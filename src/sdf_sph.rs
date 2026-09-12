//! Minimal SPH particle fluid with SDF boundary repulsion.
//!
//! Companion of [`crate::fluid`] (position-based fluid) and
//! [`crate::sdf_collider`] (SDF surface primitives) that exposes a
//! **Smoothed Particle Hydrodynamics** solver small enough to embed
//! in gameplay code but faithful to the standard Müller / Charypar /
//! Gross (2003) forcing model.
//!
//! # Scope
//!
//! MVP: fixed kernel radius, uniform particle mass, single-phase
//! fluid, Poly6 density kernel, Spiky pressure gradient, viscosity
//! Laplacian, and SDF boundary repulsion. Neighbour queries use a
//! naive `O(N²)` all-pairs pass so callers get correct results out
//! of the box; production callers with more than a few hundred
//! particles should reach for [`crate::spatial`] or the PBF solver.
//!
//! Deferred to future work: surface tension, XSPH viscosity,
//! adaptive time stepping, boundary tangential friction, and a
//! spatial-hash acceleration structure.

use crate::sdf_collider::SdfField;

/// A single SPH particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphParticle {
    /// Position (m).
    pub position: [f32; 3],
    /// Velocity (m/s).
    pub velocity: [f32; 3],
    /// Sampled density (kg/m³). Written back by `SphSolver::step`.
    pub density: f32,
    /// Sampled pressure (Pa). Written back by `SphSolver::step`.
    pub pressure: f32,
}

impl SphParticle {
    /// Convenience constructor for a resting particle.
    #[must_use]
    pub const fn at_rest(position: [f32; 3]) -> Self {
        Self {
            position,
            velocity: [0.0, 0.0, 0.0],
            density: 0.0,
            pressure: 0.0,
        }
    }
}

/// Solver parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphConfig {
    /// Smoothing radius `h` (m).
    pub kernel_radius: f32,
    /// Uniform particle mass (kg).
    pub particle_mass: f32,
    /// Rest density (kg/m³) — the pressure equation drives excess
    /// density back to this value.
    pub rest_density: f32,
    /// Gas / stiffness constant `k` in `p = k (ρ − ρ_0)`.
    pub gas_stiffness: f32,
    /// Viscosity `μ` in the Laplacian force term.
    pub viscosity: f32,
    /// Gravity vector (m/s²).
    pub gravity: [f32; 3],
    /// Boundary repulsion strength. Force magnitude scales with
    /// `boundary_strength · (repel_range − sdf_distance)`.
    pub boundary_strength: f32,
    /// SDF distance below which boundary repulsion activates.
    pub repel_range: f32,
}

impl SphConfig {
    /// Reasonable defaults for a water-like fluid at 1 m³-scale.
    #[must_use]
    pub const fn water_like() -> Self {
        Self {
            kernel_radius: 0.05,
            particle_mass: 0.02,
            rest_density: 1000.0,
            gas_stiffness: 20.0,
            viscosity: 0.1,
            gravity: [0.0, -9.81, 0.0],
            boundary_strength: 500.0,
            repel_range: 0.02,
        }
    }
}

/// SPH solver state.
pub struct SphSolver<'a, F: SdfField + ?Sized> {
    /// Particle array (mutated in place).
    pub particles: Vec<SphParticle>,
    /// Solver parameters.
    pub config: SphConfig,
    /// SDF field describing the fluid container / obstacles.
    pub boundary_sdf: &'a F,
}

impl<'a, F: SdfField + ?Sized> SphSolver<'a, F> {
    /// Construct a solver.
    pub fn new(particles: Vec<SphParticle>, config: SphConfig, boundary_sdf: &'a F) -> Self {
        Self {
            particles,
            config,
            boundary_sdf,
        }
    }

    /// Advance the solver by `dt` seconds using symplectic Euler.
    pub fn step(&mut self, dt: f32) {
        self.compute_density_and_pressure();
        let accelerations = self.compute_accelerations();
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.velocity[0] += accelerations[i][0] * dt;
            p.velocity[1] += accelerations[i][1] * dt;
            p.velocity[2] += accelerations[i][2] * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }
    }

    /// Advance one step using a **spatial-hash-accelerated** neighbour
    /// search sized to `kernel_radius`.
    ///
    /// Behaviourally equivalent to [`step`](Self::step) up to
    /// floating-point ordering; asymptotically `O(N · k)` where `k` is
    /// the average neighbour count per cell instead of `O(N²)`.
    pub fn step_hashed(&mut self, dt: f32) {
        let hash = SphSpatialHash::build(&self.particles, self.config.kernel_radius);
        self.compute_density_and_pressure_hashed(&hash);
        let accelerations = self.compute_accelerations_hashed(&hash);
        for (i, p) in self.particles.iter_mut().enumerate() {
            p.velocity[0] += accelerations[i][0] * dt;
            p.velocity[1] += accelerations[i][1] * dt;
            p.velocity[2] += accelerations[i][2] * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }
    }

    fn compute_density_and_pressure_hashed(&mut self, hash: &SphSpatialHash) {
        let h = self.config.kernel_radius;
        let m = self.config.particle_mass;
        let h_sq = h * h;
        for i in 0..self.particles.len() {
            let pi = self.particles[i].position;
            let mut rho = 0.0_f32;
            hash.for_each_neighbour(pi, |j| {
                let pj = self.particles[j].position;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let r_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                if r_sq < h_sq {
                    rho += m * poly6(r_sq.sqrt(), h);
                }
            });
            self.particles[i].density = rho;
            let excess = rho - self.config.rest_density;
            self.particles[i].pressure = (self.config.gas_stiffness * excess).max(0.0);
        }
    }

    fn compute_accelerations_hashed(&self, hash: &SphSpatialHash) -> Vec<[f32; 3]> {
        let h = self.config.kernel_radius;
        let m = self.config.particle_mass;
        let h_sq = h * h;
        let mut out = Vec::with_capacity(self.particles.len());
        for i in 0..self.particles.len() {
            let pi = self.particles[i];
            let mut ax = self.config.gravity[0];
            let mut ay = self.config.gravity[1];
            let mut az = self.config.gravity[2];
            hash.for_each_neighbour(pi.position, |j| {
                if i == j {
                    return;
                }
                let pj = self.particles[j];
                let dx = pi.position[0] - pj.position[0];
                let dy = pi.position[1] - pj.position[1];
                let dz = pi.position[2] - pj.position[2];
                let r_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                if r_sq < 1.0e-8 || r_sq >= h_sq {
                    return;
                }
                let r = r_sq.sqrt();
                let pressure_scale = -m
                    * ((pi.pressure + pj.pressure) / (2.0 * pj.density.max(1.0e-6)))
                    * spiky_grad(r, h);
                ax += pressure_scale * dx / r;
                ay += pressure_scale * dy / r;
                az += pressure_scale * dz / r;
                let viscosity_scale =
                    self.config.viscosity * m / pj.density.max(1.0e-6) * viscosity_lap(r, h);
                ax += viscosity_scale * (pj.velocity[0] - pi.velocity[0]);
                ay += viscosity_scale * (pj.velocity[1] - pi.velocity[1]);
                az += viscosity_scale * (pj.velocity[2] - pi.velocity[2]);
            });
            let d = self
                .boundary_sdf
                .distance(pi.position[0], pi.position[1], pi.position[2]);
            if d < self.config.repel_range {
                let (nx, ny, nz) =
                    self.boundary_sdf
                        .normal(pi.position[0], pi.position[1], pi.position[2]);
                let push = self.config.boundary_strength * (self.config.repel_range - d);
                ax += nx * push;
                ay += ny * push;
                az += nz * push;
            }
            out.push([ax, ay, az]);
        }
        out
    }

    fn compute_density_and_pressure(&mut self) {
        let h = self.config.kernel_radius;
        let m = self.config.particle_mass;
        for i in 0..self.particles.len() {
            let mut rho = 0.0_f32;
            let pi = self.particles[i].position;
            for j in 0..self.particles.len() {
                let pj = self.particles[j].position;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let r_sq = dx * dx + dy * dy + dz * dz;
                if r_sq < h * h {
                    let r = r_sq.sqrt();
                    rho += m * poly6(r, h);
                }
            }
            self.particles[i].density = rho;
            let excess = rho - self.config.rest_density;
            self.particles[i].pressure = (self.config.gas_stiffness * excess).max(0.0);
        }
    }

    fn compute_accelerations(&self) -> Vec<[f32; 3]> {
        let h = self.config.kernel_radius;
        let m = self.config.particle_mass;
        let mut out = Vec::with_capacity(self.particles.len());
        for i in 0..self.particles.len() {
            let pi = self.particles[i];
            let mut ax = self.config.gravity[0];
            let mut ay = self.config.gravity[1];
            let mut az = self.config.gravity[2];
            for j in 0..self.particles.len() {
                if i == j {
                    continue;
                }
                let pj = self.particles[j];
                let dx = pi.position[0] - pj.position[0];
                let dy = pi.position[1] - pj.position[1];
                let dz = pi.position[2] - pj.position[2];
                let r_sq = dx * dx + dy * dy + dz * dz;
                if r_sq < 1.0e-8 || r_sq >= h * h {
                    continue;
                }
                let r = r_sq.sqrt();
                // Pressure force: symmetric Spiky gradient.
                let pressure_scale = -m
                    * ((pi.pressure + pj.pressure) / (2.0 * pj.density.max(1.0e-6)))
                    * spiky_grad(r, h);
                ax += pressure_scale * dx / r;
                ay += pressure_scale * dy / r;
                az += pressure_scale * dz / r;
                // Viscosity force: Laplacian of the velocity difference.
                let viscosity_scale =
                    self.config.viscosity * m / pj.density.max(1.0e-6) * viscosity_lap(r, h);
                ax += viscosity_scale * (pj.velocity[0] - pi.velocity[0]);
                ay += viscosity_scale * (pj.velocity[1] - pi.velocity[1]);
                az += viscosity_scale * (pj.velocity[2] - pi.velocity[2]);
            }
            // Boundary repulsion.
            let d = self
                .boundary_sdf
                .distance(pi.position[0], pi.position[1], pi.position[2]);
            if d < self.config.repel_range {
                let (nx, ny, nz) =
                    self.boundary_sdf
                        .normal(pi.position[0], pi.position[1], pi.position[2]);
                let push = self.config.boundary_strength * (self.config.repel_range - d);
                ax += nx * push;
                ay += ny * push;
                az += nz * push;
            }
            out.push([ax, ay, az]);
        }
        out
    }
}

/// Poly6 density kernel `W(r, h) = (315 / (64 π h⁹)) (h² − r²)³`.
#[must_use]
pub fn poly6(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let scale = 315.0 / (64.0 * core::f32::consts::PI * h.powi(9));
    let diff = h * h - r * r;
    scale * diff * diff * diff
}

/// Magnitude of the Spiky pressure kernel gradient:
/// `|∇W_spiky(r, h)| = (45 / (π h⁶)) (h − r)²`.
#[must_use]
pub fn spiky_grad(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let scale = 45.0 / (core::f32::consts::PI * h.powi(6));
    let diff = h - r;
    scale * diff * diff
}

/// Uniform-cell spatial hash sized to the SPH kernel radius.
///
/// Cells are indexed by a `HashMap<(i32,i32,i32), Vec<usize>>`; a
/// query iterates the 3×3×3 neighbourhood around a probe position and
/// yields every particle in those cells. The cell size is set to the
/// kernel radius so each 3×3×3 neighbourhood strictly covers the SPH
/// interaction sphere.
///
/// Ephemeral by design: rebuild each step (`build`).
pub struct SphSpatialHash {
    cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SphSpatialHash {
    /// Build the hash from the current particle positions.
    #[must_use]
    pub fn build(particles: &[SphParticle], kernel_radius: f32) -> Self {
        let cell_size = kernel_radius.max(1.0e-6);
        let mut cells: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, p) in particles.iter().enumerate() {
            let key = position_to_cell(p.position, cell_size);
            cells.entry(key).or_default().push(i);
        }
        Self { cell_size, cells }
    }

    /// Cell size (== SPH kernel radius).
    #[must_use]
    pub const fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Number of populated cells.
    #[must_use]
    pub fn populated_cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Iterate every particle in the 3×3×3 cell neighbourhood
    /// around `position`.
    pub fn for_each_neighbour<F: FnMut(usize)>(&self, position: [f32; 3], mut visit: F) {
        let (cx, cy, cz) = position_to_cell(position, self.cell_size);
        for dz in -1..=1_i32 {
            for dy in -1..=1_i32 {
                for dx in -1..=1_i32 {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx in bucket {
                            visit(idx);
                        }
                    }
                }
            }
        }
    }
}

fn position_to_cell(pos: [f32; 3], cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    (
        (pos[0] * inv).floor() as i32,
        (pos[1] * inv).floor() as i32,
        (pos[2] * inv).floor() as i32,
    )
}

/// Viscosity Laplacian: `∇²W_visc(r, h) = (45 / (π h⁶)) (h − r)`.
#[must_use]
pub fn viscosity_lap(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let scale = 45.0 / (core::f32::consts::PI * h.powi(6));
    scale * (h - r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    #[test]
    fn poly6_zero_beyond_radius() {
        assert!(poly6(0.1, 0.05).abs() < 1.0e-6);
    }

    #[test]
    fn poly6_peak_at_zero() {
        let center = poly6(0.0, 0.1);
        let mid = poly6(0.05, 0.1);
        assert!(center > mid);
    }

    #[test]
    fn spiky_grad_zero_beyond_radius() {
        assert!(spiky_grad(0.2, 0.1) < 1.0e-6);
    }

    #[test]
    fn spatial_hash_bucket_count_matches_particle_placement() {
        let particles = vec![
            SphParticle::at_rest([0.0, 0.0, 0.0]),
            SphParticle::at_rest([0.01, 0.0, 0.0]),
            SphParticle::at_rest([5.0, 5.0, 5.0]),
        ];
        let hash = SphSpatialHash::build(&particles, 0.05);
        assert_eq!(hash.cell_size(), 0.05);
        // First two share one cell; third is far away.
        assert_eq!(hash.populated_cell_count(), 2);
    }

    #[test]
    fn spatial_hash_neighbours_cover_kernel_ball() {
        let particles = vec![
            SphParticle::at_rest([0.0, 0.0, 0.0]),
            SphParticle::at_rest([0.03, 0.0, 0.0]),
            SphParticle::at_rest([0.06, 0.0, 0.0]),
            SphParticle::at_rest([1.0, 1.0, 1.0]),
        ];
        let h = 0.05;
        let hash = SphSpatialHash::build(&particles, h);
        let mut hits = Vec::new();
        hash.for_each_neighbour([0.0, 0.0, 0.0], |i| hits.push(i));
        hits.sort_unstable();
        hits.dedup();
        // Particles at 0, 0.03, 0.06 should all fall in the 3×3×3
        // neighbourhood around the origin (indices 0, 1, 2), while
        // index 3 must be excluded.
        assert!(hits.contains(&0));
        assert!(hits.contains(&1));
        assert!(hits.contains(&2));
        assert!(!hits.contains(&3));
    }

    #[test]
    fn hashed_step_matches_naive_step_on_small_cluster() {
        let particles = vec![
            SphParticle::at_rest([0.00, 0.10, 0.00]),
            SphParticle::at_rest([0.02, 0.10, 0.00]),
            SphParticle::at_rest([0.00, 0.12, 0.02]),
        ];
        let config = SphConfig {
            gravity: [0.0, 0.0, 0.0], // isolate neighbour path
            boundary_strength: 0.0,
            ..SphConfig::water_like()
        };
        let sdf = ground_plane();
        let mut naive = SphSolver::new(particles.clone(), config, &sdf);
        let mut hashed = SphSolver::new(particles, config, &sdf);
        naive.step(1.0e-4);
        hashed.step_hashed(1.0e-4);
        for i in 0..naive.particles.len() {
            for axis in 0..3 {
                let n = naive.particles[i].position[axis];
                let h = hashed.particles[i].position[axis];
                assert!(
                    (n - h).abs() < 1.0e-4,
                    "particle {i} axis {axis}: naive={n} hashed={h}"
                );
            }
        }
    }

    #[test]
    fn viscosity_lap_positive_inside_radius() {
        assert!(viscosity_lap(0.05, 0.1) > 0.0);
    }

    #[test]
    fn config_water_like_has_reasonable_defaults() {
        let c = SphConfig::water_like();
        assert!(c.kernel_radius > 0.0);
        assert!(c.rest_density > 0.0);
        assert!(c.gravity[1] < 0.0);
    }

    #[test]
    fn single_particle_falls_under_gravity() {
        let plane = ground_plane();
        let particles = vec![SphParticle::at_rest([0.0, 5.0, 0.0])];
        let mut solver = SphSolver::new(particles, SphConfig::water_like(), &plane);
        solver.step(0.01);
        assert!(solver.particles[0].velocity[1] < 0.0);
        assert!(solver.particles[0].position[1] < 5.0);
    }

    #[test]
    fn boundary_repulsion_reverses_falling_particle() {
        let plane = ground_plane();
        let mut cfg = SphConfig::water_like();
        cfg.gravity = [0.0, -1.0, 0.0]; // gentler gravity
        cfg.boundary_strength = 1000.0;
        // Particle already near ground.
        let particles = vec![SphParticle::at_rest([0.0, 0.005, 0.0])];
        let mut solver = SphSolver::new(particles, cfg, &plane);
        for _ in 0..20 {
            solver.step(0.001);
        }
        // Boundary should have pushed it above the ground.
        assert!(solver.particles[0].position[1] > 0.005 - 0.02);
    }

    #[test]
    fn density_increases_when_particles_cluster() {
        let plane = ground_plane();
        // Two overlapping particles.
        let particles = vec![
            SphParticle::at_rest([0.0, 1.0, 0.0]),
            SphParticle::at_rest([0.01, 1.0, 0.0]),
        ];
        let mut solver = SphSolver::new(particles, SphConfig::water_like(), &plane);
        solver.compute_density_and_pressure();
        assert!(solver.particles[0].density > 0.0);
        assert!(solver.particles[1].density > 0.0);
    }
}
