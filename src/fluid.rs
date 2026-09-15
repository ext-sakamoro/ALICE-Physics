//! Position-Based Fluids (PBF)
//!
//! SPH-based fluid simulation using XPBD density constraints.
//! Supports SDF boundary containment and surface tension.
//!
//! # Algorithm (Macklin & Muller 2013)
//!
//! 1. Predict particle positions
//! 2. Find neighbors (spatial hash grid)
//! 3. Iterate: compute density, solve density constraints
//! 4. Apply viscosity and vorticity confinement
//! 5. Update velocities
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;
use crate::spatial::SpatialGrid;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Fluid Configuration
// ============================================================================

/// Fluid simulation configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidConfig {
    /// Rest density (kg/m^3)
    pub rest_density: Fix128,
    /// Smoothing kernel radius
    pub kernel_radius: Fix128,
    /// Number of solver iterations
    pub iterations: usize,
    /// Number of substeps
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity retention per frame (`step()` call), applied once per frame
    /// since 1.2.0 (per substep before, which made the result depend on `substeps`)
    pub damping: Fix128,
    /// Viscosity coefficient (XSPH)
    pub viscosity: Fix128,
    /// Vorticity confinement strength ε (`Δv = ε · dt · N × ω`, PBF §5)
    pub vorticity_strength: Fix128,
    /// Surface tension / cohesion coefficient κ
    /// (`Δv_i = κ · dt · Σ_j (m_j/ρ₀) (x_j − x_i)/r · W_ij`)
    pub surface_tension: Fix128,
    /// Particle mass
    pub particle_mass: Fix128,
    /// Constraint relaxation parameter
    pub relaxation: Fix128,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            rest_density: Fix128::from_int(1000),
            kernel_radius: Fix128::from_ratio(1, 5), // 0.2
            iterations: 4,
            substeps: 2,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            viscosity: Fix128::from_ratio(1, 100),
            vorticity_strength: Fix128::from_ratio(1, 1000),
            surface_tension: Fix128::from_ratio(1, 100),
            particle_mass: Fix128::from_ratio(1, 100),
            relaxation: Fix128::ONE,
        }
    }
}

// ============================================================================
// SPH Kernels (Deterministic)
// ============================================================================

/// Poly6 kernel (density estimation)
#[inline(always)]
fn poly6(r_sq: Fix128, h: Fix128) -> Fix128 {
    let h_sq = h * h;
    if r_sq >= h_sq {
        return Fix128::ZERO;
    }
    let diff = h_sq - r_sq;
    // W = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
    // Simplified constant for deterministic computation
    let h9 = h_sq * h_sq * h_sq * h_sq * h;
    let coeff = Fix128::from_ratio(315, 64) / h9;
    coeff * diff * diff * diff
}

/// Spiky kernel gradient magnitude (pressure)
#[inline(always)]
fn spiky_grad(r: Fix128, h: Fix128) -> Fix128 {
    if r >= h || r.is_zero() {
        return Fix128::ZERO;
    }
    let diff = h - r;
    // grad_W = -45 / (pi * h^6) * (h - r)^2
    let h6 = h * h * h * h * h * h;
    let coeff = Fix128::from_ratio(45, 1) / h6;
    -coeff * diff * diff
}

// ============================================================================
// Fluid Simulation
// ============================================================================

/// Position-Based Fluid simulation
#[repr(C, align(64))]
pub struct Fluid {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Predicted positions
    predicted: Vec<Vec3Fix>,
    /// Velocities
    pub velocities: Vec<Vec3Fix>,
    /// Per-particle density
    pub densities: Vec<Fix128>,
    /// Per-particle lambda (constraint multiplier)
    lambdas: Vec<Fix128>,
    /// Spatial hash grid
    grid: SpatialGrid,
    /// Configuration
    pub config: FluidConfig,
    /// Cached reciprocal of `rest_density` to avoid repeated division in hot loops
    inv_rest_density: Fix128,
}

impl Fluid {
    /// Create fluid with initial particle positions
    #[must_use]
    pub fn new(positions: Vec<Vec3Fix>, config: FluidConfig) -> Self {
        let n = positions.len();
        let grid_dim = 32;
        let inv_rest_density = if config.rest_density.is_zero() {
            Fix128::ONE
        } else {
            Fix128::ONE / config.rest_density
        };

        Self {
            predicted: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            densities: vec![Fix128::ZERO; n],
            lambdas: vec![Fix128::ZERO; n],
            grid: SpatialGrid::new(config.kernel_radius, grid_dim),
            positions,
            config,
            inv_rest_density,
        }
    }

    /// Create a block of fluid particles
    #[must_use]
    pub fn new_block(min: Vec3Fix, max: Vec3Fix, spacing: Fix128, config: FluidConfig) -> Self {
        let mut positions = Vec::new();

        let mut x = min.x;
        while x <= max.x {
            let mut y = min.y;
            while y <= max.y {
                let mut z = min.z;
                while z <= max.z {
                    positions.push(Vec3Fix::new(x, y, z));
                    z = z + spacing;
                }
                y = y + spacing;
            }
            x = x + spacing;
        }

        Self::new(positions, config)
    }

    /// Number of particles
    #[inline(always)]
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Step fluid simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
        self.apply_frame_damping();
    }

    /// Step with SDF boundary
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_boundary(sdf_colliders);
        }
        self.apply_frame_damping();
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;

        // 1. Predict positions (damping is applied once per frame in `step`,
        //    1.2.0 — per substep it made the terminal velocity depend on
        //    `substeps`, the same defect as the rigid-body solver's R2-1)
        for i in 0..n {
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.predicted[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Build spatial grid
        self.grid.clear();
        for i in 0..n {
            self.grid.insert(i, self.predicted[i]);
        }
        self.grid.build();

        // 3. Density constraint iterations
        let mut neighbors_buf = Vec::new();
        for _ in 0..self.config.iterations {
            // Compute densities and lambdas
            for i in 0..n {
                self.grid
                    .query_neighbors_into(self.predicted[i], h_sq, &mut neighbors_buf);
                let mut density = Fix128::ZERO;
                let mut sum_grad_sq = Fix128::ZERO;
                let mut grad_i = Vec3Fix::ZERO;

                for &j in &neighbors_buf {
                    let delta = self.predicted[i] - self.predicted[j];
                    let r_sq = delta.length_squared();

                    density = density + self.config.particle_mass * poly6(r_sq, h);

                    if i != j && r_sq < h_sq {
                        let r = r_sq.sqrt();
                        let grad_mag = spiky_grad(r, h);
                        if !r.is_zero() {
                            let grad = delta / r * grad_mag;
                            grad_i = grad_i + grad;
                            sum_grad_sq = sum_grad_sq + grad.length_squared();
                        }
                    }
                }

                self.densities[i] = density;

                // Constraint: C = density / rest_density - 1
                // Use precomputed reciprocal to avoid repeated division
                let constraint = density * self.inv_rest_density - Fix128::ONE;
                sum_grad_sq = sum_grad_sq + grad_i.length_squared();

                let epsilon = Fix128::from_ratio(1, 10000);
                self.lambdas[i] = if sum_grad_sq.is_zero() {
                    Fix128::ZERO
                } else {
                    -constraint / (sum_grad_sq + epsilon)
                };
            }

            // Apply position corrections
            for i in 0..n {
                self.grid
                    .query_neighbors_into(self.predicted[i], h_sq, &mut neighbors_buf);
                let mut correction = Vec3Fix::ZERO;

                for &j in &neighbors_buf {
                    if i == j {
                        continue;
                    }
                    let delta = self.predicted[i] - self.predicted[j];
                    let r_sq = delta.length_squared();
                    if r_sq >= h_sq || r_sq.is_zero() {
                        continue;
                    }

                    let r = r_sq.sqrt();
                    let grad_mag = spiky_grad(r, h);
                    let grad = delta / r * grad_mag;

                    let lambda_sum = self.lambdas[i] + self.lambdas[j];
                    correction = correction + grad * lambda_sum;
                }

                // Reuse precomputed reciprocal of rest_density
                self.predicted[i] =
                    self.predicted[i] + correction * self.inv_rest_density * self.config.relaxation;
            }
        }

        // 4. Update velocities and positions
        let inv_dt = Fix128::ONE / dt;
        for i in 0..n {
            self.velocities[i] = (self.predicted[i] - self.positions[i]) * inv_dt;
            self.positions[i] = self.predicted[i];
        }

        // 5. Apply viscosity (XSPH)
        self.apply_viscosity();

        // 6. Surface tension (cohesion acceleration toward neighbors)
        if !self.config.surface_tension.is_zero() {
            self.apply_surface_tension(dt);
        }

        // 7. Vorticity confinement (re-inject rotational energy lost to damping)
        if !self.config.vorticity_strength.is_zero() {
            self.apply_vorticity_confinement(dt);
        }
    }

    /// `damping` once per frame (velocity retention per `step()` call).
    fn apply_frame_damping(&mut self) {
        let d = self.config.damping;
        for v in &mut self.velocities {
            *v = *v * d;
        }
    }

    /// Particle volume `m / ρ₀` — the SPH normalisation that turns a kernel
    /// sum `Σ_j W_ij` into a dimensionless number of order 1 (and `Σ_j ∇W_ij`
    /// into `O(1/h)`), so the cohesion and confinement terms below are
    /// accelerations with the documented coefficients, not raw kernel sums.
    #[inline]
    fn particle_volume(&self) -> Fix128 {
        self.config.particle_mass * self.inv_rest_density
    }

    /// XSPH viscosity smoothing
    fn apply_viscosity(&mut self) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let c = self.config.viscosity;

        let velocities_copy: Vec<Vec3Fix> = self.velocities.clone();
        let mut neighbors_buf = Vec::new();

        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut avg_vel = Vec3Fix::ZERO;
            let mut weight_sum = Fix128::ZERO;

            for &j in &neighbors_buf {
                if i == j {
                    continue;
                }
                let delta = self.positions[i] - self.positions[j];
                let r_sq = delta.length_squared();
                let w = poly6(r_sq, h);
                avg_vel = avg_vel + (velocities_copy[j] - velocities_copy[i]) * w;
                weight_sum = weight_sum + w;
            }

            if !weight_sum.is_zero() {
                self.velocities[i] = self.velocities[i] + avg_vel * (c / weight_sum);
            }
        }
    }

    /// Surface tension via pairwise cohesion accelerations
    /// (Becker & Teschner 2007 form):
    /// `Δv_i = κ · dt · Σ_j (m_j / ρ₀) · (x_j − x_i)/r · W(r)`.
    ///
    /// Pairwise antisymmetric, so the centre of mass is untouched. Before
    /// 1.2.0 the un-normalised kernel sum (`W ≈ 260` at `r = h/2`,
    /// `h = 0.2`) was added straight to the velocity without `m/ρ₀` or `dt`,
    /// which threw a resting 5×5×5 block apart by ±4 m in a single frame at
    /// the default coefficient (`tests/default_configs.rs`).
    fn apply_surface_tension(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let coeff = self.config.surface_tension * dt * self.particle_volume();
        let mut neighbors_buf = Vec::new();

        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut force = Vec3Fix::ZERO;

            for &j in &neighbors_buf {
                if i == j {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let w = poly6(r_sq, h);
                force = force + delta / r * w;
            }

            self.velocities[i] = self.velocities[i] + force * coeff;
        }
    }

    /// Vorticity confinement (Macklin & Müller 2013, §5):
    /// `ω_i = Σ_j (m_j/ρ₀) (v_j − v_i) × ∇W_ij`,
    /// `η_i = Σ_j (m_j/ρ₀) |ω_j| ∇W_ij`, `N = η/|η|`,
    /// `Δv_i = ε · dt · (N × ω_i)`.
    ///
    /// Same 1.2.0 normalisation fix as `apply_surface_tension`: the kernel
    /// sums carry `m/ρ₀` and the confinement acceleration is integrated over
    /// `dt` instead of being added to the velocity as-is.
    fn apply_vorticity_confinement(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let vol = self.particle_volume();
        let epsilon = self.config.vorticity_strength * dt;
        let mut neighbors_buf = Vec::new();

        // Compute per-particle curl of velocity
        let mut curls: Vec<Vec3Fix> = vec![Vec3Fix::ZERO; n];
        for (i, curl_out) in curls.iter_mut().enumerate() {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut curl = Vec3Fix::ZERO;
            for &j in &neighbors_buf {
                if i == j {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let grad_mag = spiky_grad(r, h) * vol;
                let grad = delta / r * grad_mag;
                let vel_diff = self.velocities[j] - self.velocities[i];
                curl = curl + vel_diff.cross(grad);
            }
            *curl_out = curl;
        }

        // Apply confinement force: f = epsilon * (N x omega)
        // where N = normalize(gradient of |omega|)
        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut grad_mag_curl = Vec3Fix::ZERO;
            for &j in &neighbors_buf {
                if i == j {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let grad_w = spiky_grad(r, h) * vol;
                // PBF eq. 16 uses |ω_j| at the neighbour
                grad_mag_curl = grad_mag_curl + delta / r * (grad_w * curls[j].length());
            }
            if grad_mag_curl.length_squared().is_zero() {
                continue;
            }
            let n_vec = grad_mag_curl.normalize();
            let force = n_vec.cross(curls[i]) * epsilon;
            self.velocities[i] = self.velocities[i] + force;
        }
    }

    /// Resolve SDF boundary collisions
    #[cfg(feature = "std")]
    fn resolve_sdf_boundary(&mut self, sdf_colliders: &[SdfCollider]) {
        for i in 0..self.particle_count() {
            for sdf in sdf_colliders {
                let (lx, ly, lz) = sdf.world_to_local(self.positions[i]);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;

                // For containment: push particles inside if they escape
                if dist > 0.0 {
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let push = Fix128::from_f32(-dist);
                    self.positions[i] = self.positions[i] + normal * push;
                    self.predicted[i] = self.positions[i];

                    // Reflect velocity
                    let vn = normal * self.velocities[i].dot(normal);
                    self.velocities[i] = self.velocities[i] - vn * Fix128::from_ratio(15, 10);
                }
            }
        }
    }
}

impl core::fmt::Debug for Fluid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fluid")
            .field(
                "positions",
                &format_args!("[{} items]", self.positions.len()),
            )
            .field(
                "predicted",
                &format_args!("[{} items]", self.predicted.len()),
            )
            .field(
                "velocities",
                &format_args!("[{} items]", self.velocities.len()),
            )
            .field(
                "densities",
                &format_args!("[{} items]", self.densities.len()),
            )
            .field("lambdas", &format_args!("[{} items]", self.lambdas.len()))
            .field("grid", &"<SpatialGrid>")
            .field("config", &self.config)
            .field("inv_rest_density", &self.inv_rest_density)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_creation() {
        let config = FluidConfig::default();
        let fluid = Fluid::new_block(
            Vec3Fix::from_f32(-0.5, 0.0, -0.5),
            Vec3Fix::from_f32(0.5, 1.0, 0.5),
            Fix128::from_ratio(1, 5),
            config,
        );

        assert!(fluid.particle_count() > 0, "Should create particles");
    }

    #[test]
    fn test_fluid_gravity() {
        let config = FluidConfig {
            iterations: 2,
            substeps: 1,
            ..Default::default()
        };
        let mut fluid = Fluid::new(vec![Vec3Fix::from_f32(0.0, 5.0, 0.0)], config);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            fluid.step(dt);
        }

        let y = fluid.positions[0].y.to_f32();
        assert!(y < 5.0, "Particle should fall under gravity");
    }

    #[test]
    fn test_poly6_kernel() {
        let h = Fix128::from_ratio(1, 5);
        let zero = poly6(Fix128::ZERO, h);
        assert!(zero > Fix128::ZERO, "Poly6 at r=0 should be positive");

        let far = poly6(h * h, h);
        assert!(
            far.is_zero() || far <= Fix128::ZERO,
            "Poly6 at r=h should be zero"
        );
    }

    #[test]
    fn test_spatial_grid() {
        let mut grid = SpatialGrid::new(Fix128::from_ratio(1, 5), 32);
        grid.insert(0, Vec3Fix::ZERO);
        grid.insert(1, Vec3Fix::from_f32(0.1, 0.0, 0.0));
        grid.insert(2, Vec3Fix::from_f32(10.0, 0.0, 0.0));
        grid.build();

        let h_sq = Fix128::from_ratio(1, 5) * Fix128::from_ratio(1, 5);
        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, h_sq, &mut neighbors);
        assert!(neighbors.contains(&0), "Should find self");
        assert!(neighbors.contains(&1), "Should find nearby particle");
    }

    /// 原点中心の単位球 SDF (f32 sqrt のみ、det_math gate 対象外)
    #[cfg(feature = "std")]
    fn unit_sphere_collider() -> crate::sdf_collider::SdfCollider {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        let field = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len > 1e-6 {
                    (x / len, y / len, z / len)
                } else {
                    (0.0, 1.0, 0.0)
                }
            },
        );
        SdfCollider::new_static(
            Box::new(field),
            Vec3Fix::ZERO,
            crate::math::QuatFix::IDENTITY,
        )
    }

    /// 単位球からの符号付き距離 (f32 oracle)
    #[cfg(feature = "std")]
    fn sphere_dist(p: Vec3Fix) -> f32 {
        let (x, y, z) = p.to_f32();
        (x * x + y * y + z * z).sqrt() - 1.0
    }

    #[cfg(feature = "std")]
    #[test]
    fn step_with_sdf_contains_fluid_particles_inside_unit_sphere() {
        // kernel 半径 0.2 より離した 3 粒子 (相互作用なし) を球内に置き、重力で落とす
        // fluid の SDF は「容器」: 外に出た粒子 (dist > 0) を表面へ戻し法線速度を反射
        let make = || {
            Fluid::new(
                vec![
                    Vec3Fix::from_f32(0.0, 0.5, 0.0),
                    Vec3Fix::from_f32(0.3, 0.2, 0.0),
                    Vec3Fix::from_f32(-0.3, 0.1, 0.3),
                ],
                FluidConfig::default(),
            )
        };
        let sphere = [unit_sphere_collider()];
        let mut fluid = make();
        let dt = Fix128::from_ratio(1, 60);
        let mut max_dist = f32::MIN;
        for frame in 0..120 {
            fluid.step_with_sdf(dt, &sphere);
            for (i, p) in fluid.positions.iter().enumerate() {
                let d = sphere_dist(*p);
                max_dist = max_dist.max(d);
                assert!(
                    d <= 1e-3,
                    "frame {frame} particle {i} escaped sphere: dist {d}"
                );
            }
        }
        // 実際に壁に当たっている (2 s で 20 m 落ちるはずが半径 1 の球内)
        assert!(
            max_dist > -0.05,
            "never reached the wall: max dist {max_dist}"
        );
        for (i, p) in fluid.positions.iter().enumerate() {
            let (_, y, _) = p.to_f32();
            assert!(y >= -1.0 - 1e-3, "particle {i} below sphere bottom: y {y}");
        }

        // SDF なしでは粒子は落下して球外へ
        let mut free = make();
        for _ in 0..120 {
            free.step(dt);
        }
        for (i, p) in free.positions.iter().enumerate() {
            let d = sphere_dist(*p);
            assert!(d > 1.0, "particle {i} should have fallen out: dist {d}");
        }

        // collider が空なら step と bit 一致
        let mut a = make();
        let mut b = make();
        for _ in 0..10 {
            a.step_with_sdf(dt, &[]);
            b.step(dt);
        }
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.velocities, b.velocities);
    }
}
