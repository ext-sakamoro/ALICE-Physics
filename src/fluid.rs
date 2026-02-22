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
#[derive(Clone, Copy, Debug)]
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
    /// Velocity damping
    pub damping: Fix128,
    /// Viscosity coefficient (XSPH)
    pub viscosity: Fix128,
    /// Vorticity confinement strength
    pub vorticity_strength: Fix128,
    /// Surface tension coefficient
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
    /// Cached reciprocal of rest_density to avoid repeated division in hot loops
    inv_rest_density: Fix128,
}

impl Fluid {
    /// Create fluid with initial particle positions
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
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Step fluid simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
    }

    /// Step with SDF boundary
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_boundary(sdf_colliders);
        }
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;

        // 1. Predict positions
        for i in 0..n {
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.velocities[i] = self.velocities[i] * self.config.damping;
            self.predicted[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Build spatial grid
        self.grid.clear();
        for i in 0..n {
            self.grid.insert(i, self.predicted[i]);
        }

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

        // 6. Surface tension (cohesion force toward neighbors)
        if !self.config.surface_tension.is_zero() {
            self.apply_surface_tension();
        }

        // 7. Vorticity confinement (re-inject rotational energy lost to damping)
        if !self.config.vorticity_strength.is_zero() {
            self.apply_vorticity_confinement();
        }
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

    /// Surface tension via pairwise cohesion forces.
    /// Particles attract neighbors, creating surface-minimizing behavior.
    fn apply_surface_tension(&mut self) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let coeff = self.config.surface_tension;
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

    /// Vorticity confinement: re-inject rotational energy lost to damping.
    fn apply_vorticity_confinement(&mut self) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let epsilon = self.config.vorticity_strength;
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
                let grad_mag = spiky_grad(r, h);
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
                let grad_w = spiky_grad(r, h);
                let curl_mag_diff = curls[j].length() - curls[i].length();
                grad_mag_curl = grad_mag_curl + delta / r * (grad_w * curl_mag_diff);
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

        let h_sq = Fix128::from_ratio(1, 5) * Fix128::from_ratio(1, 5);
        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, h_sq, &mut neighbors);
        assert!(neighbors.contains(&0), "Should find self");
        assert!(neighbors.contains(&1), "Should find nearby particle");
    }
}
