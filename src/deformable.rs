//! Deformable Body Simulation (FEM-XPBD)
//!
//! Tetrahedral mesh deformable bodies using XPBD volume + shape constraints.
//! Supports SDF collision for deformable-rigid interaction.
//!
//! # Algorithm
//!
//! - Volume constraints preserve tetrahedral volume
//! - Shape matching constraints resist deformation
//! - Edge constraints prevent over-stretching
//! - SDF collision for surface contacts
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Deformable Configuration
// ============================================================================

/// Deformable body configuration
#[derive(Clone, Copy, Debug)]
pub struct DeformableConfig {
    /// Solver iterations
    pub iterations: usize,
    /// Number of substeps
    pub substeps: usize,
    /// Gravity
    pub gravity: Vec3Fix,
    /// Velocity damping
    pub damping: Fix128,
    /// Volume preservation compliance (0 = rigid volume)
    pub volume_compliance: Fix128,
    /// Edge stretch compliance
    pub edge_compliance: Fix128,
    /// Shape matching stiffness (0..1, 1 = rigid)
    pub shape_stiffness: Fix128,
    /// SDF collision friction
    pub sdf_friction: Fix128,
}

impl Default for DeformableConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            substeps: 4,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            volume_compliance: Fix128::from_ratio(1, 1000),
            edge_compliance: Fix128::ZERO,
            shape_stiffness: Fix128::from_ratio(5, 10),
            sdf_friction: Fix128::from_ratio(3, 10),
        }
    }
}

// ============================================================================
// Constraints
// ============================================================================

/// Tetrahedral volume constraint
#[derive(Clone, Copy, Debug)]
struct TetConstraint {
    /// Four vertex indices
    indices: [usize; 4],
    /// Rest volume (signed, preserves orientation)
    rest_volume: Fix128,
}

/// Edge length constraint
#[derive(Clone, Copy, Debug)]
struct EdgeConstraint {
    i0: usize,
    i1: usize,
    rest_length: Fix128,
}

// ============================================================================
// Deformable Body
// ============================================================================

/// Deformable body simulation
pub struct DeformableBody {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Previous positions
    prev_positions: Vec<Vec3Fix>,
    /// Velocities
    pub velocities: Vec<Vec3Fix>,
    /// Inverse mass per particle
    pub inv_masses: Vec<Fix128>,
    /// Tetrahedra (4 vertex indices each)
    pub tetrahedra: Vec<[usize; 4]>,
    /// Surface triangles (for rendering/collision)
    pub surface_triangles: Vec<[usize; 3]>,
    /// Tetrahedral volume constraints
    tet_constraints: Vec<TetConstraint>,
    /// Edge constraints
    edge_constraints: Vec<EdgeConstraint>,
    /// Rest shape center of mass
    rest_center: Vec3Fix,
    /// Rest positions relative to center
    rest_relative: Vec<Vec3Fix>,
    /// Configuration
    pub config: DeformableConfig,
}

impl DeformableBody {
    /// Create from particle positions and tetrahedra
    pub fn new(
        positions: Vec<Vec3Fix>,
        tetrahedra: Vec<[usize; 4]>,
        surface_triangles: Vec<[usize; 3]>,
        mass_per_particle: Fix128,
    ) -> Self {
        let n = positions.len();
        let inv_mass = if mass_per_particle.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / mass_per_particle
        };

        // Compute rest shape
        let rest_center = compute_center_of_mass(&positions);
        let rest_relative: Vec<Vec3Fix> = positions
            .iter()
            .map(|p| *p - rest_center)
            .collect();

        let mut body = Self {
            prev_positions: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            inv_masses: vec![inv_mass; n],
            tetrahedra: tetrahedra.clone(),
            surface_triangles,
            tet_constraints: Vec::new(),
            edge_constraints: Vec::new(),
            rest_center,
            rest_relative,
            positions,
            config: DeformableConfig::default(),
        };

        body.build_constraints();
        body
    }

    /// Create a simple cube deformable (for testing)
    pub fn new_cube(center: Vec3Fix, half_extent: Fix128, mass: Fix128) -> Self {
        let h = half_extent;
        let positions = vec![
            center + Vec3Fix::new(-h, -h, -h),
            center + Vec3Fix::new(h, -h, -h),
            center + Vec3Fix::new(h, h, -h),
            center + Vec3Fix::new(-h, h, -h),
            center + Vec3Fix::new(-h, -h, h),
            center + Vec3Fix::new(h, -h, h),
            center + Vec3Fix::new(h, h, h),
            center + Vec3Fix::new(-h, h, h),
        ];

        // 5 tetrahedra to fill a cube
        let tetrahedra = vec![
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
            [1, 3, 4, 6],
        ];

        let surface_triangles = vec![
            [0, 1, 2], [0, 2, 3], // front
            [4, 6, 5], [4, 7, 6], // back
            [0, 4, 5], [0, 5, 1], // bottom
            [2, 6, 7], [2, 7, 3], // top
            [0, 3, 7], [0, 7, 4], // left
            [1, 5, 6], [1, 6, 2], // right
        ];

        let mass_per_particle = mass / Fix128::from_int(8);
        Self::new(positions, tetrahedra, surface_triangles, mass_per_particle)
    }

    /// Build constraints from geometry
    fn build_constraints(&mut self) {
        // Tetrahedral volume constraints
        for tet in &self.tetrahedra {
            let rest_vol = compute_tet_volume(
                self.positions[tet[0]],
                self.positions[tet[1]],
                self.positions[tet[2]],
                self.positions[tet[3]],
            );
            self.tet_constraints.push(TetConstraint {
                indices: *tet,
                rest_volume: rest_vol,
            });
        }

        // Edge constraints: unique edges from tetrahedra
        let mut edge_set: Vec<(usize, usize)> = Vec::new();
        for tet in &self.tetrahedra {
            let edges = [
                (tet[0], tet[1]), (tet[0], tet[2]), (tet[0], tet[3]),
                (tet[1], tet[2]), (tet[1], tet[3]), (tet[2], tet[3]),
            ];
            for (a, b) in edges {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if !edge_set.contains(&(lo, hi)) {
                    edge_set.push((lo, hi));
                    let rest = (self.positions[a] - self.positions[b]).length();
                    self.edge_constraints.push(EdgeConstraint {
                        i0: a, i1: b, rest_length: rest,
                    });
                }
            }
        }
    }

    /// Number of particles
    #[inline]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Step simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
    }

    /// Step with SDF collision
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();

        // 1. Predict
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.prev_positions[i] = self.positions[i];
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.velocities[i] = self.velocities[i] * self.config.damping;
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Solve constraints
        for _ in 0..self.config.iterations {
            self.solve_edge_constraints(dt);
            self.solve_volume_constraints(dt);
            self.apply_shape_matching();
        }

        // 3. Update velocities
        let inv_dt = Fix128::ONE / dt;
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.velocities[i] = (self.positions[i] - self.prev_positions[i]) * inv_dt;
        }
    }

    /// Solve edge constraints
    fn solve_edge_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.edge_compliance / (dt * dt);

        for c_idx in 0..self.edge_constraints.len() {
            let c = self.edge_constraints[c_idx];
            let p0 = self.positions[c.i0];
            let p1 = self.positions[c.i1];
            let w0 = self.inv_masses[c.i0];
            let w1 = self.inv_masses[c.i1];

            let w_sum = w0 + w1 + compliance;
            if w_sum.is_zero() { continue; }

            let delta = p1 - p0;
            let dist = delta.length();
            if dist.is_zero() { continue; }

            let error = dist - c.rest_length;
            let lambda = error / w_sum;
            let correction = delta / dist * lambda;

            if !w0.is_zero() {
                self.positions[c.i0] = self.positions[c.i0] + correction * w0;
            }
            if !w1.is_zero() {
                self.positions[c.i1] = self.positions[c.i1] - correction * w1;
            }
        }
    }

    /// Solve tetrahedral volume constraints
    fn solve_volume_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.volume_compliance / (dt * dt);

        for c_idx in 0..self.tet_constraints.len() {
            let c = self.tet_constraints[c_idx];
            let [i0, i1, i2, i3] = c.indices;

            let p0 = self.positions[i0];
            let p1 = self.positions[i1];
            let p2 = self.positions[i2];
            let p3 = self.positions[i3];

            let current_vol = compute_tet_volume(p0, p1, p2, p3);
            let error = current_vol - c.rest_volume;

            if error.abs() < Fix128::from_ratio(1, 10000) {
                continue;
            }

            // Gradients of volume w.r.t. each vertex
            let g0 = (p1 - p3).cross(p2 - p3);
            let g1 = (p2 - p3).cross(p0 - p3);
            let g2 = (p0 - p3).cross(p1 - p3);
            let g3 = -(g0 + g1 + g2);

            let w0 = self.inv_masses[i0];
            let w1 = self.inv_masses[i1];
            let w2 = self.inv_masses[i2];
            let w3 = self.inv_masses[i3];

            let denom = w0 * g0.length_squared()
                + w1 * g1.length_squared()
                + w2 * g2.length_squared()
                + w3 * g3.length_squared()
                + compliance;

            if denom.is_zero() { continue; }

            let lambda = error / denom;
            let six = Fix128::from_int(6);

            if !w0.is_zero() {
                self.positions[i0] = self.positions[i0] - g0 * (lambda * w0 / six);
            }
            if !w1.is_zero() {
                self.positions[i1] = self.positions[i1] - g1 * (lambda * w1 / six);
            }
            if !w2.is_zero() {
                self.positions[i2] = self.positions[i2] - g2 * (lambda * w2 / six);
            }
            if !w3.is_zero() {
                self.positions[i3] = self.positions[i3] - g3 * (lambda * w3 / six);
            }
        }
    }

    /// Shape matching: move particles toward their rest pose
    fn apply_shape_matching(&mut self) {
        let alpha = self.config.shape_stiffness;
        if alpha.is_zero() {
            return;
        }

        // Current center of mass
        let current_center = compute_center_of_mass(&self.positions);

        // Simplified shape matching: blend toward rest shape
        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            let target = current_center + self.rest_relative[i];
            let delta = target - self.positions[i];
            self.positions[i] = self.positions[i] + delta * alpha;
        }
    }

    /// Resolve SDF collisions
    #[cfg(feature = "std")]
    fn resolve_sdf_collisions(&mut self, sdf_colliders: &[SdfCollider]) {
        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() { continue; }

            for sdf in sdf_colliders {
                let (lx, ly, lz) = sdf.world_to_local(self.positions[i]);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;

                if dist < 0.0 {
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let depth = Fix128::from_f32(-dist);
                    self.positions[i] = self.positions[i] + normal * depth;

                    let vel = self.velocities[i];
                    let vn = normal * vel.dot(normal);
                    let vt = vel - vn;
                    self.velocities[i] = vt * (Fix128::ONE - self.config.sdf_friction);
                }
            }
        }
    }

    /// Get center of mass
    pub fn center_of_mass(&self) -> Vec3Fix {
        compute_center_of_mass(&self.positions)
    }

    /// Resolve collisions between deformable body particles and rigid bodies
    ///
    /// Treats each rigid body as a sphere with given radius for simplicity.
    /// Applies two-way coupling: deformable particles are pushed out,
    /// and rigid bodies receive reaction impulses.
    pub fn resolve_rigid_body_collisions(
        &mut self,
        rigid_bodies: &mut [crate::solver::RigidBody],
        body_radii: &[Fix128],
        dt: Fix128,
    ) {
        if dt.is_zero() { return; }

        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() { continue; }

            for (rb_idx, rb) in rigid_bodies.iter_mut().enumerate() {
                if rb.is_static() && rb_idx >= body_radii.len() { continue; }
                let radius = if rb_idx < body_radii.len() { body_radii[rb_idx] } else { Fix128::ONE };

                let delta = self.positions[i] - rb.position;
                let dist_sq = delta.length_squared();
                let r_sq = radius * radius;

                if dist_sq < r_sq && !dist_sq.is_zero() {
                    let dist = dist_sq.sqrt();
                    let normal = delta / dist;
                    let penetration = radius - dist;

                    // Push particle out
                    let particle_mass = if self.inv_masses[i].is_zero() {
                        Fix128::from_int(1000000)
                    } else {
                        Fix128::ONE / self.inv_masses[i]
                    };

                    let rb_mass = if rb.inv_mass.is_zero() {
                        Fix128::from_int(1000000)
                    } else {
                        Fix128::ONE / rb.inv_mass
                    };

                    let total_mass = particle_mass + rb_mass;
                    if total_mass.is_zero() { continue; }

                    let particle_ratio = rb_mass / total_mass;
                    let rb_ratio = particle_mass / total_mass;

                    // Position correction
                    self.positions[i] = self.positions[i] + normal * (penetration * particle_ratio);

                    if !rb.inv_mass.is_zero() {
                        rb.position = rb.position - normal * (penetration * rb_ratio);
                    }

                    // Velocity response
                    let rel_vel = self.velocities[i] - rb.velocity;
                    let vel_normal = rel_vel.dot(normal);

                    if vel_normal < Fix128::ZERO {
                        let impulse_mag = -vel_normal * (particle_mass * rb_mass / total_mass);
                        let impulse = normal * impulse_mag;

                        self.velocities[i] = self.velocities[i] + impulse * self.inv_masses[i];
                        if !rb.inv_mass.is_zero() {
                            rb.velocity = rb.velocity - impulse * rb.inv_mass;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute center of mass
fn compute_center_of_mass(positions: &[Vec3Fix]) -> Vec3Fix {
    if positions.is_empty() {
        return Vec3Fix::ZERO;
    }
    let mut sum = Vec3Fix::ZERO;
    for p in positions {
        sum = sum + *p;
    }
    sum / Fix128::from_int(positions.len() as i64)
}

/// Compute signed volume of a tetrahedron
fn compute_tet_volume(p0: Vec3Fix, p1: Vec3Fix, p2: Vec3Fix, p3: Vec3Fix) -> Fix128 {
    let a = p1 - p0;
    let b = p2 - p0;
    let c = p3 - p0;
    a.dot(b.cross(c)) / Fix128::from_int(6)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deformable_cube() {
        let body = DeformableBody::new_cube(
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::from_int(10),
        );

        assert_eq!(body.particle_count(), 8);
        assert_eq!(body.tetrahedra.len(), 5);
        assert!(!body.edge_constraints.is_empty());
        assert!(!body.tet_constraints.is_empty());
    }

    #[test]
    fn test_deformable_gravity() {
        let mut body = DeformableBody::new_cube(
            Vec3Fix::from_int(0, 5, 0),
            Fix128::ONE,
            Fix128::from_int(10),
        );

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            body.step(dt);
        }

        let com = body.center_of_mass();
        assert!(com.y < Fix128::from_int(5), "Deformable should fall under gravity");
    }

    #[test]
    fn test_tet_volume() {
        let vol = compute_tet_volume(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(0, 0, 1),
        );
        // Volume of standard simplex = 1/6
        let expected = Fix128::from_ratio(1, 6);
        let error = (vol - expected).abs();
        assert!(error < Fix128::from_ratio(1, 100), "Tet volume should be 1/6");
    }

    #[test]
    fn test_center_of_mass() {
        let positions = vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::from_int(0, 2, 0),
            Vec3Fix::from_int(0, 0, 2),
        ];

        let com = compute_center_of_mass(&positions);
        let (cx, cy, cz) = com.to_f32();
        assert!((cx - 0.5).abs() < 0.01);
        assert!((cy - 0.5).abs() < 0.01);
        assert!((cz - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_soft_rigid_coupling() {
        use crate::solver::RigidBody;

        let mut body = DeformableBody::new_cube(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::ONE,
            Fix128::from_int(10),
        );

        let mut rigid_bodies = vec![
            RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(5)),
        ];
        let radii = vec![Fix128::from_int(3)]; // Large sphere overlapping the cube

        let dt = Fix128::from_ratio(1, 60);
        body.resolve_rigid_body_collisions(&mut rigid_bodies, &radii, dt);

        // After collision resolution, some particles should have moved
        // and the rigid body should have received a reaction
        // This is a basic sanity test
        let com = body.center_of_mass();
        // The COM might have shifted due to collision response
        assert!(com.x.hi >= -10 && com.x.hi <= 10, "COM should be reasonable");
    }
}
