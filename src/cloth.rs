//! Cloth / Shell Simulation (XPBD Triangle Mesh)
//!
//! Position-based cloth using distance + bending constraints
//! on a triangle mesh. Supports SDF collision.
//!
//! # Features
//!
//! - Distance constraints on mesh edges (stretch resistance)
//! - Bending constraints on adjacent triangles
//! - SDF collision for cloth-body interaction
//! - Pin constraints for attaching cloth to bodies
//! - Wind force and gravity
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Cloth Configuration
// ============================================================================

/// Cloth simulation configuration
#[derive(Clone, Copy, Debug)]
pub struct ClothConfig {
    /// Solver iterations per substep
    pub iterations: usize,
    /// Number of substeps per step
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity damping (0..1)
    pub damping: Fix128,
    /// Stretch constraint compliance (0 = rigid edges)
    pub stretch_compliance: Fix128,
    /// Bending constraint compliance (higher = more flexible)
    pub bend_compliance: Fix128,
    /// SDF collision friction
    pub sdf_friction: Fix128,
    /// Cloth thickness for collision
    pub thickness: Fix128,
    /// Enable self-collision (particle-vs-particle)
    pub self_collision: bool,
    /// Minimum distance between particles for self-collision
    pub self_collision_distance: Fix128,
}

impl Default for ClothConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            substeps: 4,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            stretch_compliance: Fix128::ZERO,
            bend_compliance: Fix128::from_ratio(1, 100),
            sdf_friction: Fix128::from_ratio(2, 10),
            thickness: Fix128::from_ratio(1, 100),
            self_collision: false,
            self_collision_distance: Fix128::from_ratio(2, 100), // 0.02
        }
    }
}

// ============================================================================
// Cloth Constraints
// ============================================================================

/// Edge (stretch) constraint
#[derive(Clone, Copy, Debug)]
struct EdgeConstraint {
    i0: usize,
    i1: usize,
    rest_length: Fix128,
    /// Precomputed reciprocal of rest_length — avoids per-iteration division.
    #[allow(dead_code)]
    inv_rest_length: Fix128,
}

/// Bending constraint between two triangles sharing an edge
#[derive(Clone, Copy, Debug)]
struct BendConstraint {
    /// The four vertices: shared edge (i0, i1) and opposite vertices (i2, i3)
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
    /// Rest dihedral angle
    rest_angle: Fix128,
}

// ============================================================================
// Cloth
// ============================================================================

/// Cloth simulation
#[repr(C, align(64))]
pub struct Cloth {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Previous positions
    pub prev_positions: Vec<Vec3Fix>,
    /// Velocities
    pub velocities: Vec<Vec3Fix>,
    /// Inverse mass per particle
    pub inv_masses: Vec<Fix128>,
    /// Triangle indices (i0, i1, i2)
    pub triangles: Vec<[usize; 3]>,
    /// Edge constraints
    edge_constraints: Vec<EdgeConstraint>,
    /// Bending constraints
    bend_constraints: Vec<BendConstraint>,
    /// Pinned particles
    pub pinned: Vec<usize>,
    /// Wind force
    pub wind: Vec3Fix,
    /// Configuration
    pub config: ClothConfig,
}

impl Cloth {
    /// Create a rectangular cloth grid
    ///
    /// `width` x `height` in world units, `res_x` x `res_y` particles
    pub fn new_grid(
        origin: Vec3Fix,
        width: Fix128,
        height: Fix128,
        res_x: usize,
        res_y: usize,
        mass_per_particle: Fix128,
    ) -> Self {
        let n = res_x * res_y;
        let inv_mass = if mass_per_particle.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / mass_per_particle
        };

        // Precompute reciprocals for grid UV generation — avoids per-iteration division.
        let recip_res_x = if res_x > 1 {
            Fix128::ONE / Fix128::from_int((res_x - 1) as i64)
        } else {
            Fix128::ZERO
        };
        let recip_res_y = if res_y > 1 {
            Fix128::ONE / Fix128::from_int((res_y - 1) as i64)
        } else {
            Fix128::ZERO
        };

        // Generate particles
        let mut positions = Vec::with_capacity(n);
        for j in 0..res_y {
            for i in 0..res_x {
                let u = Fix128::from_int(i as i64) * recip_res_x;
                let v = Fix128::from_int(j as i64) * recip_res_y;
                positions.push(Vec3Fix::new(
                    origin.x + width * u,
                    origin.y,
                    origin.z + height * v,
                ));
            }
        }

        // Generate triangles
        let mut triangles = Vec::new();
        for j in 0..(res_y - 1) {
            for i in 0..(res_x - 1) {
                let i00 = j * res_x + i;
                let i10 = i00 + 1;
                let i01 = i00 + res_x;
                let i11 = i01 + 1;

                triangles.push([i00, i10, i01]);
                triangles.push([i10, i11, i01]);
            }
        }

        let mut cloth = Self {
            prev_positions: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            positions,
            inv_masses: vec![inv_mass; n],
            triangles,
            edge_constraints: Vec::new(),
            bend_constraints: Vec::new(),
            pinned: Vec::new(),
            wind: Vec3Fix::ZERO,
            config: ClothConfig::default(),
        };

        cloth.build_constraints();
        cloth
    }

    /// Build edge and bending constraints from triangle mesh
    fn build_constraints(&mut self) {
        // Edge constraints: collect unique edges from triangles
        let mut edge_set: Vec<(usize, usize)> = Vec::new();

        for tri in &self.triangles {
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for (a, b) in edges {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if !edge_set.contains(&(lo, hi)) {
                    edge_set.push((lo, hi));
                    let rest = (self.positions[a] - self.positions[b]).length();
                    // Precompute inv_rest_length once at construction time.
                    let inv_rest = if rest.is_zero() {
                        Fix128::ZERO
                    } else {
                        Fix128::ONE / rest
                    };
                    self.edge_constraints.push(EdgeConstraint {
                        i0: a,
                        i1: b,
                        rest_length: rest,
                        inv_rest_length: inv_rest,
                    });
                }
            }
        }

        // Bending constraints: find triangles sharing edges
        for i in 0..self.triangles.len() {
            for j in (i + 1)..self.triangles.len() {
                if let Some(bend) =
                    find_shared_edge(&self.triangles[i], &self.triangles[j], &self.positions)
                {
                    self.bend_constraints.push(bend);
                }
            }
        }
    }

    /// Number of particles
    #[inline(always)]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Pin a particle in place
    pub fn pin(&mut self, idx: usize) {
        self.inv_masses[idx] = Fix128::ZERO;
        if !self.pinned.contains(&idx) {
            self.pinned.push(idx);
        }
    }

    /// Pin the top row (for curtain-like behavior)
    pub fn pin_top_row(&mut self, res_x: usize) {
        for i in 0..res_x {
            self.pin(i);
        }
    }

    /// Step cloth simulation
    #[inline(always)]
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
    }

    /// Step with SDF collision
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
    }

    /// Single substep
    #[inline(always)]
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();

        // 1. Predict positions
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.prev_positions[i] = self.positions[i];

            // Gravity + wind
            let wind_force = self.compute_wind_force(i);
            self.velocities[i] =
                self.velocities[i] + (self.config.gravity + wind_force * self.inv_masses[i]) * dt;
            self.velocities[i] = self.velocities[i] * self.config.damping;
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Solve constraints
        for _ in 0..self.config.iterations {
            self.solve_edge_constraints(dt);
            self.solve_bend_constraints(dt);
            if self.config.self_collision {
                self.solve_self_collision();
            }
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

    /// Compute approximate wind force on a particle
    #[inline(always)]
    fn compute_wind_force(&self, _particle_idx: usize) -> Vec3Fix {
        // Simplified: uniform wind force
        self.wind
    }

    /// Solve edge (stretch) constraints
    #[inline(always)]
    fn solve_edge_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.stretch_compliance / (dt * dt);

        for c_idx in 0..self.edge_constraints.len() {
            let c = self.edge_constraints[c_idx];
            let p0 = self.positions[c.i0];
            let p1 = self.positions[c.i1];
            let w0 = self.inv_masses[c.i0];
            let w1 = self.inv_masses[c.i1];

            let w_sum = w0 + w1 + compliance;
            if w_sum.is_zero() {
                continue;
            }

            let delta = p1 - p0;
            let dist = delta.length();
            if dist.is_zero() {
                continue;
            }

            let error = dist - c.rest_length;
            let lambda = error / w_sum;
            // Use precomputed reciprocal to replace per-iteration division by dist.
            let inv_dist = Fix128::ONE / dist;
            let correction = delta * inv_dist * lambda;

            if !w0.is_zero() {
                self.positions[c.i0] = self.positions[c.i0] + correction * w0;
            }
            if !w1.is_zero() {
                self.positions[c.i1] = self.positions[c.i1] - correction * w1;
            }
        }
    }

    /// Solve bending constraints (dihedral angle)
    #[inline(always)]
    fn solve_bend_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.bend_compliance / (dt * dt);

        for c_idx in 0..self.bend_constraints.len() {
            let c = self.bend_constraints[c_idx];
            let p0 = self.positions[c.i0];
            let p1 = self.positions[c.i1];
            let p2 = self.positions[c.i2];
            let p3 = self.positions[c.i3];

            // Compute dihedral angle between triangles (p0,p1,p2) and (p0,p1,p3)
            let edge = p1 - p0;
            let n1 = edge.cross(p2 - p0);
            let n2 = edge.cross(p3 - p0);

            let n1_len = n1.length();
            let n2_len = n2.length();
            if n1_len.is_zero() || n2_len.is_zero() {
                continue;
            }

            // Precompute reciprocals to replace two per-constraint divisions.
            let inv_n1_len = Fix128::ONE / n1_len;
            let inv_n2_len = Fix128::ONE / n2_len;
            let n1_norm = n1 * inv_n1_len;
            let n2_norm = n2 * inv_n2_len;

            let cos_angle = n1_norm.dot(n2_norm);
            let cos_rest = c.rest_angle.cos();
            let error = cos_angle - cos_rest;

            // Simplified bending: push opposite vertices toward rest angle
            let w2 = self.inv_masses[c.i2];
            let w3 = self.inv_masses[c.i3];
            let w_sum = w2 + w3 + compliance;
            if w_sum.is_zero() {
                continue;
            }

            let lambda = error / w_sum;
            let grad2 = n1_norm * lambda;
            let grad3 = n2_norm * (-lambda);

            if !w2.is_zero() {
                self.positions[c.i2] = self.positions[c.i2] - grad2 * w2;
            }
            if !w3.is_zero() {
                self.positions[c.i3] = self.positions[c.i3] - grad3 * w3;
            }
        }
    }

    /// Solve self-collision using spatial hash grid.
    ///
    /// Particles closer than `self_collision_distance` are pushed apart.
    /// Uses a spatial hash grid (same pattern as fluid.rs) for O(n) neighbor search.
    #[inline(always)]
    fn solve_self_collision(&mut self) {
        let n = self.particle_count();
        if n < 2 {
            return;
        }

        let min_dist = self.config.self_collision_distance;
        let min_dist_sq = min_dist * min_dist;
        let cell_size = min_dist * Fix128::from_int(2);
        let inv_cell = if cell_size.is_zero() {
            Fix128::ONE
        } else {
            Fix128::ONE / cell_size
        };
        let grid_dim: usize = 64;
        let grid_dim_i64 = grid_dim as i64;
        let half = grid_dim_i64 / 2;

        // Build spatial hash grid
        let total_cells = grid_dim * grid_dim * grid_dim;
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); total_cells];

        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            let p = self.positions[i];
            let ix = ((p.x * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as usize;
            let iy = ((p.y * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as usize;
            let iz = ((p.z * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as usize;
            let h = ix + iy * grid_dim + iz * grid_dim * grid_dim;
            if h < total_cells {
                cells[h].push(i);
            }
        }

        // Build edge set for quick lookup (skip connected particles)
        // Use a simple O(1) hash check: particle pairs that share an edge
        // should not have self-collision applied
        let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
        for c in &self.edge_constraints {
            let (lo, hi) = if c.i0 < c.i1 {
                (c.i0, c.i1)
            } else {
                (c.i1, c.i0)
            };
            edge_pairs.push((lo, hi));
        }

        // Check neighbors and apply separation constraints
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            let p = self.positions[i];
            let cx = ((p.x * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as i32;
            let cy = ((p.y * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as i32;
            let cz = ((p.z * inv_cell).hi + half).clamp(0, grid_dim_i64 - 1) as i32;

            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = cx + dx;
                        let ny = cy + dy;
                        let nz = cz + dz;
                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        let nz = nz as usize;
                        if nx >= grid_dim || ny >= grid_dim || nz >= grid_dim {
                            continue;
                        }

                        let h = nx + ny * grid_dim + nz * grid_dim * grid_dim;
                        for &j in &cells[h] {
                            if j <= i {
                                continue; // avoid duplicate pairs
                            }
                            if self.inv_masses[j].is_zero() {
                                continue;
                            }

                            // Skip particles connected by an edge
                            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                            if edge_pairs.contains(&(lo, hi)) {
                                continue;
                            }

                            let delta = self.positions[j] - self.positions[i];
                            let dist_sq = delta.length_squared();

                            if dist_sq < min_dist_sq && !dist_sq.is_zero() {
                                let dist = dist_sq.sqrt();
                                let error = min_dist - dist;
                                // Precompute reciprocals to replace per-pair divisions.
                                let inv_dist = Fix128::ONE / dist;
                                let normal = delta * inv_dist;

                                let w_sum = self.inv_masses[i] + self.inv_masses[j];
                                if w_sum.is_zero() {
                                    continue;
                                }

                                let inv_w_sum = Fix128::ONE / w_sum;
                                let correction = normal * (error * inv_w_sum);

                                self.positions[i] =
                                    self.positions[i] - correction * self.inv_masses[i];
                                self.positions[j] =
                                    self.positions[j] + correction * self.inv_masses[j];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Resolve SDF collisions for all particles
    #[cfg(feature = "std")]
    #[inline(always)]
    fn resolve_sdf_collisions(&mut self, sdf_colliders: &[SdfCollider]) {
        let thickness = self.config.thickness.to_f32();

        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            for sdf in sdf_colliders {
                let (lx, ly, lz) = sdf.world_to_local(self.positions[i]);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;

                if dist < thickness {
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let push = Fix128::from_f32(thickness - dist);

                    self.positions[i] = self.positions[i] + normal * push;

                    // Friction
                    let vel = self.velocities[i];
                    let vn = normal * vel.dot(normal);
                    let vt = vel - vn;
                    self.velocities[i] = vt * (Fix128::ONE - self.config.sdf_friction);
                }
            }
        }
    }

    /// Compute per-triangle normals (for rendering)
    pub fn compute_normals(&self) -> Vec<Vec3Fix> {
        let mut normals = vec![Vec3Fix::ZERO; self.particle_count()];

        for tri in &self.triangles {
            let e1 = self.positions[tri[1]] - self.positions[tri[0]];
            let e2 = self.positions[tri[2]] - self.positions[tri[0]];
            let face_normal = e1.cross(e2);

            normals[tri[0]] = normals[tri[0]] + face_normal;
            normals[tri[1]] = normals[tri[1]] + face_normal;
            normals[tri[2]] = normals[tri[2]] + face_normal;
        }

        for n in &mut normals {
            *n = n.normalize();
        }

        normals
    }
}

/// Find shared edge between two triangles and create a bending constraint
fn find_shared_edge(
    tri_a: &[usize; 3],
    tri_b: &[usize; 3],
    positions: &[Vec3Fix],
) -> Option<BendConstraint> {
    for ia in 0..3 {
        for ib in 0..3 {
            let a0 = tri_a[ia];
            let a1 = tri_a[(ia + 1) % 3];
            let b0 = tri_b[ib];
            let b1 = tri_b[(ib + 1) % 3];

            if (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0) {
                // Shared edge found
                let opposite_a = tri_a[(ia + 2) % 3];
                let opposite_b = tri_b[(ib + 2) % 3];

                // Compute rest dihedral angle
                let edge = positions[a1] - positions[a0];
                let n1 = edge.cross(positions[opposite_a] - positions[a0]);
                let n2 = edge.cross(positions[opposite_b] - positions[a0]);

                let n1_len = n1.length();
                let n2_len = n2.length();
                let rest_angle = if n1_len.is_zero() || n2_len.is_zero() {
                    Fix128::ZERO
                } else {
                    
                    (n1 / n1_len).dot(n2 / n2_len) // Store cos(angle) directly
                };

                return Some(BendConstraint {
                    i0: a0,
                    i1: a1,
                    i2: opposite_a,
                    i3: opposite_b,
                    rest_angle,
                });
            }
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloth_creation() {
        let cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            5,
            5,
            Fix128::from_ratio(1, 100),
        );

        assert_eq!(cloth.particle_count(), 25);
        assert!(
            !cloth.edge_constraints.is_empty(),
            "Should have edge constraints"
        );
    }

    #[test]
    fn test_cloth_pinning() {
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            5,
            5,
            Fix128::from_ratio(1, 100),
        );

        cloth.pin_top_row(5);
        assert_eq!(cloth.pinned.len(), 5);

        for i in 0..5 {
            assert!(cloth.inv_masses[i].is_zero(), "Top row should be pinned");
        }
    }

    #[test]
    fn test_cloth_drape() {
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            5,
            5,
            Fix128::from_ratio(1, 100),
        );
        cloth.pin_top_row(5);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            cloth.step(dt);
        }

        // Bottom row should drape below starting position
        // With constraints, the cloth stretches slowly; verify it moved downward
        let initial_y = cloth.positions[0].y.to_f32(); // top row (pinned)
        let bottom_y = cloth.positions[20].y.to_f32(); // bottom row
        assert!(
            bottom_y < initial_y,
            "Bottom should be below top, top={}, bottom={}",
            initial_y,
            bottom_y
        );
    }

    #[test]
    fn test_cloth_self_collision() {
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            5,
            5,
            Fix128::from_ratio(1, 100),
        );
        cloth.config.self_collision = true;
        cloth.config.self_collision_distance = Fix128::from_ratio(5, 100); // 0.05
        cloth.pin_top_row(5);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            cloth.step(dt);
        }

        // Verify no two non-connected particles are closer than self_collision_distance
        // (This is a basic sanity check; exact enforcement depends on iterations)
        let min_d = cloth.config.self_collision_distance.to_f32();
        let n = cloth.particle_count();
        let mut too_close = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (cloth.positions[i] - cloth.positions[j]).length().to_f32();
                if d < min_d * 0.5 {
                    // allow some tolerance
                    too_close += 1;
                }
            }
        }
        // With self-collision enabled, severely overlapping particles should be rare
        assert!(
            too_close < n,
            "Self-collision should prevent extreme particle overlap"
        );
    }

    #[test]
    fn test_cloth_normals() {
        let cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            3,
            3,
            Fix128::from_ratio(1, 100),
        );

        let normals = cloth.compute_normals();
        assert_eq!(normals.len(), 9);

        // For a flat grid in XZ plane, normals should point in Y
        for n in &normals {
            let (_, ny, _) = n.to_f32();
            assert!(
                ny.abs() > 0.5,
                "Normals should point mostly in Y for flat grid"
            );
        }
    }
}
