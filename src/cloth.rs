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
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Cloth Configuration
// ============================================================================

/// Cloth simulation configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClothConfig {
    /// Solver iterations per substep
    pub iterations: usize,
    /// Number of substeps per step
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity retention per frame (`step()` call), applied once per frame since 1.2.0
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
#[derive(Clone, Copy, Debug, PartialEq)]
struct EdgeConstraint {
    i0: usize,
    i1: usize,
    rest_length: Fix128,
}

/// Bending constraint between two triangles sharing an edge
#[derive(Clone, Copy, Debug, PartialEq)]
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
    #[must_use]
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
                    self.edge_constraints.push(EdgeConstraint {
                        i0: a,
                        i1: b,
                        rest_length: rest,
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
    #[must_use]
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
        self.apply_frame_damping();
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
        self.apply_frame_damping();
    }

    /// `config.damping` once per frame (velocity retention per `step()` call).
    ///
    /// 1.2.0: applied per substep before, which made the terminal velocity
    /// depend on `substeps` (`g·h·d/(1−d)`, `h = dt/substeps`) — the same
    /// defect as the rigid-body solver's frame damping fix.
    fn apply_frame_damping(&mut self) {
        let d = self.config.damping;
        for v in &mut self.velocities {
            *v = *v * d;
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
    const fn compute_wind_force(&self, _particle_idx: usize) -> Vec3Fix {
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

    /// Solve bending constraints (dihedral angle).
    ///
    /// Standard position-based dihedral constraint (Müller et al. 2007,
    /// "Position Based Dynamics", Appendix A): with the shared edge `p0 p1`
    /// and the opposite vertices `p2 p3`,
    ///
    /// ```text
    /// n1 = normalize((p1-p0) × (p2-p0)),  n2 = normalize((p1-p0) × (p3-p0)),
    /// d  = n1·n2,  C = atan2(|n1×n2|, d) - φ0
    /// ```
    ///
    /// and the four gradients `q_i` of Appendix A. The correction is
    /// `Δp_i = -w_i · sin(angle) · C / (Σ w_j |q_j|² + α/dt²) · q_i`, so a
    /// flat rest configuration (`d = -1`, `sin = 0`) produces **no** correction
    /// and cannot pump energy.
    ///
    /// Before 1.1.1 this used the simplified `cos(angle) - cos(φ0)` push along
    /// the face normals. With the correct `atan2` (fixed in 1.1.1) the flat rest
    /// angle is exactly `π`, so that error term became `1 + cos(angle) ≥ 0` —
    /// sign-blind — and the cloth accumulated energy until it flew upward.
    #[inline(always)]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    fn solve_bend_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.bend_compliance / (dt * dt);

        for c_idx in 0..self.bend_constraints.len() {
            let c = self.bend_constraints[c_idx];
            let p0 = self.positions[c.i0];
            // Translate so that p0 is the origin (Appendix A convention p1 = 0)
            let p1 = self.positions[c.i1] - p0;
            let p2 = self.positions[c.i2] - p0;
            let p3 = self.positions[c.i3] - p0;

            let c12 = p1.cross(p2);
            let c13 = p1.cross(p3);
            let len12 = c12.length();
            let len13 = c13.length();
            if len12.is_zero() || len13.is_zero() {
                continue;
            }
            let inv12 = Fix128::ONE / len12;
            let inv13 = Fix128::ONE / len13;
            let n1 = c12 * inv12;
            let n2 = c13 * inv13;

            let d = n1.dot(n2);
            let sin_angle = n1.cross(n2).length();
            let angle = Fix128::atan2(sin_angle, d);
            let error = angle - c.rest_angle;

            // Gradients (Appendix A, with p1 := our p1 (edge end), p3 := p2, p4 := p3)
            let q2 = (p1.cross(n2) + (n1.cross(p1)) * d) * inv12;
            let q3 = (p1.cross(n1) + (n2.cross(p1)) * d) * inv13;
            let q1 = (p2.cross(n2) + (n1.cross(p2)) * d) * (-inv12)
                + (p3.cross(n1) + (n2.cross(p3)) * d) * (-inv13);
            let q0 = -(q1 + q2 + q3);

            let w0 = self.inv_masses[c.i0];
            let w1 = self.inv_masses[c.i1];
            let w2 = self.inv_masses[c.i2];
            let w3 = self.inv_masses[c.i3];
            let denom = w0 * q0.length_squared()
                + w1 * q1.length_squared()
                + w2 * q2.length_squared()
                + w3 * q3.length_squared()
                + compliance;
            if denom.is_zero() {
                continue;
            }

            // s = sin(angle) · C / denom (sqrt(1 - d²) == |n1 × n2|)
            let scale = sin_angle * error / denom;
            if scale.is_zero() {
                continue;
            }

            if !w0.is_zero() {
                self.positions[c.i0] = self.positions[c.i0] - q0 * (w0 * scale);
            }
            if !w1.is_zero() {
                self.positions[c.i1] = self.positions[c.i1] - q1 * (w1 * scale);
            }
            if !w2.is_zero() {
                self.positions[c.i2] = self.positions[c.i2] - q2 * (w2 * scale);
            }
            if !w3.is_zero() {
                self.positions[c.i3] = self.positions[c.i3] - q3 * (w3 * scale);
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

        // Build spatial hash grid (2-pass CSR)
        let total_cells = grid_dim * grid_dim * grid_dim;

        // Pass 1: count particles per cell
        let mut cell_counts = vec![0usize; total_cells];
        // Temporary (cell, particle) pairs to avoid re-hashing in pass 2
        let mut cell_particle: Vec<(usize, usize)> = Vec::with_capacity(n);
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
                cell_counts[h] += 1;
                cell_particle.push((h, i));
            }
        }
        // Prefix sum → cell_offsets
        let mut cell_offsets = vec![0usize; total_cells + 1];
        for h in 0..total_cells {
            cell_offsets[h + 1] = cell_offsets[h] + cell_counts[h];
            cell_counts[h] = 0; // reuse as write cursor
        }
        // Pass 2: fill flat index buffer
        let total_particles = cell_offsets[total_cells];
        let mut indices = vec![0usize; total_particles];
        for (h, i) in &cell_particle {
            let slot = cell_offsets[*h] + cell_counts[*h];
            indices[slot] = *i;
            cell_counts[*h] += 1;
        }

        // Build sorted edge set for O(log n) lookup (skip connected particles)
        let mut edge_pairs: Vec<(usize, usize)> = Vec::with_capacity(self.edge_constraints.len());
        for c in &self.edge_constraints {
            let (lo, hi) = if c.i0 < c.i1 {
                (c.i0, c.i1)
            } else {
                (c.i1, c.i0)
            };
            edge_pairs.push((lo, hi));
        }
        edge_pairs.sort_unstable();

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
                        let start = cell_offsets[h];
                        let end = cell_offsets[h + 1];
                        for &j in &indices[start..end] {
                            if j <= i {
                                continue; // avoid duplicate pairs
                            }
                            if self.inv_masses[j].is_zero() {
                                continue;
                            }

                            // Skip particles connected by an edge (binary search)
                            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                            if edge_pairs.binary_search(&(lo, hi)).is_ok() {
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
    #[must_use]
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

impl core::fmt::Debug for Cloth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cloth")
            .field(
                "positions",
                &format_args!("[{} items]", self.positions.len()),
            )
            .field(
                "prev_positions",
                &format_args!("[{} items]", self.prev_positions.len()),
            )
            .field(
                "velocities",
                &format_args!("[{} items]", self.velocities.len()),
            )
            .field(
                "inv_masses",
                &format_args!("[{} items]", self.inv_masses.len()),
            )
            .field(
                "triangles",
                &format_args!("[{} items]", self.triangles.len()),
            )
            .field(
                "edge_constraints",
                &format_args!("[{} items]", self.edge_constraints.len()),
            )
            .field(
                "bend_constraints",
                &format_args!("[{} items]", self.bend_constraints.len()),
            )
            .field("pinned", &format_args!("[{} items]", self.pinned.len()))
            .field("wind", &self.wind)
            .field("config", &self.config)
            .finish()
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
                    // Compute actual dihedral angle via atan2(sin, cos)
                    let n1u = n1 / n1_len;
                    let n2u = n2 / n2_len;
                    let cos_val = n1u.dot(n2u);
                    let sin_val = n1u.cross(n2u).length();
                    Fix128::atan2(sin_val, cos_val)
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
            "Bottom should be below top, top={initial_y}, bottom={bottom_y}"
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

    // ---- bending (dihedral) constraint, 1.1.1 ------------------------

    /// 2 三角形 (0,1,2) / (1,0,3) が edge 0-1 を共有する最小 cloth、全 particle 質量 1
    fn two_triangle_cloth() -> Cloth {
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ONE,
            2,
            2,
            Fix128::from_ratio(1, 100),
        );
        assert_eq!(cloth.particle_count(), 4);
        assert_eq!(
            cloth.bend_constraints.len(),
            1,
            "2x2 grid = 2 triangles = 1 shared edge"
        );
        cloth.config.bend_compliance = Fix128::ZERO;
        cloth
    }

    fn dihedral(cloth: &Cloth) -> Fix128 {
        let c = cloth.bend_constraints[0];
        let p0 = cloth.positions[c.i0];
        let e = cloth.positions[c.i1] - p0;
        let n1 = e.cross(cloth.positions[c.i2] - p0).normalize();
        let n2 = e.cross(cloth.positions[c.i3] - p0).normalize();
        Fix128::atan2(n1.cross(n2).length(), n1.dot(n2))
    }

    #[test]
    fn bend_flat_rest_is_pi_and_flat_cloth_is_a_fixed_point() {
        let mut cloth = two_triangle_cloth();
        // 平面の隣接三角形: 法線は逆向き → rest = π (1.1.1 の atan2 修正で正確になった)
        let rest = cloth.bend_constraints[0].rest_angle;
        assert!(
            (rest - Fix128::PI).abs() < Fix128::from_ratio(1, 1_000_000),
            "rest {rest:?}"
        );
        let before = cloth.positions.clone();
        for _ in 0..50 {
            cloth.solve_bend_constraints(Fix128::from_ratio(1, 60));
        }
        // sin(π) = 0 → 補正 0、bit 単位で不動 (旧実装は 1 + cos ≥ 0 で押し続けた)
        assert_eq!(cloth.positions, before);
    }

    #[test]
    fn bend_folded_cloth_relaxes_monotonically_toward_flat() {
        let mut cloth = two_triangle_cloth();
        let c = cloth.bend_constraints[0];
        // grid は x-z 平面なので法線方向 = y に持ち上げて折る
        cloth.positions[c.i3].y = Fix128::from_ratio(1, 2);
        let rest = cloth.bend_constraints[0].rest_angle;
        let mut prev_err = (dihedral(&cloth) - rest).abs();
        assert!(
            prev_err > Fix128::from_ratio(1, 10),
            "初期折れ角 {prev_err:?}"
        );
        for iter in 0..40 {
            cloth.solve_bend_constraints(Fix128::from_ratio(1, 60));
            let err = (dihedral(&cloth) - rest).abs();
            assert!(
                err <= prev_err,
                "iter {iter}: error grew {prev_err:?} -> {err:?}"
            );
            prev_err = err;
        }
        assert!(
            prev_err < Fix128::from_ratio(1, 100),
            "40 iter 後も {prev_err:?}"
        );
        // 位置が有限範囲に留まる (発散なし)
        for p in &cloth.positions {
            assert!(p.length() < Fix128::from_int(4), "{p:?}");
        }
    }

    #[test]
    fn bend_pinned_vertices_do_not_move() {
        let mut cloth = two_triangle_cloth();
        let c = cloth.bend_constraints[0];
        cloth.inv_masses[c.i0] = Fix128::ZERO;
        cloth.inv_masses[c.i1] = Fix128::ZERO;
        cloth.inv_masses[c.i2] = Fix128::ZERO;
        cloth.positions[c.i3].y = Fix128::from_ratio(1, 2);
        let fixed = [
            cloth.positions[c.i0],
            cloth.positions[c.i1],
            cloth.positions[c.i2],
        ];
        let before_i3 = cloth.positions[c.i3];
        for _ in 0..10 {
            cloth.solve_bend_constraints(Fix128::from_ratio(1, 60));
        }
        assert_eq!(cloth.positions[c.i0], fixed[0]);
        assert_eq!(cloth.positions[c.i1], fixed[1]);
        assert_eq!(cloth.positions[c.i2], fixed[2]);
        assert!(cloth.positions[c.i3] != before_i3, "自由頂点だけが動く");
    }

    #[test]
    fn drape_bottom_row_ends_below_pinned_top_and_above_free_fall() {
        // 上段 pin、120 step: 下段は下がる (drape) が、拘束があるので自由落下 (y = -g t²/2) より上
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            5,
            5,
            Fix128::from_ratio(1, 100),
        );
        cloth.pin_top_row(5);
        let start_bottom = cloth.positions[20].y;
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            cloth.step(dt);
        }
        let bottom = cloth.positions[20].y;
        assert!(
            bottom < start_bottom,
            "下がる: {start_bottom:?} -> {bottom:?}"
        );
        assert!(bottom < cloth.positions[0].y, "top より下");
        // 2 秒の自由落下 = -19.6 より上 (布は吊られている)
        assert!(bottom > Fix128::from_int(-20), "発散していない {bottom:?}");
        for p in &cloth.positions {
            assert!(p.length() < Fix128::from_int(30), "{p:?}");
        }
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
    fn step_with_sdf_drapes_cloth_over_unit_sphere_without_penetration() {
        // 2×2 の 5×5 grid を y = 2 に水平配置、中心 particle (index 12) が (0, 2, 0)
        let make = || {
            Cloth::new_grid(
                Vec3Fix::from_int(-1, 2, -1),
                Fix128::from_int(2),
                Fix128::from_int(2),
                5,
                5,
                Fix128::from_ratio(1, 100),
            )
        };
        let sphere = [unit_sphere_collider()];
        let mut cloth = make();
        let thickness = cloth.config.thickness.to_f32();
        let dt = Fix128::from_ratio(1, 60);
        let mut min_dist = f32::MAX;
        for frame in 0..60 {
            cloth.step_with_sdf(dt, &sphere);
            // 各 frame 終了時: 全 free particle は thickness 以上 球の外 (push-out は substep 末尾)
            for (i, p) in cloth.positions.iter().enumerate() {
                let d = sphere_dist(*p);
                min_dist = min_dist.min(d);
                assert!(
                    d >= thickness - 1e-3,
                    "frame {frame} particle {i} dist {d} < thickness {thickness}"
                );
            }
        }
        // 実際に接触している (自明に真ではない)
        assert!(
            min_dist < thickness + 0.05,
            "never touched: min dist {min_dist}"
        );
        // 中心 particle は球頂点 (y = 1 + thickness) 付近で静止
        let (cx, cy, cz) = cloth.positions[12].to_f32();
        assert!(
            (0.99..=1.1).contains(&cy),
            "center particle y {cy} (x {cx}, z {cz})"
        );
        assert!(
            cx.abs() < 0.1 && cz.abs() < 0.1,
            "center drifted: ({cx}, {cz})"
        );

        // SDF なしなら同じ布は球を素通りして落下する (1 s、-10 m/s² → y ≈ -3)
        let mut free = make();
        for _ in 0..60 {
            free.step(dt);
        }
        let (_, fy, _) = free.positions[12].to_f32();
        assert!(
            fy < 0.0,
            "without SDF the cloth should fall through: y {fy}"
        );

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
