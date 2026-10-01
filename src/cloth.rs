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
    /// Enable self-collision
    ///
    /// Covers both proximity passes — vertex-vs-face and edge-vs-edge — plus the
    /// frame-level repair of vertex-face tunnelling. ⚠️ Edge-edge is **proximity only**:
    /// there is no swept (continuous) edge-edge test, so a crossing that happens entirely
    /// inside one frame and ends further apart than `self_collision_distance` is not
    /// repaired. Two parallel (or, in `Fix128`, numerically parallel) edges are also left
    /// to the vertex-face pass — see `closest_points_on_segments`.
    pub self_collision: bool,
    /// Contact thickness: a vertex is kept at least this far from every non-incident triangle
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
        let frame_start = self.snapshot_for_self_contact();
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
        if let Some(start) = frame_start {
            self.resolve_self_contact_over_frame(&start, dt);
        }
        self.apply_frame_damping();
    }

    /// Step with SDF collision
    #[cfg(feature = "std")]
    #[inline(always)]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let frame_start = self.snapshot_for_self_contact();
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
        if let Some(start) = frame_start {
            self.resolve_self_contact_over_frame(&start, dt);
        }
        self.apply_frame_damping();
    }

    /// Frame-start positions, kept only when self-contact is enabled.
    ///
    /// `None` (rather than an unconditional clone) so that `self_collision = false`
    /// allocates nothing and stays bit-identical to the pre-1.4.x behaviour.
    fn snapshot_for_self_contact(&self) -> Option<Vec<Vec3Fix>> {
        if self.config.self_collision && !self.config.self_collision_distance.is_zero() {
            Some(self.positions.clone())
        } else {
            None
        }
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
        //
        // Self-contact candidates are collected once here rather than once per
        // iteration: the AABB margin (2·thickness) covers how far the iterations can
        // move a particle, and the collection is the O(V·T) part.
        let margin = self.config.self_collision_distance.double();
        let (candidates, edge_candidates) = if self.config.self_collision {
            (
                self.collect_vertex_face_candidates(margin),
                self.collect_edge_edge_candidates(margin),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        for _ in 0..self.config.iterations {
            self.solve_edge_constraints(dt);
            self.solve_bend_constraints(dt);
            if self.config.self_collision {
                self.solve_self_collision(&candidates, &edge_candidates);
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

    /// Candidate (vertex, triangle) pairs for vertex-face self-contact.
    ///
    /// Built **once per substep** (not once per solver iteration) from the predicted
    /// positions: the iterations only move particles by the constraint residual, so an
    /// AABB inflated by `margin` covers the whole iteration sweep. A pair is kept when
    /// the vertex lies inside the triangle's AABB grown by `margin` on every axis.
    ///
    /// Excluded: vertices incident to the triangle (they are handled by the stretch and
    /// bending constraints), and pairs in which all four participants are pinned
    /// (`inv_mass == 0`), which can produce no correction.
    ///
    /// ⚠️ **Pinned vertices are kept as collision participants.** They cannot move, but a
    /// kinematically driven vertex sweeping into a free triangle must still push that
    /// triangle away; dropping them was the reason the old particle-particle pass never
    /// saw the driven boundary of a crumpling sheet.
    fn collect_vertex_face_candidates(&self, margin: Fix128) -> Vec<(u32, u32)> {
        let n = self.particle_count();
        let mut out: Vec<(u32, u32)> = Vec::new();
        for (ti, tri) in self.triangles.iter().enumerate() {
            let (a, b, c) = (
                self.positions[tri[0]],
                self.positions[tri[1]],
                self.positions[tri[2]],
            );
            let tri_pinned = self.inv_masses[tri[0]].is_zero()
                && self.inv_masses[tri[1]].is_zero()
                && self.inv_masses[tri[2]].is_zero();
            let lo = Vec3Fix::new(
                min3(a.x, b.x, c.x) - margin,
                min3(a.y, b.y, c.y) - margin,
                min3(a.z, b.z, c.z) - margin,
            );
            let hi = Vec3Fix::new(
                max3(a.x, b.x, c.x) + margin,
                max3(a.y, b.y, c.y) + margin,
                max3(a.z, b.z, c.z) + margin,
            );
            for i in 0..n {
                if i == tri[0] || i == tri[1] || i == tri[2] {
                    continue;
                }
                if tri_pinned && self.inv_masses[i].is_zero() {
                    continue; // nothing in this pair can move
                }
                let p = self.positions[i];
                if p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y || p.z < lo.z || p.z > hi.z
                {
                    continue;
                }
                out.push((i as u32, ti as u32));
            }
        }
        out
    }

    /// Unique undirected mesh edges, as sorted `[lo, hi]` vertex pairs.
    ///
    /// Rebuilt from `triangles` rather than read off `edge_constraints`, because the two
    /// are not the same set in practice: a caller (or a test) may clear the stretch
    /// constraints to look at collision in isolation, and collision topology must not
    /// disappear with them. The result is sorted, so the pair enumeration below is a
    /// function of the mesh alone and not of triangle insertion order.
    fn mesh_edges(&self) -> Vec<[u32; 2]> {
        let mut edges: Vec<[u32; 2]> = Vec::with_capacity(self.triangles.len() * 3);
        for tri in &self.triangles {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                edges.push([lo as u32, hi as u32]);
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// Edge pairs whose `margin`-expanded AABBs overlap, as `[i0, i1, j0, j1]`.
    ///
    /// Two exclusions, both load-bearing:
    ///
    /// - ⚠️ **Edges that share a vertex are dropped.** Their closest distance is zero by
    ///   construction, so every one of them would report a contact at every iteration and
    ///   the pass would do nothing but fight the stretch constraints
    ///   (`edge_edge_candidates_never_pair_two_edges_that_share_a_vertex` pins this; the
    ///   mutation that removes it moves the vertices of a scene whose closed form says
    ///   they must not move).
    /// - Pairs in which all four participants are pinned, which can produce no correction.
    ///
    /// Collinear and parallel pairs are *not* excluded here — they are rejected inside
    /// `closest_points_on_segments`, which is where "no isolated closest pair" is decided.
    /// A flat grid's row-mates are exactly that case, which is why this pass does not fire
    /// on an undeformed sheet.
    fn collect_edge_edge_candidates(&self, margin: Fix128) -> Vec<[u32; 4]> {
        let edges = self.mesh_edges();
        let mut out: Vec<[u32; 4]> = Vec::new();
        for (ei, e1) in edges.iter().enumerate() {
            let (a0, a1) = (e1[0] as usize, e1[1] as usize);
            let (pa0, pa1) = (self.positions[a0], self.positions[a1]);
            let lo1 = Vec3Fix::new(
                min2(pa0.x, pa1.x) - margin,
                min2(pa0.y, pa1.y) - margin,
                min2(pa0.z, pa1.z) - margin,
            );
            let hi1 = Vec3Fix::new(
                max2(pa0.x, pa1.x) + margin,
                max2(pa0.y, pa1.y) + margin,
                max2(pa0.z, pa1.z) + margin,
            );
            let both_pinned_1 = self.inv_masses[a0].is_zero() && self.inv_masses[a1].is_zero();
            for e2 in &edges[ei + 1..] {
                let (b0, b1) = (e2[0] as usize, e2[1] as usize);
                if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                    continue; // shares a vertex: distance is 0 by construction
                }
                if both_pinned_1 && self.inv_masses[b0].is_zero() && self.inv_masses[b1].is_zero() {
                    continue; // nothing in this pair can move
                }
                let (pb0, pb1) = (self.positions[b0], self.positions[b1]);
                if min2(pb0.x, pb1.x) > hi1.x
                    || max2(pb0.x, pb1.x) < lo1.x
                    || min2(pb0.y, pb1.y) > hi1.y
                    || max2(pb0.y, pb1.y) < lo1.y
                    || min2(pb0.z, pb1.z) > hi1.z
                    || max2(pb0.z, pb1.z) < lo1.z
                {
                    continue;
                }
                out.push([a0 as u32, a1 as u32, b0 as u32, b1 as u32]);
            }
        }
        out
    }

    /// Solve vertex-face (point-triangle) self-contact, Jacobi accumulation.
    ///
    /// For every candidate pair the constraint is `C = |p − q| − thickness ≥ 0`, where
    /// `q` is the closest point of the triangle to `p` and `thickness` is
    /// `config.self_collision_distance`. With the barycentric weights `w` of `q`
    /// (`q = w₀a + w₁b + w₂c`, `Σw = 1`, `w ≥ 0`) the gradients are `∇_p C = n`,
    /// `∇_a C = −w₀n`, … with `n = (p − q)/|p − q|`, so the projection is
    ///
    /// ```text
    /// s   = (thickness − |p − q|) / (w_p + w_a w₀² + w_b w₁² + w_c w₂²)
    /// Δp  = +s·w_p·n,   Δa = −s·w_a·w₀·n,   Δb = −s·w_b·w₁·n,   Δc = −s·w_c·w₂·n
    /// ```
    ///
    /// # Determinism
    ///
    /// Corrections are accumulated into a per-vertex `Δ` buffer with `+` and applied
    /// after the whole sweep, then averaged by the number of contacts that touched the
    /// vertex. `Fix128` addition is the exact group operation of `Z/2¹²⁸`, so the sum is
    /// bit-identical for **any** enumeration order of the pairs
    /// (`tests/reduction_order_independence.rs`); the pair order is not load-bearing and
    /// does not need to be sorted. Each contact normal is computed **once** and reused
    /// for all four gradients — `Fix128::Mul` is not associative, so a normal built by
    /// accumulation would depend on the order it was summed in.
    ///
    /// Before 1.4.x this was a Gauss-Seidel particle-particle pass (`positions[i]`
    /// written in place and read by later pairs). Point-point is subsumed: the distance
    /// from `p` to a triangle containing `q` is never larger than `|p − q|`, and in this
    /// mesh every pair of vertices that share a triangle is also joined by an edge, so
    /// any non-adjacent pair closer than `thickness` is reported by at least one
    /// vertex-face pair at the same threshold. `test_point_triangle_subsumes_point_point`
    /// pins that inequality.
    ///
    /// Edge-edge contact is accumulated by `accumulate_edge_edge_contacts` into the **same**
    /// `Δ`/`hits` buffer: a vertex can be within `thickness` of no face at all while two
    /// edges interpenetrate (the X crossing), so neither stage subsumes the other, and
    /// ⚠️ **they must not average separately** (see `solve_self_collision`).
    fn accumulate_vertex_face_contacts(
        &self,
        candidates: &[(u32, u32)],
        thickness: Fix128,
        delta: &mut [Vec3Fix],
        hits: &mut [u32],
    ) {
        let thickness_sq = thickness * thickness;
        for &(vi, ti) in candidates {
            let vi = vi as usize;
            let tri = self.triangles[ti as usize];
            let p = self.positions[vi];
            let (a, b, c) = (
                self.positions[tri[0]],
                self.positions[tri[1]],
                self.positions[tri[2]],
            );
            let Some((q, w)) = closest_point_on_triangle(p, a, b, c) else {
                continue; // degenerate (zero-area) triangle
            };
            let diff = p - q;
            let dist_sq = diff.length_squared();
            if dist_sq >= thickness_sq {
                continue;
            }
            // One independent computation of the normal, reused for all four gradients.
            let (normal, dist) = if dist_sq.is_zero() {
                // `p` sits exactly on the triangle: no separation direction survives, so
                // fall back to the face normal. `closest_point_on_triangle` already
                // rejected a zero-area triangle, but the normalisation can still
                // underflow, and a pair with no normal is a pair with no correction.
                match (b - a).cross(c - a).try_normalize() {
                    Some(nrm) => (nrm, Fix128::ZERO),
                    None => continue,
                }
            } else {
                let d = dist_sq.sqrt();
                (diff * (Fix128::ONE / d), d)
            };

            let (wp, wa, wb, wc) = (
                self.inv_masses[vi],
                self.inv_masses[tri[0]],
                self.inv_masses[tri[1]],
                self.inv_masses[tri[2]],
            );
            let denom = wp + wa * w[0] * w[0] + wb * w[1] * w[1] + wc * w[2] * w[2];
            if denom.is_zero() {
                continue;
            }
            let s = (thickness - dist) / denom;

            delta[vi] = delta[vi] + normal * (s * wp);
            hits[vi] += 1;
            for (k, &idx) in tri.iter().enumerate() {
                let wk = self.inv_masses[idx];
                delta[idx] = delta[idx] - normal * (s * wk * w[k]);
                hits[idx] += 1;
            }
        }
    }

    /// Accumulate edge-edge (segment-segment) self-contact, Jacobi accumulation.
    ///
    /// This is the half of cloth self-contact that vertex-face cannot see. Two sheets can
    /// cross so that each sheet's vertices stay far from every triangle of the other while
    /// an edge of one passes within `thickness` of an edge of the other — the X crossing.
    /// The vertex-face pass reports nothing there, because the closest feature pair is
    /// (edge, edge) and neither edge's endpoints are the witnesses.
    ///
    /// For every candidate pair the constraint is `C = |pₐ − p_b| − thickness ≥ 0` with
    /// `pₐ = p₁ + s(q₁ − p₁)`, `p_b = p₂ + t(q₂ − p₂)` at the closest parameters `(s, t)`.
    /// Holding `(s, t)` fixed (the usual treatment: they are stationary, so their
    /// derivatives do not enter `∇C` to first order) the gradients are
    ///
    /// ```text
    /// ∇_{p₁}C = (1−s)n   ∇_{q₁}C = s·n   ∇_{p₂}C = −(1−t)n   ∇_{q₂}C = −t·n
    /// n       = (pₐ − p_b)/|pₐ − p_b|
    /// λ       = (thickness − |pₐ − p_b|) / (w₁(1−s)² + w_{q₁}s² + w₂(1−t)² + w_{q₂}t²)
    /// Δp₁     = +λ·w₁·(1−s)·n   …   Δq₂ = −λ·w_{q₂}·t·n
    /// ```
    ///
    /// # Determinism
    ///
    /// Same construction as the vertex-face stage, into the **same** buffer: each pair's
    /// normal is computed **once** and reused for all four gradients, corrections are summed
    /// into a per-vertex `Δ` with `+` (the exact group operation of `Z/2¹²⁸`) and applied
    /// after the sweep, averaged by the number of contacts that touched the vertex. The pair
    /// order is therefore not load-bearing
    /// (`edge_edge_result_is_independent_of_the_pair_order`,
    /// `self_collision_is_independent_of_the_pair_order_across_both_stages`). No
    /// Gauss-Seidel in-place write, so no dependence on the traversal order of the edge list.
    ///
    /// # What it does not cover
    ///
    /// ⚠️ **Proximity only — there is no swept edge-edge test.** A crossing that both
    /// begins and ends inside one frame, with the edges more than `thickness` apart at the
    /// end of it, is not detected; the frame-level repair
    /// (`resolve_self_contact_over_frame`) tests vertex chords against triangles, not edge
    /// pairs against each other. ⚠️ **Parallel pairs are skipped** by
    /// `closest_points_on_segments`.
    fn accumulate_edge_edge_contacts(
        &self,
        candidates: &[[u32; 4]],
        thickness: Fix128,
        delta: &mut [Vec3Fix],
        hits: &mut [u32],
    ) {
        let thickness_sq = thickness * thickness;
        for &[i0, i1, j0, j1] in candidates {
            let (i0, i1, j0, j1) = (i0 as usize, i1 as usize, j0 as usize, j1 as usize);
            let (p1, q1) = (self.positions[i0], self.positions[i1]);
            let (p2, q2) = (self.positions[j0], self.positions[j1]);
            let Some((s, t)) = closest_points_on_segments(p1, q1, p2, q2) else {
                continue; // degenerate or parallel: no isolated closest pair
            };
            let (d1, d2) = (q1 - p1, q2 - p2);
            let pa = p1 + d1 * s;
            let pb = p2 + d2 * t;
            let diff = pa - pb;
            let dist_sq = diff.length_squared();
            if dist_sq >= thickness_sq {
                continue;
            }
            // One independent computation of the normal, reused for all four gradients.
            let (normal, dist) = if dist_sq.is_zero() {
                // The segments meet. The direction that separates them without rotating
                // either one is their common perpendicular; it exists because
                // `closest_points_on_segments` already rejected the parallel case, but the
                // normalisation can still underflow, and a pair with no normal is a pair
                // with no correction.
                match d1.cross(d2).try_normalize() {
                    Some(nrm) => (nrm, Fix128::ZERO),
                    None => continue,
                }
            } else {
                let d = dist_sq.sqrt();
                (diff * (Fix128::ONE / d), d)
            };

            let (w0, w1, w2, w3) = (
                self.inv_masses[i0],
                self.inv_masses[i1],
                self.inv_masses[j0],
                self.inv_masses[j1],
            );
            let (u, v) = (Fix128::ONE - s, Fix128::ONE - t);
            let denom = w0 * u * u + w1 * s * s + w2 * v * v + w3 * t * t;
            if denom.is_zero() {
                continue;
            }
            let lambda = (thickness - dist) / denom;

            delta[i0] = delta[i0] + normal * (lambda * w0 * u);
            delta[i1] = delta[i1] + normal * (lambda * w1 * s);
            delta[j0] = delta[j0] - normal * (lambda * w2 * v);
            delta[j1] = delta[j1] - normal * (lambda * w3 * t);
            hits[i0] += 1;
            hits[i1] += 1;
            hits[j0] += 1;
            hits[j1] += 1;
        }
    }

    /// One self-contact iteration: both stages into **one** `Δ`/`hits` buffer, applied once.
    ///
    /// ⚠️ **The shared buffer is load-bearing, not a tidiness choice.** With a buffer per
    /// stage, a vertex touched by 5 vertex-face contacts and 1 edge-edge contact receives
    /// `(Σ_vf)/5 + (Σ_ee)/1`, so the single edge-edge contact is weighted five times what
    /// each vertex-face contact gets. The averaging denominator is supposed to be "the
    /// number of contacts that touched **the vertex**", which is 6, not "the number that
    /// touched it within its own stage".
    ///
    /// Measured on the 9x9 crumple scene (`tests/analytic_self_contact.rs`, radius 0.05,
    /// 60 steps) when edge-edge was first wired as a second, separately averaged pass:
    ///
    /// | | vertex-face only | separate buffers | shared buffer |
    /// |---|---|---|---|
    /// | min vertex-face distance², self-contact ON | 8.64e-4 | **1.79e-5** | see the test |
    /// | same, OFF | 6.08e-5 | 6.08e-5 | 6.08e-5 |
    ///
    /// ⚠️ With separate buffers, **turning self-contact on made the vertex-face separation
    /// worse than turning it off** — and 61 of 61 firing edge-edge contacts were genuine
    /// long-range pairs (0 in-surface neighbours), so it was the weighting, not spurious
    /// contacts. `self_collision_improves_the_vertex_face_separation_it_constrains` is the
    /// test that catches it.
    ///
    /// ⚠️ **Two guards, and they cover different things.** The single-stage entry points are
    /// `#[cfg(test)]`, so a release build has exactly one way in and cannot average per
    /// stage by accident. The *wiring* in `substep` is guarded only by the crumple
    /// integration test: the mutation that replaces the one call with two
    /// (`SPLIT-average-per-stage`) leaves `cargo test --lib cloth::` **green** — including
    /// `the_averaging_denominator_counts_contacts_from_both_stages`, which pins this
    /// function's contract and not its caller — and turns
    /// `self_collision_improves_the_vertex_face_separation_it_constrains` red. Do not delete
    /// that integration test as redundant.
    fn solve_self_collision(&mut self, vertex_face: &[(u32, u32)], edge_edge: &[[u32; 4]]) {
        let thickness = self.config.self_collision_distance;
        if thickness.is_zero() || (vertex_face.is_empty() && edge_edge.is_empty()) {
            return;
        }
        let n = self.particle_count();
        let mut delta = vec![Vec3Fix::ZERO; n];
        let mut hits = vec![0u32; n];

        self.accumulate_vertex_face_contacts(vertex_face, thickness, &mut delta, &mut hits);
        self.accumulate_edge_edge_contacts(edge_edge, thickness, &mut delta, &mut hits);

        for i in 0..n {
            if hits[i] == 0 || self.inv_masses[i].is_zero() {
                continue;
            }
            self.positions[i] = self.positions[i] + delta[i] / Fix128::from_int(i64::from(hits[i]));
        }
    }

    /// Vertex-face stage on its own — the closed-form tests drive one stage at a time
    #[cfg(test)]
    fn solve_vertex_face_self_collision(&mut self, candidates: &[(u32, u32)]) {
        self.solve_self_collision(candidates, &[]);
    }

    /// Edge-edge stage on its own — the closed-form tests drive one stage at a time
    #[cfg(test)]
    fn solve_edge_edge_self_collision(&mut self, candidates: &[[u32; 4]]) {
        self.solve_self_collision(&[], candidates);
    }

    /// Number of repair passes `resolve_self_contact_over_frame` is allowed.
    ///
    /// Each pass shortens the offending chords (a pierced vertex is pulled back toward
    /// where it started), so the passes are contractive and in the crumple scene the
    /// loop reaches its fixed point well inside this budget. The cap exists so a
    /// pathological configuration degrades into "some tunnelling survives this frame"
    /// instead of hanging; `remaining_self_contact_crossings` makes that observable.
    const SELF_CONTACT_PASSES: usize = 16;

    /// Push back every vertex whose **frame chord** pierced a non-incident triangle.
    ///
    /// This is the discrete-collision half of the pair (Bridson et al. 2002, *Robust
    /// Treatment of Collisions, Contact and Friction for Cloth Animation*): the
    /// proximity repulsion inside the substeps keeps surfaces apart while they are
    /// close, and this pass catches what a large time step let through anyway. It runs
    /// on the **frame**, not the substep, because the frame is the interval the caller
    /// integrates over — a vertex that ends the frame on the far side of a triangle has
    /// tunnelled regardless of which substep it happened in.
    ///
    /// For every pierce it takes the entry point `q` on the triangle, orients the face
    /// normal `n` toward the side the chord started on, and projects the constraint
    /// `C = n·(p − q) − thickness ≥ 0` exactly as the proximity pass does (same
    /// gradients, same Jacobi accumulation, same averaging). Both endpoints of the chord
    /// then lie strictly on the entry side of the triangle's plane, and a segment whose
    /// endpoints share a side cannot cross it.
    ///
    /// ⚠️ This repair is a **position** projection: it can stretch an edge past what
    /// `solve_edge_constraints` would allow, and it does not build rigid impact zones,
    /// so a pile-up of simultaneous contacts is resolved approximately rather than
    /// exactly. The invariant it does guarantee is the one
    /// `remaining_self_contact_crossings` measures.
    ///
    /// The correction is mirrored into the velocities (`Δ/dt`), otherwise the next frame
    /// re-integrates the velocity that caused the tunnelling and the cloth jitters
    /// against the repair.
    fn resolve_self_contact_over_frame(&mut self, start: &[Vec3Fix], dt: Fix128) {
        let thickness = self.config.self_collision_distance;
        let n = self.particle_count();
        let inv_dt = if dt.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / dt
        };
        let mut delta = vec![Vec3Fix::ZERO; n];
        let mut hits = vec![0u32; n];
        let mut total = vec![Vec3Fix::ZERO; n];

        for _ in 0..Self::SELF_CONTACT_PASSES {
            delta.fill(Vec3Fix::ZERO);
            hits.fill(0);
            // Corrections actually accumulated this pass — not pierces detected. A pierce
            // the projection cannot act on is not a reason to run another pass.
            let mut found = 0usize;

            for tri in &self.triangles {
                let (a, b, c) = (
                    self.positions[tri[0]],
                    self.positions[tri[1]],
                    self.positions[tri[2]],
                );
                let lo = Vec3Fix::new(
                    min3(a.x, b.x, c.x) - thickness,
                    min3(a.y, b.y, c.y) - thickness,
                    min3(a.z, b.z, c.z) - thickness,
                );
                let hi = Vec3Fix::new(
                    max3(a.x, b.x, c.x) + thickness,
                    max3(a.y, b.y, c.y) + thickness,
                    max3(a.z, b.z, c.z) + thickness,
                );
                let Some(face) = (b - a).cross(c - a).try_normalize() else {
                    continue; // zero-area triangle has no side to be on
                };

                for i in 0..n {
                    if i == tri[0] || i == tri[1] || i == tri[2] {
                        continue;
                    }
                    let p0 = start[i];
                    let p1 = self.positions[i];
                    // Chord AABB vs triangle AABB (cheap reject before the predicate)
                    if min2(p0.x, p1.x) > hi.x
                        || max2(p0.x, p1.x) < lo.x
                        || min2(p0.y, p1.y) > hi.y
                        || max2(p0.y, p1.y) < lo.y
                        || min2(p0.z, p1.z) > hi.z
                        || max2(p0.z, p1.z) < lo.z
                    {
                        continue;
                    }
                    let Some((t_scaled, det)) = segment_pierces_triangle(p0, p1, a, b, c) else {
                        continue;
                    };

                    // Entry point, and the barycentric weights the gradients need
                    let t = t_scaled / det;
                    let entry = p0 + (p1 - p0) * t;
                    let Some((q, w)) = closest_point_on_triangle(entry, a, b, c) else {
                        continue;
                    };
                    // One independent computation, oriented toward the side the chord
                    // started on (accumulating a normal would make it order-dependent).
                    let normal = if face.dot(p0 - q) < Fix128::ZERO {
                        -face
                    } else {
                        face
                    };

                    let (wp, wa, wb, wc) = (
                        self.inv_masses[i],
                        self.inv_masses[tri[0]],
                        self.inv_masses[tri[1]],
                        self.inv_masses[tri[2]],
                    );
                    let denom = wp + wa * w[0] * w[0] + wb * w[1] * w[1] + wc * w[2] * w[2];
                    if denom.is_zero() {
                        continue;
                    }
                    let c_val = normal.dot(p1 - q) - thickness;
                    if c_val >= Fix128::ZERO {
                        // The chord only grazed the plane at its start (`p0` exactly on
                        // it), so the end is already clear and there is nothing to undo.
                        // This does **not** count as progress: counting it would keep the
                        // loop spinning for all `SELF_CONTACT_PASSES` with no correction.
                        continue;
                    }
                    let s = c_val / denom;
                    found += 1;

                    delta[i] = delta[i] - normal * (s * wp);
                    hits[i] += 1;
                    for (k, &idx) in tri.iter().enumerate() {
                        delta[idx] = delta[idx] + normal * (s * self.inv_masses[idx] * w[k]);
                        hits[idx] += 1;
                    }
                }
            }

            if found == 0 {
                break;
            }
            for i in 0..n {
                if hits[i] == 0 || self.inv_masses[i].is_zero() {
                    continue;
                }
                let applied = delta[i] / Fix128::from_int(i64::from(hits[i]));
                self.positions[i] = self.positions[i] + applied;
                total[i] = total[i] + applied;
            }
        }

        for (i, moved) in total.iter().enumerate() {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.velocities[i] = self.velocities[i] + *moved * inv_dt;
        }
    }

    /// Number of frame chords `start[i] → positions[i]` that pierce a non-incident
    /// triangle at its current position.
    ///
    /// This is the invariant `step` maintains, exposed so that a caller (or a test) can
    /// check it instead of trusting that it holds. `0` is the only value that means
    /// "the cloth did not pass through itself over this frame"; any positive value is
    /// the number of surviving tunnelling events, not a quality score.
    #[must_use]
    pub fn remaining_self_contact_crossings(&self, start: &[Vec3Fix]) -> usize {
        let n = self.particle_count().min(start.len());
        let mut count = 0usize;
        for tri in &self.triangles {
            let (a, b, c) = (
                self.positions[tri[0]],
                self.positions[tri[1]],
                self.positions[tri[2]],
            );
            for (i, from) in start.iter().enumerate().take(n) {
                if i == tri[0] || i == tri[1] || i == tri[2] {
                    continue;
                }
                if segment_pierces_triangle(*from, self.positions[i], a, b, c).is_some() {
                    count += 1;
                }
            }
        }
        count
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

/// Smaller of two `Fix128`
fn min2(a: Fix128, b: Fix128) -> Fix128 {
    if a < b {
        a
    } else {
        b
    }
}

/// Larger of two `Fix128`
fn max2(a: Fix128, b: Fix128) -> Fix128 {
    if a > b {
        a
    } else {
        b
    }
}

/// Does the segment `p0 → p1` pass through the interior of triangle `(a, b, c)`?
///
/// Möller–Trumbore without the division: the determinant is normalised to a positive
/// sign and every acceptance test is then a comparison, so no rounding enters the
/// decision. Returns `(t_scaled, det)` with the crossing parameter `t = t_scaled / det`
/// for callers that need the entry point; `det` is strictly positive on return.
///
/// `None` covers three distinct rejections that a caller never needs to tell apart —
/// parallel to the plane, outside the barycentric triangle, outside the segment's
/// parameter range — because in all three the segment does not cross this triangle and
/// there is nothing to repair.
#[allow(clippy::many_single_char_names)]
fn segment_pierces_triangle(
    p0: Vec3Fix,
    p1: Vec3Fix,
    a: Vec3Fix,
    b: Vec3Fix,
    c: Vec3Fix,
) -> Option<(Fix128, Fix128)> {
    let dir = p1 - p0;
    let e1 = b - a;
    let e2 = c - a;
    let h = dir.cross(e2);
    let mut det = e1.dot(h);
    if det.is_zero() {
        return None; // segment parallel to the triangle's plane
    }
    let s = p0 - a;
    let q = s.cross(e1);
    let mut u = s.dot(h);
    let mut v = dir.dot(q);
    let mut t = e2.dot(q);

    if det < Fix128::ZERO {
        det = Fix128::ZERO - det;
        u = Fix128::ZERO - u;
        v = Fix128::ZERO - v;
        t = Fix128::ZERO - t;
    }
    if u < Fix128::ZERO || v < Fix128::ZERO || u + v > det {
        return None; // outside the triangle
    }
    if t < Fix128::ZERO || t > det {
        return None; // outside the segment
    }
    Some((t, det))
}

/// Smallest of three `Fix128`
fn min3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let m = if a < b { a } else { b };
    if m < c {
        m
    } else {
        c
    }
}

/// Largest of three `Fix128`
fn max3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let m = if a > b { a } else { b };
    if m > c {
        m
    } else {
        c
    }
}

/// Closest point of triangle `(a, b, c)` to `p`, with its barycentric weights.
///
/// Voronoi-region form (Ericson, *Real-Time Collision Detection* §5.1.5): the seven
/// regions (3 vertex, 3 edge, 1 face) are separated by sign tests on dot products, so
/// only the region that actually wins performs a division.
///
/// Returns `(q, [w₀, w₁, w₂])` with `q = w₀a + w₁b + w₂c`, `Σw = 1` and `w ≥ 0`. The
/// weights are what the contact gradients are built from, so they are returned rather
/// than recomputed: `q` alone cannot distinguish "on edge `ab`" from "inside the face
/// next to `ab`", and those two have different gradients.
///
/// `None` means one thing only: **this triangle cannot be projected onto in `Fix128`** —
/// it has zero area, or an edge so short that its squared length underflows Q64.64 (the
/// region denominators are `|ab|²`, `|ac|²`, `|bc|²` and `2·area`). The caller's action is
/// the same in every case: skip the pair. It is deliberately not "return vertex `a`",
/// which used to be four separate silent fallbacks returning a point that is not the
/// closest one.
///
/// ⚠️ The clamping into vertex and edge regions is load-bearing: without it an
/// unclamped face-region solve returns a point outside the triangle, and the contact is
/// reported against a plane instead of against the triangle
/// (`closest_point_clamps_into_the_edge_and_vertex_regions` pins each region).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn closest_point_on_triangle(
    p: Vec3Fix,
    a: Vec3Fix,
    b: Vec3Fix,
    c: Vec3Fix,
) -> Option<(Vec3Fix, [Fix128; 3])> {
    const ZERO: Fix128 = Fix128::ZERO;
    const ONE: Fix128 = Fix128::ONE;
    let ab = b - a;
    let ac = c - a;

    // A zero-area triangle has no face to project onto. Rejecting it here is what makes
    // every `den.is_zero()` below a genuine underflow guard rather than a silent answer.
    let area2 = ab.cross(ac);
    if area2.x.is_zero() && area2.y.is_zero() && area2.z.is_zero() {
        return None;
    }

    // Vertex region A
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= ZERO && d2 <= ZERO {
        return Some((a, [ONE, ZERO, ZERO]));
    }

    // Vertex region B
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= ZERO && d4 <= d3 {
        return Some((b, [ZERO, ONE, ZERO]));
    }

    // Edge region AB
    let vc = d1 * d4 - d3 * d2;
    if vc <= ZERO && d1 >= ZERO && d3 <= ZERO {
        let den = d1 - d3;
        if den.is_zero() {
            return None; // |ab|² underflowed
        }
        let v = d1 / den;
        return Some((a + ab * v, [ONE - v, v, ZERO]));
    }

    // Vertex region C
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= ZERO && d5 <= d6 {
        return Some((c, [ZERO, ZERO, ONE]));
    }

    // Edge region AC
    let vb = d5 * d2 - d1 * d6;
    if vb <= ZERO && d2 >= ZERO && d6 <= ZERO {
        let den = d2 - d6;
        if den.is_zero() {
            return None; // |ac|² underflowed
        }
        let w = d2 / den;
        return Some((a + ac * w, [ONE - w, ZERO, w]));
    }

    // Edge region BC
    let va = d3 * d6 - d5 * d4;
    let e_b = d4 - d3;
    let e_c = d5 - d6;
    if va <= ZERO && e_b >= ZERO && e_c >= ZERO {
        let den = e_b + e_c;
        if den.is_zero() {
            return None; // |bc|² underflowed
        }
        let w = e_b / den;
        return Some((b + (c - b) * w, [ZERO, ONE - w, w]));
    }

    // Face region
    let den = va + vb + vc;
    if den.is_zero() {
        return None; // 2·area underflowed
    }
    let inv = ONE / den;
    let v = vb * inv;
    let w = vc * inv;
    Some((a + ab * v + ac * w, [ONE - v - w, v, w]))
}

/// Closest parameters `(s, t)` of segments `p₁→q₁` and `p₂→q₂`, both in `[0, 1]`.
///
/// Ericson, *Real-Time Collision Detection* §5.1.9: solve the unconstrained 2x2 system,
/// clamp `s` into the segment, recompute `t`, and if `t` left its range pin it and resolve
/// `s`. The clamping is what turns a line-line answer into a segment-segment one.
///
/// ⚠️ The clamping is load-bearing. Without it the returned point can lie off the end of
/// the segment, and the contact is then reported between two infinite lines — in a mesh
/// that means a correction applied to an edge that is nowhere near the other one
/// (`closest_points_on_segments_clamps_into_the_endpoint_regions` pins each region).
///
/// `None` means one thing only: **this pair has no isolated closest point pair in
/// `Fix128`**. Three inputs produce it and the caller's action is the same for all three
/// (skip the pair):
///
/// - either segment has zero length, or a length whose square underflows Q64.64
///   (`|d|² < 2⁻⁶⁴`, i.e. `|d| < 2⁻³²`) — there is no direction to parametrise along;
/// - the two directions are parallel, so `a·e − b² = 0` and the minimum is attained on a
///   whole interval rather than at a point. Returning an arbitrary member of that interval
///   would make the contact gradients a function of which member was picked, not of the
///   geometry. ⚠️ Parallel pairs are therefore **not** covered by the edge-edge pass; they
///   are covered by vertex-face, where two parallel sheets put each sheet's vertices over
///   the other's triangles. A flat grid's row-mates are the common instance, and this
///   rejection is why the pass is silent on an undeformed sheet;
/// - the same determinant underflowing to zero for a merely near-parallel pair, which is
///   the fixed-point form of the previous case and gets the same answer.
fn closest_points_on_segments(
    p1: Vec3Fix,
    q1: Vec3Fix,
    p2: Vec3Fix,
    q2: Vec3Fix,
) -> Option<(Fix128, Fix128)> {
    const ZERO: Fix128 = Fix128::ZERO;
    const ONE: Fix128 = Fix128::ONE;

    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    if a.is_zero() || e.is_zero() {
        return None; // zero-length (or underflowed) segment
    }
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    if denom.is_zero() {
        return None; // parallel: the minimum is an interval, not a point
    }

    let r = p1 - p2;
    let c = d1.dot(r);
    let f = d2.dot(r);

    let clamp01 = |x: Fix128| {
        if x < ZERO {
            ZERO
        } else if x > ONE {
            ONE
        } else {
            x
        }
    };

    let mut s = clamp01((b * f - c * e) / denom);
    let mut t = (b * s + f) / e;
    if t < ZERO {
        t = ZERO;
        s = clamp01(-c / a);
    } else if t > ONE {
        t = ONE;
        s = clamp01((b - c) / a);
    }
    Some((s, t))
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

    /// 折り返して 2 層が重なった布で、自己接触が層を引き離すことを確かめる
    ///
    /// ⚠️ **旧 scene (5x5 を上端 pin で吊るす) は自己交差しなかったので、
    /// `self_collision = false` にしても green だった** (2026-09-29 破壊試験で実測:
    /// 最小粒子間距離 0.49994 に対し判定閾値 0.025 = 余裕 20 倍、`too_close` は 0 で
    /// assert は `0 < 25`) 余裕を締め直しても「自己交差しない場面で自己接触を測っている」
    /// ことは変わらないので、**scene を置換**した
    ///
    /// 折り返しは**等長**なので辺拘束と喧嘩しない (行 3 → `(y=1, z=2)`、行 4 → `(y=1, z=1)`
    /// で全ての辺と対角が元の長さを保つ) 行 1 と行 4 は辺で繋がっていない重なり対になる
    ///
    /// ⚠️ この test が言えるのは **粒子-粒子の斥力が働いていること**だけです
    /// 頂点が三角形の内部を通り抜ける経路は粒子-粒子では原理的に見られないので、
    /// そちらは `tests/analytic_self_contact.rs` の目標 oracle が担当する
    #[test]
    fn test_cloth_self_collision() {
        fn folded(self_collision: bool) -> Fix128 {
            const R: usize = 5;
            let mut cloth = Cloth::new_grid(
                Vec3Fix::ZERO,
                Fix128::from_int(4),
                Fix128::from_int(4),
                R,
                R,
                Fix128::from_ratio(1, 100),
            );
            cloth.config.self_collision = self_collision;
            // 層間 (1.0) より大きく取って、重なりが判定に入るようにする
            cloth.config.self_collision_distance = Fix128::from_ratio(3, 2);
            cloth.config.gravity = Vec3Fix::ZERO;
            for x in 0..R {
                cloth.positions[3 * R + x] =
                    Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::from_int(2));
                cloth.positions[4 * R + x] =
                    Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::ONE);
            }
            cloth.prev_positions = cloth.positions.clone();
            for x in 0..R {
                cloth.inv_masses[x] = Fix128::ZERO;
            }
            cloth.step(Fix128::from_ratio(1, 60));
            // 行 1 と行 4 の層間最小距離
            let mut gap = Fix128::from_int(1 << 20);
            for x in 0..R {
                let d = (cloth.positions[4 * R + x] - cloth.positions[R + x]).length();
                if d < gap {
                    gap = d;
                }
            }
            gap
        }

        let gap_on = folded(true);
        let gap_off = folded(false);
        assert!(
            gap_on > gap_off,
            "自己接触を有効にしても層が引き離されていない (ON {} <= OFF {})",
            gap_on.to_f32(),
            gap_off.to_f32()
        );
    }

    // ---- vertex-face self-contact -----------------------------------

    /// `closest_point_on_triangle` の 7 領域を閉形式で固定する
    ///
    /// 三角形は `a = (0,0,0)`, `b = (1,0,0)`, `c = (0,0,1)` (XZ 平面) 座標は全て 2 進小数
    /// なので `Fix128` で厳密、許容差なしの等値で比較できる
    ///
    /// ⚠️ **重心座標も一緒に pin する** 最近接点だけを見ると「辺 `ab` の上」と「面領域で
    /// 辺 `ab` のすぐ内側」が区別できないが、この 2 つは**勾配が違う** (前者は `c` に
    /// 補正が行かない) 出所: 手計算
    #[test]
    fn closest_point_clamps_into_the_edge_and_vertex_regions() {
        let a = Vec3Fix::from_int(0, 0, 0);
        let b = Vec3Fix::from_int(1, 0, 0);
        let c = Vec3Fix::from_int(0, 0, 1);
        let h = Fix128::from_ratio(1, 2);
        let q = Fix128::from_ratio(1, 4);
        let two = Fix128::from_int(2);
        let neg_h = Fix128::ZERO - h;
        let neg_one = Fix128::from_int(-1);
        let o = Fix128::ZERO;
        let i = Fix128::ONE;

        let cases: [(Vec3Fix, Vec3Fix, [Fix128; 3], &str); 7] = [
            // 面領域: 垂線の足 (1/4, 0, 1/4)、w = (1/2, 1/4, 1/4)
            (
                Vec3Fix::new(q, h, q),
                Vec3Fix::new(q, o, q),
                [h, q, q],
                "face",
            ),
            // 辺 ab: p は -z 側に外れる
            (
                Vec3Fix::new(h, h, neg_one),
                Vec3Fix::new(h, o, o),
                [h, h, o],
                "edge ab",
            ),
            // 辺 bc (斜辺): 平面への投影は u + v = 2 > 1
            (
                Vec3Fix::new(i, h, i),
                Vec3Fix::new(h, o, h),
                [o, h, h],
                "edge bc",
            ),
            // 辺 ca: p は -x 側に外れる
            (
                Vec3Fix::new(neg_one, h, h),
                Vec3Fix::new(o, o, h),
                [h, o, h],
                "edge ca",
            ),
            // 頂点 a: ab·ap = ac·ap = -1/2 < 0
            (Vec3Fix::new(neg_h, h, neg_h), a, [i, o, o], "vertex a"),
            // 頂点 b
            (Vec3Fix::new(two, h, neg_h), b, [o, i, o], "vertex b"),
            // 頂点 c
            (Vec3Fix::new(neg_h, h, two), c, [o, o, i], "vertex c"),
        ];

        for (p, expect_q, expect_w, name) in cases {
            let Some((got_q, got_w)) = closest_point_on_triangle(p, a, b, c) else {
                panic!("{name}: 退化していない三角形で None が返った");
            };
            assert_eq!(got_q, expect_q, "{name}: 最近接点");
            assert_eq!(got_w, expect_w, "{name}: 重心座標");
            // 重心座標と最近接点が同じ点を指していること (勾配の整合性の前提)
            assert_eq!(
                a * got_w[0] + b * got_w[1] + c * got_w[2],
                expect_q,
                "{name}: w が最近接点を再構成しない"
            );
        }

        // 退化 (面積 0) は None `Some(a)` 等の「最近接でない点」で誤魔化さない
        for (x, y, z, name) in [
            (a, b, b, "b と c が同一"),
            (a, a, a, "3 頂点が同一"),
            (a, b, Vec3Fix::from_int(2, 0, 0), "3 頂点が同一直線上"),
        ] {
            assert!(
                closest_point_on_triangle(Vec3Fix::new(q, h, q), x, y, z).is_none(),
                "{name}: 面積 0 の三角形で Some が返った"
            );
        }
    }

    /// 頂点-面の判定が頂点-頂点の判定を含む (同じ閾値で取りこぼさない)
    ///
    /// 2 つの根拠を一緒に pin する:
    ///
    /// 1. 点-三角形距離 ≤ 点-頂点距離 (三角形の頂点も三角形の一部なので自明だが、
    ///    `closest_point_on_triangle` の領域分岐が壊れるとここが破れる)
    /// 2. `new_grid` の mesh では、**同じ三角形に属する頂点対は必ず辺拘束にもなっている**
    ///    ので、旧 particle-particle 判定が除外していた対 (= 辺で繋がった対) と、
    ///    新判定が除外する対 (= 三角形に属する頂点) が一致する
    #[test]
    fn point_triangle_subsumes_point_point() {
        let cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            3,
            3,
            Fix128::from_ratio(1, 100),
        );

        // (2) 三角形内の頂点対は全て辺拘束にある
        let mut edges: Vec<(usize, usize)> = cloth
            .edge_constraints
            .iter()
            .map(|c| (c.i0.min(c.i1), c.i0.max(c.i1)))
            .collect();
        edges.sort_unstable();
        for tri in &cloth.triangles {
            for (x, y) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                assert!(
                    edges.binary_search(&(x.min(y), x.max(y))).is_ok(),
                    "三角形 {tri:?} の頂点対 ({x}, {y}) が辺拘束に無い"
                );
            }
        }

        // (1) どの頂点から測っても、三角形までの距離の方が近い
        let a = Vec3Fix::from_int(0, 0, 0);
        let b = Vec3Fix::from_int(1, 0, 0);
        let c = Vec3Fix::from_int(0, 0, 1);
        let probes = [
            Vec3Fix::new(
                Fix128::from_ratio(1, 4),
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(1, 4),
            ),
            Vec3Fix::new(
                Fix128::from_int(3),
                Fix128::from_int(1),
                Fix128::from_int(-2),
            ),
            Vec3Fix::new(
                Fix128::from_ratio(-3, 4),
                Fix128::ZERO,
                Fix128::from_ratio(7, 8),
            ),
            Vec3Fix::new(
                Fix128::from_int(2),
                Fix128::from_int(2),
                Fix128::from_int(2),
            ),
        ];
        for p in probes {
            let (q, _) = closest_point_on_triangle(p, a, b, c).expect("退化していない");
            let face_d2 = (p - q).length_squared();
            for v in [a, b, c] {
                assert!(
                    face_d2 <= (p - v).length_squared(),
                    "点 {p:?}: 三角形までの距離² {face_d2:?} が頂点までの距離²より大きい"
                );
            }
        }
    }

    /// 接触対の列挙順が結果を変えない (Jacobi 蓄積の決定性)
    ///
    /// ⚠️ **これは「単一スレッドで順序が固定だから決定的」とは別の主張です** 順序が
    /// **どうであっても**同じ、を要求している broad-phase の実装を差し替えても、
    /// SIMD で分割しても、結果が動かないことがここで担保される
    /// (`Fix128` の加算は `Z/2¹²⁸` の厳密な群演算、`tests/reduction_order_independence.rs`)
    #[test]
    fn self_contact_result_is_independent_of_the_pair_order() {
        fn folded_cloth() -> Cloth {
            const R: usize = 5;
            let mut cloth = Cloth::new_grid(
                Vec3Fix::ZERO,
                Fix128::from_int(4),
                Fix128::from_int(4),
                R,
                R,
                Fix128::from_ratio(1, 100),
            );
            cloth.config.self_collision = true;
            cloth.config.self_collision_distance = Fix128::from_ratio(3, 2);
            for x in 0..R {
                cloth.positions[3 * R + x] =
                    Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::from_int(2));
                cloth.positions[4 * R + x] =
                    Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::ONE);
            }
            cloth
        }

        let mut forward = folded_cloth();
        let mut reverse = folded_cloth();
        let margin = forward.config.self_collision_distance.double();
        let mut candidates = forward.collect_vertex_face_candidates(margin);
        assert!(
            !candidates.is_empty(),
            "折り返した布で接触候補が 0 件 この scene は順序非依存性を試せない"
        );
        forward.solve_vertex_face_self_collision(&candidates);
        candidates.reverse();
        reverse.solve_vertex_face_self_collision(&candidates);

        assert_eq!(
            forward.positions, reverse.positions,
            "接触対を逆順に処理すると結果が変わる = 蓄積が順序依存"
        );
    }

    /// 1 frame で布を貫くはずの頂点が、貫かずに入射側へ戻される
    ///
    /// # scene (閉形式)
    ///
    /// 5x5 の格子を XZ 平面 (`y = 0`) に置き、**辺拘束と曲げ拘束を外して**衝突だけを見る
    /// 粒子 0 以外は全て pin、粒子 0 は far corner の三角形の**内部**の真上
    /// `(3.5, 2, 3.25)` に置いて `v = (0, -240, 0)` で落とす (`dt = 1/60` なので 1 frame の
    /// 変位は `-4`) 落下線 `x + z = 6.75 < 7` は対角線 `x + z = 7` の内側なので、
    /// 三角形 `[(3,3), (4,3), (3,4)]` の内部を通る 粒子 0 はこの三角形の頂点ではない
    ///
    /// 期待: 自己接触 OFF なら `y < 0` (素通り)、ON なら `y > 0` (入射側に残る) で
    /// `remaining_self_contact_crossings` が 0
    #[test]
    fn a_vertex_driven_through_the_sheet_is_pushed_back_to_the_entry_side() {
        fn scene(self_collision: bool) -> (Cloth, Vec<Vec3Fix>) {
            let mut cloth = Cloth::new_grid(
                Vec3Fix::ZERO,
                Fix128::from_int(4),
                Fix128::from_int(4),
                5,
                5,
                Fix128::from_ratio(1, 100),
            );
            // 衝突だけを測るため、内力を外す (布としてでなく三角形集合として見る)
            cloth.edge_constraints.clear();
            cloth.bend_constraints.clear();
            cloth.config.gravity = Vec3Fix::ZERO;
            cloth.config.damping = Fix128::ONE;
            cloth.config.self_collision = self_collision;
            cloth.config.self_collision_distance = Fix128::from_ratio(1, 10);
            for i in 1..cloth.particle_count() {
                cloth.inv_masses[i] = Fix128::ZERO;
            }
            cloth.positions[0] = Vec3Fix::new(
                Fix128::from_ratio(7, 2),
                Fix128::from_int(2),
                Fix128::from_ratio(13, 4),
            );
            cloth.prev_positions[0] = cloth.positions[0];
            cloth.velocities[0] = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-240), Fix128::ZERO);
            let start = cloth.positions.clone();
            cloth.step(Fix128::from_ratio(1, 60));
            (cloth, start)
        }

        let (off, start_off) = scene(false);
        assert!(
            off.positions[0].y < Fix128::ZERO,
            "自己接触 OFF で粒子が布を素通りしなかった (y = {}) この scene は貫通を試せていない",
            off.positions[0].y.to_f32()
        );
        assert!(
            off.remaining_self_contact_crossings(&start_off) > 0,
            "自己接触 OFF で貫通が検出されない"
        );

        let (on, start_on) = scene(true);
        assert_eq!(
            on.remaining_self_contact_crossings(&start_on),
            0,
            "自己接触 ON で貫通が残った (y = {})",
            on.positions[0].y.to_f32()
        );
        assert!(
            on.positions[0].y > Fix128::ZERO,
            "自己接触 ON で粒子が布の裏側に居る (y = {}) 押し戻しの向きが入射側でない",
            on.positions[0].y.to_f32()
        );
    }

    // -----------------------------------------------------------------------
    // 近接斥力段の強さ — 自分自身の三角形を除外していることを測る
    //
    // ⚠️ 2026-10-01 の破壊試験で、変異 `M3` (`collect_vertex_face_candidates` から
    // `if i == tri[0] || ... { continue }` を外す) が **lib / 統合 oracle とも green**
    // で生き残りました 自分自身の対では最近接点が頂点自身になり、`delta` への寄与が
    // 同じ index で厳密に相殺するので、**位置の差として出るのは `hits` の水増しによる
    // 平均化の弱まりだけ**です 以下 2 本はその弱まりを直接測ります
    // (詳細 [[feedback_degenerate_case_as_silent_wrong_answer]] 同 session の Backlog 行)
    // -----------------------------------------------------------------------

    /// 候補対に「その三角形自身の頂点」が現れない (契約)
    ///
    /// ⚠️ **これは `collect_vertex_face_candidates` の内部条件の言い換えですが、それが
    /// 言い換えでしか固定できない契約です** 下の閉形式 test が「水増しの量」を測り、
    /// 本 test が「水増しが 1 件でも起きていないこと」を測ります 片方だけでは
    /// `M3` が生き残ります (統合 scene では相殺して見えない)
    #[test]
    fn candidate_pairs_never_contain_a_vertex_of_their_own_triangle() {
        let mut cloth = folded_cloth_with_contacts();
        let margin = cloth.config.self_collision_distance.double();
        let candidates = cloth.collect_vertex_face_candidates(margin);
        assert!(
            !candidates.is_empty(),
            "折り返した布で接触候補が 0 件 この scene は契約を試せない"
        );
        for &(vi, ti) in &candidates {
            let tri = cloth.triangles[ti as usize];
            assert!(
                vi as usize != tri[0] && vi as usize != tri[1] && vi as usize != tri[2],
                "候補対 ({vi}, {ti}) が三角形 {tri:?} 自身の頂点を含む \
                 自分の面に対する斥力は delta では相殺するが hits を水増しして \
                 他の接触の補正を薄める"
            );
        }
        // 自己対が混ざっていないことを件数でも押さえる (不在でなく数で判定する)
        let self_pairs = cloth
            .triangles
            .iter()
            .enumerate()
            .map(|(ti, tri)| {
                candidates
                    .iter()
                    .filter(|&&(vi, cti)| {
                        cti as usize == ti
                            && (vi as usize == tri[0]
                                || vi as usize == tri[1]
                                || vi as usize == tri[2])
                    })
                    .count()
            })
            .sum::<usize>();
        assert_eq!(self_pairs, 0, "自己対が {self_pairs} 件混ざっている");
        // 実際に解いても三角形の頂点が自分の面から押されない (念のため後段でも測る)
        let before = cloth.positions.clone();
        cloth.solve_vertex_face_self_collision(&candidates);
        assert!(
            before != cloth.positions,
            "折り返した布で斥力が 1 bit も発火しない この scene は判別に向かない"
        );
    }

    /// 近接斥力 1 回で、頂点-面の隙間が **厳密に** `thickness` になる (閉形式)
    ///
    /// # scene (全ての値が 2 進小数なので許容差なしの等値 assert が書ける)
    ///
    /// 三角形 `a = (0,0,0)`, `b = (1,0,0)`, `c = (0,0,1)` (XZ 平面)、頂点
    /// `p = (1/4, 1/8, 1/4)` 最近接点は面領域の `q = (1/4, 0, 1/4)` で重心座標は
    /// `(w₀, w₁, w₂) = (1/2, 1/4, 1/4)` (`tests/analytic_self_contact.rs` 節 A と同じ配置)
    ///
    /// 逆質量を `w_p = 1/2`, `w_a = 1`, `w_b = w_c = 2` に選ぶと分母が **厳密に 1** になる:
    ///
    /// ```text
    /// denom = w_p + w_a w₀² + w_b w₁² + w_c w₂² = 1/2 + 1/4 + 2/16 + 2/16 = 1
    /// s     = (thickness − |p−q|) / denom = (1/4 − 1/8) / 1 = 1/8
    /// Δp    = +s·w_p·n = +1/16·n      (n = (0,1,0))
    /// Δa    = −s·w_a·w₀·n = −1/16·n   Δb = −s·w_b·w₁·n = −1/16·n   Δc = −1/16·n
    /// ```
    ///
    /// `Δa = Δb = Δc` なので三角形は XZ 平面に平行なまま `−1/16` 平行移動し、隙間は
    /// `1/8 + 1/16 + 1/16 = 1/4 = thickness` に**なるはず**です
    ///
    /// # ⚠️ 変異 `M3` でこの等式が壊れる量
    ///
    /// 自己対 3 件が加わると `hits[a] = hits[b] = hits[c] = 1 + 4 = 5` になり
    /// (`delta` は相殺するので値は変わらない)、三角形の移動量が `1/16` → `1/80` に
    /// 縮んで隙間が `1/8 + 1/16 + 1/80 ≠ 1/4` になります **`delta` でなく `hits` を
    /// 測るのが本 test の役目**です
    #[test]
    fn the_proximity_projection_restores_exactly_the_contact_thickness() {
        let eighth = Fix128::from_ratio(1, 8);
        let quarter = Fix128::from_ratio(1, 4);
        let sixteenth = Fix128::from_ratio(1, 16);

        let mut cloth = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 2, 2, Fix128::ONE);
        cloth.edge_constraints.clear();
        cloth.bend_constraints.clear();
        cloth.triangles = vec![[1, 2, 3]];
        cloth.config.self_collision = true;
        cloth.config.self_collision_distance = quarter;
        cloth.positions[0] = Vec3Fix::new(quarter, eighth, quarter);
        cloth.positions[1] = Vec3Fix::from_int(0, 0, 0);
        cloth.positions[2] = Vec3Fix::from_int(1, 0, 0);
        cloth.positions[3] = Vec3Fix::from_int(0, 0, 1);
        cloth.inv_masses[0] = Fix128::from_ratio(1, 2);
        cloth.inv_masses[1] = Fix128::ONE;
        cloth.inv_masses[2] = Fix128::from_int(2);
        cloth.inv_masses[3] = Fix128::from_int(2);

        let candidates = cloth.collect_vertex_face_candidates(quarter.double());
        // ⚠️ `M3` を入れるとここが 4 件 ((1,0) / (2,0) / (3,0) が増える) になり red
        assert_eq!(
            candidates,
            vec![(0u32, 0u32)],
            "候補は (頂点 0, 三角形 0) の 1 件だけであるべき"
        );

        cloth.solve_vertex_face_self_collision(&candidates);

        let neg_sixteenth = -sixteenth;
        assert_eq!(
            cloth.positions[0],
            Vec3Fix::new(quarter, eighth + sixteenth, quarter),
            "oracle: Δp = +1/16 (手計算、denom = 1 なので厳密)"
        );
        for (i, expected) in [
            (
                1usize,
                Vec3Fix::new(Fix128::ZERO, neg_sixteenth, Fix128::ZERO),
            ),
            (2, Vec3Fix::new(Fix128::ONE, neg_sixteenth, Fix128::ZERO)),
            (3, Vec3Fix::new(Fix128::ZERO, neg_sixteenth, Fix128::ONE)),
        ] {
            assert_eq!(
                cloth.positions[i], expected,
                "oracle: 三角形は −1/16 平行移動する (頂点 {i})"
            );
        }
        // 三角形は XZ 平面に平行なままなので、隙間は y 差がそのまま距離になる
        assert_eq!(
            cloth.positions[0].y - cloth.positions[1].y,
            quarter,
            "oracle: 1 回の投影で隙間が厳密に thickness = 1/4 になる"
        );
    }

    // -----------------------------------------------------------------------
    // 辺-辺の自己接触
    // -----------------------------------------------------------------------

    /// 辺-辺の最近接パラメータが閉形式と厳密一致する (内部点で最小を取る場合)
    ///
    /// 線分 A `(-1,0,0) → (1,0,0)`、線分 B `(0,1/8,-1) → (0,1/8,1)` は直交し、
    /// 最近接は両方の**中点** `s* = t* = 1/2`、最近接点は `(0,0,0)` と `(0,1/8,0)`、
    /// 距離² は `1/64` (手計算: `a = e = 4`, `b = 0`, `c = −2`, `f = 2`, `denom = 16`、
    /// `s = (bf − ce)/denom = 8/16`、`t = (bs + f)/e = 2/4`)
    #[test]
    fn closest_points_on_segments_matches_the_closed_form_in_the_interior() {
        let eighth = Fix128::from_ratio(1, 8);
        let half = Fix128::from_ratio(1, 2);
        let a0 = Vec3Fix::from_int(-1, 0, 0);
        let a1 = Vec3Fix::from_int(1, 0, 0);
        let b0 = Vec3Fix::new(Fix128::ZERO, eighth, Fix128::from_int(-1));
        let b1 = Vec3Fix::new(Fix128::ZERO, eighth, Fix128::ONE);

        let Some((s, t)) = closest_points_on_segments(a0, a1, b0, b1) else {
            panic!("直交する 2 線分で最近接対が得られなかった");
        };
        assert_eq!(s, half, "oracle: s* = 1/2");
        assert_eq!(t, half, "oracle: t* = 1/2");
        let pa = a0 + (a1 - a0) * s;
        let pb = b0 + (b1 - b0) * t;
        assert_eq!(pa, Vec3Fix::ZERO, "oracle: 最近接点 A = (0,0,0)");
        assert_eq!(
            pb,
            Vec3Fix::new(Fix128::ZERO, eighth, Fix128::ZERO),
            "oracle: 最近接点 B = (0,1/8,0)"
        );
        assert_eq!(
            (pa - pb).length_squared(),
            Fix128::from_ratio(1, 64),
            "oracle: 最近接距離² = 1/64"
        );
    }

    /// `s` / `t` が区間外に出る配置で端点へ clamp される (閉形式)
    ///
    /// | 配置 | 閉形式の素の解 | clamp 後 |
    /// |---|---|---|
    /// | A `(0,0,0)→(1,0,0)` / B `(2,1,-1)→(2,1,1)` | `s = 2` | `(s, t) = (1, 1/2)` |
    /// | A `(0,0,0)→(1,0,0)` / B `(1/2,1,2)→(1/2,1,3)` | `t = −2` | `(s, t) = (1/2, 0)` |
    ///
    /// ⚠️ **clamp を外すと線分でなく無限直線の答えになります** 変異 `EE1`
    /// (`clamp01` を恒等関数にする) では 1 行目が `s = 2` を返して red
    #[test]
    fn closest_points_on_segments_clamps_into_the_endpoint_regions() {
        let half = Fix128::from_ratio(1, 2);
        let a0 = Vec3Fix::from_int(0, 0, 0);
        let a1 = Vec3Fix::from_int(1, 0, 0);

        // s が 1 を超える: 線分 B は A の先 (x = 2) にある
        let Some((s, t)) = closest_points_on_segments(
            a0,
            a1,
            Vec3Fix::from_int(2, 1, -1),
            Vec3Fix::from_int(2, 1, 1),
        ) else {
            panic!("端点領域の配置で最近接対が得られなかった");
        };
        assert_eq!(
            s,
            Fix128::ONE,
            "oracle: 素の s = 2 が端点 1 へ clamp される"
        );
        assert_eq!(t, half, "oracle: t = 1/2 は区間内なので動かない");

        // t が 0 を下回る: 線分 B は z = 2..3 にあり A から見て手前の端点が最近接
        let Some((s, t)) = closest_points_on_segments(
            a0,
            a1,
            Vec3Fix::new(half, Fix128::ONE, Fix128::from_int(2)),
            Vec3Fix::new(half, Fix128::ONE, Fix128::from_int(3)),
        ) else {
            panic!("端点領域の配置で最近接対が得られなかった");
        };
        assert_eq!(
            t,
            Fix128::ZERO,
            "oracle: 素の t = −2 が端点 0 へ clamp される"
        );
        assert_eq!(s, half, "oracle: t を固定して解き直した s = 1/2");
    }

    /// 退化入力と平行入力は `None` (「もっともらしい値」を返さない契約)
    ///
    /// ⚠️ 退化は統合 scene では作られないので、**専用の契約 test でしか固定できません**
    /// (同 session の `M7` 実測: 入口の退化判定を落としても crumple の統合 oracle は green
    /// [[feedback_degenerate_case_as_silent_wrong_answer]])
    #[test]
    fn closest_points_on_segments_rejects_degenerate_and_parallel_pairs() {
        let o = Vec3Fix::ZERO;
        let x1 = Vec3Fix::from_int(1, 0, 0);
        let half = Fix128::from_ratio(1, 2);

        // (a) 片方が長さ 0
        assert!(
            closest_points_on_segments(
                o,
                o,
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::from_int(1, 1, 0)
            )
            .is_none(),
            "長さ 0 の線分が受け付けられた"
        );
        // (b) 平行 (直交する離れた 2 直線でなく、同じ向き)
        assert!(
            closest_points_on_segments(
                o,
                x1,
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::from_int(1, 1, 0)
            )
            .is_none(),
            "平行な 2 線分が受け付けられた 最小は区間なので (s,t) は幾何で決まらない"
        );
        // (c) 同一直線上 (平行の特別な場合)
        assert!(
            closest_points_on_segments(
                o,
                x1,
                Vec3Fix::new(half, Fix128::ZERO, Fix128::ZERO),
                Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO)
            )
            .is_none(),
            "同一直線上の 2 線分が受け付けられた"
        );
        // (d) 長さ² が Q64.64 で underflow する (|d| = 2⁻³³ < 2⁻³²)
        let tiny = Fix128::from_raw(0, 1 << 31);
        assert!(
            !tiny.is_zero() && (tiny * tiny).is_zero(),
            "この scene の前提 (|d| ≠ 0 だが |d|² が underflow) が崩れた"
        );
        assert!(
            closest_points_on_segments(
                o,
                Vec3Fix::new(tiny, Fix128::ZERO, Fix128::ZERO),
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::from_int(0, 1, 1)
            )
            .is_none(),
            "長さ² が underflow した線分が受け付けられた"
        );
    }

    /// **2 本目**の線分の退化を `e.is_zero()` が拾う (`denom` の判定では拾えない領域がある)
    ///
    /// ⚠️ `closest_points_on_segments_rejects_degenerate_and_parallel_pairs` の (a) / (d) は
    /// どちらも **1 本目**を退化させるので、入口の判定が `a.is_zero() || e.is_zero()` から
    /// `a.is_zero()` に減っても green のまま通ります (変異
    /// `EE24-only-first-segment-zero-check`)
    ///
    /// # ⚠️ `e.is_zero()` が死んで**いない**領域は狭い
    ///
    /// 2 本目が素直に長さ 0 なら `d₂ = 0` ⇒ `b = d₁·d₂ = 0` ⇒
    /// `denom = a·e − b² = 0` なので、**平行の判定が先に拾います** ⇒ そこを突いても
    /// `e.is_zero()` の有無は観測できません (実測: 本 test の初版はこの配置だったため
    /// 変異が green で生き残りました)
    ///
    /// `e` だけが 0 になるには、`|d₂|²` が Q64.64 で underflow する一方で `b²` が
    /// 生き残る必要があります `|d₂| = 2⁻³³` のとき `|d₂|² = 2⁻⁶⁶ → 0` で、
    /// `b² = (|d₁|·2⁻³³)² = |d₁|²·2⁻⁶⁶ ≥ 2⁻⁶⁴` は **`|d₁| ≥ 2`** で成立します
    /// ⇒ 1 本目を長さ 4 に取ると `denom = 16·0 − 2⁻⁶² = −2⁻⁶² ≠ 0` で平行判定を素通りします
    ///
    /// ⚠️ **素通りした先は panic でなく silent な誤答**です `Fix128::div` は 0 除算で
    /// `ZERO` を返すので (`src/math.rs` の `impl Div`)、`t = (b·s + f)/e` が無言で 0 になり
    /// `Some((s, 0))` という**もっともらしい値**が返ります
    /// ([[feedback_degenerate_case_as_silent_wrong_answer]] と同じ型)
    ///
    /// # 過剰 pin の線引き
    ///
    /// 固定するのは **「孤立した最近接対を持たない入力は `None`」という契約だけ**で、
    /// **どの guard が拒否するかは固定しません** ⇒ 退化の判定を `denom` 側へ寄せる /
    /// `d₂` を先に正規化する等の実装変更は `None` を返す限り green のままです
    /// red にするのは「もっともらしい値を返すようになった時」だけです
    #[test]
    fn closest_points_on_segments_rejects_a_degenerate_second_segment() {
        let o = Vec3Fix::ZERO;
        let x4 = Vec3Fix::from_int(4, 0, 0);
        let up = Vec3Fix::from_int(0, 1, 0);
        let tiny = Fix128::from_raw(0, 1 << 31); // 2⁻³³
        assert!(
            !tiny.is_zero() && (tiny * tiny).is_zero(),
            "この scene の前提 (|d| ≠ 0 だが |d|² が underflow) が崩れた"
        );

        // 前提: 1 本目を長さ 4 に取ると b² が生き残り、`denom` は 0 にならない
        // (= 平行判定では拾えない ⇒ `e.is_zero()` だけが防いでいる領域)
        let b = Fix128::from_int(4) * tiny;
        assert!(
            !(b * b).is_zero(),
            "b² まで underflow した この配置では denom = 0 なので平行判定が先に拾ってしまう"
        );

        // (a) 2 本目の長さ² だけが underflow する (1 本目は長さ 4)
        assert!(
            closest_points_on_segments(o, x4, up, Vec3Fix::new(tiny, Fix128::ONE, Fix128::ZERO))
                .is_none(),
            "2 本目の長さ² が underflow した対が受け付けられた \
             (e = 0 のまま t = (b·s + f)/e を踏むと 0 除算が silent に 0 を返す)"
        );
        // (b) 2 本目が素直に長さ 0 (こちらは平行判定でも拾えるが、契約として固定する)
        assert!(
            closest_points_on_segments(o, x4, up, up).is_none(),
            "2 本目が長さ 0 の線分が受け付けられた"
        );
        // (c) 退化の拒否は引数の順序に依らない
        assert!(
            closest_points_on_segments(up, Vec3Fix::new(tiny, Fix128::ONE, Fix128::ZERO), o, x4)
                .is_none(),
            "同じ退化対を引数順を入れ替えると受け付けられた = 判定が非対称"
        );
    }

    /// `t` を端点に固定した後、`s` を**解き直す** (clamp したままにしない)
    ///
    /// ⚠️ `closest_points_on_segments_clamps_into_the_endpoint_regions` は
    /// **解き直しの前後で `s` が一致する配置**しか持っていません (2 件目は素の `s = 1/2` を
    /// clamp せず、`t = 0` に固定した後の `−c/a` も 1/2) ⇒ 解き直しを丸ごと落としても
    /// green のまま通ります (変異 `EE7-no-resolve-s-after-pinning-t`)
    ///
    /// # 閉形式 (2 本とも `|d₁| = 1` の x 軸線分 A `(0,0,0) → (1,0,0)` に対して)
    ///
    /// | 枝 | 線分 B | `a, e, b` | 素の `s` | 固定後の `t` | **解き直した `s`** |
    /// |---|---|---|---|---|---|
    /// | `t > 1` | `(-5/2, 1/8, -5/2) → (1/2, 1/8, -3/2)` | `1, 10, 3` | `5 → 1` | `1` | `(b − c)/a = 1/2` |
    /// | `t < 0` | `(1/4, 1/8, -5/2) → (-11/4, 1/8, -11/2)` | `1, 18, −3` | `11/4 → 1` | `0` | `−c/a = 1/4` |
    ///
    /// 幾何でも同じ: どちらの枝でも最近接は **B の端点**で、その端点の `x` 座標
    /// (`1/2` と `1/4`、どちらも `[0, 1]` の内部) が `s` になります 解き直さないと
    /// `s` は clamp された `1` に居残り、A 上の最近接点が端点 `(1,0,0)` にずれます
    ///
    /// # 過剰 pin の線引き
    ///
    /// 期待値は **Ericson の手順でなく幾何そのもの** (真の最近接点) から来ているので、
    /// アルゴリズムを別形式 (例: 端点 clamp 4 候補 + 内部解の総当たり) に差し替えても
    /// **正しければ同じ値になります** ⇒ 実装の自由度 (clamp の書き方 / 中間変数 / 分岐順) は
    /// 固定していません 対照 `CTL3-closest-points-equivalent-rewrite` (同関数の等価な
    /// 書き換え) は green であることを実測済みです
    #[test]
    fn closest_points_on_segments_resolves_s_again_after_pinning_t() {
        let half = Fix128::from_ratio(1, 2);
        let quarter = Fix128::from_ratio(1, 4);
        let eighth = Fix128::from_ratio(1, 8);
        let a0 = Vec3Fix::from_int(0, 0, 0);
        let a1 = Vec3Fix::from_int(1, 0, 0);

        // 枝 t > 1: B の端点 q₂ = (1/2, 1/8, -3/2) が最近接 ⇒ s = 1/2
        let b0 = Vec3Fix::new(-Fix128::from_ratio(5, 2), eighth, -Fix128::from_ratio(5, 2));
        let b1 = Vec3Fix::new(half, eighth, -Fix128::from_ratio(3, 2));
        let Some((s, t)) = closest_points_on_segments(a0, a1, b0, b1) else {
            panic!("t > 1 の枝で最近接対が得られなかった");
        };
        assert_eq!(
            t,
            Fix128::ONE,
            "oracle: 素の t = 13/10 が端点 1 へ固定される"
        );
        assert_eq!(
            s, half,
            "oracle: t を 1 に固定した後 s は (b − c)/a = 1/2 に解き直される \
             (clamp 後の 1 に居残ってはいけない)"
        );
        // 幾何の裏取り: 最近接点は B の端点 q₂、A 側はその x 座標
        assert_eq!(
            a0 + (a1 - a0) * s,
            Vec3Fix::new(half, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(b0 + (b1 - b0) * t, b1);

        // 枝 t < 0: B の端点 p₂ = (1/4, 1/8, -5/2) が最近接 ⇒ s = 1/4
        let b0 = Vec3Fix::new(quarter, eighth, -Fix128::from_ratio(5, 2));
        let b1 = Vec3Fix::new(
            -Fix128::from_ratio(11, 4),
            eighth,
            -Fix128::from_ratio(11, 2),
        );
        let Some((s, t)) = closest_points_on_segments(a0, a1, b0, b1) else {
            panic!("t < 0 の枝で最近接対が得られなかった");
        };
        assert_eq!(
            t,
            Fix128::ZERO,
            "oracle: 素の t = −13/24 が端点 0 へ固定される"
        );
        assert_eq!(
            s, quarter,
            "oracle: t を 0 に固定した後 s は −c/a = 1/4 に解き直される"
        );
        assert_eq!(
            a0 + (a1 - a0) * s,
            Vec3Fix::new(quarter, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(b0 + (b1 - b0) * t, b0);
    }

    /// 辺-辺の候補対が、頂点を共有する辺対を含まない (契約)
    ///
    /// ⚠️ 共有頂点の対は距離 0 が**構成上の事実**なので、残すと毎 iteration 発火して
    /// 伸び拘束と喧嘩するだけになります 変異 `EE2` でここが red
    #[test]
    fn edge_edge_candidates_never_pair_two_edges_that_share_a_vertex() {
        let cloth = folded_cloth_with_contacts();
        let margin = cloth.config.self_collision_distance.double();
        let candidates = cloth.collect_edge_edge_candidates(margin);
        assert!(
            !candidates.is_empty(),
            "折り返した布で辺-辺候補が 0 件 この scene は契約を試せない"
        );
        let shared = candidates
            .iter()
            .filter(|&&[i0, i1, j0, j1]| i0 == j0 || i0 == j1 || i1 == j0 || i1 == j1)
            .count();
        assert_eq!(shared, 0, "頂点を共有する辺対が {shared} 件混ざっている");

        // 辺集合そのものが mesh から正しく作られていること (重複なし / 昇順)
        let edges = cloth.mesh_edges();
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(edges, sorted, "mesh_edges が昇順かつ重複なしでない");
        for e in &edges {
            assert!(e[0] < e[1], "辺 {e:?} が [lo, hi] に正規化されていない");
        }
    }

    /// X 字交差の scene — **どの頂点もどの面にも近づかないまま、辺どうしが交差する**
    ///
    /// ```text
    ///   T0 = [0,1,2]   0 = (-1, 0, 0)   1 = (1, 0, 0)   2 = (0, -8, 0)      (XY 平面)
    ///   T1 = [3,4,5]   3 = (0, h, -1)   4 = (0, h, 1)   5 = (0, h+8, 0)     (YZ 平面)
    /// ```
    ///
    /// 辺 `(0,1)` は x 軸に沿い、辺 `(3,4)` は z 軸に沿って高さ `h` を通るので、
    /// 2 辺は上から見て直交して交差し、距離はちょうど `h` です
    ///
    /// # なぜ頂点-面では見えないか (閉形式)
    ///
    /// `T1` は平面 `x = 0` に載り `y ≥ h` なので、頂点 0 = `(-1,0,0)` の最近接点は
    /// 辺 `(3,4)` 上の `(0,h,0)` で距離² は `1 + h²` 頂点 1 も対称に同じ 頂点 3 / 4 も
    /// `T0` (平面 `z = 0`、`y ≤ 0`) に対して同じ値 残る頂点 2 / 5 は `|y|` が 8 以上
    /// 離れています **`h = 1/8`、`thickness = 1/4` では `1 + h² = 65/64 ≫ 1/16 = thickness²`
    /// なので、頂点-面の対は 1 つも閾値に入りません**
    fn x_crossing_scene(h: Fix128, thickness: Fix128) -> Cloth {
        let mut cloth = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 3, 2, Fix128::ONE);
        assert_eq!(cloth.particle_count(), 6);
        cloth.edge_constraints.clear();
        cloth.bend_constraints.clear();
        cloth.triangles = vec![[0, 1, 2], [3, 4, 5]];
        cloth.config.gravity = Vec3Fix::ZERO;
        cloth.config.damping = Fix128::ONE;
        cloth.config.self_collision = true;
        cloth.config.self_collision_distance = thickness;
        let eight = Fix128::from_int(8);
        cloth.positions[0] = Vec3Fix::from_int(-1, 0, 0);
        cloth.positions[1] = Vec3Fix::from_int(1, 0, 0);
        cloth.positions[2] = Vec3Fix::new(Fix128::ZERO, -eight, Fix128::ZERO);
        cloth.positions[3] = Vec3Fix::new(Fix128::ZERO, h, Fix128::from_int(-1));
        cloth.positions[4] = Vec3Fix::new(Fix128::ZERO, h, Fix128::ONE);
        cloth.positions[5] = Vec3Fix::new(Fix128::ZERO, h + eight, Fix128::ZERO);
        cloth.prev_positions = cloth.positions.clone();
        cloth.velocities.fill(Vec3Fix::ZERO);
        cloth
    }

    /// 2 辺が軸平行なままである限り、符号付き隙間は `y₃ − y₀` そのもの
    ///
    /// ⚠️ **実装の距離関数を呼ばずに測るための独立な量です** 辺 `(0,1)` は x 軸、
    /// 辺 `(3,4)` は z 軸に沿い、上から見て交差しているので、`y₀ = y₁` かつ `y₃ = y₄`
    /// である限り 2 辺の最近接距離は `|y₃ − y₀|` に等しくなります (符号は交差の有無を
    /// 表すので捨てません) 前提の 2 つの等式は呼び出し側で assert します
    fn x_crossing_signed_gap(cloth: &Cloth) -> Fix128 {
        assert_eq!(
            cloth.positions[0].y, cloth.positions[1].y,
            "辺 (0,1) が x 軸に平行でなくなった この計器は使えない"
        );
        assert_eq!(
            cloth.positions[3].y, cloth.positions[4].y,
            "辺 (3,4) が z 軸に平行でなくなった この計器は使えない"
        );
        cloth.positions[3].y - cloth.positions[0].y
    }

    /// 辺-辺の投影 1 回で、辺どうしの隙間が **厳密に** `thickness` になる (閉形式)
    ///
    /// # 閉形式 (`h = 1/8`, `thickness = 1/4`、逆質量は全て 1)
    ///
    /// ```text
    /// s = t = 1/2        n = (pₐ − p_b)/|pₐ − p_b| = (0,−1,0)
    /// denom = 1·(1/2)² + 1·(1/2)² + 1·(1/2)² + 1·(1/2)² = 1
    /// λ     = (1/4 − 1/8) / 1 = 1/8
    /// Δ₀ = Δ₁ = +λ·1·(1/2)·n = (0,−1/16,0)   Δ₃ = Δ₄ = −λ·1·(1/2)·n = (0,+1/16,0)
    /// ```
    ///
    /// 隙間は `1/8 + 1/16 + 1/16 = 1/4 = thickness` になり、接触に参加しない頂点 2 / 5 は
    /// `hits = 0` なので 1 bit も動きません
    ///
    /// ⚠️ **この 1 本が辺-辺側の変異 3 つを全部捕まえます** `EE1` (clamp 除去) は
    /// `s = 2` 側に飛んで Δ が変わる、`EE2` (共有頂点の除外を落とす) は T0 / T1 の
    /// 隣接辺対が距離 0 で発火して頂点 2 / 5 まで動く、`EE3` (閾値の比較反転) は
    /// 接触が skip されて 1 bit も動かない
    #[test]
    fn the_edge_edge_projection_restores_exactly_the_contact_thickness() {
        let eighth = Fix128::from_ratio(1, 8);
        let quarter = Fix128::from_ratio(1, 4);
        let sixteenth = Fix128::from_ratio(1, 16);
        let mut cloth = x_crossing_scene(eighth, quarter);
        let before = cloth.positions.clone();
        assert_eq!(
            x_crossing_signed_gap(&cloth),
            eighth,
            "scene の初期隙間は h = 1/8"
        );

        // ⚠️ 頂点-面の段は、この scene では 1 bit も動かせない (閉形式で距離² = 65/64)
        let margin = quarter.double();
        let vf = cloth.collect_vertex_face_candidates(margin);
        cloth.solve_vertex_face_self_collision(&vf);
        assert_eq!(
            cloth.positions,
            before,
            "頂点-面の段がこの scene を動かした X 字交差の盲点を示す scene になっていない \
             (候補 {} 件)",
            vf.len()
        );

        let ee = cloth.collect_edge_edge_candidates(margin);
        assert_eq!(
            ee.iter()
                .filter(|&&[i0, i1, j0, j1]| [i0, i1, j0, j1] == [0, 1, 3, 4])
                .count(),
            1,
            "辺 (0,1) × (3,4) の対が候補に 1 件だけ入っていない (候補 {ee:?})"
        );
        cloth.solve_edge_edge_self_collision(&ee);

        let neg_sixteenth = -sixteenth;
        assert_eq!(
            cloth.positions[0],
            Vec3Fix::new(Fix128::from_int(-1), neg_sixteenth, Fix128::ZERO),
            "oracle: Δ₀ = −1/16 (denom = 1 なので厳密)"
        );
        assert_eq!(
            cloth.positions[1],
            Vec3Fix::new(Fix128::ONE, neg_sixteenth, Fix128::ZERO),
            "oracle: Δ₁ = −1/16"
        );
        assert_eq!(
            cloth.positions[3],
            Vec3Fix::new(Fix128::ZERO, eighth + sixteenth, Fix128::from_int(-1)),
            "oracle: Δ₃ = +1/16"
        );
        assert_eq!(
            cloth.positions[4],
            Vec3Fix::new(Fix128::ZERO, eighth + sixteenth, Fix128::ONE),
            "oracle: Δ₄ = +1/16"
        );
        assert_eq!(
            cloth.positions[2], before[2],
            "接触に参加しない頂点 2 が動いた (hits = 0 のはず)"
        );
        assert_eq!(
            cloth.positions[5], before[5],
            "接触に参加しない頂点 5 が動いた (hits = 0 のはず)"
        );
        assert_eq!(
            x_crossing_signed_gap(&cloth),
            quarter,
            "oracle: 1 回の投影で辺-辺の隙間が厳密に thickness = 1/4 になる"
        );
    }

    /// 辺-辺の投影が `s` と `t` を**別々に**使い、逆質量で重み付けする (非対称な閉形式)
    ///
    /// ⚠️ `the_edge_edge_projection_restores_exactly_the_contact_thickness` の scene は
    /// `s = t = 1/2` かつ逆質量が全て 1 なので、**`s` と `t` を取り違えても / 逆質量を
    /// 無視して等分しても同じ答えになります** 実測でその 2 変異
    /// (`EE18-use-s-in-place-of-t` / `EE15-ignore-inv-mass-equal-split`) を red にしていたのは
    /// `test_cloth_self_collision` の `gap_on > gap_off` だけで、**閉形式の歯が無い**状態でした
    /// 本 test は `s ≠ t` かつ逆質量 4 種を全て違う値にして、その層に歯を置きます
    ///
    /// # scene
    ///
    /// ```text
    /// T0 = [0,1,2]   0 = (0,0,0)        1 = (1,0,0)        2 = (0,-8,0)
    /// T1 = [3,4,5]   3 = (1/4,1/8,-1/2) 4 = (1/4,1/8,1/2)  5 = (1/4,1/8+8,0)
    /// ```
    ///
    /// 辺 A = `(0,1)` は x 軸、辺 B = `(3,4)` は x = 1/4 を通る z 方向 ⇒ A 上の最近接は
    /// `x = 1/4` (= `s = 1/4`)、B 上の最近接は `z = 0` (= `t = 1/2`) で **`s ≠ t`**
    ///
    /// 距離は y 差の `1/8` のみ `thickness = 1/4` に対し他の対は全て閾値外です
    /// (頂点 0 → 辺 (3,4) が `1/16 + 1/64 = 5/64 > 4/64 = thickness²` で最接近、
    /// 辺 (0,2) × (3,4) も同じ `5/64`)
    ///
    /// # 閉形式 (逆質量 `w₀ = 0`, `w₁ = 8`, `w₂ = 1/2`, `w₃ = 3/2`)
    ///
    /// ```text
    /// u = 1 − s = 3/4   v = 1 − t = 1/2   n = (pₐ − p_b)/|pₐ − p_b| = (0,−1,0)
    /// denom = 0·(3/4)² + 8·(1/4)² + (1/2)·(1/2)² + (3/2)·(1/2)² = 1/2 + 1/8 + 3/8 = 1
    /// λ     = (1/4 − 1/8) / 1 = 1/8
    /// Δ₀ = +λ·0·u·n     = 0          Δ₁ = +λ·8·s·n   = (0,−1/4,0)
    /// Δ₃ = −λ·(1/2)·v·n = (0,1/32,0) Δ₄ = −λ·(3/2)·t·n = (0,3/32,0)
    /// ```
    ///
    /// `hits` は 4 頂点とも 1 なので平均で値は変わらず、頂点 0 は `inv_mass = 0` のため
    /// 適用時に skip されます `denom = 1` は偶然でなく重みの選び方で、全段が 2 進小数です
    ///
    /// # 過剰 pin の線引き
    ///
    /// 固定するのは `accumulate_edge_edge_contacts` の doc が宣言している **XPBD の投影式**
    /// (`(s, t)` を固定した 1 次の勾配 + 逆質量重み) の値です 候補の列挙順 / 法線の求め方 /
    /// 平均の実装は固定していません (本 scene は `hits = 1` なので平均の寄与が無い)
    /// ⚠️ **拘束の定式化そのものを変えると red になります** (例: `(s, t)` の微分項を入れる)
    /// それは近似モデルの変更なので、red は差し戻しでなく**裁定を求める合図**として正しい形です
    /// 対照として辺-辺の外側の変異 (`CTL1-bend-constraint-gradient-sign-flipped` /
    /// `CTL2-stretch-rest-length-doubled`) では本 test は green のままです (誤爆なしを実測)
    #[test]
    fn the_edge_edge_projection_weights_s_t_and_the_inverse_masses_separately() {
        let quarter = Fix128::from_ratio(1, 4);
        let half = Fix128::from_ratio(1, 2);
        let eighth = Fix128::from_ratio(1, 8);
        let eight = Fix128::from_int(8);

        let scene = || {
            let mut c = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 3, 2, Fix128::ONE);
            assert_eq!(c.particle_count(), 6);
            c.edge_constraints.clear();
            c.bend_constraints.clear();
            c.triangles = vec![[0, 1, 2], [3, 4, 5]];
            c.config.gravity = Vec3Fix::ZERO;
            c.config.damping = Fix128::ONE;
            c.config.self_collision = true;
            c.config.self_collision_distance = quarter;
            c.positions[0] = Vec3Fix::from_int(0, 0, 0);
            c.positions[1] = Vec3Fix::from_int(1, 0, 0);
            c.positions[2] = Vec3Fix::new(Fix128::ZERO, -eight, Fix128::ZERO);
            c.positions[3] = Vec3Fix::new(quarter, eighth, -half);
            c.positions[4] = Vec3Fix::new(quarter, eighth, half);
            c.positions[5] = Vec3Fix::new(quarter, eighth + eight, Fix128::ZERO);
            // ⚠️ 4 つとも違う値 これが等分との差を作る
            c.inv_masses[0] = Fix128::ZERO;
            c.inv_masses[1] = eight;
            c.inv_masses[3] = half;
            c.inv_masses[4] = Fix128::from_ratio(3, 2);
            c.prev_positions = c.positions.clone();
            c.velocities.fill(Vec3Fix::ZERO);
            c
        };

        // 契約: 最近接パラメータが閉形式どおり **s ≠ t** であること (scene の前提)
        let probe = scene();
        let Some((s, t)) = closest_points_on_segments(
            probe.positions[0],
            probe.positions[1],
            probe.positions[3],
            probe.positions[4],
        ) else {
            panic!("辺 (0,1) × (3,4) で最近接対が得られなかった");
        };
        assert_eq!(s, quarter, "oracle: s = 1/4");
        assert_eq!(t, half, "oracle: t = 1/2");
        assert!(
            s != t,
            "この scene は s = t なので s / t の取り違えを試せない"
        );

        // ⚠️ 頂点-面の段はこの scene を 1 bit も動かせない (最接近が 5/64 > 4/64)
        let mut cloth = scene();
        let before = cloth.positions.clone();
        let margin = quarter.double();
        let vf = cloth.collect_vertex_face_candidates(margin);
        cloth.solve_vertex_face_self_collision(&vf);
        assert_eq!(
            cloth.positions,
            before,
            "頂点-面の段がこの scene を動かした (候補 {} 件) 辺-辺だけを測れていない",
            vf.len()
        );

        // 発火するのは辺 (0,1) × (3,4) の 1 対だけ
        let ee = cloth.collect_edge_edge_candidates(margin);
        assert_eq!(
            ee.iter()
                .filter(|&&[i0, i1, j0, j1]| [i0, i1, j0, j1] == [0, 1, 3, 4])
                .count(),
            1,
            "辺 (0,1) × (3,4) の対が候補に 1 件だけ入っていない (候補 {ee:?})"
        );
        cloth.solve_edge_edge_self_collision(&ee);

        let thirty_second = Fix128::from_ratio(1, 32);
        assert_eq!(
            cloth.positions[0], before[0],
            "oracle: 頂点 0 は inv_mass = 0 なので 1 bit も動かない"
        );
        assert_eq!(
            cloth.positions[1],
            Vec3Fix::new(Fix128::ONE, -quarter, Fix128::ZERO),
            "oracle: Δ₁ = −1/4 (= λ·w₁·s、w₁ = 8 と s = 1/4 の積)"
        );
        assert_eq!(
            cloth.positions[3],
            Vec3Fix::new(quarter, eighth + thirty_second, -half),
            "oracle: Δ₃ = +1/32 (= λ·w₂·v、w₂ = 1/2 と v = 1 − t = 1/2 の積)"
        );
        assert_eq!(
            cloth.positions[4],
            Vec3Fix::new(quarter, eighth + thirty_second * Fix128::from_int(3), half),
            "oracle: Δ₄ = +3/32 (= λ·w₃·t、w₃ = 3/2 と t = 1/2 の積)"
        );
        assert_eq!(
            cloth.positions[2], before[2],
            "接触に参加しない頂点 2 が動いた (hits = 0 のはず)"
        );
        assert_eq!(
            cloth.positions[5], before[5],
            "接触に参加しない頂点 5 が動いた (hits = 0 のはず)"
        );
        // 4 つの Δ は全部違う大きさ = 重みが潰れていない (等分だと |Δ| が揃う)
        assert!(
            (cloth.positions[1].y - before[1].y) != (cloth.positions[3].y - before[3].y)
                && (cloth.positions[3].y - before[3].y) != (cloth.positions[4].y - before[4].y),
            "Δ の大きさが揃っている = 逆質量 / s / t の重みが潰れている"
        );
    }

    /// 辺-辺の対の列挙順が結果を変えない (Jacobi 蓄積の決定性)
    #[test]
    fn edge_edge_result_is_independent_of_the_pair_order() {
        let mut forward = folded_cloth_with_contacts();
        let mut reverse = folded_cloth_with_contacts();
        let margin = forward.config.self_collision_distance.double();
        let mut candidates = forward.collect_edge_edge_candidates(margin);
        assert!(
            candidates.len() > 1,
            "辺-辺候補が {} 件では順序非依存性を試せない",
            candidates.len()
        );
        let before = forward.positions.clone();
        forward.solve_edge_edge_self_collision(&candidates);
        assert!(
            forward.positions != before,
            "辺-辺の斥力が 1 bit も発火しない この scene は判別に向かない"
        );
        candidates.reverse();
        reverse.solve_edge_edge_self_collision(&candidates);
        assert_eq!(
            forward.positions, reverse.positions,
            "辺-辺の対を逆順に処理すると結果が変わる = 蓄積が順序依存"
        );
    }

    /// 2 段を 1 本の buffer に統合しても、両方の対の列挙順が結果を変えない
    ///
    /// ⚠️ **buffer を統合すると蓄積の順序が変わる**ので、段ごとの順序非依存
    /// (`self_contact_result_is_independent_of_the_pair_order` /
    /// `edge_edge_result_is_independent_of_the_pair_order`) とは別に測り直す必要があります
    /// 成立の根拠は `Fix128` の加算が `Z/2¹²⁸` の厳密な群演算であること
    /// (`tests/reduction_order_independence.rs`) で、段をまたいでも同じです
    #[test]
    fn self_collision_is_independent_of_the_pair_order_across_both_stages() {
        let mut forward = folded_cloth_with_contacts();
        let mut reverse = folded_cloth_with_contacts();
        let margin = forward.config.self_collision_distance.double();
        let mut vf = forward.collect_vertex_face_candidates(margin);
        let mut ee = forward.collect_edge_edge_candidates(margin);
        assert!(
            vf.len() > 1 && ee.len() > 1,
            "候補が 頂点-面 {} 件 / 辺-辺 {} 件 では順序非依存性を試せない",
            vf.len(),
            ee.len()
        );
        let before = forward.positions.clone();
        forward.solve_self_collision(&vf, &ee);
        assert!(
            forward.positions != before,
            "2 段を統合した投影が 1 bit も発火しない この scene は判別に向かない"
        );
        vf.reverse();
        ee.reverse();
        reverse.solve_self_collision(&vf, &ee);
        assert_eq!(
            forward.positions, reverse.positions,
            "両段の対を逆順に処理すると結果が変わる = 統合 buffer の蓄積が順序依存"
        );
    }

    /// 統合 buffer では、同じ頂点に触れた**両段**の件数で割る (段ごとに割らない)
    ///
    /// # 閉形式
    ///
    /// `x_crossing_scene` に、辺 `(0,1)` の両端 0 / 1 を巻き込む頂点-面接触を 1 件
    /// 追加した状態で 1 回投影します 頂点 0 / 1 は **辺-辺 1 件 + 頂点-面 1 件 = 2 件**に
    /// 触られるので、統合 buffer なら両方の寄与の和を **2** で割ります
    ///
    /// ⚠️ 段ごとに割ると `(Σ_ee)/1 + (Σ_vf)/1` になって **辺-辺の寄与が 2 倍**残ります
    /// それが crumple で「自己接触 ON が OFF より悪い」を生んだ形です
    /// (詳細は `solve_self_collision` の doc の表)
    #[test]
    fn the_averaging_denominator_counts_contacts_from_both_stages() {
        let eighth = Fix128::from_ratio(1, 8);
        let quarter = Fix128::from_ratio(1, 4);
        let ee_pair = [[0u32, 1, 3, 4]];
        let vf_pair = [(0u32, 1u32)]; // 頂点 0 と三角形 T1 = [3,4,5]

        // 頂点 0 を x = −1 から x = −1/8 に寄せる T1 は平面 x = 0 に載るので、
        // 頂点 0 から辺 (3,4) までの距離² は (1/8)² + (1/8)² = 1/32 < 1/16 = thickness²
        // になり、頂点-面の接触が 1 件立つ 辺 (0,1) x (3,4) の交差は保たれる
        let scene = || {
            let mut c = x_crossing_scene(eighth, quarter);
            c.positions[0] = Vec3Fix::new(-eighth, Fix128::ZERO, Fix128::ZERO);
            c.prev_positions = c.positions.clone();
            c
        };

        let base = scene().positions[0];
        let margin = quarter.double();
        let probe = scene();
        assert!(
            probe
                .collect_vertex_face_candidates(margin)
                .contains(&(0, 1)),
            "頂点 0 と三角形 1 の対が候補に入らない"
        );
        assert!(
            probe
                .collect_edge_edge_candidates(margin)
                .contains(&ee_pair[0]),
            "辺 (0,1) x (3,4) の対が候補に入らない"
        );

        let mut vf_only = scene();
        vf_only.solve_self_collision(&vf_pair, &[]);
        let d_vf = vf_only.positions[0] - base;
        let mut ee_only = scene();
        ee_only.solve_self_collision(&[], &ee_pair);
        let d_ee = ee_only.positions[0] - base;
        assert!(
            d_vf != Vec3Fix::ZERO && d_ee != Vec3Fix::ZERO,
            "片方の段が発火していない (頂点-面 {d_vf:?} / 辺-辺 {d_ee:?}) この scene は判別に向かない"
        );

        let mut both = scene();
        both.solve_self_collision(&vf_pair, &ee_pair);
        assert_eq!(
            both.positions[0] - base,
            (d_vf + d_ee) / Fix128::from_int(2),
            "統合 buffer の補正が「段ごとの寄与の和の 1/2」になっていない \
             = 平均の分母が両段の件数 2 になっていない \
             (段ごとに割ると和そのものが残り、辺-辺が 2 倍過大評価される)"
        );
    }

    /// 1 frame で辺どうしが入れ替わる X 字交差が、辺-辺の段だけで止まる
    ///
    /// # scene
    ///
    /// `x_crossing_scene(h = 1/2, thickness = 1/4)` の `T1` を 1 frame で `−3/4`
    /// 平行移動させる (`v = (0, −48, 0)`, `dt = 1/64`) 符号付き隙間は `+1/2` から
    /// 自由落下なら `−1/4` になる = **2 辺が入れ替わる (貫通する)**
    ///
    /// ⚠️ **`dt = 1/64` は「60 fps を模す」よりも厳密性を優先した選択です** `1/60` は
    /// 2 進小数でないので `Fix128::from_ratio(1, 60)` が切り捨てられ、`substep_dt = dt/4`
    /// と `v·dt` を経た変位が閉形式から **12 ulp** ずれます (実測、`-0.25` と表示される
    /// のに `assert_eq!` が red) `1/64` なら `substep_dt = 1/256`、1 substep の変位は
    /// `−48/256 = −3/16` で全て厳密です
    ///
    /// # ⚠️ 頂点-面の不変量はこの貫通を 0 と報告する
    ///
    /// `remaining_self_contact_crossings` が測るのは「頂点の掃過線分が三角形を貫いたか」
    /// なので、辺どうしの交差は **自己接触 OFF でも 0** です これが
    /// `src/cloth.rs` に辺-辺の段が必要だった理由そのもので、本 test の反 vacuous
    /// guard も兼ねます (0 なのは正しいからでなく、その量を見ていないから)
    #[test]
    fn an_edge_pair_that_swaps_sides_in_one_frame_is_caught_only_by_the_edge_edge_pass() {
        let half = Fix128::from_ratio(1, 2);
        let quarter = Fix128::from_ratio(1, 4);
        let dt = Fix128::from_ratio(1, 64);

        let run = |self_collision: bool| -> (Fix128, usize) {
            let mut cloth = x_crossing_scene(half, quarter);
            cloth.config.self_collision = self_collision;
            // T1 を 1 frame で −3/4 平行移動 (剛体的に動かすので T1 の形は変わらない)
            for i in 3..6 {
                cloth.velocities[i] =
                    Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-48), Fix128::ZERO);
            }
            let start = cloth.positions.clone();
            cloth.step(dt);
            let crossings = cloth.remaining_self_contact_crossings(&start);
            (x_crossing_signed_gap(&cloth), crossings)
        };

        let (gap_off, crossings_off) = run(false);
        assert_eq!(
            gap_off,
            -quarter,
            "自己接触 OFF で符号付き隙間が −1/4 (= 2 辺が入れ替わった) にならない \
             この scene は貫通を試せていない (実測 {})",
            gap_off.to_f32()
        );
        assert_eq!(
            crossings_off, 0,
            "頂点-面の不変量が辺-辺の交差を検出した この scene は盲点を示していない"
        );

        let (gap_on, crossings_on) = run(true);
        assert!(
            gap_on >= quarter,
            "自己接触 ON で辺-辺の隙間が thickness = 1/4 を下回った (実測 {})",
            gap_on.to_f32()
        );
        assert_eq!(crossings_on, 0, "自己接触 ON で頂点-面の貫通が残った");
    }

    /// 折り返して自己接触させた 5x5 の布 (頂点-面 / 辺-辺 の両方に候補が出る)
    fn folded_cloth_with_contacts() -> Cloth {
        const R: usize = 5;
        let mut cloth = Cloth::new_grid(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(4),
            R,
            R,
            Fix128::from_ratio(1, 100),
        );
        cloth.config.self_collision = true;
        cloth.config.self_collision_distance = Fix128::from_ratio(3, 2);
        for x in 0..R {
            cloth.positions[3 * R + x] =
                Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::from_int(2));
            cloth.positions[4 * R + x] =
                Vec3Fix::new(cloth.positions[x].x, Fix128::ONE, Fix128::ONE);
        }
        cloth
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
