//! Density-map boundaries for the PBF fluid: a static solid given by its SDF becomes a field
//! of boundary density and its gradient, and the force the fluid exerts is shared between
//! solids along their exact surface normals.
//!
//! # MODEL
//!
//! **Equations.** Boundary density (Koschier & Bender 2017, density maps): a solid `B` with signed
//! distance `d` (negative inside) contributes to the density of a fluid particle at `x`
//!
//! ```text
//!   rho_B(x) = rho_0 * integral chi_B(y) W(x - y) dy,      grad rho_B(x) = rho_0 * integral chi_B(y) grad W(x - y) dy
//! ```
//!
//! with `W` the fluid's poly6 kernel (`integral W = 1`), its gradient taken with the spiky
//! kernel as in the fluid-fluid terms, and `chi_B` the volume fraction of a grid cell,
//! `clamp(1/2 - d / step, 0, 1)` (linear across one cell, so a surface move smaller than one
//! cell still moves the map). The integral is a sum over the grid nodes within the kernel
//! radius (`step` = rest spacing / 4), evaluated once when the solid is added and read back
//! by trilinear interpolation. The map is a deterministic
//! function of the SDF samples (the SDF must be 1-Lipschitz, as an exact distance is: nodes
//! farther than `h + step / 2` outside are skipped as exactly zero): the SDF is evaluated in `f32` at the grid nodes (the same
//! values on every platform for the same field) and everything after that is `Fix128`.
//!
//! **Units.** SI: `rho_B` in kg m^-3, `grad rho_B` in kg m^-4, forces on the solids in N.
//!
//! Fluid side: a fluid particle adds `rho_B` to its density and `grad rho_B / rho_0` to its
//! constraint gradient for every solid; the PBF position correction from solid `B` is
//! `lambda_i * grad rho_B / rho_0` (relaxation applied). Nothing in this module changes what
//! the fluid feels.
//!
//! Open, the map term in the constraint gradient: whether `grad rho_B / rho_0` belongs in
//! the constraint gradient (the `lambda` denominator) is not settled. Without it the force
//! oracles give the same answers (lying on the floor 0.9964, 2 h above it 1.0175), and the
//! density next to the walls converges better. Mean `|rho / rho_0 - 1|` after `k` iterations,
//! one substep from the rest lattice in a 0.8 m tank given by its SDF
//! (`examples/fluid_sdf_wall_convergence.rs`), particles within `h` of a wall / interior:
//!
//! ```text
//!   k        1      2      4      6      8      10
//!   with   4.25   3.51   3.41   3.56   3.68   3.70  %  (walls)   4.77  7.06  6.96  7.81  8.40  8.34  % (interior)
//!   without 4.25  3.57   3.26   3.17   3.16   3.21  %  (walls)   4.77  6.09  5.84  6.01  6.28  6.80  % (interior)
//! ```
//!
//! The interior error rises with `k` in both, so the base PBF iteration itself does not
//! converge here; the term is kept until it is measured again on a converging solver.
//!
//! Force on the solids (normal-decomposed attribution): the correction a particle gets
//! from all solids together, `F = sum_B lambda_i grad rho_B / rho_0`, is the reaction of the
//! solids (Newton 3: the solids receive `-F * m / dt^2` in total). It is shared out as:
//!
//! - one solid within the kernel: that solid gets all of `F`;
//! - two solids within the kernel (e.g. the fluid at the foot of a body resting on a floor):
//!   `F = a_1 n_1 + a_2 n_2 + r`, least squares over the two unit outward normals of the
//!   solids at the particle (from their SDFs), with the remainder `r` (the part of `F` out of
//!   the plane of the normals) split by the solids' density shares. Each solid is credited a
//!   push along its own surface. Credited by its own gradient instead, a body resting on a
//!   floor took the floor's vertical reaction on the fluid at its foot: the vertical force on
//!   a submerged cube lying on the floor was 0.44 of the closed form, and is 1.02 with this
//!   split (same fluid state, prototype at s = 0.1 m, h = 2 s);
//! - two solids whose normals are near-parallel or opposed (Gram determinant
//!   `1 - (n_1 . n_2)^2 < GRAM_MIN = 1/10`, e.g. the fluid in a gap between a floor and the
//!   bottom of a body): the split is ill-posed, and each solid keeps its own push
//!   `lambda_i grad rho_B / rho_0` (opposed pushes are each solid's actual load; they partly
//!   cancel in `F`, and splitting `F` would lose the gap pressure);
//! - three or more solids within one kernel: each keeps its own push (not covered by an
//!   oracle).
//!
//! The second solid's share is computed as `F` minus the first's, so the shares sum to `F`
//! exactly in fixed point on every branch; `tests/analytic_fluid_sdf_body.rs` checks on every
//! frame that the per-solid forces sum to minus the fluid-side total, which is accumulated
//! separately.
//!
//! **Conservation.** Momentum between the fluid and the solids: the forces credited to the
//! solids sum exactly (fixed point) to minus the solids' push on the fluid on every branch of
//! the split (Newton 3); the solids are static, so they take that momentum out, as a wall
//! does. Mass: the maps add no particles. Torque on a solid is not reported.
//!
//! **Approximations.** The volume fraction is linear across one grid cell; the map is read
//! by trilinear interpolation; the SDF is sampled in `f32`; nodes farther than `h + step / 2`
//! outside are taken as exactly zero (1-Lipschitz SDF); interior and far nodes are stored
//! although constant; three or more solids within one kernel each keep their own push.
//!
//! Cost: one map per solid, built when the solid is added: one `Fix128` and one `Vec3Fix`
//! per grid node over the given box. Measured (release, arm64 desktop, s = 0.1 m): the box
//! around a 0.4 m cube is 68,921 nodes, 302 ms, 4.4 MB. Interior and far nodes are constant
//! but stored; a band-only store is not implemented. Per frame (same machine, 1984 particles,
//! 2 substeps x 10 iterations, a tank and a cube): the map lookups take 11.8 ms and the SDF
//! normals at most 3.6 ms, against 692 ms for the base PBF frame without solids, measured
//! while other work shared the machine (`examples/fluid_sdf_body_frame_cost.rs`). The working
//! frame budget for a fluid resident is 4 ms on the target device's CPU at 60 Hz, an
//! assumption until the per-element budget is set: the base PBF dominates.
//!
//! **Range and open limitation.** The coupling is verified at s = 0.1 m, h = 2 s for a cube of
//! 4 spacings: lying on the floor (vertical force within 5 % of the top-only closed form) and
//! 2 h above it (Archimedes within 5 % for the placement the test uses: 0.990 at 900 frames).
//! The 2 h result depends on where the particles start: shifting the initial lattice by a tenth
//! of a spacing in +x, in +z, or in -x and +z gives 0.934, 0.930 and 1.032 at 900 frames
//! (measured 2026-11-19, cube 2 h up, mean of the last 60 frames; the lattice in reverse order
//! gives the unshifted 0.990 bit for bit), so two of four placements are outside 5 %. Whether
//! that spread shrinks with more frames, particles or iterations (sampling) or stays (model)
//! is not measured yet. With only 1 h of fluid between the floor and the
//! body's bottom (two particle layers), the buoyancy is 0.79-0.81 of Archimedes in the
//! prototype on both PBF and DFSPH: this case is open and not covered by this split. Solids
//! are static; moving bodies are not covered.
//!
//! **Validation.** `tests/analytic_fluid_sdf_body.rs` (lying on the floor against the top-only
//! closed form, 2 h up against Archimedes, Newton 3 on every frame, bit-equal map rebuild, the
//! flat-wall map against `rho_0 / 2`, degenerate inputs), and the split itself:
//! `split_at_a_corner_gives_each_solid_its_own_axis`, `opposed_and_parallel_normals_fall_back`,
//! `gram_min_is_one_tenth`.
//!
//! **Tiers.** L0 only.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Below this Gram determinant of two solids' normals the split is ill-posed and each solid
/// keeps its own push
pub const GRAM_MIN: Fix128 = Fix128::from_raw(0, 0x1999_9999_9999_9999);

/// Boundary density `rho_B` and its gradient on a regular grid (see the module MODEL)
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryMap {
    lo: Vec3Fix,
    step: Fix128,
    inv_step: Fix128,
    dims: [usize; 3],
    rho: Vec<Fix128>,
    grad: Vec<Vec3Fix>,
}

fn poly6(r_sq: Fix128, h: Fix128) -> Fix128 {
    super::fluid::poly6_kernel(r_sq, h)
}

fn spiky_grad(r: Fix128, h: Fix128) -> Fix128 {
    super::fluid::spiky_grad_kernel(r, h)
}

/// Signed distance of the solid at a world point (negative inside), as `Fix128`
pub(crate) fn sdf_distance(sdf: &SdfCollider, p: Vec3Fix) -> Fix128 {
    let (lx, ly, lz) = sdf.world_to_local(p);
    Fix128::from_f32(sdf.field.distance(lx, ly, lz) * sdf.scale_f32)
}

/// Unit outward normal of the solid at a world point
pub(crate) fn sdf_normal(sdf: &SdfCollider, p: Vec3Fix) -> Vec3Fix {
    let (lx, ly, lz) = sdf.world_to_local(p);
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
    sdf.local_normal_to_world(nx, ny, nz)
}

impl BoundaryMap {
    /// Build the map of one solid over the box `[lo, hi]` (world space) with grid `step`.
    /// The box should cover every point a fluid particle can reach within `kernel_radius`
    /// of the solid; outside it the map reads zero.
    ///
    /// The field must be 1-Lipschitz (it never over-estimates the distance), as an exact
    /// distance is: grid nodes farther than `kernel_radius + step / 2` outside are skipped as
    /// exactly zero, which a field that over-estimates (a twist, a displacement) would make
    /// wrong.
    ///
    /// Returns an empty map (reads zero everywhere) when `step` or `kernel_radius` is not
    /// positive or the box is inverted.
    #[must_use]
    pub fn from_sdf(
        sdf: &SdfCollider,
        lo: Vec3Fix,
        hi: Vec3Fix,
        step: Fix128,
        rest_density: Fix128,
        kernel_radius: Fix128,
    ) -> Self {
        let empty = Self {
            lo,
            step,
            inv_step: Fix128::ZERO,
            dims: [0; 3],
            rho: Vec::new(),
            grad: Vec::new(),
        };
        if step <= Fix128::ZERO
            || kernel_radius <= Fix128::ZERO
            || hi.x < lo.x
            || hi.y < lo.y
            || hi.z < lo.z
        {
            return empty;
        }
        let h = kernel_radius;
        let count = |a: Fix128, b: Fix128| ((b - a) / step).ceil().hi as usize + 1;
        let dims = [count(lo.x, hi.x), count(lo.y, hi.y), count(lo.z, hi.z)];
        let k = (h / step).ceil().hi;
        // indicator on the grid padded by the kernel reach on every side
        let pad = k as usize;
        let pd = [dims[0] + 2 * pad, dims[1] + 2 * pad, dims[2] + 2 * pad];
        let node = |i: usize, axis: Fix128| axis + step * Fix128::from_int(i as i64 - k);
        let mut chi = vec![Fix128::ZERO; pd[0] * pd[1] * pd[2]];
        let mut dist = vec![Fix128::ZERO; pd[0] * pd[1] * pd[2]];
        let inv_step = Fix128::ONE / step;
        for a in 0..pd[0] {
            for b in 0..pd[1] {
                for c in 0..pd[2] {
                    let p = Vec3Fix::new(node(a, lo.x), node(b, lo.y), node(c, lo.z));
                    let d = sdf_distance(sdf, p);
                    dist[(a * pd[1] + b) * pd[2] + c] = d;
                    let v = Fix128::ONE.half() - d * inv_step;
                    chi[(a * pd[1] + b) * pd[2] + c] = if v <= Fix128::ZERO {
                        Fix128::ZERO
                    } else if v >= Fix128::ONE {
                        Fix128::ONE
                    } else {
                        v
                    };
                }
            }
        }
        // kernel weights per offset inside the support (rho_0 dv W and rho_0 dv grad W)
        let dv = step * step * step * rest_density;
        let h_sq = h * h;
        let mut offsets: Vec<(i64, i64, i64, Fix128, Vec3Fix)> = Vec::new();
        for di in -k..=k {
            for dj in -k..=k {
                for dk in -k..=k {
                    let off = Vec3Fix::new(
                        step * Fix128::from_int(di),
                        step * Fix128::from_int(dj),
                        step * Fix128::from_int(dk),
                    );
                    let r_sq = off.length_squared();
                    if r_sq >= h_sq {
                        continue;
                    }
                    let w = poly6(r_sq, h) * dv;
                    // x - y = off (y = x - off), gradient along x - y
                    let g = if r_sq.is_zero() {
                        Vec3Fix::ZERO
                    } else {
                        let r = r_sq.sqrt();
                        off / r * (spiky_grad(r, h) * dv)
                    };
                    offsets.push((di, dj, dk, w, g));
                }
            }
        }
        let far = h + step.half();
        let n_nodes = dims[0] * dims[1] * dims[2];
        let mut rho = vec![Fix128::ZERO; n_nodes];
        let mut grad = vec![Vec3Fix::ZERO; n_nodes];
        for a in 0..dims[0] {
            for b in 0..dims[1] {
                for c in 0..dims[2] {
                    let (pa, pb, pc) = (a as i64 + k, b as i64 + k, c as i64 + k);
                    // a node farther than h + step/2 outside the solid has no solid cell in
                    // its support (the SDF is 1-Lipschitz), so its map is exactly zero
                    if dist[(pa as usize * pd[1] + pb as usize) * pd[2] + pc as usize] >= far {
                        continue;
                    }
                    let mut r = Fix128::ZERO;
                    let mut g = Vec3Fix::ZERO;
                    for &(di, dj, dk, w, gw) in &offsets {
                        // y = x - off
                        let ya = (pa - di) as usize;
                        let yb = (pb - dj) as usize;
                        let yc = (pc - dk) as usize;
                        let x = chi[(ya * pd[1] + yb) * pd[2] + yc];
                        if x.is_zero() {
                            continue;
                        }
                        if x == Fix128::ONE {
                            r = r + w;
                            g = g + gw;
                        } else {
                            r = r + w * x;
                            g = g + gw * x;
                        }
                    }
                    let idx = (a * dims[1] + b) * dims[2] + c;
                    rho[idx] = r;
                    grad[idx] = g;
                }
            }
        }
        Self {
            lo,
            step,
            inv_step,
            dims,
            rho,
            grad,
        }
    }

    /// Number of grid nodes (memory: one `Fix128` and one `Vec3Fix` per node)
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.rho.len()
    }

    /// `rho_B` and `grad rho_B` at `p` by trilinear interpolation; zero outside the grid
    #[must_use]
    pub fn sample(&self, p: Vec3Fix) -> (Fix128, Vec3Fix) {
        if self.rho.is_empty() || self.dims.iter().any(|&d| d < 2) {
            return (Fix128::ZERO, Vec3Fix::ZERO);
        }
        let f = [
            (p.x - self.lo.x) * self.inv_step,
            (p.y - self.lo.y) * self.inv_step,
            (p.z - self.lo.z) * self.inv_step,
        ];
        let mut i0 = [0usize; 3];
        let mut t = [Fix128::ZERO; 3];
        for a in 0..3 {
            if f[a] < Fix128::ZERO || f[a] > Fix128::from_int(self.dims[a] as i64 - 1) {
                return (Fix128::ZERO, Vec3Fix::ZERO);
            }
            let fl = f[a].floor();
            let mut i = fl.hi as usize;
            let mut frac = f[a] - fl;
            if i >= self.dims[a] - 1 {
                i = self.dims[a] - 2;
                frac = Fix128::ONE;
            }
            i0[a] = i;
            t[a] = frac;
        }
        let mut r = Fix128::ZERO;
        let mut g = Vec3Fix::ZERO;
        for dx in 0..2usize {
            let wx = if dx == 1 { t[0] } else { Fix128::ONE - t[0] };
            for dy in 0..2usize {
                let wy = if dy == 1 { t[1] } else { Fix128::ONE - t[1] };
                for dz in 0..2usize {
                    let wz = if dz == 1 { t[2] } else { Fix128::ONE - t[2] };
                    let w = wx * wy * wz;
                    if w.is_zero() {
                        continue;
                    }
                    let idx =
                        ((i0[0] + dx) * self.dims[1] + (i0[1] + dy)) * self.dims[2] + (i0[2] + dz);
                    r = r + self.rho[idx] * w;
                    g = g + self.grad[idx] * w;
                }
            }
        }
        (r, g)
    }
}

/// A static solid coupled to the fluid: its SDF (for the normals) and its density map
pub struct FluidBody {
    sdf: SdfCollider,
    map: BoundaryMap,
}

impl FluidBody {
    /// Pair a solid's SDF with the density map built from it
    #[must_use]
    pub fn new(sdf: SdfCollider, map: BoundaryMap) -> Self {
        Self { sdf, map }
    }

    /// The solid's SDF
    #[must_use]
    pub fn sdf(&self) -> &SdfCollider {
        &self.sdf
    }

    /// The solid's density map
    #[must_use]
    pub fn map(&self) -> &BoundaryMap {
        &self.map
    }
}

/// Share the total push `f` of two solids along their unit normals `n_a`, `n_b` (see the
/// module MODEL): returns the share of solid `b`, or `None` when the normals are within
/// [`GRAM_MIN`] of parallel or opposed (each solid then keeps its own push). Solid `a`'s
/// share is `f` minus the returned value. `share_b` is solid `b`'s density share in `[0, 1]`
/// and splits the part of `f` out of the plane of the normals.
#[must_use]
pub fn split_two(f: Vec3Fix, n_a: Vec3Fix, n_b: Vec3Fix, share_b: Fix128) -> Option<Vec3Fix> {
    let c = n_a.dot(n_b);
    let gram = Fix128::ONE - c * c;
    if gram < GRAM_MIN {
        return None;
    }
    let fa = f.dot(n_a);
    let fb = f.dot(n_b);
    // normal equations of [n_a n_b] (a_a, a_b) = f: [[1 c] [c 1]] a = (f.n_a, f.n_b)
    let a_a = (fa - c * fb) / gram;
    let a_b = (fb - c * fa) / gram;
    let rem = f - n_a * a_a - n_b * a_b;
    Some(n_b * a_b + rem * share_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
        Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        )
    }

    #[test]
    fn split_at_a_corner_gives_each_solid_its_own_axis() {
        // floor normal +y, wall normal -x; f has both components and a z part (remainder)
        let f = v(-3.0, 5.0, 2.0);
        let b = split_two(
            f,
            v(0.0, 1.0, 0.0),
            v(-1.0, 0.0, 0.0),
            Fix128::from_ratio(1, 4),
        )
        .expect("orthogonal normals are well posed");
        let a = f - b;
        assert!((b.x.to_f64() + 3.0).abs() < 1e-12 && b.y.to_f64().abs() < 1e-12);
        assert!(
            (b.z.to_f64() - 0.5).abs() < 1e-12,
            "remainder by density share"
        );
        assert!((a.y.to_f64() - 5.0).abs() < 1e-12 && a.x.to_f64().abs() < 1e-12);
        assert!((a.z.to_f64() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn opposed_and_parallel_normals_fall_back() {
        let f = v(0.0, 1.0, 0.0);
        assert!(split_two(f, v(0.0, 1.0, 0.0), v(0.0, -1.0, 0.0), Fix128::ONE.half()).is_none());
        assert!(split_two(f, v(0.0, 1.0, 0.0), v(0.0, 1.0, 0.0), Fix128::ONE.half()).is_none());
        // 1 - cos^2 just above 1/10 is split, just below is not
        let s_hi = 0.33_f64; // sin^2 = 0.1089
        let s_lo = 0.30_f64; // sin^2 = 0.09
        let n = |s: f64| v(s, (1.0 - s * s).sqrt(), 0.0);
        assert!(split_two(f, v(0.0, 1.0, 0.0), n(s_hi), Fix128::ONE.half()).is_some());
        assert!(split_two(f, v(0.0, 1.0, 0.0), n(s_lo), Fix128::ONE.half()).is_none());
    }

    #[test]
    fn gram_min_is_one_tenth() {
        assert!((GRAM_MIN.to_f64() - 0.1).abs() < 1e-15);
    }
}
