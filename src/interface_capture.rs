//! Sharp Interface Capturing (Level Set FSM + PLIC VOF Reconstruction)
//!
//! Phase G6 of the ALICE-Physics completeness project. Sharpens the diffuse
//! interfaces of `multiphase` by:
//!
//! - **Fast Sweeping Method** for level-set reinitialisation — a single
//!   Gauss-Seidel sweep restoring `|∇φ| = 1` more efficiently than the
//!   pseudo-time iteration in `multiphase::reinitialize_level_set`.
//! - **PLIC** (Piecewise Linear Interface Calculation) reconstruction of
//!   the interface plane inside each VOF cell — the standard high-fidelity
//!   sub-cell representation for two-phase flow.
//!
//! # PLIC in brief
//!
//! Inside a cell with fluid fraction `f ∈ (0, 1)`, place a plane
//! `n̂·x = d` such that the volume of fluid on the negative side matches
//! `f · Δx³`. The normal `n̂` comes from the gradient of the smoothed VOF
//! field; the offset `d` is determined by inverting the truncation volume
//! formula.
//!
//! # References
//!
//! - Zhao, "A fast sweeping method for eikonal equations", Math. Comp. 74,
//!   2005 (FSM).
//! - Youngs, "Time-dependent multi-material flow with large fluid
//!   distortion", *Numerical Methods for Fluid Dynamics*, Academic Press,
//!   1982 (PLIC).
//! - Rider & Kothe, "Reconstructing volume tracking", J. Comp. Phys. 141,
//!   1998.
//!
//! # Integration status
//!
//! Only `fast_sweeping_reinit` is currently wired into `cfd_solver.rs`.
//! The PLIC helpers (`plic_normal`, `plic_plane_offset`,
//! `truncated_cube_volume`) are reserved crate-internal API awaiting
//! downstream integration; item-level `#[allow(dead_code)]` documents this.

use crate::math::{Fix128, Vec3Fix};
use crate::math_util::cbrt_fix;
use crate::multiphase::Grid3d;

// ============================================================================
// Fast Sweeping Method
// ============================================================================

/// Fast Sweeping reinitialisation of a level-set field. Performs `sweeps`
/// Gauss-Seidel passes in each of the 8 sweep directions, restoring
/// `|∇φ| = 1` more efficiently than the pseudo-time reinit.
///
/// `sweeps` typical value: 2-3 (each pass covers one octant).
pub fn fast_sweeping_reinit(field: &mut Grid3d, sweeps: u32) {
    for _ in 0..sweeps {
        for dir_k in [true, false] {
            for dir_j in [true, false] {
                for dir_i in [true, false] {
                    fsm_pass(field, dir_i, dir_j, dir_k);
                }
            }
        }
    }
}

fn fsm_pass(field: &mut Grid3d, forward_i: bool, forward_j: bool, forward_k: bool) {
    let ni = field.nx;
    let nj = field.ny;
    let nk = field.nz;
    for kk in 0..nk {
        let k = if forward_k { kk } else { nk - 1 - kk };
        for jj in 0..nj {
            let j = if forward_j { jj } else { nj - 1 - jj };
            for ii in 0..ni {
                let i = if forward_i { ii } else { ni - 1 - ii };
                let old = field.get(i, j, k);
                // Compute neighbor minima (absolute distances)
                let a = min_neighbor(field, i, j, k, 0);
                let b = min_neighbor(field, i, j, k, 1);
                let c = min_neighbor(field, i, j, k, 2);
                // Solve quadratic |φ - a|² + |φ - b|² + |φ - c|² = dx²
                // Simplification: use one-neighbour formula for the smallest
                // of a, b, c. This is the standard FSM approximation.
                let new = solve_fsm(a, b, c, field.dx);
                // Take the sign of the current φ to preserve inside/outside
                let signed = if old.is_negative() {
                    Fix128::ZERO - new
                } else {
                    new
                };
                // Only replace if magnitude decreased (Godunov)
                if signed.abs() < old.abs() {
                    field.set(i, j, k, signed);
                }
            }
        }
    }
}

fn min_neighbor(field: &Grid3d, i: usize, j: usize, k: usize, axis: usize) -> Fix128 {
    let (a, b) = match axis {
        0 => (
            if i > 0 {
                field.get(i - 1, j, k)
            } else {
                Fix128::from_int(1_000_000)
            },
            if i + 1 < field.nx {
                field.get(i + 1, j, k)
            } else {
                Fix128::from_int(1_000_000)
            },
        ),
        1 => (
            if j > 0 {
                field.get(i, j - 1, k)
            } else {
                Fix128::from_int(1_000_000)
            },
            if j + 1 < field.ny {
                field.get(i, j + 1, k)
            } else {
                Fix128::from_int(1_000_000)
            },
        ),
        _ => (
            if k > 0 {
                field.get(i, j, k - 1)
            } else {
                Fix128::from_int(1_000_000)
            },
            if k + 1 < field.nz {
                field.get(i, j, k + 1)
            } else {
                Fix128::from_int(1_000_000)
            },
        ),
    };
    let aa = a.abs();
    let bb = b.abs();
    if aa < bb {
        aa
    } else {
        bb
    }
}

/// Solve the 3-D Godunov Eikonal update `|∇φ| = 1` at a cell given the
/// absolute-distance neighbours `(a, b, c)` and cell spacing `dx`
/// (Session 3 I5 upgrade — replaces the 1-neighbour `min + dx`).
///
/// Algorithm (Sethian, *Level Set Methods* 2nd ed. §8.4):
/// 1. Sort ascending: `a ≤ b ≤ c`.
/// 2. Try 1-neighbour: `φ = a + h`. Accept if `φ ≤ b`.
/// 3. Try 2-neighbour: solve `(φ-a)² + (φ-b)² = h²`:
///    `φ = ½·(a + b + √(2h² − (a−b)²))`. Accept if `φ ≤ c` and radicand ≥ 0.
/// 4. Try 3-neighbour: solve `(φ-a)² + (φ-b)² + (φ-c)² = h²`:
///    `φ = ⅓·(a + b + c + √(3h² − (a−b)² − (b−c)² − (a−c)²))`.
fn solve_fsm(a: Fix128, b: Fix128, c: Fix128, dx: Fix128) -> Fix128 {
    // Sort ascending
    let mut sorted = [a, b, c];
    sorted.sort();
    let (aa, bb, cc) = (sorted[0], sorted[1], sorted[2]);

    let h = dx;
    let h_sq = h * h;

    // 1-neighbour attempt
    let phi_1 = aa + h;
    if phi_1 <= bb {
        return phi_1;
    }

    // 2-neighbour attempt: 2·h² − (a−b)² must be ≥ 0
    let diff_ab = aa - bb;
    let radicand_2 = h_sq.double() - diff_ab * diff_ab;
    if radicand_2 >= Fix128::ZERO {
        let phi_2 = ((aa + bb) + radicand_2.sqrt()).half();
        if phi_2 <= cc {
            return phi_2;
        }
    }

    // 3-neighbour attempt
    let diff_bc = bb - cc;
    let diff_ac = aa - cc;
    let radicand_3 =
        Fix128::from_int(3) * h_sq - diff_ab * diff_ab - diff_bc * diff_bc - diff_ac * diff_ac;
    if radicand_3 >= Fix128::ZERO {
        let sum = aa + bb + cc;
        return (sum + radicand_3.sqrt()) / Fix128::from_int(3);
    }

    // Fallback (degenerate case): stick with the 1-neighbour estimate
    phi_1
}

// ============================================================================
// PLIC reconstruction
// ============================================================================

/// PLIC interface normal for a VOF cell: gradient of `f` via central diffs (crate-internal).
#[allow(dead_code)] // Reserved PLIC helper — awaiting integration
#[must_use]
pub(crate) fn plic_normal(vof: &Grid3d, i: usize, j: usize, k: usize) -> Vec3Fix {
    if i == 0 || j == 0 || k == 0 || i + 1 >= vof.nx || j + 1 >= vof.ny || k + 1 >= vof.nz {
        return Vec3Fix::default();
    }
    let two_dx = vof.dx + vof.dx;
    if two_dx.is_zero() {
        return Vec3Fix::default();
    }
    // Interface normal is antiparallel to gradient of f (fluid = large f
    // volumes; interface normal points from fluid toward gas).
    let dfx = -(vof.get(i + 1, j, k) - vof.get(i - 1, j, k)) / two_dx;
    let dfy = -(vof.get(i, j + 1, k) - vof.get(i, j - 1, k)) / two_dx;
    let dfz = -(vof.get(i, j, k + 1) - vof.get(i, j, k - 1)) / two_dx;
    let mag = (dfx * dfx + dfy * dfy + dfz * dfz).sqrt();
    if mag.is_zero() {
        return Vec3Fix::default();
    }
    Vec3Fix::new(dfx / mag, dfy / mag, dfz / mag)
}

/// Position of the PLIC plane inside a cubic cell such that the volume of
/// fluid on the negative side matches `f · Δx³`.
///
/// Session 3 I4 upgrade: uses the Scardovelli-Zaleski / Rider-Kothe
/// analytical formula (2000) for the small-volume regime via
/// [`crate::math_util::cbrt_fix`] and falls back to a coarse bisection for
/// the intermediate regime. Overall convergence is O(1) for the corner
/// case (previously O(40) bisection with 4³ sub-sampling per iteration).
#[must_use]
#[allow(dead_code)] // Reserved PLIC helper — awaiting integration
pub(crate) fn plic_plane_offset(normal: Vec3Fix, f: Fix128, dx: Fix128) -> Fix128 {
    if f <= Fix128::ZERO {
        return dx * Fix128::from_int(-2);
    }
    if f >= Fix128::ONE {
        return dx * Fix128::from_int(2);
    }

    // Analytical branch (Rider & Kothe 1998, eq. 15):
    // Sort |normal| components ascending as (m1 ≤ m2 ≤ m3), rescale to
    // sum to 1, and compute the small-V threshold `v1 = m1² / (6·m2·m3)`.
    // For target ≤ v1 (plane cuts only a corner tetrahedron):
    //   d_norm = cbrt(6·m1·m2·m3·V)
    let na = normal.x.abs();
    let nb = normal.y.abs();
    let nc = normal.z.abs();
    let sum = na + nb + nc;
    if sum > Fix128::ZERO {
        // Rescale to Σm_i = 1
        let m_a = na / sum;
        let m_b = nb / sum;
        let m_c = nc / sum;
        // Sort ascending (m1 ≤ m2 ≤ m3)
        let (m1, m2, m3) = {
            let mut arr = [m_a, m_b, m_c];
            arr.sort();
            (arr[0], arr[1], arr[2])
        };
        if !m2.is_zero() && !m3.is_zero() {
            let v1 = m1 * m1 / (Fix128::from_int(6) * m2 * m3);
            // Use symmetry: if f > 0.5 → analyse (1 − f) then negate d.
            let (target, sign_flip) = if f <= Fix128::from_ratio(5, 10) {
                (f, false)
            } else {
                (Fix128::ONE - f, true)
            };
            if target <= v1 {
                // Corner-tetrahedron regime: analytical solution
                let radicand = Fix128::from_int(6) * m1 * m2 * m3 * target;
                let d_norm = cbrt_fix(radicand);
                // d_norm is expressed in the unit-cube-centered form
                // (plane origin at cube corner); shift to cube-centered
                // convention used by this module (origin at cube centre).
                let half_sum_scaled = sum.half();
                let mut d = d_norm * sum - half_sum_scaled;
                if sign_flip {
                    d = Fix128::ZERO - d;
                }
                return d * dx;
            }
        }
    }

    // Intermediate regime: bisection (fewer iterations than before since
    // Fix128 precision is already saturated in ~20 iterations).
    let mut lo = Fix128::ZERO - dx.double();
    let mut hi = dx.double();
    let target_vol = f * dx * dx * dx;
    for _ in 0..20 {
        let mid = (lo + hi).half();
        let v = truncated_cube_volume(normal, mid, dx);
        if v < target_vol {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi).half()
}

/// Volume of the region `{x ∈ cube : n̂·x ≤ d}` inside a cube of side `dx`
/// centred at the origin. Computed by trilinear sub-sampling (Monte-Carlo-
/// free deterministic approximation via 4×4×4 grid).
#[must_use]
#[allow(dead_code)] // Reserved PLIC helper — awaiting integration
pub(crate) fn truncated_cube_volume(normal: Vec3Fix, d: Fix128, dx: Fix128) -> Fix128 {
    let half = dx.half();
    let subdiv = 4u32;
    let step = dx / Fix128::from_int(subdiv as i64);
    let sub_vol = step * step * step;
    let mut vol = Fix128::ZERO;
    let neg_half = Fix128::ZERO - half;
    for kk in 0..subdiv {
        for jj in 0..subdiv {
            for ii in 0..subdiv {
                let x = neg_half + step * (Fix128::from_int(ii as i64) + Fix128::from_ratio(1, 2));
                let y = neg_half + step * (Fix128::from_int(jj as i64) + Fix128::from_ratio(1, 2));
                let z = neg_half + step * (Fix128::from_int(kk as i64) + Fix128::from_ratio(1, 2));
                if normal.x * x + normal.y * y + normal.z * z <= d {
                    vol = vol + sub_vol;
                }
            }
        }
    }
    vol
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiphase::initialize_level_set_sphere;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn fast_sweeping_preserves_sign() {
        let mut g = Grid3d::new(7, 7, 7, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(2),
        );
        fast_sweeping_reinit(&mut g, 2);
        // Inside cell (3,3,3) should stay negative (interior of sphere)
        assert!(g.get(3, 3, 3) < Fix128::ZERO);
        // Corner should stay positive
        assert!(g.get(0, 0, 0) > Fix128::ZERO);
    }

    #[test]
    fn fast_sweeping_smoke_bounded() {
        let mut g = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::from_int(100));
        // Uniformly high level set → after reinit still positive
        fast_sweeping_reinit(&mut g, 1);
        assert!(g.get(2, 2, 2) > Fix128::ZERO);
    }

    #[test]
    fn plic_normal_out_of_range_returns_default() {
        let g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::ZERO);
        assert_eq!(plic_normal(&g, 0, 0, 0), Vec3Fix::default());
    }

    #[test]
    fn plic_normal_uniform_field_is_zero() {
        let g = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::from_ratio(5, 10));
        // f is uniform → grad = 0 → normal = default
        assert_eq!(plic_normal(&g, 2, 2, 2), Vec3Fix::default());
    }

    #[test]
    fn plic_normal_gradient_direction() {
        // f increasing in +X → gradient points +X → normal (antiparallel) is -X
        let mut g = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::ZERO);
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    g.set(
                        i,
                        j,
                        k,
                        Fix128::from_int(i as i64) * Fix128::from_ratio(1, 10),
                    );
                }
            }
        }
        let n = plic_normal(&g, 2, 2, 2);
        // Normal should be primarily -X
        assert!(n.x < Fix128::ZERO);
        assert!(n.y.abs() < Fix128::from_ratio(1, 100));
        assert!(n.z.abs() < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn plic_offset_empty_cell() {
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let d = plic_plane_offset(n, Fix128::ZERO, Fix128::ONE);
        assert!(d < Fix128::ZERO); // plane pushed out
    }

    #[test]
    fn plic_offset_full_cell() {
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let d = plic_plane_offset(n, Fix128::ONE, Fix128::ONE);
        assert!(d > Fix128::ZERO); // plane pushed out on positive side
    }

    #[test]
    fn plic_offset_half_full() {
        // f = 0.5, normal +X, dx = 1 → plane near x = 0 (centre).
        // 4×4×4 sub-sampling gives step 0.25, so the bisection converges
        // to the nearest sub-cell boundary rather than the exact 0.
        // Allow ±0.2 tolerance to accept either -0.125 or +0.125 branch.
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let d = plic_plane_offset(n, Fix128::from_ratio(5, 10), Fix128::ONE);
        assert!(approx_eq(d, Fix128::ZERO, Fix128::from_ratio(2, 10)));
    }

    #[test]
    fn truncated_cube_volume_all_below() {
        // d = +∞ → whole cube captured
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let v = truncated_cube_volume(n, Fix128::from_int(1000), Fix128::ONE);
        assert!(approx_eq(v, Fix128::ONE, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn truncated_cube_volume_none_below() {
        // d = -∞ → nothing captured
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let v = truncated_cube_volume(n, Fix128::from_int(-1000), Fix128::ONE);
        assert_eq!(v, Fix128::ZERO);
    }

    #[test]
    fn truncated_cube_volume_half() {
        // d = 0 (plane through centre) → half cube
        let n = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let v = truncated_cube_volume(n, Fix128::ZERO, Fix128::ONE);
        assert!(approx_eq(
            v,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(1, 10)
        ));
    }
}
