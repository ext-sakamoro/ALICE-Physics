//! Multi-Phase Flow (VOF + Level Set)
//!
//! Phase F2 of the ALICE-Physics completeness project. Tracks the interface
//! between two immiscible fluids (water/air, oil/water, molten polymer/gas)
//! using both **Volume of Fluid** (VOF) and **Level Set** representations.
//!
//! # VOF
//!
//! Store a scalar `f ∈ [0, 1]` per cell — the fraction of the cell occupied
//! by fluid A. Interface cells have `0 < f < 1`. Advection with upwind
//! discretisation preserves mass (bounded to [0, 1]).
//!
//! # Level Set
//!
//! Store a signed distance `φ` per cell — negative inside fluid A, positive
//! outside, zero on the interface. Naturally captures curvature via
//! `κ = ∇·(∇φ / |∇φ|)`, needed for surface tension in `surface_tension_csf.rs`.
//!
//! Periodic **reinitialisation** restores the `|∇φ| = 1` invariant that
//! advection destroys.
//!
//! # Implementation scope
//!
//! Fully 3D grids with explicit `Vec<Fix128>` storage. Simple upwind advection
//! (first-order); higher-order MUSCL-Hancock deferred to Phase G6 sharp
//! interface capture.
//!
//! # References
//!
//! - Hirt & Nichols, "Volume of fluid (VOF) method for the dynamics of free
//!   boundaries", J. Comp. Phys. 39 (1981) — original VOF.
//! - Osher & Sethian, "Fronts propagating with curvature-dependent speed:
//!   Algorithms based on Hamilton-Jacobi formulations", J. Comp. Phys. 79
//!   (1988) — Level Set.
//! - Sussman, Smereka, Osher, "A Level Set approach for computing solutions
//!   to incompressible two-phase flow", J. Comp. Phys. 114 (1994) —
//!   reinitialisation.
//!
//! # Integration status
//!
//! `Grid3d`, `trilinear_range`, `trilinear_sample` and `curvature_at` are
//! consumed by `cfd_solver.rs` (level-set and temperature advection) and
//! `surface_tension_csf.rs`. [`initialize_level_set_sphere`] is how a caller
//! seeds `CfdSolver::level_set` (`examples/vof_level_set_transport.rs`). The
//! pseudo-time `reinitialize_level_set` (crate-internal) is the second
//! reinitialisation scheme of the solver, selected per step through
//! `cfd_solver::StepOptions::with_level_set_reinit`
//! (`LevelSetReinit::PseudoTime`); the default remains
//! `interface_capture::fast_sweeping_reinit`. The two uniform-velocity VOF
//! schemes and `total_volume_vof` are reached through [`advect_vof_rigid`],
//! the rigid-convection transport of a VOF field (translation test of a
//! fraction field, the same example).

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Grid
// ============================================================================

/// Regular 3D grid with `Vec<Fix128>` storage in `[x + nx·(y + ny·z)]` order.
#[derive(Clone, Debug)]
pub struct Grid3d {
    /// Number of cells in X.
    pub nx: usize,
    /// Number of cells in Y.
    pub ny: usize,
    /// Number of cells in Z.
    pub nz: usize,
    /// Cell spacing (m).
    pub dx: Fix128,
    /// Cell values.
    pub data: Vec<Fix128>,
}

impl Grid3d {
    /// Create an empty grid initialised to `value`.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Fix128, value: Fix128) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            data: vec![value; nx * ny * nz],
        }
    }

    /// Linear index into `data`.
    #[inline]
    #[must_use]
    pub const fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Read value at (i, j, k). Returns `Fix128::ZERO` for out-of-range.
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.data[self.idx(i, j, k)]
    }

    /// Write value at (i, j, k). No-op if out of range.
    pub fn set(&mut self, i: usize, j: usize, k: usize, v: Fix128) {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return;
        }
        let ix = self.idx(i, j, k);
        self.data[ix] = v;
    }

    /// Total number of cells.
    #[inline]
    #[must_use]
    pub fn total(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

// ============================================================================
// VOF advection
// ============================================================================

/// Advect a scalar field using first-order upwind for a time step `dt` with
/// constant velocity `(ux, uy, uz)` (m/s). Values are clamped to `[0, 1]`
/// after update (VOF hygiene).
///
// LIMITATION(COV-MULTIPHASE-001): Simplification: assumes uniform velocity across the grid (rigid-body convection).
/// Simplification: assumes uniform velocity across the grid (rigid-body
/// convection). For a spatially varying field pair this routine with an
/// `EulerianGrid` velocity field from `eulerian_grid.rs` (Phase F4).
pub(crate) fn advect_vof_uniform(
    field: &mut Grid3d,
    ux_m_per_s: Fix128,
    uy_m_per_s: Fix128,
    uz_m_per_s: Fix128,
    dt_s: Fix128,
) {
    let mut next = field.data.clone();
    let sign_x = ux_m_per_s > Fix128::ZERO;
    let sign_y = uy_m_per_s > Fix128::ZERO;
    let sign_z = uz_m_per_s > Fix128::ZERO;
    let abs_ux = ux_m_per_s.abs();
    let abs_uy = uy_m_per_s.abs();
    let abs_uz = uz_m_per_s.abs();
    if field.dx.is_zero() {
        return;
    }
    let cx = abs_ux * dt_s / field.dx;
    let cy = abs_uy * dt_s / field.dx;
    let cz = abs_uz * dt_s / field.dx;

    for k in 0..field.nz {
        for j in 0..field.ny {
            for i in 0..field.nx {
                let f = field.get(i, j, k);
                let ix = field.idx(i, j, k);

                // Upwind samples; use zero at boundaries (empty inflow)
                let fu_x = if sign_x && i > 0 {
                    field.get(i - 1, j, k)
                } else if !sign_x && i + 1 < field.nx {
                    field.get(i + 1, j, k)
                } else {
                    Fix128::ZERO
                };
                let fu_y = if sign_y && j > 0 {
                    field.get(i, j - 1, k)
                } else if !sign_y && j + 1 < field.ny {
                    field.get(i, j + 1, k)
                } else {
                    Fix128::ZERO
                };
                let fu_z = if sign_z && k > 0 {
                    field.get(i, j, k - 1)
                } else if !sign_z && k + 1 < field.nz {
                    field.get(i, j, k + 1)
                } else {
                    Fix128::ZERO
                };
                let new = f - cx * (f - fu_x) - cy * (f - fu_y) - cz * (f - fu_z);
                let clamped = if new < Fix128::ZERO {
                    Fix128::ZERO
                } else if new > Fix128::ONE {
                    Fix128::ONE
                } else {
                    new
                };
                next[ix] = clamped;
            }
        }
    }
    field.data = next;
}

/// Return `(min, max)` of the 8 corner values surrounding `(cx, cy, cz)`.
///
/// Used by MacCormack to clamp corrector results into the pre-advection
/// local range (Fedkiw's monotonicity guard).
///
/// A grid with a zero extent has no corner values: the range is
/// `(ZERO, ZERO)` (AUD-A-S2W3-007).
#[must_use]
pub fn trilinear_range(field: &Grid3d, cx: Fix128, cy: Fix128, cz: Fix128) -> (Fix128, Fix128) {
    if field.nx == 0 || field.ny == 0 || field.nz == 0 {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let clamp_neg = |v: Fix128| if v.is_negative() { Fix128::ZERO } else { v };
    let cxx = clamp_neg(cx);
    let cyy = clamp_neg(cy);
    let czz = clamp_neg(cz);
    let ix = cxx.hi as usize;
    let iy = cyy.hi as usize;
    let iz = czz.hi as usize;
    let i0 = ix.min(field.nx - 1);
    let i1 = (ix + 1).min(field.nx - 1);
    let j0 = iy.min(field.ny - 1);
    let j1 = (iy + 1).min(field.ny - 1);
    let k0 = iz.min(field.nz - 1);
    let k1 = (iz + 1).min(field.nz - 1);
    let corners = [
        field.get(i0, j0, k0),
        field.get(i1, j0, k0),
        field.get(i0, j1, k0),
        field.get(i1, j1, k0),
        field.get(i0, j0, k1),
        field.get(i1, j0, k1),
        field.get(i0, j1, k1),
        field.get(i1, j1, k1),
    ];
    let mut lo = corners[0];
    let mut hi = corners[0];
    for &c in &corners[1..] {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    (lo, hi)
}

/// Trilinear-interpolate a scalar `Grid3d` at continuous cell coordinates
/// `(cx, cy, cz)`. Coordinates outside the grid clamp to boundary values.
///
/// A grid with a zero extent has no value to sample: the result is `ZERO`
/// (AUD-A-S2W3-007).
#[must_use]
pub fn trilinear_sample(field: &Grid3d, cx: Fix128, cy: Fix128, cz: Fix128) -> Fix128 {
    if field.nx == 0 || field.ny == 0 || field.nz == 0 {
        return Fix128::ZERO;
    }
    let clamp_neg = |v: Fix128| if v.is_negative() { Fix128::ZERO } else { v };
    let cxx = clamp_neg(cx);
    let cyy = clamp_neg(cy);
    let czz = clamp_neg(cz);
    let ix = cxx.hi as usize;
    let iy = cyy.hi as usize;
    let iz = czz.hi as usize;
    let u = Fix128 { hi: 0, lo: cxx.lo };
    let v = Fix128 { hi: 0, lo: cyy.lo };
    let w = Fix128 { hi: 0, lo: czz.lo };
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let i0 = ix.min(field.nx - 1);
    let i1 = (ix + 1).min(field.nx - 1);
    let j0 = iy.min(field.ny - 1);
    let j1 = (iy + 1).min(field.ny - 1);
    let k0 = iz.min(field.nz - 1);
    let k1 = (iz + 1).min(field.nz - 1);
    let c00 = field.get(i0, j0, k0) * om_u + field.get(i1, j0, k0) * u;
    let c10 = field.get(i0, j1, k0) * om_u + field.get(i1, j1, k0) * u;
    let c01 = field.get(i0, j0, k1) * om_u + field.get(i1, j0, k1) * u;
    let c11 = field.get(i0, j1, k1) * om_u + field.get(i1, j1, k1) * u;
    let c0 = c00 * om_v + c10 * v;
    let c1 = c01 * om_v + c11 * v;
    c0 * om_w + c1 * w
}

/// Semi-Lagrangian VOF advection (Session 3 I3 upgrade).
///
/// For each cell, trace back along the constant velocity by `dt`, sample
/// the previous field trilinearly, and write to the new field. This is
/// **unconditionally stable** (no CFL restriction) unlike first-order
/// upwind — you can take arbitrarily large `dt`, though very large steps
/// dilute detail.
pub(crate) fn advect_vof_uniform_semi_lagrangian(
    field: &mut Grid3d,
    ux_m_per_s: Fix128,
    uy_m_per_s: Fix128,
    uz_m_per_s: Fix128,
    dt_s: Fix128,
) {
    if field.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / field.dx;
    // Displacement in cell units
    let dx_cells = ux_m_per_s * dt_s * inv_dx;
    let dy_cells = uy_m_per_s * dt_s * inv_dx;
    let dz_cells = uz_m_per_s * dt_s * inv_dx;

    let old = field.data.clone();
    let old_grid = Grid3d {
        nx: field.nx,
        ny: field.ny,
        nz: field.nz,
        dx: field.dx,
        data: old,
    };
    for k in 0..field.nz {
        for j in 0..field.ny {
            for i in 0..field.nx {
                // Back-trace: sample position - u·dt (in cell units)
                let cx = Fix128::from_int(i as i64) - dx_cells;
                let cy = Fix128::from_int(j as i64) - dy_cells;
                let cz = Fix128::from_int(k as i64) - dz_cells;
                let val = trilinear_sample(&old_grid, cx, cy, cz);
                let clamped = if val < Fix128::ZERO {
                    Fix128::ZERO
                } else if val > Fix128::ONE {
                    Fix128::ONE
                } else {
                    val
                };
                let ix = field.idx(i, j, k);
                field.data[ix] = clamped;
            }
        }
    }
}

/// Total volume of fluid A in a VOF field: `V = Σ f · dx³` (crate-internal).
#[must_use]
pub(crate) fn total_volume_vof(field: &Grid3d) -> Fix128 {
    let cell_vol = field.dx * field.dx * field.dx;
    let sum: Fix128 = field.data.iter().copied().fold(Fix128::ZERO, |a, b| a + b);
    sum * cell_vol
}

/// Which discretisation [`advect_vof_rigid`] transports the fraction with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VofScheme {
    /// First-order upwind: `f_i ← f_i − c (f_i − f_up)` per axis with the
    /// Courant number `c = |u| dt / dx`, the upwind neighbour `f_up` taken
    /// as `0` outside the grid (empty inflow), then clamped to `[0, 1]`.
    /// Exact at `c = 1` (the field translates by one cell), diffusive below
    /// it, and only the clamp bounds it above it.
    Upwind,
    /// Semi-Lagrangian back-trace with trilinear sampling of the previous
    /// field, the sample point clamped to the grid (the boundary value is
    /// extended outside it), then clamped to `[0, 1]`. Exact at integer
    /// displacements `u dt / dx`, unconditionally stable, diffusive at
    /// fractional ones.
    SemiLagrangian,
}

/// Rigid convection of a VOF fraction field: every cell is carried by the
/// same velocity `velocity_m_per_s` (m/s) for `dt_s` (s) with `scheme`, and
/// the fluid volume `Σ f · dx³` after the step is returned.
///
/// This is the translation test of a fraction field: a slab of fraction `1`
/// carried by `u` for `dt = k dx / u` lands `k` cells over with its profile
/// intact under either scheme, and its volume is unchanged as long as it
/// stays inside the grid (a cell carried past the last cell is lost, and the
/// volume drops by that cell). A spatially varying velocity is the level set
/// path of `CfdSolver` (semi-Lagrangian on the cell-centred MAC velocity),
/// not this routine.
///
/// # Degenerate input
///
/// A zero spacing returns with the field **untouched** (no clamp is applied
/// either) and a zero volume; a zero velocity or a zero `dt` leaves the field
/// bit-identical under both schemes; a grid with a zero extent has no cells
/// and returns zero. A velocity far above `dx / dt` does not panic: the
/// upwind scheme reduces to the sign rule `f ← 1` where `f_up > f`, `0`
/// where `f_up < f` (the clamp), and the semi-Lagrangian one samples the
/// boundary cell the back-trace is clamped to.
#[must_use]
pub fn advect_vof_rigid(
    field: &mut Grid3d,
    scheme: VofScheme,
    velocity_m_per_s: Vec3Fix,
    dt_s: Fix128,
) -> Fix128 {
    let Vec3Fix { x, y, z } = velocity_m_per_s;
    match scheme {
        VofScheme::Upwind => advect_vof_uniform(field, x, y, z, dt_s),
        VofScheme::SemiLagrangian => advect_vof_uniform_semi_lagrangian(field, x, y, z, dt_s),
    }
    total_volume_vof(field)
}

// ============================================================================
// Level Set
// ============================================================================

/// Initialise a level set field as the signed distance to a sphere centred
/// at `(cx, cy, cz)` with radius `r`. Negative inside, positive outside.
pub fn initialize_level_set_sphere(
    field: &mut Grid3d,
    cx_m: Fix128,
    cy_m: Fix128,
    cz_m: Fix128,
    radius_m: Fix128,
) {
    for k in 0..field.nz {
        for j in 0..field.ny {
            for i in 0..field.nx {
                let x = Fix128::from_int(i as i64) * field.dx;
                let y = Fix128::from_int(j as i64) * field.dx;
                let z = Fix128::from_int(k as i64) * field.dx;
                let dx = x - cx_m;
                let dy = y - cy_m;
                let dz = z - cz_m;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                field.set(i, j, k, dist - radius_m);
            }
        }
    }
}

/// Level set reinitialisation by `iterations` explicit pseudo-time steps of
/// the Sussman–Smereka–Osher equation `φ_τ = sgn(φ) (1 − |∇φ|)` with
/// `Δτ = dx / 2` and central differences for `|∇φ|`, the sign taken from the
/// current field (`0` on the zero level, which therefore never moves).
/// Restores `|∇φ| ≈ 1` while preserving the zero level set.
///
/// Only interior cells (`1 ≤ i ≤ nx − 2`, likewise `j`, `k`) are updated;
/// the outermost layer is frozen, so a grid with fewer than three cells on
/// any axis has no interior and is returned untouched. A zero spacing is
/// also a no-op (`Δτ = 0`). A field that is already a signed distance with
/// `|∇φ| = 1` to the bit in the central differences is a fixed point to the
// LIMITATION(COV-MULTIPHASE-013): Adequate for coarse grids (< 64³) and moderate deformation
/// bit. Adequate for coarse grids (< 64³) and moderate deformation; higher
/// accuracy needs the full fast sweeping of
/// `interface_capture::fast_sweeping_reinit`, which is the solver's default.
/// Compiled with the `std` feature, as its consumer `cfd_solver` is.
#[cfg(feature = "std")]
pub(crate) fn reinitialize_level_set(field: &mut Grid3d, iterations: u32) {
    if field.nx < 3 || field.ny < 3 || field.nz < 3 {
        return;
    }
    let dtau = field.dx * Fix128::from_ratio(1, 2);
    for _ in 0..iterations {
        let mut next = field.data.clone();
        for k in 1..field.nz - 1 {
            for j in 1..field.ny - 1 {
                for i in 1..field.nx - 1 {
                    let phi = field.get(i, j, k);
                    let sign = if phi > Fix128::ZERO {
                        Fix128::ONE
                    } else if phi < Fix128::ZERO {
                        Fix128::NEG_ONE
                    } else {
                        Fix128::ZERO
                    };
                    // Approximate |∇φ| via central differences
                    let dpdx =
                        (field.get(i + 1, j, k) - field.get(i - 1, j, k)) / (field.dx + field.dx);
                    let dpdy =
                        (field.get(i, j + 1, k) - field.get(i, j - 1, k)) / (field.dx + field.dx);
                    let dpdz =
                        (field.get(i, j, k + 1) - field.get(i, j, k - 1)) / (field.dx + field.dx);
                    let grad_sq = dpdx * dpdx + dpdy * dpdy + dpdz * dpdz;
                    let grad_mag = grad_sq.sqrt();
                    let ix = field.idx(i, j, k);
                    // dφ/dτ = sign(φ_0) · (1 − |∇φ|)
                    next[ix] = phi + dtau * sign * (Fix128::ONE - grad_mag);
                }
            }
        }
        field.data = next;
    }
}

/// Estimate interface curvature at cell (i, j, k) via central differences.
// LIMITATION(COV-MULTIPHASE-028): scaled by `1/|∇φ|` (assumed close to 1 for a well-reinitialised field)
/// Returns `κ = (∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²)` scaled by `1/|∇φ|` (assumed
/// close to 1 for a well-reinitialised field).
#[must_use]
pub fn curvature_at(field: &Grid3d, i: usize, j: usize, k: usize) -> Fix128 {
    if i == 0 || j == 0 || k == 0 || i + 1 >= field.nx || j + 1 >= field.ny || k + 1 >= field.nz {
        return Fix128::ZERO;
    }
    let phi = field.get(i, j, k);
    let dx2 = field.dx * field.dx;
    if dx2.is_zero() {
        return Fix128::ZERO;
    }
    let d2x = field.get(i + 1, j, k) - phi.double() + field.get(i - 1, j, k);
    let d2y = field.get(i, j + 1, k) - phi.double() + field.get(i, j - 1, k);
    let d2z = field.get(i, j, k + 1) - phi.double() + field.get(i, j, k - 1);
    (d2x + d2y + d2z) / dx2
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_new_all_default_value() {
        let g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::from_int(5));
        assert_eq!(g.total(), 27);
        assert_eq!(g.get(0, 0, 0), Fix128::from_int(5));
        assert_eq!(g.get(2, 2, 2), Fix128::from_int(5));
    }

    #[test]
    fn grid_out_of_range_returns_zero() {
        let g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::from_int(5));
        assert_eq!(g.get(5, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn grid_set_get_roundtrip() {
        let mut g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::ZERO);
        g.set(1, 1, 1, Fix128::from_int(42));
        assert_eq!(g.get(1, 1, 1), Fix128::from_int(42));
    }

    #[test]
    fn vof_uniform_advection_shifts_field() {
        // 4×1×1 grid, initialise f = [1, 0, 0, 0]
        let mut g = Grid3d::new(4, 1, 1, Fix128::ONE, Fix128::ZERO);
        g.set(0, 0, 0, Fix128::ONE);
        // Advect with ux = 1 m/s for dt = 0.5 → half-cell shift
        advect_vof_uniform(
            &mut g,
            Fix128::ONE,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::from_ratio(5, 10),
        );
        // Cell 0 loses half, cell 1 gains half
        assert!(g.get(0, 0, 0) < Fix128::ONE);
        assert!(g.get(1, 0, 0) > Fix128::ZERO);
    }

    #[test]
    fn vof_advection_conserves_volume_bounded_step() {
        // Interior conservation for a full cell shift: total volume ≈ preserved
        let mut g = Grid3d::new(6, 1, 1, Fix128::ONE, Fix128::ZERO);
        g.set(1, 0, 0, Fix128::ONE);
        g.set(2, 0, 0, Fix128::ONE);
        let vol_initial = total_volume_vof(&g);
        // Small step so nothing exits domain
        for _ in 0..3 {
            advect_vof_uniform(
                &mut g,
                Fix128::from_ratio(1, 10),
                Fix128::ZERO,
                Fix128::ZERO,
                Fix128::ONE,
            );
        }
        let vol_after = total_volume_vof(&g);
        // First-order upwind is diffusive but conserves mass in interior;
        // allow ~5% due to boundary treatment.
        let ratio = vol_after / vol_initial;
        assert!(ratio > Fix128::from_ratio(90, 100));
        assert!(ratio < Fix128::from_ratio(110, 100));
    }

    #[test]
    fn vof_bounded_to_zero_one() {
        let mut g = Grid3d::new(3, 1, 1, Fix128::ONE, Fix128::from_int(5)); // > 1
        advect_vof_uniform(
            &mut g,
            Fix128::from_ratio(1, 10),
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ONE,
        );
        for i in 0..3 {
            assert!(g.get(i, 0, 0) <= Fix128::ONE);
            assert!(g.get(i, 0, 0) >= Fix128::ZERO);
        }
    }

    #[test]
    fn level_set_sphere_zero_on_surface() {
        let mut g = Grid3d::new(11, 11, 11, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(3),
        );
        // Inside cell (5,5,5): distance 0 - 3 = -3
        assert_eq!(g.get(5, 5, 5), Fix128::from_int(-3));
        // Point at (8,5,5): distance = 3 - 3 = 0
        assert_eq!(g.get(8, 5, 5), Fix128::ZERO);
        // Point at (10,5,5): distance = 5 - 3 = 2
        assert_eq!(g.get(10, 5, 5), Fix128::from_int(2));
    }

    #[test]
    fn total_volume_vof_uniform_fill() {
        let g = Grid3d::new(4, 4, 4, Fix128::from_int(2), Fix128::ONE);
        // 64 cells × 2^3 = 8 mm³ each → 512
        assert_eq!(total_volume_vof(&g), Fix128::from_int(512));
    }

    #[cfg(feature = "std")]
    #[test]
    fn reinitialize_level_set_smoke() {
        let mut g = Grid3d::new(7, 7, 7, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(2),
        );
        // Run reinitialisation — should not crash and not blow up
        reinitialize_level_set(&mut g, 3);
        // Zero level should still be preserved at (5, 3, 3) approximately
        let phi_surface = g.get(5, 3, 3);
        assert!(phi_surface.abs() < Fix128::from_int(2));
    }

    /// Oracle for the interior guard: a grid with fewer than three cells on
    /// any axis has no interior and is returned untouched, the zero-extent
    /// grids included (without the guard `1..0 − 1` wraps and the first
    /// interior write indexes an empty `Vec`).
    #[cfg(feature = "std")]
    #[test]
    fn reinitialize_without_an_interior_is_untouched_and_does_not_panic() {
        for (nx, ny, nz) in [
            (0, 0, 0),
            (0, 3, 3),
            (3, 0, 3),
            (3, 3, 0),
            (2, 3, 3),
            (3, 2, 3),
            (3, 3, 2),
            (1, 1, 1),
        ] {
            let mut g = Grid3d::new(nx, ny, nz, Fix128::ONE, Fix128::from_int(5));
            let before = g.data.clone();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reinitialize_level_set(&mut g, 3);
            }));
            assert!(res.is_ok(), "{nx}x{ny}x{nz} panicked");
            assert_eq!(g.data, before, "{nx}x{ny}x{nz}");
        }
    }

    #[test]
    fn curvature_zero_at_flat_interface() {
        // Level set = constant → all derivatives zero → curvature 0
        let g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::from_int(5));
        assert_eq!(curvature_at(&g, 1, 1, 1), Fix128::ZERO);
    }

    #[test]
    fn semi_lagrangian_uniform_shifts_field() {
        // Unit velocity, dt=1 → shift by exactly one cell (semi-Lagrangian
        // is exact at integer displacements).
        let mut g = Grid3d::new(6, 1, 1, Fix128::ONE, Fix128::ZERO);
        g.set(1, 0, 0, Fix128::ONE);
        advect_vof_uniform_semi_lagrangian(
            &mut g,
            Fix128::ONE,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ONE,
        );
        assert!(g.get(2, 0, 0) > Fix128::from_ratio(90, 100));
    }

    #[test]
    fn trilinear_sample_at_grid_point_returns_value() {
        let mut g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::ZERO);
        g.set(1, 1, 1, Fix128::from_int(5));
        let v = trilinear_sample(
            &g,
            Fix128::from_int(1),
            Fix128::from_int(1),
            Fix128::from_int(1),
        );
        assert_eq!(v, Fix128::from_int(5));
    }

    #[test]
    fn semi_lagrangian_zero_velocity_preserves_field() {
        let mut g = Grid3d::new(4, 4, 4, Fix128::ONE, Fix128::from_ratio(3, 10));
        let before = g.data.clone();
        advect_vof_uniform_semi_lagrangian(
            &mut g,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ONE,
        );
        assert_eq!(g.data, before);
    }

    #[test]
    fn curvature_bounded_at_edges() {
        // Level set with linear gradient still has zero curvature
        let mut g = Grid3d::new(5, 1, 1, Fix128::ONE, Fix128::ZERO);
        for i in 0..5 {
            g.set(i, 0, 0, Fix128::from_int(i as i64));
        }
        // Only interior cells checked; boundaries return 0
        assert_eq!(curvature_at(&g, 0, 0, 0), Fix128::ZERO);
    }
}
