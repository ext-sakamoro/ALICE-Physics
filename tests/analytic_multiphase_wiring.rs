//! Oracles for the `multiphase` items reached through production entry
//! points: `advect_vof_rigid` (the two uniform-velocity VOF schemes and the
//! volume `Σ f dx³`), `initialize_level_set_sphere` (the seed of
//! `CfdSolver::level_set`) and the pseudo-time `reinitialize_level_set`
//! selected through `StepOptions::with_level_set_reinit`
//! (`LevelSetReinit::PseudoTime`) and run by `CfdSolver::step_with_options`.
//!
//! # Closed forms
//!
//! * **Upwind at Courant number 1** (`u dt = dx`): `f_i ← f_i − 1·(f_i −
//!   f_{i−1}) = f_{i−1}`, the field translates by exactly one cell per step
//!   on every axis and `Σ f dx³` is unchanged while the slab is inside.
//! * **Upwind at Courant number 1/2**: `f_i ← (f_i + f_{i−1}) / 2`, so a unit
//!   cell spreads into the binomial profile `2^{−n} C(n, m)`.
//! * **Semi-Lagrangian at an integer displacement** `k = u dt / dx`: the
//!   back-traced point is a cell centre and the trilinear weights are `1`
//!   and `0`, so `f_i ← f_{i−k}` to the bit. At half a cell the weights are
//!   `1/2, 1/2` and `f_i ← (f_i + f_{i−1}) / 2`.
//! * **Above Courant number 1 the schemes differ**: for `u dt = 2 dx` the
//!   upwind update of a unit cell is `[0, −1 → 0, 2 → 1, 0]` (the clamp) and
//!   the semi-Lagrangian one is the exact two-cell shift.
//! * **Volume**: `Σ f · dx³` with `dx = 1/2` on `2×3×4` cells of `1/4` is
//!   `24 · 1/4 · 1/8 = 3/4`; with `dx = 3` it is `162`.
//! * **Sphere seed**: `φ = dx √(Δi² + Δj² + Δk²) − r`, exact (integer square
//!   root of a perfect square) wherever `Δi² + Δj² + Δk²` is a perfect
//!   square; the cells with `φ < 0` are the lattice points inside the sphere,
//!   counted independently.
//! * **Pseudo-time reinitialisation on a plane of slope 2** along one axis
//!   with `Δτ = dx/2`: the central difference is exactly `2`, so one
//!   iteration moves every interior cell by `−sgn(φ) · dx/2` and leaves the
//!   zero level and the frozen outer layer alone; the second iteration is
//!   worked out cell by cell in the test. A plane of slope 1 is a fixed point
//!   to the bit.
//! * **The default is unchanged**: `step_with_options` with
//!   `StepOptions::new` and with an explicit `FastSweeping { sweeps: 2 }` is
//!   bit-identical to `step` on a level-set solver, and `PseudoTime` is
//!   observable (differs).
//!
//! # Degenerate input
//!
//! Zero spacing (field untouched, no clamp, zero volume), zero velocity and
//! zero `dt` (bit-identical), zero-extent grids (no cells, zero volume), a
//! velocity far above `dx / dt` (the sign rule and the clamped boundary
//! sample), a zero reinitialisation count (refused, solver untouched, on
//! `step_with_options` and `step_rans`, with and without a level set), a
//! level set without an interior or with zero spacing (untouched), and a
//! sphere seeded outside the grid or with a non-positive radius.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::{
    CfdSolver, LevelSetReinit, PressureSolver, RansState, StepError, StepOptions, TurbulenceModel,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{advect_vof_rigid, initialize_level_set_sphere, Grid3d, VofScheme};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

const AXES: [usize; 3] = [0, 1, 2];

/// A velocity of `u` along `axis`.
fn along(axis: usize, u: Fix128) -> Vec3Fix {
    match axis {
        0 => Vec3Fix::new(u, Fix128::ZERO, Fix128::ZERO),
        1 => Vec3Fix::new(Fix128::ZERO, u, Fix128::ZERO),
        _ => Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, u),
    }
}

/// A line of cells along `axis` (the other extents are 1), holding `values`
/// in order. The storage order of a line is the cell order on every axis, so
/// `g.data` compares directly with a `Vec` of the expected values.
fn line(axis: usize, values: &[Fix128], dx: Fix128) -> Grid3d {
    let n = values.len();
    let (nx, ny, nz) = match axis {
        0 => (n, 1, 1),
        1 => (1, n, 1),
        _ => (1, 1, n),
    };
    Grid3d {
        nx,
        ny,
        nz,
        dx,
        data: values.to_vec(),
    }
}

/// `n` cells of which `[lo, hi)` hold fraction 1.
fn slab(n: usize, lo: usize, hi: usize) -> Vec<Fix128> {
    (0..n)
        .map(|i| {
            if (lo..hi).contains(&i) {
                Fix128::ONE
            } else {
                Fix128::ZERO
            }
        })
        .collect()
}

fn ints(v: &[i64]) -> Vec<Fix128> {
    v.iter().map(|&x| int(x)).collect()
}

fn ratios(v: &[(i64, i64)], d: i64) -> Vec<Fix128> {
    v.iter().map(|&(n, k)| q(n, k * d)).collect()
}

// ============================================================================
// VOF: advect_vof_rigid
// ============================================================================

#[test]
fn upwind_courant_one_translates_the_slab_by_one_cell_per_step_on_every_axis() {
    let dx = q(1, 4);
    let u = q(1, 2);
    let dt = q(1, 2); // u dt / dx = 1
    let cell = dx * dx * dx;
    for axis in AXES {
        let mut f = line(axis, &slab(8, 2, 4), dx);
        for step in 1..=3 {
            let volume = advect_vof_rigid(&mut f, VofScheme::Upwind, along(axis, u), dt);
            assert_eq!(
                f.data,
                slab(8, 2 + step, 4 + step),
                "axis {axis}, step {step}"
            );
            assert_eq!(volume, int(2) * cell, "axis {axis}, step {step}: volume");
        }
    }
}

#[test]
fn upwind_half_courant_gives_the_binomial_profile() {
    // dx = 1, u = 1, dt = 1/2: f_i ← (f_i + f_{i-1}) / 2
    let mut f = line(0, &ints(&[1, 0, 0, 0, 0, 0]), Fix128::ONE);
    let expected = [
        ratios(&[(1, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1)], 2),
        ratios(&[(1, 1), (2, 1), (1, 1), (0, 1), (0, 1), (0, 1)], 4),
        ratios(&[(1, 1), (3, 1), (3, 1), (1, 1), (0, 1), (0, 1)], 8),
    ];
    for (step, want) in expected.iter().enumerate() {
        let volume = advect_vof_rigid(&mut f, VofScheme::Upwind, along(0, Fix128::ONE), q(1, 2));
        assert_eq!(&f.data, want, "step {}", step + 1);
        assert_eq!(volume, Fix128::ONE, "step {}: volume", step + 1);
    }
}

#[test]
fn upwind_negative_velocity_takes_the_upwind_neighbour_from_the_high_side() {
    for axis in AXES {
        let mut f = line(axis, &ints(&[0, 0, 1, 0]), Fix128::ONE);
        let volume = advect_vof_rigid(&mut f, VofScheme::Upwind, along(axis, int(-1)), Fix128::ONE);
        assert_eq!(f.data, ints(&[0, 1, 0, 0]), "axis {axis}");
        assert_eq!(volume, Fix128::ONE);
    }
}

#[test]
fn a_cell_carried_past_the_last_cell_is_lost_and_the_volume_drops_by_it() {
    let dx = q(1, 2);
    let cell = dx * dx * dx;
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        let mut f = line(0, &ints(&[0, 0, 1, 1]), dx);
        // u dt / dx = 1
        let v1 = advect_vof_rigid(&mut f, scheme, along(0, int(2)), q(1, 4));
        assert_eq!(f.data, ints(&[0, 0, 0, 1]), "{scheme:?}: one cell left");
        assert_eq!(v1, cell, "{scheme:?}: one cell of volume left");
        let v2 = advect_vof_rigid(&mut f, scheme, along(0, int(2)), q(1, 4));
        assert_eq!(f.data, ints(&[0, 0, 0, 0]), "{scheme:?}: empty");
        assert_eq!(v2, Fix128::ZERO, "{scheme:?}: no volume left");
    }
}

#[test]
fn semi_lagrangian_integer_displacement_translates_exactly_on_every_axis() {
    // u dt / dx = 2 cells per step
    let dx = q(1, 4);
    let u = int(2);
    let dt = q(1, 4);
    let cell = dx * dx * dx;
    for axis in AXES {
        let mut f = line(axis, &slab(10, 1, 3), dx);
        for step in 1..=3 {
            let volume = advect_vof_rigid(&mut f, VofScheme::SemiLagrangian, along(axis, u), dt);
            assert_eq!(
                f.data,
                slab(10, 1 + 2 * step, 3 + 2 * step),
                "axis {axis}, step {step}"
            );
            assert_eq!(volume, int(2) * cell, "axis {axis}, step {step}: volume");
        }
    }
}

#[test]
fn semi_lagrangian_half_cell_displacement_averages_the_two_neighbours() {
    // dx = 1, u = 1, dt = 1/2: f_i ← (f_i + f_{i-1}) / 2 with the boundary
    // cell extended outside the grid (cell 0 samples itself, which is 0 here).
    let mut f = line(0, &ints(&[0, 1, 0, 0]), Fix128::ONE);
    let v1 = advect_vof_rigid(
        &mut f,
        VofScheme::SemiLagrangian,
        along(0, Fix128::ONE),
        q(1, 2),
    );
    assert_eq!(f.data, ratios(&[(0, 1), (1, 1), (1, 1), (0, 1)], 2));
    assert_eq!(v1, Fix128::ONE);
    let v2 = advect_vof_rigid(
        &mut f,
        VofScheme::SemiLagrangian,
        along(0, Fix128::ONE),
        q(1, 2),
    );
    assert_eq!(f.data, ratios(&[(0, 1), (1, 1), (2, 1), (1, 1)], 4));
    assert_eq!(v2, Fix128::ONE);
}

#[test]
fn above_courant_one_the_two_schemes_give_different_documented_answers() {
    // u dt / dx = 2: upwind f_i ← f_i − 2 (f_i − f_{i−1}), clamped;
    // semi-Lagrangian shifts by two cells exactly.
    let mut up = line(0, &ints(&[0, 1, 0, 0, 0, 0]), Fix128::ONE);
    let mut sl = up.clone();
    let vu = advect_vof_rigid(&mut up, VofScheme::Upwind, along(0, int(2)), Fix128::ONE);
    let vs = advect_vof_rigid(
        &mut sl,
        VofScheme::SemiLagrangian,
        along(0, int(2)),
        Fix128::ONE,
    );
    assert_eq!(up.data, ints(&[0, 0, 1, 0, 0, 0]), "upwind: −1 → 0, 2 → 1");
    assert_eq!(
        sl.data,
        ints(&[0, 0, 0, 1, 0, 0]),
        "semi-Lagrangian: exact shift"
    );
    assert_eq!(vu, Fix128::ONE);
    assert_eq!(vs, Fix128::ONE);
}

#[test]
fn the_returned_volume_is_the_fraction_sum_times_dx_cubed() {
    for (dx, want) in [(q(1, 2), q(3, 4)), (int(3), int(162))] {
        for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
            let mut f = Grid3d::new(2, 3, 4, dx, q(1, 4));
            let volume = advect_vof_rigid(&mut f, scheme, Vec3Fix::ZERO, Fix128::ONE);
            assert_eq!(volume, want, "{scheme:?}, dx = {}", dx.to_f64());
        }
    }
}

#[test]
fn zero_velocity_and_zero_dt_leave_the_field_bit_identical_under_both_schemes() {
    let values = [
        q(3, 8),
        Fix128::ZERO,
        Fix128::ONE,
        q(1, 2),
        q(1, 16),
        q(7, 8),
    ];
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        for axis in AXES {
            let mut f = line(axis, &values, q(1, 4));
            let before = f.data.clone();
            let v = advect_vof_rigid(&mut f, scheme, Vec3Fix::ZERO, q(1, 8));
            assert_eq!(f.data, before, "{scheme:?}, axis {axis}: u = 0");
            assert_eq!(
                v,
                q(45, 16) * q(1, 64),
                "{scheme:?}: Σf = 45/16, dx³ = 1/64"
            );
            let v = advect_vof_rigid(&mut f, scheme, along(axis, int(3)), Fix128::ZERO);
            assert_eq!(f.data, before, "{scheme:?}, axis {axis}: dt = 0");
            assert_eq!(v, q(45, 16) * q(1, 64));
        }
    }
}

#[test]
fn zero_spacing_returns_the_field_untouched_without_the_clamp_and_a_zero_volume() {
    // Values outside [0, 1] show whether the clamp ran: it must not.
    let values = [int(5), int(-2), q(1, 2), Fix128::ONE];
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        let mut f = line(0, &values, Fix128::ZERO);
        let v = advect_vof_rigid(&mut f, scheme, along(0, Fix128::ONE), Fix128::ONE);
        assert_eq!(
            f.data,
            values.to_vec(),
            "{scheme:?}: dx = 0 is an early return"
        );
        assert_eq!(v, Fix128::ZERO, "{scheme:?}: dx³ = 0");
    }
}

#[test]
fn zero_extent_grids_have_no_cells_and_return_a_zero_volume() {
    for (nx, ny, nz) in [(0, 0, 0), (0, 3, 3), (3, 0, 3), (3, 3, 0)] {
        for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
            let mut f = Grid3d::new(nx, ny, nz, Fix128::ONE, Fix128::ONE);
            let res = catch_unwind(AssertUnwindSafe(|| {
                advect_vof_rigid(&mut f, scheme, along(0, Fix128::ONE), Fix128::ONE)
            }));
            let v = res.unwrap_or_else(|_| panic!("{scheme:?} panicked on {nx}x{ny}x{nz}"));
            assert_eq!(v, Fix128::ZERO, "{scheme:?} on {nx}x{ny}x{nz}");
            assert!(f.data.is_empty());
        }
    }
}

#[test]
fn a_velocity_far_above_dx_over_dt_follows_the_sign_rule_upwind_and_samples_the_clamped_boundary_semi_lagrangian(
) {
    for huge in [int(1 << 40), int(i64::MAX)] {
        for (u, up_want, sl_want) in [
            // upwind: 1 where f_up > f, 0 where f_up < f, f where equal;
            // semi-Lagrangian: every cell samples the boundary cell on the
            // inflow side (cell 0 for u > 0, cell 3 for u < 0).
            (huge, ints(&[0, 1, 0, 0]), ints(&[1, 1, 1, 1])),
            (
                Fix128::ZERO - huge,
                ints(&[0, 0, 1, 0]),
                ints(&[1, 1, 1, 1]),
            ),
        ] {
            let mut up = line(0, &ints(&[1, 0, 0, 1]), Fix128::ONE);
            let mut sl = up.clone();
            let vu = catch_unwind(AssertUnwindSafe(|| {
                advect_vof_rigid(&mut up, VofScheme::Upwind, along(0, u), Fix128::ONE)
            }))
            .expect("upwind must not panic");
            let vs = catch_unwind(AssertUnwindSafe(|| {
                advect_vof_rigid(&mut sl, VofScheme::SemiLagrangian, along(0, u), Fix128::ONE)
            }))
            .expect("semi-Lagrangian must not panic");
            assert_eq!(up.data, up_want, "upwind, u = {}", u.to_f64());
            assert_eq!(sl.data, sl_want, "semi-Lagrangian, u = {}", u.to_f64());
            assert_eq!(vu, Fix128::ONE);
            assert_eq!(vs, int(4));
        }
    }
}

// ============================================================================
// Level set: initialize_level_set_sphere
// ============================================================================

/// Expected `φ` at offset `(di, dj, dk)` from the centre when
/// `di² + dj² + dk²` is a perfect square, else `None`.
fn exact_sphere_phi(dx: Fix128, r: Fix128, d: (i64, i64, i64)) -> Option<Fix128> {
    let s2 = d.0 * d.0 + d.1 * d.1 + d.2 * d.2;
    let root = (0..=s2).find(|m| m * m == s2)?;
    Some(dx * int(root) - r)
}

#[test]
fn the_sphere_seed_is_the_exact_signed_distance_at_perfect_square_offsets_and_signed_elsewhere() {
    for (dx, r, n, c) in [(Fix128::ONE, int(3), 11, 5), (q(1, 2), q(3, 2), 11, 5)] {
        let mut g = Grid3d::new(n, n, n, dx, int(99));
        let centre = dx * int(c);
        initialize_level_set_sphere(&mut g, centre, centre, centre, r);
        let r_cells2 = 9; // (r / dx)² = 9 in both cases
        let mut inside = 0usize;
        let mut exact_checked = 0usize;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d = (i as i64 - c, j as i64 - c, k as i64 - c);
                    let s2 = d.0 * d.0 + d.1 * d.1 + d.2 * d.2;
                    let phi = g.get(i, j, k);
                    if let Some(want) = exact_sphere_phi(dx, r, d) {
                        assert_eq!(phi, want, "cell {i},{j},{k}");
                        exact_checked += 1;
                    }
                    match s2.cmp(&r_cells2) {
                        core::cmp::Ordering::Less => {
                            assert!(phi < Fix128::ZERO, "cell {i},{j},{k} is inside");
                            inside += 1;
                        }
                        core::cmp::Ordering::Equal => assert_eq!(phi, Fix128::ZERO),
                        core::cmp::Ordering::Greater => {
                            assert!(phi > Fix128::ZERO, "cell {i},{j},{k} is outside");
                        }
                    }
                }
            }
        }
        // Lattice points with di² + dj² + dk² < 9: counted independently.
        let lattice = (-3..=3)
            .flat_map(|a| (-3..=3).flat_map(move |b| (-3..=3).map(move |c| (a, b, c))))
            .filter(|(a, b, c)| a * a + b * b + c * c < 9)
            .count();
        assert_eq!(inside, lattice, "dx = {}", dx.to_f64());
        assert_eq!(lattice, 93);
        // Offsets in [−5, 5]³ whose squared length is a perfect square,
        // counted independently (the exact cells must not be a vacuous set).
        let perfect = (-5..=5)
            .flat_map(|a| (-5..=5).flat_map(move |b| (-5..=5).map(move |c| (a, b, c))))
            .filter(|&d| exact_sphere_phi(Fix128::ONE, Fix128::ZERO, d).is_some())
            .count();
        assert_eq!(exact_checked, perfect, "dx = {}", dx.to_f64());
        assert!(perfect >= 100, "exact cells checked: {perfect}");
    }
}

#[test]
fn a_sphere_outside_the_grid_or_without_a_radius_encloses_no_cell() {
    // Centre at (10, 0, 0) on 5³ cells of spacing 1, r = 2: along the x axis
    // φ(i, 0, 0) = (10 − i) − 2, and nothing is inside.
    let mut g = Grid3d::new(5, 5, 5, Fix128::ONE, Fix128::ZERO);
    initialize_level_set_sphere(&mut g, int(10), Fix128::ZERO, Fix128::ZERO, int(2));
    for i in 0..5 {
        assert_eq!(g.get(i, 0, 0), int(8 - i as i64));
    }
    assert!(g.data.iter().all(|&phi| phi > Fix128::ZERO));

    // r = 0: φ is the distance, zero only at the centre cell.
    let mut g = Grid3d::new(5, 5, 5, Fix128::ONE, int(7));
    initialize_level_set_sphere(&mut g, int(2), int(2), int(2), Fix128::ZERO);
    assert_eq!(g.get(2, 2, 2), Fix128::ZERO);
    assert_eq!(g.data.iter().filter(|&&phi| phi.is_zero()).count(), 1);
    assert!(g.data.iter().all(|&phi| phi >= Fix128::ZERO));

    // r < 0: φ = distance + |r| > 0 everywhere (no inside exists).
    let mut g = Grid3d::new(5, 5, 5, Fix128::ONE, int(7));
    initialize_level_set_sphere(&mut g, int(2), int(2), int(2), int(-1));
    assert_eq!(g.get(2, 2, 2), Fix128::ONE);
    assert_eq!(g.get(4, 2, 2), int(3));
    assert!(g.data.iter().all(|&phi| phi > Fix128::ZERO));

    // Seeding the solver with it: a resting step keeps it positive.
    let mut s = resting(5, 5, 5, Fix128::ONE);
    s.level_set = Some(g);
    s.step_with_options(q(1, 16), &gs()).expect("steps");
    let ls = s.level_set.as_ref().expect("kept");
    assert!(ls.data.iter().all(|&phi| phi > Fix128::ZERO));
}

// ============================================================================
// Level set: reinitialize_level_set through StepOptions::with_level_set_reinit
// ============================================================================

fn gs() -> StepOptions {
    StepOptions::new(PressureSolver::RedBlackGs { sweeps: 30 })
}

fn pseudo(iterations: u32) -> StepOptions {
    gs().with_level_set_reinit(LevelSetReinit::PseudoTime { iterations })
}

/// A resting fluid (no gravity, no surface tension) that reinitialises its
/// level set on every step after the first.
fn resting(nx: usize, ny: usize, nz: usize, dx: Fix128) -> CfdSolver {
    let mut s = CfdSolver::new(nx, ny, nz, dx);
    s.gravity = Vec3Fix::ZERO;
    s.surface_tension_n_m = Fix128::ZERO;
    s.reinit_every_n_steps = 1;
    s
}

/// A plane `φ = slope · dx · (c_axis − 3)` on a `7`-cell axis with the other
/// two extents 3 (one interior line).
fn plane(axis: usize, slope: i64, dx: Fix128) -> Grid3d {
    let (nx, ny, nz) = match axis {
        0 => (7, 3, 3),
        1 => (3, 7, 3),
        _ => (3, 3, 7),
    };
    let mut g = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = [i, j, k][axis] as i64;
                g.set(i, j, k, int(slope) * dx * int(c - 3));
            }
        }
    }
    g
}

/// The same plane with the interior line replaced by `interior` (indexed by
/// the position along the axis, 0..7; the outer entries are ignored because
/// those cells are frozen).
fn plane_with_interior(axis: usize, slope: i64, dx: Fix128, interior: &[Fix128; 7]) -> Grid3d {
    let mut g = plane(axis, slope, dx);
    for (c, &value) in interior.iter().enumerate().take(6).skip(1) {
        let (i, j, k) = match axis {
            0 => (c, 1, 1),
            1 => (1, c, 1),
            _ => (1, 1, c),
        };
        g.set(i, j, k, value);
    }
    g
}

fn assert_resting_velocity(s: &CfdSolver, what: &str) {
    assert!(
        s.grid
            .u
            .iter()
            .chain(&s.grid.v)
            .chain(&s.grid.w)
            .all(|v| v.is_zero()),
        "{what}: the fluid moved"
    );
}

#[test]
fn pseudo_time_one_iteration_on_a_slope_two_plane_moves_the_interior_by_half_dx_toward_the_zero_level(
) {
    let dx = q(1, 4);
    for axis in AXES {
        let mut s = resting([7, 3, 3][axis], [3, 7, 3][axis], [3, 3, 7][axis], dx);
        s.level_set = Some(plane(axis, 2, dx));
        // Step 1 never reinitialises (step_count = 0), step 2 does.
        s.step_with_options(q(1, 16), &pseudo(1)).expect("step 1");
        assert_eq!(
            s.level_set.as_ref().expect("kept").data,
            plane(axis, 2, dx).data,
            "axis {axis}: step 1 is the identity on a resting fluid"
        );
        s.step_with_options(q(1, 16), &pseudo(1)).expect("step 2");
        assert_resting_velocity(&s, "slope-2 plane");
        // Interior: φ = (c − 3)/2 → φ − sgn(φ) · dx/2 = φ ∓ 1/8; the zero level stays.
        let want = plane_with_interior(
            axis,
            2,
            dx,
            &[int(0), q(-7, 8), q(-3, 8), int(0), q(3, 8), q(7, 8), int(0)],
        );
        assert_eq!(
            s.level_set.as_ref().expect("kept").data,
            want.data,
            "axis {axis}"
        );
        assert_eq!(s.step_count, 2);
    }
}

#[test]
fn pseudo_time_two_iterations_follow_the_jacobi_update_cell_by_cell() {
    // dx = 1, slope 2: after one iteration the interior line is
    // [−6, −7/2, −3/2, 0, 3/2, 7/2, 6]; the second iteration reads it with
    // central differences (9/4, 7/4, 3/2, 7/4, 9/4) and gives
    // [−6, −23/8, −9/8, 0, 9/8, 23/8, 6].
    let dx = Fix128::ONE;
    let mut s = resting(7, 3, 3, dx);
    s.level_set = Some(plane(0, 2, dx));
    s.step_with_options(q(1, 16), &pseudo(2)).expect("step 1");
    s.step_with_options(q(1, 16), &pseudo(2)).expect("step 2");
    let want = plane_with_interior(
        0,
        2,
        dx,
        &[
            int(0),
            q(-23, 8),
            q(-9, 8),
            int(0),
            q(9, 8),
            q(23, 8),
            int(0),
        ],
    );
    assert_eq!(s.level_set.as_ref().expect("kept").data, want.data);
}

#[test]
fn a_signed_distance_plane_is_a_fixed_point_of_pseudo_time_to_the_bit() {
    let dx = q(1, 4);
    for axis in AXES {
        let mut s = resting([7, 3, 3][axis], [3, 7, 3][axis], [3, 3, 7][axis], dx);
        s.level_set = Some(plane(axis, 1, dx));
        for _ in 0..4 {
            s.step_with_options(q(1, 16), &pseudo(5)).expect("step");
        }
        assert_eq!(
            s.level_set.as_ref().expect("kept").data,
            plane(axis, 1, dx).data,
            "axis {axis}"
        );
        assert_eq!(s.step_count, 4);
    }
}

/// A moving level-set solver: a sphere carried by a uniform `u`, reinitialised
/// every second step.
fn moving_sphere() -> CfdSolver {
    let mut s = CfdSolver::new(6, 6, 6, Fix128::ONE);
    s.gravity = Vec3Fix::ZERO;
    s.reinit_every_n_steps = 2;
    let mut ls = Grid3d::new(6, 6, 6, Fix128::ONE, Fix128::ZERO);
    initialize_level_set_sphere(&mut ls, int(3), int(3), int(3), int(2));
    s.level_set = Some(ls);
    for u in s.grid.u.iter_mut() {
        *u = Fix128::ONE;
    }
    s
}

fn level_set_bits(s: &CfdSolver) -> Vec<(i64, u64)> {
    s.level_set
        .as_ref()
        .expect("kept")
        .data
        .iter()
        .map(|v| (v.hi, v.lo))
        .collect()
}

#[test]
fn the_default_reinitialisation_of_step_with_options_is_bit_identical_to_step_and_pseudo_time_is_observable(
) {
    let dt = q(1, 2);
    let mut plain = moving_sphere();
    let mut new = moving_sphere();
    let mut explicit = moving_sphere();
    let mut pseudo_s = moving_sphere();
    let explicit_opts = gs().with_level_set_reinit(LevelSetReinit::FastSweeping { sweeps: 2 });
    for _ in 0..4 {
        plain.step(dt);
        new.step_with_options(dt, &gs()).expect("default options");
        explicit
            .step_with_options(dt, &explicit_opts)
            .expect("explicit default");
        pseudo_s
            .step_with_options(dt, &pseudo(2))
            .expect("pseudo-time");
    }
    assert_eq!(level_set_bits(&plain), level_set_bits(&new));
    assert_eq!(level_set_bits(&plain), level_set_bits(&explicit));
    assert_eq!(plain.grid.u, new.grid.u);
    assert_eq!(plain.grid.pressure, explicit.grid.pressure);
    assert_ne!(
        level_set_bits(&plain),
        level_set_bits(&pseudo_s),
        "the pseudo-time scheme must be observable"
    );
    // `StepOptions` is `PartialEq`: the builder stores what it was given, and
    // `new` carries the default the solver runs.
    assert_eq!(
        gs(),
        gs().with_level_set_reinit(LevelSetReinit::FastSweeping { sweeps: 2 })
    );
    assert_ne!(gs(), pseudo(3));
}

#[test]
fn a_zero_reinitialisation_count_is_refused_with_the_solver_untouched() {
    for reinit in [
        LevelSetReinit::FastSweeping { sweeps: 0 },
        LevelSetReinit::PseudoTime { iterations: 0 },
    ] {
        for with_level_set in [true, false] {
            let mut s = moving_sphere();
            if !with_level_set {
                s.level_set = None;
            }
            let u0 = s.grid.u.clone();
            let ls0 = s.level_set.clone().map(|g| g.data);
            let err = s
                .step_with_options(q(1, 2), &gs().with_level_set_reinit(reinit))
                .expect_err("refused");
            assert_eq!(err, StepError::ZeroReinitCount, "{reinit:?}");
            assert_eq!(s.step_count, 0, "{reinit:?}: not stepped");
            assert_eq!(s.grid.u, u0, "{reinit:?}: grid untouched");
            assert_eq!(s.level_set.clone().map(|g| g.data), ls0, "{reinit:?}");

            let mut state = RansState::new(6, 6, 6, TurbulenceModel::Smagorinsky);
            let err = s
                .step_rans(q(1, 2), &gs().with_level_set_reinit(reinit), &mut state)
                .expect_err("refused on the RANS path too");
            assert_eq!(err, StepError::ZeroReinitCount, "{reinit:?} (rans)");
            assert_eq!(s.step_count, 0);
        }
    }
}

#[test]
fn pseudo_time_on_a_level_set_without_an_interior_leaves_it_untouched_and_does_not_panic() {
    // The level set has to share the grid's extents (the CSF stage indexes
    // it with them); the zero-extent cases live in the module's unit tests.
    for (nx, ny, nz) in [(2, 3, 3), (3, 2, 3), (3, 3, 2), (2, 2, 2), (1, 1, 1)] {
        let mut s = resting(nx, ny, nz, Fix128::ONE);
        let ls = Grid3d::new(nx, ny, nz, Fix128::ONE, int(5));
        s.level_set = Some(ls.clone());
        let res = catch_unwind(AssertUnwindSafe(|| {
            for _ in 0..3 {
                s.step_with_options(q(1, 16), &pseudo(4)).expect("steps");
            }
        }));
        assert!(res.is_ok(), "{nx}x{ny}x{nz} panicked");
        assert_eq!(
            s.level_set.as_ref().expect("kept").data,
            ls.data,
            "{nx}x{ny}x{nz}"
        );
        assert_eq!(s.step_count, 3);
    }
}

#[test]
fn pseudo_time_with_a_zero_spacing_level_set_leaves_it_untouched() {
    // Δτ = dx/2 = 0: the documented no-op (the gradient quotient is 0/0 → 0
    // in Fix128, and 0 · anything leaves φ).
    let mut s = resting(7, 3, 3, Fix128::ONE);
    let ls = plane(0, 2, Fix128::ZERO);
    let mut seeded = ls.clone();
    seeded.data = plane(0, 2, Fix128::ONE).data; // a real profile with dx = 0
    s.level_set = Some(seeded.clone());
    for _ in 0..3 {
        s.step_with_options(q(1, 16), &pseudo(4)).expect("steps");
    }
    assert_eq!(s.level_set.as_ref().expect("kept").data, seeded.data);
    assert!(ls.dx.is_zero());
}
