//! Oracles for the sealed-box helpers of `MacGrid` and the adaptive time
//! step of `CfdSolver`:
//! `compute_max_dt`, `step_adaptive`, `enforce_solid_faces`, the face setters'
//! out-of-range rule, and an interior solid face under `step`.
//!
//! # Closed forms
//!
//! * `compute_max_dt(c) = c · dx / max |u_face|`, capped at
//!   `CfdSolver::MAX_DT_CAP = 1_000_000` s. With a dyadic `c`, `dx` and peak
//!   the quotient is exact, so the asserts are `assert_eq!`. The cap boundary
//!   is pinned on both sides: a peak of `2^-20` on a unit grid gives
//!   `2^20 = 1_048_576 > cap`, a peak of `2^-19` gives `524_288 < cap`.
//! * `step_adaptive(c, ceiling)` returns `min(compute_max_dt(c), ceiling)`
//!   and the solver state afterwards equals `step(that dt)` bit for bit.
//! * `enforce_solid_faces` zeroes exactly the faces whose solid flag is set
//!   and leaves every other face, inflow and outflow included, as stored:
//!   the contrast with `enforce_face_boundaries`, which rewrites those, is
//!   asserted on the same grid.
//!
//! # Degenerate input
//!
//! A peak of a few ulp on a grid of order-one spacing would push
//! `c · dx / peak` past `2^63` and wrap the quotient into a wrong, possibly
//! negative, time step; the cap is what keeps that path out, and the test
//! for it uses exactly that input (`dx = 3/2`, one face at one ulp). A
//! non-positive `cfl_target`, a zero `dx`, or a non-positive ceiling give a
//! zero `dt`, and `step_adaptive` then takes no step at all (grid and
//! `step_count` untouched). The setters ignore out-of-range faces without
//! panicking, up to `usize::MAX`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn faces(g: &MacGrid) -> (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>) {
    (g.u.clone(), g.v.clone(), g.w.clone())
}

fn iu(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + (g.nx + 1) * (j + g.ny * k)
}

fn iv(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + g.nx * (j + (g.ny + 1) * k)
}

fn iw(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + g.nx * (j + g.ny * k)
}

/// A sealed 4 × 4 × 2 box, `dx = 1/4`, no gravity, with an interior peak of
/// `|w| = 8` (and a smaller `u = 4`) so the Courant bound is `c / 32`.
fn seeded() -> CfdSolver {
    let mut s = CfdSolver::new(4, 4, 2, q(1, 4));
    s.gravity = Vec3Fix::ZERO;
    s.jacobi_iterations = 10;
    s.grid.set_closed_box_walls();
    let iu = iu(&s.grid, 1, 1, 0);
    s.grid.u[iu] = int(4);
    let iw = iw(&s.grid, 1, 1, 1);
    s.grid.w[iw] = int(-8);
    s
}

// ===========================================================================
// compute_max_dt
// ===========================================================================

#[test]
fn max_dt_is_cfl_times_dx_over_the_peak_face_speed() {
    let s = seeded();
    // c · dx / |peak| = (1/2)(1/4) / 8 = 1/64, the negative peak taken by magnitude.
    assert_eq!(s.compute_max_dt(q(1, 2)), q(1, 64));
    assert_eq!(s.compute_max_dt(int(1)), q(1, 32));
    assert_eq!(s.compute_max_dt(int(2)), q(1, 16));
    // Raising the smaller component past the peak moves the bound with it.
    let mut t = seeded();
    let iu = iu(&t.grid, 2, 2, 1);
    t.grid.u[iu] = int(16);
    assert_eq!(t.compute_max_dt(q(1, 2)), q(1, 128));
}

#[test]
fn a_resting_field_reports_the_cap() {
    let mut s = CfdSolver::new(3, 3, 3, q(1, 2));
    s.grid.set_closed_box_walls();
    assert_eq!(s.compute_max_dt(q(1, 2)), CfdSolver::MAX_DT_CAP);
    assert_eq!(CfdSolver::MAX_DT_CAP, int(1_000_000));
}

#[test]
fn a_nearly_resting_field_reports_the_cap_not_a_wrapped_quotient() {
    // 1 · (3/2) / 2^-64 = 3 · 2^63 does not fit a Fix128; the unchecked
    // quotient would come out negative.
    let mut s = CfdSolver::new(2, 2, 2, q(3, 2));
    let ix = iu(&s.grid, 1, 0, 0);
    s.grid.u[ix] = Fix128::from_raw(0, 1);
    let dt = s.compute_max_dt(int(1));
    assert!(dt > Fix128::ZERO, "wrapped into {dt:?}");
    assert_eq!(dt, CfdSolver::MAX_DT_CAP);

    // Both sides of the cap boundary on a unit grid with c = 1.
    let mut above = CfdSolver::new(2, 2, 2, Fix128::ONE);
    let ix = iv(&above.grid, 0, 1, 0);
    above.grid.v[ix] = Fix128::from_raw(0, 1 << 44); // 2^-20
    assert_eq!(above.compute_max_dt(int(1)), CfdSolver::MAX_DT_CAP);
    let mut below = CfdSolver::new(2, 2, 2, Fix128::ONE);
    let ix = iv(&below.grid, 0, 1, 0);
    below.grid.v[ix] = Fix128::from_raw(0, 1 << 45); // 2^-19
    assert_eq!(below.compute_max_dt(int(1)), int(524_288));
}

#[test]
fn non_positive_cfl_and_zero_dx_give_zero() {
    let s = seeded();
    assert_eq!(s.compute_max_dt(Fix128::ZERO), Fix128::ZERO);
    assert_eq!(s.compute_max_dt(int(-1)), Fix128::ZERO);
    let mut flat = CfdSolver::new(2, 2, 2, Fix128::ZERO);
    let ix = iu(&flat.grid, 1, 0, 0);
    flat.grid.u[ix] = int(3);
    assert_eq!(flat.compute_max_dt(q(1, 2)), Fix128::ZERO);
}

// ===========================================================================
// step_adaptive
// ===========================================================================

#[test]
fn step_adaptive_takes_the_smaller_of_cfl_and_ceiling_and_steps_exactly_like_step() {
    for (ceiling, want) in [
        (int(1), q(1, 64)),
        (q(1, 256), q(1, 256)),
        (q(1, 64), q(1, 64)),
    ] {
        let mut adaptive = seeded();
        let got = adaptive.step_adaptive(q(1, 2), ceiling);
        assert_eq!(got, want, "ceiling {ceiling:?}");
        let mut plain = seeded();
        plain.step(want);
        assert_eq!(
            faces(&adaptive.grid),
            faces(&plain.grid),
            "ceiling {ceiling:?}"
        );
        assert_eq!(adaptive.grid.pressure, plain.grid.pressure);
        assert_eq!(adaptive.step_count, 1);
        // Not vacuous: the step changed the seeded field.
        assert_ne!(faces(&adaptive.grid), faces(&seeded().grid));
    }
}

#[test]
fn step_adaptive_refuses_a_non_positive_time_step_and_leaves_the_solver_untouched() {
    type Case = (fn() -> CfdSolver, Fix128, Fix128, &'static str);
    let cases: [Case; 5] = [
        (seeded, q(1, 2), Fix128::ZERO, "ceiling = 0"),
        (seeded, q(1, 2), int(-1), "ceiling < 0"),
        (seeded, Fix128::ZERO, int(1), "cfl = 0"),
        (seeded, int(-2), int(1), "cfl < 0"),
        (
            || {
                let mut s = CfdSolver::new(2, 2, 2, Fix128::ZERO);
                let ix = iu(&s.grid, 1, 0, 0);
                s.grid.u[ix] = int(3);
                s
            },
            q(1, 2),
            int(1),
            "dx = 0",
        ),
    ];
    for (make, cfl, ceiling, what) in cases {
        let mut s = make();
        let before = faces(&s.grid);
        let res = catch_unwind(AssertUnwindSafe(|| s.step_adaptive(cfl, ceiling)));
        let dt = res.unwrap_or_else(|_| panic!("{what}: panicked"));
        assert_eq!(dt, Fix128::ZERO, "{what}");
        assert_eq!(faces(&s.grid), before, "{what}: grid changed");
        assert_eq!(s.step_count, 0, "{what}: a step was counted");
    }
    // A resting field under a ceiling above the cap integrates the cap.
    let mut rest = CfdSolver::new(2, 2, 2, q(1, 2));
    rest.gravity = Vec3Fix::ZERO;
    rest.grid.set_closed_box_walls();
    assert_eq!(
        rest.step_adaptive(q(1, 2), int(2_000_000)),
        CfdSolver::MAX_DT_CAP
    );
    assert_eq!(rest.step_count, 1);
}

// ===========================================================================
// enforce_solid_faces
// ===========================================================================

#[test]
fn enforce_solid_faces_zeroes_walls_only_and_leaves_every_other_face_as_stored() {
    let n = 3usize;
    let mut g = MacGrid::new(n, n, n, q(1, 4));
    g.set_closed_box_walls();
    g.set_u_solid(1, 1, 1, true);
    g.set_v_solid(2, 2, 0, true);
    g.set_w_solid(0, 1, 2, true);
    let inflow = FaceBc::Inflow {
        normal_velocity: int(2),
    };
    g.set_u_bc(0, 1, 1, inflow);
    g.set_u_bc(n, 1, 1, FaceBc::Outflow);
    g.set_v_bc(1, 0, 2, FaceBc::SlipWall);
    for (ix, u) in g.u.iter_mut().enumerate() {
        *u = int(ix as i64 + 1);
    }
    for (ix, v) in g.v.iter_mut().enumerate() {
        *v = int(-(ix as i64) - 1);
    }
    for (ix, w) in g.w.iter_mut().enumerate() {
        *w = q(ix as i64 + 1, 2);
    }
    let seed = faces(&g);
    let mut contrast = g.clone();

    g.enforce_solid_faces();
    let (mut walls, mut others) = (0, 0);
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let ix = iu(&g, i, j, k);
                if g.is_u_solid(i, j, k) {
                    walls += 1;
                    assert_eq!(g.u[ix], Fix128::ZERO, "u wall ({i}, {j}, {k})");
                } else {
                    others += 1;
                    assert_eq!(g.u[ix], seed.0[ix], "u non-wall ({i}, {j}, {k})");
                }
            }
        }
    }
    for k in 0..n {
        for j in 0..=n {
            for i in 0..n {
                let ix = iv(&g, i, j, k);
                if g.is_v_solid(i, j, k) {
                    walls += 1;
                    assert_eq!(g.v[ix], Fix128::ZERO, "v wall ({i}, {j}, {k})");
                } else {
                    others += 1;
                    assert_eq!(g.v[ix], seed.1[ix], "v non-wall ({i}, {j}, {k})");
                }
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                let ix = iw(&g, i, j, k);
                if g.is_w_solid(i, j, k) {
                    walls += 1;
                    assert_eq!(g.w[ix], Fix128::ZERO, "w wall ({i}, {j}, {k})");
                } else {
                    others += 1;
                    assert_eq!(g.w[ix], seed.2[ix], "w non-wall ({i}, {j}, {k})");
                }
            }
        }
    }
    // 6 layers of 9 faces, minus the inflow and outflow faces, plus 3
    // interior walls; the slip wall is still a wall.
    assert_eq!(walls, 54 - 2 + 3);
    assert_eq!(others, 3 * 36 - walls);
    // The inflow face kept its stored value, not its prescribed one; the
    // outflow face kept its stored value, not the inner one.
    assert_eq!(g.u(0, 1, 1), seed.0[iu(&g, 0, 1, 1)]);
    assert_eq!(g.u(n, 1, 1), seed.0[iu(&g, n, 1, 1)]);

    // The superset does rewrite them.
    contrast.enforce_face_boundaries();
    assert_eq!(contrast.u(0, 1, 1), int(2));
    assert_eq!(contrast.u(n, 1, 1), contrast.u(n - 1, 1, 1));
    assert_eq!(contrast.u(1, 1, 1), Fix128::ZERO);

    // No walls: bit-unchanged. Zero-sized: no panic.
    let mut open = MacGrid::new(2, 2, 2, Fix128::ONE);
    open.u.fill(int(5));
    let before = faces(&open);
    open.enforce_solid_faces();
    assert_eq!(faces(&open), before);
    for (nx, ny, nz) in [(0, 2, 2), (2, 0, 2), (2, 2, 0), (0, 0, 0)] {
        let mut z = MacGrid::new(nx, ny, nz, Fix128::ONE);
        z.set_closed_box_walls();
        assert!(catch_unwind(AssertUnwindSafe(|| z.enforce_solid_faces())).is_ok());
    }
}

// ===========================================================================
// The setters' out-of-range rule
// ===========================================================================

#[test]
fn setters_ignore_out_of_range_faces_without_panicking() {
    let (nx, ny, nz) = (2usize, 3usize, 4usize);
    let mut g = MacGrid::new(nx, ny, nz, Fix128::ONE);
    let before = faces(&g);
    let wall = FaceBc::Wall {
        velocity: Vec3Fix::new(int(1), Fix128::ZERO, Fix128::ZERO),
    };
    let big = usize::MAX;
    let res = catch_unwind(AssertUnwindSafe(|| {
        // u: i in 0..=nx, j < ny, k < nz
        g.set_u_bc(nx + 1, 0, 0, wall);
        g.set_u_bc(0, ny, 0, wall);
        g.set_u_solid(0, 0, nz, true);
        g.set_u_bc(big, big, big, wall);
        // v: i < nx, j in 0..=ny, k < nz
        g.set_v_bc(nx, 0, 0, wall);
        g.set_v_solid(0, ny + 1, 0, true);
        g.set_v_bc(0, 0, nz, wall);
        g.set_v_solid(big, 0, 0, true);
        // w: i < nx, j < ny, k in 0..=nz
        g.set_w_solid(nx, 0, 0, true);
        g.set_w_bc(0, ny, 0, wall);
        g.set_w_bc(0, 0, nz + 1, wall);
        g.set_w_solid(0, 0, big, true);
    }));
    assert!(res.is_ok(), "an out-of-range setter panicked");
    assert_eq!(faces(&g), before);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                assert_eq!(g.u_bc(i, j, k), FaceBc::Fluid);
                assert!(!g.is_u_solid(i, j, k));
            }
        }
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                assert_eq!(g.v_bc(i, j, k), FaceBc::Fluid);
                assert!(!g.is_v_solid(i, j, k));
            }
        }
    }
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                assert_eq!(g.w_bc(i, j, k), FaceBc::Fluid);
                assert!(!g.is_w_solid(i, j, k));
            }
        }
    }
    // The edge faces themselves are in range.
    g.set_u_solid(nx, ny - 1, nz - 1, true);
    g.set_v_bc(nx - 1, ny, nz - 1, wall);
    g.set_w_solid(nx - 1, ny - 1, nz, true);
    assert!(g.is_u_solid(nx, ny - 1, nz - 1));
    assert_eq!(g.v_bc(nx - 1, ny, nz - 1), wall);
    assert!(g.is_w_solid(nx - 1, ny - 1, nz));
}

// ===========================================================================
// An interior solid face under the full step
// ===========================================================================

/// A lid-driven sealed box with a baffle: the X-faces `(3, j, 0)` for
/// `j < 3` are marked solid through `set_u_solid`. After every step the flux
/// through each baffle face is exactly zero while the lid keeps the rest of
/// the fluid moving.
#[test]
fn an_interior_solid_face_carries_no_flux_after_a_step() {
    let n = 6usize;
    let mut s = CfdSolver::new(n, n, 1, q(1, 8));
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = q(1, 100);
    s.jacobi_iterations = 30;
    s.grid.set_closed_box_walls();
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    };
    for i in 0..n {
        s.grid.set_v_bc(i, n, 0, lid);
    }
    for j in 0..3 {
        s.grid.set_u_solid(3, j, 0, true);
    }
    for step in 0..6 {
        let dt = s.step_adaptive(q(1, 2), q(1, 32));
        assert!(dt > Fix128::ZERO);
        for j in 0..3 {
            assert_eq!(
                s.grid.u(3, j, 0),
                Fix128::ZERO,
                "step {step}: baffle face j = {j}"
            );
        }
        // The face just above the baffle is open and, after the first step,
        // the cavity is in motion.
        if step > 0 {
            let moving = s.grid.u.iter().any(|&u| !u.is_zero());
            assert!(moving, "step {step}: the lid did not move the fluid");
        }
    }
    // Without the baffle the same faces carry flux, so the assert has teeth.
    let mut open = CfdSolver::new(n, n, 1, q(1, 8));
    open.gravity = Vec3Fix::ZERO;
    open.density_kg_m3 = Fix128::ONE;
    open.dynamic_viscosity_pas = q(1, 100);
    open.jacobi_iterations = 30;
    open.grid.set_closed_box_walls();
    for i in 0..n {
        open.grid.set_v_bc(i, n, 0, lid);
    }
    for _ in 0..6 {
        open.step_adaptive(q(1, 2), q(1, 32));
    }
    assert!((0..3).any(|j| !open.grid.u(3, j, 0).is_zero()));
}
