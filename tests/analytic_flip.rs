//! Oracles for the FLIP / PIC particle path `CfdSolver::step_flip`.
//!
//! `step_flip` is the first production caller of
//! `eulerian_grid::p2g_normalized`: particles are scattered onto the MAC grid,
//! the grid takes one body-force + projection step, and each particle gets
//! `v_p = (1 - r) G2P(u_new) + r (v_p + G2P(u_new - u_old))` before it is
//! advected `x += v dt` and clamped to the box.
//!
//! # Contract of the first stage (what these tests do and do not claim)
//!
//! The pressure solver has no "classify cells as fluid or air, make the air
//! cells `p = 0`" machinery, so a free surface is a different feature. The
//! step is therefore only meaningful when the particles fill the whole
//! domain; a face no particle reaches is cleared to zero and treated as a
//! fluid face. `a_partial_lattice_...` pins the clearing; it does not claim the
//! result is physical. Particles lying outside `[0, N dx]` on any axis take
//! no part and are left bit-identical.
//!
//! # Degenerate input: early return, grid and particles bit-unchanged
//!
//! An empty particle list, `dt <= 0`, `dx = 0`, a zero-sized grid, a zero
//! density and a `flip_ratio` outside `[0, 1]` are not clamped into something
//! plausible (clamping would run a different scheme than the one asked for);
//! the call returns with the grid and the particles untouched, the same
//! convention `CfdSolver::step` has for `dt = 0`. The panic tests below
//! pin both halves: no panic, and bit-unchanged.
//!
//! # Arithmetic
//!
//! Positions and velocities are dyadic rationals, so every product and sum in
//! the transfer and in the advection is exact in `Fix128` and those asserts
//! are `assert_eq!`. The pressure solve is iterative with `1/deg` factors that
//! are not dyadic, so the hydrostatic oracle is the one place a tolerance is
//! used; its derivation and the measured error are in that test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

type Particle = (Vec3Fix, Vec3Fix);

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// Eight particles per cell at `(2a + 1) / 4` along each axis (`dx = 1`):
/// spacing 1/2, spanning `[1/4, n - 1/4]`, every one carrying `vel`.
fn lattice(nx: i64, ny: i64, nz: i64, vel: Vec3Fix) -> Vec<Particle> {
    let mut out = Vec::new();
    for a in 0..2 * nx {
        for b in 0..2 * ny {
            for c in 0..2 * nz {
                out.push((v3(q(2 * a + 1, 4), q(2 * b + 1, 4), q(2 * c + 1, 4)), vel));
            }
        }
    }
    out
}

fn solver(nx: usize, ny: usize, nz: usize) -> CfdSolver {
    CfdSolver::new(nx, ny, nz, Fix128::ONE)
}

type GridBits = (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>, Vec<Fix128>);

fn grid_bits(g: &MacGrid) -> GridBits {
    (g.u.clone(), g.v.clone(), g.w.clone(), g.pressure.clone())
}

fn abs_diff(a: Fix128, b: Fix128) -> Fix128 {
    (a - b).abs()
}

// ---------------------------------------------------------------------------
// 1. Hydrostatic column
// ---------------------------------------------------------------------------

/// Closed box, top faces `Outflow` (`p = 0`), filled with resting particles,
/// gravity `g = -8`, `rho = 1000`, `dx = 1`, `dt = 1/16`.
///
/// Derivation of the discrete steady state. The projection sets
/// `v_j = v*_j - (dt/rho) (p_j - p_{j-1}) / dx` on an interior y-face, with
/// `v*_j = g dt` after the body force. Resting fluid means `v_j = 0`, so
/// `p_j - p_{j-1} = rho g dx` (negative: pressure falls upward). The top face
/// reads `p_ny = 0` (`subtract_pressure_gradient` uses `0` for the cell above
/// the domain and `Outflow` extrapolates `v*` from the face below), which
/// telescopes to
///
/// ```text
/// p_j = rho |g| dx (ny - j)
/// ```
///
/// The continuum answer at the cell centre `y_j = (j + 1/2) dx` is
/// `rho |g| (H - y_j) = rho |g| dx (ny - j - 1/2)`. They differ by exactly
/// `rho |g| dx / 2`: the Dirichlet value sits at the ghost-cell centre half a
/// cell above the free end, not at the face. That offset is a property of the
/// existing projection (first-order boundary placement), pinned below as the
/// second assertion so it cannot drift unnoticed.
///
/// Tolerance, two cases.
///
/// 1. `mu = 0` (the derivation above holds exactly). The Gauss-Seidel fixed
///    point is exact; the run differs from it by the residual of 1500 sweeps
///    (contraction about 0.97 per sweep for this 5-cell column, below
///    `1e-17`) plus `Fix128` rounding (`2^-64` per operation, amplified by the
///    `rho dx^2 / dt = 16000` scale of the right-hand side). The bound `1e-9`
///    is that budget with a wide margin and is tiny against the quantity
///    measured (`rho |g| dx = 8000`, relative `1e-13`): a wrong boundary
///    placement or a wrong sign moves the answer by thousands.
/// 2. Default viscosity `mu = 1e-3`. The explicit diffusion acts on `v*`
///    before the projection, and `v*` jumps from `0` (wall) to `g dt` one face
///    away, so it perturbs the right-hand side by up to
///    `6 (nu dt / dx^2) |g| dt` per face (six neighbours, `nu = mu / rho`).
///    The pressure is a sum over the `ny` cells of the column of
///    `rho dx / dt` times that, so
///    `|dp| <= 6 ny mu |g| dt / dx = 6 * 5 * 1e-3 * 8 / 16 = 0.015`.
///    Measured with 1500 and with 6000 sweeps alike: `6.7e-3` (45% of the
///    bound), so the residual is the viscous term and not the iteration. The
///    particle displacement bound follows: `|v| <= 2 (dt / rho) |dp| / dx`,
///    three steps of `dt`, below `1e-6`.
#[test]
fn hydrostatic_column_pressure_and_particles_at_rest() {
    let (nx, ny, nz) = (3usize, 5usize, 3usize);
    let make = |mu: Fix128| {
        let mut s = solver(nx, ny, nz);
        s.dynamic_viscosity_pas = mu;
        s.grid.set_closed_box_walls();
        for k in 0..nz {
            for i in 0..nx {
                s.grid.set_v_bc(i, ny, k, FaceBc::Outflow);
            }
        }
        s.gravity = v3(Fix128::ZERO, int(-8), Fix128::ZERO);
        s.jacobi_iterations = 1500;
        s
    };
    let dt = q(1, 16);
    let initial = lattice(nx as i64, ny as i64, nz as i64, Vec3Fix::ZERO);
    let cases = [
        (Fix128::ZERO, q(1, 1_000_000_000), q(1, 1_000_000_000)),
        (q(1, 1000), q(15, 1000), q(1, 1_000_000)),
    ];
    for (r, (mu, tol, move_tol)) in [Fix128::ZERO, q(1, 2), Fix128::ONE]
        .into_iter()
        .flat_map(|r| cases.iter().map(move |c| (r, *c)))
    {
        let mut s = make(mu);
        let mut ps = initial.clone();
        for _ in 0..3 {
            s.step_flip(&mut ps, dt, r);
        }
        let rho_g_dx = int(8000);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = s.grid.pressure(i, j, k);
                    let discrete = rho_g_dx * int((ny - j) as i64);
                    assert!(
                        abs_diff(p, discrete) < tol,
                        "r={r} mu={mu}: p({i},{j},{k}) = {p}, discrete steady state {discrete}"
                    );
                    let continuum = rho_g_dx * (int((ny - j) as i64) - q(1, 2));
                    assert!(
                        abs_diff(p - continuum, rho_g_dx * q(1, 2)) < tol,
                        "r={r} mu={mu}: offset from the continuum profile at ({i},{j},{k})"
                    );
                }
            }
        }
        // Resting fluid: nothing moves (bounds in the doc above).
        for (p0, p1) in initial.iter().zip(ps.iter()) {
            assert!(
                abs_diff(p0.0.x, p1.0.x) < move_tol,
                "x moved: {:?} -> {:?}",
                p0.0,
                p1.0
            );
            assert!(
                abs_diff(p0.0.y, p1.0.y) < move_tol,
                "y moved: {:?} -> {:?}",
                p0.0,
                p1.0
            );
            assert!(
                abs_diff(p0.0.z, p1.0.z) < move_tol,
                "z moved: {:?} -> {:?}",
                p0.0,
                p1.0
            );
            assert!(p1.1.y.abs() < move_tol, "particle velocity {:?}", p1.1);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Uniform-flow translation
// ---------------------------------------------------------------------------

/// Duct along x: inflow on one x-end carrying `u_in`, outflow on the other,
/// free-slip walls on the y and z faces, no body force, no viscosity.
fn duct(n: usize, inflow_at_low_x: bool, speed: i64) -> CfdSolver {
    let mut s = solver(n, n, n);
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    let (lo, hi) = if inflow_at_low_x {
        (
            FaceBc::Inflow {
                normal_velocity: int(speed),
            },
            FaceBc::Outflow,
        )
    } else {
        (
            FaceBc::Outflow,
            FaceBc::Inflow {
                normal_velocity: int(-speed),
            },
        )
    };
    for k in 0..n {
        for j in 0..n {
            s.grid.set_u_bc(0, j, k, lo);
            s.grid.set_u_bc(n, j, k, hi);
        }
    }
    for k in 0..n {
        for i in 0..n {
            s.grid.set_v_bc(i, 0, k, FaceBc::SlipWall);
            s.grid.set_v_bc(i, n, k, FaceBc::SlipWall);
        }
    }
    for j in 0..n {
        for i in 0..n {
            s.grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            s.grid.set_w_bc(i, j, n, FaceBc::SlipWall);
        }
    }
    s
}

/// A uniform stream `U` with `p = 0` is divergence free, so the projection
/// changes nothing, `G2P(u_new) = U` and `G2P(u_new - u_old) = 0`, and every
/// particle must move by exactly `U dt` per step: `U = 2`, `dt = 1/16` gives
/// `1/8`, so three steps are `3/8`. A particle whose `x + 3/8` would pass the
/// far wall at `x = 4` stops on it (the clamp). The same holds mirrored.
#[test]
fn uniform_stream_translates_particles_by_u_dt_and_clamps_at_the_wall() {
    let n = 4usize;
    let dt = q(1, 16);
    let steps = 3i64;
    for (inflow_low, sign) in [(true, 1i64), (false, -1i64)] {
        for r in [Fix128::ZERO, q(1, 4), Fix128::ONE] {
            let mut s = duct(n, inflow_low, 2);
            let u_vec = v3(int(2 * sign), Fix128::ZERO, Fix128::ZERO);
            let initial = lattice(n as i64, n as i64, n as i64, u_vec);
            let mut ps = initial.clone();
            for step in 1..=steps {
                s.step_flip(&mut ps, dt, r);
                // Every face of the grid carries the stream.
                for (idx, &u) in s.grid.u.iter().enumerate() {
                    assert_eq!(u, u_vec.x, "r={r} step {step}: u[{idx}]");
                }
                for &v in s.grid.v.iter().chain(s.grid.w.iter()) {
                    assert_eq!(v, Fix128::ZERO, "r={r} step {step}");
                }
                for &p in &s.grid.pressure {
                    assert_eq!(p, Fix128::ZERO, "r={r} step {step}: pressure");
                }
            }
            let shift = q(steps * 2 * sign, 16);
            let mut clamped = 0;
            for (p0, p1) in initial.iter().zip(ps.iter()) {
                let free = p0.0.x + shift;
                let expected = if free > int(n as i64) {
                    clamped += 1;
                    int(n as i64)
                } else if free < Fix128::ZERO {
                    clamped += 1;
                    Fix128::ZERO
                } else {
                    free
                };
                assert_eq!(p1.0.x, expected, "r={r} sign={sign}: x {:?}", p0.0);
                assert_eq!(p1.0.y, p0.0.y);
                assert_eq!(p1.0.z, p0.0.z);
                assert_eq!(p1.1, u_vec, "r={r}: velocity of {:?}", p0.0);
            }
            // The test must actually reach the clamp, or it says nothing about it:
            // the 8 x-layers within 3/8 of the wall times 16 y-z positions.
            assert!(clamped > 0, "no particle reached the wall");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. FLIP versus PIC
// ---------------------------------------------------------------------------

/// Index of the lattice particle at `(5/4, 5/4, 5/4)` in `lattice(6, 6, 6, ..)`
/// (`a = b = c = 2`, `a` outermost).
fn target_index() -> usize {
    (2 * 12 + 2) * 12 + 2
}

/// A uniform stream `U` with one particle's x-velocity raised by `delta`,
/// projection switched off (`jacobi_iterations = 0`) and no force, so the grid
/// does not change in the step: `u_new = u_old`, `G2P(u_new - u_old) = 0`.
/// (With the projection on, the extra particle makes the field divergent and
/// the projection is a separate effect; `a_free_fall_...` below covers a
/// non-zero `u_new - u_old`.)
///
/// Expected, derived by hand. The target sits at `(5/4, 5/4, 5/4)`. For the
/// u-faces the trilinear weights per axis are `x: (3/4, 1/4)` on faces
/// `i = 1, 2`, `y: (1/4, 3/4)` and `z: (1/4, 3/4)` on `j, k = 1, 2` (stagger
/// 0, 1/2, 1/2). All 8 faces are interior, where the weight sum of the
/// lattice is `2 * 2 * 2 = 8`, so after normalisation each face holds
///
/// ```text
/// u_f = (8 U + delta w_f) / 8 = U + delta w_f / 8
/// ```
///
/// and `G2P` at the target reads the same 8 faces with the same `w_f`:
///
/// ```text
/// sum w_f^2 = (9/16 + 1/16) (1/16 + 9/16) (1/16 + 9/16) = (5/8)^3 = 125/512
/// PIC = U + delta (sum w_f^2) / 8
/// PIC = U + delta * 125/4096
/// FLIP = U + delta                     (grid unchanged, so the particle keeps its velocity)
/// v(r) = (1 - r) PIC + r FLIP
/// ```
///
/// Particles outside the target's support keep `U` exactly.
#[test]
fn flip_keeps_the_particle_velocity_and_pic_takes_the_smoothed_grid_value() {
    let delta = Fix128::ONE;
    let u = v3(int(3), int(-2), q(5, 2));
    let pic_gain = q(125, 4096);
    for r in [Fix128::ZERO, q(1, 4), q(1, 2), q(3, 4), Fix128::ONE] {
        let mut s = solver(6, 6, 6);
        s.gravity = Vec3Fix::ZERO;
        s.dynamic_viscosity_pas = Fix128::ZERO;
        s.jacobi_iterations = 0;
        let mut ps = lattice(6, 6, 6, u);
        let t = target_index();
        assert_eq!(ps[t].0, v3(q(5, 4), q(5, 4), q(5, 4)));
        ps[t].1.x = u.x + delta;
        let before = ps.clone();
        s.step_flip(&mut ps, q(1, 64), r);
        let flip = u.x + delta;
        let pic = u.x + delta * pic_gain;
        let expected = (Fix128::ONE - r) * pic + r * flip;
        assert_eq!(ps[t].1.x, expected, "r={r}");
        assert_eq!(ps[t].1.y, u.y, "r={r}");
        assert_eq!(ps[t].1.z, u.z, "r={r}");
        // A particle well outside the target's support is untouched.
        let far = lattice(6, 6, 6, u).len() - 1;
        assert_eq!(ps[far].1, u, "r={r}: far particle");
        if r == Fix128::ONE {
            for (a, b) in before.iter().zip(ps.iter()) {
                assert_eq!(a.1, b.1, "FLIP with r=1 must leave every velocity alone");
            }
        }
    }
}

/// Wall faces are imposed before `u_old` is taken. Closed box, uniform
/// `u = 4` plug, no force, projection off: the transfer leaves `4` on the wall
/// face `i = 0`, `enforce_face_boundaries` makes it `0`, and nothing changes
/// afterwards, so `u_new - u_old = 0`. The particle at `(1/4, 5/4, 5/4)` has
/// x-weights `3/4` on the wall face (value 0) and `1/4` on face 1 (value 4),
/// and unit weight sum in y and z, so
///
/// ```text
/// PIC  (r = 0): G2P(u_new) = 4 * 1/4 = 1
/// FLIP (r = 1): v_p + G2P(0) = 4
/// ```
///
/// If the wall were imposed only after `u_old` was taken, `u_old` would keep
/// `4` on the wall, `u_new - u_old = -4` there and FLIP would also read
/// `4 - 3 = 1`.
#[test]
fn the_wall_is_imposed_before_the_old_velocity_is_stored() {
    for (r, expected) in [(Fix128::ZERO, int(1)), (Fix128::ONE, int(4))] {
        let mut s = solver(6, 6, 6);
        s.grid.set_closed_box_walls();
        s.gravity = Vec3Fix::ZERO;
        s.dynamic_viscosity_pas = Fix128::ZERO;
        s.jacobi_iterations = 0;
        let mut ps = lattice(6, 6, 6, v3(int(4), Fix128::ZERO, Fix128::ZERO));
        let t = 2 * 12 + 2; // a = 0, b = 2, c = 2
        assert_eq!(ps[t].0, v3(q(1, 4), q(5, 4), q(5, 4)));
        s.step_flip(&mut ps, q(1, 64), r);
        assert_eq!(ps[t].1.x, expected, "r={r}");
    }
}

/// The viscous term runs between the body force and the projection, as in
/// `step`. Closed box, uniform `u = 4` plug, `nu = mu / rho = 2`, `dt = 1/16`,
/// `dx = 1`, so `c = nu dt / dx^2 = 1/8`; projection off. The u-face
/// `(3, 0, 2)` touches the no-slip floor: its down-neighbour is the ghost
/// `2 u_wall - u = -4`, the other five neighbours are `4`, so
/// `lap = (5 * 4 - 4) - 6 * 4 = -8`, and `u' = u + c lap = 4 - 1 = 3`. A face
/// away from every wall has `lap = 0` and keeps `4`.
#[test]
fn molecular_diffusion_acts_on_the_transferred_velocity() {
    let mut s = solver(6, 6, 6);
    s.grid.set_closed_box_walls();
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = int(2000);
    s.jacobi_iterations = 0;
    let mut ps = lattice(6, 6, 6, v3(int(4), Fix128::ZERO, Fix128::ZERO));
    s.step_flip(&mut ps, q(1, 16), Fix128::ONE);
    assert_eq!(s.grid.u(3, 0, 2), int(3));
    assert_eq!(s.grid.u(3, 2, 2), int(4));
}

// ---------------------------------------------------------------------------
// 4. A step where u_new - u_old is not zero
// ---------------------------------------------------------------------------

/// Open box (no face conditions, `p = 0` outside), uniform stream
/// `U0 = (3/2, 1, -1/2)`, gravity `g = (0, -8, 0)`, `dt = 1/16`, so `g dt` is
/// `-1/2` on y. A uniform field has zero divergence in every cell, so the
/// pressure stays `0` and the grid change is exactly the body force:
/// `u_new - u_old = (0, -1/2, 0)`. Both blends then give `U0 + g dt` for every
/// `r`; a sign error on the increment, an exchange of the two `G2P` calls, or
/// a missed clear of the stale grid each give a different number.
#[test]
fn a_free_fall_step_gives_every_particle_u0_plus_g_dt() {
    let n = 4usize;
    let u0 = v3(q(3, 2), Fix128::ONE, q(-1, 2));
    let g = v3(Fix128::ZERO, int(-8), Fix128::ZERO);
    let dt = q(1, 16);
    let expected_v = v3(u0.x + g.x * dt, u0.y + g.y * dt, u0.z + g.z * dt);
    for r in [Fix128::ZERO, q(1, 2), Fix128::ONE] {
        let mut s = solver(n, n, n);
        s.gravity = g;
        s.dynamic_viscosity_pas = Fix128::ZERO;
        // Stale grid content that must not survive the transfer.
        s.grid.u.iter_mut().for_each(|x| *x = int(11));
        s.grid.v.iter_mut().for_each(|x| *x = int(-13));
        s.grid.w.iter_mut().for_each(|x| *x = int(17));
        let initial = lattice(n as i64, n as i64, n as i64, u0);
        let mut ps = initial.clone();
        s.step_flip(&mut ps, dt, r);
        for (p0, p1) in initial.iter().zip(ps.iter()) {
            assert_eq!(p1.1, expected_v, "r={r}");
            assert_eq!(p1.0.x, p0.0.x + expected_v.x * dt);
            assert_eq!(p1.0.y, p0.0.y + expected_v.y * dt);
            assert_eq!(p1.0.z, p0.0.z + expected_v.z * dt);
        }
        assert!(s.grid.u.iter().all(|&x| x == expected_v.x), "r={r}: u grid");
        assert!(s.grid.v.iter().all(|&x| x == expected_v.y), "r={r}: v grid");
        assert!(s.grid.w.iter().all(|&x| x == expected_v.z), "r={r}: w grid");
        assert_eq!(s.step_count, 1);
    }
}

/// The grid is cleared before the transfer, not only overwritten where a
/// particle reaches. Particles fill only the cells `x < 2` of a `4^3` box (a
/// partial fill is outside the stated contract, so only the *transfer* is
/// asserted, with the projection off): the x-faces `i = 3, 4` are reached by
/// nobody and, after the clear and a body force `g_x dt = -1/2`, must read
/// exactly `-1/2`, not `stale - 1/2`.
#[test]
fn a_partial_lattice_leaves_unreached_faces_cleared_not_stale() {
    let n = 4usize;
    let mut s = solver(n, n, n);
    s.gravity = v3(int(-8), int(-8), int(-8));
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.jacobi_iterations = 0;
    s.grid.u.iter_mut().for_each(|x| *x = int(7));
    s.grid.v.iter_mut().for_each(|x| *x = int(-9));
    s.grid.w.iter_mut().for_each(|x| *x = int(5));
    let mut ps = lattice(2, 4, 4, v3(int(1), Fix128::ZERO, Fix128::ZERO));
    s.step_flip(&mut ps, q(1, 16), Fix128::ONE);
    for k in 0..n {
        for j in 0..n {
            assert_eq!(s.grid.u(4, j, k), q(-1, 2), "u(4,{j},{k})");
            assert_eq!(s.grid.u(3, j, k), q(-1, 2), "u(3,{j},{k})");
            // v / w faces at i = 3 are reached by no particle either
            // (their stagger puts the last reached face at i = 2).
            assert_eq!(s.grid.v(3, j, k), q(-1, 2), "v(3,{j},{k})");
            assert_eq!(s.grid.w(3, j, k), q(-1, 2), "w(3,{j},{k})");
        }
    }
}

// ---------------------------------------------------------------------------
// 4b. The clamp, on every axis and both ends
// ---------------------------------------------------------------------------

/// Open box, uniform stream `(6, 8, -10)`, no force, no viscosity: divergence
/// free, so nothing but the advection acts. With `dt = 1/16` the particles
/// move `(3/8, 1/2, -5/8)` per step; the expected position is the running
/// `x <- min(max(x + v dt, 0), N)` per axis, written here independently of the
/// solver. The x and y ends reached are the upper walls, the z end the lower.
/// One step only: a stream this fast opens a gap in the lattice within two
/// steps (a face nobody reaches), which is outside the stated contract.
#[test]
fn the_clamp_holds_particles_on_the_walls_on_every_axis() {
    let n = 4i64;
    let dt = q(1, 16);
    let vel = v3(int(6), int(8), int(-10));
    for r in [Fix128::ZERO, Fix128::ONE] {
        let mut s = solver(n as usize, n as usize, n as usize);
        s.gravity = Vec3Fix::ZERO;
        s.dynamic_viscosity_pas = Fix128::ZERO;
        let mut ps = lattice(n, n, n, vel);
        let mut expected: Vec<Vec3Fix> = ps.iter().map(|p| p.0).collect();
        let step_axis = |x: Fix128, v: Fix128| {
            let moved = x + v * dt;
            if moved > int(n) {
                int(n)
            } else if moved < Fix128::ZERO {
                Fix128::ZERO
            } else {
                moved
            }
        };
        for _ in 0..1 {
            s.step_flip(&mut ps, dt, r);
            for e in expected.iter_mut() {
                *e = v3(
                    step_axis(e.x, vel.x),
                    step_axis(e.y, vel.y),
                    step_axis(e.z, vel.z),
                );
            }
            for (p, e) in ps.iter().zip(&expected) {
                assert_eq!(p.0, *e, "r={r}");
                assert_eq!(p.1, vel, "r={r}");
            }
        }
        let on = |f: &dyn Fn(&Particle) -> bool| ps.iter().filter(|p| f(p)).count();
        assert!(on(&|p| p.0.x == int(n)) > 0, "x wall never reached");
        assert!(on(&|p| p.0.y == int(n)) > 0, "y wall never reached");
        assert!(on(&|p| p.0.z == Fix128::ZERO) > 0, "z wall never reached");
    }
}

// ---------------------------------------------------------------------------
// 5. Determinism
// ---------------------------------------------------------------------------

#[test]
fn identical_input_gives_bit_identical_particles_and_grid() {
    let run = || {
        let mut s = solver(4, 4, 4);
        s.grid.set_closed_box_walls();
        s.jacobi_iterations = 20;
        let mut ps = lattice(4, 4, 4, Vec3Fix::ZERO);
        // A non-trivial velocity field so the projection has work to do.
        for (n, p) in ps.iter_mut().enumerate() {
            p.1 = v3(
                q((n % 7) as i64 - 3, 8),
                q((n % 5) as i64 - 2, 8),
                q((n % 3) as i64 - 1, 8),
            );
        }
        for _ in 0..3 {
            s.step_flip(&mut ps, q(1, 64), q(1, 2));
        }
        (ps, grid_bits(&s.grid), s.step_count)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(a.2, 3);
}

// ---------------------------------------------------------------------------
// 6. Degenerate input (panic tests, with teeth: see the mutation table in the
//    commit notes)
// ---------------------------------------------------------------------------

/// Run `step_flip` and require that it neither panics nor changes anything.
fn assert_untouched(mut s: CfdSolver, mut ps: Vec<Particle>, dt: Fix128, r: Fix128, what: &str) {
    // Stale content, so a grid that is cleared or rewritten shows up.
    s.grid.u.iter_mut().for_each(|x| *x = int(5));
    s.grid.v.iter_mut().for_each(|x| *x = int(-3));
    s.grid.w.iter_mut().for_each(|x| *x = int(2));
    s.grid.pressure.iter_mut().for_each(|x| *x = int(7));
    let grid0 = grid_bits(&s.grid);
    let ps0 = ps.clone();
    let steps0 = s.step_count;
    let res = catch_unwind(AssertUnwindSafe(|| s.step_flip(&mut ps, dt, r)));
    assert!(res.is_ok(), "{what}: step_flip panicked");
    assert_eq!(grid_bits(&s.grid), grid0, "{what}: grid changed");
    assert_eq!(ps, ps0, "{what}: particles changed");
    assert_eq!(s.step_count, steps0, "{what}: step_count changed");
}

/// A `3^3` lattice plus one particle on the origin, which lies inside the box
/// of every grid below, a degenerate (`dx = 0`, zero-sized) one included, so
/// that the in-domain filter cannot be what makes the call a no-op.
fn small_lattice() -> Vec<Particle> {
    let vel = v3(Fix128::ONE, q(1, 2), Fix128::ZERO);
    let mut ps = lattice(3, 3, 3, vel);
    ps.push((Vec3Fix::ZERO, vel));
    ps
}

#[test]
fn an_empty_particle_list_returns_with_the_grid_untouched() {
    let mut s = solver(3, 3, 3);
    s.grid.u.iter_mut().for_each(|x| *x = int(5));
    assert_untouched(s, Vec::new(), q(1, 16), q(1, 2), "no particles");
}

#[test]
fn dt_zero_and_negative_dt_return_unchanged() {
    assert_untouched(
        solver(3, 3, 3),
        small_lattice(),
        Fix128::ZERO,
        q(1, 2),
        "dt = 0",
    );
    assert_untouched(
        solver(3, 3, 3),
        small_lattice(),
        q(-1, 16),
        q(1, 2),
        "dt < 0",
    );
}

#[test]
fn zero_dx_and_zero_sized_grids_return_unchanged() {
    let s = CfdSolver::new(3, 3, 3, Fix128::ZERO);
    assert_untouched(s, small_lattice(), q(1, 16), q(1, 2), "dx = 0");
    for dims in [(0, 3, 3), (3, 0, 3), (3, 3, 0)] {
        let s = CfdSolver::new(dims.0, dims.1, dims.2, Fix128::ONE);
        assert_untouched(s, small_lattice(), q(1, 16), q(1, 2), "zero-sized grid");
    }
}

/// Every particle outside the box: nothing to transfer, so the stale grid must
/// not be cleared either.
#[test]
fn a_list_of_only_outside_particles_returns_unchanged() {
    let ps = vec![
        (v3(q(-1, 2), q(1, 2), q(1, 2)), v3(int(1), int(2), int(3))),
        (v3(int(100), q(1, 2), q(1, 2)), v3(int(1), int(2), int(3))),
    ];
    assert_untouched(solver(3, 3, 3), ps, q(1, 16), q(1, 2), "only strays");
}

#[test]
fn zero_density_returns_unchanged() {
    let mut s = solver(3, 3, 3);
    s.density_kg_m3 = Fix128::ZERO;
    assert_untouched(s, small_lattice(), q(1, 16), q(1, 2), "rho = 0");
}

#[test]
fn flip_ratio_outside_zero_one_returns_unchanged() {
    for r in [q(-1, 4), q(5, 4), int(100), int(-100)] {
        assert_untouched(solver(3, 3, 3), small_lattice(), q(1, 16), r, "flip_ratio");
    }
}

/// Particles outside `[0, N dx]` (negative coordinates included, and ones
/// just beyond the far side that `p2g_normalized` would partly accept) take no
/// part: the result, grid and in-domain particles, equals the run without
/// them bit for bit, and the strays stay exactly where they were.
#[test]
fn particles_outside_the_domain_are_ignored_and_left_untouched() {
    let n = 3usize;
    let strays = [
        (v3(q(-1, 2), q(3, 2), q(3, 2)), v3(int(9), int(9), int(9))),
        (v3(q(3, 2), q(-1, 4), q(3, 2)), v3(int(-9), int(9), int(9))),
        (v3(q(3, 2), q(3, 2), q(-3, 1)), v3(int(9), int(-9), int(9))),
        (v3(q(7, 2), q(3, 2), q(3, 2)), v3(int(9), int(9), int(-9))),
        (v3(q(3, 2), q(7, 2), q(3, 2)), v3(int(5), int(5), int(5))),
        (v3(q(3, 2), q(3, 2), q(13, 4)), v3(int(5), int(-5), int(5))),
        (
            v3(int(-1_000_000), int(1_000_000), q(1, 2)),
            v3(int(1), int(1), int(1)),
        ),
    ];
    let run = |with_strays: bool| {
        let mut s = solver(n, n, n);
        s.grid.set_closed_box_walls();
        s.jacobi_iterations = 20;
        let mut ps = lattice(
            n as i64,
            n as i64,
            n as i64,
            v3(Fix128::ONE, Fix128::ZERO, q(1, 2)),
        );
        let inside = ps.len();
        if with_strays {
            ps.extend_from_slice(&strays);
        }
        let res = catch_unwind(AssertUnwindSafe(|| s.step_flip(&mut ps, q(1, 64), q(1, 2))));
        assert!(res.is_ok(), "stray particle made step_flip panic");
        (ps, inside, grid_bits(&s.grid))
    };
    let (with, inside, grid_with) = run(true);
    let (without, _, grid_without) = run(false);
    assert_eq!(grid_with, grid_without, "strays leaked into the grid");
    assert_eq!(
        with[..inside],
        without[..],
        "strays changed the in-domain particles"
    );
    assert_eq!(with[inside..], strays[..], "strays were modified");
}

/// Every particle at one point: eight faces get all the weight and the rest
/// stay unreached. Must not panic; with the projection off and `r = 1` the
/// particles keep their velocity.
#[test]
fn all_particles_at_one_point_do_not_panic() {
    let mut s = solver(4, 4, 4);
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.jacobi_iterations = 0;
    let p = (v3(q(3, 2), q(3, 2), q(3, 2)), v3(int(2), int(-1), q(1, 2)));
    let mut ps = vec![p; 50];
    let res = catch_unwind(AssertUnwindSafe(|| {
        s.step_flip(&mut ps, q(1, 16), Fix128::ONE)
    }));
    assert!(res.is_ok());
    for &(_, v) in &ps {
        assert_eq!(v, p.1);
    }
    // And with the projection on (the divergent pile is the worst case).
    let mut s = solver(4, 4, 4);
    let mut ps = vec![p; 50];
    let res = catch_unwind(AssertUnwindSafe(|| s.step_flip(&mut ps, q(1, 16), q(1, 2))));
    assert!(res.is_ok(), "same-point pile panicked with projection on");
}

/// Coordinates and velocities of order `i64::MAX / 4`. `Fix128` wraps mod 2^128
/// instead of panicking, so "does not panic" proves nothing; the expected
/// values are fixed instead:
///
/// * a particle outside the box is not touched at all (position and velocity
///   bit for bit): it deposits nothing and the grid never reaches it
/// * an in-domain particle with a huge velocity leaves the box within one step
///   (|v|*dt is far above the box), so the advection clamp puts it on the
///   corner its velocity signs point to: `+` -> the far wall (3), `-` -> 0
/// * its velocity stays within 1e-3 relative of the input and keeps its sign.
///   The transfer, projection and blend only redistribute a velocity of
///   this size, a wrapped intermediate would be off by orders of magnitude
///   (the measured drift is ~1.5e-7)
#[test]
fn huge_positions_and_velocities_have_pinned_results() {
    let big = int(i64::MAX / 4);
    let mut s = solver(3, 3, 3);
    let mut ps = vec![
        (v3(big, big, big), v3(big, big, big)),
        (v3(-big, q(1, 2), q(1, 2)), v3(big, big, big)),
        (v3(q(3, 2), q(3, 2), q(3, 2)), v3(big, -big, big)),
        (v3(q(1, 2), q(1, 2), q(1, 2)), v3(-big, big, -big)),
    ];
    let before = ps.clone();
    let res = catch_unwind(AssertUnwindSafe(|| {
        s.step_flip(&mut ps, q(1, 16), q(1, 2));
    }));
    assert!(res.is_ok(), "huge input made step_flip panic");
    // Outside the box: bit-for-bit untouched.
    assert_eq!(ps[0], before[0], "far-side particle was modified");
    assert_eq!(ps[1], before[1], "left-of-domain particle was modified");
    // Inside the box: clamped to the corner the velocity points at.
    assert_eq!(ps[2].0, v3(int(3), Fix128::ZERO, int(3)));
    assert_eq!(ps[3].0, v3(Fix128::ZERO, int(3), Fix128::ZERO));
    // Velocity: sign kept, magnitude within 1e-3 relative of the input.
    let tol = big.to_f64() * 1e-3;
    for idx in [2usize, 3] {
        let (got, want) = (ps[idx].1, before[idx].1);
        for (c, g, w) in [
            ("x", got.x, want.x),
            ("y", got.y, want.y),
            ("z", got.z, want.z),
        ] {
            assert_eq!(
                g.to_f64().signum(),
                w.to_f64().signum(),
                "particle {idx} v.{c}: sign flipped ({g:?})"
            );
            assert!(
                (g.to_f64() - w.to_f64()).abs() <= tol,
                "particle {idx} v.{c}: {} vs input {}",
                g.to_f64(),
                w.to_f64()
            );
        }
    }
}
