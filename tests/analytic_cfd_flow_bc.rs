//! Tangential no-slip and the inflow / outflow face conditions
//! (`src/eulerian_grid.rs`, `src/cfd_solver.rs`).
//!
//! The face mask that landed first said *where* a wall is, which is what the
//! pressure projection needs. It did not say what the wall does to the
//! velocity along it, and it had no way to say "fluid enters here" — so the
//! lid-driven cavity oracle had to add the viscous wall term from the test
//! side, and a duct could not be written down at all.
//!
//! Every expected value below is a closed form derived here, never a number
//! read off a run:
//!
//! - **Poiseuille.** Plane channel of height `H`, both walls no-slip, driven
//!   by a uniform body force `G` (`ρ = 1`). Steady state is `ν u'' + G = 0`,
//!   so `u(y) = (G / 2ν) · y (H − y)`.
//! - **The same channel discretely.** `u` lives at `y_j = (j + ½) dx` and the
//!   wall ghost is `u_−1 = −u_0`. Write the fixed point as the parabola plus
//!   a constant, `u_j = f(y_j) + B`, and read off what `B` has to be:
//!   1. the second difference of a quadratic is exact, so `f` satisfies every
//!      **interior** row on its own, and a constant satisfies them too;
//!   2. at the wall row the ghost `−f(h)` differs from the parabola's own
//!      value `f(−h)` by `+2 A h²`, with `A = G/2ν` and `h = dx/2`, and the
//!      constant `B` shifts the same row by `−2B` (interior rows cannot see
//!      a constant, the wall row can — that asymmetry is the whole effect);
//!   3. the solver applies the body force **before** the viscous term, so
//!      the Laplacian is evaluated at `u + G dt`; by the same argument that
//!      uniform `G dt` shows up only at the wall row, as `−2 G dt`.
//!
//!   Balancing (2) against (3): `2 A h² − 2B = 2 G dt`, i.e.
//!
//!   ```text
//!   u_j = (G / 2ν) · y_j (H − y_j)  +  G dx² / (8 ν)  −  G dt
//!   ```
//!
//!   Term by term, with where each one comes from:
//!
//!   | term | origin | provenance |
//!   |---|---|---|
//!   | `(G/2ν) y_j (H − y_j)` | the continuum solution of `ν u'' + G = 0` | analytic |
//!   | `+ G dx² / (8 ν)` | spatial discretisation: `A h²` from step (2), the `u_−1 = −u_0` wall ghost against a parabola that is not odd about the wall | analytic |
//!   | `− G dt` | operator splitting: the `G dt` of step (3), because the body force is added before the viscous term and only the wall row can see a constant | analytic |
//!
//!   Everything printed by the tests below — `3.206e-9`, `0.007812499`,
//!   `1.033`, `4.841e-5` — is **measured**, and every threshold is a
//!   convergence or arithmetic allowance, never a fitted discretisation error.
//!
//!   ⚠️ **`dt = dx²/(8ν)` is a forbidden parameter point for every test in
//!   this file, and not because it is unstable.** At that step the `dx²` term
//!   and the `dt` term are equal and opposite, `B = 0`, and the run lands
//!   *exactly* on the continuum parabola — which looks like the strongest
//!   possible evidence of correctness and is in fact two unverified errors
//!   cancelling. The first draft of this file used `ny = 8`, `ν = 1/8`,
//!   `dt = 1/64`, which is precisely that point. Do not "tidy" the step size
//!   back to a rounder number: with `B = 0` neither term is pinned any more,
//!   and an oracle that cannot see either error is not measuring whether
//!   no-slip reached the solver. The tests use `dt = dx²/(16ν)`, which leaves
//!   `B = +G dx²/(16ν)` — the splitting term at exactly half the spatial one,
//!   so a mistake in either shows up.
//! - **Free slip.** With a zero-gradient mirror at both walls the Laplacian
//!   vanishes identically, so `u` is uniform and grows as `G t` for ever.
//!   That is the control: the no-slip result is not "a bit different", it is
//!   a different solution.
//! - **Duct.** Whatever enters through the prescribed faces has to leave
//!   through the Dirichlet-pressure ones, because summing `∇·u = 0` over
//!   every cell telescopes into the net flux across the boundary.

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

// ===========================================================================
// Shared setup
// ===========================================================================

/// Kinematic viscosity `ν = 1/8` (with `ρ = 1`), exactly representable.
const NU_RECIPROCAL: i64 = 8;

/// The two axes that are not `axis`, in increasing order.
fn other_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Index triple of a face of component `axis` sitting at `p` along that axis
/// and at `(q, r)` along [`other_axes`].
fn face_index(axis: usize, p: usize, q: usize, r: usize) -> (usize, usize, usize) {
    match axis {
        0 => (p, q, r),
        1 => (q, p, r),
        _ => (q, r, p),
    }
}

/// Cell count along `axis`.
fn extent(grid: &MacGrid, axis: usize) -> usize {
    match axis {
        0 => grid.nx,
        1 => grid.ny,
        _ => grid.nz,
    }
}

fn set_face_bc(grid: &mut MacGrid, axis: usize, p: usize, q: usize, r: usize, bc: FaceBc) {
    let (i, j, k) = face_index(axis, p, q, r);
    match axis {
        0 => grid.set_u_bc(i, j, k, bc),
        1 => grid.set_v_bc(i, j, k, bc),
        _ => grid.set_w_bc(i, j, k, bc),
    }
}

fn read_face(grid: &MacGrid, axis: usize, p: usize, q: usize, r: usize) -> Fix128 {
    let (i, j, k) = face_index(axis, p, q, r);
    match axis {
        0 => grid.u(i, j, k),
        1 => grid.v(i, j, k),
        _ => grid.w(i, j, k),
    }
}

/// Mark both boundary face layers normal to `axis` with `bc`.
///
/// Used for the two no-slip walls of the channel and for the two symmetry
/// planes of the direction the problem is uniform in. The symmetry planes must
/// still block flow — otherwise the pressure Poisson problem degenerates into
/// the screened one a fixed `1/6` divisor produces — but they are not walls of
/// the physical problem, so they must not exert shear, which is what
/// [`FaceBc::SlipWall`] says.
fn mark_boundary_layers(grid: &mut MacGrid, axis: usize, bc: FaceBc) {
    let (o0, o1) = other_axes(axis);
    let (n0, n1) = (extent(grid, o0), extent(grid, o1));
    let n = extent(grid, axis);
    for q in 0..n0 {
        for r in 0..n1 {
            set_face_bc(grid, axis, 0, q, r, bc);
            set_face_bc(grid, axis, n, q, r, bc);
        }
    }
}

/// Largest `|∇·u|` over every cell, rim included.
fn max_abs_divergence(grid: &MacGrid) -> f64 {
    let mut worst = 0.0f64;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                worst = worst.max(grid.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    worst
}

/// The six ways to lay a plane channel out on the grid: `(flow axis, wall
/// axis)`, the third axis carrying the symmetry planes.
///
/// ⚠️ All six exist because a single orientation exercises exactly one of the
/// six `MacGrid::{u,v,w}_wall_across_{x,y,z}` lookups and one of the three
/// component branches of the viscous term. Measured: with only the
/// `x`-flow / `y`-wall case, deleting the no-slip ghost from the `v` **and**
/// from the `w` branch left every oracle in this crate green — the `w` ghost
/// never even ran on a non-zero value, because a quasi-2-D scene holds `w` at
/// zero, so "it is zero" was true for the wrong reason.
const CHANNEL_LAYOUTS: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];

/// Name of an axis, for the test output.
fn axis_name(axis: usize) -> char {
    ['x', 'y', 'z'][axis]
}

/// Run a plane channel driven by the body force `G = 1` along `flow_axis`,
/// walled along `wall_axis`, and return the velocity profile across the walls
/// plus the worst divergence.
///
/// `wall` is the condition on the two wall layers: [`FaceBc::Wall`] for the
/// no-slip channel, [`FaceBc::SlipWall`] for the free-slip control. The
/// boundary faces along the flow are left [`FaceBc::Fluid`], i.e. open with
/// exterior `p = 0`; the field is uniform along the flow, so the projection
/// has nothing to do and the run is the one-dimensional problem the closed
/// form describes. The remaining axis is one cell thick with symmetry planes.
fn run_plane_channel(
    n_wall: usize,
    dt_reciprocal: i64,
    steps: u32,
    wall: FaceBc,
    flow_axis: usize,
    wall_axis: usize,
) -> (Vec<f64>, f64) {
    assert_ne!(flow_axis, wall_axis);
    let slip_axis = 3 - flow_axis - wall_axis;
    let mut dims = [0usize; 3];
    dims[flow_axis] = 3;
    dims[wall_axis] = n_wall;
    dims[slip_axis] = 1;

    let dx = Fix128::from_ratio(1, n_wall as i64);
    let mut gravity = [Fix128::ZERO; 3];
    gravity[flow_axis] = Fix128::ONE; // G = 1

    let mut solver = CfdSolver::new(dims[0], dims[1], dims[2], dx);
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::from_ratio(1, NU_RECIPROCAL); // ρ = 1, so μ = ν
    solver.gravity = Vec3Fix::new(gravity[0], gravity[1], gravity[2]);
    solver.jacobi_iterations = 10;
    solver.use_turbulence = false;
    mark_boundary_layers(&mut solver.grid, wall_axis, wall);
    mark_boundary_layers(&mut solver.grid, slip_axis, FaceBc::SlipWall);

    let dt = Fix128::from_ratio(1, dt_reciprocal);
    for _ in 0..steps {
        solver.step(dt);
    }

    // Sample the flow component halfway along the flow, across the walls.
    let (o0, _) = other_axes(flow_axis);
    let mid = dims[flow_axis] / 2;
    let profile = (0..n_wall)
        .map(|m| {
            let (q, r) = if o0 == wall_axis { (m, 0) } else { (0, m) };
            read_face(&solver.grid, flow_axis, mid, q, r).to_f64()
        })
        .collect();
    (profile, max_abs_divergence(&solver.grid))
}

/// The constant the fixed point of the split scheme sits above the continuum
/// parabola: `G dx²/(8ν) − G dt`, with `G = 1`. See the module header.
fn channel_offset(ny: usize, dt_reciprocal: i64) -> f64 {
    let dx = 1.0 / ny as f64;
    let nu = 1.0 / NU_RECIPROCAL as f64;
    dx * dx / (8.0 * nu) - 1.0 / dt_reciprocal as f64
}

/// `u(y_j)` of the fixed point the solver must reach: continuum parabola plus
/// [`channel_offset`]. `G = H = 1`.
fn discrete_poiseuille(ny: usize, dt_reciprocal: i64, j: usize) -> f64 {
    continuum_poiseuille(ny, j) + channel_offset(ny, dt_reciprocal)
}

/// `u(y_j)` of the continuum parabola alone, for the convergence statement.
fn continuum_poiseuille(ny: usize, j: usize) -> f64 {
    let y = (j as f64 + 0.5) / ny as f64;
    let nu = 1.0 / NU_RECIPROCAL as f64;
    y * (1.0 - y) / (2.0 * nu)
}

// ===========================================================================
// Oracle 1 — no-slip reproduces the Poiseuille parabola
// ===========================================================================

/// Oracle: the steady state of a body-force-driven plane channel with
/// no-slip walls is `u(y) = (G/2ν) y (H − y)` plus the two closed-form
/// offsets derived in the module header.
///
/// The tolerance is the distance left to the fixed point after `t = 16` (the
/// slowest mode decays as `exp(−ν π² t) = 2.7e-9`) plus the fixed-point
/// arithmetic — **not** the discretisation error, which is predicted rather
/// than tolerated. The distance to the continuum parabola is checked against
/// the same closed form, which is what makes the deviation an explained
/// quantity instead of a tuned threshold.
///
/// `dt = dx²/(16ν)`, deliberately half of the step at which the spatial and
/// the splitting error cancel; at the cancelling step the run lands on the
/// continuum parabola while both errors are still there.
#[test]
fn no_slip_channel_reaches_the_poiseuille_parabola() {
    let n_wall = 8usize;
    let dt_reciprocal = 128i64;
    let offset = channel_offset(n_wall, dt_reciprocal);
    println!(
        "plane channel, n={n_wall}, dt=1/{dt_reciprocal}, t = 16, closed-form offset \
         {offset:.9} (analytic)"
    );

    for &(flow_axis, wall_axis) in &CHANNEL_LAYOUTS {
        let (profile, worst_div) = run_plane_channel(
            n_wall,
            dt_reciprocal,
            2048,
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
            flow_axis,
            wall_axis,
        );

        let mut worst_fixed_point = 0.0f64;
        let mut worst_continuum = 0.0f64;
        for (m, &u) in profile.iter().enumerate() {
            worst_fixed_point =
                worst_fixed_point.max((u - discrete_poiseuille(n_wall, dt_reciprocal, m)).abs());
            worst_continuum = worst_continuum.max((u - continuum_poiseuille(n_wall, m)).abs());
        }
        println!(
            "  flow {} / walls {}: max |u − predicted| {worst_fixed_point:.3e} (measured), \
             max |u − continuum| {worst_continuum:.9} (measured), max |div| {worst_div:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );

        assert!(
            worst_fixed_point < 1e-6,
            "flow {} / walls {}: the no-slip channel must settle on the closed-form \
             profile, worst deviation {worst_fixed_point:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );
        // The gap to the continuum solution is an explained quantity, not an
        // unexplained residue: it must equal the closed form, not merely be
        // small. A zero-gradient mirror would leave no parabola to compare.
        assert!(
            (worst_continuum - offset.abs()).abs() < 1e-6,
            "flow {} / walls {}: the gap to the continuum parabola must be the \
             predicted G dx²/(8ν) − G dt = {offset:.9}, measured {worst_continuum:.9}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );
        assert!(
            worst_div < 1e-9,
            "flow {} / walls {}: a unidirectional channel is divergence free by \
             construction, worst |div| {worst_div:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );
    }
}

// ===========================================================================
// Oracle 2 — free slip is a different solution, not a slightly worse one
// ===========================================================================

/// Control for oracle 1. With free-slip walls the discrete Laplacian of a
/// `y`-uniform field is identically zero, so the body force is unopposed and
/// `u(t) = G t` exactly — `16` after `1024` steps of `dt = 1/64`.
///
/// This is what the solver did at *every* wall before the no-slip ghost
/// landed, which is why the cavity oracle had to add the viscous wall term
/// from the test side. It also keeps oracle 1 honest: a mirror that quietly
/// reverted to zero gradient would show up here as `16` and there as a flat
/// profile, not as a small numerical drift.
#[test]
fn free_slip_channel_accelerates_without_bound() {
    let n_wall = 8usize;
    let steps = 1024u32;
    let dt_reciprocal = 64i64;
    let expected = f64::from(steps) / dt_reciprocal as f64; // G · t, analytic

    for &(flow_axis, wall_axis) in &CHANNEL_LAYOUTS {
        let (profile, worst_div) = run_plane_channel(
            n_wall,
            dt_reciprocal,
            steps,
            FaceBc::SlipWall,
            flow_axis,
            wall_axis,
        );
        let worst = profile
            .iter()
            .fold(0.0f64, |acc, &u| acc.max((u - expected).abs()));
        println!(
            "free-slip, flow {} / walls {}: u {:+.9} everywhere (expected G t = \
             {expected:+.3} analytic), spread {worst:.3e}, max |div| {worst_div:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis),
            profile[0]
        );

        assert!(
            worst < 1e-9,
            "flow {} / walls {}: with free-slip walls the profile must stay flat at \
             G t = {expected}, worst deviation {worst:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );
        assert!(
            worst_div < 1e-9,
            "flow {} / walls {}: the free-slip control must also stay divergence free, \
             worst |div| {worst_div:.3e}",
            axis_name(flow_axis),
            axis_name(wall_axis)
        );
    }
}

// ===========================================================================
// Oracle 3 — second order in the cell size
// ===========================================================================

/// Oracle: the distance from the continuum parabola is `G dx²/(8ν) − G dt`,
/// so with `dt` held at `dx²/(16ν)` — the explicit viscous limit scales the
/// same way, there is no choice about this — the whole error is `G dx²/(16ν)`
/// and halving the cell size quarters it.
///
/// Every row runs to the same physical time, so the rows differ only in the
/// cell size. Both terms of the closed form are exercised: the spatial one
/// through the `dx²` scaling, the splitting one because it is half of what
/// is being measured.
///
/// `#[ignore]`d **for runtime only** — the `ny = 32` row is `32768` steps — not
/// because the accuracy it asks for is out of reach; the measured ratios are in
/// the commit that added this file. Run it with
/// `cargo test --release --test analytic_cfd_flow_bc -- --ignored --nocapture`.
#[test]
#[ignore = "runtime only: 43k steps across three resolutions, release-only; the accuracy it asks for is reached, see the doc comment"]
fn channel_error_is_second_order_in_the_cell_size() {
    let mut previous: Option<(usize, f64)> = None;
    for &(ny, dt_reciprocal, steps) in &[
        (8usize, 128i64, 2048u32),
        (16, 512, 8192),
        (32, 2048, 32768),
    ] {
        let (profile, worst_div) = run_plane_channel(
            ny,
            dt_reciprocal,
            steps,
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
            0,
            1,
        );
        let error = profile.iter().enumerate().fold(0.0f64, |acc, (j, &u)| {
            acc.max((u - continuum_poiseuille(ny, j)).abs())
        });
        let predicted = channel_offset(ny, dt_reciprocal).abs();
        println!(
            "ny={ny:3}  dt=1/{dt_reciprocal:<5}  max |u − continuum| {error:.6e}  \
             closed form {predicted:.6e}  max |div| {worst_div:.3e}"
        );
        assert!(
            (error - predicted).abs() < 1e-6,
            "ny={ny}: the error must be the predicted {predicted:.6e}, measured {error:.6e}"
        );
        if let Some((prev_ny, prev_error)) = previous {
            let ratio = prev_error / error;
            println!("  ratio to ny={prev_ny}: {ratio:.3} (second order = 4)");
            assert!(
                (ratio - 4.0).abs() < 0.05,
                "halving the cell size must quarter the error, ratio {ratio:.3}"
            );
        }
        previous = Some((ny, error));
    }
}

// ===========================================================================
// Oracle 4 — a duct conserves mass between inflow and outflow
// ===========================================================================

/// Build a plane duct: no-slip walls top and bottom, symmetry planes in `z`,
/// a prescribed `u = inflow` across the whole `i = 0` layer and, when
/// `mark_outflow`, an outflow at `i = nx` (otherwise that layer is left as a
/// plain open rim face, which is the control for oracle 6).
fn duct_with(nx: usize, ny: usize, inflow: Fix128, mark_outflow: bool) -> CfdSolver {
    let dx = Fix128::from_ratio(1, ny as i64);
    let mut solver = CfdSolver::new(nx, ny, 1, dx);
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::from_ratio(1, NU_RECIPROCAL);
    solver.gravity = Vec3Fix::ZERO;
    solver.jacobi_iterations = 60;
    solver.use_turbulence = false;
    mark_boundary_layers(
        &mut solver.grid,
        1,
        FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        },
    );
    mark_boundary_layers(&mut solver.grid, 2, FaceBc::SlipWall);
    for k in 0..solver.grid.nz {
        for j in 0..ny {
            solver.grid.set_u_bc(
                0,
                j,
                k,
                FaceBc::Inflow {
                    normal_velocity: inflow,
                },
            );
            if mark_outflow {
                solver.grid.set_u_bc(nx, j, k, FaceBc::Outflow);
            }
        }
    }
    solver
}

/// The duct every oracle but number 6 uses: outflow marked.
fn duct(nx: usize, ny: usize, inflow: Fix128) -> CfdSolver {
    duct_with(nx, ny, inflow, true)
}

/// Total `u` flux through the face column `i` (the cell area `dx²` is common
/// to every term, so the bare sum is the comparison).
fn column_flux(grid: &MacGrid, i: usize) -> f64 {
    let mut flux = 0.0f64;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            flux += grid.u(i, j, k).to_f64();
        }
    }
    flux
}

/// Oracle: summing `∇·u = 0` over every cell of the duct telescopes into the
/// net flux across its boundary, and the walls carry none, so **whatever
/// enters through the prescribed faces leaves through the outflow**.
///
/// The prescribed value has to survive the step for that to mean anything —
/// advection, the viscous term and the pressure correction all write to the
/// `i = 0` faces — so the inflow is checked for *exact* equality first. A
/// mass balance between two numbers the solver was free to choose would hold
/// for the trivial reason.
#[test]
fn duct_carries_its_inflow_through_to_the_outflow() {
    let (nx, ny) = (12usize, 8usize);
    let inflow = Fix128::from_ratio(1, 2);
    let mut solver = duct(nx, ny, inflow);
    let dt = Fix128::from_ratio(1, 64);

    for _ in 0..64 {
        solver.step(dt);
    }

    // 1. The prescribed faces still carry exactly what was prescribed.
    let mut worst_inflow_drift = Fix128::ZERO;
    for k in 0..solver.grid.nz {
        for j in 0..ny {
            let drift = (solver.grid.u(0, j, k) - inflow).abs();
            if drift > worst_inflow_drift {
                worst_inflow_drift = drift;
            }
        }
    }

    let flux_in = column_flux(&solver.grid, 0);
    let flux_out = column_flux(&solver.grid, nx);
    let worst_div = max_abs_divergence(&solver.grid);
    let expected_in = ny as f64 * inflow.to_f64();
    println!(
        "duct {nx}x{ny}: flux in {flux_in:+.9} (prescribed {expected_in:+.9}), \
         flux out {flux_out:+.9}, imbalance {:.3e}, max |div| {worst_div:.3e}",
        (flux_out - flux_in).abs()
    );
    for j in 0..ny {
        println!(
            "  j={j}  u_in {:+.9}  u_out {:+.9}",
            solver.grid.u(0, j, 0).to_f64(),
            solver.grid.u(nx, j, 0).to_f64()
        );
    }

    assert_eq!(
        worst_inflow_drift,
        Fix128::ZERO,
        "the prescribed inflow must survive advection, diffusion and the \
         pressure correction unchanged"
    );
    // Not vacuous: something is actually flowing.
    assert!(
        (flux_in - expected_in).abs() < 1e-12 && flux_in > 0.0,
        "the inflow flux must be the prescribed {expected_in}, got {flux_in}"
    );
    assert!(
        (flux_out - flux_in).abs() < 1e-6,
        "mass in must equal mass out, imbalance {:.3e}",
        (flux_out - flux_in).abs()
    );
    // Every interior cross-section carries the same flux, for the same
    // reason — this is where an inflow that the Poisson stencil treats as an
    // ordinary open face shows up.
    for i in 1..nx {
        let flux = column_flux(&solver.grid, i);
        assert!(
            (flux - flux_in).abs() < 1e-6,
            "cross-section i={i} carries {flux:+.9}, inflow carries {flux_in:+.9}"
        );
    }
    assert!(
        worst_div < 1e-6,
        "the projection must leave the duct divergence free, \
         worst |div| {worst_div:.3e}"
    );
}

/// Oracle: far enough downstream a plane duct forgets its inlet profile. A
/// uniform inflow of mean `U` develops into the parabola of the same mean,
/// `u(y) = 6 U y (H − y) / H²`, whose peak is `1.5 U` — a flat profile has
/// the ratio `1`, so the ratio alone separates "developed" from "the inlet
/// slid downstream".
///
/// `#[ignore]`d **for runtime only**: `24 × 16` cells for `4096` steps with 200
/// Gauss-Seidel sweeps is minutes in a debug build. The accuracy it asks for is
/// reached; the measured values are in the commit that added this file. Run with
/// `cargo test --release --test analytic_cfd_flow_bc -- --ignored --nocapture`.
#[test]
#[ignore = "runtime only: 4096 steps on 24x16 cells with 200 GS sweeps, release-only; the accuracy it asks for is reached, see the doc comment"]
fn duct_profile_develops_toward_the_parabola() {
    let (nx, ny) = (24usize, 16usize);
    let inflow = Fix128::from_ratio(1, 2);
    let mut solver = duct(nx, ny, inflow);
    solver.jacobi_iterations = 200;
    let dt = Fix128::from_ratio(1, 128);

    for _ in 0..4096 {
        solver.step(dt);
    }

    let u_mean = column_flux(&solver.grid, nx) / ny as f64;
    let mut peak = 0.0f64;
    let mut worst = 0.0f64;
    println!("duct {nx}x{ny} outlet profile, mean {u_mean:+.6}");
    for j in 0..ny {
        let y = (j as f64 + 0.5) / ny as f64;
        let u = solver.grid.u(nx, j, 0).to_f64();
        let parabola = 6.0 * u_mean * y * (1.0 - y);
        peak = peak.max(u);
        worst = worst.max((u - parabola).abs());
        println!("  y={y:.4}  u {u:+.6}  parabola {parabola:+.6}");
    }
    let ratio = peak / u_mean;
    println!("  peak/mean {ratio:.4} (parabola 1.5, plug 1.0), worst |u − parabola| {worst:.4}");

    assert!(
        (ratio - 1.5).abs() < 0.1,
        "a developed plane duct has peak/mean = 1.5, got {ratio:.4}"
    );
    assert!(
        worst < 0.1 * u_mean,
        "the outlet profile must be the parabola of its own mean, \
         worst deviation {worst:.4} against mean {u_mean:.4}"
    );
}

// ===========================================================================
// Oracle 6 — what `Outflow` actually buys over a plain open rim face
// ===========================================================================

/// Oracle: the zero-gradient extrapolation of an [`FaceBc::Outflow`] face is
/// a **better iterate**, not a different condition.
///
/// ⚠️ This oracle exists because the honest answer is uncomfortable. On the
/// domain boundary a plain [`FaceBc::Fluid`] face already couples to the
/// exterior `p = 0`, so the Dirichlet pressure that makes an outflow an
/// outflow was there before the variant was. Measured (this test, 12×8 duct,
/// `dt = 1/64`, 64 steps): the two fields differ by `4.99e-5` with 60
/// Gauss-Seidel sweeps and `1.33e-2` with one. **At convergence they agree.**
///
/// What the extrapolation does buy is falsifiable, so it is what gets gated:
/// starting the projection from `u_nx = u_{nx−1}` leaves less for the
/// pressure solve to do, so with a deliberately under-resolved solve the
/// marked run is closer to closing its mass balance. Measured imbalances
/// against the prescribed inflow of `4.0`: one sweep `0.831` marked vs
/// `0.915` unmarked, five sweeps `0.305` vs `0.339`.
///
/// Deleting the extrapolation from `enforce_face_boundaries` makes
/// [`FaceBc::Outflow`] byte-identical to [`FaceBc::Fluid`] and **no other
/// oracle in this crate notices**, which is what this one is for.
#[test]
fn outflow_extrapolation_is_a_better_iterate_than_a_bare_open_face() {
    let (nx, ny) = (12usize, 8usize);
    let inflow = Fix128::from_ratio(1, 2);
    let expected = ny as f64 * inflow.to_f64();
    let dt = Fix128::from_ratio(1, 64);

    for &sweeps in &[1u32, 5] {
        let mut marked = duct_with(nx, ny, inflow, true);
        let mut bare = duct_with(nx, ny, inflow, false);
        marked.jacobi_iterations = sweeps;
        bare.jacobi_iterations = sweeps;
        for _ in 0..64 {
            marked.step(dt);
            bare.step(dt);
        }
        let gap_marked = (column_flux(&marked.grid, nx) - expected).abs();
        let gap_bare = (column_flux(&bare.grid, nx) - expected).abs();
        let mut field_gap = 0.0f64;
        for j in 0..ny {
            for i in 0..=nx {
                field_gap = field_gap
                    .max((marked.grid.u(i, j, 0).to_f64() - bare.grid.u(i, j, 0).to_f64()).abs());
            }
        }
        println!(
            "{sweeps} Gauss-Seidel sweep(s): outflow imbalance {gap_marked:.6} vs bare open \
             {gap_bare:.6}, field gap {field_gap:.3e}"
        );
        assert!(
            gap_marked < gap_bare,
            "the zero-gradient outflow must leave the pressure solve less to do: \
             imbalance {gap_marked:.6} marked vs {gap_bare:.6} bare, at {sweeps} sweep(s)"
        );
    }

    // The other half of the claim: converge the solve and the distinction
    // washes out. This is the characterisation that stops a reader believing
    // the variant changes the answer.
    let mut marked = duct_with(nx, ny, inflow, true);
    let mut bare = duct_with(nx, ny, inflow, false);
    marked.jacobi_iterations = 200;
    bare.jacobi_iterations = 200;
    for _ in 0..256 {
        marked.step(dt);
        bare.step(dt);
    }
    let mut field_gap = 0.0f64;
    for j in 0..ny {
        for i in 0..=nx {
            field_gap = field_gap
                .max((marked.grid.u(i, j, 0).to_f64() - bare.grid.u(i, j, 0).to_f64()).abs());
        }
    }
    println!("200 sweeps, 256 steps: field gap {field_gap:.3e} (converged, they agree)");
    assert!(
        field_gap < 1e-3,
        "at a converged solve an outflow face and an open rim face must agree, \
         gap {field_gap:.3e}"
    );
}

// ===========================================================================
// Oracle 5 — the type says what the solver does
// ===========================================================================

/// Oracle: the conditions that fix a normal velocity are exactly the ones the
/// pressure problem must drop, and the ones that leave it free are exactly
/// the ones that see the exterior `p = 0`.
///
/// This is the contract the projection is built on, stated where a reader
/// will find it: an inflow is a velocity Dirichlet and therefore a pressure
/// Neumann, an outflow is the other way round.
#[test]
fn prescribed_velocity_means_neumann_pressure() {
    assert!(FaceBc::Wall {
        velocity: Vec3Fix::ZERO
    }
    .blocks_pressure());
    assert!(FaceBc::SlipWall.blocks_pressure());
    assert!(FaceBc::Inflow {
        normal_velocity: Fix128::ONE
    }
    .blocks_pressure());
    assert!(!FaceBc::Outflow.blocks_pressure());
    assert!(!FaceBc::Fluid.blocks_pressure());

    assert!(FaceBc::Wall {
        velocity: Vec3Fix::ZERO
    }
    .is_wall());
    assert!(FaceBc::SlipWall.is_wall());
    assert!(!FaceBc::Inflow {
        normal_velocity: Fix128::ONE
    }
    .is_wall());

    // Only a no-slip wall has a velocity to impose on the viscous term.
    let lid = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    assert_eq!(
        FaceBc::Wall { velocity: lid }.no_slip_velocity(),
        Some(lid),
        "a moving wall must hand its velocity to the no-slip ghost"
    );
    assert_eq!(FaceBc::SlipWall.no_slip_velocity(), None);
    assert_eq!(FaceBc::Outflow.no_slip_velocity(), None);
}

/// Oracle: every condition round-trips through the setters, and the legacy
/// solid flag keeps agreeing with the condition that replaced it.
///
/// Building all five variants from an integration test — a separate crate —
/// is also the check that `#[non_exhaustive]` on [`FaceBc`] has not made them
/// unconstructable downstream.
#[test]
fn face_conditions_round_trip_and_agree_with_the_solid_flag() {
    let mut grid = MacGrid::new(4, 4, 4, Fix128::from_ratio(1, 4));
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    };
    let inflow = FaceBc::Inflow {
        normal_velocity: Fix128::from_ratio(1, 2),
    };

    for bc in [
        FaceBc::Fluid,
        FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        },
        lid,
        FaceBc::SlipWall,
        inflow,
        FaceBc::Outflow,
    ] {
        grid.set_u_bc(1, 2, 3, bc);
        assert_eq!(grid.u_bc(1, 2, 3), bc, "X-face round trip of {bc:?}");
        assert_eq!(
            grid.is_u_solid(1, 2, 3),
            bc.is_wall(),
            "the solid flag must follow {bc:?}"
        );

        grid.set_v_bc(1, 2, 3, bc);
        assert_eq!(grid.v_bc(1, 2, 3), bc, "Y-face round trip of {bc:?}");
        assert_eq!(grid.is_v_solid(1, 2, 3), bc.is_wall());

        grid.set_w_bc(1, 2, 3, bc);
        assert_eq!(grid.w_bc(1, 2, 3), bc, "Z-face round trip of {bc:?}");
        assert_eq!(grid.is_w_solid(1, 2, 3), bc.is_wall());
    }

    // `set_u_solid` is the old spelling of "wall at rest" / "plain fluid",
    // and clears whatever else was on the face.
    grid.set_u_bc(0, 0, 0, inflow);
    grid.set_u_solid(0, 0, 0, true);
    assert_eq!(
        grid.u_bc(0, 0, 0),
        FaceBc::Wall {
            velocity: Vec3Fix::ZERO
        }
    );
    grid.set_u_solid(0, 0, 0, false);
    assert_eq!(grid.u_bc(0, 0, 0), FaceBc::Fluid);
    assert!(!grid.is_u_solid(0, 0, 0));

    // A fresh grid is fluid everywhere, which is what keeps every existing
    // caller on the old behaviour.
    let plain = MacGrid::new(2, 2, 2, Fix128::ONE);
    assert_eq!(plain.u_bc(0, 0, 0), FaceBc::Fluid);
    assert_eq!(plain.v_bc(0, 0, 0), FaceBc::Fluid);
    assert_eq!(plain.w_bc(0, 0, 0), FaceBc::Fluid);
}
