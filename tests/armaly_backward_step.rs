//! Flow over a backward-facing step: the parts of the oracle that do not read
//! a number off a paper (`src/eulerian_grid.rs`, `src/cfd_solver.rs`).
//!
//! # Where the literature gate comes from
//!
//! ⚠️ **The numerical gate for this geometry is Gartling (1990) at `Re = 800`,
//! not Armaly (1983).** Armaly, Durst, Pereira & Schönung, *J. Fluid Mech.*
//! **127** (1983) 473-496 was read in full from the printed scan, and it
//! **contains no table of reattachment lengths** — all `x_r` data in that
//! paper is plotted (figures 4, 9, 11, 13, 14, 18) and the paper has no tables
//! at all. Every "Armaly `x_r/S` = …" in circulation is therefore somebody's
//! digitisation of a figure, which cannot be confirmed by finding a second
//! source that agrees to the last printed digit. The values that *can* be
//! confirmed that way are Gartling's, and they belong to a **different
//! problem**: Armaly's expansion ratio is 1:1.94, Gartling's is exactly 2.
//! The file name kept the word "armaly" because the backlog entry that asked
//! for this work uses it; the scene below is Gartling's.
//!
//! The comparison against Gartling's numbers is not in this file. What is
//! here are three statements that hold whatever the literature says, so that
//! when the literature comparison misses it is already known whether the
//! solver conserves mass, whether it reaches the right developed profile, and
//! what the reattachment detector means.
//!
//! # The scene
//!
//! Gartling's geometry, `H = 1` for the downstream channel and `S = H/2` for
//! the step:
//!
//! ```text
//!   y=H  +---------------------------------------------   no-slip
//!        | inflow  ->
//!   y=S  +- - - - - - - - - - - - - - - - - - - - - - -   (open)   -> outflow
//!        | step face (no-slip)
//!   y=0  +---------------------------------------------   no-slip
//!        x=0                                          x=L
//! ```
//!
//! There is **no upstream channel section**: the inflow plane *is* the step
//! plane, its lower half a wall and its upper half a prescribed inflow. That
//! is Gartling's problem, and with the current API it needs no interior solid
//! cells at all — `set_u_bc(0, j, k, Wall)` for `j dx < S` is the whole step.
//! The two long walls are `FaceBc::Wall`, `i = nx` is `FaceBc::Outflow`, and
//! `nz = 1` with both `z` layers `FaceBc::SlipWall` makes it two-dimensional.
//!
//! ⚠️ **`nz = 1` is safe now and was not always.** Before the face mask went
//! into the Poisson stencil the divisor was a fixed `1/6`, so one cell along
//! `z` left both `z` neighbours missing and turned the pressure problem into a
//! screened one; `nz <= 2` then made the projection almost inert. Since
//! `inverse_degrees` counts the faces that actually carry a pressure degree of
//! freedom, one cell along `z` gives degree 4, which is the correct
//! two-dimensional stencil. Measured on the `240 x 8` Gartling scene:
//! `max |div u| = 1.798e-13`.
//!
//! ## Reynolds number
//!
//! `Re = u_mean_inlet * H / nu` with the mean taken over the **inlet** cross
//! section (height `S`) and `H` the **downstream** channel height — this is
//! Gartling's definition, and the inlet column below is normalised so that
//! the inlet mean is `1` exactly, which makes `Re = 1/nu` by construction.
//! With `nu = 1/800` that is `Re = 800`.
//!
//! ## Inlet profile
//!
//! The inflow is **a developed profile, not a plug**: the discrete Poiseuille
//! profile of a channel of height `S`, normalised to mean `1`. In exact
//! integers, with `r = 1/dx` and `n = S/dx` faces,
//!
//! ```text
//!   k_m = (2m + 1)(2n - 2m - 1) + 1          u_m = n k_m / sum(k)
//! ```
//!
//! which is `C [y(S - y) + dx^2/4]` over the common denominator `4 r^2` (the
//! derivation of the bracket is in the layer-3 section). Gartling states the
//! inlet as the continuum parabola `u(y) = 12y - 24y^2`; that is the same
//! thing without the discrete offset, because `24 y (S - y)` with `S = 1/2`
//! **is** `12y - 24y^2`. So the two differ by the offset alone,
//! `24 * dx^2/4 = 6 dx^2` to leading order, and measured:
//!
//! | `dx` | worst abs difference from `12y - 24y^2` | `/dx^2` | discrete peak |
//! |---|---|---|---|
//! | 1/8 | 7.2917e-2 | 4.667 | 1.3333333 |
//! | 1/16 | 2.2017e-2 | 5.636 | 1.4545455 |
//! | 1/32 | 5.7685e-3 | 5.907 | 1.4883721 |
//!
//! approaching the predicted `6` and the continuum peak `1.5`. The discrete
//! form is the one prescribed here because it is the profile the solver's own
//! inlet channel would hold — a fixed point rather than a shape that starts
//! developing in the first few cells — and because its mean is exactly `1`,
//! which is what fixes `Re`.
//!
//! # Layer 2 — every cross-section carries the inflow flux
//!
//! Summing `div u` over all cells with first index below `i` telescopes: the
//! `v` faces cancel in `j` and leave `j = 0` and `j = ny`, both walls, hence
//! zero; the `w` faces leave the two `SlipWall` layers, also zero; the `u`
//! faces leave the two columns. So, **exactly and at every step**,
//!
//! ```text
//!   sum_j u(i, j) - sum_j u(0, j)  =  dx * sum_{i' < i, j} div u(i', j)
//! ```
//!
//! ⚠️ **This is an identity, not a steady-state property**, which is what
//! makes it cheap: it does not need the run to converge. Measured on the
//! `32 x 8` Gartling scene, worst over all `i`:
//!
//! | steps | identity residual | flux imbalance |
//! |---|---|---|
//! | 1 | 2.220e-15 | 4.000e0 |
//! | 16 | 1.443e-15 | 8.591e-1 |
//! | 256 | 1.027e-15 | 1.763e-4 |
//!
//! (those two columns measured in `f64`; the test compares in `Fix128`, where
//! the residual is **`0` ulp** at every step count). The left column is flat
//! at the arithmetic floor while the right one falls by four orders — the
//! identity holds long before the flow is steady, and only the right-hand
//! side shrinks. At steady state the right-hand side vanishes and the identity
//! becomes "every cross-section carries the inflow flux": measured
//! `4.441e-16` on a `64 x 8` run and `1.331e-11` on the `240 x 8` Gartling
//! one.
//!
//! ## What the leftover divergence is
//!
//! At step 256 the field still carries `max |div u| = 3.145e-5`, which is
//! **the Gauss-Seidel residual** and not discretisation or a boundary
//! condition. Measured, same scene and step count, varying only the sweep
//! count:
//!
//! | sweeps | worst abs divergence | flux imbalance |
//! |---|---|---|
//! | 15 | 1.8576e-2 | 1.7794e-1 |
//! | 60 | 3.1453e-5 | 1.7628e-4 |
//! | 240 | 2.3201e-6 | 1.7784e-5 |
//! | 960 | 2.9094e-7 | 2.8009e-6 |
//! | 3840 | 5.2930e-9 | 5.0977e-8 |
//!
//! — monotone over a factor of 256 in the sweep count with **no plateau**,
//! which is what distinguishes an under-converged solve from a discretisation
//! floor (a floor shows up as two very different iteration counts giving the
//! same answer). At a fixed 60 sweeps it also falls with time, because a
//! steady field has almost nothing left to project: `3.1453e-5` at `t = 8`,
//! `2.3885e-5` at `t = 32`, `1.8385e-13` at `t = 128`, `4.3368e-19` at
//! `t = 512`.
//!
//! ⚠️ So the test does **not** carry a tolerance on the divergence. It
//! asserts the mechanism instead: quadrupling the sweeps must reduce the
//! residual by at least four (measured 13.6x). A residual that failed to move
//! with the sweep count would be the case worth stopping for, and that
//! assertion is what would notice.
//!
//! # Layer 3 — the developed outlet profile
//!
//! Fully developed means `v = 0` and `u` a function of `y` alone, so the
//! momentum balance is `nu u_yy = dp/dx = -K` with `K` constant across the
//! channel. Discretely, with `u` at `y_j = (j + 1/2) dx` and the no-slip ghost
//! `u_-1 = -u_0`, write the fixed point as a parabola plus a constant,
//! `u_j = A y_j (H - y_j) + B`:
//!
//! 1. the second difference of a quadratic is exact, so every **interior** row
//!    gives `nu (-2A) = -K`, i.e. `A = K/2nu`, and a constant is invisible
//!    there;
//! 2. at the **wall** row the ghost `-f(h)` differs from the parabola's own
//!    continuation `f(-h)` by `+2 A h^2` with `h = dx/2`, and the constant
//!    shifts that row by `-2B` (interior rows cannot see a constant, the wall
//!    row can — that asymmetry is the whole effect), so `2 A h^2 = 2B` and
//!    `B = A dx^2/4`.
//!
//! ```text
//!   u_j = C * [ y_j (H - y_j) + dx^2/4 ]        C = Q / sum_j[ ... ]
//! ```
//!
//! `C` is **not** a free parameter: `Q` is the flux, and layer 2 says the flux
//! is the inflow flux, which is prescribed. So the whole profile is a closed
//! form in prescribed quantities.
//!
//! ⚠️ **This is a two-term form, and the three-term form in
//! `tests/analytic_cfd_flow_bc.rs` does not carry over.** That file's third
//! term `- G dt` comes from the body force being added *before* the viscous
//! term, so the Laplacian sees `u + G dt` and the wall row keeps `-2 G dt` of
//! it. A duct driven by a prescribed inflow has no body force: the pressure
//! gradient enters in the projection, *after* the viscous term, so the
//! Laplacian sees `u` and nothing of the kind appears. Two measurements say
//! so independently:
//!
//! - seeding a `12 x 8` duct with the dyadic two-term profile
//!   `{1/16, 5/32, 7/32, 1/4, 1/4, 7/32, 5/32, 1/16}` and prescribing it as
//!   the inflow leaves it there: `max |u - closed form|` falls monotonically
//!   to **3.0358e-18** (`nu = 1/64`, `dt = 1/32`, 1024 steps), the fixed-point
//!   resolution. The three-term form would sit `G dt` away.
//! - ⚠️ the `1.4885` that `duct_profile_develops_toward_the_parabola` in
//!   `tests/analytic_cfd_flow_bc.rs` reports is **not** "close to the
//!   continuum ratio `1.5`": the exact discrete peak-to-mean ratio at
//!   `ny = 16` is `0.25 / 0.16796875 = 1.488372...`, which agrees with the
//!   measurement to four digits. That run was already on the discrete fixed
//!   point; the test tolerates 0.8 % of discretisation error it could have
//!   predicted. Left alone here — that file belongs to another task — and
//!   filed in the backlog instead.
//!
//! ## The step outlet approaches it, and the residual is entrance length
//!
//! On the step scene the outlet deviation from the closed form settles at a
//! non-zero value. That is either incomplete development or a wrong closed
//! form, and the two are told apart by varying the channel **length**
//! (`Re = 96`, `ny = 8`, an earlier geometry with a short upstream section):
//!
//! | `L/H` | outlet deviation | deviation at `x = L/2` | `x_r/S` |
//! |---|---|---|---|
//! | 6 | 5.6131e-4 | 1.7295e-2 | 1.8520 |
//! | 8 | 6.7426e-5 | 3.2370e-3 | 1.8520 |
//! | 12 | 7.6976e-7 | 4.0055e-4 | 1.8520 |
//! | 16 | 8.0062e-9 | 4.8517e-5 | 1.8520 |
//!
//! The deviation decays geometrically in the distance from the step (about
//! `9.6` per `2H` there) and, at a **fixed** `x`, does not depend on where the
//! outflow was put — `3.2e-3` at `x = 4H`, `4.0e-4` at `6H`, `4.9e-5` at `8H`
//! whichever `L` produced them. So it is entrance length, not the closed
//! form, and the tolerance below is an explained quantity. (⚠️ `x_r/S` is
//! unchanged to four digits across a factor of nearly three in `L`, which is
//! worth knowing for a problem whose original purpose was to test outflow
//! conditions; at `Re = 800` on this geometry `x_1 = 4.3944` for both
//! `L = 16` and `L = 30`.)
//!
//! ⚠️ **The outlet does not develop at `Re = 800`.** The entrance length is
//! roughly `0.05 Re H = 40H`, longer than Gartling's `L = 30` — which is why
//! his paper is titled *a test problem for outflow boundary conditions*. The
//! development oracle therefore runs the same geometry at **lower `Re`, i.e.
//! larger `nu`** (⚠️ not smaller: lowering `nu` raises `Re` and lengthens the
//! entrance, which is the wrong direction). `Re = 800` is where the detector
//! and the literature comparison live.
//!
//! # Layer 4 — the reattachment detector
//!
//! [`reattachment_x`] takes the row of `u` faces nearest the lower wall and
//! returns the first place the flow stops going backwards, linearly
//! interpolated: `u(i, 0, k)` sits at `x = i dx`, and between the last
//! negative face `i` and the first non-negative one `i + 1`,
//!
//! ```text
//!   x_r = dx * ( i + (-u_i) / (u_{i+1} - u_i) )
//! ```
//!
//! Its absolute value depends on the mesh, so it is not asserted against a
//! constant here. What is pinned is (a) the contract, against rows written
//! down by hand, (b) that the crossing it reports lies in the interval its own
//! input brackets it in, (c) that the reverse flow is real, and (d) that `x_r`
//! grows with `Re`, which no paper is needed for. Measured on `24 x 8`,
//! `L = 3H`:
//!
//! | `Re` | `x_r/S` | reversed faces |
//! |---|---|---|
//! | 16 | 0.28595 | 1 |
//! | 32 | 0.56334 | 2 |
//! | 96 | 1.50055 | 6 |
//!
//! ## Grid refinement at `Re = 800`
//!
//! Measured, Gartling geometry, `L = 16`, `dt = 1/32` for `ny <= 16` and
//! `1/64` for `ny = 32`, `jacobi_iterations = 60`, all read at `t = 256`:
//!
//! | `ny` | `dx` | `x_1` (lower wall) | upper separation | upper reattachment | `L_u` |
//! |---|---|---|---|---|---|
//! | 8 | 1/8 | 4.3944 (settled) | none | none | — |
//! | 16 | 1/16 | >= 4.7220 (rising) | 4.4375 | 5.7626 | 1.325 |
//! | 32 | 1/32 | >= 5.3032 (rising) | 4.3750 | 7.6936 | 3.319 |
//! | reference | — | 6.10 | 4.85 | 10.48 | 5.63 |
//!
//! ⚠️ **Only the `ny = 8` row is converged in time.** At `ny = 16`, `x_1` is
//! `4.631712` at `t = 128` against `4.721966` at `t = 256` — still climbing by
//! `9.0e-2` — where `ny = 8` moves by `2.9e-4` over the same interval. So the
//! finer rows are **lower bounds**, and the step count at which they settle
//! was not found (filed in the backlog). ⚠️ **The flux imbalance does not
//! reveal this**: at `ny = 16` it is `7.70e-7`, small enough to look settled,
//! because it tracks the Gauss-Seidel residual rather than the slowest mode of
//! the flow. Steadiness has to be read off the quantity being measured.
//!
//! ⚠️ **The upper bubble is missing at `ny = 8` because eight cells cannot
//! carry the boundary layer, not because there is nothing there** — that run
//! *is* steady (four digits in `x_1`, `max |div u| = 1.8e-13`), so its absence
//! is resolution and not convergence. Both quantities move toward the
//! reference under refinement and neither has arrived; whether and where they
//! do is the literature comparison's business, not this file's.
//!
//! # Destructive testing
//!
//! One mutation at a time, `src/` restored after each, and every run prints
//! `### MUTATION:` so a deliberate failure cannot be mistaken for a real one
//! (the convention is recorded in the module header of
//! `tests/analytic_corotational.rs`).
//!
//! ⚠️ **Put that marker in the test's own output, not in the shell's.** The
//! mutations below were announced with a shell `echo` before each run, which
//! is invisible to anything reading the `cargo test` stream — and a session
//! watching this file's output from outside picked up three of the mutation
//! failures as genuine ones. Whatever reads the failure has to see the marker
//! in the same stream, or the mutation run has to write somewhere that is not
//! being watched. ⚠️ A run of several mutations in a row also spends the
//! "same test failed three times" budget that a watcher uses to decide when to
//! intervene, so say beforehand how many are coming.
//!
//! | # | mutation | red | still green | restored |
//! |---|---|---|---|---|
//! | M1 | `subtract_pressure_gradient` no longer holds solid `v` faces at zero, so a wall carries flux | layer 2, **the identity itself**, already at step 1 | layers 3 and 4 | green |
//! | M2 | the no-slip ghost below a `u` face becomes the zero-gradient mirror (`2 u_wall - u_in` -> `u_in`) | layer 3 fixed point (drift `3.1e18` ulp = `1.69e-1`), layer 3 step outlet (decay ratio `1.0` and `1.0`), layer 4 (no reattachment at all) | layer 2 | green |
//! | M3 | `FaceBc::Outflow` also blocks the pressure, removing the exterior `p = 0` at the outlet | layer 2 imbalance (`1.763e-4` -> `1.816e2`), layer 3 fixed point (`3.55e1`), layer 3 step outlet (decay `0.5` and `0.0`) | layer 4 | green |
//! | M4 | *(test side)* `bottom_row` reads `u(i, 1, 0)`, the layer above the wall | layer 4 (`Re = 16` has no crossing one cell up — that row is positive everywhere) | layers 2 and 3 | green |
//!
//! ⚠️ **M1 and M3 are caught by different halves of layer 2, and that is the
//! point of splitting it.** The telescoping identity is an algebraic statement
//! about whatever field is present, so removing the outlet's pressure
//! condition (M3) leaves it at `0` ulp — what catches M3 is the flux
//! imbalance. Letting a wall carry flux (M1) instead breaks the identity's own
//! premise, and it goes red at the **first step**, before any convergence
//! could be involved. An oracle with only one of the two would miss one of
//! them.
//!
//! ⚠️ **M2 leaving layer 2 green is correct, not a gap**: a free-slip wall
//! still conserves mass. A mass-conservation oracle cannot see a missing shear
//! condition, which is why layer 3 exists.

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

/// A still no-slip wall.
const WALL: FaceBc = FaceBc::Wall {
    velocity: Vec3Fix::ZERO,
};

/// One unit in the last place of [`Fix128`].
const ULP: f64 = 5.421_010_862_427_522e-20;

/// A [`Fix128`] as a raw two's-complement 128-bit integer, so two answers can
/// be differenced in units in the last place rather than in `f64`, which
/// cannot represent the difference. Same helper as the one in
/// `tests/analytic_corotational.rs`.
fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

/// Distance between two fixed-point values in units in the last place.
///
/// ⚠️ Used instead of `to_f64` wherever the bound is at the arithmetic floor:
/// at these magnitudes the `f64` conversion rounds the difference away, so a
/// comparison made after it would be reporting the rounding, not the drift.
fn ulp_gap(a: Fix128, b: Fix128) -> u128 {
    (raw(a) - raw(b)).unsigned_abs()
}

// ===========================================================================
// Closed forms
// ===========================================================================

/// `k_j = (2j + 1)(2n - 2j - 1) + 1`, the numerator of the discrete Poiseuille
/// shape of an `n`-cell channel over the common denominator `4 r^2`
/// (`r = 1/dx`).
///
/// Derivation: with `y_j = (2j+1)/(2r)` and `h = n/r`,
/// `y_j (h - y_j) = (2j+1)(2n - 2j - 1) / (4 r^2)` and `dx^2/4 = 1/(4 r^2)`,
/// so the bracket of the layer-3 closed form is `k_j / (4 r^2)` — integers
/// throughout, which is what lets the profile be prescribed without rounding.
fn shape_numerator(n: usize, j: usize) -> i64 {
    let n = n as i64;
    let j = j as i64;
    (2 * j + 1) * (2 * n - 2 * j - 1) + 1
}

/// The discrete Poiseuille shape `y_j (h - y_j) + dx^2/4` of an `n`-cell
/// channel, as a ratio of exact integers. `dx = 1/dx_recip`.
fn shape(n: usize, dx_recip: i64, j: usize) -> Fix128 {
    Fix128::from_ratio(shape_numerator(n, j), 4 * dx_recip * dx_recip)
}

/// Inlet column of `n` faces: the discrete Poiseuille profile of a channel of
/// height `n dx`, normalised so that the **discrete** mean is `1`.
///
/// `u_m = n k_m / sum(k)`, exact rationals. Normalising the mean rather than
/// the peak is what makes `Re = u_mean H / nu` equal `1/nu` for `H = 1`.
fn inlet_column(n: usize) -> Vec<Fix128> {
    let total: i64 = (0..n).map(|m| shape_numerator(n, m)).sum();
    (0..n)
        .map(|m| Fix128::from_ratio(n as i64 * shape_numerator(n, m), total))
        .collect()
}

/// Gartling's continuum inlet parabola `u(y) = 12y - 24y^2`, with `y` measured
/// from the step surface. Only used for the second-order comparison against
/// the discrete inlet column.
fn gartling_inlet_parabola(y: f64) -> f64 {
    12.0 * y - 24.0 * y * y
}

/// The developed profile of the tall channel, `C [y_j (H - y_j) + dx^2/4]`
/// with `C` fixed by the flux `q` — which layer 2 says is the inflow flux.
///
/// In `f64` because it is compared against a run that has only approached it;
/// the exact-fixed-point oracle uses [`shape`] directly and compares in
/// [`Fix128`].
fn developed_profile(ny: usize, q: f64) -> Vec<f64> {
    let total: i64 = (0..ny).map(|j| shape_numerator(ny, j)).sum();
    (0..ny)
        .map(|j| q * shape_numerator(ny, j) as f64 / total as f64)
        .collect()
}

// ===========================================================================
// The scene
// ===========================================================================

/// Gartling's backward-facing step: downstream height `H = 1` in `2 s_cells`
/// cells, step height `S = H/2`, channel length `nx dx`, `nu = 1/nu_recip`
/// and therefore `Re = nu_recip`.
fn backward_facing_step(s_cells: usize, nx: usize, nu_recip: i64) -> CfdSolver {
    let ny = 2 * s_cells;
    let dx_recip = ny as i64;
    let mut solver = CfdSolver::new(nx, ny, 1, Fix128::from_ratio(1, dx_recip));
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::from_ratio(1, nu_recip); // rho = 1, so mu = nu
    solver.gravity = Vec3Fix::ZERO;
    solver.jacobi_iterations = 60;
    solver.use_turbulence = false;

    let grid = &mut solver.grid;
    // Two-dimensional: the single z layer gets symmetry planes. They still
    // block the pressure — they have to, or the Poisson problem degenerates
    // into a screened one — they just exert no shear.
    for j in 0..ny {
        for i in 0..nx {
            grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            grid.set_w_bc(i, j, 1, FaceBc::SlipWall);
        }
    }
    for i in 0..nx {
        grid.set_v_bc(i, 0, 0, WALL);
        grid.set_v_bc(i, ny, 0, WALL);
    }
    // The step: the lower half of the inflow plane is the step face.
    for j in 0..s_cells {
        grid.set_u_bc(0, j, 0, WALL);
    }
    for (m, &u) in inlet_column(s_cells).iter().enumerate() {
        grid.set_u_bc(0, s_cells + m, 0, FaceBc::Inflow { normal_velocity: u });
    }
    for j in 0..ny {
        grid.set_u_bc(nx, j, 0, FaceBc::Outflow);
    }
    solver
}

/// A plain channel of the full height `H` with the developed profile both
/// prescribed at the inflow and seeded everywhere — the scene in which that
/// profile has to be a fixed point.
fn tall_channel_at_its_fixed_point(ny: usize, nx: usize, nu_recip: i64) -> CfdSolver {
    let dx_recip = ny as i64;
    let mut solver = CfdSolver::new(nx, ny, 1, Fix128::from_ratio(1, dx_recip));
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::from_ratio(1, nu_recip);
    solver.gravity = Vec3Fix::ZERO;
    solver.jacobi_iterations = 60;
    solver.use_turbulence = false;

    let grid = &mut solver.grid;
    for j in 0..ny {
        for i in 0..nx {
            grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            grid.set_w_bc(i, j, 1, FaceBc::SlipWall);
        }
    }
    for i in 0..nx {
        grid.set_v_bc(i, 0, 0, WALL);
        grid.set_v_bc(i, ny, 0, WALL);
    }
    for j in 0..ny {
        let u = shape(ny, dx_recip, j);
        grid.set_u_bc(0, j, 0, FaceBc::Inflow { normal_velocity: u });
        grid.set_u_bc(nx, j, 0, FaceBc::Outflow);
        for i in 0..=nx {
            grid.u[i + (nx + 1) * j] = u;
        }
    }
    solver
}

// ===========================================================================
// Readouts
// ===========================================================================

/// `sum_{j,k} u(i, j, k)`, exactly. The cell area `dx^2` is common to every
/// cross-section, so the bare sum is the comparison.
fn column_flux(grid: &MacGrid, i: usize) -> Fix128 {
    let mut flux = Fix128::ZERO;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            flux = flux + grid.u(i, j, k);
        }
    }
    flux
}

/// `sum_{i' < i, j, k} div u(i', j, k)`, the right-hand side of the
/// telescoping identity of layer 2.
fn divergence_upstream_of(grid: &MacGrid, i: usize) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for ii in 0..i {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                acc = acc + grid.divergence(ii, j, k);
            }
        }
    }
    acc
}

/// Largest `|div u|` over every cell, rim included.
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

/// The row of `u` faces nearest the lower wall, `u(i, 0, 0)` for every `i`.
///
/// ⚠️ The detector's whole input is this row, so which row it is *is* part of
/// the contract: `row[i]` is the face at `x = i dx` in the cell layer that
/// touches the wall. Reading the layer above instead moves or loses the
/// crossing entirely — measured at `Re = 16` and `Re = 32` the layer above has
/// no crossing at all.
fn bottom_row(grid: &MacGrid) -> Vec<f64> {
    (0..=grid.nx).map(|i| grid.u(i, 0, 0).to_f64()).collect()
}

/// Reattachment abscissa: the first place `row` stops being negative, in the
/// units `dx` is given in. `row[i]` sits at `x = i dx`.
///
/// `None` when the row never goes negative (nothing separated, as far as this
/// row can tell) and when it is still negative at the end (the separation is
/// longer than the row, so there is no answer rather than the last index).
fn reattachment_x(row: &[f64], dx: f64) -> Option<f64> {
    (0..row.len().saturating_sub(1))
        .find(|&i| row[i] < 0.0 && row[i + 1] >= 0.0)
        .map(|i| dx * (i as f64 + (-row[i]) / (row[i + 1] - row[i])))
}

// ===========================================================================
// The inlet closed form
// ===========================================================================

/// Oracle: the prescribed inlet column is the discrete Poiseuille profile of
/// the short channel with **mean exactly one**, which is what makes
/// `Re = u_mean H / nu` equal `1/nu`; and it is Gartling's continuum parabola
/// `12y - 24y^2` plus the discrete offset, so the two agree to `O(dx^2)` with
/// the predicted coefficient `24 * dx^2/4 = 6`.
///
/// Both facts are needed before any Reynolds number in this file means
/// anything, and neither involves running the solver.
#[test]
fn inlet_column_has_unit_mean_and_is_gartlings_parabola_to_second_order() {
    for &s_cells in &[4usize, 8, 16] {
        let dx_recip = 2 * s_cells as i64;
        let dx = 1.0 / dx_recip as f64;
        let column = inlet_column(s_cells);

        let mut sum = Fix128::ZERO;
        for &u in &column {
            sum = sum + u;
        }
        let mean = sum / Fix128::from_int(s_cells as i64);
        let mean_error = ulp_gap(mean, Fix128::ONE);

        let mut worst = 0.0f64;
        for (m, &u) in column.iter().enumerate() {
            let y = (m as f64 + 0.5) * dx;
            worst = worst.max((u.to_f64() - gartling_inlet_parabola(y)).abs());
        }
        let peak = column.iter().map(|u| u.to_f64()).fold(0.0f64, f64::max);
        println!(
            "inlet n={s_cells:3} dx=1/{dx_recip:<3}  |mean - 1| {mean_error} ulp  \
             max |discrete - (12y-24y^2)| {worst:.4e} = {:.3} dx^2  peak {peak:.7}",
            worst / (dx * dx)
        );

        // `n k_m / sum(k)` is an exact rational whose mean is exactly 1; the
        // only thing that can move it is the rounding of each term into
        // `Fix128`, which is one unit in the last place per face. Measured: 1
        // ulp at all three resolutions, i.e. Re = 800 to nineteen digits.
        assert!(
            mean_error <= 8,
            "n={s_cells}: the discrete inlet mean must be 1 so that Re = 1/nu, \
             off by {mean_error} ulp ({:.3e})",
            mean_error as f64 * ULP
        );
        // 6 dx^2 is the offset term 24 * dx^2/4; the remainder is the
        // normalisation, which the measured ratios 4.667 / 5.636 / 5.907
        // approach 6 from below. 7 dx^2 is the predicted bound with room for
        // that approach, and it is not a fitted number: at dx = 1/8 it is
        // 1.09e-1 against a measured 7.29e-2.
        assert!(
            worst <= 7.0 * dx * dx,
            "n={s_cells}: the discrete and continuum inlets must differ by the \
             offset 6 dx^2 = {:.4e}, measured {worst:.4e}",
            6.0 * dx * dx
        );
        // Not vacuous: the profile is a parabola, not a plug.
        assert!(
            peak > 1.3 && peak < 1.5,
            "n={s_cells}: the discrete peak must approach the continuum 1.5 \
             from below, got {peak}"
        );
    }
}

// ===========================================================================
// Layer 2 — the flux telescopes
// ===========================================================================

/// Oracle: `sum_j u(i,j) - sum_j u(0,j) = dx * sum_{i'<i,j} div u(i',j)` for
/// every cross-section, **at every step**, because summing the divergence over
/// the cells upstream of `i` telescopes and the walls and symmetry planes
/// carry no flux. At steady state the right-hand side vanishes and it becomes
/// "every cross-section carries the inflow flux".
///
/// Run at Gartling's `nu = 1/800`, which the identity does not care about, and
/// checked after 1, 16 and 256 steps so that the two halves are visibly
/// separate: the identity residual stays at the arithmetic floor while the
/// flux imbalance falls by four orders of magnitude. An oracle that only
/// looked at the imbalance would be measuring convergence, not conservation.
#[test]
fn every_cross_section_telescopes_to_the_inflow_flux() {
    let (s_cells, nx) = (4usize, 32usize);
    let mut solver = backward_facing_step(s_cells, nx, 800);
    let dx = solver.grid.dx;
    let dt = Fix128::from_ratio(1, 32);

    let mut done = 0u32;
    let mut imbalance_now = f64::INFINITY;
    for &steps in &[1u32, 16, 256] {
        for _ in 0..steps - done {
            solver.step(dt);
        }
        done = steps;

        let inflow = column_flux(&solver.grid, 0);
        let mut worst_identity = 0u128;
        let mut worst_imbalance = Fix128::ZERO;
        for i in 1..=nx {
            let flux = column_flux(&solver.grid, i);
            let predicted = inflow + dx * divergence_upstream_of(&solver.grid, i);
            worst_identity = worst_identity.max(ulp_gap(flux, predicted));
            let imbalance = (flux - inflow).abs();
            if imbalance > worst_imbalance {
                worst_imbalance = imbalance;
            }
        }
        imbalance_now = worst_imbalance.to_f64();
        println!(
            "steps={steps:4}  telescoping residual {worst_identity} ulp  \
             flux imbalance {imbalance_now:.3e}  inflow flux {:.12} ({} ulp from {s_cells})",
            inflow.to_f64(),
            ulp_gap(inflow, Fix128::from_int(s_cells as i64))
        );

        // Measured `0` ulp at every step count, so equality is the honest
        // bound: the identity is the telescoping sum of exact fixed-point
        // additions and there is nothing for it to round.
        assert_eq!(
            worst_identity, 0,
            "steps={steps}: the telescoping identity is exact in fixed point"
        );
        // The prescribed inflow survives advection, diffusion and the pressure
        // correction, so the flux it carries is the closed-form one: s_cells
        // faces of mean exactly 1.
        // The closed form is `s_cells` faces of mean exactly 1, so the flux is
        // `s_cells`; the only slack is the rounding of each prescribed face
        // into `Fix128`, at most one ulp per face. Measured: 4 ulp for 4 faces.
        let flux_error = ulp_gap(inflow, Fix128::from_int(s_cells as i64));
        assert!(
            flux_error <= 2 * s_cells as u128,
            "the prescribed inflow flux must be {s_cells} to within one ulp per \
             prescribed face, off by {flux_error} ulp"
        );
        // Not vacuous: the imbalance really is large to begin with, so the
        // identity above is not holding because the field is already balanced
        // (or empty).
        if steps == 1 {
            assert!(
                imbalance_now > 1.0,
                "after one step the flux cannot already be balanced, or the \
                 identity is being checked on a converged field; got \
                 {imbalance_now:.3e}"
            );
        }
    }
    assert!(
        imbalance_now < 1e-3,
        "by 256 steps the imbalance must have fallen by orders of magnitude, \
         got {imbalance_now:.3e}"
    );
    // What is left of the divergence at this point is the **Gauss-Seidel
    // residual**, and rather than tolerate a number, say so and let it be
    // falsified: quadrupling the sweeps has to reduce it. ⚠️ A residual that
    // did *not* move with the sweep count would be discretisation or a
    // boundary condition — the case worth stopping for — and this is the
    // assertion that would notice. Measured at 256 steps on this scene:
    // `1.86e-2` at 15 sweeps, `3.15e-5` at 60, `2.32e-6` at 240, `2.91e-7` at
    // 960, `5.29e-9` at 3840, i.e. monotone over a factor of 256 with no
    // plateau. (At steady state there is almost nothing left to project and 60
    // sweeps reach `4.34e-19`; see the module header.)
    let coarse = max_abs_divergence(&solver.grid);
    let mut refined_solver = backward_facing_step(s_cells, nx, 800);
    refined_solver.jacobi_iterations = 4 * solver.jacobi_iterations;
    for _ in 0..done {
        refined_solver.step(dt);
    }
    let refined = max_abs_divergence(&refined_solver.grid);
    println!(
        "max |div u| after {done} steps: {coarse:.4e} at {} sweeps, {refined:.4e} at {} \
         sweeps (ratio {:.1})",
        solver.jacobi_iterations,
        refined_solver.jacobi_iterations,
        coarse / refined
    );
    assert!(
        refined < coarse / 4.0,
        "the divergence left after {done} steps must be the Gauss-Seidel \
         residual, so four times the sweeps must reduce it by at least four: \
         {coarse:.4e} -> {refined:.4e}"
    );
}

// ===========================================================================
// Layer 3 — the developed profile
// ===========================================================================

/// Oracle: `C [y_j (H - y_j) + dx^2/4]` is an **exact** fixed point of the
/// solver for a channel driven by a prescribed inflow — the two-term form,
/// with no `- G dt`, because there is no body force here. The derivation and
/// the reason the three-term form of `tests/analytic_cfd_flow_bc.rs` does not
/// carry over are in the module header.
///
/// The field starts on the closed form, so what is measured is whether it
/// stays: the only transient is the projection building its streamwise
/// pressure gradient up from `p = 0`. The bound is the fixed-point
/// arithmetic, not a discretisation allowance — the point of the exercise is
/// that there is no discretisation error left to allow for.
#[test]
fn the_two_term_developed_profile_is_an_exact_fixed_point() {
    let (ny, nx) = (8usize, 12usize);
    let dx_recip = ny as i64;
    let mut solver = tall_channel_at_its_fixed_point(ny, nx, 64);
    let dt = Fix128::from_ratio(1, 32);
    for _ in 0..1024 {
        solver.step(dt);
    }

    let mut worst = 0u128;
    let mut worst_at = (0usize, 0usize);
    for j in 0..ny {
        let want = shape(ny, dx_recip, j);
        for i in 0..=nx {
            let drift = ulp_gap(solver.grid.u(i, j, 0), want);
            if drift > worst {
                worst = drift;
                worst_at = (i, j);
            }
        }
    }
    println!(
        "tall channel {nx}x{ny}, nu=1/64, t = 32: max |u - closed form| {worst} ulp \
         = {:.4e} at (i={}, j={}), max |div| {:.3e}",
        worst as f64 * ULP,
        worst_at.0,
        worst_at.1,
        max_abs_divergence(&solver.grid)
    );
    for j in 0..ny {
        println!(
            "  j={j}  u_outflow {:+.15}  closed form {:+.15}",
            solver.grid.u(nx, j, 0).to_f64(),
            shape(ny, dx_recip, j).to_f64()
        );
    }

    // ⚠️ The budget is a **floor**, not an allowance per step, and that is a
    // measured distinction rather than an assumption. Drift against step
    // count, same scene:
    //
    //   N     256        512        1024   2048   4096
    //   ulp   9377357473 8601095    56     54     54
    //
    // The approach is exponential — about three orders per doubling while the
    // projection builds its streamwise pressure gradient up from `p = 0` — and
    // then it stops at **54 ulp and does not move again**, confirmed over a
    // fourfold range in `N`. So the bound must be an `N`-independent constant:
    // `128` is 2.4x the floor, and `N = 1024` (56 ulp) is just past the knee.
    // ⚠️ A budget of the form `k · N` would be the wrong shape here — nothing
    // accumulates — and `0` is not reachable, unlike the Couette fixed point in
    // `tests/analytic_cfd_flow_bc.rs`, because the pressure gradient the
    // projection subtracts does not round to a value that closes exactly.
    // ⚠️ Reversal condition: if a future arithmetic change moves the floor,
    // widen this constant and record the new floor — do **not** make it
    // proportional to `N` (measured not to grow with `N`) and do **not**
    // convert it to a tolerance in `to_f64`, because at `3e-18` against a
    // field of order `0.25` the `f64` conversion cannot see the drift at all
    // and such a comparison would pass on anything.
    assert!(
        worst <= 128,
        "the two-term profile must be a fixed point to the arithmetic floor \
         (54 ulp, N-independent), drifted {worst} ulp = {:.4e}",
        worst as f64 * ULP
    );
    // Second: the profile being held is a sheared one, not a uniform field
    // that every wall condition would preserve.
    assert!(
        shape(ny, dx_recip, ny / 2) > shape(ny, dx_recip, 0) * Fix128::from_int(3),
        "the profile the run is held against must actually be sheared"
    );
}

/// Oracle: on the step scene the outlet approaches the same closed form, and
/// the residual is entrance length rather than a mismatch — the deviation
/// falls geometrically with distance from the step.
///
/// Run at `Re = 16`, i.e. **`nu` raised** relative to Gartling's `1/800`.
/// ⚠️ Lowering `nu` would raise `Re` and lengthen the entrance; at `Re = 800`
/// the entrance is about `40H`, longer than the channel, which is the whole
/// subject of Gartling's paper. The closed form itself does not depend on
/// `nu`: `nu` only sets the constant `K`, and `K` is eliminated by the flux.
#[test]
fn the_step_outlet_develops_into_the_two_term_profile() {
    let (s_cells, nx) = (4usize, 48usize);
    let ny = 2 * s_cells;
    let mut solver = backward_facing_step(s_cells, nx, 16);
    let dt = Fix128::from_ratio(1, 32);
    for _ in 0..640 {
        solver.step(dt);
    }

    let q = column_flux(&solver.grid, 0).to_f64();
    let want = developed_profile(ny, q);
    let deviation_at = |i: usize| {
        (0..ny)
            .map(|j| (solver.grid.u(i, j, 0).to_f64() - want[j]).abs())
            .fold(0.0f64, f64::max)
    };
    // x = 0, 2H, 4H and the outlet at 6H.
    let at_inlet = deviation_at(0);
    let at_2h = deviation_at(2 * ny);
    let at_4h = deviation_at(4 * ny);
    let at_outlet = deviation_at(nx);
    println!(
        "step outlet, Re=16, {nx}x{ny}, L=6H, t=20: deviation from the closed \
         form — inlet {at_inlet:.3e}  2H {at_2h:.3e}  4H {at_4h:.3e}  \
         outlet(6H) {at_outlet:.3e},  max |div| {:.3e}",
        max_abs_divergence(&solver.grid)
    );
    for (j, &target) in want.iter().enumerate() {
        println!(
            "  j={j}  u_outlet {:+.9}  closed form {target:+.9}",
            solver.grid.u(nx, j, 0).to_f64()
        );
    }

    // Geometric decay in the distance from the step: this is what separates
    // "not developed yet" from "the closed form is wrong". A constant offset
    // would hold the ratio near 1. Measured here: 246 then 173 per 2H.
    let first_ratio = at_2h / at_4h;
    let second_ratio = at_4h / at_outlet;
    println!("  decay per 2H: {first_ratio:.1} then {second_ratio:.1}");
    assert!(
        first_ratio > 20.0 && second_ratio > 20.0,
        "the deviation must decay geometrically downstream, got \
         {first_ratio:.1} and {second_ratio:.1}"
    );
    assert!(
        at_outlet < 1e-6,
        "the outlet must have reached the closed form, off by {at_outlet:.3e}"
    );
    // Not vacuous: the inflow column is nowhere near the tall-channel profile
    // — its lower half is the step face — so "the outlet matches" is a
    // statement about what the run did, not about the scene.
    assert!(
        at_inlet > 0.5 * want[ny / 2],
        "the inflow column must be far from the developed profile, else the \
         outlet matching it says nothing; deviation {at_inlet:.3e} against a \
         profile peak of {:.3e}",
        want[ny / 2]
    );
}

// ===========================================================================
// Layer 4 — the reattachment detector
// ===========================================================================

/// Oracle: [`reattachment_x`] on rows written down here, so that what it means
/// is fixed independently of any flow.
///
/// The interpolation weights are asymmetric on purpose: a row whose crossing
/// sat at the midpoint would pass just as well with the two neighbours swapped
/// or with the index-to-`x` mapping off by half a cell.
#[test]
fn the_reattachment_detector_reports_the_first_sign_change() {
    let dx = 0.25f64;
    // Crossing between index 2 (-3) and 3 (+1): 2 + 3/4 cells.
    assert_eq!(
        reattachment_x(&[0.0, -1.0, -3.0, 1.0, 2.0], dx),
        Some(dx * 2.75),
        "row[i] sits at x = i dx and the crossing is interpolated linearly \
         between the bracketing faces"
    );
    // Same magnitudes mirrored: 1 + 1/4 cells. An interpolation that used the
    // wrong endpoint would give 1.75 here and 2.25 above.
    assert_eq!(reattachment_x(&[0.0, -1.0, 3.0, 4.0], dx), Some(dx * 1.25));
    // Exactly zero counts as attached: the first face is a wall and reads 0,
    // and a run that never separates must not report a reattachment.
    assert_eq!(reattachment_x(&[0.0, 1.0, 2.0, 3.0], dx), None);
    assert_eq!(reattachment_x(&[0.0, 0.0, 0.0], dx), None);
    // Still reversed at the end of the row: the separation is longer than the
    // channel, so there is no answer rather than the last index.
    assert_eq!(reattachment_x(&[0.0, -1.0, -2.0, -3.0], dx), None);
    // Two bubbles: the first reattachment, not the last.
    assert_eq!(
        reattachment_x(&[0.0, -1.0, 1.0, -1.0, 1.0], dx),
        Some(dx * 1.5)
    );
    // Rows too short to bracket anything.
    assert_eq!(reattachment_x(&[-1.0], dx), None);
    assert_eq!(reattachment_x(&[], dx), None);
    println!("reattachment_x contract: 8 hand-written rows agree");
}

/// Oracle: the detector finds a real separation bubble on the step field, the
/// crossing it reports lies inside the cell its own input brackets it in, and
/// `x_r` **grows with `Re`** — none of which needs a number from a paper.
///
/// The absolute values are printed rather than asserted against constants:
/// they depend on the mesh, the grid-refinement measurements are in the module
/// header, and the comparison against Gartling belongs to a separate test.
#[test]
fn reattachment_grows_with_reynolds_number_on_the_step_field() {
    let (s_cells, nx) = (4usize, 24usize);
    let step_height = 0.5f64;
    let dx = 1.0 / (2 * s_cells) as f64;

    let mut previous: Option<(i64, f64)> = None;
    for &(re, steps) in &[(16i64, 640u32), (32, 960), (96, 1600)] {
        let mut solver = backward_facing_step(s_cells, nx, re);
        let dt = Fix128::from_ratio(1, 32);
        for _ in 0..steps {
            solver.step(dt);
        }
        let row = bottom_row(&solver.grid);
        let Some(x_r) = reattachment_x(&row, dx) else {
            panic!("Re={re}: no reattachment found on the bottom row {row:?}");
        };
        let reversed = row.iter().filter(|u| **u < 0.0).count();
        println!(
            "Re={re:4} {nx}x{ny}: x_r = {x_r:.6} = {:.5} S, {reversed} reversed \
             faces, max |div| {:.3e}",
            x_r / step_height,
            max_abs_divergence(&solver.grid),
            ny = 2 * s_cells
        );

        // The reported crossing must lie in the cell the row brackets it in —
        // a self-consistency check on the index-to-x mapping that needs no
        // reference value.
        let bracket = (0..row.len() - 1)
            .find(|&i| row[i] < 0.0 && row[i + 1] >= 0.0)
            .expect("a reattachment was found, so a bracketing pair exists");
        assert!(
            x_r > dx * bracket as f64 && x_r < dx * (bracket + 1) as f64,
            "Re={re}: x_r = {x_r} must lie between x = {} and {}",
            dx * bracket as f64,
            dx * (bracket + 1) as f64
        );
        // The reverse flow is real and starts at the step, not somewhere down
        // the channel.
        assert!(
            reversed >= 1 && row[1] < 0.0,
            "Re={re}: the face next to the step must be reversed, row {row:?}"
        );
        // Physics, no literature: a longer bubble at higher Re.
        if let Some((prev_re, prev_x)) = previous {
            assert!(
                x_r > prev_x,
                "x_r must grow with Re: Re={prev_re} gave {prev_x:.6}, Re={re} \
                 gave {x_r:.6}"
            );
        }
        previous = Some((re, x_r));
    }
}

/// The grid-refinement sweep behind the table in the module header, at
/// Gartling's `Re = 800`.
///
/// `#[ignore]`d **for runtime only**: `256 x 16` cells for 8192 steps with 60
/// Gauss-Seidel sweeps is minutes in release and far longer in a debug build.
/// Run with
/// `cargo test --release --test armaly_backward_step -- --ignored --nocapture`.
///
/// # What can honestly be asserted, and why it is not simply "x_1 grew"
///
/// ⚠️ **`ny = 8` settles within this step budget and `ny = 16` does not.**
/// Measured, `L = 16`, `dt = 1/32`:
///
/// | `ny` | `x_1` at `t = 128` | `x_1` at `t = 256` |
/// |---|---|---|
/// | 8 | 4.394114 | 4.394400 |
/// | 16 | 4.631712 | 4.721966 |
///
/// so at `ny = 8` the answer has stopped moving (`2.9e-4` apart) while at
/// `ny = 16` it is still climbing by `9.0e-2` over the same interval. Reading
/// both off at `t = 256` and calling the pair a refinement study would be
/// comparing a converged number with an unconverged one.
///
/// ⚠️ **It is still possible to conclude something rigorous, because the
/// direction of the remaining drift is known.** `x_1` at `ny = 16`
/// *increases* with time, so its value at `t = 256` is a **lower bound** on
/// its converged value, and
///
/// ```text
///   x_1(ny=16, converged)  >=  4.7220  >  4.3944  =  x_1(ny=8, converged)
/// ```
///
/// — refinement lengthens `x_1`, a fortiori.
///
/// ⚠️ **The gate is that inequality alone, checked at both sample points, and
/// deliberately not "`ny = 16` is still climbing".** Being mid-transient is a
/// property of the step budget, not of the discretisation: a faster solver, a
/// longer run or a different sweep count would let `ny = 16` settle, and an
/// assertion that pinned the climbing would turn that correct improvement into
/// a failure. The inequality holds either way. What the test does assert is
/// convergence in time at `ny = 8` — that one is a property of the scene, not
/// of the budget, and dropping it to make the pair symmetric would remove the
/// reference the inequality is measured against.
///
/// ⚠️ **`ny = 32` is deliberately not run here.** Measured once at `t = 256`:
/// `x_1 = 5.3032`, upper-wall bubble `4.3750 .. 7.6936`. By the same argument
/// that is a lower bound, but the step count at which it settles was not
/// found (one block of `t = 128` at that resolution is minutes), so it is a
/// recorded measurement rather than a gate. Finding that budget is filed in
/// the backlog.
#[test]
#[ignore = "runtime only: 256x16 cells for 8192 steps, release-only; the measured values are in the module header"]
fn reattachment_lengthens_under_grid_refinement() {
    let length = 16.0f64;
    let dt = Fix128::from_ratio(1, 32);
    let steps = 8192u32;

    // ny = 8: settled, and the value is the reference for the inequality.
    let coarse = {
        let (s_cells, dx) = (4usize, 1.0 / 8.0);
        let nx = (length / dx).round() as usize;
        let mut solver = backward_facing_step(s_cells, nx, 800);
        for _ in 0..steps / 2 {
            solver.step(dt);
        }
        let halfway =
            reattachment_x(&bottom_row(&solver.grid), dx).expect("ny=8: no reattachment half-way");
        for _ in steps / 2..steps {
            solver.step(dt);
        }
        let end = reattachment_x(&bottom_row(&solver.grid), dx)
            .expect("ny=8: no reattachment at the end");
        let upper_has_bubble =
            (1..=nx).any(|i| solver.grid.u(i, 2 * s_cells - 1, 0).to_f64() < 0.0);
        let mut imbalance = 0.0f64;
        let inflow = column_flux(&solver.grid, 0);
        for i in 1..=nx {
            imbalance = imbalance.max((column_flux(&solver.grid, i) - inflow).abs().to_f64());
        }
        println!(
            "ny= 8 dx=1/8  nx={nx:4} steps={steps}  x_1 {end:.6} (half-way {halfway:.6}, \
             delta {:.2e})  upper-wall reverse flow {upper_has_bubble}  flux imbalance \
             {imbalance:.2e}  max |div| {:.2e}",
            (end - halfway).abs(),
            max_abs_divergence(&solver.grid)
        );
        assert!(
            (end - halfway).abs() < 1e-3,
            "ny=8 must have settled within this budget — {halfway:.6} half-way against \
             {end:.6} at the end"
        );
        assert!(
            !upper_has_bubble,
            "ny=8 must show no upper-wall reverse flow: eight cells cannot carry that \
             boundary layer, and the reference bubble at 4.85..10.48 is what refinement \
             has to bring in"
        );
        end
    };

    // ny = 16: still climbing, so its value is a lower bound on the converged
    // one, and the upper-wall bubble has appeared.
    let (s_cells, dx) = (8usize, 1.0 / 16.0);
    let nx = (length / dx).round() as usize;
    let mut solver = backward_facing_step(s_cells, nx, 800);
    for _ in 0..steps / 2 {
        solver.step(dt);
    }
    let halfway =
        reattachment_x(&bottom_row(&solver.grid), dx).expect("ny=16: no reattachment half-way");
    for _ in steps / 2..steps {
        solver.step(dt);
    }
    let fine =
        reattachment_x(&bottom_row(&solver.grid), dx).expect("ny=16: no reattachment at the end");
    let upper: Vec<f64> = (0..=nx)
        .map(|i| solver.grid.u(i, 2 * s_cells - 1, 0).to_f64())
        .collect();
    let separation = (1..=nx).find(|&i| upper[i] < 0.0).map(|i| i as f64 * dx);
    let reattachment = reattachment_x(&upper, dx);
    println!(
        "ny=16 dx=1/16 nx={nx:4} steps={steps}  x_1 {fine:.6} (half-way {halfway:.6}, \
         still climbing by {:.2e})  upper bubble {separation:?} .. {reattachment:?}  \
         max |div| {:.2e}",
        fine - halfway,
        max_abs_divergence(&solver.grid)
    );

    // ⚠️ The gate is the inequality alone, at **both** sample points. It was
    // tempting to assert that `ny = 16` is still climbing, since that is what
    // measured — but "still climbing" is a property of this step budget, so a
    // faster solver, a longer run or a different sweep count would make it
    // settle and turn a correct improvement into a red. Requiring the
    // inequality at half-way *and* at the end keeps the statement true however
    // the run converges, and still catches a run that has gone wild.
    assert!(
        halfway > coarse && fine > coarse,
        "refinement must lengthen x_1: ny=8 settled at {coarse:.6}, ny=16 reads \
         {halfway:.6} half-way and {fine:.6} at the end"
    );
    // The upper-wall bubble appears once the mesh can carry it. Absent at
    // ny=8 (asserted above), present here — that contrast is the statement.
    let separation = separation.expect("ny=16 must show upper-wall separation");
    let reattachment = reattachment.expect("ny=16 must show upper-wall reattachment");
    assert!(
        reattachment > separation,
        "the upper-wall bubble must close downstream of where it opens: \
         {separation:.4} .. {reattachment:.4}"
    );
}
