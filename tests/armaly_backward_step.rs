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
//! Three of the four layers here hold whatever the literature says, so that
//! when the literature comparison misses it is already known whether the
//! solver conserves mass, whether it reaches the right developed profile, and
//! what the reattachment detector means. The fourth, layer 1, is the
//! comparison itself — and it misses, by a lot.
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
//! # Layer 1 — the comparison against Gartling, as a pair of twins
//!
//! The solver does not reproduce Gartling's numbers, so the comparison is
//! written as two tests over **the same scene, budget and readout**: one
//! asserting the literature values and carrying `#[ignore]`, one asserting
//! what the solver produces today and carrying none. Exactly one of the pair
//! is enabled at a time, so the scene is simulated once.
//!
//! | | asserts | runs |
//! |---|---|---|
//! | `the_reattachment_length_matches_gartling` | `x_1 = 6.10`, bubble `4.85 .. 10.48`, `L_u = 5.63`, each to within a cell | no — `#[ignore = "src gap: …"]`, it fails |
//! | `the_reattachment_length_is_pinned_at_the_resolution_ci_can_afford` | `x_1 = 4.392103184 ± 1e-3`, no upper-wall bubble | **yes**, at `+154 s` per CI `Test` job |
//!
//! ⚠️ **The pinned twin is deliberately expensive, and the decision to keep it
//! running was made explicitly.** It was landed enabled, `#[ignore]`d for one
//! commit when CI measured `+154 s` per `Test` job against a 60 s budget, and
//! then enabled again with the budget raised instead. The cost is entirely
//! that one test: measured locally, the 6 tests predating layer 1 take
//! `14.20 s`, adding the `separation_x` contract and the reflected-step
//! control takes `14.81 s`, and adding the pin takes `69.57 s`.
//!
//! ⚠️ **The argument that settled it was not about seconds.** `#[ignore]` on a
//! test that *passes* is a deletion performed procedurally: there is no runner
//! in this repository for an ignored test to be deferred *to*, so "ignored
//! for cost" and "removed" have identical consequences. Cost belongs in a
//! schedule — `quality-deep.yml` already has one — and not in an attribute
//! that reads as configuration.
//!
//! Measured on the shared scene (`ny = 8`, `L = 12H`, `dt = 1/16`, `t = 256`,
//! 30 sweeps):
//!
//! | quantity | solver | Gartling | gap |
//! |---|---|---|---|
//! | `x_1` | 4.392103184 | 6.10 | 72.0 %, short by 1.708 = 13.7 cells |
//! | upper `x_2` | none | 4.85 | no bubble at all |
//! | upper `x_3` | none | 10.48 | — |
//! | `L_u` | — | 5.63 | — |
//!
//! ⚠️ **The two ignore reasons in this repository mean different things and
//! are labelled differently.** `runtime:` passes but is too slow for the
//! per-push matrix — the refinement sweep at the bottom of this file.
//! `src gap:` **fails**, and would cost nothing extra to run — the literature
//! twin above. Reading `#[ignore]` as a single category loses the difference
//! between something waiting on a schedule and something waiting on the
//! solver, and those want opposite handling: the first should be run
//! somewhere, the second should be *reported* when it starts passing.
//!
//! ⚠️ **Nothing in this repository runs `#[ignore]`d tests** — there is no
//! `--ignored` or `--include-ignored` anywhere in `scripts/` or `.github/`, so
//! the attribute keeps a test compiling and nothing more. In particular
//! **nobody will be told when the gap closes and the ignored twin starts
//! passing**; writing a reversal condition into a doc comment does not create
//! anything that fires it. There are **51** `#[ignore]`s in this repository and
//! nothing executes any of them; giving them somewhere to run is filed in the
//! backlog. ⚠️ **That is the reason the pinned twin is not ignored despite
//! costing what it costs** — it is the half of layer 1 that can notice a
//! change, and suppressing it would have left the whole layer as prose the
//! compiler type-checks.
//!
//! ## Why the pinned scene is not Gartling's
//!
//! Gartling's channel is `L = 30` and his own computations are far finer than
//! `ny = 8`. Both departures are measured rather than assumed — the length
//! tables are in the doc comment of the pinned twin. The short version:
//! `x_1` stops depending on `L` by `L = 12` (`L = 12` and `L = 16` agree to
//! `1e-6` at `Re = 800`), and `dt = 1/16` rather than `1/32` halves the cost
//! while moving `x_1` by `2.3e-3`, whereas `dt = 1/8` leaves the range where
//! the time discretisation is converging and was rejected.
//!
//! ⚠️ **The gap is dominated by resolution and the remainder is unknown.**
//! Refinement moves both quantities toward the reference (`x_1 = 4.3944` at
//! `ny = 8`, `>= 4.7220` at `ny = 16`, `>= 5.3032` at `ny = 32`, and the upper
//! bubble appears at `ny = 16`), but the finer two are lower bounds that never
//! settled in time, so **no convergence order can be read off them** and
//! nothing here shows whether a residual survives refinement. Saying the
//! literature gap is "discretisation" would be claiming more than was
//! measured.
//!
//! ## The upper wall reads zero, and why that is not self-certifying
//!
//! The pinned twin asserts that there is **no** upper-wall bubble, which is a
//! quantity reading zero — the shape of statement this repository has been
//! caught by before, where zero meant nothing was reaching the measurement
//! rather than that the measurement was zero. Three things keep it honest, and
//! the first two are not enough on their own:
//!
//! 1. the same field's lower wall carries 35 reversed faces, so the readout
//!    is not blind to reverse flow in general;
//! 2. the upper row is checked to be a *wall-adjacent* row (its outlet value
//!    is below mid-channel), which excludes an interior row, while the "no
//!    reversal" assertion itself excludes the opposite wall's row;
//! 3. **a positive control on a field that does have a bubble there** —
//!    reflecting the step to the other wall must reflect the answer, and
//!    `the_upper_wall_readout_finds_the_bubble_of_the_reflected_step` checks
//!    that the reflected reattachment equals the original to every printed
//!    digit (`0.1429734239751133` both ways) with the fields agreeing to
//!    43 ulp.
//!
//! ⚠️ **Mutation M5 below is the reason (3) exists**: pointing the upper
//! readout one row away from the wall leaves the pinned twin **green** — it
//! still finds no reversal, and that row is still below mid-channel — and only
//! the reflected-step control turns red. Without it "no bubble at `ny = 8`"
//! and "the upper readout does not work" are the same test result.
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
//! `9.0e-2` — where `ny = 8` moves by `2.47e-4` over the same interval. So the
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
//!
//! ## Layer 1, added later
//!
//! Four more mutations, same protocol, `src/` restored after each and verified
//! with `git diff --stat -- src/` coming back empty. The marker is printed
//! from a test this time rather than from the shell, for the reason recorded
//! above.
//!
//! | # | mutation | red | still green | restored |
//! |---|---|---|---|---|
//! | M5 | *(test side)* `top_row` reads `ny - 2`, one row in from the upper wall | the reflected-step control (`Re = 16` reflected shows no bubble on that row) | ⚠️ **the pinned twin, including its "no upper bubble" and wall-adjacency assertions** — and layers 2, 3, 4 | green |
//! | M6 | *(test side)* `separation_x` interpolates from the wrong end of the bracketing pair | the `separation_x` contract (`0.3125` against `0.4375`), and the control's "the reflected bubble opens at the step" | the pinned twin, layers 2, 3, 4 | green |
//! | M7 | `u_wall_across_y` returns `None` on the `-y` side, so the lower wall alone loses its no-slip ghost | the pinned twin **via its settling assertion** (`x_1` drifts `1.47e-2` between the two sample points, against a `1e-3` budget) and the control (`Re = 16` no longer separates at all) | the `separation_x` contract | green |
//! | M8 | `diffuse_velocity` doubles the viscous coefficient (`y`-reflection preserved) | the pinned twin **via its window** (`x_1 = 2.414514151`, settled to `5.3e-6`, so settling passes and the pin catches it) and the control | the `separation_x` contract | green |
//!
//! ⚠️ **M7 and M8 are caught by different assertions of the same test, and
//! that is why it carries both.** M7 leaves `x_1` near its pinned value
//! (`4.3175` against `4.3921`, inside a window three times wider) but destroys
//! its steadiness; M8 leaves it perfectly steady and moves it by `2.0`. A test
//! with only the window would miss M7; one with only the settling check would
//! miss M8.
//!
//! ⚠️ **M5 was re-run after the reflected-step control was changed to print
//! its measurements before asserting**, because the first version of that test
//! unwrapped the abscissae before printing and so produced a panic message
//! with **no measured value in it at all**. The failure now reads:
//!
//! ```text
//! Re=  16 24x8 steps=640: reflection residual 43 ulp = 2.331e-18
//!   lower x_r Some(0.1429734239751133)
//!   upper x_r None  upper x_s None
//!   reversed faces: upper wall of the reflected scene 0, same row of the
//!                   unreflected scene 0 (expected >= 1 and 0)
//!   upper row (reflected)   [0.0, 0.0311…, 0.1056…, …]   <- attached throughout
//!   bottom row (unreflected) [0.0, -0.0032…, 0.0194…, …] <- reverses at once
//! ```
//!
//! — from which the cause is readable rather than guessable: the residual is
//! at its usual floor, so the reflection itself is intact; the reference half
//! still locates its crossing; and the row being read as "the upper wall" is
//! positive everywhere while the bottom row is not. That combination says the
//! readout is pointing at the wrong row, which is what M5 did. ⚠️ **An
//! assertion can be correct and still leave a regression undiagnosable**, and
//! nothing about a green run reveals it.
//!
//! ⚠️ **M8 went red on the reflected-step control too, which was not the
//! prediction** — the expectation was that a `y`-symmetric mutation would
//! leave it green and thereby show the pin catching something the control
//! cannot. It failed earlier than that instead: doubling the viscosity removes
//! the `Re = 16` bubble altogether, so the control's own precondition ("the
//! step wall must separate and reattach") fires before any reflection residual
//! is computed, and **the residual was never measured under M8**. So this set
//! demonstrates the control catching what the pin cannot (M5) but not the
//! converse; a mutation that moves `x_1` at `Re = 800` while leaving a bubble
//! at `Re = 16` would be needed for that and was not constructed.

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

/// Which wall the step sits against. [`StepSide::Upper`] is the same problem
/// reflected in `y`, which is what makes it a positive control for the
/// upper-wall readout: the wall that carries the bubble swaps over, and nothing
/// else about the discretisation does.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum StepSide {
    Lower,
    Upper,
}

/// Gartling's backward-facing step: downstream height `H = 1` in `2 s_cells`
/// cells, step height `S = H/2`, channel length `nx dx`, `nu = 1/nu_recip`
/// and therefore `Re = nu_recip`.
fn backward_facing_step(s_cells: usize, nx: usize, nu_recip: i64) -> CfdSolver {
    step_scene(s_cells, nx, nu_recip, StepSide::Lower)
}

/// [`backward_facing_step`] with the step against either wall. Row `j` of the
/// `Upper` scene is row `ny - 1 - j` of the `Lower` one, inflow column
/// included, so the two fields are reflections of each other and any
/// difference between them is arithmetic.
fn step_scene(s_cells: usize, nx: usize, nu_recip: i64, side: StepSide) -> CfdSolver {
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
    // Half of the inflow plane is the step face and the other half carries the
    // inlet column — which half, and in which order, is the reflection.
    let column = inlet_column(s_cells);
    for m in 0..s_cells {
        let (step_j, flow_j, source) = match side {
            StepSide::Lower => (m, s_cells + m, m),
            StepSide::Upper => (s_cells + m, m, s_cells - 1 - m),
        };
        grid.set_u_bc(0, step_j, 0, WALL);
        grid.set_u_bc(
            0,
            flow_j,
            0,
            FaceBc::Inflow {
                normal_velocity: column[source],
            },
        );
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

/// The row of `u` faces nearest the **upper** wall, `u(i, ny - 1, 0)`.
///
/// ⚠️ Which row this is cannot be checked by "it found no bubble" — that is
/// what an inoperative readout also reports. The two statements are separated
/// here by asserting, on the same field, that this row is a *near-wall* row
/// (its outlet value is below the one at mid-channel, which excludes an
/// interior row) while carrying no reversal (which excludes the opposite
/// wall's row, where 35 faces are reversed); and by
/// [`the_upper_wall_readout_finds_the_bubble_of_the_reflected_step`], which
/// runs it on a field that does have a bubble against this wall.
fn top_row(grid: &MacGrid) -> Vec<f64> {
    (0..=grid.nx)
        .map(|i| grid.u(i, grid.ny - 1, 0).to_f64())
        .collect()
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

/// Separation abscissa: the mirror image of [`reattachment_x`] — the first
/// place `row` *starts* being negative, between the last non-negative face `i`
/// and the first negative one `i + 1`.
///
/// Needed because Gartling's upper-wall bubble is detached from the inflow
/// plane (`x_2 = 4.85`), so unlike the lower wall its opening is an interior
/// crossing and has to be located rather than assumed to be at `x = 0`.
fn separation_x(row: &[f64], dx: f64) -> Option<f64> {
    (0..row.len().saturating_sub(1))
        .find(|&i| row[i] >= 0.0 && row[i + 1] < 0.0)
        .map(|i| dx * (i as f64 + row[i] / (row[i] - row[i + 1])))
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
/// | 8 | 4.394114 | 4.394360 |
/// | 16 | 4.631712 | 4.721966 |
///
/// (⚠️ the `ny = 8` entry at `t = 256` read `4.394400` when this table first
/// landed, and the `2.9e-4` computed from it appeared here and in the module
/// header. Re-measured on the same scene: `4.394113515` at `t = 128`, which
/// reproduces the landed `t = 128` figure exactly, and `4.394360266` at
/// `t = 256`. Since the same scene reproduces bit-for-bit — this is
/// fixed-point arithmetic — the disagreement was a transcription slip and not
/// a loss of determinism. The rounded `4.3944` used in the inequality below is
/// unaffected either way.)
///
/// ⚠️ **The step count at which `ny = 16` settles has since been found, by a
/// separate sweep of 40 sample points from `t = 16` to `t = 640`: it reaches
/// `4.723804046` and is bit-stable from `t = 464`, with `1e-6` agreement by
/// `t ~ 480`. `ny = 8` settles to `4.394360271`, and the approach is monotone
/// in time at both resolutions (39 consecutive differences, none negative).**
/// That means the a fortiori inequality below could be replaced by the direct
/// comparison `4.723804046 > 4.394360271`. ⚠️ **Deliberately not done here** —
/// it changes what the gate claims rather than correcting a number, so it
/// belongs to its own increment and its own review. Recorded so the next
/// person does not re-measure it.
///
/// so at `ny = 8` the answer has stopped moving (`2.47e-4` apart) while at
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

// ===========================================================================
// Layer 1 — the comparison against Gartling (1990) at Re = 800
// ===========================================================================

/// Gartling (1990), `Re = 800`, normalised to `H = 1`. Transcribed in
/// `memory/reference_armaly_gartling_values.md`, where each value is recorded
/// with the sources that agree on it; only the ones that agree to the last
/// printed digit are used as targets here.
///
/// Lower-wall reattachment. Three independent transcriptions agree
/// (`6.10` on `H = 1`, `12.20` and `12.2` on the step height `S = 0.5`).
const GARTLING_X_1: f64 = 6.10;
/// Upper-wall separation, two transcriptions (`4.85`, and `9.7` on `S`).
const GARTLING_X_2: f64 = 4.85;
/// Upper-wall bubble length `x_3 - x_2`, two transcriptions agreeing to the
/// last digit (`5.63`, and `11.26` on `S`).
const GARTLING_L_U: f64 = 5.63;
/// Upper-wall reattachment. ⚠️ **Rounding-consistent only**, not agreeing to
/// the last printed digit: one source prints `10.48` (`20.96` on `S`) and the
/// other `21.0` on `S`, i.e. `10.5`. Carried with the uncertainty below rather
/// than used as an exact target, which is why the bubble *length* is the hard
/// statement and this is the soft one.
const GARTLING_X_3: f64 = 10.48;
/// Spread between the two transcriptions of [`GARTLING_X_3`].
const GARTLING_X_3_UNCERTAINTY: f64 = 0.02;

// The scene and budget both layer-1 twins run. They share these so that the
// twins differ **only** in what they assert — see the module header.
/// `ny = 2 * 4 = 8`, `dx = 1/8`.
const LIT_S_CELLS: usize = 4;
/// `L = 12 H`. Gartling's channel is `L = 30`; measured, `x_1` does not depend
/// on it once the outflow is far enough away (table in the module header).
const LIT_LENGTH: f64 = 12.0;
/// `dt = 1/16`, i.e. a CFL number of `0.75` against the inlet peak `1.5`.
const LIT_DT_RECIP: i64 = 16;
/// `t = 256`. `x_1` is settled to better than `1e-3` by half this (asserted).
const LIT_STEPS: u32 = 4096;
/// Half the 60 sweeps the other scenes in this file use, because this is the
/// one run that has to fit in a CI budget and the sweeps are the bulk of its
/// cost. ⚠️ Measured not to move the answer: `x_1 = 4.392_103_184` here
/// against `4.392_103_202` at 60 sweeps, agreeing to `1.8e-8`, while the flux
/// imbalance is `2.4e-8` — four orders inside the `1e-3` this test asserts.
/// ⚠️ 15 sweeps was rejected: it reaches the same `x_1` (`4.392_103_100`) but
/// gets there more slowly in time, so the settling check below reads `1.6e-3`
/// between its two sample points and no longer demonstrates what it claims.
const LIT_JACOBI_SWEEPS: u32 = 30;

/// What one run of the shared layer-1 scene yields.
///
/// The twins share the scene, the budget and the readout — not one execution:
/// each calls [`measure_gartling_re_800`] itself, so enabling both runs the
/// simulation twice. Today exactly one of them is enabled, and the flip
/// described on the ignored twin keeps that true.
struct StepMeasurement {
    /// Lower-wall reattachment at the end of the budget.
    x_1: f64,
    /// The same, half-way through, so that settling is visible in one run.
    x_1_halfway: f64,
    /// Upper-wall bubble, if the field has one.
    upper_separation: Option<f64>,
    upper_reattachment: Option<f64>,
    /// Reversed faces in the wall-adjacent rows.
    upper_reversed: usize,
    lower_reversed: usize,
    /// First face downstream of the step on the lower wall; negative while the
    /// bubble starts at the step, as it must.
    lower_first_face: f64,
    /// Outlet `u` in the upper wall-adjacent row and at mid-channel. The first
    /// must be the smaller — that is what says which row was read.
    outlet_near_upper_wall: f64,
    outlet_mid_channel: f64,
    max_abs_div: f64,
    worst_flux_imbalance: f64,
}

/// Runs the shared layer-1 scene once and reads everything off it.
fn measure_gartling_re_800() -> StepMeasurement {
    let ny = 2 * LIT_S_CELLS;
    let dx = 1.0 / ny as f64;
    let nx = (LIT_LENGTH / dx).round() as usize;
    let dt = Fix128::from_ratio(1, LIT_DT_RECIP);
    let mut solver = backward_facing_step(LIT_S_CELLS, nx, 800);
    solver.jacobi_iterations = LIT_JACOBI_SWEEPS;

    for _ in 0..LIT_STEPS / 2 {
        solver.step(dt);
    }
    let x_1_halfway = reattachment_x(&bottom_row(&solver.grid), dx)
        .expect("the lower wall must separate at Re = 800 half-way through the budget");
    for _ in LIT_STEPS / 2..LIT_STEPS {
        solver.step(dt);
    }

    let lower = bottom_row(&solver.grid);
    let upper = top_row(&solver.grid);
    let x_1 =
        reattachment_x(&lower, dx).expect("the lower wall must separate and reattach at Re = 800");

    let inflow = column_flux(&solver.grid, 0);
    let mut worst_flux_imbalance = 0.0f64;
    for i in 1..=nx {
        let imbalance = (column_flux(&solver.grid, i) - inflow).abs().to_f64();
        worst_flux_imbalance = worst_flux_imbalance.max(imbalance);
    }

    StepMeasurement {
        x_1,
        x_1_halfway,
        upper_separation: separation_x(&upper, dx),
        upper_reattachment: reattachment_x(&upper, dx),
        upper_reversed: upper.iter().filter(|u| **u < 0.0).count(),
        lower_reversed: lower.iter().filter(|u| **u < 0.0).count(),
        lower_first_face: lower[1],
        outlet_near_upper_wall: upper[nx],
        outlet_mid_channel: solver.grid.u(nx, ny / 2, 0).to_f64(),
        max_abs_div: max_abs_divergence(&solver.grid),
        worst_flux_imbalance,
    }
}

/// One printed summary, shared by the twins so their outputs are comparable.
fn report(measurement: &StepMeasurement) {
    let m = measurement;
    let dx = 1.0 / (2 * LIT_S_CELLS) as f64;
    println!(
        "Gartling Re=800, ny={} dx=1/{} L={} dt=1/{} N={} sweeps={LIT_JACOBI_SWEEPS}:\n  \
         x_1 = {:.9}  (half-way {:.9}, delta {:.3e})  reference {GARTLING_X_1} \
         => {:.1} % of it, short by {:.4} = {:.1} cells\n  \
         upper wall: {} reversed faces, separation {:?}, reattachment {:?}  \
         reference {GARTLING_X_2} .. {GARTLING_X_3} (L_u = {GARTLING_L_U})\n  \
         lower wall: {} reversed faces, first face {:+.6e}\n  \
         outlet u: near upper wall {:.6}, mid-channel {:.6}\n  \
         max |div u| {:.3e}, worst flux imbalance {:.3e}",
        2 * LIT_S_CELLS,
        2 * LIT_S_CELLS,
        LIT_LENGTH,
        LIT_DT_RECIP,
        LIT_STEPS,
        m.x_1,
        m.x_1_halfway,
        (m.x_1 - m.x_1_halfway).abs(),
        100.0 * m.x_1 / GARTLING_X_1,
        GARTLING_X_1 - m.x_1,
        (GARTLING_X_1 - m.x_1) / dx,
        m.upper_reversed,
        m.upper_separation,
        m.upper_reattachment,
        m.lower_reversed,
        m.lower_first_face,
        m.outlet_near_upper_wall,
        m.outlet_mid_channel,
        m.max_abs_div,
        m.worst_flux_imbalance,
    );
}

/// Oracle: the contract of [`separation_x`], against rows written down here,
/// so that locating the *opening* of the upper-wall bubble is pinned
/// independently of any flow — the counterpart of
/// [`the_reattachment_detector_reports_the_first_sign_change`].
///
/// ⚠️ This is one of the two halves that keep "the upper wall has no bubble"
/// from being the same statement as "the upper-wall readout does not work".
/// This half fixes what the locator means; the other half,
/// [`the_upper_wall_readout_finds_the_bubble_of_the_reflected_step`], shows
/// the readout firing on a real field.
#[test]
fn the_separation_detector_reports_the_first_crossing_into_reverse_flow() {
    let dx = 0.25f64;
    // Crossing between index 1 (+3) and 2 (-1): 1 + 3/4 cells. Asymmetric on
    // purpose — swapping the endpoints or sliding the index-to-x mapping by
    // half a cell gives 1.25 or 2.25 instead.
    assert_eq!(
        separation_x(&[1.0, 3.0, -1.0, -2.0], dx),
        Some(dx * 1.75),
        "row[i] sits at x = i dx and the crossing into reverse flow is \
         interpolated between the bracketing faces"
    );
    // Mirrored magnitudes one cell further along: 2 + 1/4 cells. Using the
    // wrong endpoint of the pair gives 2.75 here and 1.25 above.
    assert_eq!(
        separation_x(&[2.0, 1.0, 1.0, -3.0], dx),
        Some(dx * 2.25),
        "the weight is the non-negative face's share of the drop across the pair"
    );
    // Zero counts as attached, so a row that opens at the very first face —
    // which is what a step at the inflow plane gives — separates at x = 0.
    assert_eq!(separation_x(&[0.0, -1.0, -2.0], dx), Some(0.0));
    // Never reverses, and reverses only after it has already come back.
    assert_eq!(separation_x(&[1.0, 2.0, 3.0], dx), None);
    assert_eq!(
        separation_x(&[1.0, -1.0, 1.0, -1.0], dx),
        Some(dx * 0.5),
        "the first opening, not the second"
    );
    // Rows too short to bracket anything.
    assert_eq!(separation_x(&[1.0], dx), None);
    assert_eq!(separation_x(&[], dx), None);
    println!("separation_x contract: 7 hand-written rows agree");
}

/// Oracle: **the positive control for the upper-wall readout.** Reflecting the
/// step to the other wall has to reflect the answer, so the row that reports
/// "no bubble" on Gartling's scene reports the *same* bubble the lower wall
/// had, once the bubble is against it.
///
/// This is the statement that makes the absence of an upper-wall bubble at
/// `Re = 800` mean something. Without it, `upper_reversed == 0` is equally
/// well explained by a readout that looks at the wrong row, or a locator that
/// never fires — the failure mode this repository keeps meeting, where a
/// quantity reads zero because nothing reaches it rather than because it is
/// zero.
///
/// Measured, `24 x 8`, `L = 3H`, `dt = 1/32`:
///
/// | `Re` | steps | reflection residual | lower `x_r` (`Lower`) | upper `x_r` (`Upper`) |
/// |---|---|---|---|---|
/// | 16 | 640 | 43 ulp | 0.1429734239751133 | 0.1429734239751133 |
/// | 96 | 1600 | 142 ulp | 0.7502761449264386 | 0.7502761449264386 |
///
/// — the two abscissae agree in **every printed digit**, and the fields agree
/// to the arithmetic floor rather than exactly, which is expected: the two
/// runs execute different traversal orders over the same arithmetic, so the
/// Gauss-Seidel sweeps accumulate their roundings in a different sequence.
/// The field residual is the honest bound (`142 ulp = 7.7e-18`); the
/// abscissae, being interpolated in `f64`, land on the same double.
///
/// ⚠️ **Only the `Re = 16` row is run.** `Re = 96` needs 1600 steps on two
/// scenes, which is `36.9 M` Poisson cell-sweeps — as much as every other test
/// in this file put together — and it is a second instance of a statement the
/// first row already makes. It is recorded above rather than gated, the same
/// way `ny = 32` is recorded rather than gated in the refinement sweep.
#[test]
fn the_upper_wall_readout_finds_the_bubble_of_the_reflected_step() {
    let (s_cells, nx) = (4usize, 24usize);
    let ny = 2 * s_cells;
    let dx = 1.0 / ny as f64;
    let dt = Fix128::from_ratio(1, 32);

    for &(re, steps) in &[(16i64, 640u32)] {
        let mut lower = step_scene(s_cells, nx, re, StepSide::Lower);
        let mut upper = step_scene(s_cells, nx, re, StepSide::Upper);
        for _ in 0..steps {
            lower.step(dt);
            upper.step(dt);
        }

        // The fields are reflections: row j of one is row ny-1-j of the other.
        let mut worst = 0u128;
        for j in 0..ny {
            for i in 0..=nx {
                worst = worst.max(ulp_gap(
                    lower.grid.u(i, j, 0),
                    upper.grid.u(i, ny - 1 - j, 0),
                ));
            }
        }

        let lower_row = bottom_row(&lower.grid);
        let upper_row = top_row(&upper.grid);
        let control_row = top_row(&lower.grid);
        let lower_x_r = reattachment_x(&lower_row, dx);
        let upper_x_r = reattachment_x(&upper_row, dx);
        let upper_x_s = separation_x(&upper_row, dx);
        let upper_reversed = upper_row.iter().filter(|u| **u < 0.0).count();
        let control_reversed = control_row.iter().filter(|u| **u < 0.0).count();

        // ⚠️ Everything is printed *before* the first assertion, and the
        // abscissae are still `Option` here on purpose. An earlier version
        // unwrapped them first and printed afterwards, which meant a
        // regression produced a panic message and **not one measured number** —
        // the rows, the residual and the reversed-face counts were all
        // unreachable on the failing path. Whatever broke this has to be
        // readable from the output alone.
        println!(
            "Re={re:4} {nx}x{ny} steps={steps}: reflection residual {worst} ulp = {:.3e}\n  \
             lower x_r {lower_x_r:?}\n  upper x_r {upper_x_r:?}  upper x_s {upper_x_s:?}\n  \
             reversed faces: upper wall of the reflected scene {upper_reversed}, \
             same row of the unreflected scene {control_reversed} (expected >= 1 and 0)\n  \
             upper row (reflected)   {upper_row:?}\n  \
             bottom row (unreflected) {lower_row:?}",
            worst as f64 * ULP,
        );

        // The readout fires: there is a bubble against the upper wall, it
        // opens at the step, and it is located — this is the control.
        assert!(
            upper_reversed >= 1,
            "Re={re}: the reflected step must reverse the flow against the upper \
             wall, or the upper-wall readout is not reading that wall; found \
             {upper_reversed} reversed faces in the row printed above"
        );
        let Some(upper_x_s) = upper_x_s else {
            panic!(
                "Re={re}: the reflected bubble must open at the step plane, but no \
                 crossing into reverse flow was located at all in a row with \
                 {upper_reversed} reversed faces — so `separation_x` is not \
                 finding what the row contains"
            );
        };
        assert!(
            upper_x_s < dx,
            "Re={re}: the reflected bubble opens at the step plane, i.e. below \
             x = {dx}, got {upper_x_s}"
        );
        // And it is the *same* bubble, which is what makes it a control on the
        // value and not only on the readout firing.
        let Some(lower_x_r) = lower_x_r else {
            panic!(
                "Re={re}: the step wall must separate and reattach — no crossing \
                 on the bottom row printed above, so the reference half of the \
                 reflection is missing and the comparison cannot be made"
            );
        };
        let Some(upper_x_r) = upper_x_r else {
            panic!(
                "Re={re}: the reflected step must put the same bubble against the \
                 upper wall, but no reattachment was located there while the \
                 unreflected scene reattaches at {lower_x_r:.16}; the upper row \
                 printed above has {upper_reversed} reversed faces"
            );
        };
        assert_eq!(
            upper_x_r,
            lower_x_r,
            "Re={re}: reflecting the scene must reflect the reattachment point \
             (difference {:.3e}), so either the discretisation is not symmetric \
             in y or the two readouts are not reading mirrored rows",
            upper_x_r - lower_x_r
        );
        // The reflection is exact to the arithmetic floor. Measured 43 ulp at
        // Re = 16 and 142 at Re = 96; the budget is a constant because this is
        // rounding in the pressure sweeps, not something that accumulates with
        // the mesh. ⚠️ Reversal condition: if this grows, widen the constant
        // and record the new value — do not convert it to a `to_f64`
        // tolerance, which at 1e-17 against a field of order 1 would pass on
        // anything.
        assert!(
            worst <= 512,
            "Re={re}: reflecting the scene must reflect the field to the \
             arithmetic floor, off by {worst} ulp = {:.3e}",
            worst as f64 * ULP
        );
        // Not vacuous: the unreflected field has nothing against that wall, so
        // the assertions above are about the reflection and not about every
        // field having a bubble everywhere.
        assert_eq!(
            control_reversed, 0,
            "Re={re}: the unreflected scene must have no upper-wall reverse \
             flow, else the control says nothing — found {control_reversed} \
             reversed faces in a row that should be attached everywhere, which \
             would mean the two scenes are not reflections of one another"
        );
    }
}

/// Oracle: **the target.** `x_1` and the upper-wall bubble equal Gartling's
/// values, to within one cell of the mesh that produced them.
///
/// # Why this is `#[ignore]`d, and what would remove the attribute
///
/// ⚠️ **`src gap`, not runtime.** This test runs the same scene and the same
/// number of steps as its twin
/// [`the_reattachment_length_is_pinned_at_the_resolution_ci_can_afford`],
/// which is not ignored, so it costs the same and would run in CI. It is
/// ignored because **it does not pass**: the solver reaches `x_1 = 4.3921`
/// against Gartling's `6.10`, i.e. 72.0 % of it, short by `1.708 = 13.7`
/// cells, and produces **no upper-wall bubble at all** where Gartling has one
/// spanning `4.85 .. 10.48`.
///
/// ⚠️ **Reversal condition, verbatim: when `x_1` reaches `6.10` to within one
/// cell, delete the `#[ignore]` on this test and put
/// `#[ignore = "superseded by the literature comparison"]` on the pinned twin
/// below.** The twin exists to notice the solver moving at all; this one
/// exists to say where it has to get to. They are written against one shared
/// measurement so that the flip is those two attribute edits and nothing else.
///
/// ⚠️ **`#[ignore]`d tests are not run anywhere in this repository** — there
/// is no `--ignored` or `--include-ignored` invocation in `scripts/` or
/// `.github/`, so nothing will tell anyone when this starts passing. The
/// attribute keeps it compiling, and the twin is what actually guards the
/// number. Giving the gap-ignored twins somewhere to run is filed in the
/// backlog; until that exists, this test is a written-down target that the
/// compiler keeps honest, and saying otherwise would overstate it.
///
/// # Where the tolerance comes from
///
/// One cell, `dx`. Not a fitted number and not the literature's own precision
/// — Gartling prints `6.10`, so his rounding is `±0.005`, two orders tighter
/// than anything claimable here. A reattachment point is located by
/// interpolating between two faces `dx` apart, so agreement to within a cell
/// of the mesh is the strongest statement a mesh of that spacing supports,
/// and it tightens automatically under refinement, which is the direction the
/// gap has to close in. The bubble length gets `2 dx`, being a difference of
/// two located crossings. [`GARTLING_X_3`] additionally carries
/// [`GARTLING_X_3_UNCERTAINTY`], because its two transcriptions agree only
/// after rounding — which is why the hard statement is the bubble *length*
/// [`GARTLING_L_U`], where they agree to the last digit.
///
/// # What is known about the gap
///
/// It is **dominated by resolution**, and how much of it survives refinement
/// is undetermined. Measured (`L = 16`, `t = 256`, in the module header):
/// `x_1 = 4.3944` at `ny = 8`, `>= 4.7220` at `ny = 16`, `>= 5.3032` at
/// `ny = 32` — moving toward `6.10`, and the upper-wall bubble appears at
/// `ny = 16` and lengthens at `ny = 32`. ⚠️ **The finer two are lower bounds
/// and no convergence order can be read off them**, because neither had
/// settled in time within the step budget that was affordable; that is a
/// different situation from "the order is wrong", and conflating the two would
/// claim the residual is not discretisation when nothing here shows that.
/// So the honest statement is: refinement moves both quantities the right way,
/// and whether `ny = 8` could reach `6.10` under a better wall treatment or a
/// higher-order advection — rather than only under refinement — is not known.
#[test]
#[ignore = "src gap: x_1 = 4.3921 against Gartling's 6.10 (72.0 %, short by 13.7 cells) and no upper-wall bubble at all; the gap is dominated by resolution and the remainder is undetermined. Distinct from the runtime ignore on its twin: this one fails, and costs the same to run"]
fn the_reattachment_length_matches_gartling() {
    let dx = 1.0 / (2 * LIT_S_CELLS) as f64;
    let m = measure_gartling_re_800();
    report(&m);

    assert!(
        (m.x_1 - GARTLING_X_1).abs() <= dx,
        "x_1 must equal Gartling's {GARTLING_X_1} to within one cell ({dx}), got \
         {:.6} — short by {:.4}",
        m.x_1,
        GARTLING_X_1 - m.x_1
    );
    let x_2 = m
        .upper_separation
        .expect("Gartling's field separates from the upper wall at x_2 = 4.85");
    let x_3 = m
        .upper_reattachment
        .expect("Gartling's upper-wall bubble closes again at x_3 = 10.48");
    assert!(
        (x_2 - GARTLING_X_2).abs() <= dx,
        "the upper wall must separate at {GARTLING_X_2} to within one cell, got {x_2:.6}"
    );
    assert!(
        (x_3 - GARTLING_X_3).abs() <= dx + GARTLING_X_3_UNCERTAINTY,
        "the upper wall must reattach at {GARTLING_X_3} to within one cell plus \
         the {GARTLING_X_3_UNCERTAINTY} spread between its two transcriptions, got {x_3:.6}"
    );
    assert!(
        (x_3 - x_2 - GARTLING_L_U).abs() <= 2.0 * dx,
        "the upper-wall bubble must be {GARTLING_L_U} long to within two cells \
         (it is a difference of two located crossings), got {:.6}",
        x_3 - x_2
    );
}

/// Oracle: the twin of the above — what the solver does produce on that scene
/// today, so that it moving is noticed even though the target is out of reach.
///
/// # ⚠️ It runs, and it is the most expensive test in this file
///
/// It was landed enabled, `#[ignore]`d for one commit when CI measured it at
/// `+154 s` per `Test` job against a 60 s budget, and then enabled again: the
/// budget was raised rather than the test dropped. The reasoning was that a
/// **passing** change-detector suppressed by `#[ignore]` is a deletion carried
/// out procedurally, and that this repository has no runner for ignored tests
/// to be suppressed *into* — of the 51 `#[ignore]`s here, none are executed by
/// anything in `scripts/` or `.github/`. The cost is recorded below so the
/// trade stays visible rather than becoming folklore.
///
/// Measured, one `cargo test --test armaly_backward_step` per row, same
/// machine, load average 3.2:
///
/// | tests run | wall |
/// |---|---|
/// | the 6 that predate layer 1 | 14.20 s |
/// | + `separation_x` contract + reflected-step control | 14.81 s |
/// | + this test | **69.57 s** |
///
/// so the two other layer-1 tests cost `+0.6 s` between them and **this one
/// costs `+54.8 s` by itself** (`76.99 s` of binary time when isolated on CI
/// hardware).
///
/// ⚠️ **The `Test` job runs the integration tests twice**, which is where a
/// factor of two between two honest measurements came from: `ci.yml` has both
/// `cargo test` and `cargo test --features "parallel"`, while its other two
/// steps are `--lib` and skip this file. The binary's own increment is
/// `76.99 s`; the job's is `2 x 76.99 = 154 s`. ⚠️ **Quoting a binary's
/// `finished in` as the CI cost understates it by that factor**, and a third
/// feature set would make it three.
///
/// On CI, measured three ways because two of them are not trustworthy on
/// their own: against its own parent commit, and then with this test ignored
/// for one commit as a control before it was enabled again.
///
/// | `Test` job | parent | this test on | this test off |
/// |---|---|---|---|
/// | ubuntu-latest | 547 | 701 | **547** |
/// | ubuntu-24.04-arm | 586 | 788 | 583 |
/// | macos-latest | 633 | 860 | 616 |
/// | macos-15-intel | 956 | 1155 | 869 |
/// | windows-latest | 442 | 751 | ⚠️ 716 |
///
/// ⚠️ **Only the third column establishes that the increment is this test.**
/// ubuntu-latest goes `547 -> 701 -> 547`, returning exactly; arm and
/// macos-latest return to within 3 s and 17 s. ⚠️ **windows stays 274 s above
/// its parent with the test switched off, and macos-15-intel lands 87 s
/// below** — those two jobs move by as much as the thing being measured, so
/// they cannot be used as instruments. The jobs that run no integration tests
/// at all moved `+1 s` and `+2 s`.
///
/// ⚠️ **A before/after pair cannot separate a real increment from runner
/// noise here; the third point — putting it back — is what does.** That is
/// also why the figure quoted above is `+154 s` from the one job that
/// round-tripped, rather than the `+154..+309` range the pair suggested.
///
/// ## Why the scene is the cheapest one that works
///
/// `ny = 8`, `L = 12 H`, `dt = 1/16`, `t = 256`, 30 sweeps — each measured
/// rather than assumed, because a gate nobody runs is indistinguishable from
/// an assertion nobody wrote, and in this repository `#[ignore]` means nobody
/// runs it, there being no `--ignored` invocation in `scripts/` or
/// `.github/`.
///
/// ⚠️ **Local seconds could not be trusted while this was being developed** —
/// load average 17.7 on 8 cores from sibling sessions, under which the same
/// 4096-step run timed anywhere between 65 s and 183 s and a 15-sweep run
/// appeared *slower* than a 60-sweep one. Poisson cell-sweeps were used
/// instead, being load-independent: this test is
/// `96 x 8 x 4096 x 30 = 94.37 M`, against
///
/// | test | cell-sweeps |
/// |---|---|
/// | `reattachment_grows_with_reynolds_number_on_the_step_field` | 36.86 M |
/// | `every_cross_section_telescopes_to_the_inflow_flux` | 19.66 M |
/// | `the_step_outlet_develops_into_the_two_term_profile` | 14.75 M |
/// | `the_upper_wall_readout_finds_the_bubble_of_the_reflected_step` | 14.75 M |
/// | `the_two_term_developed_profile_is_an_exact_fixed_point` | 5.90 M |
/// | **rest of this file** | **91.91 M** |
///
/// so this one test is `1.027x` the rest of the file put together. (⚠️ that
/// figure first read `1.22x` against `77.2 M`, which is the four solver tests
/// that **predate** layer 1 — it left out the reflected-step control, which
/// landed in the same commit and is part of "the rest of the file" from the
/// moment it exists. `1.22x` is still the right number against the
/// pre-layer-1 four, and that is the comparison `+141%` in the memory note
/// refers to. ⚠️ The two `14.75 M` entries agreeing is a coincidence:
/// `48 x 8` cells for 640 steps and `24 x 8` for 640 steps on two scenes are
/// both `384` cell-columns, so it is not double counting.)
///
/// ⚠️ **That proxy under-predicted the real cost by 2.8x** (it implied
/// `+20 s`; the measured figure once the machine went quiet was `+54.8 s`), so
/// it is fine for ranking two variants and not for deciding whether something
/// fits in a budget. The numbers above are wall clock.
///
/// ## `L = 12` rather than Gartling's `30`
///
/// `x_1` stops depending on the channel length well before Gartling's `L`.
/// Measured at `Re = 800`, `ny = 8`, `t = 128`, `dt = 1/32` and `dt = 1/16`:
///
/// | `L/H` | `x_1` at `dt = 1/32` | `x_1` at `dt = 1/16` |
/// |---|---|---|
/// | 6 | 4.403020 | 4.393623 |
/// | 8 | 4.394267 | 4.392235 |
/// | 12 | 4.394114 | 4.392103 |
/// | 16 | 4.394114 | 4.392102 |
/// | 30 | — | 4.392172 (still falling) |
///
/// `L = 12` and `L = 16` agree to `1e-6`, so the outflow has stopped mattering
/// by `12`; `L = 6` is `8.7e-3` away and `L = 8` still `1.5e-4`, so it has not
/// by `8`. ⚠️ **`Re = 96` cannot be used to make this argument** even though
/// `x_r/S` there is unchanged to four digits from `L = 6H` to `16H` (module
/// header): the Reynolds number differs by a factor of eight and with it the
/// length of everything downstream. The rows above are all at `Re = 800`.
/// `L = 30` reads `4.392172` while still descending toward the others, which
/// is its own transient and not a disagreement.
///
/// ## `dt = 1/16` rather than `1/32`
///
/// Halving the step count by doubling `dt` moves the answer in the fourth
/// digit and settles roughly twice as fast. Measured, `L = 12`, `ny = 8`:
///
/// | `dt` | CFL | `x_1` once settled | settled by |
/// |---|---|---|---|
/// | 1/8 | 1.5 | 4.358627 | `t = 256` (`N = 2048`) |
/// | 1/16 | 0.75 | 4.392103 | `t = 256` (`N = 4096`) |
/// | 1/32 | 0.375 | 4.394360 | `t = 256` (`N = 8192`) |
///
/// ⚠️ **`dt = 1/8` was rejected although it is half the cost again.** From the
/// settled column above, the successive differences are
/// `4.392103 - 4.358627 = 3.348e-2` and then
/// `4.394360 - 4.392103 = 2.257e-3`, a ratio of **14.8** per halving where a
/// first-order scheme gives 2 and a second-order one 4 — so `dt = 1/8` is not
/// in the range where the time discretisation is converging, which is
/// unsurprising at CFL 1.5, and pinning there would pin a number outside the
/// regime the other two share. `1/16` and `1/32` differ by `2.3e-3`,
/// consistent with each other.
///
/// (⚠️ this paragraph first read `2.0e-3` and a ratio of `16.7`, computed
/// against `4.394114`, which is the `dt = 1/32` value at `t = 128` — **not
/// settled**; the settled one at `t = 256` is `4.394360`. The `16.7` also
/// reached the message of the commit that landed this test and is **not being
/// rewritten there**, history being kept as it is, so this note is the record.
/// The conclusion does not depend on which figure is used: 14.8 and 16.7 are
/// both far outside the 2-to-4 band that would indicate convergence.)
///
/// ## Settling
///
/// Asserted rather than asserted-around: `x_1` is read at `t = 128` and
/// `t = 256` of the same run and the two must agree to `1e-3`. Measured on the
/// **30 sweeps this test actually uses**, `5.131e-4`. ⚠️ The figure depends on
/// the sweep count, because sweeps buy time-convergence rate here and not a
/// different answer:
///
/// | sweeps | `x_1(t=128)` vs `x_1(t=256)` | same `x_1` at `t = 256`? |
/// |---|---|---|
/// | 15 | `1.559e-3` — **outside the `1e-3` budget** | yes, `4.392103100` |
/// | **30** | **`5.131e-4`** | yes, `4.392103184` |
/// | 60 | `3.129e-4` | yes, `4.392103202` |
///
/// ⚠️ **An earlier draft quoted the 60-sweep `3.1e-4` next to the 30-sweep
/// configuration**, which made the margin look twice as comfortable as it is.
/// All three reach the same `x_1` to `1.8e-8`; what differs is how fast they
/// get there, which is exactly what this assertion measures and why 15 sweeps
/// was rejected.
///
/// ⚠️ **Steadiness cannot be taken from the flux imbalance**, which tracks the
/// Gauss-Seidel residual and not the slowest mode of the flow — measured at
/// `ny = 16` it reads `7.7e-7`, small enough to look settled, while `x_1` was
/// still climbing by `9.0e-2` per doubling. The quantity being pinned is the
/// quantity whose steadiness is checked.
///
/// ## The window
///
/// `1e-3` around the measured value, which is `1.95x` the `5.131e-4` the run
/// still moves by between its two sample points: anything inside it is the
/// same answer reached with a slightly different budget, and anything outside
/// it is the solver having moved. That is `0.02 %` of `x_1`. ⚠️ **`1.95x` is
/// tighter cover than it sounds** — the settling assertion above fails first if
/// the run stops settling, so this window only has to separate "same answer"
/// from "different answer", not to absorb the transient as well.
///
/// ⚠️ **If this test fails after a deliberate improvement, it has done its
/// job** — read the twin above and flip the two `#[ignore]` attributes rather
/// than widening this window.
#[test]
fn the_reattachment_length_is_pinned_at_the_resolution_ci_can_afford() {
    // Measured on this scene and budget. See the doc comment for the window.
    const PINNED_X_1: f64 = 4.392_103_184;
    const WINDOW: f64 = 1e-3;

    let m = measure_gartling_re_800();
    report(&m);

    // Settled, read off the pinned quantity itself.
    assert!(
        (m.x_1 - m.x_1_halfway).abs() < 1e-3,
        "x_1 must have settled within this budget, {:.9} at t = {} against \
         {:.9} at t = {}",
        m.x_1_halfway,
        LIT_STEPS / 2 / LIT_DT_RECIP as u32,
        m.x_1,
        LIT_STEPS / LIT_DT_RECIP as u32
    );
    // The pin.
    assert!(
        (m.x_1 - PINNED_X_1).abs() <= WINDOW,
        "x_1 was {PINNED_X_1} on this scene and is now {:.9} ({:+.3e}). If this \
         is a deliberate improvement toward Gartling's {GARTLING_X_1}, do not \
         widen the window — remove the #[ignore] from \
         `the_reattachment_length_matches_gartling` and mark this test \
         superseded",
        m.x_1,
        m.x_1 - PINNED_X_1
    );
    // The upper wall carries no bubble at this resolution, which is the other
    // half of the distance to Gartling and the thing refinement brings in.
    // ⚠️ Meaningful only next to the two statements that keep it from being
    // "the readout does not work": the row identity checked just below, and
    // the reflected-step control in
    // `the_upper_wall_readout_finds_the_bubble_of_the_reflected_step`.
    assert_eq!(
        m.upper_reversed, 0,
        "eight cells cannot carry the upper-wall boundary layer, so there must \
         be no reverse flow there yet; Gartling's bubble at \
         {GARTLING_X_2}..{GARTLING_X_3} is what refinement has to bring in. \
         Found {} reversed faces — if this is refinement or an improvement, \
         see the twin above",
        m.upper_reversed
    );
    // Which row `top_row` read. A near-wall row carries less than mid-channel;
    // an interior row would not, and the opposite wall's row would have failed
    // the assertion above with 35 reversed faces. Together those two pin the
    // row without needing a bubble to be present.
    assert!(
        m.outlet_near_upper_wall < m.outlet_mid_channel,
        "the upper-wall row must be the wall-adjacent one — its outlet value \
         {:.6} has to be below the mid-channel {:.6}, or the row above reports \
         'no bubble' about the wrong part of the channel",
        m.outlet_near_upper_wall,
        m.outlet_mid_channel
    );
    // Not vacuous: the lower wall does separate, and the bubble starts at the
    // step rather than somewhere down the channel.
    assert!(
        m.lower_reversed >= 8 && m.lower_first_face < 0.0,
        "the lower wall must carry a real separation starting at the step: \
         {} reversed faces, first face {:+.3e}",
        m.lower_reversed,
        m.lower_first_face
    );
    // And the run is a converged solve, not a field that stopped being
    // projected.
    assert!(
        m.worst_flux_imbalance < 1e-3,
        "the run must conserve mass across every cross-section, worst \
         imbalance {:.3e}",
        m.worst_flux_imbalance
    );
}
