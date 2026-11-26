//! Thick chain: a planar chain of links, each of a fixed area, moving between walls given as
//! a signed distance field. A stretched link is thinner, a link held between walls is longer:
//! a soft body that cannot tear, squeezed through a passage narrower than itself.
//!
//! # MODEL
//!
//! - **Equations.** Particles `x_i` (inverse mass `w_i`) joined by links `k = (i, i + 1)` of
//!   rest length `l0_k` and area `A_k = 2 r0 l0_k` (a strip of half-width `r0` at rest). A
//!   link of length `l` has the radius `r = max(A / (2 l), r_min)` ([`ThickChain::radius`]).
//!   A substep `h` (position based dynamics with compliance, Macklin et al. 2016, "XPBD"):
//!   `v ← v max(0, 1 − k h) + h (g + a_i)`, `x ← x + h v`; then per iteration, for each link
//!   the stretch constraint `C` with compliance `α`, its multiplier accumulated over the
//!   substep's iterations, `Δλ = (−C − (α / h²) λ) / ((w_i + w_j) |∇C|² + α / h²)`,
//!   `Δx = w ∇C Δλ` (so the tension `T = C |∇C| / α` does not depend on the number of
//!   iterations). With a retardation time `τ` (Kelvin-Voigt: a viscous stress `τ` times the
//!   rate of the elastic one) the same solve carries `γ = τ / h`:
//!   `Δλ = (−C − (α / h²) λ − γ ∇C·(x − x_start)) / ((1 + γ)(w_i + w_j) |∇C|² + α / h²)`,
//!   so a mode of frequency `ω` under the drag `k` goes by
//!   `e₊ (1 + ω² h² + τ ω² h) = e (1 + τ ω² h) + (1 − k h)(e − e₋)` a substep: it loses
//!   `k + τ ω²` of its rate, the fast modes first. The law of the stretch ([`ThickChainStretch`]) is either linear, `C = l − l0`
//!   with `α` given, `T = (l − l0) / α`; or the incompressible neo-Hookean solid in plane
//!   strain: with `λ = l / l0` its energy per rest area is `W = (μ / 2)(λ² + λ⁻² − 2) =
//!   (μ / 2)(λ − 1/λ)²` (`μ` the shear modulus; the width follows as `1/λ`), so `C = λ − 1/λ`
//!   and `α = 1 / (μ A)` exactly, `T = μ w0 (λ − λ⁻³)` with `w0 = 2 r0` the rest width
//!   (`4 μ w0 (λ − 1)` at small stretch, the plane strain modulus; without bound as `λ → 0`);
//!   and for each sample `p = (1 − t) x_i + t x_j` of each link (samples at most `r_min / 2`
//!   apart, the ends included) the wall constraint `C = d(p) − r ≥ 0`, `d` the distance to
//!   the wall: a disc of the link's radius fits at every sample. It is one-sided and rigid,
//!   `Δλ = −C / (w_i |∇_i C|² + w_j |∇_j C|²)` when `C < 0`, with the whole gradient:
//!   `∇_i C = (1 − t) G − (r / l) u`, `∇_j C = t G + (r / l) u`, `u` the link's direction, `G`
//!   the wall's slope as the central difference of `d` over `±r_min / 8` (the normal at a
//!   single wall, zero midway between two parallel walls), the `r / l` term absent at
//!   `r = r_min`. So a sample is moved off the wall and its link made longer and thinner in
//!   the proportion of the gradients; between two walls only the second is left, and the link
//!   settles at `l = A / (2 c)` in a channel of half-width `c`. The walls press on the link
//!   with the sum of the multipliers: in that channel `T l / c` for both walls together,
//!   `T` the link's tension ([`ThickChain::wall_force`]). Then `v ← (x − x_start) / h`, and friction (Coulomb, per link): the
//!   velocity of the link's centre of mass across the walls' mean slope `Σ Δλ G / Σ Δλ` drops
//!   by at most `μ Σ Δλ / (h (m_i + m_j))`, both ends alike; the lengthening itself meets no
//!   friction.
//!   The conforming rule ([`ThickChainFit::Conforming`], neo-Hookean links) lets a link take
//!   the shape of the space it has. Its samples' shares `ω_j` (trapezoid weights that add up
//!   to 1) are each as long as the wall there makes them, `μ_j = r_rest / d_j` (a planar
//!   share keeps its area, width times length; not the `λ^(−1/2)` of a rod in space), or as
//!   the link's tension does, `λ*`, whichever is more: `l / l0 = Σ ω_j max(λ*, μ_j)`. The
//!   link's energy is `A Σ ω_j W(max(λ*, μ_j))` and its tension `T(λ*)`: one constraint
//!   `C = √(Σ ω_j (m_j − 1/m_j)²)` (`m_j` the larger of `λ*` and `μ_j`) with the compliance
//!   `1 / (μ A)` and one multiplier, whichever shares are held. Its slope moves the ends
//!   along the link with the tension and each held share's sample along the wall's slope
//!   with what the share carries above the tension, `∝ (m̃_j s_j − m̃* s*) r_rest / d_j²`
//!   (`m̃ = m − 1/m`, `s = 1 + m⁻²`): the gel's own press on the walls, and what throws a
//!   squeezed body out of an open end. A link cannot be shorter than `l0 Σ ω_j μ_j`: the
//!   hard wall (its radius at most the harmonic mean of the distances). At that length `λ*`
//!   is the least `μ_j` (its limit from above), so the tension has no jump there: the link
//!   pulls, the hard wall answers, and that push is the walls' force, `T l / c` in a
//!   channel as under the disc rule. The samples of a link are laid out once, for the
//!   longest it may be (`l0 reach`, `ceil(2 l0 reach / r_min) + 1` spans): their number does
//!   not follow the link's length, so the energy has no jump as the link stretches; a link
//!   beyond `l0 reach` is refused. A pass takes every link's energy step, then the hard
//!   walls of all links at once, then keeps each middle line `r_min` off the walls. The hard
//!   walls are solved together because links side by side share a particle: a push on one
//!   changes its neighbours' lengths. With each condition made linear where the links are,
//!   the pushes `p` satisfy `p ≥ 0`, no link left short, and a push only on a link exactly
//!   at its walls' length; the links on their walls form runs, each run one tridiagonal
//!   system solved directly, a link whose push would pull taken off and a link the others
//!   would leave short put on, at most 8 rounds in the links' order. A pass whose rounds end
//!   unsettled is counted ([`ThickChain::walls_unsettled`]). Solved one link after another
//!   instead, a run of held links rests short of its walls' length by a part that goes with
//!   the substep squared (measured: up to 4.6e-4 of `l0` at 1/960 s, a quarter of that at
//!   half the substep), and the stated energy is then not the state's.
//!   What is left under the conforming rule is a term of the time step: in a pass every
//!   link takes its energy step in the links' order before the walls are solved; a link's
//!   own step shortens it and its two neighbours' steps lengthen it back, equally to first
//!   order in `(ω h)²`, and beside an end link (half the mass at its end, and first in the
//!   order) a second-order part of one sign remains, so that link rests a little longer
//!   than its walls ask and the walls' force is short by a part of `(ω h)²` (measured, six
//!   links of 10 mm held from 100 to 54 mm wide: the excess length 8.6e-6, 3.0e-7 and 5.4e-9
//!   of the length and the force 0.9919, 0.9978 and 0.9994 of `T l / c` at 8, 16 and 32
//!   substeps of 1/60 s; more passes do not change it).
//!   Bending ([`ThickChain::bending`], [`ThickChainBend::Section`]; a chain is made
//!   without it, and then a joint turns freely). A link stretched to `λ` at the middle of
//!   its width and bent to the curvature `κ` is Rivlin's flexure of an incompressible
//!   neo-Hookean block in plane strain (1949: a fibre at `y` from the middle has the stretch
//!   `λ (1 − κ y)` and the length `l (1 − κ y)`): its energy is
//!   `(μ A / 2) [λ² (1 + a²) + λ⁻² artanh(a) / a − 2]`, `a = κ r`, `r = r0 / λ`. At `a = 0`
//!   that is the stretch energy above; the rest, `(μ A / 2) κ² r0² [1 + λ⁻⁴ g(a)]` with
//!   `g(a) = (artanh(a) / a − 1) / a²`, is the bending: `(B / 2) κ² l` at rest and for small
//!   `κ`, `B = (8/3) μ r0³` (the plane strain modulus `4 μ` times `(2 r0)³ / 12`), and
//!   without bound as `a → 1`, as `½ ln(2 / (1 − a))`: a link cannot be bent to the radius
//!   of its half-width, where its inner side would have no length. The joint at particle
//!   `i` has the curvature `κ = 2 tan(θ / 2) / l̄` (`θ` the angle its two links turn by,
//!   `tan(θ / 2)` from their cross and dot products; `l̄` the mean of their lengths), and
//!   each of the two links gives the joint half of its bending energy at that curvature,
//!   with its own `λ` and `r`: the joint's energy is `(μ / 2) C²`, `C = κ S`,
//!   `S² = Σ (A / 2) r0² (1 + λ⁻⁴ g(κ r0 / λ))` over the two links, a constraint of
//!   compliance `1 / μ` solved as a link's stretch is (the Kelvin-Voigt term on it too),
//!   after the stretches in each pass. `C` has the sign of the turn and no corner at a
//!   straight joint. Its slope has every term: the angle's, and the lengths' through `l̄`,
//!   `λ` and `a` (a bent link is pulled longer and thinner by its bending). `g` is summed
//!   as `Σ a²ⁿ / (2n + 3)` to 18 terms with `a² < 1/16` (what is left out is below `2⁻⁸¹`;
//!   there the closed form cancels) and taken from [`Fix128::ln`] from there on; the two
//!   are within `2⁻⁵⁰` of each other where one hands over. A joint with `a ≥ 1 − 2⁻²⁰` for
//!   either of its links, or folded flat, is [`ThickChainError::Folded`]: refused when the
//!   law is set, and in a step when the chain is there at the start of a pass, after the
//!   substep has been halved as often as the setting allows
//!   ([`ThickChain::substep_retries`] counts the substeps taken again in halves).
//!   With the law the constraints of a substep are solved another way: all the particles
//!   together, as steps down the substep's potential. With `x̃` the particles after the
//!   velocity step (and on by what the walls push them in each pass) the substep is the
//!   minimum of `Ψ(x) = Σ m |x − x̃|² / 2 + Σ [C² + γ (C − C_start)²] h² / (2 α)` over the
//!   links' stretches and the joints (`C_start` a constraint's value where the substep
//!   started): its stationary point is the backward Euler step of the energy itself with
//!   the Kelvin-Voigt stress `γ (C − C_start) / α`, and at rest the energy's stationary
//!   point whatever the substep. (The Kelvin-Voigt term of a link without the law is
//!   `γ ∇C · (x − x_start)`, the same to the first order of the substep's move; in this
//!   form the step has a potential, and a part of a step can be judged by it.) A pass
//!   takes the step `d` of `[M / h² + (1 + γ) Gᵀ diag(1 / α) G] d = −∇Ψ / h²` (`G` the
//!   constraints' slopes, `M` the masses): the second slopes of `Ψ` without those of the
//!   constraints (Gauss-Newton), positive definite whatever the state. In the coordinates'
//!   order along the chain the system has no entry further than 5 from its diagonal (a
//!   joint ties a particle to the next but one); it is factored as `L D Lᵀ` in that order,
//!   without a square root, the same operations whatever the state. Then the pass moves
//!   the chain by the first of `d`, `d / 2`, `d / 4`, `d / 8` and `d / 16` that lowers `Ψ`
//!   by a ten-thousandth of what its slope along `d` promises for that part (Armijo's
//!   rule; all five are worked out whichever is taken, a part that would bend a joint past
//!   the radius of a link's half-width lowers nothing, and two values of `Ψ / h²` within
//!   `2⁻⁵⁶` are not told apart). So no pass raises `Ψ`: [`ThickChain::bend_halved`] counts
//!   the halvings taken, [`ThickChain::bend_stalled`] the passes in which no part lowered
//!   it and the chain stayed. Without the constraints' second slopes the passes close on
//!   the minimum at a linear rate that is slow where the law's shape is steep (measured,
//!   an arc of twelve links at `a = 0.86` let go from rest, the distance from the minimum
//!   of one substep after 8 and after 16 passes: 7.4e-6 and 1.3e-8 m with `μ` 6667 Pa and
//!   a substep of 1/240 s, 4.6e-8 and 3.4e-11 m at 1/3840 s; 1.5e-6 and 5.4e-7 m with `μ`
//!   1000 Pa at 1/1920 s, the slowest found; with `τ = 0.05 s` 1.5e-11 m or less after 8:
//!   `tests/analytic_thick_chain_bend_step.rs`, against the minimum found apart in `f64`).
//!   The step of the multipliers of all the constraints at once that this replaces (the
//!   matrix `(1 + γ) G W Gᵀ + diag(α / h²)`, each multiplier put along its slope of now
//!   before the step) is the same matrix seen from the constraints, taken whole: near the
//!   wall of the law it overshot, and `Ψ` rose from pass to pass with the step still
//!   pointing down it (the same arc at 1/3840 s: `Ψ` fell for 5 passes and rose in the 11
//!   after, the chain 1.9e-5 m from the minimum at 16), and putting the multipliers along
//!   the slopes of now carried a joint past the wall at larger substeps, which were then
//!   halved unseen.
//!   Two things this is for. Stepped constraint by constraint along the slope where the
//!   chain was when each step was taken, the sum of the steps is not the last slope times
//!   the multiplier, and the rest state is off by a share that goes with `(ω h)²` and that
//!   more passes do not remove (measured, a couple bending ten links of 10 mm, 20 mm in
//!   half-width, to `a = 0.5`: the joints turned 5.1, 1.5 and 0.36 % too far at 16, 32 and
//!   64 substeps of 1/60 s, the same at 16 and 64 passes; the energy of a chain let go grew
//!   in 7 of 180 frames at 32 substeps). And solved joint after joint, even with the
//!   multipliers put along the slopes of now, the passes of a substep do not reach its
//!   solution when `γ` is large: with `τ = 0.05 s` (`γ = 96` at 32 substeps of 1/60 s) the
//!   first bending swing of twelve links was left, a period on, 5.4 % above what the
//!   recurrence of the Kelvin-Voigt mode leaves at 8 passes, 16 % at 16 substeps, and
//!   0.003 % at 64 passes; solved together it is within 0.003 % at 1, 8, 16 and 64 passes
//!   alike (`tests/analytic_thick_chain_bend_statics.rs`; the same to six digits with the
//!   multipliers' step and with the particles'). The links' stretches alone did
//!   not show it (a swing along a chain of twenty links: the same to four digits at 8, 16,
//!   32 and 64 passes).
//!   The law is refused with the conforming rule, which solves each link with its walls
//!   inside the link's own step.
//!   A chain without the law is solved as before, bit for bit:
//!   its links' slopes turn too little in a substep for the difference to show, and its
//!   bits are those of the chains made before the law. The walls are solved as before in
//!   either case (their projections are not kept).
//! - **Units.** SI: m, s, kg, N; planar, so masses, forces and the compliance are per metre
//!   of depth (kg/m, N/m, `α` in m²/N); `k` in 1/s.
//! - **Conservation.** A link's area is `2 r l` by the definition of `r`, exactly, down to
//!   `r_min` (below it the link is drawn wider than its area). Linear momentum is conserved
//!   by the stretch constraint; the walls and their friction take it. Energy is not
//!   conserved: the constraints are solved implicitly (backward Euler), which damps a free
//!   swing by `1 / √(1 + (ω h)²)` a substep (`ω² = (w_i + w_j) / α`, linear law), the drag `k`
//!   and the friction dissipate, and a wall projection takes the approach speed (the walls do not bounce a
//!   particle; a body thrown at a wall is flattened and springs back by its own stretch).
//! - **Range.** Planar. `α > 0`, `μ > 0` (a rigid link cannot be squeezed). The bending law
//!   needs neo-Hookean links and every joint inside `a < 1 − 2⁻²⁰`; the halvings of a
//!   substep count the joints' `(ω h)²` as the links'. A substep in
//!   which a particle would travel more than `r_min / 2`, or a link has `(ω h)² > 1/2`
//!   (`(ω h)² = (w_i + w_j) |∇C|² h² / α` at its length then), is taken in halves, again as
//!   needed up to `splits` times, and refused beyond that. Between walls the stretch and the
//!   walls are solved in turn, and what a pass leaves unsolved grows with `(ω h)²`: with
//!   `splits = 0` the caller keeps it small (measured, with the largest `(ω h)²` of the
//!   chain's links, pushed 5 and 10 % below the least push that can feed it into a narrower
//!   channel: not fed at 0.50 and 0.69, a chain of 80 links creeping in at 1.0 and fed at
//!   1.44 and above, a chain of 40 links not fed up to 1.44). Entering a
//!   passage: a chain of rest width `w0` is fed into a frictionless channel of half-width
//!   `c < r0` only by a push that pays for the energy of the squeeze. Pushed on its tail by
//!   `F` (neo-Hookean law), the part outside is shortened to `λ_F` with `T(λ_F) = −F`, and
//!   the chain goes in when `F λ_F > (W(λc) − W(λ_F)) w0`, `λc = r0 / c`: with `q = λ_F²`,
//!   `q² + (λc² + λc⁻²) q − 3 = 0` and the least push is `μ w0 (λ_F⁻³ − λ_F)` (`0.609 μ w0`
//!   at `λc = 5/3`). Under a body force `a` on a chain of density `ρ` and length `L` the same
//!   balance reads about `a ρ L > W(λc)`: a passage narrower than the body sets a least push
//!   in proportion to the shear modulus. A particle may travel at
//!   most `r_min / 2` in a substep before the constraints
//!   ([`ThickChainError::TooFast`]), which with the sample spacing keeps a wall from passing
//!   between samples; a passage narrower than `2 r_min` is closed to the chain. A link
//!   stretched beyond 2048 samples (or, under the conforming rule, beyond `l0 reach`), a link
//!   of no length, a field value that is not a number
//!   and an invalid argument are refused; a refused step changes nothing. The distance field
//!   must not overstate the distance (Lipschitz constant at most 1); it is read in `f32`.
//!   The chain's own arithmetic is `Fix128`; the field is the caller's and is read in `f32`,
//!   so a run is the same bits on every machine when the field is made of the basic IEEE
//!   operations (`+ − × ÷ √`, comparisons), and not when it calls a platform's mathematical
//!   library. Across machines this is stated, not yet tested.
//! - **Approximations.** A link has one radius along its length (the outline steps between
//!   links). Without a bending law the body has no bending stiffness; with it the
//!   curvature is taken as one value over a joint's share of the chain (exact for a chain
//!   bent evenly; a clamped chain under its weight sags above the beam's
//!   `q L⁴ / (8 B)` by about `2.1 / N` for `N` links), and what keeps links far apart along
//!   the chain from overlapping is not part of it. The body has no shear and no rotary
//!   inertia; its thickness follows its
//!   length at once (no inertia across the link). Between samples the outline may be in a
//!   wall by at most `r − √(r² − s² / 4)`, `s ≤ r_min / 2` the spacing. A link's one radius is
//!   that of its narrowest sample, so where a passage narrows the squeeze is paid a link at a
//!   time: the mean push is the closed form of the Range, the largest lump is above it,
//!   and a link that clears its lump jumps on and loses the speed to the drag (measured at
//!   a resolved step, a mouth from 30 to 12 mm half-width over 0.1 m for a 20 mm body, the
//!   tail moved slowly: the largest push 3.3, 1.7 and 1.2 times the closed form with links
//!   of 20, 10 and 5 mm, the push's work 1.57, 1.22 and 1.04 times the energy taken, and
//!   the energy taken over the way equal to the closed form to four digits). The wall's slope is
//!   taken over `±r_min / 8`, so the chain is centred between two walls to that width. The
//!   wall projections are not accumulated: what the last pass of an iteration leaves is
//!   carried to the next substep. The stretch and the walls are solved in turn, so at rest
//!   between walls a link is left a little short of the walls' length and the walls carry
//!   a little less than `T l / c`, by a share that goes with `(ω h)²`. It is a property
//!   of the state the turn-taking settles in, not of unfinished passes: more passes do not
//!   remove it (measured, links of 10 mm held from 100 to 54 mm wide at `(ω h)² = 0.13`: the
//!   walls' force over `T l / c` for one link 0.764, 0.873, 0.888, 0.889 and 0.889 with 1, 2,
//!   4, 8 and 16 passes, flat from 8; for six links 0.94 to 0.97 with 4 passes and 0.94 to
//!   0.98 with 16; the links up to 0.46 % short with 4 passes). The squeeze friction is low
//!   by the same share.
//! - **Validation.** `tests/analytic_thick_chain.rs::a_link_of_fixed_area_is_thinner_when_longer::a_hanging_link_stretches_by_its_compliance_times_the_weight::a_free_link_swings_on_the_backward_euler_solution_whatever_the_iterations::a_chain_in_a_narrow_channel_is_longer_by_the_ratio_of_the_widths::a_chain_sliding_on_a_floor_stops_after_the_coulomb_distance::a_chain_held_thin_by_a_channel_stops_after_the_squeeze_friction_distance::a_chain_driven_through_a_wider_pipe_stays_out_of_the_walls::a_chain_driven_through_a_narrower_pipe_is_squeezed_and_comes_back::a_run_repeats_bit_for_bit::a_refused_chain_or_step_changes_nothing` `tests/analytic_thick_chain_neo_hookean.rs::a_hanging_link_settles_where_the_neo_hookean_tension_is_its_weight::a_small_stretch_meets_the_plane_strain_modulus::a_standing_link_is_shortened_to_where_the_tension_is_minus_its_weight::a_neo_hookean_chain_held_thin_stops_after_the_squeeze_friction_distance::the_stretch_law_is_part_of_the_state::held_by_a_channel_a_chain_comes_to_rest_and_the_walls_carry_most_of_its_tension` `tests/analytic_thick_chain_anchor.rs::a_fast_link_is_refused_without_split_substeps_and_flies_straight_with_them::a_run_that_needs_no_split_is_the_same_bits_with_and_without::a_link_thrown_at_a_floor_stops_on_it::a_link_too_stiff_for_its_substep_is_given_the_halvings_that_resolve_it::a_free_link_with_inner_friction_swings_on_the_recurrence_of_its_implicit_step::the_fixed_tail_keeps_its_bits_through_a_pull_between_walls::every_free_link_of_a_pulled_chain_is_where_its_tension_is_the_pull::the_reach_in_a_channel_under_less_than_the_walls_hold_is_the_walls_hold::the_reach_in_a_channel_under_more_than_the_walls_hold_is_the_pulls_stretch::let_go_the_free_part_swings_back_on_its_first_modes_recurrence::after_the_pull_ends_the_energy_never_grows` `tests/analytic_thick_chain_entry.rs::a_chain_of_short_links_is_not_fed_below_the_push_of_its_squeeze_energy_and_is_fed_above_it::moved_slowly_the_push_works_no_less_than_the_links_take::fed_every_link_is_as_long_as_the_channel_makes_it::a_conforming_chain_pays_the_squeeze_as_it_goes_not_a_link_at_a_time::moved_slowly_a_conforming_chains_push_works_what_its_shares_take::at_rest_no_link_is_shorter_than_its_walls_make_it` `tests/analytic_thick_chain_shares.rs::a_chain_in_a_channel_is_as_long_under_either_rule::a_conforming_chain_held_thin_stops_after_the_squeeze_friction_distance::the_walls_of_a_straight_channel_do_not_change_the_momentum_along_it::held_by_a_channel_a_conforming_chain_comes_to_rest_and_the_walls_carry_its_tension::the_free_stretch_is_the_shares_equation_and_has_no_jump_across_a_mouth::the_fit_rule_is_part_of_the_state_and_needs_neo_hookean_links::a_links_samples_are_laid_out_once_for_its_reach_and_counted_as_stated::the_walls_of_these_tests_are_fields_the_chain_can_be_given::the_reach_is_part_of_the_state::walls_that_take_more_rounds_than_a_pass_has_are_counted::a_link_partly_held_presses_on_its_walls_with_what_its_held_shares_carry_above_its_tension` `tests/analytic_thick_chain_pocket.rs::let_go_at_the_open_end_a_squeezed_chain_leaves_with_no_more_than_the_squeezes_energy::between_links_held_by_a_channel_the_links_in_a_pocket_are_off_their_walls::the_walls_of_the_pocket_tests_are_fields_the_chain_can_be_given` `tests/analytic_thick_chain_bend.rs::a_chain_is_made_without_the_law_and_its_bytes_are_those_before_the_law::with_the_law_the_bytes_say_so::given_the_law_after_steps_a_chain_is_the_chain_made_in_that_state_with_the_law::a_straight_chain_held_by_two_particles_does_not_move_by_a_bit::off_its_line_either_way_a_straight_chain_runs_mirrored_bit_for_bit::what_cannot_be_bent_is_refused_when_the_law_is_set::a_step_that_meets_a_fold_is_refused_and_writes_nothing::a_chain_of_one_link_has_no_joint_and_swings_as_without_the_law::the_law_is_refused_with_the_conforming_fit` `tests/analytic_thick_chain_bend_statics.rs::a_couple_bends_the_inner_links_to_the_arc_of_least_energy_gently::a_couple_bends_the_inner_links_to_the_arc_of_least_energy_halfway_to_the_fold::a_couple_bends_the_inner_links_to_the_arc_of_least_energy_near_the_fold::the_arc_under_a_couple_is_the_same_whatever_the_substep_and_the_passes::a_clamped_chain_sags_by_the_discrete_statics_and_nears_the_beam_as_its_links_shorten::pushed_along_below_the_buckling_load_a_clamped_chain_leans_by_the_beam_columns_share::pushed_along_above_the_buckling_load_a_clamped_chain_leaves_its_line::at_rest_under_a_dead_load_the_load_on_every_particle_is_the_slope_of_the_energy::let_go_bent_and_stretched_the_energy_never_grows::let_go_a_little_off_its_line_the_head_swings_with_the_first_modes_period::with_a_retardation_time_the_bending_swing_dies_at_the_kelvin_voigt_rate_whatever_the_passes::with_a_retardation_time_the_stretch_swing_dies_at_the_kelvin_voigt_rate_whatever_the_passes::let_down_onto_a_floor_from_a_fixed_tail_no_link_turns_back::pushed_with_twenty_times_the_buckling_load_no_joint_reaches_the_radius_of_its_half_width::a_joint_whose_thicker_link_alone_is_past_the_radius_of_its_half_width_is_refused` `tests/analytic_thick_chain_bend_step.rs::the_reference_minimum_is_a_minimum_and_the_scenes_are_near_the_wall::near_the_wall_no_pass_of_a_substep_raises_its_potential::near_the_wall_the_passes_of_a_substep_close_on_its_minimum`
//! - **Tiers.** L0 only.

use crate::math::Fix128;
use crate::sdf_collider::SdfQuery;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A point or a vector of the plane.
pub type P2 = [Fix128; 2];

const ZERO2: P2 = [Fix128::ZERO, Fix128::ZERO];

/// The most samples a link may need (its length over half the least radius).
const MAX_SAMPLES: i64 = 2048;
/// The most times the walls of a run of links are solved in a pass (a link whose wall would
/// pull is taken off it each time).
const WALL_ROUNDS: usize = 8;

fn add(a: P2, b: P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(a: P2, s: Fix128) -> P2 {
    [a[0] * s, a[1] * s]
}

fn dot(a: P2, b: P2) -> Fix128 {
    a[0] * b[0] + a[1] * b[1]
}

/// `(−v_y, v_x)`: `v` turned a quarter to the left.
fn perp(v: P2) -> P2 {
    [-v[1], v[0]]
}

/// Below this `a²` the bending law's shape is summed as a series, above it taken from the
/// logarithm: 1/16.
fn bend_switch() -> Fix128 {
    Fix128::ONE.shr_bits(4)
}

/// The terms of the series of [`bend_shape`]: with `a² < 1/16` what is left out is below
/// `16⁻¹⁹ / 41 < 2⁻⁸¹`.
const BEND_TERMS: i64 = 18;

/// A joint is folded from `a = 1 − 2⁻²⁰` on: the wall of the law is at `a = 1`, and the
/// shape's slope `1 / (1 − a²)` stays below `2²⁰` on this side of it.
fn bend_limit() -> Fix128 {
    Fix128::ONE - Fix128::ONE.shr_bits(20)
}

/// The shape of the bending law and its slope: `g(a) = (artanh(a) / a − 1) / a²` and
/// `g′(a) = (1 / (1 − a²) − 3 g) / a`, for `|a| < 1` (`g(0) = 1/3`, `g′(0) = 0`): by
/// [`bend_shape_series`] with `a² < 1/16`, by [`bend_shape_log`] from there on.
fn bend_shape(a: Fix128) -> (Fix128, Fix128) {
    if a * a < bend_switch() {
        bend_shape_series(a)
    } else {
        bend_shape_log(a)
    }
}

/// [`bend_shape`] as the series `g = Σ a²ⁿ / (2n + 3)`, `g′ = Σ 2n a²ⁿ⁻¹ / (2n + 3)` to
/// [`BEND_TERMS`]: for small `a`, where the closed form cancels (at `a = 10⁻⁵` half an ulp
/// of the logarithm is 3·10⁻⁴ of `g − 1/3`).
fn bend_shape_series(a: Fix128) -> (Fix128, Fix128) {
    let a2 = a * a;
    // Horner from the last term: g = 1/3 + a²(1/5 + a²(1/7 + …)); g′ likewise in a² with
    // the weights 2n / (2n + 3), times a
    let (mut g, mut slope) = (Fix128::ZERO, Fix128::ZERO);
    for n in (0..=BEND_TERMS).rev() {
        g = g * a2 + Fix128::from_ratio(1, 2 * n + 3);
        if n > 0 {
            slope = slope * a2 + Fix128::from_ratio(2 * n, 2 * n + 3);
        }
    }
    (g, slope * a)
}

/// [`bend_shape`] in closed form, `artanh(a) = ln((1 + a) / (1 − a)) / 2` with
/// [`Fix128::ln`]: for `a` away from 0 (and below 1).
fn bend_shape_log(a: Fix128) -> (Fix128, Fix128) {
    let a2 = a * a;
    let size = a.abs();
    let artanh = ((Fix128::ONE + size) / (Fix128::ONE - size)).ln().half();
    let g = (artanh / size - Fix128::ONE) / a2;
    let slope = (Fix128::ONE / (Fix128::ONE - a2) - (g + g + g)) / a;
    (g, slope)
}

/// One joint's bending constraint where the chain is: its value, its slope on the three
/// particles (the one before, the joint's, the one after).
struct BendTerm {
    value: Fix128,
    slope: [P2; 3],
}

/// A constraint of a substep's solve with a bending law: a link's stretch (two particles
/// from `first`) or a joint (three), its value and slope where the chain is, and its
/// stiffness `1 / α` (one over its compliance).
#[derive(Clone, Copy)]
struct Row {
    first: usize,
    count: usize,
    slope: [P2; 3],
    value: Fix128,
    stiff: Fix128,
    live: bool,
}

/// How far from the diagonal the matrix of a substep's step reaches, in the coordinates'
/// order along the chain (two to a particle): a joint ties a particle to the next but one,
/// so a coordinate to the fifth after it, nothing further.
const BAND: usize = 5;

/// The parts of a step that are tried, each half the one before, the whole step first.
const STEP_PARTS: u32 = 5;

/// What a part of a step must lower the merit by, as a share of what the merit's slope
/// along the step promises for that part.
fn step_share() -> Fix128 {
    Fix128::from_ratio(1, 10_000)
}

/// The merit's sum of a few dozen products is settled to this (2⁻⁵⁶): two merits closer
/// than it are not told apart.
const MERIT_GRAIN: Fix128 = Fix128 { hi: 0, lo: 1 << 8 };

/// The solution of a symmetric system given in band (`band[i][b]` the entry of rows `i`
/// and `i − b`, `b < W`), factored as `L D Lᵀ` in the rows' order: no square root, the
/// same operations in the same order whatever the entries. None when a pivot is not
/// positive (the system is not positive definite to the arithmetic's resolution).
fn solve_band<const W: usize>(band: &[[Fix128; W]], right: &[Fix128]) -> Option<Vec<Fix128>> {
    let n = band.len();
    let reach = W - 1;
    let mut low = vec![[Fix128::ZERO; W]; n];
    let mut pivot = vec![Fix128::ZERO; n];
    for i in 0..n {
        for back in (1..=reach.min(i)).rev() {
            let j = i - back;
            let mut entry = band[i][back];
            for further in back + 1..=reach.min(i) {
                let k = i - further;
                if j - k <= reach {
                    entry = entry - low[i][further] * pivot[k] * low[j][j - k];
                }
            }
            low[i][back] = entry / pivot[j];
        }
        let mut entry = band[i][0];
        for back in 1..=reach.min(i) {
            entry = entry - low[i][back] * low[i][back] * pivot[i - back];
        }
        if entry <= Fix128::ZERO {
            return None;
        }
        pivot[i] = entry;
    }
    let mut out = right.to_vec();
    for i in 0..n {
        for back in 1..=reach.min(i) {
            out[i] = out[i] - low[i][back] * out[i - back];
        }
    }
    for i in 0..n {
        out[i] = out[i] / pivot[i];
    }
    for i in (0..n).rev() {
        for back in 1..=reach.min(n - 1 - i) {
            out[i] = out[i] - low[i + back][back] * out[i + back];
        }
    }
    Some(out)
}

/// The settings of a [`ThickChain`]; [`ThickChain::new`] checks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThickChainConfig {
    /// Substeps to a step, at least 1
    pub substeps: u32,
    /// Constraint passes to a substep, at least 1
    pub iterations: u32,
    /// Gravity, m/s²
    pub gravity: P2,
    /// The law of a link's stretch
    pub stretch: ThickChainStretch,
    /// Drag, 1/s: a velocity loses `drag · h` of itself a substep
    pub drag: Fix128,
    /// Coulomb friction coefficient against the walls
    pub friction: Fix128,
    /// The least radius of a link, m, above 0: the chain's middle line keeps this far from a
    /// wall
    pub min_radius: Fix128,
    /// How often a substep may be halved when a particle would travel too far in it (0:
    /// never, the step is refused instead), at most 16
    pub splits: u32,
    /// The retardation time of the links' material, s (viscosity over stiffness, the
    /// Kelvin-Voigt solid): the stretch is resisted in proportion to its rate. 0: none
    pub relaxation: Fix128,
    /// How a link meets the walls
    pub fit: ThickChainFit,
    /// The largest stretch (length over rest length) a conforming link is sampled for: its
    /// samples are laid out once for that length (`ceil(2 l0 reach / r_min)` spans), so their
    /// number does not follow the link's length and the link's energy has no jumps in it. A
    /// link longer than `l0 reach` is refused as stretched. At least 1; not read by the disc
    /// rule
    pub reach: Fix128,
}

/// How a link answers a stretch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThickChainStretch {
    /// Tension in proportion to the extension: `(l − l0) / compliance`
    Linear {
        /// m²/N (extension per tension, per metre of depth), above 0
        compliance: Fix128,
    },
    /// The incompressible neo-Hookean solid in plane strain: tension `shear · w0 (λ − λ⁻³)`,
    /// `λ = l / l0`, `w0` the link's rest width
    NeoHookean {
        /// The shear modulus, Pa, above 0 (a third of Young's modulus)
        shear: Fix128,
    },
}

/// How a link meets the walls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThickChainFit {
    /// A body of one width along a link: a disc of the link's radius fits at every sample
    Disc,
    /// A body that takes the shape of the space it has: each sample's share of the link is as
    /// long as the wall there makes it or as the link's tension makes it, whichever is more
    /// (neo-Hookean links only)
    Conforming,
}

/// How a chain answers being bent ([`ThickChain::bending`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThickChainBend {
    /// No bending stiffness: a joint turns freely
    None,
    /// The strip's own section: at each joint the links' neo-Hookean energy over the width
    /// of a strip bent to the joint's curvature (neo-Hookean links only)
    Section,
}

/// Why a chain cannot be made or stepped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThickChainError {
    /// An argument out of its range (the one named)
    Invalid(&'static str),
    /// A link of no length
    Degenerate {
        /// The link index
        link: usize,
    },
    /// A link stretched beyond the samples a link may have
    Stretched {
        /// The link index
        link: usize,
    },
    /// A particle would travel more than half the least radius in a substep
    TooFast {
        /// The particle index
        particle: usize,
    },
    /// A link is too stiff for the substep at its most halvings
    TooStiff {
        /// The link index
        link: usize,
    },
    /// The distance field gave a value that is not a number at a sample of the link
    Field {
        /// The link index
        link: usize,
    },
    /// A joint bent to a radius of its half-width or less (the joint between links
    /// `joint − 1` and `joint`): the inner side of the strip would have no length
    Folded {
        /// The joint: the index of the particle the two links share
        joint: usize,
    },
}

/// The shares of one link under the conforming rule (see `ThickChain::shares_of`).
/// The hard wall of one link, made linear where the link is.
struct WallTerm {
    lack: Fix128,
    u: P2,
    off0: P2,
    off1: P2,
    by_walls: Fix128,
    leaning: P2,
}

struct Shares {
    weight: Vec<Fix128>,
    place: Vec<Fix128>,
    room: Vec<Fix128>,
    short: Vec<Fix128>,
    asked: Vec<Fix128>,
    need: Fix128,
}

/// The pushes of the walls on a row of links, each condition made linear: link `k` is short
/// of its walls' length by `lack[k]` (not short when that is not above 0), its own push
/// lengthens it by `diag[k]` per unit, and the push of the link before or after changes its
/// length by `next[k − 1]` or `next[k]` per unit through the particle they share. Finds
/// pushes `p` with `p ≥ 0`, no link left short (`diag p + next p' − lack ≥ 0`), and a push
/// only on a link exactly at its walls' length: the links that are short start on their
/// walls; each run of links on their walls is one tridiagonal system, solved directly; a
/// link whose push comes out pulling is taken off, a link the others' pushes would leave
/// short is put on, and the systems are solved again, at most `rounds` times, in the links'
/// order. `able[k]` false keeps link `k` off (no condition of its own). Returns the pushes,
/// which links ended on their walls, and whether the last round changed nothing.
///
/// Between straight walls a push on a link shortens its neighbours (`next < 0`), every short
/// link then gets a push and none comes out pulling; where the walls turn or narrow along a
/// link the walls' part of the slopes can make `next` positive, and a short link beside a
/// strongly pushed one can come out pulling: it is then held up by its neighbour, not by its
/// own walls.
fn settle_walls(
    diag: &[Fix128],
    next: &[Fix128],
    lack: &[Fix128],
    able: &[bool],
    rounds: usize,
) -> (Vec<Fix128>, Vec<bool>, bool) {
    let links = diag.len();
    let mut on: Vec<bool> = (0..links)
        .map(|k| able[k] && lack[k] > Fix128::ZERO)
        .collect();
    let mut push = vec![Fix128::ZERO; links];
    let mut carry = vec![Fix128::ZERO; links];
    let mut partial = vec![Fix128::ZERO; links];
    let mut settled = false;
    for _ in 0..rounds {
        push.fill(Fix128::ZERO);
        let mut dropped = false;
        let mut k = 0;
        while k < links {
            if !on[k] {
                k += 1;
                continue;
            }
            let first = k;
            // forward
            let mut pivot = diag[k];
            carry[k] = next[k] / pivot;
            partial[k] = lack[k] / pivot;
            k += 1;
            while k < links && on[k] {
                pivot = diag[k] - next[k - 1] * carry[k - 1];
                if pivot <= Fix128::ZERO {
                    // not a condition of its own beside its neighbour: off its wall
                    on[k] = false;
                    dropped = true;
                    break;
                }
                carry[k] = next[k] / pivot;
                partial[k] = (lack[k] - next[k - 1] * partial[k - 1]) / pivot;
                k += 1;
            }
            // back
            let mut i = k - 1;
            push[i] = partial[i];
            while i > first {
                i -= 1;
                push[i] = partial[i] - carry[i] * push[i + 1];
            }
        }
        for k in 0..links {
            if on[k] && push[k] < Fix128::ZERO {
                on[k] = false;
                dropped = true;
            }
        }
        // a link not on its wall that the others' pushes would leave shorter than its walls
        // allow (they move the particles it shares) comes onto it
        let mut joined = false;
        for k in 0..links {
            if on[k] || !able[k] {
                continue;
            }
            let mut after = -lack[k];
            if k > 0 && on[k - 1] {
                after = after + next[k - 1] * push[k - 1];
            }
            if k + 1 < links && on[k + 1] {
                after = after + next[k] * push[k + 1];
            }
            if after < Fix128::ZERO {
                on[k] = true;
                joined = true;
            }
        }
        if !dropped && !joined {
            settled = true;
            break;
        }
    }
    (push, on, settled)
}

/// Which shares the walls hold, the weight of the others, and their stretch `λ*`:
/// `ratio = Σ ω_j max(λ*, μ_j)`. When the link is no longer than the walls ask
/// (`ratio ≤ need`), `λ*` is the least `μ_j`, its limit as the link comes down to that
/// length: the link's tension has no jump there, and the shares at the least are the free
/// ones.
#[allow(
    clippy::needless_range_loop,
    reason = "a share's index reads three rows"
)]
fn held_shares(
    ratio: Fix128,
    need: Fix128,
    weight: &[Fix128],
    asked: &[Fix128],
) -> (Vec<bool>, Fix128, Fix128) {
    let count = weight.len();
    if ratio <= need {
        let mut least = asked[0];
        for &mu in asked {
            if mu < least {
                least = mu;
            }
        }
        let mut open = Fix128::ZERO;
        let mut is_held = vec![false; count];
        for j in 0..count {
            if asked[j] > least {
                is_held[j] = true;
            } else {
                open = open + weight[j];
            }
        }
        return (is_held, open, least);
    }
    let mut is_held = vec![false; count];
    loop {
        let (mut sum, mut open) = (Fix128::ZERO, Fix128::ZERO);
        for j in 0..count {
            if is_held[j] {
                sum = sum + weight[j] * asked[j];
            } else {
                open = open + weight[j];
            }
        }
        if open.is_zero() {
            return (is_held, Fix128::ZERO, Fix128::ZERO);
        }
        let stretch = (ratio - sum) / open;
        let mut more = false;
        for j in 0..count {
            if !is_held[j] && asked[j] > stretch {
                is_held[j] = true;
                more = true;
            }
        }
        if !more {
            return (is_held, open, stretch);
        }
    }
}

/// A planar chain of links of fixed area.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThickChain {
    config: ThickChainConfig,
    x: Vec<P2>,
    v: Vec<P2>,
    inv_mass: Vec<Fix128>,
    rest_length: Vec<Fix128>,
    area: Vec<Fix128>,
    /// the walls' press on each link in the last substep taken (read out, not state)
    wall_force: Vec<Fix128>,
    /// the passes whose hard walls were not settled within their rounds (read out, not state)
    walls_unsettled: u64,
    bend: ThickChainBend,
    /// the halvings of the steps taken with the bending law (read out, not state)
    bend_halved: u64,
    /// the passes with the bending law in which no part of the step was taken (read out,
    /// not state)
    bend_stalled: u64,
    /// the tries of a substep that were refused and taken again in halves (read out, not
    /// state)
    substep_retries: u64,
    /// the furthest a particle was moved along an axis by the last pass of the last substep
    /// taken (read out, not state)
    last_pass_move: Fix128,
}

impl ThickChain {
    const SNAPSHOT_MAGIC: [u8; 4] = *b"TCHN";
    const SNAPSHOT_VERSION: u32 = 1;
    /// the version of a chain with a bending law: the law's tag follows the settings
    const SNAPSHOT_VERSION_BEND: u32 = 2;

    /// A chain through `points` at rest: each link's rest length is the distance between its
    /// ends, its area `2 · radius · length`, its mass `density` (kg/m³) times its area, half
    /// on each end.
    ///
    /// # Errors
    ///
    /// [`ThickChainError::Invalid`] for a setting out of range, fewer than two points, a
    /// radius below the least radius or a density not above 0;
    /// [`ThickChainError::Degenerate`] for two points in one place.
    pub fn new(
        config: ThickChainConfig,
        points: &[P2],
        radius: Fix128,
        density: Fix128,
    ) -> Result<Self, ThickChainError> {
        let zero = Fix128::ZERO;
        if config.substeps == 0 {
            return Err(ThickChainError::Invalid("substeps"));
        }
        if config.iterations == 0 {
            return Err(ThickChainError::Invalid("iterations"));
        }
        match config.stretch {
            ThickChainStretch::Linear { compliance } if compliance <= zero => {
                return Err(ThickChainError::Invalid("compliance"));
            }
            ThickChainStretch::NeoHookean { shear } if shear <= zero => {
                return Err(ThickChainError::Invalid("shear"));
            }
            _ => {}
        }
        if config.drag < zero {
            return Err(ThickChainError::Invalid("drag"));
        }
        if config.friction < zero {
            return Err(ThickChainError::Invalid("friction"));
        }
        if config.min_radius <= zero {
            return Err(ThickChainError::Invalid("min_radius"));
        }
        if config.splits > 16 {
            return Err(ThickChainError::Invalid("splits"));
        }
        if config.relaxation < zero {
            return Err(ThickChainError::Invalid("relaxation"));
        }
        if config.fit == ThickChainFit::Conforming
            && !matches!(config.stretch, ThickChainStretch::NeoHookean { .. })
        {
            return Err(ThickChainError::Invalid("fit"));
        }
        if config.fit == ThickChainFit::Conforming && config.reach < Fix128::ONE {
            return Err(ThickChainError::Invalid("reach"));
        }
        if points.len() < 2 {
            return Err(ThickChainError::Invalid("points"));
        }
        if radius < config.min_radius {
            return Err(ThickChainError::Invalid("radius"));
        }
        if density <= zero {
            return Err(ThickChainError::Invalid("density"));
        }
        let links = points.len() - 1;
        let mut rest_length = Vec::with_capacity(links);
        let mut area = Vec::with_capacity(links);
        let mut mass = vec![zero; points.len()];
        for k in 0..links {
            let d = sub(points[k + 1], points[k]);
            let l = dot(d, d).sqrt();
            if l.is_zero() {
                return Err(ThickChainError::Degenerate { link: k });
            }
            let a = radius * l + radius * l;
            let half = (density * a).half();
            if half.is_zero() {
                return Err(ThickChainError::Invalid("density"));
            }
            mass[k] = mass[k] + half;
            mass[k + 1] = mass[k + 1] + half;
            rest_length.push(l);
            area.push(a);
        }
        Ok(Self {
            config,
            x: points.to_vec(),
            v: vec![ZERO2; points.len()],
            inv_mass: mass.iter().map(|&m| Fix128::ONE / m).collect(),
            wall_force: vec![zero; rest_length.len()],
            walls_unsettled: 0,
            bend: ThickChainBend::None,
            bend_halved: 0,
            bend_stalled: 0,
            substep_retries: 0,
            last_pass_move: Fix128::ZERO,
            rest_length,
            area,
        })
    }

    /// The chain with the bending law `bend` from now on. A chain is made with
    /// [`ThickChainBend::None`]; this is the only way to another law. It may be called at any
    /// time: the particles, their velocities and the links are unchanged, and the steps taken
    /// after it carry the law. The law is part of the state ([`Self::snapshot`]).
    ///
    /// # Errors
    ///
    /// [`ThickChainError::Invalid`] for [`ThickChainBend::Section`] on links that are not
    /// neo-Hookean (`"bend"`) or with [`ThickChainFit::Conforming`] (`"bend with the
    /// conforming fit"`); [`ThickChainError::Degenerate`] for a link of no length and
    /// [`ThickChainError::Folded`] for a joint already bent to a radius of its half-width or
    /// less (a joint folded flat among them), under `Section`.
    pub fn bending(mut self, bend: ThickChainBend) -> Result<Self, ThickChainError> {
        if bend == ThickChainBend::Section {
            if !matches!(self.config.stretch, ThickChainStretch::NeoHookean { .. }) {
                return Err(ThickChainError::Invalid("bend"));
            }
            if self.config.fit == ThickChainFit::Conforming {
                return Err(ThickChainError::Invalid("bend with the conforming fit"));
            }
            for k in 0..self.rest_length.len() {
                if self.length(k).is_zero() {
                    return Err(ThickChainError::Degenerate { link: k });
                }
            }
            self.bend = bend;
            for joint in 1..self.rest_length.len() {
                self.bend_term(joint)?;
            }
        }
        self.bend = bend;
        Ok(self)
    }

    /// The bending law.
    #[must_use]
    pub fn bend(&self) -> ThickChainBend {
        self.bend
    }

    /// How many times, since the chain was made, a pass with the bending law took a half
    /// of its step in place of the whole (a pass that took a quarter counts 2, and so on):
    /// the whole step did not lower the substep's potential as its slope promised. Read
    /// out of the steps, not part of the state.
    #[must_use]
    pub fn bend_halved(&self) -> u64 {
        self.bend_halved
    }

    /// How many passes with the bending law, since the chain was made, took no step: none
    /// of the parts tried lowered the substep's potential. Read out of the steps, not part
    /// of the state.
    #[must_use]
    pub fn bend_stalled(&self) -> u64 {
        self.bend_stalled
    }

    /// How many times, since the chain was made, a substep was refused (a particle too
    /// fast, a joint folded) and taken again in halves. Read out of the steps, not part
    /// of the state.
    #[must_use]
    pub fn substep_retries(&self) -> u64 {
        self.substep_retries
    }

    /// The furthest a particle was moved along an axis by the last pass of the last substep
    /// taken, m: what the passes of that substep left unsettled (a substep whose passes
    /// settled moves nothing in its last one). 0 before the first step. Read out of the
    /// step, not part of the state.
    #[must_use]
    pub fn last_pass_move(&self) -> Fix128 {
        self.last_pass_move
    }

    /// The settings.
    #[must_use]
    pub fn config(&self) -> ThickChainConfig {
        self.config
    }

    /// The number of particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.x.len()
    }

    /// The number of links (one less than the particles).
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.rest_length.len()
    }

    /// The particles' positions.
    #[must_use]
    pub fn positions(&self) -> &[P2] {
        &self.x
    }

    /// The particles' velocities.
    #[must_use]
    pub fn velocities(&self) -> &[P2] {
        &self.v
    }

    /// Puts particle `i` at `x` (its velocity is kept).
    ///
    /// # Panics
    ///
    /// When `i` is not a particle.
    pub fn set_position(&mut self, i: usize, x: P2) {
        self.x[i] = x;
    }

    /// Gives particle `i` the velocity `v`.
    ///
    /// # Panics
    ///
    /// When `i` is not a particle.
    pub fn set_velocity(&mut self, i: usize, v: P2) {
        self.v[i] = v;
    }

    /// Holds particle `i` where it is: no force or constraint moves it from now on.
    ///
    /// # Panics
    ///
    /// When `i` is not a particle.
    pub fn pin(&mut self, i: usize) {
        self.inv_mass[i] = Fix128::ZERO;
        self.v[i] = ZERO2;
    }

    /// The rest length of link `k`.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    #[must_use]
    pub fn rest_length(&self, k: usize) -> Fix128 {
        self.rest_length[k]
    }

    /// The area of link `k`.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    #[must_use]
    pub fn area(&self, k: usize) -> Fix128 {
        self.area[k]
    }

    /// The walls' press on link `k` in the last substep taken, N per metre of depth (the sum
    /// of the walls' forces across them, whatever their sides): `T l / c` for a link held in
    /// a channel of half-width `c`, less the substep's share stated in the MODEL section. 0
    /// before the first step and without walls. It is read out of the step, not part of the
    /// state: a chain made from a snapshot has 0 until it is stepped.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    #[must_use]
    pub fn wall_force(&self, k: usize) -> Fix128 {
        self.wall_force[k]
    }

    /// How many passes of the conforming rule, since the chain was made, ended their rounds
    /// with the hard walls not settled: a link still pulling on its wall, or one left
    /// shorter than its walls allow by the others' pushes (see `walls_together`). 0 in a run
    /// whose walls were solved as stated; anything else says the bound on the rounds was
    /// met. Read out of the steps, not part of the state.
    #[must_use]
    pub fn walls_unsettled(&self) -> u64 {
        self.walls_unsettled
    }

    /// The length of link `k` now.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    #[must_use]
    pub fn length(&self, k: usize) -> Fix128 {
        let d = sub(self.x[k + 1], self.x[k]);
        dot(d, d).sqrt()
    }

    /// The radius of link `k` now: its area over twice its length, at least the least radius
    /// (a link of no length is given the least radius).
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    #[must_use]
    pub fn radius(&self, k: usize) -> Fix128 {
        self.radius_at(k, self.length(k)).0
    }

    /// the radius of link `k` at length `l`, and whether the area sets it (not the floor)
    fn radius_at(&self, k: usize, l: Fix128) -> (Fix128, bool) {
        let floor = self.config.min_radius;
        if l.is_zero() {
            return (floor, false);
        }
        let r = self.area[k] / (l + l);
        if r > floor {
            (r, true)
        } else {
            (floor, false)
        }
    }

    /// One step of `dt` seconds with no walls and no drive.
    ///
    /// # Errors
    ///
    /// As [`Self::step_in`]; the chain is unchanged then.
    pub fn step(&mut self, dt: Fix128) -> Result<(), ThickChainError> {
        self.advance(dt, None, &[])
    }

    /// One step of `dt` seconds between the walls of `walls` (the distance to the nearest
    /// wall, positive in the free space, read at `z = 0`), each particle driven by its
    /// acceleration in `accel` (m/s², one per particle, or none at all).
    ///
    /// # Errors
    ///
    /// [`ThickChainError::Invalid`] for a `dt` not above 0 or an `accel` of another length,
    /// and the errors of the range (see the module's MODEL section). The chain is unchanged
    /// then.
    pub fn step_in<F: SdfQuery>(
        &mut self,
        dt: Fix128,
        walls: &F,
        accel: &[P2],
    ) -> Result<(), ThickChainError> {
        self.advance(dt, Some(walls), accel)
    }

    fn advance(
        &mut self,
        dt: Fix128,
        walls: Option<&dyn SdfQuery>,
        accel: &[P2],
    ) -> Result<(), ThickChainError> {
        if dt <= Fix128::ZERO {
            return Err(ThickChainError::Invalid("dt"));
        }
        if !accel.is_empty() && accel.len() != self.x.len() {
            return Err(ThickChainError::Invalid("accel"));
        }
        let h = dt / Fix128::from_int(i64::from(self.config.substeps));
        if (h * h).is_zero() {
            return Err(ThickChainError::Invalid("dt"));
        }
        // the substeps run on a copy: a refused one leaves the chain as it was
        let mut next = self.clone();
        for _ in 0..self.config.substeps {
            next.split_substep(h, walls, accel)?;
        }
        *self = next;
        Ok(())
    }

    /// One substep of `h`, in 2ⁿ equal parts: `n` is the least number of halvings (up to the
    /// setting) with which every link's stretch is resolved at the substep's start and no
    /// part is refused for a particle's speed. The parts of a refused try are dropped, so the
    /// result depends on the state alone.
    fn split_substep(
        &mut self,
        h: Fix128,
        walls: Option<&dyn SdfQuery>,
        accel: &[P2],
    ) -> Result<(), ThickChainError> {
        if self.config.splits == 0 {
            return self.substep(h, walls, accel);
        }
        let mut halvings = self.halvings_for_stiffness(h)?;
        let mut retries = 0u64;
        loop {
            let parts = 1i64 << halvings;
            let part = h / Fix128::from_int(parts);
            if (part * part).is_zero() {
                return Err(ThickChainError::Invalid("dt"));
            }
            let mut trial = self.clone();
            let mut refused = None;
            for _ in 0..parts {
                if let Err(error) = trial.substep(part, walls, accel) {
                    refused = Some(error);
                    break;
                }
            }
            match refused {
                None => {
                    *self = trial;
                    self.substep_retries += retries;
                    return Ok(());
                }
                Some(ThickChainError::TooFast { .. } | ThickChainError::Folded { .. })
                    if halvings < self.config.splits =>
                {
                    halvings += 1;
                    retries += 1;
                }
                Some(error) => return Err(error),
            }
        }
    }

    /// The bending constraint of the joint at particle `joint` (between links `joint − 1`
    /// and `joint`) where the chain is. The joint's energy is `(μ / 2) C²`, `C = κ S`:
    /// `κ = 2 tan(θ / 2) / l̄` its curvature (`θ` the angle the links turn by, its tangent
    /// from the links' cross and dot products; `l̄` the mean of their lengths) and
    /// `S² = Σ (A / 2) r0² (1 + λ⁻⁴ g(κ r0 / λ))` over the two links. The slope has every
    /// term: the angle's, and the lengths' through `l̄`, `λ` and `a`.
    fn bend_term(&self, joint: usize) -> Result<BendTerm, ThickChainError> {
        let folded = ThickChainError::Folded { joint };
        let (before, after) = (joint - 1, joint);
        let da = sub(self.x[joint], self.x[joint - 1]);
        let db = sub(self.x[joint + 1], self.x[joint]);
        let (la, lb) = (dot(da, da).sqrt(), dot(db, db).sqrt());
        if la.is_zero() {
            return Err(ThickChainError::Degenerate { link: before });
        }
        if lb.is_zero() {
            return Err(ThickChainError::Degenerate { link: after });
        }
        let cross = da[0] * db[1] - da[1] * db[0];
        // 1 + cos θ, times the lengths: nothing left of it when the joint is folded flat
        let open = la * lb + dot(da, db);
        if open <= Fix128::ZERO {
            return Err(folded);
        }
        let tan_half = cross / open;
        let mean = (la + lb).half();
        let kappa = (tan_half + tan_half) / mean;
        // per link: its share of S², and that share's slopes in κ and in the link's length
        let mut sum = Fix128::ZERO;
        let mut sum_kappa = Fix128::ZERO;
        let mut by_length = [Fix128::ZERO; 2];
        for (side, (k, l)) in [(before, la), (after, lb)].into_iter().enumerate() {
            let rest_radius = self.area[k] / (self.rest_length[k] + self.rest_length[k]);
            let back = self.rest_length[k] / l;
            let a = kappa * rest_radius * back;
            if a.abs() >= bend_limit() {
                return Err(folded);
            }
            let (g, g_slope) = bend_shape(a);
            let back4 = (back * back) * (back * back);
            let weight = self.area[k].half() * rest_radius * rest_radius;
            sum = sum + weight * (Fix128::ONE + back4 * g);
            sum_kappa = sum_kappa + weight * back4 * g_slope * rest_radius * back;
            by_length[side] = -(weight * back4 * (g.double().double() + a * g_slope) / l);
        }
        let s = sum.sqrt();
        if s.is_zero() {
            return Err(ThickChainError::Invalid("bend"));
        }
        let two_s = s + s;
        // dC = with_kappa dκ + κ (S_la dl_a + S_lb dl_b)
        let with_kappa = s + kappa * sum_kappa / two_s;
        // dκ = (1 + tan²) / l̄ dθ − κ / (2 l̄) (dl_a + dl_b)
        let by_angle = with_kappa * (Fix128::ONE + tan_half * tan_half) / mean;
        let shared = with_kappa * kappa / (mean + mean);
        let along_a = kappa * by_length[0] / two_s - shared;
        let along_b = kappa * by_length[1] / two_s - shared;
        // ∂θ/∂d_a = −perp(d_a) / l_a², ∂θ/∂d_b = perp(d_b) / l_b²
        let ga = add(
            scale(perp(da), -(by_angle / (la * la))),
            scale(da, along_a / la),
        );
        let gb = add(
            scale(perp(db), by_angle / (lb * lb)),
            scale(db, along_b / lb),
        );
        Ok(BendTerm {
            value: kappa * s,
            slope: [scale(ga, -Fix128::ONE), sub(ga, gb), gb],
        })
    }

    /// The shear modulus of the bending law's links (the law is set on neo-Hookean links
    /// only).
    fn bend_shear(&self) -> Fix128 {
        match self.config.stretch {
            ThickChainStretch::NeoHookean { shear } => shear,
            ThickChainStretch::Linear { .. } => Fix128::ZERO,
        }
    }

    /// The rows of the substep's constraints where the chain is, in their order along it:
    /// link 0's stretch, the joint at particle 1, link 1's stretch, the joint at particle 2,
    /// and so on (`2 N − 1` for `N` links). A row none of whose particles is free is left
    /// out of the solve (`live` false, nothing else read).
    fn rows(&self, h: Fix128) -> Result<Vec<Row>, ThickChainError> {
        let links = self.rest_length.len();
        let mut rows = Vec::with_capacity(2 * links);
        for at in 0..2 * links - 1 {
            let mut row = Row {
                first: at / 2,
                count: 2 + at % 2,
                slope: [ZERO2; 3],
                value: Fix128::ZERO,
                stiff: Fix128::ZERO,
                live: false,
            };
            let mut free = Fix128::ZERO;
            for n in 0..row.count {
                free = free + self.inv_mass[row.first + n];
            }
            if !free.is_zero() {
                row.live = true;
                if at % 2 == 0 {
                    let k = at / 2;
                    let d = sub(self.x[k + 1], self.x[k]);
                    let l = dot(d, d).sqrt();
                    if l.is_zero() {
                        return Err(ThickChainError::Degenerate { link: k });
                    }
                    let u = scale(d, Fix128::ONE / l);
                    let (value, slope, _) = self.stretch_terms(k, l, h);
                    row.value = value;
                    row.stiff = match self.config.stretch {
                        ThickChainStretch::Linear { compliance } => Fix128::ONE / compliance,
                        ThickChainStretch::NeoHookean { shear } => shear * self.area[k],
                    };
                    row.slope = [scale(u, -slope), scale(u, slope), ZERO2];
                } else {
                    let term = self.bend_term(at / 2 + 1)?;
                    row.value = term.value;
                    row.stiff = self.bend_shear();
                    row.slope = term.slope;
                }
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// The merit of a substep's step with a bending law where the chain is, of the `rows`
    /// there: the substep's potential over `h²`,
    /// `Σ m |x − x̃|² / (2 h²) + Σ [C² + γ (C − C_start)²] / (2 α)`: `per` is `1 / h`,
    /// `anchor` the particles after the velocity step (`x̃`, and what the walls have pushed
    /// them by since), `first` the rows' values where the substep started, `rate` is `γ`.
    fn merit(
        &self,
        rows: &[Row],
        per: Fix128,
        rate: Fix128,
        anchor: &[P2],
        first: &[Fix128],
    ) -> Fix128 {
        let mut sum = Fix128::ZERO;
        for (i, at) in anchor.iter().enumerate() {
            if self.inv_mass[i].is_zero() {
                continue;
            }
            let off = scale(sub(self.x[i], *at), per);
            sum = sum + (dot(off, off) / self.inv_mass[i]).half();
        }
        for (row, &was) in rows.iter().zip(first) {
            if !row.live {
                continue;
            }
            let change = row.value - was;
            sum = sum
                + (row.stiff * row.value * row.value + rate * (row.stiff * change) * change).half();
        }
        sum
    }

    /// The system of a step of the particles with a bending law, in band over their
    /// coordinates (`band[at][back]` the entry of coordinates `at` and `at − back`, two to
    /// a particle) and its right side: the masses over `h²` and the constraints' slopes
    /// weighted by their stiffness, `M / h² + (1 + γ) Gᵀ diag(1 / α) G` (the second slopes
    /// of the merit without those of the constraints), and the merit's slope with its sign
    /// turned, `−(M (x − x̃) / h² + Gᵀ diag(1 / α) (C + γ (C − C_start)))`. A coordinate
    /// of a held particle stands alone with a step of 0. The arguments are [`Self::merit`]'s.
    fn system(
        &self,
        rows: &[Row],
        per: Fix128,
        rate: Fix128,
        anchor: &[P2],
        first: &[Fix128],
    ) -> (Vec<[Fix128; BAND + 1]>, Vec<Fix128>) {
        let mut band = vec![[Fix128::ZERO; BAND + 1]; 2 * self.x.len()];
        let mut right = vec![Fix128::ZERO; 2 * self.x.len()];
        for (i, (place, anchored)) in self.x.iter().zip(anchor).enumerate() {
            for (axis, (here, kept)) in place.iter().zip(anchored).enumerate() {
                let at = 2 * i + axis;
                if self.inv_mass[i].is_zero() {
                    band[at][0] = Fix128::ONE;
                    continue;
                }
                let mass = per * per / self.inv_mass[i];
                band[at][0] = mass;
                right[at] = -(mass * (*here - *kept));
            }
        }
        for (row, &was) in rows.iter().zip(first) {
            if !row.live {
                continue;
            }
            let pull = row.stiff * (row.value + rate * (row.value - was));
            let weight = (Fix128::ONE + rate) * row.stiff;
            for n in 0..row.count {
                let i = row.first + n;
                if self.inv_mass[i].is_zero() {
                    continue;
                }
                for axis in 0..2 {
                    let at = 2 * i + axis;
                    right[at] = right[at] - row.slope[n][axis] * pull;
                    // the entries at or before the diagonal
                    for m in 0..=n {
                        let j = row.first + m;
                        if self.inv_mass[j].is_zero() {
                            continue;
                        }
                        for other in 0..2 {
                            let to = 2 * j + other;
                            if to <= at {
                                band[at][at - to] = band[at][at - to]
                                    + weight * row.slope[n][axis] * row.slope[m][other];
                            }
                        }
                    }
                }
            }
        }
        (band, right)
    }

    /// One step of all the particles together, with a bending law (see the MODEL section):
    /// the step of [`Self::system`] where the chain is ([`solve_band`]), then the first of
    /// the whole step, its half, quarter, eighth and sixteenth that lowers the merit by
    /// [`step_share`] of what its slope along the step promises (to [`MERIT_GRAIN`]). All
    /// five are worked out whichever is taken, so the work of a pass does not depend on the
    /// chain; a part that carries a joint past the radius of a link's half-width lowers
    /// nothing. With none of them the chain stays where it is ([`Self::bend_stalled`]).
    fn together(
        &mut self,
        h: Fix128,
        anchor: &[P2],
        first: &[Fix128],
    ) -> Result<(), ThickChainError> {
        let per = Fix128::ONE / h;
        let rate = self.config.relaxation / h;
        let rows = self.rows(h)?;
        let (band, right) = self.system(&rows, per, rate, anchor, first);
        let Some(step) = solve_band(&band, &right) else {
            return Err(ThickChainError::Degenerate { link: 0 });
        };
        let here = self.merit(&rows, per, rate, anchor, first);
        // the merit's slope along the step
        let mut along = Fix128::ZERO;
        for (slope, moved) in right.iter().zip(&step) {
            along = along - *slope * *moved;
        }
        let from = self.x.clone();
        let mut taken: Option<(u32, Vec<P2>)> = None;
        let mut part = Fix128::ONE;
        for halvings in 0..STEP_PARTS {
            for (i, x) in self.x.iter_mut().enumerate() {
                *x = [
                    from[i][0] + part * step[2 * i],
                    from[i][1] + part * step[2 * i + 1],
                ];
            }
            let lowered = self.rows(h).ok().is_some_and(|rows| {
                self.merit(&rows, per, rate, anchor, first)
                    <= here + step_share() * part * along + MERIT_GRAIN
            });
            if lowered && taken.is_none() {
                taken = Some((halvings, self.x.clone()));
            }
            part = part.half();
        }
        match taken {
            Some((halvings, at)) => {
                self.x = at;
                self.bend_halved += u64::from(halvings);
            }
            None => {
                self.x = from;
                self.bend_stalled += 1;
            }
        }
        Ok(())
    }

    /// The value of link `k`'s stretch constraint at the length `l`, its slope along the
    /// link, and its compliance over `h²`.
    fn stretch_terms(&self, k: usize, l: Fix128, h: Fix128) -> (Fix128, Fix128, Fix128) {
        match self.config.stretch {
            ThickChainStretch::Linear { compliance } => {
                (l - self.rest_length[k], Fix128::ONE, compliance / (h * h))
            }
            ThickChainStretch::NeoHookean { shear } => {
                let ratio = l / self.rest_length[k];
                let back = Fix128::ONE / ratio;
                (
                    ratio - back,
                    (Fix128::ONE + back * back) / self.rest_length[k],
                    Fix128::ONE / (shear * self.area[k] * h * h),
                )
            }
        }
    }

    /// The least halvings of `h` at which every link has `(ω h)² ≤ 1/2` now,
    /// `(ω h)² = (w_i + w_j) |∇C|² h² / α`.
    fn halvings_for_stiffness(&self, h: Fix128) -> Result<u32, ThickChainError> {
        let mut halvings = 0;
        for k in 0..self.rest_length.len() {
            let both = self.inv_mass[k] + self.inv_mass[k + 1];
            let l = self.length(k);
            if both.is_zero() || l.is_zero() {
                continue;
            }
            let (_, slope, softness) = self.stretch_terms(k, l, h);
            // (ω h)² for the whole substep; a halving quarters it
            let mut stiff = both * slope * slope / softness;
            let mut needed = 0;
            while stiff > Fix128::ONE.half() {
                if needed == self.config.splits {
                    return Err(ThickChainError::TooStiff { link: k });
                }
                stiff = stiff.half().half();
                needed += 1;
            }
            if needed > halvings {
                halvings = needed;
            }
        }
        if self.bend == ThickChainBend::Section {
            let softness = Fix128::ONE / (self.bend_shear() * h * h);
            for joint in 1..self.rest_length.len() {
                let term = self.bend_term(joint)?;
                let mut stiff = Fix128::ZERO;
                for (n, slope) in term.slope.iter().enumerate() {
                    stiff = stiff + self.inv_mass[joint - 1 + n] * dot(*slope, *slope);
                }
                stiff = stiff / softness;
                let mut needed = 0;
                while stiff > Fix128::ONE.half() {
                    if needed == self.config.splits {
                        return Err(ThickChainError::TooStiff { link: joint });
                    }
                    stiff = stiff.half().half();
                    needed += 1;
                }
                if needed > halvings {
                    halvings = needed;
                }
            }
        }
        Ok(halvings)
    }

    fn substep(
        &mut self,
        h: Fix128,
        walls: Option<&dyn SdfQuery>,
        accel: &[P2],
    ) -> Result<(), ThickChainError> {
        let c = self.config;
        let start = self.x.clone();
        let whole = self.bend == ThickChainBend::Section;
        // with a bending law, the constraints' values where the substep starts
        let first: Vec<Fix128> = if whole {
            self.rows(h)?.iter().map(|row| row.value).collect()
        } else {
            Vec::new()
        };
        let reach = c.min_radius.half();
        let keep = {
            let k = Fix128::ONE - c.drag * h;
            if k > Fix128::ZERO {
                k
            } else {
                Fix128::ZERO
            }
        };
        for i in 0..self.x.len() {
            if self.inv_mass[i].is_zero() {
                continue;
            }
            let a = accel.get(i).copied().unwrap_or(ZERO2);
            let v = add(scale(self.v[i], keep), scale(add(c.gravity, a), h));
            let step = scale(v, h);
            if dot(step, step) > reach * reach {
                return Err(ThickChainError::TooFast { particle: i });
            }
            self.v[i] = v;
            self.x[i] = add(self.x[i], step);
        }
        let links = self.rest_length.len();
        // the samples of each link for this substep, from its length now
        let mut samples: Vec<i64> = Vec::with_capacity(links);
        if walls.is_some() {
            for k in 0..links {
                samples.push(self.spans(k)?);
            }
        }
        let mut lambda = vec![Fix128::ZERO; links];
        // with a bending law, where the particles' inertia would leave them: after the
        // velocity step, and on by what the walls push them in each pass
        let mut anchor = if whole { self.x.clone() } else { Vec::new() };
        // per link, the walls' multipliers of this substep: their sum and their sum by slope
        let mut pressed = vec![Fix128::ZERO; links];
        let mut leaned = vec![ZERO2; links];
        let conforming = c.fit == ThickChainFit::Conforming;
        let mut before_pass = self.x.clone();
        for pass in 0..c.iterations {
            if pass > 0 {
                // what the pass before this one moved (after the last pass: read below)
                before_pass.clone_from(&self.x);
            }
            if let (true, Some(walls)) = (conforming, walls) {
                // every link's energy, then every link's walls (as the disc rule: a pass
                // ends with the links on the walls)
                for k in 0..links {
                    // the held shares' press on the walls
                    let (press, lean) =
                        self.conform(k, samples[k], h, walls, &start, &mut lambda[k], false)?;
                    pressed[k] = pressed[k] + press;
                    leaned[k] = add(leaned[k], lean);
                }
                self.walls_together(&samples, walls, &mut pressed, &mut leaned)?;
                for k in 0..links {
                    let (press, lean) =
                        self.conform(k, samples[k], h, walls, &start, &mut lambda[k], true)?;
                    pressed[k] = pressed[k] + press;
                    leaned[k] = add(leaned[k], lean);
                }
                continue;
            }
            if whole {
                self.together(h, &anchor, &first)?;
            } else {
                for (k, held) in lambda.iter_mut().enumerate() {
                    let (w0, w1) = (self.inv_mass[k], self.inv_mass[k + 1]);
                    if (w0 + w1).is_zero() {
                        continue;
                    }
                    let d = sub(self.x[k + 1], self.x[k]);
                    let l = dot(d, d).sqrt();
                    if l.is_zero() {
                        return Err(ThickChainError::Degenerate { link: k });
                    }
                    let u = scale(d, Fix128::ONE / l);
                    let (value, slope, softness) = self.stretch_terms(k, l, h);
                    let dl = if c.relaxation.is_zero() {
                        (-value - softness * *held) / ((w0 + w1) * slope * slope + softness)
                    } else {
                        // the Kelvin-Voigt term: the constraint's change over the substep so far
                        let rate = c.relaxation / h;
                        let moved = dot(
                            u,
                            sub(sub(self.x[k + 1], start[k + 1]), sub(self.x[k], start[k])),
                        ) * slope;
                        (-value - softness * *held - rate * moved)
                            / ((Fix128::ONE + rate) * (w0 + w1) * slope * slope + softness)
                    };
                    *held = *held + dl;
                    let step = [scale(u, -(w0 * slope * dl)), scale(u, w1 * slope * dl)];
                    self.x[k] = add(self.x[k], step[0]);
                    self.x[k + 1] = add(self.x[k + 1], step[1]);
                }
            }
            if let Some(walls) = walls {
                let before_walls = if whole { self.x.clone() } else { Vec::new() };
                for (k, &n) in samples.iter().enumerate() {
                    for j in 0..=n {
                        if let Some((push, slope)) = self.fit(k, Fix128::from_ratio(j, n), walls)? {
                            pressed[k] = pressed[k] + push;
                            leaned[k] = add(leaned[k], scale(slope, push));
                        }
                    }
                }
                for (i, was) in before_walls.iter().enumerate() {
                    anchor[i] = add(anchor[i], sub(self.x[i], *was));
                }
            }
        }
        let mut furthest = Fix128::ZERO;
        for (now, was) in self.x.iter().zip(&before_pass) {
            for axis in 0..2 {
                let moved = (now[axis] - was[axis]).abs();
                if moved > furthest {
                    furthest = moved;
                }
            }
        }
        self.last_pass_move = furthest;
        // a held share pulled by its link's tension may leave the walls' press below zero
        for press in &mut pressed {
            if *press < Fix128::ZERO {
                *press = Fix128::ZERO;
            }
        }
        // a multiplier is a force times h²
        for (force, press) in self.wall_force.iter_mut().zip(&pressed) {
            *force = *press / (h * h);
        }
        let rate = Fix128::ONE / h;
        for (i, &from) in start.iter().enumerate() {
            if !self.inv_mass[i].is_zero() {
                self.v[i] = scale(sub(self.x[i], from), rate);
            }
        }
        // friction, on each link as a whole: the velocity of its centre of mass across the
        // walls' mean slope drops by at most μ times the walls' impulse over its mass (both
        // ends alike, so the stretch is untouched); a link with a held end does not slide
        let mu = c.friction;
        for k in 0..links {
            let (w0, w1) = (self.inv_mass[k], self.inv_mass[k + 1]);
            if mu.is_zero() || pressed[k].is_zero() || w0.is_zero() || w1.is_zero() {
                continue;
            }
            let share = w1 / (w0 + w1);
            let whole = add(
                scale(self.v[k], share),
                scale(self.v[k + 1], Fix128::ONE - share),
            );
            let lean = scale(leaned[k], Fix128::ONE / pressed[k]);
            let across = sub(whole, scale(lean, dot(whole, lean)));
            let speed = dot(across, across).sqrt();
            if speed.is_zero() {
                continue;
            }
            let most = mu * pressed[k] * rate * (w0 * w1 / (w0 + w1));
            let drop = if speed < most { speed } else { most };
            let change = scale(across, -(drop / speed));
            self.v[k] = add(self.v[k], change);
            self.v[k + 1] = add(self.v[k + 1], change);
        }
        Ok(())
    }

    /// The wall's slope over ±e at `p` (central differences of the distance), for link `k`.
    fn slope_of(&self, k: usize, p: P2, walls: &dyn SdfQuery) -> Result<P2, ThickChainError> {
        let e = self.config.min_radius.shr_bits(3);
        let over = Fix128::ONE / (e + e);
        let field = |q: P2| -> Result<Fix128, ThickChainError> {
            let d = walls.query_distance(q[0].to_f32(), q[1].to_f32(), 0.0);
            if d.is_finite() {
                Ok(Fix128::from_f32(d))
            } else {
                Err(ThickChainError::Field { link: k })
            }
        };
        Ok([
            (field([p[0] + e, p[1]])? - field([p[0] - e, p[1]])?) * over,
            (field([p[0], p[1] + e])? - field([p[0], p[1] - e])?) * over,
        ])
    }

    /// The hard wall of link `k` as it is now: how far the link is short of the length its
    /// walls make it, `l0 Σ ω_j μ_j − l`, and the slope of that condition in its two ends
    /// (`∓u` along the link plus the walls' part of each end). None for a link with both
    /// ends held.
    fn wall_term(
        &self,
        k: usize,
        n: i64,
        walls: &dyn SdfQuery,
    ) -> Result<Option<WallTerm>, ThickChainError> {
        if (self.inv_mass[k] + self.inv_mass[k + 1]).is_zero() {
            return Ok(None);
        }
        let along = sub(self.x[k + 1], self.x[k]);
        let l = dot(along, along).sqrt();
        if l.is_zero() {
            return Err(ThickChainError::Degenerate { link: k });
        }
        let u = scale(along, Fix128::ONE / l);
        let rest = self.rest_length[k];
        let rest_radius = self.area[k] / (rest + rest);
        let Shares {
            weight,
            place,
            room,
            need,
            ..
        } = self.shares_of(k, n, walls)?;
        let (mut off0, mut off1) = (ZERO2, ZERO2);
        let mut by_walls = Fix128::ZERO;
        let mut leaning = ZERO2;
        for j in 0..weight.len() {
            let t = place[j];
            let p = add(scale(self.x[k], Fix128::ONE - t), scale(self.x[k + 1], t));
            let slope = self.slope_of(k, p, walls)?;
            // a held share's stretch changes by −(r_rest / d²) G per metre of its sample's way
            let by_wall = rest * weight[j] * (rest_radius / (room[j] * room[j]));
            off0 = add(off0, scale(slope, by_wall * (Fix128::ONE - t)));
            off1 = add(off1, scale(slope, by_wall * t));
            by_walls = by_walls + by_wall;
            leaning = add(leaning, scale(slope, by_wall));
        }
        Ok(Some(WallTerm {
            lack: rest * need - l,
            u,
            off0,
            off1,
            by_walls,
            leaning,
        }))
    }

    /// The hard walls of all links at once: no link is shorter than its shares held by the
    /// walls, `l ≥ l0 Σ ω_j μ_j`. Links side by side share a particle, so each one's push
    /// moves its neighbours' lengths: the pushes of a run of links on their walls are the
    /// solution of one tridiagonal system (the conditions made linear at the positions now),
    /// solved directly. A wall only pushes, and a link off its wall is no shorter than it
    /// allows: a link whose push comes out pulling is taken off its wall, a link the others'
    /// pushes would leave too short is put on it, and the system is solved again, at most
    /// [`WALL_ROUNDS`] times, in the links' order; a link still pulling then gets no push in
    /// this pass. Adds each link's press
    /// and its sum by slope.
    #[allow(
        clippy::needless_range_loop,
        reason = "a link's index reads the rows of the system and its two particles"
    )]
    fn walls_together(
        &mut self,
        samples: &[i64],
        walls: &dyn SdfQuery,
        pressed: &mut [Fix128],
        leaned: &mut [P2],
    ) -> Result<(), ThickChainError> {
        let links = self.rest_length.len();
        let mut terms = Vec::with_capacity(links);
        for k in 0..links {
            terms.push(self.wall_term(k, samples[k], walls)?);
        }
        // the system's rows: a link's own answer to its push, and what its push does to the
        // next link's length through the particle they share
        let mut diag = vec![Fix128::ZERO; links];
        let mut next = vec![Fix128::ZERO; links];
        let mut lack = vec![Fix128::ZERO; links];
        let mut able = vec![false; links];
        for k in 0..links {
            let Some(t) = &terms[k] else { continue };
            let (g0, g1) = (sub(t.off0, t.u), add(t.off1, t.u));
            diag[k] = self.inv_mass[k] * dot(g0, g0) + self.inv_mass[k + 1] * dot(g1, g1);
            lack[k] = t.lack;
            able[k] = !diag[k].is_zero();
            if let Some(Some(after)) = terms.get(k + 1) {
                next[k] = self.inv_mass[k + 1] * dot(g1, sub(after.off0, after.u));
            }
        }
        let (push, on, settled) = settle_walls(&diag, &next, &lack, &able, WALL_ROUNDS);
        if !settled {
            self.walls_unsettled += 1;
        }
        for k in 0..links {
            let Some(t) = &terms[k] else { continue };
            if !on[k] || push[k] <= Fix128::ZERO {
                continue;
            }
            let p = push[k];
            let (w0, w1) = (self.inv_mass[k], self.inv_mass[k + 1]);
            // the move per unit inverse mass first (a length times a mass, rounded once),
            // then each end's share: along the link one vector, taken from one end and given
            // to the other
            let shift = scale(t.u, p);
            self.x[k] = add(
                sub(self.x[k], scale(shift, w0)),
                scale(scale(t.off0, p), w0),
            );
            self.x[k + 1] = add(
                add(self.x[k + 1], scale(shift, w1)),
                scale(scale(t.off1, p), w1),
            );
            pressed[k] = pressed[k] + p * t.by_walls;
            leaned[k] = add(leaned[k], scale(t.leaning, p));
        }
        Ok(())
    }

    /// One pass of the conforming rule over link `k` with `n + 1` samples: the link's shares
    /// (trapezoid weights along it) are each as long as the wall there makes them,
    /// `μ_j = r_rest / d_j`, or as the link's tension does, `λ*`, whichever is more, with
    /// `l / l0 = Σ ω_j max(λ*, μ_j)`. Without `wall`: the link's energy as one constraint
    /// (the root of `Σ ω_j (m_j − 1/m_j)²`, `m_j` the larger of the two for share `j`). With
    /// it: the hard wall `l ≥ l0 Σ ω_j μ_j` and the middle line kept the least radius off the
    /// walls, on the link as it is then. Returns the walls' press on the link and its sum by
    /// slope.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::needless_range_loop,
        reason = "one link's pass: the samples, the shares' stretch and the three constraints on them belong together; a sample's index reads six rows"
    )]
    fn conform(
        &mut self,
        k: usize,
        n: i64,
        h: Fix128,
        walls: &dyn SdfQuery,
        start: &[P2],
        held: &mut Fix128,
        wall: bool,
    ) -> Result<(Fix128, P2), ThickChainError> {
        let c = self.config;
        let ThickChainStretch::NeoHookean { shear } = c.stretch else {
            return Err(ThickChainError::Invalid("fit"));
        };
        let field = |p: P2| -> Result<Fix128, ThickChainError> {
            let d = walls.query_distance(p[0].to_f32(), p[1].to_f32(), 0.0);
            if d.is_finite() {
                Ok(Fix128::from_f32(d))
            } else {
                Err(ThickChainError::Field { link: k })
            }
        };
        // the wall's slope over ±e
        let e = c.min_radius.shr_bits(3);
        let over = Fix128::ONE / (e + e);
        let slope_at = |p: P2| -> Result<P2, ThickChainError> {
            Ok([
                (field([p[0] + e, p[1]])? - field([p[0] - e, p[1]])?) * over,
                (field([p[0], p[1] + e])? - field([p[0], p[1] - e])?) * over,
            ])
        };
        let (w0, w1) = (self.inv_mass[k], self.inv_mass[k + 1]);
        let mut press = Fix128::ZERO;
        let mut lean = ZERO2;
        if (w0 + w1).is_zero() {
            return Ok((press, lean));
        }
        let along = sub(self.x[k + 1], self.x[k]);
        let l = dot(along, along).sqrt();
        if l.is_zero() {
            return Err(ThickChainError::Degenerate { link: k });
        }
        let u = scale(along, Fix128::ONE / l);
        let rest = self.rest_length[k];
        let area = self.area[k];
        let rest_radius = area / (rest + rest);
        let Shares {
            weight,
            place,
            room,
            short,
            asked,
            need,
        } = self.shares_of(k, n, walls)?;
        let count = weight.len();
        let mut slope = Vec::with_capacity(count);
        for t in &place {
            let p = add(scale(self.x[k], Fix128::ONE - *t), scale(self.x[k + 1], *t));
            slope.push(slope_at(p)?);
        }
        // what a held share's stretch changes by per metre of its sample's way along the
        // wall's slope: dμ/dp = −(r_rest / d²) G
        let turn = |j: usize| rest_radius / (room[j] * room[j]);
        if !wall {
            // the link's energy over μ A / 2 is S = Σ ω_j (m_j − 1/m_j)², m_j the held share's
            // μ_j or the free shares' λ*: one constraint C = √S with the compliance 1 / (μ A),
            // its multiplier kept through the passes whichever shares are held
            let (is_held, free, tension_stretch) = held_shares(l / rest, need, &weight, &asked);
            let term = |m: Fix128| {
                let back = Fix128::ONE / m;
                (m - back, Fix128::ONE + back * back)
            };
            let (free_value, free_rise) = if free.is_zero() {
                (Fix128::ZERO, Fix128::ZERO)
            } else {
                term(tension_stretch)
            };
            let mut sum = free * free_value * free_value;
            for j in 0..count {
                if is_held[j] {
                    let (value, _) = term(asked[j]);
                    sum = sum + weight[j] * value * value;
                }
            }
            let value = sum.sqrt();
            if !value.is_zero() {
                // dS/2 = Σ_held ω_j (m̃_j s_j − m̃* s*) dμ_j + m̃* s* d(l / l0)
                let pull_by_length = free_value * free_rise / (value * rest);
                // the slope's part along the link is one vector, taken from one end and
                // given to the other; the walls' part is each end's own
                let axial = scale(u, pull_by_length);
                let (mut off0, mut off1) = (ZERO2, ZERO2);
                let mut by_walls = Fix128::ZERO;
                let mut leaning = ZERO2;
                for j in 0..count {
                    if is_held[j] {
                        let (held_value, held_rise) = term(asked[j]);
                        let by_wall = weight[j] * (held_value * held_rise - free_value * free_rise)
                            / value
                            * turn(j);
                        off0 = sub(off0, scale(slope[j], by_wall * (Fix128::ONE - place[j])));
                        off1 = sub(off1, scale(slope[j], by_wall * place[j]));
                        by_walls = by_walls + by_wall;
                        leaning = add(leaning, scale(slope[j], by_wall));
                    }
                }
                let softness = Fix128::ONE / (shear * area * h * h);
                let (g0, g1) = (sub(off0, axial), add(off1, axial));
                let denom = w0 * dot(g0, g0) + w1 * dot(g1, g1);
                let dl = if c.relaxation.is_zero() {
                    (-value - softness * *held) / (denom + softness)
                } else {
                    let rate = c.relaxation / h;
                    let moved = dot(g0, sub(self.x[k], start[k]))
                        + dot(g1, sub(self.x[k + 1], start[k + 1]));
                    (-value - softness * *held - rate * moved)
                        / ((Fix128::ONE + rate) * denom + softness)
                };
                *held = *held + dl;
                // as above: near the walls' own length the slope is large and the step
                // small, and their product is the quantity to round, not the step times an
                // inverse mass
                let shift = scale(axial, dl);
                self.x[k] = add(sub(self.x[k], scale(shift, w0)), scale(scale(off0, dl), w0));
                self.x[k + 1] = add(
                    add(self.x[k + 1], scale(shift, w1)),
                    scale(scale(off1, dl), w1),
                );
                // the held shares press on the walls with what they carry above the tension
                press = press - dl * by_walls;
                lean = add(lean, scale(leaning, -dl));
            }
            return Ok((press, lean));
        }
        // the middle line stays the least radius off the walls
        for j in 0..count {
            if short[j].is_zero() {
                continue;
            }
            let g0 = scale(slope[j], Fix128::ONE - place[j]);
            let g1 = scale(slope[j], place[j]);
            let denom = w0 * dot(g0, g0) + w1 * dot(g1, g1);
            if denom.is_zero() {
                continue;
            }
            let push = short[j] / denom;
            self.x[k] = add(self.x[k], scale(g0, w0 * push));
            self.x[k + 1] = add(self.x[k + 1], scale(g1, w1 * push));
            press = press + push;
            lean = add(lean, scale(slope[j], push));
        }
        Ok((press, lean))
    }

    /// The spans between the samples of link `k`. The disc rule lays them out for the link's
    /// length now (at most `r_min / 2` apart); the conforming rule for the longest the link
    /// may be (`l0 reach`), so the number is the same at every length:
    /// `ceil(2 l0 reach / r_min) + 1`, the ceiling taken in Q64.64.
    ///
    /// # Errors
    ///
    /// [`ThickChainError::Stretched`] for a link beyond `l0 reach` (conforming) or beyond the
    /// samples a link may have.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    pub fn spans(&self, k: usize) -> Result<i64, ThickChainError> {
        let c = &self.config;
        let l = self.length(k);
        let sampled = if c.fit == ThickChainFit::Conforming {
            let most = self.rest_length[k] * c.reach;
            if l > most {
                return Err(ThickChainError::Stretched { link: k });
            }
            most
        } else {
            l
        };
        let n = ((sampled + sampled) / c.min_radius).ceil();
        if n > Fix128::from_int(MAX_SAMPLES) {
            return Err(ThickChainError::Stretched { link: k });
        }
        Ok(n.hi + 1)
    }

    /// The shares of link `k` at `n + 1` samples between `walls`: trapezoid weights that add
    /// up to 1 exactly, each sample's place along the link, the wall's distance there (not
    /// below the least radius), how far the middle line is short of the least radius, the
    /// stretch the wall asks of the share (`r_rest / d`, the planar relation of a share that
    /// keeps its area: width times length), and the sum of the weights times those.
    fn shares_of(&self, k: usize, n: i64, walls: &dyn SdfQuery) -> Result<Shares, ThickChainError> {
        let floor = self.config.min_radius;
        let rest = self.rest_length[k];
        let rest_radius = self.area[k] / (rest + rest);
        let count = n as usize + 1;
        let step = Fix128::ONE / Fix128::from_int(n);
        let mut out = Shares {
            weight: Vec::with_capacity(count),
            place: Vec::with_capacity(count),
            room: Vec::with_capacity(count),
            short: Vec::with_capacity(count),
            asked: Vec::with_capacity(count),
            need: Fix128::ZERO,
        };
        let mut given = Fix128::ZERO;
        for j in 0..=n {
            let t = Fix128::from_ratio(j, n);
            let p = add(scale(self.x[k], Fix128::ONE - t), scale(self.x[k + 1], t));
            let d = walls.query_distance(p[0].to_f32(), p[1].to_f32(), 0.0);
            if !d.is_finite() {
                return Err(ThickChainError::Field { link: k });
            }
            let d = Fix128::from_f32(d);
            // the last share takes what is left, so the weights add up to 1 exactly
            let w = if j == n {
                Fix128::ONE - given
            } else if j == 0 {
                step.half()
            } else {
                step
            };
            given = given + w;
            let (kept, lack) = if d > floor {
                (d, Fix128::ZERO)
            } else {
                (floor, floor - d)
            };
            let mu = rest_radius / kept;
            out.need = out.need + w * mu;
            out.weight.push(w);
            out.place.push(t);
            out.room.push(kept);
            out.short.push(lack);
            out.asked.push(mu);
        }
        Ok(out)
    }

    /// The stretch of link `k`'s free shares between `walls` now (the conforming rule's
    /// `λ*`: what the link's tension answers to; the least stretch the walls ask of a share
    /// when the link is no longer than they make it).
    ///
    /// # Errors
    ///
    /// [`ThickChainError::Degenerate`] for a link of no length, [`ThickChainError::Stretched`]
    /// for one beyond the samples a link may have, [`ThickChainError::Field`] for a distance
    /// that is not a number.
    ///
    /// # Panics
    ///
    /// When `k` is not a link.
    pub fn free_stretch<F: SdfQuery>(
        &self,
        k: usize,
        walls: &F,
    ) -> Result<Fix128, ThickChainError> {
        let l = self.length(k);
        if l.is_zero() {
            return Err(ThickChainError::Degenerate { link: k });
        }
        let shares = self.shares_of(k, self.spans(k)?, walls)?;
        let (_, _, stretch) = held_shares(
            l / self.rest_length[k],
            shares.need,
            &shares.weight,
            &shares.asked,
        );
        Ok(stretch)
    }

    /// The wall constraint of link `k` at its sample `t`: a disc of the link's radius fits
    /// there (the wall constraint of the MODEL section). Returns the multiplier and the wall's
    /// slope when the sample was short of its radius.
    fn fit(
        &mut self,
        k: usize,
        t: Fix128,
        walls: &dyn SdfQuery,
    ) -> Result<Option<(Fix128, P2)>, ThickChainError> {
        let field = |p: P2| -> Result<Fix128, ThickChainError> {
            let d = walls.query_distance(p[0].to_f32(), p[1].to_f32(), 0.0);
            if d.is_finite() {
                Ok(Fix128::from_f32(d))
            } else {
                Err(ThickChainError::Field { link: k })
            }
        };
        let (w0, w1) = (self.inv_mass[k], self.inv_mass[k + 1]);
        let (a, b) = (Fix128::ONE - t, t);
        let d = sub(self.x[k + 1], self.x[k]);
        let l = dot(d, d).sqrt();
        if l.is_zero() {
            return Err(ThickChainError::Degenerate { link: k });
        }
        let u = scale(d, Fix128::ONE / l);
        let (r, by_area) = self.radius_at(k, l);
        let p = add(scale(self.x[k], a), scale(self.x[k + 1], b));
        let short = field(p)? - r;
        if short >= Fix128::ZERO {
            return Ok(None);
        }
        // the wall's slope over ±e
        let e = self.config.min_radius.shr_bits(3);
        let over = Fix128::ONE / (e + e);
        let slope = [
            (field([p[0] + e, p[1]])? - field([p[0] - e, p[1]])?) * over,
            (field([p[0], p[1] + e])? - field([p[0], p[1] - e])?) * over,
        ];
        let thin = if by_area { r / l } else { Fix128::ZERO };
        let g0 = sub(scale(slope, a), scale(u, thin));
        let g1 = add(scale(slope, b), scale(u, thin));
        let denom = w0 * dot(g0, g0) + w1 * dot(g1, g1);
        if denom.is_zero() {
            return Ok(None);
        }
        let push = -short / denom;
        self.x[k] = add(self.x[k], scale(g0, w0 * push));
        self.x[k + 1] = add(self.x[k + 1], scale(g1, w1 * push));
        Ok(Some((push, slope)))
    }

    /// The chain's state as bytes: the settings, then per particle its position, velocity and
    /// inverse mass, then per link its rest length and area (`Q64.64`, little endian).
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&Self::SNAPSHOT_MAGIC);
        let version = match self.bend {
            ThickChainBend::None => Self::SNAPSHOT_VERSION,
            ThickChainBend::Section => Self::SNAPSHOT_VERSION_BEND,
        };
        out.extend_from_slice(&version.to_le_bytes());
        let c = &self.config;
        out.extend_from_slice(&c.substeps.to_le_bytes());
        out.extend_from_slice(&c.iterations.to_le_bytes());
        out.extend_from_slice(&(self.x.len() as u64).to_le_bytes());
        let mut put = |x: Fix128| {
            out.extend_from_slice(&x.hi.to_le_bytes());
            out.extend_from_slice(&x.lo.to_le_bytes());
        };
        // the stretch law: 0 and the compliance, or 1 and the shear modulus
        let (law, stiffness) = match c.stretch {
            ThickChainStretch::Linear { compliance } => (Fix128::ZERO, compliance),
            ThickChainStretch::NeoHookean { shear } => (Fix128::ONE, shear),
        };
        for x in [
            c.gravity[0],
            c.gravity[1],
            law,
            stiffness,
            c.drag,
            c.friction,
            c.min_radius,
            Fix128::from_int(i64::from(c.splits)),
            c.relaxation,
            match c.fit {
                ThickChainFit::Disc => Fix128::ZERO,
                ThickChainFit::Conforming => Fix128::ONE,
            },
            c.reach,
        ] {
            put(x);
        }
        // a chain with a bending law: the law's tag (1: the strip's section)
        if self.bend == ThickChainBend::Section {
            put(Fix128::ONE);
        }
        for i in 0..self.x.len() {
            put(self.x[i][0]);
            put(self.x[i][1]);
            put(self.v[i][0]);
            put(self.v[i][1]);
            put(self.inv_mass[i]);
        }
        for k in 0..self.rest_length.len() {
            put(self.rest_length[k]);
            put(self.area[k]);
        }
        out
    }

    /// The SHA-256 of [`Self::snapshot`]: the same on every machine for the same run.
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        Sha256::digest(self.snapshot()).into()
    }
}

// The closed-form oracles of this module (`tests/analytic_thick_chain.rs` and
// `tests/analytic_thick_chain_neo_hookean.rs`) also run as lib tests, so the module's lines
// count in the lib's coverage. `tests/analytic_thick_chain_anchor.rs` and
// `tests/analytic_thick_chain_entry.rs` are integration tests only: they settle chains over
// many frames (70 s and 80 s at opt-level 2), and the lib tests also run without
// optimisation (the minimum-toolchain job, twice), where the anchor file alone added six
// minutes to each run. Their longest tests (the two studies of the push at a mouth over
// several resolutions) are `#[ignore = "runtime: …"]` and run by the weekly job, each with a
// one-resolution test that runs every time: on the
// slowest test runner the four files added five and a half minutes to each full pass.
// `tests/analytic_thick_chain_shares.rs` (the conforming rule) runs as a lib test too; its two
// long-settling oracles are in `tests/analytic_thick_chain_pocket.rs`, an integration test only
// (four of the shares file's five minutes without optimisation).
#[cfg(test)]
#[path = "../tests/analytic_thick_chain_neo_hookean.rs"]
mod analytic_thick_chain_neo_hookean_oracle;
#[cfg(test)]
#[path = "../tests/analytic_thick_chain.rs"]
mod analytic_thick_chain_oracle;
#[cfg(test)]
#[path = "../tests/analytic_thick_chain_shares.rs"]
mod analytic_thick_chain_shares_oracle;

#[cfg(test)]
mod bend_shape_oracle {
    //! The shape of the bending law, `g(a) = (artanh(a) / a − 1) / a²`, against its series
    //! and its closed form in f64, and the two ways of taking it against each other where
    //! one hands over to the other.
    #![allow(
        clippy::float_arithmetic,
        clippy::cast_precision_loss,
        clippy::disallowed_methods,
        reason = "the reference values are f64"
    )]
    use super::{bend_shape, bend_shape_log, bend_shape_series, bend_switch, Fix128};

    /// oracle: the series, summed until the terms are below f64's resolution
    fn series(a: f64) -> (f64, f64) {
        let (mut g, mut slope) = (0.0, 0.0);
        for n in (0..400i32).rev() {
            g += a.powi(2 * n) / f64::from(2 * n + 3);
            if n > 0 {
                slope += f64::from(2 * n) * a.powi(2 * n - 1) / f64::from(2 * n + 3);
            }
        }
        (g, slope)
    }

    #[test]
    fn the_shape_is_the_series_of_the_law_on_both_sides_of_the_switch() {
        // the series in f64 is good to its last digits below about 0.9
        for a in [
            1.0e-9, 1.0e-5, 1.0e-3, 0.03, 0.1, 0.2, 0.2499, 0.25, 0.2501, 0.3, 0.45, 0.6, 0.8,
        ] {
            let (want_g, want_slope) = series(a);
            let at = Fix128::from_f64(a);
            let (g, slope) = bend_shape(at);
            assert!(
                (g.to_f64() / want_g - 1.0).abs() < 4.0e-15,
                "g({a}) = {}, the series {want_g}",
                g.to_f64()
            );
            // (the value read is of `at`, which is `a` to 2⁻⁶⁴: a part in 10¹⁰ at 10⁻⁹)
            assert!(
                (slope.to_f64() / want_slope - 1.0).abs() < 4.0e-15 + 1.0e-19 / a,
                "g'({a}) = {}, the series {want_slope}",
                slope.to_f64()
            );
            // the shape is even and its slope odd, to the bit (so the other side is not read
            // through a small negative number as f64)
            assert_eq!(bend_shape(-at), (g, -slope));
        }
        // near the wall the closed form in f64: g grows as the logarithm, its slope as
        // 1 / (1 − a²)
        for a in [0.95f64, 0.999, 0.999_999] {
            let want_g = (a.atanh() / a - 1.0) / (a * a);
            let want_slope = (1.0 / (1.0 - a * a) - 3.0 * want_g) / a;
            let (g, slope) = bend_shape(Fix128::from_f64(a));
            assert!((g.to_f64() / want_g - 1.0).abs() < 1.0e-12);
            assert!((slope.to_f64() / want_slope - 1.0).abs() < 1.0e-9);
        }
    }

    #[test]
    fn a_straight_joint_has_the_shape_a_third_and_no_slope_exactly() {
        assert_eq!(
            bend_shape(Fix128::ZERO),
            (Fix128::from_ratio(1, 3), Fix128::ZERO)
        );
    }

    #[test]
    fn the_series_is_taken_below_a_quarter_and_the_logarithm_from_it_on() {
        let quarter = Fix128::from_ratio(1, 4);
        let ulp = Fix128::from_raw(0, 1);
        // the switch is on a²: the last a whose square is below 1/16 as stored is four
        // ulps under a quarter (one ulp under, the square rounds to 1/16)
        let under = quarter - Fix128::from_raw(0, 4);
        assert_eq!(quarter * quarter, bend_switch());
        assert!(under * under < bend_switch());
        assert_eq!((quarter - ulp) * (quarter - ulp), bend_switch());
        for a in [
            Fix128::from_ratio(1, 64),
            Fix128::from_ratio(1, 8),
            under,
            -under,
        ] {
            assert_eq!(bend_shape(a), bend_shape_series(a));
            // (and there the two differ: the comparison above is not of a thing with itself)
            assert_ne!(bend_shape_series(a), bend_shape_log(a));
        }
        for a in [
            quarter,
            -quarter,
            quarter - ulp,
            quarter + ulp,
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(9, 10),
        ] {
            assert_eq!(bend_shape(a), bend_shape_log(a));
        }
    }

    #[test]
    fn where_one_hands_over_the_two_agree_to_the_logarithms_resolution() {
        // at a = 1/4 half an ulp of the logarithm (2⁻⁶⁵) is 2⁻⁵⁸ of g − 1/3 and what the
        // series leaves out is below 2⁻⁸¹: the two are within 2⁻⁵⁰ of each other, and the
        // slopes (a further division by a and a difference of 0.03) within 2⁻⁴⁶
        let quarter = Fix128::from_ratio(1, 4);
        let (g_series, slope_series) = bend_shape_series(quarter);
        let (g_log, slope_log) = bend_shape_log(quarter);
        assert!((g_series - g_log).abs() < Fix128::ONE.shr_bits(50));
        assert!((slope_series - slope_log).abs() < Fix128::ONE.shr_bits(46));
        // far below the switch the closed form has lost its digits: the reason for the series
        let small = Fix128::from_ratio(1, 4096);
        let lost = (bend_shape_series(small).0 - bend_shape_log(small).0).abs();
        assert!(lost > Fix128::ONE.shr_bits(40), "{}", lost.to_f64());
    }
}

#[cfg(test)]
mod band_oracle {
    //! The step of all the particles of a substep with a bending law, solved in band,
    //! against the same system put together entry by entry over every pair of coordinates
    //! and solved by elimination with pivoting, in f64; and the system's right side against
    //! the merit's slope by differences.
    #![allow(
        clippy::float_arithmetic,
        clippy::cast_precision_loss,
        clippy::needless_range_loop,
        clippy::disallowed_methods,
        reason = "the reference solve is f64 over a few dozen rows"
    )]
    use super::{
        solve_band, Fix128, ThickChain, ThickChainBend, ThickChainConfig, ThickChainFit,
        ThickChainStretch, BAND, P2,
    };

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    /// a number in [−1, 1) from a counter (a linear congruence, the same on every run)
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
    }

    const LINKS: usize = 20;
    const H: f64 = 1.0 / 60.0 / 32.0;

    /// a chain of 20 links off its line and off its lengths, three of its particles held
    /// (two together, one alone); where its inertia would leave its particles; and its
    /// constraints' values where the substep started
    fn state(seed: u64) -> (ThickChain, Vec<P2>, Vec<Fix128>) {
        let mut seed = seed;
        let cfg = ThickChainConfig {
            substeps: 32,
            iterations: 8,
            gravity: [Fix128::ZERO, Fix128::ZERO],
            stretch: ThickChainStretch::NeoHookean { shear: fx(1000.0) },
            drag: Fix128::ZERO,
            friction: Fix128::ZERO,
            min_radius: fx(0.002),
            splits: 0,
            relaxation: fx(0.05),
            fit: ThickChainFit::Disc,
            reach: Fix128::ONE,
        };
        let points: Vec<P2> = (0..=LINKS)
            .map(|i| [fx(0.01 * i as f64), Fix128::ZERO])
            .collect();
        let mut chain = ThickChain::new(cfg, &points, fx(0.02), fx(1000.0))
            .expect("a chain")
            .bending(ThickChainBend::Section)
            .expect("the law");
        chain.pin(0);
        chain.pin(1);
        chain.pin(9);
        let mut anchor = Vec::new();
        let mut places = Vec::new();
        for i in 0..=LINKS {
            let at = [
                0.01 * i as f64 + 0.001 * noise(&mut seed),
                0.0003 * noise(&mut seed),
            ];
            anchor.push([
                fx(at[0] + 0.00005 * noise(&mut seed)),
                fx(at[1] + 0.00005 * noise(&mut seed)),
            ]);
            // where the substep started: a little off where the chain is
            chain.set_position(
                i,
                [
                    fx(at[0] + 0.00003 * noise(&mut seed)),
                    fx(at[1] + 0.00003 * noise(&mut seed)),
                ],
            );
            places.push([fx(at[0]), fx(at[1])]);
        }
        let first = chain
            .rows(fx(H))
            .expect("rows")
            .iter()
            .map(|row| row.value)
            .collect();
        for (i, at) in places.into_iter().enumerate() {
            chain.set_position(i, at);
        }
        (chain, anchor, first)
    }

    /// the system's matrix with its right side as a last column, entry by entry over every
    /// pair of coordinates: `δ m / h² + (1 + γ) Σ_c g_c(a) g_c(b) / α_c` and
    /// `−(m (x − x̃) / h² + Σ_c g_c(a) (C_c + γ (C_c − C_c,start)) / α_c)`
    fn dense(chain: &ThickChain, anchor: &[P2], first: &[Fix128]) -> Vec<Vec<f64>> {
        let rows = chain.rows(fx(H)).expect("rows");
        let n = 2 * chain.x.len();
        let gamma = chain.config.relaxation.to_f64() / H;
        // each row's slope at every coordinate
        let mut g = vec![vec![0.0f64; n]; rows.len()];
        for (r, row) in rows.iter().enumerate() {
            for k in 0..row.count {
                for axis in 0..2 {
                    g[r][2 * (row.first + k) + axis] = row.slope[k][axis].to_f64();
                }
            }
        }
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for at in 0..n {
            let i = at / 2;
            if chain.inv_mass[i].is_zero() {
                a[at][at] = 1.0;
                continue;
            }
            let mass = 1.0 / chain.inv_mass[i].to_f64() / (H * H);
            a[at][at] = mass;
            a[at][n] = -mass * (chain.x[i][at % 2] - anchor[i][at % 2]).to_f64();
            for (r, row) in rows.iter().enumerate() {
                let (stiff, value) = (row.stiff.to_f64(), row.value.to_f64());
                a[at][n] -= g[r][at] * stiff * (value + gamma * (value - first[r].to_f64()));
                for to in 0..n {
                    if !chain.inv_mass[to / 2].is_zero() {
                        a[at][to] += (1.0 + gamma) * stiff * g[r][at] * g[r][to];
                    }
                }
            }
        }
        a
    }

    /// Gaussian elimination with partial pivoting
    fn eliminate(mut a: Vec<Vec<f64>>) -> Vec<f64> {
        let n = a.len();
        for col in 0..n {
            let mut best = col;
            for r in col + 1..n {
                if a[r][col].abs() > a[best][col].abs() {
                    best = r;
                }
            }
            a.swap(col, best);
            for r in col + 1..n {
                let f = a[r][col] / a[col][col];
                for c in col..=n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
        let mut out = vec![0.0f64; n];
        for r in (0..n).rev() {
            let mut v = a[r][n];
            for c in r + 1..n {
                v -= a[r][c] * out[c];
            }
            out[r] = v / a[r][r];
        }
        out
    }

    #[test]
    fn the_step_solved_in_band_is_the_step_of_the_whole_system_by_elimination() {
        let mut compared = 0;
        let mut largest = 0.0f64;
        for seed in 1..=8u64 {
            let (chain, anchor, first) = state(seed);
            let rows = chain.rows(fx(H)).expect("rows");
            let per = Fix128::ONE / fx(H);
            let rate = chain.config.relaxation / fx(H);
            let (band, right) = chain.system(&rows, per, rate, &anchor, &first);
            let got = solve_band(&band, &right).expect("positive definite");
            let want = eliminate(dense(&chain, &anchor, &first));
            assert_eq!(got.len(), 2 * (LINKS + 1));
            // the coordinates of held particles are out of both solves
            for at in [0, 1, 2, 3, 18, 19] {
                assert!(got[at].is_zero() && want[at] == 0.0);
            }
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(scale > 0.0);
            for (g, w) in got.iter().zip(&want) {
                // f64 through an elimination of 42 rows: nine digits of the largest step
                assert!(
                    (g.to_f64() - w).abs() <= 1e-9 * scale,
                    "seed {seed}: {} against {w} (largest {scale})",
                    g.to_f64()
                );
                compared += 1;
            }
            largest = largest.max(scale);
        }
        assert_eq!(compared, 8 * 2 * (LINKS + 1));
        assert!(largest > 0.0);
    }

    #[test]
    fn the_right_side_is_the_merits_slope_by_differences() {
        let mut compared = 0;
        for seed in 1..=4u64 {
            let (chain, anchor, first) = state(seed);
            let per = Fix128::ONE / fx(H);
            let rate = chain.config.relaxation / fx(H);
            let rows = chain.rows(fx(H)).expect("rows");
            let (_, right) = chain.system(&rows, per, rate, &anchor, &first);
            let merit = |chain: &ThickChain| {
                let rows = chain.rows(fx(H)).expect("rows");
                chain.merit(&rows, per, rate, &anchor, &first).to_f64()
            };
            let largest = right.iter().fold(0.0f64, |m, v| m.max(v.to_f64().abs()));
            assert!(largest > 0.0);
            for i in 0..=LINKS {
                for axis in 0..2 {
                    let step = 1.0e-7;
                    let (mut up, mut down) = (chain.clone(), chain.clone());
                    up.x[i][axis] = up.x[i][axis] + fx(step);
                    down.x[i][axis] = down.x[i][axis] - fx(step);
                    let slope = (merit(&up) - merit(&down)) / (2.0 * step);
                    let got = right[2 * i + axis].to_f64();
                    if chain.inv_mass[i].is_zero() {
                        assert!(got == 0.0);
                        continue;
                    }
                    // central differences over 1e-7 m: the third slopes leave 1e-6 of the largest
                    assert!(
                        (got + slope).abs() <= 1e-6 * largest,
                        "seed {seed} particle {i} axis {axis}: {got} against {}",
                        -slope
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 4 * 2 * (LINKS + 1 - 3));
    }

    #[test]
    fn a_system_that_is_not_positive_definite_is_refused() {
        let one = Fix128::ONE;
        let mut band = vec![[Fix128::ZERO; BAND + 1]; 3];
        for row in &mut band {
            row[0] = one;
        }
        let right = vec![one; 3];
        assert!(solve_band(&band, &right).is_some());
        // 1 2; 2 1 has a negative pivot
        band[1][1] = one + one;
        assert!(solve_band(&band, &right).is_none());
        band[1][1] = Fix128::ZERO;
        band[2][0] = Fix128::ZERO;
        assert!(solve_band(&band, &right).is_none());
    }
}

#[cfg(test)]
mod wall_solve {
    //! The walls' pushes on a row of links against all the ways of choosing which links are
    //! on their walls, each solved by elimination in f64: the one choice with no pull and no
    //! link left short.
    #![allow(
        clippy::float_arithmetic,
        clippy::cast_precision_loss,
        clippy::needless_range_loop,
        reason = "the reference solve is f64 over a few rows"
    )]
    use super::{settle_walls, Fix128};

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    /// every subset of links on their walls, the pushes of each by elimination; the subset
    /// whose pushes do not pull and leave no other link short
    fn by_trying_all(diag: &[f64], next: &[f64], lack: &[f64]) -> (Vec<f64>, Vec<bool>) {
        let n = diag.len();
        let entry = |i: usize, j: usize| {
            if i == j {
                diag[i]
            } else if i + 1 == j {
                next[i]
            } else if j + 1 == i {
                next[j]
            } else {
                0.0
            }
        };
        let mut found = Vec::new();
        for set in 0u32..(1 << n) {
            let rows: Vec<usize> = (0..n).filter(|k| set >> k & 1 == 1).collect();
            let m = rows.len();
            let mut a = vec![vec![0.0f64; m + 1]; m];
            for (r, &i) in rows.iter().enumerate() {
                for (c, &j) in rows.iter().enumerate() {
                    a[r][c] = entry(i, j);
                }
                a[r][m] = lack[i];
            }
            for c in 0..m {
                let pivot = (c..m)
                    .max_by(|x, y| a[*x][c].abs().total_cmp(&a[*y][c].abs()))
                    .expect("a row");
                a.swap(c, pivot);
                for r in 0..m {
                    if r != c {
                        let by = a[r][c] / a[c][c];
                        for q in c..=m {
                            a[r][q] -= by * a[c][q];
                        }
                    }
                }
            }
            let mut p = vec![0.0f64; n];
            for (r, &i) in rows.iter().enumerate() {
                p[i] = a[r][m] / a[r][r];
            }
            let fits = (0..n).all(|i| {
                let after: f64 = (0..n).map(|j| entry(i, j) * p[j]).sum::<f64>() - lack[i];
                if set >> i & 1 == 1 {
                    p[i] >= 0.0
                } else {
                    after >= 0.0
                }
            });
            if fits {
                found.push((p, (0..n).map(|k| set >> k & 1 == 1).collect()));
            }
        }
        assert_eq!(found.len(), 1, "one choice fits");
        found.remove(0)
    }

    fn check(diag: &[f64], next: &[f64], lack: &[f64], rounds_needed: usize) {
        let (want, want_on) = by_trying_all(diag, next, lack);
        let to = |v: &[f64]| v.iter().map(|x| fx(*x)).collect::<Vec<_>>();
        let able = vec![true; diag.len()];
        let (push, on, settled) = settle_walls(&to(diag), &to(next), &to(lack), &able, 8);
        assert!(settled, "not settled in 8 rounds");
        assert_eq!(on, want_on);
        for k in 0..diag.len() {
            let got = if on[k] { push[k].to_f64() } else { 0.0 };
            assert!(
                (got - want[k]).abs() <= 1.0e-12,
                "link {k}: push {got}, expected {}",
                want[k]
            );
        }
        // one round fewer than the choice takes is reported, not passed off as settled
        if rounds_needed > 1 {
            let (_, _, early) =
                settle_walls(&to(diag), &to(next), &to(lack), &able, rounds_needed - 1);
            assert!(!early, "settled in fewer rounds than the choice takes");
        }
    }

    #[test]
    fn short_links_side_by_side_are_pushed_together() {
        // a push on a link shortens its neighbours: every short link gets a push, each more
        // than it would alone (alone: 0.5 each)
        check(&[2.0, 2.0, 2.0], &[-1.0, -1.0, 0.0], &[1.0, 1.0, 1.0], 1);
    }

    #[test]
    fn a_link_its_neighbours_would_leave_short_comes_onto_its_wall() {
        // the second link is longer than its walls ask by 0.2; the first's push (0.5 alone)
        // would shorten it by 0.5: it comes on, and the two are solved together (0.6, 0.2)
        check(&[2.0, 2.0], &[-1.0, 0.0], &[1.0, -0.2], 2);
    }

    #[test]
    fn a_short_link_held_up_by_its_neighbour_is_taken_off_its_wall() {
        // where a neighbour's push lengthens a link (a positive coupling), the three solved
        // together give the middle one a pull (−0.4): it is taken off, the outer two are
        // solved alone (0.5 each) and leave it 0.3 longer than its walls ask
        check(&[2.0, 2.0, 2.0], &[1.5, -0.5, 0.0], &[1.0, 0.2, 1.0], 2);
    }

    #[test]
    fn a_link_with_no_condition_of_its_own_stays_off() {
        let to = |v: &[f64]| v.iter().map(|x| fx(*x)).collect::<Vec<_>>();
        let (push, on, settled) = settle_walls(
            &to(&[2.0, 0.0, 2.0]),
            &to(&[0.0, 0.0, 0.0]),
            &to(&[1.0, 1.0, 1.0]),
            &[true, false, true],
            8,
        );
        assert!(settled);
        assert_eq!(on, [true, false, true]);
        assert_eq!(push[1], Fix128::ZERO);
        assert!((push[0].to_f64() - 0.5).abs() <= 1.0e-12);
    }
}
