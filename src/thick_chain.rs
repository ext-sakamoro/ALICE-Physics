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
//!   `T` the link's tension. Then `v ← (x − x_start) / h`, and friction (Coulomb, per link): the
//!   velocity of the link's centre of mass across the walls' mean slope `Σ Δλ G / Σ Δλ` drops
//!   by at most `μ Σ Δλ / (h (m_i + m_j))`, both ends alike; the lengthening itself meets no
//!   friction.
//! - **Units.** SI: m, s, kg, N; planar, so masses, forces and the compliance are per metre
//!   of depth (kg/m, N/m, `α` in m²/N); `k` in 1/s.
//! - **Conservation.** A link's area is `2 r l` by the definition of `r`, exactly, down to
//!   `r_min` (below it the link is drawn wider than its area). Linear momentum is conserved
//!   by the stretch constraint; the walls and their friction take it. Energy is not
//!   conserved: the constraints are solved implicitly (backward Euler), which damps a free
//!   swing by `1 / √(1 + (ω h)²)` a substep (`ω² = (w_i + w_j) / α`, linear law), the drag `k`
//!   and the friction dissipate, and a wall projection takes the approach speed (the walls do not bounce a
//!   particle; a body thrown at a wall is flattened and springs back by its own stretch).
//! - **Range.** Planar. `α > 0`, `μ > 0` (a rigid link cannot be squeezed). A substep in
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
//!   stretched beyond 2048 samples, a link of no length, a field value that is not a number
//!   and an invalid argument are refused; a refused step changes nothing. The distance field
//!   must not overstate the distance (Lipschitz constant at most 1); it is read in `f32`.
//! - **Approximations.** A link has one radius along its length (the outline steps between
//!   links). The body has no bending stiffness and no shear; its thickness follows its
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
//!   carried to the next substep.
//! - **Validation.** `tests/analytic_thick_chain.rs::a_link_of_fixed_area_is_thinner_when_longer::a_hanging_link_stretches_by_its_compliance_times_the_weight::a_free_link_swings_on_the_backward_euler_solution_whatever_the_iterations::a_chain_in_a_narrow_channel_is_longer_by_the_ratio_of_the_widths::a_chain_sliding_on_a_floor_stops_after_the_coulomb_distance::a_chain_held_thin_by_a_channel_stops_after_the_squeeze_friction_distance::a_chain_driven_through_a_wider_pipe_stays_out_of_the_walls::a_chain_driven_through_a_narrower_pipe_is_squeezed_and_comes_back::a_run_repeats_bit_for_bit::a_refused_chain_or_step_changes_nothing` `tests/analytic_thick_chain_neo_hookean.rs::a_hanging_link_settles_where_the_neo_hookean_tension_is_its_weight::a_small_stretch_meets_the_plane_strain_modulus::a_standing_link_is_shortened_to_where_the_tension_is_minus_its_weight::a_neo_hookean_chain_held_thin_stops_after_the_squeeze_friction_distance::the_stretch_law_is_part_of_the_state` `tests/analytic_thick_chain_anchor.rs::a_fast_link_is_refused_without_split_substeps_and_flies_straight_with_them::a_run_that_needs_no_split_is_the_same_bits_with_and_without::a_link_thrown_at_a_floor_stops_on_it::a_link_too_stiff_for_its_substep_is_given_the_halvings_that_resolve_it::a_free_link_with_inner_friction_swings_on_the_recurrence_of_its_implicit_step::the_fixed_tail_keeps_its_bits_through_a_pull_between_walls::every_free_link_of_a_pulled_chain_is_where_its_tension_is_the_pull::the_reach_in_a_channel_is_the_longer_of_the_walls_hold_and_the_pulls_stretch::let_go_the_free_part_swings_back_on_its_first_modes_recurrence::after_the_pull_ends_the_energy_never_grows` `tests/analytic_thick_chain_entry.rs::pushed_slowly_no_chain_is_fed_below_the_push_of_its_squeeze_energy::moved_slowly_the_push_works_no_less_than_the_links_take_and_the_loss_goes_with_the_links_length::fed_every_link_is_as_long_as_the_channel_makes_it`
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
}

impl ThickChain {
    const SNAPSHOT_MAGIC: [u8; 4] = *b"TCHN";
    const SNAPSHOT_VERSION: u32 = 1;

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
            rest_length,
            area,
        })
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
                    return Ok(());
                }
                Some(ThickChainError::TooFast { .. }) if halvings < self.config.splits => {
                    halvings += 1;
                }
                Some(error) => return Err(error),
            }
        }
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
                let l = self.length(k);
                let n = ((l + l) / c.min_radius).ceil();
                if n > Fix128::from_int(MAX_SAMPLES) {
                    return Err(ThickChainError::Stretched { link: k });
                }
                samples.push(n.hi + 1);
            }
        }
        let mut lambda = vec![Fix128::ZERO; links];
        // per link, the walls' multipliers of this substep: their sum and their sum by slope
        let mut pressed = vec![Fix128::ZERO; links];
        let mut leaned = vec![ZERO2; links];
        for _ in 0..c.iterations {
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
                self.x[k] = sub(self.x[k], scale(u, w0 * slope * dl));
                self.x[k + 1] = add(self.x[k + 1], scale(u, w1 * slope * dl));
            }
            if let Some(walls) = walls {
                for (k, &n) in samples.iter().enumerate() {
                    for j in 0..=n {
                        if let Some((push, slope)) = self.fit(k, Fix128::from_ratio(j, n), walls)? {
                            pressed[k] = pressed[k] + push;
                            leaned[k] = add(leaned[k], scale(slope, push));
                        }
                    }
                }
            }
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
        out.extend_from_slice(&Self::SNAPSHOT_VERSION.to_le_bytes());
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
        ] {
            put(x);
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

// The closed-form oracles of this module (`tests/analytic_thick_chain.rs`) also run as lib
// tests, so the module's lines count in the lib's coverage.
#[cfg(test)]
#[path = "../tests/analytic_thick_chain_anchor.rs"]
mod analytic_thick_chain_anchor_oracle;
#[cfg(test)]
#[path = "../tests/analytic_thick_chain_neo_hookean.rs"]
mod analytic_thick_chain_neo_hookean_oracle;
#[cfg(test)]
#[path = "../tests/analytic_thick_chain.rs"]
mod analytic_thick_chain_oracle;
