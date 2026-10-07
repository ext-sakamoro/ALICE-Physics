//! Social force model of pedestrian motion (Helbing–Molnár 1995,
//! Helbing–Farkas–Vicsek 2000).
//!
//! Each pedestrian `i` is a disc of radius `r_i` and mass `m_i` moving in a
//! plane. Its acceleration is the sum of a driving term, the interaction with
//! every other pedestrian `j` and the interaction with every wall `W`:
//!
//! ```text
//! m_i dv_i/dt = m_i (v0_i ê_i − v_i)/τ_i  +  Σ_j f_ij  +  Σ_W f_iW
//! ```
//!
//! # Pedestrian–pedestrian interaction
//!
//! With `d_ij = |x_i − x_j|`, `r_ij = r_i + r_j`, `n_ij = (x_i − x_j)/d_ij`
//! (from `j` to `i`), the tangent `t_ij = (−n_ij,y, n_ij,x)`, the tangential
//! relative velocity `Δv^t_ji = (v_j − v_i)·t_ij` and `g(x) = max(x, 0)`:
//!
//! ```text
//! f_ij = w_i A e^{(r_ij − d_ij)/B} n_ij            social repulsion
//!      + k g(r_ij − d_ij) n_ij                       body force (contact only)
//!      + κ g(r_ij − d_ij) Δv^t_ji t_ij               sliding friction (contact only)
//! ```
//!
//! The view-angle weight on the social term (Helbing–Molnár 1995; the
//! continuous form of Johansson, Helbing, Shukla 2007) is
//!
//! ```text
//! w_i = λ + (1 − λ)(1 + cos φ_ij)/2,    cos φ_ij = −n_ij · ê_i
//! ```
//!
//! so a pedestrian straight ahead (`cos φ = 1`) acts with weight 1 and one
//! straight behind (`cos φ = −1`) with weight `λ ∈ [0, 1]`. The heading `ê_i`
//! is the desired direction, not the velocity, so a pedestrian at rest still
//! has a view angle. The body force and the friction are reciprocal
//! (`f_ji = −f_ij` for those terms, bit for bit); the social term is reciprocal
//! only for `λ = 1`, because `w_i ≠ w_j` in general.
//!
//! # Pedestrian–wall interaction
//!
//! A wall is a line segment (a degenerate segment is a point). With `d_iW` the
//! distance to the nearest point of the segment, `n_iW` the unit vector from
//! that point to the pedestrian and `t_iW` its tangent:
//!
//! ```text
//! f_iW = {A_W e^{(r_i − d_iW)/B_W} + k_W g(r_i − d_iW)} n_iW − κ_W g(r_i − d_iW)(v_i·t_iW) t_iW
//! ```
//!
//! The wall has its own [`InteractionParams`] and no view-angle weight. Walls
//! are two-sided.
//!
//! # Plane
//!
//! The model is written in a 2D plane ([`Vec2Fix`]). For a 3D scene the
//! caller maps the horizontal plane (for example `(x, z)` of a y-up world)
//! to it and back; heights do not enter the law.
//!
//! # Cutoff and neighbour search
//!
//! The exponential has infinite range. [`SocialForce::total_forces`] and
//! [`SocialForce::step`] only include pedestrian pairs with
//! `d_ij ≤ cutoff`; this cutoff is part of the model (choose it so that
//! `A e^{(r_ij − cutoff)/B}` is negligible, e.g. 2 m for `A = 2000 N`,
//! `B = 0.08 m` leaves `5e-5 N`). Two searches give the same pair set:
//!
//! - [`NeighborSearch::Direct`]: every pair `i < j`, `O(N²)`.
//! - [`NeighborSearch::CellList`]: a sparse cell list sorted by cell, `O(N log N)`.
//!
//! Both visit pairs as `(i, j)` with `i < j` and evaluate each pair with the
//! same code, and `Fix128` addition is an exact integer addition, so both
//! return bit-identical forces. The cell list is local to this module rather
//! than [`crate::spatial::SpatialGrid`]: that grid is a dense 3D array of
//! `dim³` cells centred on the origin (a plane would use one layer of it and
//! the whole array is cleared on every build), while a crowd domain is planar
//! and can be long (a corridor of hundreds of metres). The cells here are
//! `cutoff · (1 + 2⁻²⁰)` wide, so the rounding of `x / cell` cannot push a pair
//! at distance `cutoff` two cells apart (for coordinates below `2⁴³` cells)
//! whatever the rounding direction of `Fix128` division and multiplication.
//! With the current truncating `1/cell` the margin is not needed (a search of
//! 5.6·10⁶ near-boundary pairs found no pair two cells apart without it); it
//! keeps the guarantee independent of that rounding.
//! Walls are summed directly (there are few of them) with the same cutoff on
//! the distance `d_iW` from the centre to the segment.
//!
//! # Integration
//!
//! [`SocialForce::step`] advances with semi-implicit Euler,
//! `v ← v + (F/m) h`, then the optional speed cap `|v| ≤ c v0` (HFV 2000 use
//! `v_max = 1.3 v0`), then `x ← x + v h`. The contact terms are stiff: the
//! body force has a relative frequency `√(2k/m)` (≈ 55 rad/s for
//! `k = 1.2·10⁵ kg/s²`, `m = 80 kg`) and the friction a tangential damping
//! rate `2 κ g/m` (≈ 300 1/s at an overlap of 5 cm), so `h` of a few
//! milliseconds is needed for the representative values.
//!
//! # Not in this module
//!
//! - Route choice, goals and navigation meshes: the caller sets `ê_i` (and
//!   `v0_i`) every step.
//! - ORCA / velocity-obstacle avoidance: it chooses a velocity by a linear
//!   program over half-planes, i.e. it is a collision-avoidance *policy*,
//!   not a force law, and belongs with the navigation layer.
//! - A [`crate::solver::PhysicsWorld`] entry: pedestrians are integrated
//!   here as point masses in the plane. Bodies of a world can be driven by
//!   the forces from [`SocialForce::total_forces`] through the world's
//!   external-force entry, but that entry applies the force once at the head
//!   of a frame while gravity and contacts are sub-stepped, which shifts
//!   positions by `O(F/m · dt²)` per frame; the driving-term relaxation is
//!   then no longer `e^{−t/τ}` exactly. The crowd itself takes part in the
//!   world's substep loop as [`CrowdParticipant`] (pedestrians only, no
//!   coupling to rigid bodies).
//!
//! # Degenerate inputs
//!
//! - Two pedestrians at the same position: `n_ij` is undefined; it is taken
//!   as 0, so the pair force is exactly 0 on both.
//! - A pedestrian on the wall line (`d_iW = 0`): the wall force is 0.
//! - `ê_i = 0` (no preferred direction): driving towards rest
//!   (`m (0 − v)/τ`) and isotropic view weight `w_i = 1`. A non-unit `ê_i`
//!   is normalised.
//! - Invalid parameters are errors, never clamped: `B ≤ 0`, `A, k, κ < 0`,
//!   `λ ∉ [0, 1]`, `cutoff ≤ 0` at [`SocialForce::new`]; `τ ≤ 0`, `m ≤ 0`,
//!   `r < 0`, `v0 < 0` per pedestrian; `h ≤ 0` and a speed cap `≤ 0` at
//!   [`SocialForce::step`]. A step that fails leaves the crowd unchanged.
//! - Values out of the `Fix128` range: [`SocialForce::step`] and
//!   [`SocialForce::total_forces`] use the wrapping operators, so a force,
//!   velocity or position that leaves the range wraps silently.
//!   [`SocialForce::try_step`] evaluates the same expressions in the same
//!   order with every product, quotient, sum and difference checked, returns
//!   [`CrowdStepError::Overflow`] instead and leaves the crowd unchanged;
//!   where it succeeds its result is bit-identical to `step`'s.
//!   [`CrowdParticipant`] advances with `try_step`.
//! - `N = 0` returns an empty force list; `N = 1` has only the driving term
//!   and the walls.
//! - A social magnitude `A e^{(r − d)/B}` above the `Fix128` range (only for
//!   overlaps beyond about `36 B` with `A = 2000 N`) saturates at the largest
//!   representable value instead of wrapping.
//!
//! # Symmetry
//!
//! Mirror and rotation symmetry hold only up to rounding: a [`Fix128`] product
//! rounds towards −∞, so `(−a)·b` and `−(a·b)` can differ by one unit of
//! `2⁻⁶⁴`, and a mirrored or rotated input can give results that differ in
//! the last bits.
//!
//! # References
//!
//! - D. Helbing, P. Molnár, *Social force model for pedestrian dynamics*,
//!   Phys. Rev. E 51, 4282 (1995).
//! - D. Helbing, I. Farkas, T. Vicsek, *Simulating dynamical features of
//!   escape panic*, Nature 407, 487 (2000) (`A = 2·10³ N`, `B = 0.08 m`,
//!   `k = 1.2·10⁵ kg/s²`, `κ = 2.4·10⁵ kg/(m s)`, `τ = 0.5 s`, `m = 80 kg`).
//! - A. Johansson, D. Helbing, P. K. Shukla, *Specification of the social
//!   force pedestrian model by evolutionary adjustment to video tracking
//!   data*, Adv. Complex Syst. 10, 271 (2007).
//!
//! Author: Moroya Sakamoto

use crate::math::Fix128;
use crate::molecular_dynamics::{checked_quotient, checked_sub};
use crate::physics2d::Vec2Fix;
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Strength, range, stiffness and friction of one kind of interaction
/// (pedestrian–pedestrian or pedestrian–wall).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractionParams {
    /// `A` (N): magnitude of the social repulsion at zero gap.
    pub strength_n: Fix128,
    /// `B` (m): range of the social repulsion. Must be positive.
    pub range_m: Fix128,
    /// `k` (kg/s²): body force per unit overlap.
    pub body_stiffness: Fix128,
    /// `κ` (kg/(m s)): sliding friction per unit overlap and unit tangential speed.
    pub sliding_friction: Fix128,
}

/// One pedestrian: a disc in the plane with its own desired motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pedestrian {
    /// Position of the centre (m).
    pub position: Vec2Fix,
    /// Velocity (m/s).
    pub velocity: Vec2Fix,
    /// Radius `r_i` (m), `≥ 0`.
    pub radius_m: Fix128,
    /// Mass `m_i` (kg), `> 0`.
    pub mass_kg: Fix128,
    /// Desired speed `v0_i` (m/s), `≥ 0`.
    pub desired_speed_m_s: Fix128,
    /// Desired direction `ê_i`; normalised internally, `0` means "come to rest".
    pub desired_direction: Vec2Fix,
    /// Relaxation time `τ_i` (s), `> 0`.
    pub relaxation_time_s: Fix128,
}

/// A wall: the segment from `start` to `end` (a point if they coincide).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WallSegment {
    /// First end point (m).
    pub start: Vec2Fix,
    /// Second end point (m).
    pub end: Vec2Fix,
}

/// How [`SocialForce::total_forces`] finds the pedestrian pairs within the cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSearch {
    /// Every pair `i < j` (`O(N²)`).
    Direct,
    /// Sparse cell list of cells `cutoff · (1 + 2⁻²⁰)` wide (`O(N log N)`);
    /// bit-identical to [`NeighborSearch::Direct`].
    CellList,
}

/// Invalid model, pedestrian or step parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrowdForceError {
    /// `B ≤ 0` in an [`InteractionParams`].
    NonPositiveRange,
    /// `A`, `k` or `κ` negative in an [`InteractionParams`].
    NegativeCoefficient,
    /// View-angle weight `λ` outside `[0, 1]`.
    AnisotropyOutOfRange,
    /// Pair cutoff `≤ 0`.
    NonPositiveCutoff,
    /// `τ ≤ 0` for the pedestrian at `index` (0 for a single-pedestrian call).
    NonPositiveRelaxationTime {
        /// Index of the pedestrian in the slice.
        index: usize,
    },
    /// `m ≤ 0` for the pedestrian at `index` (0 for a single-pedestrian call).
    NonPositiveMass {
        /// Index of the pedestrian in the slice.
        index: usize,
    },
    /// `r < 0` for the pedestrian at `index`.
    NegativeRadius {
        /// Index of the pedestrian in the slice.
        index: usize,
    },
    /// `v0 < 0` for the pedestrian at `index`.
    NegativeDesiredSpeed {
        /// Index of the pedestrian in the slice.
        index: usize,
    },
    /// Time step `h ≤ 0`.
    NonPositiveTimeStep,
    /// Speed-cap ratio `≤ 0`.
    NonPositiveSpeedCap,
}

impl core::fmt::Display for CrowdForceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveRange => f.write_str("interaction range B must be positive"),
            Self::NegativeCoefficient => {
                f.write_str("interaction strength, stiffness and friction must be non-negative")
            }
            Self::AnisotropyOutOfRange => f.write_str("view-angle weight must be in [0, 1]"),
            Self::NonPositiveCutoff => f.write_str("pair cutoff must be positive"),
            Self::NonPositiveRelaxationTime { index } => {
                write!(f, "pedestrian {index}: relaxation time must be positive")
            }
            Self::NonPositiveMass { index } => {
                write!(f, "pedestrian {index}: mass must be positive")
            }
            Self::NegativeRadius { index } => {
                write!(f, "pedestrian {index}: radius must be non-negative")
            }
            Self::NegativeDesiredSpeed { index } => {
                write!(f, "pedestrian {index}: desired speed must be non-negative")
            }
            Self::NonPositiveTimeStep => f.write_str("time step must be positive"),
            Self::NonPositiveSpeedCap => f.write_str("speed-cap ratio must be positive"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CrowdForceError {}

/// Why [`SocialForce::try_step`] did not advance the crowd. The crowd is
/// unchanged in every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrowdStepError {
    /// An error that [`SocialForce::step`] reports as well.
    Crowd(CrowdForceError),
    /// A value of the step left the `Fix128` range (`|x| ≥ 2⁶³`): a force
    /// term (driving, pair, wall) or its sum, a squared distance or the
    /// squared cutoff, a cell key, the velocity update `v + F h/m`, the speed
    /// cap or the position update `x + v h`.
    Overflow,
}

impl From<CrowdForceError> for CrowdStepError {
    fn from(e: CrowdForceError) -> Self {
        Self::Crowd(e)
    }
}

impl core::fmt::Display for CrowdStepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Crowd(e) => core::fmt::Display::fmt(e, f),
            Self::Overflow => f.write_str("a value of the step is outside the fixed-point range"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CrowdStepError {}

/// The social force model: pedestrian and wall interaction parameters, the
/// view-angle weight `λ` and the pair cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocialForce {
    pedestrian: InteractionParams,
    wall: InteractionParams,
    anisotropy: Fix128,
    cutoff_m: Fix128,
}

/// Largest representable `Fix128`, the saturation value of the social magnitude.
const FIX_MAX: Fix128 = Fix128::from_raw(i64::MAX, u64::MAX);

fn check_params(p: &InteractionParams) -> Result<(), CrowdForceError> {
    if p.range_m <= Fix128::ZERO {
        return Err(CrowdForceError::NonPositiveRange);
    }
    if p.strength_n.is_negative()
        || p.body_stiffness.is_negative()
        || p.sliding_friction.is_negative()
    {
        return Err(CrowdForceError::NegativeCoefficient);
    }
    Ok(())
}

fn check_pedestrian(p: &Pedestrian, index: usize) -> Result<(), CrowdForceError> {
    if p.relaxation_time_s <= Fix128::ZERO {
        return Err(CrowdForceError::NonPositiveRelaxationTime { index });
    }
    if p.mass_kg <= Fix128::ZERO {
        return Err(CrowdForceError::NonPositiveMass { index });
    }
    if p.radius_m.is_negative() {
        return Err(CrowdForceError::NegativeRadius { index });
    }
    if p.desired_speed_m_s.is_negative() {
        return Err(CrowdForceError::NegativeDesiredSpeed { index });
    }
    Ok(())
}

/// `A e^{x/B}`, saturating instead of wrapping.
fn social_magnitude(p: &InteractionParams, x: Fix128) -> Fix128 {
    let e = (x / p.range_m).exp();
    p.strength_n.checked_mul(e).unwrap_or(FIX_MAX)
}

impl SocialForce {
    /// Build the model.
    ///
    /// `anisotropy` is `λ ∈ [0, 1]` (1 = isotropic); `cutoff_m` is the largest
    /// pair distance included by [`Self::total_forces`] and [`Self::step`].
    ///
    /// # Errors
    ///
    /// [`CrowdForceError::NonPositiveRange`], [`CrowdForceError::NegativeCoefficient`],
    /// [`CrowdForceError::AnisotropyOutOfRange`] or [`CrowdForceError::NonPositiveCutoff`].
    pub fn new(
        pedestrian: InteractionParams,
        wall: InteractionParams,
        anisotropy: Fix128,
        cutoff_m: Fix128,
    ) -> Result<Self, CrowdForceError> {
        check_params(&pedestrian)?;
        check_params(&wall)?;
        if anisotropy.is_negative() || anisotropy > Fix128::ONE {
            return Err(CrowdForceError::AnisotropyOutOfRange);
        }
        if cutoff_m <= Fix128::ZERO {
            return Err(CrowdForceError::NonPositiveCutoff);
        }
        Ok(Self {
            pedestrian,
            wall,
            anisotropy,
            cutoff_m,
        })
    }

    /// Driving force `m (v0 ê − v)/τ` with `ê` normalised (0 if `ê = 0`).
    ///
    /// # Errors
    ///
    /// An invalid pedestrian (`τ ≤ 0`, `m ≤ 0`, `r < 0`, `v0 < 0`), reported with `index: 0`.
    pub fn driving_force(&self, p: &Pedestrian) -> Result<Vec2Fix, CrowdForceError> {
        check_pedestrian(p, 0)?;
        Ok(driving(p, p.desired_direction.normalize()))
    }

    /// View-angle weight `λ + (1 − λ)(1 + cos φ)/2` of pedestrian `p` for a
    /// neighbour in the direction `−n_ij` (`n_ij` points from the neighbour
    /// to `p`), `cos φ = −n_ij · ê`. `ê = 0` gives 1.
    #[must_use]
    pub fn anisotropy_weight(&self, p: &Pedestrian, n_ij: Vec2Fix) -> Fix128 {
        self.weight(p.desired_direction.normalize(), n_ij)
    }

    fn weight(&self, e_hat: Vec2Fix, n_ij: Vec2Fix) -> Fix128 {
        if e_hat == Vec2Fix::ZERO {
            return Fix128::ONE;
        }
        let cos_phi = -n_ij.dot(e_hat);
        self.anisotropy + (Fix128::ONE - self.anisotropy) * (Fix128::ONE + cos_phi).half()
    }

    /// Forces of the pair `(a, b)`: `(f_ab on a, f_ba on b)`, without cutoff.
    ///
    /// The body force and the friction are computed once and applied with
    /// opposite signs; the social term carries each side's own view weight.
    #[must_use]
    pub fn pair_forces(&self, a: &Pedestrian, b: &Pedestrian) -> (Vec2Fix, Vec2Fix) {
        self.pair_kernel(
            a,
            b,
            a.desired_direction.normalize(),
            b.desired_direction.normalize(),
        )
    }

    fn pair_kernel(
        &self,
        a: &Pedestrian,
        b: &Pedestrian,
        ea: Vec2Fix,
        eb: Vec2Fix,
    ) -> (Vec2Fix, Vec2Fix) {
        let diff = a.position - b.position;
        let d2 = diff.length_squared();
        // Explicit, although `Fix128` division by 0 returns 0 and would give
        // n = 0 too: the result must not depend on that convention.
        if d2.is_zero() {
            return (Vec2Fix::ZERO, Vec2Fix::ZERO);
        }
        let d = d2.sqrt();
        let n = diff / d;
        let overlap = a.radius_m + b.radius_m - d;
        let social = social_magnitude(&self.pedestrian, overlap);
        let mut fa = n * (social * self.weight(ea, n));
        let mut fb = -(n * (social * self.weight(eb, -n)));
        if overlap > Fix128::ZERO {
            let t = n.perpendicular();
            let dvt = (b.velocity - a.velocity).dot(t);
            let contact = n * (self.pedestrian.body_stiffness * overlap)
                + t * (self.pedestrian.sliding_friction * overlap * dvt);
            fa = fa + contact;
            fb = fb - contact;
        }
        (fa, fb)
    }

    /// Force of wall `w` on pedestrian `p`.
    #[must_use]
    pub fn wall_force(&self, p: &Pedestrian, w: &WallSegment) -> Vec2Fix {
        let diff = p.position - nearest_point(p, w);
        let d2 = diff.length_squared();
        if d2.is_zero() {
            return Vec2Fix::ZERO;
        }
        let d = d2.sqrt();
        let n = diff / d;
        let overlap = p.radius_m - d;
        let mut f = n * social_magnitude(&self.wall, overlap);
        if overlap > Fix128::ZERO {
            let t = n.perpendicular();
            f = f + n * (self.wall.body_stiffness * overlap)
                - t * (self.wall.sliding_friction * overlap * p.velocity.dot(t));
        }
        f
    }

    /// Total force on every pedestrian (driving + pairs within the cutoff +
    /// walls within the cutoff), written to `out` (cleared first, one entry per pedestrian).
    ///
    /// # Errors
    ///
    /// The first invalid pedestrian, with its index; `out` is then empty.
    pub fn total_forces(
        &self,
        peds: &[Pedestrian],
        walls: &[WallSegment],
        search: NeighborSearch,
        out: &mut Vec<Vec2Fix>,
    ) -> Result<(), CrowdForceError> {
        out.clear();
        for (i, p) in peds.iter().enumerate() {
            check_pedestrian(p, i)?;
        }
        let cutoff2 = self.cutoff_m * self.cutoff_m;
        let headings: Vec<Vec2Fix> = peds
            .iter()
            .map(|p| p.desired_direction.normalize())
            .collect();
        for (p, &e) in peds.iter().zip(&headings) {
            let mut f = driving(p, e);
            for w in walls {
                if wall_distance_squared(p, w) <= cutoff2 {
                    f = f + self.wall_force(p, w);
                }
            }
            out.push(f);
        }
        let mut add_pair = |i: usize, j: usize| {
            let (a, b) = (&peds[i], &peds[j]);
            if (a.position - b.position).length_squared() > cutoff2 {
                return;
            }
            let (fa, fb) = self.pair_kernel(a, b, headings[i], headings[j]);
            out[i] = out[i] + fa;
            out[j] = out[j] + fb;
        };
        match search {
            NeighborSearch::Direct => {
                for i in 0..peds.len() {
                    for j in i + 1..peds.len() {
                        add_pair(i, j);
                    }
                }
            }
            NeighborSearch::CellList => {
                for (i, j) in cell_list_pairs(peds, self.cutoff_m) {
                    add_pair(i, j);
                }
            }
        }
        Ok(())
    }

    /// Advance the crowd by `h` with semi-implicit Euler (see the module doc).
    ///
    /// `speed_cap_ratio = Some(c)` limits `|v|` to `c · v0` after the velocity
    /// update.
    ///
    /// The arithmetic is not range-checked: a force, velocity or position
    /// that leaves the `Fix128` range wraps. [`Self::try_step`] checks the
    /// range and is bit-identical to this step where it succeeds.
    ///
    /// # Errors
    ///
    /// `h ≤ 0`, `c ≤ 0` or an invalid pedestrian; the crowd is left unchanged.
    pub fn step(
        &self,
        peds: &mut [Pedestrian],
        walls: &[WallSegment],
        h: Fix128,
        search: NeighborSearch,
        speed_cap_ratio: Option<Fix128>,
    ) -> Result<(), CrowdForceError> {
        if h <= Fix128::ZERO {
            return Err(CrowdForceError::NonPositiveTimeStep);
        }
        if let Some(c) = speed_cap_ratio {
            if c <= Fix128::ZERO {
                return Err(CrowdForceError::NonPositiveSpeedCap);
            }
        }
        let mut forces = Vec::with_capacity(peds.len());
        self.total_forces(peds, walls, search, &mut forces)?;
        for (p, f) in peds.iter_mut().zip(forces) {
            let mut v = p.velocity + f * (h / p.mass_kg);
            if let Some(c) = speed_cap_ratio {
                let v_max = c * p.desired_speed_m_s;
                let speed = v.length();
                if speed > v_max {
                    v = if speed.is_zero() {
                        Vec2Fix::ZERO
                    } else {
                        v * (v_max / speed)
                    };
                }
            }
            p.velocity = v;
            p.position = p.position + v * h;
        }
        Ok(())
    }

    /// [`Self::step`] with the range checked: the same expressions in the
    /// same order, with every product, quotient, sum and difference of the
    /// forces (driving, pair, wall and their sums, the squared distances of
    /// the cutoff tests, the cell keys) and of the update (`v + F h/m`, the
    /// speed cap, `x + v h`) checked against the `Fix128` range.
    ///
    /// Where it returns `Ok` the crowd is bit-identical to the one
    /// [`Self::step`] produces. On error the crowd is not changed.
    ///
    /// # Errors
    ///
    /// [`CrowdStepError::Crowd`] with the errors of [`Self::step`];
    /// [`CrowdStepError::Overflow`] for a value out of range.
    pub fn try_step(
        &self,
        peds: &mut [Pedestrian],
        walls: &[WallSegment],
        h: Fix128,
        search: NeighborSearch,
        speed_cap_ratio: Option<Fix128>,
    ) -> Result<(), CrowdStepError> {
        if h <= Fix128::ZERO {
            return Err(CrowdForceError::NonPositiveTimeStep.into());
        }
        if let Some(c) = speed_cap_ratio {
            if c <= Fix128::ZERO {
                return Err(CrowdForceError::NonPositiveSpeedCap.into());
            }
        }
        for (i, p) in peds.iter().enumerate() {
            check_pedestrian(p, i)?;
        }
        let forces = self
            .total_forces_checked(peds, walls, search)
            .ok_or(CrowdStepError::Overflow)?;
        let next = peds
            .iter()
            .zip(forces)
            .map(|(p, f)| advance_checked(p, f, h, speed_cap_ratio))
            .collect::<Option<Vec<Pedestrian>>>()
            .ok_or(CrowdStepError::Overflow)?;
        peds.copy_from_slice(&next);
        Ok(())
    }

    /// [`Self::total_forces`] for validated pedestrians with every operation
    /// range-checked (`None` if a value leaves the range).
    fn total_forces_checked(
        &self,
        peds: &[Pedestrian],
        walls: &[WallSegment],
        search: NeighborSearch,
    ) -> Option<Vec<Vec2Fix>> {
        let cutoff2 = self.cutoff_m.checked_mul(self.cutoff_m)?;
        let headings = peds
            .iter()
            .map(|p| normalize_c(p.desired_direction))
            .collect::<Option<Vec<Vec2Fix>>>()?;
        let mut out = Vec::with_capacity(peds.len());
        for (p, &e) in peds.iter().zip(&headings) {
            let mut f = driving_c(p, e)?;
            for w in walls {
                let near = nearest_point_c(p, w)?;
                if len_sq_c(sub_c(p.position, near)?)? <= cutoff2 {
                    f = add_c(f, self.wall_force_c(p, w)?)?;
                }
            }
            out.push(f);
        }
        let mut add_pair = |i: usize, j: usize| -> Option<()> {
            let (a, b) = (&peds[i], &peds[j]);
            if len_sq_c(sub_c(a.position, b.position)?)? > cutoff2 {
                return Some(());
            }
            let (fa, fb) = self.pair_kernel_c(a, b, headings[i], headings[j])?;
            out[i] = add_c(out[i], fa)?;
            out[j] = add_c(out[j], fb)?;
            Some(())
        };
        match search {
            NeighborSearch::Direct => {
                for i in 0..peds.len() {
                    for j in i + 1..peds.len() {
                        add_pair(i, j)?;
                    }
                }
            }
            NeighborSearch::CellList => {
                for (i, j) in cell_list_pairs_checked(peds, self.cutoff_m)? {
                    add_pair(i, j)?;
                }
            }
        }
        Some(out)
    }

    /// [`Self::weight`] range-checked.
    fn weight_c(&self, e_hat: Vec2Fix, n_ij: Vec2Fix) -> Option<Fix128> {
        if e_hat == Vec2Fix::ZERO {
            return Some(Fix128::ONE);
        }
        let cos_phi = neg_c(dot_c(n_ij, e_hat)?)?;
        let opening = Fix128::ONE.checked_add(cos_phi)?.half();
        let spread = checked_sub(Fix128::ONE, self.anisotropy)?.checked_mul(opening)?;
        self.anisotropy.checked_add(spread)
    }

    /// [`Self::pair_kernel`] range-checked.
    fn pair_kernel_c(
        &self,
        a: &Pedestrian,
        b: &Pedestrian,
        ea: Vec2Fix,
        eb: Vec2Fix,
    ) -> Option<(Vec2Fix, Vec2Fix)> {
        let diff = sub_c(a.position, b.position)?;
        let d2 = len_sq_c(diff)?;
        if d2.is_zero() {
            return Some((Vec2Fix::ZERO, Vec2Fix::ZERO));
        }
        let d = d2.sqrt();
        let n = div_c(diff, d)?;
        let overlap = checked_sub(a.radius_m.checked_add(b.radius_m)?, d)?;
        let social = social_magnitude_c(&self.pedestrian, overlap)?;
        let mut fa = scale_c(n, social.checked_mul(self.weight_c(ea, n)?)?)?;
        let mut fb = neg2_c(scale_c(
            n,
            social.checked_mul(self.weight_c(eb, neg2_c(n)?)?)?,
        )?)?;
        if overlap > Fix128::ZERO {
            let t = perpendicular_c(n)?;
            let dvt = dot_c(sub_c(b.velocity, a.velocity)?, t)?;
            let normal = scale_c(n, self.pedestrian.body_stiffness.checked_mul(overlap)?)?;
            let tangential = scale_c(
                t,
                self.pedestrian
                    .sliding_friction
                    .checked_mul(overlap)?
                    .checked_mul(dvt)?,
            )?;
            let contact = add_c(normal, tangential)?;
            fa = add_c(fa, contact)?;
            fb = sub_c(fb, contact)?;
        }
        Some((fa, fb))
    }

    /// [`Self::wall_force`] range-checked.
    fn wall_force_c(&self, p: &Pedestrian, w: &WallSegment) -> Option<Vec2Fix> {
        let diff = sub_c(p.position, nearest_point_c(p, w)?)?;
        let d2 = len_sq_c(diff)?;
        if d2.is_zero() {
            return Some(Vec2Fix::ZERO);
        }
        let d = d2.sqrt();
        let n = div_c(diff, d)?;
        let overlap = checked_sub(p.radius_m, d)?;
        let mut f = scale_c(n, social_magnitude_c(&self.wall, overlap)?)?;
        if overlap > Fix128::ZERO {
            let t = perpendicular_c(n)?;
            let normal = scale_c(n, self.wall.body_stiffness.checked_mul(overlap)?)?;
            let tangential = scale_c(
                t,
                self.wall
                    .sliding_friction
                    .checked_mul(overlap)?
                    .checked_mul(dot_c(p.velocity, t)?)?,
            )?;
            f = sub_c(add_c(f, normal)?, tangential)?;
        }
        Some(f)
    }
}

// Range-checked `Vec2Fix` operations: the same values as the operators where
// they return `Some`.

fn add_c(a: Vec2Fix, b: Vec2Fix) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(a.x.checked_add(b.x)?, a.y.checked_add(b.y)?))
}

fn sub_c(a: Vec2Fix, b: Vec2Fix) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(checked_sub(a.x, b.x)?, checked_sub(a.y, b.y)?))
}

fn scale_c(a: Vec2Fix, s: Fix128) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(a.x.checked_mul(s)?, a.y.checked_mul(s)?))
}

fn div_c(a: Vec2Fix, s: Fix128) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(
        checked_quotient(a.x, s)?,
        checked_quotient(a.y, s)?,
    ))
}

/// `−x` as `Neg` (`0 − x`), `None` for `x = −2⁶³`.
fn neg_c(x: Fix128) -> Option<Fix128> {
    checked_sub(Fix128::ZERO, x)
}

fn neg2_c(a: Vec2Fix) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(neg_c(a.x)?, neg_c(a.y)?))
}

fn perpendicular_c(a: Vec2Fix) -> Option<Vec2Fix> {
    Some(Vec2Fix::new(neg_c(a.y)?, a.x))
}

fn dot_c(a: Vec2Fix, b: Vec2Fix) -> Option<Fix128> {
    a.x.checked_mul(b.x)?.checked_add(a.y.checked_mul(b.y)?)
}

fn len_sq_c(a: Vec2Fix) -> Option<Fix128> {
    dot_c(a, a)
}

/// [`Vec2Fix::normalize`] range-checked.
fn normalize_c(a: Vec2Fix) -> Option<Vec2Fix> {
    let len = len_sq_c(a)?.sqrt();
    if len.is_zero() {
        Some(Vec2Fix::ZERO)
    } else {
        div_c(a, len)
    }
}

/// [`social_magnitude`] with the quotient `x / B` range-checked (the
/// saturation of the product is the same).
fn social_magnitude_c(p: &InteractionParams, x: Fix128) -> Option<Fix128> {
    let e = checked_quotient(x, p.range_m)?.exp();
    Some(p.strength_n.checked_mul(e).unwrap_or(FIX_MAX))
}

/// [`driving`] range-checked.
fn driving_c(p: &Pedestrian, e_hat: Vec2Fix) -> Option<Vec2Fix> {
    let rate = checked_quotient(p.mass_kg, p.relaxation_time_s)?;
    scale_c(
        sub_c(scale_c(e_hat, p.desired_speed_m_s)?, p.velocity)?,
        rate,
    )
}

/// [`nearest_point`] range-checked.
fn nearest_point_c(p: &Pedestrian, w: &WallSegment) -> Option<Vec2Fix> {
    let ab = sub_c(w.end, w.start)?;
    let len2 = len_sq_c(ab)?;
    let s = if len2.is_zero() {
        Fix128::ZERO
    } else {
        checked_quotient(dot_c(sub_c(p.position, w.start)?, ab)?, len2)?
            .clamp(Fix128::ZERO, Fix128::ONE)
    };
    add_c(w.start, scale_c(ab, s)?)
}

/// The velocity and position update of [`SocialForce::step`] for one
/// pedestrian, range-checked.
fn advance_checked(
    p: &Pedestrian,
    f: Vec2Fix,
    h: Fix128,
    speed_cap_ratio: Option<Fix128>,
) -> Option<Pedestrian> {
    let mut v = add_c(p.velocity, scale_c(f, checked_quotient(h, p.mass_kg)?)?)?;
    if let Some(c) = speed_cap_ratio {
        let v_max = c.checked_mul(p.desired_speed_m_s)?;
        let speed = len_sq_c(v)?.sqrt();
        if speed > v_max {
            v = if speed.is_zero() {
                Vec2Fix::ZERO
            } else {
                scale_c(v, checked_quotient(v_max, speed)?)?
            };
        }
    }
    let mut next = *p;
    next.velocity = v;
    next.position = add_c(p.position, scale_c(v, h)?)?;
    Some(next)
}

/// Nearest point of segment `w` to the centre of `p`.
fn nearest_point(p: &Pedestrian, w: &WallSegment) -> Vec2Fix {
    let ab = w.end - w.start;
    let len2 = ab.length_squared();
    let s = if len2.is_zero() {
        Fix128::ZERO
    } else {
        ((p.position - w.start).dot(ab) / len2).clamp(Fix128::ZERO, Fix128::ONE)
    };
    w.start + ab * s
}

/// `d_iW²`, the squared distance from the centre of `p` to segment `w`.
fn wall_distance_squared(p: &Pedestrian, w: &WallSegment) -> Fix128 {
    (p.position - nearest_point(p, w)).length_squared()
}

/// `m (v0 ê − v)/τ` for a validated pedestrian and normalised `ê`.
fn driving(p: &Pedestrian, e_hat: Vec2Fix) -> Vec2Fix {
    (e_hat * p.desired_speed_m_s - p.velocity) * (p.mass_kg / p.relaxation_time_s)
}

/// Pairs `(i, j)`, `i < j`, in adjacent cells of width `cutoff (1 + 2⁻²⁰)`,
/// sorted ascending. Every pair with `d_ij ≤ cutoff` is included.
fn cell_list_pairs(peds: &[Pedestrian], cutoff: Fix128) -> Vec<(usize, usize)> {
    let cell = cutoff + cutoff.shr_bits(20);
    let inv = Fix128::ONE / cell;
    let keys: Vec<(i64, i64)> = peds
        .iter()
        .map(|p| ((p.position.y * inv).hi, (p.position.x * inv).hi))
        .collect();
    pairs_in_adjacent_cells(&keys)
}

/// [`cell_list_pairs`] with the cell keys range-checked.
fn cell_list_pairs_checked(peds: &[Pedestrian], cutoff: Fix128) -> Option<Vec<(usize, usize)>> {
    let cell = cutoff.checked_add(cutoff.shr_bits(20))?;
    let inv = checked_quotient(Fix128::ONE, cell)?;
    let keys = peds
        .iter()
        .map(|p| {
            Some((
                p.position.y.checked_mul(inv)?.hi,
                p.position.x.checked_mul(inv)?.hi,
            ))
        })
        .collect::<Option<Vec<(i64, i64)>>>()?;
    Some(pairs_in_adjacent_cells(&keys))
}

/// Pairs `(i, j)`, `i < j`, whose cell keys `(y, x)` differ by at most 1 on
/// each axis, sorted ascending.
fn pairs_in_adjacent_cells(keys: &[(i64, i64)]) -> Vec<(usize, usize)> {
    let mut sorted: Vec<((i64, i64), usize)> =
        keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    sorted.sort_unstable();
    let mut pairs = Vec::new();
    for (i, &(cy, cx)) in keys.iter().enumerate() {
        for dy in -1i64..=1 {
            let Some(ny) = cy.checked_add(dy) else {
                continue;
            };
            for dx in -1i64..=1 {
                let Some(nx) = cx.checked_add(dx) else {
                    continue;
                };
                let target = (ny, nx);
                let start = sorted.partition_point(|&(k, _)| k < target);
                for &(k, j) in &sorted[start..] {
                    if k != target {
                        break;
                    }
                    if j > i {
                        pairs.push((i, j));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

// ---------------------------------------------------------------------------
// World participant
// ---------------------------------------------------------------------------

/// Snapshot tag of [`CrowdParticipant`]: the ASCII code `CRWD`, big endian.
pub const CROWD_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"CRWD"));

/// Observation channel of [`CrowdParticipant`]: the number of pedestrians.
pub const CROWD_OBS_COUNT: u32 = 0;

/// Observation channel of [`CrowdParticipant`]: the mean speed `Σ|v_i| / N`
/// (m/s), not reported for an empty crowd.
pub const CROWD_OBS_MEAN_SPEED: u32 = 1;

/// Payload layout version written by [`CrowdParticipant`].
const CROWD_STATE_VERSION: u32 = 1;
/// `version: u32`, `digest: u64`, `count: u64`.
const CROWD_HEADER_LEN: usize = 4 + 8 + 8;
/// Ten `Fix128` per pedestrian.
const CROWD_PEDESTRIAN_LEN: usize = 10 * 16;

/// A crowd as a participant of the world's substep loop: it owns the
/// pedestrians, the walls and the [`SocialForce`] model and advances them by
/// one [`SocialForce::try_step`] of the world substep width `h` per substep
/// ([`StepRule::FollowSubstep`](crate::world_participant::StepRule::FollowSubstep)).
///
/// # Time step
///
/// The step is semi-implicit Euler with the stiff contact terms of the
/// module doc, so `h` of a few milliseconds is needed for the representative
/// parameters; the world's substep width is that `h`. A non-positive `h`
/// never reaches the participant: the world checks the step rule of every
/// participant before any of them runs.
///
/// # Rigid bodies
///
/// The crowd does not interact with the world's rigid bodies: it neither
/// reads them nor stages forces on them. The model has no circular obstacle
/// (a degenerate wall segment is a point whose force uses the pedestrian
/// radius alone, not a body radius), and the plane is not tied to a world
/// axis pair. Route choice stays with the caller, who sets the desired
/// direction and speed through [`Self::pedestrians_mut`] between steps.
///
/// # Faults
///
/// [`SocialForce::try_step`] refuses an invalid pedestrian (`τ ≤ 0`,
/// `m ≤ 0`, `r < 0`, `v0 < 0`, for instance set through
/// [`Self::pedestrians_mut`]) and leaves the crowd unchanged; the participant
/// then returns [`ParticipantFault::InvalidState`] and is unchanged. A force,
/// velocity or position of the step that leaves the `Fix128` range
/// ([`CrowdStepError::Overflow`]) is [`ParticipantFault::OutOfRange`], and the
/// participant is unchanged as well.
///
/// # Snapshot payload
///
/// Little endian: `version: u32` (1), `digest: u64`, `count: u64`, then per
/// pedestrian position, velocity, radius, mass, desired speed, desired
/// direction and relaxation time (`Fix128` as `hi: i64`, `lo: u64`). The
/// digest is FNV-1a 64 over the configuration (both interaction parameter
/// sets, `λ`, the cutoff, every wall, the neighbour search and the speed cap);
/// [`Participant::check_state`] refuses a payload of another configuration
/// ([`StateError::InvalidValue`]) or of a length that does not match its count
/// ([`StateError::Length`]). The number of pedestrians and every pedestrian
/// field are state and may differ from the current crowd.
#[derive(Debug, Clone)]
pub struct CrowdParticipant {
    model: SocialForce,
    pedestrians: Vec<Pedestrian>,
    walls: Vec<WallSegment>,
    search: NeighborSearch,
    speed_cap_ratio: Option<Fix128>,
    digest: u64,
}

impl CrowdParticipant {
    /// A participant owning `pedestrians`, `walls` and `model`, stepping
    /// with `search` and the optional speed cap `|v| ≤ c · v0`.
    ///
    /// # Errors
    ///
    /// [`CrowdForceError::NonPositiveSpeedCap`] or an invalid pedestrian
    /// (the errors of [`SocialForce::step`]).
    pub fn new(
        model: SocialForce,
        pedestrians: Vec<Pedestrian>,
        walls: Vec<WallSegment>,
        search: NeighborSearch,
        speed_cap_ratio: Option<Fix128>,
    ) -> Result<Self, CrowdForceError> {
        if let Some(c) = speed_cap_ratio {
            if c <= Fix128::ZERO {
                return Err(CrowdForceError::NonPositiveSpeedCap);
            }
        }
        for (i, p) in pedestrians.iter().enumerate() {
            check_pedestrian(p, i)?;
        }
        let digest = crowd_digest(&model, &walls, search, speed_cap_ratio);
        Ok(Self {
            model,
            pedestrians,
            walls,
            search,
            speed_cap_ratio,
            digest,
        })
    }

    /// The pedestrians.
    #[must_use]
    pub fn pedestrians(&self) -> &[Pedestrian] {
        &self.pedestrians
    }

    /// The pedestrians, for the caller's route choice between steps.
    pub fn pedestrians_mut(&mut self) -> &mut [Pedestrian] {
        &mut self.pedestrians
    }

    /// The walls.
    #[must_use]
    pub fn walls(&self) -> &[WallSegment] {
        &self.walls
    }

    /// The model.
    #[must_use]
    pub const fn model(&self) -> &SocialForce {
        &self.model
    }
}

impl Participant for CrowdParticipant {
    fn kind(&self) -> ParticipantKind {
        CROWD_PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        self.model
            .try_step(
                &mut self.pedestrians,
                &self.walls,
                h,
                self.search,
                self.speed_cap_ratio,
            )
            .map_err(|e| match e {
                CrowdStepError::Overflow => ParticipantFault::OutOfRange,
                _ => ParticipantFault::InvalidState,
            })
    }

    fn observe(&self, out: &mut ObservationSink) {
        let n = self.pedestrians.len();
        out.push(CROWD_OBS_COUNT, Fix128::from_int(n as i64));
        if n > 0 {
            let total = self
                .pedestrians
                .iter()
                .fold(Fix128::ZERO, |acc, p| acc + p.velocity.length());
            out.push(CROWD_OBS_MEAN_SPEED, total / Fix128::from_int(n as i64));
        }
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&CROWD_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&(self.pedestrians.len() as u64).to_le_bytes());
        for p in &self.pedestrians {
            for f in pedestrian_fields(p) {
                put_fix(out, f);
            }
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() < CROWD_HEADER_LEN {
            return Err(StateError::Length {
                expected: CROWD_HEADER_LEN,
                found: bytes.len(),
            });
        }
        if read_u32(bytes, 0) != CROWD_STATE_VERSION || read_u64(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        let count = read_u64(bytes, 12);
        let expected = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(CROWD_PEDESTRIAN_LEN))
            .and_then(|b| b.checked_add(CROWD_HEADER_LEN))
            .unwrap_or(usize::MAX);
        if bytes.len() != expected {
            return Err(StateError::Length {
                expected,
                found: bytes.len(),
            });
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.pedestrians = bytes[CROWD_HEADER_LEN..]
            .chunks_exact(CROWD_PEDESTRIAN_LEN)
            .map(|c| {
                let f = |k: usize| get_fix(c, 16 * k);
                Pedestrian {
                    position: Vec2Fix::new(f(0), f(1)),
                    velocity: Vec2Fix::new(f(2), f(3)),
                    radius_m: f(4),
                    mass_kg: f(5),
                    desired_speed_m_s: f(6),
                    desired_direction: Vec2Fix::new(f(7), f(8)),
                    relaxation_time_s: f(9),
                }
            })
            .collect();
    }
}

fn pedestrian_fields(p: &Pedestrian) -> [Fix128; 10] {
    [
        p.position.x,
        p.position.y,
        p.velocity.x,
        p.velocity.y,
        p.radius_m,
        p.mass_kg,
        p.desired_speed_m_s,
        p.desired_direction.x,
        p.desired_direction.y,
        p.relaxation_time_s,
    ]
}

fn crowd_digest(
    model: &SocialForce,
    walls: &[WallSegment],
    search: NeighborSearch,
    speed_cap_ratio: Option<Fix128>,
) -> u64 {
    let mut b = Vec::new();
    for p in [&model.pedestrian, &model.wall] {
        for f in [
            p.strength_n,
            p.range_m,
            p.body_stiffness,
            p.sliding_friction,
        ] {
            put_fix(&mut b, f);
        }
    }
    put_fix(&mut b, model.anisotropy);
    put_fix(&mut b, model.cutoff_m);
    b.extend_from_slice(&(walls.len() as u64).to_le_bytes());
    for w in walls {
        for f in [w.start.x, w.start.y, w.end.x, w.end.y] {
            put_fix(&mut b, f);
        }
    }
    b.push(match search {
        NeighborSearch::Direct => 0,
        NeighborSearch::CellList => 1,
    });
    match speed_cap_ratio {
        None => b.push(0),
        Some(c) => {
            b.push(1);
            put_fix(&mut b, c);
        }
    }
    fnv1a64(&b)
}

/// FNV-1a 64 (offset basis `0xcbf2_9ce4_8422_2325`, prime `0x100_0000_01b3`).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(a)
}

fn read_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn get_fix(b: &[u8], at: usize) -> Fix128 {
    Fix128::from_raw(read_u64(b, at) as i64, read_u64(b, at + 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_list_pairs_cover_all_pairs_within_cutoff() {
        let cutoff = Fix128::from_ratio(3, 2);
        let mk = |x: i64, y: i64| Pedestrian {
            position: Vec2Fix::new(Fix128::from_ratio(x, 4), Fix128::from_ratio(y, 4)),
            velocity: Vec2Fix::ZERO,
            radius_m: Fix128::ZERO,
            mass_kg: Fix128::ONE,
            desired_speed_m_s: Fix128::ZERO,
            desired_direction: Vec2Fix::ZERO,
            relaxation_time_s: Fix128::ONE,
        };
        let peds: Vec<Pedestrian> = (-6..6)
            .flat_map(|x| (-6..6).map(move |y| mk(3 * x + y % 2, 2 * y - x % 3)))
            .collect();
        let pairs = cell_list_pairs(&peds, cutoff);
        let c2 = cutoff * cutoff;
        for i in 0..peds.len() {
            for j in i + 1..peds.len() {
                let near = (peds[i].position - peds[j].position).length_squared() <= c2;
                if near {
                    assert!(pairs.binary_search(&(i, j)).is_ok(), "missing ({i}, {j})");
                }
            }
        }
        let mut dedup = pairs.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), pairs.len(), "no duplicate pairs");
    }

    fn params(strength: i64, range: i64, k: i64, kappa: i64) -> InteractionParams {
        InteractionParams {
            strength_n: Fix128::from_int(strength),
            range_m: Fix128::from_int(range),
            body_stiffness: Fix128::from_int(k),
            sliding_friction: Fix128::from_int(kappa),
        }
    }

    fn v2(x: i64, y: i64) -> Vec2Fix {
        Vec2Fix::new(Fix128::from_int(x), Fix128::from_int(y))
    }

    fn ped_at(position: Vec2Fix) -> Pedestrian {
        Pedestrian {
            position,
            velocity: Vec2Fix::ZERO,
            radius_m: Fix128::from_ratio(1, 2),
            mass_kg: Fix128::ONE,
            desired_speed_m_s: Fix128::ZERO,
            desired_direction: Vec2Fix::ZERO,
            relaxation_time_s: Fix128::ONE,
        }
    }

    fn model(ped: InteractionParams, wall: InteractionParams, anisotropy: Fix128) -> SocialForce {
        SocialForce::new(ped, wall, anisotropy, Fix128::from_int(5)).expect("valid model")
    }

    /// oracle: every parameter check of `SocialForce::new` and of a
    /// pedestrian names its own error, and each error prints the message of
    /// its documentation (a step error forwards the crowd error's message).
    #[test]
    fn construction_errors_and_messages() {
        let ok = params(1, 1, 0, 0);
        let new = |p, w, a: Fix128, c: Fix128| SocialForce::new(p, w, a, c).err();
        assert_eq!(
            new(params(1, 0, 0, 0), ok, Fix128::ZERO, Fix128::ONE),
            Some(CrowdForceError::NonPositiveRange)
        );
        assert_eq!(
            new(ok, params(1, 1, -1, 0), Fix128::ZERO, Fix128::ONE),
            Some(CrowdForceError::NegativeCoefficient)
        );
        assert_eq!(
            new(ok, ok, Fix128::from_int(2), Fix128::ONE),
            Some(CrowdForceError::AnisotropyOutOfRange)
        );
        assert_eq!(
            new(ok, ok, -Fix128::ONE, Fix128::ONE),
            Some(CrowdForceError::AnisotropyOutOfRange)
        );
        assert_eq!(
            new(ok, ok, Fix128::ZERO, Fix128::ZERO),
            Some(CrowdForceError::NonPositiveCutoff)
        );
        let m = model(ok, ok, Fix128::ZERO);
        let base = ped_at(Vec2Fix::ZERO);
        let bad = [
            (
                Pedestrian {
                    relaxation_time_s: Fix128::ZERO,
                    ..base
                },
                CrowdForceError::NonPositiveRelaxationTime { index: 0 },
            ),
            (
                Pedestrian {
                    mass_kg: Fix128::ZERO,
                    ..base
                },
                CrowdForceError::NonPositiveMass { index: 0 },
            ),
            (
                Pedestrian {
                    radius_m: -Fix128::ONE,
                    ..base
                },
                CrowdForceError::NegativeRadius { index: 0 },
            ),
            (
                Pedestrian {
                    desired_speed_m_s: -Fix128::ONE,
                    ..base
                },
                CrowdForceError::NegativeDesiredSpeed { index: 0 },
            ),
        ];
        for (p, e) in bad {
            assert_eq!(m.driving_force(&p), Err(e));
        }
        let messages = [
            (
                CrowdForceError::NonPositiveRange,
                "interaction range B must be positive",
            ),
            (
                CrowdForceError::NegativeCoefficient,
                "interaction strength, stiffness and friction must be non-negative",
            ),
            (
                CrowdForceError::AnisotropyOutOfRange,
                "view-angle weight must be in [0, 1]",
            ),
            (
                CrowdForceError::NonPositiveCutoff,
                "pair cutoff must be positive",
            ),
            (
                CrowdForceError::NonPositiveRelaxationTime { index: 3 },
                "pedestrian 3: relaxation time must be positive",
            ),
            (
                CrowdForceError::NonPositiveMass { index: 3 },
                "pedestrian 3: mass must be positive",
            ),
            (
                CrowdForceError::NegativeRadius { index: 3 },
                "pedestrian 3: radius must be non-negative",
            ),
            (
                CrowdForceError::NegativeDesiredSpeed { index: 3 },
                "pedestrian 3: desired speed must be non-negative",
            ),
            (
                CrowdForceError::NonPositiveTimeStep,
                "time step must be positive",
            ),
            (
                CrowdForceError::NonPositiveSpeedCap,
                "speed-cap ratio must be positive",
            ),
        ];
        for (e, text) in messages {
            assert_eq!(e.to_string(), text);
            assert_eq!(CrowdStepError::from(e).to_string(), text);
        }
        assert_eq!(
            CrowdStepError::Overflow.to_string(),
            "a value of the step is outside the fixed-point range"
        );
    }

    /// oracle: the driving force is `m·(v₀·ê − v)/τ`: `m = 2, τ = 1/2,
    /// v₀ = 3, ê = (0, 1), v = (1, 0)` gives `(−1, 3)·4 = (−4, 12)`. The view
    /// weight is `λ + (1 − λ)(1 + cos φ)/2` with `cos φ = −n·ê`: for
    /// `λ = 1/4, ê = (1, 0)` it is 1 for `n = (−1, 0)`, 1/4 for `n = (1, 0)`
    /// and 5/8 for `n = (0, 1)`; a pedestrian with no heading weighs 1.
    #[test]
    fn driving_force_and_view_weight_closed_form() {
        let m = model(
            params(1, 1, 0, 0),
            params(1, 1, 0, 0),
            Fix128::from_ratio(1, 4),
        );
        let p = Pedestrian {
            velocity: v2(1, 0),
            mass_kg: Fix128::from_int(2),
            desired_speed_m_s: Fix128::from_int(3),
            desired_direction: v2(0, 1),
            relaxation_time_s: Fix128::from_ratio(1, 2),
            ..ped_at(Vec2Fix::ZERO)
        };
        assert_eq!(m.driving_force(&p), Ok(v2(-4, 12)));
        let walker = Pedestrian {
            desired_direction: v2(1, 0),
            ..ped_at(Vec2Fix::ZERO)
        };
        assert_eq!(m.anisotropy_weight(&walker, v2(-1, 0)), Fix128::ONE);
        assert_eq!(
            m.anisotropy_weight(&walker, v2(1, 0)),
            Fix128::from_ratio(1, 4)
        );
        assert_eq!(
            m.anisotropy_weight(&walker, v2(0, 1)),
            Fix128::from_ratio(5, 8)
        );
        assert_eq!(
            m.anisotropy_weight(&ped_at(Vec2Fix::ZERO), v2(1, 0)),
            Fix128::ONE
        );
    }

    /// oracle: two pedestrians of radius 1/2 at distance 2 (no overlap,
    /// `r − d = −1`) repel with `A·exp((r − d)/B) = 2/e` along the line
    /// between them, equal and opposite; with strength 0 and centres 1/2
    /// apart (overlap 1/2), the contact force on A is `k·(1/2)·n +
    /// κ·(1/2)·Δvₜ·t` with `n = (−1, 0)`, `t = n⊥ = (0, −1)`, `Δvₜ =
    /// (v_B − v_A)·t = −2`: `(−2, 0) + (0, 3) = (−2, 3)` for `k = 4, κ = 3`.
    /// Coincident centres give no force.
    #[test]
    fn pair_forces_closed_form() {
        let social = model(params(2, 1, 0, 0), params(1, 1, 0, 0), Fix128::ONE);
        let a = ped_at(Vec2Fix::ZERO);
        let b = ped_at(v2(2, 0));
        let (fa, fb) = social.pair_forces(&a, &b);
        let want = 2.0 / core::f64::consts::E;
        // the fixed-point exp is accurate to about 1e-8
        assert!((fa.x.to_f64() + want).abs() < 1e-7, "{fa:?}");
        assert!(fa.y.is_zero());
        assert_eq!(fb, -fa);

        let contact = model(params(0, 1, 4, 3), params(1, 1, 0, 0), Fix128::ONE);
        let b = Pedestrian {
            velocity: v2(0, 2),
            ..ped_at(Vec2Fix::new(Fix128::from_ratio(1, 2), Fix128::ZERO))
        };
        let (fa, fb) = contact.pair_forces(&a, &b);
        assert_eq!(fa, v2(-2, 3));
        assert_eq!(fb, v2(2, -3));
        assert_eq!(contact.pair_forces(&a, &a), (Vec2Fix::ZERO, Vec2Fix::ZERO));
    }

    /// oracle: a pedestrian of radius 1/2 at `(0, 1)` above the wall
    /// `(−1, 0)–(1, 0)` (distance 1, `r − d = −1/2`) is pushed along `+y` by
    /// `A·exp(−1/2)` (`A = 2, B = 1`); at `(0, 1/4)` (overlap 1/4) moving at
    /// `(2, 0)` with strength 0, `k = 4, κ = 3`, the force is
    /// `n·k·(1/4) − t·κ·(1/4)·(v·t) = (0, 1) − (−1, 0)·(−3/2) = (−3/2, 1)`.
    /// A wall of zero length acts from its start point; a pedestrian on the
    /// wall line feels nothing.
    #[test]
    fn wall_force_closed_form() {
        let wall = WallSegment {
            start: v2(-1, 0),
            end: v2(1, 0),
        };
        let social = model(params(1, 1, 0, 0), params(2, 1, 0, 0), Fix128::ONE);
        let f = social.wall_force(&ped_at(v2(0, 1)), &wall);
        assert!(f.x.is_zero());
        assert!(
            // 2·exp(−1/2)
            (f.y.to_f64() - 1.213_061_319_425_266_8).abs() < 1e-7,
            "{f:?}"
        );

        let contact = model(params(1, 1, 0, 0), params(0, 1, 4, 3), Fix128::ONE);
        let p = Pedestrian {
            velocity: v2(2, 0),
            ..ped_at(Vec2Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 4)))
        };
        assert_eq!(
            contact.wall_force(&p, &wall),
            Vec2Fix::new(Fix128::from_ratio(-3, 2), Fix128::ONE)
        );
        let point = WallSegment {
            start: v2(0, -1),
            end: v2(0, -1),
        };
        let f = social.wall_force(&ped_at(v2(0, 1)), &point);
        assert!(f.x.is_zero() && f.y > Fix128::ZERO);
        assert_eq!(social.wall_force(&ped_at(v2(0, 0)), &wall), Vec2Fix::ZERO);
    }

    /// oracle: with view weight 1 every pair force is equal and opposite and
    /// nobody is driven (`v₀ = 0, v = 0`), so the forces of a crowd sum to
    /// exactly zero while the neighbours push each other; a pedestrian
    /// beyond the cutoff 5 feels nothing. Both neighbour searches agree,
    /// and an invalid pedestrian is reported with its index.
    #[test]
    fn total_forces_cancel_in_pairs() {
        let m = model(params(2, 1, 0, 0), params(1, 1, 0, 0), Fix128::ONE);
        let peds = [
            ped_at(v2(0, 0)),
            ped_at(v2(1, 0)),
            ped_at(v2(0, 2)),
            ped_at(v2(40, 0)),
        ];
        let mut direct = Vec::new();
        m.total_forces(&peds, &[], NeighborSearch::Direct, &mut direct)
            .expect("valid crowd");
        let mut cells = Vec::new();
        m.total_forces(&peds, &[], NeighborSearch::CellList, &mut cells)
            .expect("valid crowd");
        assert_eq!(direct, cells);
        let sum = direct.iter().fold(Vec2Fix::ZERO, |s, f| s + *f);
        assert_eq!(sum, Vec2Fix::ZERO);
        assert_ne!(direct[0], Vec2Fix::ZERO);
        assert_eq!(direct[3], Vec2Fix::ZERO);
        let mut bad = peds;
        bad[2].mass_kg = Fix128::ZERO;
        assert_eq!(
            m.total_forces(&bad, &[], NeighborSearch::Direct, &mut direct),
            Err(CrowdForceError::NonPositiveMass { index: 2 })
        );
    }

    fn walker() -> Pedestrian {
        Pedestrian {
            desired_speed_m_s: Fix128::ONE,
            desired_direction: v2(1, 0),
            relaxation_time_s: Fix128::from_ratio(1, 2),
            ..ped_at(Vec2Fix::ZERO)
        }
    }

    /// oracle: a lone walker at rest (`m = 1, τ = 1/2, v₀ = 1, ê = (1, 0)`)
    /// feels `(2, 0)`; one step of `h = 1/4` (semi-implicit Euler) gives
    /// `v = (1/2, 0)` and `x = (1/8, 0)`, for `step` and `try_step` alike.
    /// A speed cap of `1/4·v₀` limits `v` to `(1/4, 0)`, `x = (1/16, 0)`.
    /// A wall beyond the cutoff changes nothing.
    #[test]
    fn step_and_try_step_closed_form() {
        let m = model(params(2, 1, 0, 0), params(2, 1, 0, 0), Fix128::ONE);
        let far_wall = [WallSegment {
            start: v2(-1, 30),
            end: v2(1, 30),
        }];
        let h = Fix128::from_ratio(1, 4);
        for cap in [None, Some(Fix128::from_ratio(1, 4))] {
            let (v, x) = if cap.is_some() {
                (Fix128::from_ratio(1, 4), Fix128::from_ratio(1, 16))
            } else {
                (Fix128::from_ratio(1, 2), Fix128::from_ratio(1, 8))
            };
            let want_v = Vec2Fix::new(v, Fix128::ZERO);
            let want_x = Vec2Fix::new(x, Fix128::ZERO);
            for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
                let mut a = [walker()];
                m.step(&mut a, &far_wall, h, search, cap).expect("step");
                assert_eq!((a[0].velocity, a[0].position), (want_v, want_x), "{cap:?}");
                let mut b = [walker()];
                m.try_step(&mut b, &far_wall, h, search, cap)
                    .expect("try_step");
                assert_eq!((b[0].velocity, b[0].position), (want_v, want_x), "{cap:?}");
            }
        }
    }

    /// oracle: `try_step` refuses a non-positive step, a non-positive cap and
    /// an invalid pedestrian, and reports `Overflow` (leaving the crowd as
    /// it was) when the new position `x + v·h` with `v = 2⁶²`, `h = 4`
    /// leaves the fixed-point range; `step` refuses the same inputs.
    #[test]
    fn try_step_refusals_leave_the_crowd_unchanged() {
        let m = model(params(2, 1, 0, 0), params(2, 1, 0, 0), Fix128::ONE);
        let mut peds = [walker()];
        let h = Fix128::from_ratio(1, 4);
        assert_eq!(
            m.try_step(&mut peds, &[], Fix128::ZERO, NeighborSearch::Direct, None),
            Err(CrowdStepError::Crowd(CrowdForceError::NonPositiveTimeStep))
        );
        assert_eq!(
            m.try_step(
                &mut peds,
                &[],
                h,
                NeighborSearch::Direct,
                Some(Fix128::ZERO)
            ),
            Err(CrowdStepError::Crowd(CrowdForceError::NonPositiveSpeedCap))
        );
        assert_eq!(
            m.step(&mut peds, &[], Fix128::ZERO, NeighborSearch::Direct, None),
            Err(CrowdForceError::NonPositiveTimeStep)
        );
        assert_eq!(
            m.step(
                &mut peds,
                &[],
                h,
                NeighborSearch::Direct,
                Some(Fix128::ZERO)
            ),
            Err(CrowdForceError::NonPositiveSpeedCap)
        );
        let mut bad = [Pedestrian {
            mass_kg: Fix128::ZERO,
            ..walker()
        }];
        assert_eq!(
            m.try_step(&mut bad, &[], h, NeighborSearch::Direct, None),
            Err(CrowdStepError::Crowd(CrowdForceError::NonPositiveMass {
                index: 0
            }))
        );
        let fast = Pedestrian {
            velocity: Vec2Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO),
            ..walker()
        };
        let mut runaway = [fast];
        assert_eq!(
            m.try_step(
                &mut runaway,
                &[],
                Fix128::from_int(4),
                NeighborSearch::Direct,
                None
            ),
            Err(CrowdStepError::Overflow)
        );
        assert_eq!(runaway, [fast]);
        assert_eq!(peds, [walker()]);
    }

    fn crowd() -> CrowdParticipant {
        let m = model(params(2, 1, 0, 0), params(2, 1, 0, 0), Fix128::ONE);
        CrowdParticipant::new(
            m,
            vec![walker()],
            vec![WallSegment {
                start: v2(-1, 30),
                end: v2(1, 30),
            }],
            NeighborSearch::Direct,
            None,
        )
        .expect("valid crowd")
    }

    /// oracle: `CrowdParticipant::new` refuses a non-positive cap and an
    /// invalid pedestrian; the accessors hand back what was given. Its kind
    /// is `"CRWD"` read big endian; the payload is version 1, the model
    /// digest, the count and 10 `Fix128` per pedestrian (20 + 160 bytes for
    /// one), read back into the same pedestrians; a short payload, another
    /// version or digest, or a count that does not match the length is
    /// refused. Observations: the count (2) and the mean speed (`|(3, 4)|/2 =
    /// 5/2`).
    #[test]
    fn crowd_participant_state_and_observations() {
        let m = model(params(2, 1, 0, 0), params(2, 1, 0, 0), Fix128::ONE);
        assert_eq!(
            CrowdParticipant::new(
                m,
                vec![],
                vec![],
                NeighborSearch::Direct,
                Some(Fix128::ZERO)
            )
            .err(),
            Some(CrowdForceError::NonPositiveSpeedCap)
        );
        let bad = Pedestrian {
            radius_m: -Fix128::ONE,
            ..walker()
        };
        assert_eq!(
            CrowdParticipant::new(
                m,
                vec![walker(), bad],
                vec![],
                NeighborSearch::CellList,
                None
            )
            .err(),
            Some(CrowdForceError::NegativeRadius { index: 1 })
        );

        let mut c = crowd();
        assert_eq!(c.pedestrians(), &[walker()]);
        assert_eq!(c.walls().len(), 1);
        assert_eq!(*c.model(), m);
        assert_eq!(Participant::kind(&c), ParticipantKind::new(0x4352_5744));
        c.pedestrians_mut()[0].velocity = v2(3, 4);
        let mut bytes = Vec::new();
        c.write_state(&mut bytes);
        assert_eq!(bytes.len(), 20 + 160);
        assert_eq!(bytes[..4], 1_u32.to_le_bytes());
        assert_eq!(bytes[12..20], 1_u64.to_le_bytes());
        assert_eq!(c.check_state(&bytes), Ok(()));
        let mut d = crowd();
        d.read_state(&bytes);
        assert_eq!(d.pedestrians(), c.pedestrians());

        assert_eq!(
            c.check_state(&bytes[..10]),
            Err(StateError::Length {
                expected: 20,
                found: 10
            })
        );
        let mut version = bytes.clone();
        version[0] = 2;
        assert_eq!(c.check_state(&version), Err(StateError::InvalidValue));
        let mut digest = bytes.clone();
        digest[4] ^= 1;
        assert_eq!(c.check_state(&digest), Err(StateError::InvalidValue));
        assert_eq!(
            c.check_state(&bytes[..100]),
            Err(StateError::Length {
                expected: 180,
                found: 100
            })
        );
        let mut huge = bytes.clone();
        huge[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            c.check_state(&huge),
            Err(StateError::Length { .. })
        ));
        // a different cap gives a different model digest
        let capped = CrowdParticipant::new(
            m,
            vec![walker()],
            c.walls().to_vec(),
            NeighborSearch::Direct,
            Some(Fix128::ONE),
        )
        .expect("valid crowd");
        assert_eq!(capped.check_state(&bytes), Err(StateError::InvalidValue));

        c.pedestrians_mut()[0].velocity = v2(3, 4);
        let mut two = crowd();
        two.pedestrians = vec![c.pedestrians()[0], walker()];
        let mut sink = ObservationSink::new();
        two.observe(&mut sink);
        assert_eq!(
            sink.values(),
            &[
                (CROWD_OBS_COUNT, Fix128::from_int(2)),
                (CROWD_OBS_MEAN_SPEED, Fix128::from_ratio(5, 2))
            ]
        );
        two.pedestrians.clear();
        let mut sink = ObservationSink::new();
        two.observe(&mut sink);
        assert_eq!(sink.values(), &[(CROWD_OBS_COUNT, Fix128::ZERO)]);
    }

    /// oracle: in a world of one substep `h = 1/4` the lone walker reaches
    /// `v = (1/2, 0)`, so the mean speed reads 1/2; a walker whose step
    /// leaves the fixed-point range records an out-of-range participant
    /// fault.
    #[test]
    fn crowd_participant_in_a_world() {
        let world_of = |c: CrowdParticipant| {
            let mut world = crate::solver::PhysicsWorld::new(crate::solver::SolverConfig {
                substeps: 1,
                ..Default::default()
            });
            world.add_participant(Box::new(c)).expect("register");
            world.step(Fix128::from_ratio(1, 4));
            world
        };
        let world = world_of(crowd());
        let Some(crate::world_participant::Observed::Exact(sink)) = world.observe_participant(0)
        else {
            panic!("observation");
        };
        assert_eq!(
            sink.values(),
            &[
                (CROWD_OBS_COUNT, Fix128::ONE),
                (CROWD_OBS_MEAN_SPEED, Fix128::from_ratio(1, 2))
            ]
        );

        let mut runaway = crowd();
        runaway.pedestrians_mut()[0].position =
            Vec2Fix::new(Fix128::from_int(i64::MAX - 1), Fix128::ZERO);
        runaway.pedestrians_mut()[0].velocity =
            Vec2Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO);
        let world = world_of(runaway);
        assert!(
            matches!(
                world.fault(),
                Some(crate::world_participant::WorldFault::Participant {
                    fault: ParticipantFault::OutOfRange,
                    ..
                })
            ),
            "{:?}",
            world.fault()
        );
    }
}
