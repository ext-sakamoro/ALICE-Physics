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
    let key = |p: &Pedestrian| ((p.position.y * inv).hi, (p.position.x * inv).hi);
    let mut sorted: Vec<((i64, i64), usize)> =
        peds.iter().enumerate().map(|(i, p)| (key(p), i)).collect();
    sorted.sort_unstable();
    let mut pairs = Vec::new();
    for (i, p) in peds.iter().enumerate() {
        let (cy, cx) = key(p);
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
/// one [`SocialForce::step`] of the world substep width `h` per substep
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
/// [`SocialForce::step`] refuses an invalid pedestrian (`τ ≤ 0`, `m ≤ 0`,
/// `r < 0`, `v0 < 0`, for instance set through [`Self::pedestrians_mut`])
/// and leaves the crowd unchanged; the participant then returns
/// [`ParticipantFault::InvalidState`] and is unchanged.
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
            .step(
                &mut self.pedestrians,
                &self.walls,
                h,
                self.search,
                self.speed_cap_ratio,
            )
            .map_err(|_| ParticipantFault::InvalidState)
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
}
