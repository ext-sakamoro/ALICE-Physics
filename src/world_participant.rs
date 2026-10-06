//! The contract a law signs to take part in [`crate::solver::PhysicsWorld`]'s
//! substep loop: [`Participant`] and the small types around it.
//!
//! A participant owns its own state (a particle set, a grid, a pedestrian
//! crowd, a force field) and advances it once per world substep. It reads the
//! rigid bodies as they were at the start of the substep and hands its effect
//! on them back as forces and torques in a [`ForceAccumulator`]; it never
//! writes a body's position or velocity. Snapshots carry its state as an
//! opaque payload tagged with its [`ParticipantKind`].
//!
//! # Contract
//!
//! | rule | how it is enforced |
//! |---|---|
//! | participants run in registration order, once per substep (before the rigid integration of that substep) | world side |
//! | [`Participant::substep`] returning `Err` leaves the participant unchanged | participant side, checked by the conformance tests |
//! | forces a participant staged in a substep that returned `Err` never reach the bodies | world side (staged per participant, merged on `Ok`) |
//! | a participant cannot write a body | type: [`SubstepCtx`] only hands out `&[RigidBody]` and the accumulator |
//! | [`Participant::check_state`] accepts every payload [`Participant::write_state`] produced, and [`Participant::read_state`] then restores the participant bit for bit | participant side, checked by the conformance tests |
//! | a restore checks the target world holds the same number of participants, of the same kinds, in the same order | world side, [`ParticipantMismatch::classify`] |
//!
//! A participant cannot write a body because [`SubstepCtx`] has no way to
//! reach one mutably:
//!
//! ```compile_fail
//! use alice_physics::world_participant::{ForceAccumulator, SubstepCtx};
//! use alice_physics::{Fix128, RigidBody, Vec3Fix};
//!
//! let bodies = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
//! let mut forces = ForceAccumulator::new(1);
//! let ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, Fix128::ONE).unwrap();
//! ctx.bodies()[0].velocity = Vec3Fix::ZERO; // `bodies()` is `&[RigidBody]`
//! ```

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Stable tag of a participant type, written in front of its snapshot
/// payload. A type keeps its tag across versions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParticipantKind(u32);

impl ParticipantKind {
    /// The tag `raw`.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw tag.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// How a participant's own time step relates to the world substep `h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepRule {
    /// One call per world substep, advancing by `h`.
    FollowSubstep,
    /// The participant advances by a fixed `Δt` of its own; registration is
    /// refused unless `h = n·Δt` for an integer `n ≥ 1` (exactly, in
    /// [`Fix128`]). The world still calls it once per substep with `h`; the
    /// participant takes `n` internal steps.
    Fixed(Fix128),
    /// One call per world substep with `h`; the participant splits `h`
    /// itself, deterministically from its own state (e.g. a CFL limit).
    Subcycle,
}

impl StepRule {
    /// Internal steps per world substep of width `h`, or why this rule cannot
    /// follow `h`. `FollowSubstep` and `Subcycle` give 1 (the participant is
    /// called once); `Fixed(Δt)` gives `n` with `h = n·Δt`.
    ///
    /// # Errors
    ///
    /// [`RegisterError::NonPositiveStep`] when `h ≤ 0` or `Δt ≤ 0`;
    /// [`RegisterError::StepRuleMismatch`] when `h` is not an integer multiple
    /// of `Δt`.
    pub fn steps_per_substep(self, h: Fix128) -> Result<u64, RegisterError> {
        if h <= Fix128::ZERO {
            return Err(RegisterError::NonPositiveStep);
        }
        match self {
            Self::FollowSubstep | Self::Subcycle => Ok(1),
            Self::Fixed(step) => {
                if step <= Fix128::ZERO {
                    return Err(RegisterError::NonPositiveStep);
                }
                let n = h / step;
                let whole = n.lo == 0 && n.hi >= 1;
                if whole && Fix128::from_int(n.hi) * step == h {
                    Ok(n.hi.unsigned_abs())
                } else {
                    Err(RegisterError::StepRuleMismatch { substep: h, step })
                }
            }
        }
    }
}

/// Why a participant could not be registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegisterError {
    /// The world substep or the participant's own step is not positive.
    NonPositiveStep,
    /// `StepRule::Fixed(step)` does not divide the world substep.
    StepRuleMismatch {
        /// World substep `h`.
        substep: Fix128,
        /// The participant's fixed step.
        step: Fix128,
    },
}

/// Why a participant's substep failed. The participant is unchanged and its
/// forces of that substep are dropped; the world records the fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParticipantFault {
    /// A value left the representable or valid range.
    OutOfRange,
    /// The participant found its own state inconsistent.
    InvalidState,
}

impl ParticipantFault {
    /// Snapshot tag of this fault.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::OutOfRange => 1,
            Self::InvalidState => 2,
        }
    }

    /// The fault with snapshot tag `tag`, if any.
    #[must_use]
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::OutOfRange),
            2 => Some(Self::InvalidState),
            _ => None,
        }
    }
}

/// The first fault the world recorded (sticky until cleared).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldFault {
    /// Registration index of the participant that failed.
    pub index: usize,
    /// Its kind.
    pub kind: ParticipantKind,
    /// What failed.
    pub fault: ParticipantFault,
}

/// Why a checked world step did not run normally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepError {
    /// A fault was already recorded; the world was not touched.
    Faulted(WorldFault),
    /// A participant failed during this step. The step ran to the end (the
    /// failed participant was frozen at its state before the failing call and
    /// its forces of that substep were dropped); the fault is now recorded.
    FaultRaised(WorldFault),
    /// The participant at `index` has a [`StepRule::Fixed`] step that does not
    /// divide this frame's substep `dt / substeps`; nothing ran. The step is
    /// only known when the world steps, so the check runs at the start of
    /// every step rather than at registration.
    Rule {
        /// Registration index of the participant.
        index: usize,
        /// Why its rule cannot follow the substep.
        error: RegisterError,
    },
}

/// Why a participant rejected a snapshot payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StateError {
    /// The payload has a different length than the participant needs.
    Length {
        /// Bytes needed.
        expected: usize,
        /// Bytes given.
        found: usize,
    },
    /// A value in the payload is not valid for the participant.
    InvalidValue,
}

/// Why a restore refused the participants of the target world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParticipantMismatch {
    /// Different numbers of participants.
    Count {
        /// Participants in the snapshot.
        snapshot: usize,
        /// Participants in the target world.
        world: usize,
    },
    /// The same number, but not the same kinds (as multisets).
    Kind {
        /// First index whose kind differs.
        index: usize,
        /// Kind in the snapshot.
        snapshot: ParticipantKind,
        /// Kind in the target world.
        world: ParticipantKind,
    },
    /// The same kinds, in another order.
    Order {
        /// First index whose kind differs.
        index: usize,
        /// Kind in the snapshot.
        snapshot: ParticipantKind,
        /// Kind in the target world.
        world: ParticipantKind,
    },
}

impl ParticipantMismatch {
    /// Compare the kinds stored in a snapshot with those registered in the
    /// target world. `None` when they agree (same count, same kinds, same
    /// order).
    #[must_use]
    pub fn classify(snapshot: &[ParticipantKind], world: &[ParticipantKind]) -> Option<Self> {
        if snapshot.len() != world.len() {
            return Some(Self::Count {
                snapshot: snapshot.len(),
                world: world.len(),
            });
        }
        let index = snapshot.iter().zip(world).position(|(s, w)| s != w)?;
        let (s, w) = (snapshot[index], world[index]);
        let mut a: Vec<ParticipantKind> = snapshot.to_vec();
        let mut b: Vec<ParticipantKind> = world.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        Some(if a == b {
            Self::Order {
                index,
                snapshot: s,
                world: w,
            }
        } else {
            Self::Kind {
                index,
                snapshot: s,
                world: w,
            }
        })
    }
}

/// Why a force could not be staged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccumulateError {
    /// The body index is past the end of the body list.
    BodyOutOfRange {
        /// The index.
        index: usize,
        /// Number of bodies.
        len: usize,
    },
    /// The accumulator does not have one slot per body.
    LengthMismatch {
        /// Slots in the accumulator.
        slots: usize,
        /// Bodies.
        bodies: usize,
    },
}

/// Per-body sums of force and torque (world frame, [`Fix128`]) for one
/// substep.
///
/// The sums use the wrapping addition of [`Fix128`], which is associative and
/// commutative, so the totals do not depend on the order the forces were
/// added in. The world still fixes the order (registration order, then each
/// participant's own order) so that any later checked accumulation stays
/// deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForceAccumulator {
    force: Vec<Vec3Fix>,
    torque: Vec<Vec3Fix>,
}

impl ForceAccumulator {
    /// Zero force and torque for `bodies` bodies.
    #[must_use]
    pub fn new(bodies: usize) -> Self {
        Self {
            force: vec![Vec3Fix::ZERO; bodies],
            torque: vec![Vec3Fix::ZERO; bodies],
        }
    }

    /// Number of body slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.force.len()
    }

    /// Whether there are no body slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.force.is_empty()
    }

    fn slot(&self, body: usize) -> Result<usize, AccumulateError> {
        if body < self.force.len() {
            Ok(body)
        } else {
            Err(AccumulateError::BodyOutOfRange {
                index: body,
                len: self.force.len(),
            })
        }
    }

    /// Add `force` to body `body`.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::BodyOutOfRange`] when `body` has no slot (nothing
    /// is added).
    pub fn add_force(&mut self, body: usize, force: Vec3Fix) -> Result<(), AccumulateError> {
        let i = self.slot(body)?;
        self.force[i] = self.force[i] + force;
        Ok(())
    }

    /// Add `torque` to body `body`.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::BodyOutOfRange`] when `body` has no slot (nothing
    /// is added).
    pub fn add_torque(&mut self, body: usize, torque: Vec3Fix) -> Result<(), AccumulateError> {
        let i = self.slot(body)?;
        self.torque[i] = self.torque[i] + torque;
        Ok(())
    }

    /// Summed force on `body` (`None` past the end).
    #[must_use]
    pub fn force(&self, body: usize) -> Option<Vec3Fix> {
        self.force.get(body).copied()
    }

    /// Summed torque on `body` (`None` past the end).
    #[must_use]
    pub fn torque(&self, body: usize) -> Option<Vec3Fix> {
        self.torque.get(body).copied()
    }

    /// Add every slot of `other` into `self`, slot by slot.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::LengthMismatch`] when the slot counts differ
    /// (`self` is unchanged).
    pub fn merge(&mut self, other: &Self) -> Result<(), AccumulateError> {
        if other.len() != self.len() {
            return Err(AccumulateError::LengthMismatch {
                slots: other.len(),
                bodies: self.len(),
            });
        }
        for i in 0..self.len() {
            self.force[i] = self.force[i] + other.force[i];
            self.torque[i] = self.torque[i] + other.torque[i];
        }
        Ok(())
    }

    /// Reset every slot to zero, keeping the slot count.
    pub fn clear(&mut self) {
        self.force.fill(Vec3Fix::ZERO);
        self.torque.fill(Vec3Fix::ZERO);
    }
}

/// What a participant sees during one substep: the bodies as they were at the
/// start of the substep (read only) and an accumulator for its forces.
#[derive(Debug)]
pub struct SubstepCtx<'a> {
    bodies: &'a [RigidBody],
    forces: &'a mut ForceAccumulator,
    substep_index: usize,
    substeps: usize,
    h: Fix128,
}

impl<'a> SubstepCtx<'a> {
    /// A context over `bodies`, staging into `forces` (one slot per body),
    /// for substep `substep_index` of `substeps` with width `h`.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::LengthMismatch`] when `forces` does not have one
    /// slot per body.
    pub fn new(
        bodies: &'a [RigidBody],
        forces: &'a mut ForceAccumulator,
        substep_index: usize,
        substeps: usize,
        h: Fix128,
    ) -> Result<Self, AccumulateError> {
        if forces.len() != bodies.len() {
            return Err(AccumulateError::LengthMismatch {
                slots: forces.len(),
                bodies: bodies.len(),
            });
        }
        Ok(Self {
            bodies,
            forces,
            substep_index,
            substeps,
            h,
        })
    }

    /// The bodies at the start of this substep.
    #[must_use]
    pub fn bodies(&self) -> &[RigidBody] {
        self.bodies
    }

    /// Index of this substep within the frame (`0..substeps`).
    #[must_use]
    pub fn substep_index(&self) -> usize {
        self.substep_index
    }

    /// Substeps in the frame.
    #[must_use]
    pub fn substeps(&self) -> usize {
        self.substeps
    }

    /// Width of this substep.
    #[must_use]
    pub fn h(&self) -> Fix128 {
        self.h
    }

    /// Stage `force` on body `body` for this substep.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::BodyOutOfRange`] for a body that does not exist.
    pub fn add_force(&mut self, body: usize, force: Vec3Fix) -> Result<(), AccumulateError> {
        self.forces.add_force(body, force)
    }

    /// Stage `torque` on body `body` for this substep.
    ///
    /// # Errors
    ///
    /// [`AccumulateError::BodyOutOfRange`] for a body that does not exist.
    pub fn add_torque(&mut self, body: usize, torque: Vec3Fix) -> Result<(), AccumulateError> {
        self.forces.add_torque(body, torque)
    }
}

/// A value read from the world: exact, or undecided because a fault (a value
/// out of range) was recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observed<T> {
    /// The exact value.
    Exact(T),
    /// No value can be stated: the world recorded a fault.
    Undecided,
}

/// Three-valued outcome of a predicate over an [`Observed`] value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The value is exact and the predicate holds.
    Holds,
    /// The value is exact and the predicate does not hold.
    Violated,
    /// The value is undecided.
    Undecided,
}

impl<T> Observed<T> {
    /// Evaluate `predicate` on the exact value; [`Verdict::Undecided`] when
    /// there is none.
    pub fn verdict(&self, predicate: impl FnOnce(&T) -> bool) -> Verdict {
        match self {
            Self::Exact(v) if predicate(v) => Verdict::Holds,
            Self::Exact(_) => Verdict::Violated,
            Self::Undecided => Verdict::Undecided,
        }
    }
}

/// Channels a participant reports from [`Participant::observe`], in the order
/// it pushed them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObservationSink {
    values: Vec<(u32, Fix128)>,
}

impl ObservationSink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Report `value` on `channel`.
    pub fn push(&mut self, channel: u32, value: Fix128) {
        self.values.push((channel, value));
    }

    /// Every reported `(channel, value)`, in push order.
    #[must_use]
    pub fn values(&self) -> &[(u32, Fix128)] {
        &self.values
    }
}

/// A law that advances its own state inside the world's substep loop.
///
/// See the module documentation for the contract.
pub trait Participant: Send {
    /// Snapshot tag of this participant type.
    fn kind(&self) -> ParticipantKind;

    /// How this participant's time step relates to the world substep.
    fn step_rule(&self) -> StepRule {
        StepRule::FollowSubstep
    }

    /// Advance by one world substep of width `h` (`ctx.h()`). Read bodies
    /// from `ctx`, stage forces on them through `ctx`.
    ///
    /// # Errors
    ///
    /// A [`ParticipantFault`]; the participant must then be exactly as it
    /// was before the call (the forces it staged are dropped by the world).
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault>;

    /// Report observable values. The world reports them as
    /// [`Observed::Undecided`] while a fault is recorded.
    fn observe(&self, out: &mut ObservationSink);

    /// Append the whole state to `out`.
    fn write_state(&self, out: &mut Vec<u8>);

    /// Check that `bytes` is a payload [`Self::read_state`] can apply. Must
    /// not change `self`.
    ///
    /// # Errors
    ///
    /// A [`StateError`] saying why the payload is unusable.
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError>;

    /// Replace the state with `bytes`. Only called with a payload
    /// [`Self::check_state`] accepted.
    fn read_state(&mut self, bytes: &[u8]);
}
