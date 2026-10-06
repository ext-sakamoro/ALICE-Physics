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
//! | participants run in the order [`execution_order`] gives for their [`Participant::ports`]: a writer of a port before its readers, registration order otherwise; with no ports declared that is registration order | world side calls [`execution_order`] at registration |
//! | a loop of ports (a cycle) is refused at registration, never ordered silently | type: [`execution_order`] returns [`OrderError::Cycle`], the world returns it as [`RegisterError::Order`] |
//! | the value behind a port that is not a shared field is state of the participant that holds it, so it is in that participant's [`Participant::write_state`] payload | participant side |
//! | the summed force and torque reach a body through `v += F·h·inv_mass`, `ω += I⁻¹·τ·h`; a product out of range is a fault ([`WorldFault::ForceOutOfRange`]), never clamped | world side |
//! | the rigid integration going out of range ([`WorldFault::RigidOverflow`]) is a fault like a participant's | world side |
//! | a parked (sleeping) body is woken by participant forces only when the velocity change they make in this substep is above the sleep threshold ([`wakes_parked_body`]: the squared norm of `F·inv_mass·h` above `linear_threshold²`, or that of `I⁻¹τ·h` above `angular_threshold²`); below it the force has no effect on the body | world side calls [`wakes_parked_body`] |
//! | a shared field ([`FieldBoard`]) is owned by the world alone; a participant reads the value committed before the substep began and writes only into its own [`FieldStage`] | type: [`SubstepCtx::field`] hands out `&[Fix128]` of the board, [`SubstepCtx::stage_field`] a buffer of the stage |
//! | staged field writes are committed at the end of the substep, in [`execution_order`], all fields at once; a commit that would leave the range of [`Fix128`] changes no field and is a fault ([`WorldFault::FieldOutOfRange`]) | [`run_substep`] |
//! | a participant that returned `Err` has its staged field writes dropped with its forces | [`run_substep`] |
//! | a [`FieldMode::Replace`] field has at most one writer, a plain [`PortAccess::Read`] of a field and a [`PortAccess::ReadCommitted`] of a port that is not a field are refused at registration | [`check_field_ports`], returned as [`RegisterError::Field`] |
//! | moving an extensive amount between layouts keeps its total exactly | [`remap_conserving`], [`deposit_bodies`] |
//! | a snapshot holds the committed field values once, in the world's section, never in a participant payload | world side, bytes from [`FieldBoard::write_values`] |
//! | participants can be registered only in builds with the `std` feature; a `no_std` world has no participants (the world-side participant API requires `std`) | world side |
//!
//! # Coupling through ports
//!
//! A participant declares the values it reads and writes as [`Port`]s
//! ([`Participant::ports`], empty by default). A port is named by a
//! [`PortId`], which, like [`ParticipantKind`], keeps its value across
//! snapshots and versions. Within a substep, a participant that reads a port
//! runs after every participant that writes it, so it sees the value of this
//! substep. Participants that do not depend on each other keep their
//! registration order. A participant that reads and writes the same port has
//! no ordering constraint with itself.
//!
//! When the ports form a loop (A writes what B reads and B writes what A
//! reads), no order lets every read see the value of this substep. The v1
//! rule for such a loop is that its reads see the previous substep's values
//! (explicit coupling); [`execution_order`] does not pick an order for it
//! and returns [`OrderError::Cycle`] instead, so a loop is never coupled
//! without being declared as one. How a participant declares a lagged read is
//! not part of this version ([`PortAccess`] is `#[non_exhaustive]`). Implicit
//! iteration between coupled participants is not part of this module either.
//!
//! # Shared fields
//!
//! A port whose value several participants exchange (a temperature on a grid,
//! a heat source per body) is a *field* on the world's [`FieldBoard`]. A field
//! is named by the same [`PortId`]: the id is what participants declare in
//! [`Participant::ports`], so the execution order, the registration checks and
//! the snapshot all speak of one name, and a port is either held by its
//! participant (no field with that id) or held by the world (a field). The
//! board, not a participant, owns the value, so a snapshot carries each field
//! once however many participants use it.
//!
//! * **Values.** A field is an array of [`Fix128`] on a [`FieldLayout`]: one
//!   sample per rigid body ([`FieldLayout::PerBody`], the way to hand a body a
//!   quantity other than a force: heat, charge, deposited mass) or a uniform
//!   grid of cubic cells ([`FieldLayout::Grid`]). A vector quantity is three
//!   fields, one per component, so every rule below (sums, conservation,
//!   snapshot) is the scalar one.
//! * **Reads.** [`SubstepCtx::field`] returns the value committed at the end of
//!   the previous substep, whoever runs first. A field read is declared as
//!   [`Port::reads_committed`], which adds no ordering constraint, so two
//!   participants may each read what the other writes (thermal ⇄ structure)
//!   without forming a refused loop. The coupling through a field is explicit:
//!   one substep of lag, the same as through the rigid bodies.
//! * **Writes.** [`SubstepCtx::stage_field`] hands out the participant's own
//!   staged buffer. Staged buffers are committed at the end of the substep in
//!   [`execution_order`], all fields at once ([`run_substep`]); a participant
//!   that returned `Err` has its staged writes dropped, as its forces are.
//! * **Several writers.** A [`FieldMode::Replace`] field (a state some
//!   participant integrates, e.g. a temperature) has one writer, checked at
//!   registration; its staged buffer starts as the committed value and
//!   replaces it. A [`FieldMode::Sum`] field (a load or source, e.g. Joule
//!   heat) takes any number of writers; each staged buffer starts at zero and
//!   the committed value is the sum of this substep's buffers, partial sums
//!   taken in execution order and checked, so the first writer that leaves the
//!   range is named the same way on every run. Nothing written in a substep
//!   gives zero, like the force accumulator.
//! * **Layouts.** Moving an extensive amount (heat in J per cell, not a
//!   temperature) between nested grids or from bodies into a grid keeps the
//!   total exactly ([`remap_conserving`], [`deposit_bodies`]); converting a
//!   density to an amount is the participant's job.
//! * **Snapshot.** Proposed section of snapshot format version 2, after
//!   `participants` and `fault`: `fields` = [`FieldBoard::write_values`]
//!   (`count: u64`, then per field in ascending id: `id: u32`, `mode: u8`,
//!   `layout: u8` and its parameters, the values as `hi: i64`, `lo: u64`
//!   little endian). Staged buffers are empty between steps and are not
//!   stored. A restore checks the layouts against the target world's board
//!   ([`FieldBoard::check_values`]) before reading any value, next to the
//!   participant checks. The `fault` section codes, with the field fault
//!   added: `0` none, `1` [`WorldFault::Participant`] (`index: u64`,
//!   `kind: u32`, fault tag `u8`), `2` [`WorldFault::RigidOverflow`], `3`
//!   [`WorldFault::ForceOutOfRange`] (`body: u64`), `4`
//!   [`WorldFault::FieldOutOfRange`] (`field: u32`, `index: u64`).
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
use crate::sleeping::SleepConfig;
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec, vec::Vec};

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

/// Stable name of a value participants exchange (a field, a set of loads).
/// A port keeps its id across snapshots and versions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PortId(u32);

impl PortId {
    /// The port `raw`.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw id.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Whether a participant reads or writes a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PortAccess {
    /// Reads the value of the current substep (runs after every writer).
    Read,
    /// Writes the value (runs before every reader).
    Write,
    /// Reads a shared field's value as committed at the end of the previous
    /// substep. Adds no ordering constraint; the port must be a field of the
    /// world's [`FieldBoard`].
    ReadCommitted,
}

/// One declared access of a participant to a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Port {
    id: PortId,
    access: PortAccess,
}

impl Port {
    /// A read of `id`.
    #[must_use]
    pub const fn reads(id: PortId) -> Self {
        Self {
            id,
            access: PortAccess::Read,
        }
    }

    /// A read of the committed value of the shared field `id`.
    #[must_use]
    pub const fn reads_committed(id: PortId) -> Self {
        Self {
            id,
            access: PortAccess::ReadCommitted,
        }
    }

    /// A write of `id`.
    #[must_use]
    pub const fn writes(id: PortId) -> Self {
        Self {
            id,
            access: PortAccess::Write,
        }
    }

    /// The port.
    #[must_use]
    pub const fn id(self) -> PortId {
        self.id
    }

    /// Read or write.
    #[must_use]
    pub const fn access(self) -> PortAccess {
        self.access
    }
}

/// Why the participants' ports admit no execution order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OrderError {
    /// The ports form a loop within one substep.
    Cycle {
        /// Registration indices of every participant on a loop, ascending.
        /// Participants that only depend on a loop are not listed.
        members: Vec<usize>,
    },
}

/// The order participants run in within a substep, from their declared
/// ports: `ports[i]` is what [`Participant::ports`] returns for the
/// participant registered at index `i`, and the result lists registration
/// indices in run order.
///
/// A participant that writes a port runs before every other participant that
/// reads it. Among participants free to run, the one registered first runs
/// first, so with no ports at all the order is `0, 1, …, n-1`, and the result
/// depends only on `ports`. Two writers of one port are not ordered against
/// each other by that port.
///
/// # Errors
///
/// [`OrderError::Cycle`] when the dependencies form a loop; no order is
/// chosen.
pub fn execution_order(ports: &[&[Port]]) -> Result<Vec<usize>, OrderError> {
    let n = ports.len();
    // succ[w] = readers that must run after writer w
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (w, wp) in ports.iter().enumerate() {
        for p in wp.iter().filter(|p| p.access == PortAccess::Write) {
            for (r, rp) in ports.iter().enumerate() {
                if r != w
                    && rp
                        .iter()
                        .any(|q| q.access == PortAccess::Read && q.id == p.id)
                {
                    succ[w].push(r);
                }
            }
        }
    }
    let mut indegree = vec![0_usize; n];
    for s in &mut succ {
        s.sort_unstable();
        s.dedup();
        for &r in s.iter() {
            indegree[r] += 1;
        }
    }
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    while let Some(next) = (0..n).find(|&i| !done[i] && indegree[i] == 0) {
        done[next] = true;
        order.push(next);
        for &r in &succ[next] {
            indegree[r] -= 1;
        }
    }
    if order.len() == n {
        return Ok(order);
    }
    // A participant left over is on a loop iff it reaches itself through
    // other left-over participants.
    let members = (0..n)
        .filter(|&v| !done[v] && reaches_itself(v, &succ, &done))
        .collect();
    Err(OrderError::Cycle { members })
}

fn reaches_itself(start: usize, succ: &[Vec<usize>], done: &[bool]) -> bool {
    let mut seen = vec![false; succ.len()];
    let mut stack: Vec<usize> = succ[start].clone();
    while let Some(v) = stack.pop() {
        if v == start {
            return true;
        }
        if done[v] || seen[v] {
            continue;
        }
        seen[v] = true;
        stack.extend_from_slice(&succ[v]);
    }
    false
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// With the new participant the declared ports form a loop
    /// ([`execution_order`]); the participant was not registered.
    Order(OrderError),
    /// The declared ports do not fit the world's shared fields
    /// ([`check_field_ports`]); the participant was not registered.
    Field(FieldPortError),
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
#[non_exhaustive]
pub enum WorldFault {
    /// A participant's substep returned `Err`.
    Participant {
        /// Registration index of the participant that failed.
        index: usize,
        /// Its kind.
        kind: ParticipantKind,
        /// What failed.
        fault: ParticipantFault,
    },
    /// The rigid integration (position prediction, velocity from positions)
    /// left the range of [`Fix128`]; the world's overflow flag in fault form.
    /// It names no body because the flag does not.
    ///
    /// The flag covers only part of the products today: the XPBD position
    /// prediction and the velocity derived from jointed positions. The TGS
    /// path does not raise it, and a wrapping addition of a position (XPBD or
    /// TGS) is not detected.
    RigidOverflow,
    /// Applying the summed participant force or torque to body `body`
    /// (`F·h·inv_mass`, `I⁻¹·τ·h`) left the range of [`Fix128`]. Nothing is
    /// clamped.
    ForceOutOfRange {
        /// Index of the body.
        body: usize,
    },
    /// Committing the staged writes of a [`FieldMode::Sum`] field left the
    /// range of [`Fix128`] when the writes of participant `index` were added.
    /// No field changed in that substep.
    FieldOutOfRange {
        /// The field.
        field: PortId,
        /// Registration index of the participant whose write left the range.
        index: usize,
    },
}

/// Why a checked world step did not run normally.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// The [`FieldLayout::PerBody`] field `field` has `samples` samples but
    /// the world holds `bodies` bodies (a body was added or removed after the
    /// field was declared): the step is refused and the world is left
    /// unchanged, nothing ran. The world-side form of
    /// [`ExchangeError::BodyCount`], checked at the start of every step like
    /// [`Self::Rule`].
    BodyCount {
        /// The field.
        field: PortId,
        /// Samples of the field.
        samples: usize,
        /// Bodies in the world.
        bodies: usize,
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
    board: Option<&'a FieldBoard>,
    stage: Option<&'a mut FieldStage>,
    ports: &'a [Port],
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
            board: None,
            stage: None,
            ports: &[],
        })
    }

    /// The same context with the world's shared fields: reads come from
    /// `board` (committed values), writes go to `stage`, and only the ports in
    /// `ports` (what the participant declared) may be read or written.
    #[must_use]
    pub fn with_fields(
        mut self,
        board: &'a FieldBoard,
        stage: &'a mut FieldStage,
        ports: &'a [Port],
    ) -> Self {
        self.board = Some(board);
        self.stage = Some(stage);
        self.ports = ports;
        self
    }

    fn declared(&self, id: PortId, access: PortAccess) -> Result<(), FieldAccessError> {
        if self.ports.iter().any(|p| p.id == id && p.access == access) {
            Ok(())
        } else {
            Err(FieldAccessError::NotDeclared { field: id, access })
        }
    }

    /// The value of field `id` committed before this substep began.
    ///
    /// # Errors
    ///
    /// [`FieldAccessError::Unknown`] when there is no such field (or no
    /// board); [`FieldAccessError::NotDeclared`] when the participant did not
    /// declare [`Port::reads_committed`] for it.
    pub fn field(&self, id: PortId) -> Result<&'a [Fix128], FieldAccessError> {
        let board = self.board.ok_or(FieldAccessError::Unknown(id))?;
        let slot = board.slot(id).ok_or(FieldAccessError::Unknown(id))?;
        self.declared(id, PortAccess::ReadCommitted)?;
        Ok(&slot.value)
    }

    /// The participant's staged buffer for field `id`, created on first use:
    /// a copy of the committed value for a [`FieldMode::Replace`] field,
    /// zeros for a [`FieldMode::Sum`] field. What it holds when the substep
    /// returns `Ok` is committed at the end of the substep.
    ///
    /// # Errors
    ///
    /// [`FieldAccessError::Unknown`] when there is no such field (or no
    /// board); [`FieldAccessError::NotDeclared`] when the participant did not
    /// declare [`Port::writes`] for it.
    pub fn stage_field(&mut self, id: PortId) -> Result<&mut [Fix128], FieldAccessError> {
        let board = self.board.ok_or(FieldAccessError::Unknown(id))?;
        let slot = board.slot(id).ok_or(FieldAccessError::Unknown(id))?;
        self.declared(id, PortAccess::Write)?;
        let stage = self
            .stage
            .as_deref_mut()
            .ok_or(FieldAccessError::Unknown(id))?;
        Ok(stage.buffer(id, || match slot.mode {
            FieldMode::Replace => slot.value.clone(),
            FieldMode::Sum => vec![Fix128::ZERO; slot.value.len()],
        }))
    }

    /// The bodies at the start of this substep. With TGS (one detection per
    /// tick) as well as XPBD, positions and velocities are those at the head
    /// of the substep.
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

    /// The ports this participant reads and writes, empty by default. The
    /// world reads them once, at registration, to fix the execution order
    /// ([`execution_order`]); they must not change while the participant is
    /// registered.
    fn ports(&self) -> &[Port] {
        &[]
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

// ============================================================================
// Shared fields
// ============================================================================

/// Where the samples of a shared field sit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldLayout {
    /// One sample per rigid body, indexed like the world's body list.
    PerBody {
        /// Number of bodies.
        bodies: usize,
    },
    /// A uniform grid of cubic cells. Cell `(ix, iy, iz)` covers
    /// `origin + [ix, ix+1)·cell` (and so on per axis); samples are stored row
    /// major with `ix` fastest, `index = (iz·ny + iy)·nx + ix`.
    Grid {
        /// Lower corner of cell `(0, 0, 0)`.
        origin: Vec3Fix,
        /// Edge length of a cell (positive).
        cell: Fix128,
        /// Cells per axis `[nx, ny, nz]` (each at least 1).
        dims: [usize; 3],
    },
}

impl FieldLayout {
    /// Number of samples, or `None` for a grid with a non-positive cell, an
    /// axis of 0 cells, or more samples than `usize` holds.
    #[must_use]
    pub fn samples(&self) -> Option<usize> {
        match *self {
            Self::PerBody { bodies } => Some(bodies),
            Self::Grid { cell, dims, .. } => {
                if cell <= Fix128::ZERO || dims.contains(&0) {
                    return None;
                }
                dims[0].checked_mul(dims[1])?.checked_mul(dims[2])
            }
        }
    }
}

/// How the staged writes of one substep become the field's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldMode {
    /// At most one writer; its staged buffer (starting as the committed value)
    /// replaces the value. Without a write in a substep the value stays.
    Replace,
    /// Any number of writers; each staged buffer starts at zero and the value
    /// becomes the sum of this substep's buffers (zero when nobody wrote).
    Sum,
}

impl FieldMode {
    const fn tag(self) -> u8 {
        match self {
            Self::Replace => 0,
            Self::Sum => 1,
        }
    }
}

/// Why a field could not be declared or set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldError {
    /// A field with this id is already declared.
    Duplicate(PortId),
    /// The layout has no valid sample count ([`FieldLayout::samples`]).
    InvalidLayout,
    /// No field with this id.
    Unknown(PortId),
    /// The values do not have one entry per sample.
    Length {
        /// Samples of the field.
        expected: usize,
        /// Values given.
        found: usize,
    },
}

/// Why the declared ports do not fit the shared fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldPortError {
    /// A [`FieldMode::Replace`] field declared written by two participants.
    SecondWriter {
        /// The field.
        field: PortId,
        /// Registration index of the first writer.
        first: usize,
        /// Registration index of the second writer.
        second: usize,
    },
    /// A plain [`PortAccess::Read`] of a field: a field read sees the
    /// committed value and must be declared [`PortAccess::ReadCommitted`].
    ReadOfField {
        /// Registration index of the participant.
        participant: usize,
        /// The field.
        field: PortId,
    },
    /// [`PortAccess::ReadCommitted`] of a port that is not a field.
    UnknownField {
        /// Registration index of the participant.
        participant: usize,
        /// The port.
        field: PortId,
    },
}

/// Why a participant could not read or stage a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldAccessError {
    /// No field with this id in the context.
    Unknown(PortId),
    /// The participant did not declare this access in [`Participant::ports`].
    NotDeclared {
        /// The field.
        field: PortId,
        /// The access it would need.
        access: PortAccess,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FieldSlot {
    id: PortId,
    layout: FieldLayout,
    mode: FieldMode,
    value: Vec<Fix128>,
}

/// The world's shared fields: committed values, owned by the world alone.
///
/// See the module documentation (*Shared fields*) for the rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldBoard {
    // ascending id
    fields: Vec<FieldSlot>,
}

impl FieldBoard {
    /// A board without fields.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn slot(&self, id: PortId) -> Option<&FieldSlot> {
        self.fields
            .binary_search_by_key(&id, |f| f.id)
            .ok()
            .map(|i| &self.fields[i])
    }

    /// Declare field `id`, all samples zero.
    ///
    /// # Errors
    ///
    /// [`FieldError::Duplicate`] for an id already declared,
    /// [`FieldError::InvalidLayout`] for a layout without a sample count; the
    /// board is unchanged.
    pub fn declare(
        &mut self,
        id: PortId,
        layout: FieldLayout,
        mode: FieldMode,
    ) -> Result<(), FieldError> {
        let n = layout.samples().ok_or(FieldError::InvalidLayout)?;
        match self.fields.binary_search_by_key(&id, |f| f.id) {
            Ok(_) => Err(FieldError::Duplicate(id)),
            Err(at) => {
                self.fields.insert(
                    at,
                    FieldSlot {
                        id,
                        layout,
                        mode,
                        value: vec![Fix128::ZERO; n],
                    },
                );
                Ok(())
            }
        }
    }

    /// Set the committed value of field `id` (an initial condition, set
    /// between steps).
    ///
    /// # Errors
    ///
    /// [`FieldError::Unknown`] or [`FieldError::Length`]; the board is
    /// unchanged.
    pub fn set(&mut self, id: PortId, values: &[Fix128]) -> Result<(), FieldError> {
        let i = self
            .fields
            .binary_search_by_key(&id, |f| f.id)
            .map_err(|_| FieldError::Unknown(id))?;
        let slot = &mut self.fields[i];
        if values.len() != slot.value.len() {
            return Err(FieldError::Length {
                expected: slot.value.len(),
                found: values.len(),
            });
        }
        slot.value.copy_from_slice(values);
        Ok(())
    }

    /// The committed value of field `id`.
    #[must_use]
    pub fn value(&self, id: PortId) -> Option<&[Fix128]> {
        self.slot(id).map(|f| f.value.as_slice())
    }

    /// The layout of field `id`.
    #[must_use]
    pub fn layout(&self, id: PortId) -> Option<FieldLayout> {
        self.slot(id).map(|f| f.layout)
    }

    /// The mode of field `id`.
    #[must_use]
    pub fn mode(&self, id: PortId) -> Option<FieldMode> {
        self.slot(id).map(|f| f.mode)
    }

    /// Every declared field id, ascending.
    #[must_use]
    pub fn ids(&self) -> Vec<PortId> {
        self.fields.iter().map(|f| f.id).collect()
    }

    /// Commit `stages` (registration index, stage) in run order. Either every
    /// field takes its new value or, on error, none changes.
    fn commit(&mut self, stages: &[(usize, FieldStage)]) -> Result<(), WorldFault> {
        let mut next: Vec<Vec<Fix128>> = Vec::with_capacity(self.fields.len());
        for slot in &self.fields {
            let mut v = match slot.mode {
                FieldMode::Replace => slot.value.clone(),
                FieldMode::Sum => vec![Fix128::ZERO; slot.value.len()],
            };
            for (index, stage) in stages {
                let Some(staged) = stage.staged(slot.id) else {
                    continue;
                };
                match slot.mode {
                    FieldMode::Replace => v.copy_from_slice(staged),
                    FieldMode::Sum => {
                        for (acc, add) in v.iter_mut().zip(staged) {
                            *acc = checked_add(*acc, *add).ok_or(WorldFault::FieldOutOfRange {
                                field: slot.id,
                                index: *index,
                            })?;
                        }
                    }
                }
            }
            next.push(v);
        }
        for (slot, v) in self.fields.iter_mut().zip(next) {
            slot.value = v;
        }
        Ok(())
    }

    fn write_descriptor(slot: &FieldSlot, out: &mut Vec<u8>) {
        out.extend_from_slice(&slot.id.get().to_le_bytes());
        out.push(slot.mode.tag());
        match slot.layout {
            FieldLayout::PerBody { bodies } => {
                out.push(0);
                out.extend_from_slice(&(bodies as u64).to_le_bytes());
            }
            FieldLayout::Grid { origin, cell, dims } => {
                out.push(1);
                for f in [origin.x, origin.y, origin.z, cell] {
                    put_fix(out, f);
                }
                for d in dims {
                    out.extend_from_slice(&(d as u64).to_le_bytes());
                }
            }
        }
    }

    /// Append every field (descriptor and committed values) to `out`, the
    /// bytes of the snapshot section `fields`.
    pub fn write_values(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.fields.len() as u64).to_le_bytes());
        for slot in &self.fields {
            Self::write_descriptor(slot, out);
            for &f in &slot.value {
                put_fix(out, f);
            }
        }
    }

    /// Check that `bytes` (from [`Self::write_values`]) describes exactly the
    /// fields of this board (ids, modes, layouts); the values are not checked.
    /// Does not change the board.
    ///
    /// # Errors
    ///
    /// [`StateError::Length`] when the byte count differs from what this board
    /// needs, [`StateError::InvalidValue`] when a descriptor differs.
    pub fn check_values(&self, bytes: &[u8]) -> Result<(), StateError> {
        let mut own = Vec::new();
        self.write_values(&mut own);
        if own.len() != bytes.len() {
            return Err(StateError::Length {
                expected: own.len(),
                found: bytes.len(),
            });
        }
        let mut at = 8;
        if own[..at] != bytes[..at] {
            return Err(StateError::InvalidValue);
        }
        for slot in &self.fields {
            let mut d = Vec::new();
            Self::write_descriptor(slot, &mut d);
            if bytes[at..at + d.len()] != d[..] {
                return Err(StateError::InvalidValue);
            }
            at += d.len() + 16 * slot.value.len();
        }
        Ok(())
    }

    /// Replace every committed value with those in `bytes`. Only called with
    /// bytes [`Self::check_values`] accepted.
    pub fn read_values(&mut self, bytes: &[u8]) {
        let mut at = 8;
        for slot in &mut self.fields {
            let mut d = Vec::new();
            Self::write_descriptor(slot, &mut d);
            at += d.len();
            for f in &mut slot.value {
                *f = get_fix(&bytes[at..at + 16]);
                at += 16;
            }
        }
    }
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn get_fix(b: &[u8]) -> Fix128 {
    let mut hi = [0_u8; 8];
    let mut lo = [0_u8; 8];
    hi.copy_from_slice(&b[0..8]);
    lo.copy_from_slice(&b[8..16]);
    Fix128 {
        hi: i64::from_le_bytes(hi),
        lo: u64::from_le_bytes(lo),
    }
}

/// `a + b`, or `None` when the sum leaves the range of [`Fix128`].
fn checked_add(a: Fix128, b: Fix128) -> Option<Fix128> {
    let wide = |f: Fix128| (i128::from(f.hi) << 64) | i128::from(f.lo);
    let s = wide(a).checked_add(wide(b))?;
    Some(Fix128 {
        hi: (s >> 64) as i64,
        lo: s as u64,
    })
}

/// One participant's staged field writes for one substep.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldStage {
    // ascending id
    staged: Vec<(PortId, Vec<Fix128>)>,
}

impl FieldStage {
    /// An empty stage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing was staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.staged.is_empty()
    }

    /// The staged buffer of field `id`, if one was created.
    #[must_use]
    pub fn staged(&self, id: PortId) -> Option<&[Fix128]> {
        self.staged
            .binary_search_by_key(&id, |s| s.0)
            .ok()
            .map(|i| self.staged[i].1.as_slice())
    }

    fn buffer(&mut self, id: PortId, init: impl FnOnce() -> Vec<Fix128>) -> &mut [Fix128] {
        let i = match self.staged.binary_search_by_key(&id, |s| s.0) {
            Ok(i) => i,
            Err(at) => {
                self.staged.insert(at, (id, init()));
                at
            }
        };
        &mut self.staged[i].1
    }
}

/// Check the declared ports of every participant (`ports[i]` for the
/// participant registered at `i`) against the fields of `board`.
///
/// # Errors
///
/// The first [`FieldPortError`] in registration order, then port order.
pub fn check_field_ports(board: &FieldBoard, ports: &[&[Port]]) -> Result<(), FieldPortError> {
    let mut writers: Vec<(PortId, usize)> = Vec::new();
    for (i, ps) in ports.iter().enumerate() {
        for p in ps.iter() {
            let field = board.slot(p.id);
            match (p.access, field) {
                (PortAccess::Read, Some(_)) => {
                    return Err(FieldPortError::ReadOfField {
                        participant: i,
                        field: p.id,
                    })
                }
                (PortAccess::ReadCommitted, None) => {
                    return Err(FieldPortError::UnknownField {
                        participant: i,
                        field: p.id,
                    })
                }
                (PortAccess::Write, Some(slot)) if slot.mode == FieldMode::Replace => {
                    match writers.iter().find(|w| w.0 == p.id) {
                        Some(&(_, first)) if first != i => {
                            return Err(FieldPortError::SecondWriter {
                                field: p.id,
                                first,
                                second: i,
                            })
                        }
                        Some(_) => {}
                        None => writers.push((p.id, i)),
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// What the world fixes at registration for a list of participants: the run
/// order within a substep and each participant's declared ports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantPlan {
    order: Vec<usize>,
    ports: Vec<Vec<Port>>,
}

impl ParticipantPlan {
    /// The plan for `participants` (registration order) over the fields of
    /// `board`. The world builds it again whenever a participant or a field is
    /// added.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Order`] for a loop of ports, [`RegisterError::Field`]
    /// when the ports do not fit the fields.
    pub fn new(
        participants: &[Box<dyn Participant>],
        board: &FieldBoard,
    ) -> Result<Self, RegisterError> {
        let ports: Vec<Vec<Port>> = participants.iter().map(|p| p.ports().to_vec()).collect();
        let views: Vec<&[Port]> = ports.iter().map(Vec::as_slice).collect();
        check_field_ports(board, &views).map_err(RegisterError::Field)?;
        let order = execution_order(&views).map_err(RegisterError::Order)?;
        Ok(Self { order, ports })
    }

    /// Registration indices in run order.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }
}

/// Position and width of one substep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubstepTime {
    /// Index of the substep within the frame.
    pub index: usize,
    /// Substeps in the frame.
    pub count: usize,
    /// Width of the substep.
    pub h: Fix128,
}

/// Why [`run_substep`] did not run (nothing was called or changed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExchangeError {
    /// The plan was built for another number of participants.
    Plan {
        /// Participants given.
        participants: usize,
        /// Participants in the plan.
        planned: usize,
    },
    /// `frozen` does not have one flag per participant.
    Frozen {
        /// Flags given.
        flags: usize,
        /// Participants given.
        participants: usize,
    },
    /// The force accumulator does not have one slot per body.
    Accumulate(AccumulateError),
    /// A [`FieldLayout::PerBody`] field does not have one sample per body.
    BodyCount {
        /// The field.
        field: PortId,
        /// Samples of the field.
        samples: usize,
        /// Bodies.
        bodies: usize,
    },
}

/// One substep of every participant, the part of the world step that runs the
/// participants and exchanges their results.
///
/// Participants that are not `frozen` run in the plan's order. Each gets the
/// bodies at the start of the substep, the fields as committed before the
/// substep, and its own empty force accumulator and field stage. On `Ok` its
/// forces are merged into `forces`; on `Err` its forces and staged writes are
/// dropped, it is marked frozen and a [`WorldFault::Participant`] is
/// reported. After every participant ran, the staged writes of those that
/// returned `Ok` are committed in run order ([`FieldBoard`] rules); a commit
/// out of range changes no field and is reported as
/// [`WorldFault::FieldOutOfRange`].
///
/// Returns the faults of this substep in the order they happened.
///
/// # Errors
///
/// An [`ExchangeError`] when the inputs do not fit together; nothing ran.
pub fn run_substep(
    participants: &mut [Box<dyn Participant>],
    plan: &ParticipantPlan,
    frozen: &mut [bool],
    bodies: &[RigidBody],
    board: &mut FieldBoard,
    forces: &mut ForceAccumulator,
    time: SubstepTime,
) -> Result<Vec<WorldFault>, ExchangeError> {
    let n = participants.len();
    if plan.order.len() != n {
        return Err(ExchangeError::Plan {
            participants: n,
            planned: plan.order.len(),
        });
    }
    if frozen.len() != n {
        return Err(ExchangeError::Frozen {
            flags: frozen.len(),
            participants: n,
        });
    }
    if forces.len() != bodies.len() {
        return Err(ExchangeError::Accumulate(AccumulateError::LengthMismatch {
            slots: forces.len(),
            bodies: bodies.len(),
        }));
    }
    for slot in &board.fields {
        if let FieldLayout::PerBody { .. } = slot.layout {
            if slot.value.len() != bodies.len() {
                return Err(ExchangeError::BodyCount {
                    field: slot.id,
                    samples: slot.value.len(),
                    bodies: bodies.len(),
                });
            }
        }
    }
    let mut faults = Vec::new();
    let mut kept: Vec<(usize, FieldStage)> = Vec::with_capacity(n);
    for &i in &plan.order {
        if frozen[i] {
            continue;
        }
        let mut staged_forces = ForceAccumulator::new(bodies.len());
        let mut stage = FieldStage::new();
        let result = {
            let ctx = SubstepCtx::new(bodies, &mut staged_forces, time.index, time.count, time.h)
                .map_err(ExchangeError::Accumulate)?;
            let mut ctx = ctx.with_fields(board, &mut stage, &plan.ports[i]);
            participants[i].substep(&mut ctx, time.h)
        };
        match result {
            Ok(()) => {
                forces
                    .merge(&staged_forces)
                    .map_err(ExchangeError::Accumulate)?;
                kept.push((i, stage));
            }
            Err(fault) => {
                frozen[i] = true;
                faults.push(WorldFault::Participant {
                    index: i,
                    kind: participants[i].kind(),
                    fault,
                });
            }
        }
    }
    if let Err(fault) = board.commit(&kept) {
        faults.push(fault);
    }
    Ok(faults)
}

/// Why an amount could not be moved between layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemapError {
    /// The values do not have one entry per source sample.
    Length {
        /// Samples of the source.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// A layout is not a valid grid ([`FieldLayout::samples`]).
    InvalidLayout,
    /// The grids are not nested: not the same origin, or cell sizes not an
    /// integer ratio, or not covering the same box.
    NotNested,
    /// A sum left the range of [`Fix128`].
    OutOfRange,
    /// Body `body` lies outside the grid.
    OutsideGrid {
        /// Index of the body.
        body: usize,
    },
}

/// `big = n·small` exactly with an integer `n ≥ 1`.
fn integer_ratio(big: Fix128, small: Fix128) -> Option<usize> {
    let n = big / small;
    if n.lo == 0 && n.hi >= 1 && Fix128::from_int(n.hi) * small == big {
        usize::try_from(n.hi).ok()
    } else {
        None
    }
}

fn grid_of(layout: &FieldLayout) -> Result<(Vec3Fix, Fix128, [usize; 3]), RemapError> {
    match *layout {
        FieldLayout::Grid { origin, cell, dims } if layout.samples().is_some() => {
            Ok((origin, cell, dims))
        }
        _ => Err(RemapError::InvalidLayout),
    }
}

/// Move an extensive amount (per cell, not a density) from grid `src` to grid
/// `dst`, keeping the total exactly.
///
/// The grids must be nested: the same origin, cell sizes in an integer ratio
/// `r`, and the same box. Coarsening sums the `r³` fine cells of each coarse
/// cell. Refining gives each fine cell `q / r³` of its coarse cell's amount
/// `q`, and the first fine cell (lowest index) also the remainder
/// `q − r³·(q / r³)`, so the fine cells add up to `q` exactly. Equal grids
/// copy.
///
/// # Errors
///
/// [`RemapError::InvalidLayout`] when a layout is not a grid,
/// [`RemapError::Length`] for `values` of the wrong length,
/// [`RemapError::NotNested`] for grids that are not nested,
/// [`RemapError::OutOfRange`] when a coarse sum leaves the range.
pub fn remap_conserving(
    src: &FieldLayout,
    values: &[Fix128],
    dst: &FieldLayout,
) -> Result<Vec<Fix128>, RemapError> {
    let (so, sc, sd) = grid_of(src)?;
    let (d_o, dc, dd) = grid_of(dst)?;
    let n_src = src.samples().ok_or(RemapError::InvalidLayout)?;
    let n_dst = dst.samples().ok_or(RemapError::InvalidLayout)?;
    if values.len() != n_src {
        return Err(RemapError::Length {
            expected: n_src,
            found: values.len(),
        });
    }
    if so != d_o {
        return Err(RemapError::NotNested);
    }
    let coarsen = dc >= sc;
    let r = if coarsen {
        integer_ratio(dc, sc)
    } else {
        integer_ratio(sc, dc)
    }
    .ok_or(RemapError::NotNested)?;
    let (fine, coarse) = if coarsen { (sd, dd) } else { (dd, sd) };
    for a in 0..3 {
        if coarse[a].checked_mul(r) != Some(fine[a]) {
            return Err(RemapError::NotNested);
        }
    }
    let parent =
        |ix: usize, iy: usize, iz: usize| ((iz / r) * coarse[1] + iy / r) * coarse[0] + ix / r;
    if coarsen {
        let mut out = vec![Fix128::ZERO; n_dst];
        for iz in 0..fine[2] {
            for iy in 0..fine[1] {
                for ix in 0..fine[0] {
                    let f = (iz * fine[1] + iy) * fine[0] + ix;
                    let c = parent(ix, iy, iz);
                    out[c] = checked_add(out[c], values[f]).ok_or(RemapError::OutOfRange)?;
                }
            }
        }
        Ok(out)
    } else {
        let k = r
            .checked_mul(r)
            .and_then(|v| v.checked_mul(r))
            .and_then(|v| i64::try_from(v).ok())
            .ok_or(RemapError::NotNested)?;
        let kf = Fix128::from_int(k);
        let mut out = vec![Fix128::ZERO; n_dst];
        for iz in 0..fine[2] {
            for iy in 0..fine[1] {
                for ix in 0..fine[0] {
                    let q = values[parent(ix, iy, iz)];
                    let base = q / kf;
                    let first = ix % r == 0 && iy % r == 0 && iz % r == 0;
                    out[(iz * fine[1] + iy) * fine[0] + ix] =
                        if first { base + (q - base * kf) } else { base };
                }
            }
        }
        Ok(out)
    }
}

/// Deposit an extensive amount per body (`amounts[i]` for `bodies[i]`) into
/// the grid cell that holds each body's position, keeping the total exactly.
///
/// # Errors
///
/// [`RemapError::InvalidLayout`] when `dst` is not a grid,
/// [`RemapError::Length`] when `amounts` is not one per body,
/// [`RemapError::OutsideGrid`] for a body outside the grid (nothing is
/// dropped silently), [`RemapError::OutOfRange`] when a cell sum leaves the
/// range.
pub fn deposit_bodies(
    bodies: &[RigidBody],
    amounts: &[Fix128],
    dst: &FieldLayout,
) -> Result<Vec<Fix128>, RemapError> {
    let (origin, cell, dims) = grid_of(dst)?;
    let n = dst.samples().ok_or(RemapError::InvalidLayout)?;
    if amounts.len() != bodies.len() {
        return Err(RemapError::Length {
            expected: bodies.len(),
            found: amounts.len(),
        });
    }
    let mut out = vec![Fix128::ZERO; n];
    for (b, (body, &q)) in bodies.iter().zip(amounts).enumerate() {
        let p = body.position;
        let mut idx = [0_usize; 3];
        for (a, (x, o)) in [(p.x, origin.x), (p.y, origin.y), (p.z, origin.z)]
            .into_iter()
            .enumerate()
        {
            let t = (x - o) / cell;
            idx[a] = usize::try_from(t.hi)
                .ok()
                .filter(|&i| i < dims[a])
                .ok_or(RemapError::OutsideGrid { body: b })?;
        }
        let c = (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0];
        out[c] = checked_add(out[c], q).ok_or(RemapError::OutOfRange)?;
    }
    Ok(out)
}

// ============================================================================
// Waking a parked body
// ============================================================================

fn above(d: Option<Vec3Fix>, threshold: Fix128) -> bool {
    let sq = d.and_then(|d| {
        let x = d.x.checked_mul(d.x)?;
        let y = d.y.checked_mul(d.y)?;
        let z = d.z.checked_mul(d.z)?;
        checked_add(checked_add(x, y)?, z)
    });
    match (sq, threshold.checked_mul(threshold)) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(s), Some(t)) => s > t,
    }
}

/// Whether participant force `force` and torque `torque` (summed over this
/// substep) wake the parked body `body`: the velocity change they make in one
/// substep of width `h` is above the sleep threshold,
/// `|F·inv_mass·h|² > linear_threshold²` or `|I⁻¹τ·h|² > angular_threshold²`
/// (`I⁻¹` in the world frame), strictly. The change is the one the world
/// applies (`F·inv_mass` then `·h`), the squares are taken in [`Fix128`] and
/// compared without a square root, so the decision is the same on every
/// platform. A change too large to represent wakes the body (the world then
/// reports it as [`WorldFault::ForceOutOfRange`]). A static body never wakes.
/// Below the threshold the world leaves the body parked and the force has no
/// effect on it.
#[must_use]
pub fn wakes_parked_body(
    body: &RigidBody,
    force: Vec3Fix,
    torque: Vec3Fix,
    h: Fix128,
    sleep: &SleepConfig,
) -> bool {
    if body.is_static() {
        return false;
    }
    let dv = force
        .checked_scale(body.inv_mass)
        .and_then(|a| a.checked_scale(h));
    let dw = body.world_inv_inertia_apply(torque).checked_scale(h);
    above(dv, sleep.linear_threshold) || above(dw, sleep.angular_threshold)
}
