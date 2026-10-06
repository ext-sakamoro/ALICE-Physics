//! The world side of the participant contract ([`crate::world_participant`]):
//! registration, the checked step, faults, observation and shared fields.
//! Like the contract itself, this API is unstable and may change in a minor
//! release (see the module documentation of [`crate::world_participant`]).
//!
//! Where the participants run: in every step loop ([`PhysicsWorld::try_step`]
//! for XPBD and TGS, [`PhysicsWorld::try_step_parallel`],
//! [`PhysicsWorld::step_with_bridge`]) at the start of each substep, before
//! the substep body. Per substep, with `h = dt / substeps` (a [`Fix128`]
//! division on every path):
//!
//! 1. [`run_substep`] calls the participants in their run order with the
//!    bodies as they are at the start of the substep and commits the staged
//!    field writes;
//! 2. the summed forces reach the dynamic bodies, `v += F·inv_mass·h` and
//!    `ω += I⁻¹·τ·h` (checked: a result out of range leaves the body as it was
//!    and records [`WorldFault::ForceOutOfRange`]; the check covers the
//!    intermediate `F·inv_mass` and `I⁻¹·τ` too, not only the final `·h`);
//!    the sums themselves are not checked: [`ForceAccumulator`] adds with the
//!    wrapping addition of [`Fix128`] (in [`run_substep`] and in
//!    [`ForceAccumulator::merge`]), so staged forces whose sum leaves the
//!    range wrap before this step sees them; a sleeping body is woken
//!    only when [`wakes_parked_body`] says so, otherwise the force has no
//!    effect on it;
//! 3. the substep body runs as before;
//! 4. when the world's overflow flag ([`PhysicsWorld::overflow_detected`])
//!    went up during the substep, [`WorldFault::RigidOverflow`] is recorded.
//!
//! With no participant registered none of this runs: the step is the one it
//! was before participants existed, bit for bit, and no fault is recorded
//! (the overflow flag then stays a flag, as it always was).
//!
//! Participants exist only with the `std` feature (they are held behind a
//! [`std::sync::Mutex`] so that the world stays `Sync` while a participant
//! is only `Send`).

use super::{BodyObservation, PhysicsWorld};
use crate::math::Fix128;
#[cfg(feature = "std")]
use crate::math::Vec3Fix;
#[cfg(feature = "std")]
use crate::world_participant::{
    run_substep, wakes_parked_body, ForceAccumulator, ObservationSink, Observed, Participant,
    ParticipantKind, ParticipantPlan, StepRule, SubstepTime,
};
use crate::world_participant::{
    FieldBoard, FieldError, FieldLayout, FieldMode, PortId, RegisterError, StepError, WorldFault,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Why [`PhysicsWorld::declare_field`] refused a field. The world is
/// unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeclareFieldError {
    /// The board refused the field ([`FieldBoard::declare`]).
    Field(FieldError),
    /// With the new field the ports of the registered participants no longer
    /// fit (a plain read of the new field, or a second writer of a
    /// [`FieldMode::Replace`] field); see [`crate::world_participant::check_field_ports`].
    Ports(RegisterError),
}

impl core::fmt::Display for DeclareFieldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Field(e) => write!(f, "field refused by the board: {e:?}"),
            Self::Ports(e) => write!(f, "field does not fit the registered ports: {e:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeclareFieldError {}

/// `a + b` per component, or `None` when a sum leaves the range of [`Fix128`].
#[cfg(feature = "std")]
fn checked_add_vec(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    let add = |x: Fix128, y: Fix128| -> Option<Fix128> {
        let wide = |f: Fix128| (i128::from(f.hi) << 64) | i128::from(f.lo);
        let s = wide(x).checked_add(wide(y))?;
        Some(Fix128 {
            hi: (s >> 64) as i64,
            lo: s as u64,
        })
    };
    Some(Vec3Fix::new(add(a.x, b.x)?, add(a.y, b.y)?, add(a.z, b.z)?))
}

/// The participant list behind the mutex, recovering it if a participant
/// panicked while the lock was held.
#[cfg(feature = "std")]
fn list_mut(m: &mut std::sync::Mutex<Vec<Box<dyn Participant>>>) -> &mut Vec<Box<dyn Participant>> {
    m.get_mut()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl PhysicsWorld {
    // ── Registration ─────────────────────────────────────────────────────

    /// Register `participant`; it runs in every substep from the next step on.
    /// Returns its registration index.
    ///
    /// The participant's ports ([`Participant::ports`]) are read once, here,
    /// and checked with the ports of the participants already registered and
    /// the world's fields ([`ParticipantPlan::new`]). A
    /// [`StepRule::Fixed`] step is checked against the substep at every step
    /// ([`StepError::Rule`]), because the substep is only known then.
    ///
    /// # Errors
    ///
    /// [`RegisterError::NonPositiveStep`] for `StepRule::Fixed(Δt ≤ 0)`,
    /// [`RegisterError::Order`] when the ports would form a loop,
    /// [`RegisterError::Field`] when they do not fit the fields. The world is
    /// unchanged.
    #[cfg(feature = "std")]
    pub fn add_participant(
        &mut self,
        participant: Box<dyn Participant>,
    ) -> Result<usize, RegisterError> {
        if let StepRule::Fixed(step) = participant.step_rule() {
            if step <= Fix128::ZERO {
                return Err(RegisterError::NonPositiveStep);
            }
        }
        let list = list_mut(&mut self.participants);
        list.push(participant);
        match ParticipantPlan::new(list, &self.fields) {
            Ok(plan) => {
                self.participant_plan = Some(plan);
                Ok(list.len() - 1)
            }
            Err(e) => {
                list.pop();
                Err(e)
            }
        }
    }

    /// Number of registered participants.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn participant_count(&self) -> usize {
        self.lock_participants().len()
    }

    /// The kinds of the registered participants, in registration order.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn participant_kinds(&self) -> Vec<ParticipantKind> {
        self.lock_participants().iter().map(|p| p.kind()).collect()
    }

    /// The state of participant `index` as its [`Participant::write_state`]
    /// writes it (the payload a snapshot holds), or `None` for an index past
    /// the last participant.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn participant_state(&self, index: usize) -> Option<Vec<u8>> {
        let list = self.lock_participants();
        let p = list.get(index)?;
        let mut out = Vec::new();
        p.write_state(&mut out);
        Some(out)
    }

    /// What participant `index` reports ([`Participant::observe`]):
    /// [`Observed::Undecided`] while a fault is recorded, `None` for an index
    /// past the last participant.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn observe_participant(&self, index: usize) -> Option<Observed<ObservationSink>> {
        let list = self.lock_participants();
        let p = list.get(index)?;
        if self.fault.is_some() {
            return Some(Observed::Undecided);
        }
        let mut sink = ObservationSink::new();
        p.observe(&mut sink);
        Some(Observed::Exact(sink))
    }

    #[cfg(feature = "std")]
    fn lock_participants(&self) -> std::sync::MutexGuard<'_, Vec<Box<dyn Participant>>> {
        self.participants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The participants in registration order, for the snapshot writer.
    #[cfg(feature = "std")]
    pub(super) fn write_participants(&self, out: &mut Vec<(u32, Vec<u8>)>) {
        for p in self.lock_participants().iter() {
            let mut payload = Vec::new();
            p.write_state(&mut payload);
            out.push((p.kind().get(), payload));
        }
    }

    /// Check snapshot payloads against the registered participants (kinds,
    /// count, order, then every payload) without changing anything.
    pub(super) fn check_participants(
        &mut self,
        stored: &[(u32, Vec<u8>)],
    ) -> Result<(), super::WorldSnapshotError> {
        #[cfg(feature = "std")]
        {
            let list = list_mut(&mut self.participants);
            let snapshot: Vec<ParticipantKind> =
                stored.iter().map(|s| ParticipantKind::new(s.0)).collect();
            let world: Vec<ParticipantKind> = list.iter().map(|p| p.kind()).collect();
            if let Some(m) =
                crate::world_participant::ParticipantMismatch::classify(&snapshot, &world)
            {
                return Err(super::WorldSnapshotError::ParticipantMismatch(m));
            }
            for (index, (p, (_, payload))) in list.iter().zip(stored).enumerate() {
                p.check_state(payload).map_err(|error| {
                    super::WorldSnapshotError::ParticipantState { index, error }
                })?;
            }
            Ok(())
        }
        #[cfg(not(feature = "std"))]
        {
            if stored.is_empty() {
                Ok(())
            } else {
                Err(super::WorldSnapshotError::ParticipantMismatch(
                    crate::world_participant::ParticipantMismatch::Count {
                        snapshot: stored.len(),
                        world: 0,
                    },
                ))
            }
        }
    }

    /// Read payloads [`Self::check_participants`] accepted.
    pub(super) fn read_participants(&mut self, stored: &[(u32, Vec<u8>)]) {
        #[cfg(feature = "std")]
        for (p, (_, payload)) in list_mut(&mut self.participants).iter_mut().zip(stored) {
            p.read_state(payload);
        }
        #[cfg(not(feature = "std"))]
        debug_assert!(stored.is_empty(), "checked by check_participants");
    }

    // ── Faults ───────────────────────────────────────────────────────────

    /// The first fault recorded, if any. While one is recorded,
    /// [`Self::try_step`] refuses to run ([`StepError::Faulted`]) and the
    /// checked observations are undecided. It is part of the snapshot.
    #[must_use]
    pub const fn fault(&self) -> Option<WorldFault> {
        self.fault
    }

    /// Forget the recorded fault so that steps run again. The world's
    /// overflow flag ([`Self::overflow_detected`]) is not touched.
    pub fn clear_fault(&mut self) {
        self.fault = None;
    }

    /// The fault as stored (snapshot writer and reader).
    pub(super) fn set_fault(&mut self, fault: Option<WorldFault>) {
        self.fault = fault;
    }

    fn record_fault(&mut self, fault: WorldFault) {
        if self.fault.is_none() {
            self.fault = Some(fault);
        }
    }

    /// [`Self::observe_body`], or [`Observed::Undecided`] while a fault is
    /// recorded or the overflow flag is up (the values may have left the
    /// range of [`Fix128`]); `None` for an index past the last body.
    #[must_use]
    pub fn observe_body_checked(
        &self,
        body_idx: usize,
    ) -> Option<crate::world_participant::Observed<BodyObservation>> {
        let o = self.observe_body(body_idx)?;
        Some(if self.fault.is_some() || self.overflow_detected {
            crate::world_participant::Observed::Undecided
        } else {
            crate::world_participant::Observed::Exact(o)
        })
    }

    // ── Shared fields ────────────────────────────────────────────────────

    /// The world's shared fields.
    #[must_use]
    pub const fn fields(&self) -> &FieldBoard {
        &self.fields
    }

    /// Declare shared field `id`, all samples zero. Declare a field before
    /// the participants that use it.
    ///
    /// # Errors
    ///
    /// [`DeclareFieldError::Field`] when the board refuses it,
    /// [`DeclareFieldError::Ports`] when the registered participants' ports
    /// no longer fit with it. The world is unchanged.
    pub fn declare_field(
        &mut self,
        id: PortId,
        layout: FieldLayout,
        mode: FieldMode,
    ) -> Result<(), DeclareFieldError> {
        let mut board = self.fields.clone();
        board
            .declare(id, layout, mode)
            .map_err(DeclareFieldError::Field)?;
        #[cfg(feature = "std")]
        {
            let list = list_mut(&mut self.participants);
            if !list.is_empty() {
                let plan = ParticipantPlan::new(list, &board).map_err(DeclareFieldError::Ports)?;
                self.participant_plan = Some(plan);
            }
        }
        self.fields = board;
        Ok(())
    }

    /// Set the committed value of field `id` (an initial condition, between
    /// steps).
    ///
    /// # Errors
    ///
    /// [`FieldError::Unknown`] or [`FieldError::Length`]; the field is
    /// unchanged.
    pub fn set_field(&mut self, id: PortId, values: &[Fix128]) -> Result<(), FieldError> {
        self.fields.set(id, values)
    }

    /// The field section as stored (snapshot writer and reader).
    pub(super) fn fields_mut(&mut self) -> &mut FieldBoard {
        &mut self.fields
    }

    // ── Checked step ─────────────────────────────────────────────────────

    /// One step of `dt`, refusing to run when it cannot run normally.
    ///
    /// In order: while a fault is recorded the step is refused
    /// ([`StepError::Faulted`]) and the world is unchanged; a non-positive
    /// `dt` changes nothing and is `Ok`; every participant's step rule is
    /// checked against `h = dt / substeps` ([`StepError::Rule`]) and every
    /// [`FieldLayout::PerBody`] field against the body count
    /// ([`StepError::BodyCount`]), refused unchanged. Then the step runs (see
    /// the module documentation of [`crate::world_participant`] and
    /// [`Self::step`]); when a fault was recorded during it the step still
    /// ran to the end and [`StepError::FaultRaised`] reports it.
    ///
    /// With no participant registered this is [`Self::step`] as it always
    /// was, bit for bit.
    ///
    /// # Errors
    ///
    /// See above.
    pub fn try_step(&mut self, dt: Fix128) -> Result<(), StepError> {
        if !self.check_step(dt)? {
            return Ok(());
        }
        self.run_step(dt);
        self.fault
            .map_or(Ok(()), |f| Err(StepError::FaultRaised(f)))
    }

    /// [`Self::try_step`] on the batched (parallel) path of
    /// [`Self::step_parallel`]. Participants are called one after the other,
    /// as in [`Self::try_step`].
    ///
    /// # Errors
    ///
    /// As [`Self::try_step`].
    #[cfg(feature = "parallel")]
    pub fn try_step_parallel(&mut self, dt: Fix128) -> Result<(), StepError> {
        if !self.check_step(dt)? {
            return Ok(());
        }
        self.run_step_parallel(dt);
        self.fault
            .map_or(Ok(()), |f| Err(StepError::FaultRaised(f)))
    }

    /// The checks at the start of a step: `Ok(true)` to run, `Ok(false)` for
    /// a non-positive `dt` (nothing to do), `Err` to refuse. Changes nothing.
    pub(super) fn check_step(&self, dt: Fix128) -> Result<bool, StepError> {
        if let Some(f) = self.fault {
            return Err(StepError::Faulted(f));
        }
        if dt <= Fix128::ZERO {
            return Ok(false);
        }
        #[cfg(feature = "std")]
        {
            let list = self.lock_participants();
            if !list.is_empty() {
                let h = dt / Fix128::from_int(self.config.substeps as i64);
                for (index, p) in list.iter().enumerate() {
                    p.step_rule()
                        .steps_per_substep(h)
                        .map_err(|error| StepError::Rule { index, error })?;
                }
                for field in self.fields.ids() {
                    if let Some(FieldLayout::PerBody { .. }) = self.fields.layout(field) {
                        let samples = self.fields.value(field).map_or(0, <[Fix128]>::len);
                        if samples != self.bodies.len() {
                            return Err(StepError::BodyCount {
                                field,
                                samples,
                                bodies: self.bodies.len(),
                            });
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    /// One `false` per registered participant: the participants that failed
    /// during the step being run (they are not called again in it). Empty
    /// without participants, which turns the substep hooks off.
    pub(super) fn participant_flags(&mut self) -> Vec<bool> {
        #[cfg(feature = "std")]
        {
            alloc_flags(list_mut(&mut self.participants).len())
        }
        #[cfg(not(feature = "std"))]
        {
            Vec::new()
        }
    }

    /// Steps 1 and 2 of a substep (module documentation): run the
    /// participants and apply their forces. Returns the overflow flag as it
    /// was at the start of the substep. Does nothing without participants.
    pub(super) fn participants_begin_substep(
        &mut self,
        index: usize,
        count: usize,
        h: Fix128,
        frozen: &mut [bool],
    ) -> bool {
        let at_start = self.overflow_detected;
        if frozen.is_empty() {
            return at_start;
        }
        #[cfg(feature = "std")]
        {
            let Some(plan) = self.participant_plan.as_ref() else {
                return at_start;
            };
            let mut forces = ForceAccumulator::new(self.bodies.len());
            let faults = run_substep(
                list_mut(&mut self.participants),
                plan,
                frozen,
                &self.bodies,
                &mut self.fields,
                &mut forces,
                SubstepTime { index, count, h },
            );
            match faults {
                Ok(faults) => {
                    for f in faults {
                        self.record_fault(f);
                    }
                }
                // Every input is built by the world for this substep and the
                // per-body fields were checked against the body count before
                // the step began (`check_step`), so the inputs always fit.
                Err(e) => unreachable!("participant inputs checked before the step: {e:?}"),
            }
            self.apply_participant_forces(&forces, h);
        }
        #[cfg(not(feature = "std"))]
        let _ = (index, count, h);
        at_start
    }

    /// Step 4 of a substep: record [`WorldFault::RigidOverflow`] when the
    /// overflow flag went up during the substep. Does nothing without
    /// participants.
    pub(super) fn participants_end_substep(&mut self, overflow_at_start: bool, frozen: &[bool]) {
        if frozen.is_empty() {
            return;
        }
        if !overflow_at_start && self.overflow_detected {
            self.record_fault(WorldFault::RigidOverflow);
        }
    }

    /// `v += F·inv_mass·h`, `ω += I⁻¹·τ·h` on every dynamic body. Every
    /// operation is checked, the intermediate `F·inv_mass` and `I⁻¹·τ`
    /// included (the products of [`Fix128`] wrap, so checking only the last
    /// `·h` would let an out-of-range intermediate through): a result out of
    /// range leaves the body as it was and records
    /// [`WorldFault::ForceOutOfRange`].
    #[cfg(feature = "std")]
    fn apply_participant_forces(&mut self, forces: &ForceAccumulator, h: Fix128) {
        for i in 0..self.bodies.len() {
            let (Some(force), Some(torque)) = (forces.force(i), forces.torque(i)) else {
                continue;
            };
            let body = &self.bodies[i];
            // Static bodies ignore forces; a kinematic body follows its
            // target, not forces.
            if !body.is_dynamic() {
                continue;
            }
            let parked = self.park.is_parked(i);
            if parked || self.islands.is_sleeping(i) {
                if !wakes_parked_body(body, force, torque, h, &self.islands.config) {
                    continue;
                }
                if parked {
                    self.islands.wake_body(i);
                    self.park.unpark(i, &mut self.stage_work);
                } else {
                    self.islands.wake_island(i);
                }
            }
            let body = &self.bodies[i];
            let v = force
                .checked_scale(body.inv_mass)
                .and_then(|a| a.checked_scale(h))
                .and_then(|dv| checked_add_vec(body.velocity, dv));
            let w = body
                .checked_world_inv_inertia_apply(torque)
                .and_then(|a| a.checked_scale(h))
                .and_then(|dw| checked_add_vec(body.angular_velocity, dw));
            match (v, w) {
                (Some(v), Some(w)) => {
                    let body = &mut self.bodies[i];
                    body.velocity = v;
                    body.angular_velocity = w;
                }
                _ => self.record_fault(WorldFault::ForceOutOfRange { body: i }),
            }
        }
    }
}

#[cfg(feature = "std")]
fn alloc_flags(n: usize) -> Vec<bool> {
    vec![false; n]
}
