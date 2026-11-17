//! A [`Cloth`] as a resident of the world: it steps inside the world's substeps
//! through the participant contract, like every other resident.
//!
//! # MODEL
//!
//! - **Equations**: the cloth's own position-based step ([`Cloth::step`]: stretch
//!   and bend constraints, rest tether, self contact, damping), run once per world
//!   step of the cloth's own rate: `m` steps per frame (the cloth's
//!   [`crate::cloth::ClothConfig::substeps`]), placed on the world's substeps by
//!   [`StepRule::PerFrame`], each of the width of the substeps it covers. So the
//!   cloth's steps, and its cost, do not depend on how many substeps the world cuts a
//!   frame into. The damping is a rate ([`crate::cloth::damping_over`]: the retention
//!   per 1/60 s raised to `h · 60`), so a second of motion decays the same at any `m`.
//! - **Units**: SI. Mass kg, length m, time s.
//! - **Conservation**: as [`Cloth::step`]; the participant adds nothing and takes
//!   nothing.
//! - **Range**: as [`Cloth::step`].
//! - **Contact with the bodies**: after each cloth step every body is a sphere of
//!   the world's collision radius `r` for it ([`SubstepCtx::body_radius`]); a particle
//!   closer than `R = r + thickness` to its centre is pushed out along the centre line
//!   by the depth `p`, the correction shared by inverse mass (`w_p` particle, `w_b`
//!   body, 0 for a static or kinematic body), its velocity relative to the body loses
//!   the inward normal part (inelastic, no friction), and a dynamic body is given the
//!   reaction `F = −n · p / ((w_p + w_b) · h²)` with `h` the width of the call, so it
//!   moves by its share `p · w_b / (w_p + w_b)` over the step.
//! - **Coupling limit**: the bodies are read at the start of the cloth's step, so a
//!   body crossing the cloth faster than the contact band can pass through it: a step
//!   catches a body moving at `|v_rel|` relative to the cloth while
//!   `|v_rel| · dt / m < 2 R` (the band is `2 R` thick at a particle); above that a
//!   body can go from one side to the other between two steps.
//! - **Approximations**: bodies are spheres of their collision radius (no box or
//!   mesh shapes), no friction, contact only with particles (not with the triangle
//!   faces between them); its own `gravity` is used.
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/cloth_resident_compare.rs` (bit equality with the
//!   world's own cloth step at one step per frame for any world substeps, the settle
//!   within 2 % across world substeps 1 / 2 / 4 / 8 for 1 / 2 / 4 steps per frame,
//!   rewind and fork bits).
//! - **Known defects**: none recorded.
//!
//! **Unstable.** This type exists to compare two ways of making cloth a resident
//! and may change or go away.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::cloth::{Cloth, Cursor};
use crate::math::{Fix128, Vec3Fix};
use crate::solver::BodyType;
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepRule,
    SubstepCtx,
};

/// Snapshot tag of [`ClothParticipant`]: the ASCII code `CLTH`, big endian.
pub const CLOTH_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"CLTH"));

/// Version of the payload [`ClothParticipant`] writes.
const CLOTH_STATE_VERSION: u32 = 1;

/// A cloth stepped by the world's substeps.
pub struct ClothParticipant {
    cloth: Cloth,
    /// The cloth's steps per frame ([`StepRule::PerFrame`])
    steps_per_frame: u32,
}

impl ClothParticipant {
    /// A resident for `cloth`.
    #[must_use]
    pub fn new(cloth: Cloth) -> Self {
        let mut cloth = cloth;
        let steps_per_frame = u32::try_from(cloth.config.substeps.max(1)).unwrap_or(u32::MAX);
        // the world places the steps, so each call is one step of the cloth
        cloth.config.substeps = 1;
        Self {
            cloth,
            steps_per_frame,
        }
    }

    /// The cloth as it is now.
    #[must_use]
    pub fn cloth(&self) -> &Cloth {
        &self.cloth
    }
}

impl Participant for ClothParticipant {
    fn kind(&self) -> ParticipantKind {
        CLOTH_PARTICIPANT_KIND
    }

    fn step_rule(&self) -> StepRule {
        StepRule::PerFrame(self.steps_per_frame)
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        if h <= Fix128::ZERO {
            return Err(ParticipantFault::InvalidState);
        }
        if ctx.bodies().is_empty() {
            self.cloth.step_by_its_own(h);
            return Ok(());
        }
        // work the contact out on a copy first: a fault leaves the cloth unchanged
        let mut next = self.cloth.clone();
        next.step_by_its_own(h);
        let reactions = contact(&mut next, ctx, h).ok_or(ParticipantFault::OutOfRange)?;
        for (body, force) in reactions {
            ctx.add_force(body, force)
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        self.cloth = next;
        Ok(())
    }

    fn observe(&self, _out: &mut ObservationSink) {}

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&CLOTH_STATE_VERSION.to_le_bytes());
        self.cloth.write_rollback(out);
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let mut cur = Cursor::new(bytes, 0);
        if cur.u32() != Some(CLOTH_STATE_VERSION) {
            return Err(StateError::InvalidValue);
        }
        let mut probe = self.cloth.clone();
        probe
            .read_rollback(&mut cur, false)
            .ok_or(StateError::InvalidValue)?;
        if cur.pos != bytes.len() {
            return Err(StateError::Length {
                expected: cur.pos,
                found: bytes.len(),
            });
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let mut cur = Cursor::new(bytes, 4);
        // check_state accepted these bytes, so the read cannot fail
        let _ = self.cloth.read_rollback(&mut cur, true);
    }
}

/// Push the cloth's particles out of every body (a sphere of the world's collision
/// radius) and return the reaction each dynamic body gets; `None` when a value
/// leaves the range of [`Fix128`].
fn contact(cloth: &mut Cloth, ctx: &SubstepCtx<'_>, h: Fix128) -> Option<Vec<(usize, Vec3Fix)>> {
    let thickness = cloth.config.thickness;
    let h_sq = h.checked_mul(h)?;
    let mut reactions = Vec::new();
    for (b, body) in ctx.bodies().iter().enumerate() {
        let radius = ctx.body_radius(b)?;
        let reach = radius.checked_add(thickness)?;
        let wb = if body.body_type == BodyType::Dynamic {
            body.inv_mass
        } else {
            Fix128::ZERO
        };
        let mut push = Vec3Fix::ZERO;
        for i in 0..cloth.positions.len() {
            let wp = cloth.inv_masses[i];
            let weight = wp.checked_add(wb)?;
            if weight <= Fix128::ZERO {
                continue;
            }
            let d = cloth.positions[i] - body.position;
            let dist = d.checked_length()?;
            if dist >= reach || dist.is_zero() {
                continue;
            }
            let n = d / dist;
            let depth = reach - dist;
            let share = depth.checked_mul(wp)?.checked_div(weight)?;
            cloth.positions[i] = cloth.positions[i] + n * share;
            let vn = (cloth.velocities[i] - body.velocity).dot(n);
            if vn < Fix128::ZERO {
                cloth.velocities[i] = cloth.velocities[i] - n * vn;
            }
            if !wb.is_zero() {
                let f = depth.checked_div(weight.checked_mul(h_sq)?)?;
                push = push - n * f;
            }
        }
        if push != Vec3Fix::ZERO {
            reactions.push((b, push));
        }
    }
    Some(reactions)
}
