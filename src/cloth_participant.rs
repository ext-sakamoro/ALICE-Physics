//! A [`Cloth`] as a resident of the world: it steps inside the world's substeps
//! through the participant contract, like every other resident.
//!
//! # MODEL
//!
//! - **Equations**: the cloth's own position-based step ([`Cloth::step`]: stretch
//!   and bend constraints, rest tether, self contact, damping), run once per world
//!   step of the cloth's own rate: `m` steps per frame (the cloth's
//!   [`crate::cloth::ClothConfig::substeps`]), placed on the world's substeps by
//!   [`StepRule::PerFrame`](crate::world_participant::StepRule::PerFrame), each of
//!   the width of the substeps it covers. So the cloth's steps, and its cost, do not depend on how many substeps the world cuts a
//!   frame into. The damping is a rate ([`crate::cloth::damping_over`]: the retention
//!   per 1/60 s raised to `h · 60`), so a second of motion decays the same at any `m`.
//! - **Units**: SI. Mass kg, length m, time s.
//! - **Contact with the world's bodies**: every body is a sphere of its world
//!   collision radius ([`SubstepCtx::body_radius`](crate::world_participant::SubstepCtx::body_radius)), reach `R = radius + thickness`.
//!   The call first predicts where the world will have each dynamic body at the end
//!   of the call with no cloth: `k` steps of `solver::free_linear_substep`
//!   (the same function the world's integration calls, with the body's
//!   `gravity_scale`), `k` the world substeps the call covers and `hs` the world's
//!   substep width ([`SubstepCtx::world_substep`](crate::world_participant::SubstepCtx::world_substep)). Inside the cloth's iterations each
//!   sphere is held at `R` from the faces it was on the start side of
//!   (particle–triangle: closest point, normal from the face, the side fixed by the
//!   body's start position; a closest point on an edge or vertex that a
//!   neighbouring face owns, the projection of the centre being inside that face,
//!   is left to that face and its normal, so a flat sheet's internal edges do not
//!   push along the surface; the boundary keeps the rounded direction `c − q`),
//!   a mass-weighted XPBD projection that moves the sphere
//!   by `w_s λ` and corner `i` by `−w_i b_i λ` (`w = inv_mass`, `b` barycentric).
//!   Coulomb friction acts in the same contact with the rigid static-friction law
//!   (`xpbd_contact::static_hold`): the tangential slip of the sphere relative to the
//!   face's material point is undone while the accumulated tangential multiplier
//!   stays within `μ` times the accumulated normal one, `μ` the body's `friction`.
//!   The force acts at the centre: no torque (a sliding block, not a rolling ball).
//!   Static and kinematic bodies take no reaction (`w_s = 0`).
//! - **Reaction**: the sphere's displacement from its prediction, `Δc`, is returned to
//!   the body as one force at the first substep of the call: the world applies it
//!   once, `Δv = F · inv_mass · hs`, and the body carries `Δv` through the `k`
//!   substeps, so `F = Δc / (inv_mass · hs · k·hs)` moves it by `Δc` and gives it
//!   the impulse `M·Δc/(k·hs)`, the momentum the cloth's position-based step takes
//!   in the other direction. Of the distributions of one force over the span this
//!   is the only one that matches both the displacement and the impulse of a
//!   position-based partner (a force spread over the `k` substeps moves the body
//!   `(k+1)/(2k)` as far for the same impulse).
//! - **Span sawtooth**: at rest the span's net displacement is 0, so the body leaves
//!   the pulse with `u = −g·hs·(k+1)/2` and after substep `j` has
//!   `u + j·g·hs = g·hs·(j − (k+1)/2)`: its speed reaches `(k−1)/2 · |g| · hs` at the
//!   ends of each span while its position stays put (0 for `k = 1`).
//! - **Conservation**: as [`Cloth::step`]; the contact conserves the momentum of
//!   cloth and bodies together (equal and opposite mass-weighted moves, the impulse
//!   above).
//! - **Range**: as [`Cloth::step`]. A body whose prediction leaves the range faults
//!   the call ([`ParticipantFault::OutOfRange`](crate::world_participant::ParticipantFault::OutOfRange)).
//! - **Approximations**: a body that crosses a face's plane within one cloth step is
//!   held on its start side (the side is read from the start position, so a body
//!   that moves more than its reach plus the cloth's motion in one step against a
//!   face it was not near is not caught). Bodies are spheres; no torque. The cloth's
//!   own `gravity` is used.
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/cloth_resident_compare.rs` (bit equality with the
//!   world's own cloth step at one step per frame with no body in reach, the settle
//!   within 2 % across world substeps, including counts the steps do not divide,
//!   rewind and fork bits); `tests/cloth_body_contact.rs`
//!   (`a_dropped_ball_comes_to_rest_on_the_sheet_without_sinking_or_drifting`, rest
//!   with no drift and the span sawtooth bound at every substep;
//!   `a_ball_smaller_than_a_cell_is_held_by_its_faces`;
//!   `a_ball_crossing_a_face_within_one_step_is_held_on_its_start_side`;
//!   `a_ball_over_the_sheets_boundary_is_caught_by_the_edge_and_rolls_off`;
//!   `a_ball_on_an_incline_holds_below_tan_mu_and_slides_above_it`;
//!   `ball_and_cloth_momentum_together_is_conserved_through_the_contact`).
//!
//! Unstable like the rest of the participant API: it may change before that API is
//! declared stable.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::cloth::{Cloth, ContactSphere, Cursor};
use crate::math::{Fix128, Vec3Fix};
use crate::solver::free_linear_substep;
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
}

impl ClothParticipant {
    /// The world's bodies as contact spheres for a step of width `h`: each body of a
    /// known collision radius, its centre at the step start and predicted over the
    /// step under the world's gravity (`x + (v + g h) h`), the reach its radius plus
    /// the cloth's thickness, the inverse mass its own when it is dynamic (a static or
    /// kinematic body is not moved). Returns the spheres and, per sphere, the body
    /// index and the predicted centre the reaction is measured from.
    fn spheres(
        &self,
        ctx: &SubstepCtx<'_>,
        h: Fix128,
        covered: usize,
        hs: Fix128,
    ) -> Result<(Vec<ContactSphere>, Vec<(usize, Vec3Fix)>), ParticipantFault> {
        let g = ctx.gravity().unwrap_or(Vec3Fix::ZERO);
        let thickness = self.cloth.config.thickness;
        let mut spheres = Vec::new();
        let mut predicted = Vec::new();
        for (i, b) in ctx.bodies().iter().enumerate() {
            let Some(radius) = ctx.body_radius(i) else {
                continue;
            };
            let dynamic = b.is_dynamic();
            // where the world will have the body at the end of this call with
            // no cloth: its own `k` substeps of `free_linear_substep`; a
            // static or kinematic body is not moved by the cloth, its centre
            // is carried on its velocity
            let centre = if dynamic {
                let (mut x, mut v) = (b.position, b.velocity);
                for _ in 0..covered {
                    let (nv, nx) = free_linear_substep(x, v, g, b.gravity_scale, hs);
                    v = nv;
                    x = nx.ok_or(ParticipantFault::OutOfRange)?;
                }
                x
            } else {
                b.position + b.velocity * h
            };
            spheres.push(ContactSphere {
                start: b.position,
                centre,
                reach: radius + thickness,
                inv_mass: if dynamic { b.inv_mass } else { Fix128::ZERO },
                friction: b.friction,
            });
            predicted.push((i, centre));
        }
        Ok((spheres, predicted))
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
        // the world integrates its bodies `k` times over this call, each a
        // world substep `hs` (its own rounding of `dt / n`)
        let covered = self
            .step_rule()
            .covers(ctx.substep_index(), ctx.substeps())
            .unwrap_or(1);
        let hs = ctx
            .world_substep()
            .unwrap_or_else(|| h / Fix128::from_int(covered as i64));
        let span = hs * Fix128::from_int(covered as i64);
        let (mut spheres, predicted) = self.spheres(ctx, h, covered, hs)?;
        self.cloth.step_with_spheres(h, &mut spheres);
        // the reaction: the impulse that moves the body by what the cloth moved
        // it. The world applies a staged force once, at the first substep of the
        // call, `Δv = F·inv_mass·hs`, and the body carries `Δv` through the `k`
        // substeps, `Δx = Δv·k·hs`; so `F = Δc / (inv_mass · hs · k·hs)`
        for (k, (body, pred)) in predicted.iter().enumerate() {
            let s = spheres[k];
            let moved = s.centre - *pred;
            if s.inv_mass.is_zero() || moved == Vec3Fix::ZERO {
                continue;
            }
            let scale = Fix128::ONE / (s.inv_mass * span * hs);
            ctx.add_force(*body, moved * scale)
                .map_err(|_| ParticipantFault::OutOfRange)?;
        }
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

#[cfg(all(test, feature = "std"))]
#[path = "../tests/lib/cloth_participant_oracles.rs"]
mod lib_oracles;
