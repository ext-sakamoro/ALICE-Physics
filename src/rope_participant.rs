//! # MODEL
//!
//! - **Equations**: position-based distance constraints between neighbouring particles
//!   of a lumped-mass chain, solved by a Gauss-Seidel sweep in each world substep, under
//!   the world's gravity. A particle tied to a body stands for the anchor with the
//!   weight `1/m + (r × n) · I⁻¹ (r × n)`; the correction `dx` it receives is staged on
//!   the body as the force `F = dx / h²` (the rope side's mass) with the torque `r × F`.
//! - **Units**: SI. Mass kg, length m, time s, force N, torque N·m.
//! - **Conservation**: with no external force the total momentum of rope and bodies
//!   stays (a staged force is the reaction of the constraint's own correction); the
//!   projection dissipates energy and never adds it over long undamped runs.
//! - **Range**: rope:body mass ratios from 0.01:1 to 1000:1 hold the rope's length
//!   within 5 % and the energy within 1 %; at 0.001:1 the sweep does not converge (see
//!   the known defect). Every product and sum is checked: a value out of range is
//!   [`ParticipantFault::OutOfRange`] and changes nothing.
//! - **Approximations**: inextensible distance constraints only (no stretch modulus,
//!   no bending or torsion stiffness), no collision of the rope with itself or the
//!   world, no air drag on the rope, a fixed sweep per substep (the stiffness follows
//!   the world's substeps).
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/analytic_rope_participant.rs::a_body_on_a_rope_swings_as_a_pendulum::a_hanging_rope_pulls_with_the_weight_it_carries::the_centre_of_mass_falls_freely::total_momentum_stays_with_no_external_force::energy_does_not_grow_over_a_long_undamped_run::the_mass_ratio_stays_bounded::the_staged_torque_is_the_offset_cross_the_force::an_offset_anchor_turns_the_body_under_the_rope::rewind_and_fork_are_bit_identical::swing_scene_digest_is_pinned`
//!   (the pendulum period, the tension of a hanging load, free fall of the centre of
//!   mass, momentum, energy, the mass-ratio range, the torque `r × F`, replay bits).
//! - **Known defects**: `tests/analytic_rope_participant.rs::a_heavy_load_on_a_light_rope_holds_its_length` (at
//!   rope:body 0.001:1 the length is 5 % off at frame 9).
//!
//! A [`Rope`] as a resident of the world: its particles step inside the world's
//! substeps, and it pulls the bodies it is tied to through the participant
//! contract.
//!
//! A particle tied to a body has no mass of its own: within the substep it stands
//! for the anchor, with the body's inverse mass (and the turn of the anchor's
//! offset) as its weight, starts from where the anchor is heading and is moved by
//! the distance constraints like any other. What the constraints do to it is what
//! the body must feel: a correction `dx` for the mass of the rope side is the force
//! `dx / h^2`, staged with the torque `r x F`. The world applies staged forces
//! before it integrates, so the body makes the same move in the same substep. The
//! particle's own mass from [`Rope::new`] is not used.
//!
//! The rope's own `substeps` and `gravity` are not used: the world's substeps and
//! its gravity as it is in the substep decide. Every
//! product and sum is checked: a value out of range is
//! [`ParticipantFault::OutOfRange`] and changes nothing.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::hash::Hasher;

use crate::buoyancy_zone::hash_fix;
use crate::det_hash::DetHasher;
use crate::math::{Fix128, Vec3Fix};
use crate::rope::Rope;
use crate::solver::{BodyType, RigidBody};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

/// Snapshot tag of [`RopeParticipant`]: the ASCII code `ROPE`, big endian.
pub const ROPE_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"ROPE"));

/// First observation channel: the pull at anchor `i` is channel `i`.
pub const ROPE_OBS_TENSION: u32 = 0;

/// Payload layout version written by [`RopeParticipant`].
const ROPE_STATE_VERSION: u32 = 1;
/// `version: u32`, `digest: u64`, particle count `u32`, anchor count `u32`.
const HEADER: usize = 20;
/// One `Fix128`.
const FIX: usize = 16;
/// One particle: position and velocity.
const PARTICLE: usize = 6 * FIX;

/// Where one particle of the rope is tied to one body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeAnchor {
    /// Particle index in the rope.
    pub particle: usize,
    /// Body index in the world.
    pub body: usize,
    /// The anchor in the body's frame.
    pub local_anchor: Vec3Fix,
}

/// Why a rope cannot be a resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeAnchorError {
    /// An anchor names a particle the rope does not have.
    ParticleOutOfRange,
    /// Two anchors tie the same particle.
    ParticleTiedTwice,
    /// The rope has fixed pins: a resident is held by anchors only.
    PinsAreNotResidents,
}

/// The participant.
pub struct RopeParticipant {
    rope: Rope,
    anchors: Vec<RopeAnchor>,
    /// The size of the pull at each anchor in the last substep.
    tensions: Vec<Fix128>,
    digest: u64,
}

fn hash_vec(h: &mut DetHasher, v: Vec3Fix) {
    hash_fix(h, v.x);
    hash_fix(h, v.y);
    hash_fix(h, v.z);
}

fn checked_cross(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        a.y.checked_mul(b.z)?.checked_sub(a.z.checked_mul(b.y)?)?,
        a.z.checked_mul(b.x)?.checked_sub(a.x.checked_mul(b.z)?)?,
        a.x.checked_mul(b.y)?.checked_sub(a.y.checked_mul(b.x)?)?,
    ))
}

/// A tied particle's view of its anchor in this substep.
#[derive(Clone, Copy)]
struct Tie {
    /// World offset of the anchor from the body's centre.
    offset: Vec3Fix,
    /// Whether the body takes forces.
    dynamic: bool,
    /// Index into `anchors`.
    anchor: usize,
}

impl Tie {
    /// Weight of this end along the unit direction `n`: `1/m + (r x n)·I⁻¹(r x n)`.
    fn weight(&self, body: &RigidBody, n: Vec3Fix) -> Option<Fix128> {
        if !self.dynamic {
            return Some(Fix128::ZERO);
        }
        let arm = checked_cross(self.offset, n)?;
        let turn = body
            .checked_world_inv_inertia_apply(arm)?
            .checked_dot(arm)?;
        body.inv_mass.checked_add(turn)
    }
}

impl RopeParticipant {
    /// The participant for `rope` tied by `anchors`.
    ///
    /// # Errors
    ///
    /// See [`RopeAnchorError`].
    pub fn new(rope: Rope, anchors: Vec<RopeAnchor>) -> Result<Self, RopeAnchorError> {
        if !rope.pins.is_empty() {
            return Err(RopeAnchorError::PinsAreNotResidents);
        }
        for (i, a) in anchors.iter().enumerate() {
            if a.particle >= rope.particle_count() {
                return Err(RopeAnchorError::ParticleOutOfRange);
            }
            if anchors[..i].iter().any(|b| b.particle == a.particle) {
                return Err(RopeAnchorError::ParticleTiedTwice);
            }
        }
        let mut h = DetHasher::default();
        for f in &rope.rest_lengths {
            hash_fix(&mut h, *f);
        }
        for f in &rope.inv_masses {
            hash_fix(&mut h, *f);
        }
        hash_fix(&mut h, rope.config.damping);
        hash_fix(&mut h, rope.config.compliance);
        h.write_u64(rope.config.iterations as u64);
        for a in &anchors {
            h.write_u64(a.particle as u64);
            h.write_u64(a.body as u64);
            hash_vec(&mut h, a.local_anchor);
        }
        let tensions = vec![Fix128::ZERO; anchors.len()];
        Ok(Self {
            rope,
            anchors,
            tensions,
            digest: h.finish(),
        })
    }

    /// The positions and velocities of the particles in a payload written by
    /// [`Participant::write_state`], `None` for any other bytes.
    #[must_use]
    pub fn particles(state: &[u8]) -> Option<Vec<(Vec3Fix, Vec3Fix)>> {
        if state.len() < HEADER || u32_at(state, 0) != ROPE_STATE_VERSION {
            return None;
        }
        let n = u32_at(state, 12) as usize;
        let a = u32_at(state, 16) as usize;
        if state.len() != payload_len(n, a) {
            return None;
        }
        Some(
            (0..n)
                .map(|i| {
                    let at = HEADER + i * PARTICLE;
                    (vec_at(state, at), vec_at(state, at + 3 * FIX))
                })
                .collect(),
        )
    }

    /// The whole substep, on copies: `None` when a value leaves the range.
    #[allow(
        clippy::type_complexity,
        reason = "the three results of one substep, used once by the caller"
    )]
    fn solve(
        &self,
        bodies: &[RigidBody],
        g: Vec3Fix,
        h: Fix128,
        last_substep: bool,
    ) -> Option<(Vec<Vec3Fix>, Vec<Vec3Fix>, Vec<Vec3Fix>)> {
        let n = self.rope.particle_count();
        let mut ties: Vec<Option<Tie>> = vec![None; n];
        let mut pos = self.rope.positions.clone();
        let mut prev = self.rope.positions.clone();
        let mut vel = self.rope.velocities.clone();
        for (anchor, a) in self.anchors.iter().enumerate() {
            let body = &bodies[a.body];
            let offset = body.rotation.rotate_vec(a.local_anchor);
            let dynamic = body.body_type == BodyType::Dynamic && !body.inv_mass.is_zero();
            ties[a.particle] = Some(Tie {
                offset,
                dynamic,
                anchor,
            });
            let here = body.position.checked_add(offset)?;
            let turn = checked_cross(body.angular_velocity, offset)?;
            let mut heading = body.velocity.checked_add(turn)?;
            if dynamic {
                heading =
                    heading.checked_add(g.checked_scale(body.gravity_scale)?.checked_scale(h)?)?;
            }
            prev[a.particle] = here;
            pos[a.particle] = here.checked_add(heading.checked_scale(h)?)?;
        }
        for i in 0..n {
            if ties[i].is_some() || self.rope.inv_masses[i].is_zero() {
                continue;
            }
            vel[i] = vel[i].checked_add(g.checked_scale(h)?)?;
            pos[i] = pos[i].checked_add(vel[i].checked_scale(h)?)?;
        }

        let h_sq = h.checked_mul(h)?;
        let alpha = self.rope.config.compliance.checked_div(h_sq)?;
        let mut pulls = vec![Vec3Fix::ZERO; self.anchors.len()];
        for _ in 0..self.rope.config.iterations {
            for s in 0..self.rope.segment_count() {
                let delta = pos[s + 1].checked_sub(pos[s])?;
                let dist = delta.checked_length()?;
                if dist.is_zero() {
                    continue;
                }
                let dir = Vec3Fix::new(
                    delta.x.checked_div(dist)?,
                    delta.y.checked_div(dist)?,
                    delta.z.checked_div(dist)?,
                );
                let weight = |i: usize| match ties[i] {
                    Some(t) => t.weight(&bodies[self.anchors[t.anchor].body], dir),
                    None => Some(self.rope.inv_masses[i]),
                };
                let (w0, w1) = (weight(s)?, weight(s + 1)?);
                let w_sum = w0.checked_add(w1)?;
                if w_sum.is_zero() {
                    continue;
                }
                let error = dist.checked_sub(self.rope.rest_lengths[s])?;
                let lambda = error.checked_div(w_sum.checked_add(alpha)?)?;
                let correction = dir.checked_scale(lambda)?;
                pos[s] = pos[s].checked_add(correction.checked_scale(w0)?)?;
                pos[s + 1] = pos[s + 1].checked_sub(correction.checked_scale(w1)?)?;
                if let Some(t) = ties[s] {
                    pulls[t.anchor] = pulls[t.anchor].checked_add(correction)?;
                }
                if let Some(t) = ties[s + 1] {
                    pulls[t.anchor] = pulls[t.anchor].checked_sub(correction)?;
                }
            }
        }

        let d = self.rope.config.damping;
        for i in 0..n {
            let moved = pos[i].checked_sub(prev[i])?;
            let v = Vec3Fix::new(
                moved.x.checked_div(h)?,
                moved.y.checked_div(h)?,
                moved.z.checked_div(h)?,
            );
            vel[i] = if last_substep && ties[i].is_none() {
                v.checked_scale(d)?
            } else {
                v
            };
        }
        let forces = pulls
            .iter()
            .map(|p| {
                Some(Vec3Fix::new(
                    p.x.checked_div(h_sq)?,
                    p.y.checked_div(h_sq)?,
                    p.z.checked_div(h_sq)?,
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        Some((pos, vel, forces))
    }
}

impl Participant for RopeParticipant {
    fn kind(&self) -> ParticipantKind {
        ROPE_PARTICIPANT_KIND
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        if self.anchors.iter().any(|a| a.body >= ctx.bodies().len()) {
            return Err(ParticipantFault::InvalidState);
        }
        // Work the whole substep out first: a value out of range leaves the
        // world's force buffer and this participant as they were.
        let last_substep = ctx.substep_index() + 1 == ctx.substeps();
        let g = ctx.gravity().ok_or(ParticipantFault::InvalidState)?;
        let (pos, vel, forces) = self
            .solve(ctx.bodies(), g, h, last_substep)
            .ok_or(ParticipantFault::OutOfRange)?;
        let mut staged = Vec::with_capacity(self.anchors.len());
        let mut tensions = Vec::with_capacity(self.anchors.len());
        for (a, f) in self.anchors.iter().zip(&forces) {
            let body = &ctx.bodies()[a.body];
            let offset = body.rotation.rotate_vec(a.local_anchor);
            let torque = checked_cross(offset, *f).ok_or(ParticipantFault::OutOfRange)?;
            tensions.push(f.checked_length().ok_or(ParticipantFault::OutOfRange)?);
            if body.body_type == BodyType::Dynamic && !body.inv_mass.is_zero() {
                staged.push((a.body, *f, torque));
            }
        }
        for (i, f, t) in staged {
            ctx.add_force(i, f)
                .map_err(|_| ParticipantFault::InvalidState)?;
            ctx.add_torque(i, t)
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        self.rope.positions = pos;
        self.rope.velocities = vel;
        self.tensions = tensions;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        for (i, t) in self.tensions.iter().enumerate() {
            out.push(ROPE_OBS_TENSION + i as u32, *t);
        }
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&ROPE_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&(self.rope.particle_count() as u32).to_le_bytes());
        out.extend_from_slice(&(self.anchors.len() as u32).to_le_bytes());
        for (p, v) in self.rope.positions.iter().zip(&self.rope.velocities) {
            write_vec(out, *p);
            write_vec(out, *v);
        }
        for t in &self.tensions {
            write_fix(out, *t);
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let expected = payload_len(self.rope.particle_count(), self.anchors.len());
        if bytes.len() != expected {
            return Err(StateError::Length {
                expected,
                found: bytes.len(),
            });
        }
        if u32_at(bytes, 0) != ROPE_STATE_VERSION
            || u64_at(bytes, 4) != self.digest
            || u32_at(bytes, 12) as usize != self.rope.particle_count()
            || u32_at(bytes, 16) as usize != self.anchors.len()
        {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let n = self.rope.particle_count();
        for i in 0..n {
            let at = HEADER + i * PARTICLE;
            self.rope.positions[i] = vec_at(bytes, at);
            self.rope.velocities[i] = vec_at(bytes, at + 3 * FIX);
        }
        for (i, t) in self.tensions.iter_mut().enumerate() {
            *t = fix_at(bytes, HEADER + n * PARTICLE + i * FIX);
        }
    }
}

fn payload_len(particles: usize, anchors: usize) -> usize {
    HEADER + particles * PARTICLE + anchors * FIX
}

fn write_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn write_vec(out: &mut Vec<u8>, v: Vec3Fix) {
    write_fix(out, v.x);
    write_fix(out, v.y);
    write_fix(out, v.z);
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(b)
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

fn fix_at(bytes: &[u8], at: usize) -> Fix128 {
    Fix128 {
        hi: u64_at(bytes, at) as i64,
        lo: u64_at(bytes, at + 8),
    }
}

fn vec_at(bytes: &[u8], at: usize) -> Vec3Fix {
    Vec3Fix::new(
        fix_at(bytes, at),
        fix_at(bytes, at + FIX),
        fix_at(bytes, at + 2 * FIX),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_participant::{ForceAccumulator, ObservationSink};

    /// A rope from a fixed body at the origin down to a dynamic body.
    fn hanging() -> (RopeParticipant, [RigidBody; 2]) {
        let i = Fix128::from_int;
        let rope = Rope::new(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, i(-2), Fix128::ZERO),
            4,
            Fix128::ONE,
        );
        let anchors = vec![
            RopeAnchor {
                particle: 0,
                body: 0,
                local_anchor: Vec3Fix::ZERO,
            },
            RopeAnchor {
                particle: 4,
                body: 1,
                local_anchor: Vec3Fix::ZERO,
            },
        ];
        let bodies = [
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, i(-2), Fix128::ZERO), Fix128::ONE),
        ];
        (
            RopeParticipant::new(rope, anchors).expect("anchors"),
            bodies,
        )
    }

    /// The anchors are checked: a pinned rope, a particle past the end and a
    /// particle tied twice are refused.
    #[test]
    fn the_anchors_are_checked() {
        let i = Fix128::from_int;
        let line = || {
            Rope::new(
                Vec3Fix::ZERO,
                Vec3Fix::new(i(1), i(0), i(0)),
                2,
                Fix128::ONE,
            )
        };
        let tie = |particle| RopeAnchor {
            particle,
            body: 0,
            local_anchor: Vec3Fix::ZERO,
        };
        let mut pinned = line();
        pinned.pin_start();
        assert_eq!(
            RopeParticipant::new(pinned, vec![]).err(),
            Some(RopeAnchorError::PinsAreNotResidents)
        );
        assert_eq!(
            RopeParticipant::new(line(), vec![tie(3)]).err(),
            Some(RopeAnchorError::ParticleOutOfRange)
        );
        assert_eq!(
            RopeParticipant::new(line(), vec![tie(1), tie(1)]).err(),
            Some(RopeAnchorError::ParticleTiedTwice)
        );
    }

    /// Under gravity the hanging rope pulls the dynamic body up and stages
    /// nothing on the fixed body; one tension is observed per anchor; the
    /// snapshot reads back and refuses another length; a context without the
    /// world's gravity or with too few bodies is an invalid state.
    #[test]
    fn the_substep_pulls_the_tied_body_and_the_snapshot_round_trips() {
        let (mut p, bodies) = hanging();
        let h = Fix128::from_ratio(1, 240);
        let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-9.81), Fix128::ZERO);
        let mut forces = ForceAccumulator::new(2);
        for k in 0..4 {
            // one buffer per substep, as the world does
            forces = ForceAccumulator::new(2);
            let mut ctx = SubstepCtx::new(&bodies, &mut forces, k, 4, h)
                .expect("ctx")
                .with_gravity(g);
            p.substep(&mut ctx, h).expect("substep");
        }
        assert_eq!(forces.force(0), Some(Vec3Fix::ZERO));
        let pull = forces.force(1).expect("slot");
        assert!(
            pull.y > Fix128::ZERO,
            "the rope holds the body up: {pull:?}"
        );
        let mut sink = ObservationSink::new();
        p.observe(&mut sink);
        assert_eq!(sink.values().len(), 2);
        assert_eq!(sink.values()[1], (ROPE_OBS_TENSION + 1, pull.length()));

        let mut bytes = Vec::new();
        p.write_state(&mut bytes);
        assert_eq!(bytes.len(), payload_len(5, 2));
        let (mut fresh, _) = hanging();
        assert_eq!(fresh.check_state(&bytes), Ok(()));
        fresh.read_state(&bytes);
        let mut again = Vec::new();
        fresh.write_state(&mut again);
        assert_eq!(again, bytes);
        let particles = RopeParticipant::particles(&bytes).expect("rope payload");
        assert_eq!(particles.len(), 5);
        assert_eq!(particles[0].0, Vec3Fix::ZERO);
        assert_eq!(RopeParticipant::particles(&bytes[..10]), None);
        assert!(matches!(
            fresh.check_state(&bytes[..bytes.len() - 1]),
            Err(StateError::Length { .. })
        ));
        let mut other = bytes.clone();
        other[4] ^= 1;
        assert_eq!(fresh.check_state(&other), Err(StateError::InvalidValue));

        let mut forces = ForceAccumulator::new(2);
        let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h).expect("ctx");
        assert_eq!(
            fresh.substep(&mut ctx, h),
            Err(ParticipantFault::InvalidState)
        );
        let one = [bodies[0]];
        let mut forces = ForceAccumulator::new(1);
        let mut ctx = SubstepCtx::new(&one, &mut forces, 0, 1, h)
            .expect("ctx")
            .with_gravity(g);
        assert_eq!(
            fresh.substep(&mut ctx, h),
            Err(ParticipantFault::InvalidState)
        );
    }
}
