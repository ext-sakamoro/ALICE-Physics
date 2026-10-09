//! Reading the contacts a step resolved against SDF colliders.
//!
//! A ball dropped onto an SDF floor. After every step the world's record
//! (`PhysicsWorld::last_step_sdf_contacts`) says which body touched which
//! collider, where, along which normal, how deep and how fast it was moving
//! into the surface; a caller can turn that into damage and reshape the field
//! between steps. A participant inside the substep loop reads the same record
//! (`SubstepCtx::sdf_contacts`): the contacts of the earlier substeps of the
//! step.
//!
//! The example checks itself: the first impact speed against the closed form
//! of the integration, and the participant's view against the world record.
//!
//! ```bash
//! cargo run --example sdf_contacts --features std
//! ```

use std::sync::{Arc, Mutex};

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

const SUBSTEPS: usize = 8;
const GRAVITY: i64 = -10;

/// Counts the SDF contacts it reads, once each: the record a substep sees
/// holds every contact of the step so far, so only the entries beyond the
/// ones already counted in this step are new.
struct ContactCounter {
    counted_this_step: usize,
    total: Arc<Mutex<usize>>,
}

impl Participant for ContactCounter {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x5344_4643)
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _h: Fix128) -> Result<(), ParticipantFault> {
        if ctx.substep_index() == 0 {
            self.counted_this_step = 0;
        }
        let seen = ctx.sdf_contacts().len();
        let new = seen - self.counted_this_step;
        self.counted_this_step = seen;
        *self
            .total
            .lock()
            .map_err(|_| ParticipantFault::InvalidState)? += new;
        Ok(())
    }

    fn observe(&self, _out: &mut ObservationSink) {}

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.counted_this_step as u64).to_le_bytes());
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 8 {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 8,
                found: bytes.len(),
            })
        }
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        self.counted_this_step = u64::from_le_bytes(raw) as usize;
    }
}

fn main() {
    let config = PhysicsConfig {
        substeps: SUBSTEPS,
        gravity: Vec3Fix::from_int(0, GRAVITY, 0),
        // No frame damping, so the fall follows `v = g·h·n` exactly.
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_sdf_collision_radius(Fix128::from_ratio(1, 2));

    // The floor `y = 0` as a field.
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    // A ball of radius 1/2 resting 1.25 above the floor.
    let ball = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(7, 4), Fix128::ZERO),
        Fix128::ONE,
    ));

    let counted = Arc::new(Mutex::new(0usize));
    world
        .add_participant(Box::new(ContactCounter {
            counted_this_step: 0,
            total: Arc::clone(&counted),
        }))
        .expect("register the participant");

    let dt = Fix128::from_ratio(1, 60);
    let h = 1.0 / (60.0 * SUBSTEPS as f64);
    let mut recorded = 0usize;
    let mut in_last_substep = 0usize;
    let mut first_impact = None;
    for frame in 0..120 {
        world.step(dt);
        let contacts = world.last_step_sdf_contacts();
        for c in contacts {
            assert_eq!(c.body_index, ball);
            assert_eq!(c.collider_index, 0);
            assert!(c.depth > Fix128::ZERO);
            if c.substep == SUBSTEPS - 1 {
                in_last_substep += 1;
            }
            if first_impact.is_none() {
                first_impact = Some((frame, *c));
            }
        }
        recorded += contacts.len();
    }

    let (frame, impact) = first_impact.expect("the ball reaches the floor");
    // Closed form before the first contact: after n substeps from rest,
    // v = g·h·n, so the first impact speed is -g·h·n with n counted to the
    // substep the contact was resolved in.
    let n = frame * SUBSTEPS + impact.substep + 1;
    let expected = -(GRAVITY as f64) * h * n as f64;
    let speed = impact.approach_speed.to_f64();
    println!(
        "first impact: frame {frame}, substep {}, speed {speed:.6} m/s (closed form {expected:.6}), \
         depth {:.6} m, normal {:?}",
        impact.substep,
        impact.depth.to_f64(),
        impact.normal.to_f32(),
    );
    assert!(
        (speed - expected).abs() < 1e-9,
        "impact speed {speed} vs {expected}"
    );
    let (nx, ny, nz) = impact.normal.to_f32();
    assert!(nx == 0.0 && ny == 1.0 && nz == 0.0, "floor normal");
    let (_, py, _) = impact.point.to_f32();
    assert!(py.abs() < 1e-6, "contact point on the floor, y = {py}");

    // The participant sees every contact except those of the last substep of
    // each step (it runs before that substep's resolution, and the record is
    // emptied at the head of the next step).
    let counted = *counted.lock().expect("lock");
    println!("recorded {recorded} contacts, participant read {counted} ({in_last_substep} in last substeps)");
    assert!(recorded > 0);
    assert_eq!(counted, recorded - in_last_substep);
}
