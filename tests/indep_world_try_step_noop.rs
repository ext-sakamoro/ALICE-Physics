//! Participants that do nothing leave the world exactly as the plain step
//! does, on a scene with gravity, contacts, a ball-joint chain, unequal
//! masses and sleeping turned on, so that bodies go
//! to sleep and a zero force has to leave them asleep.
//!
//! Compared byte for byte, in every one of 240 frames, against the same scene
//! stepped with [`PhysicsWorld::step`] (or `step_parallel`) and nothing
//! registered:
//!
//! * a world where every registration was refused (zero participants through
//!   the participant API);
//! * a [`DragMedium`] coupled to every body with `c = 0` (it stages a zero
//!   force on every dynamic body in every substep), with a moving medium;
//! * a participant that stages a zero force and a zero torque on every body,
//!   the static one included;
//! * a [`DragMedium`] with no coupling.

#![cfg(feature = "std")]

use alice_physics::coupling_medium::DragMedium;
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::sleeping::SleepConfig;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, RegisterError, StateError,
    StepRule, SubstepCtx,
};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const FRAMES: usize = 240;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Stages a zero force and a zero torque on every body.
struct ZeroPush;

impl Participant for ZeroPush {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"ZPSH"))
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        for i in 0..ctx.bodies().len() {
            ctx.add_force(i, Vec3Fix::ZERO)
                .map_err(|_| ParticipantFault::InvalidState)?;
            ctx.add_torque(i, Vec3Fix::ZERO)
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// A participant whose fixed step registration refuses.
struct Refused(Fix128);

impl Participant for Refused {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"RFSD"))
    }
    fn step_rule(&self) -> StepRule {
        StepRule::Fixed(self.0)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        panic!("a refused participant must never run");
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// A static floor sphere, a chain of three bodies of masses `1 : 7 : 0.3`
/// hanging from it by ball joints, and two spheres (`2.5`, `0.75`) dropped
/// on the floor that come to rest and go to sleep. Substeps 3 (not a power
/// of two, so the TGS width differs from `dt / 3`).
fn scene(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 3,
        iterations: 5,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    // Sleep sooner than the default (20 frames, 1/4 m/s, 1/4 rad/s) so that
    // both backends put bodies to sleep within the run (TGS keeps a resting
    // sphere bouncing at up to 0.16 m/s every few frames, above the default 0.01).
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 4),
        angular_threshold: Fix128::from_ratio(1, 4),
        frames_to_sleep: 20,
    });
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, -500.0, 0.0)), fx(500.0));
    let masses = [1.0, 7.0, 0.3];
    for (k, &m) in masses.iter().enumerate() {
        let mut b = RigidBody::new_dynamic(v3(-6.0 - 1.5 * k as f64, 4.0, 0.5), fx(m));
        b.velocity = v3(0.3, 0.0, -0.2 * k as f64);
        w.add_body_with_radius(b, fx(0.4));
    }
    // Chain: 1-2, 2-3 (body 1 is the top of the chain, free).
    w.add_joint(Joint::Ball(BallJoint::new(
        1,
        2,
        v3(-0.75, 0.0, 0.0),
        v3(0.75, 0.0, 0.0),
    )));
    w.add_joint(Joint::Ball(BallJoint::new(
        2,
        3,
        v3(-0.75, 0.0, 0.0),
        v3(0.75, 0.0, 0.0),
    )));
    w.add_body_with_radius(RigidBody::new_dynamic(v3(3.0, 1.2, -1.0), fx(2.5)), fx(0.6));
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(5.5, 2.0, 1.25), fx(0.75)),
        fx(0.3),
    );
    w
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Refused,
    ZeroDrag,
    ZeroPush,
    EmptyMedium,
}

const VARIANTS: [Variant; 4] = [
    Variant::Refused,
    Variant::ZeroDrag,
    Variant::ZeroPush,
    Variant::EmptyMedium,
];

fn with_variant(backend: SolverBackend, v: Variant) -> PhysicsWorld {
    let mut w = scene(backend);
    match v {
        Variant::Refused => {
            assert_eq!(
                w.add_participant(Box::new(Refused(Fix128::ZERO))),
                Err(RegisterError::NonPositiveStep)
            );
            assert_eq!(
                w.add_participant(Box::new(Refused(fx(-0.25)))),
                Err(RegisterError::NonPositiveStep)
            );
            assert_eq!(w.participant_count(), 0);
        }
        Variant::ZeroDrag => {
            let mut m = DragMedium::new(fx(3.0), v3(2.0, -1.0, 0.5)).expect("medium");
            for b in 0..w.bodies.len() {
                m.couple(b, Fix128::ZERO).expect("couple");
            }
            w.add_participant(Box::new(m)).expect("register");
        }
        Variant::ZeroPush => {
            w.add_participant(Box::new(ZeroPush)).expect("register");
        }
        Variant::EmptyMedium => {
            let m = DragMedium::new(fx(0.5), v3(-4.0, 0.0, 9.0)).expect("medium");
            w.add_participant(Box::new(m)).expect("register");
        }
    }
    w
}

fn run(parallel: bool) {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let dt = Fix128::from_ratio(1, 60);
        let mut plain = scene(backend);
        let mut worlds: Vec<(Variant, PhysicsWorld)> = VARIANTS
            .iter()
            .map(|&v| (v, with_variant(backend, v)))
            .collect();
        let mut slept = 0usize;
        for frame in 0..FRAMES {
            if parallel {
                #[cfg(feature = "parallel")]
                plain.step_parallel(dt);
            } else {
                plain.step(dt);
            }
            for (v, w) in &mut worlds {
                if parallel {
                    #[cfg(feature = "parallel")]
                    w.try_step_parallel(dt).expect("step");
                } else {
                    w.try_step(dt).expect("step");
                }
                assert_eq!(
                    w.serialize_state(),
                    plain.serialize_state(),
                    "{backend:?} parallel={parallel} {v:?} frame {frame}"
                );
                assert_eq!(w.fault(), None);
            }
            slept += (1..plain.bodies.len())
                .filter(|&i| plain.is_sleeping(i))
                .count();
        }
        // The scene exercised sleeping dynamic bodies (a zero force must not
        // wake them) and moved (the dropped spheres fell).
        assert!(slept > 0, "{backend:?} parallel={parallel}: nothing slept");
        assert!(plain.bodies[4].position.y.to_f64() < 1.2);
        println!("{backend:?} parallel={parallel}: body-frames asleep {slept}");
    }
}

#[test]
fn no_op_participants_match_the_plain_step_byte_for_byte() {
    run(false);
}

#[cfg(feature = "parallel")]
#[test]
fn no_op_participants_match_the_plain_batched_step_byte_for_byte() {
    run(true);
}
