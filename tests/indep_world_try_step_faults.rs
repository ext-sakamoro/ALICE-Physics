//! Inputs out of the documented range: the checked step reports a fault and
//! leaves exactly the documented part of the state untouched, bit for bit.
//!
//! * A [`DragMedium`] whose fourth coupling (of five) produces a force out of
//!   the range of [`Fix128`]: the step reports
//!   [`WorldFault::Participant`] with [`ParticipantFault::OutOfRange`], the
//!   medium keeps its payload byte for byte, and no force of the first three
//!   couplings reached a body: the bodies are those of the same world
//!   without the medium.
//! * A medium whose own momentum would leave the range when it takes the
//!   reaction (`P − q(dv)` out of range while the force and `dv` are in
//!   range): the same, through the last check of the exchange.
//! * Once a fault is recorded, the next `try_step` is refused and the whole
//!   world blob is unchanged; observations are undecided.
//! * A force or torque whose application is out of range on one body
//!   ([`WorldFault::ForceOutOfRange`]): that body is left as it was, the
//!   other bodies get their forces.
//! * A step rule that does not divide the substep, a per-body field whose
//!   sample count no longer matches: refused before anything ran, blob
//!   unchanged.

#![cfg(feature = "std")]

use alice_physics::coupling_medium::{DragMedium, DRAG_MEDIUM_KIND};
use alice_physics::sleeping::SleepConfig;
use alice_physics::world_participant::{
    FieldLayout, FieldMode, ObservationSink, Observed, Participant, ParticipantFault,
    ParticipantKind, PortId, RegisterError, StateError, StepError, StepRule, SubstepCtx,
    WorldFault,
};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn pow2(k: i32) -> Fix128 {
    if k >= 0 {
        Fix128::from_int(1i64 << k)
    } else {
        Fix128::ONE / Fix128::from_int(1i64 << (-k))
    }
}

#[derive(Clone, Copy, Debug)]
enum Path {
    Step,
    #[cfg(feature = "parallel")]
    Parallel,
}

fn paths() -> Vec<Path> {
    #[allow(unused_mut)]
    let mut p = vec![Path::Step];
    #[cfg(feature = "parallel")]
    p.push(Path::Parallel);
    p
}

fn try_advance(w: &mut PhysicsWorld, path: Path, dt: Fix128) -> Result<(), StepError> {
    match path {
        Path::Step => w.try_step(dt),
        #[cfg(feature = "parallel")]
        Path::Parallel => w.try_step_parallel(dt),
    }
}

fn plain_advance(w: &mut PhysicsWorld, path: Path, dt: Fix128) {
    match path {
        Path::Step => w.step(dt),
        #[cfg(feature = "parallel")]
        Path::Parallel => w.step_parallel(dt),
    }
}

fn world(backend: SolverBackend, substeps: usize, gravity: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: if gravity {
            v3(0.0, -9.81, 0.0)
        } else {
            Vec3Fix::ZERO
        },
        substeps,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    w
}

/// Five free bodies, masses `1 : 7 : 0.3 : 2.5 : 4`; body 3 is very fast.
fn five(backend: SolverBackend) -> PhysicsWorld {
    let mut w = world(backend, 3, true);
    let masses = [1.0, 7.0, 0.3, 2.5, 4.0];
    for (k, &m) in masses.iter().enumerate() {
        let mut b = RigidBody::new_dynamic(v3(20.0 * k as f64, 5.0, -3.0 * k as f64), fx(m));
        b.velocity = v3(0.5 * k as f64, -0.25, 1.0);
        w.add_body(b);
    }
    w.bodies[3].velocity = Vec3Fix::new(pow2(40), Fix128::ZERO, Fix128::ZERO);
    w
}

fn body_bits(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix, Vec3Fix)> {
    w.bodies
        .iter()
        .map(|b| (b.position, b.velocity, b.angular_velocity))
        .collect()
}

fn assert_refused_unchanged(w: &mut PhysicsWorld, path: Path, dt: Fix128, fault: WorldFault) {
    let before = w.snapshot_world();
    assert_eq!(try_advance(w, path, dt), Err(StepError::Faulted(fault)));
    assert!(
        w.snapshot_world() == before,
        "refused step changed the world"
    );
    assert!(matches!(
        w.observe_participant(0),
        Some(Observed::Undecided)
    ));
    assert!(matches!(
        w.observe_body_checked(0),
        Some(Observed::Undecided)
    ));
}

/// The fourth coupling's force `c (u − v)` with `|v| = 2⁴⁰`, `c = 2³⁰` is
/// `2⁷⁰`, out of range; the first three are in range.
#[test]
fn a_force_out_of_range_in_a_later_coupling_stages_nothing() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for path in paths() {
            let dt = Fix128::from_ratio(1, 60);
            let mut w = five(backend);
            let mut bare = five(backend);
            let mut m = DragMedium::new(fx(2.0), v3(1.0, 0.5, -0.5)).expect("medium");
            m.couple(0, fx(3.0)).expect("couple");
            m.couple(4, fx(5.0)).expect("couple");
            m.couple(1, fx(9.0)).expect("couple");
            m.couple(3, pow2(30)).expect("couple");
            m.couple(2, fx(0.5)).expect("couple");
            w.add_participant(Box::new(m)).expect("register");
            let payload = w.participant_state(0).expect("payload");

            let fault = WorldFault::Participant {
                index: 0,
                kind: DRAG_MEDIUM_KIND,
                fault: ParticipantFault::OutOfRange,
            };
            assert_eq!(
                try_advance(&mut w, path, dt),
                Err(StepError::FaultRaised(fault))
            );
            plain_advance(&mut bare, path, dt);
            assert_eq!(w.participant_state(0), Some(payload.clone()));
            assert_eq!(body_bits(&w), body_bits(&bare), "{backend:?} {path:?}");
            assert_eq!(w.fault(), Some(fault));

            assert_refused_unchanged(&mut w, path, dt, fault);

            // Clearing the fault lets the step run again; the medium still
            // cannot exchange and faults again in the same way.
            w.clear_fault();
            assert_eq!(
                try_advance(&mut w, path, dt),
                Err(StepError::FaultRaised(fault))
            );
            plain_advance(&mut bare, path, dt);
            assert_eq!(w.participant_state(0), Some(payload));
            assert_eq!(body_bits(&w), body_bits(&bare), "{backend:?} {path:?} 2");
        }
    }
}

/// `P = 3·2⁶¹` (mass `2⁶¹`, `u = 3`), one body of mass `2⁴⁵` moving at
/// `2²⁰`, `c = 1.25·2⁴²`, `h = 1/2`: the force `c (u − v) ≈ −1.25·2⁶²`,
/// `dv ≈ −1.25·2¹⁶` and `q(dv) ≈ −1.25·2⁶¹` are in range, but
/// `P − q(dv) ≈ 4.25·2⁶¹ > 2⁶³` is not.
#[test]
fn a_medium_momentum_out_of_range_is_refused_after_the_force_checks() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for path in paths() {
            let dt = Fix128::ONE;
            let build = || {
                let mut w = world(backend, 2, false);
                let mut b = RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), pow2(45));
                b.velocity = Vec3Fix::new(pow2(20), Fix128::ZERO, Fix128::ZERO);
                w.add_body(b);
                let mut b = RigidBody::new_dynamic(v3(50.0, 0.0, 0.0), fx(7.0));
                b.velocity = v3(-1.0, 0.0, 0.0);
                w.add_body(b);
                w
            };
            let mut w = build();
            let mut bare = build();
            let mut m = DragMedium::new(pow2(61), Vec3Fix::from_int(3, 0, 0)).expect("medium");
            m.couple(1, fx(2.0)).expect("couple");
            m.couple(0, pow2(42) * fx(1.25)).expect("couple");
            w.add_participant(Box::new(m)).expect("register");
            let payload = w.participant_state(0).expect("payload");
            let fault = WorldFault::Participant {
                index: 0,
                kind: DRAG_MEDIUM_KIND,
                fault: ParticipantFault::OutOfRange,
            };
            assert_eq!(
                try_advance(&mut w, path, dt),
                Err(StepError::FaultRaised(fault)),
                "{backend:?} {path:?}"
            );
            plain_advance(&mut bare, path, dt);
            assert_eq!(w.participant_state(0), Some(payload));
            assert_eq!(body_bits(&w), body_bits(&bare), "{backend:?} {path:?}");
            assert_refused_unchanged(&mut w, path, dt, fault);
        }
    }
}

/// Pushes `force` on `body_a` and `torque` on `body_b`, and a small force on
/// every other body.
struct Pusher {
    kind: u32,
    big: Option<(usize, Vec3Fix, Vec3Fix)>,
    small: Vec3Fix,
}

impl Participant for Pusher {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(self.kind)
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        for i in 0..ctx.bodies().len() {
            match self.big {
                Some((b, f, t)) if b == i => {
                    ctx.add_force(i, f)
                        .map_err(|_| ParticipantFault::InvalidState)?;
                    ctx.add_torque(i, t)
                        .map_err(|_| ParticipantFault::InvalidState)?;
                }
                _ => ctx
                    .add_force(i, self.small)
                    .map_err(|_| ParticipantFault::InvalidState)?,
            }
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

/// A light body (mass `2⁻²⁰`) pushed with `2⁵⁰` (so `F·inv_mass = 2⁷⁰`), or
/// twisted with `2⁵⁰` while pushed with `1/4` (so `I⁻¹τ` is out of range,
/// the force alone is not): that body is left as in
/// a world where it got no force at all; the others get theirs.
#[test]
fn a_force_or_torque_out_of_range_leaves_only_that_body_alone() {
    let cases = [
        (
            Vec3Fix::new(Fix128::ZERO, pow2(50), Fix128::ZERO),
            Vec3Fix::ZERO,
        ),
        // An in-range force with an out-of-range torque: the force must not
        // be applied either (no partial write of the body).
        (
            Vec3Fix::new(Fix128::from_ratio(1, 4), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, pow2(50)),
        ),
    ];
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for path in paths() {
            for (force, torque) in cases {
                let dt = Fix128::from_ratio(1, 50);
                let build = || {
                    let mut w = world(backend, 4, true);
                    for (k, m) in [fx(1.0), pow2(-20), fx(7.0), fx(0.3)].iter().enumerate() {
                        let mut b = RigidBody::new_dynamic(v3(30.0 * k as f64, 0.0, 0.0), *m);
                        b.velocity = v3(0.0, 0.0, 0.25 * k as f64);
                        w.add_body(b);
                    }
                    w
                };
                let small = v3(0.375, -1.25, 0.5);
                let mut w = build();
                w.add_participant(Box::new(Pusher {
                    kind: 1,
                    big: Some((1, force, torque)),
                    small,
                }))
                .expect("register");
                // Reference: body 1 gets nothing, the others the same force.
                let mut r = build();
                r.add_participant(Box::new(Pusher {
                    kind: 1,
                    big: Some((1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
                    small,
                }))
                .expect("register");
                let fault = WorldFault::ForceOutOfRange { body: 1 };
                assert_eq!(
                    try_advance(&mut w, path, dt),
                    Err(StepError::FaultRaised(fault)),
                    "{backend:?} {path:?}"
                );
                try_advance(&mut r, path, dt).expect("reference");
                assert_eq!(
                    body_bits(&w),
                    body_bits(&r),
                    "{backend:?} {path:?} {force:?} {torque:?}"
                );
                // The other bodies really moved by the small force.
                let mut none = build();
                plain_advance(&mut none, path, dt);
                assert_ne!(w.bodies[0].velocity, none.bodies[0].velocity);
                assert_refused_unchanged(&mut w, path, dt, fault);
            }
        }
    }
}

/// A participant with a fixed step of its own.
struct Fixed(Fix128);

impl Participant for Fixed {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"FIXD"))
    }
    fn step_rule(&self) -> StepRule {
        StepRule::Fixed(self.0)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// A fixed step of `1/96` does not divide `h = (1/60)/3 = 1/180`: refused,
/// nothing ran (the medium in front of it did not exchange either). A
/// per-body field declared for five bodies after a sixth was added:
/// refused, nothing ran.
#[test]
fn checks_before_the_step_refuse_without_running_anything() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for path in paths() {
            let dt = Fix128::from_ratio(1, 60);
            let mut w = five(backend);
            w.bodies[3].velocity = v3(1.0, 0.0, 0.0);
            let mut m = DragMedium::new(fx(2.0), v3(-3.0, 0.0, 1.0)).expect("medium");
            for b in 0..5 {
                m.couple(b, fx(1.5)).expect("couple");
            }
            w.add_participant(Box::new(m)).expect("register");
            w.add_participant(Box::new(Fixed(Fix128::from_ratio(1, 96))))
                .expect("register");
            let before = w.snapshot_world();
            let r = try_advance(&mut w, path, dt);
            assert!(
                matches!(
                    r,
                    Err(StepError::Rule {
                        index: 1,
                        error: RegisterError::StepRuleMismatch { .. }
                    })
                ),
                "{backend:?} {path:?}: {r:?}"
            );
            assert!(w.snapshot_world() == before, "{backend:?} {path:?}");
            assert_eq!(w.fault(), None);

            let mut w = five(backend);
            w.bodies[3].velocity = v3(1.0, 0.0, 0.0);
            let id = PortId::new(77);
            w.declare_field(id, FieldLayout::PerBody { bodies: 5 }, FieldMode::Sum)
                .expect("field");
            let mut m = DragMedium::new(fx(2.0), v3(-3.0, 0.0, 1.0)).expect("medium");
            m.couple(2, fx(1.5)).expect("couple");
            w.add_participant(Box::new(m)).expect("register");
            w.add_body(RigidBody::new_dynamic(v3(-40.0, 0.0, 0.0), fx(1.0)));
            let before = w.snapshot_world();
            assert_eq!(
                try_advance(&mut w, path, dt),
                Err(StepError::BodyCount {
                    field: id,
                    samples: 5,
                    bodies: 6
                })
            );
            assert!(w.snapshot_world() == before, "{backend:?} {path:?}");
        }
    }
}

/// A recorded fault travels with the snapshot: a fresh world restored from
/// the blob refuses to step, unchanged, until the fault is cleared.
#[test]
fn a_recorded_fault_is_restored_and_refuses_the_step() {
    let dt = Fix128::from_ratio(1, 60);
    let medium = || {
        let mut m = DragMedium::new(fx(2.0), v3(1.0, 0.5, -0.5)).expect("medium");
        m.couple(3, pow2(30)).expect("couple");
        m
    };
    let mut w = five(SolverBackend::Tgs);
    w.add_participant(Box::new(medium())).expect("register");
    assert!(w.try_step(dt).is_err());
    let fault = w.fault().expect("fault");
    let blob = w.snapshot_world();

    let mut fresh = PhysicsWorld::new(PhysicsConfig::default());
    fresh.add_participant(Box::new(medium())).expect("register");
    fresh.restore_world(&blob).expect("restore");
    assert_eq!(fresh.fault(), Some(fault));
    assert_refused_unchanged(&mut fresh, Path::Step, dt, fault);
}
