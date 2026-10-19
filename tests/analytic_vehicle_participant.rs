//! Oracles for the vehicle as a world participant (`VehicleParticipant`) and
//! for actions as world state (`FieldLayout::Channels`).
//!
//! What is pinned, and against what:
//!
//! * the channel field: its length, its refusals and its place in the
//!   snapshot;
//! * the suspension at rest against `k·c = m·g` (as the frame-head path in
//!   `tests/default_configs.rs`), and the frame-end vertical velocity, which
//!   the frame-head impulse leaves near `−g·dt/2` and the per-substep force
//!   brings to rest;
//! * aerodynamic drag alone against `v = v₀/(1 + (c/m)·v₀·t)`, first order in
//!   the substep width;
//! * the drive from a throttle written into the action field against
//!   `v = v*·tanh(t·√(F·c)/m)`, `v* = √(F/c)`;
//! * actions as world state: held for a frame, carried by snapshots and
//!   forks, replayed bit for bit from a recorded list;
//! * the participant contract: refused inputs are faults that change
//!   nothing, a snapshot of another vehicle is refused, the parallel step
//!   agrees, a digest of a drive is pinned, and the crate's semantics
//!   identifier is the one main had.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 throttle velocity v* tanh(t sqrt(F c)/m) closed form for VehicleParticipant drive"
)]

use alice_physics_core::solver::DeclareFieldError;
use alice_physics_core::vehicle::{
    Vehicle, VehicleConfig, VehicleParticipant, WheelConfig, VEHICLE_INPUTS, VEHICLE_INPUT_BRAKE,
    VEHICLE_INPUT_STEERING, VEHICLE_INPUT_THROTTLE, VEHICLE_OBS_SPEED_KMH, VEHICLE_OBS_THROTTLE,
    VEHICLE_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{
    FieldError, FieldLayout, FieldMode, FieldPortError, Observed, ParticipantFault, PortId,
    RegisterError, StepError, WorldFault,
};
use alice_physics_core::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix, WorldSnapshotError,
    PHYSICS_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

const ACTIONS: PortId = PortId::new(0x5649_4e30);

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn f64_of(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn dt60() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn channels(world: &mut PhysicsWorld) {
    world
        .declare_field(
            ACTIONS,
            FieldLayout::Channels {
                count: VEHICLE_INPUTS,
            },
            FieldMode::Replace,
        )
        .expect("declare");
}

fn act(world: &mut PhysicsWorld, throttle: f64, brake: f64, steering: f64) {
    let mut v = [Fix128::ZERO; VEHICLE_INPUTS];
    v[VEHICLE_INPUT_THROTTLE] = fx(throttle);
    v[VEHICLE_INPUT_BRAKE] = fx(brake);
    v[VEHICLE_INPUT_STEERING] = fx(steering);
    world.set_field(ACTIONS, &v).expect("set");
}

/// A world with one default wheel under a chassis of `mass`, the wheel just
/// touching the ground (the scene of `tests/default_configs.rs`).
fn single_wheel(mass: i64, backend: SolverBackend) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.config.solver_backend = backend;
    let wheel = WheelConfig::default();
    let start_y = wheel.suspension_rest + wheel.radius - Fix128::from_ratio(1, 1000);
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(mass),
    ));
    let vehicle = Vehicle::new(VehicleConfig {
        wheels: vec![wheel],
        ..VehicleConfig::default()
    });
    world
        .add_participant(Box::new(VehicleParticipant::new(vehicle, chassis)))
        .expect("register");
    world
}

/// The default four-wheel vehicle on the ground, its inputs read from the
/// action field. No frame damping (`damping = 1`), so the drive and drag
/// closed forms hold without a per-frame velocity factor.
fn car(backend: SolverBackend) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    world.config.solver_backend = backend;
    channels(&mut world);
    let config = VehicleConfig::default();
    let w = config.wheels[0];
    let start_y = w.suspension_rest + w.radius - w.local_position.y - Fix128::from_ratio(1, 100);
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(1200),
    ));
    world
        .add_participant(Box::new(
            VehicleParticipant::new(Vehicle::new(config), chassis).with_inputs(ACTIONS),
        ))
        .expect("register");
    world
}

fn observed(world: &PhysicsWorld, channel: u32) -> Fix128 {
    match world.observe_participant(0).expect("participant") {
        Observed::Exact(sink) => sink
            .values()
            .iter()
            .find(|(c, _)| *c == channel)
            .map(|(_, v)| *v)
            .expect("channel"),
        Observed::Undecided => panic!("undecided"),
    }
}

// ---------------------------------------------------------------------------
// The channel layout
// ---------------------------------------------------------------------------

/// oracle (layout): `Channels { count }` holds `count` samples, starts at
/// zero, takes a write of exactly `count` values, and a count of zero is an
/// invalid layout.
#[test]
fn channel_field_length_and_refusals() {
    assert_eq!(FieldLayout::Channels { count: 3 }.samples(), Some(3));
    assert_eq!(FieldLayout::Channels { count: 0 }.samples(), None);
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    channels(&mut w);
    assert_eq!(w.fields().value(ACTIONS), Some(&[Fix128::ZERO; 3][..]));
    act(&mut w, 0.25, 0.5, -0.5);
    assert_eq!(
        w.fields().value(ACTIONS),
        Some(&[fx(0.25), fx(0.5), fx(-0.5)][..])
    );
    assert_eq!(
        w.set_field(ACTIONS, &[Fix128::ONE; 2]),
        Err(FieldError::Length {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        w.declare_field(
            PortId::new(9),
            FieldLayout::Channels { count: 0 },
            FieldMode::Replace
        ),
        Err(DeclareFieldError::Field(FieldError::InvalidLayout))
    );
}

/// oracle (snapshot): the channel values travel in the snapshot; a world
/// declaring the channels restores them, a world declaring another count
/// refuses the blob and is left unchanged.
#[test]
fn channel_values_are_in_the_snapshot() {
    let mut a = PhysicsWorld::new(PhysicsConfig::default());
    channels(&mut a);
    act(&mut a, 0.25, 0.5, -0.5);
    let snap = a.snapshot_world();
    let mut b = PhysicsWorld::new(PhysicsConfig::default());
    channels(&mut b);
    b.restore_world(&snap).expect("restore");
    assert_eq!(b.fields().value(ACTIONS), a.fields().value(ACTIONS));
    let mut c = PhysicsWorld::new(PhysicsConfig::default());
    c.declare_field(
        ACTIONS,
        FieldLayout::Channels { count: 4 },
        FieldMode::Replace,
    )
    .expect("declare");
    let before = c.snapshot_world();
    assert!(
        c.restore_world(&snap).is_err(),
        "another count must be refused"
    );
    assert_eq!(c.snapshot_world(), before);
}

// ---------------------------------------------------------------------------
// Suspension at rest
// ---------------------------------------------------------------------------

/// oracle (closed form, statics): with the vehicle stepped inside the world,
/// one default wheel settles at compression `c = m·g/k` and ride height
/// `radius + rest·(1 − c)`, and the chassis comes to rest: after 15 s the
/// ride height is the closed form to `1e-9` m and the frame-end vertical
/// velocity is below `1e-12` m/s (measured: the error decays by about 300
/// per 2.5 s, `vy` reaches `1e-17`). The frame-head impulse of
/// `Vehicle::update` meets the closed form only to 0.01
/// (`tests/default_configs.rs`) and leaves the frame-end velocity near
/// `−g·dt/2 = −0.082` m/s (asserted below `−0.05` here so the difference is
/// visible).
#[test]
fn suspension_rests_with_zero_frame_end_velocity() {
    let wheel = WheelConfig::default();
    let (k, rest, radius) = (
        f64_of(wheel.spring_stiffness),
        f64_of(wheel.suspension_rest),
        f64_of(wheel.radius),
    );
    let g = -f64_of(PhysicsConfig::default().gravity.y);
    for mass in [1000_i64, 500] {
        let mut world = single_wheel(mass, SolverBackend::Xpbd);
        let mut max_dy = 0.0_f64;
        for frame in 0..900 {
            let before = f64_of(world.bodies[0].position.y);
            world.try_step(dt60()).expect("step");
            if frame >= 840 {
                max_dy = max_dy.max((f64_of(world.bodies[0].position.y) - before).abs());
            }
        }
        let c_ref = mass as f64 * g / k;
        let y_ref = radius + rest * (1.0 - c_ref);
        let y = f64_of(world.bodies[0].position.y);
        assert!(
            (y - y_ref).abs() < 1e-9,
            "mass {mass}: ride height {y}, closed form {y_ref}"
        );
        assert!(max_dy < 1e-12, "mass {mass}: still moving {max_dy} m/frame");
        let vy = f64_of(world.bodies[0].velocity.y);
        assert!(vy.abs() < 1e-12, "mass {mass}: frame-end vy {vy}");
    }

    // the frame-head path for contrast (same scene, `Vehicle::update`)
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let start_y = wheel.suspension_rest + wheel.radius - Fix128::from_ratio(1, 1000);
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(1000),
    ));
    let mut vehicle = Vehicle::new(VehicleConfig {
        wheels: vec![wheel],
        ..VehicleConfig::default()
    });
    for _ in 0..300 {
        vehicle.update(&mut world.bodies[chassis], dt60());
        world.step(dt60());
    }
    let vy = f64_of(world.bodies[chassis].velocity.y);
    assert!(vy < -0.05, "frame-head saw-tooth sampled at {vy}");
}

// ---------------------------------------------------------------------------
// Drag and drive
// ---------------------------------------------------------------------------

/// A wheelless chassis of 1000 kg in zero gravity and without frame damping
/// moving at 30 m/s along its forward axis, with drag `c = 0.4` and no
/// downforce, for 10 s: only drag acts, `m·v' = −c·v²`, so `v = v₀/(1 + (c/m)·v₀·t)`.
fn coast(substeps: usize) -> f64 {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        ..PhysicsConfig::default()
    });
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(1000));
    body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(30));
    let chassis = world.add_body(body);
    let vehicle = Vehicle::new(VehicleConfig {
        wheels: vec![],
        aero_drag: fx(0.4),
        downforce: Fix128::ZERO,
        ..VehicleConfig::default()
    });
    world
        .add_participant(Box::new(VehicleParticipant::new(vehicle, chassis)))
        .expect("register");
    for _ in 0..600 {
        world.try_step(dt60()).expect("step");
    }
    let want = 30.0 / (1.0 + 0.4 / 1000.0 * 30.0 * 10.0);
    (f64_of(world.bodies[chassis].velocity.z) - want).abs()
}

/// oracle (closed form, drag, convergence): the error against
/// `v₀/(1 + (c/m)·v₀·t)` is first order in the substep width: from 1 to 4
/// substeps it shrinks by a factor between 3 and 5 (measured 4.0008), and
/// with 4 substeps it is below `2e-4` m/s (measured `1.36·10⁻⁴`).
#[test]
fn drag_follows_the_closed_form_to_first_order_in_the_substep() {
    let e1 = coast(1);
    let e4 = coast(4);
    let ratio = e1 / e4;
    assert!(
        (3.0..5.0).contains(&ratio),
        "e1 {e1:e}, e4 {e4:e}, ratio {ratio}"
    );
    assert!(e4 < 2e-4, "e4 {e4:e}");
}

/// oracle (closed form, drive): after settling, the throttle written into
/// the action field drives the car straight ahead with the force
/// `F = max_torque·throttle·gear_ratio/radius` (all driven wheels together)
/// against drag `c·v²`: `v(t) = v*·tanh(t·√(F·c)/m)`, `v* = √(F/c)`. Checked
/// at 2, 4 and 6 s after the throttle opens, to `1e-6` of `v*` (measured
/// `4·10⁻⁸`, `1.8·10⁻⁷`, `3.9·10⁻⁷`: the substep error grows with `t`).
#[test]
fn throttle_drive_follows_the_tanh_closed_form() {
    let mut world = car(SolverBackend::Xpbd);
    for _ in 0..120 {
        world.try_step(dt60()).expect("settle");
    }
    let config = VehicleConfig::default();
    let throttle = 0.5;
    let f = f64_of(config.engine.max_torque) * throttle * f64_of(config.gear_ratios[0])
        / f64_of(config.wheels[0].radius);
    let c = f64_of(config.aero_drag);
    let m = 1200.0;
    let v_star = (f / c).sqrt();
    act(&mut world, throttle, 0.0, 0.0);
    let v0 = f64_of(world.bodies[0].velocity.z);
    assert!(v0.abs() < 1e-3, "settled speed {v0}");
    let mut stepped = 0;
    for (frames, t) in [(120, 2.0), (240, 4.0), (360, 6.0)] {
        while stepped < frames {
            world.try_step(dt60()).expect("drive");
            stepped += 1;
        }
        let want = v_star * (t * (f * c).sqrt() / m).tanh();
        let got = f64_of(world.bodies[0].velocity.z);
        assert!(
            (got - want).abs() < 1e-6 * v_star,
            "t = {t}: v {got}, closed form {want} (v* {v_star})"
        );
    }
    assert_eq!(observed(&world, VEHICLE_OBS_THROTTLE), fx(throttle));
    assert!(observed(&world, VEHICLE_OBS_SPEED_KMH) > Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// Actions as world state
// ---------------------------------------------------------------------------

/// oracle (replay): a drive with actions changed every 30 frames, recorded
/// as `(frame, action)`, replayed from the initial snapshot by writing each
/// action before its frame, ends bit for bit where the original ended; a
/// fork taken mid-drive carries the action in force at that moment, so
/// both branches continue identically without writing any action.
#[test]
fn actions_are_world_state_replay_and_fork() {
    let script = |frame: u64| -> Option<(f64, f64, f64)> {
        match frame {
            0 => Some((0.6, 0.0, 0.0)),
            30 => Some((0.6, 0.0, 0.4)),
            60 => Some((0.0, 0.3, -0.2)),
            90 => Some((1.0, 0.0, 0.0)),
            _ => None,
        }
    };
    let mut a = car(SolverBackend::Xpbd);
    let initial = a.snapshot_world();
    let mut fork = None;
    for frame in 0..150 {
        if let Some((t, b, s)) = script(frame) {
            act(&mut a, t, b, s);
        }
        if frame == 75 {
            fork = Some(a.snapshot_world());
        }
        a.try_step(dt60()).expect("step");
    }
    let mut r = car(SolverBackend::Xpbd);
    r.restore_world(&initial).expect("restore");
    for frame in 0..150 {
        if let Some((t, b, s)) = script(frame) {
            act(&mut r, t, b, s);
        }
        r.try_step(dt60()).expect("step");
    }
    assert_eq!(r.snapshot_world(), a.snapshot_world(), "replay");

    let fork = fork.expect("fork");
    let mut x = car(SolverBackend::Xpbd);
    let mut y = car(SolverBackend::Xpbd);
    x.restore_world(&fork).expect("restore");
    y.restore_world(&fork).expect("restore");
    assert_eq!(
        x.fields().value(ACTIONS),
        Some(&[fx(0.0), fx(0.3), fx(-0.2)][..])
    );
    for _ in 0..40 {
        x.try_step(dt60()).expect("step");
        y.try_step(dt60()).expect("step");
    }
    assert_eq!(x.snapshot_world(), y.snapshot_world(), "fork");
}

/// oracle (hold): an action written between frames is read by every substep
/// of the next frame and by none of the previous one: from two identical
/// settled worlds, the one given throttle 1 gains `F/m·dt` more forward
/// speed over the next frame (four substeps of `F/m·h`), to `1e-6` relative
/// (measured `7·10⁻⁸`), and the worlds were identical before the write.
#[test]
fn an_action_is_held_for_the_whole_next_frame() {
    let mut a = car(SolverBackend::Xpbd);
    let mut b = car(SolverBackend::Xpbd);
    for _ in 0..120 {
        a.try_step(dt60()).expect("settle");
        b.try_step(dt60()).expect("settle");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    act(&mut a, 1.0, 0.0, 0.0);
    a.try_step(dt60()).expect("step");
    b.try_step(dt60()).expect("step");
    let config = VehicleConfig::default();
    let f = f64_of(config.engine.max_torque) * f64_of(config.gear_ratios[0])
        / f64_of(config.wheels[0].radius);
    let dv = f64_of(a.bodies[0].velocity.z) - f64_of(b.bodies[0].velocity.z);
    let want = f / 1200.0 / 60.0;
    assert!(
        (dv - want).abs() < 1e-6 * want,
        "Δv {dv}, closed form {want}"
    );
}

// ---------------------------------------------------------------------------
// The participant contract
// ---------------------------------------------------------------------------

/// oracle (refusals): a throttle above 1, a brake below 0 or a steering
/// beyond ±1 is a fault (`OutOfRange`) that leaves the participant's payload
/// unchanged; a channel field of the wrong length and a chassis index past
/// the bodies are `InvalidState`; registering against an undeclared field
/// is refused.
#[test]
fn bad_inputs_are_faults_that_change_nothing() {
    for (t, b, s) in [(1.5, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 1.25)] {
        let mut w = car(SolverBackend::Xpbd);
        w.try_step(dt60()).expect("first frame");
        let before = w.participant_state(0);
        act(&mut w, t, b, s);
        assert_eq!(
            w.try_step(dt60()),
            Err(StepError::FaultRaised(WorldFault::Participant {
                index: 0,
                kind: VEHICLE_PARTICIPANT_KIND,
                fault: ParticipantFault::OutOfRange,
            })),
            "({t}, {b}, {s})"
        );
        assert_eq!(w.participant_state(0), before);
    }

    for count in [2, 4] {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.declare_field(ACTIONS, FieldLayout::Channels { count }, FieldMode::Replace)
            .expect("declare");
        let chassis = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        w.add_participant(Box::new(
            VehicleParticipant::new(Vehicle::new(VehicleConfig::default()), chassis)
                .with_inputs(ACTIONS),
        ))
        .expect("register");
        assert!(
            matches!(
                w.try_step(dt60()),
                Err(StepError::FaultRaised(WorldFault::Participant {
                    fault: ParticipantFault::InvalidState,
                    ..
                }))
            ),
            "{count} channels"
        );
    }

    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w.add_participant(Box::new(VehicleParticipant::new(
        Vehicle::new(VehicleConfig::default()),
        5,
    )))
    .expect("register");
    assert!(matches!(
        w.try_step(dt60()),
        Err(StepError::FaultRaised(WorldFault::Participant {
            fault: ParticipantFault::InvalidState,
            ..
        }))
    ));

    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let chassis = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    assert_eq!(
        w.add_participant(Box::new(
            VehicleParticipant::new(Vehicle::new(VehicleConfig::default()), chassis)
                .with_inputs(ACTIONS),
        )),
        Err(RegisterError::Field(FieldPortError::UnknownField {
            participant: 0,
            field: ACTIONS
        }))
    );
}

/// oracle (static chassis): a static chassis gets no force and the vehicle
/// state stays as constructed, through the world and through
/// `Vehicle::chassis_force` called directly.
#[test]
fn a_static_chassis_is_left_alone() {
    // at ride height, every wheel on the ground (the guard is what keeps the
    // suspension from acting)
    let config = VehicleConfig::default();
    let w0 = config.wheels[0];
    let y = w0.suspension_rest + w0.radius - w0.local_position.y - Fix128::from_ratio(1, 10);
    let body = RigidBody::new_static(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO));
    let mut v = Vehicle::new(config);
    v.throttle = Fix128::ONE;
    assert_eq!(v.chassis_force(&body, dt60()), Vec3Fix::ZERO);
    assert_eq!(
        v.wheel_states,
        Vehicle::new(VehicleConfig::default()).wheel_states
    );
    assert_eq!(v.speed_kmh, Fix128::ZERO);

    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let chassis = w.add_body(RigidBody::new_static(Vec3Fix::new(
        Fix128::ZERO,
        Fix128::ONE,
        Fix128::ZERO,
    )));
    w.add_participant(Box::new(VehicleParticipant::new(
        Vehicle::new(VehicleConfig::default()),
        chassis,
    )))
    .expect("register");
    let before = w.participant_state(0);
    for _ in 0..10 {
        w.try_step(dt60()).expect("step");
    }
    assert_eq!(w.participant_state(0), before);
}

/// oracle (snapshot of the vehicle): a fork after 90 frames of driving is
/// bit for bit the original 60 frames later (blob and payload); a world
/// whose vehicle has another spring refuses the snapshot unchanged.
#[test]
fn snapshot_fork_and_another_vehicle_is_refused() {
    let mut a = car(SolverBackend::Xpbd);
    act(&mut a, 0.7, 0.0, 0.3);
    for _ in 0..90 {
        a.try_step(dt60()).expect("step");
    }
    let snap = a.snapshot_world();
    let mut b = car(SolverBackend::Xpbd);
    b.restore_world(&snap).expect("restore");
    // the payload carries the vehicle state the observation reports
    for channel in 0..7 {
        assert_eq!(
            observed(&b, channel),
            observed(&a, channel),
            "channel {channel}"
        );
    }
    for _ in 0..60 {
        a.try_step(dt60()).expect("step");
        b.try_step(dt60()).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    assert_eq!(a.participant_state(0), b.participant_state(0));

    let mut other = PhysicsWorld::new(PhysicsConfig::default());
    channels(&mut other);
    let chassis = other.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut config = VehicleConfig::default();
    for w in &mut config.wheels {
        w.spring_stiffness = w.spring_stiffness + Fix128::ONE;
    }
    other
        .add_participant(Box::new(
            VehicleParticipant::new(Vehicle::new(config), chassis).with_inputs(ACTIONS),
        ))
        .expect("register");
    let before = other.snapshot_world();
    assert!(matches!(
        other.restore_world(&snap),
        Err(WorldSnapshotError::ParticipantState { index: 0, .. })
    ));
    assert_eq!(other.snapshot_world(), before);
}

/// oracle (identity): the vehicle enters the law identifier, and the crate's
/// semantics identifier is the value main pinned before this resident
/// (`tests/physics_semantics_id.rs`): adding a resident is a new law in the
/// world, not a change of the existing ones.
#[test]
fn identity_names_the_resident_and_semantics_are_unmoved() {
    let mut plain = PhysicsWorld::new(PhysicsConfig::default());
    plain.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let with = single_wheel(1000, SolverBackend::Xpbd);
    assert_ne!(
        plain.law_id(&PHYSICS_SEMANTICS_ID),
        with.law_id(&PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(with.participant_kinds(), vec![VEHICLE_PARTICIPANT_KIND]);
    let hex: String = PHYSICS_SEMANTICS_ID
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        "7b24f99b81c4a5e1369ea5e3f99e221fca0286e05c193ba8a453f3670ce087c2"
    );
}

fn drive_digest(backend: SolverBackend) -> String {
    let mut w = car(backend);
    for frame in 0..240 {
        match frame {
            60 => act(&mut w, 0.8, 0.0, 0.0),
            120 => act(&mut w, 0.8, 0.0, 0.5),
            180 => act(&mut w, 0.0, 0.6, 0.0),
            _ => {}
        }
        w.try_step(dt60()).expect("step");
    }
    let mut bytes = w.snapshot_world();
    bytes.extend(w.participant_state(0).expect("state"));
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const DRIVE_XPBD: &str = "068c8e0ede7a6c18f4158a1c97c6de57bcbbad8f740d14caee9bc5b5260bcc05";
const DRIVE_TGS: &str = "f7081044c6ebf09159b01efabce848684eb736dd2b5de52e8c29bdd95083b42e";

/// golden (determinism): settle, accelerate, turn, brake; the world and the
/// participant payload after 240 frames, per backend (CI runs it on every
/// operating system and on wasm32).
#[test]
fn drive_digest_is_pinned() {
    assert_eq!(drive_digest(SolverBackend::Xpbd), DRIVE_XPBD);
    assert_eq!(drive_digest(SolverBackend::Tgs), DRIVE_TGS);
}

/// oracle (parallel): `try_step_parallel` gives the same blob on 1, 2 and 8
/// threads, and, without contacts (the ground is the vehicle's plane), the
/// same chassis and payload as `try_step`.
#[cfg(feature = "parallel")]
#[test]
fn parallel_agrees_with_threads_and_serial() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let mut w = car(SolverBackend::Xpbd);
            act(&mut w, 0.6, 0.0, 0.3);
            for _ in 0..120 {
                w.try_step_parallel(dt60()).expect("step");
            }
            w
        })
    };
    let p1 = run(1);
    for t in [2, 8] {
        assert_eq!(run(t).snapshot_world(), p1.snapshot_world(), "{t} threads");
    }
    let mut s = car(SolverBackend::Xpbd);
    act(&mut s, 0.6, 0.0, 0.3);
    for _ in 0..120 {
        s.try_step(dt60()).expect("step");
    }
    assert_eq!(s.bodies[0].position, p1.bodies[0].position);
    assert_eq!(s.bodies[0].velocity, p1.bodies[0].velocity);
    assert_eq!(s.participant_state(0), p1.participant_state(0));
}
