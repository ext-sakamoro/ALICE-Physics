//! Oracles for `wind_zone::WindZoneParticipant`: a bounded wind that pushes the
//! world's bodies inside its substeps through the participant contract.
//!
//! What is pinned, and against what:
//!
//! * the release into a steady wind against the closed form of quadratic drag
//!   `u(t) = U / (1 + c·U·t)` (`u = U − v`, `c = k/m`, `k = ½ρC_dA`), with the
//!   first-order error of the explicit integration derived below;
//! * the terminal velocity under gravity, `v = U − √(m·g/k)`;
//! * one substep's staged force against the same formula evaluated by hand in
//!   `f64`, at the clock value the gust formula is given;
//! * the world side of the contract: outside the zone nothing changes (bit for
//!   bit against a world without the resident), static and kinematic bodies are
//!   left out, a resting body is woken, the clock is in the state digest,
//!   snapshot and fork are bit identical, a snapshot of another zone is
//!   refused, an out-of-range force is a fault that leaves the participant
//!   unchanged, the law identifier names the resident, and a digest of the
//!   stepped world is pinned for each backend.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 ln and sin evaluate the closed forms the world is compared with"
)]

use alice_physics_core::buoyancy_zone::ZoneShape;
use alice_physics_core::wind_zone::{
    WindZone, WindZoneParticipant, WIND_ZONE_OBS_BODIES, WIND_ZONE_OBS_TIME,
    WIND_ZONE_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{
    ForceAccumulator, Observed, Participant, ParticipantFault, StateError, SubstepCtx, WorldFault,
};
use alice_physics_core::{
    BodyType, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix,
    WorldSnapshotError, PHYSICS_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

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

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn big_box() -> ZoneShape {
    ZoneShape::Aabb {
        min: v(-1e4, -1e4, -1e4),
        max: v(1e4, 1e4, 1e4),
    }
}

/// `k = ½·ρ·C_d·A = 1` with `ρ = 1`, `C_d = 1`, `A = 2`.
fn zone(shape: ZoneShape, dir: Vec3Fix, speed: f64, gust: f64, hz: f64) -> WindZone {
    WindZone {
        shape,
        air_density_kg_m3: Fix128::ONE,
        direction: dir,
        base_speed_m_s: fx(speed),
        turbulence_amplitude: fx(gust),
        gust_frequency_hz: fx(hz),
        drag_coefficient: Fix128::ONE,
        reference_area_m2: fx(2.0),
    }
}

fn resident(z: WindZone) -> Box<WindZoneParticipant> {
    Box::new(WindZoneParticipant::new(z))
}

fn world(substeps: usize, gravity: Vec3Fix, backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity,
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    w.config.solver_backend = backend;
    w
}

/// The speed of a body of mass `m = 2` released from rest into a wind of
/// `U = 4` after 1 s, `dt = 1/64`.
fn released_speed(substeps: usize, backend: SolverBackend) -> f64 {
    let mut w = world(substeps, Vec3Fix::ZERO, backend);
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)));
    w.add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 4.0, 0.0, 0.0)))
        .expect("register");
    for _ in 0..64 {
        w.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    f64_of(w.bodies[0].velocity.x)
}

/// oracle (closed form): `u' = −c·u²` from `u(0) = U` is `u = U/(1 + c·U·t)`;
/// here `c = 1/2`, `U = 4`, `t = 1`, so `v(1) = 4 − 4/3 = 8/3`. The world
/// integrates with an explicit step `h`. For that step
/// `1/u_{n+1} = 1/u_n + c·h + c²·u_n·h² + …`, so `1/u` gains `c²·h·∫u dt =
/// c·h·ln(1 + c·U·t)` over the release and the exact error to first order is
/// `v_h − v = u²·c·h·ln(1 + c·U·t)`: the integrated body is ahead of the closed
/// form by that much, measured to 3 % for both backends and both widths.
#[test]
fn release_into_a_steady_wind_follows_quadratic_drag() {
    let exact = 8.0 / 3.0;
    let u = 4.0 / 3.0_f64;
    let c = 0.5;
    let log = (1.0 + c * 4.0 * 1.0_f64).ln();
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for substeps in [4usize, 16] {
            let h = 1.0 / (64.0 * substeps as f64);
            let want = u * u * c * h * log;
            let got = released_speed(substeps, backend) - exact;
            assert!(
                (got / want - 1.0).abs() < 0.03,
                "{backend:?} substeps {substeps}: error {got:e}, first-order {want:e}"
            );
        }
    }
}

/// oracle (terminal velocity): an upward wind of `U = 10` on a body of mass 2
/// under `g = 10` with `k = 1` balances at relative speed `√(m·g/k) = √20`, so
/// the body rises at `10 − √20`, which the explicit step reaches exactly (the
/// fixed point of the step does not depend on `h`).
#[test]
fn terminal_velocity_in_an_upward_wind() {
    let want = 10.0 - 20.0_f64.sqrt();
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut w = world(4, v(0.0, -10.0, 0.0), backend);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)));
        w.add_participant(resident(zone(big_box(), v(0.0, 1.0, 0.0), 10.0, 0.0, 0.0)))
            .expect("register");
        for _ in 0..1920 {
            w.try_step(Fix128::from_ratio(1, 64)).expect("step");
        }
        let got = f64_of(w.bodies[0].velocity.y);
        assert!((got - want).abs() < 1e-9, "{backend:?}: {got} want {want}");
    }
}

/// The relative speed after `steps` single-substep frames of width `h`, for a body of
/// mass 2 starting at rest in a wind of `U = 4` with `k = 1`, so `x0 = k·U·h/m = 2h`;
/// `None` when a step faults (the force left the fixed-point range).
fn relative_speeds(h: f64, steps: usize) -> Option<Vec<f64>> {
    let mut w = world(1, Vec3Fix::ZERO, SolverBackend::Xpbd);
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)));
    w.add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 4.0, 0.0, 0.0)))
        .expect("register");
    let mut out = Vec::new();
    for _ in 0..steps {
        w.try_step(fx(h)).ok()?;
        out.push(4.0 - f64_of(w.bodies[0].velocity.x));
    }
    Some(out)
}

/// oracle (the explicit step's range, both sides close in): one substep maps `u` to
/// `u·(1 − c·h·|u|)` with `c = k/m`. At `x0 = 1.9` the speed flips sign and
/// shrinks, at `x0 = 2.1` it grows, at `x0 = 0.9` it keeps its sign and at
/// `x0 = 1.1` it flips: the bounds `x < 2` and `x < 1` of the module doc.
#[test]
fn the_explicit_drag_is_bounded_below_x_2_and_grows_above() {
    let below = relative_speeds(0.95, 40).expect("bounded below 2");
    assert!(below.iter().all(|u| u.abs() < 4.0), "{below:?}");
    assert!(below[39].abs() < 0.5, "{below:?}");
    assert!(below[0] < 0.0, "x0 = 1.9 flips the sign: {below:?}");
    match relative_speeds(1.05, 4) {
        None => {}
        Some(above) => assert!(above[3].abs() > 4.0, "x0 = 2.1 grows: {above:?}"),
    }
    assert!(relative_speeds(0.45, 40)
        .expect("x0 = 0.9")
        .iter()
        .all(|u| *u > 0.0));
    assert!(relative_speeds(0.55, 1).expect("x0 = 1.1")[0] < 0.0);
}

/// oracle (the force of one substep, by hand in `f64`): with a gust at clock
/// `t`, a body at velocity `(1, 2, 0)` in a wind along `(0.6, 0, 0.8)` of
/// `3 + 0.5·sin(2π·2·t)` takes `k·|w − v|·(w − v)`. The clock is advanced by
/// `h` after each substep, so the third call uses `t = 2h`. Static and
/// kinematic bodies and a body outside the zone take nothing.
#[test]
fn staged_force_matches_the_formula_at_the_clock() {
    let shape = ZoneShape::Aabb {
        min: v(-1.0, -1.0, -1.0),
        max: v(1.0, 1.0, 1.0),
    };
    let z = zone(shape, v(0.6, 0.0, 0.8), 3.0, 0.5, 2.0);
    let mut inside = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    inside.velocity = v(1.0, 2.0, 0.0);
    let mut kin = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    kin.body_type = BodyType::Kinematic;
    let bodies = vec![
        inside,
        RigidBody::new_static(Vec3Fix::ZERO),
        kin,
        RigidBody::new_dynamic(v(5.0, 0.0, 0.0), Fix128::ONE),
    ];
    let h = Fix128::from_ratio(1, 64);
    let mut p = WindZoneParticipant::new(z);
    let mut acc = ForceAccumulator::new(bodies.len());
    for i in 0..3 {
        acc.clear();
        let mut ctx = SubstepCtx::new(&bodies, &mut acc, i, 3, h).expect("ctx");
        p.substep(&mut ctx, h).expect("substep");
    }
    let t = 2.0 / 64.0;
    let w = 3.0 + 0.5 * (2.0 * core::f64::consts::PI * 2.0 * t).sin();
    let rel = [0.6 * w - 1.0, -2.0, 0.8 * w];
    let speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
    let f = acc.force(0).expect("slot");
    for (got, r) in [f.x, f.y, f.z].into_iter().zip(rel) {
        assert!(
            (f64_of(got) - speed * r).abs() < 1e-9,
            "{} {}",
            f64_of(got),
            speed * r
        );
    }
    for i in 1..4 {
        assert_eq!(acc.force(i), Some(Vec3Fix::ZERO), "body {i}");
    }
    assert_eq!(p.time(), h + h + h);
}

/// oracle (bounded): a body that never enters the zone moves bit for bit as in
/// a world without the resident, on both backends, with gravity and a floor of
/// other bodies out of the way. A body leaving the zone stops accelerating.
#[test]
fn outside_the_zone_nothing_changes() {
    let far = ZoneShape::Sphere {
        centre: v(100.0, 0.0, 0.0),
        radius: Fix128::ONE,
    };
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let build = || {
            let mut w = world(4, v(0.0, -10.0, 0.0), backend);
            w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)));
            w
        };
        let mut plain = build();
        let mut with = build();
        with.add_participant(resident(zone(far, v(1.0, 0.0, 0.0), 4.0, 0.5, 1.0)))
            .expect("register");
        for _ in 0..64 {
            plain.step(Fix128::from_ratio(1, 64));
            with.try_step(Fix128::from_ratio(1, 64)).expect("step");
        }
        assert_eq!(plain.bodies[0].position, with.bodies[0].position);
        assert_eq!(plain.bodies[0].velocity, with.bodies[0].velocity);
    }
    let edge = ZoneShape::Aabb {
        min: v(-1.0, -1.0, -1.0),
        max: v(1.0, 1.0, 1.0),
    };
    let mut w = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)));
    w.add_participant(resident(zone(edge, v(1.0, 0.0, 0.0), 4.0, 0.0, 0.0)))
        .expect("register");
    let mut left_at = None;
    for _ in 0..640 {
        w.try_step(Fix128::from_ratio(1, 64)).expect("step");
        if w.bodies[0].position.x > Fix128::ONE + fx(0.5) {
            left_at = Some(w.bodies[0].velocity);
            break;
        }
    }
    let left_at = left_at.expect("the wind carries the body out");
    for _ in 0..32 {
        w.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    assert_eq!(w.bodies[0].velocity, left_at, "no force past the edge");
    let bodies = match w.observe_participant(0) {
        Some(Observed::Exact(sink)) => {
            sink.values()
                .iter()
                .find(|(c, _)| *c == WIND_ZONE_OBS_BODIES)
                .expect("channel")
                .1
        }
        other => panic!("{other:?}"),
    };
    assert_eq!(bodies, Fix128::ZERO);
}

/// oracle (count): the bodies channel counts the dynamic bodies inside the zone
/// in the last substep: two of the three here, the third being outside.
#[test]
fn the_observed_count_is_the_bodies_inside() {
    let edge = ZoneShape::Aabb {
        min: v(-1.0, -1.0, -1.0),
        max: v(1.0, 1.0, 1.0),
    };
    let mut w = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    w.add_body(RigidBody::new_dynamic(v(-0.5, 0.0, 0.0), fx(2.0)));
    w.add_body(RigidBody::new_dynamic(v(0.5, 0.0, 0.0), fx(2.0)));
    w.add_body(RigidBody::new_dynamic(v(50.0, 0.0, 0.0), fx(2.0)));
    w.add_participant(resident(zone(edge, v(0.0, 0.0, 1.0), 0.001, 0.0, 0.0)))
        .expect("register");
    w.try_step(Fix128::from_ratio(1, 64)).expect("step");
    match w.observe_participant(0) {
        Some(Observed::Exact(sink)) => {
            let n = sink
                .values()
                .iter()
                .find(|(c, _)| *c == WIND_ZONE_OBS_BODIES)
                .expect("channel")
                .1;
            assert_eq!(n, Fix128::from_int(2));
        }
        other => panic!("{other:?}"),
    }
}

/// oracle (sleep): a body resting long enough to sleep is moved by a wind that
/// reaches it: the fan part placed next to a resting ball.
#[test]
fn a_resting_body_is_woken() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    for _ in 0..240 {
        w.step(Fix128::from_ratio(1, 60));
    }
    w.add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 1.0, 0.0, 0.0)))
        .expect("register");
    for _ in 0..60 {
        w.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    assert!(
        f64_of(w.bodies[0].position.x) > 0.1,
        "{:?}",
        w.bodies[0].position
    );
}

/// The scene the fork and the digests use: two bodies in a gusting wind, one
/// falling out of the zone.
fn gust_scene(backend: SolverBackend) -> PhysicsWorld {
    let shape = ZoneShape::Aabb {
        min: v(-5.0, -3.0, -5.0),
        max: v(5.0, 5.0, 5.0),
    };
    let mut w = world(4, v(0.0, -10.0, 0.0), backend);
    w.add_body(RigidBody::new_dynamic(v(-1.0, 1.0, 0.0), fx(2.0)));
    w.add_body(RigidBody::new_dynamic(v(1.0, 2.0, 0.5), fx(0.5)));
    w.add_participant(resident(zone(shape, v(0.6, 0.0, 0.8), 3.0, 1.0, 1.5)))
        .expect("register");
    w
}

/// oracle (rewind): a world restored from a snapshot and run on is bit for bit
/// the original, so the gust is at the same phase; rewinding the *same* world
/// and running it twice gives the same state. With the clock left out of the
/// payload the gust restarts and this fails.
#[test]
fn rewind_and_fork_are_bit_identical() {
    let dt = Fix128::from_ratio(1, 64);
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut a = gust_scene(backend);
        for _ in 0..50 {
            a.try_step(dt).expect("step");
        }
        let snap = a.snapshot_world();
        let mut b = gust_scene(backend);
        b.restore_world(&snap).expect("restore");
        assert_eq!(b.snapshot_world(), snap);
        for _ in 0..100 {
            a.try_step(dt).expect("step");
            b.try_step(dt).expect("step");
        }
        assert_eq!(a.state_digest(), b.state_digest());
        assert_eq!(a.participant_state(0), b.participant_state(0));

        let end = a.state_digest();
        a.restore_world(&snap).expect("rewind");
        for _ in 0..100 {
            a.try_step(dt).expect("step");
        }
        assert_eq!(a.state_digest(), end, "the rewound run ends where it did");
    }
}

/// oracle (state digest): the clock is state. With no body in the world the
/// resident is the only thing that changes, and the digest must see it; a
/// world whose clock differs from another's digests differently.
#[test]
fn the_clock_is_in_the_state_digest() {
    let dt = Fix128::from_ratio(1, 64);
    let mut w = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    w.add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 1.0, 0.5, 1.0)))
        .expect("register");
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let before = w.state_digest();
    w.try_step(dt).expect("step");
    assert_ne!(w.state_digest(), before);
    match w.observe_participant(0) {
        Some(Observed::Exact(sink)) => {
            let t = sink
                .values()
                .iter()
                .find(|(c, _)| *c == WIND_ZONE_OBS_TIME)
                .expect("channel")
                .1;
            assert_eq!(t, dt);
        }
        other => panic!("{other:?}"),
    }
}

/// oracle (refusal): a snapshot of another zone (another wind speed), of
/// another length and of a world with no resident is refused, and the world
/// stays as it was.
#[test]
fn a_snapshot_of_another_zone_is_refused() {
    let mut a = gust_scene(SolverBackend::Xpbd);
    a.try_step(Fix128::from_ratio(1, 64)).expect("step");
    let snap = a.snapshot_world();
    let mut other = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    other
        .add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 9.0, 0.0, 0.0)))
        .expect("register");
    let before = other.snapshot_world();
    match other.restore_world(&snap) {
        Err(WorldSnapshotError::ParticipantState {
            index: 0,
            error: StateError::InvalidValue,
        }) => {}
        r => panic!("another zone must be refused, got {r:?}"),
    }
    assert_eq!(other.snapshot_world(), before);
    let mut none = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    assert!(
        none.restore_world(&snap).is_err(),
        "no participant: refused"
    );

    let p = WindZoneParticipant::new(zone(big_box(), v(1.0, 0.0, 0.0), 1.0, 0.0, 0.0));
    assert!(matches!(
        p.check_state(&[0u8; 35]),
        Err(StateError::Length {
            expected: 36,
            found: 35
        })
    ));
    let mut good = Vec::new();
    p.write_state(&mut good);
    assert_eq!(good.len(), 36);
    assert_eq!(p.check_state(&good), Ok(()));
    good[0] ^= 1;
    assert_eq!(p.check_state(&good), Err(StateError::InvalidValue));
}

/// oracle (range): a force outside the range of `Fix128` is a fault: the step
/// reports it, the world names the resident, the payload (clock included) and
/// the bodies are as in a world without the resident.
#[test]
fn an_out_of_range_force_is_a_fault_that_changes_nothing() {
    let build = |with: bool| {
        let mut w = world(2, Vec3Fix::ZERO, SolverBackend::Xpbd);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        if with {
            let mut z = zone(big_box(), v(1.0, 0.0, 0.0), 1e9, 0.0, 0.0);
            z.air_density_kg_m3 = fx(1e9);
            w.add_participant(resident(z)).expect("register");
        }
        w
    };
    let mut w = build(true);
    let mut plain = build(false);
    let before = w.participant_state(0);
    let dt = Fix128::from_ratio(1, 64);
    assert!(w.try_step(dt).is_err(), "the step must report the fault");
    plain.step(dt);
    assert_eq!(
        w.fault(),
        Some(WorldFault::Participant {
            index: 0,
            kind: WIND_ZONE_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        })
    );
    assert_eq!(w.participant_state(0), before);
    assert_eq!(w.bodies[0].position, plain.bodies[0].position);
    assert_eq!(w.bodies[0].velocity, plain.bodies[0].velocity);
}

/// oracle (checked path): in range, the checked force the resident stages is
/// bit for bit the public `WindZone::force_on`, inside and outside the zone.
#[test]
fn the_checked_force_equals_force_on_in_range() {
    let z = zone(big_box(), v(0.6, 0.0, 0.8), 3.0, 0.5, 2.0);
    for (p, vel, t) in [
        (Vec3Fix::ZERO, v(1.0, 2.0, 0.0), 0.0),
        (v(1.0, 1.0, 1.0), v(-3.0, 0.5, 7.0), 0.37),
        (v(2e4, 0.0, 0.0), v(1.0, 0.0, 0.0), 1.0),
        (Vec3Fix::ZERO, v(1.8, 0.0, 2.4), 0.0),
    ] {
        let mut b = RigidBody::new_dynamic(p, Fix128::ONE);
        b.velocity = vel;
        assert_eq!(
            z.try_force_on(&b, fx(t)),
            Some(z.force_on(&b, fx(t))),
            "{t}"
        );
    }
}

/// oracle (identity): the resident's kind enters the world's law identifier.
#[test]
fn the_law_identifier_names_the_resident() {
    let plain = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    let mut with = world(4, Vec3Fix::ZERO, SolverBackend::Xpbd);
    with.add_participant(resident(zone(big_box(), v(1.0, 0.0, 0.0), 1.0, 0.0, 0.0)))
        .expect("register");
    assert_ne!(
        plain.law_id(&PHYSICS_SEMANTICS_ID),
        with.law_id(&PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(with.participant_kinds(), vec![WIND_ZONE_PARTICIPANT_KIND]);
}

fn gust_digest(backend: SolverBackend) -> String {
    let mut w = gust_scene(backend);
    for _ in 0..256 {
        w.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    let mut bytes = w.snapshot_world();
    bytes.extend(w.participant_state(0).expect("state"));
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// The snapshot's header holds `PHYSICS_SEMANTICS_ID` and the law id, so these move with the
// identifier: the values pinned under `e90000da…` come back bit for bit from today's snapshot
// with that identifier and its law id written into the header and the checksum (the last 8
// bytes, FNV-1a 64 over the rest) recomputed, so only the header moved.
// re-pinned under aa8c72e9… (world snapshot version 9): from today's blob the main 5fc440bf bytes
// come back exactly with the header ids, the sleep rule and idle times as frames, no recurrence
// count and no lock or cloth section; the bodies and the participant state did not move
const GUST_XPBD: &str = "9022b0380817868c381a80f87069399824c552d155455d930310820106da44b6";
const GUST_TGS: &str = "42b75b30fba2f35497d37e2e6c8f1dadd6bd4c78ca69a967784f3e207e827def";

/// golden (determinism): the world snapshot and the payload of the gust scene
/// after 256 frames, for each backend. CI runs this on every operating system
/// and on wasm32, so the bits are the same everywhere.
#[test]
fn gust_scene_digest_is_pinned() {
    let xpbd = gust_digest(SolverBackend::Xpbd);
    let tgs = gust_digest(SolverBackend::Tgs);
    println!("gust digest xpbd {xpbd}");
    println!("gust digest tgs {tgs}");
    assert_eq!(xpbd, GUST_XPBD);
    assert_eq!(tgs, GUST_TGS);
}

/// oracle (parallel): `try_step_parallel` gives the same blob on 1, 2 and 8
/// threads and, without contacts, the same bodies and payload as `try_step`.
#[cfg(feature = "parallel")]
#[test]
fn parallel_steps_agree_with_each_other_and_with_the_serial_path() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let mut w = gust_scene(SolverBackend::Xpbd);
            for _ in 0..120 {
                w.try_step_parallel(Fix128::from_ratio(1, 64))
                    .expect("step");
            }
            w
        })
    };
    let p1 = run(1);
    for t in [2, 8] {
        assert_eq!(run(t).snapshot_world(), p1.snapshot_world(), "{t} threads");
    }
    let mut s = gust_scene(SolverBackend::Xpbd);
    for _ in 0..120 {
        s.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    for k in 0..2 {
        assert_eq!(s.bodies[k].position, p1.bodies[k].position, "body {k}");
        assert_eq!(s.bodies[k].velocity, p1.bodies[k].velocity, "body {k}");
    }
    assert_eq!(s.participant_state(0), p1.participant_state(0));
}
