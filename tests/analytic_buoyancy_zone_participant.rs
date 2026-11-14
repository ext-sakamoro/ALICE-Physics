//! Oracles for `buoyancy_zone::BuoyancyZoneParticipant`: a bounded fluid that
//! lifts and drags the world's bodies inside its substeps through the
//! participant contract.
//!
//! What is pinned, and against what:
//!
//! * the floating depth of a sphere against Archimedes: `f(h) = m/(ρV)` with the
//!   cap fraction `f = h²(3r − h)/(4r³)`, so a sphere of half the fluid's density
//!   floats with its centre on the surface, and a quarter-density one at the root
//!   of `x³ − 3x² + 1 = 0`, found here by bisection;
//! * the terminal velocity of a sinking sphere against the balance of weight,
//!   lift and drag, for linear and for quadratic drag;
//! * the displaced volume against the world's own radius of each body;
//! * the world side of the contract: outside the zone nothing changes (bit for
//!   bit against a world without the resident), static and kinematic bodies are
//!   left out, a resting body is lifted, the count of bodies in the fluid is in
//!   the state digest, snapshot and fork are bit identical, a snapshot of another
//!   zone is refused, an out-of-range force is a fault that leaves the
//!   participant unchanged, the law identifier names the resident, and a digest
//!   of the stepped world is pinned for each backend.

use alice_physics_core::buoyancy_zone::{
    BuoyancyZone, BuoyancyZoneParticipant, ZoneShape, BUOYANCY_ZONE_OBS_BODIES,
    BUOYANCY_ZONE_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{
    ForceAccumulator, Observed, Participant, ParticipantFault, StateError, SubstepCtx, WorldFault,
};
use alice_physics_core::{
    BodyType, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix,
    WorldSnapshotError, PHYSICS_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

const G: f64 = 9.81;
const FOUR_PI_THIRD: f64 = 4.0 * core::f64::consts::PI / 3.0;

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

/// A deep pool with its surface at `y = 0`.
fn pool() -> ZoneShape {
    ZoneShape::Aabb {
        min: v(-1e3, -1e3, -1e3),
        max: v(1e3, 0.0, 1e3),
    }
}

/// A fluid of density 1, with the world's gravity.
fn fluid(shape: ZoneShape, linear: f64, quadratic: f64) -> BuoyancyZone {
    BuoyancyZone {
        shape,
        density_kg_m3: Fix128::ONE,
        gravity: fx(G),
        drag_linear: fx(linear),
        drag_quadratic: fx(quadratic),
    }
}

fn resident(z: BuoyancyZone) -> Box<BuoyancyZoneParticipant> {
    Box::new(BuoyancyZoneParticipant::new(z))
}

fn world(substeps: usize, backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v(0.0, -G, 0.0),
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    w.config.solver_backend = backend;
    // Kept awake: the floating body approaches its height slowly (most slowly
    // under the temporal Gauss-Seidel backend), and the velocity threshold
    // would put it to sleep, held still, short of the height Archimedes gives
    // (3.4 mm short: the sleep rule reads speed only, not the unbalanced
    // force). Open defect "sleep under unbalanced force": once sleep also
    // requires a balanced net force, this override goes.
    w.set_sleep_config(alice_physics_core::SleepConfig {
        time_to_sleep: Fix128::from_int(1_000_000),
        ..alice_physics_core::SleepConfig::default()
    });
    w
}

/// The height a sphere of radius 1 and the given density settles at.
fn float_height(density: f64, backend: SolverBackend) -> f64 {
    let mut w = world(8, backend);
    let mass = density * FOUR_PI_THIRD;
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, 0.25, 0.0), fx(mass)),
        Fix128::ONE,
    );
    w.add_participant(resident(fluid(pool(), 8.0, 0.0)))
        .expect("register");
    for _ in 0..1800 {
        w.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    f64_of(w.bodies[0].position.y)
}

/// oracle (Archimedes): half the fluid's density floats with the centre on the
/// surface, where the cap fraction is exactly one half.
#[test]
fn a_half_density_sphere_floats_on_its_centre() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let y = float_height(0.5, backend);
        assert!(y.abs() < 1e-6, "{backend:?}: {y}");
    }
}

/// oracle (Archimedes): a quarter of the density floats with `f(h) = 1/4`, that
/// is `x²(3 − x)/4 = 1/4` for `x = h/r`, solved by bisection and independent of
/// the crate's cap formula being right: the
/// centre sits `1 − x` above the surface.
#[test]
fn a_quarter_density_sphere_floats_at_the_cap_root() {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let x = 0.5 * (lo + hi);
        if x * x * (3.0 - x) / 4.0 < 0.25 {
            lo = x;
        } else {
            hi = x;
        }
    }
    let want = 1.0 - lo;
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let y = float_height(0.25, backend);
        assert!((y - want).abs() < 1e-6, "{backend:?}: {y} want {want}");
    }
}

/// The sinking speed of a sphere of radius ½ and twice the fluid's density.
fn sinking_speed(linear: f64, quadratic: f64) -> f64 {
    let mut w = world(8, SolverBackend::Xpbd);
    let volume = FOUR_PI_THIRD * 0.125;
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, -2.0, 0.0), fx(2.0 * volume)),
        fx(0.5),
    );
    w.add_participant(resident(fluid(pool(), linear, quadratic)))
        .expect("register");
    for _ in 0..600 {
        w.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    assert_eq!(f64_of(w.bodies[0].velocity.x), 0.0);
    -f64_of(w.bodies[0].velocity.y)
}

/// oracle (force balance): the net weight `(m − ρV)·g` equals `c·v` at terminal
/// speed under linear drag.
#[test]
fn terminal_speed_under_linear_drag() {
    let net = FOUR_PI_THIRD * 0.125 * G;
    let got = sinking_speed(5.0, 0.0);
    let want = net / 5.0;
    assert!((got - want).abs() < 1e-6 * want, "{got} want {want}");
}

/// oracle (force balance): the net weight equals `c·v²` under quadratic drag.
#[test]
fn terminal_speed_under_quadratic_drag() {
    let net = FOUR_PI_THIRD * 0.125 * G;
    let got = sinking_speed(0.0, 3.0);
    let want = (net / 3.0).sqrt();
    assert!((got - want).abs() < 1e-6 * want, "{got} want {want}");
}

/// oracle (the world's radius): lift goes with the cube of each body's own
/// radius, and a body with none takes the world's default. One substep from
/// rest, fully submerged: `Δv = (ρ·(4/3)π·r³·g/m − g)·h`.
#[test]
fn the_displaced_volume_is_the_worlds_radius_of_each_body() {
    let mut w = world(1, SolverBackend::Xpbd);
    w.set_sdf_collision_radius(fx(0.75));
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(-20.0, -50.0, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, -50.0, 0.0), Fix128::ONE),
        Fix128::ONE,
    );
    w.add_body(RigidBody::new_dynamic(v(20.0, -50.0, 0.0), Fix128::ONE));
    w.add_participant(resident(fluid(pool(), 0.0, 0.0)))
        .expect("register");
    let dt = 1.0 / 64.0;
    w.try_step(Fix128::from_ratio(1, 64)).expect("step");
    for (i, r) in [0.5, 1.0, 0.75].into_iter().enumerate() {
        let want = (FOUR_PI_THIRD * r * r * r * G - G) * dt;
        let got = f64_of(w.bodies[i].velocity.y);
        assert!((got - want).abs() < 1e-9, "body {i}: {got} want {want}");
    }
}

/// oracle (world radius required): a context that has no world's radii cannot
/// say how much a body displaces; the resident says so instead of lifting by
/// nothing.
#[test]
fn a_context_without_radii_is_an_invalid_state() {
    let bodies = [RigidBody::new_dynamic(v(0.0, -5.0, 0.0), Fix128::ONE)];
    let mut acc = ForceAccumulator::new(1);
    let h = Fix128::from_ratio(1, 64);
    let mut ctx = SubstepCtx::new(&bodies, &mut acc, 0, 1, h).expect("ctx");
    let mut p = BuoyancyZoneParticipant::new(fluid(pool(), 0.0, 0.0));
    assert_eq!(p.substep(&mut ctx, h), Err(ParticipantFault::InvalidState));
    assert_eq!(acc.force(0), Some(Vec3Fix::ZERO));
}

/// oracle (who feels it): one substep's force on a dynamic body in the fluid,
/// by hand; static, kinematic and massless bodies and a body outside feel
/// nothing.
#[test]
fn staged_force_matches_the_formula() {
    let mut b0 = RigidBody::new_dynamic(v(0.0, -5.0, 0.0), Fix128::ONE);
    b0.velocity = v(1.0, -2.0, 0.0);
    let mut kin = RigidBody::new_dynamic(v(1.0, -5.0, 0.0), Fix128::ONE);
    kin.body_type = BodyType::Kinematic;
    let massless = RigidBody {
        inv_mass: Fix128::ZERO,
        ..RigidBody::new_dynamic(v(2.0, -5.0, 0.0), Fix128::ONE)
    };
    let bodies = [
        b0,
        RigidBody::new_static(v(3.0, -5.0, 0.0)),
        kin,
        massless,
        RigidBody::new_dynamic(v(4.0, 5.0, 0.0), Fix128::ONE),
    ];
    let radii = [Some(fx(0.5)); 5];
    let mut acc = ForceAccumulator::new(5);
    let h = Fix128::from_ratio(1, 64);
    let ctx = SubstepCtx::new(&bodies, &mut acc, 0, 1, h).expect("ctx");
    let mut ctx = ctx.with_body_radii(&radii, Fix128::ONE);
    let mut p = BuoyancyZoneParticipant::new(fluid(pool(), 0.5, 2.0));
    p.substep(&mut ctx, h).expect("substep");
    let speed = 5.0_f64.sqrt();
    let lift = FOUR_PI_THIRD * 0.125 * G;
    let want = [
        -0.5 * 1.0 - 2.0 * speed * 1.0,
        lift + 0.5 * 2.0 + 2.0 * speed * 2.0,
    ];
    let f = acc.force(0).expect("slot");
    assert!((f64_of(f.x) - want[0]).abs() < 1e-9, "{}", f64_of(f.x));
    assert!((f64_of(f.y) - want[1]).abs() < 1e-9, "{}", f64_of(f.y));
    assert_eq!(f.z, Fix128::ZERO);
    for i in 1..5 {
        assert_eq!(acc.force(i), Some(Vec3Fix::ZERO), "body {i}");
    }
}

/// oracle (bounded): a body that never reaches the fluid moves bit for bit as in
/// a world without the resident, on both backends. The count of bodies in the
/// fluid goes to zero when the last one leaves.
#[test]
fn outside_the_zone_nothing_changes() {
    let far = ZoneShape::Sphere {
        centre: v(100.0, 0.0, 0.0),
        radius: Fix128::ONE,
    };
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let build = || {
            let mut w = world(4, backend);
            w.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)), Fix128::ONE);
            w
        };
        let mut plain = build();
        let mut with = build();
        with.add_participant(resident(fluid(far, 1.0, 1.0)))
            .expect("register");
        for _ in 0..64 {
            plain.step(Fix128::from_ratio(1, 64));
            with.try_step(Fix128::from_ratio(1, 64)).expect("step");
        }
        assert_eq!(plain.bodies[0].position, with.bodies[0].position);
        assert_eq!(plain.bodies[0].velocity, with.bodies[0].velocity);
        match with.observe_participant(0) {
            Some(Observed::Exact(sink)) => {
                let n = sink
                    .values()
                    .iter()
                    .find(|(c, _)| *c == BUOYANCY_ZONE_OBS_BODIES)
                    .expect("channel")
                    .1;
                assert_eq!(n, Fix128::ZERO);
            }
            other => panic!("{other:?}"),
        }
    }
}

/// oracle (sleep): a body resting long enough to sleep is lifted by water
/// placed around it.
#[test]
fn a_resting_body_is_woken() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, -5.0, 0.0), Fix128::ONE),
        Fix128::ONE,
    );
    for _ in 0..240 {
        w.step(Fix128::from_ratio(1, 60));
    }
    w.add_participant(resident(fluid(pool(), 0.0, 0.0)))
        .expect("register");
    for _ in 0..60 {
        w.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    assert!(
        f64_of(w.bodies[0].position.y) > -4.0,
        "{:?}",
        w.bodies[0].position
    );
}

/// The scene the fork and the digests use: two bodies dropped into the pool,
/// one past its side.
fn splash_scene(backend: SolverBackend) -> PhysicsWorld {
    let shape = ZoneShape::Aabb {
        min: v(-3.0, -6.0, -3.0),
        max: v(3.0, 0.0, 3.0),
    };
    let mut w = world(4, backend);
    w.add_body_with_radius(RigidBody::new_dynamic(v(-1.0, 1.0, 0.0), fx(2.0)), fx(0.6));
    w.add_body_with_radius(RigidBody::new_dynamic(v(2.9, 2.0, 0.5), fx(0.5)), fx(0.4));
    w.add_participant(resident(fluid(shape, 1.5, 0.5)))
        .expect("register");
    w
}

/// oracle (rewind): a world restored from a snapshot and run on is bit for bit
/// the original, and rewinding the same world runs the same again.
#[test]
fn rewind_and_fork_are_bit_identical() {
    let dt = Fix128::from_ratio(1, 64);
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut a = splash_scene(backend);
        for _ in 0..50 {
            a.try_step(dt).expect("step");
        }
        let snap = a.snapshot_world();
        let mut b = splash_scene(backend);
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

/// oracle (state digest): the count of bodies in the fluid is state. A static
/// world with one dynamic body parked by a zero mass change sees it change on
/// the first substep.
#[test]
fn the_count_is_in_the_state_digest() {
    let dt = Fix128::from_ratio(1, 64);
    let mut w = world(4, SolverBackend::Xpbd);
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, -5.0, 0.0), Fix128::ONE),
        Fix128::ONE,
    );
    w.add_participant(resident(fluid(pool(), 0.0, 0.0)))
        .expect("register");
    let before = w.participant_state(0);
    w.try_step(dt).expect("step");
    assert_ne!(w.participant_state(0), before);
    let p = BuoyancyZoneParticipant::new(fluid(pool(), 0.0, 0.0));
    let mut bytes = Vec::new();
    p.write_state(&mut bytes);
    assert_eq!(bytes.len(), 20);
    let after = w.participant_state(0).expect("state");
    assert_eq!(after.len(), 20);
    assert_eq!(&after[12..20], &1u64.to_le_bytes(), "one body in the fluid");
}

/// oracle (refusal): a snapshot of another zone, of another length and of a
/// world with no resident is refused, and the world stays as it was.
#[test]
fn a_snapshot_of_another_zone_is_refused() {
    let mut a = splash_scene(SolverBackend::Xpbd);
    a.try_step(Fix128::from_ratio(1, 64)).expect("step");
    let snap = a.snapshot_world();
    let mut other = world(4, SolverBackend::Xpbd);
    other
        .add_participant(resident(fluid(pool(), 9.0, 0.0)))
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
    let mut none = world(4, SolverBackend::Xpbd);
    assert!(
        none.restore_world(&snap).is_err(),
        "no participant: refused"
    );

    let p = BuoyancyZoneParticipant::new(fluid(pool(), 1.0, 0.0));
    assert!(matches!(
        p.check_state(&[0u8; 19]),
        Err(StateError::Length {
            expected: 20,
            found: 19
        })
    ));
    let mut good = Vec::new();
    p.write_state(&mut good);
    assert_eq!(p.check_state(&good), Ok(()));
    good[0] ^= 1;
    assert_eq!(p.check_state(&good), Err(StateError::InvalidValue));
}

/// oracle (range): a lift outside the range of `Fix128` is a fault: the step
/// reports it, the world names the resident, the payload and the bodies are as
/// in a world without the resident.
#[test]
fn an_out_of_range_force_is_a_fault_that_changes_nothing() {
    let build = |with: bool| {
        let mut w = world(2, SolverBackend::Xpbd);
        w.add_body_with_radius(
            RigidBody::new_dynamic(v(0.0, -5.0, 0.0), Fix128::ONE),
            Fix128::ONE,
        );
        if with {
            let mut z = fluid(pool(), 0.0, 0.0);
            z.density_kg_m3 = fx(1e18);
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
            kind: BUOYANCY_ZONE_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        })
    );
    assert_eq!(w.participant_state(0), before);
    assert_eq!(w.bodies[0].position, plain.bodies[0].position);
    assert_eq!(w.bodies[0].velocity, plain.bodies[0].velocity);
}

/// oracle: a body beyond the box on any side feels nothing, a body in it the
/// full lift of `rho * g * V`.
#[test]
fn the_box_bounds_the_fluid_on_every_side() {
    let z = fluid(pool(), 0.0, 0.0);
    let at = |x: f64, y: f64, zz: f64| RigidBody::new_dynamic(v(x, y, zz), Fix128::ONE);
    for (x, y, zz) in [
        (2e3, -5.0, 0.0),
        (-2e3, -5.0, 0.0),
        (0.0, -5.0, 2e3),
        (0.0, -5.0, -2e3),
        (0.0, -2e3, 0.0),
        (0.0, 5.0, 0.0),
    ] {
        assert_eq!(
            z.try_force_on(&at(x, y, zz), Fix128::ONE),
            Some(Vec3Fix::ZERO),
            "at ({x}, {y}, {zz})"
        );
    }
    let lift = f64_of(
        z.try_force_on(&at(0.0, -5.0, 0.0), Fix128::ONE)
            .expect("in range")
            .y,
    );
    assert!((lift - G * FOUR_PI_THIRD).abs() < 1e-9, "lift {lift}");
}

/// oracle: a spherical zone holds what is within its radius horizontally and
/// nothing else. A body far along one axis is outside even where the square of
/// that distance is not representable, because no square is needed to see it.
#[test]
fn the_sphere_bounds_the_fluid_and_far_bodies_stay_in_range() {
    let z = fluid(
        ZoneShape::Sphere {
            centre: Vec3Fix::ZERO,
            radius: fx(10.0),
        },
        0.0,
        0.0,
    );
    let at = |x: f64, y: f64, zz: f64| RigidBody::new_dynamic(v(x, y, zz), Fix128::ONE);
    for (x, y, zz) in [
        (20.0, 0.0, 0.0),
        (0.0, 0.0, -20.0),
        (8.0, 0.0, 8.0),
        (0.0, -20.0, 0.0),
        (4e9, 0.0, 0.0),
        (0.0, 0.0, -4e9),
    ] {
        assert_eq!(
            z.try_force_on(&at(x, y, zz), Fix128::ONE),
            Some(Vec3Fix::ZERO),
            "at ({x}, {y}, {zz})"
        );
    }
    let lift = f64_of(
        z.try_force_on(&at(0.0, 0.0, 0.0), Fix128::ONE)
            .expect("in range")
            .y,
    );
    assert!((lift - G * FOUR_PI_THIRD).abs() < 1e-9, "lift {lift}");
}

/// oracle: a cap fraction out of range (a radius whose cube is not
/// representable, half under the surface) is a fault, not a body with no lift.
#[test]
fn an_out_of_range_fraction_is_a_fault() {
    let mut w = world(2, SolverBackend::Xpbd);
    w.add_body_with_radius(
        RigidBody::new_dynamic(v(0.0, 0.0, 0.0), Fix128::ONE),
        fx(3e6),
    );
    w.add_participant(resident(fluid(pool(), 0.0, 0.0)))
        .expect("register");
    assert!(w.try_step(Fix128::from_ratio(1, 64)).is_err());
    assert_eq!(
        w.fault(),
        Some(WorldFault::Participant {
            index: 0,
            kind: BUOYANCY_ZONE_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        })
    );
}

/// oracle: a payload of another layout version or another zone is refused byte
/// by byte, whatever the rest of it says.
#[test]
fn a_tampered_payload_is_refused() {
    let p = BuoyancyZoneParticipant::new(fluid(pool(), 0.0, 0.0));
    let mut good = Vec::new();
    p.write_state(&mut good);
    assert_eq!(p.check_state(&good), Ok(()));
    let mut version = good.clone();
    version[0] ^= 1;
    assert_eq!(p.check_state(&version), Err(StateError::InvalidValue));
    let mut digest = good.clone();
    digest[4] ^= 1;
    assert_eq!(p.check_state(&digest), Err(StateError::InvalidValue));
}

/// oracle (identity): in range the checked force is `force_on`, bit for bit,
/// at the surface, inside, outside and at rest.
#[test]
fn the_checked_force_equals_force_on_in_range() {
    let z = fluid(pool(), 0.7, 1.3);
    for (y, vel) in [
        (-5.0, v(1.0, -2.0, 0.5)),
        (0.0, v(0.0, 0.0, 0.0)),
        (0.25, v(-3.0, 1.0, 2.0)),
        (5.0, v(1.0, 0.0, 0.0)),
    ] {
        let mut b = RigidBody::new_dynamic(v(0.0, y, 0.0), Fix128::ONE);
        b.velocity = vel;
        assert_eq!(
            z.try_force_on(&b, fx(0.5)),
            Some(z.force_on(&b, fx(0.5))),
            "{y}"
        );
    }
}

/// oracle (identity): the resident's kind enters the world's law identifier.
#[test]
fn the_law_identifier_names_the_resident() {
    let plain = world(4, SolverBackend::Xpbd);
    let mut with = world(4, SolverBackend::Xpbd);
    with.add_participant(resident(fluid(pool(), 1.0, 0.0)))
        .expect("register");
    assert_ne!(
        plain.law_id(&PHYSICS_SEMANTICS_ID),
        with.law_id(&PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(
        with.participant_kinds(),
        vec![BUOYANCY_ZONE_PARTICIPANT_KIND]
    );
}

fn splash_digest(backend: SolverBackend) -> String {
    let mut w = splash_scene(backend);
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
const SPLASH_XPBD: &str = "761d13b1772155c1ff5eaf4a3c98400cfee2277e46f21ff5cccdc10f614637bb";
const SPLASH_TGS: &str = "ad2723f9aeb1de2026b4535bbf5b97460bbeb449c023b2fdc699d0ab1ae60e6b";

/// golden (determinism): the world snapshot and the payload of the splash scene
/// after 256 frames, for each backend. CI runs this on every operating system
/// and on wasm32, so the bits are the same everywhere.
#[test]
fn splash_scene_digest_is_pinned() {
    let xpbd = splash_digest(SolverBackend::Xpbd);
    let tgs = splash_digest(SolverBackend::Tgs);
    println!("splash digest xpbd {xpbd}");
    println!("splash digest tgs {tgs}");
    assert_eq!(xpbd, SPLASH_XPBD);
    assert_eq!(tgs, SPLASH_TGS);
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
            let mut w = splash_scene(SolverBackend::Xpbd);
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
    let mut s = splash_scene(SolverBackend::Xpbd);
    for _ in 0..120 {
        s.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    for k in 0..2 {
        assert_eq!(s.bodies[k].position, p1.bodies[k].position, "body {k}");
        assert_eq!(s.bodies[k].velocity, p1.bodies[k].velocity, "body {k}");
    }
    assert_eq!(s.participant_state(0), p1.participant_state(0));
}
