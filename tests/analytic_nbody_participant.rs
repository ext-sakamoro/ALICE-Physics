//! Oracles for `nbody::NBodyParticipant`: direct-sum gravity among the
//! world's bodies, run inside the world's substeps through the participant
//! contract.
//!
//! What is pinned, and against what:
//!
//! * the staged forces of one substep against Newton's third law (they sum
//!   to exactly zero) and against `DirectSum::accelerations` /
//!   `DirectSum::potential_energy` on the same positions;
//! * the motion of a circular pair against the closed form of symplectic
//!   Euler with the substep width `h`: the orbit becomes an ellipse of
//!   eccentricity `h/2`, so the separation swings `±d·h/2` (first order in
//!   the substep width, for both backends);
//! * the angular momentum (conserved exactly by symplectic Euler under a
//!   central force, so only rounding is left), the linear momentum (the
//!   staged forces cancel, so only rounding is left) and the energy (bounded,
//!   no drift over ten orbits);
//! * the world side of the contract: static and kinematic bodies are left
//!   out, a slow attraction wakes sleeping bodies, snapshot and fork are bit
//!   identical, a snapshot of another law is refused, an out-of-range force
//!   is a fault that leaves the participant unchanged, the law identifier
//!   names the participant, and a digest of the stepped world is pinned.

use alice_physics_core::nbody::{
    kinetic_energy, total_momentum, DirectSum, NBodyParticipant, NBODY_OBS_BODIES,
    NBODY_OBS_POTENTIAL, NBODY_PARTICIPANT_KIND,
};
use alice_physics_core::world_participant::{
    ForceAccumulator, ObservationSink, Observed, Participant, ParticipantFault, StateError,
    SubstepCtx, WorldFault,
};
use alice_physics_core::{
    BodyType, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix,
    WorldSnapshotError, PHYSICS_SEMANTICS_ID,
};
use core::f64::consts::PI;
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

fn v3(v: Vec3Fix) -> [f64; 3] {
    [f64_of(v.x), f64_of(v.y), f64_of(v.z)]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn law(g: f64, eps: f64) -> DirectSum {
    DirectSum::new(fx(g), fx(eps)).expect("law")
}

fn resident(g: f64, eps: f64) -> Box<NBodyParticipant> {
    Box::new(NBodyParticipant::new(law(g, eps)))
}

/// A world with no gravity, no damping and no sleep for orbits.
fn empty_world(substeps: usize, backend: SolverBackend) -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.config.solver_backend = backend;
    world
}

/// Two bodies of mass `1/2` at `(±1/2, 0, 0)` moving at `(0, ±1/2, 0)` with
/// `G = 1`: a circular orbit of separation `d = 1`, angular rate `ω = 1`,
/// period `2π`.
fn circular_pair(substeps: usize, backend: SolverBackend) -> PhysicsWorld {
    let half = Fix128::from_ratio(1, 2);
    let mut world = empty_world(substeps, backend);
    for s in [Fix128::ONE, -Fix128::ONE] {
        let mut b =
            RigidBody::new_dynamic(Vec3Fix::new(half * s, Fix128::ZERO, Fix128::ZERO), half);
        b.velocity = Vec3Fix::new(Fix128::ZERO, half * s, Fix128::ZERO);
        world.add_body(b);
    }
    world.add_participant(resident(1.0, 0.0)).expect("register");
    world
}

/// A hierarchical three-body scene with unequal masses (`4`, `1`, `1/2`:
/// their inverses are exact): a body of mass 1 on a near-circular orbit of
/// radius 2 and one of mass 1/2 at radius 5 around a body of mass 4, the
/// latter moving so that the total momentum is zero. The two light bodies
/// perturb each other but never come close (periods about 9 and 35).
fn three_bodies(substeps: usize, backend: SolverBackend) -> PhysicsWorld {
    let mut world = empty_world(substeps, backend);
    let (v1, v2) = (2.0_f64.sqrt(), (4.0_f64 / 5.0).sqrt());
    let spec = [
        (
            4.0,
            [0.0, 0.0, 0.0],
            [0.0, -(v1 - 0.5 * v2) / 4.0, -0.05 / 4.0],
        ),
        (1.0, [2.0, 0.0, 0.0], [0.0, v1, 0.05]),
        (0.5, [-5.0, 0.0, 0.0], [0.0, -v2, 0.0]),
    ];
    for (m, p, v) in spec {
        let mut b = RigidBody::new_dynamic(Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2])), fx(m));
        b.velocity = Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]));
        world.add_body(b);
    }
    world.add_participant(resident(1.0, 0.0)).expect("register");
    world
}

fn dynamic_state(world: &PhysicsWorld) -> (Vec<Vec3Fix>, Vec<Vec3Fix>, Vec<Fix128>) {
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut mass = Vec::new();
    for b in &world.bodies {
        if b.body_type == BodyType::Dynamic {
            pos.push(b.position);
            vel.push(b.velocity);
            mass.push(Fix128::ONE / b.inv_mass);
        }
    }
    (pos, vel, mass)
}

fn observed(world: &PhysicsWorld, index: usize) -> Vec<(u32, Fix128)> {
    match world.observe_participant(index).expect("participant") {
        Observed::Exact(sink) => sink.values().to_vec(),
        Observed::Undecided => panic!("participant observation undecided"),
    }
}

/// `(potential, body count)` a participant observes.
fn seen(p: &NBodyParticipant) -> (Fix128, Fix128) {
    let mut sink = ObservationSink::new();
    p.observe(&mut sink);
    let v = sink.values();
    assert_eq!(v.len(), 2);
    assert_eq!((v[0].0, v[1].0), (NBODY_OBS_POTENTIAL, NBODY_OBS_BODIES));
    (v[0].1, v[1].1)
}

/// oracle (Newton's third law, one substep): for three dynamic bodies of
/// masses 1, 2, 4 and a static body, the staged forces sum to exactly zero,
/// the static body gets none, each force equals `mᵢ·aᵢ` of
/// `DirectSum::accelerations` to `1e-15` relative, and the observed
/// potential is bit for bit `DirectSum::potential_energy` (the masses are
/// read back from `inv_mass` exactly because their inverses are exact).
#[test]
fn staged_forces_are_equal_and_opposite_and_match_the_direct_sum() {
    let mut bodies = Vec::new();
    let spec = [
        (1.0, [2.0, 0.0, 0.0]),
        (2.0, [-1.0, 0.5, 0.0]),
        (4.0, [0.0, -0.25, 0.5]),
    ];
    for (m, p) in spec {
        bodies.push(RigidBody::new_dynamic(
            Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2])),
            fx(m),
        ));
    }
    bodies.insert(
        1,
        RigidBody::new_static(Vec3Fix::new(fx(0.5), fx(0.5), fx(0.5))),
    );
    let mut p = NBodyParticipant::new(law(1.5, 0.0));
    let mut acc = ForceAccumulator::new(bodies.len());
    {
        let mut ctx =
            SubstepCtx::new(&bodies, &mut acc, 0, 1, Fix128::from_ratio(1, 64)).expect("ctx");
        p.substep(&mut ctx, Fix128::from_ratio(1, 64))
            .expect("substep");
    }
    let mut sum = Vec3Fix::ZERO;
    for i in 0..bodies.len() {
        sum = sum + acc.force(i).expect("slot");
        assert_eq!(acc.torque(i), Some(Vec3Fix::ZERO), "no torque on body {i}");
    }
    assert_eq!(sum, Vec3Fix::ZERO, "staged forces must cancel exactly");
    assert_eq!(
        acc.force(1),
        Some(Vec3Fix::ZERO),
        "static body takes no force"
    );

    let dynamic = [0usize, 2, 3];
    let pos: Vec<Vec3Fix> = dynamic.iter().map(|&i| bodies[i].position).collect();
    let mass: Vec<Fix128> = spec.iter().map(|s| fx(s.0)).collect();
    let mut a = Vec::new();
    law(1.5, 0.0)
        .accelerations(&pos, &mass, &mut a)
        .expect("accelerations");
    for (k, &i) in dynamic.iter().enumerate() {
        let f = v3(acc.force(i).expect("slot"));
        let want = v3(a[k]);
        let m = spec[k].0;
        for c in 0..3 {
            let w = m * want[c];
            assert!(
                (f[c] - w).abs() <= 1e-15 * (1.0 + w.abs()),
                "body {i} component {c}: force {} vs m·a {w}",
                f[c]
            );
        }
    }
    assert_eq!(
        seen(&p),
        (
            law(1.5, 0.0)
                .potential_energy(&pos, &mass)
                .expect("potential"),
            Fix128::from_int(3)
        ),
        "observed potential is the direct sum's"
    );
}

/// Separation extremes of the circular pair over one period of `frames`
/// frames.
fn swing(substeps: usize, frames: i64, backend: SolverBackend) -> (f64, f64) {
    let dt = Fix128::TWO_PI / Fix128::from_int(frames);
    let mut world = circular_pair(substeps, backend);
    let (mut rmax, mut rmin) = (1.0_f64, 1.0_f64);
    for _ in 0..frames {
        world.try_step(dt).expect("step");
        let r = norm(v3(world.bodies[0].position - world.bodies[1].position));
        rmax = rmax.max(r);
        rmin = rmin.min(r);
    }
    (rmax - 1.0, 1.0 - rmin)
}

/// oracle (closed form, symplectic Euler): kick then drift with width `h`
/// starts the circular pair with the extra radial velocity `ω²d·h/2`, which
/// is an ellipse of eccentricity `e = h/2`: the separation swings `±d·h/2`.
/// Inside the world the width is the substep `h = dt/s`, so the swing is
/// asserted to 5 % of `h/2` for `s = 1` and `s = 4` (measured 1.6 % and
/// 0.4 %: the next term is `O(h)` relative; the frame-head kick of
/// `DirectSum::kick_bodies` keeps `dt/2` whatever `s`, see
/// `analytic_nbody::frame_head_kick_is_first_order`). Both backends (TGS
/// divides the frame with `1/s` rounded to `f32`, exact for `s = 1, 4`).
#[test]
fn swing_is_half_the_substep_width() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for substeps in [1usize, 4] {
            let frames = 400_i64;
            let h = 2.0 * PI / (frames as f64 * substeps as f64);
            let (up, down) = swing(substeps, frames, backend);
            for (name, dev) in [("max |d| − 1", up), ("1 − min |d|", down)] {
                assert!(
                    (dev / (h / 2.0) - 1.0).abs() < 0.05,
                    "{backend:?} s={substeps}: {name} = {dev:e}, h/2 = {:e}",
                    h / 2.0
                );
            }
        }
    }
}

/// `L = Σ m·x×v` in `f64`.
fn angular_momentum(pos: &[Vec3Fix], vel: &[Vec3Fix], mass: &[Fix128]) -> [f64; 3] {
    let mut l = [0.0; 3];
    for ((p, v), m) in pos.iter().zip(vel).zip(mass) {
        let (p, v, m) = (v3(*p), v3(*v), f64_of(*m));
        l[0] += m * (p[1] * v[2] - p[2] * v[1]);
        l[1] += m * (p[2] * v[0] - p[0] * v[2]);
        l[2] += m * (p[0] * v[1] - p[1] * v[0]);
    }
    l
}

fn energy(world: &PhysicsWorld) -> f64 {
    let (pos, vel, mass) = dynamic_state(world);
    f64_of(kinetic_energy(&vel, &mass).expect("ke"))
        + f64_of(law(1.0, 0.0).potential_energy(&pos, &mass).expect("pe"))
}

/// oracle (conservation): over 100 time units of the three-body scene
/// (`dt = 1/64`, 4 substeps, 6400 frames, about 11 inner orbits):
///
/// * symplectic Euler under a central pair force keeps `Σ m·x×v` exactly in
///   exact arithmetic (the kick is parallel to each pair's separation, the
///   drift is parallel to the velocity), so only the rounding of `F·h/m`
///   and `x + v·h` is left: `< 2e-14` (each of the `25 600` substeps
///   truncates a velocity by `≤ 2⁻⁶³` and a position by `≤ 2⁻⁶⁴`, times
///   masses `≤ 4`, lever arms `≤ 5` and speeds `≤ 1.5`: a linear worst case
///   of `≈ 1.4·10⁻¹⁴` for the largest term, so the assertion sits above
///   it; measured `9·10⁻¹⁶`);
/// * the staged forces cancel exactly, so `Σ m·v` changes only by the
///   truncation of `F·inv_mass·h`: `≤ m·2⁻⁶³` per body and substep, `Σ m =
///   5.5`, so `≤ 1.5·10⁻¹⁴` over the run (asserted `< 2e-14`, measured
///   `3.8·10⁻¹⁵`);
/// * the energy error of a symplectic scheme stays bounded: its largest
///   excursion in the last 640 frames is within 1.5 times that of the first
///   640 frames (measured 1.16) and below `h = 1/256` (measured `3·10⁻⁴`):
///   no secular drift.
#[test]
fn momenta_are_conserved_and_energy_is_bounded() {
    let mut world = three_bodies(4, SolverBackend::Xpbd);
    let dt = Fix128::from_ratio(1, 64);
    let (pos, vel, mass) = dynamic_state(&world);
    let l0 = angular_momentum(&pos, &vel, &mass);
    let p0 = v3(total_momentum(&vel, &mass).expect("p"));
    let e0 = energy(&world);
    let (mut first, mut last) = (0.0_f64, 0.0_f64);
    let frames = 6400;
    for n in 0..frames {
        world.try_step(dt).expect("step");
        let de = (energy(&world) - e0).abs();
        if n < frames / 10 {
            first = first.max(de);
        } else if n >= frames - frames / 10 {
            last = last.max(de);
        }
    }
    let (pos, vel, mass) = dynamic_state(&world);
    let l1 = angular_momentum(&pos, &vel, &mass);
    let p1 = v3(total_momentum(&vel, &mass).expect("p"));
    for c in 0..3 {
        assert!(
            (l1[c] - l0[c]).abs() < 2e-14,
            "L[{c}] {} vs {}",
            l1[c],
            l0[c]
        );
        assert!(
            (p1[c] - p0[c]).abs() < 2e-14,
            "P[{c}] {} vs {}",
            p1[c],
            p0[c]
        );
    }
    assert!(first > 0.0, "the scene must actually exchange energy");
    assert!(
        last <= 1.5 * first,
        "energy drift: first {first:e}, last {last:e}"
    );
    assert!(last < 1.0 / 256.0, "energy error {last:e}");
}

/// oracle (who takes part): a lone dynamic body next to a static and a
/// kinematic body feels nothing and the participant reports one body (the
/// static and kinematic bodies have no finite mass, as in
/// `DirectSum::kick_bodies`).
#[test]
fn static_and_kinematic_bodies_neither_feel_nor_exert() {
    let mut world = empty_world(4, SolverBackend::Xpbd);
    world.add_body(RigidBody::new_static(Vec3Fix::new(
        fx(1.0),
        Fix128::ZERO,
        Fix128::ZERO,
    )));
    let mut kin =
        RigidBody::new_dynamic(Vec3Fix::new(fx(-1.0), Fix128::ZERO, Fix128::ZERO), fx(1.0));
    // the body type alone must exclude it: its `inv_mass` stays finite
    kin.body_type = BodyType::Kinematic;
    world.add_body(kin);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, fx(1.0)));
    world.add_participant(resident(1.0, 0.0)).expect("register");
    for _ in 0..30 {
        world.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    assert_eq!(world.bodies[2].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[2].velocity, Vec3Fix::ZERO);
    let obs = observed(&world, 0);
    assert_eq!(obs[1], (NBODY_OBS_BODIES, Fix128::ONE));
    assert_eq!(obs[0], (NBODY_OBS_POTENTIAL, Fix128::ZERO));
}

/// oracle (sleep): two bodies released from rest with `G = 10⁻⁴`, masses 1
/// and separation 2 attract with `a = G·m/d² = 2.5·10⁻⁵`, slower than the
/// default sleep threshold, so a world without the attraction puts them to
/// sleep after 60 frames. The participant's force is non-zero, the world
/// wakes them, and after 240 frames (4 s) each has moved toward the other by
/// `a·t²/2 = 2·10⁻⁴` to 2 % (the separation shrinks by `4·10⁻⁸` relative,
/// so the acceleration is constant to well within that; measured 0.7 % off
/// the per-frame model, which kicks once per frame where the world kicks
/// per substep). The world's sleep flag is not asserted: at a frame end it
/// may report a body put to sleep that the next substep's force wakes; the
/// displacement is what shows the attraction acted.
#[test]
fn a_slow_attraction_keeps_bodies_awake() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));
    world
        .add_participant(resident(1e-4, 0.0))
        .expect("register");
    let dt = Fix128::from_ratio(1, 60);
    let frames = 240;
    for _ in 0..frames {
        world.try_step(dt).expect("step");
    }
    let t = f64::from(frames) / 60.0;
    let want = 2.5e-5 * t * t / 2.0;
    let damping = f64_of((-(PhysicsConfig::default().damping_rate / Fix128::from_int(60))).exp());
    let moved = f64_of(world.bodies[0].position.x) + 1.0;
    // the default frame damping scales the velocity each frame; with it the
    // displacement is the damped sum, computed here the same way
    let mut v = 0.0_f64;
    let mut x = 0.0_f64;
    for _ in 0..frames {
        v += 2.5e-5 / 60.0;
        x += v / 60.0;
        v *= damping;
    }
    assert!(moved > 0.0, "body 0 must move toward body 1");
    assert!(
        (moved / x - 1.0).abs() < 0.02,
        "moved {moved:e}, damped closed form {x:e} (undamped {want:e})"
    );
}

/// oracle (snapshot): a world forked from a snapshot taken after 50 frames
/// and stepped 100 more is bit for bit the original (blob and participant
/// payload); a world whose participant has another `G` refuses the
/// snapshot without changing, as does a world with no participant.
#[test]
fn snapshot_fork_is_bit_identical_and_another_law_is_refused() {
    let dt = Fix128::from_ratio(1, 64);
    let mut a = three_bodies(4, SolverBackend::Xpbd);
    for _ in 0..50 {
        a.try_step(dt).expect("step");
    }
    let snap = a.snapshot_world();
    let mut b = three_bodies(4, SolverBackend::Xpbd);
    b.restore_world(&snap).expect("restore");
    assert_eq!(b.snapshot_world(), snap);
    // the payload carries what the participant last observed
    assert_eq!(observed(&b, 0), observed(&a, 0));
    for _ in 0..100 {
        a.try_step(dt).expect("step");
        b.try_step(dt).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    assert_eq!(a.participant_state(0), b.participant_state(0));
    assert_eq!(observed(&a, 0), observed(&b, 0));

    let mut other = empty_world(4, SolverBackend::Xpbd);
    other.add_participant(resident(2.0, 0.0)).expect("register");
    let before = other.snapshot_world();
    match other.restore_world(&snap) {
        Err(WorldSnapshotError::ParticipantState {
            index: 0,
            error: StateError::InvalidValue,
        }) => {}
        r => panic!("another law must be refused, got {r:?}"),
    }
    assert_eq!(other.snapshot_world(), before);

    let mut none = empty_world(4, SolverBackend::Xpbd);
    assert!(
        none.restore_world(&snap).is_err(),
        "no participant: refused"
    );
}

/// oracle (range): `G = 2⁶⁰` between bodies `2⁻¹⁰` apart makes the pair
/// force leave the range of `Fix128`. The substep fails with `OutOfRange`,
/// the world records the fault naming the participant, the participant's
/// payload is the one it had before, and the bodies move as in a world
/// without the participant (nothing was staged).
#[test]
fn an_out_of_range_force_is_a_fault_that_changes_nothing() {
    let build = |with: bool| {
        let mut w = empty_world(2, SolverBackend::Xpbd);
        let d = Fix128::from_ratio(1, 1024);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut b =
            RigidBody::new_dynamic(Vec3Fix::new(d, Fix128::ZERO, Fix128::ZERO), Fix128::ONE);
        b.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        w.add_body(b);
        if with {
            w.add_participant(Box::new(NBodyParticipant::new(
                DirectSum::new(Fix128::from_int(1 << 60), Fix128::ZERO).expect("law"),
            )))
            .expect("register");
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
            kind: NBODY_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        })
    );
    assert_eq!(w.participant_state(0), before);
    for k in 0..2 {
        assert_eq!(w.bodies[k].position, plain.bodies[k].position, "body {k}");
        assert_eq!(w.bodies[k].velocity, plain.bodies[k].velocity, "body {k}");
    }
}

/// oracle (identity): the participant's kind enters the world's law
/// identifier, and the crate's semantics identifier is the one the world is
/// asked with (a resident is a new law in the world, not a change of the
/// existing ones' semantics).
#[test]
fn the_law_identifier_names_the_resident() {
    let mut plain = empty_world(4, SolverBackend::Xpbd);
    plain.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut with = empty_world(4, SolverBackend::Xpbd);
    with.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    with.add_participant(resident(1.0, 0.0)).expect("register");
    assert_ne!(
        plain.law_id(&PHYSICS_SEMANTICS_ID),
        with.law_id(&PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(with.participant_kinds(), vec![NBODY_PARTICIPANT_KIND]);
}

/// Digest of the three-body scene after 256 frames.
fn three_body_digest(backend: SolverBackend) -> String {
    let mut world = three_bodies(4, backend);
    for _ in 0..256 {
        world.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    let mut bytes = world.snapshot_world();
    bytes.extend(world.participant_state(0).expect("state"));
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// world snapshot version 5: the version field and the checksum moved, the rest of the bytes did not
// re-pinned when the secondary chain was pinned: the header's semantics identifier (88cf6f85… ->
// 31eb4ab8…), its law_id and the checksum moved; the state did not (tests/world_snapshot_xv_88cf6f85.rs)
// re-pinned for world snapshot version 6 (damping a rate per second, body rates in the blob) and
// the semantics identifier e90000da…: the header, the version, the damping fields and the checksum
// moved; the bodies and the participant payload did not (no damping, no static collider here; the
// body state and payload after 256 frames hash the same as the build before that change)
const THREE_BODY_XPBD: &str = "86851a66c8670a77b7c3d8862bbfb11a2130b794a7987a2c195ac5c003832dc0";
const THREE_BODY_TGS: &str = "43c48a9f25149e94911785cfe967fe362e44af5d96add71c5a1227f1d7ed737c";

/// golden (determinism): the world snapshot and the participant payload of
/// the three-body scene after 256 frames, for each backend. CI runs this on
/// every operating system and on wasm32, so the bits are the same
/// everywhere.
#[test]
fn three_body_scene_digest_is_pinned() {
    assert_eq!(three_body_digest(SolverBackend::Xpbd), THREE_BODY_XPBD);
    assert_eq!(three_body_digest(SolverBackend::Tgs), THREE_BODY_TGS);
}

/// oracle (parallel): `try_step_parallel` gives the same blob on 1, 2 and 8
/// threads, and, in this scene without contacts, the same bodies and
/// participant payload as `try_step` (the whole blob differs because the
/// parallel path stores its batch bookkeeping, a known world-side gap).
#[cfg(feature = "parallel")]
#[test]
fn parallel_steps_agree_with_each_other_and_with_the_serial_path() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let mut w = three_bodies(4, SolverBackend::Xpbd);
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
    let mut s = three_bodies(4, SolverBackend::Xpbd);
    for _ in 0..120 {
        s.try_step(Fix128::from_ratio(1, 64)).expect("step");
    }
    for k in 0..3 {
        assert_eq!(s.bodies[k].position, p1.bodies[k].position, "body {k}");
        assert_eq!(s.bodies[k].velocity, p1.bodies[k].velocity, "body {k}");
    }
    assert_eq!(s.participant_state(0), p1.participant_state(0));
}

/// oracle (degenerate inputs, each with its stated result): no bodies and
/// one body stage nothing, observe a zero potential and report 0 and 1
/// bodies; two bodies at exactly the same position with `ε = 0` are skipped
/// (no force, zero potential, as `DirectSum::accelerations`); with `ε > 0`
/// the same pair has zero force (zero separation) and the finite potential
/// `−G·m₁·m₂/ε`.
#[test]
fn degenerate_inputs_have_their_stated_results() {
    let h = Fix128::from_ratio(1, 64);
    let run = |bodies: &[RigidBody], eps: f64| {
        let mut p = NBodyParticipant::new(law(1.0, eps));
        let mut acc = ForceAccumulator::new(bodies.len());
        {
            let mut ctx = SubstepCtx::new(bodies, &mut acc, 0, 1, h).expect("ctx");
            p.substep(&mut ctx, h).expect("substep");
        }
        let forces: Vec<Vec3Fix> = (0..bodies.len())
            .map(|i| acc.force(i).expect("slot"))
            .collect();
        let (u, n) = seen(&p);
        (forces, u, n)
    };
    let int = Fix128::from_int;
    assert_eq!(run(&[], 0.0), (vec![], Fix128::ZERO, int(0)));
    let one = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    assert_eq!(run(&one, 0.0), (vec![Vec3Fix::ZERO], Fix128::ZERO, int(1)));
    let same = [
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0)),
    ];
    assert_eq!(
        run(&same, 0.0),
        (vec![Vec3Fix::ZERO; 2], Fix128::ZERO, int(2))
    );
    let (f, u, n) = run(&same, 0.5);
    assert_eq!(f, vec![Vec3Fix::ZERO; 2]);
    assert_eq!(u, -fx(4.0), "−G·m₁·m₂/ε = −1·1·2/0.5");
    assert_eq!(n, int(2));
}
