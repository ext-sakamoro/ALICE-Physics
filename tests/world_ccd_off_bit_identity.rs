//! World-step continuous collision (`PhysicsWorld::set_continuous_collision`)
//! switched off: the step must be the step without the setting.
//!
//! oracle: the same scene run in the same build without any sweep. Each
//! scene is built several times and stepped 90 frames with `step` (and, with
//! `--features parallel`, also with `step_parallel`); the `serialize_state`
//! bytes of the copies must be equal bit for bit:
//!
//! - (a) the default world, which never calls the setting;
//! - (b) the setting written as off, both as `WorldCcdConfig::new()` and as
//!   `WorldCcdConfig::on().with_enabled(false)`, so that a default that is on
//!   (in `PhysicsWorld::new` or in `WorldCcdConfig::new`) separates (a) or
//!   one of the two from the others;
//! - (d) the setting on with a threshold no body reaches: the sweep chooses
//!   its bodies, finds none and moves nothing. This copy is the reference
//!   that does not go through the off branch, so a step that sweeps while the
//!   setting is off separates (a) and (b) from it in the fast scenes (each
//!   fast scene has a body the default threshold sweeps);
//! - (c) in the slow scenes only, the setting on with its default threshold:
//!   no body moves more than its radius in a substep, so nothing is swept and
//!   the step is the step without the setting (the test checks the speeds of
//!   the slow scenes against that bound).
//!
//! Two controls show that the comparison can fail: the fast scene with the
//! setting on differs from off, and a slow scene with the threshold `0`
//! (every moving body swept) differs from off.
//!
//! When the setting was added, the off step was also compared with the engine
//! without the setting at the same base, and 30 of 30 state hashes were equal
//! (5 scenes, XPBD and TGS, `step` in the default build, `step` and
//! `step_parallel` in the parallel build). That comparison is not kept here
//! as fixed hashes: a change of the engine itself moves them, and such
//! changes are the business of the engine's golden tests.

#![cfg(feature = "std")]

use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix, WorldCcdConfig,
};

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

/// A sphere at 300 m/s toward a static box 2 cm thick, plus a slow sphere
/// resting on a ground plane.
fn thin_wall(backend: SolverBackend) -> PhysicsWorld {
    let config = PhysicsConfig {
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    let wall = w.add_body(RigidBody::new_static(v(5, 1, 0)));
    w.set_body_shape(
        wall,
        &Shape::Box {
            half_extents: Vec3Fix::new(r(1, 100), Fix128::from_int(2), Fix128::from_int(2)),
        },
    );
    let mut fast = RigidBody::new_dynamic(v(0, 1, 0), Fix128::ONE);
    fast.velocity = v(300, 0, 0);
    w.add_body_with_radius(fast, r(1, 4));
    w.add_body_with_radius(RigidBody::new_dynamic(v(-3, 1, 0), Fix128::ONE), r(1, 2));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v(0, 1, 0),
        Fix128::ZERO,
    )));
    w
}

/// Two spheres closing at 400 m/s, a third crossing their line, and a thin
/// static triangle mesh in the path of a fourth.
fn head_on() -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    let mut a = RigidBody::new_dynamic(v(-6, 0, 0), Fix128::ONE);
    a.velocity = v(200, 0, 0);
    w.add_body_with_radius(a, r(1, 2));
    let mut b = RigidBody::new_dynamic(v(6, 0, 0), Fix128::from_int(2));
    b.velocity = v(-200, 0, 0);
    w.add_body_with_radius(b, r(1, 2));
    let mut c = RigidBody::new_dynamic(v(0, -6, 0), Fix128::ONE);
    c.velocity = v(0, 150, 1);
    w.add_body_with_radius(c, r(1, 3));
    let mut d = RigidBody::new_dynamic(v(0, 0, -8), Fix128::ONE);
    d.velocity = v(0, 0, 250);
    w.add_body_with_radius(d, r(1, 4));
    let mesh = TriMesh::from_indexed(&[v(-5, -5, 5), v(5, -5, 5), v(0, 5, 5)], &[0, 1, 2]);
    w.add_static_collider(StaticCollider::TriMesh(mesh));
    w
}

/// A small pile: spheres and a box dropped on a plane, one of them thrown down
/// fast.
fn pile() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v(0, 1, 0),
        Fix128::ZERO,
    )));
    for k in 0..5 {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(r(k, 3), Fix128::from_int(1 + k), r(k, 7)),
            Fix128::ONE,
        );
        w.add_body_with_radius(body, r(1, 2));
    }
    let boxed = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: Vec3Fix::new(r(1, 2), r(1, 4), r(1, 2)),
            },
            Fix128::ONE,
            v(0, 8, 0),
        )
        .unwrap();
    w.bodies[boxed].velocity = v(0, -120, 0);
    w
}

/// Spheres and a box dropped onto a plane from rest, the spheres starting on
/// or just above it. Nothing falls far enough to move its radius in one
/// substep.
fn slow_pile(backend: SolverBackend) -> PhysicsWorld {
    let config = PhysicsConfig {
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v(0, 1, 0),
        Fix128::ZERO,
    )));
    w.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, r(1, 2), Fix128::ZERO),
            Fix128::ONE,
        ),
        r(1, 2),
    );
    for k in 1..5 {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(r(k, 5), Fix128::from_int(k) + r(1, 2), r(k, 9)),
            Fix128::ONE,
        );
        w.add_body_with_radius(body, r(1, 2));
    }
    w.add_shaped_body(
        &Shape::Box {
            half_extents: Vec3Fix::new(r(1, 2), r(1, 4), r(1, 2)),
        },
        Fix128::ONE,
        v(0, 8, 0),
    )
    .unwrap();
    w
}

/// On a plane: a sphere sliding at 4 m/s into a static box, and two spheres
/// closing at 2 m/s each. Every sphere starts `1/8` above the plane.
fn slow_slide(backend: SolverBackend) -> PhysicsWorld {
    let config = PhysicsConfig {
        solver_backend: backend,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v(0, 1, 0),
        Fix128::ZERO,
    )));
    let wall = w.add_body(RigidBody::new_static(v(2, 1, 0)));
    w.set_body_shape(
        wall,
        &Shape::Box {
            half_extents: Vec3Fix::new(r(1, 10), Fix128::ONE, Fix128::from_int(2)),
        },
    );
    let mut slider = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, r(5, 8), Fix128::ZERO),
        Fix128::ONE,
    );
    slider.velocity = v(4, 0, 0);
    w.add_body_with_radius(slider, r(1, 2));
    let mut a = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(-2), r(5, 8), Fix128::from_int(3)),
        Fix128::ONE,
    );
    a.velocity = v(2, 0, 0);
    w.add_body_with_radius(a, r(1, 2));
    let mut b = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(2), r(5, 8), Fix128::from_int(3)),
        Fix128::from_int(2),
    );
    b.velocity = v(-2, 0, 0);
    w.add_body_with_radius(b, r(1, 2));
    w
}

type Build = fn() -> PhysicsWorld;

fn thin_wall_xpbd() -> PhysicsWorld {
    thin_wall(SolverBackend::Xpbd)
}

fn thin_wall_tgs() -> PhysicsWorld {
    thin_wall(SolverBackend::Tgs)
}

fn slow_pile_xpbd() -> PhysicsWorld {
    slow_pile(SolverBackend::Xpbd)
}

fn slow_pile_tgs() -> PhysicsWorld {
    slow_pile(SolverBackend::Tgs)
}

fn slow_slide_xpbd() -> PhysicsWorld {
    slow_slide(SolverBackend::Xpbd)
}

fn slow_slide_tgs() -> PhysicsWorld {
    slow_slide(SolverBackend::Tgs)
}

/// Scenes with a body the default threshold sweeps.
const FAST: &[(&str, Build)] = &[
    ("thin_wall_xpbd", thin_wall_xpbd),
    ("thin_wall_tgs", thin_wall_tgs),
    ("head_on", head_on),
    ("pile", pile),
];

/// Scenes in which no body moves its radius (`1/2`, the smallest in them) in
/// one substep.
const SLOW: &[(&str, Build)] = &[
    ("slow_pile_xpbd", slow_pile_xpbd),
    ("slow_pile_tgs", slow_pile_tgs),
    ("slow_slide_xpbd", slow_slide_xpbd),
    ("slow_slide_tgs", slow_slide_tgs),
];

fn slow_min_radius() -> Fix128 {
    r(1, 2)
}

const FRAMES: usize = 90;

#[derive(Clone, Copy, Debug)]
enum Stepper {
    Step,
    #[cfg(feature = "parallel")]
    StepParallel,
}

fn steppers() -> Vec<Stepper> {
    #[cfg(feature = "parallel")]
    return vec![Stepper::Step, Stepper::StepParallel];
    #[cfg(not(feature = "parallel"))]
    vec![Stepper::Step]
}

/// The setting is on, with a threshold (in radii) no body of these scenes
/// reaches: the fastest moves `300 / 480` m in a substep with a radius of
/// `1/4`, a ratio of `2.5`.
fn on_unreached() -> WorldCcdConfig {
    WorldCcdConfig::on().with_motion_threshold(Fix128::from_int(1 << 20))
}

/// The copies that must step alike in every scene: (a), (b) twice, (d).
fn off_like() -> Vec<(&'static str, Option<WorldCcdConfig>)> {
    vec![
        ("on, threshold not reached", Some(on_unreached())),
        ("default (setting never called)", None),
        ("WorldCcdConfig::new()", Some(WorldCcdConfig::new())),
        (
            "WorldCcdConfig::on().with_enabled(false)",
            Some(WorldCcdConfig::on().with_enabled(false)),
        ),
    ]
}

/// Runs a fresh copy of the scene and returns its state and the largest
/// squared body speed seen at the end of any frame.
fn run(build: Build, ccd: Option<WorldCcdConfig>, stepper: Stepper) -> (Vec<u8>, Fix128) {
    let mut w = build();
    assert!(
        !w.continuous_collision().is_enabled(),
        "a new world must start with the setting off"
    );
    if let Some(config) = ccd {
        w.set_continuous_collision(config);
    }
    let mut max_speed_sq = Fix128::ZERO;
    for _ in 0..FRAMES {
        match stepper {
            Stepper::Step => w.step(dt()),
            #[cfg(feature = "parallel")]
            Stepper::StepParallel => w.step_parallel(dt()),
        }
        for b in &w.bodies {
            let s = b.velocity.length_squared();
            if s > max_speed_sq {
                max_speed_sq = s;
            }
        }
    }
    (w.serialize_state(), max_speed_sq)
}

fn assert_all_equal(name: &str, build: Build, copies: &[(&str, Option<WorldCcdConfig>)]) {
    for stepper in steppers() {
        let (reference, _) = run(build, copies[0].1, stepper);
        for (label, ccd) in &copies[1..] {
            let (got, _) = run(build, *ccd, stepper);
            assert!(
                got == reference,
                "{name} ({stepper:?}): `{label}` differs from `{}`",
                copies[0].0
            );
        }
    }
}

/// (a), (b) and (d) in the scenes with a fast body.
#[test]
fn off_steps_like_the_step_without_a_sweep_in_fast_scenes() {
    assert_eq!(FAST.len(), 4);
    for (name, build) in FAST {
        assert_all_equal(name, *build, &off_like());
    }
}

/// (a), (b), (d) and (c) in the slow scenes.
#[test]
fn on_without_a_fast_body_steps_like_off() {
    assert_eq!(SLOW.len(), 4);
    // |v| h <= r / 2 with h = 1 / (60 substeps), as |v|^2 <= (60 substeps r / 2)^2.
    let substeps = PhysicsConfig::default().substeps as i64;
    let bound = Fix128::from_int(60 * substeps) * slow_min_radius() * r(1, 2);
    let bound_sq = bound * bound;
    for (name, build) in SLOW {
        let mut copies = off_like();
        copies.push(("on, default threshold", Some(WorldCcdConfig::on())));
        assert_all_equal(name, *build, &copies);
        for stepper in steppers() {
            let (_, max_speed_sq) = run(*build, None, stepper);
            assert!(
                max_speed_sq <= bound_sq,
                "{name} ({stepper:?}): a squared speed of {} reached the bound {} \
                 (half the radius per substep): the scene is not slow",
                max_speed_sq.to_f64(),
                bound_sq.to_f64()
            );
        }
    }
}

/// Control: with the default threshold the fast scene is swept and the step
/// differs from off, so the comparison above is not blind to a sweep.
#[test]
fn control_fast_scene_swept_differs_from_off() {
    for stepper in steppers() {
        let (off, _) = run(thin_wall_xpbd, None, stepper);
        let (on, _) = run(thin_wall_xpbd, Some(WorldCcdConfig::on()), stepper);
        assert!(
            off != on,
            "thin_wall_xpbd ({stepper:?}): the sweep changed nothing"
        );
    }
}

/// Control: with the threshold `0` every moving body of a slow XPBD scene is
/// swept and the step differs from off, so the slow scenes are not blind to a
/// sweep either; they agree with off under the default threshold only because
/// nothing reaches it. (The TGS backend does not sweep, so its slow scenes
/// have no such control.)
#[test]
fn control_slow_scenes_swept_at_threshold_zero_differ_from_off() {
    let xpbd: [(&str, Build); 2] = [
        ("slow_pile_xpbd", slow_pile_xpbd),
        ("slow_slide_xpbd", slow_slide_xpbd),
    ];
    for (name, build) in xpbd {
        for stepper in steppers() {
            let (off, _) = run(build, None, stepper);
            let (on, _) = run(
                build,
                Some(WorldCcdConfig::on().with_motion_threshold(Fix128::ZERO)),
                stepper,
            );
            assert!(off != on, "{name} ({stepper:?}): the sweep changed nothing");
        }
    }
}
