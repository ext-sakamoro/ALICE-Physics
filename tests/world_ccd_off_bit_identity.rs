//! World-step continuous collision (`PhysicsWorld::set_continuous_collision`)
//! switched off: the step must be the step that existed before the setting.
//!
//! oracle: the `serialize_state` bytes of the same scenes run by the engine
//! before the setting was added (the `GOLDEN_*` hashes below were recorded on
//! that tree and on `main` of the time, with and without `--features
//! parallel`; all four scenes were equal on both). Every scene contains a
//! body fast enough to be swept when the setting is on, so a step that ran any
//! part of the sweep while it is off changes the hash.

#![cfg(feature = "std")]

use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};
use sha2::{Digest, Sha256};

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

fn hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
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

fn run(mut w: PhysicsWorld, steps: usize, parallel: bool) -> String {
    for _ in 0..steps {
        if parallel {
            #[cfg(feature = "parallel")]
            w.step_parallel(dt());
            #[cfg(not(feature = "parallel"))]
            unreachable!();
        } else {
            w.step(dt());
        }
    }
    hex(&w.serialize_state())
}

fn scenes() -> Vec<(&'static str, PhysicsWorld)> {
    vec![
        ("thin_wall_xpbd", thin_wall(SolverBackend::Xpbd)),
        ("thin_wall_tgs", thin_wall(SolverBackend::Tgs)),
        ("head_on", head_on()),
        ("pile", pile()),
    ]
}

fn check(golden: &[(&str, &str)], parallel: bool) {
    let mut actual = Vec::new();
    for (name, w) in scenes() {
        actual.push((name, run(w, 90, parallel)));
    }
    for (name, got) in &actual {
        println!("(\"{name}\", \"{got}\"),");
    }
    for ((name, got), (gname, want)) in actual.iter().zip(golden) {
        assert_eq!(name, gname);
        assert_eq!(got, want, "{name}: the off step changed");
    }
}

/// `step`, the same with and without `--features parallel`.
const GOLDEN_STEP: &[(&str, &str)] = &[
    (
        "thin_wall_xpbd",
        "55bf63428684de990a46cab906181db609f5127e35c20004b9c5f32438eb812a",
    ),
    (
        "thin_wall_tgs",
        "4cf2c5b4d8bada719209223f8110ae1e03b3726cf2ddf9304270e7ce473da421",
    ),
    (
        "head_on",
        "40ad6c54c6f1fb20b70227745cec259ebb39cd7030dbe679c44bbb63c17f4f62",
    ),
    (
        "pile",
        "a5fad8e02f869406312cdbc13a375ee3f98fede54caac7194066cfcf6a1bce73",
    ),
];

/// `step_parallel`, which runs the XPBD substep whatever the backend, so the
/// TGS scene gives the XPBD hash.
#[cfg(feature = "parallel")]
const GOLDEN_STEP_PARALLEL: &[(&str, &str)] = &[
    (
        "thin_wall_xpbd",
        "55bf63428684de990a46cab906181db609f5127e35c20004b9c5f32438eb812a",
    ),
    (
        "thin_wall_tgs",
        "55bf63428684de990a46cab906181db609f5127e35c20004b9c5f32438eb812a",
    ),
    (
        "head_on",
        "40ad6c54c6f1fb20b70227745cec259ebb39cd7030dbe679c44bbb63c17f4f62",
    ),
    (
        "pile",
        "a5fad8e02f869406312cdbc13a375ee3f98fede54caac7194066cfcf6a1bce73",
    ),
];

#[test]
fn off_step_matches_the_engine_before_the_setting() {
    check(GOLDEN_STEP, false);
    assert_eq!(GOLDEN_STEP.len(), 4, "golden table incomplete");
}

#[cfg(feature = "parallel")]
#[test]
fn off_step_parallel_matches_the_engine_before_the_setting() {
    check(GOLDEN_STEP_PARALLEL, true);
    assert_eq!(GOLDEN_STEP_PARALLEL.len(), 4, "golden table incomplete");
}
