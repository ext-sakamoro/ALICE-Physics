//! A rollback through the end of a contact replays bit for bit.
//!
//! `serialize_state` / `deserialize_state` is the rollback state: save at step `N`, step on,
//! restore, step again, and the world must take the same steps as the run that never left `N`.
//! A contact carries memory from one substep to the next: the contact points found in the
//! previous substep (a point found again keeps the overlap the compliant contact holds at rest,
//! only a new point has its overlap pushed out by position) and the warm-start caches. When that
//! memory is not part of the state, the memory of the moment of the restore leaks into the
//! replay: while the contact at the restore is the contact at the save the leak is invisible,
//! and once the contact has ended in between, the resting contact counts as new on the replay.
//!
//! Scene (the three kinds of contact point the world remembers, both backends): a ball of
//! radius 1/4 m set on a 2 m square ledge (its top at y = 0) rolls toward the edge at 4 m/s
//! and falls off it, as a ball rolls off a ledge in a game: the ledge is a triangle mesh (static
//! collider points), an SDF box (SDF collider points) or a static box body (body pair points).
//! The ball is saved on a step where it rests on the ledge (centre within 1 mm of its radius
//! above the top) and restored on a later step where it has left it (centre past the edge by
//! more than its radius and below the top), then stepped 10 more. A bounce does not serve: a
//! bounce turns around inside one step, so at no step boundary is the ball in contact. Oracle: position, velocity,
//! rotation and angular velocity bit for bit equal to the straight run. Control: restoring right
//! after the save (no contact change in between) replays bit for bit, so the scene itself is
//! deterministic and the first oracle measures only what the state leaves behind.

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::trimesh::{TriMesh, Triangle};

const RADIUS: f64 = 0.25;
const REPLAY: usize = 10;
const STEPS: usize = 240;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

#[derive(Clone, Copy, Debug)]
enum Ledge {
    Mesh,
    StaticBox,
    Sdf,
}

/// What rolls or slides off the ledge: a ball (one contact point) or a cube of the same half
/// size (a manifold of points that the TGS solve couples, so its warm start shows in the bits).
#[derive(Clone, Copy, Debug)]
enum Body {
    Ball,
    Cube,
}

const BODIES: [Body; 2] = [Body::Ball, Body::Cube];
const LEDGES: [Ledge; 3] = [Ledge::Mesh, Ledge::StaticBox, Ledge::Sdf];
const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];
/// Half the side of the ledge's top (m).
const HALF: f64 = 1.0;

/// The box `|x| <= HALF`, `-1 <= y <= 0`, `|z| <= HALF`: per axis the signed excess over the
/// half extent and the sign of the coordinate from the box's centre `(0, -1/2, 0)`.
fn box_excess(x: f32, y: f32, z: f32) -> ([f32; 3], [f32; 3]) {
    let c = [x, y + 0.5, z];
    let h = [HALF as f32, 0.5, HALF as f32];
    let q = [c[0].abs() - h[0], c[1].abs() - h[1], c[2].abs() - h[2]];
    (q, [c[0].signum(), c[1].signum(), c[2].signum()])
}

/// The exact distance to the box.
fn box_distance(x: f32, y: f32, z: f32) -> f32 {
    let (q, _) = box_excess(x, y, z);
    let outside = q.iter().map(|c| c.max(0.0) * c.max(0.0)).sum::<f32>().sqrt();
    outside + q[0].max(q[1]).max(q[2]).min(0.0)
}

/// The gradient of [`box_distance`] in closed form: outside, the direction of the positive
/// excesses; inside, the axis of the largest excess.
fn box_normal(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let (q, s) = box_excess(x, y, z);
    let g = if q.iter().any(|&c| c > 0.0) {
        let o = [q[0].max(0.0) * s[0], q[1].max(0.0) * s[1], q[2].max(0.0) * s[2]];
        let l = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        [o[0] / l, o[1] / l, o[2] / l]
    } else {
        let k = if q[0] >= q[1] && q[0] >= q[2] { 0 } else if q[1] >= q[2] { 1 } else { 2 };
        let mut g = [0.0; 3];
        g[k] = s[k];
        g
    };
    (g[0], g[1], g[2])
}

/// The world and the ball's index.
fn scene(ledge: Ledge, backend: SolverBackend, body: Body) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -10.0, 0.0),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    match ledge {
        Ledge::Mesh => {
            let (a, b, c, d) = (
                v3(-HALF, 0.0, -HALF),
                v3(HALF, 0.0, -HALF),
                v3(HALF, 0.0, HALF),
                v3(-HALF, 0.0, HALF),
            );
            w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_triangles(vec![
                Triangle::new(a, c, b),
                Triangle::new(a, d, c),
            ])));
        }
        Ledge::StaticBox => {
            let f = w.add_body(RigidBody::new_static(v3(0.0, -0.5, 0.0)));
            w.set_body_shape(
                f,
                &Shape::Box {
                    half_extents: v3(HALF, 0.5, HALF),
                },
            );
        }
        Ledge::Sdf => {
            let field = ClosureSdf::new(box_distance, box_normal);
            w.add_sdf_collider(SdfCollider::new_static(
                Box::new(field),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            ));
        }
    }
    let mut b = RigidBody::new_dynamic(v3(0.25, RADIUS, 0.0), Fix128::ONE);
    b.velocity = v3(4.0, 0.0, 0.0);
    let ball = match body {
        Body::Ball => w.add_body_with_radius(b, fx(RADIUS)),
        Body::Cube => {
            let i = w.add_body(b);
            w.set_body_shape(
                i,
                &Shape::Box {
                    half_extents: v3(RADIUS, RADIUS, RADIUS),
                },
            );
            i
        }
    };
    (w, ball)
}

/// The ball's centre `(x, y)` after each step of the straight run (index `s` is the state after
/// `s + 1` steps).
fn path(ledge: Ledge, backend: SolverBackend, body: Body) -> Vec<(f64, f64)> {
    let (mut w, ball) = scene(ledge, backend, body);
    (0..STEPS)
        .map(|_| {
            w.step(dt());
            let p = w.bodies[ball].position;
            (p.x.to_f64(), p.y.to_f64())
        })
        .collect()
}

type Motion = (Vec3Fix, Vec3Fix, QuatFix, Vec3Fix);

fn motion(w: &PhysicsWorld) -> Vec<Motion> {
    w.bodies
        .iter()
        .map(|b| (b.position, b.velocity, b.rotation, b.angular_velocity))
        .collect()
}

/// Steps `save` times, saves, steps to `restore`, restores, steps `REPLAY` more; and the straight
/// run of `save + REPLAY` steps.
fn replay(ledge: Ledge, backend: SolverBackend, body: Body, save: usize, restore: usize) -> (Vec<Motion>, Vec<Motion>) {
    let (mut straight, _) = scene(ledge, backend, body);
    for _ in 0..save + REPLAY {
        straight.step(dt());
    }
    let (mut w, _) = scene(ledge, backend, body);
    for _ in 0..save {
        w.step(dt());
    }
    let blob = w.serialize_state();
    for _ in save..restore {
        w.step(dt());
    }
    assert!(w.deserialize_state(&blob), "{ledge:?} {backend:?} {body:?}: the state was refused");
    for _ in 0..REPLAY {
        w.step(dt());
    }
    (motion(&straight), motion(&w))
}

/// A step resting on the ledge and a later step off it.
fn on_then_off(ledge: Ledge, backend: SolverBackend, body: Body) -> (usize, usize) {
    let c = path(ledge, backend, body);
    // the middle of the run of steps resting on the ledge, so the contact is an old one
    let resting: Vec<usize> = (0..c.len())
        .filter(|&s| c[s].0 < HALF && (c[s].1 - RADIUS).abs() <= 1e-3)
        .collect();
    assert!(resting.len() >= 3, "{ledge:?} {backend:?} {body:?}: rested on {} steps", resting.len());
    let on = resting[resting.len() / 2];
    let off = (on + 1..c.len())
        .find(|&s| c[s].0 > HALF + RADIUS && c[s].1 < 0.0)
        .unwrap_or_else(|| panic!("{ledge:?} {backend:?} {body:?}: never left the ledge"));
    // steps counted from the start: the state after `on + 1` steps rests on the ledge
    (on + 1, off + 1)
}

#[test]
fn a_rollback_through_the_end_of_a_contact_replays_bit_for_bit() {
    let mut split = Vec::new();
    for ledge in LEDGES {
        for backend in BACKENDS {
            for body in BODIES {
                let (save, restore) = on_then_off(ledge, backend, body);
                let (straight, replayed) = replay(ledge, backend, body, save, restore);
                if straight != replayed {
                    split.push(format!("{ledge:?} {backend:?} {body:?} (saved at step {save}, restored at step {restore})"));
                }
            }
        }
    }
    assert!(split.is_empty(), "the replay left the straight run: {split:?}");
}

#[test]
fn a_rollback_right_after_the_save_replays_bit_for_bit() {
    for ledge in LEDGES {
        for backend in BACKENDS {
            for body in BODIES {
                let (save, _) = on_then_off(ledge, backend, body);
                let (straight, replayed) = replay(ledge, backend, body, save, save);
                assert_eq!(straight, replayed, "{ledge:?} {backend:?} {body:?}: restored at once at step {save}");
            }
        }
    }
}



/// A state whose step memory does not fit this world is refused with the world unchanged:
/// one byte short, one byte more, a contact point past the colliders, and two contact points out
/// of order. The blob is the Mesh ledge's state after the ball came to touch it (one static
/// contact point); the step memory ends the blob, so its last 4 bytes are the TGS impulse count
/// (0 under XPBD) and the 20 bytes before them the one static contact point.
#[test]
fn a_state_with_step_memory_of_another_world_is_refused_unchanged() {
    let (ledge, backend, body) = (Ledge::Mesh, SolverBackend::Xpbd, Body::Ball);
    let (save, _) = on_then_off(ledge, backend, body);
    let (mut w, _) = scene(ledge, backend, body);
    for _ in 0..save {
        w.step(dt());
    }
    let blob = w.serialize_state();
    let len = blob.len();
    // the static set has one point and the SDF and body sets none: count 1 + 20 byte point,
    // then the empty counts of the other two sets and of the impulses
    let point = len - 4 - 4 - 4 - 20;
    assert_eq!(&blob[point - 4..point], &1u32.to_le_bytes(), "the static point count");
    assert_eq!(&blob[len - 12..], &[0u8; 12], "the empty SDF, body and impulse counts");
    let before = w.serialize_state();
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("one byte short", blob[..len - 1].to_vec()),
        ("one byte more", [blob.clone(), vec![0]].concat()),
    ];
    // the collider index (the second u64 of the point) past the one static collider
    let mut past = blob.clone();
    past[point + 8..point + 16].copy_from_slice(&1u64.to_le_bytes());
    cases.push(("a collider past the world's", past));
    // a second static point equal to the first: not strictly sorted
    let mut twice = blob[..point - 4].to_vec();
    twice.extend_from_slice(&2u32.to_le_bytes());
    twice.extend_from_slice(&blob[point..point + 20]);
    twice.extend_from_slice(&blob[point..]);
    cases.push(("a repeated point", twice));
    for (what, bad) in cases {
        assert!(!w.deserialize_state(&bad), "{what}: accepted");
        assert_eq!(w.serialize_state(), before, "{what}: the world changed");
    }
    assert!(w.deserialize_state(&blob), "the state itself was refused");
}
