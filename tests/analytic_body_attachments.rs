//! `PhysicsWorld::remove_body_with_attachments` / `add_body_with_attachments`:
//! a body moved from one world into another with its per-body settings
//! (material, collision radius, collider, collision filter) steps in the
//! destination bit for bit like the same body built there directly.
//!
//! oracle (twin): world B receives the body through the two calls; world C
//! builds the same body in place with `add_shaped_body`, `set_body_material`
//! and `set_body_filter`. Every setting reaches the trajectory in this scene:
//! the upper floor collides only with layer 1, so the body (layer 2) falls
//! through it onto the lower floor, where its box collider (not the bounding
//! sphere) and its rubber material decide how it bounces. A body moved without
//! its settings (`remove_body` + `add_body`) lands elsewhere, which the last
//! test asserts, so the twin comparison has teeth.

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::material::MaterialId;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn mover_shape() -> Shape {
    Shape::Box {
        half_extents: Vec3Fix::new(
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(1, 2),
        ),
    }
}

/// The id `register_rubber` returns as the first registration of a world.
const RUBBER: MaterialId = 1;

const MOVER_FILTER: CollisionFilter = CollisionFilter {
    layer: 2,
    mask: u32::MAX,
    group: 0,
};

/// A world with the rubber material registered once, first: the carried
/// material id names an entry of the destination's table, so every world
/// registers the same materials in the same order.
fn new_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let rubber = w.material_table.register_rubber();
    assert_eq!(rubber, RUBBER);
    w
}

/// Two static box floors: the upper one (y = 0) collides only with layer 1,
/// the lower one (y = -6) with everything.
fn floors() -> PhysicsWorld {
    let mut w = new_world();
    let slab = Shape::Box {
        half_extents: Vec3Fix::new(Fix128::from_int(8), Fix128::ONE, Fix128::from_int(8)),
    };
    let upper = w.add_body(RigidBody::new_static(v3(0, 0, 0)));
    assert!(w.set_body_shape(upper, &slab));
    w.set_body_filter(upper, CollisionFilter::new(1, 1));
    let lower = w.add_body(RigidBody::new_static(v3(0, -6, 0)));
    assert!(w.set_body_shape(lower, &slab));
    w
}

/// Gives the body at `idx` its material and filter (its shape is set by
/// `add_shaped_body`).
fn configure(w: &mut PhysicsWorld, idx: usize) {
    w.set_body_material(idx, RUBBER);
    w.set_body_filter(idx, MOVER_FILTER);
}

/// The source world: a decoy body first, so the moved body is not at index 0.
fn source() -> (PhysicsWorld, usize) {
    let mut a = new_world();
    a.add_body(RigidBody::new_static(v3(50, 50, 50)));
    let idx = a
        .add_shaped_body(&mover_shape(), Fix128::ONE, v3(0, 3, 0))
        .expect("valid shape");
    configure(&mut a, idx);
    (a, idx)
}

/// Sets some of the moved body's settings by hand (the variants below)
type Setter = fn(&mut PhysicsWorld, usize);

fn trajectory(w: &mut PhysicsWorld, idx: usize, steps: usize) -> Vec<(Vec3Fix, Vec3Fix)> {
    (0..steps)
        .map(|_| {
            w.step(dt());
            let b = w.get_body(idx).expect("body");
            (b.position, b.velocity)
        })
        .collect()
}

fn twin() -> (PhysicsWorld, usize) {
    let mut c = floors();
    let idx = c
        .add_shaped_body(&mover_shape(), Fix128::ONE, v3(0, 3, 0))
        .expect("valid shape");
    configure(&mut c, idx);
    (c, idx)
}

#[test]
fn a_moved_body_steps_like_the_same_body_built_in_place() {
    let (mut a, idx) = source();
    let (body, attachments) = a.remove_body_with_attachments(idx).expect("in range");
    assert_eq!(a.bodies.len(), 1, "the source world lost the body");
    let mut b = floors();
    let moved = b.add_body_with_attachments(body, attachments);
    assert_eq!(b.body_filter(moved), MOVER_FILTER);

    let (mut c, built) = twin();
    assert_eq!(moved, built);
    let got = trajectory(&mut b, moved, 300);
    let want = trajectory(&mut c, built, 300);
    assert_eq!(got, want);

    // the scene reaches every setting: the body passed the upper floor
    // (filter) and bounced on the lower one (material, collider)
    let lowest = got.iter().map(|(p, _)| p.y).min().expect("steps");
    assert!(
        lowest < Fix128::from_int(-3),
        "did not fall through the upper floor"
    );
    assert!(
        got.iter().any(|(_, v)| v.y > Fix128::ZERO),
        "never moved up again: no bounce"
    );
}

#[test]
fn each_setting_changes_the_trajectory_when_it_is_not_carried() {
    let (c_traj, built) = {
        let (mut c, built) = twin();
        (trajectory(&mut c, built, 300), built)
    };
    // the body alone, through remove_body + add_body: every setting dropped
    let (mut a, idx) = source();
    let body = a.remove_body(idx).expect("in range");
    let mut b = floors();
    let plain = b.add_body(body);
    assert_eq!(plain, built);
    assert_ne!(trajectory(&mut b, plain, 300), c_traj);

    // one setting missing at a time, set by hand on the plain body
    let variants: [(&str, Setter); 3] = [
        ("no filter", |w, i| {
            w.set_body_material(i, RUBBER);
            assert!(w.set_body_shape(i, &mover_shape()));
        }),
        ("no material", |w, i| {
            w.set_body_filter(i, MOVER_FILTER);
            assert!(w.set_body_shape(i, &mover_shape()));
        }),
        ("no collider (bounding sphere)", |w, i| {
            configure(w, i);
            w.set_body_collision_radius(i, mover_shape().bounding_radius());
        }),
    ];
    for (name, set) in variants {
        let (mut a, idx) = source();
        let body = a.remove_body(idx).expect("in range");
        let mut b = floors();
        let i = b.add_body(body);
        set(&mut b, i);
        assert_ne!(trajectory(&mut b, i, 300), c_traj, "{name}");
    }
}

#[test]
fn an_index_out_of_range_moves_nothing() {
    let (mut a, idx) = source();
    let before = a.snapshot_world();
    assert!(a.remove_body_with_attachments(idx + 1).is_none());
    assert!(a.remove_body_with_attachments(usize::MAX).is_none());
    assert_eq!(a.snapshot_world(), before);
    assert_eq!(a.bodies.len(), 2);
}
