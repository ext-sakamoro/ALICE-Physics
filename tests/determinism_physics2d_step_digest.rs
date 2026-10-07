//! Bit-level pin of `PhysicsWorld2D::step` on a scene that exercises every path of the
//! 2D substep: gravity, dynamic / static / kinematic bodies, circle / polygon / capsule /
//! edge contacts with restitution and friction, and all four `Joint2D` variants.
//!
//! `EXPECTED` was recorded from the step before `Tethers2D` existed (main at
//! `ee4c5a70`). `step` now runs the same substep with an empty tether set, so the digest
//! must not move. `step_with_tethers` with an empty `Tethers2D` must also match `step`
//! state for state, every frame.

use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    Joint2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v(x: Fix128, y: Fix128) -> Vec2Fix {
    Vec2Fix::new(x, y)
}

fn square(half: Fix128) -> Shape2D {
    let m = Fix128::ZERO - half;
    Shape2D::Polygon {
        vertices: vec![v(m, m), v(half, m), v(half, half), v(m, half)],
    }
}

fn scene() -> PhysicsWorld2D {
    let mut w = PhysicsWorld2D::new(PhysicsConfig2D::default());
    // ground edge + a static box
    w.add_body(RigidBody2D::new_static(
        Vec2Fix::ZERO,
        Shape2D::Edge {
            start: Vec2Fix::from_int(-20, 0),
            end: Vec2Fix::from_int(20, 0),
        },
    ));
    w.add_body(RigidBody2D::new_static(
        Vec2Fix::from_int(4, 1),
        square(Fix128::ONE),
    ));
    // a stack of circles with restitution / friction
    for i in 0..4_i64 {
        let mut b = RigidBody2D::new_dynamic(
            v(
                r(1, 10) * Fix128::from_int(i),
                r(6, 10) + Fix128::from_int(i),
            ),
            Fix128::ONE,
            Shape2D::Circle { radius: r(1, 2) },
        );
        b.restitution = r(3, 10);
        b.friction = r(1, 2);
        w.add_body(b);
    }
    // a falling box and a capsule onto the static box
    let mut boxy = RigidBody2D::new_dynamic(
        v(r(41, 10), Fix128::from_int(4)),
        Fix128::from_int(2),
        square(r(1, 2)),
    );
    boxy.angular_velocity = r(1, 1);
    boxy.friction = r(4, 10);
    let boxy = w.add_body(boxy);
    let cap = w.add_body(RigidBody2D::new_dynamic(
        v(r(-3, 1), Fix128::from_int(3)),
        Fix128::ONE,
        Shape2D::Capsule {
            radius: r(1, 4),
            half_length: r(1, 2),
        },
    ));
    // a pendulum chain: revolute + distance + weld
    let anchor = w.add_body(RigidBody2D::new_static(
        Vec2Fix::from_int(-8, 6),
        Shape2D::Circle { radius: r(1, 10) },
    ));
    let p1 = w.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(-7, 6),
        Fix128::ONE,
        Shape2D::Circle { radius: r(1, 5) },
    ));
    let p2 = w.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(-6, 6),
        Fix128::ONE,
        Shape2D::Circle { radius: r(1, 5) },
    ));
    let p3 = w.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(-5, 6),
        Fix128::ONE,
        square(r(1, 5)),
    ));
    w.add_joint(Joint2D::Revolute {
        body_a: anchor,
        body_b: p1,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::from_int(-1, 0),
        compliance: Fix128::ZERO,
    });
    w.add_joint(Joint2D::Distance {
        body_a: p1,
        body_b: p2,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::ZERO,
        target_distance: Fix128::ONE,
        compliance: r(1, 1000),
    });
    w.add_joint(Joint2D::Weld {
        body_a: p2,
        body_b: p3,
        local_anchor_a: Vec2Fix::from_int(1, 0),
        local_anchor_b: Vec2Fix::ZERO,
        reference_angle: Fix128::ZERO,
        compliance: r(1, 100),
    });
    // mouse joints: one on the capsule, one on the falling box
    w.add_joint(Joint2D::Mouse {
        body: cap,
        target: Vec2Fix::from_int(-2, 5),
        max_force: Fix128::from_int(50),
        stiffness: Fix128::from_int(30),
        damping: Fix128::from_int(4),
    });
    w.add_joint(Joint2D::Mouse {
        body: boxy,
        target: Vec2Fix::from_int(5, 5),
        max_force: Fix128::from_int(5),
        stiffness: Fix128::from_int(10),
        damping: Fix128::ONE,
    });
    // a kinematic paddle sweeping through the circle stack
    let mut paddle = RigidBody2D::new_kinematic(Vec2Fix::from_int(-2, 1), square(r(1, 4)));
    paddle.velocity = Vec2Fix::from_int(1, 0);
    paddle.angular_velocity = r(1, 2);
    w.add_body(paddle);
    w
}

fn mix(h: &mut u64, f: Fix128) {
    for word in [f.hi as u64, f.lo] {
        for byte in word.to_le_bytes() {
            *h ^= u64::from(byte);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn digest(w: &PhysicsWorld2D) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in &w.bodies {
        for f in [
            b.position.x,
            b.position.y,
            b.angle,
            b.velocity.x,
            b.velocity.y,
            b.angular_velocity,
            b.prev_position.x,
            b.prev_position.y,
            b.prev_angle,
        ] {
            mix(&mut h, f);
        }
    }
    h
}

const FRAMES: usize = 180;
/// Recorded from `PhysicsWorld2D::step` before the tether set was introduced.
const EXPECTED: u64 = 0x2656_33e6_ca23_5f9f;

#[test]
fn step_digest_is_unchanged() {
    let mut w = scene();
    let dt = r(1, 60);
    for _ in 0..FRAMES {
        w.step(dt);
    }
    let d = digest(&w);
    println!("digest {d:#018x}");
    assert_eq!(d, EXPECTED, "PhysicsWorld2D::step moved: {d:#018x}");
}

#[test]
fn step_with_an_empty_tether_set_is_step_frame_by_frame() {
    use alice_physics::physics2d::Tethers2D;
    let mut a = scene();
    let mut b = scene();
    let empty = Tethers2D::new();
    let dt = r(1, 60);
    for frame in 0..FRAMES {
        a.step(dt);
        b.step_with_tethers(dt, &empty);
        assert_eq!(digest(&a), digest(&b), "diverged at frame {frame}");
    }
}

/// Removed entries leave the set behaving as empty.
#[test]
fn step_with_a_set_whose_entries_were_removed_is_step() {
    use alice_physics::physics2d::{AngularTether2D, KinematicDrive2D, Tethers2D};
    let mut a = scene();
    let mut b = scene();
    let mut set = Tethers2D::new();
    let t = set.add_angular(AngularTether2D::new(
        3,
        Fix128::ONE,
        Fix128::from_int(5),
        Fix128::ONE,
    ));
    let d = set.add_drive(KinematicDrive2D::new(
        b.bodies.len() - 1,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::from_int(3),
    ));
    assert!(set.remove_angular(t).is_some());
    assert!(set.remove_drive(d).is_some());
    assert!(set.remove_angular(t).is_none());
    assert_eq!((set.angular_count(), set.drive_count()), (0, 0));
    let dt = r(1, 60);
    for _ in 0..FRAMES {
        a.step(dt);
        b.step_with_tethers(dt, &set);
    }
    assert_eq!(digest(&a), digest(&b));
}
