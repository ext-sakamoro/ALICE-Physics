//! A joint that breaks transmits nothing in the substep it breaks in, and a joint that does
//! not break is untouched by having a break force.
//!
//! oracle (twin worlds, bit for bit):
//! - a joint that breaks in the first substep of a frame leaves that frame exactly as a world
//!   that never had the joint: same bodies, to the bit. Before the fix the breaking joint's
//!   whole correction was applied and the joint then removed; a slider started 6.08 m off its
//!   axis snapped its body across the gap in one substep (h = 1/480 s) and left it flying at
//!   √37 × 480 = 2919.7 m/s.
//! - the same for a joint that held for a while and is overloaded by a kick: the frame after
//!   the kick equals the twin from which the joint was removed at the kick.
//! - a joint whose break force is above every force it carries is bit-identical, every frame,
//!   to the same joint without a break force (nothing changes for joints that hold).
//!
//! Bound: a holding joint transmits its solve's impulse, a breaking one none, so against an
//! ideal brittle joint (which would transmit `break_force · h` while failing) the error is at
//! most `break_force · h` of impulse.
use alice_physics_core::joint::{BallJoint, Joint, SliderJoint};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn bodies_equal(a: &PhysicsWorld, b: &PhysicsWorld) -> bool {
    a.bodies.len() == b.bodies.len()
        && a.bodies.iter().zip(&b.bodies).all(|(x, y)| {
            x.position == y.position
                && x.velocity == y.velocity
                && x.rotation == y.rotation
                && x.angular_velocity == y.angular_velocity
        })
}

/// anchor (static) at (3, 6, 0) and one body; the joint if `joint` is given
fn two_bodies(at: Vec3Fix, joint: Option<Joint>) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_body(RigidBody::new_static(v(3, 6, 0)));
    w.add_body_with_radius(
        RigidBody::new_dynamic(at, Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    if let Some(j) = joint {
        w.add_joint(j);
    }
    w
}

#[test]
fn a_joint_breaking_in_its_first_substep_leaves_the_frame_as_if_it_was_never_there() {
    let at = v(6, 5, 6); // √37 = 6.08 m off the slider's axis through the anchor
    let mut slider = SliderJoint::new(0, 1, v(1, 0, 0), Vec3Fix::ZERO, Vec3Fix::ZERO);
    slider.limit_min = Some(Fix128::from_int(-1));
    slider.break_force = Some(Fix128::from_int(1000));
    let mut with = two_bodies(at, Some(Joint::Slider(slider)));
    let mut without = two_bodies(at, None);
    let mut compared = 0;
    for frame in 0..10 {
        with.step(dt());
        without.step(dt());
        assert!(
            with.joints.is_empty(),
            "frame {frame}: the joint did not break"
        );
        assert!(
            bodies_equal(&with, &without),
            "frame {frame}: the broken joint moved the body (|v| {} m/s, free flight {} m/s)",
            with.bodies[1].velocity.length().to_f64(),
            without.bodies[1].velocity.length().to_f64()
        );
        compared += 1;
    }
    assert_eq!(compared, 10);
}

/// a body hung 1 m below the anchor by a ball joint (static load m·g = 10 N), break force
/// 100 N: it holds; then the body is kicked sideways at 50 m/s, which needs far more than
/// 100 N in the next substep, and the joint breaks
#[test]
fn a_joint_overloaded_by_a_kick_transmits_nothing_in_the_substep_it_breaks() {
    let at = v(3, 5, 0);
    let mut ball = BallJoint::new(0, 1, Vec3Fix::ZERO, v(0, 1, 0));
    ball.break_force = Some(Fix128::from_int(100));
    let mut with = two_bodies(at, Some(Joint::Ball(ball)));
    for _ in 0..30 {
        with.step(dt());
    }
    assert_eq!(
        with.joints.len(),
        1,
        "the joint broke under its static load"
    );
    let kick = Vec3Fix::new(Fix128::from_int(50), Fix128::ZERO, Fix128::ZERO);
    with.bodies[1].velocity = with.bodies[1].velocity + kick;
    // the twin: the same state with the joint removed at the kick
    let mut without = two_bodies(at, None);
    without.bodies[1] = with.bodies[1];
    let mut compared = 0;
    for frame in 0..5 {
        with.step(dt());
        without.step(dt());
        assert!(
            with.joints.is_empty(),
            "frame {frame}: the kick did not break the joint"
        );
        assert!(
            bodies_equal(&with, &without),
            "frame {frame} after the kick differs from the twin"
        );
        compared += 1;
    }
    assert_eq!(compared, 5);
}

/// A pendulum from horizontal: the largest reaction force its joint carries in 300 frames is
/// between 40.69 and 40.70 N (bisected on the break force). With a break force of 41 N, just
/// above it, the joint never breaks and every frame is bit-identical to the same joint without
/// a break force; with 40 N it breaks (the threshold is close, so the case is not vacuous).
#[test]
fn a_joint_below_its_break_force_is_bit_identical_to_one_without() {
    let mk = |break_force: Option<Fix128>| {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = w.add_body(RigidBody::new_dynamic(v(1, 0, 0), Fix128::ONE));
        let mut j = BallJoint::new(anchor, b, Vec3Fix::ZERO, v(-1, 0, 0));
        j.break_force = break_force;
        w.add_joint(Joint::Ball(j));
        w
    };
    let mut tight = mk(Some(Fix128::from_int(40)));
    for _ in 0..300 {
        tight.step(dt());
    }
    assert!(
        tight.joints.is_empty(),
        "40 N held: the case is not near its threshold"
    );
    let mut breakable = mk(Some(Fix128::from_int(41)));
    let mut plain = mk(None);
    let mut compared = 0;
    for frame in 0..300 {
        breakable.step(dt());
        plain.step(dt());
        assert_eq!(
            breakable.joints.len(),
            1,
            "frame {frame}: broke below its break force"
        );
        assert!(bodies_equal(&breakable, &plain), "frame {frame} differs");
        compared += 1;
    }
    assert_eq!(compared, 300);
}
