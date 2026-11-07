//! A sleeping body wakes when a caller moves it through the world.
//!
//! oracle (analytic): after a wake a free body moves as `x = v t` with `v = J / m`, the
//! closed form of a body with no force on it, and a body moved into the air falls
//! `y0 - g t^2 / 2`. Also pinned: what does not wake (zero, the same position, a static body), the island waking together, the bits of an awake body, and
//! the sleep configuration deciding the frame a body falls asleep.

use alice_physics_core::sleeping::SleepConfig;
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::{
    BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, PlaneCollider, RigidBody, Vec3Fix,
};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world() -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    PhysicsWorld::new(config)
}

fn free_body(w: &mut PhysicsWorld, x: i64, mass: i64) -> usize {
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(x, 0, 0), Fix128::from_int(mass));
    b.inv_inertia = Vec3Fix::from_int(1, 1, 1);
    w.add_body(b)
}

/// Steps until body `i` sleeps; a body at rest with no force falls asleep at once the
/// default frame count has passed.
fn sleep(w: &mut PhysicsWorld, i: usize) {
    for _ in 0..200 {
        w.step(dt());
        if w.is_sleeping(i) {
            return;
        }
    }
    panic!("body {i} did not fall asleep");
}

fn x_of(w: &PhysicsWorld, i: usize) -> f64 {
    let (hi, lo) = (w.bodies[i].position.x.hi, w.bodies[i].position.x.lo);
    hi as f64 + lo as f64 / 2f64.powi(64)
}

/// oracle: a nonzero velocity wakes the body and it moves `v t` from where it slept.
#[test]
fn a_velocity_wakes_a_sleeping_body_and_it_moves_v_t() {
    let mut w = world();
    let b = free_body(&mut w, 0, 2);
    sleep(&mut w, b);
    assert!(w.set_body_velocity(b, Vec3Fix::from_int(3, 0, 0)));
    assert!(!w.is_sleeping(b), "woken by the call, before any step");
    for _ in 0..60 {
        w.step(dt());
    }
    assert!(!w.is_sleeping(b));
    assert!((x_of(&w, b) - 3.0).abs() < 1e-3, "x {}", x_of(&w, b));
}

/// oracle: an impulse wakes the body and gives it `J / m`.
#[test]
fn an_impulse_wakes_a_sleeping_body_and_it_moves_j_over_m() {
    let mut w = world();
    let b = free_body(&mut w, 0, 2);
    sleep(&mut w, b);
    assert!(w.apply_body_impulse(b, Vec3Fix::from_int(6, 0, 0)));
    assert!(!w.is_sleeping(b));
    for _ in 0..60 {
        w.step(dt());
    }
    assert!((x_of(&w, b) - 3.0).abs() < 1e-3, "x {}", x_of(&w, b));
}

/// oracle: an impulse at a point wakes the body with `J / m` and the turn `r x J`.
#[test]
fn an_impulse_at_a_point_wakes_and_turns_the_body() {
    let mut w = world();
    let b = free_body(&mut w, 0, 2);
    sleep(&mut w, b);
    assert!(w.apply_body_impulse_at(b, Vec3Fix::from_int(6, 0, 0), Vec3Fix::from_int(0, 1, 0)));
    assert!(!w.is_sleeping(b));
    assert_eq!(w.bodies[b].velocity, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(w.bodies[b].angular_velocity, Vec3Fix::from_int(0, 0, -6));
}

/// A zero velocity or impulse leaves a sleeping body asleep, so a caller that stops a
/// body every frame does not keep it awake.
#[test]
fn zero_does_not_wake() {
    let mut w = world();
    let b = free_body(&mut w, 0, 2);
    sleep(&mut w, b);
    assert!(w.set_body_velocity(b, Vec3Fix::ZERO));
    assert!(w.apply_body_impulse(b, Vec3Fix::ZERO));
    assert!(w.apply_body_impulse_at(b, Vec3Fix::ZERO, Vec3Fix::UNIT_Y));
    assert!(w.is_sleeping(b));
}

fn pair() -> (PhysicsWorld, usize, usize) {
    let mut w = world();
    let a = free_body(&mut w, 0, 1);
    let c = free_body(&mut w, 2, 1);
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        c,
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(-1, 0, 0),
    )));
    sleep(&mut w, a);
    sleep(&mut w, c);
    (w, a, c)
}

/// The bodies joined to the one that moved wake with it, through each of the three calls.
#[test]
fn the_island_wakes_together() {
    let (mut w, a, c) = pair();
    w.set_body_velocity(a, Vec3Fix::UNIT_Y);
    assert!(!w.is_sleeping(c), "set_body_velocity");
    let (mut w, a, c) = pair();
    w.apply_body_impulse(a, Vec3Fix::UNIT_Y);
    assert!(!w.is_sleeping(c), "apply_body_impulse");
    let (mut w, a, c) = pair();
    w.apply_body_impulse_at(a, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_X);
    assert!(!w.is_sleeping(c), "apply_body_impulse_at");
}

/// A static body does not wake anything: it does not move and has no island to start.
#[test]
fn a_static_body_wakes_nothing() {
    let mut w = world();
    let top = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 4, 0)));
    let a = free_body(&mut w, 0, 1);
    w.add_joint(Joint::Ball(BallJoint::new(
        top,
        a,
        Vec3Fix::from_int(0, -4, 0),
        Vec3Fix::ZERO,
    )));
    sleep(&mut w, a);
    w.set_body_velocity(top, Vec3Fix::UNIT_Y);
    assert!(w.is_sleeping(a), "set_body_velocity");
    w.apply_body_impulse(top, Vec3Fix::UNIT_Y);
    assert!(w.is_sleeping(a), "apply_body_impulse");
    w.apply_body_impulse_at(top, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_X);
    assert!(w.is_sleeping(a), "apply_body_impulse_at");
}

/// A ball of radius 1/2 asleep on the floor `y = 0`, under gravity.
fn ball_asleep_on_a_floor() -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let start = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO);
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(start, Fix128::ONE),
        Fix128::from_ratio(1, 2),
    );
    sleep(&mut w, b);
    (w, b)
}

fn y_of(w: &PhysicsWorld, i: usize) -> f64 {
    let y = w.bodies[i].position.y;
    y.hi as f64 + y.lo as f64 / 2f64.powi(64)
}

/// Measured: the field write leaves the body hanging in the air, asleep. This is the
/// defect the world's call removes, and the field stays as it was by design.
#[test]
fn a_position_written_to_the_field_leaves_the_body_hanging() {
    let (mut w, b) = ball_asleep_on_a_floor();
    w.bodies[b].position.y = Fix128::from_int(5);
    for _ in 0..30 {
        w.step(dt());
    }
    println!("field write: y {} asleep {}", y_of(&w, b), w.is_sleeping(b));
    assert!(w.is_sleeping(b));
    assert_eq!(w.bodies[b].position.y, Fix128::from_int(5));
}

/// oracle: a body moved through the world wakes and falls `y0 - g t^2 / 2`.
#[test]
fn a_moved_body_wakes_and_falls_in_free_fall() {
    let (mut w, b) = ball_asleep_on_a_floor();
    let g = -w.config.gravity.y.to_f64();
    let to = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(5), Fix128::ZERO);
    assert!(w.set_body_position(b, to));
    assert!(!w.is_sleeping(b));
    let frames = 30;
    for _ in 0..frames {
        w.step(dt());
    }
    let t = f64::from(frames) / 60.0;
    let expect = 5.0 - g * t * t / 2.0;
    println!("moved: y {} closed form {expect}", y_of(&w, b));
    assert!((y_of(&w, b) - expect).abs() < 0.05, "y {}", y_of(&w, b));
}

/// The position the body already has does not wake it, so a caller that writes the
/// same place every frame still lets it sleep.
#[test]
fn the_same_position_does_not_wake() {
    let (mut w, b) = ball_asleep_on_a_floor();
    let here = w.bodies[b].position;
    assert!(w.set_body_position(b, here));
    assert!(w.is_sleeping(b));
}

/// A moved body wakes the bodies joined to it, and a moved static body wakes nothing.
#[test]
fn a_move_wakes_the_island_and_a_static_move_wakes_nothing() {
    let (mut w, a, c) = pair();
    assert!(w.set_body_position(a, Vec3Fix::from_int(0, 1, 0)));
    assert!(!w.is_sleeping(c));
    let mut w = world();
    let top = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 4, 0)));
    let a = free_body(&mut w, 0, 1);
    w.add_joint(Joint::Ball(BallJoint::new(
        top,
        a,
        Vec3Fix::from_int(0, -4, 0),
        Vec3Fix::ZERO,
    )));
    sleep(&mut w, a);
    assert!(w.set_body_position(top, Vec3Fix::from_int(0, 5, 0)));
    assert!(w.is_sleeping(a));
}

/// An index past the last body answers `false` and changes nothing.
#[test]
fn an_index_past_the_last_body_is_refused() {
    let mut w = world();
    let b = free_body(&mut w, 0, 1);
    let before = w.state_digest();
    assert!(!w.set_body_position(b + 1, Vec3Fix::UNIT_X));
    assert!(!w.set_body_velocity(b + 1, Vec3Fix::UNIT_X));
    assert!(!w.apply_body_impulse(b + 1, Vec3Fix::UNIT_X));
    assert!(!w.apply_body_impulse_at(b + 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y));
    assert_eq!(w.state_digest(), before);
}

/// On an awake body the calls are the field writes they replace, to the bit.
#[test]
fn an_awake_body_gets_the_same_bits_as_the_field_write() {
    let (mut a, mut b) = (world(), world());
    let (i, j) = (free_body(&mut a, 0, 3), free_body(&mut b, 0, 3));
    let v = Vec3Fix::new(Fix128::from_ratio(1, 3), Fix128::ONE, Fix128::ZERO);
    a.set_body_position(i, v);
    b.bodies[j].position = v;
    a.set_body_velocity(i, v);
    b.bodies[j].velocity = v;
    a.apply_body_impulse(i, v);
    b.bodies[j].apply_impulse(v);
    a.apply_body_impulse_at(i, v, Vec3Fix::UNIT_Y);
    b.bodies[j].apply_impulse_at(v, Vec3Fix::UNIT_Y);
    for _ in 0..30 {
        a.step(dt());
        b.step(dt());
    }
    assert_eq!(a.state_digest(), b.state_digest());
}

/// A body slower than the thresholds sleeps after exactly the configured frames.
#[test]
fn the_sleep_config_decides_the_frame() {
    let frames_until_asleep = |frames: u32| {
        let mut w = world();
        let b = free_body(&mut w, 0, 1);
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: frames,
            ..SleepConfig::default()
        });
        (1..=400).find(|_| {
            w.step(dt());
            w.is_sleeping(b)
        })
    };
    let (ten, twenty) = (frames_until_asleep(10), frames_until_asleep(20));
    println!("asleep after {ten:?} and {twenty:?}");
    assert_eq!(ten.zip(twenty).map(|(a, b)| b - a), Some(10));
}
