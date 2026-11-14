//! A joint error that the dynamics did not make is removed by position, without velocity.
//!
//! A joint created between bodies that do not satisfy it, or a body of a joint moved by a pose
//! write, starts a substep with an error the previous substep's dynamics did not leave. Carrying
//! that correction into the velocity (`v = Δx / h`) turns a placement error into kinetic energy:
//! a 1 m error corrected in one substep of `h = 1/480 s` becomes 480 m/s. The world removes such
//! an error by position at the head of the substep, so the velocity is untouched; an error the
//! dynamics made in the previous substep is solved as before.
//!
//! Oracles (closed form, the scene's own numbers, gravity 0). The pairs drift at 1 m/s along `z`
//! (a body at rest falls asleep, and a sleeping body derives no velocity from a move, which would
//! hide the velocity a projection adds); "the velocity stays" is then the velocity of a lone body
//! with the same drift, mass and steps, bit for bit:
//!
//! * a ball joint with both anchors at the centres, error `e` along `x`: body A moves by
//!   `w_a / (w_a + w_b) · e`, body B by `−w_b / (w_a + w_b) · e` (the XPBD projection of one
//!   rigid row, Macklin, Müller, Chentanez, MIG 2016), and both velocities stay as they were. With
//!   equal masses the shares are 1/2 and the poses are exact; with masses 1 kg and 3 kg the shares
//!   are 3/4 and 1/4, each a `Fix128` quotient, so each coordinate is within a few rounding steps
//!   of `2⁻⁶⁴` of the closed form (bound `2⁻⁵⁶` m, a factor 2⁸ over the handful of roundings);
//! * the same joint, body B then moved 2 m by a pose write: B alone returns onto A (A is where the
//!   dynamics left it), and both velocities stay as they were;
//! * a fixed joint and a slider joint chained over three bodies, created violated (the scene of
//!   the browser package's comparison that reached 300 m/s): the kinetic energy after 60 steps is
//!   not above the energy at the start, which is 0;
//! * a spring created stretched holds that stretch as elastic energy, not as an error: it is
//!   left to the dynamics, which pull the pair together (each body gains speed toward the other),
//!   and so is a compliant ball joint;
//! * a slider with a 1000 N break force created 6.08 m off its axis: the placement error is not
//!   a load, so the joint holds, the body sits on the axis, and no energy appears;
//! * a ball joint the dynamics stretches (a body pulled by its own velocity) still carries its
//!   correction into the velocity: the pair moves together after one step.

use alice_physics_core::joint::{BallJoint, FixedJoint, Joint, SliderJoint, SpringJoint};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    })
}

fn body(w: &mut PhysicsWorld, x: i64, y: i64, z: i64, mass: i64) -> usize {
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(x, y, z),
        Fix128::from_int(mass),
    ))
}

fn centre_ball(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO))
}

/// The drift both bodies of a pair start with (0, 0, 1) m/s: a body at rest can fall asleep, and a
/// sleeping body derives no velocity from a move, which would hide the velocity a projection adds.
fn drift() -> Vec3Fix {
    Vec3Fix::from_int(0, 0, 1)
}

/// Velocity and z after `steps` steps of a lone body of `mass` with the drift: what each body of a
/// pair must show when the pair's correction adds no velocity (the same damping and rounding).
fn lone(mass: i64, steps: usize) -> (Vec3Fix, Fix128) {
    let mut w = world();
    let i = body(&mut w, 0, 0, 0, mass);
    w.bodies[i].velocity = drift();
    for _ in 0..steps {
        w.step(dt());
    }
    (w.bodies[i].velocity, w.bodies[i].position.z)
}

fn drifting_pair(w: &mut PhysicsWorld, xb: i64, mass_b: i64) -> (usize, usize) {
    let a = body(w, 0, 0, 0, 1);
    let b = body(w, xb, 0, 0, mass_b);
    w.bodies[a].velocity = drift();
    w.bodies[b].velocity = drift();
    w.add_joint(centre_ball(a, b));
    (a, b)
}

fn kinetic_energy(w: &PhysicsWorld) -> f64 {
    w.bodies
        .iter()
        .filter(|b| b.is_dynamic())
        .map(|b| {
            let v = b.velocity;
            let m = 1.0 / b.inv_mass.to_f64();
            0.5 * m
                * (v.x.to_f64() * v.x.to_f64()
                    + v.y.to_f64() * v.y.to_f64()
                    + v.z.to_f64() * v.z.to_f64())
        })
        .sum()
}

#[test]
fn a_joint_created_violated_closes_by_position_with_equal_shares() {
    let mut w = world();
    let (a, b) = drifting_pair(&mut w, 1, 1);
    w.step(dt());
    let (v, z) = lone(1, 1);
    let half = Fix128::from_ratio(1, 2);
    for i in [a, b] {
        let p = w.bodies[i].position;
        assert_eq!((p.x, p.y, p.z), (half, Fix128::ZERO, z), "body {i}");
        assert_eq!(
            w.bodies[i].velocity, v,
            "body {i}: the placement error became velocity"
        );
    }
}

#[test]
fn a_joint_created_violated_closes_by_inverse_mass_shares() {
    let mut w = world();
    let (a, b) = drifting_pair(&mut w, 1, 3);
    w.step(dt());
    // w_a = 1, w_b = 1/3: A takes 3/4 of the 1 m error, B 1/4, both meet at x = 3/4
    let bound = 1.0 / (1u64 << 56) as f64;
    for (i, mass) in [(a, 1), (b, 3)] {
        let (v, z) = lone(mass, 1);
        let p = w.bodies[i].position;
        assert!(
            (p.x.to_f64() - 0.75).abs() <= bound,
            "body {i} at x = {}",
            p.x.to_f64()
        );
        assert_eq!((p.y, p.z), (Fix128::ZERO, z), "body {i}");
        assert_eq!(
            w.bodies[i].velocity, v,
            "body {i}: the placement error became velocity"
        );
    }
}

#[test]
fn a_body_moved_by_a_pose_write_alone_returns_to_its_partner() {
    let mut w = world();
    let (a, b) = drifting_pair(&mut w, 0, 1);
    w.step(dt());
    let before = w.bodies[a].position;
    let moved = Vec3Fix::new(before.x, before.y + Fix128::from_int(2), before.z);
    w.bodies[b].set_position(moved);
    w.step(dt());
    let (v, z) = lone(1, 2);
    // A is where the dynamics alone take it; B returns onto A
    let expected = Vec3Fix::new(before.x, before.y, z);
    assert_eq!(
        w.bodies[a].position, expected,
        "the partner the dynamics left was moved"
    );
    assert_eq!(
        w.bodies[b].position, expected,
        "the moved body did not return"
    );
    for i in [a, b] {
        assert_eq!(
            w.bodies[i].velocity, v,
            "body {i}: the pose write became velocity"
        );
    }
}

#[test]
fn a_violated_fixed_and_slider_chain_gains_no_energy() {
    let mut w = world();
    let f = |x: f64| Fix128::from_f64(x);
    let mut b = Vec::new();
    for i in 0..4 {
        b.push(w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(f(i as f64 * 0.7), f(2.0 + i as f64), f(0.3 * i as f64)),
            Fix128::from_int(1 + i),
        )));
    }
    w.add_joint(Joint::Fixed(FixedJoint::new(
        b[1],
        b[2],
        Vec3Fix::new(Fix128::ZERO, f(0.4), Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, f(-0.4), Fix128::ZERO),
        QuatFix::IDENTITY,
    )));
    w.add_joint(Joint::Slider(SliderJoint::new(
        b[2],
        b[3],
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    assert_eq!(kinetic_energy(&w), 0.0);
    for _ in 0..60 {
        w.step(dt());
        assert!(
            kinetic_energy(&w) <= 0.0,
            "kinetic energy {} from a placement error",
            kinetic_energy(&w)
        );
    }
}

#[test]
fn an_error_the_dynamics_made_still_moves_the_partner() {
    let mut w = world();
    let a = body(&mut w, 0, 0, 0, 1);
    let b = body(&mut w, 0, 0, 0, 1);
    w.add_joint(centre_ball(a, b));
    w.step(dt());
    w.bodies[a].velocity = Vec3Fix::from_int(1, 0, 0);
    w.step(dt());
    // the velocity write is not a pose move: the stretch is the dynamics', so the pair shares
    // the momentum (1 kg·m/s over 2 kg) and moves together
    for i in [a, b] {
        assert!(
            w.bodies[i].velocity.x > Fix128::ZERO,
            "body {i} did not move with the pair"
        );
    }
}

/// A spring of rest length 1 m created 2 m long, or a compliant ball joint 1 m apart: the stretch
/// is stored energy, so the bodies start moving toward each other (body A along +x, body B along
/// −x) instead of being placed.
#[test]
fn a_stretched_spring_and_a_compliant_joint_are_left_to_the_dynamics() {
    let joints = [
        |a, b| {
            Joint::Spring(SpringJoint::new(
                a,
                b,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Fix128::ONE,
                Fix128::from_int(100),
                Fix128::ZERO,
            ))
        },
        |a, b| {
            Joint::Ball(
                BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO)
                    .with_compliance(Fix128::from_ratio(1, 1000)),
            )
        },
    ];
    for (n, make) in joints.into_iter().enumerate() {
        let mut w = world();
        let a = body(&mut w, 0, 0, 0, 1);
        let b = body(&mut w, 2, 0, 0, 1);
        w.add_joint(make(a, b));
        w.step(dt());
        assert!(
            w.bodies[a].velocity.x > Fix128::ZERO,
            "joint {n}: body A not pulled"
        );
        assert!(
            w.bodies[b].velocity.x < Fix128::ZERO,
            "joint {n}: body B not pulled"
        );
        assert!(
            w.bodies[b].position.x > w.bodies[a].position.x + Fix128::ONE,
            "joint {n}: placed"
        );
    }
}

/// A slider created 6.08 m off its axis (√37 m, the scene that once broke a 1000 N joint in its
/// first substep) with a break force of 1000 N: the placement error is closed by position, which
/// carries no load, so the joint holds, the body sits on the axis after the first step (y and z
/// of the anchor, within rounding), and the kinetic energy stays at its starting 0.
#[test]
fn a_joint_created_far_off_settles_without_breaking_or_gaining_energy() {
    let mut w = world();
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(3, 6, 0)));
    let body = body(&mut w, 6, 5, 6, 1);
    let mut slider = SliderJoint::new(
        anchor,
        body,
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    slider.break_force = Some(Fix128::from_int(1000));
    w.add_joint(Joint::Slider(slider));
    // the correction's direction is the error over its length √37, a rounded quotient: a few
    // roundings of 2⁻⁶⁴ each (bound 2⁻⁵⁶ m, as for the inverse-mass shares)
    let bound = 1.0 / (1u64 << 56) as f64;
    for step in 0..60 {
        w.step(dt());
        assert_eq!(
            w.joints.len(),
            1,
            "step {step}: the placement error broke the joint"
        );
        let p = w.bodies[body].position;
        let off = ((p.y.to_f64() - 6.0).abs()).max(p.z.to_f64().abs());
        assert!(off <= bound, "step {step}: {off} m off the axis");
        assert!(
            kinetic_energy(&w) <= 0.0,
            "step {step}: kinetic energy {}",
            kinetic_energy(&w)
        );
    }
}
