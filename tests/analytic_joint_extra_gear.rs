//! The gear joint of `joint_extra` against its contract: `GearJoint` names two hinge
//! joints (`joint_a`, `joint_b`) and promises `θ_a + ratio · θ_b = const` on the
//! twists of its bodies about those hinges' axes. `solve_gear` reads each hinge's
//! axis from the joint list it is given (the hinge's local axis on the gear body's
//! side, in that body's current rotation), takes each body's twist over the step
//! about that axis (`rotation · prev_rotation⁻¹` projected on the axis) and corrects
//! each body about its own axis. The coupling must therefore hold about world `z`,
//! world `x` and a tilted axis alike, and a gear whose hinges are not in the list
//! must be refused, not skipped.
//!
//! Closed form of one projection with ratio `r`, inverse inertias `w_a`, `w_b`
//! about the axis and a twist `θ` on body A alone. `solve_gear` does not read `θ`
//! itself: `extract_angle_around_axis` returns `2 · (xyz · axis)` of the `w ≥ 0`
//! quaternion, which is `2 sin(θ/2)`, so `λ = 2 sin(θ/2) / (w_a + r² w_b)`. With
//! `a = λ w_a` and `b = λ r w_b` the two corrections are the quaternions
//! `(axis · a/2, 1)` and `(axis · b/2, 1)` normalised (`rotate_by`), whose angles
//! are `2 · atan(a/2) = a − a³/12 + a⁵/80 − …` (the same in `b`). The twists
//! measured afterwards, `2 · atan2` of the composed quaternion, are
//! `θ_a = θ − 2 atan(a/2)` and `θ_b = −2 atan(b/2)`; with
//! `a + r b = 2 sin(θ/2) = θ − θ³/24 + θ⁵/1920 − …` they satisfy
//!
//! `θ_a + r θ_b = [θ − 2 sin(θ/2)] + (a³ + r b³)/12 − …`
//! `= θ³/24 + (a³ + r b³)/12 + O(θ⁵)`.
//!
//! The fifth-order terms dropped, `θ⁵/1920 + (a⁵ + r b⁵)/80`, enter the residual
//! with a minus sign, so `θ³/24 + (a³ + r b³)/12` is an upper bound in exact
//! arithmetic with that much margin (about `4e-8` at `θ = 1/8`, `r = 1`), far
//! above the `~2⁻⁴⁸` error of the one CORDIC `sin_cos` in `from_axis_angle` and
//! the `2⁻⁶⁴` rounding of `Fix128`. The bound is evaluated in f64 from `θ`, `r`,
//! `w_a` and `w_b`, never by the solver under test.
//!
//! Rate over many steps: a driver A that the solver cannot move (`w_a = 0`) turned
//! by `θ` every step makes the follower B turn by `2 atan(b/2)` with
//! `b = 2 sin(θ/2) / r` every step, so after `N` steps `θ_B = −N · 2 atan(sin(θ/2) / r)`
//! exactly in exact arithmetic (every turn is about the same axis, so the angles
//! add), and the rate ratio `θ_B / (N θ)` differs from `−1/r` by the one-step
//! residual bound above (with `a = 0`) per step at most.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the bound is a closed form in f64, never the solver under test"
)]

use alice_physics_core::joint::{BallJoint, HingeJoint, Joint};
use alice_physics_core::joint_extra::{solve_extra_joints, ExtraJoint, ExtraJointError, GearJoint};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::solver::RigidBody;

/// bodies 0 and 1 with unit mass and unit isotropic inverse inertia, body 2 a static
/// frame both hinges attach to; A (body 0) carries a twist `theta` about `axis` over
/// the step (`prev_rotation` identity), B (body 1) is at rest
fn pair_with_twist(axis: Vec3Fix, theta: Fix128) -> Vec<RigidBody> {
    let mut a = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    a.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    a.prev_rotation = QuatFix::IDENTITY;
    a.rotation = QuatFix::from_axis_angle(axis, theta);
    let mut b = RigidBody::new(
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    );
    b.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    b.prev_rotation = QuatFix::IDENTITY;
    b.rotation = QuatFix::IDENTITY;
    let mut frame = RigidBody::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    frame.inv_mass = Fix128::ZERO;
    frame.inv_inertia = Vec3Fix::ZERO;
    vec![a, b, frame]
}

/// the two hinges the gear names: index 0 joins body 0 to the frame, index 1 body 1
/// to the frame, both about `axis` (given in the bodies' local frames, identity at
/// rest)
fn hinges(axis: Vec3Fix) -> Vec<Joint> {
    let o = Vec3Fix::ZERO;
    vec![
        Joint::Hinge(HingeJoint::new(0, 2, o, o, axis, axis)),
        Joint::Hinge(HingeJoint::new(1, 2, o, o, axis, axis)),
    ]
}

/// the signed twist of `rotation · prev_rotation⁻¹` about a unit `axis`:
/// `2 · atan2(xyz · axis, w)` from the `w ≥ 0` representative
fn twist_about(body: &RigidBody, axis: Vec3Fix) -> f64 {
    let q = body.rotation.mul(body.prev_rotation.conjugate());
    let (x, y, z, w) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64());
    let (x, y, z, w) = if w < 0.0 {
        (-x, -y, -z, -w)
    } else {
        (x, y, z, w)
    };
    let (ax, ay, az) = (axis.x.to_f64(), axis.y.to_f64(), axis.z.to_f64());
    2.0 * (x * ax + y * ay + z * az).atan2(w)
}

fn gear(ratio: Fix128) -> Vec<ExtraJoint> {
    vec![ExtraJoint::Gear(GearJoint::new(0, 1, 0, 1, ratio))]
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// bound on `|θ_a + r θ_b|` after one projection (module doc): the third-order
/// terms of the twist read as `2 sin(θ/2)` and of the two corrections applied as
/// `(axis · x/2, 1)` normalised; the fifth-order terms dropped only lower the
/// residual. `w_a`, `w_b` are the inverse inertias about the hinge axis
fn residual_bound(theta: f64, ratio: f64, w_a: f64, w_b: f64) -> f64 {
    let theta = theta.abs();
    let lambda = 2.0 * (theta / 2.0).sin() / (w_a + ratio * ratio * w_b);
    let (a, b) = (lambda * w_a, lambda * ratio * w_b);
    theta.powi(3) / 24.0 + (a.powi(3) + ratio * b.powi(3)) / 12.0
}

/// one projection about `axis`, ratio 1 and 2: the residual is within the bound and
/// B turned
fn one_projection_is_coupled(axis: Vec3Fix, name: &str) {
    for (num, den) in [(1, 1), (2, 1)] {
        let ratio = Fix128::from_ratio(num, den);
        let theta = Fix128::from_ratio(1, 8);
        let mut bodies = pair_with_twist(axis, theta);
        solve_extra_joints(&mut bodies, &gear(ratio), &hinges(axis), dt())
            .expect("both hinges are in the list");
        let ta = twist_about(&bodies[0], axis);
        let tb = twist_about(&bodies[1], axis);
        let residual = ta + ratio.to_f64() * tb;
        let bound = residual_bound(theta.to_f64(), ratio.to_f64(), 1.0, 1.0);
        eprintln!(
            "{name}, ratio {num}/{den}: θ_a {ta:.6e} θ_b {tb:.6e} \
             residual {residual:.3e} bound {bound:.3e}"
        );
        assert!(
            tb != 0.0,
            "{name}: B stayed at rest, the twist was not seen"
        );
        assert!(
            residual.abs() <= bound,
            "{name}: residual {residual:.3e} above {bound:.3e}"
        );
    }
}

/// hinges about world `z`
#[test]
fn a_twist_about_world_z_is_coupled_in_the_ratio() {
    one_projection_is_coupled(Vec3Fix::UNIT_Z, "z");
}

/// hinges about world `x`: the axis the old solver never read
#[test]
fn a_twist_about_the_hinge_axis_x_is_coupled_in_the_ratio() {
    one_projection_is_coupled(Vec3Fix::UNIT_X, "x");
}

/// hinges about `(1, 1, 1)/√3`: no component is special
#[test]
fn a_twist_about_a_tilted_hinge_axis_is_coupled_in_the_ratio() {
    let axis = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE).normalize();
    one_projection_is_coupled(axis, "tilted");
}

/// Rate over 120 steps about world `x`: a driver the solver cannot move is turned
/// by `θ = 1/64` rad every step, the follower's total turn is `−N θ / r` within the
/// per-step residual bound summed, for `r = 2` and `r = 1/2`.
#[test]
fn a_driven_gear_turns_the_follower_at_the_ratio_over_many_steps() {
    const STEPS: usize = 120;
    let axis = Vec3Fix::UNIT_X;
    let theta = Fix128::from_ratio(1, 64);
    for (num, den) in [(2, 1), (1, 2)] {
        let ratio = Fix128::from_ratio(num, den);
        let mut bodies = pair_with_twist(axis, Fix128::ZERO);
        // the driver: a kinematic body, no mass and no inertia for the solver to move
        bodies[0].inv_mass = Fix128::ZERO;
        bodies[0].inv_inertia = Vec3Fix::ZERO;
        let turn = QuatFix::from_axis_angle(axis, theta);
        let joints = gear(ratio);
        let hinge_list = hinges(axis);
        for _ in 0..STEPS {
            for b in bodies.iter_mut() {
                b.prev_rotation = b.rotation;
            }
            bodies[0].rotation = turn.mul(bodies[0].rotation).normalize();
            solve_extra_joints(&mut bodies, &joints, &hinge_list, dt())
                .expect("both hinges are in the list");
        }
        // the follower only ever turned about `x`, so its angle is read from its
        // quaternion directly
        let q = bodies[1].rotation;
        let total_b = 2.0 * q.x.to_f64().atan2(q.w.to_f64());
        let total_a = STEPS as f64 * theta.to_f64();
        let residual = total_a + ratio.to_f64() * total_b;
        let bound = STEPS as f64 * residual_bound(theta.to_f64(), ratio.to_f64(), 0.0, 1.0);
        let rate = total_b / total_a;
        eprintln!(
            "ratio {num}/{den}: θ_A {total_a:.6e} θ_B {total_b:.6e} rate {rate:.6} \
             (closed form {:.6}) residual {residual:.3e} bound {bound:.3e}",
            -1.0 / ratio.to_f64()
        );
        assert!(
            residual.abs() <= bound,
            "ratio {num}/{den}: residual {residual:.3e} above {bound:.3e}"
        );
        assert!(
            (bodies[1].rotation.y.to_f64()).abs() < 1e-15
                && (bodies[1].rotation.z.to_f64()).abs() < 1e-15,
            "the follower turned off its hinge axis"
        );
    }
}

/// A gear whose hinges are not there to read is refused before it touches a body:
/// an index past the list, a joint that is not a hinge, a body on neither side.
#[test]
fn a_gear_without_its_hinges_is_refused_and_moves_nothing() {
    let axis = Vec3Fix::UNIT_X;
    let theta = Fix128::from_ratio(1, 8);
    let o = Vec3Fix::ZERO;
    let cases: [(Vec<Joint>, GearJoint, ExtraJointError); 3] = [
        (
            vec![Joint::Hinge(HingeJoint::new(0, 2, o, o, axis, axis))],
            GearJoint::new(0, 1, 0, 1, Fix128::ONE),
            ExtraJointError::HingeIndexOutOfRange { index: 1, count: 1 },
        ),
        (
            vec![
                Joint::Hinge(HingeJoint::new(0, 2, o, o, axis, axis)),
                Joint::Ball(BallJoint::new(1, 2, o, o)),
            ],
            GearJoint::new(0, 1, 0, 1, Fix128::ONE),
            ExtraJointError::NotAHinge { index: 1 },
        ),
        (
            hinges(axis),
            GearJoint::new(0, 1, 0, 0, Fix128::ONE),
            ExtraJointError::BodyNotOnHinge { body: 1, hinge: 0 },
        ),
    ];
    for (list, gear, want) in cases {
        let mut bodies = pair_with_twist(axis, theta);
        let before = bodies.clone();
        let got = solve_extra_joints(&mut bodies, &[ExtraJoint::Gear(gear)], &list, dt());
        assert_eq!(got, Err(want), "{want}");
        assert_eq!(bodies, before, "{want}: a refused gear must move nothing");
    }
}
