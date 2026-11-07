//! The gear joint of `joint_extra` against its own contract: `GearJoint` names two
//! hinge joints (`joint_a`, `joint_b`) and promises `θ_a + ratio · θ_b = const` on
//! their angles. `solve_gear` reads neither hinge: it takes each body's twist over
//! the step about world `z` (`rotation · prev_rotation⁻¹` projected on `z`) and
//! corrects about world `z`. So the coupling holds only for hinges whose axis is
//! world `z`; about any other hinge axis the bodies are not coupled at all.
//!
//! Closed form of one projection with ratio `r`, inverse inertias `w_a`, `w_b`
//! about the axis and a twist `θ` on body A alone: `λ = θ / (w_a + r² w_b)`,
//! A turns by `−λ w_a` and B by `−λ r w_b`, so the twists after the solve satisfy
//! `θ_a + r θ_b = 0` exactly in real arithmetic. The correction is applied as the
//! quaternion `(0, 0, −λw/2, 1)` normalised, whose angle is `2·atan(λw/2) = λw −
//! (λw)³/12 + …`, so the residual of the sum is below `(|θ|)³ / 6` for `|θ| < 1`
//! (both bodies' cubic terms, each `≤ |θ|³/12`), plus the `2⁻⁴⁰` of the one CORDIC
//! `sin_cos` in `from_axis_angle`.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the twist bound θ³/6 is a closed form evaluated in f64, never by the solver under test"
)]

use alice_physics_core::joint_extra::{solve_extra_joints, ExtraJoint, GearJoint};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::solver::RigidBody;

/// `2^-40`: 256x the documented `~2^-48` CORDIC error of the one `sin_cos` call.
fn cordic_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24)
}

/// two unit-mass bodies with unit inverse inertia; A carries a twist `theta`
/// about `axis` over the step (`prev_rotation` identity), B is at rest
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
    vec![a, b]
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
    // hinge indices 0 and 1: the joints the gear claims to couple
    vec![ExtraJoint::Gear(GearJoint::new(0, 1, 0, 1, ratio))]
}

/// bound on `|θ_a + r θ_b|` after one projection: the cubic term of the
/// small-angle quaternion on each body plus the CORDIC tolerance
fn residual_bound(theta: f64) -> f64 {
    theta.abs().powi(3) / 6.0 + cordic_tol().to_f64()
}

/// Today's behaviour, pinned: a twist about world `z` is coupled by the
/// gear (`θ_a + r θ_b = 0` within the small-angle bound), ratio 1 and 2.
#[test]
fn a_twist_about_world_z_is_coupled_in_the_ratio() {
    for (num, den) in [(1, 1), (2, 1)] {
        let ratio = Fix128::from_ratio(num, den);
        let theta = Fix128::from_ratio(1, 8);
        let mut bodies = pair_with_twist(Vec3Fix::UNIT_Z, theta);
        solve_extra_joints(&mut bodies, &gear(ratio), Fix128::from_ratio(1, 60));
        let ta = twist_about(&bodies[0], Vec3Fix::UNIT_Z);
        let tb = twist_about(&bodies[1], Vec3Fix::UNIT_Z);
        let residual = ta + ratio.to_f64() * tb;
        let bound = residual_bound(theta.to_f64());
        eprintln!("z, ratio {num}/{den}: θ_a {ta:.6e} θ_b {tb:.6e} residual {residual:.3e} bound {bound:.3e}");
        assert!(
            residual.abs() <= bound,
            "residual {residual:.3e} above {bound:.3e}"
        );
        assert!(tb != 0.0, "B must turn");
    }
}

/// The contract: the gear couples the two hinges it names, whatever their axis.
/// With the hinge axis along world `x` the same twist must be coupled the same way.
/// Red today: `solve_gear` reads the twist about world `z` only, sees `0`, and
/// returns without touching either body, so B stays at rest and the residual is
/// the whole `θ`.
#[test]
#[ignore = "known defect: solve_gear reads each body's per-step twist about world z and never the hinge joints it names (joint_a / joint_b), so a gear pair on hinges about any other axis is not coupled: a twist of 1/8 rad about world x on body A leaves body B at rest and the residual θ_a + ratio θ_b equals the twist"]
fn a_twist_about_the_hinge_axis_x_is_coupled_in_the_ratio() {
    let ratio = Fix128::ONE;
    let theta = Fix128::from_ratio(1, 8);
    let mut bodies = pair_with_twist(Vec3Fix::UNIT_X, theta);
    solve_extra_joints(&mut bodies, &gear(ratio), Fix128::from_ratio(1, 60));
    let ta = twist_about(&bodies[0], Vec3Fix::UNIT_X);
    let tb = twist_about(&bodies[1], Vec3Fix::UNIT_X);
    let residual = ta + ratio.to_f64() * tb;
    let bound = residual_bound(theta.to_f64());
    eprintln!("x: θ_a {ta:.6e} θ_b {tb:.6e} residual {residual:.3e} bound {bound:.3e}");
    assert!(tb != 0.0, "B stayed at rest: the x twist was not seen");
    assert!(
        residual.abs() <= bound,
        "residual {residual:.3e} above {bound:.3e}"
    );
}
