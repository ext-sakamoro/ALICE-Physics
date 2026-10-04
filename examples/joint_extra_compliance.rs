//! Joint Extra Compliance Example
//!
//! Production entry point for the constructors and compliance builders of
//! `src/joint_extra.rs` that `examples/joint_extra_wiring.rs` does not use:
//! `GearJoint::new`, `RackAndPinionJoint::new` and `with_compliance` on
//! `PulleyJoint`, `GearJoint`, `WeldJoint` and `RackAndPinionJoint`.
//!
//! Every joint here is a single XPBD projection, so the expected state is
//! written out from the constraint and the compliance alone:
//! `λ = C / (Σ w + α / dt²)` and each body moves by `λ w` along its gradient.
//! The compliance is chosen so that `α / dt²` is a round number relative to the
//! generalized inverse mass, and every scene is set up so that the inverse
//! mass along the constrained direction is unambiguous (inverse inertia only
//! about `z`, the axis the gear and the pinion turn about).
//!
//! - pulley: `C = len_a + r len_b − L`, both bodies lifted along their ropes
//! - gear: `C = 2 z_a + r · 2 z_b` (the small-angle twist of each body's turn
//!   since the previous step), body A turned back by the quaternion
//!   `(0, 0, −λ w_a / 2, 1)`, normalized
//! - weld: anchors pulled together by `λ = d / (w_a + w_b + α / dt²)`
//! - rack and pinion: `C = d − r · 2 z_p`, the rack moved back along its axis
//!
//! Run with: `cargo run --example joint_extra_compliance`

use alice_physics::joint_extra::{
    solve_extra_joints, solve_pulley_to_length, ExtraJoint, GearJoint, PulleyJoint,
    RackAndPinionJoint, WeldJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn close(got: Fix128, want: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= 1e-12,
        "{what}: got {g:.15}, closed form {want:.15}"
    );
}

/// A dynamic unit-mass body that can only turn about `z` (inverse inertia 1).
fn spinner(position: Vec3Fix) -> RigidBody {
    let mut b = RigidBody::new(position, Fix128::ONE);
    b.inv_inertia = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    b
}

/// A static body turned about `z` by the unit quaternion `(0, 0, z, w)` since
/// its previous step.
fn turned_static(z: Fix128, w: Fix128) -> RigidBody {
    let mut b = RigidBody::new_static(Vec3Fix::ZERO);
    b.inv_inertia = Vec3Fix::ZERO;
    b.prev_rotation = QuatFix::IDENTITY;
    b.rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, z, w);
    b
}

/// `(z, w)` of `normalize((1 − h k)(w0 + z0 k))`.
fn turned_back(z0: f64, w0: f64, h: f64) -> (f64, f64) {
    let (w, z) = (w0 + h * z0, z0 - h * w0);
    let n = (w * w + z * z).sqrt();
    (z / n, w / n)
}

fn pulley(dt: Fix128) {
    // len_a = 5, len_b = 4, ratio 2: total 13 against a rest length of 10.
    let joint = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(10, 0, 0),
        Fix128::from_int(2),
    )
    .with_compliance(dt * dt);
    assert_eq!(joint.compliance, dt * dt, "with_compliance stores α");
    let mut bodies = [
        RigidBody::new(Vec3Fix::from_int(0, -5, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(10, -4, 0), Fix128::ONE),
    ];
    solve_pulley_to_length(&joint, &mut bodies, Fix128::from_int(10), dt);
    // C = 3, Σw = 1 + 2² · 1 = 5, α/dt² = 1: λ = 1/2.
    // A rises by λ w_a = 1/2, B by λ r w_b = 1.
    let lambda = 3.0 / (1.0 + 4.0 + 1.0);
    close(
        bodies[0].position.y,
        -5.0 + lambda,
        "pulley: body A lifted by λ",
    );
    close(
        bodies[1].position.y,
        -4.0 + 2.0 * lambda,
        "pulley: body B lifted by λ r",
    );
    println!(
        "pulley: λ = {lambda}, A y = {}, B y = {}",
        bodies[0].position.y.to_f64(),
        bodies[1].position.y.to_f64()
    );
}

fn gear(dt: Fix128) {
    // A turned by (0, 0, 7/25, 24/25), B (static) by (0, 0, 3/5, 4/5).
    let (za, wa, zb, wb) = (q(7, 25), q(24, 25), q(3, 5), q(4, 5));
    let ratio = q(1, 2);
    let mut a = spinner(Vec3Fix::ZERO);
    a.prev_rotation = QuatFix::IDENTITY;
    a.rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, za, wa);
    let b = turned_static(zb, wb);

    let error = 2.0 * za.to_f64() + ratio.to_f64() * 2.0 * zb.to_f64();
    for (compliance_over_dt2, label) in [(0i64, "rigid"), (1, "compliant")] {
        let joint = GearJoint::new(0, 1, 7, 8, ratio)
            .with_compliance(Fix128::from_int(compliance_over_dt2) * dt * dt);
        assert_eq!(
            (
                joint.body_a,
                joint.body_b,
                joint.joint_a,
                joint.joint_b,
                joint.ratio
            ),
            (0, 1, 7, 8, ratio),
            "GearJoint::new stores its arguments"
        );
        let mut bodies = [a, b];
        solve_extra_joints(&mut bodies, &[ExtraJoint::Gear(joint)], dt);
        // w_a = 1, w_b = 0: λ w_a = C / (1 + α/dt²).
        let h = error / (1.0 + compliance_over_dt2 as f64) / 2.0;
        let (z, w) = turned_back(za.to_f64(), wa.to_f64(), h);
        close(bodies[0].rotation.z, z, "gear: body A turned back (z)");
        close(bodies[0].rotation.w, w, "gear: body A turned back (w)");
        assert_eq!(
            bodies[1].rotation, b.rotation,
            "a static gear does not turn"
        );
        println!("gear ({label}): C = {error}, A rotation z = {z:.12}, w = {w:.12}");
    }
}

fn weld(dt: Fix128) {
    let mut bodies = [
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE),
    ];
    let joint = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_compliance(Fix128::from_int(2) * dt * dt);
    solve_extra_joints(&mut bodies, &[ExtraJoint::Weld(joint)], dt);
    // d = 5 along (3/5, 4/5, 0), Σw = 2, α/dt² = 2: λ = 5/4, each body moves λ.
    let lambda = 5.0 / (1.0 + 1.0 + 2.0);
    close(bodies[0].position.x, 0.6 * lambda, "weld: A toward B (x)");
    close(bodies[0].position.y, 0.8 * lambda, "weld: A toward B (y)");
    close(
        bodies[1].position.x,
        3.0 - 0.6 * lambda,
        "weld: B toward A (x)",
    );
    close(
        bodies[1].position.y,
        4.0 - 0.8 * lambda,
        "weld: B toward A (y)",
    );
    println!(
        "weld: λ = {lambda}, gap after = {:.6}",
        (bodies[1].position - bodies[0].position).length().to_f64()
    );
}

fn rack_and_pinion(dt: Fix128) {
    let ratio = q(1, 2);
    let (zp, wp) = (q(7, 25), q(24, 25));
    let mut rack = RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    rack.prev_position = Vec3Fix::ZERO;
    let pinion = turned_static(zp, wp);

    let error = 1.0 - ratio.to_f64() * 2.0 * zp.to_f64();
    for (compliance_over_dt2, label) in [(0i64, "rigid"), (1, "compliant")] {
        let joint = RackAndPinionJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, ratio)
            .with_compliance(Fix128::from_int(compliance_over_dt2) * dt * dt);
        assert_eq!(
            (
                joint.body_rack,
                joint.body_pinion,
                joint.rack_axis,
                joint.pinion_axis
            ),
            (0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z),
            "RackAndPinionJoint::new stores its arguments"
        );
        let mut bodies = [rack, pinion];
        solve_extra_joints(&mut bodies, &[ExtraJoint::RackAndPinion(joint)], dt);
        // w_linear = 1, w_angular = 0: the rack moves back by C / (1 + α/dt²).
        let want = 1.0 - error / (1.0 + compliance_over_dt2 as f64);
        close(bodies[0].position.x, want, "rack moved back along its axis");
        println!("rack and pinion ({label}): C = {error}, rack x = {want:.12}");
    }
}

fn main() {
    let dt = q(1, 4);
    pulley(dt);
    gear(dt);
    weld(dt);
    rack_and_pinion(dt);
    println!("all closed forms hold");
}
