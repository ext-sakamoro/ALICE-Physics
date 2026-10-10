//! Joint limits, D6 motion modes and breakable joints — the production
//! entry point for the `joint` builders (`with_limits`, `with_linear_limits`,
//! `with_angular_limits`, `with_linear_motion`, `with_angular_motion`,
//! `with_break_force`), the `JointType` / `joint_type` / `break_force`
//! queries and `solve_joints_breaking_on_force`.
//!
//! Every section prints the closed-form value next to the solver output.
//!
//! ```bash
//! cargo run --example joint_limits_and_breaking --features std
//! ```
//!
//! # Closed forms (all values dyadic, exact in `Fix128` unless noted)
//!
//! * **Breakable ramp**: body B (mass `m = 2`) starts on the anchor of a
//!   rigid ball joint to a static body A. On step `k` a pull `F_k = k·F0`
//!   (`F0 = 1`) acting over `dt = 1/4` from rest predicts a separation
//!   `d_k = F_k·dt²/m = k/32`. The XPBD multiplier of the one-constraint
//!   scene is `λ = d_k / w = d_k·m` and the constraint force `λ/dt² = F_k`
//!   exactly. `solve_joints_breaking_on_force` solves the joint, takes the
//!   reaction force that solve transmitted (`solve_joints_with_reaction_forces`,
//!   N) and reports it broken on the first step with `F_k > break_force`
//!   (strict). With
//!   `break_force = 5` N the joint holds through `k = 5` (equality) and
//!   breaks on `k* = ⌊F_b/F0⌋ + 1 = 6`. `PhysicsWorld::step` applies the same
//!   check to the world's joints and reports each break as a
//!   `JointBreakEvent`. While held, the solve moves B back onto the
//!   anchor bit for bit (`normal·λ·inv_mass = d_k`).
//! * **Slider / D6 linear limit `[lo, hi]`**: with A static and B at
//!   `inv_mass = 1/2`, one solve moves B by exactly the overshoot
//!   (`error/inv_w · inv_mass = error`), so a predicted travel `3` with
//!   `hi = 1` lands on `1`, travel `−3` with `lo = −1/2` lands on `−1/2`,
//!   and a travel inside the interval is untouched. `D6Motion::Locked`
//!   lands on `0`, `Free` is untouched.
//! * **Hinge / D6 angular limit / cone-twist**: B rotated by `θ = 1 rad`
//!   about the limited axis with limit `1/2` lands on `1/2`; the rotation
//!   is a true-angle quaternion rotation and the angle is read back with
//!   `2·atan2(q_z, q_w)`, both 48-iteration CORDIC (≈ 2⁻⁴⁸ each), so the
//!   printed difference is of order 2⁻⁴⁶.
//!
//! Author: Moroya Sakamoto

use alice_physics::joint::{
    solve_joints, solve_joints_breaking_on_force, BallJoint, ConeTwistJoint, D6Joint, D6Motion,
    FixedJoint, HingeJoint, Joint, JointType, SliderJoint, SpringJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Static A at the origin and a dynamic B with the given inverse mass and
/// unit inverse inertia, both at the origin.
fn pair(inv_mass_b: Fix128) -> Vec<RigidBody> {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_mass = inv_mass_b;
    b.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    vec![RigidBody::new_static(Vec3Fix::ZERO), b]
}

/// Signed twist of a rotation about +z, read independently of the joint
/// module: `2·atan2(q_z, q_w)` on the `w ≥ 0` cover.
fn twist_about_z(rot: QuatFix) -> Fix128 {
    let (z, w) = if rot.w.is_negative() {
        (-rot.z, -rot.w)
    } else {
        (rot.z, rot.w)
    };
    Fix128::atan2(z, w).double()
}

/// Angle between the rotated +z axis and +z: `atan2(|a × z|, a · z)`.
fn cone_angle_from_z(rot: QuatFix) -> Fix128 {
    let a = rot.rotate_vec(Vec3Fix::UNIT_Z);
    Fix128::atan2(a.cross(Vec3Fix::UNIT_Z).length(), a.dot(Vec3Fix::UNIT_Z))
}

fn kind_name(kind: JointType) -> &'static str {
    match kind {
        JointType::Ball => "Ball",
        JointType::Hinge => "Hinge",
        JointType::Fixed => "Fixed",
        JointType::Slider => "Slider",
        JointType::Spring => "Spring",
        JointType::D6 => "D6",
        JointType::ConeTwist => "ConeTwist",
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let dt = q(1, 4);
    let half = q(1, 2);
    let o = Vec3Fix::ZERO;

    // ---- 1. every joint type carries a break force and reports its kind ----
    let threshold = Fix128::from_int(5); // N
    let catalogue = [
        Joint::Ball(BallJoint::new(0, 1, o, o).with_break_force(threshold)),
        Joint::Hinge(
            HingeJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
                .with_limits(-half, half)
                .with_break_force(threshold),
        ),
        Joint::Fixed(FixedJoint::new(0, 1, o, o, QuatFix::IDENTITY).with_break_force(threshold)),
        Joint::Slider(
            SliderJoint::new(0, 1, Vec3Fix::UNIT_X, o, o)
                .with_limits(-half, Fix128::ONE)
                .with_break_force(threshold),
        ),
        Joint::Spring(
            SpringJoint::new(0, 1, o, o, Fix128::ONE, Fix128::from_int(4), Fix128::ZERO)
                .with_break_force(threshold),
        ),
        Joint::D6(
            D6Joint::new(0, 1, o, o)
                .with_linear_motion(D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
                .with_linear_limits(
                    Vec3Fix::new(-half, -half, -half),
                    Vec3Fix::new(half, Fix128::ONE, half),
                )
                .with_angular_motion(D6Motion::Free, D6Motion::Free, D6Motion::Limited)
                .with_angular_limits(
                    Vec3Fix::new(-half, -half, -half),
                    Vec3Fix::new(half, half, half),
                )
                .with_break_force(threshold),
        ),
        Joint::ConeTwist(
            ConeTwistJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
                .with_limits(half, q(1, 4))
                .with_break_force(threshold),
        ),
    ];
    println!("[joint] catalogue: kind / break_force (expect 7 distinct kinds, all Some(5.0))");
    for j in &catalogue {
        let bf = j.break_force().map(|f| f.to_f64());
        println!(
            "[joint]   {:<9} break_force={bf:?}",
            kind_name(j.joint_type())
        );
    }

    // ---- 2. breakable ramp: closed form k* = floor(F_b / F0) + 1 = 6 ----
    let m = Fix128::from_int(2);
    let f0 = Fix128::ONE;
    let fb_force = Fix128::from_int(5);
    println!(
        "[joint] breakable ramp: m={} dt={} F0={} break_force={} N (the reaction force lambda/dt^2), k*=6",
        m.to_f64(),
        dt.to_f64(),
        f0.to_f64(),
        fb_force.to_f64(),
    );
    let ramp = [Joint::Ball(
        BallJoint::new(0, 1, o, o).with_break_force(fb_force),
    )];
    let mut bodies = pair(Fix128::ONE / m);
    let mut broke_at = None;
    for k in 1..=8i64 {
        // predicted separation under F_k = k·F0 from rest over dt: d_k = k F0 dt² / m
        let d_k = Fix128::from_int(k) * f0 * dt * dt / m;
        bodies[1].position = Vec3Fix::new(Fix128::ZERO, -d_k, Fix128::ZERO);
        let lambda = d_k * m;
        let force = lambda / (dt * dt);
        let broken = solve_joints_breaking_on_force(&ramp, &mut bodies, dt);
        println!(
            "[joint]   k={k} d_k={:.5} lambda/dt^2={} (expect F_k={k}) broken={broken:?} B.y after={:.5}",
            d_k.to_f64(),
            force.to_f64(),
            bodies[1].position.y.to_f64()
        );
        if !broken.is_empty() && broke_at.is_none() {
            broke_at = Some(k);
        }
    }
    println!("[joint]   first break at k={broke_at:?} (expect Some(6))");

    // ---- 2b. the world breaks an overloaded joint inside step ----
    // a 2 kg body hung by a rigid ball joint from a static anchor: after one
    // gravity substep (h = 1/64, g = 8) the joint's reaction force is m·g = 16 N
    {
        use alice_physics::{PhysicsConfig, PhysicsWorld};
        for limit in [15, 17] {
            let mut world = PhysicsWorld::new(PhysicsConfig {
                substeps: 1,
                gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
                damping: Fix128::ONE,
                ..PhysicsConfig::default()
            });
            world.add_body(RigidBody::new_static(o));
            world.add_body(RigidBody::new(o, Fix128::from_int(2)));
            world.add_joint(Joint::Ball(
                BallJoint::new(0, 1, o, o).with_break_force(Fix128::from_int(limit)),
            ));
            world.step(Fix128::from_ratio(1, 64));
            let seen = world.events.joint_break_events().len();
            let drained = world.events.drain_joint_break_events();
            println!(
                "[joint] world step, break_force {limit} N vs m*g = 16 N: joints left {} (expect {}), events {seen}, drained force {:?} (expect {})",
                world.joints.len(),
                usize::from(limit > 16),
                drained.first().map(|e| e.force.to_f64()),
                if limit > 16 { "None" } else { "Some(16.0)" },
            );
        }
    }

    // ---- 2c. migrating from the separation-based check ----
    // `solve_joints_breakable` (deprecated) compares `Joint::compute_force`, the
    // anchor separation in metres; `solve_joints_breaking_on_force` compares the
    // reaction force in newtons. The same k = 6 scene measures both ways.
    {
        let m = Fix128::from_int(2);
        let d6 = Fix128::from_int(6) * dt * dt / m;
        let mut bodies = pair(Fix128::ONE / m);
        bodies[1].position = Vec3Fix::new(Fix128::ZERO, -d6, Fix128::ZERO);
        let probe = Joint::Ball(BallJoint::new(0, 1, o, o));
        println!(
            "[joint] migration: separation {} m (compute_force), reaction force {} N (solve_joints_with_reaction_forces); expect 0.1875 m and 6 N",
            probe.compute_force(&bodies).to_f64(),
            alice_physics::joint::solve_joints_with_reaction_forces(&[probe], &mut bodies.clone(), dt)[0].to_f64(),
        );
        // the old threshold for the same 5 N limit was the separation 5/32 m
        let old = [Joint::Ball(
            BallJoint::new(0, 1, o, o).with_break_force(q(5, 32)),
        )];
        #[allow(deprecated)] // shown for migration; removed in 3.0
        let broken = alice_physics::joint::solve_joints_breakable(&old, &mut bodies, dt);
        println!("[joint] migration: deprecated separation check with 5/32 m breaks too: {broken:?} (expect [0])");
    }

    // ---- 3. slider limits: land exactly on the violated limit ----
    let slider = [Joint::Slider(
        SliderJoint::new(0, 1, Vec3Fix::UNIT_X, o, o).with_limits(-half, Fix128::ONE),
    )];
    for (travel, expect) in [
        (q(3, 1), Fix128::ONE),
        (q(-3, 1), -half),
        (q(1, 4), q(1, 4)),
    ] {
        let mut b = pair(half);
        b[1].position = Vec3Fix::new(travel, Fix128::ZERO, Fix128::ZERO);
        solve_joints(&slider, &mut b, dt);
        println!(
            "[joint] slider limits [-1/2, 1]: travel {} -> x={} (expect {})",
            travel.to_f64(),
            b[1].position.x.to_f64(),
            expect.to_f64()
        );
    }

    // ---- 4. hinge angular limit [-1/2, 1/2]: θ = 1 lands on 1/2 ----
    let hinge = [Joint::Hinge(
        HingeJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z).with_limits(-half, half),
    )];
    let mut b = pair(Fix128::ONE);
    b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
    solve_joints(&hinge, &mut b, dt);
    let landed = twist_about_z(b[1].rotation);
    println!(
        "[joint] hinge limits [-1/2, 1/2]: theta 1 -> {} (expect 0.5, |diff| {:e} ~ 2^-46)",
        landed.to_f64(),
        (landed - half).to_f64()
    );

    // ---- 5. D6 motion modes + linear / angular limits ----
    let d6 = [Joint::D6(
        D6Joint::new(0, 1, o, o)
            .with_linear_motion(D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
            .with_linear_limits(
                Vec3Fix::new(-half, -half, -half),
                Vec3Fix::new(half, Fix128::ONE, half),
            )
            .with_angular_motion(D6Motion::Free, D6Motion::Free, D6Motion::Limited)
            .with_angular_limits(
                Vec3Fix::new(-half, -half, -half),
                Vec3Fix::new(half, half, half),
            ),
    )];
    let mut b = pair(half);
    b[1].position = Vec3Fix::from_int(4, 3, 5);
    b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
    solve_joints(&d6, &mut b, dt);
    println!(
        "[joint] D6 linear (Locked, Limited[-1/2,1], Free) from (4,3,5) -> ({}, {}, {}) (expect (0, 1, 5))",
        b[1].position.x.to_f64(),
        b[1].position.y.to_f64(),
        b[1].position.z.to_f64()
    );
    println!(
        "[joint] D6 angular z Limited[-1/2,1/2]: theta 1 -> {} (expect 0.5)",
        twist_about_z(b[1].rotation).to_f64()
    );

    // ---- 6. cone-twist limits (cone 1/2, twist 1/4) ----
    let cone_twist = [Joint::ConeTwist(
        ConeTwistJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
            .with_limits(half, q(1, 4)),
    )];
    let mut b = pair(Fix128::ONE);
    b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::ONE);
    solve_joints(&cone_twist, &mut b, dt);
    println!(
        "[joint] cone limit 1/2: tilt 1 about x -> cone angle {} (expect 0.5)",
        cone_angle_from_z(b[1].rotation).to_f64()
    );
    let mut b = pair(Fix128::ONE);
    b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
    solve_joints(&cone_twist, &mut b, dt);
    println!(
        "[joint] twist limit 1/4: twist 1 about z -> {} (expect 0.25)",
        twist_about_z(b[1].rotation).to_f64()
    );
}
