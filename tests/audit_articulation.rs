//! Audit S3-1 oracles for `articulation` (bookkeeping, FK, motors, ragdoll, solver flags).
//!
//! `analytic_multibody_dynamics` pins the Featherstone physics (14 closed
//! forms) and `analytic_articulation_wiring` the plumbing; this file adds
//! ragdoll geometry, motor closed forms, mass-splitting boundary and flag
//! behaviour, and the defect oracles for what those files do not ask.

use alice_physics::articulation::{build_ragdoll, ArticulatedBody, FeatherstoneSolver, LINK_ROOT};
use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::motor::PdController;
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn ball(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO))
}

// ---------------------------------------------------------------------------
// build_ragdoll
// ---------------------------------------------------------------------------

/// Structure: 12 bodies / 12 links / 11 joints, indices start at `body_start_index`
/// and are contiguous, masses are the documented 31 in total, and every body sits at the
/// pelvis position plus its fixed offset.
#[test]
fn ragdoll_structure_indices_and_mass() {
    let pelvis = v3(3.0, 5.0, -2.0);
    let (artic, bodies) = build_ragdoll(pelvis, 7);
    assert_eq!(bodies.len(), 12);
    assert_eq!(artic.link_count(), 12);
    assert_eq!(artic.joints().len(), 11);
    assert_eq!(artic.body_indices(), (7..19).collect::<Vec<_>>());
    assert_eq!(artic.links[artic.root].body_index, 7);
    assert!(!artic.fixed_base);
    let total: f64 = bodies.iter().map(|b| 1.0 / b.inv_mass.to_f64()).sum();
    assert!((total - 31.0).abs() < 1e-9, "total mass {total}");
    assert_eq!(arr(bodies[0].position), [3.0, 5.0, -2.0]);
    assert_eq!(arr(bodies[3].position), [3.0, 11.0, -2.0], "head");
    assert_eq!(
        arr(bodies[11].position),
        [4.0, 1.0, -2.0],
        "right lower leg"
    );
}

/// Hierarchy: each link's joint connects the parent's body to the child's body (indices
/// carry `body_start_index`), and `local_offset` is the child-minus-parent position
/// offset of the built pose.
#[test]
fn ragdoll_links_are_consistent_with_their_joints_and_pose() {
    let start = 4;
    let (artic, bodies) = build_ragdoll(Vec3Fix::ZERO, start);
    for (i, link) in artic.links.iter().enumerate() {
        if link.parent == LINK_ROOT {
            assert_eq!(i, artic.root);
            assert!(link.joint.is_none());
            continue;
        }
        let parent = &artic.links[link.parent];
        let (a, b) = link.joint.as_ref().unwrap().bodies();
        assert_eq!(
            a, parent.body_index,
            "link {i}: joint body_a is the parent's body"
        );
        assert_eq!(
            b, link.body_index,
            "link {i}: joint body_b is the child's body"
        );
        assert!(parent.children.contains(&i));
        let d =
            bodies[link.body_index - start].position - bodies[parent.body_index - start].position;
        assert_eq!(
            arr(d),
            arr(link.local_offset),
            "link {i}: local_offset vs pose"
        );
    }
}

/// A freshly built ragdoll satisfies its own joints: the two anchors of every joint
/// coincide in world space (otherwise the first constraint solve snaps the body).
#[test]
#[ignore = "known defect: AUD-A-S3W1-018: build_ragdoll spine / chest / head / leg joints put anchor_a at +1 (or -1) on the parent and anchor_b at the child's origin, so the anchors are 1 unit apart at build time (arms are consistent)"]
fn ragdoll_joint_anchors_coincide_at_build_time() {
    let (artic, bodies) = build_ragdoll(Vec3Fix::ZERO, 0);
    let mut worst = 0.0f64;
    let mut which = 0;
    for (i, link) in artic.links.iter().enumerate() {
        let Some(Joint::Ball(j)) = &link.joint else {
            if let Some(Joint::Hinge(j)) = &link.joint {
                let wa = bodies[j.body_a].position
                    + bodies[j.body_a].rotation.rotate_vec(j.local_anchor_a);
                let wb = bodies[j.body_b].position
                    + bodies[j.body_b].rotation.rotate_vec(j.local_anchor_b);
                let d = arr(wb - wa);
                let g = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if g > worst {
                    worst = g;
                    which = i;
                }
            }
            continue;
        };
        let wa = bodies[j.body_a].position + bodies[j.body_a].rotation.rotate_vec(j.local_anchor_a);
        let wb = bodies[j.body_b].position + bodies[j.body_b].rotation.rotate_vec(j.local_anchor_b);
        let d = arr(wb - wa);
        let g = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if g > worst {
            worst = g;
            which = i;
        }
    }
    assert!(worst < 1e-9, "largest anchor gap {worst} at link {which}");
}

// ---------------------------------------------------------------------------
// dof_count
// ---------------------------------------------------------------------------

/// `dof_count` is documented "approximate: each non-root link's joint": it counts
/// joints, so a ball (3 DOF) and a hinge (1 DOF) give 2, not 4.
#[test]
#[ignore = "known defect: AUD-A-S3W1-019: dof_count returns the number of jointed links (2 for Ball + Hinge), not the degrees of freedom (4); the name promises DOFs, the doc admits 'approximate', the existing wiring test pins the joint count"]
fn dof_count_sums_joint_degrees_of_freedom() {
    let mut a = ArticulatedBody::new(0, true);
    let l1 = a.add_link(0, 1, ball(0, 1), Vec3Fix::ZERO);
    a.add_link(
        l1,
        2,
        Joint::Hinge(HingeJoint::new(
            1,
            2,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
        Vec3Fix::ZERO,
    );
    assert_eq!(a.dof_count(), 4);
}

// ---------------------------------------------------------------------------
// forward_kinematics
// ---------------------------------------------------------------------------

/// Three levels and a branch: each child = parent (already updated) + R_parent * offset;
/// orientations and the root are untouched.
#[test]
fn forward_kinematics_chains_through_updated_parents() {
    let mut bodies: Vec<RigidBody> = (0..4)
        .map(|_| RigidBody::new(v3(100.0, 100.0, 100.0), Fix128::ONE))
        .collect();
    bodies[0].position = v3(1.0, 2.0, 3.0);
    // 90 degrees about z for the parent of link 2: (x,y) -> (-y, x)
    let h = std::f64::consts::FRAC_1_SQRT_2;
    bodies[1].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, fx(h), fx(h));
    let mut a = ArticulatedBody::new(0, true);
    let l1 = a.add_link(0, 1, ball(0, 1), v3(0.0, 1.0, 0.0));
    a.add_link(l1, 2, ball(1, 2), v3(2.0, 0.0, 0.0));
    a.add_link(0, 3, ball(0, 3), v3(0.0, 0.0, 5.0));
    let rot_before: Vec<_> = bodies.iter().map(|b| b.rotation).collect();
    a.forward_kinematics(&mut bodies);
    assert_eq!(arr(bodies[0].position), [1.0, 2.0, 3.0]);
    assert_eq!(arr(bodies[1].position), [1.0, 3.0, 3.0]);
    let p2 = arr(bodies[2].position);
    // R(90deg) (2,0,0) = (0,2,0): body 2 = (1,3,3) + (0,2,0)
    assert!(
        (p2[0] - 1.0).abs() < 1e-12 && (p2[1] - 5.0).abs() < 1e-12 && (p2[2] - 3.0).abs() < 1e-12,
        "{p2:?}"
    );
    assert_eq!(arr(bodies[3].position), [1.0, 2.0, 8.0]);
    for (b, r) in bodies.iter().zip(rot_before) {
        assert_eq!(b.rotation, r, "orientations are not touched");
    }
}

// ---------------------------------------------------------------------------
// apply_motors
// ---------------------------------------------------------------------------

fn motor_pos(kp: f64, kd: f64, max: f64, target: f64) -> PdController {
    let mut m = PdController::new(fx(kp), fx(kd), fx(max));
    m.set_position_target(fx(target));
    m
}

/// Position mode closed form: `F = kp (target - dist) + kd (0 - v_rel)`, impulse `F dt`
/// along a->b: `v_a -= F dt w_a`, `v_b += F dt w_b`.
#[test]
fn motor_position_mode_pushes_the_pair_apart_by_f_dt_inverse_mass() {
    let mut bodies = vec![
        RigidBody::new(v3(0.0, 0.0, 0.0), fx(1.0)),
        RigidBody::new(v3(3.0, 0.0, 0.0), fx(2.0)),
    ];
    let mut a = ArticulatedBody::new(0, false);
    a.add_link(0, 1, ball(0, 1), v3(3.0, 0.0, 0.0));
    a.set_motor(1, motor_pos(2.0, 0.0, 100.0, 5.0));
    a.apply_motors(&mut bodies, fx(0.25));
    // F = 2*(5-3) = 4 ; impulse 1.0
    assert!(
        (bodies[0].velocity.x.to_f64() + 1.0).abs() < 1e-12,
        "{}",
        bodies[0].velocity.x.to_f64()
    );
    assert!((bodies[1].velocity.x.to_f64() - 0.5).abs() < 1e-12);
    // momentum conserved
    let p = bodies[0].velocity.x.to_f64() / bodies[0].inv_mass.to_f64()
        + bodies[1].velocity.x.to_f64() / bodies[1].inv_mass.to_f64();
    assert!(p.abs() < 1e-12);
}

/// Velocity damping term and clamp: `F = clamp(kp err + kd (v_t - v_rel), +-max)`.
#[test]
fn motor_clamps_and_damps_along_the_line_between_bodies() {
    let mut bodies = vec![
        RigidBody::new(v3(0.0, 0.0, 0.0), fx(1.0)),
        RigidBody::new(v3(0.0, 4.0, 0.0), fx(1.0)),
    ];
    bodies[1].velocity = v3(0.0, 2.0, 0.0);
    let mut a = ArticulatedBody::new(0, false);
    a.add_link(0, 1, ball(0, 1), Vec3Fix::ZERO);
    // error = 4 - 4 = 0 ; vel term kd*(0 - 2) = -6 ; max 5 -> -5
    a.set_motor(1, motor_pos(10.0, 3.0, 5.0, 4.0));
    a.apply_motors(&mut bodies, fx(0.5));
    // impulse = -5 * 0.5 along +y: a gets +2.5, b gets -2.5
    assert!((bodies[0].velocity.y.to_f64() - 2.5).abs() < 1e-12);
    assert!((bodies[1].velocity.y.to_f64() - (2.0 - 2.5)).abs() < 1e-12);
}

/// Mode Off, zero force, coincident bodies and a static partner are all left alone.
#[test]
fn motor_off_zero_force_and_static_partner() {
    let mk = |gap: f64| {
        vec![
            RigidBody::new(v3(0.0, 0.0, 0.0), fx(1.0)),
            RigidBody::new(v3(gap, 0.0, 0.0), fx(1.0)),
        ]
    };
    let mut a = ArticulatedBody::new(0, false);
    a.add_link(0, 1, ball(0, 1), Vec3Fix::ZERO);
    // off
    a.set_motor(1, PdController::new(fx(10.0), fx(1.0), fx(100.0)));
    let mut b = mk(3.0);
    a.apply_motors(&mut b, fx(0.25));
    assert_eq!(arr(b[1].velocity), [0.0; 3]);
    // zero force (error 0, no velocity)
    a.set_motor(1, motor_pos(10.0, 1.0, 100.0, 3.0));
    let mut b = mk(3.0);
    a.apply_motors(&mut b, fx(0.25));
    assert_eq!(arr(b[0].velocity), [0.0; 3]);
    // coincident bodies: no direction
    let mut b = mk(0.0);
    a.apply_motors(&mut b, fx(0.25));
    assert_eq!(arr(b[1].velocity), [0.0; 3]);
    // static a: only b moves
    a.set_motor(1, motor_pos(2.0, 0.0, 100.0, 5.0));
    let mut b = mk(3.0);
    b[0] = RigidBody::new_static(v3(0.0, 0.0, 0.0));
    a.apply_motors(&mut b, fx(0.25));
    assert_eq!(arr(b[0].velocity), [0.0; 3]);
    assert!((b[1].velocity.x.to_f64() - 1.0).abs() < 1e-12);
}

/// A motor on a hinge link should drive the hinge angle: a velocity-mode motor with a
/// target of 1 rad/s makes the child spin about the hinge axis.
#[test]
#[ignore = "known defect: AUD-A-S3W1-017: apply_motors uses the distance between the two body centres as the joint coordinate for every joint type; a hinge/ball motor changes linear velocity along the bone and never the angular velocity (angular_velocity stays 0 here)"]
fn hinge_motor_drives_the_hinge_angle() {
    let mut bodies = vec![
        RigidBody::new(v3(0.0, 0.0, 0.0), fx(1.0)),
        RigidBody::new(v3(1.0, 0.0, 0.0), fx(1.0)),
    ];
    let mut a = ArticulatedBody::new(0, false);
    a.add_link(
        0,
        1,
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            v3(1.0, 0.0, 0.0),
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
        v3(1.0, 0.0, 0.0),
    );
    let mut m = PdController::new(fx(10.0), fx(0.0), fx(100.0));
    m.set_velocity_target(fx(1.0));
    a.set_motor(1, m);
    a.apply_motors(&mut bodies, fx(0.25));
    let w = bodies[1].angular_velocity.z.to_f64() - bodies[0].angular_velocity.z.to_f64();
    assert!(
        w.abs() > 1e-6,
        "relative angular velocity about the hinge axis {w}"
    );
}

// ---------------------------------------------------------------------------
// FeatherstoneSolver flags
// ---------------------------------------------------------------------------

/// A single free root integrates semi-implicitly: `v = g dt`, `x += v dt`; with
/// `fixed_base = true` or a static body it is held.
#[test]
fn single_root_free_fall_fixed_base_and_static_base() {
    let g = v3(0.0, -10.0, 0.0);
    let dt = fx(0.125);
    let mut bodies = vec![RigidBody::new(v3(0.0, 5.0, 0.0), fx(2.0))];
    FeatherstoneSolver::new().solve(&ArticulatedBody::new(0, false), &mut bodies, g, dt);
    assert!((bodies[0].velocity.y.to_f64() + 1.25).abs() < 1e-12);
    assert!((bodies[0].position.y.to_f64() - (5.0 - 1.25 * 0.125)).abs() < 1e-12);
    let mut held = vec![RigidBody::new(v3(0.0, 5.0, 0.0), fx(2.0))];
    FeatherstoneSolver::new().solve(&ArticulatedBody::new(0, true), &mut held, g, dt);
    assert_eq!(arr(held[0].position), [0.0, 5.0, 0.0]);
    assert_eq!(arr(held[0].velocity), [0.0; 3]);
    let mut st = vec![RigidBody::new_static(v3(0.0, 5.0, 0.0))];
    FeatherstoneSolver::new().solve(&ArticulatedBody::new(0, false), &mut st, g, dt);
    assert_eq!(arr(st[0].position), [0.0, 5.0, 0.0]);
}

/// `solve_with_mass_splitting` splits exactly when `heavy/light >= threshold` (>=, not >).
#[test]
fn mass_splitting_triggers_at_the_ratio_boundary() {
    let g = v3(0.0, -10.0, 0.0);
    let dt = fx(0.25);
    let scene = || {
        let b = vec![
            RigidBody::new(v3(0.0, 0.0, 0.0), fx(1.0)),
            RigidBody::new(v3(0.0, 1.0, 0.0), fx(4.0)),
        ];
        let mut a = ArticulatedBody::new(0, false);
        a.add_link(0, 1, ball(0, 1), v3(0.0, 1.0, 0.0));
        (a, b)
    };
    let (a, mut full) = scene();
    FeatherstoneSolver::new().solve(&a, &mut full, g, dt);
    let (a, mut half) = scene();
    let mut s = FeatherstoneSolver::new();
    s.solve(&a, &mut half, g, dt.half());
    s.solve(&a, &mut half, g, dt.half());
    // ratio exactly 4: threshold 4 splits, 4.5 does not, 0 / negative never split
    let run = |t: f64| {
        let (a, mut b) = scene();
        FeatherstoneSolver::new().solve_with_mass_splitting(&a, &mut b, g, dt, fx(t));
        b
    };
    for (t, expect_split) in [
        (4.0, true),
        (3.0, true),
        (4.5, false),
        (0.0, false),
        (-1.0, false),
    ] {
        let got = run(t);
        let want = if expect_split { &half } else { &full };
        for k in 0..2 {
            assert_eq!(got[k].position, want[k].position, "threshold {t} body {k}");
            assert_eq!(got[k].velocity, want[k].velocity, "threshold {t} body {k}");
        }
    }
    assert_ne!(
        full[1].position, half[1].position,
        "scene must distinguish the two paths"
    );
}
