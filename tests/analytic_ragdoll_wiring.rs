//! Oracles for `ragdoll`: presets, T-pose layout and the joint table.
//!
//! The layout is the module's documented construction, recomputed here in
//! `f64` from the segment-length fractions of the standing height (feet at
//! the bottom, T-pose facing +Z). Mass per bone is `mass_kg * fraction`.
//!
//! NOT pinned (open Backlog entries, `RagdollBuilder`): the fractions summing
//! to 1 (they sum to 0.958 today) and the joints keeping their bones apart
//! (anchors are body centres today).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::joint::Joint;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::ragdoll::{Bone, RagdollBuilder, RagdollHandle, RagdollProportions};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};

fn f(x: Fix128) -> f64 {
    x.to_f64()
}

#[test]
fn presets_have_the_documented_height_mass_and_shared_fractions() {
    let m = RagdollProportions::human_male();
    let w = RagdollProportions::human_female();
    let c = RagdollProportions::child();
    assert!((f(m.height_m) - 1.75).abs() < 1e-9 && (f(m.mass_kg) - 75.0).abs() < 1e-9);
    assert!((f(w.height_m) - 1.62).abs() < 1e-9 && (f(w.mass_kg) - 62.0).abs() < 1e-9);
    assert!((f(c.height_m) - 1.30).abs() < 1e-9 && (f(c.mass_kg) - 30.0).abs() < 1e-9);
    // Fractions (Winter table in the module doc) are shared by all presets.
    let fr = |p: &RagdollProportions| {
        [
            p.mass_pelvis,
            p.mass_torso,
            p.mass_head,
            p.mass_upper_arm,
            p.mass_forearm,
            p.mass_hand,
            p.mass_thigh,
            p.mass_shin,
            p.mass_foot,
        ]
        .map(f)
    };
    let want = [
        0.100, 0.355, 0.081, 0.028, 0.016, 0.006, 0.100, 0.0465, 0.0145,
    ];
    for (g, e) in fr(&m).iter().zip(want) {
        assert!((g - e).abs() < 1e-9, "{g} vs {e}");
    }
    assert_eq!(fr(&w), fr(&m));
    assert_eq!(fr(&c), fr(&m));
}

fn build(p: RagdollProportions, at: Vec3Fix) -> (PhysicsWorld, RagdollHandle) {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let h = RagdollBuilder::build(&mut w, p, at);
    (w, h)
}

fn check_layout(p: RagdollProportions, mass_kg: f64, h_m: f64) {
    let (px, py, pz) = (1.0, 2.0, 3.0);
    let (w, hd) = build(p, Vec3Fix::from_int(1, 2, 3));
    let l = |x: f64| x * h_m;
    let y_torso = py + (l(0.08) + l(0.30)) / 2.0;
    let y_head = y_torso + (l(0.30) + l(0.13)) / 2.0;
    let y_sh = y_torso + l(0.30) * 0.35;
    let y_hip = py - l(0.08) / 2.0;
    let want = [
        (Bone::Pelvis, px, py, 0.100),
        (Bone::Torso, px, y_torso, 0.355),
        (Bone::Head, px, y_head, 0.081),
        (
            Bone::LeftUpperArm,
            px - l(0.12),
            y_sh - l(0.18) / 2.0,
            0.028,
        ),
        (
            Bone::LeftForearm,
            px - l(0.12),
            y_sh - l(0.18) - l(0.15) / 2.0,
            0.016,
        ),
        (
            Bone::LeftHand,
            px - l(0.12),
            y_sh - l(0.18) - l(0.15) - l(0.11) / 2.0,
            0.006,
        ),
        (
            Bone::RightUpperArm,
            px + l(0.12),
            y_sh - l(0.18) / 2.0,
            0.028,
        ),
        (
            Bone::RightForearm,
            px + l(0.12),
            y_sh - l(0.18) - l(0.15) / 2.0,
            0.016,
        ),
        (
            Bone::RightHand,
            px + l(0.12),
            y_sh - l(0.18) - l(0.15) - l(0.11) / 2.0,
            0.006,
        ),
        (Bone::LeftThigh, px - l(0.07), y_hip - l(0.24) / 2.0, 0.100),
        (
            Bone::LeftShin,
            px - l(0.07),
            y_hip - l(0.24) - l(0.23) / 2.0,
            0.0465,
        ),
        (
            Bone::LeftFoot,
            px - l(0.07),
            y_hip - l(0.24) - l(0.23) - l(0.05) / 2.0,
            0.0145,
        ),
        (Bone::RightThigh, px + l(0.07), y_hip - l(0.24) / 2.0, 0.100),
        (
            Bone::RightShin,
            px + l(0.07),
            y_hip - l(0.24) - l(0.23) / 2.0,
            0.0465,
        ),
        (
            Bone::RightFoot,
            px + l(0.07),
            y_hip - l(0.24) - l(0.23) - l(0.05) / 2.0,
            0.0145,
        ),
    ];
    for (bone, x, y, frac) in want {
        let b = &w.bodies[hd.body(bone)];
        assert!((f(b.position.x) - x).abs() < 1e-6, "{bone:?} x");
        assert!(
            (f(b.position.y) - y).abs() < 1e-6,
            "{bone:?} y {} vs {y}",
            f(b.position.y)
        );
        assert!((f(b.position.z) - pz).abs() < 1e-9, "{bone:?} z");
        let mass = 1.0 / f(b.inv_mass);
        assert!(
            (mass - mass_kg * frac).abs() < 1e-6 * mass_kg,
            "{bone:?} mass {mass} vs {}",
            mass_kg * frac
        );
        assert!(b.velocity == Vec3Fix::ZERO);
    }
}

#[test]
fn t_pose_layout_and_masses_for_every_preset() {
    check_layout(RagdollProportions::human_male(), 75.0, 1.75);
    check_layout(RagdollProportions::human_female(), 62.0, 1.62);
    check_layout(RagdollProportions::child(), 30.0, 1.30);
}

#[test]
fn preset_height_scales_the_layout_linearly() {
    // Vertical span head-centre minus foot-centre is proportional to height:
    // y_head - y_foot = h * (0.19 + 0.215 + 0.04 + 0.24 + 0.23 + 0.025) = 0.94 h
    for (p, h) in [
        (RagdollProportions::human_male(), 1.75),
        (RagdollProportions::human_female(), 1.62),
        (RagdollProportions::child(), 1.30),
    ] {
        let (w, hd) = build(p, Vec3Fix::ZERO);
        let span = f(w.bodies[hd.body(Bone::Head)].position.y)
            - f(w.bodies[hd.body(Bone::LeftFoot)].position.y);
        assert!((span - 0.94 * h).abs() < 1e-6, "{span} vs {}", 0.94 * h);
    }
}

#[test]
fn joint_table_connects_each_bone_to_its_parent() {
    let (w, hd) = build(RagdollProportions::child(), Vec3Fix::ZERO);
    use Bone::*;
    // joints[i] attaches bone i + 1 to its parent; (is_hinge, parent, child).
    let table = [
        (false, Pelvis, Torso),
        (false, Torso, Head),
        (false, Torso, LeftUpperArm),
        (true, LeftUpperArm, LeftForearm),
        (false, LeftForearm, LeftHand),
        (false, Torso, RightUpperArm),
        (true, RightUpperArm, RightForearm),
        (false, RightForearm, RightHand),
        (false, Pelvis, LeftThigh),
        (true, LeftThigh, LeftShin),
        (false, LeftShin, LeftFoot),
        (false, Pelvis, RightThigh),
        (true, RightThigh, RightShin),
        (false, RightShin, RightFoot),
    ];
    assert_eq!(w.joints.len(), 14);
    assert_eq!(w.bodies.len(), 15);
    for (i, (hinge, a, b)) in table.iter().enumerate() {
        match (&w.joints[hd.joints[i]], hinge) {
            (Joint::Ball(j), false) => {
                assert_eq!(
                    (j.body_a, j.body_b),
                    (hd.body(*a), hd.body(*b)),
                    "joint {i}"
                );
            }
            (Joint::Hinge(j), true) => {
                assert_eq!(
                    (j.body_a, j.body_b),
                    (hd.body(*a), hd.body(*b)),
                    "joint {i}"
                );
                // Hinge axis is X on both bodies (elbows / knees fold about the lateral axis).
                assert_eq!(j.local_axis_a, Vec3Fix::from_int(1, 0, 0));
                assert_eq!(j.local_axis_b, Vec3Fix::from_int(1, 0, 0));
            }
            (other, _) => panic!("joint {i}: unexpected {other:?}"),
        }
    }
    // Bone indices follow the enum order and are one-to-one with the world's bodies.
    for (k, bone) in [
        Pelvis,
        Torso,
        Head,
        LeftUpperArm,
        LeftForearm,
        LeftHand,
        RightUpperArm,
        RightForearm,
        RightHand,
        LeftThigh,
        LeftShin,
        LeftFoot,
        RightThigh,
        RightShin,
        RightFoot,
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(bone.index(), k);
        assert_eq!(hd.body(*bone), hd.bones[k]);
    }
}

#[test]
fn building_into_a_non_empty_world_offsets_the_indices() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let first = RagdollBuilder::build(&mut w, RagdollProportions::human_male(), Vec3Fix::ZERO);
    let second = RagdollBuilder::build(
        &mut w,
        RagdollProportions::child(),
        Vec3Fix::from_int(5, 0, 0),
    );
    assert_eq!(second.bones[0], first.bones[14] + 1);
    assert_eq!(second.joints[0], first.joints[13] + 1);
    match &w.joints[second.joints[0]] {
        Joint::Ball(j) => assert_eq!((j.body_a, j.body_b), (second.bones[0], second.bones[1])),
        other => panic!("{other:?}"),
    }
    assert_eq!(w.bodies.len(), 30);
    assert_eq!(w.joints.len(), 28);
}

#[test]
fn degenerate_proportions_do_not_panic() {
    let mut p = RagdollProportions::human_male();
    p.mass_kg = Fix128::ZERO;
    let (w, hd) = build(p, Vec3Fix::ZERO);
    // Zero total mass: every bone is immovable (inv_mass 0), still 15 bodies.
    assert_eq!(w.bodies.len(), 15);
    assert!(hd.bones.iter().all(|&b| w.bodies[b].inv_mass.is_zero()));
    let mut p = RagdollProportions::human_male();
    p.height_m = Fix128::ZERO;
    let (w, hd) = build(p, Vec3Fix::from_int(1, 2, 3));
    // Zero height: every segment length is 0, all bones sit at the pelvis position.
    for &b in &hd.bones {
        assert_eq!(w.bodies[b].position, Vec3Fix::from_int(1, 2, 3));
    }
}
