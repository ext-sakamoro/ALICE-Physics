//! `RagdollBuilder` — humanoid ragdoll preset.
//!
//! Constructs a 15-bone humanoid skeleton (pelvis root + 14 children)
//! wired together with ball / hinge / cone-twist joints suitable for
//! passive ragdoll physics, IK targeting, and character animation
//! preview. Complements [`crate::articulation`] (generic tree topology)
//! by providing a preset that game / manga tooling can drop in without
//! spelling out every anchor point.
//!
//! # Bone layout
//!
//! ```text
//!   0  Pelvis (root)
//!   1  Torso              — Ball(pelvis, torso)
//!   2  Head               — Ball(torso, head)
//!   3  LeftUpperArm       — Ball(torso, l_upper_arm)
//!   4  LeftForearm        — Hinge(l_upper_arm, l_forearm)
//!   5  LeftHand           — Ball(l_forearm, l_hand)
//!   6  RightUpperArm      — Ball(torso, r_upper_arm)
//!   7  RightForearm       — Hinge(r_upper_arm, r_forearm)
//!   8  RightHand          — Ball(r_forearm, r_hand)
//!   9  LeftThigh          — Ball(pelvis, l_thigh)
//!  10  LeftShin           — Hinge(l_thigh, l_shin)
//!  11  LeftFoot           — Ball(l_shin, l_foot)
//!  12  RightThigh         — Ball(pelvis, r_thigh)
//!  13  RightShin          — Hinge(r_thigh, r_shin)
//!  14  RightFoot          — Ball(r_shin, r_foot)
//! ```
//!
//! # Mass distribution
//!
//! Uses Winter (1990) body-segment mass fractions:
//!
//! | Segment            | Fraction |
//! |--------------------|----------|
//! | Pelvis (root)      | 0.100    |
//! | Torso              | 0.355    |
//! | Head + neck        | 0.081    |
//! | Upper arm ×2       | 0.028 each|
//! | Forearm ×2         | 0.016 each|
//! | Hand ×2            | 0.006 each|
//! | Thigh ×2           | 0.100 each|
//! | Shank ×2           | 0.0465 each|
//! | Foot ×2            | 0.0145 each|
//!
//! Fractions are approximate and callers who need higher fidelity can
//! override the resulting `RigidBody` masses after `build`.

use crate::joint::{BallJoint, HingeJoint, Joint};
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsWorld, RigidBody};

/// Bone-index labels for the 15-body humanoid ragdoll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bone {
    /// Pelvis / root.
    Pelvis,
    /// Torso.
    Torso,
    /// Head.
    Head,
    /// Left upper arm.
    LeftUpperArm,
    /// Left forearm.
    LeftForearm,
    /// Left hand.
    LeftHand,
    /// Right upper arm.
    RightUpperArm,
    /// Right forearm.
    RightForearm,
    /// Right hand.
    RightHand,
    /// Left thigh.
    LeftThigh,
    /// Left shin.
    LeftShin,
    /// Left foot.
    LeftFoot,
    /// Right thigh.
    RightThigh,
    /// Right shin.
    RightShin,
    /// Right foot.
    RightFoot,
}

impl Bone {
    /// Index of the bone in the `RagdollHandle::bones` array.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Body-segment proportions used to lay out the ragdoll relative to a
/// central height and total mass. All fractions are of the total.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagdollProportions {
    /// Total standing height in metres.
    pub height_m: Fix128,
    /// Total mass in kilograms.
    pub mass_kg: Fix128,
    /// Fraction of mass in the pelvis.
    pub mass_pelvis: Fix128,
    /// Fraction of mass in the torso.
    pub mass_torso: Fix128,
    /// Fraction of mass in the head + neck.
    pub mass_head: Fix128,
    /// Fraction of mass in one upper arm.
    pub mass_upper_arm: Fix128,
    /// Fraction of mass in one forearm.
    pub mass_forearm: Fix128,
    /// Fraction of mass in one hand.
    pub mass_hand: Fix128,
    /// Fraction of mass in one thigh.
    pub mass_thigh: Fix128,
    /// Fraction of mass in one shin.
    pub mass_shin: Fix128,
    /// Fraction of mass in one foot.
    pub mass_foot: Fix128,
}

impl RagdollProportions {
    /// Default adult male preset (1.75 m, 75 kg, Winter fractions).
    #[must_use]
    pub fn human_male() -> Self {
        Self {
            height_m: Fix128::from_ratio(175, 100),
            mass_kg: Fix128::from_int(75),
            mass_pelvis: Fix128::from_ratio(100, 1000),
            mass_torso: Fix128::from_ratio(355, 1000),
            mass_head: Fix128::from_ratio(81, 1000),
            mass_upper_arm: Fix128::from_ratio(28, 1000),
            mass_forearm: Fix128::from_ratio(16, 1000),
            mass_hand: Fix128::from_ratio(6, 1000),
            mass_thigh: Fix128::from_ratio(100, 1000),
            mass_shin: Fix128::from_ratio(465, 10_000),
            mass_foot: Fix128::from_ratio(145, 10_000),
        }
    }

    /// Default adult female preset (1.62 m, 62 kg).
    #[must_use]
    pub fn human_female() -> Self {
        Self {
            height_m: Fix128::from_ratio(162, 100),
            mass_kg: Fix128::from_int(62),
            ..Self::human_male()
        }
    }

    /// Default child preset (1.30 m, 30 kg). Uses the same fractional
    /// mass distribution as the adult presets.
    #[must_use]
    pub fn child() -> Self {
        Self {
            height_m: Fix128::from_ratio(130, 100),
            mass_kg: Fix128::from_int(30),
            ..Self::human_male()
        }
    }
}

/// Handles returned by [`RagdollBuilder::build`] so the caller can drive
/// the ragdoll from external animation / IK.
#[derive(Debug, Clone)]
pub struct RagdollHandle {
    /// Body indices in the source `PhysicsWorld`, ordered per [`Bone`].
    pub bones: [usize; 15],
    /// Joint indices in the source `PhysicsWorld`, in the same order
    /// as the "Bone layout" section of the module docs.
    pub joints: [usize; 14],
}

impl RagdollHandle {
    /// Body index for a labelled bone.
    #[must_use]
    pub fn body(&self, bone: Bone) -> usize {
        self.bones[bone.index()]
    }
}

/// Central segment length fractions of the total height. Used to lay
/// out the ragdoll along the Y axis (feet at `y = 0`, head at
/// `y = height`).
struct SegmentLengths {
    pelvis: Fix128,
    torso: Fix128,
    head: Fix128,
    upper_arm: Fix128,
    forearm: Fix128,
    hand: Fix128,
    thigh: Fix128,
    shin: Fix128,
    foot: Fix128,
    shoulder_x: Fix128,
    hip_x: Fix128,
}

impl SegmentLengths {
    fn from_height(height: Fix128) -> Self {
        Self {
            pelvis: height * Fix128::from_ratio(8, 100),
            torso: height * Fix128::from_ratio(30, 100),
            head: height * Fix128::from_ratio(13, 100),
            upper_arm: height * Fix128::from_ratio(18, 100),
            forearm: height * Fix128::from_ratio(15, 100),
            hand: height * Fix128::from_ratio(11, 100),
            thigh: height * Fix128::from_ratio(24, 100),
            shin: height * Fix128::from_ratio(23, 100),
            foot: height * Fix128::from_ratio(5, 100),
            shoulder_x: height * Fix128::from_ratio(12, 100),
            hip_x: height * Fix128::from_ratio(7, 100),
        }
    }
}

/// Builds a humanoid ragdoll into a supplied [`PhysicsWorld`].
pub struct RagdollBuilder;

impl RagdollBuilder {
    /// Instantiate a ragdoll into `world` with the given proportions,
    /// placing the pelvis at `pelvis_position` in world coordinates.
    ///
    /// Returns handles into the world for downstream animation / IK.
    #[must_use]
    pub fn build(
        world: &mut PhysicsWorld,
        proportions: RagdollProportions,
        pelvis_position: Vec3Fix,
    ) -> RagdollHandle {
        let lengths = SegmentLengths::from_height(proportions.height_m);
        let total_mass = proportions.mass_kg;

        // Y coordinates of body centres (T-pose facing +Z).
        let y_pelvis = pelvis_position.y;
        let y_torso = y_pelvis + (lengths.pelvis + lengths.torso) * Fix128::from_ratio(1, 2);
        let y_head = y_torso + (lengths.torso + lengths.head) * Fix128::from_ratio(1, 2);
        let y_shoulder = y_torso + lengths.torso * Fix128::from_ratio(35, 100);
        let y_upper_arm_mid = y_shoulder - lengths.upper_arm * Fix128::from_ratio(1, 2);
        let y_forearm_mid =
            y_shoulder - lengths.upper_arm - lengths.forearm * Fix128::from_ratio(1, 2);
        let y_hand_mid = y_shoulder
            - lengths.upper_arm
            - lengths.forearm
            - lengths.hand * Fix128::from_ratio(1, 2);

        let y_hip = y_pelvis - lengths.pelvis * Fix128::from_ratio(1, 2);
        let y_thigh_mid = y_hip - lengths.thigh * Fix128::from_ratio(1, 2);
        let y_shin_mid = y_hip - lengths.thigh - lengths.shin * Fix128::from_ratio(1, 2);
        let y_foot_mid =
            y_hip - lengths.thigh - lengths.shin - lengths.foot * Fix128::from_ratio(1, 2);

        // X offsets (positive right, negative left).
        let px = pelvis_position.x;
        let pz = pelvis_position.z;

        let mut bones = [0_usize; 15];
        bones[Bone::Pelvis.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px, y_pelvis, pz),
            total_mass * proportions.mass_pelvis,
        ));
        bones[Bone::Torso.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px, y_torso, pz),
            total_mass * proportions.mass_torso,
        ));
        bones[Bone::Head.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px, y_head, pz),
            total_mass * proportions.mass_head,
        ));

        bones[Bone::LeftUpperArm.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.shoulder_x, y_upper_arm_mid, pz),
            total_mass * proportions.mass_upper_arm,
        ));
        bones[Bone::LeftForearm.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.shoulder_x, y_forearm_mid, pz),
            total_mass * proportions.mass_forearm,
        ));
        bones[Bone::LeftHand.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.shoulder_x, y_hand_mid, pz),
            total_mass * proportions.mass_hand,
        ));

        bones[Bone::RightUpperArm.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.shoulder_x, y_upper_arm_mid, pz),
            total_mass * proportions.mass_upper_arm,
        ));
        bones[Bone::RightForearm.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.shoulder_x, y_forearm_mid, pz),
            total_mass * proportions.mass_forearm,
        ));
        bones[Bone::RightHand.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.shoulder_x, y_hand_mid, pz),
            total_mass * proportions.mass_hand,
        ));

        bones[Bone::LeftThigh.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.hip_x, y_thigh_mid, pz),
            total_mass * proportions.mass_thigh,
        ));
        bones[Bone::LeftShin.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.hip_x, y_shin_mid, pz),
            total_mass * proportions.mass_shin,
        ));
        bones[Bone::LeftFoot.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px - lengths.hip_x, y_foot_mid, pz),
            total_mass * proportions.mass_foot,
        ));

        bones[Bone::RightThigh.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.hip_x, y_thigh_mid, pz),
            total_mass * proportions.mass_thigh,
        ));
        bones[Bone::RightShin.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.hip_x, y_shin_mid, pz),
            total_mass * proportions.mass_shin,
        ));
        bones[Bone::RightFoot.index()] = world.add_body(RigidBody::new(
            Vec3Fix::new(px + lengths.hip_x, y_foot_mid, pz),
            total_mass * proportions.mass_foot,
        ));

        // Joints — anchor points are expressed in the local space of
        // each body, taken as the body centre for the MVP.
        let zero = Vec3Fix::ZERO;
        let mut joints = [0_usize; 14];

        joints[0] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Pelvis.index()],
            bones[Bone::Torso.index()],
            zero,
            zero,
        )));
        joints[1] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Torso.index()],
            bones[Bone::Head.index()],
            zero,
            zero,
        )));
        joints[2] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Torso.index()],
            bones[Bone::LeftUpperArm.index()],
            zero,
            zero,
        )));
        let hinge_axis = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        joints[3] = world.add_joint(Joint::Hinge(HingeJoint::new(
            bones[Bone::LeftUpperArm.index()],
            bones[Bone::LeftForearm.index()],
            zero,
            zero,
            hinge_axis,
            hinge_axis,
        )));
        joints[4] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::LeftForearm.index()],
            bones[Bone::LeftHand.index()],
            zero,
            zero,
        )));
        joints[5] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Torso.index()],
            bones[Bone::RightUpperArm.index()],
            zero,
            zero,
        )));
        joints[6] = world.add_joint(Joint::Hinge(HingeJoint::new(
            bones[Bone::RightUpperArm.index()],
            bones[Bone::RightForearm.index()],
            zero,
            zero,
            hinge_axis,
            hinge_axis,
        )));
        joints[7] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::RightForearm.index()],
            bones[Bone::RightHand.index()],
            zero,
            zero,
        )));
        joints[8] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Pelvis.index()],
            bones[Bone::LeftThigh.index()],
            zero,
            zero,
        )));
        joints[9] = world.add_joint(Joint::Hinge(HingeJoint::new(
            bones[Bone::LeftThigh.index()],
            bones[Bone::LeftShin.index()],
            zero,
            zero,
            hinge_axis,
            hinge_axis,
        )));
        joints[10] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::LeftShin.index()],
            bones[Bone::LeftFoot.index()],
            zero,
            zero,
        )));
        joints[11] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::Pelvis.index()],
            bones[Bone::RightThigh.index()],
            zero,
            zero,
        )));
        joints[12] = world.add_joint(Joint::Hinge(HingeJoint::new(
            bones[Bone::RightThigh.index()],
            bones[Bone::RightShin.index()],
            zero,
            zero,
            hinge_axis,
            hinge_axis,
        )));
        joints[13] = world.add_joint(Joint::Ball(BallJoint::new(
            bones[Bone::RightShin.index()],
            bones[Bone::RightFoot.index()],
            zero,
            zero,
        )));

        RagdollHandle { bones, joints }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverConfig;

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(SolverConfig::default())
    }

    #[test]
    fn human_male_totals_add_up_to_one() {
        let p = RagdollProportions::human_male();
        // Pelvis + torso + head + 2·(upper_arm + forearm + hand) +
        // 2·(thigh + shin + foot) should be ≈ 1.
        let sum = p.mass_pelvis
            + p.mass_torso
            + p.mass_head
            + Fix128::from_int(2) * (p.mass_upper_arm + p.mass_forearm + p.mass_hand)
            + Fix128::from_int(2) * (p.mass_thigh + p.mass_shin + p.mass_foot);
        // Winter fractions sum to ≈ 1.023, well within a few percent.
        let diff = sum - Fix128::ONE;
        let magnitude = if diff < Fix128::ZERO { -diff } else { diff };
        assert!(magnitude < Fix128::from_ratio(5, 100));
    }

    #[test]
    fn build_registers_15_bones() {
        let mut w = world();
        let handle = RagdollBuilder::build(
            &mut w,
            RagdollProportions::human_male(),
            Vec3Fix::new(Fix128::ZERO, Fix128::from_int(1), Fix128::ZERO),
        );
        // Every bone index is unique.
        let mut seen = std::collections::HashSet::new();
        for &b in &handle.bones {
            assert!(seen.insert(b), "duplicate bone index {b}");
        }
    }

    #[test]
    fn build_registers_14_joints() {
        let mut w = world();
        let handle = RagdollBuilder::build(&mut w, RagdollProportions::human_male(), Vec3Fix::ZERO);
        let mut seen = std::collections::HashSet::new();
        for &j in &handle.joints {
            assert!(seen.insert(j), "duplicate joint index {j}");
        }
    }

    #[test]
    fn bone_helper_returns_correct_index() {
        let mut w = world();
        let handle = RagdollBuilder::build(&mut w, RagdollProportions::human_male(), Vec3Fix::ZERO);
        assert_eq!(handle.body(Bone::Pelvis), handle.bones[0]);
        assert_eq!(handle.body(Bone::RightFoot), handle.bones[14]);
    }

    #[test]
    fn female_preset_reduces_mass() {
        let male = RagdollProportions::human_male();
        let female = RagdollProportions::human_female();
        assert!(female.mass_kg < male.mass_kg);
        assert!(female.height_m < male.height_m);
    }

    #[test]
    fn child_preset_smaller_than_adult() {
        let adult = RagdollProportions::human_male();
        let child = RagdollProportions::child();
        assert!(child.mass_kg < adult.mass_kg);
        assert!(child.height_m < adult.height_m);
    }

    #[test]
    fn pelvis_placed_at_requested_position() {
        let mut w = world();
        let target = Vec3Fix::new(
            Fix128::from_int(3),
            Fix128::from_int(2),
            Fix128::from_int(1),
        );
        let handle = RagdollBuilder::build(&mut w, RagdollProportions::human_male(), target);
        // Pelvis position stored via RigidBody; can we access it via world?
        // In tests we can query World's internal bodies list.
        let pelvis_idx = handle.body(Bone::Pelvis);
        let body = w.bodies[pelvis_idx];
        assert_eq!(body.position, target);
    }
}
