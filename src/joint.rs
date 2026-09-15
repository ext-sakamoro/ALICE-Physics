//! Joint System for Rigid Body Connections
//!
//! XPBD-based joint constraints with angular limits and compliance.
//!
//! # Joint Types
//!
//! - **`BallJoint`**: 3-DOF rotation (shoulder, hip)
//! - **`HingeJoint`**: 1-DOF rotation with angle limits (knee, door)
//! - **`FixedJoint`**: 0-DOF (weld)
//! - **`SliderJoint`**: 1-DOF translation along an axis (piston)
//! - **`SpringJoint`**: Distance spring with damping
//! - **`D6Joint`**: 6-DOF configurable joint with per-axis locking
//! - **`ConeTwistJoint`**: Cone-twist constraint for ragdoll shoulders/hips

use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Joint type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    /// Ball-and-socket (3 rotational DOF)
    Ball,
    /// Hinge (1 rotational DOF)
    Hinge,
    /// Fixed / weld (0 DOF)
    Fixed,
    /// Slider / prismatic (1 translational DOF)
    Slider,
    /// Spring with damping
    Spring,
    /// 6-DOF configurable joint
    D6,
    /// Cone-twist (ragdoll)
    ConeTwist,
}

/// Ball-and-socket joint (3 rotational DOF)
///
/// Constrains two anchor points to coincide while allowing free rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BallJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Compliance (inverse stiffness, 0 = rigid)
    pub compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl BallJoint {
    /// Create a new ball joint
    #[inline]
    #[must_use]
    pub const fn new(body_a: usize, body_b: usize, anchor_a: Vec3Fix, anchor_b: Vec3Fix) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set compliance (inverse stiffness)
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Hinge joint (1 rotational DOF around an axis, with optional angle limits)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HingeJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Hinge axis in body A's local space
    pub local_axis_a: Vec3Fix,
    /// Hinge axis in body B's local space
    pub local_axis_b: Vec3Fix,
    /// Minimum angle (radians, None = no limit)
    pub angle_min: Option<Fix128>,
    /// Maximum angle (radians, None = no limit)
    pub angle_max: Option<Fix128>,
    /// Positional compliance
    pub compliance: Fix128,
    /// Angular compliance
    pub angular_compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl HingeJoint {
    /// Create a new hinge joint
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        axis_a: Vec3Fix,
        axis_b: Vec3Fix,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            local_axis_a: axis_a,
            local_axis_b: axis_b,
            angle_min: None,
            angle_max: None,
            compliance: Fix128::ZERO,
            angular_compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set angular limits (radians)
    #[must_use]
    pub const fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        self.angle_min = Some(min);
        self.angle_max = Some(max);
        self
    }

    /// Set positional compliance (inverse stiffness)
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Fixed joint (0 DOF, weld two bodies)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Relative rotation at the time of creation (to maintain)
    pub relative_rotation: QuatFix,
    /// Positional compliance (inverse stiffness)
    pub compliance: Fix128,
    /// Angular compliance (inverse stiffness)
    pub angular_compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl FixedJoint {
    /// Create a new fixed joint (weld)
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        relative_rotation: QuatFix,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            relative_rotation,
            compliance: Fix128::ZERO,
            angular_compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Slider (prismatic) joint: translation along a single axis
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliderJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Slide axis in body A's local space
    pub local_axis: Vec3Fix,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Minimum translation distance (None = no limit)
    pub limit_min: Option<Fix128>,
    /// Maximum translation distance (None = no limit)
    pub limit_max: Option<Fix128>,
    /// Positional compliance (inverse stiffness)
    pub compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl SliderJoint {
    /// Create a new slider joint along an axis
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        axis: Vec3Fix,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_axis: axis,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            limit_min: None,
            limit_max: None,
            compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set translation limits
    #[must_use]
    pub const fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        self.limit_min = Some(min);
        self.limit_max = Some(max);
        self
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Spring joint: distance spring with damping
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpringJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Rest length of the spring
    pub rest_length: Fix128,
    /// Spring stiffness
    pub stiffness: Fix128,
    /// Damping coefficient
    pub damping: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl SpringJoint {
    /// Create a new spring joint
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        rest_length: Fix128,
        stiffness: Fix128,
        damping: Fix128,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            rest_length,
            stiffness,
            damping,
            break_force: None,
        }
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Axis freedom mode for D6 joints
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum D6Motion {
    /// Axis is locked (no motion allowed)
    Locked,
    /// Axis is free (unlimited motion)
    #[default]
    Free,
    /// Axis is limited (within min/max bounds)
    Limited,
}

/// D6 Joint (6-DOF configurable joint)
///
/// Each of the 6 axes (3 linear + 3 angular) can be independently
/// set to Free, Locked, or Limited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D6Joint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Reference frame for body A
    pub local_frame_a: QuatFix,
    /// Reference frame for body B
    pub local_frame_b: QuatFix,
    /// Linear X axis motion
    pub linear_x: D6Motion,
    /// Linear Y axis motion
    pub linear_y: D6Motion,
    /// Linear Z axis motion
    pub linear_z: D6Motion,
    /// Angular X (twist) motion
    pub angular_x: D6Motion,
    /// Angular Y (swing1) motion
    pub angular_y: D6Motion,
    /// Angular Z (swing2) motion
    pub angular_z: D6Motion,
    /// Linear limits (min/max for limited axes)
    pub linear_limit_min: Vec3Fix,
    /// Linear limit max
    pub linear_limit_max: Vec3Fix,
    /// Angular limits (min for limited axes, radians)
    pub angular_limit_min: Vec3Fix,
    /// Angular limit max
    pub angular_limit_max: Vec3Fix,
    /// Positional compliance
    pub compliance: Fix128,
    /// Angular compliance
    pub angular_compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl D6Joint {
    /// Create a new D6 joint with all axes free
    #[must_use]
    pub fn new(body_a: usize, body_b: usize, anchor_a: Vec3Fix, anchor_b: Vec3Fix) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            local_frame_a: QuatFix::IDENTITY,
            local_frame_b: QuatFix::IDENTITY,
            linear_x: D6Motion::Free,
            linear_y: D6Motion::Free,
            linear_z: D6Motion::Free,
            angular_x: D6Motion::Free,
            angular_y: D6Motion::Free,
            angular_z: D6Motion::Free,
            linear_limit_min: Vec3Fix::new(-Fix128::ONE, -Fix128::ONE, -Fix128::ONE),
            linear_limit_max: Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE),
            angular_limit_min: Vec3Fix::new(-Fix128::PI, -Fix128::PI, -Fix128::PI),
            angular_limit_max: Vec3Fix::new(Fix128::PI, Fix128::PI, Fix128::PI),
            compliance: Fix128::ZERO,
            angular_compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set linear motion on all axes
    #[must_use]
    pub const fn with_linear_motion(mut self, x: D6Motion, y: D6Motion, z: D6Motion) -> Self {
        self.linear_x = x;
        self.linear_y = y;
        self.linear_z = z;
        self
    }

    /// Set angular motion on all axes
    #[must_use]
    pub const fn with_angular_motion(mut self, x: D6Motion, y: D6Motion, z: D6Motion) -> Self {
        self.angular_x = x;
        self.angular_y = y;
        self.angular_z = z;
        self
    }

    /// Set linear limits
    #[must_use]
    pub const fn with_linear_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        self.linear_limit_min = min;
        self.linear_limit_max = max;
        self
    }

    /// Set angular limits
    #[must_use]
    pub const fn with_angular_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        self.angular_limit_min = min;
        self.angular_limit_max = max;
        self
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Cone-Twist joint (ball joint with cone angle limit + twist limit)
///
/// Used for ragdoll shoulders and hips where rotation is constrained
/// to a cone around the twist axis, with an additional twist limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConeTwistJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Twist axis in body A's local space
    pub twist_axis_a: Vec3Fix,
    /// Twist axis in body B's local space
    pub twist_axis_b: Vec3Fix,
    /// Maximum cone angle (radians, half-angle)
    pub cone_limit: Fix128,
    /// Maximum twist angle (radians)
    pub twist_limit: Fix128,
    /// Positional compliance
    pub compliance: Fix128,
    /// Angular compliance
    pub angular_compliance: Fix128,
    /// Maximum force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
}

impl ConeTwistJoint {
    /// Create a new cone-twist joint
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        twist_axis_a: Vec3Fix,
        twist_axis_b: Vec3Fix,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            twist_axis_a,
            twist_axis_b,
            cone_limit: Fix128::HALF_PI,
            twist_limit: Fix128::PI,
            compliance: Fix128::ZERO,
            angular_compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Set cone and twist limits
    #[must_use]
    pub const fn with_limits(mut self, cone: Fix128, twist: Fix128) -> Self {
        self.cone_limit = cone;
        self.twist_limit = twist;
        self
    }

    /// Set break force threshold
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Unified joint enum for storage in `PhysicsWorld`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Joint {
    /// Ball-and-socket joint
    Ball(BallJoint),
    /// Hinge joint
    Hinge(HingeJoint),
    /// Fixed / weld joint
    Fixed(FixedJoint),
    /// Slider / prismatic joint
    Slider(SliderJoint),
    /// Spring joint
    Spring(SpringJoint),
    /// 6-DOF configurable joint
    D6(D6Joint),
    /// Cone-twist joint (ragdoll)
    ConeTwist(ConeTwistJoint),
}

impl Joint {
    /// Get body indices for this joint
    #[inline]
    #[must_use]
    pub const fn bodies(&self) -> (usize, usize) {
        match self {
            Self::Ball(j) => (j.body_a, j.body_b),
            Self::Hinge(j) => (j.body_a, j.body_b),
            Self::Fixed(j) => (j.body_a, j.body_b),
            Self::Slider(j) => (j.body_a, j.body_b),
            Self::Spring(j) => (j.body_a, j.body_b),
            Self::D6(j) => (j.body_a, j.body_b),
            Self::ConeTwist(j) => (j.body_a, j.body_b),
        }
    }

    /// Get joint type
    #[inline]
    #[must_use]
    pub const fn joint_type(&self) -> JointType {
        match self {
            Self::Ball(_) => JointType::Ball,
            Self::Hinge(_) => JointType::Hinge,
            Self::Fixed(_) => JointType::Fixed,
            Self::Slider(_) => JointType::Slider,
            Self::Spring(_) => JointType::Spring,
            Self::D6(_) => JointType::D6,
            Self::ConeTwist(_) => JointType::ConeTwist,
        }
    }

    /// Get the break force threshold (None = unbreakable)
    #[inline]
    #[must_use]
    pub const fn break_force(&self) -> Option<Fix128> {
        match self {
            Self::Ball(j) => j.break_force,
            Self::Hinge(j) => j.break_force,
            Self::Fixed(j) => j.break_force,
            Self::Slider(j) => j.break_force,
            Self::Spring(j) => j.break_force,
            Self::D6(j) => j.break_force,
            Self::ConeTwist(j) => j.break_force,
        }
    }

    /// Compute current constraint force (distance between anchor points).
    ///
    /// Returns the force magnitude used to determine if the joint should break.
    #[must_use]
    pub fn compute_force(&self, bodies: &[crate::solver::RigidBody]) -> Fix128 {
        let (a_idx, b_idx) = self.bodies();
        let body_a = &bodies[a_idx];
        let body_b = &bodies[b_idx];

        match self {
            Self::Ball(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Self::Hinge(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Self::Fixed(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Self::Slider(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                let delta = anchor_b - anchor_a;
                let world_axis = body_a.rotation.rotate_vec(j.local_axis).normalize();
                let along = delta.dot(world_axis);
                let perp = delta - world_axis * along;
                perp.length()
            }
            Self::Spring(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                let dist = (anchor_b - anchor_a).length();
                let displacement = dist - j.rest_length;
                (j.stiffness * displacement).abs()
            }
            Self::D6(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Self::ConeTwist(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
        }
    }
}

/// Solve all joints for one XPBD iteration
///
/// Modifies body positions/rotations in-place to satisfy constraints.
pub fn solve_joints(joints: &[Joint], bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    for joint in joints {
        match joint {
            Joint::Ball(j) => solve_ball_joint(j, bodies, dt),
            Joint::Hinge(j) => solve_hinge_joint(j, bodies, dt),
            Joint::Fixed(j) => solve_fixed_joint(j, bodies, dt),
            Joint::Slider(j) => solve_slider_joint(j, bodies, dt),
            Joint::Spring(j) => solve_spring_joint(j, bodies, dt),
            Joint::D6(j) => solve_d6_joint(j, bodies, dt),
            Joint::ConeTwist(j) => solve_cone_twist_joint(j, bodies, dt),
        }
    }
}

/// Solve joints and return indices of joints that should be removed (broken).
///
/// Checks each breakable joint's constraint force BEFORE solving.
/// If the force exceeds the threshold, the joint is marked as broken
/// and skipped during solving. Returns indices of broken joints in
/// descending order (safe for sequential removal).
pub fn solve_joints_breakable(
    joints: &[Joint],
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec<usize> {
    // Check which joints exceeded their break force BEFORE solving
    let mut broken = Vec::new();
    for (i, joint) in joints.iter().enumerate() {
        if let Some(max_force) = joint.break_force() {
            let force = joint.compute_force(bodies);
            if force > max_force {
                broken.push(i);
            }
        }
    }

    // Sort for binary search lookup
    broken.sort_unstable();

    // Solve only non-broken joints
    for (i, joint) in joints.iter().enumerate() {
        if broken.binary_search(&i).is_ok() {
            continue;
        }
        match joint {
            Joint::Ball(j) => solve_ball_joint(j, bodies, dt),
            Joint::Hinge(j) => solve_hinge_joint(j, bodies, dt),
            Joint::Fixed(j) => solve_fixed_joint(j, bodies, dt),
            Joint::Slider(j) => solve_slider_joint(j, bodies, dt),
            Joint::Spring(j) => solve_spring_joint(j, bodies, dt),
            Joint::D6(j) => solve_d6_joint(j, bodies, dt),
            Joint::ConeTwist(j) => solve_cone_twist_joint(j, bodies, dt),
        }
    }

    // Sort descending for safe removal
    broken.sort_unstable_by(|a, b| b.cmp(a));
    broken
}

/// Solve ball joint: constrain anchor points to coincide
fn solve_ball_joint(joint: &BallJoint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // World-space anchor positions
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if distance.is_zero() {
        return;
    }

    let compliance_term = joint.compliance / (dt * dt);
    let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

    if w_sum.is_zero() {
        return;
    }

    let inv_w_sum = Fix128::ONE / w_sum;
    let lambda = distance * inv_w_sum;
    let correction = normal * lambda;

    if !body_a.inv_mass.is_zero() {
        bodies[joint.body_a].position =
            bodies[joint.body_a].position + correction * body_a.inv_mass;
    }
    if !body_b.inv_mass.is_zero() {
        bodies[joint.body_b].position =
            bodies[joint.body_b].position - correction * body_b.inv_mass;
    }
}

/// Solve hinge joint: positional + angular constraint along axis
fn solve_hinge_joint(joint: &HingeJoint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint (same as ball joint)
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if !w_sum.is_zero() {
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = distance * inv_w_sum;
            let correction = normal * lambda;

            if !body_a.inv_mass.is_zero() {
                bodies[joint.body_a].position =
                    bodies[joint.body_a].position + correction * body_a.inv_mass;
            }
            if !body_b.inv_mass.is_zero() {
                bodies[joint.body_b].position =
                    bodies[joint.body_b].position - correction * body_b.inv_mass;
            }
        }
    }

    // 2. Angular constraint: align axes
    let world_axis_a = body_a.rotation.rotate_vec(joint.local_axis_a);
    let world_axis_b = body_b.rotation.rotate_vec(joint.local_axis_b);

    let axis_error = world_axis_a.cross(world_axis_b);
    let (correction_axis, error_mag) = axis_error.normalize_with_length();

    if !error_mag.is_zero() {
        let angular_compliance = joint.angular_compliance / (dt * dt);
        let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length() + angular_compliance;

        if !w_ang.is_zero() {
            let inv_w_ang = Fix128::ONE / w_ang;
            let angular_lambda = error_mag * inv_w_ang;

            apply_angular_correction(
                bodies,
                joint.body_a,
                joint.body_b,
                correction_axis,
                angular_lambda,
            );
        }
    }

    // 3. Angle limits
    if let (Some(min_angle), Some(max_angle)) = (joint.angle_min, joint.angle_max) {
        let body_a = bodies[joint.body_a];
        let body_b = bodies[joint.body_b];
        let world_axis_a = body_a.rotation.rotate_vec(joint.local_axis_a);

        // Compute relative angle around hinge axis
        let rel_quat = body_b.rotation.mul(body_a.rotation.conjugate());
        let angle = compute_twist_angle(rel_quat, world_axis_a);

        if angle < min_angle {
            let error = min_angle - angle;
            let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length();
            if !w_ang.is_zero() {
                let inv_w_ang = Fix128::ONE / w_ang;
                apply_angular_correction(
                    bodies,
                    joint.body_a,
                    joint.body_b,
                    world_axis_a,
                    -(error * inv_w_ang),
                );
            }
        } else if angle > max_angle {
            let error = angle - max_angle;
            let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length();
            if !w_ang.is_zero() {
                let inv_w_ang = Fix128::ONE / w_ang;
                apply_angular_correction(
                    bodies,
                    joint.body_a,
                    joint.body_b,
                    world_axis_a,
                    error * inv_w_ang,
                );
            }
        }
    }
}

/// Solve fixed joint: positional + full rotational lock
fn solve_fixed_joint(joint: &FixedJoint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if !w_sum.is_zero() {
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = distance * inv_w_sum;
            let correction = normal * lambda;

            if !body_a.inv_mass.is_zero() {
                bodies[joint.body_a].position =
                    bodies[joint.body_a].position + correction * body_a.inv_mass;
            }
            if !body_b.inv_mass.is_zero() {
                bodies[joint.body_b].position =
                    bodies[joint.body_b].position - correction * body_b.inv_mass;
            }
        }
    }

    // 2. Rotational constraint: maintain relative rotation
    let target_rot_b = body_a.rotation.mul(joint.relative_rotation);
    let rot_error = body_b.rotation.mul(target_rot_b.conjugate());

    // Extract error as rotation vector (axis * angle)
    let error_vec = Vec3Fix::new(rot_error.x, rot_error.y, rot_error.z);
    let (correction_axis, error_mag) = error_vec.normalize_with_length();

    if !error_mag.is_zero() {
        let angular_compliance = joint.angular_compliance / (dt * dt);
        let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length() + angular_compliance;

        if !w_ang.is_zero() {
            let inv_w_ang = Fix128::ONE / w_ang;
            let two = Fix128::from_int(2);
            let angular_lambda = (error_mag * two) * inv_w_ang;

            apply_angular_correction(
                bodies,
                joint.body_a,
                joint.body_b,
                correction_axis,
                angular_lambda,
            );
        }
    }
}

/// Solve slider joint: constrain to 1-DOF translation along axis
fn solve_slider_joint(joint: &SliderJoint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    let world_axis = body_a.rotation.rotate_vec(joint.local_axis).normalize();

    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);

    let delta = anchor_b - anchor_a;

    // Project delta onto axis
    let along_axis = delta.dot(world_axis);

    // Perpendicular error (must be zero)
    let perp = delta - world_axis * along_axis;
    let (perp_normal, perp_dist) = perp.normalize_with_length();

    if !perp_dist.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if !w_sum.is_zero() {
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = perp_dist * inv_w_sum;
            let correction = perp_normal * lambda;

            if !body_a.inv_mass.is_zero() {
                bodies[joint.body_a].position =
                    bodies[joint.body_a].position + correction * body_a.inv_mass;
            }
            if !body_b.inv_mass.is_zero() {
                bodies[joint.body_b].position =
                    bodies[joint.body_b].position - correction * body_b.inv_mass;
            }
        }
    }

    // Enforce translation limits
    if let (Some(min_d), Some(max_d)) = (joint.limit_min, joint.limit_max) {
        if along_axis < min_d {
            let error = min_d - along_axis;
            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if !w_sum.is_zero() {
                let inv_w_sum = Fix128::ONE / w_sum;
                let correction = world_axis * (error * inv_w_sum);
                if !body_a.inv_mass.is_zero() {
                    bodies[joint.body_a].position =
                        bodies[joint.body_a].position - correction * body_a.inv_mass;
                }
                if !body_b.inv_mass.is_zero() {
                    bodies[joint.body_b].position =
                        bodies[joint.body_b].position + correction * body_b.inv_mass;
                }
            }
        } else if along_axis > max_d {
            let error = along_axis - max_d;
            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if !w_sum.is_zero() {
                let inv_w_sum = Fix128::ONE / w_sum;
                let correction = world_axis * (error * inv_w_sum);
                if !body_a.inv_mass.is_zero() {
                    bodies[joint.body_a].position =
                        bodies[joint.body_a].position + correction * body_a.inv_mass;
                }
                if !body_b.inv_mass.is_zero() {
                    bodies[joint.body_b].position =
                        bodies[joint.body_b].position - correction * body_b.inv_mass;
                }
            }
        }
    }
}

/// Solve spring joint: spring force with damping
fn solve_spring_joint(joint: &SpringJoint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if distance.is_zero() {
        return;
    }

    // Spring force: F = -k * (x - rest_length)
    let displacement = distance - joint.rest_length;
    let spring_force = joint.stiffness * displacement;

    // Damping force: F = -c * v_relative_along_normal
    let rel_vel = body_b.velocity - body_a.velocity;
    let vel_along_normal = rel_vel.dot(normal);
    let damping_force = joint.damping * vel_along_normal;

    let total_force = spring_force + damping_force;

    // Convert to impulse (force * dt)
    let impulse = normal * (total_force * dt);

    let w_sum = body_a.inv_mass + body_b.inv_mass;
    if w_sum.is_zero() {
        return;
    }

    if !body_a.inv_mass.is_zero() {
        bodies[joint.body_a].position = bodies[joint.body_a].position + impulse * body_a.inv_mass;
    }
    if !body_b.inv_mass.is_zero() {
        bodies[joint.body_b].position = bodies[joint.body_b].position - impulse * body_b.inv_mass;
    }
}

/// Solve D6 joint: per-axis locking/limiting for all 6 DOF
#[allow(clippy::too_many_lines)]
fn solve_d6_joint(joint: &D6Joint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // World-space anchors
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);
    let delta = anchor_b - anchor_a;

    // Get local frame axes in world space
    let frame_a = body_a.rotation.mul(joint.local_frame_a);
    let axis_x = frame_a.rotate_vec(Vec3Fix::UNIT_X);
    let axis_y = frame_a.rotate_vec(Vec3Fix::UNIT_Y);
    let axis_z = frame_a.rotate_vec(Vec3Fix::UNIT_Z);

    let compliance_term = joint.compliance / (dt * dt);
    let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

    if !w_sum.is_zero() {
        let inv_w_sum = Fix128::ONE / w_sum;

        // Linear constraints per axis
        let axes = [
            (
                joint.linear_x,
                axis_x,
                joint.linear_limit_min.x,
                joint.linear_limit_max.x,
            ),
            (
                joint.linear_y,
                axis_y,
                joint.linear_limit_min.y,
                joint.linear_limit_max.y,
            ),
            (
                joint.linear_z,
                axis_z,
                joint.linear_limit_min.z,
                joint.linear_limit_max.z,
            ),
        ];

        for &(motion, axis, limit_min, limit_max) in &axes {
            let proj = delta.dot(axis);
            let error = match motion {
                D6Motion::Locked => proj,
                D6Motion::Limited => {
                    if proj < limit_min {
                        proj - limit_min
                    } else if proj > limit_max {
                        proj - limit_max
                    } else {
                        Fix128::ZERO
                    }
                }
                D6Motion::Free => Fix128::ZERO,
            };

            if !error.is_zero() {
                let lambda = error * inv_w_sum;
                let correction = axis * lambda;

                if !body_a.inv_mass.is_zero() {
                    bodies[joint.body_a].position =
                        bodies[joint.body_a].position + correction * body_a.inv_mass;
                }
                if !body_b.inv_mass.is_zero() {
                    bodies[joint.body_b].position =
                        bodies[joint.body_b].position - correction * body_b.inv_mass;
                }
            }
        }
    }

    // Angular constraints per axis
    let angular_compliance = joint.angular_compliance / (dt * dt);
    let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length() + angular_compliance;

    if !w_ang.is_zero() {
        let inv_w_ang = Fix128::ONE / w_ang;
        let rel_quat = body_b.rotation.mul(body_a.rotation.conjugate());

        let ang_axes = [
            (
                joint.angular_x,
                axis_x,
                joint.angular_limit_min.x,
                joint.angular_limit_max.x,
            ),
            (
                joint.angular_y,
                axis_y,
                joint.angular_limit_min.y,
                joint.angular_limit_max.y,
            ),
            (
                joint.angular_z,
                axis_z,
                joint.angular_limit_min.z,
                joint.angular_limit_max.z,
            ),
        ];

        for &(motion, axis, limit_min, limit_max) in &ang_axes {
            let angle = compute_twist_angle(rel_quat, axis);

            let error = match motion {
                D6Motion::Locked => angle,
                D6Motion::Limited => {
                    if angle < limit_min {
                        angle - limit_min
                    } else if angle > limit_max {
                        angle - limit_max
                    } else {
                        Fix128::ZERO
                    }
                }
                D6Motion::Free => Fix128::ZERO,
            };

            if !error.is_zero() {
                apply_angular_correction(
                    bodies,
                    joint.body_a,
                    joint.body_b,
                    axis,
                    error * inv_w_ang,
                );
            }
        }
    }
}

/// Solve cone-twist joint: positional + cone + twist constraints
fn solve_cone_twist_joint(
    joint: &ConeTwistJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint (same as ball joint)
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);
    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if !w_sum.is_zero() {
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = distance * inv_w_sum;
            let correction = normal * lambda;

            if !body_a.inv_mass.is_zero() {
                bodies[joint.body_a].position =
                    bodies[joint.body_a].position + correction * body_a.inv_mass;
            }
            if !body_b.inv_mass.is_zero() {
                bodies[joint.body_b].position =
                    bodies[joint.body_b].position - correction * body_b.inv_mass;
            }
        }
    }

    // 2. Cone constraint
    let world_axis_a = body_a.rotation.rotate_vec(joint.twist_axis_a).normalize();
    let world_axis_b = body_b.rotation.rotate_vec(joint.twist_axis_b).normalize();

    let dot = world_axis_a.dot(world_axis_b);
    let cross = world_axis_a.cross(world_axis_b);
    let (correction_axis, cross_len) = cross.normalize_with_length();
    // Use atan2(sin, cos) instead of acos for deterministic fixed-point
    let cone_angle = Fix128::atan2(cross_len, dot);
    let cone_angle = if cone_angle < Fix128::ZERO {
        -cone_angle
    } else {
        cone_angle
    };

    if cone_angle > joint.cone_limit {
        let error = cone_angle - joint.cone_limit;

        if !cross_len.is_zero() {
            let angular_compliance = joint.angular_compliance / (dt * dt);
            let w_ang =
                body_a.inv_inertia.length() + body_b.inv_inertia.length() + angular_compliance;

            if !w_ang.is_zero() {
                let inv_w_ang = Fix128::ONE / w_ang;
                apply_angular_correction(
                    bodies,
                    joint.body_a,
                    joint.body_b,
                    correction_axis,
                    error * inv_w_ang,
                );
            }
        }
    }

    // 3. Twist constraint
    let rel_quat = bodies[joint.body_b]
        .rotation
        .mul(bodies[joint.body_a].rotation.conjugate());
    let twist_angle = compute_twist_angle(rel_quat, world_axis_a);

    if twist_angle.abs() > joint.twist_limit {
        let error = if twist_angle > Fix128::ZERO {
            twist_angle - joint.twist_limit
        } else {
            twist_angle + joint.twist_limit
        };

        let angular_compliance = joint.angular_compliance / (dt * dt);
        let w_ang = body_a.inv_inertia.length() + body_b.inv_inertia.length() + angular_compliance;

        if !w_ang.is_zero() {
            let inv_w_ang = Fix128::ONE / w_ang;
            apply_angular_correction(
                bodies,
                joint.body_a,
                joint.body_b,
                world_axis_a,
                error * inv_w_ang,
            );
        }
    }
}

/// Apply angular correction to two bodies (utility)
///
/// Uses `split_at_mut` to safely obtain two mutable references.
fn apply_angular_correction(
    bodies: &mut [crate::solver::RigidBody],
    idx_a: usize,
    idx_b: usize,
    axis: Vec3Fix,
    magnitude: Fix128,
) {
    let half_mag = magnitude.half();
    let inv_mass_a = bodies[idx_a].inv_mass;
    let inv_mass_b = bodies[idx_b].inv_mass;

    if !inv_mass_a.is_zero() {
        let delta_q = QuatFix::new(
            axis.x * half_mag,
            axis.y * half_mag,
            axis.z * half_mag,
            Fix128::ONE,
        );
        bodies[idx_a].rotation = delta_q.mul(bodies[idx_a].rotation).normalize();
    }
    if !inv_mass_b.is_zero() {
        let delta_q = QuatFix::new(
            -(axis.x * half_mag),
            -(axis.y * half_mag),
            -(axis.z * half_mag),
            Fix128::ONE,
        );
        bodies[idx_b].rotation = delta_q.mul(bodies[idx_b].rotation).normalize();
    }
}

/// Compute twist angle of a quaternion around a given axis
fn compute_twist_angle(q: QuatFix, axis: Vec3Fix) -> Fix128 {
    let qv = Vec3Fix::new(q.x, q.y, q.z);
    let proj = axis * qv.dot(axis);
    let twist = QuatFix::new(proj.x, proj.y, proj.z, q.w).normalize();

    // angle = 2 * atan2(|twist.xyz|, twist.w)
    let xyz_len = Vec3Fix::new(twist.x, twist.y, twist.z).length();
    Fix128::atan2(xyz_len, twist.w).double()
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::solver::RigidBody;

    #[test]
    fn test_ball_joint_holds() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        ];

        let joint = Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO));
        let dt = Fix128::from_ratio(1, 60);

        for _ in 0..100 {
            solve_joints(&[joint], &mut bodies, dt);
        }

        // Body 1 should be pulled toward body 0
        let dist = (bodies[1].position - bodies[0].position).length();
        assert!(
            dist < Fix128::from_int(5),
            "Ball joint should pull bodies together"
        );
    }

    #[test]
    fn test_fixed_joint() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
        ];

        let joint = Joint::Fixed(FixedJoint::new(
            0,
            1,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        let dt = Fix128::from_ratio(1, 60);

        for _ in 0..100 {
            solve_joints(&[joint], &mut bodies, dt);
        }

        // Body 1 should be near anchor_a (1,0,0)
        let dist = (bodies[1].position - Vec3Fix::from_int(1, 0, 0)).length();
        assert!(
            dist < Fix128::from_int(3),
            "Fixed joint should hold position"
        );
    }

    #[test]
    fn test_spring_joint() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE),
        ];

        let joint = Joint::Spring(SpringJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(3),  // rest length = 3
            Fix128::from_int(10), // stiffness = 10
            Fix128::from_int(1),  // damping = 1
        ));
        let dt = Fix128::from_ratio(1, 60);

        for _ in 0..200 {
            solve_joints(&[joint], &mut bodies, dt);
        }

        // Should oscillate toward rest length = 3
        let dist = bodies[1].position.length();
        assert!(
            dist < Fix128::from_int(10),
            "Spring should pull body closer"
        );
    }

    #[test]
    fn test_slider_joint() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(3, 5, 0), Fix128::ONE),
        ];

        let joint = Joint::Slider(SliderJoint::new(
            0,
            1,
            Vec3Fix::UNIT_X,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ));
        let dt = Fix128::from_ratio(1, 60);

        for _ in 0..100 {
            solve_joints(&[joint], &mut bodies, dt);
        }

        // Y component should be constrained toward 0 (perpendicular to axis)
        let y_abs = bodies[1].position.y.abs();
        assert!(
            y_abs < Fix128::from_int(5),
            "Slider should constrain perpendicular motion"
        );
    }

    #[test]
    fn test_joint_bodies() {
        let ball = Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO));
        assert_eq!(ball.bodies(), (0, 1));
        assert_eq!(ball.joint_type(), JointType::Ball);
    }

    #[test]
    fn test_breakable_joint() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(20, 0, 0), Fix128::ONE), // far away
        ];

        // Ball joint with low break force (should break immediately)
        let joint = Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
                .with_break_force(Fix128::from_int(5)),
        );
        let dt = Fix128::from_ratio(1, 60);

        let broken = solve_joints_breakable(&[joint], &mut bodies, dt);
        // Distance is ~20, break force is 5 → should break
        assert_eq!(broken.len(), 1, "Joint should break under high force");
        assert_eq!(broken[0], 0);
    }

    #[test]
    fn test_unbreakable_joint() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
        ];

        // Ball joint with high break force (should NOT break)
        let joint = Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
                .with_break_force(Fix128::from_int(100)),
        );
        let dt = Fix128::from_ratio(1, 60);

        let broken = solve_joints_breakable(&[joint], &mut bodies, dt);
        assert!(broken.is_empty(), "Joint should not break under low force");
    }

    #[test]
    fn test_no_break_force() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(100, 0, 0), Fix128::ONE),
        ];

        // No break force → never breaks
        let joint = Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO));
        let dt = Fix128::from_ratio(1, 60);

        let broken = solve_joints_breakable(&[joint], &mut bodies, dt);
        assert!(
            broken.is_empty(),
            "Joint without break_force should never break"
        );
    }

    // ---- mutation-score tests (2026-09-15) ----------------------------
    // dt = 1/4 (inv exact)、位置は整数、質量は inv_mass を直接設定して閉形式と bit 単位一致

    fn fi(n: i64) -> Fix128 {
        Fix128::from_int(n)
    }

    fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    const DT: Fix128 = Fix128 { hi: 0, lo: 1 << 62 }; // 1/4

    /// A: inv_mass ia at pa、B: inv_mass ib at pb (inv_inertia は (1,1,1))
    fn pair(pa: Vec3Fix, ia: i64, pb: Vec3Fix, ib: i64) -> Vec<RigidBody> {
        let mut a = RigidBody::new_dynamic(pa, Fix128::ONE);
        let mut b = RigidBody::new_dynamic(pb, Fix128::ONE);
        a.inv_mass = fi(ia);
        b.inv_mass = fi(ib);
        a.inv_inertia = v3i(1, 1, 1);
        b.inv_inertia = v3i(1, 1, 1);
        if ia == 0 {
            a.body_type = crate::solver::BodyType::Static;
        }
        if ib == 0 {
            b.body_type = crate::solver::BodyType::Static;
        }
        vec![a, b]
    }

    fn near(a: Fix128, b: Fix128) -> bool {
        (a - b).abs() < Fix128 { hi: 0, lo: 1 << 24 }
    }

    fn near_v(a: Vec3Fix, b: Vec3Fix) -> bool {
        near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z)
    }

    // ---- ball ----------------------------------------------------------

    #[test]
    fn ball_joint_splits_gap_by_inverse_mass() {
        // A inv 1 at 0、B inv 3 at (4,0,0)、anchor 0 → gap 4、λ = 1 → A +1、B -3 → 両方 (1,0,0)
        let mut bodies = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        let j = BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        solve_ball_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[0].position, v3i(1, 0, 0));
        assert_eq!(bodies[1].position, v3i(1, 0, 0));
    }

    #[test]
    fn ball_joint_local_anchors_are_rotated_and_static_side_fixed() {
        // A static at 0 with anchor (1,0,0) rotated 180° about z → world anchor (-1,0,0)
        // B inv 1 at (3,0,0) anchor (0,0,0) → gap 4 → B moves to (-1,0,0)
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(3, 0, 0), 1);
        bodies[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let j = BallJoint::new(0, 1, v3i(1, 0, 0), Vec3Fix::ZERO);
        solve_ball_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[0].position, Vec3Fix::ZERO);
        assert_eq!(bodies[1].position, v3i(-1, 0, 0));
        // B 側 anchor (0,2,0): B の world anchor が A anchor に重なる位置 = (-1,-2,0)
        let mut bodies2 = pair(Vec3Fix::ZERO, 0, v3i(3, 0, 0), 1);
        bodies2[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let j2 = BallJoint::new(0, 1, v3i(1, 0, 0), v3i(0, 2, 0));
        solve_ball_joint(&j2, &mut bodies2, DT);
        assert!(
            near_v(bodies2[1].position, v3i(-1, -2, 0)),
            "{:?}",
            bodies2[1].position
        );
    }

    #[test]
    fn ball_joint_compliance_and_degenerate_cases() {
        // compliance 1/16 → term 1、A static、B inv 1 gap 4 → w 2 → λ 2 → B at (2,0,0)
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        let mut j = BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        j.compliance = Fix128::from_ratio(1, 16);
        solve_ball_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[1].position, v3i(2, 0, 0));
        // 一致点 / 両 static → 変化なし
        let mut same = pair(v3i(1, 1, 1), 1, v3i(1, 1, 1), 1);
        solve_ball_joint(
            &BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO),
            &mut same,
            DT,
        );
        assert_eq!(same[0].position, v3i(1, 1, 1));
        let mut statics = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 0);
        solve_ball_joint(
            &BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO),
            &mut statics,
            DT,
        );
        assert_eq!(statics[1].position, v3i(4, 0, 0));
    }

    // ---- hinge ---------------------------------------------------------

    #[test]
    fn hinge_joint_positional_part_matches_ball_joint() {
        let mut bodies = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        solve_hinge_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[0].position, v3i(1, 0, 0));
        assert_eq!(bodies[1].position, v3i(1, 0, 0));
        // 軸が揃っていれば回転は不変
        assert_eq!(bodies[0].rotation, QuatFix::IDENTITY);
        assert_eq!(bodies[1].rotation, QuatFix::IDENTITY);
    }

    #[test]
    fn hinge_joint_aligns_axes_monotonically() {
        // B の軸を x 軸周りに 0.5 rad 傾ける → 反復で軸誤差が単調減少、A (static) は不動
        let mut bodies = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(1, 2));
        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        let err = |b: &[RigidBody]| {
            let a = b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            let c = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            a.cross(c).length()
        };
        let mut prev = err(&bodies);
        assert!(prev > Fix128::from_ratio(1, 10));
        for i in 0..30 {
            solve_hinge_joint(&j, &mut bodies, DT);
            let e = err(&bodies);
            assert!(e <= prev, "iter {i}: {prev:?} -> {e:?}");
            prev = e;
        }
        assert!(prev < Fix128::from_ratio(1, 1000));
        assert_eq!(bodies[0].rotation, QuatFix::IDENTITY);
        // 両 dynamic なら両方が回る (A も IDENTITY から離れる)
        let mut both = pair(Vec3Fix::ZERO, 1, Vec3Fix::ZERO, 1);
        both[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(1, 2));
        solve_hinge_joint(&j, &mut both, DT);
        assert!(both[0].rotation != QuatFix::IDENTITY);
    }

    #[test]
    fn hinge_joint_limits_push_angle_back_into_range() {
        // 軸 z、B を z 周りに +1 rad 回す、limit [-0.5, 0.5] → max 超過 → 角度が減る
        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(Fix128::from_ratio(-1, 2), Fix128::from_ratio(1, 2));
        let angle = |b: &[RigidBody]| {
            let rel = b[1].rotation.mul(b[0].rotation.conjugate());
            compute_twist_angle(rel, Vec3Fix::UNIT_Z)
        };
        let mut over = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        over[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
        let a0 = angle(&over);
        assert!(near(a0, Fix128::ONE), "{a0:?}");
        solve_hinge_joint(&j, &mut over, DT);
        let a1 = angle(&over);
        assert!(
            a1 < a0 && a1 >= Fix128::from_ratio(1, 2) - Fix128::from_ratio(1, 100),
            "{a0:?} -> {a1:?}"
        );
        // min 側: -1 rad → 増える
        let mut under = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        under[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fi(-1));
        let b0 = angle(&under);
        solve_hinge_joint(&j, &mut under, DT);
        let b1 = angle(&under);
        assert!(b1 > b0, "{b0:?} -> {b1:?}");
        // 範囲内 (0.25) は不変、limit なしも不変
        let mut inside = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        inside[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 4));
        let r0 = inside[1].rotation;
        solve_hinge_joint(&j, &mut inside, DT);
        assert_eq!(inside[1].rotation, r0);
        let mut free = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        free[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
        let f0 = free[1].rotation;
        let nolimit = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        solve_hinge_joint(&nolimit, &mut free, DT);
        assert_eq!(free[1].rotation, f0);
    }

    // ---- fixed ---------------------------------------------------------

    #[test]
    fn fixed_joint_restores_relative_rotation_and_position() {
        let mut bodies = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        let j = FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        solve_fixed_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[0].position, v3i(1, 0, 0));
        assert_eq!(bodies[1].position, v3i(1, 0, 0));
        // 相対回転 identity で B が z 周り 0.5 rad ずれている → 反復で相対回転誤差が単調減少
        let mut rot = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        rot[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 2));
        let err = |b: &[RigidBody]| {
            let e = b[1].rotation.mul(b[0].rotation.conjugate());
            Vec3Fix::new(e.x, e.y, e.z).length()
        };
        let mut prev = err(&rot);
        for i in 0..30 {
            solve_fixed_joint(&j, &mut rot, DT);
            let e = err(&rot);
            assert!(e <= prev, "iter {i}");
            prev = e;
        }
        assert!(prev < Fix128::from_ratio(1, 1000));
        // 目標相対回転が 90° なら、B が 90° 回っている状態は不変
        let target = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
        let j90 = FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, target);
        let mut ok = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        ok[1].rotation = target;
        let before = ok[1].rotation;
        solve_fixed_joint(&j90, &mut ok, DT);
        assert!(
            near(ok[1].rotation.x, before.x)
                && near(ok[1].rotation.z, before.z)
                && near(ok[1].rotation.w, before.w)
        );
    }

    // ---- slider --------------------------------------------------------

    #[test]
    fn slider_joint_removes_perpendicular_offset_only() {
        // 軸 x、B が (3, 4, 0): along 3 は自由、perp (0,4,0) を除去 → inv 1 / static → B (3,0,0)
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let j = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO);
        solve_slider_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[1].position, v3i(3, 0, 0));
        // 両 dynamic inv 1 / 3: perp 4 → λ 1 → A +1 (y)、B -3 (y) → A (0,1,0)、B (3,1,0)
        let mut both = pair(Vec3Fix::ZERO, 1, v3i(3, 4, 0), 3);
        solve_slider_joint(&j, &mut both, DT);
        assert_eq!(both[0].position, v3i(0, 1, 0));
        assert_eq!(both[1].position, v3i(3, 1, 0));
        // 軸は A の回転で回る: A を z 周り 180° → world 軸 -x、perp は同じ y → 同結果
        let mut rot = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        rot[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        solve_slider_joint(&j, &mut rot, DT);
        assert_eq!(rot[1].position, v3i(3, 0, 0));
    }

    #[test]
    fn slider_joint_limits_clamp_travel_along_axis() {
        let j = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_limits(fi(-1), fi(2));
        // along 5 > max 2 → error 3、A static / B inv 1 → B -3 → (2,0,0)
        let mut over = pair(Vec3Fix::ZERO, 0, v3i(5, 0, 0), 1);
        solve_slider_joint(&j, &mut over, DT);
        assert_eq!(over[1].position, v3i(2, 0, 0));
        // along -4 < min -1 → error 3 → B +3 → (-1,0,0)
        let mut under = pair(Vec3Fix::ZERO, 0, v3i(-4, 0, 0), 1);
        solve_slider_joint(&j, &mut under, DT);
        assert_eq!(under[1].position, v3i(-1, 0, 0));
        // 両 dynamic inv 1/3、along 6 → error 4、λ 1 → A +1、B -3 → A (1,0,0)、B (3,0,0)
        let mut both = pair(Vec3Fix::ZERO, 1, v3i(6, 0, 0), 3);
        solve_slider_joint(&j, &mut both, DT);
        assert_eq!(both[0].position, v3i(1, 0, 0));
        assert_eq!(both[1].position, v3i(3, 0, 0));
        // 境界 (along == max) と範囲内は不変
        let mut edge = pair(Vec3Fix::ZERO, 0, v3i(2, 0, 0), 1);
        solve_slider_joint(&j, &mut edge, DT);
        assert_eq!(edge[1].position, v3i(2, 0, 0));
        let mut inside = pair(Vec3Fix::ZERO, 0, v3i(1, 0, 0), 1);
        solve_slider_joint(&j, &mut inside, DT);
        assert_eq!(inside[1].position, v3i(1, 0, 0));
    }

    // ---- spring --------------------------------------------------------

    #[test]
    fn spring_joint_force_is_stiffness_times_displacement_plus_damping() {
        // rest 1、k 2、c 0、距離 4 → F = 2*3 = 6、impulse = n*6*dt(1/4) = 1.5 → A +1.5 (inv 1)、B -4.5 (inv 3)
        let mut bodies = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        let j = SpringJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fi(1),
            fi(2),
            Fix128::ZERO,
        );
        solve_spring_joint(&j, &mut bodies, DT);
        assert_eq!(
            bodies[0].position,
            Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            bodies[1].position,
            Vec3Fix::new(Fix128::from_ratio(-1, 2), Fix128::ZERO, Fix128::ZERO)
        );
        // 圧縮 (距離 4 < rest 8) → 負の力で離れる: F = 2*(-4) = -8 → impulse -2 → A -2、B +2 (inv 1/1)
        let mut comp = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 1);
        let jc = SpringJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fi(8),
            fi(2),
            Fix128::ZERO,
        );
        solve_spring_joint(&jc, &mut comp, DT);
        assert_eq!(comp[0].position, v3i(-2, 0, 0));
        assert_eq!(comp[1].position, v3i(6, 0, 0));
        // damping: rest 4 (力 0)、c 2、B が +x に 3 で離れる → F = 2*3 = 6 → impulse 1.5 → A +1.5、B -1.5
        let mut damp = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 1);
        damp[1].velocity = v3i(3, 0, 0);
        let jd = SpringJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, fi(4), fi(2), fi(2));
        solve_spring_joint(&jd, &mut damp, DT);
        assert_eq!(
            damp[0].position,
            Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            damp[1].position,
            Vec3Fix::new(Fix128::from_ratio(5, 2), Fix128::ZERO, Fix128::ZERO)
        );
        // rest にあり速度 0 → 不変、一致点 → 不変
        let mut rest = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 1);
        solve_spring_joint(&jd, &mut rest, DT);
        assert_eq!(rest[1].position, v3i(4, 0, 0));
    }

    // ---- D6 ------------------------------------------------------------

    #[test]
    fn d6_joint_linear_locked_limited_and_free_axes() {
        let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        j.linear_x = D6Motion::Locked;
        j.linear_y = D6Motion::Limited; // [-1, 1]
        j.linear_z = D6Motion::Free;
        // B at (4, 3, 5): x locked → -4、y limited → -(3-1) = -2、z free → 0 ⇒ B (0, 1, 5) (A static / B inv 1)
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(4, 3, 5), 1);
        solve_d6_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[1].position, v3i(0, 1, 5));
        // y が下限側: (0, -6, 0) → error -6-(-1) = -5 → B +5 → (0,-1,0)
        let mut low = pair(Vec3Fix::ZERO, 0, v3i(0, -6, 0), 1);
        solve_d6_joint(&j, &mut low, DT);
        assert_eq!(low[1].position, v3i(0, -1, 0));
        // 両 dynamic inv 1/3、x locked、B (4,0,0) → λ 1 → A +1、B -3
        let mut both = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        solve_d6_joint(&j, &mut both, DT);
        assert_eq!(both[0].position, v3i(1, 0, 0));
        assert_eq!(both[1].position, v3i(1, 0, 0));
        // frame_a を z 周り 180° 回転 → world x 軸が反転しても locked の結果は同じ (符号が両方で反転)
        let mut jf = j;
        jf.local_frame_a = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let mut flipped = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        solve_d6_joint(&jf, &mut flipped, DT);
        assert_eq!(flipped[1].position, Vec3Fix::ZERO);
    }

    #[test]
    fn d6_joint_angular_locked_axis_pulls_rotation_back() {
        let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        j.angular_z = D6Motion::Locked;
        let angle = |b: &[RigidBody]| {
            let rel = b[1].rotation.mul(b[0].rotation.conjugate());
            compute_twist_angle(rel, Vec3Fix::UNIT_Z)
        };
        let mut bodies = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 2));
        let mut prev = angle(&bodies).abs();
        for i in 0..30 {
            solve_d6_joint(&j, &mut bodies, DT);
            let a = angle(&bodies).abs();
            assert!(a <= prev, "iter {i}");
            prev = a;
        }
        assert!(prev < Fix128::from_ratio(1, 1000));
        // Limited [-0.25, 0.25] で 0.5 → 減る、0.1 → 不変、Free → 不変
        let mut jl = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        jl.angular_z = D6Motion::Limited;
        jl.angular_limit_min = Vec3Fix::new(
            Fix128::from_ratio(-1, 4),
            Fix128::from_ratio(-1, 4),
            Fix128::from_ratio(-1, 4),
        );
        jl.angular_limit_max = Vec3Fix::new(
            Fix128::from_ratio(1, 4),
            Fix128::from_ratio(1, 4),
            Fix128::from_ratio(1, 4),
        );
        let mut over = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        over[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 2));
        let o0 = angle(&over);
        solve_d6_joint(&jl, &mut over, DT);
        assert!(angle(&over) < o0);
        let mut inside = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        inside[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 10));
        let r0 = inside[1].rotation;
        solve_d6_joint(&jl, &mut inside, DT);
        assert_eq!(inside[1].rotation, r0);
        let free = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        let mut f = pair(Vec3Fix::ZERO, 0, v3i(4, 3, 5), 1);
        f[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 2));
        let (p0, q0) = (f[1].position, f[1].rotation);
        solve_d6_joint(&free, &mut f, DT);
        assert_eq!((f[1].position, f[1].rotation), (p0, q0));
    }

    // ---- cone twist ----------------------------------------------------

    #[test]
    fn cone_twist_joint_enforces_cone_and_twist_limits() {
        // twist 軸 z、cone limit 0.25 rad: B を x 周り 1 rad 傾ける → cone 角 1 > 0.25 → 減る
        let cone_angle = |b: &[RigidBody]| {
            let a = b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            let c = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            Fix128::atan2(a.cross(c).length(), a.dot(c))
        };
        let j = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(Fix128::from_ratio(1, 4), Fix128::from_ratio(1, 4));
        let mut tilted = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        tilted[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::ONE);
        let c0 = cone_angle(&tilted);
        assert!(near(c0, Fix128::ONE), "{c0:?}");
        solve_cone_twist_joint(&j, &mut tilted, DT);
        let c1 = cone_angle(&tilted);
        assert!(c1 < c0, "{c0:?} -> {c1:?}");
        // 反復で cone limit 近傍まで単調に戻る
        let mut prev = c1;
        for _ in 0..40 {
            solve_cone_twist_joint(&j, &mut tilted, DT);
            let c = cone_angle(&tilted);
            assert!(c <= prev + Fix128 { hi: 0, lo: 1 << 30 });
            prev = c;
        }
        assert!(
            prev < Fix128::from_ratio(1, 4) + Fix128::from_ratio(1, 50),
            "{prev:?}"
        );
        // twist: z 周り 1 rad (cone 0) → twist 1 > 0.25 → 減る
        let twist = |b: &[RigidBody]| {
            let rel = b[1].rotation.mul(b[0].rotation.conjugate());
            compute_twist_angle(rel, Vec3Fix::UNIT_Z)
        };
        let mut twisted = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        twisted[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE);
        let t0 = twist(&twisted);
        solve_cone_twist_joint(&j, &mut twisted, DT);
        assert!(twist(&twisted) < t0);
        // 範囲内 (cone 0.1、twist 0.1) は不変、位置部は ball と同じ
        let mut inside = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        inside[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(1, 10));
        let r0 = inside[1].rotation;
        solve_cone_twist_joint(&j, &mut inside, DT);
        assert_eq!(inside[1].rotation, r0);
        let mut pos = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 3);
        solve_cone_twist_joint(&j, &mut pos, DT);
        assert_eq!(pos[0].position, v3i(1, 0, 0));
        assert_eq!(pos[1].position, v3i(1, 0, 0));
    }

    // ---- Joint enum: bodies / compute_force / breakable ----------------

    #[test]
    fn joint_bodies_and_compute_force_per_kind() {
        let bodies = pair(Vec3Fix::ZERO, 1, v3i(3, 4, 0), 1); // 距離 5
        let ball = Joint::Ball(BallJoint::new(2, 7, Vec3Fix::ZERO, Vec3Fix::ZERO));
        assert_eq!(ball.bodies(), (2, 7));
        let ball01 = Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO));
        assert_eq!(ball01.compute_force(&bodies), fi(5));
        let hinge = Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        ));
        assert_eq!(hinge.compute_force(&bodies), fi(5));
        let fixed = Joint::Fixed(FixedJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        assert_eq!(fixed.compute_force(&bodies), fi(5));
        // slider (軸 x): perp = (0,4,0) → 4
        let slider = Joint::Slider(SliderJoint::new(
            0,
            1,
            Vec3Fix::UNIT_X,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ));
        assert_eq!(slider.compute_force(&bodies), fi(4));
        // spring: |k (dist - rest)| = |2 (5 - 8)| = 6
        let spring = Joint::Spring(SpringJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fi(8),
            fi(2),
            Fix128::ZERO,
        ));
        assert_eq!(spring.compute_force(&bodies), fi(6));
        let d6 = Joint::D6(D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO));
        assert_eq!(d6.compute_force(&bodies), fi(5));
        let ct = Joint::ConeTwist(ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        ));
        assert_eq!(ct.compute_force(&bodies), fi(5));
        // anchor が効く: A anchor (3,4,0) なら距離 0
        let anchored = Joint::Ball(BallJoint::new(0, 1, v3i(3, 4, 0), Vec3Fix::ZERO));
        assert_eq!(anchored.compute_force(&bodies), Fix128::ZERO);
    }

    #[test]
    fn solve_joints_breakable_breaks_strictly_above_threshold_and_skips_broken() {
        // 距離 5 の ball joint: break 5 → 壊れない (`>` は false)、break 4 → 壊れて解かれない
        let mk = |bf: i64| {
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(fi(bf)))
        };
        let mut hold = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let broken = solve_joints_breakable(&[mk(5)], &mut hold, DT);
        assert!(broken.is_empty());
        assert!(
            near_v(hold[1].position, Vec3Fix::ZERO),
            "解かれて A に一致 {:?}",
            hold[1].position
        );
        let mut snap = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let broken2 = solve_joints_breakable(&[mk(4)], &mut snap, DT);
        assert_eq!(broken2, vec![0]);
        assert_eq!(snap[1].position, v3i(3, 4, 0), "壊れた joint は解かれない");
        // 複数: index 降順で返る、壊れていない方は解かれる
        let mut multi = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let js = [mk(4), mk(100), mk(1)];
        let broken3 = solve_joints_breakable(&js, &mut multi, DT);
        assert_eq!(broken3, vec![2, 0]);
        assert!(near_v(multi[1].position, Vec3Fix::ZERO));
        // solve_joints (非 breakable) は全部解く
        let mut all = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        solve_joints(&[mk(1)], &mut all, DT);
        assert!(near_v(all[1].position, Vec3Fix::ZERO));
    }
}
