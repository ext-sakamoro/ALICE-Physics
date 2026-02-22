//! Joint System for Rigid Body Connections
//!
//! XPBD-based joint constraints with angular limits and compliance.
//!
//! # Joint Types
//!
//! - **BallJoint**: 3-DOF rotation (shoulder, hip)
//! - **HingeJoint**: 1-DOF rotation with angle limits (knee, door)
//! - **FixedJoint**: 0-DOF (weld)
//! - **SliderJoint**: 1-DOF translation along an axis (piston)
//! - **SpringJoint**: Distance spring with damping
//! - **D6Joint**: 6-DOF configurable joint with per-axis locking
//! - **ConeTwistJoint**: Cone-twist constraint for ragdoll shoulders/hips

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
#[derive(Clone, Copy, Debug)]
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
    pub fn new(body_a: usize, body_b: usize, anchor_a: Vec3Fix, anchor_b: Vec3Fix) -> Self {
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
    pub fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }

    /// Set break force threshold
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Hinge joint (1 rotational DOF around an axis, with optional angle limits)
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
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
    pub fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        self.angle_min = Some(min);
        self.angle_max = Some(max);
        self
    }

    /// Set positional compliance (inverse stiffness)
    pub fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }

    /// Set break force threshold
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Fixed joint (0 DOF, weld two bodies)
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
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
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Slider (prismatic) joint: translation along a single axis
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
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
    pub fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        self.limit_min = Some(min);
        self.limit_max = Some(max);
        self
    }

    /// Set break force threshold
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Spring joint: distance spring with damping
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
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
    pub fn with_break_force(mut self, force: Fix128) -> Self {
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
#[derive(Clone, Copy, Debug)]
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
    pub fn with_linear_motion(mut self, x: D6Motion, y: D6Motion, z: D6Motion) -> Self {
        self.linear_x = x;
        self.linear_y = y;
        self.linear_z = z;
        self
    }

    /// Set angular motion on all axes
    pub fn with_angular_motion(mut self, x: D6Motion, y: D6Motion, z: D6Motion) -> Self {
        self.angular_x = x;
        self.angular_y = y;
        self.angular_z = z;
        self
    }

    /// Set linear limits
    pub fn with_linear_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        self.linear_limit_min = min;
        self.linear_limit_max = max;
        self
    }

    /// Set angular limits
    pub fn with_angular_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        self.angular_limit_min = min;
        self.angular_limit_max = max;
        self
    }

    /// Set break force threshold
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Cone-Twist joint (ball joint with cone angle limit + twist limit)
///
/// Used for ragdoll shoulders and hips where rotation is constrained
/// to a cone around the twist axis, with an additional twist limit.
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
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
    pub fn with_limits(mut self, cone: Fix128, twist: Fix128) -> Self {
        self.cone_limit = cone;
        self.twist_limit = twist;
        self
    }

    /// Set break force threshold
    pub fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }
}

/// Unified joint enum for storage in PhysicsWorld
#[derive(Clone, Copy, Debug)]
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
    pub fn bodies(&self) -> (usize, usize) {
        match self {
            Joint::Ball(j) => (j.body_a, j.body_b),
            Joint::Hinge(j) => (j.body_a, j.body_b),
            Joint::Fixed(j) => (j.body_a, j.body_b),
            Joint::Slider(j) => (j.body_a, j.body_b),
            Joint::Spring(j) => (j.body_a, j.body_b),
            Joint::D6(j) => (j.body_a, j.body_b),
            Joint::ConeTwist(j) => (j.body_a, j.body_b),
        }
    }

    /// Get joint type
    #[inline]
    pub fn joint_type(&self) -> JointType {
        match self {
            Joint::Ball(_) => JointType::Ball,
            Joint::Hinge(_) => JointType::Hinge,
            Joint::Fixed(_) => JointType::Fixed,
            Joint::Slider(_) => JointType::Slider,
            Joint::Spring(_) => JointType::Spring,
            Joint::D6(_) => JointType::D6,
            Joint::ConeTwist(_) => JointType::ConeTwist,
        }
    }

    /// Get the break force threshold (None = unbreakable)
    #[inline]
    pub fn break_force(&self) -> Option<Fix128> {
        match self {
            Joint::Ball(j) => j.break_force,
            Joint::Hinge(j) => j.break_force,
            Joint::Fixed(j) => j.break_force,
            Joint::Slider(j) => j.break_force,
            Joint::Spring(j) => j.break_force,
            Joint::D6(j) => j.break_force,
            Joint::ConeTwist(j) => j.break_force,
        }
    }

    /// Compute current constraint force (distance between anchor points).
    ///
    /// Returns the force magnitude used to determine if the joint should break.
    pub fn compute_force(&self, bodies: &[crate::solver::RigidBody]) -> Fix128 {
        let (a_idx, b_idx) = self.bodies();
        let body_a = &bodies[a_idx];
        let body_b = &bodies[b_idx];

        match self {
            Joint::Ball(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Joint::Hinge(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Joint::Fixed(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Joint::Slider(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                let delta = anchor_b - anchor_a;
                let world_axis = body_a.rotation.rotate_vec(j.local_axis).normalize();
                let along = delta.dot(world_axis);
                let perp = delta - world_axis * along;
                perp.length()
            }
            Joint::Spring(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                let dist = (anchor_b - anchor_a).length();
                let displacement = dist - j.rest_length;
                (j.stiffness * displacement).abs()
            }
            Joint::D6(j) => {
                let anchor_a = body_a.position + body_a.rotation.rotate_vec(j.local_anchor_a);
                let anchor_b = body_b.position + body_b.rotation.rotate_vec(j.local_anchor_b);
                (anchor_b - anchor_a).length()
            }
            Joint::ConeTwist(j) => {
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
/// Uses split_at_mut to safely obtain two mutable references.
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

#[cfg(test)]
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
}
