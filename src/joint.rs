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

/// `a > b` for `Fix128`, usable in a `const fn` (the `PartialOrd` impl is
/// not `const`). `Fix128` is `(hi << 64) | lo` as a signed two's-complement
/// 128-bit integer (src/math.rs), so comparing the high word first and
/// breaking ties on the low word is exact.
const fn fix128_min_max_violated(min: Fix128, max: Fix128) -> bool {
    min.hi > max.hi || (min.hi == max.hi && min.lo > max.lo)
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
    ///
    /// # Panics
    ///
    /// When `min > max`. `min == max` is allowed (a locked joint).
    #[must_use]
    pub const fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        if fix128_min_max_violated(min, max) {
            panic!("HingeJoint::with_limits: min must not exceed max");
        }
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
    ///
    /// # Panics
    ///
    /// When `min > max`. `min == max` is allowed (a locked joint).
    #[must_use]
    pub const fn with_limits(mut self, min: Fix128, max: Fix128) -> Self {
        if fix128_min_max_violated(min, max) {
            panic!("SliderJoint::with_limits: min must not exceed max");
        }
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

/// Spring joint: a spring-damper between two anchor points
///
/// The joint applies the force (on B; A receives the opposite)
///
/// - `rest_length == 0`: `F = −k d − c u`, with `d = anchor_b − anchor_a`
///   and `u = d'` the relative anchor velocity. Spring and damper are both
///   isotropic, so the damper removes the tangential relative velocity as
///   well as the radial one and a body held by the spring does not orbit
///   its anchor.
/// - `rest_length > 0`: `F = −(k (|d| − L) + c (u · n)) n`, `n = d / |d|`,
///   the usual spring with a dashpot along its axis. The direction across
///   the axis is free (a pendulum stays a pendulum), as for a physical
///   spring-damper element; use `rest_length = 0` to hold a body at a point.
///
/// For a body of mass `m` held against a static or kinematic anchor (or
/// two bodies with `m_eff = 1 / (1/m_a + 1/m_b)`), angular frequency `ω`
/// and damping ratio `ζ`, use `stiffness = m_eff ω²` and
/// `damping = 2 ζ m_eff ω`; `ζ = 1` (`damping = 2 √(k m_eff)`) is critical
/// damping, `x(t) = (x0 + (v0 + ω x0) t) e^(−ωt)` with no overshoot.
///
/// It is solved once per substep as an XPBD constraint with damping
/// (Macklin, Müller, Chentanez 2016, eq. 26), which is a backward Euler step
/// of the force law: the result converges to the same continuous solution
/// for any `substeps`, does not depend on `iterations`, and is stable for
/// any `k` and `c`. Each joint carries its own `stiffness` / `damping`, so
/// springs of different `ω` can share one world. `SolverConfig::damping`
/// and the per-body damping factors are applied on top, once per frame.
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
    /// Rest length `L` of the spring in metres. `0` makes the spring and the
    /// damper isotropic (see the type doc)
    pub rest_length: Fix128,
    /// Spring stiffness `k` in newtons per metre (`m_eff ω²` for angular
    /// frequency `ω`). Independent of the step length and of `substeps`
    pub stiffness: Fix128,
    /// Damping coefficient `c` in newton-seconds per metre, acting on the
    /// relative anchor velocity (every component for `rest_length = 0`, the
    /// component along the spring otherwise). Critical damping is
    /// `c = 2 √(k m_eff)`
    pub damping: Fix128,
    /// Maximum spring force `|k (|d| − L)|` in newtons before the joint
    /// breaks (None = unbreakable), see [`solve_joints_breakable`]
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
    ///
    /// # Panics
    ///
    /// When any axis has `min > max`. `min == max` is allowed (a locked
    /// axis).
    #[must_use]
    pub const fn with_linear_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        if fix128_min_max_violated(min.x, max.x)
            || fix128_min_max_violated(min.y, max.y)
            || fix128_min_max_violated(min.z, max.z)
        {
            panic!("D6Joint::with_linear_limits: min must not exceed max on any axis");
        }
        self.linear_limit_min = min;
        self.linear_limit_max = max;
        self
    }

    /// Set angular limits
    ///
    /// # Panics
    ///
    /// When any axis has `min > max`. `min == max` is allowed (a locked
    /// axis).
    #[must_use]
    pub const fn with_angular_limits(mut self, min: Vec3Fix, max: Vec3Fix) -> Self {
        if fix128_min_max_violated(min.x, max.x)
            || fix128_min_max_violated(min.y, max.y)
            || fix128_min_max_violated(min.z, max.z)
        {
            panic!("D6Joint::with_angular_limits: min must not exceed max on any axis");
        }
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
    ///
    /// # Panics
    ///
    /// When `cone` or `twist` is negative. `0` is allowed (no rotation on
    /// that axis).
    #[must_use]
    pub const fn with_limits(mut self, cone: Fix128, twist: Fix128) -> Self {
        if cone.is_negative() {
            panic!("ConeTwistJoint::with_limits: cone must not be negative");
        }
        if twist.is_negative() {
            panic!("ConeTwistJoint::with_limits: twist must not be negative");
        }
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

    /// Current anchor separation of the joint's positional row (metres).
    ///
    /// For the spring joint this is the spring force `k · |d − rest|` (N); for
    /// every other joint it is a length, not a force: the gap between the two
    /// anchors (the slider measures only the part off its axis). Breaking uses
    /// the reaction force of the solve ([`solve_joints_with_reaction_forces`]),
    /// which is a force for every joint.
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
///
/// # Joint compliance contract
///
/// - One call is one XPBD solve of every joint starting from `λ = 0`
///   (Macklin, Müller, Chentanez, MIG 2016, eq. 18): a compliant row moves by
///   `Δλ = −C / (w + α̃)` with `α̃ = compliance / dt²`.
/// - It is meant to be called once per substep, with `dt` the substep length.
/// - Calling it several times at the same `dt` is not equivalent to a longer
///   solve: no `λ` is carried between calls, so each call removes the fraction
///   `w / (w + α̃)` of the gap again and leaves `d · (α̃ / (w + α̃))^k` after `k`
///   calls. The effective stiffness therefore grows with the number of calls.
pub fn solve_joints(joints: &[Joint], bodies: &mut [crate::solver::RigidBody], dt: Fix128) {
    for joint in joints {
        let _ = solve_one(joint, bodies, dt);
    }
}

/// Solve one joint. A joint whose two bodies are the same body constrains
/// nothing a motion could satisfy (a body cannot move relative to itself), so
/// it is skipped and the body stays a free body; solving it would apply both
/// sides' corrections to that one body and inject motion.
/// Solve one joint and return the linear impulse of its positional rows,
/// `Σ λ · n` (Macklin, Müller, Chentanez, MIG 2016: the row multiplier `λ`
/// along its unit direction `n`; body A moved by `+w_a λ n`, body B by
/// `−w_b λ n`). Angular rows are not included.
fn solve_one(joint: &Joint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) -> Vec3Fix {
    let (a, b) = joint.bodies();
    if a == b {
        return Vec3Fix::ZERO;
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

/// Solve joints and return indices of joints that should be removed (broken),
/// comparing each joint's **anchor separation** with its `break_force`.
///
/// Deprecated: the threshold here is the length [`Joint::compute_force`]
/// returns (metres, `k · x` for a spring), not a force, although
/// `break_force` is documented as a force. Use
/// [`solve_joints_breaking_on_force`], which compares the reaction force in
/// newtons, as [`crate::solver::PhysicsWorld::step`] does.
///
/// Checks each breakable joint's separation BEFORE solving.
/// If the force exceeds the threshold, the joint is marked as broken
/// and skipped during solving. Returns indices of broken joints in
/// descending order (safe for sequential removal).
///
/// # Joint compliance contract
///
/// - One call is one XPBD solve of every joint starting from `λ = 0`
///   (Macklin, Müller, Chentanez, MIG 2016, eq. 18): a compliant row moves by
///   `Δλ = −C / (w + α̃)` with `α̃ = compliance / dt²`.
/// - It is meant to be called once per substep, with `dt` the substep length.
/// - Calling it several times at the same `dt` is not equivalent to a longer
///   solve: no `λ` is carried between calls, so each call removes the fraction
///   `w / (w + α̃)` of the gap again and leaves `d · (α̃ / (w + α̃))^k` after `k`
///   calls. The effective stiffness therefore grows with the number of calls.
#[deprecated(
    since = "2.2.0",
    note = "the threshold is the anchor separation (m), not a force; use solve_joints_breaking_on_force, whose break_force is the reaction force λ/dt² in N"
)]
pub fn solve_joints_breakable(
    joints: &[Joint],
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec<usize> {
    // Check which joints exceeded their break force BEFORE solving
    let mut broken = Vec::new();
    for (i, joint) in joints.iter().enumerate() {
        let (a, b) = joint.bodies();
        if a == b {
            // a joint on one body constrains nothing, so it carries no force
            continue;
        }
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
        let _ = solve_one(joint, bodies, dt);
    }

    // Sort descending for safe removal
    broken.sort_unstable_by(|a, b| b.cmp(a));
    broken
}

/// Solve every joint once (as [`solve_joints`]) and return each joint's
/// reaction force in newtons, in joint order.
///
/// The force of a joint is the length of the linear impulse its positional
/// rows applied in this solve divided by `dt²`: `|Σ λ · n| / dt²`, where `λ` is
/// the XPBD multiplier of a row along its unit direction `n` (Macklin, Müller,
/// Chentanez, MIG 2016, eq. 10 and 18). It is the force the solve actually
/// transmitted, so a joint further down a chain carries the load passed to it
/// through the joints above in the same sweep. A body of mass `m` hung at rest
/// from a static anchor by a rigid ball joint reports `m · g` per substep.
///
/// - Angular rows (hinge axis, fixed orientation, cone and twist limits, D6
///   angular axes) produce torques and are not counted.
/// - A joint on one body, or between two bodies that cannot move, applies
///   nothing and reports 0.
/// - `dt` is the substep length the joints are solved at.
/// - The force is what the solve actually transmitted, transient included:
///   joints are solved once per substep in one Gauss–Seidel sweep, so a chain
///   starting from rest overshoots its static load before settling (2 kg +
///   2 kg hung from a static anchor, `g = 8`: lower joint 8, 16, 20, 20, 18,
///   16 N, upper 16, 32, 40, 40, 36, 32 N, the same on both backends and for
///   any `substeps` / `iterations`). A `break_force` between the static load
///   and that peak breaks during the start-up; set it above the peak.
// LIMITATION(COV-RIGID-032): angular rows produce torques and are not reported; only the positional reaction force is.
pub fn solve_joints_with_reaction_forces(
    joints: &[Joint],
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec<Fix128> {
    let h2 = dt * dt;
    joints
        .iter()
        .map(|joint| {
            let impulse = solve_one(joint, bodies, dt);
            if h2.is_zero() {
                Fix128::ZERO
            } else {
                impulse.length() / h2
            }
        })
        .collect()
}

/// Solve joints once and return the indices of those whose reaction force
/// exceeded their `break_force` (N), in descending order (safe for sequential
/// `swap_remove`).
///
/// Every joint is solved first ([`solve_joints_with_reaction_forces`]); a
/// joint is broken when its reaction force from that solve is strictly
/// greater than its `break_force`. The broken joint's correction from this
/// solve has been applied: the caller removes it so that it no longer acts on
/// later solves. This is the check [`crate::solver::PhysicsWorld::step`]
/// applies to the world's joints after each substep's joint solve.
pub fn solve_joints_breaking_on_force(
    joints: &[Joint],
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec<usize> {
    let forces = solve_joints_with_reaction_forces(joints, bodies, dt);
    let mut broken: Vec<usize> = joints
        .iter()
        .zip(&forces)
        .enumerate()
        .filter(|(_, (joint, force))| joint.break_force().is_some_and(|max| **force > max))
        .map(|(i, _)| i)
        .collect();
    broken.reverse();
    broken
}

/// Solve ball joint: constrain anchor points to coincide
fn solve_ball_joint(
    joint: &BallJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // World-space anchor positions and lever arms (COM → anchor)
    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let anchor_a = body_a.position + r_a;
    let anchor_b = body_b.position + r_b;

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if distance.is_zero() {
        return impulse;
    }

    let compliance_term = joint.compliance / (dt * dt);
    let w_sum = point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, normal) + compliance_term;

    if w_sum.is_zero() {
        return impulse;
    }

    let lambda = distance / w_sum;
    apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, normal, lambda);
    impulse = impulse + normal * lambda;
    impulse
}

/// Solve hinge joint: positional + angular constraint along axis
///
/// # Claims
/// - `angle_min` and `angle_max` are enforced independently: `None` leaves that side unbounded
/// - A violated side moves the relative twist angle (radians) back onto the bound, split by inverse inertia
/// - With both `None` no angle limit is applied
fn solve_hinge_joint(
    joint: &HingeJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint (same as ball joint)
    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let anchor_a = body_a.position + r_a;
    let anchor_b = body_b.position + r_b;

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum =
            point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, normal) + compliance_term;

        if !w_sum.is_zero() {
            let lambda = distance / w_sum;
            apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, normal, lambda);
            impulse = impulse + normal * lambda;
        }
    }

    // 2. Angular constraint: align axes
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];
    let world_axis_a = body_a.rotation.rotate_vec(joint.local_axis_a);
    let world_axis_b = body_b.rotation.rotate_vec(joint.local_axis_b);

    let axis_error = world_axis_a.cross(world_axis_b);
    let (correction_axis, sin_err) = axis_error.normalize_with_length();

    if !sin_err.is_zero() {
        // true misalignment angle, not its sine (exact in one rigid solve)
        let error_mag = Fix128::atan2(sin_err, world_axis_a.dot(world_axis_b));
        let angular_compliance = joint.angular_compliance / (dt * dt);
        let w_ang =
            angular_w_sum(bodies, joint.body_a, joint.body_b, correction_axis) + angular_compliance;

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

    // 3. Angle limits (rigid: no angular compliance, a limit is a stop)
    if joint.angle_min.is_some() || joint.angle_max.is_some() {
        let body_a = bodies[joint.body_a];
        let body_b = bodies[joint.body_b];
        let world_axis_a = body_a.rotation.rotate_vec(joint.local_axis_a);

        // Compute relative angle around hinge axis
        let rel_quat = body_b.rotation.mul(body_a.rotation.conjugate());
        let angle = compute_twist_angle(rel_quat, world_axis_a);

        if let Some(min_angle) = joint.angle_min.filter(|m| angle < *m) {
            let error = min_angle - angle;
            let w_ang = angular_w_sum(bodies, joint.body_a, joint.body_b, world_axis_a);
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
        } else if let Some(max_angle) = joint.angle_max.filter(|m| angle > *m) {
            let error = angle - max_angle;
            let w_ang = angular_w_sum(bodies, joint.body_a, joint.body_b, world_axis_a);
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
    impulse
}

/// Solve fixed joint: positional + full rotational lock
fn solve_fixed_joint(
    joint: &FixedJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint
    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let anchor_a = body_a.position + r_a;
    let anchor_b = body_b.position + r_b;

    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum =
            point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, normal) + compliance_term;

        if !w_sum.is_zero() {
            let lambda = distance / w_sum;
            apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, normal, lambda);
            impulse = impulse + normal * lambda;
        }
    }

    // 2. Rotational constraint: maintain relative rotation
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];
    let target_rot_b = body_a.rotation.mul(joint.relative_rotation);
    let rot_error = body_b.rotation.mul(target_rot_b.conjugate());

    // Extract error as rotation vector (axis * angle)
    let error_vec = Vec3Fix::new(rot_error.x, rot_error.y, rot_error.z);
    let (correction_axis, sin_half) = error_vec.normalize_with_length();

    if !sin_half.is_zero() {
        // rotation angle of the error quaternion (shortest arc: flip to w ≥ 0)
        let (sin_half, cos_half) = if rot_error.w.is_negative() {
            (-sin_half, -rot_error.w)
        } else {
            (sin_half, rot_error.w)
        };
        let error_mag = Fix128::atan2(sin_half, cos_half).double();
        let angular_compliance = joint.angular_compliance / (dt * dt);
        let w_ang =
            angular_w_sum(bodies, joint.body_a, joint.body_b, correction_axis) + angular_compliance;

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
    impulse
}

/// Solve slider joint: constrain to 1-DOF translation along axis
///
/// # Claims
/// - `limit_min` and `limit_max` are enforced independently: `None` leaves that side unbounded
/// - A violated side moves the axial offset (metres) back onto the bound, split by inverse mass
/// - With both `None` no translation limit is applied
fn solve_slider_joint(
    joint: &SliderJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    let world_axis = body_a.rotation.rotate_vec(joint.local_axis).normalize();

    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let anchor_a = body_a.position + r_a;
    let anchor_b = body_b.position + r_b;

    let delta = anchor_b - anchor_a;

    // Project delta onto axis
    let along_axis = delta.dot(world_axis);

    // Perpendicular error (must be zero)
    let perp = delta - world_axis * along_axis;
    let (perp_normal, perp_dist) = perp.normalize_with_length();

    if !perp_dist.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum = point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, perp_normal)
            + compliance_term;

        if !w_sum.is_zero() {
            let lambda = perp_dist / w_sum;
            apply_point_correction(
                bodies,
                joint.body_a,
                joint.body_b,
                r_a,
                r_b,
                perp_normal,
                lambda,
            );
            impulse = impulse + perp_normal * lambda;
        }
    }

    // Enforce translation limits
    if joint.limit_min.is_some() || joint.limit_max.is_some() {
        if let Some(min_d) = joint.limit_min.filter(|m| along_axis < *m) {
            let error = min_d - along_axis;
            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if !w_sum.is_zero() {
                let inv_w_sum = Fix128::ONE / w_sum;
                let correction = world_axis * (error * inv_w_sum);
                // A moves by −correction, the point-row convention with λ = −error / w
                impulse = impulse - correction;
                if !body_a.inv_mass.is_zero() {
                    bodies[joint.body_a].position =
                        bodies[joint.body_a].position - correction * body_a.inv_mass;
                }
                if !body_b.inv_mass.is_zero() {
                    bodies[joint.body_b].position =
                        bodies[joint.body_b].position + correction * body_b.inv_mass;
                }
            }
        } else if let Some(max_d) = joint.limit_max.filter(|m| along_axis > *m) {
            let error = along_axis - max_d;
            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if !w_sum.is_zero() {
                let inv_w_sum = Fix128::ONE / w_sum;
                let correction = world_axis * (error * inv_w_sum);
                impulse = impulse + correction;
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
    impulse
}

/// Solve spring joint: XPBD with damping (Macklin, Müller, Chentanez 2016,
/// eq. 26), one solve per substep starting from `λ = 0`
///
/// With `h = dt`, `α̃ = 1/(k h²)` and `γ = c/(k h)`, a constraint row with
/// gradient `n`, error `C` and displacement `∇C · (x − x_n) = h (u · n)` moves
/// by `Δλ = −(C + γ h (u · n)) / ((1 + γ) w + α̃)`. Multiplied through by
/// `k h²`, so that `k = 0` (a pure damper) needs no division:
///
/// `λ = h² (k C + c (u · n)) / (1 + (k h² + c h) w)`
///
/// with `w` the generalised inverse mass along `n` (lever arms included,
/// see `point_w_sum`). Body A moves by `+w_a λ n`, body B by `−w_b λ n`.
///
/// - `rest_length == 0`: the three rows `C = d` (gradient `I`) are solved as
///   one row along `n = (k d + c u) / |k d + c u|`, exact when the anchors
///   are on the centres of mass (the inverse mass is isotropic). This is
///   the vector form of the force `−k d − c u`.
/// - `rest_length > 0`: the single row `C = |d| − L`, `n = d / |d|`.
///
/// `u` is the relative anchor velocity `(v_b + ω_b × r_b) − (v_a + ω_a × r_a)`
/// of the substep: the predicted velocity in the XPBD substep and the
/// velocity the TGS substep advanced the positions with, so that
/// `h u = x − x_n` in both. The velocity update that follows the solve
/// (`v = (x − x_n) / h`, or the TGS carry `v += Δx / h`) then makes one
/// substep the backward Euler step of `m x'' = F`. For a single body of
/// mass `m` against a static anchor, with `a = ω h`, `k = m ω²` and
/// `c = 2 ζ m ω`, the map is `x' = (x (1 + 2ζa) + h v) / (1 + 2ζa + a²)`.
///
/// # Claims
/// - The effective stiffness is `k` [N/m] and the damping `c` [N·s/m], for any
///   `dt`; a hanging body settles at the static extension `m g / k` exactly
/// - `rest_length = 0`: the damper acts on every component of the relative
///   velocity, so tangential motion decays like radial motion (no orbit)
/// - `rest_length > 0`: the damper acts along the spring only
/// - Two static bodies, or a coincident anchor pair at rest with
///   `rest_length > 0` (no direction), leave both bodies unchanged
/// - Unconditionally stable and free of overshoot for `ζ >= 1`
fn solve_spring_joint(
    joint: &SpringJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];
    if body_a.inv_mass.is_zero() && body_b.inv_mass.is_zero() {
        return impulse;
    }

    // World-space lever arms (COM → anchor) and anchors
    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let delta = (body_b.position + r_b) - (body_a.position + r_a);

    // Relative anchor velocity of the substep
    let vel_a = body_a.velocity + body_a.angular_velocity.cross(r_a);
    let vel_b = body_b.velocity + body_b.angular_velocity.cross(r_b);
    let rel_vel = vel_b - vel_a;

    let k = joint.stiffness;
    let c = joint.damping;
    let h2 = dt * dt;

    // `force` is the scalar `k C + c (u · n)` along `normal` (B is pulled
    // toward A for a positive value)
    let (normal, force) = if joint.rest_length.is_zero() {
        // Vector constraint C = d: the row along the force direction
        let (normal, force) = (delta * k + rel_vel * c).normalize_with_length();
        if force.is_zero() {
            return impulse;
        }
        (normal, force)
    } else {
        let (normal, distance) = delta.normalize_with_length();
        if distance.is_zero() {
            return impulse;
        }
        let displacement = distance - joint.rest_length;
        (normal, k * displacement + c * rel_vel.dot(normal))
    };

    let w = point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, normal);
    let denom = Fix128::ONE + (k * h2 + c * dt) * w;
    if denom.is_zero() {
        return impulse;
    }
    let lambda = force * h2 / denom;
    apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, normal, lambda);
    impulse = impulse + normal * lambda;
    impulse
}

/// Solve D6 joint: per-axis locking/limiting for all 6 DOF
#[allow(clippy::too_many_lines)]
fn solve_d6_joint(joint: &D6Joint, bodies: &mut [crate::solver::RigidBody], dt: Fix128) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // World-space anchors (position gap used for `proj` below is computed
    // once up front and reused across the sequential per-axis corrections,
    // same pre-existing Gauss-Seidel approximation as before this fix)
    let anchor_a = body_a.position + body_a.rotation.rotate_vec(joint.local_anchor_a);
    let anchor_b = body_b.position + body_b.rotation.rotate_vec(joint.local_anchor_b);
    let delta = anchor_b - anchor_a;

    // Get local frame axes in world space
    let frame_a = body_a.rotation.mul(joint.local_frame_a);
    let axis_x = frame_a.rotate_vec(Vec3Fix::UNIT_X);
    let axis_y = frame_a.rotate_vec(Vec3Fix::UNIT_Y);
    let axis_z = frame_a.rotate_vec(Vec3Fix::UNIT_Z);

    let compliance_term = joint.compliance / (dt * dt);

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
            // Lever arms are recomputed from each body's CURRENT rotation on every
            // iteration: an earlier axis in this same loop can rotate a body (the
            // lever-arm split this fix adds), and the arm is orientation-only, so a
            // value cached before the loop would go stale as soon as that happens.
            let r_a = bodies[joint.body_a]
                .rotation
                .rotate_vec(joint.local_anchor_a);
            let r_b = bodies[joint.body_b]
                .rotation
                .rotate_vec(joint.local_anchor_b);
            let w_sum =
                point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, axis) + compliance_term;
            if !w_sum.is_zero() {
                let lambda = error / w_sum;
                apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, axis, lambda);
                impulse = impulse + axis * lambda;
            }
        }
    }

    // Angular constraints per axis (w is axis-dependent: n · I⁻¹ n)
    //
    // Zero error is where each body's *frame* (body rotation composed with
    // its `local_frame_*`) lines up with the other's, not where the raw
    // body rotations coincide: `rel_quat` is frame_b_world ⊗ frame_a_world⁻¹
    // so a non-identity `local_frame_b` shifts the pose B is pulled toward.
    let angular_compliance = joint.angular_compliance / (dt * dt);
    {
        let frame_a_now = bodies[joint.body_a].rotation.mul(joint.local_frame_a);
        let frame_b_now = bodies[joint.body_b].rotation.mul(joint.local_frame_b);
        let rel_quat = frame_b_now.mul(frame_a_now.conjugate());

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
                let w_ang =
                    angular_w_sum(bodies, joint.body_a, joint.body_b, axis) + angular_compliance;
                if !w_ang.is_zero() {
                    apply_angular_correction(
                        bodies,
                        joint.body_a,
                        joint.body_b,
                        axis,
                        error / w_ang,
                    );
                }
            }
        }
    }
    impulse
}

/// Solve cone-twist joint: positional + cone + twist constraints
fn solve_cone_twist_joint(
    joint: &ConeTwistJoint,
    bodies: &mut [crate::solver::RigidBody],
    dt: Fix128,
) -> Vec3Fix {
    let mut impulse = Vec3Fix::ZERO;
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // 1. Positional constraint (same as ball joint)
    let r_a = body_a.rotation.rotate_vec(joint.local_anchor_a);
    let r_b = body_b.rotation.rotate_vec(joint.local_anchor_b);
    let anchor_a = body_a.position + r_a;
    let anchor_b = body_b.position + r_b;
    let delta = anchor_b - anchor_a;
    let (normal, distance) = delta.normalize_with_length();

    if !distance.is_zero() {
        let compliance_term = joint.compliance / (dt * dt);
        let w_sum =
            point_w_sum(bodies, joint.body_a, joint.body_b, r_a, r_b, normal) + compliance_term;

        if !w_sum.is_zero() {
            let lambda = distance / w_sum;
            apply_point_correction(bodies, joint.body_a, joint.body_b, r_a, r_b, normal, lambda);
            impulse = impulse + normal * lambda;
        }
    }

    // 2. Cone constraint
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];
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
            let w_ang = angular_w_sum(bodies, joint.body_a, joint.body_b, correction_axis)
                + angular_compliance;

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
        let w_ang =
            angular_w_sum(bodies, joint.body_a, joint.body_b, world_axis_a) + angular_compliance;

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
    impulse
}

/// Generalised inverse mass of a body for a rotation about the world axis
/// `axis`: `n · I⁻¹ n` with the diagonal inverse inertia expressed in the
/// body frame (Macklin et al. 2020 eq. 3). Zero for static bodies. An
/// isotropic inverse inertia uses `axis` itself (no rotation into the body
/// frame), so the value does not depend on the orientation, as in
/// [`crate::solver::inv_inertia_world_apply`].
fn angular_inverse_mass(body: &crate::solver::RigidBody, axis: Vec3Fix) -> Fix128 {
    if body.inv_mass.is_zero() {
        return Fix128::ZERO;
    }
    let local = if crate::solver::inv_inertia_is_isotropic(body.inv_inertia) {
        axis
    } else {
        body.rotation.conjugate().rotate_vec(axis)
    };
    local.x * local.x * body.inv_inertia.x
        + local.y * local.y * body.inv_inertia.y
        + local.z * local.z * body.inv_inertia.z
}

/// Sum of the two bodies' angular inverse masses about `axis` (the `w` of
/// an XPBD angular constraint before the compliance term).
fn angular_w_sum(
    bodies: &[crate::solver::RigidBody],
    idx_a: usize,
    idx_b: usize,
    axis: Vec3Fix,
) -> Fix128 {
    angular_inverse_mass(&bodies[idx_a], axis) + angular_inverse_mass(&bodies[idx_b], axis)
}

/// Apply the angular XPBD correction `λ` about the world axis `axis`: body
/// A rotates by `w_a λ`, body B by `−w_b λ` (`w_i = n · I_i⁻¹ n`), so the
/// *relative* rotation changes by `(w_a + w_b) λ` — exactly the error when
/// the caller used `λ = error / (w_a + w_b + α̃)`. Before 1.2.0 both bodies
/// received the full `λ` regardless of inertia and `w` was the vector
/// length of the inverse-inertia diagonal, so a unit-inertia hinge removed
/// only `1/√3` of its error per step.
fn apply_angular_correction(
    bodies: &mut [crate::solver::RigidBody],
    idx_a: usize,
    idx_b: usize,
    axis: Vec3Fix,
    lambda: Fix128,
) {
    let w_a = angular_inverse_mass(&bodies[idx_a], axis);
    let w_b = angular_inverse_mass(&bodies[idx_b], axis);
    if !w_a.is_zero() {
        bodies[idx_a].rotation = rotate_by_angle(bodies[idx_a].rotation, axis, w_a * lambda);
    }
    if !w_b.is_zero() {
        bodies[idx_b].rotation = rotate_by_angle(bodies[idx_b].rotation, axis, -(w_b * lambda));
    }
}

/// `q ← normalize(axis_angle(axis, θ) ⊗ q)`: rotation by exactly `θ` about
/// the world axis (CORDIC `sin_cos`, deterministic), so a rigid constraint
/// whose error is a true angle is satisfied in a single solve. (The
/// first-order `(n θ/2, 1)` update used before 1.2.0 rotates by
/// `2·atan(θ/2)`, 2 % short at 0.3 rad.)
fn rotate_by_angle(q: QuatFix, axis: Vec3Fix, theta: Fix128) -> QuatFix {
    if theta.is_zero() {
        return q;
    }
    QuatFix::from_axis_angle(axis, theta).mul(q).normalize()
}

/// Lever-arm term of the XPBD generalised inverse mass for a point
/// constraint corrected along `normal`: `(r × n)^T I⁻¹ (r × n)`, where `r`
/// is the world-space vector from the body's centre of mass to the anchor
/// point (Macklin, Müller & Chentanez 2016, "XPBD", §3.4). Zero for a
/// static body or when the anchor sits exactly on the centre of mass.
fn point_angular_w(body: &crate::solver::RigidBody, r: Vec3Fix, normal: Vec3Fix) -> Fix128 {
    angular_inverse_mass(body, r.cross(normal))
}

/// Generalised inverse mass `w_a + w_b` of a point (anchor-to-anchor)
/// constraint corrected along `normal`: the plain `inv_mass` of each body
/// plus its lever-arm term `point_angular_w`. This is the `w` XPBD uses
/// before the compliance term is added (`λ = C / (w_a + w_b + α̃)`). `r_a`
/// / `r_b` are the world-space vectors from each body's centre of mass to
/// its anchor (`rotation.rotate_vec(local_anchor)`), taken at the *start*
/// of the joint's solve: the lever arm depends only on orientation, not on
/// position, so it stays correct even when the caller applies several
/// corrections to the same bodies before the orientation itself changes
/// (e.g. the D6 joint's per-axis linear loop).
fn point_w_sum(
    bodies: &[crate::solver::RigidBody],
    idx_a: usize,
    idx_b: usize,
    r_a: Vec3Fix,
    r_b: Vec3Fix,
    normal: Vec3Fix,
) -> Fix128 {
    bodies[idx_a].inv_mass
        + bodies[idx_b].inv_mass
        + point_angular_w(&bodies[idx_a], r_a, normal)
        + point_angular_w(&bodies[idx_b], r_b, normal)
}

/// Apply one body's share of the rotation half of a point-constraint
/// correction: the angular displacement `I⁻¹ (r × n) · λ` decomposed into
/// an axis and an exact angle, applied with `rotate_by_angle`. `λ` carries
/// the sign (positive for body A, negative for body B, matching the
/// translation convention below) so a static or infinite-inertia body
/// (whose `world_inv_inertia_apply` is the zero vector) is left untouched.
fn apply_point_rotation(
    bodies: &mut [crate::solver::RigidBody],
    idx: usize,
    r: Vec3Fix,
    normal: Vec3Fix,
    lambda: Fix128,
) {
    let body = bodies[idx];
    if body.inv_mass.is_zero() {
        return;
    }
    let omega = body.world_inv_inertia_apply(r.cross(normal)) * lambda;
    let (axis, angle) = omega.normalize_with_length();
    if !angle.is_zero() {
        bodies[idx].rotation = rotate_by_angle(bodies[idx].rotation, axis, angle);
    }
}

/// Apply the XPBD point-constraint correction `λ` along `normal` to both
/// bodies, splitting it between translation (`± inv_m · λ · n`, the
/// pre-1.3.0 behaviour) and rotation (`± I⁻¹ (r × n) · λ`) by the lever arm
/// `r_a` / `r_b` (anchor minus centre of mass, see `point_w_sum` for why
/// these are passed in rather than recomputed from position). An anchor on
/// the centre of mass (`r = 0`) reduces exactly to the translation-only
/// update every positional joint used before.
fn apply_point_correction(
    bodies: &mut [crate::solver::RigidBody],
    idx_a: usize,
    idx_b: usize,
    r_a: Vec3Fix,
    r_b: Vec3Fix,
    normal: Vec3Fix,
    lambda: Fix128,
) {
    let body_a = bodies[idx_a];
    let body_b = bodies[idx_b];

    if !body_a.inv_mass.is_zero() {
        bodies[idx_a].position = body_a.position + normal * (lambda * body_a.inv_mass);
    }
    if !body_b.inv_mass.is_zero() {
        bodies[idx_b].position = body_b.position - normal * (lambda * body_b.inv_mass);
    }
    apply_point_rotation(bodies, idx_a, r_a, normal, lambda);
    apply_point_rotation(bodies, idx_b, r_b, normal, -lambda);
}

/// Signed twist angle of `q` about `axis` in `(−π, π]`: the swing–twist
/// decomposition keeps the component of the rotation vector along `axis`,
/// `angle = 2 · atan2(q_xyz · axis, q_w)` after flipping `q` to the
/// `w ≥ 0` cover. Before 1.2.0 this returned `2 · atan2(|proj|, w) ≥ 0`, so
/// a rotation of −1 rad measured as +1 and every angle limit pushed
/// negative rotations the wrong way.
pub(crate) fn compute_twist_angle(q: QuatFix, axis: Vec3Fix) -> Fix128 {
    let qv = Vec3Fix::new(q.x, q.y, q.z);
    let s = qv.dot(axis);
    let proj = axis * s;
    let twist = QuatFix::new(proj.x, proj.y, proj.z, q.w).normalize();
    let signed = Vec3Fix::new(twist.x, twist.y, twist.z).dot(axis);
    let (signed, w) = if twist.w.is_negative() {
        (-signed, -twist.w)
    } else {
        (signed, twist.w)
    };
    Fix128::atan2(signed, w).double()
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

    #[allow(deprecated)] // pins the separation-based solve_joints_breakable
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

    #[allow(deprecated)] // pins the separation-based solve_joints_breakable
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
    fn test_unbreakable_joint_on_force() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
        ];

        // Ball joint with high break force (should NOT break)
        let joint = Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
                .with_break_force(Fix128::from_int(4000)),
        );
        let dt = Fix128::from_ratio(1, 60);

        // reaction force = 1 m / (1 kg⁻¹ · dt²) = 3600 N < 4000 N
        let broken = solve_joints_breaking_on_force(&[joint], &mut bodies, dt);
        assert!(broken.is_empty(), "Joint should not break under low force");
    }

    #[allow(deprecated)] // pins the separation-based solve_joints_breakable
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
        // B 側 anchor (0,2,0) (B の inv_inertia は pair() 既定の (1,1,1)、lever arm 非 0):
        // world anchor_a = (-1,0,0) (上と同じ、A static), world anchor_b = (3,0,0)+(0,2,0) = (3,2,0)
        // delta = (4,2,0), distance = sqrt(20), normal = (4,2,0)/sqrt(20)
        // r_b = (0,2,0) (B の rotation は identity なので local anchor そのもの)
        // r_b × normal = (0, 0, -8/sqrt(20)) → point_angular_w = (8/sqrt(20))^2 * 1 = 64/20 = 16/5
        // w_sum = inv_mass_a(0) + inv_mass_b(1) + 0 (A static) + 16/5 = 21/5
        // lambda = distance / w_sum = sqrt(20) / (21/5) = 10 sqrt(5) / 21 (sqrt(20) = 2 sqrt(5))
        // translation: normal * lambda = (4/sqrt(20), 2/sqrt(20), 0) * 10 sqrt(5)/21
        //   = (20/21, 10/21, 0) (sqrt(5) cancels) → B.position -= that → (3 - 20/21, -10/21, 0)
        //   = (43/21, -10/21, 0)
        // rotation (apply_point_rotation(.., r_b, normal, -lambda)):
        //   omega = world_inv_inertia_apply(r_b × normal) * (-lambda)
        //         = (0,0,-8/sqrt(20)) * (-10 sqrt(5)/21) = (0, 0, 80 sqrt(5) / (2 sqrt(5) * 21))
        //         = (0, 0, 40/21) → signed twist about +Z = 40/21 (positive, nonzero: the lever
        //   arm now produces a rotation, where the pre-fix code left B unrotated)
        let mut bodies2 = pair(Vec3Fix::ZERO, 0, v3i(3, 0, 0), 1);
        bodies2[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let j2 = BallJoint::new(0, 1, v3i(1, 0, 0), v3i(0, 2, 0));
        solve_ball_joint(&j2, &mut bodies2, DT);
        assert!(
            near_v(
                bodies2[1].position,
                Vec3Fix::new(
                    Fix128::from_ratio(43, 21),
                    Fix128::from_ratio(-10, 21),
                    Fix128::ZERO
                )
            ),
            "{:?}",
            bodies2[1].position
        );
        assert!(near(
            compute_twist_angle(bodies2[1].rotation, Vec3Fix::UNIT_Z),
            Fix128::from_ratio(40, 21)
        ));
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
        // XPBD: lambda = dt^2 F / (1 + (k dt^2 + c dt) w)、dt = 1/4
        // rest 1、k 2、c 0、距離 4、inv (1, 3): F = 6、w = 4 → lambda = (6/16)/(1 + 2/16*4) = 1/4 → A +1/4、B -3/4
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
            Vec3Fix::new(Fix128::from_ratio(1, 4), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            bodies[1].position,
            Vec3Fix::new(Fix128::from_ratio(13, 4), Fix128::ZERO, Fix128::ZERO)
        );
        // 圧縮 (距離 4 < rest 8): F = 2*(-4) = -8、w = 2 → lambda = (-8/16)/(1 + 2/16*2) = -2/5 → A -2/5、B +2/5
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
        assert!(near_v(
            comp[0].position,
            Vec3Fix::new(Fix128::from_ratio(-2, 5), Fix128::ZERO, Fix128::ZERO)
        ));
        assert!(near_v(
            comp[1].position,
            Vec3Fix::new(Fix128::from_ratio(22, 5), Fix128::ZERO, Fix128::ZERO)
        ));
        // damping: rest 4 (ばね力 0)、k 2、c 2、B が +x に 3 で離れる: F = 2*3 = 6、w = 2
        // 減衰は陰的 (Macklin 2016 eq. 26): lambda = dt^2 F / (1 + (k dt^2 + c dt) w)
        // → lambda = (6/16)/(1 + (2/16 + 2/4)*2) = 1/6 → A +1/6、B -1/6
        let mut damp = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 1);
        damp[1].velocity = v3i(3, 0, 0);
        let jd = SpringJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, fi(4), fi(2), fi(2));
        solve_spring_joint(&jd, &mut damp, DT);
        assert!(near_v(
            damp[0].position,
            Vec3Fix::new(Fix128::from_ratio(1, 6), Fix128::ZERO, Fix128::ZERO)
        ));
        assert!(near_v(
            damp[1].position,
            Vec3Fix::new(Fix128::from_ratio(23, 6), Fix128::ZERO, Fix128::ZERO)
        ));
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

    /// With a nonzero lever arm, a Locked x axis followed by a Locked y axis is no
    /// longer two independent 1-D projections: the x-step's lever-arm rotation (part
    /// of this fix) changes where B's anchor sits before the y-step solves, so the
    /// y-step's lever arm must be recomputed from B's CURRENT rotation, not a value
    /// cached from before the loop.
    ///
    /// A static at origin (anchor 0, irrelevant). B at (4,3,0), inv_mass 1,
    /// inv_inertia (1,1,1), anchor (0,2,0). delta = (4,5,0) (cached once, as before
    /// this fix). x-step: r_b=(0,2,0), r_b×axis_x=(0,0,-2), w=1+4=5, lambda=4/5 ⇒
    /// B.x = 4 - 4/5 = 16/5, and B rotates by theta = 8/5 about z (same closed form as
    /// `every_joint_positional_part_uses_rotated_local_anchors`'s ball case).
    /// y-step (fresh r_b): r_b_fresh = Rz(theta).rotate((0,2,0)) = (-2 sin(theta),
    /// 2 cos(theta), 0); r_b_fresh × axis_y = (0,0,-2 sin(theta)); w = 1 + 4 sin(theta)^2;
    /// lambda_y = 5 / w ⇒ B.y = 3 - lambda_y. (A stale r_b=(0,2,0) would instead give
    /// r_b×axis_y = 0, w = 1, lambda_y = 5, B.y = -2 — a full, unmoderated jump.)
    #[test]
    fn d6_joint_second_locked_axis_uses_the_rotation_from_the_first() {
        let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, v3i(0, 2, 0));
        j.linear_x = D6Motion::Locked;
        j.linear_y = D6Motion::Locked;
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(4, 3, 0), 1);
        solve_d6_joint(&j, &mut bodies, DT);

        // sin/cos via Fix128's own CORDIC (not the bug under test: these are generic
        // trig primitives, not solve_d6_joint) so the expected value is bit-exact
        // comparable, and deterministic cross-platform (see clippy::disallowed_methods).
        let theta = Fix128::from_ratio(8, 5);
        let sin_theta = theta.sin();
        let want_y = Fix128::from_int(3)
            - Fix128::from_int(5) / (Fix128::ONE + Fix128::from_int(4) * sin_theta * sin_theta);
        assert!(
            near(bodies[1].position.x, Fix128::from_ratio(16, 5)),
            "x {:?}",
            bodies[1].position.x
        );
        assert!(
            near(bodies[1].position.y, want_y),
            "y {:?} (want {want_y:?})",
            bodies[1].position.y
        );
        // the stale (pre-fix-refinement) value a cached r_b would have produced
        assert!(
            !near(bodies[1].position.y, Fix128::from_int(-2)),
            "y landed on the stale-cache value: {:?}",
            bodies[1].position.y
        );
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
        // 1.2.0: the rigid solve is exact, so the angle is ~0 after one pass
        // and later passes only see ulp-level noise (2⁻⁴⁰ tolerance)
        let noise = Fix128::from_raw(0, 1 << 24);
        for i in 0..30 {
            solve_d6_joint(&j, &mut bodies, DT);
            let a = angle(&bodies).abs();
            assert!(a <= prev + noise, "iter {i}");
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

    #[allow(deprecated)] // pins the separation-based solve_joints_breakable
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

    #[test]
    fn breaking_on_force_breaks_strictly_above_threshold_and_skips_broken() {
        // 距離 5・質量 1・DT 1/4 の ball joint の反力は 5 / (1/16) = 80: break 80 → 壊れない (`>` は false)、break 64 → 壊れて解かれない
        let mk = |bf: i64| {
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(fi(bf)))
        };
        let mut hold = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let broken = solve_joints_breaking_on_force(&[mk(80)], &mut hold, DT);
        assert!(broken.is_empty());
        assert!(
            near_v(hold[1].position, Vec3Fix::ZERO),
            "解かれて A に一致 {:?}",
            hold[1].position
        );
        let mut snap = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let broken2 = solve_joints_breaking_on_force(&[mk(64)], &mut snap, DT);
        assert_eq!(broken2, vec![0]);
        // 判定は solve の後: 壊れた joint もこの呼び出しで解かれている
        assert!(
            near_v(snap[1].position, Vec3Fix::ZERO),
            "{:?}",
            snap[1].position
        );
        // 同じ 2 body の joint を 3 本: 1 本目が gap を閉じる (80 N) ので後の 2 本は
        // gap 0 で反力 0 (Gauss-Seidel の順で荷重を受ける) ⇒ 壊れるのは 1 本目だけ
        let mut multi = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        let js = [mk(64), mk(1600), mk(16)];
        let broken3 = solve_joints_breaking_on_force(&js, &mut multi, DT);
        assert_eq!(broken3, vec![0]);
        assert!(near_v(multi[1].position, Vec3Fix::ZERO));
        // solve_joints (非 breakable) は全部解く
        let mut all = pair(Vec3Fix::ZERO, 0, v3i(3, 4, 0), 1);
        solve_joints(&[mk(16)], &mut all, DT);
        assert!(near_v(all[1].position, Vec3Fix::ZERO));
    }

    // ---- D6 builders ---------------------------------------------------

    #[test]
    fn d6_with_linear_motion_and_limits_builders_drive_solver() {
        // builder は各軸に個別に格納する (x/y/z の取り違えを検出)
        let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_linear_motion(D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
            .with_linear_limits(v3i(-7, -2, -9), v3i(7, 2, 9));
        assert_eq!(
            (j.linear_x, j.linear_y, j.linear_z),
            (D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
        );
        assert_eq!(j.linear_limit_min, v3i(-7, -2, -9));
        assert_eq!(j.linear_limit_max, v3i(7, 2, 9));
        // angular 側は default (Free / ±π) のまま
        assert_eq!(
            (j.angular_x, j.angular_y, j.angular_z),
            (D6Motion::Free, D6Motion::Free, D6Motion::Free)
        );
        assert_eq!(j.angular_limit_max.y, Fix128::PI);

        // 手書き field 設定と builder は同一 joint を作る
        let mut manual = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        manual.linear_x = D6Motion::Locked;
        manual.linear_y = D6Motion::Limited;
        manual.linear_z = D6Motion::Free;
        manual.linear_limit_min = v3i(-7, -2, -9);
        manual.linear_limit_max = v3i(7, 2, 9);
        assert_eq!(j, manual);

        // solver: B (4, 3, 5)、A static / B inv 1 → x locked → 0、y limited [-2, 2] → 2、z free → 5
        let mut bodies = pair(Vec3Fix::ZERO, 0, v3i(4, 3, 5), 1);
        solve_d6_joint(&j, &mut bodies, DT);
        assert_eq!(bodies[1].position, v3i(0, 2, 5));
        // 下限側: y = -6 → error -6 - (-2) = -4 → (0, -2, 0)
        let mut low = pair(Vec3Fix::ZERO, 0, v3i(0, -6, 0), 1);
        solve_d6_joint(&j, &mut low, DT);
        assert_eq!(low[1].position, v3i(0, -2, 0));
        // 限界内 (0, 1, 0) は不変
        let mut inside = pair(Vec3Fix::ZERO, 0, v3i(0, 1, 0), 1);
        solve_d6_joint(&j, &mut inside, DT);
        assert_eq!(inside[1].position, v3i(0, 1, 0));
    }

    #[test]
    fn d6_with_angular_motion_and_limits_builders_drive_solver() {
        let q = Fix128::from_ratio(1, 4);
        let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_angular_motion(D6Motion::Free, D6Motion::Locked, D6Motion::Limited)
            .with_angular_limits(Vec3Fix::new(-q, -q, -q), Vec3Fix::new(q, q, q));
        assert_eq!(
            (j.angular_x, j.angular_y, j.angular_z),
            (D6Motion::Free, D6Motion::Locked, D6Motion::Limited)
        );
        assert_eq!(j.angular_limit_min, Vec3Fix::new(-q, -q, -q));
        assert_eq!(j.angular_limit_max, Vec3Fix::new(q, q, q));
        // linear 側は default (Free / ±1) のまま
        assert_eq!(
            (j.linear_x, j.linear_y, j.linear_z),
            (D6Motion::Free, D6Motion::Free, D6Motion::Free)
        );
        assert_eq!(j.linear_limit_min, v3i(-1, -1, -1));

        let angle = |b: &[RigidBody], axis: Vec3Fix| {
            let rel = b[1].rotation.mul(b[0].rotation.conjugate());
            compute_twist_angle(rel, axis)
        };
        // z Limited [-1/4, 1/4]: 1/2 回転は限界に向かって減る
        let mut over = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        over[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 2));
        let o0 = angle(&over, Vec3Fix::UNIT_Z);
        solve_d6_joint(&j, &mut over, DT);
        let o1 = angle(&over, Vec3Fix::UNIT_Z);
        assert!(o1 < o0, "{o1:?} < {o0:?}");
        // 1.2.0: exact solve lands on the limit (±2⁻⁴⁰ rounding), never below
        assert!(
            o1 >= q - Fix128::from_raw(0, 1 << 24),
            "limit は over-correct しない: {o1:?}"
        );
        // z 1/10 は限界内 → z 角は不変 (y Locked が拾う twist は数値 noise 程度、2^-40 以下)
        let mut inside = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        inside[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 10));
        let z0 = angle(&inside, Vec3Fix::UNIT_Z);
        solve_d6_joint(&j, &mut inside, DT);
        let z1 = angle(&inside, Vec3Fix::UNIT_Z);
        assert!(near(z0, z1), "{z0:?} vs {z1:?}");
        // y Locked: 1/10 でも引き戻される、x Free: 1/2 でも不変
        let mut locked = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        locked[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 10));
        let l0 = angle(&locked, Vec3Fix::UNIT_Y).abs();
        solve_d6_joint(&j, &mut locked, DT);
        assert!(angle(&locked, Vec3Fix::UNIT_Y).abs() < l0);
        let mut free = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        free[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(1, 2));
        let x0 = angle(&free, Vec3Fix::UNIT_X);
        solve_d6_joint(&j, &mut free, DT);
        let x1 = angle(&free, Vec3Fix::UNIT_X);
        assert!(near(x0, x1), "{x0:?} vs {x1:?}");
        assert!(
            x1 > Fix128::from_ratio(49, 100),
            "x Free は 1/2 のまま: {x1:?}"
        );
    }

    // ---- batch 6: anchors / compliance / limit equality / compute_force ----

    /// A: static at origin、z 軸 180° 回転、anchor (1,0,0) → world anchor (-1,0,0)
    /// B: inv 1 at (3,-2,0)、anchor (0,2,0) → world anchor (3,0,0)  ⇒ gap 4 (x 方向)
    fn anchored_pair() -> Vec<RigidBody> {
        let mut b = pair(Vec3Fix::ZERO, 0, v3i(3, -2, 0), 1);
        b[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        b
    }

    const A_ANCHOR: Vec3Fix = Vec3Fix {
        x: Fix128::ONE,
        y: Fix128::ZERO,
        z: Fix128::ZERO,
    };
    const B_ANCHOR: Vec3Fix = Vec3Fix {
        x: Fix128::ZERO,
        y: Fix128 { hi: 2, lo: 0 },
        z: Fix128::ZERO,
    };

    #[test]
    fn every_joint_positional_part_uses_rotated_local_anchors() {
        // Shared geometry for ball/hinge/fixed/cone-twist/d6 below: world anchor_a =
        // (-1,0,0) (A static, 180° about z), world anchor_b = (3,-2,0)+(0,2,0) = (3,0,0)
        // (B_ANCHOR's rotate is identity) ⇒ delta = (4,0,0), distance = 4 (no sqrt: the
        // y components cancel exactly), normal = (1,0,0).
        // r_b = (0,2,0) (B's local anchor, B's rotation starts identity).
        // r_b × normal = (0,2,0) × (1,0,0) = (0,0,-2) ⇒ point_angular_w =
        // (-2)^2 * inv_inertia.z(1) = 4. w_sum = inv_m_a(0) + inv_m_b(1) + 0(A static) + 4 = 5.
        // lambda = distance / w_sum = 4/5. translation: normal * lambda * inv_m_b = (4/5,0,0)
        // ⇒ B.position -= that = (3 - 4/5, -2, 0) = (11/5, -2, 0).
        // rotation: apply_point_rotation(.., r_b, normal, -lambda): omega =
        // world_inv_inertia_apply((0,0,-2)) * (-4/5) = (0,0,-2) * (-4/5) = (0,0,8/5)
        // ⇒ signed twist about +z = 8/5 (nonzero: this is exactly the lever-arm rotation
        // the pre-fix code never produced).
        let want_pos = Vec3Fix::new(
            Fix128::from_ratio(11, 5),
            -Fix128::from_int(2),
            Fix128::ZERO,
        );
        let want_twist = Fix128::from_ratio(8, 5);

        let mut b = anchored_pair();
        solve_ball_joint(&BallJoint::new(0, 1, A_ANCHOR, B_ANCHOR), &mut b, DT);
        assert!(near_v(b[1].position, want_pos), "ball {:?}", b[1].position);
        assert!(near(
            compute_twist_angle(b[1].rotation, Vec3Fix::UNIT_Z),
            want_twist
        ));

        // hinge's axis-alignment step (its own step 2) is a no-op here: rotating UNIT_Z
        // by a rotation that is itself about Z leaves UNIT_Z unchanged, for both bodies,
        // so world_axis_a == world_axis_b after step 1 and the cross product is zero.
        let mut h = anchored_pair();
        solve_hinge_joint(
            &HingeJoint::new(0, 1, A_ANCHOR, B_ANCHOR, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z),
            &mut h,
            DT,
        );
        assert!(near_v(h[1].position, want_pos), "hinge {:?}", h[1].position);
        assert!(near(
            compute_twist_angle(h[1].rotation, Vec3Fix::UNIT_Z),
            want_twist
        ));

        // fixed's position is identical to ball/hinge (step 1 is the same point
        // constraint), but its own step 2 ("maintain relative rotation") then measures
        // B against target_rot_b = A.rotation * relative_rotation = Rz(pi) * Rz(-pi) =
        // identity exactly, so rot_error = B.rotation = Rz(8/5) (the step-1 result above).
        // angular_inverse_mass(B, UNIT_Z) is exactly 1 (inv_inertia.z = 1, and Z is
        // invariant under a rotation about Z), so this step's lambda = 8/5 exactly
        // cancels the step-1 rotation: B ends up back at identity, same as pre-fix.
        let mut f = anchored_pair();
        solve_fixed_joint(
            &FixedJoint::new(
                0,
                1,
                A_ANCHOR,
                B_ANCHOR,
                QuatFix::new(Fix128::ZERO, Fix128::ZERO, -Fix128::ONE, Fix128::ZERO),
            ),
            &mut f,
            DT,
        );
        assert!(near_v(f[1].position, want_pos), "fixed {:?}", f[1].position);
        assert_eq!(
            f[1].rotation,
            QuatFix::IDENTITY,
            "fixed rotation {:?}",
            f[1].rotation
        );

        // cone-twist's cone/twist steps are no-ops here too: world_axis_a == world_axis_b
        // (both UNIT_Z, same reasoning as hinge) makes cone_angle exactly 0 (<= the
        // default pi/2 limit), and the resulting twist_angle (8/5 - pi) is within the
        // default pi twist_limit.
        let mut ct = anchored_pair();
        solve_cone_twist_joint(
            &ConeTwistJoint::new(0, 1, A_ANCHOR, B_ANCHOR, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z),
            &mut ct,
            DT,
        );
        assert!(
            near_v(ct[1].position, want_pos),
            "cone {:?}",
            ct[1].position
        );
        assert!(near(
            compute_twist_angle(ct[1].rotation, Vec3Fix::UNIT_Z),
            want_twist
        ));

        // d6 locked on all 3 linear axes: axis_x = frame_a.rotate(UNIT_X) = (-1,0,0) (A's
        // 180° frame), and the x-step alone reproduces the ball-joint computation above
        // (same delta, same lever arm) since delta's y/z components are both exactly 0
        // (anchor_b.y = -2+2 = 0 = anchor_a.y, anchor_b.z = anchor_a.z = 0), so the y and
        // z steps see error == 0 and are no-ops; angular is all Free (default), so no
        // angular correction either.
        let mut d6j = D6Joint::new(0, 1, A_ANCHOR, B_ANCHOR);
        d6j.linear_x = D6Motion::Locked;
        d6j.linear_y = D6Motion::Locked;
        d6j.linear_z = D6Motion::Locked;
        let mut d = anchored_pair();
        solve_d6_joint(&d6j, &mut d, DT);
        assert!(near_v(d[1].position, want_pos), "d6 {:?}", d[1].position);
        assert!(near(
            compute_twist_angle(d[1].rotation, Vec3Fix::UNIT_Z),
            want_twist
        ));
        // slider (軸 x、A 回転で world 軸 -x): perp は y 成分 0 なので along のみ → 移動なし、
        // B anchor を (0,3,0) にして perp 1 を作る → B の y が -1 動く
        //
        // Unlike the 5 cases above, this sub-case is unaffected by the lever-arm fix:
        // r_b = (0,3,0) and the correction direction perp_normal = (0,1,0) are parallel,
        // so r_b x perp_normal = 0 and point_angular_w is exactly 0 (a lever arm aligned
        // with the correction direction produces no torque) — translation-only is still
        // the exact answer here, same as before this fix.
        let mut sl = anchored_pair();
        solve_slider_joint(
            &SliderJoint::new(0, 1, Vec3Fix::UNIT_X, A_ANCHOR, v3i(0, 3, 0)),
            &mut sl,
            DT,
        );
        assert!(
            near_v(sl[1].position, v3i(3, -3, 0)),
            "slider {:?}",
            sl[1].position
        );
        // spring: rest 4 なら力 0 (anchor 込みの距離が 4)、rest 2 なら k=1 で 2 → impulse 1/2
        let mut sp0 = anchored_pair();
        solve_spring_joint(
            &SpringJoint::new(0, 1, A_ANCHOR, B_ANCHOR, fi(4), Fix128::ONE, Fix128::ZERO),
            &mut sp0,
            DT,
        );
        assert_eq!(sp0[1].position, v3i(3, -2, 0));
        let mut sp = anchored_pair();
        solve_spring_joint(
            &SpringJoint::new(0, 1, A_ANCHOR, B_ANCHOR, fi(2), Fix128::ONE, Fix128::ZERO),
            &mut sp,
            DT,
        );
        // The spring takes the same lever-arm coupling as the point joints above:
        // C = 2, w = 5, lambda = (dt^2 k C) / (1 + k dt^2 w) = (1/8) / (21/16) = 2/21
        // ⇒ B moves -2/21 along x and turns by I⁻¹ (r_b × n) (−λ) = +4/21 about z.
        assert!(
            near_v(
                sp[1].position,
                Vec3Fix::new(Fix128::from_ratio(61, 21), fi(-2), Fix128::ZERO)
            ),
            "spring {:?}",
            sp[1].position
        );
        assert!(near(
            compute_twist_angle(sp[1].rotation, Vec3Fix::UNIT_Z),
            Fix128::from_ratio(4, 21)
        ));
    }

    #[test]
    fn compliance_softens_every_positional_joint_identically() {
        // A static、B inv 1、gap 4、compliance 1/16 (dt 1/4 → term 1) → w 2 → 移動 2
        let expect = v3i(2, 0, 0);
        let mut h = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        h.compliance = Fix128::from_ratio(1, 16);
        let mut bh = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        solve_hinge_joint(&h, &mut bh, DT);
        assert_eq!(bh[1].position, expect, "hinge");
        let mut f = FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        f.compliance = Fix128::from_ratio(1, 16);
        let mut bf = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        solve_fixed_joint(&f, &mut bf, DT);
        assert_eq!(bf[1].position, expect, "fixed");
        let mut c = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        c.compliance = Fix128::from_ratio(1, 16);
        let mut bc = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        solve_cone_twist_joint(&c, &mut bc, DT);
        assert_eq!(bc[1].position, expect, "cone");
        let mut s = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO);
        s.compliance = Fix128::from_ratio(1, 16);
        let mut bs = pair(Vec3Fix::ZERO, 0, v3i(0, 4, 0), 1); // perp 4
        solve_slider_joint(&s, &mut bs, DT);
        assert_eq!(bs[1].position, v3i(0, 2, 0), "slider");
        let mut d = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        d.linear_x = D6Motion::Locked;
        d.compliance = Fix128::from_ratio(1, 16);
        let mut bd = pair(Vec3Fix::ZERO, 0, v3i(4, 0, 0), 1);
        solve_d6_joint(&d, &mut bd, DT);
        assert_eq!(bd[1].position, expect, "d6");
        // angular compliance: hinge 軸誤差の補正量が小さくなる (compliance 0 との比較)
        let mk = |ac: Fix128| {
            let mut j = HingeJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            );
            j.angular_compliance = ac;
            let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
            b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(1, 2));
            solve_hinge_joint(&j, &mut b, DT);
            let a = b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            let c = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
            a.cross(c).length()
        };
        assert!(
            mk(Fix128::from_ratio(1, 4)) > mk(Fix128::ZERO),
            "angular compliance で補正が弱まる"
        );
    }

    #[test]
    fn limit_equality_is_inside_the_allowed_range() {
        // slider along == min ちょうど → 不変 (`<` false)、d6 proj == limit_max → 不変、cone == limit → 不変
        let sj = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_limits(fi(-1), fi(2));
        let mut at_min = pair(Vec3Fix::ZERO, 0, v3i(-1, 0, 0), 1);
        solve_slider_joint(&sj, &mut at_min, DT);
        assert_eq!(at_min[1].position, v3i(-1, 0, 0));
        let mut dj = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        dj.linear_y = D6Motion::Limited; // [-1, 1]
        let mut at_max = pair(Vec3Fix::ZERO, 0, v3i(0, 1, 0), 1);
        solve_d6_joint(&dj, &mut at_max, DT);
        assert_eq!(at_max[1].position, v3i(0, 1, 0));
        let mut at_lo = pair(Vec3Fix::ZERO, 0, v3i(0, -1, 0), 1);
        solve_d6_joint(&dj, &mut at_lo, DT);
        assert_eq!(at_lo[1].position, v3i(0, -1, 0));
        // d6 limited で下限割れ: (0,-3,0) → -3 - (-1) = -2 → B += 2 → (0,-1,0) (上限側は既存 test)
        let mut below = pair(Vec3Fix::ZERO, 0, v3i(0, -3, 0), 1);
        solve_d6_joint(&dj, &mut below, DT);
        assert_eq!(below[1].position, v3i(0, -1, 0));
    }

    #[test]
    fn compute_force_honours_anchors_and_rotation_for_every_kind() {
        let bodies = anchored_pair(); // world anchor 距離 4 (x)
        assert_eq!(
            Joint::Ball(BallJoint::new(0, 1, A_ANCHOR, B_ANCHOR)).compute_force(&bodies),
            fi(4)
        );
        assert_eq!(
            Joint::Hinge(HingeJoint::new(
                0,
                1,
                A_ANCHOR,
                B_ANCHOR,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z
            ))
            .compute_force(&bodies),
            fi(4)
        );
        assert_eq!(
            Joint::Fixed(FixedJoint::new(0, 1, A_ANCHOR, B_ANCHOR, QuatFix::IDENTITY))
                .compute_force(&bodies),
            fi(4)
        );
        assert_eq!(
            Joint::D6(D6Joint::new(0, 1, A_ANCHOR, B_ANCHOR)).compute_force(&bodies),
            fi(4)
        );
        assert_eq!(
            Joint::ConeTwist(ConeTwistJoint::new(
                0,
                1,
                A_ANCHOR,
                B_ANCHOR,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z
            ))
            .compute_force(&bodies),
            fi(4)
        );
        // spring: |k (4 - rest)| = |3 (4 - 1)| = 9
        assert_eq!(
            Joint::Spring(SpringJoint::new(
                0,
                1,
                A_ANCHOR,
                B_ANCHOR,
                fi(1),
                fi(3),
                Fix128::ZERO
            ))
            .compute_force(&bodies),
            fi(9)
        );
        // slider 軸 y (A 回転で world -y): delta (4,0,0) の perp は 4、軸 x なら perp 0
        assert_eq!(
            Joint::Slider(SliderJoint::new(0, 1, Vec3Fix::UNIT_Y, A_ANCHOR, B_ANCHOR))
                .compute_force(&bodies),
            fi(4)
        );
        assert_eq!(
            Joint::Slider(SliderJoint::new(0, 1, Vec3Fix::UNIT_X, A_ANCHOR, B_ANCHOR))
                .compute_force(&bodies),
            Fix128::ZERO
        );
        // anchor なしなら距離は |(3,-2,0)| = √13 ≠ 4
        assert_ne!(
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)).compute_force(&bodies),
            fi(4)
        );
    }

    // ---- batch 7: angular / limit / compliance arithmetic (mutation kills, 2026-09-15) ----
    //
    // Conventions used below (all derived by hand, see each test's comment):
    // * `DT = 1/4` so `dt * dt = 1/16` and `compliance / dt²` is `compliance * 16`.
    // * `set_inv_inertia(i_a, i_b)` sets an isotropic `inv_inertia = (i, i, i)`, so the
    //   angular inverse mass `w = n · I⁻¹ n` is exactly `i` for every unit axis and every
    //   body orientation (static bodies contribute 0 regardless).
    // * `q35(axis)` is the unit quaternion `(axis * 3/5, 4/5)`: a rotation of
    //   `θ = 2·atan(3/4) ≈ 1.287 002 rad` with `sin θ = 24/25`, `cos θ = 7/25`.
    // * `apply_angular_correction(λ)` rotates A by exactly `+w_a λ` and B by `−w_b λ`, and
    //   every caller uses `λ = error / (w_a + w_b + α̃)`, so with `α̃ = 0` one solve changes
    //   the relative angle by exactly `error` (lands on the limit / on zero).

    /// Isotropic `inv_inertia = (i, i, i)` on both bodies: `n · I⁻¹ n == i` for any unit `n`.
    fn set_inv_inertia(b: &mut [RigidBody], ia: Fix128, ib: Fix128) {
        b[0].inv_inertia = Vec3Fix::new(ia, ia, ia);
        b[1].inv_inertia = Vec3Fix::new(ib, ib, ib);
    }

    /// Unit quaternion `(axis * 3/5, 4/5)`: rotation by `2·atan(3/4) ≈ 1.287 rad`.
    fn q35(axis: Vec3Fix) -> QuatFix {
        let s = Fix128::from_ratio(3, 5);
        QuatFix::new(axis.x * s, axis.y * s, axis.z * s, Fix128::from_ratio(4, 5))
    }

    /// Relative twist of B w.r.t. A about `axis` (same formula as the solvers, signed).
    fn rel_twist(b: &[RigidBody], axis: Vec3Fix) -> Fix128 {
        compute_twist_angle(b[1].rotation.mul(b[0].rotation.conjugate()), axis)
    }

    /// Angle between the bodies' world z axes (cone angle of a z twist axis).
    fn tilt_z(b: &[RigidBody]) -> Fix128 {
        let a = b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z);
        let c = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
        Fix128::atan2(a.cross(c).length(), a.dot(c))
    }

    /// Angle between world z and `q`'s z axis.
    fn tilt_from_world_z(q: QuatFix) -> Fix128 {
        let c = q.rotate_vec(Vec3Fix::UNIT_Z);
        Fix128::atan2(Vec3Fix::UNIT_Z.cross(c).length(), Vec3Fix::UNIT_Z.dot(c))
    }

    /// `θ = 2·atan2(3, 4)`, the rotation angle of `q35`.
    fn theta35() -> Fix128 {
        Fix128::atan2(fi(3), fi(4)).double()
    }

    /// Kills `776:64` / `876:64` / `950:64` / `1104:68` / `1200:64` (`correction * inv_mass_a`
    /// → `/`) and `1030:81` (spring `impulse * inv_mass_a` → `/`).
    ///
    /// A inv_mass 3 at 0, B inv_mass 1 at gap 4 → w = 4, λ = 1, correction = 1:
    /// A moves `1 * 3 = 3` (mutant: `1 / 3`), B moves `1 * 1 = 1` → both land on 3.
    /// Spring: rest 1, k 2, dist 4 → F = 6, impulse = 6 · ¼ = 3/2 → A += 3/2 · 3 = 9/2
    /// (mutant: 3/2 / 3 = 1/2), B −= 3/2 → 5/2.
    #[test]
    fn positional_body_a_update_multiplies_by_inverse_mass() {
        let expect = v3i(3, 0, 0);
        let mut h = pair(Vec3Fix::ZERO, 3, v3i(4, 0, 0), 1);
        solve_hinge_joint(
            &HingeJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            ),
            &mut h,
            DT,
        );
        assert_eq!((h[0].position, h[1].position), (expect, expect), "hinge");
        let mut f = pair(Vec3Fix::ZERO, 3, v3i(4, 0, 0), 1);
        solve_fixed_joint(
            &FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY),
            &mut f,
            DT,
        );
        assert_eq!((f[0].position, f[1].position), (expect, expect), "fixed");
        let mut c = pair(Vec3Fix::ZERO, 3, v3i(4, 0, 0), 1);
        solve_cone_twist_joint(
            &ConeTwistJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            ),
            &mut c,
            DT,
        );
        assert_eq!((c[0].position, c[1].position), (expect, expect), "cone");
        let mut d6 = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        d6.linear_x = D6Motion::Locked;
        let mut d = pair(Vec3Fix::ZERO, 3, v3i(4, 0, 0), 1);
        solve_d6_joint(&d6, &mut d, DT);
        assert_eq!((d[0].position, d[1].position), (expect, expect), "d6");
        // slider: axis x, gap (0,4,0) is entirely perpendicular → same split along y
        let mut s = pair(Vec3Fix::ZERO, 3, v3i(0, 4, 0), 1);
        solve_slider_joint(
            &SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO),
            &mut s,
            DT,
        );
        assert_eq!(
            (s[0].position, s[1].position),
            (v3i(0, 3, 0), v3i(0, 3, 0)),
            "slider"
        );
        let mut sp = pair(Vec3Fix::ZERO, 3, v3i(4, 0, 0), 1);
        solve_spring_joint(
            &SpringJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                fi(1),
                fi(2),
                Fix128::ZERO,
            ),
            &mut sp,
            DT,
        );
        assert_eq!(
            sp[0].position,
            Vec3Fix::new(Fix128::from_ratio(3, 4), Fix128::ZERO, Fix128::ZERO),
            "spring A"
        );
        assert_eq!(
            sp[1].position,
            Vec3Fix::new(Fix128::from_ratio(15, 4), Fix128::ZERO, Fix128::ZERO),
            "spring B"
        );
    }

    /// Kills `795:59` (`angular_compliance / dt²` → `*`), `795:65` (`dt * dt` → `/`, `+`),
    /// `797` (`angular_w_sum + angular_compliance` → `-`, `*`), `1296` (`w_a + w_b` → `*`)
    /// and the `w_a * λ` / `w_b * λ` products in `apply_angular_correction` (1316 / 1319).
    ///
    /// Both dynamic, `w_a = 1/2`, `w_b = 1/4`, `angular_compliance = 1/64` → term 1/4,
    /// `w = 1/2 + 1/4 + 1/4 = 1`. B = q35(x) tilts its z axis by `θ` about +x, the
    /// misalignment angle is `atan2(24/25, 7/25) = θ`, `λ = θ / 1 = θ`:
    /// A rotates `+θ/2` about x, B rotates `−θ/4` → B's tilt from world z is `3θ/4`,
    /// the residual angle between the two axes is `θ/4` (compliance keeps 1/4 of the error).
    /// Mutants: `w ∈ {0.751, 0.766, 0.781 (795), 0.5, 0.1875 (797), 0.375 (1296)}` →
    /// A's tilt `∈ {0.666θ, 0.653θ, 0.64θ, θ, 2.67θ, 1.33θ}` instead of `θ/2`.
    #[test]
    fn hinge_axis_alignment_uses_angular_compliance_and_inertia_sum() {
        let mut j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        );
        j.angular_compliance = Fix128::from_ratio(1, 64);
        let mut b = pair(Vec3Fix::ZERO, 1, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::from_ratio(1, 2), Fix128::from_ratio(1, 4));
        b[1].rotation = q35(Vec3Fix::UNIT_X);
        let theta = theta35();
        assert!(near(tilt_z(&b), theta));
        solve_hinge_joint(&j, &mut b, DT);
        let a_tilt = tilt_from_world_z(b[0].rotation);
        let b_tilt = tilt_from_world_z(b[1].rotation);
        let rel = tilt_z(&b);
        assert!(near(a_tilt, theta.half()), "A tilt {a_tilt:?} vs θ/2");
        assert!(
            near(b_tilt, theta - theta / fi(4)),
            "B tilt {b_tilt:?} vs 3θ/4"
        );
        assert!(near(rel, theta / fi(4)), "residual {rel:?} vs θ/4");
        // A rotated about +x (its z axis tipped toward −y), B rotated back about −x
        assert!(b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z).y < Fix128::ZERO);
    }

    /// Kills `900` (`error_mag` from `atan2(...).double()`), `901:59` (`angular_compliance / dt²`
    /// → `*`), `901:65` (`dt * dt` → `/`, `+`), `903` (`+ angular_compliance` → `-`, `*`) and
    /// `1319` (`w_b * λ` → `/`).
    ///
    /// A static, `w_b = 3/4`, `angular_compliance = 1/64` → term 1/4, `w = 1`. B = q35(z),
    /// target identity → error quaternion q35(z), `error_mag = 2·atan2(3/5, 4/5) = θ`,
    /// `λ = θ`, B rotates `−3θ/4` → residual relative twist exactly `θ/4 ≈ 0.3218`.
    /// Mutants: `w ∈ {0.751, 0.766, 0.781, 0.5, 0.1875}` → residual
    /// `∈ {0.001θ, 0.02θ, 0.04θ, −θ/2, −3θ (wraps to 2.42)}`; `w_b / λ` → `θ − 0.583`.
    #[test]
    fn fixed_joint_rotation_lock_uses_angular_compliance_and_inertia_sum() {
        let mut j = FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        j.angular_compliance = Fix128::from_ratio(1, 64);
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, Fix128::from_ratio(3, 4));
        b[1].rotation = q35(Vec3Fix::UNIT_Z);
        let theta = theta35();
        assert!(near(rel_twist(&b, Vec3Fix::UNIT_Z), theta));
        solve_fixed_joint(&j, &mut b, DT);
        let after = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(after, theta / fi(4)), "{after:?} vs θ/4");
        assert_eq!(b[0].rotation, QuatFix::IDENTITY, "static A never rotates");
        // rigid (compliance 0) + w_b = 1: one solve restores the identity exactly
        let rigid = FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let mut r = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut r, Fix128::ZERO, Fix128::ONE);
        r[1].rotation = q35(Vec3Fix::UNIT_Z);
        solve_fixed_joint(&rigid, &mut r, DT);
        assert!(near(rel_twist(&r, Vec3Fix::UNIT_Z), Fix128::ZERO));
        assert!(near(r[1].rotation.w.abs(), Fix128::ONE));
    }

    /// Kills `823:18` (`<` → `==`), `824:35` (`min − angle` → `/`, `+`), `826:16`
    /// (`!w_ang.is_zero()` negation), `827:45` (`1 / w` → `*`), `833:21` (sign of λ),
    /// `833:29` (`error * inv_w` → `/`, `+`), `1296` (`w_a + w_b` → `*`) and `1316` (`w_a * λ`).
    ///
    /// Static A, `w_b = 2`: B = q35(z) (`θ ≈ 1.287`), limits `[2, 3]` → `e = 2 − θ ≈ 0.713`,
    /// `λ = −e / 2`, B rotates `−w_b λ = +e` → the angle lands on the limit, exactly 2.
    /// Mutants: `==` / `!` → 1.287; `/` (`2/θ`) → 2.841; `+` → θ+3.287 (wraps to −1.709);
    /// `inv_w = 2` and `e / inv_w` → B rotates 4e → wraps to −2.144; `e + 1/2` → wraps to −2.57;
    /// sign → 0.574; `w_a * w_b = 0` → no move.
    /// Two dynamic bodies, `w_a = 2`, `w_b = 6` (`w = 8`): A rotates `2λ = −e/4`, B rotates
    /// `+6e/8`, relative angle again exactly 2 and A's own angle is `−e/4`.
    #[test]
    fn hinge_min_limit_lands_exactly_on_the_limit() {
        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(fi(2), fi(3));
        let theta = theta35();
        let e = fi(2) - theta;
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, fi(2));
        b[1].rotation = q35(Vec3Fix::UNIT_Z);
        assert!(near(rel_twist(&b, Vec3Fix::UNIT_Z), theta));
        solve_hinge_joint(&j, &mut b, DT);
        let after = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(after, fi(2)), "{after:?} vs 2");
        assert_eq!(b[0].rotation, QuatFix::IDENTITY);

        let mut both = pair(Vec3Fix::ZERO, 1, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut both, fi(2), fi(6));
        both[1].rotation = q35(Vec3Fix::UNIT_Z);
        solve_hinge_joint(&j, &mut both, DT);
        let rel = rel_twist(&both, Vec3Fix::UNIT_Z);
        let a_angle = compute_twist_angle(both[0].rotation, Vec3Fix::UNIT_Z);
        let b_angle = compute_twist_angle(both[1].rotation, Vec3Fix::UNIT_Z);
        assert!(near(rel, fi(2)), "rel {rel:?} vs 2");
        assert!(near(a_angle, -(e / fi(4))), "A {a_angle:?} vs −e/4");
        assert!(
            near(b_angle, theta + (e * fi(3)) / fi(4)),
            "B {b_angle:?} vs θ + 3e/4"
        );
    }

    /// Kills `837:31` (`angle − max` → `+`) and `846` (`error * inv_w` → `/`, `+`).
    ///
    /// Static A, `w_b = 2`, limits `[1/4, 1/2]`: `e = θ − 1/2 ≈ 0.787`, `λ = e / 2`,
    /// B rotates `−2λ = −e` → lands exactly on 1/2.
    /// Mutants: `+` → `e = θ + 1/2` → B ends at `−1/2`; `e / inv_w = 2e` → B rotates `−4e`
    /// → wraps to 2.14; `e + inv_w` → 1.287 − 2.574 = −1.287.
    #[test]
    fn hinge_max_limit_lands_exactly_on_the_limit() {
        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(Fix128::from_ratio(1, 4), Fix128::from_ratio(1, 2));
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, fi(2));
        b[1].rotation = q35(Vec3Fix::UNIT_Z);
        solve_hinge_joint(&j, &mut b, DT);
        let after = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(after, Fix128::from_ratio(1, 2)), "{after:?} vs 1/2");
    }

    /// Negative-angle branch (live since `compute_twist_angle` became signed): B = q35(z)⁻¹
    /// reads `−θ ≈ −1.287`, limits `[−1/2, 1/2]` → `angle < min`, `e = −1/2 + θ ≈ 0.787`,
    /// `λ = −e / w_b`, B rotates `+e` → lands exactly on `−1/2`.
    /// Also pins the sign convention itself: `q` and `−q` (same rotation) read the same angle,
    /// and the inverse reads the negated angle (kills the `w < 0` flip in `compute_twist_angle`,
    /// 1347-1348, and `.double()` at 1352).
    #[test]
    fn hinge_min_limit_with_negative_angle_pushes_toward_min() {
        let theta = theta35();
        let q = q35(Vec3Fix::UNIT_Z);
        let neg_q = QuatFix::new(-q.x, -q.y, -q.z, -q.w);
        assert!(near(compute_twist_angle(q, Vec3Fix::UNIT_Z), theta));
        assert!(near(compute_twist_angle(neg_q, Vec3Fix::UNIT_Z), theta));
        assert!(near(
            compute_twist_angle(q.conjugate(), Vec3Fix::UNIT_Z),
            -theta
        ));

        let j = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(Fix128::from_ratio(-1, 2), Fix128::from_ratio(1, 2));
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, Fix128::ONE);
        b[1].rotation = q.conjugate();
        assert!(near(rel_twist(&b, Vec3Fix::UNIT_Z), -theta));
        solve_hinge_joint(&j, &mut b, DT);
        let after = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(after, Fix128::from_ratio(-1, 2)), "{after:?} vs −1/2");
        // positive side: +θ lands on +1/2
        let mut p = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut p, Fix128::ZERO, Fix128::ONE);
        p[1].rotation = q;
        solve_hinge_joint(&j, &mut p, DT);
        assert!(near(
            rel_twist(&p, Vec3Fix::UNIT_Z),
            Fix128::from_ratio(1, 2)
        ));
    }

    /// Regression pin: with the angle exactly on a limit nothing happens, bit for bit.
    /// (`<` → `<=` / `>` → `>=` at 823 / 836 are equivalent mutants: they enter the branch
    /// with `error = 0`, `λ = 0`, and `rotate_by_angle` returns `q` unchanged for `θ = 0`.)
    #[test]
    fn hinge_limit_equality_is_a_bitwise_no_op() {
        let mk = |min: Fix128, max: Fix128| {
            HingeJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_limits(min, max)
        };
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, fi(3), fi(1));
        b[1].rotation = q35(Vec3Fix::UNIT_Z);
        let before = b[1].rotation;
        let theta = rel_twist(&b, Vec3Fix::UNIT_Z);
        // angle == min
        solve_hinge_joint(&mk(theta, theta + fi(1)), &mut b, DT);
        assert_eq!(b[1].rotation, before, "angle == min must not touch B");
        // angle == max
        solve_hinge_joint(&mk(theta - fi(1), theta), &mut b, DT);
        assert_eq!(b[1].rotation, before, "angle == max must not touch B");
    }

    /// Kills `927:36` (`pos_a + rot(anchor_a)` → `-`) and `930:26` (`anchor_b − anchor_a` → `+`).
    ///
    /// A static at 0 with `local_anchor_a = (0, 1, 0)`; B inv_mass 1 at `(3, 5, 0)`, axis x.
    /// `delta = (3, 4, 0)`, perpendicular part `(0, 4, 0)` (power of two → exact `1/len`)
    /// → B moves to `(3, 1, 0)`. Either mutant yields `delta = (3, 6, 0)` → B near `(3, −1, 0)`.
    #[test]
    fn slider_adds_anchor_a_and_uses_b_minus_a() {
        let j = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, v3i(0, 1, 0), Vec3Fix::ZERO);
        let mut b = pair(Vec3Fix::ZERO, 0, v3i(3, 5, 0), 1);
        solve_slider_joint(&j, &mut b, DT);
        assert_eq!(b[1].position, v3i(3, 1, 0));
    }

    /// Kills `965:45` (`1 / w` → `*`), `966:54` (`error * inv_w` → `/`), `967:20`
    /// (`!inv_mass_a.is_zero()` negation), `969:55` (A `−` → `+`), `969:68` (A `* inv_mass` → `/`),
    /// `973:68` (B `* inv_mass` → `/`) and `984:68` (max branch, A `* inv_mass` → `/`).
    ///
    /// A inv_mass 3 at 0, B inv_mass 5, limits `[−1, 2]`.
    /// min: B at `(−5, 0, 0)` → along −5, `e = 4`, `w = 8`, `correction = 4/8 = 1/2` along x:
    /// A −= 1/2 · 3 = 3/2 → `(−3/2, 0, 0)`, B += 1/2 · 5 = 5/2 → `(−5/2, 0, 0)` (along = −1).
    /// Mutants: `inv_w = 8` / `e / inv_w` → correction 32; `!` → A stays at 0; `+` → A at +3/2;
    /// A `/` → 1/6; B `/` → 1/10.
    /// max: B at `(6, 0, 0)` → `e = 4`, correction 1/2: A += 3/2, B −= 5/2 → along = 2.
    #[test]
    fn slider_limit_corrections_split_by_inverse_mass() {
        let j = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_limits(fi(-1), fi(2));
        let half = Fix128::from_ratio(1, 2);
        let mut under = pair(Vec3Fix::ZERO, 3, v3i(-5, 0, 0), 5);
        solve_slider_joint(&j, &mut under, DT);
        assert_eq!(
            under[0].position,
            Vec3Fix::new(-(fi(3) * half), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            under[1].position,
            Vec3Fix::new(-(fi(5) * half), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(under[1].position.x - under[0].position.x, fi(-1));
        let mut over = pair(Vec3Fix::ZERO, 3, v3i(6, 0, 0), 5);
        solve_slider_joint(&j, &mut over, DT);
        assert_eq!(
            over[0].position,
            Vec3Fix::new(fi(3) * half, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            over[1].position,
            Vec3Fix::new(fi(7) * half, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(over[1].position.x - over[0].position.x, fi(2));
    }

    /// Kills `1015:35` (`v_b − v_a` → `+`).
    ///
    /// rest 4 = distance (spring force 0), damping 2, `v_a = (1, 0, 0)`, `v_b = (3, 0, 0)`:
    /// relative velocity along the normal is `3 − 1 = 2` → `F = 4`; XPBD multiplier with
    /// implicit damping `λ = dt² F / (1 + (k dt² + c dt) w) = (4/16) / (1 + (2/16 + 2/4)·2)
    /// = 1/9` → A `(1/9, 0, 0)`, B `(4 − 1/9, 0, 0)`. Mutant: `3 + 1 = 4` → `F = 8` →
    /// `λ = 2/9` → A `(2/9, 0, 0)`.
    #[test]
    fn spring_damping_uses_relative_velocity_b_minus_a() {
        let j = SpringJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, fi(4), fi(2), fi(2));
        let mut b = pair(Vec3Fix::ZERO, 1, v3i(4, 0, 0), 1);
        b[0].velocity = v3i(1, 0, 0);
        b[1].velocity = v3i(3, 0, 0);
        solve_spring_joint(&j, &mut b, DT);
        assert!(
            near_v(
                b[0].position,
                Vec3Fix::new(Fix128::from_ratio(1, 9), Fix128::ZERO, Fix128::ZERO)
            ),
            "A {:?}",
            b[0].position
        );
        assert!(
            near_v(
                b[1].position,
                Vec3Fix::new(Fix128::from_ratio(35, 9), Fix128::ZERO, Fix128::ZERO)
            ),
            "B {:?}",
            b[1].position
        );
    }

    /// Kills `1115:55` (`angular_compliance / dt²` → `*`), `1115:61` (`dt * dt` → `/`, `+`),
    /// `1159` (`angular_w_sum + angular_compliance` → `-`, `*`) and `1166` (`error / w_ang` → `*`).
    ///
    /// Static A, `w_b = 3/4`, `angular_compliance = 3/64` → term 3/4, `w = 3/2`. z Locked,
    /// B = q35(z): `error = θ`, `λ = 2θ/3`, B rotates `−(3/4)(2θ/3) = −θ/2` → residual `θ/2`.
    /// Mutants: `w ∈ {0.753, 0.797, 0.844}` → residual `{0.004θ, 0.059θ, 0.111θ}`;
    /// `w = 0` → no move (θ); `w = 0.5625` → `−0.33θ`; `error * w` → `λ = 1.5θ` → `−0.125θ`.
    #[test]
    fn d6_locked_angular_axis_uses_angular_compliance_and_inertia_sum() {
        let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
        j.angular_z = D6Motion::Locked;
        j.angular_compliance = Fix128::from_ratio(3, 64);
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, Fix128::from_ratio(3, 4));
        b[1].rotation = q35(Vec3Fix::UNIT_Z);
        let theta = theta35();
        solve_d6_joint(&j, &mut b, DT);
        let after = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(after, theta.half()), "{after:?} vs θ/2");
    }

    /// Kills `1146:30` (`<` → `==`), `1147:31` (`angle − min` → `/`, `+`) and `1149:31`
    /// (`angle − max` → `+`); also exercises the live negative-angle `angle < min` branch.
    ///
    /// Static A, `w_b = 1`, no compliance → `λ = error`, B rotates `−error`: one solve lands
    /// exactly on the violated limit.
    /// min: limits `[2, 3]`, B = q35(z) → `error = θ − 2 < 0` → angle becomes 2.
    /// Mutants: `==` → 1.287; `/` (`θ/2`) → 0.6435; `+` (`θ + 2`) → −2.0.
    /// max: limits `[1/4, 1/2]` → `error = θ − 1/2` → angle 1/2. Mutant `+` → −1/2.
    /// negative: B = q35(z)⁻¹ (angle −θ), limits `[−1/2, 1/2]` → `error = −θ + 1/2` → −1/2.
    #[test]
    fn d6_limited_angular_axis_lands_exactly_on_the_limit() {
        let mk = |min: Fix128, max: Fix128| {
            let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
            j.angular_z = D6Motion::Limited;
            j.angular_limit_min.z = min;
            j.angular_limit_max.z = max;
            j
        };
        let half = Fix128::from_ratio(1, 2);
        let run = |rot: QuatFix, min: Fix128, max: Fix128| {
            let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
            set_inv_inertia(&mut b, Fix128::ZERO, Fix128::ONE);
            b[1].rotation = rot;
            solve_d6_joint(&mk(min, max), &mut b, DT);
            rel_twist(&b, Vec3Fix::UNIT_Z)
        };
        let q = q35(Vec3Fix::UNIT_Z);
        let under = run(q, fi(2), fi(3));
        assert!(near(under, fi(2)), "min: {under:?} vs 2");
        let over = run(q, Fix128::from_ratio(1, 4), half);
        assert!(near(over, half), "max: {over:?} vs 1/2");
        let negative = run(q.conjugate(), -half, half);
        assert!(near(negative, -half), "negative: {negative:?} vs −1/2");
    }

    /// Kills `1225:32` (`cone − limit` → `/`, `+`), `1228:63` (`angular_compliance / dt²` → `*`),
    /// `1228:69` (`dt * dt` → `/`, `+`), `1230` (`+ angular_compliance` → `-`, `*`), `1233:45`
    /// (`1 / w` → `*`) and `1239:27` (`error * inv_w` → `/`, `+`).
    ///
    /// Static A, `w_b = 1/2`, `angular_compliance = 1/64` → term 1/4, `w = 3/4`, `inv_w = 4/3`.
    /// B = q35(x) → cone angle `φ = θ`, `cone_limit = 1/2` → `e = θ − 1/2 ≈ 0.787`,
    /// `λ = 4e/3`, B rotates `−(1/2)(4e/3) = −2e/3` about +x → `φ' = θ − 2e/3 = (θ + 1)/3 ≈ 0.7623`.
    /// Mutants: `/` (`2θ`) → 0.429; `+` → 0.096; `w ∈ {0.501, 0.516, 0.531}` → 0.502 / 0.524 /
    /// 0.546; `w = 1/4` → 0.287; `w = 1/8` → 1.861; `inv_w = 3/4` and `e / inv_w` → 0.992;
    /// `e + inv_w` → 0.227.
    #[test]
    fn cone_limit_correction_uses_angular_compliance_and_inertia_sum() {
        let half = Fix128::from_ratio(1, 2);
        let mut j = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(half, half);
        j.angular_compliance = Fix128::from_ratio(1, 64);
        let mut b = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut b, Fix128::ZERO, half);
        b[1].rotation = q35(Vec3Fix::UNIT_X);
        let theta = theta35();
        assert!(near(tilt_z(&b), theta));
        let expected = (theta + Fix128::ONE) / fi(3);
        solve_cone_twist_joint(&j, &mut b, DT);
        let after = tilt_z(&b);
        assert!(near(after, expected), "{after:?} vs {expected:?}");
        // twist stayed zero: B still rotates about x only
        assert!(near(rel_twist(&b, Vec3Fix::UNIT_Z), Fix128::ZERO));
        // rigid, w_b = 1: one solve lands exactly on the cone limit
        let rigid = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(half, half);
        let mut r = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut r, Fix128::ZERO, Fix128::ONE);
        r[1].rotation = q35(Vec3Fix::UNIT_X);
        solve_cone_twist_joint(&rigid, &mut r, DT);
        assert!(near(tilt_z(&r), half));
    }

    /// Kills `1252:36` (`twist_angle > 0` → `<`, `==`), `1253:25` (`twist − limit` → `/`, `+`),
    /// `1255` (`twist + limit` → `-`, now live), `1258` / `1260` (compliance term) and
    /// `1269` (`error * inv_w`).
    ///
    /// Static A, `w_b = 1/2`, `angular_compliance = 1/64` → `w = 3/4`, `inv_w = 4/3`,
    /// `twist_limit = 1/2`.
    /// positive: B = q35(z), twist `θ`, `e = θ − 1/2`, B rotates `−2e/3` → `(θ + 1)/3 ≈ 0.7623`.
    /// `<` / `==` take the `twist + limit` branch (`e = θ + 1/2`) → 0.096; `/` → −0.429.
    /// negative: B = q35(z)⁻¹, twist `−θ`, `e = −θ + 1/2`, B rotates `+2|e|/3` → `−(θ + 1)/3`.
    /// Mutant `twist − limit` there → `e = −θ − 1/2` → −0.096.
    #[test]
    fn twist_limit_correction_error_arithmetic() {
        let half = Fix128::from_ratio(1, 2);
        let mut j = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(half, half);
        j.angular_compliance = Fix128::from_ratio(1, 64);
        let theta = theta35();
        let expected = (theta + Fix128::ONE) / fi(3);
        let q = q35(Vec3Fix::UNIT_Z);

        let mut pos = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut pos, Fix128::ZERO, half);
        pos[1].rotation = q;
        solve_cone_twist_joint(&j, &mut pos, DT);
        let after = rel_twist(&pos, Vec3Fix::UNIT_Z);
        assert!(near(after, expected), "positive: {after:?} vs {expected:?}");
        assert!(near(tilt_z(&pos), Fix128::ZERO));

        let mut neg = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        set_inv_inertia(&mut neg, Fix128::ZERO, half);
        neg[1].rotation = q.conjugate();
        assert!(near(rel_twist(&neg, Vec3Fix::UNIT_Z), -theta));
        solve_cone_twist_joint(&j, &mut neg, DT);
        let after = rel_twist(&neg, Vec3Fix::UNIT_Z);
        assert!(
            near(after, -expected),
            "negative: {after:?} vs {:?}",
            -expected
        );

        // rigid, w_b = 1: ±θ land exactly on ±1/2
        let rigid = ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(half, half);
        for (rot, limit) in [(q, half), (q.conjugate(), -half)] {
            let mut r = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
            set_inv_inertia(&mut r, Fix128::ZERO, Fix128::ONE);
            r[1].rotation = rot;
            solve_cone_twist_joint(&rigid, &mut r, DT);
            let t = rel_twist(&r, Vec3Fix::UNIT_Z);
            assert!(near(t, limit), "{t:?} vs {limit:?}");
        }
    }

    /// Regression pin: cone / twist angle exactly on the limit is a bitwise no-op.
    /// (`>` → `>=` at 1224 / 1251 are equivalent mutants: `error = 0`, `λ = 0`, and
    /// `rotate_by_angle` returns `q` unchanged for `θ = 0`.)
    #[test]
    fn cone_and_twist_limit_equality_is_a_bitwise_no_op() {
        let mk = |cone: Fix128, twist: Fix128| {
            ConeTwistJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_limits(cone, twist)
        };
        // cone == limit (twist is 0 < π)
        let mut c = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        c[1].rotation = q35(Vec3Fix::UNIT_X);
        let a = c[0].rotation.rotate_vec(Vec3Fix::UNIT_Z).normalize();
        let bz = c[1].rotation.rotate_vec(Vec3Fix::UNIT_Z).normalize();
        let (_, cross_len) = a.cross(bz).normalize_with_length();
        let cone = Fix128::atan2(cross_len, a.dot(bz));
        let before = c[1].rotation;
        solve_cone_twist_joint(&mk(cone, Fix128::PI), &mut c, DT);
        assert_eq!(c[1].rotation, before, "cone == limit must not touch B");
        // twist == limit (cone is 0 < π/2)
        let mut t = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        t[1].rotation = q35(Vec3Fix::UNIT_Z);
        let twist = rel_twist(&t, Vec3Fix::UNIT_Z);
        let before = t[1].rotation;
        solve_cone_twist_joint(&mk(Fix128::HALF_PI, twist), &mut t, DT);
        assert_eq!(t[1].rotation, before, "twist == limit must not touch B");
    }

    /// Kills the anisotropic terms of `angular_inverse_mass` (1282-1285: body-frame
    /// `conjugate()` rotation, `x²·I_x + y²·I_y + z²·I_z` → `-`, `/`), its static
    /// early-return (1279), and the `w_a λ` / `−w_b λ` split in `apply_angular_correction`
    /// (1313-1319).
    ///
    /// A = q35(x), B = q35(z) ⊗ q35(x) (B is A twisted by `θ` about world z), both hinge
    /// local axes are `(0, 24/25, 7/25)` = world z in either body frame, so the axes stay
    /// aligned and the limit branch measures a relative twist of exactly `θ` about world z.
    /// `inv_inertia_a = (5, 2, 7)` → `w_a = (24/25)²·2 + (7/25)²·7 = (1152 + 343)/625 = 2.392`,
    /// `inv_inertia_b = (11, 13, 3)` → `w_b = (24/25)²·13 + (7/25)²·3 = (7488 + 147)/625 = 12.216`,
    /// `w = 9130/625`. Limits `[2, 3]`: `e = 2 − θ`, A rotates `−w_a e / w` about z, B
    /// `+w_b e / w`; the relative angle lands exactly on 2 and A's rotation is `−1495 e / 9130`.
    /// (Isotropic-inertia mutants would give `w_a = 5` or `7`, `w_b = 11` or `3`; dropping
    /// `conjugate()` rotates the axis the wrong way: `(0, −24/25, 7/25)` has the same squares
    /// here, which is why the sign of A's rotation and the `x²I_x − y²I_y` mutants are pinned
    /// through `w_a` / `w_b` and the exact landing on 2.)
    #[test]
    fn angular_inverse_mass_is_anisotropic_and_split_by_body() {
        let local_axis = Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_ratio(24, 25),
            Fix128::from_ratio(7, 25),
        );
        let j = HingeJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, local_axis, local_axis)
            .with_limits(fi(2), fi(3));
        let theta = theta35();
        let e = fi(2) - theta;
        let qa = q35(Vec3Fix::UNIT_X);
        let mut b = pair(Vec3Fix::ZERO, 1, Vec3Fix::ZERO, 1);
        b[0].inv_inertia = v3i(5, 2, 7);
        b[1].inv_inertia = v3i(11, 13, 3);
        b[0].rotation = qa;
        b[1].rotation = q35(Vec3Fix::UNIT_Z).mul(qa).normalize();
        assert!(near_v(qa.rotate_vec(local_axis), Vec3Fix::UNIT_Z));
        assert!(near_v(
            b[1].rotation.rotate_vec(local_axis),
            Vec3Fix::UNIT_Z
        ));
        assert!(near(rel_twist(&b, Vec3Fix::UNIT_Z), theta));
        let w_a = Fix128::from_ratio(1152 + 343, 625);
        let w_b = Fix128::from_ratio(7488 + 147, 625);
        assert!(near(angular_inverse_mass(&b[0], Vec3Fix::UNIT_Z), w_a));
        assert!(near(angular_inverse_mass(&b[1], Vec3Fix::UNIT_Z), w_b));
        let a_before = b[0].rotation;
        solve_hinge_joint(&j, &mut b, DT);
        let rel = rel_twist(&b, Vec3Fix::UNIT_Z);
        assert!(near(rel, fi(2)), "rel {rel:?} vs 2");
        let a_delta = b[0].rotation.mul(a_before.conjugate());
        let a_angle = compute_twist_angle(a_delta, Vec3Fix::UNIT_Z);
        let expected_a = -(w_a * e) / (w_a + w_b);
        assert!(near(a_angle, expected_a), "A {a_angle:?} vs {expected_a:?}");
        // A rotated about world z only: its own z axis direction is unchanged
        assert!(near_v(
            b[0].rotation.rotate_vec(local_axis),
            Vec3Fix::UNIT_Z
        ));
        // static A contributes 0 even with non-zero inv_inertia
        let mut s = pair(Vec3Fix::ZERO, 0, Vec3Fix::ZERO, 1);
        s[0].inv_inertia = v3i(5, 2, 7);
        assert_eq!(angular_inverse_mass(&s[0], Vec3Fix::UNIT_Z), Fix128::ZERO);
    }

    /// oracle: for an isotropic `inv_inertia = (c, c, c)` the generalised
    /// inverse mass `n · R diag(c, c, c) Rᵀ n` is `c (n·n)` for every
    /// rotation; the value must be the orientation-free sum
    /// `n.x²·c + n.y²·c + n.z²·c`, bit for bit, for 64 non-identity
    /// orientations (the body-frame rotation of `n` rounds differently for
    /// each of them).
    #[test]
    fn angular_inverse_mass_of_an_isotropic_body_does_not_depend_on_orientation() {
        let c = Fix128::from_ratio(5, 2);
        let n = Vec3Fix::new(
            Fix128::from_ratio(3, 7),
            Fix128::from_ratio(-11, 13),
            Fix128::from_ratio(1, 3),
        );
        let expected = n.x * n.x * c + n.y * n.y * c + n.z * n.z * c;
        let mut differ = 0;
        for i in 0..16_i64 {
            let axis = Vec3Fix::new(
                Fix128::from_ratio(1 + i, 3),
                Fix128::from_ratio(2 - i, 5),
                Fix128::from_ratio(3 + 2 * i, 7),
            )
            .normalize();
            for k in 1..=4_i64 {
                let q =
                    QuatFix::from_axis_angle(axis, Fix128::from_ratio(7 * k + i, 9)).normalize();
                let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
                b.inv_inertia = Vec3Fix::new(c, c, c);
                b.rotation = q;
                differ += usize::from(angular_inverse_mass(&b, n) != expected);
            }
        }
        assert_eq!(differ, 0, "{differ}/64 orientations differ from c·(n·n)");
    }
}
