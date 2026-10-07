//! Extended Joint Types for Mechanical Linkages
//!
//! Additional joint constraints beyond the core set in [`crate::joint`]:
//!
//! - **`PulleyJoint`**: Two-body pulley with configurable ratio
//! - **`GearJoint`**: Couples two hinge joint angles
//! - **`WeldJoint`**: Rigid weld with breakable force and torque thresholds
//! - **`RackAndPinionJoint`**: Converts linear motion to rotary motion
//! - **`MouseJoint`**: Soft constraint that drives a body toward a target position
//!
//! All constraints use XPBD-style position-level corrections with compliance.

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::motor::shortest_arc;
use crate::solver::RigidBody;

// ============================================================================
// PulleyJoint
// ============================================================================

/// Pulley joint constraining two bodies via a virtual pulley system.
///
/// The constraint maintains: `len_a + ratio * len_b = total_length`
/// where `len_a` is the rope length from `ground_anchor_a` to `anchor_a`
/// and `len_b` is the rope length from `ground_anchor_b` to `anchor_b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PulleyJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Attachment point on body A (world space)
    pub anchor_a: Vec3Fix,
    /// Attachment point on body B (world space)
    pub anchor_b: Vec3Fix,
    /// Fixed pulley attachment point for body A's rope (world space)
    pub ground_anchor_a: Vec3Fix,
    /// Fixed pulley attachment point for body B's rope (world space)
    pub ground_anchor_b: Vec3Fix,
    /// Pulley ratio: length_a + ratio * length_b = constant
    pub ratio: Fix128,
    /// Compliance (inverse stiffness, 0 = rigid)
    pub compliance: Fix128,
}

impl PulleyJoint {
    /// Create a new pulley joint.
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        ground_anchor_a: Vec3Fix,
        ground_anchor_b: Vec3Fix,
        ratio: Fix128,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            ground_anchor_a,
            ground_anchor_b,
            ratio,
            compliance: Fix128::ZERO,
        }
    }

    /// Compute the current total rope length: `len_a + ratio * len_b`.
    #[must_use]
    pub fn total_length(&self, bodies: &[RigidBody]) -> Fix128 {
        let world_a =
            bodies[self.body_a].position + bodies[self.body_a].rotation.rotate_vec(self.anchor_a);
        let world_b =
            bodies[self.body_b].position + bodies[self.body_b].rotation.rotate_vec(self.anchor_b);

        let len_a = (world_a - self.ground_anchor_a).length();
        let len_b = (world_b - self.ground_anchor_b).length();

        len_a + self.ratio * len_b
    }

    /// Set compliance (inverse stiffness).
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

// ============================================================================
// GearJoint
// ============================================================================

/// Gear joint coupling the angular displacement of two hinge joints.
///
/// Constraint: `angle_a + ratio * angle_b = constant`
///
/// The joint references two hinge joints by index and enforces a fixed
/// ratio between their angular displacements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GearJoint {
    /// Index of the first body (connected to hinge A)
    pub body_a: usize,
    /// Index of the second body (connected to hinge B)
    pub body_b: usize,
    /// Index of the first hinge joint in the world joint list
    pub joint_a: usize,
    /// Index of the second hinge joint in the world joint list
    pub joint_b: usize,
    /// Gear ratio: angle_a + ratio * angle_b = constant
    pub ratio: Fix128,
    /// Compliance (inverse stiffness, 0 = rigid)
    pub compliance: Fix128,
}

impl GearJoint {
    /// Create a new gear joint coupling two hinge joints.
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        joint_a: usize,
        joint_b: usize,
        ratio: Fix128,
    ) -> Self {
        Self {
            body_a,
            body_b,
            joint_a,
            joint_b,
            ratio,
            compliance: Fix128::ZERO,
        }
    }

    /// Set compliance (inverse stiffness).
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

// ============================================================================
// WeldJoint
// ============================================================================

/// Weld joint that rigidly locks both position and rotation between two bodies.
///
/// Unlike `FixedJoint` from the core module, this variant supports separate
/// break-force and break-torque thresholds so the weld can snap under
/// translational stress, rotational stress, or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeldJoint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Relative rotation of body B in body A's frame at creation time
    pub local_rotation: QuatFix,
    /// Compliance (inverse stiffness, 0 = rigid)
    pub compliance: Fix128,
    /// Maximum translational force before the joint breaks (None = unbreakable)
    pub break_force: Option<Fix128>,
    /// Maximum torque before the joint breaks (None = unbreakable)
    pub break_torque: Option<Fix128>,
}

impl WeldJoint {
    /// Create a new weld joint.
    #[inline]
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        local_anchor_a: Vec3Fix,
        local_anchor_b: Vec3Fix,
        local_rotation: QuatFix,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            local_rotation,
            compliance: Fix128::ZERO,
            break_force: None,
            break_torque: None,
        }
    }

    /// Set break force threshold.
    #[must_use]
    pub const fn with_break_force(mut self, force: Fix128) -> Self {
        self.break_force = Some(force);
        self
    }

    /// Set break torque threshold.
    #[must_use]
    pub const fn with_break_torque(mut self, torque: Fix128) -> Self {
        self.break_torque = Some(torque);
        self
    }

    /// Set compliance (inverse stiffness).
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }

    /// Compute translational constraint force (anchor separation distance).
    #[must_use]
    pub fn compute_force(&self, bodies: &[RigidBody]) -> Fix128 {
        let body_a = &bodies[self.body_a];
        let body_b = &bodies[self.body_b];
        let anchor_a = body_a.position + body_a.rotation.rotate_vec(self.local_anchor_a);
        let anchor_b = body_b.position + body_b.rotation.rotate_vec(self.local_anchor_b);
        (anchor_b - anchor_a).length()
    }

    /// Compute rotational constraint error (angular deviation magnitude).
    #[must_use]
    pub fn compute_torque(&self, bodies: &[RigidBody]) -> Fix128 {
        let body_a = &bodies[self.body_a];
        let body_b = &bodies[self.body_b];
        let target_rot_b = body_a.rotation.mul(self.local_rotation);
        let rot_error = body_b.rotation.mul(target_rot_b.conjugate());
        Vec3Fix::new(rot_error.x, rot_error.y, rot_error.z).length()
    }

    /// Check if the joint should break under current forces.
    #[must_use]
    pub fn is_broken(&self, bodies: &[RigidBody]) -> bool {
        if let Some(max_force) = self.break_force {
            if self.compute_force(bodies) > max_force {
                return true;
            }
        }
        if let Some(max_torque) = self.break_torque {
            if self.compute_torque(bodies) > max_torque {
                return true;
            }
        }
        false
    }
}

// ============================================================================
// RackAndPinionJoint
// ============================================================================

/// Rack-and-pinion joint converting linear displacement to angular displacement.
///
/// Constraint: `linear_displacement = ratio * angular_displacement`
///
/// The rack body translates along `rack_axis` while the pinion body
/// rotates around `pinion_axis`. The ratio converts between
/// linear units and radians.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RackAndPinionJoint {
    /// Index of the rack body (linear motion)
    pub body_rack: usize,
    /// Index of the pinion body (rotary motion)
    pub body_pinion: usize,
    /// Axis of linear travel in the rack body's local space
    pub rack_axis: Vec3Fix,
    /// Axis of rotation in the pinion body's local space
    pub pinion_axis: Vec3Fix,
    /// Conversion ratio: linear_displacement = ratio * angular_displacement
    pub ratio: Fix128,
    /// Compliance (inverse stiffness, 0 = rigid)
    pub compliance: Fix128,
}

impl RackAndPinionJoint {
    /// Create a new rack-and-pinion joint.
    #[inline]
    #[must_use]
    pub const fn new(
        body_rack: usize,
        body_pinion: usize,
        rack_axis: Vec3Fix,
        pinion_axis: Vec3Fix,
        ratio: Fix128,
    ) -> Self {
        Self {
            body_rack,
            body_pinion,
            rack_axis,
            pinion_axis,
            ratio,
            compliance: Fix128::ZERO,
        }
    }

    /// Set compliance (inverse stiffness).
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

// ============================================================================
// MouseJoint
// ============================================================================

/// Mouse joint (spring-damper) that drives a body toward a target position.
///
/// Used for interactive dragging. The body is pulled toward `target_position`
/// with configurable stiffness and damping, capped by `max_force`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MouseJoint {
    /// Index of the controlled body
    pub body: usize,
    /// Target position in world space
    pub target_position: Vec3Fix,
    /// Maximum force magnitude applied per step
    pub max_force: Fix128,
    /// Spring stiffness (force per unit displacement)
    pub stiffness: Fix128,
    /// Damping coefficient (force per unit velocity)
    pub damping: Fix128,
}

impl MouseJoint {
    /// Create a new mouse joint.
    #[inline]
    #[must_use]
    pub const fn new(
        body: usize,
        target_position: Vec3Fix,
        max_force: Fix128,
        stiffness: Fix128,
        damping: Fix128,
    ) -> Self {
        Self {
            body,
            target_position,
            max_force,
            stiffness,
            damping,
        }
    }

    /// Update the target position (e.g. following mouse cursor).
    pub fn set_target(&mut self, target: Vec3Fix) {
        self.target_position = target;
    }
}

// ============================================================================
// ExtraJoint enum
// ============================================================================

/// Unified enum for extended joint types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtraJoint {
    /// Pulley constraint
    Pulley(PulleyJoint),
    /// Gear coupling
    Gear(GearJoint),
    /// Rigid weld with break thresholds
    Weld(WeldJoint),
    /// Rack-and-pinion linkage
    RackAndPinion(RackAndPinionJoint),
    /// Mouse / spring-damper target tracking
    Mouse(MouseJoint),
}

// ============================================================================
// Solver
// ============================================================================

/// Solve all extra joints for one XPBD iteration.
///
/// Modifies body positions and rotations in-place to satisfy constraints.
/// Broken weld joints are skipped but not removed from the slice.
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
pub fn solve_extra_joints(bodies: &mut [RigidBody], joints: &[ExtraJoint], dt: Fix128) {
    for joint in joints {
        match joint {
            ExtraJoint::Pulley(j) => solve_pulley(j, bodies, dt),
            ExtraJoint::Gear(j) => solve_gear(j, bodies, dt),
            ExtraJoint::Weld(j) => solve_weld(j, bodies, dt),
            ExtraJoint::RackAndPinion(j) => solve_rack_and_pinion(j, bodies, dt),
            ExtraJoint::Mouse(j) => solve_mouse(j, bodies, dt),
        }
    }
}

/// Solve pulley joint inside [`solve_extra_joints`].
///
/// # Claims
///
/// - The rope keeps the total `len_a + ratio · len_b` it had at the start of
///   the step, evaluated at the bodies' `prev_position` / `prev_rotation`; one
///   XPBD projection along the two ropes (split by inverse mass, the second
///   scaled by `ratio`) moves the bodies back onto it. A pulley whose bodies
///   have not moved since the previous pose is left as it is. With a
///   `compliance > 0` the projection closes only part of the gap, and the rest
///   becomes the next step's starting length: a compliant pulley stretches for
///   good rather than springing back (a persistent rest length would need a
///   field on `PulleyJoint`).
/// - To hold an explicit rest length instead use [`solve_pulley_to_length`].
fn solve_pulley(joint: &PulleyJoint, bodies: &mut [RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    let world_a = body_a.position + body_a.rotation.rotate_vec(joint.anchor_a);
    let world_b = body_b.position + body_b.rotation.rotate_vec(joint.anchor_b);

    let delta_a = world_a - joint.ground_anchor_a;
    let delta_b = world_b - joint.ground_anchor_b;

    let (dir_a, len_a) = delta_a.normalize_with_length();
    let (dir_b, len_b) = delta_b.normalize_with_length();

    if len_a.is_zero() && len_b.is_zero() {
        return;
    }

    // The rope keeps the total it had at the start of the step (the bodies'
    // previous pose): C = (len_a + ratio len_b) − total at prev_position. One
    // XPBD projection along the two ropes moves the bodies back onto it, so
    // whatever moved them during the step cannot lengthen or shorten the rope.
    let prev_a = body_a.prev_position + body_a.prev_rotation.rotate_vec(joint.anchor_a);
    let prev_b = body_b.prev_position + body_b.prev_rotation.rotate_vec(joint.anchor_b);
    let rest_total = (prev_a - joint.ground_anchor_a).length()
        + joint.ratio * (prev_b - joint.ground_anchor_b).length();
    let error = len_a + joint.ratio * len_b - rest_total;
    if error.is_zero() {
        return;
    }

    // Effective mass: w_a + ratio^2 * w_b
    let compliance_term = joint.compliance / (dt * dt);
    let w_a = body_a.inv_mass;
    let w_b = body_b.inv_mass;
    let w_sum = w_a + joint.ratio * joint.ratio * w_b + compliance_term;
    if w_sum.is_zero() {
        return;
    }
    let lambda = error / w_sum;
    if !w_a.is_zero() {
        bodies[joint.body_a].position = bodies[joint.body_a].position - dir_a * (lambda * w_a);
    }
    if !w_b.is_zero() {
        bodies[joint.body_b].position =
            bodies[joint.body_b].position - dir_b * (lambda * joint.ratio * w_b);
    }
}

/// Constrain the pulley rope to a maximum total length (rope cannot stretch).
///
/// The constraint is `C = len_a + ratio * len_b - rest_length <= 0` with
/// `len_a = |p_a + R_a anchor_a - ground_anchor_a|` and
/// `len_b = |p_b + R_b anchor_b - ground_anchor_b|`. One XPBD position
/// projection is applied along the two rope directions:
/// `lambda = C / (w_a + ratio^2 w_b + compliance / dt^2)`,
/// `p_a -= dir_a lambda w_a`, `p_b -= ratio dir_b lambda w_b`.
///
/// # Claims
///
/// - Units: lengths in the same unit as the body positions, `dt` in seconds.
/// - Only an excess (`C > 0`) is corrected; a slack rope (`C <= 0`) moves nothing.
/// - Corrections are split by inverse mass; a body with `inv_mass = 0` does not move.
/// - Positional only: the anchors' lever arms are used for the length but no
///   rotation is applied.
/// - If both ropes have zero length, or the generalized inverse mass is zero,
///   nothing is moved.
pub fn solve_pulley_to_length(
    joint: &PulleyJoint,
    bodies: &mut [RigidBody],
    rest_length: Fix128,
    dt: Fix128,
) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    let world_a = body_a.position + body_a.rotation.rotate_vec(joint.anchor_a);
    let world_b = body_b.position + body_b.rotation.rotate_vec(joint.anchor_b);

    let (dir_a, len_a) = (world_a - joint.ground_anchor_a).normalize_with_length();
    let (dir_b, len_b) = (world_b - joint.ground_anchor_b).normalize_with_length();

    if len_a.is_zero() && len_b.is_zero() {
        return;
    }

    let c = len_a + joint.ratio * len_b - rest_length;
    if c <= Fix128::ZERO {
        return;
    }

    let w_a = body_a.inv_mass;
    let w_b = body_b.inv_mass;
    let w_sum = w_a + joint.ratio * joint.ratio * w_b + joint.compliance / (dt * dt);
    if w_sum.is_zero() {
        return;
    }
    let lambda = c / w_sum;

    if !w_a.is_zero() {
        bodies[joint.body_a].position = bodies[joint.body_a].position - dir_a * (lambda * w_a);
    }
    if !w_b.is_zero() {
        bodies[joint.body_b].position =
            bodies[joint.body_b].position - dir_b * (lambda * joint.ratio * w_b);
    }
}

/// Solve gear joint: `angle_a + ratio * angle_b = constant`.
fn solve_gear(joint: &GearJoint, bodies: &mut [RigidBody], dt: Fix128) {
    let body_a = bodies[joint.body_a];
    let body_b = bodies[joint.body_b];

    // Extract angular displacements around each body's local Z axis
    // (simplified: use the body's angular velocity integrated over dt as proxy)
    let rel_quat_a = body_a.rotation.mul(body_a.prev_rotation.conjugate());
    let rel_quat_b = body_b.rotation.mul(body_b.prev_rotation.conjugate());

    // Signed twists about z, the axis the correction below rotates about
    let angle_a = extract_angle(rel_quat_a);
    let angle_b = extract_angle(rel_quat_b);

    // Constraint: angle_a + ratio * angle_b should be zero (relative to initial)
    let error = angle_a + joint.ratio * angle_b;

    if error.abs().is_zero() {
        return;
    }

    let compliance_term = joint.compliance / (dt * dt);
    let w_a = inv_inertia_about(&body_a, Vec3Fix::UNIT_Z);
    let w_b = inv_inertia_about(&body_b, Vec3Fix::UNIT_Z);
    let w_sum = w_a + joint.ratio * joint.ratio * w_b + compliance_term;

    if w_sum.is_zero() {
        return;
    }

    let inv_w = Fix128::ONE / w_sum;
    let lambda = error * inv_w;

    // Apply angular corrections around the Z axis (simplified).
    // grad C = (1, ratio), so d_angle_a = -lambda w_a and d_angle_b = -lambda ratio w_b.
    if !body_a.inv_mass.is_zero() {
        let half_lambda = (lambda * w_a).half();
        let dq = QuatFix::new(Fix128::ZERO, Fix128::ZERO, -half_lambda, Fix128::ONE);
        bodies[joint.body_a].rotation = dq.mul(bodies[joint.body_a].rotation).normalize();
    }
    if !body_b.inv_mass.is_zero() {
        let scaled = lambda * joint.ratio * w_b;
        let half_lambda = scaled.half();
        let dq = QuatFix::new(Fix128::ZERO, Fix128::ZERO, -half_lambda, Fix128::ONE);
        bodies[joint.body_b].rotation = dq.mul(bodies[joint.body_b].rotation).normalize();
    }
}

/// Solve weld joint: lock position and rotation, skip if broken.
fn solve_weld(joint: &WeldJoint, bodies: &mut [RigidBody], dt: Fix128) {
    // Check break conditions before solving
    if joint.is_broken(bodies) {
        return;
    }

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
    let target_rot_b = body_a.rotation.mul(joint.local_rotation);
    let rot_error = shortest_arc(body_b.rotation.mul(target_rot_b.conjugate()));
    let error_vec = Vec3Fix::new(rot_error.x, rot_error.y, rot_error.z);
    let (correction_axis, error_mag) = error_vec.normalize_with_length();

    if !error_mag.is_zero() {
        // XPBD with the full 3x3 effective mass: K = W_a + W_b + compliance/dt^2
        // (W the world inverse inertia, zero for a static body), lambda = K^-1
        // theta, and the bodies turn by +W_a lambda and -W_b lambda. The
        // correction's angular momentum is lambda - lambda = 0 for any inertia,
        // and a rigid weld closes the error to first order in one solve
        // ((W_a + W_b) lambda = theta); two moving anisotropic bodies leave a
        // second-order rest, as their two turns do not commute, gone in 2-3
        let compliance_term = joint.compliance / (dt * dt);
        let world = |body: &RigidBody| crate::math::Mat3Fix {
            col0: world_inv_inertia_times(body, Vec3Fix::UNIT_X),
            col1: world_inv_inertia_times(body, Vec3Fix::UNIT_Y),
            col2: world_inv_inertia_times(body, Vec3Fix::UNIT_Z),
        };
        let (wa, wb) = (world(&body_a), world(&body_b));
        let k = crate::math::Mat3Fix {
            col0: wa.col0 + wb.col0 + Vec3Fix::new(compliance_term, Fix128::ZERO, Fix128::ZERO),
            col1: wa.col1 + wb.col1 + Vec3Fix::new(Fix128::ZERO, compliance_term, Fix128::ZERO),
            col2: wa.col2 + wb.col2 + Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, compliance_term),
        };
        let theta = correction_axis * (error_mag * Fix128::from_int(2));
        if let Some((turn_a, turn_b)) = weld_turns(k, wa, wb, theta) {
            rotate_by(bodies, joint.body_a, turn_a);
            rotate_by(bodies, joint.body_b, turn_b);
        }
    }
}

/// The weld's rotation corrections `(+W_a lambda, -W_b lambda)` for the error
/// rotation vector `theta`, with `K lambda = theta` solved in the eigenbasis of
/// `K` ([`crate::mass_properties::principal_axes`], which scales `K` to unit
/// size, so the answer does not depend on how heavy or light the bodies are).
///
/// - Directions whose eigenvalue is at most `max(2^-50 largest, 2^-56)` are
///   locked for both bodies (inverse inertia zero there, as for bodies held to
///   planar rotation about any axis). The eigenvalue of such a direction comes
///   back as rounding, positive or negative, a few steps of `2^-64` in `K`'s own
///   units; read as free, it turned bodies by up to 0.06 rad for an error that
///   lay wholly in the locked direction. No correction is asked for along a
///   locked direction and the rest of the error is closed.
/// - The undamped solution is used unless a body would turn by more than
///   `max(4 |theta|, 1/8 rad)`, which every well-conditioned weld stays within
///   (rods whose cheap long axes take several times the error included).
/// - Beyond that the correction leaves the small-angle range it is linearised
///   in: when the nearly locked axes of the two bodies are slightly apart the
///   linear solution turns both by radians (0.01 rad of error gave 1.5 rad per
///   body). The eigenvalues are then damped by `mu`, starting at `2^-10` of the
///   largest and doubling, until the turns fit; a fixed count of steps keeps it
///   deterministic, and if none fits no correction is applied.
fn weld_turns(
    k: crate::math::Mat3Fix,
    wa: crate::math::Mat3Fix,
    wb: crate::math::Mat3Fix,
    theta: Vec3Fix,
) -> Option<(Vec3Fix, Vec3Fix)> {
    let (moments, axes) = crate::mass_properties::principal_axes(k);
    let pairs = [
        (moments.x, axes.col0),
        (moments.y, axes.col1),
        (moments.z, axes.col2),
    ];
    let largest = pairs
        .iter()
        .map(|&(e, _)| e)
        .fold(Fix128::ZERO, |m, e| if e > m { e } else { m });
    if largest <= Fix128::ZERO {
        return None;
    }
    let locked = {
        let relative = largest / Fix128::from_int(1i64 << 50);
        let floor = Fix128::ONE / Fix128::from_int(1i64 << 56);
        if relative > floor {
            relative
        } else {
            floor
        }
    };
    let theta_len = theta.length();
    let limit = {
        let four = theta_len * Fix128::from_int(4);
        let eighth = Fix128::from_ratio(1, 8);
        if four > eighth {
            four
        } else {
            eighth
        }
    };
    let limit_sq = limit * limit;
    let turns = |mu: Fix128| {
        let mut lambda = Vec3Fix::ZERO;
        for &(e, q) in &pairs {
            if e > locked {
                lambda = lambda + q * (q.dot(theta) / (e + mu));
            }
        }
        let (turn_a, turn_b) = (wa.mul_vec(lambda), -wb.mul_vec(lambda));
        (turn_a.length_squared() <= limit_sq && turn_b.length_squared() <= limit_sq)
            .then_some((turn_a, turn_b))
    };
    if let Some(found) = turns(Fix128::ZERO) {
        return Some(found);
    }
    let mut mu = largest / Fix128::from_int(1 << 10);
    for _ in 0..64 {
        if let Some(found) = turns(mu) {
            return Some(found);
        }
        mu = mu * Fix128::from_int(2);
    }
    None
}

/// Solve rack-and-pinion: `linear_displacement = ratio * angular_displacement`.
fn solve_rack_and_pinion(joint: &RackAndPinionJoint, bodies: &mut [RigidBody], dt: Fix128) {
    let body_rack = bodies[joint.body_rack];
    let body_pinion = bodies[joint.body_pinion];

    // Linear displacement of rack along its axis
    let world_rack_axis = body_rack.rotation.rotate_vec(joint.rack_axis).normalize();
    let linear_disp = (body_rack.position - body_rack.prev_position).dot(world_rack_axis);

    // Angular displacement of pinion around its axis
    let rel_quat = body_pinion
        .rotation
        .mul(body_pinion.prev_rotation.conjugate());
    let world_pinion_axis = body_pinion
        .rotation
        .rotate_vec(joint.pinion_axis)
        .normalize();
    let angular_disp = extract_angle_around_axis(rel_quat, world_pinion_axis);

    // Constraint: linear_disp - ratio * angular_disp = 0
    let error = linear_disp - joint.ratio * angular_disp;

    if error.abs().is_zero() {
        return;
    }

    let compliance_term = joint.compliance / (dt * dt);
    let w_linear = body_rack.inv_mass;
    let w_angular = inv_inertia_about(&body_pinion, world_pinion_axis);
    let w_sum = w_linear + joint.ratio * joint.ratio * w_angular + compliance_term;

    if w_sum.is_zero() {
        return;
    }

    let inv_w = Fix128::ONE / w_sum;
    let lambda = error * inv_w;

    // Correct rack position along rack axis
    if !body_rack.inv_mass.is_zero() {
        bodies[joint.body_rack].position =
            bodies[joint.body_rack].position - world_rack_axis * (lambda * w_linear);
    }

    // Correct pinion rotation around pinion axis
    if !body_pinion.inv_mass.is_zero() {
        let angular_correction = (lambda * joint.ratio * w_angular).half();
        let dq = QuatFix::new(
            world_pinion_axis.x * angular_correction,
            world_pinion_axis.y * angular_correction,
            world_pinion_axis.z * angular_correction,
            Fix128::ONE,
        );
        bodies[joint.body_pinion].rotation = dq.mul(bodies[joint.body_pinion].rotation).normalize();
    }
}

/// Solve mouse joint: soft spring-damper toward target.
fn solve_mouse(joint: &MouseJoint, bodies: &mut [RigidBody], dt: Fix128) {
    let body = bodies[joint.body];

    if body.inv_mass.is_zero() {
        return;
    }

    let delta = joint.target_position - body.position;
    let (direction, distance) = delta.normalize_with_length();

    if distance.is_zero() {
        return;
    }

    // Spring force: F = stiffness * displacement
    let spring_force = joint.stiffness * distance;

    // Damping force: F = -damping * velocity_along_direction
    let vel_along = body.velocity.dot(direction);
    let damping_force = joint.damping * vel_along;

    let total_force = spring_force - damping_force;

    // Clamp to max_force
    let clamped_force = if total_force > joint.max_force {
        joint.max_force
    } else if total_force < -joint.max_force {
        -joint.max_force
    } else {
        total_force
    };

    // Apply as a position-based step: the force acting for `dt` on inverse mass
    // `w` changes the position by `F dt^2 w` (`F dt w` would be a velocity change)
    let impulse = direction * (clamped_force * dt);
    bodies[joint.body].position = bodies[joint.body].position + impulse * (dt * body.inv_mass);
}

// ============================================================================
// Helpers
// ============================================================================

/// Minimum xyz-length-squared below which a quaternion is treated as zero rotation.
/// Prevents noise from near-identity quaternions.
const ANGLE_EPSILON_SQ: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0000_0001_0000,
};

/// Extract the signed rotation angle about the z axis from a quaternion.
///
/// This is the axis [`solve_gear`] applies its correction about, so reading and
/// writing use the same sign convention. See [`extract_angle_around_axis`].
#[must_use]
fn extract_angle(q: QuatFix) -> Fix128 {
    extract_angle_around_axis(q, Vec3Fix::UNIT_Z)
}

/// Extract the signed rotation angle around a specific axis from a quaternion.
///
/// Projects the quaternion's imaginary part onto the given (unit) axis to isolate
/// the twist component and returns `2 * (xyz . axis)`, the small-angle form of
/// `2 * atan2(xyz . axis, w)`, read from the `w >= 0` representative so `q` and
/// `-q` give the same angle. A rotation in the negative sense about `axis` returns
/// a negative angle.
#[must_use]
fn extract_angle_around_axis(q: QuatFix, axis: Vec3Fix) -> Fix128 {
    let q = shortest_arc(q);
    let qv = Vec3Fix::new(q.x, q.y, q.z);
    let twist = qv.dot(axis);
    if twist * twist < ANGLE_EPSILON_SQ {
        return Fix128::ZERO;
    }
    twist.double()
}

/// Inverse moment of inertia of `body` about the world axis `axis` (unit):
/// `n_localᵀ diag(inv_inertia) n_local` with `n_local` the axis in the body
/// frame. Zero for a static body (it takes no correction).
fn inv_inertia_about(body: &RigidBody, axis: Vec3Fix) -> Fix128 {
    if body.inv_mass.is_zero() {
        return Fix128::ZERO;
    }
    let n = body.rotation.conjugate().rotate_vec(axis);
    let i = body.inv_inertia;
    i.x * n.x * n.x + i.y * n.y * n.y + i.z * n.z * n.z
}

/// `I⁻¹ v` with the body's inverse inertia in world space (`R diag(i) Rᵀ v`);
/// zero for a static body.
fn world_inv_inertia_times(body: &RigidBody, v: Vec3Fix) -> Vec3Fix {
    if body.inv_mass.is_zero() {
        return Vec3Fix::ZERO;
    }
    let q = body.rotation;
    let local = q.conjugate().rotate_vec(v);
    let i = body.inv_inertia;
    q.rotate_vec(Vec3Fix::new(i.x * local.x, i.y * local.y, i.z * local.z))
}

/// Turn a dynamic body by the small rotation vector `w` (radians).
fn rotate_by(bodies: &mut [RigidBody], idx: usize, w: Vec3Fix) {
    if bodies[idx].inv_mass.is_zero() || w.length_squared().is_zero() {
        return;
    }
    let dq = QuatFix::new(w.x.half(), w.y.half(), w.z.half(), Fix128::ONE);
    bodies[idx].rotation = dq.mul(bodies[idx].rotation).normalize();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn dt() -> Fix128 {
        Fix128::from_ratio(1, 60)
    }

    // --- PulleyJoint ---

    #[test]
    fn test_pulley_joint_creation() {
        let pj = PulleyJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 10, 0),
            Fix128::from_int(2),
        );
        assert_eq!(pj.body_a, 0);
        assert_eq!(pj.body_b, 1);
        assert_eq!(pj.ratio.hi, 2);
        assert!(pj.compliance.is_zero());
    }

    #[test]
    fn test_pulley_joint_total_length() {
        let bodies = vec![
            RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(0, 3, 0), Fix128::ONE),
        ];
        let pj = PulleyJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 10, 0),
            Fix128::ONE,
        );
        let total = pj.total_length(&bodies);
        // len_a = |5 - 10| = 5, len_b = |3 - 10| = 7, total = 5 + 1*7 = 12
        assert_eq!(total.hi, 12);
    }

    #[test]
    fn test_pulley_joint_with_compliance() {
        let pj = PulleyJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::ONE,
        )
        .with_compliance(Fix128::from_int(5));
        assert_eq!(pj.compliance.hi, 5);
    }

    #[test]
    fn test_pulley_solve_does_not_panic() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(0, 3, 0), Fix128::ONE),
        ];
        let joints = vec![ExtraJoint::Pulley(PulleyJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 10, 0),
            Fix128::from_int(2),
        ))];
        solve_extra_joints(&mut bodies, &joints, dt());
    }

    #[test]
    fn test_pulley_static_bodies() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)),
            RigidBody::new_static(Vec3Fix::from_int(0, 3, 0)),
        ];
        let joints = vec![ExtraJoint::Pulley(PulleyJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 10, 0),
            Fix128::ONE,
        ))];
        let pos_before = (bodies[0].position, bodies[1].position);
        solve_extra_joints(&mut bodies, &joints, dt());
        // Static bodies should not move
        assert_eq!(bodies[0].position.y.hi, pos_before.0.y.hi);
        assert_eq!(bodies[1].position.y.hi, pos_before.1.y.hi);
    }

    // --- GearJoint ---

    #[test]
    fn test_gear_joint_creation() {
        let gj = GearJoint::new(0, 1, 0, 1, Fix128::from_int(2));
        assert_eq!(gj.body_a, 0);
        assert_eq!(gj.body_b, 1);
        assert_eq!(gj.joint_a, 0);
        assert_eq!(gj.joint_b, 1);
        assert_eq!(gj.ratio.hi, 2);
    }

    #[test]
    fn test_gear_joint_with_compliance() {
        let gj = GearJoint::new(0, 1, 0, 1, Fix128::ONE).with_compliance(Fix128::from_int(3));
        assert_eq!(gj.compliance.hi, 3);
    }

    #[test]
    fn test_gear_solve_no_rotation_no_correction() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        ];
        let joints = vec![ExtraJoint::Gear(GearJoint::new(0, 1, 0, 1, Fix128::ONE))];
        solve_extra_joints(&mut bodies, &joints, dt());
        // No rotation change (both identity) means no gear error.
        // Rotations should remain near identity (w close to 1).
        let w_a = bodies[0].rotation.w;
        let w_b = bodies[1].rotation.w;
        let tolerance = Fix128::from_ratio(1, 100);
        assert!(
            (w_a - Fix128::ONE).abs() < tolerance,
            "Body A rotation should remain near identity"
        );
        assert!(
            (w_b - Fix128::ONE).abs() < tolerance,
            "Body B rotation should remain near identity"
        );
    }

    #[test]
    fn test_gear_solve_with_rotation() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        ];
        // Rotate body A around Z axis
        bodies[0].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 4));
        // prev_rotation stays IDENTITY, so there's angular displacement on A
        let joints = vec![ExtraJoint::Gear(GearJoint::new(0, 1, 0, 1, Fix128::ONE))];
        solve_extra_joints(&mut bodies, &joints, dt());
        // Body B should have received some angular correction
        // (the gear couples them, so B's rotation should change)
        let b_rot = bodies[1].rotation;
        // At minimum, verify the solver ran without error and modified B's rotation
        assert!(
            b_rot.x != Fix128::ZERO || b_rot.y != Fix128::ZERO || b_rot.z != Fix128::ZERO,
            "Gear joint should apply angular correction to body B"
        );
    }

    // --- WeldJoint ---

    /// A body whose inverse inertia is `inv` and orientation `rotation`.
    fn weld_body(inv: Vec3Fix, rotation: QuatFix, dynamic: bool) -> RigidBody {
        let mut b = if dynamic {
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)
        } else {
            RigidBody::new_static(Vec3Fix::ZERO)
        };
        b.inv_inertia = inv;
        b.rotation = rotation;
        b
    }

    /// Angle of the rotation taking `from` to `to`, in radians.
    fn turn_angle(from: QuatFix, to: QuatFix) -> f64 {
        let d = shortest_arc(to.mul(from.conjugate()));
        2.0 * Vec3Fix::new(d.x, d.y, d.z).length().to_f64()
    }

    fn solve_one_weld(bodies: &mut [RigidBody]) {
        let joints = [ExtraJoint::Weld(WeldJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ))];
        solve_extra_joints(bodies, &joints, dt());
    }

    /// A compliant weld removes the fraction `w / (w + alpha)` of the error about
    /// every axis (`w = 2` for two unit inverse inertias, `alpha = compliance /
    /// dt^2 = 1` here), so `1/3` of it is left whichever axis it is about.
    #[test]
    fn compliant_weld_leaves_the_closed_form_fraction_about_every_axis() {
        let d = dt();
        let compliance = d * d;
        for axis in [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z] {
            let err = QuatFix::from_axis_angle(axis, Fix128::from_ratio(1, 1000));
            let one = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
            let mut bodies = [
                weld_body(one, QuatFix::IDENTITY, true),
                weld_body(one, err, true),
            ];
            let before = turn_angle(bodies[0].rotation, bodies[1].rotation);
            let joints = [ExtraJoint::Weld(
                WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
                    .with_compliance(compliance),
            )];
            solve_extra_joints(&mut bodies, &joints, d);
            let ratio = turn_angle(bodies[0].rotation, bodies[1].rotation) / before;
            assert!(
                (ratio - 1.0 / 3.0).abs() < 1e-5,
                "about {axis:?} a compliant weld left {ratio} of the error, expected 1/3"
            );
        }
    }

    /// Bodies held to planar rotation (inverse inertia only about z) have a
    /// singular effective mass; the weld must still close a z error between
    /// them, and with a static partner, in one solve.
    #[test]
    fn weld_closes_the_error_of_bodies_locked_to_planar_rotation() {
        let planar = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        let err = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 100));
        for a_dynamic in [true, false] {
            let mut bodies = [
                weld_body(planar, QuatFix::IDENTITY, a_dynamic),
                weld_body(planar, err, true),
            ];
            solve_one_weld(&mut bodies);
            let residual = turn_angle(bodies[0].rotation, bodies[1].rotation);
            // 6e-8 is the small-angle step of `rotate_by`, the same as for a
            // well-conditioned weld; an unsolved error stays at 1e-2.
            assert!(
                residual < 1e-6,
                "planar weld (a dynamic: {a_dynamic}) left {residual:e} rad of a 0.01 rad error"
            );
        }
    }

    /// `solves` solves of a weld whose second body is off by `err` (applied
    /// after its own rotation), returning the error left.
    fn weld_residual(
        inv_a: Vec3Fix,
        rot_a: QuatFix,
        inv_b: Vec3Fix,
        rot_b: QuatFix,
        a_dynamic: bool,
        err: QuatFix,
        solves: usize,
    ) -> f64 {
        let mut bodies = [
            weld_body(inv_a, rot_a, a_dynamic),
            weld_body(inv_b, rot_b.mul(err), true),
        ];
        // the weld holds b at a's orientation times this
        let joints = [ExtraJoint::Weld(WeldJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            rot_a.conjugate().mul(rot_b),
        ))];
        for _ in 0..solves {
            solve_extra_joints(&mut bodies, &joints, dt());
        }
        let target = bodies[0].rotation.mul(rot_a.conjugate().mul(rot_b));
        turn_angle(target, bodies[1].rotation)
    }

    fn tilted(axis: (i64, i64, i64), angle: (i64, i64)) -> QuatFix {
        QuatFix::from_axis_angle(
            Vec3Fix::from_int(axis.0, axis.1, axis.2).normalize(),
            Fix128::from_ratio(angle.0, angle.1),
        )
    }

    /// An error that lies wholly in a direction neither body can turn about
    /// asks for no correction: the bodies stay where they are (the eigenvalue
    /// of a locked direction is rounding and must not read as free).
    #[test]
    fn weld_does_not_turn_bodies_for_an_error_they_cannot_correct() {
        let planar = |w: Fix128| Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, w);
        let flat = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ZERO);
        let cases = [
            (planar(Fix128::ONE), false, Vec3Fix::UNIT_X),
            (planar(Fix128::from_f64(1e-9)), false, Vec3Fix::UNIT_X),
            (flat, true, Vec3Fix::UNIT_Z),
            (flat, false, Vec3Fix::UNIT_Z),
        ];
        for (inv, a_dynamic, axis) in cases {
            for i in 0..60i64 {
                let frame = tilted((1 + i % 7, -3 + i % 5, 2 + i % 3), (i * 37 % 314 + 1, 100));
                let err = QuatFix::from_axis_angle(axis, Fix128::from_ratio(1, 100));
                let mut bodies = [
                    weld_body(inv, frame, a_dynamic),
                    weld_body(inv, frame.mul(err), true),
                ];
                let before = [bodies[0].rotation, bodies[1].rotation];
                solve_one_weld(&mut bodies);
                for (k, b) in bodies.iter().enumerate() {
                    let turned = turn_angle(before[k], b.rotation);
                    assert!(
                        turned < 1e-9,
                        "inverse inertia {inv:?} (a dynamic: {a_dynamic}), frame {i}: body {k} \
                         turned {turned:e} rad for an error about the locked {axis:?}"
                    );
                }
            }
        }
    }

    /// Planar bodies whose free axis is tilted away from every world axis: the
    /// error about that axis closes in one solve, for two dynamic bodies and
    /// with a static partner (the rounding of a singular `K` must not read as
    /// a solvable direction).
    #[test]
    fn weld_closes_planar_errors_about_a_tilted_axis() {
        let planar = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        for frame in [
            tilted((1, 2, 3), (7, 10)),
            tilted((-3, 1, 2), (23, 10)),
            tilted((2, -5, 1), (11, 4)),
        ] {
            let err = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 100));
            for a_dynamic in [true, false] {
                let left = weld_residual(planar, frame, planar, frame, a_dynamic, err, 1);
                assert!(
                    left < 1e-6,
                    "tilted planar weld (a dynamic: {a_dynamic}) left {left:e} of 1e-2"
                );
            }
        }
    }

    /// The correction does not depend on how heavy the bodies are: equal
    /// isotropic inertia from 1e-9 to 1e9 closes 0.01 rad in one solve.
    #[test]
    fn weld_closes_the_error_at_every_inertia_scale() {
        let err = tilted((1, 1, 1), (1, 100));
        for w in [1e-9, 1e-6, 1.0, 1e6, 1e9] {
            let inv = Vec3Fix::new(
                Fix128::from_f64(w),
                Fix128::from_f64(w),
                Fix128::from_f64(w),
            );
            let left = weld_residual(inv, QuatFix::IDENTITY, inv, QuatFix::IDENTITY, true, err, 1);
            assert!(left < 1e-6, "inverse inertia {w:e} left {left:e} of 1e-2");
        }
    }

    /// Rods whose cheap long axes are 0.06 rad apart need turns several times
    /// the error; that is a well-conditioned weld, solved undamped, and the
    /// second-order rest of the first solve is gone after the second.
    #[test]
    fn weld_closes_the_error_between_rods_with_apart_long_axes() {
        // the undamped turn is 7.8 times the error: past 4 |theta| = 0.04 rad,
        // inside the 1/8 rad floor
        let rod = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::from_int(1000));
        let apart = tilted((1, 0, 0), (3, 50));
        let err = tilted((0, 1, 0), (1, 100));
        let one = weld_residual(rod, QuatFix::IDENTITY, rod, apart, true, err, 1);
        let two = weld_residual(rod, QuatFix::IDENTITY, rod, apart, true, err, 2);
        assert!(
            two < 1e-6,
            "rods left {one:e} after one solve and {two:e} after two"
        );

        // A 0.1 rad error: the undamped turns (about 3 times the error) stay
        // inside 4 |theta|, so three solves close it (measured 8.7e-7); a limit
        // of 2 |theta| would damp them and leave 1.9e-2.
        let big = tilted((0, 1, 1), (1, 10));
        let closer = tilted((1, 0, 0), (1, 10));
        let left = weld_residual(rod, QuatFix::IDENTITY, rod, closer, true, big, 3);
        assert!(
            left < 1e-5,
            "rods with a 0.1 rad error left {left:e} after three solves"
        );
    }

    /// Nearly locked axes a little apart make the linear solve turn both bodies
    /// by radians; the correction stays within `max(4 |theta|, 1/8 rad)` and
    /// still reduces the error.
    #[test]
    fn weld_turns_stay_bounded_when_the_effective_mass_is_nearly_singular() {
        let base = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_ratio(3, 10));
        // the last case needs mu doubled twice before the turns fit
        for (small, error) in [(1e-6, 100), (1e-12, 100), (1e-12, 5)] {
            let small = Fix128::from_f64(small);
            let err = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, error));
            let inv = Vec3Fix::new(Fix128::ONE, Fix128::ONE, small);
            let mut bodies = [
                weld_body(inv, base, true),
                weld_body(inv, err.mul(base), true),
            ];
            let before = [bodies[0].rotation, bodies[1].rotation];
            let error_before = turn_angle(before[0], before[1]);
            solve_one_weld(&mut bodies);
            for (i, b) in bodies.iter().enumerate() {
                let turned = turn_angle(before[i], b.rotation);
                assert!(
                    turned <= (4.0 * error_before).max(0.125) * (1.0 + 1e-9),
                    "inverse inertia {:e}: body {i} turned {turned:e} rad for a \
                     {error_before:e} rad error",
                    small.to_f64()
                );
            }
            let error_after = turn_angle(bodies[0].rotation, bodies[1].rotation);
            // Measured 1.00e-2 -> 9.5e-3: the bounded step still corrects,
            // where giving up would leave the error as it was.
            assert!(
                error_after < 0.99 * error_before,
                "inverse inertia {:e}: error {error_before:e} -> {error_after:e}, \
                 no correction within the bound",
                small.to_f64()
            );
        }
    }

    #[test]
    fn test_weld_joint_creation() {
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert_eq!(wj.body_a, 0);
        assert_eq!(wj.body_b, 1);
        assert!(wj.break_force.is_none());
        assert!(wj.break_torque.is_none());
    }

    #[test]
    fn test_weld_joint_holds_position() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        ];
        let joints = vec![ExtraJoint::Weld(WeldJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ))];
        for _ in 0..100 {
            solve_extra_joints(&mut bodies, &joints, dt());
        }
        let dist = (bodies[1].position - bodies[0].position).length();
        assert!(
            dist < Fix128::from_int(5),
            "Weld joint should pull bodies together"
        );
    }

    #[test]
    fn test_weld_joint_breaks_under_force() {
        let bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(20, 0, 0), Fix128::ONE),
        ];
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_force(Fix128::from_int(5));
        // Distance is 20, break force is 5 -> should be broken
        assert!(wj.is_broken(&bodies));
    }

    #[test]
    fn test_weld_joint_does_not_break_under_low_force() {
        let bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
        ];
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_force(Fix128::from_int(100));
        assert!(!wj.is_broken(&bodies));
    }

    #[test]
    fn test_weld_joint_break_torque() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        ];
        // Rotate body B significantly
        bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_int(1));
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_torque(Fix128::from_ratio(1, 100));
        assert!(wj.is_broken(&bodies), "Should break under high torque");
    }

    #[test]
    fn test_weld_joint_compute_force() {
        let bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE),
        ];
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let force = wj.compute_force(&bodies);
        // Distance = sqrt(9+16) = 5
        assert_eq!(force.hi, 5);
    }

    #[test]
    fn test_weld_joint_with_compliance() {
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_compliance(Fix128::from_int(10));
        assert_eq!(wj.compliance.hi, 10);
    }

    #[test]
    fn test_weld_broken_skips_solve() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(20, 0, 0), Fix128::ONE),
        ];
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_force(Fix128::from_int(5));
        let joints = vec![ExtraJoint::Weld(wj)];
        let pos_before = bodies[1].position;
        solve_extra_joints(&mut bodies, &joints, dt());
        // Should be broken, so body should not move
        assert_eq!(
            bodies[1].position.x.hi, pos_before.x.hi,
            "Broken weld should not correct position"
        );
    }

    // --- RackAndPinionJoint ---

    #[test]
    fn test_rack_and_pinion_creation() {
        let rp =
            RackAndPinionJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, Fix128::from_int(2));
        assert_eq!(rp.body_rack, 0);
        assert_eq!(rp.body_pinion, 1);
        assert_eq!(rp.ratio.hi, 2);
    }

    #[test]
    fn test_rack_and_pinion_with_compliance() {
        let rp = RackAndPinionJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, Fix128::ONE)
            .with_compliance(Fix128::from_int(7));
        assert_eq!(rp.compliance.hi, 7);
    }

    #[test]
    fn test_rack_and_pinion_no_displacement() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        ];
        let joints = vec![ExtraJoint::RackAndPinion(RackAndPinionJoint::new(
            0,
            1,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Z,
            Fix128::ONE,
        ))];
        // No displacement from prev_position => no error => no significant correction
        let pos_before = bodies[0].position;
        solve_extra_joints(&mut bodies, &joints, dt());
        let moved = (bodies[0].position.x - pos_before.x).abs();
        assert!(
            moved < Fix128::from_ratio(1, 10),
            "No displacement should mean minimal correction"
        );
    }

    #[test]
    fn test_rack_and_pinion_with_displacement() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE),
        ];
        // Simulate rack having moved along X from origin
        bodies[0].prev_position = Vec3Fix::ZERO;
        // Pinion has not rotated
        let joints = vec![ExtraJoint::RackAndPinion(RackAndPinionJoint::new(
            0,
            1,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Z,
            Fix128::ONE,
        ))];
        solve_extra_joints(&mut bodies, &joints, dt());
        // There should be some correction applied (rack moved without pinion rotating)
        // The pinion should receive angular correction or the rack should be pulled back
        // Just verify no panic and some change occurred
        let rack_moved = bodies[0].position.x.hi != 3;
        let pinion_rotated = bodies[1].rotation.z != Fix128::ZERO;
        assert!(
            rack_moved || pinion_rotated,
            "Rack-and-pinion should apply correction when constraint is violated"
        );
    }

    // --- MouseJoint ---

    #[test]
    fn test_mouse_joint_creation() {
        let mj = MouseJoint::new(
            0,
            Vec3Fix::from_int(10, 0, 0),
            Fix128::from_int(100),
            Fix128::from_int(50),
            Fix128::from_int(5),
        );
        assert_eq!(mj.body, 0);
        assert_eq!(mj.target_position.x.hi, 10);
        assert_eq!(mj.max_force.hi, 100);
        assert_eq!(mj.stiffness.hi, 50);
        assert_eq!(mj.damping.hi, 5);
    }

    #[test]
    fn test_mouse_joint_set_target() {
        let mut mj = MouseJoint::new(
            0,
            Vec3Fix::ZERO,
            Fix128::from_int(100),
            Fix128::from_int(50),
            Fix128::from_int(5),
        );
        mj.set_target(Vec3Fix::from_int(20, 30, 40));
        assert_eq!(mj.target_position.x.hi, 20);
        assert_eq!(mj.target_position.y.hi, 30);
        assert_eq!(mj.target_position.z.hi, 40);
    }

    #[test]
    fn test_mouse_joint_pulls_toward_target() {
        let mut bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];
        let joints = vec![ExtraJoint::Mouse(MouseJoint::new(
            0,
            Vec3Fix::from_int(10, 0, 0),
            Fix128::from_int(1000),
            Fix128::from_int(100),
            Fix128::from_int(1),
        ))];
        for _ in 0..50 {
            solve_extra_joints(&mut bodies, &joints, dt());
        }
        // Body should have moved toward target (10, 0, 0)
        assert!(
            bodies[0].position.x > Fix128::ZERO,
            "Mouse joint should pull body toward target"
        );
    }

    #[test]
    fn test_mouse_joint_max_force_clamp() {
        let mut bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];
        let joints = vec![ExtraJoint::Mouse(MouseJoint::new(
            0,
            Vec3Fix::from_int(1000, 0, 0), // Far away target
            Fix128::from_ratio(1, 100),    // Very small max force
            Fix128::from_int(1000),
            Fix128::ZERO,
        ))];
        solve_extra_joints(&mut bodies, &joints, dt());
        // With tiny max_force and large distance, movement should be small
        let moved = bodies[0].position.x;
        assert!(
            moved < Fix128::from_int(1),
            "Max force should clamp the applied correction"
        );
    }

    #[test]
    fn test_mouse_joint_static_body_no_move() {
        let mut bodies = vec![RigidBody::new_static(Vec3Fix::ZERO)];
        let joints = vec![ExtraJoint::Mouse(MouseJoint::new(
            0,
            Vec3Fix::from_int(10, 0, 0),
            Fix128::from_int(100),
            Fix128::from_int(50),
            Fix128::from_int(5),
        ))];
        solve_extra_joints(&mut bodies, &joints, dt());
        assert!(
            bodies[0].position.x.is_zero(),
            "Static body should not be moved by mouse joint"
        );
    }

    #[test]
    fn test_mouse_joint_at_target_no_correction() {
        let mut bodies = vec![RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE)];
        let joints = vec![ExtraJoint::Mouse(MouseJoint::new(
            0,
            Vec3Fix::from_int(5, 0, 0), // Already at target
            Fix128::from_int(100),
            Fix128::from_int(50),
            Fix128::from_int(5),
        ))];
        let pos_before = bodies[0].position;
        solve_extra_joints(&mut bodies, &joints, dt());
        assert_eq!(
            bodies[0].position.x.hi, pos_before.x.hi,
            "Body at target should not move"
        );
    }

    // --- Mixed joints ---

    #[test]
    fn test_solve_multiple_joint_types() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
            RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE),
        ];
        let joints = vec![
            ExtraJoint::Weld(WeldJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            )),
            ExtraJoint::Mouse(MouseJoint::new(
                2,
                Vec3Fix::from_int(0, 10, 0),
                Fix128::from_int(100),
                Fix128::from_int(50),
                Fix128::from_int(5),
            )),
        ];
        // Should not panic and should process both joints
        for _ in 0..10 {
            solve_extra_joints(&mut bodies, &joints, dt());
        }
        // Weld should pull body 1 closer
        let dist = (bodies[1].position - bodies[0].position).length();
        assert!(dist < Fix128::from_int(5), "Weld should constrain");
        // Mouse should push body 2 toward (0,10,0)
        assert!(
            bodies[2].position.y > Fix128::from_int(5),
            "Mouse should push body toward target"
        );
    }

    // --- Helper function tests ---

    #[test]
    fn test_extract_angle_identity() {
        let angle = extract_angle(QuatFix::IDENTITY);
        assert!(
            angle.is_zero(),
            "Identity quaternion should have zero angle"
        );
    }

    #[test]
    fn test_extract_angle_nonzero() {
        let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_int(1));
        let angle = extract_angle(q);
        // Should be a positive, non-zero angle
        assert!(
            angle > Fix128::ZERO,
            "Angle from rotated quaternion should be positive"
        );
        // 2*sin(0.5) ~ 0.958, close to the input angle of 1 radian
        assert!(
            angle < Fix128::from_int(2),
            "Extracted angle should be in the right ballpark"
        );
    }

    #[test]
    fn test_extract_angle_around_axis() {
        let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 2));
        let angle = extract_angle_around_axis(q, Vec3Fix::UNIT_Y);
        // Should be positive and non-zero
        assert!(
            angle > Fix128::ZERO,
            "Angle from rotated quaternion should be positive"
        );
        // Should be less than pi (reasonable bound)
        assert!(angle < Fix128::PI, "Extracted angle should be less than pi");
    }
}
