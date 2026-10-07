//! XPBD (Extended Position Based Dynamics) Solver - Batched Edition
//!
//! Unified solver for rigid bodies, cloth, and ropes.
//!
//! # Key Features
//!
//! - Unconditionally stable (no stiffness-induced explosions)
//! - Supports substeps for stiff constraints
//! - Deterministic with fixed iteration counts
//! - **Constraint Batching**: Graph-colored constraint groups for parallel solving
//! - **Optional Parallelism**: Enable `parallel` feature for Rayon-based solving
//!
//! # Batching Strategy
//!
//! Constraints are grouped by "color" where constraints of the same color
//! share no bodies (independent). This allows:
//! - Sequential processing within each color (data dependency)
//! - Parallel processing across colors (no conflicts)

use crate::bvh::{BroadphaseHybrid, BvhPrimitive, LinearBvh};
use crate::collider::{Contact, AABB};
use crate::event::EventCollector;
use crate::filter::CollisionFilter;
use crate::force::{apply_force_fields, ForceFieldInstance};
use crate::joint::{solve_joints, Joint};
use crate::math::{select_vec3, Fix128, QuatFix, Vec3Fix};
use crate::sdf_collider::SdfCollider;
use crate::sleeping::{IslandManager, SleepConfig, SleepData, SleepState};

/// Minimum effective inverse-mass sum below which constraint solving is skipped.
/// Prevents division explosion when two near-static bodies are in contact.
/// Value: 2^-40 ≈ 9.1e-13 in Fix128 (`lo = 1 << 24`). Before 1.2.0 the raw
/// value was `1 << 40` = 2^-24 ≈ 6e-8 while the doc said 2^-40, so a dynamic
/// body heavier than 2^24 kg against a static one (or two bodies above
/// 2^25 kg) had its contacts, distance constraints and restitution
/// silently skipped.
const W_SUM_EPSILON: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000000001000000,
};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Rayon parallel iterator support
#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod participants;
mod world_ccd;
mod world_snapshot;
pub use participants::DeclareFieldError;
pub use world_ccd::WorldCcdConfig;
pub use world_snapshot::WorldSnapshotError;

// ============================================================================
// Body Type
// ============================================================================

/// Type of rigid body
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BodyType {
    /// Moved by physics (gravity, constraints, impulses)
    Dynamic = 0,
    /// Never moves
    Static = 1,
    /// Moved by user code, pushes dynamic bodies but is not affected by them
    Kinematic = 2,
}

// ============================================================================
// Rigid Body
// ============================================================================

/// Default [`RigidBody::friction`] of a body built by `new`, `new_dynamic`,
/// `new_static` and `new_kinematic`: 0.5, the `dynamic_friction` of the default
/// material, so a body that keeps the defaults contacts like one with no body value.
const DEFAULT_BODY_FRICTION: Fix128 = Fix128::from_raw(0, 9_223_372_036_854_775_808);

/// Default [`RigidBody::restitution`] of a body built by `new`, `new_dynamic`,
/// `new_static` and `new_kinematic`: 0.3, the `restitution` of the default material.
const DEFAULT_BODY_RESTITUTION: Fix128 = Fix128::from_raw(0, 5_534_023_222_112_865_484);

/// Rigid body state
///
/// Field layout is optimized for cache performance (Gap 1.1):
/// - HOT fields (accessed every solver iteration) are placed first.
/// - COLD fields (accessed occasionally) follow after.
///
/// `#[repr(C, align(64))]` ensures the struct starts on a cache-line boundary
/// so the hot fields are always in the first 64-byte cache line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct RigidBody {
    // --- HOT fields (accessed every iteration) ---
    /// Position (center of mass)
    pub position: Vec3Fix,
    /// Linear velocity
    pub velocity: Vec3Fix,
    /// Inverse mass (0 = static/infinite mass)
    pub inv_mass: Fix128,
    /// Inverse inertia tensor (diagonal, in local space)
    pub inv_inertia: Vec3Fix,
    /// Previous position (for XPBD)
    pub prev_position: Vec3Fix,

    // --- COLD fields (accessed occasionally) ---
    /// Orientation
    pub rotation: QuatFix,
    /// Angular velocity
    pub angular_velocity: Vec3Fix,
    /// Previous rotation (for XPBD)
    pub prev_rotation: QuatFix,
    /// Coefficient of restitution (bounciness), default 0.3
    ///
    /// Read by contacts only while the body has the default material
    /// (see [`PhysicsWorld::combined_material`]).
    pub restitution: Fix128,
    /// Friction coefficient, default 0.5
    ///
    /// Read by contacts only while the body has the default material
    /// (see [`PhysicsWorld::combined_material`]).
    pub friction: Fix128,
    /// Gravity scale multiplier (1.0 = normal, 0.0 = no gravity, 2.0 = double)
    pub gravity_scale: Fix128,
    /// Per-body linear damping factor (1.0 = no damping, 0.0 = full damping)
    /// Multiplied with the global damping, applied once per frame (`step()`).
    pub linear_damping: Fix128,
    /// Per-body angular damping factor (1.0 = no damping, 0.0 = full damping)
    /// Multiplied with the global damping, applied once per frame (`step()`).
    pub angular_damping: Fix128,
    /// Whether this body is a sensor/trigger (detects overlap but no physics response)
    pub is_sensor: bool,
    /// Body type (Dynamic, Static, Kinematic)
    pub body_type: BodyType,
    /// Kinematic target position and rotation
    pub kinematic_target: Option<(Vec3Fix, QuatFix)>,
}

/// `a - b`, or `None` when the difference leaves the range of [`Fix128`];
/// when it is `Some` it is bit for bit `a - b`.
fn checked_fix_sub(a: Fix128, b: Fix128) -> Option<Fix128> {
    let s = fix_wide(a).checked_sub(fix_wide(b))?;
    Some(fix_narrow(s))
}

/// The Q64.64 bits of `f` as one `i128`.
fn fix_wide(f: Fix128) -> i128 {
    (i128::from(f.hi) << 64) | i128::from(f.lo)
}

/// The inverse of [`fix_wide`].
fn fix_narrow(s: i128) -> Fix128 {
    Fix128::from_raw((s >> 64) as i64, s as u64)
}

/// [`QuatFix::mul`] with every product, sum and difference checked, in the
/// same order.
fn checked_quat_mul(a: QuatFix, b: QuatFix) -> Option<QuatFix> {
    let m = |x: Fix128, y: Fix128| x.checked_mul(y);
    let x = checked_fix_sub(
        Fix128::checked_add(
            Fix128::checked_add(m(a.w, b.x)?, m(a.x, b.w)?)?,
            m(a.y, b.z)?,
        )?,
        m(a.z, b.y)?,
    )?;
    let y = Fix128::checked_add(
        Fix128::checked_add(checked_fix_sub(m(a.w, b.y)?, m(a.x, b.z)?)?, m(a.y, b.w)?)?,
        m(a.z, b.x)?,
    )?;
    let z = Fix128::checked_add(
        checked_fix_sub(
            Fix128::checked_add(m(a.w, b.z)?, m(a.x, b.y)?)?,
            m(a.y, b.x)?,
        )?,
        m(a.z, b.w)?,
    )?;
    let w = checked_fix_sub(
        checked_fix_sub(checked_fix_sub(m(a.w, b.w)?, m(a.x, b.x)?)?, m(a.y, b.y)?)?,
        m(a.z, b.z)?,
    )?;
    Some(QuatFix::new(x, y, z, w))
}

/// Whether a body-frame diagonal inverse inertia is isotropic (`x = y = z`).
///
/// For such a body `R · diag(c, c, c) · Rᵀ = c · E` for every rotation `R`,
/// so a world-frame inverse inertia applies as the plain product with `c`.
#[inline]
#[must_use]
pub(crate) fn inv_inertia_is_isotropic(inv_inertia: Vec3Fix) -> bool {
    inv_inertia.x == inv_inertia.y && inv_inertia.y == inv_inertia.z
}

/// World-frame `I⁻¹ v` for the body-frame diagonal `inv_inertia` and the
/// body orientation `rotation`: `R · diag(inv_inertia) · R⁻¹ v`.
///
/// # Claims
/// - an isotropic `inv_inertia = (c, c, c)` gives the component-wise product
///   `(v.x·c, v.y·c, v.z·c)` without rotating, so the result does not depend
///   on `rotation` (the rotate, scale, rotate-back evaluation rounds
///   differently for every non-identity rotation)
/// - any other `inv_inertia` rotates `v` into the body frame by `R⁻¹`,
///   scales it component-wise and rotates it back by `R`
#[inline]
#[must_use]
pub(crate) fn inv_inertia_world_apply(
    rotation: QuatFix,
    inv_inertia: Vec3Fix,
    v: Vec3Fix,
) -> Vec3Fix {
    if inv_inertia_is_isotropic(inv_inertia) {
        let c = inv_inertia.x;
        return Vec3Fix::new(v.x * c, v.y * c, v.z * c);
    }
    let local = rotation.conjugate().rotate_vec(v);
    let scaled = Vec3Fix::new(
        local.x * inv_inertia.x,
        local.y * inv_inertia.y,
        local.z * inv_inertia.z,
    );
    rotation.rotate_vec(scaled)
}

/// [`QuatFix::rotate_vec`] (`q v q*`) with every operation checked.
fn checked_rotate_vec(q: QuatFix, v: Vec3Fix) -> Option<Vec3Fix> {
    let qv = QuatFix::new(v.x, v.y, v.z, Fix128::ZERO);
    let r = checked_quat_mul(checked_quat_mul(q, qv)?, q.conjugate())?;
    Some(Vec3Fix::new(r.x, r.y, r.z))
}

impl RigidBody {
    /// Create a new dynamic rigid body at the given position with the given mass.
    ///
    /// The inverse mass is computed automatically. A unit-sphere inertia tensor
    /// is used as the default. Use `new_dynamic` as a more descriptive alias.
    ///
    /// # Examples
    ///
    /// ```
    /// use alice_physics::{Fix128, RigidBody, Vec3Fix};
    ///
    /// let body = RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
    /// // inv_mass should be 1 / 1 = 1
    /// assert_eq!(body.inv_mass.hi, 1);
    /// // Position should match
    /// assert_eq!(body.position.y.hi, 10);
    /// // Dynamic body is not static
    /// assert!(!body.is_static());
    /// ```
    #[must_use]
    pub fn new(position: Vec3Fix, mass: Fix128) -> Self {
        let inv_mass = if mass.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / mass
        };

        // Default inertia (unit sphere)
        let inertia = mass * Fix128::from_ratio(2, 5);
        let inv_inertia = if inertia.is_zero() {
            Vec3Fix::ZERO
        } else {
            let inv_i = Fix128::ONE / inertia;
            Vec3Fix::new(inv_i, inv_i, inv_i)
        };

        Self {
            position,
            velocity: Vec3Fix::ZERO,
            inv_mass,
            inv_inertia,
            prev_position: position,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: DEFAULT_BODY_RESTITUTION,
            friction: DEFAULT_BODY_FRICTION,
            gravity_scale: Fix128::ONE,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        }
    }

    /// Create new dynamic rigid body (alias for new)
    #[inline]
    #[must_use]
    pub fn new_dynamic(position: Vec3Fix, mass: Fix128) -> Self {
        Self::new(position, mass)
    }

    /// Create static (immovable) rigid body
    #[must_use]
    pub const fn new_static(position: Vec3Fix) -> Self {
        Self {
            position,
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: DEFAULT_BODY_RESTITUTION,
            friction: DEFAULT_BODY_FRICTION,
            gravity_scale: Fix128::ZERO,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Static,
            kinematic_target: None,
        }
    }

    /// Create a sensor (trigger) body - detects overlap but no physics response
    #[must_use]
    pub const fn new_sensor(position: Vec3Fix) -> Self {
        Self {
            position,
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ZERO,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: true,
            body_type: BodyType::Static,
            kinematic_target: None,
        }
    }

    /// Create a kinematic body (moved by user code, not physics)
    ///
    /// Kinematic bodies have infinite mass and are unaffected by forces,
    /// but can push dynamic bodies. Set target via `set_kinematic_target`.
    #[must_use]
    pub const fn new_kinematic(position: Vec3Fix) -> Self {
        Self {
            position,
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: DEFAULT_BODY_RESTITUTION,
            friction: DEFAULT_BODY_FRICTION,
            gravity_scale: Fix128::ZERO,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Kinematic,
            kinematic_target: None,
        }
    }

    /// Check if body is static
    #[inline]
    #[must_use]
    pub const fn is_static(&self) -> bool {
        self.inv_mass.is_zero()
    }

    /// Check if body is kinematic
    #[inline]
    #[must_use]
    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType::Kinematic
    }

    /// Check if body is dynamic
    #[inline]
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    /// Set kinematic target position and rotation
    ///
    /// The body will be moved to this target during the next simulation step.
    /// Velocity is automatically computed from the position change.
    ///
    /// `rotation` is stored as a unit quaternion (see [`Self::set_rotation`]).
    pub fn set_kinematic_target(&mut self, position: Vec3Fix, rotation: QuatFix) {
        self.kinematic_target = Some((position, rotation.unit_rotation()));
    }

    /// Apply impulse at center of mass
    pub fn apply_impulse(&mut self, impulse: Vec3Fix) {
        if !self.is_static() {
            self.velocity = self.velocity + impulse * self.inv_mass;
        }
    }

    /// World-frame `I⁻¹ τ` for a body whose `inv_inertia` is diagonal in the body frame.
    ///
    /// # Claims
    /// - `τ` is a world-frame vector; the result is a world-frame vector
    /// - computed as `R · diag(inv_inertia) · R⁻¹ τ` with `R = self.rotation`
    /// - for an identity rotation it equals the component-wise product with `inv_inertia`
    /// - a zero `inv_inertia` (or a zero `τ`) gives the zero vector
    /// - an isotropic `inv_inertia = (c, c, c)` gives the component-wise
    ///   product `c · τ` for every rotation (see [`inv_inertia_world_apply`])
    #[inline]
    pub(crate) fn world_inv_inertia_apply(&self, torque: Vec3Fix) -> Vec3Fix {
        inv_inertia_world_apply(self.rotation, self.inv_inertia, torque)
    }

    /// [`Self::world_inv_inertia_apply`] with every product, sum and
    /// difference checked: `None` when one of them leaves the range of
    /// [`Fix128`].
    ///
    /// # Claims
    /// - when it is `Some`, the value is bit for bit the one of
    ///   [`Self::world_inv_inertia_apply`] (the same operations in the same
    ///   order; a wrapping sum whose every partial sum is in range is exact)
    /// - a product goes through [`Fix128::checked_mul`]; an intermediate
    ///   partial sum out of range is `None` even if the final value would be
    ///   in range (the rotation of a unit quaternion keeps the partial sums
    ///   within a small factor of `|τ|`, so this only matters near the edge
    ///   of the range)
    #[must_use]
    pub(crate) fn checked_world_inv_inertia_apply(&self, torque: Vec3Fix) -> Option<Vec3Fix> {
        if inv_inertia_is_isotropic(self.inv_inertia) {
            let c = self.inv_inertia.x;
            return Some(Vec3Fix::new(
                torque.x.checked_mul(c)?,
                torque.y.checked_mul(c)?,
                torque.z.checked_mul(c)?,
            ));
        }
        let q = self.rotation;
        let local = checked_rotate_vec(q.conjugate(), torque)?;
        let scaled = Vec3Fix::new(
            local.x.checked_mul(self.inv_inertia.x)?,
            local.y.checked_mul(self.inv_inertia.y)?,
            local.z.checked_mul(self.inv_inertia.z)?,
        );
        checked_rotate_vec(q, scaled)
    }

    /// Apply impulse at world-space point
    ///
    /// # Claims
    /// - the linear part is `impulse * inv_mass`
    /// - the angular part is `I_world⁻¹ ((point - position) × impulse)`, with the body-frame
    ///   diagonal `inv_inertia` rotated by `rotation` into the world frame
    /// - a static body is unchanged
    pub fn apply_impulse_at(&mut self, impulse: Vec3Fix, point: Vec3Fix) {
        if !self.is_static() {
            self.velocity = self.velocity + impulse * self.inv_mass;

            let r = point - self.position;
            let torque = r.cross(impulse);
            self.angular_velocity = self.angular_velocity + self.world_inv_inertia_apply(torque);
        }
    }

    /// Apply a continuous force (accumulated, applied during next step)
    ///
    /// Force is converted to velocity change: v += (F * `inv_mass`) * dt
    /// For one-shot velocity changes, use `apply_impulse` instead.
    #[inline]
    pub fn add_force(&mut self, force: Vec3Fix, dt: Fix128) {
        if !self.is_static() {
            self.velocity = self.velocity + force * self.inv_mass * dt;
        }
    }

    /// Apply a continuous torque
    ///
    /// # Claims
    /// - `torque` is a world-frame vector; `angular_velocity += I_world⁻¹ torque * dt`
    /// - `I_world⁻¹ = R diag(inv_inertia) R⁻¹` with `R = rotation`
    /// - a static body is unchanged
    #[inline]
    pub fn add_torque(&mut self, torque: Vec3Fix, dt: Fix128) {
        if !self.is_static() {
            self.angular_velocity =
                self.angular_velocity + self.world_inv_inertia_apply(torque) * dt;
        }
    }

    /// Set the linear velocity directly
    #[inline]
    pub fn set_velocity(&mut self, velocity: Vec3Fix) {
        self.velocity = velocity;
    }

    /// Set the angular velocity directly
    #[inline]
    pub fn set_angular_velocity(&mut self, angular_velocity: Vec3Fix) {
        self.angular_velocity = angular_velocity;
    }

    /// Set position directly (teleport)
    #[inline]
    pub fn set_position(&mut self, position: Vec3Fix) {
        self.position = position;
        self.prev_position = position;
    }

    /// Set rotation directly
    ///
    /// The stored rotation is `rotation` as a unit quaternion: one whose
    /// squared length is within `2^-32` of one is kept bit for bit, any other
    /// is normalized (the zero quaternion becomes the identity). Applying a
    /// quaternion `q` scales what it rotates by `|q|^2`.
    #[inline]
    pub fn set_rotation(&mut self, rotation: QuatFix) {
        let rotation = rotation.unit_rotation();
        self.rotation = rotation;
        self.prev_rotation = rotation;
    }

    /// The stored rotations (`rotation`, `prev_rotation` and the rotation of
    /// `kinematic_target`) as unit quaternions, see
    /// [`QuatFix::unit_rotation`]: a rotation already of unit length (within
    /// `2^-32` in squared length) is left bit for bit.
    #[inline]
    pub(crate) fn make_rotations_unit(&mut self) {
        self.rotation = self.rotation.unit_rotation();
        self.prev_rotation = self.prev_rotation.unit_rotation();
        if let Some((position, rotation)) = self.kinematic_target {
            self.kinematic_target = Some((position, rotation.unit_rotation()));
        }
    }

    /// Get mass (inverse of `inv_mass`, returns infinity for static bodies)
    #[inline]
    #[must_use]
    pub fn mass(&self) -> Fix128 {
        if self.inv_mass.is_zero() {
            Fix128::ZERO // Represents infinite mass
        } else {
            Fix128::ONE / self.inv_mass
        }
    }

    /// Get linear speed (magnitude of velocity)
    #[inline]
    #[must_use]
    pub fn speed(&self) -> Fix128 {
        self.velocity.length()
    }

    /// Set per-body linear damping (1.0 = no extra damping)
    #[inline]
    #[must_use]
    pub const fn with_linear_damping(mut self, damping: Fix128) -> Self {
        self.linear_damping = damping;
        self
    }

    /// Set per-body angular damping (1.0 = no extra damping)
    #[inline]
    #[must_use]
    pub const fn with_angular_damping(mut self, damping: Fix128) -> Self {
        self.angular_damping = damping;
        self
    }

    /// Builder: set restitution (bounciness)
    #[inline]
    #[must_use]
    pub const fn with_restitution(mut self, restitution: Fix128) -> Self {
        self.restitution = restitution;
        self
    }

    /// Builder: set friction coefficient
    #[inline]
    #[must_use]
    pub const fn with_friction(mut self, friction: Fix128) -> Self {
        self.friction = friction;
        self
    }

    /// Builder: set gravity scale
    #[inline]
    #[must_use]
    pub const fn with_gravity_scale(mut self, scale: Fix128) -> Self {
        self.gravity_scale = scale;
        self
    }

    /// Builder: set as sensor (trigger)
    #[inline]
    #[must_use]
    pub const fn with_sensor(mut self, is_sensor: bool) -> Self {
        self.is_sensor = is_sensor;
        self
    }

    /// Builder: set initial velocity
    #[inline]
    #[must_use]
    pub const fn with_velocity(mut self, velocity: Vec3Fix) -> Self {
        self.velocity = velocity;
        self
    }

    /// Builder: set initial rotation
    #[inline]
    #[must_use]
    pub const fn with_rotation(mut self, rotation: QuatFix) -> Self {
        self.rotation = rotation;
        self.prev_rotation = rotation;
        self
    }
}

impl Default for RigidBody {
    /// Default creates a 1kg dynamic body at the origin.
    fn default() -> Self {
        Self::new(Vec3Fix::ZERO, Fix128::ONE)
    }
}

// ============================================================================
// Constraints
// ============================================================================

/// Distance constraint between two bodies
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistanceConstraint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Anchor point in body A's local space
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local space
    pub local_anchor_b: Vec3Fix,
    /// Target distance between anchors
    pub target_distance: Fix128,
    /// Inverse stiffness (0 = infinitely stiff)
    pub compliance: Fix128,
    /// XPBD Lagrange multiplier accumulated over the current substep.
    ///
    /// Zeroed at the start of every substep and incremented by `dlambda` in each
    /// solve iteration (Macklin 2016), so the compliance term `alpha~ * lambda`
    /// sees the total force applied so far and the effective stiffness is
    /// independent of `iterations`. Before 1.2.0 it held only the last
    /// iteration's increment and was carried across substeps.
    pub cached_lambda: Fix128,
}

impl DistanceConstraint {
    /// Create a new distance constraint
    #[must_use]
    pub const fn new(
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Fix,
        anchor_b: Vec3Fix,
        distance: Fix128,
    ) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: anchor_a,
            local_anchor_b: anchor_b,
            target_distance: distance,
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        }
    }

    /// Set compliance (inverse stiffness)
    #[must_use]
    pub const fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

/// Contact constraint (from collision detection)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactConstraint {
    /// Index of the first body
    pub body_a: usize,
    /// Index of the second body
    pub body_b: usize,
    /// Contact information from collision detection
    pub contact: Contact,
    /// Friction coefficient
    pub friction: Fix128,
    /// Restitution (bounciness) coefficient
    pub restitution: Fix128,
    /// Contact multiplier accumulated over the substep the contact lives in
    /// (contacts are re-detected every substep since 1.2.0): the separation
    /// along `contact.normal` already applied by earlier solver iterations.
    pub cached_lambda: Fix128,
}

impl ContactConstraint {
    /// Create a new contact constraint with default friction and restitution
    #[must_use]
    pub fn new(body_a: usize, body_b: usize, contact: Contact) -> Self {
        Self {
            body_a,
            body_b,
            contact,
            friction: Fix128::from_ratio(3, 10), // 0.3 friction
            restitution: Fix128::from_ratio(2, 10), // 0.2 restitution
            cached_lambda: Fix128::ZERO,
        }
    }
}

// ============================================================================
// XPBD Solver
// ============================================================================

/// Selects which rigid-body integrator [`PhysicsWorld::step`] dispatches to.
///
/// Both variants read and write the same [`RigidBody`] fields, so a world
/// can be migrated between backends between steps without changing any
/// other configuration. They are **not** numerically interchangeable —
/// XPBD is position-based (Macklin/Müller) and TGS is a sub-stepping
/// impulse-based projected Gauss-Seidel integrator (Müller et al. 2020,
/// see the crate-internal `solver_tgs` module) — so the same scene produces different
/// (both physically valid) trajectories under each backend. Pick one and
/// stay on it for a given simulation; do not expect bit-identical replay
/// across a backend switch.
///
/// # Notes on the `Tgs` path
///
/// [`DistanceConstraint`]s are enforced as bilateral velocity-level impulse
/// joints (`JointOriented` in `solver_tgs_hooks_6dof_oriented`), solved
/// every TGS inner sub-step alongside contacts, plus a Baumgarte position
/// correction for residual drift — a genuinely different algorithm from
/// XPBD's position-based compliance solve, so the two backends converge to
/// slightly different equilibria for the same scene (not a bug; see
/// `tests/analytic_tgs_wiring.rs`'s joint tests for both backends' measured
/// bands).
///
/// Pre-solve hooks and contact modifiers run once over the
/// tick's contacts before the solve (`tests/analytic_tgs_backend_coverage.rs`).
///
/// The world's [`Joint`]s are solved inside the TGS substep loop: after each
/// substep's impulse solve, the same position-level joint projection the XPBD
/// substep runs is applied and its correction is added to the bodies' linear
/// and angular velocities (`tests/analytic_tgs_joints_in_substep.rs`).
///
/// * Requires the `std` feature (`solver_tgs` is `std`-gated, same as the
///   rest of the TGS family). Selecting `Tgs` in a build without `std`
///   is **not** silently ignored at the type level — the variant still
///   exists — but [`PhysicsWorld::step`] falls back to running `Xpbd`
///   for that build, since there is no TGS implementation to dispatch to.
///   This is a documented, defined fallback (a different already-correct
///   solver runs), not state corruption.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SolverBackend {
    /// Extended Position-Based Dynamics — the solver this crate has shipped
    /// since 0.1.0. Default.
    #[default]
    Xpbd,
    /// Sub-stepping impulse-based temporal Gauss-Seidel. See the gaps listed
    /// on [`SolverBackend`] itself before using this in production.
    Tgs,
}

/// The broad-phase [`PhysicsWorld`] uses to find the pairs of bodies whose
/// colliders may touch.
///
/// A body's box is the closed-form world box of its collider when it carries a
/// shape or a compound, and the cube of its collision sphere otherwise. Every
/// kind hands its candidates (pairs whose boxes may overlap) to the
/// same exact narrow-phase in ascending pair order, and a pair whose boxes do not
/// meet has no contact, so a world steps to **bit-identical** results whichever
/// it uses; they differ in cost. Set with [`PhysicsWorld::set_broadphase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Broadphase {
    /// A linear BVH over Morton codes, rebuilt from scratch every step. Default.
    #[default]
    Bvh,
    /// A persistent dynamic AABB tree ([`crate::dynamic_bvh::DynamicAabbTree`]):
    /// each body keeps a fattened proxy that is only re-inserted when the body
    /// leaves it, so a world where most bodies move little pays for the few that do.
    DynamicTree,
    /// Static bodies in a BVH rebuilt only when the static set changes, small
    /// dynamic bodies in a sparse hash grid whose cell size follows the median
    /// body, and large dynamic bodies in a per-step BVH; reports exactly the
    /// pairs whose boxes overlap. Fastest when many bodies are static or bodies
    /// are of similar size. The sleep skip ([`PhysicsWorld::set_sleep_skip`])
    /// parks bodies only with [`Broadphase::Bvh`]; with this kind every body is
    /// staged every step.
    Hybrid,
}

/// What the [`Broadphase::DynamicTree`] broad-phase currently holds, returned by
/// [`PhysicsWorld::broadphase_stats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BroadphaseStats {
    /// Bodies with a proxy in the tree.
    pub proxies: usize,
    /// Height of the tree (0 when empty).
    pub height: i32,
}

/// How much work the last [`PhysicsWorld::step`] did, stage by stage.
///
/// Every counter is a number of items a stage visited, summed over the substeps
/// of that one call (reset when `step` starts). They count work, not time, so a
/// test can assert on them without measuring a clock. `step_parallel`,
/// `step_with_bridge`, the TGS backend and the public substep API add to the
/// same counters but do not reset them.
///
/// With the sleep skip on ([`PhysicsWorld::set_sleep_skip`], the default) a
/// sleeping body that is at rest and not attached to a joint or distance
/// constraint is *parked*: every stage below except `sleep_scanned` and
/// `parked_sleep_updates` leaves it out, and the broad-phase finds its contacts
/// through a persistent tree instead of rebuilding it. The counters then depend
/// on the awake bodies only, except those two, which stay one per sleeping body
/// per step (a check that the parked state was not edited between steps, and
/// the `idle_frames` count the snapshot carries).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StageWork {
    /// Bodies examined at the start of the step to decide which are parked.
    pub sleep_scanned: u64,
    /// Bodies parked for this step (not counting ones woken during it).
    pub parked: u64,
    /// Parked bodies woken by a contact during the step.
    pub unparked: u64,
    /// Bodies the force fields were applied to.
    pub force_field_bodies: u64,
    /// Bodies visited by position integration.
    pub integrated: u64,
    /// Bodies put into the per-substep broad-phase BVH.
    pub broadphase_primitives: u64,
    /// Candidate pairs the broad-phase handed to the narrow-phase.
    pub broadphase_pairs: u64,
    /// Pairs of bodies whose exact boxes the [`Broadphase::Hybrid`] broad-phase
    /// compared to find `broadphase_pairs` (0 with the other broad-phases).
    pub broadphase_box_tests: u64,
    /// Bodies tested against the static and SDF colliders.
    pub resolution_bodies: u64,
    /// Bodies whose velocity was derived from their position change.
    pub velocity_bodies: u64,
    /// Bodies visited by the frame damping.
    pub damping_bodies: u64,
    /// Bodies whose sleep state was evaluated from their velocity.
    pub sleep_evaluated: u64,
    /// Parked bodies whose sleep bookkeeping was advanced without evaluation.
    pub parked_sleep_updates: u64,
    /// Proxies inserted into the parked-body tree.
    pub tree_inserts: u64,
    /// Proxies removed from the parked-body tree.
    pub tree_removes: u64,
}

/// What a parked body looked like when it was last found at rest, so the next
/// step can tell with a few compares that nothing has edited it since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParkCache {
    position: Vec3Fix,
    rotation: QuatFix,
    /// The angular velocity the velocity derivation gives a body whose rotation
    /// did not change (not always exactly zero: `x·(−y)` and `−(x·y)` can differ
    /// by one ulp in fixed point).
    angular_velocity: Vec3Fix,
    radius: Option<Fix128>,
    /// Recorded for a `BodyType::Static` body (only position and radius matter,
    /// and the rotation when the body carries a collider: it turns the box).
    static_kind: bool,
}

/// Everything a parked body's "a step changes nothing but `idle_frames`" verdict
/// depends on besides its own state. Any change drops every [`ParkCache`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct ParkKey {
    substep_dt: Fix128,
    sleep: SleepConfig,
    generation: u64,
    sdf_radius: Fix128,
    /// Pose, scale, attachment and field address of each SDF collider (the
    /// `sdf_colliders` field is public, so it is compared every step).
    sdf: Vec<(Vec3Fix, QuatFix, Fix128, usize, usize)>,
}

/// Book-keeping of the sleep skip (see [`StageWork`]).
#[derive(Default)]
struct ParkState {
    /// Parking is in effect for the step being run.
    active: bool,
    /// Per body: parked for the rest of the current step.
    parked: Vec<bool>,
    /// The bodies parked at the start of the step, ascending.
    parked_list: Vec<usize>,
    /// The bodies that are not parked, ascending; the stages iterate these.
    awake: Vec<usize>,
    cache: Vec<Option<ParkCache>>,
    proxies: Vec<Option<u32>>,
    /// Number of `Some` in `proxies` (`DynamicAabbTree::proxy_count` walks
    /// every node, which would put the parked bodies back into each substep).
    proxy_live: usize,
    tree: crate::dynamic_bvh::DynamicAabbTree,
    key: Option<ParkKey>,
    /// Scratch: body referenced by a joint or a distance constraint.
    constrained: Vec<bool>,
    /// The substep being run is the first of the step.
    first_substep: bool,
    /// The full `dt` of the step (force fields act on it).
    frame_dt: Fix128,
}

impl ParkState {
    /// [`Self::clear`], counting the dropped proxies as tree removals.
    fn clear_counted(&mut self, stats: &mut StageWork) {
        stats.tree_removes += self.proxy_live as u64;
        self.clear();
    }

    /// Drop the parked-body tree and every cache.
    fn clear(&mut self) {
        self.tree = crate::dynamic_bvh::DynamicAabbTree::new();
        self.proxies.clear();
        self.proxy_live = 0;
        self.cache.clear();
        self.parked.clear();
        self.parked_list.clear();
        self.awake.clear();
        self.key = None;
        self.active = false;
    }

    /// Remove body `i`'s proxy from the tree, if it has one.
    fn remove_proxy(&mut self, i: usize, stats: &mut StageWork) {
        if let Some(proxy) = self.proxies.get_mut(i).and_then(Option::take) {
            self.tree.remove(proxy);
            self.proxy_live -= 1;
            stats.tree_removes += 1;
        }
    }

    /// Body `i` was woken during the step: from now on every stage sees it.
    fn unpark(&mut self, i: usize, stats: &mut StageWork) {
        if !self.active || !self.parked.get(i).copied().unwrap_or(false) {
            return;
        }
        self.parked[i] = false;
        self.remove_proxy(i, stats);
        if let Err(pos) = self.awake.binary_search(&i) {
            self.awake.insert(pos, i);
        }
        stats.unparked += 1;
    }

    /// Whether body `i` is parked in the current step.
    #[inline]
    fn is_parked(&self, i: usize) -> bool {
        self.active && self.parked.get(i).copied().unwrap_or(false)
    }
}

/// Below this `|v|²` (`|v| < 2⁻⁸`) the rotation angle `2·asin|v|` is taken
/// from its series, whose first omitted term is below `2⁻⁶⁴` relative.
const LOG_SERIES_LIMIT_SQ: Fix128 = Fix128::from_raw(0, 1 << 48);

/// Angular velocity derived from a rotation change by the exact logarithm of
/// the rotation: with `dq = q·p⁻¹ = (v, w)`, its sign folded so that `w ≥ 0`
/// (the shorter of the two equivalent rotations), the rotation is
/// `θ = 2·atan2(|v|, w)` about `v / |v|`, so `ω = v̂ θ / dt`.
///
/// The angle is evaluated as the factor `θ / |v| = 2·asin(|v|)/|v|`
/// multiplying `v`: from the series `2 (1 + s²/6 + 3s⁴/40 + 5s⁶/112)`
/// (`s² = |v|²`) below `|v| = 2⁻⁸`, which needs no square root and keeps
/// small rotations exact to rounding, and from `2·atan2(|v|, w)/|v|` above.
/// A rotation predicted as `from_axis_angle(ω̂, |ω| dt)` therefore gives back
/// `ω` (the earlier chord `2 v / dt` gave `(2/dt) sin(|ω| dt / 2)`, which
/// lost `|ω|³ dt² / 24` per call).
#[inline]
fn angular_from_rotations(rotation: QuatFix, prev_rotation: QuatFix, inv_dt: Fix128) -> Vec3Fix {
    let dq = rotation.mul(prev_rotation.conjugate());
    let (v, w) = if dq.w < Fix128::ZERO {
        (Vec3Fix::new(-dq.x, -dq.y, -dq.z), -dq.w)
    } else {
        (Vec3Fix::new(dq.x, dq.y, dq.z), dq.w)
    };
    let s2 = v.dot(v);
    let factor = if s2 < LOG_SERIES_LIMIT_SQ {
        let series = Fix128::from_ratio(1, 6)
            + s2 * (Fix128::from_ratio(3, 40) + s2 * Fix128::from_ratio(5, 112));
        (Fix128::ONE + s2 * series).double()
    } else {
        let s = s2.sqrt();
        Fix128::atan2(s, w).double() / s
    };
    v * (factor * inv_dt)
}

/// Kind tags for [`tgs_cache_key`]: contacts and distance constraints share
/// one warm-start cache.
#[cfg(feature = "std")]
const TGS_KEY_CONTACT: u64 = 0;
#[cfg(feature = "std")]
const TGS_KEY_DISTANCE: u64 = 1;

/// Warm-start cache key for the `ordinal`-th entry of kind `tag` joining the
/// bodies with stable ids `a` → `b` this tick (`ordinals` counts per pair and
/// is reset every tick by the caller creating it afresh).
///
/// The four words are mixed with the SplitMix64 finaliser, so the key is a
/// deterministic function of what the entry joins — never of where it sits
/// in a vector. Two distinct entries collide with probability about
/// `n² / 2⁶⁵` per tick for `n` entries.
#[cfg(feature = "std")]
fn tgs_cache_key(
    tag: u64,
    a: u64,
    b: u64,
    ordinals: &mut std::collections::BTreeMap<(u64, u64), u64>,
) -> u64 {
    let slot = ordinals.entry((a, b)).or_insert(0);
    let ordinal = *slot;
    *slot += 1;
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for w in [tag, a, b, ordinal] {
        h ^= w;
        h = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
        h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 31;
    }
    h
}

/// Predict a dynamic body's rotation over one substep `dt`.
///
/// A body with a gyroscopic response is advanced by the symplectic splitting
/// of [`crate::gyroscopic::split_free_rotation`]: its rotation and angular
/// velocity become the split's end state, and the split is returned so that
/// `update_velocities` can rebuild `ω` from it. Any other body (isotropic,
/// at rest, infinite moments) turns about `ω` by `|ω| dt`, as before.
///
/// `|ω|² ≥ 2⁶³` (`|ω| ≥ 2³¹·⁵ ≈ 3.04e9` rad/s) では `normalize_with_length` の
/// 長さが wrap して 0 になり、回転が止まって速度導出で ω まで 0 に消える
/// その範囲だけ、2 乗を経ない長さと向き ([`Vec3Fix::checked_length_scaled`] /
/// [`Vec3Fix::try_normalize_scaled`]) で回す 長さ自体か `|ω|·dt` が表せない
/// 時は回さずに `overflow` を立てる 範囲内は従来の式のまま (bit 不変)
#[inline]
fn predict_rotation(
    body: &mut RigidBody,
    dt: Fix128,
    overflow: &mut bool,
) -> Option<crate::gyroscopic::FreeRotation> {
    let split = crate::gyroscopic::split_free_rotation(
        body.angular_velocity,
        body.rotation,
        body.inv_inertia,
        dt,
    );
    match split {
        Some(f) => {
            body.angular_velocity = f.omega;
            body.rotation = f.rotation;
        }
        None => {
            let omega = body.angular_velocity;
            if let Some(speed_sq) = omega.checked_length_squared() {
                // `normalize_with_length` の式を展開したもの (single sqrt、
                // `Some` の 2 乗は `length_squared` と bit 一致)
                let ang_speed = speed_sq.sqrt();
                if !ang_speed.is_zero() {
                    let axis = omega * (Fix128::ONE / ang_speed);
                    let delta_rot = QuatFix::from_axis_angle(axis, ang_speed * dt);
                    body.rotation = delta_rot.mul(body.rotation).normalize();
                }
            } else {
                // |ω|² が範囲外: ω は非零 (零なら 2 乗は 0) なので向きは必ずある
                let turn = omega
                    .checked_length_scaled()
                    .and_then(|s| s.checked_mul(dt))
                    .zip(omega.try_normalize_scaled());
                match turn {
                    Some((angle, axis)) => {
                        let delta_rot = QuatFix::from_axis_angle(axis, angle);
                        body.rotation = delta_rot.mul(body.rotation).normalize();
                    }
                    None => *overflow = true,
                }
            }
        }
    }
    split
}

/// Linear velocity at the end of a substep, written into `body.velocity`.
///
/// A body predicted this substep (`predicted = Some(x_pred)`, with
/// `body.velocity` still the predicted velocity `v_pred` the prediction
/// `x_pred = x_prev + v_pred·h` used) gets `v = v_pred + (x − x_pred) / h`:
/// the predicted velocity plus the correction the solve applied. A body no
/// constraint or contact moved keeps `v_pred` bit for bit. In exact
/// arithmetic this equals `(x − x_prev) / h`; re-deriving from the position
/// difference instead dropped the low bits that the truncating product
/// `v_pred·h` lost, every substep, so free bodies did not conserve momentum.
/// Any other body (`None`) derives `v = (x − x_prev) / h`, as before.
/// A substep whose position add left the range keeps the in-range predicted
/// velocity: the position stays put, so it equals `x_pred` and the
/// correction is zero.
///
/// Returns `false` when the scaled difference leaves the `Fix128` range; the
/// velocity is then left unchanged.
#[inline]
fn derive_linear_velocity(
    body: &mut RigidBody,
    predicted: Option<Vec3Fix>,
    inv_dt: Fix128,
) -> bool {
    match predicted {
        Some(x_pred) if body.position == x_pred => true,
        Some(x_pred) => match (body.position - x_pred).checked_scale(inv_dt) {
            Some(dv) => {
                body.velocity = body.velocity + dv;
                true
            }
            None => false,
        },
        None => match (body.position - body.prev_position).checked_scale(inv_dt) {
            Some(v) => {
                body.velocity = v;
                true
            }
            None => false,
        },
    }
}

/// Angular velocity at the end of a substep. A body predicted by the
/// splitting keeps the split's end velocity plus the rotation the position
/// solve added beyond the split's end orientation (zero when no constraint
/// turned it). Any other body predicted this substep (`predicted =
/// Some(q_pred)`, `body.angular_velocity` still the `ω_pred` that predicted
/// `q_pred`) does the same with `ω_pred` and `q_pred`: a body the solve did
/// not turn keeps `ω_pred` bit for bit (re-deriving it from
/// `log(q · q_prev⁻¹)` rounds through the axis-angle, the normalisation and
/// the logarithm). A body not predicted (`None`: static, kinematic,
/// sleeping, parked) derives `ω` from its rotation change, as
/// [`angular_from_rotations`] always did.
#[inline]
fn derived_angular_velocity(
    body: &RigidBody,
    free: Option<crate::gyroscopic::FreeRotation>,
    predicted: Option<QuatFix>,
    inv_dt: Fix128,
) -> Vec3Fix {
    match (free, predicted) {
        (Some(f), _) if body.rotation == f.rotation => f.omega,
        (Some(f), _) => f.omega + angular_from_rotations(body.rotation, f.rotation, inv_dt),
        (None, Some(q)) if body.rotation == q => body.angular_velocity,
        (None, Some(q)) => body.angular_velocity + angular_from_rotations(body.rotation, q, inv_dt),
        (None, None) => angular_from_rotations(body.rotation, body.prev_rotation, inv_dt),
    }
}

/// Snapshot of [`SolverBackend::Tgs`]'s per-frame warm-start impulse cache
/// effectiveness, returned by [`PhysicsWorld::tgs_cache_stats`].
///
/// Mirrors `crate::solver_tgs::ImpulseCacheStats` (which stays
/// crate-internal along with the rest of the `solver_tgs` family, see
/// [`SolverBackend`]'s module doc) so a host can read warm-start
/// diagnostics without depending on any `pub(crate)` type.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TgsCacheStats {
    /// Number of contacts whose previous-tick impulse was found in the
    /// cache (a successful warm start).
    pub hits: u64,
    /// Number of contacts not found in the cache (new contact, or one
    /// whose entry was evicted by [`PhysicsWorld::step`]'s per-tick
    /// internal `ImpulseCache::sweep` call).
    pub misses: u64,
}

#[cfg(feature = "std")]
impl TgsCacheStats {
    /// Constructs from raw hit/miss counts. Prefer
    /// [`PhysicsWorld::tgs_cache_stats`] in production; this exists so
    /// test/bench code can construct a specific ratio directly —
    /// `#[non_exhaustive]` blocks the struct-literal syntax outside this
    /// crate.
    #[must_use]
    pub fn new(hits: u64, misses: u64) -> Self {
        Self { hits, misses }
    }

    /// Ratio in `[0.0, 1.0]`. Returns `0.0` when `hits + misses == 0`
    /// (no lookups performed yet). Delegates to the internal
    /// `ImpulseCacheStats::hit_rate` (the same `hits / (hits + misses)`
    /// formula) rather than recomputing it here, so this public wrapper
    /// has a real dependency on the internal type instead of merely
    /// duplicating its arithmetic.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        crate::solver_tgs::ImpulseCacheStats {
            hits: self.hits,
            misses: self.misses,
        }
        .hit_rate()
    }
}

#[cfg(feature = "std")]
impl From<crate::solver_tgs::ImpulseCacheStats> for TgsCacheStats {
    fn from(s: crate::solver_tgs::ImpulseCacheStats) -> Self {
        Self {
            hits: s.hits,
            misses: s.misses,
        }
    }
}

/// XPBD physics solver configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolverConfig {
    /// Number of substeps per frame
    pub substeps: usize,
    /// Number of constraint solver passes per substep (default 1 since 1.2.0,
    /// Small Steps: prefer more `substeps` over more `iterations`)
    pub iterations: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Global velocity-retention factor, applied **once per frame** (`step()`),
    /// independent of `substeps` (1.0 = no damping). Before 1.2.0 it was applied
    /// per substep, which made the result depend on `substeps`.
    pub damping: Fix128,
    /// **No effect since 1.2.0** (kept for struct compatibility).
    ///
    /// Distance constraints follow standard XPBD (`lambda = 0` at substep
    /// start, accumulated over iterations) and contacts are re-detected every
    /// substep and accumulate their multiplier within it, so there is no
    /// previous-substep multiplier left to warm-start from. Before 1.2.0 the
    /// factor biased both solvers and was the source of the iteration-dependent
    /// stiffness (distance) and the energy-creating re-push (contact).
    /// 0.8〜0.95 が安定的。デフォルト 0.85。
    pub warm_start_factor: Fix128,
    /// Which integrator [`PhysicsWorld::step`] dispatches to. Default
    /// [`SolverBackend::Xpbd`] — selecting it is structurally a no-op (the
    /// TGS dispatch branch is behind a runtime `matches!` check that is
    /// never taken), so every existing caller that does not set this field
    /// keeps today's `step` behavior bit-for-bit. See [`SolverBackend`] for
    /// the `Tgs` path's documented gaps.
    pub solver_backend: SolverBackend,
}

/// Physics configuration (alias for `SolverConfig`)
pub type PhysicsConfig = SolverConfig;

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            substeps: 8,
            // 1.2.0: Small Steps (Müller et al. 2020) — with collision detection
            // and constraint re-evaluation per substep, one Gauss-Seidel pass per
            // substep beats several; the previous 8 × 4 = 32 passes cost 4× for
            // no accuracy gain. Raise `iterations` only for stiff *rigid*
            // constraint chains that must converge within a single substep.
            iterations: 1,
            gravity: Vec3Fix::new(
                Fix128::ZERO,
                Fix128::from_int(-10), // -10 m/s^2
                Fix128::ZERO,
            ),
            damping: Fix128::from_ratio(99, 100), // 0.99 velocity retention
            warm_start_factor: Fix128::from_ratio(85, 100), // 0.85
            solver_backend: SolverBackend::Xpbd,
        }
    }
}

// ============================================================================
// Constraint Batching (Graph Coloring)
// ============================================================================

/// Constraint batch for parallel processing
///
/// A batch contains constraint indices that share no bodies,
/// allowing safe parallel modification.
#[derive(Clone, Debug, Default)]
pub struct ConstraintBatch {
    /// Indices into `distance_constraints`
    pub distance_indices: Vec<usize>,
    /// Indices into `contact_constraints`
    pub contact_indices: Vec<usize>,
}

/// Pointer wrapper enabling parallel access to disjoint body slots.
///
/// # Safety
///
/// The graph coloring invariant (`rebuild_batches`) guarantees that within
/// a single constraint batch, no two constraints share a *dynamic* body
/// index. Static / kinematic bodies (`inv_mass == 0`) are excluded from
/// coloring and are only ever handed out as shared references
/// (`BodyRef::Static`), so many threads may read them concurrently while
/// no thread writes them.
#[cfg(feature = "parallel")]
struct BodySlicePtr {
    ptr: *mut RigidBody,
    len: usize,
}

// SAFETY: `BodySlicePtr` is only used inside the parallel constraint solver
// where graph-coloring guarantees that no two threads take `&mut` to the
// same index. Static bodies are shared read-only, dynamic bodies appear in
// at most one constraint per batch, so all accesses are data-race-free.
#[cfg(feature = "parallel")]
unsafe impl Send for BodySlicePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for BodySlicePtr {}

#[cfg(feature = "parallel")]
impl BodySlicePtr {
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    /// # Safety
    ///
    /// Caller must ensure `idx < self.len` and that no other thread
    /// concurrently accesses the same index (shared or exclusive).
    unsafe fn get_mut(&self, idx: usize) -> &mut RigidBody {
        debug_assert!(idx < self.len);
        &mut *self.ptr.add(idx)
    }

    #[inline(always)]
    /// # Safety
    ///
    /// Caller must ensure `idx < self.len` and that no other thread
    /// concurrently holds a `&mut` to the same index.
    unsafe fn get(&self, idx: usize) -> &RigidBody {
        debug_assert!(idx < self.len);
        &*self.ptr.add(idx)
    }

    #[inline(always)]
    /// Borrow slot `idx` as shared (static body) or exclusive (dynamic body).
    ///
    /// # Safety
    ///
    /// `is_static` must be the value recorded by `rebuild_batches` for this
    /// index (see `PhysicsWorld::batch_static_bodies`); passing `false` for a
    /// body that another thread also borrows is undefined behaviour.
    unsafe fn borrow(&self, idx: usize, is_static: bool) -> BodyRef<'_> {
        if is_static {
            BodyRef::Static(self.get(idx))
        } else {
            BodyRef::Dynamic(self.get_mut(idx))
        }
    }
}

/// Static friction of a contact at the position level (Müller, Macklin,
/// Chentanez, Jeschke, Kim, "Detailed Rigid Body Simulation with Extended
/// Position Based Dynamics", SCA 2020, eq. (26)).
///
/// `disp_a` / `disp_b` are the displacements of the two bodies since the
/// start of the substep (`position − prev_position`), `lambda` the contact's
/// accumulated normal separation this substep (`cached_lambda`) and `mu` its
/// friction coefficient. The relative tangential displacement
/// `Δp_t = (Δx_a − Δx_b)` minus its normal part is cancelled when
/// `|Δp_t| < μ λ` (both are distances: the paper compares the multipliers
/// `λ_t < μ_s λ_n`, which share the factor `1 / (w_a + w_b)`), split by
/// inverse mass like the normal correction. Returns the corrections to
/// subtract from `a` and add to `b`, or `None` when the contact slides (or
/// has no tangential motion / no normal multiplier). Contact points are
/// taken at the body centres: the contact solve is translational only.
///
/// `mu` is the contact's friction (`ContactConstraint::friction`, after the
/// contact modifiers); the combined material carries one coefficient, so the
/// static and the kinetic coefficients are equal.
// LIMITATION(COV-RIGID-080): the combined material carries one coefficient, so the static and the kinetic coefficients are equal.
// LIMITATION(COV-RIGID-083): Contact points are taken at the body centres: the contact solve is translational only.
#[allow(clippy::too_many_arguments)]
#[inline]
fn static_friction_correction(
    disp_a: Vec3Fix,
    disp_b: Vec3Fix,
    normal: Vec3Fix,
    inv_mass_a: Fix128,
    inv_mass_b: Fix128,
    inv_w_sum: Fix128,
    mu: Fix128,
    lambda: Fix128,
) -> Option<(Vec3Fix, Vec3Fix)> {
    let limit = mu * lambda;
    if limit <= Fix128::ZERO {
        return None;
    }
    let dp = disp_a - disp_b;
    let dp_t = dp - normal * dp.dot(normal);
    let len_sq = dp_t.length_squared();
    if len_sq.is_zero() || len_sq >= limit * limit {
        return None;
    }
    Some((
        dp_t * (inv_mass_a * inv_w_sum),
        dp_t * (inv_mass_b * inv_w_sum),
    ))
}

/// Body access handed to the parallel pair solvers.
///
/// Dynamic bodies are borrowed exclusively and receive position writes;
/// static / kinematic bodies are borrowed shared and position writes are
/// dropped (they would be no-ops anyway: `inv_mass == 0` makes every
/// correction zero and the sequential path writes the unchanged position).
#[cfg(feature = "parallel")]
enum BodyRef<'a> {
    Dynamic(&'a mut RigidBody),
    Static(&'a RigidBody),
}

#[cfg(feature = "parallel")]
impl BodyRef<'_> {
    #[inline(always)]
    fn get(&self) -> &RigidBody {
        match self {
            BodyRef::Dynamic(b) => b,
            BodyRef::Static(b) => b,
        }
    }

    #[inline(always)]
    fn set_position(&mut self, position: Vec3Fix) {
        if let BodyRef::Dynamic(b) = self {
            b.position = position;
        }
    }
}

/// Pointer wrapper enabling parallel write-back of cached lambda.
///
/// Each constraint index appears in exactly one batch, so concurrent
/// mutation of distinct constraint slots is safe.
#[cfg(feature = "parallel")]
struct DistConstraintSlicePtr {
    ptr: *mut DistanceConstraint,
    len: usize,
}

// SAFETY: `DistConstraintSlicePtr` is only used inside the parallel
// constraint solver where each constraint index appears in exactly one
// graph-colored batch. Concurrent `get_mut` calls target disjoint indices,
// preventing data races.
#[cfg(feature = "parallel")]
unsafe impl Send for DistConstraintSlicePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for DistConstraintSlicePtr {}

#[cfg(feature = "parallel")]
impl DistConstraintSlicePtr {
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    /// # Safety
    ///
    /// Caller must ensure `idx < self.len` and that no other thread
    /// concurrently accesses the same index.
    unsafe fn get_mut(&self, idx: usize) -> &mut DistanceConstraint {
        debug_assert!(idx < self.len);
        &mut *self.ptr.add(idx)
    }
}

/// Pointer wrapper enabling parallel write-back of cached lambda for contact constraints.
#[cfg(feature = "parallel")]
struct ContactConstraintSlicePtr {
    ptr: *mut ContactConstraint,
    len: usize,
}

// SAFETY: Same guarantee as `DistConstraintSlicePtr` — each constraint index
// appears in exactly one graph-colored batch.
#[cfg(feature = "parallel")]
unsafe impl Send for ContactConstraintSlicePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for ContactConstraintSlicePtr {}

#[cfg(feature = "parallel")]
impl ContactConstraintSlicePtr {
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    /// # Safety
    ///
    /// Caller must ensure `idx < self.len` and that no other thread
    /// concurrently accesses the same index.
    unsafe fn get_mut(&self, idx: usize) -> &mut ContactConstraint {
        debug_assert!(idx < self.len);
        &mut *self.ptr.add(idx)
    }
}

/// Pre-solve contact hook callback type
///
/// Called before each contact is solved. Return `false` to skip this contact.
/// This allows game logic to filter contacts (e.g., one-way platforms,
/// character controllers that ignore certain collisions).
#[cfg(feature = "std")]
pub type PreSolveHook = Box<dyn Fn(usize, usize, &Contact) -> bool + Send + Sync>;

/// Contact modification callback trait
///
/// Implement to modify contact properties before solving.
/// More powerful than `PreSolveHook`: can mutate normal, depth,
/// friction, and restitution per-contact.
#[cfg(feature = "std")]
pub trait ContactModifier: Send + Sync {
    /// Modify a contact before solving.
    ///
    /// Return `false` to discard the contact entirely.
    /// Modify the mutable references to change contact properties.
    fn modify_contact(
        &self,
        body_a: usize,
        body_b: usize,
        contact: &mut Contact,
        friction: &mut Fix128,
        restitution: &mut Fix128,
    ) -> bool;
}

/// Typed, read-only observation of a single body's law-relevant state.
///
/// Public入口 for [`PhysicsWorld::observe_body`] / [`PhysicsWorld::observe_bodies`]
/// — 「Law が読む形の観測型」 observation 以前は blob の
/// parse か field 直読みしかなく、公開 interface が無かった
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyObservation {
    /// Body index this observation was taken from.
    ///
    /// ⚠️ **フレーム内でのみ安定** — [`PhysicsWorld::remove_body`] は
    /// `swap_remove` なので、body の生成・破棄を跨いで同じ index が同じ
    /// body を指す保証はない
    pub body_index: usize,
    /// Position (center of mass).
    pub position: Vec3Fix,
    /// Linear velocity.
    pub velocity: Vec3Fix,
    /// Orientation.
    pub rotation: QuatFix,
    /// Angular velocity.
    pub angular_velocity: Vec3Fix,
    /// Whether the body is currently asleep (island manager).
    pub sleeping: bool,
    /// Whether the body has at least one active contact this frame
    /// (`Begin` or `Persist`, from [`crate::event::EventCollector`]).
    ///
    /// ⚠️ **`grounded` ではない** — 一般 body に地面接触の定義は無いため、
    /// 「何らかの body と接触している」という弱い述語に留める
    pub in_contact: bool,
}

/// The per-body settings of a [`PhysicsWorld`] kept outside [`RigidBody`],
/// moved together with a body between worlds (see `MultiWorld::transfer_body`).
#[derive(Clone, Debug)]
pub(crate) struct BodyAttachments {
    material: crate::material::MaterialId,
    collision_radius: Option<Fix128>,
    collider: Option<crate::body_collider::BodyCollider>,
    filter: CollisionFilter,
}

/// XPBD physics world with batched constraint solving
///
/// `PhysicsWorld` integrates all physics subsystems:
/// - Rigid body dynamics with XPBD solver
/// - Automatic collision detection (BVH broad-phase + sphere narrow-phase)
/// - Joint constraints (Ball, Hinge, Fixed, Slider, Spring, D6, `ConeTwist`)
/// - Force fields (wind, gravity wells, buoyancy, vortex, drag)
/// - Collision filtering (layer/mask bitmasks)
/// - Contact events (begin/persist/end)
/// - Sleeping/island management
pub struct PhysicsWorld {
    /// Solver configuration
    pub config: SolverConfig,
    /// All rigid bodies in the world
    pub bodies: Vec<RigidBody>,
    /// Distance constraints between bodies
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Contact constraints from collision detection
    pub contact_constraints: Vec<ContactConstraint>,
    /// SDF colliders for implicit collision detection
    pub sdf_colliders: Vec<SdfCollider>,
    /// Default collision radius for body-vs-SDF queries
    pub sdf_collision_radius: Fix128,
    /// Planes, height fields and triangle meshes that bodies rest on (see
    /// [`crate::static_collider`])
    static_colliders: Vec<crate::static_collider::StaticCollider>,
    /// Pre-colored constraint batches (computed on demand)
    constraint_batches: Vec<ConstraintBatch>,
    /// Whether batches need recomputation
    batches_dirty: bool,
    /// Per-body snapshot of `inv_mass == 0` taken by `rebuild_batches`.
    ///
    /// Bodies flagged here were excluded from graph coloring (they are never
    /// written by the pair solvers), so the parallel path must only ever take
    /// shared references to them. `solve_constraints_batched` re-validates
    /// this snapshot against the live bodies before dispatch and rebuilds the
    /// batches if any body changed static-ness in between.
    batch_static_bodies: Vec<bool>,
    /// Contact manifold cache for warm starting
    pub contact_cache: crate::contact_cache::ContactCache,
    /// Material pair lookup table
    pub material_table: crate::material::MaterialTable,
    /// Per-body material IDs
    pub body_materials: Vec<crate::material::MaterialId>,
    /// Pre-solve contact hooks (called before contact resolution)
    #[cfg(feature = "std")]
    pre_solve_hooks: Vec<PreSolveHook>,
    /// Contact modifiers (called before contact resolution, can mutate contact)
    #[cfg(feature = "std")]
    contact_modifiers: Vec<Box<dyn ContactModifier>>,
    /// Constraints the hooks / modifiers discarded in the last
    /// [`Self::apply_contact_filters`] pre-pass (indexed like
    /// `contact_constraints`; an index past the end counts as kept)
    #[cfg(feature = "std")]
    contact_discarded: Vec<bool>,
    /// Pre-solve normal velocity `v̄_n` of each contact constraint in the
    /// current substep (indexed like `contact_constraints`), recorded by
    /// [`Self::update_velocities`] before it derives the post-solve
    /// velocities; the restitution target is `−e v̄_n`
    contact_pre_vn: Vec<Fix128>,
    /// Free rotation of each body in the current substep, written by
    /// `integrate_positions` and read by `update_velocities` (indexed like
    /// `bodies`; `None`: no gyroscopic response). Transient: rebuilt every
    /// substep, so it carries no state across steps.
    free_rotation: Vec<Option<crate::gyroscopic::FreeRotation>>,
    /// Predicted pose (position, rotation) of each body in the current
    /// substep, written by `integrate_positions` after `x += v·h` and the
    /// rotation prediction, and read by `update_velocities` (indexed like
    /// `bodies`; `None`: the body was not predicted this substep — static,
    /// kinematic, sleeping or parked — and derives its velocities from
    /// `prev_position` / `prev_rotation`). Transient: rebuilt every substep,
    /// so it carries no state across steps.
    predicted_pose: Vec<Option<(Vec3Fix, QuatFix)>>,
    /// v0.11.0: installed GPU solver bridge for automatic contact-solve
    /// routing. When `Some`, every call to [`Self::step`] /
    /// [`Self::substep`] transparently routes contact-solve through the
    /// bridge instead of running the CPU solver inline. Installed and
    /// removed via [`Self::set_gpu_solver_bridge`] and
    /// [`Self::take_gpu_solver_bridge`]. Callers who want explicit
    /// control can bypass the field entirely by using
    /// [`Self::step_with_bridge`] / [`Self::substep_with_bridge`] /
    /// [`Self::solve_contact_constraints_with_bridge`] directly — those
    /// helpers ignore the installed bridge and use the borrowed one
    /// passed in.
    #[cfg(feature = "gpu-solver-bridge")]
    gpu_solver_bridge: Option<Box<dyn crate::gpu_bridge::GpuSolverBridge + Send + Sync>>,

    // ── Integrated Subsystems ──────────────────────────────────────────
    /// Joint constraints (solved each substep alongside distance/contact)
    pub joints: Vec<Joint>,
    /// Joint motors (one-axis PD controllers), applied every substep by
    /// [`Self::step`]; see [`Self::add_joint_motor`]
    joint_motors: Vec<crate::motor::JointMotor>,
    /// Three-axis rotation motors `(joint index, controller)`, applied every
    /// substep by [`Self::step`]; see [`Self::add_joint_motor_3d`]
    joint_motors_3d: Vec<(usize, crate::motor::PdController3D)>,
    /// Force fields applied at the start of each step
    pub force_fields: Vec<ForceFieldInstance>,
    /// Contact and trigger event collector
    pub events: EventCollector,
    /// Island manager for sleeping and connectivity tracking
    pub islands: IslandManager,
    /// Per-body collision radius for automatic sphere-based collision detection.
    /// `None` means the body does not participate in auto-detection.
    body_collision_radii: Vec<Option<Fix128>>,
    /// Which broad-phase finds the candidate pairs.
    broadphase: Broadphase,
    /// The persistent tree of [`Broadphase::DynamicTree`] (unused otherwise).
    broadphase_tree: crate::dynamic_bvh::DynamicAabbTree,
    /// Body index → its proxy in `broadphase_tree`.
    broadphase_proxies: Vec<Option<u32>>,
    /// The layers of [`Broadphase::Hybrid`] (unused otherwise).
    broadphase_hybrid: BroadphaseHybrid,
    /// The convex collider a body carries, when it has one — one shape or a
    /// compound of them: the narrow-phase works on it instead of the bounding
    /// sphere (see [`crate::shape`], [`crate::compound`]).
    body_colliders: Vec<Option<crate::body_collider::BodyCollider>>,
    /// Per-body collision filter (layer/mask/group)
    body_filters: Vec<CollisionFilter>,
    /// 範囲外の積を踏んだか (sticky、[`PhysicsWorld::overflow_detected`])
    ///
    /// ⚠️ **`Fix128` の `Mul` 自体からは world に到達できない** (純粋な trait
    /// impl、crate は `no_std` 対応なので `thread_local` も使えない) ので、
    /// **発散が起きる経路** (積分と速度導出) で `checked_mul` を呼んでここに
    /// 立てる doctrine B-12 の「`mul` の中で world の flag を立てる」は
    /// 到達不能なため、WM-01 の結論「`mul` の積が範囲外を演算側で見る」を
    /// 呼び出し側で実現した形
    overflow_detected: bool,
    /// Substeps still to run in the current frame, counting the one being run
    /// (0 outside a substep loop). A kinematic body with a target closes
    /// `1 / kinematic_substeps_left` of its remaining gap in each substep, so
    /// the velocity derived after every substep is the frame velocity
    /// (see [`kinematic_substep_pose`]).
    kinematic_substeps_left: usize,
    /// Warm-start impulse cache for the [`SolverBackend::Tgs`] path, persisted
    /// across frames the same way [`Self::contact_cache`] is for XPBD.
    /// Unused (and empty) while `config.solver_backend` is `Xpbd`.
    #[cfg(feature = "std")]
    tgs_impulse_cache: crate::solver_tgs::ImpulseCache,
    /// Per-body identity that survives `remove_body`'s `swap_remove` (the
    /// index does not). The TGS warm-start cache keys contacts and distance
    /// constraints by these, so a reorder of `bodies` cannot hand one
    /// contact's cached impulse to another. Kept parallel to `bodies` by
    /// [`Self::sync_body_stable_ids`], which also covers bodies pushed onto
    /// the public `bodies` field directly.
    #[cfg(feature = "std")]
    body_stable_ids: Vec<u64>,
    /// The next id [`Self::sync_body_stable_ids`] hands out.
    #[cfg(feature = "std")]
    next_body_stable_id: u64,
    /// Skip parked sleeping bodies in `step` (see [`StageWork`]).
    sleep_skip: bool,
    /// Parked bodies, their tree and caches.
    park: ParkState,
    /// Bumped by every change that can invalidate a parked body's verdict but is
    /// not visible in its own state (shapes, radii, static colliders, removal).
    park_generation: u64,
    /// Work counters of the last `step`.
    stage_work: StageWork,
    /// Participants in registration order (see [`crate::world_participant`]).
    /// Held behind a mutex so that the world stays `Sync` while a participant
    /// is only `Send`; every access during a step goes through `&mut self`
    /// ([`std::sync::Mutex::get_mut`], no locking).
    #[cfg(feature = "std")]
    participants: std::sync::Mutex<Vec<Box<dyn crate::world_participant::Participant>>>,
    /// Run order and declared ports of `participants`, rebuilt whenever a
    /// participant or a field is added; `None` while no participant is
    /// registered.
    #[cfg(feature = "std")]
    participant_plan: Option<crate::world_participant::ParticipantPlan>,
    /// The first fault recorded (sticky until [`Self::clear_fault`]).
    fault: Option<crate::world_participant::WorldFault>,
    /// Shared fields, owned by the world.
    fields: crate::world_participant::FieldBoard,
    /// Continuous collision of the step ([`Self::set_continuous_collision`]).
    ccd: WorldCcdConfig,
}

/// Fold `bytes` into `hash` with FNV-1a (64-bit).
///
/// Used by [`PhysicsWorld::population_fingerprint`]; kept as a free
/// function (not a closure) to avoid a double-borrow of `hash`.
fn fnv1a_fold(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
}

/// The contact of one body with one SDF collider: as its shape or compound when it
/// has one, otherwise as a sphere of `radius` at its position.
#[cfg(feature = "std")]
fn sdf_contact_of(
    collider: Option<&crate::body_collider::BodyCollider>,
    body: &RigidBody,
    radius: Fix128,
    sdf: &crate::sdf_collider::SdfCollider,
) -> Option<crate::collider::Contact> {
    match collider {
        Some(c) => c.sdf_contact(body.position, body.rotation, sdf),
        None => crate::sdf_collider::collide_sphere_sdf(body.position, radius, sdf),
    }
}

/// Pose of a kinematic body after one substep towards its target.
///
/// `left` is the number of substeps left in the frame including the current
/// one. Each substep closes `1 / left` of the remaining gap, which makes the
/// per-substep displacement the same in every substep (the gap shrinks by
/// `(left - 1) / left` each time), and the last substep (`left <= 1`, also the
/// value outside a substep loop) lands on the target exactly.
fn kinematic_substep_pose(
    position: Vec3Fix,
    rotation: QuatFix,
    target_pos: Vec3Fix,
    target_rot: QuatFix,
    left: usize,
) -> (Vec3Fix, QuatFix) {
    if left <= 1 {
        return (target_pos, target_rot);
    }
    let frac = Fix128::ONE / Fix128::from_int(left as i64);
    (
        position + (target_pos - position) * frac,
        crate::interpolation::slerp(rotation, target_rot, frac),
    )
}

impl PhysicsWorld {
    /// Create a new, empty physics world with the given solver configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
    ///
    /// let config = PhysicsConfig::default();
    /// let mut world = PhysicsWorld::new(config);
    ///
    /// // World starts with no bodies
    /// assert_eq!(world.bodies.len(), 0);
    ///
    /// // Add a body and verify it is tracked
    /// let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
    /// let id = world.add_body(body);
    /// assert_eq!(id, 0);
    /// assert_eq!(world.bodies.len(), 1);
    /// ```
    #[must_use]
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            bodies: Vec::new(),
            distance_constraints: Vec::new(),
            contact_constraints: Vec::new(),
            sdf_colliders: Vec::new(),
            sdf_collision_radius: Fix128::from_ratio(1, 2), // 0.5 default
            static_colliders: Vec::new(),
            constraint_batches: Vec::new(),
            batches_dirty: true,
            batch_static_bodies: Vec::new(),
            contact_cache: crate::contact_cache::ContactCache::new(),
            material_table: crate::material::MaterialTable::new(),
            body_materials: Vec::new(),
            #[cfg(feature = "std")]
            pre_solve_hooks: Vec::new(),
            #[cfg(feature = "std")]
            contact_modifiers: Vec::new(),
            #[cfg(feature = "std")]
            contact_discarded: Vec::new(),
            contact_pre_vn: Vec::new(),
            free_rotation: Vec::new(),
            predicted_pose: Vec::new(),
            #[cfg(feature = "gpu-solver-bridge")]
            gpu_solver_bridge: None,
            joints: Vec::new(),
            joint_motors: Vec::new(),
            joint_motors_3d: Vec::new(),
            force_fields: Vec::new(),
            events: EventCollector::new(),
            islands: IslandManager::new(0, SleepConfig::default()),
            body_collision_radii: Vec::new(),
            broadphase: Broadphase::default(),
            broadphase_tree: crate::dynamic_bvh::DynamicAabbTree::new(),
            broadphase_proxies: Vec::new(),
            broadphase_hybrid: BroadphaseHybrid::new(),
            body_colliders: Vec::new(),
            body_filters: Vec::new(),
            overflow_detected: false,
            kinematic_substeps_left: 0,
            #[cfg(feature = "std")]
            tgs_impulse_cache: crate::solver_tgs::ImpulseCache::new(),
            #[cfg(feature = "std")]
            body_stable_ids: Vec::new(),
            #[cfg(feature = "std")]
            next_body_stable_id: 0,
            sleep_skip: true,
            park: ParkState::default(),
            park_generation: 0,
            stage_work: StageWork::default(),
            #[cfg(feature = "std")]
            participants: std::sync::Mutex::new(Vec::new()),
            #[cfg(feature = "std")]
            participant_plan: None,
            fault: None,
            fields: crate::world_participant::FieldBoard::new(),
            ccd: WorldCcdConfig::new(),
        }
    }

    /// Reset the world to the same empty state [`Self::new`] produces,
    /// keeping the current [`SolverConfig`].
    ///
    /// # World Auditor の `reset()` 契約 (WM-07)
    ///
    /// 「同じ初期状態から N step を 2 回実行すると bit 一致する」を成立させる
    /// ための入口 `*self = Self::new(self.config)` と等価 (`config` は
    /// `Copy`) — `bodies` / `joints` / `force_fields` / `islands` /
    /// `overflow_detected` を含む **全 field が `new` と同じ既定値**に戻る
    /// (installed hook / GPU bridge も含む、呼び出し側が明示的に残したい
    /// 状態があれば `reset_world` の前に退避すること)
    ///
    /// [`Self::deserialize_state`] による rollback とは異なる経路 — rollback
    /// は「population を caller が replay で合わせてから特定 frame の連続状態を
    /// 復元する」契約だが、`reset_world` は「population ごと空にする」契約
    /// 両者は **「population を目標状態まで合わせてから連続状態を復元する」**
    /// という同じ形の特殊例 (`reset_world` は目標 population が空の場合)
    ///
    /// # Examples
    ///
    /// ```
    /// use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
    ///
    /// let config = PhysicsConfig::default();
    /// let mut world = PhysicsWorld::new(config);
    /// world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE));
    /// assert_eq!(world.bodies.len(), 1);
    ///
    /// world.reset_world();
    /// assert_eq!(world.bodies.len(), 0);
    /// assert!(!world.overflow_detected());
    /// ```
    pub fn reset_world(&mut self) {
        *self = Self::new(self.config);
    }

    /// v0.11.0: install a GPU solver bridge for automatic contact-solve
    /// routing. Subsequent calls to [`Self::step`] / the substep loop
    /// route contact-solve through this bridge instead of the CPU
    /// solver. Pass `Some(bridge)` to install; use
    /// [`Self::take_gpu_solver_bridge`] to remove.
    ///
    /// The bridge is stored as `Box<dyn GpuSolverBridge + Send + Sync>`,
    /// so implementers must satisfy `Send + Sync`. For ALICE-TRT v3.0.0+
    /// this is guaranteed because `TrtSolverAdapter` uses
    /// `Arc<GpuDevice>` and holds only `Send + Sync` fields.
    ///
    /// Callers who want explicit control without installing the bridge
    /// on the world (game-engine wrapper hosts, hot-swap harnesses)
    /// can continue to use [`Self::step_with_bridge`] /
    /// [`Self::substep_with_bridge`] /
    /// [`Self::solve_contact_constraints_with_bridge`] with an
    /// externally-owned bridge — those methods ignore the installed
    /// bridge and take a `&mut B` parameter.
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn set_gpu_solver_bridge(
        &mut self,
        bridge: Option<Box<dyn crate::gpu_bridge::GpuSolverBridge + Send + Sync>>,
    ) {
        self.gpu_solver_bridge = bridge;
    }

    /// v0.11.0: remove and return the currently installed GPU solver
    /// bridge, if any. Subsequent calls to [`Self::step`] /
    /// the substep loop revert to the CPU contact solver.
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn take_gpu_solver_bridge(
        &mut self,
    ) -> Option<Box<dyn crate::gpu_bridge::GpuSolverBridge + Send + Sync>> {
        self.gpu_solver_bridge.take()
    }

    /// v0.11.0: whether a GPU solver bridge is currently installed on
    /// this world. Cheap; does not touch the bridge.
    #[cfg(feature = "gpu-solver-bridge")]
    #[must_use]
    pub const fn gpu_solver_bridge_installed(&self) -> bool {
        self.gpu_solver_bridge.is_some()
    }

    /// Bring `body_stable_ids` to `bodies.len()`: drop ids past the end and
    /// give every body without one a fresh id. Bodies keep their id for as
    /// long as they live; ids are never reused.
    #[cfg(feature = "std")]
    fn sync_body_stable_ids(&mut self) {
        let n = self.bodies.len();
        self.body_stable_ids.truncate(n);
        while self.body_stable_ids.len() < n {
            self.body_stable_ids.push(self.next_body_stable_id);
            self.next_body_stable_id += 1;
        }
    }

    /// Add rigid body, returns index
    ///
    /// The body's rotations are stored as unit quaternions (see
    /// [`RigidBody::set_rotation`]).
    pub fn add_body(&mut self, mut body: RigidBody) -> usize {
        body.make_rotations_unit();
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        self.body_collision_radii.push(None);
        self.body_colliders.push(None);
        self.body_filters.push(CollisionFilter::DEFAULT);
        self.islands.resize(idx + 1);
        #[cfg(feature = "std")]
        self.sync_body_stable_ids();
        idx
    }

    /// Add rigid body with a collision sphere radius for automatic detection.
    ///
    /// Bodies with a collision radius participate in BVH broad-phase and
    /// sphere-sphere narrow-phase collision detection during `step()`.
    ///
    /// The body's rotations are stored as unit quaternions, as in
    /// [`Self::add_body`].
    pub fn add_body_with_radius(&mut self, mut body: RigidBody, radius: Fix128) -> usize {
        body.make_rotations_unit();
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        self.body_collision_radii.push(Some(radius));
        self.body_colliders.push(None);
        self.body_filters.push(CollisionFilter::DEFAULT);
        self.islands.resize(idx + 1);
        #[cfg(feature = "std")]
        self.sync_body_stable_ids();
        idx
    }

    /// Add a rigid body shaped like `shape`, made of material of `density`, with
    /// its centre of mass at `position`.
    ///
    /// The body gets the shape's mass (`density × volume`), the shape's principal
    /// inertia about the centre of mass (not the unit-sphere inertia
    /// [`RigidBody::new`] assigns), and the shape's bounding sphere about the
    /// centre of mass as its collision radius, so it takes part in the sphere
    /// broad- and narrow-phase like any body added with
    /// [`Self::add_body_with_radius`]. See [`crate::shape`] for the frames.
    ///
    /// # Errors
    ///
    /// [`ShapeError`](crate::shape::ShapeError) when the density is not positive,
    /// a dimension is not, or the mass or inertia does not fit `Fix128`. Nothing is
    /// added in that case.
    pub fn add_shaped_body(
        &mut self,
        shape: &crate::shape::Shape,
        density: Fix128,
        position: Vec3Fix,
    ) -> Result<usize, crate::shape::ShapeError> {
        let (mass, inertia) = shape.mass_and_inertia(density)?;
        let mut body = RigidBody::new_dynamic(position, mass);
        body.inv_inertia = Vec3Fix::new(
            Fix128::ONE / inertia.x,
            Fix128::ONE / inertia.y,
            Fix128::ONE / inertia.z,
        );
        let idx = self.add_body_with_radius(body, shape.bounding_radius());
        self.body_colliders[idx] = Some(crate::body_collider::BodyCollider::Shape(*shape));
        Ok(idx)
    }

    /// Give an existing body a convex `shape` as its collider: its collision radius
    /// becomes the shape's bounding radius about the centre of mass, and a pair of
    /// bodies of which either carries a shape is decided by the shape
    /// ([`crate::collider::contact`]; against a body without one, by its collision
    /// sphere) instead of by the bounding spheres. Returns `false` for an out-of-range index.
    ///
    /// The body's position is taken as the shape's **centre of mass** (see
    /// [`crate::shape`]); its mass and inertia are not changed — use
    /// [`Self::add_shaped_body`] for a body whose mass properties come from the shape.
    pub fn set_body_shape(&mut self, body_idx: usize, shape: &crate::shape::Shape) -> bool {
        if body_idx >= self.bodies.len() || body_idx >= self.body_colliders.len() {
            return false;
        }
        self.park_generation = self.park_generation.wrapping_add(1);
        self.body_collision_radii[body_idx] = Some(shape.bounding_radius());
        self.body_colliders[body_idx] = Some(crate::body_collider::BodyCollider::Shape(*shape));
        true
    }

    /// Whether the colliders of two bodies overlap, the same way the contact
    /// generation decides it: GJK on their shapes when both carry one
    /// ([`Self::set_body_shape`], [`Self::add_shaped_body`]), GJK on the shape and
    /// the other body's collision sphere when one does, otherwise the test of
    /// their collision spheres. `false` for an index that is not a body or a body
    /// without a collider. Touching exactly counts as overlapping when a shape is
    /// involved (GJK's convention) and as clear for two spheres (the contact
    /// generation's).
    #[must_use]
    pub fn colliders_overlap(&self, a: usize, b: usize) -> bool {
        let (Some(body_a), Some(body_b)) = (self.bodies.get(a), self.bodies.get(b)) else {
            return false;
        };
        let collider = |i: usize| self.body_colliders.get(i).and_then(Option::as_ref);
        let radius = |i: usize| self.body_collision_radii.get(i).copied().flatten();
        let pose_a = (body_a.position, body_a.rotation);
        let pose_b = (body_b.position, body_b.rotation);
        match (collider(a), collider(b)) {
            (Some(ca), Some(cb)) => crate::body_collider::colliders_meet(ca, pose_a, cb, pose_b),
            (Some(ca), None) => radius(b).is_some_and(|rb| {
                crate::body_collider::collider_meets_sphere(
                    ca,
                    pose_a,
                    crate::collider::Sphere::new(body_b.position, rb),
                )
            }),
            (None, Some(cb)) => radius(a).is_some_and(|ra| {
                crate::body_collider::collider_meets_sphere(
                    cb,
                    pose_b,
                    crate::collider::Sphere::new(body_a.position, ra),
                )
            }),
            (None, None) => match (radius(a), radius(b)) {
                (Some(ra), Some(rb)) => {
                    let reach = ra + rb;
                    (body_a.position - body_b.position).length_squared() < reach * reach
                }
                _ => false,
            },
        }
    }

    /// Add a rigid body made of a [`CompoundShape`](crate::compound::CompoundShape):
    /// material of `density`, the compound's centre of mass at `position`.
    ///
    /// The mass is the children's total and the inertia is their summed tensor about
    /// the common centre of mass, which for an asymmetric compound has products of
    /// inertia; the body stores the **principal** moments, so its local frame is the
    /// principal frame, and its initial rotation is the rotation from that frame to
    /// the compound's as authored — the compound appears in the world exactly as it
    /// was built, translated so its centre of mass is at `position`. The body
    /// collides as its children (`body_collider`, internal): a gap between two
    /// children is not part of the body, and its collision radius is the sphere about
    /// the centre of mass that contains them all.
    ///
    /// # Errors
    ///
    /// [`ShapeError`](crate::shape::ShapeError): `NonPositiveDensity`;
    /// `DegenerateShape` for an empty compound or one with no volume;
    /// `MassNotRepresentable` when the mass or a principal moment is not positive or
    /// the sizes are too large for `Fix128`. Nothing is added in those cases.
    pub fn add_compound_body(
        &mut self,
        compound: &crate::compound::CompoundShape,
        density: Fix128,
        position: Vec3Fix,
    ) -> Result<usize, crate::shape::ShapeError> {
        use crate::shape::ShapeError;
        if density <= Fix128::ZERO {
            return Err(ShapeError::NonPositiveDensity);
        }
        if compound.is_empty() {
            return Err(ShapeError::DegenerateShape);
        }
        let props = compound.mass_properties(density);
        if props.mass <= Fix128::ZERO {
            return Err(ShapeError::DegenerateShape);
        }
        let (moments, axes) = crate::mass_properties::principal_axes(props.inertia_tensor);
        if !(moments.x > Fix128::ZERO && moments.y > Fix128::ZERO && moments.z > Fix128::ZERO) {
            return Err(ShapeError::MassNotRepresentable);
        }
        // The compound in the principal frame, centre of mass at the origin.
        let frame = crate::body_collider::quat_from_rotation(axes);
        let to_principal = frame.conjugate();
        let mut stored = crate::compound::CompoundShape::new();
        for child in &compound.children {
            stored.children.push(crate::compound::CompoundChild {
                shape: child.shape.clone(),
                local_position: to_principal
                    .rotate_vec(child.local_position - props.center_of_mass),
                local_rotation: to_principal.mul(child.local_rotation),
            });
        }
        let collider = crate::body_collider::BodyCollider::Compound(stored);
        let radius = collider.bounding_radius();
        // The largest quantity the solve forms, as an `f64`, before it is formed.
        let estimate = props.mass.to_f64() * radius.to_f64() * radius.to_f64();
        if !(estimate.is_finite() && estimate < (1u64 << 61) as f64) {
            return Err(ShapeError::MassNotRepresentable);
        }
        let mut body = RigidBody::new_dynamic(position, props.mass);
        body.rotation = frame;
        body.inv_inertia = Vec3Fix::new(
            Fix128::ONE / moments.x,
            Fix128::ONE / moments.y,
            Fix128::ONE / moments.z,
        );
        let idx = self.add_body_with_radius(body, radius);
        self.body_colliders[idx] = Some(collider);
        Ok(idx)
    }

    /// Extend the per-body side tables with the defaults `add_body` gives, up
    /// to `bodies.len()` (bodies pushed onto the public field have none).
    fn pad_body_tables(&mut self) {
        let n = self.bodies.len();
        if self.body_materials.len() < n {
            self.body_materials
                .resize(n, crate::material::DEFAULT_MATERIAL);
        }
        if self.body_collision_radii.len() < n {
            self.body_collision_radii.resize(n, None);
        }
        if self.body_colliders.len() < n {
            self.body_colliders.resize(n, None);
        }
        if self.body_filters.len() < n {
            self.body_filters.resize(n, CollisionFilter::DEFAULT);
        }
    }

    /// [`Self::remove_body`] that also returns the body's per-body settings
    /// (material, collision radius, collider, collision filter), for moving it
    /// into another world with [`Self::add_body_with_attachments`]. The stable
    /// id stays behind: ids belong to a world.
    pub(crate) fn remove_body_with_attachments(
        &mut self,
        idx: usize,
    ) -> Option<(RigidBody, BodyAttachments)> {
        // a body pushed onto the public `bodies` field may have no entries yet:
        // it carries the defaults `add_body` would give it
        let attachments = BodyAttachments {
            material: self
                .body_materials
                .get(idx)
                .copied()
                .unwrap_or(crate::material::DEFAULT_MATERIAL),
            collision_radius: self.body_collision_radii.get(idx).copied().flatten(),
            collider: self.body_colliders.get(idx).cloned().flatten(),
            filter: self.body_filters.get(idx).copied().unwrap_or_default(),
        };
        let body = self.remove_body(idx)?;
        Some((body, attachments))
    }

    /// [`Self::add_body`] with the per-body settings taken by
    /// [`Self::remove_body_with_attachments`]; the body gets a new stable id.
    pub(crate) fn add_body_with_attachments(
        &mut self,
        body: RigidBody,
        attachments: BodyAttachments,
    ) -> usize {
        let idx = self.add_body(body);
        self.body_materials[idx] = attachments.material;
        self.body_collision_radii[idx] = attachments.collision_radius;
        self.body_colliders[idx] = attachments.collider;
        self.body_filters[idx] = attachments.filter;
        idx
    }

    /// Remove a body by index (swap-remove).
    ///
    /// The last body is moved to fill the gap. All constraints and joints
    /// referencing the old last index are remapped. Returns the removed body,
    /// or `None` if the index is out of bounds.
    pub fn remove_body(&mut self, idx: usize) -> Option<RigidBody> {
        if idx >= self.bodies.len() {
            return None;
        }
        let last = self.bodies.len() - 1;

        // 1. Drop every constraint / joint that references the body being removed.
        //    This must happen BEFORE the `last -> idx` remap: after `swap_remove`
        //    the moved last body occupies `idx`, so a constraint that still says
        //    `idx` would silently re-attach to the wrong body (2026-09-15 bug fix,
        //    found by the remove_body mutation tests; v1.1.0 and earlier produced
        //    `(idx, idx)` self-constraints for the swapped-in body).
        self.distance_constraints
            .retain(|c| c.body_a != idx && c.body_b != idx);
        self.contact_constraints
            .retain(|c| c.body_a != idx && c.body_b != idx);
        if !self.joint_motors.is_empty() || !self.joint_motors_3d.is_empty() {
            // Motors follow their joint: new index of each kept joint, `None`
            // for a joint dropped with the body.
            let mut next = 0;
            let joint_map: Vec<Option<usize>> = self
                .joints
                .iter()
                .map(|j| {
                    let (a, b) = j.bodies();
                    if a != idx && b != idx {
                        next += 1;
                        Some(next - 1)
                    } else {
                        None
                    }
                })
                .collect();
            self.remap_motor_joints(|j| joint_map.get(j).copied().flatten());
        }
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != idx && b != idx
        });

        // Bring the ids level with `bodies` before both lose `idx` together
        // (a body pushed onto the public field has no id yet).
        #[cfg(feature = "std")]
        self.sync_body_stable_ids();
        // a body pushed onto the public `bodies` field has no side-table
        // entries yet: give it the defaults so the tables swap-remove in step
        self.pad_body_tables();
        let removed = self.bodies.swap_remove(idx);
        self.body_materials.swap_remove(idx);
        self.body_collision_radii.swap_remove(idx);
        self.body_colliders.swap_remove(idx);
        self.body_filters.swap_remove(idx);
        #[cfg(feature = "std")]
        self.body_stable_ids.swap_remove(idx);

        // 2. Remap references from `last` -> `idx` in all remaining constraints and joints
        if idx != last {
            for c in &mut self.distance_constraints {
                if c.body_a == last {
                    c.body_a = idx;
                }
                if c.body_b == last {
                    c.body_b = idx;
                }
            }
            for c in &mut self.contact_constraints {
                if c.body_a == last {
                    c.body_a = idx;
                }
                if c.body_b == last {
                    c.body_b = idx;
                }
            }
            self.remap_joint_indices(last, idx);
        }

        // Rebuild IslandManager to match new body count and connectivity
        let new_len = self.bodies.len();
        self.islands = IslandManager::new(new_len, self.islands.config);
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < new_len && b < new_len {
                self.islands.union(a, b);
            }
        }

        self.park_generation = self.park_generation.wrapping_add(1);
        self.batches_dirty = true;
        Some(removed)
    }

    /// Remap joint body indices after swap-remove
    fn remap_joint_indices(&mut self, from: usize, to: usize) {
        /// Remap a single joint's `body_a` and `body_b` fields.
        macro_rules! remap {
            ($j:expr) => {
                if $j.body_a == from {
                    $j.body_a = to;
                }
                if $j.body_b == from {
                    $j.body_b = to;
                }
            };
        }
        for joint in &mut self.joints {
            match joint {
                Joint::Ball(j) => {
                    remap!(j);
                }
                Joint::Hinge(j) => {
                    remap!(j);
                }
                Joint::Fixed(j) => {
                    remap!(j);
                }
                Joint::Slider(j) => {
                    remap!(j);
                }
                Joint::Spring(j) => {
                    remap!(j);
                }
                Joint::D6(j) => {
                    remap!(j);
                }
                Joint::ConeTwist(j) => {
                    remap!(j);
                }
            }
        }
    }

    /// Number of bodies in the world
    #[inline]
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of active (non-sleeping) bodies
    #[must_use]
    pub fn active_body_count(&self) -> usize {
        self.bodies.len() - self.islands.sleeping_count().min(self.bodies.len())
    }

    /// Set a body's material ID
    pub fn set_body_material(&mut self, body_idx: usize, material_id: crate::material::MaterialId) {
        if body_idx < self.body_materials.len() {
            self.body_materials[body_idx] = material_id;
        }
    }

    /// Add a pre-solve contact hook
    ///
    /// The hook is called with (`body_a_index`, `body_b_index`, &contact).
    /// Return `false` to skip the contact (e.g., for one-way platforms).
    #[cfg(feature = "std")]
    pub fn add_pre_solve_hook(&mut self, hook: PreSolveHook) {
        self.pre_solve_hooks.push(hook);
    }

    /// Clear all pre-solve hooks
    #[cfg(feature = "std")]
    pub fn clear_pre_solve_hooks(&mut self) {
        self.pre_solve_hooks.clear();
    }

    /// Add a contact modifier
    ///
    /// Contact modifiers can mutate contact properties (normal, depth, friction,
    /// restitution) before solving. Return false from `modify_contact` to discard.
    #[cfg(feature = "std")]
    pub fn add_contact_modifier(&mut self, modifier: Box<dyn ContactModifier>) {
        self.contact_modifiers.push(modifier);
    }

    /// Clear all contact modifiers
    #[cfg(feature = "std")]
    pub fn clear_contact_modifiers(&mut self) {
        self.contact_modifiers.clear();
    }

    /// Begin a new simulation frame (updates contact cache lifecycle)
    pub fn begin_frame(&mut self) {
        self.contact_cache.begin_frame();
    }

    /// End a simulation frame (prune old contacts from cache)
    pub fn end_frame(&mut self) {
        self.contact_cache.end_frame();
    }

    // ── Joint Management ───────────────────────────────────────────────

    /// Add a joint constraint, returns joint index.
    ///
    /// # Panics
    ///
    /// Panics if either body index in the joint is out of bounds.
    pub fn add_joint(&mut self, joint: Joint) -> usize {
        let (a, b) = joint.bodies();
        assert!(
            a < self.bodies.len() && b < self.bodies.len(),
            "Joint body indices ({}, {}) out of bounds (body count = {})",
            a,
            b,
            self.bodies.len()
        );
        let idx = self.joints.len();
        self.islands.resize(self.bodies.len());
        self.islands.union(a, b);
        self.joints.push(joint);
        idx
    }

    /// Remove a joint by index (swap-remove), returns the removed joint
    ///
    /// Motors follow their joint: a motor on the removed joint is removed with
    /// it, and a motor on the last joint (moved into `idx`) is re-pointed to
    /// `idx`.
    pub fn remove_joint(&mut self, idx: usize) -> Option<Joint> {
        if idx >= self.joints.len() {
            return None;
        }
        let last = self.joints.len() - 1;
        self.remap_motor_joints(|j| {
            if j == idx {
                None
            } else if j == last {
                Some(idx)
            } else {
                Some(j)
            }
        });
        Some(self.joints.swap_remove(idx))
    }

    /// Re-point (or drop, on `None`) every motor's joint index through `map`.
    fn remap_motor_joints(&mut self, map: impl Fn(usize) -> Option<usize>) {
        self.joint_motors.retain_mut(|m| match map(m.joint_index) {
            Some(j) => {
                m.joint_index = j;
                true
            }
            None => false,
        });
        self.joint_motors_3d
            .retain_mut(|(joint, _)| match map(*joint) {
                Some(j) => {
                    *joint = j;
                    true
                }
                None => false,
            });
    }

    // ── Joint motors ──────────────────────────────────────────────────

    /// Attach a one-axis PD motor to joint `joint_index`, returns the motor
    /// index.
    ///
    /// The motor is applied by [`Self::step`] every substep (see
    /// [`crate::motor::apply_motors`] for the generalised coordinate per joint
    /// type: the twist angle for a hinge, the centre distance otherwise). The
    /// controller is used as given, so a controller still in
    /// [`crate::motor::MotorMode::Off`] does nothing until a target is set
    /// ([`Self::set_joint_motor_velocity_target`] / [`Self::joint_motor_mut`]).
    /// Motors follow their joint through [`Self::remove_joint`] and
    /// [`Self::remove_body`]; a motor whose index no longer names a joint
    /// (after editing the public `joints` field directly) does nothing.
    pub fn add_joint_motor(
        &mut self,
        joint_index: usize,
        controller: crate::motor::PdController,
    ) -> usize {
        self.joint_motors
            .push(crate::motor::JointMotor::new(joint_index, controller));
        self.joint_motors.len() - 1
    }

    /// The motor added as `motor` by [`Self::add_joint_motor`], for changing its
    /// gains or targets between steps.
    #[must_use]
    pub fn joint_motor_mut(&mut self, motor: usize) -> Option<&mut crate::motor::JointMotor> {
        self.joint_motors.get_mut(motor)
    }

    /// Switch motor `motor` to velocity mode with `target` (rad/s for a hinge,
    /// m/s along the centre line otherwise). Returns `false` if there is no
    /// such motor.
    pub fn set_joint_motor_velocity_target(&mut self, motor: usize, target: Fix128) -> bool {
        match self.joint_motors.get_mut(motor) {
            Some(m) => {
                m.controller.set_velocity_target(target);
                true
            }
            None => false,
        }
    }

    /// Turn motor `motor` off (it applies no force until a target is set
    /// again). Returns `false` if there is no such motor.
    pub fn disable_joint_motor(&mut self, motor: usize) -> bool {
        match self.joint_motors.get_mut(motor) {
            Some(m) => {
                m.controller.disable();
                true
            }
            None => false,
        }
    }

    /// Attach a three-axis rotation motor with per-axis gains `kp` / `kd` and
    /// torque cap `max_torque` to joint `joint_index`, returns its index.
    ///
    /// The motor starts `Off`; [`Self::set_joint_motor_3d_rotation_target`]
    /// drives the joint's relative rotation `rotation_b * rotation_a⁻¹` to a
    /// target (see [`crate::motor::PdController3D::compute_torque`]). It
    /// follows its joint through removals like [`Self::add_joint_motor`].
    pub fn add_joint_motor_3d(
        &mut self,
        joint_index: usize,
        kp: Vec3Fix,
        kd: Vec3Fix,
        max_torque: Fix128,
    ) -> usize {
        self.joint_motors_3d.push((
            joint_index,
            crate::motor::PdController3D::new(kp, kd, max_torque),
        ));
        self.joint_motors_3d.len() - 1
    }

    /// The controller of rotation motor `motor` ([`Self::add_joint_motor_3d`]).
    #[must_use]
    pub fn joint_motor_3d_mut(
        &mut self,
        motor: usize,
    ) -> Option<&mut crate::motor::PdController3D> {
        self.joint_motors_3d.get_mut(motor).map(|(_, c)| c)
    }

    /// Drive rotation motor `motor` to the relative rotation `target`
    /// (position mode). Returns `false` if there is no such motor.
    pub fn set_joint_motor_3d_rotation_target(&mut self, motor: usize, target: QuatFix) -> bool {
        match self.joint_motors_3d.get_mut(motor) {
            Some((_, c)) => {
                c.set_rotation_target(target);
                true
            }
            None => false,
        }
    }

    /// Apply every joint motor for one (sub)step of length `dt`.
    ///
    /// A sleeping body a motor moves is woken with its island, otherwise the
    /// motor's velocity change would be discarded by the sleep skip in
    /// `integrate_positions`. A motor whose output is zero does not wake
    /// anything, so a motor resting at its target lets the joint sleep.
    fn apply_joint_motors(&mut self, dt: Fix128) {
        if self.joint_motors.is_empty() && self.joint_motors_3d.is_empty() {
            return;
        }
        let mut asleep: Vec<(usize, Vec3Fix, Vec3Fix)> = Vec::new();
        let n = self.bodies.len();
        let pairs = self
            .joint_motors
            .iter()
            .map(|m| m.joint_index)
            .chain(self.joint_motors_3d.iter().map(|(j, _)| *j));
        for j in pairs {
            if let Some(joint) = self.joints.get(j) {
                let (a, b) = joint.bodies();
                for x in [a, b] {
                    if x < n && self.islands.is_sleeping(x) {
                        asleep.push((x, self.bodies[x].velocity, self.bodies[x].angular_velocity));
                    }
                }
            }
        }
        crate::motor::apply_motors(&self.joint_motors, &self.joints, &mut self.bodies, dt);
        crate::motor::apply_motors_3d(&self.joint_motors_3d, &self.joints, &mut self.bodies, dt);
        for (x, v, w) in asleep {
            let body = &self.bodies[x];
            if (body.velocity != v || body.angular_velocity != w)
                && x < self.islands.sleep_data.len()
            {
                self.islands.wake_island(x);
            }
        }
    }

    /// Number of joints
    #[inline]
    #[must_use]
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    // ── Force Field Management ────────────────────────────────────────

    /// Add a force field, returns field index
    pub fn add_force_field(&mut self, field: ForceFieldInstance) -> usize {
        let idx = self.force_fields.len();
        self.force_fields.push(field);
        idx
    }

    /// Remove a force field by index
    pub fn remove_force_field(&mut self, idx: usize) -> Option<ForceFieldInstance> {
        if idx >= self.force_fields.len() {
            return None;
        }
        Some(self.force_fields.swap_remove(idx))
    }

    // ── Per-Body Collision Shape / Filter ──────────────────────────────

    /// Set collision radius for a body (enables automatic sphere collision detection)
    pub fn set_body_collision_radius(&mut self, body_idx: usize, radius: Fix128) {
        if body_idx < self.body_collision_radii.len() {
            self.park_generation = self.park_generation.wrapping_add(1);
            self.body_collision_radii[body_idx] = Some(radius);
        }
    }

    /// Remove collision radius (disable automatic collision detection for this body)
    pub fn clear_body_collision_radius(&mut self, body_idx: usize) {
        if body_idx < self.body_collision_radii.len() {
            self.park_generation = self.park_generation.wrapping_add(1);
            self.body_collision_radii[body_idx] = None;
        }
    }

    /// Set collision filter for a body
    pub fn set_body_filter(&mut self, body_idx: usize, filter: CollisionFilter) {
        if body_idx < self.body_filters.len() {
            self.body_filters[body_idx] = filter;
        }
    }

    /// Get collision filter for a body
    #[must_use]
    pub fn body_filter(&self, body_idx: usize) -> CollisionFilter {
        self.body_filters
            .get(body_idx)
            .copied()
            .unwrap_or(CollisionFilter::DEFAULT)
    }

    // ── Sleeping ──────────────────────────────────────────────────────

    /// Check if a body is sleeping
    #[inline]
    #[must_use]
    pub fn is_sleeping(&self, body_idx: usize) -> bool {
        self.islands.is_sleeping(body_idx)
    }

    /// Wake up a body and all connected bodies in its island
    pub fn wake_body(&mut self, body_idx: usize) {
        // An index past the last body is ignored, like every other per-body
        // setter; the island union-find would index out of range otherwise.
        if body_idx >= self.bodies.len() {
            return;
        }
        self.islands.wake_island(body_idx);
    }

    /// Set the sleep configuration
    pub fn set_sleep_config(&mut self, config: SleepConfig) {
        self.islands.config = config;
    }

    /// Turn the sleep skip of [`Self::step`] on (the default) or off.
    ///
    /// With it on, a sleeping body at rest that no joint or distance constraint
    /// references is left out of every stage of the step, and its contacts are
    /// found through a persistent tree of such bodies instead of rebuilding the
    /// broad-phase over it (see [`StageWork`]). The simulation is bit-identical
    /// either way: the skip only drops work whose result is already known (a body
    /// at rest stays where it is, with the velocity it already has). Turning it off
    /// drops the tree; it is rebuilt on the next step with the skip on.
    ///
    /// Only `step` with the XPBD backend and [`Broadphase::Bvh`] skips; the other
    /// entry points and broad-phases ([`Broadphase::DynamicTree`],
    /// [`Broadphase::Hybrid`]) visit every body as before.
    pub fn set_sleep_skip(&mut self, enabled: bool) {
        self.sleep_skip = enabled;
        if !enabled {
            self.park.clear();
        }
    }

    /// Whether the sleep skip of [`Self::step`] is on (see [`Self::set_sleep_skip`]).
    #[must_use]
    pub const fn sleep_skip(&self) -> bool {
        self.sleep_skip
    }

    /// Work counters of the last [`Self::step`] (see [`StageWork`]).
    #[must_use]
    pub const fn stage_work(&self) -> StageWork {
        self.stage_work
    }

    // ── Typed Observation (World Auditor WM-10, Physics 版) ────────────

    /// Observe a single body's law-relevant state.
    ///
    /// Returns `None` if `body_idx` is out of range (there is no body at
    /// that index — note that [`Self::remove_body`] uses `swap_remove`, so
    /// `body_idx` is only stable within the frame it was observed).
    ///
    /// ⚠️ **`grounded` は意図的に持たない** — 一般 [`RigidBody`] に地面接触の
    /// 定義は存在せず (`character.rs` の `CharacterController::detect_ground`
    /// は特化アルゴリズムで、全 body に適用できる基準ではない) Law / goal
    /// 述語が「接地」を必要とする場合は `in_contact` + 接触法線
    /// ([`EventCollector::contact_events`] の [`crate::event::ContactEvent::normal`]) から
    /// 自前で構成する
    #[must_use]
    pub fn observe_body(&self, body_idx: usize) -> Option<BodyObservation> {
        let body = self.bodies.get(body_idx)?;
        let in_contact = self
            .events
            .contact_events()
            .iter()
            .any(|e| e.body_a == body_idx || e.body_b == body_idx);
        Some(BodyObservation {
            body_index: body_idx,
            position: body.position,
            velocity: body.velocity,
            rotation: body.rotation,
            angular_velocity: body.angular_velocity,
            sleeping: self.is_sleeping(body_idx),
            in_contact,
        })
    }

    /// Observe every body in the world, in body index order.
    #[must_use]
    pub fn observe_bodies(&self) -> Vec<BodyObservation> {
        (0..self.bodies.len())
            .map(|i| {
                self.observe_body(i)
                    .expect("index came from 0..bodies.len(), always in range")
            })
            .collect()
    }

    // ── Event Access ──────────────────────────────────────────────────

    /// Get contact events from the last step
    #[inline]
    #[must_use]
    pub fn contact_events(&self) -> &[crate::event::ContactEvent] {
        self.events.contact_events()
    }

    /// Get trigger events from the last step
    #[inline]
    #[must_use]
    pub fn trigger_events(&self) -> &[crate::event::TriggerEvent] {
        self.events.trigger_events()
    }

    /// Drain (consume) contact events from the last step
    #[inline]
    pub fn drain_contact_events(&mut self) -> Vec<crate::event::ContactEvent> {
        self.events.drain_contact_events()
    }

    /// Drain (consume) trigger events from the last step
    #[inline]
    pub fn drain_trigger_events(&mut self) -> Vec<crate::event::TriggerEvent> {
        self.events.drain_trigger_events()
    }

    // ── Raycast on World ──────────────────────────────────────────────

    /// What a ray query tests for body `i`: its collision radius and the shape or
    /// compound it carries, or `None` when the body has no collision radius (it
    /// takes part in no collision, so no ray sees it either).
    pub(crate) fn ray_geometry(
        &self,
        i: usize,
    ) -> Option<(Fix128, Option<&crate::body_collider::BodyCollider>)> {
        let radius = self.body_collision_radii.get(i).copied().flatten()?;
        Some((radius, self.body_colliders.get(i).and_then(Option::as_ref)))
    }

    /// The static colliders, in index order.
    pub(crate) fn static_colliders_slice(&self) -> &[crate::static_collider::StaticCollider] {
        &self.static_colliders
    }

    /// The normal force of one contact of the last step (see
    /// [`contact_forces`](Self::contact_forces)), or `None` when neither body can move.
    pub(crate) fn contact_constraint_force(
        &self,
        c: &ContactConstraint,
        dt: Fix128,
    ) -> Option<Fix128> {
        let h = self.substep_dt(dt);
        self.contact_force(c, h * h).map(|(_, _, force)| force)
    }

    /// Cast a ray against all body collision spheres, returns (`body_index`, distance).
    ///
    /// Uses a BVH broad-phase to cull bodies outside the ray's bounding box,
    /// then performs exact ray-sphere intersection on candidates.
    /// Returns `None` if direction is zero or no body is hit.
    ///
    /// Every body is tested as its **bounding sphere** (its collision radius), even
    /// when it carries a shape or a compound, and static colliders and SDF
    /// colliders are not tested. This differs from the contacts, which a shaped
    /// body makes with its shape: a ray can hit the bounding sphere of a body
    /// that nothing touches there. [`Self::cast_ray`] tests the actual geometry
    /// (see [`crate::shape_raycast`]).
    #[must_use]
    pub fn raycast(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_distance: Fix128,
    ) -> Option<(usize, Fix128)> {
        if direction.length_squared().is_zero() {
            return None;
        }
        let dir_norm = direction.normalize();

        // Build BVH from collidable bodies for broad-phase culling
        let mut primitives = Vec::new();
        for i in 0..self.bodies.len() {
            if let Some(radius) = self.body_collision_radii.get(i).and_then(|r| *r) {
                let pos = self.bodies[i].position;
                let half = Vec3Fix::new(radius, radius, radius);
                let aabb = crate::collider::AABB::from_center_half(pos, half);
                primitives.push(crate::bvh::BvhPrimitive {
                    aabb,
                    index: i as u32,
                    morton: 0,
                });
            }
        }

        if primitives.is_empty() {
            return None;
        }

        // Compute ray AABB (bounding box of the ray segment)
        let endpoint = origin + dir_norm * max_distance;
        let ray_min = Vec3Fix::new(
            if origin.x < endpoint.x {
                origin.x
            } else {
                endpoint.x
            },
            if origin.y < endpoint.y {
                origin.y
            } else {
                endpoint.y
            },
            if origin.z < endpoint.z {
                origin.z
            } else {
                endpoint.z
            },
        );
        let ray_max = Vec3Fix::new(
            if origin.x > endpoint.x {
                origin.x
            } else {
                endpoint.x
            },
            if origin.y > endpoint.y {
                origin.y
            } else {
                endpoint.y
            },
            if origin.z > endpoint.z {
                origin.z
            } else {
                endpoint.z
            },
        );
        let ray_aabb = crate::collider::AABB {
            min: ray_min,
            max: ray_max,
        };

        let bvh = crate::bvh::LinearBvh::build(primitives);
        let candidates = bvh.query(&ray_aabb);

        // Narrow-phase: exact ray-sphere intersection on BVH candidates
        let mut best: Option<(usize, Fix128)> = None;
        for &prim_idx in &candidates {
            let i = prim_idx as usize;
            let radius = match self.body_collision_radii.get(i) {
                Some(Some(r)) => *r,
                _ => continue,
            };
            let oc = origin - self.bodies[i].position;
            let b = oc.dot(dir_norm);
            let c = oc.dot(oc) - radius * radius;
            let discriminant = b * b - c;
            if discriminant < Fix128::ZERO {
                continue;
            }
            let sqrt_d = discriminant.sqrt();
            // Try the nearest intersection first
            let mut t = -b - sqrt_d;
            // If nearest t is behind origin, try the far intersection (ray inside sphere)
            if t < Fix128::ZERO {
                t = -b + sqrt_d;
            }
            if t < Fix128::ZERO || t > max_distance {
                continue;
            }
            let dominated = match best {
                None => true,
                Some((_, prev_t)) => t < prev_t,
            };
            if dominated {
                best = Some((i, t));
            }
        }
        best
    }

    /// Get combined material properties for a body pair
    ///
    /// # Claims
    ///
    /// - A body whose material is [`crate::material::DEFAULT_MATERIAL`] contributes
    ///   its own [`RigidBody::friction`] and [`RigidBody::restitution`] (the
    ///   default material is "no material assigned"); a body with any other
    ///   material contributes that material's `dynamic_friction` / `restitution`
    ///   and its own fields are not read.
    /// - Bodies built with the defaults (friction 0.3, restitution 0.5) give the
    ///   same result as before, since the default material holds the same values.
    /// - The two contributions are combined with the materials' combine rules; a
    ///   pair override registered for the two materials takes precedence over
    ///   both.
    /// - An index outside the body list counts as the default material with the
    ///   material's own values.
    #[must_use]
    pub fn combined_material(
        &self,
        body_a: usize,
        body_b: usize,
    ) -> crate::material::CombinedMaterial {
        let side = |i: usize| -> (crate::material::MaterialId, Option<(Fix128, Fix128)>) {
            let Some(&mat) = self.body_materials.get(i) else {
                return (crate::material::DEFAULT_MATERIAL, None);
            };
            if mat != crate::material::DEFAULT_MATERIAL {
                return (mat, None);
            }
            (mat, self.bodies.get(i).map(|b| (b.friction, b.restitution)))
        };
        let (mat_a, surface_a) = side(body_a);
        let (mat_b, surface_b) = side(body_b);
        self.material_table
            .combine_with_surface(mat_a, surface_a, mat_b, surface_b)
    }

    /// Add distance constraint
    pub fn add_distance_constraint(&mut self, constraint: DistanceConstraint) {
        self.distance_constraints.push(constraint);
        self.batches_dirty = true;
    }

    /// Add contact constraint
    pub fn add_contact(&mut self, contact: ContactConstraint) {
        // Update contact cache for warm starting
        let key = crate::contact_cache::BodyPairKey::new(contact.body_a, contact.body_b);
        let manifold = self
            .contact_cache
            .get_or_create(key, contact.friction, contact.restitution);
        manifold.add_or_update(
            &contact.contact,
            contact.contact.point_a,
            contact.contact.point_b,
        );

        self.contact_constraints.push(contact);
        self.batches_dirty = true;
    }

    /// Add contact constraint with material lookup
    ///
    /// Automatically looks up friction/restitution from the material table.
    pub fn add_contact_with_material(&mut self, body_a: usize, body_b: usize, contact: Contact) {
        let combined = self.combined_material(body_a, body_b);
        let constraint = ContactConstraint {
            body_a,
            body_b,
            contact,
            friction: combined.friction,
            restitution: combined.restitution,
            cached_lambda: Fix128::ZERO,
        };
        self.add_contact(constraint);
    }

    /// Clear all contact constraints. `step()` does this at the start of every
    /// substep (1.2.0) before `detect_collisions`, so contacts added manually
    /// between frames are not seen by `step()`; drive the substep API yourself
    /// if you generate contacts outside the built-in sphere / SDF detection.
    pub fn clear_contacts(&mut self) {
        self.contact_constraints.clear();
        #[cfg(feature = "std")]
        self.contact_discarded.clear();
        self.batches_dirty = true;
    }

    /// Rebuild constraint batches using greedy graph coloring
    ///
    /// Each batch contains constraints that share no bodies,
    /// allowing safe parallel modification.
    pub fn rebuild_batches(&mut self) {
        if !self.batches_dirty {
            return;
        }

        self.constraint_batches.clear();

        // Snapshot which bodies are static / kinematic. They are never written
        // by the pair solvers (every correction is multiplied by
        // `inv_mass == 0`), so the parallel path borrows them shared and they
        // impose no coloring constraint. Without this exclusion a single
        // static floor touched by N bodies would force N sequential batches.
        let num_bodies = self.bodies.len();
        self.batch_static_bodies.clear();
        self.batch_static_bodies
            .extend(self.bodies.iter().map(|b| b.inv_mass.is_zero()));

        // Track which colors each dynamic body participates in. One growable
        // bitset per body (64 colors per word) — no upper bound on colors, so a
        // hub body shared by 65+ constraints still gets a unique color per
        // constraint instead of aliasing into a single overflow batch.
        let mut body_colors: Vec<Vec<u64>> = vec![Vec::new(); num_bodies];

        // Color distance constraints
        for (constraint_idx, constraint) in self.distance_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            // Find first color where both bodies are free
            let color =
                Self::find_free_color(&body_colors, &self.batch_static_bodies, body_a, body_b);

            // Ensure we have enough batches
            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            // Add constraint to batch
            self.constraint_batches[color]
                .distance_indices
                .push(constraint_idx);

            // Mark bodies as used in this color (set bit)
            Self::mark_color(&mut body_colors, &self.batch_static_bodies, body_a, color);
            Self::mark_color(&mut body_colors, &self.batch_static_bodies, body_b, color);
        }

        // Color contact constraints
        for (constraint_idx, constraint) in self.contact_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            let color =
                Self::find_free_color(&body_colors, &self.batch_static_bodies, body_a, body_b);

            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            self.constraint_batches[color]
                .contact_indices
                .push(constraint_idx);

            Self::mark_color(&mut body_colors, &self.batch_static_bodies, body_a, color);
            Self::mark_color(&mut body_colors, &self.batch_static_bodies, body_b, color);
        }

        self.batches_dirty = false;

        debug_assert!(
            self.batches_are_body_disjoint(),
            "graph coloring produced a batch with two constraints sharing a dynamic body"
        );
    }

    /// Occupancy word `word` of body `idx`, or 0 if the body is static,
    /// out of range, or has no color at that word yet.
    #[inline(always)]
    fn color_word(
        body_colors: &[Vec<u64>],
        static_bodies: &[bool],
        idx: usize,
        word: usize,
    ) -> u64 {
        if static_bodies.get(idx).copied().unwrap_or(false) {
            return 0;
        }
        body_colors
            .get(idx)
            .and_then(|words| words.get(word))
            .copied()
            .unwrap_or(0)
    }

    /// Find first color where both bodies are free (greedy coloring).
    ///
    /// Scans the per-body occupancy bitsets word by word (64 colors per
    /// word); the first word with a free bit yields the color. If every
    /// allocated word is saturated the next fresh word is used, so the
    /// number of colors is unbounded and two constraints sharing a dynamic
    /// body never land in the same batch.
    fn find_free_color(
        body_colors: &[Vec<u64>],
        static_bodies: &[bool],
        body_a: usize,
        body_b: usize,
    ) -> usize {
        let words_a = body_colors.get(body_a).map_or(0, Vec::len);
        let words_b = body_colors.get(body_b).map_or(0, Vec::len);
        let words = words_a.max(words_b);

        for word in 0..words {
            let occupied = Self::color_word(body_colors, static_bodies, body_a, word)
                | Self::color_word(body_colors, static_bodies, body_b, word);
            if occupied != u64::MAX {
                return word * 64 + (!occupied).trailing_zeros() as usize;
            }
        }

        words * 64
    }

    /// Record that dynamic body `idx` participates in `color`.
    ///
    /// Static bodies and out-of-range indices are ignored (they are not
    /// coloring constraints). Grows the body's bitset on demand.
    #[inline(always)]
    fn mark_color(body_colors: &mut [Vec<u64>], static_bodies: &[bool], idx: usize, color: usize) {
        if static_bodies.get(idx).copied().unwrap_or(false) {
            return;
        }
        if let Some(words) = body_colors.get_mut(idx) {
            let word = color / 64;
            if words.len() <= word {
                words.resize(word + 1, 0);
            }
            words[word] |= 1u64 << (color % 64);
        }
    }

    /// Check the graph coloring invariant: within one batch no two
    /// constraints reference the same dynamic body (static bodies as
    /// recorded in `batch_static_bodies` may repeat, they are read-only).
    ///
    /// O(total constraint references) with a per-body scratch marker.
    /// Used by `rebuild_batches`' `debug_assert!` and by tests.
    pub(crate) fn batches_are_body_disjoint(&self) -> bool {
        let num_bodies = self.bodies.len();
        // usize::MAX = untouched; otherwise the batch index that last used it.
        let mut last_batch: Vec<usize> = vec![usize::MAX; num_bodies];

        for (batch_idx, batch) in self.constraint_batches.iter().enumerate() {
            let dist_bodies = batch.distance_indices.iter().map(|&i| {
                (
                    self.distance_constraints[i].body_a,
                    self.distance_constraints[i].body_b,
                )
            });
            let contact_bodies = batch.contact_indices.iter().map(|&i| {
                (
                    self.contact_constraints[i].body_a,
                    self.contact_constraints[i].body_b,
                )
            });

            for (a, b) in dist_bodies.chain(contact_bodies) {
                for (k, idx) in [a, b].into_iter().enumerate() {
                    // A constraint referencing the same body twice counts once.
                    if (k == 1 && a == b) || idx >= num_bodies || self.batch_static_bodies[idx] {
                        continue;
                    }
                    if last_batch[idx] == batch_idx {
                        return false;
                    }
                    last_batch[idx] = batch_idx;
                }
            }
        }
        true
    }

    /// Get number of constraint batches (colors used)
    #[must_use]
    pub fn num_batches(&self) -> usize {
        self.constraint_batches.len()
    }

    /// Step simulation by dt (in fixed-point seconds)
    ///
    /// Integrated pipeline:
    /// 1. Begin event frame
    /// 2. Apply force fields to body velocities
    /// 3. Detect collisions (BVH broad-phase + sphere narrow-phase)
    /// 4. Substep loop (joint motors, integrate, solve constraints + joints,
    ///    update velocities)
    /// 5. Update sleeping states
    /// 6. End event frame (generates end-of-contact events)
    ///
    /// Deterministic: same inputs always produce same outputs.
    ///
    /// # Limitation: joints
    ///
    /// - `step` solves the joints once per substep, after and outside the
    ///   `iterations` loop that solves distance and contact constraints.
    /// - Joint compliance is therefore exact XPBD with respect to substeps:
    ///   each substep's single solve starts from `λ = 0`, so a compliant joint
    ///   under a constant load `F` settles at the stretch `F · compliance`
    ///   whatever `iterations` is.
    /// - Convergence of a rigid joint chain depends on `substeps` only, not on
    ///   `iterations`.
    ///
    /// # Participants
    ///
    /// `step` is [`Self::try_step`] with the result dropped: while a fault is
    /// recorded ([`Self::fault`]) it leaves the world unchanged. Without
    /// registered participants no fault is ever recorded and `step` runs as
    /// it always has.
    pub fn step(&mut self, dt: Fix128) {
        let _ = self.try_step(dt);
    }

    /// Bring every body's rotations to unit length before a step reads them
    /// ([`RigidBody::make_rotations_unit`]). `rotation` is a public field, so
    /// a value assigned to it directly between steps reaches the step as it
    /// was written; a rotation already of unit length is left bit for bit.
    fn make_body_rotations_unit(&mut self) {
        for body in &mut self.bodies {
            body.make_rotations_unit();
        }
    }

    /// The step body behind [`Self::try_step`] (checks already done, `dt > 0`).
    fn run_step(&mut self, dt: Fix128) {
        self.make_body_rotations_unit();
        self.stage_work = StageWork::default();
        let mut frozen = self.participant_flags();

        // `SolverBackend::Tgs` dispatch (std-only, see `SolverBackend` doc for
        // the no-std fallback). This `matches!` is `false` for every caller
        // that leaves `config.solver_backend` at its `Default` (`Xpbd`), so
        // the rest of this function — the pre-1.1 XPBD body — is reached and
        // executed exactly as before: no new code runs on the default path.
        #[cfg(feature = "std")]
        if matches!(self.config.solver_backend, SolverBackend::Tgs) {
            self.step_tgs(dt, &mut frozen);
            return;
        }

        // Phase 0: Event frame lifecycle (contacts are cleared and re-detected
        // in every substep since 1.2.0, see `substep`)
        self.events.begin_frame();

        // Phase 0.5: Rebuild island connectivity from current joints
        self.islands.resize(self.bodies.len());
        self.islands.reset_unions();
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < self.bodies.len() && b < self.bodies.len() {
                self.islands.union(a, b);
            }
        }

        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        // Phase 0.75: Park the sleeping bodies at rest (sleep skip). Every
        // stage below iterates `park.awake` while `park.active` is set.
        self.park_begin(dt, substep_dt);

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            if self.park.active {
                for k in 0..self.park.awake.len() {
                    let i = self.park.awake[k];
                    let body = &mut self.bodies[i];
                    if body.is_static() {
                        continue;
                    }
                    self.stage_work.force_field_bodies += 1;
                    body.velocity =
                        crate::force::force_field_velocity(&self.force_fields, i, body, dt);
                }
            } else {
                self.stage_work.force_field_bodies += self.bodies.len() as u64;
                apply_force_fields(&self.force_fields, &mut self.bodies, dt);
            }
        }

        // Phase 2 (collision detection) moved into the substep — Small Steps
        // (Müller et al. 2020): a frame-level contact set solved 8× with a
        // stale depth re-pushed the penetration every substep and turned a
        // 5 m/s head-on collision into 700 m/s (1.2.0, R4-2).

        // Phase 3: Substep loop
        let n = self.config.substeps;
        for i in 0..n {
            self.kinematic_substeps_left = n - i;
            self.park.first_substep = i == 0;
            // Participants run before the substep body (not inside `substep`,
            // which the unit tests call on its own).
            let overflow_at_start = self.participants_begin_substep(i, n, substep_dt, &mut frozen);
            self.substep(substep_dt);
            self.participants_end_substep(overflow_at_start, &frozen);
        }
        self.kinematic_substeps_left = 0;
        self.park.first_substep = false;

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.update_sleep_states();
        self.park.active = false;

        // Phase 4.5: Leave every SDF collider attached to a body at that
        // body's final pose, which is what the queries between steps
        // (`sdf_contacts`, `sdf_ccd_hits`, the shape casts) read.
        self.sync_sdf_colliders();

        // Phase 5: End event frame
        self.events.end_frame();
    }

    /// Run [`Self::step`] `n` times with the same `dt`.
    ///
    /// The native counterpart of the Python `step_n` and WASM `stepN`
    /// bindings, which call this: the result is bit-identical to calling
    /// `step(dt)` `n` times in a loop, `n = 0` leaves the world unchanged, and
    /// a non-positive `dt` leaves it unchanged for every `n` (as `step` does).
    ///
    /// # Examples
    ///
    /// ```
    /// use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
    ///
    /// let mut a = PhysicsWorld::new(PhysicsConfig::default());
    /// a.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE));
    /// let mut b = PhysicsWorld::from_world_snapshot(&a.snapshot_world()).unwrap();
    ///
    /// let dt = Fix128::from_ratio(1, 60);
    /// a.step_n(3, dt);
    /// for _ in 0..3 {
    ///     b.step(dt);
    /// }
    /// assert_eq!(a.bodies[0].position, b.bodies[0].position);
    /// ```
    pub fn step_n(&mut self, n: usize, dt: Fix128) {
        for _ in 0..n {
            self.step(dt);
        }
    }

    /// Phase 4 of [`Self::step`]: [`IslandManager::update_sleep`] over the bodies
    /// that are not parked, and for a parked body the same outcome without the
    /// evaluation (its velocity is the at-rest one its cache recorded as idle).
    fn update_sleep_states(&mut self) {
        if !self.park.active {
            self.stage_work.sleep_evaluated += self.bodies.len() as u64;
            self.islands.update_sleep(&self.bodies);
            return;
        }
        for k in 0..self.park.awake.len() {
            let i = self.park.awake[k];
            self.stage_work.sleep_evaluated += 1;
            self.islands.update_sleep_body(i, &self.bodies[i]);
        }
        for k in 0..self.park.parked_list.len() {
            let i = self.park.parked_list[k];
            if !self.park.parked[i] {
                continue; // woken during the step: evaluated above
            }
            self.stage_work.parked_sleep_updates += 1;
            self.islands
                .update_sleep_idle(i, self.bodies[i].is_static());
        }
    }

    /// Decide which bodies are parked for this step (sleep skip, see
    /// [`StageWork`]) and bring the parked-body tree up to date.
    ///
    /// A body is parked when the step would change nothing about it but its
    /// `idle_frames`: it is asleep, not kinematic, not referenced by a joint or a
    /// distance constraint, and — for a dynamic body — `prev == position`,
    /// `prev_rotation == rotation`, a zero velocity, the angular velocity the
    /// derivation gives an unchanged rotation, an idle verdict on that velocity,
    /// and no static / SDF collider touching it. A contact with an awake body
    /// wakes it during the step like any sleeping body ([`ParkState::unpark`]).
    /// Nothing is parked while an SDF collider is attached to a body.
    fn park_begin(&mut self, dt: Fix128, substep_dt: Fix128) {
        self.park.active = false;
        let n = self.bodies.len();
        if !self.sleep_skip || self.broadphase != Broadphase::Bvh || n == 0 {
            if !self.park.proxies.is_empty() || self.park.key.is_some() {
                self.park.clear_counted(&mut self.stage_work);
            }
            return;
        }
        // A zero velocity must be idle, otherwise every sleeping body wakes this
        // step and there is nothing to skip.
        let cfg = self.islands.config;
        if !(Vec3Fix::ZERO.length() < cfg.linear_threshold) {
            self.park.clear_counted(&mut self.stage_work);
            return;
        }
        // A collider attached to a body moves with it during the step, after
        // this check has passed a parked body as clear of it: no parking while
        // such a collider is in the world.
        if self.sdf_colliders.iter().any(|c| c.body_index < n) {
            self.park.clear_counted(&mut self.stage_work);
            return;
        }

        let key = ParkKey {
            substep_dt,
            sleep: cfg,
            generation: self.park_generation,
            sdf_radius: self.sdf_collision_radius,
            sdf: self
                .sdf_colliders
                .iter()
                .map(|c| {
                    (
                        c.position,
                        c.rotation,
                        c.scale,
                        c.body_index,
                        core::ptr::from_ref::<dyn crate::sdf_collider::SdfField>(c.field.as_ref())
                            .cast::<u8>() as usize,
                    )
                })
                .collect(),
        };
        // Fewer bodies than last time: indices may now name other bodies.
        if self.park.key.as_ref() != Some(&key) || n < self.park.parked.len() {
            self.park.clear_counted(&mut self.stage_work);
            self.park.key = Some(key);
        }

        self.park.parked.resize(n, false);
        self.park.cache.resize(n, None);
        self.park.proxies.resize(n, None);
        self.park.constrained.clear();
        self.park.constrained.resize(n, false);
        for j in &self.joints {
            let (a, b) = j.bodies();
            for x in [a, b] {
                if x < n {
                    self.park.constrained[x] = true;
                }
            }
        }
        for c in &self.distance_constraints {
            for x in [c.body_a, c.body_b] {
                if x < n {
                    self.park.constrained[x] = true;
                }
            }
        }

        self.park.frame_dt = dt;
        self.park.awake.clear();
        self.park.parked_list.clear();
        let inv_dt = Fix128::ONE / substep_dt;
        for i in 0..n {
            self.stage_work.sleep_scanned += 1;
            let (ok, recached) = self.park_check(i, inv_dt);
            self.park.parked[i] = ok;
            let radius = self.body_collision_radii.get(i).and_then(|r| *r);
            match (ok, radius) {
                (true, Some(r)) => {
                    self.park.parked_list.push(i);
                    if recached || self.park.proxies[i].is_none() {
                        self.park.remove_proxy(i, &mut self.stage_work);
                        let aabb = self.broadphase_box(i, r);
                        self.park.proxies[i] = Some(self.park.tree.insert(aabb, i as u32));
                        self.park.proxy_live += 1;
                        self.stage_work.tree_inserts += 1;
                    }
                }
                (true, None) => {
                    self.park.parked_list.push(i);
                    self.park.remove_proxy(i, &mut self.stage_work);
                }
                (false, _) => {
                    self.park.awake.push(i);
                    self.park.remove_proxy(i, &mut self.stage_work);
                }
            }
        }
        self.stage_work.parked = self.park.parked_list.len() as u64;
        self.park.active = true;
    }

    /// `(parked, cache rewritten)` for body `i` (see [`Self::park_begin`]).
    fn park_check(&mut self, i: usize, inv_dt: Fix128) -> (bool, bool) {
        if !self.islands.is_sleeping(i) || self.park.constrained[i] {
            return (false, false);
        }
        let body = &self.bodies[i];
        let radius = self.body_collision_radii.get(i).and_then(|r| *r);
        match body.body_type {
            BodyType::Kinematic => (false, false),
            BodyType::Static => {
                if !body.is_static() {
                    return (false, false);
                }
                // A collider's box turns with the body; a sphere cube does not.
                let has_collider = self.body_colliders.get(i).is_some_and(Option::is_some);
                let c = ParkCache {
                    position: body.position,
                    rotation: body.rotation,
                    angular_velocity: Vec3Fix::ZERO,
                    radius,
                    static_kind: true,
                };
                match self.park.cache[i] {
                    Some(old)
                        if old.static_kind
                            && old.position == c.position
                            && old.radius == radius
                            && (old.rotation == c.rotation || !has_collider) =>
                    {
                        (true, false)
                    }
                    _ => {
                        self.park.cache[i] = Some(c);
                        (true, true)
                    }
                }
            }
            BodyType::Dynamic => {
                if body.prev_position != body.position
                    || body.prev_rotation != body.rotation
                    || body.velocity != Vec3Fix::ZERO
                {
                    return (false, false);
                }
                if let Some(c) = self.park.cache[i] {
                    if !c.static_kind
                        && c.position == body.position
                        && c.rotation == body.rotation
                        && c.radius == radius
                    {
                        return (body.angular_velocity == c.angular_velocity, false);
                    }
                }
                let omega = angular_from_rotations(body.rotation, body.rotation, inv_dt);
                if body.angular_velocity != omega
                    || !(omega.length() < self.islands.config.angular_threshold)
                    || !self.resolution_clear(i)
                {
                    self.park.cache[i] = None;
                    return (false, false);
                }
                self.park.cache[i] = Some(ParkCache {
                    position: body.position,
                    rotation: body.rotation,
                    angular_velocity: omega,
                    radius,
                    static_kind: false,
                });
                (true, true)
            }
        }
    }

    /// No static or SDF collider would move body `i` where it is now (the
    /// resolution passes test every collider at the same position when none
    /// returns a contact).
    fn resolution_clear(&self, i: usize) -> bool {
        let body = &self.bodies[i];
        if body.is_static() || body.is_sensor {
            return true; // both resolution passes skip it
        }
        let radius = self
            .body_collision_radii
            .get(i)
            .and_then(|r| *r)
            .unwrap_or(self.sdf_collision_radius);
        for collider in &self.static_colliders {
            if collider.collide_sphere(body.position, radius).is_some() {
                return false;
            }
        }
        #[cfg(feature = "std")]
        {
            let collider = self.body_colliders.get(i).and_then(Option::as_ref);
            for sdf in &self.sdf_colliders {
                // Its own field never moves it (`resolve_sdf_collisions`).
                if sdf.body_index == i {
                    continue;
                }
                if sdf_contact_of(collider, body, self.sdf_collision_radius, sdf).is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// How many bodies a stage visits: the unparked ones while parking is in
    /// effect, otherwise all of them (pair with [`Self::stage_body`]).
    #[inline]
    fn stage_count(&self) -> usize {
        if self.park.active {
            self.park.awake.len()
        } else {
            self.bodies.len()
        }
    }

    /// The `k`-th body a stage visits (see [`Self::stage_count`]).
    #[inline]
    fn stage_body(&self, k: usize) -> usize {
        if self.park.active {
            self.park.awake[k]
        } else {
            k
        }
    }

    /// Number of bodies that are not parked (all of them without parking).
    #[cfg(feature = "parallel")]
    fn unparked_count(&self) -> u64 {
        self.stage_count() as u64
    }

    /// Advances every [`BodyType::Kinematic`] body with a
    /// [`RigidBody::kinematic_target`] set, mirroring [`Self::integrate_positions`]'s
    /// `Kinematic` branch: velocity is derived from the position delta
    /// (`(target - position) / dt`), then position/rotation are snapped to
    /// the target. Runs once per full `dt` here (the `Tgs` path detects
    /// contacts once per tick, not once per sub-step, so there is no
    /// per-sub-step granularity to match) — called before collision
    /// detection so a moving kinematic body's contacts use its new
    /// position, same ordering as `step`'s per-sub-step kinematic advance
    /// relative to its per-sub-step contact handling.
    #[cfg(feature = "std")]
    fn advance_kinematic_targets_for_tgs(&mut self, dt: Fix128) {
        for body in &mut self.bodies {
            if body.body_type != BodyType::Kinematic {
                continue;
            }
            body.prev_position = body.position;
            body.prev_rotation = body.rotation;
            if let Some((target_pos, target_rot)) = body.kinematic_target {
                body.velocity = (target_pos - body.position) * (Fix128::ONE / dt);
                body.position = target_pos;
                body.rotation = target_rot;
            }
        }
    }
    /// Static colliders for the TGS path, with their positional correction
    /// carried into the velocities (see `step_tgs` Phase 3.2): a static
    /// contact pushes the body out and removes the velocity into the surface.
    #[cfg(feature = "std")]
    fn tgs_static_corrections(&mut self) {
        if !self.static_colliders.is_empty() {
            self.resolve_static_collisions(true);
        }
    }

    /// The bodies the world's joints refer to, ascending and without
    /// repeats (indices past the body list are left out).
    #[cfg(feature = "std")]
    fn jointed_bodies(&self) -> Vec<usize> {
        let n = self.bodies.len();
        let mut out: Vec<usize> = self
            .joints
            .iter()
            .flat_map(|j| {
                let (a, b) = j.bodies();
                [a, b]
            })
            .filter(|&i| i < n)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// One TGS substep's joint solve: copies the jointed bodies' TGS state to
    /// the world bodies, applies the joint projection
    /// ([`Self::solve_joints_dispatch`], the call the XPBD substep makes) and
    /// writes the result back, adding each correction to the velocity as the
    /// XPBD velocity update would: `Δx / h` to the linear velocity and the
    /// angular velocity of the rotation change to the angular one. TGS keeps
    /// its velocities across substeps, so without the carry a corrected body
    /// would keep the velocity that took it off the constraint.
    #[cfg(feature = "std")]
    fn tgs_project_joints(
        &mut self,
        tgs_bodies: &mut [crate::solver_tgs_hooks_6dof_oriented::Body6DofOrientedState],
        jointed: &[usize],
        sub_dt: Fix128,
    ) {
        use crate::solver_tgs_backend::tgs_to_body;
        for &i in jointed {
            tgs_to_body(&tgs_bodies[i], &mut self.bodies[i]);
        }
        let before: Vec<(Vec3Fix, QuatFix)> = jointed
            .iter()
            .map(|&i| (self.bodies[i].position, self.bodies[i].rotation))
            .collect();
        self.solve_joints_dispatch(sub_dt);
        let inv_dt = Fix128::ONE / sub_dt;
        let mut overflow = false;
        for (&i, (p0, q0)) in jointed.iter().zip(before) {
            let body = &mut self.bodies[i];
            if body.is_dynamic() {
                if body.position != p0 {
                    match (body.position - p0).checked_scale(inv_dt) {
                        Some(dv) => body.velocity = body.velocity + dv,
                        None => overflow = true,
                    }
                }
                if body.rotation != q0 {
                    body.angular_velocity =
                        body.angular_velocity + angular_from_rotations(body.rotation, q0, inv_dt);
                }
            }
            let state = &mut tgs_bodies[i];
            state.position = [body.position.x, body.position.y, body.position.z];
            state.orientation = body.rotation;
            state.linear_velocity = [body.velocity.x, body.velocity.y, body.velocity.z];
            state.angular_velocity = [
                body.angular_velocity.x,
                body.angular_velocity.y,
                body.angular_velocity.z,
            ];
        }
        if overflow {
            self.note_rigid_overflow();
        }
    }

    /// `SolverBackend::Tgs` body of [`Self::step`]. Mirrors `step`'s phase
    /// numbering so the two are easy to diff, but is a genuinely different
    /// algorithm: contacts are detected once per full `dt` (not re-detected
    /// every sub-step — the TGS family owns its own sub-stepping internally
    /// via [`crate::solver_tgs::tgs_step`]), and bodies are advanced by
    /// per-island impulse-based Gauss-Seidel instead of XPBD position
    /// projection. Joints, static colliders, contact filters, kinematic
    /// targets and SDF colliders are all handled — see [`SolverBackend`]'s
    /// notes for the joint solve and filter handling, Phase 3.2 below for the
    /// static colliders, and
    /// [`Self::advance_kinematic_targets_for_tgs`] /
    /// [`Self::resolve_sdf_collisions`] for the last two.
    #[cfg(feature = "std")]
    fn step_tgs(&mut self, dt: Fix128, frozen: &mut [bool]) {
        use crate::solver_tgs::{build_islands, DistanceRef};
        use crate::solver_tgs_backend::{body_to_tgs, contact_to_tgs, tgs_to_body};
        use crate::solver_tgs_hooks_6dof_oriented::Pgs6DofOrientedConfig;
        use crate::solver_tgs_hooks_6dof_oriented_scoped::solve_oriented_islands_serial;

        // Phase 0: Event frame lifecycle (mirrors `step`; TGS detects contacts
        // once below instead of once per sub-step).
        self.events.begin_frame();

        // Phase 0.5: Rebuild island connectivity from current joints (same
        // `IslandManager` bookkeeping `step` performs; the TGS islands used
        // for solving below are a separate, local structure built from the
        // same joint list via `build_islands`).
        self.islands.resize(self.bodies.len());
        self.islands.reset_unions();
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < self.bodies.len() && b < self.bodies.len() {
                self.islands.union(a, b);
            }
        }

        // Phase 1: Apply force fields (identical call to `step`), then
        // advance kinematic targets before contacts are detected below.
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }
        // Joint motors: once per tick with the full `dt`, like the force
        // fields (TGS owns its sub-stepping, see the method doc).
        self.apply_joint_motors(dt);
        self.advance_kinematic_targets_for_tgs(dt);

        // Phase 2: Collision detection once for the whole tick (not
        // re-detected per sub-step — see the method doc).
        self.clear_contacts();
        self.detect_collisions();

        // Phase 2.5: Resolve SDF collisions (implicit surface contacts),
        // mirroring `substep`'s Phase 1.5. `resolve_sdf_collisions` pushes
        // bodies directly out of SDF overlap via `body.position +=
        // normal*depth` — it does not go through the contact-constraint /
        // impulse pipeline at all, so it is correct regardless of which
        // `SolverBackend` subsequently integrates velocities, and is called
        // once here (not once per TGS inner sub-step) for the same reason
        // `detect_collisions` above is: TGS owns its own sub-stepping
        // internally and this path only detects/resolves once per full
        // `dt`, same as every other per-tick (not per-substep) phase here.
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }

        // Phase 2.7: Pre-solve hooks and contact modifiers, once over the
        // tick's contact set (the same pre-pass `substep` runs over each
        // substep's set). A modifier's result is written back to the
        // constraint and read by `contact_to_tgs` below; a vetoed contact is
        // left out of the TGS contact list.
        self.apply_contact_filters();

        // Phase 3: Convert to the TGS body/contact representation, solve
        // every island, convert back.
        let n = self.bodies.len();
        let mut tgs_bodies: Vec<_> = self
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| body_to_tgs(b, i as u64))
            .collect();
        // Warm-start keys name a contact / constraint by what it joins, not
        // by its position in a vector: the stable ids of its two bodies and
        // its ordinal among the entries that join the same pair this tick.
        self.sync_body_stable_ids();
        let ids = &self.body_stable_ids;
        // An out-of-range index (a hand-built constraint) gets a sentinel;
        // `build_islands` below rejects it before anything is solved.
        let id_of = |i: usize| ids.get(i).copied().unwrap_or(u64::MAX);
        let mut contact_ordinals = std::collections::BTreeMap::new();
        // Contacts vetoed by a hook or a modifier (`apply_contact_filters`
        // above) are not solved, as in the XPBD contact solve. The key is
        // taken before the veto is applied, so vetoing one contact of a pair
        // does not shift the ordinal (and so the warm-start entry) of the
        // pair's other contacts. A modified contact is converted from the
        // values the modifier wrote back. (Sensor pairs never reach
        // `contact_constraints`: detection reports them as triggers.)
        let discarded = &self.contact_discarded;
        let mut tgs_contacts: Vec<_> = self
            .contact_constraints
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                // Orient every contact from the lower stable id to the higher
                // one. Detection orders a pair by index, and `swap_remove`
                // can flip the index order of two bodies whose ids did not
                // change; without this the same contact would come back
                // reversed (opposite normal and tangent basis) and miss.
                let (ia, ib) = (id_of(c.body_a), id_of(c.body_b));
                let flipped;
                let c = if ia > ib {
                    flipped = ContactConstraint {
                        body_a: c.body_b,
                        body_b: c.body_a,
                        contact: Contact {
                            depth: c.contact.depth,
                            normal: -c.contact.normal,
                            point_a: c.contact.point_b,
                            point_b: c.contact.point_a,
                        },
                        ..*c
                    };
                    &flipped
                } else {
                    c
                };
                let key = tgs_cache_key(
                    TGS_KEY_CONTACT,
                    ia.min(ib),
                    ia.max(ib),
                    &mut contact_ordinals,
                );
                if discarded.get(i).copied().unwrap_or(false) {
                    return None;
                }
                Some(contact_to_tgs(c, &self.bodies, key))
            })
            .collect();
        let joint_refs: Vec<DistanceRef<'_>> = self
            .distance_constraints
            .iter()
            .map(|joint| DistanceRef { joint })
            .collect();
        // Joints share `self.tgs_impulse_cache` with contacts; the kind tag
        // in the key keeps the two apart.
        let mut joint_ordinals = std::collections::BTreeMap::new();
        let mut tgs_joints: Vec<crate::solver_tgs_hooks_6dof_oriented::JointOriented> = self
            .distance_constraints
            .iter()
            .map(|j| crate::solver_tgs_hooks_6dof_oriented::JointOriented {
                body_a: j.body_a,
                body_b: j.body_b,
                stable_id: tgs_cache_key(
                    TGS_KEY_DISTANCE,
                    id_of(j.body_a),
                    id_of(j.body_b),
                    &mut joint_ordinals,
                ),
                local_anchor_a: [j.local_anchor_a.x, j.local_anchor_a.y, j.local_anchor_a.z],
                local_anchor_b: [j.local_anchor_b.x, j.local_anchor_b.y, j.local_anchor_b.z],
                target_distance: j.target_distance,
                accum: Fix128::ZERO,
            })
            .collect();

        // `build_islands` only errs when a contact/joint references a body
        // index `>= n`; every index here comes from this same world's own
        // `detect_collisions` / `add_distance_constraint`, both of which are
        // bounds-checked against `self.bodies.len()` at insertion time, so
        // this is unreachable in practice. Treat it as "nothing to solve"
        // rather than panicking an FFI host.
        let islands = build_islands(&tgs_bodies, &tgs_contacts, &joint_refs).unwrap_or_default();

        let cfg = Pgs6DofOrientedConfig {
            gravity: [
                self.config.gravity.x,
                self.config.gravity.y,
                self.config.gravity.z,
            ],
            ..Pgs6DofOrientedConfig::default()
        };
        let tgs_cfg = crate::solver_tgs::TgsConfig {
            substeps: self.config.substeps.max(1) as u32,
            velocity_iters: self.config.iterations.max(1) as u32,
            ..crate::solver_tgs::TgsConfig::default()
        };
        // Participants need a substep loop of their own: with none registered
        // the joint-free path below is the single call it always was. The
        // loop splits `dt` exactly as that call does, so a world whose
        // participants stage nothing gives the same bits either way.
        let with_participants = !frozen.is_empty();
        if self.joints.is_empty() && !with_participants {
            solve_oriented_islands_serial(
                &mut tgs_bodies,
                &mut tgs_contacts,
                &mut tgs_joints,
                &islands,
                &mut self.tgs_impulse_cache,
                cfg,
                &tgs_cfg,
                dt,
            );
        } else {
            // The world's joints (`Joint`: ball, hinge, fixed, slider, spring,
            // D6, cone-twist) are solved inside the substep loop: after each
            // substep's impulse solve and position advance, the joint
            // projection the XPBD substep runs is applied to the jointed
            // bodies and carried into their velocities. The loop splits `dt`
            // exactly as `tgs_step` does, so each island sees the same
            // substep sequence it would in a single call.
            let one_substep = crate::solver_tgs::TgsConfig {
                substeps: 1,
                ..tgs_cfg
            };
            //
            // The width is the one `tgs_step` uses, with or without
            // participants, and participants are handed the same `h`
            // ([`Self::participant_substep_width`]): a participant that
            // stages nothing then leaves the bodies bit for bit as in a world
            // without it, for every substep count.
            let substeps = self.config.substeps;
            let sub_dt = dt * Fix128::from_f32(1.0 / tgs_cfg.substeps as f32);
            let jointed = self.jointed_bodies();
            for s in 0..tgs_cfg.substeps {
                let overflow_at_start = if with_participants {
                    // Participants read the bodies as they are at the start of
                    // this substep, and their forces change the velocities
                    // the TGS solve starts from.
                    for (body, state) in self.bodies.iter_mut().zip(tgs_bodies.iter()) {
                        tgs_to_body(state, body);
                    }
                    let flag =
                        self.participants_begin_substep(s as usize, substeps, sub_dt, frozen);
                    for (state, body) in tgs_bodies.iter_mut().zip(self.bodies.iter()) {
                        state.linear_velocity = [body.velocity.x, body.velocity.y, body.velocity.z];
                        state.angular_velocity = [
                            body.angular_velocity.x,
                            body.angular_velocity.y,
                            body.angular_velocity.z,
                        ];
                    }
                    flag
                } else {
                    false
                };
                solve_oriented_islands_serial(
                    &mut tgs_bodies,
                    &mut tgs_contacts,
                    &mut tgs_joints,
                    &islands,
                    &mut self.tgs_impulse_cache,
                    cfg,
                    &one_substep,
                    sub_dt,
                );
                if !self.joints.is_empty() {
                    self.tgs_project_joints(&mut tgs_bodies, &jointed, sub_dt);
                }
                if with_participants {
                    // Keep `self.bodies` in step with the TGS state after every
                    // substep. The end-of-substep check reads only the
                    // overflow flag, not the bodies; the next substep copies
                    // the bodies out again before its participants run, and
                    // the loop end copies them once more.
                    for (body, state) in self.bodies.iter_mut().zip(tgs_bodies.iter()) {
                        tgs_to_body(state, body);
                    }
                    // この substep の範囲外を、参加者が読む前に world の印へ畳む
                    if tgs_bodies.iter().any(|s| s.overflow) {
                        self.note_rigid_overflow();
                    }
                    self.participants_end_substep(overflow_at_start, frozen);
                }
            }
        }
        // Every contact for this tick has now been visited (each contact's
        // warm-start entry was `take`n in `begin_substep` and re-`set` in
        // `end_substep` for every sub-step above) — evict any cache entry
        // that was not touched this tick, per `ImpulseCache::sweep`'s own
        // documented contract. Without this, a contact that stops recurring
        // (bodies separate, then the same body pair contacts again later)
        // would resurrect the old, stale impulse as a warm-start "hit"
        // instead of correctly starting from zero.
        self.tgs_impulse_cache.sweep();
        debug_assert_eq!(tgs_bodies.len(), n);
        for (body, state) in self.bodies.iter_mut().zip(tgs_bodies.iter()) {
            tgs_to_body(state, body);
        }
        // TGS は複製 `tgs_bodies` の上で積分するので、範囲外は複製の各 body の
        // `overflow` に溜まる (その body の位置は据え置き、XPBD の積分と同じ扱い)
        // 戻す時に world の sticky flag へ畳み込む
        if tgs_bodies.iter().any(|s| s.overflow) {
            self.note_rigid_overflow();
        }

        // Phase 3.2: The world's static colliders. A static contact is a
        // position-level correction (the call `substep` makes); TGS owns
        // velocities, so it also removes the velocity into the surface.
        // Without that a body resting on a plane keeps the velocity gravity
        // gave it and sinks again next tick. (The joints were solved inside
        // the substep loop above.)
        self.tgs_static_corrections();

        // Phase 3.5: Frame-level damping (identical call to `step`; TGS's own
        // per-substep gravity/impulse integration does not apply the global
        // `damping` factor, so this still needs to run here).
        self.apply_frame_damping();

        // Phase 4: Update sleeping (identical call to `step`).
        self.islands.update_sleep(&self.bodies);

        // Phase 4.5: Leave every SDF collider attached to a body at that
        // body's final pose, which is what the queries between steps
        // (`sdf_contacts`, `sdf_ccd_hits`, the shape casts) read.
        self.sync_sdf_colliders();

        // Phase 5: End event frame (identical call to `step`).
        self.events.end_frame();
    }

    /// Warm-start hit-rate diagnostics for the [`SolverBackend::Tgs`] path's
    /// per-contact impulse cache (see [`Self::reset_tgs_cache_stats`] to
    /// isolate a specific measurement window, e.g. after warm-up frames).
    ///
    /// Always `hits: 0, misses: 0` while `config.solver_backend` is
    /// `Xpbd` — the internal `step_tgs` (the only writer of this cache)
    /// is never called on that path.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn tgs_cache_stats(&self) -> TgsCacheStats {
        self.tgs_impulse_cache.stats().into()
    }

    /// Clears the [`Self::tgs_cache_stats`] hit/miss counters without
    /// touching the cached impulses themselves — the same contract as the
    /// internal `ImpulseCache::reset_stats`. Useful when a host wants to
    /// measure hit rate over a specific window instead of the cache's
    /// entire lifetime (e.g. after a scene's warm-up frames).
    #[cfg(feature = "std")]
    pub fn reset_tgs_cache_stats(&mut self) {
        self.tgs_impulse_cache.reset_stats();
    }

    /// Step with batched constraint solving
    ///
    /// When `parallel` feature is enabled, processes independent constraint
    /// batches in parallel using Rayon. Includes the same integrated pipeline
    /// as `step()`: force fields, collision detection, joints, events, sleeping.
    ///
    /// # Determinism contract
    ///
    /// `step_parallel` is deterministic — the same world stepped twice, on any
    /// number of threads and on every platform, produces the same bits (graph
    /// colouring is deterministic and constraints inside a batch touch
    /// disjoint bodies). It is **not** bit-identical to [`Self::step`] once two
    /// constraints share a body: the batches are a different Gauss–Seidel
    /// ordering than constraint index order, and Gauss–Seidel is order
    /// dependent. Lockstep / rollback peers must therefore all use the same
    /// path (`fuzz/fuzz_targets/fuzz_step_parity.rs` checks both properties).
    ///
    /// # Participants
    ///
    /// `step_parallel` is [`Self::try_step_parallel`] with the result dropped.
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, dt: Fix128) {
        let _ = self.try_step_parallel(dt);
    }

    /// The step body behind [`Self::try_step_parallel`] (checks already done,
    /// `dt > 0`).
    #[cfg(feature = "parallel")]
    fn run_step_parallel(&mut self, dt: Fix128) {
        self.make_body_rotations_unit();
        let mut frozen = self.participant_flags();

        // Phase 0: Event frame lifecycle (contacts are cleared and re-detected
        // in every substep since 1.2.0, see `substep`)
        self.events.begin_frame();

        // Phase 0.5: Rebuild island connectivity from current joints
        self.islands.resize(self.bodies.len());
        self.islands.reset_unions();
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < self.bodies.len() && b < self.bodies.len() {
                self.islands.union(a, b);
            }
        }

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }

        // Phase 2 (collision detection) is per substep since 1.2.0, see `step`.

        // Phase 3: Substep loop (batches are rebuilt inside each substep after
        // detection, `solve_constraints_batched` re-colours on dirty)
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        let n = self.config.substeps;
        for i in 0..n {
            self.kinematic_substeps_left = n - i;
            let overflow_at_start = self.participants_begin_substep(i, n, substep_dt, &mut frozen);
            self.substep_batched(substep_dt);
            self.participants_end_substep(overflow_at_start, &frozen);
        }
        self.kinematic_substeps_left = 0;

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

        // Phase 4.5: Leave every SDF collider attached to a body at that
        // body's final pose, which is what the queries between steps
        // (`sdf_contacts`, `sdf_ccd_hits`, the shape casts) read.
        self.sync_sdf_colliders();

        // Phase 5: End event frame
        self.events.end_frame();
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        // 0. Joint motors act as external forces / torques of this substep.
        self.apply_joint_motors(dt);
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.1. Continuous collision of fast bodies (`world_ccd`, off by
        //      default: nothing runs while it is off).
        let ccd_hits = if self.ccd.is_enabled() {
            self.ccd_sweep()
        } else {
            Vec::new()
        };

        // 1.2. Collision detection on the predicted positions (Small Steps).
        //      Contacts live for exactly one substep.
        self.clear_contacts();
        self.detect_collisions();
        if !ccd_hits.is_empty() {
            self.ccd_add_contacts(&ccd_hits);
        }

        // 1.5. Resolve SDF collisions (implicit surface contacts)
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }
        if !self.static_colliders.is_empty() {
            self.resolve_static_collisions(false);
        }

        // 2. Solve constraints (sequential). Hooks and modifiers run once per
        //    substep, ahead of the iteration loop.
        #[cfg(feature = "std")]
        self.apply_contact_filters();
        for _ in 0..self.config.iterations {
            self.solve_distance_constraints(dt);
            self.solve_contact_constraints(dt);
        }

        // 3. Solve joint constraints (v0.12.0: auto-routes through
        //    installed bridge if any, otherwise falls back to CPU
        //    `solve_joints`).
        self.solve_joints_dispatch(dt);

        self.update_velocities(dt);
    }

    /// Single substep with batched constraint solving
    #[cfg(feature = "parallel")]
    fn substep_batched(&mut self, dt: Fix128) {
        self.apply_joint_motors(dt);
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.1. Continuous collision of fast bodies, as in `substep`.
        let ccd_hits = if self.ccd.is_enabled() {
            self.ccd_sweep()
        } else {
            Vec::new()
        };

        // 1.2. Collision detection on the predicted positions (Small Steps).
        self.clear_contacts();
        self.detect_collisions();
        if !ccd_hits.is_empty() {
            self.ccd_add_contacts(&ccd_hits);
        }
        self.rebuild_batches();

        // 1.5. Resolve SDF collisions
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }
        if !self.static_colliders.is_empty() {
            self.resolve_static_collisions(false);
        }

        // 2. Solve constraints (batched). Hooks and modifiers run once per
        //    substep, ahead of the iteration loop, as in `substep`.
        #[cfg(feature = "std")]
        self.apply_contact_filters();
        for _ in 0..self.config.iterations {
            self.solve_constraints_batched(dt);
        }

        // 3. Solve joint constraints (v0.12.0: auto-routes through
        //    installed bridge if any, otherwise falls back to CPU
        //    `solve_joints`).
        self.solve_joints_dispatch(dt);

        self.update_velocities(dt);
    }

    /// Integrate positions (shared between sequential and batched)
    ///
    /// # Claims
    ///
    /// - A kinematic body with a `kinematic_target` closes `1 / k` of the
    ///   remaining gap to the target in each substep, `k` being the substeps
    ///   left in the frame including this one, so it reaches the target
    ///   exactly in the last substep.
    /// - The velocity derived after every substep of that frame equals
    ///   `(target - start) / frame_dt` in m/s, so it is that value after
    ///   `step`, up to fixed-point rounding of the division by `k`.
    /// - A call outside a `step` substep loop (`k = 0`) snaps to the target
    ///   in one call, as before.
    /// - A kinematic body without a target keeps its pose and velocity.
    ///
    /// Sleeping dynamic bodies are skipped: only `prev_position/prev_rotation`
    /// are saved so that `update_velocities` derives zero velocity.
    #[inline]
    fn integrate_positions(&mut self, dt: Fix128) {
        // ⚠️ 範囲外の積を集める受け皿 — parallel branch の closure から
        // `self` に書けないので Atomic で受けて関数末尾で flag に畳み込む
        // (bool の OR は結合的・可換なので rayon の実行順に依存しない)
        let overflow = core::sync::atomic::AtomicBool::new(false);
        // Free rotation of this substep per body (`None`: no gyroscopic
        // response, the rotation is predicted from `ω` as before); read back
        // by `update_velocities` in the same substep.
        let mut free = core::mem::take(&mut self.free_rotation);
        free.clear();
        free.resize(self.bodies.len(), None);
        let mut predicted = core::mem::take(&mut self.predicted_pose);
        predicted.clear();
        predicted.resize(self.bodies.len(), None);
        #[cfg(feature = "parallel")]
        {
            let gravity = self.config.gravity;
            let left = self.kinematic_substeps_left;
            let sleep_data = &self.islands.sleep_data;
            let parked: &[bool] = if self.park.active {
                &self.park.parked
            } else {
                &[]
            };
            self.stage_work.integrated += self.unparked_count();
            self.bodies
                .par_iter_mut()
                .zip(free.par_iter_mut())
                .zip(predicted.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((body, free), predicted))| {
                    if parked.get(i).copied().unwrap_or(false) {
                        return;
                    }
                    match body.body_type {
                        BodyType::Static => return,
                        BodyType::Kinematic => {
                            body.prev_position = body.position;
                            body.prev_rotation = body.rotation;
                            if let Some((target_pos, target_rot)) = body.kinematic_target {
                                let (p, r) = kinematic_substep_pose(
                                    body.position,
                                    body.rotation,
                                    target_pos,
                                    target_rot,
                                    left,
                                );
                                body.velocity = (p - body.position) * (Fix128::ONE / dt);
                                body.position = p;
                                body.rotation = r;
                            }
                            return;
                        }
                        BodyType::Dynamic => {}
                    }

                    // Skip sleeping bodies (preserve prev for zero-velocity derivation)
                    if sleep_data
                        .get(i)
                        .is_some_and(super::sleeping::SleepData::is_sleeping)
                    {
                        body.prev_position = body.position;
                        body.prev_rotation = body.rotation;
                        return;
                    }

                    // Store previous state
                    body.prev_position = body.position;
                    body.prev_rotation = body.rotation;

                    // Apply gravity (with per-body scale)
                    body.velocity = body.velocity + gravity * body.gravity_scale * dt;

                    // Damping is applied once per frame in `apply_frame_damping`
                    // (1.2.0). Applying it here, per substep, made the terminal
                    // velocity depend on `substeps` (v_inf = g*h*d/(1-d)).

                    // Predict position
                    // ⚠️ parallel branch でも同じ検出を行う — 片方だけだと
                    // `--features parallel` で guard が silent に消える
                    // (closure から `self` に書けないので bool を reduce する)
                    // 位置の加算も検査する (`|x| ≥ 2⁶³` で反対側の端へ wrap するため)
                    match body
                        .velocity
                        .checked_scale(dt)
                        .and_then(|d| body.position.checked_add(d))
                    {
                        Some(p) => body.position = p,
                        None => overflow.store(true, core::sync::atomic::Ordering::Relaxed),
                    }

                    // Predict rotation, with the gyroscopic term ω × Iω
                    let mut spin_overflow = false;
                    *free = predict_rotation(body, dt, &mut spin_overflow);
                    if spin_overflow {
                        overflow.store(true, core::sync::atomic::Ordering::Relaxed);
                    }

                    // The predicted pose, against which `update_velocities`
                    // measures the solve's correction
                    *predicted = Some((body.position, body.rotation));
                });
            self.free_rotation = free;
            self.predicted_pose = predicted;
        }

        #[cfg(not(feature = "parallel"))]
        {
            let count = self.stage_count();
            for k in 0..count {
                let i = self.stage_body(k);
                self.stage_work.integrated += 1;
                match self.bodies[i].body_type {
                    BodyType::Static => continue,
                    BodyType::Kinematic => {
                        self.bodies[i].prev_position = self.bodies[i].position;
                        self.bodies[i].prev_rotation = self.bodies[i].rotation;
                        if let Some((target_pos, target_rot)) = self.bodies[i].kinematic_target {
                            let (p, r) = kinematic_substep_pose(
                                self.bodies[i].position,
                                self.bodies[i].rotation,
                                target_pos,
                                target_rot,
                                self.kinematic_substeps_left,
                            );
                            self.bodies[i].velocity =
                                (p - self.bodies[i].position) * (Fix128::ONE / dt);
                            self.bodies[i].position = p;
                            self.bodies[i].rotation = r;
                        }
                        continue;
                    }
                    BodyType::Dynamic => {}
                }

                // Skip sleeping bodies (preserve prev for zero-velocity derivation)
                if self.islands.is_sleeping(i) {
                    self.bodies[i].prev_position = self.bodies[i].position;
                    self.bodies[i].prev_rotation = self.bodies[i].rotation;
                    continue;
                }

                // Store previous state
                self.bodies[i].prev_position = self.bodies[i].position;
                self.bodies[i].prev_rotation = self.bodies[i].rotation;

                // Apply gravity (with per-body scale)
                let grav = self.config.gravity * self.bodies[i].gravity_scale * dt;
                self.bodies[i].velocity = self.bodies[i].velocity + grav;

                // Damping: once per frame in `apply_frame_damping` (see the
                // `parallel` branch above for the rationale).

                // Predict position
                // ⚠️ `checked_scale` で範囲外を検出する (WM-01 / B-12)
                // 範囲外なら **位置を動かさず** flag を立てる — 0 加算されて
                // 「原点で静止した自己整合な状態」に落ちるのを表に出すため
                // 位置の加算も検査する (`|x| ≥ 2⁶³` で反対側の端へ wrap するため)
                let body = &mut self.bodies[i];
                match body
                    .velocity
                    .checked_scale(dt)
                    .and_then(|d| body.position.checked_add(d))
                {
                    Some(p) => body.position = p,
                    None => overflow.store(true, core::sync::atomic::Ordering::Relaxed),
                }

                // Predict rotation, with the gyroscopic term ω × Iω
                let mut spin_overflow = false;
                free[i] = predict_rotation(&mut self.bodies[i], dt, &mut spin_overflow);
                if spin_overflow {
                    overflow.store(true, core::sync::atomic::Ordering::Relaxed);
                }

                // The predicted pose, against which `update_velocities`
                // measures the solve's correction
                predicted[i] = Some((self.bodies[i].position, self.bodies[i].rotation));
            }
            self.free_rotation = free;
            self.predicted_pose = predicted;
        }

        // ⚠️ 受け皿を sticky flag に畳み込む (1 度立ったら落ちない)
        if overflow.load(core::sync::atomic::Ordering::Relaxed) {
            self.note_rigid_overflow();
        }
    }

    /// Reset the XPBD Lagrange multipliers of the distance constraints at the
    /// start of a substep (Macklin 2016, Algorithm 1: `lambda = 0`).
    ///
    /// Within the substep `solve_distance_constraints` / `solve_distance_pair`
    /// accumulate `dlambda` into `cached_lambda`. Carrying a fraction of the
    /// previous substep's `lambda` over (the pre-1.2.0 "warm start") is not
    /// valid for a position-based multiplier: the carried value is treated as
    /// force already applied this substep although it was not, which inflates
    /// the steady-state extension of compliant constraints
    /// (`C = mg/k * (1 + wsf / (1 - wsf))`, 6.7x at `wsf = 0.85`). For rigid
    /// constraints (`compliance == 0`) `lambda` never enters the correction, so
    /// their behaviour is unchanged. `warm_start_factor` now only affects the
    /// contact solver (byte-exact parity contract with `GpuSolverBridge`
    /// implementations, see `solve_contact_pair`).
    fn reset_lambdas(&mut self) {
        for c in &mut self.distance_constraints {
            c.cached_lambda = Fix128::ZERO;
        }
    }

    /// Apply global and per-body velocity damping **once per frame**.
    ///
    /// `SolverConfig::damping` / `RigidBody::linear_damping` /
    /// `RigidBody::angular_damping` are velocity-retention factors per
    /// `step()` call. Before 1.2.0 they were multiplied in every substep, so
    /// the default `substeps: 8, damping: 0.99` gave every body a terminal
    /// velocity of `g * h * d / (1 - d)` with `h = dt / substeps` — about
    /// 2 m/s under default gravity — and doubling `substeps` halved the fall
    /// speed. A precision parameter must not change the physics, so damping
    /// now runs after the substep loop, on the velocities derived by the last
    /// `update_velocities`. Static / kinematic / sleeping bodies are skipped.
    fn apply_frame_damping(&mut self) {
        let damping = self.config.damping;
        let count = self.stage_count();
        for k in 0..count {
            let i = self.stage_body(k);
            self.stage_work.damping_bodies += 1;
            if self.bodies[i].body_type != BodyType::Dynamic || self.islands.is_sleeping(i) {
                continue;
            }
            let body = &mut self.bodies[i];
            body.velocity = body.velocity * damping * body.linear_damping;
            body.angular_velocity = body.angular_velocity * damping * body.angular_damping;
        }
    }

    /// Update velocities from position changes, then apply restitution and friction
    /// for active contact constraints.
    #[inline]
    fn update_velocities(&mut self, dt: Fix128) {
        let inv_dt = Fix128::ONE / dt;
        // ⚠️ 範囲外の積を集める受け皿 — parallel branch の closure から
        // `self` に書けないので Atomic で受けて関数末尾で flag に畳み込む
        // (bool の OR は結合的・可換なので rayon の実行順に依存しない)
        let overflow = core::sync::atomic::AtomicBool::new(false);

        // --- Phase 0: Record the pre-solve normal velocity v̄_n per contact ---
        // `velocity` still holds the substep's predicted velocity (gravity
        // applied in `integrate_positions`, untouched by the position solve),
        // so this is the approach velocity before any constraint acted.
        self.contact_pre_vn.clear();
        for constraint in &self.contact_constraints {
            let relative =
                self.bodies[constraint.body_a].velocity - self.bodies[constraint.body_b].velocity;
            self.contact_pre_vn
                .push(relative.dot(constraint.contact.normal));
        }

        // --- Phase 1: Derive velocities from position/rotation changes ---
        #[cfg(feature = "parallel")]
        {
            let parked: &[bool] = if self.park.active {
                &self.park.parked
            } else {
                &[]
            };
            self.stage_work.velocity_bodies += self.unparked_count();
            let free: &[Option<crate::gyroscopic::FreeRotation>] = &self.free_rotation;
            let predicted: &[Option<(Vec3Fix, QuatFix)>] = &self.predicted_pose;
            self.bodies
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, body)| {
                    if body.body_type == BodyType::Static || parked.get(i).copied().unwrap_or(false)
                    {
                        return;
                    }
                    // ⚠️ 速度導出も範囲外を検出する (WM-01 経路 2)
                    // 範囲外なら速度を更新しない (0 に上書きされるのを防ぐ)
                    let pose = predicted.get(i).copied().flatten();
                    if !derive_linear_velocity(body, pose.map(|(x, _)| x), inv_dt) {
                        overflow.store(true, core::sync::atomic::Ordering::Relaxed);
                    }
                    // Angular velocity from rotation change:
                    // delta_q = rotation * prev_rotation^-1
                    // angular_velocity = 2 * delta_q.xyz / dt  (when delta_q.w > 0)
                    // (split bodies: the split's ω plus the solve's change)
                    let f = free.get(i).copied().flatten();
                    body.angular_velocity =
                        derived_angular_velocity(body, f, pose.map(|(_, q)| q), inv_dt);
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            let count = self.stage_count();
            for k in 0..count {
                let i = self.stage_body(k);
                self.stage_work.velocity_bodies += 1;
                let body = &mut self.bodies[i];
                if body.body_type == BodyType::Static {
                    continue;
                }
                // ⚠️ 速度導出も範囲外を検出する (WM-01 経路 2)
                // 範囲外なら速度を更新しない (0 に上書きされるのを防ぐ)
                let pose = self.predicted_pose.get(i).copied().flatten();
                if !derive_linear_velocity(body, pose.map(|(x, _)| x), inv_dt) {
                    overflow.store(true, core::sync::atomic::Ordering::Relaxed);
                }
                // Angular velocity from rotation change (split bodies: the
                // split's ω plus the solve's change)
                let f = self.free_rotation.get(i).copied().flatten();
                let body = &mut self.bodies[i];
                body.angular_velocity =
                    derived_angular_velocity(body, f, pose.map(|(_, q)| q), inv_dt);
            }
        }

        // --- Phase 2: Apply restitution and friction at contacts ---
        // Restitution threshold of Müller et al. 2020 (below): an approach
        // slower than `2 |g| h` is a resting contact and gets `e = 0`, so
        // resting bodies do not bounce on the gravity of one substep
        // (the same rule as the 2D solver).
        let restitution_threshold = self.config.gravity.length() * dt * Fix128::from_int(2);
        let num_contacts = self.contact_constraints.len();
        for i in 0..num_contacts {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            if body_a.is_sensor || body_b.is_sensor {
                continue;
            }

            // A contact the substep's hook / modifier pre-pass discarded gets
            // no velocity response either (the position pass skipped it).
            // Kept contacts read the friction / restitution the pre-pass
            // wrote back, so a modifier's values apply here too.
            #[cfg(feature = "std")]
            if self.contact_discarded.get(i).copied().unwrap_or(false) {
                continue;
            }

            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if w_sum < W_SUM_EPSILON {
                continue;
            }

            let n = constraint.contact.normal;
            let relative_vel = body_a.velocity - body_b.velocity;
            let vn = relative_vel.dot(n);

            // Restitution (Müller, Macklin, Chentanez, Jeschke, Kim,
            // "Detailed Rigid Body Simulation with Extended Position Based
            // Dynamics", SCA 2020, eq. (34)): the target normal velocity is
            // `max(−e v̄_n, 0)` with `v̄_n` the pre-solve normal velocity. The
            // post-solve `vn` only reflects how far into the substep the
            // bodies first touched (`vn = −gap / h`), so reversing `vn`
            // itself made the bounce depend on that phase. The correction is
            // one-sided (only raises `vn` to the target), as in the 2D solver.
            let pre_vn = self.contact_pre_vn.get(i).copied().unwrap_or(vn);
            let restitution = if pre_vn < -restitution_threshold {
                constraint.restitution
            } else {
                Fix128::ZERO
            };
            let target = -restitution * pre_vn;
            let target = if target.is_negative() {
                Fix128::ZERO
            } else {
                target
            };
            if vn < target {
                let delta_vn = target - vn;
                let impulse_n = n * delta_vn;
                let inv_w = Fix128::ONE / w_sum;
                self.bodies[constraint.body_a].velocity =
                    self.bodies[constraint.body_a].velocity + impulse_n * (body_a.inv_mass * inv_w);
                self.bodies[constraint.body_b].velocity =
                    self.bodies[constraint.body_b].velocity - impulse_n * (body_b.inv_mass * inv_w);
            }

            // Friction: reduce tangential velocity
            let relative_vel2 =
                self.bodies[constraint.body_a].velocity - self.bodies[constraint.body_b].velocity;
            let vn2 = relative_vel2.dot(n);
            let tangent_vel = relative_vel2 - n * vn2;
            let tangent_speed_sq = tangent_vel.length_squared();
            if tangent_speed_sq > Fix128::ZERO {
                let tangent_speed = tangent_speed_sq.sqrt();
                let friction = constraint.friction;
                // Coulomb friction (Müller, Macklin, Chentanez, Jeschke, Kim,
                // "Detailed Rigid Body Simulation with Extended Position Based
                // Dynamics", SCA 2020, eq. (30)): the tangential correction
                // is capped by `h μ |f_n| w = μ λ_n / h` in relative velocity.
                // `cached_lambda` is this substep's accumulated normal
                // separation from the position solve (the paper's multiplier
                // times `w_sum`), so the cap is `μ λ / h`. The normal
                // velocity after the solve (`vn2`) is not the normal force:
                // it is about 0 for a resting contact, which removed the
                // friction of sliding and resting bodies.
                let max_friction_impulse = friction * constraint.cached_lambda * inv_dt;
                let applied = if tangent_speed < max_friction_impulse {
                    tangent_speed
                } else {
                    max_friction_impulse
                };
                let friction_dir = tangent_vel / tangent_speed;
                let friction_impulse = friction_dir * applied;
                let inv_w = Fix128::ONE / w_sum;
                self.bodies[constraint.body_a].velocity = self.bodies[constraint.body_a].velocity
                    - friction_impulse * (body_a.inv_mass * inv_w);
                self.bodies[constraint.body_b].velocity = self.bodies[constraint.body_b].velocity
                    + friction_impulse * (body_b.inv_mass * inv_w);
            }
        }

        // ⚠️ 受け皿を sticky flag に畳み込む (1 度立ったら落ちない)
        if overflow.load(core::sync::atomic::Ordering::Relaxed) {
            self.note_rigid_overflow();
        }
    }

    /// Run the pre-solve hooks and contact modifiers once over every
    /// contact constraint and keep the result.
    ///
    /// The serial, bridge and batched (`step_parallel`) substeps call this
    /// ahead of their iteration loop, so a relative change such as
    /// `friction *= 0.5` applies once per substep and not once per
    /// iteration, and every path calls each hook the same number of times.
    ///
    /// # Claims
    ///
    /// - Each hook and each modifier runs at most once per contact per call.
    /// - Sensor contacts are skipped without calling hooks or modifiers.
    /// - A modifier's result (contact, friction, restitution) is written back
    ///   to the constraint, and the solver passes read that value without
    ///   calling the modifier again.
    /// - A contact vetoed by a hook or discarded by a modifier is left
    ///   unmodified and is skipped by the solver passes (position and
    ///   velocity) until the next call.
    /// - With no hooks and no modifiers the call changes nothing.
    #[cfg(feature = "std")]
    fn apply_contact_filters(&mut self) {
        self.contact_discarded.clear();
        if self.pre_solve_hooks.is_empty() && self.contact_modifiers.is_empty() {
            return;
        }
        let num = self.contact_constraints.len();
        self.contact_discarded.resize(num, false);
        for i in 0..num {
            let constraint = self.contact_constraints[i];
            if self.bodies[constraint.body_a].is_sensor || self.bodies[constraint.body_b].is_sensor
            {
                continue;
            }

            let mut contact = constraint.contact;
            let mut friction = constraint.friction;
            let mut restitution = constraint.restitution;

            let mut skip = false;
            for hook in &self.pre_solve_hooks {
                if !hook(constraint.body_a, constraint.body_b, &contact) {
                    skip = true;
                    break;
                }
            }
            if !skip {
                for modifier in &self.contact_modifiers {
                    if !modifier.modify_contact(
                        constraint.body_a,
                        constraint.body_b,
                        &mut contact,
                        &mut friction,
                        &mut restitution,
                    ) {
                        skip = true;
                        break;
                    }
                }
            }
            if skip {
                self.contact_discarded[i] = true;
                continue;
            }

            let slot = &mut self.contact_constraints[i];
            slot.contact = contact;
            slot.friction = friction;
            slot.restitution = restitution;
        }
    }

    /// Solve constraints in batched parallel mode via Rayon.
    ///
    /// Within each colored batch, constraints share no body indices
    /// (guaranteed by `rebuild_batches` graph coloring), so they are
    /// dispatched to Rayon's thread pool for parallel execution.
    /// Between batches, a synchronization barrier ensures that
    /// earlier batch results are visible to later batches.
    ///
    /// Pre-solve hooks and contact modifiers are not called here: the
    /// substep runs [`Self::apply_contact_filters`] once before its
    /// iterations, and this pass skips the contacts it discarded and reads
    /// the values it wrote back.
    #[cfg(feature = "parallel")]
    fn solve_constraints_batched(&mut self, dt: Fix128) {
        // The coloring excluded bodies that were static when `rebuild_batches`
        // ran. If any body changed static-ness since (e.g. the user edited
        // `inv_mass` through `bodies` after the rebuild) the snapshot no longer
        // matches the live world and the batches must be recomputed, otherwise
        // two threads could take `&mut` to a body that is now dynamic.
        let snapshot_stale = self.batch_static_bodies.len() != self.bodies.len()
            || self
                .bodies
                .iter()
                .zip(&self.batch_static_bodies)
                .any(|(body, &was_static)| body.inv_mass.is_zero() != was_static);
        if snapshot_stale {
            self.batches_dirty = true;
            self.rebuild_batches();
        }

        let num_batches = self.constraint_batches.len();

        let bodies = BodySlicePtr {
            ptr: self.bodies.as_mut_ptr(),
            len: self.bodies.len(),
        };
        let dists = DistConstraintSlicePtr {
            ptr: self.distance_constraints.as_mut_ptr(),
            len: self.distance_constraints.len(),
        };
        let static_bodies: &[bool] = &self.batch_static_bodies;
        // Contacts the substep's hook / modifier pre-pass discarded (an index
        // past the end counts as kept)
        #[cfg(feature = "std")]
        let discarded: &[bool] = &self.contact_discarded;
        #[cfg(not(feature = "std"))]
        let discarded: &[bool] = &[];

        for batch_idx in 0..num_batches {
            // Phase 1: Distance constraints — parallel within batch
            {
                let indices = &self.constraint_batches[batch_idx].distance_indices;
                indices.par_iter().for_each(|&idx| {
                    // SAFETY: Graph coloring guarantees no two constraints in
                    // this batch share a dynamic body index; static bodies are
                    // borrowed shared only (`static_bodies` is the snapshot the
                    // coloring was built from, validated above). Each
                    // constraint index appears in exactly one batch, so
                    // cached_lambda writes are also disjoint.
                    unsafe {
                        let constraint = dists.get_mut(idx);
                        let body_a =
                            bodies.borrow(constraint.body_a, static_bodies[constraint.body_a]);
                        let body_b =
                            bodies.borrow(constraint.body_b, static_bodies[constraint.body_b]);
                        Self::solve_distance_pair(body_a, body_b, constraint, dt);
                    }
                });
            }

            // Phase 2: Contact constraints — parallel within batch
            {
                let contacts = ContactConstraintSlicePtr {
                    ptr: self.contact_constraints.as_mut_ptr(),
                    len: self.contact_constraints.len(),
                };
                let indices = &self.constraint_batches[batch_idx].contact_indices;
                indices.par_iter().for_each(|&idx| {
                    if discarded.get(idx).copied().unwrap_or(false) {
                        return;
                    }
                    // SAFETY: Same invariant as Phase 1 — disjoint dynamic
                    // bodies per batch, static bodies shared read-only,
                    // each constraint index in exactly one batch.
                    unsafe {
                        let constraint = contacts.get_mut(idx);
                        let body_a =
                            bodies.borrow(constraint.body_a, static_bodies[constraint.body_a]);
                        let body_b =
                            bodies.borrow(constraint.body_b, static_bodies[constraint.body_b]);
                        Self::solve_contact_pair(body_a, body_b, constraint);
                    }
                });
            }
        }
    }

    /// Solve a single distance constraint given mutable body references.
    ///
    /// Extracted as a static method (no `&self`) for parallel dispatch.
    /// Warm-starting seeds the solver with the cached lambda from the
    /// previous substep, reducing iterations needed for convergence.
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn solve_distance_pair(
        mut body_a: BodyRef<'_>,
        mut body_b: BodyRef<'_>,
        constraint: &mut DistanceConstraint,
        dt: Fix128,
    ) {
        let (pos_a, rot_a, inv_mass_a) = {
            let a = body_a.get();
            (a.position, a.rotation, a.inv_mass)
        };
        let (pos_b, rot_b, inv_mass_b) = {
            let b = body_b.get();
            (b.position, b.rotation, b.inv_mass)
        };

        let anchor_a = pos_a + rot_a.rotate_vec(constraint.local_anchor_a);
        let anchor_b = pos_b + rot_b.rotate_vec(constraint.local_anchor_b);

        let delta = anchor_b - anchor_a;
        let (normal, distance) = delta.normalize_with_length();

        if distance.is_zero() {
            return;
        }

        let error = distance - constraint.target_distance;

        let compliance_term = constraint.compliance / (dt * dt);
        let w_sum = inv_mass_a + inv_mass_b + compliance_term;

        if w_sum < W_SUM_EPSILON {
            return;
        }

        // XPBD (Macklin 2016): `cached_lambda` is the Lagrange multiplier
        // accumulated over this substep (zeroed by `reset_lambdas`).
        //   dlambda = (C - alpha~ * lambda) / (w_a + w_b + alpha~)
        //   lambda += dlambda,  dx = dlambda * w * grad C
        // Before 1.2.0 `lambda` was overwritten instead of accumulated, so the
        // alpha~*lambda term was always under-estimated and the effective
        // stiffness of compliant constraints depended on `iterations`.
        let inv_w_sum = Fix128::ONE / w_sum;
        let lambda = constraint.cached_lambda;
        let dlambda = (error - lambda * compliance_term) * inv_w_sum;
        constraint.cached_lambda = lambda + dlambda;
        let correction = normal * dlambda;

        // Branchless: static bodies have inv_mass == ZERO, correction * ZERO == ZERO.
        // `set_position` is a no-op for `BodyRef::Static`, matching the
        // sequential path which writes the unchanged position.
        let delta_a = correction * inv_mass_a;
        let delta_b = correction * inv_mass_b;
        body_a.set_position(select_vec3(!inv_mass_a.is_zero(), pos_a + delta_a, pos_a));
        body_b.set_position(select_vec3(!inv_mass_b.is_zero(), pos_b - delta_b, pos_b));
    }

    /// Solve a single contact constraint given mutable body references.
    ///
    /// Extracted as a static method (no `&self`) for parallel dispatch.
    /// Warm-starting seeds the solver with the cached lambda from the
    /// previous substep, reducing iterations needed for convergence.
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn solve_contact_pair(
        mut body_a: BodyRef<'_>,
        mut body_b: BodyRef<'_>,
        constraint: &mut ContactConstraint,
    ) {
        let (pos_a, prev_a, inv_mass_a, sensor_a) = {
            let a = body_a.get();
            (a.position, a.prev_position, a.inv_mass, a.is_sensor)
        };
        let (pos_b, prev_b, inv_mass_b, sensor_b) = {
            let b = body_b.get();
            (b.position, b.prev_position, b.inv_mass, b.is_sensor)
        };

        // Skip physics response for sensor/trigger bodies
        if sensor_a || sensor_b {
            return;
        }

        let contact = constraint.contact;

        if contact.depth <= Fix128::ZERO {
            return;
        }

        let w_sum = inv_mass_a + inv_mass_b;
        if w_sum < W_SUM_EPSILON {
            return;
        }

        // Accumulated non-negative contact multiplier (1.2.0). The contact is
        // re-detected every substep, so `depth` is the penetration at the start
        // of this substep and `cached_lambda` the separation already applied in
        // earlier iterations: dlambda = depth - lambda, clamped to lambda >= 0.
        // Before 1.2.0 every iteration re-pushed `depth - 0.85 * lambda_prev`
        // against a frame-stale depth, which multiplied the separation by the
        // iteration and substep counts (5 m/s collision → 700 m/s at defaults).
        let inv_w_sum = Fix128::ONE / w_sum;
        let lambda = constraint.cached_lambda;
        let dlambda = contact.depth - lambda;
        let (mut pos_a, mut pos_b) = (pos_a, pos_b);
        if dlambda > Fix128::ZERO {
            constraint.cached_lambda = lambda + dlambda;

            let correction = contact.normal * dlambda;
            let correction_a = correction * (inv_mass_a * inv_w_sum);
            let correction_b = correction * (inv_mass_b * inv_w_sum);

            // Branchless: inv_mass == ZERO for static bodies, correction_x will be ZERO.
            pos_a = select_vec3(!inv_mass_a.is_zero(), pos_a + correction_a, pos_a);
            pos_b = select_vec3(!inv_mass_b.is_zero(), pos_b - correction_b, pos_b);
        }

        // Static friction (Müller et al. 2020, eq. (26)), see
        // `static_friction_correction`.
        if let Some((fa, fb)) = static_friction_correction(
            pos_a - prev_a,
            pos_b - prev_b,
            contact.normal,
            inv_mass_a,
            inv_mass_b,
            inv_w_sum,
            constraint.friction,
            constraint.cached_lambda,
        ) {
            pos_a = select_vec3(!inv_mass_a.is_zero(), pos_a - fa, pos_a);
            pos_b = select_vec3(!inv_mass_b.is_zero(), pos_b + fb, pos_b);
        }

        // `set_position` is a no-op for `BodyRef::Static`.
        body_a.set_position(pos_a);
        body_b.set_position(pos_b);
    }

    /// Solve distance constraints (sequential) with warm-starting (Gap 3.1).
    ///
    /// Warm-starting seeds each constraint with its cached lambda from the
    /// previous substep, reducing iterations needed for convergence.
    #[inline(always)]
    fn solve_distance_constraints(&mut self, dt: Fix128) {
        let num_constraints = self.distance_constraints.len();
        for i in 0..num_constraints {
            let constraint = self.distance_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            // Get world-space anchor positions
            let anchor_a = body_a.position + body_a.rotation.rotate_vec(constraint.local_anchor_a);
            let anchor_b = body_b.position + body_b.rotation.rotate_vec(constraint.local_anchor_b);

            // Compute constraint error (single sqrt via normalize_with_length)
            let delta = anchor_b - anchor_a;
            let (normal, distance) = delta.normalize_with_length();

            if distance.is_zero() {
                continue;
            }

            let error = distance - constraint.target_distance;

            // Compute compliance term (XPBD)
            let compliance_term = constraint.compliance / (dt * dt);
            let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

            if w_sum < W_SUM_EPSILON {
                continue;
            }

            // XPBD: accumulate the Lagrange multiplier over the substep
            // (see `solve_distance_pair` for the derivation).
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = constraint.cached_lambda;
            let dlambda = (error - lambda * compliance_term) * inv_w_sum;
            self.distance_constraints[i].cached_lambda = lambda + dlambda;
            let correction = normal * dlambda;

            // Branchless apply corrections: static bodies have inv_mass == ZERO.
            let delta_a = correction * body_a.inv_mass;
            let delta_b = correction * body_b.inv_mass;
            self.bodies[constraint.body_a].position = select_vec3(
                !body_a.inv_mass.is_zero(),
                self.bodies[constraint.body_a].position + delta_a,
                self.bodies[constraint.body_a].position,
            );
            self.bodies[constraint.body_b].position = select_vec3(
                !body_b.inv_mass.is_zero(),
                self.bodies[constraint.body_b].position - delta_b,
                self.bodies[constraint.body_b].position,
            );
        }
    }

    /// One serial contact pass (one PGS iteration) over the constraints that
    /// [`Self::apply_contact_filters`] kept (Gap 2.3).
    ///
    /// # Claims
    ///
    /// - The pass does not call hooks or modifiers; it reads the values the
    ///   pre-pass wrote back, so the substep's iterations all see the same
    ///   friction, restitution and contact.
    /// - A modifier therefore runs once per substep, not once per iteration,
    ///   and a relative change such as `friction *= 0.5` does not compound
    ///   with `iterations`.
    /// - Sensor contacts and contacts discarded by the pre-pass are skipped.
    ///
    /// # v0.11.0 auto-routing
    ///
    /// If a GPU solver bridge has been installed on this world via
    /// [`Self::set_gpu_solver_bridge`], contact-solve is routed through
    /// the bridge (one pass, same filters as the pre-pass)
    /// instead of running the CPU inline solver below. The bridge is
    /// briefly taken out of `self` via `Option::take()` so the borrow
    /// checker sees `&mut self` for the routed call, then reinstalled
    /// on exit — the field is unchanged from the caller's perspective.
    ///
    /// If no bridge is installed, the existing CPU-only code path runs
    /// unchanged.
    fn solve_contact_constraints(&mut self, _dt: Fix128) {
        #[cfg(feature = "gpu-solver-bridge")]
        if let Some(mut bridge) = self.gpu_solver_bridge.take() {
            self.solve_contact_pass_with_bridge(bridge.as_mut());
            self.gpu_solver_bridge = Some(bridge);
            return;
        }
        let num_constraints = self.contact_constraints.len();
        for i in 0..num_constraints {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            // Skip physics response for sensor/trigger bodies
            if body_a.is_sensor || body_b.is_sensor {
                continue;
            }

            // Discarded by the substep's hook / modifier pre-pass
            #[cfg(feature = "std")]
            if self.contact_discarded.get(i).copied().unwrap_or(false) {
                continue;
            }

            // The modifiers' result was written back to the constraint by the
            // pre-pass, so this is the value every iteration sees.
            let contact = constraint.contact;

            // Only resolve if penetrating
            if contact.depth <= Fix128::ZERO {
                continue;
            }

            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if w_sum < W_SUM_EPSILON {
                continue;
            }

            // Accumulated contact multiplier, see `solve_contact_pair`.
            let inv_w_sum = Fix128::ONE / w_sum;
            let lambda = constraint.cached_lambda;
            let dlambda = contact.depth - lambda;
            if dlambda > Fix128::ZERO {
                self.contact_constraints[i].cached_lambda = lambda + dlambda;

                let correction = contact.normal * dlambda;
                let correction_a = correction * (body_a.inv_mass * inv_w_sum);
                let correction_b = correction * (body_b.inv_mass * inv_w_sum);

                // Branchless: inv_mass == ZERO for static bodies, correction_x will be ZERO.
                self.bodies[constraint.body_a].position = select_vec3(
                    !body_a.inv_mass.is_zero(),
                    self.bodies[constraint.body_a].position + correction_a,
                    self.bodies[constraint.body_a].position,
                );
                self.bodies[constraint.body_b].position = select_vec3(
                    !body_b.inv_mass.is_zero(),
                    self.bodies[constraint.body_b].position - correction_b,
                    self.bodies[constraint.body_b].position,
                );
            }

            // Static friction (Müller et al. 2020, eq. (26)), see
            // `static_friction_correction`.
            self.apply_static_friction(i, inv_w_sum);
        }
    }

    /// Position-level static friction of contact `i` against the current
    /// body positions, shared by the serial and bridge contact passes (the
    /// batched pass calls [`static_friction_correction`] on its pair
    /// references). The caller has already skipped sensor, discarded and
    /// non-penetrating contacts.
    fn apply_static_friction(&mut self, i: usize, inv_w_sum: Fix128) {
        let constraint = self.contact_constraints[i];
        let a = self.bodies[constraint.body_a];
        let b = self.bodies[constraint.body_b];
        if let Some((fa, fb)) = static_friction_correction(
            a.position - a.prev_position,
            b.position - b.prev_position,
            constraint.contact.normal,
            a.inv_mass,
            b.inv_mass,
            inv_w_sum,
            constraint.friction,
            constraint.cached_lambda,
        ) {
            self.bodies[constraint.body_a].position =
                select_vec3(!a.inv_mass.is_zero(), a.position - fa, a.position);
            self.bodies[constraint.body_b].position =
                select_vec3(!b.inv_mass.is_zero(), b.position + fb, b.position);
        }
    }

    /// v0.10.0 opt-in: run one PGS contact-solve iteration via a
    /// caller-supplied [`GpuSolverBridge`](crate::gpu_bridge::GpuSolverBridge) instead of the CPU-side
    /// `solve_contact_constraints` hot loop.
    ///
    /// # Claims
    ///
    /// - Each call runs the hooks and modifiers once, writes the result back
    ///   to the constraints, then runs one bridge pass over the survivors.
    /// - `substep_with_bridge` runs the hooks and modifiers once per substep
    ///   and then only the pass per iteration, so a relative modifier does
    ///   not compound with `iterations`.
    ///
    /// # Byte-exact CPU parity
    ///
    /// The bridge-routed path applies Stage A filters (sensor skip,
    /// pre-solve hooks, contact modifiers) on the CPU side upfront
    /// exactly as `solve_contact_constraints` does inline. Because
    /// the closures the hooks and modifiers evaluate depend only on
    /// their function inputs — never on live body state that
    /// changes during the loop — a batched upfront application is
    /// byte-exact equivalent to the CPU's per-constraint
    /// application. Stage B (depth ≤ 0 skip, w_sum < ε skip,
    /// warm-start biased lambda, cached_lambda write, position
    /// correction) is routed through the bridge, which runs the
    /// same sequential Gauss-Seidel semantics because bridge
    /// backends dispatch a single-workgroup single-thread compute
    /// kernel.
    ///
    /// # Constraint contract
    ///
    /// - The bridge receives a filtered `Vec<ContactConstraint>`
    ///   containing only the constraints that survived Stage A
    ///   (sensor-free, hook-approved, modifier-approved). The
    ///   contact / friction / restitution fields may be mutated
    ///   from the original slot values if a contact modifier
    ///   ran.
    /// - After readback the method writes each updated
    ///   `cached_lambda` back into `self.contact_constraints` via
    ///   an index map, mirroring the CPU write that lands at
    ///   `self.contact_constraints[i].cached_lambda = lambda` at
    ///   the corresponding slot.
    /// - Body positions are written back to `self.bodies[i].position`.
    ///
    /// # Panics
    ///
    /// Panics if the bridge implementation does not override the
    /// v0.9.0 contact-solve methods (`send_contact_constraints`,
    /// `send_body_state`, `dispatch_contact_solve_iteration`,
    /// `recv_contact_constraints`, `recv_body_positions`).
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn solve_contact_constraints_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
    ) {
        self.apply_contact_filters();
        self.solve_contact_pass_with_bridge(bridge);
    }

    /// One bridge-routed PGS pass over the constraints that
    /// [`Self::apply_contact_filters`] kept. Does not call hooks or
    /// modifiers, so the iteration loop of a substep sees the values the
    /// pre-pass wrote.
    #[cfg(feature = "gpu-solver-bridge")]
    fn solve_contact_pass_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
    ) {
        // ---- Stage A: select the surviving constraints ----
        let num_constraints = self.contact_constraints.len();
        let mut filtered: Vec<ContactConstraint> = Vec::with_capacity(num_constraints);
        let mut mapping: Vec<usize> = Vec::with_capacity(num_constraints);

        for i in 0..num_constraints {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            if body_a.is_sensor || body_b.is_sensor {
                continue;
            }
            if self.contact_discarded.get(i).copied().unwrap_or(false) {
                continue;
            }

            let contact = constraint.contact;
            let friction = constraint.friction;
            let restitution = constraint.restitution;

            filtered.push(ContactConstraint {
                body_a: constraint.body_a,
                body_b: constraint.body_b,
                contact,
                friction,
                restitution,
                cached_lambda: constraint.cached_lambda,
            });
            mapping.push(i);
        }

        if filtered.is_empty() {
            return;
        }

        // ---- Extract per-body state ----
        let mut positions: Vec<[Fix128; 3]> = Vec::with_capacity(self.bodies.len());
        let mut inv_masses: Vec<Fix128> = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            positions.push([body.position.x, body.position.y, body.position.z]);
            inv_masses.push(body.inv_mass);
        }

        // ---- Stage B: route through bridge ----
        bridge.send_contact_constraints(&filtered);
        bridge.send_body_state(&positions, &inv_masses);
        bridge.dispatch_contact_solve_iteration(self.config.warm_start_factor);

        let mut updated_constraints = filtered;
        let mut updated_positions = positions;
        bridge.recv_contact_constraints(&mut updated_constraints);
        bridge.recv_body_positions(&mut updated_positions);

        // ---- Write back cached_lambda + positions ----
        for (filtered_idx, orig_slot) in mapping.iter().enumerate() {
            self.contact_constraints[*orig_slot].cached_lambda =
                updated_constraints[filtered_idx].cached_lambda;
        }
        for (i, pos) in updated_positions.iter().enumerate() {
            self.bodies[i].position = Vec3Fix::new(pos[0], pos[1], pos[2]);
        }

        // Static friction (Müller et al. 2020, eq. (26)) on the CPU after
        // the bridge's normal pass, over the same filtered contacts (the
        // bridge contract carries only the normal multiplier).
        for &orig_slot in &mapping {
            let c = self.contact_constraints[orig_slot];
            let w_sum = self.bodies[c.body_a].inv_mass + self.bodies[c.body_b].inv_mass;
            if c.contact.depth <= Fix128::ZERO || w_sum < W_SUM_EPSILON {
                continue;
            }
            self.apply_static_friction(orig_slot, Fix128::ONE / w_sum);
        }
    }

    /// v0.10.0 opt-in: run one full simulation step with the
    /// contact-solve stage routed through a caller-supplied
    /// [`GpuSolverBridge`](crate::gpu_bridge::GpuSolverBridge). Semantically equivalent to
    /// [`Self::step`] with `substep(_)` replaced by
    /// [`Self::substep_with_bridge`] inside the substep loop.
    ///
    /// # Byte-exact CPU parity
    ///
    /// When the bridge implementation is byte-exact vs the CPU
    /// `solve_contact_constraints`, a full `step_with_bridge`
    /// produces `self.bodies` and `self.contact_constraints` state
    /// byte-identical to what a CPU-only `step(dt)` call produces
    /// from the same initial state and configuration.
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn step_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
        dt: Fix128,
    ) {
        // The checks of `try_step` (a recorded fault, a non-positive `dt`, a
        // participant step rule or a per-body field that does not fit): when
        // one fails nothing runs, as `step` does. This path runs the XPBD
        // substep loop whatever the backend, so the step rules are checked
        // against the width it hands out, `dt / substeps`.
        let h = Self::divided_substep_width(dt, self.config.substeps);
        if !matches!(self.check_step(dt, h), Ok(true)) {
            return;
        }
        self.make_body_rotations_unit();
        let mut frozen = self.participant_flags();

        // Phase 0: Event frame lifecycle (contacts are cleared and re-detected
        // in every substep since 1.2.0, see `substep`)
        self.events.begin_frame();

        // Phase 0.5: Rebuild island connectivity from current joints
        self.islands.resize(self.bodies.len());
        self.islands.reset_unions();
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < self.bodies.len() && b < self.bodies.len() {
                self.islands.union(a, b);
            }
        }

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }

        // Phase 2 (collision detection) is per substep since 1.2.0, see `step`.

        // Phase 3: Substep loop with bridge-routed contact solve
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        let n = self.config.substeps;
        for i in 0..n {
            self.kinematic_substeps_left = n - i;
            let overflow_at_start = self.participants_begin_substep(i, n, substep_dt, &mut frozen);
            self.substep_with_bridge(bridge, substep_dt);
            self.participants_end_substep(overflow_at_start, &frozen);
        }
        self.kinematic_substeps_left = 0;

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

        // Phase 4.5: Leave every SDF collider attached to a body at that
        // body's final pose, which is what the queries between steps
        // (`sdf_contacts`, `sdf_ccd_hits`, the shape casts) read.
        self.sync_sdf_colliders();

        // Phase 5: End event frame
        self.events.end_frame();
    }

    /// v0.10.0 opt-in: run one substep with the contact-solve stage
    /// routed through a caller-supplied [`GpuSolverBridge`](crate::gpu_bridge::GpuSolverBridge). The
    /// integrate + distance-projection + joint + velocity-update
    /// stages stay on the CPU; only the inner PGS contact-solve
    /// iterations are dispatched through the bridge.
    ///
    /// # Byte-exact CPU parity
    ///
    /// When the bridge implementation is byte-exact vs the CPU
    /// `solve_contact_constraints` (which every ALICE-TRT
    /// `TrtSolverAdapter` release since v2.6.0 asserts on the
    /// 3-platform CI matrix), a full `substep_with_bridge` sequence
    /// produces `self.bodies` and `self.contact_constraints` state
    /// byte-identical to what a CPU-only `substep(dt)` call produces
    /// from the same initial state and configuration.
    ///
    /// Called on its own, this runs no participant: the participants of
    /// [`crate::world_participant`] are called by the step loops
    /// ([`Self::step_with_bridge`], [`Self::try_step`]) around each substep.
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn substep_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
        dt: Fix128,
    ) {
        // Callable on its own (outside `step_with_bridge`), so it is an entry
        // of its own for the rotations too.
        self.make_body_rotations_unit();
        self.apply_joint_motors(dt);
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.2. Collision detection on the predicted positions (Small Steps).
        self.clear_contacts();
        self.detect_collisions();

        // 1.5. Resolve SDF collisions (implicit surface contacts)
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }
        if !self.static_colliders.is_empty() {
            self.resolve_static_collisions(false);
        }

        // 2. Solve constraints (sequential). Distance stays CPU;
        //    contact routes through the bridge. The first iteration runs the
        //    hooks and modifiers once and solves; later iterations only solve,
        //    so a relative modifier does not compound with `iterations`.
        for iteration in 0..self.config.iterations {
            self.solve_distance_constraints(dt);
            if iteration == 0 {
                self.solve_contact_constraints_with_bridge(bridge);
            } else {
                self.solve_contact_pass_with_bridge(bridge);
            }
        }

        // 3. Solve joint constraints — route through the same bridge.
        self.solve_joints_with_bridge(bridge, dt);

        self.update_velocities(dt);
    }

    /// v0.12.0: run one joint-solve pass with the joint stage routed
    /// through a caller-supplied [`GpuSolverBridge`](crate::gpu_bridge::GpuSolverBridge). Extracts body
    /// positions, rotations and inverse masses into the shape the
    /// bridge expects, uploads them via `send_joints`, `send_body_state`
    /// and `send_body_rotations`, dispatches
    /// `dispatch_joint_solve_iteration(dt)`, reads back the
    /// post-solve positions via `recv_body_positions`, and writes
    /// them back into `self.bodies[i].position`.
    ///
    /// # Contract
    ///
    /// - The bridge must implement the v0.12.0 joint-solve methods
    ///   (`send_joints`, `send_body_rotations`,
    ///   `dispatch_joint_solve_iteration`); the default `panic!`
    ///   implementations surface backends that do not.
    /// - `dt` is the substep length used to compute
    ///   `compliance / (dt * dt)` inside the kernel.
    /// - A joint whose two ends are the same body is not uploaded
    ///   (the CPU solve skips it too, the body stays free); the
    ///   remaining joints keep their order.
    /// - When no joint remains (`self.joints` is empty or holds only
    ///   such joints), this method is a no-op and returns without
    ///   touching the bridge — useful for callers that
    ///   unconditionally route through a bridge and let this method
    ///   decide whether there's work to do.
    ///
    /// # Byte-exact CPU parity
    ///
    /// The CPU-side `solve_joints` runs each joint independently in
    /// index order; the bridge is required to reproduce that exact
    /// order and per-joint arithmetic. ALICE-TRT v3.1.0 satisfies
    /// this by dispatching a single-workgroup single-thread kernel
    /// that walks the uploaded joint list top to bottom.
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn solve_joints_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
        dt: Fix128,
    ) {
        // A joint whose two ends are the same body constrains nothing (a body
        // cannot move relative to itself); the CPU solve skips it, and a
        // backend walking the list would apply both sides' corrections to that
        // one body. Upload only the other joints, in their original order.
        let joints: Vec<crate::joint::Joint> = self
            .joints
            .iter()
            .filter(|j| {
                let (a, b) = j.bodies();
                a != b
            })
            .copied()
            .collect();
        if joints.is_empty() {
            return;
        }

        // ---- Extract per-body state ----
        let mut positions: Vec<[Fix128; 3]> = Vec::with_capacity(self.bodies.len());
        let mut inv_masses: Vec<Fix128> = Vec::with_capacity(self.bodies.len());
        let mut rotations: Vec<[Fix128; 4]> = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            positions.push([body.position.x, body.position.y, body.position.z]);
            inv_masses.push(body.inv_mass);
            rotations.push([
                body.rotation.x,
                body.rotation.y,
                body.rotation.z,
                body.rotation.w,
            ]);
        }

        // ---- Stage B: route through bridge ----
        bridge.send_joints(&joints);
        bridge.send_body_state(&positions, &inv_masses);
        bridge.send_body_rotations(&rotations);
        bridge.dispatch_joint_solve_iteration(dt);

        let mut updated_positions = positions;
        bridge.recv_body_positions(&mut updated_positions);

        // ---- Write back positions ----
        for (i, pos) in updated_positions.iter().enumerate() {
            self.bodies[i].position = crate::math::Vec3Fix::new(pos[0], pos[1], pos[2]);
        }
    }

    /// v0.12.0 private helper: solve joints either through the
    /// installed GPU solver bridge (if any) or via the CPU
    /// `solve_joints` fallback. Used by [`Self::step`] / substep
    /// entry points to auto-route without every caller needing to
    /// know about the bridge.
    fn solve_joints_dispatch(&mut self, dt: Fix128) {
        if self.joints.is_empty() {
            return;
        }
        #[cfg(feature = "gpu-solver-bridge")]
        if let Some(mut bridge) = self.gpu_solver_bridge.take() {
            self.solve_joints_with_bridge(bridge.as_mut(), dt);
            self.gpu_solver_bridge = Some(bridge);
            return;
        }
        solve_joints(&self.joints, &mut self.bodies, dt);
    }

    /// Add an SDF collider to the world
    ///
    /// A collider attached to a body ([`SdfCollider::new_dynamic`]) is placed
    /// at that body's current pose when the body exists, and follows the body
    /// from then on: every step copies the body's pose into it before SDF
    /// overlap is resolved and again at the end of the step.
    pub fn add_sdf_collider(&mut self, mut collider: SdfCollider) -> usize {
        collider.sync_to_body(&self.bodies);
        let idx = self.sdf_colliders.len();
        self.sdf_colliders.push(collider);
        idx
    }

    /// Copy each body's pose into the SDF colliders attached to it
    /// ([`crate::sdf_collider::sync_dynamic_sdf_colliders`]); static colliders
    /// are not touched.
    fn sync_sdf_colliders(&mut self) {
        crate::sdf_collider::sync_dynamic_sdf_colliders(&mut self.sdf_colliders, &self.bodies);
    }

    /// Remove an SDF collider by index
    pub fn remove_sdf_collider(&mut self, idx: usize) -> Option<SdfCollider> {
        if idx < self.sdf_colliders.len() {
            Some(self.sdf_colliders.remove(idx))
        } else {
            None
        }
    }

    /// Set the collision radius for body-vs-SDF queries
    pub fn set_sdf_collision_radius(&mut self, radius: Fix128) {
        self.sdf_collision_radius = radius;
    }

    /// Resolve SDF collisions by directly correcting body positions.
    ///
    /// SDF colliders are treated as immovable surfaces (infinite mass).
    /// Each penetrating body is pushed out along the SDF gradient.
    ///
    /// When `parallel` feature is enabled, bodies are processed in parallel via Rayon.
    #[cfg(feature = "std")]
    fn resolve_sdf_collisions(&mut self) {
        // A collider attached to a body is resolved at that body's current
        // (predicted) pose.
        self.sync_sdf_colliders();
        let sdf_colliders = &self.sdf_colliders;
        let colliders = &self.body_colliders;
        let collision_radius = self.sdf_collision_radius;

        let push_out = |idx: usize, body: &mut RigidBody| {
            if body.is_static() || body.is_sensor {
                return;
            }
            let collider = colliders.get(idx).and_then(Option::as_ref);
            for sdf in sdf_colliders {
                // A body is never pushed out of its own field.
                if sdf.body_index == idx {
                    continue;
                }
                if let Some(contact) = sdf_contact_of(collider, body, collision_radius, sdf) {
                    body.position = body.position + contact.normal * contact.depth;
                }
            }
        };

        #[cfg(feature = "parallel")]
        {
            let parked: &[bool] = if self.park.active {
                &self.park.parked
            } else {
                &[]
            };
            self.stage_work.resolution_bodies += self.unparked_count();
            self.bodies
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, body)| {
                    if !parked.get(idx).copied().unwrap_or(false) {
                        push_out(idx, body);
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            let count = if self.park.active {
                self.park.awake.len()
            } else {
                self.bodies.len()
            };
            for k in 0..count {
                let idx = if self.park.active {
                    self.park.awake[k]
                } else {
                    k
                };
                self.stage_work.resolution_bodies += 1;
                push_out(idx, &mut self.bodies[idx]);
            }
        }
    }

    /// The contacts of the bodies with the SDF colliders, as `(body index, contact)`:
    /// what [`step`](Self::step) would push each body out of, without moving
    /// anything. A body with a shape or a compound is tested as that collider, any
    /// other body as a sphere of [`sdf_collision_radius`](Self::sdf_collision_radius);
    /// static bodies, sensors and an SDF attached to the body itself are skipped.
    /// An SDF attached to a body is queried at the pose it was given when it was
    /// added or at the end of the last step; after moving a body by hand, call
    /// [`crate::sdf_collider::sync_dynamic_sdf_colliders`] on
    /// [`sdf_colliders`](Self::sdf_colliders) first.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn sdf_contacts(&self) -> Vec<(usize, crate::collider::Contact)> {
        let mut contacts = Vec::new();
        for (idx, body) in self.bodies.iter().enumerate() {
            if body.is_static() || body.is_sensor {
                continue;
            }
            let collider = self.body_colliders.get(idx).and_then(Option::as_ref);
            for sdf in &self.sdf_colliders {
                if sdf.body_index == idx {
                    continue;
                }
                if let Some(c) = sdf_contact_of(collider, body, self.sdf_collision_radius, sdf) {
                    contacts.push((idx, c));
                }
            }
        }
        contacts
    }

    /// The normal force each contact of the last step carries, as
    /// `(contact point, normal, force)` — the input [`crate::contact_viz`] draws.
    ///
    /// `dt` is the frame step passed to [`step`](Self::step). The solver applies a
    /// separation `λ` (`cached_lambda`) along the normal in the last substep, of
    /// length `h = dt / substeps`, which for bodies of inverse masses `wₐ, w_b` is
    /// the displacement of a force `F` over `h`: `λ = (wₐ + w_b)·F·h²`, so
    /// `F = λ / ((wₐ + w_b)·h²)`. A body resting under gravity then carries its
    /// weight: `m·g`. The normal points from B to A and the force acts on A. Contacts
    /// between two immovable bodies carry none and are left out. A body that has gone
    /// to sleep (about a minute of rest) is not solved, so its contacts are not there
    /// to report.
    #[must_use]
    pub fn contact_forces(&self, dt: Fix128) -> Vec<(Vec3Fix, Vec3Fix, Fix128)> {
        let h2 = self.substep_dt(dt) * self.substep_dt(dt);
        self.contact_constraints
            .iter()
            .filter_map(|c| self.contact_force(c, h2))
            .collect()
    }

    /// The length `h` of one substep of a frame of `dt`.
    fn substep_dt(&self, dt: Fix128) -> Fix128 {
        dt / Fix128::from_int(self.config.substeps.max(1) as i64)
    }

    /// `(point, normal, force)` of one contact, or `None` when neither body can move.
    fn contact_force(
        &self,
        c: &ContactConstraint,
        h2: Fix128,
    ) -> Option<(Vec3Fix, Vec3Fix, Fix128)> {
        let w = self.bodies.get(c.body_a)?.inv_mass + self.bodies.get(c.body_b)?.inv_mass;
        if w < W_SUM_EPSILON || h2.is_zero() {
            return None;
        }
        Some((
            c.contact.point_a,
            c.contact.normal,
            c.cached_lambda / (w * h2),
        ))
    }

    /// Arrows for the normal forces of the last step's contacts
    /// ([`contact_forces`](Self::contact_forces) through
    /// [`crate::contact_viz::generate_contact_arrows`]).
    #[must_use]
    pub fn contact_arrows(&self, dt: Fix128) -> Vec<crate::contact_viz::ContactArrow> {
        crate::contact_viz::generate_contact_arrows(&self.contact_forces(dt))
    }

    /// Two tangential arrows of magnitude `μ·F` for each contact of the last step,
    /// with the friction coefficient of that contact (they differ between material
    /// pairs).
    #[must_use]
    pub fn contact_friction_arrows(&self, dt: Fix128) -> Vec<crate::contact_viz::ContactArrow> {
        let h2 = self.substep_dt(dt) * self.substep_dt(dt);
        self.contact_constraints
            .iter()
            .filter_map(|c| Some((self.contact_force(c, h2)?, c.friction)))
            .flat_map(|(force, mu)| crate::contact_viz::generate_friction_arrows(&[force], mu))
            .collect()
    }

    /// A friction cone (half-angle `atan μ`, height the normal force) for each
    /// contact of the last step, with that contact's friction coefficient.
    #[must_use]
    pub fn contact_friction_cones(&self, dt: Fix128) -> Vec<crate::contact_viz::FrictionCone> {
        let h2 = self.substep_dt(dt) * self.substep_dt(dt);
        self.contact_constraints
            .iter()
            .filter_map(|c| Some((self.contact_force(c, h2)?, c.friction)))
            .flat_map(|(force, mu)| crate::contact_viz::generate_friction_cones(&[force], mu))
            .collect()
    }

    /// Add an immovable collision surface — a plane, a height field or a triangle
    /// mesh — and return its index. See [`crate::static_collider`].
    pub fn add_static_collider(
        &mut self,
        collider: crate::static_collider::StaticCollider,
    ) -> usize {
        let idx = self.static_colliders.len();
        self.park_generation = self.park_generation.wrapping_add(1);
        self.static_colliders.push(collider);
        idx
    }

    /// Remove a static collider by index, returning it, or `None` when the index
    /// is out of range. Later colliders shift down by one.
    pub fn remove_static_collider(
        &mut self,
        idx: usize,
    ) -> Option<crate::static_collider::StaticCollider> {
        if idx < self.static_colliders.len() {
            self.park_generation = self.park_generation.wrapping_add(1);
            Some(self.static_colliders.remove(idx))
        } else {
            None
        }
    }

    /// Number of static colliders.
    #[must_use]
    pub fn static_collider_count(&self) -> usize {
        self.static_colliders.len()
    }

    /// Resolve every non-static, non-sensor body's collision sphere against the
    /// static colliders, in order, pushing it out along each contact normal.
    ///
    /// The sphere is the body's collision radius, or the world's default
    /// ([`Self::set_sdf_collision_radius`]) for a body without one.
    ///
    /// `remove_approach`: also drop the part of the body's velocity that
    /// points into the surface (an inelastic contact). The XPBD substep
    /// passes `false` — its `update_velocities` turns the push into velocity
    /// — and the TGS path passes `true`, because TGS keeps the velocity it
    /// integrated and a pushed-out body would otherwise keep falling into
    /// the surface.
    fn resolve_static_collisions(&mut self, remove_approach: bool) {
        let default_radius = self.sdf_collision_radius;
        let count = self.stage_count();
        for k in 0..count {
            let i = self.stage_body(k);
            self.stage_work.resolution_bodies += 1;
            let body = &mut self.bodies[i];
            if body.is_static() || body.is_sensor {
                continue;
            }
            let radius = self
                .body_collision_radii
                .get(i)
                .and_then(|r| *r)
                .unwrap_or(default_radius);
            for collider in &self.static_colliders {
                if let Some(contact) = collider.collide_sphere(body.position, radius) {
                    body.position = body.position + contact.normal * contact.depth;
                    if remove_approach {
                        let vn = body.velocity.dot(contact.normal);
                        if vn < Fix128::ZERO {
                            body.velocity = body.velocity - contact.normal * vn;
                        }
                    }
                }
            }
        }
    }

    /// Select the broad-phase. Every kind gives bit-identical simulations (see
    /// [`Broadphase`]); switching drops the persistent tree and the
    /// [`Broadphase::Hybrid`] layers, which are rebuilt on the next step if
    /// that kind is chosen.
    pub fn set_broadphase(&mut self, kind: Broadphase) {
        self.park_generation = self.park_generation.wrapping_add(1);
        self.broadphase = kind;
        self.broadphase_reset();
    }

    /// The broad-phase in use.
    #[must_use]
    pub fn broadphase(&self) -> Broadphase {
        self.broadphase
    }

    /// What the [`Broadphase::DynamicTree`] tree holds: how many bodies have a
    /// proxy and how tall the tree is. Both are 0 until a step has run with that
    /// broad-phase.
    #[must_use]
    pub fn broadphase_stats(&self) -> BroadphaseStats {
        BroadphaseStats {
            proxies: self.broadphase_tree.proxy_count(),
            height: self.broadphase_tree.height(),
        }
    }

    /// The fattened box the [`Broadphase::DynamicTree`] tree keeps for a body, or
    /// `None` when the body has no proxy (no collision radius, or no step yet).
    #[must_use]
    pub fn broadphase_proxy_aabb(&self, body_idx: usize) -> Option<AABB> {
        let proxy = (*self.broadphase_proxies.get(body_idx)?)?;
        if self.broadphase_tree.user_data(proxy) as usize != body_idx {
            return None;
        }
        Some(self.broadphase_tree.get_aabb(proxy))
    }

    /// Forget the persistent tree and the hybrid layers: the next
    /// [`Broadphase::DynamicTree`] or [`Broadphase::Hybrid`] step rebuilds them
    /// from the bodies. Called when the broad-phase is switched.
    fn broadphase_reset(&mut self) {
        self.broadphase_tree = crate::dynamic_bvh::DynamicAabbTree::new();
        self.broadphase_proxies.clear();
        self.broadphase_hybrid = BroadphaseHybrid::new();
    }

    /// The box the broad-phase keeps for body `i`, whose collision radius is
    /// `radius`: the closed-form world box of its collider when it carries one
    /// ([`crate::body_collider::BodyCollider::world_aabb`]), clipped to the cube
    /// of the collision sphere, otherwise that cube. Both contain the body's
    /// position, so the clip is never empty. Every pair the narrow-phase can
    /// report has overlapping boxes: a pair involving a collider is decided by
    /// GJK on the collider (and on the plain side's sphere), which lies inside
    /// these boxes.
    fn broadphase_box(&self, i: usize, radius: Fix128) -> AABB {
        let pos = self.bodies[i].position;
        let cube = AABB::from_center_half(pos, Vec3Fix::new(radius, radius, radius));
        match self.body_colliders.get(i).and_then(Option::as_ref) {
            Some(collider) => {
                let tight = collider.world_aabb(pos, self.bodies[i].rotation);
                let max = |x: Fix128, y: Fix128| if x > y { x } else { y };
                let min = |x: Fix128, y: Fix128| if x < y { x } else { y };
                AABB {
                    min: Vec3Fix::new(
                        max(tight.min.x, cube.min.x),
                        max(tight.min.y, cube.min.y),
                        max(tight.min.z, cube.min.z),
                    ),
                    max: Vec3Fix::new(
                        min(tight.max.x, cube.max.x),
                        min(tight.max.y, cube.max.y),
                        min(tight.max.z, cube.max.z),
                    ),
                }
            }
            None => cube,
        }
    }

    /// The sorted candidate pairs of bodies whose boxes may overlap, from the
    /// primitives (body index + tight box) of every body with a collision radius.
    fn broadphase_pairs(&mut self, primitives: Vec<BvhPrimitive>) -> Vec<(u32, u32)> {
        match self.broadphase {
            Broadphase::Bvh => LinearBvh::build(primitives).find_pairs(),
            Broadphase::DynamicTree => {
                let n = self.bodies.len();
                // A proxy belongs to a body *index* (its user data), and re-boxing it
                // each step follows whichever body now has that index, so removing
                // or restoring bodies needs no rebuild; only the proxies of indices
                // that no longer exist have to go.
                for proxy in self
                    .broadphase_proxies
                    .drain(n.min(self.broadphase_proxies.len())..)
                    .flatten()
                {
                    self.broadphase_tree.remove(proxy);
                }
                self.broadphase_proxies.resize(n, None);
                let mut present = vec![false; n];
                for p in &primitives {
                    let i = p.index as usize;
                    present[i] = true;
                    match self.broadphase_proxies[i] {
                        Some(proxy) => {
                            self.broadphase_tree.update(proxy, p.aabb);
                        }
                        None => {
                            self.broadphase_proxies[i] =
                                Some(self.broadphase_tree.insert(p.aabb, p.index));
                        }
                    }
                }
                // A body that lost its collision radius leaves the tree.
                for (i, here) in present.iter().enumerate() {
                    if !here {
                        if let Some(proxy) = self.broadphase_proxies[i].take() {
                            self.broadphase_tree.remove(proxy);
                        }
                    }
                }
                self.broadphase_tree.find_pairs()
            }
            Broadphase::Hybrid => {
                let hybrid = &mut self.broadphase_hybrid;
                hybrid.clear_dynamic();
                for p in &primitives {
                    let is_static = self.bodies[p.index as usize].is_static();
                    hybrid.insert_dynamic(p.index, p.aabb, is_static);
                }
                hybrid.build_dynamic();
                let mut pairs = Vec::new();
                self.stage_work.broadphase_box_tests += hybrid.query_pairs(&mut pairs);
                pairs
            }
        }
    }

    /// The sorted candidate pairs while parking is in effect: the pairs among the
    /// unparked `primitives` from a [`LinearBvh`] over them, plus every parked
    /// body whose proxy overlaps one of them, from `park.tree`. Pairs of two
    /// parked bodies are left out; the narrow-phase drops pairs of two sleeping
    /// bodies anyway, so the contacts and their order are those of
    /// [`Self::broadphase_pairs`] over every body.
    fn parked_broadphase_pairs(&mut self, primitives: Vec<BvhPrimitive>) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for p in &primitives {
            let i = p.index;
            self.park.tree.query_callback(&p.aabb, |j| {
                pairs.push(if i < j { (i, j) } else { (j, i) });
            });
        }
        if primitives.len() >= 2 {
            pairs.extend(LinearBvh::build(primitives).find_pairs());
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    // ── Automatic Collision Detection ─────────────────────────────────

    /// Detect collisions between bodies with collision radii.
    ///
    /// The broad-phase boxes are the collider's tight world box for a body that
    /// carries one and the cube of the collision sphere otherwise; the
    /// narrow-phase is exact sphere-sphere for two plain bodies and GJK/EPA for a
    /// pair involving a collider (the plain side as its sphere).
    /// Generates contact constraints and events automatically.
    /// Bodies without a collision radius are skipped.
    #[allow(clippy::too_many_lines, clippy::items_after_statements)]
    fn detect_collisions(&mut self) {
        let n = self.bodies.len();
        if n < 2 {
            return;
        }

        // Build BVH from bodies that have collision radii (the unparked ones while
        // parking is in effect: parked bodies sit in `park.tree`)
        let mut primitives = Vec::new();
        let count = self.stage_count();
        for k in 0..count {
            let i = self.stage_body(k);
            if let Some(radius) = self.body_collision_radii.get(i).and_then(|r| *r) {
                primitives.push(BvhPrimitive {
                    aabb: self.broadphase_box(i, radius),
                    index: i as u32,
                    morton: 0,
                });
            }
        }
        self.stage_work.broadphase_primitives += primitives.len() as u64;

        let parked_with_radius = if self.park.active {
            self.park.proxy_live
        } else {
            0
        };
        if primitives.len() + parked_with_radius < 2 {
            return;
        }

        let pairs = if self.park.active {
            self.parked_broadphase_pairs(primitives)
        } else {
            self.broadphase_pairs(primitives)
        };
        self.stage_work.broadphase_pairs += pairs.len() as u64;

        // Velocity of a body as the pre-skip step saw it here: a parked body
        // still carries the force-field kick of phase 1 in the first substep
        // (the velocity derivation zeroes it at the end of that substep).
        let velocity_of = |i: usize| -> Vec3Fix {
            let body = &self.bodies[i];
            if self.park.first_substep
                && !self.force_fields.is_empty()
                && self.park.is_parked(i)
                && !body.is_static()
            {
                crate::force::force_field_velocity(&self.force_fields, i, body, self.park.frame_dt)
            } else {
                body.velocity
            }
        };

        // Collect results to avoid borrow conflicts
        struct ContactInfo {
            body_a: usize,
            body_b: usize,
            contact: Contact,
            normal: Vec3Fix,
            point: Vec3Fix,
            depth: Fix128,
            rel_vel: Fix128,
            is_sensor: bool,
        }
        let mut results: Vec<ContactInfo> = Vec::new();
        let mut radius_overflow = false;

        for (a32, b32) in pairs {
            let a = a32 as usize;
            let b = b32 as usize;
            if a >= n || b >= n {
                continue;
            }

            // Sphere-sphere narrow phase (safe indexing for deserialization robustness)
            let radius_a = self
                .body_collision_radii
                .get(a)
                .and_then(|r| *r)
                .unwrap_or(Fix128::ZERO);
            let radius_b = self
                .body_collision_radii
                .get(b)
                .and_then(|r| *r)
                .unwrap_or(Fix128::ZERO);

            // Contact normal points from B to A (`Contact::normal` contract, same
            // as the EPA / SDF paths and what `solve_contact_constraints` /
            // `update_velocities` assume: A is moved along +n, B along -n).
            // Before 1.2.0 this path used `pos_b - pos_a` (A → B), so every
            // sphere-sphere contact pushed the bodies *into* each other and a
            // plain head-on collision accelerated both bodies (4 m/s → 114 m/s
            // in 4 frames). `tests/analytic_physics.rs::head_on_collision_*`.
            let delta = self.bodies[a].position - self.bodies[b].position;
            // 半径和が `|·| ≥ 2⁶³` だと wrap して負になり、どの距離とも比べられない
            // (`velocity_of` が `self` を借りているので、印は loop の後で立てる)
            let Some(combined_radius) = radius_a.checked_add(radius_b) else {
                radius_overflow = true;
                continue;
            };
            // Squared-distance early out *first*: the BVH candidate set is a
            // superset of the overlapping pairs (49k candidates for 2.7k contacts
            // on the 1000-sphere grid) and the quantised leaf AABBs cannot be
            // tightened without a BVH API change, so every candidate pays only
            // 4 checked multiplies + 2 checked adds + a compare here; the filter / static / sleep lookups
            // and the sqrt + 3 divisions of `normalize_with_length` run only for
            // real overlaps. Same `dist < combined_radius` decision (both sides
            // exact for |delta| < 2^31), same contact order.
            //
            // `|delta|` か半径和が `2³¹·⁵ ≈ 3.04e9` 以上だと 2 乗が wrap して
            // 重なりを見逃す (両辺が黙って別の値になる) 2 乗がどちらも範囲内
            // なら従来の比較のまま (bit 不変)、どちらかが範囲外の対だけ 2 乗を
            // 経ない長さで比べ、法線と距離もその長さから作る (`scaled`)
            let mut scaled: Option<(Vec3Fix, Fix128)> = None;
            // A pair involving a collider is decided by GJK on the collider, with
            // the other body's collider or, for a plain body, its sphere; such a
            // pair can overlap with coincident centres, where the sphere path
            // cannot give a normal. Its early out is the overlap of the two
            // broad-phase boxes (the collider's tight box, the plain body's
            // sphere cube), which contain everything GJK looks at.
            let collider_a = self.body_colliders.get(a).and_then(Option::as_ref);
            let collider_b = self.body_colliders.get(b).and_then(Option::as_ref);
            if collider_a.is_none() && collider_b.is_none() {
                let squares = delta
                    .checked_length_squared()
                    .zip(combined_radius.checked_mul(combined_radius));
                match squares {
                    Some((d2, r2)) => {
                        if d2 >= r2 || d2.is_zero() {
                            continue;
                        }
                    }
                    None => {
                        // 中心距離が `≥ 2⁶³` (表せない) なら、どの半径和
                        // (`< 2⁶³`、上で検査済) とも重ならない
                        let Some(dist) = delta.checked_length_scaled() else {
                            continue;
                        };
                        if dist >= combined_radius || dist.is_zero() {
                            continue;
                        }
                        let Some(normal) = delta.try_normalize_scaled() else {
                            continue;
                        };
                        scaled = Some((normal, dist));
                    }
                }
            } else if !self
                .broadphase_box(a, radius_a)
                .intersects(&self.broadphase_box(b, radius_b))
            {
                continue;
            }

            // Filter check
            let filter_a = self
                .body_filters
                .get(a)
                .copied()
                .unwrap_or(CollisionFilter::DEFAULT);
            let filter_b = self
                .body_filters
                .get(b)
                .copied()
                .unwrap_or(CollisionFilter::DEFAULT);
            if !CollisionFilter::can_collide(&filter_a, &filter_b) {
                continue;
            }

            // Skip static-static
            if self.bodies[a].is_static() && self.bodies[b].is_static() {
                continue;
            }

            // Skip if both sleeping
            if self.islands.is_sleeping(a) && self.islands.is_sleeping(b) {
                continue;
            }

            // The boxes overlap, which is necessary but not sufficient for the
            // solids to: let the shapes decide.
            if collider_a.is_some() || collider_b.is_some() {
                let pose_a = (self.bodies[a].position, self.bodies[a].rotation);
                let pose_b = (self.bodies[b].position, self.bodies[b].rotation);
                let hit = match (collider_a, collider_b) {
                    (Some(ca), Some(cb)) => {
                        crate::body_collider::contact_between(ca, pose_a, cb, pose_b)
                    }
                    (Some(ca), None) => crate::body_collider::contact_with_sphere(
                        ca,
                        pose_a,
                        crate::collider::Sphere::new(pose_b.0, radius_b),
                        true,
                    ),
                    (None, Some(cb)) => crate::body_collider::contact_with_sphere(
                        cb,
                        pose_b,
                        crate::collider::Sphere::new(pose_a.0, radius_a),
                        false,
                    ),
                    (None, None) => None,
                };
                if let Some(contact) = hit {
                    if contact.depth > Fix128::ZERO {
                        let rel_vel = (velocity_of(a) - velocity_of(b)).dot(contact.normal);
                        results.push(ContactInfo {
                            body_a: a,
                            body_b: b,
                            contact,
                            normal: contact.normal,
                            point: contact.point_a,
                            depth: contact.depth,
                            rel_vel,
                            is_sensor: self.bodies[a].is_sensor || self.bodies[b].is_sensor,
                        });
                    }
                }
                continue;
            }

            let (normal, dist) = scaled.unwrap_or_else(|| delta.normalize_with_length());

            if dist < combined_radius && !dist.is_zero() {
                let depth = combined_radius - dist;
                let point_a = self.bodies[a].position - normal * radius_a;
                let point_b = self.bodies[b].position + normal * radius_b;
                // Approach speed along the normal: negative while closing.
                let rel_vel = (velocity_of(a) - velocity_of(b)).dot(normal);
                let is_sensor = self.bodies[a].is_sensor || self.bodies[b].is_sensor;

                let contact = Contact {
                    depth,
                    normal,
                    point_a,
                    point_b,
                };

                results.push(ContactInfo {
                    body_a: a,
                    body_b: b,
                    contact,
                    normal,
                    point: point_a,
                    depth,
                    rel_vel,
                    is_sensor,
                });
            }
        }

        if radius_overflow {
            self.note_rigid_overflow();
        }

        // Apply results
        for info in results {
            // Report contact event
            self.events.report_contact(
                info.body_a,
                info.body_b,
                info.normal,
                info.point,
                info.depth,
                info.rel_vel,
            );

            // Wake a *sleeping* island touched by this contact. Waking
            // unconditionally (pre-1.2.0) reset `idle_frames` of every body in
            // resting contact on every detection, so a stack could never fall
            // asleep, and `wake_island` walks all bodies — O(contacts × bodies)
            // per detection (3.7 ms of a 6 ms step at 2700 contacts / 1000 bodies).
            for x in [info.body_a, info.body_b] {
                if !self.islands.is_sleeping(x) {
                    continue;
                }
                if self.park.is_parked(x) {
                    // Parked bodies are on no joint, so their island is
                    // themselves: same outcome as `wake_island`, without its
                    // walk over every body.
                    self.islands.wake_body(x);
                    self.park.unpark(x, &mut self.stage_work);
                } else {
                    self.islands.wake_island(x);
                }
            }

            if info.is_sensor {
                self.events.report_trigger(info.body_a, info.body_b);
            } else {
                self.add_contact_with_material(info.body_a, info.body_b, info.contact);
            }
        }
    }

    /// Get body by index
    #[inline]
    #[must_use]
    pub fn get_body(&self, idx: usize) -> Option<&RigidBody> {
        self.bodies.get(idx)
    }

    /// Get mutable body by index
    #[inline]
    pub fn get_body_mut(&mut self, idx: usize) -> Option<&mut RigidBody> {
        self.bodies.get_mut(idx)
    }

    /// 範囲外の積が起きた step があったか (sticky、WM-01 / doctrine B-12)
    ///
    /// `true` は **この world の軌道が信用できない** ことを意味する
    /// doctrine §3 の 3 値で言えば `undecided` — 探索では「この枝は未決定」
    ///
    /// # なぜ値ベースの検出器では足りないか
    ///
    /// `Fix128::mul` の積が範囲外で 2 の冪だと `hi` が厳密に 0 になり
    /// (`2⁴⁰ × 2⁴⁰`)、`position += velocity*dt` が 0 加算 →
    /// `update_velocities` が速度も 0 に上書き →
    /// **「原点で完全に静止した自己整合な状態」** に落ちる
    /// ⚠️ **静止は物理的に妥当なのでどの不変条件でも red にならない**
    ///
    /// # sticky である理由
    ///
    /// 1 度立ったら `step` を重ねても落ちない 落ちると探索が
    /// 「未決定だった枝」を後の step で決定済と誤認する
    ///
    /// ⚠️ **flag は状態の一部**なので [`Self::serialize_state`] の被覆に
    /// 入れる必要がある (入れないと巻き戻した先で `undecided` が消える、
    /// doctrine B-12 の指摘) — **format v2 で収録済** (world ごと 1 byte、
    /// [`Self::STATE_VERSION`] 参照)
    #[must_use]
    pub const fn overflow_detected(&self) -> bool {
        self.overflow_detected
    }

    /// 剛体の経路が範囲外の値を踏んだことを [`Self::overflow_detected`] に立てる
    ///
    /// sticky flag を立てるのはこの関数だけ (積分の `v·dt` と位置の加算、
    /// 速度導出、球の接触の距離と半径和、角速度の大きさ、TGS の積分、
    /// TGS の関節投影の速度への繰り越し) 落とす経路は [`Self::reset_world`] と
    /// 状態の復元だけ
    #[inline]
    pub(crate) fn note_rigid_overflow(&mut self) {
        self.overflow_detected = true;
    }

    /// [`Self::serialize_state`] blob の magic
    ///
    /// ⚠️ 旧 format は先頭が body 数だったので、magic を検査しないと
    /// **新実装が旧 blob を誤って解釈する**
    pub const STATE_MAGIC: [u8; 4] = *b"APHY";

    /// [`Self::serialize_state`] blob の format version
    ///
    /// v1 = header 12 + body ごと 208 + body ごと sleep 5
    /// v2 = v1 + **world ごと overflow flag 1 byte** (doctrine B-12)
    /// v3 = v2 + **world ごと population fingerprint 8 byte**
    /// ([`Self::population_fingerprint`]、rollback の body 数不一致検査強化)
    ///
    /// 被覆を足す時はここを上げ、[`Self::deserialize_state`] で分岐する
    /// (不一致を silent に読み替えない — 旧 version の blob は `false` で拒否、
    /// v1 / v2 の blob は v3 実装に対して version 不一致で拒否される、
    /// 既存の v1→v2 bump と同じ方針)
    pub const STATE_VERSION: u16 = 3;

    /// Serialize world state (for rollback netcode).
    ///
    /// # 被覆 (v1、2026-09-30)
    ///
    /// | 範囲 | 内容 |
    /// |---|---|
    /// | `[0..4)` | magic [`Self::STATE_MAGIC`] (`b"APHY"`) |
    /// | `[4..6)` | version u16 = [`Self::STATE_VERSION`] |
    /// | `[6..8)` | reserved u16 = 0 (将来の flag 用、8 byte 境界揃え) |
    /// | `[8..12)` | body 数 u32 |
    /// | `[12..)` | body ごとに 208 byte = position 48 + velocity 48 + rotation 64 + angular_velocity 48 |
    /// | 続き | body ごとに 5 byte = [`SleepState`] u8 + `idle_frames` u32 |
    /// | 続き | world ごと overflow flag 1 byte (v2) |
    /// | 続き | world ごと population fingerprint u64 (v3、[`Self::population_fingerprint`]) |
    ///
    /// ⚠️ **`SleepState` + `idle_frames` を v1 で被覆に入れた** (WM-08)
    /// 旧 format は body の運動状態だけを持ち、`deserialize_state` 末尾の
    /// `IslandManager::new` が sleep 状態を 0 に戻していたため、
    /// **巻き戻した先で眠るタイミングがずれて軌道が割れた**
    /// (`tests/wm08_state_coverage.rs` が対照実験付きで red を実測してから実装)
    ///
    /// ⚠️ **magic と version を入れた理由**: 旧 blob は先頭が body 数なので、
    /// 検査しないと**新実装が旧 blob を誤って解釈する** magic 不一致 /
    /// version 不一致は `deserialize_state` が `false` を返して明示的に拒否する
    ///
    /// 依然として保存 (= 復元) しないもの: constraints / joints / force
    /// fields / collision radii / filters / materials (rollback netcode
    /// では game state から毎 frame 再構築される前提) ⚠️ **v3 の population
    /// fingerprint はこれらの値を検査にのみ使う** (復元はしない) ので
    /// 「保存しない」契約自体は変わらない
    ///
    /// ⚠️ [`crate::netcode::SimulationChecksum`] は **本 blob の全 byte から
    /// 導出**される (被覆の単一源化、B-13) ので、ここに state を足すと
    /// checksum も自動で追従する
    #[must_use]
    pub fn serialize_state(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header: magic + version + reserved
        data.extend_from_slice(&Self::STATE_MAGIC);
        data.extend_from_slice(&Self::STATE_VERSION.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());

        // Body count
        let count = self.bodies.len() as u32;
        data.extend_from_slice(&count.to_le_bytes());

        // Body states
        for body in &self.bodies {
            // Position (3 x Fix128 = 48 bytes)
            data.extend_from_slice(&body.position.x.hi.to_le_bytes());
            data.extend_from_slice(&body.position.x.lo.to_le_bytes());
            data.extend_from_slice(&body.position.y.hi.to_le_bytes());
            data.extend_from_slice(&body.position.y.lo.to_le_bytes());
            data.extend_from_slice(&body.position.z.hi.to_le_bytes());
            data.extend_from_slice(&body.position.z.lo.to_le_bytes());

            // Velocity (3 x Fix128 = 48 bytes)
            data.extend_from_slice(&body.velocity.x.hi.to_le_bytes());
            data.extend_from_slice(&body.velocity.x.lo.to_le_bytes());
            data.extend_from_slice(&body.velocity.y.hi.to_le_bytes());
            data.extend_from_slice(&body.velocity.y.lo.to_le_bytes());
            data.extend_from_slice(&body.velocity.z.hi.to_le_bytes());
            data.extend_from_slice(&body.velocity.z.lo.to_le_bytes());

            // Rotation (4 x Fix128 = 64 bytes)
            data.extend_from_slice(&body.rotation.x.hi.to_le_bytes());
            data.extend_from_slice(&body.rotation.x.lo.to_le_bytes());
            data.extend_from_slice(&body.rotation.y.hi.to_le_bytes());
            data.extend_from_slice(&body.rotation.y.lo.to_le_bytes());
            data.extend_from_slice(&body.rotation.z.hi.to_le_bytes());
            data.extend_from_slice(&body.rotation.z.lo.to_le_bytes());
            data.extend_from_slice(&body.rotation.w.hi.to_le_bytes());
            data.extend_from_slice(&body.rotation.w.lo.to_le_bytes());

            // Angular velocity (3 x Fix128 = 48 bytes)
            data.extend_from_slice(&body.angular_velocity.x.hi.to_le_bytes());
            data.extend_from_slice(&body.angular_velocity.x.lo.to_le_bytes());
            data.extend_from_slice(&body.angular_velocity.y.hi.to_le_bytes());
            data.extend_from_slice(&body.angular_velocity.y.lo.to_le_bytes());
            data.extend_from_slice(&body.angular_velocity.z.hi.to_le_bytes());
            data.extend_from_slice(&body.angular_velocity.z.lo.to_le_bytes());
        }

        // Sleep 状態 (body ごとに 5 byte、WM-08)
        //
        // ⚠️ `sleep_data` は body 数に合わせて resize されている前提だが、
        // 短い場合は既定 (Awake / 0) で埋める — blob の長さを body 数だけで
        // 決められるようにして、読み側の長さ検査を単純に保つ
        for i in 0..self.bodies.len() {
            let sd = self
                .islands
                .sleep_data
                .get(i)
                .copied()
                .unwrap_or_else(SleepData::new);
            data.push(match sd.state {
                SleepState::Awake => 0,
                SleepState::Sleeping => 1,
            });
            data.extend_from_slice(&sd.idle_frames.to_le_bytes());
        }

        // overflow flag (world ごと 1 byte、v2、doctrine B-12)
        //
        // ⚠️ **flag は状態の一部** — blob に入れないと overflow した枝を
        // 巻き戻した先で `undecided` が消えて B-12 が目的を達成しない
        data.push(u8::from(self.overflow_detected));

        // population fingerprint (world ごと 8 byte、v3)
        //
        // ⚠️ **位置 / 速度 / 回転 / sleep は含めない** ([`Self::population_fingerprint`]
        // 参照) — blob 自身が復元する量なので、ここに混ぜると検査が
        // 「基準が内側」になり population 不一致を検出できなくなる
        data.extend_from_slice(&self.population_fingerprint().to_le_bytes());

        data
    }

    /// FNV-1a 64-bit hash of per-body quantities [`Self::serialize_state`]
    /// does **not** restore (shape/mass/inertia/collider/filter/material),
    /// in body index order.
    ///
    /// # なぜこれが要るか (rollback body 数不一致、gap #3)
    ///
    /// [`Self::deserialize_state`] は body **数**の一致しか見ていない
    /// `remove_body` は `swap_remove` なので、count が元に戻っても
    /// index↔body の対応 (= population) が変わっていることがある —
    /// その場合 count 一致だけでは検出できず、**別の body に誤った状態を
    /// 書き込んで silent に通る** (fail-fast になっていない)
    ///
    /// ⚠️ **位置 / 速度 / 回転 / sleep を含めない** — それらは blob 自身が
    /// 復元する量なので、ここに混ぜると「自分で組んだ量で自分を検査する」
    /// 循環になり population 不一致を検出できなくなる
    #[must_use]
    pub fn population_fingerprint(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for i in 0..self.bodies.len() {
            let body = &self.bodies[i];
            fnv1a_fold(&mut hash, &body.inv_mass.hi.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_mass.lo.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.x.hi.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.x.lo.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.y.hi.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.y.lo.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.z.hi.to_le_bytes());
            fnv1a_fold(&mut hash, &body.inv_inertia.z.lo.to_le_bytes());
            fnv1a_fold(&mut hash, &[body.body_type as u8]);
            fnv1a_fold(&mut hash, &[u8::from(body.is_sensor)]);
            match self.body_collision_radii.get(i).copied().flatten() {
                Some(r) => {
                    fnv1a_fold(&mut hash, &[1u8]);
                    fnv1a_fold(&mut hash, &r.hi.to_le_bytes());
                    fnv1a_fold(&mut hash, &r.lo.to_le_bytes());
                }
                None => fnv1a_fold(&mut hash, &[0u8]),
            }
            let filter = self.body_filter(i);
            fnv1a_fold(&mut hash, &filter.layer.to_le_bytes());
            fnv1a_fold(&mut hash, &filter.mask.to_le_bytes());
            fnv1a_fold(&mut hash, &filter.group.to_le_bytes());
            let material = self
                .body_materials
                .get(i)
                .copied()
                .unwrap_or(crate::material::DEFAULT_MATERIAL);
            fnv1a_fold(&mut hash, &material.to_le_bytes());
        }
        hash
    }

    /// Deserialize world state (for rollback netcode).
    ///
    /// Restores per-body transforms. Parallel arrays (collision radii,
    /// filters, materials, island manager) are resized to match the body
    /// count, preserving existing entries and zero-filling new ones.
    ///
    /// # Rollback の契約 (gap #3、2026-10-02)
    ///
    /// body **数**が一致していても **population** (どの body がどんな
    /// `inv_mass` / `inv_inertia` / collider / filter / material を持つか)
    /// が一致していなければ拒否する ([`Self::population_fingerprint`]、v3)
    /// caller は body の生成・破棄 (`add_body` / `remove_body`) と joints /
    /// force fields / filters / collider radii / materials の変更を、目標
    /// frame まで決定論的に replay してから本関数を呼ぶこと ⚠️
    /// **joints の再構築は本関数の呼び出し前に行う** — 本関数の末尾で
    /// island を joints から作り直すため、順序が逆だと sleep の島が
    /// 異なる形になる
    ///
    /// 不一致 (`false`) の場合、`self` は一切変更されない (すべての検査は
    /// body 状態を書き込む前に完了する)
    pub fn deserialize_state(&mut self, data: &[u8]) -> bool {
        // Header 12 byte: magic 4 + version 2 + reserved 2 + count 4
        if data.len() < 12 {
            return false;
        }
        // ⚠️ magic / version を検査しないと **旧 blob (先頭が body 数) を
        // 新 format として誤って解釈する** 不一致は明示的に拒否する
        if data[0..4] != Self::STATE_MAGIC {
            return false;
        }
        if u16::from_le_bytes([data[4], data[5]]) != Self::STATE_VERSION {
            return false;
        }
        // ⚠️ reserved も検査する — 検査しないと **blob の中に silent に
        // 無視される byte が残り**、「1 byte でも壊れたらどれかの field に
        // 反映される」という不変条件に穴が空く (将来 flag に使う時は
        // `STATE_VERSION` を上げて分岐する)
        if u16::from_le_bytes([data[6], data[7]]) != 0 {
            return false;
        }

        let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        if count != self.bodies.len() {
            return false;
        }

        // 長さ検査は body 数から一意に決まる (header + 運動状態 + sleep + overflow + fingerprint)
        let expected = 12 + count * 208 + count * 5 + 1 + 8;
        if data.len() < expected {
            return false;
        }

        // ⚠️ **population fingerprint は body 状態を 1 byte も書く前に検査する**
        // (v3、gap #3) count が一致していても `swap_remove` + `add_body` で
        // population (= inv_mass / inertia / collider 等の組み合わせ) が
        // 変わっていれば拒否する — ここで return すれば self は未変更のまま
        let fp_offset = expected - 8;
        let stored_fingerprint = u64::from_le_bytes([
            data[fp_offset],
            data[fp_offset + 1],
            data[fp_offset + 2],
            data[fp_offset + 3],
            data[fp_offset + 4],
            data[fp_offset + 5],
            data[fp_offset + 6],
            data[fp_offset + 7],
        ]);
        if stored_fingerprint != self.population_fingerprint() {
            return false;
        }

        // Per-body: position(48) + velocity(48) + rotation(64) + angular_velocity(48) = 208 bytes
        let mut offset = 12;
        for body in &mut self.bodies {
            if offset + 208 > data.len() {
                return false;
            }

            // Helper to read Fix128
            let read_fix128 = |o: &mut usize| -> Fix128 {
                let hi = i64::from_le_bytes([
                    data[*o],
                    data[*o + 1],
                    data[*o + 2],
                    data[*o + 3],
                    data[*o + 4],
                    data[*o + 5],
                    data[*o + 6],
                    data[*o + 7],
                ]);
                *o += 8;
                let lo = u64::from_le_bytes([
                    data[*o],
                    data[*o + 1],
                    data[*o + 2],
                    data[*o + 3],
                    data[*o + 4],
                    data[*o + 5],
                    data[*o + 6],
                    data[*o + 7],
                ]);
                *o += 8;
                Fix128 { hi, lo }
            };

            body.position.x = read_fix128(&mut offset);
            body.position.y = read_fix128(&mut offset);
            body.position.z = read_fix128(&mut offset);

            body.velocity.x = read_fix128(&mut offset);
            body.velocity.y = read_fix128(&mut offset);
            body.velocity.z = read_fix128(&mut offset);

            body.rotation.x = read_fix128(&mut offset);
            body.rotation.y = read_fix128(&mut offset);
            body.rotation.z = read_fix128(&mut offset);
            body.rotation.w = read_fix128(&mut offset);

            body.angular_velocity.x = read_fix128(&mut offset);
            body.angular_velocity.y = read_fix128(&mut offset);
            body.angular_velocity.z = read_fix128(&mut offset);
        }

        // Resync parallel arrays to match body count (extend AND truncate)
        let n = self.bodies.len();
        while self.body_collision_radii.len() < n {
            self.body_collision_radii.push(None);
        }
        self.body_collision_radii.truncate(n);
        self.body_colliders.resize(n, None);
        while self.body_filters.len() < n {
            self.body_filters.push(CollisionFilter::DEFAULT);
        }
        self.body_filters.truncate(n);
        while self.body_materials.len() < n {
            self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        }
        self.body_materials.truncate(n);
        #[cfg(feature = "std")]
        self.sync_body_stable_ids();
        // ⚠️ `IslandManager::new` は sleep_data を既定 (Awake / idle_frames 0) に
        // 戻すので、**この後に blob から復元する** 順序を逆にすると復元が消える
        self.islands = IslandManager::new(n, self.islands.config);

        // Sleep 状態の復元 (WM-08) — body の運動状態を読んだ直後から続く
        for i in 0..n {
            let state = match data[offset] {
                0 => SleepState::Awake,
                1 => SleepState::Sleeping,
                // 未知の値は拒否する (silent に Awake へ倒すと被覆の穴が
                // 「復元できた」ように見える)
                _ => return false,
            };
            offset += 1;
            let idle_frames = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;
            if let Some(sd) = self.islands.sleep_data.get_mut(i) {
                sd.state = state;
                sd.idle_frames = idle_frames;
            }
        }

        // overflow flag の復元 (v2)
        //
        // ⚠️ **読み込んだ状態に従う** (立てるだけでなく下げる) — sticky は
        // 「step を重ねても落ちない」ことであって、**別の枝を読み込んでも
        // 残る**ことではない 残すと探索で枝を跨いで汚染する
        self.overflow_detected = match data[offset] {
            0 => false,
            1 => true,
            // 未知の値は拒否 (silent に false へ倒すと undecided が消える)
            _ => return false,
        };

        true
    }
}

impl core::fmt::Debug for PhysicsWorld {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PhysicsWorld")
            .field("config", &self.config)
            .field("bodies", &self.bodies.len())
            .field("distance_constraints", &self.distance_constraints.len())
            .field("contact_constraints", &self.contact_constraints.len())
            .field("sdf_colliders", &self.sdf_colliders.len())
            .field("static_colliders", &self.static_colliders.len())
            .field("joints", &self.joints.len())
            .field("force_fields", &self.force_fields.len())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// oracle: [`RigidBody::world_inv_inertia_apply`] (unchecked). On inputs
    /// in range the checked form gives the same bits, for rotations about
    /// several axes and an anisotropic `inv_inertia` (`1, 2, 4`), so a
    /// dropped rotation, a component taken from the wrong axis or a sign
    /// error in the checked quaternion product changes the result. The
    /// fixture asserts that rotation and anisotropy both matter here.
    #[test]
    fn checked_world_inv_inertia_apply_is_the_unchecked_one_in_range() {
        let torques = [
            Vec3Fix::from_int(3, -1, 2),
            Vec3Fix::new(
                Fix128::from_ratio(-5, 7),
                Fix128::from_ratio(11, 3),
                Fix128::from_int(-9),
            ),
        ];
        let rotations = [
            QuatFix::from_axis_angle(Vec3Fix::from_int(1, 2, 3), Fix128::from_ratio(7, 10)),
            QuatFix::from_axis_angle(Vec3Fix::from_int(-2, 1, 0), Fix128::from_ratio(-13, 5)),
            QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), Fix128::from_ratio(1, 3)),
        ];
        for q in rotations {
            let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
            body.inv_inertia = Vec3Fix::from_int(1, 2, 4);
            body.rotation = q;
            for tau in torques {
                let unchecked = body.world_inv_inertia_apply(tau);
                assert_eq!(
                    body.checked_world_inv_inertia_apply(tau),
                    Some(unchecked),
                    "q {q:?} tau {tau:?}"
                );
                let mut unrotated = body;
                unrotated.rotation = QuatFix::IDENTITY;
                assert_ne!(
                    unrotated.world_inv_inertia_apply(tau),
                    unchecked,
                    "fixture: the rotation must matter"
                );
                let mut isotropic = body;
                isotropic.inv_inertia = Vec3Fix::from_int(2, 2, 2);
                assert_ne!(
                    isotropic.world_inv_inertia_apply(tau),
                    unchecked,
                    "fixture: the anisotropy must matter"
                );
            }
        }
    }

    /// oracle: the key is a function of (kind, ids, ordinal) only. Entries
    /// of one pair get distinct keys in arrival order, a second pass with a
    /// fresh counter reproduces them, and kind or direction separate keys.
    #[test]
    fn tgs_cache_key_separates_ordinal_kind_and_pair() {
        let mut ord = std::collections::BTreeMap::new();
        let k0 = tgs_cache_key(TGS_KEY_CONTACT, 3, 7, &mut ord);
        let k1 = tgs_cache_key(TGS_KEY_CONTACT, 3, 7, &mut ord);
        let other = tgs_cache_key(TGS_KEY_CONTACT, 3, 8, &mut ord);
        assert_ne!(k0, k1, "two entries of one pair must not share a key");
        let mut again = std::collections::BTreeMap::new();
        assert_eq!(tgs_cache_key(TGS_KEY_CONTACT, 3, 7, &mut again), k0);
        assert_eq!(tgs_cache_key(TGS_KEY_CONTACT, 3, 7, &mut again), k1);
        let mut fresh = std::collections::BTreeMap::new();
        assert_ne!(tgs_cache_key(TGS_KEY_DISTANCE, 3, 7, &mut fresh), k0);
        let mut fresh = std::collections::BTreeMap::new();
        assert_ne!(tgs_cache_key(TGS_KEY_CONTACT, 7, 3, &mut fresh), k0);
        assert_ne!(other, k0);
        assert_eq!(ord.get(&(3, 7)), Some(&2));
    }

    #[test]
    fn test_rigid_body_creation() {
        let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        assert!(!body.is_static());

        let static_body = RigidBody::new_static(Vec3Fix::ZERO);
        assert!(static_body.is_static());
    }

    #[test]
    fn test_gravity_integration() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        world.add_body(body);

        // Step for 1 second (in 10 steps of 0.1s)
        for _ in 0..10 {
            world.step(Fix128::from_ratio(1, 10));
        }

        // Body should have fallen
        let body = world.get_body(0).unwrap();
        assert!(
            body.position.y < Fix128::from_int(10),
            "Body should have fallen"
        );
    }

    #[test]
    fn test_distance_constraint() {
        let config = SolverConfig {
            substeps: 4,
            iterations: 8,
            gravity: Vec3Fix::ZERO, // No gravity for constraint test
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        // Two bodies connected by a distance constraint
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));

        world.add_distance_constraint(DistanceConstraint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(3), // Target distance: 3
        ));

        // Step simulation
        for _ in 0..100 {
            world.step(Fix128::from_ratio(1, 60));
        }

        // Body B should be approximately 3 units from body A
        let body_b = world.get_body(b).unwrap();
        let distance = body_b.position.length();

        // Allow some tolerance due to gravity
        assert!(
            distance < Fix128::from_int(5),
            "Constraint should pull body closer"
        );
    }

    #[test]
    fn test_state_serialization() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        world.add_body(RigidBody::new(Vec3Fix::from_int(1, 2, 3), Fix128::ONE));
        world.add_body(RigidBody::new(
            Vec3Fix::from_int(4, 5, 6),
            Fix128::from_int(2),
        ));

        let state = world.serialize_state();

        // Deserialize into another world with the same population (same
        // mass per body index) but different starting positions — v3 の
        // population fingerprint は mass 等の組み合わせが一致している
        // ことを要求する (gap #3、mass が異なれば別 population として拒否)
        let mut world2 = PhysicsWorld::new(config);
        world2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        world2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));

        assert!(world2.deserialize_state(&state));

        assert_eq!(world2.bodies[0].position.x.hi, 1);
        assert_eq!(world2.bodies[0].position.y.hi, 2);
        assert_eq!(world2.bodies[1].position.x.hi, 4);
    }

    #[test]
    fn test_determinism() {
        // Run the same simulation twice and verify identical results
        let config = SolverConfig::default();

        let run_simulation = || {
            let mut world = PhysicsWorld::new(config);
            world.add_body(RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE));
            world.add_body(RigidBody::new(
                Vec3Fix::from_int(5, 10, 0),
                Fix128::from_int(2),
            ));

            for _ in 0..100 {
                world.step(Fix128::from_ratio(1, 60));
            }

            world.serialize_state()
        };

        let state1 = run_simulation();
        let state2 = run_simulation();

        assert_eq!(state1, state2, "Simulation must be deterministic");
    }

    #[test]
    fn test_body_material_assignment() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        let metal_id = world.material_table.register_metal();
        let rubber_id = world.material_table.register_rubber();

        let a = world.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE));

        world.set_body_material(a, metal_id);
        world.set_body_material(b, rubber_id);

        let combined = world.combined_material(a, b);
        assert!(
            combined.friction > Fix128::ZERO,
            "Combined friction should be positive"
        );
    }

    #[test]
    fn test_contact_cache_integration() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        let a = world.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE));

        world.begin_frame();

        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_X,
            point_a: Vec3Fix::from_int(1, 0, 0),
            point_b: Vec3Fix::from_int(1, 0, 0),
        };
        world.add_contact_with_material(a, b, contact);

        world.end_frame();

        // Contact cache should have the manifold
        let key = crate::contact_cache::BodyPairKey::new(a, b);
        let manifold = world.contact_cache.find(&key);
        assert!(
            manifold.is_some(),
            "Contact cache should contain the manifold"
        );
    }

    #[test]
    fn test_kinematic_body() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        let kinematic = RigidBody::new_kinematic(Vec3Fix::ZERO);
        let k_id = world.add_body(kinematic);

        // Set kinematic target
        world.bodies[k_id].set_kinematic_target(Vec3Fix::from_int(5, 0, 0), QuatFix::IDENTITY);

        world.step(Fix128::from_ratio(1, 60));

        // Should have moved to target
        let pos = world.bodies[k_id].position;
        assert_eq!(pos.x.hi, 5, "Kinematic body should move to target");
    }

    #[test]
    fn test_body_type_enum() {
        let dynamic = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        assert!(dynamic.is_dynamic());
        assert!(!dynamic.is_kinematic());

        let static_b = RigidBody::new_static(Vec3Fix::ZERO);
        assert!(static_b.is_static());
        assert!(!static_b.is_dynamic());

        let kinematic = RigidBody::new_kinematic(Vec3Fix::ZERO);
        assert!(kinematic.is_kinematic());
        assert!(kinematic.is_static()); // inv_mass is zero
    }

    #[test]
    fn test_per_body_gravity_scale() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Body with zero gravity
        let mut no_grav = RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        no_grav.gravity_scale = Fix128::ZERO;
        let ng_id = world.add_body(no_grav);

        // Body with normal gravity
        let normal = RigidBody::new(Vec3Fix::from_int(5, 10, 0), Fix128::ONE);
        let n_id = world.add_body(normal);

        for _ in 0..60 {
            world.step(Fix128::from_ratio(1, 60));
        }

        // No-gravity body should not have fallen
        let ng_y = world.bodies[ng_id].position.y;
        assert!(
            ng_y > Fix128::from_int(9),
            "Zero-gravity body should stay near y=10, got {ng_y:?}"
        );

        // Normal body should have fallen (damping is strong, just check it fell at all)
        let n_y = world.bodies[n_id].position.y;
        assert!(
            n_y < Fix128::from_int(10),
            "Normal gravity body should fall"
        );
    }

    #[test]
    fn test_sdf_ground_collision() {
        use crate::math::QuatFix;
        use crate::sdf_collider::{ClosureSdf, SdfCollider};

        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.set_sdf_collision_radius(Fix128::from_ratio(1, 2)); // 0.5

        // Add falling body at y=5
        let body = RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
        let body_id = world.add_body(body);

        // Add ground plane SDF at y=0
        let ground = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
        world.add_sdf_collider(SdfCollider::new_static(
            Box::new(ground),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));

        // Simulate 2 seconds (120 frames at 60fps)
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            world.step(dt);
        }

        // Body should have fallen but been stopped by SDF ground
        let pos = world.bodies[body_id].position;
        let y = pos.y.to_f32();

        // Body should be near the ground (y ~ collision_radius = 0.5)
        assert!(y < 5.0, "Body should have fallen from y=5");
        assert!(
            y > -1.0,
            "Body should not have fallen through SDF ground, y={y}"
        );
    }

    #[test]
    fn test_warm_start_factor_default() {
        let config = SolverConfig::default();
        // デフォルト 0.85
        let expected = Fix128::from_ratio(85, 100);
        assert_eq!(config.warm_start_factor, expected);
    }

    #[test]
    fn test_warm_start_factor_convergence() {
        // warm_start_factor = 0.85 vs 0.0（無効）で収束速度を比較
        let make_world = |wsf: Fix128| {
            let config = SolverConfig {
                substeps: 4,
                iterations: 2, // 少ないイテレーションで差が出やすい
                gravity: Vec3Fix::ZERO,
                warm_start_factor: wsf,
                ..Default::default()
            };
            let mut world = PhysicsWorld::new(config);
            let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            let b = world.add_body(RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));
            let constraint =
                DistanceConstraint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(3));
            world.add_distance_constraint(constraint);
            world
        };

        let mut world_warm = make_world(Fix128::from_ratio(85, 100));
        let mut world_cold = make_world(Fix128::ZERO);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..30 {
            world_warm.step(dt);
            world_cold.step(dt);
        }

        // Both should converge, but warm-started should be closer to target distance
        let dist_warm = (world_warm.bodies[1].position - world_warm.bodies[0].position)
            .length()
            .to_f32();
        let dist_cold = (world_cold.bodies[1].position - world_cold.bodies[0].position)
            .length()
            .to_f32();

        let target = 3.0_f32;
        let err_warm = (dist_warm - target).abs();
        let err_cold = (dist_cold - target).abs();
        // warm-start版の誤差が小さいか、同等であること
        assert!(
            err_warm <= err_cold + 0.1,
            "Warm-start should converge at least as well: err_warm={err_warm}, err_cold={err_cold}"
        );
    }

    #[test]
    fn test_contact_warm_start_cached_lambda() {
        // ContactConstraint の cached_lambda がデフォルトでゼロ
        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        let cc = ContactConstraint::new(0, 1, contact);
        assert_eq!(cc.cached_lambda, Fix128::ZERO);
    }

    #[test]
    fn test_manual_contact_pushes_body_out_by_depth() {
        // 手動 contact は substep が clear_contacts するので (1.2.0)、solver level の
        // `solve_contact_constraints` を直接 iterations 回呼んで検証
        let config = SolverConfig {
            substeps: 1,
            iterations: 2,
            gravity: Vec3Fix::ZERO,
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);
        let _a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let _b = world.add_body(RigidBody::new(
            Vec3Fix::from_f32(0.0, 0.05, 0.0),
            Fix128::ONE,
        ));

        // normal は B→A 方向（ドキュメント仕様）。body_a が下にあるので下向き。
        // correction は normal 方向。body_a += correction, body_b -= correction。
        // body_a=static, body_b は -(-Y) = +Y に押し出される。
        let contact = Contact {
            depth: Fix128::from_ratio(5, 100), // 0.05 m 侵入
            normal: -Vec3Fix::UNIT_Y,          // B→A = 下向き
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::from_f32(0.0, 0.05, 0.0),
        };
        world.add_contact(ContactConstraint::new(0, 1, contact));

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..world.config.iterations {
            world.solve_contact_constraints(dt);
        }

        // body_b は depth ちょうど 1 回分 (0.05) 上に押し出される、2 iteration でも 2 倍にならない
        let y = world.bodies[1].position.y.to_f32();
        assert!(
            (y - 0.10).abs() < 1e-5,
            "Body should be pushed up by depth once: y={y}"
        );
    }

    // -------------------------------------------------------------------
    // v0.11.0: PhysicsWorld auto-routing to installed GPU solver bridge
    // -------------------------------------------------------------------

    /// Recording bridge implementation for the v0.11.0 auto-routing
    /// tests. Counts invocations of contact-solve methods via shared
    /// `Arc<AtomicUsize>` counters so the test can assert routing
    /// occurred without needing to reach inside the boxed trait object.
    ///
    /// All methods are no-ops on world state (positions are not
    /// modified), so a test that observes position drift while this
    /// bridge is installed is proof that the CPU path ran — the bridge
    /// intentionally does not solve.
    #[cfg(feature = "gpu-solver-bridge")]
    struct RecordingBridge {
        send_contact_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
        send_body_state_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
        dispatch_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
        // v0.12.0 joint-solve counters (default zero; used only by the
        // joint auto-routing tests below)
        send_joints_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
        send_body_rotations_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
        dispatch_joint_count: std::sync::Arc<core::sync::atomic::AtomicUsize>,
    }

    #[cfg(feature = "gpu-solver-bridge")]
    impl crate::gpu_bridge::GpuSolverBridge for RecordingBridge {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(
            &self,
            _fixture: &crate::gpu_bridge::DiffFixture,
        ) -> Result<(), crate::gpu_bridge::GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {
            self.send_contact_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {
            self.send_body_state_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
            self.dispatch_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
        fn send_joints(&mut self, _j: &[crate::joint::Joint]) {
            self.send_joints_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn send_body_rotations(&mut self, _r: &[[Fix128; 4]]) {
            self.send_body_rotations_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
        fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {
            self.dispatch_joint_count
                .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Two overlapping unit spheres in zero gravity: since 1.2.0 contacts are
    /// re-detected in every substep, so the contact the bridge tests observe
    /// has to come from `detect_collisions` (a manually added contact would
    /// be cleared at the start of the substep).
    #[cfg(feature = "gpu-solver-bridge")]
    fn build_one_contact_world() -> PhysicsWorld {
        let mut world = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let body_a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        let body_b = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_ratio(19, 10), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        world.add_body_with_radius(body_a, Fix128::ONE);
        world.add_body_with_radius(body_b, Fix128::ONE);
        world
    }

    /// v0.11.0: attaching a bridge via `set_gpu_solver_bridge` routes
    /// contact-solve through the bridge on every subsequent `step`.
    /// Verified by counting bridge method invocations.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_step_auto_routes_through_bridge_when_attached() {
        let send_contact_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let send_body_state_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let dispatch_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));

        let bridge = RecordingBridge {
            send_contact_count: std::sync::Arc::clone(&send_contact_count),
            send_body_state_count: std::sync::Arc::clone(&send_body_state_count),
            dispatch_count: std::sync::Arc::clone(&dispatch_count),
            send_joints_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_rotations_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_joint_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
        };

        let mut world = build_one_contact_world();
        world.set_gpu_solver_bridge(Some(Box::new(bridge)));
        assert!(world.gpu_solver_bridge_installed());

        // substep は solve_contact_constraints を `config.iterations`
        // 回呼ぶ (default 4)。auto-routing 経路はその全てを bridge に
        // 転送するはずなので、send / dispatch は
        // `config.iterations` 回発火する。
        let iters = world.config.iterations;
        let dt = Fix128::from_ratio(1, 60);
        world.substep(dt);

        assert_eq!(
            send_contact_count.load(core::sync::atomic::Ordering::SeqCst),
            iters,
            "send_contact_constraints must be invoked once per PGS iteration"
        );
        assert_eq!(
            send_body_state_count.load(core::sync::atomic::Ordering::SeqCst),
            iters,
            "send_body_state must be invoked once per PGS iteration"
        );
        assert_eq!(
            dispatch_count.load(core::sync::atomic::Ordering::SeqCst),
            iters,
            "dispatch_contact_solve_iteration must fire once per PGS iteration"
        );
    }

    /// v0.11.0: without a bridge installed, `step` falls back to the CPU
    /// contact solver. Verified by observing that contact-affected body
    /// positions drift — the CPU path applies the position correction
    /// that the recording bridge above would not.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_step_falls_back_to_cpu_when_bridge_detached() {
        let mut world = build_one_contact_world();
        assert!(!world.gpu_solver_bridge_installed());

        let x_before = world.bodies[0].position.x;
        let dt = Fix128::from_ratio(1, 60);
        world.substep(dt);
        let x_after = world.bodies[0].position.x;

        // CPU contact solve applies a position correction along the
        // contact normal (x axis). Any non-zero drift confirms the CPU
        // path was taken instead of the (absent) bridge path.
        assert_ne!(
            x_before, x_after,
            "CPU contact solver must adjust position when no bridge is installed"
        );
    }

    /// v0.11.0: attach / take lifecycle preserves the field invariant
    /// (`gpu_solver_bridge_installed` reflects the current state) and
    /// `take_gpu_solver_bridge` returns the same box on the first
    /// call and `None` on subsequent calls.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_bridge_attach_detach_lifecycle() {
        let mut world = PhysicsWorld::new(SolverConfig::default());
        assert!(!world.gpu_solver_bridge_installed());

        let bridge = RecordingBridge {
            send_contact_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_state_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_joints_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_rotations_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_joint_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
        };
        world.set_gpu_solver_bridge(Some(Box::new(bridge)));
        assert!(world.gpu_solver_bridge_installed());

        let taken = world.take_gpu_solver_bridge();
        assert!(taken.is_some(), "first take must return the installed box");
        assert!(!world.gpu_solver_bridge_installed());

        let taken_again = world.take_gpu_solver_bridge();
        assert!(
            taken_again.is_none(),
            "subsequent take must return None when nothing is installed"
        );
    }

    // -------------------------------------------------------------------
    // v0.12.0: PhysicsWorld auto-routing joint solve through installed bridge
    // -------------------------------------------------------------------

    #[cfg(feature = "gpu-solver-bridge")]
    fn build_one_ball_joint_world() -> PhysicsWorld {
        use crate::joint::{BallJoint, Joint};
        use crate::math::Vec3Fix;
        let mut world = PhysicsWorld::new(SolverConfig::default());
        let body_a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        let body_b = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        world.add_body(body_a);
        world.add_body(body_b);
        world.joints.push(Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
        world
    }

    /// v0.12.0: attaching a bridge routes joint-solve through the
    /// bridge on every subsequent `substep` — verified by counting
    /// bridge method invocations. Joint solve runs **once per
    /// substep** (unlike contact solve which iterates
    /// `config.iterations` times), so the counters should each
    /// increment by exactly one after a single substep.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_substep_auto_routes_joint_solve_through_bridge() {
        let send_joints_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let send_body_rotations_count =
            std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let dispatch_joint_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));

        let bridge = RecordingBridge {
            send_contact_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_state_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_joints_count: std::sync::Arc::clone(&send_joints_count),
            send_body_rotations_count: std::sync::Arc::clone(&send_body_rotations_count),
            dispatch_joint_count: std::sync::Arc::clone(&dispatch_joint_count),
        };

        let mut world = build_one_ball_joint_world();
        world.set_gpu_solver_bridge(Some(Box::new(bridge)));

        let dt = Fix128::from_ratio(1, 60);
        world.substep(dt);

        assert_eq!(
            send_joints_count.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "send_joints must be invoked exactly once per substep"
        );
        assert_eq!(
            send_body_rotations_count.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "send_body_rotations must be invoked exactly once per substep"
        );
        assert_eq!(
            dispatch_joint_count.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "dispatch_joint_solve_iteration must fire exactly once per substep (single Baumgarte projection, not PGS)"
        );
    }

    /// v0.12.0: without a bridge installed, joint solve falls back to
    /// the CPU `solve_joints`. Verified by observing that the two
    /// bodies (initially 1 unit apart with a zero-anchor ball joint)
    /// get pulled together by the joint constraint.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_substep_falls_back_to_cpu_joint_solve_when_bridge_detached() {
        let mut world = build_one_ball_joint_world();
        assert!(!world.gpu_solver_bridge_installed());

        let a_before = world.bodies[0].position.x;
        let b_before = world.bodies[1].position.x;
        let dt = Fix128::from_ratio(1, 60);
        world.substep(dt);
        let a_after = world.bodies[0].position.x;
        let b_after = world.bodies[1].position.x;

        // Zero-anchor ball joint constrains anchor_a == anchor_b,
        // which for zero local anchors is body positions themselves.
        // CPU `solve_ball_joint` therefore pulls each body toward
        // the other by half the separation — bodies should move
        // toward each other but not past each other.
        let a_delta = (a_after - a_before).to_f32();
        let b_delta = (b_after - b_before).to_f32();
        assert!(
            a_delta > 0.0,
            "body_a should move in +x toward body_b (delta = {a_delta})"
        );
        assert!(
            b_delta < 0.0,
            "body_b should move in -x toward body_a (delta = {b_delta})"
        );
    }

    /// v0.12.0: attach → observe joint routing → take →
    /// verify subsequent substep uses CPU path — a mixed-lifecycle
    /// test that exercises both the auto-routing and fallback code
    /// paths on the same `PhysicsWorld` instance.
    #[cfg(feature = "gpu-solver-bridge")]
    #[test]
    fn physics_world_substep_joint_solve_bridge_attach_take_lifecycle() {
        let dispatch_joint_count = std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let bridge = RecordingBridge {
            send_contact_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_state_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_joints_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            send_body_rotations_count: std::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0)),
            dispatch_joint_count: std::sync::Arc::clone(&dispatch_joint_count),
        };

        let mut world = build_one_ball_joint_world();
        world.set_gpu_solver_bridge(Some(Box::new(bridge)));

        let dt = Fix128::from_ratio(1, 60);
        world.substep(dt);
        assert_eq!(
            dispatch_joint_count.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "first substep must route through bridge"
        );

        let taken = world.take_gpu_solver_bridge();
        assert!(taken.is_some(), "bridge take must return Some");

        // After detach, the bridge's counter must not increment on
        // subsequent substeps — the CPU path takes over.
        let a_before = world.bodies[0].position.x;
        world.substep(dt);
        let a_after = world.bodies[0].position.x;
        assert_eq!(
            dispatch_joint_count.load(core::sync::atomic::Ordering::SeqCst),
            1,
            "after take_gpu_solver_bridge, no further bridge dispatch may occur"
        );
        assert_ne!(
            a_before, a_after,
            "CPU joint solve should move body_a toward body_b after detach"
        );
    }

    // ------------------------------------------------------------------
    // Graph coloring soundness (v1.0.1): >64 colors + static exclusion
    // ------------------------------------------------------------------

    /// Hub body + `spokes` distance constraints, each spoke its own body.
    fn hub_world(hub: RigidBody, spokes: usize) -> PhysicsWorld {
        let mut world = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..Default::default()
        });
        let hub_idx = world.add_body(hub);
        for i in 0..spokes {
            let spoke = world.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(i as i64 + 1, 0, 0),
                Fix128::ONE,
            ));
            world.add_distance_constraint(DistanceConstraint::new(
                hub_idx,
                spoke,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Fix128::ONE,
            ));
        }
        world
    }

    #[test]
    fn coloring_dynamic_hub_over_64_spokes_gets_one_color_per_constraint() {
        // Pre-1.0.1: colors saturated at 64 and every further constraint
        // aliased into batch 64 while sharing the hub body.
        for spokes in [65usize, 70, 200] {
            let mut world = hub_world(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), spokes);
            world.rebuild_batches();
            assert_eq!(
                world.num_batches(),
                spokes,
                "dynamic hub with {spokes} spokes needs {spokes} batches"
            );
            assert!(world.batches_are_body_disjoint());
        }
    }

    #[test]
    fn coloring_static_hub_imposes_no_constraint() {
        // A static floor touched by many bodies must not serialize the
        // solver: all spokes are distinct dynamic bodies → single batch.
        let mut world = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 200);
        world.rebuild_batches();
        assert_eq!(world.num_batches(), 1);
        assert!(world.batches_are_body_disjoint());
    }

    #[test]
    fn coloring_contacts_and_distances_share_color_space() {
        // Two constraint kinds on the same dynamic pair must land in
        // different batches (the contact pass runs after the distance pass
        // but colors are shared across both).
        let mut world = PhysicsWorld::new(SolverConfig::default());
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(1, 0, 0),
            Fix128::ONE,
        ));
        world.add_distance_constraint(DistanceConstraint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::ONE,
        ));
        world.add_contact(ContactConstraint::new(
            a,
            b,
            Contact {
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
                normal: Vec3Fix::from_int(1, 0, 0),
                depth: Fix128::from_ratio(1, 10),
            },
        ));
        world.rebuild_batches();
        assert_eq!(world.num_batches(), 2);
        assert!(world.batches_are_body_disjoint());
    }

    #[test]
    fn coloring_disjoint_check_detects_violation() {
        // Hand-craft an invalid batch to prove the checker is not vacuous.
        let mut world = hub_world(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), 2);
        world.rebuild_batches();
        assert!(world.batches_are_body_disjoint());
        let mut bad = ConstraintBatch::default();
        bad.distance_indices.extend([0usize, 1]);
        world.constraint_batches = vec![bad];
        assert!(!world.batches_are_body_disjoint());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_step_static_hub_matches_sequential_solver() {
        // Static hub + 70 spokes: pre-1.0.1 this scene created a batch with
        // 6 constraints all holding `&mut` to the hub. Step both solver
        // paths and require bit-exact agreement.
        let sub_dt = Fix128::from_ratio(1, 240);
        let mut par = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 70);
        let mut seq = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 70);
        par.rebuild_batches();
        seq.rebuild_batches();
        assert_eq!(par.num_batches(), 1, "static hub → single batch");
        for _ in 0..120 {
            par.substep_batched(sub_dt);
            seq.substep(sub_dt);
        }
        for (p, s) in par.bodies.iter().zip(&seq.bodies) {
            assert_eq!(p.position, s.position);
            assert_eq!(p.velocity, s.velocity);
        }
        assert!(par.batches_are_body_disjoint());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_step_rebuilds_when_body_becomes_dynamic_after_coloring() {
        // Coloring excluded the hub as static; flipping it to dynamic
        // without touching constraints must not leave the stale snapshot
        // in place (two threads would otherwise take `&mut` to the hub).
        let mut world = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 70);
        world.rebuild_batches();
        assert_eq!(world.num_batches(), 1);
        world.bodies[0].inv_mass = Fix128::ONE;
        world.solve_constraints_batched(Fix128::from_ratio(1, 60));
        assert_eq!(world.num_batches(), 70);
        assert!(world.batches_are_body_disjoint());
    }

    // ------------------------------------------------------------------
    // Mutation-score tests (2026-09-15, quality-deep core score 32% 対応)
    // 各 test は cargo-mutants の missed 変異 (演算子置換 / 分岐反転 / 符号削除)
    // を値の厳密一致で殺すことを目的にする  dt は 2 の冪の逆数 (1/4) にして
    // inv_dt が Fix128 で exact になるようにし、expected は独立に手計算した値
    // ------------------------------------------------------------------

    /// gravity 0 / damping なしの world (integrate の副作用を消して単機能を観測)
    fn quiet_world() -> PhysicsWorld {
        let config = SolverConfig {
            gravity: Vec3Fix::ZERO,
            damping: Fix128::ONE,
            ..SolverConfig::default()
        };
        PhysicsWorld::new(config)
    }

    fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    fn r(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    #[test]
    fn update_velocities_linear_velocity_is_position_delta_times_inv_dt() {
        let mut world = quiet_world();
        let dyn_idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let static_idx = world.add_body(RigidBody::new_static(v3(5, 5, 5)));
        let dt = r(1, 4); // inv_dt = 4 exactly

        // prev → position の差分 (1/4, -1/2, 3/4) → velocity は (1, -2, 3)
        world.bodies[dyn_idx].prev_position = v3(1, 1, 1);
        world.bodies[dyn_idx].position = Vec3Fix::new(
            Fix128::ONE + r(1, 4),
            Fix128::ONE - r(1, 2),
            Fix128::ONE + r(3, 4),
        );
        // static body にも差分を仕込む: 処理対象外なので velocity は ZERO のまま
        world.bodies[static_idx].prev_position = v3(0, 0, 0);
        world.bodies[static_idx].position = v3(5, 5, 5);

        world.update_velocities(dt);

        assert_eq!(world.bodies[dyn_idx].velocity, v3(1, -2, 3));
        assert_eq!(world.bodies[static_idx].velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn update_velocities_angular_velocity_from_rotation_delta_positive_w() {
        let mut world = quiet_world();
        let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let dt = r(1, 4);
        // 小さい回転 (θ = 1/2 rad) → dq.w > 0 の分岐
        let q = QuatFix::from_axis_angle(v3(1, 2, 3), r(1, 2));
        world.bodies[idx].prev_rotation = QuatFix::IDENTITY;
        world.bodies[idx].rotation = q;

        world.update_velocities(dt);

        // dq = q: θ = 1/2 rad about (1, 2, 3)/√14 ⇒ ω = axis · θ / dt = axis · 2
        let got = world.bodies[idx].angular_velocity;
        let n = crate::det_math::sqrt64(14.0);
        let want = [2.0 / n, 4.0 / n, 6.0 / n];
        for (g, e) in [got.x, got.y, got.z].iter().zip(want) {
            assert!((g.to_f64() - e).abs() < 1e-12, "{} vs {e}", g.to_f64());
        }
    }

    #[test]
    fn update_velocities_angular_velocity_negative_w_flips_sign() {
        let mut world = quiet_world();
        let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let dt = r(1, 4);
        // θ = 3π/2 → w = cos(3π/4) < 0 → 符号反転分岐
        let theta = Fix128::PI + Fix128::HALF_PI;
        let q = QuatFix::from_axis_angle(v3(1, -2, 3), theta);
        assert!(q.w < Fix128::ZERO, "test precondition: w must be negative");
        world.bodies[idx].prev_rotation = QuatFix::IDENTITY;
        world.bodies[idx].rotation = q;

        world.update_velocities(dt);

        // 3π/2 about a = (1, −2, 3)/√14 equals π/2 about −a (w folded to
        // ≥ 0) ⇒ ω = −a · (π/2) / dt = −a · 2π; every component non-zero so
        // each sign is checked
        let got = world.bodies[idx].angular_velocity;
        let n = crate::det_math::sqrt64(14.0);
        let k = 2.0 * core::f64::consts::PI / n;
        let want = [-k, 2.0 * k, -3.0 * k];
        for (g, e) in [got.x, got.y, got.z].iter().zip(want) {
            assert!((g.to_f64() - e).abs() < 1e-12, "{} vs {e}", g.to_f64());
        }
    }

    #[test]
    fn update_velocities_angular_velocity_zero_w_takes_positive_branch() {
        let mut world = quiet_world();
        let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let dt = r(1, 4);
        // w == 0 ちょうど (π 回転) → `<` は false = 正の分岐 (`<=` / `==` 変異はここで死ぬ)
        let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        world.bodies[idx].prev_rotation = QuatFix::IDENTITY;
        world.bodies[idx].rotation = q;

        world.update_velocities(dt);

        // π about z (w = 0 takes the positive branch) ⇒ ω = (0, 0, π / dt) = (0, 0, 4π)
        let got = world.bodies[idx].angular_velocity;
        assert_eq!((got.x, got.y), (Fix128::ZERO, Fix128::ZERO));
        let want = 4.0 * core::f64::consts::PI;
        assert!(
            (got.z.to_f64() - want).abs() < 1e-12,
            "{} vs {want}",
            got.z.to_f64()
        );
    }

    /// 接触 1 本を直接差し込む helper (normal は B → A 方向 = +x)
    fn push_contact(
        world: &mut PhysicsWorld,
        a: usize,
        b: usize,
        friction: Fix128,
        restitution: Fix128,
    ) {
        let contact = Contact {
            depth: r(1, 100),
            normal: v3(1, 0, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        let mut c = ContactConstraint::new(a, b, contact);
        c.friction = friction;
        c.restitution = restitution;
        world.contact_constraints.push(c);
    }

    /// Phase 1 は velocity を (position - prev) * inv_dt で上書きするので、
    /// 速度 v を与えるには prev_position = position - v * dt を仕込む (dt = 1/4 固定)
    /// 反発は pre-solve の法線速度 v̄_n を使うので `velocity` にも同じ v を入れる
    /// (拘束が位置を動かさなかった substep = pre と post が一致する場合)
    fn give_velocity(world: &mut PhysicsWorld, idx: usize, v: Vec3Fix) {
        let b = &mut world.bodies[idx];
        b.velocity = v;
        b.prev_position = b.position - v * r(1, 4);
        b.prev_rotation = b.rotation;
    }

    /// 全 body の prev を現在値に揃える (速度 0)
    fn freeze_positions(world: &mut PhysicsWorld) {
        for b in &mut world.bodies {
            b.prev_position = b.position;
            b.prev_rotation = b.rotation;
        }
    }

    #[test]
    fn update_velocities_restitution_splits_impulse_by_inverse_mass() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        push_contact(&mut world, a, b, Fix128::ZERO, r(1, 2));
        freeze_positions(&mut world);
        // A が B に向かって -x に 1 で接近 (vn = -1 < 0)
        give_velocity(&mut world, a, v3(-1, 0, 0));

        world.update_velocities(r(1, 4));

        // delta_vn = -(1 + 0.5) * (-1) = 1.5、inv_w = 1/2 → A += 0.75、B -= 0.75
        assert_eq!(
            world.bodies[a].velocity,
            Vec3Fix::new(-r(1, 4), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[b].velocity,
            Vec3Fix::new(-r(3, 4), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn update_velocities_restitution_skips_separating_contact() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        push_contact(&mut world, a, b, Fix128::ZERO, r(1, 2));
        freeze_positions(&mut world);
        // vn = +1 > 0 (離れていく) → restitution なし、tangent なし → 無変化
        give_velocity(&mut world, a, v3(1, 0, 0));

        world.update_velocities(r(1, 4));

        assert_eq!(world.bodies[a].velocity, v3(1, 0, 0));
        assert_eq!(world.bodies[b].velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn update_velocities_restitution_uses_mass_ratio() {
        let mut world = quiet_world();
        // A: inv_mass 1、B: inv_mass 3 (直接設定、1/(1/3) の丸めを避ける) → inv_w = 1/4 exact
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        world.bodies[b].inv_mass = Fix128::from_int(3);
        push_contact(&mut world, a, b, Fix128::ZERO, Fix128::ZERO);
        freeze_positions(&mut world);
        give_velocity(&mut world, a, v3(-2, 0, 0)); // vn = -2、e = 0 → delta_vn = 2

        world.update_velocities(r(1, 4));

        // A += 2 * (1 * 1/4) = 0.5 → -1.5、B -= 2 * (3 * 1/4) = 1.5 → -1.5 (完全非弾性で速度一致)
        assert_eq!(
            world.bodies[a].velocity,
            Vec3Fix::new(-r(3, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[b].velocity,
            Vec3Fix::new(-r(3, 2), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn update_velocities_friction_clamped_by_coulomb_limit() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_static(v3(1, 0, 0))); // inv_mass 0 → w_sum = 1
        push_contact(&mut world, a, b, r(1, 2), Fix128::ZERO);
        // 位置解の法線 λ = 1 → Coulomb 上限 μ λ / dt = 0.5 * 1 * 4 = 2
        world.contact_constraints[0].cached_lambda = Fix128::ONE;
        freeze_positions(&mut world);
        // 法線成分は正 (離れる) にして restitution を素通りさせる
        give_velocity(&mut world, a, v3(4, 3, 0));

        world.update_velocities(r(1, 4));

        // tangent_speed = 3、max = 2 → clamp 2、A.y -= 2 * 1 → 1、x は不変
        assert_eq!(
            world.bodies[a].velocity,
            Vec3Fix::new(Fix128::from_int(4), Fix128::ONE, Fix128::ZERO)
        );
        assert_eq!(world.bodies[b].velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn update_velocities_friction_below_limit_removes_all_tangential_velocity() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_static(v3(1, 0, 0)));
        push_contact(&mut world, a, b, Fix128::ONE, Fix128::ZERO);
        // μ = 1、λ = 1 → max = μ λ / dt = 4
        world.contact_constraints[0].cached_lambda = Fix128::ONE;
        freeze_positions(&mut world);
        give_velocity(&mut world, a, v3(4, 0, 3));

        world.update_velocities(r(1, 4));

        // tangent_speed = 3 < max 4 → applied = 3 → z 成分 0
        assert_eq!(
            world.bodies[a].velocity,
            Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn update_velocities_friction_boundary_equal_to_limit_uses_limit() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_static(v3(1, 0, 0)));
        push_contact(&mut world, a, b, Fix128::ONE, Fix128::ZERO);
        // μ = 1、λ = 3/4 → max = μ λ / dt = 3
        world.contact_constraints[0].cached_lambda = r(3, 4);
        freeze_positions(&mut world);
        // tangent_speed == max (3 == 3): `<` は false → max 側 (値は同じ 3 だが `<=`/`>` 変異で経路が変わる)
        give_velocity(&mut world, a, v3(3, 3, 0));

        world.update_velocities(r(1, 4));

        assert_eq!(
            world.bodies[a].velocity,
            Vec3Fix::new(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn update_velocities_sensor_on_either_side_skips_response() {
        for sensor_side in 0..2 {
            let mut world = quiet_world();
            let a = world.add_body(RigidBody::new_dynamic(v3(0, 0, 0), Fix128::ONE));
            let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
            world.bodies[if sensor_side == 0 { a } else { b }].is_sensor = true;
            push_contact(&mut world, a, b, r(1, 2), r(1, 2));
            freeze_positions(&mut world);
            give_velocity(&mut world, a, v3(-1, 2, 0));

            world.update_velocities(r(1, 4));

            assert_eq!(
                world.bodies[a].velocity,
                v3(-1, 2, 0),
                "sensor side {sensor_side}"
            );
            assert_eq!(
                world.bodies[b].velocity,
                Vec3Fix::ZERO,
                "sensor side {sensor_side}"
            );
        }
    }

    #[test]
    fn update_velocities_two_static_bodies_are_skipped() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(v3(0, 0, 0)));
        let b = world.add_body(RigidBody::new_static(v3(1, 0, 0)));
        push_contact(&mut world, a, b, Fix128::ONE, Fix128::ONE);
        // static は Phase 1 で skip、Phase 2 は w_sum < ε で skip → velocity は不変 (手書き値のまま)
        freeze_positions(&mut world);
        world.bodies[a].velocity = v3(-1, 1, 0);

        world.update_velocities(r(1, 4));

        assert_eq!(world.bodies[a].velocity, v3(-1, 1, 0));
    }

    // ---- raycast ------------------------------------------------------

    /// 半径 2 の球を pos に置いた world (gravity なし)
    fn sphere_world(positions: &[Vec3Fix]) -> PhysicsWorld {
        let mut world = quiet_world();
        for &p in positions {
            world.add_body_with_radius(RigidBody::new_static(p), Fix128::from_int(2));
        }
        world
    }

    #[test]
    fn raycast_hit_distance_is_exact_closed_form() {
        // 原点から +x、球 (10,0,0) r=2 → b=-10, c=96, disc=4, t = 10 - 2 = 8
        let world = sphere_world(&[v3(10, 0, 0)]);
        let hit = world.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100));
        assert_eq!(hit, Some((0, Fix128::from_int(8))));
        // direction は正規化される: (3,0,0) でも同じ t
        let hit2 = world.raycast(Vec3Fix::ZERO, v3(3, 0, 0), Fix128::from_int(100));
        assert_eq!(hit2, Some((0, Fix128::from_int(8))));
    }

    #[test]
    fn raycast_along_every_axis_and_direction() {
        // ray AABB の min/max 選択 (x/y/z × </>) を全て通す
        for (dir, pos) in [
            (v3(1, 0, 0), v3(10, 0, 0)),
            (v3(-1, 0, 0), v3(-10, 0, 0)),
            (v3(0, 1, 0), v3(0, 10, 0)),
            (v3(0, -1, 0), v3(0, -10, 0)),
            (v3(0, 0, 1), v3(0, 0, 10)),
            (v3(0, 0, -1), v3(0, 0, -10)),
        ] {
            let world = sphere_world(&[pos]);
            assert_eq!(
                world.raycast(Vec3Fix::ZERO, dir, Fix128::from_int(100)),
                Some((0, Fix128::from_int(8))),
                "dir {dir:?}"
            );
            // 逆向きは外れる
            assert_eq!(
                world.raycast(
                    Vec3Fix::ZERO,
                    dir * Fix128::from_int(-1),
                    Fix128::from_int(100)
                ),
                None
            );
        }
    }

    #[test]
    fn raycast_origin_on_surface_hits_at_zero_not_far_side() {
        // origin (8,0,0)、球 (10,0,0) r=2: oc=(-2,0,0), b=-2, c=0, disc=4 → t_near = 0 (`<` は false)
        let world = sphere_world(&[v3(10, 0, 0)]);
        let hit = world.raycast(v3(8, 0, 0), v3(1, 0, 0), Fix128::from_int(100));
        assert_eq!(hit, Some((0, Fix128::ZERO)));
    }

    #[test]
    fn raycast_from_inside_sphere_uses_far_intersection() {
        // origin = 球心: b=0, c=-4, disc=4 → t_near = -2 < 0 → t_far = 2
        let world = sphere_world(&[v3(10, 0, 0)]);
        let hit = world.raycast(v3(10, 0, 0), v3(1, 0, 0), Fix128::from_int(100));
        assert_eq!(hit, Some((0, Fix128::from_int(2))));
    }

    #[test]
    fn raycast_max_distance_boundary_inclusive() {
        let world = sphere_world(&[v3(10, 0, 0)]);
        // t = 8 == max → hit (`>` は false)
        assert_eq!(
            world.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(8)),
            Some((0, Fix128::from_int(8)))
        );
        // t = 8 > max 7 → miss (ray AABB は球に届かない or 距離で reject)
        assert_eq!(
            world.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(7)),
            None
        );
        // AABB は届くが距離で reject: max = 7.9 (AABB は球 [8,12] と 7.9 で非交差)、
        // 代わりに max = 9 で AABB 交差 + t=8 ≤ 9 → hit、max = 8 - ε 相当は上で網羅
        assert_eq!(
            world.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(9)),
            Some((0, Fix128::from_int(8)))
        );
    }

    #[test]
    fn raycast_returns_nearest_regardless_of_body_order() {
        // 手前 (10) と奥 (20) の球、index 順を両方試す
        let near = v3(10, 0, 0);
        let far = v3(20, 0, 0);
        let w1 = sphere_world(&[far, near]);
        assert_eq!(
            w1.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            Some((1, Fix128::from_int(8)))
        );
        let w2 = sphere_world(&[near, far]);
        assert_eq!(
            w2.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            Some((0, Fix128::from_int(8)))
        );
    }

    #[test]
    fn raycast_misses_sphere_beside_the_ray() {
        // 球 (10, 5, 0) r=2: 最接近距離 5 > 2 → disc < 0 → None
        let world = sphere_world(&[v3(10, 5, 0)]);
        assert_eq!(
            world.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            None
        );
        // 球 (10, 2, 0): 接線 (disc = 0) → t = 10 で hit
        let tangent = sphere_world(&[v3(10, 2, 0)]);
        assert_eq!(
            tangent.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            Some((0, Fix128::from_int(10)))
        );
    }

    #[test]
    fn raycast_zero_direction_and_no_collidable_bodies_return_none() {
        let world = sphere_world(&[v3(10, 0, 0)]);
        assert_eq!(
            world.raycast(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(100)),
            None
        );
        let mut no_radius = quiet_world();
        no_radius.add_body(RigidBody::new_static(v3(10, 0, 0)));
        assert_eq!(
            no_radius.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            None
        );
        // radius を後から消すと当たらなくなる
        let mut cleared = sphere_world(&[v3(10, 0, 0)]);
        cleared.clear_body_collision_radius(0);
        assert_eq!(
            cleared.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            None
        );
        // set で復活
        cleared.set_body_collision_radius(0, Fix128::from_int(2));
        assert_eq!(
            cleared.raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(100)),
            Some((0, Fix128::from_int(8)))
        );
    }

    // ---- remove_body ---------------------------------------------------

    fn dc(a: usize, b: usize) -> DistanceConstraint {
        DistanceConstraint {
            body_a: a,
            body_b: b,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::ONE,
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        }
    }

    fn three_body_world_with_all_pairs() -> PhysicsWorld {
        let mut world = quiet_world();
        for i in 0..3 {
            world.add_body(RigidBody::new_dynamic(v3(i, 0, 0), Fix128::ONE));
        }
        world.add_distance_constraint(dc(0, 1));
        world.add_distance_constraint(dc(1, 2));
        world.add_distance_constraint(dc(0, 2));
        world.contact_constraints.push(ContactConstraint::new(
            0,
            1,
            Contact {
                depth: Fix128::ZERO,
                normal: v3(1, 0, 0),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
        ));
        world.contact_constraints.push(ContactConstraint::new(
            1,
            2,
            Contact {
                depth: Fix128::ZERO,
                normal: v3(1, 0, 0),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
        ));
        world.contact_constraints.push(ContactConstraint::new(
            0,
            2,
            Contact {
                depth: Fix128::ZERO,
                normal: v3(1, 0, 0),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
        ));
        world
    }

    fn dist_pairs(world: &PhysicsWorld) -> Vec<(usize, usize)> {
        world
            .distance_constraints
            .iter()
            .map(|c| (c.body_a, c.body_b))
            .collect()
    }

    fn contact_pairs(world: &PhysicsWorld) -> Vec<(usize, usize)> {
        world
            .contact_constraints
            .iter()
            .map(|c| (c.body_a, c.body_b))
            .collect()
    }

    #[test]
    fn remove_body_middle_drops_its_constraints_and_remaps_last() {
        let mut world = three_body_world_with_all_pairs();
        let removed = world.remove_body(1).expect("index 1 exists");
        assert_eq!(removed.position, v3(1, 0, 0));
        assert_eq!(world.bodies.len(), 2);
        // 旧 A2 が index 1 に移動
        assert_eq!(world.bodies[1].position, v3(2, 0, 0));
        // A1 に触る 2 本は消え、(0-2) だけが (0,1) に remap
        assert_eq!(dist_pairs(&world), vec![(0, 1)]);
        assert_eq!(contact_pairs(&world), vec![(0, 1)]);
        assert_eq!(world.body_materials.len(), 2);
        assert_eq!(world.body_collision_radii.len(), 2);
        assert_eq!(world.body_filters.len(), 2);
    }

    #[test]
    fn remove_body_first_remaps_last_into_slot_zero() {
        let mut world = three_body_world_with_all_pairs();
        world.remove_body(0).expect("index 0 exists");
        assert_eq!(world.bodies[0].position, v3(2, 0, 0));
        assert_eq!(world.bodies[1].position, v3(1, 0, 0));
        // 残るのは (1-2) → (1,0)
        assert_eq!(dist_pairs(&world), vec![(1, 0)]);
        assert_eq!(contact_pairs(&world), vec![(1, 0)]);
    }

    #[test]
    fn remove_body_last_needs_no_remap() {
        let mut world = three_body_world_with_all_pairs();
        world.remove_body(2).expect("index 2 exists");
        assert_eq!(world.bodies.len(), 2);
        assert_eq!(world.bodies[1].position, v3(1, 0, 0));
        assert_eq!(dist_pairs(&world), vec![(0, 1)]);
        assert_eq!(contact_pairs(&world), vec![(0, 1)]);
    }

    #[test]
    fn remove_body_out_of_range_is_none_and_leaves_world_untouched() {
        let mut world = three_body_world_with_all_pairs();
        assert!(world.remove_body(3).is_none());
        assert!(world.remove_body(usize::MAX).is_none());
        assert_eq!(world.bodies.len(), 3);
        assert_eq!(dist_pairs(&world), vec![(0, 1), (1, 2), (0, 2)]);
    }

    #[test]
    fn remove_body_never_leaves_self_or_dangling_references() {
        // 4 body 全 pair、各 index を順に消して不変条件を検査
        for victim in 0..4 {
            let mut world = quiet_world();
            for i in 0..4 {
                world.add_body(RigidBody::new_dynamic(v3(i, 0, 0), Fix128::ONE));
            }
            for a in 0..4 {
                for b in (a + 1)..4 {
                    world.add_distance_constraint(dc(a, b));
                }
            }
            world.remove_body(victim).expect("in range");
            let n = world.bodies.len();
            assert_eq!(n, 3);
            // 残 3 body の全 pair = 3 本、自己参照なし、範囲内
            assert_eq!(world.distance_constraints.len(), 3, "victim {victim}");
            for c in &world.distance_constraints {
                assert!(
                    c.body_a < n && c.body_b < n,
                    "victim {victim}: dangling {:?}",
                    (c.body_a, c.body_b)
                );
                assert_ne!(c.body_a, c.body_b, "victim {victim}: self constraint");
            }
            // 残った body の position 集合 = 元 4 点から victim を除いたもの
            let mut xs: Vec<i64> = world.bodies.iter().map(|b| b.position.x.hi).collect();
            xs.sort_unstable();
            let mut expected: Vec<i64> = (0..4).filter(|&i| i != victim as i64).collect();
            expected.sort_unstable();
            assert_eq!(xs, expected);
        }
    }

    #[test]
    fn remove_body_with_joint_drops_joint_and_rebuilds_islands() {
        let mut world = quiet_world();
        for i in 0..3 {
            world.add_body(RigidBody::new_dynamic(v3(i, 0, 0), Fix128::ONE));
        }
        let j = crate::joint::BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO);
        world.add_joint(Joint::Ball(j));
        let j2 = crate::joint::BallJoint::new(0, 2, Vec3Fix::ZERO, Vec3Fix::ZERO);
        world.add_joint(Joint::Ball(j2));
        assert_eq!(world.joint_count(), 2);
        world.remove_body(1).expect("in range");
        // (1-2) は消え、(0-2) は (0,1) に remap
        assert_eq!(world.joint_count(), 1);
        assert_eq!(world.joints[0].bodies(), (0, 1));
        // island は新 body 数で再構築され、joint で 0 と 1 が同一 island
        assert_eq!(world.islands.find(0), world.islands.find(1));
    }

    // ---- detect_collisions --------------------------------------------

    /// |a - b| が 2^-60 未満 (normalize の sqrt 丸め 数 ulp を許容、変異の差は桁違いなので検出力は不変)
    fn near(a: Fix128, b: Fix128) -> bool {
        let d = (a - b).abs();
        d.hi == 0 && d.lo < (1u64 << 4)
    }

    fn assert_near_vec(a: Vec3Fix, b: Vec3Fix, what: &str) {
        assert!(
            near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z),
            "{what}: {a:?} vs {b:?}"
        );
    }

    /// 半径 r の dynamic 球 2 個を x 軸上 dist 離して置く
    fn two_spheres(dist_num: i64, dist_den: i64, r_a: Fix128, r_b: Fix128) -> PhysicsWorld {
        let mut world = quiet_world();
        world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r_a);
        world.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(r(dist_num, dist_den), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            r_b,
        );
        world
    }

    #[test]
    fn detect_collisions_overlapping_spheres_produce_exact_contact() {
        // r=1, r=1、距離 1.5 → depth = 0.5、normal = -x (B → A、`Contact::normal` 契約)、
        // point_a = pos_a - n·r_a = (1,0,0)、point_b = pos_b + n·r_b = (0.5,0,0)
        // (1.2.0 以前は a → b で pin していた = 押し込み方向の bug を test が固定していた)
        let mut world = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        world.detect_collisions();
        assert_eq!(world.contact_constraints.len(), 1);
        let c = &world.contact_constraints[0];
        assert_eq!((c.body_a, c.body_b), (0, 1));
        assert!(
            near(c.contact.depth, r(1, 2)),
            "depth {:?}",
            c.contact.depth
        );
        assert_near_vec(c.contact.normal, v3(-1, 0, 0), "normal");
        assert_near_vec(c.contact.point_a, v3(1, 0, 0), "point_a");
        assert_near_vec(
            c.contact.point_b,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO),
            "point_b",
        );
        // event も 1 件、rel_vel は 0
        assert_eq!(world.contact_events().len(), 1);
        assert!(world.trigger_events().is_empty());
    }

    #[test]
    fn detect_collisions_asymmetric_radii_split_points_correctly() {
        // r_a = 2, r_b = 1/2、距離 2 → combined 2.5、depth 0.5
        let mut world = two_spheres(2, 1, Fix128::from_int(2), r(1, 2));
        world.detect_collisions();
        assert_eq!(world.contact_constraints.len(), 1);
        let c = world.contact_constraints[0].contact;
        assert!(near(c.depth, r(1, 2)), "depth {:?}", c.depth);
        assert_near_vec(c.point_a, v3(2, 0, 0), "point_a"); // a.pos + n * r_a
        assert_near_vec(
            c.point_b,
            Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO),
            "point_b",
        ); // b.pos - n * r_b
    }

    #[test]
    fn detect_collisions_touching_exactly_is_not_a_contact() {
        // dist == combined (2 == 1+1): `<` は false → 接触なし (`<=` 変異はここで死ぬ)
        let mut world = two_spheres(2, 1, Fix128::ONE, Fix128::ONE);
        world.detect_collisions();
        assert!(world.contact_constraints.is_empty());
        assert!(world.contact_events().is_empty());
        // 少しでも近ければ接触
        let mut world2 = two_spheres(199, 100, Fix128::ONE, Fix128::ONE);
        world2.detect_collisions();
        assert_eq!(world2.contact_constraints.len(), 1);
    }

    #[test]
    fn detect_collisions_coincident_centers_are_skipped() {
        // dist == 0 → normal 不定なので skip (`!dist.is_zero()`)
        let mut world = two_spheres(0, 1, Fix128::ONE, Fix128::ONE);
        world.detect_collisions();
        assert!(world.contact_constraints.is_empty());
    }

    #[test]
    fn detect_collisions_relative_velocity_is_reported_along_normal() {
        let mut world = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        world.bodies[0].velocity = v3(2, 7, 0); // a → b 方向 +2 (y 成分は normal 直交で無視)
        world.bodies[1].velocity = v3(-1, 0, 0);
        world.detect_collisions();
        let ev = &world.contact_events()[0];
        // rel_vel = (v_a - v_b) · n = (3, 7, 0) · (-1,0,0) = -3 (接近中は負、B → A normal)
        assert!(
            near(ev.relative_velocity, Fix128::from_int(-3)),
            "rel_vel {:?}",
            ev.relative_velocity
        );
        assert!(near(ev.depth, r(1, 2)), "depth {:?}", ev.depth);
    }

    #[test]
    fn detect_collisions_sensor_reports_trigger_instead_of_constraint() {
        for sensor_side in 0..2 {
            let mut world = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
            world.bodies[sensor_side].is_sensor = true;
            world.detect_collisions();
            assert!(world.contact_constraints.is_empty(), "side {sensor_side}");
            assert_eq!(world.trigger_events().len(), 1, "side {sensor_side}");
            assert_eq!(world.contact_events().len(), 1, "side {sensor_side}");
        }
    }

    #[test]
    fn detect_collisions_static_static_pair_is_skipped_but_static_dynamic_is_not() {
        let mut world = quiet_world();
        world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
        world.add_body_with_radius(
            RigidBody::new_static(Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO)),
            Fix128::ONE,
        );
        world.detect_collisions();
        assert!(world.contact_constraints.is_empty());

        let mut mixed = quiet_world();
        mixed.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
        mixed.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            Fix128::ONE,
        );
        mixed.detect_collisions();
        assert_eq!(mixed.contact_constraints.len(), 1);
    }

    #[test]
    fn detect_collisions_filter_blocks_same_group_and_masked_layers() {
        // 同一 group (非 0) → 衝突しない
        let mut same_group = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        let g = CollisionFilter {
            layer: 1,
            mask: u32::MAX,
            group: 7,
        };
        same_group.set_body_filter(0, g);
        same_group.set_body_filter(1, g);
        same_group.detect_collisions();
        assert!(same_group.contact_constraints.is_empty());

        // 片方向 mask 不一致 (a.layer & b.mask == 0) → 衝突しない (`&&` の右側も検査)
        for side in 0..2 {
            let mut masked = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
            masked.set_body_filter(
                side,
                CollisionFilter {
                    layer: 1 << 5,
                    mask: u32::MAX,
                    group: 0,
                },
            );
            masked.set_body_filter(
                1 - side,
                CollisionFilter {
                    layer: 1,
                    mask: !(1 << 5),
                    group: 0,
                },
            );
            masked.detect_collisions();
            assert!(masked.contact_constraints.is_empty(), "side {side}");
        }

        // 異なる group + 相互 mask 一致 → 衝突する
        let mut ok = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        ok.set_body_filter(
            0,
            CollisionFilter {
                layer: 1,
                mask: u32::MAX,
                group: 1,
            },
        );
        ok.set_body_filter(
            1,
            CollisionFilter {
                layer: 1,
                mask: u32::MAX,
                group: 2,
            },
        );
        ok.detect_collisions();
        assert_eq!(ok.contact_constraints.len(), 1);
    }

    #[test]
    fn detect_collisions_both_sleeping_skipped_one_sleeping_wakes() {
        let mut both = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        both.islands.sleep_data[0].state = crate::sleeping::SleepState::Sleeping;
        both.islands.sleep_data[1].state = crate::sleeping::SleepState::Sleeping;
        both.detect_collisions();
        assert!(both.contact_constraints.is_empty());

        let mut one = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        one.islands.sleep_data[0].state = crate::sleeping::SleepState::Sleeping;
        one.detect_collisions();
        assert_eq!(one.contact_constraints.len(), 1);
        // 接触で起こされる
        assert!(!one.islands.is_sleeping(0));
    }

    #[test]
    fn detect_collisions_needs_two_collidable_bodies() {
        // body 1 個 / radius 付き 1 個だけ → 何も起きない
        let mut single = quiet_world();
        single.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            Fix128::ONE,
        );
        single.detect_collisions();
        assert!(single.contact_constraints.is_empty());

        let mut one_radius = quiet_world();
        one_radius.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            Fix128::ONE,
        );
        one_radius.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        one_radius.detect_collisions();
        assert!(one_radius.contact_constraints.is_empty());
    }

    #[test]
    fn detect_collisions_uses_combined_material_of_pair() {
        let mut world = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        world.detect_collisions();
        let c = world.contact_constraints[0];
        let combined = world.combined_material(0, 1);
        assert_eq!(c.friction, combined.friction);
        assert_eq!(c.restitution, combined.restitution);
    }

    // ---- solve_distance_constraints -----------------------------------

    #[test]
    fn solve_distance_constraints_rigid_splits_error_by_inverse_mass() {
        // A (inv 1) at 0、B (inv 3) at x=4、target 2 → error 2、w_sum 4、lambda 1/2
        // A += n*λ*1 = +0.5、B -= n*λ*3 = -1.5 → 距離 4 - 2 = 2 で一発収束
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(4, 0, 0), Fix128::ONE));
        world.bodies[b].inv_mass = Fix128::from_int(3);
        let mut c = dc(a, b);
        c.target_distance = Fix128::from_int(2);
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(
            world.bodies[a].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[b].position,
            Vec3Fix::new(r(5, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(world.distance_constraints[0].cached_lambda, r(1, 2));
    }

    #[test]
    fn solve_distance_constraints_static_side_does_not_move() {
        // A static at 0、B (inv 1) at 4、target 1 → error 3、w_sum 1 → B -= 3 → x = 1
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new_dynamic(v3(4, 0, 0), Fix128::ONE));
        world.add_distance_constraint(dc(a, b)); // target 1

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, v3(1, 0, 0));
    }

    #[test]
    fn solve_distance_constraints_negative_error_pushes_apart() {
        // 距離 1 < target 3 → error -2、両 inv 1 → w_sum 2、λ = -1 → A -= 1 (x=-1)、B += 1 (x=2)
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let mut c = dc(a, b);
        c.target_distance = Fix128::from_int(3);
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(world.bodies[a].position, v3(-1, 0, 0));
        assert_eq!(world.bodies[b].position, v3(2, 0, 0));
    }

    #[test]
    fn solve_distance_constraints_compliance_softens_correction() {
        // compliance 1/16、dt 1/4 → compliance_term = (1/16)/(1/16) = 1
        // A static、B inv 1 at 4、target 2 → error 2、w_sum = 1 + 1 = 2、λ = 1 → B: 4 - 1 = 3
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new_dynamic(v3(4, 0, 0), Fix128::ONE));
        let mut c = dc(a, b);
        c.target_distance = Fix128::from_int(2);
        c.compliance = r(1, 16);
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(world.bodies[b].position, v3(3, 0, 0));
        assert_eq!(world.distance_constraints[0].cached_lambda, Fix128::ONE);
    }

    #[test]
    fn solve_distance_constraints_accumulates_lambda_xpbd() {
        // substep 途中 (前 iteration で λ = 1 が累積済)、compliance_term = 1 (上と同設定)
        // dλ = (C - α̃λ) / w_sum = (2 - 1*1) / 2 = 1/2 → B: 4 - 0.5 = 3.5、λ = 1 + 1/2 = 3/2
        // (1.2.0 以前は λ を上書きしていたので cached_lambda = 1/2 になっていた)
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new_dynamic(v3(4, 0, 0), Fix128::ONE));
        let mut c = dc(a, b);
        c.target_distance = Fix128::from_int(2);
        c.compliance = r(1, 16);
        c.cached_lambda = Fix128::ONE;
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(
            world.bodies[b].position,
            Vec3Fix::new(r(7, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(world.distance_constraints[0].cached_lambda, r(3, 2));
    }

    #[test]
    fn reset_lambdas_zeroes_distance_lambda() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new_dynamic(v3(4, 0, 0), Fix128::ONE));
        let mut c = dc(a, b);
        c.cached_lambda = Fix128::from_int(3);
        world.add_distance_constraint(c);
        world.reset_lambdas();
        assert_eq!(world.distance_constraints[0].cached_lambda, Fix128::ZERO);
    }

    #[test]
    fn xpbd_compliant_extension_is_iteration_independent() {
        // 1 kg を compliance 0.01 (k = 100 N/m) の距離拘束で吊る → 定常伸び mg/k = 0.1 m
        // iterations を 1..16 と変えても同じ伸びに収束する (1.2.0 以前は iter 倍で伸び半減)
        let run = |iters: usize| {
            let mut world = quiet_world();
            world.config.gravity = v3(0, -10, 0);
            world.config.damping = Fix128::ONE;
            world.config.substeps = 4;
            world.config.iterations = iters;
            let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            let b = world.add_body(RigidBody::new_dynamic(v3(0, -1, 0), Fix128::ONE));
            world.bodies[b].linear_damping = r(9, 10); // settle the oscillation
            let mut c = dc(a, b);
            c.target_distance = Fix128::ONE;
            c.compliance = r(1, 100);
            world.add_distance_constraint(c);
            for _ in 0..600 {
                world.step(r(1, 60));
            }
            (-world.bodies[b].position.y - Fix128::ONE).to_f64()
        };
        let ext: Vec<f64> = [1usize, 2, 4, 8, 16].iter().map(|&i| run(i)).collect();
        for (i, e) in ext.iter().enumerate() {
            assert!(
                (e - 0.1).abs() < 0.01,
                "iterations index {i}: extension {e} m, expected 0.1 ± 0.01 (all: {ext:?})"
            );
        }
    }

    #[test]
    fn solve_distance_constraints_anchor_offsets_are_rotated_into_world() {
        // B に local anchor (0,1,0)、B を z 軸 π 回転 → world anchor は B.pos + (0,-1,0)
        // A static at 0、B at (0,2,0) → anchor_b = (0,1,0)、距離 1 == target 1 → 補正なし
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new_dynamic(v3(0, 2, 0), Fix128::ONE));
        world.bodies[b].rotation =
            QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let mut c = dc(a, b);
        c.local_anchor_b = v3(0, 1, 0);
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_eq!(world.bodies[b].position, v3(0, 2, 0));
        // 回転していなければ anchor は (0,3,0)、距離 3 → error 2 → B は -2 動く
        let mut unrotated = quiet_world();
        let a2 = unrotated.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b2 = unrotated.add_body(RigidBody::new_dynamic(v3(0, 2, 0), Fix128::ONE));
        let mut c2 = dc(a2, b2);
        c2.local_anchor_b = v3(0, 1, 0);
        unrotated.add_distance_constraint(c2);
        unrotated.solve_distance_constraints(r(1, 4));
        assert_near_vec(unrotated.bodies[b2].position, v3(0, 0, 0), "unrotated B");
        // normalize の sqrt 丸め 数 ulp
    }

    #[test]
    fn solve_distance_constraints_skips_coincident_and_double_static() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)); // 同一点
        world.add_distance_constraint(dc(a, b));
        world.solve_distance_constraints(r(1, 4));
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, Vec3Fix::ZERO);
        assert_eq!(world.distance_constraints[0].cached_lambda, Fix128::ZERO);

        let mut statics = quiet_world();
        let s1 = statics.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let s2 = statics.add_body(RigidBody::new_static(v3(5, 0, 0)));
        statics.add_distance_constraint(dc(s1, s2));
        statics.solve_distance_constraints(r(1, 4));
        assert_eq!(statics.bodies[s2].position, v3(5, 0, 0));
        assert_eq!(statics.distance_constraints[0].cached_lambda, Fix128::ZERO);
    }

    // ---- solve_contact_constraints ------------------------------------

    fn contact_world(inv_a: Fix128, inv_b: Fix128, depth: Fix128) -> PhysicsWorld {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        world.bodies[a].inv_mass = inv_a;
        world.bodies[b].inv_mass = inv_b;
        let contact = Contact {
            depth,
            normal: v3(1, 0, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        world
            .contact_constraints
            .push(ContactConstraint::new(a, b, contact));
        world
    }

    #[test]
    fn solve_contact_constraints_pushes_apart_by_depth_weighted_by_inverse_mass() {
        // inv 1 / inv 3、depth 1 → λ = 1、inv_w = 1/4 → A += n * 1/4、B -= n * 3/4
        let mut world = contact_world(Fix128::ONE, Fix128::from_int(3), Fix128::ONE);
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(
            world.bodies[0].position,
            Vec3Fix::new(r(1, 4), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[1].position,
            Vec3Fix::new(r(1, 4), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(world.contact_constraints[0].cached_lambda, Fix128::ONE);
    }

    #[test]
    fn solve_contact_constraints_zero_or_negative_depth_is_skipped() {
        for depth in [Fix128::ZERO, -r(1, 2)] {
            let mut world = contact_world(Fix128::ONE, Fix128::ONE, depth);
            world.solve_contact_constraints(r(1, 4));
            assert_eq!(world.bodies[0].position, Vec3Fix::ZERO, "depth {depth:?}");
            assert_eq!(world.bodies[1].position, v3(1, 0, 0), "depth {depth:?}");
            assert_eq!(world.contact_constraints[0].cached_lambda, Fix128::ZERO);
        }
    }

    #[test]
    fn solve_contact_constraints_static_side_absorbs_nothing() {
        // A static (inv 0)、B inv 1、depth 1/2 → B -= 1/2、A 不動
        let mut world = contact_world(Fix128::ZERO, Fix128::ONE, r(1, 2));
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);
        assert_eq!(
            world.bodies[1].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
        // 両 static は skip、cached_lambda も 0 のまま
        let mut both = contact_world(Fix128::ZERO, Fix128::ZERO, r(1, 2));
        both.solve_contact_constraints(r(1, 4));
        assert_eq!(both.bodies[1].position, v3(1, 0, 0));
        assert_eq!(both.contact_constraints[0].cached_lambda, Fix128::ZERO);
    }

    #[test]
    fn solve_contact_constraints_accumulates_lambda_and_pushes_depth_once() {
        // depth 1/2、λ 累積済 1 (≥ depth) → dλ ≤ 0 → 動かない、λ 不変
        let mut world = contact_world(Fix128::ONE, Fix128::ONE, r(1, 2));
        world.contact_constraints[0].cached_lambda = Fix128::ONE;
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);
        assert_eq!(world.contact_constraints[0].cached_lambda, Fix128::ONE);

        // λ == depth ちょうど → dλ = 0 → 動かない
        let mut edge = contact_world(Fix128::ONE, Fix128::ONE, r(1, 2));
        edge.contact_constraints[0].cached_lambda = r(1, 2);
        edge.solve_contact_constraints(r(1, 4));
        assert_eq!(edge.bodies[0].position, Vec3Fix::ZERO);

        // λ 1/4 → dλ = 1/2 - 1/4 = 1/4 → A += 1/8 (w_a / w_sum = 1/2)、λ = 1/2
        let mut partial = contact_world(Fix128::ONE, Fix128::ONE, r(1, 2));
        partial.contact_constraints[0].cached_lambda = r(1, 4);
        partial.solve_contact_constraints(r(1, 4));
        assert_eq!(
            partial.bodies[0].position,
            Vec3Fix::new(r(1, 8), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(partial.contact_constraints[0].cached_lambda, r(1, 2));

        // 4 iteration 回しても総押し出し量は depth 1 回分 (1.2.0 以前は iteration 毎に再 push)
        let mut iters = contact_world(Fix128::ONE, Fix128::ONE, r(1, 2));
        for _ in 0..4 {
            iters.solve_contact_constraints(r(1, 4));
        }
        assert_eq!(
            iters.bodies[0].position,
            Vec3Fix::new(r(1, 4), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            iters.bodies[1].position,
            Vec3Fix::new(r(3, 4), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn solve_contact_constraints_sensor_skips_response() {
        for side in 0..2 {
            let mut world = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
            world.bodies[side].is_sensor = true;
            world.solve_contact_constraints(r(1, 4));
            assert_eq!(world.bodies[0].position, Vec3Fix::ZERO, "side {side}");
            assert_eq!(world.bodies[1].position, v3(1, 0, 0), "side {side}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn solve_contact_constraints_pre_solve_hook_can_discard_and_modifier_can_rescale() {
        // hook が false → skip
        let mut vetoed = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        vetoed.add_pre_solve_hook(Box::new(|_a, _b, _c| false));
        vetoed.apply_contact_filters();
        vetoed.solve_contact_constraints(r(1, 4));
        assert_eq!(vetoed.bodies[0].position, Vec3Fix::ZERO);

        // hook が true → 通常通り
        let mut allowed = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        allowed.add_pre_solve_hook(Box::new(|_a, _b, _c| true));
        allowed.apply_contact_filters();
        allowed.solve_contact_constraints(r(1, 4));
        assert_eq!(
            allowed.bodies[0].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );

        // modifier が depth を半分に → λ = 1/2 → A += 1/4
        struct Halve;
        impl ContactModifier for Halve {
            fn modify_contact(
                &self,
                _a: usize,
                _b: usize,
                c: &mut Contact,
                _f: &mut Fix128,
                _r: &mut Fix128,
            ) -> bool {
                c.depth = c.depth.half();
                true
            }
        }
        let mut halved = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        halved.add_contact_modifier(Box::new(Halve));
        halved.apply_contact_filters();
        halved.solve_contact_constraints(r(1, 4));
        assert_eq!(
            halved.bodies[0].position,
            Vec3Fix::new(r(1, 4), Fix128::ZERO, Fix128::ZERO)
        );

        // modifier が false → 捨てる
        struct Discard;
        impl ContactModifier for Discard {
            fn modify_contact(
                &self,
                _a: usize,
                _b: usize,
                _c: &mut Contact,
                _f: &mut Fix128,
                _r: &mut Fix128,
            ) -> bool {
                false
            }
        }
        let mut discarded = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        discarded.add_contact_modifier(Box::new(Discard));
        discarded.apply_contact_filters();
        discarded.solve_contact_constraints(r(1, 4));
        assert_eq!(discarded.bodies[0].position, Vec3Fix::ZERO);
    }

    // ---- RigidBody force / torque / impulse ---------------------------

    /// inv_mass 2、inv_inertia (1, 2, 4) の body、初期 velocity / angular を非零にして加算を観測
    fn loaded_body() -> RigidBody {
        let mut b = RigidBody::new_dynamic(v3(1, 2, 3), Fix128::ONE);
        b.inv_mass = Fix128::from_int(2);
        b.inv_inertia = v3(1, 2, 4);
        b.velocity = v3(10, 20, 30);
        b.angular_velocity = v3(-1, -2, -3);
        b
    }

    #[test]
    fn add_force_scales_by_inverse_mass_and_dt() {
        let mut b = loaded_body();
        b.add_force(v3(4, -8, 16), r(1, 4)); // Δv = F * 2 * 1/4 = (2, -4, 8)
        assert_eq!(b.velocity, v3(12, 16, 38));
        assert_eq!(b.angular_velocity, v3(-1, -2, -3));
        let mut st = RigidBody::new_static(Vec3Fix::ZERO);
        st.add_force(v3(4, 4, 4), Fix128::ONE);
        assert_eq!(st.velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn add_torque_scales_each_axis_by_inverse_inertia_and_dt() {
        let mut b = loaded_body();
        b.add_torque(v3(8, 8, 8), r(1, 4)); // Δω = (8*1, 8*2, 8*4) * 1/4 = (2, 4, 8)
        assert_eq!(b.angular_velocity, v3(1, 2, 5));
        assert_eq!(b.velocity, v3(10, 20, 30));
        let mut st = RigidBody::new_static(Vec3Fix::ZERO);
        st.add_torque(v3(8, 8, 8), Fix128::ONE);
        assert_eq!(st.angular_velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn apply_impulse_scales_by_inverse_mass_only() {
        let mut b = loaded_body();
        b.apply_impulse(v3(1, 2, 3)); // Δv = (2, 4, 6)
        assert_eq!(b.velocity, v3(12, 24, 36));
        assert_eq!(b.angular_velocity, v3(-1, -2, -3));
        let mut st = RigidBody::new_static(Vec3Fix::ZERO);
        st.apply_impulse(v3(1, 1, 1));
        assert_eq!(st.velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn apply_impulse_at_adds_lever_arm_torque() {
        // position (1,2,3)、point (2,2,3) → r = (1,0,0)、impulse (0,3,0) → r × J = (0,0,3)
        // Δω = (0*1, 0*2, 3*4) = (0,0,12)、Δv = J * 2 = (0,6,0)
        let mut b = loaded_body();
        b.apply_impulse_at(v3(0, 3, 0), v3(2, 2, 3));
        assert_eq!(b.velocity, v3(10, 26, 30));
        assert_eq!(b.angular_velocity, v3(-1, -2, 9));

        // r = (0,1,0)、J = (0,0,5) → r × J = (5,0,0) → Δω = (5,0,0)
        let mut c = loaded_body();
        c.apply_impulse_at(v3(0, 0, 5), v3(1, 3, 3));
        assert_eq!(c.angular_velocity, v3(4, -2, -3));
        assert_eq!(c.velocity, v3(10, 20, 40));

        // r = (0,0,1)、J = (7,0,0) → r × J = (0,7,0) → Δω = (0,14,0)
        let mut d = loaded_body();
        d.apply_impulse_at(v3(7, 0, 0), v3(1, 2, 4));
        assert_eq!(d.angular_velocity, v3(-1, 12, -3));

        // 着力点 = 重心 → torque なし
        let mut e = loaded_body();
        e.apply_impulse_at(v3(7, 0, 0), v3(1, 2, 3));
        assert_eq!(e.angular_velocity, v3(-1, -2, -3));
        assert_eq!(e.velocity, v3(24, 20, 30));

        let mut st = RigidBody::new_static(Vec3Fix::ZERO);
        st.apply_impulse_at(v3(1, 1, 1), v3(1, 0, 0));
        assert_eq!(st.velocity, Vec3Fix::ZERO);
        assert_eq!(st.angular_velocity, Vec3Fix::ZERO);
    }

    // ---- serialize / deserialize ---------------------------------------

    fn snapshot_world() -> PhysicsWorld {
        let mut world = quiet_world();
        let mut a = RigidBody::new_dynamic(v3(1, 2, 3), Fix128::ONE);
        a.velocity = v3(4, 5, 6);
        a.angular_velocity = v3(7, 8, 9);
        a.rotation = QuatFix::from_axis_angle(v3(1, 1, 0), r(1, 3));
        world.add_body(a);
        let mut b = RigidBody::new_dynamic(v3(-1, -2, -3), Fix128::ONE);
        b.velocity = Vec3Fix::new(r(1, 3), r(-2, 7), r(5, 11));
        world.add_body(b);
        world
    }

    #[test]
    fn deserialize_state_roundtrip_restores_every_field_bit_exact() {
        let src = snapshot_world();
        let bytes = src.serialize_state();
        // v3: header 12 + body ごと 208 + body ごと sleep 5 + world ごと flag 1 + fingerprint 8
        assert_eq!(bytes.len(), 12 + 2 * 208 + 2 * 5 + 1 + 8);
        let mut dst = quiet_world();
        dst.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        dst.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        assert!(dst.deserialize_state(&bytes));
        for i in 0..2 {
            assert_eq!(dst.bodies[i].position, src.bodies[i].position, "pos {i}");
            assert_eq!(dst.bodies[i].velocity, src.bodies[i].velocity, "vel {i}");
            assert_eq!(dst.bodies[i].rotation, src.bodies[i].rotation, "rot {i}");
            assert_eq!(
                dst.bodies[i].angular_velocity, src.bodies[i].angular_velocity,
                "ang {i}"
            );
        }
        // 並列配列は body 数に同期
        assert_eq!(dst.body_collision_radii.len(), 2);
        assert_eq!(dst.body_filters.len(), 2);
        assert_eq!(dst.body_materials.len(), 2);
    }

    #[test]
    fn deserialize_state_every_single_byte_flip_changes_some_field() {
        // 1 byte でも壊れた snapshot は必ずどれかの field に反映される (silent 無視 / 別 field 混入を検出)
        //
        // ⚠️ v3 format は 5 領域に分かれる — **どの領域にも silent に無視される
        // byte が無い**ことを領域ごとに確かめる
        //   [0..12)                              header (magic / version / reserved) → **拒否される**
        //   [12, 12+n*208)                       body の運動状態 → 該当 body の該当 field が変わる
        //   [12+n*208, 12+n*208+n*5)             sleep 状態 → 該当 body の sleep_data が変わる
        //   [12+n*208+n*5, 12+n*208+n*5+1)       overflow flag (1 byte)
        //   [12+n*208+n*5+1, ..+9)               population fingerprint (8 byte) → **拒否される**
        let src = snapshot_world();
        let bytes = src.serialize_state();
        let body_end = 12 + 2 * 208;
        let flag_pos = bytes.len() - 9;

        // 領域 1: header — 1 byte 壊れたら受け付けない
        for pos in 0..12 {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(
                !dst.deserialize_state(&bad),
                "header byte {pos} を壊しても受け付けている (silent に無視される byte がある)"
            );
        }

        // 領域 4: flag (fingerprint 直前の 1 byte) — 壊したら overflow_detected が変わる
        //
        // ⚠️ v2 で足した領域も「silent に無視される byte が無い」不変条件の
        // 対象に含める (flag は 0/1 なので 0x01 の flip で必ず値が変わる)
        {
            let pos = flag_pos;
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(dst.deserialize_state(&bad), "flag byte {pos}");
            assert_ne!(
                dst.overflow_detected(),
                src.overflow_detected(),
                "flag byte {pos} を壊しても overflow_detected が変わらない"
            );
        }

        // 領域 5: population fingerprint (末尾 8 byte) — 1 byte でも壊れたら
        // 拒否する (= count は一致しているが fingerprint 不一致で fail-fast、
        // gap #3 の population 検査そのもの)
        for pos in bytes.len() - 8..bytes.len() {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(
                !dst.deserialize_state(&bad),
                "fingerprint byte {pos} を壊しても受け付けている"
            );
            // 拒否時は元の状態を保つ
            assert_eq!(dst.bodies[0].position, src.bodies[0].position);
        }

        // 領域 3: sleep — 1 byte 壊れたら sleep_data が変わる (flag の手前まで)
        for pos in body_end..flag_pos {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(dst.deserialize_state(&bad), "sleep byte {pos}");
            let i = (pos - body_end) / 5;
            assert_ne!(
                dst.islands.sleep_data[i], src.islands.sleep_data[i],
                "sleep byte {pos} は body {i} の sleep_data を変えるべき"
            );
        }

        // 領域 2: body の運動状態
        for pos in 12..body_end {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(dst.deserialize_state(&bad), "byte {pos}");
            let body = (pos - 12) / 208;
            let field = ((pos - 12) % 208) / 48; // 0 pos / 1 vel / 2 rot(64 byte = 2 slot 相当は 128..192) / 3 ang
            let changed_body = (0..2)
                .filter(|&i| {
                    let (s, d) = (&src.bodies[i], &dst.bodies[i]);
                    s.position != d.position
                        || s.velocity != d.velocity
                        || s.rotation != d.rotation
                        || s.angular_velocity != d.angular_velocity
                })
                .collect::<Vec<_>>();
            assert_eq!(
                changed_body,
                vec![body],
                "byte {pos} must change only body {body}"
            );
            let (s, d) = (&src.bodies[body], &dst.bodies[body]);
            let off = (pos - 12) % 208;
            match off {
                0..=47 => assert!(
                    s.position != d.position && s.velocity == d.velocity,
                    "byte {pos} → position"
                ),
                48..=95 => assert!(
                    s.velocity != d.velocity && s.position == d.position,
                    "byte {pos} → velocity"
                ),
                96..=159 => assert!(
                    s.rotation != d.rotation && s.angular_velocity == d.angular_velocity,
                    "byte {pos} → rotation"
                ),
                _ => assert!(
                    s.angular_velocity != d.angular_velocity && s.rotation == d.rotation,
                    "byte {pos} → angular"
                ),
            }
            let _ = field;
        }
    }

    #[test]
    fn deserialize_state_rejects_short_or_mismatched_input() {
        let src = snapshot_world();
        let bytes = src.serialize_state();
        let mut dst = snapshot_world();
        assert!(!dst.deserialize_state(&bytes[..11])); // header 未満 (v1 は 12 byte)
        assert!(!dst.deserialize_state(&bytes[..12 + 208 + 100])); // 2 体目が途中で切れる
        assert!(
            !dst.deserialize_state(&bytes[..bytes.len() - 1]),
            "末尾 (v3 の fingerprint) が 1 byte 欠けても拒否する (長さ検査が body 数から一意に決まる)"
        );
        // ⚠️ 旧 format (先頭が body 数、header なし) は magic 不一致で拒否される
        // 検査しないと **旧 blob の先頭 4 byte を magic と読んで誤解釈する**
        let mut legacy = 2u32.to_le_bytes().to_vec();
        legacy.extend_from_slice(&bytes[12..12 + 2 * 208]);
        assert!(
            !dst.deserialize_state(&legacy),
            "旧 format の blob は magic 不一致で拒否されるべき"
        );
        // ⚠️ v1 blob (magic は合うが version が 1) も拒否される
        // version を検査しないと **flag 1 byte 短い blob を v2 として読む**
        let mut v1 = bytes.clone();
        v1[4] = 1;
        v1.truncate(v1.len() - 1);
        assert!(
            !dst.deserialize_state(&v1),
            "v1 blob は version 不一致で拒否されるべき"
        );
        let mut one_body = quiet_world();
        one_body.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        assert!(!one_body.deserialize_state(&bytes)); // count 不一致
                                                      // 拒否時は元の状態を保つ (先頭 body だけ書き換わっていない)
        assert_eq!(one_body.bodies[0].position, Vec3Fix::ZERO);
        // ぴったり 1 body 分 (header + 運動状態 + sleep) なら OK
        let mut exact = quiet_world();
        exact.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut one = PhysicsWorld::STATE_MAGIC.to_vec();
        one.extend_from_slice(&PhysicsWorld::STATE_VERSION.to_le_bytes());
        one.extend_from_slice(&0u16.to_le_bytes());
        one.extend_from_slice(&1u32.to_le_bytes());
        one.extend_from_slice(&bytes[12..12 + 208]);
        one.extend_from_slice(&bytes[12 + 2 * 208..12 + 2 * 208 + 5]);
        one.push(0); // v2: flag
        one.extend_from_slice(&exact.population_fingerprint().to_le_bytes()); // v3: fingerprint
        assert!(exact.deserialize_state(&one));
        assert_eq!(exact.bodies[0].position, v3(1, 2, 3));
    }

    // ---- integrate_positions -----------------------------------------

    #[test]
    fn integrate_positions_applies_gravity_damping_and_predicts_position() {
        let mut world = quiet_world();
        world.config.gravity = v3(0, -8, 0);
        world.config.damping = r(1, 2);
        let i = world.add_body(RigidBody::new_dynamic(v3(1, 1, 1), Fix128::ONE));
        world.bodies[i].velocity = v3(4, 0, 0);
        world.bodies[i].gravity_scale = r(1, 2);
        world.bodies[i].linear_damping = r(1, 2);
        let dt = r(1, 4);

        world.integrate_positions(dt);

        // v = (4,0,0) + g*scale*dt = (4, -1, 0)  (damping は integrate では掛からない、1.2.0)
        // pos = (1,1,1) + v*dt = (2, 3/4, 1)、prev = (1,1,1)
        let b = &world.bodies[i];
        assert_eq!(
            b.velocity,
            Vec3Fix::new(Fix128::from_int(4), -Fix128::ONE, Fix128::ZERO)
        );
        assert_eq!(
            b.position,
            Vec3Fix::new(Fix128::from_int(2), r(3, 4), Fix128::ONE)
        );
        assert_eq!(b.prev_position, v3(1, 1, 1));

        // frame damping: global 1/2 * per-body 1/2 → (1, -1/4, 0)
        world.apply_frame_damping();
        assert_eq!(
            world.bodies[i].velocity,
            Vec3Fix::new(Fix128::ONE, -r(1, 4), Fix128::ZERO)
        );
    }

    #[test]
    fn integrate_positions_angular_damping_and_rotation_prediction() {
        let mut world = quiet_world();
        world.config.damping = Fix128::ONE;
        let i = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        world.bodies[i].angular_velocity = v3(0, 0, 2);
        world.bodies[i].angular_damping = r(1, 2); // frame damping → (0,0,1)
        let dt = r(1, 4);

        world.integrate_positions(dt);

        let b = &world.bodies[i];
        // integrate は damping を掛けない (1.2.0)、角速度は 2 のまま
        assert_eq!(b.angular_velocity, v3(0, 0, 2));
        // 回転は z 軸 angle = 2 * 1/4 の quaternion (正規化済) と一致
        let expected = QuatFix::from_axis_angle(v3(0, 0, 1), r(1, 2))
            .mul(QuatFix::IDENTITY)
            .normalize();
        assert_eq!(b.rotation, expected);
        assert_eq!(b.prev_rotation, QuatFix::IDENTITY);
        assert!(b.rotation != QuatFix::IDENTITY);
        world.apply_frame_damping();
        assert_eq!(world.bodies[i].angular_velocity, v3(0, 0, 1));
    }

    #[test]
    fn apply_frame_damping_skips_static_kinematic_and_sleeping() {
        let mut world = quiet_world();
        world.config.damping = r(1, 2);
        let st = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mut kin = RigidBody::new_static(Vec3Fix::ZERO);
        kin.body_type = BodyType::Kinematic;
        let ki = world.add_body(kin);
        let dy = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        for &i in &[st, ki, dy] {
            world.bodies[i].velocity = v3(2, 0, 0);
            world.bodies[i].angular_velocity = v3(0, 4, 0);
        }
        world.apply_frame_damping();
        assert_eq!(world.bodies[st].velocity, v3(2, 0, 0));
        assert_eq!(world.bodies[ki].velocity, v3(2, 0, 0));
        assert_eq!(world.bodies[dy].velocity, v3(1, 0, 0));
        assert_eq!(world.bodies[dy].angular_velocity, v3(0, 2, 0));
    }

    #[test]
    fn free_fall_is_substep_independent_with_frame_damping() {
        // 重力 -10、damping ONE、1 秒 (60 frame) → y = -5.0、substeps を変えても同じ
        // (1.2.0 以前は substep 内 damping で終端速度 g*h*d/(1-d) が substeps 依存だった)
        let fall = |substeps: usize| {
            let mut world = quiet_world();
            world.config.gravity = v3(0, -10, 0);
            world.config.damping = Fix128::ONE;
            world.config.substeps = substeps;
            let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
            for _ in 0..60 {
                world.step(r(1, 60));
            }
            world.bodies[b].position.y.to_f64()
        };
        let ys: Vec<f64> = [1usize, 2, 4, 8, 16].iter().map(|&n| fall(n)).collect();
        for y in &ys {
            // symplectic Euler の離散化誤差 g*dt/2 = 1/12 → -5 - 0.083、substeps 増で -5 に寄る
            assert!(
                (y + 5.0).abs() < 0.1,
                "free fall y = {y}, expected -5.0 ± 0.1 (all: {ys:?})"
            );
        }
        // 既定 config (damping 0.99 / frame) でも 1 秒で 3.5 m 以上落ちる (旧: -1.64 m)
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        for _ in 0..60 {
            world.step(r(1, 60));
        }
        let y = world.bodies[b].position.y.to_f64();
        assert!(y < -3.5, "default config fell only to y = {y}");
    }

    #[test]
    fn integrate_positions_static_kinematic_and_sleeping_paths() {
        let mut world = quiet_world();
        world.config.gravity = v3(0, -8, 0);
        let st = world.add_body(RigidBody::new_static(v3(0, 5, 0)));
        let mut kin = RigidBody::new_static(v3(0, 0, 0));
        kin.body_type = BodyType::Kinematic;
        kin.set_kinematic_target(v3(2, 0, 0), QuatFix::IDENTITY);
        let ki = world.add_body(kin);
        let sl = world.add_body(RigidBody::new_dynamic(v3(9, 9, 9), Fix128::ONE));
        world.islands.sleep_data[sl].state = crate::sleeping::SleepState::Sleeping;
        let dt = r(1, 4);

        world.integrate_positions(dt);

        // static: 完全不動
        assert_eq!(world.bodies[st].position, v3(0, 5, 0));
        assert_eq!(world.bodies[st].velocity, Vec3Fix::ZERO);
        // kinematic: target に snap、velocity = Δ/dt = (8,0,0)、prev は旧位置
        assert_eq!(world.bodies[ki].position, v3(2, 0, 0));
        assert_eq!(world.bodies[ki].velocity, v3(8, 0, 0));
        assert_eq!(world.bodies[ki].prev_position, Vec3Fix::ZERO);
        // sleeping: 重力なし、prev == position
        assert_eq!(world.bodies[sl].position, v3(9, 9, 9));
        assert_eq!(world.bodies[sl].prev_position, v3(9, 9, 9));
        assert_eq!(world.bodies[sl].velocity, Vec3Fix::ZERO);
    }

    // ---- step (substep 分割 / 重力の閉形式) ----------------------------

    #[test]
    fn step_free_fall_matches_closed_form_over_substeps() {
        // gravity -8、damping 1、substeps 4、dt 1 → substep_dt 1/4
        // 半陰的 Euler: v_k = -8 * k/4、x_k = x_{k-1} + v_k/4 → 1 step 後 v = -8、x = -(2+4+6+8)/4 = -5
        let mut world = quiet_world();
        world.config.gravity = v3(0, -8, 0);
        world.config.damping = Fix128::ONE;
        world.config.substeps = 4;
        world.config.iterations = 1;
        let i = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

        world.step(Fix128::ONE);

        assert_eq!(world.bodies[i].velocity, v3(0, -8, 0));
        assert_eq!(world.bodies[i].position, v3(0, -5, 0));
        // substeps 1 なら x = -8 (1 回で落ちる)
        let mut single = quiet_world();
        single.config.gravity = v3(0, -8, 0);
        single.config.damping = Fix128::ONE;
        single.config.substeps = 1;
        let j = single.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        single.step(Fix128::ONE);
        assert_eq!(single.bodies[j].position, v3(0, -8, 0));
    }

    // ---- setters / counters ------------------------------------------

    #[test]
    fn combined_material_uses_registered_pair_override_and_ignores_out_of_range() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let metal = world.material_table.register_metal();
        world
            .material_table
            .set_pair_override(0, metal, r(1, 8), r(7, 8));
        world.set_body_material(b, metal);
        let c = world.combined_material(a, b);
        assert_eq!((c.friction, c.restitution), (r(1, 8), r(7, 8)));
        // 順序を入れ替えても同じ (対称)
        let c2 = world.combined_material(b, a);
        assert_eq!((c2.friction, c2.restitution), (r(1, 8), r(7, 8)));
        // 範囲外 index は DEFAULT_MATERIAL 扱い = (default, metal) の組ではなく (default, default)
        let d = world.combined_material(a, 99);
        let dd = world.material_table.combine(0, 0);
        assert_eq!((d.friction, d.restitution), (dd.friction, dd.restitution));
        // set_body_material の範囲外は無視
        world.set_body_material(99, metal);
        assert_eq!(world.body_materials.len(), 2);
    }

    #[test]
    fn body_setters_ignore_out_of_range_and_apply_in_range() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        world.set_body_collision_radius(a, r(3, 2));
        assert_eq!(world.body_collision_radii[a], Some(r(3, 2)));
        world.set_body_collision_radius(5, Fix128::ONE);
        assert_eq!(world.body_collision_radii.len(), 1);
        world.clear_body_collision_radius(5);
        assert_eq!(world.body_collision_radii[a], Some(r(3, 2)));
        world.clear_body_collision_radius(a);
        assert_eq!(world.body_collision_radii[a], None);

        let f = CollisionFilter {
            layer: 4,
            mask: 8,
            group: 2,
        };
        world.set_body_filter(a, f);
        assert_eq!(world.body_filters[a], f);
        world.set_body_filter(5, CollisionFilter::DEFAULT);
        assert_eq!(world.body_filters[a], f);
    }

    #[test]
    fn active_body_count_excludes_sleeping() {
        let mut world = quiet_world();
        for i in 0..3 {
            world.add_body(RigidBody::new_dynamic(v3(i, 0, 0), Fix128::ONE));
        }
        assert_eq!(world.active_body_count(), 3);
        world.islands.sleep_data[1].state = crate::sleeping::SleepState::Sleeping;
        assert_eq!(world.active_body_count(), 2);
        world.islands.sleep_data[0].state = crate::sleeping::SleepState::Sleeping;
        world.islands.sleep_data[2].state = crate::sleeping::SleepState::Sleeping;
        assert_eq!(world.active_body_count(), 0);
    }

    // ---- hook / modifier clearing, RigidBody builders -------------------

    #[cfg(feature = "std")]
    #[test]
    fn clear_pre_solve_hooks_restores_normal_contact_response() {
        // veto hook が入っている間は contact が捨てられて A は動かない
        let mut world = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        world.add_pre_solve_hook(Box::new(|_a, _b, _c| false));
        world.add_pre_solve_hook(Box::new(|_a, _b, _c| false));
        assert_eq!(world.pre_solve_hooks.len(), 2);
        world.apply_contact_filters();
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);

        // clear 後は hook 0 個 → 通常通り λ = 1、inv_w = 1/2 → A += 1/2
        world.clear_pre_solve_hooks();
        assert!(world.pre_solve_hooks.is_empty());
        world.apply_contact_filters();
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(
            world.bodies[0].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[1].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn clear_contact_modifiers_restores_normal_contact_response() {
        struct Discard;
        impl ContactModifier for Discard {
            fn modify_contact(
                &self,
                _a: usize,
                _b: usize,
                _c: &mut Contact,
                _f: &mut Fix128,
                _r: &mut Fix128,
            ) -> bool {
                false
            }
        }
        let mut world = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        world.add_contact_modifier(Box::new(Discard));
        assert_eq!(world.contact_modifiers.len(), 1);
        world.apply_contact_filters();
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);

        world.clear_contact_modifiers();
        assert!(world.contact_modifiers.is_empty());
        world.apply_contact_filters();
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(
            world.bodies[0].position,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
    }

    #[test]
    fn set_angular_velocity_overwrites_and_drives_rotation_prediction() {
        let mut world = quiet_world();
        let i = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        world.bodies[i].angular_velocity = v3(5, 5, 5);
        world.bodies[i].set_angular_velocity(v3(0, 0, 2));
        assert_eq!(world.bodies[i].angular_velocity, v3(0, 0, 2));
        // 直接代入 (apply_impulse 系と違い static でも通る)
        let mut st = RigidBody::new_static(Vec3Fix::ZERO);
        st.set_angular_velocity(v3(1, 0, 0));
        assert_eq!(st.angular_velocity, v3(1, 0, 0));

        // 設定した ω が積分で使われる: z 軸 angle = 2 * 1/4 の回転
        world.integrate_positions(r(1, 4));
        let expected = QuatFix::from_axis_angle(v3(0, 0, 1), r(1, 2))
            .mul(QuatFix::IDENTITY)
            .normalize();
        assert_eq!(world.bodies[i].rotation, expected);
    }

    #[test]
    fn with_angular_damping_is_stored_and_applied_once_per_frame_by_step() {
        let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_angular_damping(r(1, 2));
        assert_eq!(body.angular_damping, r(1, 2));
        assert_eq!(body.linear_damping, Fix128::ONE);

        // substeps を変えても damping は frame 毎に 1 回だけ: ω_damped == ω_undamped * 1 * 1/2
        // (substep 毎なら substeps=4 で 1/16 になる)
        for substeps in [1usize, 4] {
            let mut undamped = quiet_world();
            undamped.config.substeps = substeps;
            let u = undamped.add_body(
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(v3(3, 0, 0)),
            );
            undamped.bodies[u].set_angular_velocity(v3(0, 0, 2));

            let mut damped = quiet_world();
            damped.config.substeps = substeps;
            let d = damped.add_body(
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)
                    .with_velocity(v3(3, 0, 0))
                    .with_angular_damping(r(1, 2)),
            );
            damped.bodies[d].set_angular_velocity(v3(0, 0, 2));

            undamped.step(r(1, 4));
            damped.step(r(1, 4));

            let expected = undamped.bodies[u].angular_velocity * Fix128::ONE * r(1, 2);
            assert_eq!(
                damped.bodies[d].angular_velocity, expected,
                "substeps {substeps}"
            );
            // 角 damping は線速度に影響しない
            assert_eq!(
                damped.bodies[d].velocity, undamped.bodies[u].velocity,
                "substeps {substeps}"
            );
            let ratio = damped.bodies[d].angular_velocity.z.to_f64()
                / undamped.bodies[u].angular_velocity.z.to_f64();
            assert!(
                (ratio - 0.5).abs() < 1e-9,
                "substeps {substeps}: ratio {ratio}"
            );
        }
    }

    #[test]
    fn with_sensor_stores_flag_and_sensor_body_skips_contact_response() {
        let sensor = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_sensor(true);
        assert!(sensor.is_sensor);
        assert!(!sensor.with_sensor(false).is_sensor);

        // contact_world と同じ配置だが A を builder で sensor にする
        let mut world = quiet_world();
        let a =
            world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_sensor(true));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let contact = Contact {
            depth: Fix128::ONE,
            normal: v3(1, 0, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        world
            .contact_constraints
            .push(ContactConstraint::new(a, b, contact));
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, v3(1, 0, 0));

        // 自動検出でも sensor は trigger event になり contact constraint を作らない
        let mut auto = quiet_world();
        auto.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_sensor(true),
            Fix128::ONE,
        );
        auto.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(r(3, 2), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            Fix128::ONE,
        );
        auto.detect_collisions();
        assert!(auto.contact_constraints.is_empty());
        assert_eq!(auto.trigger_events().len(), 1);
    }

    #[test]
    fn raycast_far_intersection_off_center_and_max_distance_cull_inside_aabb() {
        // origin (9,0,0) が球 (10,0,0) r=2 の内側 (中心から 1 ずれ): b=-1, c=-3, disc=4 → t_near=-1<0 → t_far = 1+2 = 3
        let world = sphere_world(&[v3(10, 0, 0)]);
        assert_eq!(
            world.raycast(v3(9, 0, 0), v3(1, 0, 0), Fix128::from_int(100)),
            Some((0, Fix128::from_int(3)))
        );
        // 球 (10,0,0) r=3 に y=2 の平行 ray: AABB [7,13] は max 7.5 の ray box と重なるが t = 10 - √5 ≈ 7.76 > 7.5 → None
        let mut big = quiet_world();
        big.add_body_with_radius(RigidBody::new_static(v3(10, 0, 0)), Fix128::from_int(3));
        assert_eq!(big.raycast(v3(0, 2, 0), v3(1, 0, 0), r(15, 2)), None);
        assert!(big
            .raycast(v3(0, 2, 0), v3(1, 0, 0), Fix128::from_int(8))
            .is_some());
    }

    // ---- mutation-kill tests (cargo-mutants missed list, 2026-09-15) ----

    /// `RigidBody::set_velocity` / `set_position` / `set_rotation` → `()`,
    /// `mass` → `Default` / `/` → `*`, `speed` → `Default`: every setter stores
    /// the exact value (position / rotation also reset the XPBD `prev_*`),
    /// mass 4 has `inv_mass` 1/4 and `mass()` must invert it back to 4 (the
    /// `*` mutant returns 1/4), a static body reports 0, and |(3, -4, 0)| = 5.
    #[test]
    fn rigid_body_setters_and_mass_speed_report_exact_values() {
        let mut body = RigidBody::new(v3(1, 1, 1), Fix128::from_int(4));
        assert_eq!(body.inv_mass, r(1, 4));
        assert_eq!(body.mass(), Fix128::from_int(4));
        assert_eq!(RigidBody::new_static(Vec3Fix::ZERO).mass(), Fix128::ZERO);

        body.set_velocity(v3(3, -4, 0));
        assert_eq!(body.velocity, v3(3, -4, 0));
        assert_eq!(body.speed(), Fix128::from_int(5));

        body.set_position(v3(7, -8, 9));
        assert_eq!(body.position, v3(7, -8, 9));
        assert_eq!(body.prev_position, v3(7, -8, 9));

        let q = QuatFix::from_axis_angle(v3(0, 1, 0), r(1, 3));
        assert_ne!(q, QuatFix::IDENTITY);
        body.set_rotation(q);
        assert_eq!(body.rotation, q);
        assert_eq!(body.prev_rotation, q);
    }

    /// `apply_impulse_at` line 297 / `add_torque` line 321 `*` → `/` on the x
    /// component: with `inv_inertia = (4, 4, 4)` and torque x = 2 the product
    /// is 8 (impulse) / 8 · dt = 4 (torque, dt = 1/2); the quotient would be
    /// 1/2 / 1/4. Lever arm r = (0, 1, 0), impulse (0, 0, 2) → r × F = (2, 0, 0).
    #[test]
    fn apply_impulse_at_and_add_torque_multiply_x_by_inverse_inertia() {
        let mut body = RigidBody::new(v3(5, 5, 5), Fix128::ONE);
        body.inv_inertia = v3(4, 4, 4);
        body.apply_impulse_at(v3(0, 0, 2), v3(5, 6, 5));
        assert_eq!(body.velocity, v3(0, 0, 2));
        assert_eq!(body.angular_velocity, v3(8, 0, 0));

        let mut spun = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        spun.inv_inertia = v3(4, 4, 4);
        spun.add_torque(v3(2, 0, 0), r(1, 2));
        assert_eq!(spun.angular_velocity, v3(4, 0, 0));
    }

    /// `body_count` → `0` / `1`, `get_body` → `Some(Default)`, `get_body_mut` →
    /// `None` / `Some(Default)`: counts 0 → 2 → 1 across add / remove, the
    /// accessors return the body at that slot (position (3, 4, 5), not the
    /// default origin) and `None` past the end; a write through `get_body_mut`
    /// must land in `bodies`.
    #[test]
    fn body_count_and_get_body_accessors_track_the_body_vector() {
        let mut world = quiet_world();
        assert_eq!(world.body_count(), 0);
        assert!(world.get_body(0).is_none());
        assert!(world.get_body_mut(0).is_none());

        world.add_body(RigidBody::new_dynamic(v3(3, 4, 5), Fix128::ONE));
        world.add_body(RigidBody::new_dynamic(v3(6, 7, 8), Fix128::ONE));
        assert_eq!(world.body_count(), 2);
        assert_eq!(world.get_body(0).map(|b| b.position), Some(v3(3, 4, 5)));
        assert_eq!(world.get_body(1).map(|b| b.position), Some(v3(6, 7, 8)));
        assert!(world.get_body(2).is_none());

        world
            .get_body_mut(1)
            .expect("body 1 exists")
            .set_velocity(v3(-1, -2, -3));
        assert_eq!(world.bodies[1].velocity, v3(-1, -2, -3));
        assert!(world.get_body_mut(2).is_none());

        assert!(world.remove_body(0).is_some());
        assert_eq!(world.body_count(), 1);
    }

    /// `set_body_material` line 1109, `set_body_collision_radius` line 1213,
    /// `clear_body_collision_radius` line 1220, `set_body_filter` line 1227,
    /// `combined_material` lines 1413 / 1418 `<` → `<=`: an index equal to the
    /// body count is out of range and must be ignored (the mutants index one
    /// past the end and panic). `body_filter` → `Default` is killed by reading
    /// back a non-default filter.
    #[test]
    fn index_equal_to_body_count_is_out_of_range_for_every_setter() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let len = world.body_count();
        let metal = world.material_table.register_metal();

        world.set_body_material(len, metal);
        world.set_body_collision_radius(len, Fix128::ONE);
        world.clear_body_collision_radius(len);
        world.set_body_filter(len, CollisionFilter::NONE);
        assert_eq!(
            world.body_materials,
            vec![crate::material::DEFAULT_MATERIAL]
        );
        assert_eq!(world.body_collision_radii, vec![None]);
        assert_eq!(world.body_filters, vec![CollisionFilter::DEFAULT]);

        world.set_body_material(a, metal);
        let default_metal = world
            .material_table
            .combine(crate::material::DEFAULT_MATERIAL, metal);
        let default_default = world.material_table.combine(0, 0);
        assert_ne!(default_metal.friction, default_default.friction);
        for (x, y, want) in [
            (len, a, default_metal),
            (a, len, default_metal),
            (len, len, default_default),
        ] {
            let c = world.combined_material(x, y);
            assert_eq!(
                (c.friction, c.restitution),
                (want.friction, want.restitution),
                "({x}, {y}) must treat the out-of-range side as the default material"
            );
        }
        let custom = CollisionFilter {
            layer: 1 << 3,
            mask: 1 << 9,
            group: 5,
        };
        world.set_body_filter(a, custom);
        assert_eq!(world.body_filter(a), custom);
        assert_eq!(world.body_filter(len), CollisionFilter::DEFAULT);
    }

    /// `begin_frame` / `end_frame` → `()` and `clear_contacts` → `()`: a cached
    /// manifold ages by one per `begin_frame` (4 calls → `stale_frames == 4`)
    /// and `end_frame` prunes it once it exceeds `max_stale_frames` (3);
    /// `clear_contacts` empties the constraint list.
    #[test]
    fn contact_cache_frame_lifecycle_ages_and_prunes_manifolds() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        push_contact(&mut world, a, b, r(1, 2), r(1, 2));
        let c = world.contact_constraints.pop().expect("pushed");
        world.add_contact(c);
        assert_eq!(world.contact_constraints.len(), 1);
        assert_eq!(world.contact_cache.manifold_count(), 1);
        assert_eq!(world.contact_cache.manifolds[0].stale_frames, 0);

        for _ in 0..4 {
            world.begin_frame();
        }
        assert_eq!(world.contact_cache.manifolds[0].stale_frames, 4);
        world.end_frame();
        assert_eq!(world.contact_cache.manifold_count(), 0);

        world.clear_contacts();
        assert!(world.contact_constraints.is_empty());
    }

    /// `remove_joint` → `None` and line 1179 `>=` → `<`: two joints get
    /// indices 0 / 1, index 2 is out of range (`None`), removing 0 returns
    /// `Some` and leaves one joint, after which index 1 is out of range.
    #[test]
    fn add_and_remove_joint_roundtrip() {
        use crate::joint::{BallJoint, Joint};
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let j0 = world.add_joint(Joint::Ball(BallJoint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
        let j1 = world.add_joint(Joint::Ball(BallJoint::new(
            b,
            a,
            v3(1, 0, 0),
            Vec3Fix::ZERO,
        )));
        assert_eq!((j0, j1), (0, 1));
        assert_eq!(world.joint_count(), 2);
        assert!(world.remove_joint(2).is_none());
        assert_eq!(world.joint_count(), 2);
        let removed = world.remove_joint(0).expect("joint 0 exists");
        assert_eq!(removed.bodies(), (a, b));
        assert_eq!(world.joint_count(), 1);
        assert_eq!(world.joints[0].bodies(), (b, a));
        assert!(world.remove_joint(1).is_none());
    }

    /// `add_force_field` → `0` / `1`, `remove_force_field` → `None` and line
    /// 1203 `>=` → `<`, `step` line 1697 `!` deleted: fields get indices 0 / 1,
    /// index 2 is `None`, removing 1 leaves one field, and `step` applies the
    /// remaining directional field once per frame: F = (2, 0, 0), m = 1,
    /// dt = 1/4 → v = 1/2, then 8 substeps of 1/32 move x by 8 · (1/2)(1/32) =
    /// 1/8 (quiet world: no gravity, no damping).
    #[test]
    fn force_field_add_remove_and_step_applies_directional_force() {
        use crate::force::{ForceField, ForceFieldInstance};
        let mut world = quiet_world();
        let body = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let wind = ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(2),
        });
        let drag = ForceFieldInstance::new(ForceField::Drag {
            coefficient: Fix128::ONE,
        });
        assert_eq!(world.add_force_field(wind), 0);
        assert_eq!(world.add_force_field(drag), 1);
        assert_eq!(world.force_fields.len(), 2);
        assert!(world.remove_force_field(2).is_none());
        assert!(matches!(
            world.remove_force_field(1).map(|f| f.field),
            Some(ForceField::Drag { .. })
        ));
        assert!(world.remove_force_field(1).is_none());
        assert_eq!(world.force_fields.len(), 1);

        world.step(r(1, 4));
        assert_eq!(
            world.bodies[body].velocity,
            Vec3Fix::new(r(1, 2), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(
            world.bodies[body].position,
            Vec3Fix::new(r(1, 8), Fix128::ZERO, Fix128::ZERO)
        );
    }

    /// The live-proxy count the broad-phase reads every substep equals the
    /// number of proxies in the parked-body tree through parking, a contact
    /// unpark, a manual wake, a removal and turning the skip off.
    #[test]
    fn park_proxy_live_count_tracks_the_tree() {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            ..PhysicsConfig::default()
        });
        let r = Fix128::from_ratio(1, 2);
        for i in 0..20 {
            w.add_body_with_radius(
                RigidBody::new_dynamic(Vec3Fix::from_int(i * 3, 0, 0), Fix128::ONE),
                r,
            );
            w.islands.sleep_data[i as usize].state = SleepState::Sleeping;
        }
        let check = |w: &PhysicsWorld, ctx: &str| {
            assert_eq!(w.park.proxy_live, w.park.tree.proxy_count(), "{ctx}");
            assert_eq!(
                w.park.proxy_live,
                w.park.proxies.iter().flatten().count(),
                "{ctx}"
            );
        };
        w.step(Fix128::from_ratio(1, 60));
        check(&w, "parked");
        assert_eq!(w.park.proxy_live, 20);
        // Awake body 0 moved into body 1: the contact unparks it.
        w.wake_body(0);
        w.bodies[0].set_position(Vec3Fix::new(
            Fix128::from_ratio(22, 10),
            Fix128::ZERO,
            Fix128::ZERO,
        ));
        w.step(Fix128::from_ratio(1, 60));
        check(&w, "contact");
        assert!(w.stage_work().unparked > 0);
        w.wake_body(7);
        w.step(Fix128::from_ratio(1, 60));
        check(&w, "manual wake");
        w.remove_body(3);
        w.step(Fix128::from_ratio(1, 60));
        check(&w, "remove");
        w.set_sleep_skip(false);
        check(&w, "off");
        assert_eq!(w.park.proxy_live, 0);
    }

    /// `is_sleeping` → `false` / `true` and `wake_body` → `()`: after one step
    /// the static body is asleep (static bodies always are), the moving
    /// dynamic body and an out-of-range index are not, and `wake_body` wakes
    /// the static one again.
    #[test]
    fn is_sleeping_and_wake_body_reflect_island_state() {
        let mut world = quiet_world();
        let floor = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mover = world.add_body(RigidBody::new_dynamic(v3(0, 5, 0), Fix128::ONE));
        world.bodies[mover].velocity = v3(5, 0, 0);
        world.step(r(1, 4));
        assert!(world.is_sleeping(floor));
        assert!(!world.is_sleeping(mover));
        assert!(!world.is_sleeping(99));
        world.wake_body(floor);
        assert!(!world.is_sleeping(floor));
        assert!(!world.is_sleeping(mover));
    }

    /// `set_sleep_config` → `()`: with `frames_to_sleep = 2` an idle body is
    /// asleep after two frames; under the default 60 frames it is not.
    #[test]
    fn set_sleep_config_changes_frames_to_sleep() {
        let cfg = SleepConfig {
            frames_to_sleep: 2,
            ..SleepConfig::default()
        };
        let mut quick = quiet_world();
        let idle = quick.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        quick.set_sleep_config(cfg);
        assert_eq!(quick.islands.config, cfg);
        quick.step(r(1, 4));
        assert!(!quick.is_sleeping(idle));
        quick.step(r(1, 4));
        assert!(quick.is_sleeping(idle));

        let mut slow = quiet_world();
        let idle2 = slow.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        slow.step(r(1, 4));
        slow.step(r(1, 4));
        assert!(!slow.is_sleeping(idle2));
    }

    /// `drain_contact_events` / `drain_trigger_events` → `vec![]`: draining
    /// returns the events of the last detection and leaves the collector
    /// empty.
    #[test]
    fn drain_events_return_the_events_and_empty_the_collector() {
        let mut world = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        world.detect_collisions();
        assert_eq!(world.contact_events().len(), 1);
        let drained = world.drain_contact_events();
        assert_eq!(drained.len(), 1);
        assert_eq!((drained[0].body_a, drained[0].body_b), (0, 1));
        assert_eq!(drained[0].event_type, crate::event::ContactEventType::Begin);
        assert!(world.contact_events().is_empty());
        assert!(world.drain_contact_events().is_empty());

        let mut sensor = two_spheres(3, 2, Fix128::ONE, Fix128::ONE);
        sensor.bodies[1].is_sensor = true;
        sensor.detect_collisions();
        let triggers = sensor.drain_trigger_events();
        assert_eq!(triggers.len(), 1);
        assert!(triggers[0].entered);
        assert!(sensor.trigger_events().is_empty());
        assert!(sensor.drain_trigger_events().is_empty());
    }

    /// `step` line 1691 `<` → `==` / `>`: the joint island rebuild at the start
    /// of every step (`reset_unions` then `union` for every in-range joint)
    /// must leave both bodies of a joint in one island.
    #[test]
    fn step_rebuilds_joint_islands_from_scratch() {
        use crate::joint::{BallJoint, Joint};
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let c = world.add_body(RigidBody::new_dynamic(v3(9, 0, 0), Fix128::ONE));
        world.add_joint(Joint::Ball(BallJoint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
        // `add_joint` already unions; wipe that so only `step`'s rebuild counts
        world.islands.reset_unions();
        assert_ne!(world.islands.find(a), world.islands.find(b));
        world.step(r(1, 4));
        assert_eq!(world.islands.find(a), world.islands.find(b));
        assert_ne!(world.islands.find(a), world.islands.find(c));
    }

    /// `integrate_positions` line 1905 `-` → `+`: a kinematic body's velocity
    /// is (target - position) / dt = ((5, 2, 3) - (1, 2, 3)) · 4 = (16, 0, 0)
    /// right after integration (the sum would give (24, 16, 24)); the body
    /// lands on the target and `prev_position` keeps the old position.
    #[test]
    fn integrate_positions_kinematic_velocity_is_target_minus_position_over_dt() {
        let mut world = quiet_world();
        let k = world.add_body(RigidBody::new_kinematic(v3(1, 2, 3)));
        world.bodies[k].set_kinematic_target(v3(5, 2, 3), QuatFix::IDENTITY);
        world.integrate_positions(r(1, 4));
        assert_eq!(world.bodies[k].velocity, v3(16, 0, 0));
        assert_eq!(world.bodies[k].position, v3(5, 2, 3));
        assert_eq!(world.bodies[k].prev_position, v3(1, 2, 3));
    }

    /// Same closed form as the `integrate_positions` test immediately above,
    /// but through the `Tgs`-path helper instead: velocity is
    /// (target - position) / dt = ((5, 2, 3) - (1, 2, 3)) · 4 = (16, 0, 0),
    /// the body lands exactly on the target, and `prev_position` keeps the
    /// pre-advance position (mirrors `integrate_positions`' `Kinematic`
    /// branch bit-for-bit; this is the oracle for
    /// `advance_kinematic_targets_for_tgs` itself, not for `step_tgs`).
    #[cfg(feature = "std")]
    #[test]
    fn advance_kinematic_targets_for_tgs_matches_integrate_positions_closed_form() {
        let mut world = quiet_world();
        let k = world.add_body(RigidBody::new_kinematic(v3(1, 2, 3)));
        world.bodies[k].set_kinematic_target(v3(5, 2, 3), QuatFix::IDENTITY);
        world.advance_kinematic_targets_for_tgs(r(1, 4));
        assert_eq!(world.bodies[k].velocity, v3(16, 0, 0));
        assert_eq!(world.bodies[k].position, v3(5, 2, 3));
        assert_eq!(world.bodies[k].prev_position, v3(1, 2, 3));
    }

    /// A dynamic body with no target is untouched by the `Tgs`-path helper
    /// (only `BodyType::Kinematic` bodies are visited at all).
    #[cfg(feature = "std")]
    #[test]
    fn advance_kinematic_targets_for_tgs_leaves_dynamic_bodies_untouched() {
        let mut world = quiet_world();
        let d = world.add_body(RigidBody::new_dynamic(v3(1, 2, 3), Fix128::ONE));
        world.bodies[d].velocity = v3(7, 8, 9);
        world.advance_kinematic_targets_for_tgs(r(1, 4));
        assert_eq!(world.bodies[d].position, v3(1, 2, 3));
        assert_eq!(world.bodies[d].velocity, v3(7, 8, 9));
    }

    /// A kinematic body with no target set (`kinematic_target == None`) is
    /// left at rest — only `prev_position`/`prev_rotation` are refreshed,
    /// matching `integrate_positions`' `Kinematic` branch (the `if let
    /// Some(...)` guards the actual advance in both).
    #[cfg(feature = "std")]
    #[test]
    fn advance_kinematic_targets_for_tgs_without_a_target_stays_at_rest() {
        let mut world = quiet_world();
        let k = world.add_body(RigidBody::new_kinematic(v3(1, 2, 3)));
        world.advance_kinematic_targets_for_tgs(r(1, 4));
        assert_eq!(world.bodies[k].position, v3(1, 2, 3));
        assert_eq!(world.bodies[k].velocity, Vec3Fix::ZERO);
        assert_eq!(world.bodies[k].prev_position, v3(1, 2, 3));
    }

    /// `update_velocities` lines 2055 / 2073 `-` → `+` (relative velocity),
    /// 2089 `/` → `*` (`inv_w`), 2091 / 2093 `*` → `/` and 2093 `+` → `-`
    /// (friction split): both bodies dynamic (inv 1 / inv 3 → inv_w = 1/4),
    /// contact normal +y (B → A), vA = (3, -4, 0), vB = (1, -2, 0), e = 1/2,
    /// μ = 1/2.
    /// Restitution: vn = (2, -2, 0)·n = -2 → Δvn = 3 → A += 3/4, B -= 9/4 →
    /// A (3, -13/4, 0), B (1, -17/4, 0). Friction: rel = (2, 1, 0), vn2 = 1,
    /// tangent (2, 0, 0) speed 2, limit μ·|vn2| = 1/2 → impulse (1/2, 0, 0);
    /// A -= 1/2 · 1/4, B += 1/2 · 3/4 → A (23/8, -13/4, 0), B (11/8, -17/4, 0).
    /// (`inv_w = w_sum` would give A.x = 1 / B.x = 7, `+` → `-` on B gives 5/8.)
    #[test]
    fn update_velocities_friction_and_restitution_split_between_two_dynamic_bodies() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 1, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        world.bodies[b].inv_mass = Fix128::from_int(3);
        let contact = Contact {
            depth: r(1, 100),
            normal: v3(0, 1, 0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        let mut c = ContactConstraint::new(a, b, contact);
        c.friction = r(1, 2);
        c.restitution = r(1, 2);
        // 位置解の法線 λ = 1/4 → Coulomb 上限 μ λ / dt = 1/2 * 1/4 * 4 = 1/2
        c.cached_lambda = r(1, 4);
        world.contact_constraints.push(c);
        freeze_positions(&mut world);
        give_velocity(&mut world, a, v3(3, -4, 0));
        give_velocity(&mut world, b, v3(1, -2, 0));

        world.update_velocities(r(1, 4));

        assert_near_vec(
            world.bodies[a].velocity,
            Vec3Fix::new(r(23, 8), r(-13, 4), Fix128::ZERO),
            "A",
        );
        assert_near_vec(
            world.bodies[b].velocity,
            Vec3Fix::new(r(11, 8), r(-17, 4), Fix128::ZERO),
            "B",
        );
    }

    /// Half of `W_SUM_EPSILON` (an inverse mass strictly below the skip
    /// threshold).
    fn half_epsilon() -> Fix128 {
        Fix128 {
            hi: 0,
            lo: W_SUM_EPSILON.lo >> 1,
        }
    }

    /// `update_velocities` line 2050 `<` → `<=` / `==`: a contact whose
    /// `w_sum` is exactly `W_SUM_EPSILON` is solved (A: inv_mass = ε against a
    /// static B, vn = -4, e = 1/2 → Δvn = 6, inv_w = 1/ε, ε · 1/ε = 1 → A.y =
    /// -4 + 6 = 2), one with `w_sum = ε/2` is skipped (A keeps (0, -4, 0)).
    #[test]
    fn update_velocities_w_sum_epsilon_boundary() {
        for (inv_mass, expected_y) in [(W_SUM_EPSILON, 2), (half_epsilon(), -4)] {
            let mut world = quiet_world();
            let a = world.add_body(RigidBody::new_dynamic(v3(0, 1, 0), Fix128::ONE));
            let b = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            world.bodies[a].inv_mass = inv_mass;
            let contact = Contact {
                depth: r(1, 100),
                normal: v3(0, 1, 0),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            };
            let mut c = ContactConstraint::new(a, b, contact);
            c.friction = Fix128::ZERO;
            c.restitution = r(1, 2);
            world.contact_constraints.push(c);
            freeze_positions(&mut world);
            give_velocity(&mut world, a, v3(0, -4, 0));

            world.update_velocities(r(1, 4));

            assert_eq!(
                world.bodies[a].velocity,
                v3(0, expected_y, 0),
                "inv_mass {inv_mass:?}"
            );
        }
    }

    /// `solve_distance_constraints` line 2394 `<` → `<=` / `==`: with A at
    /// (0, 4, 0), inv_mass = ε, static B at the origin and target 1 the error
    /// is 3 and `w_sum == ε` is solved: dlambda = 3/ε, correction · ε moves A
    /// by the full error to (0, 1, 0). With inv_mass = ε/2 the constraint is
    /// skipped and A stays at (0, 4, 0).
    #[test]
    fn solve_distance_constraints_w_sum_epsilon_boundary() {
        for (inv_mass, expected) in [(W_SUM_EPSILON, v3(0, 1, 0)), (half_epsilon(), v3(0, 4, 0))] {
            let mut world = quiet_world();
            let a = world.add_body(RigidBody::new_dynamic(v3(0, 4, 0), Fix128::ONE));
            let b = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            world.bodies[a].inv_mass = inv_mass;
            world.add_distance_constraint(dc(a, b));
            world.solve_distance_constraints(r(1, 4));
            assert_near_vec(world.bodies[a].position, expected, "A position");
            assert_eq!(world.bodies[b].position, Vec3Fix::ZERO);
        }
    }

    /// `solve_contact_constraints` line 2497 `<` → `<=` / `==`: depth 1 along
    /// +x against a static B; `w_sum == ε` pushes A by the full depth
    /// (ε · 1/ε = 1) to (1, 0, 0), `w_sum = ε/2` is skipped.
    #[test]
    fn solve_contact_constraints_w_sum_epsilon_boundary() {
        for (inv_mass, expected) in [
            (W_SUM_EPSILON, v3(1, 0, 0)),
            (half_epsilon(), Vec3Fix::ZERO),
        ] {
            let mut world = contact_world(inv_mass, Fix128::ZERO, Fix128::ONE);
            world.solve_contact_constraints(r(1, 4));
            assert_eq!(world.bodies[0].position, expected, "inv_mass {inv_mass:?}");
            assert_eq!(world.bodies[1].position, v3(1, 0, 0));
        }
    }

    /// `solve_distance_constraints` line 2377 `+` → `-` (anchor A) and line
    /// 2407 `*` → `/` (`delta_a`): A at (0, 2, 0) with local anchor (0, 1, 0)
    /// and inv_mass 2, static B at the origin, target 1 → world anchor
    /// (0, 3, 0), error 2, w_sum 2, dlambda 1, correction (0, -1, 0) · 2 →
    /// A lands on the origin. The `-` anchor gives error 0 (A stays), the `/`
    /// gives half the move (A at (0, 3/2, 0)).
    #[test]
    fn solve_distance_constraints_anchor_a_offset_and_inverse_mass_scaling() {
        let mut world = quiet_world();
        let a = world.add_body(RigidBody::new_dynamic(v3(0, 2, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        world.bodies[a].inv_mass = Fix128::from_int(2);
        let mut c = dc(a, b);
        c.local_anchor_a = v3(0, 1, 0);
        world.add_distance_constraint(c);

        world.solve_distance_constraints(r(1, 4));

        assert_near_vec(world.bodies[a].position, Vec3Fix::ZERO, "A");
        assert_eq!(world.bodies[b].position, Vec3Fix::ZERO);
        assert_eq!(world.distance_constraints[0].cached_lambda, Fix128::ONE);
    }

    /// `detect_collisions` line 2933 / 2952 `<` → `>` (early return with 3
    /// bodies / 3 primitives) and line 3009 `*` → `+` (squared-distance early
    /// out): three collidable bodies, A at the origin and B at (2, 1, 1)
    /// (dist² = 6) with radii 3/2 each (combined 3: 6 < 3² = 9 overlaps, but
    /// 6 ≥ 3 + 3 = 6 would be culled), C far away. Exactly one contact, depth
    /// 3 - √6.
    #[test]
    fn detect_collisions_three_bodies_squared_distance_cull_uses_squared_radius() {
        let mut world = quiet_world();
        let radius = r(3, 2);
        world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), radius);
        world.add_body_with_radius(RigidBody::new_dynamic(v3(2, 1, 1), Fix128::ONE), radius);
        world.add_body_with_radius(RigidBody::new_dynamic(v3(50, 50, 50), Fix128::ONE), radius);
        world.detect_collisions();
        assert_eq!(world.contact_constraints.len(), 1);
        let c = world.contact_constraints[0];
        assert_eq!((c.body_a, c.body_b), (0, 1));
        let expected_depth = Fix128::from_int(3) - Fix128::from_int(6).sqrt();
        assert!(
            near(c.contact.depth, expected_depth),
            "depth {:?} vs {expected_depth:?}",
            c.contact.depth
        );
        assert_eq!(world.contact_events().len(), 1);
    }

    /// `remove_sdf_collider` → `None` and line 2863 `<` → `<=` / `==` / `>`,
    /// `set_sdf_collision_radius` → `()`: index 1 of a single collider is out
    /// of range (`None`, no panic), index 0 is removed, and the radius setter
    /// stores 3/4.
    #[test]
    fn remove_sdf_collider_boundary_and_sdf_radius_setter() {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        let mut world = quiet_world();
        let ground = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
        assert_eq!(
            world.add_sdf_collider(SdfCollider::new_static(
                Box::new(ground),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            )),
            0
        );
        assert!(world.remove_sdf_collider(1).is_none());
        assert_eq!(world.sdf_colliders.len(), 1);
        assert!(world.remove_sdf_collider(0).is_some());
        assert!(world.sdf_colliders.is_empty());
        assert!(world.remove_sdf_collider(0).is_none());

        world.set_sdf_collision_radius(r(3, 4));
        assert_eq!(world.sdf_collision_radius, r(3, 4));
    }

    /// `deserialize_state` line 3173 `<` → `<=`: a 4-byte snapshot (header
    /// only, count 0) of an empty world is valid.
    #[test]
    fn deserialize_state_accepts_header_only_snapshot_of_empty_world() {
        let src = quiet_world();
        let bytes = src.serialize_state();
        // v3: body 0 体でも header 12 + flag 1 + fingerprint 8
        assert_eq!(bytes.len(), 21);
        let mut dst = quiet_world();
        assert!(dst.deserialize_state(&bytes));
        assert_eq!(dst.body_count(), 0);
    }

    /// `deserialize_state` lines 3237 / 3241 / 3245 `<` → `==` / `>`: when the
    /// parallel arrays are shorter than `bodies` (a body pushed straight into
    /// `bodies`), the resync loops must extend them to the body count with
    /// default entries.
    #[test]
    fn deserialize_state_extends_short_parallel_arrays_to_body_count() {
        let mut world = quiet_world();
        world.add_body(RigidBody::new_dynamic(v3(1, 2, 3), Fix128::ONE));
        world.set_body_collision_radius(0, r(1, 2));
        world
            .bodies
            .push(RigidBody::new_dynamic(v3(4, 5, 6), Fix128::ONE));
        world
            .bodies
            .push(RigidBody::new_dynamic(v3(7, 8, 9), Fix128::ONE));
        assert_eq!(world.body_collision_radii.len(), 1);
        let bytes = world.serialize_state();
        assert!(world.deserialize_state(&bytes));
        assert_eq!(world.body_collision_radii, vec![Some(r(1, 2)), None, None]);
        assert_eq!(world.body_filters, vec![CollisionFilter::DEFAULT; 3]);
        assert_eq!(
            world.body_materials,
            vec![crate::material::DEFAULT_MATERIAL; 3]
        );
        assert_eq!(world.islands.sleep_data.len(), 3);
        assert_eq!(world.bodies[2].position, v3(7, 8, 9));
    }

    // ---- observe_body / observe_bodies (World Auditor WM-10 Physics 版) ----

    #[test]
    fn observe_body_returns_none_out_of_range_and_fields_for_in_range() {
        let mut world = quiet_world();
        let mut a = RigidBody::new_dynamic(v3(1, 2, 3), Fix128::ONE);
        a.velocity = v3(4, 5, 6);
        a.angular_velocity = v3(7, 8, 9);
        world.add_body(a);

        assert!(world.observe_body(1).is_none(), "範囲外 index は None");

        let obs = world.observe_body(0).expect("範囲内 index は Some");
        assert_eq!(obs.body_index, 0);
        assert_eq!(obs.position, v3(1, 2, 3));
        assert_eq!(obs.velocity, v3(4, 5, 6));
        assert_eq!(obs.angular_velocity, v3(7, 8, 9));
        assert!(!obs.sleeping);
        assert!(!obs.in_contact, "接触していない body は in_contact = false");
    }

    #[test]
    fn observe_bodies_returns_one_entry_per_body_in_index_order() {
        let mut world = quiet_world();
        world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        world.add_body(RigidBody::new_dynamic(v3(2, 0, 0), Fix128::ONE));
        world.add_body(RigidBody::new_dynamic(v3(3, 0, 0), Fix128::ONE));

        let obs = world.observe_bodies();
        assert_eq!(obs.len(), 3);
        for (i, o) in obs.iter().enumerate() {
            assert_eq!(o.body_index, i);
            assert_eq!(o.position, world.bodies[i].position);
        }
    }

    #[test]
    fn observe_body_reports_in_contact_from_this_frame_events() {
        // 重なった 2 球を 1 step 進めると contact event (Begin) が出る
        // (`tests/wm07_rollback_event_parity.rs` と同じ idiom — 最初から重なっている配置)
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            ..PhysicsConfig::default()
        });
        let radius = Fix128::from_int(2);
        let a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        world.add_body_with_radius(a, radius);
        let b = RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE);
        world.add_body_with_radius(b, radius);
        let c = RigidBody::new_dynamic(v3(100, 100, 100), Fix128::ONE);
        world.add_body_with_radius(c, Fix128::from_ratio(1, 10));

        world.step(Fix128::from_ratio(1, 60));

        assert!(world.observe_body(0).unwrap().in_contact);
        assert!(world.observe_body(1).unwrap().in_contact);
        assert!(
            !world.observe_body(2).unwrap().in_contact,
            "遠くの body は接触していないはず"
        );
    }

    /// `batches_are_body_disjoint` line 1646 `&&` → `||` / `a == b` → `a != b`
    /// (conflict through the *second* body of both constraints) and `||` →
    /// `&&` (a self-constraint counts its body once): constraints (1, 0) and
    /// (2, 0) share body 0 only as `body_b`, so one batch holding both is
    /// invalid; a batch holding only the self-constraint (1, 1) is valid.
    #[test]
    fn batches_are_body_disjoint_checks_second_body_and_self_constraints() {
        let mut world = quiet_world();
        for i in 0..3i64 {
            world.add_body(RigidBody::new_dynamic(v3(i, 0, 0), Fix128::ONE));
        }
        world.add_distance_constraint(dc(1, 0));
        world.add_distance_constraint(dc(2, 0));
        world.rebuild_batches();
        assert_eq!(world.num_batches(), 2);
        assert!(world.batches_are_body_disjoint());

        let mut shared_b = ConstraintBatch::default();
        shared_b.distance_indices.extend([0usize, 1]);
        world.constraint_batches = vec![shared_b];
        assert!(!world.batches_are_body_disjoint());

        world.distance_constraints.push(dc(1, 1));
        let mut self_only = ConstraintBatch::default();
        self_only.distance_indices.push(2);
        world.constraint_batches = vec![self_only];
        assert!(world.batches_are_body_disjoint());

        let mut self_and_other = ConstraintBatch::default();
        self_and_other.distance_indices.extend([2usize, 0]);
        world.constraint_batches = vec![self_and_other];
        assert!(!world.batches_are_body_disjoint());
    }

    /// `raycast` line 1397 `<` → `<=` / `==`: (1) two spheres hit at exactly
    /// the same t (mirror images across the ray) — the first BVH candidate
    /// wins; the candidates come in Morton order, so the sphere with the
    /// smaller y (body 1) is reported. (2) A ray cast in -x meets the far
    /// sphere first in Morton order (smaller x) and the nearer sphere second;
    /// the nearer one (body 1, t = 10 - 2 = 8) must replace it.
    #[test]
    fn raycast_ties_keep_first_candidate_and_nearer_later_candidate_replaces() {
        let mut tie = quiet_world();
        tie.add_body_with_radius(
            RigidBody::new_static(Vec3Fix::new(Fix128::from_int(5), r(1, 2), Fix128::ZERO)),
            Fix128::ONE,
        );
        tie.add_body_with_radius(
            RigidBody::new_static(Vec3Fix::new(Fix128::from_int(5), r(-1, 2), Fix128::ZERO)),
            Fix128::ONE,
        );
        let hit = tie
            .raycast(Vec3Fix::ZERO, v3(1, 0, 0), Fix128::from_int(20))
            .expect("both spheres straddle the ray");
        assert_eq!(hit.0, 1);
        // t = 5 - sqrt(3/4) for both
        let expected_t = Fix128::from_int(5) - r(3, 4).sqrt();
        assert!(near(hit.1, expected_t), "t {:?} vs {expected_t:?}", hit.1);

        let far_first = sphere_world(&[v3(5, 0, 0), v3(10, 0, 0)]);
        assert_eq!(
            far_first.raycast(v3(20, 0, 0), v3(-1, 0, 0), Fix128::from_int(100)),
            Some((1, Fix128::from_int(8)))
        );
    }

    /// `Debug for PhysicsWorld` → `Ok(())`: the output names the type and the
    /// body count.
    #[test]
    fn physics_world_debug_lists_type_and_counts() {
        let mut world = quiet_world();
        world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        world.add_body(RigidBody::new_dynamic(v3(1, 0, 0), Fix128::ONE));
        let text = format!("{world:?}");
        assert!(text.starts_with("PhysicsWorld"), "{text}");
        assert!(text.contains("bodies: 2"), "{text}");
        assert!(text.contains("joints: 0"), "{text}");
    }

    // ── closed-form unit tests of the shape / motor / contact-force /
    //    participant paths (they are otherwise exercised only from tests/) ──

    fn half_unit_box() -> crate::shape::Shape {
        crate::shape::Shape::Box {
            half_extents: Vec3Fix::from_int(1, 1, 1),
        }
    }

    /// oracle: a box of half-extent 1 at the origin and a sphere of radius 1
    /// meet when the sphere centre is within `1 + 1 = 2` of the box face along
    /// `x` (1.5: overlap, 2.5: clear), whichever side carries the shape; two
    /// shaped boxes meet when the centres are under `1 + 1 = 2` apart; two
    /// plain spheres of radius 1 meet when the centres are under 2 apart.
    /// An index past the bodies, or a body with no collider, never overlaps,
    /// and `set_body_shape` refuses an index past the bodies.
    #[test]
    fn colliders_overlap_follows_the_closed_form_gaps() {
        let mut world = quiet_world();
        let shaped = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(world.set_body_shape(shaped, &half_unit_box()));
        assert!(!world.set_body_shape(99, &half_unit_box()));
        let near = world.add_body_with_radius(
            RigidBody::new(
                Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            Fix128::ONE,
        );
        let far = world.add_body_with_radius(
            RigidBody::new(
                Vec3Fix::new(Fix128::from_ratio(5, 2), Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            Fix128::ONE,
        );
        let bare = world.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        // shape against sphere, both argument orders
        assert!(world.colliders_overlap(shaped, near));
        assert!(world.colliders_overlap(near, shaped));
        assert!(!world.colliders_overlap(shaped, far));
        assert!(!world.colliders_overlap(far, shaped));
        // two plain spheres 1 apart (< 2) overlap
        assert!(world.colliders_overlap(near, far));
        // a body without a collider, or an index past the bodies
        assert!(!world.colliders_overlap(shaped, bare));
        assert!(!world.colliders_overlap(near, bare));
        assert!(!world.colliders_overlap(near, 99));
        // shape against shape: the far body becomes a box 2.5 away (clear),
        // then the near one a box 1.5 away (overlapping)
        assert!(world.set_body_shape(far, &half_unit_box()));
        assert!(!world.colliders_overlap(shaped, far));
        assert!(world.set_body_shape(near, &half_unit_box()));
        assert!(world.colliders_overlap(shaped, near));
    }

    /// oracle: a sphere of radius 1/2 dropped onto the top face (`y = 1`) of a
    /// static box of half-extent 1 comes to rest with its centre at
    /// `1 + 1/2 = 3/2`, the contact of a shaped body with a plain sphere
    /// (the GJK path of the narrow phase), for either body order.
    #[test]
    fn sphere_rests_on_a_shaped_box_at_face_plus_radius() {
        for box_first in [true, false] {
            let mut world = PhysicsWorld::new(SolverConfig::default());
            let ball = || {
                RigidBody::new(
                    Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(8, 5), Fix128::ZERO),
                    Fix128::ONE,
                )
            };
            let (floor, b) = if box_first {
                let f = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
                let b = world.add_body_with_radius(ball(), Fix128::from_ratio(1, 2));
                (f, b)
            } else {
                let b = world.add_body_with_radius(ball(), Fix128::from_ratio(1, 2));
                let f = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
                (f, b)
            };
            assert!(world.set_body_shape(floor, &half_unit_box()));
            for _ in 0..120 {
                world.step(Fix128::from_ratio(1, 60));
            }
            let y = world.bodies[b].position.y.to_f64();
            assert!((y - 1.5).abs() < 0.02, "box_first {box_first}: y {y}");
            assert!(world.bodies[b].position.x.abs() < Fix128::from_ratio(1, 1000));
        }
    }

    /// oracle: the motor accessors report `false` / `None` for an index that
    /// was never returned, set the mode they name, and `remove_joint`
    /// drops the motors of the removed joint and re-points those of the
    /// last joint to the freed index (swap-remove).
    #[test]
    fn joint_motor_accessors_and_removal_remap() {
        let mut world = quiet_world();
        for x in 0..3 {
            world.add_body(RigidBody::new(Vec3Fix::from_int(x, 0, 0), Fix128::ONE));
        }
        let ball = |a, b| {
            Joint::Ball(crate::joint::BallJoint::new(
                a,
                b,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
            ))
        };
        let j0 = world.add_joint(ball(0, 1));
        let j1 = world.add_joint(ball(1, 2));
        let pd = crate::motor::PdController::new(Fix128::ONE, Fix128::ONE, Fix128::from_int(10));
        let m0 = world.add_joint_motor(j0, pd);
        let m1 = world.add_joint_motor(j1, pd);
        assert!(world.set_joint_motor_velocity_target(m1, Fix128::from_int(2)));
        assert!(!world.set_joint_motor_velocity_target(9, Fix128::ONE));
        assert_eq!(
            world.joint_motor_mut(m1).unwrap().controller.mode,
            crate::motor::MotorMode::Velocity
        );
        assert!(world.disable_joint_motor(m1));
        assert!(!world.disable_joint_motor(9));
        assert_eq!(
            world.joint_motor_mut(m1).unwrap().controller.mode,
            crate::motor::MotorMode::Off
        );
        assert!(world.joint_motor_mut(9).is_none());

        let r0 = world.add_joint_motor_3d(
            j0,
            Vec3Fix::from_int(1, 1, 1),
            Vec3Fix::from_int(1, 1, 1),
            Fix128::from_int(5),
        );
        let r1 = world.add_joint_motor_3d(
            j1,
            Vec3Fix::from_int(1, 1, 1),
            Vec3Fix::from_int(1, 1, 1),
            Fix128::from_int(5),
        );
        let target = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 1, 0), Fix128::from_ratio(1, 4));
        assert!(world.set_joint_motor_3d_rotation_target(r1, target));
        assert!(!world.set_joint_motor_3d_rotation_target(9, target));
        assert_eq!(
            world.joint_motor_3d_mut(r1).unwrap().target_rotation,
            target
        );
        assert!(world.joint_motor_3d_mut(9).is_none());
        let _ = (m0, r0);

        assert!(world.remove_joint(7).is_none());
        assert!(world.remove_joint(j0).is_some());
        // the motors of joint 0 are gone, the ones of joint 1 now name joint 0
        assert_eq!(world.joint_motors.len(), 1);
        assert_eq!(world.joint_motors[0].joint_index, 0);
        assert_eq!(world.joint_motors_3d.len(), 1);
        assert_eq!(world.joint_motors_3d[0].0, 0);
        assert_eq!(world.joint_motors_3d[0].1.target_rotation, target);
    }

    /// oracle: a contact carrying `λ` between a body of mass 1 (`w = 1`) and a
    /// static body (`w = 0`), in a frame `dt = 1` of 8 substeps (`h = 1/8`),
    /// carries the force `F = λ / (w·h²) = 64·λ`; with `λ = 1/2` that is 32.
    /// It gives one normal arrow of 32, two friction arrows of `μ·F = 0.3·32`
    /// and one cone of height 32. A contact between two static bodies
    /// carries none.
    #[test]
    fn contact_forces_are_lambda_over_w_h_squared() {
        let mut world = PhysicsWorld::new(SolverConfig::default());
        let a = world.add_body(RigidBody::new(Vec3Fix::from_int(0, 1, 0), Fix128::ONE));
        let b = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let s = world.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)));
        let contact = Contact {
            depth: Fix128::from_ratio(1, 100),
            normal: Vec3Fix::from_int(0, 1, 0),
            point_a: Vec3Fix::from_int(0, 1, 0),
            point_b: Vec3Fix::ZERO,
        };
        let mut c = ContactConstraint::new(a, b, contact);
        c.cached_lambda = Fix128::from_ratio(1, 2);
        let mut dead = ContactConstraint::new(b, s, contact);
        dead.cached_lambda = Fix128::ONE;
        world.contact_constraints = vec![c, dead];
        let dt = Fix128::ONE;
        let forces = world.contact_forces(dt);
        assert_eq!(forces.len(), 1);
        assert_eq!(forces[0].0, Vec3Fix::from_int(0, 1, 0));
        assert_eq!(forces[0].1, Vec3Fix::from_int(0, 1, 0));
        assert_eq!(forces[0].2, Fix128::from_int(32));
        assert_eq!(
            world.contact_constraint_force(&world.contact_constraints[0], dt),
            Some(Fix128::from_int(32))
        );
        assert_eq!(
            world.contact_constraint_force(&world.contact_constraints[1], dt),
            None
        );
        let arrows = world.contact_arrows(dt);
        assert_eq!(arrows.len(), 1);
        assert_eq!(arrows[0].force_magnitude, Fix128::from_int(32));
        let friction = world.contact_friction_arrows(dt);
        assert_eq!(friction.len(), 2);
        let mu_f = (Fix128::from_int(32) * Fix128::from_ratio(3, 10)).to_f64();
        for f in &friction {
            assert!((f.force_magnitude.to_f64() - 9.6).abs() < 1e-9 && (mu_f - 9.6).abs() < 1e-9);
        }
        let cones = world.contact_friction_cones(dt);
        assert_eq!(cones.len(), 1);
        assert_eq!(cones[0].height, Fix128::from_int(32));
    }

    /// oracle: static colliders are indexed in insertion order, `remove`
    /// shifts the later ones down and refuses an index past the end.
    #[test]
    fn static_collider_add_remove_count() {
        let mut world = quiet_world();
        let plane = |h: i64| {
            crate::static_collider::StaticCollider::Plane(
                crate::plane_collider::PlaneCollider::new(
                    Vec3Fix::from_int(0, 1, 0),
                    Fix128::from_int(h),
                ),
            )
        };
        assert_eq!(world.add_static_collider(plane(0)), 0);
        assert_eq!(world.add_static_collider(plane(1)), 1);
        assert_eq!(world.static_collider_count(), 2);
        assert!(world.remove_static_collider(2).is_none());
        assert!(world.remove_static_collider(0).is_some());
        assert_eq!(world.static_collider_count(), 1);
        // the plane at height 1 moved to index 0: a sphere of radius 1/2 at
        // y = 1 (its centre on that plane) touches it with depth 1/2
        let c = world.static_colliders_slice()[0]
            .collide_sphere(Vec3Fix::from_int(0, 1, 0), Fix128::from_ratio(1, 2))
            .expect("touching the plane at y = 1");
        assert!((c.depth.to_f64() - 0.5).abs() < 1e-9, "{c:?}");
    }

    /// oracle: with the dynamic-tree broad-phase every body with a collision
    /// radius has a proxy after a step, and its fattened box contains the
    /// sphere's own box (`centre ± r`); a body without a radius has none.
    /// With the hybrid broad-phase two overlapping spheres (gap −1/2) are
    /// found and pushed apart, while a far one is left at rest.
    #[test]
    fn dynamic_tree_and_hybrid_broadphases_find_the_pairs() {
        for kind in [Broadphase::DynamicTree, Broadphase::Hybrid] {
            let mut world = quiet_world();
            world.set_broadphase(kind);
            assert_eq!(world.broadphase(), kind);
            let a =
                world.add_body_with_radius(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE), Fix128::ONE);
            let b = world.add_body_with_radius(
                RigidBody::new(
                    Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
                    Fix128::ONE,
                ),
                Fix128::ONE,
            );
            let far = world.add_body_with_radius(
                RigidBody::new(Vec3Fix::from_int(50, 0, 0), Fix128::ONE),
                Fix128::ONE,
            );
            let bare = world.add_body(RigidBody::new(Vec3Fix::from_int(-50, 0, 0), Fix128::ONE));
            world.step(Fix128::from_ratio(1, 60));
            let gap = (world.bodies[b].position - world.bodies[a].position).length();
            assert!(gap > Fix128::from_ratio(3, 2), "{kind:?}: gap {gap:?}");
            assert_eq!(world.bodies[far].position, Vec3Fix::from_int(50, 0, 0));
            if kind == Broadphase::DynamicTree {
                assert_eq!(world.broadphase_stats().proxies, 3);
                let bx = world.broadphase_proxy_aabb(far).expect("proxy");
                assert!(bx.min.x <= Fix128::from_int(49) && bx.max.x >= Fix128::from_int(51));
                assert!(world.broadphase_proxy_aabb(bare).is_none());
                assert!(world.broadphase_proxy_aabb(99).is_none());
            }
        }
    }

    /// A participant adding a constant force to one body (or failing).
    struct ConstantPush {
        body: usize,
        force: Vec3Fix,
        fail: bool,
    }

    impl crate::world_participant::Participant for ConstantPush {
        fn kind(&self) -> crate::world_participant::ParticipantKind {
            crate::world_participant::ParticipantKind::new(0x5055_5348)
        }
        fn substep(
            &mut self,
            ctx: &mut crate::world_participant::SubstepCtx<'_>,
            _h: Fix128,
        ) -> Result<(), crate::world_participant::ParticipantFault> {
            if self.fail {
                return Err(crate::world_participant::ParticipantFault::InvalidState);
            }
            ctx.add_force(self.body, self.force)
                .map_err(|_| crate::world_participant::ParticipantFault::InvalidState)
        }
        fn observe(&self, _: &mut crate::world_participant::ObservationSink) {}
        fn write_state(&self, _: &mut Vec<u8>) {}
        fn check_state(&self, _: &[u8]) -> Result<(), crate::world_participant::StateError> {
            Ok(())
        }
        fn read_state(&mut self, _: &[u8]) {}
    }

    /// oracle: under the TGS backend, with no gravity and no damping, a
    /// participant applying `F = (4, 0, 0)` to a body of mass 2 for one frame
    /// `dt = 1/64` (8 dyadic substeps) gives `Δv = F·dt/m = 1/32` along `x`,
    /// with and without a joint in the world; a participant that fails
    /// records a participant fault and applies nothing.
    #[test]
    fn tgs_participant_force_gives_f_dt_over_m() {
        for with_joint in [false, true] {
            for fail in [false, true] {
                let mut world = PhysicsWorld::new(SolverConfig {
                    gravity: Vec3Fix::ZERO,
                    damping: Fix128::ONE,
                    solver_backend: SolverBackend::Tgs,
                    ..Default::default()
                });
                let body = world.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
                if with_joint {
                    let other =
                        world.add_body(RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE));
                    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 12, 0)));
                    world.add_joint(Joint::Ball(crate::joint::BallJoint::new(
                        other,
                        anchor,
                        Vec3Fix::ZERO,
                        Vec3Fix::from_int(0, -2, 0),
                    )));
                }
                world
                    .add_participant(Box::new(ConstantPush {
                        body,
                        force: Vec3Fix::from_int(4, 0, 0),
                        fail,
                    }))
                    .expect("register");
                world.step(Fix128::from_ratio(1, 64));
                let v = world.bodies[body].velocity;
                if fail {
                    assert_eq!(v, Vec3Fix::ZERO, "joint {with_joint}");
                    assert!(
                        matches!(
                            world.fault(),
                            Some(crate::world_participant::WorldFault::Participant {
                                index: 0,
                                ..
                            })
                        ),
                        "joint {with_joint}: {:?}",
                        world.fault()
                    );
                } else {
                    assert!(
                        (v.x.to_f64() - 1.0 / 32.0).abs() < 1e-12,
                        "joint {with_joint}: v {v:?}"
                    );
                    assert!(
                        v.y.is_zero() && v.z.is_zero(),
                        "joint {with_joint}: v {v:?}"
                    );
                    assert_eq!(world.fault(), None);
                }
            }
        }
    }

    /// oracle: for an isotropic `inv_inertia = (c, c, c)` the checked world
    /// inverse inertia is `Some(c · τ)` (component-wise `checked_mul`) for 64
    /// non-identity orientations, equal to the unchecked value, and `None`
    /// when a product leaves the range.
    #[test]
    fn checked_world_inv_inertia_apply_of_an_isotropic_body_is_c_tau() {
        let c = Fix128::from_ratio(5, 2);
        let tau = Vec3Fix::new(
            Fix128::from_ratio(3, 7),
            Fix128::from_ratio(-11, 13),
            Fix128::from_ratio(1, 3),
        );
        let expected = Vec3Fix::new(tau.x * c, tau.y * c, tau.z * c);
        let mut differ = 0;
        for i in 0..16_i64 {
            let axis = Vec3Fix::new(
                Fix128::from_ratio(1 + i, 3),
                Fix128::from_ratio(2 - i, 5),
                Fix128::from_ratio(3 + 2 * i, 7),
            )
            .normalize();
            for k in 1..=4_i64 {
                let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
                b.inv_inertia = Vec3Fix::new(c, c, c);
                b.rotation =
                    QuatFix::from_axis_angle(axis, Fix128::from_ratio(7 * k + i, 9)).normalize();
                let got = b.checked_world_inv_inertia_apply(tau);
                assert_eq!(got, Some(b.world_inv_inertia_apply(tau)));
                differ += usize::from(got != Some(expected));
            }
        }
        assert_eq!(differ, 0, "{differ}/64 orientations differ from c·τ");
        let mut big = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        big.inv_inertia = Vec3Fix::new(
            Fix128::from_int(4),
            Fix128::from_int(4),
            Fix128::from_int(4),
        );
        let huge = Vec3Fix::new(Fix128::from_int(1_i64 << 62), Fix128::ZERO, Fix128::ZERO);
        assert_eq!(big.checked_world_inv_inertia_apply(huge), None);
    }
}

/// loom model of the invariant `BodySlicePtr` / `DistConstraintSlicePtr` /
/// `ContactConstraintSlicePtr` rely on: every constraint of one colour batch
/// touches disjoint dynamic bodies, so the batch can be written by any
/// number of threads without synchronisation.
///
/// The raw-pointer wrappers themselves are invisible to loom (plain memory);
/// what loom can exhaustively check is the *protocol*: take the real batches
/// `rebuild_batches` produces for scenes that exercise the hard cases (a
/// shared static floor, a dynamic hub, rods under contact), give every
/// constraint of a batch to its own loom thread, and let each thread write
/// its two body slots through a `loom::cell::UnsafeCell`. If the colouring
/// ever placed two constraints sharing a dynamic body in one batch, loom
/// reports the unsynchronised concurrent access on one of its interleavings.
/// Run with `RUSTFLAGS="--cfg loom" cargo test --lib loom_` (quality-deep).
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell;
    use loom::sync::Arc;
    use loom::thread;

    fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    /// (dynamic-body pairs per batch) for a scene: static bodies are excluded
    /// because the solver borrows them shared (`BodyRef::Static`).
    fn batches_of(world: &mut PhysicsWorld) -> Vec<Vec<(Option<usize>, Option<usize>)>> {
        world.rebuild_batches();
        let dynamic = |i: usize| (!world.bodies[i].inv_mass.is_zero()).then_some(i);
        world
            .constraint_batches
            .iter()
            .map(|b| {
                let mut pairs = Vec::new();
                for &ci in &b.distance_indices {
                    let c = world.distance_constraints[ci];
                    pairs.push((dynamic(c.body_a), dynamic(c.body_b)));
                }
                for &ci in &b.contact_indices {
                    let c = world.contact_constraints[ci];
                    pairs.push((dynamic(c.body_a), dynamic(c.body_b)));
                }
                pairs
            })
            .collect()
    }

    fn check_batches(batches: Vec<Vec<(Option<usize>, Option<usize>)>>, n_bodies: usize) {
        for batch in batches {
            // loom explores every interleaving of up to a handful of threads
            for chunk in batch.chunks(3) {
                let chunk: Vec<_> = chunk.to_vec();
                let n = n_bodies;
                loom::model(move || {
                    let slots: Arc<Vec<UnsafeCell<u32>>> =
                        Arc::new((0..n).map(|_| UnsafeCell::new(0)).collect());
                    let handles: Vec<_> = chunk
                        .iter()
                        .copied()
                        .map(|(a, b)| {
                            let slots = Arc::clone(&slots);
                            thread::spawn(move || {
                                for body in [a, b].into_iter().flatten() {
                                    // the solver writes position / velocity of
                                    // each dynamic body of its constraint
                                    slots[body].with_mut(|p| unsafe { *p += 1 });
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            }
        }
    }

    #[test]
    fn loom_batches_touch_disjoint_dynamic_bodies_static_floor_hub() {
        // 6 dynamic spheres resting on one static sphere + 2 rods
        let mut world = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let floor =
            world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::from_int(4));
        let mut ids = Vec::new();
        for i in 0..6i64 {
            let angle = Fix128::from_ratio(i, 6) * Fix128::TWO_PI;
            let (s, c) = angle.sin_cos();
            let pos = Vec3Fix::new(
                c * Fix128::from_int(4),
                s * Fix128::from_int(4),
                Fix128::ZERO,
            );
            ids.push(
                world.add_body_with_radius(RigidBody::new_dynamic(pos, Fix128::ONE), Fix128::ONE),
            );
        }
        for w in ids.windows(2).take(2) {
            world.add_distance_constraint(DistanceConstraint {
                body_a: w[0],
                body_b: w[1],
                local_anchor_a: Vec3Fix::ZERO,
                local_anchor_b: Vec3Fix::ZERO,
                target_distance: Fix128::from_int(4),
                compliance: Fix128::ZERO,
                cached_lambda: Fix128::ZERO,
            });
        }
        let _ = floor;
        world.clear_contacts();
        world.detect_collisions();
        assert!(
            !world.contact_constraints.is_empty(),
            "spheres must touch the floor"
        );
        let n = world.bodies.len();
        check_batches(batches_of(&mut world), n);
    }

    #[test]
    fn loom_batches_touch_disjoint_dynamic_bodies_dynamic_hub() {
        // one dynamic hub connected to 5 dynamic spokes: every spoke rod
        // shares the hub, so the colouring must serialise them into 5 batches
        let mut world = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let hub = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        for i in 0..5i64 {
            let spoke = world.add_body(RigidBody::new_dynamic(v3(3 * (i + 1), 0, 0), Fix128::ONE));
            world.add_distance_constraint(DistanceConstraint {
                body_a: hub,
                body_b: spoke,
                local_anchor_a: Vec3Fix::ZERO,
                local_anchor_b: Vec3Fix::ZERO,
                target_distance: Fix128::from_int(3 * (i + 1)),
                compliance: Fix128::ZERO,
                cached_lambda: Fix128::ZERO,
            });
        }
        let n = world.bodies.len();
        let batches = batches_of(&mut world);
        assert_eq!(batches.len(), 5, "hub forces one batch per spoke");
        check_batches(batches, n);
    }
}
