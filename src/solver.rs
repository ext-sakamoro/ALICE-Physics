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

use crate::bvh::{BvhPrimitive, LinearBvh};
use crate::collider::{Contact, AABB};
use crate::event::EventCollector;
use crate::filter::CollisionFilter;
use crate::force::{apply_force_fields, ForceFieldInstance};
use crate::joint::{solve_joints, Joint};
use crate::math::{select_vec3, Fix128, QuatFix, Vec3Fix};
use crate::sdf_collider::SdfCollider;
use crate::sleeping::{IslandManager, SleepConfig};

/// Minimum effective inverse-mass sum below which constraint solving is skipped.
/// Prevents division explosion when two near-static bodies are in contact.
/// Value: ~2^-40 ≈ 9.1e-13 in Fix128.
const W_SUM_EPSILON: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000010000000000,
};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Rayon parallel iterator support
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    /// Coefficient of restitution (bounciness)
    pub restitution: Fix128,
    /// Friction coefficient
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
            restitution: Fix128::from_ratio(5, 10), // 0.5 default
            friction: Fix128::from_ratio(3, 10),    // 0.3 default
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
            restitution: Fix128::ZERO,
            friction: Fix128::ONE,
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
            restitution: Fix128::ZERO,
            friction: Fix128::ONE,
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
    pub fn set_kinematic_target(&mut self, position: Vec3Fix, rotation: QuatFix) {
        self.kinematic_target = Some((position, rotation));
    }

    /// Apply impulse at center of mass
    pub fn apply_impulse(&mut self, impulse: Vec3Fix) {
        if !self.is_static() {
            self.velocity = self.velocity + impulse * self.inv_mass;
        }
    }

    /// Apply impulse at world-space point
    pub fn apply_impulse_at(&mut self, impulse: Vec3Fix, point: Vec3Fix) {
        if !self.is_static() {
            self.velocity = self.velocity + impulse * self.inv_mass;

            let r = point - self.position;
            let torque = r.cross(impulse);
            self.angular_velocity = self.angular_velocity
                + Vec3Fix::new(
                    torque.x * self.inv_inertia.x,
                    torque.y * self.inv_inertia.y,
                    torque.z * self.inv_inertia.z,
                );
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
    #[inline]
    pub fn add_torque(&mut self, torque: Vec3Fix, dt: Fix128) {
        if !self.is_static() {
            self.angular_velocity = self.angular_velocity
                + Vec3Fix::new(
                    torque.x * self.inv_inertia.x * dt,
                    torque.y * self.inv_inertia.y * dt,
                    torque.z * self.inv_inertia.z * dt,
                );
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
    #[inline]
    pub fn set_rotation(&mut self, rotation: QuatFix) {
        self.rotation = rotation;
        self.prev_rotation = rotation;
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
    /// Force fields applied at the start of each step
    pub force_fields: Vec<ForceFieldInstance>,
    /// Contact and trigger event collector
    pub events: EventCollector,
    /// Island manager for sleeping and connectivity tracking
    pub islands: IslandManager,
    /// Per-body collision radius for automatic sphere-based collision detection.
    /// `None` means the body does not participate in auto-detection.
    body_collision_radii: Vec<Option<Fix128>>,
    /// Per-body collision filter (layer/mask/group)
    body_filters: Vec<CollisionFilter>,
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
            #[cfg(feature = "gpu-solver-bridge")]
            gpu_solver_bridge: None,
            joints: Vec::new(),
            force_fields: Vec::new(),
            events: EventCollector::new(),
            islands: IslandManager::new(0, SleepConfig::default()),
            body_collision_radii: Vec::new(),
            body_filters: Vec::new(),
        }
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

    /// Add rigid body, returns index
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        self.body_collision_radii.push(None);
        self.body_filters.push(CollisionFilter::DEFAULT);
        self.islands.resize(idx + 1);
        idx
    }

    /// Add rigid body with a collision sphere radius for automatic detection.
    ///
    /// Bodies with a collision radius participate in BVH broad-phase and
    /// sphere-sphere narrow-phase collision detection during `step()`.
    pub fn add_body_with_radius(&mut self, body: RigidBody, radius: Fix128) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        self.body_collision_radii.push(Some(radius));
        self.body_filters.push(CollisionFilter::DEFAULT);
        self.islands.resize(idx + 1);
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
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a != idx && b != idx
        });

        let removed = self.bodies.swap_remove(idx);
        self.body_materials.swap_remove(idx);
        self.body_collision_radii.swap_remove(idx);
        self.body_filters.swap_remove(idx);

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
    pub fn remove_joint(&mut self, idx: usize) -> Option<Joint> {
        if idx >= self.joints.len() {
            return None;
        }
        Some(self.joints.swap_remove(idx))
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
            self.body_collision_radii[body_idx] = Some(radius);
        }
    }

    /// Remove collision radius (disable automatic collision detection for this body)
    pub fn clear_body_collision_radius(&mut self, body_idx: usize) {
        if body_idx < self.body_collision_radii.len() {
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
        self.islands.wake_island(body_idx);
    }

    /// Set the sleep configuration
    pub fn set_sleep_config(&mut self, config: SleepConfig) {
        self.islands.config = config;
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

    /// Cast a ray against all body collision spheres, returns (`body_index`, distance).
    ///
    /// Uses a BVH broad-phase to cull bodies outside the ray's bounding box,
    /// then performs exact ray-sphere intersection on candidates.
    /// Returns `None` if direction is zero or no body is hit.
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
    #[must_use]
    pub fn combined_material(
        &self,
        body_a: usize,
        body_b: usize,
    ) -> crate::material::CombinedMaterial {
        let mat_a = if body_a < self.body_materials.len() {
            self.body_materials[body_a]
        } else {
            crate::material::DEFAULT_MATERIAL
        };
        let mat_b = if body_b < self.body_materials.len() {
            self.body_materials[body_b]
        } else {
            crate::material::DEFAULT_MATERIAL
        };
        self.material_table.combine(mat_a, mat_b)
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
    /// 4. Substep loop (integrate, solve constraints + joints, update velocities)
    /// 5. Update sleeping states
    /// 6. End event frame (generates end-of-contact events)
    ///
    /// Deterministic: same inputs always produce same outputs.
    pub fn step(&mut self, dt: Fix128) {
        // Guard: non-positive dt produces no physics update
        if dt <= Fix128::ZERO {
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

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }

        // Phase 2 (collision detection) moved into the substep — Small Steps
        // (Müller et al. 2020): a frame-level contact set solved 8× with a
        // stale depth re-pushed the penetration every substep and turned a
        // 5 m/s head-on collision into 700 m/s (1.2.0, R4-2).

        // Phase 3: Substep loop
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

        // Phase 5: End event frame
        self.events.end_frame();
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
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, dt: Fix128) {
        // Guard: non-positive dt produces no physics update
        if dt <= Fix128::ZERO {
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

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }

        // Phase 2 (collision detection) is per substep since 1.2.0, see `step`.

        // Phase 3: Substep loop (batches are rebuilt inside each substep after
        // detection, `solve_constraints_batched` re-colours on dirty)
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep_batched(substep_dt);
        }

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

        // Phase 5: End event frame
        self.events.end_frame();
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.2. Collision detection on the predicted positions (Small Steps).
        //      Contacts live for exactly one substep.
        self.clear_contacts();
        self.detect_collisions();

        // 1.5. Resolve SDF collisions (implicit surface contacts)
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }

        // 2. Solve constraints (sequential)
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
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.2. Collision detection on the predicted positions (Small Steps).
        self.clear_contacts();
        self.detect_collisions();
        self.rebuild_batches();

        // 1.5. Resolve SDF collisions
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }

        // 2. Solve constraints (batched)
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
    /// Sleeping dynamic bodies are skipped: only `prev_position/prev_rotation`
    /// are saved so that `update_velocities` derives zero velocity.
    #[inline]
    fn integrate_positions(&mut self, dt: Fix128) {
        #[cfg(feature = "parallel")]
        {
            let gravity = self.config.gravity;
            let sleep_data = &self.islands.sleep_data;
            self.bodies
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, body)| {
                    match body.body_type {
                        BodyType::Static => return,
                        BodyType::Kinematic => {
                            body.prev_position = body.position;
                            body.prev_rotation = body.rotation;
                            if let Some((target_pos, target_rot)) = body.kinematic_target {
                                body.velocity = (target_pos - body.position) * (Fix128::ONE / dt);
                                body.position = target_pos;
                                body.rotation = target_rot;
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
                    body.position = body.position + body.velocity * dt;

                    // Predict rotation (single sqrt via normalize_with_length)
                    let (axis, ang_speed) = body.angular_velocity.normalize_with_length();
                    if !ang_speed.is_zero() {
                        let angle = ang_speed * dt;
                        let delta_rot = QuatFix::from_axis_angle(axis, angle);
                        body.rotation = delta_rot.mul(body.rotation).normalize();
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..self.bodies.len() {
                match self.bodies[i].body_type {
                    BodyType::Static => continue,
                    BodyType::Kinematic => {
                        self.bodies[i].prev_position = self.bodies[i].position;
                        self.bodies[i].prev_rotation = self.bodies[i].rotation;
                        if let Some((target_pos, target_rot)) = self.bodies[i].kinematic_target {
                            self.bodies[i].velocity =
                                (target_pos - self.bodies[i].position) * (Fix128::ONE / dt);
                            self.bodies[i].position = target_pos;
                            self.bodies[i].rotation = target_rot;
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
                self.bodies[i].position = self.bodies[i].position + self.bodies[i].velocity * dt;

                // Predict rotation (single sqrt via normalize_with_length)
                let (axis, ang_speed) = self.bodies[i].angular_velocity.normalize_with_length();
                if !ang_speed.is_zero() {
                    let angle = ang_speed * dt;
                    let delta_rot = QuatFix::from_axis_angle(axis, angle);
                    self.bodies[i].rotation = delta_rot.mul(self.bodies[i].rotation).normalize();
                }
            }
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
        for i in 0..self.bodies.len() {
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

        // --- Phase 1: Derive velocities from position/rotation changes ---
        #[cfg(feature = "parallel")]
        {
            self.bodies.par_iter_mut().for_each(|body| {
                if body.body_type == BodyType::Static {
                    return;
                }
                body.velocity = (body.position - body.prev_position) * inv_dt;
                // Angular velocity from rotation change:
                // delta_q = rotation * prev_rotation^-1
                // angular_velocity = 2 * delta_q.xyz / dt  (when delta_q.w > 0)
                let dq = body.rotation.mul(body.prev_rotation.conjugate());
                let two_inv_dt = inv_dt + inv_dt;
                if dq.w < Fix128::ZERO {
                    body.angular_velocity =
                        Vec3Fix::new(-dq.x * two_inv_dt, -dq.y * two_inv_dt, -dq.z * two_inv_dt);
                } else {
                    body.angular_velocity =
                        Vec3Fix::new(dq.x * two_inv_dt, dq.y * two_inv_dt, dq.z * two_inv_dt);
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for body in &mut self.bodies {
                if body.body_type == BodyType::Static {
                    continue;
                }
                body.velocity = (body.position - body.prev_position) * inv_dt;
                // Angular velocity from rotation change
                let dq = body.rotation.mul(body.prev_rotation.conjugate());
                let two_inv_dt = inv_dt + inv_dt;
                if dq.w < Fix128::ZERO {
                    body.angular_velocity =
                        Vec3Fix::new(-dq.x * two_inv_dt, -dq.y * two_inv_dt, -dq.z * two_inv_dt);
                } else {
                    body.angular_velocity =
                        Vec3Fix::new(dq.x * two_inv_dt, dq.y * two_inv_dt, dq.z * two_inv_dt);
                }
            }
        }

        // --- Phase 2: Apply restitution and friction at contacts ---
        let num_contacts = self.contact_constraints.len();
        for i in 0..num_contacts {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            if body_a.is_sensor || body_b.is_sensor {
                continue;
            }

            let w_sum = body_a.inv_mass + body_b.inv_mass;
            if w_sum < W_SUM_EPSILON {
                continue;
            }

            let n = constraint.contact.normal;
            let relative_vel = body_a.velocity - body_b.velocity;
            let vn = relative_vel.dot(n);

            // Restitution: apply bounce on separating velocity
            if vn < Fix128::ZERO {
                let restitution = constraint.restitution;
                // delta_vn = -(1 + e) * vn
                let delta_vn = -(Fix128::ONE + restitution) * vn;
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
                // Coulomb friction: clamp tangential impulse to friction * normal impulse
                let max_friction_impulse = friction * vn2.abs();
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
    }

    /// Apply pre-solve hooks and contact modifiers to contact constraints.
    ///
    /// Runs sequentially before parallel dispatch. Contacts that are
    /// filtered out have their depth set to zero, causing the parallel
    /// solver to skip them via its existing early-return check.
    #[cfg(all(feature = "parallel", feature = "std"))]
    fn pre_process_contacts(&mut self) {
        let num = self.contact_constraints.len();
        for i in 0..num {
            let constraint = &mut self.contact_constraints[i];
            let body_a_idx = constraint.body_a;
            let body_b_idx = constraint.body_b;

            let mut skip = false;
            for hook in &self.pre_solve_hooks {
                if !hook(body_a_idx, body_b_idx, &constraint.contact) {
                    skip = true;
                    break;
                }
            }
            if !skip {
                for modifier in &self.contact_modifiers {
                    if !modifier.modify_contact(
                        body_a_idx,
                        body_b_idx,
                        &mut constraint.contact,
                        &mut constraint.friction,
                        &mut constraint.restitution,
                    ) {
                        skip = true;
                        break;
                    }
                }
            }
            if skip {
                // Mark as non-penetrating so the solver skips it
                constraint.contact.depth = Fix128::ZERO;
            }
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
    /// Pre-solve hooks and contact modifiers are applied in a sequential
    /// pre-pass before parallel dispatch begins.
    #[cfg(feature = "parallel")]
    fn solve_constraints_batched(&mut self, dt: Fix128) {
        // Apply contact modifiers before parallel dispatch
        #[cfg(feature = "std")]
        if !self.pre_solve_hooks.is_empty() || !self.contact_modifiers.is_empty() {
            self.pre_process_contacts();
        }

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
        let (pos_a, inv_mass_a, sensor_a) = {
            let a = body_a.get();
            (a.position, a.inv_mass, a.is_sensor)
        };
        let (pos_b, inv_mass_b, sensor_b) = {
            let b = body_b.get();
            (b.position, b.inv_mass, b.is_sensor)
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
        if dlambda <= Fix128::ZERO {
            return;
        }
        constraint.cached_lambda = lambda + dlambda;

        let correction = contact.normal * dlambda;
        let correction_a = correction * (inv_mass_a * inv_w_sum);
        let correction_b = correction * (inv_mass_b * inv_w_sum);

        // Branchless: inv_mass == ZERO for static bodies, correction_x will be ZERO.
        // `set_position` is a no-op for `BodyRef::Static`.
        body_a.set_position(select_vec3(
            !inv_mass_a.is_zero(),
            pos_a + correction_a,
            pos_a,
        ));
        body_b.set_position(select_vec3(
            !inv_mass_b.is_zero(),
            pos_b - correction_b,
            pos_b,
        ));
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

    /// Solve contact constraints with pre-solve hook support (Gap 2.3).
    ///
    /// # v0.11.0 auto-routing
    ///
    /// If a GPU solver bridge has been installed on this world via
    /// [`Self::set_gpu_solver_bridge`], contact-solve is routed through
    /// the bridge (via [`Self::solve_contact_constraints_with_bridge`])
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
            self.solve_contact_constraints_with_bridge(bridge.as_mut());
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

            #[cfg_attr(not(feature = "std"), allow(unused_mut))]
            let mut contact = constraint.contact;
            #[cfg_attr(not(feature = "std"), allow(unused_variables, unused_mut))]
            let mut friction = constraint.friction;
            #[cfg_attr(not(feature = "std"), allow(unused_variables, unused_mut))]
            let mut restitution = constraint.restitution;

            // Pre-solve hook: allow game logic to filter contacts
            #[cfg(feature = "std")]
            {
                let mut skip = false;
                for hook in &self.pre_solve_hooks {
                    if !hook(constraint.body_a, constraint.body_b, &contact) {
                        skip = true;
                        break;
                    }
                }
                // Contact modifiers: can mutate contact properties
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
                    continue;
                }
            }

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
            if dlambda <= Fix128::ZERO {
                continue;
            }
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
    }

    /// v0.10.0 opt-in: run one PGS contact-solve iteration via a
    /// caller-supplied [`GpuSolverBridge`](crate::gpu_bridge::GpuSolverBridge) instead of the CPU-side
    /// `solve_contact_constraints` hot loop.
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
        // ---- Stage A: filter + mutate on CPU ----
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
                continue;
            }

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
        // Guard: non-positive dt produces no physics update
        if dt <= Fix128::ZERO {
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

        // Phase 1: Apply force fields
        if !self.force_fields.is_empty() {
            apply_force_fields(&self.force_fields, &mut self.bodies, dt);
        }

        // Phase 2 (collision detection) is per substep since 1.2.0, see `step`.

        // Phase 3: Substep loop with bridge-routed contact solve
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep_with_bridge(bridge, substep_dt);
        }

        // Phase 3.5: Frame-level damping (1.2.0, was per substep)
        self.apply_frame_damping();

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

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
    #[cfg(feature = "gpu-solver-bridge")]
    pub fn substep_with_bridge<B: crate::gpu_bridge::GpuSolverBridge + ?Sized>(
        &mut self,
        bridge: &mut B,
        dt: Fix128,
    ) {
        self.integrate_positions(dt);
        self.reset_lambdas();

        // 1.2. Collision detection on the predicted positions (Small Steps).
        self.clear_contacts();
        self.detect_collisions();

        // 1.5. Resolve SDF collisions (implicit surface contacts)
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }

        // 2. Solve constraints (sequential). Distance stays CPU;
        //    contact routes through the bridge.
        for _ in 0..self.config.iterations {
            self.solve_distance_constraints(dt);
            self.solve_contact_constraints_with_bridge(bridge);
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
    /// - When `self.joints.is_empty()`, this method is a no-op and
    ///   returns without touching the bridge — useful for callers
    ///   that unconditionally route through a bridge and let this
    ///   method decide whether there's work to do.
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
        if self.joints.is_empty() {
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
        bridge.send_joints(&self.joints);
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
    pub fn add_sdf_collider(&mut self, collider: SdfCollider) -> usize {
        let idx = self.sdf_colliders.len();
        self.sdf_colliders.push(collider);
        idx
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
        let sdf_colliders = &self.sdf_colliders;
        let collision_radius = self.sdf_collision_radius;

        #[cfg(feature = "parallel")]
        {
            self.bodies.par_iter_mut().for_each(|body| {
                if body.is_static() || body.is_sensor {
                    return;
                }
                for sdf in sdf_colliders {
                    if let Some(contact) = crate::sdf_collider::collide_sphere_sdf(
                        body.position,
                        collision_radius,
                        sdf,
                    ) {
                        body.position = body.position + contact.normal * contact.depth;
                    }
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for body in &mut self.bodies {
                if body.is_static() || body.is_sensor {
                    continue;
                }
                for sdf in sdf_colliders {
                    if let Some(contact) = crate::sdf_collider::collide_sphere_sdf(
                        body.position,
                        collision_radius,
                        sdf,
                    ) {
                        body.position = body.position + contact.normal * contact.depth;
                    }
                }
            }
        }
    }

    // ── Automatic Collision Detection ─────────────────────────────────

    /// Detect collisions between bodies with collision radii.
    ///
    /// Uses BVH broad-phase with Morton codes and sphere-sphere narrow-phase.
    /// Generates contact constraints and events automatically.
    /// Bodies without a collision radius are skipped.
    #[allow(clippy::too_many_lines, clippy::items_after_statements)]
    fn detect_collisions(&mut self) {
        let n = self.bodies.len();
        if n < 2 {
            return;
        }

        // Build BVH from bodies that have collision radii
        let mut primitives = Vec::new();
        for i in 0..n {
            if let Some(radius) = self.body_collision_radii.get(i).and_then(|r| *r) {
                let pos = self.bodies[i].position;
                let half = Vec3Fix::new(radius, radius, radius);
                let aabb = AABB::from_center_half(pos, half);
                primitives.push(BvhPrimitive {
                    aabb,
                    index: i as u32,
                    morton: 0,
                });
            }
        }

        if primitives.len() < 2 {
            return;
        }

        let bvh = LinearBvh::build(primitives);
        let pairs = bvh.find_pairs();

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
            let combined_radius = radius_a + radius_b;
            // Squared-distance early out *first*: the BVH candidate set is a
            // superset of the overlapping pairs (49k candidates for 2.7k contacts
            // on the 1000-sphere grid) and the quantised leaf AABBs cannot be
            // tightened without a BVH API change, so every candidate pays only
            // 3 multiplies + a compare here; the filter / static / sleep lookups
            // and the sqrt + 3 divisions of `normalize_with_length` run only for
            // real overlaps. Same `dist < combined_radius` decision (both sides
            // exact for |delta| < 2^31), same contact order.
            let dist_sq = delta.length_squared();
            if dist_sq >= combined_radius * combined_radius || dist_sq.is_zero() {
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

            let (normal, dist) = delta.normalize_with_length();

            if dist < combined_radius && !dist.is_zero() {
                let depth = combined_radius - dist;
                let point_a = self.bodies[a].position - normal * radius_a;
                let point_b = self.bodies[b].position + normal * radius_b;
                // Approach speed along the normal: negative while closing.
                let rel_vel = (self.bodies[a].velocity - self.bodies[b].velocity).dot(normal);
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
            if self.islands.is_sleeping(info.body_a) {
                self.islands.wake_island(info.body_a);
            }
            if self.islands.is_sleeping(info.body_b) {
                self.islands.wake_island(info.body_b);
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

    /// Serialize world state (for rollback netcode).
    ///
    /// Saves per-body: position, velocity, rotation, angular velocity.
    /// Does NOT save constraints, joints, force fields, collision radii,
    /// or filters — in rollback netcode these are derived from game state
    /// and re-created each frame.
    #[must_use]
    pub fn serialize_state(&self) -> Vec<u8> {
        let mut data = Vec::new();

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

        data
    }

    /// Deserialize world state (for rollback netcode).
    ///
    /// Restores per-body transforms. Parallel arrays (collision radii,
    /// filters, materials, island manager) are resized to match the body
    /// count, preserving existing entries and zero-filling new ones.
    pub fn deserialize_state(&mut self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if count != self.bodies.len() {
            return false;
        }

        // Per-body: position(48) + velocity(48) + rotation(64) + angular_velocity(48) = 208 bytes
        let mut offset = 4;
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
        while self.body_filters.len() < n {
            self.body_filters.push(CollisionFilter::DEFAULT);
        }
        self.body_filters.truncate(n);
        while self.body_materials.len() < n {
            self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        }
        self.body_materials.truncate(n);
        self.islands = IslandManager::new(n, self.islands.config);

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
            .field("joints", &self.joints.len())
            .field("force_fields", &self.force_fields.len())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

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

        // Deserialize into another world
        let mut world2 = PhysicsWorld::new(config);
        world2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        world2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));

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

        // dq = q * conj(I) = q、angular = 2 * dq.xyz / dt = dq.xyz * 8
        let eight = Fix128::from_int(8);
        let expected = Vec3Fix::new(q.x * eight, q.y * eight, q.z * eight);
        assert_eq!(world.bodies[idx].angular_velocity, expected);
        assert!(expected.x > Fix128::ZERO && expected.z > Fix128::ZERO);
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

        let eight = Fix128::from_int(8);
        let expected = Vec3Fix::new(-q.x * eight, -q.y * eight, -q.z * eight);
        assert_eq!(world.bodies[idx].angular_velocity, expected);
        // 3 成分とも非零で符号が独立に検証されること
        assert!(
            expected.x != Fix128::ZERO && expected.y != Fix128::ZERO && expected.z != Fix128::ZERO
        );
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

        assert_eq!(world.bodies[idx].angular_velocity, v3(0, 0, 8));
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
    fn give_velocity(world: &mut PhysicsWorld, idx: usize, v: Vec3Fix) {
        let b = &mut world.bodies[idx];
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
        freeze_positions(&mut world);
        // 法線成分は正 (離れる) にして restitution を素通りさせ、vn2 = +4 を Coulomb 上限の元にする
        give_velocity(&mut world, a, v3(4, 3, 0));

        world.update_velocities(r(1, 4));

        // tangent_speed = 3、max = 0.5 * 4 = 2 → clamp 2、A.y -= 2 * 1 → 1、x は不変
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
        push_contact(&mut world, a, b, Fix128::ONE, Fix128::ZERO); // μ = 1 → max = |vn2| = 4
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
        push_contact(&mut world, a, b, Fix128::ONE, Fix128::ZERO); // max = |vn2| = 3
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
        vetoed.solve_contact_constraints(r(1, 4));
        assert_eq!(vetoed.bodies[0].position, Vec3Fix::ZERO);

        // hook が true → 通常通り
        let mut allowed = contact_world(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        allowed.add_pre_solve_hook(Box::new(|_a, _b, _c| true));
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
        assert_eq!(bytes.len(), 4 + 2 * 208);
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
        let src = snapshot_world();
        let bytes = src.serialize_state();
        for pos in 4..bytes.len() {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            let mut dst = snapshot_world();
            assert!(dst.deserialize_state(&bad), "byte {pos}");
            let body = (pos - 4) / 208;
            let field = ((pos - 4) % 208) / 48; // 0 pos / 1 vel / 2 rot(64 byte = 2 slot 相当は 128..192) / 3 ang
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
            let off = (pos - 4) % 208;
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
        assert!(!dst.deserialize_state(&bytes[..3])); // header 未満
        assert!(!dst.deserialize_state(&bytes[..4 + 208 + 100])); // 2 体目が途中で切れる
        let mut one_body = quiet_world();
        one_body.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        assert!(!one_body.deserialize_state(&bytes)); // count 不一致
                                                      // 拒否時は元の状態を保つ (先頭 body だけ書き換わっていない)
        assert_eq!(one_body.bodies[0].position, Vec3Fix::ZERO);
        // ぴったり 1 body 分 + header なら OK
        let mut exact = quiet_world();
        exact.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut one = 1u32.to_le_bytes().to_vec();
        one.extend_from_slice(&bytes[4..4 + 208]);
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
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);

        // clear 後は hook 0 個 → 通常通り λ = 1、inv_w = 1/2 → A += 1/2
        world.clear_pre_solve_hooks();
        assert!(world.pre_solve_hooks.is_empty());
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
        world.solve_contact_constraints(r(1, 4));
        assert_eq!(world.bodies[0].position, Vec3Fix::ZERO);

        world.clear_contact_modifiers();
        assert!(world.contact_modifiers.is_empty());
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
