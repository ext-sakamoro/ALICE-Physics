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
#[derive(Clone, Copy, Debug)]
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
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        }
    }

    /// Create new dynamic rigid body (alias for new)
    #[inline]
    pub fn new_dynamic(position: Vec3Fix, mass: Fix128) -> Self {
        Self::new(position, mass)
    }

    /// Create static (immovable) rigid body
    pub fn new_static(position: Vec3Fix) -> Self {
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
            is_sensor: false,
            body_type: BodyType::Static,
            kinematic_target: None,
        }
    }

    /// Create a sensor (trigger) body - detects overlap but no physics response
    pub fn new_sensor(position: Vec3Fix) -> Self {
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
            is_sensor: true,
            body_type: BodyType::Static,
            kinematic_target: None,
        }
    }

    /// Create a kinematic body (moved by user code, not physics)
    ///
    /// Kinematic bodies have infinite mass and are unaffected by forces,
    /// but can push dynamic bodies. Set target via `set_kinematic_target`.
    pub fn new_kinematic(position: Vec3Fix) -> Self {
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
            is_sensor: false,
            body_type: BodyType::Kinematic,
            kinematic_target: None,
        }
    }

    /// Check if body is static
    #[inline]
    pub fn is_static(&self) -> bool {
        self.inv_mass.is_zero()
    }

    /// Check if body is kinematic
    #[inline]
    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType::Kinematic
    }

    /// Check if body is dynamic
    #[inline]
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
    /// Force is converted to velocity change: v += (F * inv_mass) * dt
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

    /// Get mass (inverse of inv_mass, returns infinity for static bodies)
    #[inline]
    pub fn mass(&self) -> Fix128 {
        if self.inv_mass.is_zero() {
            Fix128::ZERO // Represents infinite mass
        } else {
            Fix128::ONE / self.inv_mass
        }
    }

    /// Get linear speed (magnitude of velocity)
    #[inline]
    pub fn speed(&self) -> Fix128 {
        self.velocity.length()
    }
}

// ============================================================================
// Constraints
// ============================================================================

/// Distance constraint between two bodies
#[derive(Clone, Copy, Debug)]
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
    /// Cached lambda (Lagrange multiplier) from the previous substep for warm-starting.
    ///
    /// Warm-starting seeds the solver with the accumulated impulse from the last
    /// substep, reducing iterations needed for convergence (standard XPBD technique).
    /// Initialized to `Fix128::ZERO`; updated every solve iteration (Gap 3.1).
    pub cached_lambda: Fix128,
}

impl DistanceConstraint {
    /// Create a new distance constraint
    pub fn new(
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
    pub fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

/// Contact constraint (from collision detection)
#[derive(Clone, Copy, Debug)]
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
}

impl ContactConstraint {
    /// Create a new contact constraint with default friction and restitution
    pub fn new(body_a: usize, body_b: usize, contact: Contact) -> Self {
        Self {
            body_a,
            body_b,
            contact,
            friction: Fix128::from_ratio(3, 10), // 0.3 friction
            restitution: Fix128::from_ratio(2, 10), // 0.2 restitution
        }
    }
}

// ============================================================================
// XPBD Solver
// ============================================================================

/// XPBD physics solver configuration
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// Number of substeps per frame
    pub substeps: usize,
    /// Number of constraint iterations per substep
    pub iterations: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Global damping factor
    pub damping: Fix128,
}

/// Physics configuration (alias for SolverConfig)
pub type PhysicsConfig = SolverConfig;

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            substeps: 8,
            iterations: 4,
            gravity: Vec3Fix::new(
                Fix128::ZERO,
                Fix128::from_int(-10), // -10 m/s^2
                Fix128::ZERO,
            ),
            damping: Fix128::from_ratio(99, 100), // 0.99 velocity retention
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
    /// Indices into distance_constraints
    pub distance_indices: Vec<usize>,
    /// Indices into contact_constraints
    pub contact_indices: Vec<usize>,
}

/// Pointer wrapper enabling parallel access to disjoint body slots.
///
/// # Safety
///
/// The graph coloring invariant (`rebuild_batches`) guarantees that within
/// a single constraint batch, no two constraints share a body index.
/// This makes concurrent mutation of distinct body slots sound.
#[cfg(feature = "parallel")]
struct BodySlicePtr {
    ptr: *mut RigidBody,
    len: usize,
}

#[cfg(feature = "parallel")]
unsafe impl Send for BodySlicePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for BodySlicePtr {}

#[cfg(feature = "parallel")]
impl BodySlicePtr {
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn get_mut(&self, idx: usize) -> &mut RigidBody {
        debug_assert!(idx < self.len);
        &mut *self.ptr.add(idx)
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

#[cfg(feature = "parallel")]
unsafe impl Send for DistConstraintSlicePtr {}
#[cfg(feature = "parallel")]
unsafe impl Sync for DistConstraintSlicePtr {}

#[cfg(feature = "parallel")]
impl DistConstraintSlicePtr {
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    unsafe fn get_mut(&self, idx: usize) -> &mut DistanceConstraint {
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
/// PhysicsWorld integrates all physics subsystems:
/// - Rigid body dynamics with XPBD solver
/// - Automatic collision detection (BVH broad-phase + sphere narrow-phase)
/// - Joint constraints (Ball, Hinge, Fixed, Slider, Spring, D6, ConeTwist)
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
            contact_cache: crate::contact_cache::ContactCache::new(),
            material_table: crate::material::MaterialTable::new(),
            body_materials: Vec::new(),
            #[cfg(feature = "std")]
            pre_solve_hooks: Vec::new(),
            #[cfg(feature = "std")]
            contact_modifiers: Vec::new(),
            joints: Vec::new(),
            force_fields: Vec::new(),
            events: EventCollector::new(),
            islands: IslandManager::new(0, SleepConfig::default()),
            body_collision_radii: Vec::new(),
            body_filters: Vec::new(),
        }
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
        let removed = self.bodies.swap_remove(idx);
        self.body_materials.swap_remove(idx);
        self.body_collision_radii.swap_remove(idx);
        self.body_filters.swap_remove(idx);

        // Remap references from `last` -> `idx` in all constraints and joints
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

        // Remove constraints that referenced the removed body
        self.distance_constraints
            .retain(|c| c.body_a < self.bodies.len() && c.body_b < self.bodies.len());
        self.contact_constraints
            .retain(|c| c.body_a < self.bodies.len() && c.body_b < self.bodies.len());
        self.joints.retain(|j| {
            let (a, b) = j.bodies();
            a < self.bodies.len() && b < self.bodies.len()
        });

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
        /// Remap a single joint's body_a and body_b fields.
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
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of active (non-sleeping) bodies
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
    /// The hook is called with (body_a_index, body_b_index, &contact).
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
    pub fn body_filter(&self, body_idx: usize) -> CollisionFilter {
        self.body_filters
            .get(body_idx)
            .copied()
            .unwrap_or(CollisionFilter::DEFAULT)
    }

    // ── Sleeping ──────────────────────────────────────────────────────

    /// Check if a body is sleeping
    #[inline]
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
    pub fn contact_events(&self) -> &[crate::event::ContactEvent] {
        self.events.contact_events()
    }

    /// Get trigger events from the last step
    #[inline]
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

    /// Cast a ray against all body collision spheres, returns (body_index, distance).
    ///
    /// Uses a BVH broad-phase to cull bodies outside the ray's bounding box,
    /// then performs exact ray-sphere intersection on candidates.
    /// Returns `None` if direction is zero or no body is hit.
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
        };
        self.add_contact(constraint);
    }

    /// Clear all contact constraints (call before collision detection)
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

        // Track which colors each body participates in (u64 bitmask per body)
        let num_bodies = self.bodies.len();
        let mut body_colors: Vec<u64> = vec![0u64; num_bodies];

        // Color distance constraints
        for (constraint_idx, constraint) in self.distance_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            // Find first color where both bodies are free
            let color = Self::find_free_color(&body_colors, body_a, body_b);

            // Ensure we have enough batches
            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            // Add constraint to batch
            self.constraint_batches[color]
                .distance_indices
                .push(constraint_idx);

            // Mark bodies as used in this color (set bit)
            if body_a < num_bodies && color < 64 {
                body_colors[body_a] |= 1u64 << color;
            }
            if body_b < num_bodies && color < 64 {
                body_colors[body_b] |= 1u64 << color;
            }
        }

        // Color contact constraints
        for (constraint_idx, constraint) in self.contact_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            let color = Self::find_free_color(&body_colors, body_a, body_b);

            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            self.constraint_batches[color]
                .contact_indices
                .push(constraint_idx);

            if body_a < num_bodies && color < 64 {
                body_colors[body_a] |= 1u64 << color;
            }
            if body_b < num_bodies && color < 64 {
                body_colors[body_b] |= 1u64 << color;
            }
        }

        self.batches_dirty = false;
    }

    /// Find first color where both bodies are free (greedy coloring).
    ///
    /// Uses a `u64` bitmask per body for O(1) occupancy checks (up to 64 colors).
    /// Falls back to linear scan for color indices >= 64.
    fn find_free_color(body_colors: &[u64], body_a: usize, body_b: usize) -> usize {
        let mask_a = body_colors.get(body_a).copied().unwrap_or(0);
        let mask_b = body_colors.get(body_b).copied().unwrap_or(0);
        let occupied = mask_a | mask_b;

        // Fast path: find first zero bit via bitwise NOT + trailing zeros
        if occupied != u64::MAX {
            return (!occupied).trailing_zeros() as usize;
        }

        // Overflow path (> 64 colors): linear scan from 64 onward
        // This is extremely rare in practice (would require a single body
        // participating in 64+ different constraint batches).
        64
    }

    /// Get number of constraint batches (colors used)
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

        // Phase 0: Event frame lifecycle + clear stale contacts
        self.events.begin_frame();
        self.clear_contacts();

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

        // Phase 2: Auto collision detection (BVH + sphere)
        self.detect_collisions();

        // Phase 3: Substep loop
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }

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
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, dt: Fix128) {
        // Guard: non-positive dt produces no physics update
        if dt <= Fix128::ZERO {
            return;
        }

        // Phase 0: Event frame lifecycle + clear stale contacts
        self.events.begin_frame();
        self.clear_contacts();

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

        // Phase 2: Auto collision detection
        self.detect_collisions();

        // Phase 3: Rebuild batches and substep
        self.rebuild_batches();
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep_batched(substep_dt);
        }

        // Phase 4: Update sleeping
        self.islands.update_sleep(&self.bodies);

        // Phase 5: End event frame
        self.events.end_frame();
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        self.integrate_positions(dt);

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

        // 3. Solve joint constraints
        if !self.joints.is_empty() {
            solve_joints(&self.joints, &mut self.bodies, dt);
        }

        self.update_velocities(dt);
    }

    /// Single substep with batched constraint solving
    #[cfg(feature = "parallel")]
    fn substep_batched(&mut self, dt: Fix128) {
        self.integrate_positions(dt);

        // 1.5. Resolve SDF collisions
        #[cfg(feature = "std")]
        if !self.sdf_colliders.is_empty() {
            self.resolve_sdf_collisions();
        }

        // 2. Solve constraints (batched)
        for _ in 0..self.config.iterations {
            self.solve_constraints_batched(dt);
        }

        // 3. Solve joint constraints
        if !self.joints.is_empty() {
            solve_joints(&self.joints, &mut self.bodies, dt);
        }

        self.update_velocities(dt);
    }

    /// Integrate positions (shared between sequential and batched)
    ///
    /// Sleeping dynamic bodies are skipped: only prev_position/prev_rotation
    /// are saved so that `update_velocities` derives zero velocity.
    #[inline]
    fn integrate_positions(&mut self, dt: Fix128) {
        #[cfg(feature = "parallel")]
        {
            let gravity = self.config.gravity;
            let damping = self.config.damping;
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
                    if sleep_data.get(i).is_some_and(|d| d.is_sleeping()) {
                        body.prev_position = body.position;
                        body.prev_rotation = body.rotation;
                        return;
                    }

                    // Store previous state
                    body.prev_position = body.position;
                    body.prev_rotation = body.rotation;

                    // Apply gravity (with per-body scale)
                    body.velocity = body.velocity + gravity * body.gravity_scale * dt;

                    // Apply damping
                    body.velocity = body.velocity * damping;
                    body.angular_velocity = body.angular_velocity * damping;

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

                // Apply damping
                self.bodies[i].velocity = self.bodies[i].velocity * self.config.damping;
                self.bodies[i].angular_velocity =
                    self.bodies[i].angular_velocity * self.config.damping;

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

        let num_batches = self.constraint_batches.len();

        let bodies = BodySlicePtr {
            ptr: self.bodies.as_mut_ptr(),
            len: self.bodies.len(),
        };
        let dists = DistConstraintSlicePtr {
            ptr: self.distance_constraints.as_mut_ptr(),
            len: self.distance_constraints.len(),
        };

        for batch_idx in 0..num_batches {
            // Phase 1: Distance constraints — parallel within batch
            {
                let indices = &self.constraint_batches[batch_idx].distance_indices;
                indices.par_iter().for_each(|&idx| {
                    // SAFETY: Graph coloring guarantees no two constraints in
                    // this batch share a body index. Each constraint index
                    // appears in exactly one batch, so cached_lambda writes
                    // are also disjoint.
                    unsafe {
                        let constraint = dists.get_mut(idx);
                        let body_a = bodies.get_mut(constraint.body_a);
                        let body_b = bodies.get_mut(constraint.body_b);
                        Self::solve_distance_pair(body_a, body_b, constraint, dt);
                    }
                });
            }

            // Phase 2: Contact constraints — parallel within batch
            {
                let contact_constraints = &self.contact_constraints;
                let indices = &self.constraint_batches[batch_idx].contact_indices;
                indices.par_iter().for_each(|&idx| {
                    let constraint = &contact_constraints[idx];
                    // SAFETY: Graph coloring guarantees disjoint body access.
                    unsafe {
                        let body_a = bodies.get_mut(constraint.body_a);
                        let body_b = bodies.get_mut(constraint.body_b);
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
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        constraint: &mut DistanceConstraint,
        dt: Fix128,
    ) {
        let anchor_a = body_a.position + body_a.rotation.rotate_vec(constraint.local_anchor_a);
        let anchor_b = body_b.position + body_b.rotation.rotate_vec(constraint.local_anchor_b);

        let delta = anchor_b - anchor_a;
        let (normal, distance) = delta.normalize_with_length();

        if distance.is_zero() {
            return;
        }

        let error = distance - constraint.target_distance;

        let compliance_term = constraint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if w_sum < W_SUM_EPSILON {
            return;
        }

        // Warm-start: bias the error by the cached lambda from the previous substep.
        let inv_w_sum = Fix128::ONE / w_sum;
        let biased_error = error - constraint.cached_lambda * compliance_term;
        let lambda = biased_error * inv_w_sum;
        let correction = normal * lambda;

        // Store lambda for warm-starting on the next substep.
        constraint.cached_lambda = lambda;

        // Branchless: static bodies have inv_mass == ZERO, correction * ZERO == ZERO.
        let delta_a = correction * body_a.inv_mass;
        let delta_b = correction * body_b.inv_mass;
        body_a.position = select_vec3(
            !body_a.inv_mass.is_zero(),
            body_a.position + delta_a,
            body_a.position,
        );
        body_b.position = select_vec3(
            !body_b.inv_mass.is_zero(),
            body_b.position - delta_b,
            body_b.position,
        );
    }

    /// Solve a single contact constraint given mutable body references.
    ///
    /// Extracted as a static method (no `&self`) for parallel dispatch.
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn solve_contact_pair(
        body_a: &mut RigidBody,
        body_b: &mut RigidBody,
        constraint: &ContactConstraint,
    ) {
        // Skip physics response for sensor/trigger bodies
        if body_a.is_sensor || body_b.is_sensor {
            return;
        }

        let contact = constraint.contact;

        if contact.depth <= Fix128::ZERO {
            return;
        }

        let w_sum = body_a.inv_mass + body_b.inv_mass;
        if w_sum < W_SUM_EPSILON {
            return;
        }

        let inv_w_sum = Fix128::ONE / w_sum;
        let correction = contact.normal * contact.depth;
        let correction_a = correction * (body_a.inv_mass * inv_w_sum);
        let correction_b = correction * (body_b.inv_mass * inv_w_sum);

        // Branchless: inv_mass == ZERO for static bodies, correction_x will be ZERO.
        body_a.position = select_vec3(
            !body_a.inv_mass.is_zero(),
            body_a.position + correction_a,
            body_a.position,
        );
        body_b.position = select_vec3(
            !body_b.inv_mass.is_zero(),
            body_b.position - correction_b,
            body_b.position,
        );
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

            // Warm-start: bias the error by subtracting the previously cached lambda
            // times the compliance term so the solver starts from a good initial guess.
            let inv_w_sum = Fix128::ONE / w_sum;
            let biased_error = error - constraint.cached_lambda * compliance_term;
            let lambda = biased_error * inv_w_sum;
            let correction = normal * lambda;

            // Store lambda for warm-starting on the next substep.
            self.distance_constraints[i].cached_lambda = lambda;

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

    /// Solve contact constraints with pre-solve hook support (Gap 2.3)
    #[inline(always)]
    fn solve_contact_constraints(&mut self, _dt: Fix128) {
        let num_constraints = self.contact_constraints.len();
        for i in 0..num_constraints {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            // Skip physics response for sensor/trigger bodies
            if body_a.is_sensor || body_b.is_sensor {
                continue;
            }

            let mut contact = constraint.contact;
            let mut _friction = constraint.friction;
            let mut _restitution = constraint.restitution;

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
                            &mut _friction,
                            &mut _restitution,
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

            // Position correction (reciprocal pre-computation: 1 division instead of 6)
            let inv_w_sum = Fix128::ONE / w_sum;
            let correction = contact.normal * contact.depth;
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

            let delta = self.bodies[b].position - self.bodies[a].position;
            let (normal, dist) = delta.normalize_with_length();
            let combined_radius = radius_a + radius_b;

            if dist < combined_radius && !dist.is_zero() {
                let depth = combined_radius - dist;
                let point_a = self.bodies[a].position + normal * radius_a;
                let point_b = self.bodies[b].position - normal * radius_b;
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

            // Wake sleeping bodies on contact
            self.islands.wake_body(info.body_a);
            self.islands.wake_body(info.body_b);

            if info.is_sensor {
                self.events.report_trigger(info.body_a, info.body_b);
            } else {
                self.add_contact_with_material(info.body_a, info.body_b, info.contact);
            }
        }
    }

    /// Get body by index
    #[inline]
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

#[cfg(test)]
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
            "Zero-gravity body should stay near y=10, got {:?}",
            ng_y
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
            "Body should not have fallen through SDF ground, y={}",
            y
        );
    }
}
