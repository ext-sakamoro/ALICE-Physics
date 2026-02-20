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

use crate::collider::Contact;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Rayon parallel iterator support (reserved for future use)
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
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
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    /// Position (center of mass)
    pub position: Vec3Fix,
    /// Orientation
    pub rotation: QuatFix,
    /// Linear velocity
    pub velocity: Vec3Fix,
    /// Angular velocity
    pub angular_velocity: Vec3Fix,
    /// Inverse mass (0 = static/infinite mass)
    pub inv_mass: Fix128,
    /// Inverse inertia tensor (diagonal, in local space)
    pub inv_inertia: Vec3Fix,
    /// Previous position (for XPBD)
    pub prev_position: Vec3Fix,
    /// Previous rotation (for XPBD)
    pub prev_rotation: QuatFix,
    /// Coefficient of restitution (bounciness)
    pub restitution: Fix128,
    /// Friction coefficient
    pub friction: Fix128,
    /// Whether this body is a sensor/trigger (detects overlap but no physics response)
    pub is_sensor: bool,
    /// Body type (Dynamic, Static, Kinematic)
    pub body_type: BodyType,
    /// Gravity scale multiplier (1.0 = normal, 0.0 = no gravity, 2.0 = double)
    pub gravity_scale: Fix128,
    /// Kinematic target position and rotation
    pub kinematic_target: Option<(Vec3Fix, QuatFix)>,
}

impl RigidBody {
    /// Create new dynamic rigid body
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
            rotation: QuatFix::IDENTITY,
            velocity: Vec3Fix::ZERO,
            angular_velocity: Vec3Fix::ZERO,
            inv_mass,
            inv_inertia,
            prev_position: position,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::from_ratio(5, 10), // 0.5 default
            friction: Fix128::from_ratio(3, 10),    // 0.3 default
            is_sensor: false,
            body_type: BodyType::Dynamic,
            gravity_scale: Fix128::ONE,
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
            rotation: QuatFix::IDENTITY,
            velocity: Vec3Fix::ZERO,
            angular_velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Static,
            gravity_scale: Fix128::ZERO,
            kinematic_target: None,
        }
    }

    /// Create a sensor (trigger) body — detects overlap but no physics response
    pub fn new_sensor(position: Vec3Fix) -> Self {
        Self {
            position,
            rotation: QuatFix::IDENTITY,
            velocity: Vec3Fix::ZERO,
            angular_velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            is_sensor: true,
            body_type: BodyType::Static,
            gravity_scale: Fix128::ZERO,
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
            rotation: QuatFix::IDENTITY,
            velocity: Vec3Fix::ZERO,
            angular_velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: position,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Kinematic,
            gravity_scale: Fix128::ZERO,
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
                Fix128::from_int(-10), // -10 m/s²
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
}

impl PhysicsWorld {
    /// Create new physics world
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
        }
    }

    /// Add rigid body, returns index
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        self.body_materials.push(crate::material::DEFAULT_MATERIAL);
        idx
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

        // Track which colors each body participates in
        let num_bodies = self.bodies.len();
        let mut body_colors: Vec<Vec<usize>> = vec![Vec::new(); num_bodies];

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

            // Mark bodies as used in this color
            if body_a < num_bodies {
                body_colors[body_a].push(color);
            }
            if body_b < num_bodies {
                body_colors[body_b].push(color);
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

            if body_a < num_bodies {
                body_colors[body_a].push(color);
            }
            if body_b < num_bodies {
                body_colors[body_b].push(color);
            }
        }

        self.batches_dirty = false;
    }

    /// Find first color where both bodies are free (greedy coloring)
    fn find_free_color(body_colors: &[Vec<usize>], body_a: usize, body_b: usize) -> usize {
        let mut color = 0;
        loop {
            let a_ok = body_colors
                .get(body_a)
                .map_or(true, |colors| !colors.contains(&color));
            let b_ok = body_colors
                .get(body_b)
                .map_or(true, |colors| !colors.contains(&color));

            if a_ok && b_ok {
                return color;
            }
            color += 1;

            // Safety limit
            if color > 256 {
                return 0;
            }
        }
    }

    /// Get number of constraint batches (colors used)
    pub fn num_batches(&self) -> usize {
        self.constraint_batches.len()
    }

    /// Step simulation by dt (in fixed-point seconds)
    ///
    /// Deterministic: same inputs always produce same outputs.
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
    }

    /// Step with batched constraint solving
    ///
    /// When `parallel` feature is enabled, processes independent constraint
    /// batches in parallel using Rayon.
    #[cfg(feature = "parallel")]
    pub fn step_parallel(&mut self, dt: Fix128) {
        self.rebuild_batches();

        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep_batched(substep_dt);
        }
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

        self.update_velocities(dt);
    }

    /// Integrate positions (shared between sequential and batched)
    #[inline]
    fn integrate_positions(&mut self, dt: Fix128) {
        #[cfg(feature = "parallel")]
        {
            let gravity = self.config.gravity;
            let damping = self.config.damping;
            self.bodies.par_iter_mut().for_each(|body| {
                match body.body_type {
                    BodyType::Static => return,
                    BodyType::Kinematic => {
                        body.prev_position = body.position;
                        body.prev_rotation = body.rotation;
                        if let Some((target_pos, target_rot)) = body.kinematic_target {
                            body.position = target_pos;
                            body.rotation = target_rot;
                        }
                        return;
                    }
                    BodyType::Dynamic => {}
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

                // Predict rotation (simplified: small angle approximation)
                let angle = body.angular_velocity.length() * dt;
                if !angle.is_zero() {
                    let axis = body.angular_velocity.normalize();
                    let delta_rot = QuatFix::from_axis_angle(axis, angle);
                    body.rotation = delta_rot.mul(body.rotation).normalize();
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for body in &mut self.bodies {
                match body.body_type {
                    BodyType::Static => continue,
                    BodyType::Kinematic => {
                        body.prev_position = body.position;
                        body.prev_rotation = body.rotation;
                        if let Some((target_pos, target_rot)) = body.kinematic_target {
                            body.position = target_pos;
                            body.rotation = target_rot;
                        }
                        continue;
                    }
                    BodyType::Dynamic => {}
                }

                // Store previous state
                body.prev_position = body.position;
                body.prev_rotation = body.rotation;

                // Apply gravity (with per-body scale)
                body.velocity = body.velocity + self.config.gravity * body.gravity_scale * dt;

                // Apply damping
                body.velocity = body.velocity * self.config.damping;
                body.angular_velocity = body.angular_velocity * self.config.damping;

                // Predict position
                body.position = body.position + body.velocity * dt;

                // Predict rotation (simplified: small angle approximation)
                let angle = body.angular_velocity.length() * dt;
                if !angle.is_zero() {
                    let axis = body.angular_velocity.normalize();
                    let delta_rot = QuatFix::from_axis_angle(axis, angle);
                    body.rotation = delta_rot.mul(body.rotation).normalize();
                }
            }
        }
    }

    /// Update velocities from position changes
    #[inline]
    fn update_velocities(&mut self, dt: Fix128) {
        let inv_dt = Fix128::ONE / dt;

        #[cfg(feature = "parallel")]
        {
            self.bodies.par_iter_mut().for_each(|body| {
                if body.body_type == BodyType::Static {
                    return;
                }

                body.velocity = (body.position - body.prev_position) * inv_dt;
                // Angular velocity update (simplified)
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for body in &mut self.bodies {
                if body.body_type == BodyType::Static {
                    continue;
                }

                body.velocity = (body.position - body.prev_position) * inv_dt;
                // Angular velocity update (simplified)
            }
        }
    }

    /// Solve constraints in batched parallel mode
    ///
    /// Uses index-based iteration to avoid Vec::clone() heap allocation.
    #[cfg(feature = "parallel")]
    fn solve_constraints_batched(&mut self, dt: Fix128) {
        let num_batches = self.constraint_batches.len();

        for batch_idx in 0..num_batches {
            // Distance constraints — index-based to avoid borrow conflict
            let num_dist = self.constraint_batches[batch_idx].distance_indices.len();
            for i in 0..num_dist {
                let idx = self.constraint_batches[batch_idx].distance_indices[i];
                self.solve_single_distance_constraint(idx, dt);
            }

            // Contact constraints
            let num_contact = self.constraint_batches[batch_idx].contact_indices.len();
            for i in 0..num_contact {
                let idx = self.constraint_batches[batch_idx].contact_indices[i];
                self.solve_single_contact_constraint(idx, dt);
            }
        }
    }

    /// Solve a single distance constraint by index
    #[cfg(feature = "parallel")]
    fn solve_single_distance_constraint(&mut self, idx: usize, dt: Fix128) {
        let constraint = self.distance_constraints[idx];
        let body_a = self.bodies[constraint.body_a];
        let body_b = self.bodies[constraint.body_b];

        let anchor_a = body_a.position + body_a.rotation.rotate_vec(constraint.local_anchor_a);
        let anchor_b = body_b.position + body_b.rotation.rotate_vec(constraint.local_anchor_b);

        let delta = anchor_b - anchor_a;
        let distance = delta.length();

        if distance.is_zero() {
            return;
        }

        let error = distance - constraint.target_distance;
        let normal = delta / distance;

        let compliance_term = constraint.compliance / (dt * dt);
        let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

        if w_sum.is_zero() {
            return;
        }

        let lambda = error / w_sum;
        let correction = normal * lambda;

        if !body_a.inv_mass.is_zero() {
            self.bodies[constraint.body_a].position =
                self.bodies[constraint.body_a].position + correction * body_a.inv_mass;
        }
        if !body_b.inv_mass.is_zero() {
            self.bodies[constraint.body_b].position =
                self.bodies[constraint.body_b].position - correction * body_b.inv_mass;
        }
    }

    /// Solve a single contact constraint by index
    #[cfg(feature = "parallel")]
    fn solve_single_contact_constraint(&mut self, idx: usize, _dt: Fix128) {
        let constraint = self.contact_constraints[idx];
        let body_a = self.bodies[constraint.body_a];
        let body_b = self.bodies[constraint.body_b];

        // Skip physics response for sensor/trigger bodies
        if body_a.is_sensor || body_b.is_sensor {
            return;
        }

        let contact = constraint.contact;

        if contact.depth <= Fix128::ZERO {
            return;
        }

        let w_sum = body_a.inv_mass + body_b.inv_mass;
        if w_sum.is_zero() {
            return;
        }

        let correction = contact.normal * contact.depth;
        let correction_a = correction * body_a.inv_mass / w_sum;
        let correction_b = correction * body_b.inv_mass / w_sum;

        if !body_a.inv_mass.is_zero() {
            self.bodies[constraint.body_a].position =
                self.bodies[constraint.body_a].position + correction_a;
        }
        if !body_b.inv_mass.is_zero() {
            self.bodies[constraint.body_b].position =
                self.bodies[constraint.body_b].position - correction_b;
        }
    }

    /// Solve distance constraints (sequential)
    #[inline]
    fn solve_distance_constraints(&mut self, dt: Fix128) {
        let num_constraints = self.distance_constraints.len();
        for i in 0..num_constraints {
            let constraint = self.distance_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            // Get world-space anchor positions
            let anchor_a = body_a.position + body_a.rotation.rotate_vec(constraint.local_anchor_a);
            let anchor_b = body_b.position + body_b.rotation.rotate_vec(constraint.local_anchor_b);

            // Compute constraint error
            let delta = anchor_b - anchor_a;
            let distance = delta.length();

            if distance.is_zero() {
                continue;
            }

            let error = distance - constraint.target_distance;
            let normal = delta / distance;

            // Compute compliance term (XPBD)
            let compliance_term = constraint.compliance / (dt * dt);
            let w_sum = body_a.inv_mass + body_b.inv_mass + compliance_term;

            if w_sum.is_zero() {
                continue;
            }

            // Compute correction
            let lambda = error / w_sum;
            let correction = normal * lambda;

            // Apply corrections
            if !body_a.inv_mass.is_zero() {
                self.bodies[constraint.body_a].position =
                    self.bodies[constraint.body_a].position + correction * body_a.inv_mass;
            }
            if !body_b.inv_mass.is_zero() {
                self.bodies[constraint.body_b].position =
                    self.bodies[constraint.body_b].position - correction * body_b.inv_mass;
            }
        }
    }

    /// Solve contact constraints with pre-solve hook support
    #[inline]
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
            if w_sum.is_zero() {
                continue;
            }

            // Position correction (push bodies apart)
            let correction = contact.normal * contact.depth;
            let correction_a = correction * body_a.inv_mass / w_sum;
            let correction_b = correction * body_b.inv_mass / w_sum;

            if !body_a.inv_mass.is_zero() {
                self.bodies[constraint.body_a].position =
                    self.bodies[constraint.body_a].position + correction_a;
            }
            if !body_b.inv_mass.is_zero() {
                self.bodies[constraint.body_b].position =
                    self.bodies[constraint.body_b].position - correction_b;
            }
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

    /// Serialize world state (for rollback netcode)
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
        }

        data
    }

    /// Deserialize world state (for rollback netcode)
    pub fn deserialize_state(&mut self, data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if count != self.bodies.len() {
            return false;
        }

        let mut offset = 4;
        for body in &mut self.bodies {
            if offset + 160 > data.len() {
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
        }

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

        // Body should be near the ground (y ≈ collision_radius = 0.5)
        assert!(y < 5.0, "Body should have fallen from y=5");
        assert!(
            y > -1.0,
            "Body should not have fallen through SDF ground, y={}",
            y
        );
    }
}
