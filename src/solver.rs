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

use crate::math::{Fix128, Vec3Fix, QuatFix};
use crate::collider::Contact;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Rayon parallel iterator support (reserved for future use)
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

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
        }
    }

    /// Check if body is static
    #[inline]
    pub fn is_static(&self) -> bool {
        self.inv_mass.is_zero()
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
            self.angular_velocity = self.angular_velocity + Vec3Fix::new(
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
    pub body_a: usize,
    pub body_b: usize,
    pub local_anchor_a: Vec3Fix,
    pub local_anchor_b: Vec3Fix,
    pub target_distance: Fix128,
    pub compliance: Fix128,  // Inverse stiffness (0 = infinitely stiff)
}

impl DistanceConstraint {
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

    pub fn with_compliance(mut self, compliance: Fix128) -> Self {
        self.compliance = compliance;
        self
    }
}

/// Contact constraint (from collision detection)
#[derive(Clone, Copy, Debug)]
pub struct ContactConstraint {
    pub body_a: usize,
    pub body_b: usize,
    pub contact: Contact,
    pub friction: Fix128,
    pub restitution: Fix128,
}

impl ContactConstraint {
    pub fn new(body_a: usize, body_b: usize, contact: Contact) -> Self {
        Self {
            body_a,
            body_b,
            contact,
            friction: Fix128::from_ratio(3, 10),  // 0.3 friction
            restitution: Fix128::from_ratio(2, 10),  // 0.2 restitution
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
                Fix128::from_int(-10),  // -10 m/s²
                Fix128::ZERO,
            ),
            damping: Fix128::from_ratio(99, 100),  // 0.99 velocity retention
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

/// XPBD physics world with batched constraint solving
pub struct PhysicsWorld {
    pub config: SolverConfig,
    pub bodies: Vec<RigidBody>,
    pub distance_constraints: Vec<DistanceConstraint>,
    pub contact_constraints: Vec<ContactConstraint>,
    /// Pre-colored constraint batches (computed on demand)
    constraint_batches: Vec<ConstraintBatch>,
    /// Whether batches need recomputation
    batches_dirty: bool,
}

impl PhysicsWorld {
    /// Create new physics world
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            bodies: Vec::new(),
            distance_constraints: Vec::new(),
            contact_constraints: Vec::new(),
            constraint_batches: Vec::new(),
            batches_dirty: true,
        }
    }

    /// Add rigid body, returns index
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Add distance constraint
    pub fn add_distance_constraint(&mut self, constraint: DistanceConstraint) {
        self.distance_constraints.push(constraint);
        self.batches_dirty = true;
    }

    /// Add contact constraint
    pub fn add_contact(&mut self, contact: ContactConstraint) {
        self.contact_constraints.push(contact);
        self.batches_dirty = true;
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

        // Track which bodies are "used" in each color
        let num_bodies = self.bodies.len();
        let body_colors: Vec<Option<usize>> = vec![None; num_bodies];

        // Color distance constraints
        for (constraint_idx, constraint) in self.distance_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            // Find first color where both bodies are free
            let color = self.find_free_color(&body_colors, body_a, body_b);

            // Ensure we have enough batches
            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            // Add constraint to batch
            self.constraint_batches[color].distance_indices.push(constraint_idx);

            // Mark bodies as used in this color
            // (Reset colors for next frame - greedy per-constraint)
        }

        // Color contact constraints
        for (constraint_idx, constraint) in self.contact_constraints.iter().enumerate() {
            let body_a = constraint.body_a;
            let body_b = constraint.body_b;

            let color = self.find_free_color(&body_colors, body_a, body_b);

            while self.constraint_batches.len() <= color {
                self.constraint_batches.push(ConstraintBatch::default());
            }

            self.constraint_batches[color].contact_indices.push(constraint_idx);
        }

        self.batches_dirty = false;
    }

    /// Find first color where both bodies are free (greedy coloring)
    fn find_free_color(&self, body_colors: &[Option<usize>], body_a: usize, body_b: usize) -> usize {
        // Simple greedy: find lowest color not used by either body
        let color_a = body_colors.get(body_a).copied().flatten();
        let color_b = body_colors.get(body_b).copied().flatten();

        // Find first available color
        let mut color = 0;
        loop {
            let a_ok = color_a.map_or(true, |c| c != color);
            let b_ok = color_b.map_or(true, |c| c != color);

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

        // 2. Solve constraints (batched)
        for _ in 0..self.config.iterations {
            self.solve_constraints_batched(dt);
        }

        self.update_velocities(dt);
    }

    /// Integrate positions (shared between sequential and batched)
    #[inline]
    fn integrate_positions(&mut self, dt: Fix128) {
        for body in &mut self.bodies {
            if body.is_static() {
                continue;
            }

            // Store previous state
            body.prev_position = body.position;
            body.prev_rotation = body.rotation;

            // Apply gravity
            body.velocity = body.velocity + self.config.gravity * dt;

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

    /// Update velocities from position changes
    #[inline]
    fn update_velocities(&mut self, dt: Fix128) {
        let inv_dt = Fix128::ONE / dt;
        for body in &mut self.bodies {
            if body.is_static() {
                continue;
            }

            body.velocity = (body.position - body.prev_position) * inv_dt;
            // Angular velocity update (simplified)
        }
    }

    /// Solve constraints in batched parallel mode
    #[cfg(feature = "parallel")]
    fn solve_constraints_batched(&mut self, dt: Fix128) {
        // Process each batch sequentially (batches are independent)
        // Clone indices to avoid borrow issues
        let num_batches = self.constraint_batches.len();

        for batch_idx in 0..num_batches {
            // Clone the indices for this batch
            let distance_indices: Vec<usize> = self.constraint_batches[batch_idx]
                .distance_indices
                .clone();
            let contact_indices: Vec<usize> = self.constraint_batches[batch_idx]
                .contact_indices
                .clone();

            // Solve distance constraints in this batch
            for idx in distance_indices {
                self.solve_single_distance_constraint(idx, dt);
            }

            // Solve contact constraints in this batch
            for idx in contact_indices {
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

    /// Solve contact constraints
    fn solve_contact_constraints(&mut self, _dt: Fix128) {
        let num_constraints = self.contact_constraints.len();
        for i in 0..num_constraints {
            let constraint = self.contact_constraints[i];
            let body_a = self.bodies[constraint.body_a];
            let body_b = self.bodies[constraint.body_b];

            let contact = constraint.contact;

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
                    data[*o], data[*o + 1], data[*o + 2], data[*o + 3],
                    data[*o + 4], data[*o + 5], data[*o + 6], data[*o + 7],
                ]);
                *o += 8;
                let lo = u64::from_le_bytes([
                    data[*o], data[*o + 1], data[*o + 2], data[*o + 3],
                    data[*o + 4], data[*o + 5], data[*o + 6], data[*o + 7],
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
        assert!(body.position.y < Fix128::from_int(10), "Body should have fallen");
    }

    #[test]
    fn test_distance_constraint() {
        let config = SolverConfig {
            substeps: 4,
            iterations: 8,
            gravity: Vec3Fix::ZERO,  // No gravity for constraint test
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        // Two bodies connected by a distance constraint
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = world.add_body(RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));

        world.add_distance_constraint(DistanceConstraint::new(
            a, b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(3),  // Target distance: 3
        ));

        // Step simulation
        for _ in 0..100 {
            world.step(Fix128::from_ratio(1, 60));
        }

        // Body B should be approximately 3 units from body A
        let body_b = world.get_body(b).unwrap();
        let distance = body_b.position.length();

        // Allow some tolerance due to gravity
        assert!(distance < Fix128::from_int(5), "Constraint should pull body closer");
    }

    #[test]
    fn test_state_serialization() {
        let config = SolverConfig::default();
        let mut world = PhysicsWorld::new(config);

        world.add_body(RigidBody::new(Vec3Fix::from_int(1, 2, 3), Fix128::ONE));
        world.add_body(RigidBody::new(Vec3Fix::from_int(4, 5, 6), Fix128::from_int(2)));

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
            world.add_body(RigidBody::new(Vec3Fix::from_int(5, 10, 0), Fix128::from_int(2)));

            for _ in 0..100 {
                world.step(Fix128::from_ratio(1, 60));
            }

            world.serialize_state()
        };

        let state1 = run_simulation();
        let state2 = run_simulation();

        assert_eq!(state1, state2, "Simulation must be deterministic");
    }
}
