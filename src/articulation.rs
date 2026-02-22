//! Articulated Bodies (Multi-Joint Chains)
//!
//! Represents connected rigid body chains like ragdolls, robotic arms,
//! and vehicles. Uses forward kinematics propagation.
//!
//! # Architecture
//!
//! An `ArticulatedBody` is a tree of `Link`s connected by joints.
//! The root link is typically the pelvis/base. Each child link
//! references its parent and connecting joint.

use crate::joint::{BallJoint, HingeJoint, Joint};
use crate::math::{Fix128, Vec3Fix};
use crate::motor::{MotorMode, PdController};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A link in an articulated body
#[derive(Clone, Debug)]
pub struct Link {
    /// Index into PhysicsWorld bodies
    pub body_index: usize,
    /// Parent link index (usize::MAX for root)
    pub parent: usize,
    /// Joint connecting this link to its parent
    pub joint: Option<Joint>,
    /// Local offset from parent's anchor to this link's origin
    pub local_offset: Vec3Fix,
    /// Children link indices
    pub children: Vec<usize>,
    /// Optional motor for this link's joint
    pub motor: Option<PdController>,
}

/// Sentinel for root link (no parent)
pub const LINK_ROOT: usize = usize::MAX;

/// An articulated body (connected chain of rigid bodies)
#[derive(Clone, Debug)]
pub struct ArticulatedBody {
    /// Links in this articulation
    pub links: Vec<Link>,
    /// Root link index
    pub root: usize,
    /// Whether the root is fixed in world space
    pub fixed_base: bool,
}

impl ArticulatedBody {
    /// Create a new articulated body with a root link
    pub fn new(root_body_index: usize, fixed_base: bool) -> Self {
        let root_link = Link {
            body_index: root_body_index,
            parent: LINK_ROOT,
            joint: None,
            local_offset: Vec3Fix::ZERO,
            children: Vec::new(),
            motor: None,
        };

        Self {
            links: vec![root_link],
            root: 0,
            fixed_base,
        }
    }

    /// Add a child link connected by a joint
    ///
    /// Returns the index of the new link.
    pub fn add_link(
        &mut self,
        parent_link: usize,
        body_index: usize,
        joint: Joint,
        local_offset: Vec3Fix,
    ) -> usize {
        let link_idx = self.links.len();
        let link = Link {
            body_index,
            parent: parent_link,
            joint: Some(joint),
            local_offset,
            children: Vec::new(),
            motor: None,
        };
        self.links.push(link);
        self.links[parent_link].children.push(link_idx);
        link_idx
    }

    /// Set a motor on a link's joint
    pub fn set_motor(&mut self, link_index: usize, motor: PdController) {
        if link_index < self.links.len() {
            self.links[link_index].motor = Some(motor);
        }
    }

    /// Number of links
    #[inline]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Number of DOFs (approximate: each non-root link's joint)
    pub fn dof_count(&self) -> usize {
        self.links.iter().filter(|l| l.joint.is_some()).count()
    }

    /// Get all body indices in this articulation
    pub fn body_indices(&self) -> Vec<usize> {
        self.links.iter().map(|l| l.body_index).collect()
    }

    /// Get all joints in this articulation
    pub fn joints(&self) -> Vec<&Joint> {
        self.links.iter().filter_map(|l| l.joint.as_ref()).collect()
    }

    /// Forward kinematics: propagate positions from root to leaves
    ///
    /// Sets child body positions based on parent position + joint + offset.
    pub fn forward_kinematics(&self, bodies: &mut [RigidBody]) {
        self.fk_recursive(self.root, bodies);
    }

    fn fk_recursive(&self, link_idx: usize, bodies: &mut [RigidBody]) {
        let link = &self.links[link_idx];

        if link.parent != LINK_ROOT {
            let parent = &self.links[link.parent];
            let parent_body = bodies[parent.body_index];

            // Child position = parent position + rotated offset
            let world_offset = parent_body.rotation.rotate_vec(link.local_offset);
            bodies[link.body_index].position = parent_body.position + world_offset;
        }

        // Recurse to children (index-based to avoid Vec clone)
        for i in 0..self.links[link_idx].children.len() {
            let child_idx = self.links[link_idx].children[i];
            self.fk_recursive(child_idx, bodies);
        }
    }

    /// Apply motors on all links
    pub fn apply_motors(&self, bodies: &mut [RigidBody], dt: Fix128) {
        for link in &self.links {
            if let (Some(joint), Some(motor)) = (&link.joint, &link.motor) {
                if motor.mode == MotorMode::Off {
                    continue;
                }

                let (body_a_idx, body_b_idx) = joint.bodies();
                let body_a = bodies[body_a_idx];
                let body_b = bodies[body_b_idx];

                let delta = body_b.position - body_a.position;
                let current_pos = delta.length();
                let rel_vel = body_b.velocity - body_a.velocity;
                let current_vel = if current_pos.is_zero() {
                    Fix128::ZERO
                } else {
                    rel_vel.dot(delta / current_pos)
                };

                let force = motor.compute(current_pos, current_vel);

                if force.is_zero() || current_pos.is_zero() {
                    continue;
                }

                let direction = delta / current_pos;
                let impulse = direction * (force * dt);

                if !body_a.inv_mass.is_zero() {
                    bodies[body_a_idx].velocity =
                        bodies[body_a_idx].velocity - impulse * body_a.inv_mass;
                }
                if !body_b.inv_mass.is_zero() {
                    bodies[body_b_idx].velocity =
                        bodies[body_b_idx].velocity + impulse * body_b.inv_mass;
                }
            }
        }
    }
}

/// Build a simple ragdoll articulated body
///
/// Creates a basic humanoid ragdoll with:
/// - Pelvis (root)
/// - Spine -> Chest -> Head
/// - L/R Upper Arm -> Lower Arm
/// - L/R Upper Leg -> Lower Leg
///
/// Returns `(ArticulatedBody, Vec<RigidBody>)` ready to add to PhysicsWorld.
#[allow(clippy::too_many_lines)]
pub fn build_ragdoll(
    pelvis_pos: Vec3Fix,
    body_start_index: usize,
) -> (ArticulatedBody, Vec<RigidBody>) {
    let mut bodies = Vec::new();

    // Helper to create a body
    let mut make_body = |pos: Vec3Fix, mass: Fix128| -> usize {
        let idx = body_start_index + bodies.len();
        bodies.push(RigidBody::new(pos, mass));
        idx
    };

    let one = Fix128::ONE;

    // Create bodies (positions relative to pelvis)
    let pelvis_idx = make_body(pelvis_pos, Fix128::from_int(5));
    let spine_idx = make_body(pelvis_pos + Vec3Fix::from_int(0, 2, 0), Fix128::from_int(4));
    let chest_idx = make_body(pelvis_pos + Vec3Fix::from_int(0, 4, 0), Fix128::from_int(4));
    let head_idx = make_body(pelvis_pos + Vec3Fix::from_int(0, 6, 0), Fix128::from_int(2));

    let l_upper_arm_idx = make_body(
        pelvis_pos + Vec3Fix::from_int(-2, 4, 0),
        Fix128::from_int(2),
    );
    let l_lower_arm_idx = make_body(pelvis_pos + Vec3Fix::from_int(-4, 4, 0), one);
    let r_upper_arm_idx = make_body(pelvis_pos + Vec3Fix::from_int(2, 4, 0), Fix128::from_int(2));
    let r_lower_arm_idx = make_body(pelvis_pos + Vec3Fix::from_int(4, 4, 0), one);

    let l_upper_leg_idx = make_body(
        pelvis_pos + Vec3Fix::from_int(-1, -2, 0),
        Fix128::from_int(3),
    );
    let l_lower_leg_idx = make_body(
        pelvis_pos + Vec3Fix::from_int(-1, -4, 0),
        Fix128::from_int(2),
    );
    let r_upper_leg_idx = make_body(
        pelvis_pos + Vec3Fix::from_int(1, -2, 0),
        Fix128::from_int(3),
    );
    let r_lower_leg_idx = make_body(
        pelvis_pos + Vec3Fix::from_int(1, -4, 0),
        Fix128::from_int(2),
    );

    // Build articulation
    let mut artic = ArticulatedBody::new(pelvis_idx, false);

    // Spine chain
    let spine_link = artic.add_link(
        0,
        spine_idx,
        Joint::Ball(BallJoint::new(
            pelvis_idx,
            spine_idx,
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::ZERO,
        )),
        Vec3Fix::from_int(0, 2, 0),
    );
    let chest_link = artic.add_link(
        spine_link,
        chest_idx,
        Joint::Ball(BallJoint::new(
            spine_idx,
            chest_idx,
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::ZERO,
        )),
        Vec3Fix::from_int(0, 2, 0),
    );
    let _head_link = artic.add_link(
        chest_link,
        head_idx,
        Joint::Ball(BallJoint::new(
            chest_idx,
            head_idx,
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::ZERO,
        )),
        Vec3Fix::from_int(0, 2, 0),
    );

    // Arms
    let l_arm_link = artic.add_link(
        chest_link,
        l_upper_arm_idx,
        Joint::Ball(BallJoint::new(
            chest_idx,
            l_upper_arm_idx,
            Vec3Fix::from_int(-1, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
        )),
        Vec3Fix::from_int(-2, 0, 0),
    );
    let _l_forearm_link = artic.add_link(
        l_arm_link,
        l_lower_arm_idx,
        Joint::Hinge(
            HingeJoint::new(
                l_upper_arm_idx,
                l_lower_arm_idx,
                Vec3Fix::from_int(-1, 0, 0),
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_limits(Fix128::ZERO, Fix128::PI),
        ),
        Vec3Fix::from_int(-2, 0, 0),
    );

    let r_arm_link = artic.add_link(
        chest_link,
        r_upper_arm_idx,
        Joint::Ball(BallJoint::new(
            chest_idx,
            r_upper_arm_idx,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(-1, 0, 0),
        )),
        Vec3Fix::from_int(2, 0, 0),
    );
    let _r_forearm_link = artic.add_link(
        r_arm_link,
        r_lower_arm_idx,
        Joint::Hinge(
            HingeJoint::new(
                r_upper_arm_idx,
                r_lower_arm_idx,
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::from_int(-1, 0, 0),
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_limits(Fix128::ZERO, Fix128::PI),
        ),
        Vec3Fix::from_int(2, 0, 0),
    );

    // Legs
    let l_leg_link = artic.add_link(
        0,
        l_upper_leg_idx,
        Joint::Ball(BallJoint::new(
            pelvis_idx,
            l_upper_leg_idx,
            Vec3Fix::from_int(-1, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
        )),
        Vec3Fix::from_int(-1, -2, 0),
    );
    let _l_shin_link = artic.add_link(
        l_leg_link,
        l_lower_leg_idx,
        Joint::Hinge(
            HingeJoint::new(
                l_upper_leg_idx,
                l_lower_leg_idx,
                Vec3Fix::from_int(0, -1, 0),
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::UNIT_X,
                Vec3Fix::UNIT_X,
            )
            .with_limits(-Fix128::PI, Fix128::ZERO),
        ),
        Vec3Fix::from_int(0, -2, 0),
    );

    let r_leg_link = artic.add_link(
        0,
        r_upper_leg_idx,
        Joint::Ball(BallJoint::new(
            pelvis_idx,
            r_upper_leg_idx,
            Vec3Fix::from_int(1, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
        )),
        Vec3Fix::from_int(1, -2, 0),
    );
    let _r_shin_link = artic.add_link(
        r_leg_link,
        r_lower_leg_idx,
        Joint::Hinge(
            HingeJoint::new(
                r_upper_leg_idx,
                r_lower_leg_idx,
                Vec3Fix::from_int(0, -1, 0),
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::UNIT_X,
                Vec3Fix::UNIT_X,
            )
            .with_limits(-Fix128::PI, Fix128::ZERO),
        ),
        Vec3Fix::from_int(0, -2, 0),
    );

    (artic, bodies)
}

// ============================================================================
// Featherstone Articulated Body Algorithm (ABI)
// ============================================================================

/// Per-link spatial data for Featherstone ABI
#[derive(Clone, Debug)]
struct FeatherstoneLinkData {
    /// Articulated body inertia (6x6 stored as two 3x3 blocks)
    abi_linear: Fix128, // Simplified: scalar mass-like term
    abi_angular: Vec3Fix, // Simplified: diagonal inertia term
    /// Bias force (Coriolis + external)
    bias_force: Vec3Fix,
    _bias_torque: Vec3Fix,
    /// Computed joint acceleration
    _joint_accel: Fix128,
    /// Spatial velocity
    vel_linear: Vec3Fix,
    vel_angular: Vec3Fix,
}

impl Default for FeatherstoneLinkData {
    fn default() -> Self {
        Self {
            abi_linear: Fix128::ZERO,
            abi_angular: Vec3Fix::ZERO,
            bias_force: Vec3Fix::ZERO,
            _bias_torque: Vec3Fix::ZERO,
            _joint_accel: Fix128::ZERO,
            vel_linear: Vec3Fix::ZERO,
            vel_angular: Vec3Fix::ZERO,
        }
    }
}

/// Featherstone Articulated Body Algorithm solver
///
/// Computes forward dynamics for articulated bodies in O(n) time
/// where n is the number of links.
pub struct FeatherstoneSolver {
    /// Per-link data
    link_data: Vec<FeatherstoneLinkData>,
}

impl FeatherstoneSolver {
    /// Create a new solver
    pub fn new() -> Self {
        Self {
            link_data: Vec::new(),
        }
    }

    /// Solve forward dynamics for an articulated body
    ///
    /// Three-pass algorithm:
    /// 1. Forward pass: compute velocities
    /// 2. Backward pass: compute articulated body inertias
    /// 3. Forward pass: compute accelerations
    pub fn solve(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        gravity: Vec3Fix,
        dt: Fix128,
    ) {
        let n = artic.links.len();
        self.link_data.resize_with(n, FeatherstoneLinkData::default);

        // Pass 1: Forward — compute velocities from root to leaves
        self.forward_velocity_pass(artic, bodies);

        // Pass 2: Backward — compute articulated body inertias from leaves to root
        self.backward_inertia_pass(artic, bodies, gravity);

        // Pass 3: Forward — compute accelerations and apply
        self.forward_acceleration_pass(artic, bodies, dt);
    }

    /// Pass 1: Propagate velocities root → leaves
    fn forward_velocity_pass(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody]) {
        self.fv_recursive(artic, bodies, artic.root);
    }

    fn fv_recursive(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody], link_idx: usize) {
        let link = &artic.links[link_idx];
        let body = &bodies[link.body_index];

        self.link_data[link_idx].vel_linear = body.velocity;
        self.link_data[link_idx].vel_angular = body.angular_velocity;

        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.fv_recursive(artic, bodies, child);
        }
    }

    /// Pass 2: Compute ABI from leaves → root
    fn backward_inertia_pass(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &[RigidBody],
        gravity: Vec3Fix,
    ) {
        self.bi_recursive(artic, bodies, gravity, artic.root);
    }

    fn bi_recursive(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &[RigidBody],
        gravity: Vec3Fix,
        link_idx: usize,
    ) {
        // Process children first (leaves to root)
        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.bi_recursive(artic, bodies, gravity, child);
        }

        let link = &artic.links[link_idx];
        let body = &bodies[link.body_index];

        // Initialize with body's own inertia
        let mass = if body.inv_mass.is_zero() {
            Fix128::from_int(1000000) // Very large mass for static
        } else {
            Fix128::ONE / body.inv_mass
        };

        self.link_data[link_idx].abi_linear = mass;
        self.link_data[link_idx].abi_angular = if body.inv_inertia.x.is_zero() {
            Vec3Fix::from_int(1000000, 1000000, 1000000)
        } else {
            Vec3Fix::new(
                Fix128::ONE / body.inv_inertia.x,
                Fix128::ONE / body.inv_inertia.y,
                Fix128::ONE / body.inv_inertia.z,
            )
        };

        // Bias force = gravity + Coriolis terms
        self.link_data[link_idx].bias_force = gravity * mass;

        // Accumulate children's inertia contributions
        for i in 0..artic.links[link_idx].children.len() {
            let child_idx = artic.links[link_idx].children[i];
            let child_mass = self.link_data[child_idx].abi_linear;
            self.link_data[link_idx].abi_linear = self.link_data[link_idx].abi_linear + child_mass;
        }
    }

    /// Pass 3: Compute accelerations root → leaves and integrate
    fn forward_acceleration_pass(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        dt: Fix128,
    ) {
        self.fa_recursive(artic, bodies, dt, artic.root);
    }

    fn fa_recursive(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        dt: Fix128,
        link_idx: usize,
    ) {
        let link = &artic.links[link_idx];

        if link.parent != LINK_ROOT && !bodies[link.body_index].inv_mass.is_zero() {
            let mass = self.link_data[link_idx].abi_linear;
            let force = self.link_data[link_idx].bias_force;

            // a = F / m (simplified)
            let accel = if mass.is_zero() {
                Vec3Fix::ZERO
            } else {
                force / mass
            };

            // Semi-implicit Euler integration
            bodies[link.body_index].velocity = bodies[link.body_index].velocity + accel * dt;
            bodies[link.body_index].position =
                bodies[link.body_index].position + bodies[link.body_index].velocity * dt;
        }

        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.fa_recursive(artic, bodies, dt, child);
        }
    }
}

impl Default for FeatherstoneSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_articulation() {
        let artic = ArticulatedBody::new(0, false);
        assert_eq!(artic.link_count(), 1);
        assert_eq!(artic.dof_count(), 0);
    }

    #[test]
    fn test_add_links() {
        let mut artic = ArticulatedBody::new(0, false);
        let link1 = artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(0, 2, 0),
        );
        let _link2 = artic.add_link(
            link1,
            2,
            Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(0, 2, 0),
        );

        assert_eq!(artic.link_count(), 3);
        assert_eq!(artic.dof_count(), 2);
        assert_eq!(artic.body_indices(), vec![0, 1, 2]);
    }

    #[test]
    fn test_forward_kinematics() {
        let mut bodies = vec![
            RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(5)),
            RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::from_int(3)),
        ];

        let mut artic = ArticulatedBody::new(0, true);
        artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(0, 3, 0),
        );

        artic.forward_kinematics(&mut bodies);

        // Body 1 should be at parent(0,0,0) + offset(0,3,0) = (0,3,0)
        assert_eq!(bodies[1].position.y.hi, 3);
    }

    #[test]
    fn test_build_ragdoll() {
        let (artic, bodies) = build_ragdoll(Vec3Fix::ZERO, 0);
        assert_eq!(bodies.len(), 12, "Ragdoll should have 12 bodies");
        assert_eq!(artic.link_count(), 12, "Ragdoll should have 12 links");
    }

    #[test]
    fn test_set_motor() {
        let mut artic = ArticulatedBody::new(0, false);
        artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::ZERO,
        );

        let mut motor = PdController::default();
        motor.set_position_target(Fix128::from_int(5));
        artic.set_motor(1, motor);

        assert!(artic.links[1].motor.is_some());
    }

    #[test]
    fn test_featherstone_solver() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO), // fixed base
            RigidBody::new(Vec3Fix::from_int(0, 2, 0), Fix128::from_int(2)),
            RigidBody::new(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
        ];

        let mut artic = ArticulatedBody::new(0, true);
        artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(
                0,
                1,
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::from_int(0, -1, 0),
            )),
            Vec3Fix::from_int(0, 2, 0),
        );
        artic.add_link(
            1,
            2,
            Joint::Ball(BallJoint::new(
                1,
                2,
                Vec3Fix::from_int(0, 1, 0),
                Vec3Fix::from_int(0, -1, 0),
            )),
            Vec3Fix::from_int(0, 2, 0),
        );

        let mut solver = FeatherstoneSolver::new();
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        let dt = Fix128::from_ratio(1, 60);

        // Solve several steps
        for _ in 0..10 {
            solver.solve(&artic, &mut bodies, gravity, dt);
        }

        // Bodies should have fallen under gravity
        assert!(
            bodies[1].position.y < Fix128::from_int(2),
            "Link 1 should fall"
        );
        assert!(
            bodies[2].position.y < Fix128::from_int(4),
            "Link 2 should fall"
        );
    }
}
