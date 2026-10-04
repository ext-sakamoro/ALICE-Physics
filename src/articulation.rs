//! Articulated Bodies (Multi-Joint Chains)
//!
//! Represents connected rigid body chains like ragdolls, robotic arms,
//! and vehicles.
//!
//! # Architecture
//!
//! An `ArticulatedBody` is a tree of `Link`s connected by joints.
//! The root link is typically the pelvis/base. Each child link
//! references its parent and connecting joint.
//!
//! [`ArticulatedBody::forward_kinematics`] propagates positions down the tree
//! from a pose. Forward *dynamics* — turning gravity and the joint constraints
//! into accelerations — is [`FeatherstoneSolver`], which reads `Link::joint` and
//! is documented at its own definition.

use crate::joint::{BallJoint, D6Motion, HingeJoint, Joint};
use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
use crate::motor::{MotorMode, PdController};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A link in an articulated body
#[derive(Clone, Debug)]
pub struct Link {
    /// Index into `PhysicsWorld` bodies
    pub body_index: usize,
    /// Parent link index (`usize::MAX` for root)
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
    #[must_use]
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
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    /// Number of DOFs (approximate: each non-root link's joint)
    #[must_use]
    pub fn dof_count(&self) -> usize {
        self.links.iter().filter(|l| l.joint.is_some()).count()
    }

    /// Get all body indices in this articulation
    #[must_use]
    pub fn body_indices(&self) -> Vec<usize> {
        self.links.iter().map(|l| l.body_index).collect()
    }

    /// Get all joints in this articulation
    #[must_use]
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
    ///
    /// # Claims
    /// - A `Joint::Hinge` link's generalised coordinate is the relative twist angle
    ///   about the hinge axis (`crate::joint::compute_twist_angle`), the same
    ///   quantity `solve_hinge_joint`'s own angle-limit step reads, and the motor
    ///   drives the bodies' `angular_velocity`, not their linear `velocity`.
    /// - Every other joint type keeps the centre-to-centre distance as its
    ///   generalised coordinate (unchanged from before this method gained the
    ///   hinge case above), since only the hinge's single rotational DOF has an
    ///   established angle convention already defined elsewhere in this crate.
    pub fn apply_motors(&self, bodies: &mut [RigidBody], dt: Fix128) {
        for link in &self.links {
            if let (Some(joint), Some(motor)) = (&link.joint, &link.motor) {
                if motor.mode == MotorMode::Off {
                    continue;
                }

                let (body_a_idx, body_b_idx) = joint.bodies();
                let body_a = bodies[body_a_idx];
                let body_b = bodies[body_b_idx];

                if let Joint::Hinge(hinge) = joint {
                    let axis = body_a.rotation.rotate_vec(hinge.local_axis_a);
                    let rel_quat = body_b.rotation.mul(body_a.rotation.conjugate());
                    let current_pos = crate::joint::compute_twist_angle(rel_quat, axis);
                    let current_vel = (body_b.angular_velocity - body_a.angular_velocity).dot(axis);

                    let torque = motor.compute(current_pos, current_vel);
                    if torque.is_zero() {
                        continue;
                    }

                    let angular_impulse = axis * (torque * dt);
                    if !body_a.inv_mass.is_zero() {
                        bodies[body_a_idx].angular_velocity = bodies[body_a_idx].angular_velocity
                            - body_a.world_inv_inertia_apply(angular_impulse);
                    }
                    if !body_b.inv_mass.is_zero() {
                        bodies[body_b_idx].angular_velocity = bodies[body_b_idx].angular_velocity
                            + body_b.world_inv_inertia_apply(angular_impulse);
                    }
                    continue;
                }

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
/// Returns `(ArticulatedBody, Vec<RigidBody>)` ready to add to `PhysicsWorld`.
#[allow(clippy::too_many_lines)]
#[must_use]
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
// Featherstone Articulated Body Algorithm
// ============================================================================
//
// The solver below is the Articulated Body Algorithm (Featherstone, *Rigid Body
// Dynamics Algorithms*, ch. 7) written in **absolute (world) coordinates**: every
// spatial quantity is expressed in the inertial frame with its moment taken about
// the world origin, so the Plücker transform between a parent and its child is the
// identity and no per-link coordinate frame has to be maintained. That choice is
// what lets the algorithm run directly on this crate's maximal-coordinate
// `RigidBody` state (world position, world velocity, world orientation) without
// introducing a parallel set of minimal coordinates.
//
// Spatial vectors are ordered `[angular; linear]`. A motion vector is
// `[ω; v_O]`, where `v_O` is the velocity of the body-fixed point currently at
// the origin; a force vector is `[n_O; f]`. The pairing `Sᵀf = ω·n_O + v_O·f`
// is power, which is why [`SpatialVec::pair`] serves for both `SᵀU` and `Sᵀp`.
//
// # Gravity
//
// Gravity is applied by Featherstone's base-acceleration substitution (RBDA
// §7.3) rather than as a per-link external wrench. For a rigid body,
// `I·[0; g]` is exactly the gravitational wrench `[m c×g; m g]`, so writing
// `a = ã + a_g` turns the equations of motion into the same equations with no
// gravity and a base acceleration shifted by `-a_g`. Two things follow, and the
// second is the reason the substitution is used here rather than the wrench:
//
// 1. A chain in free fall has `p^A = 0` everywhere, so every joint acceleration
//    is `D⁻¹·0`, which is exactly zero in `Fix128` no matter what `D⁻¹` rounded
//    to. Free fall therefore comes out bit-exact instead of within a few ulp.
// 2. The same is true of any configuration in static equilibrium.
//
// The substitution assumes a uniform field, so `RigidBody::gravity_scale` is not
// consulted by this solver — a per-body scale is not a uniform field.

/// Maximum number of degrees of freedom one joint can contribute.
const MAX_JOINT_DOF: usize = 6;

/// Stand-in for an infinite mass or rotational inertia.
///
/// A zero inverse mass (or inverse inertia) means "cannot accelerate" —
/// infinite inertia. Infinity is not representable in [`Fix128`], so the
/// articulated-body inertia substitutes a large finite value: big enough that
/// the resulting acceleration is negligible, small enough that its products stay
/// far from the [`Fix128`] range. Such a body is also skipped by the integrator,
/// so the stand-in only ever affects how much its neighbours can move it.
const INFINITE_INERTIA_STANDIN: i64 = 1_000_000;

/// A spatial (6D) vector, ordered `[angular; linear]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpatialVec {
    /// Angular part: `ω` for a motion vector, the moment about the origin for a
    /// force vector.
    ang: Vec3Fix,
    /// Linear part: `v_O` for a motion vector, the resultant for a force vector.
    lin: Vec3Fix,
}

impl SpatialVec {
    const ZERO: Self = Self {
        ang: Vec3Fix::ZERO,
        lin: Vec3Fix::ZERO,
    };

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            ang: self.ang + rhs.ang,
            lin: self.lin + rhs.lin,
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            ang: self.ang - rhs.ang,
            lin: self.lin - rhs.lin,
        }
    }

    #[inline]
    fn scaled(self, s: Fix128) -> Self {
        Self {
            ang: self.ang * s,
            lin: self.lin * s,
        }
    }

    /// The motion/force pairing `ω·n_O + v_O·f`, i.e. power.
    #[inline]
    fn pair(self, force: Self) -> Fix128 {
        self.ang.dot(force.ang) + self.lin.dot(force.lin)
    }

    /// Spatial motion cross product `self × m`.
    #[inline]
    fn cross_motion(self, m: Self) -> Self {
        Self {
            ang: self.ang.cross(m.ang),
            lin: self.ang.cross(m.lin) + self.lin.cross(m.ang),
        }
    }

    /// Spatial force cross product `self ×* f`.
    #[inline]
    fn cross_force(self, f: Self) -> Self {
        Self {
            ang: self.ang.cross(f.ang) + self.lin.cross(f.lin),
            lin: self.ang.cross(f.lin),
        }
    }
}

/// A spatial (6x6) matrix, stored as four 3x3 blocks in the `[angular; linear]`
/// ordering: `[[aa, al], [la, ll]]`.
#[derive(Clone, Copy, Debug)]
struct SpatialMat {
    /// Angular row block, angular column block.
    aa: Mat3Fix,
    /// Angular row block, linear column block.
    al: Mat3Fix,
    /// Linear row block, angular column block.
    la: Mat3Fix,
    /// Linear row block, linear column block.
    ll: Mat3Fix,
}

impl SpatialMat {
    const ZERO: Self = Self {
        aa: Mat3Fix::ZERO,
        al: Mat3Fix::ZERO,
        la: Mat3Fix::ZERO,
        ll: Mat3Fix::ZERO,
    };

    #[inline]
    fn mul_vec(self, v: SpatialVec) -> SpatialVec {
        SpatialVec {
            ang: self.aa.mul_vec(v.ang) + self.al.mul_vec(v.lin),
            lin: self.la.mul_vec(v.ang) + self.ll.mul_vec(v.lin),
        }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            aa: mat_add(self.aa, rhs.aa),
            al: mat_add(self.al, rhs.al),
            la: mat_add(self.la, rhs.la),
            ll: mat_add(self.ll, rhs.ll),
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            aa: mat_sub(self.aa, rhs.aa),
            al: mat_sub(self.al, rhs.al),
            la: mat_sub(self.la, rhs.la),
            ll: mat_sub(self.ll, rhs.ll),
        }
    }

    /// The outer product `x yᵀ` of a spatial column and a spatial row.
    #[inline]
    fn outer(x: SpatialVec, y: SpatialVec) -> Self {
        Self {
            aa: outer3(x.ang, y.ang),
            al: outer3(x.ang, y.lin),
            la: outer3(x.lin, y.ang),
            ll: outer3(x.lin, y.lin),
        }
    }
}

#[inline]
fn mat_add(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(a.col0 + b.col0, a.col1 + b.col1, a.col2 + b.col2)
}

#[inline]
fn mat_sub(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(a.col0 - b.col0, a.col1 - b.col1, a.col2 - b.col2)
}

/// The outer product `a bᵀ`, whose column `j` is `a * b_j`.
#[inline]
fn outer3(a: Vec3Fix, b: Vec3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(a * b.x, a * b.y, a * b.z)
}

/// The skew-symmetric matrix `c×`, so that `skew(c) * v == c.cross(v)`.
#[inline]
fn skew(c: Vec3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(
        c.cross(Vec3Fix::UNIT_X),
        c.cross(Vec3Fix::UNIT_Y),
        c.cross(Vec3Fix::UNIT_Z),
    )
}

/// `1 / v`, or [`INFINITE_INERTIA_STANDIN`] when `v` is zero.
#[inline]
fn reciprocal_or_standin(v: Fix128) -> Fix128 {
    if v.is_zero() {
        Fix128::from_int(INFINITE_INERTIA_STANDIN)
    } else {
        Fix128::ONE / v
    }
}

/// The rotation matrix of `q`, built by rotating the three basis vectors.
///
/// Exact for `QuatFix::IDENTITY`, which is what keeps an unrotated scene's
/// inertia equal to the stored diagonal bit for bit.
fn rotation_matrix(q: QuatFix) -> Mat3Fix {
    Mat3Fix::from_cols(
        q.rotate_vec(Vec3Fix::UNIT_X),
        q.rotate_vec(Vec3Fix::UNIT_Y),
        q.rotate_vec(Vec3Fix::UNIT_Z),
    )
}

/// The body's rotational inertia about its centre of mass, in world axes:
/// `R · diag(I_local) · Rᵀ`.
fn world_inertia(body: &RigidBody) -> Mat3Fix {
    let local = Mat3Fix::diagonal(
        reciprocal_or_standin(body.inv_inertia.x),
        reciprocal_or_standin(body.inv_inertia.y),
        reciprocal_or_standin(body.inv_inertia.z),
    );
    let r = rotation_matrix(body.rotation);
    r.mul_mat(local).mul_mat(r.transpose())
}

/// The body's spatial inertia, taken about the world origin.
///
/// With `m` the mass, `c` the centre of mass and `Ī_c` the rotational inertia
/// about the centre of mass, the momentum `h = I v` reads
///
/// ```text
///   L_O = (Ī_c - m c× c×) ω + m c× v_O
///   P   =        - m c×    ω +   m   v_O
/// ```
///
/// which is the block layout below. The matrix is symmetric because
/// `(c×)ᵀ = -(c×)`.
fn body_spatial_inertia(body: &RigidBody) -> SpatialMat {
    let mass = reciprocal_or_standin(body.inv_mass);
    let c = skew(body.position);
    let mc = c.scale(mass);
    SpatialMat {
        aa: mat_sub(world_inertia(body), c.mul_mat(c).scale(mass)),
        al: mc,
        la: mat_sub(Mat3Fix::ZERO, mc),
        ll: Mat3Fix::diagonal(mass, mass, mass),
    }
}

#[inline]
fn vec_component(v: Vec3Fix, i: usize) -> Fix128 {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[inline]
fn mat_component(m: Mat3Fix, row: usize, col: usize) -> Fix128 {
    let column = match col {
        0 => m.col0,
        1 => m.col1,
        _ => m.col2,
    };
    vec_component(column, row)
}

/// Flatten a spatial matrix into row-major `[row][col]` order, angular first.
fn spatial_to_rows(m: SpatialMat) -> [[Fix128; MAX_JOINT_DOF]; MAX_JOINT_DOF] {
    let mut out = [[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = mat_component(m.aa, row, col);
            out[row][col + 3] = mat_component(m.al, row, col);
            out[row + 3][col] = mat_component(m.la, row, col);
            out[row + 3][col + 3] = mat_component(m.ll, row, col);
        }
    }
    out
}

/// Inverse of the leading `n`x`n` block of `mat`, or `None` when it is singular.
///
/// Gauss-Jordan elimination with partial pivoting.
///
/// # Determinism
/// The pivot is the row with the largest magnitude in the current column, ties
/// broken by the lowest row index; `Fix128` comparison is an exact integer
/// comparison, so the pivot sequence depends only on the input bits. Every
/// arithmetic step is `Fix128`, so the routine is bit-identical on every
/// platform (skill §1 経路 2 — no transcendental, no float).
fn invert_small(
    mat: &[[Fix128; MAX_JOINT_DOF]; MAX_JOINT_DOF],
    n: usize,
) -> Option<[[Fix128; MAX_JOINT_DOF]; MAX_JOINT_DOF]> {
    let mut a = *mat;
    let mut inv = [[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF];
    for (i, row) in inv.iter_mut().enumerate().take(n) {
        row[i] = Fix128::ONE;
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut best = a[col][col].abs();
        for (row, candidate) in a.iter().enumerate().take(n).skip(col + 1) {
            let magnitude = candidate[col].abs();
            if magnitude > best {
                best = magnitude;
                pivot_row = row;
            }
        }
        if best.is_zero() {
            return None;
        }
        if pivot_row != col {
            a.swap(pivot_row, col);
            inv.swap(pivot_row, col);
        }

        let pivot = a[col][col];
        for k in 0..n {
            a[col][k] = a[col][k] / pivot;
            inv[col][k] = inv[col][k] / pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.is_zero() {
                continue;
            }
            for k in 0..n {
                a[row][k] = a[row][k] - a[col][k] * factor;
                inv[row][k] = inv[row][k] - inv[col][k] * factor;
            }
        }
    }

    Some(inv)
}

/// The motion subspace `S` of one joint, in world coordinates, as up to six
/// spatial basis vectors. `dof == 0` is a weld: no admissible relative motion.
#[derive(Clone, Copy, Debug)]
struct MotionSubspace {
    cols: [SpatialVec; MAX_JOINT_DOF],
    dof: usize,
}

impl Default for MotionSubspace {
    fn default() -> Self {
        Self {
            cols: [SpatialVec::ZERO; MAX_JOINT_DOF],
            dof: 0,
        }
    }
}

impl MotionSubspace {
    /// Rotation about the line through `anchor` with direction `axis`, as
    /// Plücker coordinates `[axis; anchor × axis]`.
    ///
    /// `axis` is deliberately **not** normalised: scaling a column by `k` scales
    /// the joint acceleration by `1/k` and leaves `S q̈` unchanged, so
    /// normalising would only introduce a square root and the rounding with it.
    #[inline]
    fn revolute(anchor: Vec3Fix, axis: Vec3Fix) -> SpatialVec {
        SpatialVec {
            ang: axis,
            lin: anchor.cross(axis),
        }
    }

    /// Translation along `axis`.
    #[inline]
    fn prismatic(axis: Vec3Fix) -> SpatialVec {
        SpatialVec {
            ang: Vec3Fix::ZERO,
            lin: axis,
        }
    }

    fn push(&mut self, col: SpatialVec) {
        if self.dof < MAX_JOINT_DOF {
            self.cols[self.dof] = col;
            self.dof += 1;
        }
    }

    /// The three rotational degrees of freedom of a ball-type joint.
    fn push_ball(&mut self, anchor: Vec3Fix) {
        for axis in [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z] {
            self.push(Self::revolute(anchor, axis));
        }
    }
}

/// The world position of a point given in a body's local frame.
#[inline]
fn world_point(body: &RigidBody, local: Vec3Fix) -> Vec3Fix {
    body.position + body.rotation.rotate_vec(local)
}

/// The joint's two anchor points, in their own bodies' local frames.
const fn local_anchors(joint: &Joint) -> (Vec3Fix, Vec3Fix) {
    match joint {
        Joint::Ball(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::Hinge(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::Fixed(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::Slider(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::Spring(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::D6(j) => (j.local_anchor_a, j.local_anchor_b),
        Joint::ConeTwist(j) => (j.local_anchor_a, j.local_anchor_b),
    }
}

/// The motion subspace of `joint`, in world coordinates.
///
/// The anchor is taken from whichever side of the joint is the parent link, so
/// that the subspace is anchored to the frame the child moves relative to. Joint
/// limits are not part of the subspace: a limit is a unilateral constraint that
/// only binds at the stop, and forward dynamics away from the stop is the
/// unlimited joint. Compliance is likewise ignored — a compliant joint is still
/// this joint kinematically.
fn subspace_for(joint: &Joint, bodies: &[RigidBody], parent_body: usize) -> MotionSubspace {
    let (body_a, body_b) = joint.bodies();
    let (anchor_a, anchor_b) = local_anchors(joint);
    let anchor = if body_b == parent_body && body_a != parent_body {
        world_point(&bodies[body_b], anchor_b)
    } else {
        world_point(&bodies[body_a], anchor_a)
    };
    // Axes are declared in body A's local frame by every joint that has one.
    let frame = bodies[body_a].rotation;

    let mut s = MotionSubspace::default();
    match joint {
        // Zero degrees of freedom: the child is rigidly carried by the parent.
        Joint::Fixed(_) => {}
        // Three rotational degrees of freedom about the anchor.
        Joint::Ball(_) | Joint::ConeTwist(_) => s.push_ball(anchor),
        Joint::Hinge(j) => {
            s.push(MotionSubspace::revolute(
                anchor,
                frame.rotate_vec(j.local_axis_a),
            ));
        }
        Joint::Slider(j) => {
            s.push(MotionSubspace::prismatic(frame.rotate_vec(j.local_axis)));
        }
        // A spring imposes no kinematic constraint at all — it supplies a force
        // between two otherwise free bodies — so its subspace is the whole of
        // the spatial motion space. The spring force itself is applied outside
        // forward dynamics.
        Joint::Spring(_) => {
            s.push_ball(anchor);
            for axis in [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z] {
                s.push(MotionSubspace::prismatic(axis));
            }
        }
        Joint::D6(j) => {
            let joint_frame = frame.mul(j.local_frame_a);
            let angular = [j.angular_x, j.angular_y, j.angular_z];
            let linear = [j.linear_x, j.linear_y, j.linear_z];
            let basis = [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z];
            for (motion, axis) in angular.iter().zip(basis.iter()) {
                if *motion != D6Motion::Locked {
                    s.push(MotionSubspace::revolute(
                        anchor,
                        joint_frame.rotate_vec(*axis),
                    ));
                }
            }
            for (motion, axis) in linear.iter().zip(basis.iter()) {
                if *motion != D6Motion::Locked {
                    s.push(MotionSubspace::prismatic(joint_frame.rotate_vec(*axis)));
                }
            }
        }
    }
    s
}

/// Per-link working state for one `solve` call.
#[derive(Clone, Debug)]
struct FeatherstoneLinkData {
    /// Articulated body inertia `I^A`, accumulated from the leaves.
    inertia: SpatialMat,
    /// Articulated bias force `p^A`, accumulated from the leaves.
    bias: SpatialVec,
    /// Motion subspace `S` of the joint to this link's parent.
    subspace: MotionSubspace,
    /// `U = I^A S`, one spatial force vector per degree of freedom.
    u: [SpatialVec; MAX_JOINT_DOF],
    /// `D⁻¹ = (Sᵀ I^A S)⁻¹`.
    d_inv: [[Fix128; MAX_JOINT_DOF]; MAX_JOINT_DOF],
    /// `u = τ - Sᵀ p^A`, the joint-space residual force.
    u_term: [Fix128; MAX_JOINT_DOF],
    /// Velocity-product acceleration `c = v × (v - v_parent)`.
    c_bias: SpatialVec,
    /// Spatial velocity of this link.
    vel: SpatialVec,
    /// Spatial acceleration of this link in the gravity-free frame.
    accel: SpatialVec,
}

impl Default for FeatherstoneLinkData {
    fn default() -> Self {
        Self {
            inertia: SpatialMat::ZERO,
            bias: SpatialVec::ZERO,
            subspace: MotionSubspace::default(),
            u: [SpatialVec::ZERO; MAX_JOINT_DOF],
            d_inv: [[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF],
            u_term: [Fix128::ZERO; MAX_JOINT_DOF],
            c_bias: SpatialVec::ZERO,
            vel: SpatialVec::ZERO,
            accel: SpatialVec::ZERO,
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
    #[must_use]
    pub const fn new() -> Self {
        Self {
            link_data: Vec::new(),
        }
    }

    /// Solve forward dynamics for an articulated body and integrate one step.
    ///
    /// Three-pass algorithm:
    /// 1. Forward pass: spatial velocities and the velocity-product
    ///    acceleration `c = v × (v - v_parent)`, root to leaves.
    /// 2. Backward pass: articulated body inertia `I^A` and bias force `p^A`,
    ///    leaves to root, each child folded into its parent through the rank-`n`
    ///    update `I^a = I^A - U D⁻¹ Uᵀ`.
    /// 3. Forward pass: joint accelerations `q̈ = D⁻¹(u - Uᵀ a')` and the link
    ///    accelerations they produce, then a semi-implicit Euler step.
    ///
    /// Joint torques are zero here: motors are applied separately through
    /// [`ArticulatedBody::apply_motors`], which writes velocities directly.
    ///
    /// # Determinism
    /// Links are visited in `children` order at every level, which is insertion
    /// order and independent of address or thread. All arithmetic is `Fix128`;
    /// the only inversions go through one Gauss-Jordan routine whose pivot rule
    /// is an exact integer comparison, so the pivot sequence depends only on the
    /// input bits.
    pub fn solve(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        gravity: Vec3Fix,
        dt: Fix128,
    ) {
        let n = artic.links.len();
        if n == 0 || artic.root >= n {
            return;
        }
        self.link_data.clear();
        self.link_data.resize_with(n, FeatherstoneLinkData::default);

        self.forward_velocity_pass(artic, bodies);
        self.backward_inertia_pass(artic, bodies);
        self.forward_acceleration_pass(artic, bodies, gravity, dt);
    }

    /// Featherstone forward dynamics with a Mass Splitting hint for
    /// large mass ratio stacks (Phase F 11.2 skeleton).
    ///
    /// Currently forwards to [`Self::solve`] unchanged. The follow-up
    /// commit implements the split policy: contact pairs whose mass
    /// ratio (heavier / lighter) exceeds `mass_ratio_split_threshold`
    /// are decomposed into two separately solved constraints so that
    /// PGS iteration count no longer has to fight the ill-conditioning
    /// of extreme mass ratios (see
    /// `deterministic-physics-lockstep-discipline` skill §11.2).
    ///
    /// # Determinism
    /// The Mass Splitting policy will canonicalise pivot order via
    /// `sort_by_key(mass) + index tie-break` and process the split
    /// pairs in flat-array index order, matching skill §1 経路 5.
    /// The current forwarding path inherits [`Self::solve`]'s
    /// determinism guarantee unchanged.
    ///
    /// # Status
    /// Skeleton API committed as part of Phase F 11.2 rollout. The
    /// signature is stable so downstream integrations (extreme mass
    /// ratio stacking scenes, ragdoll systems on soft ground) can
    /// begin experimenting; the actual splitting policy is scheduled
    /// for a follow-up commit.
    pub fn solve_with_mass_splitting(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        gravity: Vec3Fix,
        dt: Fix128,
        mass_ratio_split_threshold: Fix128,
    ) {
        // Body of Phase F 11.2: proxy Mass Splitting policy via dt
        // halving. Rationale: with an extreme mass ratio across the
        // articulated links (heaviest / lightest > threshold), a
        // single Featherstone forward-dynamics pass at the full `dt`
        // suffers from stiffness-induced ill-conditioning that shows
        // up as jitter at contact points. Splitting the step into two
        // half-steps of `dt / 2` acts as a first-order proxy for the
        // full Mass Splitting decomposition (which would decompose
        // contact impulses instead), relieving the effective spectral
        // radius without altering any per-link inertia.
        //
        // # Determinism
        // - The min / max reciprocal mass scan iterates over
        //   `artic.links` in index order (skill §1 経路 5).
        // - The ratio comparison uses Fix128 multiplication rather
        //   than division so the branch decision is bit-exact across
        //   platforms (skill §1 経路 2 — no CORDIC / rounding).
        // - `dt.half()` is a Fix128 arithmetic shift, itself bit-exact.
        // - Sub-stepping is a straight recursion into the existing
        //   `solve`, whose determinism guarantees carry through.
        let (min_inv_mass, max_inv_mass, has_dynamic) = {
            let mut lo = Fix128::ZERO;
            let mut hi = Fix128::ZERO;
            let mut seen_any = false;
            for link in &artic.links {
                let inv_m = bodies[link.body_index].inv_mass;
                if inv_m <= Fix128::ZERO {
                    continue; // static / kinematic — skip.
                }
                if seen_any {
                    if inv_m < lo {
                        lo = inv_m;
                    }
                    if inv_m > hi {
                        hi = inv_m;
                    }
                } else {
                    lo = inv_m;
                    hi = inv_m;
                    seen_any = true;
                }
            }
            (lo, hi, seen_any)
        };

        // Ratio (heavy / light) is `(1/min_inv_mass) / (1/max_inv_mass)
        // = max_inv_mass / min_inv_mass`. Compare via multiplication
        // to avoid a Fix128 division on the branch predicate.
        let use_splitting = has_dynamic
            && mass_ratio_split_threshold > Fix128::ZERO
            && min_inv_mass > Fix128::ZERO
            && max_inv_mass >= mass_ratio_split_threshold * min_inv_mass;

        if use_splitting {
            let half_dt = dt.half();
            self.solve(artic, bodies, gravity, half_dt);
            self.solve(artic, bodies, gravity, half_dt);
        } else {
            self.solve(artic, bodies, gravity, dt);
        }
    }

    /// Pass 1: spatial velocities and velocity-product accelerations, root to
    /// leaves.
    ///
    /// The joint velocity `S q̇` is recovered as `v_i - v_parent` rather than
    /// stored, which is exact and needs no minimal coordinates; the
    /// velocity-product term is then `c = v_i × (v_i - v_parent)`.
    fn forward_velocity_pass(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody]) {
        self.fv_recursive(artic, bodies, artic.root);
    }

    fn fv_recursive(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody], link_idx: usize) {
        let link = &artic.links[link_idx];
        let body = &bodies[link.body_index];

        // v_O is the velocity of the body-fixed point at the origin.
        let vel = SpatialVec {
            ang: body.angular_velocity,
            lin: body.velocity - body.angular_velocity.cross(body.position),
        };
        self.link_data[link_idx].vel = vel;

        if link.parent == LINK_ROOT {
            self.link_data[link_idx].c_bias = SpatialVec::ZERO;
            self.link_data[link_idx].subspace = MotionSubspace::default();
        } else {
            let parent_vel = self.link_data[link.parent].vel;
            self.link_data[link_idx].c_bias = vel.cross_motion(vel.sub(parent_vel));
            let parent_body = artic.links[link.parent].body_index;
            self.link_data[link_idx].subspace = link
                .joint
                .as_ref()
                .map_or_else(MotionSubspace::default, |joint| {
                    subspace_for(joint, bodies, parent_body)
                });
        }

        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.fv_recursive(artic, bodies, child);
        }
    }

    /// Pass 2: articulated body inertia and bias force, leaves to root.
    fn backward_inertia_pass(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody]) {
        self.bi_recursive(artic, bodies, artic.root);
    }

    fn bi_recursive(&mut self, artic: &ArticulatedBody, bodies: &[RigidBody], link_idx: usize) {
        // Children first: a child's `I^A` must be complete before it can be
        // folded into its parent.
        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.bi_recursive(artic, bodies, child);
        }

        let body = &bodies[artic.links[link_idx].body_index];
        let mut inertia = body_spatial_inertia(body);
        let vel = self.link_data[link_idx].vel;
        // Gravity is carried by the base-acceleration substitution, so the only
        // bias force here is the velocity product.
        let mut bias = vel.cross_force(inertia.mul_vec(vel));

        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            let (child_inertia, child_bias) = self.fold_child(child);
            inertia = inertia.add(child_inertia);
            bias = bias.add(child_bias);
        }

        self.link_data[link_idx].inertia = inertia;
        self.link_data[link_idx].bias = bias;
    }

    /// Reduce `child` across its joint, returning the `(I^a, p^a)` that its
    /// parent must add to its own.
    ///
    /// This is the rank-`n` articulated-body update
    ///
    /// ```text
    ///   U   = I^A S
    ///   D   = Sᵀ U
    ///   u   = τ - Sᵀ p^A            (τ = 0 — see `solve`)
    ///   I^a = I^A - U D⁻¹ Uᵀ
    ///   p^a = p^A + I^a c + U D⁻¹ u
    /// ```
    ///
    /// with `U`, `D⁻¹` and `u` stored on the child for the acceleration pass.
    ///
    /// A weld contributes `dof == 0`, for which the update degenerates to
    /// `I^a = I^A`, `p^a = p^A + I^A c` — the child's whole inertia is carried
    /// by the parent, which is what a zero-DOF joint means.
    ///
    /// If `D` is singular the joint carries no inertia along some direction of
    /// its subspace; `D⁻¹` is then taken as zero, which is the infinite-inertia
    /// limit and locks that joint for this step. Locking cannot inject energy,
    /// which is why it is the safe degenerate choice.
    fn fold_child(&mut self, child: usize) -> (SpatialMat, SpatialVec) {
        let data = &self.link_data[child];
        let inertia = data.inertia;
        let bias = data.bias;
        let c_bias = data.c_bias;
        let subspace = data.subspace;
        let dof = subspace.dof;

        let mut u = [SpatialVec::ZERO; MAX_JOINT_DOF];
        for (slot, column) in u.iter_mut().zip(subspace.cols.iter()).take(dof) {
            *slot = inertia.mul_vec(*column);
        }

        let mut d = [[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF];
        for (row, column) in d.iter_mut().zip(subspace.cols.iter()).take(dof) {
            for (slot, force) in row.iter_mut().zip(u.iter()).take(dof) {
                *slot = column.pair(*force);
            }
        }
        let d_inv = if dof == 0 {
            [[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF]
        } else {
            invert_small(&d, dof).unwrap_or([[Fix128::ZERO; MAX_JOINT_DOF]; MAX_JOINT_DOF])
        };

        let mut u_term = [Fix128::ZERO; MAX_JOINT_DOF];
        for (slot, column) in u_term.iter_mut().zip(subspace.cols.iter()).take(dof) {
            // τ = 0, so u = -Sᵀ p^A.
            *slot = Fix128::ZERO - column.pair(bias);
        }

        // W = U D⁻¹, column a of which is Σ_b U_b (D⁻¹)_{b,a}.
        let mut w = [SpatialVec::ZERO; MAX_JOINT_DOF];
        for a in 0..dof {
            let mut acc = SpatialVec::ZERO;
            for b in 0..dof {
                acc = acc.add(u[b].scaled(d_inv[b][a]));
            }
            w[a] = acc;
        }

        let mut correction = SpatialMat::ZERO;
        for a in 0..dof {
            correction = correction.add(SpatialMat::outer(w[a], u[a]));
        }
        let articulated_inertia = inertia.sub(correction);

        let mut u_d_u = SpatialVec::ZERO;
        for a in 0..dof {
            u_d_u = u_d_u.add(w[a].scaled(u_term[a]));
        }
        let articulated_bias = bias.add(articulated_inertia.mul_vec(c_bias)).add(u_d_u);

        let data = &mut self.link_data[child];
        data.u = u;
        data.d_inv = d_inv;
        data.u_term = u_term;

        (articulated_inertia, articulated_bias)
    }

    /// Pass 3: accelerations root to leaves, then integrate.
    fn forward_acceleration_pass(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        gravity: Vec3Fix,
        dt: Fix128,
    ) {
        let root = artic.root;
        let root_body = artic.links[root].body_index;
        // A base is held either because the articulation says so or because the
        // body itself is static.
        let root_fixed = artic.fixed_base || bodies[root_body].inv_mass.is_zero();

        let root_accel = if root_fixed {
            // Gravity-free frame: holding the base means `a = 0`, i.e. `ã = -a_g`.
            SpatialVec {
                ang: Vec3Fix::ZERO,
                lin: -gravity,
            }
        } else {
            // A floating base has no external force left once gravity is carried
            // by the substitution, so `I^A ã + p^A = 0`.
            self.solve_floating_base(root)
        };
        self.link_data[root].accel = root_accel;

        if !root_fixed {
            Self::integrate(&mut bodies[root_body], root_accel, gravity, dt);
        }

        for i in 0..artic.links[root].children.len() {
            let child = artic.links[root].children[i];
            self.fa_recursive(artic, bodies, gravity, dt, child);
        }
    }

    /// `ã_root = -(I^A)⁻¹ p^A`, or the held base when `I^A` is singular.
    ///
    /// A singular articulated-body inertia at a floating root means the root has
    /// no admissible acceleration at all; holding it is the only choice that does
    /// not invent one. It cannot arise from a body with positive mass, because
    /// the linear block is then `m·1` and the whole matrix is positive definite.
    fn solve_floating_base(&self, root: usize) -> SpatialVec {
        let rows = spatial_to_rows(self.link_data[root].inertia);
        let Some(inv) = invert_small(&rows, MAX_JOINT_DOF) else {
            return SpatialVec::ZERO;
        };
        let bias = self.link_data[root].bias;
        let rhs = [
            Fix128::ZERO - bias.ang.x,
            Fix128::ZERO - bias.ang.y,
            Fix128::ZERO - bias.ang.z,
            Fix128::ZERO - bias.lin.x,
            Fix128::ZERO - bias.lin.y,
            Fix128::ZERO - bias.lin.z,
        ];
        let mut out = [Fix128::ZERO; MAX_JOINT_DOF];
        for (row, slot) in inv.iter().zip(out.iter_mut()) {
            let mut acc = Fix128::ZERO;
            for (coefficient, value) in row.iter().zip(rhs.iter()) {
                acc = acc + *coefficient * *value;
            }
            *slot = acc;
        }
        SpatialVec {
            ang: Vec3Fix::new(out[0], out[1], out[2]),
            lin: Vec3Fix::new(out[3], out[4], out[5]),
        }
    }

    fn fa_recursive(
        &mut self,
        artic: &ArticulatedBody,
        bodies: &mut [RigidBody],
        gravity: Vec3Fix,
        dt: Fix128,
        link_idx: usize,
    ) {
        let parent = artic.links[link_idx].parent;
        let parent_accel = self.link_data[parent].accel;
        let data = &self.link_data[link_idx];
        let subspace = data.subspace;
        let dof = subspace.dof;

        // a' = a_parent + c
        let prior = parent_accel.add(data.c_bias);

        // q̈ = D⁻¹ (u - Uᵀ a')
        let mut accel = prior;
        for a in 0..dof {
            let mut q_ddot = Fix128::ZERO;
            for b in 0..dof {
                let residual = data.u_term[b] - data.u[b].pair(prior);
                q_ddot = q_ddot + data.d_inv[a][b] * residual;
            }
            accel = accel.add(subspace.cols[a].scaled(q_ddot));
        }
        self.link_data[link_idx].accel = accel;

        let body_index = artic.links[link_idx].body_index;
        if !bodies[body_index].inv_mass.is_zero() {
            Self::integrate(&mut bodies[body_index], accel, gravity, dt);
        }

        for i in 0..artic.links[link_idx].children.len() {
            let child = artic.links[link_idx].children[i];
            self.fa_recursive(artic, bodies, gravity, dt, child);
        }
    }

    /// Semi-implicit Euler step from a gravity-free spatial acceleration.
    ///
    /// `accel` is `ã`; the true spatial acceleration is `ã + a_g` with
    /// `a_g = [0; g]`. The classical acceleration of the centre of mass follows
    /// from the spatial one by `a_cm = a_O + α × c + ω × v_cm`, which is the
    /// time derivative of `v_O = v_cm - ω × c` rearranged.
    fn integrate(body: &mut RigidBody, accel: SpatialVec, gravity: Vec3Fix, dt: Fix128) {
        let alpha = accel.ang;
        let a_origin = accel.lin + gravity;
        let a_cm =
            a_origin + alpha.cross(body.position) + body.angular_velocity.cross(body.velocity);

        body.velocity = body.velocity + a_cm * dt;
        body.angular_velocity = body.angular_velocity + alpha * dt;
        body.position = body.position + body.velocity * dt;

        let (axis, ang_speed) = body.angular_velocity.normalize_with_length();
        if !ang_speed.is_zero() {
            let delta_rot = QuatFix::from_axis_angle(axis, ang_speed * dt);
            body.rotation = delta_rot.mul(body.rotation).normalize();
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

    /// A two-link chain stacked vertically over a static base, each link's
    /// centre of mass directly above its own joint anchor.
    ///
    /// The weight of each link then acts along a line through its anchor and
    /// exerts no moment about it, so the configuration is an equilibrium — an
    /// inverted pendulum balanced exactly on its pivot — and nothing moves.
    ///
    /// This scene used to be asserted the other way round ("Link 1 should
    /// fall"), which was only true while `solve` ignored `Link::joint` and
    /// free-fell every link. The assertion below is what mechanics says about
    /// this configuration, and it is exact because zero moment about the anchor
    /// is an exact statement about parallel vectors, not a near-cancellation.
    ///
    /// Holding an *unstable* equilibrium to the bit is worth pinning on its own.
    /// In floating point the balance would not survive: the moment is a
    /// difference of nearly equal quantities, the residue would be a few ulp of
    /// the wrong sign, and an inverted pendulum amplifies exactly that — it
    /// would visibly topple within a few hundred steps. Staying bit-identical to
    /// the starting state after ten steps is therefore not a weak assertion but
    /// a consequence of `Fix128` being exact, and it fails the moment the solver
    /// stops computing the moment exactly.
    ///
    /// Asserted together with
    /// `featherstone_swings_a_link_offset_from_its_anchor`, never alone:
    /// "nothing moves" on its own is satisfied by a solver that does nothing.
    #[test]
    fn featherstone_holds_a_chain_balanced_over_its_anchors() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO), // fixed base
            RigidBody::new(Vec3Fix::from_int(0, 2, 0), Fix128::from_int(2)),
            RigidBody::new(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
        ];
        let start: Vec<Vec3Fix> = bodies.iter().map(|b| b.position).collect();

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

        for _ in 0..10 {
            solver.solve(&artic, &mut bodies, gravity, dt);
        }

        for i in 1..bodies.len() {
            assert_eq!(
                bodies[i].velocity,
                Vec3Fix::ZERO,
                "link {i} balances over its anchor, so its weight has no moment \
                 there and it must not start moving"
            );
            assert_eq!(
                bodies[i].position, start[i],
                "link {i} must not drift off the balance point"
            );
        }
    }

    /// The same joint, with the link's centre of mass moved off the vertical
    /// through its anchor.
    ///
    /// Gravity now has a nonzero moment about the anchor, so the link must swing
    /// down and around it. Only the signs are asserted, because they follow from
    /// the geometry alone: a link at `+x` from its anchor falls, swings back
    /// toward the anchor, and turns clockwise in the xy plane (angular velocity
    /// along `-z`). The magnitudes depend on how the link's mass is distributed,
    /// which is a modelling choice, so they are not pinned here.
    #[test]
    fn featherstone_swings_a_link_offset_from_its_anchor() {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(2)),
        ];
        let start = bodies[1].position;

        let mut artic = ArticulatedBody::new(0, true);
        artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::from_int(-2, 0, 0),
            )),
            Vec3Fix::from_int(2, 0, 0),
        );

        let mut solver = FeatherstoneSolver::new();
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        let dt = Fix128::from_ratio(1, 60);

        for _ in 0..10 {
            solver.solve(&artic, &mut bodies, gravity, dt);
        }

        assert!(
            bodies[1].position.y < start.y,
            "gravity has a moment about the anchor here, so the link must fall, \
             got y = {}",
            bodies[1].position.y
        );
        assert!(
            bodies[1].position.x < start.x,
            "the link swings back toward the vertical through its anchor, so x \
             must decrease, got x = {}",
            bodies[1].position.x
        );
        assert!(
            bodies[1].angular_velocity.z < Fix128::ZERO,
            "a link at +x turning down about an anchor at the origin rotates \
             clockwise in the xy plane, so angular velocity is along -z, got {}",
            bodies[1].angular_velocity.z
        );
        assert_eq!(
            bodies[0].velocity,
            Vec3Fix::ZERO,
            "the static base must not be integrated"
        );
    }

    /// Two-link chain fixture matching the `test_featherstone_solver`
    /// scene shape: a static root (index 0) with two dynamic child
    /// links (indices 1 and 2) whose masses parametrise the ratio for
    /// `solve_with_mass_splitting` tests.
    fn build_mass_splitting_scene(
        mass_root_child: Fix128,
        mass_leaf: Fix128,
    ) -> (Vec<RigidBody>, ArticulatedBody) {
        let bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO), // fixed base
            RigidBody::new(Vec3Fix::from_int(0, 2, 0), mass_root_child),
            RigidBody::new(Vec3Fix::from_int(0, 4, 0), mass_leaf),
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
        (bodies, artic)
    }

    /// When `mass_ratio_split_threshold` is zero, the splitting branch
    /// must be inert and byte-for-byte identical to a plain `solve`
    /// (Phase F 11.2 opt-in semantics).
    #[test]
    fn mass_splitting_matches_plain_solve_when_threshold_zero() {
        let (mut bodies1, artic1) = build_mass_splitting_scene(Fix128::ONE, Fix128::ONE);
        let (mut bodies2, artic2) = build_mass_splitting_scene(Fix128::ONE, Fix128::ONE);
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        let dt = Fix128::from_ratio(1, 60);

        let mut plain = FeatherstoneSolver::new();
        plain.solve(&artic1, &mut bodies1, gravity, dt);

        let mut splitting = FeatherstoneSolver::new();
        splitting.solve_with_mass_splitting(&artic2, &mut bodies2, gravity, dt, Fix128::ZERO);

        for i in 0..bodies1.len() {
            assert_eq!(
                bodies1[i].position.y.hi, bodies2[i].position.y.hi,
                "body {i} y.hi must match"
            );
            assert_eq!(
                bodies1[i].position.y.lo, bodies2[i].position.y.lo,
                "body {i} y.lo must match"
            );
        }
    }

    /// When the mass ratio does not exceed the split threshold, the
    /// splitting branch is inert — behaviour matches the plain
    /// `solve` byte-for-byte.
    #[test]
    fn mass_splitting_matches_plain_solve_when_ratio_below_threshold() {
        let (mut bodies1, artic1) = build_mass_splitting_scene(Fix128::ONE, Fix128::from_int(2));
        let (mut bodies2, artic2) = build_mass_splitting_scene(Fix128::ONE, Fix128::from_int(2));
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        let dt = Fix128::from_ratio(1, 60);

        let mut plain = FeatherstoneSolver::new();
        plain.solve(&artic1, &mut bodies1, gravity, dt);

        let mut splitting = FeatherstoneSolver::new();
        // heavy/light = 2, threshold = 100 → splitting inert.
        splitting.solve_with_mass_splitting(
            &artic2,
            &mut bodies2,
            gravity,
            dt,
            Fix128::from_int(100),
        );

        for i in 0..bodies1.len() {
            assert_eq!(bodies1[i].position.y.hi, bodies2[i].position.y.hi);
        }
    }

    /// When the mass ratio exceeds the split threshold, the two
    /// half-step recursion runs; both dynamic links must still fall
    /// under gravity (positional invariant preserved) and the solver
    /// must not diverge.
    #[test]
    fn mass_splitting_applies_when_ratio_exceeds_threshold() {
        // heavy child (mass 100) over light middle link (mass 1) → ratio 100.
        let (mut bodies, artic) = build_mass_splitting_scene(Fix128::ONE, Fix128::from_int(100));
        let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        let dt = Fix128::from_ratio(1, 60);
        let y0_1 = bodies[1].position.y;
        let y0_2 = bodies[2].position.y;

        let mut splitting = FeatherstoneSolver::new();
        splitting.solve_with_mass_splitting(
            &artic,
            &mut bodies,
            gravity,
            dt,
            Fix128::from_int(10), // threshold 10 < ratio 100 → active
        );

        assert!(
            bodies[1].position.y <= y0_1,
            "link 1 must fall or stay under gravity"
        );
        assert!(
            bodies[2].position.y <= y0_2,
            "link 2 must fall or stay under gravity"
        );
        // No divergence sanity check.
        assert!(bodies[1].position.y > Fix128::from_int(-1000));
        assert!(bodies[2].position.y > Fix128::from_int(-1000));
    }

    #[test]
    fn apply_motors_drives_each_motored_link_along_its_joint_axis() {
        // root (static) at origin、link 1 (inv 1) at (4,0,0)、link 2 (inv 2) at (4,3,0)
        // link1 motor: kp 10 / target 距離 6 (現在 4) → force 20、dt 1/4 → impulse 5 → body1 += (5,0,0)
        // link2 motor: kp 10 / target 距離 1 (現在 3、軸 +y) → force -20 → impulse (0,-5,0)
        //   body1 (inv 1) -= (0,-5,0) → (5, 5, 0)、body2 (inv 2) += (0,-10,0)
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new_dynamic(Vec3Fix::from_int(4, 0, 0), Fix128::ONE),
            RigidBody::new_dynamic(Vec3Fix::from_int(4, 3, 0), Fix128::ONE),
        ];
        bodies[2].inv_mass = Fix128::from_int(2);

        let mut artic = ArticulatedBody::new(0, true);
        let l1 = artic.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(4, 0, 0),
        );
        let l2 = artic.add_link(
            l1,
            2,
            Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(0, 3, 0),
        );
        let mut m1 = PdController::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(100));
        m1.set_position_target(Fix128::from_int(6));
        let mut m2 = m1;
        m2.set_position_target(Fix128::ONE);
        artic.set_motor(l1, m1);
        artic.set_motor(l2, m2);
        // 範囲外 link への set_motor は無視
        artic.set_motor(99, m1);
        assert_eq!(artic.link_count(), 3);

        let dt = Fix128::from_ratio(1, 4);
        artic.apply_motors(&mut bodies, dt);
        assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);
        assert_eq!(bodies[1].velocity, Vec3Fix::from_int(5, 5, 0));
        assert_eq!(bodies[2].velocity, Vec3Fix::from_int(0, -10, 0));

        // 手計算と同じ結果を motor::apply_motors (free fn) でも得る = 両実装の等価性
        let mut flat = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new_dynamic(Vec3Fix::from_int(4, 0, 0), Fix128::ONE),
            RigidBody::new_dynamic(Vec3Fix::from_int(4, 3, 0), Fix128::ONE),
        ];
        flat[2].inv_mass = Fix128::from_int(2);
        let joints: Vec<Joint> = artic.joints().into_iter().copied().collect();
        let motors = [
            crate::motor::JointMotor::new(0, m1),
            crate::motor::JointMotor::new(1, m2),
        ];
        crate::motor::apply_motors(&motors, &joints, &mut flat, dt);
        assert_eq!(flat[1].velocity, bodies[1].velocity);
        assert_eq!(flat[2].velocity, bodies[2].velocity);

        // Off motor / motor なし link は何もしない
        let mut idle = ArticulatedBody::new(0, true);
        let li = idle.add_link(
            0,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::from_int(4, 0, 0),
        );
        let mut quiet = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new_dynamic(Vec3Fix::from_int(4, 0, 0), Fix128::ONE),
        ];
        idle.apply_motors(&mut quiet, dt);
        assert_eq!(quiet[1].velocity, Vec3Fix::ZERO);
        let mut off = m1;
        off.disable();
        idle.set_motor(li, off);
        idle.apply_motors(&mut quiet, dt);
        assert_eq!(quiet[1].velocity, Vec3Fix::ZERO);
    }
}
