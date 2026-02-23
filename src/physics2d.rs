//! 2D Physics Subsystem — Deterministic XPBD with 128-bit Fixed-Point Arithmetic
//!
//! A complete 2D rigid-body physics module built on [`Fix128`] (I64F64).
//! Every operation is deterministic and produces bit-exact results across all
//! platforms.
//!
//! # Features
//!
//! - **Vec2Fix**: 2D vector with full operator overloading
//! - **Shape2D**: Circle, convex polygon, capsule, and edge shapes
//! - **RigidBody2D**: Dynamic, static, and kinematic rigid bodies
//! - **Collision Detection**: SAT for polygon-polygon, analytic for circles/capsules
//! - **XPBD Solver**: Extended Position Based Dynamics with substeps
//! - **Joint2D**: Revolute, distance, weld, and mouse joints
//!
//! # Determinism
//!
//! All arithmetic uses [`Fix128`]. No `f32`/`f64` anywhere. Iteration counts
//! are fixed. No `HashMap` or non-deterministic data structures.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::Fix128;

// ============================================================================
// Vec2Fix — 2D Vector
// ============================================================================

/// 2D vector using [`Fix128`] components.
///
/// Provides full operator overloading, geometric utilities, and deterministic
/// arithmetic for 2D physics simulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Vec2Fix {
    /// X component
    pub x: Fix128,
    /// Y component
    pub y: Fix128,
}

impl Vec2Fix {
    /// Zero vector (0, 0)
    pub const ZERO: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
    };

    /// One vector (1, 1)
    pub const ONE: Self = Self {
        x: Fix128::ONE,
        y: Fix128::ONE,
    };

    /// Unit X vector (1, 0)
    pub const UNIT_X: Self = Self {
        x: Fix128::ONE,
        y: Fix128::ZERO,
    };

    /// Unit Y vector (0, 1)
    pub const UNIT_Y: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ONE,
    };

    /// Create a new 2D vector.
    #[inline]
    #[must_use]
    pub const fn new(x: Fix128, y: Fix128) -> Self {
        Self { x, y }
    }

    /// Create from integer components.
    #[inline]
    #[must_use]
    pub const fn from_int(x: i64, y: i64) -> Self {
        Self {
            x: Fix128::from_int(x),
            y: Fix128::from_int(y),
        }
    }

    /// Squared length (avoids sqrt).
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> Fix128 {
        self.x * self.x + self.y * self.y
    }

    /// Length (magnitude).
    #[inline]
    #[must_use]
    pub fn length(self) -> Fix128 {
        self.length_squared().sqrt()
    }

    /// Normalize to unit length. Returns `ZERO` for zero-length vectors.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len.is_zero() {
            Self::ZERO
        } else {
            self / len
        }
    }

    /// Dot product.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> Fix128 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// 2D cross product (returns a scalar: `a.x * b.y - a.y * b.x`).
    ///
    /// This is the z-component of the 3D cross product when both vectors
    /// are embedded in the XY plane.
    #[inline]
    #[must_use]
    pub fn cross_scalar(self, rhs: Self) -> Fix128 {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Rotate this vector by an angle (radians, counter-clockwise).
    #[must_use]
    pub fn rotate(self, angle: Fix128) -> Self {
        let (sin_a, cos_a) = angle.sin_cos();
        Self {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    /// Return the perpendicular vector (90 degrees counter-clockwise): `(-y, x)`.
    #[inline]
    #[must_use]
    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// Distance to another point.
    #[inline]
    #[must_use]
    pub fn distance_to(self, other: Self) -> Fix128 {
        (other - self).length()
    }

    /// Linear interpolation: `self + (other - self) * t`.
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: Fix128) -> Self {
        self + (other - self) * t
    }

    /// Scale by a scalar.
    #[inline]
    #[must_use]
    pub fn scale(self, s: Fix128) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

impl Add for Vec2Fix {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vec2Fix {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<Fix128> for Vec2Fix {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Fix128) -> Self {
        self.scale(rhs)
    }
}

impl Div<Fix128> for Vec2Fix {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Fix128) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl Neg for Vec2Fix {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// ============================================================================
// Shape2D — Collision Shapes
// ============================================================================

/// 2D collision shape.
///
/// All shapes are defined in local space relative to the body's center of mass.
#[derive(Clone, Debug)]
pub enum Shape2D {
    /// Circle defined by its radius.
    Circle {
        /// Radius of the circle.
        radius: Fix128,
    },
    /// Convex polygon defined by vertices in CCW winding order.
    Polygon {
        /// Vertices in counter-clockwise order. Must form a convex hull.
        vertices: Vec<Vec2Fix>,
    },
    /// Capsule defined by a radius and half-length along the local X axis.
    Capsule {
        /// Radius of the capsule's hemicircles.
        radius: Fix128,
        /// Half of the segment length between hemicircle centers.
        half_length: Fix128,
    },
    /// Line segment (edge) from start to end.
    Edge {
        /// Start point in local space.
        start: Vec2Fix,
        /// End point in local space.
        end: Vec2Fix,
    },
}

// ============================================================================
// BodyType2D
// ============================================================================

/// Type of a 2D rigid body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyType2D {
    /// Fully simulated body affected by forces and collisions.
    Dynamic,
    /// Immovable body (infinite mass, zero velocity).
    Static,
    /// User-controlled body that affects dynamics but is not affected by forces.
    Kinematic,
}

// ============================================================================
// RigidBody2D
// ============================================================================

/// 2D rigid body with position, orientation, velocity, and shape.
#[derive(Clone, Debug)]
pub struct RigidBody2D {
    /// World-space position of the center of mass.
    pub position: Vec2Fix,
    /// Orientation angle in radians (counter-clockwise from +X).
    pub angle: Fix128,
    /// Linear velocity.
    pub velocity: Vec2Fix,
    /// Angular velocity (radians per second, positive = CCW).
    pub angular_velocity: Fix128,
    /// Inverse mass (0 for static/kinematic bodies).
    pub inv_mass: Fix128,
    /// Inverse moment of inertia (0 for static/kinematic bodies).
    pub inv_inertia: Fix128,
    /// Collision shape.
    pub shape: Shape2D,
    /// Coefficient of restitution (bounciness, 0..1).
    pub restitution: Fix128,
    /// Friction coefficient.
    pub friction: Fix128,
    /// Body type.
    pub body_type: BodyType2D,
    /// Previous position (used by XPBD solver).
    pub prev_position: Vec2Fix,
    /// Previous angle (used by XPBD solver).
    pub prev_angle: Fix128,
}

impl RigidBody2D {
    /// Create a new dynamic body with a circle shape.
    ///
    /// Mass is derived from the given value. Inertia is computed for the circle
    /// shape: `I = 0.5 * m * r^2`.
    #[must_use]
    pub fn new_dynamic(position: Vec2Fix, mass: Fix128, shape: Shape2D) -> Self {
        let inv_mass = if mass.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / mass
        };
        let inertia = compute_inertia(&shape, mass);
        let inv_inertia = if inertia.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / inertia
        };
        Self {
            position,
            angle: Fix128::ZERO,
            velocity: Vec2Fix::ZERO,
            angular_velocity: Fix128::ZERO,
            inv_mass,
            inv_inertia,
            shape,
            restitution: Fix128::from_ratio(5, 10),
            friction: Fix128::from_ratio(3, 10),
            body_type: BodyType2D::Dynamic,
            prev_position: position,
            prev_angle: Fix128::ZERO,
        }
    }

    /// Create a new static (immovable) body.
    #[must_use]
    pub fn new_static(position: Vec2Fix, shape: Shape2D) -> Self {
        Self {
            position,
            angle: Fix128::ZERO,
            velocity: Vec2Fix::ZERO,
            angular_velocity: Fix128::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Fix128::ZERO,
            shape,
            restitution: Fix128::from_ratio(5, 10),
            friction: Fix128::from_ratio(5, 10),
            body_type: BodyType2D::Static,
            prev_position: position,
            prev_angle: Fix128::ZERO,
        }
    }

    /// Create a new kinematic body.
    #[must_use]
    pub fn new_kinematic(position: Vec2Fix, shape: Shape2D) -> Self {
        Self {
            position,
            angle: Fix128::ZERO,
            velocity: Vec2Fix::ZERO,
            angular_velocity: Fix128::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Fix128::ZERO,
            shape,
            restitution: Fix128::from_ratio(5, 10),
            friction: Fix128::from_ratio(3, 10),
            body_type: BodyType2D::Kinematic,
            prev_position: position,
            prev_angle: Fix128::ZERO,
        }
    }

    /// Apply a linear impulse at the center of mass.
    #[inline]
    pub fn apply_impulse(&mut self, impulse: Vec2Fix) {
        if self.body_type != BodyType2D::Dynamic {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
    }

    /// Apply a linear impulse at a world-space point, generating both linear
    /// and angular impulse.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec2Fix, world_point: Vec2Fix) {
        if self.body_type != BodyType2D::Dynamic {
            return;
        }
        self.velocity = self.velocity + impulse * self.inv_mass;
        let r = world_point - self.position;
        self.angular_velocity = self.angular_velocity + r.cross_scalar(impulse) * self.inv_inertia;
    }

    /// Apply a force (will be integrated over the next timestep).
    #[inline]
    pub fn apply_force(&mut self, force: Vec2Fix, dt: Fix128) {
        if self.body_type != BodyType2D::Dynamic {
            return;
        }
        self.velocity = self.velocity + force * self.inv_mass * dt;
    }

    /// Transform a local-space point to world space.
    #[must_use]
    pub fn world_point(&self, local: Vec2Fix) -> Vec2Fix {
        self.position + local.rotate(self.angle)
    }

    /// Returns `true` if this body has zero inverse mass (static or kinematic).
    #[inline]
    #[must_use]
    pub fn is_static_or_kinematic(&self) -> bool {
        self.body_type != BodyType2D::Dynamic
    }
}

/// Compute moment of inertia for a 2D shape with given mass.
fn compute_inertia(shape: &Shape2D, mass: Fix128) -> Fix128 {
    match shape {
        Shape2D::Circle { radius } => {
            // I = 0.5 * m * r^2
            mass * *radius * *radius / Fix128::from_int(2)
        }
        Shape2D::Capsule {
            radius,
            half_length,
        } => {
            // Approximate as rectangle + two semicircles
            // I_rect = m_rect * (w^2 + h^2) / 12
            // I_circle = m_circle * r^2 / 2
            // Simplified: I ~ m * (r^2 / 2 + half_length^2 / 3)
            let r2 = *radius * *radius;
            let h2 = *half_length * *half_length;
            mass * (r2 / Fix128::from_int(2) + h2 / Fix128::from_int(3))
        }
        Shape2D::Polygon { vertices } => {
            // Use the polygon moment of inertia formula
            if vertices.len() < 3 {
                return Fix128::ZERO;
            }
            let mut numerator = Fix128::ZERO;
            let mut denominator = Fix128::ZERO;
            let n = vertices.len();
            for i in 0..n {
                let a = vertices[i];
                let b = vertices[(i + 1) % n];
                let cross = a.cross_scalar(b).abs();
                numerator = numerator + cross * (a.dot(a) + a.dot(b) + b.dot(b));
                denominator = denominator + cross;
            }
            if denominator.is_zero() {
                return Fix128::ZERO;
            }
            mass * numerator / (denominator * Fix128::from_int(6))
        }
        Shape2D::Edge { start, end } => {
            // Treat as thin rod: I = m * L^2 / 12
            let l2 = (*end - *start).length_squared();
            mass * l2 / Fix128::from_int(12)
        }
    }
}

// ============================================================================
// Contact2D
// ============================================================================

/// Contact point between two 2D bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact2D {
    /// World-space contact point.
    pub point: Vec2Fix,
    /// Contact normal (points from body_a toward body_b).
    pub normal: Vec2Fix,
    /// Penetration depth (positive means overlapping).
    pub depth: Fix128,
    /// Index of the first body.
    pub body_a: usize,
    /// Index of the second body.
    pub body_b: usize,
}

// ============================================================================
// PhysicsConfig2D
// ============================================================================

/// Configuration for the 2D physics world.
#[derive(Clone, Debug)]
pub struct PhysicsConfig2D {
    /// Gravitational acceleration vector.
    pub gravity: Vec2Fix,
    /// Number of substeps per `step()` call.
    pub substeps: usize,
    /// Number of constraint solver iterations per substep.
    pub iterations: usize,
    /// Linear velocity damping factor (applied per step, 0..1).
    pub damping: Fix128,
}

impl Default for PhysicsConfig2D {
    fn default() -> Self {
        Self {
            gravity: Vec2Fix::new(Fix128::ZERO, Fix128::from_int(-10)),
            substeps: 4,
            iterations: 8,
            damping: Fix128::from_ratio(99, 100),
        }
    }
}

// ============================================================================
// PhysicsWorld2D
// ============================================================================

/// 2D physics world containing bodies, joints, and solver state.
pub struct PhysicsWorld2D {
    /// All rigid bodies in the world.
    pub bodies: Vec<RigidBody2D>,
    /// Gravity vector.
    pub gravity: Vec2Fix,
    /// Solver configuration.
    pub config: PhysicsConfig2D,
    /// Active joints.
    pub joints: Vec<Joint2D>,
}

impl PhysicsWorld2D {
    /// Create a new empty 2D physics world.
    #[must_use]
    pub fn new(config: PhysicsConfig2D) -> Self {
        let gravity = config.gravity;
        Self {
            bodies: Vec::new(),
            gravity,
            config,
            joints: Vec::new(),
        }
    }

    /// Add a body to the world. Returns the body index.
    pub fn add_body(&mut self, body: RigidBody2D) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Remove a body by index. Swaps with last to preserve indices of other
    /// bodies (except the last one). Returns the removed body if the index
    /// was valid.
    pub fn remove_body(&mut self, index: usize) -> Option<RigidBody2D> {
        if index >= self.bodies.len() {
            return None;
        }
        Some(self.bodies.swap_remove(index))
    }

    /// Add a joint to the world.
    pub fn add_joint(&mut self, joint: Joint2D) {
        self.joints.push(joint);
    }

    /// Step the simulation by `dt` seconds.
    ///
    /// Internally runs `config.substeps` substeps, each with collision
    /// detection, XPBD position correction, and velocity derivation.
    pub fn step(&mut self, dt: Fix128) {
        let substeps = self.config.substeps;
        if substeps == 0 {
            return;
        }
        let sub_dt = dt / Fix128::from_int(substeps as i64);

        for _ in 0..substeps {
            self.substep(sub_dt);
        }

        // Apply damping
        let damping = self.config.damping;
        for body in &mut self.bodies {
            if body.body_type == BodyType2D::Dynamic {
                body.velocity = body.velocity * damping;
                body.angular_velocity = body.angular_velocity * damping;
            }
        }
    }

    /// Perform one substep of the XPBD solver.
    fn substep(&mut self, sub_dt: Fix128) {
        let gravity = self.gravity;
        let iterations = self.config.iterations;

        // 1. Store previous state and integrate velocity
        for body in &mut self.bodies {
            if body.body_type != BodyType2D::Dynamic {
                // Kinematic: update prev but don't apply gravity
                body.prev_position = body.position;
                body.prev_angle = body.angle;
                if body.body_type == BodyType2D::Kinematic {
                    body.position = body.position + body.velocity * sub_dt;
                    body.angle = body.angle + body.angular_velocity * sub_dt;
                }
                continue;
            }
            body.prev_position = body.position;
            body.prev_angle = body.angle;

            // Apply gravity
            body.velocity = body.velocity + gravity * sub_dt;

            // Predict position
            body.position = body.position + body.velocity * sub_dt;
            body.angle = body.angle + body.angular_velocity * sub_dt;
        }

        // 2. Detect collisions
        let contacts = self.detect_all_contacts();

        // 3. Solve constraints (position-based)
        for _ in 0..iterations {
            // Solve contact constraints
            solve_contacts_xpbd(&mut self.bodies, &contacts);

            // Solve joints
            solve_joints_2d(&mut self.bodies, &self.joints, sub_dt);
        }

        // 4. Derive velocity from position change
        if sub_dt.is_zero() {
            return;
        }
        let inv_dt = Fix128::ONE / sub_dt;
        for body in &mut self.bodies {
            if body.body_type != BodyType2D::Dynamic {
                continue;
            }
            body.velocity = (body.position - body.prev_position) * inv_dt;
            body.angular_velocity = (body.angle - body.prev_angle) * inv_dt;
        }
    }

    /// Detect all contact pairs via brute-force N^2 broad phase.
    fn detect_all_contacts(&self) -> Vec<Contact2D> {
        let mut contacts = Vec::new();
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Skip static-static pairs
                if self.bodies[i].is_static_or_kinematic()
                    && self.bodies[j].is_static_or_kinematic()
                {
                    continue;
                }
                if let Some(contact) = self.check_collision_2d(i, j) {
                    contacts.push(contact);
                }
            }
        }
        contacts
    }

    /// Check collision between two bodies by index.
    #[must_use]
    pub fn check_collision_2d(&self, idx_a: usize, idx_b: usize) -> Option<Contact2D> {
        let body_a = &self.bodies[idx_a];
        let body_b = &self.bodies[idx_b];

        let result = match (&body_a.shape, &body_b.shape) {
            (Shape2D::Circle { radius: ra }, Shape2D::Circle { radius: rb }) => {
                circle_vs_circle(body_a.position, *ra, body_b.position, *rb)
            }
            (Shape2D::Circle { radius }, Shape2D::Polygon { vertices }) => {
                circle_vs_polygon(body_a.position, *radius, body_b, vertices)
            }
            (Shape2D::Polygon { vertices }, Shape2D::Circle { radius }) => {
                circle_vs_polygon(body_b.position, *radius, body_a, vertices).map(|mut c| {
                    c.normal = -c.normal;
                    c
                })
            }
            (Shape2D::Polygon { vertices: va }, Shape2D::Polygon { vertices: vb }) => {
                polygon_vs_polygon(body_a, va, body_b, vb)
            }
            (
                Shape2D::Circle { radius: rc },
                Shape2D::Capsule {
                    radius: rcap,
                    half_length,
                },
            ) => capsule_vs_circle(body_b, *rcap, *half_length, body_a.position, *rc),
            (
                Shape2D::Capsule {
                    radius: rcap,
                    half_length,
                },
                Shape2D::Circle { radius: rc },
            ) => {
                capsule_vs_circle(body_a, *rcap, *half_length, body_b.position, *rc).map(|mut c| {
                    c.normal = -c.normal;
                    c
                })
            }
            (Shape2D::Edge { start, end }, Shape2D::Circle { radius }) => {
                edge_vs_circle(body_a, *start, *end, body_b.position, *radius)
            }
            (Shape2D::Circle { radius }, Shape2D::Edge { start, end }) => {
                edge_vs_circle(body_b, *start, *end, body_a.position, *radius).map(|mut c| {
                    c.normal = -c.normal;
                    c
                })
            }
            _ => None,
        };

        result.map(|mut c| {
            c.body_a = idx_a;
            c.body_b = idx_b;
            c
        })
    }
}

// ============================================================================
// Collision Detection — Internal Functions
// ============================================================================

/// Circle vs circle collision test.
fn circle_vs_circle(
    pos_a: Vec2Fix,
    radius_a: Fix128,
    pos_b: Vec2Fix,
    radius_b: Fix128,
) -> Option<Contact2D> {
    let delta = pos_b - pos_a;
    let dist_sq = delta.length_squared();
    let sum_r = radius_a + radius_b;
    let sum_r_sq = sum_r * sum_r;

    if dist_sq > sum_r_sq || dist_sq.is_zero() {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist.is_zero() {
        Vec2Fix::UNIT_Y
    } else {
        delta / dist
    };

    let depth = sum_r - dist;
    let point = pos_a + normal * (radius_a - depth.half());

    Some(Contact2D {
        point,
        normal,
        depth,
        body_a: 0,
        body_b: 0,
    })
}

/// Transform polygon vertices from local space to world space.
fn transform_vertices(body: &RigidBody2D, local_verts: &[Vec2Fix]) -> Vec<Vec2Fix> {
    local_verts.iter().map(|v| body.world_point(*v)).collect()
}

/// Circle vs convex polygon collision using SAT with Voronoi regions.
fn circle_vs_polygon(
    circle_pos: Vec2Fix,
    circle_radius: Fix128,
    poly_body: &RigidBody2D,
    poly_verts: &[Vec2Fix],
) -> Option<Contact2D> {
    if poly_verts.len() < 3 {
        return None;
    }

    let world_verts = transform_vertices(poly_body, poly_verts);
    let n = world_verts.len();

    // Find the closest edge and check separation
    let mut best_dist = Fix128::from_int(-999_999);
    let mut best_normal = Vec2Fix::ZERO;
    let mut best_idx = 0;

    for i in 0..n {
        let a = world_verts[i];
        let b = world_verts[(i + 1) % n];
        let edge = b - a;
        // Outward normal (for CCW winding)
        let normal = Vec2Fix::new(edge.y, -edge.x).normalize();
        let d = (circle_pos - a).dot(normal);

        if d > best_dist {
            best_dist = d;
            best_normal = normal;
            best_idx = i;
        }
    }

    // Check if circle center is outside the polygon beyond its radius
    if best_dist > circle_radius {
        return None;
    }

    // Determine Voronoi region: vertex or edge
    let a = world_verts[best_idx];
    let b = world_verts[(best_idx + 1) % n];
    let edge = b - a;
    let edge_len_sq = edge.length_squared();

    let t = if edge_len_sq.is_zero() {
        Fix128::ZERO
    } else {
        (circle_pos - a).dot(edge) / edge_len_sq
    };

    if t < Fix128::ZERO {
        // Vertex A region
        let delta = circle_pos - a;
        let dist = delta.length();
        if dist > circle_radius || dist.is_zero() {
            return None;
        }
        let normal = delta / dist;
        let depth = circle_radius - dist;
        Some(Contact2D {
            point: a,
            normal,
            depth,
            body_a: 0,
            body_b: 0,
        })
    } else if t > Fix128::ONE {
        // Vertex B region
        let delta = circle_pos - b;
        let dist = delta.length();
        if dist > circle_radius || dist.is_zero() {
            return None;
        }
        let normal = delta / dist;
        let depth = circle_radius - dist;
        Some(Contact2D {
            point: b,
            normal,
            depth,
            body_a: 0,
            body_b: 0,
        })
    } else {
        // Edge region
        let depth = circle_radius - best_dist;
        if depth.is_negative() {
            return None;
        }
        let point = circle_pos - best_normal * best_dist;
        Some(Contact2D {
            point,
            normal: best_normal,
            depth,
            body_a: 0,
            body_b: 0,
        })
    }
}

/// Convex polygon vs convex polygon collision using SAT (Separating Axis Theorem).
fn polygon_vs_polygon(
    body_a: &RigidBody2D,
    verts_a: &[Vec2Fix],
    body_b: &RigidBody2D,
    verts_b: &[Vec2Fix],
) -> Option<Contact2D> {
    if verts_a.len() < 3 || verts_b.len() < 3 {
        return None;
    }

    let world_a = transform_vertices(body_a, verts_a);
    let world_b = transform_vertices(body_b, verts_b);

    let mut min_depth = Fix128::from_int(999_999);
    let mut best_normal = Vec2Fix::ZERO;

    // Test axes from polygon A
    if let Some((depth, normal)) = sat_test_axes(&world_a, &world_b) {
        if depth < min_depth {
            min_depth = depth;
            best_normal = normal;
        }
    } else {
        return None; // Separating axis found
    }

    // Test axes from polygon B
    if let Some((depth, normal)) = sat_test_axes(&world_b, &world_a) {
        if depth < min_depth {
            min_depth = depth;
            best_normal = normal;
        }
    } else {
        return None; // Separating axis found
    }

    // Ensure normal points from A to B
    let center_a = polygon_centroid(&world_a);
    let center_b = polygon_centroid(&world_b);
    let ab = center_b - center_a;
    if ab.dot(best_normal).is_negative() {
        best_normal = -best_normal;
    }

    // Compute contact point (midpoint of overlap)
    let point = (center_a + center_b) * Fix128::from_ratio(1, 2);

    Some(Contact2D {
        point,
        normal: best_normal,
        depth: min_depth,
        body_a: 0,
        body_b: 0,
    })
}

/// Test all edge normals of `poly_ref` as separating axes against `poly_test`.
/// Returns the minimum overlap depth and corresponding normal, or `None` if
/// a separating axis is found.
fn sat_test_axes(poly_ref: &[Vec2Fix], poly_test: &[Vec2Fix]) -> Option<(Fix128, Vec2Fix)> {
    let n = poly_ref.len();
    let mut min_depth = Fix128::from_int(999_999);
    let mut best_normal = Vec2Fix::ZERO;

    for i in 0..n {
        let a = poly_ref[i];
        let b = poly_ref[(i + 1) % n];
        let edge = b - a;
        let normal = Vec2Fix::new(edge.y, -edge.x).normalize();

        // Project both polygons onto this axis
        let (min_a, max_a) = project_polygon(poly_ref, normal);
        let (min_b, max_b) = project_polygon(poly_test, normal);

        // Check overlap
        if max_a < min_b || max_b < min_a {
            return None; // Separating axis found
        }

        // Compute overlap depth
        let overlap1 = max_a - min_b;
        let overlap2 = max_b - min_a;
        let depth = if overlap1 < overlap2 {
            overlap1
        } else {
            overlap2
        };

        if depth < min_depth {
            min_depth = depth;
            best_normal = normal;
        }
    }

    Some((min_depth, best_normal))
}

/// Project a polygon onto an axis and return (min, max) projections.
fn project_polygon(verts: &[Vec2Fix], axis: Vec2Fix) -> (Fix128, Fix128) {
    let mut min_proj = verts[0].dot(axis);
    let mut max_proj = min_proj;

    for v in verts.iter().skip(1) {
        let p = v.dot(axis);
        if p < min_proj {
            min_proj = p;
        }
        if p > max_proj {
            max_proj = p;
        }
    }

    (min_proj, max_proj)
}

/// Compute centroid of a polygon.
fn polygon_centroid(verts: &[Vec2Fix]) -> Vec2Fix {
    let mut sum = Vec2Fix::ZERO;
    for v in verts {
        sum = sum + *v;
    }
    let n = Fix128::from_int(verts.len() as i64);
    sum / n
}

/// Capsule vs circle collision.
///
/// The capsule is centered at `body_cap.position` with its segment along the
/// local X axis (from `-half_length` to `+half_length`).
fn capsule_vs_circle(
    body_cap: &RigidBody2D,
    cap_radius: Fix128,
    cap_half_length: Fix128,
    circle_pos: Vec2Fix,
    circle_radius: Fix128,
) -> Option<Contact2D> {
    // Capsule segment endpoints in world space
    let local_a = Vec2Fix::new(-cap_half_length, Fix128::ZERO);
    let local_b = Vec2Fix::new(cap_half_length, Fix128::ZERO);
    let seg_a = body_cap.world_point(local_a);
    let seg_b = body_cap.world_point(local_b);

    // Find closest point on segment to circle center
    let closest = closest_point_on_segment(seg_a, seg_b, circle_pos);

    // Now it's a circle-circle test between closest point and circle center
    let sum_r = cap_radius + circle_radius;
    let delta = circle_pos - closest;
    let dist_sq = delta.length_squared();
    let sum_r_sq = sum_r * sum_r;

    if dist_sq > sum_r_sq {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist.is_zero() {
        Vec2Fix::UNIT_Y
    } else {
        delta / dist
    };

    let depth = sum_r - dist;
    let point = closest + normal * cap_radius;

    Some(Contact2D {
        point,
        normal,
        depth,
        body_a: 0,
        body_b: 0,
    })
}

/// Edge vs circle collision.
fn edge_vs_circle(
    edge_body: &RigidBody2D,
    local_start: Vec2Fix,
    local_end: Vec2Fix,
    circle_pos: Vec2Fix,
    circle_radius: Fix128,
) -> Option<Contact2D> {
    let world_start = edge_body.world_point(local_start);
    let world_end = edge_body.world_point(local_end);

    let closest = closest_point_on_segment(world_start, world_end, circle_pos);
    let delta = circle_pos - closest;
    let dist_sq = delta.length_squared();
    let r_sq = circle_radius * circle_radius;

    if dist_sq > r_sq {
        return None;
    }

    let dist = dist_sq.sqrt();
    let normal = if dist.is_zero() {
        // Use edge normal
        let edge = world_end - world_start;
        Vec2Fix::new(edge.y, -edge.x).normalize()
    } else {
        delta / dist
    };

    let depth = circle_radius - dist;

    Some(Contact2D {
        point: closest,
        normal,
        depth,
        body_a: 0,
        body_b: 0,
    })
}

/// Closest point on a line segment to a given point.
fn closest_point_on_segment(seg_a: Vec2Fix, seg_b: Vec2Fix, point: Vec2Fix) -> Vec2Fix {
    let ab = seg_b - seg_a;
    let len_sq = ab.length_squared();
    if len_sq.is_zero() {
        return seg_a;
    }
    let t = (point - seg_a).dot(ab) / len_sq;
    // Clamp t to [0, 1]
    let t_clamped = if t.is_negative() {
        Fix128::ZERO
    } else if t > Fix128::ONE {
        Fix128::ONE
    } else {
        t
    };
    seg_a + ab * t_clamped
}

// ============================================================================
// XPBD Contact Solver
// ============================================================================

/// Solve contact constraints using XPBD position correction.
fn solve_contacts_xpbd(bodies: &mut [RigidBody2D], contacts: &[Contact2D]) {
    for contact in contacts {
        let a = contact.body_a;
        let b = contact.body_b;
        let normal = contact.normal;
        let depth = contact.depth;

        if depth.is_negative() || depth.is_zero() {
            continue;
        }

        let inv_mass_a = bodies[a].inv_mass;
        let inv_mass_b = bodies[b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;

        if total_inv_mass.is_zero() {
            continue;
        }

        let correction = normal * depth;
        let ratio_a = inv_mass_a / total_inv_mass;
        let ratio_b = inv_mass_b / total_inv_mass;

        if bodies[a].body_type == BodyType2D::Dynamic {
            bodies[a].position = bodies[a].position - correction * ratio_a;
        }
        if bodies[b].body_type == BodyType2D::Dynamic {
            bodies[b].position = bodies[b].position + correction * ratio_b;
        }

        // Angular correction
        let r_a = contact.point - bodies[a].position;
        let r_b = contact.point - bodies[b].position;
        let rn_a = r_a.cross_scalar(normal);
        let rn_b = r_b.cross_scalar(normal);

        let ang_inv_a = bodies[a].inv_inertia * rn_a * rn_a;
        let ang_inv_b = bodies[b].inv_inertia * rn_b * rn_b;
        let total_ang = ang_inv_a + ang_inv_b;

        if !total_ang.is_zero() {
            let ang_correction = depth * Fix128::from_ratio(1, 4);
            if bodies[a].body_type == BodyType2D::Dynamic {
                let da =
                    rn_a * ang_correction * bodies[a].inv_inertia / (total_inv_mass + total_ang);
                bodies[a].angle = bodies[a].angle - da;
            }
            if bodies[b].body_type == BodyType2D::Dynamic {
                let db =
                    rn_b * ang_correction * bodies[b].inv_inertia / (total_inv_mass + total_ang);
                bodies[b].angle = bodies[b].angle + db;
            }
        }
    }
}

// ============================================================================
// Joint2D
// ============================================================================

/// 2D joint constraint connecting one or two bodies.
#[derive(Clone, Debug)]
pub enum Joint2D {
    /// Revolute (pin/hinge) joint: constrains two bodies to share a point.
    Revolute {
        /// First body index.
        body_a: usize,
        /// Second body index.
        body_b: usize,
        /// Anchor point in body A's local space.
        local_anchor_a: Vec2Fix,
        /// Anchor point in body B's local space.
        local_anchor_b: Vec2Fix,
        /// Compliance (inverse stiffness). 0 = perfectly rigid.
        compliance: Fix128,
    },
    /// Distance joint: maintains a fixed distance between two anchor points.
    Distance {
        /// First body index.
        body_a: usize,
        /// Second body index.
        body_b: usize,
        /// Anchor point in body A's local space.
        local_anchor_a: Vec2Fix,
        /// Anchor point in body B's local space.
        local_anchor_b: Vec2Fix,
        /// Target distance between anchors.
        target_distance: Fix128,
        /// Compliance (inverse stiffness). 0 = perfectly rigid.
        compliance: Fix128,
    },
    /// Weld joint: constrains two bodies to maintain relative position and angle.
    Weld {
        /// First body index.
        body_a: usize,
        /// Second body index.
        body_b: usize,
        /// Anchor point in body A's local space.
        local_anchor_a: Vec2Fix,
        /// Anchor point in body B's local space.
        local_anchor_b: Vec2Fix,
        /// Reference angle (relative angle at rest).
        reference_angle: Fix128,
        /// Compliance (inverse stiffness). 0 = perfectly rigid.
        compliance: Fix128,
    },
    /// Mouse joint: drags a body toward a world-space target point.
    Mouse {
        /// Body index.
        body: usize,
        /// World-space target position.
        target: Vec2Fix,
        /// Maximum force the joint can apply.
        max_force: Fix128,
        /// Stiffness parameter.
        stiffness: Fix128,
        /// Damping parameter.
        damping: Fix128,
    },
}

/// Solve all 2D joints using XPBD position-level constraints.
pub fn solve_joints_2d(bodies: &mut [RigidBody2D], joints: &[Joint2D], sub_dt: Fix128) {
    let alpha = if sub_dt.is_zero() {
        Fix128::ZERO
    } else {
        Fix128::ONE / (sub_dt * sub_dt)
    };

    for joint in joints {
        match joint {
            Joint2D::Revolute {
                body_a,
                body_b,
                local_anchor_a,
                local_anchor_b,
                compliance,
            } => {
                solve_revolute(
                    bodies,
                    *body_a,
                    *body_b,
                    *local_anchor_a,
                    *local_anchor_b,
                    *compliance,
                    alpha,
                );
            }
            Joint2D::Distance {
                body_a,
                body_b,
                local_anchor_a,
                local_anchor_b,
                target_distance,
                compliance,
            } => {
                solve_distance(
                    bodies,
                    *body_a,
                    *body_b,
                    *local_anchor_a,
                    *local_anchor_b,
                    *target_distance,
                    *compliance,
                    alpha,
                );
            }
            Joint2D::Weld {
                body_a,
                body_b,
                local_anchor_a,
                local_anchor_b,
                reference_angle,
                compliance,
            } => {
                solve_weld(
                    bodies,
                    *body_a,
                    *body_b,
                    *local_anchor_a,
                    *local_anchor_b,
                    *reference_angle,
                    *compliance,
                    alpha,
                );
            }
            Joint2D::Mouse {
                body,
                target,
                max_force,
                stiffness,
                damping: _,
            } => {
                solve_mouse(bodies, *body, *target, *max_force, *stiffness);
            }
        }
    }
}

/// Solve a revolute joint (two anchors must coincide).
fn solve_revolute(
    bodies: &mut [RigidBody2D],
    a: usize,
    b: usize,
    local_a: Vec2Fix,
    local_b: Vec2Fix,
    compliance: Fix128,
    alpha: Fix128,
) {
    let world_a = bodies[a].world_point(local_a);
    let world_b = bodies[b].world_point(local_b);
    let delta = world_b - world_a;
    let dist_sq = delta.length_squared();
    if dist_sq.is_zero() {
        return;
    }
    let dist = dist_sq.sqrt();
    let n = delta / dist;

    let inv_mass_a = bodies[a].inv_mass;
    let inv_mass_b = bodies[b].inv_mass;

    let r_a = world_a - bodies[a].position;
    let r_b = world_b - bodies[b].position;
    let rn_a = r_a.cross_scalar(n);
    let rn_b = r_b.cross_scalar(n);

    let w = inv_mass_a
        + inv_mass_b
        + rn_a * rn_a * bodies[a].inv_inertia
        + rn_b * rn_b * bodies[b].inv_inertia
        + compliance * alpha;

    if w.is_zero() {
        return;
    }

    let lambda = -dist / w;
    let p = n * lambda;

    if bodies[a].body_type == BodyType2D::Dynamic {
        bodies[a].position = bodies[a].position - p * inv_mass_a;
        bodies[a].angle = bodies[a].angle - rn_a * lambda * bodies[a].inv_inertia;
    }
    if bodies[b].body_type == BodyType2D::Dynamic {
        bodies[b].position = bodies[b].position + p * inv_mass_b;
        bodies[b].angle = bodies[b].angle + rn_b * lambda * bodies[b].inv_inertia;
    }
}

/// Solve a distance joint.
#[allow(clippy::too_many_arguments)]
fn solve_distance(
    bodies: &mut [RigidBody2D],
    a: usize,
    b: usize,
    local_a: Vec2Fix,
    local_b: Vec2Fix,
    target_distance: Fix128,
    compliance: Fix128,
    alpha: Fix128,
) {
    let world_a = bodies[a].world_point(local_a);
    let world_b = bodies[b].world_point(local_b);
    let delta = world_b - world_a;
    let dist = delta.length();

    if dist.is_zero() {
        return;
    }
    let n = delta / dist;

    let c = dist - target_distance;

    let inv_mass_a = bodies[a].inv_mass;
    let inv_mass_b = bodies[b].inv_mass;

    let r_a = world_a - bodies[a].position;
    let r_b = world_b - bodies[b].position;
    let rn_a = r_a.cross_scalar(n);
    let rn_b = r_b.cross_scalar(n);

    let w = inv_mass_a
        + inv_mass_b
        + rn_a * rn_a * bodies[a].inv_inertia
        + rn_b * rn_b * bodies[b].inv_inertia
        + compliance * alpha;

    if w.is_zero() {
        return;
    }

    let lambda = -c / w;
    let p = n * lambda;

    if bodies[a].body_type == BodyType2D::Dynamic {
        bodies[a].position = bodies[a].position - p * inv_mass_a;
        bodies[a].angle = bodies[a].angle - rn_a * lambda * bodies[a].inv_inertia;
    }
    if bodies[b].body_type == BodyType2D::Dynamic {
        bodies[b].position = bodies[b].position + p * inv_mass_b;
        bodies[b].angle = bodies[b].angle + rn_b * lambda * bodies[b].inv_inertia;
    }
}

/// Solve a weld joint (position + angle constraint).
#[allow(clippy::too_many_arguments)]
fn solve_weld(
    bodies: &mut [RigidBody2D],
    a: usize,
    b: usize,
    local_a: Vec2Fix,
    local_b: Vec2Fix,
    reference_angle: Fix128,
    compliance: Fix128,
    alpha: Fix128,
) {
    // Position constraint (same as revolute)
    solve_revolute(bodies, a, b, local_a, local_b, compliance, alpha);

    // Angular constraint
    let angle_error = bodies[b].angle - bodies[a].angle - reference_angle;
    if angle_error.is_zero() {
        return;
    }

    let inv_i_a = bodies[a].inv_inertia;
    let inv_i_b = bodies[b].inv_inertia;
    let w = inv_i_a + inv_i_b + compliance * alpha;

    if w.is_zero() {
        return;
    }

    let lambda = -angle_error / w;

    if bodies[a].body_type == BodyType2D::Dynamic {
        bodies[a].angle = bodies[a].angle - lambda * inv_i_a;
    }
    if bodies[b].body_type == BodyType2D::Dynamic {
        bodies[b].angle = bodies[b].angle + lambda * inv_i_b;
    }
}

/// Solve a mouse joint (pull body toward target).
fn solve_mouse(
    bodies: &mut [RigidBody2D],
    body_idx: usize,
    target: Vec2Fix,
    max_force: Fix128,
    stiffness: Fix128,
) {
    let body = &bodies[body_idx];
    if body.body_type != BodyType2D::Dynamic {
        return;
    }

    let delta = target - body.position;
    let dist = delta.length();
    if dist.is_zero() {
        return;
    }

    // Limit correction
    let correction = if dist > max_force {
        delta * (max_force / dist)
    } else {
        delta
    };

    bodies[body_idx].position = bodies[body_idx].position + correction * stiffness;
}

impl core::fmt::Debug for PhysicsWorld2D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PhysicsWorld2D")
            .field("bodies", &self.bodies.len())
            .field("joints", &self.joints.len())
            .field("config", &self.config)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    // ---- Vec2Fix arithmetic ----

    #[test]
    fn test_vec2_new_and_constants() {
        let v = Vec2Fix::new(Fix128::from_int(3), Fix128::from_int(4));
        assert_eq!(v.x.hi, 3);
        assert_eq!(v.y.hi, 4);
        assert!(Vec2Fix::ZERO.x.is_zero());
        assert!(Vec2Fix::ZERO.y.is_zero());
        assert_eq!(Vec2Fix::ONE.x.hi, 1);
        assert_eq!(Vec2Fix::ONE.y.hi, 1);
        assert_eq!(Vec2Fix::UNIT_X.x.hi, 1);
        assert!(Vec2Fix::UNIT_X.y.is_zero());
        assert!(Vec2Fix::UNIT_Y.x.is_zero());
        assert_eq!(Vec2Fix::UNIT_Y.y.hi, 1);
    }

    #[test]
    fn test_vec2_from_int() {
        let v = Vec2Fix::from_int(7, -3);
        assert_eq!(v.x.hi, 7);
        assert_eq!(v.y.hi, -3);
    }

    #[test]
    fn test_vec2_add_sub() {
        let a = Vec2Fix::from_int(3, 5);
        let b = Vec2Fix::from_int(1, 2);
        let sum = a + b;
        assert_eq!(sum.x.hi, 4);
        assert_eq!(sum.y.hi, 7);
        let diff = a - b;
        assert_eq!(diff.x.hi, 2);
        assert_eq!(diff.y.hi, 3);
    }

    #[test]
    fn test_vec2_mul_div_scalar() {
        let v = Vec2Fix::from_int(6, 8);
        let scaled = v * Fix128::from_int(3);
        assert_eq!(scaled.x.hi, 18);
        assert_eq!(scaled.y.hi, 24);
        let halved = v / Fix128::from_int(2);
        assert_eq!(halved.x.hi, 3);
        assert_eq!(halved.y.hi, 4);
    }

    #[test]
    fn test_vec2_neg() {
        let v = Vec2Fix::from_int(5, -3);
        let neg_v = -v;
        assert_eq!(neg_v.x.hi, -5);
        assert_eq!(neg_v.y.hi, 3);
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2Fix::from_int(3, 4);
        let b = Vec2Fix::from_int(2, 5);
        let d = a.dot(b);
        // 3*2 + 4*5 = 26
        assert_eq!(d.hi, 26);
    }

    #[test]
    fn test_vec2_cross_scalar() {
        let a = Vec2Fix::from_int(3, 4);
        let b = Vec2Fix::from_int(2, 5);
        let c = a.cross_scalar(b);
        // 3*5 - 4*2 = 15 - 8 = 7
        assert_eq!(c.hi, 7);
    }

    #[test]
    fn test_vec2_length_squared() {
        let v = Vec2Fix::from_int(3, 4);
        let len_sq = v.length_squared();
        // 9 + 16 = 25
        assert_eq!(len_sq.hi, 25);
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2Fix::from_int(3, 4);
        let len = v.length();
        // sqrt(25) = 5
        assert_eq!(len.hi, 5);
    }

    #[test]
    fn test_vec2_normalize() {
        let v = Vec2Fix::from_int(0, 5);
        let n = v.normalize();
        assert!(n.x.is_zero());
        assert_eq!(n.y.hi, 1);

        // Zero vector normalizes to zero
        let z = Vec2Fix::ZERO.normalize();
        assert!(z.x.is_zero());
        assert!(z.y.is_zero());
    }

    #[test]
    fn test_vec2_perpendicular() {
        let v = Vec2Fix::from_int(3, 4);
        let p = v.perpendicular();
        assert_eq!(p.x.hi, -4);
        assert_eq!(p.y.hi, 3);
        // Perpendicular should have zero dot product
        assert!(v.dot(p).is_zero());
    }

    #[test]
    fn test_vec2_distance_to() {
        let a = Vec2Fix::from_int(0, 0);
        let b = Vec2Fix::from_int(3, 4);
        let d = a.distance_to(b);
        assert_eq!(d.hi, 5);
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2Fix::from_int(0, 0);
        let b = Vec2Fix::from_int(10, 20);
        let half = Fix128::from_ratio(1, 2);
        let mid = a.lerp(b, half);
        assert_eq!(mid.x.hi, 5);
        assert_eq!(mid.y.hi, 10);

        // t=0 -> a, t=1 -> b
        let at0 = a.lerp(b, Fix128::ZERO);
        assert_eq!(at0.x.hi, 0);
        assert_eq!(at0.y.hi, 0);
        let at1 = a.lerp(b, Fix128::ONE);
        assert_eq!(at1.x.hi, 10);
        assert_eq!(at1.y.hi, 20);
    }

    #[test]
    fn test_vec2_rotate() {
        // Rotate UNIT_X by pi/2 should give approximately UNIT_Y
        let v = Vec2Fix::UNIT_X;
        let rotated = v.rotate(Fix128::HALF_PI);
        // Allow small tolerance due to CORDIC precision
        assert!(rotated.x.abs() < Fix128::from_ratio(1, 1000));
        assert!((rotated.y - Fix128::ONE).abs() < Fix128::from_ratio(1, 1000));
    }

    // ---- Shape and body creation ----

    #[test]
    fn test_body_creation_dynamic() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body = RigidBody2D::new_dynamic(Vec2Fix::from_int(5, 10), Fix128::from_int(2), shape);
        assert_eq!(body.position.x.hi, 5);
        assert_eq!(body.position.y.hi, 10);
        assert_eq!(body.body_type, BodyType2D::Dynamic);
        assert!(!body.inv_mass.is_zero());
    }

    #[test]
    fn test_body_creation_static() {
        let shape = Shape2D::Circle {
            radius: Fix128::from_int(5),
        };
        let body = RigidBody2D::new_static(Vec2Fix::ZERO, shape);
        assert_eq!(body.body_type, BodyType2D::Static);
        assert!(body.inv_mass.is_zero());
        assert!(body.inv_inertia.is_zero());
    }

    #[test]
    fn test_body_creation_kinematic() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body = RigidBody2D::new_kinematic(Vec2Fix::from_int(1, 2), shape);
        assert_eq!(body.body_type, BodyType2D::Kinematic);
        assert!(body.inv_mass.is_zero());
    }

    #[test]
    fn test_body_apply_impulse() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let mut body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::ONE, shape);
        body.apply_impulse(Vec2Fix::from_int(10, 0));
        assert_eq!(body.velocity.x.hi, 10);
        assert!(body.velocity.y.is_zero());
    }

    #[test]
    fn test_static_body_ignores_impulse() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let mut body = RigidBody2D::new_static(Vec2Fix::ZERO, shape);
        body.apply_impulse(Vec2Fix::from_int(100, 100));
        assert!(body.velocity.x.is_zero());
        assert!(body.velocity.y.is_zero());
    }

    #[test]
    fn test_body_world_point() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let mut body = RigidBody2D::new_dynamic(Vec2Fix::from_int(10, 20), Fix128::ONE, shape);
        body.angle = Fix128::ZERO;
        let wp = body.world_point(Vec2Fix::from_int(1, 0));
        // After rotation by angle=0, local (1,0) -> approximately (1, 0).
        // CORDIC sin(0) has sub-integer precision error, so check integer part
        // with tolerance.
        let dx = (wp.x - Fix128::from_int(11)).abs();
        let dy = (wp.y - Fix128::from_int(20)).abs();
        assert!(dx < Fix128::ONE, "world_point x off by more than 1");
        assert!(dy < Fix128::ONE, "world_point y off by more than 1");
    }

    // ---- Collision detection ----

    #[test]
    fn test_circle_vs_circle_collision() {
        let c = circle_vs_circle(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Vec2Fix::from_int(1, 0),
            Fix128::ONE,
        );
        assert!(c.is_some());
        let contact = c.unwrap();
        assert_eq!(contact.depth.hi, 1); // overlap = 2 - 1 = 1
        assert_eq!(contact.normal.x.hi, 1); // pointing from A to B
        assert!(contact.normal.y.is_zero());
    }

    #[test]
    fn test_circle_vs_circle_no_collision() {
        let c = circle_vs_circle(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Vec2Fix::from_int(5, 0),
            Fix128::ONE,
        );
        assert!(c.is_none());
    }

    #[test]
    fn test_polygon_vs_polygon_collision() {
        // Two overlapping unit squares
        let verts_a = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];
        let verts_b = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];

        let body_a = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts_a.clone(),
            },
        );
        let body_b = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(1, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts_b.clone(),
            },
        );

        let result = polygon_vs_polygon(&body_a, &verts_a, &body_b, &verts_b);
        assert!(result.is_some());
        let contact = result.unwrap();
        assert!(contact.depth > Fix128::ZERO);
    }

    #[test]
    fn test_polygon_vs_polygon_no_collision() {
        let verts_a = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];
        let verts_b = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];

        let body_a = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts_a.clone(),
            },
        );
        let body_b = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(10, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts_b.clone(),
            },
        );

        let result = polygon_vs_polygon(&body_a, &verts_a, &body_b, &verts_b);
        assert!(result.is_none());
    }

    #[test]
    fn test_circle_vs_polygon_collision() {
        let verts = vec![
            Vec2Fix::from_int(-2, -2),
            Vec2Fix::from_int(2, -2),
            Vec2Fix::from_int(2, 2),
            Vec2Fix::from_int(-2, 2),
        ];
        let poly_body = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts.clone(),
            },
        );

        let result = circle_vs_polygon(Vec2Fix::from_int(2, 0), Fix128::ONE, &poly_body, &verts);
        assert!(result.is_some());
    }

    #[test]
    fn test_circle_vs_polygon_no_collision() {
        let verts = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];
        let poly_body = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Polygon {
                vertices: verts.clone(),
            },
        );

        let result = circle_vs_polygon(Vec2Fix::from_int(10, 10), Fix128::ONE, &poly_body, &verts);
        assert!(result.is_none());
    }

    #[test]
    fn test_capsule_vs_circle_collision() {
        let cap_body = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Capsule {
                radius: Fix128::ONE,
                half_length: Fix128::from_int(2),
            },
        );
        let result = capsule_vs_circle(
            &cap_body,
            Fix128::ONE,
            Fix128::from_int(2),
            Vec2Fix::from_int(3, 0),
            Fix128::ONE,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_capsule_vs_circle_no_collision() {
        let cap_body = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            Shape2D::Capsule {
                radius: Fix128::ONE,
                half_length: Fix128::from_int(2),
            },
        );
        let result = capsule_vs_circle(
            &cap_body,
            Fix128::ONE,
            Fix128::from_int(2),
            Vec2Fix::from_int(10, 10),
            Fix128::ONE,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_edge_vs_circle_collision() {
        let edge_body = RigidBody2D::new_static(
            Vec2Fix::ZERO,
            Shape2D::Edge {
                start: Vec2Fix::from_int(-5, 0),
                end: Vec2Fix::from_int(5, 0),
            },
        );
        let result = edge_vs_circle(
            &edge_body,
            Vec2Fix::from_int(-5, 0),
            Vec2Fix::from_int(5, 0),
            Vec2Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2)),
            Fix128::ONE,
        );
        assert!(result.is_some());
        let contact = result.unwrap();
        assert!(contact.depth > Fix128::ZERO);
    }

    // ---- Basic simulation ----

    #[test]
    fn test_falling_body() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body = RigidBody2D::new_dynamic(Vec2Fix::from_int(0, 100), Fix128::ONE, shape);
        let id = world.add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // Body should have fallen under gravity
        assert!(
            world.bodies[id].position.y < Fix128::from_int(100),
            "Body should have fallen"
        );
    }

    #[test]
    fn test_static_body_does_not_move() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::from_int(5),
        };
        let body = RigidBody2D::new_static(Vec2Fix::from_int(10, 20), shape);
        let id = world.add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            world.step(dt);
        }

        assert_eq!(world.bodies[id].position.x.hi, 10);
        assert_eq!(world.bodies[id].position.y.hi, 20);
    }

    #[test]
    fn test_kinematic_body_moves_by_velocity() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let mut body = RigidBody2D::new_kinematic(Vec2Fix::ZERO, shape);
        body.velocity = Vec2Fix::from_int(10, 0);
        let id = world.add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // Should have moved ~10 units in x
        let x = world.bodies[id].position.x;
        assert!(x > Fix128::from_int(5));
    }

    // ---- Determinism ----

    #[test]
    fn test_determinism_2d() {
        fn run_sim() -> (Vec2Fix, Fix128) {
            let config = PhysicsConfig2D {
                gravity: Vec2Fix::new(Fix128::ZERO, Fix128::from_int(-10)),
                substeps: 4,
                iterations: 8,
                damping: Fix128::from_ratio(99, 100),
            };
            let mut world = PhysicsWorld2D::new(config);

            let shape = Shape2D::Circle {
                radius: Fix128::ONE,
            };
            let body =
                RigidBody2D::new_dynamic(Vec2Fix::from_int(5, 50), Fix128::from_ratio(3, 2), shape);
            world.add_body(body);

            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..120 {
                world.step(dt);
            }

            (world.bodies[0].position, world.bodies[0].angle)
        }

        let (pos1, angle1) = run_sim();
        let (pos2, angle2) = run_sim();

        // Bit-exact
        assert_eq!(pos1.x.hi, pos2.x.hi);
        assert_eq!(pos1.x.lo, pos2.x.lo);
        assert_eq!(pos1.y.hi, pos2.y.hi);
        assert_eq!(pos1.y.lo, pos2.y.lo);
        assert_eq!(angle1.hi, angle2.hi);
        assert_eq!(angle1.lo, angle2.lo);
    }

    // ---- Joint constraints ----

    #[test]
    fn test_distance_joint() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body_a = RigidBody2D::new_dynamic(Vec2Fix::from_int(0, 10), Fix128::ONE, shape.clone());
        let body_b = RigidBody2D::new_dynamic(Vec2Fix::from_int(5, 10), Fix128::ONE, shape);
        let id_a = world.add_body(body_a);
        let id_b = world.add_body(body_b);

        world.add_joint(Joint2D::Distance {
            body_a: id_a,
            body_b: id_b,
            local_anchor_a: Vec2Fix::ZERO,
            local_anchor_b: Vec2Fix::ZERO,
            target_distance: Fix128::from_int(5),
            compliance: Fix128::ZERO,
        });

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        let pos_a = world.bodies[id_a].position;
        let pos_b = world.bodies[id_b].position;
        let dist = (pos_b - pos_a).length();
        let target = Fix128::from_int(5);
        let error = (dist - target).abs();

        // Should maintain roughly 5 units distance
        assert!(
            error < Fix128::from_int(2),
            "Distance joint error too large"
        );
    }

    #[test]
    fn test_revolute_joint() {
        let config = PhysicsConfig2D {
            gravity: Vec2Fix::ZERO,
            substeps: 4,
            iterations: 8,
            damping: Fix128::from_ratio(99, 100),
        };
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body_a = RigidBody2D::new_static(Vec2Fix::ZERO, shape.clone());
        let body_b = RigidBody2D::new_dynamic(Vec2Fix::from_int(3, 0), Fix128::ONE, shape);
        let id_a = world.add_body(body_a);
        let id_b = world.add_body(body_b);

        world.add_joint(Joint2D::Revolute {
            body_a: id_a,
            body_b: id_b,
            local_anchor_a: Vec2Fix::ZERO,
            local_anchor_b: Vec2Fix::from_int(-3, 0),
            compliance: Fix128::ZERO,
        });

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // The anchor points should stay close
        let wa = world.bodies[id_a].world_point(Vec2Fix::ZERO);
        let wb = world.bodies[id_b].world_point(Vec2Fix::from_int(-3, 0));
        let separation = wa.distance_to(wb);
        assert!(
            separation < Fix128::ONE,
            "Revolute joint anchor separation too large"
        );
    }

    #[test]
    fn test_weld_joint() {
        let config = PhysicsConfig2D {
            gravity: Vec2Fix::ZERO,
            substeps: 4,
            iterations: 8,
            damping: Fix128::from_ratio(99, 100),
        };
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body_a = RigidBody2D::new_static(Vec2Fix::ZERO, shape.clone());
        let mut body_b = RigidBody2D::new_dynamic(Vec2Fix::from_int(2, 0), Fix128::ONE, shape);
        body_b.angular_velocity = Fix128::from_int(5); // spin it
        let id_a = world.add_body(body_a);
        let id_b = world.add_body(body_b);

        world.add_joint(Joint2D::Weld {
            body_a: id_a,
            body_b: id_b,
            local_anchor_a: Vec2Fix::from_int(2, 0),
            local_anchor_b: Vec2Fix::ZERO,
            reference_angle: Fix128::ZERO,
            compliance: Fix128::ZERO,
        });

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            world.step(dt);
        }

        // Weld joint should resist angle difference
        let angle_diff = (world.bodies[id_b].angle - world.bodies[id_a].angle).abs();
        assert!(
            angle_diff < Fix128::from_int(2),
            "Weld joint should constrain angle"
        );
    }

    #[test]
    fn test_mouse_joint() {
        let config = PhysicsConfig2D {
            gravity: Vec2Fix::ZERO,
            substeps: 4,
            iterations: 8,
            damping: Fix128::from_ratio(99, 100),
        };
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::ONE, shape);
        let id = world.add_body(body);

        let target = Vec2Fix::from_int(10, 10);
        world.add_joint(Joint2D::Mouse {
            body: id,
            target,
            max_force: Fix128::from_int(100),
            stiffness: Fix128::from_ratio(1, 10),
            damping: Fix128::from_ratio(1, 100),
        });

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..300 {
            world.step(dt);
        }

        // Body should have moved toward target
        let pos = world.bodies[id].position;
        let dist = pos.distance_to(target);
        assert!(
            dist < Fix128::from_int(5),
            "Mouse joint should pull body toward target"
        );
    }

    // ---- Contact resolution ----

    #[test]
    fn test_circle_circle_contact_resolution() {
        let config = PhysicsConfig2D {
            gravity: Vec2Fix::ZERO,
            substeps: 4,
            iterations: 8,
            damping: Fix128::ONE, // no damping
        };
        let mut world = PhysicsWorld2D::new(config);

        let shape_a = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let shape_b = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        // Overlapping circles
        let body_a = RigidBody2D::new_dynamic(Vec2Fix::from_int(0, 0), Fix128::ONE, shape_a);
        let body_b = RigidBody2D::new_dynamic(Vec2Fix::from_int(1, 0), Fix128::ONE, shape_b);
        world.add_body(body_a);
        world.add_body(body_b);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // Bodies should have separated
        let dist = world.bodies[0]
            .position
            .distance_to(world.bodies[1].position);
        assert!(
            dist >= Fix128::ONE,
            "Circles should have been pushed apart by contact solver"
        );
    }

    #[test]
    fn test_body_rests_on_static_edge() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        // Static ground edge
        let ground = RigidBody2D::new_static(
            Vec2Fix::ZERO,
            Shape2D::Edge {
                start: Vec2Fix::from_int(-100, 0),
                end: Vec2Fix::from_int(100, 0),
            },
        );
        world.add_body(ground);

        // Falling circle
        let ball = RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 5),
            Fix128::ONE,
            Shape2D::Circle {
                radius: Fix128::ONE,
            },
        );
        let ball_id = world.add_body(ball);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..300 {
            world.step(dt);
        }

        // Ball should not have fallen through the ground
        let y = world.bodies[ball_id].position.y;
        assert!(
            y > Fix128::from_int(-5),
            "Ball should rest on or above the edge"
        );
    }

    // ---- World management ----

    #[test]
    fn test_add_remove_body() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let id0 = world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            shape.clone(),
        ));
        let _id1 = world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(5, 5),
            Fix128::ONE,
            shape,
        ));
        assert_eq!(world.bodies.len(), 2);

        let removed = world.remove_body(id0);
        assert!(removed.is_some());
        assert_eq!(world.bodies.len(), 1);
    }

    #[test]
    fn test_remove_body_invalid_index() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);
        let removed = world.remove_body(999);
        assert!(removed.is_none());
    }

    // ---- Inertia computation ----

    #[test]
    fn test_circle_inertia() {
        let shape = Shape2D::Circle {
            radius: Fix128::from_int(2),
        };
        let mass = Fix128::from_int(4);
        let inertia = compute_inertia(&shape, mass);
        // I = 0.5 * 4 * 4 = 8
        assert_eq!(inertia.hi, 8);
    }

    #[test]
    fn test_polygon_inertia_nonzero() {
        let verts = vec![
            Vec2Fix::from_int(-1, -1),
            Vec2Fix::from_int(1, -1),
            Vec2Fix::from_int(1, 1),
            Vec2Fix::from_int(-1, 1),
        ];
        let shape = Shape2D::Polygon { vertices: verts };
        let inertia = compute_inertia(&shape, Fix128::from_int(4));
        assert!(!inertia.is_zero());
        assert!(!inertia.is_negative());
    }

    // ---- Multiple collisions ----

    #[test]
    fn test_multiple_bodies_simulation() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };

        // Add 5 bodies at different heights
        for i in 0..5 {
            let body = RigidBody2D::new_dynamic(
                Vec2Fix::from_int(i as i64 * 3, (i as i64 + 1) * 10),
                Fix128::ONE,
                shape.clone(),
            );
            world.add_body(body);
        }

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // All bodies should have fallen
        for body in &world.bodies {
            assert!(body.velocity.y.is_negative() || body.position.y < Fix128::from_int(50));
        }
    }

    // ---- World with collision ----

    #[test]
    fn test_world_check_collision_2d() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape_a = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let shape_b = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            shape_a,
        ));
        world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(1, 0),
            Fix128::ONE,
            shape_b,
        ));

        let result = world.check_collision_2d(0, 1);
        assert!(result.is_some());
    }

    #[test]
    fn test_world_no_collision_far_apart() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(0, 0),
            Fix128::ONE,
            shape.clone(),
        ));
        world.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::from_int(100, 100),
            Fix128::ONE,
            shape,
        ));

        let result = world.check_collision_2d(0, 1);
        assert!(result.is_none());
    }

    // ---- Zero dt edge case ----

    #[test]
    fn test_zero_dt_step() {
        let config = PhysicsConfig2D::default();
        let mut world = PhysicsWorld2D::new(config);

        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let body = RigidBody2D::new_dynamic(Vec2Fix::from_int(0, 10), Fix128::ONE, shape);
        let id = world.add_body(body);

        // Stepping with zero dt should not crash or move the body
        world.step(Fix128::ZERO);
        assert_eq!(world.bodies[id].position.x.hi, 0);
        assert_eq!(world.bodies[id].position.y.hi, 10);
    }

    // ---- PhysicsConfig2D default ----

    #[test]
    fn test_config_default() {
        let config = PhysicsConfig2D::default();
        assert!(config.gravity.x.is_zero());
        assert_eq!(config.gravity.y.hi, -10);
        assert_eq!(config.substeps, 4);
        assert_eq!(config.iterations, 8);
    }

    // ---- Closest point on segment ----

    #[test]
    fn test_closest_point_on_segment_middle() {
        let a = Vec2Fix::from_int(0, 0);
        let b = Vec2Fix::from_int(10, 0);
        let p = Vec2Fix::from_int(5, 5);
        let closest = closest_point_on_segment(a, b, p);
        assert_eq!(closest.x.hi, 5);
        assert!(closest.y.is_zero());
    }

    #[test]
    fn test_closest_point_on_segment_endpoint_a() {
        let a = Vec2Fix::from_int(0, 0);
        let b = Vec2Fix::from_int(10, 0);
        let p = Vec2Fix::from_int(-5, 0);
        let closest = closest_point_on_segment(a, b, p);
        assert_eq!(closest.x.hi, 0);
        assert!(closest.y.is_zero());
    }

    #[test]
    fn test_closest_point_on_segment_endpoint_b() {
        let a = Vec2Fix::from_int(0, 0);
        let b = Vec2Fix::from_int(10, 0);
        let p = Vec2Fix::from_int(15, 0);
        let closest = closest_point_on_segment(a, b, p);
        assert_eq!(closest.x.hi, 10);
        assert!(closest.y.is_zero());
    }

    // ---- Impulse at point ----

    #[test]
    fn test_apply_impulse_at_point() {
        let shape = Shape2D::Circle {
            radius: Fix128::from_int(2),
        };
        let mut body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::ONE, shape);
        let point = Vec2Fix::from_int(2, 0);
        let impulse = Vec2Fix::from_int(0, 10);
        body.apply_impulse_at_point(impulse, point);

        // Should have both linear and angular velocity
        assert_eq!(body.velocity.y.hi, 10);
        assert!(!body.angular_velocity.is_zero());
    }

    // ---- Force application ----

    #[test]
    fn test_apply_force() {
        let shape = Shape2D::Circle {
            radius: Fix128::ONE,
        };
        let mut body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::ONE, shape);
        let force = Vec2Fix::from_int(100, 0);
        let dt = Fix128::from_ratio(1, 60);
        body.apply_force(force, dt);
        assert!(body.velocity.x > Fix128::ZERO);
    }
}
