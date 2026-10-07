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
use alloc::{vec, vec::Vec};

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
    ///
    /// Zero thickness and two-sided: it collides with circles, capsules and
    /// polygons from either side, but not with other edges (two segments
    /// enclose no area, so parallel edges never overlap).
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub const fn new(config: PhysicsConfig2D) -> Self {
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
        self.step_with_tethers(dt, &Tethers2D::new());
    }

    /// [`step`](Self::step) that also solves the angular tethers and kinematic drives in
    /// `tethers` inside every substep (drives right after the velocity integration,
    /// tethers in the constraint iterations next to the joints). With an empty set this is
    /// `step`, bit for bit.
    pub fn step_with_tethers(&mut self, dt: Fix128, tethers: &Tethers2D) {
        let substeps = self.config.substeps;
        if substeps == 0 {
            return;
        }
        let sub_dt = dt / Fix128::from_int(substeps as i64);

        for _ in 0..substeps {
            self.substep(sub_dt, tethers);
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
    fn substep(&mut self, sub_dt: Fix128, tethers: &Tethers2D) {
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
        for drive in tethers.live_drives() {
            apply_kinematic_drive(&mut self.bodies, drive, sub_dt);
        }

        // 2. Detect collisions (pairs + the pre-solve normal velocity each
        //    pair approaches with, which the restitution pass needs)
        let contacts = self.detect_all_contacts();
        let pre_normal_velocity: Vec<Fix128> = contacts
            .iter()
            .map(|c| self.contact_normal_velocity(c))
            .collect();
        let mut lambda_n = vec![Fix128::ZERO; contacts.len()];
        let mut joint_lambda = vec![Vec2Fix::ZERO; self.joints.len()];
        let mut tether_lambda = vec![Fix128::ZERO; tethers.angular.len()];

        // 3. Solve constraints (position-based). The penetration is
        //    re-evaluated from the current positions in every iteration
        //    (Macklin 2016 / 2020) instead of re-applying the depth measured
        //    before the solve: before 1.2.0 each iteration pushed the same
        //    stale depth again, so `iterations = 8` separated a 2 m/s head-on
        //    collision at 14 m/s.
        for _ in 0..iterations {
            for (k, pair) in contacts.iter().enumerate() {
                let Some(current) = self.check_collision_2d(pair.body_a, pair.body_b) else {
                    continue;
                };
                lambda_n[k] = lambda_n[k] + solve_contact_position(&mut self.bodies, &current);
            }

            // Solve joints
            solve_joints_2d_accumulated(&mut self.bodies, &self.joints, sub_dt, &mut joint_lambda);

            for (slot, lambda) in tethers.angular.iter().zip(tether_lambda.iter_mut()) {
                if let Some(tether) = slot {
                    solve_angular_tether(&mut self.bodies, tether, sub_dt, lambda);
                }
            }
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

        // 5. Velocity pass (Macklin et al. 2020 "Detailed rigid body
        //    simulation with XPBD" §3.6): Newton restitution
        //    `v_n' = max(−e v̄_n, 0)` against the pre-solve approach speed
        //    and Coulomb friction `|Δv_t| ≤ μ λ_n / h`. Before 1.2.0
        //    `restitution` / `friction` were never read.
        let restitution_threshold = self.gravity.length() * sub_dt * Fix128::from_int(2);
        for (k, pair) in contacts.iter().enumerate() {
            if lambda_n[k] <= Fix128::ZERO {
                continue;
            }
            self.apply_contact_velocity_pass(
                pair,
                pre_normal_velocity[k],
                lambda_n[k] * inv_dt,
                restitution_threshold,
            );
        }
    }

    /// Relative normal velocity of the two bodies at the contact point,
    /// `(v_b − v_a) · n` with `n` pointing from A to B (negative while
    /// approaching).
    fn contact_normal_velocity(&self, c: &Contact2D) -> Fix128 {
        let (va, vb) = (
            self.point_velocity(c.body_a, c.point),
            self.point_velocity(c.body_b, c.point),
        );
        (vb - va).dot(c.normal)
    }

    /// Velocity of a body's material point at world position `p`.
    fn point_velocity(&self, body: usize, p: Vec2Fix) -> Vec2Fix {
        let b = &self.bodies[body];
        let r = p - b.position;
        b.velocity + r.perpendicular() * b.angular_velocity
    }

    /// Restitution + friction impulses for one contact after the velocity
    /// derivation. `friction_bound` is `μ λ_n / h` before the `μ` factor
    /// (i.e. `λ_n / h`), `pre_vn` the approach speed before the solve.
    fn apply_contact_velocity_pass(
        &mut self,
        c: &Contact2D,
        pre_vn: Fix128,
        lambda_n_over_h: Fix128,
        restitution_threshold: Fix128,
    ) {
        let (a, b, n) = (c.body_a, c.body_b, c.normal);
        let dyn_a = self.bodies[a].body_type == BodyType2D::Dynamic;
        let dyn_b = self.bodies[b].body_type == BodyType2D::Dynamic;
        if !dyn_a && !dyn_b {
            return;
        }
        let r_a = c.point - self.bodies[a].position;
        let r_b = c.point - self.bodies[b].position;
        let inv_m_a = if dyn_a {
            self.bodies[a].inv_mass
        } else {
            Fix128::ZERO
        };
        let inv_m_b = if dyn_b {
            self.bodies[b].inv_mass
        } else {
            Fix128::ZERO
        };
        let inv_i_a = if dyn_a {
            self.bodies[a].inv_inertia
        } else {
            Fix128::ZERO
        };
        let inv_i_b = if dyn_b {
            self.bodies[b].inv_inertia
        } else {
            Fix128::ZERO
        };
        // generalised inverse mass along a direction d at the contact point
        let w_along = |d: Vec2Fix| {
            let ra_d = r_a.cross_scalar(d);
            let rb_d = r_b.cross_scalar(d);
            inv_m_a + inv_m_b + inv_i_a * ra_d * ra_d + inv_i_b * rb_d * rb_d
        };
        // apply an impulse `p` along `d` (+ on B, − on A)
        let apply = |bodies: &mut [RigidBody2D], d: Vec2Fix, p: Fix128| {
            if dyn_a {
                bodies[a].velocity = bodies[a].velocity - d * (p * inv_m_a);
                bodies[a].angular_velocity =
                    bodies[a].angular_velocity - r_a.cross_scalar(d) * p * inv_i_a;
            }
            if dyn_b {
                bodies[b].velocity = bodies[b].velocity + d * (p * inv_m_b);
                bodies[b].angular_velocity =
                    bodies[b].angular_velocity + r_b.cross_scalar(d) * p * inv_i_b;
            }
        };

        // --- restitution ---
        let v_rel = self.point_velocity(b, c.point) - self.point_velocity(a, c.point);
        let vn = v_rel.dot(n);
        // only a genuine approach bounces; resting contacts (|v̄_n| below
        // 2 g h) get e = 0 so stacks do not jitter
        let e = if pre_vn < -restitution_threshold {
            let ea = self.bodies[a].restitution;
            let eb = self.bodies[b].restitution;
            if ea > eb {
                ea
            } else {
                eb
            }
        } else {
            Fix128::ZERO
        };
        let target = -e * pre_vn;
        let target = if target.is_negative() {
            Fix128::ZERO
        } else {
            target
        };
        let w_n = w_along(n);
        if vn < target && !w_n.is_zero() {
            apply(&mut self.bodies, n, (target - vn) / w_n);
        }

        // --- Coulomb friction ---
        let v_rel = self.point_velocity(b, c.point) - self.point_velocity(a, c.point);
        let vt = v_rel - n * v_rel.dot(n);
        let vt_len = vt.length();
        if vt_len.is_zero() {
            return;
        }
        let mu = (self.bodies[a].friction * self.bodies[b].friction).sqrt();
        let max_dv = mu * lambda_n_over_h;
        let dv = if vt_len < max_dv { vt_len } else { max_dv };
        if dv.is_zero() {
            return;
        }
        let t = vt * (Fix128::ONE / vt_len);
        let w_t = w_along(t);
        if !w_t.is_zero() {
            apply(&mut self.bodies, t, -dv / w_t);
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
            ) => {
                // capsule_vs_circle returns capsule -> circle; here A is the circle
                capsule_vs_circle(body_b, *rcap, *half_length, body_a.position, *rc).map(|mut c| {
                    c.normal = -c.normal;
                    c
                })
            }
            (
                Shape2D::Capsule {
                    radius: rcap,
                    half_length,
                },
                Shape2D::Circle { radius: rc },
            ) => capsule_vs_circle(body_a, *rcap, *half_length, body_b.position, *rc),
            (Shape2D::Edge { start, end }, Shape2D::Circle { radius }) => {
                edge_vs_circle(body_a, *start, *end, body_b.position, *radius)
            }
            (Shape2D::Circle { radius }, Shape2D::Edge { start, end }) => {
                edge_vs_circle(body_b, *start, *end, body_a.position, *radius).map(|mut c| {
                    c.normal = -c.normal;
                    c
                })
            }
            // Two zero-thickness segments enclose no area: parallel edges never
            // overlap (a falling edge passes a parallel one between substeps)
            // and crossing edges have no orientation-independent penetration,
            // so edges deliberately do not collide with each other.
            (Shape2D::Edge { .. }, Shape2D::Edge { .. }) => None,
            (
                Shape2D::Capsule { .. } | Shape2D::Edge { .. } | Shape2D::Polygon { .. },
                Shape2D::Capsule { .. } | Shape2D::Edge { .. } | Shape2D::Polygon { .. },
            ) => match (rounded_core(body_a), rounded_core(body_b)) {
                (Some((core_a, ra)), Some((core_b, rb))) => {
                    rounded_core_vs_rounded_core(&core_a, ra, &core_b, rb)
                }
                _ => None,
            },
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
///
/// The returned normal points from the circle (first argument) toward the
/// polygon, i.e. against the polygon's outward face / vertex direction, so the
/// `(Circle, Polygon)` arm of `check_collision_2d` uses it as body_a -> body_b.
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
            normal: -normal,
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
            normal: -normal,
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
            normal: -best_normal,
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
/// The returned normal points from the capsule (first argument) toward the
/// circle. The capsule is centered at `body_cap.position` with its segment along the
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
///
/// The returned normal points from the edge (first argument) toward the circle.
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

/// World-space core and radius of a capsule, edge or convex polygon: the shape
/// is the core (a segment, or a CCW convex polygon) inflated by the radius.
///
/// A capsule is its local-X segment `[-half_length, +half_length]` inflated by
/// its radius, an edge its segment with radius 0 (zero thickness, two-sided),
/// a polygon its vertices with radius 0. Circles are handled by the dedicated
/// pair functions and return `None`, as do polygons with fewer than 3 vertices.
fn rounded_core(body: &RigidBody2D) -> Option<(Vec<Vec2Fix>, Fix128)> {
    match &body.shape {
        Shape2D::Capsule {
            radius,
            half_length,
        } => Some((
            vec![
                body.world_point(Vec2Fix::new(-*half_length, Fix128::ZERO)),
                body.world_point(Vec2Fix::new(*half_length, Fix128::ZERO)),
            ],
            *radius,
        )),
        Shape2D::Edge { start, end } => Some((
            vec![body.world_point(*start), body.world_point(*end)],
            Fix128::ZERO,
        )),
        Shape2D::Polygon { vertices } if vertices.len() >= 3 => {
            Some((transform_vertices(body, vertices), Fix128::ZERO))
        }
        _ => None,
    }
}

/// Edges `(p, q)` of a core: one for a segment, `n` (closing) for a polygon.
fn core_edges(core: &[Vec2Fix]) -> Vec<(Vec2Fix, Vec2Fix)> {
    if core.len() == 2 {
        return vec![(core[0], core[1])];
    }
    (0..core.len())
        .map(|i| (core[i], core[(i + 1) % core.len()]))
        .collect()
}

/// Contact between two rounded convex shapes, each a core (segment or convex
/// polygon) inflated by a radius. The normal points from A toward B and the
/// depth is the exact penetration of the inflated shapes:
///
/// * cores disjoint: `depth = r_a + r_b - dist(core_a, core_b)` along the
///   direction between the closest points (no contact when negative);
/// * cores overlapping: `depth = r_a + r_b + PD(core_a, core_b)`, where the
///   core penetration `PD` is the smallest SAT overlap over the edge normals of
///   both cores (the face normals of their Minkowski difference).
///
/// The contact point lies in the middle of the overlap region: along the
/// normal halfway between A's surface and B's surface, along the tangent at
/// the centre of the shared extent of the two supporting features (so a
/// segment lying flat on a face gets the midpoint of the touching interval).
fn rounded_core_vs_rounded_core(
    core_a: &[Vec2Fix],
    radius_a: Fix128,
    core_b: &[Vec2Fix],
    radius_b: Fix128,
) -> Option<Contact2D> {
    let edges_a = core_edges(core_a);
    let edges_b = core_edges(core_b);
    let radius_sum = radius_a + radius_b;

    // Closest points of the cores (on their boundaries).
    let mut closest: Option<(Fix128, Vec2Fix, Vec2Fix)> = None;
    for &(pa, qa) in &edges_a {
        for &(pb, qb) in &edges_b {
            let (ca, cb) = closest_points_segment_segment(pa, qa, pb, qb);
            let dist_sq = (cb - ca).length_squared();
            if closest.is_none_or(|(d, _, _)| dist_sq < d) {
                closest = Some((dist_sq, ca, cb));
            }
        }
    }
    let (dist_sq, ca, cb) = closest?;

    // SAT over the edge normals of both cores plus the closest-point direction.
    // The edge normals are the face normals of the Minkowski difference, so the
    // smallest overlap over them is the core penetration; the closest-point
    // direction separates disjoint cores even when a core is degenerate (a
    // point, or collinear segments), where the edge normals alone are
    // incomplete. Any extra axis only adds overlaps that are >= the true one.
    let mut axes: Vec<Vec2Fix> = edges_a
        .iter()
        .chain(edges_b.iter())
        .map(|&(p, q)| (q - p).perpendicular().normalize())
        .collect();
    axes.push((cb - ca).normalize());
    let mut separated = false;
    let mut best: Option<(Fix128, Vec2Fix)> = None;
    for axis in axes {
        if axis == Vec2Fix::ZERO {
            continue;
        }
        let (min_a, max_a) = project_polygon(core_a, axis);
        let (min_b, max_b) = project_polygon(core_b, axis);
        if max_a < min_b || max_b < min_a {
            separated = true;
            break;
        }
        // overlap removed by moving B along +axis, resp. along -axis
        for (overlap, dir) in [(max_a - min_b, axis), (max_b - min_a, -axis)] {
            if best.is_none_or(|(d, _)| overlap < d) {
                best = Some((overlap, dir));
            }
        }
    }

    let (normal, depth) = if separated {
        if dist_sq > radius_sum * radius_sum {
            return None;
        }
        let dist = dist_sq.sqrt();
        if dist.is_zero() {
            return None;
        }
        ((cb - ca) / dist, radius_sum - dist)
    } else {
        match best {
            Some((overlap, dir)) => (dir, overlap + radius_sum),
            // both cores are the same point: no direction is preferred
            None => (Vec2Fix::UNIT_Y, radius_sum),
        }
    };
    if depth.is_negative() {
        return None;
    }

    Some(Contact2D {
        point: overlap_midpoint(core_a, radius_a, core_b, radius_b, normal),
        normal,
        depth,
        body_a: 0,
        body_b: 0,
    })
}

/// Middle of the overlap region of two rounded cores along `normal` (A -> B).
fn overlap_midpoint(
    core_a: &[Vec2Fix],
    radius_a: Fix128,
    core_b: &[Vec2Fix],
    radius_b: Fix128,
    normal: Vec2Fix,
) -> Vec2Fix {
    // vertices within `eps` of the extreme projection form the supporting
    // feature (one vertex, or the two ends of an edge parallel to the tangent)
    let eps = Fix128::from_ratio(1, 1 << 30);
    let tangent = normal.perpendicular();
    let (_, top_a) = project_polygon(core_a, normal);
    let (bottom_b, _) = project_polygon(core_b, normal);
    let extent = |core: &[Vec2Fix], on_feature: &dyn Fn(Fix128) -> bool| {
        let mut lo: Option<Fix128> = None;
        let mut hi: Option<Fix128> = None;
        for v in core {
            if on_feature(v.dot(normal)) {
                let t = v.dot(tangent);
                lo = Some(lo.map_or(t, |l| if t < l { t } else { l }));
                hi = Some(hi.map_or(t, |h| if t > h { t } else { h }));
            }
        }
        (lo.unwrap_or(Fix128::ZERO), hi.unwrap_or(Fix128::ZERO))
    };
    let (lo_a, hi_a) = extent(core_a, &|d| d >= top_a - eps);
    let (lo_b, hi_b) = extent(core_b, &|d| d <= bottom_b + eps);
    let lo = if lo_a > lo_b { lo_a } else { lo_b };
    let hi = if hi_a < hi_b { hi_a } else { hi_b };
    let along_tangent = (lo + hi).half();
    let along_normal = ((top_a + radius_a) + (bottom_b - radius_b)).half();
    tangent * along_tangent + normal * along_normal
}

/// Closest points between segments `p1 q1` and `p2 q2` (Ericson, Real-Time
/// Collision Detection §5.1.9), handling zero-length segments.
fn closest_points_segment_segment(
    p1: Vec2Fix,
    q1: Vec2Fix,
    p2: Vec2Fix,
    q2: Vec2Fix,
) -> (Vec2Fix, Vec2Fix) {
    let clamp01 = |x: Fix128| {
        if x.is_negative() {
            Fix128::ZERO
        } else if x > Fix128::ONE {
            Fix128::ONE
        } else {
            x
        }
    };
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);
    let (s, t) = if a.is_zero() && e.is_zero() {
        (Fix128::ZERO, Fix128::ZERO)
    } else if a.is_zero() {
        (Fix128::ZERO, clamp01(f / e))
    } else {
        let c = d1.dot(r);
        if e.is_zero() {
            (clamp01(-c / a), Fix128::ZERO)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s = if denom.is_zero() {
                Fix128::ZERO
            } else {
                clamp01((b * f - c * e) / denom)
            };
            let t = (b * s + f) / e;
            if t.is_negative() {
                (clamp01(-c / a), Fix128::ZERO)
            } else if t > Fix128::ONE {
                (clamp01((b - c) / a), Fix128::ONE)
            } else {
                (s, t)
            }
        }
    };
    (p1 + d1 * s, p2 + d2 * t)
}

// ============================================================================
// XPBD Contact Solver
// ============================================================================

/// Resolve one contact's current penetration by a positional correction
/// (rigid XPBD contact, compliance 0) and return the positional multiplier
/// applied (the depth removed), which the velocity pass turns into the
/// normal force bound for friction.
fn solve_contact_position(bodies: &mut [RigidBody2D], contact: &Contact2D) -> Fix128 {
    {
        let a = contact.body_a;
        let b = contact.body_b;
        let normal = contact.normal;
        let depth = contact.depth;

        if depth.is_negative() || depth.is_zero() {
            return Fix128::ZERO;
        }

        let inv_mass_a = bodies[a].inv_mass;
        let inv_mass_b = bodies[b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;

        if total_inv_mass.is_zero() {
            return Fix128::ZERO;
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
        depth
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
    /// Mouse joint: drags a body toward a world-space target point with the
    /// spring-damper `F = stiffness·(target − x) − damping·v`, `|F| <= max_force`.
    ///
    /// Solved inside the step as an XPBD constraint with damping (backward Euler per
    /// substep, independent of `iterations`). For a body of mass `m`, angular frequency
    /// `ω` and damping ratio `ζ` use `stiffness = m ω²` and `damping = 2 ζ m ω`; `ζ = 1`
    /// is the critically damped return `x(t) = (x0 + (v0 + ω x0) t)·e^(−ωt)`. Each joint
    /// carries its own `stiffness` / `damping`, so mouse joints of different `ω` can share
    /// one world. `PhysicsConfig2D::damping` is applied on top (set it to 1 for the pure law).
    Mouse {
        /// Body index.
        body: usize,
        /// World-space target position.
        target: Vec2Fix,
        /// Maximum force the joint can apply, in N (`|F| <= max_force`); `<= 0` applies
        /// no force.
        max_force: Fix128,
        /// Spring stiffness `k`, in N/m (formerly a dimensionless fraction applied per
        /// solver iteration, see CHANGELOG). `k = m ω²` for angular frequency `ω`.
        stiffness: Fix128,
        /// Damping coefficient `c`, in N·s/m, acting on the body's velocity.
        /// `c = 2·sqrt(k·m)` is critically damped (no overshoot), smaller oscillates,
        /// larger creeps.
        damping: Fix128,
    },
}

/// Solve all 2D joints using XPBD position-level constraints.
///
/// One call is one solver iteration starting from zero accumulated multipliers.
/// [`PhysicsWorld2D::step`] keeps the multipliers across the iterations of a substep, so
/// compliant joints there do not stiffen with `iterations`.
pub fn solve_joints_2d(bodies: &mut [RigidBody2D], joints: &[Joint2D], sub_dt: Fix128) {
    let mut lambdas = vec![Vec2Fix::ZERO; joints.len()];
    solve_joints_2d_accumulated(bodies, joints, sub_dt, &mut lambdas);
}

/// One XPBD iteration over `joints`, accumulating each joint's multiplier in
/// `lambdas[i]` (Distance uses `.x`, Mouse the vector). `lambdas` is zeroed by the caller at
/// the start of every substep.
fn solve_joints_2d_accumulated(
    bodies: &mut [RigidBody2D],
    joints: &[Joint2D],
    sub_dt: Fix128,
    lambdas: &mut [Vec2Fix],
) {
    let alpha = if sub_dt.is_zero() {
        Fix128::ZERO
    } else {
        Fix128::ONE / (sub_dt * sub_dt)
    };

    for (joint, lambda) in joints.iter().zip(lambdas.iter_mut()) {
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
                    &mut lambda.x,
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
                damping,
            } => {
                solve_mouse(
                    bodies, *body, *target, *max_force, *stiffness, *damping, sub_dt, lambda,
                );
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
    accumulated: &mut Fix128,
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

    // XPBD (Macklin 2016): the compliance term sees the multiplier accumulated over
    // the substep, so the effective stiffness 1/compliance does not grow with iterations
    let lambda = (-c - compliance * alpha * *accumulated) / w;
    *accumulated = *accumulated + lambda;
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
#[allow(clippy::too_many_arguments)]
fn solve_mouse(
    bodies: &mut [RigidBody2D],
    body_idx: usize,
    target: Vec2Fix,
    max_force: Fix128,
    stiffness: Fix128,
    damping: Fix128,
    sub_dt: Fix128,
    accumulated: &mut Vec2Fix,
) {
    let body = &bodies[body_idx];
    if body.body_type != BodyType2D::Dynamic || sub_dt.is_zero() {
        return;
    }
    // XPBD with damping (Macklin et al. 2016, eq. 26) on the vector constraint
    // C = x − target (gradient I), multiplied through by k h² so that k = 0 (a pure
    // damper) needs no division:
    //   Δλ = (−k h² C − λ − c h (x − x_prev)) / ((k h² + c h) w + 1),   x += w Δλ
    // The force is λ / h², so |F| <= max_force is |λ| <= max_force · h².
    let h = sub_dt;
    let kh2 = stiffness * h * h;
    let ch = damping * h;
    let w = body.inv_mass;
    let c = body.position - target;
    let moved = body.position - body.prev_position;
    let denom = (kh2 + ch) * w + Fix128::ONE;
    if denom.is_zero() {
        return;
    }
    let numer = c * (Fix128::ZERO - kh2) - *accumulated - moved * ch;
    let mut total = *accumulated + numer / denom;
    let limit = if max_force > Fix128::ZERO {
        max_force * h * h
    } else {
        Fix128::ZERO
    };
    let len = total.length();
    if len > limit {
        total = if len.is_zero() {
            Vec2Fix::ZERO
        } else {
            total * (limit / len)
        };
    }
    let dlambda = total - *accumulated;
    *accumulated = total;
    bodies[body_idx].position = bodies[body_idx].position + dlambda * w;
}

// ============================================================================
// Tethers2D (angular spring-damper + kinematic drive)
// ============================================================================

/// Angular spring-damper pulling a dynamic body's angle toward `target_angle`:
/// `τ = stiffness·(target_angle − θ) − damping·ω`.
///
/// Solved inside [`PhysicsWorld2D::step_with_tethers`] as an XPBD constraint with damping
/// (Macklin et al. 2016, eq. 26) on `C = θ − target_angle`, one multiplier per tether
/// accumulated over the iterations of a substep, so the law is backward Euler per substep
/// and independent of `iterations`. For moment of inertia `I`, angular frequency `ω` and
/// damping ratio `ζ` use `stiffness = I ω²`, `damping = 2 ζ I ω`
/// ([`AngularTether2D::critically_damped`] is `ζ = 1`, the return
/// `θ(t) = (θ0 + (ω0 + ω θ0) t)·e^(−ωt)` about the target, without overshoot).
///
/// The error `θ − target_angle` is not wrapped to `(−π, π]`: a body that has spun one full
/// turn is pulled back through that turn. Combine with [`Joint2D::Mouse`] to return both
/// position and orientation. Only `Dynamic` bodies are affected (use
/// [`KinematicDrive2D`] for kinematic ones). `PhysicsConfig2D::damping` is applied on top
/// (set it to 1 for the pure law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AngularTether2D {
    /// Body index.
    pub body: usize,
    /// Target angle in radians (same unwrapped convention as `RigidBody2D::angle`).
    pub target_angle: Fix128,
    /// Angular stiffness `k_θ`, in N·m/rad.
    pub stiffness: Fix128,
    /// Angular damping `c_θ`, in N·m·s/rad, acting on the body's angular velocity.
    pub damping: Fix128,
}

impl AngularTether2D {
    /// Tether with explicit angular stiffness (N·m/rad) and damping (N·m·s/rad).
    #[must_use]
    pub const fn new(
        body: usize,
        target_angle: Fix128,
        stiffness: Fix128,
        damping: Fix128,
    ) -> Self {
        Self {
            body,
            target_angle,
            stiffness,
            damping,
        }
    }

    /// Critically damped tether for a body of moment of inertia `inertia`:
    /// `stiffness = I ω²`, `damping = 2 I ω`.
    #[must_use]
    pub fn critically_damped(
        body: usize,
        target_angle: Fix128,
        inertia: Fix128,
        omega: Fix128,
    ) -> Self {
        Self::new(
            body,
            target_angle,
            inertia * omega * omega,
            Fix128::from_int(2) * inertia * omega,
        )
    }
}

/// Drives a `Kinematic` body toward `target_position` / `target_angle` along the critically
/// damped trajectory of angular frequency `omega`, through its velocity.
///
/// A kinematic body has infinite mass, so this is not a constraint: at every substep of
/// length `h` the position error `e = x − target` and velocity `v` are advanced by the exact
/// solution of `ë = −2ω ė − ω² e`,
/// `e' = (e + (v + ω e) h)·E`, `v' = (v − ω h (v + ω e))·E`, `E = e^(−ωh)`,
/// and likewise for the angle. Composing substeps reproduces the continuous
/// `e(t) = (e0 + (v0 + ω e0) t)·e^(−ωt)` for any `substeps` (up to `Fix128` rounding), so the
/// body reaches the target without overshoot when it starts at rest. The body's `velocity`
/// / `angular_velocity` hold `v'`, so contacts see the true velocity, and the body keeps
/// pushing dynamic bodies it meets. The drive overrides any velocity set by hand; with
/// several drives on one body the one added last wins. Non-kinematic bodies are ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct KinematicDrive2D {
    /// Body index (must be `BodyType2D::Kinematic`).
    pub body: usize,
    /// Target position.
    pub target_position: Vec2Fix,
    /// Target angle in radians (unwrapped, like `AngularTether2D::target_angle`).
    pub target_angle: Fix128,
    /// Angular frequency `ω` (1/s) of the critically damped return; `<= 0` holds the body
    /// where it is with its current velocity unchanged (the drive is inactive).
    pub omega: Fix128,
}

impl KinematicDrive2D {
    /// Drive toward `target_position` / `target_angle` with angular frequency `omega`.
    #[must_use]
    pub const fn new(
        body: usize,
        target_position: Vec2Fix,
        target_angle: Fix128,
        omega: Fix128,
    ) -> Self {
        Self {
            body,
            target_position,
            target_angle,
            omega,
        }
    }
}

/// Handle of an [`AngularTether2D`] in a [`Tethers2D`] set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AngularTetherId(usize);

/// Handle of a [`KinematicDrive2D`] in a [`Tethers2D`] set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KinematicDriveId(usize);

/// Set of angular tethers and kinematic drives, solved by
/// [`PhysicsWorld2D::step_with_tethers`].
///
/// Kept outside `PhysicsWorld2D` so the world's public layout is unchanged. Handles stay
/// valid until their own removal (slots are not reused), and entries are solved in the
/// order they were added, so a replay is bit-identical.
#[derive(Clone, Debug, Default)]
pub struct Tethers2D {
    angular: Vec<Option<AngularTether2D>>,
    drives: Vec<Option<KinematicDrive2D>>,
}

impl Tethers2D {
    /// Empty set. `step_with_tethers` with an empty set is `step`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            angular: Vec::new(),
            drives: Vec::new(),
        }
    }

    /// Add an angular tether.
    pub fn add_angular(&mut self, tether: AngularTether2D) -> AngularTetherId {
        self.angular.push(Some(tether));
        AngularTetherId(self.angular.len() - 1)
    }

    /// Remove an angular tether, returning it if the handle was live.
    pub fn remove_angular(&mut self, id: AngularTetherId) -> Option<AngularTether2D> {
        self.angular.get_mut(id.0).and_then(Option::take)
    }

    /// The angular tether behind `id`, if live.
    #[must_use]
    pub fn angular(&self, id: AngularTetherId) -> Option<&AngularTether2D> {
        self.angular.get(id.0).and_then(Option::as_ref)
    }

    /// Mutable access (e.g. to move `target_angle`), if live.
    pub fn angular_mut(&mut self, id: AngularTetherId) -> Option<&mut AngularTether2D> {
        self.angular.get_mut(id.0).and_then(Option::as_mut)
    }

    /// Add a kinematic drive.
    pub fn add_drive(&mut self, drive: KinematicDrive2D) -> KinematicDriveId {
        self.drives.push(Some(drive));
        KinematicDriveId(self.drives.len() - 1)
    }

    /// Remove a kinematic drive, returning it if the handle was live. The body keeps the
    /// velocity the drive last gave it.
    pub fn remove_drive(&mut self, id: KinematicDriveId) -> Option<KinematicDrive2D> {
        self.drives.get_mut(id.0).and_then(Option::take)
    }

    /// The kinematic drive behind `id`, if live.
    #[must_use]
    pub fn drive(&self, id: KinematicDriveId) -> Option<&KinematicDrive2D> {
        self.drives.get(id.0).and_then(Option::as_ref)
    }

    /// Mutable access (e.g. to move the targets), if live.
    pub fn drive_mut(&mut self, id: KinematicDriveId) -> Option<&mut KinematicDrive2D> {
        self.drives.get_mut(id.0).and_then(Option::as_mut)
    }

    /// Number of live angular tethers.
    #[must_use]
    pub fn angular_count(&self) -> usize {
        self.angular.iter().filter(|t| t.is_some()).count()
    }

    /// Number of live kinematic drives.
    #[must_use]
    pub fn drive_count(&self) -> usize {
        self.drives.iter().filter(|d| d.is_some()).count()
    }

    fn live_drives(&self) -> impl Iterator<Item = &KinematicDrive2D> {
        self.drives.iter().flatten()
    }
}

/// `e^(−x)` for `x >= 0`, accurate to a few units of 2⁻⁶⁴: Taylor series on `x / 2^k <= 1/2`
/// followed by `k` squarings. (`Fix128::exp` is ≲ 1e-6 relative, which compounds over the
/// substeps of a kinematic drive.)
fn exp_neg(x: Fix128) -> Fix128 {
    if x <= Fix128::ZERO {
        return Fix128::ONE;
    }
    if x.hi >= 44 {
        return Fix128::ZERO;
    }
    let half = Fix128::from_ratio(1, 2);
    let mut y = x;
    let mut k = 0_u32;
    while y > half {
        y = y * half;
        k += 1;
    }
    // Σ (−y)ⁿ / n!, |y| <= 1/2: 26 terms leave a remainder below 2⁻⁶⁴·2⁻³⁰
    let mut sum = Fix128::ONE;
    let mut term = Fix128::ONE;
    for n in 1..=26_i64 {
        term = Fix128::ZERO - term * y / Fix128::from_int(n);
        sum = sum + term;
    }
    for _ in 0..k {
        sum = sum * sum;
    }
    sum
}

/// Advance a kinematic drive by one substep (closed-form critically damped step).
fn apply_kinematic_drive(bodies: &mut [RigidBody2D], drive: &KinematicDrive2D, h: Fix128) {
    let Some(body) = bodies.get_mut(drive.body) else {
        return;
    };
    if body.body_type != BodyType2D::Kinematic || drive.omega <= Fix128::ZERO {
        return;
    }
    let w = drive.omega;
    let e_factor = exp_neg(w * h);
    // linear: start of the substep is prev_position with the velocity it began with
    let e0 = body.prev_position - drive.target_position;
    let v0 = body.velocity;
    let s = v0 + e0 * w;
    body.position = drive.target_position + (e0 + s * h) * e_factor;
    body.velocity = (v0 - s * (w * h)) * e_factor;
    // angular
    let a0 = body.prev_angle - drive.target_angle;
    let om0 = body.angular_velocity;
    let sa = om0 + w * a0;
    body.angle = drive.target_angle + (a0 + sa * h) * e_factor;
    body.angular_velocity = (om0 - sa * (w * h)) * e_factor;
}

/// One XPBD-with-damping iteration of an angular tether, accumulating its multiplier.
fn solve_angular_tether(
    bodies: &mut [RigidBody2D],
    tether: &AngularTether2D,
    h: Fix128,
    accumulated: &mut Fix128,
) {
    let Some(body) = bodies.get_mut(tether.body) else {
        return;
    };
    if body.body_type != BodyType2D::Dynamic || h.is_zero() {
        return;
    }
    // Same scalar update as `solve_mouse`, multiplied through by k h²:
    //   Δλ = (−k h² C − λ − c h (θ − θ_prev)) / ((k h² + c h) w + 1),   θ += w Δλ
    let kh2 = tether.stiffness * h * h;
    let ch = tether.damping * h;
    let w = body.inv_inertia;
    let c = body.angle - tether.target_angle;
    let moved = body.angle - body.prev_angle;
    let denom = (kh2 + ch) * w + Fix128::ONE;
    if denom.is_zero() {
        return;
    }
    let numer = Fix128::ZERO - kh2 * c - *accumulated - ch * moved;
    let dlambda = numer / denom;
    *accumulated = *accumulated + dlambda;
    body.angle = body.angle + dlambda * w;
}

impl core::fmt::Debug for PhysicsWorld2D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PhysicsWorld2D")
            .field("bodies", &format_args!("[{} items]", self.bodies.len()))
            .field("gravity", &self.gravity)
            .field("config", &self.config)
            .field("joints", &format_args!("[{} items]", self.joints.len()))
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
            // k = 100 N/m and c = 20 N·s/m on a 1 kg body: omega = 10 rad/s,
            // c = 2·sqrt(k·m) (critically damped)
            stiffness: Fix128::from_int(100),
            damping: Fix128::from_int(20),
        });

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..300 {
            world.step(dt);
        }

        // Body should have moved toward target
        let pos = world.bodies[id].position;
        // critically damped from 14.14 m: 14.14·(1 + 50)·e^(−50) ≈ 1e-19 m after 5 s; with
        // the damping ignored the frame damping alone leaves ≈ 0.86 m, so the bound is 1 cm
        let dist = pos.distance_to(target);
        assert!(
            dist < Fix128::from_ratio(1, 100),
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

    // ---- Tethers2D (closed-form checks; the full oracles are in
    // tests/analytic_physics2d_tethers.rs) ----

    fn tether_world(substeps: usize) -> PhysicsWorld2D {
        PhysicsWorld2D::new(PhysicsConfig2D {
            gravity: Vec2Fix::ZERO,
            substeps,
            iterations: 2,
            damping: Fix128::ONE,
        })
    }

    #[test]
    fn exp_neg_matches_reference_values() {
        // e^-1, e^-0.25, e^-5 against f64 to well below f64 resolution of Fix128 use
        for (x, want) in [
            (1.0, 0.367_879_441_171_442_3),
            (0.25, 0.778_800_783_071_404_9),
            (5.0, 0.006_737_946_999_085_467),
        ] {
            let got = exp_neg(Fix128::from_f64(x)).to_f64();
            assert!((got - want).abs() < 1e-15, "e^-{x}: {got} vs {want}");
        }
        assert_eq!(exp_neg(Fix128::ZERO), Fix128::ONE);
        assert_eq!(exp_neg(Fix128::from_int(-3)), Fix128::ONE);
        assert_eq!(exp_neg(Fix128::from_int(50)), Fix128::ZERO);
    }

    #[test]
    fn angular_tether_one_substep_is_backward_euler() {
        // I = 2, ω = 6, ζ = 1, h = 1/60: θ' = θ (1 + 2a) / (1 + a)², a = ω h
        let mut w = tether_world(1);
        let mut b = RigidBody2D::new_dynamic(
            Vec2Fix::ZERO,
            Fix128::ONE,
            Shape2D::Circle {
                radius: Fix128::ONE,
            },
        );
        b.inv_inertia = Fix128::from_ratio(1, 2);
        b.angle = Fix128::ONE;
        let idx = w.add_body(b);
        let mut set = Tethers2D::new();
        let id = set.add_angular(AngularTether2D::critically_damped(
            idx,
            Fix128::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(6),
        ));
        assert_eq!(
            set.angular(id).map(|t| t.stiffness),
            Some(Fix128::from_int(72))
        );
        w.step_with_tethers(Fix128::from_ratio(1, 60), &set);
        let a = 0.1_f64;
        let want = (1.0 + 2.0 * a) / ((1.0 + a) * (1.0 + a));
        assert!((w.bodies[idx].angle.to_f64() - want).abs() < 1e-15);
        assert!((w.bodies[idx].angular_velocity.to_f64() - (want - 1.0) * 60.0).abs() < 1e-12);
        // retarget, then remove: the set is empty again
        set.angular_mut(id).unwrap().target_angle = Fix128::ONE;
        assert_eq!(set.angular_count(), 1);
        assert!(set.remove_angular(id).is_some());
        assert!(set.angular(id).is_none() && set.angular_mut(id).is_none());
        assert_eq!(set.angular_count(), 0);
    }

    #[test]
    fn kinematic_drive_one_frame_is_the_closed_form() {
        // e(t) = (e0 + (v0 + ω e0) t) e^(−ωt), composed over 4 substeps
        let mut w = tether_world(4);
        let mut b = RigidBody2D::new_kinematic(
            Vec2Fix::from_int(2, -1),
            Shape2D::Circle {
                radius: Fix128::ONE,
            },
        );
        b.angle = Fix128::from_int(3);
        b.velocity = Vec2Fix::from_int(1, 0);
        let idx = w.add_body(b);
        let mut set = Tethers2D::new();
        let id = set.add_drive(KinematicDrive2D::new(
            idx,
            Vec2Fix::ZERO,
            Fix128::ONE,
            Fix128::from_int(5),
        ));
        assert_eq!(set.drive_count(), 1);
        w.step_with_tethers(Fix128::from_ratio(1, 10), &set);
        let (om, t) = (5.0_f64, 0.1_f64);
        let ex = |e0: f64, v0: f64| (e0 + (v0 + om * e0) * t) * crate::det_math::exp64(-om * t);
        let body = &w.bodies[idx];
        assert!((body.position.x.to_f64() - ex(2.0, 1.0)).abs() < 1e-14);
        assert!((body.position.y.to_f64() - ex(-1.0, 0.0)).abs() < 1e-14);
        assert!((body.angle.to_f64() - 1.0 - ex(2.0, 0.0)).abs() < 1e-14);
        set.drive_mut(id).unwrap().omega = Fix128::ZERO;
        assert_eq!(set.drive(id).map(|d| d.omega), Some(Fix128::ZERO));
        assert!(set.remove_drive(id).is_some());
        assert!(
            set.drive(id).is_none()
                && set.drive_mut(id).is_none()
                && set.remove_drive(id).is_none()
        );
        assert_eq!(set.drive_count(), 0);
    }

    #[test]
    fn tethers_skip_out_of_range_and_wrong_body_types() {
        let mut w = tether_world(1);
        let d = w.add_body(RigidBody2D::new_dynamic(
            Vec2Fix::ZERO,
            Fix128::ONE,
            Shape2D::Circle {
                radius: Fix128::ONE,
            },
        ));
        let mut k = RigidBody2D::new_kinematic(
            Vec2Fix::from_int(10, 0),
            Shape2D::Circle {
                radius: Fix128::ONE,
            },
        );
        k.inv_inertia = Fix128::ONE;
        k.angle = Fix128::ONE;
        let k = w.add_body(k);
        let mut set = Tethers2D::new();
        set.add_angular(AngularTether2D::new(
            7,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
        ));
        set.add_angular(AngularTether2D::new(
            k,
            Fix128::ZERO,
            Fix128::from_int(9),
            Fix128::ONE,
        ));
        set.add_drive(KinematicDrive2D::new(
            7,
            Vec2Fix::ZERO,
            Fix128::ZERO,
            Fix128::ONE,
        ));
        set.add_drive(KinematicDrive2D::new(
            d,
            Vec2Fix::from_int(1, 1),
            Fix128::ZERO,
            Fix128::ONE,
        ));
        w.step_with_tethers(Fix128::from_ratio(1, 60), &set);
        assert_eq!(w.bodies[d].position, Vec2Fix::ZERO);
        assert_eq!(w.bodies[k].angle, Fix128::ONE);
        // dt = 0: tethers see h = 0 and leave the state alone
        let before = w.bodies[k].position;
        w.step_with_tethers(Fix128::ZERO, &set);
        assert_eq!(w.bodies[k].position, before);
    }
}
