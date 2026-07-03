//! Orientation-tracking extension for the sub-stepping TGS solver.
//!
//! Adds a unit-quaternion orientation to the 6-DOF body state and
//! provides the standard quaternion integration primitive:
//!
//! ```text
//! q_new = normalize(q + 0.5 · dt · (ω × q))
//! ```
//!
//! Here `ω × q` is the Hamilton product between the pure quaternion
//! `(ω.x, ω.y, ω.z, 0)` and the current orientation `q`. Integrating
//! the orientation at the end of a sub-step and then rebuilding the
//! world-frame inertia (`R · I_local · Rᵀ`) is the usual recipe for
//! layering full 6-DOF simulation on top of the existing hooks.
//!
//! The inertia is kept diagonal for now, matching
//! [`crate::solver_tgs_hooks_6dof::Body6DofState`]; consumers that
//! need a full symmetric world-frame inertia can rebuild one before
//! each sub-step using [`crate::math::Mat3Fix`].

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver_tgs::{BodyLike, ContactLike};

// ---------------------------------------------------------------------------
// Small `[Fix128; 3]` scratch layout used by the hooks family.
// ---------------------------------------------------------------------------

type Vec3 = [Fix128; 3];

const V_ZERO: Vec3 = [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO];

/// Convert a `[Fix128; 3]` triple to a [`Vec3Fix`].
#[inline]
#[must_use]
pub fn to_vec3fix(v: Vec3) -> Vec3Fix {
    Vec3Fix {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

/// Convert a [`Vec3Fix`] to a `[Fix128; 3]` triple.
#[inline]
#[must_use]
pub fn from_vec3fix(v: Vec3Fix) -> Vec3 {
    [v.x, v.y, v.z]
}

// ---------------------------------------------------------------------------
// Orientation integration primitive
// ---------------------------------------------------------------------------

/// Integrate a unit quaternion by an angular velocity `omega` over the
/// interval `dt` using the classic first-order rule
/// `q_new = normalize(q + 0.5 · dt · (ω_q × q))`.
#[must_use]
pub fn integrate_orientation(q: QuatFix, omega: Vec3, dt: Fix128) -> QuatFix {
    let omega_q = QuatFix::new(omega[0], omega[1], omega[2], Fix128::ZERO);
    let q_dot = omega_q.mul(q);
    let half_dt = Fix128::from_f32(0.5) * dt;
    let candidate = QuatFix::new(
        q.x + q_dot.x * half_dt,
        q.y + q_dot.y * half_dt,
        q.z + q_dot.z * half_dt,
        q.w + q_dot.w * half_dt,
    );
    candidate.normalize()
}

// ---------------------------------------------------------------------------
// Body state with orientation
// ---------------------------------------------------------------------------

/// A [`Body6DofState`] augmented with a unit quaternion orientation.
/// Callers that want full 6-DOF simulation typically build a shim hook
/// that mirrors the existing [`Pgs6DofHooks`] logic against this type
/// and finishes each sub-step by [`integrate_orientation`] over
/// `end_substep`'s `sub_dt`.
///
/// [`Body6DofState`]: crate::solver_tgs_hooks_6dof::Body6DofState
/// [`Pgs6DofHooks`]: crate::solver_tgs_hooks_6dof::Pgs6DofHooks
#[derive(Debug, Clone, Copy)]
pub struct Body6DofOrientedState {
    /// World-space centre-of-mass position.
    pub position: Vec3,
    /// Unit quaternion describing the body's world-frame orientation.
    pub orientation: QuatFix,
    /// Linear velocity of the body's centre of mass (world frame).
    pub linear_velocity: Vec3,
    /// Angular velocity around the body's centre of mass (world frame).
    pub angular_velocity: Vec3,
    /// Reciprocal of the body mass. `Fix128::ZERO` marks the body as
    /// static or kinematic (no linear response to impulses).
    pub inv_mass: Fix128,
    /// Reciprocals of the three principal moments of inertia in the
    /// body's local frame.
    pub inv_inertia_local: Vec3,
    /// `true` when the body participates in velocity/position updates.
    pub is_dynamic: bool,
    /// Stable identity used for warm-start indexing across frames.
    pub stable_id: u64,
}

impl Default for Body6DofOrientedState {
    fn default() -> Self {
        Self {
            position: V_ZERO,
            orientation: QuatFix::IDENTITY,
            linear_velocity: V_ZERO,
            angular_velocity: V_ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia_local: V_ZERO,
            is_dynamic: false,
            stable_id: 0,
        }
    }
}

impl BodyLike for Body6DofOrientedState {
    fn stable_id(&self) -> u64 {
        self.stable_id
    }
    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }
}

impl Body6DofOrientedState {
    /// Advance the body's linear position, angular velocity is left
    /// intact, and the orientation is integrated by
    /// [`integrate_orientation`].
    pub fn advance(&mut self, sub_dt: Fix128) {
        if !self.is_dynamic {
            return;
        }
        // Linear position: p += v · dt.
        self.position = [
            self.position[0] + self.linear_velocity[0] * sub_dt,
            self.position[1] + self.linear_velocity[1] * sub_dt,
            self.position[2] + self.linear_velocity[2] * sub_dt,
        ];
        // Orientation: standard quaternion integration.
        self.orientation = integrate_orientation(self.orientation, self.angular_velocity, sub_dt);
    }

    /// Rotate a body-local vector into the world frame.
    #[must_use]
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        from_vec3fix(self.orientation.rotate_vec(to_vec3fix(local)))
    }
}

// ---------------------------------------------------------------------------
// Contact (mirrors the diagonal-inertia contact, kept independent so
// callers can migrate to orientation tracking incrementally).
// ---------------------------------------------------------------------------

/// A pairwise contact for oriented bodies.
#[derive(Debug, Clone, Copy)]
pub struct ContactOriented {
    /// World-index of the first body participating in the contact.
    pub body_a: usize,
    /// World-index of the second body participating in the contact.
    pub body_b: usize,
    /// Stable identity used for warm-start indexing across frames.
    pub stable_id: u64,
    /// Contact normal, oriented from body A into body B (unit length).
    pub normal: Vec3,
    /// First tangent axis of the contact frame (unit length).
    pub tangent1: Vec3,
    /// Second tangent axis of the contact frame (unit length).
    pub tangent2: Vec3,
    /// Offset from body A's centre to the contact point (world frame).
    pub r_a: Vec3,
    /// Offset from body B's centre to the contact point (world frame).
    pub r_b: Vec3,
    /// Signed penetration depth. Positive values mean the bodies overlap.
    pub penetration: Fix128,
    /// Coulomb friction coefficient for the pair.
    pub friction: Fix128,
    /// Newton coefficient of restitution for the pair.
    pub restitution: Fix128,
    /// Accumulated normal impulse magnitude during the current sub-step.
    pub accum_normal: Fix128,
    /// Accumulated first-tangent impulse magnitude during the current sub-step.
    pub accum_tangent1: Fix128,
    /// Accumulated second-tangent impulse magnitude during the current sub-step.
    pub accum_tangent2: Fix128,
}

impl ContactLike for ContactOriented {
    fn body_a(&self) -> usize {
        self.body_a
    }
    fn body_b(&self) -> usize {
        self.body_b
    }
    fn stable_id(&self) -> u64 {
        self.stable_id
    }
}

// ---------------------------------------------------------------------------
// Vec3 helpers reused by the hook implementation.
// ---------------------------------------------------------------------------

#[inline]
fn v_add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn v_scale(v: Vec3, s: Fix128) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}
#[inline]
fn v_dot(a: Vec3, b: Vec3) -> Fix128 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn v_cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn diag_mul(diag: Vec3, v: Vec3) -> Vec3 {
    [diag[0] * v[0], diag[1] * v[1], diag[2] * v[2]]
}

/// Rotate a world-frame vector into the body-local principal-axes
/// frame using `q⁻¹`.
#[inline]
fn world_to_local(q: QuatFix, world: Vec3) -> Vec3 {
    from_vec3fix(q.conjugate().rotate_vec(to_vec3fix(world)))
}

/// Rotate a body-local vector into the world frame using `q`.
#[inline]
fn local_to_world_v(q: QuatFix, local: Vec3) -> Vec3 {
    from_vec3fix(q.rotate_vec(to_vec3fix(local)))
}

/// Apply the diagonal inverse inertia in the body's principal-axes
/// frame: `I_world⁻¹ · j = R · diag(inv_I_local) · Rᵀ · j`. The full
/// symmetric world-frame inverse inertia is realised implicitly by
/// this three-step transform without materialising a 3×3 matrix.
#[inline]
fn inv_inertia_apply(q: QuatFix, inv_i_local: Vec3, j_world: Vec3) -> Vec3 {
    let j_local = world_to_local(q, j_world);
    let dw_local = diag_mul(inv_i_local, j_local);
    local_to_world_v(q, dw_local)
}

// ---------------------------------------------------------------------------
// Pgs6DofOrientedHooks — full 6-DOF hooks with orientation tracking
// ---------------------------------------------------------------------------

use crate::solver_tgs::{CachedImpulse, ImpulseCache, TgsHooks};

/// Tunable parameters for [`Pgs6DofOrientedHooks`].
#[derive(Debug, Clone, Copy)]
pub struct Pgs6DofOrientedConfig {
    /// Gravitational acceleration applied to dynamic bodies per second.
    pub gravity: Vec3,
    /// Baumgarte positional-correction coefficient in `[0, 1]`.
    pub baumgarte: Fix128,
    /// Penetration slop below which positional correction is skipped.
    pub slop: Fix128,
    /// When `true`, warm-start impulses from [`ImpulseCache`] before the first iteration.
    pub warmstart: bool,
}

impl Default for Pgs6DofOrientedConfig {
    fn default() -> Self {
        Self {
            gravity: [Fix128::ZERO, Fix128::from_f32(-9.81), Fix128::ZERO],
            baumgarte: Fix128::from_f32(0.2),
            slop: Fix128::from_f32(0.005),
            warmstart: true,
        }
    }
}

/// Full 6-DOF [`TgsHooks`] implementation over
/// [`Body6DofOrientedState`] / [`ContactOriented`]. Angular impulses
/// are transformed into each body's principal-axes frame before being
/// scaled by the diagonal `inv_inertia_local`, and orientations are
/// integrated at the end of every sub-step.
pub struct Pgs6DofOrientedHooks<'a> {
    /// Mutable slice of oriented body states this hook operates on.
    pub bodies: &'a mut [Body6DofOrientedState],
    /// Mutable slice of oriented contacts this hook operates on.
    pub contacts: &'a mut [ContactOriented],
    /// Warm-start impulse cache reused across frames.
    pub cache: &'a mut ImpulseCache,
    /// Tunable projected Gauss-Seidel parameters for the oriented solve.
    pub cfg: Pgs6DofOrientedConfig,
    restitution_bias: Vec<Fix128>,
}

impl<'a> Pgs6DofOrientedHooks<'a> {
    /// Construct a new oriented hook binding the provided body / contact / cache slices with `cfg`.
    #[must_use]
    pub fn new(
        bodies: &'a mut [Body6DofOrientedState],
        contacts: &'a mut [ContactOriented],
        cache: &'a mut ImpulseCache,
        cfg: Pgs6DofOrientedConfig,
    ) -> Self {
        let n = contacts.len();
        Self {
            bodies,
            contacts,
            cache,
            cfg,
            restitution_bias: vec![Fix128::ZERO; n],
        }
    }

    fn apply_impulse(&mut self, c_idx: usize, impulse: Vec3) {
        let contact = self.contacts[c_idx];
        let (ia, ib) = (contact.body_a, contact.body_b);
        if self.bodies[ia].is_dynamic {
            let m = self.bodies[ia].inv_mass;
            self.bodies[ia].linear_velocity =
                v_sub(self.bodies[ia].linear_velocity, v_scale(impulse, m));
            let j_world = v_cross(contact.r_a, impulse);
            let dw = inv_inertia_apply(
                self.bodies[ia].orientation,
                self.bodies[ia].inv_inertia_local,
                j_world,
            );
            self.bodies[ia].angular_velocity = v_sub(self.bodies[ia].angular_velocity, dw);
        }
        if self.bodies[ib].is_dynamic {
            let m = self.bodies[ib].inv_mass;
            self.bodies[ib].linear_velocity =
                v_add(self.bodies[ib].linear_velocity, v_scale(impulse, m));
            let j_world = v_cross(contact.r_b, impulse);
            let dw = inv_inertia_apply(
                self.bodies[ib].orientation,
                self.bodies[ib].inv_inertia_local,
                j_world,
            );
            self.bodies[ib].angular_velocity = v_add(self.bodies[ib].angular_velocity, dw);
        }
    }

    fn contact_rel_velocity(&self, c_idx: usize) -> Vec3 {
        let contact = self.contacts[c_idx];
        let va = self.bodies[contact.body_a].linear_velocity;
        let wa = self.bodies[contact.body_a].angular_velocity;
        let vb = self.bodies[contact.body_b].linear_velocity;
        let wb = self.bodies[contact.body_b].angular_velocity;
        v_sub(
            v_add(vb, v_cross(wb, contact.r_b)),
            v_add(va, v_cross(wa, contact.r_a)),
        )
    }

    /// Effective mass along `axis` at contact `c_idx`, using the
    /// principal-axes inertia for both bodies.
    fn effective_mass(&self, c_idx: usize, axis: Vec3) -> Fix128 {
        let contact = self.contacts[c_idx];
        let (ia, ib) = (contact.body_a, contact.body_b);
        let inv_m = self.bodies[ia].inv_mass + self.bodies[ib].inv_mass;
        // Angular contribution for each body: (r × axis)ᵀ · I_world⁻¹ · (r × axis)
        // = dot(r × axis, inv_inertia_apply(r × axis))
        let ra_x = v_cross(contact.r_a, axis);
        let rb_x = v_cross(contact.r_b, axis);
        let ang_a = v_dot(
            ra_x,
            inv_inertia_apply(
                self.bodies[ia].orientation,
                self.bodies[ia].inv_inertia_local,
                ra_x,
            ),
        );
        let ang_b = v_dot(
            rb_x,
            inv_inertia_apply(
                self.bodies[ib].orientation,
                self.bodies[ib].inv_inertia_local,
                rb_x,
            ),
        );
        let denom = inv_m + ang_a + ang_b;
        if denom <= Fix128::ZERO {
            Fix128::ZERO
        } else {
            Fix128::ONE / denom
        }
    }
}

impl TgsHooks for Pgs6DofOrientedHooks<'_> {
    fn begin_substep(&mut self, sub_dt: Fix128) {
        let dv = v_scale(self.cfg.gravity, sub_dt);
        for body in self.bodies.iter_mut() {
            if body.is_dynamic {
                body.linear_velocity = v_add(body.linear_velocity, dv);
            }
        }
        for i in 0..self.contacts.len() {
            let rel_v = self.contact_rel_velocity(i);
            let vn = v_dot(rel_v, self.contacts[i].normal);
            self.restitution_bias[i] = if vn < Fix128::ZERO {
                -(self.contacts[i].restitution * vn)
            } else {
                Fix128::ZERO
            };
            if self.cfg.warmstart {
                let cached = self.cache.take(self.contacts[i].stable_id);
                self.contacts[i].accum_normal = cached.normal;
                self.contacts[i].accum_tangent1 = cached.tangent1;
                self.contacts[i].accum_tangent2 = cached.tangent2;
                if cached.normal != Fix128::ZERO
                    || cached.tangent1 != Fix128::ZERO
                    || cached.tangent2 != Fix128::ZERO
                {
                    let impulse = v_add(
                        v_add(
                            v_scale(self.contacts[i].normal, cached.normal),
                            v_scale(self.contacts[i].tangent1, cached.tangent1),
                        ),
                        v_scale(self.contacts[i].tangent2, cached.tangent2),
                    );
                    self.apply_impulse(i, impulse);
                }
            } else {
                self.contacts[i].accum_normal = Fix128::ZERO;
                self.contacts[i].accum_tangent1 = Fix128::ZERO;
                self.contacts[i].accum_tangent2 = Fix128::ZERO;
            }
        }
    }

    fn velocity_iteration(&mut self, _sub_dt: Fix128) {
        for i in 0..self.contacts.len() {
            // Normal with Newton restitution.
            let contact = self.contacts[i];
            let rel_v = self.contact_rel_velocity(i);
            let vn = v_dot(rel_v, contact.normal);
            let eff_m_n = self.effective_mass(i, contact.normal);
            if eff_m_n > Fix128::ZERO {
                let lambda = (self.restitution_bias[i] - vn) * eff_m_n;
                let prev = contact.accum_normal;
                let mut new_acc = prev + lambda;
                if new_acc < Fix128::ZERO {
                    new_acc = Fix128::ZERO;
                }
                let applied = new_acc - prev;
                if applied != Fix128::ZERO {
                    self.apply_impulse(i, v_scale(contact.normal, applied));
                    self.contacts[i].accum_normal = new_acc;
                }
            }

            let contact = self.contacts[i];
            let rel_v = self.contact_rel_velocity(i);

            let vt1 = v_dot(rel_v, contact.tangent1);
            let eff_m_t1 = self.effective_mass(i, contact.tangent1);
            let mut tentative_t1 = contact.accum_tangent1;
            if eff_m_t1 > Fix128::ZERO {
                tentative_t1 = contact.accum_tangent1 + (-vt1 * eff_m_t1);
            }
            let vt2 = v_dot(rel_v, contact.tangent2);
            let eff_m_t2 = self.effective_mass(i, contact.tangent2);
            let mut tentative_t2 = contact.accum_tangent2;
            if eff_m_t2 > Fix128::ZERO {
                tentative_t2 = contact.accum_tangent2 + (-vt2 * eff_m_t2);
            }
            // Friction cone joint clamp.
            let limit = contact.friction * contact.accum_normal;
            let mag_sq = tentative_t1 * tentative_t1 + tentative_t2 * tentative_t2;
            let limit_sq = limit * limit;
            let (new_t1, new_t2) = if mag_sq > limit_sq && limit > Fix128::ZERO {
                let mag_sq_f = mag_sq.to_f32().max(1e-12);
                let scale_f = limit.to_f32() / mag_sq_f.sqrt();
                let scale = Fix128::from_f32(scale_f);
                (tentative_t1 * scale, tentative_t2 * scale)
            } else if limit == Fix128::ZERO {
                (Fix128::ZERO, Fix128::ZERO)
            } else {
                (tentative_t1, tentative_t2)
            };
            let d_t1 = new_t1 - contact.accum_tangent1;
            let d_t2 = new_t2 - contact.accum_tangent2;
            if d_t1 != Fix128::ZERO || d_t2 != Fix128::ZERO {
                let impulse = v_add(
                    v_scale(contact.tangent1, d_t1),
                    v_scale(contact.tangent2, d_t2),
                );
                self.apply_impulse(i, impulse);
                self.contacts[i].accum_tangent1 = new_t1;
                self.contacts[i].accum_tangent2 = new_t2;
            }
        }
    }

    fn position_iteration(&mut self, _sub_dt: Fix128) {
        for i in 0..self.contacts.len() {
            let contact = self.contacts[i];
            let excess = contact.penetration - self.cfg.slop;
            if excess <= Fix128::ZERO {
                continue;
            }
            let eff_m = self.effective_mass(i, contact.normal);
            if eff_m <= Fix128::ZERO {
                continue;
            }
            let correction = self.cfg.baumgarte * excess * eff_m;
            let disp = v_scale(contact.normal, correction);
            if self.bodies[contact.body_a].is_dynamic {
                let m = self.bodies[contact.body_a].inv_mass;
                self.bodies[contact.body_a].position =
                    v_sub(self.bodies[contact.body_a].position, v_scale(disp, m));
            }
            if self.bodies[contact.body_b].is_dynamic {
                let m = self.bodies[contact.body_b].inv_mass;
                self.bodies[contact.body_b].position =
                    v_add(self.bodies[contact.body_b].position, v_scale(disp, m));
            }
        }
    }

    fn end_substep(&mut self, sub_dt: Fix128) {
        // Advance position + orientation for every dynamic body.
        for body in self.bodies.iter_mut() {
            body.advance(sub_dt);
        }
        for contact in self.contacts.iter() {
            self.cache.set(
                contact.stable_id,
                CachedImpulse {
                    normal: contact.accum_normal,
                    tangent1: contact.accum_tangent1,
                    tangent2: contact.accum_tangent2,
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_with_zero_omega_stays_identity() {
        let q = integrate_orientation(QuatFix::IDENTITY, V_ZERO, Fix128::from_f32(1.0 / 60.0));
        // Should remain the identity quaternion within Fix128 precision.
        assert_eq!(q.x, Fix128::ZERO);
        assert_eq!(q.y, Fix128::ZERO);
        assert_eq!(q.z, Fix128::ZERO);
        assert_eq!(q.w, Fix128::ONE);
    }

    #[test]
    fn spin_about_y_produces_positive_y_component() {
        // ω = 1 rad/s about +Y, integrated over a small step, gives a
        // quaternion with a small positive y component and w ≈ 1.
        let omega = [Fix128::ZERO, Fix128::ONE, Fix128::ZERO];
        let q = integrate_orientation(
            QuatFix::IDENTITY,
            omega,
            Fix128::from_f32(0.02), // 20 ms
        );
        assert!(q.y.to_f32() > 0.0, "y component should be positive");
        // Unit-length invariant.
        let mag = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).to_f32();
        assert!((mag - 1.0).abs() < 1e-3, "quaternion must stay unit-length");
    }

    #[test]
    fn integrate_orientation_is_bit_perfect_deterministic() {
        let q = QuatFix::IDENTITY;
        let omega = [
            Fix128::from_f32(0.3),
            Fix128::from_f32(-0.5),
            Fix128::from_f32(0.7),
        ];
        let dt = Fix128::from_f32(1.0 / 60.0);
        let a = integrate_orientation(q, omega, dt);
        let b = integrate_orientation(q, omega, dt);
        assert_eq!(a.x, b.x);
        assert_eq!(a.y, b.y);
        assert_eq!(a.z, b.z);
        assert_eq!(a.w, b.w);
    }

    #[test]
    fn advance_moves_position_and_rotates() {
        let mut body = Body6DofOrientedState {
            is_dynamic: true,
            linear_velocity: [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            angular_velocity: [Fix128::ZERO, Fix128::from_f32(0.5), Fix128::ZERO],
            inv_mass: Fix128::ONE,
            inv_inertia_local: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            stable_id: 1,
            ..Default::default()
        };
        let start_pos = body.position;
        let start_q = body.orientation;
        body.advance(Fix128::from_f32(1.0 / 60.0));
        assert!(
            body.position[0].to_f32() > start_pos[0].to_f32(),
            "position should advance along +X"
        );
        assert!(
            body.orientation.y.to_f32() > start_q.y.to_f32(),
            "orientation should pick up +Y component"
        );
    }

    #[test]
    fn static_body_advance_is_a_noop() {
        let mut body = Body6DofOrientedState {
            linear_velocity: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            angular_velocity: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            is_dynamic: false,
            ..Default::default()
        };
        let snapshot = body;
        body.advance(Fix128::from_f32(1.0 / 60.0));
        assert_eq!(body.position, snapshot.position);
        assert_eq!(body.orientation.x, snapshot.orientation.x);
        assert_eq!(body.orientation.w, snapshot.orientation.w);
    }

    // --- Pgs6DofOrientedHooks ---------------------------------------------

    use crate::solver_tgs::{tgs_step, ImpulseCache, TgsConfig};

    fn ori_dynamic(id: u64, inv_m: f32, inv_i: f32, y: f32) -> Body6DofOrientedState {
        Body6DofOrientedState {
            position: [Fix128::ZERO, Fix128::from_f32(y), Fix128::ZERO],
            inv_mass: Fix128::from_f32(inv_m),
            inv_inertia_local: [
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
            ],
            is_dynamic: true,
            stable_id: id,
            ..Default::default()
        }
    }
    fn ori_ground(id: u64) -> Body6DofOrientedState {
        Body6DofOrientedState {
            is_dynamic: false,
            stable_id: id,
            ..Default::default()
        }
    }
    fn ori_floor_contact(
        a: usize,
        b: usize,
        id: u64,
        mu: f32,
        e: f32,
        pen: f32,
    ) -> ContactOriented {
        ContactOriented {
            body_a: a,
            body_b: b,
            stable_id: id,
            normal: [Fix128::ZERO, Fix128::ONE, Fix128::ZERO],
            tangent1: [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            tangent2: [Fix128::ZERO, Fix128::ZERO, Fix128::ONE],
            r_a: V_ZERO,
            r_b: V_ZERO,
            penetration: Fix128::from_f32(pen),
            friction: Fix128::from_f32(mu),
            restitution: Fix128::from_f32(e),
            accum_normal: Fix128::ZERO,
            accum_tangent1: Fix128::ZERO,
            accum_tangent2: Fix128::ZERO,
        }
    }

    #[test]
    fn oriented_gravity_only_affects_dynamic_bodies() {
        let mut bodies = [ori_dynamic(1, 1.0, 1.0, 10.0), ori_ground(2)];
        let mut contacts: [ContactOriented; 0] = [];
        let mut cache = ImpulseCache::new();
        let cfg = Pgs6DofOrientedConfig {
            warmstart: false,
            ..Pgs6DofOrientedConfig::default()
        };
        let mut hooks = Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0,
                position_iters: 0,
                warmstart: false,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        assert!(bodies[0].linear_velocity[1].to_f32() < 0.0);
        assert_eq!(bodies[1].linear_velocity, V_ZERO);
    }

    #[test]
    fn oriented_restitution_bounces_body_off_floor() {
        let mut bodies = [ori_dynamic(1, 1.0, 1.0, 0.0), ori_ground(2)];
        bodies[0].linear_velocity = [Fix128::ZERO, Fix128::from_f32(-4.0), Fix128::ZERO];
        let mut contacts = [ori_floor_contact(1, 0, 100, 0.0, 1.0, 0.0)];
        let mut cache = ImpulseCache::new();
        let cfg = Pgs6DofOrientedConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..Pgs6DofOrientedConfig::default()
        };
        let mut hooks = Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 8,
                position_iters: 0,
                warmstart: false,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        assert!(bodies[0].linear_velocity[1].to_f32() > 3.5);
    }

    #[test]
    fn oriented_end_substep_integrates_orientation() {
        let mut bodies = [ori_dynamic(1, 1.0, 1.0, 0.0)];
        bodies[0].angular_velocity = [Fix128::ZERO, Fix128::from_f32(0.5), Fix128::ZERO];
        let start_q = bodies[0].orientation;
        let mut contacts: [ContactOriented; 0] = [];
        let mut cache = ImpulseCache::new();
        let cfg = Pgs6DofOrientedConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..Pgs6DofOrientedConfig::default()
        };
        let mut hooks = Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0,
                position_iters: 0,
                warmstart: false,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        // Orientation should have picked up a small +Y component.
        assert!(bodies[0].orientation.y.to_f32() > start_q.y.to_f32());
        // And the quaternion must still be unit length.
        let q = bodies[0].orientation;
        let mag = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).to_f32();
        assert!((mag - 1.0).abs() < 1e-3);
    }

    #[test]
    fn oriented_solve_is_bit_perfect_deterministic() {
        fn run() -> [Body6DofOrientedState; 3] {
            let mut bodies = [
                ori_ground(1),
                ori_dynamic(2, 1.0, 1.0, 0.05),
                ori_dynamic(3, 1.0, 1.0, 0.15),
            ];
            let mut contacts = [
                ori_floor_contact(0, 1, 500, 0.4, 0.0, 0.05),
                ori_floor_contact(1, 2, 501, 0.4, 0.0, 0.05),
            ];
            let mut cache = ImpulseCache::new();
            let cfg = Pgs6DofOrientedConfig::default();
            let tcfg = TgsConfig::default();
            let dt = Fix128::from_f32(1.0 / 60.0);
            {
                let mut h = Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
                for _ in 0..4 {
                    tgs_step(&mut h, &tcfg, dt);
                }
            }
            bodies
        }
        let a = run();
        let b = run();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.position, y.position);
            assert_eq!(x.linear_velocity, y.linear_velocity);
            assert_eq!(x.angular_velocity, y.angular_velocity);
            assert_eq!(x.orientation.x, y.orientation.x);
            assert_eq!(x.orientation.y, y.orientation.y);
            assert_eq!(x.orientation.z, y.orientation.z);
            assert_eq!(x.orientation.w, y.orientation.w);
        }
    }

    #[test]
    fn local_to_world_uses_the_orientation() {
        let mut body = Body6DofOrientedState {
            is_dynamic: true,
            stable_id: 1,
            ..Default::default()
        };
        // Rotate 90° about +Y using the axis-angle helper.
        body.orientation =
            QuatFix::from_axis_angle(Vec3Fix::from_f32(0.0, 1.0, 0.0), Fix128::from_f32(1.5708));
        // +X in the body frame becomes ≈ -Z in the world frame after a
        // 90° rotation about +Y.
        let world = body.local_to_world([Fix128::ONE, Fix128::ZERO, Fix128::ZERO]);
        assert!(world[2].to_f32() < -0.9, "expected ≈ -Z, got {:?}", world);
    }
}
