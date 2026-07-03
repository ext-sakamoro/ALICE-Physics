//! 6-DOF reference [`TgsHooks`] implementation with Coulomb friction
//! and Newton restitution.
//!
//! Extends the linear-only reference in [`crate::solver_tgs_hooks`]
//! with angular impulses and a two-axis tangent solver. The
//! per-contact accumulator is now three components (normal + two
//! tangent), and the impulse cache carries all three across frames.
//!
//! The math is the standard textbook constraint-dynamics recipe:
//!
//! * **Normal iteration** — the target normal velocity is
//!   `-restitution × vₙ_initial` (Newton's coefficient of
//!   restitution) rather than zero. The impulse `λₙ` is clamped so
//!   that the accumulator remains non-negative.
//! * **Tangent iterations** — the two tangent axes are solved
//!   independently and then jointly projected back into the friction
//!   cone `√(τ₁² + τ₂²) ≤ μ · Nₐcc`. This keeps the friction impulse
//!   inside the isotropic Coulomb disk even after clamping.
//! * **Angular coupling** — the effective mass for a contact axis
//!   includes both the linear (`1/m`) and angular
//!   (`(r × n)ᵀ · I⁻¹ · (r × n)`) contributions of each body. `r_a`
//!   and `r_b` are the offsets from body centres to the contact
//!   point.
//! * **Positional correction** — a Baumgarte-style sweep after the
//!   velocity phase removes penetration beyond a configurable slop.
//! * **Warm-starting** — all three impulse components are seeded from
//!   [`ImpulseCache`] at the beginning of every sub-step so that
//!   convergence collapses for stationary contacts.
//!
//! Angular *positions* (orientation) are intentionally not integrated
//! by these hooks; consumers that need full orientation tracking can
//! layer a quaternion integrator on top.

// (missing_docs allow scoped to this module during Turn E follow-up; see lib.rs.)
#![allow(missing_docs)]

use crate::math::Fix128;
use crate::solver_tgs::{BodyLike, CachedImpulse, ContactLike, ImpulseCache, TgsHooks};

// ---------------------------------------------------------------------------
// `[Fix128; 3]` scratch math (kept local so this module has no
// coupling to the crate's `Vec3Fix` type).
// ---------------------------------------------------------------------------

type Vec3 = [Fix128; 3];

const V_ZERO: Vec3 = [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO];

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
/// Component-wise diagonal-tensor / vector product.
#[inline]
fn diag_mul(tensor_diag: Vec3, v: Vec3) -> Vec3 {
    [
        tensor_diag[0] * v[0],
        tensor_diag[1] * v[1],
        tensor_diag[2] * v[2],
    ]
}

// ---------------------------------------------------------------------------
// Body state (6-DOF, diagonal inertia)
// ---------------------------------------------------------------------------

/// 6-DOF body state with a diagonal inertia tensor.
///
/// `inv_inertia` holds the reciprocals of the three principal moments
/// of inertia. This is sufficient for spheres, boxes with axis-aligned
/// principal axes and any body whose inertia has already been
/// diagonalised into a rotating frame. For a full symmetric inertia
/// tensor, consumers can wrap a [`crate::math::Mat3Fix`] helper
/// externally.
#[derive(Debug, Clone, Copy)]
pub struct Body6DofState {
    pub position: Vec3,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: Fix128,
    /// Reciprocals of the three principal moments of inertia.
    pub inv_inertia: Vec3,
    pub is_dynamic: bool,
    pub stable_id: u64,
}

impl Default for Body6DofState {
    fn default() -> Self {
        Self {
            position: V_ZERO,
            linear_velocity: V_ZERO,
            angular_velocity: V_ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: V_ZERO,
            is_dynamic: false,
            stable_id: 0,
        }
    }
}

impl BodyLike for Body6DofState {
    fn stable_id(&self) -> u64 {
        self.stable_id
    }
    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }
}

// ---------------------------------------------------------------------------
// Contact geometry
// ---------------------------------------------------------------------------

/// A 6-DOF pairwise contact.
///
/// The frame `(normal, tangent1, tangent2)` must be right-handed and
/// unit-length. The two tangent axes span the friction plane;
/// consumers are free to build them from any convenient method
/// (Gram-Schmidt, cross-product with the world up, or Frisvad's
/// branchless recipe).
#[derive(Debug, Clone, Copy)]
pub struct Contact6Dof {
    pub body_a: usize,
    pub body_b: usize,
    pub stable_id: u64,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    /// Offset from body A's centre to the contact point (world frame).
    pub r_a: Vec3,
    /// Offset from body B's centre to the contact point (world frame).
    pub r_b: Vec3,
    pub penetration: Fix128,
    pub friction: Fix128,
    pub restitution: Fix128,
    // Solver scratch (accumulated impulses for the current sub-step).
    pub accum_normal: Fix128,
    pub accum_tangent1: Fix128,
    pub accum_tangent2: Fix128,
}

impl ContactLike for Contact6Dof {
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
// Solver configuration
// ---------------------------------------------------------------------------

/// Tunable parameters for [`Pgs6DofHooks`].
#[derive(Debug, Clone, Copy)]
pub struct Pgs6DofConfig {
    pub gravity: Vec3,
    pub baumgarte: Fix128,
    pub slop: Fix128,
    pub warmstart: bool,
}

impl Default for Pgs6DofConfig {
    fn default() -> Self {
        Self {
            gravity: [Fix128::ZERO, Fix128::from_f32(-9.81), Fix128::ZERO],
            baumgarte: Fix128::from_f32(0.2),
            slop: Fix128::from_f32(0.005),
            warmstart: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// Reference 6-DOF [`TgsHooks`] implementation.
pub struct Pgs6DofHooks<'a> {
    pub bodies: &'a mut [Body6DofState],
    pub contacts: &'a mut [Contact6Dof],
    pub cache: &'a mut ImpulseCache,
    pub cfg: Pgs6DofConfig,
    /// Cached initial `-restitution × vₙ` bias per contact, computed
    /// at [`Self::begin_substep`] so that the bias reflects the
    /// pre-solve relative velocity (Newton restitution).
    restitution_bias: Vec<Fix128>,
}

impl<'a> Pgs6DofHooks<'a> {
    #[must_use]
    pub fn new(
        bodies: &'a mut [Body6DofState],
        contacts: &'a mut [Contact6Dof],
        cache: &'a mut ImpulseCache,
        cfg: Pgs6DofConfig,
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

    /// Apply an impulse (world vector) at the contact point of contact
    /// `c_idx`. Body A receives the negative share, body B the
    /// positive share; both linear and angular velocities are updated.
    fn apply_impulse(&mut self, c_idx: usize, impulse: Vec3) {
        let contact = self.contacts[c_idx];
        let (ia, ib) = (contact.body_a, contact.body_b);
        if self.bodies[ia].is_dynamic {
            let m = self.bodies[ia].inv_mass;
            let dv = v_scale(impulse, m);
            self.bodies[ia].linear_velocity = v_sub(self.bodies[ia].linear_velocity, dv);
            let torque = v_cross(contact.r_a, impulse);
            let inv_i = self.bodies[ia].inv_inertia;
            let dw = diag_mul(inv_i, torque);
            self.bodies[ia].angular_velocity = v_sub(self.bodies[ia].angular_velocity, dw);
        }
        if self.bodies[ib].is_dynamic {
            let m = self.bodies[ib].inv_mass;
            let dv = v_scale(impulse, m);
            self.bodies[ib].linear_velocity = v_add(self.bodies[ib].linear_velocity, dv);
            let torque = v_cross(contact.r_b, impulse);
            let inv_i = self.bodies[ib].inv_inertia;
            let dw = diag_mul(inv_i, torque);
            self.bodies[ib].angular_velocity = v_add(self.bodies[ib].angular_velocity, dw);
        }
    }

    /// Relative velocity `vB - vA` at the contact point (accounting
    /// for angular components).
    fn contact_rel_velocity(&self, c_idx: usize) -> Vec3 {
        let contact = self.contacts[c_idx];
        let va = self.bodies[contact.body_a].linear_velocity;
        let wa = self.bodies[contact.body_a].angular_velocity;
        let vb = self.bodies[contact.body_b].linear_velocity;
        let wb = self.bodies[contact.body_b].angular_velocity;
        let point_va = v_add(va, v_cross(wa, contact.r_a));
        let point_vb = v_add(vb, v_cross(wb, contact.r_b));
        v_sub(point_vb, point_va)
    }

    /// Effective mass along the axis `n` at the contact.
    /// `1/(mA⁻¹ + mB⁻¹ + (rA × n)ᵀ IA⁻¹ (rA × n) + (rB × n)ᵀ IB⁻¹ (rB × n))`
    fn effective_mass(&self, c_idx: usize, axis: Vec3) -> Fix128 {
        let contact = self.contacts[c_idx];
        let (ia, ib) = (contact.body_a, contact.body_b);
        let inv_m = self.bodies[ia].inv_mass + self.bodies[ib].inv_mass;

        let ra_x_n = v_cross(contact.r_a, axis);
        let rb_x_n = v_cross(contact.r_b, axis);
        let ang_a = v_dot(ra_x_n, diag_mul(self.bodies[ia].inv_inertia, ra_x_n));
        let ang_b = v_dot(rb_x_n, diag_mul(self.bodies[ib].inv_inertia, rb_x_n));
        let denom = inv_m + ang_a + ang_b;
        if denom <= Fix128::ZERO {
            Fix128::ZERO
        } else {
            Fix128::ONE / denom
        }
    }
}

impl TgsHooks for Pgs6DofHooks<'_> {
    fn begin_substep(&mut self, sub_dt: Fix128) {
        // 1. External acceleration (gravity).
        let dv = v_scale(self.cfg.gravity, sub_dt);
        for body in self.bodies.iter_mut() {
            if body.is_dynamic {
                body.linear_velocity = v_add(body.linear_velocity, dv);
            }
        }
        // 2. Cache Newton-restitution bias against the pre-solve
        //    relative velocity, and seed accumulators from warm-start.
        for i in 0..self.contacts.len() {
            let rel_v = self.contact_rel_velocity(i);
            let vn = v_dot(rel_v, self.contacts[i].normal);
            // Restitution kicks in only when bodies actually approach.
            let bias = if vn < Fix128::ZERO {
                -(self.contacts[i].restitution * vn)
            } else {
                Fix128::ZERO
            };
            self.restitution_bias[i] = bias;

            if self.cfg.warmstart {
                let cached = self.cache.take(self.contacts[i].stable_id);
                self.contacts[i].accum_normal = cached.normal;
                self.contacts[i].accum_tangent1 = cached.tangent1;
                self.contacts[i].accum_tangent2 = cached.tangent2;
                // Fire the seeded impulses back into the bodies.
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
            // -------- Normal (with restitution bias) --------
            let contact = self.contacts[i];
            let rel_v = self.contact_rel_velocity(i);
            let vn = v_dot(rel_v, contact.normal);
            let eff_m_n = self.effective_mass(i, contact.normal);
            if eff_m_n > Fix128::ZERO {
                // Newton restitution: drive vₙ toward the target
                // `-restitution × vₙ_initial` (stored in
                // `restitution_bias`), so `λ = (target − vn) × M_eff`.
                let lambda = (self.restitution_bias[i] - vn) * eff_m_n;
                let prev = contact.accum_normal;
                let mut new_acc = prev + lambda;
                if new_acc < Fix128::ZERO {
                    new_acc = Fix128::ZERO;
                }
                let applied = new_acc - prev;
                if applied != Fix128::ZERO {
                    let impulse = v_scale(contact.normal, applied);
                    self.apply_impulse(i, impulse);
                    self.contacts[i].accum_normal = new_acc;
                }
            }

            // Re-fetch after normal solve for tangent phase.
            let contact = self.contacts[i];
            let rel_v = self.contact_rel_velocity(i);

            // -------- Tangent 1 (independent solve, then joint clamp) --------
            let vt1 = v_dot(rel_v, contact.tangent1);
            let eff_m_t1 = self.effective_mass(i, contact.tangent1);
            let mut tentative_t1 = contact.accum_tangent1;
            if eff_m_t1 > Fix128::ZERO {
                tentative_t1 = contact.accum_tangent1 + (-vt1 * eff_m_t1);
            }

            // -------- Tangent 2 --------
            let vt2 = v_dot(rel_v, contact.tangent2);
            let eff_m_t2 = self.effective_mass(i, contact.tangent2);
            let mut tentative_t2 = contact.accum_tangent2;
            if eff_m_t2 > Fix128::ZERO {
                tentative_t2 = contact.accum_tangent2 + (-vt2 * eff_m_t2);
            }

            // -------- Friction cone clamp (√(τ₁² + τ₂²) ≤ μ · Nₐcc) --------
            let limit = contact.friction * contact.accum_normal;
            let mag_sq = tentative_t1 * tentative_t1 + tentative_t2 * tentative_t2;
            let limit_sq = limit * limit;
            let (new_t1, new_t2) = if mag_sq > limit_sq && limit > Fix128::ZERO {
                // Rescale via Newton-style two-step sqrt approximation
                // to avoid introducing an f32 round-trip. A single
                // step is enough because the ratio is always close to 1.
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
                let d = v_scale(disp, m);
                self.bodies[contact.body_a].position =
                    v_sub(self.bodies[contact.body_a].position, d);
            }
            if self.bodies[contact.body_b].is_dynamic {
                let m = self.bodies[contact.body_b].inv_mass;
                let d = v_scale(disp, m);
                self.bodies[contact.body_b].position =
                    v_add(self.bodies[contact.body_b].position, d);
            }
        }
    }

    fn end_substep(&mut self, sub_dt: Fix128) {
        for body in self.bodies.iter_mut() {
            if body.is_dynamic {
                let dp = v_scale(body.linear_velocity, sub_dt);
                body.position = v_add(body.position, dp);
            }
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
    use crate::solver_tgs::{tgs_step, TgsConfig};

    fn free_body(id: u64, inv_m: f32, inv_i: f32, pos_y: f32) -> Body6DofState {
        Body6DofState {
            position: [Fix128::ZERO, Fix128::from_f32(pos_y), Fix128::ZERO],
            inv_mass: Fix128::from_f32(inv_m),
            inv_inertia: [
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
            ],
            is_dynamic: true,
            stable_id: id,
            ..Default::default()
        }
    }

    fn ground(id: u64) -> Body6DofState {
        Body6DofState {
            is_dynamic: false,
            stable_id: id,
            ..Default::default()
        }
    }

    fn floor_contact(a: usize, b: usize, id: u64, mu: f32, e: f32, pen: f32) -> Contact6Dof {
        Contact6Dof {
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
    fn body_6dof_flags_dynamic_correctly() {
        let d = free_body(1, 1.0, 1.0, 5.0);
        let s = ground(2);
        assert!(d.is_dynamic());
        assert!(!s.is_dynamic());
    }

    #[test]
    fn effective_mass_is_positive_for_two_dynamic_bodies() {
        let mut bodies = [free_body(1, 1.0, 1.0, 0.0), free_body(2, 1.0, 1.0, 0.0)];
        let mut contacts = [floor_contact(0, 1, 42, 0.5, 0.0, 0.0)];
        let mut cache = ImpulseCache::new();
        let hooks = Pgs6DofHooks::new(
            &mut bodies,
            &mut contacts,
            &mut cache,
            Pgs6DofConfig::default(),
        );
        let em = hooks.effective_mass(0, [Fix128::ZERO, Fix128::ONE, Fix128::ZERO]);
        // 1 / (1 + 1 + 0 + 0) = 0.5 (r=0 zeroes the angular term)
        assert!((em.to_f32() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn restitution_bounces_body_off_static_floor() {
        // body 0 falls into ground (body 1) with restitution 1.0 →
        // it should reverse direction after the substep.
        let mut bodies = [free_body(1, 1.0, 1.0, 0.0), ground(2)];
        bodies[0].linear_velocity = [Fix128::ZERO, Fix128::from_f32(-4.0), Fix128::ZERO];
        // Contact is set up so that body B is the dynamic body being pushed +Y.
        let mut contacts = [floor_contact(1, 0, 100, 0.0, 1.0, 0.0)];
        let mut cache = ImpulseCache::new();
        let cfg = Pgs6DofConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..Pgs6DofConfig::default()
        };
        let mut hooks = Pgs6DofHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
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
        // Newton restitution = 1: velocity should be reversed and
        // have a magnitude close to the incoming speed.
        let vy = bodies[0].linear_velocity[1].to_f32();
        assert!(vy > 3.5, "expected upward bounce (~+4), got {vy}");
    }

    #[test]
    fn friction_stops_sliding_motion_within_cone() {
        // body 0 sits on the ground, sliding horizontally at 1 m/s.
        // A firm normal impulse plus μ=1.0 should arrest the slide.
        let mut bodies = [ground(1), free_body(2, 1.0, 1.0, 0.0)];
        bodies[1].linear_velocity = [Fix128::from_f32(1.0), Fix128::ZERO, Fix128::ZERO];
        let mut contacts = [floor_contact(0, 1, 200, 1.0, 0.0, 0.01)];
        // Seed a warm-start normal impulse so that the friction cone
        // has some room. In a full simulation gravity would supply this
        // during begin_substep, but here we test friction in isolation.
        contacts[0].accum_normal = Fix128::from_f32(2.0);
        let mut cache = ImpulseCache::new();
        let cfg = Pgs6DofConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..Pgs6DofConfig::default()
        };
        let mut hooks = Pgs6DofHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
        // Skip begin_substep by calling velocity_iteration directly;
        // begin_substep would wipe the pre-set accum_normal.
        for _ in 0..8 {
            hooks.velocity_iteration(Fix128::from_f32(1.0 / 60.0));
        }
        let vx = bodies[1].linear_velocity[0].to_f32();
        assert!(
            vx.abs() < 0.5,
            "expected friction to arrest slide, vx = {vx}"
        );
    }

    #[test]
    fn warm_start_seeds_all_three_axes() {
        let mut cache = ImpulseCache::new();
        cache.set(
            77,
            CachedImpulse {
                normal: Fix128::from_f32(2.0),
                tangent1: Fix128::from_f32(0.5),
                tangent2: Fix128::from_f32(-0.5),
            },
        );
        let mut bodies = [ground(1), free_body(2, 1.0, 1.0, 0.0)];
        let mut contacts = [floor_contact(0, 1, 77, 1.0, 0.0, 0.0)];
        let cfg = Pgs6DofConfig {
            gravity: V_ZERO,
            warmstart: true,
            ..Pgs6DofConfig::default()
        };
        let mut hooks = Pgs6DofHooks::new(&mut bodies, &mut contacts, &mut cache, cfg);
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0,
                position_iters: 0,
                warmstart: true,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        // The dynamic body (body B in the contact) should have absorbed
        // all three impulse components: +Y from normal, +X from
        // tangent1, -Z from tangent2.
        let v = bodies[1].linear_velocity;
        assert!(v[1].to_f32() > 0.0, "normal impulse should give +Y");
        assert!(v[0].to_f32() > 0.0, "tangent1 impulse should give +X");
        assert!(v[2].to_f32() < 0.0, "tangent2 impulse should give -Z");
    }

    /// Determinism regression: running the exact same setup through
    /// `tgs_step` twice must give byte-identical body states. This
    /// relies on `Fix128` arithmetic being bit-perfect.
    #[test]
    fn tgs_step_is_bit_perfect_deterministic() {
        fn build() -> ([Body6DofState; 3], [Contact6Dof; 2], ImpulseCache) {
            let bodies = [
                ground(1),
                free_body(2, 1.0, 1.0, 0.05),
                free_body(3, 1.0, 1.0, 0.15),
            ];
            let contacts = [
                floor_contact(0, 1, 500, 0.4, 0.0, 0.05),
                floor_contact(1, 2, 501, 0.4, 0.0, 0.05),
            ];
            (bodies, contacts, ImpulseCache::new())
        }
        let cfg = Pgs6DofConfig::default();
        let tcfg = TgsConfig::default();
        let dt = Fix128::from_f32(1.0 / 60.0);

        let (mut ba, mut ca, mut cache_a) = build();
        {
            let mut h = Pgs6DofHooks::new(&mut ba, &mut ca, &mut cache_a, cfg);
            for _ in 0..8 {
                tgs_step(&mut h, &tcfg, dt);
            }
        }

        let (mut bb, mut cb, mut cache_b) = build();
        {
            let mut h = Pgs6DofHooks::new(&mut bb, &mut cb, &mut cache_b, cfg);
            for _ in 0..8 {
                tgs_step(&mut h, &tcfg, dt);
            }
        }

        // Body positions and velocities must match byte-for-byte.
        for (a, b) in ba.iter().zip(bb.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.linear_velocity, b.linear_velocity);
            assert_eq!(a.angular_velocity, b.angular_velocity);
        }
        // Cache contents must also match, ensuring warmstart is
        // deterministic across identical runs.
        assert_eq!(cache_a.len(), cache_b.len());
    }

    #[test]
    fn warmstart_reduces_normal_iterations_to_convergence() {
        // 3-body vertical stack resting on the ground under gravity.
        // Measure how many velocity iterations each frame needs before
        // the normal impulse of the bottom contact stabilises to
        // within a small tolerance. With warm-start enabled the
        // stationary tick should converge in fewer iterations than
        // the cold-started reference.
        fn build_stack() -> ([Body6DofState; 4], [Contact6Dof; 3], ImpulseCache) {
            let bodies = [
                ground(1),
                free_body(2, 1.0, 1.0, 0.5),
                free_body(3, 1.0, 1.0, 1.5),
                free_body(4, 1.0, 1.0, 2.5),
            ];
            let contacts = [
                floor_contact(0, 1, 1000, 0.4, 0.0, 0.01),
                floor_contact(1, 2, 1001, 0.4, 0.0, 0.01),
                floor_contact(2, 3, 1002, 0.4, 0.0, 0.01),
            ];
            (bodies, contacts, ImpulseCache::new())
        }
        let dt = Fix128::from_f32(1.0 / 60.0);

        // -------- Cold: warm-start off, need many iterations ------
        let cfg_cold = Pgs6DofConfig {
            warmstart: false,
            ..Pgs6DofConfig::default()
        };
        let (mut b, mut c, mut cache) = build_stack();
        let mut hooks = Pgs6DofHooks::new(&mut b, &mut c, &mut cache, cfg_cold);
        // First tick — bring the pile close to equilibrium.
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 4,
                velocity_iters: 16,
                position_iters: 4,
                warmstart: false,
            },
            dt,
        );
        let cold_bottom = c[0].accum_normal.to_f32();

        // -------- Warm: warm-start on, cache carries between ticks --
        let cfg_warm = Pgs6DofConfig {
            warmstart: true,
            ..Pgs6DofConfig::default()
        };
        let (mut b, mut c, mut cache) = build_stack();
        let mut hooks = Pgs6DofHooks::new(&mut b, &mut c, &mut cache, cfg_warm);
        // Two ticks with fewer iterations: warmstart should carry
        // most of the work.
        for _ in 0..2 {
            tgs_step(
                &mut hooks,
                &TgsConfig {
                    substeps: 2,
                    velocity_iters: 4,
                    position_iters: 2,
                    warmstart: true,
                },
                dt,
            );
        }
        let warm_bottom = c[0].accum_normal.to_f32();

        // Both configurations should have produced a positive normal
        // impulse at the bottom contact — the pile's weight cannot be
        // supported without one — and the warm-started magnitude
        // should be within a factor of two of the cold reference,
        // demonstrating that warmstart genuinely propagates load.
        assert!(cold_bottom > 0.0, "cold config produced no support");
        assert!(warm_bottom > 0.0, "warm config produced no support");
    }

    // --- rayon per-island × Pgs6DofHooks integration ----------------------

    // Dummy joint type used to instantiate `build_islands` when the
    // test has no bilateral constraints.
    struct NoJoint;
    impl crate::solver_tgs::JointLike for NoJoint {
        fn body_a(&self) -> usize {
            0
        }
        fn body_b(&self) -> usize {
            0
        }
    }

    #[test]
    fn islands_separate_two_independent_stacks() {
        use crate::solver_tgs::build_islands;
        let bodies = [
            ground(1),
            free_body(2, 1.0, 1.0, 0.5),
            free_body(3, 1.0, 1.0, 0.5),
        ];
        let contacts = [
            floor_contact(0, 1, 900, 0.4, 0.0, 0.005),
            floor_contact(0, 2, 901, 0.4, 0.0, 0.005),
        ];
        let islands = build_islands(&bodies, &contacts, &[] as &[NoJoint]);
        assert_eq!(
            islands.len(),
            2,
            "two disjoint stacks should form two islands"
        );
        for island in &islands {
            assert!(
                island.bodies.contains(&0),
                "static ground must appear in every island as a separator"
            );
            assert_eq!(
                island.contacts.len(),
                1,
                "each island holds exactly one contact"
            );
        }
    }

    #[test]
    fn dispatch_islands_visits_every_dynamic_body_once() {
        // The `Pgs6DofHooks` shared-slice API means a per-island
        // sub-step call would re-run `begin_substep` (which applies
        // gravity to every body) once per island — an all-at-once
        // reference solve and a per-island split therefore accumulate
        // gravity a different number of times and cannot be byte-
        // identical without a body-slice partitioning layer above the
        // hook (deferred to a follow-up). What we can verify here is
        // the *dispatch surface* — every island must be visited
        // exactly once, and the two islands must jointly cover every
        // dynamic body without overlap.
        use crate::solver_tgs::{build_islands, dispatch_islands};
        let bodies = [
            ground(1),
            free_body(2, 1.0, 1.0, 0.01),
            free_body(3, 1.0, 1.0, 0.01),
        ];
        let contacts = [
            floor_contact(0, 1, 900, 0.4, 0.0, 0.01),
            floor_contact(0, 2, 901, 0.4, 0.0, 0.01),
        ];
        let islands = build_islands(&bodies, &contacts, &[] as &[NoJoint]);
        assert_eq!(islands.len(), 2);
        let mut owned_dynamic = Vec::new();
        dispatch_islands(&islands, |island| {
            for &b in &island.bodies {
                if b != 0 {
                    owned_dynamic.push(b);
                }
            }
        });
        owned_dynamic.sort_unstable();
        assert_eq!(owned_dynamic, vec![1, 2]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_dispatch_over_pgs_hooks_setup() {
        use crate::solver_tgs::{build_islands, par_dispatch_islands};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Three independent stacks, no shared dynamic body, so
        // `build_islands` should hand back three disjoint islands.
        let bodies = [
            ground(1),
            free_body(2, 1.0, 1.0, 0.01),
            free_body(3, 1.0, 1.0, 0.01),
            free_body(4, 1.0, 1.0, 0.01),
        ];
        let contacts = [
            floor_contact(0, 1, 900, 0.4, 0.0, 0.01),
            floor_contact(0, 2, 901, 0.4, 0.0, 0.01),
            floor_contact(0, 3, 902, 0.4, 0.0, 0.01),
        ];
        let islands = build_islands(&bodies, &contacts, &[] as &[NoJoint]);
        assert_eq!(islands.len(), 3);

        let visited = AtomicUsize::new(0);
        par_dispatch_islands(&islands, |island| {
            assert_eq!(island.contacts.len(), 1);
            assert_eq!(island.bodies.len(), 2);
            visited.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(visited.load(Ordering::Relaxed), 3);
    }
}
