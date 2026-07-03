//! Reference [`TgsHooks`] implementation for point-mass bodies.
//!
//! The module provides self-contained data structures and a projected
//! Gauss-Seidel style step so that [`crate::solver_tgs::tgs_step`] can
//! be exercised without depending on the crate's full rigid-body
//! solver in [`crate::solver`]. The math here is the standard textbook
//! constraint-dynamics recipe:
//!
//! * **Gravity + integration** — external acceleration is added at the
//!   beginning of every sub-step; positions are advanced by the
//!   solved velocities at the end of every sub-step.
//! * **Velocity iterations** — for each contact the relative normal
//!   velocity is driven to zero by an impulse `λ` computed from
//!   `-vₙ / (mᴬ⁻¹ + mᴮ⁻¹)`. The accumulated impulse is clamped so
//!   that only pushing (never pulling) forces are allowed on a
//!   unilateral contact.
//! * **Position iterations** — a Baumgarte-style positional pass
//!   corrects penetration beyond a slop threshold to prevent visible
//!   sinking.
//! * **Warm-starting** — at the beginning of every sub-step the
//!   accumulator is seeded from [`ImpulseCache`] so that the applied
//!   impulse from the previous frame acts as the initial guess. This
//!   collapses PGS convergence for stationary piles.
//!
//! Angular quantities are intentionally omitted from this reference
//! implementation. Callers that need a full 6-DOF hook can copy the
//! shape shown here and swap in `Vec3Fix` / `Mat3Fix` math.

// (missing_docs allow scoped to this module during Turn E follow-up; see lib.rs.)
#![allow(missing_docs)]

use crate::math::Fix128;
use crate::solver_tgs::{BodyLike, CachedImpulse, ContactLike, ImpulseCache, TgsHooks};

// ---------------------------------------------------------------------------
// Small `[Fix128; 3]` helpers used only by this module.
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

// ---------------------------------------------------------------------------
// Body state (linear only, angular deferred to a follow-up)
// ---------------------------------------------------------------------------

/// Linear-only body state used by [`PgsHooks`].
///
/// Angular velocity, orientation and inertia tensor are intentionally
/// omitted so that the reference hook can stay compact and easy to
/// reason about. A full 6-DOF implementation is expected to live in a
/// separate module.
#[derive(Debug, Clone, Copy)]
pub struct SimpleBodyState {
    /// World-space centre of mass.
    pub position: Vec3,
    /// World-space linear velocity.
    pub linear_velocity: Vec3,
    /// Reciprocal of mass. Set to zero for static / kinematic bodies.
    pub inv_mass: Fix128,
    /// `true` when this body participates in solver updates.
    pub is_dynamic: bool,
    /// Frame-stable identity used by [`ImpulseCache`] look-ups.
    pub stable_id: u64,
}

impl Default for SimpleBodyState {
    fn default() -> Self {
        Self {
            position: V_ZERO,
            linear_velocity: V_ZERO,
            inv_mass: Fix128::ZERO,
            is_dynamic: false,
            stable_id: 0,
        }
    }
}

impl BodyLike for SimpleBodyState {
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

/// A pairwise contact between two [`SimpleBodyState`]s. The solver
/// keeps its per-frame scratch (`accum_normal`) here so that warm-start
/// data can be written back to [`ImpulseCache`] at the end of the
/// sub-step.
#[derive(Debug, Clone, Copy)]
pub struct SimpleContact {
    pub body_a: usize,
    pub body_b: usize,
    pub stable_id: u64,
    /// Unit-length world-space normal, oriented from body A into body
    /// B (i.e. positive normal impulse pushes B along `normal`).
    pub normal: Vec3,
    /// Signed penetration depth. Positive values mean the bodies
    /// overlap.
    pub penetration: Fix128,
    /// Coulomb friction coefficient (not used by the reference
    /// linear-only hooks but present so that callers can wire in a
    /// tangential solver later).
    pub friction: Fix128,
    /// Restitution coefficient (unused by the linear-only reference).
    pub restitution: Fix128,
    /// Accumulated normal impulse for the current sub-step. Populated
    /// by [`PgsHooks::velocity_iteration`] and written back to
    /// [`ImpulseCache`] in [`PgsHooks::end_substep`].
    pub accum_normal: Fix128,
}

impl ContactLike for SimpleContact {
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

/// Tunable parameters for [`PgsHooks`].
#[derive(Debug, Clone, Copy)]
pub struct PgsConfig {
    /// External acceleration applied at the beginning of every sub-step.
    pub gravity: Vec3,
    /// Baumgarte gain used by the positional pass to remove penetration.
    pub baumgarte: Fix128,
    /// Penetration allowance below which no positional correction is applied.
    pub slop: Fix128,
    /// When `true`, the accumulator is seeded from [`ImpulseCache`] at
    /// the beginning of every sub-step.
    pub warmstart: bool,
}

impl Default for PgsConfig {
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
// Reference PGS hooks
// ---------------------------------------------------------------------------

/// Reference [`TgsHooks`] implementation over slices of
/// [`SimpleBodyState`] and [`SimpleContact`].
pub struct PgsHooks<'a> {
    pub bodies: &'a mut [SimpleBodyState],
    pub contacts: &'a mut [SimpleContact],
    pub cache: &'a mut ImpulseCache,
    pub cfg: PgsConfig,
}

impl PgsHooks<'_> {
    /// Applies an impulse of magnitude `delta_lambda` along `contact.normal`
    /// to the two bodies referenced by the contact at `c_idx`. Body A
    /// receives the negative and body B the positive share.
    fn apply_normal_impulse(&mut self, c_idx: usize, delta_lambda: Fix128) {
        let contact = self.contacts[c_idx];
        let impulse = v_scale(contact.normal, delta_lambda);
        let (a, b) = (contact.body_a, contact.body_b);
        if self.bodies[a].is_dynamic {
            let m = self.bodies[a].inv_mass;
            let dv = v_scale(impulse, m);
            self.bodies[a].linear_velocity = v_sub(self.bodies[a].linear_velocity, dv);
        }
        if self.bodies[b].is_dynamic {
            let m = self.bodies[b].inv_mass;
            let dv = v_scale(impulse, m);
            self.bodies[b].linear_velocity = v_add(self.bodies[b].linear_velocity, dv);
        }
    }
}

impl TgsHooks for PgsHooks<'_> {
    fn begin_substep(&mut self, sub_dt: Fix128) {
        // 1. External acceleration.
        let dv = v_scale(self.cfg.gravity, sub_dt);
        for body in self.bodies.iter_mut() {
            if body.is_dynamic {
                body.linear_velocity = v_add(body.linear_velocity, dv);
            }
        }
        // 2. Seed the accumulator from the warm-start cache (or reset it).
        for i in 0..self.contacts.len() {
            if self.cfg.warmstart {
                let cached = self.cache.take(self.contacts[i].stable_id);
                self.contacts[i].accum_normal = cached.normal;
                if cached.normal > Fix128::ZERO {
                    self.apply_normal_impulse(i, cached.normal);
                }
            } else {
                self.contacts[i].accum_normal = Fix128::ZERO;
            }
        }
    }

    fn velocity_iteration(&mut self, _sub_dt: Fix128) {
        for i in 0..self.contacts.len() {
            let contact = self.contacts[i];
            let va = self.bodies[contact.body_a].linear_velocity;
            let vb = self.bodies[contact.body_b].linear_velocity;
            let rel_v = v_sub(vb, va);
            let vn = v_dot(rel_v, contact.normal);
            let inv_m_sum =
                self.bodies[contact.body_a].inv_mass + self.bodies[contact.body_b].inv_mass;
            if inv_m_sum <= Fix128::ZERO {
                continue;
            }
            let lambda = -vn / inv_m_sum;
            // Unilateral clamp: the accumulator can never become negative.
            let prev = contact.accum_normal;
            let mut new_acc = prev + lambda;
            if new_acc < Fix128::ZERO {
                new_acc = Fix128::ZERO;
            }
            let applied = new_acc - prev;
            if applied == Fix128::ZERO {
                continue;
            }
            self.apply_normal_impulse(i, applied);
            self.contacts[i].accum_normal = new_acc;
        }
    }

    fn position_iteration(&mut self, _sub_dt: Fix128) {
        for i in 0..self.contacts.len() {
            let contact = self.contacts[i];
            let excess = contact.penetration - self.cfg.slop;
            if excess <= Fix128::ZERO {
                continue;
            }
            let inv_m_sum =
                self.bodies[contact.body_a].inv_mass + self.bodies[contact.body_b].inv_mass;
            if inv_m_sum <= Fix128::ZERO {
                continue;
            }
            let correction = self.cfg.baumgarte * excess;
            let per_inv_mass = correction / inv_m_sum;
            let disp = v_scale(contact.normal, per_inv_mass);
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
        // Publish the accumulated impulse to the warm-start cache so
        // that the next tick can seed itself with it.
        for contact in self.contacts.iter() {
            self.cache.set(
                contact.stable_id,
                CachedImpulse {
                    normal: contact.accum_normal,
                    tangent1: Fix128::ZERO,
                    tangent2: Fix128::ZERO,
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

    fn dynamic_body(id: u64, inv_mass_f: f32, y_pos: f32) -> SimpleBodyState {
        SimpleBodyState {
            position: [Fix128::ZERO, Fix128::from_f32(y_pos), Fix128::ZERO],
            linear_velocity: V_ZERO,
            inv_mass: Fix128::from_f32(inv_mass_f),
            is_dynamic: true,
            stable_id: id,
        }
    }

    fn static_body(id: u64) -> SimpleBodyState {
        SimpleBodyState {
            position: V_ZERO,
            linear_velocity: V_ZERO,
            inv_mass: Fix128::ZERO,
            is_dynamic: false,
            stable_id: id,
        }
    }

    fn floor_contact(a: usize, b: usize, id: u64, penetration_f: f32) -> SimpleContact {
        SimpleContact {
            body_a: a,
            body_b: b,
            stable_id: id,
            normal: [Fix128::ZERO, Fix128::ONE, Fix128::ZERO], // pushes B upward
            penetration: Fix128::from_f32(penetration_f),
            friction: Fix128::from_f32(0.5),
            restitution: Fix128::ZERO,
            accum_normal: Fix128::ZERO,
        }
    }

    #[test]
    fn simple_body_state_reports_dynamic_flag() {
        let dyn_b = dynamic_body(1, 1.0, 5.0);
        let stat_b = static_body(2);
        assert!(dyn_b.is_dynamic());
        assert!(!stat_b.is_dynamic());
        assert_eq!(dyn_b.stable_id(), 1);
        assert_eq!(stat_b.stable_id(), 2);
    }

    #[test]
    fn simple_contact_forwards_body_indices() {
        let c = floor_contact(3, 4, 100, 0.0);
        assert_eq!(c.body_a(), 3);
        assert_eq!(c.body_b(), 4);
        assert_eq!(c.stable_id(), 100);
    }

    #[test]
    fn gravity_only_affects_dynamic_bodies() {
        let mut bodies = [dynamic_body(1, 1.0, 10.0), static_body(2)];
        let mut contacts: [SimpleContact; 0] = [];
        let mut cache = ImpulseCache::new();
        let cfg = PgsConfig {
            warmstart: false,
            ..PgsConfig::default()
        };
        let mut hooks = PgsHooks {
            bodies: &mut bodies,
            contacts: &mut contacts,
            cache: &mut cache,
            cfg,
        };
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
        // Dynamic body accelerates downward, static body untouched.
        assert!(bodies[0].linear_velocity[1].to_f32() < 0.0);
        assert_eq!(bodies[1].linear_velocity, V_ZERO);
    }

    #[test]
    fn velocity_iteration_removes_relative_normal_velocity() {
        // Body A (index 0) sits below B (index 1). A is moving up, B
        // is moving down: they approach each other. With the contact
        // normal oriented "from A into B" (i.e. +Y), the relative
        // normal velocity `dot(vB - vA, n)` is negative — the sign
        // convention the hook expects for approach.
        let mut bodies = [
            SimpleBodyState {
                linear_velocity: [Fix128::ZERO, Fix128::from_f32(1.0), Fix128::ZERO],
                ..dynamic_body(1, 1.0, -0.5)
            },
            SimpleBodyState {
                linear_velocity: [Fix128::ZERO, Fix128::from_f32(-2.0), Fix128::ZERO],
                ..dynamic_body(2, 1.0, 0.5)
            },
        ];
        let mut contacts = [floor_contact(0, 1, 500, 0.0)];
        let mut cache = ImpulseCache::new();
        let cfg = PgsConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..PgsConfig::default()
        };
        let mut hooks = PgsHooks {
            bodies: &mut bodies,
            contacts: &mut contacts,
            cache: &mut cache,
            cfg,
        };
        // One tick with a handful of velocity iterations, no gravity.
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
        // Relative normal velocity along the contact normal should
        // drop below its initial magnitude.
        let rel_vn = (bodies[1].linear_velocity[1] - bodies[0].linear_velocity[1]).to_f32();
        assert!(
            rel_vn.abs() < 3.0,
            "expected relative normal velocity to shrink, got {rel_vn}"
        );
    }

    #[test]
    fn warm_start_carries_impulse_across_ticks() {
        let mut cache = ImpulseCache::new();
        // Seed a positive normal impulse for the contact.
        cache.set(
            42,
            CachedImpulse {
                normal: Fix128::from_f32(3.0),
                tangent1: Fix128::ZERO,
                tangent2: Fix128::ZERO,
            },
        );
        let mut bodies = [dynamic_body(1, 1.0, 1.0), static_body(2)];
        let mut contacts = [floor_contact(1, 0, 42, 0.0)];
        let cfg = PgsConfig {
            gravity: V_ZERO,
            warmstart: true,
            ..PgsConfig::default()
        };
        let mut hooks = PgsHooks {
            bodies: &mut bodies,
            contacts: &mut contacts,
            cache: &mut cache,
            cfg,
        };
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0, // rely on warm-start alone
                position_iters: 0,
                warmstart: true,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        // The dynamic body sits on body_b of the contact, so the
        // positive share of the warm-start normal impulse pushed it
        // along +Y.
        assert!(bodies[0].linear_velocity[1].to_f32() > 0.0);
    }

    #[test]
    fn position_correction_ignores_penetration_below_slop() {
        // penetration below the default slop (0.005) must produce no shove.
        let mut bodies = [dynamic_body(1, 1.0, 0.001), static_body(2)];
        let start_y = bodies[0].position[1];
        let mut contacts = [floor_contact(1, 0, 7, 0.001)]; // below slop
        let mut cache = ImpulseCache::new();
        let cfg = PgsConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..PgsConfig::default()
        };
        let mut hooks = PgsHooks {
            bodies: &mut bodies,
            contacts: &mut contacts,
            cache: &mut cache,
            cfg,
        };
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0,
                position_iters: 4,
                warmstart: false,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        assert_eq!(bodies[0].position[1], start_y);
    }

    #[test]
    fn position_correction_pushes_when_penetration_exceeds_slop() {
        let mut bodies = [dynamic_body(1, 1.0, -0.1), static_body(2)];
        // Contact points from body B (static, body index 1) toward body A (dynamic, body index 0)
        // along +Y, so body A gets pushed downward (v_sub) — we want it pushed upward instead.
        // The convention in the hook is: A gets `- disp`, B gets `+ disp`. So a floor contact
        // (dynamic falls into static floor) should have A = static, B = dynamic to lift B.
        let mut contacts = [floor_contact(1, 0, 8, 0.05)];
        let start_y = bodies[0].position[1];
        let mut cache = ImpulseCache::new();
        let cfg = PgsConfig {
            gravity: V_ZERO,
            warmstart: false,
            ..PgsConfig::default()
        };
        let mut hooks = PgsHooks {
            bodies: &mut bodies,
            contacts: &mut contacts,
            cache: &mut cache,
            cfg,
        };
        tgs_step(
            &mut hooks,
            &TgsConfig {
                substeps: 1,
                velocity_iters: 0,
                position_iters: 4,
                warmstart: false,
            },
            Fix128::from_f32(1.0 / 60.0),
        );
        // Body 0 is dynamic; it should have been pushed upward.
        assert!(
            bodies[0].position[1].to_f32() > start_y.to_f32(),
            "expected upward push, from {} to {}",
            start_y.to_f32(),
            bodies[0].position[1].to_f32()
        );
    }
}
