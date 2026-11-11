//! A box resting flat on a floor sinks by the contact compliance times its load,
//! carries its weight, and shares it equally between its four corners, at every
//! step rate.
//!
//! oracle (statics of springs): the four corners of the face are springs of
//! compliance `α` (the pair's, [`CombinedMaterial::compliance`], the sum of the
//! two materials') sharing the weight `m g` equally by symmetry, so each
//! carries `m g / 4` and sinks `α m g / 4`; the reaction sums to `m g`. XPBD's
//! `α̃ = α / h²` makes this independent of the substep, so 240, 480 and 960 Hz
//! agree. Tolerances: the force within `1e-9` relative, the depth within `1e-9`
//! relative of `α m g / 4` (rounding of the Fix128 solve), the four corner
//! forces equal to the bit.
//!
//! [`CombinedMaterial::compliance`]: alice_physics_core::material::CombinedMaterial::compliance

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 powi and sqrt of the tilt and the corner arm"
)]

use alice_physics_core::material::DEFAULT_CONTACT_COMPLIANCE;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepConfig;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

#[test]
fn a_resting_box_sinks_by_alpha_times_its_corner_load_at_every_rate() {
    let g = 10.0;
    let alpha = 2.0 * DEFAULT_CONTACT_COMPLIANCE.to_f64();
    let mut wrong = Vec::new();
    for hz in [240.0, 480.0, 960.0] {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::new(Fix128::ZERO, fx(-g), Fix128::ZERO),
            ..SolverConfig::default()
        });
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
            Fix128::ZERO,
        )));
        // Awake throughout: a sleeping body's contacts are not solved.
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: u32::MAX,
            ..SleepConfig::default()
        });
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: Vec3Fix::new(fx(0.45), fx(0.06), fx(0.3)),
                },
                fx(500.0),
                Vec3Fix::new(Fix128::ZERO, fx(0.06), Fix128::ZERO),
            )
            .expect("box");
        for _ in 0..hz as usize {
            w.step(fx(1.0 / hz));
        }
        let weight = g / w.get_body(b).expect("box").inv_mass.to_f64();
        let forces: Vec<Fix128> = w
            .contact_forces(fx(1.0 / hz))
            .iter()
            .map(|(_, _, f)| *f)
            .collect();
        let total: f64 = forces.iter().map(|f| f.to_f64()).sum();
        let depth = 0.06 - w.get_body(b).expect("box").position.y.to_f64();
        let want_depth = alpha * weight / 4.0;
        if forces.len() != 4
            || forces.iter().any(|f| *f != forces[0])
            || (total - weight).abs() > 1e-9 * weight
            || (depth - want_depth).abs() > 1e-9 * want_depth.max(1e-12)
        {
            wrong.push(format!(
                "{hz} Hz: {} corner forces {:?}, sum {total} (want {weight}), depth {depth:.6e} (want {want_depth:.6e})",
                forces.len(),
                forces.iter().map(|f| f.to_f64()).collect::<Vec<_>>()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The well-conditioned part of the same scene, kept as a guard apart from the
/// to-the-bit split above: the reaction sums to `m g` (within `1e-9`), the box
/// sinks `α m g / 4` (within `1e-9`), it tilts by less than `1e-9` rad in the
/// second, and the corners' loads differ by no more than that tilt can explain
/// through the springs, `θ r_max / α` with `r_max` the corner's arm.
#[test]
fn a_resting_box_keeps_its_load_depth_and_level_within_the_bounds() {
    let g = 10.0;
    let alpha = 2.0 * DEFAULT_CONTACT_COMPLIANCE.to_f64();
    let r_max = (0.45f64 * 0.45 + 0.3 * 0.3).sqrt();
    let mut wrong = Vec::new();
    for hz in [240.0, 480.0, 960.0] {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::new(Fix128::ZERO, fx(-g), Fix128::ZERO),
            ..SolverConfig::default()
        });
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
            Fix128::ZERO,
        )));
        w.set_sleep_config(SleepConfig {
            frames_to_sleep: u32::MAX,
            ..SleepConfig::default()
        });
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: Vec3Fix::new(fx(0.45), fx(0.06), fx(0.3)),
                },
                fx(500.0),
                Vec3Fix::new(Fix128::ZERO, fx(0.06), Fix128::ZERO),
            )
            .expect("box");
        for _ in 0..hz as usize {
            w.step(fx(1.0 / hz));
        }
        let body = w.get_body(b).expect("box");
        let weight = g / body.inv_mass.to_f64();
        let forces: Vec<f64> = w
            .contact_forces(fx(1.0 / hz))
            .iter()
            .map(|(_, _, f)| f.to_f64())
            .collect();
        let total: f64 = forces.iter().sum();
        let depth = 0.06 - body.position.y.to_f64();
        let up = body
            .rotation
            .rotate_vec(Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO));
        let tilt = (up.x.to_f64().powi(2) + up.z.to_f64().powi(2)).sqrt();
        let split = forces.iter().cloned().fold(f64::MIN, f64::max)
            - forces.iter().cloned().fold(f64::MAX, f64::min);
        let split_bound = (tilt * r_max / alpha).max(1e-9 * weight);
        if (total - weight).abs() > 1e-9 * weight
            || (depth - alpha * weight / 4.0).abs() > 1e-9 * alpha * weight
            || tilt > 1e-9
            || split > split_bound
        {
            wrong.push(format!(
                "{hz} Hz: sum {total}, depth {depth:.6e}, tilt {tilt:.3e}, split {split:.3e} (bound {split_bound:.3e})"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
