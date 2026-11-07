//! A steady contact's penetration per substep is `g cos θ h²`: at 8 substeps
//! of a 60 Hz frame on a 0.63 rad incline it is 1.4e-4 m, at 32 it is 8.8e-6
//! and at 64 it is 2.2e-6. The contact between a sphere body and a box goes
//! through GJK / EPA, whose termination threshold is `2⁻¹⁶ ≈ 1.5e-5`, so below
//! that the reported depth (and with it the normal multiplier `λ_n`, the
//! friction cap `μ λ_n / h` and the stick limit `μ λ_n`) is not the
//! penetration, and the contact can be dropped for a substep. This oracle pins
//! the detection: a sphere placed on the turned face of a static box with
//! penetration `δ` must be pushed out by `δ` (the normal correction equals the
//! depth) for `δ` from 1e-3 down to 1e-7, within `δ · 2⁻²⁰ + 2⁻⁴⁰` — the bound
//! of an exact closest-point depth (the sphere's centre is outside the box, so
//! the depth is `r` minus a distance the fixed-point arithmetic resolves to
//! `2⁻⁴⁰` after the normalised face normal's `2⁻²⁰`); EPA's own documented
//! accuracy is its threshold `2⁻¹⁶`, which the deep case below holds today.
//!
//! Known defect (ignored until the fix lands): the depth EPA reports is off by
//! up to its threshold. At rest the push-out is short by a constant
//! 2.3e-8 … 5.8e-8 for `δ` = 1e-7 … 1e-3 (measured). On a sliding ball at 64
//! substeps the error is larger and changes with the simplex: the ball sinks
//! to a standing penetration of 7.65e-6 while each substep reports only the
//! new increment (5.3e-7), then one substep reports the whole 8.2e-6, pushes
//! it out at once and the ball leaves the face at 2.9e-2 m/s for a few
//! substeps without a contact — the floor in the convergence sweep of the
//! sliding ball (`analytic_contact_friction_convergence.rs`) at 32 and 64
//! substeps; at 8 substeps the increment (3.5e-5) is above the threshold and
//! no kick occurs. Root fix: a sphere's depth from GJK's exact closest point
//! (`r − d`, no EPA), filed as `shallow contact depth`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 placement and bounds"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

const R: f64 = 0.5;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// The sphere's distance above the turned face minus `R` (negative = penetration)
fn gap(w: &PhysicsWorld, ball: usize, theta: f64) -> f64 {
    let p = w.bodies[ball].position;
    p.x.to_f64() * -theta.sin() + p.y.to_f64() * theta.cos() - (0.5 + R)
}

/// One substep without gravity: the only motion is the contact's normal correction.
fn pushed_out_by(theta: f64, delta: f64) -> (f64, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: Fix128::ZERO,
        substeps: 1,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.bodies[slab].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)).normalize();
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 2.0),
        }
    ));
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let up = 0.5 + R - delta;
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(normal[0] * up, normal[1] * up, 0.0), Fix128::ONE),
        fx(R),
    );
    let before = gap(&w, ball, theta);
    w.step(fx(1.0 / 480.0));
    (gap(&w, ball, theta) - before, w.contact_constraints.len())
}

const DELTAS: [f64; 6] = [1e-3, 1e-4, 1e-5, 2e-6, 1e-6, 1e-7];

/// oracle: the normal correction equals the penetration for every `δ` in the
/// sweep, within `δ · 2⁻²⁰ + 2⁻⁴⁰` (an exact depth); the contact exists at every `δ`.
#[test]
#[ignore = "known defect: shallow contact depth below the EPA threshold"]
fn a_shallow_penetration_is_detected_and_corrected_by_its_depth() {
    let theta = 0.629_f64;
    let rows: Vec<(f64, f64, usize)> = DELTAS
        .iter()
        .map(|&delta| {
            let (moved, contacts) = pushed_out_by(theta, delta);
            (delta, moved, contacts)
        })
        .collect();
    for &(delta, moved, contacts) in &rows {
        println!(
            "δ {delta:.1e}: pushed out {moved:.3e} (error {:+.3e}, exact bound {:.1e}), contacts {contacts}",
            moved - delta,
            delta * 2f64.powi(-20) + 2f64.powi(-40)
        );
    }
    let mut checked = 0;
    for &(delta, moved, contacts) in &rows {
        let bound = delta * 2f64.powi(-20) + 2f64.powi(-40);
        assert!(contacts >= 1, "δ {delta:.1e}: no contact");
        assert!(
            (moved - delta).abs() <= bound,
            "δ {delta:.1e}: pushed out {moved:.3e}, bound {bound:.1e}"
        );
        checked += 1;
    }
    assert_eq!(checked, DELTAS.len());
}

/// Today's law: EPA stops when a face is within its threshold `2⁻¹⁶` of the
/// polytope, so for a penetration above the threshold the correction equals
/// `δ` within `2⁻¹⁶` (measured 5.8e-8 at `δ = 1e-3`); this holds now and gives
/// the oracle teeth on the placement, the contact and the push-out direction.
#[test]
fn a_penetration_above_the_epa_threshold_is_corrected_within_the_threshold() {
    let mut checked = 0;
    for &delta in &[1e-3, 1e-4] {
        let (moved, contacts) = pushed_out_by(0.629, delta);
        assert_eq!(contacts, 1, "δ {delta:.1e}");
        assert!(
            (moved - delta).abs() <= 2f64.powi(-16),
            "δ {delta:.1e}: pushed out {moved:.3e}"
        );
        checked += 1;
    }
    assert_eq!(checked, 2);
}
