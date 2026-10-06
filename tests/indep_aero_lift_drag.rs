//! Independent oracles for `lift_drag::LiftDragSurface`.
//!
//! The existing file (`analytic_lift_drag.rs`) uses a thin-aerofoil wing of
//! AR 8, e 0.9, C_D0 0.03. Here the wing is a non-thin section (`a₀ = 5.9`)
//! of AR 5.5, e 0.8, S 1.7 m², C_D0 0.021, cambered (`α₀ = −0.035`), so
//! every expected number below is new. What is checked:
//!
//! - Absolute force magnitudes `|L| = ½ ρ V² S C_L`, `|D| = ½ ρ V² S C_D` of
//!   a body in a climb/descent, with `C_L` from lifting-line theory written
//!   out here.
//! - Scaling laws: `F ∝ V²`, `F ∝ ρ`, `F ∝ S` (separate surfaces).
//! - The power identity `F · û = −|D|` (lift does no work) over the whole
//!   angle range, attached, transition, separated and reverse flow.
//! - Independence of the spanwise flow component (yawed flow).
//! - Flat-plate values at 45°, 90°, 135°, the zero-lift angle, and the
//!   transition midpoint.
//! - The documented "no flow" resolution threshold for `|u_p|`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 trig computes references outside the crate.
#![allow(clippy::disallowed_methods)]

use std::f64::consts::PI;

use alice_physics::lift_drag::{LiftDragParams, LiftDragSurface};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

const S: f64 = 1.7;
const AR: f64 = 5.5;
const E: f64 = 0.8;
const A0: f64 = 5.9;
const ALPHA0: f64 = -0.035;
const STALL: f64 = 0.26;
const TRANS: f64 = 0.09;
const CD0: f64 = 0.021;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn to3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn params(area: f64) -> LiftDragParams {
    LiftDragParams {
        area_m2: fx(area),
        aspect_ratio: fx(AR),
        oswald_efficiency: fx(E),
        section_lift_slope_per_rad: fx(A0),
        zero_lift_angle_rad: fx(ALPHA0),
        stall_angle_rad: fx(STALL),
        stall_transition_rad: fx(TRANS),
        zero_lift_drag_coefficient: fx(CD0),
        chord_axis_local: Vec3Fix::UNIT_X,
        lift_axis_local: Vec3Fix::UNIT_Y,
        center_of_pressure_local: Vec3Fix::ZERO,
    }
}

fn wing(area: f64) -> LiftDragSurface {
    LiftDragSurface::new(params(area)).expect("valid wing")
}

/// Lifting-line slope `a₀ / (1 + a₀/(π e AR))`.
fn slope() -> f64 {
    A0 / (1.0 + A0 / (PI * E * AR))
}

fn rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = if expected == 0.0 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    };
    assert!(
        err <= tol,
        "{label}: {actual} vs {expected} (err {err:e} > {tol:e})"
    );
}

/// Body moving through still air so that the angle of attack is `alpha`:
/// surface velocity `V (cos α, −sin α, 0)` (forward and down for α > 0).
fn body_at(alpha: f64, speed: f64) -> RigidBody {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(3));
    b.velocity = v3(speed * alpha.cos(), -speed * alpha.sin(), 0.0);
    b
}

#[test]
fn absolute_lift_and_drag_match_dynamic_pressure_times_coefficient() {
    // oracle: q = ½ ρ V², L = q S a (α − α₀), D = q S (C_D0 + C_L²/(π e AR)).
    // ρ = 0.9 kg/m³ (about 3 km), V = 37.5 m/s. CORDIC atan2 recovers α to
    // well below 1e-9 rad; bound 1e-7 relative.
    let rho = 0.9;
    let v = 37.5;
    let w = wing(S);
    for alpha in [-0.15, 0.0, 0.12, 0.2] {
        let load = w.load(&body_at(alpha, v), Vec3Fix::ZERO, fx(rho)).unwrap();
        let q = 0.5 * rho * v * v;
        let c_l = slope() * (alpha - ALPHA0);
        let c_d = CD0 + c_l * c_l / (PI * E * AR);
        let lift = to3(load.lift);
        let drag = to3(load.drag);
        rel_close(
            &format!("α {alpha}"),
            load.angle_of_attack_rad.unwrap().to_f64(),
            alpha,
            1e-9,
        );
        rel_close(
            &format!("|L| at α {alpha}"),
            norm(lift),
            q * S * c_l.abs(),
            1e-7,
        );
        rel_close(&format!("|D| at α {alpha}"), norm(drag), q * S * c_d, 1e-7);
        // Lift "up" (+y component) for C_L > 0.
        assert_eq!(lift[1] > 0.0, c_l > 0.0, "lift sign at α {alpha}");
        rel_close("airspeed", load.airspeed_m_s.to_f64(), v, 1e-12);
    }
}

#[test]
fn forces_scale_with_speed_squared_density_and_area() {
    let alpha = 0.08;
    let base = wing(S)
        .load(&body_at(alpha, 20.0), Vec3Fix::ZERO, fx(1.1))
        .unwrap();
    let fb = norm(to3(base.force));
    let fast = wing(S)
        .load(&body_at(alpha, 60.0), Vec3Fix::ZERO, fx(1.1))
        .unwrap();
    rel_close("F(3V)/F(V)", norm(to3(fast.force)) / fb, 9.0, 1e-7);
    let dense = wing(S)
        .load(&body_at(alpha, 20.0), Vec3Fix::ZERO, fx(1.65))
        .unwrap();
    rel_close("F(1.5ρ)/F(ρ)", norm(to3(dense.force)) / fb, 1.5, 1e-9);
    let big = wing(3.0 * S)
        .load(&body_at(alpha, 20.0), Vec3Fix::ZERO, fx(1.1))
        .unwrap();
    rel_close("F(3S)/F(S)", norm(to3(big.force)) / fb, 3.0, 1e-9);
    // The direction does not change with any of the three.
    let dir0 = to3(base.force);
    for other in [fast, dense, big] {
        let a = to3(other.force);
        let n = norm(a);
        for (x, x0) in a.iter().zip(dir0) {
            assert!((x / n - x0 / fb).abs() < 1e-9);
        }
    }
}

#[test]
fn lift_does_no_work_so_force_along_flow_is_minus_drag() {
    // oracle: P = F · u = −D |u| (L ⟂ u); over the attached, transition,
    // separated and reverse ranges.
    let w = wing(S);
    let mut a = -3.0;
    while a <= 3.0 {
        let load = w.load(&body_at(a, 12.0), Vec3Fix::ZERO, fx(1.2)).unwrap();
        let u = [a.cos(), -a.sin(), 0.0];
        let f = to3(load.force);
        let along = f[0] * u[0] + f[1] * u[1] + f[2] * u[2];
        let drag = norm(to3(load.drag));
        assert!(
            (along + drag).abs() <= 1e-9 * (1.0 + drag),
            "α {a}: F·û {along} vs −D {}",
            -drag
        );
        assert!(
            along <= 0.0,
            "α {a}: aerodynamic power must not be positive"
        );
        a += 0.05;
    }
}

#[test]
fn spanwise_flow_component_does_not_change_the_load() {
    // Strip model: the span component (z here) is dropped before α and q.
    let w = wing(S);
    let straight = body_at(0.1, 25.0);
    let mut yawed = straight;
    yawed.velocity.z = fx(17.0);
    let a = w.load(&straight, Vec3Fix::ZERO, fx(1.0)).unwrap();
    let b = w.load(&yawed, Vec3Fix::ZERO, fx(1.0)).unwrap();
    assert_eq!(a.force, b.force);
    assert_eq!(a.angle_of_attack_rad, b.angle_of_attack_rad);
    assert_eq!(a.airspeed_m_s, b.airspeed_m_s);
}

#[test]
fn flat_plate_values_at_45_90_and_135_degrees() {
    // oracle: separated branch C_L = sin 2αₑ, C_D = C_D0 + 2 sin² αₑ.
    let w = wing(S);
    for (deg, cl, cd) in [
        (45.0, 1.0, CD0 + 1.0),
        (90.0, 0.0, CD0 + 2.0),
        (135.0, -1.0, CD0 + 1.0),
        (-45.0, -1.0, CD0 + 1.0),
    ] {
        let alpha_e = deg * PI / 180.0;
        let c = w.coefficients(fx(alpha_e + ALPHA0));
        assert!(
            (c.lift.to_f64() - cl).abs() < 1e-9,
            "C_L at {deg}°: {}",
            c.lift.to_f64()
        );
        assert!(
            (c.drag.to_f64() - cd).abs() < 1e-9,
            "C_D at {deg}°: {}",
            c.drag.to_f64()
        );
    }
}

#[test]
fn zero_lift_angle_gives_exactly_zero_lift_and_profile_drag() {
    let c = wing(S).coefficients(fx(ALPHA0));
    assert_eq!(c.lift, Fix128::ZERO);
    assert_eq!(c.drag, fx(CD0));
}

#[test]
fn transition_midpoint_is_the_average_of_the_two_branches() {
    // oracle: t = ½ → C_L = σ (a α_s + sin 2α_t)/2,
    //         C_D = ((C_D0 + k (a α_s)²) + (C_D0 + 2 sin² α_t))/2.
    let w = wing(S);
    let at_ = STALL + TRANS;
    let k = 1.0 / (PI * E * AR);
    let cl_s = slope() * STALL;
    let cl = 0.5 * (cl_s + (2.0 * at_).sin());
    let cd = 0.5 * ((CD0 + k * cl_s * cl_s) + (CD0 + 2.0 * at_.sin().powi(2)));
    let mid = STALL + 0.5 * TRANS;
    for sign in [1.0, -1.0] {
        let c = w.coefficients(fx(ALPHA0 + sign * mid));
        rel_close(&format!("C_L mid {sign}"), c.lift.to_f64(), sign * cl, 1e-8);
        rel_close(&format!("C_D mid {sign}"), c.drag.to_f64(), cd, 1e-8);
    }
}

#[test]
fn speed_below_fix128_resolution_counts_as_no_flow() {
    // Documented: |u_p|² below the Fix128 resolution (|u_p| ≲ 2e-10 m/s) is
    // no flow (α = None, zero force); 1e-6 m/s is flow.
    let w = wing(S);
    let mut slow = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    slow.velocity = v3(1e-10, 0.0, 0.0);
    let l = w.load(&slow, Vec3Fix::ZERO, fx(1.2)).unwrap();
    assert_eq!(l.angle_of_attack_rad, None);
    assert_eq!(l.force, Vec3Fix::ZERO);
    slow.velocity = v3(1e-6, 0.0, 0.0);
    let l = w.load(&slow, Vec3Fix::ZERO, fx(1.2)).unwrap();
    assert!(l.angle_of_attack_rad.is_some());
    assert!(l.drag.x < Fix128::ZERO, "drag opposes +x motion");
}
