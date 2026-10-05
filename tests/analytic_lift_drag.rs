//! Oracles for `lift_drag::LiftDragSurface`: thin-aerofoil theory, the
//! lifting-line finite-wing correction, induced drag, the maximum lift-to-drag
//! ratio, force directions, stall, degenerate inputs, and a steady glide run
//! through `PhysicsWorld`.
//!
//! # Where the expected values come from
//!
//! - Thin aerofoil: `C_L = 2π (α − α₀)` (Anderson, *Fundamentals of
//!   Aerodynamics*, §4.7).
//! - Lifting line (elliptic loading, Oswald `e`): `a = a₀ / (1 + a₀/(π e AR))`,
//!   `C_Di = C_L² / (π e AR)` (Anderson §5.3).
//! - Drag polar `C_D = C_D0 + C_L²/(π e AR)`: `(L/D)max = ½ √(π e AR / C_D0)`
//!   at `C_L* = √(π e AR C_D0)` (Anderson, *Aircraft Performance and Design*, §5.4).
//! - Flat plate after stall: `C_L = 2 sin α cos α`, `C_D = C_D0 + 2 sin² α`.
//! - Steady glide: `tan γ = D / L = C_D / C_L`, `½ ρ V² S √(C_L² + C_D²) = W`.
//!
//! Every expected value is evaluated here with `f64` from those formulas
//! (the reference is `ref_coeffs` below, written from the module doc, not by
//! calling the crate); the crate's `Fix128` result is compared through
//! `to_f64()`.
//!
//! # Tolerances
//!
//! Direct coefficient / force checks: relative `1e-9`. The crate's CORDIC
//! `sin_cos` / `atan2` are bounded at `1e-11` by their own sweep test
//! (`src/math.rs::transcendental_sweep_matches_f64_reference`); the rest is
//! a handful of `Fix128` multiplications (2⁻⁶⁴ each). A dropped term, a wrong
//! slope or a wrong sign is off by at least `1e-3` relative.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 transcendentals here compute closed-form references outside the
// crate, not simulation state (same convention as the other analytic tests).
#![allow(clippy::disallowed_methods)]

use std::f64::consts::PI;

use alice_physics::lift_drag::{
    AeroLoad, LiftDragError, LiftDragParams, LiftDragSurface, THIN_AIRFOIL_LIFT_SLOPE,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const TOL: f64 = 1e-9;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn to3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = if expected.abs() < 1e-12 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    };
    assert!(
        err <= tol,
        "{label}: {actual} vs {expected} (err {err} > {tol})"
    );
}

/// Wing in f64, mirrors `LiftDragParams`.
#[derive(Clone, Copy)]
struct RefWing {
    s: f64,
    ar: f64,
    e: f64,
    a0: f64,
    alpha0: f64,
    stall: f64,
    trans: f64,
    cd0: f64,
}

const WING: RefWing = RefWing {
    s: 0.5,
    ar: 8.0,
    e: 0.9,
    a0: 2.0 * PI,
    alpha0: 0.0,
    stall: 0.25,
    trans: 0.1,
    cd0: 0.03,
};

fn params(w: RefWing) -> LiftDragParams {
    LiftDragParams {
        area_m2: fx(w.s),
        aspect_ratio: fx(w.ar),
        oswald_efficiency: fx(w.e),
        section_lift_slope_per_rad: if w.a0 == 2.0 * PI {
            THIN_AIRFOIL_LIFT_SLOPE
        } else {
            fx(w.a0)
        },
        zero_lift_angle_rad: fx(w.alpha0),
        stall_angle_rad: fx(w.stall),
        stall_transition_rad: fx(w.trans),
        zero_lift_drag_coefficient: fx(w.cd0),
        chord_axis_local: Vec3Fix::UNIT_X,
        lift_axis_local: Vec3Fix::UNIT_Y,
        center_of_pressure_local: Vec3Fix::ZERO,
    }
}

fn surface(w: RefWing) -> LiftDragSurface {
    LiftDragSurface::new(params(w)).expect("valid wing")
}

fn ref_slope(w: RefWing) -> f64 {
    w.a0 / (1.0 + w.a0 / (PI * w.e * w.ar))
}

fn wrap(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    }
    if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// The coefficient model of the module doc, in f64.
fn ref_coeffs(w: RefWing, alpha: f64) -> (f64, f64) {
    let a = ref_slope(w);
    let k = 1.0 / (PI * w.e * w.ar);
    let ae = wrap(alpha - w.alpha0);
    let x = ae.abs();
    let sg = ae.signum();
    let at = w.stall + w.trans;
    if x <= w.stall {
        let cl = a * ae;
        (cl, w.cd0 + k * cl * cl)
    } else if x < at {
        let t = (x - w.stall) / w.trans;
        let cl = sg * ((1.0 - t) * a * w.stall + t * (2.0 * at).sin());
        let cd =
            (1.0 - t) * (w.cd0 + k * (a * w.stall).powi(2)) + t * (w.cd0 + 2.0 * at.sin().powi(2));
        (cl, cd)
    } else {
        (2.0 * ae.sin() * ae.cos(), w.cd0 + 2.0 * ae.sin().powi(2))
    }
}

fn cl(s: &LiftDragSurface, alpha: f64) -> f64 {
    s.coefficients(fx(alpha)).lift.to_f64()
}

fn cd(s: &LiftDragSurface, alpha: f64) -> f64 {
    s.coefficients(fx(alpha)).drag.to_f64()
}

// ---------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------

#[test]
fn thin_airfoil_slope_is_two_pi_in_the_infinite_aspect_ratio_limit() {
    // oracle: thin-aerofoil theory, C_L = 2π (α − α₀). AR = 1e9 stands in for
    // infinity: the lifting-line factor is 1 + 2/(e·1e9) = 1 + 2e-9.
    let w = RefWing {
        ar: 1e9,
        e: 1.0,
        alpha0: -0.03,
        ..WING
    };
    let s = surface(w);
    rel_close("a(AR→∞)", s.lift_slope_per_rad().to_f64(), 2.0 * PI, 3e-9);
    for alpha in [0.0, 0.02, 0.05, -0.05] {
        rel_close(
            &format!("C_L({alpha})"),
            cl(&s, alpha),
            2.0 * PI * (alpha + 0.03),
            3e-9,
        );
    }
}

#[test]
fn finite_wing_slope_follows_lifting_line() {
    // oracle: a = a₀ / (1 + a₀/(π e AR)), AR = 8, e = 0.9 → 4.917 /rad.
    let s = surface(WING);
    let a = 2.0 * PI / (1.0 + 2.0 * PI / (PI * 0.9 * 8.0));
    rel_close("a", s.lift_slope_per_rad().to_f64(), a, TOL);
    rel_close("C_L(0.1)", cl(&s, 0.1), a * 0.1, TOL);
    // Lower AR → lower slope (monotone in AR).
    let low = surface(RefWing { ar: 4.0, ..WING });
    assert!(low.lift_slope_per_rad() < s.lift_slope_per_rad());
}

#[test]
fn induced_drag_follows_the_drag_polar() {
    // oracle: C_D = C_D0 + C_L² / (π e AR) with C_L = a α (attached flow).
    let s = surface(WING);
    let a = ref_slope(WING);
    for alpha in [-0.2, -0.05, 0.0, 0.07, 0.2] {
        let c_l = a * alpha;
        let expected = 0.03 + c_l * c_l / (PI * 0.9 * 8.0);
        rel_close(&format!("C_D({alpha})"), cd(&s, alpha), expected, TOL);
    }
}

#[test]
fn maximum_lift_to_drag_matches_the_polar_closed_form() {
    // oracle: (L/D)max = ½ √(π e AR / C_D0) = 7.6845 for AR 8, e 0.9, C_D0
    // 0.03, reached at C_L* = √(π e AR C_D0) = 0.8238, α* = 0.1675 rad (< α_s).
    // Scan α in 1e-5 rad steps: L/D is flat at its maximum (quadratic), so
    // the scan error is far below the tolerance.
    let s = surface(WING);
    let mut best = (0.0_f64, 0.0_f64);
    let mut i = 0;
    while i <= 25_000 {
        let alpha = f64::from(i) * 1e-5;
        let c = s.coefficients(fx(alpha));
        let ld = c.lift.to_f64() / c.drag.to_f64();
        if ld > best.0 {
            best = (ld, c.lift.to_f64());
        }
        i += 1;
    }
    let ld_max = 0.5 * (PI * 0.9 * 8.0 / 0.03).sqrt();
    let cl_star = (PI * 0.9 * 8.0 * 0.03).sqrt();
    rel_close("(L/D)max", best.0, ld_max, 1e-8);
    rel_close("C_L*", best.1, cl_star, 1e-3);
}

#[test]
fn stall_lowers_lift_and_coefficients_are_continuous() {
    let s = surface(WING);
    let a = ref_slope(WING);
    let (st, at) = (0.25, 0.35);
    // Peak of the attached branch at α_s.
    rel_close("C_L(α_s)", cl(&s, st), a * st, TOL);
    // Past the stall angle the lift is lower than at the stall angle.
    assert!(cl(&s, st + 0.02) < cl(&s, st));
    assert!(cl(&s, at) < cl(&s, st));
    // C_L(α_t) is the flat-plate value sin 2α_t.
    rel_close("C_L(α_t)", cl(&s, at), (2.0 * at).sin(), TOL);
    // Continuity at both seams (1e-9 rad on either side).
    for seam in [st, at] {
        let d_l = (cl(&s, seam + 1e-9) - cl(&s, seam - 1e-9)).abs();
        let d_d = (cd(&s, seam + 1e-9) - cd(&s, seam - 1e-9)).abs();
        assert!(
            d_l < 1e-7 && d_d < 1e-7,
            "jump at {seam}: ΔC_L {d_l}, ΔC_D {d_d}"
        );
    }
    // Every branch against the f64 reference model.
    for alpha in [
        0.26,
        0.3,
        0.34,
        0.5,
        PI / 4.0,
        1.2,
        PI / 2.0,
        2.5,
        -0.3,
        -1.0,
        3.1,
    ] {
        let (rl, rd) = ref_coeffs(WING, alpha);
        rel_close(&format!("C_L({alpha})"), cl(&s, alpha), rl, TOL);
        rel_close(&format!("C_D({alpha})"), cd(&s, alpha), rd, TOL);
    }
    // Flat plate landmarks: C_L(45°) = 1, C_D(90°) = C_D0 + 2.
    rel_close("C_L(45°)", cl(&s, PI / 4.0), 1.0, TOL);
    rel_close("C_D(90°)", cd(&s, PI / 2.0), 2.03, TOL);
}

#[test]
fn lift_is_odd_and_drag_even_in_the_effective_angle() {
    let s = surface(WING);
    for alpha in [0.01, 0.1, 0.25, 0.3, 0.8, 2.0] {
        let p = s.coefficients(fx(alpha));
        let m = s.coefficients(fx(-alpha));
        assert!(
            (p.lift + m.lift).to_f64().abs() < 1e-15,
            "C_L odd at {alpha}"
        );
        assert!(
            (p.drag - m.drag).to_f64().abs() < 1e-15,
            "C_D even at {alpha}"
        );
    }
    // Zero-lift angle shifts the symmetry centre: C_L(α₀) = 0.
    let cambered = surface(RefWing {
        alpha0: -0.04,
        ..WING
    });
    assert!(cl(&cambered, -0.04).abs() < 1e-15);
    rel_close(
        "C_L(0) cambered",
        cl(&cambered, 0.0),
        ref_slope(WING) * 0.04,
        TOL,
    );
}

#[test]
fn angles_outside_minus_pi_pi_are_wrapped() {
    // Documented: any α is wrapped into (−π, π] before evaluation.
    let s = surface(WING);
    for alpha in [7.0, -9.5, 100.0] {
        let (rl, rd) = ref_coeffs(WING, alpha);
        rel_close(&format!("C_L({alpha})"), cl(&s, alpha), rl, 1e-8);
        rel_close(&format!("C_D({alpha})"), cd(&s, alpha), rd, 1e-8);
    }
    // α = π (flow from behind, edge on): flat plate, C_L = 0, C_D = C_D0.
    assert!(cl(&s, PI).abs() < 1e-9);
    rel_close("C_D(π)", cd(&s, PI), 0.03, 1e-8);
}

// ---------------------------------------------------------------------
// Loads on a body
// ---------------------------------------------------------------------

fn body_moving(v: [f64; 3]) -> RigidBody {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2));
    b.velocity = v3(v[0], v[1], v[2]);
    b
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

const RHO: f64 = 1.225;

#[test]
fn lift_is_perpendicular_and_drag_antiparallel_to_the_flow() {
    // oracle: L = q S C_L along ŝ × û (⟂ û), D = q S C_D along −û.
    let s = surface(WING);
    let speed = 15.0;
    for alpha in [0.1_f64, -0.12, 0.2, 0.6] {
        // Surface moving through still air at angle of attack α: u = V(cos α, −sin α, 0).
        let u = [speed * alpha.cos(), -speed * alpha.sin(), 0.0];
        let load = s.load(&body_moving(u), Vec3Fix::ZERO, fx(RHO)).unwrap();
        rel_close("α", load.angle_of_attack_rad.unwrap().to_f64(), alpha, TOL);
        rel_close("|u_p|", load.airspeed_m_s.to_f64(), speed, TOL);
        let (rl, rd) = ref_coeffs(WING, alpha);
        let q = 0.5 * RHO * speed * speed;
        let (lift, drag) = (to3(load.lift), to3(load.drag));
        assert!(
            dot(lift, u).abs() / (norm(u) * norm(lift)) < 1e-12,
            "lift ⟂ u"
        );
        rel_close("|L|", norm(lift), q * 0.5 * rl.abs(), TOL);
        rel_close("|D|", norm(drag), q * 0.5 * rd, TOL);
        rel_close("D·û", dot(drag, u) / norm(u), -q * 0.5 * rd, TOL);
        // Positive α lifts towards +y (the lift axis), negative α towards −y.
        assert_eq!(lift[1] > 0.0, alpha > 0.0, "lift sign at α = {alpha}");
        // F = L + D
        let f = to3(load.force);
        for i in 0..3 {
            assert!((f[i] - lift[i] - drag[i]).abs() < 1e-9);
        }
    }
}

#[test]
fn symmetric_wing_at_zero_incidence_has_zero_lift_and_reversed_angle_reverses_lift() {
    let s = surface(WING);
    // Zero up to the CORDIC `atan2` residual (|α| ~ 1e-13 rad for an exact
    // zero numerator, measured): |L| ≤ q S a |α| ≈ 61 · 4.9 · 1e-12.
    let load = s
        .load(&body_moving([20.0, 0.0, 0.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    assert!(load.angle_of_attack_rad.unwrap().to_f64().abs() < 1e-12);
    for c in to3(load.lift) {
        assert!(c.abs() < 1e-9, "lift component {c} at α = 0");
    }
    let up = s
        .load(&body_moving([20.0, -1.0, 0.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    let down = s
        .load(&body_moving([20.0, 1.0, 0.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    assert!(up.lift.y > Fix128::ZERO && down.lift.y < Fix128::ZERO);
    assert!((up.lift.y + down.lift.y).to_f64().abs() < 1e-12);
    assert!((up.lift.x - down.lift.x).to_f64().abs() < 1e-12);
}

#[test]
fn wind_enters_the_relative_flow() {
    // A body at rest in a wind w sees the same flow as a body moving at −w in
    // still air (u = v − w).
    let s = surface(WING);
    let moving = s
        .load(&body_moving([12.0, -1.5, 0.3]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    let windy = s
        .load(&body_moving([0.0, 0.0, 0.0]), v3(-12.0, 1.5, -0.3), fx(RHO))
        .unwrap();
    for (a, b) in to3(moving.force).iter().zip(to3(windy.force)) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    // A tailwind equal to the body velocity leaves no flow at all.
    let still = s
        .load(
            &body_moving([12.0, -1.5, 0.0]),
            v3(12.0, -1.5, 0.0),
            fx(RHO),
        )
        .unwrap();
    assert_eq!(still.force, Vec3Fix::ZERO);
    assert_eq!(still.angle_of_attack_rad, None);
}

#[test]
fn body_rotation_turns_the_wing_axes() {
    // Pitching the body nose-up by θ about +z, flying level: α = θ.
    let s = surface(WING);
    for theta in [0.05_f64, -0.1, 0.2] {
        let mut b = body_moving([15.0, 0.0, 0.0]);
        b.set_rotation(QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)));
        let load = s.load(&b, Vec3Fix::ZERO, fx(RHO)).unwrap();
        rel_close(
            "α = θ",
            load.angle_of_attack_rad.unwrap().to_f64(),
            theta,
            1e-9,
        );
    }
}

#[test]
fn angular_velocity_moves_the_centre_of_pressure_through_the_air() {
    // v_cp = v + ω × r. r = (−1, 0, 0) (tail surface 1 m behind the COM),
    // ω = (0, 0, w): ω × r = (0, −w, 0), so the tail moves down and
    // α = atan(w / V).
    let mut p = params(WING);
    p.center_of_pressure_local = v3(-1.0, 0.0, 0.0);
    let s = LiftDragSurface::new(p).unwrap();
    let (vx, w) = (10.0_f64, 0.5);
    let mut b = body_moving([vx, 0.0, 0.0]);
    b.angular_velocity = v3(0.0, 0.0, w);
    let load = s.load(&b, Vec3Fix::ZERO, fx(RHO)).unwrap();
    rel_close(
        "α",
        load.angle_of_attack_rad.unwrap().to_f64(),
        (w / vx).atan(),
        1e-9,
    );
    rel_close(
        "|u_p|",
        load.airspeed_m_s.to_f64(),
        (vx * vx + w * w).sqrt(),
        1e-9,
    );
    assert_eq!(load.application_point, v3(-1.0, 0.0, 0.0));
}

#[test]
fn offset_centre_of_pressure_gives_moment_and_apply_adds_it() {
    // τ = r × F about the COM, and `apply` changes ω by I⁻¹ τ dt (identity
    // rotation: component-wise with inv_inertia).
    let mut p = params(WING);
    p.center_of_pressure_local = v3(-0.5, 0.0, 0.0);
    let s = LiftDragSurface::new(p).unwrap();
    let alpha = 0.1_f64;
    let speed = 15.0;
    let u = [speed * alpha.cos(), -speed * alpha.sin(), 0.0];
    let mut b = body_moving(u);
    let load = s.load(&b, Vec3Fix::ZERO, fx(RHO)).unwrap();
    // Expected force from the f64 model.
    let (rl, rd) = ref_coeffs(WING, alpha);
    let q = 0.5 * RHO * speed * speed * 0.5;
    let uh = [alpha.cos(), -alpha.sin(), 0.0];
    let lift_dir = [alpha.sin(), alpha.cos(), 0.0]; // ŝ × û with ŝ = ẑ
    let f = [
        q * (rl * lift_dir[0] - rd * uh[0]),
        q * (rl * lift_dir[1] - rd * uh[1]),
        0.0,
    ];
    // r × F with r = (−0.5, 0, 0): (0, 0, −0.5 F_y)
    rel_close("τ_z", load.torque.z.to_f64(), -0.5 * f[1], 1e-9);
    assert!(load.torque.x.to_f64().abs() < 1e-12 && load.torque.y.to_f64().abs() < 1e-12);
    // Tail surface behind the COM: positive lift gives a nose-down (−z) moment.
    assert!(load.torque.z < Fix128::ZERO);

    let dt = 0.01;
    let inv_m = b.inv_mass.to_f64();
    let inv_iz = b.inv_inertia.z.to_f64();
    let applied: AeroLoad = s.apply(&mut b, Vec3Fix::ZERO, fx(RHO), fx(dt)).unwrap();
    assert_eq!(applied, load);
    rel_close(
        "Δω_z",
        b.angular_velocity.z.to_f64(),
        inv_iz * (-0.5 * f[1]) * dt,
        1e-9,
    );
    rel_close(
        "Δv_y",
        b.velocity.y.to_f64() - u[1],
        inv_m * f[1] * dt,
        1e-7,
    );
}

// ---------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------

#[test]
fn degenerate_flow_and_density() {
    let s = surface(WING);
    // No velocity at all → zero load, α undefined.
    let rest = s
        .load(&body_moving([0.0, 0.0, 0.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    assert_eq!(rest.force, Vec3Fix::ZERO);
    assert_eq!(rest.torque, Vec3Fix::ZERO);
    assert_eq!(rest.angle_of_attack_rad, None);
    assert_eq!(rest.airspeed_m_s, Fix128::ZERO);
    // Pure spanwise flow is dropped by the strip model → same as no flow.
    let span = s
        .load(&body_moving([0.0, 0.0, 30.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    assert_eq!(span.force, Vec3Fix::ZERO);
    assert_eq!(span.angle_of_attack_rad, None);
    // ρ = 0 → zero force, α still reported.
    let vac = s
        .load(&body_moving([10.0, -1.0, 0.0]), Vec3Fix::ZERO, Fix128::ZERO)
        .unwrap();
    assert_eq!(vac.force, Vec3Fix::ZERO);
    assert!(vac.angle_of_attack_rad.is_some());
    // ρ < 0 → error, and `apply` leaves the body unchanged.
    let mut b = body_moving([10.0, -1.0, 0.0]);
    let before = b;
    assert_eq!(
        s.apply(&mut b, Vec3Fix::ZERO, fx(-1.0), fx(0.01)),
        Err(LiftDragError::NegativeAirDensity)
    );
    assert_eq!(b, before);
}

#[test]
fn invalid_parameters_are_rejected() {
    let base = params(WING);
    let cases: Vec<(LiftDragParams, LiftDragError)> = vec![
        (
            LiftDragParams {
                area_m2: Fix128::ZERO,
                ..base
            },
            LiftDragError::NonPositiveArea,
        ),
        (
            LiftDragParams {
                area_m2: fx(-1.0),
                ..base
            },
            LiftDragError::NonPositiveArea,
        ),
        (
            LiftDragParams {
                aspect_ratio: Fix128::ZERO,
                ..base
            },
            LiftDragError::NonPositiveAspectRatio,
        ),
        (
            LiftDragParams {
                aspect_ratio: fx(-3.0),
                ..base
            },
            LiftDragError::NonPositiveAspectRatio,
        ),
        (
            LiftDragParams {
                oswald_efficiency: Fix128::ZERO,
                ..base
            },
            LiftDragError::OswaldEfficiencyOutOfRange,
        ),
        (
            LiftDragParams {
                oswald_efficiency: fx(1.01),
                ..base
            },
            LiftDragError::OswaldEfficiencyOutOfRange,
        ),
        (
            LiftDragParams {
                section_lift_slope_per_rad: Fix128::ZERO,
                ..base
            },
            LiftDragError::NonPositiveLiftSlope,
        ),
        (
            LiftDragParams {
                stall_angle_rad: Fix128::ZERO,
                ..base
            },
            LiftDragError::StallAngleOutOfRange,
        ),
        (
            LiftDragParams {
                stall_angle_rad: Fix128::HALF_PI,
                ..base
            },
            LiftDragError::StallAngleOutOfRange,
        ),
        (
            LiftDragParams {
                stall_transition_rad: Fix128::ZERO,
                ..base
            },
            LiftDragError::StallTransitionOutOfRange,
        ),
        (
            LiftDragParams {
                stall_transition_rad: fx(1.4),
                ..base
            },
            LiftDragError::StallTransitionOutOfRange,
        ),
        (
            LiftDragParams {
                zero_lift_drag_coefficient: fx(-0.01),
                ..base
            },
            LiftDragError::NegativeZeroLiftDrag,
        ),
        (
            LiftDragParams {
                chord_axis_local: Vec3Fix::ZERO,
                ..base
            },
            LiftDragError::DegenerateAxes,
        ),
        (
            LiftDragParams {
                lift_axis_local: Vec3Fix::ZERO,
                ..base
            },
            LiftDragError::DegenerateAxes,
        ),
        (
            LiftDragParams {
                lift_axis_local: v3(-2.0, 0.0, 0.0),
                ..base
            },
            LiftDragError::DegenerateAxes,
        ),
        // a·α_s = 0.1·0.25 is far below sin 2(α_s+Δ): no lift loss at stall.
        (
            LiftDragParams {
                section_lift_slope_per_rad: fx(0.1),
                ..base
            },
            LiftDragError::NoLiftLossAtStall,
        ),
    ];
    for (p, e) in cases {
        assert_eq!(LiftDragSurface::new(p), Err(e));
    }
    // e = 1 exactly and a non-orthogonal (but not parallel) lift axis are accepted;
    // the lift axis is made orthogonal to the chord.
    let ok = LiftDragSurface::new(LiftDragParams {
        oswald_efficiency: Fix128::ONE,
        lift_axis_local: v3(0.3, 2.0, 0.0),
        ..base
    })
    .unwrap();
    let load = ok
        .load(&body_moving([10.0, 0.0, 0.0]), Vec3Fix::ZERO, fx(RHO))
        .unwrap();
    // α = 0 up to the CORDIC atan2 residual: the chord, not the given lift
    // axis (tilted 8.5° towards it), sets the zero of α.
    assert!(load.angle_of_attack_rad.unwrap().to_f64().abs() < 1e-12);
    assert_eq!(ok.params().lift_axis_local, v3(0.3, 2.0, 0.0));
}

// ---------------------------------------------------------------------
// Steady glide through PhysicsWorld
// ---------------------------------------------------------------------

#[test]
fn world_glider_settles_on_the_closed_form_glide_path() {
    // oracle: steady glide, tan γ = C_D / C_L and ½ ρ V² S √(C_L² + C_D²) = m g.
    // The body is pitched so that flying the path γ puts the wing at α* =
    // 0.08 rad: pitch θ = α* − γ about +z. The centre of pressure is at the
    // COM, so the attitude stays fixed and only the path and speed settle.
    // At the steady state one frame's aerodynamic impulse cancels the
    // gravity impulse exactly, so the fixed point is the closed form; the
    // remaining error is the transient. The speed mode decays with
    // τ = V / (2 g sin γ) ≈ 6.9 s; 120 s is 17 τ (e^-17 ≈ 4e-8 of the
    // 10 % initial speed error).
    let alpha_star = 0.08_f64;
    let (rl, rd) = ref_coeffs(WING, alpha_star);
    let gamma = (rd / rl).atan();
    let theta = alpha_star - gamma;
    let mass = 2.0;
    let config = SolverConfig {
        damping: Fix128::ONE,
        ..SolverConfig::default()
    };
    let g = -config.gravity.y.to_f64();
    let v_ss = (2.0 * mass * g / (RHO * WING.s * (rl * rl + rd * rd).sqrt())).sqrt();

    let s = surface(WING);
    let mut world = PhysicsWorld::new(config);
    let mut body = RigidBody::new(v3(0.0, 1000.0, 0.0), fx(mass));
    body.set_rotation(QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)));
    let v0 = 0.9 * v_ss;
    body.velocity = v3(v0 * gamma.cos(), -v0 * gamma.sin(), 0.0);
    let idx = world.add_body(body);

    let dt = Fix128::from_ratio(1, 60);
    let rho = fx(RHO);
    for _ in 0..(120 * 60) {
        let b = world.get_body_mut(idx).unwrap();
        s.apply(b, Vec3Fix::ZERO, rho, dt).unwrap();
        world.step(dt);
    }
    let b = world.get_body(idx).unwrap();
    let v = to3(b.velocity);
    let speed = norm(v);
    let path = (-v[1] / v[0]).atan();
    rel_close("V steady", speed, v_ss, 1e-5);
    assert!((path - gamma).abs() < 1e-5, "γ {path} vs {gamma}");
    rel_close("L/D = 1/tan γ", 1.0 / path.tan(), rl / rd, 1e-4);
    // The body did not rotate (no moment at the COM).
    let load = s.load(b, Vec3Fix::ZERO, rho).unwrap();
    rel_close(
        "α settles at α*",
        load.angle_of_attack_rad.unwrap().to_f64(),
        alpha_star,
        1e-4,
    );
}
