//! Audit oracles for `alice_physics::anisotropic_friction`.
//! Expected values come from the friction-ellipse closed form
//! F = -N * diag(mu) * v_hat (hand derived), not from the implementation.
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::math::{Fix128, Vec3Fix};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn model() -> AnisotropicFriction {
    // distinct, dyadic-friendly coefficients: static 1.0/0.5, kinetic 0.5/0.25
    AnisotropicFriction {
        longitudinal_static: Fix128::ONE,
        longitudinal_kinetic: r(1, 2),
        transverse_static: r(1, 2),
        transverse_kinetic: r(1, 4),
        slip_threshold_m_s: r(1, 2),
    }
}

fn close(a: Fix128, want: f64, tol: f64) -> bool {
    (a.to_f64() - want).abs() <= tol
}

/// 3-4-5 oblique slip, kinetic: v = (6, 0, 8), |v| = 10, v_hat = (0.6, 0, 0.8).
/// F = -N (mu_l * 0.6, 0, mu_t * 0.8) = -100 (0.5*0.6, 0, 0.25*0.8) = (-30, 0, -20).
#[test]
fn oblique_kinetic_matches_ellipse_closed_form() {
    let f = model().friction_force(
        Fix128::from_int(100),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(6.0, 0.0, 8.0),
    );
    assert!(close(f.x, -30.0, 1e-9), "{:?}", f.x.to_f64());
    assert!(close(f.y, 0.0, 1e-12));
    assert!(close(f.z, -20.0, 1e-9), "{:?}", f.z.to_f64());
}

/// Static/kinetic switch uses |v_tan|, not a component: v = (0.4, 0, 0.4)
/// has each component < 0.5 but |v| = 0.5657 > 0.5, so kinetic applies.
/// F = -N (mu_l_k * 0.7071, 0, mu_t_k * 0.7071) with N = 10.
#[test]
fn threshold_uses_slip_magnitude_not_components() {
    let f = model().friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(0.4, 0.0, 0.4),
    );
    let s = 0.5f64.sqrt();
    assert!(close(f.x, -10.0 * 0.5 * s, 1e-9), "{}", f.x.to_f64());
    assert!(close(f.z, -10.0 * 0.25 * s, 1e-9), "{}", f.z.to_f64());
}

/// Oblique static, exactly at the threshold (inclusive): threshold 5/8, v = (3/8, 0, 1/2),
/// |v| = 5/8 exactly (all dyadic). v_hat = (0.6, 0, 0.8).
/// F = -N (mu_l_s * 0.6, 0, mu_t_s * 0.8) = -10 (0.6, 0, 0.4) = (-6, 0, -4).
#[test]
fn oblique_at_threshold_is_static() {
    let mut m = model();
    m.slip_threshold_m_s = r(5, 8);
    let f = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        Vec3Fix::new(r(3, 8), Fix128::ZERO, r(1, 2)),
    );
    assert!(close(f.x, -6.0, 1e-9), "{}", f.x.to_f64());
    assert!(close(f.z, -4.0, 1e-9), "{}", f.z.to_f64());
    // one ulp-scale above the threshold flips to kinetic: slip 0.6 > 5/8? no, use 0.7
    let f = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        Vec3Fix::new(r(21, 40), Fix128::ZERO, r(7, 10)),
    );
    // |v| = 0.875 > 5/8: kinetic, v_hat = (0.6, 0, 0.8)
    assert!(close(f.x, -3.0, 1e-9), "{}", f.x.to_f64());
    assert!(close(f.z, -2.0, 1e-9), "{}", f.z.to_f64());
}

/// Rotated tangent frame (not world axes): t_long = (0.6, 0, 0.8),
/// t_trans = (-0.8, 0, 0.6). Slip v = 10 * t_long -> pure longitudinal:
/// F = -N mu_l_k t_long = -100*0.5*(0.6,0,0.8) = (-30, 0, -40).
/// Slip v = 10 * (t_long + t_trans)/sqrt2 -> F = -N (mu_l*c t_long + mu_t*c t_trans).
#[test]
fn rotated_and_tilted_tangent_frame() {
    let tl = v3(0.6, 0.0, 0.8);
    let tt = v3(-0.8, 0.0, 0.6);
    let m = model();
    let f = m.friction_force(Fix128::from_int(100), tl, tt, v3(6.0, 0.0, 8.0));
    assert!(close(f.x, -30.0, 1e-9), "{}", f.x.to_f64());
    assert!(close(f.z, -40.0, 1e-9), "{}", f.z.to_f64());
    // oblique in the rotated frame
    let c = 0.5f64.sqrt();
    let v = v3(10.0 * c * (0.6 - 0.8), 0.0, 10.0 * c * (0.8 + 0.6));
    let f = m.friction_force(Fix128::from_int(100), tl, tt, v);
    let want_x = -100.0 * c * (0.5 * 0.6 + 0.25 * -0.8);
    let want_z = -100.0 * c * (0.5 * 0.8 + 0.25 * 0.6);
    assert!(close(f.x, want_x, 1e-8), "{} vs {}", f.x.to_f64(), want_x);
    assert!(close(f.z, want_z, 1e-8), "{} vs {}", f.z.to_f64(), want_z);
}

/// Friction never adds energy (F . v <= 0) and |F| <= N max(mu) for slip in all quadrants.
#[test]
fn dissipative_and_bounded_in_all_quadrants() {
    let m = model();
    let n = Fix128::from_int(50);
    for &(a, b) in &[
        (3.0, 4.0),
        (-3.0, 4.0),
        (3.0, -4.0),
        (-3.0, -4.0),
        (0.0, -2.0),
        (-7.0, 0.0),
    ] {
        let v = v3(a, 0.0, b);
        let f = m.friction_force(n, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, v);
        let work = f.x.to_f64() * a + f.z.to_f64() * b;
        assert!(work <= 0.0, "F.v = {work} for v=({a},{b})");
        // signs oppose the slip components
        assert!(f.x.to_f64() * a <= 0.0 && f.z.to_f64() * b <= 0.0);
        let mag = (f.x.to_f64() * f.x.to_f64() + f.z.to_f64() * f.z.to_f64()).sqrt();
        assert!(mag <= 50.0 * 0.5 + 1e-9, "|F| = {mag}");
        // power closed form: F.v = -N (mu_l vl^2 + mu_t vt^2) / |v|
        let sp = (a * a + b * b).sqrt();
        let want = -50.0 * (0.5 * a * a + 0.25 * b * b) / sp;
        assert!((work - want).abs() < 1e-8, "{work} vs {want}");
    }
}

/// Normal component of velocity (and its tangent-frame y) is ignored; negative N gives zero.
#[test]
fn normal_velocity_ignored_and_negative_load_zero() {
    let m = model();
    let a = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(6.0, 0.0, 8.0),
    );
    let b = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(6.0, 123.0, 8.0),
    );
    assert_eq!(a, b);
    let z = m.friction_force(
        -Fix128::ONE,
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(6.0, 0.0, 8.0),
    );
    assert_eq!(z, Vec3Fix::ZERO);
}

/// Force scales linearly with the normal load (Coulomb).
#[test]
fn force_is_linear_in_normal_load() {
    let m = model();
    let v = v3(6.0, 0.0, 8.0);
    let f1 = m.friction_force(Fix128::from_int(1), Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, v);
    let f8 = m.friction_force(Fix128::from_int(8), Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, v);
    assert!(close(f8.x, 8.0 * f1.x.to_f64(), 1e-9));
    assert!(close(f8.z, 8.0 * f1.z.to_f64(), 1e-9));
}

/// Preset doc claims: tyre "high longitudinal grip", ski "very low longitudinal, high
/// transverse", ice "near-zero longitudinal"; kinetic <= static per axis for every preset.
#[test]
fn presets_have_kinetic_not_above_static_and_documented_ordering() {
    for p in [
        AnisotropicFriction::tyre_asphalt(),
        AnisotropicFriction::ski_snow(),
        AnisotropicFriction::skate_ice(),
    ] {
        assert!(p.longitudinal_kinetic <= p.longitudinal_static);
        assert!(p.transverse_kinetic <= p.transverse_static);
        assert!(p.slip_threshold_m_s > Fix128::ZERO);
    }
    let ski = AnisotropicFriction::ski_snow();
    assert!(ski.longitudinal_static.to_f64() < 0.1 && ski.transverse_static.to_f64() > 0.5);
    let ice = AnisotropicFriction::skate_ice();
    assert!(ice.longitudinal_static.to_f64() < 0.05);
    let tyre = AnisotropicFriction::tyre_asphalt();
    assert!(close(tyre.longitudinal_static, 1.1, 1e-12));
    assert!(close(tyre.transverse_kinetic, 0.7, 1e-12));
}

/// Very slow oblique slip: Fix128 has ~5.4e-20 resolution, so v_long^2 + v_trans^2 for
/// |v| ~ 1e-10 underflows to 0 and the force collapses to zero although the body is sliding.
/// Closed form (static regime): F = -N (mu_l_s*0.7071, 0, mu_t_s*0.7071).
#[test]
fn tiny_oblique_slip_still_produces_static_friction() {
    let m = model();
    let e = 1.0e-10;
    let f = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(e, 0.0, e),
    );
    let s = 0.5f64.sqrt();
    assert!(close(f.x, -10.0 * 1.0 * s, 1e-6), "Fx = {}", f.x.to_f64());
    assert!(close(f.z, -10.0 * 0.5 * s, 1e-6), "Fz = {}", f.z.to_f64());
}

/// Oblique slip at 1e10 m/s: v^2 = 1e20 exceeds the Fix128 integer range (2^63 ~ 9.2e18),
/// so the slip magnitude wraps. Documented behaviour for Fix128 overflow, but friction_force
/// advertises no input range.
#[test]
fn huge_oblique_slip_keeps_ellipse_direction() {
    let m = model();
    let e = 1.0e10;
    let f = m.friction_force(
        Fix128::from_int(10),
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Z,
        v3(e, 0.0, e),
    );
    let s = 0.5f64.sqrt();
    assert!(close(f.x, -10.0 * 0.5 * s, 1e-6), "Fx = {}", f.x.to_f64());
    assert!(close(f.z, -10.0 * 0.25 * s, 1e-6), "Fz = {}", f.z.to_f64());
}

/// Pins the three preset tables (the doc names them tyre / ski / ice but gives no numbers;
/// these are the shipped values, a change here is a behaviour change for every consumer).
#[test]
fn preset_tables_are_pinned() {
    let g = |p: AnisotropicFriction| {
        [
            p.longitudinal_static.to_f64(),
            p.longitudinal_kinetic.to_f64(),
            p.transverse_static.to_f64(),
            p.transverse_kinetic.to_f64(),
            p.slip_threshold_m_s.to_f64(),
        ]
    };
    let eq = |got: [f64; 5], want: [f64; 5]| {
        for i in 0..5 {
            assert!((got[i] - want[i]).abs() < 1e-12, "{got:?} vs {want:?}");
        }
    };
    eq(
        g(AnisotropicFriction::tyre_asphalt()),
        [1.1, 0.9, 0.9, 0.7, 0.05],
    );
    eq(
        g(AnisotropicFriction::ski_snow()),
        [0.06, 0.04, 0.9, 0.75, 0.05],
    );
    eq(
        g(AnisotropicFriction::skate_ice()),
        [0.02, 0.015, 0.85, 0.7, 0.02],
    );
}
