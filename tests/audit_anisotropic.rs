//! Audit oracle for `anisotropic`: elasticity transformation (Jones 2.85), strength
//! constructors, Max-stress / Hill / Tsai-Wu failure indices. Expected values from textbook
//! closed forms and independent f64 evaluation.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::anisotropic::{
    evaluate_failure, AnisotropicStrength, FailureCriterion, OrthotropicElasticity,
    OrthotropicStress,
};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn mats() -> Vec<MaterialProperties> {
    vec![
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::cf_nylon(),
        MaterialProperties::tpu(),
    ]
}

fn ortho(
    el: f64,
    et: f64,
    ez: f64,
    nlt: f64,
    nlz: f64,
    glt: f64,
    glz: f64,
) -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: fx(el),
        e_t_mpa: fx(et),
        e_z_mpa: fx(ez),
        nu_lt: fx(nlt),
        nu_lz: fx(nlz),
        nu_tz: fx(0.3),
        g_lt_mpa: fx(glt),
        g_lz_mpa: fx(glz),
        g_tz_mpa: fx(glz),
    }
}

// ---------------- elasticity ----------------

#[test]
fn from_fdm_material_constants_follow_the_documented_rules() {
    for m in mats() {
        let o = OrthotropicElasticity::from_fdm_material(&m);
        let e = m.youngs_modulus_gpa.to_f64() * 1000.0;
        let r = m.anisotropy_z_ratio.to_f64();
        assert!((o.e_l_mpa.to_f64() - e).abs() < 1e-9 * e, "{}", m.name);
        assert_eq!(o.e_l_mpa, o.e_t_mpa);
        assert!((o.e_z_mpa.to_f64() - e * r).abs() < 1e-9 * e, "{}", m.name);
        assert!((o.nu_lt.to_f64() - 0.35).abs() < 1e-18);
        assert!((o.nu_lz.to_f64() - 0.30).abs() < 1e-18);
        assert_eq!(o.nu_tz, o.nu_lz);
        let g = e / (2.0 * (1.0 + 0.35));
        assert!((o.g_lt_mpa.to_f64() - g).abs() < 1e-9 * g, "{}", m.name);
        assert!((o.g_lz_mpa.to_f64() - g * r).abs() < 1e-9 * g, "{}", m.name);
        assert_eq!(o.g_tz_mpa, o.g_lz_mpa);
    }
}

fn jones(el: f64, et: f64, g: f64, nu: f64, th: f64) -> f64 {
    let (s, c) = (th.sin(), th.cos());
    1.0 / (c.powi(4) / el + s.powi(4) / et + c * c * s * s * (1.0 / g - 2.0 * nu / el))
}

#[test]
fn e_at_angle_lt_matches_jones_2_85_for_a_strongly_orthotropic_ply() {
    // carbon/epoxy-like ply: E_L 140 GPa, E_T 10 GPa, G 5 GPa, nu 0.3
    let o = ortho(140e3, 10e3, 10e3, 0.3, 0.3, 5e3, 5e3);
    for deg in [
        0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, -30.0, 180.0,
    ] {
        let th = deg * std::f64::consts::PI / 180.0;
        let got = o.e_at_angle_lt(fx(th)).to_f64();
        let want = jones(140e3, 10e3, 5e3, 0.3, th);
        assert!(
            (got - want).abs() < 1e-6 * want,
            "deg={deg} got={got} want={want}"
        );
    }
}

#[test]
fn e_at_angle_lt_endpoints_symmetry_and_monotonic_fall() {
    let o = ortho(140e3, 10e3, 10e3, 0.3, 0.3, 5e3, 5e3);
    let pi = std::f64::consts::PI;
    assert!((o.e_at_angle_lt(Fix128::ZERO).to_f64() - 140e3).abs() < 1e-6 * 140e3);
    assert!((o.e_at_angle_lt(fx(pi / 2.0)).to_f64() - 10e3).abs() < 1e-6 * 10e3);
    for th in [0.2, 0.7, 1.1] {
        let a = o.e_at_angle_lt(fx(th)).to_f64();
        let b = o.e_at_angle_lt(fx(-th)).to_f64();
        let c = o.e_at_angle_lt(fx(pi - th)).to_f64();
        assert!(
            (a - b).abs() < 1e-6 * a && (a - c).abs() < 1e-6 * a,
            "th={th}"
        );
    }
    let mut prev = f64::MAX;
    for i in 0..=9 {
        let e = o.e_at_angle_lt(fx(i as f64 * pi / 18.0)).to_f64();
        assert!(e < prev);
        prev = e;
    }
}

/// FDM base: E_L = E_T and G_LT = E / 2(1+nu_LT) => the in-plane material is isotropic, so E(theta) = E.
#[test]
fn fdm_in_plane_is_isotropic_for_every_angle() {
    for m in mats() {
        let o = OrthotropicElasticity::from_fdm_material(&m);
        let e = o.e_l_mpa.to_f64();
        for th in [
            0.0,
            0.3,
            std::f64::consts::FRAC_PI_4,
            1.2,
            std::f64::consts::FRAC_PI_2,
        ] {
            let got = o.e_at_angle_lt(fx(th)).to_f64();
            assert!(
                (got - e).abs() < 1e-6 * e,
                "{} th={th} got={got} e={e}",
                m.name
            );
        }
    }
}

#[test]
fn e_at_angle_lz_uses_the_lz_constants() {
    for m in mats() {
        let o = OrthotropicElasticity::from_fdm_material(&m);
        let (el, ez, g, nu) = (
            o.e_l_mpa.to_f64(),
            o.e_z_mpa.to_f64(),
            o.g_lz_mpa.to_f64(),
            o.nu_lz.to_f64(),
        );
        for th in [0.0, 0.4, 0.9, 1.3, std::f64::consts::FRAC_PI_2] {
            let got = o.e_at_angle_lz(fx(th)).to_f64();
            let want = jones(el, ez, g, nu, th);
            assert!(
                (got - want).abs() < 1e-6 * want,
                "{} th={th} got={got} want={want}",
                m.name
            );
        }
        assert!((o.e_at_angle_lz(Fix128::ZERO).to_f64() - el).abs() < 1e-6 * el);
        assert!((o.e_at_angle_lz(fx(std::f64::consts::FRAC_PI_2)).to_f64() - ez).abs() < 1e-6 * ez);
    }
}

/// L-T and L-Z planes use different constants: change only T-side data, L-Z is unaffected and vice versa.
#[test]
fn lt_and_lz_planes_are_independent() {
    let a = ortho(100e3, 10e3, 20e3, 0.3, 0.25, 4e3, 6e3);
    let th = fx(0.6);
    let mut b = a;
    b.e_z_mpa = fx(30e3);
    b.g_lz_mpa = fx(9e3);
    b.nu_lz = fx(0.2);
    assert_eq!(a.e_at_angle_lt(th), b.e_at_angle_lt(th));
    assert_ne!(a.e_at_angle_lz(th), b.e_at_angle_lz(th));
    let mut c = a;
    c.e_t_mpa = fx(15e3);
    c.g_lt_mpa = fx(7e3);
    c.nu_lt = fx(0.2);
    assert_eq!(a.e_at_angle_lz(th), c.e_at_angle_lz(th));
    assert_ne!(a.e_at_angle_lt(th), c.e_at_angle_lt(th));
}

// ---------------- strength ----------------

#[test]
fn strength_from_fdm_follows_the_documented_ratios() {
    for m in mats() {
        let s = AnisotropicStrength::from_fdm_material(&m);
        let x = m.yield_strength_mpa.to_f64();
        let r = m.anisotropy_z_ratio.to_f64();
        for v in [
            s.x_l_tension_mpa,
            s.x_l_compression_mpa,
            s.x_t_tension_mpa,
            s.x_t_compression_mpa,
        ] {
            assert_eq!(v, m.yield_strength_mpa);
        }
        assert!(
            (s.x_z_tension_mpa.to_f64() - x * r).abs() < 1e-9 * x,
            "{}",
            m.name
        );
        assert_eq!(s.x_z_tension_mpa, s.x_z_compression_mpa);
        assert!((s.s_lt_mpa.to_f64() - 0.6 * x).abs() < 1e-9 * x);
        assert!((s.s_lz_mpa.to_f64() - 0.6 * x * r).abs() < 1e-9 * x);
        assert_eq!(s.s_tz_mpa, s.s_lz_mpa);
    }
}

#[test]
fn axial_constructor_sets_normals_and_zero_shears() {
    let s = OrthotropicStress::axial(fx(1.0), fx(-2.0), fx(3.0));
    assert_eq!(
        (s.sigma_l, s.sigma_t, s.sigma_z),
        (fx(1.0), fx(-2.0), fx(3.0))
    );
    assert_eq!(
        (s.tau_lt, s.tau_lz, s.tau_tz),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
}

// ---------------- failure criteria ----------------

fn composite() -> AnisotropicStrength {
    // asymmetric, strongly anisotropic strengths (graphite/epoxy-like)
    AnisotropicStrength {
        x_l_tension_mpa: fx(1500.0),
        x_l_compression_mpa: fx(1200.0),
        x_t_tension_mpa: fx(40.0),
        x_t_compression_mpa: fx(200.0),
        x_z_tension_mpa: fx(30.0),
        x_z_compression_mpa: fx(150.0),
        s_lt_mpa: fx(68.0),
        s_lz_mpa: fx(60.0),
        s_tz_mpa: fx(45.0),
    }
}
fn st(l: f64, t: f64, z: f64, lt: f64, lz: f64, tz: f64) -> OrthotropicStress {
    OrthotropicStress {
        sigma_l: fx(l),
        sigma_t: fx(t),
        sigma_z: fx(z),
        tau_lt: fx(lt),
        tau_lz: fx(lz),
        tau_tz: fx(tz),
    }
}
fn idx(s: &OrthotropicStress, a: &AnisotropicStrength, c: FailureCriterion) -> f64 {
    evaluate_failure(s, a, c).failure_index.to_f64()
}

/// Every criterion reports exactly 1 at each single-component allowable of the matching sign.
#[test]
fn index_is_one_at_each_uniaxial_and_shear_allowable() {
    let a = composite();
    let tension = [(1500.0, 0), (40.0, 1), (30.0, 2)];
    let compression = [(1200.0, 0), (200.0, 1), (150.0, 2)];
    let crit_tension_all = [FailureCriterion::MaximumStress, FailureCriterion::TsaiWu];
    for c in crit_tension_all {
        for (x, ax) in tension {
            let mut v = [0.0; 3];
            v[ax] = x;
            let got = idx(&st(v[0], v[1], v[2], 0.0, 0.0, 0.0), &a, c);
            assert!(
                (got - 1.0).abs() < 1e-9,
                "{c:?} tension axis {ax} idx {got}"
            );
        }
        for (x, ax) in compression {
            let mut v = [0.0; 3];
            v[ax] = -x;
            let got = idx(&st(v[0], v[1], v[2], 0.0, 0.0, 0.0), &a, c);
            assert!(
                (got - 1.0).abs() < 1e-9,
                "{c:?} compression axis {ax} idx {got}"
            );
        }
    }
    for c in [
        FailureCriterion::MaximumStress,
        FailureCriterion::Hill,
        FailureCriterion::TsaiWu,
    ] {
        for (shear, which) in [(68.0, 0), (60.0, 1), (45.0, 2)] {
            for sign in [1.0, -1.0] {
                let mut t = [0.0; 3];
                t[which] = sign * shear;
                let got = idx(&st(0.0, 0.0, 0.0, t[0], t[1], t[2]), &a, c);
                assert!(
                    (got - 1.0).abs() < 1e-9,
                    "{c:?} shear {which} sign {sign}: {got}"
                );
            }
        }
    }
    // Hill uses the tensile strengths on each axis
    for (x, ax) in tension {
        let mut v = [0.0; 3];
        v[ax] = x;
        let got = idx(
            &st(v[0], v[1], v[2], 0.0, 0.0, 0.0),
            &a,
            FailureCriterion::Hill,
        );
        assert!((got - 1.0).abs() < 1e-9, "Hill tension axis {ax}: {got}");
    }
}

/// Isotropic strengths, no shear: Tsai-Wu (Hoffman) and Hill both reduce to (von Mises / X)^2.
#[test]
fn isotropic_normal_states_reduce_to_von_mises() {
    let x = 70.0;
    let a = AnisotropicStrength {
        x_l_tension_mpa: fx(x),
        x_l_compression_mpa: fx(x),
        x_t_tension_mpa: fx(x),
        x_t_compression_mpa: fx(x),
        x_z_tension_mpa: fx(x),
        x_z_compression_mpa: fx(x),
        s_lt_mpa: fx(40.0),
        s_lz_mpa: fx(40.0),
        s_tz_mpa: fx(40.0),
    };
    for (l, t, z) in [
        (30.0, -10.0, 5.0),
        (70.0, 70.0, 0.0),
        (12.0, 12.0, 12.0),
        (-20.0, 45.0, 33.0),
    ] {
        let vm2 = ((l - t) * (l - t) + (t - z) * (t - z) + (z - l) * (z - l)) / 2.0;
        let want = vm2 / (x * x);
        let s = st(l, t, z, 0.0, 0.0, 0.0);
        for c in [FailureCriterion::Hill, FailureCriterion::TsaiWu] {
            let got = idx(&s, &a, c);
            assert!(
                (got - want).abs() < 1e-9,
                "{c:?} ({l},{t},{z}) got {got} want {want}"
            );
        }
    }
}

#[test]
fn max_stress_uses_the_signed_allowable_and_reports_the_worst_ratio() {
    let a = composite();
    // sigma_t = -100 against X_t compression 200 -> 0.5 ; tau_tz = 30 / 45 = 0.667 is the worst
    let r = evaluate_failure(
        &st(0.0, -100.0, 0.0, 0.0, 0.0, 30.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert!((r.failure_index.to_f64() - 30.0 / 45.0).abs() < 1e-12);
    // switching the sign puts it against the 40 MPa tension allowable -> 2.5
    let r = evaluate_failure(
        &st(0.0, 100.0, 0.0, 0.0, 0.0, 30.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert!((r.failure_index.to_f64() - 2.5).abs() < 1e-12);
    assert!(!r.is_safe);
}

#[test]
fn is_safe_is_strictly_below_one() {
    let a = composite();
    let at = evaluate_failure(
        &st(1500.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert_eq!(at.failure_index, Fix128::ONE);
    assert!(!at.is_safe, "index = 1 is incipient failure, not safe");
    let below = evaluate_failure(
        &st(1499.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert!(below.is_safe);
}

/// Doc: reserve factor is "how much the loading could scale before failure": scaling the stress state
/// by it must land on failure index 1. Exact for the (linear) maximum-stress criterion.
#[test]
fn reserve_factor_scales_max_stress_to_incipient_failure() {
    let a = composite();
    let s = st(300.0, -20.0, 5.0, 10.0, 0.0, 0.0);
    let r = evaluate_failure(&s, &a, FailureCriterion::MaximumStress);
    let k = r.reserve_factor.to_f64();
    let scaled = st(300.0 * k, -20.0 * k, 5.0 * k, 10.0 * k, 0.0, 0.0);
    assert!((idx(&scaled, &a, FailureCriterion::MaximumStress) - 1.0).abs() < 1e-9);
}

/// Hill is homogeneous of degree 2 in the stresses: a load factor R gives index R^2 f, so the
/// factor that reaches failure is 1/sqrt(f), not 1/f.
#[test]
fn reserve_factor_scales_hill_to_incipient_failure() {
    let a = composite();
    let s = st(0.0, 20.0, 0.0, 0.0, 0.0, 0.0); // f = 0.25
    let r = evaluate_failure(&s, &a, FailureCriterion::Hill);
    let k = r.reserve_factor.to_f64();
    let scaled = st(0.0, 20.0 * k, 0.0, 0.0, 0.0, 0.0);
    let got = idx(&scaled, &a, FailureCriterion::Hill);
    assert!((got - 1.0).abs() < 1e-6, "reserve {k}: scaled index {got}");
}

/// Tsai-Wu with linear terms: R solves a R^2 + b R = 1 (not R = 1/(a+b)).
#[test]
fn reserve_factor_scales_tsai_wu_to_incipient_failure() {
    let a = composite();
    let s = st(0.0, 20.0, 0.0, 0.0, 0.0, 0.0);
    let r = evaluate_failure(&s, &a, FailureCriterion::TsaiWu);
    let k = r.reserve_factor.to_f64();
    let scaled = st(0.0, 20.0 * k, 0.0, 0.0, 0.0, 0.0);
    let got = idx(&scaled, &a, FailureCriterion::TsaiWu);
    assert!((got - 1.0).abs() < 1e-6, "reserve {k}: scaled index {got}");
}

/// Tsai-Wu with a negative linear part (compression on an axis whose compressive
/// strength exceeds the tensile one). X_Lt = 4, X_Lc = 16, sigma_L = -8:
/// b = (1/4 - 1/16)(-8) = -3/2, a = 64/64 = 1, so R^2 - (3/2) R - 1 = 0 gives
/// R = 2 and the scaled state sigma_L = -16 = -X_Lc sits exactly on the envelope.
#[test]
fn reserve_factor_tsai_wu_with_negative_linear_part_is_exact() {
    let s = AnisotropicStrength {
        x_l_tension_mpa: Fix128::from_int(4),
        x_l_compression_mpa: Fix128::from_int(16),
        x_t_tension_mpa: Fix128::from_int(8),
        x_t_compression_mpa: Fix128::from_int(32),
        x_z_tension_mpa: Fix128::from_int(16),
        x_z_compression_mpa: Fix128::from_int(64),
        s_lt_mpa: Fix128::from_int(8),
        s_lz_mpa: Fix128::from_int(16),
        s_tz_mpa: Fix128::from_int(32),
    };
    let r = evaluate_failure(
        &OrthotropicStress::axial(Fix128::from_int(-8), Fix128::ZERO, Fix128::ZERO),
        &s,
        FailureCriterion::TsaiWu,
    );
    assert_eq!(r.failure_index, Fix128::from_ratio(-1, 2));
    assert_eq!(r.reserve_factor, Fix128::from_int(2));
    let at = evaluate_failure(
        &OrthotropicStress::axial(Fix128::from_int(-16), Fix128::ZERO, Fix128::ZERO),
        &s,
        FailureCriterion::TsaiWu,
    );
    assert_eq!(at.failure_index, Fix128::ONE);
}

/// A zero allowable means "no strength in that direction": any stress there must fail.
#[test]
// AUD-A-S1W6-005
fn zero_strength_with_nonzero_stress_is_not_safe() {
    let mut a = composite();
    a.x_z_tension_mpa = Fix128::ZERO;
    a.x_z_compression_mpa = Fix128::ZERO;
    for c in [
        FailureCriterion::MaximumStress,
        FailureCriterion::Hill,
        FailureCriterion::TsaiWu,
    ] {
        let r = evaluate_failure(&st(0.0, 0.0, 10.0, 0.0, 0.0, 0.0), &a, c);
        assert!(!r.is_safe, "{c:?}: index {}", r.failure_index.to_f64());
        assert_eq!(r.failure_index, Fix128::from_int(i64::MAX >> 8), "{c:?}");
        // no stress in that direction: the zero allowable does not matter
        let r = evaluate_failure(&st(100.0, 0.0, 0.0, 0.0, 0.0, 0.0), &a, c);
        let base = evaluate_failure(&st(100.0, 0.0, 0.0, 0.0, 0.0, 0.0), &composite(), c);
        assert_eq!(r.is_safe, base.is_safe, "{c:?}");
    }
    // a zero shear allowable with shear stress, and a zero compression
    // allowable under tension (only the loaded direction counts)
    let mut b = composite();
    b.s_lt_mpa = Fix128::ZERO;
    assert!(
        !evaluate_failure(
            &st(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            &b,
            FailureCriterion::Hill
        )
        .is_safe
    );
    let mut c = composite();
    c.x_l_compression_mpa = Fix128::ZERO;
    let t = evaluate_failure(
        &st(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        &c,
        FailureCriterion::MaximumStress,
    );
    assert!(t.is_safe, "tension against a zero compression allowable");
}

#[test]
fn reserve_factor_is_one_over_index_floored_at_epsilon() {
    let a = composite();
    let r = evaluate_failure(
        &st(750.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert!((r.reserve_factor.to_f64() - 2.0).abs() < 1e-9);
    let z = evaluate_failure(
        &st(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        &a,
        FailureCriterion::MaximumStress,
    );
    assert!((z.reserve_factor.to_f64() - 1.0e6).abs() < 1.0);
    assert!(z.is_safe);
}

/// Hill/Tsai-Wu are symmetric under swapping the roles of (T,Z) when their strengths are swapped.
#[test]
fn hill_is_invariant_under_relabelling_t_and_z() {
    let a = composite();
    let b = AnisotropicStrength {
        x_t_tension_mpa: a.x_z_tension_mpa,
        x_z_tension_mpa: a.x_t_tension_mpa,
        x_t_compression_mpa: a.x_z_compression_mpa,
        x_z_compression_mpa: a.x_t_compression_mpa,
        s_lz_mpa: a.s_lt_mpa,
        s_lt_mpa: a.s_lz_mpa,
        ..a
    };
    // swapping T<->Z swaps tau_lt<->tau_lz, tau_tz stays
    let s1 = st(100.0, 12.0, -7.0, 20.0, 15.0, 9.0);
    let s2 = st(100.0, -7.0, 12.0, 15.0, 20.0, 9.0);
    for c in [
        FailureCriterion::Hill,
        FailureCriterion::TsaiWu,
        FailureCriterion::MaximumStress,
    ] {
        let (x, y) = (idx(&s1, &a, c), idx(&s2, &b, c));
        assert!((x - y).abs() < 1e-9 * x.abs().max(1.0), "{c:?} {x} vs {y}");
    }
}
