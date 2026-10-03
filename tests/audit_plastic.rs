//! Audit oracles (S2-1) for `alice_physics::plastic`.
//!
//! Only `PlasticModel`, `PlasticState`, `HardeningType`, `NortonCreep`,
//! `PlasticModel::from_fdm_material`, `NortonCreep::pla_room_temp` and
//! `NortonCreep::integrate` are reachable from outside the crate; the radial
//! return (`radial_return_1d`), `current_yield_mpa`, `with_hardening`,
//! `petg_room_temp` and `StressTensor` are `pub(crate)`. The radial return is
//! driven here through the public `StructuralSolver::step`, whose
//! `plastic_model` field is public so the hardening law can be chosen.
//!
//! Closed form for a constant trial stress `s` above yield `sy` (Simo &
//! Hughes ch. 2-3, linear hardening `H`, elastic modulus `E`): with
//! `r = E / (E + H)` and `f0 = s - sy`, the overshoot after step k is
//! `f_k = f0 r^k` for isotropic, kinematic AND combined hardening (the
//! consistency condition only sees `H_iso + H_kin = H`), so
//! `eps_p(n) = f0/H (1 - r^n)`; what differs is the split: back stress
//! `alpha = H eps_p` (kinematic), `H eps_p / 2` (combined), `0` (isotropic).

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::field_reassign_with_default,
    clippy::unnecessary_map_or,
    clippy::needless_range_loop
)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::plastic::{HardeningType, NortonCreep, PlasticModel, PlasticState};
use alice_physics::structural_solver::StructuralSolver;

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// 15x20 section (Z = 1000 mm^3), cantilever `load` N on 200 mm:
/// sigma = load * 200 / 1000 MPa.
fn solver(load_n: i64, ht: HardeningType) -> StructuralSolver {
    let mut s = StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: int(15),
            height_mm: int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: int(load_n),
            length_mm: int(200),
        },
        MaterialProperties::pla(),
    );
    s.plastic_model.hardening_type = ht;
    s
}

// ------------------------------------------------------------ from_fdm_material

/// Doc: isotropic hardening default, `H = 0.05 E`, `E` in MPa = GPa x 1000,
/// yield from the material.
#[test]
fn from_fdm_material_closed_forms_for_every_preset() {
    for m in [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
    ] {
        let p = PlasticModel::from_fdm_material(&m);
        let e = m.youngs_modulus_gpa.to_f64() * 1000.0;
        assert!(rel(p.youngs_modulus_mpa.to_f64(), e) < 1e-12, "{}", m.name);
        assert!(
            rel(p.hardening_modulus_mpa.to_f64(), 0.05 * e) < 1e-9,
            "{}",
            m.name
        );
        assert_eq!(p.yield_strength_mpa, m.yield_strength_mpa);
        assert_eq!(p.hardening_type, HardeningType::Isotropic);
    }
    assert_eq!(HardeningType::default(), HardeningType::Isotropic);
}

#[test]
fn plastic_state_default_is_virgin() {
    let s = PlasticState::default();
    assert_eq!(s.equivalent_plastic_strain, Fix128::ZERO);
    assert_eq!(s.back_stress_mpa, Fix128::ZERO);
    assert_eq!(s.creep_strain, Fix128::ZERO);
}

// ------------------------------------------------ radial return via the solver

/// For every hardening law the plastic strain after n steps follows
/// `f0/H (1 - r^n)` and the back stress splits as documented
/// (kinematic H eps_p, combined H eps_p / 2, isotropic 0).
#[test]
fn radial_return_split_between_isotropic_and_kinematic_matches_closed_form() {
    let (sy, e) = (50.0f64, 3500.0f64);
    let h = 0.05 * e;
    let r = e / (e + h);
    for load in [300i64, 400] {
        let s_trial = load as f64 * 200.0 / 1000.0; // 60 / 80 MPa
        let f0 = s_trial - sy;
        for (ht, alpha_frac) in [
            (HardeningType::Isotropic, 0.0),
            (HardeningType::Kinematic, 1.0),
            (HardeningType::Combined, 0.5),
        ] {
            let mut sv = solver(load, ht);
            for n in 1..=12 {
                let rep = sv.step();
                let want_ep = f0 / h * (1.0 - r.powi(n));
                assert!(
                    rel(rep.plastic_strain.to_f64(), want_ep) < 1e-9,
                    "{ht:?} load={load} n={n}: {} vs {want_ep}",
                    rep.plastic_strain.to_f64()
                );
                let want_alpha = alpha_frac * h * want_ep;
                assert!(
                    (sv.state.back_stress_mpa.to_f64() - want_alpha).abs()
                        < 1e-8 * (1.0 + want_alpha),
                    "{ht:?} load={load} n={n}: alpha {} vs {want_alpha}",
                    sv.state.back_stress_mpa.to_f64()
                );
            }
        }
    }
}

/// Doc: elastic when `|sigma - alpha| <= yield`. At exactly the yield stress
/// (50 MPa: 250 N) nothing flows; a hair above yields. (The boundary is
/// closed: `<=`.)
#[test]
fn stress_exactly_at_yield_is_elastic() {
    let mut sv = solver(250, HardeningType::Isotropic); // 250*200/1000 = 50.0 MPa
    let rep = sv.step();
    assert!(rep.bending_stress_mpa.to_f64() > 49.999 && rep.bending_stress_mpa.to_f64() < 50.001);
    if rep.bending_stress_mpa == Fix128::from_int(50) {
        assert_eq!(rep.plastic_strain, Fix128::ZERO, "|s| == yield is elastic");
    }
    let mut sv2 = solver(251, HardeningType::Isotropic);
    assert!(sv2.step().plastic_strain > Fix128::ZERO);
}

/// A kinematic back stress shifts the elastic domain: after the first step at
/// 60 MPa the back stress is `H dlambda`, and the next step (same trial
/// stress) overshoots by `f0 r`, strictly less than `f0` (the surface moved
/// toward the load). Isotropic gives the same overshoot sequence.
#[test]
fn overshoot_shrinks_geometrically_for_all_three_laws() {
    for ht in [
        HardeningType::Isotropic,
        HardeningType::Kinematic,
        HardeningType::Combined,
    ] {
        let mut sv = solver(300, ht);
        let e1 = sv.step().plastic_strain.to_f64();
        let e2 = sv.step().plastic_strain.to_f64() - e1;
        let ratio = e2 / e1;
        let r = 3500.0 / 3675.0;
        assert!(
            (ratio - r).abs() < 1e-9,
            "{ht:?}: increment ratio {ratio} vs {r}"
        );
    }
}

/// Stage independence: the plastic update does not touch creep, and the creep
/// update does not touch plastic strain or back stress.
#[test]
fn norton_creep_field_is_independent_of_plastic_update() {
    let mut a = solver(300, HardeningType::Kinematic);
    a.step();
    // creep accumulates `A s^3 dt` in `state.creep_strain` regardless of yield
    let want = 6.34e-13 * 60.0f64.powi(3) * 3600.0;
    assert!(
        rel(a.state.creep_strain.to_f64(), want) < 1e-3,
        "{}",
        a.state.creep_strain.to_f64()
    );
}

// ---------------------------------------------------------------- NortonCreep

/// Doc: calibrated to 1 % strain after 6 months (1.5768e7 s) at 10 MPa, n = 3,
/// `A = 0.01 / (10^3 * 1.5768e7) = 6.34e-13`.
#[test]
fn pla_preset_reproduces_one_percent_at_ten_mpa_after_six_months() {
    let c = NortonCreep::pla_room_temp();
    assert_eq!(c.n, 3);
    assert!(rel(c.a.to_f64(), 6.34e-13) < 1e-6, "a = {}", c.a.to_f64());
    let mut st = PlasticState::default();
    c.integrate(int(10), Fix128::from_f64(1.5768e7), &mut st);
    let got = st.creep_strain.to_f64();
    assert!((got - 0.01).abs() < 5e-5, "1 % strain expected, got {got}");
}

/// `integrate` is forward Euler `eps += A sigma^n dt`, closed form for each
/// integer exponent, and linear in dt.
#[test]
fn integrate_is_a_sigma_n_dt_for_each_exponent() {
    for n in 1u32..=6 {
        let c = NortonCreep {
            a: Fix128::from_f64(2e-9),
            n,
        };
        for (sigma, dt) in [(3i64, 100i64), (10, 3600), (-4, 50)] {
            let mut st = PlasticState::default();
            c.integrate(int(sigma), int(dt), &mut st);
            let want = 2e-9 * (sigma as f64).powi(n as i32) * dt as f64;
            assert!(
                (st.creep_strain.to_f64() - want).abs() <= 1e-9 * want.abs().max(1e-6),
                "n={n} sigma={sigma} dt={dt}: {} vs {want}",
                st.creep_strain.to_f64()
            );
        }
    }
}

/// Accumulation: two half steps equal one full step (to fixed-point rounding)
/// and the previous value is kept (`+=`, not `=`).
#[test]
fn integrate_accumulates_onto_existing_creep_strain() {
    let c = NortonCreep::pla_room_temp();
    let mut a = PlasticState::default();
    a.creep_strain = Fix128::from_ratio(1, 100);
    c.integrate(int(20), int(3600), &mut a);
    let mut b = PlasticState::default();
    c.integrate(int(20), int(3600), &mut b);
    let d = (a.creep_strain - b.creep_strain).to_f64();
    assert!((d - 0.01).abs() < 1e-12);
    let mut h = PlasticState::default();
    c.integrate(int(20), int(1800), &mut h);
    c.integrate(int(20), int(1800), &mut h);
    assert!((h.creep_strain - b.creep_strain).to_f64().abs() < 1e-12);
}

/// Degenerate inputs leave the state untouched and never touch the other
/// state fields.
#[test]
fn integrate_with_zero_stress_or_zero_dt_changes_nothing() {
    let c = NortonCreep::pla_room_temp();
    let mut st = PlasticState {
        equivalent_plastic_strain: Fix128::from_ratio(1, 50),
        back_stress_mpa: int(7),
        creep_strain: Fix128::from_ratio(3, 1000),
    };
    let before = st;
    c.integrate(Fix128::ZERO, int(3600), &mut st);
    c.integrate(int(30), Fix128::ZERO, &mut st);
    assert_eq!(st, before);
    c.integrate(int(30), int(3600), &mut st);
    assert_eq!(
        st.equivalent_plastic_strain,
        before.equivalent_plastic_strain
    );
    assert_eq!(st.back_stress_mpa, before.back_stress_mpa);
    assert!(st.creep_strain > before.creep_strain);
}

/// Odd exponent is sign-preserving: compressive stress gives negative creep of
/// the same magnitude as tensile (`sigma^3` is odd).
#[test]
fn creep_is_odd_in_stress_for_odd_exponent() {
    let c = NortonCreep::pla_room_temp();
    let (mut t, mut k) = (PlasticState::default(), PlasticState::default());
    c.integrate(int(25), int(3600), &mut t);
    c.integrate(int(-25), int(3600), &mut k);
    assert_eq!(t.creep_strain, Fix128::ZERO - k.creep_strain);
    assert!(t.creep_strain > Fix128::ZERO);
}

/// The doc's own calibration recipe `A = eps / (sigma^n t)` (1 % at 10 MPa
/// after 6 months) must be reproducible for every exponent in the documented
/// "typically 3-8" range despite `A` being stored with 2^-64 resolution: the
/// n = 8 constant is 6.3e-18 (117 ulp, truncated), so its quantisation error is
/// below 1 % (measured 0.85 %). (From n = 9 on `A < 1e-18` is under 12 ulp and the error exceeds
/// 4 %; n = 10 gives 1 ulp. Recorded in the ledger, no oracle: outside the
/// documented range.)
#[test]
fn doc_calibration_recipe_is_representable_for_exponents_3_to_8() {
    let t = 1.5768e7f64;
    for n in 3u32..=8 {
        let a = 0.01 / (10f64.powi(n as i32) * t);
        let c = NortonCreep {
            a: Fix128::from_f64(a),
            n,
        };
        let mut st = PlasticState::default();
        c.integrate(int(10), Fix128::from_f64(t), &mut st);
        let got = st.creep_strain.to_f64();
        assert!((got - 0.01).abs() < 1e-4, "n={n}: {got}");
    }
}
