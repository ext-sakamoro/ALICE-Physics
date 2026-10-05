//! Oracles for `warp_risk::{Footprint::rectangle, EnvConditions::enclosed_abs}`
//! (`examples/warp_risk_enclosure.rs`) and the score formula they feed.
//!
//! Hand-derived closed form (module doc, with the calibrated constant `k = 500`):
//! `raw = 500 * alpha * (E/5) * (A/50000) * (Lmax/300) * (dT/200)`,
//! `score = clamp(raw, 0, 1)`, category by `score < 0.25 / 0.5 / 0.75`,
//! `F/mm = E[GPa] * 1000 * alpha * dT`.
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::warp_risk::{analyze_warp_risk, EnvConditions, Footprint, WarpRiskCategory};

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}
fn raw(alpha: f64, e_gpa: f64, area: f64, lmax: f64, dt: f64) -> f64 {
    500.0 * alpha * (e_gpa / 5.0) * (area / 50_000.0) * (lmax / 300.0) * (dt / 200.0)
}

#[test]
fn rectangle_area_and_longest_side() {
    for (w, h) in [(100i64, 200i64), (200, 100), (150, 150), (0, 40), (7, 1)] {
        let f = Footprint::rectangle(fx(w), fx(h));
        assert_eq!(f.area_mm2, fx(w * h), "{w}x{h}");
        assert_eq!(f.max_dimension_mm, fx(w.max(h)), "{w}x{h}");
    }
    // fractional sides
    let f = Footprint::rectangle(Fix128::from_ratio(5, 2), Fix128::from_ratio(3, 2));
    assert_eq!(f.area_mm2, Fix128::from_ratio(15, 4));
    assert_eq!(f.max_dimension_mm, Fix128::from_ratio(5, 2));
}

#[test]
fn enclosed_abs_preset_values() {
    let e = EnvConditions::enclosed_abs();
    assert_eq!(
        (e.print_temp_c, e.chamber_temp_c, e.bed_temp_c),
        (fx(240), fx(55), fx(100))
    );
    let o = EnvConditions::open_air_pla();
    assert_eq!(
        (o.print_temp_c, o.chamber_temp_c, o.bed_temp_c),
        (fx(200), fx(20), fx(60))
    );
}

#[test]
fn score_matches_the_closed_form_on_a_grid() {
    let mats = [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
    ];
    let envs = [
        EnvConditions::open_air_pla(),
        EnvConditions::enclosed_abs(),
        EnvConditions {
            print_temp_c: fx(250),
            chamber_temp_c: fx(80),
            bed_temp_c: fx(110),
        },
    ];
    let mut unclamped = 0;
    for m in &mats {
        for env in &envs {
            for (w, h) in [(30i64, 30i64), (60, 40), (100, 100), (150, 90), (200, 120)] {
                let f = Footprint::rectangle(fx(w), fx(h));
                let r = analyze_warp_risk(&f, m, env);
                let dt = (env.print_temp_c - env.chamber_temp_c).to_f64();
                let want_raw = raw(
                    m.shrinkage_ratio.to_f64(),
                    m.youngs_modulus_gpa.to_f64(),
                    (w * h) as f64,
                    w.max(h) as f64,
                    dt,
                );
                let want = want_raw.clamp(0.0, 1.0);
                assert!(
                    (r.score.to_f64() - want).abs() < 1e-9,
                    "{} {w}x{h}: {} vs {want}",
                    m.name,
                    r.score.to_f64()
                );
                if want_raw < 1.0 {
                    unclamped += 1;
                }
                let want_cat = if want < 0.25 {
                    WarpRiskCategory::Low
                } else if want < 0.5 {
                    WarpRiskCategory::Medium
                } else if want < 0.75 {
                    WarpRiskCategory::High
                } else {
                    WarpRiskCategory::Critical
                };
                assert_eq!(r.category, want_cat, "{} {w}x{h} score {want}", m.name);
                let want_f =
                    m.youngs_modulus_gpa.to_f64() * 1000.0 * m.shrinkage_ratio.to_f64() * dt;
                assert!((r.warp_force_per_mm.to_f64() - want_f).abs() < 1e-6);
            }
        }
    }
    assert!(
        unclamped >= 20,
        "the grid must exercise the unclamped formula, got {unclamped}"
    );
}

#[test]
fn every_category_is_reachable_with_its_own_recommendation() {
    // set alpha to hit a target raw score on a fixed footprint / env
    let f = Footprint {
        area_mm2: fx(50_000),
        max_dimension_mm: fx(300),
    };
    let env = EnvConditions {
        print_temp_c: fx(220),
        chamber_temp_c: fx(20),
        bed_temp_c: fx(60),
    };
    let mut seen = Vec::new();
    for (target, cat) in [
        (0.10, WarpRiskCategory::Low),
        (0.24, WarpRiskCategory::Low),
        (0.26, WarpRiskCategory::Medium),
        (0.49, WarpRiskCategory::Medium),
        (0.51, WarpRiskCategory::High),
        (0.74, WarpRiskCategory::High),
        (0.76, WarpRiskCategory::Critical),
        (5.0, WarpRiskCategory::Critical),
    ] {
        // raw = 500 * alpha * (E/5 = 1) * 1 * 1 * 1 -> alpha = target / 500
        let m = MaterialProperties {
            shrinkage_ratio: Fix128::from_f64(target / 500.0),
            youngs_modulus_gpa: fx(5),
            ..MaterialProperties::pla()
        };
        let r = analyze_warp_risk(&f, &m, &env);
        let dt_norm = 200.0 / 200.0;
        assert!(
            (r.score.to_f64() - (target * dt_norm).min(1.0)).abs() < 1e-9,
            "target {target}"
        );
        assert_eq!(r.category, cat, "target {target}");
        seen.push((r.category, r.recommendation));
    }
    let mut texts: Vec<_> = seen.iter().map(|s| s.1).collect();
    texts.sort_unstable();
    texts.dedup();
    assert_eq!(texts.len(), 4, "one distinct recommendation per category");
    let text_of = |c: WarpRiskCategory| seen.iter().find(|s| s.0 == c).unwrap().1;
    assert!(text_of(WarpRiskCategory::Low).contains("No mitigation"));
    assert!(text_of(WarpRiskCategory::Medium).contains("brim"));
    assert!(text_of(WarpRiskCategory::High).contains("enclosure"));
    assert!(text_of(WarpRiskCategory::Critical).contains("failure"));
}

#[test]
fn documented_case_is_critical_and_the_enclosure_helps() {
    // module doc: 280 x 250 mm PLA plate on the open printer -> Critical, score > 0.75
    let plate = Footprint::rectangle(fx(280), fx(250));
    let r = analyze_warp_risk(
        &plate,
        &MaterialProperties::pla(),
        &EnvConditions::open_air_pla(),
    );
    assert_eq!(r.category, WarpRiskCategory::Critical);
    assert!(r.score > Fix128::from_ratio(75, 100));
    // hand value: 500 * 0.002 * 0.7 * 1.4 * (280/300) * 0.9 = 0.8232
    assert!(
        (r.score.to_f64() - 0.8232).abs() < 1e-9,
        "{}",
        r.score.to_f64()
    );
    // the same ABS part scores strictly lower in the 55 C enclosure than in open air
    let part = Footprint::rectangle(fx(100), fx(100));
    let abs = MaterialProperties::abs();
    let open = EnvConditions {
        print_temp_c: fx(240),
        chamber_temp_c: fx(20),
        bed_temp_c: fx(90),
    };
    let enclosed = analyze_warp_risk(&part, &abs, &EnvConditions::enclosed_abs());
    let opened = analyze_warp_risk(&part, &abs, &open);
    assert!(enclosed.score < opened.score);
    assert!(enclosed.warp_force_per_mm < opened.warp_force_per_mm);
}

#[test]
fn degenerate_inputs_clamp_to_zero_risk() {
    let pla = MaterialProperties::pla();
    // empty footprint
    let z = Footprint {
        area_mm2: Fix128::ZERO,
        max_dimension_mm: Fix128::ZERO,
    };
    let r = analyze_warp_risk(&z, &pla, &EnvConditions::open_air_pla());
    assert_eq!((r.score, r.category), (Fix128::ZERO, WarpRiskCategory::Low));
    // chamber hotter than the nozzle: negative dT -> negative raw -> clamped to 0
    let hot = EnvConditions {
        print_temp_c: fx(200),
        chamber_temp_c: fx(260),
        bed_temp_c: fx(60),
    };
    let big = Footprint::rectangle(fx(280), fx(250));
    let r = analyze_warp_risk(&big, &pla, &hot);
    assert_eq!((r.score, r.category), (Fix128::ZERO, WarpRiskCategory::Low));
    // zero shrinkage material never warps
    let rigid = MaterialProperties {
        shrinkage_ratio: Fix128::ZERO,
        ..pla
    };
    let r = analyze_warp_risk(&big, &rigid, &EnvConditions::open_air_pla());
    assert_eq!(r.score, Fix128::ZERO);
    assert_eq!(r.warp_force_per_mm, Fix128::ZERO);
}
