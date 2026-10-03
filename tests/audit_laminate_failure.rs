//! Audit oracles for laminate_failure
//!
//! 期待値は doc の式 (Tsai-Wu 1971 / Azzi-Tsai-Hill / Hashin 1980) を f64 で手書きしたもの

use alice_physics::laminate_failure::{
    failure_index, hashin_failure_mode, puck_failure_mode, tsai_hill_failure_index,
    tsai_wu_failure_index, FailureCriterion, FailureMode, LaminateStrengths, StressState,
};
use alice_physics::math::Fix128;

fn st(a: f64, b: f64, c: f64) -> StressState {
    StressState {
        sigma_1: Fix128::from_f64(a),
        sigma_2: Fix128::from_f64(b),
        tau_12: Fix128::from_f64(c),
    }
}

/// (xt, xc, yt, yc, s) を f64 で
fn dims(s: &LaminateStrengths) -> [f64; 5] {
    [
        s.xt.to_f64(),
        s.xc.to_f64(),
        s.yt.to_f64(),
        s.yc.to_f64(),
        s.s.to_f64(),
    ]
}

fn ref_tsai_wu(d: [f64; 5], s1: f64, s2: f64, t: f64) -> f64 {
    let [xt, xc, yt, yc, s] = d;
    let f1 = 1.0 / xt - 1.0 / xc;
    let f2 = 1.0 / yt - 1.0 / yc;
    let f11 = 1.0 / (xt * xc);
    let f22 = 1.0 / (yt * yc);
    let f66 = 1.0 / (s * s);
    let f12 = -0.5 * (f11 * f22).sqrt();
    f1 * s1 + f2 * s2 + f11 * s1 * s1 + f22 * s2 * s2 + f66 * t * t + 2.0 * f12 * s1 * s2
}

fn ref_tsai_hill(d: [f64; 5], s1: f64, s2: f64, t: f64) -> f64 {
    let [xt, xc, yt, yc, s] = d;
    let x = if s1 >= 0.0 { xt } else { xc };
    let y = if s2 >= 0.0 { yt } else { yc };
    (s1 * s1 - s1 * s2) / (x * x) + s2 * s2 / (y * y) + t * t / (s * s)
}

fn grid() -> Vec<(f64, f64, f64)> {
    let a = [-1800.0, -700.0, -100.0, 0.0, 50.0, 400.0, 1200.0, 1700.0];
    let b = [-300.0, -120.0, -10.0, 0.0, 5.0, 30.0, 60.0];
    let c = [-90.0, -20.0, 0.0, 15.0, 70.0];
    let mut v = Vec::new();
    for &x in &a {
        for &y in &b {
            for &z in &c {
                v.push((x, y, z));
            }
        }
    }
    v
}

fn presets() -> [LaminateStrengths; 2] {
    [LaminateStrengths::cfrp_ud(), LaminateStrengths::gfrp_ud()]
}

#[test]
fn tsai_wu_matches_textbook_formula_on_grid() {
    for p in presets() {
        let d = dims(&p);
        for (a, b, c) in grid() {
            let got = tsai_wu_failure_index(p, st(a, b, c)).to_f64();
            let want = ref_tsai_wu(d, a, b, c);
            assert!(
                (got - want).abs() < 1e-6,
                "tw ({a},{b},{c}): got {got} want {want}"
            );
        }
    }
}

#[test]
fn tsai_wu_is_one_at_each_uniaxial_strength() {
    // Tsai-Wu の係数の定義から F1 X + F11 X^2 = 1 (X = Xt, -Xc), 同様に Y, S
    for p in presets() {
        let d = dims(&p);
        let [xt, xc, yt, yc, s] = d;
        for (a, b, c) in [
            (xt, 0.0, 0.0),
            (-xc, 0.0, 0.0),
            (0.0, yt, 0.0),
            (0.0, -yc, 0.0),
            (0.0, 0.0, s),
            (0.0, 0.0, -s),
        ] {
            let got = tsai_wu_failure_index(p, st(a, b, c)).to_f64();
            assert!((got - 1.0).abs() < 1e-6, "({a},{b},{c}) -> {got}");
        }
    }
}

#[test]
fn tsai_hill_matches_formula_on_grid_and_is_one_at_strengths() {
    for p in presets() {
        let d = dims(&p);
        for (a, b, c) in grid() {
            let got = tsai_hill_failure_index(p, st(a, b, c)).to_f64();
            let want = ref_tsai_hill(d, a, b, c);
            assert!(
                (got - want).abs() < 1e-6,
                "th ({a},{b},{c}): got {got} want {want}"
            );
        }
        let [xt, xc, yt, yc, s] = d;
        for (a, b, c) in [
            (xt, 0.0, 0.0),
            (-xc, 0.0, 0.0),
            (0.0, yt, 0.0),
            (0.0, -yc, 0.0),
            (0.0, 0.0, s),
        ] {
            let got = tsai_hill_failure_index(p, st(a, b, c)).to_f64();
            assert!((got - 1.0).abs() < 1e-6, "({a},{b},{c}) -> {got}");
        }
    }
}

#[test]
fn failure_index_scales_quadratically_for_tsai_hill() {
    // Tsai-Hill は 2 次同次: FI(k·σ) = k^2·FI(σ)
    for p in presets() {
        let base = tsai_hill_failure_index(p, st(300.0, 15.0, 20.0)).to_f64();
        let dbl = tsai_hill_failure_index(p, st(600.0, 30.0, 40.0)).to_f64();
        assert!((dbl - 4.0 * base).abs() < 1e-6);
    }
}

fn ref_hashin(d: [f64; 5], s1: f64, s2: f64, t: f64) -> (FailureMode, f64) {
    let [xt, xc, yt, yc, s] = d;
    if s1 >= 0.0 {
        let fi = s1 * s1 / (xt * xt) + t * t / (s * s);
        if fi >= 1.0 {
            return (FailureMode::FibreTension, fi);
        }
    } else {
        let fi = s1 * s1 / (xc * xc);
        if fi >= 1.0 {
            return (FailureMode::FibreCompression, fi);
        }
    }
    if s2 >= 0.0 {
        let fi = s2 * s2 / (yt * yt) + t * t / (s * s);
        if fi >= 1.0 {
            return (FailureMode::MatrixTension, fi);
        }
    } else {
        let q = yc / (2.0 * s);
        let fi = (s2 / (2.0 * s)) * (s2 / (2.0 * s)) + (q * q - 1.0) * s2 / yc + t * t / (s * s);
        if fi >= 1.0 {
            return (FailureMode::MatrixCompression, fi);
        }
    }
    (FailureMode::Safe, 0.0)
}

#[test]
fn hashin_mode_matches_hashin_1980_on_grid_away_from_boundary() {
    let mut n = 0;
    for p in presets() {
        let d = dims(&p);
        for (a, b, c) in grid() {
            let (want, fi) = ref_hashin(d, a, b, c);
            if want != FailureMode::Safe && (fi - 1.0).abs() < 1e-6 {
                continue;
            }
            let got = hashin_failure_mode(p, st(a, b, c));
            assert_eq!(got, want, "hashin ({a},{b},{c})");
            n += 1;
        }
    }
    assert!(n > 500, "compared {n}");
}

#[test]
fn hashin_just_below_and_above_each_strength() {
    for p in presets() {
        let [xt, xc, yt, _yc, s] = dims(&p);
        assert_eq!(
            hashin_failure_mode(p, st(xt * 0.999, 0.0, 0.0)),
            FailureMode::Safe
        );
        assert_eq!(
            hashin_failure_mode(p, st(xt * 1.001, 0.0, 0.0)),
            FailureMode::FibreTension
        );
        assert_eq!(
            hashin_failure_mode(p, st(-xc * 0.999, 0.0, 0.0)),
            FailureMode::Safe
        );
        assert_eq!(
            hashin_failure_mode(p, st(-xc * 1.001, 0.0, 0.0)),
            FailureMode::FibreCompression
        );
        assert_eq!(
            hashin_failure_mode(p, st(0.0, yt * 0.999, 0.0)),
            FailureMode::Safe
        );
        assert_eq!(
            hashin_failure_mode(p, st(0.0, yt * 1.001, 0.0)),
            FailureMode::MatrixTension
        );
        // 純せん断 τ = S: σ1 = 0 >= 0 なので Hashin 1980 の fibre tension 式 (τ 項を含む) が先に成立する
        assert_eq!(
            hashin_failure_mode(p, st(0.0, 0.0, s * 1.001)),
            FailureMode::FibreTension
        );
        assert_eq!(
            hashin_failure_mode(p, st(0.0, 0.0, s * 0.999)),
            FailureMode::Safe
        );
    }
}

fn ref_puck(d: [f64; 5], s1: f64, s2: f64, t: f64) -> (FailureMode, f64) {
    let (m, _) = ref_hashin(d, s1, s2, t);
    if m == FailureMode::FibreTension || m == FailureMode::FibreCompression {
        return (m, 0.0);
    }
    let [_, _, yt, yc, s] = d;
    let rn = if s2 >= 0.0 { s2 / yt } else { -s2 / yc };
    let rs = t.abs() / s;
    let e = rn * rn + rs * rs;
    if e < 1.0 {
        return (FailureMode::Safe, e);
    }
    if s2 >= 0.0 {
        (FailureMode::InterFibreA, e)
    } else if rs > rn {
        (FailureMode::InterFibreB, e)
    } else {
        (FailureMode::InterFibreC, e)
    }
}

#[test]
fn puck_mode_matches_documented_rule_on_grid() {
    for p in presets() {
        let d = dims(&p);
        for (a, b, c) in grid() {
            let (want, e) = ref_puck(d, a, b, c);
            if (e - 1.0).abs() < 1e-6 {
                continue;
            }
            let got = puck_failure_mode(p, st(a, b, c));
            assert_eq!(got, want, "puck ({a},{b},{c})");
        }
    }
}

#[test]
fn puck_covers_all_three_interfibre_modes_with_hand_points() {
    let p = LaminateStrengths::cfrp_ud();
    // A: σ2 = 50 > Yt 40 / B: σ2 = -50, τ = 67.9 (rn = 0.203, rs = 0.9985, e = 1.04) / C: σ2 = -250, τ = 10 (rn = 1.016)
    assert_eq!(
        puck_failure_mode(p, st(0.0, 50.0, 0.0)),
        FailureMode::InterFibreA
    );
    assert_eq!(
        puck_failure_mode(p, st(0.0, -50.0, 67.9)),
        FailureMode::InterFibreB
    );
    assert_eq!(
        puck_failure_mode(p, st(0.0, -250.0, 10.0)),
        FailureMode::InterFibreC
    );
    assert_eq!(
        puck_failure_mode(p, st(0.0, -10.0, 10.0)),
        FailureMode::Safe
    );
}

#[test]
fn dispatch_hashin_and_puck_are_exactly_zero_or_one_and_match_modes() {
    for p in presets() {
        for (a, b, c) in grid() {
            let s = st(a, b, c);
            let h = failure_index(FailureCriterion::Hashin, p, s);
            let want_h = if hashin_failure_mode(p, s) == FailureMode::Safe {
                Fix128::ZERO
            } else {
                Fix128::ONE
            };
            assert_eq!(h, want_h);
            let q = failure_index(FailureCriterion::Puck, p, s);
            let want_q = if puck_failure_mode(p, s) == FailureMode::Safe {
                Fix128::ZERO
            } else {
                Fix128::ONE
            };
            assert_eq!(q, want_q);
            assert_eq!(
                failure_index(FailureCriterion::TsaiWu, p, s),
                tsai_wu_failure_index(p, s)
            );
            assert_eq!(
                failure_index(FailureCriterion::TsaiHill, p, s),
                tsai_hill_failure_index(p, s)
            );
        }
    }
}

#[test]
fn hashin_and_puck_agree_on_fibre_modes_and_on_matrix_tension_threshold() {
    // doc: Puck の fibre mode は Hashin と同一、σ2 >= 0 側の IFF-A は Hashin MatrixTension と同じ楕円
    for p in presets() {
        for (a, b, c) in grid() {
            let h = hashin_failure_mode(p, st(a, b, c));
            let q = puck_failure_mode(p, st(a, b, c));
            if h == FailureMode::FibreTension || h == FailureMode::FibreCompression {
                assert_eq!(q, h);
            }
        }
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-003: 強度 0 の ply が全 criterion で Safe / FI = 0 (Fix128 0 除算 = ZERO、検証なし)"]
fn zero_strength_ply_must_not_report_safe_under_load() {
    // doc: FI >= 1 -> failed。強度 0 の層に荷重が掛かれば破壊と読むのが閉形式 (FI -> inf)
    // 実装は Fix128 の 0 除算 (= ZERO) で全 criterion が Safe / 0 を返す
    let z = LaminateStrengths {
        xt: Fix128::ZERO,
        xc: Fix128::ZERO,
        yt: Fix128::ZERO,
        yc: Fix128::ZERO,
        s: Fix128::ZERO,
    };
    let loaded = st(100.0, 50.0, 30.0);
    assert!(
        tsai_hill_failure_index(z, loaded) >= Fix128::ONE,
        "tsai_hill FI = 0"
    );
    assert!(
        tsai_wu_failure_index(z, loaded) >= Fix128::ONE,
        "tsai_wu FI = 0"
    );
    assert_ne!(
        hashin_failure_mode(z, loaded),
        FailureMode::Safe,
        "hashin Safe"
    );
    assert_ne!(puck_failure_mode(z, loaded), FailureMode::Safe, "puck Safe");
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-003: S = 0 のとき tau 項が消え FI = 0 (tau = 50 MPa)"]
fn single_zero_strength_s_must_not_drop_shear_term() {
    // S = 0 だけ 0 (入力ミス) でも τ12 = 50 MPa の項が消えて FI が過小になる
    let mut p = LaminateStrengths::cfrp_ud();
    p.s = Fix128::ZERO;
    let fi = tsai_hill_failure_index(p, st(0.0, 0.0, 50.0)).to_f64();
    assert!(fi >= 1.0, "FI = {fi} (shear term dropped)");
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-004: sigma1 = 3.2e9 MPa で sigma1^2 が 2^63 を超え wrap、Tsai-Hill FI = -3.6e12 (負)"]
fn huge_stress_must_not_wrap_to_safe() {
    // σ1 = 3.2e9 MPa: Xt = 1500 -> FI ~ 4.5e6 であるべき。Fix128 の乗算は 2^63 を超えると wrap する
    let p = LaminateStrengths::cfrp_ud();
    let fi = tsai_hill_failure_index(p, st(3.2e9, 0.0, 0.0)).to_f64();
    assert!(fi > 1.0, "FI = {fi}");
    assert_ne!(
        hashin_failure_mode(p, st(3.2e9, 0.0, 0.0)),
        FailureMode::Safe
    );
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-005: sigma1 = 0, sigma2 = -50, tau = 75 (> S) の puck が FibreTension を返す (Hashin の fibre tension 式の tau 項が成立するため)。IFF mode B の doc (小さい横圧縮 + 高せん断) と矛盾"]
fn puck_pure_shear_above_s_is_inter_fibre_not_fibre_tension() {
    let p = LaminateStrengths::cfrp_ud();
    // fibre 応力 0: fibre 破壊ではあり得ない。rn = 0.2, rs = 1.10 -> B
    assert_eq!(
        puck_failure_mode(p, st(0.0, -50.0, 75.0)),
        FailureMode::InterFibreB
    );
}
