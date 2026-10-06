//! Audit oracles for fatigue
//!
//! 公開 API (`SnCurve` / `from_fdm_material` / `miner_damage`) だけを対象にする
//! (`cycles_to_failure` / `stress_at_cycles` / `analyze_spectrum` / 金属 preset は pub(crate))
//! 期待値は Basquin N = N_e (S_e / S)^m と Miner D = Σ n_i / N_i を f64 で独立に評価したもの

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::fatigue::{miner_damage, SnCurve};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

fn curve(se: i64, ne: u64, m: u32, uts: i64) -> SnCurve {
    SnCurve {
        ultimate_tensile_mpa: Fix128::from_int(uts),
        endurance_stress_mpa: Fix128::from_int(se),
        endurance_cycles: ne,
        fatigue_exponent_m: m,
    }
}

/// Basquin の寿命 (f64、整数 m は掛け算で)
fn ref_n(se: f64, ne: f64, m: u32, s: f64) -> f64 {
    let r = se / s;
    let mut p = 1.0;
    for _ in 0..m {
        p *= r;
    }
    ne * p
}

fn presets() -> Vec<(&'static str, SnCurve, f64, f64, u32, f64)> {
    // (name, curve, S_e, N_e, m, UTS)
    vec![
        // 実装の preset (pub(crate)) と同じ値を公開の struct 構造で組む
        (
            "sus304",
            curve(240, 10_000_000, 10, 505),
            240.0,
            1e7,
            10,
            505.0,
        ),
        ("a5052", curve(92, 5_000_000, 6, 230), 92.0, 5e6, 6, 230.0),
        ("poly", curve(18, 1_000_000, 5, 60), 18.0, 1e6, 5, 60.0),
        ("m1", curve(50, 100_000, 1, 400), 50.0, 1e5, 1, 400.0),
        ("m3", curve(50, 2_000_000, 3, 400), 50.0, 2e6, 3, 400.0),
    ]
}

#[test]
fn single_entry_damage_is_n_over_floor_of_basquin_life() {
    // 実装は N = floor(N_e (S_e/S)^m) (1 未満は 1) で D = n / N と等価
    for (name, c, se, ne, m, uts) in presets() {
        for k in [1.02, 1.1, 1.3, 1.7, 2.0, 3.0] {
            let s = se * k;
            if s > uts {
                continue;
            }
            let n_ref = ref_n(se, ne, m, s);
            let n_floor = n_ref.floor().max(1.0);
            let applied = 1000u64;
            let d = miner_damage(&[(Fix128::from_f64(s), applied)], &c).to_f64();
            // 実装側の切り捨ては ratio の Fix128 丸め由来で ±1 cycle ずれうる
            let lo = applied as f64 / (n_floor + 1.0);
            let hi = applied as f64 / (n_floor - 1.0).max(1.0);
            assert!(
                d >= lo * (1.0 - 1e-9) && d <= hi * (1.0 + 1e-9),
                "{name} k={k}: D={d} want ~{}",
                applied as f64 / n_floor
            );
        }
    }
}

#[test]
fn miner_is_linear_in_applied_cycles_and_additive_over_entries() {
    let c = curve(100, 1_000_000, 5, 400);
    let sa = Fix128::from_int(150);
    let sb = Fix128::from_int(220);
    let da = miner_damage(&[(sa, 500)], &c);
    let db = miner_damage(&[(sb, 700)], &c);
    let dab = miner_damage(&[(sa, 500), (sb, 700)], &c);
    assert_eq!(dab, da + db);
    // 順序に依らない
    assert_eq!(miner_damage(&[(sb, 700), (sa, 500)], &c), dab);
    // n を 4 倍にすると damage は 4 倍 (除算の丸め 4 ulp 以内)
    let d4 = miner_damage(&[(sa, 2000)], &c);
    let diff = (d4 - da * Fix128::from_int(4)).abs();
    assert!(diff <= Fix128::from_raw(0, 16), "{diff:?}");
    // 同じ stress の分割 (n1 + n2) は 1 ulp 級の差で一致
    let split = miner_damage(&[(sa, 200), (sa, 300)], &c);
    assert!((split - da).abs() <= Fix128::from_raw(0, 16));
}

#[test]
fn entries_at_or_below_endurance_or_non_positive_contribute_zero() {
    let c = curve(100, 1_000_000, 5, 400);
    for s in [100, 99, 1, 0, -1, -500] {
        assert_eq!(
            miner_damage(&[(Fix128::from_int(s), 1_000_000)], &c),
            Fix128::ZERO,
            "S={s}"
        );
    }
    // 境界直上は有限の正値
    let just = Fix128::from_int(100) + Fix128::from_raw(0, 1 << 20);
    assert!(miner_damage(&[(just, 1000)], &c) > Fix128::ZERO);
    assert_eq!(miner_damage(&[], &c), Fix128::ZERO);
    // n = 0 は damage 0
    assert_eq!(
        miner_damage(&[(Fix128::from_int(300), 0)], &c),
        Fix128::ZERO
    );
}

#[test]
fn damage_per_cycle_is_monotone_in_stress_and_in_exponent_direction() {
    let c = curve(100, 1_000_000, 6, 600);
    let mut prev = Fix128::ZERO;
    for s in [101, 110, 130, 160, 200, 300, 450] {
        let d = miner_damage(&[(Fix128::from_int(s), 1)], &c);
        assert!(d >= prev, "S={s}: {d:?} < {prev:?}");
        prev = d;
    }
    // 同じ S > S_e では m が大きいほど (S_e/S < 1 なので) ratio^m が小さく、寿命が短く damage は大きい
    let mut prev = Fix128::ZERO;
    for m in [1u32, 2, 3, 5, 8, 10] {
        let d = miner_damage(
            &[(Fix128::from_int(150), 1000)],
            &curve(100, 1_000_000, m, 600),
        );
        assert!(d >= prev, "m={m}");
        prev = d;
    }
}

#[test]
fn zero_exponent_gives_stress_independent_life_equal_to_endurance_cycles() {
    // N = N_e (S_e/S)^0 = N_e (S > S_e のとき)
    let c = curve(100, 1_000_000, 0, 600);
    let d = miner_damage(&[(Fix128::from_int(500), 1_000_000)], &c);
    assert_eq!(d, Fix128::ONE);
}

#[test]
fn full_life_at_doubled_endurance_stress_is_exactly_one() {
    // S = 2 S_e なので ratio = 1/2 は dyadic: N = N_e / 2^m は厳密 -> n = N で D = 1
    for (m, ne) in [
        (1u32, 1_000_000u64),
        (3, 800_000),
        (5, 1_000_000),
        (10, 10_240_000),
    ] {
        let c = curve(100, ne, m, 600);
        let n = ne >> m;
        let d = miner_damage(&[(Fix128::from_int(200), n)], &c);
        assert_eq!(d, Fix128::ONE, "m={m}");
    }
}

#[test]
fn from_fdm_material_uses_documented_polymer_defaults_for_every_preset() {
    // doc: S_e = 0.3 UTS, N_e = 1e6, m = 5, UTS はそのまま
    let mats = [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
        MaterialProperties::pc(),
        MaterialProperties::tpu(),
        MaterialProperties::nylon(),
        MaterialProperties::cf_nylon(),
        MaterialProperties::peek(),
    ];
    for m in mats {
        let c = SnCurve::from_fdm_material(&m);
        assert_eq!(c.ultimate_tensile_mpa, m.tensile_strength_mpa, "{}", m.name);
        assert_eq!(c.endurance_cycles, 1_000_000);
        assert_eq!(c.fatigue_exponent_m, 5);
        let want = m.tensile_strength_mpa.to_f64() * 0.3;
        assert!(
            (c.endurance_stress_mpa.to_f64() - want).abs() < 1e-9,
            "{}",
            m.name
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-006: ultimate_tensile_mpa は寿命計算で一切使われない。PLA 曲線で S = UTS の寿命は N_e (0.3)^5 = 2430 cycles (doc は 1 cycle 近似)、1 cycle の damage は 4e-4"]
fn static_failure_at_ultimate_strength_is_one_cycle() {
    // doc (SnCurve::ultimate_tensile_mpa): "Ultimate tensile strength (MPa). Cycle count = 1 approximate."
    // かつ cycles_to_failure: S > UTS は静的破壊として "very small cycle count"
    // PLA 由来の曲線で S = UTS を 1 cycle 掛けたときの damage は >= 1 のはず
    let c = SnCurve::from_fdm_material(&MaterialProperties::pla());
    let d = miner_damage(&[(c.ultimate_tensile_mpa, 1)], &c);
    assert!(d >= Fix128::ONE, "D(S = UTS, 1 cycle) = {}", d.to_f64());
    // S = 2 UTS でも同様
    let d2 = miner_damage(&[(c.ultimate_tensile_mpa * Fix128::from_int(2), 1)], &c);
    assert!(d2 >= Fix128::ONE, "D(S = 2 UTS, 1 cycle) = {}", d2.to_f64());
}

#[test]
fn applied_cycles_above_i64_max_do_not_flip_the_damage_sign() {
    // AUD-A-S4W2-007: n = u64::MAX used to go through `as i64` and wrap negative
    let c = curve(100, 1_000_000, 5, 400);
    let d = miner_damage(&[(Fix128::from_int(300), u64::MAX)], &c);
    assert!(d > Fix128::ZERO, "D = {d:?} (n = u64::MAX)");
    // N = 1e6 (300/100)^-5 = 4115 cycles (floor): D = n / 4115 exactly in quotient + remainder form
    let n_fail = 1_000_000u64 * 100_000 / 24_300_000; // 1e6 / 3^5 = 4115.2 -> 4115
    assert_eq!(n_fail, 4115);
    let q = u64::MAX / n_fail;
    let r = u64::MAX % n_fail;
    let want =
        Fix128::from_int(q as i64) + Fix128::from_int(r as i64) / Fix128::from_int(n_fail as i64);
    assert_eq!(d, want);
    // independent of the quotient + remainder form: n / N in f64
    let f = u64::MAX as f64 / 4115.0;
    assert!(
        (d.to_f64() - f).abs() / f < 1e-12,
        "D = {} vs {f}",
        d.to_f64()
    );
}

#[test]
fn damage_beyond_the_representable_range_saturates_instead_of_wrapping() {
    // S = 10 S_e, m = 10: N = 1 cycle, so D = n; n = u64::MAX exceeds the Fix128 integer range
    let c = curve(100, 1_000_000, 10, 6000);
    let d = miner_damage(&[(Fix128::from_int(1000), u64::MAX)], &c);
    // the largest Fix128: integer part i64::MAX, every fraction bit set
    assert_eq!(d, Fix128::from_raw(i64::MAX, u64::MAX), "D = {d:?}");
    // a sum of two near-maximal terms does not wrap either
    let i = i64::MAX as u64;
    let d2 = miner_damage(
        &[(Fix128::from_int(1000), i), (Fix128::from_int(1000), i)],
        &c,
    );
    assert!(d2 >= d, "D2 = {d2:?}");
}

#[test]
fn extreme_stress_clamps_life_to_one_cycle() {
    // S = 10 S_e, m = 10: N = 1e6 * 1e-10 = 1e-4 -> 1 cycle (下限)。D = n / 1 = n
    let c = curve(100, 1_000_000, 10, 6000);
    let d = miner_damage(&[(Fix128::from_int(1000), 7)], &c);
    assert_eq!(d, Fix128::from_int(7));
    // N が 1 ちょうどになる境界の少し上でも 1 以上
    let d2 = miner_damage(&[(Fix128::from_int(300), 100)], &curve(100, 1000, 6, 6000));
    // N = 1000 / 3^6 = 1.37 -> 1 cycle (切り捨て) なので D = 100
    assert_eq!(d2, Fix128::from_int(100));
}

#[test]
fn infinite_life_entries_do_not_stop_the_accumulation_of_later_entries() {
    // 耐久限以下の entry が先頭にあっても、後続の entry の damage は加算される (Miner の和は全 entry)
    let c = curve(100, 1_000_000, 5, 400);
    let damaging = (Fix128::from_int(200), 5000u64);
    let alone = miner_damage(&[damaging], &c);
    assert!(alone > Fix128::ZERO);
    let with_safe_first = miner_damage(&[(Fix128::from_int(50), 1_000_000), damaging], &c);
    assert_eq!(with_safe_first, alone);
    let with_safe_between = miner_damage(
        &[damaging, (Fix128::from_int(100), 1_000_000), damaging],
        &c,
    );
    assert_eq!(with_safe_between, alone + alone);
}

/// N_e = 2^64 - 4 (beyond `i64::MAX`), S_e / S = 3 / 4 exactly, m = 1:
/// N = 0.75 N_e = 3 (2^62 - 1), itself beyond `i64::MAX`. The endurance count
/// used to go through `as i64` (wrapping to -4), the life fell to 1 cycle and
/// the damage of N cycles read N instead of 1
#[test]
fn an_endurance_count_above_i64_max_keeps_its_life() {
    let ne = u64::MAX - 3;
    let n_fail = 3 * ((1u64 << 62) - 1);
    assert_eq!(n_fail, ne / 4 * 3); // oracle: N = N_e (3/4)^1, exact in integers
    let c = curve(3, ne, 1, 400);
    assert_eq!(
        miner_damage(&[(Fix128::from_int(4), n_fail)], &c),
        Fix128::ONE
    );
    // half the life (n_fail is odd: floor) is half the damage, 1/2 - 1/(2 N)
    let half = miner_damage(&[(Fix128::from_int(4), n_fail / 2)], &c).to_f64();
    assert!((half - 0.5).abs() < 1e-15, "D = {half}");
}

/// m = 2, S_e / S = 1 / 2: N = N_e / 4 with N_e = 3 * 2^62, so N = 3 * 2^60
/// (within `i64::MAX`, only the endurance count is beyond it)
#[test]
fn an_endurance_count_above_i64_max_with_a_life_within_it() {
    let ne = 3u64 << 62;
    let c = curve(100, ne, 2, 400);
    let d = miner_damage(&[(Fix128::from_int(200), 3u64 << 60)], &c);
    assert_eq!(d, Fix128::ONE);
}

/// The Basquin inverse with N_e = 3 * 2^62 and m = 1: at n = N_e / 2 the
/// stress is 2 S_e (oracle S = S_e (N_e / n)^(1/m)); the endurance count used to
/// wrap negative through `as i64`
#[test]
fn the_basquin_inverse_takes_an_endurance_count_above_i64_max() {
    let s = StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: Fix128::from_int(15),
            height_mm: Fix128::from_int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: Fix128::ONE,
            length_mm: Fix128::from_int(200),
        },
        MaterialProperties::sus304(),
    )
    .with_sn_curve(curve(100, 3u64 << 62, 1, 400));
    let got = s.fatigue_strength_mpa(3u64 << 61).unwrap();
    assert!((got.to_f64() - 200.0).abs() < 1e-9, "S = {}", got.to_f64());
    // and with n beyond i64::MAX too (n = 0.75 N_e): S = S_e / 0.75
    let got = s.fatigue_strength_mpa(9u64 << 60).unwrap();
    assert!(
        (got.to_f64() - 400.0 / 3.0).abs() < 1e-9,
        "S = {}",
        got.to_f64()
    );
}

/// A negative S_e is not a physical curve; Basquin is kept as written. With
/// m odd the ratio S_e / S < 0 gives a negative "life", answered as the
/// one-cycle minimum (D = n). With m even (-2)^2 = 4, so N = 4 N_e
#[test]
fn a_negative_endurance_stress_keeps_basquin_as_written() {
    let d = miner_damage(&[(Fix128::from_int(50), 7)], &curve(-100, 1000, 1, 400));
    assert_eq!(d, Fix128::from_int(7));
    let d = miner_damage(&[(Fix128::from_int(50), 4000)], &curve(-100, 1000, 2, 400));
    assert_eq!(d, Fix128::ONE);
}

/// A life beyond u64 saturates one below the infinite-life sentinel: S_e = -100,
/// S = 50, m = 2 gives (S_e / S)^2 = 4, so N = 4 N_e > u64::MAX for N_e = u64::MAX / 2.
/// The saturated life is u64::MAX - 1 (finite: u64::MAX means infinite and adds
/// no damage), so n = u64::MAX - 1 is exactly one life
#[test]
fn a_life_beyond_u64_saturates_below_the_infinite_sentinel() {
    let c = curve(-100, u64::MAX / 2, 2, 400);
    let d = miner_damage(&[(Fix128::from_int(50), u64::MAX - 1)], &c);
    assert_eq!(d, Fix128::ONE);
}

/// A finite life between 2^32 and i64::MAX with a small n keeps every bit:
/// N_e = 2^41 + 2, S = 2 S_e, m = 1, so N = 2^40 + 1 and D = 3 / (2^40 + 1)
#[test]
fn a_long_life_and_a_small_count_keep_their_ratio_exactly() {
    let n_fail = (1i64 << 40) + 1;
    let c = curve(100, (1u64 << 41) + 2, 1, 400);
    let d = miner_damage(&[(Fix128::from_int(200), 3)], &c);
    assert_eq!(d, Fix128::from_int(3) / Fix128::from_int(n_fail));
}
