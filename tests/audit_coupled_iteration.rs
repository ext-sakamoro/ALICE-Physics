//! Audit S3W3 oracles for `src/coupled_iteration.rs`.
//!
//! Closed forms used (all independent of the code under test):
//! * `sqrt(3^2 + 4^2) = 5` and similar Pythagorean residuals for the L2 norm;
//! * `L2_TERM_FLOOR^2 = 2^-64 = 1 ulp`, `(floor - 1 ulp)^2 < 1 ulp`;
//! * the monitor's decisions on dyadic residual sequences (`2^-k`), where every
//!   quotient and product is exact in `Q64.64`, so the expected verdict and the
//!   sweep at which it fires follow from the documented rules by hand;
//! * the model problem `e_{j+1} = -(m_f/m) e_j` of the module documentation.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_iteration::{
    residual_norm_inf, residual_norm_l2_checked, run_sub_iteration, ConfigFault,
    ContractionMonitor, CoupledIterationError, EquilibrationScale, MonitorVerdict,
    SubIterationConfig, L2_TERM_FLOOR,
};
use alice_physics::math::Fix128;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn ulp() -> Fix128 {
    Fix128::from_raw(0, 1)
}

fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

// ---------------------------------------------------------------------------
// residual_norm_inf / residual_norm_l2_checked / L2_TERM_FLOOR
// ---------------------------------------------------------------------------

#[test]
fn inf_norm_is_the_largest_magnitude_with_sign_dropped() {
    let v = [r(1, 2), r(-7, 2), r(3, 1), Fix128::ZERO];
    assert_eq!(residual_norm_inf(&v), r(7, 2));
    assert_eq!(residual_norm_inf(&[]), Fix128::ZERO);
    assert_eq!(
        residual_norm_inf(&[Fix128::ZERO, Fix128::ZERO]),
        Fix128::ZERO
    );
    // a single negative component
    assert_eq!(residual_norm_inf(&[r(-5, 1)]), r(5, 1));
}

#[test]
fn inf_norm_is_faithful_over_the_whole_representable_range() {
    // Doc: "stays faithful across the whole range in which the components
    // themselves are representable". The most negative representable
    // component has magnitude 2^63.
    let most_negative = Fix128::from_raw(i64::MIN, 0);
    let n = residual_norm_inf(&[most_negative]);
    assert!(
        n > Fix128::from_int(i64::MAX / 2),
        "norm of the most negative component is {n:?}, expected about 9.2e18"
    );
}

#[test]
fn l2_floor_is_exactly_the_square_root_of_one_ulp() {
    assert_eq!(raw(L2_TERM_FLOOR), 1_i128 << 32);
    assert_eq!(raw(L2_TERM_FLOOR * L2_TERM_FLOOR), 1);
    let below = L2_TERM_FLOOR - ulp();
    assert_eq!(
        raw(below * below),
        0,
        "just below the floor squares to zero"
    );
    assert!((L2_TERM_FLOOR.to_f64() - 2.328_306_436_538_696e-10).abs() < 1e-22);
}

#[test]
fn l2_matches_pythagoras_for_signed_components() {
    assert_eq!(
        residual_norm_l2_checked(&[r(3, 1), r(-4, 1)]).unwrap(),
        Fix128::from_int(5)
    );
    // 1,2,2 -> 3 ; 2,3,6 -> 7
    assert_eq!(
        residual_norm_l2_checked(&[r(1, 1), r(2, 1), r(-2, 1)]).unwrap(),
        Fix128::from_int(3)
    );
    assert_eq!(
        residual_norm_l2_checked(&[r(-2, 1), r(3, 1), r(6, 1)]).unwrap(),
        Fix128::from_int(7)
    );
    assert_eq!(residual_norm_l2_checked(&[]).unwrap(), Fix128::ZERO);
}

#[test]
fn l2_error_carries_the_offending_component_index() {
    // component 2^32 squares to 2^64, which wraps the integer field to zero
    let bad = Fix128::from_int(1_i64 << 32);
    let err = residual_norm_l2_checked(&[Fix128::ONE, Fix128::ONE, bad]).unwrap_err();
    assert_eq!(err, CoupledIterationError::ArithmeticWrapped { sweeps: 2 });
    let err = residual_norm_l2_checked(&[bad]).unwrap_err();
    assert_eq!(err, CoupledIterationError::ArithmeticWrapped { sweeps: 0 });
}

#[test]
fn l2_accepts_a_component_exactly_at_the_floor_and_the_largest_safe_one() {
    // at the floor: square is exactly 1 ulp, sqrt = floor
    let n = residual_norm_l2_checked(&[L2_TERM_FLOOR]).unwrap();
    assert_eq!(n, L2_TERM_FLOOR);
    // largest component whose square is representable: 3.0e9 (square 9e18 < 2^63)
    let n = residual_norm_l2_checked(&[Fix128::from_int(3_000_000_000)]).unwrap();
    assert!((n.to_f64() - 3.0e9).abs() < 1.0);
}

#[test]
fn l2_refuses_every_component_whose_square_leaves_the_representable_range() {
    // Doc: "reports the loss instead of returning a plausible number". Squares
    // of 4.5e9 and 5.0e9 exceed 2^63 (9.22e18) and wrap modulo 2^64; for
    // 4.5e9 the wrapped value (1.9e18 after wrap) is positive, so neither the
    // `next < sum` nor the `square == 0` guard fires.
    for big in [
        3_100_000_000_i64,
        4_000_000_000,
        4_500_000_000,
        5_000_000_000,
        6_000_000_000,
    ] {
        let res = residual_norm_l2_checked(&[Fix128::from_int(big)]);
        assert!(
            res.is_err(),
            "component {big} (square {:e} > 2^63) returned {:?} instead of ArithmeticWrapped",
            (big as f64) * (big as f64),
            res.map(Fix128::to_f64)
        );
    }
}

// ---------------------------------------------------------------------------
// EquilibrationScale
// ---------------------------------------------------------------------------

#[test]
fn covering_is_the_next_power_of_two_for_magnitudes_above_one() {
    for (m, e) in [
        (r(1, 1), 0u32),
        (Fix128::ONE + ulp(), 1),
        (r(2, 1), 1),
        (r(3, 1), 2),
        (r(4, 1), 2),
        (r(1000, 1), 10),
        (r(1024, 1), 10),
    ] {
        let s = EquilibrationScale::covering(m).unwrap();
        assert_eq!(s.exponent(), e, "magnitude {m:?}");
        assert_eq!(s.factor(), Fix128::from_int(1 << e));
    }
    // 2^62 is accepted, 2^62 + 1 is refused
    let top = Fix128::from_int(1_i64 << 62);
    assert_eq!(EquilibrationScale::covering(top).unwrap().exponent(), 62);
    assert_eq!(
        EquilibrationScale::covering(top + Fix128::ONE),
        Err(ConfigFault::ScaleOutOfRange)
    );
    assert_eq!(EquilibrationScale::MAX_EXPONENT, 62);
}

#[test]
fn covering_zero_and_negative_magnitudes_are_the_identity() {
    assert_eq!(
        EquilibrationScale::covering(Fix128::ZERO).unwrap(),
        EquilibrationScale::IDENTITY
    );
    assert_eq!(
        EquilibrationScale::covering(r(-9, 1)).unwrap(),
        EquilibrationScale::IDENTITY
    );
    assert_eq!(EquilibrationScale::IDENTITY.factor(), Fix128::ONE);
}

#[test]
fn covering_scales_only_down_and_leaves_a_magnitude_of_one_or_less_alone() {
    // Doc: the smallest power of two `2^e`, `e >= 0`, at or above `magnitude`;
    // above 1 the largest entry lands in (1/2, 1], at or below 1 the factor is
    // the identity (the exponent is unsigned, the scale only divides).
    for (num, den) in [(3, 10), (1, 2), (1, 1)] {
        let m = r(num, den);
        let s = EquilibrationScale::covering(m).unwrap();
        assert_eq!(s, EquilibrationScale::IDENTITY, "{num}/{den} got {s:?}");
        assert_eq!(s.scale_down(m), m);
    }
    for (num, den) in [(3, 2), (5, 1), (1_000_001, 1000)] {
        let m = r(num, den);
        let scaled = EquilibrationScale::covering(m).unwrap().scale_down(m);
        assert!(
            scaled > r(1, 2) && scaled <= Fix128::ONE,
            "{num}/{den} scaled to {} (documented range (1/2, 1])",
            scaled.to_f64()
        );
    }
}

#[test]
fn scaling_up_then_down_is_exact_and_down_then_up_is_bounded() {
    let values = [
        r(3, 7),
        r(-3, 7),
        r(123_456, 1000),
        r(-1, 3),
        Fix128::from_raw(0, 1),
        Fix128::from_raw(-1, u64::MAX),
        Fix128::from_raw(5, 0x1234_5678_9abc_def1),
    ];
    for e in [0u32, 1, 5, 17, 40, 62] {
        let s = EquilibrationScale::covering(Fix128::from_int(1_i64 << e)).unwrap();
        assert_eq!(s.exponent(), e);
        let bound = raw(s.round_trip_bound());
        assert_eq!(bound, (1_i128 << e) - 1, "bound is (2^e - 1) ulp");
        for v in values {
            if e <= 40 {
                assert_eq!(s.scale_down(s.scale_up(v)), v, "up then down, e {e}");
            }
            let back = s.scale_up(s.scale_down(v));
            let err = (raw(v) - raw(back)).abs();
            assert!(
                err <= bound,
                "down-up error {err} > bound {bound} (e {e}, v {v:?})"
            );
        }
    }
}

#[test]
fn scale_down_is_an_arithmetic_shift_flooring_toward_minus_infinity() {
    let s = EquilibrationScale::covering(Fix128::from_int(4)).unwrap(); // 2^2
    assert_eq!(s.scale_down(Fix128::from_int(8)), Fix128::from_int(2));
    assert_eq!(s.scale_down(Fix128::from_int(-8)), Fix128::from_int(-2));
    // -1 ulp floors to -1 ulp, +3 ulp floors to 0
    assert_eq!(raw(s.scale_down(Fix128::from_raw(-1, u64::MAX))), -1);
    assert_eq!(raw(s.scale_down(Fix128::from_raw(0, 3))), 0);
    assert_eq!(s.scale_up(r(3, 2)), Fix128::from_int(6));
}

// ---------------------------------------------------------------------------
// SubIterationConfig
// ---------------------------------------------------------------------------

fn good() -> SubIterationConfig {
    SubIterationConfig::default()
}

#[test]
fn validate_boundaries_are_exact() {
    let mut c = good();
    c.max_sweeps = 1;
    assert!(c.validate().is_ok());
    c.max_sweeps = 0;
    assert_eq!(c.validate(), Err(ConfigFault::ZeroSweepBudget));

    let mut c = good();
    c.relative_tolerance = ulp();
    assert!(c.validate().is_ok());
    c.relative_tolerance = Fix128::ZERO;
    assert_eq!(c.validate(), Err(ConfigFault::NonPositiveTolerance));
    c.relative_tolerance = r(-1, 2);
    assert_eq!(c.validate(), Err(ConfigFault::NonPositiveTolerance));

    let mut c = good();
    c.stagnation_fraction = ulp();
    assert!(c.validate().is_ok());
    c.stagnation_fraction = Fix128::ONE - ulp();
    assert!(c.validate().is_ok());
    c.stagnation_fraction = Fix128::ZERO;
    assert_eq!(c.validate(), Err(ConfigFault::StagnationFractionOutOfRange));
    c.stagnation_fraction = Fix128::ONE;
    assert_eq!(c.validate(), Err(ConfigFault::StagnationFractionOutOfRange));
    c.stagnation_fraction = r(3, 2);
    assert_eq!(c.validate(), Err(ConfigFault::StagnationFractionOutOfRange));
    c.stagnation_fraction = r(-1, 2);
    assert_eq!(c.validate(), Err(ConfigFault::StagnationFractionOutOfRange));

    let mut c = good();
    c.divergence_ratio = Fix128::ONE + ulp();
    assert!(c.validate().is_ok());
    c.divergence_ratio = Fix128::ONE;
    assert_eq!(c.validate(), Err(ConfigFault::DivergenceRatioNotAboveOne));
    c.divergence_ratio = r(1, 2);
    assert_eq!(c.validate(), Err(ConfigFault::DivergenceRatioNotAboveOne));

    let mut c = good();
    c.ratio_samples = 2;
    assert!(c.validate().is_ok());
    c.ratio_samples = 1;
    assert_eq!(c.validate(), Err(ConfigFault::InsufficientRatioSamples));
    c.ratio_samples = 0;
    assert_eq!(c.validate(), Err(ConfigFault::InsufficientRatioSamples));
}

#[test]
fn validate_reports_the_first_fault_in_field_order() {
    let c = SubIterationConfig {
        max_sweeps: 0,
        relative_tolerance: Fix128::ZERO,
        stagnation_fraction: Fix128::ONE,
        divergence_ratio: Fix128::ONE,
        ratio_samples: 0,
    };
    assert_eq!(c.validate(), Err(ConfigFault::ZeroSweepBudget));
    let c = SubIterationConfig { max_sweeps: 5, ..c };
    assert_eq!(c.validate(), Err(ConfigFault::NonPositiveTolerance));
    let c = SubIterationConfig {
        relative_tolerance: r(1, 8),
        ..c
    };
    assert_eq!(c.validate(), Err(ConfigFault::StagnationFractionOutOfRange));
    let c = SubIterationConfig {
        stagnation_fraction: r(1, 2),
        ..c
    };
    assert_eq!(c.validate(), Err(ConfigFault::DivergenceRatioNotAboveOne));
    let c = SubIterationConfig {
        divergence_ratio: r(2, 1),
        ..c
    };
    assert_eq!(c.validate(), Err(ConfigFault::InsufficientRatioSamples));
}

#[test]
fn new_stores_the_fields_in_argument_order() {
    let c = SubIterationConfig::new(7, r(1, 8), r(1, 4), r(3, 2), 5).unwrap();
    assert_eq!(c.max_sweeps, 7);
    assert_eq!(c.relative_tolerance, r(1, 8));
    assert_eq!(c.stagnation_fraction, r(1, 4));
    assert_eq!(c.divergence_ratio, r(3, 2));
    assert_eq!(c.ratio_samples, 5);
    assert_eq!(
        SubIterationConfig::new(0, r(1, 8), r(1, 4), r(3, 2), 5),
        Err(ConfigFault::ZeroSweepBudget)
    );
}

#[test]
fn default_values_are_the_documented_ones() {
    let d = SubIterationConfig::default();
    assert_eq!(d.max_sweeps, 64);
    assert_eq!(d.relative_tolerance, r(1, 1024));
    assert_eq!(d.stagnation_fraction, r(1, 2));
    assert_eq!(d.divergence_ratio, Fix128::ONE + r(1, 16));
    assert_eq!(d.ratio_samples, 4);
}

// ---------------------------------------------------------------------------
// ContractionMonitor / run_sub_iteration on dyadic sequences
// ---------------------------------------------------------------------------

fn feed(m: &mut ContractionMonitor, seq: &[Fix128]) -> Vec<MonitorVerdict> {
    seq.iter().map(|&x| m.observe(x)).collect()
}

fn pow2(k: i32) -> Fix128 {
    if k >= 0 {
        Fix128::from_int(1_i64 << k)
    } else {
        Fix128::from_raw(0, 1u64 << (64 + k))
    }
}

#[test]
fn halving_residuals_converge_exactly_when_they_reach_one_over_1024() {
    // r_k = 2^-(k-1): sweep 11 gives 2^-10 = first * tolerance, so `<=` fires
    // at sweep 11 and not before.
    let mut m = ContractionMonitor::new(good()).unwrap();
    let seq: Vec<_> = (0..11).map(|k| pow2(-k)).collect();
    let v = feed(&mut m, &seq);
    for (i, verdict) in v.iter().enumerate().take(10) {
        assert_eq!(*verdict, MonitorVerdict::Continue, "sweep {}", i + 1);
    }
    assert_eq!(v[10], MonitorVerdict::Converged);
    assert_eq!(m.sweeps(), 11);
    assert_eq!(m.observed_ratio(), Some(r(1, 2)));
    assert_eq!(m.best_residual(), pow2(-10));
}

#[test]
fn a_zero_first_residual_is_converged_at_sweep_one() {
    let mut m = ContractionMonitor::new(good()).unwrap();
    assert_eq!(m.observe(Fix128::ZERO), MonitorVerdict::Converged);
    assert_eq!(m.sweeps(), 1);
    assert_eq!(m.observed_ratio(), None);
}

#[test]
fn ratio_is_read_exactly_at_the_ratio_samples_sweep() {
    let mut cfg = good();
    cfg.ratio_samples = 3;
    let mut m = ContractionMonitor::new(cfg).unwrap();
    m.observe(pow2(0));
    m.observe(pow2(-1));
    assert_eq!(m.observed_ratio(), None, "not yet: 2 sweeps seen, 3 needed");
    m.observe(pow2(-3)); // ratio of the last two: 1/4
    assert_eq!(m.observed_ratio(), Some(r(1, 4)));
    // later sweeps do not overwrite it
    m.observe(pow2(-4));
    assert_eq!(m.observed_ratio(), Some(r(1, 4)));
}

#[test]
fn doubling_residuals_are_diverging_at_the_fourth_sweep() {
    let mut m = ContractionMonitor::new(good()).unwrap();
    let v = feed(&mut m, &[pow2(0), pow2(1), pow2(2), pow2(3)]);
    assert_eq!(&v[..3], &[MonitorVerdict::Continue; 3]);
    assert_eq!(
        v[3],
        MonitorVerdict::Stop(CoupledIterationError::Diverging {
            observed_ratio: Fix128::from_int(2),
            sweeps: 4
        })
    );
}

#[test]
fn a_ratio_exactly_at_the_divergence_threshold_is_not_divergent() {
    // 4096, 4352, 4624, 4913: successive ratio 17/16 = 1 + 1/16 exactly
    let seq = [4096, 4352, 4624, 4913].map(Fix128::from_int);
    let mut m = ContractionMonitor::new(good()).unwrap();
    let v = feed(&mut m, &seq);
    assert_eq!(m.observed_ratio(), Some(Fix128::ONE + r(1, 16)));
    assert_eq!(
        v[3],
        MonitorVerdict::Continue,
        "ratio == threshold must not stop"
    );
}

#[test]
fn flat_residuals_stagnate_at_twice_the_best_sweep() {
    // 1, 1/2, 1/4 (best at sweep 3) then flat: stop_at = 3 / (1 - 1/2) = 6, so
    // the first stop is at sweep 7 with 4 sweeps since improvement.
    let q = pow2(-2);
    let mut m = ContractionMonitor::new(good()).unwrap();
    let v = feed(&mut m, &[pow2(0), pow2(-1), q, q, q, q, q]);
    for (i, verdict) in v.iter().enumerate().take(6) {
        assert_eq!(*verdict, MonitorVerdict::Continue, "sweep {}", i + 1);
    }
    assert_eq!(
        v[6],
        MonitorVerdict::Stop(CoupledIterationError::Stagnated {
            sweeps: 7,
            best_residual: q,
            first_residual: pow2(0),
            sweeps_since_improvement: 4,
        })
    );
}

#[test]
fn stagnation_fraction_moves_the_stopping_point() {
    // fraction 3/4: stop_at = 3 / (1/4) = 12 -> first stop at sweep 13
    let cfg = SubIterationConfig {
        stagnation_fraction: r(3, 4),
        max_sweeps: 100,
        ..good()
    };
    let q = pow2(-2);
    let mut m = ContractionMonitor::new(cfg).unwrap();
    let mut seq = vec![pow2(0), pow2(-1)];
    seq.extend(std::iter::repeat_n(q, 11));
    let v = feed(&mut m, &seq);
    assert!(v.iter().take(12).all(|x| *x == MonitorVerdict::Continue));
    assert_eq!(
        v[12],
        MonitorVerdict::Stop(CoupledIterationError::Stagnated {
            sweeps: 13,
            best_residual: q,
            first_residual: pow2(0),
            sweeps_since_improvement: 10,
        })
    );
}

#[test]
fn a_tie_with_the_best_does_not_reset_the_best_sweep() {
    // equal residual is not an improvement: best_sweep stays 3 (see above), so
    // `sweeps_since_improvement` counts from the first occurrence.
    let q = pow2(-2);
    let mut m = ContractionMonitor::new(good()).unwrap();
    feed(&mut m, &[pow2(0), pow2(-1), q, q]);
    assert_eq!(m.best_residual(), q);
}

#[test]
fn budget_exhaustion_reports_the_last_residual_not_the_best() {
    // wobbling sequence with a wide stagnation window so that NotConverged
    // (not Stagnated) is the stop: best 1/2 at sweep 2, last 0.5625
    let cfg = SubIterationConfig {
        max_sweeps: 5,
        stagnation_fraction: r(9, 10),
        ..good()
    };
    let seq = [pow2(0), r(1, 2), r(5, 8), r(9, 16), r(9, 16)];
    let mut m = ContractionMonitor::new(cfg).unwrap();
    let v = feed(&mut m, &seq);
    assert_eq!(
        v[4],
        MonitorVerdict::Stop(CoupledIterationError::NotConverged {
            sweeps: 5,
            residual: r(9, 16)
        })
    );
}

#[test]
fn convergence_wins_over_the_budget_on_the_last_sweep() {
    let cfg = SubIterationConfig {
        max_sweeps: 11,
        ..good()
    };
    let seq: Vec<_> = (0..11).map(|k| pow2(-k)).collect();
    let mut m = ContractionMonitor::new(cfg).unwrap();
    let v = feed(&mut m, &seq);
    assert_eq!(v[10], MonitorVerdict::Converged);
}

#[test]
fn an_invalid_config_is_refused_at_construction_with_the_fault() {
    let cfg = SubIterationConfig {
        max_sweeps: 0,
        ..good()
    };
    assert_eq!(
        ContractionMonitor::new(cfg).unwrap_err(),
        CoupledIterationError::InvalidConfig {
            fault: ConfigFault::ZeroSweepBudget
        }
    );
}

#[test]
fn run_sub_iteration_passes_zero_based_sweep_indices_and_reports_the_final_state() {
    let mut seen = Vec::new();
    let report = run_sub_iteration(good(), |i| {
        seen.push(i);
        pow2(-(i as i32))
    })
    .unwrap();
    assert_eq!(seen, (0..11).collect::<Vec<u32>>());
    assert_eq!(report.sweeps, 11);
    assert_eq!(report.residual, pow2(-10));
    assert_eq!(report.observed_ratio, Some(r(1, 2)));
}

#[test]
fn run_sub_iteration_surfaces_the_monitor_stop_as_the_error() {
    let err = run_sub_iteration(good(), |i| pow2(i as i32)).unwrap_err();
    assert_eq!(
        err,
        CoupledIterationError::Diverging {
            observed_ratio: Fix128::from_int(2),
            sweeps: 4
        }
    );
    let bad = SubIterationConfig {
        ratio_samples: 1,
        ..good()
    };
    assert_eq!(
        run_sub_iteration(bad, |_| Fix128::ONE).unwrap_err(),
        CoupledIterationError::InvalidConfig {
            fault: ConfigFault::InsufficientRatioSamples
        }
    );
}

// ---------------------------------------------------------------------------
// Model problem of the module documentation
// ---------------------------------------------------------------------------

/// rho(dt) = m_f / (m + k dt^2) with m = k = 1 (module doc table).
#[test]
fn documented_rho_table_matches_the_closed_form() {
    let doc = [
        (8.0, 0.5, 0.492_307_692_307_692_3),
        (80.0, 0.5, 0.499_921_887_205_124_2),
        (800.0, 0.5, 0.499_999_218_751_220_7),
        (8000.0, 0.5, 0.499_999_992_187_500_1),
        (8.0, 2.0, 1.969_230_769_230_769_3),
        (80.0, 2.0, 1.999_687_548_820_496_8),
        (800.0, 2.0, 1.999_996_875_004_882_7),
        (8000.0, 2.0, 1.999_999_968_750_000_5),
    ];
    for (inv_dt, mf, table) in doc {
        let dt: f64 = 1.0 / inv_dt;
        let rho: f64 = mf / (1.0 + dt * dt);
        assert!(
            (rho - table).abs() < 1e-15,
            "dt 1/{inv_dt} mf {mf}: {rho} vs doc {table}"
        );
    }
}

/// First sweep (zero-based, the index `run_sub_iteration` hands to the sweep
/// closure) at which e*e is negative, for the dt-free model
/// `a' = -1 - (m_f/m) a`, `a* = -1/(1 + m_f/m)`, starting from a = 0.
fn first_negative_square(mf: Fix128) -> Option<u32> {
    let a_star = -Fix128::ONE / (Fix128::ONE + mf);
    let mut a = Fix128::ZERO;
    for j in 1..=200u32 {
        a = -Fix128::ONE - mf * a;
        let e = a - a_star;
        if (e * e).is_negative() {
            return Some(j - 1);
        }
    }
    None
}

#[test]
fn documented_wrap_sweeps_for_a_diverging_model_match_the_measurement() {
    // module doc: m_f/m = 1.5 -> 59, 2.0 -> 33, 4.0 -> 16 (zero-based sweep where e*e first goes negative)
    let got = [r(3, 2), r(2, 1), r(4, 1)].map(first_negative_square);
    assert_eq!(
        got,
        [Some(59), Some(33), Some(16)],
        "doc table: 59 / 33 / 16"
    );
}

#[test]
fn two_separate_overshoots_are_not_a_wrap() {
    // grow, shrink, grow, shrink: the growth streak resets on every decrease,
    // so neither shrink is "growth for two sweeps followed by a decrease".
    let seq = [pow2(0), pow2(1), r(3, 2), r(7, 4), r(5, 4)];
    let mut m = ContractionMonitor::new(SubIterationConfig {
        max_sweeps: 100,
        ..good()
    })
    .unwrap();
    let v = feed(&mut m, &seq);
    for (i, x) in v.iter().enumerate() {
        assert!(
            !matches!(
                x,
                MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped { .. })
            ),
            "false wrap at sweep {}",
            i + 1
        );
    }
}

#[test]
fn a_slowly_growing_residual_is_not_reported_as_stagnated() {
    // ratio 1 + 2^-8 per sweep: below the divergence threshold (1 + 1/16), so
    // the verdict must wait for the budget (NotConverged), not Stagnated: the
    // stagnation test only runs while the residual is not growing.
    let cfg = SubIterationConfig {
        max_sweeps: 9,
        ..good()
    };
    let mut m = ContractionMonitor::new(cfg).unwrap();
    let seq: Vec<Fix128> = (0..9).map(|k| Fix128::ONE + r(k, 256)).collect();
    let v = feed(&mut m, &seq);
    for x in v.iter().take(8) {
        assert_eq!(*x, MonitorVerdict::Continue);
    }
    assert_eq!(
        v[8],
        MonitorVerdict::Stop(CoupledIterationError::NotConverged {
            sweeps: 9,
            residual: Fix128::ONE + r(8, 256)
        })
    );
}

#[test]
fn constant_residual_stagnates_at_the_first_sweep_after_the_ratio_window() {
    // best sweep stays 1: stop_at = 1 / (1 - 1/2) = 2, but the stagnation test
    // only starts after `ratio_samples` (4) sweeps, so the first stop is sweep 5.
    let mut m = ContractionMonitor::new(good()).unwrap();
    let v = feed(&mut m, &[Fix128::ONE; 5]);
    for x in v.iter().take(4) {
        assert_eq!(*x, MonitorVerdict::Continue);
    }
    assert_eq!(
        v[4],
        MonitorVerdict::Stop(CoupledIterationError::Stagnated {
            sweeps: 5,
            best_residual: Fix128::ONE,
            first_residual: Fix128::ONE,
            sweeps_since_improvement: 4,
        })
    );
}

#[test]
fn stagnation_point_is_measured_from_the_first_sweep_when_nothing_improves() {
    // fraction 7/8: stop_at = 1 / (1/8) = 8 -> first stop at sweep 9
    let cfg = SubIterationConfig {
        stagnation_fraction: r(7, 8),
        max_sweeps: 100,
        ..good()
    };
    let mut m = ContractionMonitor::new(cfg).unwrap();
    let v = feed(&mut m, &[Fix128::ONE; 9]);
    assert!(v.iter().take(8).all(|x| *x == MonitorVerdict::Continue));
    assert_eq!(
        v[8],
        MonitorVerdict::Stop(CoupledIterationError::Stagnated {
            sweeps: 9,
            best_residual: Fix128::ONE,
            first_residual: Fix128::ONE,
            sweeps_since_improvement: 8,
        })
    );
}
