//! Audit oracles for `linear_elastic_fem`.
//!
//! Every expected value below is a closed form, a hand calculation, or an
//! exhaustive search over a small input set. None of them is read back from the
//! implementation under test.
//!
//! Oracles that are red on the current source carry
//! `#[ignore = "known defect: AUD-A-S1W3-NNN: ..."]`.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    mark_bulk, solve, solve_corotational, AdaptiveConfig, Axis, BoundaryConditions,
    CorotationalConfig, ElasticMaterial, ElastoplasticConfig, ElastoplasticIncrementRequest,
    ElastoplasticProblem, FemError, FemSolution, PlasticHeating, Preconditioner, SolverConfig,
    StressTensor, ThermalSoftening,
};
use alice_physics::linear_elastic_fem::{solve_with_eigenstrain, ThermalExpansion};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// `2^exponent` for `-63 <= exponent <= 62`, built from raw words.
fn pow2(exponent: i32) -> Fix128 {
    if exponent >= 0 {
        Fix128::from_int(1_i64 << exponent)
    } else {
        Fix128::from_raw(0, 1_u64 << (64 + exponent))
    }
}

fn corner_tet() -> SdfTetMesh {
    SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        tets: vec![Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    }
}

// ---------------------------------------------------------------------------
// ElasticMaterial
// ---------------------------------------------------------------------------

/// Doc: `E <= 0`, or `ν` outside the open interval `(-1, 0.5)`, is refused. The
/// endpoints and their nearest accepted neighbours pin both comparisons.
#[test]
fn material_validation_endpoints_are_exact() {
    let nu = fx(0.25);
    assert!(matches!(
        ElasticMaterial::new(Fix128::ZERO, nu),
        Err(FemError::InvalidMaterial(_))
    ));
    assert!(matches!(
        ElasticMaterial::new(-Fix128::ONE, nu),
        Err(FemError::InvalidMaterial(_))
    ));
    assert!(ElasticMaterial::new(Fix128::from_raw(0, 1), nu).is_ok());

    let e = fx(1000.0);
    assert!(matches!(
        ElasticMaterial::new(e, Fix128::NEG_ONE),
        Err(FemError::InvalidMaterial(_))
    ));
    assert!(ElasticMaterial::new(e, Fix128::NEG_ONE + pow2(-20)).is_ok());
    assert!(matches!(
        ElasticMaterial::new(e, fx(0.5)),
        Err(FemError::InvalidMaterial(_))
    ));
    assert!(ElasticMaterial::new(e, fx(0.5) - pow2(-20)).is_ok());
    assert!(matches!(
        ElasticMaterial::new(e, fx(0.75)),
        Err(FemError::InvalidMaterial(_))
    ));
    assert!(matches!(
        ElasticMaterial::new(e, fx(-1.5)),
        Err(FemError::InvalidMaterial(_))
    ));
    // nu = 0 is a legal material (no lateral contraction).
    let zero = ElasticMaterial::new(e, Fix128::ZERO).expect("nu = 0 is inside (-1, 0.5)");
    let (lambda, mu) = zero.lame();
    assert_eq!(lambda, Fix128::ZERO);
    assert_eq!(mu, fx(500.0));
}

/// Doc: `λ = Eν / ((1+ν)(1−2ν))`, `μ = E / (2(1+ν))`. With `ν = 1/4` and
/// `E = 1000` both are exactly 400 (hand calculation: `250 / (1.25 · 0.5)`).
#[test]
fn lame_parameters_match_the_closed_form() {
    let m = ElasticMaterial::new(fx(1000.0), fx(0.25)).expect("valid");
    let (lambda, mu) = m.lame();
    assert_eq!(lambda, fx(400.0));
    assert_eq!(mu, fx(400.0));

    // Steel-like, non-dyadic ratio: compare against the f64 closed form.
    let m = ElasticMaterial::new(fx(200_000.0), fx(0.3)).expect("valid");
    let (lambda, mu) = m.lame();
    let want_lambda = 200_000.0 * 0.3 / (1.3 * 0.4);
    let want_mu = 200_000.0 / 2.6;
    assert!((lambda.to_f64() - want_lambda).abs() <= 1e-9 * want_lambda);
    assert!((mu.to_f64() - want_mu).abs() <= 1e-9 * want_mu);

    // Negative Poisson's ratio: lambda is negative, mu positive.
    let m = ElasticMaterial::new(fx(1000.0), fx(-0.25)).expect("valid");
    let (lambda, mu) = m.lame();
    let want_lambda = 1000.0 * -0.25 / (0.75 * 1.5);
    assert!((lambda.to_f64() - want_lambda).abs() <= 1e-9);
    assert!((mu.to_f64() - 1000.0 / 1.5).abs() <= 1e-9);
}

/// `ElasticMaterial::new` accepts every `ν` in the open interval and `lame()` is
/// documented as the closed form on that interval. Near the ends the closed form
/// exceeds the `Fix128` range (2^63) and the division wraps silently: nothing
/// reports it. Measured 2026-10-04, E = 1e6, ν = 0.5 − 2^-k: k = 40 is correct
/// (1.83e17), k = 50 gives 3.2e18 against a true 1.9e20, k = 58 gives −3.7e18.
#[test]
fn lame_never_returns_a_wrapped_value_inside_the_accepted_interval() {
    let e = 1.0e6;
    for k in [50_i32, 56, 58, 60] {
        let nu = fx(0.5) - pow2(-k);
        if let Ok(m) = ElasticMaterial::new(fx(e), nu) {
            let (lambda, _) = m.lame();
            let want = e * 0.5 / (1.5 * 2.0_f64.powi(-k + 1));
            assert!(
                lambda.to_f64() > 0.0 && ((lambda.to_f64() - want) / want).abs() < 1e-6,
                "k={k}: lambda {:e}, closed form {want:e}",
                lambda.to_f64()
            );
        }
        let nu = Fix128::NEG_ONE + pow2(-k);
        if let Ok(m) = ElasticMaterial::new(fx(e), nu) {
            let (lambda, _) = m.lame();
            // lambda = E nu / ((1+nu)(1-2nu)) with nu near -1: large and negative.
            let want = -e / (2.0_f64.powi(-k) * 3.0);
            assert!(
                lambda.to_f64() < 0.0 && ((lambda.to_f64() - want) / want).abs() < 1e-6,
                "k={k} (nu near -1): lambda {:e}, closed form {want:e}",
                lambda.to_f64()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// StressTensor / FemSolution
// ---------------------------------------------------------------------------

/// Doc: von Mises is `sqrt(1/2[(sxx-syy)^2 + ...] + 3(sxy^2 + ...))`. Pure shear
/// tau gives `sqrt(3) tau` in each of the three planes; hydrostatic gives 0.
#[test]
fn von_mises_and_hydrostatic_closed_forms() {
    let zero = StressTensor::default();
    let tau = fx(10.0);
    for pick in 0..3 {
        let mut s = zero;
        match pick {
            0 => s.xy = tau,
            1 => s.yz = tau,
            _ => s.zx = tau,
        }
        assert!((s.von_mises().to_f64() - 10.0 * 3.0_f64.sqrt()).abs() < 1e-9);
        assert_eq!(s.hydrostatic(), Fix128::ZERO);
    }
    let hydro = StressTensor {
        xx: fx(-6.0),
        yy: fx(-6.0),
        zz: fx(-6.0),
        ..zero
    };
    assert!(hydro.von_mises().to_f64().abs() < 1e-9);
    assert_eq!(hydro.hydrostatic(), fx(-6.0));
    let mixed = StressTensor {
        xx: fx(3.0),
        yy: fx(6.0),
        zz: fx(9.0),
        ..zero
    };
    assert_eq!(mixed.hydrostatic(), fx(6.0));
    // 1/2 (9 + 9 + 36) = 27, so sqrt(27).
    assert!((mixed.von_mises().to_f64() - 27.0_f64.sqrt()).abs() < 1e-9);
    let neg = StressTensor {
        xx: fx(-3.0),
        yy: fx(-6.0),
        zz: fx(-9.0),
        ..zero
    };
    assert_eq!(neg.von_mises(), mixed.von_mises());
    assert_eq!(neg.hydrostatic(), fx(-6.0));
}

/// Doc: `FemSolution::max_von_mises_mpa` is the largest over all elements, and
/// zero for a solution with no elements.
#[test]
fn max_von_mises_is_the_largest_and_zero_when_empty() {
    let mk = |stresses: Vec<StressTensor>| FemSolution {
        displacements: Vec::new(),
        element_stress: stresses,
        iterations: 0,
        relative_residual: Fix128::ZERO,
        effective_relative_tolerance: Fix128::ZERO,
    };
    let uniaxial = |s: f64| StressTensor {
        xx: fx(s),
        ..StressTensor::default()
    };
    // The largest is neither first nor last, and one entry is negative.
    let sol = mk(vec![uniaxial(5.0), uniaxial(-40.0), uniaxial(12.0)]);
    assert!((sol.max_von_mises_mpa().to_f64() - 40.0).abs() < 1e-9);
    assert_eq!(mk(Vec::new()).max_von_mises_mpa(), Fix128::ZERO);
    let only_zero = mk(vec![StressTensor::default()]);
    assert_eq!(only_zero.max_von_mises_mpa(), Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// SolverConfig
// ---------------------------------------------------------------------------

/// The documented defaults: 10,000 iterations, relative residual 2^-30, window
/// `max(500, 0.5 x iterations)`, improvement 2^-10, no preconditioner. The
/// default stagnation fraction must be below 1 (at 1 the rule can never fire),
/// which the `with_stagnation_fraction` doc says no test pins.
#[test]
fn solver_config_defaults_are_the_documented_ones() {
    let c = SolverConfig::default();
    assert_eq!(c.max_iterations(), 10_000);
    assert_eq!(c.relative_tolerance(), pow2(-30));
    assert_eq!(c.stagnation_min_window(), 500);
    assert_eq!(c.stagnation_window_fraction(), fx(0.5));
    assert_eq!(c.stagnation_min_improvement(), pow2(-10));
    assert_eq!(c.preconditioner(), Preconditioner::None);
    assert!(c.stagnation_window_fraction() > Fix128::ZERO);
    assert!(c.stagnation_window_fraction() < Fix128::ONE);
}

/// Every refusal boundary of `SolverConfig`, one step either side.
#[test]
fn solver_config_validation_endpoints() {
    let tol = pow2(-20);
    assert!(SolverConfig::try_new(0, tol).is_err());
    assert!(SolverConfig::try_new(1, tol).is_ok());
    assert!(SolverConfig::try_new(5, Fix128::ZERO).is_err());
    assert!(SolverConfig::try_new(5, Fix128::from_raw(0, 1)).is_ok());
    assert!(SolverConfig::try_new(5, Fix128::ONE).is_err());
    assert!(SolverConfig::try_new(5, Fix128::ONE - Fix128::from_raw(0, 1)).is_ok());
    assert!(SolverConfig::try_new(5, -tol).is_err());
    let ok = SolverConfig::try_new(7, tol).expect("valid");
    assert_eq!(ok.max_iterations(), 7);
    assert_eq!(ok.relative_tolerance(), tol);
    // try_new keeps the defaults it does not take.
    assert_eq!(ok.stagnation_min_window(), 500);

    let c = SolverConfig::default();
    assert!(c.with_stagnation(0, fx(0.25)).is_err());
    assert!(c.with_stagnation(1, Fix128::ZERO).is_err());
    assert!(c.with_stagnation(1, Fix128::ONE).is_err());
    assert!(c.with_stagnation(1, -fx(0.25)).is_err());
    let s = c.with_stagnation(3, fx(0.25)).expect("valid");
    assert_eq!(s.stagnation_min_window(), 3);
    assert_eq!(s.stagnation_min_improvement(), fx(0.25));
    assert_eq!(s.max_iterations(), c.max_iterations());

    assert!(c.with_stagnation_fraction(Fix128::ZERO).is_err());
    assert!(c.with_stagnation_fraction(Fix128::ONE).is_err());
    assert!(c.with_stagnation_fraction(fx(1.5)).is_err());
    assert!(c.with_stagnation_fraction(-fx(0.5)).is_err());
    let f = c.with_stagnation_fraction(fx(0.75)).expect("valid");
    assert_eq!(f.stagnation_window_fraction(), fx(0.75));
    assert_eq!(
        f.with_preconditioner(Preconditioner::JacobiScaled)
            .preconditioner(),
        Preconditioner::JacobiScaled
    );
}

// ---------------------------------------------------------------------------
// BoundaryConditions
// ---------------------------------------------------------------------------

/// Doc: a later `prescribe` of the same `(vertex, axis)` replaces the earlier
/// value; `add_load` accumulates; `prescribed()` / `loads()` keep insertion
/// order; counts count distinct degrees of freedom.
#[test]
fn boundary_conditions_replace_accumulate_and_keep_order() {
    let mut bc = BoundaryConditions::new();
    bc.prescribe(2, Axis::Y, fx(1.0));
    bc.prescribe(1, Axis::X, fx(2.0));
    bc.prescribe(2, Axis::Y, fx(3.0));
    bc.prescribe(2, Axis::X, fx(4.0));
    assert_eq!(bc.prescribed_count(), 3);
    assert_eq!(
        bc.prescribed(),
        &[
            (2, Axis::Y, fx(3.0)),
            (1, Axis::X, fx(2.0)),
            (2, Axis::X, fx(4.0))
        ]
    );
    bc.add_load(5, Axis::Z, fx(1.5));
    bc.add_load(4, Axis::Z, fx(1.0));
    bc.add_load(5, Axis::Z, fx(2.0));
    bc.add_load(5, Axis::Y, fx(-1.0));
    assert_eq!(bc.load_count(), 3);
    assert_eq!(
        bc.loads(),
        &[
            (5, Axis::Z, fx(3.5)),
            (4, Axis::Z, fx(1.0)),
            (5, Axis::Y, fx(-1.0))
        ]
    );
    let mut all = BoundaryConditions::new();
    all.prescribe_all(7, [fx(1.0), fx(2.0), fx(3.0)]);
    assert_eq!(
        all.prescribed(),
        &[
            (7, Axis::X, fx(1.0)),
            (7, Axis::Y, fx(2.0)),
            (7, Axis::Z, fx(3.0))
        ]
    );
    let mut fixed = BoundaryConditions::new();
    fixed.fix(3);
    assert_eq!(fixed.prescribed_count(), 3);
    assert!(fixed.prescribed().iter().all(|p| p.2 == Fix128::ZERO));
    assert_eq!(Axis::ALL, [Axis::X, Axis::Y, Axis::Z]);
    assert_eq!(
        [Axis::X.index(), Axis::Y.index(), Axis::Z.index()],
        [0, 1, 2]
    );
}

/// Doc: a load on a prescribed degree of freedom is ignored by the solve.
#[test]
fn a_load_on_a_prescribed_dof_does_not_change_the_solution() {
    let mesh = corner_tet();
    let material = ElasticMaterial::new(fx(1000.0), fx(0.25)).expect("valid");
    let mut base = BoundaryConditions::new();
    base.fix(0);
    base.prescribe(1, Axis::Y, Fix128::ZERO);
    base.prescribe(1, Axis::Z, Fix128::ZERO);
    base.prescribe(2, Axis::Z, Fix128::ZERO);
    base.add_load(3, Axis::X, fx(0.5));
    let mut with_extra = base.clone();
    with_extra.add_load(0, Axis::X, fx(100.0)); // vertex 0 is fixed
    with_extra.add_load(1, Axis::Y, fx(-7.0)); // prescribed to zero
    let a = solve(&mesh, &material, &base, &SolverConfig::default()).expect("well posed");
    let b = solve(&mesh, &material, &with_extra, &SolverConfig::default()).expect("well posed");
    assert_eq!(a.displacements, b.displacements);
    assert_eq!(a.element_stress, b.element_stress);
    // The load that is not ignored does move the tip, in its own direction.
    let moved = a.displacements[3][0].to_f64();
    assert!(
        moved > 0.0,
        "a +x load must move vertex 3 toward +x: {moved}"
    );
}

// ---------------------------------------------------------------------------
// mark_bulk
// ---------------------------------------------------------------------------

/// Doc: the **smallest** set carrying at least `θ · total`, taken greedily from
/// the largest indicator with ties broken by index ascending. Expected sets come
/// from an exhaustive search over all subsets in exact integer arithmetic
/// (indicators are `n/8`, `θ = k/8`).
#[test]
fn mark_bulk_is_the_minimal_set_on_every_small_input() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..400 {
        let len = 1 + (next() % 7) as usize;
        let ints: Vec<u64> = (0..len).map(|_| next() % 6).collect();
        let ind: Vec<Fix128> = ints.iter().map(|&n| Fix128::from_raw(0, n << 61)).collect();
        let total: u64 = ints.iter().sum();
        for k in 1..=8_u64 {
            let theta = if k == 8 {
                Fix128::ONE
            } else {
                Fix128::from_raw(0, k << 61)
            };
            let got = mark_bulk(&ind, theta).expect("valid input");
            assert_eq!(got.len(), len);
            if total == 0 {
                assert!(got.iter().all(|m| !m), "total 0 marks nothing: {ints:?}");
                continue;
            }
            // Exhaustive minimal cardinality with sum * 8 >= k * total.
            let mut best = usize::MAX;
            for mask in 0_u32..(1 << len) {
                let sum: u64 = (0..len)
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| ints[i])
                    .sum();
                if sum * 8 >= k * total {
                    best = best.min(mask.count_ones() as usize);
                }
            }
            let count = got.iter().filter(|m| **m).count();
            assert_eq!(count, best, "ints {ints:?} theta {k}/8");
            let carried: u64 = got
                .iter()
                .zip(&ints)
                .filter(|(m, _)| **m)
                .map(|(_, n)| *n)
                .sum();
            assert!(carried * 8 >= k * total, "ints {ints:?} theta {k}/8");
            // Greedy order: every marked element ranks above every unmarked one
            // under (value descending, index ascending).
            for i in 0..len {
                for j in 0..len {
                    if got[i] && !got[j] {
                        assert!(
                            ints[i] > ints[j] || (ints[i] == ints[j] && i < j),
                            "ints {ints:?} theta {k}/8: marked {i} unmarked {j}"
                        );
                    }
                }
            }
        }
    }
}

/// Domain refusals of `mark_bulk`.
#[test]
fn mark_bulk_refuses_what_its_doc_says() {
    let one = [Fix128::ONE];
    assert!(matches!(mark_bulk(&[], fx(0.5)), Err(FemError::EmptyMesh)));
    assert!(matches!(
        mark_bulk(&one, Fix128::ZERO),
        Err(FemError::InvalidConfig(_))
    ));
    assert!(matches!(
        mark_bulk(&one, Fix128::ONE + pow2(-20)),
        Err(FemError::InvalidConfig(_))
    ));
    assert!(mark_bulk(&one, Fix128::ONE).is_ok());
    assert!(matches!(
        mark_bulk(&[Fix128::ONE, -pow2(-30)], fx(0.5)),
        Err(FemError::InvalidConfig(_))
    ));
    // theta = 1 marks every nonzero indicator and nothing else.
    let got = mark_bulk(&[fx(1.0), Fix128::ZERO, fx(2.0)], Fix128::ONE).expect("valid");
    assert_eq!(got, vec![true, false, true]);
}

/// Doc: the marked set carries **at least** `θ · total`. `total * θ` is formed
/// in `Fix128` and truncated, so for indicators at the scale of one `Fix128`
/// step the carried amount can fall one step short of the exact product.
/// Measured: indicators `[5, 5]` ulp, `θ = fx(0.55)`: the exact target is
/// `10 θ = 5.5` ulp, the truncated target is 5, so one element (50 %) is marked
/// where 55 % was requested.
#[test]
// AUD-A-S1W3-002
fn mark_bulk_carries_at_least_theta_times_total_even_at_ulp_scale() {
    let ind = [Fix128::from_raw(0, 5), Fix128::from_raw(0, 5)];
    let theta = fx(0.55);
    let got = mark_bulk(&ind, theta).expect("valid");
    let carried_ulps = got.iter().filter(|m| **m).count() as u128 * 5;
    // Exact: carried / 10 >= theta  <=>  carried * 2^64 >= 10 * theta_raw
    // (theta < 1, so theta_raw is the low word).
    assert_eq!(theta.hi, 0);
    assert!(
        carried_ulps << 64 >= 10 * u128::from(theta.lo),
        "carried {carried_ulps}/10 ulp, requested theta {}",
        theta.to_f64()
    );
}

// ---------------------------------------------------------------------------
// config getters and refusals that no other test reads back
// ---------------------------------------------------------------------------

#[test]
fn adaptive_and_corotational_configs_report_what_they_were_given() {
    let lin = SolverConfig::try_new(77, pow2(-12)).expect("valid");
    let a = AdaptiveConfig::try_new(lin, fx(0.5), 4, 9).expect("valid");
    assert_eq!(a.linear(), lin);
    assert_eq!(a.bulk_fraction(), fx(0.5));
    assert_eq!(a.max_rounds(), 4);
    assert_eq!(a.max_refine_passes(), 9);
    assert!(AdaptiveConfig::try_new(lin, fx(0.5), 0, 9).is_err());
    assert!(AdaptiveConfig::try_new(lin, fx(0.5), 1, 0).is_err());
    assert!(AdaptiveConfig::try_new(lin, Fix128::ZERO, 1, 1).is_err());
    assert!(AdaptiveConfig::try_new(lin, Fix128::ONE, 1, 1).is_ok());
    assert!(AdaptiveConfig::try_new(lin, Fix128::ONE + pow2(-20), 1, 1).is_err());

    let c = CorotationalConfig::try_new(lin, 11, pow2(-18), 5, 13).expect("valid");
    assert_eq!(c.linear(), lin);
    assert_eq!(c.newton_iterations(), 11);
    assert_eq!(c.newton_tolerance(), pow2(-18));
    assert_eq!(c.increments(), 5);
    assert_eq!(c.polar_iterations(), 13);
    assert!(c.hyperelastic().is_none());
    assert!(!c.consistent_tangent());
    assert!(c.with_consistent_tangent().consistent_tangent());
    assert!(CorotationalConfig::try_new(lin, 0, pow2(-18), 5, 13).is_err());
    assert!(CorotationalConfig::try_new(lin, 1, Fix128::ZERO, 5, 13).is_err());
    assert!(CorotationalConfig::try_new(lin, 1, Fix128::ONE, 5, 13).is_err());
    assert!(CorotationalConfig::try_new(lin, 1, pow2(-18), 0, 13).is_err());
    assert!(CorotationalConfig::try_new(lin, 1, pow2(-18), 5, 0).is_err());
}

/// Doc: `β` in `[0, 1]` (0 accepted, 1 accepted), `c_v > 0`, `ΔT = β W / c_v`.
#[test]
fn plastic_heating_validation_and_closed_form() {
    assert!(PlasticHeating::try_new(Fix128::ZERO, fx(4.0)).is_ok());
    assert!(PlasticHeating::try_new(Fix128::ONE, fx(4.0)).is_ok());
    assert!(PlasticHeating::try_new(-pow2(-20), fx(4.0)).is_err());
    assert!(PlasticHeating::try_new(Fix128::ONE + pow2(-20), fx(4.0)).is_err());
    assert!(PlasticHeating::try_new(fx(0.5), Fix128::ZERO).is_err());
    assert!(PlasticHeating::try_new(fx(0.5), -fx(4.0)).is_err());
    assert!(PlasticHeating::try_new(fx(0.5), Fix128::from_raw(0, 1)).is_ok());
    let h = PlasticHeating::try_new(fx(0.75), fx(4.0)).expect("valid");
    assert_eq!(h.taylor_quinney(), fx(0.75));
    assert_eq!(h.volumetric_heat_capacity_mpa_per_k(), fx(4.0));
    // 0.75 * 8 / 4 = 1.5 exactly.
    assert_eq!(h.temperature_rise(fx(8.0)), fx(1.5));
    let cold = PlasticHeating::try_new(Fix128::ZERO, fx(4.0)).expect("valid");
    assert_eq!(cold.temperature_rise(fx(8.0)), Fix128::ZERO);
}

/// Doc: yield stress in `(0, 2^30]`, hardening in `[0, 2^30]`, tolerance in
/// `(0, 1)`, budget positive.
#[test]
fn elastoplastic_config_validation_endpoints() {
    let lin = SolverConfig::default();
    let tol = pow2(-20);
    let cap = Fix128::from_int(1 << 30);
    let build =
        |n: u32, t: Fix128, y: Fix128, h: Fix128| ElastoplasticConfig::try_new(lin, n, t, y, h);
    assert!(build(1, tol, fx(250.0), fx(10.0)).is_ok());
    assert!(build(0, tol, fx(250.0), fx(10.0)).is_err());
    assert!(build(1, Fix128::ZERO, fx(250.0), fx(10.0)).is_err());
    assert!(build(1, Fix128::ONE, fx(250.0), fx(10.0)).is_err());
    assert!(build(1, tol, Fix128::ZERO, fx(10.0)).is_err());
    assert!(build(1, tol, -fx(1.0), fx(10.0)).is_err());
    assert!(build(1, tol, cap, fx(10.0)).is_ok());
    assert!(build(1, tol, cap + Fix128::ONE, fx(10.0)).is_err());
    assert!(build(1, tol, fx(250.0), Fix128::ZERO).is_ok());
    assert!(build(1, tol, fx(250.0), -pow2(-20)).is_err());
    assert!(build(1, tol, fx(250.0), cap).is_ok());
    assert!(build(1, tol, fx(250.0), cap + Fix128::ONE).is_err());
}

/// Doc: `ThermalSoftening` fractions are non-negative, at most 2^30; `none()` is
/// the zero law.
#[test]
fn thermal_softening_validation_and_getters() {
    let cap = Fix128::from_int(1 << 30);
    assert!(ThermalSoftening::try_new(Fix128::ZERO, Fix128::ZERO).is_ok());
    assert!(ThermalSoftening::try_new(-pow2(-20), Fix128::ZERO).is_err());
    assert!(ThermalSoftening::try_new(Fix128::ZERO, -pow2(-20)).is_err());
    assert!(ThermalSoftening::try_new(cap, cap).is_ok());
    assert!(ThermalSoftening::try_new(cap + Fix128::ONE, Fix128::ZERO).is_err());
    assert!(ThermalSoftening::try_new(Fix128::ZERO, cap + Fix128::ONE).is_err());
    let s = ThermalSoftening::try_new(fx(0.25), fx(0.5)).expect("valid");
    assert_eq!(s.yield_per_k(), fx(0.25));
    assert_eq!(s.hardening_per_k(), fx(0.5));
    let none = ThermalSoftening::none();
    assert_eq!(none.yield_per_k(), Fix128::ZERO);
    assert_eq!(none.hardening_per_k(), Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// solve / solve_with_eigenstrain consistency, element orientation
// ---------------------------------------------------------------------------

fn unit_cube_kuhn() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                mesh.vertices.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    let idx = |i: usize, j: usize, k: usize| (i + 2 * j + 4 * k) as u32;
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for path in PATHS {
        let mut step = [0usize; 3];
        let mut corners = [idx(0, 0, 0); 4];
        for (n, axis) in path.into_iter().enumerate() {
            step[axis] = 1;
            corners[n + 1] = idx(step[0], step[1], step[2]);
        }
        mesh.tets.push(Tetrahedron { vertices: corners });
    }
    mesh
}

fn cube_bc() -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for v in 0..4_u32 {
        bc.fix(v); // the z = 0 face
    }
    bc.add_load(7, Axis::X, fx(0.25));
    bc.add_load(7, Axis::Z, fx(-0.5));
    bc
}

/// Doc: `solve` is `solve_with_eigenstrain` with no eigenstrain, "the same
/// system as a uniform `ΔT = 0`", and bit-identical. Compare the displacement,
/// the stress and the iteration count of the two paths.
#[test]
fn solve_equals_the_eigenstrain_solve_at_zero_rise_bit_for_bit() {
    let mesh = unit_cube_kuhn();
    let material = ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("valid");
    let bc = cube_bc();
    let cfg = SolverConfig::default();
    let plain = solve(&mesh, &material, &bc, &cfg).expect("well posed");
    let field = CoupledField::try_new(
        3,
        3,
        3,
        (fx(-0.5), fx(-0.5), fx(-0.5)),
        (fx(1.5), fx(1.5), fx(1.5)),
    )
    .expect("grid");
    let rise = TemperatureRise::from_absolute(&field, Fix128::ZERO);
    let with = solve_with_eigenstrain(
        &mesh,
        &material,
        &bc,
        &cfg,
        Some(ThermalExpansion::from_rise(&rise, fx(1.0e-3))),
    )
    .expect("well posed");
    assert_eq!(plain.displacements, with.displacements);
    assert_eq!(plain.element_stress, with.element_stress);
    assert_eq!(plain.iterations, with.iterations);
}

/// `volume = |det| / 6` and the gradients use the signed determinant, so
/// reversing the vertex order of a tetrahedron (which flips the orientation)
/// describes the same body and must give the same field. Compared to 1e-12,
/// because the reordering changes the summation order inside the element.
#[test]
fn reversing_the_orientation_of_some_elements_does_not_change_the_answer() {
    let material = ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("valid");
    let bc = cube_bc();
    let cfg = SolverConfig::default();
    let base_mesh = unit_cube_kuhn();
    let mut flipped = base_mesh.clone();
    for t in [1_usize, 4] {
        flipped.tets[t].vertices.swap(0, 1);
    }
    let a = solve(&base_mesh, &material, &bc, &cfg).expect("well posed");
    let b = solve(&flipped, &material, &bc, &cfg).expect("well posed");
    for (da, db) in a.displacements.iter().zip(&b.displacements) {
        for axis in 0..3 {
            assert!((da[axis].to_f64() - db[axis].to_f64()).abs() < 1e-12);
        }
    }
    for (sa, sb) in a.element_stress.iter().zip(&b.element_stress) {
        for (x, y) in [
            (sa.xx, sb.xx),
            (sa.yy, sb.yy),
            (sa.zz, sb.zz),
            (sa.xy, sb.xy),
            (sa.yz, sb.yz),
            (sa.zx, sb.zx),
        ] {
            assert!((x.to_f64() - y.to_f64()).abs() < 1e-9);
        }
    }
}

/// Clapeyron's theorem: for a load-driven solve with every prescribed
/// displacement zero, `Σ f·u = Σ_e V_e σ_e : ε_e`. The right side is rebuilt
/// here from the mesh coordinates in `f64` with the inverse of the edge matrix
/// (cofactor form), a different route from the module's `cross / det`
/// gradients, and from the reported displacement and stress.
#[test]
fn clapeyron_theorem_ties_load_work_to_stress_work() {
    let mesh = unit_cube_kuhn();
    let material = ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("valid");
    let bc = cube_bc();
    let sol = solve(&mesh, &material, &bc, &SolverConfig::default()).expect("well posed");

    let mut load_work = 0.0_f64;
    for &(v, axis, f) in bc.loads() {
        load_work += f.to_f64() * sol.displacements[v as usize][axis.index()].to_f64();
    }
    assert!(load_work > 0.0, "work done by the loads must be positive");

    let mut stress_work = 0.0_f64;
    for (t, tet) in mesh.tets.iter().enumerate() {
        let p: Vec<[f64; 3]> = tet
            .vertices
            .iter()
            .map(|&v| mesh.vertices[v as usize].map(f64::from))
            .collect();
        let col = |i: usize| [p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]];
        let (a, b, c) = (col(1), col(2), col(3));
        // J = [a b c] as columns; J^-1 by cofactors.
        let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
            + c[0] * (a[1] * b[2] - a[2] * b[1]);
        let inv = [
            [
                (b[1] * c[2] - b[2] * c[1]) / det,
                (b[2] * c[0] - b[0] * c[2]) / det,
                (b[0] * c[1] - b[1] * c[0]) / det,
            ],
            [
                (a[2] * c[1] - a[1] * c[2]) / det,
                (a[0] * c[2] - a[2] * c[0]) / det,
                (a[1] * c[0] - a[0] * c[1]) / det,
            ],
            [
                (a[1] * b[2] - a[2] * b[1]) / det,
                (a[2] * b[0] - a[0] * b[2]) / det,
                (a[0] * b[1] - a[1] * b[0]) / det,
            ],
        ];
        // grad N_i (i = 1..3) = row i-1 of J^-1; grad N_0 = -(sum).
        let mut grad = [[0.0_f64; 3]; 4];
        grad[1..4].copy_from_slice(&inv);
        for k in 0..3 {
            grad[0][k] = -(grad[1][k] + grad[2][k] + grad[3][k]);
        }
        let mut du = [[0.0_f64; 3]; 3]; // du[a][b] = d u_a / d x_b
        for (i, &v) in tet.vertices.iter().enumerate() {
            let u = sol.displacements[v as usize].map(Fix128::to_f64);
            for ax in 0..3 {
                for bx in 0..3 {
                    du[ax][bx] += u[ax] * grad[i][bx];
                }
            }
        }
        let s = &sol.element_stress[t];
        let eps = |i: usize, j: usize| 0.5 * (du[i][j] + du[j][i]);
        let doubled = |x: Fix128| 2.0 * x.to_f64();
        let work = s.xx.to_f64() * eps(0, 0)
            + s.yy.to_f64() * eps(1, 1)
            + s.zz.to_f64() * eps(2, 2)
            + doubled(s.xy) * eps(0, 1)
            + doubled(s.yz) * eps(1, 2)
            + doubled(s.zx) * eps(0, 2);
        stress_work += det.abs() / 6.0 * work;
    }
    assert!(
        (stress_work - load_work).abs() <= 1e-9 * load_work,
        "sum f.u = {load_work:.12e}, sum V sigma:eps = {stress_work:.12e}"
    );
}

/// Thermal eigenstress is sampled at the element centroid. A fully clamped
/// tetrahedron carries `σ = −(3λ+2μ) α ΔT(centroid) I` exactly. The field is
/// linear with a different slope per axis (100 x + 10 y + z), so a centroid with
/// any wrong coordinate gives a different number. Centroid of the corner tet is
/// (1/4, 1/4, 1/4): ΔT = 25 + 2.5 + 0.25 = 27.75 K.
#[test]
fn thermal_eigenstress_reads_the_field_at_the_centroid_on_every_axis() {
    let mesh = corner_tet();
    let (e, nu, alpha) = (3500.0, 0.25, 1.0e-3);
    let material = ElasticMaterial::new(fx(e), fx(nu)).expect("valid");
    let mut bc = BoundaryConditions::new();
    for v in 0..4 {
        bc.fix(v);
    }
    // Grid nodes at -0.5, 0.5, 1.5 on each axis; the field is linear, so the
    // trilinear interpolation is exact.
    let mut field = CoupledField::try_new(
        3,
        3,
        3,
        (fx(-0.5), fx(-0.5), fx(-0.5)),
        (fx(1.5), fx(1.5), fx(1.5)),
    )
    .expect("grid");
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                let (x, y, z) = (ix as f64 - 0.5, iy as f64 - 0.5, iz as f64 - 0.5);
                field.set(ix, iy, iz, fx(100.0 * x + 10.0 * y + z));
            }
        }
    }
    let rise = TemperatureRise::from_absolute(&field, Fix128::ZERO);
    let sol = solve_with_eigenstrain(
        &mesh,
        &material,
        &bc,
        &SolverConfig::default(),
        Some(ThermalExpansion::from_rise(&rise, fx(alpha))),
    )
    .expect("well posed");
    // lambda = 3500*0.25/(1.25*0.5) = 1400, mu = 1400, 3 lambda + 2 mu = 7000.
    let want = -7000.0 * alpha * 27.75;
    let s = &sol.element_stress[0];
    for (name, v) in [("xx", s.xx), ("yy", s.yy), ("zz", s.zz)] {
        assert!(
            (v.to_f64() - want).abs() < 1e-6,
            "sigma_{name} = {}, closed form {want}",
            v.to_f64()
        );
    }
    for v in [s.xy, s.yz, s.zx] {
        assert!(v.to_f64().abs() < 1e-9);
    }
}

/// Doc: a solve with fewer than six constrained degrees of freedom is refused
/// as `UnderConstrained`. With no load the conjugate gradient never runs, so the
/// count itself is the only thing that can refuse five; six (a valid minimal
/// support) is accepted.
#[test]
fn five_constrained_dofs_are_refused_and_six_are_accepted() {
    let mesh = corner_tet();
    let material = ElasticMaterial::new(fx(1000.0), fx(0.25)).expect("valid");
    let mut five = BoundaryConditions::new();
    five.fix(0);
    five.prescribe(1, Axis::Y, Fix128::ZERO);
    five.prescribe(1, Axis::Z, Fix128::ZERO);
    assert_eq!(five.prescribed_count(), 5);
    assert!(matches!(
        solve(&mesh, &material, &five, &SolverConfig::default()),
        Err(FemError::UnderConstrained)
    ));
    let lin = SolverConfig::default();
    let cfg = CorotationalConfig::try_new(lin, 5, pow2(-20), 1, 20).expect("valid");
    assert!(matches!(
        solve_corotational(&mesh, &material, &five, &cfg),
        Err(FemError::UnderConstrained)
    ));
    let mut six = five.clone();
    six.prescribe(2, Axis::Z, Fix128::ZERO);
    assert!(solve(&mesh, &material, &six, &SolverConfig::default()).is_ok());
    assert!(solve_corotational(&mesh, &material, &six, &cfg).is_ok());
}

/// A fully clamped, purely elastic elastoplastic step under a uniform rise:
/// `σ = −(3λ+2μ) α ΔT I` with **no shear**. The eigenstrain is isotropic, so it
/// must reach the three normal strains only; reading it into the engineering
/// shears as well would give `σ_xy = −μ α ΔT`.
#[test]
fn elastoplastic_thermal_eigenstrain_touches_normal_strains_only() {
    let mesh = corner_tet();
    let (e, nu, alpha, rise_k) = (3500.0, 0.25, 1.0e-3, 40.0);
    let material = ElasticMaterial::new(fx(e), fx(nu)).expect("valid");
    let mut bc = BoundaryConditions::new();
    for v in 0..4 {
        bc.fix(v);
    }
    let config = ElastoplasticConfig::try_new(
        SolverConfig::default(),
        20,
        pow2(-20),
        fx(1.0e6),
        Fix128::ZERO,
    )
    .expect("valid");
    let field = CoupledField::try_new_filled(
        2,
        2,
        2,
        (fx(-0.5), fx(-0.5), fx(-0.5)),
        (fx(1.5), fx(1.5), fx(1.5)),
        fx(rise_k),
    )
    .expect("grid");
    let rise = TemperatureRise::from_absolute(&field, Fix128::ZERO);
    let problem = ElastoplasticProblem::try_new(&mesh, &material, &bc, &config).expect("prepares");
    let state = problem.virgin_state();
    let request = ElastoplasticIncrementRequest::new(Fix128::ONE)
        .with_thermal(ThermalExpansion::from_rise(&rise, fx(alpha)), None);
    let inc = problem.step(&state, &request).expect("solves");
    // lambda = mu = 1400, 3 lambda + 2 mu = 7000.
    let want = -7000.0 * alpha * rise_k;
    let s = &inc.field.element_stress[0];
    for v in [s.xx, s.yy, s.zz] {
        assert!((v.to_f64() - want).abs() < 1e-6, "{} vs {want}", v.to_f64());
    }
    for v in [s.xy, s.yz, s.zx] {
        assert!(v.to_f64().abs() < 1e-6, "shear stress {}", v.to_f64());
    }
}

/// The stagnation rule fires when `iterations since the last improvement` reaches
/// the window, not one more. Improvement is demanded at 99.9 % per iteration so
/// that no step after the first counts, the window is the floor of 3 (the
/// fraction is made negligible), and the report must say 3.
#[test]
fn stagnation_fires_at_the_window_not_one_after() {
    let mesh = unit_cube_kuhn();
    let material = ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("valid");
    let bc = cube_bc();
    let cfg = SolverConfig::try_new(10_000, pow2(-60))
        .expect("valid")
        .with_stagnation(3, fx(0.999))
        .expect("valid")
        .with_stagnation_fraction(pow2(-40))
        .expect("valid");
    match solve(&mesh, &material, &bc, &cfg) {
        Err(FemError::Stagnated {
            without_improvement,
            ..
        }) => assert_eq!(without_improvement, 3),
        other => panic!("expected Stagnated, got {:?}", other.map(|s| s.iterations)),
    }
}

/// A sliver tetrahedron `(0,0,0) (1,0,0) (0,1,0) (0,0,h)` with its base fully
/// fixed and a load at the apex is well posed for every `h > 0`; only an exactly
/// zero determinant is `DegenerateElement`. Measured 2026-10-04: `h = 1e-6`
/// solves, `h = 1e-8` and `h = 1e-10` return `UnderConstrained` although nine
/// degrees of freedom are fixed (the gradients are `~1/h`, their squares leave
/// the `Fix128` range and the curvature test reads a non-positive value).
#[test]
fn a_sliver_with_a_fixed_base_is_not_reported_as_under_constrained() {
    for h in [1.0e-6_f32, 1.0e-8, 1.0e-10] {
        let mut mesh = corner_tet();
        mesh.vertices[3] = [0.0, 0.0, h];
        let material = ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("valid");
        let mut bc = BoundaryConditions::new();
        for v in 0..3 {
            bc.fix(v);
        }
        bc.add_load(3, Axis::Z, fx(1.0));
        let r = solve(&mesh, &material, &bc, &SolverConfig::default());
        assert!(
            !matches!(r, Err(FemError::UnderConstrained)),
            "h = {h:e}: base fixed (9 dofs) yet UnderConstrained"
        );
    }
}

/// When `θ · total` is exact the target is not raised: indicators `[4, 4]` ulp at
/// `θ = 1/2` need exactly 4 ulp, one element. `θ = 1` marks everything, and a
/// physical-scale total (`2^40` with `θ = 0.55`) still marks the minimal set.
#[test]
fn mark_bulk_exact_targets_are_not_rounded_up() {
    let four = Fix128::from_raw(0, 4);
    let got = mark_bulk(&[four, four], fx(0.5)).expect("valid");
    assert_eq!(got.iter().filter(|m| **m).count(), 1);
    let all = mark_bulk(&[four, four, four], Fix128::ONE).expect("valid");
    assert!(all.iter().all(|m| *m));
    let big = Fix128::from_int(1 << 40);
    let got = mark_bulk(&[big, big], fx(0.55)).expect("valid");
    assert_eq!(got.iter().filter(|m| **m).count(), 2);
    let got = mark_bulk(&[big, big], fx(0.5)).expect("valid");
    assert_eq!(got.iter().filter(|m| **m).count(), 1);
}
