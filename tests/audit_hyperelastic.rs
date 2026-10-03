//! Audit oracle for `hyperelastic`. Expected values: hand closed forms, an
//! independent f64 energy derivative, and textbook simple-shear / uniaxial results.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::hyperelastic::{
    cauchy_stress, small_strain_moduli, small_strain_shear_modulus, strain_energy_density,
    uniaxial_cauchy_stress, volumetric_modulus, HyperelasticModel, Stretch,
};
use alice_physics::math::{Fix128, Mat3Fix, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn mat(r: [[f64; 3]; 3]) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(fx(r[0][0]), fx(r[1][0]), fx(r[2][0])),
        Vec3Fix::new(fx(r[0][1]), fx(r[1][1]), fx(r[2][1])),
        Vec3Fix::new(fx(r[0][2]), fx(r[1][2]), fx(r[2][2])),
    )
}
fn at(m: &Mat3Fix, i: usize, j: usize) -> f64 {
    let c = [m.col0, m.col1, m.col2][j];
    [c.x, c.y, c.z][i].to_f64()
}

fn models() -> [HyperelasticModel; 5] {
    [
        HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) },
        HyperelasticModel::MooneyRivlin {
            c1_mpa: fx(0.3),
            c2_mpa: fx(0.12),
        },
        HyperelasticModel::Yeoh {
            c1_mpa: fx(0.4),
            c2_mpa: fx(-0.02),
            c3_mpa: fx(0.003),
        },
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::natural_rubber(),
    ]
}

/// Independent f64 energy for a principal-stretch triple.
fn w_f64(m: &HyperelasticModel, l: [f64; 3]) -> f64 {
    let i1: f64 = l.iter().map(|x| x * x).sum();
    let i2: f64 = l.iter().map(|x| 1.0 / (x * x)).sum();
    match *m {
        HyperelasticModel::NeoHookean { mu_mpa } => mu_mpa.to_f64() / 2.0 * (i1 - 3.0),
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            c1_mpa.to_f64() * (i1 - 3.0) + c2_mpa.to_f64() * (i2 - 3.0)
        }
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let d = i1 - 3.0;
            c1_mpa.to_f64() * d + c2_mpa.to_f64() * d * d + c3_mpa.to_f64() * d * d * d
        }
    }
}

// ---------------- Stretch ----------------

#[test]
fn uniaxial_and_equibiaxial_stretch_closed_forms() {
    for l in [0.5, 1.0, 2.0, 3.7] {
        let s = Stretch::uniaxial(fx(l));
        assert_eq!(s.l1, fx(l));
        assert_eq!(s.l2, s.l3);
        assert!((s.l2.to_f64() - 1.0 / l.sqrt()).abs() < 1e-12, "l={l}");
        assert!((s.volume_ratio().to_f64() - 1.0).abs() < 1e-11);
        let e = Stretch::equibiaxial(fx(l));
        assert_eq!(e.l1, e.l2);
        assert_eq!(e.l1, fx(l));
        assert!((e.l3.to_f64() - 1.0 / (l * l)).abs() < 1e-12, "l={l}");
        assert!((e.volume_ratio().to_f64() - 1.0).abs() < 1e-11);
    }
}

#[test]
fn non_positive_stretch_returns_unity() {
    for l in [0.0, -1.0, -0.001] {
        assert_eq!(Stretch::uniaxial(fx(l)), Stretch::UNITY, "uniaxial {l}");
        assert_eq!(
            Stretch::equibiaxial(fx(l)),
            Stretch::UNITY,
            "equibiaxial {l}"
        );
    }
}

#[test]
fn invariants_i1_i2_closed_form_and_equivalence_at_unit_volume() {
    let s = Stretch::uniaxial(fx(2.0));
    // I1 = 4 + 0.5 + 0.5 = 5 ; I2 = 1/4 + 2 + 2 = 4.25
    assert!((s.i1().to_f64() - 5.0).abs() < 1e-11);
    assert!((s.i2().to_f64() - 4.25).abs() < 1e-11);
    let (a, b, c) = (
        s.l1.to_f64().powi(2),
        s.l2.to_f64().powi(2),
        s.l3.to_f64().powi(2),
    );
    assert!((s.i2().to_f64() - (a * b + b * c + c * a)).abs() < 1e-11);
    // non-trivial triple, J = 1: (2, 3, 1/6)
    let t = Stretch {
        l1: fx(2.0),
        l2: fx(3.0),
        l3: fx(1.0 / 6.0),
    };
    let (a, b, c) = (4.0, 9.0, 1.0 / 36.0);
    assert!((t.i1().to_f64() - (a + b + c)).abs() < 1e-11);
    assert!((t.i2().to_f64() - (a * b + b * c + c * a)).abs() < 1e-9);
    assert!((t.volume_ratio().to_f64() - 1.0).abs() < 1e-11);
    assert_eq!(Stretch::UNITY.i1(), Fix128::from_int(3));
    assert_eq!(Stretch::UNITY.i2(), Fix128::from_int(3));
}

// ---------------- presets ----------------

#[test]
fn presets_carry_their_documented_constants() {
    match HyperelasticModel::tpu_soft() {
        HyperelasticModel::NeoHookean { mu_mpa } => assert_eq!(mu_mpa, Fix128::from_int(3)),
        m => panic!("tpu_soft is not Neo-Hookean: {m:?}"),
    }
    match HyperelasticModel::silicone_soft() {
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            assert!((c1_mpa.to_f64() - 0.1).abs() < 1e-18);
            assert!((c2_mpa.to_f64() - 0.05).abs() < 1e-18);
        }
        m => panic!("silicone_soft is not Mooney-Rivlin: {m:?}"),
    }
    match HyperelasticModel::natural_rubber() {
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            assert_eq!(c1_mpa.to_f64(), 0.5);
            // literal is -0.01703125 (comment says -0.017): doc rounding, 0.18 %
            assert!((c2_mpa.to_f64() + 0.017).abs() < 1e-4);
            assert!((c2_mpa.to_f64() + 0.01703125).abs() < 1e-15);
            // literal is 0.000625 (comment says 0.00062): doc rounding, 0.8 %
            assert!((c3_mpa.to_f64() - 0.00062).abs() < 1e-5);
            assert!((c3_mpa.to_f64() - 0.000625).abs() < 1e-15);
        }
        m => panic!("natural_rubber is not Yeoh: {m:?}"),
    }
    // docs: natural rubber "~1 MPa shear modulus at small strain"
    assert!(
        (small_strain_shear_modulus(&HyperelasticModel::natural_rubber()).to_f64() - 1.0).abs()
            < 1e-15
    );
}

// ---------------- energy / uniaxial stress ----------------

#[test]
fn strain_energy_matches_independent_f64_energy() {
    for m in models() {
        for l in [0.6, 1.0, 1.4, 2.5] {
            let w = strain_energy_density(&m, &Stretch::uniaxial(fx(l))).to_f64();
            let want = w_f64(&m, [l, 1.0 / l.sqrt(), 1.0 / l.sqrt()]);
            assert!(
                (w - want).abs() < 1e-9 * (1.0 + want.abs()),
                "{m:?} l={l} w={w} want={want}"
            );
        }
        let e = Stretch::equibiaxial(fx(1.3));
        let w = strain_energy_density(&m, &e).to_f64();
        let want = w_f64(&m, [1.3, 1.3, 1.0 / 1.69]);
        assert!((w - want).abs() < 1e-9, "{m:?} equibiaxial");
        assert!(strain_energy_density(&m, &Stretch::UNITY).to_f64().abs() < 1e-12);
    }
}

/// Doc: uniaxial stress = derivative of W under incompressibility: sigma = lambda dW/dlambda.
#[test]
fn uniaxial_cauchy_stress_is_lambda_times_dw_dlambda() {
    for m in models() {
        for l in [0.7, 1.2, 2.0, 3.0] {
            let h = 1e-5;
            let wp = w_f64(&m, [l + h, 1.0 / (l + h).sqrt(), 1.0 / (l + h).sqrt()]);
            let wm = w_f64(&m, [l - h, 1.0 / (l - h).sqrt(), 1.0 / (l - h).sqrt()]);
            let want = l * (wp - wm) / (2.0 * h);
            let got = uniaxial_cauchy_stress(&m, fx(l)).to_f64();
            assert!(
                (got - want).abs() < 1e-5 * (1.0 + want.abs()),
                "{m:?} l={l} got={got} want={want}"
            );
        }
        assert!(uniaxial_cauchy_stress(&m, Fix128::ONE).to_f64().abs() < 1e-12);
    }
}

#[test]
fn uniaxial_cauchy_stress_non_positive_lambda_is_zero() {
    for m in models() {
        assert_eq!(uniaxial_cauchy_stress(&m, Fix128::ZERO), Fix128::ZERO);
        assert_eq!(uniaxial_cauchy_stress(&m, fx(-2.0)), Fix128::ZERO);
    }
}

#[test]
fn uniaxial_cauchy_stress_closed_forms() {
    // lambda = 2: base = 4 - 0.5 = 3.5
    let nh = HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) };
    assert!((uniaxial_cauchy_stress(&nh, fx(2.0)).to_f64() - 0.8 * 3.5).abs() < 1e-11);
    let mr = HyperelasticModel::MooneyRivlin {
        c1_mpa: fx(0.3),
        c2_mpa: fx(0.12),
    };
    assert!(
        (uniaxial_cauchy_stress(&mr, fx(2.0)).to_f64() - 2.0 * (0.3 + 0.12 / 2.0) * 3.5).abs()
            < 1e-11
    );
    let ye = HyperelasticModel::Yeoh {
        c1_mpa: fx(0.4),
        c2_mpa: fx(-0.02),
        c3_mpa: fx(0.003),
    };
    let d = 2.0_f64 * 2.0 + 2.0 / 2.0 - 3.0; // I1-3 = 4+0.5+0.5-3 = 2
    assert!((d - 2.0).abs() < 1e-12);
    let want = 2.0 * 3.5 * (0.4 + 2.0 * -0.02 * d + 3.0 * 0.003 * d * d);
    assert!((uniaxial_cauchy_stress(&ye, fx(2.0)).to_f64() - want).abs() < 1e-11);
}

// ---------------- small-strain moduli ----------------

#[test]
fn small_strain_shear_modulus_closed_forms_and_agrees_with_moduli() {
    let nh = HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) };
    let mr = HyperelasticModel::MooneyRivlin {
        c1_mpa: fx(0.3),
        c2_mpa: fx(0.12),
    };
    let ye = HyperelasticModel::Yeoh {
        c1_mpa: fx(0.4),
        c2_mpa: fx(-0.02),
        c3_mpa: fx(0.003),
    };
    assert!((small_strain_shear_modulus(&nh).to_f64() - 0.8).abs() < 1e-15);
    assert!((small_strain_shear_modulus(&mr).to_f64() - 2.0 * 0.42).abs() < 1e-15);
    assert!((small_strain_shear_modulus(&ye).to_f64() - 0.8).abs() < 1e-15);
    for m in models() {
        assert_eq!(
            small_strain_moduli(&m).1,
            small_strain_shear_modulus(&m),
            "{m:?}"
        );
    }
    // the slope of the uniaxial curve at lambda -> 1 is 3 mu0 (E = 3 mu for incompressible)
    for m in models() {
        let h = 1e-6;
        let slope = (uniaxial_cauchy_stress(&m, fx(1.0 + h)).to_f64()
            - uniaxial_cauchy_stress(&m, fx(1.0 - h)).to_f64())
            / (2.0 * h);
        let mu0 = small_strain_shear_modulus(&m).to_f64();
        assert!(
            (slope - 3.0 * mu0).abs() < 1e-6,
            "{m:?} slope={slope} 3mu0={}",
            3.0 * mu0
        );
    }
}

#[test]
fn offset_is_zero_4c2_8c2() {
    let nh = HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) };
    let mr = HyperelasticModel::MooneyRivlin {
        c1_mpa: fx(0.3),
        c2_mpa: fx(0.12),
    };
    let ye = HyperelasticModel::Yeoh {
        c1_mpa: fx(0.4),
        c2_mpa: fx(-0.02),
        c3_mpa: fx(0.003),
    };
    assert!(small_strain_moduli(&nh).0.to_f64().abs() < 1e-15);
    assert!((small_strain_moduli(&mr).0.to_f64() - 4.0 * 0.12).abs() < 1e-14);
    assert!((small_strain_moduli(&ye).0.to_f64() - 8.0 * -0.02).abs() < 1e-14);
}

#[test]
fn volumetric_modulus_is_lambda_minus_offset_and_none_below_zero() {
    let mr = HyperelasticModel::MooneyRivlin {
        c1_mpa: fx(0.5),
        c2_mpa: fx(0.25),
    };
    let ye = HyperelasticModel::Yeoh {
        c1_mpa: fx(0.4),
        c2_mpa: fx(0.125),
        c3_mpa: fx(0.003),
    };
    let nh = HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) };
    assert_eq!(volumetric_modulus(&mr, fx(10.0)), Some(fx(9.0)));
    assert_eq!(volumetric_modulus(&ye, fx(10.0)), Some(fx(9.0)));
    assert_eq!(volumetric_modulus(&nh, fx(10.0)), Some(fx(10.0)));
    // boundary: lambda = offset -> kappa = 0 is allowed ("None when kappa < 0")
    assert_eq!(volumetric_modulus(&mr, fx(1.0)), Some(Fix128::ZERO));
    assert_eq!(volumetric_modulus(&ye, fx(1.0)), Some(Fix128::ZERO));
    assert_eq!(volumetric_modulus(&mr, fx(0.5)), None);
    assert_eq!(volumetric_modulus(&ye, fx(0.5)), None);
    assert_eq!(volumetric_modulus(&nh, Fix128::ZERO), Some(Fix128::ZERO));
    assert_eq!(volumetric_modulus(&nh, fx(-0.001)), None);
}

// ---------------- tensor Cauchy stress ----------------

#[test]
fn cauchy_stress_vanishes_at_identity_for_every_model_and_kappa() {
    for m in models() {
        for k in [0.0, 1.0, 50.0, 1000.0] {
            let s = cauchy_stress(&m, fx(k), Mat3Fix::IDENTITY).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(at(&s, i, j).abs() < 1e-12, "{m:?} k={k} ({i},{j})");
                }
            }
        }
    }
}

#[test]
fn cauchy_stress_refuses_det_not_positive() {
    let m = models()[0];
    assert!(cauchy_stress(
        &m,
        fx(10.0),
        mat([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    .is_none());
    assert!(cauchy_stress(
        &m,
        fx(10.0),
        mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    )
    .is_none());
    assert!(cauchy_stress(
        &m,
        fx(10.0),
        mat([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    )
    .is_none());
    // barely positive det is accepted
    assert!(cauchy_stress(
        &m,
        fx(10.0),
        mat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]])
    )
    .is_some());
}

/// Simple shear F = I + g e1 (x) e2 (J = 1): textbook results for the three families.
///  Neo-Hookean/Yeoh: s12 = 2 W1 g, N1 = s11 - s22 = 2 W1 g^2, N2 = 0
///  Mooney-Rivlin  : s12 = 2 (C1 + C2) g, N1 = 2 (C1 + C2) g^2, N2 = -2 C2 g^2
#[test]
fn simple_shear_stresses_match_textbook() {
    let g = 0.4;
    let f = mat([[1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let check = |m: HyperelasticModel, w1: f64, w2: f64| {
        let s = cauchy_stress(&m, fx(37.0), f).unwrap();
        let s12 = at(&s, 0, 1);
        let n1 = at(&s, 0, 0) - at(&s, 1, 1);
        let n2 = at(&s, 1, 1) - at(&s, 2, 2);
        assert!((s12 - 2.0 * (w1 + w2) * g).abs() < 1e-9, "{m:?} s12={s12}");
        assert!((n1 - 2.0 * (w1 + w2) * g * g).abs() < 1e-9, "{m:?} N1={n1}");
        assert!((n2 + 2.0 * w2 * g * g).abs() < 1e-9, "{m:?} N2={n2}");
        // symmetric, no 13/23 shear
        assert!((at(&s, 0, 1) - at(&s, 1, 0)).abs() < 1e-12);
        assert!(at(&s, 0, 2).abs() < 1e-12 && at(&s, 1, 2).abs() < 1e-12);
    };
    check(HyperelasticModel::NeoHookean { mu_mpa: fx(0.8) }, 0.4, 0.0);
    check(
        HyperelasticModel::MooneyRivlin {
            c1_mpa: fx(0.3),
            c2_mpa: fx(0.12),
        },
        0.3,
        0.12,
    );
    let i1m3 = g * g;
    let (c1, c2, c3) = (0.4, -0.02, 0.003);
    check(
        HyperelasticModel::Yeoh {
            c1_mpa: fx(c1),
            c2_mpa: fx(c2),
            c3_mpa: fx(c3),
        },
        c1 + 2.0 * c2 * i1m3 + 3.0 * c3 * i1m3 * i1m3,
        0.0,
    );
}

/// Route independence: the tensor stress under an incompressible uniaxial gradient reproduces
/// the scalar `uniaxial_cauchy_stress` as the axial minus lateral difference.
#[test]
fn tensor_uniaxial_difference_equals_scalar_uniaxial_stress() {
    for m in models() {
        for l in [0.8_f64, 1.5, 2.2] {
            let t = 1.0 / l.sqrt();
            let f = mat([[l, 0.0, 0.0], [0.0, t, 0.0], [0.0, 0.0, t]]);
            let s = cauchy_stress(&m, fx(100.0), f).unwrap();
            let diff = at(&s, 0, 0) - at(&s, 1, 1);
            let want = uniaxial_cauchy_stress(&m, fx(l)).to_f64();
            assert!(
                (diff - want).abs() < 1e-8 * (1.0 + want.abs()),
                "{m:?} l={l} diff={diff} want={want}"
            );
            assert!((at(&s, 1, 1) - at(&s, 2, 2)).abs() < 1e-9);
        }
    }
}

/// Small strain: sigma = lambda_eff tr(eps) I + 2 mu_eff eps with lambda_eff = kappa + offset.
#[test]
fn small_strain_limit_is_the_linear_solid_with_the_documented_moduli() {
    let e = 1e-6;
    let f = mat([
        [1.0 + e, 0.5 * e, 0.0],
        [0.5 * e, 1.0 - 0.3 * e, 0.0],
        [0.0, 0.0, 1.0 + 0.7 * e],
    ]);
    let eps = [
        [e, 0.5 * e, 0.0],
        [0.5 * e, -0.3 * e, 0.0],
        [0.0, 0.0, 0.7 * e],
    ];
    let tr = e - 0.3 * e + 0.7 * e;
    for m in models() {
        let kappa = 40.0;
        let (offset, mu) = small_strain_moduli(&m);
        let lam = kappa + offset.to_f64();
        let s = cauchy_stress(&m, fx(kappa), f).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want =
                    lam * tr * if i == j { 1.0 } else { 0.0 } + 2.0 * mu.to_f64() * eps[i][j];
                assert!(
                    (at(&s, i, j) - want).abs() < 1e-9 * (1.0 + lam),
                    "{m:?} ({i},{j}) got {} want {}",
                    at(&s, i, j),
                    want
                );
            }
        }
    }
}

/// Objectivity: sigma(Q F) = Q sigma(F) Q^T for a rotation Q (90 deg about z).
#[test]
fn cauchy_stress_is_objective_under_rotation() {
    let f = mat([[1.2, 0.3, 0.1], [0.0, 0.9, 0.2], [0.1, 0.0, 1.1]]);
    let q = mat([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let qf = q.mul_mat(f);
    for m in models() {
        let s = cauchy_stress(&m, fx(25.0), f).unwrap();
        let sr = cauchy_stress(&m, fx(25.0), qf).unwrap();
        let want = q.mul_mat(s).mul_mat(q.transpose());
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (at(&sr, i, j) - at(&want, i, j)).abs() < 1e-9,
                    "{m:?} ({i},{j})"
                );
            }
        }
    }
}

/// Barrier: compression drives sigma -> -inf (documented); stress becomes more compressive.
#[test]
fn compression_barrier_is_monotone_hydrostatic() {
    let m = HyperelasticModel::NeoHookean { mu_mpa: fx(1.0) };
    let mut prev = 0.0;
    for s in [0.9, 0.7, 0.5, 0.3, 0.1] {
        let f = mat([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]]);
        let sg = cauchy_stress(&m, fx(5.0), f).unwrap();
        let p = at(&sg, 0, 0);
        assert!(p < prev, "s={s} p={p}");
        prev = p;
    }
}
