//! Oracles for `metric::MetricWeights::{weights, norm, lipschitz}`
//! (`examples/metric_clearance_bounds.rs`), independent of the `alice_det_math` port check
//! in `tests/analytic_metric_broadphase.rs`:
//! * `norm` against the definition `w1*|v|_1 + w2*|v|_2 + winf*|v|_inf` evaluated in f64
//! * `lipschitz` / `minimum` against a brute-force sweep of the Euclidean unit sphere
//!   AND against the explicit extremal directions derived by hand:
//!   max at `h ~ (w1+winf, w1, w1)`, min at `(1,..,1)/sqrt(k)` over `k` axes
//! * metric axioms for non-negative weights (homogeneity, triangle inequality, symmetry)
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::metric::{MetricError, MetricWeights};

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn vec(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn g(w: (f64, f64, f64), v: (f64, f64, f64)) -> f64 {
    let (x, y, z) = (v.0.abs(), v.1.abs(), v.2.abs());
    w.0 * (x + y + z) + w.1 * (x * x + y * y + z * z).sqrt() + w.2 * x.max(y).max(z)
}
const WEIGHTS: &[(f64, f64, f64)] = &[
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 2.0, 3.0),
    (0.5, 0.0, 2.0),
    (0.25, 4.0, 0.0),
    (3.0, 0.0, 0.0),
    (0.0, 0.5, 0.125),
];

fn mw(w: (f64, f64, f64)) -> MetricWeights {
    MetricWeights::new(fx(w.0), fx(w.1), fx(w.2)).unwrap()
}

#[test]
fn weights_round_trip_and_constants() {
    for &w in WEIGHTS {
        let (a, b, c) = mw(w).weights();
        assert_eq!((a, b, c), (fx(w.0), fx(w.1), fx(w.2)));
    }
    assert_eq!(
        MetricWeights::L1.weights(),
        (Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
    assert_eq!(
        MetricWeights::L2.weights(),
        (Fix128::ZERO, Fix128::ONE, Fix128::ZERO)
    );
    assert_eq!(
        MetricWeights::LINF.weights(),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ONE)
    );
    assert_eq!(MetricWeights::default(), MetricWeights::L2);
    assert!(MetricWeights::L2.is_euclidean());
    assert!(!mw((0.0, 1.0, 0.5)).is_euclidean());
    assert!(!mw((0.5, 1.0, 0.0)).is_euclidean());
    assert!(!mw((0.0, 2.0, 0.0)).is_euclidean());
}

#[test]
fn constructor_rejects_non_metrics() {
    let z = Fix128::ZERO;
    let n = Fix128::from_ratio(-1, 1_000_000);
    assert_eq!(
        MetricWeights::new(n, Fix128::ONE, z),
        Err(MetricError::NotConvex)
    );
    assert_eq!(
        MetricWeights::new(z, n, Fix128::ONE),
        Err(MetricError::NotConvex)
    );
    assert_eq!(
        MetricWeights::new(Fix128::ONE, z, n),
        Err(MetricError::NotConvex)
    );
    assert_eq!(MetricWeights::new(z, z, z), Err(MetricError::Degenerate));
    assert!(MetricWeights::new(z, z, Fix128::from_ratio(1, 1_000_000)).is_ok());
}

#[test]
fn norm_matches_the_definition() {
    let pts = [
        (1.0, -2.0, 3.0),
        (0.0, 0.0, 0.0),
        (-5.0, 0.0, 0.0),
        (0.0, 4.0, -3.0),
        (2.5, 2.5, -2.5),
        (-0.125, 7.0, 0.5),
    ];
    for &w in WEIGHTS {
        for &p in &pts {
            let got = mw(w).norm(vec(p.0, p.1, p.2)).to_f64();
            assert!(
                (got - g(w, p)).abs() < 1e-9,
                "w={w:?} p={p:?}: {got} vs {}",
                g(w, p)
            );
        }
    }
    // hand values: (3,4,0) -> L1 7, L2 5, Linf 4
    let v = vec(3.0, -4.0, 0.0);
    assert_eq!(MetricWeights::L1.norm(v), Fix128::from_int(7));
    assert_eq!(MetricWeights::L2.norm(v), Fix128::from_int(5));
    assert_eq!(MetricWeights::LINF.norm(v), Fix128::from_int(4));
    // each axis alone as the maximum (the three comparisons of the L-inf branch)
    for (v, want) in [
        ((9.0, 2.0, 3.0), 9.0),
        ((2.0, 9.0, 3.0), 9.0),
        ((2.0, 3.0, 9.0), 9.0),
        ((-9.0, 2.0, 3.0), 9.0),
        ((2.0, -3.0, -9.0), 9.0),
    ] {
        assert_eq!(
            MetricWeights::LINF.norm(vec(v.0, v.1, v.2)),
            Fix128::from_int(want as i64)
        );
    }
}

#[test]
fn norm_is_a_norm() {
    let m = mw((1.0, 2.0, 3.0));
    let a = vec(1.0, -2.0, 3.0);
    let b = vec(-0.5, 4.0, 1.0);
    let w = (1.0, 2.0, 3.0);
    assert_eq!(m.norm(Vec3Fix::ZERO), Fix128::ZERO);
    for s in [0.5f64, 2.0, 7.0] {
        let scaled = vec(s, -2.0 * s, 3.0 * s);
        assert!((m.norm(scaled).to_f64() - s * m.norm(a).to_f64()).abs() < 1e-9);
    }
    let neg = vec(-1.0, 2.0, -3.0);
    assert_eq!(m.norm(neg), m.norm(a));
    let sum = vec(1.0 - 0.5, -2.0 + 4.0, 3.0 + 1.0);
    assert!(m.norm(sum).to_f64() <= m.norm(a).to_f64() + m.norm(b).to_f64() + 1e-12);
    assert!((g(w, (0.5, 2.0, 4.0)) - m.norm(sum).to_f64()).abs() < 1e-9);
}

#[test]
fn lipschitz_and_minimum_match_a_sphere_sweep_and_the_closed_form_extremisers() {
    for &w in WEIGHTS {
        let m = mw(w);
        let (lip, min) = (m.lipschitz().to_f64(), m.minimum().to_f64());
        // brute-force sweep of the unit sphere
        let (mut hi, mut lo) = (f64::MIN, f64::MAX);
        let (nt, np) = (180, 360);
        for i in 0..=nt {
            let th = std::f64::consts::PI * i as f64 / nt as f64;
            for j in 0..np {
                let ph = 2.0 * std::f64::consts::PI * j as f64 / np as f64;
                let p = (th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                let val = g(w, p);
                hi = hi.max(val);
                lo = lo.min(val);
            }
        }
        assert!(
            hi <= lip + 1e-9 && lip - hi < 1e-2,
            "w={w:?} sweep max {hi} vs lipschitz {lip}"
        );
        assert!(
            lo >= min - 1e-9 && lo - min < 1e-2,
            "w={w:?} sweep min {lo} vs minimum {min}"
        );
        // explicit maximiser u = (w1+winf, w1, w1) (normalised): g(u/|u|) = lipschitz
        let u = if w.0 + w.2 == 0.0 {
            (1.0, 0.0, 0.0)
        } else {
            (w.0 + w.2, w.0, w.0)
        }; // pure L2: any direction
        let n = (u.0 * u.0 + u.1 * u.1 + u.2 * u.2).sqrt();
        let at_max = g(w, (u.0 / n, u.1 / n, u.2 / n));
        assert!(
            (at_max - lip).abs() < 1e-9,
            "w={w:?} maximiser {at_max} vs {lip}"
        );
        // minimisers: first k axes equal, 1/sqrt(k)
        let mut best = f64::MAX;
        for k in 1..=3 {
            let c = 1.0 / (k as f64).sqrt();
            let p = (
                c,
                if k >= 2 { c } else { 0.0 },
                if k >= 3 { c } else { 0.0 },
            );
            best = best.min(g(w, p));
        }
        assert!(
            (best - min).abs() < 1e-9,
            "w={w:?} minimiser {best} vs {min}"
        );
    }
}

#[test]
fn lipschitz_hand_values() {
    // L1: sqrt(1 + 2) = sqrt3; L2: 1; Linf: 1
    assert!((MetricWeights::L1.lipschitz().to_f64() - 3f64.sqrt()).abs() < 1e-12);
    assert!((MetricWeights::L2.lipschitz().to_f64() - 1.0).abs() < 1e-12);
    assert!((MetricWeights::LINF.lipschitz().to_f64() - 1.0).abs() < 1e-12);
    // (1,2,3): sqrt(18) + 2
    let m = mw((1.0, 2.0, 3.0));
    assert!((m.lipschitz().to_f64() - (18f64.sqrt() + 2.0)).abs() < 1e-12);
    // the Lipschitz bound: any Euclidean step s moves the norm by at most lip * |s|
    let a = vec(1.0, 2.0, 3.0);
    let step = vec(0.3, -0.2, 0.4);
    let b = Vec3Fix::new(a.x + step.x, a.y + step.y, a.z + step.z);
    let d = (m.norm(b) - m.norm(a)).abs().to_f64();
    assert!(d <= m.lipschitz().to_f64() * step.length().to_f64() + 1e-12);
}
