//! Closed-form oracles for the fixed-point metric and the metric-aware
//! broadphase.
//!
//! Written before the implementation. Two quantities are involved and they
//! are *not* the same number; conflating them is the whole point of these
//! tests.
//!
//! 1. **The axis half-extent of a metric ball.** `{x : N(x) ≤ r}` reaches
//!    `r / (w₁ + w₂ + w∞)` along each axis, exactly: every basis norm gives
//!    `N(h) ≥ (w₁ + w₂ + w∞)·|hₓ|`, so `|hₓ|/N(h) ≤ 1/Σw`, with equality at
//!    `h = e₁`. For the cube metric that is `r` — a cube-metric ball is the
//!    cube `[−r, r]³` and its box is *not* grown at all.
//! 2. **A distance measured in the metric, converted to a Euclidean
//!    guarantee.** Points at metric distance `d` sit at Euclidean distance
//!    between `d/Lip` and `d/min`, so a clearance of `d` in the metric only
//!    bounds a Euclidean region of radius `d/min` — `√3·d` for the cube
//!    metric. The broadphase margin is a *distance*, so this is the factor
//!    it has to grow by, and skipping it is what lets a pair through.
//!
//! The closed forms themselves (`lipschitz`, `minimum`) are ported from
//! `alice_det_math::metric`, which derives and brute-force-checks them in
//! `f32`; the tests below re-check the `Fix128` port against that crate so
//! the two cannot drift.

use alice_physics::collider::AABB;
use alice_physics::dynamic_bvh::DynamicAabbTree;
use alice_physics::metric::{MetricError, MetricWeights};
use alice_physics::{Fix128, Vec3Fix};

use alice_det_math::metric::MetricWeights as RefWeights;

fn fix(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fix(x), fix(y), fix(z))
}

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() < tol,
        "{what}: got {g}, want {want} (tol {tol})"
    );
}

#[test]
fn pure_bases_have_the_derived_constants() {
    // L1: max ‖h‖₁ on the Euclidean unit sphere is √3, min is 1
    close(
        MetricWeights::L1.lipschitz(),
        3.0_f64.sqrt(),
        1e-9,
        "L1 lipschitz",
    );
    close(MetricWeights::L1.minimum(), 1.0, 1e-9, "L1 minimum");
    // L2: the Euclidean metric is 1-Lipschitz in both directions
    close(MetricWeights::L2.lipschitz(), 1.0, 1e-12, "L2 lipschitz");
    close(MetricWeights::L2.minimum(), 1.0, 1e-12, "L2 minimum");
    // L∞: max 1, min 1/√3 — a metric clearance of 1 only bounds √3 Euclidean
    close(MetricWeights::LINF.lipschitz(), 1.0, 1e-9, "Linf lipschitz");
    close(
        MetricWeights::LINF.minimum(),
        1.0 / 3.0_f64.sqrt(),
        1e-9,
        "Linf minimum",
    );
    close(
        MetricWeights::LINF.euclidean_radius(Fix128::ONE),
        3.0_f64.sqrt(),
        1e-8,
        "Linf euclidean_radius(1)",
    );
    // …while the ball itself is exactly the unit cube: no axis growth
    close(
        MetricWeights::LINF.axis_extent(Fix128::ONE),
        1.0,
        1e-12,
        "Linf axis_extent(1)",
    );
    close(
        MetricWeights::L1.axis_extent(Fix128::ONE),
        1.0,
        1e-12,
        "L1 axis_extent(1)",
    );
}

#[test]
fn fixed_point_port_agrees_with_the_f32_reference() {
    // the same law lives in alice-det-math (f32, brute-force checked there);
    // this is the port parity oracle that stops the two from drifting
    for (a, b, c) in [
        (1.0_f64, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.3, 0.5, 0.2),
        (2.0, 0.0, 5.0),
        (0.125, 0.25, 0.625),
    ] {
        let got = MetricWeights::new(fix(a), fix(b), fix(c)).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let want = RefWeights::new(a as f32, b as f32, c as f32).unwrap();
        close(
            got.lipschitz(),
            f64::from(want.lipschitz()),
            1e-6,
            "lipschitz port",
        );
        close(
            got.minimum(),
            f64::from(want.minimum()),
            1e-6,
            "minimum port",
        );
    }
}

#[test]
fn axis_extent_is_the_exact_half_width_of_the_metric_ball() {
    // brute force: walk directions, put the point at metric distance 1 along
    // each, and take the largest x it reaches. Shares no code with the
    // closed form.
    for (a, b, c) in [
        (1.0_f64, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.3, 0.5, 0.2),
        (1.0, 1.0, 1.0),
    ] {
        let w = MetricWeights::new(fix(a), fix(b), fix(c)).unwrap();
        let mut widest = 0.0_f64;
        let n = 24;
        for i in -n..=n {
            for j in -n..=n {
                for k in -n..=n {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let h = v(f64::from(i), f64::from(j), f64::from(k));
                    let norm = w.norm(h).to_f64();
                    // scale the direction onto the unit sphere of the metric
                    widest = widest.max((f64::from(i) / norm).abs());
                }
            }
        }
        close(
            w.axis_extent(Fix128::ONE),
            widest,
            1e-6,
            "axis_extent vs brute force",
        );
    }
}

#[test]
fn norm_reproduces_the_basis_definitions() {
    let p = v(1.0, 2.0, -3.0);
    close(MetricWeights::L1.norm(p), 6.0, 1e-9, "L1 norm");
    close(MetricWeights::L2.norm(p), 14.0_f64.sqrt(), 1e-9, "L2 norm");
    close(MetricWeights::LINF.norm(p), 3.0, 1e-9, "Linf norm");
}

#[test]
fn a_negative_weight_is_not_a_metric() {
    assert_eq!(
        MetricWeights::new(fix(-0.5), Fix128::ONE, Fix128::ZERO),
        Err(MetricError::NotConvex)
    );
    assert_eq!(
        MetricWeights::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        Err(MetricError::Degenerate)
    );
}

#[test]
fn the_euclidean_metric_leaves_every_proxy_box_bit_identical() {
    // determinism: turning the feature on without changing the metric must
    // not move a single bit, or every recorded golden breaks
    let boxes = [
        AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0)),
        AABB::new(v(3.25, -0.5, 2.0), v(4.75, 0.5, 2.5)),
        AABB::new(v(-10.0, 7.0, -3.0), v(-9.0, 8.0, -2.75)),
    ];

    let mut plain = DynamicAabbTree::new();
    let mut explicit = DynamicAabbTree::new();
    explicit.metric = MetricWeights::L2;

    for (i, b) in boxes.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let id_a = plain.insert(*b, i as u32);
        #[allow(clippy::cast_possible_truncation)]
        let id_b = explicit.insert(*b, i as u32);
        assert_eq!(
            plain.get_aabb(id_a),
            explicit.get_aabb(id_b),
            "Euclidean metric perturbed proxy {i}"
        );
    }
}

#[test]
fn a_cube_metric_clearance_is_missed_without_the_expansion_and_caught_with_it() {
    // Two boxes 1.2 apart, with a margin of 1.0 *in the metric*. Under the
    // Euclidean reading the fattened boxes reach 1.0 each way and the gap
    // (1.2) is covered — so pick a gap that only the metric reading covers:
    // 2.4 is beyond 2×1.0 but inside 2×√3 ≈ 3.46.
    let gap = 2.4;
    let a = AABB::new(v(-0.5, -0.5, -0.5), v(0.5, 0.5, 0.5));
    let b = AABB::new(v(0.5 + gap, -0.5, -0.5), v(1.5 + gap, 0.5, 0.5));

    let mut euclidean = DynamicAabbTree::new();
    euclidean.margin = Fix128::ONE;
    let ea = euclidean.insert(a, 0);
    let eb = euclidean.insert(b, 1);
    assert!(
        !euclidean.get_aabb(ea).intersects(&euclidean.get_aabb(eb)),
        "the Euclidean margin should not reach across {gap}"
    );

    let mut cube = DynamicAabbTree::new();
    cube.margin = Fix128::ONE;
    cube.metric = MetricWeights::LINF;
    let ca = cube.insert(a, 0);
    let cb = cube.insert(b, 1);
    assert!(
        cube.get_aabb(ca).intersects(&cube.get_aabb(cb)),
        "a clearance of 1 in the cube metric reaches √3 in Euclidean space, \
         so this pair must survive the broadphase"
    );

    // and the growth is the derived factor, not an arbitrary fudge
    let half_e = (euclidean.get_aabb(ea).max.x - euclidean.get_aabb(ea).min.x).to_f64() / 2.0;
    let half_c = (cube.get_aabb(ca).max.x - cube.get_aabb(ca).min.x).to_f64() / 2.0;
    // 0.5 + margin vs 0.5 + √3·margin
    close(fix(half_e), 1.5, 1e-9, "euclidean fattened half width");
    close(
        fix(half_c),
        0.5 + 3.0_f64.sqrt(),
        1e-6,
        "cube fattened half width",
    );
}

#[test]
fn metric_ball_box_is_tight_and_never_smaller_than_the_ball() {
    // AABB::from_metric_ball must contain the ball and touch it: sample the
    // ball's surface and check both directions
    for (a, b, c) in [(1.0_f64, 0.0, 0.0), (0.0, 0.0, 1.0), (0.4, 0.4, 0.2)] {
        let w = MetricWeights::new(fix(a), fix(b), fix(c)).unwrap();
        let r = fix(2.0);
        let boxed = AABB::from_metric_ball(v(1.0, -2.0, 0.5), r, w);
        let mut touched = false;
        let n = 16;
        for i in -n..=n {
            for j in -n..=n {
                for k in -n..=n {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let h = v(f64::from(i), f64::from(j), f64::from(k));
                    let scale = r.to_f64() / w.norm(h).to_f64();
                    let p = v(
                        1.0 + f64::from(i) * scale,
                        -2.0 + f64::from(j) * scale,
                        0.5 + f64::from(k) * scale,
                    );
                    let over = [
                        boxed.min.x.to_f64() - p.x.to_f64(),
                        p.x.to_f64() - boxed.max.x.to_f64(),
                        boxed.min.y.to_f64() - p.y.to_f64(),
                        p.y.to_f64() - boxed.max.y.to_f64(),
                        boxed.min.z.to_f64() - p.z.to_f64(),
                        p.z.to_f64() - boxed.max.z.to_f64(),
                    ]
                    .into_iter()
                    .fold(f64::NEG_INFINITY, f64::max);
                    assert!(
                        over < 1e-9,
                        "surface point escaped the box by {over} for {a},{b},{c}"
                    );
                    if (p.x.to_f64() - boxed.max.x.to_f64()).abs() < 1e-6 {
                        touched = true;
                    }
                }
            }
        }
        assert!(touched, "box is not tight for {a},{b},{c}");
    }
}
