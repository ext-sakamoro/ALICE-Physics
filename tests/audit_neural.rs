//! Audit oracles for `neural` (deterministic ternary network + ragdoll
//! controller). Expected values are closed forms evaluated independently
//! (`f64` tanh, hand-written integer dot products), never the crate's own
//! formula.
//!
//! Audited here (already pinned elsewhere and not repeated: ternary matvec
//! as exact dot product, ReLU / HardTanh / LeakyReLU characteristic points,
//! two-layer forward on a tiny network, torque clamp):
//! - accuracy / boundedness / monotonicity of `fix128_tanh_approx` against
//!   the true `tanh` (the doc claims a Pade approximant with ~0.004 error)
//! - the controller's feature layout (position, velocity, rotation xyzw,
//!   angular velocity) for every one of the 13 slots of two bodies
//! - the `features_per_body` configuration field
//! - joint-major torque layout and sign preservation
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "std", feature = "neural"))]
#![allow(clippy::disallowed_methods)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_ml::TernaryWeight;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::neural::{
    fix128_hard_tanh, fix128_relu, fix128_tanh_approx, fix128_ternary_matvec, Activation,
    ControllerConfig, DeterministicNetwork, FixedTernaryWeight, RagdollController,
    FEATURES_PER_BODY,
};
use alice_physics::solver::RigidBody;

fn tanh_approx_at(x: Fix128) -> f64 {
    let mut v = [x];
    fix128_tanh_approx(&mut v);
    v[0].to_f64()
}

fn identity_layer(n_out: usize, n_in: usize) -> FixedTernaryWeight {
    let mut vals = vec![0i8; n_out * n_in];
    for r in 0..n_out.min(n_in) {
        vals[r * n_in + r] = 1;
    }
    FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&vals, n_out, n_in),
        Fix128::ONE,
    )
}

// ---------------------------------------------------------------------------
// tanh approximation
// ---------------------------------------------------------------------------

#[test]
fn tanh_approx_is_odd_and_zero_at_the_origin() {
    assert_eq!(tanh_approx_at(Fix128::ZERO), 0.0);
    for k in 1..=40 {
        let x = Fix128::from_ratio(k, 8);
        let pos = tanh_approx_at(x);
        let neg = tanh_approx_at(Fix128::ZERO - x);
        assert!((pos + neg).abs() < 1e-15, "x = {k}/8: {pos} vs {neg}");
    }
}

#[test]
fn tanh_approx_error_near_zero_is_the_cubic_gap_between_the_two_taylor_series() {
    // f(x) = x (27 + x^2) / (27 + 9 x^2) = x - 8 x^3 / 27 + 8 x^5 / 81 - ...
    // tanh(x)                              = x -   x^3 / 3  + 2 x^5 / 15 - ...
    // so f - tanh = x^3 / 27 + O(x^5): the formula is not a Pade approximant of tanh
    // (whose error starts at x^5). Checked against the analytic leading term.
    for k in 1..=8 {
        let x = k as f64 / 32.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 32));
        let err = got - x.tanh();
        assert!(
            (err - x * x * x / 27.0).abs() < 0.04 * x.powi(5),
            "x = {x}: err {err}"
        );
    }
}

#[test]
fn tanh_approx_stays_within_three_hundredths_of_tanh_up_to_four() {
    // Measured maximum is ~0.0236 (near |x| = 1.5); this pins the real accuracy
    // class so a regression to a worse rational is caught.
    let mut worst = 0.0_f64;
    for k in 0..=320 {
        let xf = k as f64 / 80.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 80));
        worst = worst.max((got - xf.tanh()).abs());
    }
    assert!(worst < 0.03, "worst error {worst}");
}

#[test]
fn tanh_approx_clamps_to_unit_beyond_four() {
    assert_eq!(tanh_approx_at(Fix128::from_ratio(41, 10)), 1.0);
    assert_eq!(tanh_approx_at(Fix128::from_ratio(-41, 10)), -1.0);
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-003: fix128_tanh_approx doc claims a Pade approximant with ~0.004 max error for |x| < 4.5; the formula x(27+x^2)/(27+9x^2) is not the Pade[3/2] of tanh (x(15+x^2)/(15+6x^2)) and its max error vs tanh is 0.0235 at |x| = 1.56 (0.0201 at 2.0)"]
fn tanh_approx_max_error_matches_the_documented_0_004() {
    let mut worst = 0.0_f64;
    let mut at = 0.0_f64;
    for k in 0..=360 {
        let xf = k as f64 / 80.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 80));
        let e = (got - xf.tanh()).abs();
        if e > worst {
            worst = e;
            at = xf;
        }
    }
    assert!(worst < 0.0045, "worst error {worst} at x = {at}");
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-004: fix128_tanh_approx (Activation::TanhApprox, doc 'smooth bounded output') returns values above 1 for 3 < x <= 4 (1.0058 at x = 4) and then drops back to exactly 1 for x > 4, so it is neither bounded by 1 nor monotone"]
fn tanh_approx_is_bounded_by_one_and_monotone() {
    let mut prev = -1.0_f64;
    for k in -400..=400 {
        let y = tanh_approx_at(Fix128::from_ratio(k, 80));
        assert!(y.abs() <= 1.0, "|f({})| = {y} exceeds 1", k as f64 / 80.0);
        assert!(
            y >= prev,
            "not monotone at x = {}: {y} < {prev}",
            k as f64 / 80.0
        );
        prev = y;
    }
}

// ---------------------------------------------------------------------------
// matvec / activation algebra
// ---------------------------------------------------------------------------

#[test]
fn ternary_matvec_is_linear_in_the_input() {
    let vals = [1i8, -1, 0, 1, 1, -1];
    let w = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&vals, 2, 3),
        Fix128::from_ratio(1, 4),
    );
    let a = [
        Fix128::from_int(3),
        Fix128::from_int(-5),
        Fix128::from_int(7),
    ];
    let b = [
        Fix128::from_int(-2),
        Fix128::from_int(8),
        Fix128::from_int(1),
    ];
    let ab = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let (mut oa, mut ob, mut oab) = ([Fix128::ZERO; 2], [Fix128::ZERO; 2], [Fix128::ZERO; 2]);
    fix128_ternary_matvec(&a, &w, &mut oa);
    fix128_ternary_matvec(&b, &w, &mut ob);
    fix128_ternary_matvec(&ab, &w, &mut oab);
    for i in 0..2 {
        assert_eq!(oab[i], oa[i] + ob[i], "row {i}");
    }
    // row 0 = [+1, -1, 0] . [3, -5, 7] / 4 = (3 + 5) / 4 = 2
    assert_eq!(oa[0], Fix128::from_int(2));
    // row 1 = [+1, +1, -1] . [3, -5, 7] / 4 = (3 - 5 - 7) / 4 = -9/4
    assert_eq!(oa[1], Fix128::from_ratio(-9, 4));
}

#[test]
fn relu_is_idempotent_and_hard_tanh_is_idempotent() {
    let src = [
        Fix128::from_ratio(-7, 3),
        Fix128::from_ratio(1, 3),
        Fix128::from_ratio(5, 3),
        Fix128::from_int(-1),
        Fix128::ONE,
    ];
    let mut once = src;
    fix128_relu(&mut once);
    let mut twice = once;
    fix128_relu(&mut twice);
    assert_eq!(once, twice);
    let mut once = src;
    fix128_hard_tanh(&mut once);
    let mut twice = once;
    fix128_hard_tanh(&mut twice);
    assert_eq!(once, twice);
    for v in once {
        assert!(v >= Fix128::NEG_ONE && v <= Fix128::ONE);
    }
}

#[test]
fn network_accessors_and_activation_dispatch_follow_the_layer_list() {
    // 4 -> 3 (None) -> 2 (HardTanh); identity-like layers, scale 3 on layer 2
    let l1 = identity_layer(3, 4);
    let l2 = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&[1, 0, 0, 0, 1, 0], 2, 3),
        Fix128::from_int(3),
    );
    let mut net =
        DeterministicNetwork::new(vec![l1, l2], vec![Activation::None, Activation::HardTanh]);
    assert_eq!(net.num_layers(), 2);
    assert_eq!(net.input_size(), 4);
    assert_eq!(net.output_size(), 2);
    let out = net
        .forward(&[
            Fix128::from_ratio(1, 2),
            Fix128::from_int(-2),
            Fix128::from_int(9),
            Fix128::from_int(9),
        ])
        .to_vec();
    // layer1 = [1/2, -2, 9]; layer2 = 3*[1/2, -2] = [3/2, -6] ; HardTanh -> [1, -1]
    assert_eq!(out, vec![Fix128::ONE, Fix128::NEG_ONE]);
}

// ---------------------------------------------------------------------------
// ragdoll controller
// ---------------------------------------------------------------------------

fn controller_selecting_all_features(
    num_bodies: usize,
    num_joints: usize,
    max_torque: i64,
) -> RagdollController {
    let n_in = num_bodies * FEATURES_PER_BODY;
    let net = DeterministicNetwork::new(
        vec![identity_layer(num_joints * 3, n_in)],
        vec![Activation::None],
    );
    RagdollController::new(
        net,
        ControllerConfig {
            max_torque: Fix128::from_int(max_torque),
            num_joints,
            num_bodies,
            features_per_body: FEATURES_PER_BODY,
        },
    )
}

fn labelled_body(base: i64) -> RigidBody {
    let mut b =
        RigidBody::new_dynamic(Vec3Fix::from_int(base + 1, base + 2, base + 3), Fix128::ONE);
    b.velocity = Vec3Fix::from_int(base + 4, base + 5, base + 6);
    b.rotation = QuatFix {
        x: Fix128::from_int(base + 7),
        y: Fix128::from_int(base + 8),
        z: Fix128::from_int(base + 9),
        w: Fix128::from_int(base + 10),
    };
    b.angular_velocity = Vec3Fix::from_int(base + 11, base + 12, base + 13);
    b
}

#[test]
fn feature_layout_is_position_velocity_rotation_xyzw_angular_velocity_per_body() {
    // 2 bodies -> 26 features; 9 joints -> 27 outputs (the last output reads nothing).
    let mut c = controller_selecting_all_features(2, 9, 1000);
    let out = c.compute(&[labelled_body(0), labelled_body(100)]);
    let flat: Vec<i64> = out
        .torques
        .iter()
        .flat_map(|t| [t.x, t.y, t.z])
        .map(|v| v.hi)
        .collect();
    let mut want: Vec<i64> = (1..=13).collect();
    want.extend(101..=113);
    want.push(0);
    assert_eq!(flat, want);
}

#[test]
fn controller_output_is_joint_major_with_independent_axis_clamps() {
    // 1 body, 5 joints; torque limit 6: values 1..13 clamp at 6 per axis.
    let mut c = controller_selecting_all_features(1, 5, 6);
    let out = c.compute(&[labelled_body(0)]);
    assert_eq!(out.torques.len(), 5);
    let flat: Vec<i64> = out
        .torques
        .iter()
        .flat_map(|t| [t.x, t.y, t.z])
        .map(|v| v.hi)
        .collect();
    assert_eq!(flat, vec![1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0]);
}

#[test]
fn controller_clamp_is_symmetric_for_negative_features() {
    let mut c = controller_selecting_all_features(1, 1, 3);
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(-10, 2, -3), Fix128::ONE);
    b.velocity = Vec3Fix::ZERO;
    let out = c.compute(&[b]);
    assert_eq!(out.torques[0].x, Fix128::from_int(-3));
    assert_eq!(out.torques[0].y, Fix128::from_int(2));
    assert_eq!(out.torques[0].z, Fix128::from_int(-3));
}

#[test]
fn controller_with_fewer_bodies_zero_fills_the_missing_slots_not_stale_data() {
    let mut c = controller_selecting_all_features(2, 9, 1000);
    let _ = c.compute(&[labelled_body(0), labelled_body(100)]);
    let out = c.compute(&[labelled_body(0)]);
    let flat: Vec<i64> = out
        .torques
        .iter()
        .flat_map(|t| [t.x, t.y, t.z])
        .map(|v| v.hi)
        .collect();
    let mut want: Vec<i64> = (1..=13).collect();
    want.resize(27, 0);
    assert_eq!(flat, want);
}

#[test]
fn features_per_body_below_thirteen_does_not_panic_or_alias() {
    let n_bodies = 2usize;
    let fpb = 10usize;
    let net = DeterministicNetwork::new(
        vec![identity_layer(6, n_bodies * fpb)],
        vec![Activation::None],
    );
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut c = RagdollController::new(
            net,
            ControllerConfig {
                max_torque: Fix128::from_int(1000),
                num_joints: 2,
                num_bodies: n_bodies,
                features_per_body: fpb,
            },
        );
        c.compute(&[labelled_body(0), labelled_body(100)])
    }));
    assert!(r.is_ok(), "compute panicked for features_per_body = 10");
}

#[test]
fn mismatched_layer_chain_is_rejected_at_construction() {
    let l1 = identity_layer(3, 4); // out 3
    let l2 = identity_layer(2, 5); // in 5 != 3
    let r = catch_unwind(AssertUnwindSafe(|| {
        DeterministicNetwork::new(vec![l1, l2], vec![Activation::None, Activation::None])
    }));
    assert!(
        r.is_err(),
        "layer 2 expects 5 inputs but layer 1 produces 3"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-007: DeterministicNetwork::forward on a zero-layer network panics (index out of bounds on buf_offsets[1] / usize underflow of n - 1) although new(vec![], vec![]) succeeds and the doc lists only the length-mismatch panic; also input_size()/output_size() panic on it (pinned as current behaviour in analytic_neural_wiring.rs, not recorded as a defect there) -- escalated: tests/analytic_neural_wiring.rs::deterministic_network_zero_layers_constructs_but_forward_panics pins the panic as current behaviour and names this exact choice (validate in new() vs return &[] in forward()) a design decision, not a silent fix"]
fn zero_layer_network_forward_does_not_panic() {
    let mut net = DeterministicNetwork::new(vec![], vec![]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = net.forward(&[]);
    }));
    assert!(r.is_ok());
}

#[test]
fn matvec_ignores_extra_input_columns_and_leaves_extra_output_slots_untouched() {
    // 2 x 3 weight, input longer than 3 and output longer than 2
    let w = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&[1, 0, -1, 1, 1, 1], 2, 3),
        Fix128::ONE,
    );
    let input = [
        Fix128::from_int(5),
        Fix128::from_int(7),
        Fix128::from_int(11),
        Fix128::from_int(1000), // beyond in_features: must not contribute
    ];
    let sentinel = Fix128::from_int(-777);
    let mut output = [sentinel; 4];
    fix128_ternary_matvec(&input, &w, &mut output);
    assert_eq!(output[0], Fix128::from_int(5 - 11));
    assert_eq!(output[1], Fix128::from_int(5 + 7 + 11));
    assert_eq!(output[2], sentinel, "slot beyond out_features was written");
    assert_eq!(output[3], sentinel, "slot beyond out_features was written");
}

#[test]
fn documented_constructor_panics_fire_for_every_dimension_mismatch() {
    // DeterministicNetwork::new: layers.len() != activations.len()
    let r = catch_unwind(AssertUnwindSafe(|| {
        DeterministicNetwork::new(vec![identity_layer(2, 2)], vec![])
    }));
    assert!(r.is_err(), "length mismatch must panic");
    let r = catch_unwind(AssertUnwindSafe(|| {
        DeterministicNetwork::new(vec![], vec![Activation::None])
    }));
    assert!(r.is_err(), "length mismatch must panic");

    // RagdollController::new: input size must be num_bodies * features_per_body,
    // output size must be num_joints * 3
    let make = |n_in: usize, n_out: usize, bodies: usize, joints: usize| {
        catch_unwind(AssertUnwindSafe(|| {
            let net = DeterministicNetwork::new(
                vec![identity_layer(n_out, n_in)],
                vec![Activation::None],
            );
            RagdollController::new(
                net,
                ControllerConfig {
                    max_torque: Fix128::ONE,
                    num_joints: joints,
                    num_bodies: bodies,
                    features_per_body: FEATURES_PER_BODY,
                },
            )
        }))
    };
    assert!(make(13, 3, 1, 1).is_ok());
    assert!(make(12, 3, 1, 1).is_err(), "input too small");
    assert!(make(14, 3, 1, 1).is_err(), "input too large");
    assert!(make(13, 2, 1, 1).is_err(), "output too small");
    assert!(make(13, 4, 1, 1).is_err(), "output too large");
    assert!(make(26, 3, 2, 1).is_ok());
    assert!(make(13, 6, 1, 2).is_ok());
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-016: fix128_ternary_matvec doc says 'No rounding error', but the final `acc * scale` is a truncating Q64.64 multiply: input 1/3 (raw 0x5555..) times f32 scale 0.1 loses the low 88-64 fraction bits (output differs from the exact product by 1 ulp = 2^-64)"]
fn matvec_output_is_the_exact_product_of_the_integer_sum_and_the_scale() {
    // exact value: raw_x * raw_scale / 2^64 (raw_scale = f32 0.1 widened exactly)
    let x = Fix128::from_ratio(1, 3);
    let scale = Fix128::from_f64(f64::from(0.1_f32));
    let w = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&[1], 1, 1),
        scale,
    );
    let mut out = [Fix128::ZERO];
    fix128_ternary_matvec(&[x], &w, &mut out);
    let raw = |v: Fix128| ((v.hi as i128) << 64) | i128::from(v.lo);
    let exact_num = raw(x) * raw(scale);
    // exact product in units of 2^-128; the output carries 2^-64 resolution
    let got_scaled = raw(out[0]) << 64;
    assert_eq!(got_scaled, exact_num, "product was rounded");
}

#[test]
fn tanh_approx_layer_activation_is_the_smooth_function_not_the_hard_clamp() {
    // x = 0.5: tanh = 0.4621, smooth approximant within 0.01 of it, hard clamp would return 0.5
    let mut net =
        DeterministicNetwork::new(vec![identity_layer(1, 1)], vec![Activation::TanhApprox]);
    let y = net.forward(&[Fix128::from_ratio(1, 2)])[0].to_f64();
    assert!((y - 0.5_f64.tanh()).abs() < 0.01, "y = {y}");
    assert!((y - 0.5).abs() > 0.02, "y = {y} equals the hard clamp");
    // |x| > 4 saturates to +-1, so x = 9 cannot tell the two apart; x = 2 can
    let y2 = net.forward(&[Fix128::from_int(2)])[0].to_f64();
    assert!((y2 - 2.0_f64.tanh()).abs() < 0.03 && y2 < 1.0, "y2 = {y2}");
}

#[test]
fn controller_config_default_matches_the_documented_ragdoll_shape() {
    // doc example: 8 joints, 9 bodies, 13 features per body, 100 N*m torque limit
    let c = ControllerConfig::default();
    assert_eq!(c.max_torque, Fix128::from_int(100));
    assert_eq!(c.num_joints, 8);
    assert_eq!(c.num_bodies, 9);
    assert_eq!(c.features_per_body, FEATURES_PER_BODY);
    assert_eq!(FEATURES_PER_BODY, 13);
    // the default is self-consistent: a network of the matching shape is accepted
    let net = DeterministicNetwork::new(
        vec![identity_layer(
            c.num_joints * 3,
            c.num_bodies * c.features_per_body,
        )],
        vec![Activation::None],
    );
    let ctrl = RagdollController::new(net, c);
    assert_eq!(ctrl.config().num_joints, 8);
}
