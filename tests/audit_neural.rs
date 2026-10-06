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
fn tanh_approx_error_near_zero_is_below_the_f64_resolution() {
    // The Pade [7/6] approximant agrees with the Taylor series of tanh through
    // x^13, so its error starts at x^15 / O(10^11): below 1e-15 for |x| <= 1/2
    // (f64 tanh itself is good to about 1e-16 there), and 1.6e-12 at x = 1.
    for k in 1..=16 {
        let x = k as f64 / 32.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 32));
        assert!(
            (got - x.tanh()).abs() < 1e-15,
            "x = {x}: err {}",
            got - x.tanh()
        );
    }
    let at_one = tanh_approx_at(Fix128::ONE) - 1.0_f64.tanh();
    assert!(
        at_one.abs() < 2e-12 && at_one.abs() > 1e-12,
        "x = 1: err {at_one}"
    );
}

#[test]
fn tanh_approx_stays_within_three_hundredths_of_tanh_up_to_four() {
    // A coarse bound kept from the earlier formula: any rational in this
    // accuracy class passes; the documented bound is pinned below.
    let mut worst = 0.0_f64;
    for k in 0..=320 {
        let xf = k as f64 / 80.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 80));
        worst = worst.max((got - xf.tanh()).abs());
    }
    assert!(worst < 0.03, "worst error {worst}");
}

#[test]
fn tanh_approx_clamps_to_unit_beyond_nine_halves() {
    assert_eq!(tanh_approx_at(Fix128::from_ratio(46, 10)), 1.0);
    assert_eq!(tanh_approx_at(Fix128::from_ratio(-46, 10)), -1.0);
    // 4.1 is inside the ratio's range now, not clamped
    assert!(tanh_approx_at(Fix128::from_ratio(41, 10)) < 1.0);
}

#[test]
fn tanh_approx_max_error_matches_the_documented_bound() {
    // AUD-A-S3W2-003: the doc claims a Pade approximant with max error about
    // 2.5e-4; the worst point is just past the 9/2 clamp, 1 - tanh(4.5) = 2.47e-4.
    let mut worst = 0.0_f64;
    let mut at = 0.0_f64;
    for k in 0..=800 {
        let xf = k as f64 / 80.0;
        let got = tanh_approx_at(Fix128::from_ratio(k, 80));
        let e = (got - xf.tanh()).abs();
        if e > worst {
            worst = e;
            at = xf;
        }
    }
    assert!(worst < 2.5e-4, "worst error {worst} at x = {at}");
}

#[test]
// AUD-A-S3W2-004
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
// AUD-A-S3W2-007: an empty network has no outputs
fn zero_layer_network_forward_does_not_panic() {
    let mut net = DeterministicNetwork::new(vec![], vec![]);
    let r = catch_unwind(AssertUnwindSafe(|| {
        assert!(net.forward(&[]).is_empty());
    }));
    assert!(r.is_ok());
    assert_eq!(net.input_size(), 0);
    assert_eq!(net.output_size(), 0);
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
// AUD-A-S3W2-016: the doc promised no rounding; the sum is exact and the scale
// multiply rounds down to the 2^-64 grid, as the doc now says
fn matvec_output_is_the_floor_of_the_exact_product_of_the_sum_and_the_scale() {
    // exact value: raw_x * raw_scale / 2^64 (raw_scale = f32 0.1 widened exactly),
    // floored to the 2^-64 grid; 1/3 * 0.1 is not on the grid, so the floor is
    // strictly below the exact value
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
    assert_ne!(
        exact_num & ((1_i128 << 64) - 1),
        0,
        "the product is off the grid"
    );
    assert_eq!(
        raw(out[0]),
        exact_num >> 64,
        "not the floor of the exact product"
    );
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
