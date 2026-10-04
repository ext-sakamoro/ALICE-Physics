//! Oracles for the wiring of `neural`: the Fix128 activation kernels, the
//! ternary matrix-vector multiply, `DeterministicNetwork`, and the
//! `RagdollController` that wraps it.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * `fix128_relu(x) = max(0, x)`: negative clamps to `0`, non-negative is
//!   unchanged (checked at the exact `-1/0/+1` and extreme-magnitude points).
//! * `fix128_hard_tanh(x) = clamp(x, -1, 1)`: the comparisons in the source
//!   are strict `>` / `<`, so `x = +-1` passes through unchanged and only
//!   `|x| > 1` is clamped.
//! * `fix128_tanh_approx(x) = x*(27+x^2)/(27+9x^2)` for `|x| <= 4`, else
//!   `+-1`. At `x = 3`: `3*(27+9)/(27+81) = 108/108 = 1` exactly, and by the
//!   formula's oddness `x = -3` gives exactly `-1`. At `x = 0` the formula is
//!   `0/27 = 0` exactly. `x = 4` is *not* claimed by the clamp branch
//!   (`4 > 4` is false), so it still evaluates the Padé ratio
//!   `4*43/171 = 172/171`, computed independently here with the same
//!   `Fix128` primitives (not by calling the function under test).
//! * `fix128_leaky_relu(x, alpha) = x if x >= 0 else alpha*x`: the negative
//!   branch's expected value is `alpha * x`, computed in the test via the
//!   `Fix128` `Mul` operator directly (the same primitive the function
//!   composes, not the function itself) — this is the established idiom
//!   already used by `src/neural.rs`'s own unit tests.
//! * `fix128_ternary_matvec`: with ternary entries restricted to `{-1,0,1}`
//!   the accumulation is pure add/sub, so for
//!   `[[+1,-1,0],[0,+1,+1]] * [7,3,5] = [7-3, 3+5] = [4, 8]` (unscaled); a
//!   scale of `3/4` (an exact power-of-two-denominator fraction) gives
//!   `[3, 6]` bit-exactly.
//! * `DeterministicNetwork::forward`: a hand-built 2-layer net
//!   (`3 -> 2` ReLU `-> 2` TanhApprox) on input `[3,0,0]` gives layer-1 raw
//!   `[3,0]` (ReLU leaves both unchanged), layer-2 raw `[3,-3]`, and
//!   `TanhApprox(3)=1`, `TanhApprox(-3)=-1` exactly (the same Padé root used
//!   above), so `forward() == [1,-1]` bit-exactly.
//! * `RagdollController::compute`: a hand-built `13 -> 4` ReLU `-> 3`
//!   HardTanh network driven by a single body with `velocity = (2,4,6)` and
//!   `angular_velocity.z = 5` gives layer-1 raw `[2,4,-6,5]`, ReLU
//!   `[2,4,0,5]`, layer-2 raw `[2,-4,-5]`, scale `1/8` gives
//!   `[0.25,-0.5,-0.625]` (all within `[-1,1]`, so HardTanh and the
//!   `max_torque = 1` clamp are both no-ops).
//! * `FixedTernaryWeight::from_ternary_weight` reads the wrapped
//!   `TernaryWeight`'s `f32` `scale()` (here `2.5`, exact in `f64`/`Fix128`);
//!   `from_ternary_weight_with_scale` stores the given `Fix128` verbatim
//!   (here `7`) and never consults the wrapped weight's `f32` scale — the
//!   two constructors must therefore disagree on the same `TernaryWeight`.
//!
//! # Degenerate inputs (documented result, not just "no panic")
//!
//! * Extreme-magnitude `Fix128` inputs (`from_raw(i64::MAX, u64::MAX)` /
//!   `from_raw(i64::MIN, 0)`) never panic in any activation: `relu` and
//!   `hard_tanh` only compare (no arithmetic), and `tanh_approx`'s clamp
//!   branch is checked *before* the squaring multiply, so the risky
//!   multiply is never reached for out-of-range `x`.
//! * `fix128_leaky_relu` and `fix128_ternary_matvec` use `Fix128::Mul`/`Add`,
//!   which are wrapping (`Fix128` is a 128-bit wrapping group, not a
//!   saturating one) — extreme-magnitude operands wrap silently instead of
//!   panicking, and the wrapped value (not just "no panic") is asserted.
//! * `fix128_ternary_matvec` with a 0x0 matrix, an output buffer shorter
//!   than `out_features`, or an input slice shorter than `in_features`: the
//!   function iterates `.take(out_n)` / `.take(in_n)`, so it silently
//!   computes only the rows that fit the output buffer, and treats missing
//!   input columns as if they contributed `0` to the sum (mathematically
//!   the same as zero-padding) — never a panic or an `Err`.
//! * A layer whose ternary weight is entirely `Ternary::Zero`: this
//!   architecture has no bias term, so the accumulator is always `0` and
//!   the output is exactly `ZERO` regardless of the scale or the input.
//! * `DeterministicNetwork::new(vec![], vec![])` succeeds (`num_layers() ==
//!   0`), but `forward()` on the resulting zero-layer network **panics**
//!   (index out of bounds on `buf_offsets[1]` / `buf_offsets[n - 1]`) — this
//!   is pinned as the current, undocumented behaviour, not fixed here (a
//!   fix would be a design decision: validate at `new()`, or make
//!   `forward()` return `&[]`).
//! * `RagdollController::compute(&[])` (fewer bodies than configured):
//!   `extract_features` zero-fills the whole input buffer, and because this
//!   architecture has no bias term, the output is exactly `ZERO` for every
//!   joint axis — true for *any* network weights, not just the one tested
//!   here.
//! * `RagdollController::compute` with *more* bodies than `config.num_bodies`:
//!   `extract_features` takes only `num_bodies.min(bodies.len())`, so the
//!   extra bodies are silently ignored — the output is identical to the
//!   call without them.
//! * `num_joints = 0` (and a network whose last layer has `out_features =
//!   0`): `RagdollController::new` accepts it (`0 == 0 * 3`), and
//!   `compute()` returns a `ControllerOutput` with an empty `torques` Vec.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "std", feature = "neural"))]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_ml::TernaryWeight;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::neural::{
    fix128_hard_tanh, fix128_leaky_relu, fix128_relu, fix128_tanh_approx, fix128_ternary_matvec,
    Activation, ControllerConfig, ControllerOutput, DeterministicNetwork, FixedTernaryWeight,
    RagdollController,
};
use alice_physics::solver::RigidBody;

// ---------------------------------------------------------------------------
// Activation kernels
// ---------------------------------------------------------------------------

#[test]
fn fix128_relu_characteristic_points_and_extremes() {
    let mut values = [
        Fix128::from_int(-7),
        Fix128::ZERO,
        Fix128::from_int(7),
        Fix128::from_raw(i64::MIN, 0),
        Fix128::from_raw(i64::MAX, u64::MAX),
    ];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_relu(&mut values);
    }));
    assert!(r.is_ok(), "relu is comparison-only, must never panic");
    assert_eq!(values[0], Fix128::ZERO, "negative clamps to 0");
    assert_eq!(values[1], Fix128::ZERO, "0 stays 0");
    assert_eq!(values[2], Fix128::from_int(7), "non-negative is unchanged");
    assert_eq!(
        values[3],
        Fix128::ZERO,
        "extreme negative magnitude clamps to 0"
    );
    assert_eq!(
        values[4],
        Fix128::from_raw(i64::MAX, u64::MAX),
        "extreme positive magnitude is unchanged"
    );
}

#[test]
fn fix128_hard_tanh_clamps_strictly_outside_unit_interval() {
    let mut values = [
        Fix128::ONE,                          // boundary: unchanged
        Fix128::NEG_ONE,                      // boundary: unchanged
        Fix128::from_ratio(1, 2),             // inside: unchanged
        Fix128::from_int(2),                  // outside: clamp to 1
        Fix128::from_int(-2),                 // outside: clamp to -1
        Fix128::from_raw(i64::MAX, u64::MAX), // extreme: clamp to 1
        Fix128::from_raw(i64::MIN, 0),        // extreme: clamp to -1
    ];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_hard_tanh(&mut values);
    }));
    assert!(r.is_ok(), "hard_tanh is comparison-only, must never panic");
    assert_eq!(values[0], Fix128::ONE, "x == 1 passes through (strict >)");
    assert_eq!(
        values[1],
        Fix128::NEG_ONE,
        "x == -1 passes through (strict <)"
    );
    assert_eq!(values[2], Fix128::from_ratio(1, 2));
    assert_eq!(values[3], Fix128::ONE);
    assert_eq!(values[4], Fix128::NEG_ONE);
    assert_eq!(values[5], Fix128::ONE, "extreme magnitude still clamps");
    assert_eq!(values[6], Fix128::NEG_ONE, "extreme magnitude still clamps");
}

#[test]
fn fix128_tanh_approx_exact_pade_roots_and_clamp_branch() {
    // x = 0: 0*(27+0)/(27+0) = 0 exactly.
    // x = 3: 3*(27+9)/(27+81) = 108/108 = 1 exactly.
    // x = -3: oddness of the formula gives exactly -1.
    // x = 10 / -10: beyond the |x| > 4 clamp guard, independent of the Padé
    //   ratio entirely.
    let mut values = [
        Fix128::ZERO,
        Fix128::from_int(3),
        Fix128::from_int(-3),
        Fix128::from_int(10),
        Fix128::from_int(-10),
    ];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_tanh_approx(&mut values);
    }));
    assert!(r.is_ok());
    assert_eq!(values[0], Fix128::ZERO);
    assert_eq!(values[1], Fix128::ONE);
    assert_eq!(values[2], Fix128::NEG_ONE);
    assert_eq!(values[3], Fix128::ONE, "x > 4 clamps to 1");
    assert_eq!(values[4], Fix128::NEG_ONE, "x < -4 clamps to -1");
}

#[test]
fn fix128_tanh_approx_boundary_x_equals_four_is_not_clamped() {
    // x = 4 fails the strict `x > 4` clamp guard, so it still evaluates the
    // Padé ratio: 4*(27+16)/(27+144) = 4*43/171 = 172/171. Computed here
    // independently with the same Fix128 primitives (Add/Mul/Div), not by
    // calling fix128_tanh_approx.
    let x = Fix128::from_int(4);
    let c27 = Fix128::from_int(27);
    let c9 = Fix128::from_int(9);
    let x2 = x * x;
    let expected = (x * (c27 + x2)) / (c27 + c9 * x2);
    assert_eq!(expected, Fix128::from_int(172) / Fix128::from_int(171));
    assert_ne!(
        expected,
        Fix128::ONE,
        "the boundary is NOT clamped, so it must not equal exactly 1"
    );

    let mut values = [x];
    fix128_tanh_approx(&mut values);
    assert_eq!(
        values[0], expected,
        "x == 4 takes the Pade branch, not the clamp branch"
    );
}

#[test]
fn fix128_tanh_approx_extreme_magnitude_never_reaches_the_squaring_multiply() {
    // The clamp check `x > 4` / `x < -4` runs before x*x, so extreme
    // magnitudes can never overflow the squaring multiply — they are always
    // claimed by the clamp branch.
    let mut values = [
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::from_raw(i64::MIN, 0),
    ];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_tanh_approx(&mut values);
    }));
    assert!(r.is_ok());
    assert_eq!(values[0], Fix128::ONE);
    assert_eq!(values[1], Fix128::NEG_ONE);
}

#[test]
fn fix128_leaky_relu_slope_is_alpha_times_x_for_negative_inputs() {
    let alpha = Fix128::from_ratio(1, 8); // exact power-of-two fraction
    let x_pos = Fix128::from_int(8);
    let x_neg = Fix128::from_int(-8);
    let x_zero = Fix128::ZERO;

    // alpha = 0: the negative branch degenerates to 0 (hard ReLU).
    let mut v0 = [x_pos, x_neg, x_zero];
    fix128_leaky_relu(&mut v0, Fix128::ZERO);
    assert_eq!(v0, [x_pos, Fix128::ZERO, x_zero]);

    // alpha = 1: the negative branch becomes the identity (no leak at all),
    // so the function is just a no-op for every sign.
    let mut v1 = [x_pos, x_neg, x_zero];
    fix128_leaky_relu(&mut v1, Fix128::ONE);
    assert_eq!(v1, [x_pos, x_neg, x_zero]);

    // alpha = 1/8: expected value is alpha * x, computed with the same
    // Fix128 Mul primitive the function composes (not by calling the
    // function under test).
    let mut v2 = [x_pos, x_neg, x_zero];
    fix128_leaky_relu(&mut v2, alpha);
    assert_eq!(v2, [x_pos, x_neg * alpha, x_zero]);
    assert_eq!(v2[1], Fix128::from_int(-1), "−8 * 1/8 = −1 exactly");
}

#[test]
fn fix128_leaky_relu_extreme_alpha_wraps_instead_of_panicking() {
    // Fix128 Mul is a wrapping 128-bit operation, not saturating, so an
    // extreme alpha wraps the product around instead of panicking. The
    // expected value is computed with the same Mul primitive (the point is
    // to pin *what* it wraps to, not merely that it doesn't crash).
    let huge_alpha = Fix128::from_raw(i64::MAX, u64::MAX);
    let x_neg = Fix128::from_int(-3);
    let expected = x_neg * huge_alpha;

    let mut values = [x_neg];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_leaky_relu(&mut values, huge_alpha);
    }));
    assert!(r.is_ok(), "wrapping Mul must not panic");
    assert_eq!(values[0], expected);
}

// ---------------------------------------------------------------------------
// fix128_ternary_matvec
// ---------------------------------------------------------------------------

#[test]
fn fix128_ternary_matvec_is_an_exact_dot_product() {
    // [[+1,-1,0],[0,+1,+1]] * [7,3,5] = [7-3, 3+5] = [4,8] unscaled.
    // scale = 3/4 (exact power-of-two-denominator fraction) -> [3,6].
    let w = TernaryWeight::from_ternary(&[1, -1, 0, 0, 1, 1], 2, 3);
    let scale = Fix128::from_ratio(3, 4);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, scale);
    let input = [
        Fix128::from_int(7),
        Fix128::from_int(3),
        Fix128::from_int(5),
    ];
    let mut output = [Fix128::ZERO; 2];
    fix128_ternary_matvec(&input, &ftw, &mut output);
    assert_eq!(output, [Fix128::from_int(3), Fix128::from_int(6)]);
}

#[test]
fn fix128_ternary_matvec_zero_sized_matrix_is_a_no_op() {
    let w = TernaryWeight::from_ternary(&[], 0, 0);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, Fix128::from_int(99));
    let input: [Fix128; 0] = [];
    let mut output: [Fix128; 0] = [];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_ternary_matvec(&input, &ftw, &mut output);
    }));
    assert!(r.is_ok(), "0x0 matvec must not panic");
    assert_eq!(output.len(), 0);
}

#[test]
fn fix128_ternary_matvec_short_output_buffer_computes_only_what_fits() {
    // out_features = 2, but the caller only provides room for 1 output —
    // the function silently stops at `.take(out_n)` clamped by the slice
    // length, it never panics or errors.
    let w = TernaryWeight::from_ternary(&[1, -1, 0, 0, 1, 1], 2, 3);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, Fix128::ONE);
    let input = [
        Fix128::from_int(7),
        Fix128::from_int(3),
        Fix128::from_int(5),
    ];
    let mut output = [Fix128::ZERO; 1];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_ternary_matvec(&input, &ftw, &mut output);
    }));
    assert!(r.is_ok());
    assert_eq!(output[0], Fix128::from_int(4), "row 0 only: 7 - 3 = 4");
}

#[test]
fn fix128_ternary_matvec_short_input_treats_missing_columns_as_zero() {
    // in_features = 3, but only 1 input element is provided — the loop's
    // `.take(in_n)` is clamped by the input slice length, so the missing
    // columns contribute nothing to the sum (the same result as if they
    // had been zero-padded).
    let w = TernaryWeight::from_ternary(&[1, -1, 0, 0, 1, 1], 2, 3);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, Fix128::ONE);
    let input = [Fix128::from_int(7)]; // column 1 and 2 are "missing"
    let mut output = [Fix128::ZERO; 2];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_ternary_matvec(&input, &ftw, &mut output);
    }));
    assert!(r.is_ok());
    assert_eq!(
        output[0],
        Fix128::from_int(7),
        "row0: +1*7 (cols 1,2 missing)"
    );
    assert_eq!(
        output[1],
        Fix128::ZERO,
        "row1: col0 weight is 0, cols 1,2 missing"
    );
}

#[test]
fn fix128_ternary_matvec_extreme_magnitude_accumulation_wraps() {
    // Both weight entries are +1, so the accumulator sums the same extreme
    // value with itself: a*a is not computed (ternary matvec never
    // multiplies two Fix128 values), but a+a overflows the 128-bit range
    // and wraps per Fix128::Add's documented wrapping semantics. The
    // expected value is computed with the same Add primitive directly (not
    // by calling fix128_ternary_matvec).
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);
    let expected_wrapped_sum = huge + huge;

    let w = TernaryWeight::from_ternary(&[1, 1], 1, 2);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, Fix128::ONE);
    let input = [huge, huge];
    let mut output = [Fix128::ZERO; 1];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fix128_ternary_matvec(&input, &ftw, &mut output);
    }));
    assert!(r.is_ok(), "wrapping Add must not panic");
    assert_eq!(output[0], expected_wrapped_sum);
}

// ---------------------------------------------------------------------------
// FixedTernaryWeight constructors
// ---------------------------------------------------------------------------

#[test]
fn from_ternary_weight_reads_the_wrapped_f32_scale_from_ternary_weight_with_scale_does_not() {
    let base = TernaryWeight::from_ternary(&[1, -1, 0, 0, 1, 1], 2, 3);
    let w_scaled = TernaryWeight::from_packed(base.packed().to_vec(), 2, 3, 2.5);

    let ftw_a = FixedTernaryWeight::from_ternary_weight(w_scaled.clone());
    assert_eq!(
        ftw_a.scale(),
        Fix128::from_f64(2.5),
        "from_ternary_weight consults the wrapped f32 scale()"
    );

    let ftw_b = FixedTernaryWeight::from_ternary_weight_with_scale(w_scaled, Fix128::from_int(7));
    assert_eq!(
        ftw_b.scale(),
        Fix128::from_int(7),
        "from_ternary_weight_with_scale stores the explicit scale verbatim"
    );
    assert_ne!(
        ftw_a.scale(),
        ftw_b.scale(),
        "the two constructors must disagree on the same TernaryWeight"
    );

    // Linearity sanity check: the raw (unscaled) ternary sums are
    // row0 = 5-3 = 2, row1 = 3+7 = 10.
    let input = [
        Fix128::from_int(5),
        Fix128::from_int(3),
        Fix128::from_int(7),
    ];
    let mut out_a = [Fix128::ZERO; 2];
    fix128_ternary_matvec(&input, &ftw_a, &mut out_a);
    assert_eq!(out_a, [Fix128::from_int(5), Fix128::from_int(25)]);

    let mut out_b = [Fix128::ZERO; 2];
    fix128_ternary_matvec(&input, &ftw_b, &mut out_b);
    assert_eq!(out_b, [Fix128::from_int(14), Fix128::from_int(70)]);
}

// ---------------------------------------------------------------------------
// DeterministicNetwork
// ---------------------------------------------------------------------------

fn build_tiny_relu_tanh_network() -> DeterministicNetwork {
    // Layer 1: 3 -> 2, ReLU. Row0 = [+1,0,0], Row1 = [0,0,-1].
    let w1 = TernaryWeight::from_ternary(&[1, 0, 0, 0, 0, -1], 2, 3);
    let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);
    // Layer 2: 2 -> 2, TanhApprox. Row0 = [+1,0], Row1 = [-1,0].
    let w2 = TernaryWeight::from_ternary(&[1, 0, -1, 0], 2, 2);
    let ftw2 = FixedTernaryWeight::from_ternary_weight(w2);
    DeterministicNetwork::new(
        vec![ftw1, ftw2],
        vec![Activation::ReLU, Activation::TanhApprox],
    )
}

#[test]
fn deterministic_network_forward_matches_hand_derived_layers() {
    let mut net = build_tiny_relu_tanh_network();
    assert_eq!(net.num_layers(), 2);

    // input = [3,0,0]
    // layer1 raw = [3*1+0+0, 0+0+0*-1] = [3,0]; ReLU leaves both unchanged.
    // layer2 raw = [3*1+0*0, 3*-1+0*0] = [3,-3];
    // TanhApprox(3) = 1, TanhApprox(-3) = -1 exactly (same Pade root as
    // the activation-kernel test above).
    let input = [Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO];
    let out = net.forward(&input);
    assert_eq!(out, [Fix128::ONE, Fix128::NEG_ONE]);
}

// PIN: AUD-A-S3W2-007
#[test]
fn deterministic_network_zero_layers_constructs_but_forward_panics() {
    let mut net = DeterministicNetwork::new(vec![], vec![]);
    assert_eq!(net.num_layers(), 0, "an empty network is constructible");

    // forward() unconditionally indexes buf_offsets[1] (and later
    // buf_offsets[n - 1] with n = 0, an underflow), neither of which exist
    // for a zero-layer network. This is the CURRENT behaviour — pinned as
    // a documented gap, not silently patched here (changing it is a design
    // decision: validate eagerly in `new()`, or make `forward()` return
    // `&[]`).
    let r = catch_unwind(AssertUnwindSafe(|| {
        let _ = net.forward(&[]);
    }));
    assert!(
        r.is_err(),
        "forward() on a zero-layer network panics (index out of bounds), it does not return &[]"
    );
}

#[test]
fn deterministic_network_all_zero_weight_layer_outputs_zero_regardless_of_scale_or_input() {
    // All-Ternary::Zero row: the accumulator is always 0 (no bias term
    // exists in this architecture), so the scaled output is exactly ZERO
    // no matter the scale or the input values.
    let w = TernaryWeight::from_ternary(&[0, 0, 0, 0, 0, 0], 2, 3);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, Fix128::from_int(1_000_000));
    let mut net = DeterministicNetwork::new(vec![ftw], vec![Activation::None]);

    let input = [
        Fix128::from_int(-999),
        Fix128::from_int(999),
        Fix128::from_raw(i64::MAX, u64::MAX),
    ];
    let out = net.forward(&input);
    assert_eq!(out, [Fix128::ZERO, Fix128::ZERO]);
}

// ---------------------------------------------------------------------------
// RagdollController
// ---------------------------------------------------------------------------

/// `13 -> 4` (ReLU) `-> 3` (HardTanh) network, one joint, one body.
///
/// Row selections (see module doc for the derivation):
/// layer1 picks out velocity.x/.y/.z and angular_velocity.z;
/// layer2 maps the ReLU'd hidden vector to 3 torque axes with scale 1/8.
fn build_controller_with_max_torque(max_torque: Fix128) -> RagdollController {
    let mut w1_values = [0i8; 52]; // 4 rows * 13 cols
    w1_values[3] = 1; // row0 . velocity.x   (col 3)
    w1_values[17] = 1; // row1 . velocity.y   (col 4, offset 13+4)
    w1_values[31] = -1; // row2 . velocity.z   (col 5, offset 26+5)
    w1_values[51] = 1; // row3 . angular_velocity.z (col 12, offset 39+12)
    let w1 = TernaryWeight::from_ternary(&w1_values, 4, 13);
    let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);

    // row1 depends on BOTH col1 and col2: col2 (hidden[2]) is 0 after a
    // correctly-applied ReLU on layer 1's negative raw value (-6), so this
    // row's value is unchanged from a col1-only design in the passing case,
    // but it exposes a skipped/short-circuited ReLU on layer 1 (hidden[2]
    // would be -6 instead of 0, changing this row's dot product).
    let w2_values = [1i8, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1];
    let w2 = TernaryWeight::from_ternary(&w2_values, 3, 4);
    let ftw2 = FixedTernaryWeight::from_ternary_weight_with_scale(w2, Fix128::from_ratio(1, 8));

    let network = DeterministicNetwork::new(
        vec![ftw1, ftw2],
        vec![Activation::ReLU, Activation::HardTanh],
    );
    let config = ControllerConfig {
        max_torque,
        num_joints: 1,
        num_bodies: 1,
        features_per_body: alice_physics::neural::FEATURES_PER_BODY,
    };
    RagdollController::new(network, config)
}

fn build_controller() -> RagdollController {
    build_controller_with_max_torque(Fix128::ONE)
}

fn body_with_velocity_and_angular_velocity(v: (i64, i64, i64), w: (i64, i64, i64)) -> RigidBody {
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(v.0, v.1, v.2);
    body.angular_velocity = Vec3Fix::from_int(w.0, w.1, w.2);
    body
}

#[test]
fn ragdoll_controller_accessors_and_compute_match_hand_derived_torque() {
    let mut controller = build_controller();
    assert_eq!(controller.network().num_layers(), 2);
    assert_eq!(controller.config().num_joints, 1);
    assert_eq!(controller.config().num_bodies, 1);
    assert_eq!(controller.config().max_torque, Fix128::ONE);

    // velocity = (2,4,6), angular_velocity.z = 5:
    // layer1 raw = [2, 4, -6, 5]; ReLU -> [2, 4, 0, 5]
    // layer2 raw = [2, -4, -5]; scale 1/8 -> [0.25, -0.5, -0.625]
    // all within [-1,1], so HardTanh and the max_torque=1 clamp are no-ops.
    let body = body_with_velocity_and_angular_velocity((2, 4, 6), (0, 0, 5));
    let output: ControllerOutput = controller.compute(&[body]);
    let torque = output.torques[0];
    assert_eq!(torque.x, Fix128::from_ratio(1, 4));
    assert_eq!(torque.y, Fix128::from_ratio(-1, 2));
    assert_eq!(torque.z, Fix128::from_ratio(-5, 8));
}

#[test]
fn ragdoll_controller_compute_with_empty_bodies_yields_zero_torque() {
    let mut controller = build_controller();
    let output = controller.compute(&[]);
    let torque = output.torques[0];
    assert_eq!(
        torque,
        Vec3Fix::ZERO,
        "no bias term exists, so zero-filled features always yield zero torque"
    );
}

#[test]
fn ragdoll_controller_compute_ignores_bodies_beyond_num_bodies() {
    let mut controller_a = build_controller();
    let mut controller_b = build_controller();
    let body0 = body_with_velocity_and_angular_velocity((2, 4, 6), (0, 0, 5));
    let body1 = body_with_velocity_and_angular_velocity((100, 200, 300), (9, 9, 9));

    let with_one = controller_a.compute(&[body0]);
    let with_two = controller_b.compute(&[body0, body1]);
    assert_eq!(
        with_one.torques[0], with_two.torques[0],
        "config.num_bodies == 1, so the second body must be silently ignored"
    );
}

#[test]
fn ragdoll_controller_clamps_torque_to_max_torque() {
    // velocity.x = 100 drives layer1 row0 to 100; ReLU keeps it; layer2
    // row0 scales by 1/8 -> 12.5; HardTanh clamps that to exactly 1 (the
    // network's own [-1,1] bound) — which still exceeds max_torque = 1/2,
    // so only the controller's own clamp_fix128 can bring it down further.
    let mut controller = build_controller_with_max_torque(Fix128::from_ratio(1, 2));
    let body = body_with_velocity_and_angular_velocity((100, 0, 0), (0, 0, 0));
    let output = controller.compute(&[body]);
    let torque = output.torques[0];
    assert_eq!(
        torque.x,
        Fix128::from_ratio(1, 2),
        "controller clamp engages beyond HardTanh's own bound"
    );
    assert_eq!(torque.y, Fix128::ZERO);
    assert_eq!(torque.z, Fix128::ZERO);
}

#[test]
fn ragdoll_controller_zero_joints_yields_empty_controller_output() {
    // out_features = 0 on the last layer => num_joints must be 0 (0 == 0*3).
    let w = TernaryWeight::from_ternary(&[], 0, 13);
    let ftw = FixedTernaryWeight::from_ternary_weight(w);
    let network = DeterministicNetwork::new(vec![ftw], vec![Activation::None]);
    let config = ControllerConfig {
        max_torque: Fix128::ONE,
        num_joints: 0,
        num_bodies: 1,
        features_per_body: alice_physics::neural::FEATURES_PER_BODY,
    };
    let mut controller = RagdollController::new(network, config);

    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let output: ControllerOutput = controller.compute(&[body]);
    assert_eq!(output.torques.len(), 0);
}
