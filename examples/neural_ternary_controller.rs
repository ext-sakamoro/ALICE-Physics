//! Deterministic Neural Ternary Controller — production wiring for
//! `src/neural.rs`'s fixed-point ternary-weight controller.
//!
//! Builds a tiny hand-picked 2-layer `DeterministicNetwork`
//! (13 inputs -> 4 hidden [ReLU] -> 3 outputs [HardTanh]), wraps it in a
//! `RagdollController`, and prints the controller's torque output next to
//! the closed-form value derived by hand in the comments below. Also
//! exercises each activation kernel and the raw ternary matvec directly,
//! so every public item in `src/neural.rs` has a production call site.
//!
//! ```bash
//! cargo run --example neural_ternary_controller --features std,neural
//! ```

use alice_ml::TernaryWeight;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::neural::{
    fix128_hard_tanh, fix128_leaky_relu, fix128_relu, fix128_tanh_approx, fix128_ternary_matvec,
    Activation, ControllerConfig, ControllerOutput, DeterministicNetwork, FixedTernaryWeight,
    RagdollController, FEATURES_PER_BODY,
};
use alice_physics::solver::RigidBody;

fn main() {
    println!("[neural] -- direct kernel wiring --");

    // fix128_relu: ReLU(x) = max(0, x).
    let mut relu_vals = [Fix128::from_int(5), Fix128::from_int(-5), Fix128::ZERO];
    fix128_relu(&mut relu_vals);
    println!(
        "[neural] fix128_relu([5,-5,0]) = [{}, {}, {}] (expect [5, 0, 0])",
        relu_vals[0].to_f64(),
        relu_vals[1].to_f64(),
        relu_vals[2].to_f64()
    );
    assert_eq!(relu_vals, [Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO]);

    // fix128_hard_tanh: clamp(x, -1, 1). The comparisons are strict `>` /
    // `<`, so values already at the +-1 boundary pass through unchanged.
    let mut tanh_vals = [
        Fix128::from_int(2),
        Fix128::from_int(-2),
        Fix128::ONE,
        Fix128::NEG_ONE,
    ];
    fix128_hard_tanh(&mut tanh_vals);
    println!(
        "[neural] fix128_hard_tanh([2,-2,1,-1]) = [{}, {}, {}, {}] (expect [1, -1, 1, -1])",
        tanh_vals[0].to_f64(),
        tanh_vals[1].to_f64(),
        tanh_vals[2].to_f64(),
        tanh_vals[3].to_f64()
    );
    assert_eq!(
        tanh_vals,
        [Fix128::ONE, Fix128::NEG_ONE, Fix128::ONE, Fix128::NEG_ONE]
    );

    // fix128_tanh_approx: Pade approximant x*(27+x^2)/(27+9x^2) for |x| <= 4,
    // else clamp to +-1. At x = 3: 3*(27+9)/(27+81) = 108/108 = 1 exactly
    // (and by oddness of the formula, x = -3 gives exactly -1).
    let mut pade_vals = [Fix128::from_int(3), Fix128::from_int(-3), Fix128::ZERO];
    fix128_tanh_approx(&mut pade_vals);
    println!(
        "[neural] fix128_tanh_approx([3,-3,0]) = [{}, {}, {}] (expect [1, -1, 0], exact Pade root)",
        pade_vals[0].to_f64(),
        pade_vals[1].to_f64(),
        pade_vals[2].to_f64()
    );
    assert_eq!(pade_vals, [Fix128::ONE, Fix128::NEG_ONE, Fix128::ZERO]);

    // fix128_leaky_relu: x if x >= 0, alpha * x otherwise. alpha = 1/8 keeps
    // the closed form exact in Fix128 (a power-of-two fraction), unlike the
    // doc's typical alpha = 1/100.
    let alpha = Fix128::from_ratio(1, 8);
    let mut leaky_vals = [Fix128::from_int(8), Fix128::from_int(-8)];
    fix128_leaky_relu(&mut leaky_vals, alpha);
    let expected_leaky = [Fix128::from_int(8), Fix128::from_int(-8) * alpha];
    println!(
        "[neural] fix128_leaky_relu([8,-8], alpha=1/8) = [{}, {}] (expect [8, -1])",
        leaky_vals[0].to_f64(),
        leaky_vals[1].to_f64()
    );
    assert_eq!(leaky_vals, expected_leaky);

    // fix128_ternary_matvec: pure ternary accumulate-then-scale, exact.
    //   [+1, -1,  0]   [7]     [7 - 3]      [4]
    //   [ 0, +1, +1] * [3]  =  [3 + 5]  =   [8]   (unscaled)
    //                  [5]
    // scale = 3/4 (exact power-of-two-denominator fraction) -> [3, 6].
    let w = TernaryWeight::from_ternary(&[1, -1, 0, 0, 1, 1], 2, 3);
    let scale = Fix128::from_ratio(3, 4);
    let ftw = FixedTernaryWeight::from_ternary_weight_with_scale(w, scale);
    let mv_input = [
        Fix128::from_int(7),
        Fix128::from_int(3),
        Fix128::from_int(5),
    ];
    let mut mv_out = [Fix128::ZERO; 2];
    fix128_ternary_matvec(&mv_input, &ftw, &mut mv_out);
    println!(
        "[neural] fix128_ternary_matvec([7,3,5]) = [{}, {}] (expect [3, 6])",
        mv_out[0].to_f64(),
        mv_out[1].to_f64()
    );
    assert_eq!(mv_out, [Fix128::from_int(3), Fix128::from_int(6)]);

    println!("\n[neural] -- DeterministicNetwork + RagdollController wiring --");

    // Layer 1: 13 inputs (the ragdoll feature layout: position[0..3],
    // velocity[3..6], rotation[6..10], angular_velocity[10..13]) -> 4
    // hidden, ReLU. Every row has exactly one nonzero column so each dot
    // product is a single hand-checkable multiply; all other columns are
    // Ternary::Zero (unused).
    let mut w1_values = [0i8; 52]; // 4 rows * 13 cols, row-major
    w1_values[0] = 1; // row0 . position.x   (col 0)
    w1_values[14] = 1; // row1 . position.y   (col 1, offset 13+1)
    w1_values[28] = -1; // row2 . position.z   (col 2, offset 26+2)
    w1_values[48] = 1; // row3 . rotation.w   (col 9, offset 39+9)
    let w1 = TernaryWeight::from_ternary(&w1_values, 4, 13);
    let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);
    println!(
        "[neural] FixedTernaryWeight::from_ternary_weight layer1 scale = {} \
         (default from TernaryWeight::scale(), unscaled i8 input)",
        ftw1.scale().to_f64()
    );

    // Layer 2: 4 hidden -> 3 outputs (one joint, 3 torque axes), HardTanh.
    let w2_values = [1i8, 1, 0, -1, -1, 0, 0, 1, 0, -1, 1, 1];
    let w2 = TernaryWeight::from_ternary(&w2_values, 3, 4);
    let scale2 = Fix128::from_ratio(1, 4);
    let ftw2 = FixedTernaryWeight::from_ternary_weight_with_scale(w2, scale2);

    let mut network = DeterministicNetwork::new(
        vec![ftw1, ftw2],
        vec![Activation::ReLU, Activation::HardTanh],
    );
    println!("[neural] network.num_layers() = {}", network.num_layers());
    assert_eq!(network.num_layers(), 2);

    // Hand-picked body: position (1,2,3), velocity/angular_velocity zero,
    // rotation identity (so rotation.w = 1). Feature vector is therefore
    // [1,2,3, 0,0,0, 0,0,0,1, 0,0,0].
    //
    // layer1 raw = [1*1, 2*1, 3*-1, 1*1] = [1, 2, -3, 1]; ReLU -> [1, 2, 0, 1]
    // layer2 raw = [1+2+0-1, -1+0+0+1, 0-2+0+1] = [2, 0, -1]; * scale(1/4)
    //            = [0.5, 0, -0.25]; HardTanh leaves all three unchanged
    //            (all within [-1, 1]).
    let raw_input = [
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_int(3),
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ONE,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
    ];
    let expected_forward = [
        Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::from_ratio(-1, 4),
    ];
    let raw_out = network.forward(&raw_input).to_vec();
    println!(
        "[neural] network.forward(hand input) = [{}, {}, {}] (expect [0.5, 0, -0.25])",
        raw_out[0].to_f64(),
        raw_out[1].to_f64(),
        raw_out[2].to_f64()
    );
    assert_eq!(raw_out, expected_forward);

    let config = ControllerConfig {
        max_torque: Fix128::from_int(10),
        num_joints: 1,
        num_bodies: 1,
        features_per_body: FEATURES_PER_BODY,
    };
    let mut controller = RagdollController::new(network, config);
    println!(
        "[neural] controller.config() = max_torque {} num_joints {} num_bodies {}",
        controller.config().max_torque.to_f64(),
        controller.config().num_joints,
        controller.config().num_bodies
    );
    println!(
        "[neural] controller.network().num_layers() = {}",
        controller.network().num_layers()
    );

    let body = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);
    let bodies = [body];
    let output: ControllerOutput = controller.compute(&bodies);
    let torque = output.torques[0];
    println!(
        "[neural] controller.compute(1 body).torques[0] = ({}, {}, {}) (expect (0.5, 0, -0.25))",
        torque.x.to_f64(),
        torque.y.to_f64(),
        torque.z.to_f64()
    );
    assert_eq!(torque.x, Fix128::from_ratio(1, 2));
    assert_eq!(torque.y, Fix128::ZERO);
    assert_eq!(torque.z, Fix128::from_ratio(-1, 4));

    println!("\n[neural] all closed-form checks passed.");
}
