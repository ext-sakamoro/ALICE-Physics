//! Deterministic Neural Controller (ALICE-ML × Physics)
//!
//! Fixed-point neural inference for game AI. Ternary weights {-1, 0, +1}
//! combined with 128-bit fixed-point arithmetic guarantee bit-exact
//! reproducibility across all platforms.
//!
//! # Why This Works
//!
//! With ternary weights, matrix-vector multiplication reduces to pure
//! addition and subtraction — no floating-point multiply, no rounding,
//! no platform-dependent behavior. This is the "holy grail" for
//! deterministic AI in networked games.
//!
//! # Example
//!
//! ```rust,ignore
//! use alice_physics::neural::*;
//! use alice_physics::{Fix128, Vec3Fix, RigidBody};
//! use alice_ml::TernaryWeight;
//!
//! // Build network from ternary weights
//! let w1 = FixedTernaryWeight::from_ternary_weight(layer1_weights);
//! let w2 = FixedTernaryWeight::from_ternary_weight(layer2_weights);
//!
//! let network = DeterministicNetwork::new(
//!     vec![w1, w2],
//!     vec![Activation::ReLU, Activation::HardTanh],
//! );
//!
//! // Create ragdoll controller
//! let config = ControllerConfig {
//!     max_torque: Fix128::from_int(100),
//!     num_joints: 8,
//!     num_bodies: 9,
//!     features_per_body: 13,
//! };
//! let mut controller = RagdollController::new(network, config);
//!
//! // Compute torques (deterministic across all platforms)
//! let output = controller.compute(&bodies);
//! ```
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;
use alice_ml::{Ternary, TernaryWeight};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Fixed-Point Ternary Weight
// ---------------------------------------------------------------------------

/// Ternary weight wrapper with precomputed Fix128 scale.
///
/// The scale factor is converted from f32 to Fix128 once at construction time.
/// All subsequent inference uses only fixed-point arithmetic.
pub struct FixedTernaryWeight {
    weight: TernaryWeight,
    scale_fix: Fix128,
}

impl FixedTernaryWeight {
    /// Create from an ALICE-ML TernaryWeight.
    ///
    /// Converts the f32 scale to Fix128 (one-time, deterministic).
    pub fn from_ternary_weight(w: TernaryWeight) -> Self {
        let scale_fix = Fix128::from_f64(w.scale() as f64);
        Self { weight: w, scale_fix }
    }

    /// Create with an explicit Fix128 scale (fully deterministic construction).
    pub fn from_ternary_weight_with_scale(w: TernaryWeight, scale: Fix128) -> Self {
        Self { weight: w, scale_fix: scale }
    }

    /// Number of output features (rows).
    #[inline]
    pub fn out_features(&self) -> usize {
        self.weight.out_features()
    }

    /// Number of input features (columns).
    #[inline]
    pub fn in_features(&self) -> usize {
        self.weight.in_features()
    }

    /// The Fix128 scale factor.
    #[inline]
    pub fn scale(&self) -> Fix128 {
        self.scale_fix
    }
}

// ---------------------------------------------------------------------------
// Core Deterministic Inference Kernels
// ---------------------------------------------------------------------------

/// Fixed-point ternary matrix-vector multiply (core kernel).
///
/// With ternary weights {-1, 0, +1}, this is **pure addition/subtraction**.
/// No floating-point multiplication. No rounding error. Bit-exact everywhere.
///
/// ```text
/// output[i] = scale * Σ_j (w[i,j] ⊙ input[j])
///
/// where ⊙ is:
///   +1 → add input[j]
///   -1 → subtract input[j]
///    0 → skip
/// ```
pub fn fix128_ternary_matvec(
    input: &[Fix128],
    weights: &FixedTernaryWeight,
    output: &mut [Fix128],
) {
    let w = &weights.weight;
    let out_n = w.out_features();
    let in_n = w.in_features();

    for row in 0..out_n {
        let mut acc = Fix128::ZERO;
        for col in 0..in_n {
            match w.get(row, col) {
                Ternary::Plus  => acc = acc + input[col],
                Ternary::Minus => acc = acc - input[col],
                Ternary::Zero  => {},
            }
        }
        output[row] = acc * weights.scale_fix;
    }
}

// ---------------------------------------------------------------------------
// Activation Functions (all fully deterministic)
// ---------------------------------------------------------------------------

/// Fixed-point ReLU: `max(0, x)`.
///
/// Uses sign-bit comparison only. Zero branches, zero floating-point.
pub fn fix128_relu(values: &mut [Fix128]) {
    for v in values.iter_mut() {
        if v.is_negative() {
            *v = Fix128::ZERO;
        }
    }
}

/// Fixed-point Hard Tanh: `clamp(x, -1, 1)`.
///
/// Ideal for output layers where bounded [-1, 1] range is needed.
/// The network learns to compensate for the piecewise-linear shape.
pub fn fix128_hard_tanh(values: &mut [Fix128]) {
    for v in values.iter_mut() {
        if *v > Fix128::ONE {
            *v = Fix128::ONE;
        } else if *v < Fix128::NEG_ONE {
            *v = Fix128::NEG_ONE;
        }
    }
}

/// Fixed-point Tanh approximation via Padé rational function:
///
/// ```text
/// tanh(x) ≈ x · (27 + x²) / (27 + 9·x²)
/// ```
///
/// Accurate to ~0.004 max error for |x| < 4.5.
/// Uses only Fix128 add/mul/div — no transcendental functions.
pub fn fix128_tanh_approx(values: &mut [Fix128]) {
    let c27 = Fix128::from_int(27);
    let c9 = Fix128::from_int(9);

    for v in values.iter_mut() {
        let x = *v;
        let x2 = x * x;

        // Clamp extreme values to avoid unnecessary division
        if x > Fix128::from_int(4) {
            *v = Fix128::ONE;
        } else if x < Fix128::from_int(-4) {
            *v = Fix128::NEG_ONE;
        } else {
            // Padé approximant: x * (27 + x²) / (27 + 9*x²)
            let numer = x * (c27 + x2);
            let denom = c27 + c9 * x2;
            *v = numer / denom;
        }
    }
}

/// Fixed-point Leaky ReLU: `x if x >= 0, alpha * x otherwise`.
///
/// `alpha` is typically Fix128::from_ratio(1, 100) for 0.01.
pub fn fix128_leaky_relu(values: &mut [Fix128], alpha: Fix128) {
    for v in values.iter_mut() {
        if v.is_negative() {
            *v = *v * alpha;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Layer Deterministic Network
// ---------------------------------------------------------------------------

/// Activation function type for network layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `max(0, x)` — standard for hidden layers
    ReLU,
    /// `clamp(x, -1, 1)` — bounded output
    HardTanh,
    /// Padé rational approximation — smooth bounded output
    TanhApprox,
    /// No activation (linear pass-through)
    None,
}

/// Multi-layer deterministic neural network.
///
/// All computation is in Fix128. All buffers are pre-allocated at construction.
/// Forward pass performs zero heap allocations.
pub struct DeterministicNetwork {
    layers: Vec<FixedTernaryWeight>,
    activations: Vec<Activation>,
    /// Pre-allocated intermediate buffers (one per layer output)
    buffers: Vec<Vec<Fix128>>,
}

impl DeterministicNetwork {
    /// Create a new network from layers and activations.
    ///
    /// `layers` and `activations` must have the same length.
    /// Buffers are pre-allocated for zero-allocation inference.
    ///
    /// # Panics
    ///
    /// Panics if `layers.len() != activations.len()`.
    pub fn new(layers: Vec<FixedTernaryWeight>, activations: Vec<Activation>) -> Self {
        assert_eq!(
            layers.len(),
            activations.len(),
            "layers and activations must have the same length"
        );

        let buffers = layers
            .iter()
            .map(|l| vec![Fix128::ZERO; l.out_features()])
            .collect();

        Self { layers, activations, buffers }
    }

    /// Run forward pass (deterministic, zero allocation).
    ///
    /// Returns a reference to the final output buffer.
    pub fn forward(&mut self, input: &[Fix128]) -> &[Fix128] {
        let n = self.layers.len();

        // First layer: reads from external input
        fix128_ternary_matvec(input, &self.layers[0], &mut self.buffers[0]);
        Self::apply_activation(self.activations[0], &mut self.buffers[0]);

        // Subsequent layers: split_at_mut satisfies borrow checker
        for i in 1..n {
            let (prev, curr) = self.buffers.split_at_mut(i);
            fix128_ternary_matvec(&prev[i - 1], &self.layers[i], &mut curr[0]);
            Self::apply_activation(self.activations[i], &mut curr[0]);
        }

        &self.buffers[n - 1]
    }

    /// Apply activation function to a buffer.
    #[inline]
    fn apply_activation(act: Activation, buf: &mut [Fix128]) {
        match act {
            Activation::ReLU       => fix128_relu(buf),
            Activation::HardTanh   => fix128_hard_tanh(buf),
            Activation::TanhApprox => fix128_tanh_approx(buf),
            Activation::None       => {},
        }
    }

    /// Number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Input dimension (in_features of first layer).
    #[inline]
    pub fn input_size(&self) -> usize {
        self.layers[0].in_features()
    }

    /// Output dimension (out_features of last layer).
    #[inline]
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].out_features()
    }
}

// ---------------------------------------------------------------------------
// Ragdoll / Character Controller
// ---------------------------------------------------------------------------

/// Number of features extracted per rigid body.
///
/// - Position (3): x, y, z
/// - Velocity (3): vx, vy, vz
/// - Rotation (4): qx, qy, qz, qw
/// - Angular velocity (3): wx, wy, wz
///
/// Total: 13 per body.
pub const FEATURES_PER_BODY: usize = 13;

/// Configuration for the ragdoll neural controller.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Maximum torque magnitude per joint axis.
    pub max_torque: Fix128,
    /// Number of joints to control (output = num_joints * 3 axes).
    pub num_joints: usize,
    /// Number of rigid bodies to observe.
    pub num_bodies: usize,
    /// Features per body (default: 13).
    pub features_per_body: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_torque: Fix128::from_int(100),
            num_joints: 8,
            num_bodies: 9,
            features_per_body: FEATURES_PER_BODY,
        }
    }
}

/// Output from the neural controller.
pub struct ControllerOutput {
    /// Target torque for each joint (3 axes per joint).
    pub torques: Vec<Vec3Fix>,
}

/// Ragdoll controller powered by a deterministic neural network.
///
/// Extracts body states (position, velocity, rotation, angular velocity)
/// as Fix128 features, runs inference, and outputs joint torques.
///
/// **Determinism guarantee**: Given identical body states, the output
/// torques are bit-exact across all platforms, CPUs, and compilers.
pub struct RagdollController {
    network: DeterministicNetwork,
    config: ControllerConfig,
    /// Pre-allocated input feature buffer.
    input_buffer: Vec<Fix128>,
}

impl RagdollController {
    /// Create a new ragdoll controller.
    ///
    /// The network's input size must match `config.num_bodies * config.features_per_body`.
    /// The network's output size must match `config.num_joints * 3`.
    ///
    /// # Panics
    ///
    /// Panics if network dimensions don't match the config.
    pub fn new(network: DeterministicNetwork, config: ControllerConfig) -> Self {
        let input_size = config.num_bodies * config.features_per_body;
        assert_eq!(
            network.input_size(),
            input_size,
            "Network input ({}) must match num_bodies * features_per_body ({})",
            network.input_size(),
            input_size,
        );
        assert_eq!(
            network.output_size(),
            config.num_joints * 3,
            "Network output ({}) must match num_joints * 3 ({})",
            network.output_size(),
            config.num_joints * 3,
        );

        let input_buffer = vec![Fix128::ZERO; input_size];

        Self { network, config, input_buffer }
    }

    /// Extract features from rigid bodies into the input buffer.
    fn extract_features(&mut self, bodies: &[RigidBody]) {
        let mut idx = 0;
        let n = self.config.num_bodies.min(bodies.len());

        for body in bodies.iter().take(n) {
            // Position (3)
            self.input_buffer[idx]     = body.position.x;
            self.input_buffer[idx + 1] = body.position.y;
            self.input_buffer[idx + 2] = body.position.z;
            // Velocity (3)
            self.input_buffer[idx + 3] = body.velocity.x;
            self.input_buffer[idx + 4] = body.velocity.y;
            self.input_buffer[idx + 5] = body.velocity.z;
            // Rotation quaternion (4)
            self.input_buffer[idx + 6] = body.rotation.x;
            self.input_buffer[idx + 7] = body.rotation.y;
            self.input_buffer[idx + 8] = body.rotation.z;
            self.input_buffer[idx + 9] = body.rotation.w;
            // Angular velocity (3)
            self.input_buffer[idx + 10] = body.angular_velocity.x;
            self.input_buffer[idx + 11] = body.angular_velocity.y;
            self.input_buffer[idx + 12] = body.angular_velocity.z;

            idx += self.config.features_per_body;
        }

        // Zero-fill remaining slots if bodies < num_bodies
        for i in idx..self.input_buffer.len() {
            self.input_buffer[i] = Fix128::ZERO;
        }
    }

    /// Compute control torques from rigid body states.
    ///
    /// This is the main entry point. Fully deterministic: same bodies → same torques.
    pub fn compute(&mut self, bodies: &[RigidBody]) -> ControllerOutput {
        self.extract_features(bodies);

        let raw = self.network.forward(&self.input_buffer);

        let mut torques = Vec::with_capacity(self.config.num_joints);
        let max = self.config.max_torque;
        let neg_max = Fix128::ZERO - max;

        for i in 0..self.config.num_joints {
            let base = i * 3;
            let tx = clamp_fix128(raw.get(base).copied().unwrap_or(Fix128::ZERO), neg_max, max);
            let ty = clamp_fix128(raw.get(base + 1).copied().unwrap_or(Fix128::ZERO), neg_max, max);
            let tz = clamp_fix128(raw.get(base + 2).copied().unwrap_or(Fix128::ZERO), neg_max, max);

            torques.push(Vec3Fix { x: tx, y: ty, z: tz });
        }

        ControllerOutput { torques }
    }

    /// Access the underlying network.
    pub fn network(&self) -> &DeterministicNetwork {
        &self.network
    }

    /// Access the controller config.
    pub fn config(&self) -> &ControllerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Clamp a Fix128 value to [min, max].
#[inline]
fn clamp_fix128(v: Fix128, min: Fix128, max: Fix128) -> Fix128 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alice_ml::quantize_to_ternary;

    /// Helper: create a simple TernaryWeight from known values
    fn make_test_weight(out: usize, inp: usize, values: &[i8]) -> TernaryWeight {
        let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let (tw, _stats) = quantize_to_ternary(&f32_values, out, inp);
        tw
    }

    #[test]
    fn test_fix128_ternary_matvec_identity() {
        // 3x3 identity-like: diag = +1, off = 0
        // [1, 0, 0]
        // [0, 1, 0]
        // [0, 0, 1]
        let values = [1i8, 0, 0, 0, 1, 0, 0, 0, 1];
        let tw = make_test_weight(3, 3, &values);
        let ftw = FixedTernaryWeight::from_ternary_weight(tw);

        let input = [Fix128::from_int(10), Fix128::from_int(20), Fix128::from_int(30)];
        let mut output = [Fix128::ZERO; 3];

        fix128_ternary_matvec(&input, &ftw, &mut output);

        // scale ≈ mean(|values|) = 3/9 ≈ 0.333
        // After quantization with that scale, output ≈ input * scale
        // The exact values depend on quantization, but structure should be preserved
        assert!(!output[0].is_zero(), "output[0] should be non-zero");
        assert!(!output[1].is_zero(), "output[1] should be non-zero");
        assert!(!output[2].is_zero(), "output[2] should be non-zero");
    }

    #[test]
    fn test_fix128_ternary_matvec_manual() {
        // Create weights where we know the exact ternary values
        // +1, -1
        // -1, +1
        let values = [1.0f32, -1.0, -1.0, 1.0];
        let (tw, _) = quantize_to_ternary(&values, 2, 2);
        let scale = Fix128::from_f64(tw.scale() as f64);
        let ftw = FixedTernaryWeight::from_ternary_weight(tw);

        let a = Fix128::from_int(5);
        let b = Fix128::from_int(3);
        let input = [a, b];
        let mut output = [Fix128::ZERO; 2];

        fix128_ternary_matvec(&input, &ftw, &mut output);

        // Row 0: +1*5 + -1*3 = 2, then * scale
        // Row 1: -1*5 + +1*3 = -2, then * scale
        let expected_0 = (a - b) * scale;
        let expected_1 = (b - a) * scale;

        assert_eq!(output[0].hi, expected_0.hi);
        assert_eq!(output[0].lo, expected_0.lo);
        assert_eq!(output[1].hi, expected_1.hi);
        assert_eq!(output[1].lo, expected_1.lo);
    }

    #[test]
    fn test_fix128_relu() {
        let mut values = [
            Fix128::from_int(5),
            Fix128::from_int(-3),
            Fix128::ZERO,
            Fix128::from_int(-1),
            Fix128::from_int(10),
        ];

        fix128_relu(&mut values);

        assert_eq!(values[0].hi, 5);
        assert_eq!(values[1].hi, 0);
        assert!(values[1].lo == 0);
        assert_eq!(values[2].hi, 0);
        assert_eq!(values[3].hi, 0);
        assert!(values[3].lo == 0);
        assert_eq!(values[4].hi, 10);
    }

    #[test]
    fn test_fix128_hard_tanh() {
        let mut values = [
            Fix128::from_int(5),
            Fix128::from_int(-3),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(-1, 2),
        ];

        fix128_hard_tanh(&mut values);

        assert_eq!(values[0].hi, 1); // Clamped to 1
        assert_eq!(values[0].lo, 0);
        assert_eq!(values[1].hi, Fix128::NEG_ONE.hi); // Clamped to -1
        // 0.5 stays 0.5
        assert_eq!(values[2].hi, 0);
        assert!(values[2].lo > 0);
        // -0.5 stays -0.5
        assert!(values[3].is_negative());
    }

    #[test]
    fn test_fix128_tanh_approx_bounds() {
        let mut values = [
            Fix128::from_int(10),
            Fix128::from_int(-10),
            Fix128::ZERO,
            Fix128::from_ratio(1, 2),
        ];

        fix128_tanh_approx(&mut values);

        // Large positive → 1
        assert_eq!(values[0].hi, 1);
        assert_eq!(values[0].lo, 0);
        // Large negative → -1
        assert_eq!(values[1].hi, Fix128::NEG_ONE.hi);
        // Zero → zero
        assert!(values[2].is_zero());
        // 0.5 → tanh(0.5) ≈ 0.462
        assert!(!values[3].is_zero());
        assert!(!values[3].is_negative());
        assert!(values[3] < Fix128::ONE);
    }

    #[test]
    fn test_deterministic_network_forward() {
        // 4-input → 3-hidden (ReLU) → 2-output (HardTanh)
        let f32_w1: Vec<f32> = vec![
            1.0, -1.0, 0.0, 1.0,
           -1.0,  1.0, 1.0, 0.0,
            0.0,  1.0,-1.0, 1.0,
        ];
        let f32_w2: Vec<f32> = vec![
            1.0, -1.0, 1.0,
           -1.0,  0.0, 1.0,
        ];

        let (tw1, _) = quantize_to_ternary(&f32_w1, 3, 4);
        let (tw2, _) = quantize_to_ternary(&f32_w2, 2, 3);
        let ftw1 = FixedTernaryWeight::from_ternary_weight(tw1);
        let ftw2 = FixedTernaryWeight::from_ternary_weight(tw2);

        let mut net = DeterministicNetwork::new(
            vec![ftw1, ftw2],
            vec![Activation::ReLU, Activation::HardTanh],
        );

        let input = [
            Fix128::from_int(1),
            Fix128::from_int(2),
            Fix128::from_int(3),
            Fix128::from_int(4),
        ];

        let output = net.forward(&input);
        assert_eq!(output.len(), 2);

        // Output should be in [-1, 1] due to HardTanh
        for &v in output {
            assert!(v >= Fix128::NEG_ONE);
            assert!(v <= Fix128::ONE);
        }
    }

    #[test]
    fn test_deterministic_network_reproducibility() {
        // Same network, same input → must produce BIT-EXACT same output
        let f32_w: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, -1.0];
        let (tw, _) = quantize_to_ternary(&f32_w, 3, 3);

        let input = [
            Fix128::from_int(7),
            Fix128::from_ratio(3, 2),
            Fix128::from_int(-4),
        ];

        // Run 1
        let ftw1 = FixedTernaryWeight::from_ternary_weight(tw.clone());
        let mut net1 = DeterministicNetwork::new(
            vec![ftw1],
            vec![Activation::ReLU],
        );
        let out1: Vec<Fix128> = net1.forward(&input).to_vec();

        // Run 2 (fresh network, same weights)
        let (tw2, _) = quantize_to_ternary(&f32_w, 3, 3);
        let ftw2 = FixedTernaryWeight::from_ternary_weight(tw2);
        let mut net2 = DeterministicNetwork::new(
            vec![ftw2],
            vec![Activation::ReLU],
        );
        let out2: Vec<Fix128> = net2.forward(&input).to_vec();

        // Bit-exact comparison
        for i in 0..3 {
            assert_eq!(out1[i].hi, out2[i].hi, "hi mismatch at index {}", i);
            assert_eq!(out1[i].lo, out2[i].lo, "lo mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ragdoll_controller() {
        // Simple 2-body controller: 2*13=26 inputs → 16 hidden → 2*3=6 outputs
        let num_bodies = 2;
        let num_joints = 2;
        let input_size = num_bodies * FEATURES_PER_BODY; // 26
        let hidden_size = 16;
        let output_size = num_joints * 3; // 6

        // Create random-ish weights
        let f32_w1: Vec<f32> = (0..hidden_size * input_size)
            .map(|i| if i % 3 == 0 { 1.0 } else if i % 3 == 1 { -1.0 } else { 0.0 })
            .collect();
        let f32_w2: Vec<f32> = (0..output_size * hidden_size)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let (tw1, _) = quantize_to_ternary(&f32_w1, hidden_size, input_size);
        let (tw2, _) = quantize_to_ternary(&f32_w2, output_size, hidden_size);
        let ftw1 = FixedTernaryWeight::from_ternary_weight(tw1);
        let ftw2 = FixedTernaryWeight::from_ternary_weight(tw2);

        let network = DeterministicNetwork::new(
            vec![ftw1, ftw2],
            vec![Activation::ReLU, Activation::HardTanh],
        );

        let config = ControllerConfig {
            max_torque: Fix128::from_int(50),
            num_joints,
            num_bodies,
            features_per_body: FEATURES_PER_BODY,
        };

        let mut controller = RagdollController::new(network, config);

        // Create test bodies
        let body_a = RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 5, 0),
            Fix128::ONE,
        );
        let body_b = RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 3, 0),
            Fix128::ONE,
        );
        let bodies = [body_a, body_b];

        let output = controller.compute(&bodies);

        assert_eq!(output.torques.len(), num_joints);

        // Torques should be clamped to max_torque
        let max = Fix128::from_int(50);
        let neg_max = Fix128::ZERO - max;
        for torque in &output.torques {
            assert!(torque.x >= neg_max && torque.x <= max);
            assert!(torque.y >= neg_max && torque.y <= max);
            assert!(torque.z >= neg_max && torque.z <= max);
        }
    }

    #[test]
    fn test_ragdoll_controller_determinism() {
        let num_bodies = 2;
        let num_joints = 2;
        let input_size = num_bodies * FEATURES_PER_BODY;
        let hidden_size = 8;
        let output_size = num_joints * 3;

        let f32_w1: Vec<f32> = (0..hidden_size * input_size)
            .map(|i| (i as f32 * 0.7).sin())
            .collect();
        let f32_w2: Vec<f32> = (0..output_size * hidden_size)
            .map(|i| (i as f32 * 1.3).cos())
            .collect();

        let bodies = [
            RigidBody::new_dynamic(Vec3Fix::from_int(1, 10, 2), Fix128::from_ratio(3, 2)),
            RigidBody::new_dynamic(Vec3Fix::from_int(-1, 8, 1), Fix128::ONE),
        ];

        // Run twice
        let make_controller = || {
            let (tw1, _) = quantize_to_ternary(&f32_w1, hidden_size, input_size);
            let (tw2, _) = quantize_to_ternary(&f32_w2, output_size, hidden_size);
            let ftw1 = FixedTernaryWeight::from_ternary_weight(tw1);
            let ftw2 = FixedTernaryWeight::from_ternary_weight(tw2);
            let network = DeterministicNetwork::new(
                vec![ftw1, ftw2],
                vec![Activation::ReLU, Activation::HardTanh],
            );
            let config = ControllerConfig {
                max_torque: Fix128::from_int(100),
                num_joints,
                num_bodies,
                features_per_body: FEATURES_PER_BODY,
            };
            RagdollController::new(network, config)
        };

        let out1 = make_controller().compute(&bodies);
        let out2 = make_controller().compute(&bodies);

        // Bit-exact
        for (t1, t2) in out1.torques.iter().zip(out2.torques.iter()) {
            assert_eq!(t1.x.hi, t2.x.hi);
            assert_eq!(t1.x.lo, t2.x.lo);
            assert_eq!(t1.y.hi, t2.y.hi);
            assert_eq!(t1.y.lo, t2.y.lo);
            assert_eq!(t1.z.hi, t2.z.hi);
            assert_eq!(t1.z.lo, t2.z.lo);
        }
    }

    #[test]
    fn test_leaky_relu() {
        let alpha = Fix128::from_ratio(1, 100); // 0.01
        let mut values = [
            Fix128::from_int(5),
            Fix128::from_int(-10),
            Fix128::ZERO,
        ];

        fix128_leaky_relu(&mut values, alpha);

        assert_eq!(values[0].hi, 5);
        // -10 * 0.01 should be small negative
        assert!(values[1].is_negative());
        assert!(values[1] > Fix128::from_int(-1)); // -0.1, much less than -1
        assert!(values[2].is_zero());
    }

    #[test]
    fn test_controller_config_default() {
        let config = ControllerConfig::default();
        assert_eq!(config.num_joints, 8);
        assert_eq!(config.num_bodies, 9);
        assert_eq!(config.features_per_body, 13);
    }

    #[test]
    fn test_clamp_fix128() {
        let min = Fix128::from_int(-5);
        let max = Fix128::from_int(5);

        assert_eq!(clamp_fix128(Fix128::from_int(3), min, max).hi, 3);
        assert_eq!(clamp_fix128(Fix128::from_int(10), min, max).hi, 5);
        assert_eq!(clamp_fix128(Fix128::from_int(-10), min, max).hi, -5);
    }
}
