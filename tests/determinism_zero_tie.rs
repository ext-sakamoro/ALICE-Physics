//! Pins the sign of zero where two equal zeros meet a max or a min
//!
//! `f32::max` / `f32::min` leave that sign unspecified and the targets differ
//! (aarch64: max +0, min -0; x86_64: the argument), so before the crate's own
//! `det_math` max / min an equal pair of zeros could be stored with a different
//! bit on Linux x86 than on macOS ARM The expected values are the IEEE 754-2019
//! maximumNumber / minimumNumber results (max +0, min -0), written as bit
//! patterns, and every scenario is a public entry point whose output is kept
//! (a distance, a field reduction, the state bytes of a participant)
//!
//! CI runs this file on every test leg (x86_64 and aarch64), and
//! `examples/zero_tie_bits.rs` repeats the same checks in release

use alice_physics::erosion::{ErosionConfig, ErosionModifier};
use alice_physics::sdf_collider::{SdfField, SdfUnion};
use alice_physics::sim_field::ScalarField3D;
use alice_physics::world_participant::Participant;
use std::hint::black_box;

const POS: u32 = 0x0000_0000;
const NEG: u32 = 0x8000_0000;

/// A field whose distance is a fixed value (a signed zero here)
struct Constant(f32);

impl SdfField for Constant {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        black_box(self.0)
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// A surface at distance 0 everywhere with a fixed normal
struct Surface(f32, f32, f32);

impl SdfField for Surface {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        0.0
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (black_box(self.0), black_box(self.1), black_box(self.2))
    }
}

#[test]
fn a_union_of_two_zero_distances_is_negative_zero_in_both_orders() {
    // the union is the min of the two distances
    let pos_neg = SdfUnion::new(Constant(0.0), Constant(-0.0));
    let neg_pos = SdfUnion::new(Constant(-0.0), Constant(0.0));
    let pos_pos = SdfUnion::new(Constant(0.0), Constant(0.0));
    assert_eq!(pos_neg.distance(1.0, 2.0, 3.0).to_bits(), NEG);
    assert_eq!(neg_pos.distance(1.0, 2.0, 3.0).to_bits(), NEG);
    assert_eq!(pos_pos.distance(1.0, 2.0, 3.0).to_bits(), POS);
}

#[test]
fn the_maximum_of_a_field_of_signed_zeros_is_positive_zero() {
    let mut f = ScalarField3D::new(2, 2, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    for (i, v) in f.data.iter_mut().enumerate() {
        *v = if i % 2 == 0 { -0.0 } else { 0.0 };
    }
    assert_eq!(f.max_value().to_bits(), POS);
    f.data.fill(-0.0);
    assert_eq!(f.max_value().to_bits(), NEG);
    f.data.reverse();
    f.data[3] = 0.0;
    assert_eq!(f.max_value().to_bits(), POS);
}

fn exposure_state(normal: (f32, f32, f32)) -> (Vec<u32>, Vec<u8>) {
    let cfg = ErosionConfig {
        flow_direction: (1.0, 0.0, 0.0),
        ..ErosionConfig::default()
    };
    let mut er = ErosionModifier::new(cfg, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    er.compute_exposure_from_normals(&Surface(normal.0, normal.1, normal.2), 0.5);
    let bits = er.exposure.data.iter().map(|v| v.to_bits()).collect();
    let mut state = Vec::new();
    er.write_state(&mut state);
    (bits, state)
}

#[test]
fn erosion_exposure_on_flat_degenerate_and_sloped_normals_has_fixed_bits() {
    // flat and degenerate normals give an alignment of exactly -0 (a negated
    // fused multiply-add of zeros), clamped to +0; the sloped normal faces the
    // flow with alignment 0.6
    for (normal, want) in [
        ((0.0, 1.0, 0.0), POS),
        ((0.0, 0.0, 0.0), POS),
        ((-0.6, 0.8, 0.0), 0x3f19_999a),
    ] {
        let (bits, state) = exposure_state(normal);
        assert_eq!(bits.len(), 64);
        assert!(bits.iter().all(|&b| b == want), "{normal:?}: {bits:08x?}");
        // the exposure field is the last field written to the state bytes
        let tail: Vec<u32> = state[state.len() - 4 * 64..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            tail, bits,
            "{normal:?}: write_state does not end with the exposure field"
        );
    }
}
