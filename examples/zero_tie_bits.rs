//! The sign of zero where two equal zeros meet a max or a min
//!
//! Prints and checks, in the profile the example is built with (CI runs every
//! example in release on Linux x86_64), the same values that
//! `tests/determinism_zero_tie.rs` pins in the test profile on every leg: the
//! union of two zero distances is -0 in both orders, the maximum of a field of
//! mixed signed zeros is +0
//!
//! Run: `cargo run --release --example zero_tie_bits`

use alice_physics_core::sdf_collider::{SdfField, SdfUnion};
use alice_physics_core::sim_field::ScalarField3D;
use std::hint::black_box;

struct Constant(f32);

impl SdfField for Constant {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        black_box(self.0)
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

fn main() {
    let a = SdfUnion::new(Constant(0.0), Constant(-0.0)).distance(0.0, 0.0, 0.0);
    let b = SdfUnion::new(Constant(-0.0), Constant(0.0)).distance(0.0, 0.0, 0.0);
    println!(
        "union(+0, -0) = {:#010x}, union(-0, +0) = {:#010x}",
        a.to_bits(),
        b.to_bits()
    );
    assert_eq!((a.to_bits(), b.to_bits()), (0x8000_0000, 0x8000_0000));

    let mut f = ScalarField3D::new(2, 2, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    for (i, v) in f.data.iter_mut().enumerate() {
        *v = if i % 2 == 0 { -0.0 } else { 0.0 };
    }
    let m = f.max_value();
    println!(
        "max of a field of mixed signed zeros = {:#010x}",
        m.to_bits()
    );
    assert_eq!(m.to_bits(), 0x0000_0000);
}
