//! Deterministic cube root, integer power and clamp:
//! `math_util::{cbrt_fix, pow_int, clamp_fix}`.
//!
//! Hand values: `cbrt(27) = 3`, `cbrt(10^9) = 1000` (the old fixed seed returned 29701.8),
//! `cbrt(2^62) = 2^(62/3) = 1664510.6`, `2^-3 = 0.125`, `(3/2)^4 = 81/16 = 5.0625`,
//! `clamp(7, -2, 5) = 5`. A bubble-radius sizing example: the radius of a sphere of volume
//! `V` is `cbrt(3V / (4 pi))`.
//!
//! ```bash
//! cargo run --release --example math_util_roots_powers --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::math_util::{cbrt_fix, clamp_fix, pow_int};

fn main() {
    for n in [27_i64, 1_000_000_000, 1 << 62] {
        println!("cbrt({n}) = {:.4}", cbrt_fix(Fix128::from_int(n)).to_f64());
    }
    println!(
        "pow_int(2, -3) = {}, pow_int(3/2, 4) = {}",
        pow_int(Fix128::from_int(2), -3).to_f64(),
        pow_int(Fix128::from_ratio(3, 2), 4).to_f64()
    );
    println!(
        "clamp(7, -2, 5) = {}",
        clamp_fix(
            Fix128::from_int(7),
            Fix128::from_int(-2),
            Fix128::from_int(5)
        )
        .to_f64()
    );
    // Equivalent sphere radius for 1 litre (0.001 m^3): r = cbrt(3 V / (4 pi)).
    let v = Fix128::from_ratio(1, 1000);
    let r = cbrt_fix(Fix128::from_int(3) * v / (Fix128::from_int(4) * Fix128::PI));
    println!("1 L sphere radius = {:.5} m", r.to_f64());
}
