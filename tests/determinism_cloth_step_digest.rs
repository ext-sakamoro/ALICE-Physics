//! Bit-level pin of `Cloth::step` / `Cloth::step_with_sdf` without a rest tether.
//!
//! `EXPECTED_*` were recorded before the rest tether existed (main at `87b98703`). A cloth
//! that never enables the tether must keep producing the same states.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::Cloth;

fn mix(h: &mut u64, f: Fix128) {
    for word in [f.hi as u64, f.lo] {
        for byte in word.to_le_bytes() {
            *h ^= u64::from(byte);
            *h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn digest(c: &Cloth) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for v in c
        .positions
        .iter()
        .chain(&c.velocities)
        .chain(&c.prev_positions)
    {
        for f in [v.x, v.y, v.z] {
            mix(&mut h, f);
        }
    }
    h
}

fn curtain(self_collision: bool) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::from_int(0, 2, 0),
        Fix128::ONE,
        Fix128::ONE,
        6,
        5,
        Fix128::from_ratio(1, 10),
    );
    c.pin_top_row(6);
    c.wind = Vec3Fix::new(
        Fix128::from_ratio(1, 5),
        Fix128::ZERO,
        Fix128::from_ratio(1, 10),
    );
    c.config.stretch_compliance = Fix128::from_ratio(1, 10_000);
    c.config.self_collision = self_collision;
    // fold the bottom two rows back over the sheet, 1 cm below it, so the
    // self-contact passes have work to do
    for j in 3..5 {
        for i in 0..6 {
            let src = c.positions[(5 - j) * 6 + i];
            c.positions[j * 6 + i] = Vec3Fix::new(src.x, src.y - Fix128::from_ratio(1, 100), src.z);
            c.prev_positions[j * 6 + i] = c.positions[j * 6 + i];
        }
    }
    c
}

/// Recorded from `Cloth::step` before the rest tether was introduced.
const EXPECTED_PLAIN: u64 = 0x545b_6a9d_803d_31ef;
/// Same with self-collision enabled.
const EXPECTED_SELF: u64 = 0x524d_0515_e6e6_5540;

#[test]
fn cloth_step_digest_is_unchanged() {
    let mut digests = Vec::new();
    for (self_collision, expected) in [(false, EXPECTED_PLAIN), (true, EXPECTED_SELF)] {
        let mut c = curtain(self_collision);
        for _ in 0..60 {
            c.step(Fix128::from_ratio(1, 60));
        }
        let d = digest(&c);
        println!("self_collision {self_collision}: digest {d:#018x}");
        digests.push(d);
        assert_eq!(
            d, expected,
            "Cloth::step moved (self_collision {self_collision}): {d:#018x}"
        );
    }
}
