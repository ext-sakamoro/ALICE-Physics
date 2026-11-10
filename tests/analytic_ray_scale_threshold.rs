//! Oracle: the scale of a far ray query is fixed, so its result bits are.
//!
//! # What is measured
//!
//! Beyond the direct limit (2³⁰) a ray query advances the ray to one unit before
//! the body's bounding sphere and solves there at the scale `2⁻ᵏ`, with `k` from
//! `body_scale_bits`: the bits of `2 · radius + 2` beyond 2²⁹. Every scaled operand
//! is floored to `2ᵏ⁻⁶⁴`, so a different `k` gives a result within the same stated
//! bound but with different low bits, and the bits are what every platform must
//! agree on and what a stored replay compares.
//!
//! The existing far-range oracles (tests/analytic_query_box_range.rs) use exact
//! integer closed forms, whose hits come out exact for any `k`, so they cannot see
//! a change of `k`. Here the entry depths are irrational: a sphere of radius
//! `r = 2ᵉ · (4 + q) / 4` sits `L = 2⁶¹` along +x and `b = 2ᵉ/4` or `2ᵉ/2` beside
//! the ray on y and z (`2b² < r²`, so the ray meets it), and the ray enters it at
//! `t = L − √(r² − 2b²)`. Radii run over each power of
//! two from 2²⁸ to 2⁴⁶ and just above it, where `k` steps.
//!
//! Each depth `L − t` is checked against an `f64` reference (relative `2⁻⁴⁰`, so
//! the right root and the right magnitude), and the bits of every `t` from
//! `raycast` and `cast_ray` are folded into a digest recorded on the first run.
//! A `k` taken over `radius + 2` instead of `2 · radius + 2`, or a scale target of
//! 2³¹ instead of 2²⁹, moves the digest.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape_raycast::RayFilter;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fold(digest: &mut u64, t: Fix128) {
    for b in t.hi.to_le_bytes().iter().chain(t.lo.to_le_bytes().iter()) {
        *digest ^= u64::from(*b);
        *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn to_f64(v: Fix128) -> f64 {
    v.hi as f64 + v.lo as f64 / 18_446_744_073_709_551_616.0
}

/// Recorded on the first run (every platform must agree).
const SCALE_THRESHOLD_DIGEST: u64 = 0xecf88dccea8f3f35;

#[test]
fn far_ray_queries_with_irrational_depths_keep_their_bits_across_each_scale_step() {
    let l = 1i64 << 61;
    let filter = RayFilter::default();
    let dir = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0;
    for e in 28u32..47 {
        for q in 0..4i64 {
            let r = (1i64 << e) / 4 * (4 + q);
            for beside in [1i64, 2] {
                let b = (1i64 << e) / 4 * beside;
                let mut w = PhysicsWorld::new(SolverConfig {
                    gravity: Vec3Fix::ZERO,
                    ..SolverConfig::default()
                });
                w.add_body_with_radius(
                    RigidBody::new(Vec3Fix::from_int(l, b, b), Fix128::ONE),
                    Fix128::from_int(r),
                );
                // the depth into the sphere where the ray enters it
                let want = ((r as f64) * (r as f64) - 2.0 * (b as f64) * (b as f64)).sqrt();
                let tag = format!("r = 2^{e}·{}/4, b = {beside}/4·2^{e}", 4 + q);
                let (body, t) = w
                    .raycast(Vec3Fix::ZERO, dir, Fix128::from_int(2 * l))
                    .unwrap_or_else(|| panic!("raycast missed: {tag}"));
                assert_eq!(body, 0, "{tag}");
                let hit = w
                    .cast_ray(Vec3Fix::ZERO, dir, Fix128::from_int(2 * l), &filter)
                    .unwrap_or_else(|| panic!("cast_ray missed: {tag}"));
                assert_eq!(hit.body, Some(0), "{tag}");
                for (what, t) in [("raycast", t), ("cast_ray", hit.t)] {
                    let depth = to_f64(Fix128::from_int(l) - t);
                    assert!(
                        ((depth - want) / want).abs() < 1.0 / (1u64 << 40) as f64,
                        "{what} {tag}: depth {depth} against {want}"
                    );
                    fold(&mut digest, t);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 19 * 4 * 2 * 2, "compared {checked}");
    assert_eq!(
        digest, SCALE_THRESHOLD_DIGEST,
        "far-query bits moved across the scale steps (checked {checked})"
    );
}
