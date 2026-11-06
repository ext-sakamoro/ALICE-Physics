//! Oracle: a ray query finds a body whose collision box leaves the `Fix128` range.
//!
//! # What is measured
//!
//! `PhysicsWorld::raycast` and `PhysicsWorld::cast_ray` build a BVH of the bodies'
//! boxes for the query. A box built with `AABB::from_center_half` from a position
//! and a radius whose sum leaves the range wraps to the other side and comes out
//! inverted, so the BVH never offers that body as a candidate and the ray misses
//! it although the sphere's near surface is well inside the range. The broad-phase
//! of `step` builds its boxes with the saturating form since 001001f1; the query
//! paths are expected to see the same body.
//!
//! The closed form: a body at `x = 2^62 + 2^40` with radius `2^62` has its near
//! surface at `x = 2^40`; a ray from the origin along +x with reach `2^41` hits it
//! at `t = 2^40`. The mirrored body (`x = -(2^62 + 2^40)`, ray along -x) is the
//! same closed form with the box's minimum wrapping instead of its maximum. An
//! in-range twin (`x = 3`, radius `1`, hit at `t = 2`) keeps the same query honest.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape_raycast::RayFilter;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn pow2(n: u32) -> Fix128 {
    Fix128::from_raw(1i64 << n, 0)
}

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

fn world_with_sphere_at(center: Vec3Fix, radius: Fix128) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new(center, Fix128::ONE), radius);
    w
}

fn world_with_sphere(x: Fix128, radius: Fix128) -> PhysicsWorld {
    world_with_sphere_at(x_at(x), radius)
}

/// (body x, ray direction sign, expected t) for the two wrapping sides and the twin.
fn cases() -> Vec<(&'static str, Fix128, Fix128, Fix128, Fix128)> {
    let far = pow2(62) + pow2(40);
    vec![
        (
            "in range",
            Fix128::from_int(3),
            Fix128::ONE,
            Fix128::ONE,
            Fix128::from_int(2),
        ),
        ("box max wraps", far, pow2(62), Fix128::ONE, pow2(40)),
        ("box min wraps", -far, pow2(62), -Fix128::ONE, pow2(40)),
    ]
}

#[test]
fn raycast_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let reach = if want_t > pow2(41) {
            want_t * Fix128::from_int(2)
        } else {
            pow2(41)
        };
        let hit = w.raycast(Vec3Fix::ZERO, x_at(sign), reach);
        assert_eq!(hit, Some((0, want_t)), "{name}: raycast");
    }
}

#[test]
fn cast_ray_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let hit = w.cast_ray(Vec3Fix::ZERO, x_at(sign), pow2(41), &RayFilter::default());
        let hit = hit.unwrap_or_else(|| panic!("{name}: cast_ray missed"));
        assert_eq!((hit.body, hit.t), (Some(0), want_t), "{name}: cast_ray");
    }
}

/// `k` of the scaled path: the bits of the largest integer magnitude beyond 29
/// (a negative value by its one's complement, as the implementation counts it).
fn scale_bits_of(values: &[i64]) -> u32 {
    let top = values
        .iter()
        .map(|&v| 64 - (if v < 0 { !v } else { v }).leading_zeros())
        .max()
        .unwrap_or(0);
    top.saturating_sub(29)
}

/// FNV-1a over the raw bits of every distance, for the cross-platform check.
fn fold(digest: &mut u64, t: Fix128) {
    for b in t.hi.to_le_bytes().iter().chain(t.lo.to_le_bytes().iter()) {
        *digest ^= u64::from(*b);
        *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// Recorded on the first run (ubuntu / macOS / windows / wasm must agree).
const FAR_DIGEST: u64 = 0xe0016b5b9b5bff9d;

/// Exact closed forms beyond the direct limit: the centre is `L` along an axis and
/// `h·m` beside it, the radius `r·m` with `r² = h² + s²` (a Pythagorean triple),
/// so the ray from the origin along the axis enters the sphere at exactly
/// `t = L − s·m`. Every input is an integer, so the reference needs no rounding.
/// The scaled path must find every one of them within `2^(k−60)` and give the
/// same bits on every platform.
#[test]
fn raycast_beyond_the_direct_limit_is_within_the_stated_bound_of_exact_closed_forms() {
    let triples = [
        (3i64, 4i64, 5i64),
        (5, 12, 13),
        (8, 15, 17),
        (7, 24, 25),
        (20, 21, 29),
        (12, 35, 37),
    ];
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut checked = 0;
    let mut worst = Fix128::ZERO;
    for &(h, s, r) in &triples {
        for p in [40u32, 48, 52, 56, 57] {
            let m = 1i64 << p;
            for l in [1i64 << 61, 1i64 << 62, 3i64 << 60] {
                if s * m >= l {
                    // the origin would be inside the sphere (its far side beyond `l`)
                    continue;
                }
                for axis in 0..3 {
                    for sign in [1i64, -1] {
                        let mut c = [0i64; 3];
                        c[axis] = sign * l;
                        c[(axis + 1) % 3] = h * m;
                        let center = Vec3Fix::from_int(c[0], c[1], c[2]);
                        let mut d = [Fix128::ZERO; 3];
                        d[axis] = Fix128::from_int(sign);
                        let w = world_with_sphere_at(center, Fix128::from_int(r * m));
                        let want = Fix128::from_int(l - s * m);
                        let k = scale_bits_of(&[c[0], c[1], c[2], r * m, l]);
                        assert!(k > 0, "the case must be beyond the direct limit");
                        let bound = Fix128::from_raw(0, 1u64 << (k + 4)); // 2^(k−60)
                        let hit = w.raycast(
                            Vec3Fix::ZERO,
                            Vec3Fix::new(d[0], d[1], d[2]),
                            Fix128::from_int(l),
                        );
                        let (body, got) = hit.unwrap_or_else(|| panic!("silent None: triple {h},{s},{r} p={p} l={l} axis={axis} sign={sign}"));
                        assert_eq!(body, 0);
                        let err = (got - want).abs();
                        assert!(err <= bound, "triple {h},{s},{r} p={p} l={l} axis={axis} sign={sign}: |t − exact| = {err:?} > 2^({k}−60)");
                        if err > worst {
                            worst = err;
                        }
                        fold(&mut digest, got);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked >= 400, "compared {checked}");
    assert_eq!(
        digest, FAR_DIGEST,
        "far-query bits moved (checked {checked}, worst error {worst:?})"
    );
}
