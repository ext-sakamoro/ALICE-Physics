//! Independent oracle tests for `ContactCache`'s warm-start API
//! (`apply_warm_start`, `manifold_count`, `point_count`, `store_impulses`,
//! `tangent_frame`, `total_contact_points`, `warm_start_impulse` --
//! `scripts/wiring-baseline.txt` `unwired src/contact_cache.rs::*`, now
//! wired through `examples/contact_warm_start_cache.rs`, see that file's
//! module doc for why none of this is called from `PhysicsWorld::step()`).
//!
//! `tangent_frame` is `pub(crate)` and cannot be named from this file (a
//! separate crate) at all, let alone called to produce an expected value --
//! every expectation below that depends on it is built from `Vec3Fix::cross`
//! / `Vec3Fix::normalize` (independently-testable primitives) applied to the
//! reference axis selected by the rule documented on `tangent_frame`
//! (`src/contact_cache.rs`): pick `UNIT_X` if `|x| <= |y| && |x| <= |z|`,
//! else `UNIT_Y` if `|y| <= |z|`, else `UNIT_Z`.

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::contact_cache::{BodyPairKey, ContactCache};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use std::panic::catch_unwind;

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn contact(normal: Vec3Fix, depth: Fix128) -> Contact {
    Contact {
        depth,
        normal,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    }
}

// ----------------------------------------------------------------------
// manifold_count / total_contact_points: empty cache, and hand-counted
// after populating 3 pairs with 2 / 4 (capped, documented max) / 1 points.
// ----------------------------------------------------------------------

#[test]
fn manifold_count_and_total_contact_points_are_zero_on_an_empty_cache() {
    let cache = ContactCache::new();
    assert_eq!(cache.manifold_count(), 0);
    assert_eq!(cache.total_contact_points(), 0);
    assert!(
        cache.find(&BodyPairKey::new(0, 1)).is_none(),
        "miss on an empty cache"
    );
}

#[test]
fn manifold_count_and_total_contact_points_sum_across_pairs_with_the_documented_cap() {
    let mut cache = ContactCache::new();
    let pair_a = BodyPairKey::new(0, 1); // 2 distinct points
    let pair_b = BodyPairKey::new(2, 3); // 5 distinct points added, capped at the documented max of 4
    let pair_c = BodyPairKey::new(4, 5); // 1 point

    {
        let m = cache.get_or_create(pair_a, Fix128::ONE, Fix128::ZERO);
        for i in 0..2 {
            m.add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                v3i(i * 10, 0, 0),
                Vec3Fix::ZERO,
            );
        }
    }
    {
        let m = cache.get_or_create(pair_b, Fix128::ONE, Fix128::ZERO);
        for i in 0..5 {
            m.add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                v3i(i * 10, 0, 0),
                Vec3Fix::ZERO,
            );
        }
    }
    {
        let m = cache.get_or_create(pair_c, Fix128::ONE, Fix128::ZERO);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
    }

    assert_eq!(cache.manifold_count(), 3);
    assert_eq!(
        cache.find(&pair_a).map(|m| m.point_count()),
        Some(2),
        "cache hit: pair_a was populated"
    );
    assert_eq!(
        cache.find(&pair_b).map(|m| m.point_count()),
        Some(4),
        "5 distinct points capped at 4"
    );
    assert_eq!(cache.find(&pair_c).map(|m| m.point_count()), Some(1));
    assert!(
        cache.find(&BodyPairKey::new(9, 10)).is_none(),
        "cache miss: never populated"
    );
    // 2 + 4 (capped) + 1 = 7, not 2 + 5 + 1 = 8 -- total_contact_points must
    // read the post-cap state, not the number of add_or_update calls.
    assert_eq!(cache.total_contact_points(), 7);
}

// ----------------------------------------------------------------------
// warm_start_impulse / store_impulses: hit vs miss, zero impulse, and the
// documented `point_idx < len` boundary (`==len` is already out of range).
// ----------------------------------------------------------------------

#[test]
fn warm_start_impulse_is_the_zero_triple_on_a_fresh_point_and_on_any_miss() {
    let mut cache = ContactCache::new();
    let pair = BodyPairKey::new(0, 1);
    let m = cache.get_or_create(pair, Fix128::ONE, Fix128::ZERO);

    // Miss: manifold exists but has no points yet.
    assert_eq!(
        m.warm_start_impulse(0),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );

    m.add_or_update(
        &contact(Vec3Fix::UNIT_Y, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );

    // Hit, but a fresh point starts at zero (explicit zero-impulse case).
    assert_eq!(
        m.warm_start_impulse(0),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );

    // Miss: index exactly at len (1 point -> len==1, idx 1 is out of range).
    assert_eq!(
        m.warm_start_impulse(1),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
    // Miss: wildly out of range.
    assert_eq!(
        m.warm_start_impulse(usize::MAX),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
}

#[test]
fn store_impulses_round_trips_exactly_and_out_of_range_indices_are_a_silent_no_op() {
    let mut cache = ContactCache::new();
    let pair = BodyPairKey::new(0, 1);
    let m = cache.get_or_create(pair, Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_Y, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );

    // Negative/fractional values round-trip exactly (Fix128 is exact for
    // dyadic rationals, no float rounding to lose).
    m.store_impulses(
        0,
        Fix128::from_ratio(-7, 2),
        fi(0),
        Fix128::from_ratio(11, 4),
    );
    assert_eq!(
        m.warm_start_impulse(0),
        (
            Fix128::from_ratio(-7, 2),
            Fix128::ZERO,
            Fix128::from_ratio(11, 4)
        )
    );

    // Out-of-range store (idx == len) must not grow the manifold or disturb
    // the in-range point.
    m.store_impulses(1, fi(999), fi(999), fi(999));
    assert_eq!(
        m.point_count(),
        1,
        "store on idx==len must not append a point"
    );
    assert_eq!(
        m.warm_start_impulse(0),
        (
            Fix128::from_ratio(-7, 2),
            Fix128::ZERO,
            Fix128::from_ratio(11, 4)
        ),
        "out-of-range store must not corrupt index 0"
    );

    // Out-of-range store on an empty manifold (idx == len == 0) must also
    // be a no-op, not a panic.
    let empty_pair = BodyPairKey::new(8, 9);
    let e = cache.get_or_create(empty_pair, Fix128::ONE, Fix128::ZERO);
    e.store_impulses(0, fi(1), fi(1), fi(1));
    assert_eq!(e.point_count(), 0);
}

// ----------------------------------------------------------------------
// apply_warm_start: tangent_frame reached transitively. Every `t1`/`t2`
// below is built from `Vec3Fix::cross` + `Vec3Fix::normalize` applied to
// the reference axis picked by `tangent_frame`'s documented rule -- never
// by calling `tangent_frame` (impossible from here anyway: `pub(crate)`).
// ----------------------------------------------------------------------

#[test]
fn apply_warm_start_normal_y_uses_reference_x_hand_derived() {
    // normal = Y: |x|=0 <= |y|=1 true, |x|=0 <= |z|=0 true -> reference = X.
    let normal = Vec3Fix::UNIT_Y;
    let reference = Vec3Fix::UNIT_X;
    let t1 = normal.cross(reference).normalize(); // Y × X = -Z, length 1
    let t2 = normal.cross(t1); // Y × -Z = -X
    assert_eq!(t1, -Vec3Fix::UNIT_Z);
    assert_eq!(t2, -Vec3Fix::UNIT_X);

    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::ONE; // factor 1.0: impulse == stored lambda exactly
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(&contact(normal, fi(1)), Vec3Fix::ZERO, Vec3Fix::ZERO);
    m.store_impulses(0, fi(3), fi(5), fi(9)); // lambda_n, lambda_t1, lambda_t2

    let expected_impulse = normal * fi(3) + t1 * fi(5) + t2 * fi(9);
    // = (0,3,0) + (0,0,-5) + (-9,0,0) = (-9, 3, -5)
    assert_eq!(expected_impulse, v3i(-9, 3, -5));

    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), // inv_mass 1
        RigidBody::new_static(v3i(1, 0, 0)),                // inv_mass 0
    ];
    cache.apply_warm_start(&mut bodies);
    assert_eq!(bodies[0].velocity, expected_impulse);
    assert_eq!(
        bodies[1].velocity,
        Vec3Fix::ZERO,
        "inv_mass==0 body must stay exactly zero"
    );
}

#[test]
fn apply_warm_start_normal_z_uses_reference_x_hand_derived_and_scales_both_bodies() {
    // normal = Z: |x|=0 <= |y|=0 true, |x|=0 <= |z|=1 true -> reference = X.
    let normal = Vec3Fix::UNIT_Z;
    let reference = Vec3Fix::UNIT_X;
    let t1 = normal.cross(reference).normalize(); // Z × X = Y
    let t2 = normal.cross(t1); // Z × Y = -X
    assert_eq!(t1, Vec3Fix::UNIT_Y);
    assert_eq!(t2, -Vec3Fix::UNIT_X);

    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::from_ratio(1, 4);
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(&contact(normal, fi(1)), Vec3Fix::ZERO, Vec3Fix::ZERO);
    m.store_impulses(0, fi(8), fi(12), fi(4)); // *0.25 -> (2, 3, 1)

    let impulse = normal * fi(2) + t1 * fi(3) + t2 * fi(1);
    // = (0,0,2) + (0,3,0) + (-1,0,0) = (-1, 3, 2)
    assert_eq!(impulse, v3i(-1, 3, 2));

    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_dynamic(v3i(1, 0, 0), Fix128::ONE),
    ];
    bodies[0].inv_mass = fi(2);
    bodies[1].inv_mass = fi(5);
    cache.apply_warm_start(&mut bodies);
    // body A: += impulse * inv_mass(2); body B: -= impulse * inv_mass(5)
    assert_eq!(bodies[0].velocity, impulse * fi(2));
    assert_eq!(bodies[1].velocity, -(impulse * fi(5)));
}

#[test]
fn apply_warm_start_tie_normal_matches_cross_product_reference_x_rule() {
    // normal = (1,1,1): |x|=1<=|y|=1 true, |x|=1<=|z|=1 true -> reference = X
    // (the `<=` means an exact 3-way tie resolves to X, documented and
    // already exercised against `tangent_frame` directly by the crate's own
    // internal unit test `tangent_frame_picks_least_parallel_axis_and_is_orthonormal`;
    // reproduced here independently through `apply_warm_start`, which is
    // the point of this file.)
    //
    // `ContactManifold::add_or_update` -> `update_normal` stores
    // `sum_of_point_normals / length` as the manifold's *shared* normal,
    // not the raw per-point `Contact::normal` passed in -- for a single
    // point that is exactly `raw.normalize()` (same formula as
    // `Vec3Fix::normalize`). `apply_warm_start` reads that shared,
    // normalized field for both the `impulse_n` term and the
    // `tangent_frame` input, so the oracle must use the normalized vector
    // too, not the raw (1,1,1) (whose cross-product *direction* happens to
    // match after normalizing `t1`/`t2`, since cross scales linearly with a
    // positive scalar, but the `impulse_n` term does not: `(1,1,1) * k` vs
    // `(1,1,1)/sqrt(3) * k` differ by the missing `1/sqrt(3)`).
    let raw_normal = v3i(1, 1, 1);
    let normal = raw_normal.normalize();
    let reference = Vec3Fix::UNIT_X;
    let t1 = normal.cross(reference).normalize();
    let t2 = normal.cross(t1);

    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::ONE;
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(&contact(raw_normal, fi(1)), Vec3Fix::ZERO, Vec3Fix::ZERO);
    assert_eq!(
        m.normal, normal,
        "manifold's shared normal must be the normalized contact normal"
    );
    m.store_impulses(0, fi(2), fi(3), fi(-1));

    let expected = normal * fi(2) + t1 * fi(3) + t2 * fi(-1);
    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_static(v3i(1, 0, 0)),
    ];
    cache.apply_warm_start(&mut bodies);
    assert_eq!(bodies[0].velocity, expected);
}

// ----------------------------------------------------------------------
// Degenerate inputs.
// ----------------------------------------------------------------------

#[test]
fn apply_warm_start_on_an_empty_cache_leaves_bodies_untouched() {
    let cache = ContactCache::new();
    let mut bodies = vec![RigidBody::new_dynamic(v3i(1, 2, 3), Fix128::ONE)];
    bodies[0].velocity = v3i(4, 5, 6);
    cache.apply_warm_start(&mut bodies);
    assert_eq!(bodies[0].velocity, v3i(4, 5, 6));
}

#[test]
fn apply_warm_start_skips_manifolds_whose_body_index_is_out_of_range() {
    let mut cache = ContactCache::new();
    let m = cache.get_or_create(BodyPairKey::new(0, 5), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_X, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    m.store_impulses(0, fi(100), fi(100), fi(100));

    // Only 2 bodies (indices 0,1) exist; the manifold references body 5.
    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_dynamic(v3i(1, 0, 0), Fix128::ONE),
    ];
    cache.apply_warm_start(&mut bodies);
    assert_eq!(
        bodies[0].velocity,
        Vec3Fix::ZERO,
        "out-of-range manifold must not be applied at all"
    );
    assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
}

#[test]
fn apply_warm_start_zero_normal_degenerate_contact_produces_exactly_zero_impulse_regardless_of_lambda_magnitude(
) {
    // A `Contact` with `normal == ZERO` (caller-provided degenerate input,
    // e.g. a buggy upstream narrow-phase): `ContactManifold::update_normal`'s
    // `if !len.is_zero()` guard means the manifold's shared normal is never
    // set away from its initial `Vec3Fix::ZERO` for this point. tangent_frame's
    // reference-axis rule for ZERO is well-defined (|0|<=|0| on both arms ->
    // X), but `ZERO.cross(X) == ZERO` and `ZERO.normalize() == ZERO`
    // (`Vec3Fix::normalize`'s documented zero-length behaviour), so both
    // `t1` and `t2` are ZERO too -> every term of the impulse is `ZERO * k
    // == ZERO`, independent of how large `lambda_n`/`lambda_t1`/`lambda_t2`
    // are. Uses extreme-magnitude lambdas (near Fix128's raw range) to also
    // cover the "extreme Fix128 magnitude must not panic" requirement: the
    // zero multiplier means this can never overflow regardless of the
    // wrapping semantics documented on `Fix128`'s `Mul`.
    let normal = Vec3Fix::ZERO;
    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);

    let mut cache = ContactCache::new();
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(&contact(normal, fi(1)), Vec3Fix::ZERO, Vec3Fix::ZERO);
    assert_eq!(
        m.normal,
        Vec3Fix::ZERO,
        "zero-normal contact must leave the manifold's shared normal at ZERO"
    );
    m.store_impulses(0, extreme, extreme, extreme);

    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_static(v3i(1, 0, 0)),
    ];
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.apply_warm_start(&mut bodies);
    }));
    assert!(
        result.is_ok(),
        "apply_warm_start must not panic on extreme-magnitude lambdas"
    );
    assert_eq!(
        bodies[0].velocity,
        Vec3Fix::ZERO,
        "zero normal -> zero impulse exactly, regardless of lambda magnitude"
    );
    assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
}

#[test]
fn apply_warm_start_zero_impulse_on_a_nonzero_normal_leaves_velocity_unchanged() {
    let mut cache = ContactCache::new();
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_X, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    m.store_impulses(0, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO); // explicit zero impulse

    let mut bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    cache.apply_warm_start(&mut bodies);
    assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);
}

#[test]
fn apply_warm_start_extreme_lambda_with_nonzero_normal_does_not_panic() {
    // Same wrapping-arithmetic contract as the zero-normal case above, but
    // here the normal is nonzero so the multiplier is not trivially zero --
    // this exercises `Fix128::Mul`'s documented wrapping (mod 2^128) path
    // for real. Only the "does not panic" contract is asserted (the exact
    // wrapped numeric value is `math.rs`'s oracle surface, not
    // `contact_cache`'s).
    let extreme = Fix128::from_raw(i64::MAX, u64::MAX);
    let mut cache = ContactCache::new();
    let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_X, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    m.store_impulses(0, extreme, extreme, extreme);

    let mut bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let result = catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.apply_warm_start(&mut bodies);
    }));
    assert!(
        result.is_ok(),
        "apply_warm_start must not panic on extreme-magnitude lambdas with a nonzero normal"
    );
}
