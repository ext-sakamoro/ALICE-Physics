//! Audit S1-5 oracles for `alice_physics::contact_cache`.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::contact_cache::{BodyPairKey, ContactCache, ContactManifold};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}
fn contact(n: Vec3Fix, depth: f64) -> Contact {
    Contact {
        depth: f(depth),
        normal: n,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    }
}
fn manifold() -> ContactManifold {
    ContactManifold::new(BodyPairKey::new(0, 1), f(0.5), f(0.1))
}

/// Match radius: doc "within 1cm" = squared distance < 1e-4 (strict).
#[test]
fn matching_radius_is_one_centimetre_strict() {
    let c = contact(Vec3Fix::UNIT_Y, 0.1);
    let mut m = manifold();
    m.add_or_update(&c, v(0.0, 0.0, 0.0), Vec3Fix::ZERO);
    m.add_or_update(&c, v(0.0099, 0.0, 0.0), Vec3Fix::ZERO);
    assert_eq!(m.point_count(), 1, "0.99 cm merges");
    m.add_or_update(&c, v(0.0, 0.0101 + 0.0099, 0.0), Vec3Fix::ZERO); // 1.01 cm from the stored point at (0.0099,0,0)? far along y
    assert_eq!(m.point_count(), 2, "> 1 cm separates");
    // exactly 1 cm is NOT a match (strict <): stored (0.0099,0,0); probe at 0.0199
    let mut m2 = manifold();
    m2.add_or_update(&c, v(0.0, 0.0, 0.0), Vec3Fix::ZERO);
    m2.add_or_update(&c, v(0.0101, 0.0, 0.0), Vec3Fix::ZERO);
    assert_eq!(m2.point_count(), 2, "1.01 cm separates");
    // match is by squared distance in 3D
    let mut m3 = manifold();
    m3.add_or_update(&c, v(0.0, 0.0, 0.0), Vec3Fix::ZERO);
    m3.add_or_update(&c, v(0.0058, 0.0058, 0.0058), Vec3Fix::ZERO); // |d| = 0.01005
    assert_eq!(m3.point_count(), 2);
    m3.add_or_update(&c, v(0.0057, 0.0057, 0.0057), Vec3Fix::ZERO); // |d| = 0.00987 from origin
    assert_eq!(
        m3.point_count(),
        2,
        "0.987 cm merges into an existing point"
    );
}

#[test]
fn update_keeps_impulses_and_bumps_age_new_point_resets() {
    let mut m = manifold();
    let c = contact(Vec3Fix::UNIT_Y, 0.1);
    m.add_or_update(&c, Vec3Fix::ZERO, Vec3Fix::ZERO);
    m.store_impulses(0, f(5.0), f(6.0), f(7.0));
    for k in 1..=3u32 {
        m.add_or_update(&c, Vec3Fix::ZERO, Vec3Fix::ZERO);
        assert_eq!(m.points[0].age, k);
    }
    assert_eq!(m.warm_start_impulse(0), (f(5.0), f(6.0), f(7.0)));
    // a replaced (full-manifold) point starts from zero lambdas and age 0
    for i in 1..4 {
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, 0.2 + i as f64),
            v(i as f64, 0.0, 0.0),
            Vec3Fix::ZERO,
        );
    }
    assert_eq!(m.point_count(), 4);
    // shallowest is point 0 (depth 0.1); replace with deeper
    m.add_or_update(
        &contact(Vec3Fix::UNIT_Y, 9.0),
        v(50.0, 0.0, 0.0),
        Vec3Fix::ZERO,
    );
    assert_eq!(m.points[0].local_point_a, v(50.0, 0.0, 0.0));
    assert_eq!(
        m.warm_start_impulse(0),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
    assert_eq!(m.points[0].age, 0);
}

/// CachedContactPoint.normal doc: "Contact normal (world space, A→B)". The stored value is
/// `Contact::normal` (documented in collider.rs as pointing from B to A) and apply_warm_start
/// pushes body A ALONG +normal. With the A→B reading, a positive normal impulse would push A
/// toward B (interpenetration).
#[test]
#[ignore = "known defect: AUD-A-S1W5-014: CachedContactPoint::normal doc says A->B but the cached value is Contact.normal (B->A) and apply_warm_start moves A along +normal; with the documented A->B convention warm start pushes the bodies together"]
fn warm_start_normal_impulse_separates_with_documented_a_to_b_normal() {
    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::ONE;
    // A at origin, B above it: the A->B normal is +y
    let n_a_to_b = Vec3Fix::UNIT_Y;
    let m = cache.get_or_create(BodyPairKey::new(0, 1), f(0.5), f(0.0));
    m.add_or_update(&contact(n_a_to_b, 0.1), Vec3Fix::ZERO, Vec3Fix::ZERO);
    m.store_impulses(0, f(1.0), Fix128::ZERO, Fix128::ZERO);
    let mut bodies = [
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
    ];
    cache.apply_warm_start(&mut bodies);
    let rel = (bodies[1].velocity - bodies[0].velocity).dot(n_a_to_b);
    assert!(
        rel > Fix128::ZERO,
        "relative velocity along A->B = {}, bodies approach",
        rel.to_f64()
    );
}

/// Pin the actual convention (matches collider::Contact B->A and solver.rs:2810): A gets +n.
#[test]
fn warm_start_follows_contact_normal_b_to_a_convention_of_the_solver() {
    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::ONE;
    let n = Vec3Fix::UNIT_Y; // Contact.normal: B -> A (A above B)
    let m = cache.get_or_create(BodyPairKey::new(0, 1), f(0.5), f(0.0));
    m.add_or_update(&contact(n, 0.1), Vec3Fix::ZERO, Vec3Fix::ZERO);
    m.store_impulses(0, f(2.0), Fix128::ZERO, Fix128::ZERO);
    let mut bodies = [
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        RigidBody::new_dynamic(Vec3Fix::ZERO, f(4.0)),
    ];
    cache.apply_warm_start(&mut bodies);
    // v_A += n * lambda / m_A = (0,2,0); v_B -= n * lambda / m_B = (0,-0.5,0)
    assert_eq!(bodies[0].velocity, v(0.0, 2.0, 0.0));
    assert_eq!(bodies[1].velocity, v(0.0, -0.5, 0.0));
    // momentum balance: sum m v = 0
    let p = bodies[0].velocity * f(1.0) + bodies[1].velocity * f(4.0);
    assert_eq!(p, Vec3Fix::ZERO);
}

/// tangent frame (pub(crate), observed through apply_warm_start): for many unit normals the
/// three impulse directions n, t1, t2 are orthonormal and right-handed (t1 x t2 = n).
#[test]
fn tangent_frame_is_orthonormal_and_right_handed_for_many_normals() {
    let dirs: Vec<Vec3Fix> = {
        let mut d = vec![];
        for (x, y, z) in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 2.0, 0.5),
            (0.3, -0.2, 0.9),
            (2.0, 2.0, -1.0),
            (0.01, 1.0, 0.02),
            (5.0, 0.1, 0.1),
        ] {
            d.push(v(x, y, z).normalize());
        }
        d
    };
    for n in dirs {
        // probe the 3 lambda channels with unit factor on a unit-mass body vs a static one
        let probe = |ln: f64, l1: f64, l2: f64| -> Vec3Fix {
            let mut cache = ContactCache::new();
            cache.warm_start_factor = Fix128::ONE;
            let m = cache.get_or_create(BodyPairKey::new(0, 1), f(0.5), Fix128::ZERO);
            m.add_or_update(&contact(n, 0.1), Vec3Fix::ZERO, Vec3Fix::ZERO);
            m.store_impulses(0, f(ln), f(l1), f(l2));
            let mut bodies = [
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
                RigidBody::new_static(Vec3Fix::ZERO),
            ];
            cache.apply_warm_start(&mut bodies);
            bodies[0].velocity
        };
        let vn = probe(1.0, 0.0, 0.0);
        let t1 = probe(0.0, 1.0, 0.0);
        let t2 = probe(0.0, 0.0, 1.0);
        let e = 1e-9;
        let dot = |a: Vec3Fix, b: Vec3Fix| a.dot(b).to_f64();
        assert!(
            (dot(vn, vn) - 1.0).abs() < e
                && (dot(t1, t1) - 1.0).abs() < 1e-6
                && (dot(t2, t2) - 1.0).abs() < 1e-6,
            "unit lengths"
        );
        assert!(
            dot(vn, t1).abs() < 1e-6 && dot(vn, t2).abs() < 1e-6 && dot(t1, t2).abs() < 1e-6,
            "orthogonal for n = {:?}",
            n
        );
        let c = t1.cross(t2) - vn;
        assert!(c.length().to_f64() < 1e-6, "t1 x t2 = n (right-handed)");
    }
}

/// `ContactCache::manifolds` is a public Vec shadowed by a private pair index. Clearing the Vec
/// directly (it is `pub`) leaves the index pointing past the end and `find` panics.
#[test]
fn find_after_direct_manifolds_mutation_does_not_panic() {
    let mut cache = ContactCache::new();
    let key = BodyPairKey::new(0, 1);
    cache.get_or_create(key, f(0.5), f(0.1));
    cache.manifolds.clear();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.find(&key).is_none()));
    assert_eq!(
        r.ok(),
        Some(true),
        "find() panicked or returned a stale manifold"
    );
}

#[test]
fn frame_lifecycle_expiry_boundary_and_update_resets_staleness() {
    let mut cache = ContactCache::new();
    assert_eq!(cache.max_stale_frames, 3);
    let k = BodyPairKey::new(2, 7);
    cache.get_or_create(k, f(0.4), f(0.2)).add_or_update(
        &contact(Vec3Fix::UNIT_Y, 0.1),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    for frame in 1..=3 {
        cache.begin_frame();
        cache.end_frame();
        assert!(cache.find(&k).is_some(), "survives stale={frame} (<= max)");
    }
    // an update before end_frame resets staleness
    cache.begin_frame();
    cache.get_or_create(k, f(9.0), f(9.0)).add_or_update(
        &contact(Vec3Fix::UNIT_Y, 0.1),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    cache.end_frame();
    assert_eq!(cache.find(&k).unwrap().stale_frames, 0);
    for _ in 0..3 {
        cache.begin_frame();
        cache.end_frame();
    }
    assert!(cache.find(&k).is_some());
    cache.begin_frame();
    cache.end_frame();
    assert!(cache.find(&k).is_none(), "stale 4 > 3 removed");
    // friction/restitution are fixed at creation; later get_or_create does not overwrite
    let mut c2 = ContactCache::new();
    c2.get_or_create(k, f(0.4), f(0.2));
    let m = c2.get_or_create(k, f(9.0), f(9.0));
    assert_eq!(m.friction, f(0.4));
    assert_eq!(m.restitution, f(0.2));
    // default warm-start factor 0.8, custom max stale respected
    assert!((ContactCache::new().warm_start_factor.to_f64() - 0.8).abs() < 1e-15);
    let mut c3 = ContactCache::new();
    c3.max_stale_frames = 0;
    c3.get_or_create(k, f(0.4), f(0.2));
    c3.begin_frame();
    c3.end_frame();
    assert_eq!(c3.manifold_count(), 0);
}

#[test]
fn body_pair_key_canonical_and_equal_indices() {
    assert_eq!(BodyPairKey::new(9, 3), BodyPairKey::new(3, 9));
    let k = BodyPairKey::new(9, 3);
    assert_eq!((k.body_a, k.body_b), (3, 9));
    let s = BodyPairKey::new(4, 4);
    assert_eq!((s.body_a, s.body_b), (4, 4));
    // hash set semantics: both orders collapse
    let mut set = std::collections::HashSet::new();
    set.insert(BodyPairKey::new(1, 2));
    set.insert(BodyPairKey::new(2, 1));
    assert_eq!(set.len(), 1);
}

#[test]
fn warm_start_factor_zero_is_off_and_scales_linearly() {
    let mk = |factor: f64| {
        let mut cache = ContactCache::new();
        cache.warm_start_factor = f(factor);
        let m = cache.get_or_create(BodyPairKey::new(0, 1), f(0.5), Fix128::ZERO);
        m.add_or_update(&contact(Vec3Fix::UNIT_X, 0.1), Vec3Fix::ZERO, Vec3Fix::ZERO);
        m.store_impulses(0, f(8.0), Fix128::ZERO, Fix128::ZERO);
        let mut bodies = [
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        ];
        cache.apply_warm_start(&mut bodies);
        bodies[0].velocity.x.to_f64()
    };
    assert_eq!(mk(0.0), 0.0);
    assert_eq!(mk(0.5), 4.0);
    assert_eq!(mk(1.0), 8.0);
    assert_eq!(mk(0.25), 2.0);
}
