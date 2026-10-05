//! Frame lifecycle of the persistent contact cache
//!
//! Reaches `ContactCache::begin_frame`, `ContactCache::end_frame`,
//! `ContactCache::clear`, `ContactManifold::clear` and
//! `ContactManifold::is_empty`.
//!
//! Closed form for expiry, from the documented contract (`begin_frame` marks
//! every manifold one frame staler, a contact update resets it to 0,
//! `end_frame` keeps manifolds with `stale_frames <= max_stale_frames`):
//! a pair last touched in frame 0 has `stale_frames = k` at the end of frame
//! `k`, so with `max_stale_frames = M` it survives frames `1..=M` and is gone
//! at the end of frame `M + 1`. A pair touched every frame never expires.
//!
//! Run with: `cargo run --example contact_cache_lifecycle`

use alice_physics::collider::Contact;
use alice_physics::contact_cache::{BodyPairKey, ContactCache, ContactManifold};
use alice_physics::math::{Fix128, Vec3Fix};

fn touch(cache: &mut ContactCache, pair: BodyPairKey) {
    let contact = Contact {
        depth: Fix128::from_ratio(1, 100),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    };
    cache
        .get_or_create(pair, Fix128::from_ratio(1, 2), Fix128::ZERO)
        .add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);
}

fn main() {
    // ---- one manifold: empty, filled, cleared ---------------------------
    let pair = BodyPairKey::new(0, 1);
    let mut m = ContactManifold::new(pair, Fix128::from_ratio(1, 2), Fix128::ZERO);
    assert!(m.is_empty(), "a new manifold has no points");
    m.add_or_update(
        &Contact {
            depth: Fix128::from_ratio(1, 50),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        },
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    assert!(!m.is_empty(), "one point after add_or_update");
    assert_eq!(m.normal, Vec3Fix::UNIT_Y, "shared normal of one point");
    m.clear();
    assert!(m.is_empty(), "clear drops every point");
    assert_eq!(m.normal, Vec3Fix::ZERO, "clear resets the shared normal");

    // ---- cache: a kept pair and an abandoned pair -----------------------
    let mut cache = ContactCache::new();
    let max_stale = cache.max_stale_frames;
    let kept = BodyPairKey::new(0, 1);
    let dropped = BodyPairKey::new(2, 3);

    cache.begin_frame();
    touch(&mut cache, kept);
    touch(&mut cache, dropped);
    cache.end_frame();
    assert_eq!(cache.manifold_count(), 2);

    for k in 1..=max_stale + 2 {
        cache.begin_frame();
        touch(&mut cache, kept);
        cache.end_frame();

        let expect_dropped_alive = k <= max_stale;
        let dropped_alive = cache.find(&dropped).is_some();
        println!(
            "[contact_cache] end of frame {k}: abandoned pair alive = {dropped_alive} \
             (closed form k <= {max_stale}: {expect_dropped_alive})"
        );
        assert_eq!(dropped_alive, expect_dropped_alive, "frame {k}");
        if let Some(d) = cache.find(&dropped) {
            assert_eq!(d.stale_frames, k, "stale_frames counts untouched frames");
        }
        assert_eq!(
            cache.find(&kept).map(|m| m.stale_frames),
            Some(0),
            "touched pair is fresh every frame"
        );
        assert_eq!(
            cache.manifold_count(),
            1 + usize::from(expect_dropped_alive)
        );
    }

    // ---- clear empties the cache and its index --------------------------
    cache.clear();
    assert_eq!(cache.manifold_count(), 0);
    assert!(cache.find(&kept).is_none(), "the index is cleared too");
    touch(&mut cache, kept);
    assert_eq!(cache.manifold_count(), 1, "a cleared cache starts over");
    assert_eq!(cache.total_contact_points(), 1);

    println!(
        "[contact_cache] abandoned pair expired after {} untouched frames, as derived",
        max_stale + 1
    );
}
