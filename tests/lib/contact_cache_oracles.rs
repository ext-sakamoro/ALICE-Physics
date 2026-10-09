//! Lib tests of `src/contact_cache.rs` for the stale pair index: `manifolds`
//! is public, so a caller can remove a manifold directly and leave the index
//! pointing at another pair's slot.
//!
//! Included from `src/contact_cache.rs` as a `#[cfg(test)]` module, so they run
//! with `cargo test --lib`, the test set the mutation run uses.

use super::*;

/// After a manifold is removed from `manifolds` directly, the pair index of
/// a pair that is gone points at the slot another pair moved into. `find`
/// and `get_or_create` must not answer with that other pair's manifold.
#[test]
fn a_stale_index_never_answers_another_pair() {
    let (a, b) = (BodyPairKey::new(0, 1), BodyPairKey::new(2, 3));
    let mut cache = ContactCache::new();
    let _ = cache.get_or_create(a, Fix128::ONE, Fix128::ZERO);
    let _ = cache.get_or_create(b, Fix128::ONE, Fix128::ZERO);
    // `a` was at slot 0; removing it moves `b` there while the index still
    // maps `a` to 0
    cache.manifolds.remove(0);
    assert!(cache.find(&a).is_none(), "the removed pair is answered");
    assert_eq!(cache.find(&b).map(|m| m.pair), Some(b));
    let created = cache.get_or_create(a, Fix128::ONE, Fix128::ZERO);
    assert_eq!(created.pair, a, "a new manifold for the removed pair");
    assert_eq!(cache.manifolds.len(), 2);
}
