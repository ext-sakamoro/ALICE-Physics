//! Contact Manifold Cache with Warm Starting
//!
//! Persistent contact manifolds that survive across frames for stable stacking
//! and improved solver convergence. Implements warm starting by carrying over
//! accumulated impulses (lambdas) from the previous frame.
//!
//! # AAA Features
//!
//! - **4-point manifold**: Up to 4 contact points per body pair
//! - **Persistent contacts**: Matching via feature IDs across frames
//! - **Warm starting**: Pre-apply previous frame's impulses for faster convergence
//! - **Contact aging**: Auto-remove stale contacts

use crate::collider::Contact;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::HashMap;

/// Maximum contact points per manifold (internal cap, currently 4).
pub(crate) const MAX_MANIFOLD_POINTS: usize = 4;

/// A single cached contact point within a manifold.
///
/// Every field is expressed with the manifold's `pair.body_a` as body A and
/// `pair.body_b` as body B (the sorted pair), whichever order the contact
/// was reported in: [`crate::solver::PhysicsWorld::add_contact`] turns a
/// contact given as `(body_a > body_b)` around before storing it. The
/// tangent impulses are coordinates in the frame
/// [`ContactCache::apply_warm_start`] builds from the manifold normal.
///
/// Marked `#[non_exhaustive]` so future warm-start / analytics fields
/// can be added without a breaking API change; construct via the internal
/// `CachedContactPoint::new` and inspect via public fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CachedContactPoint {
    /// Contact point on body A (`pair.body_a`, local space)
    pub local_point_a: Vec3Fix,
    /// Contact point on body B (`pair.body_b`, local space)
    pub local_point_b: Vec3Fix,
    /// Contact normal (world space, pointing from B to A, i.e. from
    /// `pair.body_b` to `pair.body_a` — the
    /// [`crate::collider::Contact::normal`] convention)
    pub normal: Vec3Fix,
    /// Penetration depth
    pub depth: Fix128,
    /// Accumulated normal impulse (for warm starting)
    pub lambda_n: Fix128,
    /// Accumulated tangent impulse X (for warm starting)
    pub lambda_t1: Fix128,
    /// Accumulated tangent impulse Y (for warm starting)
    pub lambda_t2: Fix128,
    /// Number of frames this contact has persisted
    pub age: u32,
}

impl CachedContactPoint {
    /// Create a new cached contact point
    #[must_use]
    pub const fn new(local_a: Vec3Fix, local_b: Vec3Fix, normal: Vec3Fix, depth: Fix128) -> Self {
        Self {
            local_point_a: local_a,
            local_point_b: local_b,
            normal,
            depth,
            lambda_n: Fix128::ZERO,
            lambda_t1: Fix128::ZERO,
            lambda_t2: Fix128::ZERO,
            age: 0,
        }
    }
}

impl CachedContactPoint {
    /// The same point seen with A and B exchanged: the points swap, the
    /// normal turns around, the normal impulse keeps its magnitude and the
    /// tangent impulses are re-expressed for the frame of the turned normal.
    ///
    /// For the frame `(t1, t2)` of a normal `n` (see [`tangent_frame`]) the
    /// frame of `-n` is exactly `(-t1, t2)`: the reference axis depends only
    /// on `|n|`, `(-n) × e = -(n × e)` for a unit axis `e` (the products are
    /// by 0 and ±1), normalising is odd, and `(-n) × (-t1) = n × t1`. The
    /// impulse on the new A must be minus the impulse on the old A, so
    /// `(λn, λt1, λt2)` becomes `(λn, λt1, -λt2)`. Every step is a negation
    /// or a swap, so turning a point twice gives it back bit for bit.
    #[must_use]
    pub(crate) fn turned(self) -> Self {
        Self {
            local_point_a: self.local_point_b,
            local_point_b: self.local_point_a,
            normal: -self.normal,
            lambda_t2: -self.lambda_t2,
            ..self
        }
    }
}

/// `contact` as reported for `(body_a, body_b)`, expressed with the smaller
/// index as A (the order of [`BodyPairKey::new`]): unchanged when
/// `body_a <= body_b`, otherwise with the points swapped and the normal
/// turned around.
#[must_use]
pub(crate) fn oriented_to_key(body_a: usize, body_b: usize, contact: &Contact) -> Contact {
    if body_a > body_b {
        Contact {
            depth: contact.depth,
            normal: -contact.normal,
            point_a: contact.point_b,
            point_b: contact.point_a,
        }
    } else {
        *contact
    }
}

/// Body pair key for manifold lookup
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyPairKey {
    /// Index of body A (always the smaller index)
    pub body_a: u32,
    /// Index of body B (always the larger index)
    pub body_b: u32,
}

impl BodyPairKey {
    /// Create a canonical body pair key (ensures a < b)
    #[inline]
    #[must_use]
    pub const fn new(a: usize, b: usize) -> Self {
        if a < b {
            Self {
                body_a: a as u32,
                body_b: b as u32,
            }
        } else {
            Self {
                body_a: b as u32,
                body_b: a as u32,
            }
        }
    }
}

/// Contact manifold: up to 4 persistent contact points between two bodies.
///
/// Marked `#[non_exhaustive]` so additional accumulator / diagnostic
/// fields can be added post-v1.0 without a breaking API change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ContactManifold {
    /// Body pair this manifold belongs to
    pub pair: BodyPairKey,
    /// Active contact points (up to 4 per manifold).
    pub points: Vec<CachedContactPoint>,
    /// Shared normal direction (average of point normals, pointing from
    /// `pair.body_b` to `pair.body_a`)
    pub normal: Vec3Fix,
    /// Friction coefficient for this pair
    pub friction: Fix128,
    /// Restitution coefficient for this pair
    pub restitution: Fix128,
    /// Number of frames since last update (for expiry)
    pub stale_frames: u32,
}

impl ContactManifold {
    /// Create a new empty manifold
    #[must_use]
    pub fn new(pair: BodyPairKey, friction: Fix128, restitution: Fix128) -> Self {
        Self {
            pair,
            points: Vec::with_capacity(MAX_MANIFOLD_POINTS),
            normal: Vec3Fix::ZERO,
            friction,
            restitution,
            stale_frames: 0,
        }
    }

    /// Add or update a contact point in the manifold
    ///
    /// If a matching point exists (within threshold), update it and preserve lambdas.
    /// Otherwise, add as new. If full (4 points), replace the shallowest.
    ///
    /// `contact`, `local_a` and `local_b` must have `pair.body_a` as A (see
    /// [`CachedContactPoint`]); [`crate::solver::PhysicsWorld::add_contact`]
    /// orients a contact reported the other way round before calling this.
    pub fn add_or_update(&mut self, contact: &Contact, local_a: Vec3Fix, local_b: Vec3Fix) {
        // Squared distance threshold for contact matching.
        // 0.0001 = (0.01m)^2, matches contacts within 1cm.
        let threshold_sq = Fix128::from_ratio(1, 10000);

        // Try to find matching existing point
        let mut best_match: Option<usize> = None;
        let mut best_dist_sq = threshold_sq;

        for (i, existing) in self.points.iter().enumerate() {
            let dist_sq = (existing.local_point_a - local_a).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_match = Some(i);
            }
        }

        if let Some(idx) = best_match {
            // Update existing point — preserve accumulated impulses (warm starting)
            let lambda_n = self.points[idx].lambda_n;
            let lambda_t1 = self.points[idx].lambda_t1;
            let lambda_t2 = self.points[idx].lambda_t2;
            let age = self.points[idx].age;

            self.points[idx] =
                CachedContactPoint::new(local_a, local_b, contact.normal, contact.depth);
            self.points[idx].lambda_n = lambda_n;
            self.points[idx].lambda_t1 = lambda_t1;
            self.points[idx].lambda_t2 = lambda_t2;
            self.points[idx].age = age + 1;
        } else if self.points.len() < MAX_MANIFOLD_POINTS {
            // Add new point
            self.points.push(CachedContactPoint::new(
                local_a,
                local_b,
                contact.normal,
                contact.depth,
            ));
        } else {
            // Replace shallowest point
            let mut shallowest_idx = 0;
            let mut shallowest_depth = self.points[0].depth;
            for (i, p) in self.points.iter().enumerate().skip(1) {
                if p.depth < shallowest_depth {
                    shallowest_depth = p.depth;
                    shallowest_idx = i;
                }
            }
            if contact.depth > shallowest_depth {
                self.points[shallowest_idx] =
                    CachedContactPoint::new(local_a, local_b, contact.normal, contact.depth);
            }
        }

        // Update shared normal
        self.update_normal();
        self.stale_frames = 0;
    }

    /// Exchange A and B: turn every point ([`CachedContactPoint::turned`])
    /// and negate the shared normal (the average of the turned point normals
    /// is exactly the negated average). The pair key is not touched.
    pub(crate) fn turn(&mut self) {
        for p in &mut self.points {
            *p = p.turned();
        }
        self.normal = -self.normal;
    }

    /// Update the shared normal (average of point normals)
    fn update_normal(&mut self) {
        if self.points.is_empty() {
            self.normal = Vec3Fix::ZERO;
            return;
        }

        let mut sum = Vec3Fix::ZERO;
        for p in &self.points {
            sum = sum + p.normal;
        }
        let len = sum.length();
        if !len.is_zero() {
            self.normal = sum / len;
        }
    }

    /// Number of active contact points
    #[inline]
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Check if this manifold is empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear all points (reset manifold)
    pub fn clear(&mut self) {
        self.points.clear();
        self.normal = Vec3Fix::ZERO;
    }

    /// Get warm-start impulse for a contact point
    #[must_use]
    pub fn warm_start_impulse(&self, point_idx: usize) -> (Fix128, Fix128, Fix128) {
        if point_idx < self.points.len() {
            let p = &self.points[point_idx];
            (p.lambda_n, p.lambda_t1, p.lambda_t2)
        } else {
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
        }
    }

    /// Store solver impulses back into the cache
    pub fn store_impulses(
        &mut self,
        point_idx: usize,
        lambda_n: Fix128,
        lambda_t1: Fix128,
        lambda_t2: Fix128,
    ) {
        if point_idx < self.points.len() {
            self.points[point_idx].lambda_n = lambda_n;
            self.points[point_idx].lambda_t1 = lambda_t1;
            self.points[point_idx].lambda_t2 = lambda_t2;
        }
    }
}

/// Contact cache: stores all active manifolds across frames
pub struct ContactCache {
    /// All active manifolds
    pub manifolds: Vec<ContactManifold>,
    /// `HashMap` index for O(1) manifold lookup (std feature only)
    #[cfg(feature = "std")]
    pub(crate) pair_index: HashMap<BodyPairKey, usize>,
    /// Maximum stale frames before manifold is removed
    pub max_stale_frames: u32,
    /// Warm starting factor (0.0 = off, 1.0 = full warm start)
    pub warm_start_factor: Fix128,
}

impl ContactCache {
    /// Create a new contact cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            manifolds: Vec::new(),
            #[cfg(feature = "std")]
            pair_index: HashMap::new(),
            max_stale_frames: 3,
            warm_start_factor: Fix128::from_ratio(8, 10), // 0.8 default
        }
    }

    /// Find or create manifold for a body pair
    ///
    /// # Panics
    ///
    /// Panics if the internal manifold `Vec` is empty after a push (should never happen).
    pub fn get_or_create(
        &mut self,
        pair: BodyPairKey,
        friction: Fix128,
        restitution: Fix128,
    ) -> &mut ContactManifold {
        // Find existing using HashMap (O(1)) with std feature, or linear scan (O(n)) without
        //
        // `manifolds` is `pub`, so an external `clear()`/`retain()`/`remove()` on it
        // can leave `pair_index` pointing past the end or at another pair's slot;
        // `locate` checks the slot and falls back to a scan, and a hit found by
        // the scan is re-indexed here.
        let pos = self.locate(&pair);
        #[cfg(feature = "std")]
        if let Some(idx) = pos {
            self.pair_index.insert(pair, idx);
        }

        if let Some(idx) = pos {
            &mut self.manifolds[idx]
        } else {
            #[cfg(feature = "std")]
            {
                let idx = self.manifolds.len();
                self.manifolds
                    .push(ContactManifold::new(pair, friction, restitution));
                self.pair_index.insert(pair, idx);
                self.manifolds
                    .last_mut()
                    .expect("just pushed, cannot be empty")
            }
            #[cfg(not(feature = "std"))]
            {
                self.manifolds
                    .push(ContactManifold::new(pair, friction, restitution));
                self.manifolds
                    .last_mut()
                    .expect("just pushed, cannot be empty")
            }
        }
    }

    /// Find manifold for a body pair (read-only)
    ///
    /// `manifolds` is `pub`, so an external `clear()`/`retain()`/`remove()`
    /// on it can leave the private `pair_index` pointing past the end or at
    /// another pair's slot; the slot is checked, so a stale index neither
    /// panics nor answers another pair.
    #[must_use]
    pub fn find(&self, pair: &BodyPairKey) -> Option<&ContactManifold> {
        self.locate(pair).map(|idx| &self.manifolds[idx])
    }

    /// Slot of the manifold for `pair`. With `std` the pair index answers in
    /// O(1): a pair it does not hold has no manifold (the miss stays O(1), so
    /// creating new pairs stays linear per frame), and an entry left stale by a
    /// direct edit of `manifolds` (past the end, or shifted onto another
    /// pair's slot) falls back to the O(n) scan, so another pair's manifold is
    /// never answered. Without `std` it is the scan.
    fn locate(&self, pair: &BodyPairKey) -> Option<usize> {
        #[cfg(feature = "std")]
        {
            let idx = *self.pair_index.get(pair)?;
            if self.manifolds.get(idx).is_some_and(|m| m.pair == *pair) {
                return Some(idx);
            }
        }
        self.manifolds.iter().position(|m| m.pair == *pair)
    }

    /// Mark all manifolds as potentially stale (call at start of frame)
    pub fn begin_frame(&mut self) {
        for manifold in &mut self.manifolds {
            manifold.stale_frames += 1;
        }
    }

    /// Remove expired manifolds (call at end of frame)
    pub fn end_frame(&mut self) {
        let max_stale = self.max_stale_frames;
        self.manifolds.retain(|m| m.stale_frames <= max_stale);
        self.rebuild_pair_index();
    }

    /// Follow `PhysicsWorld::remove_body`'s swap-remove of body `idx`, after
    /// which the body that was at `last` sits at `idx`.
    ///
    /// - A manifold whose pair involves `idx` belongs to the removed body: no
    ///   surviving body has those contacts, so it is dropped (keeping it
    ///   would hand its impulses to the moved body's next contact).
    /// - A manifold whose pair involves `last` belongs to the moved body: it
    ///   is kept and re-keyed with `last` replaced by `idx`, re-sorted through
    ///   [`BodyPairKey::new`]. When the re-sort exchanges A and B (the other
    ///   body's index lies between `idx` and `last`), every point is turned
    ///   ([`CachedContactPoint::turned`]) and the normal negated, so the data
    ///   keeps describing `pair.body_a` as A: a world that had the moved body
    ///   at `idx` from the start caches the same data under the same key. The
    ///   new key involves `idx`, which no kept manifold does, so it cannot
    ///   collide with another pair.
    ///
    /// Other manifolds and the order of the list are kept. Aging is not
    /// touched.
    pub(crate) fn swap_remove_body(&mut self, idx: usize, last: usize) {
        self.manifolds
            .retain(|m| m.pair.body_a as usize != idx && m.pair.body_b as usize != idx);
        if idx != last {
            for m in &mut self.manifolds {
                let a = m.pair.body_a as usize;
                let b = m.pair.body_b as usize;
                if a == last || b == last {
                    let a = if a == last { idx } else { a };
                    let b = if b == last { idx } else { b };
                    m.pair = BodyPairKey::new(a, b);
                    if a > b {
                        m.turn();
                    }
                }
            }
        }
        self.rebuild_pair_index();
    }

    /// Rebuild the `HashMap` index from `manifolds` (std feature only).
    fn rebuild_pair_index(&mut self) {
        #[cfg(feature = "std")]
        {
            self.pair_index.clear();
            for (idx, manifold) in self.manifolds.iter().enumerate() {
                self.pair_index.insert(manifold.pair, idx);
            }
        }
    }

    /// Total number of active manifolds
    #[inline]
    #[must_use]
    pub fn manifold_count(&self) -> usize {
        self.manifolds.len()
    }

    /// Total number of active contact points across all manifolds
    pub fn total_contact_points(&self) -> usize {
        self.manifolds
            .iter()
            .map(ContactManifold::point_count)
            .sum()
    }

    /// Clear all manifolds
    pub fn clear(&mut self) {
        self.manifolds.clear();
        #[cfg(feature = "std")]
        self.pair_index.clear();
    }

    /// Apply warm starting impulses to bodies
    ///
    /// Pre-applies accumulated impulses from the previous frame's solution,
    /// scaled by `warm_start_factor`. This dramatically improves convergence.
    pub fn apply_warm_start(&self, bodies: &mut [crate::solver::RigidBody]) {
        let factor = self.warm_start_factor;

        for manifold in &self.manifolds {
            let a_idx = manifold.pair.body_a as usize;
            let b_idx = manifold.pair.body_b as usize;

            if a_idx >= bodies.len() || b_idx >= bodies.len() {
                continue;
            }

            for point in &manifold.points {
                let impulse_n = manifold.normal * (point.lambda_n * factor);

                // Build tangent frame
                let (t1, t2) = tangent_frame(manifold.normal);
                let impulse_t = t1 * (point.lambda_t1 * factor) + t2 * (point.lambda_t2 * factor);

                let total_impulse = impulse_n + impulse_t;

                if !bodies[a_idx].inv_mass.is_zero() {
                    bodies[a_idx].velocity =
                        bodies[a_idx].velocity + total_impulse * bodies[a_idx].inv_mass;
                }
                if !bodies[b_idx].inv_mass.is_zero() {
                    bodies[b_idx].velocity =
                        bodies[b_idx].velocity - total_impulse * bodies[b_idx].inv_mass;
                }
            }
        }
    }
}

impl Default for ContactCache {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ContactCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("ContactCache");
        s.field(
            "manifolds",
            &format_args!("[{} items]", self.manifolds.len()),
        );
        #[cfg(feature = "std")]
        s.field(
            "pair_index",
            &format_args!("[{} items]", self.pair_index.len()),
        );
        s.field("max_stale_frames", &self.max_stale_frames);
        s.field("warm_start_factor", &self.warm_start_factor);
        s.finish()
    }
}

/// Build orthonormal tangent frame from a normal vector (crate-internal helper).
#[must_use]
pub(crate) fn tangent_frame(normal: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
    // Pick axis least parallel to normal
    let abs_x = normal.x.abs();
    let abs_y = normal.y.abs();
    let abs_z = normal.z.abs();

    let reference = if abs_x <= abs_y && abs_x <= abs_z {
        Vec3Fix::UNIT_X
    } else if abs_y <= abs_z {
        Vec3Fix::UNIT_Y
    } else {
        Vec3Fix::UNIT_Z
    };

    let t1 = normal.cross(reference).normalize();
    let t2 = normal.cross(t1);
    (t1, t2)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::solver::RigidBody;

    #[test]
    fn test_body_pair_key_canonical() {
        let k1 = BodyPairKey::new(3, 7);
        let k2 = BodyPairKey::new(7, 3);
        assert_eq!(k1, k2);
        assert_eq!(k1.body_a, 3);
        assert_eq!(k1.body_b, 7);
    }

    #[test]
    fn test_manifold_add_point() {
        let pair = BodyPairKey::new(0, 1);
        let mut manifold =
            ContactManifold::new(pair, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));

        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };

        manifold.add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);
        assert_eq!(manifold.point_count(), 1);
    }

    #[test]
    fn test_manifold_max_points() {
        let pair = BodyPairKey::new(0, 1);
        let mut manifold =
            ContactManifold::new(pair, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));

        // Add 5 distinct points — should cap at 4
        for i in 0..5 {
            let contact = Contact {
                depth: Fix128::from_int(i as i64 + 1),
                normal: Vec3Fix::UNIT_Y,
                point_a: Vec3Fix::from_int(i as i64 * 10, 0, 0),
                point_b: Vec3Fix::from_int(i as i64 * 10, 0, 0),
            };
            manifold.add_or_update(
                &contact,
                Vec3Fix::from_int(i as i64 * 10, 0, 0),
                Vec3Fix::from_int(i as i64 * 10, 0, 0),
            );
        }

        assert!(manifold.point_count() <= MAX_MANIFOLD_POINTS);
    }

    #[test]
    fn test_manifold_warm_start_preserved() {
        let pair = BodyPairKey::new(0, 1);
        let mut manifold =
            ContactManifold::new(pair, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));

        let contact = Contact {
            depth: Fix128::from_ratio(1, 10),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        };
        manifold.add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);

        // Store impulses
        manifold.store_impulses(0, Fix128::from_int(5), Fix128::ONE, Fix128::ONE);

        // Update same contact point — impulses should be preserved
        manifold.add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);
        let (ln, lt1, lt2) = manifold.warm_start_impulse(0);
        assert_eq!(ln.hi, 5);
        assert_eq!(lt1.hi, 1);
        assert_eq!(lt2.hi, 1);
    }

    #[test]
    fn test_contact_cache_lifecycle() {
        let mut cache = ContactCache::new();

        let pair = BodyPairKey::new(0, 1);
        {
            let manifold =
                cache.get_or_create(pair, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));
            let contact = Contact {
                depth: Fix128::from_ratio(1, 10),
                normal: Vec3Fix::UNIT_Y,
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            };
            manifold.add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);
        }

        assert_eq!(cache.manifold_count(), 1);
        assert_eq!(cache.total_contact_points(), 1);

        // Simulate stale frames — should expire after max_stale_frames
        for _ in 0..5 {
            cache.begin_frame();
            cache.end_frame();
        }

        assert_eq!(cache.manifold_count(), 0);
    }

    #[test]
    fn test_tangent_frame() {
        let (t1, t2) = tangent_frame(Vec3Fix::UNIT_Y);

        // t1 and t2 should be perpendicular to normal and each other
        let dot_n_t1 = Vec3Fix::UNIT_Y.dot(t1);
        let dot_n_t2 = Vec3Fix::UNIT_Y.dot(t2);
        let dot_t1_t2 = t1.dot(t2);

        assert!(dot_n_t1.abs() < Fix128::from_ratio(1, 100));
        assert!(dot_n_t2.abs() < Fix128::from_ratio(1, 100));
        assert!(dot_t1_t2.abs() < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn test_warm_start_application() {
        let mut cache = ContactCache::new();
        let pair = BodyPairKey::new(0, 1);

        {
            let manifold =
                cache.get_or_create(pair, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));
            let contact = Contact {
                depth: Fix128::from_ratio(1, 10),
                normal: Vec3Fix::UNIT_Y,
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            };
            manifold.add_or_update(&contact, Vec3Fix::ZERO, Vec3Fix::ZERO);
            manifold.store_impulses(0, Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        }

        let mut bodies = vec![
            RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        ];

        cache.apply_warm_start(&mut bodies);

        // Body 0 should have gained upward velocity, body 1 downward
        assert!(bodies[0].velocity.y > Fix128::ZERO);
        assert!(bodies[1].velocity.y < Fix128::ZERO);
    }

    // ---- mutation-score tests (2026-09-15) ----------------------------

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

    fn manifold() -> ContactManifold {
        ContactManifold::new(
            BodyPairKey::new(0, 1),
            Fix128::from_ratio(1, 2),
            Fix128::ZERO,
        )
    }

    #[test]
    fn manifold_add_matches_within_1cm_and_preserves_impulses() {
        let mut m = manifold();
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(1)), v3i(1, 0, 0), v3i(1, 1, 0));
        assert_eq!(m.point_count(), 1);
        assert!(!m.is_empty());
        m.store_impulses(0, fi(5), fi(6), fi(7));
        // 0.5 cm ずれた点 = 同一点として更新 (impulse 保持、age +1、depth 更新)
        let near = Vec3Fix::new(
            Fix128::ONE + Fix128::from_ratio(1, 200),
            Fix128::ZERO,
            Fix128::ZERO,
        );
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(2)), near, v3i(1, 1, 0));
        assert_eq!(m.point_count(), 1);
        assert_eq!(m.warm_start_impulse(0), (fi(5), fi(6), fi(7)));
        assert_eq!(m.points[0].age, 1);
        assert_eq!(m.points[0].depth, fi(2));
        assert_eq!(m.points[0].local_point_a, near);
        // 更新後の点 (1.005) から 2 cm 離れた点は新規点 (1 cm 閾値の外)
        let edge = Vec3Fix::new(
            Fix128::ONE + Fix128::from_ratio(5, 200),
            Fix128::ZERO,
            Fix128::ZERO,
        );
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(1)), edge, v3i(1, 1, 0));
        assert_eq!(m.point_count(), 2);
        assert_eq!(
            m.warm_start_impulse(1),
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
        );
        // 範囲外 index は 0、store は無視
        assert_eq!(
            m.warm_start_impulse(9),
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
        );
        m.store_impulses(9, fi(1), fi(1), fi(1));
        assert_eq!(m.point_count(), 2);
        m.clear();
        assert!(m.is_empty());
        assert_eq!(m.point_count(), 0);
        assert_eq!(m.normal, Vec3Fix::ZERO);
    }

    #[test]
    fn manifold_picks_nearest_of_several_candidates() {
        let mut m = manifold();
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            v3i(0, 0, 0),
            Vec3Fix::ZERO,
        );
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            v3i(5, 0, 0),
            Vec3Fix::ZERO,
        );
        m.store_impulses(0, fi(10), Fix128::ZERO, Fix128::ZERO);
        m.store_impulses(1, fi(20), Fix128::ZERO, Fix128::ZERO);
        // (5.004, 0, 0) は点 1 に一致 → 点 1 の impulse 20 が残り、点 0 は不変
        let p = Vec3Fix::new(
            fi(5) + Fix128::from_ratio(1, 250),
            Fix128::ZERO,
            Fix128::ZERO,
        );
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(3)), p, Vec3Fix::ZERO);
        assert_eq!(m.point_count(), 2);
        assert_eq!(m.warm_start_impulse(1).0, fi(20));
        assert_eq!(m.points[1].depth, fi(3));
        assert_eq!(m.points[0].depth, fi(1));
    }

    #[test]
    fn manifold_full_replaces_shallowest_only_if_deeper() {
        let mut m = manifold();
        for (i, d) in [3, 1, 4, 2].iter().enumerate() {
            m.add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(*d)),
                v3i(i as i64 * 10, 0, 0),
                Vec3Fix::ZERO,
            );
        }
        assert_eq!(m.point_count(), MAX_MANIFOLD_POINTS);
        // 5 点目 depth 0.5 < 最浅 1 → 追加されない
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 2)),
            v3i(100, 0, 0),
            Vec3Fix::ZERO,
        );
        assert_eq!(m.point_count(), 4);
        assert!(m.points.iter().all(|p| p.local_point_a.x != fi(100)));
        // depth 1 == 最浅 1 → `>` は false → 追加されない
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            v3i(100, 0, 0),
            Vec3Fix::ZERO,
        );
        assert!(m.points.iter().all(|p| p.local_point_a.x != fi(100)));
        // depth 2.5 > 最浅 1 (index 1) → index 1 が置換される
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, Fix128::from_ratio(5, 2)),
            v3i(100, 0, 0),
            Vec3Fix::ZERO,
        );
        assert_eq!(m.point_count(), 4);
        assert_eq!(m.points[1].local_point_a, v3i(100, 0, 0));
        assert_eq!(m.points[1].depth, Fix128::from_ratio(5, 2));
        let depths: Vec<Fix128> = m.points.iter().map(|p| p.depth).collect();
        assert_eq!(depths, vec![fi(3), Fix128::from_ratio(5, 2), fi(4), fi(2)]);
    }

    #[test]
    fn manifold_normal_is_normalized_average_of_point_normals() {
        let mut m = manifold();
        m.add_or_update(&contact(v3i(3, 0, 0), fi(1)), v3i(0, 0, 0), Vec3Fix::ZERO);
        assert_eq!(m.normal, Vec3Fix::UNIT_X); // (3,0,0)/3
        m.add_or_update(&contact(v3i(0, 4, 0), fi(1)), v3i(10, 0, 0), Vec3Fix::ZERO);
        // sum (3,4,0)、len 5 → (0.6, 0.8, 0)
        assert_eq!(
            m.normal,
            Vec3Fix::new(fi(3) / fi(5), fi(4) / fi(5), Fix128::ZERO)
        );
        // 打ち消し合う normal → len 0 → 直前の normal を維持
        let before = m.normal;
        m.add_or_update(
            &contact(v3i(-3, -4, 0), fi(1)),
            v3i(20, 0, 0),
            Vec3Fix::ZERO,
        );
        assert_eq!(m.normal, before);
        assert_eq!(m.stale_frames, 0);
    }

    #[test]
    fn cache_get_or_create_find_frames_and_counts() {
        let mut cache = ContactCache::new();
        let k01 = BodyPairKey::new(0, 1);
        let k23 = BodyPairKey::new(2, 3);
        cache
            .get_or_create(k01, Fix128::ONE, Fix128::ZERO)
            .add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
            );
        cache
            .get_or_create(k01, Fix128::ONE, Fix128::ZERO)
            .add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                v3i(5, 0, 0),
                Vec3Fix::ZERO,
            );
        cache
            .get_or_create(k23, Fix128::ONE, Fix128::ZERO)
            .add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
            );
        assert_eq!(cache.manifold_count(), 2, "同じ pair は再利用");
        assert_eq!(cache.total_contact_points(), 3);
        assert_eq!(cache.find(&k01).map(ContactManifold::point_count), Some(2));
        assert_eq!(cache.find(&k23).map(ContactManifold::point_count), Some(1));
        assert!(cache.find(&BodyPairKey::new(4, 5)).is_none());
        // 順序違いの key は同一 (正規化)
        assert_eq!(BodyPairKey::new(1, 0), k01);
        assert!(cache.find(&BodyPairKey::new(1, 0)).is_some());
        // stale: begin_frame で +1、max 3 を超えると end_frame で落ちる (k23 だけ touch し続ける)
        for _ in 0..3 {
            cache.begin_frame();
            cache
                .get_or_create(k23, Fix128::ONE, Fix128::ZERO)
                .add_or_update(
                    &contact(Vec3Fix::UNIT_Y, fi(1)),
                    Vec3Fix::ZERO,
                    Vec3Fix::ZERO,
                );
            cache.end_frame();
        }
        assert_eq!(cache.manifold_count(), 2, "stale 3 == max 3 は残る");
        cache.begin_frame();
        cache.end_frame();
        assert_eq!(cache.manifold_count(), 1, "stale 4 > 3 で落ちる");
        assert!(cache.find(&k01).is_none());
        assert!(cache.find(&k23).is_some(), "index は再構築される");
        assert_eq!(cache.total_contact_points(), 1);
        cache.clear();
        assert_eq!(cache.manifold_count(), 0);
        assert_eq!(cache.total_contact_points(), 0);
        assert!(cache.find(&k23).is_none());
        let dbg = format!("{cache:?}");
        assert!(
            dbg.contains("ContactCache") && dbg.contains("max_stale_frames: 3"),
            "{dbg}"
        );
    }

    #[test]
    fn apply_warm_start_applies_scaled_cached_impulses() {
        let mut cache = ContactCache::new();
        cache.warm_start_factor = Fix128::from_ratio(1, 2);
        let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Z, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
        // normal z: tangent_frame(z) = (z × x = y, z × y = -x) → t1 = (0,1,0), t2 = (-1,0,0)
        m.store_impulses(0, fi(4), fi(2), fi(6));
        let mut bodies = vec![
            crate::solver::RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            crate::solver::RigidBody::new_dynamic(v3i(1, 0, 0), Fix128::ONE),
        ];
        bodies[1].inv_mass = fi(3);
        cache.apply_warm_start(&mut bodies);
        // total = z*(4*0.5) + y*(2*0.5) + (-x)*(6*0.5) = (-3, 1, 2)、A += total*1、B -= total*3
        assert_eq!(bodies[0].velocity, v3i(-3, 1, 2));
        assert_eq!(bodies[1].velocity, v3i(9, -3, -6));
        // static 側は不変、body index 範囲外の manifold は skip
        let mut st = vec![
            crate::solver::RigidBody::new_static(Vec3Fix::ZERO),
            crate::solver::RigidBody::new_dynamic(v3i(1, 0, 0), Fix128::ONE),
        ];
        cache.apply_warm_start(&mut st);
        assert_eq!(st[0].velocity, Vec3Fix::ZERO);
        assert_eq!(st[1].velocity, v3i(3, -1, -2));
        let mut short = vec![crate::solver::RigidBody::new_dynamic(
            Vec3Fix::ZERO,
            Fix128::ONE,
        )];
        cache.apply_warm_start(&mut short);
        assert_eq!(short[0].velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn tangent_frame_picks_least_parallel_axis_and_is_orthonormal() {
        // normal = x → reference は y (abs_x が最小でない、abs_y <= abs_z) → t1 = x × y = z、t2 = x × z = -y
        assert_eq!(
            tangent_frame(Vec3Fix::UNIT_X),
            (Vec3Fix::UNIT_Z, -Vec3Fix::UNIT_Y)
        );
        // normal = y → reference x (abs_x <= abs_y && abs_x <= abs_z) → t1 = y × x = -z、t2 = y × -z = -x
        assert_eq!(
            tangent_frame(Vec3Fix::UNIT_Y),
            (-Vec3Fix::UNIT_Z, -Vec3Fix::UNIT_X)
        );
        // normal = z → reference x → t1 = z × x = y、t2 = z × y = -x
        assert_eq!(
            tangent_frame(Vec3Fix::UNIT_Z),
            (Vec3Fix::UNIT_Y, -Vec3Fix::UNIT_X)
        );
        // 同率 (abs_x == abs_y == abs_z): `<=` で x が選ばれる
        let d = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        let (t1, t2) = tangent_frame(d);
        let expect_t1 = d.cross(Vec3Fix::UNIT_X).normalize();
        assert_eq!(t1, expect_t1);
        assert_eq!(t2, d.cross(t1));
        // 一般 normal: t1 ⊥ n、t2 ⊥ n、t1 ⊥ t2 (数 ulp)
        let n = Vec3Fix::new(
            Fix128::from_ratio(1, 3),
            Fix128::from_ratio(-2, 3),
            Fix128::from_ratio(2, 3),
        );
        let (t1, t2) = tangent_frame(n);
        let small = Fix128 { hi: 0, lo: 1 << 24 };
        assert!(n.dot(t1).abs() < small && n.dot(t2).abs() < small && t1.dot(t2).abs() < small);
        assert!((t1.length() - Fix128::ONE).abs() < small);
    }

    #[test]
    fn apply_warm_start_factor_scales_normal_and_both_tangents() {
        let mut cache = ContactCache::new();
        cache.warm_start_factor = Fix128::from_ratio(1, 4);
        let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_X, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
        // normal x: t1 = x × y = z、t2 = x × z = -y
        m.store_impulses(0, fi(8), fi(12), fi(20));
        let mut bodies = vec![
            crate::solver::RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            crate::solver::RigidBody::new_static(v3i(1, 0, 0)),
        ];
        cache.apply_warm_start(&mut bodies);
        // total = x*2 + z*3 + (-y)*5 = (2, -5, 3)
        assert_eq!(bodies[0].velocity, v3i(2, -5, 3));
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
    }

    // ---- mutation-score tests batch 7 (2026-09-15、cargo-mutants missed 分) ----

    /// `dist_sq < best_dist_sq` (厳密 less-than): 2 点と等距離の新規点は
    /// 先に見つかった index 0 に一致する (`<=` なら後勝ちで index 1 が更新される)
    #[test]
    fn manifold_match_tie_prefers_first_candidate() {
        let mut m = manifold();
        // A at 0、B at 1/64 (= 1.56 cm 離れ → 1 cm 閾値の外、別点として登録)
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
        let b = Vec3Fix::new(Fix128::from_ratio(1, 64), Fix128::ZERO, Fix128::ZERO);
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(2)), b, Vec3Fix::ZERO);
        assert_eq!(m.point_count(), 2);
        // 中点 1/128: 両点との dist_sq が厳密に 1/16384 で等しい (dyadic、丸めなし)
        let mid = Vec3Fix::new(Fix128::from_ratio(1, 128), Fix128::ZERO, Fix128::ZERO);
        let d_a = (m.points[0].local_point_a - mid).length_squared();
        let d_b = (m.points[1].local_point_a - mid).length_squared();
        assert_eq!(d_a, d_b);
        assert_eq!(d_a, Fix128::from_ratio(1, 16384));
        m.add_or_update(&contact(Vec3Fix::UNIT_Y, fi(3)), mid, Vec3Fix::ZERO);
        assert_eq!(m.point_count(), 2);
        assert_eq!(m.points[0].depth, fi(3), "先勝ち: index 0 が更新される");
        assert_eq!(m.points[0].local_point_a, mid);
        assert_eq!(m.points[0].age, 1);
        assert_eq!(m.points[1].depth, fi(2), "index 1 は不変");
        assert_eq!(m.points[1].local_point_a, b);
        assert_eq!(m.points[1].age, 0);
    }

    /// 満杯時の最浅探索 `p.depth < shallowest_depth` (厳密): 最浅が同点なら
    /// 先頭 (index 0) が置換される (`<=` なら最後の同点 index 1 が置換される)
    #[test]
    fn manifold_full_replaces_first_of_equal_shallowest() {
        let mut m = manifold();
        for (i, d) in [1, 1, 3, 4].iter().enumerate() {
            m.add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(*d)),
                v3i(i as i64 * 10, 0, 0),
                Vec3Fix::ZERO,
            );
        }
        assert_eq!(m.point_count(), MAX_MANIFOLD_POINTS);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(2)),
            v3i(100, 0, 0),
            Vec3Fix::ZERO,
        );
        assert_eq!(m.point_count(), 4);
        assert_eq!(m.points[0].local_point_a, v3i(100, 0, 0));
        assert_eq!(m.points[0].depth, fi(2));
        assert_eq!(m.points[1].local_point_a, v3i(10, 0, 0), "同点の後方は不変");
        assert_eq!(m.points[1].depth, fi(1));
        let depths: Vec<Fix128> = m.points.iter().map(|p| p.depth).collect();
        assert_eq!(depths, vec![fi(2), fi(1), fi(3), fi(4)]);
    }

    /// `point_idx < len` の境界: idx == len は範囲外 (`<=` なら index out of bounds panic)
    #[test]
    fn warm_start_and_store_at_exact_len_are_out_of_range() {
        let mut m = manifold();
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
        m.store_impulses(0, fi(2), fi(3), fi(4));
        assert_eq!(m.point_count(), 1);
        // idx 1 == len 1
        assert_eq!(
            m.warm_start_impulse(1),
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
        );
        m.store_impulses(1, fi(9), fi(9), fi(9));
        assert_eq!(m.point_count(), 1);
        assert_eq!(m.warm_start_impulse(0), (fi(2), fi(3), fi(4)));
        // 空 manifold: idx 0 == len 0
        let e = manifold();
        assert_eq!(
            e.warm_start_impulse(0),
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
        );
        let mut e2 = manifold();
        e2.store_impulses(0, fi(1), fi(1), fi(1));
        assert!(e2.is_empty());
    }

    /// body A の `total_impulse * inv_mass` (inv_mass ≠ 1) と tangent `t1 * (λ * factor)`
    /// (λ * factor ≠ 1): 乗算と除算で結果が異なる値を使う
    #[test]
    fn apply_warm_start_scales_by_body_a_inv_mass() {
        let mut cache = ContactCache::new();
        cache.warm_start_factor = Fix128::from_ratio(1, 2);
        let m = cache.get_or_create(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Z, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
        // normal z: t1 = z × x = y、t2 = z × y = -x、λ_t1 = 6 → 6 * 0.5 = 3、他 0
        m.store_impulses(0, Fix128::ZERO, fi(6), Fix128::ZERO);
        let mut bodies = vec![
            crate::solver::RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
            crate::solver::RigidBody::new_static(v3i(1, 0, 0)),
        ];
        bodies[0].inv_mass = fi(4);
        cache.apply_warm_start(&mut bodies);
        // total = y * 3、A += total * 4 = (0, 12, 0)  (÷ なら (0, 3/4, 0) / t1 ÷ 3 なら (0, 4/3, 0))
        assert_eq!(bodies[0].velocity, v3i(0, 12, 0));
        assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
    }

    /// reference 軸選択 `abs_x <= abs_y && abs_x <= abs_z`: 片側だけ真の normal で
    /// X が選ばれない (`||` なら X が選ばれ t1 の零成分が変わる)
    #[test]
    fn tangent_frame_reference_requires_both_comparisons() {
        // (2, 4, 1): abs_x <= abs_y 真、abs_x <= abs_z 偽 → else-if 4 <= 1 偽 → Z
        //   t1 = normalize(n × Z) = normalize((4, -2, 0)) → z == 0、x > 0、y < 0
        //   (X 参照なら n × X = (0, 1, -4) → x == 0)
        let n = v3i(2, 4, 1);
        let (t1, t2) = tangent_frame(n);
        assert_eq!(t1.z, Fix128::ZERO);
        assert!(t1.x > Fix128::ZERO && t1.y < Fix128::ZERO, "{t1:?}");
        assert_eq!(t2, n.cross(t1));
        // (2, 1, 4): abs_x <= abs_y 偽、abs_x <= abs_z 真 → else-if 1 <= 4 真 → Y
        //   t1 = normalize(n × Y) = normalize((-4, 0, 2)) → y == 0、x < 0、z > 0
        //   (X 参照なら n × X = (0, 4, -1) → x == 0)
        let n2 = v3i(2, 1, 4);
        let (u1, u2) = tangent_frame(n2);
        assert_eq!(u1.y, Fix128::ZERO);
        assert!(u1.x < Fix128::ZERO && u1.z > Fix128::ZERO, "{u1:?}");
        assert_eq!(u2, n2.cross(u1));
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_stale_index_found_by_the_scan_is_re_indexed() {
        let mut cache = ContactCache::new();
        let (a, b) = (BodyPairKey::new(0, 1), BodyPairKey::new(2, 3));
        cache.get_or_create(a, Fix128::ONE, Fix128::ZERO);
        cache.get_or_create(b, Fix128::ONE, Fix128::ZERO);
        cache.manifolds.remove(0); // b moves to slot 0, its index still says 1
        assert_eq!(cache.pair_index.get(&b), Some(&1));
        assert_eq!(cache.get_or_create(b, Fix128::ONE, Fix128::ZERO).pair, b);
        assert_eq!(cache.pair_index.get(&b), Some(&0));
        assert_eq!(cache.manifolds.len(), 1, "no duplicate manifold for b");
    }

    /// Cache holding `pairs` in order, each with friction `k + 1` (k = its
    /// position), so a manifold's origin stays visible after re-keying.
    fn cache_of(pairs: &[(usize, usize)]) -> ContactCache {
        let mut cache = ContactCache::new();
        for (k, &(a, b)) in pairs.iter().enumerate() {
            cache.get_or_create(
                BodyPairKey::new(a, b),
                Fix128::from_int(k as i64 + 1),
                Fix128::ZERO,
            );
        }
        cache
    }

    #[test]
    fn swap_remove_body_drops_the_removed_and_re_keys_the_moved_body() {
        // remove 1 of 5 (last = 4): (0,1) (1,3) (1,4) dropped; (0,4) -> (0,1)
        // keeps order; (2,4) -> (1,2) and (3,4) -> (1,3) re-sorted; (2,3) kept
        let pairs = [(0, 1), (0, 4), (2, 4), (1, 3), (2, 3), (1, 4), (4, 3)];
        let mut cache = cache_of(&pairs);
        cache.swap_remove_body(1, 4);
        let got: Vec<(u32, u32, Fix128)> = cache
            .manifolds
            .iter()
            .map(|m| (m.pair.body_a, m.pair.body_b, m.friction))
            .collect();
        let f = |k: i64| Fix128::from_int(k + 1);
        assert_eq!(
            got,
            vec![(0, 1, f(1)), (1, 2, f(2)), (2, 3, f(4)), (1, 3, f(6))]
        );
        assert_eq!(cache.pair_index.len(), 4);
        for (slot, m) in cache.manifolds.iter().enumerate() {
            assert_eq!(cache.pair_index.get(&m.pair), Some(&slot));
        }
        // the removed body's (1,4) and the moved body's old (0,4) are gone
        assert!(cache.find(&BodyPairKey::new(1, 4)).is_none());
        assert!(cache.find(&BodyPairKey::new(0, 4)).is_none());
    }

    #[test]
    fn swap_remove_body_of_the_last_index_only_drops_its_pairs() {
        let mut cache = cache_of(&[(0, 4), (1, 2), (4, 2), (3, 1)]);
        cache.swap_remove_body(4, 4);
        let got: Vec<BodyPairKey> = cache.manifolds.iter().map(|m| m.pair).collect();
        assert_eq!(got, vec![BodyPairKey::new(1, 2), BodyPairKey::new(1, 3)]);
        assert_eq!(cache.pair_index.len(), 2);
        assert_eq!(cache.pair_index.get(&BodyPairKey::new(1, 3)), Some(&1));
    }

    fn point(k: i64) -> CachedContactPoint {
        let mut p = CachedContactPoint::new(
            Vec3Fix::new(Fix128::from_ratio(k, 3), Fix128::ONE, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-k, 5), Fix128::ONE),
            Vec3Fix::new(
                Fix128::from_ratio(1, 3),
                Fix128::from_ratio(k, 7),
                Fix128::from_ratio(-2, 3),
            )
            .normalize(),
            Fix128::from_ratio(k + 1, 100),
        );
        p.lambda_n = Fix128::from_ratio(k + 2, 3);
        p.lambda_t1 = Fix128::from_ratio(-k, 11);
        p.lambda_t2 = Fix128::from_ratio(k + 5, 13);
        p.age = k as u32;
        p
    }

    #[test]
    fn a_turned_point_swaps_the_points_turns_the_normal_and_negates_lambda_t2() {
        let p = point(3);
        let t = p.turned();
        assert_eq!(t.local_point_a, p.local_point_b);
        assert_eq!(t.local_point_b, p.local_point_a);
        assert_eq!(t.normal, -p.normal);
        assert_eq!(
            (t.depth, t.lambda_n, t.lambda_t1, t.age),
            (p.depth, p.lambda_n, p.lambda_t1, p.age)
        );
        assert_eq!(t.lambda_t2, -p.lambda_t2);
        assert_eq!(t.turned(), p, "turning twice gives the point back");
    }

    #[test]
    fn the_frame_of_a_turned_normal_is_minus_t1_and_t2_bit_for_bit() {
        // the claim `turned` rests on: tangent_frame(-n) == (-t1, t2) exactly
        for i in -6i64..=6 {
            for j in [-5i64, -1, 2, 9] {
                let n = Vec3Fix::new(
                    Fix128::from_ratio(i, 7),
                    Fix128::from_ratio(j, 5),
                    Fix128::from_ratio(3, 11),
                )
                .normalize();
                let (t1, t2) = tangent_frame(n);
                assert_eq!(tangent_frame(-n), (-t1, t2), "n = ({i}/7, {j}/5, 3/11)");
            }
        }
    }

    #[test]
    fn a_turned_manifold_pushes_each_body_the_same_way() {
        // turning the data and exchanging the bodies' roles keeps the impulse
        // each body receives (to the rounding of the Fix128 products)
        let mut m = ContactManifold::new(BodyPairKey::new(0, 1), Fix128::ONE, Fix128::ZERO);
        m.points.push(point(2));
        m.update_normal();
        let mut cache = ContactCache::new();
        cache.warm_start_factor = Fix128::ONE;
        cache.manifolds.push(m.clone());
        let bodies = || {
            vec![
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)),
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(4)),
            ]
        };
        let mut before = bodies();
        cache.apply_warm_start(&mut before);
        let mut turned = m.clone();
        turned.turn();
        assert_eq!(turned.normal, -m.normal);
        // the turned data describes body 1 as A: apply with the roles swapped
        let mut swapped = vec![bodies()[1], bodies()[0]];
        cache.manifolds[0] = turned.clone();
        cache.apply_warm_start(&mut swapped);
        let tol = Fix128::from_raw(0, 1 << 4);
        for (x, y) in [(&before[0], &swapped[1]), (&before[1], &swapped[0])] {
            let d = x.velocity - y.velocity;
            assert!(
                d.x.abs() <= tol && d.y.abs() <= tol && d.z.abs() <= tol,
                "{d:?}"
            );
            assert!(!x.velocity.length_squared().is_zero());
        }
        turned.turn();
        assert_eq!(turned.points, m.points);
        assert_eq!(turned.normal, m.normal);
    }

    #[test]
    fn oriented_to_key_turns_only_a_contact_reported_with_the_larger_index_first() {
        let c = Contact {
            depth: Fix128::from_ratio(1, 8),
            normal: Vec3Fix::UNIT_Y,
            point_a: Vec3Fix::UNIT_X,
            point_b: Vec3Fix::UNIT_Z,
        };
        assert_eq!(oriented_to_key(1, 4, &c), c);
        assert_eq!(oriented_to_key(3, 3, &c), c);
        let t = oriented_to_key(4, 1, &c);
        assert_eq!(
            (t.normal, t.point_a, t.point_b, t.depth),
            (-c.normal, c.point_b, c.point_a, c.depth)
        );
    }

    #[test]
    fn swap_remove_body_turns_a_moved_pair_whose_order_flips() {
        // remove 1 of 5: (2,4) -> (1,2) exchanges A and B and is turned;
        // (0,4) -> (0,1) keeps the order and the data
        let mut cache = ContactCache::new();
        for (k, &(a, b)) in [(0usize, 4usize), (2, 4)].iter().enumerate() {
            cache
                .get_or_create(BodyPairKey::new(a, b), Fix128::ONE, Fix128::ZERO)
                .points
                .push(point(k as i64));
        }
        cache.manifolds[0].update_normal();
        cache.manifolds[1].update_normal();
        let kept = cache.manifolds[0].clone();
        let mut flipped = cache.manifolds[1].clone();
        cache.swap_remove_body(1, 4);
        assert_eq!(cache.manifolds[0].points, kept.points);
        assert_eq!(cache.manifolds[0].normal, kept.normal);
        flipped.turn();
        assert_eq!(cache.manifolds[1].pair, BodyPairKey::new(1, 2));
        assert_eq!(cache.manifolds[1].points, flipped.points);
        assert_eq!(cache.manifolds[1].normal, flipped.normal);
    }
}
