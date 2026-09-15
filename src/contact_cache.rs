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
/// Marked `#[non_exhaustive]` so future warm-start / analytics fields
/// can be added without a breaking API change; construct via the internal
/// `CachedContactPoint::new` and inspect via public fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CachedContactPoint {
    /// Contact point on body A (local space)
    pub local_point_a: Vec3Fix,
    /// Contact point on body B (local space)
    pub local_point_b: Vec3Fix,
    /// Contact normal (world space, A→B)
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
    /// Shared normal direction (average of point normals)
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
    pair_index: HashMap<BodyPairKey, usize>,
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
        #[cfg(feature = "std")]
        let pos = self.pair_index.get(&pair).copied();

        #[cfg(not(feature = "std"))]
        let pos = self.manifolds.iter().position(|m| m.pair == pair);

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
    #[must_use]
    pub fn find(&self, pair: &BodyPairKey) -> Option<&ContactManifold> {
        // Use HashMap (O(1)) with std feature, or linear scan (O(n)) without
        #[cfg(feature = "std")]
        {
            self.pair_index.get(pair).map(|&idx| &self.manifolds[idx])
        }

        #[cfg(not(feature = "std"))]
        {
            self.manifolds.iter().find(|m| m.pair == *pair)
        }
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

        // Rebuild the HashMap index after retain (std feature only)
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
}
