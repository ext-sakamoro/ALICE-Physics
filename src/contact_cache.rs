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

/// Maximum contact points per manifold
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// A single cached contact point within a manifold
#[derive(Clone, Copy, Debug)]
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
    pub fn new(local_a: Vec3Fix, local_b: Vec3Fix, normal: Vec3Fix, depth: Fix128) -> Self {
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
    pub fn new(a: usize, b: usize) -> Self {
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

/// Contact manifold: up to 4 persistent contact points between two bodies
#[derive(Clone, Debug)]
pub struct ContactManifold {
    /// Body pair this manifold belongs to
    pub pair: BodyPairKey,
    /// Active contact points (up to MAX_MANIFOLD_POINTS)
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
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Check if this manifold is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear all points (reset manifold)
    pub fn clear(&mut self) {
        self.points.clear();
        self.normal = Vec3Fix::ZERO;
    }

    /// Get warm-start impulse for a contact point
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
    /// HashMap index for O(1) manifold lookup (std feature only)
    #[cfg(feature = "std")]
    pair_index: HashMap<BodyPairKey, usize>,
    /// Maximum stale frames before manifold is removed
    pub max_stale_frames: u32,
    /// Warm starting factor (0.0 = off, 1.0 = full warm start)
    pub warm_start_factor: Fix128,
}

impl ContactCache {
    /// Create a new contact cache
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
                self.manifolds.last_mut().unwrap()
            }
            #[cfg(not(feature = "std"))]
            {
                self.manifolds
                    .push(ContactManifold::new(pair, friction, restitution));
                self.manifolds.last_mut().unwrap()
            }
        }
    }

    /// Find manifold for a body pair (read-only)
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

/// Build orthonormal tangent frame from a normal vector
pub fn tangent_frame(normal: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
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

#[cfg(test)]
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
}
