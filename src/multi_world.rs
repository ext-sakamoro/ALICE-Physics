//! Multiple Independent Physics Worlds
//!
//! Manages several independent physics worlds that can run simultaneously.
//! Supports transferring rigid bodies between worlds and portal connections.
//!
//! # Features
//!
//! - Create and manage multiple independent physics worlds
//! - Step all worlds sequentially or in parallel (`parallel` feature)
//! - Transfer bodies between worlds with position remapping
//! - Portal connections between world pairs

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::{PhysicsConfig, PhysicsWorld};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// Multi-World Types
// ============================================================================

/// Container managing multiple independent physics worlds.
pub struct MultiWorld {
    /// All physics worlds
    pub worlds: Vec<PhysicsWorld>,
}

/// A portal connecting two worlds.
///
/// Objects passing through the portal in world A appear in world B
/// at a transformed position/orientation, and vice versa.
#[derive(Clone, Debug)]
pub struct Portal {
    /// Index of the first world
    pub world_a: usize,
    /// Index of the second world
    pub world_b: usize,
    /// Transform applied when crossing from A to B: (translation, rotation)
    pub transform: (Vec3Fix, QuatFix),
}

impl MultiWorld {
    /// Create a new empty multi-world container.
    #[must_use]
    pub const fn new() -> Self {
        Self { worlds: Vec::new() }
    }

    /// Add a new physics world with the given configuration.
    ///
    /// Returns the index of the newly added world.
    pub fn add_world(&mut self, config: PhysicsConfig) -> usize {
        let idx = self.worlds.len();
        self.worlds.push(PhysicsWorld::new(config));
        idx
    }

    /// Step all worlds sequentially.
    ///
    /// Each world is advanced by `dt` independently.
    pub fn step_all(&mut self, dt: Fix128) {
        for world in &mut self.worlds {
            world.step(dt);
        }
    }

    /// Step all worlds in parallel using Rayon.
    ///
    /// Each world is advanced by `dt` independently on separate threads.
    /// Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn step_all_parallel(&mut self, dt: Fix128) {
        self.worlds.par_iter_mut().for_each(|world| {
            world.step(dt);
        });
    }

    /// Transfer a rigid body from one world to another.
    ///
    /// Removes the body from `from_world` and adds it to `to_world`
    /// at the given `new_position`. Returns the body's new index in
    /// the destination world, or `None` if the indices are invalid.
    ///
    /// # Arguments
    ///
    /// - `from_world`: Source world index
    /// - `body_id`: Body index within the source world
    /// - `to_world`: Destination world index
    /// - `new_position`: Position in the destination world
    pub fn transfer_body(
        &mut self,
        from_world: usize,
        body_id: usize,
        to_world: usize,
        new_position: Vec3Fix,
    ) -> Option<usize> {
        if from_world >= self.worlds.len() || to_world >= self.worlds.len() {
            return None;
        }
        if from_world == to_world {
            return None;
        }

        // Remove body from source world
        // We need to handle the borrow checker: read then modify
        let body_opt = {
            let src = &mut self.worlds[from_world];
            if body_id >= src.bodies.len() {
                return None;
            }
            src.remove_body(body_id)
        };

        let mut body = body_opt?;

        // Update position for destination world
        body.position = new_position;
        body.prev_position = new_position;

        // Add to destination world
        let new_id = self.worlds[to_world].add_body(body);

        Some(new_id)
    }

    /// Get the number of worlds.
    #[must_use]
    pub fn world_count(&self) -> usize {
        self.worlds.len()
    }

    /// Get the total number of bodies across all worlds.
    #[must_use]
    pub fn total_body_count(&self) -> usize {
        self.worlds.iter().map(|w| w.bodies.len()).sum()
    }
}

impl Default for MultiWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for MultiWorld {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultiWorld")
            .field("worlds", &self.worlds.len())
            .finish()
    }
}

impl Portal {
    /// Create a new portal between two worlds.
    #[must_use]
    pub const fn new(
        world_a: usize,
        world_b: usize,
        translation: Vec3Fix,
        rotation: QuatFix,
    ) -> Self {
        Self {
            world_a,
            world_b,
            transform: (translation, rotation),
        }
    }

    /// Transform a position from world A coordinates to world B coordinates.
    #[must_use]
    pub fn transform_a_to_b(&self, position: Vec3Fix) -> Vec3Fix {
        let (translation, rotation) = &self.transform;
        rotation.rotate_vec(position) + *translation
    }

    /// Transform a position from world B coordinates to world A coordinates.
    #[must_use]
    pub fn transform_b_to_a(&self, position: Vec3Fix) -> Vec3Fix {
        let (translation, rotation) = &self.transform;
        let inv_rot = rotation.conjugate();
        inv_rot.rotate_vec(position - *translation)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::solver::RigidBody;

    #[test]
    fn test_new_multi_world() {
        let mw = MultiWorld::new();
        assert_eq!(mw.world_count(), 0);
    }

    #[test]
    fn test_default_multi_world() {
        let mw = MultiWorld::default();
        assert_eq!(mw.world_count(), 0);
    }

    #[test]
    fn test_add_world() {
        let mut mw = MultiWorld::new();
        let idx = mw.add_world(PhysicsConfig::default());
        assert_eq!(idx, 0);
        assert_eq!(mw.world_count(), 1);
    }

    #[test]
    fn test_add_multiple_worlds() {
        let mut mw = MultiWorld::new();
        let idx0 = mw.add_world(PhysicsConfig::default());
        let idx1 = mw.add_world(PhysicsConfig::default());
        let idx2 = mw.add_world(PhysicsConfig::default());
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(mw.world_count(), 3);
    }

    #[test]
    fn test_step_all_empty() {
        let mut mw = MultiWorld::new();
        mw.step_all(Fix128::from_ratio(1, 60));
        // Should not panic
    }

    #[test]
    fn test_step_all_with_body() {
        let mut mw = MultiWorld::new();
        mw.add_world(PhysicsConfig::default());

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        mw.worlds[0].add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        mw.step_all(dt);

        // Body should have moved under gravity
        let pos = mw.worlds[0].bodies[0].position;
        assert!(pos.y < Fix128::from_int(10));
    }

    #[test]
    fn test_transfer_body() {
        let mut mw = MultiWorld::new();
        mw.add_world(PhysicsConfig::default());
        mw.add_world(PhysicsConfig::default());

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
        mw.worlds[0].add_body(body);

        assert_eq!(mw.worlds[0].bodies.len(), 1);
        assert_eq!(mw.worlds[1].bodies.len(), 0);

        let new_pos = Vec3Fix::from_int(10, 20, 30);
        let new_id = mw.transfer_body(0, 0, 1, new_pos);

        assert!(new_id.is_some());
        assert_eq!(new_id.unwrap(), 0);
        assert_eq!(mw.worlds[0].bodies.len(), 0);
        assert_eq!(mw.worlds[1].bodies.len(), 1);
        assert_eq!(mw.worlds[1].bodies[0].position.x.hi, 10);
        assert_eq!(mw.worlds[1].bodies[0].position.y.hi, 20);
    }

    #[test]
    fn test_transfer_invalid_world() {
        let mut mw = MultiWorld::new();
        mw.add_world(PhysicsConfig::default());

        let result = mw.transfer_body(0, 0, 99, Vec3Fix::ZERO);
        assert!(result.is_none());
    }

    #[test]
    fn test_transfer_same_world() {
        let mut mw = MultiWorld::new();
        mw.add_world(PhysicsConfig::default());

        let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        mw.worlds[0].add_body(body);

        let result = mw.transfer_body(0, 0, 0, Vec3Fix::ZERO);
        assert!(result.is_none());
    }

    #[test]
    fn test_total_body_count() {
        let mut mw = MultiWorld::new();
        mw.add_world(PhysicsConfig::default());
        mw.add_world(PhysicsConfig::default());

        mw.worlds[0].add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        mw.worlds[0].add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        mw.worlds[1].add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

        assert_eq!(mw.total_body_count(), 3);
    }

    #[test]
    fn test_portal_transform_identity() {
        let portal = Portal::new(0, 1, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let pos = Vec3Fix::from_int(5, 10, 15);
        let transformed = portal.transform_a_to_b(pos);
        assert_eq!(transformed.x.hi, 5);
        assert_eq!(transformed.y.hi, 10);
        assert_eq!(transformed.z.hi, 15);
    }

    #[test]
    fn test_portal_transform_translation() {
        let portal = Portal::new(0, 1, Vec3Fix::from_int(100, 0, 0), QuatFix::IDENTITY);
        let pos = Vec3Fix::from_int(5, 10, 15);
        let transformed = portal.transform_a_to_b(pos);
        assert_eq!(transformed.x.hi, 105);
        assert_eq!(transformed.y.hi, 10);
    }

    #[test]
    fn test_portal_roundtrip() {
        let portal = Portal::new(0, 1, Vec3Fix::from_int(50, -20, 10), QuatFix::IDENTITY);
        let pos = Vec3Fix::from_int(7, 3, -5);
        let to_b = portal.transform_a_to_b(pos);
        let back = portal.transform_b_to_a(to_b);

        let eps = Fix128::from_ratio(1, 1000);
        assert!((back.x - pos.x).abs() < eps);
        assert!((back.y - pos.y).abs() < eps);
        assert!((back.z - pos.z).abs() < eps);
    }

    /// 3 world (それぞれ異なる gravity / 衝突 / joint 構成) を同じ内容で 2 組作る
    #[cfg(feature = "parallel")]
    fn build_scene() -> MultiWorld {
        let mut mw = MultiWorld::new();
        // world 0: 既定重力、球 2 個が衝突する
        let w0 = mw.add_world(PhysicsConfig::default());
        mw.worlds[w0].add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
            Fix128::ONE,
        );
        mw.worlds[w0].add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
        // world 1: 横向き重力、3 body が距離拘束で連結
        let w1 = mw.add_world(PhysicsConfig {
            gravity: Vec3Fix::from_int(3, 0, 0),
            ..PhysicsConfig::default()
        });
        let a = mw.worlds[w1].add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = mw.worlds[w1].add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, -2, 0),
            Fix128::ONE,
        ));
        let c = mw.worlds[w1].add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, -4, 0),
            Fix128::from_int(2),
        ));
        mw.worlds[w1].add_distance_constraint(crate::solver::DistanceConstraint::new(
            a,
            b,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(2),
        ));
        mw.worlds[w1].add_distance_constraint(crate::solver::DistanceConstraint::new(
            b,
            c,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(2),
        ));
        // world 2: 空 (step が no-op でも壊れない)
        mw.add_world(PhysicsConfig::default());
        mw
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn step_all_parallel_is_bit_identical_to_sequential_step_all() {
        let mut seq = build_scene();
        let mut par = build_scene();
        let dt = Fix128::from_ratio(1, 60);
        for frame in 0..30 {
            seq.step_all(dt);
            par.step_all_parallel(dt);
            assert_eq!(seq.world_count(), par.world_count());
            for (w, (ws, wp)) in seq.worlds.iter().zip(par.worlds.iter()).enumerate() {
                assert_eq!(ws.bodies.len(), wp.bodies.len(), "frame {frame} world {w}");
                for (i, (bs, bp)) in ws.bodies.iter().zip(wp.bodies.iter()).enumerate() {
                    assert_eq!(bs.position, bp.position, "frame {frame} world {w} body {i}");
                    assert_eq!(bs.velocity, bp.velocity, "frame {frame} world {w} body {i}");
                    assert_eq!(bs.rotation, bp.rotation, "frame {frame} world {w} body {i}");
                    assert_eq!(
                        bs.angular_velocity, bp.angular_velocity,
                        "frame {frame} world {w} body {i}"
                    );
                }
                assert_eq!(
                    ws.contact_events().len(),
                    wp.contact_events().len(),
                    "frame {frame} world {w}"
                );
            }
        }
        // 実際に何かが動いた (自明な一致ではない): world 0 は落下して球に接触、world 1 は +x へ振れる
        let w0 = &par.worlds[0].bodies[0].position;
        assert!(w0.y < Fix128::from_int(4) && w0.y > Fix128::ONE, "{w0:?}");
        assert!(par.worlds[1].bodies[2].position.x > Fix128::ZERO);
        // world 間の独立性: world 2 は空のまま、world 0 の body 数も不変
        assert_eq!(par.worlds[2].bodies.len(), 0);
        assert_eq!(par.total_body_count(), 5);
        // 空 MultiWorld でも parallel step は no-op
        let mut empty = MultiWorld::new();
        empty.step_all_parallel(dt);
        assert_eq!(empty.world_count(), 0);
    }
}
