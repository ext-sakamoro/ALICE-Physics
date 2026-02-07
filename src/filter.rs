//! Collision Filtering (Layer/Mask System)
//!
//! Bitmask-based collision filtering for controlling which bodies can interact.
//!
//! # Usage
//!
//! ```ignore
//! use alice_physics::filter::CollisionFilter;
//!
//! // Layer 0 = player, Layer 1 = enemy, Layer 2 = projectile
//! let player = CollisionFilter::new(1 << 0, (1 << 1) | (1 << 2)); // collides with enemy + projectile
//! let enemy  = CollisionFilter::new(1 << 1, (1 << 0) | (1 << 2)); // collides with player + projectile
//! let ghost  = CollisionFilter::new(1 << 3, 0);                    // collides with nothing
//! ```

/// Collision filter using layer/mask bitmasks.
///
/// Two bodies can collide iff:
///   `(a.layer & b.mask) != 0 && (b.layer & a.mask) != 0`
///
/// This provides fine-grained bidirectional control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollisionFilter {
    /// Which layer(s) this body belongs to (bitmask)
    pub layer: u32,
    /// Which layers this body can collide with (bitmask)
    pub mask: u32,
    /// Collision group ID (bodies in the same non-zero group never collide)
    pub group: u32,
}

impl CollisionFilter {
    /// Default filter: layer 1, collides with everything
    pub const DEFAULT: Self = Self {
        layer: 1,
        mask: u32::MAX,
        group: 0,
    };

    /// Filter that collides with nothing
    pub const NONE: Self = Self {
        layer: 0,
        mask: 0,
        group: 0,
    };

    /// Filter that collides with everything
    pub const ALL: Self = Self {
        layer: u32::MAX,
        mask: u32::MAX,
        group: 0,
    };

    /// Create a new collision filter
    #[inline]
    pub const fn new(layer: u32, mask: u32) -> Self {
        Self { layer, mask, group: 0 }
    }

    /// Create filter with a collision group
    #[inline]
    pub const fn with_group(mut self, group: u32) -> Self {
        self.group = group;
        self
    }

    /// Check if two filters allow collision
    #[inline]
    pub fn can_collide(a: &Self, b: &Self) -> bool {
        // Same non-zero group => never collide
        if a.group != 0 && a.group == b.group {
            return false;
        }
        // Bidirectional layer/mask check
        (a.layer & b.mask) != 0 && (b.layer & a.mask) != 0
    }
}

impl Default for CollisionFilter {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Predefined collision layers for common game setups
pub mod layers {
    /// Default collision layer
    pub const DEFAULT: u32     = 1 << 0;
    /// Static geometry layer
    pub const STATIC: u32      = 1 << 1;
    /// Kinematic body layer
    pub const KINEMATIC: u32   = 1 << 2;
    /// Player layer
    pub const PLAYER: u32      = 1 << 3;
    /// Enemy layer
    pub const ENEMY: u32       = 1 << 4;
    /// Projectile layer
    pub const PROJECTILE: u32  = 1 << 5;
    /// Trigger volume layer
    pub const TRIGGER: u32     = 1 << 6;
    /// Debris layer
    pub const DEBRIS: u32      = 1 << 7;
    /// Sensor layer
    pub const SENSOR: u32      = 1 << 8;
    /// Vehicle layer
    pub const VEHICLE: u32     = 1 << 9;
    /// All layers combined
    pub const ALL: u32         = u32::MAX;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_filter() {
        let a = CollisionFilter::DEFAULT;
        let b = CollisionFilter::DEFAULT;
        assert!(CollisionFilter::can_collide(&a, &b));
    }

    #[test]
    fn test_none_filter() {
        let a = CollisionFilter::NONE;
        let b = CollisionFilter::DEFAULT;
        assert!(!CollisionFilter::can_collide(&a, &b));
    }

    #[test]
    fn test_layer_mask() {
        let player = CollisionFilter::new(layers::PLAYER, layers::ENEMY | layers::PROJECTILE | layers::STATIC);
        let enemy = CollisionFilter::new(layers::ENEMY, layers::PLAYER | layers::PROJECTILE | layers::STATIC);
        let wall = CollisionFilter::new(layers::STATIC, layers::ALL);

        assert!(CollisionFilter::can_collide(&player, &enemy));
        assert!(CollisionFilter::can_collide(&player, &wall));
        assert!(CollisionFilter::can_collide(&enemy, &wall));
    }

    #[test]
    fn test_one_way_mask() {
        // A can see B, but B cannot see A
        let a = CollisionFilter::new(1 << 0, 1 << 1);
        let b = CollisionFilter::new(1 << 1, 0); // mask=0: B doesn't want to collide with anything
        assert!(!CollisionFilter::can_collide(&a, &b));
    }

    #[test]
    fn test_collision_group() {
        let a = CollisionFilter::new(layers::ALL, layers::ALL).with_group(1);
        let b = CollisionFilter::new(layers::ALL, layers::ALL).with_group(1);
        let c = CollisionFilter::new(layers::ALL, layers::ALL).with_group(2);

        // Same group => no collision
        assert!(!CollisionFilter::can_collide(&a, &b));
        // Different group => collision allowed
        assert!(CollisionFilter::can_collide(&a, &c));
    }

    #[test]
    fn test_group_zero_always_checks_mask() {
        let a = CollisionFilter::new(layers::ALL, layers::ALL).with_group(0);
        let b = CollisionFilter::new(layers::ALL, layers::ALL).with_group(0);
        assert!(CollisionFilter::can_collide(&a, &b));
    }
}
