//! Simulation Modifier Core System
//!
//! Trait and wrapper for physics-driven SDF modifiers.
//! `ModifiedSdf` chains multiple modifiers that read simulation
//! fields (temperature, pressure, stress) and alter the SDF
//! distance at evaluation time.
//!
//! # Pattern
//!
//! Each modifier:
//! 1. Owns its simulation field data (ScalarField3D)
//! 2. Has `update(dt)` to advance simulation state
//! 3. `modify_distance(x, y, z, dist)` alters SDF distance
//!
//! # Example
//!
//! ```
//! use alice_physics::sdf_collider::ClosureSdf;
//! use alice_physics::sim_modifier::ModifiedSdf;
//! use alice_physics::thermal::{ThermalModifier, ThermalConfig};
//! use alice_physics::pressure::{PressureModifier, PressureConfig};
//! use alice_physics::sdf_collider::SdfField;
//!
//! let sphere_sdf = ClosureSdf::new(
//!     |x, y, z| (x*x + y*y + z*z).sqrt() - 1.0,
//!     |x, y, z| { let l = (x*x + y*y + z*z).sqrt().max(1e-6); (x/l, y/l, z/l) },
//! );
//! let bounds = (-2.0, -2.0, -2.0);
//! let bounds_max = (2.0, 2.0, 2.0);
//! let thermal = ThermalModifier::new(ThermalConfig::default(), 4, bounds, bounds_max);
//! let pressure = PressureModifier::new(PressureConfig::default(), 4, bounds, bounds_max);
//!
//! let modified = ModifiedSdf::new(Box::new(sphere_sdf))
//!     .with_modifier(Box::new(thermal))
//!     .with_modifier(Box::new(pressure));
//!
//! let dist = modified.distance(0.0, 0.0, 0.0);
//! ```
//!
//! Author: Moroya Sakamoto

use crate::sdf_collider::SdfField;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// PhysicsModifier Trait
// ============================================================================

/// Trait for simulation-driven SDF modifiers
pub trait PhysicsModifier: Send + Sync {
    /// Modify the SDF distance at a world-space point.
    ///
    /// Returns the modified distance. Positive offset = surface recedes
    /// (material removed). Negative offset = surface expands.
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32;

    /// Advance the simulation state by dt seconds.
    fn update(&mut self, dt: f32);

    /// Name of this modifier (for debugging)
    fn name(&self) -> &str;

    /// Whether this modifier has any active effect
    fn is_active(&self) -> bool {
        true
    }
}

// ============================================================================
// ModifiedSdf
// ============================================================================

/// SDF wrapper that applies a chain of physics modifiers.
///
/// Implements `SdfField`, so it can be used anywhere an SDF is expected.
pub struct ModifiedSdf {
    /// Original SDF field
    original: Box<dyn SdfField>,
    /// Chain of modifiers (applied in order)
    modifiers: Vec<Box<dyn PhysicsModifier>>,
    /// Epsilon for normal computation
    normal_eps: f32,
}

impl ModifiedSdf {
    /// Create a new modified SDF wrapping the original field
    pub fn new(original: Box<dyn SdfField>) -> Self {
        Self {
            original,
            modifiers: Vec::new(),
            normal_eps: 0.001,
        }
    }

    /// Add a modifier to the chain
    pub fn with_modifier(mut self, modifier: Box<dyn PhysicsModifier>) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Add a modifier (mutable)
    pub fn add_modifier(&mut self, modifier: Box<dyn PhysicsModifier>) {
        self.modifiers.push(modifier);
    }

    /// Remove all modifiers
    pub fn clear_modifiers(&mut self) {
        self.modifiers.clear();
    }

    /// Number of active modifiers
    pub fn modifier_count(&self) -> usize {
        self.modifiers.len()
    }

    /// Update all modifier simulations
    pub fn update(&mut self, dt: f32) {
        for m in &mut self.modifiers {
            m.update(dt);
        }
    }

    /// Get mutable access to a modifier by index
    pub fn modifier_mut(&mut self, index: usize) -> Option<&mut dyn PhysicsModifier> {
        match self.modifiers.get_mut(index) {
            Some(m) => Some(m.as_mut()),
            None => None,
        }
    }

    /// Evaluate the modified distance at a point
    #[inline]
    fn eval_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut d = self.original.distance(x, y, z);
        for m in &self.modifiers {
            if m.is_active() {
                d = m.modify_distance(x, y, z, d);
            }
        }
        d
    }
}

impl SdfField for ModifiedSdf {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval_distance(x, y, z)
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let e = self.normal_eps;
        let dx = self.eval_distance(x + e, y, z) - self.eval_distance(x - e, y, z);
        let dy = self.eval_distance(x, y + e, z) - self.eval_distance(x, y - e, z);
        let dz = self.eval_distance(x, y, z + e) - self.eval_distance(x, y, z - e);

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-10 {
            (0.0, 1.0, 0.0)
        } else {
            (dx / len, dy / len, dz / len)
        }
    }

    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let d = self.eval_distance(x, y, z);
        let n = self.normal(x, y, z);
        (d, n)
    }
}

// ============================================================================
// SingleModifiedSdf (lightweight single-modifier wrapper)
// ============================================================================

/// Lightweight SDF wrapper for a single modifier (avoids Vec overhead)
pub struct SingleModifiedSdf<M: PhysicsModifier> {
    /// Original SDF
    pub original: Box<dyn SdfField>,
    /// The modifier
    pub modifier: M,
    /// Epsilon for normals
    normal_eps: f32,
}

impl<M: PhysicsModifier> SingleModifiedSdf<M> {
    /// Create wrapper with a single modifier
    pub fn new(original: Box<dyn SdfField>, modifier: M) -> Self {
        Self {
            original,
            modifier,
            normal_eps: 0.001,
        }
    }

    /// Update the modifier simulation
    pub fn update(&mut self, dt: f32) {
        self.modifier.update(dt);
    }

    #[inline]
    fn eval_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let d = self.original.distance(x, y, z);
        self.modifier.modify_distance(x, y, z, d)
    }
}

impl<M: PhysicsModifier> SdfField for SingleModifiedSdf<M> {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval_distance(x, y, z)
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let e = self.normal_eps;
        let dx = self.eval_distance(x + e, y, z) - self.eval_distance(x - e, y, z);
        let dy = self.eval_distance(x, y + e, z) - self.eval_distance(x, y - e, z);
        let dz = self.eval_distance(x, y, z + e) - self.eval_distance(x, y, z - e);

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-10 {
            (0.0, 1.0, 0.0)
        } else {
            (dx / len, dy / len, dz / len)
        }
    }

    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let d = self.eval_distance(x, y, z);
        let n = self.normal(x, y, z);
        (d, n)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    /// Simple modifier that uniformly expands the SDF
    struct ExpandModifier {
        amount: f32,
    }

    impl PhysicsModifier for ExpandModifier {
        fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
            d - self.amount
        }
        fn update(&mut self, _dt: f32) {}
        fn name(&self) -> &str {
            "expand"
        }
    }

    #[test]
    fn test_modified_sdf_no_modifiers() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere));
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 1.0).abs() < 0.01,
            "No modifiers should pass through, got {}",
            d
        );
    }

    #[test]
    fn test_modified_sdf_expand() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere))
            .with_modifier(Box::new(ExpandModifier { amount: 0.5 }));

        // Original: distance at (2,0,0) = 1.0
        // After expand by 0.5: distance = 0.5
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 0.5).abs() < 0.01,
            "Expand should reduce distance, got {}",
            d
        );
    }

    #[test]
    fn test_modifier_chain() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere))
            .with_modifier(Box::new(ExpandModifier { amount: 0.3 }))
            .with_modifier(Box::new(ExpandModifier { amount: 0.2 }));

        // Total expand = 0.5
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 0.5).abs() < 0.01,
            "Chain should sum expansions, got {}",
            d
        );
    }

    #[test]
    fn test_single_modified_sdf() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = SingleModifiedSdf::new(Box::new(sphere), ExpandModifier { amount: 0.5 });

        let d = modified.distance(2.0, 0.0, 0.0);
        assert!((d - 0.5).abs() < 0.01, "Single modifier expand, got {}", d);
    }
}
