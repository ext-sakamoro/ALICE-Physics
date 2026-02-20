//! Material System with Pair-wise Combine Rules
//!
//! Provides friction and restitution lookup per body-pair, with configurable
//! combine rules (Average, Min, Max, Multiply).
//!
//! # AAA Features
//!
//! - **Material IDs**: Assign material types to rigid bodies
//! - **Combine Rules**: Average, Min, Max, Multiply for friction/restitution
//! - **Pair Overrides**: Custom friction/restitution for specific material pairs
//! - **Default Materials**: Predefined materials (Metal, Wood, Rubber, Ice, etc.)

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Material ID (u16 for compact storage)
pub type MaterialId = u16;

/// Default material ID
pub const DEFAULT_MATERIAL: MaterialId = 0;

/// Combine rule for friction/restitution when two materials interact
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineRule {
    /// Average of two values
    Average,
    /// Minimum of two values
    Min,
    /// Maximum of two values
    Max,
    /// Multiply two values
    Multiply,
}

impl CombineRule {
    /// Apply the combine rule to two values
    #[inline]
    pub fn apply(&self, a: Fix128, b: Fix128) -> Fix128 {
        match self {
            CombineRule::Average => (a + b).half(),
            CombineRule::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            CombineRule::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            CombineRule::Multiply => a * b,
        }
    }
}

impl Default for CombineRule {
    fn default() -> Self {
        CombineRule::Average
    }
}

/// Physics material definition
#[derive(Clone, Copy, Debug)]
pub struct PhysicsMaterial {
    /// Material identifier
    pub id: MaterialId,
    /// Static friction coefficient
    pub static_friction: Fix128,
    /// Dynamic friction coefficient
    pub dynamic_friction: Fix128,
    /// Restitution (bounciness)
    pub restitution: Fix128,
    /// Friction combine rule
    pub friction_combine: CombineRule,
    /// Restitution combine rule
    pub restitution_combine: CombineRule,
}

impl PhysicsMaterial {
    /// Create a new material with given properties
    pub fn new(id: MaterialId, friction: Fix128, restitution: Fix128) -> Self {
        Self {
            id,
            static_friction: friction,
            dynamic_friction: friction,
            restitution,
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
        }
    }

    /// Set combine rules
    pub fn with_combine_rules(mut self, friction: CombineRule, restitution: CombineRule) -> Self {
        self.friction_combine = friction;
        self.restitution_combine = restitution;
        self
    }

    /// Set separate static/dynamic friction
    pub fn with_static_friction(mut self, static_friction: Fix128) -> Self {
        self.static_friction = static_friction;
        self
    }
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self::new(
            DEFAULT_MATERIAL,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(3, 10),
        )
    }
}

/// Pair override entry
#[derive(Clone, Copy, Debug)]
struct PairOverride {
    mat_a: MaterialId,
    mat_b: MaterialId,
    friction: Fix128,
    restitution: Fix128,
}

/// Combined material result for a contact pair
#[derive(Clone, Copy, Debug)]
pub struct CombinedMaterial {
    /// Combined friction
    pub friction: Fix128,
    /// Combined restitution
    pub restitution: Fix128,
}

/// Material pair lookup table
pub struct MaterialTable {
    /// Registered materials (indexed by MaterialId)
    materials: Vec<PhysicsMaterial>,
    /// Pair-specific overrides
    pair_overrides: Vec<PairOverride>,
    /// Global friction combine rule (fallback)
    pub default_friction_combine: CombineRule,
    /// Global restitution combine rule (fallback)
    pub default_restitution_combine: CombineRule,
}

impl MaterialTable {
    /// Create a new material table with a default material
    pub fn new() -> Self {
        let mut table = Self {
            materials: Vec::new(),
            pair_overrides: Vec::new(),
            default_friction_combine: CombineRule::Average,
            default_restitution_combine: CombineRule::Average,
        };
        // Register default material at index 0
        table.register(PhysicsMaterial::default());
        table
    }

    /// Register a material, returns its ID
    pub fn register(&mut self, material: PhysicsMaterial) -> MaterialId {
        let id = self.materials.len() as MaterialId;
        let mut mat = material;
        mat.id = id;
        self.materials.push(mat);
        id
    }

    /// Get material by ID
    pub fn get(&self, id: MaterialId) -> &PhysicsMaterial {
        self.materials
            .get(id as usize)
            .unwrap_or(&self.materials[0])
    }

    /// Set a pair-specific override
    pub fn set_pair_override(
        &mut self,
        mat_a: MaterialId,
        mat_b: MaterialId,
        friction: Fix128,
        restitution: Fix128,
    ) {
        let (a, b) = if mat_a <= mat_b {
            (mat_a, mat_b)
        } else {
            (mat_b, mat_a)
        };

        // Update existing or add new
        if let Some(p) = self
            .pair_overrides
            .iter_mut()
            .find(|p| p.mat_a == a && p.mat_b == b)
        {
            p.friction = friction;
            p.restitution = restitution;
        } else {
            self.pair_overrides.push(PairOverride {
                mat_a: a,
                mat_b: b,
                friction,
                restitution,
            });
        }
    }

    /// Combine materials for a contact pair
    pub fn combine(&self, mat_a: MaterialId, mat_b: MaterialId) -> CombinedMaterial {
        let (a, b) = if mat_a <= mat_b {
            (mat_a, mat_b)
        } else {
            (mat_b, mat_a)
        };

        // Check pair overrides first
        if let Some(p) = self
            .pair_overrides
            .iter()
            .find(|p| p.mat_a == a && p.mat_b == b)
        {
            return CombinedMaterial {
                friction: p.friction,
                restitution: p.restitution,
            };
        }

        // Use combine rules
        let mat_a = self.get(a);
        let mat_b = self.get(b);

        // Use the higher-priority combine rule
        let friction_rule = combine_rule_priority(mat_a.friction_combine, mat_b.friction_combine);
        let restitution_rule =
            combine_rule_priority(mat_a.restitution_combine, mat_b.restitution_combine);

        CombinedMaterial {
            friction: friction_rule.apply(mat_a.dynamic_friction, mat_b.dynamic_friction),
            restitution: restitution_rule.apply(mat_a.restitution, mat_b.restitution),
        }
    }

    /// Number of registered materials
    #[inline]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    // ===== Predefined materials =====

    /// Register a "Metal" material
    pub fn register_metal(&mut self) -> MaterialId {
        self.register(PhysicsMaterial::new(
            0,
            Fix128::from_ratio(4, 10),
            Fix128::from_ratio(1, 10),
        ))
    }

    /// Register a "Wood" material
    pub fn register_wood(&mut self) -> MaterialId {
        self.register(PhysicsMaterial::new(
            0,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(3, 10),
        ))
    }

    /// Register a "Rubber" material
    pub fn register_rubber(&mut self) -> MaterialId {
        self.register(
            PhysicsMaterial::new(0, Fix128::from_ratio(8, 10), Fix128::from_ratio(8, 10))
                .with_combine_rules(CombineRule::Max, CombineRule::Max),
        )
    }

    /// Register an "Ice" material
    pub fn register_ice(&mut self) -> MaterialId {
        self.register(
            PhysicsMaterial::new(0, Fix128::from_ratio(5, 100), Fix128::from_ratio(1, 10))
                .with_combine_rules(CombineRule::Min, CombineRule::Min),
        )
    }

    /// Register a "Concrete" material
    pub fn register_concrete(&mut self) -> MaterialId {
        self.register(PhysicsMaterial::new(
            0,
            Fix128::from_ratio(6, 10),
            Fix128::from_ratio(2, 10),
        ))
    }
}

impl Default for MaterialTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Priority: Max > Multiply > Average > Min
fn combine_rule_priority(a: CombineRule, b: CombineRule) -> CombineRule {
    fn priority(r: CombineRule) -> u8 {
        match r {
            CombineRule::Min => 0,
            CombineRule::Average => 1,
            CombineRule::Multiply => 2,
            CombineRule::Max => 3,
        }
    }

    if priority(a) >= priority(b) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_rules() {
        let a = Fix128::from_ratio(4, 10);
        let b = Fix128::from_ratio(8, 10);

        let avg = CombineRule::Average.apply(a, b);
        assert_eq!(avg.hi, 0);
        // Average of 0.4 and 0.8 = 0.6

        let min = CombineRule::Min.apply(a, b);
        assert!(min <= a);

        let max = CombineRule::Max.apply(a, b);
        assert!(max >= b);
    }

    #[test]
    fn test_material_table() {
        let mut table = MaterialTable::new();
        let metal = table.register_metal();
        let rubber = table.register_rubber();

        let combined = table.combine(metal, rubber);
        // Rubber uses Max combine → should pick highest friction
        assert!(combined.friction > Fix128::ZERO);
    }

    #[test]
    fn test_pair_override() {
        let mut table = MaterialTable::new();
        let a = table.register(PhysicsMaterial::new(
            0,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(5, 10),
        ));
        let b = table.register(PhysicsMaterial::new(
            0,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(5, 10),
        ));

        // Override: ice-on-ice → near-zero friction
        table.set_pair_override(a, b, Fix128::from_ratio(1, 100), Fix128::ZERO);

        let combined = table.combine(a, b);
        assert!(combined.friction < Fix128::from_ratio(1, 10));
    }

    #[test]
    fn test_default_material() {
        let table = MaterialTable::new();
        let mat = table.get(DEFAULT_MATERIAL);
        assert!(mat.dynamic_friction > Fix128::ZERO);
        assert!(mat.restitution > Fix128::ZERO);
    }

    #[test]
    fn test_combine_rule_priority() {
        let result = combine_rule_priority(CombineRule::Min, CombineRule::Max);
        assert_eq!(result, CombineRule::Max);

        let result = combine_rule_priority(CombineRule::Average, CombineRule::Multiply);
        assert_eq!(result, CombineRule::Multiply);
    }
}
