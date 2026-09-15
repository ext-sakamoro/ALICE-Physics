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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CombineRule {
    /// Average of two values
    #[default]
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
    #[must_use]
    pub fn apply(&self, a: Fix128, b: Fix128) -> Fix128 {
        match self {
            Self::Average => (a + b).half(),
            Self::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            Self::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            Self::Multiply => a * b,
        }
    }
}

/// Physics material definition
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    #[must_use]
    pub const fn new(id: MaterialId, friction: Fix128, restitution: Fix128) -> Self {
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
    #[must_use]
    pub const fn with_combine_rules(
        mut self,
        friction: CombineRule,
        restitution: CombineRule,
    ) -> Self {
        self.friction_combine = friction;
        self.restitution_combine = restitution;
        self
    }

    /// Set separate static/dynamic friction
    #[must_use]
    pub const fn with_static_friction(mut self, static_friction: Fix128) -> Self {
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
#[derive(Clone, Copy, Debug, PartialEq)]
struct PairOverride {
    mat_a: MaterialId,
    mat_b: MaterialId,
    friction: Fix128,
    restitution: Fix128,
}

/// Combined material result for a contact pair
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CombinedMaterial {
    /// Combined friction
    pub friction: Fix128,
    /// Combined restitution
    pub restitution: Fix128,
}

/// Material pair lookup table
pub struct MaterialTable {
    /// Registered materials (indexed by `MaterialId`)
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
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
const fn combine_rule_priority(a: CombineRule, b: CombineRule) -> CombineRule {
    const fn priority(r: CombineRule) -> u8 {
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

    /// 非 2 冪の有理数同士の平均は Fix128 で 1 ulp 丸まり得る
    fn near(a: Fix128, b: Fix128) -> bool {
        (a - b).abs() < Fix128 { hi: 0, lo: 1 << 8 }
    }

    #[test]
    fn register_wood_ice_concrete_store_documented_coefficients_and_rules() {
        let mut table = MaterialTable::new();
        let wood = table.register_wood();
        let ice = table.register_ice();
        let concrete = table.register_concrete();
        // id は登録順 (default が 0)、struct 側 id も同じ値に書き換わる
        assert_eq!((wood, ice, concrete), (1, 2, 3));
        assert_eq!(table.len(), 4);
        assert_eq!(table.get(wood).id, wood);
        assert_eq!(table.get(ice).id, ice);
        assert_eq!(table.get(concrete).id, concrete);

        let w = table.get(wood);
        assert_eq!(w.dynamic_friction, Fix128::from_ratio(5, 10));
        assert_eq!(w.static_friction, Fix128::from_ratio(5, 10));
        assert_eq!(w.restitution, Fix128::from_ratio(3, 10));
        assert_eq!(w.friction_combine, CombineRule::Average);
        assert_eq!(w.restitution_combine, CombineRule::Average);

        let i = table.get(ice);
        assert_eq!(i.dynamic_friction, Fix128::from_ratio(5, 100));
        assert_eq!(i.restitution, Fix128::from_ratio(1, 10));
        assert_eq!(i.friction_combine, CombineRule::Min);
        assert_eq!(i.restitution_combine, CombineRule::Min);

        let c = table.get(concrete);
        assert_eq!(c.dynamic_friction, Fix128::from_ratio(6, 10));
        assert_eq!(c.restitution, Fix128::from_ratio(2, 10));
        assert_eq!(c.friction_combine, CombineRule::Average);
        assert_eq!(c.restitution_combine, CombineRule::Average);

        // wood × concrete: Average → friction (0.5+0.6)/2 = 0.55、restitution (0.3+0.2)/2 = 0.25
        let wc = table.combine(wood, concrete);
        assert!(near(wc.friction, Fix128::from_ratio(55, 100)), "{wc:?}");
        assert!(near(wc.restitution, Fix128::from_ratio(25, 100)), "{wc:?}");
        // ice × concrete: Min は Average より低優先 → Average 採用 → (0.05+0.6)/2 = 0.325
        let ic = table.combine(ice, concrete);
        assert!(near(ic.friction, Fix128::from_ratio(325, 1000)), "{ic:?}");
        assert!(near(ic.restitution, Fix128::from_ratio(15, 100)), "{ic:?}");
        // ice × ice: 両方 Min → 0.05 / 0.1
        let ii = table.combine(ice, ice);
        assert_eq!(ii.friction, Fix128::from_ratio(5, 100));
        assert_eq!(ii.restitution, Fix128::from_ratio(1, 10));
        // 対称
        assert_eq!(table.combine(concrete, ice), ic);
    }

    #[test]
    fn with_static_friction_only_changes_static_coefficient() {
        let base = PhysicsMaterial::new(7, Fix128::from_ratio(3, 10), Fix128::from_ratio(2, 10));
        let m = base.with_static_friction(Fix128::from_ratio(9, 10));
        assert_eq!(m.static_friction, Fix128::from_ratio(9, 10));
        assert_eq!(m.dynamic_friction, Fix128::from_ratio(3, 10));
        assert_eq!(m.restitution, Fix128::from_ratio(2, 10));
        assert_eq!(m.id, 7);
        assert_eq!(m.friction_combine, base.friction_combine);
        assert_eq!(m.restitution_combine, base.restitution_combine);
        // static_friction 以外は元と同一
        let mut expected = base;
        expected.static_friction = Fix128::from_ratio(9, 10);
        assert_eq!(m, expected);

        // combine は dynamic_friction を使う: static を変えても pair 結果は不変
        let mut table = MaterialTable::new();
        let plain = table.register(base);
        let sticky = table.register(m);
        let plain_pair = table.combine(plain, DEFAULT_MATERIAL);
        let sticky_pair = table.combine(sticky, DEFAULT_MATERIAL);
        assert_eq!(plain_pair, sticky_pair);
        // (0.3 + 0.5) / 2 = 0.4
        assert!(
            near(sticky_pair.friction, Fix128::from_ratio(4, 10)),
            "{sticky_pair:?}"
        );
    }
}
