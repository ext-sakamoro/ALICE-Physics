//! Audit oracle for `material`: combine rules, pair table, preset constants,
//! ID assignment. Expected values come from hand arithmetic / integer rational checks.
#![cfg(feature = "std")]
#![allow(clippy::type_complexity)]

use alice_physics::material::{CombineRule, MaterialTable, PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics::math::Fix128;

/// `f` is the Q64.64 floor of n/d (n,d > 0): checked with integer arithmetic only.
fn is_floor_ratio(f: Fix128, n: u128, d: u128) -> bool {
    let hi = (n / d) as i64;
    let rem = n % d;
    let lo = ((rem << 64) / d) as u64;
    f.hi == hi && f.lo == lo
}

#[test]
fn combine_rules_on_dyadic_values_are_exact() {
    let a = Fix128::from_raw(0, 1u64 << 62); // 0.25
    let b = Fix128::from_raw(0, 3u64 << 62); // 0.75
    assert_eq!(
        CombineRule::Average.apply(a, b),
        Fix128::from_raw(0, 1u64 << 63)
    ); // 0.5
    assert_eq!(CombineRule::Min.apply(a, b), a);
    assert_eq!(CombineRule::Min.apply(b, a), a);
    assert_eq!(CombineRule::Max.apply(a, b), b);
    assert_eq!(CombineRule::Max.apply(b, a), b);
    // 0.25 * 0.75 = 3/16
    assert_eq!(
        CombineRule::Multiply.apply(a, b),
        Fix128::from_raw(0, 3u64 << 60)
    );
    // negative values: min/max follow the signed order
    let n = Fix128::from_int(-2);
    assert_eq!(CombineRule::Min.apply(n, a), n);
    assert_eq!(CombineRule::Max.apply(n, a), a);
}

#[test]
fn combine_rules_are_commutative_and_idempotent_where_expected() {
    let vals = [
        Fix128::ZERO,
        Fix128::from_raw(0, 1 << 60),
        Fix128::ONE,
        Fix128::from_int(3),
    ];
    for r in [
        CombineRule::Average,
        CombineRule::Min,
        CombineRule::Max,
        CombineRule::Multiply,
    ] {
        for &x in &vals {
            for &y in &vals {
                assert_eq!(r.apply(x, y), r.apply(y, x), "{r:?} not commutative");
            }
        }
    }
    for r in [CombineRule::Average, CombineRule::Min, CombineRule::Max] {
        for &x in &vals {
            assert_eq!(r.apply(x, x), x, "{r:?} not idempotent");
        }
    }
}

#[test]
fn register_returns_sequential_ids_and_ignores_the_material_own_id() {
    let mut t = MaterialTable::new();
    assert_eq!(t.len(), 1);
    assert!(!t.is_empty());
    let a = t.register(PhysicsMaterial::new(77, Fix128::ONE, Fix128::ZERO));
    let b = t.register(PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ONE));
    assert_eq!((a, b), (1, 2));
    assert_eq!(t.get(a).id, 1);
    assert_eq!(t.get(b).id, 2);
    assert_eq!(t.get(a).dynamic_friction, Fix128::ONE);
    assert_eq!(t.get(b).restitution, Fix128::ONE);
    assert_eq!(t.len(), 3);
}

/// Number of distinct `MaterialId = u16` values: the closed form the capacity
/// boundary is derived from (AUD-A-S1W6-002).
const MATERIAL_ID_COUNT: usize = u16::MAX as usize + 1;

/// Friction tag that identifies the material registered with id `i`.
fn tag(i: usize) -> Fix128 {
    Fix128::from_int(i as i64)
}

/// Fill a fresh table up to the last free id. `new()` holds the default material at
/// id 0, so `MATERIAL_ID_COUNT - 1` further registrations fit, and registration `k`
/// (1-based) must get id `k` because ids are handed out in order and never reused.
fn full_table() -> MaterialTable {
    let mut t = MaterialTable::new();
    for i in 1..MATERIAL_ID_COUNT {
        let id = t
            .try_register(PhysicsMaterial::new(0, tag(i), Fix128::ZERO))
            .unwrap_or_else(|e| {
                panic!("registration {i} of {} failed: {e}", MATERIAL_ID_COUNT - 1)
            });
        assert_eq!(usize::from(id), i, "registration {i} got id {id}");
    }
    assert_eq!(t.len(), MATERIAL_ID_COUNT);
    t
}

/// AUD-A-S1W6-002: with `u16` ids the table holds at most 65,536 materials (the
/// default at id 0 plus 65,535 registered). The first registration past that is
/// refused instead of handing out a wrapped id, and every earlier id still finds
/// the material registered under it.
#[test]
fn material_table_try_register_refuses_past_the_u16_id_range() {
    let mut t = full_table();
    let overflow = t.try_register(PhysicsMaterial::new(0, Fix128::from_int(-1), Fix128::ZERO));
    assert_eq!(
        overflow,
        Err(alice_physics::PhysicsError::CapacityExceeded {
            resource: "materials",
            limit: MATERIAL_ID_COUNT,
        })
    );
    assert_eq!(
        t.len(),
        MATERIAL_ID_COUNT,
        "a refused registration must not grow the table"
    );
    // The default material at id 0 is untouched by the refused registration.
    assert_eq!(*t.get(DEFAULT_MATERIAL), PhysicsMaterial::default());
    for i in 1..MATERIAL_ID_COUNT {
        let id = i as u16;
        let m = t.get(id);
        assert_eq!(
            m.id, id,
            "id {id} reads back a material carrying id {}",
            m.id
        );
        assert_eq!(
            m.dynamic_friction,
            tag(i),
            "id {id} does not find its material"
        );
    }
}

/// AUD-A-S1W6-002: `register` panics on the registration that has no free id
/// instead of silently returning a wrapped id that aliases another material.
#[test]
#[should_panic(expected = "MaterialTable::register")]
fn register_panics_when_every_material_id_is_in_use() {
    let mut t = full_table();
    let _ = t.register(PhysicsMaterial::new(0, Fix128::from_int(-1), Fix128::ZERO));
}

#[test]
fn pair_override_is_symmetric_update_in_place_and_wins_over_rules() {
    let mut t = MaterialTable::new();
    let a = t.register_rubber();
    let b = t.register_ice();
    let f1 = Fix128::from_raw(0, 1 << 62);
    let r1 = Fix128::from_raw(0, 1 << 61);
    t.set_pair_override(b, a, f1, r1); // reversed order
    let c1 = t.combine(a, b);
    assert_eq!((c1.friction, c1.restitution), (f1, r1));
    assert_eq!(t.combine(b, a), c1);
    // update, not duplicate
    let f2 = Fix128::from_raw(0, 3 << 62);
    t.set_pair_override(a, b, f2, r1);
    assert_eq!(t.combine(a, b).friction, f2);
    assert_eq!(t.combine(b, a).friction, f2);
    // other pairs unaffected
    let w = t.register_wood();
    assert_ne!(t.combine(a, w).friction, f2);
    // self pair with override
    t.set_pair_override(w, w, f1, r1);
    assert_eq!(t.combine(w, w).friction, f1);
}

#[test]
fn override_on_unregistered_ids_is_still_matched_by_key() {
    let mut t = MaterialTable::new();
    let f = Fix128::from_raw(0, 1 << 61);
    t.set_pair_override(900, 40, f, f);
    assert_eq!(t.combine(40, 900).friction, f);
}

#[test]
fn combine_without_override_uses_dynamic_friction_and_priority_max_multiply_average_min() {
    let q = |n: u64| Fix128::from_raw(0, n << 60); // n/16
    let mk = |fr, re, fc, rc| PhysicsMaterial::new(0, fr, re).with_combine_rules(fc, rc);
    let mut t = MaterialTable::new();
    let lo = q(2);
    let hi = q(6);
    // (rule_a, rule_b, expected winner rule) per documented priority
    use CombineRule::*;
    let rules = [Min, Average, Multiply, Max];
    for (i, &ra) in rules.iter().enumerate() {
        for (j, &rb) in rules.iter().enumerate() {
            let a = t.register(mk(lo, lo, ra, ra));
            let b = t.register(mk(hi, hi, rb, rb));
            let winner = rules[i.max(j)];
            let c = t.combine(a, b);
            assert_eq!(
                c.friction,
                winner.apply(lo, hi),
                "friction {ra:?} vs {rb:?}"
            );
            assert_eq!(
                c.restitution,
                winner.apply(lo, hi),
                "restitution {ra:?} vs {rb:?}"
            );
        }
    }
}

#[test]
fn friction_and_restitution_choose_their_rules_independently() {
    let mut t = MaterialTable::new();
    let a = t.register(
        PhysicsMaterial::new(0, Fix128::from_int(2), Fix128::from_int(2))
            .with_combine_rules(CombineRule::Max, CombineRule::Min),
    );
    let b = t.register(
        PhysicsMaterial::new(0, Fix128::from_int(4), Fix128::from_int(4))
            .with_combine_rules(CombineRule::Min, CombineRule::Min),
    );
    let c = t.combine(a, b);
    assert_eq!(c.friction, Fix128::from_int(4)); // Max wins priority -> max(2,4)
    assert_eq!(c.restitution, Fix128::from_int(2)); // Min vs Min -> min
}

#[test]
fn static_friction_does_not_enter_combine() {
    // `with_static_friction` sets only the static coefficient; combine() reads dynamic.
    let mut t = MaterialTable::new();
    let a = t.register(
        PhysicsMaterial::new(0, Fix128::from_int(1), Fix128::ZERO)
            .with_static_friction(Fix128::from_int(9)),
    );
    let c = t.combine(a, a);
    assert_eq!(c.friction, Fix128::from_int(1));
    assert_eq!(t.get(a).static_friction, Fix128::from_int(9));
    assert_eq!(t.get(a).dynamic_friction, Fix128::from_int(1));
}

#[test]
fn default_material_and_table_values() {
    let d = PhysicsMaterial::default();
    assert_eq!(d.id, DEFAULT_MATERIAL);
    assert!(is_floor_ratio(d.static_friction, 5, 10));
    assert!(is_floor_ratio(d.dynamic_friction, 5, 10));
    assert!(is_floor_ratio(d.restitution, 3, 10));
    assert_eq!(d.friction_combine, CombineRule::Average);
    assert_eq!(d.restitution_combine, CombineRule::Average);
    let t = MaterialTable::default();
    assert_eq!(t.len(), 1);
    assert_eq!(t.get(0), &PhysicsMaterial::default());
    assert_eq!(t.default_friction_combine, CombineRule::Average);
    assert_eq!(t.default_restitution_combine, CombineRule::Average);
}

#[test]
fn preset_constants_are_the_floor_of_their_decimal_values_and_rules() {
    let mut t = MaterialTable::new();
    let cases: [(u16, (u128, u128), (u128, u128), CombineRule); 5] = [
        (t.register_metal(), (4, 10), (1, 10), CombineRule::Average),
        (t.register_wood(), (5, 10), (3, 10), CombineRule::Average),
        (t.register_rubber(), (8, 10), (8, 10), CombineRule::Max),
        (t.register_ice(), (5, 100), (1, 10), CombineRule::Min),
        (
            t.register_concrete(),
            (6, 10),
            (2, 10),
            CombineRule::Average,
        ),
    ];
    for (id, (fn_, fd), (rn, rd), rule) in cases {
        let m = t.get(id);
        assert_eq!(m.id, id);
        assert!(
            is_floor_ratio(m.dynamic_friction, fn_, fd),
            "id {id} friction"
        );
        assert_eq!(m.static_friction, m.dynamic_friction, "id {id}");
        assert!(is_floor_ratio(m.restitution, rn, rd), "id {id} restitution");
        assert_eq!(m.friction_combine, rule, "id {id}");
        assert_eq!(m.restitution_combine, rule, "id {id}");
    }
}

/// Real-world sanity: a Min-ruled material (ice) combined with an Average partner uses the
/// partner's rule (priority Average > Min), so ice x concrete is the plain average 0.325,
/// far from ice's 0.05.
#[test]
fn ice_against_average_partner_is_not_slippery_by_documented_priority() {
    let mut t = MaterialTable::new();
    let ice = t.register_ice();
    let con = t.register_concrete();
    let f = t.combine(ice, con).friction.to_f64();
    assert!((f - 0.325).abs() < 1e-12);
}

#[test]
fn unknown_id_reads_slot_zero_in_combine() {
    let mut t = MaterialTable::new();
    let m = t.register_metal();
    assert_eq!(t.combine(m, 5000), t.combine(m, DEFAULT_MATERIAL));
    assert_eq!(t.get(5000), t.get(0));
}
