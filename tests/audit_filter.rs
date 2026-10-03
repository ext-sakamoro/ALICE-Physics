//! Audit oracles for `filter`.
//!
//! Expected values are derived from the documented predicate
//! "`(a.layer & b.mask) != 0 && (b.layer & a.mask) != 0`, and the same
//! non-zero group never collides" by per-bit enumeration (a different
//! evaluation order from the bit-AND in the crate).

use alice_physics::{layers, CollisionFilter};

/// Per-bit reference: exists bit i with a.layer[i] and b.mask[i].
fn any_common_bit(x: u32, y: u32) -> bool {
    (0..32).any(|i| (x >> i) & 1 == 1 && (y >> i) & 1 == 1)
}

fn reference(a: &CollisionFilter, b: &CollisionFilter) -> bool {
    if a.group != 0 && a.group == b.group {
        return false;
    }
    any_common_bit(a.layer, b.mask) && any_common_bit(b.layer, a.mask)
}

#[test]
fn can_collide_matches_per_bit_reference_exhaustively_on_four_bits() {
    let mut checked = 0u32;
    for al in 0..16u32 {
        for am in 0..16u32 {
            for bl in 0..16u32 {
                for bm in 0..16u32 {
                    for ag in 0..3u32 {
                        for bg in 0..3u32 {
                            let a = CollisionFilter::new(al, am).with_group(ag);
                            let b = CollisionFilter::new(bl, bm).with_group(bg);
                            assert_eq!(
                                CollisionFilter::can_collide(&a, &b),
                                reference(&a, &b),
                                "a=({al:#x},{am:#x},g{ag}) b=({bl:#x},{bm:#x},g{bg})"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 16 * 16 * 16 * 16 * 9);
}

#[test]
fn can_collide_is_symmetric() {
    for al in 0..8u32 {
        for am in 0..8u32 {
            for bl in 0..8u32 {
                for bm in 0..8u32 {
                    for g in 0..3u32 {
                        let a = CollisionFilter::new(al, am).with_group(g);
                        let b = CollisionFilter::new(bl, bm).with_group(2 - g);
                        assert_eq!(
                            CollisionFilter::can_collide(&a, &b),
                            CollisionFilter::can_collide(&b, &a)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn high_bits_participate_in_the_check() {
    // Bit 31 only: a mask that is a high bit must still match.
    let a = CollisionFilter::new(1 << 31, 1 << 30);
    let b = CollisionFilter::new(1 << 30, 1 << 31);
    assert!(CollisionFilter::can_collide(&a, &b));
    let c = CollisionFilter::new(1 << 30, 1 << 30);
    assert!(!CollisionFilter::can_collide(&a, &c));
}

#[test]
fn documented_constants_have_documented_values() {
    assert_eq!(CollisionFilter::DEFAULT.layer, 1);
    assert_eq!(CollisionFilter::DEFAULT.mask, u32::MAX);
    assert_eq!(CollisionFilter::DEFAULT.group, 0);
    assert_eq!(CollisionFilter::NONE.layer, 0);
    assert_eq!(CollisionFilter::NONE.mask, 0);
    assert_eq!(CollisionFilter::NONE.group, 0);
    assert_eq!(CollisionFilter::ALL.layer, u32::MAX);
    assert_eq!(CollisionFilter::ALL.mask, u32::MAX);
    assert_eq!(CollisionFilter::ALL.group, 0);
    assert_eq!(CollisionFilter::default(), CollisionFilter::DEFAULT);
    assert_eq!(layers::ALL, u32::MAX);
}

#[test]
fn new_stores_layer_and_mask_in_that_order_with_group_zero() {
    let f = CollisionFilter::new(0xA5, 0x5A);
    assert_eq!((f.layer, f.mask, f.group), (0xA5, 0x5A, 0));
    let g = f.with_group(7);
    assert_eq!((g.layer, g.mask, g.group), (0xA5, 0x5A, 7));
}

#[test]
fn default_filter_collides_with_every_named_layer() {
    let named = [
        layers::DEFAULT,
        layers::STATIC,
        layers::KINEMATIC,
        layers::PLAYER,
        layers::ENEMY,
        layers::PROJECTILE,
        layers::TRIGGER,
        layers::DEBRIS,
        layers::SENSOR,
        layers::VEHICLE,
    ];
    for l in named {
        let other = CollisionFilter::new(l, u32::MAX);
        assert!(CollisionFilter::can_collide(
            &CollisionFilter::DEFAULT,
            &other
        ));
    }
}

#[test]
fn groups_differing_where_one_is_zero_do_not_block() {
    let a = CollisionFilter::ALL.with_group(0);
    let b = CollisionFilter::ALL.with_group(5);
    assert!(CollisionFilter::can_collide(&a, &b));
    assert!(CollisionFilter::can_collide(&b, &a));
}

#[test]
fn group_uses_all_32_bits() {
    let a = CollisionFilter::ALL.with_group(0x8000_0000);
    let b = CollisionFilter::ALL.with_group(0x8000_0000);
    assert_eq!(a.group, 0x8000_0000);
    assert!(!CollisionFilter::can_collide(&a, &b));
    let c = CollisionFilter::ALL.with_group(0x0001_0000);
    let d = CollisionFilter::ALL.with_group(0x0001_0000);
    assert_eq!(c.group, 0x0001_0000);
    assert!(!CollisionFilter::can_collide(&c, &d));
}
