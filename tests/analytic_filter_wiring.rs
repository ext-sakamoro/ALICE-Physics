//! Closed-form oracles for `filter::layers` (the nine named collision
//! category constants) and `CollisionFilter::with_group`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * **Pairwise distinctness**: `src/filter.rs::layers` documents each
//!   constant as `1 << n` for a distinct `n` in `0..=9` (`DEFAULT = 1<<0`,
//!   `STATIC = 1<<1`, `KINEMATIC = 1<<2`, `PLAYER = 1<<3`, `ENEMY = 1<<4`,
//!   `PROJECTILE = 1<<5`, `TRIGGER = 1<<6`, `DEBRIS = 1<<7`, `SENSOR = 1<<8`,
//!   `VEHICLE = 1<<9`). Nine hand-written `1u32 << n` literals must equal
//!   the nine constants and must be pairwise distinct single-bit values
//!   (`popcount == 1`).
//! * **`can_collide`** (`src/filter.rs` doc): `(a.layer & b.mask) != 0 &&
//!   (b.layer & a.mask) != 0`, gated first by "same non-zero group never
//!   collides". For a `PLAYER` filter `(layer = PLAYER, mask = ENEMY |
//!   STATIC | TRIGGER)` against a category filter `(layer = category, mask
//!   = u32::MAX)`: `(PLAYER & MAX) != 0` is always true, so the outcome is
//!   exactly `category & (ENEMY | STATIC | TRIGGER) != 0` — true for
//!   `ENEMY`, `STATIC`, `TRIGGER` and false for the other six
//!   (`DEBRIS`, `KINEMATIC`, `PROJECTILE`, `SENSOR`, `VEHICLE`, `DEFAULT`).
//! * **`with_group`** (`self.group = group`): overwriting, not additive —
//!   `f.with_group(1).with_group(2).group == 2`, never `1` or `1 | 2 == 3`.
//!   A non-zero group short-circuits the layer/mask check in both
//!   directions: two filters with matching `layer`/`mask` bits but the same
//!   non-zero `group` do not collide; the same two filters with *different*
//!   non-zero groups do.
//! * **Integration**: a `PhysicsWorld` scene with one `PLAYER` body
//!   (`mask = ENEMY | STATIC | TRIGGER`) at the origin and one body per
//!   other category collocated at `x = 1.5` (radius `1` each, so distance
//!   `1.5` < combined radius `2`, depth exactly `0.5`) reproduces the same
//!   collide / no-collide split end to end through `set_body_filter` +
//!   `step`. `substeps = 1` is required so `contact_constraints` (cleared
//!   and rebuilt every substep, see `substep`'s "Small Steps" comment)
//!   still holds the single detection pass when the frame ends — with the
//!   default 8 substeps the first substep's XPBD correction already
//!   separates a depth-`0.5` pair by `8 * 0.5 / 2 = 2.0`
//!   (`tests/analytic_world_api.rs`'s own closed form for that quantity),
//!   so a later substep's re-detection finds nothing even though a contact
//!   genuinely happened; `contact_events()` (deduped to "first substep that
//!   sees it", `EventCollector::report_contact`) is substep-count
//!   independent and is the primary oracle for "did this pair ever
//!   collide".
//!
//! # Degenerate inputs (documented result, not "no panic")
//!
//! * `CollisionFilter::NONE` (`layer = 0, mask = 0, group = 0`): matches
//!   nothing in either role — `can_collide(NONE, ALL) == false` and
//!   `can_collide(ALL, NONE) == false` (the module's own doc example: a
//!   "ghost" with `mask = 0` collides with nothing, not "everything").
//! * Two filters built identically from the same category constant collide
//!   (bidirectional check trivially passes on matching bits, `group == 0`
//!   so the group branch is skipped).
//! * `with_group` called twice overwrites; it is never additive.
//! * Extreme bit patterns (`layer = mask = group = u32::MAX`) do not panic
//!   (`catch_unwind`); two such filters share group `u32::MAX` (non-zero,
//!   equal) so they do **not** collide despite every layer/mask bit
//!   overlapping — the group check runs first and is absolute.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{layers, CollisionFilter};

// ---------------------------------------------------------------------
// Pairwise distinctness of the nine named category constants
// ---------------------------------------------------------------------

#[test]
fn layer_constants_match_documented_shift_amounts() {
    assert_eq!(layers::DEFAULT, 1u32 << 0);
    assert_eq!(layers::STATIC, 1u32 << 1);
    assert_eq!(layers::KINEMATIC, 1u32 << 2);
    assert_eq!(layers::PLAYER, 1u32 << 3);
    assert_eq!(layers::ENEMY, 1u32 << 4);
    assert_eq!(layers::PROJECTILE, 1u32 << 5);
    assert_eq!(layers::TRIGGER, 1u32 << 6);
    assert_eq!(layers::DEBRIS, 1u32 << 7);
    assert_eq!(layers::SENSOR, 1u32 << 8);
    assert_eq!(layers::VEHICLE, 1u32 << 9);
}

#[test]
fn layer_constants_are_pairwise_distinct_single_bits() {
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
    for &bit in &named {
        assert_eq!(bit.count_ones(), 1, "{bit:#x} is not a single bit");
    }
    for i in 0..named.len() {
        for j in (i + 1)..named.len() {
            assert_ne!(
                named[i], named[j],
                "layer constants at index {i} and {j} collide ({:#x})",
                named[i]
            );
        }
    }
}

// ---------------------------------------------------------------------
// can_collide: PLAYER mask of ENEMY | STATIC | TRIGGER against each
// category constant used as the partner's `layer`, partner `mask = MAX`.
// ---------------------------------------------------------------------

#[test]
fn can_collide_category_split_matches_bitmask_and() {
    let player = CollisionFilter::new(
        layers::PLAYER,
        layers::ENEMY | layers::STATIC | layers::TRIGGER,
    );

    let expect_collide = [
        (layers::ENEMY, true, "ENEMY"),
        (layers::STATIC, true, "STATIC"),
        (layers::TRIGGER, true, "TRIGGER"),
        (layers::DEBRIS, false, "DEBRIS"),
        (layers::KINEMATIC, false, "KINEMATIC"),
        (layers::PROJECTILE, false, "PROJECTILE"),
        (layers::SENSOR, false, "SENSOR"),
        (layers::VEHICLE, false, "VEHICLE"),
        (layers::DEFAULT, false, "DEFAULT"),
    ];

    for (category, expected, name) in expect_collide {
        let other = CollisionFilter::new(category, u32::MAX);
        assert_eq!(
            CollisionFilter::can_collide(&player, &other),
            expected,
            "player vs {name} (a, b order)"
        );
        // Bidirectional by construction: swapping sides must agree.
        assert_eq!(
            CollisionFilter::can_collide(&other, &player),
            expected,
            "player vs {name} (b, a order)"
        );
    }
}

// ---------------------------------------------------------------------
// with_group: overwrite semantics + group exclusion priority
// ---------------------------------------------------------------------

#[test]
fn with_group_overwrites_does_not_accumulate() {
    let once = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(1);
    assert_eq!(once.group, 1);

    let twice = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE)
        .with_group(1)
        .with_group(2);
    assert_eq!(
        twice.group, 2,
        "with_group must overwrite, not add (1|2 == 3)"
    );
    assert_ne!(twice.group, 1 | 2);
}

#[test]
fn with_group_same_nonzero_group_blocks_despite_matching_layer_mask() {
    let a = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(42);
    let b = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(42);
    // Layer/mask overlap fully (VEHICLE & VEHICLE != 0 both ways); the
    // matching non-zero group must still block collision.
    assert!(CollisionFilter::can_collide(
        &a,
        &CollisionFilter::new(layers::VEHICLE, layers::VEHICLE)
    ));
    assert!(!CollisionFilter::can_collide(&a, &b));
}

#[test]
fn with_group_different_nonzero_groups_do_not_block() {
    let a = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(42);
    let c = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(7);
    assert!(CollisionFilter::can_collide(&a, &c));
}

// ---------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------

#[test]
fn none_filter_matches_nothing_in_either_role() {
    let none = CollisionFilter::NONE;
    let all = CollisionFilter::ALL;
    assert!(!CollisionFilter::can_collide(&none, &all));
    assert!(!CollisionFilter::can_collide(&all, &none));
    // NONE has group == 0, so this is the mask == 0 branch, not the group
    // branch: zero groups *and* zero mask is "matches nothing", not
    // "default matches everything".
    assert_eq!(none.group, 0);
    assert_eq!(none.layer, 0);
    assert_eq!(none.mask, 0);
}

#[test]
fn identical_filters_built_from_the_same_category_collide() {
    let a = CollisionFilter::new(layers::ENEMY, layers::ENEMY);
    let b = CollisionFilter::new(layers::ENEMY, layers::ENEMY);
    assert_eq!(a, b);
    assert!(CollisionFilter::can_collide(&a, &b));
}

#[test]
fn extreme_bit_patterns_do_not_panic_and_group_wins() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let a = CollisionFilter::new(u32::MAX, u32::MAX).with_group(u32::MAX);
        let b = CollisionFilter::new(u32::MAX, u32::MAX).with_group(u32::MAX);
        CollisionFilter::can_collide(&a, &b)
    }));
    assert!(
        matches!(result, Ok(false)),
        "matching non-zero group must still win (no panic, result false)"
    );

    let result2 = catch_unwind(AssertUnwindSafe(|| {
        let a = CollisionFilter::new(u32::MAX, u32::MAX).with_group(u32::MAX);
        let c = CollisionFilter::new(u32::MAX, u32::MAX).with_group(0);
        CollisionFilter::can_collide(&a, &c)
    }));
    assert!(
        matches!(result2, Ok(true)),
        "group 0 on one side is not a match, falls through to layer/mask (no panic, result true)"
    );
}

// ---------------------------------------------------------------------
// Integration: production PhysicsWorld scene, through set_body_filter +
// step, mirroring examples/collision_filter_categories.rs
// ---------------------------------------------------------------------

/// `substeps = 1` so `contact_constraints` (cleared and rebuilt at the
/// start of every substep) still holds the single detection pass at the
/// end of the frame; see the module doc for why 8 substeps would not.
fn single_substep_weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..PhysicsConfig::default()
    }
}

struct CategoryWorld {
    world: PhysicsWorld,
    player: usize,
    enemy: usize,
    wall: usize,
    trigger: usize,
    debris: usize,
    kinematic: usize,
    projectile: usize,
    sensor: usize,
    vehicle: usize,
}

/// Mirrors `examples/collision_filter_categories.rs::category_scene`: one
/// `PLAYER` body at the origin (radius 1, mask `ENEMY | STATIC | TRIGGER`)
/// and one body per other named category collocated at `x = 1.5` (radius
/// 1, mask `u32::MAX`). Distance `1.5` < combined radius `2`: depth exactly
/// `0.5` for every pair the filter allows through.
fn build_category_world() -> CategoryWorld {
    let mut world = PhysicsWorld::new(single_substep_weightless());
    let r = Fix128::ONE;
    let origin = Vec3Fix::ZERO;
    let beside = Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO);

    let player = world.add_body_with_radius(RigidBody::new_dynamic(origin, Fix128::ONE), r);
    world.set_body_filter(
        player,
        CollisionFilter::new(
            layers::PLAYER,
            layers::ENEMY | layers::STATIC | layers::TRIGGER,
        ),
    );

    // `mask = layers::PLAYER` (not `u32::MAX`), so this pair's outcome also
    // pins the PLAYER constant's own bit: see the mirroring comment in
    // `examples/collision_filter_categories.rs::category_scene`.
    let enemy = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(enemy, CollisionFilter::new(layers::ENEMY, layers::PLAYER));

    let wall = world.add_body_with_radius(RigidBody::new_static(beside), r);
    world.set_body_filter(wall, CollisionFilter::new(layers::STATIC, u32::MAX));

    let trigger = world.add_body_with_radius(
        RigidBody {
            is_sensor: true,
            ..RigidBody::new_dynamic(beside, Fix128::ONE)
        },
        r,
    );
    world.set_body_filter(trigger, CollisionFilter::new(layers::TRIGGER, u32::MAX));

    let debris = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(debris, CollisionFilter::new(layers::DEBRIS, u32::MAX));

    let kinematic = world.add_body_with_radius(RigidBody::new_kinematic(beside), r);
    world.set_body_filter(kinematic, CollisionFilter::new(layers::KINEMATIC, u32::MAX));

    let projectile = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(
        projectile,
        CollisionFilter::new(layers::PROJECTILE, u32::MAX),
    );

    let sensor = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(sensor, CollisionFilter::new(layers::SENSOR, u32::MAX));

    let vehicle = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(vehicle, CollisionFilter::new(layers::VEHICLE, u32::MAX));

    CategoryWorld {
        world,
        player,
        enemy,
        wall,
        trigger,
        debris,
        kinematic,
        projectile,
        sensor,
        vehicle,
    }
}

#[test]
fn integration_scene_contact_events_match_bitmask_split() {
    let mut scene = build_category_world();
    scene.world.step(Fix128::from_ratio(1, 64));

    let has_contact_event = |a: usize, b: usize| {
        scene
            .world
            .contact_events()
            .iter()
            .any(|e| (e.body_a, e.body_b) == (a.min(b), a.max(b)))
    };

    // Allowed by the player's mask (ENEMY | STATIC | TRIGGER): a contact
    // event is generated for every one, regardless of sensor status
    // (`report_contact` runs unconditionally before the sensor branch).
    assert!(has_contact_event(scene.player, scene.enemy), "ENEMY");
    assert!(has_contact_event(scene.player, scene.wall), "STATIC");
    assert!(has_contact_event(scene.player, scene.trigger), "TRIGGER");

    // Not in the player's mask: no event at all.
    assert!(!has_contact_event(scene.player, scene.debris), "DEBRIS");
    assert!(
        !has_contact_event(scene.player, scene.kinematic),
        "KINEMATIC"
    );
    assert!(
        !has_contact_event(scene.player, scene.projectile),
        "PROJECTILE"
    );
    assert!(!has_contact_event(scene.player, scene.sensor), "SENSOR");
    assert!(!has_contact_event(scene.player, scene.vehicle), "VEHICLE");
}

#[test]
fn integration_scene_contact_constraints_exclude_sensor_pair() {
    let mut scene = build_category_world();
    scene.world.step(Fix128::from_ratio(1, 64));

    let has_constraint = |a: usize, b: usize| {
        scene
            .world
            .contact_constraints
            .iter()
            .any(|c| (c.body_a, c.body_b) == (a.min(b), a.max(b)))
    };

    // ENEMY and STATIC produce a physics-response contact constraint.
    assert!(has_constraint(scene.player, scene.enemy), "ENEMY");
    assert!(has_constraint(scene.player, scene.wall), "STATIC");
    // TRIGGER is a sensor: a contact event fires but never a constraint.
    assert!(
        !has_constraint(scene.player, scene.trigger),
        "TRIGGER must not produce a contact constraint"
    );
}

#[test]
fn integration_scene_trigger_event_reports_the_sensor_pair() {
    let mut scene = build_category_world();
    scene.world.step(Fix128::from_ratio(1, 64));

    // `report_trigger(info.body_a, info.body_b)` keeps the detection order
    // (the lower body index first in this scene, since the BVH pair comes
    // back `(player, trigger)` with `player == 0`), not "the sensor side
    // first" — `TriggerEvent.trigger_body` is `player` here, not `trigger`.
    let found = scene
        .world
        .trigger_events()
        .iter()
        .any(|e| (e.trigger_body, e.other_body) == (scene.player, scene.trigger));
    assert!(
        found,
        "expected a TriggerEvent for (player, trigger), got {:?}",
        scene
            .world
            .trigger_events()
            .iter()
            .map(|e| (e.trigger_body, e.other_body))
            .collect::<Vec<_>>()
    );
}

#[test]
fn integration_scene_degenerate_mask_zero_partner_never_collides() {
    // A ghost partner (`layer = DEBRIS`, `mask = 0`) added alongside the
    // player: even though DEBRIS is in nobody's story here, a `mask == 0`
    // partner cannot collide with *any* mask on the other side, including
    // `u32::MAX` on the player's filter.
    let mut world = PhysicsWorld::new(single_substep_weightless());
    let r = Fix128::ONE;
    let player = world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r);
    world.set_body_filter(player, CollisionFilter::new(u32::MAX, u32::MAX));

    let ghost = world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        r,
    );
    world.set_body_filter(ghost, CollisionFilter::new(layers::DEBRIS, 0));

    world.step(Fix128::from_ratio(1, 64));
    assert!(world.contact_events().is_empty());
    assert!(world.contact_constraints.is_empty());
}
