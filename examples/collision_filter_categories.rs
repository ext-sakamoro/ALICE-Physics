//! Category-based collision filtering with the `filter::layers` constants
//!
//! Wires the nine named layer constants (`layers::{DEBRIS, ENEMY, KINEMATIC,
//! PLAYER, PROJECTILE, SENSOR, STATIC, TRIGGER, VEHICLE}`) and the
//! `CollisionFilter::with_group` builder into a production `PhysicsWorld`
//! scene, through `PhysicsWorld::set_body_filter` / `PhysicsWorld::step`.
//!
//! `can_collide` is `(a.layer & b.mask) != 0 && (b.layer & a.mask) != 0`
//! (`src/filter.rs` doc), checked bidirectionally and gated first by the
//! collision-group rule ("same non-zero group never collides"). Every
//! outcome printed below follows from that formula and the hand-derived
//! pair geometry; `tests/analytic_filter_wiring.rs` pins the same values.
//!
//! ```bash
//! cargo run --example collision_filter_categories --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{layers, CollisionFilter};

/// Zero gravity (so the only thing that can change whether a pair reports a
/// contact is the filter) and a single substep. `step()` clears and
/// re-detects `contact_constraints` at the start of *every* substep
/// (`substep`'s "Small Steps" comment); with the default 8 substeps the
/// first substep's XPBD position correction already separates a depth-`0.5`
/// pair by `8 * 0.5 / 2 = 2.0` (`tests/analytic_world_api.rs`'s own closed
/// form), so by the last substep `detect_collisions` finds nothing and
/// `contact_constraints` ends the frame empty even though a contact
/// happened. One substep makes `contact_constraints` reflect the single
/// detection pass deterministically.
fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..PhysicsConfig::default()
    }
}

/// `1/64 s`, exact in `Fix128` (same frame used by `examples/world_api_tour.rs`).
fn frame_dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// One `PLAYER` body at the origin (radius 1) and one body of every other
/// named category collocated at `x = 1.5` (radius 1): distance `1.5` <
/// combined radius `2`, so every pair the filter allows overlaps by exactly
/// `0.5` (`tests/analytic_filter_wiring.rs::PLAYER_PARTNER_DEPTH`). The eight
/// "other" bodies are mutually coincident (distance `0`), which
/// `detect_collisions` skips outright (`dist_sq.is_zero()`), so only the
/// nine `(player, other)` pairs are observable.
fn category_scene() -> (PhysicsWorld, [usize; 9]) {
    let mut world = PhysicsWorld::new(weightless());
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

    // `mask = layers::PLAYER` (not `u32::MAX`) so this pair's outcome also
    // pins the PLAYER constant's own bit: the other eight partners use
    // `u32::MAX`, under which `(player.layer & partner.mask) != 0` holds for
    // *any* non-zero `player.layer`, making PLAYER unobservable through
    // them alone (`rules/analytic-oracle-tests.md`'s "scene must let the
    // quantity reach the output").
    let enemy = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(enemy, CollisionFilter::new(layers::ENEMY, layers::PLAYER));

    let wall = world.add_body_with_radius(RigidBody::new_static(beside), r);
    world.set_body_filter(wall, CollisionFilter::new(layers::STATIC, u32::MAX));

    // A trigger volume: tagged with the TRIGGER layer and flagged `is_sensor`
    // so the engine reports a `TriggerEvent` instead of a contact constraint.
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

    let turret = world.add_body_with_radius(RigidBody::new_kinematic(beside), r);
    world.set_body_filter(turret, CollisionFilter::new(layers::KINEMATIC, u32::MAX));

    let bullet = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(bullet, CollisionFilter::new(layers::PROJECTILE, u32::MAX));

    let radar = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(radar, CollisionFilter::new(layers::SENSOR, u32::MAX));

    let car = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(car, CollisionFilter::new(layers::VEHICLE, u32::MAX));

    (
        world,
        [
            player, enemy, wall, trigger, debris, turret, bullet, radar, car,
        ],
    )
}

fn contacts_and_triggers() {
    println!("[filter] -- category layers vs a PLAYER mask of ENEMY|STATIC|TRIGGER --");
    let (mut world, ids) = category_scene();
    let [player, enemy, wall, trigger, debris, turret, bullet, radar, car] = ids;

    world.step(frame_dt());

    let contact_pairs: Vec<(usize, usize)> = world
        .contact_events()
        .iter()
        .map(|e| (e.body_a, e.body_b))
        .collect();
    let constraint_pairs: Vec<(usize, usize)> = world
        .contact_constraints
        .iter()
        .map(|c| (c.body_a, c.body_b))
        .collect();
    let trigger_pairs: Vec<(usize, usize)> = world
        .trigger_events()
        .iter()
        .map(|e| (e.trigger_body, e.other_body))
        .collect();

    let label = |idx: usize| -> &'static str {
        match idx {
            x if x == enemy => "ENEMY",
            x if x == wall => "STATIC",
            x if x == trigger => "TRIGGER",
            x if x == debris => "DEBRIS",
            x if x == turret => "KINEMATIC",
            x if x == bullet => "PROJECTILE",
            x if x == radar => "SENSOR",
            x if x == car => "VEHICLE",
            _ => "?",
        }
    };

    for other in [enemy, wall, trigger, debris, turret, bullet, radar, car] {
        let has_contact_event = contact_pairs.contains(&(player, other));
        let has_constraint = constraint_pairs.contains(&(player, other));
        let has_trigger =
            trigger_pairs.contains(&(player, other)) || trigger_pairs.contains(&(other, player));
        println!(
            "[filter] PLAYER vs {:<10} contact_event={has_contact_event:<5} contact_constraint={has_constraint:<5} trigger_event={has_trigger}",
            label(other)
        );
    }
}

/// `with_group` is an overwrite (`self.group = group`), not additive, and a
/// same non-zero `group` on both sides blocks collision regardless of
/// matching `layer`/`mask` bits (group check runs before the bitmask check,
/// `CollisionFilter::can_collide`).
fn group_builder() {
    println!("[filter] -- with_group: overwrite semantics + group exclusion --");

    let overwritten = CollisionFilter::new(layers::VEHICLE, layers::VEHICLE)
        .with_group(1)
        .with_group(2);
    println!(
        "[filter] with_group(1).with_group(2) -> group={} (overwrite, not additive)",
        overwritten.group
    );

    let mut world = PhysicsWorld::new(weightless());
    let origin = Vec3Fix::ZERO;
    let beside = Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO);
    let behind = Vec3Fix::new(Fix128::from_ratio(-3, 2), Fix128::ZERO, Fix128::ZERO);
    let r = Fix128::ONE;

    let hero = world.add_body_with_radius(RigidBody::new_dynamic(origin, Fix128::ONE), r);
    world.set_body_filter(
        hero,
        CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(42),
    );

    // Same layer/mask bits as `hero`, same non-zero group: blocked.
    let convoy_mate = world.add_body_with_radius(RigidBody::new_dynamic(beside, Fix128::ONE), r);
    world.set_body_filter(
        convoy_mate,
        CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(42),
    );

    // Same layer/mask bits, a *different* non-zero group: allowed.
    let rival = world.add_body_with_radius(RigidBody::new_dynamic(behind, Fix128::ONE), r);
    world.set_body_filter(
        rival,
        CollisionFilter::new(layers::VEHICLE, layers::VEHICLE).with_group(7),
    );

    world.step(frame_dt());
    let constraint_pairs: Vec<(usize, usize)> = world
        .contact_constraints
        .iter()
        .map(|c| (c.body_a, c.body_b))
        .collect();
    println!(
        "[filter] hero vs convoy_mate (same group 42): contact={}",
        constraint_pairs.contains(&(hero, convoy_mate))
    );
    println!(
        "[filter] hero vs rival (group 42 vs 7): contact={}",
        constraint_pairs.contains(&(hero, rival))
    );
}

fn main() {
    contacts_and_triggers();
    group_builder();
}
