//! Analytic-solution / behavioral-difference oracle tests for the
//! `solver_tgs` dispatch + warm-start cache wiring: `dispatch_islands`,
//! `par_dispatch_islands`, `ImpulseCache::{stats,hit_rate,reset_stats,sweep}`,
//! and the new `PhysicsWorld::{tgs_cache_stats,reset_tgs_cache_stats}`
//! accessors (`TgsCacheStats`).
//!
//! `dispatch_islands`, `par_dispatch_islands`, `ImpulseCache` and
//! `solver_tgs::{HasVelocity, AdaptiveSubStepConfig, ...}` are all
//! `pub(crate)` — unreachable from this external `tests/` crate, which
//! only sees the library's public surface. Their degenerate cases
//! (empty island list, single island, cache with zero entries, sweep
//! with nothing stale) are therefore covered directly in
//! `src/solver_tgs.rs`'s own `#[cfg(test)]` module
//! (`dispatch_islands_degenerate_empty_and_single`,
//! `impulse_cache_sweep_degenerate_empty_and_nothing_stale`,
//! `par_dispatch_islands_panics_on_aux_length_mismatch`, and the existing
//! empty-slice case folded into
//! `par_dispatch_islands_runs_closure_for_each_island_exactly_once`).
//! This file instead proves the wiring end-to-end through the one public
//! entry point that reaches all of them: `PhysicsWorld::step` under
//! `SolverBackend::Tgs`, plus the new `tgs_cache_stats` /
//! `reset_tgs_cache_stats` accessors.
//!
//! Rule (analytic-oracle-tests): every hit/miss count below is derived by
//! hand from `ImpulseCache`'s own documented contract (`take` marks an
//! entry alive and counts a hit/miss; `set` stores without marking alive;
//! `sweep` drops anything not marked alive this tick and clears the alive
//! set) and from `Pgs6DofOrientedHooks`'s hook wiring (`cache.take` runs
//! once per contact per sub-step in `begin_substep`, `cache.set` runs
//! once per contact per sub-step in `end_substep` — see
//! `src/solver_tgs_hooks_6dof_oriented.rs`), never by calling `sweep`,
//! `stats`, or `step` itself to produce the expected side.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn tgs_world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    })
}

/// oracle: this is the behavioral difference the `sweep()` wiring exists
/// for. With exactly two bodies (ground + ball), the sole contact's
/// `stable_id` is always `0` (there is only ever one possible pair, so
/// `detect_collisions`'s single-pair output always lands at index `0`
/// when the pair is in contact).
///
/// 1. Frame 1 (first-ever contact): `cache.take(0)` on an empty cache →
///    miss. `(hits, misses) = (0, 1)`.
/// 2. Frame 2 (same overlapping position, same id `0`): the entry
///    `sweep`-survived frame 1 (it was `take`n, hence marked alive, hence
///    not evicted) → `cache.take(0)` is a hit. `(hits, misses) = (1, 1)`.
/// 3. Frame 3 (ball moved far away, zero contacts this tick): no `take`
///    call happens at all, so `live` is empty when `sweep()` runs — the
///    entry for id `0` is evicted. `(hits, misses)` unchanged `(1, 1)`.
/// 4. Frame 4 (ball moved back into contact, same id `0` again): the
///    entry was evicted in frame 3, so `cache.take(0)` is a **miss**, not
///    a hit. `(hits, misses) = (1, 2)`.
///
/// Before this wiring landed, `step_tgs` never called `sweep()`, so the
/// frame-2 entry would never have been evicted in frame 3, and frame 4
/// would have incorrectly scored a hit (`(2, 1)` instead of `(1, 2)`),
/// resurrecting a warm-start impulse from a contact that no longer has
/// any physical relationship to the new one.
#[test]
fn sweep_evicts_a_contact_that_stops_recurring() {
    let mut w = tgs_world();
    let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.set_body_collision_radius(ground, Fix128::ONE);
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, Fix128::ONE);
    let dt = r(1, 60);
    let touching = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    let far_away = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(1000), Fix128::ZERO);

    // Frame 1: first touch.
    w.step(dt);
    let s1 = w.tgs_cache_stats();
    assert_eq!(
        (s1.hits, s1.misses),
        (0, 1),
        "frame 1: first touch is a miss"
    );

    // Frame 2: re-pin to the same overlapping position (the solve may
    // have nudged it slightly) — same contact, same id 0, now a hit.
    w.bodies[ball].position = touching;
    w.bodies[ball].velocity = Vec3Fix::ZERO;
    w.step(dt);
    let s2 = w.tgs_cache_stats();
    assert_eq!(
        (s2.hits, s2.misses),
        (1, 1),
        "frame 2: warm-start hit on the recurring contact"
    );

    // Frame 3: move far away — zero contacts this tick, so `sweep` finds
    // nothing alive and evicts the frame-2 entry.
    w.bodies[ball].position = far_away;
    w.bodies[ball].velocity = Vec3Fix::ZERO;
    w.step(dt);
    let s3 = w.tgs_cache_stats();
    assert_eq!(
        (s3.hits, s3.misses),
        (1, 1),
        "frame 3: zero contacts, cache untouched (no take/set calls at all)"
    );

    // Frame 4: move back into contact — same id 0, but the entry was
    // evicted in frame 3, so this is a fresh miss, not a stale hit.
    w.bodies[ball].position = touching;
    w.bodies[ball].velocity = Vec3Fix::ZERO;
    w.step(dt);
    let s4 = w.tgs_cache_stats();
    assert_eq!(
        (s4.hits, s4.misses),
        (1, 2),
        "frame 4: the evicted entry forces a miss instead of resurrecting stale data"
    );
}

/// oracle: `TgsCacheStats::hit_rate` is `hits / (hits + misses)`, `0.0`
/// when both are zero — checked directly against hand-picked ratios
/// without going through `PhysicsWorld` at all (the formula itself does
/// not depend on how the counts were produced).
#[test]
fn tgs_cache_stats_hit_rate_matches_hand_derived_ratio() {
    use alice_physics::TgsCacheStats;

    let empty = TgsCacheStats::default();
    assert_eq!((empty.hits, empty.misses), (0, 0));
    assert_eq!(empty.hit_rate(), 0.0, "0/0 is defined as 0.0, not NaN");

    let three_of_four = TgsCacheStats::new(3, 1);
    assert!((three_of_four.hit_rate() - 0.75).abs() < 1e-12);

    let all_misses = TgsCacheStats::new(0, 5);
    assert_eq!(all_misses.hit_rate(), 0.0);

    let all_hits = TgsCacheStats::new(5, 0);
    assert_eq!(all_hits.hit_rate(), 1.0);
}

/// oracle: `reset_tgs_cache_stats` clears only the hit/miss **counters**,
/// not the cached impulses themselves — so a contact that recurs right
/// after a reset still warm-starts (a hit), it just is not counted
/// against the pre-reset totals.
///
/// Frame 1 (first touch): miss, `(hits, misses) = (0, 1)`.
/// `reset_tgs_cache_stats()`: counters go to `(0, 0)`; the entry for id
/// `0` that `sweep` just preserved (it was `take`n this tick) is
/// untouched.
/// Frame 2 (same contact recurs): the preserved entry makes this a hit,
/// so post-frame-2 stats are `(1, 0)` relative to the reset baseline —
/// if `reset_tgs_cache_stats` had also cleared the entries, this would
/// have been a miss (`(0, 1)`) instead.
#[test]
fn reset_tgs_cache_stats_clears_counters_without_touching_cached_impulses() {
    let mut w = tgs_world();
    let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.set_body_collision_radius(ground, Fix128::ONE);
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, Fix128::ONE);
    let dt = r(1, 60);

    w.step(dt); // frame 1: miss.
    let s1 = w.tgs_cache_stats();
    assert_eq!((s1.hits, s1.misses), (0, 1));

    w.reset_tgs_cache_stats();
    let after_reset = w.tgs_cache_stats();
    assert_eq!((after_reset.hits, after_reset.misses), (0, 0));

    w.bodies[ball].position = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    w.bodies[ball].velocity = Vec3Fix::ZERO;
    w.step(dt); // frame 2: the entry survived the reset, so this is a hit.
    let s2 = w.tgs_cache_stats();
    assert_eq!(
        (s2.hits, s2.misses),
        (1, 0),
        "reset clears counters only; the cached impulse from frame 1 still warm-starts frame 2"
    );
}

/// Degenerate end-to-end case: an **empty** world (zero bodies) stepped
/// under `SolverBackend::Tgs` builds zero islands and makes zero
/// `ImpulseCache` calls — `dispatch_islands`/`par_dispatch_islands`
/// (reached transitively through `solve_oriented_islands_serial`) must
/// not panic on an empty island list, and the cache stats must stay at
/// the all-zero default.
#[test]
fn tgs_step_on_an_empty_world_does_not_panic_and_reports_zero_stats() {
    let mut w = tgs_world();
    for _ in 0..5 {
        w.step(r(1, 60));
    }
    let s = w.tgs_cache_stats();
    assert_eq!((s.hits, s.misses), (0, 0));
}

/// Degenerate end-to-end case: a single free dynamic body with no
/// collision radius set (so it can never generate a contact) still gets
/// its own single-body island (an isolated body is its own connected
/// component in `build_islands`) and falls under gravity — exercising
/// the single-island dispatch path — while the `ImpulseCache` stays
/// completely untouched across many steps (zero contacts ever occur).
#[test]
fn tgs_step_with_a_single_contact_free_body_reports_zero_stats_but_still_falls() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        substeps: 1,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    });
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO),
        Fix128::ONE,
    ));
    let dt = r(1, 60);
    for _ in 0..10 {
        w.step(dt);
    }
    let s = w.tgs_cache_stats();
    assert_eq!(
        (s.hits, s.misses),
        (0, 0),
        "no collider radius set => never a contact => cache never touched"
    );
    assert!(
        w.bodies[b].position.y < Fix128::from_int(10),
        "the lone body's single-body island must still receive gravity"
    );
}
