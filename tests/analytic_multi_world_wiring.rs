//! Oracles for the wiring of `multi_world`'s surface: `MultiWorld::{new,
//! add_world, step_all, step_all_parallel, total_body_count, transfer_body,
//! world_count}` and `Portal::{new, transform_a_to_b, transform_b_to_a}`.
//! All of `add_world`, `step_all`, `step_all_parallel`, `total_body_count`,
//! `transfer_body`, `transform_a_to_b`, `transform_b_to_a`, `world_count`
//! had zero production callers before this crate's
//! `examples/multi_world_management.rs` (`scripts/wiring-baseline.txt`
//! `unwired src/multi_world.rs::*`, 8 items).
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * **`add_world`** returns `self.worlds.len()` *before* the push, so the
//!   n-th call (0-indexed) returns exactly `n`.
//! * **`world_count`** is exactly the number of `add_world` calls made so
//!   far (it does not track removals -- there is no `remove_world`).
//! * **`total_body_count`** is `sum(w.bodies.len() for w in worlds)`; the
//!   oracle sums each world's `bodies.len()` independently (not by calling
//!   `total_body_count` itself) and cross-checks.
//! * **`transfer_body(from, body_id, to, new_position)`** removes
//!   `bodies[body_id]` from `worlds[from]` via `swap_remove` semantics (the
//!   last body takes the removed slot) and pushes it onto `worlds[to]`, so
//!   the returned index is exactly `worlds[to].bodies.len()` *before* the
//!   push, `worlds[from].bodies.len()` drops by exactly 1, `worlds[to]`'s
//!   rises by exactly 1, and the moved body's `position` AND
//!   `prev_position` both become `new_position` exactly (bit-for-bit, plain
//!   field assignment -- no XPBD blending).
//! * **`step_all`** advances every world independently by the same `dt`;
//!   it does not touch `MultiWorld` bookkeeping (`worlds.len()` is
//!   unchanged by stepping).
//! * **`step_all_parallel`** (feature `parallel`) is documented to run the
//!   exact same per-world step as `step_all`, just via Rayon; the closed
//!   form is bit-identity against `step_all` on an equivalent scene, frame
//!   by frame, for every body's `position`/`velocity`.
//! * **`Portal::transform_a_to_b`/`transform_b_to_a`** apply
//!   `rotate_vec(rotation, p) [+-] translation`. With `rotation = (x=0,
//!   y=0, z=1, w=0)` -- the pure unit quaternion along +Z, i.e. a 180
//!   degree rotation about Z -- the Hamilton product `q*v*q^-1` (this
//!   crate's `QuatFix::mul`/`conjugate`) reduces, hand-derived below, to
//!   exactly `(-v.x, -v.y, v.z)` for any `v`:
//!
//!   ```text
//!   q = (0,0,1,0), q^-1 = conjugate(q) = (0,0,-1,0), qv = (vx,vy,vz,0)
//!   temp  = q * qv  = (w=0,x=0,y=0,z=1) * (w=0,x=vx,y=vy,z=vz)
//!         = (x: 0*vx+0*0+0*vz-1*vy, y: 0*vy-0*vz+0*0+1*vx,
//!            z: 0*vz+0*vy-0*vx+1*0, w: 0*0-0*vx-0*vy-1*vz)
//!         = (-vy, vx, 0, -vz)
//!   result = temp * q^-1, q^-1 = (x=0,y=0,z=-1,w=0)
//!          = (x: (-vz)*0+(-vy)*0+vx*(-1)-0*0,
//!             y: (-vz)*0-(-vy)*(-1)+vx*0+0*0,
//!             z: (-vz)*(-1)+(-vy)*0-vx*0+0*0,
//!             w: 0-(-vy)*0-vx*0-0*(-1))
//!          = (-vx, -vy, vz, 0)
//!   ```
//!
//!   so `transform_a_to_b(p) = (tx - p.x, ty - p.y, p.z + tz)` and
//!   `transform_b_to_a(p) = (tx - p.x, ty - p.y, p.z - tz)` (both map `+x ->
//!   -x`, `+y -> -y`, `+z -> +z` before/after the translation), and their
//!   composition is the identity: every multiplication involved is by
//!   `Fix128::ZERO`/`Fix128::ONE`/`-Fix128::ONE`, which this crate's
//!   128x128-bit fixed-point multiply performs exactly (no truncation),
//!   and addition/subtraction of `Fix128` is exact (wrapping integer
//!   arithmetic) regardless of magnitude.
//!
//! # Degenerate inputs (documented result, not merely "no panic")
//!
//! * **Zero worlds**: `step_all`/`step_all_parallel` over an empty
//!   `MultiWorld` is a no-op (`world_count()` stays `0`, no panic).
//! * **`transfer_body` with an out-of-range `from`/`to` world index**
//!   returns `None` and mutates neither world.
//! * **`transfer_body` with an out-of-range `body_id`** returns `None`.
//! * **`transfer_body(w, id, w, ..)` (same source and destination)**
//!   returns `None` unconditionally, even when `id` is valid.
//! * **Transferring "the same `body_id` twice"**: because removal is
//!   `swap_remove`, the slot vacated by the first transfer is immediately
//!   occupied by whatever body used to be last; a second
//!   `transfer_body(w, id, ..)` with the same `id` therefore succeeds
//!   again and moves *that* body, it does not error and does not refer to
//!   the already-moved body.
//! * **`Portal`'s `world_a`/`world_b` fields are not validated against
//!   any `MultiWorld`** by `transform_a_to_b`/`transform_b_to_a` -- those
//!   two methods only read `self.transform`, so a `Portal` built with
//!   out-of-range `world_a`/`world_b` indices still transforms positions
//!   correctly (the indices are caller-facing metadata, not used for
//!   array indexing inside these methods). Asserted explicitly below so a
//!   future change that *does* start indexing with them is caught.
//! * **Extreme offsets**: `Fix128` add/sub is a wrapping group operation
//!   (mod 2^128), so `transform_a_to_b`/`transform_b_to_a` never panic
//!   regardless of magnitude (checked via `catch_unwind`), and because
//!   wrapping subtraction is the exact group inverse of wrapping addition,
//!   the round trip through an extreme translation still lands back on
//!   the exact original value even if the intermediate result wraps
//!   around the 128-bit range.

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::multi_world::{MultiWorld, Portal};
use alice_physics::solver::{PhysicsConfig, RigidBody};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn dt60() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn body_at(x: i64, y: i64, z: i64) -> RigidBody {
    RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE)
}

// ---------------------------------------------------------------------
// add_world / world_count: closed form idx == worlds.len() before push
// ---------------------------------------------------------------------

#[test]
fn add_world_returns_sequential_indices_and_world_count_matches_call_count() {
    let mut mw = MultiWorld::new();
    assert_eq!(mw.world_count(), 0);
    for expected_idx in 0..5usize {
        let idx = mw.add_world(PhysicsConfig::default());
        assert_eq!(
            idx, expected_idx,
            "add_world must return worlds.len() before the push"
        );
        assert_eq!(mw.world_count(), expected_idx + 1);
    }
}

#[test]
fn world_count_zero_worlds_is_not_a_special_case() {
    let mw = MultiWorld::new();
    assert_eq!(mw.world_count(), 0);
    assert_eq!(mw.total_body_count(), 0);
}

// ---------------------------------------------------------------------
// total_body_count: independent per-world sum, including an empty world
// ---------------------------------------------------------------------

#[test]
fn total_body_count_matches_independent_per_world_sum() {
    let mut mw = MultiWorld::new();
    let a = mw.add_world(PhysicsConfig::default());
    let b = mw.add_world(PhysicsConfig::default());
    let c = mw.add_world(PhysicsConfig::default()); // stays empty

    mw.worlds[a].add_body(body_at(0, 0, 0));
    mw.worlds[a].add_body(body_at(1, 0, 0));
    mw.worlds[a].add_body(body_at(2, 0, 0));
    mw.worlds[b].add_body(body_at(0, 1, 0));

    let independent: usize = [a, b, c].iter().map(|&w| mw.worlds[w].bodies.len()).sum();
    assert_eq!(independent, 4);
    assert_eq!(mw.total_body_count(), independent);
    assert_eq!(
        mw.worlds[c].bodies.len(),
        0,
        "sanity: world c really is empty"
    );
}

// ---------------------------------------------------------------------
// step_all: per-world independence, zero-world no-op
// ---------------------------------------------------------------------

#[test]
fn step_all_on_zero_worlds_is_a_no_op() {
    let mut mw = MultiWorld::new();
    let result = catch_unwind(AssertUnwindSafe(move || {
        mw.step_all(dt60());
        mw.world_count()
    }));
    assert_eq!(
        result.expect("step_all on an empty MultiWorld must not panic"),
        0
    );
}

#[test]
fn step_all_steps_every_world_independently() {
    // The "drifting" world's body carries a nonzero initial velocity under
    // zero gravity: if `step_all` ever skipped that world's `world.step()`
    // call (e.g. only stepping the first/last world), its position would
    // stay put -- unlike a merely-static body, for which "not stepped" and
    // "stepped with zero net force" are observationally identical. This
    // closes that blind spot.
    let mut mw = MultiWorld::new();
    let falling = mw.add_world(PhysicsConfig::default());
    let still = mw.add_world(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    let drifting = mw.add_world(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    mw.worlds[falling].add_body(body_at(0, 10, 0));
    mw.worlds[still].add_body(body_at(0, 10, 0));
    let mut drifter = body_at(0, 10, 0);
    drifter.velocity = Vec3Fix::from_int(2, 0, 0);
    mw.worlds[drifting].add_body(drifter);

    for _ in 0..10 {
        mw.step_all(dt60());
    }

    assert!(
        mw.worlds[falling].bodies[0].position.y < Fix128::from_int(10),
        "body under gravity must fall"
    );
    assert_eq!(
        mw.worlds[still].bodies[0].position.y,
        Fix128::from_int(10),
        "body in a zero-gravity world must not move: step_all must not leak gravity across worlds"
    );
    assert!(
        mw.worlds[drifting].bodies[0].position.x > Fix128::ZERO,
        "body with nonzero velocity must actually be advanced by step_all \
         (catches a mutation that silently skips a world's step() call)"
    );
}

// ---------------------------------------------------------------------
// step_all_parallel: bit-identity oracle against step_all (feature "parallel")
// ---------------------------------------------------------------------

#[cfg(feature = "parallel")]
#[test]
fn step_all_parallel_is_bit_identical_to_step_all() {
    fn build() -> MultiWorld {
        let mut m = MultiWorld::new();
        let a = m.add_world(PhysicsConfig::default());
        m.worlds[a].add_body(body_at(0, 10, 0));
        m.worlds[a].add_body(body_at(5, 10, 0));
        let b = m.add_world(PhysicsConfig {
            gravity: Vec3Fix::from_int(0, -20, 0),
            ..PhysicsConfig::default()
        });
        m.worlds[b].add_body(body_at(0, 3, 0));
        m.add_world(PhysicsConfig::default()); // empty, must stay empty
        m
    }

    let mut seq = build();
    let mut par = build();
    for frame in 0..20 {
        seq.step_all(dt60());
        par.step_all_parallel(dt60());
        for (w, (ws, wp)) in seq.worlds.iter().zip(par.worlds.iter()).enumerate() {
            assert_eq!(ws.bodies.len(), wp.bodies.len(), "frame {frame} world {w}");
            for (i, (bs, bp)) in ws.bodies.iter().zip(wp.bodies.iter()).enumerate() {
                assert_eq!(bs.position, bp.position, "frame {frame} world {w} body {i}");
                assert_eq!(bs.velocity, bp.velocity, "frame {frame} world {w} body {i}");
            }
        }
    }
    assert_eq!(par.worlds[2].bodies.len(), 0);
}

#[cfg(feature = "parallel")]
#[test]
fn step_all_parallel_on_zero_worlds_is_a_no_op() {
    let mut mw = MultiWorld::new();
    let result = catch_unwind(AssertUnwindSafe(move || {
        mw.step_all_parallel(dt60());
        mw.world_count()
    }));
    assert_eq!(
        result.expect("step_all_parallel on an empty MultiWorld must not panic"),
        0
    );
}

// ---------------------------------------------------------------------
// transfer_body: closed-form index/count bookkeeping + exact position write
// ---------------------------------------------------------------------

#[test]
fn transfer_body_moves_exactly_one_body_with_exact_new_position() {
    let mut mw = MultiWorld::new();
    let from = mw.add_world(PhysicsConfig::default());
    let to = mw.add_world(PhysicsConfig::default());
    mw.worlds[from].add_body(body_at(0, 5, 0));
    mw.worlds[to].add_body(body_at(9, 9, 9)); // pre-existing body in destination

    let before_from = mw.worlds[from].bodies.len();
    let before_to = mw.worlds[to].bodies.len();
    let new_pos = Vec3Fix::from_int(10, 20, 30);

    let new_id = mw
        .transfer_body(from, 0, to, new_pos)
        .expect("valid transfer must return Some");

    assert_eq!(
        new_id, before_to,
        "closed form: new_id == destination length before transfer"
    );
    assert_eq!(mw.worlds[from].bodies.len(), before_from - 1);
    assert_eq!(mw.worlds[to].bodies.len(), before_to + 1);
    assert_eq!(mw.worlds[to].bodies[new_id].position, new_pos);
    assert_eq!(
        mw.worlds[to].bodies[new_id].prev_position, new_pos,
        "prev_position must also be overwritten (no stale XPBD state)"
    );
}

#[test]
fn transfer_body_moves_the_body_at_a_nonzero_body_id_not_body_zero() {
    // All the other transfer_body tests happen to pass body_id == 0 for a
    // successful move, which cannot distinguish "removed the requested
    // body_id" from "always removed index 0, ignoring body_id". This test
    // transfers the *middle* body of three and checks that exactly that
    // one is the one that moved.
    let mut mw = MultiWorld::new();
    let from = mw.add_world(PhysicsConfig::default());
    let to = mw.add_world(PhysicsConfig::default());
    mw.worlds[from].add_body(body_at(0, 0, 0)); // body_id 0
    mw.worlds[from].add_body(body_at(1, 0, 0)); // body_id 1 -- the one transferred
    mw.worlds[from].add_body(body_at(2, 0, 0)); // body_id 2

    let new_pos = Vec3Fix::from_int(42, 0, 0);
    let new_id = mw
        .transfer_body(from, 1, to, new_pos)
        .expect("valid transfer must return Some");

    assert_eq!(mw.worlds[to].bodies[new_id].position, new_pos);
    assert_eq!(
        mw.worlds[from].bodies.len(),
        2,
        "exactly one body left `from`"
    );
    // swap_remove(1) moves the last body (originally x=2) into slot 1;
    // body_id 0 (x=0) is untouched.
    let remaining_x: std::collections::BTreeSet<i64> = mw.worlds[from]
        .bodies
        .iter()
        .map(|b| b.position.x.hi)
        .collect();
    assert_eq!(
        remaining_x,
        [0, 2].into_iter().collect(),
        "the body at x=1 (body_id 1) must be the one that left, not body_id 0"
    );
}

#[test]
fn transfer_body_degenerate_world_indices_return_none_and_do_not_mutate() {
    let mut mw = MultiWorld::new();
    let a = mw.add_world(PhysicsConfig::default());
    mw.worlds[a].add_body(body_at(0, 0, 0));

    let before = mw.worlds[a].bodies.len();
    assert!(
        mw.transfer_body(a, 0, 99, Vec3Fix::ZERO).is_none(),
        "destination out of range"
    );
    assert!(
        mw.transfer_body(99, 0, a, Vec3Fix::ZERO).is_none(),
        "source out of range"
    );
    assert!(
        mw.transfer_body(a, 0, a, Vec3Fix::ZERO).is_none(),
        "same source and destination, even though body_id is valid"
    );
    assert_eq!(
        mw.worlds[a].bodies.len(),
        before,
        "none of the rejected calls may mutate state"
    );
}

#[test]
fn transfer_body_out_of_range_body_id_returns_none() {
    let mut mw = MultiWorld::new();
    let a = mw.add_world(PhysicsConfig::default());
    let b = mw.add_world(PhysicsConfig::default());
    assert!(
        mw.transfer_body(a, 0, b, Vec3Fix::ZERO).is_none(),
        "world a has zero bodies"
    );

    mw.worlds[a].add_body(body_at(0, 0, 0));
    assert!(
        mw.transfer_body(a, 1, b, Vec3Fix::ZERO).is_none(),
        "body_id 1 is out of range for a single-body world"
    );
    assert_eq!(
        mw.worlds[a].bodies.len(),
        1,
        "rejected call must not mutate"
    );
}

#[test]
fn transfer_body_same_body_id_twice_moves_the_swap_remove_replacement() {
    // 3 bodies in `from`; transfer index 0 (swap_remove relocates the
    // *last* body, index 2, into slot 0). A second transfer of index 0
    // must therefore move what was originally body index 2, not error,
    // not silently no-op, and not refer to the already-moved body.
    let mut mw = MultiWorld::new();
    let from = mw.add_world(PhysicsConfig::default());
    let to = mw.add_world(PhysicsConfig::default());
    mw.worlds[from].add_body(body_at(0, 0, 0)); // idx 0: moved first
    mw.worlds[from].add_body(body_at(1, 0, 0)); // idx 1: stays
    mw.worlds[from].add_body(body_at(2, 0, 0)); // idx 2: swap_remove target -> becomes idx 0

    let first_target = Vec3Fix::from_int(100, 0, 0);
    let first_id = mw
        .transfer_body(from, 0, to, first_target)
        .expect("first transfer must succeed");
    assert_eq!(mw.worlds[to].bodies[first_id].position, first_target);
    // original body idx 2 (x=2) is now at slot 0 in `from` (swap_remove)
    assert_eq!(mw.worlds[from].bodies[0].position.x, Fix128::from_int(2));
    assert_eq!(mw.worlds[from].bodies.len(), 2);

    let second_target = Vec3Fix::from_int(200, 0, 0);
    let second_id = mw
        .transfer_body(from, 0, to, second_target)
        .expect("second transfer of body_id 0 must succeed");
    assert_ne!(
        second_id, first_id,
        "second transfer must land in a fresh destination slot"
    );
    assert_eq!(mw.worlds[to].bodies[second_id].position, second_target);
    // the body that moved second must be the one that was at x=2 (relocated by swap_remove),
    // not the body at x=0 (already moved) or x=1 (untouched, still in `from`).
    assert_eq!(mw.worlds[from].bodies.len(), 1);
    assert_eq!(
        mw.worlds[from].bodies[0].position.x,
        Fix128::from_int(1),
        "the untouched body (x=1) must remain in `from`"
    );
}

// ---------------------------------------------------------------------
// Portal::transform_a_to_b / transform_b_to_a: closed form + exact round trip
// ---------------------------------------------------------------------

/// `q = (0,0,1,0)`: pure unit quaternion along +Z, a 180 degree rotation
/// about Z. See module doc comment for the hand-derived Hamilton product.
fn rot_180_about_z() -> QuatFix {
    QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO)
}

#[test]
fn transform_a_to_b_matches_hand_derived_180_about_z_closed_form() {
    let translation = Vec3Fix::from_int(50, -20, 10);
    let portal = Portal::new(0, 1, translation, rot_180_about_z());
    let p = Vec3Fix::from_int(7, 3, -5);

    let expected = Vec3Fix::new(
        -p.x + translation.x,
        -p.y + translation.y,
        p.z + translation.z,
    );
    assert_eq!(portal.transform_a_to_b(p), expected);
}

#[test]
fn transform_b_to_a_matches_hand_derived_180_about_z_closed_form() {
    let translation = Vec3Fix::from_int(50, -20, 10);
    let portal = Portal::new(0, 1, translation, rot_180_about_z());
    let p = Vec3Fix::from_int(7, 3, -5);

    let expected = Vec3Fix::new(
        translation.x - p.x,
        translation.y - p.y,
        p.z - translation.z,
    );
    assert_eq!(portal.transform_b_to_a(p), expected);
}

#[test]
fn transform_a_to_b_then_b_to_a_round_trips_exactly() {
    let translation = Vec3Fix::from_int(50, -20, 10);
    let portal = Portal::new(0, 1, translation, rot_180_about_z());
    let p = Vec3Fix::from_int(7, 3, -5);

    let to_b = portal.transform_a_to_b(p);
    assert_eq!(
        portal.transform_b_to_a(to_b),
        p,
        "round trip must be bit-exact, not merely approximate"
    );
}

#[test]
fn transform_a_to_b_identity_rotation_reduces_to_pure_translation() {
    let translation = Vec3Fix::from_int(100, 0, 0);
    let portal = Portal::new(0, 1, translation, QuatFix::IDENTITY);
    let p = Vec3Fix::from_int(5, 10, 15);
    assert_eq!(portal.transform_a_to_b(p), Vec3Fix::from_int(105, 10, 15));
}

#[test]
fn transform_methods_ignore_world_a_world_b_indices() {
    // world_a / world_b are caller-facing metadata, not used by either
    // transform method -- a Portal built with indices that refer to no
    // existing MultiWorld (or to each other) must still transform
    // correctly. If a future change starts indexing with them, this must
    // start failing loudly (not silently return a different value).
    let translation = Vec3Fix::from_int(1, 2, 3);
    let rot = rot_180_about_z();
    let in_range = Portal::new(0, 1, translation, rot);
    let out_of_range = Portal::new(usize::MAX - 1, usize::MAX, translation, rot);
    let same_index = Portal::new(7, 7, translation, rot);

    let p = Vec3Fix::from_int(4, 5, 6);
    let expected = in_range.transform_a_to_b(p);
    assert_eq!(out_of_range.transform_a_to_b(p), expected);
    assert_eq!(same_index.transform_a_to_b(p), expected);
}

#[test]
fn transform_round_trip_holds_exactly_under_extreme_offsets_without_panicking() {
    // Fix128 add/sub is a wrapping group operation (mod 2^128): no
    // magnitude can make it panic, and wrapping sub is the exact group
    // inverse of wrapping add, so the round trip must hold exactly even if
    // the intermediate `transform_a_to_b` result wraps around.
    let extreme_translation = Vec3Fix::new(
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::from_raw(i64::MIN, 0),
        Fix128::from_raw(i64::MAX / 2, 1),
    );
    let portal = Portal::new(0, 1, extreme_translation, rot_180_about_z());
    let extreme_p = Vec3Fix::new(
        Fix128::from_raw(i64::MIN / 2, 7),
        Fix128::from_raw(i64::MAX / 2, 0),
        Fix128::from_raw(i64::MIN, u64::MAX),
    );

    let result = catch_unwind(|| {
        let to_b = portal.transform_a_to_b(extreme_p);
        portal.transform_b_to_a(to_b)
    });
    match result {
        Ok(back) => assert_eq!(back, extreme_p, "wrapping add/sub round trip must hold exactly even under wraparound"),
        Err(_) => panic!("Fix128 add/sub is documented as wrapping (mod 2^128); transform round trip must not panic"),
    }
}
