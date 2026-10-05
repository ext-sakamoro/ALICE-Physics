//! `ContactCache` is a standalone, opt-in warm-start bookkeeping API: a host
//! builds its own `ContactCache`, feeds it contacts via `get_or_create` /
//! `ContactManifold::add_or_update` (both already wired in production
//! through `PhysicsWorld::add_contact`, `src/solver.rs`), and -- if it wants
//! to pre-apply the previous frame's accumulated impulses to its own body
//! array before its own solve -- reads/writes the per-point lambdas with
//! `warm_start_impulse` / `store_impulses` / `apply_warm_start`.
//!
//! Wiring: `apply_warm_start`, `manifold_count`, `point_count`,
//! `store_impulses`, `tangent_frame`, `total_contact_points`,
//! `warm_start_impulse` had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/contact_cache.rs::*`). This
//! is their production entry point.
//!
//! `tangent_frame` is `pub(crate)`, not `pub` -- it cannot be called
//! directly from an example (a separate crate). It is reached transitively:
//! `apply_warm_start`'s body calls it (`src/contact_cache.rs` line ~405), so
//! once `apply_warm_start` has a live caller here, `tangent_frame` becomes
//! reachable from that same root and `scripts/wiring_guard.py`'s fixed-point
//! reachability analysis marks it wired too.
//!
//! ⚠️ This is deliberately **not** wired into `PhysicsWorld::step()`'s
//! XPBD contact-solving loop. Two independent reasons, both already on
//! record before this file existed:
//!
//! 1. The *production* contact warm-start is a different mechanism:
//!    `ContactConstraint::cached_lambda`, reset once per substep in
//!    `reset_lambdas` and accumulated by `solve_contact_pair` (see
//!    `src/solver.rs` around line 2560-2580 and 2933-2980). `ContactCache`'s
//!    own per-point `lambda_n`/`lambda_t1`/`lambda_t2` fields are written by
//!    `ContactManifold::add_or_update` (preserved across frames when a point
//!    matches) but never read back into that solve -- it is a second,
//!    parallel ledger `step()` does not consult.
//! 2. Reviving this exact cache into the CPU step loop was already
//!    investigated (2026-09-30) and decided against: doing so would make
//!    simulation results depend on `ContactCache` state that is explicitly
//!    *excluded* from `serialize_state`/`deserialize_state` snapshot
//!    coverage, which breaks the bit-exact rollback property the World
//!    Model doctrine requires (replaying from a snapshot would silently
//!    diverge from the original run because the cache's warm-start history
//!    is not part of what gets restored). That ruling is specifically about
//!    the *production, snapshot-governed* `step()` loop; it does not apply
//!    here because this file never calls `PhysicsWorld::step()` at all --
//!    it demonstrates the cache as the separate, self-contained API it
//!    already is, exactly as `TgsCacheStats` is demonstrated as a read-only
//!    diagnostic in `examples/tgs_solver_backend.rs` rather than forced into
//!    the hot path.
//!
//! ```bash
//! cargo run --example contact_warm_start_cache --features std
//! ```

use alice_physics::collider::Contact;
use alice_physics::contact_cache::{BodyPairKey, ContactCache};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn contact(normal: Vec3Fix, depth: Fix128) -> Contact {
    Contact {
        depth,
        normal,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    }
}

// ======================================================================
// Scene 1: `manifold_count` / `point_count` / `total_contact_points`.
//
// Two body pairs, 3 points on the first pair's manifold (capped at
// `MAX_MANIFOLD_POINTS` = 4, so 3 distinct points all survive) and 1 on the
// second. Hand count: 2 manifolds, 3 + 1 = 4 points total.
// ======================================================================
fn scene_counts() {
    let mut cache = ContactCache::new();
    let pair_a = BodyPairKey::new(0, 1);
    let pair_b = BodyPairKey::new(2, 3);

    {
        let m = cache.get_or_create(pair_a, Fix128::ONE, Fix128::ZERO);
        for i in 0..3 {
            m.add_or_update(
                &contact(Vec3Fix::UNIT_Y, fi(1)),
                v3i(i * 10, 0, 0), // 10 units apart: each is a distinct point
                Vec3Fix::ZERO,
            );
        }
    }
    {
        let m = cache.get_or_create(pair_b, Fix128::ONE, Fix128::ZERO);
        m.add_or_update(
            &contact(Vec3Fix::UNIT_Y, fi(1)),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        );
    }

    let manifold_count = cache.manifold_count();
    let point_count_a = cache.find(&pair_a).map_or(0, |m| m.point_count());
    let point_count_b = cache.find(&pair_b).map_or(0, |m| m.point_count());
    let total = cache.total_contact_points();

    assert_eq!(manifold_count, 2, "2 distinct body pairs -> 2 manifolds");
    assert_eq!(
        point_count_a, 3,
        "3 points added, all >1cm apart -> all distinct"
    );
    assert_eq!(point_count_b, 1);
    assert_eq!(
        total, 4,
        "total_contact_points is the sum across all manifolds (3+1)"
    );

    println!(
        "[contact_warm_start_cache counts] manifold_count={manifold_count} point_count(pair_a)={point_count_a} point_count(pair_b)={point_count_b} total_contact_points={total}"
    );
}

// ======================================================================
// Scene 2: `store_impulses` / `warm_start_impulse` round trip, plus the
// documented out-of-range behaviour (both are `point_idx < len` guarded,
// returning/no-op'ing on miss rather than panicking).
// ======================================================================
fn scene_warm_start_round_trip() {
    let mut cache = ContactCache::new();
    let pair = BodyPairKey::new(5, 9);
    let m = cache.get_or_create(pair, Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_Y, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );

    // Before any store: a freshly-added point starts at (0,0,0).
    let before = m.warm_start_impulse(0);
    assert_eq!(before, (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO));

    m.store_impulses(0, fi(11), fi(-4), fi(7));
    let after = m.warm_start_impulse(0);
    assert_eq!(
        after,
        (fi(11), fi(-4), fi(7)),
        "store_impulses/warm_start_impulse must round-trip exactly"
    );

    // Out-of-range point_idx: warm_start_impulse returns the zero triple
    // (not a panic), store_impulses is a silent no-op (point_count unchanged,
    // and index 0's impulses above are untouched).
    let miss = m.warm_start_impulse(99);
    assert_eq!(miss, (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO));
    m.store_impulses(99, fi(-1), fi(-1), fi(-1));
    assert_eq!(
        m.point_count(),
        1,
        "store_impulses(out_of_range) must not grow the manifold"
    );
    assert_eq!(
        m.warm_start_impulse(0),
        (fi(11), fi(-4), fi(7)),
        "out-of-range store must not corrupt index 0"
    );

    println!(
        "[contact_warm_start_cache round_trip] before={before:?} after_store={after:?} out_of_range_read={miss:?} point_count_after_oob_store={}",
        m.point_count()
    );
}

// ======================================================================
// Scene 3: `apply_warm_start` -- the actual impulse-to-velocity step.
//
// normal = UNIT_X. `tangent_frame`'s reference-axis rule (hand-derived from
// `src/contact_cache.rs`'s documented selection, NOT by calling the
// function under test):
//   abs_x=1, abs_y=0, abs_z=0
//   abs_x <= abs_y?  1 <= 0 -> false
//   abs_y <= abs_z?  0 <= 0 -> true  => reference = UNIT_Y
//   t1 = normal × reference = X × Y = Z
//   t2 = normal × t1        = X × Z = -Y
// factor = 1/2 (cache.warm_start_factor overridden for a round number).
// lambda_n=8, lambda_t1=12, lambda_t2=20 on body pair (0,1), body 0 dynamic
// (inv_mass=1), body 1 static (inv_mass=0, must stay exactly zero).
//   impulse = X*(8*0.5) + Z*(12*0.5) + (-Y)*(20*0.5) = (4,0,0)+(0,0,6)+(0,-10,0)
//           = (4, -10, 6)
//   body0.velocity += impulse * inv_mass(1) = (4, -10, 6)
//   body1.velocity unaffected (static, inv_mass==0 guard)
// A second pair (2,3) is added with body index 3 out of range for a
// shorter `bodies` slice passed separately, to exercise the documented
// "a_idx/b_idx >= bodies.len() -> skip this manifold" guard without panicking.
// ======================================================================
fn scene_apply_warm_start() {
    let mut cache = ContactCache::new();
    cache.warm_start_factor = Fix128::from_ratio(1, 2);

    let pair = BodyPairKey::new(0, 1);
    let m = cache.get_or_create(pair, Fix128::ONE, Fix128::ZERO);
    m.add_or_update(
        &contact(Vec3Fix::UNIT_X, fi(1)),
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    );
    m.store_impulses(0, fi(8), fi(12), fi(20));

    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), // inv_mass = 1
        RigidBody::new_static(v3i(1, 0, 0)),                // inv_mass = 0
    ];
    cache.apply_warm_start(&mut bodies);

    let expected_body0 = v3i(4, -10, 6);
    assert_eq!(
        bodies[0].velocity, expected_body0,
        "hand-derived impulse must match apply_warm_start's output exactly"
    );
    assert_eq!(
        bodies[1].velocity,
        Vec3Fix::ZERO,
        "static body (inv_mass=0) must stay exactly zero"
    );

    println!(
        "[contact_warm_start_cache apply_warm_start] factor=0.5 lambda=(8,12,20) normal=+X -> body0.velocity={:?} (closed form {:?}) body1(static).velocity={:?}",
        bodies[0].velocity, expected_body0, bodies[1].velocity
    );

    // Degenerate: manifold references a body index past the end of the
    // slice actually passed to apply_warm_start -- must skip, not panic or
    // index out of bounds.
    let mut short = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    cache.apply_warm_start(&mut short); // pair (0,1): body 1 >= short.len()==1
    assert_eq!(
        short[0].velocity,
        Vec3Fix::ZERO,
        "out-of-range manifold must be skipped, not applied partially"
    );
    println!("[contact_warm_start_cache apply_warm_start] body-index-out-of-range manifold skipped cleanly (short.len()=1)");

    // Degenerate: empty cache against nonempty bodies -- a true no-op.
    let empty = ContactCache::new();
    let mut untouched = vec![RigidBody::new_dynamic(v3i(5, 5, 5), Fix128::ONE)];
    untouched[0].velocity = v3i(1, 2, 3);
    empty.apply_warm_start(&mut untouched);
    assert_eq!(
        untouched[0].velocity,
        v3i(1, 2, 3),
        "empty cache must leave velocities untouched"
    );
    println!("[contact_warm_start_cache apply_warm_start] empty cache is a true no-op");
}

fn main() {
    println!("ALICE-Physics ContactCache warm-start API wiring demo");
    println!("======================================================");
    println!();
    println!("-- Scene 1: manifold_count / point_count / total_contact_points --");
    scene_counts();
    println!();
    println!("-- Scene 2: store_impulses / warm_start_impulse round trip --");
    scene_warm_start_round_trip();
    println!();
    println!("-- Scene 3: apply_warm_start (and tangent_frame, reached transitively) --");
    scene_apply_warm_start();
    println!();
    println!(
        "Done. ContactCache is a standalone opt-in API -- see this file's module doc for why it is not wired into PhysicsWorld::step()."
    );
}
