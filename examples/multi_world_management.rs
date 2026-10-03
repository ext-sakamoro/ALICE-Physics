//! Multi-world management: independent `PhysicsWorld` instances with
//! cross-world body transfer and portal coordinate-frame conversion.
//!
//! Wiring: `MultiWorld::{add_world, step_all, step_all_parallel,
//! total_body_count, transfer_body, world_count}` and
//! `Portal::{transform_a_to_b, transform_b_to_a}` had zero production
//! callers (`scripts/wiring-baseline.txt` `unwired src/multi_world.rs::*`,
//! 8 items). This is their production entry point.
//!
//! ```bash
//! cargo run --example multi_world_management --features std
//! cargo run --example multi_world_management --features std,parallel
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::multi_world::{MultiWorld, Portal};
use alice_physics::solver::{PhysicsConfig, RigidBody};

fn main() {
    // ------------------------------------------------------------------
    // add_world / world_count: closed form -- the returned index equals
    // the number of worlds already present, and world_count() equals the
    // number of add_world() calls made so far.
    // ------------------------------------------------------------------
    let mut mw = MultiWorld::new();
    let w0 = mw.add_world(PhysicsConfig::default());
    let w1 = mw.add_world(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -20, 0),
        ..PhysicsConfig::default()
    });
    let w2 = mw.add_world(PhysicsConfig::default());
    assert_eq!(
        (w0, w1, w2),
        (0, 1, 2),
        "add_world index == worlds.len() before push"
    );
    assert_eq!(mw.world_count(), 3, "world_count == #add_world calls");
    println!(
        "[multi_world] world_count after 3x add_world = {}",
        mw.world_count()
    );

    // ------------------------------------------------------------------
    // total_body_count: cross-checked against a sum computed independently
    // in this example (not by calling total_body_count itself).
    // ------------------------------------------------------------------
    mw.worlds[w0].add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    mw.worlds[w0].add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 10, 0),
        Fix128::ONE,
    ));
    mw.worlds[w1].add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 5, 0),
        Fix128::ONE,
    ));
    // w2 stays empty on purpose (exercises the "some worlds empty" path).
    let independent_sum: usize = mw.worlds.iter().map(|w| w.bodies.len()).sum();
    assert_eq!(independent_sum, 3);
    assert_eq!(
        mw.total_body_count(),
        independent_sum,
        "total_body_count == independent per-world sum"
    );
    println!(
        "[multi_world] total_body_count = {} (independent sum = {})",
        mw.total_body_count(),
        independent_sum
    );

    // ------------------------------------------------------------------
    // step_all: sequential stepping of every world, each independently.
    // ------------------------------------------------------------------
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        mw.step_all(dt);
    }
    let w0_y_after_seq = mw.worlds[w0].bodies[0].position.y;
    assert!(
        w0_y_after_seq < Fix128::from_int(10),
        "body in world0 must have fallen under gravity: y={w0_y_after_seq}"
    );
    // world2 is still empty: step_all over zero bodies is a documented no-op.
    assert_eq!(mw.worlds[w2].bodies.len(), 0);
    println!("[multi_world] after 30x step_all, world0 body0 y = {w0_y_after_seq}");

    // ------------------------------------------------------------------
    // step_all_parallel (feature "parallel"): closed-form oracle is
    // bit-identity against step_all on an equivalent scene -- the module
    // documents per-world independence, so stepping in parallel across
    // worlds must reproduce the sequential trajectory exactly, frame by
    // frame, for every body in every world.
    // ------------------------------------------------------------------
    #[cfg(feature = "parallel")]
    {
        fn build_scene() -> MultiWorld {
            let mut m = MultiWorld::new();
            let a = m.add_world(PhysicsConfig::default());
            m.worlds[a].add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(0, 10, 0),
                Fix128::ONE,
            ));
            m.worlds[a].add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(3, 10, 0),
                Fix128::ONE,
            ));
            let b = m.add_world(PhysicsConfig {
                gravity: Vec3Fix::from_int(0, -20, 0),
                ..PhysicsConfig::default()
            });
            m.worlds[b].add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(0, 5, 0),
                Fix128::ONE,
            ));
            m.add_world(PhysicsConfig::default()); // empty world: must stay empty under parallel too
            m
        }

        let mut seq = build_scene();
        let mut par = build_scene();
        for frame in 0..30 {
            seq.step_all(dt);
            par.step_all_parallel(dt);
            assert_eq!(seq.world_count(), par.world_count(), "frame {frame}");
            for (w, (ws, wp)) in seq.worlds.iter().zip(par.worlds.iter()).enumerate() {
                assert_eq!(ws.bodies.len(), wp.bodies.len(), "frame {frame} world {w}");
                for (i, (bs, bp)) in ws.bodies.iter().zip(wp.bodies.iter()).enumerate() {
                    assert_eq!(
                        bs.position, bp.position,
                        "frame {frame} world {w} body {i}: position diverged"
                    );
                    assert_eq!(
                        bs.velocity, bp.velocity,
                        "frame {frame} world {w} body {i}: velocity diverged"
                    );
                }
            }
        }
        assert_eq!(
            par.worlds[2].bodies.len(),
            0,
            "the untouched 3rd world must stay empty under parallel stepping"
        );
        println!("[multi_world] step_all_parallel bit-identical to step_all over 30 frames (2 worlds, 3 bodies): ok");

        // Degenerate: stepping zero worlds in parallel is a documented no-op.
        let mut empty = MultiWorld::new();
        empty.step_all_parallel(dt);
        assert_eq!(empty.world_count(), 0);
        println!("[multi_world] step_all_parallel on an empty MultiWorld: no-op, ok");
    }
    #[cfg(not(feature = "parallel"))]
    println!("[multi_world] step_all_parallel skipped (build without --features parallel)");

    // ------------------------------------------------------------------
    // transfer_body: moves a body from one world to another, re-homing its
    // position. Closed form: the returned index equals the destination
    // world's body count *before* the transfer (it is pushed onto the end),
    // the source count drops by 1, the destination count rises by 1, and
    // the body lands at exactly `new_position` (bit-exact field write).
    // ------------------------------------------------------------------
    let before_from = mw.worlds[w0].bodies.len();
    let before_to = mw.worlds[w2].bodies.len();
    let new_pos = Vec3Fix::from_int(100, 200, 300);
    let new_id = mw
        .transfer_body(w0, 0, w2, new_pos)
        .expect("valid transfer must return Some");
    assert_eq!(
        new_id, before_to,
        "closed form: new_id == destination length before transfer"
    );
    assert_eq!(
        mw.worlds[w0].bodies.len(),
        before_from - 1,
        "source world lost exactly one body"
    );
    assert_eq!(
        mw.worlds[w2].bodies.len(),
        before_to + 1,
        "destination world gained exactly one body"
    );
    assert_eq!(
        mw.worlds[w2].bodies[new_id].position, new_pos,
        "transferred body must land at new_position exactly"
    );
    assert_eq!(
        mw.worlds[w2].bodies[new_id].prev_position, new_pos,
        "prev_position must match too (no stale XPBD state)"
    );
    println!(
        "[multi_world] transfer_body world{w0}->world{w2}: new_id={new_id}, position={:?}",
        mw.worlds[w2].bodies[new_id].position
    );

    // Degenerate inputs (documented via the Option<usize> return: None on
    // any invalid index or a same-world "transfer").
    assert!(
        mw.transfer_body(w0, 0, 99, Vec3Fix::ZERO).is_none(),
        "destination world out of range -> None"
    );
    assert!(
        mw.transfer_body(99, 0, w2, Vec3Fix::ZERO).is_none(),
        "source world out of range -> None"
    );
    assert!(
        mw.transfer_body(w0, 0, w0, Vec3Fix::ZERO).is_none(),
        "same-world transfer -> None (no-op by design)"
    );
    let body_count_before_oob = mw.worlds[w0].bodies.len();
    assert!(
        mw.transfer_body(w0, 999, w2, Vec3Fix::ZERO).is_none(),
        "out-of-range body_id -> None"
    );
    assert_eq!(
        mw.worlds[w0].bodies.len(),
        body_count_before_oob,
        "rejected transfer must not mutate either world"
    );
    println!("[multi_world] degenerate transfer_body inputs (bad world / same world / bad body_id) all return None, unchanged state: ok");

    // Transferring the "same body" twice: the first transfer removes body
    // 0 from world w0 via swap_remove, so a second call with the same
    // body_id now targets whatever body (if any) now occupies that slot,
    // not the original body. With exactly one body left in w0 at this
    // point, body_id 0 is still valid and the second transfer must succeed
    // and move *that* body (not panic, not silently no-op).
    assert_eq!(
        mw.worlds[w0].bodies.len(),
        1,
        "setup: exactly one body left in w0"
    );
    let second_new_id = mw
        .transfer_body(w0, 0, w2, Vec3Fix::from_int(1, 1, 1))
        .expect("second transfer of body_id 0 must succeed");
    assert_eq!(mw.worlds[w0].bodies.len(), 0, "w0 now empty");
    assert_eq!(
        mw.worlds[w2].bodies[second_new_id].position,
        Vec3Fix::from_int(1, 1, 1)
    );
    println!("[multi_world] repeated transfer_body(w0, 0, ..) after a prior removal moves the body that now occupies slot 0: ok");

    // ------------------------------------------------------------------
    // Portal::transform_a_to_b / transform_b_to_a: a non-identity,
    // non-axis-aligned-trivial rotation with an EXACT fixed-point closed
    // form. q = (x=0, y=0, z=1, w=0) is the pure unit quaternion along +Z,
    // i.e. a 180 degree rotation about the Z axis. Hand-derived via the
    // Hamilton product q*v*q^-1 (see this crate's `QuatFix::mul`):
    //
    //   q = (0,0,1,0), q^-1 = conjugate(q) = (0,0,-1,0)
    //   temp = q*qv = (-vy, vx, 0, -vz)
    //   result = temp*q^-1 = (-vx, -vy, vz, 0)
    //
    // so rotate_vec((0,0,1,0), v) = (-v.x, -v.y, v.z) -- exactly, because
    // every multiplication involved is by Fix128::ZERO or Fix128::ONE (or
    // its negation), which Fix128's 128x128 fixed-point multiply performs
    // with no truncation. transform_a_to_b/b_to_a add/subtract a
    // translation on top, which is also exact (wrapping integer add).
    // ------------------------------------------------------------------
    let rot_180_z = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    let translation = Vec3Fix::from_int(50, -20, 10);
    let portal = Portal::new(w0, w2, translation, rot_180_z);

    let p = Vec3Fix::from_int(7, 3, -5);
    let expected_a_to_b = Vec3Fix::new(
        -p.x + translation.x,
        -p.y + translation.y,
        p.z + translation.z,
    );
    let a_to_b = portal.transform_a_to_b(p);
    assert_eq!(
        a_to_b, expected_a_to_b,
        "transform_a_to_b must match the hand-derived 180deg-about-Z closed form"
    );

    let expected_b_to_a_of_a_to_b = p; // exact round trip, derived algebraically above
    let back = portal.transform_b_to_a(a_to_b);
    assert_eq!(
        back, expected_b_to_a_of_a_to_b,
        "transform_b_to_a must exactly invert transform_a_to_b"
    );
    println!(
        "[multi_world] Portal 180deg-Z: transform_a_to_b({p:?}) = {a_to_b:?}, transform_b_to_a(..) round-trips to {back:?}"
    );

    // Extreme offsets: Fix128 arithmetic is a wrapping group under add/sub
    // (documented), so round-tripping through an extreme translation must
    // still land back on the exact original value even if the
    // intermediate `a_to_b` wraps around the 128-bit range, and must not
    // panic doing so.
    let extreme_translation = Vec3Fix::new(
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::from_raw(i64::MIN, 0),
        Fix128::from_raw(i64::MAX / 2, 0),
    );
    let extreme_portal = Portal::new(w0, w2, extreme_translation, rot_180_z);
    let extreme_p = Vec3Fix::new(
        Fix128::from_raw(i64::MIN / 2, 0),
        Fix128::from_raw(i64::MAX / 2, 0),
        Fix128::from_raw(i64::MIN, 0),
    );
    let result = std::panic::catch_unwind(|| {
        let b = extreme_portal.transform_a_to_b(extreme_p);
        let a = extreme_portal.transform_b_to_a(b);
        (b, a)
    });
    match result {
        Ok((_wrapped, roundtrip)) => {
            assert_eq!(roundtrip, extreme_p, "wrapping add/sub is its own group inverse: round trip must hold exactly even under wraparound");
            println!("[multi_world] extreme-offset portal round trip holds exactly under wraparound (no panic): ok");
        }
        Err(_) => {
            panic!("Fix128 add/sub is documented as wrapping (mod 2^128); this must not panic")
        }
    }

    println!("[multi_world] all oracle checks passed");
}
