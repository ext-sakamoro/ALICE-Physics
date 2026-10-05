//! Manual frame lifecycle of the world's contact cache
//!
//! Reaches `PhysicsWorld::begin_frame` and `PhysicsWorld::end_frame`.
//!
//! `PhysicsWorld::contact_cache` is filled only by the manual contact APIs
//! (`add_contact` / `add_contact_with_material`), and `step` neither fills,
//! reads, ages nor prunes it: the built-in detection writes straight into the
//! constraint list, and the solver's own warm start lives on
//! `ContactConstraint::cached_lambda`. A host that injects contacts therefore
//! owns the cache lifecycle and drives it with `begin_frame` / `end_frame`
//! around its own frame.
//!
//! Closed form, from the documented contract (`begin_frame` makes every
//! manifold one frame staler, a contact update resets its manifold to 0,
//! `end_frame` keeps manifolds with `stale_frames <= max_stale_frames`):
//!
//! - a pair last touched in frame 0 has `stale_frames = k` at the end of frame
//!   `k`, so with `max_stale_frames = M` it survives frames `1..=M` and is gone
//!   at the end of frame `M + 1`;
//! - a pair touched every frame keeps `stale_frames = 0`, and a point that is
//!   re-reported at the same local position keeps the impulses stored on it
//!   (`warm_start_impulse` returns exactly what `store_impulses` wrote) while
//!   its `age` counts the frames it persisted, `age = k` after frame `k`;
//! - `step` between the two calls leaves every `stale_frames` unchanged.
//!
//! Run with: `cargo run --example world_contact_cache_frame`

use alice_physics::collider::Contact;
use alice_physics::contact_cache::BodyPairKey;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};

fn contact_at(x: i64) -> Contact {
    let p = Vec3Fix::from_int(x, 0, 0);
    Contact {
        depth: Fix128::from_ratio(1, 100),
        normal: Vec3Fix::UNIT_Y,
        point_a: p,
        point_b: p,
    }
}

fn stale(world: &PhysicsWorld, pair: BodyPairKey) -> Option<u32> {
    world.contact_cache.find(&pair).map(|m| m.stale_frames)
}

fn main() {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(cfg);
    // Four bodies far apart: the built-in detection finds nothing, so `step`
    // has no contact of its own.
    let b: Vec<usize> = (0..4)
        .map(|i| {
            world.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(100 * i, 0, 0),
                Fix128::ONE,
            ))
        })
        .collect();
    let kept = BodyPairKey::new(b[0], b[1]);
    let abandoned = BodyPairKey::new(b[2], b[3]);
    let max_stale = world.contact_cache.max_stale_frames;
    let dt = Fix128::from_ratio(1, 60);

    // ---- frame 0: both pairs reported -----------------------------------
    world.begin_frame();
    world.add_contact(ContactConstraint::new(b[0], b[1], contact_at(1)));
    world.add_contact(ContactConstraint::new(b[2], b[3], contact_at(2)));
    world.step(dt);
    world.end_frame();
    assert_eq!(world.contact_cache.manifold_count(), 2);
    assert_eq!(stale(&world, kept), Some(0));
    assert_eq!(stale(&world, abandoned), Some(0));

    // The host's solver writes its impulses back into the cache.
    let (ln, lt1, lt2) = (
        Fix128::from_ratio(3, 4),
        Fix128::from_ratio(-1, 8),
        Fix128::from_ratio(1, 16),
    );
    world
        .contact_cache
        .get_or_create(kept, Fix128::ZERO, Fix128::ZERO)
        .store_impulses(0, ln, lt1, lt2);

    // ---- frames 1..=M+1: only `kept` is reported ------------------------
    for k in 1..=max_stale + 1 {
        world.begin_frame();
        world.add_contact(ContactConstraint::new(b[0], b[1], contact_at(1)));
        let before_step = (stale(&world, kept), stale(&world, abandoned));
        world.step(dt);
        let after_step = (stale(&world, kept), stale(&world, abandoned));
        assert_eq!(before_step, after_step, "frame {k}: step touched the cache");
        world.end_frame();

        let expect_alive = k <= max_stale;
        let alive = stale(&world, abandoned).is_some();
        println!(
            "[world_contact_cache_frame] end of frame {k}: abandoned alive = {alive} \
             (closed form k <= {max_stale}: {expect_alive}), stale = {:?}",
            stale(&world, abandoned)
        );
        assert_eq!(alive, expect_alive, "frame {k}: expiry");
        if expect_alive {
            assert_eq!(stale(&world, abandoned), Some(k), "frame {k}: stale count");
        }

        let m = world
            .contact_cache
            .find(&kept)
            .expect("kept pair is never pruned");
        assert_eq!(m.stale_frames, 0, "frame {k}: reported pair is fresh");
        assert_eq!(
            m.point_count(),
            1,
            "frame {k}: same point, matched not added"
        );
        assert_eq!(
            m.warm_start_impulse(0),
            (ln, lt1, lt2),
            "frame {k}: stored impulses survive the frame"
        );
        assert_eq!(m.points[0].age, k, "frame {k}: age counts persisted frames");
    }
    assert_eq!(world.contact_cache.manifold_count(), 1);

    // ---- without end_frame nothing is pruned -----------------------------
    for _ in 0..=max_stale {
        world.begin_frame();
    }
    assert_eq!(stale(&world, kept), Some(max_stale + 1));
    assert_eq!(
        world.contact_cache.manifold_count(),
        1,
        "begin_frame alone never prunes"
    );
    world.end_frame();
    assert_eq!(
        world.contact_cache.manifold_count(),
        0,
        "end_frame prunes past the limit"
    );

    println!("[world_contact_cache_frame] all closed-form checks passed");
}
