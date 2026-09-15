#![no_main]
//! Evaluation-path determinism fuzz (1.2.0, ALICE-SDF `fuzz_eval_parity` pattern).
//!
//! The rigid-body solver has two CPU ways to advance the same world: `step`
//! (sequential Gauss–Seidel in constraint index order) and `step_parallel`
//! (graph-coloured batches, `parallel` feature). They are **not** bit-identical
//! once two constraints share a body — the batch order is a different
//! Gauss–Seidel ordering, and the crate documents that lockstep peers must
//! agree on the path — but each of them must be *deterministic*: the same
//! scene stepped twice, on however many Rayon threads, gives the same bits.
//! That is the property the raw-pointer batch solver (`BodySlicePtr` et al.)
//! can silently lose, so this target drives arbitrary scenes — overlapping
//! spheres, resting stacks, distance constraints under contact, mixed masses,
//! every substep / iteration count — through both paths, twice each, and
//! compares the serialised state after every frame.
//!
//! Bit-exact `step` vs `step_parallel` parity is asserted only when the
//! constraint graph makes the two orderings equivalent: no rods and no
//! contact at any point of the frame (every constraint then touches its own
//! bodies only, so the batch order cannot matter).
use alice_physics::{DistanceConstraint, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Body {
    pos: (i8, i8, i8),
    vel: (i8, i8, i8),
    /// 0 → static, otherwise mass = n / 4
    mass_q: u8,
    /// sphere radius = 1 + r / 64 (never zero)
    radius_q: u8,
}

#[derive(Debug, Arbitrary)]
struct Rod {
    a: u8,
    b: u8,
    /// target distance = 1 + d / 16
    dist_q: u8,
    /// compliance = c / 256 (0 = rigid)
    comp_q: u8,
}

#[derive(Debug, Arbitrary)]
struct Scene {
    bodies: Vec<Body>,
    rods: Vec<Rod>,
    substeps: u8,
    iterations: u8,
    frames: u8,
}

fn build(scene: &Scene) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        substeps: (scene.substeps as usize % 8) + 1,
        iterations: (scene.iterations as usize % 4) + 1,
        ..PhysicsConfig::default()
    });
    for b in scene.bodies.iter().take(24) {
        let pos = Vec3Fix::new(
            Fix128::from_ratio(b.pos.0 as i64, 4),
            Fix128::from_ratio(b.pos.1 as i64, 4),
            Fix128::from_ratio(b.pos.2 as i64, 4),
        );
        let radius = Fix128::ONE + Fix128::from_ratio(b.radius_q as i64, 64);
        if b.mass_q == 0 {
            world.add_body_with_radius(RigidBody::new_static(pos), radius);
        } else {
            let mut body = RigidBody::new_dynamic(pos, Fix128::from_ratio(b.mass_q as i64, 4));
            body.velocity = Vec3Fix::new(
                Fix128::from_ratio(b.vel.0 as i64, 8),
                Fix128::from_ratio(b.vel.1 as i64, 8),
                Fix128::from_ratio(b.vel.2 as i64, 8),
            );
            world.add_body_with_radius(body, radius);
        }
    }
    let n = world.bodies.len();
    if n >= 2 {
        for r in scene.rods.iter().take(16) {
            let a = r.a as usize % n;
            let b = r.b as usize % n;
            if a == b {
                continue;
            }
            world.add_distance_constraint(DistanceConstraint {
                body_a: a,
                body_b: b,
                local_anchor_a: Vec3Fix::ZERO,
                local_anchor_b: Vec3Fix::ZERO,
                target_distance: Fix128::ONE + Fix128::from_ratio(r.dist_q as i64, 16),
                compliance: Fix128::from_ratio(r.comp_q as i64, 256),
                cached_lambda: Fix128::ZERO,
            });
        }
    }
    world
}

fuzz_target!(|scene: Scene| {
    if scene.bodies.is_empty() {
        return;
    }
    let dt = Fix128::from_ratio(1, 60);
    let frames = (scene.frames as usize % 16) + 1;

    let mut seq_a = build(&scene);
    let mut seq_b = build(&scene);
    #[cfg(feature = "parallel")]
    let (mut par_a, mut par_b) = (build(&scene), build(&scene));
    let order_free = scene.rods.is_empty() && seq_a.distance_constraints.is_empty();

    for frame in 0..frames {
        seq_a.step(dt);
        seq_b.step(dt);
        let golden = seq_a.serialize_state();
        assert_eq!(golden, seq_b.serialize_state(), "step not deterministic at frame {frame}");
        assert_eq!(golden, seq_a.serialize_state(), "serialize_state not idempotent at frame {frame}");
        #[cfg(feature = "parallel")]
        {
            par_a.step_parallel(dt);
            par_b.step_parallel(dt);
            let par = par_a.serialize_state();
            assert_eq!(
                par,
                par_b.serialize_state(),
                "step_parallel not deterministic at frame {frame} (bodies {}, substeps {}, iterations {})",
                seq_a.bodies.len(),
                seq_a.config.substeps,
                seq_a.config.iterations
            );
            // No rods and no contact seen by either path this frame → the two
            // Gauss–Seidel orderings coincide and the bits must match.
            if order_free
                && seq_a.contact_constraints.is_empty()
                && par_a.contact_constraints.is_empty()
                && seq_a.contact_events().is_empty()
                && par_a.contact_events().is_empty()
            {
                assert_eq!(
                    golden, par,
                    "step vs step_parallel diverged on an order-free scene at frame {frame}"
                );
            }
        }
    }
});
