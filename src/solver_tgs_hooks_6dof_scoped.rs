//! Per-island scoped solve helpers for [`Pgs6DofHooks`].
//!
//! The base [`crate::solver_tgs_hooks_6dof::Pgs6DofHooks`] takes a
//! mutable slice covering every body in the world and applies gravity
//! and impulses across the whole slice. Calling it once per island
//! would therefore accumulate gravity multiple times.
//!
//! This module bridges that gap by extracting an *isolated* body /
//! contact subset for a given island, remapping contact body indices
//! into the local range, running the standard hook + [`tgs_step`]
//! against the subset, and finally writing the updated body states
//! back to the world slice.
//!
//! Because the two islands built by
//! [`crate::solver_tgs::build_islands`] never share a dynamic body,
//! parallel dispatch is safe: each island scribbles into its own
//! local buffer, and the write-back stage touches disjoint world
//! indices.

// (missing_docs allow scoped to this module during Turn E follow-up; see lib.rs.)
#![allow(missing_docs)]

use crate::math::Fix128;
use crate::solver_tgs::{tgs_step, ImpulseCache, Island, TgsConfig};
use crate::solver_tgs_hooks_6dof::{Body6DofState, Contact6Dof, Pgs6DofConfig, Pgs6DofHooks};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Single-island solve
// ---------------------------------------------------------------------------

/// Run [`tgs_step`] with a hook that only sees the bodies and contacts
/// belonging to `island`.
///
/// The world body/contact slices are read to build a local copy that
/// mirrors only what the island refers to; body indices inside the
/// island's contacts are remapped to the [0..N) range of the local
/// body buffer. After the sub-step the updated body states are
/// written back to `world_bodies` and the contact accumulators are
/// written back to `world_contacts`.
///
/// # Panics
/// Panics if a contact in `island.contacts` refers to a body that is
/// not listed in `island.bodies`.
pub fn solve_island_isolated(
    world_bodies: &mut [Body6DofState],
    world_contacts: &mut [Contact6Dof],
    island: &Island,
    cache: &mut ImpulseCache,
    cfg: Pgs6DofConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    // 1. Local body buffer + reverse-map into world indices.
    let world_indices: Vec<usize> = island.bodies.clone();
    let mut local_bodies: Vec<Body6DofState> =
        world_indices.iter().map(|&i| world_bodies[i]).collect();
    let world_to_local: HashMap<usize, usize> = world_indices
        .iter()
        .enumerate()
        .map(|(local, &world)| (world, local))
        .collect();

    // 2. Local contact buffer with body indices remapped into
    //    [0..local_bodies.len()).
    let mut local_contacts: Vec<Contact6Dof> = island
        .contacts
        .iter()
        .map(|&ci| {
            let mut c = world_contacts[ci];
            c.body_a = *world_to_local
                .get(&c.body_a)
                .expect("contact body_a not in island");
            c.body_b = *world_to_local
                .get(&c.body_b)
                .expect("contact body_b not in island");
            c
        })
        .collect();

    // 3. Standard hook + tgs_step against the local buffers.
    {
        let mut hooks = Pgs6DofHooks::new(&mut local_bodies, &mut local_contacts, cache, cfg);
        tgs_step(&mut hooks, tgs_cfg, dt);
    }

    // 4. Write body updates back to the world slice at the correct
    //    world indices (dynamic and static alike; the hook already
    //    skipped movement for the static ones).
    for (local, &world) in world_indices.iter().enumerate() {
        world_bodies[world] = local_bodies[local];
    }

    // 5. Write contact accumulators back to the world slice, but
    //    restore the original world body indices so downstream code
    //    keeps pointing at the world bodies.
    for (local_i, &world_i) in island.contacts.iter().enumerate() {
        let orig = world_contacts[world_i];
        let updated = local_contacts[local_i];
        world_contacts[world_i] = Contact6Dof {
            body_a: orig.body_a,
            body_b: orig.body_b,
            ..updated
        };
    }
}

// ---------------------------------------------------------------------------
// Bulk solve — serial and parallel
// ---------------------------------------------------------------------------

/// Serially solves every island using [`solve_island_isolated`],
/// sharing a single [`ImpulseCache`] across islands. Because the
/// islands are disjoint, this is equivalent to calling
/// [`solve_island_isolated`] once per island in
/// [`Island`]-canonical order.
pub fn solve_islands_serial(
    world_bodies: &mut [Body6DofState],
    world_contacts: &mut [Contact6Dof],
    islands: &[Island],
    cache: &mut ImpulseCache,
    cfg: Pgs6DofConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    for island in islands {
        solve_island_isolated(
            world_bodies,
            world_contacts,
            island,
            cache,
            cfg,
            tgs_cfg,
            dt,
        );
    }
}

/// Solves every island in parallel via `rayon`.
///
/// Each island receives its own dedicated [`ImpulseCache`] so that
/// there is no shared state between threads; callers that want a
/// unified warm-start pool can merge the per-island caches after the
/// call. `caches.len()` must equal `islands.len()`.
///
/// Determinism: the per-island solve is a pure function of the
/// island's inputs, so the aggregated result is byte-identical to the
/// serial variant [`solve_islands_serial`] modulo the shared-vs-split
/// cache split (which is a caller-visible policy choice).
///
/// # Panics
/// Panics when `caches.len() != islands.len()`.
#[cfg(feature = "parallel")]
pub fn solve_islands_parallel(
    world_bodies: &mut [Body6DofState],
    world_contacts: &mut [Contact6Dof],
    islands: &[Island],
    caches: &mut [ImpulseCache],
    cfg: Pgs6DofConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    use rayon::prelude::*;
    assert_eq!(
        caches.len(),
        islands.len(),
        "one cache per island is required"
    );

    // 1. Parallel-solve into local buffers. Each thread produces
    //    (world_indices, updated_local_bodies, contact_writeback[]).
    let updates: Vec<(Vec<usize>, Vec<Body6DofState>, Vec<(usize, Contact6Dof)>)> = islands
        .par_iter()
        .zip(caches.par_iter_mut())
        .map(|(island, cache)| {
            let world_indices: Vec<usize> = island.bodies.clone();
            let mut local_bodies: Vec<Body6DofState> =
                world_indices.iter().map(|&i| world_bodies[i]).collect();
            let world_to_local: HashMap<usize, usize> = world_indices
                .iter()
                .enumerate()
                .map(|(local, &world)| (world, local))
                .collect();
            let mut local_contacts: Vec<Contact6Dof> = island
                .contacts
                .iter()
                .map(|&ci| {
                    let mut c = world_contacts[ci];
                    c.body_a = *world_to_local.get(&c.body_a).unwrap();
                    c.body_b = *world_to_local.get(&c.body_b).unwrap();
                    c
                })
                .collect();
            {
                let mut hooks =
                    Pgs6DofHooks::new(&mut local_bodies, &mut local_contacts, cache, cfg);
                tgs_step(&mut hooks, tgs_cfg, dt);
            }
            let writeback: Vec<(usize, Contact6Dof)> = island
                .contacts
                .iter()
                .zip(local_contacts.into_iter())
                .map(|(&world_i, updated)| {
                    let orig = world_contacts[world_i];
                    (
                        world_i,
                        Contact6Dof {
                            body_a: orig.body_a,
                            body_b: orig.body_b,
                            ..updated
                        },
                    )
                })
                .collect();
            (world_indices, local_bodies, writeback)
        })
        .collect();

    // 2. Serial write-back over disjoint world indices — safe because
    //    islands share no dynamic body.
    for (indices, bodies, contacts) in updates {
        for (local, &world) in indices.iter().enumerate() {
            world_bodies[world] = bodies[local];
        }
        for (world_i, contact) in contacts {
            world_contacts[world_i] = contact;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver_tgs::{build_islands, JointLike};

    struct NoJoint;
    impl JointLike for NoJoint {
        fn body_a(&self) -> usize {
            0
        }
        fn body_b(&self) -> usize {
            0
        }
    }

    fn ground(id: u64) -> Body6DofState {
        Body6DofState {
            is_dynamic: false,
            stable_id: id,
            ..Default::default()
        }
    }
    fn free_body(id: u64, inv_m: f32, inv_i: f32, y: f32) -> Body6DofState {
        Body6DofState {
            position: [Fix128::ZERO, Fix128::from_f32(y), Fix128::ZERO],
            inv_mass: Fix128::from_f32(inv_m),
            inv_inertia: [
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
                Fix128::from_f32(inv_i),
            ],
            is_dynamic: true,
            stable_id: id,
            ..Default::default()
        }
    }
    fn floor_contact(a: usize, b: usize, id: u64, mu: f32, pen: f32) -> Contact6Dof {
        Contact6Dof {
            body_a: a,
            body_b: b,
            stable_id: id,
            normal: [Fix128::ZERO, Fix128::ONE, Fix128::ZERO],
            tangent1: [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            tangent2: [Fix128::ZERO, Fix128::ZERO, Fix128::ONE],
            r_a: [Fix128::ZERO; 3],
            r_b: [Fix128::ZERO; 3],
            penetration: Fix128::from_f32(pen),
            friction: Fix128::from_f32(mu),
            restitution: Fix128::ZERO,
            accum_normal: Fix128::ZERO,
            accum_tangent1: Fix128::ZERO,
            accum_tangent2: Fix128::ZERO,
        }
    }

    fn build_two_disjoint_stacks() -> ([Body6DofState; 3], [Contact6Dof; 2]) {
        (
            [
                ground(1),
                free_body(2, 1.0, 1.0, 0.01),
                free_body(3, 1.0, 1.0, 0.01),
            ],
            [
                floor_contact(0, 1, 900, 0.4, 0.01),
                floor_contact(0, 2, 901, 0.4, 0.01),
            ],
        )
    }

    #[test]
    fn scoped_single_island_advances_only_its_body() {
        let (mut bodies, mut contacts) = build_two_disjoint_stacks();
        let islands = build_islands(&bodies, &contacts, &[] as &[NoJoint]);
        assert_eq!(islands.len(), 2);
        let cfg = Pgs6DofConfig {
            warmstart: false,
            ..Pgs6DofConfig::default()
        };
        let tcfg = TgsConfig::default();
        let dt = Fix128::from_f32(1.0 / 60.0);
        let mut cache = ImpulseCache::new();

        // The `islands[0]` case owns body index 1 (or 2). Whichever it is,
        // the sibling dynamic body must remain untouched.
        let sibling = *islands[0].bodies.iter().find(|&&b| b != 0).unwrap();
        let untouched = if sibling == 1 { 2 } else { 1 };
        let snapshot = bodies[untouched];

        solve_island_isolated(
            &mut bodies,
            &mut contacts,
            &islands[0],
            &mut cache,
            cfg,
            &tcfg,
            dt,
        );

        assert_eq!(
            bodies[untouched].position, snapshot.position,
            "sibling body must not have moved when only one island was solved"
        );
        assert_eq!(bodies[untouched].linear_velocity, snapshot.linear_velocity);
    }

    #[test]
    fn scoped_serial_dispatch_matches_all_at_once_after_scoping() {
        // Reference: `solve_islands_serial` over both islands.
        let (mut ref_bodies, mut ref_contacts) = build_two_disjoint_stacks();
        let islands = build_islands(&ref_bodies, &ref_contacts, &[] as &[NoJoint]);
        let cfg = Pgs6DofConfig {
            warmstart: false,
            ..Pgs6DofConfig::default()
        };
        let tcfg = TgsConfig::default();
        let dt = Fix128::from_f32(1.0 / 60.0);
        let mut ref_cache = ImpulseCache::new();
        solve_islands_serial(
            &mut ref_bodies,
            &mut ref_contacts,
            &islands,
            &mut ref_cache,
            cfg,
            &tcfg,
            dt,
        );

        // Direct call chain of `solve_island_isolated` in the same
        // canonical order must be byte-identical.
        let (mut direct_bodies, mut direct_contacts) = build_two_disjoint_stacks();
        let mut direct_cache = ImpulseCache::new();
        for island in &islands {
            solve_island_isolated(
                &mut direct_bodies,
                &mut direct_contacts,
                island,
                &mut direct_cache,
                cfg,
                &tcfg,
                dt,
            );
        }
        for (a, b) in ref_bodies.iter().zip(direct_bodies.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.linear_velocity, b.linear_velocity);
            assert_eq!(a.angular_velocity, b.angular_velocity);
        }
    }

    #[test]
    fn scoped_solve_is_bit_perfect_deterministic() {
        // Same input, twice, must give byte-identical output.
        fn run() -> [Body6DofState; 3] {
            let (mut bodies, mut contacts) = build_two_disjoint_stacks();
            let islands = build_islands(&bodies, &contacts, &[] as &[NoJoint]);
            let cfg = Pgs6DofConfig::default();
            let tcfg = TgsConfig::default();
            let dt = Fix128::from_f32(1.0 / 60.0);
            let mut cache = ImpulseCache::new();
            for _ in 0..4 {
                solve_islands_serial(
                    &mut bodies,
                    &mut contacts,
                    &islands,
                    &mut cache,
                    cfg,
                    &tcfg,
                    dt,
                );
            }
            bodies
        }
        let a = run();
        let b = run();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.position, y.position);
            assert_eq!(x.linear_velocity, y.linear_velocity);
            assert_eq!(x.angular_velocity, y.angular_velocity);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial_bit_perfect() {
        // rayon parallel dispatch must give the same result as the
        // serial fallback thanks to island disjointness + Fix128
        // determinism.
        let (mut serial_bodies, mut serial_contacts) = build_two_disjoint_stacks();
        let (mut par_bodies, mut par_contacts) = build_two_disjoint_stacks();
        let islands = build_islands(&serial_bodies, &serial_contacts, &[] as &[NoJoint]);
        assert_eq!(islands.len(), 2);
        let cfg = Pgs6DofConfig::default();
        let tcfg = TgsConfig::default();
        let dt = Fix128::from_f32(1.0 / 60.0);

        let mut serial_cache = ImpulseCache::new();
        solve_islands_serial(
            &mut serial_bodies,
            &mut serial_contacts,
            &islands,
            &mut serial_cache,
            cfg,
            &tcfg,
            dt,
        );

        // Each island gets its own cache — the parallel API's policy.
        let mut par_caches: Vec<ImpulseCache> =
            (0..islands.len()).map(|_| ImpulseCache::new()).collect();
        solve_islands_parallel(
            &mut par_bodies,
            &mut par_contacts,
            &islands,
            &mut par_caches,
            cfg,
            &tcfg,
            dt,
        );

        for (s, p) in serial_bodies.iter().zip(par_bodies.iter()) {
            assert_eq!(s.position, p.position, "position drift in parallel path");
            assert_eq!(s.linear_velocity, p.linear_velocity);
            assert_eq!(s.angular_velocity, p.angular_velocity);
        }
    }
}
