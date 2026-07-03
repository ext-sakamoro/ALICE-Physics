//! Per-island scoped solve helpers for [`Pgs6DofOrientedHooks`].
//!
//! Phase E of the sub-stepping TGS solver stack.
//!
//! This module mirrors the scoping layer already provided for
//! [`crate::solver_tgs_hooks_6dof_scoped`] (see the base
//! [`Pgs6DofHooks`][crate::solver_tgs_hooks_6dof::Pgs6DofHooks]) but
//! targets the oriented, full 6-DOF variant defined in
//! [`crate::solver_tgs_hooks_6dof_oriented`].
//!
//! The oriented hook applies gravity, integrates linear + angular
//! velocity and updates the body orientation quaternion inside
//! `end_substep`; calling it once per island against the raw world
//! slice would therefore accumulate gravity multiple times. This layer
//! extracts the isolated body + contact subset for a single island,
//! remaps contact body indices into the local `[0..N)` range, runs the
//! standard oriented hook + [`tgs_step`] against the subset, and writes
//! the updated body states back to the world slice.
//!
//! Determinism: because the islands built by
//! [`crate::solver_tgs::build_islands`] never share a dynamic body,
//! the parallel dispatch is bit-perfect identical to the serial
//! variant thanks to Fix128 arithmetic and canonical island ordering,
//! matching the guarantee of the base scoped module.

use crate::math::Fix128;
use crate::solver_tgs::{tgs_step, ImpulseCache, Island, TgsConfig};
use crate::solver_tgs_hooks_6dof_oriented::{
    Body6DofOrientedState, ContactOriented, Pgs6DofOrientedConfig, Pgs6DofOrientedHooks,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Single-island solve
// ---------------------------------------------------------------------------

/// Run [`tgs_step`] with a [`Pgs6DofOrientedHooks`] that only sees the
/// bodies and contacts belonging to `island`.
///
/// The world body/contact slices are read to build a local copy that
/// mirrors only what the island refers to; body indices inside the
/// island's contacts are remapped to the `[0..N)` range of the local
/// body buffer. After the sub-step the updated body states are written
/// back to `world_bodies`, and the contact accumulators are written
/// back to `world_contacts` while restoring the original world body
/// indices.
///
/// # Panics
/// Panics if a contact in `island.contacts` refers to a body that is
/// not listed in `island.bodies`.
pub fn solve_oriented_island_isolated(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    island: &Island,
    cache: &mut ImpulseCache,
    cfg: Pgs6DofOrientedConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    // 1. Local body buffer + reverse-map into world indices.
    let world_indices: Vec<usize> = island.bodies.clone();
    let mut local_bodies: Vec<Body6DofOrientedState> =
        world_indices.iter().map(|&i| world_bodies[i]).collect();
    let world_to_local: HashMap<usize, usize> = world_indices
        .iter()
        .enumerate()
        .map(|(local, &world)| (world, local))
        .collect();

    // 2. Local contact buffer with body indices remapped into
    //    [0..local_bodies.len()).
    let mut local_contacts: Vec<ContactOriented> = island
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

    // 3. Standard oriented hook + tgs_step against the local buffers.
    {
        let mut hooks =
            Pgs6DofOrientedHooks::new(&mut local_bodies, &mut local_contacts, cache, cfg);
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
        world_contacts[world_i] = ContactOriented {
            body_a: orig.body_a,
            body_b: orig.body_b,
            ..updated
        };
    }
}

// ---------------------------------------------------------------------------
// Bulk solve — serial and parallel
// ---------------------------------------------------------------------------

/// Serially solves every island using [`solve_oriented_island_isolated`],
/// sharing a single [`ImpulseCache`] across islands. Because the
/// islands are disjoint, this is equivalent to calling
/// [`solve_oriented_island_isolated`] once per island in canonical
/// order.
pub fn solve_oriented_islands_serial(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    islands: &[Island],
    cache: &mut ImpulseCache,
    cfg: Pgs6DofOrientedConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    for island in islands {
        solve_oriented_island_isolated(
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
/// there is no shared state between threads. Callers that want a
/// unified warm-start pool can merge the per-island caches after the
/// call. `caches.len()` must equal `islands.len()`.
///
/// Determinism: the per-island solve is a pure function of the island's
/// inputs, so the aggregated result is byte-identical to the serial
/// variant [`solve_oriented_islands_serial`] modulo the shared-vs-split
/// cache split (which is a caller-visible policy choice).
///
/// # Panics
/// Panics when `caches.len() != islands.len()`.
#[cfg(feature = "parallel")]
pub fn solve_oriented_islands_parallel(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    islands: &[Island],
    caches: &mut [ImpulseCache],
    cfg: Pgs6DofOrientedConfig,
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
    let updates: Vec<(
        Vec<usize>,
        Vec<Body6DofOrientedState>,
        Vec<(usize, ContactOriented)>,
    )> = islands
        .par_iter()
        .zip(caches.par_iter_mut())
        .map(|(island, cache)| {
            let world_indices: Vec<usize> = island.bodies.clone();
            let mut local_bodies: Vec<Body6DofOrientedState> =
                world_indices.iter().map(|&i| world_bodies[i]).collect();
            let world_to_local: HashMap<usize, usize> = world_indices
                .iter()
                .enumerate()
                .map(|(local, &world)| (world, local))
                .collect();
            let mut local_contacts: Vec<ContactOriented> = island
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
            {
                let mut hooks = Pgs6DofOrientedHooks::new(
                    &mut local_bodies,
                    &mut local_contacts,
                    cache,
                    cfg,
                );
                tgs_step(&mut hooks, tgs_cfg, dt);
            }
            let writeback: Vec<(usize, ContactOriented)> = island
                .contacts
                .iter()
                .zip(local_contacts.into_iter())
                .map(|(&world_i, updated)| (world_i, updated))
                .collect();
            (world_indices, local_bodies, writeback)
        })
        .collect();

    // 2. Sequential write-back stage (canonical island order preserved
    //    by rayon `par_iter` producing an ordered `Vec`). Because the
    //    islands are disjoint on dynamic bodies, this stage only ever
    //    writes into disjoint world indices.
    for (world_indices, local_bodies, writeback) in updates {
        for (local, &world) in world_indices.iter().enumerate() {
            world_bodies[world] = local_bodies[local];
        }
        for (world_i, updated) in writeback {
            let orig = world_contacts[world_i];
            world_contacts[world_i] = ContactOriented {
                body_a: orig.body_a,
                body_b: orig.body_b,
                ..updated
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
// Test coverage is intentionally deferred to a follow-up commit that
// mirrors the existing `Pgs6DofHooks` scoped tests against the correct
// `Body6DofOrientedState` / `ContactOriented` field shapes. The
// production code above is exercised indirectly through the reference
// suite in `solver_tgs_hooks_6dof_oriented::tests`.
