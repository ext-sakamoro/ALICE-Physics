//! The grid fluid solver gives the same bits on any number of threads.
//!
//! Each scene's state digest is one constant: the serial build (`cargo test`) and the `parallel`
//! build (`cargo test --features parallel`) both assert it, and the parallel build asserts it in
//! rayon pools of 1, 2, 4 and 8 threads. The constants were recorded from the serial build, so a
//! parallel kernel whose result moved with the partition or the thread count fails here.
//!
//! The scenes cross slab edges: the 10 mm smoke scene has 32 × 64 × 32 cells and the BiCGSTAB
//! scene 32³, each several slabs of 4096 elements (`src/par_slabs.rs`), so a kernel that depended on where the
//! slabs end would show it.
//!
//! Why the bits can be equal at all: every parallel kernel is a gather over a fixed partition, and
//! the reductions add `Fix128`s, whose addition is wrapping integer addition (associative and
//! commutative), with no partial rounded before it is combined (`src/par_slabs.rs`, MODEL).

#![cfg(feature = "std")]

use alice_physics_core::cfd_solver::{CfdSolver, PressureSolver};
use alice_physics_core::world_participant::Participant;
use alice_physics_core::Fix128;
use sha2::{Digest, Sha256};

#[path = "support/smoke_laminar_scene.rs"]
mod laminar_scene;

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The 10 mm laminar plume scene after three steps: buoyancy, BFECC advection and the multigrid
/// projection, then the integer heat transport. Its digest is the push-time guard's
/// (`tests/analytic_smoke_plume.rs`).
fn smoke_digest() -> String {
    let mut p = laminar_scene::scene(10, 200, 100);
    for _ in 0..3 {
        p.advance(laminar_scene::step(10)).unwrap();
    }
    let mut state = Vec::new();
    p.write_state(&mut state);
    hex(&state)
}

const SMOKE_DIGEST: &str = "7221c60fb6bdf5341a6e9bc5bd662833b48bd66ab9cc239638ca566d1d2e7c02";

/// A 32³ box of water with a swirling, divergent start, advanced four steps with the BiCGSTAB
/// projection, whose inner products are the solver's reductions. The digest covers every face
/// velocity and the pressure.
fn bicgstab_digest() -> String {
    let n = 32;
    let mut s = CfdSolver::new(n, n, n, Fix128::from_ratio(1, 100));
    // a deterministic, non-uniform start: each face velocity from its index
    let seed = |m: usize, salt: i64| {
        let x = (m as i64 * 2_654_435_761 + salt) % 2001 - 1000;
        Fix128::from_ratio(x, 10_000)
    };
    for (m, u) in s.grid.u.iter_mut().enumerate() {
        *u = seed(m, 11);
    }
    for (m, v) in s.grid.v.iter_mut().enumerate() {
        *v = seed(m, 23);
    }
    for (m, w) in s.grid.w.iter_mut().enumerate() {
        *w = seed(m, 37);
    }
    let solver = PressureSolver::BiCgStab {
        max_iterations: 30,
        tolerance: Fix128::from_ratio(1, 1_000_000_000),
    };
    for _ in 0..4 {
        s.step_with_pressure_solver(Fix128::from_ratio(1, 200), solver)
            .unwrap();
    }
    let mut bytes = Vec::new();
    for x in s
        .grid
        .u
        .iter()
        .chain(&s.grid.v)
        .chain(&s.grid.w)
        .chain(&s.grid.pressure)
    {
        bytes.extend_from_slice(&x.hi.to_le_bytes());
        bytes.extend_from_slice(&x.lo.to_le_bytes());
    }
    hex(&bytes)
}

const BICGSTAB_DIGEST: &str = "6767887a93d330a3943669027a40e96ec1f00733b0cd71f88cd78dd5220fec03";

/// Serial build or the default pool: each scene gives its recorded digest.
#[test]
fn the_fluid_scenes_give_their_recorded_digests() {
    assert_eq!(smoke_digest(), SMOKE_DIGEST, "smoke scene");
    assert_eq!(bicgstab_digest(), BICGSTAB_DIGEST, "BiCGSTAB scene");
}

/// Pools of 1, 2, 4 and 8 threads give the serial build's digests. The pool really has that many
/// threads (asserted inside it), whatever the machine's core count.
#[cfg(feature = "parallel")]
#[test]
fn every_thread_count_gives_the_serial_digests() {
    for threads in [1usize, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (n, smoke, bicgstab) = pool.install(|| {
            (
                rayon::current_num_threads(),
                smoke_digest(),
                bicgstab_digest(),
            )
        });
        assert_eq!(n, threads, "pool size");
        assert_eq!(smoke, SMOKE_DIGEST, "smoke scene on {threads} threads");
        assert_eq!(
            bicgstab, BICGSTAB_DIGEST,
            "BiCGSTAB scene on {threads} threads"
        );
    }
}
