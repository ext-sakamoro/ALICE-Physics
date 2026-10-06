//! Wall-time measurement harness for the `step_parallel` investigation.
//!
//! Scenario matches `benches/physics_bench.rs`'s
//! `thousand_overlapping_spheres_1_step` (10x10x10 grid, spacing 1.5,
//! radius 1, every neighbour pair overlapping) with
//! `Broadphase::DynamicTree`, reproducing the configuration measured as
//! step 23.6 ms vs step_parallel 22.8 ms at 1000 bodies / 8 substeps.
//!
//! Thread count is read from `RAYON_NUM_THREADS` by rayon itself (not set
//! here), so the same binary is invoked once per thread count from the
//! shell to get independent process measurements.
//!
//! Run with (release, parallel feature required for the step_parallel mode):
//!   cargo run --release --features parallel --example step_parallel_profile -- <mode> <substeps> <frames>
//!   mode: "step" | "step_parallel"
//!
//! Prints a single line: `TOTAL_MS <f64> FRAMES <usize> SUBSTEPS <u32>`

use std::time::Instant;

use alice_physics::solver::Broadphase;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn build_world(substeps: usize) -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_broadphase(Broadphase::DynamicTree);
    for x in 0..10i64 {
        for y in 0..10i64 {
            for z in 0..10i64 {
                let pos = Vec3Fix::new(
                    Fix128::from_ratio(3 * x, 2),
                    Fix128::from_ratio(3 * y, 2) + Fix128::from_int(20),
                    Fix128::from_ratio(3 * z, 2),
                );
                world.add_body_with_radius(RigidBody::new_dynamic(pos, Fix128::ONE), Fix128::ONE);
            }
        }
    }
    world
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("step");
    let substeps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let frames: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);

    let mut world = build_world(substeps);
    let dt = Fix128::from_ratio(1, 60);

    let start = Instant::now();
    match mode {
        "step" => {
            for _ in 0..frames {
                world.step(dt);
            }
        }
        #[cfg(feature = "parallel")]
        "step_parallel" => {
            for _ in 0..frames {
                world.step_parallel(dt);
            }
        }
        #[cfg(not(feature = "parallel"))]
        "step_parallel" => {
            eprintln!("step_parallel mode requires --features parallel");
            std::process::exit(1);
        }
        other => {
            eprintln!("unknown mode: {other} (expected step|step_parallel)");
            std::process::exit(1);
        }
    }
    let elapsed = start.elapsed();

    // Keep the final state live so the compiler cannot elide the loop.
    std::hint::black_box(&world.bodies[0].position);

    println!(
        "TOTAL_MS {:.3} FRAMES {} SUBSTEPS {}",
        elapsed.as_secs_f64() * 1000.0,
        frames,
        substeps
    );
}
