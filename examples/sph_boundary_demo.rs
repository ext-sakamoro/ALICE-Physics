//! SDF-boundary SPH fluid demo — drops a cube of 512 water particles
//! inside a hemispherical bowl and compares the naive `O(N²)`
//! neighbour search (`step`) against the spatial-hash-accelerated
//! `O(N·k)` variant (`step_hashed`).
//!
//! Illustrates the v0.13.0 Session 4 (S3 Tier ★★) and
//! v0.14.0-preview.1 (`SphSpatialHash` / `step_hashed`) additions.
//!
//! ```bash
//! cargo run --example sph_boundary_demo --features std --release
//! ```

use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_sph::{SphConfig, SphParticle, SphSolver};
use std::time::Instant;

const GRID: usize = 8; // 8³ = 512 particles

/// Build a hemispherical bowl SDF at the origin, radius = 0.6 m.
/// Inside the bowl (below y = 0, above the hemisphere): negative distance.
/// A container geometry that keeps fluid inside.
fn hemisphere_bowl() -> ClosureSdf {
    let radius: f32 = 0.6;
    ClosureSdf::new(
        move |x, y, z| {
            // Distance to sphere of radius `radius`, negated so inside is positive
            let d_sphere = radius - (x * x + y * y + z * z).sqrt();
            // Cap the top at y = 0 (only the lower half is a container)
            d_sphere.min(-y)
        },
        move |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1e-6);
            (-x / len, -y / len, -z / len)
        },
    )
}

/// Build an 8×8×8 = 512-particle water block at (0, -0.3, 0) with 0.03 m spacing.
fn seed_particles() -> Vec<SphParticle> {
    let spacing = 0.03_f32;
    let base_x = -(GRID as f32 * spacing) * 0.5;
    let base_y = -0.45_f32;
    let base_z = base_x;
    let mut particles = Vec::with_capacity(GRID * GRID * GRID);
    for k in 0..GRID {
        for j in 0..GRID {
            for i in 0..GRID {
                particles.push(SphParticle::at_rest([
                    base_x + i as f32 * spacing,
                    base_y + j as f32 * spacing,
                    base_z + k as f32 * spacing,
                ]));
            }
        }
    }
    particles
}

fn simulate<F>(label: &str, mut solver_step: F) -> f32
where
    F: FnMut(f32),
{
    let start = Instant::now();
    let dt = 1.0 / 240.0;
    for _ in 0..60 {
        solver_step(dt);
    }
    let elapsed = start.elapsed().as_secs_f32();
    println!(
        "{:>20} : {:>7.2} ms for 60 steps × 512 particles",
        label,
        elapsed * 1000.0
    );
    elapsed
}

fn main() {
    let bowl = hemisphere_bowl();
    let config = SphConfig::water_like();

    println!("ALICE-Physics — SDF-Boundary SPH Fluid Demo");
    println!("===========================================");
    println!("Particles:   {} (water_like preset)", GRID * GRID * GRID);
    println!("Boundary:    hemispherical bowl, r = 0.6 m at origin");
    println!("Kernel h:    {} m", config.kernel_radius);
    println!("Time step:   1/240 s = 4.17 ms");
    println!("Total steps: 60 (≈ 0.25 s simulation)\n");

    // Naive O(N²) neighbour search.
    let particles_naive = seed_particles();
    let mut solver_naive = SphSolver::new(particles_naive, config, &bowl);
    let t_naive = simulate("O(N²) step", |dt| solver_naive.step(dt));

    // Spatial-hash-accelerated O(N·k).
    let particles_hashed = seed_particles();
    let mut solver_hashed = SphSolver::new(particles_hashed, config, &bowl);
    let t_hashed = simulate("O(N·k) step_hashed", |dt| solver_hashed.step_hashed(dt));

    let speedup = t_naive / t_hashed;
    println!("\nSpeedup (naive / hashed): {:.2}×", speedup);
    println!(
        "Final particle count check: naive = {} / hashed = {}",
        solver_naive.particles.len(),
        solver_hashed.particles.len()
    );

    // Report average density (should be near rest_density = 1000 kg/m³ after
    // enough steps, though 60 steps is short for equilibration).
    let avg_naive: f32 = solver_naive
        .particles
        .iter()
        .map(|p| p.density)
        .sum::<f32>()
        / solver_naive.particles.len() as f32;
    let avg_hashed: f32 = solver_hashed
        .particles
        .iter()
        .map(|p| p.density)
        .sum::<f32>()
        / solver_hashed.particles.len() as f32;
    println!(
        "Average density (rest = 1000 kg/m³): naive = {:.1} / hashed = {:.1}",
        avg_naive, avg_hashed
    );
}
