//! Cloth Simulation Example
//!
//! Demonstrates creating a cloth grid with pinned top row,
//! simulating drape under gravity.
//!
//! ```bash
//! cargo run --example cloth_simulation --features std
//! ```

use alice_physics::cloth::{Cloth, ClothConfig};
use alice_physics::math::{Fix128, Vec3Fix};

fn main() {
    // Create a 2m x 2m cloth grid with 10x10 particles
    let mut cloth = Cloth::new_grid(
        Vec3Fix::new(
            Fix128::from_int(-1),
            Fix128::from_int(5),
            Fix128::from_int(-1),
        ),
        Fix128::from_int(2),        // width
        Fix128::from_int(2),        // height
        10,                         // res_x
        10,                         // res_y
        Fix128::from_ratio(1, 100), // mass per particle
    );

    // Pin the top row (curtain-like behavior)
    cloth.pin_top_row(10);

    // Configure
    cloth.config = ClothConfig {
        iterations: 8,
        substeps: 4,
        ..ClothConfig::default()
    };

    // Add some wind
    cloth.wind = Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ONE);

    println!("ALICE-Physics Cloth Simulation");
    println!("==============================");
    println!("Particles: {}", cloth.particle_count());
    println!("Pinned top row: 10 particles");
    println!();

    // Simulate 3 seconds at 60 FPS
    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..180 {
        cloth.step(dt);

        if frame % 30 == 0 {
            // Show position of bottom-center particle
            let mid_bottom = 9 * 10 + 5; // row 9, col 5
            let pos = cloth.positions[mid_bottom];
            println!(
                "Frame {:3}: bottom-center y={:.4}, z={:.4}",
                frame,
                pos.y.to_f64(),
                pos.z.to_f64(),
            );
        }
    }

    let normals = cloth.compute_normals();
    println!();
    println!("Simulation complete (180 frames, 3 seconds).");
    println!("Final normals computed: {} vertices", normals.len());
    println!("Debug: {:?}", cloth);
}
