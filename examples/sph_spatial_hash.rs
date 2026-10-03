//! SPH spatial hash: cell occupancy and neighbourhood queries.
//!
//! Particles on a 4 x 4 x 4 lattice of spacing 0.5 h (h = kernel radius 0.1)
//! fall two per axis into each hash cell, so 4^3 / 2^3 = 8 cells are populated
//! (cell size == h). The 3 x 3 x 3 neighbourhood of an interior particle covers
//! every particle within h of it, checked against a brute-force distance scan.
//!
//! ```bash
//! cargo run --release --example sph_spatial_hash --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_sph::{SphParticle, SphSpatialHash};

fn main() {
    let h = 0.1_f32;
    let spacing = 0.5 * h;
    let mut ps = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                // +0.01 keeps lattice points off the cell boundaries.
                ps.push(SphParticle::at_rest([
                    i as f32 * spacing + 0.01,
                    j as f32 * spacing + 0.01,
                    k as f32 * spacing + 0.01,
                ]));
            }
        }
    }
    let hash = SphSpatialHash::build(&ps, h);
    println!(
        "[sph_hash] cell size {} populated cells {}",
        hash.cell_size(),
        hash.populated_cell_count()
    );
    assert_eq!(hash.cell_size(), h);
    assert_eq!(hash.populated_cell_count(), 8);

    let probe = ps[21].position; // lattice (1, 1, 1)
    let mut visited = vec![false; ps.len()];
    hash.for_each_neighbour(probe, |j| visited[j] = true);
    let mut within = 0;
    for (j, p) in ps.iter().enumerate() {
        let d2: f32 = (0..3).map(|a| (p.position[a] - probe[a]).powi(2)).sum();
        if d2 < h * h {
            within += 1;
            assert!(
                visited[j],
                "particle {j} inside the kernel sphere was not visited"
            );
        }
    }
    println!(
        "[sph_hash] visited {} of {} particles, {} inside the kernel sphere",
        visited.iter().filter(|v| **v).count(),
        ps.len(),
        within
    );
}
