//! Impact-driven fracture: `FractureModifier::{apply_stress_at, stress_at, active_crack_count}`.
//!
//! A 5^3 stress grid over `[0, 4]^3` (cell size 1) takes a 120 unit impact at its centre node.
//! The smoothstep splat of radius 1.5 puts `A` on the centre node and `A * 7/27` on its six face
//! neighbours (distance 1: `t = 1 - 1/1.5 = 1/3`, `t^2 (3 - 2t) = 7/27`). With toughness 50 only
//! the centre node (120) is above threshold, so exactly one crack is seeded. It grows at
//! `propagation_speed * dt` per step until `max_crack_length`, then `active_crack_count` drops to 0
//! while the crack stays in the SDF.
//!
//! ```bash
//! cargo run --release --example fracture_impact --features std
//! ```

use alice_physics::fracture::{FractureConfig, FractureModifier};
use alice_physics::sim_modifier::PhysicsModifier;

fn main() {
    let config = FractureConfig {
        stress_diffusion: 0.0,
        stress_decay: 0.0,
        propagation_speed: 2.0,
        max_crack_length: 1.0,
        ..FractureConfig::default()
    };
    let mut m = FractureModifier::new(config, 5, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0));
    m.apply_stress_at(2.0, 2.0, 2.0, 120.0, 1.5);
    println!(
        "stress: centre {:.3}, face neighbour {:.3} (expect {:.3})",
        m.stress_at(2.0, 2.0, 2.0),
        m.stress_at(3.0, 2.0, 2.0),
        120.0 * 7.0 / 27.0
    );
    for step in 0..6 {
        m.update(0.25);
        let len = m.cracks.first().map_or(0.0, |c| c.length);
        println!(
            "step {step}: active cracks {}, total {}, length {len:.2}",
            m.active_crack_count(),
            m.cracks.len()
        );
    }
    // the crack stays carved into the surface after it stops growing
    let c = m.cracks[0];
    let mid = (
        c.start.0 + 0.5 * (c.end.0 - c.start.0),
        c.start.1 + 0.5 * (c.end.1 - c.start.1),
        c.start.2 + 0.5 * (c.end.2 - c.start.2),
    );
    println!(
        "distance on the crack axis: {:.3} (was -1.000, receded to crack_width {:.3})",
        m.modify_distance(mid.0, mid.1, mid.2, -1.0),
        m.config.crack_width
    );
}
