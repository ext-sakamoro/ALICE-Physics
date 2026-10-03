//! Rope-to-rigid-body attachments: `RopeAttachment::compliance` (builder) and
//! `solve_rope_attachments`, driven together with `Rope::step`.
//!
//! A rope hangs from a static anchor body rotated 180 degrees about Z. The
//! anchor's local point `(1, 1, 0)` maps to world `body.position + (-1, -1, 0)`.
//! A rigid attachment (compliance 0) drops the particle exactly on that point; a
//! compliant one closes only `1 / (1 + compliance / dt^2)` of the gap per solve.
//!
//! ```bash
//! cargo run --release --example rope_body_attachment --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rope::Rope;
use alice_physics::rope_attach::{solve_rope_attachments, RopeAttachment};
use alice_physics::solver::RigidBody;

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let mut anchor = RigidBody::new_static(Vec3Fix::from_int(10, 5, 0));
    anchor.rotation = QuatFix {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::ONE,
        w: Fix128::ZERO,
    };
    let bodies = [anchor];
    let world_anchor = Vec3Fix::from_int(9, 4, 0);

    for (label, compliance) in [
        ("rigid", Fix128::ZERO),
        ("compliant", Fix128::from_ratio(1, 3600)),
    ] {
        let mut rope = Rope::new(
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(4, 0, 0),
            4,
            Fix128::ONE,
        );
        let att = [RopeAttachment::new(0, 0, Vec3Fix::from_int(1, 1, 0)).compliance(compliance)];
        let before = (rope.positions[0] - world_anchor).length().to_f64();
        let broken =
            solve_rope_attachments(&att, &mut rope.positions, &mut rope.velocities, &bodies, dt);
        let after = (rope.positions[0] - world_anchor).length().to_f64();
        println!(
            "{label}: gap {before:.6} -> {after:.6} (broken: {})",
            broken.len()
        );
    }

    // Breakable attachment: force estimate = gap / (1 + alpha) / dt^2 = 9.848858 * 3600.
    let mut rope = Rope::new(
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(4, 0, 0),
        4,
        Fix128::ONE,
    );
    let att = [RopeAttachment::with_break_force(
        0,
        0,
        Vec3Fix::from_int(1, 1, 0),
        Fix128::from_int(30_000),
    )];
    let broken =
        solve_rope_attachments(&att, &mut rope.positions, &mut rope.velocities, &bodies, dt);
    println!("breakable (threshold 30000, force ~35456): broken = {broken:?}");
}
