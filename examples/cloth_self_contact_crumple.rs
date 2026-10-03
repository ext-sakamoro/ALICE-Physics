//! Self-contact of a crumpling cloth, measured with
//! `Cloth::remaining_self_contact_crossings`.
//!
//! A 9x9 sheet is pinned on its boundary, which is driven toward the centre until the
//! interior must buckle (a deterministic 1/8 bump breaks the symmetry). After every
//! frame the invariant "the cloth did not pass through itself this frame" is counted:
//! vertex chords piercing a triangle plus edge pairs that swapped sides. With
//! `self_collision` on the frame repair keeps that count at zero; with it off the same
//! scene tunnels, so a zero is not a vacuous result of the instrument.
//!
//! ```bash
//! cargo run --release --example cloth_self_contact_crumple --features std
//! ```

use alice_physics::cloth::Cloth;
use alice_physics::math::Fix128;

const RES: usize = 9;
const STEPS: usize = 120;

fn crossings(self_collision: bool) -> usize {
    let mut cloth = Cloth::new_grid(
        alice_physics::math::Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        RES,
        RES,
        Fix128::from_ratio(1, 100),
    );
    cloth.config.self_collision = self_collision;
    cloth.config.self_collision_distance = Fix128::from_ratio(5, 100);
    let bump = Fix128::from_ratio(1, 8);
    cloth.positions[4 * RES + 4].y = bump;
    cloth.prev_positions[4 * RES + 4].y = bump;

    let is_boundary = |i: usize| {
        let (r, c) = (i / RES, i % RES);
        r == 0 || c == 0 || r == RES - 1 || c == RES - 1
    };
    let rest = cloth.positions.clone();
    for i in 0..RES * RES {
        if is_boundary(i) {
            cloth.inv_masses[i] = Fix128::ZERO;
        }
    }
    let dt = Fix128::from_ratio(1, 60);
    let centre = Fix128::from_int(4);
    let total = Fix128::from_int(STEPS as i64);
    let mut count = 0usize;
    for s in 0..STEPS {
        let shrink = Fix128::ONE - Fix128::from_ratio(7, 8) * Fix128::from_int(s as i64) / total;
        for (i, r) in rest.iter().enumerate() {
            if is_boundary(i) {
                cloth.positions[i].x = centre + (r.x - centre) * shrink;
                cloth.positions[i].z = centre + (r.z - centre) * shrink;
            }
        }
        let before = cloth.positions.clone();
        cloth.step(dt);
        count += cloth.remaining_self_contact_crossings(&before);
    }
    count
}

fn main() {
    let on = crossings(true);
    let off = crossings(false);
    println!("self-contact crossings over {STEPS} frames: ON = {on}, OFF = {off}");
    assert_eq!(
        on, 0,
        "the frame repair must keep the cloth from passing through itself"
    );
    assert!(
        off > 0,
        "without self-contact the same scene tunnels, and the invariant sees it"
    );
}
