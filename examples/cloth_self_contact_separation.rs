//! A pinned square of cloth pushed into folds, with the frame-end contact separation counted
//!
//! Reaches `Cloth::self_contact_stats` (and its counters `SelfContactStats`).
//!
//! A 9 × 9 grid, 8 m square, has its boundary pinned and pulled toward the centre over 60
//! frames, so the inside folds onto itself. With self-contact on, every frame ends with the
//! contact separation; the example prints its counters (calls, calls that reached the pass
//! budget, cluster stages and how many of them converged, corrections dropped because
//! they would cross) and checks that the cloth never passed through itself.
//!
//! Run: `cargo run --example cloth_self_contact_separation --features std`
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::math::{Fix128, Vec3Fix};

const RES: usize = 9;
const STEPS: usize = 60;

fn main() {
    let mut cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        RES,
        RES,
        Fix128::from_ratio(1, 100),
    );
    cloth.config.self_collision = true;
    cloth.config.self_collision_distance = Fix128::from_ratio(5, 100);
    let bump = Fix128::from_ratio(1, 8);
    cloth.positions[4 * RES + 4].y = bump;
    cloth.prev_positions[4 * RES + 4].y = bump;
    let boundary = |i: usize| {
        let (r, c) = (i / RES, i % RES);
        r == 0 || c == 0 || r == RES - 1 || c == RES - 1
    };
    let rest = cloth.positions.clone();
    for i in 0..RES * RES {
        if boundary(i) {
            cloth.inv_masses[i] = Fix128::ZERO;
        }
    }
    let dt = Fix128::from_ratio(1, 60);
    let centre = Fix128::from_int(4);
    let mut crossings = 0usize;
    for s in 0..STEPS {
        let shrink = Fix128::ONE
            - Fix128::from_ratio(7, 8) * Fix128::from_int(s as i64)
                / Fix128::from_int(STEPS as i64);
        for (i, r) in rest.iter().enumerate() {
            if boundary(i) {
                cloth.positions[i].x = centre + (r.x - centre) * shrink;
                cloth.positions[i].z = centre + (r.z - centre) * shrink;
            }
        }
        let before = cloth.positions.clone();
        cloth.step(dt);
        crossings += cloth.remaining_self_contact_crossings(&before);
    }
    let st = cloth.self_contact_stats();
    println!("frames {STEPS}, crossings {crossings}");
    println!(
        "separation calls {}, reached the pass budget {}, cluster stages {} (converged {}), dropped corrections {}",
        st.separation_calls, st.capped_calls, st.cluster_solves, st.cluster_converged, st.clipped_corrections
    );
    assert_eq!(crossings, 0, "the cloth passed through itself");
    assert!(
        st.separation_calls >= STEPS as u64,
        "a frame ended without the separation"
    );
    assert_eq!(
        st.cluster_solves, st.capped_calls,
        "a capped call skipped the cluster stage"
    );
}
