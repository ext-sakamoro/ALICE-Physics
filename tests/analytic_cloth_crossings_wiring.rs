//! Oracles for `Cloth::remaining_self_contact_crossings` (previously without a
//! production caller): the count of "the cloth passed through itself this frame" events.
//!
//! Geometry (closed form): a flat sheet in the plane `y = 0`. A vertex whose frame chord
//! `start -> end` crosses that plane inside a non-incident triangle is a vertex-face
//! crossing; a chord that stays on one side, or an undisturbed frame (`start == end`),
//! has none.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::cloth::Cloth;
use alice_physics::math::{Fix128, Vec3Fix};

fn sheet() -> Cloth {
    // 5 x 5 particles on the plane y = 0 spanning [0, 4]^2, spacing 1
    Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(4),
        Fix128::from_int(4),
        5,
        5,
        Fix128::from_ratio(1, 100),
    )
}

fn fx(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

#[test]
fn an_undisturbed_frame_has_no_crossings() {
    let cloth = sheet();
    let start = cloth.positions.clone();
    assert_eq!(cloth.remaining_self_contact_crossings(&start), 0);
}

#[test]
fn a_vertex_pushed_through_a_distant_triangle_is_counted() {
    // particle 0 sits at the (0, 0) corner; carry it from above the sheet to below it at
    // (3.3, y, 3.3), inside a triangle it does not share a vertex with
    let mut cloth = sheet();
    let start = {
        let mut s = cloth.positions.clone();
        s[0] = fx(3.3, 1.0, 3.3);
        s
    };
    cloth.positions[0] = fx(3.3, -1.0, 3.3);
    assert!(cloth.remaining_self_contact_crossings(&start) >= 1);
}

#[test]
fn a_chord_that_stays_above_the_sheet_is_not_a_crossing() {
    let mut cloth = sheet();
    let start = {
        let mut s = cloth.positions.clone();
        s[0] = fx(3.3, 2.0, 3.3);
        s
    };
    cloth.positions[0] = fx(3.3, 0.5, 3.3); // never reaches y = 0
    assert_eq!(cloth.remaining_self_contact_crossings(&start), 0);
}

#[test]
fn the_count_is_symmetric_in_the_direction_of_the_pierce() {
    let down = {
        let mut cloth = sheet();
        let mut s = cloth.positions.clone();
        s[0] = fx(3.3, 1.0, 3.3);
        cloth.positions[0] = fx(3.3, -1.0, 3.3);
        cloth.remaining_self_contact_crossings(&s)
    };
    let up = {
        let mut cloth = sheet();
        let mut s = cloth.positions.clone();
        s[0] = fx(3.3, -1.0, 3.3);
        cloth.positions[0] = fx(3.3, 1.0, 3.3);
        cloth.remaining_self_contact_crossings(&s)
    };
    assert!(down >= 1);
    assert_eq!(down, up);
}

#[test]
fn a_start_snapshot_shorter_than_the_cloth_is_tolerated() {
    // only the first `start.len()` particles are examined (documented `min`)
    let cloth = sheet();
    let short = cloth.positions[..3].to_vec();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cloth.remaining_self_contact_crossings(&short)
    }));
    assert_eq!(r.expect("a short snapshot must not panic"), 0);
    let empty: Vec<Vec3Fix> = Vec::new();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cloth.remaining_self_contact_crossings(&empty)
    }));
    assert_eq!(r.expect("an empty snapshot must not panic"), 0);
}
