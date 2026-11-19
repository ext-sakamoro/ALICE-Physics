//! A strip pulled to 1.25 times its length, with and without the triangle area constraint
//!
//! Reaches `Cloth::set_area_constraint`, `Cloth::area_constraint`,
//! `Cloth::clear_area_constraint` and `Cloth::total_area`.
//!
//! A 2 m × 1 m strip (stiff edges, no gravity) has both short ends pinned and moved apart
//! over one second, then held. With the area constraint on, every triangle keeps its rest
//! area, so the strip keeps its area and narrows toward `W₀·L₀/L = 0.8 m` in the middle.
//! Without it, the area grows with the pull. The example runs both and checks that.
//!
//! Run: `cargo run --example cloth_area_constraint --features std`
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::math::{Fix128, Vec3Fix};

const NX: usize = 9;
const NZ: usize = 5;

fn strip(area: bool) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(2),
        Fix128::ONE,
        NX,
        NZ,
        Fix128::from_ratio(1, 100),
    );
    c.config.gravity = Vec3Fix::ZERO;
    c.config.iterations = 20;
    c.config.damping = Fix128::from_ratio(9, 10);
    c.config.stretch_compliance = Fix128::from_ratio(1, 100);
    if area {
        c.set_area_constraint(Fix128::ZERO);
    }
    c
}

/// Pull the strip's end columns from length 2 to 2.5 over 60 frames, hold 120 frames,
/// and return `(total area, width at the middle column)`.
fn pull(c: &mut Cloth) -> (f64, f64) {
    let ends: Vec<(usize, Fix128)> = (0..NZ)
        .flat_map(|j| {
            [
                (j * NX, c.positions[j * NX].x),
                (j * NX + NX - 1, c.positions[j * NX + NX - 1].x),
            ]
        })
        .collect();
    let centre = (c.positions[0].x + c.positions[NX - 1].x).half();
    for &(i, _) in &ends {
        c.pin(i);
    }
    for f in 1..=180_i64 {
        let s = Fix128::ONE + Fix128::from_ratio(f.min(60), 240);
        for &(i, x0) in &ends {
            c.positions[i].x = centre + (x0 - centre) * s;
        }
        c.step(Fix128::from_ratio(1, 60));
    }
    let mid = NX / 2;
    let width = c.positions[(NZ - 1) * NX + mid].z - c.positions[mid].z;
    (c.total_area().to_f64(), width.abs().to_f64())
}

fn main() {
    let mut kept = strip(true);
    assert_eq!(kept.area_constraint(), Some(Fix128::ZERO));
    let a0 = kept.total_area().to_f64();
    let (a_on, w_on) = pull(&mut kept);

    let mut free = strip(false);
    assert_eq!(free.area_constraint(), None);
    let (a_off, w_off) = pull(&mut free);

    println!("rest area {a0:.4} m²");
    println!(
        "area constraint on : area {a_on:.4} m², middle width {w_on:.4} m (incompressible 0.8000)"
    );
    println!("area constraint off: area {a_off:.4} m², middle width {w_off:.4} m");
    assert!(
        // within 5 %: the compliant edges keep pulling against the rigid area constraint
        // each iteration (the bound is derived in tests/analytic_cloth_area_constraint.rs);
        // without it the area grows by about 20 %
        (a_on - a0).abs() < 0.05 * a0,
        "on: the area is kept ({a_on} vs {a0})"
    );
    assert!(
        a_off > a0 * 1.1,
        "off: the area grows with the pull ({a_off} vs {a0})"
    );
    assert!(
        w_on < w_off,
        "on: the strip narrows more ({w_on} vs {w_off})"
    );

    kept.clear_area_constraint();
    assert_eq!(kept.area_constraint(), None);
    println!("cleared: the constraint is off again");
}
