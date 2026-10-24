//! Checkpoint and restore of a Yee lattice through its exact state bytes.
//!
//! A lattice with a graded PML, a lossy dielectric slab and a current is run
//! for a while, written with [`YeeGrid::write_state`], read back with
//! [`YeeGrid::read_state`], and both copies are run on. The restored copy
//! stays bit-identical to the original.
//!
//! The second copy shows why the state bytes are needed: rebuilding the same
//! lattice through [`YeeGrid::get`] and [`YeeGrid::set`] loses the absorber's
//! split half fields (only their sum is visible), so it takes different bits
//! from its next step on.
//!
//! ```bash
//! cargo run --example maxwell_state_checkpoint --features std
//! ```

use alice_physics_core::math::Fix128;
use alice_physics_core::maxwell_fdtd::{
    Absorber, Component, Material, MaterialMap, YeeGrid, COURANT_3D,
};

const COMPONENTS: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

fn lattice() -> YeeGrid {
    let mut map = MaterialMap::vacuum(12, 10, 8);
    map.fill(
        [5, 4, 3],
        [7, 6, 5],
        Material::new(
            Fix128::from_int(2),
            Fix128::ONE,
            Fix128::ONE / Fix128::from_int(8),
        ),
    );
    YeeGrid::new_with_absorber(
        12,
        10,
        8,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 2, 2],
            sigma_max: Fix128::from_int(3),
        },
    )
    .with_materials(&map)
    .expect("the slab is inside the absorber and under the Courant limit")
}

fn differing_samples(a: &YeeGrid, b: &YeeGrid) -> usize {
    let mut n = 0;
    for c in COMPONENTS {
        let (ni, nj, nk) = a.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    if a.get(c, i, j, k) != b.get(c, i, j, k) {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

fn main() {
    let mut original = lattice();
    original.set(Component::Ez, 4, 5, 4, Fix128::ONE);
    original.set_current(Component::Ex, 4, 4, 4, Fix128::ONE / Fix128::from_int(3));
    for _ in 0..40 {
        original.step();
    }

    let mut bytes = Vec::new();
    original.write_state(&mut bytes);
    let mut restored = YeeGrid::read_state(&bytes).expect("bytes written by write_state");

    let mut rebuilt = lattice();
    for c in COMPONENTS {
        let (ni, nj, nk) = original.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    rebuilt.set(c, i, j, k, original.get(c, i, j, k));
                }
            }
        }
    }
    rebuilt.set_current(Component::Ex, 4, 4, 4, Fix128::ONE / Fix128::from_int(3));

    println!("state bytes: {}", bytes.len());
    println!("step  restored≠original  rebuilt≠original");
    for step in 1..=60 {
        original.step();
        restored.step();
        rebuilt.step();
        if step % 10 == 0 {
            println!(
                "{step:>4}  {:>17}  {:>16}",
                differing_samples(&restored, &original),
                differing_samples(&rebuilt, &original)
            );
        }
    }
    assert_eq!(restored, original, "the restored lattice drifted");
}
