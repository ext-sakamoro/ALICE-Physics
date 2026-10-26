//! The content hash of the stepping semantics: `PHYSICS_SEMANTICS_ID`.
//!
//! A host that stores results (a replay, a saved world, a cache of stepped
//! states) can record this identifier next to them. When a later build reports
//! a different value, at least one pinned stepping path (or the det-math
//! functions it evaluates) gives different bits, so the stored results are not
//! reproducible by that build. Pass a previously recorded hex value as the
//! first argument to compare. A recorded value that is the identifier of an
//! earlier `alice-det-math` generation is named, with the pins whose recorded
//! value differs under it.
//!
//! ```bash
//! cargo run --release --example physics_semantics_id
//! cargo run --release --example physics_semantics_id -- <64 hex digits>
//! ```

use alice_physics_core::{
    PHYSICS_PINS_BY_DET_MATH, PHYSICS_SEMANTICS_BY_DET_MATH, PHYSICS_SEMANTICS_ID,
    PHYSICS_SEMANTICS_PINS,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let id = hex(&PHYSICS_SEMANTICS_ID);
    println!("PHYSICS_SEMANTICS_ID {id}");
    println!("{} entries:", PHYSICS_SEMANTICS_PINS.len());
    for (name, digest) in PHYSICS_SEMANTICS_PINS {
        println!("  {name:<26} {}", hex(digest));
    }
    if let Some(recorded) = std::env::args().nth(1) {
        if recorded.eq_ignore_ascii_case(&id) {
            println!("matches the recorded identifier: same stepping semantics");
        } else {
            println!(
                "differs from the recorded {recorded}: results recorded under it may not reproduce"
            );
            let earlier = PHYSICS_SEMANTICS_BY_DET_MATH
                .iter()
                .find(|(_, row)| recorded.eq_ignore_ascii_case(&hex(row)));
            if let Some((det_math, _)) = earlier {
                println!("  the identifier with alice-det-math {}", hex(det_math));
                for (_, name, digest) in PHYSICS_PINS_BY_DET_MATH
                    .iter()
                    .filter(|(det, _, _)| det == det_math)
                {
                    println!("  {name:<26} was {}", hex(digest));
                }
            }
            std::process::exit(1);
        }
    }
}
