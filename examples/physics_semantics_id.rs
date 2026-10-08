//! The content hash of the stepping semantics: `PHYSICS_SEMANTICS_ID`.
//!
//! A host that stores results (a replay, a saved world, a cache of stepped
//! states) can record this identifier next to them. When a later build reports
//! a different value, at least one pinned stepping path (or the det-math
//! functions it evaluates) gives different bits, so the stored results are not
//! reproducible by that build. Pass a previously recorded hex value as the
//! first argument to compare.
//!
//! ```bash
//! cargo run --release --example physics_semantics_id
//! cargo run --release --example physics_semantics_id -- <64 hex digits>
//! ```

use alice_physics::{PHYSICS_SEMANTICS_ID, PHYSICS_SEMANTICS_PINS};

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
            std::process::exit(1);
        }
    }
}
