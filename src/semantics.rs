//! Content hash of the stepping semantics: [`PHYSICS_SEMANTICS_ID`].
//!
//! The determinism goldens under `tests/determinism_*.rs` pin the bits a world
//! has after it is stepped, one digest per stepping path. This module folds
//! those digests, plus the semantics identifier of `alice-det-math` (the
//! transcendental functions the solvers evaluate), into one 32-byte value.
//! Two builds with the same identifier step every pinned path to the same
//! bits; a build that steps any of them differently has a different value.
//!
//! # Hex form
//!
//! ```text
//! 3aa1464559166369539c6c972b5140f35ab7d3adc181badcbe18d02a1fa8df72
//! ```
//!
//! # Entries
//!
//! [`PHYSICS_SEMANTICS_PINS`] has one entry per row of the `PINS` table of
//! `tests/determinism_golden_coverage.rs` (23 rows), named by the
//! combination that row pins, with the SHA-256 golden of the test it names,
//! plus one entry `"alice-det-math"` holding `alice_det_math::SEMANTICS_ID`.
//! Several combinations share one digest (for example `step`, `step_n` and
//! `try_step` give the same bits); each is still its own entry, so the set
//! of pinned paths is part of the value.
//!
//! `physics2d` is pinned by a `u64` digest; its entry is the SHA-256 of the
//! same final state (`EXPECTED_SHA256` in
//! `tests/determinism_physics2d_step_digest.rs`).
//!
//! The rows the coverage table lists as known gaps (stepping combinations
//! that no golden pins yet) are not entries. **The identifier therefore
//! changes when a gap is filled (a row moves from the gaps to the pins) or
//! when a pin is moved (a golden is re-recorded)**, and it does not notice a
//! change on a path that has no golden.
//!
//! # Fold
//!
//! The same procedure as the `SEMANTICS_ID` fold of `alice-det-math`: sort
//! the entries by name (ascending byte order), reject an empty table, a
//! repeated name, an empty name or a name that is not ASCII, then SHA-256
//! over, for each entry in that order, the name length as 4 bytes
//! big-endian (`u32`), the name bytes, and the 32 digest bytes. The length
//! prefix keeps two different tables from producing the same byte stream;
//! sorting makes the value independent of the order the table is written in.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants that no `cfg` gates. The
//! entries of the paths that need a feature to run (`step_parallel` and
//! `try_step_parallel` with `parallel`, `step_with_bridge` and
//! `substep_with_bridge` with `gpu-solver-bridge`) are always present, so a
//! build with any feature set reports the same identifier.
//!
//! # Checked by
//!
//! `tests/physics_semantics_id.rs` reads every digest from the golden test
//! the coverage table names (the source, not a copy), checks that the entry
//! names are exactly the pinned combinations plus `"alice-det-math"`, that
//! the det-math entry equals `alice_det_math::SEMANTICS_ID`, and recomputes
//! the fold. Re-recording order after an intended change of a stepping
//! path: update the golden first, then the entry here, then the identifier.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The `(name, digest)` entries [`PHYSICS_SEMANTICS_ID`] folds: one per
/// pinned stepping combination, plus `"alice-det-math"`. See the module
/// documentation for where each digest comes from.
pub const PHYSICS_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "Xpbd step",
        hex32("cf46cd7596ad87225bc7f1e17e1296622b8ed25659d7df968f063f06885a056b"),
    ),
    (
        "Xpbd step_n",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "Xpbd try_step",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "Xpbd step_parallel",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "Xpbd try_step_parallel",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "Xpbd step_with_bridge",
        hex32("d73751a75b8b95b567fc4c212638e7fd994e1506e0abbb5af49a662032c0d9fd"),
    ),
    (
        "Xpbd substep_with_bridge",
        hex32("d73751a75b8b95b567fc4c212638e7fd994e1506e0abbb5af49a662032c0d9fd"),
    ),
    (
        "Tgs step",
        hex32("cfb9f3e814a4b0a55c96019f1df345eb4aaba233fa5321e021dbcb916ff730fe"),
    ),
    (
        "Tgs step_n",
        hex32("cfb9f3e814a4b0a55c96019f1df345eb4aaba233fa5321e021dbcb916ff730fe"),
    ),
    (
        "Tgs try_step",
        hex32("cfb9f3e814a4b0a55c96019f1df345eb4aaba233fa5321e021dbcb916ff730fe"),
    ),
    (
        "contacts",
        hex32("532dc2852b71eb9da1ca3a62a4f4063f2e1151561744d15852f9fd9dee6b28ad"),
    ),
    (
        "distance constraint",
        hex32("4c28a6fb32ae076f393c762e590466e9d2e9786a5efacc2c1a114cdc66be5978"),
    ),
    (
        "joint",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "continuous collision",
        hex32("afb100a521cc3d308a842b963c1e97112ac2cd824040bb2a65fdb81c846b4d7a"),
    ),
    (
        "sleeping",
        hex32("2d2fbb89f587cd27872e85f4ed2d856c72bb3f0d62a3d0d5d2f5360260ac7a86"),
    ),
    (
        "broadphase Bvh",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "broadphase DynamicTree",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "broadphase Hybrid",
        hex32("7b962ba6d3dcd404946bcc8c6c423e75d0bdd4dfc8597026eb443c6974016d39"),
    ),
    (
        "participant",
        hex32("3de49332adc74a203ca1c416e3bc162bc432aa692ff5b61a58f5aa13c14bd684"),
    ),
    (
        "cloth",
        hex32("ba4fd85c6e390518ef3326802fb3a29498e1994245901e8636cdad9c0ae1d556"),
    ),
    (
        "physics2d",
        hex32("9bf77c1406346913dd64841bddcb1e53c458259dd9314250ba33cc75a4216536"),
    ),
    (
        "step_parallel shared-body order",
        hex32("412bed1309a8709f63637bbe6b1a796a249594c880f024f2b814834089ef16ca"),
    ),
    (
        "installed bridge",
        hex32("d73751a75b8b95b567fc4c212638e7fd994e1506e0abbb5af49a662032c0d9fd"),
    ),
    (
        "alice-det-math",
        hex32("d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e"),
    ),
];

/// Content hash of the stepping semantics: the fold of
/// [`PHYSICS_SEMANTICS_PINS`] described in the [module documentation](self).
///
/// Hex: `3aa1464559166369539c6c972b5140f35ab7d3adc181badcbe18d02a1fa8df72`.
/// The same value in every build, whatever cargo features are enabled.
pub const PHYSICS_SEMANTICS_ID: [u8; 32] =
    hex32("3aa1464559166369539c6c972b5140f35ab7d3adc181badcbe18d02a1fa8df72");
