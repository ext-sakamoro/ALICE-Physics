//! Content hash of the stepping semantics: [`PHYSICS_SEMANTICS_ID`].
//!
//! The determinism goldens under `tests/determinism_*.rs` pin the bits a world
//! has after it is stepped, one digest per stepping path. This module folds
//! those digests, plus the semantics identifier of `alice-det-math` (the
//! transcendental functions the solvers evaluate), into one 32-byte value.
//! Two builds with the same identifier step every pinned path to the same
//! bits; a build that steps any of them differently has a different value.
//!
//! # Hex form (with `alice-det-math` 0.5.1; other generations in [`PHYSICS_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! 7b24f99b81c4a5e1369ea5e3f99e221fca0286e05c193ba8a453f3670ce087c2
//! ```
//!
//! # Entries
//!
//! [`PHYSICS_SEMANTICS_PINS`] has one entry per row of the `PINS` table of
//! `tests/determinism_golden_coverage.rs` (23 rows), named by the
//! combination that row pins, with the SHA-256 golden of the test it names,
//! plus one entry per Law output: the golden of
//! `tests/determinism_law_outputs.rs` for each src file that calls the Law
//! max / min / clamp (named `"law output: <file>"`, listed in its
//! `LAW_OUTPUT_PINS` table; a file with such a call and no row fails there),
//! plus one entry `"alice-det-math"` holding `alice_det_math::SEMANTICS_ID`,
//! plus one entry per optional Law-bearing peer this crate exchanges values
//! with ([`PHYSICS_PEER_IDS`]: `"alice-analytics"`).
//! **A peer's identifier moving moves this identifier, by design**: a world
//! stepped by this build and handed to a peer of another arithmetic is a
//! different exchange.
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
pub(crate) const fn hex32(s: &str) -> [u8; 32] {
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
    // the Law outputs: one golden per src file that calls the Law max / min /
    // clamp (`LAW_OUTPUT_PINS` in tests/determinism_law_outputs.rs)
    (
        "law output: anomaly",
        hex32("ba0f9ef3fde0df57aa55af0ecd88608875d326ad8b9ae88dd98ca257954375e1"),
    ),
    (
        "law output: convex_decompose",
        hex32("504fba138f56a4bc99a3141402bea5d794fd1e3b69f0224c415e50d95817350d"),
    ),
    (
        "law output: erosion",
        hex32("3d4bbcb2c3107120bb33a7b91821dbeda2c564a13dd3c640e25c30bd38042746"),
    ),
    (
        "law output: fracture",
        hex32("6547e36692326d189c321022b065b2567490da1fc3994bcd11ccc1d4bf684f5f"),
    ),
    (
        "law output: heatmap",
        hex32("32d50d9108e2fce2cc0d4b55f82987e70625d97f15173cd37142b765ab4f8be7"),
    ),
    (
        "law output: phase_change",
        hex32("298c35d5de20f49859c56389809987bcfbf3c66a8ef9ec9e3efd8e5bb623c677"),
    ),
    (
        "law output: pressure",
        hex32("ced6e512e72ddfdda18be4db0eee969765011b3b38d9e54c268b053c1177635f"),
    ),
    (
        "law output: privacy",
        hex32("9c8f5949b5346420ca73e4ac029ddd07da9c4e74d0c08ddd4a0e1d4c8c46bff8"),
    ),
    (
        "law output: sdf_character",
        hex32("b17e24f1544ea96ab55ed1b6949cb4fc2f3beabd8966f9a4981c8993c2174fa0"),
    ),
    (
        "law output: sdf_collider",
        hex32("b6feffa57022484b7a84e13fec2faa7290e264f273d895c01dac963986e4db1e"),
    ),
    (
        "law output: sdf_destruction",
        hex32("684e6d18da512f696f3620a1540adce95165550a37a97dbde8de60b30557a717"),
    ),
    (
        "law output: sdf_fem_mesh",
        hex32("641feebc7a04d2f8a0c5367d2f687db315f1bb2c37a0b24d4d01bc7b800354f3"),
    ),
    (
        "law output: sdf_sph",
        hex32("d20c7444cc5a18c5548881c46015bca1768e13e15857ce94ab791a32c473c051"),
    ),
    (
        "law output: sdf_wind_field",
        hex32("41b4e95ba70ff419c2f1c2083df0999e6e34b110e2fb430431dd578900f483a8"),
    ),
    (
        "law output: shape",
        hex32("d3a0cb188c4331c280ac3bf3678570c850c7ac03563c8c44830569560596f36a"),
    ),
    (
        "law output: sim_field",
        hex32("98ea603b87edf5f774011a75b426b7545561e06d0870e72bc2275f0cfe41768d"),
    ),
    (
        "law output: sim_modifier observe",
        hex32("28068d73ca7e51dc188a25c39562709bf56620c522ae317b7c42b013cc5022e4"),
    ),
    (
        "law output: sketch",
        hex32("83275eec34cb7e18cf3c5db2582b25047ba6cbc254fd08036db12fb4b9291b31"),
    ),
    (
        "law output: tgs oriented 6dof hooks",
        hex32("79144576c7e17440faaaa14e5e1496c501bd5e929e83ebbdaece82ad87f115ff"),
    ),
    (
        "law output: thermal",
        hex32("9c50b43dc262b5c078215b91c01c4914a18f13fbc05092a987ac2cd28a492040"),
    ),
    (
        "law output: thin_wall",
        hex32("5fd41dcdab079b2632a3e36f1f093949d451504b043eedd11230b27e1bad68f8"),
    ),
    (
        "law output: transient_thermal",
        hex32("5c8159bac8ee768397456149e0ed151cf7b5975a3f3a0f13981b32bb975175b3"),
    ),
    ("alice-analytics", PHYSICS_PEER_IDS[0].1),
    // the resolved alice-det-math's own identifier, not a copy of it: the
    // identifier below is looked up by it
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// The identifier each optional Law-bearing peer exports, as this build
/// expects it: `alice-analytics` (feature `analytics`, read by
/// `crate::analytics_bridge`).
///
/// The values are written out, not read from the peers, so the table (and
/// [`PHYSICS_SEMANTICS_ID`], which folds it) is the same in every feature set.
/// `tests/physics_semantics_id.rs` checks under each peer's feature that its
/// entry equals the identifier the resolved peer exports; at run time the
/// bridges compare the linked peer's identifier with its entry
/// ([`check_peer`]) and refuse a peer whose identifier moved.
pub const PHYSICS_PEER_IDS: &[(&str, [u8; 32])] = &[
    // alice-analytics 0.5 / 0.6: `SEMANTICS_ID`, the alice-det-math 0.5.1 identifier it re-exports
    (
        "alice-analytics",
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
    ),
];

/// A Law-bearing peer whose identifier is not the one this build expects
/// ([`PHYSICS_PEER_IDS`]): the bridge refuses to exchange values with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerMismatch {
    /// The bridge that refused the peer (`"analytics_bridge"`, ...).
    pub bridge: &'static str,
    /// The peer crate (`"alice-analytics"`, ...).
    pub peer: &'static str,
    /// The identifier the linked peer exports.
    pub found: [u8; 32],
    /// The identifier this build expects of it.
    pub expected: [u8; 32],
}

impl core::fmt::Display for PeerMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {} identifier ", self.bridge, self.peer)?;
        for b in self.found {
            write!(f, "{b:02x}")?;
        }
        f.write_str(" is not the expected ")?;
        for b in self.expected {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl core::error::Error for PeerMismatch {}

/// The entry of [`PHYSICS_PEER_IDS`] for `peer`, or `None` when the table has
/// no such peer.
#[must_use]
pub fn expected_peer_id(peer: &str) -> Option<[u8; 32]> {
    PHYSICS_PEER_IDS
        .iter()
        .find(|(name, _)| *name == peer)
        .map(|(_, id)| *id)
}

/// Compare the identifier `found` that the linked `peer` exports with the one
/// this build expects of it.
///
/// # Errors
///
/// [`PeerMismatch`] naming both identifiers when they differ.
///
/// # Panics
///
/// When `peer` has no entry in [`PHYSICS_PEER_IDS`] (a bridge to a peer the
/// table does not list is a programming error, caught by the bridge's tests).
pub fn check_peer(
    bridge: &'static str,
    peer: &'static str,
    found: &[u8; 32],
) -> Result<(), PeerMismatch> {
    let expected =
        expected_peer_id(peer).unwrap_or_else(|| panic!("{peer} has no entry in PHYSICS_PEER_IDS"));
    if *found == expected {
        Ok(())
    } else {
        Err(PeerMismatch {
            bridge,
            peer,
            found: *found,
            expected,
        })
    }
}

/// [`PHYSICS_SEMANTICS_ID`] for each `alice-det-math` generation this crate
/// knows: `(alice_det_math::SEMANTICS_ID, fold of PHYSICS_SEMANTICS_PINS with
/// that det-math entry)`.
///
/// The det-math entry of [`PHYSICS_SEMANTICS_PINS`] is the identifier of the
/// `alice-det-math` the build resolved, so the identifier is looked up here
/// by it ([`semantics_id_for`]); a build against a generation with no row
/// does not compile, rather than reporting an identifier of another
/// generation. `alice-det-math` moves `SEMANTICS_ID` in patch releases when
/// the old identifier is listed in its `PREVIOUS_SEMANTICS_IDS` and the
/// existing functions are bit-identical (0.5.1 added `max` / `min`); this
/// table maps each generation to this crate's identifier. A later release of
/// that kind is selected by the `"0.5.1"` requirement and stops the compile
/// here until this crate adds its row.
/// `tests/physics_semantics_id.rs` recomputes every row from the pins in one
/// build.
pub const PHYSICS_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.0
    (
        hex32("135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000"),
        hex32("a19e22d46dfd87d84d36b1e083332d03f43c2360253d294876fef83de1945b27"),
    ),
    // alice-det-math 0.5.1 (adds `max` / `min` / `max64` / `min64`)
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("7b24f99b81c4a5e1369ea5e3f99e221fca0286e05c193ba8a453f3670ce087c2"),
    ),
];

/// Content hash of the stepping semantics: the fold of
/// [`PHYSICS_SEMANTICS_PINS`] described in the [module documentation](self),
/// taken from [`PHYSICS_SEMANTICS_BY_DET_MATH`] by the resolved
/// `alice-det-math`.
///
/// Hex (with `alice-det-math` 0.5.1):
/// `7b24f99b81c4a5e1369ea5e3f99e221fca0286e05c193ba8a453f3670ce087c2`.
/// The same value in every build, whatever cargo features are enabled.
pub const PHYSICS_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_det_math::SEMANTICS_ID);

/// The [`PHYSICS_SEMANTICS_ID`] of a build that resolved the `alice-det-math`
/// with identifier `det_math`.
///
/// # Panics
///
/// When [`PHYSICS_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a
/// const context this stops the compile). A new `alice-det-math` generation
/// needs its row, recomputed from the pins.
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < PHYSICS_SEMANTICS_BY_DET_MATH.len() {
        if same32(&PHYSICS_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return PHYSICS_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in PHYSICS_SEMANTICS_BY_DET_MATH: add its fold");
}

const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Earlier values of [`PHYSICS_SEMANTICS_ID`] under which every pinned path
/// steps to the same bits as this build, newest first.
///
/// A world snapshot whose header carries one of them restores
/// ([`crate::LawCheck::VerifiedUnderPrevious`] /
/// [`crate::LawCheck::SemanticsPrevious`]); any other identifier is still
/// [`crate::WorldSnapshotError::SemanticsMismatch`].
///
/// An identifier is listed only when the change that replaced it moved no
/// determinism golden, so the stepping bits are the same under both. Each is
/// a value of [`PHYSICS_SEMANTICS_BY_DET_MATH`] other than this build's:
///
/// * `79ca5297…` (3.1.0) and the 0.5.0-generation `cd9aa3b8…` (3.0.0 and
///   3.0.1, 3.1.0 with `alice-det-math` 0.5.0): replaced when the peer
///   identifiers ([`PHYSICS_PEER_IDS`]) were folded in, which moved no golden.
/// * `cd9aa3b8…` was itself replaced when `alice-det-math` went from 0.5.0
///   to 0.5.1, which added `max` / `min` / `max64` / `min64` and left the
///   existing functions unchanged (`alice-det-math` 0.5.1 itself lists the
///   0.5.0 identifier in its `PREVIOUS_SEMANTICS_IDS`). No golden moved.
pub const PREVIOUS_PHYSICS_SEMANTICS_IDS: &[[u8; 32]] = &[
    hex32("79ca52976f404dfb453908970ebe45f584970ff9344a56a06aae3fc9c3c38108"),
    hex32("cd9aa3b89257fc8b3e5d6714362ce4e52d4ff142baf30de2f80324e44bdd3469"),
];
