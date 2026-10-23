//! A world snapshot as a build with another `PHYSICS_SEMANTICS_ID` would write it.
//!
//! A version 4 snapshot carries the writing build's semantics identifier at
//! `[16..48)`, the world's `law_id` under it at `[48..80)`, and ends with the
//! FNV-1a 64 of every preceding byte (little-endian). Everything else is the
//! world's state. Rewriting those three parts with another identifier gives
//! the bytes the build of that identifier wrote for the same state, so a digest
//! pinned before an identifier moved can be reproduced on the new code: equal
//! digests show that only the identifier moved, not the stepped state.
//!
//! Included through a `path` attribute by the tests that
//! pin snapshot digests.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::solver::PhysicsWorld;

/// `w.snapshot_world()` with the header's semantics identifier set to `id`, its
/// `law_id` set to `w.law_id(id)`, and the trailing checksum recomputed.
pub fn snapshot_under(w: &PhysicsWorld, id: &[u8; 32]) -> Vec<u8> {
    let mut bytes = w.snapshot_world();
    assert!(
        bytes.len() >= 88,
        "a version 4 snapshot has an 80-byte header and a checksum"
    );
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        4,
        "the header layout here is that of version 4"
    );
    bytes[16..48].copy_from_slice(id);
    bytes[48..80].copy_from_slice(&w.law_id(id));
    let end = bytes.len() - 8;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &bytes[..end] {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
    bytes[end..].copy_from_slice(&hash.to_le_bytes());
    bytes
}
