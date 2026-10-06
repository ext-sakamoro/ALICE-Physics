//! Oracles for the material table section of the whole-world snapshot.
//!
//! oracle: the invariants `MaterialTable` keeps through its own API.
//! `MaterialTable::new` registers the default material at id 0 and
//! `try_register` assigns `id = len` and refuses once 65,536 ids are in use,
//! so every table the public API can build holds `1..=65_536` materials with
//! `materials[i].id == i`. A blob whose material section breaks either
//! invariant was not written by `snapshot_world`; `restore_world` must reject
//! it with `WorldSnapshotError::InvalidValue { section: "material_table" }`
//! and leave the target world untouched, while every table inside the
//! invariants (1 material, a few, exactly 65,536) restores to a world that
//! re-encodes to the same bytes.
//!
//! Blobs are built through the public encoder: `snapshot_world` of a known
//! world, then the material section is located by its marker entry and
//! re-spliced with the requested entries, with the payload length and the
//! FNV-1a 64 checksum recomputed (header layout documented on
//! `PhysicsWorld::snapshot_world`).

use alice_physics::{
    Fix128, PhysicsConfig, PhysicsError, PhysicsMaterial, PhysicsWorld, RigidBody, Vec3Fix,
    WorldSnapshotError,
};

/// Bytes of one encoded material: id u16 + 3 `Fix128` + 2 combine tags.
const ENTRY: usize = 2 + 3 * 16 + 2;
/// Header: magic 4 + version 2 + reserved 2 + payload length 8.
const HEADER: usize = 16;
const CHECKSUM: usize = 8;
/// Every `MaterialId` (`u16`) in use.
const CAPACITY: usize = 65_536;

const MATERIAL_ERR: WorldSnapshotError = WorldSnapshotError::InvalidValue {
    section: "material_table",
};

fn marker() -> PhysicsMaterial {
    let mut m = PhysicsMaterial::new(0, Fix128::from_ratio(7, 13), Fix128::from_ratio(3, 19));
    m.dynamic_friction = Fix128::from_ratio(5, 17);
    m
}

/// Source world: one dynamic body using the marker material (id 1).
fn source() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 3, 0),
        Fix128::ONE,
    ));
    let id = w.material_table.register(marker());
    assert_eq!(id, 1);
    w.set_body_material(b, id);
    w
}

/// Target world with different contents, so a partial restore would show.
fn target() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(4, 1, 2),
        Fix128::from_int(2),
    ));
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    w.material_table.register_rubber();
    w.material_table.register_ice();
    w.set_body_material(0, 2);
    w.step(Fix128::from_ratio(1, 60));
    w
}

fn fix_bytes(f: Fix128) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&f.hi.to_le_bytes());
    out[8..].copy_from_slice(&f.lo.to_le_bytes());
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    h
}

/// The blob of [`source`] cut around its material entries: the bytes before
/// the material count, entry 0 (default), entry 1 (marker), and the rest.
struct Cut {
    before: Vec<u8>,
    default_entry: Vec<u8>,
    marker_entry: Vec<u8>,
    after: Vec<u8>,
}

fn cut() -> (Vec<u8>, Cut) {
    let blob = source().snapshot_world();
    let m = marker();
    // id (1) + the three coefficients: unique to the marker entry.
    let mut needle = 1u16.to_le_bytes().to_vec();
    needle.extend_from_slice(&fix_bytes(m.static_friction));
    needle.extend_from_slice(&fix_bytes(m.dynamic_friction));
    needle.extend_from_slice(&fix_bytes(m.restitution));
    let hits: Vec<usize> = blob
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle.as_slice())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "marker entry must occur exactly once");
    let entry1 = hits[0];
    let entry0 = entry1 - ENTRY;
    let count_at = entry0 - 8;
    assert_eq!(&blob[count_at..entry0], &2u64.to_le_bytes());
    assert_eq!(&blob[entry0..entry0 + 2], &0u16.to_le_bytes());
    let end = blob.len() - CHECKSUM;
    let c = Cut {
        before: blob[..count_at].to_vec(),
        default_entry: blob[entry0..entry1].to_vec(),
        marker_entry: blob[entry1..entry1 + ENTRY].to_vec(),
        after: blob[entry1 + ENTRY..end].to_vec(),
    };
    (blob, c)
}

/// A blob whose material section holds `entries` (raw entry bytes, ids
/// already set), with the payload length and checksum made consistent.
fn splice(c: &Cut, entries: &[Vec<u8>]) -> Vec<u8> {
    let mut out = c.before.clone();
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        assert_eq!(e.len(), ENTRY);
        out.extend_from_slice(e);
    }
    out.extend_from_slice(&c.after);
    let payload = (out.len() - HEADER) as u64;
    out[8..16].copy_from_slice(&payload.to_le_bytes());
    let h = fnv1a(&out);
    out.extend_from_slice(&h.to_le_bytes());
    out
}

fn with_id(entry: &[u8], id: u16) -> Vec<u8> {
    let mut e = entry.to_vec();
    e[..2].copy_from_slice(&id.to_le_bytes());
    e
}

/// `n` copies of the default entry with ids `0..n`.
fn defaults(c: &Cut, n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| with_id(&c.default_entry, u16::try_from(i).expect("id fits u16")))
        .collect()
}

/// Restores `blob` into [`target`]; asserts the documented error and that
/// the target world re-encodes to the bytes it had before.
fn assert_rejected(blob: &[u8]) {
    let mut w = target();
    let before = w.snapshot_world();
    assert_eq!(w.restore_world(blob), Err(MATERIAL_ERR));
    assert_eq!(w.snapshot_world(), before, "target world changed");
    assert_eq!(w.material_table.len(), 3);
    assert_eq!(
        PhysicsWorld::from_world_snapshot(blob).err(),
        Some(MATERIAL_ERR)
    );
}

#[test]
fn the_splice_helper_is_exact_on_the_original_table() {
    let (blob, c) = cut();
    let same = splice(&c, &[c.default_entry.clone(), c.marker_entry.clone()]);
    assert_eq!(same, blob);
}

#[test]
fn a_valid_table_round_trips_exactly() {
    let (blob, _) = cut();
    let mut w = target();
    assert_eq!(w.restore_world(&blob), Ok(()));
    assert_eq!(w.snapshot_world(), blob);
    assert_eq!(w.material_table.len(), 2);
    assert_eq!(*w.material_table.get(1), {
        let mut m = marker();
        m.id = 1;
        m
    });
    assert_eq!(*w.material_table.get(0), PhysicsMaterial::default());
}

#[test]
fn a_table_holding_only_the_default_material_is_accepted() {
    let (_, c) = cut();
    let blob = splice(&c, &defaults(&c, 1));
    let w = PhysicsWorld::from_world_snapshot(&blob).expect("1 material is valid");
    assert_eq!(w.material_table.len(), 1);
    assert_eq!(w.snapshot_world(), blob);
}

#[test]
fn exactly_65536_materials_are_accepted_and_the_table_is_full() {
    let (_, c) = cut();
    let blob = splice(&c, &defaults(&c, CAPACITY));
    let mut w = PhysicsWorld::from_world_snapshot(&blob).expect("65,536 materials are valid");
    assert_eq!(w.material_table.len(), CAPACITY);
    assert_eq!(w.material_table.get(u16::MAX).id, u16::MAX);
    assert_eq!(w.snapshot_world(), blob);
    assert_eq!(
        w.material_table.try_register(marker()),
        Err(PhysicsError::CapacityExceeded {
            resource: "materials",
            limit: CAPACITY,
        })
    );
}

#[test]
fn a_table_with_no_materials_is_rejected() {
    let (_, c) = cut();
    assert_rejected(&splice(&c, &[]));
}

#[test]
fn a_table_with_65537_materials_is_rejected() {
    let (_, c) = cut();
    let mut entries = defaults(&c, CAPACITY);
    // The 65,537th entry cannot carry id 65,536 (`u16`), so `id == index`
    // rejects it too: the upper count bound only makes the rejection happen
    // before the entries are read, the error is the same either way.
    entries.push(with_id(&c.default_entry, 0));
    assert_rejected(&splice(&c, &entries));
}

#[test]
fn permuted_ids_are_rejected() {
    let (_, c) = cut();
    let entries = vec![
        with_id(&c.default_entry, 0),
        with_id(&c.marker_entry, 2),
        with_id(&c.default_entry, 1),
    ];
    assert_rejected(&splice(&c, &entries));
}

#[test]
fn a_duplicated_id_is_rejected() {
    let (_, c) = cut();
    let entries = vec![c.default_entry.clone(), with_id(&c.marker_entry, 0)];
    assert_rejected(&splice(&c, &entries));
}

#[test]
fn a_first_id_other_than_zero_is_rejected() {
    let (_, c) = cut();
    let entries = vec![with_id(&c.default_entry, 1), with_id(&c.marker_entry, 0)];
    assert_rejected(&splice(&c, &entries));
}

#[test]
fn a_gap_in_the_ids_is_rejected() {
    let (_, c) = cut();
    let entries = vec![c.default_entry.clone(), with_id(&c.marker_entry, 5)];
    assert_rejected(&splice(&c, &entries));
}
