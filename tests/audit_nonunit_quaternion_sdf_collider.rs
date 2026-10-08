//! A static SDF collider whose orientation is given as a non-unit quaternion
//! (norm 2, 1/2, 1.3) is stored as the unit quaternion it stands for,
//! whichever way the orientation came in:
//!
//! - `SdfCollider::new_static`;
//! - `SdfCollider::set_pose`;
//! - a direct write to the public `rotation` field followed by
//!   `SdfCollider::update_cache`;
//! - a direct write to the public `rotation` field (without `update_cache`)
//!   before `PhysicsWorld::add_sdf_collider`;
//! - a direct write to `PhysicsWorld::sdf_colliders[i].rotation` (without
//!   `update_cache`) between steps, which the next step brings to unit length
//!   (before that step the contact normal is rotated by the value as written);
//! - a `PhysicsWorld::restore_world` blob holding a non-unit rotation and
//!   inverse rotation for the collider, which the next step brings to unit
//!   length (before that step the queries use the rotation as restored).
//!
//! The field is evaluated at `R⁻¹ (p - position)`, so a stored non-unit
//! inverse rotation scaled the local point by `|q|^2`: with the scene below a
//! norm-2 rotation lost the contact and a norm-1/2 rotation deepened it.
//!
//! # Expected values
//!
//! Each scene is run once with the scaled rotation `s·q` and once with
//! `normalize(s·q)`, and the observations (contact count, depth, normal, and
//! the probe position after one step) must agree bit for bit: the collider
//! stores `normalize(s·q)` for the first, and a quaternion within `2^-32` of
//! unit length is stored unchanged. Against the original unit `q` (which
//! differs from `normalize(s·q)` by the rounding of the normalization) they
//! agree to `1e-4`, the field being evaluated in `f32`. Every read asserts a
//! contact, so a scene that misses the field fails instead of comparing two
//! empty lists.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{collide_sphere_sdf, ClosureSdf, SdfCollider};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const NEAR_F32: f64 = 1e-4;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scaled(q: QuatFix, s: Fix128) -> QuatFix {
    QuatFix::new(q.x * s, q.y * s, q.z * s, q.w * s)
}

/// A turn about a slanted axis, so every component of the quaternion is non-zero.
fn turn() -> QuatFix {
    QuatFix::from_axis_angle(v3(1.0, 2.0, 3.0).normalize(), fx(0.7))
}

const SCALES: [f64; 3] = [2.0, 0.5, 1.3];

/// Collider origin.
fn origin() -> Vec3Fix {
    v3(0.3, -0.2, 0.1)
}

/// Probe radius (the world's `sdf_collision_radius`).
fn radius() -> Fix128 {
    fx(0.5)
}

/// A ball of radius 1 centred at local `(1.5, 0, 0)`: off the origin, so the
/// orientation decides where it is.
fn field() -> Box<ClosureSdf> {
    Box::new(ClosureSdf::new(
        |x, y, z| ((x - 1.5) * (x - 1.5) + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = ((x - 1.5) * (x - 1.5) + y * y + z * z).sqrt().max(1e-6);
            ((x - 1.5) / l, y / l, z / l)
        },
    ))
}

/// The probe centre: local `(2.2, 0.2, 0.1)` in the unit frame of `turn()`,
/// 0.735 from the ball centre, so a sphere of radius 0.5 there overlaps the
/// ball by about 0.765.
fn probe_at() -> Vec3Fix {
    origin() + turn().rotate_vec(v3(2.2, 0.2, 0.1))
}

/// How the orientation of the collider under test reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    NewStatic,
    SetPose,
    UpdateCache,
    FieldBeforeAdd,
    WorldField,
    Restore,
}

const ENTRIES: [Entry; 6] = [
    Entry::NewStatic,
    Entry::SetPose,
    Entry::UpdateCache,
    Entry::FieldBeforeAdd,
    Entry::WorldField,
    Entry::Restore,
];

/// The collider with orientation `q`, built through the collider-level
/// entries (the world-level entries start from `q` given to `new_static`).
fn collider(q: QuatFix, entry: Entry) -> SdfCollider {
    match entry {
        Entry::SetPose => {
            let mut c = SdfCollider::new_static(field(), Vec3Fix::ZERO, QuatFix::IDENTITY);
            c.set_pose(origin(), q);
            c
        }
        Entry::UpdateCache => {
            let mut c = SdfCollider::new_static(field(), origin(), QuatFix::IDENTITY);
            c.rotation = q;
            c.update_cache();
            c
        }
        Entry::FieldBeforeAdd => {
            // `rotation` only: the cached inverse keeps the orientation given
            // to `new_static`, which is the unit `q` itself (see `scene`).
            let mut c = SdfCollider::new_static(field(), origin(), unit_of(q));
            c.rotation = q;
            c
        }
        Entry::NewStatic | Entry::WorldField | Entry::Restore => {
            SdfCollider::new_static(field(), origin(), q)
        }
    }
}

/// `q` at unit length: kept as given when its squared length is within
/// `2^-32` of one, normalized otherwise (the storage rule under test, written
/// out here so the scaled and the reference run start from the same bits).
fn unit_of(q: QuatFix) -> QuatFix {
    let tol = Fix128 { hi: 0, lo: 1 << 32 };
    if (q.length_squared() - Fix128::ONE).abs() <= tol {
        q
    } else {
        q.normalize()
    }
}

struct Obs {
    values: Vec<Fix128>,
}

impl Obs {
    fn s(&mut self, s: Fix128) {
        self.values.push(s);
    }
    fn v(&mut self, v: Vec3Fix) {
        self.s(v.x);
        self.s(v.y);
        self.s(v.z);
    }
    fn contact(&mut self, depth: Fix128, normal: Vec3Fix) {
        assert!(depth > Fix128::ZERO, "the probe overlaps the field");
        self.s(depth);
        self.v(normal);
    }
}

fn read_world(w: &PhysicsWorld, o: &mut Obs) {
    let contacts = w.sdf_contacts();
    assert_eq!(contacts.len(), 1, "the probe touches the field once");
    o.s(Fix128::from_int(contacts.len() as i64));
    for (_, c) in contacts {
        o.contact(c.depth, c.normal);
    }
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.sdf_collision_radius = radius();
    w
}

/// `bytes` replaced by `with` at its only occurrence in `blob`.
fn patch_once(blob: &mut [u8], bytes: &[u8], with: &[u8]) {
    let at: Vec<usize> = (0..=blob.len() - bytes.len())
        .filter(|&i| &blob[i..i + bytes.len()] == bytes)
        .collect();
    assert_eq!(at.len(), 1, "the pattern occurs once in the blob");
    blob[at[0]..at[0] + bytes.len()].copy_from_slice(with);
}

fn quat_bytes(q: QuatFix) -> Vec<u8> {
    let mut out = Vec::new();
    for c in [q.x, q.y, q.z, q.w] {
        out.extend_from_slice(&c.hi.to_le_bytes());
        out.extend_from_slice(&c.lo.to_le_bytes());
    }
    out
}

/// The blob's trailing FNV-1a checksum recomputed over the rest.
fn reseal(blob: &mut [u8]) {
    let n = blob.len() - 8;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &blob[..n] {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
    blob[n..].copy_from_slice(&hash.to_le_bytes());
}

/// Observations of the collider with orientation `q` entered through `entry`.
/// `before` is false where the rotation as given is documented to apply until
/// the next step (`WorldField`, `Restore`), so only the after-step reads are
/// compared.
fn scene(q: QuatFix, entry: Entry, before: bool) -> Obs {
    let mut o = Obs { values: Vec::new() };
    let mut w = world();
    let probe = w.add_body(RigidBody::new_dynamic(probe_at(), Fix128::ONE));
    match entry {
        Entry::NewStatic | Entry::SetPose | Entry::UpdateCache => {
            let c = collider(q, entry);
            let hit = collide_sphere_sdf(probe_at(), radius(), &c).expect("the probe overlaps");
            o.contact(hit.depth, hit.normal);
            w.add_sdf_collider(c);
        }
        Entry::FieldBeforeAdd => {
            w.add_sdf_collider(collider(q, entry));
        }
        Entry::WorldField => {
            let i = w.add_sdf_collider(collider(unit_of(q), entry));
            w.sdf_colliders[i].rotation = q;
        }
        Entry::Restore => {
            let unit = unit_of(q);
            w.add_sdf_collider(collider(unit, entry));
            let mut blob = w.snapshot_world();
            patch_once(&mut blob, &quat_bytes(unit), &quat_bytes(q));
            patch_once(
                &mut blob,
                &quat_bytes(unit.conjugate()),
                &quat_bytes(q.conjugate()),
            );
            reseal(&mut blob);
            w.restore_world(&blob).expect("the patched blob restores");
        }
    }
    if before {
        read_world(&w, &mut o);
    }
    w.step(dt());
    o.v(w.bodies[probe].position);
    w.bodies[probe].set_position(probe_at());
    read_world(&w, &mut o);
    o
}

fn check(entry: Entry) {
    let q = turn();
    // A rotation written to the world's collider (or restored) applies as
    // given until the next step: only the after-step reads are compared.
    let before = !matches!(entry, Entry::WorldField | Entry::Restore);
    let unit = scene(q, entry, before);
    for s in SCALES {
        let sq = scaled(q, fx(s));
        let got = scene(sq, entry, before);
        let reference = scene(sq.normalize(), entry, before);
        assert_eq!(
            got.values, reference.values,
            "{entry:?}, |q| = {s}: same bits as normalize(s·q)"
        );
        assert_eq!(got.values.len(), unit.values.len());
        for (i, (a, b)) in got.values.iter().zip(&unit.values).enumerate() {
            let d = (*a - *b).abs().to_f64();
            assert!(
                d <= NEAR_F32,
                "{entry:?}, |q| = {s}, value {i}: {} vs unit {} (|d| = {d:e})",
                a.to_f64(),
                b.to_f64()
            );
        }
    }
}

#[test]
fn new_static_stores_a_unit_rotation() {
    check(Entry::NewStatic);
}

#[test]
fn set_pose_stores_a_unit_rotation() {
    check(Entry::SetPose);
}

#[test]
fn update_cache_stores_a_unit_rotation() {
    check(Entry::UpdateCache);
}

#[test]
fn add_sdf_collider_stores_a_unit_rotation() {
    check(Entry::FieldBeforeAdd);
}

#[test]
fn step_brings_a_field_written_rotation_to_unit_length() {
    check(Entry::WorldField);
}

#[test]
fn step_brings_a_restored_rotation_to_unit_length() {
    check(Entry::Restore);
}

/// The restored rotation applies as restored until the next step (the
/// documented behaviour): with `|q| = 2` the probe loses the contact before
/// the step and has it back after.
#[test]
fn a_restored_rotation_applies_as_restored_until_the_next_step() {
    let q = scaled(turn(), fx(2.0));
    let mut w = world();
    let probe = w.add_body(RigidBody::new_dynamic(probe_at(), Fix128::ONE));
    w.add_sdf_collider(SdfCollider::new_static(field(), origin(), turn()));
    let mut blob = w.snapshot_world();
    patch_once(&mut blob, &quat_bytes(turn()), &quat_bytes(q));
    patch_once(
        &mut blob,
        &quat_bytes(turn().conjugate()),
        &quat_bytes(q.conjugate()),
    );
    reseal(&mut blob);
    w.restore_world(&blob).expect("the patched blob restores");
    assert!(w.sdf_contacts().is_empty(), "|q|^2 = 4 moves the ball away");
    w.step(dt());
    w.bodies[probe].set_position(probe_at());
    assert_eq!(w.sdf_contacts().len(), 1);
}

#[test]
fn every_entry_is_covered() {
    assert_eq!(ENTRIES.len(), 6);
    for e in ENTRIES {
        let _ = collider(turn(), e);
    }
}
