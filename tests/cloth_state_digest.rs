//! Oracle: a [`Cloth`] has one state digest — SHA-256 over every bit a step reads or
//! writes — that a replay reproduces exactly and any change moves.
//!
//! # What is measured
//!
//! Two cloths built the same and driven by the same script carry the same digest at
//! every frame; a third run agrees; any difference in state (a step, a grab target, a
//! mass, a configuration value, the tether) gives another digest; the digest is the
//! SHA-256 of the documented domain and `state_bytes`, recomputed here independently;
//! a cloth restored from a world snapshot has its original digest; the exact sleep is
//! invisible to it; and the script's digest at frame 300 is frozen for every target.

use alice_physics_core::cloth::{Cloth, ClothConfig, GrabId, OutlineDesc};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use sha2::{Digest as _, Sha256};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

fn at(x: f32, z: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, 0.0, z)
}

/// A block letter E, spacing 0.1, the site's mochi configuration and tether.
fn letter() -> Cloth {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    let omega = Fix128::from_f32(core::f32::consts::TAU / 0.5);
    Cloth::from_outline(&OutlineDesc {
        origin: Vec3Fix::ZERO,
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours: vec![vec![
            p(0, 0),
            p(10, 0),
            p(10, 2),
            p(2, 2),
            p(2, 4),
            p(10, 4),
            p(10, 6),
            p(2, 6),
            p(2, 8),
            p(10, 8),
            p(10, 10),
            p(0, 10),
        ]],
        spacing: r(1, 10),
        mass_per_particle: Fix128::ONE,
        config: ClothConfig {
            iterations: 4,
            substeps: 1,
            gravity: Vec3Fix::ZERO,
            damping: r(985, 1000),
            stretch_compliance: r(1, 100),
            bend_compliance: r(1, 4),
            ..ClothConfig::default()
        },
        tether: Some((omega * omega, omega * Fix128::from_f32(2.0 * 0.42))),
    })
    .expect("the E keeps triangles")
}

/// Grab the corner, drag it for 30 frames, throw it, run to `frames`; the digest after
/// every frame.
fn script(c: &mut Cloth, frames: usize) -> Vec<[u8; 32]> {
    let mut out = Vec::with_capacity(frames);
    let id = GrabId(7);
    c.grab(id, at(0.0, 0.0), r(34, 100))
        .expect("corner in reach");
    for f in 0..frames {
        if f < 30 {
            c.move_grab(id, at(-0.9 + 0.01 * f as f32, -0.5));
        } else if f == 30 {
            c.release_grab(id);
        }
        c.step(dt());
        out.push(c.state_digest());
    }
    out
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// oracle: the same script replays to the same digest at every frame, three times over.
#[test]
fn replay_gives_the_same_digest_every_frame() {
    let (mut a, mut b, mut c) = (letter(), letter(), letter());
    assert_eq!(a.state_digest(), b.state_digest());
    let (da, db, dc) = (
        script(&mut a, 300),
        script(&mut b, 300),
        script(&mut c, 300),
    );
    assert_eq!(da, db);
    assert_eq!(db, dc);
    // and the frames differ among themselves while the cloth moves
    assert_ne!(da[0], da[1]);
    assert_ne!(da[30], da[31]);
}

/// oracle: every kind of difference moves the digest — one more awake step, a grab
/// target, one inverse mass, a configuration value, the tether, a triangle.
#[test]
fn any_change_moves_the_digest() {
    let base = letter();
    let d0 = base.state_digest();
    let mut stepped = letter();
    stepped.grab(GrabId(1), at(0.0, 0.0), r(34, 100)).unwrap();
    stepped.move_grab(GrabId(1), at(-0.2, -0.1));
    stepped.step(dt());
    assert_ne!(stepped.state_digest(), d0, "a step that moved a particle");
    let mut grabbed = letter();
    grabbed.grab(GrabId(1), at(0.0, 0.0), r(34, 100)).unwrap();
    let d1 = grabbed.state_digest();
    assert_ne!(d1, d0, "a grab is state");
    grabbed.move_grab(GrabId(1), at(0.3, 0.0));
    assert_ne!(grabbed.state_digest(), d1, "a grab target is state");
    let mut moving = letter();
    moving.velocities[3] = at(0.0, 1.0);
    assert_ne!(moving.state_digest(), d0, "a velocity");
    let mut prev = letter();
    prev.prev_positions[3] = at(0.5, 0.5);
    assert_ne!(prev.state_digest(), d0, "a previous position");
    let mut pinned = letter();
    pinned.pin(3);
    assert_ne!(pinned.state_digest(), d0, "a pin");
    let mut mass = letter();
    mass.inv_masses[3] = r(1, 2);
    assert_ne!(mass.state_digest(), d0, "an inverse mass");
    let mut config = letter();
    config.config.damping = r(99, 100);
    assert_ne!(config.state_digest(), d0, "a configuration value");
    let mut tether = letter();
    tether.clear_rest_tether();
    assert_ne!(tether.state_digest(), d0, "the tether");
    let mut wind = letter();
    wind.wind = at(0.0, 1.0);
    assert_ne!(wind.state_digest(), d0, "the wind");
    let mut tri = letter();
    tri.triangles.swap(0, 1);
    assert_ne!(tri.state_digest(), d0, "the triangle order");
}

/// oracle: the digest is SHA-256(domain ‖ state_bytes) with the documented domain,
/// recomputed here with another SHA-256 call sequence.
#[test]
fn digest_is_sha256_of_domain_and_state_bytes() {
    assert_eq!(
        Cloth::CLOTH_DIGEST_DOMAIN,
        <[u8; 32]>::from(Sha256::digest(b"alice-physics/cloth-digest/v1")),
        "the domain is the hash of its name"
    );
    let mut c = letter();
    script(&mut c, 40);
    let mut h = Sha256::new();
    h.update(Cloth::CLOTH_DIGEST_DOMAIN);
    h.update(c.state_bytes());
    assert_eq!(<[u8; 32]>::from(h.finalize()), c.state_digest());
    // the bytes start with the particle count and hold every particle thrice plus a mass
    let n = c.particle_count();
    let bytes = c.state_bytes();
    assert_eq!(&bytes[..8], &(n as u64).to_le_bytes());
    assert!(bytes.len() > 8 + n * (48 * 3 + 16));
}

/// oracle: a cloth restored from a world snapshot has the digest it had, and its
/// section in the world blob is exactly its state bytes.
#[test]
fn restored_cloth_keeps_its_digest() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let i = w.add_cloth(letter());
    let id = w.cloth_grab(i, at(0.0, 0.0), r(34, 100)).unwrap();
    for f in 0..20 {
        w.cloth_move_grab(id, at(-0.5 + 0.02 * f as f32, -0.3));
        w.step(dt());
    }
    let d = w.cloths[i].state_digest();
    let blob = w.snapshot_world();
    let bytes = w.cloths[i].state_bytes();
    assert!(
        blob.windows(bytes.len()).any(|win| win == bytes.as_slice()),
        "the cloth's state bytes are its section of the world snapshot"
    );
    let mut t = PhysicsWorld::new(SolverConfig::default());
    t.restore_world(&blob).expect("restore");
    assert_eq!(t.cloths[i].state_digest(), d);
    // the restored cloth steps on to the same digest
    w.step(dt());
    t.step(dt());
    assert_eq!(t.cloths[i].state_digest(), w.cloths[i].state_digest());
}

/// oracle: the exact sleep changes no digest — sleep on and off agree every frame,
/// including after the fixed point is reached.
#[test]
fn sleep_is_invisible_to_the_digest() {
    let mut a = letter();
    let mut b = letter();
    b.set_sleep(false);
    assert_eq!(a.state_digest(), b.state_digest());
    let (da, db) = (script(&mut a, 600), script(&mut b, 600));
    assert_eq!(da, db);
    assert!(a.is_asleep() && !b.is_asleep());
    assert_eq!(a.state_digest(), b.state_digest());
}

/// Frozen digest of the script at frame 300; every CI target must agree. It moves only
/// with an announced change to the cloth's law or to `state_bytes`.
const SCRIPT_DIGEST_300: &str = "f72b6c695d1759943cc7aa96d661b64832203bd66e4ca936924315165f5bc98c";

/// oracle: the script's digest at frame 300 is the frozen value on every target.
#[test]
fn script_digest_is_frozen() {
    let mut c = letter();
    let d = script(&mut c, 300);
    assert_eq!(hex(&d[299]), SCRIPT_DIGEST_300);
}
