//! Snapshot version 2 round trip with a [`DragMedium`], a participant with
//! its own state and a shared field that feeds back into the dynamics.
//!
//! The scene: gravity, a large static sphere as the floor, five bodies of
//! masses `1 : 7 : 0.3 : 2.5 : 0.75` (two joined by a ball joint), a medium
//! coupled to four of them, and a `Tracker` that integrates every body's
//! `x` displacement into a per-body [`FieldMode::Replace`] field and pushes
//! the body back by a force proportional to that field, with a sign that
//! flips with its own substep counter. Every piece of state (bodies, medium
//! momentum, the tracker's counter, the field) changes the later steps, so a
//! snapshot that dropped any of them would not continue bit for bit.
//!
//! Run `N = 53` frames, snapshot, run `M = 71` frames recording the whole
//! world blob after every frame; then restore the snapshot into the same
//! world and into fresh worlds (participants of the same configuration but
//! other state, the field set to other values) and run `M` again: every
//! frame's blob, and the medium's and tracker's payloads, are the same bytes.

#![cfg(feature = "std")]

use alice_physics::coupling_medium::DragMedium;
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::world_participant::{
    FieldLayout, FieldMode, ObservationSink, Participant, ParticipantFault, ParticipantKind, Port,
    PortId, StateError, SubstepCtx,
};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const N: usize = 53;
const M: usize = 71;
const FIELD: PortId = PortId::new(0x7472_6163);
const BODIES: usize = 6;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 45)
}

/// Integrates `v_x · h` into the field and pushes back with
/// `F_x = ∓k · field` (sign from the parity of its substep counter).
struct Tracker {
    counter: u64,
    ports: Vec<Port>,
}

impl Tracker {
    fn new(counter: u64) -> Self {
        Self {
            counter,
            ports: vec![Port::writes(FIELD)],
        }
    }
}

impl Participant for Tracker {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"TRAK"))
    }
    fn ports(&self) -> &[Port] {
        &self.ports
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let vx: Vec<Fix128> = ctx.bodies().iter().map(|b| b.velocity.x).collect();
        let k = if self.counter % 2 == 0 {
            fx(-0.8)
        } else {
            fx(0.5)
        };
        let mut forces = Vec::with_capacity(vx.len());
        {
            let staged = ctx
                .stage_field(FIELD)
                .map_err(|_| ParticipantFault::InvalidState)?;
            for (s, &v) in staged.iter_mut().zip(&vx) {
                *s = *s + v * h;
                forces.push(*s * k);
            }
        }
        for (i, f) in forces.into_iter().enumerate() {
            ctx.add_force(i, Vec3Fix::new(f, Fix128::ZERO, Fix128::ZERO))
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        self.counter += 1;
        Ok(())
    }
    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, Fix128::from_int(self.counter as i64));
    }
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.counter.to_le_bytes());
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 8 {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 8,
                found: bytes.len(),
            })
        }
    }
    fn read_state(&mut self, bytes: &[u8]) {
        let mut b = [0u8; 8];
        b.copy_from_slice(bytes);
        self.counter = u64::from_le_bytes(b);
    }
}

fn medium(velocity: Vec3Fix) -> DragMedium {
    let mut m = DragMedium::new(fx(0.9), velocity).expect("medium");
    for (b, c) in [(3, 1.7), (1, 0.6), (5, 0.45), (2, 2.9)] {
        m.couple(b, fx(c)).expect("couple");
    }
    m
}

fn config(backend: SolverBackend) -> PhysicsConfig {
    PhysicsConfig {
        substeps: 5,
        iterations: 4,
        solver_backend: backend,
        ..PhysicsConfig::default()
    }
}

/// Participants and the field on an empty world (a restore brings the
/// bodies); `seed` changes their state, not their configuration.
fn attach(w: &mut PhysicsWorld, seed: i64) {
    w.declare_field(
        FIELD,
        FieldLayout::PerBody { bodies: BODIES },
        FieldMode::Replace,
    )
    .expect("field");
    let values: Vec<Fix128> = (0..BODIES as i64)
        .map(|i| Fix128::from_ratio(seed * (i + 1), 7))
        .collect();
    w.set_field(FIELD, &values).expect("set");
    w.add_participant(Box::new(medium(Vec3Fix::from_int(seed, -2 * seed, 1))))
        .expect("medium");
    w.add_participant(Box::new(Tracker::new(seed as u64)))
        .expect("tracker");
}

fn scene(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(config(backend));
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, -400.0, 0.0)), fx(400.0));
    let masses = [1.0, 7.0, 0.3, 2.5, 0.75];
    let vel = [
        [1.25, 0.0, -0.5],
        [-0.75, 0.5, 0.25],
        [2.5, 1.0, 0.0],
        [0.0, -0.5, 1.5],
        [-1.5, 2.0, -0.75],
    ];
    for (k, (&m, v)) in masses.iter().zip(vel).enumerate() {
        let mut b = RigidBody::new_dynamic(
            v3(
                -4.0 + 2.25 * k as f64,
                0.6 + 0.5 * (k % 3) as f64,
                0.3 * k as f64,
            ),
            fx(m),
        );
        b.velocity = v3(v[0], v[1], v[2]);
        w.add_body_with_radius(b, fx(0.35));
    }
    w.add_joint(Joint::Ball(BallJoint::new(
        2,
        3,
        v3(1.1, 0.0, 0.0),
        v3(-1.1, 0.0, 0.0),
    )));
    attach(&mut w, 3);
    w
}

#[derive(Clone, Copy, Debug)]
enum Path {
    Step,
    #[cfg(feature = "parallel")]
    Parallel,
}

fn advance(w: &mut PhysicsWorld, path: Path) {
    match path {
        Path::Step => w.try_step(dt()).expect("step"),
        #[cfg(feature = "parallel")]
        Path::Parallel => w.try_step_parallel(dt()).expect("step"),
    }
}

fn record(w: &mut PhysicsWorld, path: Path) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    (0..M)
        .map(|_| {
            advance(w, path);
            (
                w.snapshot_world(),
                w.participant_state(0).expect("medium"),
                w.participant_state(1).expect("tracker"),
            )
        })
        .collect()
}

fn round_trip(backend: SolverBackend, path: Path) {
    let mut w = scene(backend);
    for _ in 0..N {
        advance(&mut w, path);
    }
    let blob = w.snapshot_world();
    let reference = record(&mut w, path);
    // The run moved the medium, the tracker and the field (none of them is
    // trivially preserved).
    assert_ne!(reference[0].1, reference[M - 1].1, "medium did not move");
    assert_ne!(reference[0].2, reference[M - 1].2, "tracker did not move");
    assert_ne!(
        w.fields().value(FIELD).expect("field")[1],
        Fix128::from_ratio(6, 7)
    );

    // Into the same world.
    w.restore_world(&blob).expect("restore same");
    let again = record(&mut w, path);
    for (f, (a, b)) in reference.iter().zip(&again).enumerate() {
        assert!(a == b, "{backend:?} {path:?} same world, frame {f}");
    }

    // Into a fresh world with the same configuration and other state.
    let mut fresh = scene(backend);
    for _ in 0..7 {
        advance(&mut fresh, path);
    }
    fresh.restore_world(&blob).expect("restore fresh");
    let other = record(&mut fresh, path);
    for (f, (a, b)) in reference.iter().zip(&other).enumerate() {
        assert!(a == b, "{backend:?} {path:?} fresh world, frame {f}");
    }

    // Into an empty world that only holds the participants and the field.
    let mut empty = PhysicsWorld::new(PhysicsConfig::default());
    attach(&mut empty, 11);
    empty.restore_world(&blob).expect("restore empty");
    let from_empty = record(&mut empty, path);
    for (f, (a, b)) in reference.iter().zip(&from_empty).enumerate() {
        assert!(a == b, "{backend:?} {path:?} empty world, frame {f}");
    }
}

#[test]
fn snapshot_v2_continues_bit_for_bit_on_both_backends() {
    round_trip(SolverBackend::Xpbd, Path::Step);
    round_trip(SolverBackend::Tgs, Path::Step);
}

#[cfg(feature = "parallel")]
#[test]
fn snapshot_v2_continues_bit_for_bit_on_the_batched_path() {
    round_trip(SolverBackend::Xpbd, Path::Parallel);
    round_trip(SolverBackend::Tgs, Path::Parallel);
}

/// Recompute the trailing FNV-1a 64 of a blob after an edit.
fn reseal(mut b: Vec<u8>) -> Vec<u8> {
    let n = b.len() - 8;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in &b[..n] {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

/// Offset of the payload of the participant of kind `kind` with payload
/// length `len` (`kind: u32`, `payload_len: u64`, payload; little endian).
fn payload_at(blob: &[u8], kind: &[u8; 4], len: u64) -> usize {
    let mut key = u32::from_be_bytes(*kind).to_le_bytes().to_vec();
    key.extend_from_slice(&len.to_le_bytes());
    let hits: Vec<usize> = blob
        .windows(key.len())
        .enumerate()
        .filter(|(_, w)| *w == key.as_slice())
        .map(|(i, _)| i + key.len())
        .collect();
    assert_eq!(hits.len(), 1, "participant header {kind:?} not unique");
    hits[0]
}

fn diverges(blob: &[u8], reference: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> bool {
    let mut x = PhysicsWorld::new(PhysicsConfig::default());
    attach(&mut x, 5);
    x.restore_world(blob).expect("restore edited blob");
    let run = record(&mut x, Path::Step);
    run.iter().zip(reference).any(|(a, b)| a.0 != b.0)
}

/// Each piece of participant-side state matters: a blob whose medium
/// momentum or field differs by one raw unit, or whose tracker counter
/// differs by one, continues differently (so the bit-identical continuation
/// above is not a property of a scene that ignores them). The unedited blob
/// continues identically through the same helper.
#[test]
fn every_piece_of_state_changes_the_continuation() {
    let mut w = scene(SolverBackend::Xpbd);
    for _ in 0..N {
        advance(&mut w, Path::Step);
    }
    let blob = w.snapshot_world();
    let reference = record(&mut w, Path::Step);
    assert!(!diverges(&blob, &reference), "unedited blob diverged");

    // Medium momentum P.x: payload offset 12 (version u32, digest u64), lo
    // word first.
    let at = payload_at(&blob, b"MEDM", 60);
    let mut b = blob.clone();
    b[at + 12 + 8] ^= 1;
    assert!(diverges(&reseal(b), &reference), "medium momentum ignored");

    // Tracker counter.
    let at = payload_at(&blob, b"TRAK", 8);
    let mut b = blob.clone();
    b[at] ^= 1;
    assert!(diverges(&reseal(b), &reference), "tracker counter ignored");

    // Field sample 2 nudged by one raw unit (set after the restore).
    let mut x = PhysicsWorld::new(PhysicsConfig::default());
    attach(&mut x, 5);
    x.restore_world(&blob).expect("restore");
    let mut values = x.fields().value(FIELD).expect("field").to_vec();
    values[2] = values[2] + Fix128::from_raw(0, 1);
    x.set_field(FIELD, &values).expect("set");
    let run = record(&mut x, Path::Step);
    assert!(
        run.iter().zip(&reference).any(|(a, b)| a.0 != b.0),
        "field ignored"
    );
}
