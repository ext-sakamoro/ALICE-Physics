//! `MultiWorld::try_step_all`: every world is stepped with
//! `PhysicsWorld::try_step` and its result is returned in world order, so a
//! participant fault in one world reaches the caller instead of being
//! dropped as `step_all` drops it.
//!
//! The oracle for the worlds themselves is `step_all`: `step` is `try_step`
//! with the result dropped, so after any number of frames every world of a
//! `try_step_all` container is bit for bit the same world of a `step_all`
//! container, with or without faults. The oracle for the results is the
//! world's own `try_step` on a copy of the same world.

use alice_physics::multi_world::MultiWorld;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepError,
    SubstepCtx, WorldFault,
};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const PROBE: ParticipantKind = ParticipantKind::new(u32::from_be_bytes(*b"PRBE"));

/// Pushes body 0 with a constant force and fails on call `fail_on` (never
/// when `None`); a call counter is its whole state.
struct Probe {
    calls: u64,
    fail_on: Option<u64>,
}

impl Participant for Probe {
    fn kind(&self) -> ParticipantKind {
        PROBE
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        if self.fail_on == Some(self.calls) {
            return Err(ParticipantFault::OutOfRange);
        }
        ctx.add_force(0, Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO))
            .map_err(|_| ParticipantFault::InvalidState)?;
        self.calls += 1;
        Ok(())
    }
    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, Fix128::from_int(self.calls as i64));
    }
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.calls.to_le_bytes());
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
        self.calls = u64::from_le_bytes(b);
    }
}

/// World `i` of the scene: a falling body, the probe on worlds 0 to 2, and
/// on world 1 the probe fails on its 6th call (frame 2 with 4 substeps).
fn scene() -> MultiWorld {
    let mut m = MultiWorld::new();
    for i in 0..4_usize {
        let w = m.add_world(PhysicsConfig {
            substeps: 4,
            ..PhysicsConfig::default()
        });
        let world = &mut m.worlds[w];
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_int(i as i64),
                Fix128::from_int(10),
                Fix128::ZERO,
            ),
            Fix128::ONE,
        ));
        if i < 3 {
            let fail_on = (i == 1).then_some(5);
            world
                .add_participant(Box::new(Probe { calls: 0, fail_on }))
                .expect("register");
        }
    }
    m
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn snapshots(m: &MultiWorld) -> Vec<Vec<u8>> {
    m.worlds.iter().map(PhysicsWorld::snapshot_world).collect()
}

/// oracle (results): frame 1 is `Ok` everywhere; in frame 2 world 1's probe
/// fails (call 5) and that world returns `FaultRaised` naming the probe while
/// the others stay `Ok`; from frame 3 world 1 returns `Faulted` with the same
/// fault. Each result equals the one `try_step` gives on a world built the
/// same way and stepped alone.
#[test]
fn each_world_returns_its_own_result() {
    let mut m = scene();
    let mut alone = scene();
    let fault = WorldFault::Participant {
        index: 0,
        kind: PROBE,
        fault: ParticipantFault::OutOfRange,
    };
    for frame in 0..5 {
        let results = m.try_step_all(dt());
        let want: Vec<Result<(), StepError>> =
            alone.worlds.iter_mut().map(|w| w.try_step(dt())).collect();
        assert_eq!(results, want, "frame {frame}");
        let world1 = match frame {
            0 => Ok(()),
            1 => Err(StepError::FaultRaised(fault)),
            _ => Err(StepError::Faulted(fault)),
        };
        assert_eq!(
            results,
            vec![Ok(()), world1, Ok(()), Ok(())],
            "frame {frame}"
        );
    }
}

/// oracle (worlds): after 30 frames every world of the `try_step_all`
/// container is bit for bit the same world of a `step_all` container,
/// including the faulted world 1 and the world without participants.
#[test]
fn worlds_match_step_all_bit_for_bit() {
    let mut a = scene();
    let mut b = scene();
    for _ in 0..30 {
        let _ = a.try_step_all(dt());
        b.step_all(dt());
    }
    assert_eq!(snapshots(&a), snapshots(&b));
    assert!(a.worlds[1].fault().is_some(), "world 1 must be faulted");
    assert!(a.worlds[0].fault().is_none());
}

/// oracle (degenerate): an empty container returns no results; a
/// non-positive `dt` returns `Ok` for every world and changes nothing (as
/// `try_step`).
#[test]
fn empty_container_and_non_positive_dt() {
    let mut empty = MultiWorld::new();
    assert!(empty.try_step_all(dt()).is_empty());
    let mut m = scene();
    let before = snapshots(&m);
    assert_eq!(m.try_step_all(Fix128::ZERO), vec![Ok(()); 4]);
    assert_eq!(m.try_step_all(-dt()), vec![Ok(()); 4]);
    assert_eq!(snapshots(&m), before);
}

/// oracle (parallel): `try_step_all_parallel` returns the sequential results
/// in world order and leaves the worlds bit for bit as `try_step_all` does,
/// on 1, 2 and 8 threads.
#[cfg(feature = "parallel")]
#[test]
fn parallel_matches_sequential() {
    let mut seq = scene();
    let mut seq_results = Vec::new();
    for _ in 0..10 {
        seq_results.push(seq.try_step_all(dt()));
    }
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        let (par, results) = pool.install(|| {
            let mut par = scene();
            let results: Vec<_> = (0..10).map(|_| par.try_step_all_parallel(dt())).collect();
            (par, results)
        });
        assert_eq!(results, seq_results, "{threads} threads");
        assert_eq!(snapshots(&par), snapshots(&seq), "{threads} threads");
    }
}
