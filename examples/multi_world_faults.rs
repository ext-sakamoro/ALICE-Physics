//! Stepping several worlds and reading each world's result: one world holds
//! a participant that fails on its second substep, and `try_step_all`
//! reports that fault for that world only while the others keep stepping.
//!
//! Run with `cargo run --example multi_world_faults` (add
//! `--features parallel` to step the worlds on several threads).

use alice_physics::multi_world::MultiWorld;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use alice_physics::{Fix128, PhysicsConfig, RigidBody, Vec3Fix};

/// Fails on its second call.
struct FailsOnce {
    calls: u64,
}

impl Participant for FailsOnce {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"FAIL"))
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        if self.calls == 1 {
            return Err(ParticipantFault::OutOfRange);
        }
        self.calls += 1;
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
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

fn worlds() -> Result<MultiWorld, String> {
    let mut m = MultiWorld::new();
    for i in 0..3 {
        let w = m.add_world(PhysicsConfig::default());
        m.worlds[w].add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(i), Fix128::from_int(10), Fix128::ZERO),
            Fix128::ONE,
        ));
    }
    m.worlds[1]
        .add_participant(Box::new(FailsOnce { calls: 0 }))
        .map_err(|e| format!("{e:?}"))?;
    Ok(m)
}

fn main() -> Result<(), String> {
    let dt = Fix128::from_ratio(1, 60);
    let mut m = worlds()?;
    for frame in 0..3 {
        for (i, result) in m.try_step_all(dt).iter().enumerate() {
            match result {
                Ok(()) => println!("frame {frame}, world {i}: ok"),
                Err(e) => println!("frame {frame}, world {i}: {e:?}"),
            }
        }
    }
    #[cfg(feature = "parallel")]
    {
        let mut p = worlds()?;
        let results = p.try_step_all_parallel(dt);
        println!(
            "parallel, first frame: {} worlds failed",
            results.iter().filter(|r| r.is_err()).count()
        );
    }
    Ok(())
}
