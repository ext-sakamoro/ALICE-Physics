//! A law written against the participant contract
//! ([`alice_physics::world_participant`]) and driven by hand through the
//! substep context, the way the world will drive it: a point oscillator that
//! pulls a body along y, its snapshot payload, a failing call that leaves it
//! unchanged, and the checks a restore makes on the participant list.
//!
//! Run: `cargo run --example world_participant_contract`

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use alice_physics::world_participant::{
    AccumulateError, ForceAccumulator, ObservationSink, Observed, Participant, ParticipantFault,
    ParticipantKind, ParticipantMismatch, RegisterError, StateError, StepError, StepRule,
    SubstepCtx, Verdict, WorldFault,
};

const KIND: ParticipantKind = ParticipantKind::new(1);

/// `x'' = -k x` (symplectic Euler); pulls `target` with `c (x - y_body)`.
struct Oscillator {
    target: usize,
    x: Fix128,
    v: Fix128,
    limit: Fix128,
}

impl Participant for Oscillator {
    fn kind(&self) -> ParticipantKind {
        KIND
    }

    fn step_rule(&self) -> StepRule {
        StepRule::FollowSubstep
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let v = self.v - Fix128::from_int(4) * self.x * h;
        let x = self.x + v * h;
        if x > self.limit {
            // nothing committed: the participant stays as it was
            return Err(ParticipantFault::OutOfRange);
        }
        let y = ctx
            .bodies()
            .get(self.target)
            .ok_or(ParticipantFault::InvalidState)?
            .position
            .y;
        ctx.add_force(
            self.target,
            Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2) * (x - y), Fix128::ZERO),
        )
        .map_err(|_: AccumulateError| ParticipantFault::InvalidState)?;
        ctx.add_torque(self.target, Vec3Fix::ZERO)
            .map_err(|_| ParticipantFault::InvalidState)?;
        self.x = x;
        self.v = v;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, self.x);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        for f in [self.x, self.v] {
            out.extend_from_slice(&f.hi.to_le_bytes());
            out.extend_from_slice(&f.lo.to_le_bytes());
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 32 {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 32,
                found: bytes.len(),
            })
        }
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let fix = |b: &[u8]| Fix128 {
            hi: i64::from_le_bytes(b[0..8].try_into().expect("8 bytes")),
            lo: u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")),
        };
        self.x = fix(&bytes[0..16]);
        self.v = fix(&bytes[16..32]);
    }
}

fn state(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

fn main() {
    let bodies = vec![RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 1, 0),
        Fix128::ONE,
    )];
    let mut p = Oscillator {
        target: 0,
        x: Fix128::from_int(2),
        v: Fix128::ZERO,
        limit: Fix128::from_int(10),
    };
    let substeps = 4;
    let h = Fix128::from_ratio(1, 240);
    let per = p
        .step_rule()
        .steps_per_substep(h)
        .expect("follows the substep");
    println!(
        "[world_participant] kind={} internal steps per substep={per}",
        p.kind().get()
    );

    // One frame: each substep stages into its own accumulator, merged on Ok.
    let mut frame = ForceAccumulator::new(bodies.len());
    for i in 0..substeps {
        let mut staged = ForceAccumulator::new(bodies.len());
        let mut ctx =
            SubstepCtx::new(&bodies, &mut staged, i, substeps, h).expect("one slot per body");
        assert_eq!(
            (ctx.substep_index(), ctx.substeps(), ctx.h()),
            (i, substeps, h)
        );
        if p.substep(&mut ctx, h).is_ok() {
            frame.merge(&staged).expect("same body count");
        }
    }
    println!(
        "[world_participant] frame force on body 0 = {:?} torque = {:?} (slots {}, empty {})",
        frame.force(0).map(|f| f.y.to_f64()),
        frame.torque(0).map(|t| t.y.to_f64()),
        frame.len(),
        frame.is_empty()
    );
    frame.clear();

    // Snapshot payload round trip.
    let blob = state(&p);
    let mut copy = Oscillator {
        x: Fix128::ZERO,
        v: Fix128::ZERO,
        ..p
    };
    copy.check_state(&blob).expect("own payload");
    copy.read_state(&blob);
    println!(
        "[world_participant] payload {} bytes, round trip equal = {}",
        blob.len(),
        state(&copy) == blob
    );
    println!(
        "[world_participant] short payload = {:?}",
        copy.check_state(&blob[..31])
    );

    // A failing call leaves the participant unchanged.
    let mut tight = Oscillator {
        limit: Fix128::from_int(-100),
        ..copy
    };
    let before = state(&tight);
    let mut staged = ForceAccumulator::new(bodies.len());
    let mut ctx = SubstepCtx::new(&bodies, &mut staged, 0, 1, h).expect("slots");
    let fault = tight.substep(&mut ctx, h).expect_err("out of range");
    println!(
        "[world_participant] fault={fault:?} tag={} unchanged={}",
        fault.tag(),
        state(&tight) == before
    );
    assert_eq!(ParticipantFault::from_tag(fault.tag()), Some(fault));
    let recorded = WorldFault {
        index: 0,
        kind: KIND,
        fault,
    };
    let refused = StepError::Faulted(recorded);
    let mut sink = ObservationSink::new();
    tight.observe(&mut sink);
    let seen = Observed::Exact(sink.values()[0].1);
    let while_faulted: Observed<Fix128> = Observed::Undecided;
    println!(
        "[world_participant] {refused:?}: x<10 exact={:?} faulted={:?}",
        seen.verdict(|x| *x < Fix128::from_int(10)),
        while_faulted.verdict(|_| true)
    );
    assert_eq!(while_faulted.verdict(|_| true), Verdict::Undecided);

    // What a restore checks on the participant list, and a registration a
    // fixed-step law cannot have.
    let other = ParticipantKind::new(2);
    for (snapshot, world) in [
        (vec![KIND, other], vec![KIND, other]),
        (vec![KIND], vec![KIND, other]),
        (vec![KIND, other], vec![KIND, KIND]),
        (vec![KIND, other], vec![other, KIND]),
    ] {
        let verdict: Option<ParticipantMismatch> = ParticipantMismatch::classify(&snapshot, &world);
        println!("[world_participant] restore check {verdict:?}");
    }
    let rule: Result<u64, RegisterError> =
        StepRule::Fixed(Fix128::from_ratio(2, 3)).steps_per_substep(Fix128::ONE);
    println!(
        "[world_participant] Fixed(2/3) in a substep of 1 = {rule:?} (Subcycle: {:?})",
        StepRule::Subcycle.steps_per_substep(Fix128::ONE)
    );
}
