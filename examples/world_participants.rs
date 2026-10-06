//! A law that takes part in `PhysicsWorld`'s substep loop: a thruster with a
//! fuel tank pushes a body until the tank is empty, publishes the fuel it
//! burnt into a shared field, and is saved and restored with the world.
//!
//! ```bash
//! cargo run --example world_participants --features std
//! cargo run --example world_participants --features std,parallel
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{
    FieldLayout, FieldMode, ObservationSink, Observed, Participant, ParticipantFault,
    ParticipantKind, Port, PortId, StateError, SubstepCtx,
};

const KIND_THRUSTER: ParticipantKind = ParticipantKind::new(0x5448_5255);
const BURNT: PortId = PortId::new(1);

/// Pushes `body` with `thrust` along +x while fuel lasts; one unit of fuel
/// per second of thrust.
struct Thruster {
    body: usize,
    thrust: Fix128,
    fuel: Fix128,
    ports: [Port; 1],
}

impl Thruster {
    fn new(body: usize, thrust: Fix128, fuel: Fix128) -> Self {
        Self {
            body,
            thrust,
            fuel,
            ports: [Port::writes(BURNT)],
        }
    }
}

impl Participant for Thruster {
    fn kind(&self) -> ParticipantKind {
        KIND_THRUSTER
    }

    fn ports(&self) -> &[Port] {
        &self.ports
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let burn = if self.fuel < h { self.fuel } else { h };
        if burn > Fix128::ZERO {
            ctx.add_force(
                self.body,
                Vec3Fix::new(self.thrust, Fix128::ZERO, Fix128::ZERO),
            )
            .map_err(|_| ParticipantFault::InvalidState)?;
        }
        ctx.stage_field(BURNT)
            .map_err(|_| ParticipantFault::InvalidState)?
            .fill(burn);
        self.fuel = self.fuel - burn;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, self.fuel);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.fuel.hi.to_le_bytes());
        out.extend_from_slice(&self.fuel.lo.to_le_bytes());
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 16 {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 16,
                found: bytes.len(),
            })
        }
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let mut hi = [0u8; 8];
        let mut lo = [0u8; 8];
        hi.copy_from_slice(&bytes[0..8]);
        lo.copy_from_slice(&bytes[8..16]);
        self.fuel = Fix128 {
            hi: i64::from_le_bytes(hi),
            lo: u64::from_le_bytes(lo),
        };
    }
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w.declare_field(BURNT, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare the burnt-fuel field");
    w.add_participant(Box::new(Thruster::new(
        0,
        Fix128::from_int(2),
        Fix128::from_ratio(1, 2),
    )))
    .expect("register the thruster");
    w
}

fn fuel_of(w: &PhysicsWorld) -> Fix128 {
    match w.observe_participant(0) {
        Some(Observed::Exact(sink)) => sink.values()[0].1,
        other => panic!("the thruster is not observable: {other:?}"),
    }
}

fn main() {
    let dt = Fix128::from_ratio(1, 64);
    let mut w = world();
    println!(
        "[world_participants] registered {} participant(s): {:?}",
        w.participant_count(),
        w.participant_kinds()
    );

    for _ in 0..16 {
        w.try_step(dt).expect("step");
    }
    let saved = w.snapshot_world();
    let state = w.participant_state(0).expect("thruster state");
    println!(
        "[world_participants] after 0.25 s: x = {:.6}, v = {:.6}, fuel = {:.6}, burnt in last substep = {:.6}, state bytes = {}",
        w.bodies[0].position.x.to_f64(),
        w.bodies[0].velocity.x.to_f64(),
        fuel_of(&w).to_f64(),
        w.fields().value(BURNT).expect("field")[0].to_f64(),
        state.len()
    );

    for _ in 0..32 {
        w.try_step(dt).expect("step");
    }
    // the tank empties at t = 0.5 s: v = F·t/m = 2 · 0.5 = 1
    assert_eq!(fuel_of(&w), Fix128::ZERO);
    assert_eq!(w.bodies[0].velocity.x, Fix128::ONE);
    println!(
        "[world_participants] after 0.75 s: x = {:.6}, v = {:.6} (tank empty)",
        w.bodies[0].position.x.to_f64(),
        w.bodies[0].velocity.x.to_f64()
    );

    // a branch restored from the snapshot runs the same 32 steps bit for bit
    let mut branch = world();
    branch.restore_world(&saved).expect("restore");
    for _ in 0..32 {
        #[cfg(feature = "parallel")]
        branch.try_step_parallel(dt).expect("step");
        #[cfg(not(feature = "parallel"))]
        branch.try_step(dt).expect("step");
    }
    assert_eq!(branch.bodies[0].position, w.bodies[0].position);
    assert_eq!(branch.fault(), None);
    match branch.observe_body_checked(0) {
        Some(Observed::Exact(o)) => println!(
            "[world_participants] restored branch agrees: x = {:.6}",
            o.position.x.to_f64()
        ),
        other => panic!("the body is not observable: {other:?}"),
    }
    branch.clear_fault();
    branch
        .set_field(BURNT, &[Fix128::ZERO])
        .expect("reset the field");
}
