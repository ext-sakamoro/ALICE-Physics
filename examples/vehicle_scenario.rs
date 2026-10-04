//! Two cars in one lane: the follower closes in, brakes when the time to
//! collision drops below 2 s, and its stopping distance is measured. The run
//! is recorded, encoded to bytes, decoded and replayed into a freshly built
//! scenario; the replay reproduces every state bit for bit.
//!
//! ```bash
//! cargo run --release --example vehicle_scenario
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle_dynamics::scenario::{
    longitudinal_state, time_headway, time_to_collision, up_from_gravity, Recording, ReplayError,
    Scenario, ScenarioVehicle, StoppingDistanceMeter, VehicleSnapshot, RECORDING_MAGIC,
    RECORDING_VERSION,
};
use alice_physics::vehicle_dynamics::surface::{FlatGround, RoadCondition};
use alice_physics::vehicle_dynamics::{DriverInput, DynamicVehicle, DynamicVehicleConfig};

const CAR_LENGTH: f64 = 4.5;

/// Same definition every time: world, road, two cars 40 m apart.
fn build() -> Scenario<FlatGround> {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    // The default frame damping keeps 0.99 of the velocity per frame (a
    // 0.6 /s drag at 60 Hz); the vehicle model carries its own resistances.
    world.config.damping = Fix128::ONE;
    let mut sc = Scenario::new(
        world,
        FlatGround {
            height: Fix128::ZERO,
        },
        RoadCondition::dry_asphalt(),
        Fix128::from_ratio(1, 60),
    );
    for z in [0.0, 40.0] {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(0.78), Fix128::from_f64(z)),
            Fix128::from_int(1200),
        );
        sc.add_vehicle(
            DynamicVehicle::new(DynamicVehicleConfig::passenger_car()),
            body,
        );
    }
    sc
}

fn state(sc: &Scenario<FlatGround>) -> Vec<(Vec3Fix, Vec3Fix)> {
    (0..sc.vehicles.len())
        .filter_map(|i| sc.chassis(i).map(|b| (b.position, b.velocity)))
        .collect()
}

fn main() {
    let mut sc = build();
    // Settle on the suspension, then give the follower 20 m/s, the leader 8.
    sc.run_tracks(&[], 60);
    let speeds = [20.0, 8.0];
    for (v, &speed) in sc.vehicles.iter_mut().zip(&speeds) {
        let ScenarioVehicle { vehicle, body } = v;
        sc.world.bodies[*body].velocity =
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_f64(speed));
        let r = vehicle.config.base.wheels[0].radius;
        for w in &mut vehicle.wheels {
            w.omega = Fix128::from_f64(speed) / r;
        }
    }

    sc.start_recording();
    let dir = Vec3Fix::UNIT_Z;
    let extent = Fix128::from_f64(CAR_LENGTH);
    let up = up_from_gravity(sc.world.config.gravity).expect("gravity");
    let mut meter: Option<StoppingDistanceMeter> = None;
    let mut braking = false;
    for _ in 0..600 {
        let rear = *sc.chassis(0).expect("follower");
        let front = *sc.chassis(1).expect("leader");
        let (gap, v_rear, v_front) = longitudinal_state(&rear, &front, dir, extent);
        let ttc = time_to_collision(gap, v_rear, v_front);
        if !braking && ttc.is_some_and(|t| t < Fix128::from_int(2)) {
            braking = true;
            meter = Some(StoppingDistanceMeter::new(&rear, up, Fix128::ZERO));
            println!(
                "frame {}: {:.2} m/s, TTC {:.2} s, headway {:.2} s, gap {:.2} m -> brake",
                sc.frame,
                DynamicVehicle::forward_speed(&rear).to_f64(),
                ttc.map_or(f64::INFINITY, Fix128::to_f64),
                time_headway(gap, v_rear).map_or(f64::INFINITY, Fix128::to_f64),
                gap.to_f64()
            );
        }
        let brake = braking;
        sc.step_with(1, |_frame, i, _veh, _body| {
            if i == 0 && brake {
                DriverInput {
                    brake: Fix128::ONE,
                    ..DriverInput::default()
                }
            } else {
                DriverInput {
                    throttle: Fix128::from_ratio(1, 10),
                    ..DriverInput::default()
                }
            }
        });
        if let (Some(m), Some(b)) = (meter.as_mut(), sc.chassis(0)) {
            if !m.is_stopped() && m.sample(b) {
                println!(
                    "stopped after {} frames, {:.2} m",
                    m.frames_to_stop().unwrap_or(0),
                    m.distance().map_or(0.0, Fix128::to_f64)
                );
            }
        }
    }
    if let Some(m) = meter.filter(|m| !m.is_stopped()) {
        println!("still moving, {:.2} m so far", m.travelled().to_f64());
    }
    if let Some(t) = sc.chassis(1).map(|front| {
        let rear = sc.chassis(0).expect("follower");
        let (gap, v_rear, _) = longitudinal_state(rear, front, dir, extent);
        time_headway(gap, v_rear)
    }) {
        println!("final headway {:?}", t.map(Fix128::to_f64));
    }

    assert!(sc.is_recording());
    let mut rec = sc.stop_recording().expect("recording");
    let bytes = rec.to_bytes();
    assert_eq!(bytes[..4], RECORDING_MAGIC);
    println!(
        "recording v{RECORDING_VERSION}: {} frames x {} vehicles, dt {} s, {} bytes",
        rec.frame_count(),
        rec.vehicle_count(),
        rec.dt().to_f64(),
        bytes.len()
    );
    let wheels: usize = rec
        .vehicles()
        .iter()
        .map(|v: &VehicleSnapshot| v.wheels.len())
        .sum();
    println!(
        "initial state: {wheels} wheel states, first inputs {:?}",
        rec.inputs(0).map(<[_]>::len)
    );

    let decoded = Recording::from_bytes(&bytes).expect("decode");
    let mut fresh = build();
    decoded.replay(&mut fresh).expect("replay");
    let exact = state(&fresh) == state(&sc)
        && fresh
            .vehicles
            .iter()
            .zip(&sc.vehicles)
            .all(|(a, b)| a.vehicle.wheels == b.vehicle.wheels);
    println!("replay bit-exact: {exact}");
    assert!(exact);

    // An edited recording replays to a different end state.
    if let Some(inp) = rec.input_mut(rec.frame_count() / 2, 0) {
        inp.steering = Fix128::from_ratio(1, 10);
    }
    let mut edited = build();
    rec.restore(&mut edited).expect("restore");
    for k in 0..rec.frame_count() {
        edited.step(rec.inputs(k).unwrap_or(&[]));
    }
    println!("edited replay differs: {}", state(&edited) != state(&sc));

    // A truncated byte stream is rejected.
    let err = Recording::from_bytes(&bytes[..bytes.len() - 1]);
    assert_eq!(err, Err(ReplayError::Truncated));
    println!("truncated bytes: {err:?}");
}
