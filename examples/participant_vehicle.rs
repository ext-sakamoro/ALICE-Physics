//! A vehicle driven from outside the world: the throttle, brake and
//! steering live in a world field that the caller writes between frames,
//! and the vehicle participant reads them in every substep.
//!
//! This is the loop an external controller (a person, a script or a
//! learner) runs: observe, decide, write the action field, step. Because
//! the actions are world state, the snapshot taken mid-drive carries the
//! action in force, and the drive replays bit for bit.
//!
//! Run with `cargo run --example participant_vehicle`.

use alice_physics::vehicle::{
    Vehicle, VehicleConfig, VehicleParticipant, VEHICLE_INPUTS, VEHICLE_INPUT_BRAKE,
    VEHICLE_INPUT_STEERING, VEHICLE_INPUT_THROTTLE, VEHICLE_OBS_BRAKE, VEHICLE_OBS_ENGINE_RPM,
    VEHICLE_OBS_GEAR, VEHICLE_OBS_GROUNDED, VEHICLE_OBS_SPEED_KMH, VEHICLE_OBS_STEERING,
    VEHICLE_OBS_THROTTLE,
};
use alice_physics::world_participant::{FieldLayout, FieldMode, Observed, PortId};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const ACTIONS: PortId = PortId::new(1);

fn main() -> Result<(), String> {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world
        .declare_field(
            ACTIONS,
            FieldLayout::Channels {
                count: VEHICLE_INPUTS,
            },
            FieldMode::Replace,
        )
        .map_err(|e| e.to_string())?;
    let config = VehicleConfig::default();
    let wheel = config.wheels[0];
    let start_y = wheel.suspension_rest + wheel.radius - wheel.local_position.y;
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(1200),
    ));
    let resident = VehicleParticipant::new(Vehicle::new(config), chassis).with_inputs(ACTIONS);
    println!(
        "{} wheels, aero drag {:.2}",
        resident.vehicle().config.wheels.len(),
        resident.vehicle().config.aero_drag.to_f64()
    );
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("{e:?}"))?;

    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..360 {
        // decide the action for the next frame (here: a fixed script)
        let (throttle, brake, steering) = match frame {
            0..=59 => (0.0, 0.0, 0.0),
            60..=239 => (0.8, 0.0, if frame >= 180 { 0.3 } else { 0.0 }),
            _ => (0.0, 0.5, 0.0),
        };
        let mut action = [Fix128::ZERO; VEHICLE_INPUTS];
        action[VEHICLE_INPUT_THROTTLE] = Fix128::from_f64(throttle);
        action[VEHICLE_INPUT_BRAKE] = Fix128::from_f64(brake);
        action[VEHICLE_INPUT_STEERING] = Fix128::from_f64(steering);
        world
            .set_field(ACTIONS, &action)
            .map_err(|e| format!("{e:?}"))?;
        world.try_step(dt).map_err(|e| format!("{e:?}"))?;

        if frame % 60 == 59 {
            if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
                let get = |c: u32| {
                    sink.values()
                        .iter()
                        .find(|(k, _)| *k == c)
                        .map_or(0.0, |(_, v)| {
                            if v.is_negative() {
                                -(-*v).to_f64()
                            } else {
                                v.to_f64()
                            }
                        })
                };
                println!(
                    "t = {:>4.1} s  speed {:>6.2} km/h  rpm {:>7.1}  gear {}  throttle {:.2}  brake {:.2}  steering {:+.2}  wheels down {}",
                    (frame + 1) as f64 / 60.0,
                    get(VEHICLE_OBS_SPEED_KMH),
                    get(VEHICLE_OBS_ENGINE_RPM),
                    get(VEHICLE_OBS_GEAR),
                    get(VEHICLE_OBS_THROTTLE),
                    get(VEHICLE_OBS_BRAKE),
                    get(VEHICLE_OBS_STEERING),
                    get(VEHICLE_OBS_GROUNDED),
                );
            }
        }
    }
    Ok(())
}
