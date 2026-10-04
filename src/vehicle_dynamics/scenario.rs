//! Scenarios: several [`DynamicVehicle`]s in one [`PhysicsWorld`], following
//! metrics (time to collision, time headway), a stopping-distance meter and a
//! lossless input replay.
//!
//! # Runner
//!
//! [`Scenario`] owns the world, the road, the road condition and the
//! vehicles. One frame ([`Scenario::step`]):
//!
//! 1. every vehicle gets its [`DriverInput`] for the frame (decided before
//!    any vehicle is updated, so a closed-loop controller in
//!    [`Scenario::step_with`] sees the state at the start of the frame for
//!    every vehicle)
//! 2. [`DynamicVehicle::update`] runs for every vehicle **in index order**
//!    (`vehicles[0]`, `vehicles[1]`, …) with `Environment::time = time`
//! 3. `PhysicsWorld::step(dt)` once, then `time += dt`, `frame += 1`
//!
//! Each `update` touches only its own chassis body, so vehicles that do not
//! interact through the world (no colliders between them) give bit-for-bit
//! the same result whatever their order or company.
//!
//! # Replay
//!
//! [`Scenario::start_recording`] snapshots the initial state — the world
//! state blob ([`PhysicsWorld::serialize_state`]: every body's position,
//! velocity, rotation, angular velocity and sleep state, raw `Fix128`), every
//! vehicle's wheel states, engine speed, gear and current input, plus `dt`,
//! `time` and `frame` — and from then on every frame appends the inputs that
//! were applied. [`Recording::restore`] puts that state back into a scenario
//! built from the same definition (same world configuration and bodies, same
//! vehicle configurations, road, condition and wind, which are not
//! recorded); stepping it with the recorded inputs ([`Recording::replay`])
//! reproduces every state bit for bit. [`Recording::to_bytes`] /
//! [`Recording::from_bytes`] store all values as raw `Fix128` (`hi` / `lo`),
//! never as floating point.
//!
//! Everything here is `Fix128` arithmetic on `alloc` only (no I/O).

use super::surface::{RoadCondition, RoadSurface};
use super::{DriverInput, DynamicVehicle, Environment, WheelDynamicsState};
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsWorld, RigidBody};
use crate::wind_zone::WindZone;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Following metrics
// ---------------------------------------------------------------------------

/// Time to collision of two vehicles on one line (s):
/// `max(gap, 0) / (v_rear − v_front)`.
///
/// `gap` is the bumper-to-bumper distance (m, negative when the bodies
/// already overlap), `v_rear` / `v_front` the speeds along the line (m/s).
/// `None` when the vehicles are not closing (`v_rear ≤ v_front`) or the
/// quotient is outside the `Fix128` range (no collision within ~9.2e18 s).
/// An overlap that is still closing gives `Some(0)`.
#[must_use]
pub fn time_to_collision(gap: Fix128, v_rear: Fix128, v_front: Fix128) -> Option<Fix128> {
    let closing = v_rear - v_front;
    if closing <= Fix128::ZERO {
        return None;
    }
    gap.max(Fix128::ZERO).checked_div(closing)
}

/// Time headway (s): `max(gap, 0) / v_rear`, the time the following vehicle
/// needs to reach the leader's current rear bumper. `None` when the
/// following vehicle does not move forward (`v_rear ≤ 0`) or the quotient is
/// outside the `Fix128` range.
#[must_use]
pub fn time_headway(gap: Fix128, v_rear: Fix128) -> Option<Fix128> {
    if v_rear <= Fix128::ZERO {
        return None;
    }
    gap.max(Fix128::ZERO).checked_div(v_rear)
}

/// Gap and speeds of two bodies along the unit direction `dir` (rear →
/// front): `((p_front − p_rear)·dir − extent, v_rear·dir, v_front·dir)`.
///
/// `extent` is the centre-to-centre distance at contact (front overhang of
/// the rear vehicle + rear overhang of the leader; the car length for two
/// identical cars with the body origin at mid-length). The result feeds
/// [`time_to_collision`] and [`time_headway`].
#[must_use]
pub fn longitudinal_state(
    rear: &RigidBody,
    front: &RigidBody,
    dir: Vec3Fix,
    extent: Fix128,
) -> (Fix128, Fix128, Fix128) {
    let gap = (front.position - rear.position).dot(dir) - extent;
    (gap, rear.velocity.dot(dir), front.velocity.dot(dir))
}

/// Unit up vector `−g / |g|` of a gravity vector, `None` for zero gravity.
#[must_use]
pub fn up_from_gravity(gravity: Vec3Fix) -> Option<Vec3Fix> {
    let len = gravity.length();
    if len.is_zero() {
        return None;
    }
    Some(-gravity / len)
}

/// Remove the component of `v` along the unit `up`.
fn horizontal(v: Vec3Fix, up: Vec3Fix) -> Vec3Fix {
    v - up * v.dot(up)
}

// ---------------------------------------------------------------------------
// Stopping distance
// ---------------------------------------------------------------------------

/// Stopping-distance meter: the horizontal path length travelled from the
/// frame braking starts until the vehicle stops.
///
/// Create it on the brake-on frame ([`Self::new`]) and call
/// [`Self::sample`] after every frame. The distance is the sum of the
/// per-frame displacements of the chassis origin with their component along
/// `up` removed (suspension dive and bounce are not travel); for a straight
/// stop it equals the displacement along the heading.
///
/// # Stop
///
/// The vehicle counts as stopped on the first sample whose horizontal
/// velocity, projected on the horizontal velocity direction of the previous
/// sample, is `≤ rest_speed` — the forward speed of the travel reached the
/// threshold or reversed within the frame. When the previous horizontal
/// velocity is exactly zero the criterion is `|v_h| ≤ rest_speed`. The
/// distance is latched there; later samples do not change it.
///
/// `rest_speed = 0` is the exact criterion and needs no tuning: a car on its
/// suspension does not settle to exactly zero velocity at the end of a stop
/// — the body pitches back from the brake dive while the tyre contacts are
/// held, so the chassis origin swings backward (measured: `−0.08 m/s` a few
/// frames after a 20 m/s locked stop, decaying over ~1 s) — and the sign
/// change of the forward speed marks the stop on the frame it happens. A
/// threshold only latches earlier: a car still decelerating at `a` when its
/// speed crosses `rest_speed` travels at most `rest_speed² / (2 a)` further
/// (the error added to the distance). Comparing against the previous
/// sample's direction rather than a fixed heading follows curved paths and
/// sliding cars whose body yaws away from the direction of travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoppingDistanceMeter {
    up: Vec3Fix,
    rest_speed: Fix128,
    last: Vec3Fix,
    last_velocity: Vec3Fix,
    travelled: Fix128,
    samples: u64,
    stopped_at: Option<u64>,
}

impl StoppingDistanceMeter {
    /// Start measuring at the current state of `chassis`. `up` must be a
    /// unit vector (see [`up_from_gravity`]); a negative `rest_speed` is
    /// treated as 0.
    #[must_use]
    pub fn new(chassis: &RigidBody, up: Vec3Fix, rest_speed: Fix128) -> Self {
        Self {
            up,
            rest_speed: rest_speed.max(Fix128::ZERO),
            last: chassis.position,
            last_velocity: horizontal(chassis.velocity, up),
            travelled: Fix128::ZERO,
            samples: 0,
            stopped_at: None,
        }
    }

    /// Add the frame that just ran. Returns `true` once the vehicle is
    /// stopped (and stays `true`).
    pub fn sample(&mut self, chassis: &RigidBody) -> bool {
        if self.stopped_at.is_some() {
            return true;
        }
        self.samples += 1;
        self.travelled =
            self.travelled + horizontal(chassis.position - self.last, self.up).length();
        self.last = chassis.position;
        let v = horizontal(chassis.velocity, self.up);
        let (dir, len) = self.last_velocity.normalize_with_length();
        let forward = if len.is_zero() {
            v.length()
        } else {
            v.dot(dir)
        };
        if forward <= self.rest_speed {
            self.stopped_at = Some(self.samples);
        }
        self.last_velocity = v;
        self.stopped_at.is_some()
    }

    /// Whether the vehicle has stopped.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.stopped_at.is_some()
    }

    /// Stopping distance (m), `None` until stopped.
    #[must_use]
    pub fn distance(&self) -> Option<Fix128> {
        self.stopped_at.map(|_| self.travelled)
    }

    /// Horizontal path length so far (m), stopped or not.
    #[must_use]
    pub fn travelled(&self) -> Fix128 {
        self.travelled
    }

    /// Number of samples (frames) from brake-on to the stop, `None` until
    /// stopped.
    #[must_use]
    pub fn frames_to_stop(&self) -> Option<u64> {
        self.stopped_at
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// One vehicle of a [`Scenario`] and the index of its chassis in
/// `Scenario::world.bodies`.
#[derive(Clone, Debug)]
pub struct ScenarioVehicle {
    /// The vehicle.
    pub vehicle: DynamicVehicle,
    /// Chassis body index in the world.
    pub body: usize,
}

/// Several vehicles in one world, stepped deterministically (see the module
/// doc for the frame order).
pub struct Scenario<R: RoadSurface> {
    /// The world (chassis bodies and anything else the caller adds).
    pub world: PhysicsWorld,
    /// Road surface shared by all vehicles.
    pub road: R,
    /// Road material, weather, rolling resistance.
    pub condition: RoadCondition,
    /// Wind acting on every vehicle, if any.
    pub wind: Option<WindZone>,
    /// Vehicles, in processing order.
    pub vehicles: Vec<ScenarioVehicle>,
    /// Frame length (s). `dt ≤ 0` makes [`Self::step`] a no-op.
    pub dt: Fix128,
    /// Simulation time (s), passed to `Environment::time`.
    pub time: Fix128,
    /// Frames stepped so far.
    pub frame: u64,
    recording: Option<Recording>,
}

impl<R: RoadSurface> Scenario<R> {
    /// Scenario without vehicles at `time = 0`, `frame = 0`, no wind.
    #[must_use]
    pub fn new(world: PhysicsWorld, road: R, condition: RoadCondition, dt: Fix128) -> Self {
        Self {
            world,
            road,
            condition,
            wind: None,
            vehicles: Vec::new(),
            dt,
            time: Fix128::ZERO,
            frame: 0,
            recording: None,
        }
    }

    /// Add `chassis` to the world and `vehicle` on it; returns the vehicle
    /// index (its position in the processing order).
    pub fn add_vehicle(&mut self, vehicle: DynamicVehicle, chassis: RigidBody) -> usize {
        let body = self.world.add_body(chassis);
        self.vehicles.push(ScenarioVehicle { vehicle, body });
        self.vehicles.len() - 1
    }

    /// Chassis body of vehicle `i`, `None` when out of range.
    #[must_use]
    pub fn chassis(&self, i: usize) -> Option<&RigidBody> {
        self.vehicles
            .get(i)
            .and_then(|v| self.world.bodies.get(v.body))
    }

    /// One frame with `inputs[i]` for vehicle `i`. Vehicles past the end of
    /// `inputs` get `DriverInput::default()` (pedals released, wheel
    /// straight); extra entries are ignored. With `dt ≤ 0` nothing happens
    /// (no update, no step, `frame` / `time` unchanged, nothing recorded).
    pub fn step(&mut self, inputs: &[DriverInput]) {
        if self.dt <= Fix128::ZERO {
            return;
        }
        for (i, v) in self.vehicles.iter_mut().enumerate() {
            v.vehicle.input = inputs.get(i).copied().unwrap_or_default();
        }
        self.advance();
    }

    /// `frames` frames whose inputs come from `control(frame, i, vehicle,
    /// chassis)`, called for every vehicle in index order with the state at
    /// the start of the frame (all inputs of a frame are decided before any
    /// vehicle moves). `frame` is the absolute [`Self::frame`] number.
    pub fn step_with<F>(&mut self, frames: usize, mut control: F)
    where
        F: FnMut(u64, usize, &DynamicVehicle, &RigidBody) -> DriverInput,
    {
        if self.dt <= Fix128::ZERO {
            return;
        }
        for _ in 0..frames {
            let frame = self.frame;
            let mut inputs = Vec::with_capacity(self.vehicles.len());
            for (i, v) in self.vehicles.iter().enumerate() {
                inputs.push(control(frame, i, &v.vehicle, &self.world.bodies[v.body]));
            }
            for (v, inp) in self.vehicles.iter_mut().zip(inputs) {
                v.vehicle.input = inp;
            }
            self.advance();
        }
    }

    /// `frames` frames where vehicle `i` takes `tracks[i][k]` on the `k`-th
    /// frame of this call. A missing track, or a track shorter than
    /// `frames`, gives `DriverInput::default()` for the missing frames
    /// (pedals released, wheel straight — not the last input held).
    pub fn run_tracks(&mut self, tracks: &[&[DriverInput]], frames: usize) {
        if self.dt <= Fix128::ZERO {
            return;
        }
        for k in 0..frames {
            for (i, v) in self.vehicles.iter_mut().enumerate() {
                v.vehicle.input = tracks
                    .get(i)
                    .and_then(|t| t.get(k))
                    .copied()
                    .unwrap_or_default();
            }
            self.advance();
        }
    }

    /// Update every vehicle with its current input, step the world, record.
    fn advance(&mut self) {
        let env = Environment {
            condition: &self.condition,
            wind: self.wind.as_ref(),
            time: self.time,
        };
        for v in &mut self.vehicles {
            v.vehicle
                .update(&mut self.world.bodies[v.body], &self.road, &env, self.dt);
        }
        self.world.step(self.dt);
        self.time = self.time + self.dt;
        self.frame += 1;
        if let Some(rec) = self.recording.as_mut() {
            rec.inputs
                .extend(self.vehicles.iter().map(|v| v.vehicle.input));
            rec.frames += 1;
        }
    }

    /// Start a new recording from the current state (replacing one in
    /// progress).
    pub fn start_recording(&mut self) {
        self.recording = Some(Recording {
            dt: self.dt,
            start_time: self.time,
            start_frame: self.frame,
            world_state: self.world.serialize_state(),
            vehicles: self
                .vehicles
                .iter()
                .map(|v| VehicleSnapshot {
                    body: v.body,
                    engine_rpm: v.vehicle.engine_rpm,
                    current_gear: v.vehicle.config.powertrain.current_gear,
                    input: v.vehicle.input,
                    wheels: v.vehicle.wheels.clone(),
                })
                .collect(),
            frames: 0,
            inputs: Vec::new(),
        });
    }

    /// Whether a recording is in progress.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    /// Stop recording and return it, `None` when none was in progress.
    pub fn stop_recording(&mut self) -> Option<Recording> {
        self.recording.take()
    }
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// First 4 bytes of [`Recording::to_bytes`].
pub const RECORDING_MAGIC: [u8; 4] = *b"AVSR";
/// Format version, bytes 4..6 (little endian) of [`Recording::to_bytes`].
pub const RECORDING_VERSION: u16 = 1;

/// Recorded state of one vehicle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VehicleSnapshot {
    /// Chassis body index.
    pub body: usize,
    /// Engine speed (rpm).
    pub engine_rpm: Fix128,
    /// `config.powertrain.current_gear`.
    pub current_gear: usize,
    /// Input held by the vehicle when recording started.
    pub input: DriverInput,
    /// Wheel states.
    pub wheels: Vec<WheelDynamicsState>,
}

/// Initial state + per-frame inputs of a scenario run (see the module doc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording {
    dt: Fix128,
    start_time: Fix128,
    start_frame: u64,
    world_state: Vec<u8>,
    vehicles: Vec<VehicleSnapshot>,
    frames: u64,
    /// Frame-major: `inputs[k * vehicles.len() + i]`.
    inputs: Vec<DriverInput>,
}

/// Why decoding or restoring a [`Recording`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// The bytes do not start with [`RECORDING_MAGIC`].
    BadMagic,
    /// A format version other than [`RECORDING_VERSION`].
    UnsupportedVersion(u16),
    /// The bytes end before the record does.
    Truncated,
    /// Bytes left after the record.
    TrailingBytes,
    /// A flag byte that is neither 0 nor 1.
    InvalidFlag,
    /// A count or index does not fit this platform's `usize`.
    Oversized,
    /// The scenario has another number of vehicles.
    VehicleCountMismatch {
        /// Vehicles in the recording.
        recorded: usize,
        /// Vehicles in the scenario.
        scenario: usize,
    },
    /// Vehicle `vehicle` sits on another chassis body index.
    BodyMismatch {
        /// Vehicle index.
        vehicle: usize,
        /// Recorded body index.
        recorded: usize,
        /// Body index in the scenario.
        scenario: usize,
    },
    /// Vehicle `vehicle` has another number of wheels.
    WheelCountMismatch {
        /// Vehicle index.
        vehicle: usize,
        /// Wheels in the recording.
        recorded: usize,
        /// Wheels in the scenario.
        scenario: usize,
    },
    /// `PhysicsWorld::deserialize_state` refused the recorded world state
    /// (other body count or world population).
    WorldStateRejected,
}

impl Recording {
    /// Number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames as usize
    }

    /// Number of vehicles.
    #[must_use]
    pub fn vehicle_count(&self) -> usize {
        self.vehicles.len()
    }

    /// Recorded vehicle states at the start.
    #[must_use]
    pub fn vehicles(&self) -> &[VehicleSnapshot] {
        &self.vehicles
    }

    /// Frame length (s) of the recorded run.
    #[must_use]
    pub fn dt(&self) -> Fix128 {
        self.dt
    }

    /// Inputs applied on recorded frame `k` (one per vehicle), `None` past
    /// the end.
    #[must_use]
    pub fn inputs(&self, k: usize) -> Option<&[DriverInput]> {
        if k >= self.frame_count() {
            return None;
        }
        let n = self.vehicles.len();
        Some(&self.inputs[k * n..(k + 1) * n])
    }

    /// Mutable input of vehicle `i` on frame `k` (editing a recording).
    pub fn input_mut(&mut self, k: usize, i: usize) -> Option<&mut DriverInput> {
        let n = self.vehicles.len();
        if k >= self.frame_count() || i >= n {
            return None;
        }
        self.inputs.get_mut(k * n + i)
    }

    /// Put the recorded initial state into `scenario`: world state, every
    /// vehicle's wheels / engine speed / gear / input, `dt`, `time`,
    /// `frame`. Shape mismatches are reported before anything is written,
    /// so an `Err` leaves `scenario` unchanged.
    ///
    /// # Errors
    ///
    /// [`ReplayError::VehicleCountMismatch`], [`ReplayError::BodyMismatch`],
    /// [`ReplayError::WheelCountMismatch`], [`ReplayError::WorldStateRejected`].
    pub fn restore<R: RoadSurface>(&self, scenario: &mut Scenario<R>) -> Result<(), ReplayError> {
        if scenario.vehicles.len() != self.vehicles.len() {
            return Err(ReplayError::VehicleCountMismatch {
                recorded: self.vehicles.len(),
                scenario: scenario.vehicles.len(),
            });
        }
        for (i, (snap, v)) in self.vehicles.iter().zip(&scenario.vehicles).enumerate() {
            if snap.body != v.body {
                return Err(ReplayError::BodyMismatch {
                    vehicle: i,
                    recorded: snap.body,
                    scenario: v.body,
                });
            }
            let wheels = v.vehicle.config.base.wheels.len();
            if snap.wheels.len() != wheels {
                return Err(ReplayError::WheelCountMismatch {
                    vehicle: i,
                    recorded: snap.wheels.len(),
                    scenario: wheels,
                });
            }
        }
        // `deserialize_state` validates the whole blob before writing.
        if !scenario.world.deserialize_state(&self.world_state) {
            return Err(ReplayError::WorldStateRejected);
        }
        for (snap, v) in self.vehicles.iter().zip(&mut scenario.vehicles) {
            v.vehicle.wheels.clone_from(&snap.wheels);
            v.vehicle.engine_rpm = snap.engine_rpm;
            v.vehicle.config.powertrain.current_gear = snap.current_gear;
            v.vehicle.input = snap.input;
        }
        scenario.dt = self.dt;
        scenario.time = self.start_time;
        scenario.frame = self.start_frame;
        Ok(())
    }

    /// [`Self::restore`], then step `scenario` through every recorded frame.
    ///
    /// # Errors
    ///
    /// Those of [`Self::restore`]; nothing is stepped then.
    pub fn replay<R: RoadSurface>(&self, scenario: &mut Scenario<R>) -> Result<(), ReplayError> {
        self.restore(scenario)?;
        for k in 0..self.frame_count() {
            if let Some(inputs) = self.inputs(k) {
                scenario.step(inputs);
            }
        }
        Ok(())
    }

    /// Encode as bytes (little endian, `Fix128` as `hi: i64` then `lo: u64`):
    ///
    /// | field | bytes |
    /// |---|---|
    /// | magic [`RECORDING_MAGIC`] | 4 |
    /// | version [`RECORDING_VERSION`] (u16) + reserved 0 (u16) | 4 |
    /// | `dt`, start time | 16 + 16 |
    /// | start frame (u64) | 8 |
    /// | world blob length (u32) + world blob | 4 + n |
    /// | vehicle count (u32) | 4 |
    /// | per vehicle: body (u64), engine rpm, gear (u64), input, wheel count (u32), wheels | 8 + 16 + 8 + 64 + 4 + 307 w |
    /// | frame count (u64) | 8 |
    /// | inputs, frame-major | 64 per vehicle per frame |
    ///
    /// An input is throttle, brake, handbrake, steering (4 `Fix128`). A
    /// wheel is `grounded` (u8 0/1), contact point (3), contact normal (3),
    /// compression, normal load, steer angle, omega, spin angle, slip ratio,
    /// slip tan α, longitudinal force, lateral force, brake torque (10
    /// `Fix128`), `abs_active` (u8), anchor tag (u8 0/1) + anchor (3
    /// `Fix128`, zero when absent): 307 bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&RECORDING_MAGIC);
        out.extend_from_slice(&RECORDING_VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        put_fix(&mut out, self.dt);
        put_fix(&mut out, self.start_time);
        out.extend_from_slice(&self.start_frame.to_le_bytes());
        out.extend_from_slice(&(self.world_state.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.world_state);
        out.extend_from_slice(&(self.vehicles.len() as u32).to_le_bytes());
        for v in &self.vehicles {
            out.extend_from_slice(&(v.body as u64).to_le_bytes());
            put_fix(&mut out, v.engine_rpm);
            out.extend_from_slice(&(v.current_gear as u64).to_le_bytes());
            put_input(&mut out, &v.input);
            out.extend_from_slice(&(v.wheels.len() as u32).to_le_bytes());
            for w in &v.wheels {
                put_wheel(&mut out, w);
            }
        }
        out.extend_from_slice(&self.frames.to_le_bytes());
        for inp in &self.inputs {
            put_input(&mut out, inp);
        }
        out
    }

    /// Decode [`Self::to_bytes`] output.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Truncated`] for any strict prefix (including fewer
    /// than 4 bytes), [`ReplayError::BadMagic`],
    /// [`ReplayError::UnsupportedVersion`], [`ReplayError::InvalidFlag`] for
    /// a flag byte other than 0 / 1, [`ReplayError::Oversized`] when a count
    /// does not fit `usize` or the input table would exceed the remaining
    /// bytes' addressable size, [`ReplayError::TrailingBytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != RECORDING_MAGIC {
            return Err(ReplayError::BadMagic);
        }
        let version = r.u16()?;
        if version != RECORDING_VERSION {
            return Err(ReplayError::UnsupportedVersion(version));
        }
        let _reserved = r.u16()?;
        let dt = r.fix()?;
        let start_time = r.fix()?;
        let start_frame = r.u64()?;
        let blob_len = r.u32()? as usize;
        let world_state = r.take(blob_len)?.to_vec();
        let n = r.u32()? as usize;
        let mut vehicles = Vec::new();
        for _ in 0..n {
            let body = usize::try_from(r.u64()?).map_err(|_| ReplayError::Oversized)?;
            let engine_rpm = r.fix()?;
            let current_gear = usize::try_from(r.u64()?).map_err(|_| ReplayError::Oversized)?;
            let input = r.input()?;
            let nw = r.u32()? as usize;
            let mut wheels = Vec::new();
            for _ in 0..nw {
                wheels.push(r.wheel()?);
            }
            vehicles.push(VehicleSnapshot {
                body,
                engine_rpm,
                current_gear,
                input,
                wheels,
            });
        }
        let frames = r.u64()?;
        let count = usize::try_from(frames)
            .ok()
            .and_then(|f| f.checked_mul(n))
            .ok_or(ReplayError::Oversized)?;
        // Check the length before allocating the table.
        let need = count
            .checked_mul(INPUT_BYTES)
            .ok_or(ReplayError::Oversized)?;
        if r.remaining() < need {
            return Err(ReplayError::Truncated);
        }
        let mut inputs = Vec::with_capacity(count);
        for _ in 0..count {
            inputs.push(r.input()?);
        }
        if r.remaining() != 0 {
            return Err(ReplayError::TrailingBytes);
        }
        Ok(Self {
            dt,
            start_time,
            start_frame,
            world_state,
            vehicles,
            frames,
            inputs,
        })
    }
}

/// Encoded size of one [`DriverInput`].
const INPUT_BYTES: usize = 64;

fn put_fix(out: &mut Vec<u8>, x: Fix128) {
    out.extend_from_slice(&x.hi.to_le_bytes());
    out.extend_from_slice(&x.lo.to_le_bytes());
}

fn put_vec(out: &mut Vec<u8>, v: Vec3Fix) {
    put_fix(out, v.x);
    put_fix(out, v.y);
    put_fix(out, v.z);
}

fn put_input(out: &mut Vec<u8>, i: &DriverInput) {
    put_fix(out, i.throttle);
    put_fix(out, i.brake);
    put_fix(out, i.handbrake);
    put_fix(out, i.steering);
}

fn put_wheel(out: &mut Vec<u8>, w: &WheelDynamicsState) {
    out.push(u8::from(w.grounded));
    put_vec(out, w.contact_point);
    put_vec(out, w.contact_normal);
    for x in [
        w.compression,
        w.normal_load,
        w.steer_angle,
        w.omega,
        w.spin_angle,
        w.slip_ratio,
        w.slip_tan_alpha,
        w.longitudinal_force,
        w.lateral_force,
        w.brake_torque,
    ] {
        put_fix(out, x);
    }
    out.push(u8::from(w.abs_active));
    match w.anchor {
        Some(a) => {
            out.push(1);
            put_vec(out, a);
        }
        None => {
            out.push(0);
            put_vec(out, Vec3Fix::ZERO);
        }
    }
}

/// Cursor over the encoded bytes; every read checks the length.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ReplayError> {
        if self.remaining() < n {
            return Err(ReplayError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ReplayError> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u16(&mut self) -> Result<u16, ReplayError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, ReplayError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, ReplayError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn flag(&mut self) -> Result<bool, ReplayError> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ReplayError::InvalidFlag),
        }
    }

    fn fix(&mut self) -> Result<Fix128, ReplayError> {
        let hi = i64::from_le_bytes(self.array()?);
        let lo = u64::from_le_bytes(self.array()?);
        Ok(Fix128::from_raw(hi, lo))
    }

    fn vec(&mut self) -> Result<Vec3Fix, ReplayError> {
        Ok(Vec3Fix::new(self.fix()?, self.fix()?, self.fix()?))
    }

    fn input(&mut self) -> Result<DriverInput, ReplayError> {
        Ok(DriverInput {
            throttle: self.fix()?,
            brake: self.fix()?,
            handbrake: self.fix()?,
            steering: self.fix()?,
        })
    }

    fn wheel(&mut self) -> Result<WheelDynamicsState, ReplayError> {
        let grounded = self.flag()?;
        let contact_point = self.vec()?;
        let contact_normal = self.vec()?;
        let compression = self.fix()?;
        let normal_load = self.fix()?;
        let steer_angle = self.fix()?;
        let omega = self.fix()?;
        let spin_angle = self.fix()?;
        let slip_ratio = self.fix()?;
        let slip_tan_alpha = self.fix()?;
        let longitudinal_force = self.fix()?;
        let lateral_force = self.fix()?;
        let brake_torque = self.fix()?;
        let abs_active = self.flag()?;
        let has_anchor = self.flag()?;
        let a = self.vec()?;
        Ok(WheelDynamicsState {
            grounded,
            contact_point,
            contact_normal,
            compression,
            normal_load,
            steer_angle,
            omega,
            spin_angle,
            slip_ratio,
            slip_tan_alpha,
            longitudinal_force,
            lateral_force,
            brake_torque,
            abs_active,
            anchor: has_anchor.then_some(a),
        })
    }
}
