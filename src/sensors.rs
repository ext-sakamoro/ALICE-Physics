//! Simulated sensors over a [`PhysicsWorld`]: a scanning range finder (lidar), a
//! contact sensor and an inertial measurement unit (IMU), each with optional
//! deterministic Gaussian noise.
//!
//! Every reading is a function of the world state, the sensor's settings and, for
//! noise, a seeded [`DeterministicRng`]: the same scene and seed give the same bits
//! on every platform.
//!
//! # Lidar
//!
//! [`Lidar`] casts one ray per point of an azimuth × elevation grid with
//! [`PhysicsWorld::ray_caster`] (the body BVH is built once per scan), against the
//! geometry bodies actually collide as (see [`crate::shape_raycast`]). In the
//! sensor's frame a ray of azimuth `a` and elevation `e` points along
//!
//! ```text
//! (cos e · cos a,  sin e,  cos e · sin a)
//! ```
//!
//! so azimuth turns from `+X` toward `+Z` and elevation lifts toward `+Y`. Angles
//! are spaced evenly from the minimum to the maximum inclusive
//! (`min + (max − min)·i / (n − 1)`; a count of 1 is the minimum alone). Rays are
//! cast and stored elevation-major: index `e_i · azimuth_count + a_i`.
//!
//! # Contact sensor
//!
//! [`ContactSensor`] reads the contact constraints of the last step that involve
//! its body (the solver keeps those of the last substep) and the normal force of
// LIMITATION(COV-SENSE-034): the contact sensor does not report contacts with static colliders (resolved by projection, no contact constraint)
//! each, as [`PhysicsWorld::contact_forces`] computes it. A contact with a static
//! collider ([`crate::static_collider`]) is resolved by projection, not by a
//! contact constraint, and a sensor body makes no constraints: neither is
// LIMITATION(COV-SENSE-035): a sleeping body reads no contact even while it rests on another body
//! reported. A body that has gone to sleep has no contacts to report.
//!
//! # IMU
//!
//! [`Imu`] differences the body's linear velocity between two samples, so it
//! reports the mean acceleration over the step. The accelerometer reading is the
//! **specific force**: the acceleration minus the world's gravity
//! (`config.gravity`), in the body's frame. A body at rest under gravity
//! `(0, −g, 0)` (a static body included) reads `(0, +g, 0)` turned into its frame;
//! a body in free fall reads zero. A body's `gravity_scale` is not gravity to the
//! sensor: the part of its weight beyond `1 ×` (or missing below it) acts like any
//! other applied force, so a body falling at `gravity_scale = 2` reads `(0, −g, 0)`.
//! The gyroscope reading is the body's angular velocity in its own frame, as the
//! solver keeps it (the XPBD backend recovers it from each substep's rotation, to
//! `O((ω·h)²)` relative).
//!
//! # Noise
//!
//! [`GaussianNoise`] adds `σ·z` with `z` from [`DeterministicRng::next_gaussian`]
//! (Box–Muller in `Fix128`; see that method for the precision and the 9.4 σ
//! range). A noisy sensor draws a fixed number of values per reading, so its
//! stream does not depend on what the reading was.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::rng::DeterministicRng;
use crate::shape_raycast::{RayFilter, WorldRayHit};
use crate::solver::PhysicsWorld;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Additive zero-mean Gaussian noise with its own seeded generator.
#[derive(Clone, Debug)]
pub struct GaussianNoise {
    /// Standard deviation of the added value.
    pub std_dev: Fix128,
    rng: DeterministicRng,
}

impl GaussianNoise {
    /// Noise of standard deviation `std_dev`, seeded with `seed`.
    #[must_use]
    pub fn new(std_dev: Fix128, seed: u64) -> Self {
        Self {
            std_dev,
            rng: DeterministicRng::new(seed),
        }
    }

    /// One draw `σ·z`.
    pub fn sample(&mut self) -> Fix128 {
        self.std_dev * self.rng.next_gaussian()
    }

    /// `value + σ·z`.
    pub fn apply(&mut self, value: Fix128) -> Fix128 {
        value + self.sample()
    }

    /// Each component plus its own draw (x, y, z in that order).
    pub fn apply_vec(&mut self, v: Vec3Fix) -> Vec3Fix {
        let x = self.apply(v.x);
        let y = self.apply(v.y);
        let z = self.apply(v.z);
        Vec3Fix::new(x, y, z)
    }
}

/// Evenly spaced angles from `min` to `max` inclusive.
fn angle_grid(min: Fix128, max: Fix128, count: usize) -> Vec<Fix128> {
    match count {
        0 => Vec::new(),
        1 => vec![min],
        n => {
            let span = max - min;
            let last = Fix128::from_int((n - 1) as i64);
            (0..n)
                .map(|i| min + span * Fix128::from_int(i as i64) / last)
                .collect()
        }
    }
}

/// A scanning range finder: one ray per azimuth × elevation grid point (see the
/// module doc for the frame and the order).
#[derive(Clone, Debug)]
pub struct Lidar {
    /// Where the sensor is: in the world for [`Self::scan`], in the body's frame
    /// (relative to its position) for [`Self::scan_from_body`].
    pub position: Vec3Fix,
    /// How the sensor is turned, in the same frame as `position`.
    pub rotation: QuatFix,
    /// Longest range measured; farther surfaces read as no return.
    pub max_range: Fix128,
    /// Which colliders the rays see.
    pub filter: RayFilter,
    azimuths: Vec<Fix128>,
    elevations: Vec<Fix128>,
    noise: Option<GaussianNoise>,
}

/// One lidar scan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LidarScan {
    /// Number of azimuth steps (the inner index).
    pub azimuth_count: usize,
    /// Number of elevation steps (the outer index).
    pub elevation_count: usize,
    /// Measured range of each ray (noise included), `None` for no return.
    pub ranges: Vec<Option<Fix128>>,
    /// The exact hit of each ray (no noise), `None` for no return.
    pub hits: Vec<Option<WorldRayHit>>,
}

impl LidarScan {
    /// The range at one grid point, `None` for no return or an index out of range.
    #[must_use]
    pub fn range(&self, azimuth_index: usize, elevation_index: usize) -> Option<Fix128> {
        if azimuth_index >= self.azimuth_count {
            return None;
        }
        self.ranges
            .get(elevation_index * self.azimuth_count + azimuth_index)
            .copied()
            .flatten()
    }
}

impl Lidar {
    /// A lidar at the origin, unturned, with `azimuth_count` azimuths over
    /// `[azimuth_min, azimuth_max]` and `elevation_count` elevations over
    /// `[elevation_min, elevation_max]` (radians), seeing up to `max_range` with the
    /// default [`RayFilter`] and no noise.
    #[must_use]
    pub fn new(
        (azimuth_min, azimuth_max, azimuth_count): (Fix128, Fix128, usize),
        (elevation_min, elevation_max, elevation_count): (Fix128, Fix128, usize),
        max_range: Fix128,
    ) -> Self {
        Self {
            position: Vec3Fix::ZERO,
            rotation: QuatFix::IDENTITY,
            max_range,
            filter: RayFilter::default(),
            azimuths: angle_grid(azimuth_min, azimuth_max, azimuth_count),
            elevations: angle_grid(elevation_min, elevation_max, elevation_count),
            noise: None,
        }
    }

    /// Place the sensor (see [`Self::position`] for the frame).
    #[must_use]
    pub fn with_pose(mut self, position: Vec3Fix, rotation: QuatFix) -> Self {
        self.position = position;
        self.rotation = rotation;
        self
    }

    /// Which colliders the rays see.
    #[must_use]
    pub fn with_filter(mut self, filter: RayFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Add range noise: one draw per ray, in scan order, misses included; a noisy
    /// range is clamped to `[0, max_range]`.
    #[must_use]
    pub fn with_noise(mut self, noise: GaussianNoise) -> Self {
        self.noise = Some(noise);
        self
    }

    /// The unit ray directions in the sensor's frame, in scan order.
    #[must_use]
    pub fn directions(&self) -> Vec<Vec3Fix> {
        let mut dirs = Vec::with_capacity(self.azimuths.len() * self.elevations.len());
        for &e in &self.elevations {
            let (sin_e, cos_e) = e.sin_cos();
            for &a in &self.azimuths {
                let (sin_a, cos_a) = a.sin_cos();
                dirs.push(Vec3Fix::new(cos_e * cos_a, sin_e, cos_e * sin_a));
            }
        }
        dirs
    }

    /// Scan from the sensor's pose in the world.
    pub fn scan(&mut self, world: &PhysicsWorld) -> LidarScan {
        self.scan_at(world, self.position, self.rotation, self.filter)
    }

    /// Scan from the sensor mounted on `body` (its pose relative to the body), not
    /// seeing that body. `None` for an index that is not a body.
    pub fn scan_from_body(&mut self, world: &PhysicsWorld, body: usize) -> Option<LidarScan> {
        let b = world.bodies.get(body)?;
        let position = b.position + b.rotation.rotate_vec(self.position);
        let rotation = b.rotation.mul(self.rotation);
        let filter = self.filter.excluding_body(body);
        Some(self.scan_at(world, position, rotation, filter))
    }

    fn scan_at(
        &mut self,
        world: &PhysicsWorld,
        position: Vec3Fix,
        rotation: QuatFix,
        filter: RayFilter,
    ) -> LidarScan {
        let caster = world.ray_caster(filter);
        let mut ranges = Vec::new();
        let mut hits = Vec::new();
        for dir in self.directions() {
            let hit = caster.closest(position, rotation.rotate_vec(dir), self.max_range);
            let mut range = hit.map(|h| h.t);
            if let Some(noise) = self.noise.as_mut() {
                let draw = noise.sample();
                range = range.map(|r| clamp(r + draw, Fix128::ZERO, self.max_range));
            }
            ranges.push(range);
            hits.push(hit);
        }
        LidarScan {
            azimuth_count: self.azimuths.len(),
            elevation_count: self.elevations.len(),
            ranges,
            hits,
        }
    }
}

fn clamp(x: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Reports whether a body is touching anything and how hard it is pressed.
#[derive(Clone, Debug)]
pub struct ContactSensor {
    /// The body sensed.
    pub body: usize,
    noise: Option<GaussianNoise>,
}

/// One contact-sensor reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContactReading {
    /// Whether the body has at least one contact.
    pub in_contact: bool,
    /// Number of contacts.
    pub contact_count: usize,
    /// Sum of the normal-force magnitudes on the body (noise included, clamped at
    /// zero).
    pub normal_force: Fix128,
    /// Vector sum of the normal forces on the body (no noise).
    pub net_force: Vec3Fix,
}

impl ContactSensor {
    /// A sensor on `body`, without noise.
    #[must_use]
    pub const fn new(body: usize) -> Self {
        Self { body, noise: None }
    }

    /// Add noise to the normal-force sum: one draw per reading.
    #[must_use]
    pub fn with_noise(mut self, noise: GaussianNoise) -> Self {
        self.noise = Some(noise);
        self
    }

    /// The reading after a step of `dt` (the `dt` passed to
    /// [`PhysicsWorld::step`]; the forces are computed from it).
    pub fn read(&mut self, world: &PhysicsWorld, dt: Fix128) -> ContactReading {
        let mut count = 0usize;
        let mut sum = Fix128::ZERO;
        let mut net = Vec3Fix::ZERO;
        for c in &world.contact_constraints {
            // The normal points from B to A and the force acts on A.
            let sign = if c.body_a == self.body {
                Fix128::ONE
            } else if c.body_b == self.body {
                Fix128::NEG_ONE
            } else {
                continue;
            };
            count += 1;
            if let Some(force) = world.contact_constraint_force(c, dt) {
                sum = sum + force;
                net = net + c.contact.normal * (force * sign);
            }
        }
        if let Some(noise) = self.noise.as_mut() {
            sum = clamp(
                noise.apply(sum),
                Fix128::ZERO,
                Fix128::from_raw(i64::MAX, u64::MAX),
            );
        }
        ContactReading {
            in_contact: count > 0,
            contact_count: count,
            normal_force: sum,
            net_force: net,
        }
    }
}

/// An inertial measurement unit on one body (see the module doc).
#[derive(Clone, Debug)]
pub struct Imu {
    /// The body carrying the unit.
    pub body: usize,
    prev_velocity: Vec3Fix,
    accel_noise: Option<GaussianNoise>,
    gyro_noise: Option<GaussianNoise>,
}

/// One IMU reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImuReading {
    /// Mean linear acceleration over the step, world frame (no noise).
    pub acceleration: Vec3Fix,
    /// Accelerometer: acceleration minus gravity, body frame (noise included).
    pub specific_force: Vec3Fix,
    /// Gyroscope: angular velocity, body frame (noise included).
    pub angular_velocity: Vec3Fix,
}

impl Imu {
    /// A unit on `body`, starting from its current velocity. `None` for an index
    /// that is not a body.
    #[must_use]
    pub fn new(world: &PhysicsWorld, body: usize) -> Option<Self> {
        let b = world.bodies.get(body)?;
        Some(Self {
            body,
            prev_velocity: b.velocity,
            accel_noise: None,
            gyro_noise: None,
        })
    }

    /// Add accelerometer noise: three draws per reading (x, y, z).
    #[must_use]
    pub fn with_accel_noise(mut self, noise: GaussianNoise) -> Self {
        self.accel_noise = Some(noise);
        self
    }

    /// Add gyroscope noise: three draws per reading (x, y, z).
    #[must_use]
    pub fn with_gyro_noise(mut self, noise: GaussianNoise) -> Self {
        self.gyro_noise = Some(noise);
        self
    }

    /// The reading after a step of `dt`; the current velocity becomes the reference
    /// for the next one. `None` (and nothing changes) when the body is gone or
    /// `dt ≤ 0`.
    pub fn sample(&mut self, world: &PhysicsWorld, dt: Fix128) -> Option<ImuReading> {
        if dt <= Fix128::ZERO {
            return None;
        }
        let b = world.bodies.get(self.body)?;
        let acceleration = (b.velocity - self.prev_velocity) / dt;
        self.prev_velocity = b.velocity;
        let to_body = b.rotation.conjugate();
        let mut specific_force = to_body.rotate_vec(acceleration - world.config.gravity);
        let mut angular_velocity = to_body.rotate_vec(b.angular_velocity);
        if let Some(noise) = self.accel_noise.as_mut() {
            specific_force = noise.apply_vec(specific_force);
        }
        if let Some(noise) = self.gyro_noise.as_mut() {
            angular_velocity = noise.apply_vec(angular_velocity);
        }
        Some(ImuReading {
            acceleration,
            specific_force,
            angular_velocity,
        })
    }
}
