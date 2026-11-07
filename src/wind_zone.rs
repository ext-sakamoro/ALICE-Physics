//! # MODEL
//!
//! - **Equations**: quadratic drag `F = ½ ρ C_d A |v_rel| v_rel` with
//!   `v_rel = u(t) − v_body`, in a prescribed wind field
//!   `u(t) = d · U + d · a · sin(2π f t)`, uniform inside the zone and zero
//!   outside. No flow equation is solved for the air.
//! - **Units**: SI. ρ kg/m³, A m², U and a m/s, f Hz, t s, F N; `C_d` is
//!   dimensionless.
//! - **Conservation**: the air is an external reservoir, so the bodies'
//!   momentum and energy are not conserved: drag adds momentum along the wind
//!   and dissipates relative kinetic energy. The air carries no state except the
//!   gust clock, which is in the world's state and digest.
//! - **Range**: the force is integrated explicitly once per substep: with
//!   `x = k |v_rel| inv_mass h` and `k = ½ ρ C_d A`, a substep multiplies `v_rel`
//!   by `1 − x`, so the relative speed is bounded while `x < 2` and grows past it
//!   (no clamp: the caller picks `h`). A force out of the
//!   fixed-point range is a participant fault that changes nothing. `C_d` and
//!   `A` are constants, so the model holds where the drag coefficient does not
//!   vary with the Reynolds number over the speeds used.
//! - **Approximations**: the wind passes through bodies and shapes (no shelter,
//!   no wake, no flow around an obstacle); no height profile (no boundary layer
//!   or log law near the ground); one sinusoidal gust along the mean direction
//!   (no turbulence spectrum such as von Kármán or Kaimal, no cross-wind); the
//!   force acts at the body's centre with a constant area, independent of the
//!   surface orientation.
//! - **Tiers**: L0 only. No lighter tier is built, so no tier has an error to
//!   measure against L0.
//! - **Validation** (`tests/analytic_wind_zone_participant.rs`):
//!   `release_into_a_steady_wind_follows_quadratic_drag` (closed form
//!   `v(t) = U − U/(1 + c U t)` and the first-order step error),
//!   `terminal_velocity_in_an_upward_wind` (`v = U − √(m g / k)`),
//!   `staged_force_matches_the_formula_at_the_clock`,
//!   `outside_the_zone_nothing_changes`,
//!   `the_explicit_drag_is_bounded_below_x_2_and_grows_above`,
//!   `rewind_and_fork_are_bit_identical`,
//!   `gust_scene_digest_is_pinned`. Not validated, because not modelled: the log
//!   law, the shelter and wake of an obstacle, and a gust spectrum.
//!
//! `WindZone` — bounded aerodynamic force field applied to rigid bodies
//! and cloth / hair particles.
//!
//! Companion of [`crate::buoyancy_zone`]: reuses the same
//! [`crate::buoyancy_zone::ZoneShape`] volume primitive but replaces
//! the buoyancy + drag model with a directional wind + turbulent gust
//! model tailored for cloth simulation, hair strand advection, and
//! kinematic character wind effects.
//!
//! Aerodynamic force acting on a particle / body inside the zone
//!
//! ```text
//! F = ½ · ρ_air · C_d · A · |v_rel| · v_rel
//! ```
//!
//! where `v_rel = wind_velocity(t) − body_velocity` and
//! `wind_velocity(t) = base_direction · base_speed
//!                     + turbulence · sin(2π · gust_frequency · t)`.
//!
// LIMITATION(COV-SOFT-072): The MVP evaluates the aerodynamic force in the body's centre and ignores surface-orientation dependence
// LIMITATION(COV-ENV-103): The MVP evaluates the aerodynamic force in the body's centre and ignores surface-orientation dependence
//! The MVP evaluates the aerodynamic force in the body's centre and
//! ignores surface-orientation dependence; full projected-area
//! integration and streamwise-turbulence spectral models are future
//! work.

use crate::buoyancy_zone::{hash_fix, ZoneShape};
use crate::det_hash::DetHasher;
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{BodyType, RigidBody};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use core::hash::Hasher;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Directional wind + turbulent gust volume.
#[derive(Debug, Clone, Copy)]
pub struct WindZone {
    /// Geometry of the wind pocket.
    pub shape: ZoneShape,
    /// Air density (kg/m³). Sea-level ISA = 1.225.
    pub air_density_kg_m3: Fix128,
    /// Nominal wind direction (unit vector). The builder normalises
    /// on construction; the field can be non-unit if the caller wants
    /// to bake wind magnitude in.
    pub direction: Vec3Fix,
    /// Base wind speed (m/s) applied along `direction`.
    pub base_speed_m_s: Fix128,
    /// Turbulence amplitude scaled by `sin(2π · gust_frequency · t)`
    /// and added to the base wind vector.
    pub turbulence_amplitude: Fix128,
    /// Gust temporal frequency in Hz.
    pub gust_frequency_hz: Fix128,
    /// Drag coefficient `C_d`. Cloth panels are often ~1.2; unit
    /// spheres are 0.47.
    pub drag_coefficient: Fix128,
    /// Reference cross-sectional area `A` (m²). Callers who want
    /// particle-sized cloth can pass the mesh triangle area; for
    /// bodies the projected area of the enclosing sphere is a
    /// reasonable proxy.
    pub reference_area_m2: Fix128,
}

impl WindZone {
    /// Preset for a light outdoor breeze (3 m/s base + 0.5 m/s gust,
    /// C_d = 1.2, A = 0.5 m²).
    #[must_use]
    pub fn light_breeze(shape: ZoneShape) -> Self {
        Self {
            shape,
            air_density_kg_m3: Fix128::from_ratio(1225, 1000),
            direction: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            base_speed_m_s: Fix128::from_int(3),
            turbulence_amplitude: Fix128::from_ratio(1, 2),
            gust_frequency_hz: Fix128::from_ratio(1, 2),
            drag_coefficient: Fix128::from_ratio(12, 10),
            reference_area_m2: Fix128::from_ratio(1, 2),
        }
    }

    /// Preset for a storm-force wind (20 m/s base + 5 m/s gust).
    #[must_use]
    pub fn storm(shape: ZoneShape) -> Self {
        Self {
            shape,
            air_density_kg_m3: Fix128::from_ratio(1225, 1000),
            direction: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            base_speed_m_s: Fix128::from_int(20),
            turbulence_amplitude: Fix128::from_int(5),
            gust_frequency_hz: Fix128::from_ratio(3, 2),
            drag_coefficient: Fix128::from_ratio(12, 10),
            reference_area_m2: Fix128::from_ratio(1, 2),
        }
    }

    /// Instantaneous wind velocity vector at simulation time `t_s`.
    ///
    /// The base direction is scaled by `base_speed_m_s`, then the
    // LIMITATION(COV-ENV-032): the turbulence gust `turbulence · sin(2π · f · t)` is added along the same direction
    /// turbulence gust `turbulence · sin(2π · f · t)` is added along
    /// the same direction. Callers who need cross-wind gusts can
    /// override this via [`WindZone::instantaneous_wind_vector`] after
    /// construction.
    #[must_use]
    pub fn instantaneous_wind_vector(&self, t_s: Fix128) -> Vec3Fix {
        let base = self.direction * self.base_speed_m_s;
        // sin(2π · f · t). We can call the crate's fixed-point sine
        // because it is CORDIC-based and cheap.
        let two_pi = Fix128::PI * Fix128::from_int(2);
        let phase = two_pi * self.gust_frequency_hz * t_s;
        let gust_scalar = self.turbulence_amplitude * phase.sin();
        base + self.direction * gust_scalar
    }

    /// Aerodynamic force on `body` at simulation time `t_s`. Returns
    /// zero if the body centre is outside the zone.
    #[must_use]
    pub fn force_on(&self, body: &RigidBody, t_s: Fix128) -> Vec3Fix {
        if !self.shape.contains(body.position) {
            return Vec3Fix::ZERO;
        }
        let wind = self.instantaneous_wind_vector(t_s);
        self.aerodynamic_force(wind, body.velocity)
    }

    /// Aerodynamic force on a cloth / hair particle at simulation time
    /// `t_s`. `position` is checked against the zone; `velocity` is
    /// the particle's current velocity.
    #[must_use]
    pub fn force_on_particle(&self, position: Vec3Fix, velocity: Vec3Fix, t_s: Fix128) -> Vec3Fix {
        if !self.shape.contains(position) {
            return Vec3Fix::ZERO;
        }
        let wind = self.instantaneous_wind_vector(t_s);
        self.aerodynamic_force(wind, velocity)
    }

    /// [`Self::force_on`] with every product and sum checked: `None` when a
    /// value leaves the range of [`Fix128`] (where `force_on` wraps), and the
    /// same bits as `force_on` otherwise.
    #[must_use]
    pub fn try_force_on(&self, body: &RigidBody, t_s: Fix128) -> Option<Vec3Fix> {
        if !self.shape.contains(body.position) {
            return Some(Vec3Fix::ZERO);
        }
        let wind = self.try_instantaneous_wind_vector(t_s)?;
        let relative = wind.checked_sub(body.velocity)?;
        let half = Fix128::from_ratio(1, 2);
        let coeff = half
            .checked_mul(self.air_density_kg_m3)?
            .checked_mul(self.drag_coefficient)?
            .checked_mul(self.reference_area_m2)?;
        if coeff.is_zero() {
            return Some(Vec3Fix::ZERO);
        }
        let k = coeff.sqrt();
        let scaled = relative.checked_scale(k)?;
        let scaled_speed_sq = scaled.checked_length_squared()?;
        if scaled_speed_sq.is_zero() {
            return Some(Vec3Fix::ZERO);
        }
        relative.checked_scale(k.checked_mul(scaled_speed_sq.sqrt())?)
    }

    /// [`Self::instantaneous_wind_vector`] with every product and sum checked.
    fn try_instantaneous_wind_vector(&self, t_s: Fix128) -> Option<Vec3Fix> {
        let base = self.direction.checked_scale(self.base_speed_m_s)?;
        let two_pi = Fix128::PI * Fix128::from_int(2);
        let phase = two_pi
            .checked_mul(self.gust_frequency_hz)?
            .checked_mul(t_s)?;
        let gust_scalar = self.turbulence_amplitude.checked_mul(phase.sin())?;
        base.checked_add(self.direction.checked_scale(gust_scalar)?)
    }

    fn aerodynamic_force(&self, wind_velocity: Vec3Fix, body_velocity: Vec3Fix) -> Vec3Fix {
        let relative = wind_velocity - body_velocity;
        let half = Fix128::from_ratio(1, 2);
        let coeff = half * self.air_density_kg_m3 * self.drag_coefficient * self.reference_area_m2;
        if coeff.is_zero() {
            return Vec3Fix::ZERO;
        }
        // `relative.length_squared()` (|v_rel|^2) overflows Fix128 once
        // |v_rel| exceeds ~3.04e9, well before the coefficient-weighted
        // magnitude — the quantity that actually needs to stay
        // representable — does (e.g. |v_rel| = 4e9 gives a finite,
        // representable 6.0e18 N once scaled by `coeff`, but the bare
        // |v_rel|^2 = 1.6e19 does not fit). Scale the velocity by
        // `sqrt(coeff)` *before* squaring instead of squaring first and
        // scaling down afterward: `(k * relative).length_squared() == coeff
        // * |v_rel|^2` exactly (k = sqrt(coeff)), without ever materializing
        // the bare |v_rel|^2.
        let k = coeff.sqrt();
        let scaled = relative * k;
        let scaled_speed_sq = scaled.length_squared();
        if scaled_speed_sq.is_zero() {
            return Vec3Fix::ZERO;
        }
        let scaled_speed = scaled_speed_sq.sqrt();
        // magnitude / speed == coeff * speed == k * scaled_speed (both
        // `k` and `scaled_speed` carry one factor of the coefficient /
        // velocity scaling respectively, so their product recovers
        // coeff * |v_rel| without ever computing |v_rel| or |v_rel|^2 bare).
        // Direction of the aerodynamic force is aligned with the relative
        // velocity.
        relative * (k * scaled_speed)
    }
}

// ---------------------------------------------------------------------------
// World participant
// ---------------------------------------------------------------------------

/// Snapshot tag of [`WindZoneParticipant`]: the ASCII code `WNDZ`, big endian.
pub const WIND_ZONE_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"WNDZ"));

/// Observation channel of [`WindZoneParticipant`]: the zone's clock, the time
/// it has been registered and stepped for (seconds).
pub const WIND_ZONE_OBS_TIME: u32 = 0;

/// Observation channel of [`WindZoneParticipant`]: how many bodies felt the
/// zone in the last substep it ran.
pub const WIND_ZONE_OBS_BODIES: u32 = 1;

/// Payload layout version written by [`WindZoneParticipant`].
const WIND_ZONE_STATE_VERSION: u32 = 1;

/// `version: u32`, `digest: u64`, the clock (`Fix128`), the body count (`u64`).
const WIND_ZONE_STATE_LEN: usize = 4 + 8 + 16 + 8;

/// [`WindZone`] as a participant of a [`crate::solver::PhysicsWorld`]: a
/// bounded wind that pushes the world's dynamic bodies inside every substep.
///
/// Register it with `PhysicsWorld::add_participant` and advance the world with
/// `try_step` (or `try_step_parallel`). In every substep of width `h` it reads
/// the bodies at the head of the substep and stages, for each dynamic body
/// with a finite mass whose centre is inside the zone, the force
/// [`WindZone::force_on`] at the zone's clock `t`, then advances `t` by `h`.
/// The world adds `F·inv_mass·h` to the velocity before it integrates the
/// position, so the drag is integrated explicitly. With `x = k·|v_rel|·inv_mass·h`
/// and `k = ½·ρ·C_d·A`, one substep multiplies `v_rel` by `1 − x`: the relative
/// speed falls without changing sign while `x < 1` and stays bounded while
/// `x < 2`; past 2 it grows step after step.
///
/// The clock starts at zero when the participant is built, is part of the
/// world's state and goes back with a restored snapshot, so a gust is at the
/// same phase in every replay. Rewind with `snapshotWorld` / `restore_world`:
/// `serialize_state` keeps the motion of the bodies only and leaves the clock
/// where it is.
///
/// Static and kinematic bodies are left out. A sleeping body is woken by the
/// world when the force changes its velocity. Every body takes the zone's one
/// reference area `A` whatever its shape or size.
///
/// # Range
///
/// A force or clock outside the range of [`Fix128`] makes the substep fail with
/// [`ParticipantFault::OutOfRange`]: nothing is staged, the participant keeps
/// its state and the world records the fault.
///
/// # Snapshot payload
///
/// Little endian, 36 bytes: `version: u32` (1), `digest: u64`, the clock as a
/// [`Fix128`] (`hi: i64`, `lo: u64`), the number of bodies that felt the zone
/// (`u64`). The digest is FNV-1a 64 over the zone's shape and values;
/// [`Participant::check_state`] refuses a payload of another length
/// ([`StateError::Length`]), version or zone ([`StateError::InvalidValue`]).
#[derive(Clone, Debug)]
pub struct WindZoneParticipant {
    zone: WindZone,
    digest: u64,
    time: Fix128,
    bodies: u64,
}

impl WindZoneParticipant {
    /// The participant for `zone`, with its clock at zero.
    #[must_use]
    pub fn new(zone: WindZone) -> Self {
        let mut h = DetHasher::default();
        zone.shape.hash_into(&mut h);
        for f in [
            zone.air_density_kg_m3,
            zone.direction.x,
            zone.direction.y,
            zone.direction.z,
            zone.base_speed_m_s,
            zone.turbulence_amplitude,
            zone.gust_frequency_hz,
            zone.drag_coefficient,
            zone.reference_area_m2,
        ] {
            hash_fix(&mut h, f);
        }
        Self {
            zone,
            digest: h.finish(),
            time: Fix128::ZERO,
            bodies: 0,
        }
    }

    /// The zone it applies.
    #[must_use]
    pub fn zone(&self) -> &WindZone {
        &self.zone
    }

    /// The zone's clock (seconds).
    #[must_use]
    pub fn time(&self) -> Fix128 {
        self.time
    }
}

impl Participant for WindZoneParticipant {
    fn kind(&self) -> ParticipantKind {
        WIND_ZONE_PARTICIPANT_KIND
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        // Work the whole substep out first: a value out of range leaves the
        // world's force buffer and this participant as they were.
        let next = self
            .time
            .checked_add(h)
            .ok_or(ParticipantFault::OutOfRange)?;
        let mut staged: Vec<(usize, Vec3Fix)> = Vec::new();
        let mut bodies = 0u64;
        for (i, body) in ctx.bodies().iter().enumerate() {
            if body.body_type != BodyType::Dynamic
                || body.inv_mass.is_zero()
                || !self.zone.shape.contains(body.position)
            {
                continue;
            }
            let force = self
                .zone
                .try_force_on(body, self.time)
                .ok_or(ParticipantFault::OutOfRange)?;
            staged.push((i, force));
            bodies += 1;
        }
        for (i, force) in staged {
            ctx.add_force(i, force)
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        self.time = next;
        self.bodies = bodies;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        out.push(WIND_ZONE_OBS_TIME, self.time);
        out.push(WIND_ZONE_OBS_BODIES, Fix128::from_int(self.bodies as i64));
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&WIND_ZONE_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&self.time.hi.to_le_bytes());
        out.extend_from_slice(&self.time.lo.to_le_bytes());
        out.extend_from_slice(&self.bodies.to_le_bytes());
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() != WIND_ZONE_STATE_LEN {
            return Err(StateError::Length {
                expected: WIND_ZONE_STATE_LEN,
                found: bytes.len(),
            });
        }
        if u32_at(bytes, 0) != WIND_ZONE_STATE_VERSION || u64_at(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.time = Fix128 {
            hi: u64_at(bytes, 12) as i64,
            lo: u64_at(bytes, 20),
        };
        self.bodies = u64_at(bytes, 28);
    }
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(b)
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buoyancy_zone::ZoneShape;
    use crate::math::QuatFix;
    use crate::solver::{BodyType, RigidBody};

    fn wide_aabb() -> ZoneShape {
        ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-10),
                Fix128::from_int(-10),
                Fix128::from_int(-10),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(10),
                Fix128::from_int(10),
                Fix128::from_int(10),
            ),
        }
    }

    fn resting_body_at(x: i64, y: i64, z: i64) -> RigidBody {
        RigidBody {
            position: Vec3Fix::new(
                Fix128::from_int(x),
                Fix128::from_int(y),
                Fix128::from_int(z),
            ),
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ONE,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: Vec3Fix::ZERO,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ONE,
            linear_damping_rate: Fix128::ZERO,
            angular_damping_rate: Fix128::ZERO,
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        }
    }

    #[test]
    fn light_breeze_preset_configured() {
        let zone = WindZone::light_breeze(wide_aabb());
        assert_eq!(zone.base_speed_m_s, Fix128::from_int(3));
        assert!(zone.direction.x > Fix128::ZERO);
    }

    #[test]
    fn storm_preset_stronger_than_breeze() {
        let breeze = WindZone::light_breeze(wide_aabb());
        let storm = WindZone::storm(wide_aabb());
        assert!(storm.base_speed_m_s > breeze.base_speed_m_s);
        assert!(storm.turbulence_amplitude > breeze.turbulence_amplitude);
    }

    #[test]
    fn wind_vector_at_zero_time_matches_base() {
        let zone = WindZone::light_breeze(wide_aabb());
        let v = zone.instantaneous_wind_vector(Fix128::ZERO);
        // sin(0) is close to 0 under CORDIC, so only base term remains
        // to within one ULP of the CORDIC accuracy.
        let expected = zone.direction * zone.base_speed_m_s;
        let dx = v.x - expected.x;
        let dy = v.y - expected.y;
        let dz = v.z - expected.z;
        let tolerance = Fix128::from_ratio(1, 100);
        assert!(dx.abs() < tolerance);
        assert!(dy.abs() < tolerance);
        assert!(dz.abs() < tolerance);
    }

    #[test]
    fn force_zero_outside_zone() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(100, 0, 0); // outside AABB.
        let f = zone.force_on(&body, Fix128::ZERO);
        assert_eq!(f, Vec3Fix::ZERO);
    }

    #[test]
    fn force_direction_aligned_with_wind() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(0, 0, 0);
        let f = zone.force_on(&body, Fix128::ZERO);
        // Wind is +x, body is resting, so force must be +x.
        assert!(f.x > Fix128::ZERO);
        assert_eq!(f.y, Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn force_reverses_when_body_moves_faster_than_wind() {
        let zone = WindZone::light_breeze(wide_aabb());
        let mut body = resting_body_at(0, 0, 0);
        body.velocity = Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        let f = zone.force_on(&body, Fix128::ZERO);
        // Body outruns the wind → drag pulls back (−x).
        assert!(f.x < Fix128::ZERO);
    }

    #[test]
    fn particle_force_matches_body_force_for_same_state() {
        let zone = WindZone::light_breeze(wide_aabb());
        let body = resting_body_at(0, 0, 0);
        let f_body = zone.force_on(&body, Fix128::ZERO);
        let f_particle = zone.force_on_particle(body.position, body.velocity, Fix128::ZERO);
        assert_eq!(f_body, f_particle);
    }

    #[test]
    fn zero_force_when_wind_matches_body_velocity() {
        let zone = WindZone::light_breeze(wide_aabb());
        let mut body = resting_body_at(0, 0, 0);
        body.velocity = zone.direction * zone.base_speed_m_s;
        let f = zone.force_on(&body, Fix128::ZERO);
        assert_eq!(f, Vec3Fix::ZERO);
    }

    fn resident() -> WindZoneParticipant {
        WindZoneParticipant::new(WindZone::light_breeze(wide_aabb()))
    }

    #[test]
    fn the_kind_is_the_ascii_code_of_wndz() {
        assert_eq!(resident().kind(), ParticipantKind::new(0x574e_445a));
    }

    #[test]
    fn the_state_round_trips_with_clock_and_count() {
        let mut a = resident();
        a.time = Fix128::from_ratio(-7, 3);
        a.bodies = 5;
        let mut bytes = Vec::new();
        a.write_state(&mut bytes);
        assert_eq!(bytes.len(), WIND_ZONE_STATE_LEN);
        let mut b = resident();
        assert_eq!(b.check_state(&bytes), Ok(()));
        b.read_state(&bytes);
        assert_eq!(b.time(), a.time());
        assert_eq!(b.bodies, 5);
    }

    #[test]
    fn check_state_refuses_length_version_and_zone() {
        let a = resident();
        let mut bytes = Vec::new();
        a.write_state(&mut bytes);
        assert_eq!(
            a.check_state(&bytes[..35]),
            Err(StateError::Length {
                expected: 36,
                found: 35
            })
        );
        let mut wrong_version = bytes.clone();
        wrong_version[0] ^= 1;
        assert_eq!(a.check_state(&wrong_version), Err(StateError::InvalidValue));
        let other = WindZoneParticipant::new(WindZone::storm(wide_aabb()));
        assert_eq!(other.check_state(&bytes), Err(StateError::InvalidValue));
    }

    /// Every value of the zone changes the digest a snapshot is checked with.
    #[test]
    fn every_value_of_the_zone_is_in_the_digest() {
        let base = WindZone::light_breeze(wide_aabb());
        let d = WindZoneParticipant::new(base).digest;
        let bump = Fix128::from_ratio(1, 1024);
        let mut changed: Vec<WindZone> = Vec::new();
        for edit in [
            |z: &mut WindZone, b: Fix128| z.air_density_kg_m3 = z.air_density_kg_m3 + b,
            |z: &mut WindZone, b: Fix128| z.direction.x = z.direction.x + b,
            |z: &mut WindZone, b: Fix128| z.direction.y = z.direction.y + b,
            |z: &mut WindZone, b: Fix128| z.direction.z = z.direction.z + b,
            |z: &mut WindZone, b: Fix128| z.base_speed_m_s = z.base_speed_m_s + b,
            |z: &mut WindZone, b: Fix128| z.turbulence_amplitude = z.turbulence_amplitude + b,
            |z: &mut WindZone, b: Fix128| z.gust_frequency_hz = z.gust_frequency_hz + b,
            |z: &mut WindZone, b: Fix128| z.drag_coefficient = z.drag_coefficient + b,
            |z: &mut WindZone, b: Fix128| z.reference_area_m2 = z.reference_area_m2 + b,
        ] {
            let mut z = base;
            edit(&mut z, bump);
            changed.push(z);
        }
        let mut moved = base;
        moved.shape = ZoneShape::Sphere {
            centre: Vec3Fix::ZERO,
            radius: Fix128::ONE,
        };
        changed.push(moved);
        for (i, z) in changed.into_iter().enumerate() {
            assert_ne!(WindZoneParticipant::new(z).digest, d, "value {i}");
        }
    }

    /// The checked force gives up where the unchecked one would wrap: a wind
    /// of 2⁶² m/s has no representable relative velocity against a body moving
    /// the other way at the same speed.
    #[test]
    fn the_checked_force_is_none_out_of_range() {
        let mut zone = WindZone::light_breeze(wide_aabb());
        zone.turbulence_amplitude = Fix128::ZERO;
        let huge = Fix128::from_int(1_i64 << 62);
        zone.base_speed_m_s = huge;
        let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        body.velocity = Vec3Fix::new(-huge, Fix128::ZERO, Fix128::ZERO);
        assert_eq!(zone.try_force_on(&body, Fix128::ZERO), None);
        body.velocity = Vec3Fix::ZERO;
        assert_eq!(zone.try_force_on(&body, Fix128::ZERO), None);
    }

    #[test]
    fn the_checked_force_is_zero_without_coefficient_or_outside() {
        let mut zone = WindZone::light_breeze(wide_aabb());
        zone.reference_area_m2 = Fix128::ZERO;
        let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        assert_eq!(zone.try_force_on(&body, Fix128::ZERO), Some(Vec3Fix::ZERO));
        let zone = WindZone::light_breeze(wide_aabb());
        let far = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(100), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        assert_eq!(zone.try_force_on(&far, Fix128::ZERO), Some(Vec3Fix::ZERO));
        let mut still = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        let mut calm = WindZone::light_breeze(wide_aabb());
        calm.base_speed_m_s = Fix128::ZERO;
        calm.turbulence_amplitude = Fix128::ZERO;
        still.velocity = Vec3Fix::ZERO;
        assert_eq!(calm.try_force_on(&still, Fix128::ZERO), Some(Vec3Fix::ZERO));
    }

    /// The substep stages `force_on` at the clock for each dynamic body inside the zone,
    /// leaves static and outside bodies alone, advances the clock by `h` and reports the
    /// count; a force out of range is a fault that stages nothing and keeps the clock.
    #[test]
    fn the_substep_stages_the_force_and_advances_the_clock() {
        use crate::world_participant::{ForceAccumulator, ObservationSink, SubstepCtx};
        let zone = WindZone::light_breeze(wide_aabb());
        let mut inside = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        inside.velocity = Vec3Fix::from_int(1, 0, 0);
        let outside = RigidBody::new_dynamic(Vec3Fix::from_int(100, 0, 0), Fix128::ONE);
        let fixed = RigidBody::new_static(Vec3Fix::ZERO);
        let bodies = [inside, outside, fixed];
        let mut p = WindZoneParticipant::new(zone);
        let h = Fix128::from_ratio(1, 64);
        let mut forces = ForceAccumulator::new(bodies.len());
        {
            let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h).expect("ctx");
            p.substep(&mut ctx, h).expect("substep");
        }
        assert_eq!(
            forces.force(0),
            Some(zone.force_on(&bodies[0], Fix128::ZERO))
        );
        assert_eq!(forces.force(1), Some(Vec3Fix::ZERO));
        assert_eq!(forces.force(2), Some(Vec3Fix::ZERO));
        assert_eq!(p.time(), h);
        let mut sink = ObservationSink::new();
        p.observe(&mut sink);
        assert_eq!(
            sink.values(),
            &[(WIND_ZONE_OBS_TIME, h), (WIND_ZONE_OBS_BODIES, Fix128::ONE)]
        );

        let mut huge = zone;
        huge.base_speed_m_s = Fix128::from_int(1 << 40);
        let mut q = WindZoneParticipant::new(huge);
        let mut forces = ForceAccumulator::new(bodies.len());
        {
            let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h).expect("ctx");
            assert_eq!(q.substep(&mut ctx, h), Err(ParticipantFault::OutOfRange));
        }
        assert_eq!(forces.force(0), Some(Vec3Fix::ZERO));
        assert_eq!(q.time(), Fix128::ZERO);
    }
}
