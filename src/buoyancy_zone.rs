//! # MODEL
//!
//! - **Equations**: Archimedes buoyancy plus linear and quadratic drag on each dynamic
//!   body inside the zone, `F = ρ V_sub |g| ŷ − c_lin v − c_quad |v| v`, with
//!   `V_sub = f · (4/3) π r³` and `f` the submerged fraction of a sphere of the world's
//!   radius `r` of that body, `f = h² (3r − h) / (4 r³)`, `h = clamp(d + r, 0, 2r)`, `d`
//!   the signed depth of the centre below the surface.
//! - **Units**: SI. ρ kg/m³, r m, V m³, g m/s², v m/s, `c_lin` kg/s, `c_quad` kg/m, F N.
//! - **Conservation**: the fluid is an external reservoir at rest, so the bodies'
//!   momentum is not conserved; buoyancy is conservative (a potential in depth) and
//!   the drag dissipates kinetic energy. The fluid carries no state of its own except
//!   the count of bodies inside, which is in the world's state and digest.
//! - **Range**: the force is integrated explicitly once per substep of width `h`; the
//!   drag is stable while `c_lin h / m < 2` and `c_quad |v| h / m < 2` (a substep
//!   multiplies the velocity by `1 − x`). A force out of the fixed-point range is a
//!   participant fault that changes nothing; the submerged fraction is clamped to
//!   `[0, 1]` by construction. The density and coefficients are constants.
//! - **Approximations**: a static fluid (no flow, no waves, no free surface motion),
//!   a sphere of the body's radius stands for the body's shape, the force acts at the
//!   body's centre (no torque from an offset centre of buoyancy), no added mass, no
//!   surface tension, isotropic drag.
//! - **Tiers**: prescribed static fluid volume (no flow): a tier below the fluid L0.
//!   This is not the fluid model: the L0, a fluid solved with a free surface around
//!   SDF solids, is not built, so this tier's error against it is not measured.
//! - **Validation**: `tests/analytic_buoyancy_zone_participant.rs::a_half_density_sphere_floats_on_its_centre::a_quarter_density_sphere_floats_at_the_cap_root::terminal_speed_under_linear_drag::terminal_speed_under_quadratic_drag::the_displaced_volume_is_the_worlds_radius_of_each_body::staged_force_matches_the_formula::outside_the_zone_nothing_changes::the_box_bounds_the_fluid_on_every_side::the_sphere_bounds_the_fluid_and_far_bodies_stay_in_range::rewind_and_fork_are_bit_identical::splash_scene_digest_is_pinned`
//!   (Archimedes at ½ and ¼ density against the cap root, terminal speeds against the
//!   closed forms, the formula at one substep, the bounds of the volume, replay bits).
//! - validation missing: the hydrostatic pressure field `p = ρ g h` inside the fluid
//!   (the fluid L0; this tier has no pressure field)
//! - validation missing: added mass and the torque of an offset centre of buoyancy
//!   (the fluid L0)
//!
//! `BuoyancyZone` — 3-D fluid volume that applies buoyancy + drag to
//! immersed rigid bodies.
//!
//! [`BuoyancyZoneParticipant`] is the same law as a resident of the world: it
//! works inside the substeps, with the world's own radius of each body, and is
//! in the digest and the snapshot.
//!
//! Complements the existing [`crate::force::ForceField::Buoyancy`]
//! variant (infinite half-space with a horizontal water surface) by
//! modelling a **bounded** fluid volume with an explicit shape. Two
//! shapes are supported today:
//!
//! - [`ZoneShape::Aabb`] — axis-aligned box, e.g. a pool or tank.
//! - [`ZoneShape::Sphere`] — spherical pocket, e.g. a droplet or
//!   underwater cave.
//!
//! For each dynamic body inside the zone the resulting force is
//!
//! ```text
//! F = ρ_water · V_displaced · |g|  · ŷ_up
//!   − c_lin  · v
//!   − c_quad · v · |v|
//! ```
//!
// LIMITATION(COV-COUPLE-066): where `V_displaced` is estimated from a spherical body proxy of radius `body_radius`; the exact submerged volume of an arbitrary mesh is out of scope for this module and remains future work.
//! where `V_displaced` is estimated from a spherical body proxy of
//! radius `body_radius`; the exact submerged volume of an arbitrary
//! mesh is out of scope for this module and remains future work.

use crate::det_hash::DetHasher;
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{BodyType, RigidBody};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use core::hash::Hasher;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Feed a value's bits to `h`.
pub(crate) fn hash_fix(h: &mut crate::det_hash::DetHasher, f: Fix128) {
    core::hash::Hasher::write(h, &f.hi.to_le_bytes());
    core::hash::Hasher::write(h, &f.lo.to_le_bytes());
}

/// Shape of the buoyancy zone volume.
#[derive(Debug, Clone, Copy)]
pub enum ZoneShape {
    /// Axis-aligned box defined by `min` and `max` corners.
    Aabb {
        /// Lower corner (inclusive).
        min: Vec3Fix,
        /// Upper corner (inclusive).
        max: Vec3Fix,
    },
    /// Sphere with a centre and radius.
    Sphere {
        /// Centre position.
        centre: Vec3Fix,
        /// Radius.
        radius: Fix128,
    },
}

impl ZoneShape {
    /// Feed the shape's variant and values to `h`: the part of a resident's
    /// law digest that says where the zone is.
    pub(crate) fn hash_into(&self, h: &mut crate::det_hash::DetHasher) {
        match self {
            Self::Aabb { min, max } => {
                core::hash::Hasher::write(h, &[0]);
                for v in [*min, *max] {
                    hash_fix(h, v.x);
                    hash_fix(h, v.y);
                    hash_fix(h, v.z);
                }
            }
            Self::Sphere { centre, radius } => {
                core::hash::Hasher::write(h, &[1]);
                hash_fix(h, centre.x);
                hash_fix(h, centre.y);
                hash_fix(h, centre.z);
                hash_fix(h, *radius);
            }
        }
    }

    /// True when `point` lies strictly inside the shape.
    #[must_use]
    pub fn contains(&self, point: Vec3Fix) -> bool {
        match self {
            Self::Aabb { min, max } => {
                point.x >= min.x
                    && point.x <= max.x
                    && point.y >= min.y
                    && point.y <= max.y
                    && point.z >= min.z
                    && point.z <= max.z
            }
            Self::Sphere { centre, radius } => {
                let delta = point - *centre;
                delta.length_squared() <= (*radius) * (*radius)
            }
        }
    }

    /// Signed depth of `point` below the zone's upper surface (negative when
    /// the point is above it), or `None` when the point is outside the zone's
    /// horizontal footprint / below its floor — i.e. cannot be touching the
    /// fluid at all. Used by [`BuoyancyZone::submerged_fraction`].
    #[must_use]
    pub fn signed_depth_below_surface(&self, point: Vec3Fix) -> Option<Fix128> {
        match self {
            Self::Aabb { min, max } => {
                if point.x < min.x || point.x > max.x || point.z < min.z || point.z > max.z {
                    return None;
                }
                if point.y < min.y {
                    return None;
                }
                Some(max.y - point.y)
            }
            Self::Sphere { centre, radius } => {
                let dx = point.x - centre.x;
                let dz = point.z - centre.z;
                if dx * dx + dz * dz >= *radius * *radius {
                    return None;
                }
                if point.y < centre.y - *radius {
                    return None;
                }
                Some(centre.y + *radius - point.y)
            }
        }
    }

    /// Unsigned depth below the upper surface, `ZERO` when the point is not
    /// inside the fluid (outside the footprint, above the surface or below
    /// the floor).
    #[must_use]
    pub fn depth_below_surface(&self, point: Vec3Fix) -> Fix128 {
        match self {
            Self::Aabb { min, max } => {
                if point.x < min.x
                    || point.x > max.x
                    || point.z < min.z
                    || point.z > max.z
                    || point.y > max.y
                {
                    return Fix128::ZERO;
                }
                if point.y < min.y {
                    // below the pool floor: not in the water (pre-1.2.0 clamped
                    // to the floor and reported full depth)
                    return Fix128::ZERO;
                }
                max.y - point.y
            }
            Self::Sphere { centre, radius } => {
                let delta = point - *centre;
                let dist_squared = delta.length_squared();
                let r_squared = (*radius) * (*radius);
                if dist_squared >= r_squared {
                    return Fix128::ZERO;
                }
                let surface_y = centre.y + *radius;
                if point.y > surface_y {
                    Fix128::ZERO
                } else {
                    surface_y - point.y
                }
            }
        }
    }
}

/// Bounded fluid volume applying buoyancy + drag.
#[derive(Debug, Clone, Copy)]
pub struct BuoyancyZone {
    /// Geometry of the fluid pocket.
    pub shape: ZoneShape,
    /// Fluid density (kg/m³). Water at 20 °C is ≈ 998.
    pub density_kg_m3: Fix128,
    /// Gravity magnitude (m/s²). Positive; direction is fixed to
    /// `+y` for buoyancy (world coordinates assume `+y = up`).
    pub gravity: Fix128,
    /// Linear drag coefficient `c_lin`. Applied as `-c_lin · v`.
    pub drag_linear: Fix128,
    /// Quadratic drag coefficient `c_quad`. Applied as `-c_quad · v · |v|`.
    pub drag_quadratic: Fix128,
}

impl BuoyancyZone {
    /// Construct a zone with reasonable defaults for a water pool
    /// (ρ = 1000 kg/m³, |g| = 9.81 m/s², linear drag 0.5, quadratic drag 1.0).
    #[must_use]
    pub fn water_pool(shape: ZoneShape) -> Self {
        Self {
            shape,
            density_kg_m3: Fix128::from_int(1000),
            gravity: Fix128::from_ratio(981, 100),
            drag_linear: Fix128::from_ratio(1, 2),
            drag_quadratic: Fix128::ONE,
        }
    }

    /// Compute the buoyancy + drag force acting on `body`. `body_radius`
    /// approximates the immersed volume as a sphere; callers can pass
    /// the enclosing-sphere radius of a compound collider.
    #[must_use]
    pub fn force_on(&self, body: &RigidBody, body_radius: Fix128) -> Vec3Fix {
        let submerged_fraction = self.submerged_fraction(body.position, body_radius);
        if submerged_fraction.is_zero() {
            return Vec3Fix::ZERO;
        }

        // V_displaced = fraction · (4/3) π r³. Before this fix the constant was
        // `from_ratio(41888, 10000)` = 4.1888, 2.3e-6 above the true 4.18879020…
        // (the comment even quoted a different value, 4.18879); it now uses the
        // full-precision `Fix128::PI`.
        let four_pi_third = Fix128::PI * Fix128::from_int(4) / Fix128::from_int(3);
        let radius_cubed = body_radius * body_radius * body_radius;
        let full_volume = four_pi_third * radius_cubed;
        let displaced = full_volume * submerged_fraction;
        let buoy_magnitude = self.density_kg_m3 * displaced * self.gravity;
        let buoyancy = Vec3Fix::new(Fix128::ZERO, buoy_magnitude, Fix128::ZERO);

        // Linear drag: -c_lin · v.
        let linear_drag = if body.velocity.length_squared().is_zero() {
            Vec3Fix::ZERO
        } else {
            body.velocity * (-self.drag_linear)
        };

        // Quadratic drag: -c_quad · v · |v|.
        let speed_sq = body.velocity.length_squared();
        let quadratic_drag = if speed_sq.is_zero() {
            Vec3Fix::ZERO
        } else {
            let speed = speed_sq.sqrt();
            body.velocity * (-self.drag_quadratic * speed)
        };

        buoyancy + linear_drag + quadratic_drag
    }

    /// Submerged volume fraction (`0..=1`) of a sphere of `radius` centred
    /// at `position`: the exact spherical-cap ratio `h²(3r − h) / (4r³)` with
    /// `h = clamp(d + r, 0, 2r)`, where `d` is the signed depth of the centre
    /// below the zone's upper surface (negative above it).
    ///
    /// Continuous through the surface: a centre `r` above the surface gives
    /// 0, at the surface ½, `r` below 1. Before 1.2.0 the fraction was the
    /// linear estimate `(d + r) / 2r` (up to 9.4 % off the cap volume) and
    /// jumped from 0 to ½ as the centre crossed the surface, so a body lighter
    /// than half the fluid density could never settle
    /// (`tests/engineering_oracles_fluid.rs`). For AABB zones the centre must
    /// lie inside the horizontal footprint and above the floor; for sphere
    /// zones inside the ball's horizontal extent.
    #[must_use]
    pub fn submerged_fraction(&self, position: Vec3Fix, radius: Fix128) -> Fix128 {
        if radius <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        let Some(depth) = self.shape.signed_depth_below_surface(position) else {
            return Fix128::ZERO;
        };
        let two_r = radius + radius;
        let mut h = depth + radius;
        if h <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        if h >= two_r {
            return Fix128::ONE;
        }
        // cap volume / sphere volume = h²(3r − h) / (4 r³)
        h = h.min(two_r);
        let three_r_minus_h = Fix128::from_int(3) * radius - h;
        (h * h * three_r_minus_h) / (Fix128::from_int(4) * radius * radius * radius)
    }

    /// [`Self::force_on`] with every product and sum checked: `None` when a
    /// value leaves the range of [`Fix128`]. In range it is `force_on`, bit for
    /// bit.
    #[must_use]
    pub fn try_force_on(&self, body: &RigidBody, body_radius: Fix128) -> Option<Vec3Fix> {
        let fraction = self.try_submerged_fraction(body.position, body_radius)?;
        self.try_force_at(body, body_radius, fraction)
    }

    /// The force of [`Self::try_force_on`] for an already known fraction.
    fn try_force_at(
        &self,
        body: &RigidBody,
        body_radius: Fix128,
        fraction: Fix128,
    ) -> Option<Vec3Fix> {
        if fraction.is_zero() {
            return Some(Vec3Fix::ZERO);
        }
        let four_pi_third = Fix128::PI * Fix128::from_int(4) / Fix128::from_int(3);
        let radius_cubed = body_radius
            .checked_mul(body_radius)?
            .checked_mul(body_radius)?;
        let displaced = four_pi_third
            .checked_mul(radius_cubed)?
            .checked_mul(fraction)?;
        let lift = self
            .density_kg_m3
            .checked_mul(displaced)?
            .checked_mul(self.gravity)?;
        let buoyancy = Vec3Fix::new(Fix128::ZERO, lift, Fix128::ZERO);
        let speed_sq = body.velocity.checked_length_squared()?;
        if speed_sq.is_zero() {
            return Some(buoyancy);
        }
        let linear = body.velocity.checked_scale(-self.drag_linear)?;
        let quadratic = body
            .velocity
            .checked_scale((-self.drag_quadratic).checked_mul(speed_sq.sqrt())?)?;
        buoyancy.checked_add(linear)?.checked_add(quadratic)
    }

    /// [`Self::submerged_fraction`] with every product and sum checked: `None`
    /// when a value leaves the range of [`Fix128`], zero when the sphere does
    /// not touch the fluid.
    fn try_submerged_fraction(&self, position: Vec3Fix, radius: Fix128) -> Option<Fix128> {
        if radius <= Fix128::ZERO {
            return Some(Fix128::ZERO);
        }
        let depth = match self.shape {
            ZoneShape::Aabb { min, max } => {
                if position.x < min.x
                    || position.x > max.x
                    || position.z < min.z
                    || position.z > max.z
                    || position.y < min.y
                {
                    return Some(Fix128::ZERO);
                }
                max.y.checked_sub(position.y)?
            }
            ZoneShape::Sphere {
                centre,
                radius: zone_radius,
            } => {
                let dx = position.x.checked_sub(centre.x)?;
                let dz = position.z.checked_sub(centre.z)?;
                // Further than the radius along one axis is outside, however
                // far: the squares below need not be representable for that.
                let reach = zone_radius.abs();
                if dx.abs() >= reach || dz.abs() >= reach {
                    return Some(Fix128::ZERO);
                }
                let across = dx.checked_mul(dx)?.checked_add(dz.checked_mul(dz)?)?;
                if across >= zone_radius.checked_mul(zone_radius)?
                    || position.y < centre.y.checked_sub(zone_radius)?
                {
                    return Some(Fix128::ZERO);
                }
                centre.y.checked_add(zone_radius)?.checked_sub(position.y)?
            }
        };
        let two_r = radius.checked_add(radius)?;
        let h = depth.checked_add(radius)?;
        if h <= Fix128::ZERO {
            return Some(Fix128::ZERO);
        }
        if h >= two_r {
            return Some(Fix128::ONE);
        }
        let three_r_minus_h = Fix128::from_int(3).checked_mul(radius)?.checked_sub(h)?;
        let four_r_cubed = Fix128::from_int(4)
            .checked_mul(radius)?
            .checked_mul(radius)?
            .checked_mul(radius)?;
        h.checked_mul(h)?
            .checked_mul(three_r_minus_h)?
            .checked_div(four_r_cubed)
    }
}

// ---------------------------------------------------------------------------
// World participant
// ---------------------------------------------------------------------------

/// Snapshot tag of [`BuoyancyZoneParticipant`]: the ASCII code `BUOY`, big
/// endian.
pub const BUOYANCY_ZONE_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"BUOY"));

/// Observation channel of [`BuoyancyZoneParticipant`]: how many bodies touched
/// the fluid in the last substep it ran.
pub const BUOYANCY_ZONE_OBS_BODIES: u32 = 0;

/// Payload layout version written by [`BuoyancyZoneParticipant`].
const BUOYANCY_ZONE_STATE_VERSION: u32 = 1;

/// `version: u32`, `digest: u64`, the body count (`u64`).
const BUOYANCY_ZONE_STATE_LEN: usize = 4 + 8 + 8;

/// [`BuoyancyZone`] as a participant of a [`crate::solver::PhysicsWorld`]: a
/// bounded fluid that lifts and drags the world's dynamic bodies inside every
/// substep.
///
/// Register it with `PhysicsWorld::add_participant` and advance the world with
/// `try_step` (or `try_step_parallel`). In every substep of width `h` it reads
/// the bodies at the head of the substep and stages, for each dynamic body with
/// a finite mass that touches the fluid, the force [`BuoyancyZone::force_on`]
/// with the world's collision radius of that body (the radius given with
/// `add_body_with_radius` or `set_collision_radius`, else the world's
/// `sdf_collision_radius`). The drag is integrated explicitly: it stays stable
/// while `(c_lin + 2·c_quad·|v|)·inv_mass·h < 2`.
///
/// The zone's `gravity` is its own value: a fluid that should balance the
/// world's weight is given the magnitude of the world's gravity.
///
/// Static and kinematic bodies are left out. A sleeping body is woken by the
/// world when the force changes its velocity.
///
/// # Range
///
/// A force outside the range of [`Fix128`] makes the substep fail with
/// [`ParticipantFault::OutOfRange`]: nothing is staged, the participant keeps
/// its state and the world records the fault. A substep context built without
/// the world's radii fails with [`ParticipantFault::InvalidState`] when a body
/// is a candidate, since the displaced volume is unknown.
///
/// # Snapshot payload
///
/// Little endian, 20 bytes: `version: u32` (1), `digest: u64`, the number of
/// bodies that touched the fluid (`u64`). The digest is FNV-1a 64 over the
/// zone's shape and values; [`Participant::check_state`] refuses a payload of
/// another length ([`StateError::Length`]), version or zone
/// ([`StateError::InvalidValue`]).
#[derive(Clone, Debug)]
pub struct BuoyancyZoneParticipant {
    zone: BuoyancyZone,
    digest: u64,
    bodies: u64,
}

impl BuoyancyZoneParticipant {
    /// The participant for `zone`.
    #[must_use]
    pub fn new(zone: BuoyancyZone) -> Self {
        let mut h = DetHasher::default();
        zone.shape.hash_into(&mut h);
        for f in [
            zone.density_kg_m3,
            zone.gravity,
            zone.drag_linear,
            zone.drag_quadratic,
        ] {
            hash_fix(&mut h, f);
        }
        Self {
            zone,
            digest: h.finish(),
            bodies: 0,
        }
    }

    /// The zone it applies.
    #[must_use]
    pub fn zone(&self) -> &BuoyancyZone {
        &self.zone
    }

    /// How many bodies touched the fluid in the last substep it ran.
    #[must_use]
    pub fn bodies_in_fluid(&self) -> u64 {
        self.bodies
    }
}

impl Participant for BuoyancyZoneParticipant {
    fn kind(&self) -> ParticipantKind {
        BUOYANCY_ZONE_PARTICIPANT_KIND
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _h: Fix128) -> Result<(), ParticipantFault> {
        // Work the whole substep out first: a value out of range leaves the
        // world's force buffer and this participant as they were.
        let mut staged: Vec<(usize, Vec3Fix)> = Vec::new();
        for (i, body) in ctx.bodies().iter().enumerate() {
            if body.body_type != BodyType::Dynamic || body.inv_mass.is_zero() {
                continue;
            }
            let radius = ctx.body_radius(i).ok_or(ParticipantFault::InvalidState)?;
            let fraction = self
                .zone
                .try_submerged_fraction(body.position, radius)
                .ok_or(ParticipantFault::OutOfRange)?;
            if fraction.is_zero() {
                continue;
            }
            let force = self
                .zone
                .try_force_at(body, radius, fraction)
                .ok_or(ParticipantFault::OutOfRange)?;
            staged.push((i, force));
        }
        let bodies = staged.len() as u64;
        for (i, force) in staged {
            ctx.add_force(i, force)
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        self.bodies = bodies;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        out.push(
            BUOYANCY_ZONE_OBS_BODIES,
            Fix128::from_int(self.bodies as i64),
        );
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&BUOYANCY_ZONE_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&self.bodies.to_le_bytes());
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() != BUOYANCY_ZONE_STATE_LEN {
            return Err(StateError::Length {
                expected: BUOYANCY_ZONE_STATE_LEN,
                found: bytes.len(),
            });
        }
        if u32_at(bytes, 0) != BUOYANCY_ZONE_STATE_VERSION || u64_at(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.bodies = u64_at(bytes, 12);
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
    use crate::solver::{BodyType, RigidBody};

    fn floating_body(y: i64) -> RigidBody {
        let mut body = RigidBody {
            position: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(y), Fix128::ZERO),
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ONE,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: Vec3Fix::ZERO,
            rotation: crate::math::QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: crate::math::QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ONE,
            linear_damping_rate: Fix128::ZERO,
            angular_damping_rate: Fix128::ZERO,
            is_sensor: false,
            body_type: BodyType::Dynamic,
            kinematic_target: None,
        };
        body.prev_position = body.position;
        body
    }

    #[test]
    fn aabb_contains_interior_point() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        assert!(shape.contains(Vec3Fix::ZERO));
        assert!(!shape.contains(Vec3Fix::new(
            Fix128::from_int(2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
    }

    #[test]
    fn sphere_contains_interior_point() {
        let shape = ZoneShape::Sphere {
            centre: Vec3Fix::ZERO,
            radius: Fix128::from_int(1),
        };
        assert!(shape.contains(Vec3Fix::new(
            Fix128::from_ratio(1, 2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
        assert!(!shape.contains(Vec3Fix::new(
            Fix128::from_int(2),
            Fix128::ZERO,
            Fix128::ZERO,
        )));
    }

    #[test]
    fn aabb_depth_zero_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        let above = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO);
        assert_eq!(shape.depth_below_surface(above), Fix128::ZERO);
    }

    #[test]
    fn aabb_depth_matches_expected_at_centre() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-2),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(2),
                Fix128::from_int(1),
            ),
        };
        // Point at (0, 0, 0). Zone top at y = 2. Depth should be 2.
        let depth = shape.depth_below_surface(Vec3Fix::ZERO);
        assert_eq!(depth, Fix128::from_int(2));
    }

    #[test]
    fn buoyancy_produces_upward_force_when_submerged() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let body = floating_body(-1);
        let force = zone.force_on(&body, Fix128::ONE);
        // Upward buoyancy => positive y component.
        assert!(force.y > Fix128::ZERO);
        // No horizontal force in resting state.
        assert_eq!(force.x, Fix128::ZERO);
        assert_eq!(force.z, Fix128::ZERO);
    }

    #[test]
    fn buoyancy_zero_when_body_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        // Body at y = 10, zone max y = 5 → outside.
        let body = floating_body(10);
        let force = zone.force_on(&body, Fix128::ONE);
        assert_eq!(force, Vec3Fix::ZERO);
    }

    #[test]
    fn drag_opposes_velocity_when_submerged() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let mut body = floating_body(0);
        body.velocity = Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO);
        let force = zone.force_on(&body, Fix128::ONE);
        // Drag should be negative x (opposing velocity).
        assert!(force.x < Fix128::ZERO);
    }

    #[test]
    fn submerged_fraction_capped_at_one() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let position = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-4), Fix128::ZERO);
        // Deep inside — fraction should be 1.
        let fraction = zone.submerged_fraction(position, Fix128::from_ratio(1, 2));
        assert_eq!(fraction, Fix128::ONE);
    }

    #[test]
    fn submerged_fraction_zero_above_surface() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-5),
                Fix128::from_int(-5),
                Fix128::from_int(-5),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let position = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO);
        let fraction = zone.submerged_fraction(position, Fix128::ONE);
        assert_eq!(fraction, Fix128::ZERO);
    }

    #[test]
    fn zero_radius_has_zero_fraction() {
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(-1),
                Fix128::from_int(-1),
            ),
            max: Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let fraction = zone.submerged_fraction(Vec3Fix::ZERO, Fix128::ZERO);
        assert_eq!(fraction, Fix128::ZERO);
    }

    /// The substep stages `force_on` with the world's radius of each dynamic body in the
    /// fluid, leaves static and outside bodies alone and reports the count; a force out of
    /// range is a fault that stages nothing and keeps the count.
    #[test]
    fn the_substep_stages_the_force_with_the_worlds_radius() {
        use crate::world_participant::{ForceAccumulator, ObservationSink, SubstepCtx};
        let i = Fix128::from_int;
        let shape = ZoneShape::Aabb {
            min: Vec3Fix::new(i(-10), i(-10), i(-10)),
            max: Vec3Fix::new(i(10), i(0), i(10)),
        };
        let zone = BuoyancyZone::water_pool(shape);
        let mut inside = RigidBody::new_dynamic(Vec3Fix::new(i(0), i(-2), i(0)), Fix128::ONE);
        inside.velocity = Vec3Fix::new(Fix128::ZERO, i(-1), Fix128::ZERO);
        let outside = RigidBody::new_dynamic(Vec3Fix::new(i(50), i(-2), i(0)), Fix128::ONE);
        let fixed = RigidBody::new_static(Vec3Fix::new(i(0), i(-2), i(0)));
        let bodies = [inside, outside, fixed];
        let radius = Fix128::from_ratio(1, 2);
        let radii = [Some(radius), None, None];
        let h = Fix128::from_ratio(1, 64);
        let mut p = BuoyancyZoneParticipant::new(zone);
        let mut forces = ForceAccumulator::new(bodies.len());
        {
            let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h)
                .expect("ctx")
                .with_body_radii(&radii, Fix128::ONE);
            p.substep(&mut ctx, h).expect("substep");
        }
        assert_eq!(forces.force(0), Some(zone.force_on(&bodies[0], radius)));
        assert_eq!(forces.force(1), Some(Vec3Fix::ZERO));
        assert_eq!(forces.force(2), Some(Vec3Fix::ZERO));
        let mut sink = ObservationSink::new();
        p.observe(&mut sink);
        assert_eq!(sink.values(), &[(BUOYANCY_ZONE_OBS_BODIES, Fix128::ONE)]);

        let mut huge = zone;
        huge.density_kg_m3 = Fix128::from_int(1 << 62);
        let mut q = BuoyancyZoneParticipant::new(huge);
        let mut forces = ForceAccumulator::new(bodies.len());
        {
            let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h)
                .expect("ctx")
                .with_body_radii(&radii, Fix128::ONE);
            assert_eq!(q.substep(&mut ctx, h), Err(ParticipantFault::OutOfRange));
        }
        assert_eq!(forces.force(0), Some(Vec3Fix::ZERO));
    }
}
