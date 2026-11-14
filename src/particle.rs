//! General-Purpose Particle System
//!
//! A deterministic particle system supporting emitters, gravity, damping,
//! force fields, and particle lifetimes. Uses the deterministic RNG for
//! reproducible emission patterns.
//!
//! # Features
//!
//! - Multiple particle emitters with configurable spread, speed, and rate
//! - Particle aging and lifetime management
//! - Gravity and linear damping
//! - Integration with the force field system
//! - Deterministic via `DeterministicRng`
//!
//! # MODEL — landing and running off a surface
//!
//! **Equations.** A free particle moves by semi-implicit Euler, `v ← (v + g·dt)·d`,
//! `x ← x + v·dt` (`d` = [`ParticleSystem::damping`]). It lands where its straight path in
//! that step meets a [`LandingTarget`]: the path is sampled at most `max_travel` apart, the
//! piece between the last free sample and the first occupied one is halved
//! [`LANDING_BISECTIONS`] times, and the occupied end (at most `max_travel / 2^24` inside) is
//! put on the surface along the normal. The target's surface is the zero set of an SDF, with the
//! outward normal `n` taken from the SDF's gradient. The landing keeps only the tangential
//! part of the velocity, `v ← v − n (v·n)`. On the surface
//! ([`ParticleSystem::step_with_run_off`]) the particle feels gravity less its normal part,
//! `g_t = g − n (g·n)`, and a viscous drag `c` along the surface, integrated implicitly:
//! `v ← (v_t + g_t·dt) / (1 + c·dt)`. On a plane this converges to the terminal speed
//! `|g_t| / c = |g| sin α / c` (α the incline) for every `dt > 0`. The step's move `v·dt`
//! is made along the surface in pieces of at most `max_travel`, each in the tangent plane
//! where it starts, and after each piece the particle is projected back onto the surface,
//! again while a projection still moves it more than `max_travel / 2^24` (at most
//! [`RUN_OFF_PROJECTIONS`] times). It drips (becomes free, keeping its velocity) when the
//! normal where it ends satisfies `n·ĝ ≥ drip_cos`, or when the step's straight move ends
//! further than `max_travel` outside the surface (it ran past an edge; a straight move that ends
//! inside, into the other face of a crotch, is no edge and slides on). Two faces meeting in a
//! concave crease (a valley, the inner corner of a stroke) hold a drop between them: a piece of the
//! move crosses a crease when the normal turns by more than any curve the field resolves can turn
//! over it (a surface held within `max_travel` curves with a radius of at least `max_travel`, so
//! over a piece `p` its normal turns at most `p / max_travel`; the chord `|n − n'|` is compared,
//! which never exceeds the angle) and the piece goes into the second face (`fn crease`). The impact
//! on the second face takes the part of the velocity into it, `v' = v − n' (v·n')`, as a landing
//! does. Its part across the crease line `l = n × n' / |n × n'|` is `w = v' − l (v'·l)`; gravity's
//! part on the second face across the line, `p = g_B − l (g_B·l)` with `g_B = g − n' (g·n')`,
//! pulls it back to the line at `|p|` (in a level crotch `|p| = g sin θ = a`). Moving away from
//! the first face (`w·n > 0`) at `|w| > |p| dt`, it climbs the far face for more than a step and
//! goes on along the second face alone. Otherwise (pressing into the first face, or `|w| ≤ |p| dt`)
//! it is back on the line within the step: the excursion lasts `2 |w| / |p| ≤ 2 dt` and reaches
//! `|w|² / (2 |p|) ≤ |p| dt² / 2` up the face, below what one step resolves, so both faces hold
//! it: it stops at the crease (bisected to `max_travel / 2^24`), goes on along `l` and keeps only
//! `v ← l (v·l)`. The same `a dt` ends the climbs below (`N`), and the `2 dt` of the bound is this
//! last excursion. In a
//! symmetric V of face angle θ, a drop running down one face keeps `r = cos 2θ` of its speed up
//! the other (`d_A · u_B = cos² θ − sin² θ`): at θ ≥ 45° (`r ≤ 0`) it stops in the crease at once,
//! below it climbs the far face and comes back, each time with at most `r` of the speed it
//! arrived with (drag only takes more). An excursion at speed `v` lasts at most `2 v / a + 1 / c`
//! (`a = g sin θ`: the climb ends within `v / a`, and the descent from the height reached, at
//! most `v² / (2 a)` along the face, within `v / a` plus the drag time `1 / c`, since under the
//! drag `s(t) ≥ (a / c)(t − 1 / c)`). It stops climbing once what is left, `r^n v₀`, is no more
//! than one step of the pull undoes, `a dt`, after `N = ⌈ln(a dt / v₀) / ln r⌉` crossings; so
//! after first reaching the crease at `v₀` it is at rest within `Σ_{n<N} (2 r^n v₀ / a + 1 / c) +
//! 2 dt`. Landing from the first
//! occupied sample itself, a sample can lie up to `max_travel` inside, where the gradient
//! belongs to another face (near an edge, or the nearest face is not the one the path
//! crossed): one projection from there missed the surface by millimetres or put the drop on
//! another face (measured on a 20 mm deep stroke: up to 2.5 mm inside, 10 mm off the path).
//! Sliding by one straight move and one projection had the same two faults: on a curve
//! the move cuts across it (an arc of `r atan(c / r)` for a move `c`), and from deep inside
//! the projection misses the surface (measured on a curved stroke: up to 2.1 mm inside).
//!
//! **Units.** SI: positions m, velocities m/s, `g` m/s², `c` 1/s, `dt` s, `drip_cos`
//! dimensionless. The particle's state and the integration are `Fix128` (Q64.64). The surface
//! is read from the target as `f32` (an SDF's distance and normal, converted once with
//! `Fix128::from_f32`), so a run replays bit for bit when the field's own `f32` evaluation is
//! deterministic (the same code and the same `f32` operations on every machine; a field that
//! uses fused multiply-add on one machine and not on another can differ in its last bit).
//! [`ParticleSystem::state_digest`] fixes everything the next step depends on.
//!
//! **Conservation.** Mass per particle is constant: particles are neither split nor merged.
//! Energy is not conserved: drag dissipates it, and the landing discards the normal
//! velocity (a perfectly inelastic impact). Momentum is not conserved: the surface is an
//! external constraint.
//!
//! **Valid range.** Rain-sized drops on rigid surfaces given as an SDF, with `c > 0` and
//! `0 ≤ drip_cos ≤ 1`. The SDF must be close to a true distance within `max_travel`, so
//! that the projection lands on the surface. A [`LandingTarget::Query`] gives occupancy
//! only, so a particle on it is held where it landed.
//!
//! **Approximations.** No particle-particle interaction (no coalescence, no film). No
//! contact angle or hysteresis: a drop on a gentle slope always moves (pinning by
//! `ρ V g sin α < γ w (cos θ_r − cos θ_a)` is not modelled). No splash or bounce. The drag
//! is linear (a viscous film), not inertial.
//!
//! **Validation.** `tests/analytic_particle_run_off.rs`:
//! - `on_an_incline_the_speed_reaches_the_closed_form` (`|g| sin α / c` at steps of
//!   1/60, 1/240 and 1/960 s)
//! - `a_drop_that_died_and_came_back_is_free` (a surface is not carried into a new life)
//! - `on_a_sphere_the_drop_drips_at_the_cosine_it_was_told`
//! - `past_an_edge_the_drop_falls_free`
//! - `a_query_surface_holds_the_drop_where_it_is_occupied`
//! - `rain_scene_digest_is_pinned`
//! - `tests/analytic_particle_run_off_crotch.rs`: in a V valley the energy `|v|²/2 + g y` never
//!   rises; the drop comes to rest in a level crease within the bound above and stays (at once
//!   at 45° and 60°, after climbing the far face at 30°); in a gutter it runs along the crease at
//!   `g s / (√(1 + s²) c)`; on a single slope the bits are those before the crease was handled
//! - `a_steep_gutter_holds_by_the_pull_across_its_line` (unit): in a gutter whose line falls at 2
//!   (faces of slope 1) the hold compares with gravity's pull across the line `|p| = g / √30`, not
//!   all of it on the face (`g √30 / 6`): a drop between the two is released
//! - `held_drops_stay_on_square_and_curved_strokes` (wind-blown rain on a square-edged and a
//!   curved stroke 20 mm deep, normal from a 2 mm difference: every held step within
//!   `2 max_travel / 2^24` + 8 f32 ulps of the surface)
//! - `a_slide_round_a_curve_does_not_cut_across_it` (round a cylinder each step's arc is at
//!   least `|v| dt (1 − (max_travel / r)² / 3)` and at most `|v| dt`)
//!
//! `tests/analytic_particle_landing.rs`:
//! - `a_landing_is_on_the_surface_at_faces_edges_and_corners` (7262 landings of wind-blown
//!   rain on a box, 412 on edges, with the exact normal and with a 2 mm central difference:
//!   on the surface within `max_travel / 2^24` + 8 f32 ulps, on the drop's path within twice that)
//! - `one_projection_from_a_deep_sample_misses_the_surface_at_an_edge` (the bound's teeth)
//! - `the_same_calls_give_the_same_bits`
//! - `every_part_of_the_state_moves_the_digest`
//!
//! **Tiers.** L0 only.

use crate::force::{field_force_at, ForceField};
use crate::math::{Fix128, Vec3Fix};
use crate::rng::DeterministicRng;
use crate::sdf_collider::SdfField;
#[cfg(feature = "std")]
use crate::sdf_collider::{collide_point_sdf, SdfCollider};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Particle Types
// ============================================================================

/// Configuration for a particle emitter.
#[derive(Clone, Debug)]
pub struct ParticleEmitter {
    /// Emitter position (world space)
    pub position: Vec3Fix,
    /// Emission direction (normalized)
    pub direction: Vec3Fix,
    /// Full apex angle of the emission cone in radians (0 = focused beam, PI = hemisphere);
    /// the largest deflection from `direction` is `spread_angle / 2`, capped at PI/2
    pub spread_angle: Fix128,
    /// Number of particles emitted per second
    pub emission_rate: Fix128,
    /// Initial speed of emitted particles
    pub initial_speed: Fix128,
    /// Lifetime of emitted particles (seconds)
    pub lifetime: Fix128,
    /// Mass of emitted particles
    pub particle_mass: Fix128,
    /// Accumulated fractional emission (sub-frame emission tracking)
    emission_accumulator: Fix128,
}

impl ParticleEmitter {
    /// Create a new particle emitter.
    #[must_use]
    pub fn new(
        position: Vec3Fix,
        direction: Vec3Fix,
        spread_angle: Fix128,
        emission_rate: Fix128,
        initial_speed: Fix128,
        lifetime: Fix128,
        particle_mass: Fix128,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            spread_angle,
            emission_rate,
            initial_speed,
            lifetime,
            particle_mass,
            emission_accumulator: Fix128::ZERO,
        }
    }
}

/// A single particle in the system.
#[derive(Clone, Debug)]
pub struct Particle {
    /// Current position
    pub position: Vec3Fix,
    /// Current velocity
    pub velocity: Vec3Fix,
    /// Time since emission (seconds)
    pub age: Fix128,
    /// Maximum lifetime (seconds)
    pub lifetime: Fix128,
    /// Particle mass
    pub mass: Fix128,
    /// Whether this particle is active
    pub alive: bool,
}

impl Particle {
    /// Create a new active particle.
    #[must_use]
    pub const fn new(position: Vec3Fix, velocity: Vec3Fix, lifetime: Fix128, mass: Fix128) -> Self {
        Self {
            position,
            velocity,
            age: Fix128::ZERO,
            lifetime,
            mass,
            alive: true,
        }
    }
}

/// Something a moving particle can land on, for [`ParticleSystem::step_with_landing`].
///
/// Each target answers one question per sample point: "is this point occupied, and if
/// so, which way is out?".
#[non_exhaustive]
pub enum LandingTarget<'a> {
    /// A world-space signed distance field (unit scale, no transform). A sample point
    /// lands when `distance < 0`; the normal is the field's outward normal and the
    /// reported position is the point projected back onto the surface.
    Field(&'a dyn SdfField),
    /// A placed SDF collider (position, rotation, scale), tested with
    /// [`collide_point_sdf`]; the reported position is the contact's surface point.
    #[cfg(feature = "std")]
    Collider(&'a SdfCollider),
    /// A non-SDF occupant (cloth particles, a voxel mask, ...): returns the outward
    /// normal when the point is occupied, `None` when it is free. The reported
    /// position is the sample point itself.
    Query(&'a dyn Fn(Vec3Fix) -> Option<Vec3Fix>),
}

impl core::fmt::Debug for LandingTarget<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Field(_) => f.write_str("LandingTarget::Field"),
            #[cfg(feature = "std")]
            Self::Collider(_) => f.write_str("LandingTarget::Collider"),
            Self::Query(_) => f.write_str("LandingTarget::Query"),
        }
    }
}

impl LandingTarget<'_> {
    /// `(surface position, outward normal)` when `point` is occupied by this target.
    fn probe(&self, point: Vec3Fix) -> Option<(Vec3Fix, Vec3Fix)> {
        match self {
            Self::Field(field) => {
                let (x, y, z) = point.to_f32();
                let d = field.distance(x, y, z);
                if d >= 0.0 {
                    return None;
                }
                let (nx, ny, nz) = field.normal(x, y, z);
                let normal = Vec3Fix::from_f32(nx, ny, nz).normalize();
                Some((point + normal * Fix128::from_f32(-d), normal))
            }
            #[cfg(feature = "std")]
            Self::Collider(collider) => {
                collide_point_sdf(point, collider).map(|c| (c.point_b, c.normal))
            }
            Self::Query(query) => query(point).map(|normal| (point, normal)),
        }
    }

    /// `(point on the surface, outward normal)` when `point` is on this target or within
    /// `reach` of it, on either side. A query has no distance: it answers for occupied points.
    /// The signed distance of `point` from the target (negative inside), at any range; `None` for a query, which
    /// says only whether a point is occupied
    fn distance(&self, point: Vec3Fix) -> Option<Fix128> {
        match self {
            Self::Field(field) => {
                let (x, y, z) = point.to_f32();
                Some(Fix128::from_f32(field.distance(x, y, z)))
            }
            #[cfg(feature = "std")]
            Self::Collider(collider) => {
                let (x, y, z) = collider.world_to_local(point);
                Some(Fix128::from_f32(
                    collider.field.distance(x, y, z) * collider.scale_f32,
                ))
            }
            Self::Query(_) => None,
        }
    }

    fn surface(&self, point: Vec3Fix, reach: Fix128) -> Option<(Vec3Fix, Vec3Fix)> {
        let on_field = |field: &dyn SdfField, x: f32, y: f32, z: f32| {
            let d = Fix128::from_f32(field.distance(x, y, z));
            (d.abs() <= reach).then(|| (d, field.normal(x, y, z)))
        };
        match self {
            Self::Field(field) => {
                let (x, y, z) = point.to_f32();
                let (d, (nx, ny, nz)) = on_field(*field, x, y, z)?;
                let normal = Vec3Fix::from_f32(nx, ny, nz).normalize();
                Some((point - normal * d, normal))
            }
            #[cfg(feature = "std")]
            Self::Collider(collider) => {
                let (x, y, z) = collider.world_to_local(point);
                let d = Fix128::from_f32(collider.field.distance(x, y, z) * collider.scale_f32);
                if d.abs() > reach {
                    return None;
                }
                let (nx, ny, nz) = collider.field.normal(x, y, z);
                let normal = collider.local_normal_to_world(nx, ny, nz);
                Some((point - normal * d, normal))
            }
            Self::Query(query) => query(point).map(|normal| (point, normal)),
        }
    }
}

/// One particle landing on a [`LandingTarget`] during [`ParticleSystem::step_with_landing`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LandingEvent {
    /// Index into [`ParticleSystem::particles`] of the particle that landed (the slot now
    /// holds the particle re-emitted in its place)
    pub particle: usize,
    /// Index into the `targets` slice of the target that was hit
    pub target: usize,
    /// Landing position: where the step's path meets the surface, on it for SDF targets (within
    /// `max_travel / 2^`[`LANDING_BISECTIONS`] and the field's f32 reading), the occupied point
    /// nearest that crossing for queries
    pub position: Vec3Fix,
    /// Outward surface normal at the landing position
    pub normal: Vec3Fix,
    /// Time from the start of the step to the crossing (to `max_travel / 2^`[`LANDING_BISECTIONS`]
    /// of the path), in `(0, dt]`
    pub time_in_step: Fix128,
}

/// How a landed particle runs off, for [`ParticleSystem::step_with_run_off`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunOff {
    /// Viscous drag along the surface, per second: the terminal speed is `|g_t| / drag`.
    pub drag: Fix128,
    /// The particle drips when the surface normal, against the direction of gravity, reaches
    /// this cosine: 0 is a vertical wall, 1 the very underside of a surface.
    pub drip_cos: Fix128,
}

/// What happened in one [`ParticleSystem::step_with_run_off`].
#[derive(Clone, Debug, Default)]
pub struct RunOffEvents {
    /// Particles that landed on a target in this step.
    pub landed: Vec<LandingEvent>,
    /// Indices of the particles that left their surface in this step.
    pub dripped: Vec<usize>,
}

/// Which target a particle ran down, and where and at what age the run-off step left it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SurfaceRecord {
    target: usize,
    age: Fix128,
    position: Vec3Fix,
}

/// A particle system managing particles and emitters.
pub struct ParticleSystem {
    /// All particles (alive and dead)
    pub particles: Vec<Particle>,
    /// Particle emitters
    pub emitters: Vec<ParticleEmitter>,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Linear velocity damping (0 = full damping, 1 = no damping)
    pub damping: Fix128,
    /// Maximum number of particles allowed
    pub max_particles: usize,
    /// The target each particle is running down in [`Self::step_with_run_off`], by
    /// particle index, with the particle's age and position as that step left them; shorter
    /// than `particles` when the later ones are not on anything. A record counts only while
    /// the particle is alive and still at that age and position ([`Self::particle_on_surface`]):
    /// any other step, a death, a new life in the slot or a change through `particles` moves
    /// one of them, so a drop that left the surface any other way is free.
    on_surface: Vec<Option<SurfaceRecord>>,
}

impl ParticleSystem {
    /// Create a new particle system.
    ///
    /// # Arguments
    ///
    /// - `max_particles`: Maximum number of live particles
    /// - `gravity`: Gravity vector applied to all particles
    #[must_use]
    pub fn new(max_particles: usize, gravity: Vec3Fix) -> Self {
        Self {
            particles: Vec::new(),
            emitters: Vec::new(),
            gravity,
            damping: Fix128::from_ratio(99, 100),
            max_particles,
            on_surface: Vec::new(),
        }
    }

    /// Add a particle emitter and return its index.
    pub fn add_emitter(&mut self, emitter: ParticleEmitter) -> usize {
        let idx = self.emitters.len();
        self.emitters.push(emitter);
        idx
    }

    /// Count the number of alive particles.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.particles.iter().filter(|p| p.alive).count()
    }

    /// Step the particle system by `dt` seconds.
    ///
    /// 1. Emit new particles from each emitter
    /// 2. Apply gravity and damping to all alive particles
    /// 3. Integrate positions (Euler)
    /// 4. Age particles and kill expired ones
    ///
    /// The `rng` parameter ensures deterministic emission patterns.
    pub fn step(&mut self, dt: Fix128, rng: &mut DeterministicRng) {
        if dt.is_zero() {
            return;
        }

        // --- Emit new particles ---
        let num_emitters = self.emitters.len();
        for ei in 0..num_emitters {
            self.emit_particles(ei, dt, rng);
        }

        // --- Update existing particles ---
        for particle in &mut self.particles {
            if !particle.alive {
                continue;
            }

            // Apply gravity
            particle.velocity = particle.velocity + self.gravity * dt;

            // Apply damping
            particle.velocity = particle.velocity * self.damping;

            // Integrate position
            particle.position = particle.position + particle.velocity * dt;

            // Age
            particle.age = particle.age + dt;

            // Kill expired particles
            if particle.age >= particle.lifetime {
                particle.alive = false;
            }
        }
    }

    /// Step like [`step`](Self::step), but stop each particle at the first
    /// [`LandingTarget`] its path enters, report it and re-emit it from
    /// `emitters[respawn_emitter]` (rain hitting shapes and falling again).
    ///
    /// The velocity update is the same as `step` (`v += g·dt`, `v *= damping`). The
    /// straight path `x → x + v·dt` is then sampled at `n = ceil(|v·dt| / max_travel)`
    /// evenly spaced points (`k/n` for `k = 1..=n`, the last one is exactly the position
    /// `step` would produce), and the targets are probed at each sample in order. At the
    /// first occupied sample the piece of path back to the last free one is halved
    /// [`LANDING_BISECTIONS`] times and the [`LandingEvent`] is at the crossing; the slot is then refilled with
    /// a fresh particle from the respawn emitter (same position / velocity law as normal
    /// emission, drawing from `rng`), so landing never changes the number of live
    /// particles. Particles that land are not aged or expired in that step.
    ///
    /// A target whose extent along the path is longer than `max_travel` cannot be
    /// skipped, whatever `dt` is: consecutive samples are at most `max_travel` apart.
    /// The cost is `O(|v·dt| / max_travel)` probes per particle per step. With no
    /// targets the result is bit-identical to [`step`](Self::step).
    ///
    /// # Panics
    ///
    /// If `max_travel <= 0`, if `respawn_emitter` is not a valid emitter index, or if a
    /// particle's step would need more than 2^20 samples (`|v·dt| / max_travel > 2^20`).
    ///
    /// # Example
    ///
    /// ```
    /// use alice_physics_core::math::{Fix128, Vec3Fix};
    /// use alice_physics_core::particle::{LandingTarget, ParticleEmitter, ParticleSystem};
    /// use alice_physics_core::rng::DeterministicRng;
    /// use alice_physics_core::sdf_collider::ClosureSdf;
    ///
    /// // floor y = 0
    /// let floor = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    /// let mut rain = ParticleSystem::new(64, Vec3Fix::from_int(0, -10, 0));
    /// rain.damping = Fix128::ONE;
    /// rain.add_emitter(ParticleEmitter::new(
    ///     Vec3Fix::from_int(0, 5, 0),
    ///     Vec3Fix::from_int(0, -1, 0),
    ///     Fix128::from_ratio(1, 4),
    ///     Fix128::from_int(120),
    ///     Fix128::ONE,
    ///     Fix128::from_int(100),
    ///     Fix128::ONE,
    /// ));
    /// let mut rng = DeterministicRng::new(7);
    /// let targets = [LandingTarget::Field(&floor)];
    /// let mut landed = 0;
    /// for _ in 0..240 {
    ///     let events = rain.step_with_landing(
    ///         Fix128::from_ratio(1, 60),
    ///         &mut rng,
    ///         &targets,
    ///         Fix128::from_ratio(1, 20),
    ///         0,
    ///     );
    ///     for e in &events {
    ///         assert!(e.normal.y > Fix128::ZERO);
    ///     }
    ///     landed += events.len();
    /// }
    /// assert!(landed > 0);
    /// ```
    pub fn step_with_landing(
        &mut self,
        dt: Fix128,
        rng: &mut DeterministicRng,
        targets: &[LandingTarget<'_>],
        max_travel: Fix128,
        respawn_emitter: usize,
    ) -> Vec<LandingEvent> {
        assert!(
            max_travel > Fix128::ZERO,
            "step_with_landing: max_travel must be positive"
        );
        assert!(
            respawn_emitter < self.emitters.len(),
            "step_with_landing: respawn_emitter {respawn_emitter} out of range ({} emitters)",
            self.emitters.len()
        );
        let mut events = Vec::new();
        if dt.is_zero() {
            return events;
        }

        let num_emitters = self.emitters.len();
        for ei in 0..num_emitters {
            self.emit_particles(ei, dt, rng);
        }

        let gravity = self.gravity;
        let damping = self.damping;
        let respawn = &self.emitters[respawn_emitter];
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if !particle.alive {
                continue;
            }
            particle.velocity = particle.velocity + gravity * dt;
            particle.velocity = particle.velocity * damping;
            let start = particle.position;
            let displacement = particle.velocity * dt;

            let hit = if targets.is_empty() {
                None
            } else {
                first_landing(start, displacement, dt, targets, max_travel)
            };
            if let Some((target, position, normal, time_in_step)) = hit {
                events.push(LandingEvent {
                    particle: index,
                    target,
                    position,
                    normal,
                    time_in_step,
                });
                particle.position = respawn.position;
                particle.velocity = compute_emission_velocity(
                    respawn.direction,
                    respawn.initial_speed,
                    respawn.spread_angle,
                    rng,
                );
                particle.age = Fix128::ZERO;
                particle.lifetime = respawn.lifetime;
                particle.mass = respawn.particle_mass;
                continue;
            }

            particle.position = start + particle.velocity * dt;
            particle.age = particle.age + dt;
            if particle.age >= particle.lifetime {
                particle.alive = false;
            }
        }
        events
    }

    /// Step like [`Self::step_with_landing`], except that a particle that lands stays on what
    /// it hit: it runs down the surface and drips off it instead of being re-emitted.
    ///
    /// On a surface with outward normal `n`, `g` less its part along `n` accelerates the
    /// particle and `run_off.drag` slows it (implicitly, so the terminal speed on a plane is
    /// `|g_t| / drag` for any `dt`). After each move the particle is put back on the surface.
    /// It leaves as a free particle, keeping its velocity, when it gets further than
    /// `max_travel` from the surface (it ran past an edge) or when `n·ĝ >= run_off.drip_cos`
    /// (it hangs under an overhang). A particle on a [`LandingTarget::Query`] stays where the
    /// query says the point is occupied, since a query gives no distance to the surface.
    ///
    /// Particles die at the end of their lifetime wherever they are; emitters replace them.
    /// [`Self::particle_on_surface`] says which target a particle is on.
    ///
    /// # Panics
    ///
    /// When `max_travel` is not positive, or a free particle's step needs more samples than
    /// the cap of [`Self::step_with_landing`].
    pub fn step_with_run_off(
        &mut self,
        dt: Fix128,
        rng: &mut DeterministicRng,
        targets: &[LandingTarget<'_>],
        max_travel: Fix128,
        run_off: RunOff,
    ) -> RunOffEvents {
        assert!(
            max_travel > Fix128::ZERO,
            "step_with_run_off: max_travel must be positive"
        );
        let mut events = RunOffEvents::default();
        if dt.is_zero() {
            return events;
        }
        for ei in 0..self.emitters.len() {
            self.emit_particles(ei, dt, rng);
        }
        self.on_surface.resize(self.particles.len(), None);

        let gravity = self.gravity;
        let damping = self.damping;
        let down = gravity.normalize();
        let slowing = Fix128::ONE / (Fix128::ONE + run_off.drag * dt);
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if !particle.alive {
                self.on_surface[index] = None;
                continue;
            }
            let on = self.on_surface[index]
                .filter(|r| r.age == particle.age && r.position == particle.position)
                .map(|r| r.target)
                .filter(|t| *t < targets.len());
            let mut still_on = None;
            if let Some(t) = on {
                let surface = targets[t].surface(particle.position, max_travel);
                let normal = surface.map_or(down, |(_, n)| n);
                let pull = gravity - normal * gravity.dot(normal);
                let along = particle.velocity - normal * particle.velocity.dot(normal);
                particle.velocity = (along + pull * dt) * slowing;
                let travel = particle.velocity * dt;
                let moved = particle.position + travel;
                // it leaves the surface when the step's straight move ends further than `max_travel` outside it (it ran
                // past an edge); otherwise it slides along the surface and drips where the normal tips past `drip_cos`.
                // A straight move that ends deep inside (into a second face of a crotch) is no edge: it slides, and the
                // crease holds it
                let past_edge = match targets[t].distance(moved) {
                    Some(d) => d > max_travel,
                    None => targets[t].surface(moved, max_travel).is_none(),
                };
                let slid = if past_edge {
                    None
                } else {
                    slide(
                        &targets[t],
                        particle.position,
                        travel,
                        max_travel,
                        Hold {
                            velocity: particle.velocity,
                            gravity,
                            dt,
                        },
                    )
                };
                match slid {
                    Some((position, n, line)) if n.dot(down) < run_off.drip_cos => {
                        particle.position = position;
                        // held by two faces at a crease, it keeps only the velocity along the line where they meet
                        // (each face takes the part into it, as a single face does on landing)
                        if let Some(l) = line {
                            particle.velocity = l * particle.velocity.dot(l);
                        }
                        still_on = Some(t);
                    }
                    _ => {
                        particle.position = moved;
                        events.dripped.push(index);
                    }
                }
            } else {
                particle.velocity = (particle.velocity + gravity * dt) * damping;
                let start = particle.position;
                let displacement = particle.velocity * dt;
                let hit = if targets.is_empty() {
                    None
                } else {
                    first_landing(start, displacement, dt, targets, max_travel)
                };
                if let Some((target, position, normal, time_in_step)) = hit {
                    events.landed.push(LandingEvent {
                        particle: index,
                        target,
                        position,
                        normal,
                        time_in_step,
                    });
                    particle.position = position;
                    particle.velocity = particle.velocity - normal * particle.velocity.dot(normal);
                    still_on = Some(target);
                } else {
                    particle.position = start + displacement;
                }
            }
            particle.age = particle.age + dt;
            if particle.age >= particle.lifetime {
                particle.alive = false;
                still_on = None;
            }
            self.on_surface[index] = still_on.map(|target| SurfaceRecord {
                target,
                age: particle.age,
                position: particle.position,
            });
        }
        events
    }

    /// The target particle `index` is running down, `None` when it is free.
    #[must_use]
    pub fn particle_on_surface(&self, index: usize) -> Option<usize> {
        let record = self.on_surface.get(index).copied().flatten()?;
        let p = self.particles.get(index)?;
        (p.alive && p.age == record.age && p.position == record.position).then_some(record.target)
    }

    /// SHA-256 of everything that decides the next step: gravity, damping, the cap, every
    /// emitter (with its fractional emission), every particle, which surface each is on, and
    /// `rng`, which the caller holds. Equal for two systems exactly when they will go on to
    /// the same bits under the same calls.
    #[must_use]
    pub fn state_digest(&self, rng: &DeterministicRng) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        let fix = |h: &mut Sha256, f: Fix128| {
            h.update(f.hi.to_le_bytes());
            h.update(f.lo.to_le_bytes());
        };
        let vec = |h: &mut Sha256, v: Vec3Fix| {
            fix(h, v.x);
            fix(h, v.y);
            fix(h, v.z);
        };
        h.update(b"alice-physics/particle-system-digest/v1");
        let (state, inc) = rng.raw_state();
        h.update(state.to_le_bytes());
        h.update(inc.to_le_bytes());
        vec(&mut h, self.gravity);
        fix(&mut h, self.damping);
        h.update((self.max_particles as u64).to_le_bytes());
        h.update((self.emitters.len() as u64).to_le_bytes());
        for e in &self.emitters {
            vec(&mut h, e.position);
            vec(&mut h, e.direction);
            for f in [
                e.spread_angle,
                e.emission_rate,
                e.initial_speed,
                e.lifetime,
                e.particle_mass,
                e.emission_accumulator,
            ] {
                fix(&mut h, f);
            }
        }
        h.update((self.particles.len() as u64).to_le_bytes());
        for (i, p) in self.particles.iter().enumerate() {
            vec(&mut h, p.position);
            vec(&mut h, p.velocity);
            fix(&mut h, p.age);
            fix(&mut h, p.lifetime);
            fix(&mut h, p.mass);
            h.update([u8::from(p.alive)]);
            let on = self.particle_on_surface(i).map_or(u64::MAX, |t| t as u64);
            h.update(on.to_le_bytes());
        }
        h.finalize().into()
    }

    /// Emit particles from a specific emitter.
    fn emit_particles(&mut self, emitter_index: usize, dt: Fix128, rng: &mut DeterministicRng) {
        let emitter = &mut self.emitters[emitter_index];

        // Accumulate fractional emission
        emitter.emission_accumulator = emitter.emission_accumulator + emitter.emission_rate * dt;

        // Determine how many particles to emit this frame
        let to_emit = emitter.emission_accumulator.hi.max(0) as usize;
        if to_emit == 0 {
            return;
        }
        emitter.emission_accumulator =
            emitter.emission_accumulator - Fix128::from_int(to_emit as i64);

        let direction = emitter.direction;
        let speed = emitter.initial_speed;
        let lifetime = emitter.lifetime;
        let mass = emitter.particle_mass;
        let position = emitter.position;
        let spread = emitter.spread_angle;

        for _ in 0..to_emit {
            // `max_particles` bounds the number of live particles: no emission at the cap
            if self.alive_count() >= self.max_particles {
                break;
            }
            let vel = compute_emission_velocity(direction, speed, spread, rng);
            // reuse a dead slot before growing the pool, so `particles.len()` stays bounded
            if !self.recycle_dead_particle(position, vel, lifetime, mass) {
                self.particles
                    .push(Particle::new(position, vel, lifetime, mass));
            }
        }
    }

    /// Revive the first dead slot with a fresh particle; `false` when every slot is alive.
    fn recycle_dead_particle(
        &mut self,
        position: Vec3Fix,
        velocity: Vec3Fix,
        lifetime: Fix128,
        mass: Fix128,
    ) -> bool {
        for (i, p) in self.particles.iter_mut().enumerate() {
            if !p.alive {
                if let Some(record) = self.on_surface.get_mut(i) {
                    *record = None;
                }
                p.position = position;
                p.velocity = velocity;
                p.age = Fix128::ZERO;
                p.lifetime = lifetime;
                p.mass = mass;
                p.alive = true;
                return true;
            }
        }
        false
    }

    /// Apply a force field to all alive particles.
    ///
    /// The force is [`crate::force::field_force_at`] (the same laws as
    /// [`crate::force::compute_force`] for rigid bodies), applied as a velocity
    /// impulse: `v += F/m * dt`.
    /// Uses a fixed dt of 1/60 for the impulse (since force fields are
    /// typically applied once per frame).
    pub fn apply_force_field(&mut self, field: &ForceField) {
        let dt = Fix128::from_ratio(1, 60);

        for particle in &mut self.particles {
            if !particle.alive || particle.mass.is_zero() {
                continue;
            }

            let force = field_force_at(field, particle.position, particle.velocity);
            let inv_mass = Fix128::ONE / particle.mass;
            particle.velocity = particle.velocity + force * inv_mass * dt;
        }
    }
}

/// Upper bound on the samples one particle may take in one
/// [`ParticleSystem::step_with_landing`] call (`|v·dt| / max_travel`); a larger request
/// panics instead of looping for an unbounded time.
const MAX_LANDING_SAMPLES: usize = 1 << 20;

/// Halvings of the path between the last free sample and the first occupied one before a landing is
/// put on the surface: the occupied end is then at most `max_travel / 2^24` inside (1.2e-9 for a
/// `max_travel` of 0.02), below the f32 resolution the fields are read at, so the landing point no
/// longer depends on how far past the surface the sample fell nor on the field's normal there
pub const LANDING_BISECTIONS: u32 = 24;

/// Projections of a sliding particle back onto the surface after each piece of its move: repeated while the last one
/// moved it more than `max_travel / 2^LANDING_BISECTIONS`, at most this many times
pub const RUN_OFF_PROJECTIONS: u32 = 8;

/// A point put back on `target` and the outward normal there: projected along the field's normal until a projection
/// moves it no more than `max_travel / 2^LANDING_BISECTIONS` (one projection from a point well inside, where the
/// field's normal belongs to another face, does not reach the surface), `None` when it is no longer within reach
fn settle(
    target: &LandingTarget<'_>,
    point: Vec3Fix,
    max_travel: Fix128,
) -> Option<(Vec3Fix, Vec3Fix)> {
    let tolerance = max_travel / Fix128::from_int(1_i64 << LANDING_BISECTIONS);
    let mut at = target.surface(point, max_travel)?;
    let mut moved = (at.0 - point).length();
    let mut k = 1;
    while moved > tolerance && k < RUN_OFF_PROJECTIONS {
        let next = target.surface(at.0, max_travel)?;
        moved = (next.0 - at.0).length();
        at = next;
        k += 1;
    }
    Some(at)
}

/// A particle on `target` moved by `travel` along it: in pieces of at most `max_travel`, each along the surface where
/// the piece starts (the move's part in the tangent plane there, at the piece's length) and put back on the surface
/// after it ([`settle`]). One straight move then a projection cuts across a curved surface (an arc of `r atan(c / r)`
/// for a move of `c` on a radius `r`) and, from deep inside, misses it. `None` when a piece leaves its reach
fn slide(
    target: &LandingTarget<'_>,
    start: Vec3Fix,
    travel: Vec3Fix,
    max_travel: Fix128,
    hold: Hold,
) -> Option<(Vec3Fix, Vec3Fix, Option<Vec3Fix>)> {
    let length = travel.length();
    let mut at = settle(target, start, max_travel)?;
    if length.is_zero() {
        return Some((at.0, at.1, None));
    }
    let ratio = length / max_travel;
    let whole = usize::try_from(ratio.hi).unwrap_or(usize::MAX);
    let pieces = if ratio.lo == 0 {
        whole
    } else {
        whole.saturating_add(1)
    }
    .max(1);
    assert!(
        pieces <= MAX_LANDING_SAMPLES,
        "step_with_run_off: |v*dt| / max_travel = {pieces} pieces exceeds {MAX_LANDING_SAMPLES}"
    );
    let piece = length / Fix128::from_int(pieces as i64);
    // once the move meets a crease it goes on along the crease line, the only way both faces let it
    let mut line: Option<Vec3Fix> = None;
    for _ in 0..pieces {
        let (point, normal) = at;
        let step = match line {
            Some(l) => l * (travel.dot(l) / length * piece),
            None => {
                let along = travel - normal * travel.dot(normal);
                let along_length = along.length();
                if along_length.is_zero() {
                    break;
                }
                along * (piece / along_length)
            }
        };
        let next = settle(target, point + step, max_travel)?;
        if line.is_none() {
            // the face the straight step entered is the one at its end before it is settled: settled, the point
            // can land on the crease itself, where the field may name the first face again
            let entered = target
                .surface(point + step, max_travel)
                .map_or(next.1, |(_, n)| n);
            if let Some(l) = crease(
                target, normal, entered, point, step, piece, max_travel, hold,
            ) {
                // the crease on this piece: the furthest point of the straight step still outside the second face
                // (bisected to `max_travel / 2^LANDING_BISECTIONS`), and the rest of the piece along the line
                let s = outside_until(target, point, step, max_travel);
                let corner = settle(target, point + step * s, max_travel)?;
                let rest = l * (travel.dot(l) / length * piece * (Fix128::ONE - s));
                at = settle(target, corner.0 + rest, max_travel)?;
                line = Some(l);
                continue;
            }
        }
        at = next;
    }
    Some((at.0, at.1, line))
}

/// What decides whether a crease holds a drop: its velocity for the step, gravity, the step
#[derive(Clone, Copy)]
struct Hold {
    velocity: Vec3Fix,
    gravity: Vec3Fix,
    dt: Fix128,
}

/// Whether the piece `point → point + step` (on the face of outward normal `normal`) crosses a crease into a second
/// face of normal `next_normal`, and if so the crease line (unit, along `normal × next_normal`). A crease turns the
/// normal by more than any curve the field resolves can over the piece: a surface the field holds curves with a radius
/// of at least `max_travel` (its reach), so over a piece of length `p` its normal turns by at most `p / max_travel`;
/// the chord `|n − n'| = 2 sin(θ/2) ≤ θ` must exceed that. And the step must go into the second face (`step · n' <
/// 0`, the straight step ends inside it), else it is an edge the drop runs over, not a crotch that holds it. `None`
/// for a target with no distance (a query) and for faces nearly opposite (no line between them)
#[allow(
    clippy::too_many_arguments,
    reason = "the crease test needs the target, both normals, the point, the step, the piece, the reach and the hold state; a struct for one private call adds nothing"
)]
fn crease(
    target: &LandingTarget<'_>,
    normal: Vec3Fix,
    next_normal: Vec3Fix,
    point: Vec3Fix,
    step: Vec3Fix,
    piece: Fix128,
    max_travel: Fix128,
    hold: Hold,
) -> Option<Vec3Fix> {
    if matches!(target, LandingTarget::Query(_)) {
        return None;
    }
    let turn = piece / max_travel;
    let chord = (normal - next_normal).length();
    if chord <= turn || step.dot(next_normal) >= Fix128::ZERO {
        return None;
    }
    if signed_distance(target, point + step, max_travel)? >= Fix128::ZERO {
        return None;
    }
    let l = normal.cross(next_normal);
    let len = l.length();
    if len <= turn {
        return None;
    }
    let l = l / len;
    // the impact on the second face takes the part of the velocity into it (inelastic, as a landing does); what is
    // left climbs away from the first face by its part across the crease line. It climbs the far face (in a shallow
    // valley) only if that part moves away from the first face faster than one step of gravity's pull back across
    // the line undoes; slower, it is back in the crease within the step, and both faces hold it (MODEL)
    let after = hold.velocity - next_normal * hold.velocity.dot(next_normal);
    let across = after - l * after.dot(l);
    let along_face = hold.gravity - next_normal * hold.gravity.dot(next_normal);
    let pull = along_face - l * along_face.dot(l);
    if across.dot(normal) > Fix128::ZERO && across.length() > pull.length() * hold.dt {
        return None;
    }
    Some(l)
}

/// The signed distance of `point` from the target's surface (negative inside), `None` beyond its reach or for a query
fn signed_distance(
    target: &LandingTarget<'_>,
    point: Vec3Fix,
    max_travel: Fix128,
) -> Option<Fix128> {
    let (on, n) = target.surface(point, max_travel)?;
    Some((point - on).dot(n))
}

/// The largest fraction `s` of the straight `step` from `point` whose end is not inside the target (to `max_travel /
/// 2^`[`LANDING_BISECTIONS`] of distance): `0` when the step enters it at once
fn outside_until(
    target: &LandingTarget<'_>,
    point: Vec3Fix,
    step: Vec3Fix,
    max_travel: Fix128,
) -> Fix128 {
    let tolerance = max_travel / Fix128::from_int(1_i64 << LANDING_BISECTIONS);
    let inside = |s: Fix128| {
        signed_distance(target, point + step * s, max_travel).is_some_and(|d| d < -tolerance)
    };
    let (mut lo, mut hi) = (Fix128::ZERO, Fix128::ONE);
    if !inside(hi) {
        return hi;
    }
    for _ in 0..LANDING_BISECTIONS {
        let mid = (lo + hi) / Fix128::from_int(2);
        if inside(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// First occupied sample on the segment `start → start + displacement`:
/// `(target index, surface position, outward normal, time from step start)`.
///
/// Samples are at `k/n` of the segment for `k = 1..=n`, `n = ceil(|displacement| /
/// max_travel)`, so two consecutive samples are never more than `max_travel` apart.
fn first_landing(
    start: Vec3Fix,
    displacement: Vec3Fix,
    dt: Fix128,
    targets: &[LandingTarget<'_>],
    max_travel: Fix128,
) -> Option<(usize, Vec3Fix, Vec3Fix, Fix128)> {
    let length = displacement.length();
    let ratio = length / max_travel;
    let whole = usize::try_from(ratio.hi).unwrap_or(usize::MAX);
    let samples = if ratio.lo == 0 {
        whole
    } else {
        whole.saturating_add(1)
    }
    .max(1);
    assert!(
        samples <= MAX_LANDING_SAMPLES,
        "step_with_landing: |v*dt| / max_travel = {samples} samples exceeds {MAX_LANDING_SAMPLES}"
    );
    let n = Fix128::from_int(samples as i64);
    let mut before = Fix128::ZERO;
    for k in 1..=samples {
        let frac = if k == samples {
            // the last sample is exactly the end point `step` would produce
            Fix128::ONE
        } else {
            Fix128::from_int(k as i64) / n
        };
        let point = start + displacement * frac;
        for (ti, target) in targets.iter().enumerate() {
            if target.probe(point).is_some() {
                // the crossing lies between the last free sample and this one: halve that piece of the path
                // until it is `max_travel / 2^LANDING_BISECTIONS` long, then put the occupied end on the surface.
                // A single projection from the sample itself can start far inside (a sample may lie up to
                // `max_travel` past the surface) where the field's normal belongs to another face
                let (mut free, mut held) = (before, frac);
                if target.probe(start + displacement * free).is_none() {
                    for _ in 0..LANDING_BISECTIONS {
                        let mid = (free + held).half();
                        if target.probe(start + displacement * mid).is_some() {
                            held = mid;
                        } else {
                            free = mid;
                        }
                    }
                }
                let (position, normal) = target
                    .probe(start + displacement * held)
                    .unwrap_or_else(|| unreachable!("`held` is only ever an occupied point"));
                let time = if held == Fix128::ONE { dt } else { dt * held };
                return Some((ti, position, normal, time));
            }
        }
        before = frac;
    }
    None
}

/// Compute emission velocity with spread.
fn compute_emission_velocity(
    direction: Vec3Fix,
    speed: Fix128,
    spread: Fix128,
    rng: &mut DeterministicRng,
) -> Vec3Fix {
    if spread.is_zero() {
        return direction * speed;
    }

    // Add random spread using the deterministic RNG
    let rand_dir = rng.next_direction();

    // Blend between exact direction and random direction based on spread
    // spread = 0 -> exact, spread = PI -> fully random hemisphere
    // `spread` is the full apex angle of the emission cone (PI = hemisphere, half-angle
    // PI/2). d + t r with a unit random r deflects by at most asin(t), so t = sin(spread/2)
    // makes the maximum deflection exactly spread/2; beyond PI the cone stays a hemisphere.
    let full = if spread > Fix128::PI {
        Fix128::PI
    } else {
        spread
    };
    let t = full.half().sin();
    let blended = Vec3Fix::new(
        direction.x + rand_dir.x * t,
        direction.y + rand_dir.y * t,
        direction.z + rand_dir.z * t,
    )
    .normalize();

    blended * speed
}

impl core::fmt::Debug for ParticleSystem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ParticleSystem")
            .field(
                "particles",
                &format_args!("[{} items]", self.particles.len()),
            )
            .field("emitters", &format_args!("[{} items]", self.emitters.len()))
            .field("gravity", &self.gravity)
            .field("damping", &self.damping)
            .field("max_particles", &self.max_particles)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// oracle (MODEL, the crease hold): in a gutter whose crease line falls steeply (faces `y = |x| − 2 z`), most of
    /// gravity on the second face runs along the line and only its part across the line pulls a drop back to it.
    /// Computed here in f64 from the faces: `g_B = g − n_B (g·n_B)`, `p = g_B − l (g_B·l)`, `|p| = g / √30` (1.791) and
    /// `|g_B| = g √30 / 6` (8.955) m/s². A drop leaving the first face across the line at a speed between `|p| dt` and `|g_B| dt`
    /// climbs the far face for more than a step (released); at half of `|p| dt`, or pressing into the first face, it is
    /// held. Comparing with all of `g_B` would hold the first drop
    #[test]
    fn a_steep_gutter_holds_by_the_pull_across_its_line() {
        let (k, s) = (1.0_f64, 2.0_f64);
        let n = (1.0 + k * k + s * s).sqrt();
        let field = crate::sdf_collider::ClosureSdf::new(
            move |x, y, z| (y - k as f32 * x.abs() + s as f32 * z) / n as f32,
            move |x, _, _| {
                let sx = if x >= 0.0 { 1.0 } else { -1.0 };
                (
                    -k as f32 * sx / n as f32,
                    1.0 / n as f32,
                    s as f32 / n as f32,
                )
            },
        );
        let target = LandingTarget::Field(&field);
        let v3 = |a: [f64; 3]| {
            Vec3Fix::new(
                Fix128::from_f64(a[0]),
                Fix128::from_f64(a[1]),
                Fix128::from_f64(a[2]),
            )
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let sub =
            |a: [f64; 3], b: [f64; 3], t: f64| [a[0] - b[0] * t, a[1] - b[1] * t, a[2] - b[2] * t];
        let len = |a: [f64; 3]| dot(a, a).sqrt();
        let n_a = [-k / n, 1.0 / n, s / n];
        let n_b = [k / n, 1.0 / n, s / n];
        let c = [
            n_a[1] * n_b[2] - n_a[2] * n_b[1],
            n_a[2] * n_b[0] - n_a[0] * n_b[2],
            n_a[0] * n_b[1] - n_a[1] * n_b[0],
        ];
        let l = [c[0] / len(c), c[1] / len(c), c[2] / len(c)];
        let g = [0.0, -9.81, 0.0];
        let g_b = sub(g, n_b, dot(g, n_b));
        let p = sub(g_b, l, dot(g_b, l));
        assert!(
            (len(p) - 9.81 / 30f64.sqrt()).abs() < 1e-12
                && (len(g_b) - 9.81 * 30f64.sqrt() / 6.0).abs() < 1e-12
        );
        // across the line on the second face, away from the first face
        let u = [
            n_b[1] * l[2] - n_b[2] * l[1],
            n_b[2] * l[0] - n_b[0] * l[2],
            n_b[0] * l[1] - n_b[1] * l[0],
        ];
        let away = if dot(u, n_a) > 0.0 {
            u
        } else {
            [-u[0], -u[1], -u[2]]
        };
        let dt = 1.0 / 60.0;
        // a piece from the first face (x = 1 mm) into the second
        let point = v3([0.001, 0.001, 0.0]);
        let step = v3([-0.004, -0.003, 0.0]);
        let decide = |speed: f64, dir: [f64; 3]| {
            crease(
                &target,
                v3(n_a),
                v3(n_b),
                point,
                step,
                step.length(),
                Fix128::from_ratio(2, 100),
                Hold {
                    velocity: v3([dir[0] * speed, dir[1] * speed, dir[2] * speed]),
                    gravity: v3(g),
                    dt: Fix128::from_ratio(1, 60),
                },
            )
        };
        let between = 0.5 * (len(p) + len(g_b)) * dt;
        assert!(
            decide(between, away).is_none(),
            "{between} m/s away across the line (between |p| dt and |g_B| dt) is held"
        );
        assert!(
            decide(0.5 * len(p) * dt, away).is_some(),
            "half of |p| dt away across the line is released"
        );
        let into = [-away[0], -away[1], -away[2]];
        assert!(
            decide(between, into).is_some(),
            "a drop pressing into the first face is released"
        );
    }

    #[test]
    fn test_new_particle_system() {
        let ps = ParticleSystem::new(100, Vec3Fix::from_int(0, -10, 0));
        assert_eq!(ps.max_particles, 100);
        assert_eq!(ps.alive_count(), 0);
        assert!(ps.particles.is_empty());
    }

    #[test]
    fn test_add_emitter() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(10),
            Fix128::from_int(5),
            Fix128::from_int(2),
            Fix128::ONE,
        );
        let idx = ps.add_emitter(emitter);
        assert_eq!(idx, 0);
        assert_eq!(ps.emitters.len(), 1);
    }

    #[test]
    fn test_step_emits_particles() {
        let mut ps = ParticleSystem::new(1000, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(120), // 120 particles per second
            Fix128::from_int(5),
            Fix128::from_int(10),
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);
        // Step for 1/60 s: expect 120/60 = 2 particles emitted
        ps.step(Fix128::from_ratio(1, 60), &mut rng);

        assert!(ps.alive_count() > 0);
    }

    #[test]
    fn test_particles_age_and_die() {
        let mut ps = ParticleSystem::new(1000, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(100),
            Fix128::from_int(1),
            Fix128::from_ratio(1, 10), // 0.1 second lifetime
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);

        // Emit some particles
        ps.step(Fix128::from_ratio(1, 60), &mut rng);
        let alive_after_emit = ps.alive_count();
        assert!(alive_after_emit > 0);

        // Step many times until particles die
        for _ in 0..30 {
            ps.step(Fix128::from_ratio(1, 60), &mut rng);
        }

        // Some initial particles should have died by now (lifetime = 0.1s, 30 frames at 1/60 = 0.5s)
        // But new ones keep getting emitted, so we just check that aging works
        let has_dead = ps.particles.iter().any(|p| !p.alive);
        assert!(has_dead, "Some particles should have expired");
    }

    #[test]
    fn test_gravity_affects_particles() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::from_int(0, -10, 0));
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_X,
            Fix128::ZERO,
            Fix128::from_int(60),
            Fix128::ZERO, // zero initial speed
            Fix128::from_int(10),
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);
        ps.step(Fix128::from_ratio(1, 60), &mut rng);

        // Step again so gravity has time to act
        ps.step(Fix128::from_ratio(1, 60), &mut rng);

        // Particles should have negative Y velocity from gravity
        for p in &ps.particles {
            if p.alive && p.age > Fix128::ZERO {
                assert!(
                    p.velocity.y.is_negative(),
                    "Gravity should pull particles down"
                );
            }
        }
    }

    #[test]
    fn test_max_particles_limit() {
        let mut ps = ParticleSystem::new(5, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(1000), // Very high rate
            Fix128::ONE,
            Fix128::from_int(100),
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);
        ps.step(Fix128::from_ratio(1, 10), &mut rng);

        assert!(ps.alive_count() <= 5);
    }

    #[test]
    fn test_zero_dt_no_emission() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(60),
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);
        ps.step(Fix128::ZERO, &mut rng);

        assert_eq!(ps.alive_count(), 0);
    }

    #[test]
    fn test_deterministic_emission() {
        // Two systems with same config and seed should produce identical results
        let create_and_step = |seed: u64| -> Vec<(i64, u64, i64, u64)> {
            let mut ps = ParticleSystem::new(100, Vec3Fix::from_int(0, -10, 0));
            let emitter = ParticleEmitter::new(
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Y,
                Fix128::from_ratio(1, 4),
                Fix128::from_int(10),
                Fix128::from_int(5),
                Fix128::from_int(2),
                Fix128::ONE,
            );
            ps.add_emitter(emitter);

            let mut rng = DeterministicRng::new(seed);
            for _ in 0..10 {
                ps.step(Fix128::from_ratio(1, 60), &mut rng);
            }

            ps.particles
                .iter()
                .map(|p| {
                    (
                        p.position.x.hi,
                        p.position.x.lo,
                        p.position.y.hi,
                        p.position.y.lo,
                    )
                })
                .collect()
        };

        let result1 = create_and_step(12345);
        let result2 = create_and_step(12345);
        assert_eq!(result1, result2, "Particle system should be deterministic");
    }

    #[test]
    fn test_apply_force_field_drag() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::ZERO);
        ps.particles.push(Particle::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(10, 0, 0),
            Fix128::from_int(10),
            Fix128::ONE,
        ));

        let drag = ForceField::Drag {
            coefficient: Fix128::from_int(5),
        };
        ps.apply_force_field(&drag);

        // Velocity should decrease due to drag
        assert!(ps.particles[0].velocity.x < Fix128::from_int(10));
    }

    #[test]
    fn test_apply_force_field_directional() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::ZERO);
        ps.particles.push(Particle::new(
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(10),
            Fix128::ONE,
        ));

        let wind = ForceField::Directional {
            direction: Vec3Fix::UNIT_X,
            strength: Fix128::from_int(100),
        };
        ps.apply_force_field(&wind);

        // Should have gained X velocity
        assert!(ps.particles[0].velocity.x > Fix128::ZERO);
    }

    #[test]
    fn test_particle_new() {
        let p = Particle::new(
            Vec3Fix::from_int(1, 2, 3),
            Vec3Fix::from_int(4, 5, 6),
            Fix128::from_int(10),
            Fix128::ONE,
        );
        assert!(p.alive);
        assert!(p.age.is_zero());
        assert_eq!(p.position.x.hi, 1);
    }

    #[test]
    fn test_emitter_direction_normalized() {
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(3, 4, 0),
            Fix128::ZERO,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
        );
        let len = emitter.direction.length();
        let eps = Fix128::from_ratio(1, 100);
        assert!((len - Fix128::ONE).abs() < eps);
    }

    #[test]
    fn test_damping_reduces_velocity() {
        let mut ps = ParticleSystem::new(100, Vec3Fix::ZERO);
        ps.damping = Fix128::from_ratio(9, 10); // 0.9 damping
        ps.particles.push(Particle::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(10, 0, 0),
            Fix128::from_int(100),
            Fix128::ONE,
        ));

        let mut rng = DeterministicRng::new(42);
        ps.step(Fix128::from_ratio(1, 60), &mut rng);

        // Velocity should be reduced by damping factor
        assert!(ps.particles[0].velocity.x < Fix128::from_int(10));
    }

    #[test]
    fn test_recycle_dead_particles() {
        let mut ps = ParticleSystem::new(2, Vec3Fix::ZERO);
        let emitter = ParticleEmitter::new(
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
            Fix128::from_int(60),
            Fix128::ONE,
            Fix128::from_ratio(1, 60), // 1 frame lifetime
            Fix128::ONE,
        );
        ps.add_emitter(emitter);

        let mut rng = DeterministicRng::new(42);

        // Emit particles
        ps.step(Fix128::from_ratio(1, 60), &mut rng);
        // Let them die
        ps.step(Fix128::from_ratio(1, 60), &mut rng);
        // Emit again (should recycle)
        ps.step(Fix128::from_ratio(1, 60), &mut rng);

        // Total particle count should not exceed max
        assert!(ps.alive_count() <= 2);
    }

    #[cfg(feature = "std")]
    mod run_off {
        use super::*;
        use crate::sdf_collider::ClosureSdf;

        const G: f64 = 9.81;

        fn drop_at(y: f64) -> ParticleSystem {
            let mut s = ParticleSystem::new(4, Vec3Fix::from_f32(0.0, -G as f32, 0.0));
            s.damping = Fix128::ONE;
            s.particles.push(Particle::new(
                Vec3Fix::from_f32(0.0, y as f32, 0.0),
                Vec3Fix::ZERO,
                Fix128::from_int(1000),
                Fix128::ONE,
            ));
            s
        }

        fn ro(drag: f64, drip_cos: f64) -> RunOff {
            RunOff {
                drag: Fix128::from_f64(drag),
                drip_cos: Fix128::from_f64(drip_cos),
            }
        }

        fn dt() -> Fix128 {
            Fix128::from_ratio(1, 120)
        }

        fn travel() -> Fix128 {
            Fix128::from_ratio(1, 4)
        }

        /// oracle: on a 30° incline the speed down it settles at `g sin 30° / c`
        #[test]
        fn incline_terminal_speed_is_the_closed_form() {
            let (nx, ny) = (0.5f32, 0.75f32.sqrt());
            let plane =
                ClosureSdf::new(move |x, y, _| nx * x + ny * y, move |_, _, _| (nx, ny, 0.0));
            let targets = [LandingTarget::Field(&plane)];
            let mut s = drop_at(0.5);
            let mut rng = DeterministicRng::new(1);
            let mut landed = 0;
            for _ in 0..1200 {
                landed += s
                    .step_with_run_off(dt(), &mut rng, &targets, travel(), ro(2.0, 0.9))
                    .landed
                    .len();
            }
            assert_eq!(landed, 1);
            assert_eq!(s.particle_on_surface(0), Some(0));
            let want = G * 0.5 / 2.0;
            let got = s.particles[0].velocity.length().to_f64();
            assert!((got - want).abs() < want * 1e-3, "{got} vs {want}");
        }

        /// under an overhang (the normal points down) the drop leaves at once
        #[test]
        fn a_drop_under_an_overhang_drips() {
            // a ceiling at y = 1 seen from below: its outward normal points down
            let ceiling = ClosureSdf::new(|_, y, _| 1.0 - y, |_, _, _| (0.0, -1.0, 0.0));
            let floor = ClosureSdf::new(|_, y, _| y + 5.0, |_, _, _| (0.0, 1.0, 0.0));
            let targets = [LandingTarget::Field(&ceiling), LandingTarget::Field(&floor)];
            let mut s = drop_at(0.99);
            s.particles[0].velocity = Vec3Fix::from_f32(0.0, 1.0, 0.0);
            let mut rng = DeterministicRng::new(2);
            let mut dripped = 0;
            let mut landed_on_ceiling = false;
            for _ in 0..4 {
                let e = s.step_with_run_off(dt(), &mut rng, &targets, travel(), ro(2.0, 0.5));
                landed_on_ceiling |= e.landed.iter().any(|l| l.target == 0);
                dripped += e.dripped.len();
            }
            assert!(landed_on_ceiling, "it rose into the ceiling");
            assert_eq!(dripped, 1, "and dripped off it");
            assert_eq!(s.particle_on_surface(0), None);
        }

        /// a drop revived through `particles` after it died on a surface is free: no drip, it falls
        #[test]
        fn a_revived_slot_is_free() {
            let floor = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
            let targets = [LandingTarget::Field(&floor)];
            let mut s = drop_at(0.02);
            s.particles[0].lifetime = Fix128::from_ratio(1, 2);
            let mut rng = DeterministicRng::new(4);
            for _ in 0..30 {
                s.step_with_run_off(dt(), &mut rng, &targets, travel(), ro(2.0, 0.9));
            }
            assert_eq!(s.particle_on_surface(0), Some(0));
            while s.particles[0].alive {
                s.step(dt(), &mut rng);
            }
            let p = &mut s.particles[0];
            p.alive = true;
            p.age = Fix128::ZERO;
            p.position = Vec3Fix::from_f32(0.0, 2.0, 0.0);
            p.velocity = Vec3Fix::ZERO;
            assert_eq!(s.particle_on_surface(0), None);
            let e = s.step_with_run_off(dt(), &mut rng, &targets, travel(), ro(2.0, 0.9));
            assert!(e.dripped.is_empty());
            assert!(s.particles[0].position.y.to_f64() < 2.0);
        }

        /// the digest moves with the surface a drop is on, and with the generator
        #[test]
        fn the_digest_covers_the_surface_and_the_generator() {
            let floor = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
            let targets = [LandingTarget::Field(&floor)];
            let mut s = drop_at(0.01);
            s.particles[0].velocity = Vec3Fix::from_f32(0.0, -2.0, 0.0);
            let mut rng = DeterministicRng::new(5);
            let free = s.state_digest(&rng);
            s.step_with_run_off(dt(), &mut rng, &targets, travel(), ro(2.0, 0.9));
            assert_eq!(s.particle_on_surface(0), Some(0));
            let on = s.state_digest(&rng);
            assert_ne!(free, on);
            let mut moved = rng.clone();
            moved.next_u32();
            assert_ne!(s.state_digest(&rng), s.state_digest(&moved));
            assert_eq!(s.state_digest(&rng), on, "the same state, the same digest");
        }

        /// a zero step does nothing; a travel that is not positive is refused
        #[test]
        fn a_zero_step_does_nothing() {
            let mut s = drop_at(1.0);
            let mut rng = DeterministicRng::new(6);
            let before = s.state_digest(&rng);
            let e = s.step_with_run_off(Fix128::ZERO, &mut rng, &[], travel(), ro(2.0, 0.9));
            assert!(e.landed.is_empty() && e.dripped.is_empty());
            assert_eq!(s.state_digest(&rng), before);
        }

        #[test]
        #[should_panic(expected = "max_travel must be positive")]
        fn a_travel_that_is_not_positive_is_refused() {
            let mut s = drop_at(1.0);
            let mut rng = DeterministicRng::new(7);
            s.step_with_run_off(dt(), &mut rng, &[], Fix128::ZERO, ro(2.0, 0.9));
        }
    }
}
