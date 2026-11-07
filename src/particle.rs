//! General-Purpose Particle System
//!
//! A deterministic particle system supporting emitters, gravity, damping,
//! force fields, and particle lifetimes. Uses the deterministic RNG for
//! reproducible emission patterns.
//!
//! # MODEL — point particles under gravity, damping and force fields
//!
//! - **方程式 (equations)**: a particle is a point with position, velocity, age,
//!   lifetime, mass and an alive flag (`Particle`, l. 164–177). Per step `dt`, in
//!   this order: `v ← v + g·dt` (`step`, l. 336), `v ← d·v` with `d = damping`
//!   (l. 339), `x ← x + v·dt` (semi-implicit Euler, l. 342), `age ← age + dt`
//!   (l. 345), dead once `age ≥ lifetime` (l. 348). Emission adds `rate·dt` to a
//!   per-emitter accumulator and releases its integer part, keeping the fraction
//!   (`emit_particles`, l. 495–503); a new particle starts at the emitter
//!   position with the emitter's lifetime and mass (l. 505–510, 519–521) and
//!   velocity `speed · normalize(dir + t·r)`, `r` from
//!   `DeterministicRng::next_direction`, `t = sin(min(spread, π)/2)`, so the
//!   deflection is at most `spread/2` capped at a hemisphere; `spread = 0` gives
//!   `speed·dir` with no random draw (`compute_emission_velocity`, l. 625–650).
//!   `apply_force_field` adds `F/m · (1/60 s)` to the velocity of every live
//!   particle with `m ≠ 0`, `F` from `crate::force::field_force_at` at the
//!   particle's position and velocity (l. 556–565). `step_with_landing` applies
//!   the same velocity update (l. 450–451), samples the segment `x → x + v·dt`
//!   at `k/n`, `k = 1..=n`, `n = max(1, ⌈|v·dt| / max_travel⌉)`, the last sample
//!   being the end point itself (`first_landing`, l. 587–607), and at the first
//!   sample occupied by a target (slice order, l. 609–611) emits a
//!   `LandingEvent` (l. 460–467) and re-emits the slot from
//!   `emitters[respawn_emitter]` (l. 468–478), else moves and ages it as in
//!   `step` (l. 481–484).
//! - **単位 (units)**: SI in `Fix128` — m, s, kg, m/s; `emission_rate` in 1/s,
//!   `spread_angle` in rad, `lifetime` in s (`ParticleEmitter`, l. 124–132);
//!   `damping` is dimensionless and multiplies `v` once per call of `step`, not
//!   per second (l. 339). A `LandingTarget::Field` is probed in f32 (sample point
//!   to f32, distance and normal back), so an SDF landing's position and normal
//!   carry f32 rounding; the particle state stays `Fix128` (`probe`, l. 230–237).
//! - **保存則 (conservation)**: nothing is conserved by construction; every
//!   force is external. Per step a live particle's momentum gains `m·g·dt`
//!   (l. 336) and then loses `(1 − d)·m·(v + g·dt)` to damping (l. 339; the
//!   default `d` is `0.99`, `new`, l. 292); `apply_force_field` adds
//!   `F · (1/60 s)` per call (l. 565). A landing replaces the particle's
//!   velocity by a fresh emission velocity — no bounce, no momentum passed to
//!   the target (l. 468–474). The live count is unchanged by a landing (the
//!   slot stays alive, l. 460–478) and never exceeds `max_particles` (l. 514).
//!   Mass is constant only from one emission to the next: a landed particle
//!   takes the respawn emitter's `particle_mass` (l. 477), a dead slot revived
//!   by `emit_particles` takes that emitter's mass (`recycle_dead_particle`,
//!   l. 540); either may differ from the old mass.
//! - **適用範囲 (range)**: 3D (`Vec3Fix` state, l. 166–168); `dt = 0` returns
//!   without emitting or ageing (l. 319, 434); any other `dt` is accepted, the
//!   integrator being first order (l. 336–342). Damping is per step, so the
//!   speed reached under gravity alone is the fixed point of `v ← d·(v + g·dt)`,
//!   `|v∞| = d·|g|·dt / (1 − d)` (l. 336, 339): it grows with `dt` instead of
//!   converging as `dt → 0`; a drag field adds a fixed `F/m · (1/60 s)` per
//!   call whatever `dt` is (l. 556, 565), so the terminal speed under drag also
//!   depends on `dt`. Landing samples are at most `max_travel` apart, so a
//!   target thinner than `max_travel` along the path can be stepped over and
//!   one at least that thick cannot (l. 587–607). Panics: `max_travel ≤ 0`
//!   (l. 425), `respawn_emitter` out of range (l. 429), more than `2²⁰` samples
//!   in a step (l. 573, 597). Emission stops at the live cap (l. 514); a dead
//!   or massless particle ignores force fields (l. 559).
//! - **近似 (approximations)**: no rotation, no particle–particle interaction —
//!   the step visits each particle alone (l. 330–351); drag only as a force
//!   field, none in the step (l. 336–342, 563); no restitution or sticking on
//!   landing, the particle is re-emitted (l. 468–478); straight-line sampling
//!   inside a step, not a continuous collision test (l. 602–607); per-step
//!   damping (l. 339); force fields at a fixed `1/60 s` whatever the step
//!   (l. 556); emission count truncated to the accumulator's integer part
//!   (l. 498); direction `normalize(dir + t·r)` with a random unit `r`, not a
//!   uniform sample of the cone (l. 630–648).
//! - **検証 (validation)**: `tests/analytic_particle_wiring.rs` (every test of
//!   the file: `::single_particle_follows_the_semi_implicit_recursion` pins the
//!   3D recursion to `1e-12`; the others the emission floor, lifetime, live cap,
//!   cone, mass and lifetime carried by emission and recycling, each force law
//!   as `F/m · (1/60 s)`, seed determinism); `tests/analytic_particle_landing.rs`
//!   (every test of the file: `::free_fall_landing_matches_the_closed_form_and_sqrt_2h_over_g`
//!   pins the sampled closed form to `1e-12`, the landing time within
//!   [t* − 3dt/2, t* + dt] for dt = 1/7, 1/30, 1/64, 1/240 and `|y| < 1e-6`;
//!   `::landing_time_error_shrinks_with_dt` sweeps dt = 1/16 … 1/256 with error
//!   ≤ 3dt/2; the others the thin band, determinism, respawn with the live count
//!   conserved, equality with `step` without targets, the panics);
//!   `tests/audit_force.rs::
//!   particle_system_agrees_with_compute_force_for_directional_point_drag_and_buoyancy`,
//!   `::particle_system_agrees_with_compute_force_for_magnetic_explosion_and_vortex`
//!   (the particle impulse equals the rigid-body force law).
//!   validation missing: projectile (parabolic) flight against the continuous
//!   closed form `x = v₀ t`, `y = v₀ t − g t²/2` with a step sweep (the
//!   recursion oracle pins the discrete map only).
//!   validation missing: terminal velocity of a falling particle under drag
//!   (`v_t = m g / c` for the linear drag field; no quadratic drag exists).
//! - **段 (tiers)**: L0 only.
//!
//! # Features
//!
//! - Multiple particle emitters with configurable spread, speed, and rate
//! - Particle aging and lifetime management
//! - Gravity and linear damping
//! - Integration with the force field system
//! - Deterministic via `DeterministicRng`

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
    /// Landing position (on the surface for SDF targets, the sample point for queries)
    pub position: Vec3Fix,
    /// Outward surface normal at the landing position
    pub normal: Vec3Fix,
    /// Time from the start of the step to the first occupied sample, in `(0, dt]`
    pub time_in_step: Fix128,
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
    /// `step` would produce), and the targets are probed at each sample in order. The
    /// first occupied sample produces a [`LandingEvent`]; the slot is then refilled with
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
        for p in &mut self.particles {
            if !p.alive {
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
    for k in 1..=samples {
        let (point, time) = if k == samples {
            // the last sample is exactly the end point `step` would produce
            (start + displacement, dt)
        } else {
            let frac = Fix128::from_int(k as i64) / n;
            (start + displacement * frac, dt * frac)
        };
        for (ti, target) in targets.iter().enumerate() {
            if let Some((position, normal)) = target.probe(point) {
                return Some((ti, position, normal, time));
            }
        }
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
}
