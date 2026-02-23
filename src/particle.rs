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

use crate::force::ForceField;
use crate::math::{Fix128, Vec3Fix};
use crate::rng::DeterministicRng;

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
    /// Spread angle in radians (0 = focused beam, PI = hemisphere)
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
    pub fn new(position: Vec3Fix, velocity: Vec3Fix, lifetime: Fix128, mass: Fix128) -> Self {
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
            if self.alive_count() >= self.max_particles {
                // Try to recycle a dead particle
                if !self
                    .recycle_dead_particle(position, direction, speed, lifetime, mass, spread, rng)
                {
                    break; // No room
                }
            } else {
                let vel = compute_emission_velocity(direction, speed, spread, rng);
                self.particles
                    .push(Particle::new(position, vel, lifetime, mass));
            }
        }
    }

    /// Try to recycle a dead particle slot.
    #[allow(clippy::too_many_arguments)]
    fn recycle_dead_particle(
        &mut self,
        position: Vec3Fix,
        direction: Vec3Fix,
        speed: Fix128,
        lifetime: Fix128,
        mass: Fix128,
        spread: Fix128,
        rng: &mut DeterministicRng,
    ) -> bool {
        for p in &mut self.particles {
            if !p.alive {
                let vel = compute_emission_velocity(direction, speed, spread, rng);
                p.position = position;
                p.velocity = vel;
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
    /// The force is applied as a velocity impulse: `v += F/m * dt`.
    /// Uses a fixed dt of 1/60 for the impulse (since force fields are
    /// typically applied once per frame).
    pub fn apply_force_field(&mut self, field: &ForceField) {
        let dt = Fix128::from_ratio(1, 60);

        for particle in &mut self.particles {
            if !particle.alive || particle.mass.is_zero() {
                continue;
            }

            let force = compute_force(field, particle.position, particle.velocity);
            let inv_mass = Fix128::ONE / particle.mass;
            particle.velocity = particle.velocity + force * inv_mass * dt;
        }
    }
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
    let t = spread / Fix128::PI;
    let blended = Vec3Fix::new(
        direction.x + rand_dir.x * t,
        direction.y + rand_dir.y * t,
        direction.z + rand_dir.z * t,
    )
    .normalize();

    blended * speed
}

/// Compute force on a particle from a force field.
fn compute_force(field: &ForceField, position: Vec3Fix, velocity: Vec3Fix) -> Vec3Fix {
    match *field {
        ForceField::Directional {
            direction,
            strength,
        } => direction * strength,

        ForceField::Point {
            center,
            strength,
            repulsive,
            max_force,
        } => {
            let diff = center - position;
            let dist_sq = diff.dot(diff);
            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }
            let dist = dist_sq.sqrt();
            let dir = diff / dist;
            let mut force_mag = strength / dist_sq;
            if force_mag > max_force {
                force_mag = max_force;
            }
            if repulsive {
                dir * (Fix128::ZERO - force_mag)
            } else {
                dir * force_mag
            }
        }

        ForceField::Drag { coefficient } => velocity * (Fix128::ZERO - coefficient),

        ForceField::Buoyancy {
            surface_y,
            density,
            drag,
        } => {
            if position.y < surface_y {
                let depth = surface_y - position.y;
                let buoyancy = Vec3Fix::new(Fix128::ZERO, density * depth, Fix128::ZERO);
                let drag_force = velocity * (Fix128::ZERO - drag);
                buoyancy + drag_force
            } else {
                Vec3Fix::ZERO
            }
        }

        ForceField::Vortex {
            center,
            axis,
            strength,
            falloff_radius,
        } => {
            let diff = position - center;
            let dist = diff.length();
            if dist.is_zero() || falloff_radius.is_zero() {
                return Vec3Fix::ZERO;
            }
            let tangent = axis.cross(diff).normalize();
            let falloff = if dist < falloff_radius {
                Fix128::ONE
            } else {
                falloff_radius / dist
            };
            tangent * (strength * falloff)
        }

        ForceField::Explosion {
            center,
            strength,
            radius,
            falloff_power,
        } => {
            let diff = position - center;
            let dist = diff.length();
            if dist.is_zero() || dist > radius || radius.is_zero() {
                return Vec3Fix::ZERO;
            }
            let dir = diff / dist;
            // (1 - dist/radius)^falloff_power
            let ratio = Fix128::ONE - dist / radius;
            let mut atten = ratio;
            // Integer power approximation for falloff
            let power_int = falloff_power.hi.max(1);
            for _ in 1..power_int {
                atten = atten * ratio;
            }
            dir * (strength * atten)
        }

        ForceField::Magnetic {
            position: dipole_pos,
            moment,
            strength,
        } => {
            let diff = position - dipole_pos;
            let dist_sq = diff.dot(diff);
            if dist_sq.is_zero() {
                return Vec3Fix::ZERO;
            }
            let dist = dist_sq.sqrt();
            let r3 = dist * dist * dist;
            // Simplified dipole: force along moment direction, magnitude ~ strength / r^3
            moment.normalize() * (strength / r3)
        }
    }
}

impl core::fmt::Debug for ParticleSystem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ParticleSystem")
            .field("particles", &self.particles.len())
            .field("emitters", &self.emitters.len())
            .field("max_particles", &self.max_particles)
            .field("gravity", &self.gravity)
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
