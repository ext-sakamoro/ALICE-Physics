//! Rope / Cable Simulation (XPBD Distance Chain)
//!
//! Position-based rope using chained distance constraints.
//! Supports SDF collision for wrapping around geometry.
//!
//! # Features
//!
//! - XPBD distance constraints between particles
//! - SDF surface collision and friction
//! - Configurable stiffness, damping, gravity
//! - Pin constraints (attach endpoints to bodies or world)
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Rope Configuration
// ============================================================================

/// Rope simulation configuration
#[derive(Clone, Copy, Debug)]
pub struct RopeConfig {
    /// Number of solver iterations per step
    pub iterations: usize,
    /// Number of substeps per step
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity damping (0..1, 1 = no damping)
    pub damping: Fix128,
    /// Distance constraint compliance (0 = rigid)
    pub compliance: Fix128,
    /// Friction coefficient against SDF surfaces
    pub sdf_friction: Fix128,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            substeps: 4,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            compliance: Fix128::ZERO,
            sdf_friction: Fix128::from_ratio(3, 10),
        }
    }
}

/// Pin constraint: attach a particle to a fixed point or body
#[derive(Clone, Copy, Debug)]
pub struct PinConstraint {
    /// Particle index
    pub particle_index: usize,
    /// Target position (world space)
    pub target: Vec3Fix,
    /// If Some, follow this body's position + offset
    pub body_index: Option<usize>,
    /// Offset from body position (local space)
    pub local_offset: Vec3Fix,
}

// ============================================================================
// Rope
// ============================================================================

/// Rope / cable simulation
pub struct Rope {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Previous positions (for Verlet integration)
    pub prev_positions: Vec<Vec3Fix>,
    /// Particle velocities (derived)
    pub velocities: Vec<Vec3Fix>,
    /// Inverse mass per particle (0 = pinned)
    pub inv_masses: Vec<Fix128>,
    /// Rest distance between consecutive particles
    pub rest_lengths: Vec<Fix128>,
    /// Pin constraints
    pub pins: Vec<PinConstraint>,
    /// Configuration
    pub config: RopeConfig,
    /// Total rope length
    pub total_length: Fix128,
}

impl Rope {
    /// Create a straight rope between two endpoints
    pub fn new(start: Vec3Fix, end: Vec3Fix, num_segments: usize, mass_per_unit: Fix128) -> Self {
        let n = num_segments + 1; // number of particles
        let total_length = (end - start).length();
        let segment_length = total_length / Fix128::from_int(num_segments as i64);
        let particle_mass = mass_per_unit * segment_length;
        let inv_mass = if particle_mass.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / particle_mass
        };

        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let t = Fix128::from_ratio(i as i64, num_segments as i64);
            let p = Vec3Fix::new(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.z + (end.z - start.z) * t,
            );
            positions.push(p);
        }

        let rest_lengths = vec![segment_length; num_segments];
        let inv_masses = vec![inv_mass; n];

        Self {
            prev_positions: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            positions,
            inv_masses,
            rest_lengths,
            pins: Vec::new(),
            config: RopeConfig::default(),
            total_length,
        }
    }

    /// Number of particles
    #[inline]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of segments
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.rest_lengths.len()
    }

    /// Add a pin constraint
    pub fn add_pin(&mut self, pin: PinConstraint) {
        self.inv_masses[pin.particle_index] = Fix128::ZERO;
        self.pins.push(pin);
    }

    /// Pin the first particle to its current position
    pub fn pin_start(&mut self) {
        let pos = self.positions[0];
        self.add_pin(PinConstraint {
            particle_index: 0,
            target: pos,
            body_index: None,
            local_offset: Vec3Fix::ZERO,
        });
    }

    /// Pin the last particle to its current position
    pub fn pin_end(&mut self) {
        let last = self.particle_count() - 1;
        let pos = self.positions[last];
        self.add_pin(PinConstraint {
            particle_index: last,
            target: pos,
            body_index: None,
            local_offset: Vec3Fix::ZERO,
        });
    }

    /// Step rope simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
    }

    /// Step rope with SDF collision
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();

        // 1. Predict positions (Verlet integration)
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            self.prev_positions[i] = self.positions[i];
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.velocities[i] = self.velocities[i] * self.config.damping;
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Apply pin constraints
        for pin in &self.pins {
            self.positions[pin.particle_index] = pin.target;
        }

        // 3. Solve distance constraints
        for _ in 0..self.config.iterations {
            self.solve_distance_constraints(dt);
        }

        // 4. Update velocities
        let inv_dt = Fix128::ONE / dt;
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.velocities[i] = (self.positions[i] - self.prev_positions[i]) * inv_dt;
        }
    }

    /// Solve distance constraints between consecutive particles
    fn solve_distance_constraints(&mut self, dt: Fix128) {
        for i in 0..self.segment_count() {
            let p0 = self.positions[i];
            let p1 = self.positions[i + 1];
            let w0 = self.inv_masses[i];
            let w1 = self.inv_masses[i + 1];

            let w_sum = w0 + w1;
            if w_sum.is_zero() {
                continue;
            }

            let delta = p1 - p0;
            let dist = delta.length();
            if dist.is_zero() {
                continue;
            }

            let rest = self.rest_lengths[i];
            let error = dist - rest;

            let compliance_term = self.config.compliance / (dt * dt);
            let lambda = error / (w_sum + compliance_term);
            let correction = delta / dist * lambda;

            if !w0.is_zero() {
                self.positions[i] = self.positions[i] + correction * w0;
            }
            if !w1.is_zero() {
                self.positions[i + 1] = self.positions[i + 1] - correction * w1;
            }
        }
    }

    /// Resolve SDF collisions for all particles
    #[cfg(feature = "std")]
    fn resolve_sdf_collisions(&mut self, sdf_colliders: &[SdfCollider]) {
        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            for sdf in sdf_colliders {
                let (lx, ly, lz) = sdf.world_to_local(self.positions[i]);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;

                if dist < 0.0 {
                    // Penetrating: push out along gradient
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let depth = Fix128::from_f32(-dist);

                    self.positions[i] = self.positions[i] + normal * depth;

                    // Friction: dampen tangential velocity
                    let vel = self.velocities[i];
                    let vn = normal * vel.dot(normal);
                    let vt = vel - vn;
                    self.velocities[i] =
                        vn * Fix128::from_f32(-0.1) + vt * (Fix128::ONE - self.config.sdf_friction);
                }
            }
        }
    }

    /// Get current rope length (sum of segment distances)
    pub fn current_length(&self) -> Fix128 {
        let mut length = Fix128::ZERO;
        for i in 0..self.segment_count() {
            length = length + (self.positions[i + 1] - self.positions[i]).length();
        }
        length
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let rope = Rope::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(10, 0, 0),
            20,
            Fix128::from_ratio(1, 10),
        );

        assert_eq!(rope.particle_count(), 21);
        assert_eq!(rope.segment_count(), 20);
    }

    #[test]
    fn test_rope_pins() {
        let mut rope = Rope::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(5, 0, 0),
            10,
            Fix128::from_ratio(1, 10),
        );

        rope.pin_start();
        rope.pin_end();

        assert_eq!(rope.pins.len(), 2);
        assert!(rope.inv_masses[0].is_zero(), "Start should be pinned");
        assert!(rope.inv_masses[10].is_zero(), "End should be pinned");
    }

    #[test]
    fn test_rope_gravity() {
        let mut rope = Rope::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(5, 0, 0),
            10,
            Fix128::from_ratio(1, 10),
        );
        rope.pin_start();
        rope.pin_end();

        // Simulate for 1 second
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            rope.step(dt);
        }

        // Middle particles should sag below the endpoints
        let mid = rope.positions[5];
        assert!(mid.y < Fix128::ZERO, "Rope should sag under gravity");
    }

    #[test]
    fn test_rope_length_preservation() {
        let mut rope = Rope::new(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(5, 0, 0),
            10,
            Fix128::from_ratio(1, 10),
        );
        rope.pin_start();
        rope.pin_end();
        rope.config.iterations = 16;

        let initial_length = rope.total_length;

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            rope.step(dt);
        }

        let current_length = rope.current_length();
        let error = (current_length - initial_length).abs();
        let tolerance = initial_length * Fix128::from_ratio(1, 10); // 10% tolerance
        assert!(
            error < tolerance,
            "Rope length should be approximately preserved: initial={:?}, current={:?}",
            initial_length.to_f32(),
            current_length.to_f32()
        );
    }
}
