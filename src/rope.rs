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
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Rope Configuration
// ============================================================================

/// Rope simulation configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeConfig {
    /// Number of solver iterations per step
    pub iterations: usize,
    /// Number of substeps per step
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity retention per frame (`step()` call), applied once per frame since 1.2.0
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Create a straight rope between two endpoints.
    ///
    /// # Panics
    ///
    /// Panics if `num_segments` is zero.
    #[must_use]
    pub fn new(start: Vec3Fix, end: Vec3Fix, num_segments: usize, mass_per_unit: Fix128) -> Self {
        assert!(num_segments > 0, "Rope requires at least one segment");
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
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of segments
    #[inline]
    #[must_use]
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

    /// Update pin targets that track dynamic bodies.
    ///
    /// Call this before `step()` when pins are attached to moving bodies.
    pub fn update_pin_targets(
        &mut self,
        body_positions: &[Vec3Fix],
        body_rotations: &[crate::math::QuatFix],
    ) {
        for pin in &mut self.pins {
            if let Some(idx) = pin.body_index {
                if idx < body_positions.len() && idx < body_rotations.len() {
                    pin.target =
                        body_positions[idx] + body_rotations[idx].rotate_vec(pin.local_offset);
                }
            }
        }
    }

    /// Step rope simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
        self.apply_frame_damping();
    }

    /// Step rope with SDF collision
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);

        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
        self.apply_frame_damping();
    }

    /// `config.damping` once per frame (velocity retention per `step()` call).
    ///
    /// 1.2.0: applied per substep before, which made the terminal velocity
    /// depend on `substeps` (`g·h·d/(1−d)`, `h = dt/substeps`) — the same
    /// defect as the rigid-body solver's frame damping fix.
    fn apply_frame_damping(&mut self) {
        let d = self.config.damping;
        for v in &mut self.velocities {
            *v = *v * d;
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
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Apply pin constraints (static targets; body-following is handled by update_pin_targets)
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
    #[must_use]
    pub fn current_length(&self) -> Fix128 {
        let mut length = Fix128::ZERO;
        for i in 0..self.segment_count() {
            length = length + (self.positions[i + 1] - self.positions[i]).length();
        }
        length
    }
}

impl core::fmt::Debug for Rope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Rope")
            .field(
                "positions",
                &format_args!("[{} items]", self.positions.len()),
            )
            .field(
                "prev_positions",
                &format_args!("[{} items]", self.prev_positions.len()),
            )
            .field(
                "velocities",
                &format_args!("[{} items]", self.velocities.len()),
            )
            .field(
                "inv_masses",
                &format_args!("[{} items]", self.inv_masses.len()),
            )
            .field(
                "rest_lengths",
                &format_args!("[{} items]", self.rest_lengths.len()),
            )
            .field("pins", &format_args!("[{} items]", self.pins.len()))
            .field("config", &self.config)
            .field("total_length", &self.total_length)
            .finish()
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

    /// 原点中心の単位球 SDF (f32 sqrt のみ、det_math gate 対象外)
    #[cfg(feature = "std")]
    fn unit_sphere_collider() -> crate::sdf_collider::SdfCollider {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        let field = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len > 1e-6 {
                    (x / len, y / len, z / len)
                } else {
                    (0.0, 1.0, 0.0)
                }
            },
        );
        SdfCollider::new_static(
            Box::new(field),
            Vec3Fix::ZERO,
            crate::math::QuatFix::IDENTITY,
        )
    }

    /// 単位球からの符号付き距離 (f32 oracle)
    #[cfg(feature = "std")]
    fn sphere_dist(p: Vec3Fix) -> f32 {
        let (x, y, z) = p.to_f32();
        (x * x + y * y + z * z).sqrt() - 1.0
    }

    #[cfg(feature = "std")]
    #[test]
    fn step_with_sdf_keeps_free_rope_particles_outside_unit_sphere() {
        // 長さ 2 の水平 rope (8 segment、粒子間 0.25) を y = 2 から球の上に落とす
        let make = || {
            Rope::new(
                Vec3Fix::from_int(-1, 2, 0),
                Vec3Fix::from_int(1, 2, 0),
                8,
                Fix128::ONE,
            )
        };
        let sphere = [unit_sphere_collider()];
        let mut rope = make();
        let dt = Fix128::from_ratio(1, 60);
        let mut min_dist = f32::MAX;
        for frame in 0..60 {
            rope.step_with_sdf(dt, &sphere);
            for (i, p) in rope.positions.iter().enumerate() {
                let d = sphere_dist(*p);
                min_dist = min_dist.min(d);
                assert!(
                    d >= -1e-3,
                    "frame {frame} particle {i} inside sphere: dist {d}"
                );
            }
        }
        assert!(min_dist < 0.05, "never touched: min dist {min_dist}");
        // rope 長は保たれる (rest 2、compliance 0)
        let len = rope.current_length().to_f32();
        assert!((len - 2.0).abs() < 0.2, "length {len}");

        // SDF なしでは球を貫通する frame が存在する
        let mut free = make();
        let mut free_min = f32::MAX;
        for _ in 0..60 {
            free.step(dt);
            for p in &free.positions {
                free_min = free_min.min(sphere_dist(*p));
            }
        }
        assert!(
            free_min < -0.1,
            "without SDF the rope should pass through: {free_min}"
        );

        // pinned particle (inv_mass 0) は SDF に押されない: 球内に pin した start は動かない
        let mut pinned = Rope::new(
            Vec3Fix::from_f32(0.0, 0.5, 0.0),
            Vec3Fix::from_int(0, 3, 0),
            5,
            Fix128::ONE,
        );
        pinned.pin_start();
        for _ in 0..10 {
            pinned.step_with_sdf(dt, &sphere);
        }
        assert_eq!(pinned.positions[0], Vec3Fix::from_f32(0.0, 0.5, 0.0));

        // collider が空なら step と bit 一致
        let mut a = make();
        let mut b = make();
        for _ in 0..10 {
            a.step_with_sdf(dt, &[]);
            b.step(dt);
        }
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.velocities, b.velocities);
    }
}
