//! Deterministic Fluid Netcode
//!
//! Extends the netcode system for deterministic fluid simulation.
//! Uses Fix128 fluid state serialization for rollback and lockstep sync.
//!
//! # Design
//!
//! Fluid state is large (N particles * position+velocity), so this module
//! provides:
//! - Delta compression (only send changed particles)
//! - Checksum validation (detect desync early)
//! - Snapshot/restore for rollback
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Fluid State Snapshot
// ============================================================================

/// Serialized fluid state for netcode
#[derive(Clone, Debug)]
pub struct FluidSnapshot {
    /// Number of particles
    pub particle_count: u32,
    /// Serialized positions (N * 3 * 16 bytes)
    pub positions: Vec<u8>,
    /// Serialized velocities (N * 3 * 16 bytes)
    pub velocities: Vec<u8>,
    /// State checksum for desync detection
    pub checksum: u64,
    /// Frame number
    pub frame: u64,
}

impl FluidSnapshot {
    /// Create snapshot from fluid particle data
    pub fn capture(
        positions: &[Vec3Fix],
        velocities: &[Vec3Fix],
        frame: u64,
    ) -> Self {
        let n = positions.len();
        let pos_data = serialize_vec3_array(positions);
        let vel_data = serialize_vec3_array(velocities);

        let checksum = compute_checksum(&pos_data, &vel_data);

        Self {
            particle_count: n as u32,
            positions: pos_data,
            velocities: vel_data,
            checksum,
            frame,
        }
    }

    /// Restore fluid state from snapshot
    pub fn restore(&self) -> Option<(Vec<Vec3Fix>, Vec<Vec3Fix>)> {
        let n = self.particle_count as usize;
        let positions = deserialize_vec3_array(&self.positions, n)?;
        let velocities = deserialize_vec3_array(&self.velocities, n)?;
        Some((positions, velocities))
    }

    /// Verify checksum matches current data
    pub fn verify(&self, positions: &[Vec3Fix], velocities: &[Vec3Fix]) -> bool {
        let pos_data = serialize_vec3_array(positions);
        let vel_data = serialize_vec3_array(velocities);
        let checksum = compute_checksum(&pos_data, &vel_data);
        checksum == self.checksum
    }

    /// Total serialized size in bytes
    pub fn size_bytes(&self) -> usize {
        4 + self.positions.len() + self.velocities.len() + 8 + 8
    }
}

// ============================================================================
// Delta Compression
// ============================================================================

/// Delta-compressed fluid state update
///
/// Only contains particles that have changed significantly
/// since the last snapshot.
#[derive(Clone, Debug)]
pub struct FluidDelta {
    /// Frame number
    pub frame: u64,
    /// Base frame this delta is relative to
    pub base_frame: u64,
    /// Changed particle indices
    pub changed_indices: Vec<u32>,
    /// New positions for changed particles
    pub positions: Vec<Vec3Fix>,
    /// New velocities for changed particles
    pub velocities: Vec<Vec3Fix>,
    /// Full state checksum (for validation)
    pub checksum: u64,
}

impl FluidDelta {
    /// Compute delta between two states
    pub fn compute(
        old_positions: &[Vec3Fix],
        old_velocities: &[Vec3Fix],
        new_positions: &[Vec3Fix],
        new_velocities: &[Vec3Fix],
        threshold: Fix128,
        base_frame: u64,
        new_frame: u64,
    ) -> Self {
        let mut changed_indices = Vec::new();
        let mut positions = Vec::new();
        let mut velocities = Vec::new();

        let threshold_sq = threshold * threshold;

        for i in 0..new_positions.len().min(old_positions.len()) {
            let pos_diff = (new_positions[i] - old_positions[i]).length_squared();
            let vel_diff = (new_velocities[i] - old_velocities[i]).length_squared();

            if pos_diff > threshold_sq || vel_diff > threshold_sq {
                changed_indices.push(i as u32);
                positions.push(new_positions[i]);
                velocities.push(new_velocities[i]);
            }
        }

        let pos_data = serialize_vec3_array(new_positions);
        let vel_data = serialize_vec3_array(new_velocities);
        let checksum = compute_checksum(&pos_data, &vel_data);

        Self {
            frame: new_frame,
            base_frame,
            changed_indices,
            positions,
            velocities,
            checksum,
        }
    }

    /// Apply delta to base state
    pub fn apply(
        &self,
        base_positions: &mut Vec<Vec3Fix>,
        base_velocities: &mut Vec<Vec3Fix>,
    ) {
        for (idx, &particle_idx) in self.changed_indices.iter().enumerate() {
            let i = particle_idx as usize;
            if i < base_positions.len() {
                base_positions[i] = self.positions[idx];
                base_velocities[i] = self.velocities[idx];
            }
        }
    }

    /// Number of changed particles
    pub fn changed_count(&self) -> usize {
        self.changed_indices.len()
    }

    /// Compression ratio (1.0 = no compression, 0.0 = perfect)
    pub fn compression_ratio(&self, total_particles: usize) -> f32 {
        if total_particles == 0 {
            return 1.0;
        }
        self.changed_indices.len() as f32 / total_particles as f32
    }
}

// ============================================================================
// Serialization Helpers
// ============================================================================

fn serialize_vec3_array(vecs: &[Vec3Fix]) -> Vec<u8> {
    let mut data = Vec::with_capacity(vecs.len() * 48); // 3 components * 16 bytes each

    for v in vecs {
        for component in &[v.x, v.y, v.z] {
            data.extend_from_slice(&component.hi.to_le_bytes());
            data.extend_from_slice(&component.lo.to_le_bytes());
        }
    }

    data
}

fn deserialize_vec3_array(data: &[u8], count: usize) -> Option<Vec<Vec3Fix>> {
    let expected = count * 48;
    if data.len() < expected {
        return None;
    }

    let mut vecs = Vec::with_capacity(count);
    let mut offset = 0;

    for _ in 0..count {
        let mut components = [Fix128::ZERO; 3];
        for c in &mut components {
            if offset + 16 > data.len() {
                return None;
            }
            let hi = i64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]);
            offset += 8;
            let lo = u64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]);
            offset += 8;
            *c = Fix128 { hi, lo };
        }
        vecs.push(Vec3Fix::new(components[0], components[1], components[2]));
    }

    Some(vecs)
}

/// FNV-1a 64-bit hash for checksum
fn compute_checksum(pos_data: &[u8], vel_data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in pos_data.iter().chain(vel_data.iter()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_roundtrip() {
        let positions = vec![
            Vec3Fix::from_int(1, 2, 3),
            Vec3Fix::from_int(4, 5, 6),
        ];
        let velocities = vec![
            Vec3Fix::from_f32(0.1, 0.2, 0.3),
            Vec3Fix::from_f32(0.4, 0.5, 0.6),
        ];

        let snapshot = FluidSnapshot::capture(&positions, &velocities, 42);
        let (restored_pos, restored_vel) = snapshot.restore().unwrap();

        assert_eq!(restored_pos.len(), 2);
        assert_eq!(restored_vel.len(), 2);
        assert_eq!(restored_pos[0].x.hi, 1);
        assert_eq!(restored_pos[1].y.hi, 5);
    }

    #[test]
    fn test_snapshot_verify() {
        let positions = vec![Vec3Fix::from_int(1, 2, 3)];
        let velocities = vec![Vec3Fix::ZERO];

        let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);
        assert!(snapshot.verify(&positions, &velocities));

        // Modified data should fail verification
        let modified = vec![Vec3Fix::from_int(99, 99, 99)];
        assert!(!snapshot.verify(&modified, &velocities));
    }

    #[test]
    fn test_delta_compression() {
        let old_pos = vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
        ];
        let old_vel = vec![Vec3Fix::ZERO; 3];

        // Only move particle 1
        let mut new_pos = old_pos.clone();
        new_pos[1] = Vec3Fix::from_int(1, 5, 0);
        let new_vel = old_vel.clone();

        let delta = FluidDelta::compute(
            &old_pos, &old_vel,
            &new_pos, &new_vel,
            Fix128::from_ratio(1, 10),
            0, 1,
        );

        assert_eq!(delta.changed_count(), 1, "Only one particle changed");
        assert!(delta.compression_ratio(3) < 0.5, "Should be well compressed");
    }

    #[test]
    fn test_delta_apply() {
        let old_pos = vec![Vec3Fix::ZERO; 3];
        let old_vel = vec![Vec3Fix::ZERO; 3];

        let new_pos = vec![
            Vec3Fix::ZERO,
            Vec3Fix::from_int(5, 5, 5),
            Vec3Fix::ZERO,
        ];
        let new_vel = old_vel.clone();

        let delta = FluidDelta::compute(
            &old_pos, &old_vel,
            &new_pos, &new_vel,
            Fix128::from_ratio(1, 10),
            0, 1,
        );

        let mut base_pos = old_pos;
        let mut base_vel = old_vel;
        delta.apply(&mut base_pos, &mut base_vel);

        assert_eq!(base_pos[1].x.hi, 5, "Delta should update changed particle");
        assert!(base_pos[0].x.is_zero(), "Unchanged particle should remain");
    }
}
