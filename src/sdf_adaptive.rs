//! Adaptive SDF Resolution
//!
//! Evaluates SDF at variable resolution based on collision proximity.
//! Near-collision regions get high-res evaluation; distant regions use cached coarse values.
//!
//! # Strategy
//!
//! - BVH broad-phase determines which bodies are near SDF surfaces
//! - Bodies near SDF → full-resolution evaluation
//! - Bodies far from SDF → cached distance (skip re-evaluation)
//! - Bodies very far → skip entirely
//!
//! Author: Moroya Sakamoto

use crate::math::Vec3Fix;
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Adaptive SDF Configuration
// ============================================================================

/// Adaptive evaluation configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveConfig {
    /// Distance threshold for high-resolution evaluation
    pub high_res_threshold: f32,
    /// Distance threshold for cached evaluation (beyond = skip)
    pub cache_threshold: f32,
    /// Maximum age (frames) before cache entry expires
    pub cache_max_age: u32,
    /// Number of refinement samples for high-res mode
    pub refinement_samples: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            high_res_threshold: 2.0,
            cache_threshold: 10.0,
            cache_max_age: 8,
            refinement_samples: 4,
        }
    }
}

// ============================================================================
// Cached SDF Entry
// ============================================================================

/// Cached SDF evaluation result
#[derive(Clone, Copy, Debug, PartialEq)]
struct CacheEntry {
    /// Cached distance value
    distance: f32,
    /// Cached normal
    normal: (f32, f32, f32),
    /// Position where this was evaluated
    position: Vec3Fix,
    /// Frame when this was last updated
    last_update: u32,
    /// Whether this entry is valid
    valid: bool,
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            distance: f32::MAX,
            normal: (0.0, 1.0, 0.0),
            position: Vec3Fix::ZERO,
            last_update: 0,
            valid: false,
        }
    }
}

// ============================================================================
// Evaluation Level
// ============================================================================

/// Resolution level for SDF evaluation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalLevel {
    /// Full evaluation with refinement (near collision)
    HighRes,
    /// Single evaluation (moderate distance)
    Standard,
    /// Use cached value (far from surface)
    Cached,
    /// Skip evaluation (very far)
    Skip,
}

// ============================================================================
// Adaptive SDF Evaluator
// ============================================================================

/// Adaptive SDF evaluation with distance-based LOD
pub struct AdaptiveSdfEvaluator {
    /// Per-body cache entries
    cache: Vec<CacheEntry>,
    /// Current frame counter
    frame: u32,
    /// Configuration
    pub config: AdaptiveConfig,
    /// Statistics: evaluations saved this frame
    pub stats_saved: usize,
    /// Statistics: total evaluations this frame
    pub stats_total: usize,
}

impl AdaptiveSdfEvaluator {
    /// Create evaluator for N bodies
    #[must_use]
    pub fn new(num_bodies: usize, config: AdaptiveConfig) -> Self {
        Self {
            cache: vec![CacheEntry::default(); num_bodies],
            frame: 0,
            config,
            stats_saved: 0,
            stats_total: 0,
        }
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        self.stats_saved = 0;
        self.stats_total = 0;
    }

    /// Resize cache for new body count
    pub fn resize(&mut self, num_bodies: usize) {
        self.cache.resize(num_bodies, CacheEntry::default());
    }

    /// Determine evaluation level for a body
    fn determine_level(&self, body_idx: usize, position: Vec3Fix) -> EvalLevel {
        if body_idx >= self.cache.len() {
            return EvalLevel::Standard;
        }

        let entry = &self.cache[body_idx];
        if !entry.valid {
            return EvalLevel::Standard;
        }

        let age = self.frame.saturating_sub(entry.last_update);
        if age > self.config.cache_max_age {
            return EvalLevel::Standard;
        }

        // Check how far the body has moved since last evaluation
        let movement = (position - entry.position).length().to_f32();

        if entry.distance > self.config.cache_threshold && movement < self.config.high_res_threshold
        {
            return EvalLevel::Skip;
        }

        if entry.distance > self.config.high_res_threshold && movement < entry.distance * 0.5 {
            return EvalLevel::Cached;
        }

        if entry.distance < self.config.high_res_threshold {
            return EvalLevel::HighRes;
        }

        EvalLevel::Standard
    }

    /// Evaluate SDF for a body with adaptive resolution
    #[cfg(feature = "std")]
    pub fn evaluate(
        &mut self,
        body_idx: usize,
        position: Vec3Fix,
        sdf: &SdfCollider,
    ) -> (f32, (f32, f32, f32)) {
        self.stats_total += 1;

        let level = self.determine_level(body_idx, position);

        match level {
            EvalLevel::Skip => {
                self.stats_saved += 1;
                if body_idx < self.cache.len() {
                    let entry = &self.cache[body_idx];
                    (entry.distance, entry.normal)
                } else {
                    (f32::MAX, (0.0, 1.0, 0.0))
                }
            }

            EvalLevel::Cached => {
                self.stats_saved += 1;
                if body_idx < self.cache.len() && self.cache[body_idx].valid {
                    let entry = &self.cache[body_idx];
                    (entry.distance, entry.normal)
                } else {
                    self.evaluate_and_cache(body_idx, position, sdf, false)
                }
            }

            EvalLevel::Standard => self.evaluate_and_cache(body_idx, position, sdf, false),

            EvalLevel::HighRes => self.evaluate_and_cache(body_idx, position, sdf, true),
        }
    }

    /// Evaluate SDF and update cache
    #[cfg(feature = "std")]
    fn evaluate_and_cache(
        &mut self,
        body_idx: usize,
        position: Vec3Fix,
        sdf: &SdfCollider,
        high_res: bool,
    ) -> (f32, (f32, f32, f32)) {
        let (lx, ly, lz) = sdf.world_to_local(position);
        let (dist, normal) = sdf.field.distance_and_normal(lx, ly, lz);
        let world_dist = dist * sdf.scale_f32;

        let final_normal = if high_res && world_dist.abs() < self.config.high_res_threshold {
            // Refined normal: use tighter finite differences
            let eps = 0.0005;
            let dx = sdf.field.distance(lx + eps, ly, lz) - sdf.field.distance(lx - eps, ly, lz);
            let dy = sdf.field.distance(lx, ly + eps, lz) - sdf.field.distance(lx, ly - eps, lz);
            let dz = sdf.field.distance(lx, ly, lz + eps) - sdf.field.distance(lx, ly, lz - eps);
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len > 1e-10 {
                (dx / len, dy / len, dz / len)
            } else {
                normal
            }
        } else {
            normal
        };

        // Update cache
        if body_idx < self.cache.len() {
            self.cache[body_idx] = CacheEntry {
                distance: world_dist,
                normal: final_normal,
                position,
                last_update: self.frame,
                valid: true,
            };
        }

        (world_dist, final_normal)
    }

    /// Get evaluation statistics as (saved, total) counts
    #[must_use]
    pub fn stats(&self) -> (usize, usize) {
        (self.stats_saved, self.stats_total)
    }

    /// Invalidate cache for a specific body (e.g., after teleport)
    pub fn invalidate(&mut self, body_idx: usize) {
        if body_idx < self.cache.len() {
            self.cache[body_idx].valid = false;
        }
    }

    /// Invalidate all cache entries
    pub fn invalidate_all(&mut self) {
        for entry in &mut self.cache {
            entry.valid = false;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::sdf_collider::{ClosureSdf, SdfCollider};

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        )
    }

    #[test]
    fn test_adaptive_evaluator() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let mut evaluator = AdaptiveSdfEvaluator::new(2, AdaptiveConfig::default());

        evaluator.begin_frame();
        let (dist, _) = evaluator.evaluate(0, Vec3Fix::from_f32(5.0, 0.0, 0.0), &sdf);
        assert!(dist > 0.0, "Should be outside sphere");

        // Second evaluation should potentially use cache
        evaluator.begin_frame();
        let (dist2, _) = evaluator.evaluate(0, Vec3Fix::from_f32(5.0, 0.0, 0.0), &sdf);
        assert!((dist2 - dist).abs() < 0.1, "Cached value should be close");
    }

    #[test]
    fn test_eval_levels() {
        let evaluator = AdaptiveSdfEvaluator::new(1, AdaptiveConfig::default());
        // Fresh cache should return Standard
        let level = evaluator.determine_level(0, Vec3Fix::ZERO);
        assert_eq!(level, EvalLevel::Standard);
    }

    #[test]
    fn test_invalidation() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let mut evaluator = AdaptiveSdfEvaluator::new(1, AdaptiveConfig::default());
        evaluator.begin_frame();
        evaluator.evaluate(0, Vec3Fix::from_f32(5.0, 0.0, 0.0), &sdf);

        evaluator.invalidate(0);
        let level = evaluator.determine_level(0, Vec3Fix::from_f32(5.0, 0.0, 0.0));
        assert_eq!(
            level,
            EvalLevel::Standard,
            "After invalidation should use Standard"
        );
    }
}
