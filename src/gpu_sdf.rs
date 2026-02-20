//! GPU-Accelerated SDF Evaluation (Compute Shader Interface)
//!
//! Provides data structures and dispatch logic for evaluating
//! SDF collision queries on the GPU. Pairs with ALICE-SDF's `gpu` feature.
//!
//! # Design
//!
//! CPU side prepares query buffers (positions, radii) and dispatches
//! to GPU compute shader. GPU evaluates all SDF queries in parallel
//! and writes back (distance, normal) results.
//!
//! This module defines the buffer layouts and dispatch interface.
//! Actual GPU execution requires ALICE-SDF's wgpu/vulkan backend.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// GPU Buffer Layouts
// ============================================================================

/// Per-query input data (CPU → GPU)
///
/// Packed for GPU transfer: 16 bytes per query
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct GpuSdfQuery {
    /// Query position (x, y, z)
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Z position
    pub z: f32,
    /// Query radius (for sphere-SDF test, 0 for point query)
    pub radius: f32,
}

/// Per-query output data (GPU → CPU)
///
/// Packed for GPU transfer: 16 bytes per result
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct GpuSdfResult {
    /// Signed distance to SDF surface
    pub distance: f32,
    /// Surface normal X
    pub normal_x: f32,
    /// Surface normal Y
    pub normal_y: f32,
    /// Surface normal Z
    pub normal_z: f32,
}

/// GPU dispatch configuration
#[derive(Clone, Copy, Debug)]
pub struct GpuDispatchConfig {
    /// Workgroup size (typically 64 or 256)
    pub workgroup_size: u32,
    /// Maximum queries per dispatch
    pub max_queries: u32,
    /// Whether to compute normals (more expensive)
    pub compute_normals: bool,
}

impl Default for GpuDispatchConfig {
    fn default() -> Self {
        Self {
            workgroup_size: 64,
            max_queries: 65536,
            compute_normals: true,
        }
    }
}

// ============================================================================
// GPU SDF Batch
// ============================================================================

/// Batch of SDF queries prepared for GPU dispatch
pub struct GpuSdfBatch {
    /// Input query buffer
    pub queries: Vec<GpuSdfQuery>,
    /// Output result buffer
    pub results: Vec<GpuSdfResult>,
    /// Body index for each query (to map results back)
    pub body_indices: Vec<usize>,
    /// Dispatch config
    pub config: GpuDispatchConfig,
}

impl GpuSdfBatch {
    /// Create a new empty batch
    pub fn new(config: GpuDispatchConfig) -> Self {
        Self {
            queries: Vec::new(),
            results: Vec::new(),
            body_indices: Vec::new(),
            config,
        }
    }

    /// Add a query to the batch
    pub fn add_query(&mut self, body_idx: usize, position: Vec3Fix, radius: Fix128) {
        let (x, y, z) = position.to_f32();
        self.queries.push(GpuSdfQuery {
            x, y, z,
            radius: radius.to_f32(),
        });
        self.body_indices.push(body_idx);
    }

    /// Number of queries in the batch
    #[inline]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }

    /// Calculate number of workgroups needed
    pub fn num_workgroups(&self) -> u32 {
        let n = self.queries.len() as u32;
        (n + self.config.workgroup_size - 1) / self.config.workgroup_size
    }

    /// Prepare output buffer (allocate space for results)
    pub fn prepare_output(&mut self) {
        self.results.resize(self.queries.len(), GpuSdfResult::default());
    }

    /// Clear batch for reuse
    pub fn clear(&mut self) {
        self.queries.clear();
        self.results.clear();
        self.body_indices.clear();
    }

    /// Get raw query buffer as bytes (for GPU upload)
    pub fn query_bytes(&self) -> &[u8] {
        // SAFETY: GpuSdfQuery is repr(C, align(16)) with no padding or uninitialized bytes.
        // The pointer is derived from a valid Vec<GpuSdfQuery>, and the byte length equals
        // queries.len() * size_of::<GpuSdfQuery>(). The returned slice lifetime is tied to &self.
        unsafe {
            core::slice::from_raw_parts(
                self.queries.as_ptr() as *const u8,
                self.queries.len() * core::mem::size_of::<GpuSdfQuery>(),
            )
        }
    }

    /// Get mutable raw result buffer as bytes (for GPU readback)
    pub fn result_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: GpuSdfResult is repr(C, align(16)) with no padding or uninitialized bytes
        // (all fields are f32, default-initialized). The pointer is derived from a valid
        // Vec<GpuSdfResult>, and the byte length equals results.len() * size_of::<GpuSdfResult>().
        // The returned slice lifetime is tied to &mut self, preventing aliased access.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.results.as_mut_ptr() as *mut u8,
                self.results.len() * core::mem::size_of::<GpuSdfResult>(),
            )
        }
    }

    /// Check results for collisions and return contacts
    pub fn extract_contacts(&self, collision_radius: f32) -> Vec<GpuSdfContact> {
        let mut contacts = Vec::new();

        for (i, result) in self.results.iter().enumerate() {
            let penetration = collision_radius - result.distance;
            if penetration > 0.0 {
                contacts.push(GpuSdfContact {
                    body_index: self.body_indices[i],
                    penetration,
                    normal: (result.normal_x, result.normal_y, result.normal_z),
                    distance: result.distance,
                });
            }
        }

        contacts
    }
}

/// Contact from GPU SDF evaluation
#[derive(Clone, Copy, Debug)]
pub struct GpuSdfContact {
    /// Body index
    pub body_index: usize,
    /// Penetration depth
    pub penetration: f32,
    /// Surface normal (pointing outward)
    pub normal: (f32, f32, f32),
    /// Signed distance
    pub distance: f32,
}

// ============================================================================
// CPU Fallback (for testing without GPU)
// ============================================================================

/// Execute batch on CPU (fallback when GPU is unavailable)
#[cfg(feature = "std")]
pub fn execute_batch_cpu(
    batch: &mut GpuSdfBatch,
    sdf: &dyn crate::sdf_collider::SdfField,
) {
    batch.prepare_output();

    for (i, query) in batch.queries.iter().enumerate() {
        let dist = sdf.distance(query.x, query.y, query.z);

        let (nx, ny, nz) = if batch.config.compute_normals {
            sdf.normal(query.x, query.y, query.z)
        } else {
            (0.0, 0.0, 0.0)
        };

        batch.results[i] = GpuSdfResult {
            distance: dist,
            normal_x: nx,
            normal_y: ny,
            normal_z: nz,
        };
    }
}

// ============================================================================
// WGSL Shader Source (for reference / code generation)
// ============================================================================

/// WGSL compute shader source for SDF evaluation.
///
/// This is provided as a reference; actual GPU dispatch uses ALICE-SDF's
/// GPU backend which compiles SDF nodes to WGSL/SPIR-V.
pub const SDF_EVAL_WGSL: &str = r#"
struct Query {
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
};

struct Result {
    distance: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
};

@group(0) @binding(0) var<storage, read> queries: array<Query>;
@group(0) @binding(1) var<storage, read_write> results: array<Result>;

// SDF evaluation function (generated per-SDF)
fn sdf_distance(x: f32, y: f32, z: f32) -> f32 {
    // Placeholder - replaced by ALICE-SDF code generation
    return length(vec3(x, y, z)) - 1.0;
}

fn sdf_normal(x: f32, y: f32, z: f32) -> vec3<f32> {
    let eps = 0.001;
    let dx = sdf_distance(x + eps, y, z) - sdf_distance(x - eps, y, z);
    let dy = sdf_distance(x, y + eps, z) - sdf_distance(x, y - eps, z);
    let dz = sdf_distance(x, y, z + eps) - sdf_distance(x, y, z - eps);
    return normalize(vec3(dx, dy, dz));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&queries) {
        return;
    }
    let q = queries[idx];
    let d = sdf_distance(q.x, q.y, q.z);
    let n = sdf_normal(q.x, q.y, q.z);
    results[idx] = Result(d, n.x, n.y, n.z);
}
"#;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    #[test]
    fn test_gpu_batch_creation() {
        let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());

        batch.add_query(0, Vec3Fix::from_f32(1.0, 0.0, 0.0), Fix128::from_f32(0.5));
        batch.add_query(1, Vec3Fix::from_f32(0.0, 2.0, 0.0), Fix128::from_f32(0.5));

        assert_eq!(batch.query_count(), 2);
        assert_eq!(batch.num_workgroups(), 1);
    }

    #[test]
    fn test_cpu_fallback() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 { (0.0, 1.0, 0.0) } else { (x / len, y / len, z / len) }
            },
        );

        let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
        batch.add_query(0, Vec3Fix::from_f32(2.0, 0.0, 0.0), Fix128::from_f32(0.5));
        batch.add_query(1, Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::from_f32(0.5));

        execute_batch_cpu(&mut batch, &sphere);

        // Point at (2,0,0): distance = 1.0 (outside)
        assert!(batch.results[0].distance > 0.0, "Should be outside sphere");
        // Point at (0.5,0,0): distance = -0.5 (inside)
        assert!(batch.results[1].distance < 0.0, "Should be inside sphere");
    }

    #[test]
    fn test_extract_contacts() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 { (0.0, 1.0, 0.0) } else { (x / len, y / len, z / len) }
            },
        );

        let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
        batch.add_query(0, Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::from_f32(0.5));

        execute_batch_cpu(&mut batch, &sphere);

        let contacts = batch.extract_contacts(0.5);
        assert!(!contacts.is_empty(), "Should detect contact for body inside sphere");
    }

    #[test]
    fn test_buffer_layout() {
        assert_eq!(core::mem::size_of::<GpuSdfQuery>(), 16);
        assert_eq!(core::mem::size_of::<GpuSdfResult>(), 16);
    }
}
