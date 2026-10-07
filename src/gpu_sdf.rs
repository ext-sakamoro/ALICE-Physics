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
#[derive(Clone, Copy, Debug, PartialEq)]
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

const _: () = {
    assert!(core::mem::size_of::<GpuSdfQuery>() == 16);
    assert!(core::mem::size_of::<GpuSdfResult>() == 16);
};

/// GPU dispatch configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    #[must_use]
    pub const fn new(config: GpuDispatchConfig) -> Self {
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
            x,
            y,
            z,
            radius: radius.to_f32(),
        });
        self.body_indices.push(body_idx);
    }

    /// Number of queries in the batch
    #[inline]
    #[must_use]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }

    /// Calculate number of workgroups needed
    #[must_use]
    pub fn num_workgroups(&self) -> u32 {
        let n = self.queries.len() as u32;
        n.div_ceil(self.config.workgroup_size)
    }

    /// Prepare output buffer (allocate space for results)
    pub fn prepare_output(&mut self) {
        self.results
            .resize(self.queries.len(), GpuSdfResult::default());
    }

    /// Clear batch for reuse
    pub fn clear(&mut self) {
        self.queries.clear();
        self.results.clear();
        self.body_indices.clear();
    }

    /// Get raw query buffer as bytes (for GPU upload)
    #[must_use]
    pub fn query_bytes(&self) -> &[u8] {
        // SAFETY: GpuSdfQuery is repr(C, align(16)) with no padding or uninitialized bytes.
        // The pointer is derived from a valid Vec<GpuSdfQuery>, and the byte length equals
        // queries.len() * size_of::<GpuSdfQuery>(). The returned slice lifetime is tied to &self.
        unsafe {
            core::slice::from_raw_parts(
                self.queries.as_ptr().cast::<u8>(),
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
                self.results.as_mut_ptr().cast::<u8>(),
                self.results.len() * core::mem::size_of::<GpuSdfResult>(),
            )
        }
    }

    /// Check results for collisions and return contacts.
    ///
    /// A query is a sphere of its own `radius` (0 for a point) grown by the
    /// common `collision_radius`: it penetrates by
    /// `collision_radius + radius − distance` when that is positive.
    #[must_use]
    pub fn extract_contacts(&self, collision_radius: f32) -> Vec<GpuSdfContact> {
        let mut contacts = Vec::new();

        for (i, result) in self.results.iter().enumerate() {
            let radius = self.queries.get(i).map_or(0.0, |q| q.radius);
            let penetration = collision_radius + radius - result.distance;
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

// ============================================================================
// GPU Instancing — Instanced SDF Batch & Multi-Dispatch
// ============================================================================

/// Instanced SDF query batch — evaluates many queries against one SDF in a
/// single GPU kernel dispatch.
///
/// Grouping queries by `sdf_id` minimises the number of kernel launches and
/// avoids redundant SDF binding changes on the GPU. Pass a collection of these
/// to [`GpuSdfMultiDispatch`] to assemble a full frame's worth of queries.
#[repr(C, align(16))]
pub struct GpuSdfInstancedBatch {
    /// Identifier of the SDF to evaluate (maps to a GPU buffer / binding slot).
    pub sdf_id: u32,
    /// Number of queries in `queries` (mirrors `queries.len()` for C FFI).
    pub query_count: u32,
    /// Query inputs for this SDF instance.
    pub queries: Vec<GpuSdfQuery>,
}

/// Multi-SDF dispatch — groups queries by SDF for the minimum number of GPU
/// kernel calls per frame (one dispatch per unique `sdf_id`).
///
/// # Usage
///
/// ```
/// use alice_physics::gpu_sdf::{GpuSdfMultiDispatch, GpuSdfQuery};
///
/// let mut dispatch = GpuSdfMultiDispatch::new();
/// let sphere_queries = vec![GpuSdfQuery { x: 0.0, y: 0.0, z: 0.0, radius: 0.5 }];
/// let box_queries = vec![GpuSdfQuery { x: 1.0, y: 1.0, z: 1.0, radius: 0.0 }];
/// dispatch.add_batch(0, sphere_queries);
/// dispatch.add_batch(1, box_queries);
///
/// assert_eq!(dispatch.total_queries(), 2);
/// assert_eq!(dispatch.total_dispatches(), 2);
/// ```
pub struct GpuSdfMultiDispatch {
    /// One entry per unique SDF — ordered by insertion.
    pub batches: Vec<GpuSdfInstancedBatch>,
}

impl GpuSdfMultiDispatch {
    /// Create an empty multi-dispatch container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            batches: Vec::new(),
        }
    }

    /// Append a batch of queries for a single SDF.
    ///
    /// `sdf_id` identifies which GPU-side SDF buffer to bind during this
    /// dispatch. Queries for an `sdf_id` that already has a batch are appended
    /// to it (in call order), so there is one batch, and one dispatch, per
    /// unique `sdf_id`; batches keep the order of their first `add_batch`.
    pub fn add_batch(&mut self, sdf_id: u32, mut queries: Vec<GpuSdfQuery>) {
        if let Some(b) = self.batches.iter_mut().find(|b| b.sdf_id == sdf_id) {
            b.queries.append(&mut queries);
            b.query_count = b.queries.len() as u32;
            return;
        }
        self.batches.push(GpuSdfInstancedBatch {
            sdf_id,
            query_count: queries.len() as u32,
            queries,
        });
    }

    /// Total number of individual SDF point queries across all batches.
    #[must_use]
    pub fn total_queries(&self) -> usize {
        self.batches.iter().map(|b| b.queries.len()).sum()
    }

    /// Number of GPU kernel dispatches that will be issued (one per batch, so
    /// one per unique `sdf_id`).
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.batches.len()
    }
}

impl Default for GpuSdfMultiDispatch {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Kernel Auto-Dispatch — SIMD-width aligned batch size
// ============================================================================

/// Returns the optimal CPU-side pre-processing batch size.
///
/// Equals [`crate::math::SIMD_WIDTH`] so inner loops that feed data to SIMD
/// registers process exactly one register's worth of items per iteration —
/// no padding, no partial loads.
///
/// Use this when chunking queries before uploading to the GPU:
///
/// ```
/// use alice_physics::gpu_sdf::batch_size;
///
/// let queries = vec![1.0f32; 64];
/// for chunk in queries.chunks(batch_size()) {
///     assert!(chunk.len() <= batch_size());
/// }
/// ```
#[inline(always)]
#[must_use]
pub const fn batch_size() -> usize {
    crate::math::SIMD_WIDTH
}

/// Contact from GPU SDF evaluation
#[derive(Clone, Copy, Debug, PartialEq)]
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
pub fn execute_batch_cpu(batch: &mut GpuSdfBatch, sdf: &dyn crate::sdf_collider::SdfField) {
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
pub const SDF_EVAL_WGSL: &str = r"
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
";

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
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
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
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
        batch.add_query(0, Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::from_f32(0.5));

        execute_batch_cpu(&mut batch, &sphere);

        let contacts = batch.extract_contacts(0.5);
        assert!(
            !contacts.is_empty(),
            "Should detect contact for body inside sphere"
        );
    }

    #[test]
    fn test_buffer_layout() {
        assert_eq!(core::mem::size_of::<GpuSdfQuery>(), 16);
        assert_eq!(core::mem::size_of::<GpuSdfResult>(), 16);
    }

    fn q(x: f32, radius: f32) -> GpuSdfQuery {
        GpuSdfQuery {
            x,
            y: 0.0,
            z: 0.0,
            radius,
        }
    }

    #[test]
    fn multi_dispatch_merges_batches_by_sdf_id_in_call_order() {
        let mut d = GpuSdfMultiDispatch::default();
        assert_eq!(d.total_queries(), 0);
        assert_eq!(d.total_dispatches(), 0);
        d.add_batch(7, vec![q(1.0, 0.0)]);
        d.add_batch(3, vec![q(2.0, 0.0), q(3.0, 0.0)]);
        d.add_batch(7, vec![q(4.0, 0.5), q(5.0, 0.0)]);
        // one dispatch per unique sdf_id, in order of first appearance
        assert_eq!(d.total_dispatches(), 2);
        assert_eq!(d.total_queries(), 5);
        assert_eq!(d.batches[0].sdf_id, 7);
        assert_eq!(d.batches[1].sdf_id, 3);
        // the repeated id appended to its batch and query_count tracks len
        assert_eq!(d.batches[0].query_count, 3);
        assert_eq!(
            d.batches[0].queries,
            vec![q(1.0, 0.0), q(4.0, 0.5), q(5.0, 0.0)]
        );
        assert_eq!(d.batches[1].query_count, 2);
    }

    #[test]
    fn byte_views_cover_every_query_and_result() {
        let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
        batch.add_query(4, Vec3Fix::from_f32(1.5, -2.0, 0.25), Fix128::from_f32(0.5));
        batch.add_query(9, Vec3Fix::from_f32(0.0, 3.0, 0.0), Fix128::ZERO);
        let bytes = batch.query_bytes();
        assert_eq!(bytes.len(), 2 * 16);
        let f = |k: usize| f32::from_ne_bytes([bytes[k], bytes[k + 1], bytes[k + 2], bytes[k + 3]]);
        assert_eq!([f(0), f(4), f(8), f(12)], [1.5, -2.0, 0.25, 0.5]);
        assert_eq!([f(16), f(20), f(24), f(28)], [0.0, 3.0, 0.0, 0.0]);

        // results are empty until prepared; a GPU readback writes through the view
        assert!(batch.result_bytes_mut().is_empty());
        batch.prepare_output();
        let out = batch.result_bytes_mut();
        assert_eq!(out.len(), 2 * 16);
        out[16..20].copy_from_slice(&(-0.75f32).to_ne_bytes());
        out[24..28].copy_from_slice(&1.0f32.to_ne_bytes());
        assert_eq!(batch.results[1].distance, -0.75);
        assert_eq!(batch.results[1].normal_y, 1.0);
        assert_eq!(batch.results[0].distance, 0.0);

        batch.clear();
        assert_eq!(batch.query_count(), 0);
        assert!(batch.results.is_empty());
        assert!(batch.body_indices.is_empty());
        assert!(batch.query_bytes().is_empty());
        assert_eq!(batch.num_workgroups(), 0);
    }

    #[test]
    fn cpu_fallback_without_normals_and_contact_depth() {
        // plane y = 0, normals disabled: distance only, normal left at zero
        let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
        let mut batch = GpuSdfBatch::new(GpuDispatchConfig {
            compute_normals: false,
            ..GpuDispatchConfig::default()
        });
        batch.add_query(2, Vec3Fix::from_f32(0.0, 0.25, 0.0), Fix128::from_f32(0.5));
        batch.add_query(5, Vec3Fix::from_f32(0.0, 2.0, 0.0), Fix128::ZERO);
        execute_batch_cpu(&mut batch, &plane);
        assert_eq!(batch.results[0].distance, 0.25);
        assert_eq!(
            (
                batch.results[0].normal_x,
                batch.results[0].normal_y,
                batch.results[0].normal_z
            ),
            (0.0, 0.0, 0.0)
        );
        // penetration = collision_radius + radius - distance = 0.25 + 0.5 - 0.25
        let contacts = batch.extract_contacts(0.25);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].body_index, 2);
        assert_eq!(contacts[0].penetration, 0.5);
        assert_eq!(contacts[0].distance, 0.25);
    }

    #[test]
    fn workgroup_count_rounds_up() {
        let mut batch = GpuSdfBatch::new(GpuDispatchConfig {
            workgroup_size: 4,
            ..GpuDispatchConfig::default()
        });
        for i in 0..9 {
            batch.add_query(i, Vec3Fix::ZERO, Fix128::ZERO);
        }
        assert_eq!(batch.num_workgroups(), 3);
        assert_eq!(batch_size(), crate::math::SIMD_WIDTH);
    }
}
