//! Oracles for the `gpu_sdf` items reached through the production entry
//! point `examples/gpu_sdf_batch_queries.rs`: `GpuSdfBatch::{add_query,
//! query_count, num_workgroups, prepare_output, query_bytes,
//! result_bytes_mut, extract_contacts}`, the free function
//! `execute_batch_cpu` (the CPU reference implementation), `GpuSdfContact`,
//! `batch_size`, `SDF_EVAL_WGSL` (the reference WGSL placeholder shader) and
//! `GpuSdfMultiDispatch::{add_batch, total_queries, total_dispatches}`
//! (including the `GpuSdfInstancedBatch::query_count` field it populates).
//!
//! # Closed forms
//!
//! * **`query_count()`**: equals the number of `add_query` calls, by
//!   construction (`self.queries.len()`), checked for several distinct
//!   counts so a mutation that returns a constant cannot survive.
//! * **`num_workgroups()`**: `ceil(n / workgroup_size)`, computed
//!   independently in this file as `(n + workgroup_size - 1) /
//!   workgroup_size` (integer division) and checked against several
//!   `(n, workgroup_size)` pairs, including a batch spanning more than one
//!   workgroup and a non-default `workgroup_size`.
//! * **Buffer marshalling**: `query_bytes().len() == queries.len() * 16` and
//!   `result_bytes_mut().len() == results.len() * 16` (both structs are
//!   `repr(C, align(16))` with four `f32` fields, no padding). A raw byte
//!   write into `result_bytes_mut()` is read back exactly through
//!   `batch.results`, proving the slice aliases the same backing memory
//!   rather than a copy.
//! * **`execute_batch_cpu`**: for the unit sphere `d(p) = |p| - 1`, computed
//!   independently in this file (not by calling the crate's `ClosureSdf` or
//!   `execute_batch_cpu` itself to produce the expected value).
//! * **`SDF_EVAL_WGSL` parity**: the placeholder `sdf_distance` in the WGSL
//!   source is the literal text `length(vec3(x, y, z)) - 1.0`, i.e. the same
//!   unit-sphere formula used above — checked both textually (guards
//!   against silent drift of the placeholder formula) and numerically
//!   (the CPU reference agrees with the closed form on the same scene).
//! * **`total_queries()` / `total_dispatches()`**: `Σ batches[i].len()` and
//!   `batches.len()`, checked with distinct per-batch sizes so a mutation
//!   that only looks at the first batch cannot survive, and the
//!   `GpuSdfInstancedBatch::query_count` field is checked against the same
//!   per-batch sizes (it mirrors `queries.len()` per the doc comment).
//! * **`batch_size()`**: equals `alice_physics::math::SIMD_WIDTH` by
//!   definition; under the default feature set (no `simd` feature) that is
//!   the documented scalar fallback, `1`.
//!
//! # Degenerate input
//!
//! `num_workgroups()` with `workgroup_size = 0` panics (integer division by
//! zero) for both an empty and a non-empty batch — this is the current,
//! undocumented-as-guarded behavior of `u32::div_ceil`, pinned here rather
//! than silently assumed. `extract_contacts()` on a batch that was never
//! `prepare_output`-ed / `execute_batch_cpu`-ed returns an empty `Vec`
//! (iterates the empty `results`) even though `body_indices` is non-empty —
//! pinned as the current behavior, not a panic. A query at extreme `f32`
//! coordinates (`f32::MAX`) round-trips through `Fix128`'s saturating
//! `f64 → i64` cast in `Vec3Fix::from_f32`; the stored position is silently
//! clamped to roughly `i64::MAX` (~9.22e18, far below `f32::MAX` ~3.4e38)
//! rather than overflowing to infinity, and the resulting distance is a
//! large *finite* number — no panic either way. A query exactly
//! on the sphere surface (`distance == 0.0`, bit-exact) is checked at the
//! `extract_contacts` boundary: `penetration = collision_radius - distance`
//! must be strictly `> 0.0`, so a `collision_radius` of `0.0` excludes it.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::gpu_sdf::{
    batch_size, execute_batch_cpu, GpuDispatchConfig, GpuSdfBatch, GpuSdfContact,
    GpuSdfMultiDispatch, GpuSdfQuery, GpuSdfResult, SDF_EVAL_WGSL,
};
use alice_physics::math::{Fix128, Vec3Fix, SIMD_WIDTH};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Closed-form unit-sphere SDF distance, independent of the crate's
/// `ClosureSdf` / `execute_batch_cpu`: `d(p) = |p| - 1`.
fn sphere_distance(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt() - 1.0
}

/// Closed-form unit-sphere SDF normal (outward), matching the fallback
/// convention used by the crate's own `gpu_sdf` module tests: `(0, 1, 0)`
/// at the origin (`len < 1e-10`), else the normalized position.
fn sphere_normal(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len = (x * x + y * y + z * z).sqrt();
    if len < 1e-10 {
        (0.0, 1.0, 0.0)
    } else {
        (x / len, y / len, z / len)
    }
}

/// A minimal `SdfField` wrapping [`sphere_distance`] / [`sphere_normal`],
/// built without going through `alice_physics::sdf_collider::ClosureSdf` so
/// that `execute_batch_cpu`'s dynamic dispatch is exercised against a
/// locally-owned, independently-written reference.
struct UnitSphere;

impl alice_physics::sdf_collider::SdfField for UnitSphere {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        sphere_distance(x, y, z)
    }

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        sphere_normal(x, y, z)
    }
}

fn batch_with_queries(workgroup_size: u32, positions: &[(f32, f32, f32)]) -> GpuSdfBatch {
    let config = GpuDispatchConfig {
        workgroup_size,
        ..GpuDispatchConfig::default()
    };
    let mut batch = GpuSdfBatch::new(config);
    for (i, &(x, y, z)) in positions.iter().enumerate() {
        batch.add_query(i, Vec3Fix::from_f32(x, y, z), Fix128::from_f32(0.0));
    }
    batch
}

// ============================================================================
// add_query / query_count
// ============================================================================

#[test]
fn test_query_count_equals_add_query_call_count() {
    for &k in &[0usize, 1, 3, 80] {
        let positions: Vec<(f32, f32, f32)> = (0..k).map(|i| (i as f32, 0.0, 0.0)).collect();
        let batch = batch_with_queries(64, &positions);
        assert_eq!(
            batch.query_count(),
            k,
            "query_count() must equal add_query call count {k}"
        );
    }
}

#[test]
fn test_add_query_round_trips_exact_fields_through_query_bytes() {
    // Dyadic rationals round-trip exactly through Fix128 (binary fixed point)
    // and back through f32, so this is a bit-exact oracle, not a tolerance.
    let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
    batch.add_query(
        7,
        Vec3Fix::from_f32(1.5, -2.25, 3.0),
        Fix128::from_f32(0.75),
    );

    assert_eq!(batch.queries[0].x, 1.5);
    assert_eq!(batch.queries[0].y, -2.25);
    assert_eq!(batch.queries[0].z, 3.0);
    assert_eq!(batch.queries[0].radius, 0.75);

    let bytes = batch.query_bytes();
    assert_eq!(bytes.len(), 16);
    let decode = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    assert_eq!(decode(0), 1.5, "query_bytes()[0..4] must be x");
    assert_eq!(decode(4), -2.25, "query_bytes()[4..8] must be y");
    assert_eq!(decode(8), 3.0, "query_bytes()[8..12] must be z");
    assert_eq!(decode(12), 0.75, "query_bytes()[12..16] must be radius");
}

// ============================================================================
// num_workgroups — closed-form ceiling division
// ============================================================================

fn expected_workgroups(n: u32, workgroup_size: u32) -> u32 {
    // `u32::div_ceil` is the stdlib's ceiling-division primitive, not the
    // crate under test — this remains an independent closed form, not a
    // call into `GpuSdfBatch::num_workgroups`.
    n.div_ceil(workgroup_size)
}

#[test]
fn test_num_workgroups_ceiling_closed_form() {
    let cases: &[(u32, u32)] = &[
        (0, 64),
        (1, 64),
        (64, 64),
        (65, 64),
        (80, 64),
        (80, 32),
        (127, 64),
        (128, 64),
    ];
    for &(n, ws) in cases {
        let positions: Vec<(f32, f32, f32)> = (0..n).map(|i| (i as f32, 0.0, 0.0)).collect();
        let batch = batch_with_queries(ws, &positions);
        assert_eq!(
            batch.num_workgroups(),
            expected_workgroups(n, ws),
            "num_workgroups() mismatch for n={n} workgroup_size={ws}"
        );
    }
}

#[test]
fn test_num_workgroups_zero_workgroup_size_panics() {
    // Non-empty batch, workgroup_size = 0: `u32::div_ceil` panics on
    // division by zero. This is the current behavior — pinned, not fixed.
    let batch = batch_with_queries(0, &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]);
    let result = catch_unwind(AssertUnwindSafe(|| batch.num_workgroups()));
    assert!(
        result.is_err(),
        "num_workgroups() must panic on workgroup_size = 0 (div by zero), got {result:?}"
    );

    // Empty batch, workgroup_size = 0: `0.div_ceil(0)` also panics (the
    // panic is on the divisor being zero, independent of the dividend).
    let empty = batch_with_queries(0, &[]);
    let result_empty = catch_unwind(AssertUnwindSafe(|| empty.num_workgroups()));
    assert!(
        result_empty.is_err(),
        "num_workgroups() on an empty batch must also panic on workgroup_size = 0"
    );
}

// ============================================================================
// prepare_output — resize to query_count (grow and shrink)
// ============================================================================

#[test]
fn test_prepare_output_resizes_results_to_query_count() {
    let mut batch = batch_with_queries(64, &[(0.0, 0.0, 0.0); 5]);
    batch.prepare_output();
    assert_eq!(
        batch.results.len(),
        5,
        "prepare_output() must grow results to query_count()"
    );
    for r in &batch.results {
        assert_eq!(r.distance, 0.0);
        assert_eq!(r.normal_x, 0.0);
        assert_eq!(r.normal_y, 0.0);
        assert_eq!(r.normal_z, 0.0);
    }

    // Manually oversize `results` beyond `queries.len()`, then confirm
    // prepare_output() shrinks it back down (Vec::resize truncates too).
    batch.results.push(GpuSdfResult {
        distance: 9.0,
        normal_x: 0.0,
        normal_y: 0.0,
        normal_z: 0.0,
    });
    batch.results.push(GpuSdfResult {
        distance: 9.0,
        normal_x: 0.0,
        normal_y: 0.0,
        normal_z: 0.0,
    });
    assert_eq!(batch.results.len(), 7);
    // `result_bytes_mut()` must track `results.len()` (7), not
    // `queries.len()` (5) — the two differ right now, which is exactly the
    // scene that catches a mutant that sources the byte length from the
    // wrong field.
    assert_eq!(
        batch.result_bytes_mut().len(),
        7 * 16,
        "result_bytes_mut() must size from results.len(), not queries.len()"
    );
    batch.prepare_output();
    assert_eq!(
        batch.results.len(),
        5,
        "prepare_output() must shrink results back to query_count()"
    );
}

// ============================================================================
// query_bytes / result_bytes_mut — length closed form + write/read round-trip
// ============================================================================

#[test]
fn test_buffer_byte_lengths_closed_form() {
    for &n in &[0usize, 1, 5] {
        let mut batch = batch_with_queries(64, &vec![(0.0, 0.0, 0.0); n]);
        assert_eq!(
            batch.query_bytes().len(),
            n * 16,
            "query_bytes() length for n={n}"
        );
        batch.prepare_output();
        assert_eq!(
            batch.result_bytes_mut().len(),
            n * 16,
            "result_bytes_mut() length for n={n}"
        );
    }
}

#[test]
fn test_result_bytes_mut_write_then_read_round_trip() {
    let mut batch = batch_with_queries(64, &[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]);
    batch.prepare_output();

    let written: [(f32, f32, f32, f32); 2] = [(1.25, 0.0, 1.0, 0.0), (-3.5, 1.0, 0.0, 0.0)];
    {
        let bytes = batch.result_bytes_mut();
        for (i, &(d, nx, ny, nz)) in written.iter().enumerate() {
            let off = i * 16;
            bytes[off..off + 4].copy_from_slice(&d.to_le_bytes());
            bytes[off + 4..off + 8].copy_from_slice(&nx.to_le_bytes());
            bytes[off + 8..off + 12].copy_from_slice(&ny.to_le_bytes());
            bytes[off + 12..off + 16].copy_from_slice(&nz.to_le_bytes());
        }
    }

    for (i, &(d, nx, ny, nz)) in written.iter().enumerate() {
        assert_eq!(
            batch.results[i].distance, d,
            "result[{i}].distance after raw write"
        );
        assert_eq!(
            batch.results[i].normal_x, nx,
            "result[{i}].normal_x after raw write"
        );
        assert_eq!(
            batch.results[i].normal_y, ny,
            "result[{i}].normal_y after raw write"
        );
        assert_eq!(
            batch.results[i].normal_z, nz,
            "result[{i}].normal_z after raw write"
        );
    }
}

// ============================================================================
// execute_batch_cpu — unit sphere closed form (bit-exact points)
// ============================================================================

#[test]
fn test_execute_batch_cpu_matches_unit_sphere_closed_form() {
    let points: &[(f32, f32, f32)] = &[
        (2.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ];
    let mut batch = batch_with_queries(64, points);
    execute_batch_cpu(&mut batch, &UnitSphere);

    for (i, &(x, y, z)) in points.iter().enumerate() {
        let (exp_dist, (exp_nx, exp_ny, exp_nz)) =
            (sphere_distance(x, y, z), sphere_normal(x, y, z));
        assert_eq!(
            batch.results[i].distance, exp_dist,
            "distance mismatch at point {i}"
        );
        assert_eq!(
            batch.results[i].normal_x, exp_nx,
            "normal_x mismatch at point {i}"
        );
        assert_eq!(
            batch.results[i].normal_y, exp_ny,
            "normal_y mismatch at point {i}"
        );
        assert_eq!(
            batch.results[i].normal_z, exp_nz,
            "normal_z mismatch at point {i}"
        );
    }
    // Point exactly on the surface is an exact zero, not an approximation.
    assert_eq!(
        batch.results[2].distance, 0.0,
        "(1,0,0) must be exactly on the unit sphere"
    );
}

#[test]
fn test_execute_batch_cpu_compute_normals_false_zeroes_normal() {
    // Non-default config field: compute_normals = false must suppress the
    // normal computation (distance is still computed) — this is the
    // configuration/wiring check: a mutation that ignores the flag and
    // always computes normals would survive without this scene.
    let config = GpuDispatchConfig {
        compute_normals: false,
        ..GpuDispatchConfig::default()
    };
    let mut batch = GpuSdfBatch::new(config);
    batch.add_query(0, Vec3Fix::from_f32(2.0, 0.0, 0.0), Fix128::from_f32(0.0));
    execute_batch_cpu(&mut batch, &UnitSphere);

    assert_eq!(
        batch.results[0].distance, 1.0,
        "distance must still be computed"
    );
    assert_eq!(
        batch.results[0].normal_x, 0.0,
        "normal must be suppressed when compute_normals = false"
    );
    assert_eq!(batch.results[0].normal_y, 0.0);
    assert_eq!(batch.results[0].normal_z, 0.0);
}

// ============================================================================
// extract_contacts — GpuSdfContact, strict `> 0.0` boundary
// ============================================================================

#[test]
fn test_extract_contacts_strict_inequality_boundary() {
    // Query exactly on the surface (distance == 0.0, bit-exact) with
    // collision_radius = 0.0: penetration = 0.0 - 0.0 = 0.0, which is NOT
    // > 0.0, so it must be excluded.
    let mut on_surface = batch_with_queries(64, &[(1.0, 0.0, 0.0)]);
    execute_batch_cpu(&mut on_surface, &UnitSphere);
    let contacts: Vec<GpuSdfContact> = on_surface.extract_contacts(0.0);
    assert!(
        contacts.is_empty(),
        "a point exactly on the surface at collision_radius=0.0 must be excluded"
    );

    // Query strictly inside: penetration = 0.0 - (-0.5) = 0.5 > 0.0, included.
    let mut inside = batch_with_queries(64, &[(0.5, 0.0, 0.0)]);
    execute_batch_cpu(&mut inside, &UnitSphere);
    let contacts: Vec<GpuSdfContact> = inside.extract_contacts(0.0);
    assert_eq!(
        contacts.len(),
        1,
        "a point strictly inside the sphere must produce exactly one contact"
    );
    assert_eq!(contacts[0].body_index, 0);
    assert_eq!(contacts[0].distance, -0.5);
    assert_eq!(
        contacts[0].penetration, 0.5,
        "penetration = collision_radius - distance"
    );
}

#[test]
fn test_extract_contacts_preserves_distinct_body_indices() {
    // `add_query`'s `body_idx` argument is deliberately not the query's
    // position in the batch (10, 20, 30 instead of 0, 1, 2), so a mutation
    // that drops or constant-folds `body_idx` (e.g. always pushing 0) is
    // caught: all three bodies are strictly inside, so all three must
    // produce a contact carrying their own original index.
    let config = GpuDispatchConfig::default();
    let mut batch = GpuSdfBatch::new(config);
    batch.add_query(10, Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::from_f32(0.0));
    batch.add_query(20, Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_f32(0.0));
    batch.add_query(30, Vec3Fix::from_f32(0.0, 0.0, 0.5), Fix128::from_f32(0.0));
    execute_batch_cpu(&mut batch, &UnitSphere);

    let contacts: Vec<GpuSdfContact> = batch.extract_contacts(0.0);
    assert_eq!(
        contacts.len(),
        3,
        "all three queries are strictly inside the sphere"
    );
    let mut body_indices: Vec<usize> = contacts.iter().map(|c| c.body_index).collect();
    body_indices.sort_unstable();
    assert_eq!(
        body_indices,
        vec![10, 20, 30],
        "extract_contacts() must preserve each query's original body_idx"
    );
}

#[test]
fn test_extract_contacts_without_results_is_empty_not_panic() {
    // Degenerate: add_query called, but neither prepare_output() nor
    // execute_batch_cpu() — `results` stays empty while `body_indices` has
    // 3 entries. extract_contacts() iterates `results`, so it returns an
    // empty Vec silently (pinned as current behavior, not a panic).
    let batch = batch_with_queries(64, &[(0.0, 0.0, 0.0); 3]);
    assert_eq!(batch.results.len(), 0);
    assert_eq!(batch.body_indices.len(), 3);
    let contacts: Vec<GpuSdfContact> = batch.extract_contacts(10.0);
    assert!(
        contacts.is_empty(),
        "extract_contacts() without prepare_output()/execute_batch_cpu() must be empty"
    );
}

// ============================================================================
// GpuSdfMultiDispatch — total_queries / total_dispatches / query_count field
// ============================================================================

#[test]
fn test_multi_dispatch_totals_closed_form() {
    let mut dispatch = GpuSdfMultiDispatch::new();
    assert_eq!(
        dispatch.total_queries(),
        0,
        "empty dispatch has 0 total queries"
    );
    assert_eq!(
        dispatch.total_dispatches(),
        0,
        "empty dispatch has 0 total dispatches"
    );

    let sizes = [2usize, 5, 80];
    for (sdf_id, &n) in sizes.iter().enumerate() {
        let queries: Vec<GpuSdfQuery> = (0..n)
            .map(|i| GpuSdfQuery {
                x: i as f32,
                y: 0.0,
                z: 0.0,
                radius: 0.0,
            })
            .collect();
        dispatch.add_batch(sdf_id as u32, queries);
    }

    assert_eq!(
        dispatch.total_queries(),
        sizes.iter().sum::<usize>(),
        "total_queries() must sum all batch lengths"
    );
    assert_eq!(
        dispatch.total_dispatches(),
        sizes.len(),
        "total_dispatches() must equal the number of add_batch calls"
    );

    for (i, &n) in sizes.iter().enumerate() {
        assert_eq!(
            dispatch.batches[i].query_count, n as u32,
            "GpuSdfInstancedBatch::query_count must mirror queries.len() for batch {i}"
        );
    }
}

// ============================================================================
// batch_size — SIMD_WIDTH contract
// ============================================================================

#[test]
fn test_batch_size_equals_simd_width() {
    assert_eq!(
        batch_size(),
        SIMD_WIDTH,
        "batch_size() must equal crate::math::SIMD_WIDTH by definition"
    );
    // Default feature set has no `simd` feature: scalar fallback is 1.
    assert_eq!(
        batch_size(),
        1,
        "default (no `simd` feature) build must use the scalar fallback width"
    );
}

// ============================================================================
// SDF_EVAL_WGSL — textual + numeric parity with the unit-sphere closed form
// ============================================================================

#[test]
fn test_sdf_eval_wgsl_placeholder_textual_parity() {
    assert!(
        SDF_EVAL_WGSL.contains("length(vec3(x, y, z)) - 1.0"),
        "SDF_EVAL_WGSL's placeholder sdf_distance formula must stay the unit-sphere formula \
         documented by the gpu_sdf module (drift here silently breaks the parity claim)"
    );
    for marker in [
        "struct Query",
        "struct Result",
        "fn sdf_distance",
        "fn sdf_normal",
        "fn main(",
    ] {
        assert!(
            SDF_EVAL_WGSL.contains(marker),
            "SDF_EVAL_WGSL must still contain `{marker}`"
        );
    }
}

#[test]
fn test_sdf_eval_wgsl_placeholder_numeric_parity_with_cpu_reference() {
    // The WGSL placeholder's `sdf_distance` is textually the same formula as
    // `sphere_distance` above; demonstrate that `execute_batch_cpu` (the
    // documented reference a GPU backend must match) agrees with that exact
    // formula numerically, closing the loop between the textual check above
    // and the CPU-side behavior.
    let points: &[(f32, f32, f32)] = &[(3.0, 4.0, 0.0), (0.0, 0.0, 5.0)];
    let mut batch = batch_with_queries(64, points);
    execute_batch_cpu(&mut batch, &UnitSphere);
    for (i, &(x, y, z)) in points.iter().enumerate() {
        assert_eq!(
            batch.results[i].distance,
            sphere_distance(x, y, z),
            "point {i}"
        );
    }
}

// ============================================================================
// Degenerate: extreme coordinates
// ============================================================================

#[test]
fn test_extreme_coordinates_do_not_panic() {
    // Real finding (not a crate bug, documented here): `add_query` stores
    // positions as `Vec3Fix` (`Fix128`), whose integer half is `i64`. The
    // `f32 -> f64 -> Fix128 -> f64 -> f32` round trip in
    // `Vec3Fix::from_f32`/`to_f32` therefore saturates at `i64::MAX`
    // (~9.22e18) through a saturating `as i64` cast — far below
    // `f32::MAX` (~3.4e38). So a query at `(f32::MAX, f32::MAX, f32::MAX)`
    // does NOT reach the GPU query buffer as `f32::MAX`; it arrives already
    // clamped to roughly `i64::MAX`, and the resulting distance is a large
    // *finite* number, not `f32::INFINITY`.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut batch = batch_with_queries(64, &[(f32::MAX, f32::MAX, f32::MAX)]);
        execute_batch_cpu(&mut batch, &UnitSphere);
        (batch.queries[0].x, batch.results[0].distance)
    }));
    assert!(
        result.is_ok(),
        "extreme f32::MAX coordinates must not panic, got {result:?}"
    );
    let (stored_x, distance) = result.unwrap();
    assert!(
        stored_x < f32::MAX,
        "the Fix128 round trip must saturate f32::MAX down towards i64::MAX, got stored x = {stored_x}"
    );
    assert!(
        distance.is_finite() && distance > 1.0e18,
        "distance at the saturated (MAX, MAX, MAX) query must be a large finite number, got {distance}"
    );
}
