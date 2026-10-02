//! GPU SDF batch-query CPU-fallback demo.
//!
//! Exercises the full `gpu_sdf` public surface: batch construction,
//! SIMD-aligned chunk sizing, CPU-side reference execution, raw buffer
//! marshalling (the GPU upload/readback contract), and multi-SDF
//! dispatch grouping.
//!
//! Also demonstrates the *parity contract* between [`execute_batch_cpu`]
//! (the CPU reference implementation) and the reference [`SDF_EVAL_WGSL`]
//! placeholder shader: both evaluate the exact same unit-sphere formula
//! (`length(p) - 1.0`) for the sphere scene below, so a GPU backend that
//! replaces the placeholder `sdf_distance` with ALICE-SDF code generation
//! must still agree with `execute_batch_cpu` on this scene.
//!
//! ```bash
//! cargo run --example gpu_sdf_batch_queries --features std
//! ```

use alice_physics::gpu_sdf::{
    batch_size, execute_batch_cpu, GpuDispatchConfig, GpuSdfBatch, GpuSdfContact,
    GpuSdfMultiDispatch, GpuSdfQuery, SDF_EVAL_WGSL,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::ClosureSdf;

/// Unit sphere centred at the origin — matches `SDF_EVAL_WGSL`'s placeholder
/// `sdf_distance` formula (`length(vec3(x, y, z)) - 1.0`) exactly.
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / len, y / len, z / len)
        },
    )
}

/// Axis-aligned unit box centred at the origin (Chebyshev distance), used
/// only to demonstrate [`GpuSdfMultiDispatch`] grouping multiple distinct
/// SDFs into separate GPU kernel dispatches.
fn unit_box() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| x.abs().max(y.abs()).max(z.abs()) - 1.0,
        |x, y, z| {
            let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
            if ax >= ay && ax >= az {
                (x.signum(), 0.0, 0.0)
            } else if ay >= az {
                (0.0, y.signum(), 0.0)
            } else {
                (0.0, 0.0, z.signum())
            }
        },
    )
}

fn main() {
    println!(
        "[gpu_sdf] SDF_EVAL_WGSL reference shader: {} bytes (placeholder sdf_distance = unit sphere)",
        SDF_EVAL_WGSL.len()
    );

    // batch_size() mirrors crate::math::SIMD_WIDTH — size CPU-side chunks to
    // exactly one SIMD register's worth of queries, no partial loads.
    let chunk = batch_size();
    println!("[gpu_sdf] batch_size() (SIMD_WIDTH) = {chunk}");

    // --- Single-SDF batch: sphere queries spanning > 1 workgroup --------
    let config = GpuDispatchConfig {
        workgroup_size: 64,
        ..GpuDispatchConfig::default()
    };
    let mut batch = GpuSdfBatch::new(config);

    // 80 points along +x from -1.0 to 6.9 step 0.1: crosses the unit-sphere
    // surface (|x| = 1.0) and exceeds one 64-wide workgroup.
    for i in 0..80u32 {
        let x = -1.0 + 0.1 * (i as f32);
        batch.add_query(
            i as usize,
            Vec3Fix::from_f32(x, 0.0, 0.0),
            Fix128::from_f32(0.0),
        );
    }
    println!(
        "[gpu_sdf] query_count() after 80 add_query calls = {}",
        batch.query_count()
    );
    println!(
        "[gpu_sdf] num_workgroups() @ workgroup_size=64 for 80 queries = {} (ceil(80/64))",
        batch.num_workgroups()
    );

    batch.prepare_output();
    let qb_len = batch.query_bytes().len();
    let rb_len = batch.result_bytes_mut().len();
    println!(
        "[gpu_sdf] query_bytes() = {qb_len} bytes, result_bytes_mut() = {rb_len} bytes (16 B/query, 16 B/result)"
    );

    let sphere = unit_sphere();
    execute_batch_cpu(&mut batch, &sphere);

    let contacts: Vec<GpuSdfContact> = batch.extract_contacts(0.05);
    println!(
        "[gpu_sdf] extract_contacts(0.05) found {} contact(s):",
        contacts.len()
    );
    for c in &contacts {
        println!(
            "[gpu_sdf]   body={} distance={:.4} penetration={:.4} normal=({:.3},{:.3},{:.3})",
            c.body_index, c.distance, c.penetration, c.normal.0, c.normal.1, c.normal.2
        );
    }

    // --- Multi-dispatch: sphere + box grouped into separate kernels ------
    let mut dispatch = GpuSdfMultiDispatch::new();
    let sphere_queries: Vec<GpuSdfQuery> = (0..5)
        .map(|i| GpuSdfQuery {
            x: i as f32 * 0.3,
            y: 0.0,
            z: 0.0,
            radius: 0.0,
        })
        .collect();
    let box_queries: Vec<GpuSdfQuery> = (0..3)
        .map(|i| GpuSdfQuery {
            x: 0.0,
            y: i as f32 * 0.5,
            z: 0.0,
            radius: 0.0,
        })
        .collect();
    dispatch.add_batch(0, sphere_queries);
    dispatch.add_batch(1, box_queries);

    println!(
        "[gpu_sdf] multi-dispatch: total_queries()={} total_dispatches()={} (5 sphere + 3 box, 2 kernels)",
        dispatch.total_queries(),
        dispatch.total_dispatches()
    );
    for b in &dispatch.batches {
        println!(
            "[gpu_sdf]   sdf_id={} query_count field={}",
            b.sdf_id, b.query_count
        );
    }

    // Evaluate the box SDF through the CPU reference too, proving `unit_box`
    // is a real second SDF (not just a label passed to add_batch).
    let mut box_batch = GpuSdfBatch::new(GpuDispatchConfig::default());
    box_batch.add_query(100, Vec3Fix::from_f32(0.5, 0.5, 0.5), Fix128::from_f32(0.0));
    execute_batch_cpu(&mut box_batch, &unit_box());
    println!(
        "[gpu_sdf] unit_box distance at (0.5,0.5,0.5) = {:.4}",
        box_batch.results[0].distance
    );
}
