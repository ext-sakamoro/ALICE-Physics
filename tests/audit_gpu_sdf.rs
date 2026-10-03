//! Audit oracles for `gpu_sdf` (query / result buffers, dispatch batch,
//! multi-dispatch, CPU fallback, WGSL reference shader).
//!
//! Expectations are derived from the documented buffer layout (16-byte
//! records, field order x, y, z, radius / distance, nx, ny, nz), the
//! ceiling-division workgroup count, and the closed-form unit-sphere SDF.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::gpu_sdf::{
    batch_size, execute_batch_cpu, GpuDispatchConfig, GpuSdfBatch, GpuSdfContact,
    GpuSdfMultiDispatch, GpuSdfQuery, GpuSdfResult, SDF_EVAL_WGSL,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::SdfField;

struct UnitSphere;

impl SdfField for UnitSphere {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        (x * x + y * y + z * z).sqrt() - 1.0
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let l = (x * x + y * y + z * z).sqrt();
        (x / l, y / l, z / l)
    }
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

#[test]
fn dispatch_config_defaults_are_64_65536_and_normals_on() {
    let c = GpuDispatchConfig::default();
    assert_eq!(c.workgroup_size, 64);
    assert_eq!(c.max_queries, 65536);
    assert!(c.compute_normals);
}

#[test]
fn record_layout_is_four_f32_in_documented_order() {
    assert_eq!(core::mem::size_of::<GpuSdfQuery>(), 16);
    assert_eq!(core::mem::align_of::<GpuSdfQuery>(), 16);
    assert_eq!(core::mem::size_of::<GpuSdfResult>(), 16);
    assert_eq!(core::mem::align_of::<GpuSdfResult>(), 16);
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    b.add_query(0, v3(1.5, -2.25, 3.125), fx(0.5));
    let bytes = b.query_bytes();
    assert_eq!(bytes.len(), 16);
    let f = |i: usize| {
        f32::from_ne_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ])
    };
    assert_eq!((f(0), f(1), f(2), f(3)), (1.5, -2.25, 3.125, 0.5));
}

#[test]
fn add_query_converts_non_dyadic_positions_to_the_nearest_f32() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    b.add_query(7, v3(0.1, -0.7, 123.456), fx(0.3));
    let q = b.queries[0];
    assert_eq!(q.x, 0.1_f64 as f32);
    assert_eq!(q.y, -0.7_f64 as f32);
    assert_eq!(q.z, 123.456_f64 as f32);
    assert_eq!(q.radius, 0.3_f64 as f32);
    assert_eq!(b.body_indices, vec![7]);
}

#[test]
fn body_indices_stay_parallel_to_queries_in_insertion_order() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    for (k, idx) in [9usize, 2, 2, 5].iter().enumerate() {
        b.add_query(*idx, v3(k as f64, 0.0, 0.0), Fix128::ZERO);
    }
    assert_eq!(b.body_indices, vec![9, 2, 2, 5]);
    assert_eq!(
        b.queries.iter().map(|q| q.x).collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(b.query_count(), 4);
}

#[test]
fn num_workgroups_is_ceiling_division_including_empty_and_exact_multiples() {
    for (n, wg, want) in [
        (0u32, 64u32, 0u32),
        (1, 64, 1),
        (64, 64, 1),
        (65, 64, 2),
        (128, 64, 2),
        (129, 256, 1),
        (257, 256, 2),
        (5, 1, 5),
    ] {
        let mut b = GpuSdfBatch::new(GpuDispatchConfig {
            workgroup_size: wg,
            ..GpuDispatchConfig::default()
        });
        for i in 0..n {
            b.add_query(i as usize, Vec3Fix::ZERO, Fix128::ZERO);
        }
        assert_eq!(b.num_workgroups(), want, "n={n} wg={wg}");
    }
}

#[test]
fn prepare_output_keeps_existing_results_and_zero_fills_new_slots() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    b.add_query(0, Vec3Fix::ZERO, Fix128::ZERO);
    b.prepare_output();
    b.results[0] = GpuSdfResult {
        distance: 3.0,
        normal_x: 1.0,
        normal_y: 0.0,
        normal_z: 0.0,
    };
    b.add_query(1, Vec3Fix::ZERO, Fix128::ZERO);
    b.prepare_output();
    assert_eq!(b.results.len(), 2);
    assert_eq!(b.results[0].distance, 3.0);
    assert_eq!(b.results[1].distance, 0.0);
    assert_eq!(b.result_bytes_mut().len(), 32);
}

#[test]
fn clear_empties_all_three_buffers_and_keeps_the_config() {
    let cfg = GpuDispatchConfig {
        workgroup_size: 256,
        max_queries: 10,
        compute_normals: false,
    };
    let mut b = GpuSdfBatch::new(cfg);
    b.add_query(1, Vec3Fix::ZERO, Fix128::ZERO);
    b.prepare_output();
    b.clear();
    assert_eq!(
        (b.queries.len(), b.results.len(), b.body_indices.len()),
        (0, 0, 0)
    );
    assert_eq!(b.query_bytes().len(), 0);
    assert_eq!(b.config, cfg);
    assert_eq!(b.num_workgroups(), 0);
}

#[test]
fn extract_contacts_reports_strictly_penetrating_results_in_query_order() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    for i in 0..4 {
        b.add_query(10 + i, Vec3Fix::ZERO, Fix128::ZERO);
    }
    b.prepare_output();
    let dists = [0.25_f32, 0.5, 0.75, -0.5];
    for (i, d) in dists.iter().enumerate() {
        b.results[i] = GpuSdfResult {
            distance: *d,
            normal_x: i as f32,
            normal_y: 1.0,
            normal_z: -1.0,
        };
    }
    let c = b.extract_contacts(0.5);
    // penetration = 0.5 - d : positive for d = 0.25 and d = -0.5 only (d = 0.5 is the boundary)
    assert_eq!(
        c,
        vec![
            GpuSdfContact {
                body_index: 10,
                penetration: 0.25,
                normal: (0.0, 1.0, -1.0),
                distance: 0.25
            },
            GpuSdfContact {
                body_index: 13,
                penetration: 1.0,
                normal: (3.0, 1.0, -1.0),
                distance: -0.5
            },
        ]
    );
}

#[test]
fn execute_batch_cpu_overwrites_stale_results_and_preserves_query_order() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    b.add_query(0, v3(3.0, 0.0, 0.0), Fix128::ZERO);
    b.add_query(1, v3(0.0, -2.0, 0.0), Fix128::ZERO);
    b.prepare_output();
    b.results[0].distance = 99.0;
    execute_batch_cpu(&mut b, &UnitSphere);
    assert_eq!(b.results[0].distance, 2.0);
    assert_eq!(
        (
            b.results[0].normal_x,
            b.results[0].normal_y,
            b.results[0].normal_z
        ),
        (1.0, 0.0, 0.0)
    );
    assert_eq!(b.results[1].distance, 1.0);
    assert_eq!(
        (
            b.results[1].normal_x,
            b.results[1].normal_y,
            b.results[1].normal_z
        ),
        (0.0, -1.0, 0.0)
    );
}

#[test]
fn multi_dispatch_keeps_insertion_order_and_mirrors_query_count() {
    let q = |n: usize| {
        vec![
            GpuSdfQuery {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                radius: 0.0
            };
            n
        ]
    };
    let mut d = GpuSdfMultiDispatch::default();
    d.add_batch(5, q(3));
    d.add_batch(2, q(0));
    d.add_batch(9, q(4));
    assert_eq!(
        d.batches.iter().map(|b| b.sdf_id).collect::<Vec<_>>(),
        vec![5, 2, 9]
    );
    assert_eq!(
        d.batches.iter().map(|b| b.query_count).collect::<Vec<_>>(),
        vec![3, 0, 4]
    );
    assert_eq!(d.total_queries(), 7);
    assert_eq!(d.total_dispatches(), 3);
}

#[test]
fn batch_size_is_the_simd_width_and_positive_power_of_two() {
    assert_eq!(batch_size(), alice_physics::math::SIMD_WIDTH);
    assert!(batch_size() >= 1 && batch_size().is_power_of_two());
}

#[test]
fn wgsl_struct_field_order_matches_the_rust_records_and_default_workgroup() {
    let pos = |needle: &str| {
        SDF_EVAL_WGSL
            .find(needle)
            .unwrap_or_else(|| panic!("missing {needle}"))
    };
    assert!(
        pos("x: f32") < pos("y: f32")
            && pos("y: f32") < pos("z: f32")
            && pos("z: f32") < pos("radius: f32")
    );
    assert!(
        pos("distance: f32") < pos("normal_x: f32")
            && pos("normal_x: f32") < pos("normal_y: f32")
            && pos("normal_y: f32") < pos("normal_z: f32")
    );
    let want = format!(
        "@workgroup_size({})",
        GpuDispatchConfig::default().workgroup_size
    );
    assert!(
        SDF_EVAL_WGSL.contains(&want),
        "shader must declare the default workgroup size"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-009: GpuDispatchConfig::max_queries is documented as \"Maximum queries per dispatch\" but is never read; a batch with max_queries = 2 accepts 5 queries and num_workgroups() counts all of them (5 workgroups at workgroup_size 1) with no split or rejection"]
fn max_queries_bounds_what_one_dispatch_covers() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig {
        workgroup_size: 1,
        max_queries: 2,
        compute_normals: true,
    });
    for i in 0..5 {
        b.add_query(i, Vec3Fix::ZERO, Fix128::ZERO);
    }
    // a single dispatch may cover at most max_queries queries: either the
    // extra queries are refused or the dispatch size is capped
    assert!(
        b.query_count() <= 2 || b.num_workgroups() <= 2,
        "queries {} workgroups {}",
        b.query_count(),
        b.num_workgroups()
    );
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-010: GpuSdfQuery::radius (\"for sphere-SDF test\") is carried to the buffer but never used by execute_batch_cpu, extract_contacts or SDF_EVAL_WGSL; penetration uses one global collision_radius, so a radius-0.5 sphere centred 0.3 from the surface produces no contact at collision_radius 0 (expected penetration 0.2)"]
fn per_query_radius_contributes_to_penetration() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    // centre at distance 0.3 outside the unit sphere
    b.add_query(0, v3(1.3, 0.0, 0.0), fx(0.5));
    b.prepare_output();
    execute_batch_cpu(&mut b, &UnitSphere);
    let c = b.extract_contacts(0.0);
    assert_eq!(
        c.len(),
        1,
        "a sphere query of radius 0.5 at distance 0.3 overlaps by 0.2"
    );
    assert!((c[0].penetration - 0.2).abs() < 1e-5);
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-011: GpuSdfMultiDispatch docs say one dispatch per unique sdf_id, but add_batch never merges, so two batches with the same sdf_id give total_dispatches() == 2 (unique ids: 1)"]
fn total_dispatches_counts_unique_sdf_ids() {
    let q = vec![GpuSdfQuery {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        radius: 0.0,
    }];
    let mut d = GpuSdfMultiDispatch::new();
    d.add_batch(7, q.clone());
    d.add_batch(7, q);
    assert_eq!(d.total_dispatches(), 1);
}

struct LinearField;

impl SdfField for LinearField {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        x + 2.0 * y + 4.0 * z
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        (x + 10.0, y + 20.0, z + 30.0)
    }
}

#[test]
fn execute_batch_cpu_passes_coordinates_to_the_field_in_x_y_z_order() {
    let mut b = GpuSdfBatch::new(GpuDispatchConfig::default());
    b.add_query(0, v3(1.0, 2.0, 3.0), Fix128::ZERO);
    b.prepare_output();
    execute_batch_cpu(&mut b, &LinearField);
    let r = b.results[0];
    assert_eq!(r.distance, 1.0 + 4.0 + 12.0);
    assert_eq!((r.normal_x, r.normal_y, r.normal_z), (11.0, 22.0, 33.0));
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-021: num_workgroups() divides by config.workgroup_size (documented as typically 64 or 256) but SDF_EVAL_WGSL fixes @workgroup_size(64); with workgroup_size 256 and 300 queries it returns 2 groups, which cover 128 invocations of the 64-wide shader (172 queries never evaluated)"]
fn dispatch_covers_every_query_with_the_shader_workgroup_size() {
    let shader_wg: u32 = SDF_EVAL_WGSL
        .split("@workgroup_size(")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .and_then(|s| s.trim().parse().ok())
        .expect("shader declares a workgroup size");
    for wg in [64u32, 256] {
        let mut b = GpuSdfBatch::new(GpuDispatchConfig {
            workgroup_size: wg,
            ..GpuDispatchConfig::default()
        });
        for i in 0..300 {
            b.add_query(i, Vec3Fix::ZERO, Fix128::ZERO);
        }
        assert!(
            b.num_workgroups() * shader_wg >= 300,
            "workgroup_size {wg}: {} groups x {shader_wg} invocations < 300 queries",
            b.num_workgroups()
        );
    }
}
