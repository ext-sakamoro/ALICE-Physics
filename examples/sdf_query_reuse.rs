//! SDF Query Reuse Example
//!
//! Production entry point for the reuse and emptiness paths of three SDF
//! query helpers: `GpuSdfBatch::clear` (`src/gpu_sdf.rs`),
//! `AdaptiveSdfEvaluator::resize` (`src/sdf_adaptive.rs`) and
//! `SdfManifold::is_empty` (`src/sdf_manifold.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - the unit sphere's distance at `(x, 0, 0)` is `|x| − 1`; a batch that is
//!   cleared and refilled reports exactly the distances of the new queries,
//!   and a cleared batch reports no contacts
//! - the adaptive evaluator skips a far body (distance above the cache
//!   threshold) on the frame after it was cached, but only bodies inside its
//!   cache are cached: with a cache of one, two far bodies save one
//!   evaluation per frame; after `resize(2)` the second body is cached on its
//!   next evaluation and both are saved from then on; `resize(0)` saves none
//! - a sphere of radius `r` centred at height `y` over the plane `y = 0`
//!   touches it exactly when `y < r`; the manifold is empty otherwise, and
//!   when it is not, every sample sees the same depth `r − y` along `+y`,
//!   and capping the manifold at 1-3 contacts keeps that many
//!
//! Run with: `cargo run --example sdf_query_reuse`

use alice_physics::gpu_sdf::{execute_batch_cpu, GpuDispatchConfig, GpuSdfBatch};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sdf_manifold::{generate_sdf_manifold, ManifoldConfig};

fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / len, y / len, z / len)
        },
    )
}

fn ground() -> ClosureSdf {
    ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))
}

fn gpu_batch_clear() {
    let sphere = unit_sphere();
    let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
    for (i, x) in [0.5_f32, 3.0, -2.0].into_iter().enumerate() {
        batch.add_query(i, Vec3Fix::from_f32(x, 0.0, 0.0), Fix128::ZERO);
    }
    execute_batch_cpu(&mut batch, &sphere);
    assert_eq!(
        batch.extract_contacts(0.0).len(),
        1,
        "only x = 0.5 is inside"
    );

    batch.clear();
    assert_eq!(batch.query_count(), 0, "clear drops the queries");
    assert!(batch.results.is_empty(), "clear drops the results");
    assert!(
        batch.body_indices.is_empty(),
        "clear drops the body indices"
    );
    assert!(
        batch.extract_contacts(10.0).is_empty(),
        "a cleared batch has no contacts, however large the radius"
    );

    // Refill with different bodies: the results are the new queries' alone.
    let xs = [4.0_f32, -1.5];
    for (k, x) in xs.into_iter().enumerate() {
        batch.add_query(10 + k, Vec3Fix::from_f32(x, 0.0, 0.0), Fix128::ZERO);
    }
    execute_batch_cpu(&mut batch, &sphere);
    assert_eq!(batch.query_count(), 2, "two queries after the refill");
    for (k, x) in xs.into_iter().enumerate() {
        let d = batch.results[k].distance;
        assert!(
            (d - (x.abs() - 1.0)).abs() < 1e-6,
            "|x| - 1 at x = {x}: got {d}"
        );
        assert_eq!(batch.body_indices[k], 10 + k, "body index of query {k}");
    }
    let contacts = batch.extract_contacts(1.0);
    assert_eq!(contacts.len(), 1, "within 1 of the sphere: only x = -1.5");
    assert_eq!(contacts[0].body_index, 11);
    println!(
        "GpuSdfBatch: cleared, refilled, d = ({}, {})",
        batch.results[0].distance, batch.results[1].distance
    );
}

fn adaptive_resize() {
    let sphere = SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);
    // Both bodies are 15 m out: distance 14, above the cache threshold 10.
    let pos = [Vec3Fix::from_int(15, 0, 0), Vec3Fix::from_int(0, 0, 15)];
    let mut ev = AdaptiveSdfEvaluator::new(1, AdaptiveConfig::default());
    let frame = |ev: &mut AdaptiveSdfEvaluator| {
        ev.begin_frame();
        for (i, p) in pos.iter().enumerate() {
            let (d, _) = ev.evaluate(i, *p, &sphere);
            assert!((d - 14.0).abs() < 1e-4, "body {i}: |p| - 1 = 14, got {d}");
        }
        ev.stats()
    };

    assert_eq!(frame(&mut ev), (0, 2), "frame 1: nothing cached yet");
    assert_eq!(frame(&mut ev), (1, 2), "cache of one: only body 0 is saved");
    ev.resize(2);
    assert_eq!(frame(&mut ev), (1, 2), "body 1 is cached this frame");
    assert_eq!(frame(&mut ev), (2, 2), "and saved from the next one");
    ev.resize(0);
    assert_eq!(frame(&mut ev), (0, 2), "an empty cache saves nothing");
    println!(
        "AdaptiveSdfEvaluator: saved 1/2 with one slot, 2/2 after resize(2), 0/2 after resize(0)"
    );
}

fn manifold_emptiness() {
    let plane = SdfCollider::new_static(Box::new(ground()), Vec3Fix::ZERO, QuatFix::IDENTITY);
    let config = ManifoldConfig::default();
    let r = Fix128::ONE;

    for (y, touching) in [(2.0_f64, false), (1.0, false), (0.5, true), (0.25, true)] {
        let m = generate_sdf_manifold(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(y), Fix128::ZERO),
            r,
            &plane,
            &config,
        );
        assert_eq!(!m.is_empty(), touching, "centre at y = {y}, radius 1");
        assert_eq!(
            m.is_empty(),
            m.contacts.is_empty(),
            "is_empty agrees with the contacts"
        );
        if touching {
            let depth = 1.0 - y;
            assert!(
                (m.avg_depth.to_f64() - depth).abs() < 1e-6,
                "y = {y}: depth r - y = {depth}, got {}",
                m.avg_depth.to_f64()
            );
            assert!((m.normal.y.to_f64() - 1.0).abs() < 1e-6, "normal along +y");
        }
        println!(
            "SdfManifold y = {y}: is_empty = {}, {} contacts",
            m.is_empty(),
            m.len()
        );
    }

    // Fewer contacts than the reduction keeps: still not empty.
    for keep in 1..=3 {
        let few = ManifoldConfig {
            max_contacts: keep,
            ..ManifoldConfig::default()
        };
        let m = generate_sdf_manifold(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
            r,
            &plane,
            &few,
        );
        assert_eq!(m.len(), keep, "max_contacts = {keep} keeps {keep}");
        assert!(!m.is_empty(), "{keep} contact(s) is not empty");
    }

    let none = ManifoldConfig {
        max_contacts: 0,
        ..ManifoldConfig::default()
    };
    let m = generate_sdf_manifold(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
        r,
        &plane,
        &none,
    );
    assert!(m.is_empty(), "max_contacts = 0 yields an empty manifold");
}

fn main() {
    gpu_batch_clear();
    adaptive_resize();
    manifold_emptiness();
    println!("all closed forms hold");
}
