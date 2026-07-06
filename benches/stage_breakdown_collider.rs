//! Narrow-phase collision cost measurement — Phase 3 gating bench
//!
//! `benches/stage_breakdown.rs` established that broad-phase and integrate
//! account for ~10 μs/body/frame each and constraint solve is 33x that. But
//! it deliberately skipped narrow-phase because none of its workloads
//! attached explicit colliders — every `RigidBody` was inserted via
//! `add_body`, so `body_collision_radii` was `None` and the BVH walk
//! early-exits before any pairwise contact resolution runs.
//!
//! This bench closes that gap. It compares the pile workload with **and
//! without** attached `Sphere` colliders (via `add_body_with_radius`) so
//! the delta between the two runs is the effective narrow-phase +
//! post-collision solver cost.
//!
//! Per the [GPU offload roadmap](../docs/GPU_OFFLOAD_ROADMAP.md) Phase 3
//! gating: narrow-phase GPU offload only justifies itself if the delta
//! exceeds 30% of total frame cost on a realistic workload. If the delta
//! stays small, joint expansion (Phase 2.5) delivers more per unit effort.
//!
//! Run: `cargo bench --bench stage_breakdown_collider -- --quick`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const FRAMES: usize = 30;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Baseline: falling column, no colliders attached (repeats the
/// `stage_breakdown::workload_pile` shape). All bodies are `add_body`,
/// so `body_collision_radii` is `None` and no narrow-phase runs.
fn workload_pile_no_collider(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    let footprint = ((n as f64).sqrt().ceil() as i64).max(4);
    for i in 0..n {
        let x = i as i64 % footprint;
        let z = (i as i64 / footprint) % footprint;
        let y = 5 + (i as i64 / (footprint * footprint));
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(x, y, z),
            Fix128::ONE,
        ));
    }
    world
}

/// With colliders: same falling column but each body is inserted via
/// `add_body_with_radius(0.5)`. Broad-phase now enumerates real pairs
/// and narrow-phase runs sphere-sphere contact resolution + PGS solve.
fn workload_pile_with_collider(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let radius = Fix128::from_ratio(1, 2);
    world.add_body_with_radius(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)), radius);
    let footprint = ((n as f64).sqrt().ceil() as i64).max(4);
    for i in 0..n {
        let x = i as i64 % footprint;
        let z = (i as i64 / footprint) % footprint;
        let y = 5 + (i as i64 / (footprint * footprint));
        world.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE),
            radius,
        );
    }
    world
}

fn run_frames(world: &mut PhysicsWorld) {
    let dt = dt();
    for _ in 0..FRAMES {
        world.step(black_box(dt));
    }
}

fn bench_size(c: &mut Criterion, label: &str, n: usize) {
    let mut group = c.benchmark_group(format!("collider_delta/{label}_{n}"));
    let samples = if n >= 10_000 { 10 } else { 20 };
    group.sample_size(samples);

    group.bench_function("pile_no_collider", |b| {
        b.iter_batched(
            || workload_pile_no_collider(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("pile_with_collider", |b| {
        b.iter_batched(
            || workload_pile_with_collider(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_size(c, "small", 100);
    bench_size(c, "medium", 1_000);
    bench_size(c, "large", 10_000);
}

criterion_group!(stage_breakdown_collider, bench_all);
criterion_main!(stage_breakdown_collider);
