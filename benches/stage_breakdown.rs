//! Stage-level workload breakdown benchmark
//!
//! Uses the **workload-differential** method to estimate the cost of each
//! internal stage (integrate / broad-phase / narrow-phase / PGS / joint /
//! sleeping) without touching the deterministic solver internals.
//!
//! For each of three body-count regimes (small / medium / large) we run
//! several distinct workloads. Each workload isolates one or two stages
//! by construction, and the per-stage cost is recovered by subtraction:
//!
//! - `sleeping`  — all bodies asleep, dispatch overhead only
//! - `falling`   — free-fall, no contacts (integrate + light broad-phase)
//! - `bvh_dense` — packed sleeping grid (broad-phase pair enumeration)
//! - `pile`      — dropping stack of colliders (narrow-phase + PGS)
//! - `chain`     — distance-constraint chain (constraint solve)
//!
//! Run with: `cargo bench --bench stage_breakdown -- --sample-size=20`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use alice_physics::sleeping::SleepConfig;
use alice_physics::{DistanceConstraint, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn fast_sleep_config() -> SleepConfig {
    let mut cfg = SleepConfig::default();
    cfg.frames_to_sleep = 1; // sleep after a single idle frame
    cfg
}

const FRAMES: usize = 30; // half-second at 60 FPS

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Workload 1: N sleeping bodies, well separated. Measures dispatch overhead
/// (event frame lifecycle, island rebuild loop, sleeping traversal).
fn workload_sleeping(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.set_sleep_config(fast_sleep_config());
    for i in 0..n {
        let x = (i as i64 % 100) * 10; // 10 m apart, well outside any collision
        let z = (i as i64 / 100) * 10;
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(x, 0, z),
            Fix128::ONE,
        ));
    }
    world
}

/// Workload 2: N bodies in free-fall from staggered heights, well separated.
/// Measures integrate cost, minimal broad-phase (nothing pairs up).
fn workload_falling(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    for i in 0..n {
        let x = (i as i64 % 100) * 10;
        let z = (i as i64 / 100) * 10;
        let y = 100 + (i as i64 % 10);
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(x, y, z),
            Fix128::ONE,
        ));
    }
    world
}

/// Workload 3: N bodies packed on a dense grid, all sleeping. Every neighbour
/// pair enters the broad-phase pair enumeration but no narrow-phase resolution
/// is needed. Measures BVH build + AABB overlap cost.
fn workload_bvh_dense(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.set_sleep_config(fast_sleep_config());
    // 1 unit apart — AABBs overlap heavily but positions stay stable
    let side = (n as f64).sqrt().ceil() as i64;
    for i in 0..n {
        let x = i as i64 % side;
        let z = i as i64 / side;
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(x, 0, z),
            Fix128::ONE,
        ));
    }
    world
}

/// Workload 4: falling stack that collides with a static ground plane.
/// Measures narrow-phase + PGS on top of everything workload_falling includes.
fn workload_pile(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    // Static ground
    world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    // Falling column of bodies over a small footprint so they collide
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

/// Workload 5: chain of N bodies linked by distance constraints. Measures
/// constraint-solve cost on top of integrate.
fn workload_chain(n: usize) -> PhysicsWorld {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let mut prev = None;
    for i in 0..n {
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_int(i as i64 * 2, 50, 0),
            Fix128::ONE,
        );
        let id = world.add_body(body);
        if let Some(p) = prev {
            world.add_distance_constraint(
                DistanceConstraint::new(p, id, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(2))
                    .with_compliance(Fix128::from_ratio(1, 1000)),
            );
        }
        prev = Some(id);
    }
    // Pin the first
    world.bodies[0].inv_mass = Fix128::ZERO;
    world
}

fn run_frames(world: &mut PhysicsWorld) {
    let dt = dt();
    for _ in 0..FRAMES {
        world.step(black_box(dt));
    }
}

fn bench_size(c: &mut Criterion, label: &str, n: usize) {
    let mut group = c.benchmark_group(format!("stage_breakdown/{label}_{n}"));
    // Fewer samples for the largest workloads so the whole run stays sane
    let samples = if n >= 10_000 { 10 } else { 20 };
    group.sample_size(samples);

    group.bench_function("1_sleeping", |b| {
        b.iter_batched(
            || workload_sleeping(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("2_falling", |b| {
        b.iter_batched(
            || workload_falling(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("3_bvh_dense", |b| {
        b.iter_batched(
            || workload_bvh_dense(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("4_pile", |b| {
        b.iter_batched(
            || workload_pile(n),
            |mut w| run_frames(&mut w),
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("5_chain", |b| {
        b.iter_batched(
            || workload_chain(n),
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

criterion_group!(stage_breakdown, bench_all);
criterion_main!(stage_breakdown);
