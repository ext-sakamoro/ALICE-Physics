//! Benchmarks for ALICE-Physics
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use alice_physics::bvh::BvhPrimitive;
use alice_physics::collider::AABB;
use alice_physics::{Fix128, LinearBvh, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

// ============================================================================
// Physics step benchmarks
// ============================================================================

fn bench_physics_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_step");

    group.bench_function("single_body_60_steps", |b| {
        b.iter(|| {
            let config = PhysicsConfig::default();
            let mut world = PhysicsWorld::new(config);
            world.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(0, 100, 0),
                Fix128::ONE,
            ));
            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..60 {
                world.step(black_box(dt));
            }
            world.bodies[0].position
        });
    });

    group.bench_function("ten_bodies_60_steps", |b| {
        b.iter(|| {
            let config = PhysicsConfig::default();
            let mut world = PhysicsWorld::new(config);
            for i in 0..10 {
                world.add_body(RigidBody::new_dynamic(
                    Vec3Fix::from_int(i * 3, 50, 0),
                    Fix128::ONE,
                ));
            }
            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..60 {
                world.step(black_box(dt));
            }
            world.bodies[0].position
        });
    });

    group.finish();
}

// ============================================================================
// Math operation benchmarks
// ============================================================================

fn bench_math_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_ops");

    let a = Fix128::from_raw(12345, 0xABCDEF0123456789);
    let b = Fix128::from_raw(67890, 0x9876543210FEDCBA);

    group.bench_function("fix128_mul", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)));
    });

    group.bench_function("fix128_div", |bench| {
        let divisor = Fix128::from_raw(67, 0x9876543210FEDCBA);
        bench.iter(|| black_box(black_box(a) / black_box(divisor)));
    });

    group.bench_function("fix128_sqrt", |bench| {
        let val = Fix128::from_int(12345);
        bench.iter(|| black_box(black_box(val).sqrt()));
    });

    let va = Vec3Fix::from_int(3, 4, 5);
    let vb = Vec3Fix::from_int(6, 7, 8);

    group.bench_function("vec3_dot", |bench| {
        bench.iter(|| black_box(black_box(va).dot(black_box(vb))));
    });

    group.bench_function("vec3_cross", |bench| {
        bench.iter(|| black_box(black_box(va).cross(black_box(vb))));
    });

    group.bench_function("vec3_normalize", |bench| {
        bench.iter(|| black_box(black_box(va).normalize()));
    });

    group.bench_function("fix128_sin_cos", |bench| {
        let angle = Fix128::from_ratio(1, 4); // ~0.25 radians
        bench.iter(|| black_box(black_box(angle).sin_cos()));
    });

    group.finish();
}

// ============================================================================
// BVH query benchmarks
// ============================================================================

fn bench_bvh_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_query");

    // Build BVH with 100 primitives
    let prims: Vec<BvhPrimitive> = (0u32..100)
        .map(|i| {
            let x = Fix128::from_int(i as i64 * 2);
            BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO),
                    Vec3Fix::new(x + Fix128::ONE, Fix128::ONE, Fix128::ONE),
                ),
                index: i,
                morton: 0,
            }
        })
        .collect();
    let bvh = LinearBvh::build(prims);

    group.bench_function("query_100_prims", |bench| {
        let q = AABB::new(Vec3Fix::from_int(10, 0, 0), Vec3Fix::from_int(20, 1, 1));
        bench.iter(|| black_box(bvh.query(black_box(&q))));
    });

    // Build BVH with 1000 primitives
    let prims_1k: Vec<BvhPrimitive> = (0u32..1000)
        .map(|i| {
            let x = Fix128::from_int(i as i64 * 2);
            let z = Fix128::from_int((i / 32) as i64 * 2);
            BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::new(x, Fix128::ZERO, z),
                    Vec3Fix::new(x + Fix128::ONE, Fix128::ONE, z + Fix128::ONE),
                ),
                index: i,
                morton: 0,
            }
        })
        .collect();
    let bvh_1k = LinearBvh::build(prims_1k);

    group.bench_function("query_1000_prims", |bench| {
        let q = AABB::new(Vec3Fix::from_int(50, 0, 0), Vec3Fix::from_int(60, 1, 5));
        bench.iter(|| black_box(bvh_1k.query(black_box(&q))));
    });

    group.finish();
}

criterion_group!(benches, bench_physics_step, bench_math_ops, bench_bvh_query);
criterion_main!(benches);
