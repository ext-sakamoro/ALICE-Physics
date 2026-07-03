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

    let a = Fix128::from_raw(12_345, 0xABCD_EF01_2345_6789);
    let b = Fix128::from_raw(67_890, 0x9876_5432_10FE_DCBA);

    group.bench_function("fix128_mul", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)));
    });

    group.bench_function("fix128_div", |bench| {
        let divisor = Fix128::from_raw(67, 0x9876_5432_10FE_DCBA);
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
            let x = Fix128::from_int(i64::from(i) * 2);
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
            let x = Fix128::from_int(i64::from(i) * 2);
            let z = Fix128::from_int(i64::from(i / 32) * 2);
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

// ============================================================================
// Phase E: per-island scoped oriented TGS solver + Warm-starting observability
// ============================================================================

fn bench_tgs_oriented_scoped(c: &mut Criterion) {
    use alice_physics::math::QuatFix;
    use alice_physics::solver_tgs::{build_islands, ImpulseCache, JointLike, TgsConfig};
    use alice_physics::solver_tgs_hooks_6dof_oriented::{
        Body6DofOrientedState, ContactOriented, Pgs6DofOrientedConfig,
    };
    use alice_physics::solver_tgs_hooks_6dof_oriented_scoped::solve_oriented_islands_serial;

    struct NoJoint;
    impl JointLike for NoJoint {
        fn body_a(&self) -> usize {
            0
        }
        fn body_b(&self) -> usize {
            0
        }
    }
    const NO_JOINTS: [NoJoint; 0] = [];

    fn make_body(px: i64, py: i64, id: u64) -> Body6DofOrientedState {
        Body6DofOrientedState {
            position: [Fix128::from_int(px), Fix128::from_int(py), Fix128::ZERO],
            orientation: QuatFix::IDENTITY,
            linear_velocity: [Fix128::ZERO; 3],
            angular_velocity: [Fix128::ZERO; 3],
            inv_mass: Fix128::from_int(1),
            inv_inertia_local: [Fix128::from_int(1); 3],
            is_dynamic: true,
            stable_id: id,
        }
    }

    fn make_contact(a: usize, b: usize, id: u64) -> ContactOriented {
        ContactOriented {
            body_a: a,
            body_b: b,
            stable_id: id,
            normal: [Fix128::ZERO, Fix128::from_int(1), Fix128::ZERO],
            tangent1: [Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO],
            tangent2: [Fix128::ZERO, Fix128::ZERO, Fix128::from_int(1)],
            r_a: [Fix128::ZERO; 3],
            r_b: [Fix128::ZERO; 3],
            penetration: Fix128::from_ratio(1, 100),
            friction: Fix128::from_ratio(1, 2),
            restitution: Fix128::ZERO,
            accum_normal: Fix128::ZERO,
            accum_tangent1: Fix128::ZERO,
            accum_tangent2: Fix128::ZERO,
        }
    }

    let mut group = c.benchmark_group("tgs_oriented_scoped");

    group.bench_function("serial_6bodies_3islands_60steps", |b| {
        b.iter(|| {
            // 6 bodies grouped into 3 disjoint islands (2 bodies each,
            // connected by one contact).
            let mut bodies: Vec<_> = (0..6)
                .map(|i| make_body((i as i64 / 2) * 10, i as i64, i as u64))
                .collect();
            let mut contacts: Vec<_> = vec![
                make_contact(0, 1, 100),
                make_contact(2, 3, 200),
                make_contact(4, 5, 300),
            ];
            let islands = build_islands(&bodies, &contacts, &NO_JOINTS);
            let mut cache = ImpulseCache::default();
            let cfg = Pgs6DofOrientedConfig::default();
            let tgs_cfg = TgsConfig::default();
            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..60 {
                solve_oriented_islands_serial(
                    &mut bodies,
                    &mut contacts,
                    &islands,
                    &mut cache,
                    cfg,
                    &tgs_cfg,
                    black_box(dt),
                );
            }
            // Report warm-starting hit rate to Criterion via the black box.
            (bodies, cache.stats().hit_rate())
        });
    });

    group.finish();
}

// ============================================================================
// Turn D next-step: LinearBvh refit vs full rebuild
// ============================================================================

fn bench_bvh_refit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_refit_vs_build");

    let prims_1k: Vec<BvhPrimitive> = (0..1000_i64)
        .map(|i| BvhPrimitive {
            aabb: AABB::new(Vec3Fix::from_int(i, 0, 0), Vec3Fix::from_int(i + 1, 1, 1)),
            index: i as u32,
            morton: 0,
        })
        .collect();
    let new_aabbs_1k: Vec<AABB> = (0..1000_i64)
        .map(|i| AABB::new(Vec3Fix::from_int(i, 10, 0), Vec3Fix::from_int(i + 1, 11, 1)))
        .collect();

    group.bench_function("refit_1000_prims", |b| {
        let mut bvh = LinearBvh::build(prims_1k.clone());
        b.iter(|| {
            bvh.refit_leaves(black_box(&new_aabbs_1k));
        });
    });

    group.bench_function("full_rebuild_1000_prims", |b| {
        b.iter(|| {
            let _bvh = LinearBvh::build(black_box(prims_1k.clone()));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_physics_step,
    bench_math_ops,
    bench_bvh_query,
    bench_tgs_oriented_scoped,
    bench_bvh_refit,
);
criterion_main!(benches);
