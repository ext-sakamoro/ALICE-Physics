//! Instruction-count benchmarks (Valgrind / Callgrind via gungraun).
//!
//! Instruction counts do not depend on the load of the machine, so
//! `.github/workflows/bench-gate.yml` runs these on the base commit and on the
//! change in the same job and fails when a count rises past the limit. Each
//! workload is small (Callgrind runs it about 50 times slower) and exercises one
//! hot path: the world step with contacts, the BVH, and the fixed-point maths.
//!
//! Linux only: `cargo bench --bench instruction_counts` needs Valgrind and the
//! `gungraun-runner` binary of the same version as the `gungraun` dev-dependency.

#[cfg(not(target_os = "linux"))]
fn main() {}

#[cfg(target_os = "linux")]
use alice_physics::bvh::BvhPrimitive;
#[cfg(target_os = "linux")]
use alice_physics::collider::AABB;
#[cfg(target_os = "linux")]
use alice_physics::{Fix128, LinearBvh, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
#[cfg(target_os = "linux")]
use gungraun::prelude::*;
#[cfg(target_os = "linux")]
use std::hint::black_box;

/// A 3×3×3 stack of touching spheres on a static floor, 30 steps: broad phase,
/// narrow phase and the contact solve in every substep.
#[cfg(target_os = "linux")]
fn stacked_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::from_int(0, -100, 0)),
        Fix128::from_int(100),
    );
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                world.add_body_with_radius(
                    RigidBody::new_dynamic(Vec3Fix::from_int(2 * x, 1 + 2 * y, 2 * z), Fix128::ONE),
                    Fix128::ONE,
                );
            }
        }
    }
    world
}

#[cfg(target_os = "linux")]
#[library_benchmark]
#[bench::stack_27(setup = stacked_world)]
fn world_step(mut world: PhysicsWorld) -> Vec3Fix {
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        world.step(black_box(dt));
    }
    world.bodies[1].position
}

#[cfg(target_os = "linux")]
fn boxes() -> Vec<BvhPrimitive> {
    (0..256)
        .map(|i: i64| {
            let c = Vec3Fix::from_int(i % 16 * 3, i / 16 * 3, (i * 7) % 5);
            let h = Vec3Fix::from_int(1, 1, 1);
            BvhPrimitive {
                aabb: AABB::new(c - h, c + h),
                index: i as u32,
                morton: 0,
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
#[library_benchmark]
#[bench::boxes_256(setup = boxes)]
fn bvh_build_and_query(prims: Vec<BvhPrimitive>) -> usize {
    let bvh = LinearBvh::build(black_box(prims));
    let q = AABB::new(Vec3Fix::from_int(10, 10, 0), Vec3Fix::from_int(20, 20, 4));
    bvh.query(&q).len()
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn fixed_point_maths() -> Fix128 {
    let mut acc = Fix128::ZERO;
    for i in 1..200 {
        let x = Fix128::from_ratio(i, 7);
        acc = acc + x.sqrt() + x.sin() * x.cos();
    }
    black_box(acc)
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = hot_paths,
    benchmarks = [world_step, bvh_build_and_query, fixed_point_maths]
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = hot_paths);
