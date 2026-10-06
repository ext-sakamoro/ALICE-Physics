//! Feasibility check for 6e's question: does an unmoving body skip the
//! margin padding recompute (`DynamicAabbTree::fatten`, which calls
//! `MetricWeights::euclidean_radius`)? Answers by direct measurement
//! against the real `DynamicAabbTree::update` (no src changes).
//!
//! `update()` already short-circuits with a containment check (fat AABB
//! contains the new tight AABB) before it ever calls `fatten`/
//! `euclidean_radius` -- that is read directly from
//! src/dynamic_bvh.rs:135-153. This measures it under two regimes:
//! zero movement (should be 0% reinsert) and the per-substep displacement
//! size seen in the 1000-overlapping-spheres bench (gravity -10 m/s^2,
//! dt = 1/60/8), to see how much of the bench's 1000 bodies actually
//! trigger `fatten` per substep.

use alice_physics::{DynamicAabbTree, Fix128, Vec3Fix, AABB};

fn sphere_aabb(center: Vec3Fix, radius: Fix128) -> AABB {
    let r = Vec3Fix::new(radius, radius, radius);
    AABB::new(center - r, center + r)
}

fn main() {
    let radius = Fix128::ONE;

    // Regime 1: body never moves. Expect 0 reinserts across many updates.
    {
        let mut tree = DynamicAabbTree::new();
        let center = Vec3Fix::new(fx(5), fx(5), fx(5));
        let proxy = tree.insert(sphere_aabb(center, radius), 0);
        let mut reinserts = 0u32;
        for _ in 0..1000 {
            if tree.update(proxy, sphere_aabb(center, radius)) {
                reinserts += 1;
            }
        }
        println!("stationary body: {reinserts}/1000 updates triggered reinsert (fatten/euclidean_radius call)");
    }

    // Regime 2: free-fall displacement per substep at the bench's dt and g,
    // matching benches/physics_bench.rs::thousand_overlapping_spheres_1_step
    // (dt = 1/60, 8 substeps/frame -> substep_dt = 1/480, g = -10).
    {
        let mut tree = DynamicAabbTree::new();
        let substep_dt = 1.0_f64 / 480.0;
        let g = -10.0_f64;
        let mut y = 20.0_f64;
        let mut v = 0.0_f64;
        let center0 = Vec3Fix::new(fx(0), ffx(y), fx(0));
        let proxy = tree.insert(sphere_aabb(center0, radius), 0);
        let mut reinserts = 0u32;
        let substeps = 8 * 60; // 60 frames worth of substeps
        for _ in 0..substeps {
            v += g * substep_dt;
            y += v * substep_dt;
            let center = Vec3Fix::new(fx(0), ffx(y), fx(0));
            if tree.update(proxy, sphere_aabb(center, radius)) {
                reinserts += 1;
            }
        }
        println!(
            "free-falling body (bench dt/g): {reinserts}/{substeps} substeps triggered reinsert"
        );
    }
}

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn ffx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
