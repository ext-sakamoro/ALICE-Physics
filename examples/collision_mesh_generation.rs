//! Production entry point for `alice_physics::collision_mesh_gen`:
//! `CollisionMesh`, `CollisionMeshConfig`, `compute_mesh_aabb`,
//! `generate_collision_mesh`, and `simplify_collision_mesh`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all five as `unwired` --
//! the module's own `#[cfg(test)]` block exercises them, but tests do not
//! count as production callers for the wiring guard, and nothing in
//! `src/` / `examples/` / `benches/` called any of the five before this
//! file existed. This example is that caller.
//!
//! `tests/analytic_collision_mesh_gen_wiring.rs` holds the closed-form /
//! invariant oracles for degenerate and simplification-ratio edge cases
//! that this file's diagnostic prints do not repeat.
//!
//! ```bash
//! cargo run --example collision_mesh_generation --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::collision_mesh_gen::{
    compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
    CollisionMeshConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};

/// A sphere's signed-distance function: negative inside, zero on the
/// surface, positive outside. `generate_collision_mesh` treats negative
/// values as "inside" per its own doc comment.
fn sphere_sdf(center: Vec3Fix, radius: Fix128) -> impl Fn(Vec3Fix) -> Fix128 {
    move |p: Vec3Fix| {
        let d = p - center;
        d.length() - radius
    }
}

fn main() {
    // ------------------------------------------------------------------
    // 1. CollisionMesh / compute_mesh_aabb -- a hand-built, non-axis-
    //    symmetric mesh whose AABB is computed independently by reading
    //    the three coordinates off each vertex directly (not by calling
    //    `compute_mesh_aabb` a second time, and not by any min/max loop
    //    that mirrors the function under test -- just literal min/max of
    //    the six numbers in each axis, written out by hand below).
    // ------------------------------------------------------------------
    let hand_mesh = CollisionMesh {
        vertices: vec![
            Vec3Fix::from_int(-3, 7, 2),
            Vec3Fix::from_int(5, -1, 9),
            Vec3Fix::from_int(1, 4, -6),
        ],
        triangles: vec![[0, 1, 2]],
    };
    let hand_aabb = compute_mesh_aabb(&hand_mesh);
    // min.x = min(-3, 5, 1) = -3; max.x = max(-3, 5, 1) = 5
    // min.y = min(7, -1, 4) = -1; max.y = max(7, -1, 4) = 7
    // min.z = min(2, 9, -6) = -6; max.z = max(2, 9, -6) = 9
    let want_min = Vec3Fix::from_int(-3, -1, -6);
    let want_max = Vec3Fix::from_int(5, 7, 9);
    assert_eq!(
        hand_aabb.min, want_min,
        "[collision_mesh_gen] MISMATCH compute_mesh_aabb min: got {:?}, want {:?}",
        hand_aabb.min, want_min
    );
    assert_eq!(
        hand_aabb.max, want_max,
        "[collision_mesh_gen] MISMATCH compute_mesh_aabb max: got {:?}, want {:?}",
        hand_aabb.max, want_max
    );
    println!(
        "[collision_mesh_gen] ok compute_mesh_aabb(hand-built 3-vertex mesh): min=({:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1})",
        hand_aabb.min.x.to_f64(),
        hand_aabb.min.y.to_f64(),
        hand_aabb.min.z.to_f64(),
        hand_aabb.max.x.to_f64(),
        hand_aabb.max.y.to_f64(),
        hand_aabb.max.z.to_f64()
    );

    // ------------------------------------------------------------------
    // 2. CollisionMeshConfig / generate_collision_mesh -- a sphere of
    //    radius 2 sampled on a 16^3 grid over [-3,3]^3. Every emitted
    //    vertex sits on a grid edge where the SDF changes sign, found by
    //    linearly interpolating the two sampled endpoint values. Because
    //    a signed distance field is 1-Lipschitz (|grad f| = 1 a.e.), the
    //    true surface crossing on an edge of length `h` cannot be more
    //    than `h` away (in SDF value, hence in Euclidean distance from
    //    the true radius) from any point on that same edge -- including
    //    the linearly-interpolated one. So `|dist(vertex, center) - r|`
    //    is bounded by the grid cell edge length, independent of the
    //    marching-cubes implementation itself.
    // ------------------------------------------------------------------
    let resolution = 16_usize;
    let bounds_min = Vec3Fix::from_int(-3, -3, -3);
    let bounds_max = Vec3Fix::from_int(3, 3, 3);
    let config = CollisionMeshConfig {
        resolution,
        bounds_min,
        bounds_max,
    };
    let radius = Fix128::from_int(2);
    let sphere_mesh = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, radius), &config);
    assert!(
        !sphere_mesh.vertices.is_empty() && !sphere_mesh.triangles.is_empty(),
        "[collision_mesh_gen] MISMATCH generate_collision_mesh(sphere): expected a non-empty mesh"
    );

    let edge_len = (bounds_max.x - bounds_min.x).to_f64() / resolution as f64;
    let lipschitz_tol = edge_len; // h, derived above -- not a tuned fudge factor
    let r_f64 = radius.to_f64();
    let mut max_dev = 0.0_f64;
    for v in &sphere_mesh.vertices {
        let dev = (v.length().to_f64() - r_f64).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    assert!(
        max_dev <= lipschitz_tol,
        "[collision_mesh_gen] MISMATCH generate_collision_mesh(sphere) vertex radius: \
         max |len-r| = {max_dev:.6} exceeds Lipschitz bound h = {lipschitz_tol:.6}"
    );
    println!(
        "[collision_mesh_gen] ok generate_collision_mesh(sphere r={r_f64}): {} vertices, {} triangles, \
         max |len-r| = {max_dev:.6} <= h = {lipschitz_tol:.6}",
        sphere_mesh.vertices.len(),
        sphere_mesh.triangles.len()
    );

    // The same Lipschitz bound applies to compute_mesh_aabb's output,
    // since it is just the componentwise min/max of exactly those
    // vertices: every coordinate of every vertex lies in
    // [-(r+h), r+h] on each axis.
    let sphere_aabb = compute_mesh_aabb(&sphere_mesh);
    let bound = r_f64 + lipschitz_tol;
    for (label, lo, hi) in [
        ("x", sphere_aabb.min.x.to_f64(), sphere_aabb.max.x.to_f64()),
        ("y", sphere_aabb.min.y.to_f64(), sphere_aabb.max.y.to_f64()),
        ("z", sphere_aabb.min.z.to_f64(), sphere_aabb.max.z.to_f64()),
    ] {
        assert!(
            lo >= -bound && hi <= bound,
            "[collision_mesh_gen] MISMATCH compute_mesh_aabb(sphere) {label}: [{lo:.6},{hi:.6}] \
             outside [-{bound:.6},{bound:.6}]"
        );
    }
    println!(
        "[collision_mesh_gen] ok compute_mesh_aabb(sphere mesh) within +-{bound:.6} of origin on every axis: \
         min=({:.4},{:.4},{:.4}) max=({:.4},{:.4},{:.4})",
        sphere_aabb.min.x.to_f64(),
        sphere_aabb.min.y.to_f64(),
        sphere_aabb.min.z.to_f64(),
        sphere_aabb.max.x.to_f64(),
        sphere_aabb.max.y.to_f64(),
        sphere_aabb.max.z.to_f64()
    );

    // ------------------------------------------------------------------
    // 3. simplify_collision_mesh -- edge-collapse midpoints are convex
    //    combinations of existing vertex positions, so every vertex of a
    //    simplified mesh lies within the convex hull of the original
    //    mesh's vertices, hence within the original mesh's AABB too.
    //    This is an invariant of the algorithm's collapse rule (replace
    //    a vertex with the componentwise average of two existing
    //    vertices), not a recomputation of which edge gets picked.
    // ------------------------------------------------------------------
    let target = sphere_mesh.triangles.len() / 2;
    let simplified = simplify_collision_mesh(&sphere_mesh, target);
    assert!(
        simplified.triangles.len() <= sphere_mesh.triangles.len(),
        "[collision_mesh_gen] MISMATCH simplify_collision_mesh: triangle count must not increase"
    );
    for tri in &simplified.triangles {
        for &idx in tri {
            assert!(
                idx < simplified.vertices.len(),
                "[collision_mesh_gen] MISMATCH simplify_collision_mesh: triangle index {idx} out of bounds"
            );
        }
    }
    let simplified_aabb = compute_mesh_aabb(&simplified);
    assert!(
        simplified_aabb.min.x >= sphere_aabb.min.x
            && simplified_aabb.min.y >= sphere_aabb.min.y
            && simplified_aabb.min.z >= sphere_aabb.min.z
            && simplified_aabb.max.x <= sphere_aabb.max.x
            && simplified_aabb.max.y <= sphere_aabb.max.y
            && simplified_aabb.max.z <= sphere_aabb.max.z,
        "[collision_mesh_gen] MISMATCH simplify_collision_mesh AABB containment: simplified {:?}..{:?} \
         must stay inside original {:?}..{:?}",
        simplified_aabb.min,
        simplified_aabb.max,
        sphere_aabb.min,
        sphere_aabb.max
    );
    println!(
        "[collision_mesh_gen] ok simplify_collision_mesh(sphere, target={target}): {} -> {} triangles, \
         AABB min=({:.4},{:.4},{:.4}) max=({:.4},{:.4},{:.4}) contained in original",
        sphere_mesh.triangles.len(),
        simplified.triangles.len(),
        simplified_aabb.min.x.to_f64(),
        simplified_aabb.min.y.to_f64(),
        simplified_aabb.min.z.to_f64(),
        simplified_aabb.max.x.to_f64(),
        simplified_aabb.max.y.to_f64(),
        simplified_aabb.max.z.to_f64()
    );

    println!("[collision_mesh_gen] all checks passed");
}
