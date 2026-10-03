//! Oracles for the production entry points of
//! `alice_physics::collision_mesh_gen` driven by
//! `examples/collision_mesh_generation.rs`: `CollisionMesh`,
//! `CollisionMeshConfig`, `compute_mesh_aabb`, `generate_collision_mesh`,
//! and `simplify_collision_mesh`.
//!
//! # What this file is and is not
//!
//! The module's own `#[cfg(test)]` block (in `src/collision_mesh_gen.rs`)
//! already covers: a sphere mesh has vertices/triangles, an always-positive
//! or always-negative SDF produces an empty mesh, `simplify_collision_mesh`
//! is a no-op when `target_triangles` already meets the current count,
//! `compute_mesh_aabb` on an empty mesh and on a single vertex, and that
//! every triangle index stays in bounds for a generated sphere mesh. None
//! of that is repeated here. What this file adds:
//!
//! * `compute_mesh_aabb` at `Fix128`'s extreme representable magnitude
//!   (confirming the componentwise `<`/`>` comparisons do not need
//!   headroom the way an arithmetic op would), and on a mesh whose
//!   vertices repeat the same point (duplicate-vertex degeneracy),
//! * `generate_collision_mesh`'s `config.resolution.max(2)` clamp: a
//!   `resolution` of `0` or `1` must produce a bit-identical mesh to an
//!   explicit `resolution: 2` config (same bounds, same SDF) -- not just
//!   "does not panic",
//! * `simplify_collision_mesh`'s simplification-ratio edge cases, using
//!   a mesh built from topologically disjoint triangles (no two triangles
//!   share a vertex). For such a mesh, each edge collapse can only ever
//!   degenerate the one triangle that owns the collapsed edge -- no other
//!   triangle references either of its endpoints -- so the loop's
//!   `triangles.len() > target_triangles` exit condition is hit exactly,
//!   never overshot: triangle count decreases by exactly one per
//!   collapse. This gives an exact (not tolerance-based, not a
//!   reimplementation of the shortest-edge search) oracle for
//!   `target_triangles` at `0`, in the interior, at the boundary
//!   (`target == len`), and above the boundary (`target > len`, already
//!   the documented no-op early return but re-checked here against the
//!   disjoint-triangle construction specifically),
//! * a single-triangle mesh collapsing to exactly zero triangles when
//!   `target_triangles == 0` (any one of its three edges collapsing makes
//!   two of the triangle's three indices equal, which the degenerate
//!   filter always removes -- true regardless of which edge the
//!   shortest-edge search happens to pick), and
//! * `CollisionMeshConfig` / `CollisionMesh` as directly constructed
//!   struct literals (not only as `generate_collision_mesh`'s return
//!   value), confirming the public fields round-trip unchanged.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::collision_mesh_gen::{
    compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
    CollisionMeshConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};

// ============================================================================
// CollisionMeshConfig / CollisionMesh -- struct literal round-trip
// ============================================================================

#[test]
fn collision_mesh_config_fields_round_trip() {
    let config = CollisionMeshConfig {
        resolution: 12,
        bounds_min: Vec3Fix::from_int(-5, -5, -5),
        bounds_max: Vec3Fix::from_int(5, 5, 5),
    };
    assert_eq!(config.resolution, 12);
    assert_eq!(config.bounds_min, Vec3Fix::from_int(-5, -5, -5));
    assert_eq!(config.bounds_max, Vec3Fix::from_int(5, 5, 5));
}

#[test]
fn collision_mesh_fields_round_trip() {
    let mesh = CollisionMesh {
        vertices: vec![Vec3Fix::from_int(1, 2, 3)],
        triangles: vec![[0, 0, 0]],
    };
    assert_eq!(mesh.vertices, vec![Vec3Fix::from_int(1, 2, 3)]);
    assert_eq!(mesh.triangles, vec![[0, 0, 0]]);
}

// ============================================================================
// compute_mesh_aabb -- degenerate inputs beyond the module's own tests
// ============================================================================

/// Duplicate vertices (every vertex the same point) must report that
/// point as both `min` and `max`, exactly -- an independent closed form
/// (min == max == the repeated point), not a call to `compute_mesh_aabb`
/// a second time.
#[test]
fn compute_mesh_aabb_all_duplicate_vertices() {
    let p = Vec3Fix::from_int(7, -4, 2);
    let mesh = CollisionMesh {
        vertices: vec![p, p, p, p],
        triangles: vec![[0, 1, 2], [1, 2, 3]],
    };
    let aabb = compute_mesh_aabb(&mesh);
    assert_eq!(aabb.min, p);
    assert_eq!(aabb.max, p);
}

/// `Fix128`'s extreme representable magnitude (`i64::MAX` / `i64::MIN` in
/// the integer half) on both ends at once -- the componentwise `<`/`>`
/// comparisons `compute_mesh_aabb` uses are plain integer-pair
/// comparisons (`Ord` on `(hi, lo)`), so this must not need any arithmetic
/// headroom the way a sum or difference would.
#[test]
fn compute_mesh_aabb_extreme_magnitude() {
    let lo_extreme = Vec3Fix {
        x: Fix128 {
            hi: i64::MIN,
            lo: 0,
        },
        y: Fix128 {
            hi: i64::MIN,
            lo: 0,
        },
        z: Fix128 {
            hi: i64::MIN,
            lo: 0,
        },
    };
    let hi_extreme = Vec3Fix {
        x: Fix128 {
            hi: i64::MAX,
            lo: 0,
        },
        y: Fix128 {
            hi: i64::MAX,
            lo: 0,
        },
        z: Fix128 {
            hi: i64::MAX,
            lo: 0,
        },
    };
    let mid = Vec3Fix::from_int(0, 0, 0);
    let mesh = CollisionMesh {
        vertices: vec![lo_extreme, mid, hi_extreme],
        triangles: vec![[0, 1, 2]],
    };
    let aabb = compute_mesh_aabb(&mesh);
    assert_eq!(aabb.min, lo_extreme);
    assert_eq!(aabb.max, hi_extreme);
}

// ============================================================================
// generate_collision_mesh -- resolution clamp + empty mesh
// ============================================================================

fn sphere_sdf(center: Vec3Fix, radius: Fix128) -> impl Fn(Vec3Fix) -> Fix128 {
    move |p: Vec3Fix| {
        let d = p - center;
        d.length() - radius
    }
}

/// `resolution.max(2)` means `resolution: 0` and `resolution: 1` must both
/// produce a mesh bit-identical to an explicit `resolution: 2` config with
/// the same bounds and the same SDF -- the oracle here is the explicit
/// `resolution: 2` config, built and called independently (not derived
/// from the `resolution: 0`/`1` runs).
#[test]
fn generate_collision_mesh_resolution_zero_and_one_clamp_to_two() {
    let bounds_min = Vec3Fix::from_int(-1, -1, -1);
    let bounds_max = Vec3Fix::from_int(1, 1, 1);
    let radius = Fix128::ONE;

    let reference_config = CollisionMeshConfig {
        resolution: 2,
        bounds_min,
        bounds_max,
    };
    let reference = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, radius), &reference_config);
    assert!(
        !reference.vertices.is_empty(),
        "sanity: the resolution=2 reference mesh must itself be non-empty"
    );

    for clamped_resolution in [0_usize, 1_usize] {
        let clamped_config = CollisionMeshConfig {
            resolution: clamped_resolution,
            bounds_min,
            bounds_max,
        };
        let clamped = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, radius), &clamped_config);
        assert_eq!(
            clamped.vertices, reference.vertices,
            "resolution={clamped_resolution} must clamp to resolution=2 (vertices differ)"
        );
        assert_eq!(
            clamped.triangles, reference.triangles,
            "resolution={clamped_resolution} must clamp to resolution=2 (triangles differ)"
        );
    }
}

/// A degenerate (zero-volume, inverted) bounding box: `bounds_min ==
/// bounds_max`. Every sampled grid point collapses onto the single point
/// `bounds_min`, so every cell has eight identical corner values -- no
/// sign change is possible -- and the mesh must be empty. This is a
/// different degeneracy from the module's own "always positive" /
/// "always negative" SDF tests: here the SDF is a perfectly ordinary
/// sphere SDF, but the sampling volume itself has zero extent.
#[test]
fn generate_collision_mesh_zero_extent_bounds_is_empty() {
    let point = Vec3Fix::from_int(3, 3, 3);
    let config = CollisionMeshConfig {
        resolution: 4,
        bounds_min: point,
        bounds_max: point,
    };
    // The sphere's surface passes nowhere near (3,3,3) at radius 1 from
    // the origin, so even if sampling were not degenerate this SDF would
    // read strictly positive everywhere -- the zero-extent bounds make
    // that doubly true (every corner samples the exact same value).
    let mesh = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, Fix128::ONE), &config);
    assert!(mesh.vertices.is_empty());
    assert!(mesh.triangles.is_empty());
}

// ============================================================================
// simplify_collision_mesh -- simplification-ratio edge cases
// ============================================================================

/// `n` triangles, no two of which share a vertex (each triangle owns its
/// own private set of three vertices). Because no triangle's indices
/// overlap another's, collapsing any one triangle's edge can only ever
/// degenerate that triangle -- the remap `if *idx == remove { *idx = keep
/// }` never touches another triangle's indices, since `remove` and `keep`
/// belong exclusively to the collapsing triangle. So every loop iteration
/// removes exactly one triangle, and the loop's `while triangles.len() >
/// target_triangles` condition is satisfied exactly, not overshot.
fn disjoint_triangles(n: usize) -> CollisionMesh {
    let mut vertices = Vec::with_capacity(n * 3);
    let mut triangles = Vec::with_capacity(n);
    for i in 0..n {
        // Distinct, non-degenerate (non-collinear), and distinct in size
        // per triangle so no two edges across different triangles are
        // ever tied for "shortest" -- irrelevant to the final count (see
        // doc comment above) but keeps the construction unambiguous.
        let base = (i as i64) * 1000;
        let i0 = vertices.len();
        vertices.push(Vec3Fix::from_int(base, 0, 0));
        vertices.push(Vec3Fix::from_int(base + 1 + i as i64, 0, 0));
        vertices.push(Vec3Fix::from_int(base, 1 + i as i64, 0));
        triangles.push([i0, i0 + 1, i0 + 2]);
    }
    CollisionMesh {
        vertices,
        triangles,
    }
}

// Contract change (marching-tetrahedra rewrite, program item 9e): an edge collapse
// must keep the surface a manifold and must not move the border of an open mesh.
// Every vertex of a disjoint triangle is on the border, so none of these edges can
// be collapsed: the mesh is returned as it is, whatever the target. (Until then a
// collapse shrank an isolated triangle to nothing, which removed it from the mesh by
// degenerating it, and the target was always reached.)
#[test]
fn simplify_disjoint_triangles_target_zero_leaves_the_border_alone() {
    let mesh = disjoint_triangles(5);
    let simplified = simplify_collision_mesh(&mesh, 0);
    assert_eq!(simplified.triangles.len(), 5);
    assert_eq!(simplified.vertices, mesh.vertices);
}

#[test]
fn simplify_disjoint_triangles_target_interior_leaves_the_border_alone() {
    let mesh = disjoint_triangles(5);
    let simplified = simplify_collision_mesh(&mesh, 2);
    assert_eq!(simplified.triangles.len(), 5);
}

#[test]
fn simplify_disjoint_triangles_target_equals_len_is_noop() {
    let mesh = disjoint_triangles(5);
    let simplified = simplify_collision_mesh(&mesh, 5);
    assert_eq!(simplified.triangles.len(), 5);
    // The documented early return (`if mesh.triangles.len() <=
    // target_triangles { return mesh.clone(); }`) means this is a literal
    // clone, not just the same count.
    assert_eq!(simplified.vertices, mesh.vertices);
    assert_eq!(simplified.triangles, mesh.triangles);
}

#[test]
fn simplify_disjoint_triangles_target_above_len_is_noop_clone() {
    let mesh = disjoint_triangles(5);
    let simplified = simplify_collision_mesh(&mesh, 9);
    assert_eq!(simplified.vertices, mesh.vertices);
    assert_eq!(simplified.triangles, mesh.triangles);
}

/// Degenerate: a single triangle. All three of its vertices are on the border and
/// each of its edges joins two border vertices, so none can be collapsed (it would
/// pinch the border): the triangle stays, whatever the target. See the contract
/// note above.
#[test]
fn simplify_single_triangle_target_zero_stays_a_triangle() {
    let mesh = CollisionMesh {
        vertices: vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
        ],
        triangles: vec![[0, 1, 2]],
    };
    let simplified = simplify_collision_mesh(&mesh, 0);
    assert_eq!(simplified.triangles.len(), 1);
}

/// Degenerate: the empty mesh. `target_triangles: 0` meets the early
/// return (`0 <= 0`) immediately, so this must stay empty without ever
/// entering the collapse loop (which would index `vertices[keep]` on an
/// empty `Vec` and panic).
#[test]
fn simplify_empty_mesh_target_zero_stays_empty() {
    let mesh = CollisionMesh {
        vertices: vec![],
        triangles: vec![],
    };
    let simplified = simplify_collision_mesh(&mesh, 0);
    assert_eq!(simplified.vertices.len(), 0);
    assert_eq!(simplified.triangles.len(), 0);
}
