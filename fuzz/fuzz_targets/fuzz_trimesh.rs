#![no_main]
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::Ray;
use alice_physics::trimesh::TriMesh;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct TriMeshInput {
    /// Vertex count (2..=32).
    vertex_count_raw: u8,
    /// Vertex coordinates (i8, mapped to Fix128 as unit steps).
    vertices: Vec<(i8, i8, i8)>,
    /// Triangle indices (u8, wrapped modulo vertex_count).
    indices: Vec<u8>,
    /// Ray origin.
    ray_origin: (i8, i8, i8),
    /// Ray direction (must be non-zero after normalisation).
    ray_direction: (i8, i8, i8),
    /// Ray max_t (u8, mapped to 1..=64).
    ray_max_t: u8,
    /// Closest-point query location.
    query_point: (i8, i8, i8),
}

// Fuzz triangle mesh construction + queries:
// - `TriMesh::from_indexed` — BVH-accelerated mesh builder
// - `TriMesh::raycast` — BVH ray-triangle intersection
// - `TriMesh::closest_point` — nearest triangle + point on triangle
//
// Must never panic under arbitrary indexed geometry (degenerate triangles,
// out-of-bounds indices modulo'd back, coincident vertices, zero rays).
fuzz_target!(|input: TriMeshInput| {
    let vertex_count = ((input.vertex_count_raw % 31) + 2) as usize; // 2..=32
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let (x, y, z) = input.vertices.get(i).copied().unwrap_or((0, 0, 0));
        vertices.push(Vec3Fix::from_int(x as i64, y as i64, z as i64));
    }

    // Build a triangle-index buffer. Cap at 96 indices (32 triangles) to
    // keep each fuzz iteration bounded. Wrap indices modulo vertex_count so
    // they are always in range (the API itself trims incomplete triangles).
    let raw_indices = &input.indices;
    let index_count = raw_indices.len().min(96) / 3 * 3; // multiple of 3
    let mut indices = Vec::with_capacity(index_count);
    for i in 0..index_count {
        indices.push(raw_indices[i] as u32 % vertex_count as u32);
    }

    // Skip degenerate configurations: no vertices or no complete triangles.
    if vertex_count < 3 || indices.len() < 3 {
        return;
    }

    let mesh = TriMesh::from_indexed(&vertices, &indices);

    // Raycast query — must not panic even if the ray direction is zero
    // (raycast handles the degenerate direction internally).
    let dir_x = input.ray_direction.0 as i64;
    let dir_y = input.ray_direction.1 as i64;
    let dir_z = input.ray_direction.2 as i64;
    if dir_x.abs() + dir_y.abs() + dir_z.abs() > 0 {
        let ray = Ray::new(
            Vec3Fix::from_int(
                input.ray_origin.0 as i64,
                input.ray_origin.1 as i64,
                input.ray_origin.2 as i64,
            ),
            Vec3Fix::from_int(dir_x, dir_y, dir_z),
        );
        let max_t = Fix128::from_int(((input.ray_max_t as i64) % 64).max(1));
        let _ = mesh.raycast(&ray, max_t);
    }

    // Closest-point query — always safe, returns a triangle index + point.
    let query = Vec3Fix::from_int(
        input.query_point.0 as i64,
        input.query_point.1 as i64,
        input.query_point.2 as i64,
    );
    let _ = mesh.closest_point(query);
});
