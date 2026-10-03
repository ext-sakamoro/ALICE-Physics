//! Mesh an SDF into tetrahedra, measure it, and refine it.
//!
//! [`generate`] dices every lattice cube that lies fully inside the SDF into
//! five tetrahedra and skips the rest, so a shape that does not line up with
//! the lattice is meshed as a staircase strictly inside it.
//! [`generate_marching_tets`] additionally clips the cubes the surface
//! crosses, so its mesh conforms to the boundary and has at least as many
//! tetrahedra.
//!
//! The run below meshes a cube that is sized to exactly fill one lattice
//! cell — so the closed form for its tet/vertex/edge counts can be checked
//! by hand against the printed numbers — and then a ball, where the two
//! generators diverge because the ball's boundary crosses cells.
//!
//! ```bash
//! cargo run --example sdf_fem_mesh_generation --features std
//! ```

use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{
    generate, generate_marching_tets, BoundaryFaceError, SdfTetMesh,
};

/// Axis-aligned cube of half-extent `half`, centred on the origin.
fn box_sdf(half: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| {
            let dx = x.abs() - half;
            let dy = y.abs() - half;
            let dz = z.abs() - half;
            dx.max(dy).max(dz)
        },
        |_x, _y, _z| (1.0, 0.0, 0.0),
    )
}

fn ball_sdf(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

fn report(label: &str, mesh: &SdfTetMesh) {
    println!(
        "[sdf_fem_mesh] {label}: {} tets, {} vertices, max edge {:.6}",
        mesh.tet_count(),
        mesh.vertex_count(),
        mesh.max_edge_length()
    );
}

fn main() {
    // --- one lattice cell, exactly -----------------------------------------
    // A cube whose corners sit exactly on the lattice points of a single
    // cell: every corner has distance 0 (inside, by the `<= 0.0` rule), so
    // `generate` meshes exactly one cube. `CUBE_FIVE_TETS` always dices one
    // cube into 5 tetrahedra sharing its 8 corners — no new vertex — so the
    // closed form is tets = 5, vertices = 8. The longest edge in either
    // 5-tet decomposition is a face diagonal of the cube, length
    // `cell * sqrt(2)`.
    let cell = 1.0_f32;
    let one_cell = generate(
        &box_sdf(cell / 2.0),
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
        cell,
    );
    report("generate (one cell)", &one_cell);
    println!(
        "[sdf_fem_mesh] closed form: 5 tets, 8 vertices, max edge {:.6} (cell * sqrt(2))",
        cell * 2.0_f32.sqrt()
    );

    // --- a ball: interior-only vs. surface-conforming -----------------------
    let radius = 0.9_f32;
    let (min, max, cell) = ([-1.2_f32; 3], [1.2_f32; 3], 0.4_f32);
    let sdf = ball_sdf(radius);
    let interior = generate(&sdf, min, max, cell);
    let surface = generate_marching_tets(&sdf, min, max, cell);
    report("generate (ball, interior-only)", &interior);
    report(
        "generate_marching_tets (ball, surface-conforming)",
        &surface,
    );
    println!(
        "[sdf_fem_mesh] surface-conforming has >= interior-only tets: {}",
        surface.tet_count() >= interior.tet_count()
    );

    match surface.boundary_faces() {
        Ok(faces) => println!(
            "[sdf_fem_mesh] boundary_faces: {} triangles on the meshed surface",
            faces.len()
        ),
        Err(BoundaryFaceError::NonManifoldFace { face, uses }) => {
            println!("[sdf_fem_mesh] boundary_faces refused: face {face:?} claimed by {uses} tets");
        }
        Err(other) => println!("[sdf_fem_mesh] boundary_faces refused: {other:?}"),
    }

    // A region with no zero crossing at all meshes to nothing — there is no
    // lattice corner for either generator to call "inside".
    let empty = generate_marching_tets(&sdf, [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], cell);
    println!(
        "[sdf_fem_mesh] generate_marching_tets outside the ball: {} tets (closed form: 0)",
        empty.tet_count()
    );

    // --- the deprecated reporting, for comparison ---------------------------
    // `refine_by_max_edge_length` performs the same conforming refinement as
    // `try_refine_conforming` and remains callable; it just cannot say
    // whether the pass budget ran out, which is why `try_refine_conforming`
    // is preferred everywhere else in this crate.
    let mut legacy = one_cell.clone();
    #[allow(deprecated)]
    let legacy_passes = legacy.refine_by_max_edge_length(0.6, 16);
    report("refine_by_max_edge_length (deprecated, finishes)", &legacy);
    println!("[sdf_fem_mesh] deprecated API reported {legacy_passes} passes");

    let mut legacy_short = one_cell.clone();
    #[allow(deprecated)]
    let legacy_short_passes = legacy_short.refine_by_max_edge_length(0.6, 1);
    println!(
        "[sdf_fem_mesh] deprecated API with a one-pass budget: reports {legacy_short_passes} \
         passes either way (clean finish or exhausted budget look the same — this is exactly \
         what try_refine_conforming's Result<u32, RefineError> fixes)"
    );
}
