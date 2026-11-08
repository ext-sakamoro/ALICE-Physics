//! Mesh a raised letter exactly: the outline eroded by a skin, extruded and
//! rounded by the same skin, as conforming tetrahedra.
//!
//! The outline here is an `L` (one reflex corner); the eroded outline, its
//! triangulation and the slab mesh are printed with the closed-form areas
//! they should match.
use alice_physics_core::extruded_tet_mesh::{mesh_rounded_slab, RoundedSlab};
use alice_physics_core::polygon_cdt::{triangulate, CdtConfig};
use alice_physics_core::polygon_offset::{erode, OffsetConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let outline = vec![vec![
        [0.0, 0.0],
        [0.2, 0.0],
        [0.2, 0.08],
        [0.08, 0.08],
        [0.08, 0.2],
        [0.0, 0.2],
    ]];
    let skin = 0.012;

    let cdt = triangulate(
        &outline,
        &CdtConfig {
            max_edge: 0.03,
            ..CdtConfig::default()
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let area = cdt.area2_grid() as f64 * 0.5 * cdt.grid * cdt.grid;
    println!(
        "outline: {} triangles, area {area:.6} (shoelace 0.025600)",
        cdt.triangles.len()
    );

    let eroded =
        erode(&outline, skin, &OffsetConfig { sagitta: 1.0e-5 }).map_err(|e| format!("{e:?}"))?;
    let pts = eroded.points();
    let eroded_area: f64 = pts
        .iter()
        .map(|r| {
            0.5 * (0..r.len())
                .map(|i| r[i][0] * r[(i + 1) % r.len()][1] - r[(i + 1) % r.len()][0] * r[i][1])
                .sum::<f64>()
        })
        .sum();
    println!(
        "eroded by {skin}: {} vertices, area {eroded_area:.6}",
        pts[0].len()
    );

    let mesh = mesh_rounded_slab(
        &outline,
        &RoundedSlab {
            skin,
            half_depth: 0.08,
            boundary_step: 0.012,
            max_edge: 0.03,
            layer: 0.015,
            grid: 1.0e-6,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    println!(
        "slab: {} vertices, {} tetrahedra",
        mesh.vertices.len(),
        mesh.tets.len()
    );
    Ok(())
}
