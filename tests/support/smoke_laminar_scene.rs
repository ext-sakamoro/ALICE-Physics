//! The laminar point-plume scene, shared by the long oracle (`examples/smoke_laminar_plume.rs`)
//! and the short push-time guard in `tests/analytic_smoke_plume.rs`, so both build the same
//! physical scene: a closed box of air 0.32 × 0.64 × 0.32 m, heat and momentum diffusing alike
//! (`ν = κ`, Prandtl 1), and a 2 cm cube of heating on the floor's centre. The box, the source
//! and the rates are physical sizes, so a finer grid resolves the same plume.

use alice_physics_core::smoke_plume::{CellBox, SmokeConfig, SmokePlume, SmokeSource, SourceKind};
use alice_physics_core::Fix128;

/// Box size (m): width along x and z, height along y.
pub const BOX_XZ_MM: i64 = 320;
pub const BOX_Y_MM: i64 = 640;
/// Side of the heated cube on the floor (mm).
pub const SOURCE_MM: i64 = 20;

/// The scene at a cell of `dx_mm` millimetres (`dx_mm` divides 20), with kinematic viscosity
/// and heat diffusivity both `nu_um2` · 10⁻⁶ m²/s and the source heating at `heat_k_per_s`.
/// The velocity step is at most 1/120 s at 10 mm and shrinks with the cell.
pub fn scene(dx_mm: i64, nu_um2: i64, heat_k_per_s: i64) -> SmokePlume {
    assert!(dx_mm > 0 && SOURCE_MM % dx_mm == 0, "the cell must divide the source");
    let n_xz = (BOX_XZ_MM / dx_mm) as usize;
    let n_y = (BOX_Y_MM / dx_mm) as usize;
    let mut c = SmokeConfig::air();
    c.step_max = step(dx_mm);
    c.viscosity_pas = c.density_kg_m3 * Fix128::from_ratio(nu_um2, 1_000_000);
    c.heat_diffusivity_m2_s = Fix128::from_ratio(nu_um2, 1_000_000);
    let mut p = SmokePlume::new(n_xz, n_y, n_xz, Fix128::from_ratio(dx_mm, 1000), c)
        .expect("the laminar scene builds");
    let s = (SOURCE_MM / dx_mm) as usize;
    let lo = n_xz / 2 - s / 2;
    p.add_source(SmokeSource {
        cells: CellBox {
            lo: [lo, 1, lo],
            hi: [lo + s, 1 + s, lo + s],
        },
        kind: SourceKind::Rates {
            heat_k_per_s: Fix128::from_int(heat_k_per_s),
            soot_kg_m3_s: Fix128::ZERO,
        },
    })
    .expect("the source fits the box");
    p
}

/// The velocity step of the scene at `dx_mm` (s): 1/120 s at 10 mm, in proportion to the cell.
pub fn step(dx_mm: i64) -> Fix128 {
    Fix128::from_ratio(dx_mm, 1200)
}

/// Height of the source's top above the floor (m), the virtual origin the plume's `z` is
/// measured from.
pub fn source_top_m(dx_mm: i64) -> f64 {
    (dx_mm + SOURCE_MM) as f64 / 1000.0
}
