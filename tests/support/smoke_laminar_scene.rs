//! The laminar point-plume scene, shared by the long oracle (`examples/smoke_laminar_plume.rs`)
//! and the short push-time guard in `tests/analytic_smoke_plume.rs`, so both build the same
//! physical scene: a box of air 0.32 × 0.64 × 0.32 m (closed; the long run may open its top),
//! heat and momentum diffusing alike (`ν = κ`, Prandtl 1), and a 2 cm cube of heating centred
//! on the floor, its bottom 1 cm above it. Every length is written in metres and the cells are
//! derived from the cell size, so every grid that divides the lengths builds the same plume
//! (a length that is not a whole number of cells is refused, not rounded).

use alice_physics_core::smoke_plume::{CellBox, SmokeConfig, SmokePlume, SmokeSource, SourceKind};
use alice_physics_core::Fix128;

/// A length in metres, held exactly as `num / den` m.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metres(pub i64, pub i64);

impl Metres {
    /// The length in micrometres, exactly (every scene length is a whole number of them).
    pub fn micrometres(self) -> i64 {
        assert_eq!(
            self.0 * 1_000_000 % self.1,
            0,
            "{self:?} is not a whole number of micrometres"
        );
        self.0 * 1_000_000 / self.1
    }
    /// The number of cells of `dx_um` micrometres the length spans, refused unless whole.
    pub fn cells(self, dx_um: i64) -> usize {
        let um = self.micrometres();
        assert!(
            dx_um > 0 && um % dx_um == 0,
            "{self:?} is not a whole number of {dx_um} µm cells"
        );
        (um / dx_um) as usize
    }
}

/// Width of the box along x and z.
pub const BOX_XZ: Metres = Metres(32, 100);
/// Height of the box.
pub const BOX_Y: Metres = Metres(64, 100);
/// Side of the heated cube.
pub const SOURCE_SIDE: Metres = Metres(2, 100);
/// Height of the heated cube's bottom above the floor.
pub const SOURCE_BOTTOM: Metres = Metres(1, 100);
/// Distance from the box's side walls to the heated cube, `(BOX_XZ − SOURCE_SIDE) / 2`.
pub const SOURCE_SIDE_GAP: Metres = Metres(15, 100);

/// The cells of the scene at a cell of `dx_um` micrometres: the box's extents and the
/// source's cell box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub n_xz: usize,
    pub n_y: usize,
    pub source: CellBox,
}

/// The scene's cells at a cell of `dx_um` micrometres, every one derived from a length in
/// metres.
pub fn geometry(dx_um: i64) -> Geometry {
    assert_eq!(
        2 * SOURCE_SIDE_GAP.micrometres() + SOURCE_SIDE.micrometres(),
        BOX_XZ.micrometres(),
        "the source is centred"
    );
    let (gap, side, bottom) = (
        SOURCE_SIDE_GAP.cells(dx_um),
        SOURCE_SIDE.cells(dx_um),
        SOURCE_BOTTOM.cells(dx_um),
    );
    Geometry {
        n_xz: BOX_XZ.cells(dx_um),
        n_y: BOX_Y.cells(dx_um),
        source: CellBox {
            lo: [gap, bottom, gap],
            hi: [gap + side, bottom + side, gap + side],
        },
    }
}

/// The scene at a cell of `dx_mm` millimetres (10 or 5; 2.5 mm through [`scene_um`]), with
/// kinematic viscosity and heat diffusivity both `nu_um2` · 10⁻⁶ m²/s and the source heating at
/// `heat_k_per_s`. The velocity step is at most 1/120 s at 10 mm and shrinks with the cell.
#[allow(
    dead_code,
    reason = "the push-time guard builds the scene by millimetres; the long run by micrometres"
)]
pub fn scene(dx_mm: i64, nu_um2: i64, heat_k_per_s: i64) -> SmokePlume {
    scene_um(dx_mm * 1000, nu_um2, heat_k_per_s)
}

/// [`scene`] at a cell of `dx_um` micrometres.
pub fn scene_um(dx_um: i64, nu_um2: i64, heat_k_per_s: i64) -> SmokePlume {
    let g = geometry(dx_um);
    let mut c = SmokeConfig::air();
    c.step_max = step_um(dx_um);
    c.viscosity_pas = c.density_kg_m3 * Fix128::from_ratio(nu_um2, 1_000_000);
    c.heat_diffusivity_m2_s = Fix128::from_ratio(nu_um2, 1_000_000);
    let mut p = SmokePlume::new(
        g.n_xz,
        g.n_y,
        g.n_xz,
        Fix128::from_ratio(dx_um, 1_000_000),
        c,
    )
    .expect("the laminar scene builds");
    p.add_source(SmokeSource {
        cells: g.source,
        kind: SourceKind::Rates {
            heat_k_per_s: Fix128::from_int(heat_k_per_s),
            soot_kg_m3_s: Fix128::ZERO,
        },
    })
    .expect("the source fits the box");
    p
}

/// The velocity step of the scene at `dx_mm` (s): 1/120 s at 10 mm, in proportion to the cell.
#[allow(
    dead_code,
    reason = "the push-time guard steps by millimetres; the long run by micrometres"
)]
pub fn step(dx_mm: i64) -> Fix128 {
    step_um(dx_mm * 1000)
}

/// [`step`] at a cell of `dx_um` micrometres.
pub fn step_um(dx_um: i64) -> Fix128 {
    Fix128::from_ratio(dx_um, 1_200_000)
}

/// Height of the source's top above the floor (m), the virtual origin the plume's `z` is
/// measured from.
#[allow(
    dead_code,
    reason = "the long run reads it; the push-time guard does not"
)]
pub fn source_top_m() -> f64 {
    (SOURCE_BOTTOM.micrometres() + SOURCE_SIDE.micrometres()) as f64 / 1e6
}

/// How closely the heat leaving the box must match the heat the source adds.
pub const STEADY_TOLERANCE: f64 = 0.05;

/// The plume's own time scale for the steady rule: the time to cross the box's height at the
/// plume's speed, `BOX_Y / w`, in whole seconds rounded up (at least 1). The outflow is
/// compared with the source over this span, and must stay steady for this many seconds before
/// the window opens: an outflow that only passes through the band while the box fills does not
/// stay there for a transit.
#[allow(
    dead_code,
    reason = "the long run and the rule's test read it; the push-time guard does not"
)]
pub fn transit_span_s(speed_m_s: f64) -> Option<usize> {
    let um_s = (speed_m_s * 1e6) as i64;
    if um_s <= 0 {
        return None;
    }
    let height = BOX_Y.micrometres();
    Some(((height + um_s - 1) / um_s).max(1) as usize)
}

/// Whether the outflow history, one `(heat that had left, heat the source had added)` pair per
/// whole second, is steady at its last second over `span` seconds: the heat that left over the
/// last `span` seconds is the heat added within [`STEADY_TOLERANCE`], on both sides (a plume
/// first reaching an open top pushes out the warm air ahead of it, so the outflow overshoots).
#[allow(
    dead_code,
    reason = "the long run and the rule's test read it; the push-time guard does not"
)]
pub fn outflow_steady(hist: &[(i64, i64)], span: usize) -> bool {
    if span == 0 || hist.len() <= span {
        return false;
    }
    let (now, then) = (hist[hist.len() - 1], hist[hist.len() - 1 - span]);
    let added = (now.1 - then.1) as f64;
    added > 0.0 && ((now.0 - then.0) as f64 / added - 1.0).abs() <= STEADY_TOLERANCE
}

/// Whether the outflow has stayed steady ([`outflow_steady`] over `span`) at each of the last
/// `span` whole seconds: the condition for opening the averaging window.
#[allow(
    dead_code,
    reason = "the long run and the rule's test read it; the push-time guard does not"
)]
pub fn sustained_steady(hist: &[(i64, i64)], span: usize) -> bool {
    span > 0
        && hist.len() >= 2 * span
        && (0..span).all(|k| outflow_steady(&hist[..hist.len() - k], span))
}
