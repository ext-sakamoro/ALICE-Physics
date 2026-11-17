//! # MODEL
//!
//! - **Equations**: a thin water layer of depth `h ≥ 0` over a fixed bed `b` on a square grid
//!   of cells of side `l`, gravity along −y. Mass: `∂h/∂t + ∇·q = r − e`, with `r` the rain
//!   rate and `e` the evaporation rate (both m/s) ([`WaterFilm::step_once`]). The regimes,
//!   chosen per face from the declared deepest film `h_film` ([`FilmConfig::film_depth_max`])
//!   and join step `J` ([`FilmConfig::pool_join_step`], [`join_step`]):
//!
//!   | regime | where | treated by | holds exactly |
//!   |---|---|---|---|
//!   | film | `h_f ≤ h_film` | explicit flux `q = sign(Δη)·min(g h_f³ \|Δη\| / (3ν l), h_f √(g h_f))`, lubrication (viscous, Nusselt) capped by critical | steady film depth (Nusselt, a stroke's sampled section), critical outflow over a free edge |
//!   | still pool | `h_f > h_film` and `\|Δη\| ≤ J`, its throughflow within its level tolerance | the faces' cells are joined (union-find) and levelled exactly in integers | level surface, overflow at the rim height, fill level against volume |
//!   | deep flow | a pool whose throughflow (rim flow, rain and deposits inside it) needs a gradient `ΔH = 3 ν Q L² / (g h̄³ A)` beyond its tolerance, or a deep face that is neither a spill nor inertial | its cells step on their own: the film flux at `⌈8 min(lub, crit) / E⌉` local substeps (the face's diffusion number) | uniform flow in a channel deeper than the film, the gradient of a pool fed from inside, a letter filling from its runoff |
//!   | free overfall | `h_f > h_film`, the lower side's surface at or below the crest, over a drop in the bed or an open edge | the broad-crested weir `q = (2/3)^{3/2} √g H^{3/2}` ([`weir_transfer`]) at `⌈8 t_w / E⌉` local substeps | the head of a pool spilling over a sill |
//!   | out of range | a deep front onto a dry floor of the same height (a dam break), or a deep face over a bed step taller than the film with the lower side submerged and the flux at critical (a drowned sill) | [`ParticipantFault::OutOfRange`], the state unchanged, the cells in [`WaterFilm::last_band_cells`] | — |
//!
//!   `η = b + h`; `h_f = max(η_L, η_R) − max(b_L, b_R)` (zero or above) is the depth over the
//!   higher bed of the two cells ([`face_depth`]), so no water crosses a face while the
//!   surface stands below the higher bed: a basin holds water until its surface passes the
//!   rim. On a steady incline of angle `α` the film flux is the Nusselt flux
//!   `q = g h³ sinα / (3ν)`. The film flux is evaluated in integers: with `H = h_f` and
//!   `E = |Δη|` in quanta, the lubrication transfer is `⌊H³E / D_l⌋` and the critical transfer
//!   `⌊⌊√(H³)⌋ · N_c / 2⁶⁴⌋`, both floored, with `D_l = 3 ν l² 2⁹⁶ / (g Δt)` and
//!   `N_c = √g Δt / l · 2⁴⁸` taken once per substep as the raw bits of their [`Fix128`] values
//!   ([`FluxConstants`]); a product past 128 bits saturates, which is exact because it then
//!   exceeds the `⌊h_f/4⌋` cap ([`MAX_LUBRICATION_DIVISOR`], [`MAX_FACE_DEPTH`]).
//!
//!   Discretisation: each substep (1) adds rain and removes evaporation, (2) joins pools
//!   and levels each exactly: the common surface `η*` is the largest with
//!   `Σ max(η* − b, 0) ≤ V` over its cells (bisection on integers), the remainder one quantum
//!   each to the lowest-index cells with `b ≤ η*`, (3) moves water across every other face
//!   at once, each face read from the same state:
//!   `t = min(⌊|q|·Δt/l⌋, ⌊|Δη|/8⌋, ⌊h_f/4⌋)` with the sign of `Δη`
//!   ([`simultaneous_transfer`]), the moves summed per cell, so the result does not depend on
//!   the order the faces are visited and no cell gives more than it holds. The potential
//!   `P = Σ (η² − b²)` falls with every film move: with `δ` the change per cell,
//!   `ΔP = −2 Σ_f t_f Δη_f + Σ_c δ_c²`; `t ≤ Δη/8` gives `Σ t Δη ≥ 8 Σ t²` and four faces per
//!   cell give `Σ δ² ≤ 8 Σ t²`, so `ΔP ≤ −8 Σ t² < 0` when anything moves; levelling a pool
//!   never raises it. (4) the faces of the deep-flow and overfall regimes then take `k` local
//!   substeps of `Δt / k` each, every one from the state the last left and with the still
//!   pools they touch levelled after it; `k` is the largest count any of them needs, from the
//!   state at the start of the substep (deterministic), at most
//!   [`FilmConfig::max_local_substeps`] ([`WaterFilm::last_local_substeps`]). The substep is
//!   `Δt = h / ⌈h / Δt_s⌉` with
//!   `Δt_s = min(Δt_max, 3 ν l² / (8 g h_film³))` ([`WaterFilm::substep`],
//!   [`WaterFilm::subcycles`]), from declared inputs only: no film face then meets the
//!   `⌊|Δη|/8⌋` cap ([`WaterFilm::last_cap_faces`] counts any that does).
//! - **Units**: SI. Lengths and depths m, time s, `q` m²/s, `g` m/s², `ν` m²/s, rates m/s
//!   (rain given in mm/h converts with [`FilmConfig::rain_mm_per_hour`]). The depth is held
//!   as an integer number of quanta of `2⁻³²` m ([`DEPTH_QUANTUM_BITS`],
//!   [`metres_to_quanta`]); fluxes and rates are [`Fix128`]. `h_film` m, `J` and the pool
//!   tolerance as fractions of a depth.
//! - **Conservation**: volume is conserved exactly, as integers: a transfer leaves one cell
//!   and enters the other, levelling redistributes a pool's own volume
//!   ([`WaterFilm::step_once`]); the ledger `Σ depth = initial + rain + deposits − evaporation
//!   − outflow over open edges` holds to the last quantum ([`FilmLedger::expected`]). Momentum
//!   is not modelled (no inertia).
//! - **Range**: cell side `l` from 1 mm to 1 m ([`FilmBed::from_heights`]). The lubrication
//!   flux holds for `Re ≤ 500` and `Fr < 1`; at `Fr ≥ 1` the critical cap gives the critical
//!   depth `(q²/g)^{1/3}`. The film regime is resolved at the derived substep (a stroke of
//!   1 mm at slope 1/200 matches the sampled-section closed form to `5·10⁻⁶` at 1 cm cells,
//!   one notch longer `Δt` misses by +9 % at 2 cm and −20 % at 1 cm). Measured on 2 cm cells:
//!   between the deepest film the substep serves and the depth where a flowing pool stays
//!   within 1 % of level lies a band (1.6–5.4 mm at slope 1/1000, 1.6–3.1 mm at 1/200, none at
//!   1/50 where the flow is supercritical); the cells in it step at local substeps (measured:
//!   uniform flow 2.5 mm deep at 1/200 and 3 mm at 1/1000 on 2 cm cells within `5·10⁻⁵` of
//!   the closed form at 6 and 8 local substeps), only where it is. A pool stays levelled while
//!   its throughflow needs `ΔH ≤ max(tolerance · h̄, 7 quanta per cell of length)`
//!   ([`FilmConfig::pool_level_tolerance`], default 1 %): a level pool stands in for that
//!   gradient, and the tolerance is the error accepted. A free overfall matches the
//!   broad-crested weir to `1.1·10⁻³` at the film substep. A drowned sill is outside: with
//!   the film flux it runs at critical, 1.84 times the weir and stiff (about 1 600 local
//!   substeps at 15 mm on 1 cm cells), and the submerged weir laws are stiff near level
//!   (Villemonte: +20 % at 1/256 s, +2.5 % as `Δt → 0`). More than [`FilmConfig::max_subcycles`]
//!   substeps, a depth leaving the integer range, a face depth above 4 m ([`MAX_FACE_DEPTH`])
//!   or local substeps beyond [`FilmConfig::max_local_substeps`], a substep so short that
//!   `D_l` passes 2⁹⁸ is [`ParticipantFault::OutOfRange`] and
//!   changes nothing ([`WaterFilm::advance`]).
//! - **Approximations**: 2.5D, one depth per column, the bed a height field sampled once per
//!   cell centre ([`FilmBed::from_height_fn`], [`FilmBed::from_sdf`]); gravity along −y only;
//!   no inertia, surface tension, contact angle or infiltration ([`simultaneous_transfer`]);
//!   rain uniform over the grid and evaporation at a fixed rate on wet cells; a still pool
//!   rises everywhere at once (no waves, surges or sloshing: a 7 m groove 3 cm deep would
//!   take 13 s for a gravity wave to cross); a point inflow must arrive gently (a deposit
//!   whose bump makes a deep face inertial is refused: feed per substep, not per second);
//!   the bed is fixed while the film steps, a new one is
//!   set with [`WaterFilm::set_bed`]. Rest: pools are level to one quantum; a film stops
//!   where its lubrication transfer rounds to zero, `H³E < D_l`, so a flat film rests at
//!   about `D_l^{1/4}` (37 µm at `l` = 1 cm and `Δt` = 1/256 s, measured 38 µm). The join
//!   step bounds the jump when two pools meet: they join only when their surfaces are within
//!   `J` of each other, so levelling at the join moves at most `J`.
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/analytic_water_film.rs::piled_deep_water_is_refused::the_potential_never_rises_without_a_source::a_column_spilling_four_ways_never_goes_negative::volume_is_conserved_to_the_last_quantum::a_groove_overflows_at_its_rim_height::a_groove_fills_in_the_closed_form_time::rewind_and_rerun_are_bit_identical::face_order_does_not_change_the_state::a_tampered_payload_is_refused::degenerate_inputs_are_refused::the_film_steps_and_rewinds_inside_a_world::a_steady_film_on_an_incline_has_the_nusselt_depth::a_wet_cell_dries_in_the_closed_form_time::an_sdf_bed_and_a_height_bed_agree::the_substep_count_depends_on_declared_inputs_only::the_mixed_scene_digest_is_pinned::the_integer_square_root_is_exact::the_flux_law_holds_at_its_bounds::a_cell_drains_over_its_edges_at_the_critical_rate::a_flowing_channel_deeper_than_the_film_is_stepped_not_levelled::a_filling_groove_is_accepted_and_levels::flow_along_a_stroke_has_the_lubrication_depth::a_dam_break_front_below_a_spill_is_refused::two_basins_join_over_a_sill_without_a_jump::the_join_step_is_one_percent_of_the_shallower_side::a_free_overfall_drains_at_the_weir_rate::flow_in_the_band_has_the_lubrication_depth::a_pool_fed_from_inside_keeps_its_gradient::a_pool_fed_by_rain_keeps_its_gradient::a_letter_fills_from_its_runoff::a_new_bed_keeps_every_cells_water::a_face_deeper_than_the_law_accepts_is_refused::a_drowned_sill_is_refused::a_gently_fed_pool_stays_level`
//!
//! Author: Moroya Sakamoto

use crate::det_hash::DetHasher;
use crate::math::Fix128;
use crate::sdf_collider::SdfField;
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepRule,
    SubstepCtx,
};
use core::hash::Hasher;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::sync::Arc;

/// Snapshot tag of [`WaterFilm`]: the ASCII code `FILM`, big endian.
pub const WATER_FILM_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"FILM"));

/// Bits below the metre in a depth quantum: one quantum is `2⁻³²` m.
pub const DEPTH_QUANTUM_BITS: u32 = 32;

/// Observation channel: total water volume (m³).
pub const WATER_FILM_OBS_VOLUME: u32 = 0;
/// Observation channel: substeps taken by the last [`Participant::substep`].
pub const WATER_FILM_OBS_SUBCYCLES: u32 = 1;
/// Observation channel: number of wet cells.
pub const WATER_FILM_OBS_WET_CELLS: u32 = 2;

const STATE_VERSION: u32 = 1;
// version, digest, generation, steps, two carries, ledger (5), last substeps, last moved
const HEADER_LEN: usize = 4 + 8 + 8 + 8 + 16 + 16 + 8 * 5 + 4 + 8;
// per cell: depth i64, substeps since wet u32, water deposited since the last advance i64
const PER_CELL: usize = 8 + 4 + 8;

/// Metres to whole depth quanta, floored.
#[must_use]
pub fn metres_to_quanta(m: Fix128) -> i64 {
    let raw = ((m.hi as i128) << 64) | (m.lo as i128);
    (raw >> (64 - DEPTH_QUANTUM_BITS)) as i64
}

/// Depth quanta to metres (exact).
#[must_use]
pub fn quanta_to_metres(q: i64) -> Fix128 {
    let raw = (q as i128) << (64 - DEPTH_QUANTUM_BITS);
    Fix128::from_raw((raw >> 64) as i64, raw as u64)
}

/// Metres as a fractional number of quanta (`m · 2³²`, exact shift).
fn metres_in_quanta(m: Fix128) -> Fix128 {
    let raw = (((m.hi as i128) << 64) | (m.lo as i128)) << DEPTH_QUANTUM_BITS;
    Fix128::from_raw((raw >> 64) as i64, raw as u64)
}

/// How the face flux is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilmScheme {
    /// Lubrication flux capped by the critical flux, every face at once from the same state,
    /// each face moving at most `⌊|Δη|/8⌋` and `⌊h_f/4⌋`; no inertia, no flux state.
    DiffusionWave,
}

/// What happens at the edge of the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmEdge {
    /// A wall: nothing crosses the edge.
    Closed,
    /// Water leaves over the edge as onto a dry cell with the same bed; it is counted in
    /// [`FilmLedger::outflow`] and never comes back.
    Open,
}

/// Why a bed or film could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmBuildError {
    /// The grid has no cells, the heights do not match it, or a new bed is on another grid
    /// or does not raise the generation.
    BadGrid,
    /// The cell side is outside 1 mm .. 1 m.
    CellOutOfRange,
    /// The SDF has no surface between the two heights at some column.
    NoSurface,
    /// A configuration value is out of range.
    BadConfig,
}

/// The bed under the film: one height per cell centre, in depth quanta, with a generation
/// number that a rebuilt bed must raise ([`WaterFilm::set_bed`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilmBed {
    nx: usize,
    nz: usize,
    cell: Fix128,
    origin_x: Fix128,
    origin_z: Fix128,
    heights: Vec<i64>,
    generation: u64,
}

impl FilmBed {
    /// A bed of `nx × nz` cells of side `cell` (m), cell `(i, k)` centred at
    /// `(origin.0 + (i + ½)·cell, origin.1 + (k + ½)·cell)`, heights in metres row-major
    /// (`i + k·nx`), quantised (floored) once.
    ///
    /// # Errors
    ///
    /// [`FilmBuildError::BadGrid`] or [`FilmBuildError::CellOutOfRange`].
    pub fn from_heights(
        nx: usize,
        nz: usize,
        cell: Fix128,
        origin: (Fix128, Fix128),
        heights_m: &[Fix128],
        generation: u64,
    ) -> Result<Self, FilmBuildError> {
        if nx == 0 || nz == 0 || heights_m.len() != nx * nz {
            return Err(FilmBuildError::BadGrid);
        }
        if cell < Fix128::from_ratio(1, 1000) || cell > Fix128::ONE {
            return Err(FilmBuildError::CellOutOfRange);
        }
        Ok(Self {
            nx,
            nz,
            cell,
            origin_x: origin.0,
            origin_z: origin.1,
            heights: heights_m.iter().map(|h| metres_to_quanta(*h)).collect(),
            generation,
        })
    }

    /// The centre of cell `(i, k)` in metres.
    #[must_use]
    pub fn centre(&self, i: usize, k: usize) -> (Fix128, Fix128) {
        let half = self.cell.half();
        (
            self.origin_x + self.cell * Fix128::from_int(i as i64) + half,
            self.origin_z + self.cell * Fix128::from_int(k as i64) + half,
        )
    }

    /// A bed sampled from a height function `height(x, z)` (m, f32) at the cell centres.
    ///
    /// # Errors
    ///
    /// As [`Self::from_heights`].
    pub fn from_height_fn(
        nx: usize,
        nz: usize,
        cell: Fix128,
        origin: (Fix128, Fix128),
        generation: u64,
        height: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, FilmBuildError> {
        let mut bed = Self::from_heights(
            nx,
            nz,
            cell,
            origin,
            &vec_of(nx * nz, Fix128::ZERO),
            generation,
        )?;
        for k in 0..nz {
            for i in 0..nx {
                let (x, z) = bed.centre(i, k);
                bed.heights[i + k * nx] =
                    metres_to_quanta(Fix128::from_f32(height(x.to_f32(), z.to_f32())));
            }
        }
        Ok(bed)
    }

    /// A bed sampled from the zero set of an SDF: at each cell centre the surface height
    /// between `y_range.0` (inside, distance < 0) and `y_range.1` (outside, distance > 0),
    /// found by 48 bisections on the distance's sign (f32).
    ///
    /// # Errors
    ///
    /// [`FilmBuildError::NoSurface`] when a column has no sign change between the two
    /// heights, otherwise as [`Self::from_heights`].
    pub fn from_sdf(
        nx: usize,
        nz: usize,
        cell: Fix128,
        origin: (Fix128, Fix128),
        generation: u64,
        field: &dyn SdfField,
        y_range: (f32, f32),
    ) -> Result<Self, FilmBuildError> {
        let mut bed = Self::from_heights(
            nx,
            nz,
            cell,
            origin,
            &vec_of(nx * nz, Fix128::ZERO),
            generation,
        )?;
        for k in 0..nz {
            for i in 0..nx {
                let (xf, zf) = bed.centre(i, k);
                let (x, z) = (xf.to_f32(), zf.to_f32());
                let (mut lo, mut hi) = y_range;
                if field.distance(x, lo, z) >= 0.0 || field.distance(x, hi, z) <= 0.0 {
                    return Err(FilmBuildError::NoSurface);
                }
                for _ in 0..48 {
                    let mid = lo + (hi - lo) * 0.5;
                    if field.distance(x, mid, z) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                bed.heights[i + k * nx] = metres_to_quanta(Fix128::from_f32(lo + (hi - lo) * 0.5));
            }
        }
        Ok(bed)
    }

    /// Cells along x.
    #[must_use]
    pub fn nx(&self) -> usize {
        self.nx
    }
    /// Cells along z.
    #[must_use]
    pub fn nz(&self) -> usize {
        self.nz
    }
    /// Cell side (m).
    #[must_use]
    pub fn cell(&self) -> Fix128 {
        self.cell
    }
    /// Generation number.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Bed heights in depth quanta, row-major.
    #[must_use]
    pub fn heights(&self) -> &[i64] {
        &self.heights
    }

    fn hash_into(&self, h: &mut DetHasher) {
        h.write(&(self.nx as u64).to_le_bytes());
        h.write(&(self.nz as u64).to_le_bytes());
        for f in [self.cell, self.origin_x, self.origin_z] {
            h.write(&f.hi.to_le_bytes());
            h.write(&f.lo.to_le_bytes());
        }
        h.write(&self.generation.to_le_bytes());
        for b in &self.heights {
            h.write(&b.to_le_bytes());
        }
    }
}

fn vec_of<T: Clone>(n: usize, v: T) -> Vec<T> {
    let mut out = Vec::with_capacity(n);
    out.resize(n, v);
    out
}

/// Constants of the film. Build it with [`FilmConfig::water`] and change fields from there;
/// [`WaterFilm::new`] checks the values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FilmConfig {
    /// Flux scheme.
    pub scheme: FilmScheme,
    /// Edge condition.
    pub edge: FilmEdge,
    /// Gravity magnitude `g` (m/s², > 0), along −y.
    pub gravity: Fix128,
    /// Kinematic viscosity `ν` (m²/s, > 0).
    pub viscosity: Fix128,
    /// Rain rate (m/s, ≥ 0), uniform over the grid.
    pub rain: Fix128,
    /// Evaporation rate (m/s, ≥ 0) on wet cells.
    pub evaporation: Fix128,
    /// Depth (quanta, ≥ 0) above which a cell counts as wet.
    pub wet_threshold: i64,
    /// Longest substep `Δt_max` (s, > 0); the substep count depends on this and the step
    /// only ([`WaterFilm::subcycles`]).
    pub max_substep: Fix128,
    /// Most substeps one [`WaterFilm::advance`] may take (≥ 1).
    pub max_subcycles: u32,
    /// Deepest film (m, > 0) the lubrication flux serves: a face deeper than this joins its
    /// cells into a pool, levelled exactly. The substep is at most `3 ν l² / (8 g h³)` for
    /// this depth, so no lubrication face meets the `⌊|Δη|/8⌋` cap.
    pub film_depth_max: Fix128,
    /// How far a pool's surface may be from level (a fraction of its mean depth, ≥ 0)
    /// before a pool carrying flow is refused as outside the model: a level pool stands in
    /// for a surface that falls by `ΔH = 3 ν Q L² / (g h̄³ A)` across it, and the error is
    /// accepted while `ΔH ≤ tolerance · h̄` (never less than 7 quanta per cell of length).
    pub pool_level_tolerance: Fix128,
    /// Largest surface step (a fraction of the shallower side's depth, ≥ 0; never less than
    /// 7 quanta) across which a face deeper than the film joins two cells into one pool. A
    /// deeper face with a larger step (water standing over a sill against a lower basin)
    /// stays a face carrying the capped (critical) flux, so the lower side fills over time
    /// and the two join when their levels meet: levelling at a join moves at most this step.
    pub pool_join_step: Fix128,
    /// Most local substeps (≥ 1) one substep may take on the faces that need a shorter step
    /// than the film's (a pool carrying throughflow, a deep lubrication face, a free
    /// overfall); more is [`ParticipantFault::OutOfRange`].
    pub max_local_substeps: u32,
}

impl FilmConfig {
    /// Water at 20 °C (`ν = 1.0e-6` m²/s) under standard gravity (`9.80665` m/s²), no rain,
    /// no evaporation, wet above 1 µm, substeps of at most 1/256 s, at most 65 536 substeps
    /// per advance.
    #[must_use]
    pub fn water(scheme: FilmScheme, edge: FilmEdge) -> Self {
        Self {
            scheme,
            edge,
            gravity: Fix128::from_ratio(980_665, 100_000),
            viscosity: Fix128::from_ratio(1, 1_000_000),
            rain: Fix128::ZERO,
            evaporation: Fix128::ZERO,
            wet_threshold: metres_to_quanta(Fix128::from_ratio(1, 1_000_000)),
            max_substep: Fix128::from_ratio(1, 256),
            max_subcycles: 1 << 16,
            film_depth_max: Fix128::from_ratio(1, 1000),
            pool_level_tolerance: Fix128::from_ratio(1, 100),
            pool_join_step: Fix128::from_ratio(1, 100),
            max_local_substeps: 1 << 12,
        }
    }

    /// A rate in mm/h as m/s (`mm_per_hour / 3 600 000`).
    #[must_use]
    pub fn rain_mm_per_hour(mm_per_hour: Fix128) -> Fix128 {
        mm_per_hour.div_int(3_600_000)
    }

    fn hash_into(&self, h: &mut DetHasher) {
        h.write(&[self.scheme as u8, self.edge as u8]);
        for f in [
            self.gravity,
            self.viscosity,
            self.rain,
            self.evaporation,
            self.max_substep,
        ] {
            h.write(&f.hi.to_le_bytes());
            h.write(&f.lo.to_le_bytes());
        }
        h.write(&self.wet_threshold.to_le_bytes());
        h.write(&self.max_subcycles.to_le_bytes());
        h.write(&self.film_depth_max.hi.to_le_bytes());
        h.write(&self.film_depth_max.lo.to_le_bytes());
        h.write(&self.pool_level_tolerance.hi.to_le_bytes());
        h.write(&self.pool_level_tolerance.lo.to_le_bytes());
        h.write(&self.pool_join_step.hi.to_le_bytes());
        h.write(&self.pool_join_step.lo.to_le_bytes());
        h.write(&self.max_local_substeps.to_le_bytes());
    }
}

/// Volume bookkeeping in depth quanta (one quantum over one cell).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilmLedger {
    /// Water set at the start ([`WaterFilm::set_initial_depth`]).
    pub initial: i64,
    /// Rain added.
    pub rain: i64,
    /// Point deposits added ([`WaterFilm::deposit`]).
    pub deposits: i64,
    /// Evaporated.
    pub evaporated: i64,
    /// Left over open edges.
    pub outflow: i64,
}

impl FilmLedger {
    /// `initial + rain + deposits − evaporated − outflow`: what the film must hold.
    #[must_use]
    pub fn expected(&self) -> i64 {
        self.initial + self.rain + self.deposits - self.evaporated - self.outflow
    }
}

/// The depth over the higher bed of two cells: `max(η_L, η_R) − max(b_L, b_R)`, zero or
/// above (quanta).
#[must_use]
pub fn face_depth(b_l: i64, h_l: i64, b_r: i64, h_r: i64) -> i64 {
    let eta = (b_l + h_l).max(b_r + h_r);
    (eta - b_l.max(b_r)).max(0)
}

/// The join step of a face in quanta: `max(⌊fraction · min(h_a, h_b)⌋, 7)`.
#[must_use]
pub fn join_step(fraction: Fix128, h_a: i64, h_b: i64) -> i64 {
    (fraction * Fix128::from_int(h_a.min(h_b))).hi.max(7)
}

/// Largest face depth the flux law accepts, `2³⁴` quanta (4 m): `h_f³` then fits in 102 bits.
pub const MAX_FACE_DEPTH: u64 = 1 << 34;

/// Largest lubrication divisor `D_l` the flux law accepts (`2⁹⁸`): with `h_f ≤ 2³⁴` a product
/// `h_f³ |Δη|` that overflows 128 bits then means a lubrication transfer above `2³⁰` quanta,
/// more than any `⌊h_f/4⌋`, so saturating it is exact.
pub const MAX_LUBRICATION_DIVISOR: u128 = 1 << 98;

/// The two per-substep constants of the integer flux law, from the cell side `l`, the
/// substep `Δt`, gravity `g` and viscosity `ν` (all [`Fix128`]):
/// `lubrication = ⌊3 ν l² 2⁹⁶ / (g Δt)⌋` and `critical = ⌊√g · Δt / l · 2⁻¹⁶ · 2⁶⁴⌋`, each the
/// raw Q64.64 bits of the [`Fix128`] value (so rounded as [`Fix128`] multiplication, division
/// and square root round).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluxConstants {
    /// `D_l`: the lubrication transfer is `⌊h_f³ |Δη| / D_l⌋` quanta.
    pub lubrication: u128,
    /// `N_c`: the critical transfer is `⌊⌊√(h_f³)⌋ · N_c / 2⁶⁴⌋` quanta.
    pub critical: u128,
    /// `N_w = C_w N_c` with `C_w = (2/3)^{3/2}`: the free-overfall (broad-crested weir)
    /// transfer is `⌊⌊√(H³)⌋ · N_w / 2⁶⁴⌋` quanta for a head `H` over the crest.
    pub weir: u128,
}

/// `(2/3)^{3/2} = √(8/27)`, the broad-crested weir coefficient of `q = C_w √g H^{3/2}`.
fn weir_coefficient() -> Fix128 {
    Fix128::from_ratio(8, 27).sqrt()
}

fn fix_bits(f: Fix128) -> u128 {
    ((f.hi as u128) << 64) | u128::from(f.lo)
}

impl FluxConstants {
    /// The constants for a substep `dt`; `None` when `dt`, `g`, `ν` or `l` is not positive or
    /// `D_l` exceeds [`MAX_LUBRICATION_DIVISOR`].
    #[must_use]
    pub fn new(cell: Fix128, dt: Fix128, g: Fix128, nu: Fix128) -> Option<Self> {
        if cell <= Fix128::ZERO || dt <= Fix128::ZERO || g <= Fix128::ZERO || nu <= Fix128::ZERO {
            return None;
        }
        let x = Fix128::from_int(3) * nu * cell * cell * Fix128::from_int(1i64 << 32) / (g * dt);
        let c = (g.sqrt() * dt / cell).shr_bits(16);
        let w = weir_coefficient() * c;
        if x.is_negative() || c.is_negative() || w.is_negative() {
            return None;
        }
        let lubrication = fix_bits(x);
        if lubrication == 0 || lubrication > MAX_LUBRICATION_DIVISOR {
            return None;
        }
        Some(Self {
            lubrication,
            critical: fix_bits(c),
            weir: fix_bits(w),
        })
    }
}

/// `⌊√n⌋` for a `u128`, by bits (exact).
#[must_use]
pub fn isqrt_u128(n: u128) -> u128 {
    let mut rem = n;
    let mut root = 0u128;
    let mut bit = 1u128 << 126;
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// Whether the critical transfer `⌊⌊√(H³)⌋ · N_c / 2⁶⁴⌋` is below `m`, without the square
/// root: it is iff `⌊√(H³)⌋ < T` with `T = ⌈m · 2⁶⁴ / N_c⌉`, iff `H³ < T²` (exact).
fn critical_below(hf: u64, h3: u128, m: u128, n_c: u128) -> bool {
    if n_c == 0 {
        return m > 0;
    }
    // a cheap lower bound first: ⌊√(H³)⌋ ≥ H ⌊√H⌋, so the critical transfer is at least
    // ⌊H ⌊√H⌋ N_c / 2⁶⁴⌋; when that already reaches m, it is not below m
    let low = (u128::from(hf) * u128::from(isqrt_u64(hf)))
        .checked_mul(n_c)
        .map_or(u128::MAX, |v| v >> 64);
    if low >= m {
        return false;
    }
    if m >= 1 << 63 {
        return critical(h3, n_c) < m;
    }
    let t = (m << 64).div_ceil(n_c);
    match t.checked_mul(t) {
        Some(t2) => h3 < t2,
        None => true,
    }
}

/// `⌊√n⌋` for a `u64`, by bits from the highest set pair (exact).
fn isqrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut rem = n;
    let mut root = 0u64;
    let mut bit = 1u64 << ((63 - n.leading_zeros()) & !1);
    while bit != 0 {
        if rem >= root + bit {
            rem -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// The critical transfer `⌊⌊√(H³)⌋ · N_c / 2⁶⁴⌋` (saturating).
fn critical(h3: u128, n_c: u128) -> u128 {
    isqrt_u128(h3)
        .checked_mul(n_c)
        .map_or(u128::MAX, |v| v >> 64)
}

/// The film transfer of a face and whether a stability cap held it below the lubrication
/// flux while that flux was below the critical one, computed once.
fn film_transfer(first: (i64, i64), second: (i64, i64), k: FluxConstants) -> Option<(i64, bool)> {
    let d_eta = (first.0 + first.1) - (second.0 + second.1);
    let hf = face_depth(first.0, first.1, second.0, second.1).unsigned_abs();
    if hf > MAX_FACE_DEPTH {
        return None;
    }
    let e = d_eta.unsigned_abs();
    if hf == 0 || e == 0 {
        return Some((0, false));
    }
    let caps = u128::from((e / 8).min(hf / 4));
    let h3 = u128::from(hf) * u128::from(hf) * u128::from(hf);
    let lub = h3
        .checked_mul(u128::from(e))
        .map_or(u128::MAX, |v| v / k.lubrication);
    // the critical transfer is computed only where it binds
    let crit_below_lub = critical_below(hf, h3, lub, k.critical);
    let t = if caps == 0 {
        0
    } else if crit_below_lub && critical_below(hf, h3, caps, k.critical) {
        critical(h3, k.critical)
    } else {
        lub.min(caps)
    };
    let capped = !crit_below_lub && t < lub;
    let t = t as i64;
    Some((if d_eta > 0 { t } else { -t }, capped))
}

/// The water (quanta) a face moves in a substep when every face moves at once from the same
/// state, in integers: with `H = h_f` and `E = |Δη|` in quanta,
/// `t = min(⌊H³E / D_l⌋, ⌊⌊√(H³)⌋·N_c / 2⁶⁴⌋, ⌊E/8⌋, ⌊H/4⌋)` with the sign of `Δη`
/// (the lubrication and the critical transfer, each floored; a product past 128 bits
/// saturates, which is exact by [`MAX_LUBRICATION_DIVISOR`]). `None` when `H` exceeds
/// [`MAX_FACE_DEPTH`].
#[must_use]
pub fn simultaneous_transfer(
    first: (i64, i64),
    second: (i64, i64),
    k: FluxConstants,
) -> Option<i64> {
    film_transfer(first, second, k).map(|(t, _)| t)
}

/// [`simultaneous_transfer`] and whether a stability cap (`⌊|Δη|/8⌋` or `⌊h_f/4⌋`) held the
/// transfer below the lubrication flux while that flux was below the critical one (a
/// lubrication face outside its diffusion number).
#[must_use]
pub fn simultaneous_transfer_report(
    first: (i64, i64),
    second: (i64, i64),
    k: FluxConstants,
) -> Option<(i64, bool)> {
    film_transfer(first, second, k)
}

/// The water (quanta) a free overfall moves in a substep: the side with the higher surface
/// spills over the higher bed (the crest) into a side whose surface is at or below it, at
/// the broad-crested weir flux `q = (2/3)^{3/2} √g H^{3/2}` per unit width with `H` the head
/// over the crest: `t = min(⌊⌊√(H³)⌋·N_w / 2⁶⁴⌋, ⌊E/8⌋, ⌊H/4⌋)` with the sign of `Δη`.
/// `None` when `H` exceeds [`MAX_FACE_DEPTH`].
#[must_use]
pub fn weir_transfer(first: (i64, i64), second: (i64, i64), k: FluxConstants) -> Option<i64> {
    let d_eta = (first.0 + first.1) - (second.0 + second.1);
    let hf = face_depth(first.0, first.1, second.0, second.1).unsigned_abs();
    if hf > MAX_FACE_DEPTH {
        return None;
    }
    let caps = (d_eta.unsigned_abs() / 8).min(hf / 4);
    if caps == 0 {
        return Some(0);
    }
    let h3 = u128::from(hf) * u128::from(hf) * u128::from(hf);
    let w = isqrt_u128(h3)
        .checked_mul(k.weir)
        .map_or(u128::MAX, |v| v >> 64);
    let t = w.min(u128::from(caps)) as i64;
    Some(if d_eta > 0 { t } else { -t })
}

/// How a face deeper than the film is stepped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaceLaw {
    /// The film flux (lubrication capped by critical).
    Film,
    /// A free overfall over a drop or an open edge.
    Weir,
}

/// Pool id of a cell in no pool.
const NOT_POOLED: u32 = u32::MAX;

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

/// A thin water film over a [`FilmBed`]; a [`Participant`] of kind `FILM`.
///
/// # Snapshot payload
///
/// Little endian: `version: u32` (1), `digest: u64` (FNV-1a 64 over the configuration and the
/// bed), the bed generation `u64`, the substep count `u64`, the rain and evaporation carries
/// (two [`Fix128`] as `hi: i64, lo: u64`), the ledger (five `i64`), the last substep count
/// `u32`, the quanta moved in the last substep `u64`, then per cell the depth `i64`, the
/// substeps since it was last wet `u32` and the water deposited since the last advance `i64`.
/// [`Participant::check_state`] refuses another length ([`StateError::Length`]), version or
/// digest (which covers the bed and its generation), or a negative depth or deposit
/// ([`StateError::InvalidValue`]).
#[derive(Clone, Debug)]
pub struct WaterFilm {
    config: FilmConfig,
    bed: FilmBed,
    digest: u64,
    depth: Vec<i64>,
    dry_age: Vec<u32>,
    rain_carry: Fix128,
    evap_carry: Fix128,
    ledger: FilmLedger,
    steps: u64,
    last_subcycles: u32,
    last_moved: u64,
    reverse_faces: bool,
    // per-substep work buffer, not state
    scratch: Vec<i64>,
    // the substep used: min(max_substep, 3 ν l² / (8 g h_film³)), from declared inputs
    substep: Fix128,
    last_pooled_cells: Vec<usize>,
    last_band_cells: Vec<usize>,
    last_cap_faces: u32,
    // water deposited per cell since the last advance (quanta): a source inside a pool counts
    // as its inflow through every substep of the advance
    fed: Vec<i64>,
    last_local_substeps: u32,
    // the substeps of the advance in progress, over which a deposit's inflow is spread (1
    // outside an advance)
    fed_share: u32,
    // the faces of the grid (x faces then z faces) as cell pairs, `NO_CELL` past an edge;
    // built once from the grid's shape and shared by every copy
    faces: Arc<[(u32, u32)]>,
}

/// A face side past the edge of the grid.
const NO_CELL: u32 = u32::MAX;

// x faces then z faces, each as (cell on the low side, cell on the high side)
fn build_faces(nx: usize, nz: usize) -> Arc<[(u32, u32)]> {
    let mut faces = Vec::with_capacity((nx + 1) * nz + nx * (nz + 1));
    let cell = |i: usize, k: usize| (i + k * nx) as u32;
    for k in 0..nz {
        for i in 0..=nx {
            faces.push((
                if i > 0 { cell(i - 1, k) } else { NO_CELL },
                if i < nx { cell(i, k) } else { NO_CELL },
            ));
        }
    }
    for k in 0..=nz {
        for i in 0..nx {
            faces.push((
                if k > 0 { cell(i, k - 1) } else { NO_CELL },
                if k < nz { cell(i, k) } else { NO_CELL },
            ));
        }
    }
    faces.into()
}

fn side_cell(c: u32) -> Option<usize> {
    if c == NO_CELL {
        None
    } else {
        Some(c as usize)
    }
}

// the cells of each pool with more than one cell, grouped (pools in order of their root,
// cells ascending)
fn pool_groups(pool: &[u32]) -> Vec<Vec<usize>> {
    let mut pairs: Vec<(u32, usize)> = pool
        .iter()
        .enumerate()
        .filter(|(_, p)| **p != NOT_POOLED)
        .map(|(c, p)| (*p, c))
        .collect();
    pairs.sort_unstable();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut last = NOT_POOLED;
    for (p, c) in pairs {
        if p != last {
            groups.push(Vec::new());
            last = p;
        }
        if let Some(g) = groups.last_mut() {
            g.push(c);
        }
    }
    groups.retain(|g| g.len() > 1);
    groups
}

impl WaterFilm {
    /// A dry film over `bed`.
    ///
    /// # Errors
    ///
    /// [`FilmBuildError::BadConfig`] for a non-positive gravity, viscosity or substep, a
    /// negative rate or threshold, or no substeps.
    pub fn new(config: FilmConfig, bed: FilmBed) -> Result<Self, FilmBuildError> {
        if config.gravity <= Fix128::ZERO
            || config.viscosity <= Fix128::ZERO
            || config.max_substep <= Fix128::ZERO
            || config.rain.is_negative()
            || config.evaporation.is_negative()
            || config.wet_threshold < 0
            || config.max_subcycles == 0
            || config.film_depth_max <= Fix128::ZERO
            || config.pool_level_tolerance.is_negative()
            || config.pool_join_step.is_negative()
            || config.max_local_substeps == 0
        {
            return Err(FilmBuildError::BadConfig);
        }
        let n = bed.nx * bed.nz;
        let mut film = Self {
            config,
            digest: 0,
            depth: vec_of(n, 0),
            dry_age: vec_of(n, u32::MAX),
            bed,
            rain_carry: Fix128::ZERO,
            evap_carry: Fix128::ZERO,
            ledger: FilmLedger::default(),
            steps: 0,
            last_subcycles: 0,
            last_moved: 0,
            reverse_faces: false,
            scratch: Vec::new(),
            substep: Fix128::ZERO,
            last_pooled_cells: Vec::new(),
            last_band_cells: Vec::new(),
            last_cap_faces: 0,
            fed: vec_of(n, 0),
            last_local_substeps: 0,
            fed_share: 1,
            faces: Arc::from(Vec::new()),
        };
        film.faces = build_faces(film.bed.nx, film.bed.nz);
        film.digest = film.config_digest();
        let l = film.bed.cell;
        let h = film.config.film_depth_max;
        let limit = Fix128::from_int(3) * film.config.viscosity * l * l
            / (Fix128::from_int(8) * film.config.gravity * h * h * h);
        film.substep = limit.min(film.config.max_substep);
        if film.substep <= Fix128::ZERO {
            return Err(FilmBuildError::BadConfig);
        }
        Ok(film)
    }

    fn config_digest(&self) -> u64 {
        let mut h = DetHasher::default();
        self.config.hash_into(&mut h);
        self.bed.hash_into(&mut h);
        h.finish()
    }

    /// Set the water of cell `(i, k)` (quanta), counted in [`FilmLedger::initial`]. Returns
    /// `false` (and changes nothing) for a cell outside the grid or a negative depth.
    pub fn set_initial_depth(&mut self, i: usize, k: usize, quanta: i64) -> bool {
        if i >= self.bed.nx || k >= self.bed.nz || quanta < 0 {
            return false;
        }
        let c = i + k * self.bed.nx;
        self.ledger.initial += quanta - self.depth[c];
        self.depth[c] = quanta;
        self.refresh_wet(c);
        true
    }

    /// Add `volume` (m³) at the point `(x, z)` to the cell under it, as whole quanta
    /// (floored); returns the quanta added (0 outside the grid or for a negative volume).
    /// Landed rain parcels from [`crate::particle`] become film water this way.
    pub fn deposit(&mut self, x: Fix128, z: Fix128, volume: Fix128) -> i64 {
        if volume.is_negative() {
            return 0;
        }
        let fi = ((x - self.bed.origin_x) / self.bed.cell).floor();
        let fk = ((z - self.bed.origin_z) / self.bed.cell).floor();
        if fi.is_negative() || fk.is_negative() {
            return 0;
        }
        let (i, k) = (fi.hi as usize, fk.hi as usize);
        if i >= self.bed.nx || k >= self.bed.nz {
            return 0;
        }
        let q = metres_to_quanta(volume / (self.bed.cell * self.bed.cell));
        let c = i + k * self.bed.nx;
        self.depth[c] += q;
        self.fed[c] += q;
        self.ledger.deposits += q;
        self.refresh_wet(c);
        q
    }

    /// Replace the bed (same grid). The water depth per cell is kept; the generation must
    /// grow, and the digest changes with the bed, so a snapshot taken over the old bed is
    /// refused.
    ///
    /// # Errors
    ///
    /// [`FilmBuildError::BadGrid`] for another grid or a generation that does not grow.
    pub fn set_bed(&mut self, bed: FilmBed) -> Result<(), FilmBuildError> {
        if bed.nx != self.bed.nx
            || bed.nz != self.bed.nz
            || bed.cell != self.bed.cell
            || bed.generation <= self.bed.generation
        {
            return Err(FilmBuildError::BadGrid);
        }
        self.bed = bed;
        self.digest = self.config_digest();
        Ok(())
    }

    /// The bed.
    #[must_use]
    pub fn bed(&self) -> &FilmBed {
        &self.bed
    }
    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &FilmConfig {
        &self.config
    }
    /// Depth per cell (quanta), row-major.
    #[must_use]
    pub fn depths(&self) -> &[i64] {
        &self.depth
    }
    /// Depth of cell `(i, k)` in metres.
    #[must_use]
    pub fn depth_m(&self, i: usize, k: usize) -> Fix128 {
        quanta_to_metres(self.depth[i + k * self.bed.nx])
    }
    /// Substeps since each cell was last wet (`0` = wet now, `u32::MAX` = never wet).
    #[must_use]
    pub fn dry_ages(&self) -> &[u32] {
        &self.dry_age
    }
    /// Total water (quanta over one cell).
    #[must_use]
    pub fn total(&self) -> i64 {
        self.depth.iter().sum()
    }
    /// The ledger.
    #[must_use]
    pub fn ledger(&self) -> FilmLedger {
        self.ledger
    }
    /// Substeps taken so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }
    /// Quanta moved across faces in the last substep (0 = no face moved water).
    #[must_use]
    pub fn last_moved(&self) -> u64 {
        self.last_moved
    }
    /// Visit the faces in reverse order. The state must not depend on it; this exists so a
    /// test can show that.
    pub fn set_reverse_faces(&mut self, reverse: bool) {
        self.reverse_faces = reverse;
    }

    fn refresh_wet(&mut self, c: usize) {
        if self.depth[c] > self.config.wet_threshold {
            self.dry_age[c] = 0;
        }
    }

    /// The number of substeps `advance(h)` takes: `⌈h / Δt_max⌉` (at least 1), from the step
    /// and [`FilmConfig::max_substep`] only; `None` above [`FilmConfig::max_subcycles`].
    #[must_use]
    pub fn subcycles(&self, h: Fix128) -> Option<u32> {
        let need = (h / self.substep).ceil();
        if need.hi < 1 {
            return Some(1);
        }
        if need.hi > i64::from(self.config.max_subcycles) {
            return None;
        }
        Some(need.hi as u32)
    }

    // (bed, depth) of one side of a face; past an open edge, a dry cell with the inside bed
    fn side(&self, me: Option<usize>, other: Option<usize>) -> Option<(i64, i64)> {
        match (me, other) {
            (Some(c), _) => Some((self.bed.heights[c], self.depth[c])),
            (None, Some(o)) if self.config.edge == FilmEdge::Open => Some((self.bed.heights[o], 0)),
            _ => None,
        }
    }

    // level each pool exactly: the common surface η* is the largest with
    // Σ max(η* − b, 0) ≤ V over its cells, and the r quanta left over go one each to the r
    // lowest-index cells whose bed is at or below η*
    fn level_pools(&mut self, pool: &[u32]) -> Result<(), ParticipantFault> {
        for cells in &pool_groups(pool) {
            let volume: i128 = cells.iter().map(|c| i128::from(self.depth[*c])).sum();
            let fill = |eta: i64| -> i128 {
                cells
                    .iter()
                    .map(|c| i128::from((eta - self.bed.heights[*c]).max(0)))
                    .sum()
            };
            let mut lo = cells
                .iter()
                .map(|c| self.bed.heights[*c])
                .min()
                .unwrap_or(0);
            let mut hi = lo
                .checked_add(i64::try_from(volume).map_err(|_| ParticipantFault::OutOfRange)?)
                .and_then(|v| v.checked_add(1))
                .ok_or(ParticipantFault::OutOfRange)?;
            // invariant: fill(lo) ≤ V < fill(hi)
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if fill(mid) <= volume {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let mut left = volume - fill(lo);
            for &c in cells {
                let b = self.bed.heights[c];
                let mut d = (lo - b).max(0);
                if left > 0 && b <= lo {
                    d += 1;
                    left -= 1;
                }
                self.depth[c] = d;
            }
        }
        Ok(())
    }

    // the cells of every pool whose throughflow needs a surface gradient beyond the rest band
    fn pool_band(&self, pool: &[u32], inflow: &[i64], outflow: &[i64], dt: Fix128) -> Vec<usize> {
        let nx = self.bed.nx;
        let l = self.bed.cell;
        let cfg = self.config;
        let mut band = Vec::new();
        for cells in &pool_groups(pool) {
            let id = pool[cells[0]] as usize;
            let through = inflow[id].min(outflow[id]);
            if through <= 0 {
                continue;
            }
            let count = Fix128::from_int(cells.len() as i64);
            let (mut i0, mut i1, mut k0, mut k1) = (usize::MAX, 0, usize::MAX, 0);
            for &c in cells {
                let (i, k) = (c % nx, c / nx);
                i0 = i0.min(i);
                i1 = i1.max(i);
                k0 = k0.min(k);
                k1 = k1.max(k);
            }
            let span = Fix128::from_int(((i1 - i0).max(k1 - k0) + 1) as i64);
            let total: i64 = cells.iter().map(|c| self.depth[*c]).sum();
            let mean_h = quanta_to_metres(total) / count;
            // Q (m³/s) = through quanta × l² / Δt; ΔH = 3 ν Q L² / (g h̄³ A), L = span·l, A = count·l²
            let q = quanta_to_metres(through) * l * l / dt;
            let need = Fix128::from_int(3) * cfg.viscosity * q * span * span
                / (cfg.gravity * mean_h * mean_h * mean_h * count);
            let floor = quanta_to_metres(7) * span;
            let relative = cfg.pool_level_tolerance * mean_h;
            let allowed = relative.max(floor);
            if mean_h.is_zero() || need > allowed {
                band.extend_from_slice(cells);
            }
        }
        band
    }

    /// Cells levelled as a pool in the last substep, ascending.
    #[must_use]
    pub fn last_pooled_cells(&self) -> &[usize] {
        &self.last_pooled_cells
    }
    /// Cells of the pools that carried throughflow beyond the rest band in the substep that
    /// was refused as out of range (empty when the last advance succeeded).
    #[must_use]
    pub fn last_band_cells(&self) -> &[usize] {
        &self.last_band_cells
    }
    /// Lubrication faces held below their flux by a stability cap in the last substep (0 when
    /// the substep is within the diffusion number of the deepest film).
    #[must_use]
    pub fn last_cap_faces(&self) -> u32 {
        self.last_cap_faces
    }
    /// Local substeps taken inside the last substep by the faces that need a shorter step
    /// (1 when none did).
    #[must_use]
    pub fn last_local_substeps(&self) -> u32 {
        self.last_local_substeps
    }
    /// The substep this film takes (s): `min(max_substep, 3 ν l² / (8 g h_film³))`.
    #[must_use]
    pub fn substep(&self) -> Fix128 {
        self.substep
    }

    /// Advance by `h` seconds in [`Self::subcycles`] substeps; returns their number.
    ///
    /// # Errors
    ///
    /// [`ParticipantFault::OutOfRange`] for a non-positive `h`, too many substeps, or a depth
    /// leaving the integer range; the film is then unchanged.
    pub fn advance(&mut self, h: Fix128) -> Result<u32, ParticipantFault> {
        if h <= Fix128::ZERO {
            return Err(ParticipantFault::OutOfRange);
        }
        let n = self.subcycles(h).ok_or(ParticipantFault::OutOfRange)?;
        let dt = h.div_int(i64::from(n));
        self.last_band_cells.clear();
        let mut next = self.clone();
        // a deposit is spread over the advance it fed: each substep counts its share as a
        // pool's inflow, so the throughflow is the deposit's rate
        next.fed_share = n;
        for _ in 0..n {
            if let Err(e) = next.step_once(dt) {
                // keep the report of a refused substep (the band), not its state
                self.last_band_cells = core::mem::take(&mut next.last_band_cells);
                return Err(e);
            }
        }
        next.last_subcycles = n;
        next.fed.iter_mut().for_each(|d| *d = 0);
        next.fed_share = 1;
        *self = next;
        Ok(n)
    }

    /// One substep of width `dt`: rain, the transfers across every face, evaporation.
    ///
    /// # Errors
    ///
    /// [`ParticipantFault::OutOfRange`] when a depth leaves the integer range (the film may
    /// then be partly stepped; [`Self::advance`] discards it).
    pub fn step_once(&mut self, dt: Fix128) -> Result<(), ParticipantFault> {
        let cfg = self.config;
        let ncell = self.depth.len();
        let mut rained = 0i64;
        // rain: the same whole quanta on every cell, the fraction carried
        if !cfg.rain.is_zero() {
            let total = self.rain_carry + metres_in_quanta(cfg.rain * dt);
            let whole = total.floor();
            self.rain_carry = total - whole;
            let k = whole.hi;
            if k > 0 {
                for d in &mut self.depth {
                    *d = d.checked_add(k).ok_or(ParticipantFault::OutOfRange)?;
                }
                self.ledger.rain += k * ncell as i64;
                rained = k;
            }
        }
        let faces = Arc::clone(&self.faces);
        let nf = faces.len();
        let face = |f: usize| -> (Option<usize>, Option<usize>) {
            let (a, b) = faces[f];
            (side_cell(a), side_cell(b))
        };
        // pools: faces whose depth over the higher bed exceeds the deepest film and whose
        // surface step is within the join step join their cells
        let h_film = metres_to_quanta(cfg.film_depth_max);
        let mut parent: Vec<u32> = (0..ncell as u32).collect();
        let mut joined = vec_of(ncell, false);
        for f in 0..nf {
            if let (Some(a), Some(b)) = face(f) {
                let hf = face_depth(
                    self.bed.heights[a],
                    self.depth[a],
                    self.bed.heights[b],
                    self.depth[b],
                );
                let (ea, eb) = (
                    self.bed.heights[a] + self.depth[a],
                    self.bed.heights[b] + self.depth[b],
                );
                if hf > h_film
                    && (ea - eb).abs()
                        <= join_step(cfg.pool_join_step, self.depth[a], self.depth[b])
                {
                    let (ra, rb) = (find(&mut parent, a as u32), find(&mut parent, b as u32));
                    if ra != rb {
                        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                        parent[hi as usize] = lo;
                    }
                    joined[a] = true;
                    joined[b] = true;
                }
            }
        }
        let mut pool = vec_of(ncell, NOT_POOLED);
        for c in 0..ncell {
            if joined[c] {
                pool[c] = find(&mut parent, c as u32);
            }
        }
        let k = FluxConstants::new(self.bed.cell, dt, cfg.gravity, cfg.viscosity)
            .ok_or(ParticipantFault::OutOfRange)?;
        // the throughflow of each pool at the state before levelling: water crossing its rim
        // (by the film flux), and the rain and deposits that fall inside it
        let mut pool_in: Vec<i64> = vec_of(ncell, 0);
        let mut pool_out: Vec<i64> = vec_of(ncell, 0);
        for f in 0..nf {
            let (a, b) = face(f);
            let (pa, pb) = (
                a.map_or(NOT_POOLED, |c| pool[c]),
                b.map_or(NOT_POOLED, |c| pool[c]),
            );
            if pa == pb {
                continue;
            }
            let (sa, sb) = match (self.side(a, b), self.side(b, a)) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            let t = simultaneous_transfer(sa, sb, k).ok_or(ParticipantFault::OutOfRange)?;
            let (from, to) = if t > 0 { (pa, pb) } else { (pb, pa) };
            if from != NOT_POOLED {
                pool_out[from as usize] += t.abs();
            }
            if to != NOT_POOLED {
                pool_in[to as usize] += t.abs();
            }
        }
        for c in 0..ncell {
            if pool[c] != NOT_POOLED {
                // deposits are never negative: the ceiling of fed / share in integers
                let share = i64::from(self.fed_share.max(1));
                pool_in[pool[c] as usize] += (self.fed[c] + share - 1) / share + rained;
            }
        }
        // the band: a pool carrying throughflow Q needs a lubrication gradient across its
        // length L, ΔH = 3 ν Q L² / (g h̄³ A); one that needs more than its tolerance is not
        // levelled, its cells step on their own at local substeps
        let mut free = vec_of(ncell, false);
        for c in self.pool_band(&pool, &pool_in, &pool_out, dt) {
            pool[c] = NOT_POOLED;
            free[c] = true;
        }
        self.level_pools(&pool)?;
        self.last_pooled_cells = (0..ncell).filter(|c| pool[*c] != NOT_POOLED).collect();
        let mut moved = 0u64;
        let mut edge_out = 0i64;
        let mut cap_faces = 0u32;
        // deep faces whose flux is inertial (a dam-break front, a submerged step): outside
        let mut steep: Vec<usize> = Vec::new();
        // faces stepped at local substeps, and how many each needs
        let mut local: Vec<(usize, FaceLaw)> = Vec::new();
        let mut need: u128 = 1;
        // every face from the same state; the moves are summed per cell, so the order the
        // faces are visited does not matter
        let mut delta = core::mem::take(&mut self.scratch);
        delta.clear();
        delta.resize(ncell, 0);
        for n in 0..nf {
            let f = if self.reverse_faces { nf - 1 - n } else { n };
            let (a, b) = face(f);
            if let (Some(a), Some(b)) = (a, b) {
                if pool[a] != NOT_POOLED && pool[a] == pool[b] {
                    continue;
                }
            }
            let (sa, sb) = match (self.side(a, b), self.side(b, a)) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            };
            if sa.1 == 0 && sb.1 == 0 {
                continue;
            }
            let hf = face_depth(sa.0, sa.1, sb.0, sb.1);
            let near_free = a.is_some_and(|c| free[c]) || b.is_some_and(|c| free[c]);
            if hf.unsigned_abs() > MAX_FACE_DEPTH {
                return Err(ParticipantFault::OutOfRange);
            }
            let h3 = u128::from(hf.unsigned_abs()).pow(3);
            if hf <= h_film && !near_free {
                let (t, capped) =
                    simultaneous_transfer_report(sa, sb, k).ok_or(ParticipantFault::OutOfRange)?;
                cap_faces += u32::from(capped);
                if t == 0 {
                    continue;
                }
                match a {
                    Some(c) => delta[c] -= t,
                    None => edge_out -= t,
                }
                match b {
                    Some(c) => delta[c] += t,
                    None => edge_out += t,
                }
                moved += t.unsigned_abs();
                continue;
            }
            if hf <= h_film {
                local.push((f, FaceLaw::Film));
                // local substeps a lubrication face needs: 8 h_f³ / D_l (its diffusion number)
                need = need.max(h3.saturating_mul(8).div_ceil(k.lubrication));
                continue;
            }
            // deeper than the film: a free overfall (the lower side's surface at or below the
            // crest, over a drop or an open edge) is a weir; otherwise the film flux holds
            // while lubrication is below critical (subcritical, viscous)
            let (hi, lo) = if sa.0 + sa.1 >= sb.0 + sb.1 {
                (sa, sb)
            } else {
                (sb, sa)
            };
            let crest = sa.0.max(sb.0);
            let open = a.is_none() || b.is_none();
            if lo.0 + lo.1 <= crest && (open || hi.0 > lo.0) {
                let e = u128::from((hi.0 + hi.1 - lo.0 - lo.1).unsigned_abs());
                let w = isqrt_u128(h3)
                    .checked_mul(k.weir)
                    .map_or(u128::MAX, |v| v >> 64);
                if e > 0 {
                    need = need.max(w.saturating_mul(8).div_ceil(e));
                }
                local.push((f, FaceLaw::Weir));
                continue;
            }
            let e = u128::from(((sa.0 + sa.1) - (sb.0 + sb.1)).unsigned_abs());
            let lub = h3.checked_mul(e).map_or(u128::MAX, |v| v / k.lubrication);
            let crit_below_lub = critical_below(hf.unsigned_abs(), h3, lub, k.critical);
            // a front onto a dry or lower floor of the same height (a dam break), or a step in
            // the bed taller than the film with the lower side submerged and the flux at
            // critical (a drowned sill): inertial flows, outside the model
            let front = lo.0 + lo.1 <= crest;
            let drowned_sill = (sa.0 - sb.0).abs() > h_film && crit_below_lub;
            if front || drowned_sill {
                steep.extend(a.into_iter().chain(b));
                continue;
            }
            // otherwise the film flux (lubrication, capped by critical) at enough local
            // substeps that it never meets the ⌊|Δη|/8⌋ cap
            local.push((f, FaceLaw::Film));
            if e > 0 {
                let moved = if crit_below_lub {
                    critical(h3, k.critical)
                } else {
                    lub
                };
                need = need.max(moved.saturating_mul(8).div_ceil(e));
            }
        }
        if !steep.is_empty() {
            steep.sort_unstable();
            steep.dedup();
            self.last_band_cells = steep;
            return Err(ParticipantFault::OutOfRange);
        }
        if need > u128::from(cfg.max_local_substeps) {
            let mut cells: Vec<usize> = local
                .iter()
                .flat_map(|(f, _)| {
                    let (a, b) = face(*f);
                    a.into_iter().chain(b)
                })
                .collect();
            cells.sort_unstable();
            cells.dedup();
            self.last_band_cells = cells;
            return Err(ParticipantFault::OutOfRange);
        }
        for (d, dd) in self.depth.iter_mut().zip(&delta) {
            *d += dd;
        }
        // local substeps: the faces that need a shorter step, `need` times at `Δt / need`,
        // each from the state the last one left; a pool they touch is levelled after each
        let count = need as u32;
        if !local.is_empty() {
            let kl = FluxConstants::new(
                self.bed.cell,
                dt.div_int(i64::from(count)),
                cfg.gravity,
                cfg.viscosity,
            )
            .ok_or(ParticipantFault::OutOfRange)?;
            let mut touches = vec_of(ncell, false);
            for (f, _) in &local {
                let (a, b) = face(*f);
                for c in a.into_iter().chain(b) {
                    if pool[c] != NOT_POOLED {
                        touches[pool[c] as usize] = true;
                    }
                }
            }
            let touched: Vec<u32> = pool
                .iter()
                .map(|p| {
                    if *p != NOT_POOLED && touches[*p as usize] {
                        *p
                    } else {
                        NOT_POOLED
                    }
                })
                .collect();
            for _ in 0..count {
                delta.iter_mut().for_each(|d| *d = 0);
                for (f, law) in &local {
                    let (a, b) = face(*f);
                    let (sa, sb) = match (self.side(a, b), self.side(b, a)) {
                        (Some(x), Some(y)) => (x, y),
                        _ => continue,
                    };
                    let t = match law {
                        FaceLaw::Film => {
                            let (t, capped) = simultaneous_transfer_report(sa, sb, kl)
                                .ok_or(ParticipantFault::OutOfRange)?;
                            cap_faces += u32::from(capped);
                            t
                        }
                        FaceLaw::Weir => {
                            weir_transfer(sa, sb, kl).ok_or(ParticipantFault::OutOfRange)?
                        }
                    };
                    if t == 0 {
                        continue;
                    }
                    match a {
                        Some(c) => delta[c] -= t,
                        None => edge_out -= t,
                    }
                    match b {
                        Some(c) => delta[c] += t,
                        None => edge_out += t,
                    }
                    moved += t.unsigned_abs();
                }
                for (d, dd) in self.depth.iter_mut().zip(&delta) {
                    *d += dd;
                }
                self.level_pools(&touched)?;
            }
        }
        self.last_local_substeps = count;
        self.scratch = delta;
        self.ledger.outflow += edge_out;
        self.last_moved = moved;
        self.last_cap_faces = cap_faces;
        // evaporation: the same whole quanta off every cell that has them, never below zero
        if !cfg.evaporation.is_zero() {
            let total = self.evap_carry + metres_in_quanta(cfg.evaporation * dt);
            let whole = total.floor();
            self.evap_carry = total - whole;
            let k = whole.hi;
            if k > 0 {
                for d in &mut self.depth {
                    let take = (*d).min(k);
                    *d -= take;
                    self.ledger.evaporated += take;
                }
            }
        }
        for (d, age) in self.depth.iter().zip(self.dry_age.iter_mut()) {
            if *d > cfg.wet_threshold {
                *age = 0;
            } else if *age != u32::MAX {
                *age = age.saturating_add(1).min(u32::MAX - 1);
            }
        }
        self.steps += 1;
        Ok(())
    }

    fn state_len(&self) -> usize {
        HEADER_LEN + self.depth.len() * PER_CELL
    }
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(x)
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(x)
}
fn i64_at(b: &[u8], at: usize) -> i64 {
    u64_at(b, at) as i64
}
fn fix_at(b: &[u8], at: usize) -> Fix128 {
    Fix128::from_raw(i64_at(b, at), u64_at(b, at + 8))
}

impl Participant for WaterFilm {
    fn kind(&self) -> ParticipantKind {
        WATER_FILM_PARTICIPANT_KIND
    }

    fn step_rule(&self) -> StepRule {
        StepRule::Subcycle
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        self.advance(h).map(|_| ())
    }

    fn observe(&self, out: &mut ObservationSink) {
        let area = self.bed.cell * self.bed.cell;
        out.push(WATER_FILM_OBS_VOLUME, quanta_to_metres(self.total()) * area);
        out.push(
            WATER_FILM_OBS_SUBCYCLES,
            Fix128::from_int(i64::from(self.last_subcycles)),
        );
        let wet = self.dry_age.iter().filter(|a| **a == 0).count();
        out.push(WATER_FILM_OBS_WET_CELLS, Fix128::from_int(wet as i64));
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&self.bed.generation.to_le_bytes());
        out.extend_from_slice(&self.steps.to_le_bytes());
        put_fix(out, self.rain_carry);
        put_fix(out, self.evap_carry);
        for v in [
            self.ledger.initial,
            self.ledger.rain,
            self.ledger.deposits,
            self.ledger.evaporated,
            self.ledger.outflow,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.last_subcycles.to_le_bytes());
        out.extend_from_slice(&self.last_moved.to_le_bytes());
        for c in 0..self.depth.len() {
            out.extend_from_slice(&self.depth[c].to_le_bytes());
            out.extend_from_slice(&self.dry_age[c].to_le_bytes());
            out.extend_from_slice(&self.fed[c].to_le_bytes());
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let expected = self.state_len();
        if bytes.len() != expected {
            return Err(StateError::Length {
                expected,
                found: bytes.len(),
            });
        }
        // the digest covers the bed and its generation
        if u32_at(bytes, 0) != STATE_VERSION || u64_at(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        if (0..self.depth.len()).any(|c| {
            i64_at(bytes, HEADER_LEN + c * PER_CELL) < 0
                || i64_at(bytes, HEADER_LEN + c * PER_CELL + 12) < 0
        }) {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.steps = u64_at(bytes, 20);
        self.rain_carry = fix_at(bytes, 28);
        self.evap_carry = fix_at(bytes, 44);
        self.ledger = FilmLedger {
            initial: i64_at(bytes, 60),
            rain: i64_at(bytes, 68),
            deposits: i64_at(bytes, 76),
            evaporated: i64_at(bytes, 84),
            outflow: i64_at(bytes, 92),
        };
        self.last_subcycles = u32_at(bytes, 100);
        self.last_moved = u64_at(bytes, 104);
        let mut at = HEADER_LEN;
        for c in 0..self.depth.len() {
            self.depth[c] = i64_at(bytes, at);
            self.dry_age[c] = u32_at(bytes, at + 8);
            self.fed[c] = i64_at(bytes, at + 12);
            at += PER_CELL;
        }
    }
}

// The film's oracles compile as lib tests too, so the film's lines count in the `--lib`
// coverage (the same file runs as the integration test `analytic_water_film`).
#[cfg(test)]
#[path = "../tests/analytic_water_film.rs"]
mod analytic_water_film_oracle;
