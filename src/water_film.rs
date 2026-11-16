//! # MODEL
//!
//! - **Equations**: a thin water layer of depth `h ≥ 0` over a fixed bed `b` on a square grid
//!   of cells of side `l`, gravity along −y. Mass: `∂h/∂t + ∇·q = r − e`, with `r` the rain
//!   rate and `e` the evaporation rate (both m/s) ([`WaterFilm::step_once`]). The flux `q`
//!   (m²/s, per unit width) across a face is the diffusion-wave flux driven by the difference
//!   of the free-surface elevation `η = b + h`: `q = sign(Δη)·min(g h_f³ |Δη| / (3ν l),
//!   h_f √(g h_f))`, the lubrication (viscous, Nusselt) flux capped by the critical flux
//!   ([`diffusion_wave_flux`]). `h_f = max(η_L, η_R) − max(b_L, b_R)` (zero or above) is the
//!   depth over the higher bed of the two cells ([`face_depth`]), so no water crosses a face
//!   while the surface stands below the higher bed: a basin holds water until its surface
//!   passes the rim. On a steady incline of angle `α` the flux is the Nusselt film flux
//!   `q = g h³ sinα / (3ν)`.
//! - **Discretisation**: every substep moves water across every face at once, each face read
//!   from the state at the start of the substep: `t = min(⌊|q|·Δt/l⌋, ⌊|Δη|/8⌋, ⌊h_f/4⌋)` with
//!   the sign of `Δη` ([`simultaneous_transfer`]), the moves summed per cell, so the result does
//!   not depend on the order the faces are visited and no cell gives more than it holds (four
//!   faces of at most `h_f/4`, `h_f ≤ h`). The potential `P = Σ (η² − b²)` falls with every move:
//!   with `δ` the change per cell, `ΔP = −2 Σ_f t_f Δη_f + Σ_c δ_c²`; `t ≤ Δη/8` gives
//!   `Σ t Δη ≥ 8 Σ t²` and four faces per cell give `Σ δ² ≤ 8 Σ t²`, so `ΔP ≤ −8 Σ t² < 0` when
//!   anything moves; the state is integer, so without rain the film reaches rest (every
//!   transfer zero) in finitely many substeps. On a uniform steady film every face reads the
//!   same state, so the scheme carries the Nusselt flux exactly at any `Δt` (measured: depth
//!   error below 10⁻⁴ at `Δt` = 1/256, 1/1024 and 1/4096 s; an earlier variant that moved
//!   disjoint face sets in turn was first order, depth error ≈ −5.2·uΔt/l).
//! - **Units**: SI. Lengths and depths m, time s, `q` m²/s, `g` m/s², `ν` m²/s, rates m/s
//!   (rain given in mm/h converts with [`FilmConfig::rain_mm_per_hour`]). The depth is held
//!   as an integer number of quanta of `2⁻³²` m ([`DEPTH_QUANTUM_BITS`],
//!   [`metres_to_quanta`]); fluxes and rates are [`Fix128`].
//! - **Conservation**: volume is conserved exactly, as integers: a transfer leaves one cell
//!   and enters the other ([`WaterFilm::step_once`]); the ledger `Σ depth = initial + rain +
//!   deposits − evaporation − outflow over open edges` holds to the last quantum
//!   ([`FilmLedger::expected`]). Momentum is not modelled (no inertia).
//! - **Range**: cell side `l` from 1 mm to 1 m ([`FilmBed::from_heights`]). The substep is
//!   `Δt = h / ⌈h / Δt_max⌉` with `Δt_max` declared in [`FilmConfig::max_substep`]
//!   ([`WaterFilm::subcycles`]): a function of the step and the configuration only, never of
//!   the water. The film flux is resolved while `u Δt ≤ l/4` (u the film speed, the `⌊h_f/4⌋`
//!   cap); above it flow is slower than the model (at `l` = 1 cm and `Δt` = 1/256 s: films up
//!   to 0.64 m/s, which a 1 mm film reaches only beyond a slope of about 11°).
//!   **Deep pools (`h ≳ 1` mm, where the critical cap binds) level at a relaxation rate set
//!   by `Δt`, not at the gravity-wave speed**: their steady states (a level surface, overflow
//!   at the rim) are physical, their levelling time is not
//!   (`a_closed_groove_levels_to_the_closed_form_surface` reports it against `L/√(g h)`).
//!   More than [`FilmConfig::max_subcycles`] substeps, or a depth leaving the integer range,
//!   is [`ParticipantFault::OutOfRange`] and changes nothing ([`WaterFilm::advance`]).
//! - **Approximations**: 2.5D, one depth per column, the bed a height field sampled once per
//!   cell centre ([`FilmBed::from_height_fn`], [`FilmBed::from_sdf`]); gravity along −y only;
//!   no inertia, surface tension, contact angle or infiltration ([`diffusion_wave_flux`]);
//!   rain uniform over the grid and evaporation at a fixed rate on wet cells
//!   ([`WaterFilm::step_once`]); the bed is fixed while the film steps, a new one is set with
//!   [`WaterFilm::set_bed`], which needs a larger generation. Rest is reached when every face
//!   moves less than one quantum: in the pool regime two neighbouring wet cells then differ by
//!   at most 7 quanta (`⌊|Δη|/8⌋ = 0`), so a pool of `L` cells along its longest path is level
//!   to within `7(L − 1)` quanta, and a thin film on a slope stops once its flux moves less
//!   than a quantum per substep. The band cannot be one quantum for any rule of this kind
//!   (local, independent of face order, integer, potential strictly falling): a staircase of
//!   unit steps `η, η + 1, η + 2, …` is at rest under every rule that moves nothing across
//!   `|Δη| ≤ 1`, and a rule that moves a quantum across `|Δη| = 1` only swaps the pair, which
//!   leaves the potential unchanged, so it never comes to rest.
//!   Measured on a closed 8-cell groove: spread 48 quanta at rest (11 nm, bound 49).
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/analytic_water_film.rs::a_closed_groove_levels_to_the_closed_form_surface::volume_is_conserved_to_the_last_quantum::a_groove_overflows_at_its_rim_height::a_groove_fills_in_the_closed_form_time::rewind_and_rerun_are_bit_identical::face_order_does_not_change_the_state::the_potential_never_rises_without_a_source::a_steady_film_on_an_incline_has_the_nusselt_depth::a_wet_cell_dries_in_the_closed_form_time::an_sdf_bed_and_a_height_bed_agree::the_substep_count_depends_on_declared_inputs_only`
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
use alloc::vec::Vec;

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

/// The diffusion-wave flux: `sign(Δη)·min(g h_f³ |Δη| / (3ν l), h_f √(g h_f))`.
#[must_use]
pub fn diffusion_wave_flux(
    h_face: Fix128,
    d_eta: Fix128,
    cell: Fix128,
    g: Fix128,
    nu: Fix128,
) -> Fix128 {
    if h_face.is_zero() || d_eta.is_zero() {
        return Fix128::ZERO;
    }
    let lub = g * h_face * h_face * h_face * d_eta.abs() / (Fix128::from_int(3) * nu * cell);
    let crit = h_face * (g * h_face).sqrt();
    let m = if lub < crit { lub } else { crit };
    if d_eta.is_negative() {
        -m
    } else {
        m
    }
}

/// The water (quanta) a face moves in a substep when every face moves at once from the same
/// state: `min(⌊|q|·Δt/l⌋, ⌊|Δη|/8⌋, ⌊h_f/4⌋)` with the sign of `Δη`.
#[must_use]
pub fn simultaneous_transfer(
    first: (i64, i64),
    second: (i64, i64),
    cell: Fix128,
    dt: Fix128,
    g: Fix128,
    nu: Fix128,
) -> i64 {
    let d_eta = (first.0 + first.1) - (second.0 + second.1);
    let hf = face_depth(first.0, first.1, second.0, second.1);
    if d_eta == 0 || hf == 0 {
        return 0;
    }
    let q = diffusion_wave_flux(
        quanta_to_metres(hf),
        quanta_to_metres(d_eta.abs()),
        cell,
        g,
        nu,
    );
    let flux = metres_to_quanta(q * dt / cell);
    let t = flux.min(d_eta.abs() / 8).min(hf / 4);
    if d_eta > 0 {
        t
    } else {
        -t
    }
}

/// A thin water film over a [`FilmBed`]; a [`Participant`] of kind `FILM`.
///
/// # Snapshot payload
///
/// Little endian: `version: u32` (1), `digest: u64` (FNV-1a 64 over the configuration and the
/// bed), the bed generation `u64`, the substep count `u64`, the rain and evaporation carries
/// (two [`Fix128`] as `hi: i64, lo: u64`), the ledger (five `i64`), the last substep count
/// `u32`, the quanta moved in the last substep `u64`, then per cell the depth `i64` and the
/// substeps since it was last wet `u32`.
/// [`Participant::check_state`] refuses another length ([`StateError::Length`]), version or
/// digest (which covers the bed and its generation), or a negative depth
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
        };
        film.digest = film.config_digest();
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
        let need = (h / self.config.max_substep).ceil();
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

    // the faces of group `colour` (0: x faces, even low index; 1: x, odd; 2: z, even; 3: z,
    // odd) as (first cell, second cell); None = past the edge
    fn face_set(&self, colour: u64) -> Vec<(Option<usize>, Option<usize>)> {
        let (nx, nz) = (self.bed.nx, self.bed.nz);
        let parity = (colour % 2) as usize;
        let mut faces = Vec::new();
        if colour % 4 < 2 {
            for k in 0..nz {
                let mut i = parity;
                while i <= nx {
                    let a = if i > 0 { Some(i - 1 + k * nx) } else { None };
                    let b = if i < nx { Some(i + k * nx) } else { None };
                    faces.push((a, b));
                    i += 2;
                }
            }
        } else {
            for i in 0..nx {
                let mut k = parity;
                while k <= nz {
                    let a = if k > 0 { Some(i + (k - 1) * nx) } else { None };
                    let b = if k < nz { Some(i + k * nx) } else { None };
                    faces.push((a, b));
                    k += 2;
                }
            }
        }
        if self.reverse_faces {
            faces.reverse();
        }
        faces
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
        let mut next = self.clone();
        for _ in 0..n {
            next.step_once(dt)?;
        }
        next.last_subcycles = n;
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
            }
        }
        let mut moved = 0u64;
        let mut edge_out = 0i64;
        {
            // every face from the same state; the deltas are summed, so order does not matter
            let mut delta: Vec<i64> = vec_of(ncell, 0);
            let mut faces = Vec::new();
            for colour in 0..4 {
                faces.extend(self.face_set(colour));
            }
            for (a, b) in faces {
                let (sa, sb) = match (self.side(a, b), self.side(b, a)) {
                    (Some(x), Some(y)) => (x, y),
                    _ => continue,
                };
                let t =
                    simultaneous_transfer(sa, sb, self.bed.cell, dt, cfg.gravity, cfg.viscosity);
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
        }
        self.ledger.outflow += edge_out;
        self.last_moved = moved;
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
        HEADER_LEN + self.depth.len() * 12
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
        for (d, a) in self.depth.iter().zip(&self.dry_age) {
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&a.to_le_bytes());
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
        if (0..self.depth.len()).any(|c| i64_at(bytes, HEADER_LEN + c * 12) < 0) {
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
            at += 12;
        }
    }
}
