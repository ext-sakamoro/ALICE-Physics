//! # MODEL
//!
//! - **Equations**: hot, sooty air in a closed box of `nx × ny × nz` cubic cells of side `l`,
//!   gravity along −y. Velocity: incompressible Navier–Stokes in the Boussinesq form,
//!   `∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + β ΔT g ŷ`, `∇·u = 0`, solved by [`CfdSolver`] on its
//!   MAC grid (BFECC advection, explicit diffusion, multigrid projection, no-through-flow
//!   walls); the buoyancy is added to the y faces from the mean `ΔT` of the two cells
//!   sharing each face before the solver step ([`SmokePlume::step_once`]). Scalars: the
//!   temperature excess `ΔT = T − T₀ ≥ 0` and the soot density `s ≥ 0` obey
//!   `∂φ/∂t + ∇·(uφ) = κ_φ ∇²φ + S_φ` (molecular diffusivity `κ_φ`,
//!   [`SmokeConfig::heat_diffusivity_m2_s`], [`SmokeConfig::soot_diffusivity_m2_s`]), with
//!   sources `S` given per box of cells ([`SmokeSource`]):
//!   either declared rates, or a burner whose single-step Arrhenius reaction
//!   ([`crate::smoke_fire`]) heats at `r Δh_c / (ρ c_p)` and makes soot at `r y_s` from the
//!   cell's own temperature. The scalar transport is finite volume in integers
//!   ([`QuantaField::step`]): every face of a cell `d` that carries flow out of it moves
//!   `c·φ_f` with the Courant number `c = |u_f| Δt / l` and the face value
//!   `φ_f = q_d + ⌊(1 − c)/2 · minmod(q_d − q_back, q_recv − q_d)⌋` (MUSCL, minmod limited);
//!   the outflow of the cell, `E = Σ c φ_f` plus its carried fraction, moves as
//!   `M = ⌊E⌋` whole quanta, shared among the outgoing faces by largest remainder, and
//!   the fraction `E − M` stays with the cell for its next substep. Every move is read from
//!   the same state and summed per cell, so the result does not depend on the order the
//!   cells are visited ([`SmokePlume::set_reverse_cells`]). The transport takes
//!   `k = ⌈2 max_d Σ_out c⌉` substeps per velocity step, so `Σ_out c ≤ 1/2`: with the
//!   limiter `φ_f ≤ 3/2 q_d`, so `E ≤ 3/4 q_d + 1` and no cell gives more than it holds
//!   ([`TransportReport::overrun_cells`] counts any that would). Diffusion follows the
//!   advection in each velocity step ([`QuantaField::diffuse`]): each cell sends
//!   `d (q_d − q_n)` toward every neighbour holding less, `d = κ Δt_d / l²`, in whole quanta
//!   with its own carried fraction, so the explicit seven-point scheme moves exactly; it
//!   takes `⌈6 κ Δt / l²⌉` substeps, so `d ≤ 1/6` (the explicit limit in 3D): a cell then
//!   sends at most `6 d q_d ≤ q_d`. The scheme's second moment grows by exactly `2 d l²` per
//!   axis per substep away from the walls, which is `2 κ t`.
//! - **Units**: SI. Lengths m, time s, velocity m/s, `ΔT` K, soot kg/m³, `ρ` kg/m³,
//!   `ν = μ/ρ` with `μ` Pa·s, `β` 1/K, `c_p` J/(kg·K), `g` m/s². `ΔT` is held as integer
//!   quanta of `2⁻³²` K ([`HEAT_QUANTUM_BITS`]) and soot as quanta of `2⁻⁴⁰` kg/m³
//!   ([`SOOT_QUANTUM_BITS`]) per cell; velocities and rates are [`Fix128`].
//! - **Conservation**: heat (`Σ ΔT`, the cells being of equal volume) and soot mass are
//!   conserved exactly, as integers: a move leaves one cell and enters its neighbour, the
//!   walls pass nothing, and the ledger `Σ q = initial + sources` holds to the last quantum
//!   ([`SmokeLedger`]). The carried fractions are below one quantum per cell, so the
//!   transported amount differs from the exact flux sum by less than one quantum per cell.
//!   Momentum is the solver's (not conserved exactly).
//! - **Range**: cells from 1 mm to 1 m, each extent a power of two from 4 to 256 (the
//!   multigrid projection), the velocity step at most [`SmokeConfig::step_max`] with the
//!   viscous diffusion number `ν Δt / l² ≤ 1/6` checked at build (the scalar diffusion is
//!   substepped instead). No turbulence model, and the plume's similarity is open: at the
//!   air's own diffusivity a plume is about 5 mm wide, below what these grids resolve, so
//!   its width comes from the scheme's diffusion and narrows with the cell (measured
//!   without molecular diffusion at 10 and 5 mm cells over 0.15–0.35 m in a 0.64 m box:
//!   `d ln R²/d ln z` 1.03 → 0.48, `d ln w_c/d ln z` 0.27 → 0.22, `d ln ΔT_c/d ln z`
//!   −0.26 → −0.18, against the laminar 1, 0, −1); neither plume law is claimed yet. More
//!   than [`SmokeConfig::max_steps`] velocity steps in an advance, more than
//!   [`SmokeConfig::max_transport_substeps`] transport substeps, a quantum count leaving the
//!   integer range or a cell past [`MAX_CELL_QUANTA`], or a cell giving more than it holds is
//!   [`ParticipantFault::OutOfRange`] and changes nothing ([`SmokePlume::advance`]).
//! - **Approximations**: Boussinesq (density variations only in the buoyancy), no radiation,
//!   constant diffusivities, no wall heat loss, soot a passive tracer (no settling, no
//!   coagulation), a burner's fuel and oxidizer held at their declared densities (a supplied
//!   burner, not a consumed one), a closed box only.
//! - **Tiers**: L0 only.
//! - **Validation**: `tests/analytic_smoke_plume.rs::heat_and_soot_are_conserved_to_the_last_quantum::the_projection_leaves_a_bounded_divergence_and_closed_walls::rewind_and_rerun_are_bit_identical::cell_order_does_not_change_the_state::a_tampered_payload_is_refused::no_cell_goes_negative_at_the_largest_courant_number::a_cell_below_one_face_quantum_still_drains::the_soot_left_behind_a_passing_puff_is_bounded::degenerate_inputs_are_refused::the_plume_steps_inside_a_world::a_burner_heats_at_the_arrhenius_rate::an_isolated_cell_drains_at_the_upwind_rate::the_transport_substeps_keep_half_a_cell_per_substep::a_puff_spreads_with_variance_two_kappa_t::a_gradient_below_one_quantum_still_diffuses`
//!
//! Author: Moroya Sakamoto

use crate::cfd_solver::{AdvectionScheme, CfdSolver};
use crate::det_hash::DetHasher;
use crate::eulerian_grid::MacGrid;
use crate::math::{Fix128, Vec3Fix};
use crate::smoke_fire::{
    heat_release_j_per_m3_s, reaction_rate_kg_per_m3_s, soot_generation_kg_per_m3_s,
    ArrheniusReaction,
};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepRule,
    SubstepCtx,
};
use core::hash::Hasher;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Snapshot tag of [`SmokePlume`]: the ASCII code `SMOK`, big endian.
pub const SMOKE_PLUME_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"SMOK"));

/// Bits below the kelvin in a heat quantum: one quantum is `2⁻³²` K of temperature excess.
pub const HEAT_QUANTUM_BITS: u32 = 32;
/// Bits below 1 kg/m³ in a soot quantum: one quantum is `2⁻⁴⁰` kg/m³.
pub const SOOT_QUANTUM_BITS: u32 = 40;
/// Most quanta one cell may hold, `2⁶⁰` (2.7·10⁸ K of excess heat, 10⁶ kg/m³ of soot): every
/// intermediate of a substep then fits an `i64` (a face value is at most `3/2` of it, a cell
/// takes at most six neighbours' outflow on top of its own).
pub const MAX_CELL_QUANTA: i64 = 1 << 60;

/// Observation channel: heat held, `Σ ΔT · V` (K·m³).
pub const SMOKE_OBS_HEAT: u32 = 0;
/// Observation channel: soot held (kg).
pub const SMOKE_OBS_SOOT: u32 = 1;
/// Observation channel: largest temperature excess of any cell (K).
pub const SMOKE_OBS_MAX_EXCESS: u32 = 2;
/// Observation channel: transport substeps of the last velocity step.
pub const SMOKE_OBS_TRANSPORT_SUBSTEPS: u32 = 3;

const STATE_VERSION: u32 = 2;

/// Whole quanta of `2^-bits` as a value (exact).
#[must_use]
pub fn from_quanta(q: i64, bits: u32) -> Fix128 {
    let raw = i128::from(q) << (64 - bits);
    Fix128::from_raw((raw >> 64) as i64, raw as u64)
}

/// A value counted in quanta of `2^-bits` (exact, the fraction kept), or the fault past
/// the range.
fn in_quanta(v: Fix128, bits: u32) -> Result<Fix128, ParticipantFault> {
    let raw = (i128::from(v.hi) << 64) | i128::from(v.lo);
    let s = raw
        .checked_mul(1i128 << bits)
        .ok_or(ParticipantFault::OutOfRange)?;
    Ok(Fix128::from_raw((s >> 64) as i64, s as u64))
}

/// The minmod limiter: the smaller of two slopes of one sign, else 0.
#[must_use]
pub fn minmod(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 || (a > 0) != (b > 0) {
        0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

/// What one transport substep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransportReport {
    /// Quanta moved across faces.
    pub moved: u64,
    /// Cells whose outflow exceeded what they held (zero when the limiter and the substep
    /// rule hold; the outflow is then cut to the cell's content).
    pub overrun_cells: u32,
    /// Smallest cell content after the substep.
    pub min: i64,
    /// Largest cell content after the substep.
    pub max: i64,
}

/// A cell-centred scalar held as integer quanta and moved by face fluxes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantaField {
    nx: usize,
    ny: usize,
    nz: usize,
    q: Vec<i64>,
    carry: Vec<Fix128>,
    dcarry: Vec<Fix128>,
}

/// Direction of a cell's face: `(axis, +1 or −1)`, in the order −x, +x, −y, +y, −z, +z.
const DIRS: [(usize, isize); 6] = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)];

impl QuantaField {
    /// An empty field of `nx × ny × nz` cells.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            q: vec![0; n],
            carry: vec![Fix128::ZERO; n],
            dcarry: vec![Fix128::ZERO; n],
        }
    }

    /// Cell index of `(i, j, k)`.
    #[must_use]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Quanta per cell.
    #[must_use]
    pub fn quanta(&self) -> &[i64] {
        &self.q
    }

    /// Sum of all cells, or `None` past the integer range.
    #[must_use]
    pub fn total(&self) -> Option<i64> {
        self.q.iter().try_fold(0i64, |a, &q| a.checked_add(q))
    }

    /// Add `n` quanta to cell `c`, or `None` when the cell would leave `0..=MAX_CELL_QUANTA`
    /// ([`MAX_CELL_QUANTA`]).
    pub fn add(&mut self, c: usize, n: i64) -> Option<()> {
        let v = self.q[c].checked_add(n)?;
        if !(0..=MAX_CELL_QUANTA).contains(&v) {
            return None;
        }
        self.q[c] = v;
        Some(())
    }

    fn neighbour(&self, i: usize, j: usize, k: usize, axis: usize, s: isize) -> Option<usize> {
        let (n, p) = match axis {
            0 => (self.nx, i),
            1 => (self.ny, j),
            _ => (self.nz, k),
        };
        let p2 = p as isize + s;
        if p2 < 0 || p2 >= n as isize {
            return None;
        }
        let p2 = p2 as usize;
        Some(match axis {
            0 => self.index(p2, j, k),
            1 => self.index(i, p2, k),
            _ => self.index(i, j, p2),
        })
    }

    /// Velocity on the face of cell `(i, j, k)` toward `(axis, s)`; boundary faces carry
    /// nothing (closed box).
    fn face_velocity(
        grid: &MacGrid,
        i: usize,
        j: usize,
        k: usize,
        axis: usize,
        s: isize,
    ) -> Fix128 {
        let (nx, ny) = (grid.nx, grid.ny);
        let up = usize::from(s > 0);
        match axis {
            0 => grid.u[i + up + (nx + 1) * (j + ny * k)],
            1 => grid.v[i + nx * (j + up + (ny + 1) * k)],
            _ => grid.w[i + nx * (j + ny * (k + up))],
        }
    }

    /// Largest summed outgoing Courant number of any cell, `max_d Σ_out |u_f| Δt / l`.
    #[must_use]
    pub fn max_out_courant(grid: &MacGrid, dt: Fix128) -> Fix128 {
        let kdt = dt / grid.dx;
        let mut m = Fix128::ZERO;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let mut out = Fix128::ZERO;
                    for &(axis, s) in &DIRS {
                        let u = Self::face_velocity(grid, i, j, k, axis, s);
                        if (s > 0 && u > Fix128::ZERO) || (s < 0 && u < Fix128::ZERO) {
                            out = out + u.abs();
                        }
                    }
                    let c = out * kdt;
                    if c > m {
                        m = c;
                    }
                }
            }
        }
        m
    }

    /// One transport substep of width `dt` through the face velocities of `grid` (the grid
    /// must have this field's extents). `reverse` visits the cells in reverse order (the
    /// result is the same). The caller keeps `Σ_out c ≤ 1/2` ([`Self::max_out_courant`]).
    pub fn step(&mut self, grid: &MacGrid, dt: Fix128, reverse: bool) -> TransportReport {
        let kdt = dt / grid.dx;
        let half = Fix128::from_ratio(1, 2);
        let q = self.q.clone();
        let mut carry = self.carry.clone();
        let report = self.exchange(&mut carry, reverse, |f, c, i, j, k, faces| {
            let qd = q[c];
            let mut nf = 0usize;
            for &(axis, s) in &DIRS {
                let u = Self::face_velocity(grid, i, j, k, axis, s);
                let outward = (s > 0 && u > Fix128::ZERO) || (s < 0 && u < Fix128::ZERO);
                if !outward {
                    continue;
                }
                let Some(recv) = f.neighbour(i, j, k, axis, s) else {
                    continue;
                };
                let qr = q[recv];
                let qb = f.neighbour(i, j, k, axis, -s).map_or(qd, |b| q[b]);
                let ca = u.abs() * kdt;
                let slope = minmod(qd - qb, qr - qd);
                let phi = qd
                    + ((Fix128::ONE - ca) * half * Fix128::from_int(slope))
                        .floor()
                        .hi;
                faces[nf] = (recv, ca * Fix128::from_int(phi.max(0)));
                nf += 1;
            }
            nf
        });
        self.carry = carry;
        report
    }

    /// One diffusion substep with diffusion number `d = κ Δt / l²`: each cell sends
    /// `d (q_d − q_n)` toward every neighbour `n` holding less, across the interior faces
    /// (the walls pass nothing), with its own carried fraction. `d ≤ 1/6` keeps the sum a
    /// cell sends within what it holds ([`SmokePlume::step_once`] takes `⌈6 d⌉` substeps).
    pub fn diffuse(&mut self, d: Fix128, reverse: bool) -> TransportReport {
        let q = self.q.clone();
        let mut carry = self.dcarry.clone();
        let report = self.exchange(&mut carry, reverse, |f, c, i, j, k, faces| {
            let qd = q[c];
            let mut nf = 0usize;
            for &(axis, s) in &DIRS {
                let Some(n) = f.neighbour(i, j, k, axis, s) else {
                    continue;
                };
                if q[n] < qd {
                    faces[nf] = (n, d * Fix128::from_int(qd - q[n]));
                    nf += 1;
                }
            }
            nf
        });
        self.dcarry = carry;
        report
    }

    /// Move quanta out of every cell along the faces `outflows` names, all read from the
    /// same state: a cell's exact outflow plus its carried fraction moves as whole quanta,
    /// shared among its faces by largest remainder (ties to the earlier direction), and the
    /// fraction left stays in `carry`.
    fn exchange<F>(
        &mut self,
        carry: &mut [Fix128],
        reverse: bool,
        mut outflows: F,
    ) -> TransportReport
    where
        F: FnMut(&Self, usize, usize, usize, usize, &mut [(usize, Fix128); 6]) -> usize,
    {
        let n = self.q.len();
        let mut delta = vec![0i64; n];
        let mut report = TransportReport::default();
        for t in 0..n {
            let c = if reverse { n - 1 - t } else { t };
            let i = c % self.nx;
            let j = (c / self.nx) % self.ny;
            let k = c / (self.nx * self.ny);
            let qd = self.q[c];
            let mut faces: [(usize, Fix128); 6] = [(0, Fix128::ZERO); 6];
            let nf = outflows(self, c, i, j, k, &mut faces);
            let e = faces[..nf].iter().fold(Fix128::ZERO, |a, f| a + f.1);
            let total = e + carry[c];
            let whole = total.floor().hi;
            let m = if whole > qd {
                report.overrun_cells += 1;
                carry[c] = Fix128::ZERO;
                qd.max(0)
            } else {
                carry[c] = total - Fix128::from_int(whole);
                whole
            };
            if m <= 0 || nf == 0 || e.is_zero() {
                continue;
            }
            let mf = Fix128::from_int(m);
            let mut part = [0i64; 6];
            let mut frac = [Fix128::ZERO; 6];
            let mut given = 0i64;
            for f in 0..nf {
                let share = faces[f].1 / e * mf;
                part[f] = share.floor().hi;
                frac[f] = share - Fix128::from_int(part[f]);
                given += part[f];
            }
            let mut rest = m - given;
            while rest > 0 {
                let mut best = 0usize;
                for f in 1..nf {
                    if frac[f] > frac[best] {
                        best = f;
                    }
                }
                part[best] += 1;
                frac[best] = Fix128::ZERO - Fix128::ONE;
                rest -= 1;
            }
            while rest < 0 {
                let mut best = 0usize;
                for f in 1..nf {
                    if part[f] > part[best] {
                        best = f;
                    }
                }
                part[best] -= 1;
                rest += 1;
            }
            for f in 0..nf {
                delta[c] -= part[f];
                delta[faces[f].0] += part[f];
            }
            report.moved += m as u64;
        }
        let (mut min, mut max) = (i64::MAX, i64::MIN);
        for (q, d) in self.q.iter_mut().zip(delta) {
            *q += d;
            min = min.min(*q);
            max = max.max(*q);
        }
        report.min = min;
        report.max = max;
        report
    }
}

/// The air and the stepping limits. Build it from [`SmokeConfig::air`] and set the fields that
/// differ (the struct is `#[non_exhaustive]`, so a field added later does not break a caller).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SmokeConfig {
    /// Air density `ρ` (kg/m³).
    pub density_kg_m3: Fix128,
    /// Dynamic viscosity `μ` (Pa·s).
    pub viscosity_pas: Fix128,
    /// Thermal expansion coefficient `β` (1/K).
    pub beta_per_k: Fix128,
    /// Ambient temperature `T₀` (K).
    pub ambient_k: Fix128,
    /// Specific heat `c_p` (J/(kg·K)), for burner heat.
    pub heat_capacity_j_kg_k: Fix128,
    /// Gravity magnitude `g` (m/s²), along −y.
    pub gravity_m_s2: Fix128,
    /// Molecular diffusivity of heat `κ` (m²/s).
    pub heat_diffusivity_m2_s: Fix128,
    /// Diffusivity of soot (m²/s).
    pub soot_diffusivity_m2_s: Fix128,
    /// Longest velocity step (s).
    pub step_max: Fix128,
    /// Multigrid W-cycles per projection.
    pub multigrid_cycles: u32,
    /// Most velocity steps in one advance.
    pub max_steps: u32,
    /// Most transport substeps per velocity step.
    pub max_transport_substeps: u32,
}

impl SmokeConfig {
    /// Air at 300 K: `ρ` 1.2 kg/m³, `μ` 1.8·10⁻⁵ Pa·s, `β` 1/300 K⁻¹, `c_p` 1005 J/(kg·K),
    /// thermal diffusivity 2.2·10⁻⁵ m²/s, `g` 9.80665 m/s², steps of at most 1/120 s,
    /// 6 W-cycles. Soot diffusivity 0 (the Brownian diffusivity of soot particles, about
    /// 10⁻⁹ m²/s, is far below what a grid of millimetres resolves).
    #[must_use]
    pub fn air() -> Self {
        Self {
            density_kg_m3: Fix128::from_ratio(12, 10),
            viscosity_pas: Fix128::from_ratio(18, 1_000_000),
            beta_per_k: Fix128::from_ratio(1, 300),
            ambient_k: Fix128::from_int(300),
            heat_capacity_j_kg_k: Fix128::from_int(1005),
            gravity_m_s2: Fix128::from_ratio(980_665, 100_000),
            heat_diffusivity_m2_s: Fix128::from_ratio(22, 1_000_000),
            soot_diffusivity_m2_s: Fix128::ZERO,
            step_max: Fix128::from_ratio(1, 120),
            multigrid_cycles: 6,
            max_steps: 1024,
            max_transport_substeps: 64,
        }
    }

    fn hash_into(&self, h: &mut DetHasher) {
        for f in [
            self.density_kg_m3,
            self.viscosity_pas,
            self.beta_per_k,
            self.ambient_k,
            self.heat_capacity_j_kg_k,
            self.gravity_m_s2,
            self.heat_diffusivity_m2_s,
            self.soot_diffusivity_m2_s,
            self.step_max,
        ] {
            h.write(&f.hi.to_le_bytes());
            h.write(&f.lo.to_le_bytes());
        }
        for v in [
            self.multigrid_cycles,
            self.max_steps,
            self.max_transport_substeps,
        ] {
            h.write(&v.to_le_bytes());
        }
    }
}

/// A box of cells `lo..hi` (half open) along each axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellBox {
    /// First cell along x, y, z.
    pub lo: [usize; 3],
    /// One past the last cell along x, y, z.
    pub hi: [usize; 3],
}

/// What a source does in its cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// Declared rates: heating (K/s) and soot (kg/(m³·s)) in each cell.
    Rates {
        /// Temperature excess added per second (K/s).
        heat_k_per_s: Fix128,
        /// Soot added per second (kg/(m³·s)).
        soot_kg_m3_s: Fix128,
    },
    /// A supplied burner: the reaction at the cell's temperature with fuel and oxidizer
    /// held at the declared densities (kg/m³).
    Burner {
        /// The single-step reaction.
        reaction: ArrheniusReaction,
        /// Fuel density (kg/m³).
        fuel_kg_m3: Fix128,
        /// Oxidizer density (kg/m³).
        oxidizer_kg_m3: Fix128,
    },
}

/// Heat and soot added in a box of cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmokeSource {
    /// Where.
    pub cells: CellBox,
    /// What.
    pub kind: SourceKind,
}

/// The rates a source gives a cell at temperature excess `excess`: (K/s, kg/(m³·s)).
#[must_use]
pub fn source_rates(kind: &SourceKind, config: &SmokeConfig, excess: Fix128) -> (Fix128, Fix128) {
    match *kind {
        SourceKind::Rates {
            heat_k_per_s,
            soot_kg_m3_s,
        } => (heat_k_per_s, soot_kg_m3_s),
        SourceKind::Burner {
            reaction,
            fuel_kg_m3,
            oxidizer_kg_m3,
        } => {
            let r = reaction_rate_kg_per_m3_s(
                &reaction,
                config.ambient_k + excess,
                fuel_kg_m3,
                oxidizer_kg_m3,
            );
            let heat = heat_release_j_per_m3_s(&reaction, r)
                / (config.density_kg_m3 * config.heat_capacity_j_kg_k);
            (heat, soot_generation_kg_per_m3_s(&reaction, r))
        }
    }
}

/// Why a plume could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmokeBuildError {
    /// An extent is not a power of two from 4 to 256, or the cell is outside 1 mm–1 m.
    BadGrid,
    /// A config value is out of its range (non-positive, or the diffusion number past 1/6).
    BadConfig,
    /// A source box is empty or leaves the grid, or a rate is negative.
    BadSource,
    /// A cell index is outside the grid, or an initial amount is negative.
    CellOutOfRange,
}

/// The quanta the plume has been given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SmokeLedger {
    /// Heat quanta set at the start.
    pub heat_initial: i64,
    /// Heat quanta added by the sources.
    pub heat_added: i64,
    /// Soot quanta set at the start.
    pub soot_initial: i64,
    /// Soot quanta added by the sources.
    pub soot_added: i64,
}

impl SmokeLedger {
    /// Heat quanta the cells must hold.
    #[must_use]
    pub fn heat_expected(&self) -> i64 {
        self.heat_initial + self.heat_added
    }
    /// Soot quanta the cells must hold.
    #[must_use]
    pub fn soot_expected(&self) -> i64 {
        self.soot_initial + self.soot_added
    }
}

/// Buoyant smoke in a closed box as a world participant (see the module MODEL).
#[derive(Clone, Debug)]
pub struct SmokePlume {
    config: SmokeConfig,
    solver: CfdSolver,
    heat: QuantaField,
    soot: QuantaField,
    sources: Vec<SmokeSource>,
    // per source cell, in source order then x-fastest: heat and soot carried fractions
    source_carry: Vec<(Fix128, Fix128)>,
    ledger: SmokeLedger,
    digest: u64,
    steps: u64,
    reverse: bool,
    last_steps: u32,
    last_transport_substeps: u32,
    last_diffusion_substeps: u32,
    last_moved: u64,
}

fn valid_extent(n: usize) -> bool {
    (4..=256).contains(&n) && n.is_power_of_two()
}

impl SmokePlume {
    /// A still, ambient box of `nx × ny × nz` cells of side `dx`.
    ///
    /// # Errors
    ///
    /// [`SmokeBuildError::BadGrid`] or [`SmokeBuildError::BadConfig`].
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: Fix128,
        config: SmokeConfig,
    ) -> Result<Self, SmokeBuildError> {
        if !(valid_extent(nx) && valid_extent(ny) && valid_extent(nz))
            || dx < Fix128::from_ratio(1, 1000)
            || dx > Fix128::ONE
        {
            return Err(SmokeBuildError::BadGrid);
        }
        let positive = [
            config.density_kg_m3,
            config.beta_per_k,
            config.ambient_k,
            config.heat_capacity_j_kg_k,
            config.gravity_m_s2,
            config.step_max,
        ];
        if positive.iter().any(|v| *v <= Fix128::ZERO)
            || config.viscosity_pas < Fix128::ZERO
            || config.heat_diffusivity_m2_s < Fix128::ZERO
            || config.soot_diffusivity_m2_s < Fix128::ZERO
            || config.step_max > Fix128::from_ratio(1, 10)
            || config.multigrid_cycles == 0
            || config.max_steps == 0
            || config.max_transport_substeps == 0
            || config.viscosity_pas / config.density_kg_m3 * config.step_max / (dx * dx)
                > Fix128::from_ratio(1, 6)
        {
            return Err(SmokeBuildError::BadConfig);
        }
        let mut solver = CfdSolver::new(nx, ny, nz, dx);
        solver.density_kg_m3 = config.density_kg_m3;
        solver.dynamic_viscosity_pas = config.viscosity_pas;
        solver.beta_per_k = config.beta_per_k;
        solver.reference_temp_k = config.ambient_k;
        // the hydrostatic part is the pressure's in a closed box; buoyancy is added here
        solver.gravity = Vec3Fix::ZERO;
        solver.temperature = None;
        solver.advection_scheme = AdvectionScheme::Bfecc;
        solver.grid.set_closed_box_walls();
        let mut plume = Self {
            config,
            solver,
            heat: QuantaField::new(nx, ny, nz),
            soot: QuantaField::new(nx, ny, nz),
            sources: Vec::new(),
            source_carry: Vec::new(),
            ledger: SmokeLedger::default(),
            digest: 0,
            steps: 0,
            reverse: false,
            last_steps: 0,
            last_transport_substeps: 0,
            last_diffusion_substeps: 0,
            last_moved: 0,
        };
        plume.digest = plume.config_digest();
        Ok(plume)
    }

    fn config_digest(&self) -> u64 {
        let mut h = DetHasher::default();
        self.config.hash_into(&mut h);
        let g = &self.solver.grid;
        for n in [g.nx, g.ny, g.nz] {
            h.write(&(n as u64).to_le_bytes());
        }
        h.write(&g.dx.hi.to_le_bytes());
        h.write(&g.dx.lo.to_le_bytes());
        for s in &self.sources {
            for v in s.cells.lo.iter().chain(s.cells.hi.iter()) {
                h.write(&(*v as u64).to_le_bytes());
            }
            let fs: [Fix128; 5] = match s.kind {
                SourceKind::Rates {
                    heat_k_per_s,
                    soot_kg_m3_s,
                } => [
                    heat_k_per_s,
                    soot_kg_m3_s,
                    Fix128::ZERO,
                    Fix128::ZERO,
                    Fix128::ZERO,
                ],
                SourceKind::Burner {
                    reaction,
                    fuel_kg_m3,
                    oxidizer_kg_m3,
                } => {
                    h.write(&reaction.pre_exponential_a.hi.to_le_bytes());
                    h.write(&reaction.pre_exponential_a.lo.to_le_bytes());
                    h.write(&reaction.activation_energy_j_per_mol.hi.to_le_bytes());
                    h.write(&reaction.activation_energy_j_per_mol.lo.to_le_bytes());
                    [
                        fuel_kg_m3,
                        oxidizer_kg_m3,
                        reaction.gas_constant_j_per_mol_k,
                        reaction.heat_of_combustion_j_per_kg,
                        reaction.soot_yield,
                    ]
                }
            };
            for f in fs {
                h.write(&f.hi.to_le_bytes());
                h.write(&f.lo.to_le_bytes());
            }
        }
        h.finish()
    }

    /// Add a source.
    ///
    /// # Errors
    ///
    /// [`SmokeBuildError::BadSource`] for an empty box, a box leaving the grid or a negative
    /// declared rate or density.
    pub fn add_source(&mut self, source: SmokeSource) -> Result<(), SmokeBuildError> {
        let ext = [self.heat.nx, self.heat.ny, self.heat.nz];
        let b = source.cells;
        if (0..3).any(|a| b.lo[a] >= b.hi[a] || b.hi[a] > ext[a]) {
            return Err(SmokeBuildError::BadSource);
        }
        let negative = match source.kind {
            SourceKind::Rates {
                heat_k_per_s,
                soot_kg_m3_s,
            } => heat_k_per_s < Fix128::ZERO || soot_kg_m3_s < Fix128::ZERO,
            SourceKind::Burner {
                reaction,
                fuel_kg_m3,
                oxidizer_kg_m3,
            } => {
                fuel_kg_m3 < Fix128::ZERO
                    || oxidizer_kg_m3 < Fix128::ZERO
                    || reaction.heat_of_combustion_j_per_kg < Fix128::ZERO
                    || reaction.soot_yield < Fix128::ZERO
            }
        };
        if negative {
            return Err(SmokeBuildError::BadSource);
        }
        let cells = (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
        self.sources.push(source);
        self.source_carry
            .extend(core::iter::repeat_n((Fix128::ZERO, Fix128::ZERO), cells));
        self.digest = self.config_digest();
        Ok(())
    }

    /// Set the initial heat (quanta of temperature excess) and soot (quanta) of cell
    /// `(i, j, k)`, counted in the ledger's initial amounts.
    ///
    /// # Errors
    ///
    /// [`SmokeBuildError::CellOutOfRange`] for a cell outside the grid, an amount outside
    /// `0..=MAX_CELL_QUANTA`, or a ledger total past the integer range; nothing changes then.
    pub fn set_initial(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        heat: i64,
        soot: i64,
    ) -> Result<(), SmokeBuildError> {
        let range = 0..=MAX_CELL_QUANTA;
        if i >= self.heat.nx
            || j >= self.heat.ny
            || k >= self.heat.nz
            || !range.contains(&heat)
            || !range.contains(&soot)
        {
            return Err(SmokeBuildError::CellOutOfRange);
        }
        let c = self.heat.index(i, j, k);
        let heat_initial = (self.ledger.heat_initial - self.heat.q[c])
            .checked_add(heat)
            .ok_or(SmokeBuildError::CellOutOfRange)?;
        let soot_initial = (self.ledger.soot_initial - self.soot.q[c])
            .checked_add(soot)
            .ok_or(SmokeBuildError::CellOutOfRange)?;
        self.ledger.heat_initial = heat_initial;
        self.ledger.soot_initial = soot_initial;
        self.heat.q[c] = heat;
        self.soot.q[c] = soot;
        Ok(())
    }

    /// Visit the cells in reverse order during transport (the state does not depend on it).
    pub fn set_reverse_cells(&mut self, reverse: bool) {
        self.reverse = reverse;
    }

    /// The config.
    #[must_use]
    pub fn config(&self) -> &SmokeConfig {
        &self.config
    }
    /// The velocity grid.
    #[must_use]
    pub fn velocity(&self) -> &MacGrid {
        &self.solver.grid
    }
    /// Heat quanta per cell (temperature excess in `2⁻³²` K).
    #[must_use]
    pub fn heat(&self) -> &QuantaField {
        &self.heat
    }
    /// Soot quanta per cell (`2⁻⁴⁰` kg/m³).
    #[must_use]
    pub fn soot(&self) -> &QuantaField {
        &self.soot
    }
    /// What the plume has been given.
    #[must_use]
    pub fn ledger(&self) -> SmokeLedger {
        self.ledger
    }
    /// Velocity steps taken.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }
    /// Velocity steps of the last advance.
    #[must_use]
    pub fn last_steps(&self) -> u32 {
        self.last_steps
    }
    /// Transport substeps of the last velocity step.
    #[must_use]
    pub fn last_transport_substeps(&self) -> u32 {
        self.last_transport_substeps
    }
    /// Diffusion substeps of the last velocity step (the larger of heat's and soot's).
    #[must_use]
    pub fn last_diffusion_substeps(&self) -> u32 {
        self.last_diffusion_substeps
    }
    /// Quanta moved (heat and soot) in the last velocity step.
    #[must_use]
    pub fn last_moved(&self) -> u64 {
        self.last_moved
    }
    /// Temperature excess of cell `(i, j, k)` (K).
    #[must_use]
    pub fn excess_k(&self, i: usize, j: usize, k: usize) -> Fix128 {
        from_quanta(self.heat.q[self.heat.index(i, j, k)], HEAT_QUANTUM_BITS)
    }

    /// Velocity steps for an advance of `h`: `⌈h / step_max⌉`, or `None` past the limit.
    #[must_use]
    pub fn step_count(&self, h: Fix128) -> Option<u32> {
        let n = (h / self.config.step_max).ceil().hi;
        if n < 1 || n > i64::from(self.config.max_steps) {
            return None;
        }
        Some(n as u32)
    }

    /// Advance by `h` in `⌈h / step_max⌉` equal velocity steps. Returns the step count.
    ///
    /// # Errors
    ///
    /// [`ParticipantFault::OutOfRange`] (see the MODEL); the plume is then unchanged.
    pub fn advance(&mut self, h: Fix128) -> Result<u32, ParticipantFault> {
        if h <= Fix128::ZERO {
            return Err(ParticipantFault::OutOfRange);
        }
        let n = self.step_count(h).ok_or(ParticipantFault::OutOfRange)?;
        let dt = h.div_int(i64::from(n));
        let mut next = self.clone();
        for _ in 0..n {
            next.step_once(dt)?;
        }
        next.last_steps = n;
        *self = next;
        Ok(n)
    }

    /// One velocity step of `dt`: sources, buoyancy, the solver step, then the transport
    /// substeps of heat and soot.
    ///
    /// # Errors
    ///
    /// [`ParticipantFault::OutOfRange`]; the plume may then be partly stepped
    /// ([`Self::advance`] discards it).
    pub fn step_once(&mut self, dt: Fix128) -> Result<(), ParticipantFault> {
        self.apply_sources(dt)?;
        self.apply_buoyancy(dt);
        self.solver.step_multigrid(dt, self.config.multigrid_cycles);
        let c = QuantaField::max_out_courant(&self.solver.grid, dt);
        let sub = (c * Fix128::from_int(2)).ceil().hi.max(1);
        if sub > i64::from(self.config.max_transport_substeps) {
            return Err(ParticipantFault::OutOfRange);
        }
        let sdt = dt.div_int(sub);
        let mut moved = 0u64;
        for _ in 0..sub {
            for field in [&mut self.heat, &mut self.soot] {
                let r = field.step(&self.solver.grid, sdt, self.reverse);
                if r.overrun_cells > 0 || r.min < 0 || r.max > MAX_CELL_QUANTA {
                    return Err(ParticipantFault::OutOfRange);
                }
                moved += r.moved;
            }
        }
        // diffusion: ⌈6 d⌉ substeps keep each at d ≤ 1/6 (no cell sends more than it holds)
        let mut most = 0i64;
        let l2 = self.solver.grid.dx * self.solver.grid.dx;
        for (field, kappa) in [
            (&mut self.heat, self.config.heat_diffusivity_m2_s),
            (&mut self.soot, self.config.soot_diffusivity_m2_s),
        ] {
            if kappa.is_zero() {
                continue;
            }
            let d = kappa * dt / l2;
            let kd = (d * Fix128::from_int(6)).ceil().hi.max(1);
            if kd > i64::from(self.config.max_transport_substeps) {
                return Err(ParticipantFault::OutOfRange);
            }
            let dsub = d.div_int(kd);
            for _ in 0..kd {
                let r = field.diffuse(dsub, self.reverse);
                if r.overrun_cells > 0 || r.min < 0 || r.max > MAX_CELL_QUANTA {
                    return Err(ParticipantFault::OutOfRange);
                }
                moved += r.moved;
            }
            most = most.max(kd);
        }
        self.last_transport_substeps = sub as u32;
        self.last_diffusion_substeps = most as u32;
        self.last_moved = moved;
        self.steps += 1;
        Ok(())
    }

    fn apply_sources(&mut self, dt: Fix128) -> Result<(), ParticipantFault> {
        let mut at = 0usize;
        for s in &self.sources {
            let b = s.cells;
            for k in b.lo[2]..b.hi[2] {
                for j in b.lo[1]..b.hi[1] {
                    for i in b.lo[0]..b.hi[0] {
                        let c = self.heat.index(i, j, k);
                        let excess = from_quanta(self.heat.q[c], HEAT_QUANTUM_BITS);
                        let (hr, sr) = source_rates(&s.kind, &self.config, excess);
                        let (hc, sc) = self.source_carry[at];
                        let th = hc + in_quanta(hr * dt, HEAT_QUANTUM_BITS)?;
                        let ts = sc + in_quanta(sr * dt, SOOT_QUANTUM_BITS)?;
                        let (wh, ws) = (th.floor(), ts.floor());
                        self.source_carry[at] = (th - wh, ts - ws);
                        self.heat
                            .add(c, wh.hi)
                            .ok_or(ParticipantFault::OutOfRange)?;
                        self.soot
                            .add(c, ws.hi)
                            .ok_or(ParticipantFault::OutOfRange)?;
                        self.ledger.heat_added = self
                            .ledger
                            .heat_added
                            .checked_add(wh.hi)
                            .ok_or(ParticipantFault::OutOfRange)?;
                        self.ledger.soot_added = self
                            .ledger
                            .soot_added
                            .checked_add(ws.hi)
                            .ok_or(ParticipantFault::OutOfRange)?;
                        at += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_buoyancy(&mut self, dt: Fix128) {
        let (nx, ny, nz) = (self.heat.nx, self.heat.ny, self.heat.nz);
        let kick = self.config.beta_per_k * self.config.gravity_m_s2 * dt;
        let half = Fix128::from_ratio(1, 2);
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    let below = self.heat.q[self.heat.index(i, j - 1, k)];
                    let above = self.heat.q[self.heat.index(i, j, k)];
                    if below == 0 && above == 0 {
                        continue;
                    }
                    let face = (from_quanta(below, HEAT_QUANTUM_BITS)
                        + from_quanta(above, HEAT_QUANTUM_BITS))
                        * half;
                    let ix = i + nx * (j + (ny + 1) * k);
                    self.solver.grid.v[ix] = self.solver.grid.v[ix] + face * kick;
                }
            }
        }
    }

    fn cells(&self) -> usize {
        self.heat.q.len()
    }
    fn faces(&self) -> usize {
        let g = &self.solver.grid;
        g.u.len() + g.v.len() + g.w.len() + g.pressure.len()
    }
    fn state_len(&self) -> usize {
        HEADER_LEN + self.cells() * PER_CELL + self.faces() * 16 + self.source_carry.len() * 32
    }
}

// version, digest, steps, solver steps, ledger (4), last steps, last substeps, last moved,
// last diffusion substeps
const HEADER_LEN: usize = 4 + 8 + 8 + 8 + 8 * 4 + 4 + 4 + 8 + 4;
// per cell: heat quanta, heat carry, heat diffusion carry, soot quanta, soot carry, soot
// diffusion carry
const PER_CELL: usize = 8 + 16 + 16 + 8 + 16 + 16;

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
fn is_fraction(f: Fix128) -> bool {
    f.hi == 0
}

impl Participant for SmokePlume {
    fn kind(&self) -> ParticipantKind {
        SMOKE_PLUME_PARTICIPANT_KIND
    }

    fn step_rule(&self) -> StepRule {
        StepRule::Subcycle
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        self.advance(h).map(|_| ())
    }

    fn observe(&self, out: &mut ObservationSink) {
        let dx = self.solver.grid.dx;
        let v = dx * dx * dx;
        let heat = self
            .heat
            .q
            .iter()
            .fold(Fix128::ZERO, |a, &q| a + from_quanta(q, HEAT_QUANTUM_BITS));
        let soot = self
            .soot
            .q
            .iter()
            .fold(Fix128::ZERO, |a, &q| a + from_quanta(q, SOOT_QUANTUM_BITS));
        let max = self.heat.q.iter().copied().max().unwrap_or(0);
        out.push(SMOKE_OBS_HEAT, heat * v);
        out.push(SMOKE_OBS_SOOT, soot * v);
        out.push(SMOKE_OBS_MAX_EXCESS, from_quanta(max, HEAT_QUANTUM_BITS));
        out.push(
            SMOKE_OBS_TRANSPORT_SUBSTEPS,
            Fix128::from_int(i64::from(self.last_transport_substeps)),
        );
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&self.steps.to_le_bytes());
        out.extend_from_slice(&self.solver.step_count.to_le_bytes());
        for v in [
            self.ledger.heat_initial,
            self.ledger.heat_added,
            self.ledger.soot_initial,
            self.ledger.soot_added,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.last_steps.to_le_bytes());
        out.extend_from_slice(&self.last_transport_substeps.to_le_bytes());
        out.extend_from_slice(&self.last_moved.to_le_bytes());
        out.extend_from_slice(&self.last_diffusion_substeps.to_le_bytes());
        for c in 0..self.cells() {
            out.extend_from_slice(&self.heat.q[c].to_le_bytes());
            put_fix(out, self.heat.carry[c]);
            put_fix(out, self.heat.dcarry[c]);
            out.extend_from_slice(&self.soot.q[c].to_le_bytes());
            put_fix(out, self.soot.carry[c]);
            put_fix(out, self.soot.dcarry[c]);
        }
        let g = &self.solver.grid;
        for f in g.u.iter().chain(&g.v).chain(&g.w).chain(&g.pressure) {
            put_fix(out, *f);
        }
        for &(h, s) in &self.source_carry {
            put_fix(out, h);
            put_fix(out, s);
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
        if u32_at(bytes, 0) != STATE_VERSION || u64_at(bytes, 4) != self.digest {
            return Err(StateError::InvalidValue);
        }
        let bad_cell = (0..self.cells()).any(|c| {
            let at = HEADER_LEN + c * PER_CELL;
            let range = 0..=MAX_CELL_QUANTA;
            !range.contains(&i64_at(bytes, at))
                || !is_fraction(fix_at(bytes, at + 8))
                || !is_fraction(fix_at(bytes, at + 24))
                || !range.contains(&i64_at(bytes, at + 40))
                || !is_fraction(fix_at(bytes, at + 48))
                || !is_fraction(fix_at(bytes, at + 64))
        });
        let carries = HEADER_LEN + self.cells() * PER_CELL + self.faces() * 16;
        let bad_source =
            (0..self.source_carry.len() * 2).any(|n| !is_fraction(fix_at(bytes, carries + n * 16)));
        if bad_cell || bad_source {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.steps = u64_at(bytes, 12);
        self.solver.step_count = u64_at(bytes, 20);
        self.ledger = SmokeLedger {
            heat_initial: i64_at(bytes, 28),
            heat_added: i64_at(bytes, 36),
            soot_initial: i64_at(bytes, 44),
            soot_added: i64_at(bytes, 52),
        };
        self.last_steps = u32_at(bytes, 60);
        self.last_transport_substeps = u32_at(bytes, 64);
        self.last_moved = u64_at(bytes, 68);
        self.last_diffusion_substeps = u32_at(bytes, 76);
        let mut at = HEADER_LEN;
        for c in 0..self.cells() {
            self.heat.q[c] = i64_at(bytes, at);
            self.heat.carry[c] = fix_at(bytes, at + 8);
            self.heat.dcarry[c] = fix_at(bytes, at + 24);
            self.soot.q[c] = i64_at(bytes, at + 40);
            self.soot.carry[c] = fix_at(bytes, at + 48);
            self.soot.dcarry[c] = fix_at(bytes, at + 64);
            at += PER_CELL;
        }
        let g = &mut self.solver.grid;
        for f in
            g.u.iter_mut()
                .chain(g.v.iter_mut())
                .chain(g.w.iter_mut())
                .chain(g.pressure.iter_mut())
        {
            *f = fix_at(bytes, at);
            at += 16;
        }
        for sc in &mut self.source_carry {
            *sc = (fix_at(bytes, at), fix_at(bytes, at + 16));
            at += 32;
        }
    }
}

// The plume's oracles compile as lib tests too, so the plume's lines count in the `--lib`
// coverage (the same file runs as the integration test `analytic_smoke_plume`).
#[cfg(test)]
#[path = "../tests/analytic_smoke_plume.rs"]
mod analytic_smoke_plume_oracle;
