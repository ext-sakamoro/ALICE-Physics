//! One-dimensional compressible Euler equations, finite-volume method.
//!
//! The first stage of an explicit shock-capturing hydrocode: the conservation
//! laws
//!
//! ```text
//! ∂U/∂t + ∂F(U)/∂x = 0,   U = (ρ, ρu, E),   F = (ρu, ρu² + p, u(E + p))
//! E = p/(γ − 1) + ½ρu²     (ideal gas, constant γ)
//! ```
//!
//! are advanced on a uniform grid of `N` cells of width `Δx`:
//!
//! ```text
//! U_i^{n+1} = U_i^n − (Δt/Δx) (F_{i+½} − F_{i−½})
//! ```
//!
//! with the interface flux `F_{i+½}` from a Riemann solver.
//!
//! [`crate::compressible`] holds the closed forms (ideal-gas relations, the
//! normal-shock jump, Riemann invariants); this module is the solver that
//! marches a flow field in time. The two meet in the oracle
//! `tests/analytic_euler_fv.rs::stationary_shock_matches_normal_shock_jump`.
//!
//! # Units — non-dimensional (recommended), and why
//!
//! The solver has no unit system of its own: only `γ` and the ratios
//! `Δt/Δx` and `p/ρ` enter it. Give it **non-dimensional** variables:
//! `ρ/ρ₀`, `p/p₀`, `u/√(p₀/ρ₀)`, `x/L`, `t·√(p₀/ρ₀)/L` for a reference state
//! `(ρ₀, p₀)` and length `L`. The reason is the number format: [`Fix128`] is
//! Q64.64, whose resolution is the *absolute* constant `2⁻⁶⁴ ≈ 5.4e-20` with
//! no exponent, so a quantity keeps about `64 + log₂|x|` significant bits.
//! Scaled to order 1, every state variable keeps about 64 bits; in SI the same
//! flow mixes `p ≈ 10⁵ Pa` with `ρ ≈ 1 kg/m³` and a `Δt` of microseconds, and
// LIMITATION(COV-SHOCK-087): a strong rarefaction drives `p` towards 0, where an SI value would keep only the bits above `2⁻⁶⁴` (a pressure of `10⁻¹⁰ Pa` keeps ≈ 31)
//! a strong rarefaction drives `p` towards 0, where an SI value would keep
//! only the bits above `2⁻⁶⁴` (a pressure of `10⁻¹⁰ Pa` keeps ≈ 31). The range
//! (±9.2e18) is never the issue in this module; precision near zero is. This
//! is the same constraint that forces normalised units in
//! [`crate::maxwell_fdtd`]. Convert at the boundary: `p = p₀ p̂`, etc.
//!
//! # Numerical flux ([`RiemannSolver`])
//!
//! - [`RiemannSolver::Exact`] — Godunov's flux from the exact solution of the
//!   Riemann problem sampled at `x/t = 0`. Toro, *Riemann Solvers and
//!   Numerical Methods for Fluid Dynamics*, 3rd ed. (Springer, 2009), ch. 4:
//!   pressure function `f(p) = f_L(p) + f_R(p) + Δu` and the star velocity
//!   (§4.2, Proposition 4.1), Newton–Raphson iteration started from the
//!   adaptive PVRS / two-rarefaction / two-shock guess (§4.3), sampling of the
//!   complete wave pattern (§4.4–4.5), and the vacuum-generation condition
//!   `(2/(γ−1))(a_L + a_R) ≤ u_R − u_L` (§4.6); the structure follows the
//!   program of §4.9 (`GUESSP`, `PREFUN`, `STARPU`, `SAMPLE`). Godunov's
//!   method itself: Toro ch. 6.
//!   Differences from that program: a Newton iterate that would be negative is
//!   replaced by half the previous iterate (not by the constant `TOLPRE`), the
//!   stopping test is a relative step below `2⁻⁴⁶` (not `10⁻⁶`), and at most
//!   [`MAX_NEWTON_ITERATIONS`] iterations are run before
//!   [`EulerError::RiemannNotConverged`].
//!   Vacuum: when the condition above holds the solution contains a vacuum
//!   region that this solver does not represent, and
//!   [`EulerError::VacuumGenerated`] is returned.
//! - [`RiemannSolver::Hllc`] — the HLLC flux (Toro 3rd ed. §10.4: star
//!   speed `S*`, star states `U*_K` and `F*_K = F_K + S_K (U*_K − U_K)`) with
//!   the Einfeldt wave-speed bounds of Toro §10.5.1:
//!   `S_L = min(u_L − a_L, ũ − ã)`, `S_R = max(u_R + a_R, ũ + ã)` with Roe
//!   averages `ũ`, `H̃`, `ã² = (γ−1)(H̃ − ½ũ²)` (Einfeldt, *SIAM J. Numer.
//!   Anal.* 25 (1988) 294–318; with these bounds HLLC is positively
//!   conservative, Batten, Clarke, Lambert, Causon, *SIAM J. Sci. Comput.* 18
//!   (1997) 1553–1570). These bounds also keep `S_L < 0 < S_R` at a
//!   reflecting wall whatever the impact speed, which is what makes the wall
//!   flux carry exactly zero mass (the pressure-based estimate of Toro §10.5.2
//!   does not: for an impact Mach number above ≈ 1.85 it puts `S_L ≥ 0`).
//!   When `S* = 0` exactly the two star fluxes are averaged, so the flux is the
//!   same whichever side is called left.
//!
//! # Second order ([`Reconstruction`], [`TimeIntegrator`])
//!
//! MUSCL reconstruction in **primitive** variables `(ρ, u, p)` (van Leer,
//! *J. Comput. Phys.* 32 (1979) 101–136; Toro 3rd ed. ch. 13–14): in each
//! cell the slope `σ = φ(Δ₋, Δ₊)` of `Δ₋ = W_i − W_{i−1}`, `Δ₊ = W_{i+1} − W_i`
//! with [`Limiter::Minmod`] (`φ = minmod(Δ₋, Δ₊)`) or [`Limiter::VanLeer`]
//! (`φ = 2Δ₋Δ₊/(Δ₋ + Δ₊)` when `Δ₋Δ₊ > 0`, else 0), and interface states
//! `W_i ± σ/2`. Both limiters keep the reconstructed values between the cell
//! and its neighbour, so a positive `ρ` and `p` stay positive.
//!
//! [`TimeIntegrator::SspRk2`] is the two-stage strong-stability-preserving
//! Runge–Kutta method (Shu & Osher, *J. Comput. Phys.* 77 (1988) 439–471;
//! Gottlieb & Shu, *Math. Comp.* 67 (1998) 73–85):
//! `U¹ = Uⁿ + Δt L(Uⁿ)`, `Uⁿ⁺¹ = ½Uⁿ + ½(U¹ + Δt L(U¹))`. It is applied in the
//! algebraically identical form `Uⁿ⁺¹ = Uⁿ − Δ(½(G⁰ + G¹))` with
//! `G = (Δt/Δx) F` per interface, so that the update stays a flux difference
//! and the conservation below stays exact. MUSCL with
//! [`TimeIntegrator::ForwardEuler`] is allowed but is not a stable pairing for
//! all CFL numbers; use SSP-RK2 with MUSCL.
//!
//! # Time step and boundaries
//!
//! [`EulerFv1d::step`] takes `Δt = C_cfl Δx / max_i(|u_i| + a_i)` with
//! `0 < C_cfl ≤ 1` (Toro 3rd ed. §6.3.2). Boundaries are imposed through two
//! ghost cells on each side: [`Boundary::Transmissive`] (zero-order
//! extrapolation), [`Boundary::ReflectiveWall`] (mirror image with `u → −u`),
//! and [`Boundary::Periodic`] (both ends must be periodic).
//!
//! # Positivity
//!
// LIMITATION(COV-SHOCK-079): A stage that would leave `ρ ≤ 0` or `p ≤ 0` in a cell returns [`EulerError::NonPositiveDensity`] / [`EulerError::NonPositivePressure`] with the cell index, and the state is not changed
//! A stage that would leave `ρ ≤ 0` or `p ≤ 0` in a cell returns
//! [`EulerError::NonPositiveDensity`] / [`EulerError::NonPositivePressure`]
//! with the cell index, and the state is not changed (all stages are computed
//! into scratch storage and committed only when every cell is valid). The
//! same holds for [`EulerError::VacuumGenerated`].
//!
//! # Exactness and symmetry
//!
//! - **Conservation.** `Fix128` addition is an exact integer addition, and the
//!   update subtracts the same interface value from one cell that it adds to
//!   the next, so with periodic boundaries `Σρ`, `Σρu`, `ΣE` are conserved
//!   bit for bit every step. Between reflecting walls the wall flux has zero
//!   mass and energy components exactly, so `Σρ` and `ΣE` are conserved bit
//!   for bit too.
//! - **Mirror symmetry.** [`Fix128`] multiplication floors toward −∞, so
//!   `(−a)·b` is one raw unit below `−(a·b)` whenever the product has a
//!   remainder. Every product in this module goes through a sign-symmetric
//!   product (`|a|·|b|` with the sign reapplied) and every halving through a
//!   sign-symmetric shift; division is already symmetric. With that, the
//!   mirrored initial value (cell `i → N−1−i`, `u → −u`) produces the mirrored
//!   solution bit for bit.
//! - **Transcendentals.** `x^e` in the exact solver is `exp(e ln x)` with a
//!   range-reduced Taylor series for `exp` and an `atanh` series for `ln`,
//!   both accurate to about `10⁻¹⁷` and integer-only (deterministic on every
//!   target). [`Fix128::powf_pos`] is not used: it truncates the exponent to
//!   24 fractional bits (relative error ≈ `|ln x|·2⁻²⁴`), which would cap the
//!   star pressure at about `10⁻⁷`.

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Maximum Newton iterations of the exact Riemann solver.
pub const MAX_NEWTON_ITERATIONS: u32 = 64;

/// Relative Newton step below which the star pressure is accepted (`2⁻⁴⁶`).
const NEWTON_TOL: Fix128 = Fix128::from_raw(0, 1 << 18);

/// Floor of the initial guess (`2⁻²⁰ ≈ 9.5e-7`, the role of Toro's `TOLPRE`).
const GUESS_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 44);

// ============================================================================
// Errors
// ============================================================================

/// Errors of the Euler solver. Every error leaves the solver state unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EulerError {
    /// No cells (`N = 0`).
    NoCells,
    /// `γ ≤ 1`.
    GammaNotAboveOne,
    /// CFL number not in `(0, 1]`.
    CflOutOfRange,
    /// Cell width `Δx ≤ 0`.
    NonPositiveSpacing,
    /// Time step `Δt ≤ 0` (also returned when the maximum wave speed is so
    /// small that `C_cfl Δx / s_max` is not representable).
    NonPositiveTimeStep,
    /// Exactly one end is [`Boundary::Periodic`].
    PeriodicMismatch,
    /// `ρ ≤ 0` in the cell at `cell` (for a two-state call, 0 = left,
    /// 1 = right; for a single state, 0).
    NonPositiveDensity {
        /// Index of the offending cell / state.
        cell: usize,
    },
    /// `p ≤ 0` in the cell at `cell` (indices as for
    /// [`EulerError::NonPositiveDensity`]).
    NonPositivePressure {
        /// Index of the offending cell / state.
        cell: usize,
    },
    /// The Riemann problem generates vacuum (Toro 3rd ed. §4.6), which the
    /// exact solver does not represent.
    VacuumGenerated,
    /// Newton iteration for the star pressure did not converge within
    /// [`MAX_NEWTON_ITERATIONS`].
    RiemannNotConverged,
}

impl core::fmt::Display for EulerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoCells => write!(f, "euler_fv: no cells"),
            Self::GammaNotAboveOne => write!(f, "euler_fv: gamma must be > 1"),
            Self::CflOutOfRange => write!(f, "euler_fv: CFL number must be in (0, 1]"),
            Self::NonPositiveSpacing => write!(f, "euler_fv: cell width must be > 0"),
            Self::NonPositiveTimeStep => write!(f, "euler_fv: time step must be > 0"),
            Self::PeriodicMismatch => {
                write!(f, "euler_fv: periodic boundary must be used on both ends")
            }
            Self::NonPositiveDensity { cell } => {
                write!(f, "euler_fv: non-positive density in cell {cell}")
            }
            Self::NonPositivePressure { cell } => {
                write!(f, "euler_fv: non-positive pressure in cell {cell}")
            }
            Self::VacuumGenerated => write!(f, "euler_fv: Riemann problem generates vacuum"),
            Self::RiemannNotConverged => {
                write!(f, "euler_fv: star-pressure iteration did not converge")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EulerError {}

// ============================================================================
// Sign-symmetric arithmetic and transcendentals
// ============================================================================

/// `a·b` rounded toward zero: `|a|·|b|` with the sign reapplied, so that
/// `mul(−a, b) = −mul(a, b)` exactly (the `Fix128` operator floors).
#[inline]
fn mul(a: Fix128, b: Fix128) -> Fix128 {
    let r = a.abs() * b.abs();
    if a.is_negative() != b.is_negative() {
        -r
    } else {
        r
    }
}

/// `x/2` rounded toward zero (`half(−x) = −half(x)` exactly).
#[inline]
fn half(x: Fix128) -> Fix128 {
    if x.is_negative() {
        -((-x).half())
    } else {
        x.half()
    }
}

fn min(a: Fix128, b: Fix128) -> Fix128 {
    if a < b {
        a
    } else {
        b
    }
}

fn max(a: Fix128, b: Fix128) -> Fix128 {
    if a > b {
        a
    } else {
        b
    }
}

/// `ln 2`, exact to `2⁻⁶⁴` (same constant as [`Fix128::ln`]).
const LN2: Fix128 = Fix128::from_raw(0, 0xB172_17F7_D1CF_79AC);
/// `√2`, truncated to `2⁻⁶⁴`.
const SQRT2: Fix128 = Fix128::from_raw(1, 0x6A09_E667_F3BC_C908);

/// Natural log of `x > 0`: reduce to `m ∈ [1/√2, √2)`, then
/// `ln m = 2 atanh t`, `t = (m−1)/(m+1)`, `|t| ≤ 0.172`, 14 terms
/// (truncation `< 0.172²⁹/29 ≈ 2e-24`). `x ≤ 0` returns `ln 2⁻⁶⁴` (the log
/// of the smallest positive value) instead of looping in the range reduction.
fn ln_pos(x: Fix128) -> Fix128 {
    if x <= Fix128::ZERO {
        return -(Fix128::from_int(64) * LN2);
    }
    let mut m = x;
    let mut k: i64 = 0;
    while m >= SQRT2 {
        m = m.half();
        k += 1;
    }
    let lower = SQRT2.half();
    while m < lower {
        m = m.double();
        k -= 1;
    }
    let t = (m - Fix128::ONE) / (m + Fix128::ONE);
    let t2 = mul(t, t);
    let mut term = t;
    let mut sum = Fix128::ZERO;
    for n in 0..14i64 {
        sum = sum + mul(term, Fix128::from_ratio(1, 2 * n + 1));
        term = mul(term, t2);
    }
    Fix128::from_int(k) * LN2 + sum.double()
}

/// `eˣ`: `x = k ln 2 + r`, `|r| ≤ ½ ln 2`, Taylor series of `eʳ` to the 20th
/// term (truncation `< 0.35²¹/21! ≈ 5e-30`), then `2ᵏ` by shifts. Saturates
/// above `x ≈ 43` and returns 0 below the resolution.
fn exp_fix(x: Fix128) -> Fix128 {
    // 1/ln 2 = 1.442 695 040 888 963 4
    const INV_LN2: Fix128 = Fix128::from_raw(1, 8_166_282_121_979_092_992);
    let q = mul(x, INV_LN2) + Fix128::ONE.half();
    let k = q.floor().hi;
    if k > 62 {
        return Fix128::from_raw(i64::MAX, u64::MAX);
    }
    if k < -65 {
        return Fix128::ZERO;
    }
    let r = x - Fix128::from_int(k) * LN2;
    let mut sum = Fix128::ONE;
    let mut term = Fix128::ONE;
    for n in 1..=20i64 {
        term = mul(mul(term, r), Fix128::from_ratio(1, n));
        sum = sum + term;
    }
    if k >= 0 {
        for _ in 0..k {
            sum = sum.double();
        }
        sum
    } else {
        sum.shr_bits((-k) as u32)
    }
}

/// `xᵉ` for `x > 0` (any sign of `e`), as `exp(e ln x)`.
fn pow_pos(x: Fix128, e: Fix128) -> Fix128 {
    exp_fix(mul(e, ln_pos(x)))
}

// ============================================================================
// States
// ============================================================================

/// Primitive state `(ρ, u, p)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitive {
    /// Density `ρ`.
    pub rho: Fix128,
    /// Velocity `u`.
    pub u: Fix128,
    /// Pressure `p`.
    pub p: Fix128,
}

/// Conserved state `(ρ, ρu, E)` (also used for fluxes, component by component).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Conserved {
    /// Density `ρ` (mass flux `ρu` for a flux).
    pub rho: Fix128,
    /// Momentum `ρu` (momentum flux `ρu² + p` for a flux).
    pub mom: Fix128,
    /// Total energy `E = p/(γ−1) + ½ρu²` (energy flux `u(E + p)` for a flux).
    pub energy: Fix128,
}

impl Conserved {
    fn add(self, o: Self) -> Self {
        Self {
            rho: self.rho + o.rho,
            mom: self.mom + o.mom,
            energy: self.energy + o.energy,
        }
    }

    fn sub(self, o: Self) -> Self {
        Self {
            rho: self.rho - o.rho,
            mom: self.mom - o.mom,
            energy: self.energy - o.energy,
        }
    }

    fn scale(self, s: Fix128) -> Self {
        Self {
            rho: mul(s, self.rho),
            mom: mul(s, self.mom),
            energy: mul(s, self.energy),
        }
    }

    fn halved(self) -> Self {
        Self {
            rho: half(self.rho),
            mom: half(self.mom),
            energy: half(self.energy),
        }
    }

    /// Conserved variables of a primitive state (no validation).
    #[must_use]
    pub fn from_primitive(gamma: Fix128, w: &Primitive) -> Self {
        let mom = mul(w.rho, w.u);
        Self {
            rho: w.rho,
            mom,
            energy: w.p / (gamma - Fix128::ONE) + half(mul(mom, w.u)),
        }
    }

    /// Primitive variables: `u = ρu/ρ`, `p = (γ−1)(E − ½ ρu·u)`.
    ///
    /// # Errors
    ///
    /// [`EulerError::NonPositiveDensity`] / [`EulerError::NonPositivePressure`]
    /// with `cell: 0`.
    pub fn to_primitive(&self, gamma: Fix128) -> Result<Primitive, EulerError> {
        to_primitive_at(gamma, self, 0)
    }
}

fn to_primitive_at(gamma: Fix128, c: &Conserved, cell: usize) -> Result<Primitive, EulerError> {
    if c.rho <= Fix128::ZERO {
        return Err(EulerError::NonPositiveDensity { cell });
    }
    let u = c.mom / c.rho;
    let p = mul(gamma - Fix128::ONE, c.energy - half(mul(c.mom, u)));
    if p <= Fix128::ZERO {
        return Err(EulerError::NonPositivePressure { cell });
    }
    Ok(Primitive { rho: c.rho, u, p })
}

fn check_state(w: &Primitive, cell: usize) -> Result<(), EulerError> {
    if w.rho <= Fix128::ZERO {
        return Err(EulerError::NonPositiveDensity { cell });
    }
    if w.p <= Fix128::ZERO {
        return Err(EulerError::NonPositivePressure { cell });
    }
    Ok(())
}

fn check_gamma(gamma: Fix128) -> Result<(), EulerError> {
    if gamma <= Fix128::ONE {
        Err(EulerError::GammaNotAboveOne)
    } else {
        Ok(())
    }
}

/// Physical flux `F(W) = (ρu, ρu² + p, u(E + p))`.
#[must_use]
pub fn physical_flux(gamma: Fix128, w: &Primitive) -> Conserved {
    let c = Conserved::from_primitive(gamma, w);
    Conserved {
        rho: c.mom,
        mom: mul(c.mom, w.u) + w.p,
        energy: mul(w.u, c.energy + w.p),
    }
}

// ============================================================================
// Exact Riemann solver (Toro 3rd ed. ch. 4)
// ============================================================================

/// γ-dependent constants `G1..G8` of Toro 3rd ed. §4.9.
#[derive(Clone, Copy, Debug)]
struct GammaConsts {
    g: Fix128,
    g1: Fix128,
    g2: Fix128,
    g3: Fix128,
    g4: Fix128,
    g5: Fix128,
    g6: Fix128,
    g7: Fix128,
    g8: Fix128,
    inv_g: Fix128,
}

impl GammaConsts {
    fn new(g: Fix128) -> Self {
        let one = Fix128::ONE;
        let two = Fix128::from_int(2);
        let gm1 = g - one;
        let gp1 = g + one;
        Self {
            g,
            g1: gm1 / g.double(),
            g2: gp1 / g.double(),
            g3: g.double() / gm1,
            g4: two / gm1,
            g5: two / gp1,
            g6: gm1 / gp1,
            g7: gm1.half(),
            g8: gm1,
            inv_g: one / g,
        }
    }

    fn sound(&self, w: &Primitive) -> Fix128 {
        (mul(self.g, w.p) / w.rho).sqrt()
    }
}

/// Star region of a Riemann problem: pressure `p*` and velocity `u*` between
/// the left and right waves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarState {
    /// Star pressure `p*`.
    pub p: Fix128,
    /// Star (contact) velocity `u*`.
    pub u: Fix128,
    /// Newton iterations used.
    pub iterations: u32,
}

/// `f_K(p)` and `f_K'(p)` (Toro 3rd ed. §4.2, Proposition 4.1; `PREFUN`).
fn prefun(c: &GammaConsts, p: Fix128, w: &Primitive, a: Fix128) -> (Fix128, Fix128) {
    if p <= w.p {
        // rarefaction
        let pr = p / w.p;
        let f = mul(mul(c.g4, a), pow_pos(pr, c.g1) - Fix128::ONE);
        let fd = pow_pos(pr, -c.g2) / mul(w.rho, a);
        (f, fd)
    } else {
        // shock
        let ak = c.g5 / w.rho;
        let bk = mul(c.g6, w.p);
        let q = (ak / (bk + p)).sqrt();
        let f = mul(p - w.p, q);
        let fd = mul(Fix128::ONE - half((p - w.p) / (bk + p)), q);
        (f, fd)
    }
}

/// Adaptive initial guess (Toro 3rd ed. §4.3; `GUESSP`): PVRS when the
/// pressure ratio is small and PVRS lies between the data, two-rarefaction
/// when PVRS is below both pressures, two-shock otherwise.
fn guess_pressure(c: &GammaConsts, l: &Primitive, r: &Primitive, al: Fix128, ar: Fix128) -> Fix128 {
    let cup = mul(mul(Fix128::ONE.half().half(), l.rho + r.rho), al + ar);
    let ppv = max(Fix128::ZERO, half(l.p + r.p) + mul(half(l.u - r.u), cup));
    let pmin = min(l.p, r.p);
    let pmax = max(l.p, r.p);
    let two = Fix128::from_int(2);
    if pmax / pmin <= two && pmin <= ppv && ppv <= pmax {
        ppv
    } else if ppv < pmin {
        // two-rarefaction approximation, written in its L/R-symmetric form
        // p_TR = [(a_L + a_R − ½(γ−1)(u_R − u_L)) / (a_L/p_L^z + a_R/p_R^z)]^{1/z},
        // z = (γ−1)/(2γ) (Toro 3rd ed. §4.3.1); `GUESSP` evaluates the same
        // value through an intermediate velocity, which is not symmetric in
        // the last bit
        let num = (al + ar) - mul(c.g7, r.u - l.u);
        if num <= Fix128::ZERO {
            return ppv;
        }
        let den = al / pow_pos(l.p, c.g1) + ar / pow_pos(r.p, c.g1);
        pow_pos(num / den, c.g3)
    } else {
        // two-shock approximation
        let gel = ((c.g5 / l.rho) / (mul(c.g6, l.p) + ppv)).sqrt();
        let ger = ((c.g5 / r.rho) / (mul(c.g6, r.p) + ppv)).sqrt();
        (mul(gel, l.p) + mul(ger, r.p) - (r.u - l.u)) / (gel + ger)
    }
}

fn check_pair(gamma: Fix128, l: &Primitive, r: &Primitive) -> Result<(), EulerError> {
    check_gamma(gamma)?;
    check_state(l, 0)?;
    check_state(r, 1)
}

/// Exact solution of the Riemann problem `left | right` for the star region.
///
/// # Errors
///
/// [`EulerError::GammaNotAboveOne`], [`EulerError::NonPositiveDensity`] /
/// [`EulerError::NonPositivePressure`] (`cell` 0 = left, 1 = right),
/// [`EulerError::VacuumGenerated`], [`EulerError::RiemannNotConverged`].
pub fn exact_riemann(
    gamma: Fix128,
    left: &Primitive,
    right: &Primitive,
) -> Result<StarState, EulerError> {
    check_pair(gamma, left, right)?;
    star_state(&GammaConsts::new(gamma), left, right)
}

fn star_state(c: &GammaConsts, l: &Primitive, r: &Primitive) -> Result<StarState, EulerError> {
    let al = c.sound(l);
    let ar = c.sound(r);
    let du = r.u - l.u;
    // Pressure positivity condition (Toro 3rd ed. §4.6).
    if mul(c.g4, al + ar) <= du {
        return Err(EulerError::VacuumGenerated);
    }
    let mut p = max(guess_pressure(c, l, r, al, ar), GUESS_FLOOR);
    let mut iterations = 0;
    loop {
        if iterations >= MAX_NEWTON_ITERATIONS {
            return Err(EulerError::RiemannNotConverged);
        }
        iterations += 1;
        let (fl, fld) = prefun(c, p, l, al);
        let (fr, frd) = prefun(c, p, r, ar);
        let mut pn = p - (fl + fr + du) / (fld + frd);
        if pn <= Fix128::ZERO {
            pn = p.half();
            if pn.is_zero() {
                // halved down to the resolution: no positive root reachable
                return Err(EulerError::RiemannNotConverged);
            }
        }
        let change = (pn - p).abs().double() / (pn + p);
        p = pn;
        if change <= NEWTON_TOL {
            break;
        }
    }
    let (fl, _) = prefun(c, p, l, al);
    let (fr, _) = prefun(c, p, r, ar);
    Ok(StarState {
        p,
        u: half((l.u + r.u) + (fr - fl)),
        iterations,
    })
}

impl StarState {
    /// The self-similar solution at speed `s = x/t` (Toro 3rd ed. §4.4–4.5;
    /// `SAMPLE`). `left` / `right` and `gamma` must be the ones `self` was
    /// solved from.
    #[must_use]
    pub fn sample(
        &self,
        gamma: Fix128,
        left: &Primitive,
        right: &Primitive,
        s: Fix128,
    ) -> Primitive {
        sample(&GammaConsts::new(gamma), left, right, self, s)
    }
}

fn sample(c: &GammaConsts, l: &Primitive, r: &Primitive, star: &StarState, s: Fix128) -> Primitive {
    let (pm, um) = (star.p, star.u);
    if s <= um {
        let al = c.sound(l);
        if pm <= l.p {
            // left rarefaction
            if s <= l.u - al {
                return *l;
            }
            let ratio = pm / l.p;
            let cml = mul(al, pow_pos(ratio, c.g1));
            if s > um - cml {
                return Primitive {
                    rho: mul(l.rho, pow_pos(ratio, c.inv_g)),
                    u: um,
                    p: pm,
                };
            }
            let cc = mul(c.g5, al + mul(c.g7, l.u - s));
            let ca = cc / al;
            Primitive {
                rho: mul(l.rho, pow_pos(ca, c.g4)),
                u: mul(c.g5, al + mul(c.g7, l.u) + s),
                p: mul(l.p, pow_pos(ca, c.g3)),
            }
        } else {
            // left shock
            let pml = pm / l.p;
            let sl = l.u - mul(al, (mul(c.g2, pml) + c.g1).sqrt());
            if s <= sl {
                return *l;
            }
            Primitive {
                rho: mul(l.rho, (pml + c.g6) / (mul(pml, c.g6) + Fix128::ONE)),
                u: um,
                p: pm,
            }
        }
    } else {
        let ar = c.sound(r);
        if pm > r.p {
            // right shock
            let pmr = pm / r.p;
            let sr = r.u + mul(ar, (mul(c.g2, pmr) + c.g1).sqrt());
            if s >= sr {
                return *r;
            }
            Primitive {
                rho: mul(r.rho, (pmr + c.g6) / (mul(pmr, c.g6) + Fix128::ONE)),
                u: um,
                p: pm,
            }
        } else {
            // right rarefaction
            if s >= r.u + ar {
                return *r;
            }
            let ratio = pm / r.p;
            let cmr = mul(ar, pow_pos(ratio, c.g1));
            if s < um + cmr {
                return Primitive {
                    rho: mul(r.rho, pow_pos(ratio, c.inv_g)),
                    u: um,
                    p: pm,
                };
            }
            let cc = mul(c.g5, ar - mul(c.g7, r.u - s));
            let ca = cc / ar;
            Primitive {
                rho: mul(r.rho, pow_pos(ca, c.g4)),
                u: mul(c.g5, -ar + mul(c.g7, r.u) + s),
                p: mul(r.p, pow_pos(ca, c.g3)),
            }
        }
    }
}

// ============================================================================
// Numerical fluxes
// ============================================================================

/// Riemann solver used for the interface flux.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiemannSolver {
    /// Godunov flux from the exact Riemann solution (Toro 3rd ed. ch. 4, 6).
    Exact,
    /// HLLC with Einfeldt wave speeds (Toro 3rd ed. §10.4, §10.5.1).
    Hllc,
}

/// Interface flux `F(left, right)` of the chosen Riemann solver.
///
/// # Errors
///
/// As [`exact_riemann`] for invalid input (`cell` 0 = left, 1 = right);
/// [`EulerError::VacuumGenerated`] / [`EulerError::RiemannNotConverged`] for
/// [`RiemannSolver::Exact`].
pub fn numerical_flux(
    solver: RiemannSolver,
    gamma: Fix128,
    left: &Primitive,
    right: &Primitive,
) -> Result<Conserved, EulerError> {
    check_pair(gamma, left, right)?;
    interface_flux(solver, &GammaConsts::new(gamma), left, right)
}

fn interface_flux(
    solver: RiemannSolver,
    c: &GammaConsts,
    l: &Primitive,
    r: &Primitive,
) -> Result<Conserved, EulerError> {
    match solver {
        RiemannSolver::Exact => {
            let star = star_state(c, l, r)?;
            let w = sample(c, l, r, &star, Fix128::ZERO);
            Ok(physical_flux(c.g, &w))
        }
        RiemannSolver::Hllc => Ok(hllc_flux(c, l, r)),
    }
}

fn hllc_flux(c: &GammaConsts, l: &Primitive, r: &Primitive) -> Conserved {
    let al = c.sound(l);
    let ar = c.sound(r);
    let ul = Conserved::from_primitive(c.g, l);
    let ur = Conserved::from_primitive(c.g, r);
    // Roe averages (Einfeldt wave-speed bounds, Toro 3rd ed. §10.5.1)
    let sql = l.rho.sqrt();
    let sqr = r.rho.sqrt();
    let den = sql + sqr;
    let ut = (mul(sql, l.u) + mul(sqr, r.u)) / den;
    let hl = (ul.energy + l.p) / l.rho;
    let hr = (ur.energy + r.p) / r.rho;
    let ht = (mul(sql, hl) + mul(sqr, hr)) / den;
    let at = max(Fix128::ZERO, mul(c.g8, ht - half(mul(ut, ut)))).sqrt();
    let sl = min(l.u - al, ut - at);
    let sr = max(r.u + ar, ut + at);
    if Fix128::ZERO <= sl {
        return physical_flux(c.g, l);
    }
    if Fix128::ZERO >= sr {
        return physical_flux(c.g, r);
    }
    // Star speed and star fluxes (Toro 3rd ed. §10.4)
    let dl = mul(l.rho, sl - l.u);
    let dr = mul(r.rho, sr - r.u);
    let s_star = ((r.p - l.p) + mul(l.u, dl) - mul(r.u, dr)) / (dl - dr);
    let star_flux = |w: &Primitive, u: &Conserved, s: Fix128, d: Fix128| -> Conserved {
        let k = d / (s - s_star);
        let e = u.energy / w.rho + mul(s_star - w.u, s_star + w.p / d);
        let u_star = Conserved {
            rho: k,
            mom: mul(k, s_star),
            energy: mul(k, e),
        };
        physical_flux(c.g, w).add(u_star.sub(*u).scale(s))
    };
    if s_star > Fix128::ZERO {
        star_flux(l, &ul, sl, dl)
    } else if s_star < Fix128::ZERO {
        star_flux(r, &ur, sr, dr)
    } else {
        star_flux(l, &ul, sl, dl)
            .add(star_flux(r, &ur, sr, dr))
            .halved()
    }
}

// ============================================================================
// Solver configuration
// ============================================================================

/// Slope limiter of the MUSCL reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limiter {
    /// `minmod(a, b)`: the smaller magnitude when `a`, `b` share a sign, else 0.
    Minmod,
    /// van Leer: `2ab/(a + b)` when `ab > 0`, else 0.
    VanLeer,
}

/// Spatial reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reconstruction {
    /// Piecewise constant (first order).
    FirstOrder,
    /// MUSCL, piecewise linear in primitive variables with a limiter.
    Muscl(Limiter),
}

/// Time integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeIntegrator {
    /// One forward-Euler stage (first order).
    ForwardEuler,
    /// Two-stage SSP Runge–Kutta (second order).
    SspRk2,
}

/// Boundary condition at one end of the domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Zero-order extrapolation (waves leave the domain).
    Transmissive,
    /// Reflecting solid wall: mirror image with `u → −u`.
    ReflectiveWall,
    /// Periodic (both ends must be periodic).
    Periodic,
}

/// Configuration of [`EulerFv1d`]. Validated by [`EulerFv1d::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EulerConfig {
    /// Ratio of specific heats `γ > 1` (e.g. `compressible::IdealGas::air().gamma`).
    pub gamma: Fix128,
    /// Interface Riemann solver.
    pub solver: RiemannSolver,
    /// Spatial reconstruction.
    pub reconstruction: Reconstruction,
    /// Time integrator.
    pub time: TimeIntegrator,
    /// CFL number `C_cfl ∈ (0, 1]` used by [`EulerFv1d::step`].
    pub cfl: Fix128,
    /// Boundary at `x = 0`.
    pub left: Boundary,
    /// Boundary at `x = N Δx`.
    pub right: Boundary,
}

impl EulerConfig {
    /// First-order Godunov (exact Riemann solver, forward Euler, CFL 0.9,
    /// transmissive ends).
    #[must_use]
    pub fn godunov(gamma: Fix128) -> Self {
        Self {
            gamma,
            solver: RiemannSolver::Exact,
            reconstruction: Reconstruction::FirstOrder,
            time: TimeIntegrator::ForwardEuler,
            cfl: Fix128::from_ratio(9, 10),
            left: Boundary::Transmissive,
            right: Boundary::Transmissive,
        }
    }

    /// Second-order MUSCL + SSP-RK2 with the given solver and limiter
    /// (CFL 0.5, transmissive ends).
    #[must_use]
    pub fn muscl(gamma: Fix128, solver: RiemannSolver, limiter: Limiter) -> Self {
        Self {
            gamma,
            solver,
            reconstruction: Reconstruction::Muscl(limiter),
            time: TimeIntegrator::SspRk2,
            cfl: Fix128::ONE.half(),
            left: Boundary::Transmissive,
            right: Boundary::Transmissive,
        }
    }

    fn validate(&self) -> Result<(), EulerError> {
        check_gamma(self.gamma)?;
        if self.cfl <= Fix128::ZERO || self.cfl > Fix128::ONE {
            return Err(EulerError::CflOutOfRange);
        }
        if (self.left == Boundary::Periodic) != (self.right == Boundary::Periodic) {
            return Err(EulerError::PeriodicMismatch);
        }
        Ok(())
    }
}

// ============================================================================
// Solver
// ============================================================================

/// 1D finite-volume solver of the compressible Euler equations on `N` uniform
/// cells. See the module doc.
#[derive(Clone, Debug)]
pub struct EulerFv1d {
    config: EulerConfig,
    consts: GammaConsts,
    dx: Fix128,
    cells: Vec<Conserved>,
    time: Fix128,
}

fn limit(limiter: Limiter, a: Fix128, b: Fix128) -> Fix128 {
    match limiter {
        Limiter::Minmod => {
            if a > Fix128::ZERO && b > Fix128::ZERO {
                min(a, b)
            } else if a < Fix128::ZERO && b < Fix128::ZERO {
                max(a, b)
            } else {
                Fix128::ZERO
            }
        }
        Limiter::VanLeer => {
            let ab = mul(a, b);
            if ab > Fix128::ZERO {
                ab.double() / (a + b)
            } else {
                Fix128::ZERO
            }
        }
    }
}

fn mirror(w: Primitive) -> Primitive {
    Primitive { u: -w.u, ..w }
}

impl EulerFv1d {
    /// A solver with the given cell averages (primitive variables), starting
    /// at `t = 0`.
    ///
    /// # Errors
    ///
    /// [`EulerError::NoCells`], [`EulerError::NonPositiveSpacing`], the
    /// configuration errors ([`EulerError::GammaNotAboveOne`],
    /// [`EulerError::CflOutOfRange`], [`EulerError::PeriodicMismatch`]) and
    /// [`EulerError::NonPositiveDensity`] / [`EulerError::NonPositivePressure`]
    /// with the index of the first invalid cell.
    pub fn new(config: EulerConfig, dx: Fix128, initial: &[Primitive]) -> Result<Self, EulerError> {
        config.validate()?;
        if initial.is_empty() {
            return Err(EulerError::NoCells);
        }
        if dx <= Fix128::ZERO {
            return Err(EulerError::NonPositiveSpacing);
        }
        for (i, w) in initial.iter().enumerate() {
            check_state(w, i)?;
        }
        let cells = initial
            .iter()
            .map(|w| Conserved::from_primitive(config.gamma, w))
            .collect();
        Ok(Self {
            config,
            consts: GammaConsts::new(config.gamma),
            dx,
            cells,
            time: Fix128::ZERO,
        })
    }

    /// Configuration.
    #[must_use]
    pub fn config(&self) -> &EulerConfig {
        &self.config
    }

    /// Cell width `Δx`.
    #[must_use]
    pub fn dx(&self) -> Fix128 {
        self.dx
    }

    /// Simulated time.
    #[must_use]
    pub fn time(&self) -> Fix128 {
        self.time
    }

    /// Cell averages of the conserved variables.
    #[must_use]
    pub fn cells(&self) -> &[Conserved] {
        &self.cells
    }

    /// Cell averages of the primitive variables.
    #[must_use]
    pub fn primitives(&self) -> Vec<Primitive> {
        self.cells.iter().map(|c| self.prim(c)).collect()
    }

    /// Sums `Σρ_i`, `Σ(ρu)_i`, `ΣE_i` (the domain integrals divided by `Δx`).
    #[must_use]
    pub fn totals(&self) -> Conserved {
        self.cells
            .iter()
            .fold(Conserved::default(), |acc, c| acc.add(*c))
    }

    /// Every stored cell is valid (checked by [`Self::new`] and by every
    /// stage before commit), so the conversion cannot fail.
    fn prim(&self, c: &Conserved) -> Primitive {
        let u = c.mom / c.rho;
        Primitive {
            rho: c.rho,
            u,
            p: mul(self.consts.g8, c.energy - half(mul(c.mom, u))),
        }
    }

    /// `max_i (|u_i| + a_i)`.
    #[must_use]
    pub fn max_wave_speed(&self) -> Fix128 {
        self.cells.iter().fold(Fix128::ZERO, |m, c| {
            let w = self.prim(c);
            max(m, w.u.abs() + self.consts.sound(&w))
        })
    }

    /// `Δt = C_cfl Δx / max_i(|u_i| + a_i)`.
    ///
    /// # Errors
    ///
    /// [`EulerError::NonPositiveTimeStep`] when the result is not positive.
    pub fn cfl_time_step(&self) -> Result<Fix128, EulerError> {
        let dt = mul(self.config.cfl, self.dx) / self.max_wave_speed();
        if dt <= Fix128::ZERO {
            return Err(EulerError::NonPositiveTimeStep);
        }
        Ok(dt)
    }

    /// One step with the CFL time step; returns `Δt`.
    ///
    /// # Errors
    ///
    /// As [`Self::step_with_dt`]; the state is unchanged on error.
    pub fn step(&mut self) -> Result<Fix128, EulerError> {
        let dt = self.cfl_time_step()?;
        self.step_with_dt(dt)?;
        Ok(dt)
    }

    /// One step with a caller-chosen `Δt` (not checked against the CFL
    /// condition).
    ///
    /// # Errors
    ///
    /// [`EulerError::NonPositiveTimeStep`]; [`EulerError::VacuumGenerated`] /
    /// [`EulerError::RiemannNotConverged`] from the exact solver;
    /// [`EulerError::NonPositiveDensity`] / [`EulerError::NonPositivePressure`]
    /// for a stage that would leave an invalid cell. The state is unchanged
    /// on error.
    pub fn step_with_dt(&mut self, dt: Fix128) -> Result<(), EulerError> {
        if dt <= Fix128::ZERO {
            return Err(EulerError::NonPositiveTimeStep);
        }
        let lambda = dt / self.dx;
        let g0 = self.scaled_fluxes(&self.cells, lambda)?;
        let new = match self.config.time {
            TimeIntegrator::ForwardEuler => self.apply(&self.cells, &g0)?,
            TimeIntegrator::SspRk2 => {
                let u1 = self.apply(&self.cells, &g0)?;
                let g1 = self.scaled_fluxes(&u1, lambda)?;
                let h: Vec<Conserved> = g0
                    .iter()
                    .zip(g1.iter())
                    .map(|(a, b)| a.add(*b).halved())
                    .collect();
                self.apply(&self.cells, &h)?
            }
        };
        self.cells = new;
        self.time = self.time + dt;
        Ok(())
    }

    /// Steps with the CFL time step until `t_end` (the last step is shortened
    /// to land on `t_end` exactly); returns the number of steps. A `t_end`
    /// not after the current time does nothing.
    ///
    /// # Errors
    ///
    /// As [`Self::step_with_dt`]; steps already taken are kept.
    pub fn advance_to(&mut self, t_end: Fix128) -> Result<usize, EulerError> {
        let mut steps = 0;
        while self.time < t_end {
            let dt = min(self.cfl_time_step()?, t_end - self.time);
            self.step_with_dt(dt)?;
            steps += 1;
        }
        Ok(steps)
    }

    /// `U_i − (G_{i+½} − G_{i−½})` for every cell, validated.
    fn apply(&self, u: &[Conserved], g: &[Conserved]) -> Result<Vec<Conserved>, EulerError> {
        let mut out = Vec::with_capacity(u.len());
        for (i, c) in u.iter().enumerate() {
            let n = c.sub(g[i + 1].sub(g[i]));
            to_primitive_at(self.config.gamma, &n, i)?;
            out.push(n);
        }
        Ok(out)
    }

    /// Primitive cell values with two ghost cells on each side.
    fn extended(&self, u: &[Conserved]) -> Vec<Primitive> {
        let n = u.len();
        let w: Vec<Primitive> = u.iter().map(|c| self.prim(c)).collect();
        let mut ext = Vec::with_capacity(n + 4);
        for j in (1..=2usize).rev() {
            ext.push(match self.config.left {
                Boundary::Transmissive => w[0],
                Boundary::ReflectiveWall => mirror(w[(j - 1).min(n - 1)]),
                Boundary::Periodic => w[(n - j % n) % n],
            });
        }
        ext.extend_from_slice(&w);
        for j in 1..=2usize {
            ext.push(match self.config.right {
                Boundary::Transmissive => w[n - 1],
                Boundary::ReflectiveWall => mirror(w[n - 1 - (j - 1).min(n - 1)]),
                Boundary::Periodic => w[(j - 1) % n],
            });
        }
        ext
    }

    /// `(Δt/Δx) F_{i−½}` for the `N + 1` interfaces `i = 0..=N`.
    fn scaled_fluxes(&self, u: &[Conserved], lambda: Fix128) -> Result<Vec<Conserved>, EulerError> {
        let ext = self.extended(u);
        let n = u.len();
        // slopes of ext[1..=n+2] (cells −1..=N)
        let slopes: Vec<Primitive> = match self.config.reconstruction {
            Reconstruction::FirstOrder => Vec::new(),
            Reconstruction::Muscl(lim) => (1..=n + 2)
                .map(|k| {
                    let (a, b, c) = (ext[k - 1], ext[k], ext[k + 1]);
                    Primitive {
                        rho: limit(lim, b.rho - a.rho, c.rho - b.rho),
                        u: limit(lim, b.u - a.u, c.u - b.u),
                        p: limit(lim, b.p - a.p, c.p - b.p),
                    }
                })
                .collect(),
        };
        let mut g = Vec::with_capacity(n + 1);
        for i in 0..=n {
            // interface between ext[i + 1] (cell i−1) and ext[i + 2] (cell i)
            let (wl, wr) = if slopes.is_empty() {
                (ext[i + 1], ext[i + 2])
            } else {
                let (a, sa) = (ext[i + 1], slopes[i]);
                let (b, sb) = (ext[i + 2], slopes[i + 1]);
                (
                    Primitive {
                        rho: a.rho + half(sa.rho),
                        u: a.u + half(sa.u),
                        p: a.p + half(sa.p),
                    },
                    Primitive {
                        rho: b.rho - half(sb.rho),
                        u: b.u - half(sb.u),
                        p: b.p - half(sb.p),
                    },
                )
            };
            let f = interface_flux(self.config.solver, &self.consts, &wl, &wr)?;
            g.push(f.scale(lambda));
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::disallowed_methods)] // f64 ln / exp are the reference here
    fn ln_and_exp_are_accurate() {
        for &x in &[1e-9, 0.001_89, 0.3, 1.0, 1.7, 7.0, 4600.0] {
            // reference at the representable input (1e-9 itself is 2⁻⁶⁴-truncated)
            let xf = Fix128::from_f64(x);
            let l = ln_pos(xf).to_f64();
            assert!((l - xf.to_f64().ln()).abs() < 1e-15, "ln {x}: {l}");
        }
        for &x in &[-30.0, -3.3, -0.2, 0.0, 0.5, 2.0, 20.0] {
            // relative 1e-15, or the 2⁻⁶⁴ resolution for small results
            let e = exp_fix(Fix128::from_f64(x)).to_f64();
            let r = Fix128::from_f64(x).to_f64().exp();
            assert!((e - r).abs() < 1e-15 * r + 1e-19, "exp {x}: {e}");
        }
    }

    #[test]
    fn symmetric_helpers() {
        let a = Fix128::from_ratio(1, 3);
        let b = Fix128::from_ratio(1, 7);
        assert_eq!(mul(-a, b), -mul(a, b));
        assert_eq!(half(-a), -half(a));
    }

    fn two() -> Fix128 {
        Fix128::from_int(2)
    }

    fn state(rho: i64, u: Fix128, p: i64) -> Primitive {
        Primitive {
            rho: Fix128::from_int(rho),
            u,
            p: Fix128::from_int(p),
        }
    }

    fn near(a: Fix128, b: Fix128, tol: f64) -> bool {
        (a - b).abs().to_f64() < tol
    }

    /// oracle: every error prints the message its documentation gives.
    #[test]
    fn error_messages() {
        for (e, text) in [
            (EulerError::NoCells, "euler_fv: no cells"),
            (EulerError::GammaNotAboveOne, "euler_fv: gamma must be > 1"),
            (
                EulerError::CflOutOfRange,
                "euler_fv: CFL number must be in (0, 1]",
            ),
            (
                EulerError::NonPositiveSpacing,
                "euler_fv: cell width must be > 0",
            ),
            (
                EulerError::NonPositiveTimeStep,
                "euler_fv: time step must be > 0",
            ),
            (
                EulerError::PeriodicMismatch,
                "euler_fv: periodic boundary must be used on both ends",
            ),
            (
                EulerError::NonPositiveDensity { cell: 4 },
                "euler_fv: non-positive density in cell 4",
            ),
            (
                EulerError::NonPositivePressure { cell: 4 },
                "euler_fv: non-positive pressure in cell 4",
            ),
            (
                EulerError::VacuumGenerated,
                "euler_fv: Riemann problem generates vacuum",
            ),
            (
                EulerError::RiemannNotConverged,
                "euler_fv: star-pressure iteration did not converge",
            ),
        ] {
            assert_eq!(e.to_string(), text);
        }
    }

    /// oracle: for `γ = 2` the state `ρ = 1, u = 2, p = 3` has momentum 2 and
    /// energy `p/(γ − 1) + ρu²/2 = 3 + 2 = 5`, and flux
    /// `(ρu, ρu² + p, u(E + p)) = (2, 7, 16)`; the conversion back gives the
    /// state again. A non-positive density or pressure is refused.
    #[test]
    fn conserved_primitive_and_flux_closed_form() {
        let w = state(1, two(), 3);
        let c = Conserved::from_primitive(two(), &w);
        let want = Conserved {
            rho: Fix128::ONE,
            mom: two(),
            energy: Fix128::from_int(5),
        };
        assert_eq!(c, want);
        assert_eq!(c.to_primitive(two()), Ok(w));
        assert_eq!(
            physical_flux(two(), &w),
            Conserved {
                rho: two(),
                mom: Fix128::from_int(7),
                energy: Fix128::from_int(16)
            }
        );
        let empty = Conserved::default();
        assert_eq!(
            empty.to_primitive(two()),
            Err(EulerError::NonPositiveDensity { cell: 0 })
        );
        let cold = Conserved {
            energy: Fix128::from_int(2),
            ..want
        };
        assert_eq!(
            cold.to_primitive(two()),
            Err(EulerError::NonPositivePressure { cell: 0 })
        );
    }

    /// oracle: minmod picks the smaller slope of equal sign and 0 across a
    /// sign change; van Leer gives `2ab/(a + b)` (`2·3/4 = 3/2` for 1 and 3)
    /// and 0 across a sign change.
    #[test]
    fn limiters_closed_form() {
        let (one, three) = (Fix128::ONE, Fix128::from_int(3));
        assert_eq!(limit(Limiter::Minmod, one, three), one);
        assert_eq!(limit(Limiter::Minmod, -one, -three), -one);
        assert_eq!(limit(Limiter::Minmod, one, -three), Fix128::ZERO);
        assert_eq!(
            limit(Limiter::VanLeer, one, three),
            Fix128::from_ratio(3, 2)
        );
        assert_eq!(limit(Limiter::VanLeer, one, -three), Fix128::ZERO);
    }

    /// oracle: a Riemann problem between equal states has the star state of
    /// that state (`p* = 3`, `u* = 1/2`) and samples to it; the Godunov and
    /// HLLC fluxes are then the physical flux. Two states flying apart with
    /// `u_R − u_L = 200 ≥ 2(a_L + a_R)/(γ − 1)` open a vacuum (6 does not); a supersonic
    /// pair (`u = ±10`, `a = √6`) takes the upwind physical flux exactly.
    #[test]
    fn riemann_problems_closed_form() {
        let w = state(1, Fix128::from_ratio(1, 2), 3);
        let star = exact_riemann(two(), &w, &w).expect("converges");
        assert!(near(star.p, Fix128::from_int(3), 1e-12), "{star:?}");
        assert!(near(star.u, Fix128::from_ratio(1, 2), 1e-12), "{star:?}");
        for s in [-Fix128::from_int(5), Fix128::ZERO, Fix128::from_int(5)] {
            let got = star.sample(two(), &w, &w, s);
            assert!(
                near(got.rho, w.rho, 1e-12) && near(got.u, w.u, 1e-12) && near(got.p, w.p, 1e-12)
            );
        }
        let f = physical_flux(two(), &w);
        for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
            let g = numerical_flux(solver, two(), &w, &w).expect("flux");
            assert!(
                near(g.rho, f.rho, 1e-12)
                    && near(g.mom, f.mom, 1e-12)
                    && near(g.energy, f.energy, 1e-12),
                "{solver:?}"
            );
        }
        let apart = (
            state(1, -Fix128::from_int(100), 3),
            state(1, Fix128::from_int(100), 3),
        );
        assert_eq!(
            exact_riemann(two(), &apart.0, &apart.1),
            Err(EulerError::VacuumGenerated)
        );
        // `u_R − u_L = 6 < 2(a_L + a_R) = 4√6 ≈ 9.8`: a star state, no vacuum
        let near_apart = (
            state(1, -Fix128::from_int(3), 3),
            state(1, Fix128::from_int(3), 3),
        );
        assert!(exact_riemann(two(), &near_apart.0, &near_apart.1).is_ok());
        let left = state(2, Fix128::from_int(-10), 3);
        let fast = state(1, Fix128::from_int(10), 3);
        assert_eq!(
            numerical_flux(
                RiemannSolver::Hllc,
                two(),
                &fast,
                &state(2, Fix128::from_int(10), 3)
            ),
            Ok(physical_flux(two(), &fast))
        );
        assert_eq!(
            numerical_flux(
                RiemannSolver::Hllc,
                two(),
                &state(2, Fix128::from_int(-10), 3),
                &left
            ),
            Ok(physical_flux(two(), &left))
        );
    }

    /// oracle: the solver refuses `γ ≤ 1`, a CFL outside `(0, 1]`, a periodic
    /// boundary on one end only, no cells, a non-positive width, a bad cell
    /// (with its index) and a non-positive step.
    #[test]
    fn solver_refusals() {
        let w = [state(1, Fix128::ZERO, 1)];
        let dx = Fix128::ONE;
        let cfg = EulerConfig::godunov(two());
        let new =
            |c: EulerConfig, dx: Fix128, cells: &[Primitive]| EulerFv1d::new(c, dx, cells).err();
        assert_eq!(
            new(EulerConfig::godunov(Fix128::ONE), dx, &w),
            Some(EulerError::GammaNotAboveOne)
        );
        assert_eq!(
            new(
                EulerConfig {
                    cfl: Fix128::ZERO,
                    ..cfg
                },
                dx,
                &w
            ),
            Some(EulerError::CflOutOfRange)
        );
        assert_eq!(
            new(EulerConfig { cfl: two(), ..cfg }, dx, &w),
            Some(EulerError::CflOutOfRange)
        );
        assert_eq!(
            new(
                EulerConfig {
                    left: Boundary::Periodic,
                    ..cfg
                },
                dx,
                &w
            ),
            Some(EulerError::PeriodicMismatch)
        );
        assert_eq!(new(cfg, dx, &[]), Some(EulerError::NoCells));
        assert_eq!(
            new(cfg, Fix128::ZERO, &w),
            Some(EulerError::NonPositiveSpacing)
        );
        assert_eq!(
            new(cfg, dx, &[w[0], state(0, Fix128::ZERO, 1)]),
            Some(EulerError::NonPositiveDensity { cell: 1 })
        );
        assert_eq!(
            new(cfg, dx, &[w[0], w[0], state(1, Fix128::ZERO, 0)]),
            Some(EulerError::NonPositivePressure { cell: 2 })
        );
        let mut s = EulerFv1d::new(cfg, dx, &w).expect("valid");
        assert_eq!(
            s.step_with_dt(Fix128::ZERO),
            Err(EulerError::NonPositiveTimeStep)
        );
    }

    /// oracle: a uniform state is a steady solution for every scheme and
    /// boundary (identical interface fluxes cancel): the cells stay exactly
    /// as given while time advances by the CFL step
    /// `cfl·dx/(|u| + √(γp/ρ))`. Walls need `u = 0`. `advance_to` stops
    /// exactly at the end time. Totals are 8 times the cell.
    #[test]
    fn uniform_state_is_steady_for_every_scheme() {
        let moving = Fix128::from_ratio(1, 2);
        let cases = [
            (EulerConfig::godunov(two()), moving),
            (
                EulerConfig::muscl(two(), RiemannSolver::Hllc, Limiter::Minmod),
                moving,
            ),
            (
                EulerConfig::muscl(two(), RiemannSolver::Exact, Limiter::VanLeer),
                moving,
            ),
            (
                EulerConfig {
                    left: Boundary::Periodic,
                    right: Boundary::Periodic,
                    ..EulerConfig::muscl(two(), RiemannSolver::Hllc, Limiter::VanLeer)
                },
                moving,
            ),
            (
                EulerConfig {
                    left: Boundary::ReflectiveWall,
                    right: Boundary::ReflectiveWall,
                    ..EulerConfig::godunov(two())
                },
                Fix128::ZERO,
            ),
            (
                EulerConfig {
                    left: Boundary::ReflectiveWall,
                    right: Boundary::ReflectiveWall,
                    ..EulerConfig::muscl(two(), RiemannSolver::Exact, Limiter::Minmod)
                },
                Fix128::ZERO,
            ),
        ];
        for (cfg, u) in cases {
            let w = state(1, u, 3);
            let dx = Fix128::from_ratio(1, 8);
            let mut s = EulerFv1d::new(cfg, dx, &[w; 8]).expect("valid");
            assert_eq!(*s.config(), cfg);
            assert_eq!(s.dx(), dx);
            let cell = Conserved::from_primitive(two(), &w);
            assert_eq!(s.totals(), cell.scale(Fix128::from_int(8)));
            let speed = u.to_f64() + 6f64.sqrt();
            assert!(
                (s.max_wave_speed().to_f64() - speed).abs() < 1e-12,
                "{cfg:?}"
            );
            let dt = s.step().expect("step");
            assert!(
                (dt.to_f64() - cfg.cfl.to_f64() / 8.0 / speed).abs() < 1e-12,
                "{cfg:?}"
            );
            assert_eq!(s.time(), dt);
            assert!(s.cells().iter().all(|c| *c == cell), "{cfg:?}");
            let t_end = Fix128::from_ratio(1, 4);
            let steps = s.advance_to(t_end).expect("advance");
            assert!(steps >= 1);
            assert_eq!(s.time(), t_end);
            assert!(s.cells().iter().all(|c| *c == cell), "{cfg:?}");
            assert!(s
                .primitives()
                .iter()
                .all(|p| near(p.p, w.p, 1e-12) && p.u == w.u));
        }
    }
}
