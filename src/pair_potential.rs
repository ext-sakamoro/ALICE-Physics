//! Pair potentials between point particles: Lennard-Jones 12-6, Morse,
//! Coulomb and screened Coulomb (Yukawa), with cutoff / shift and the
//! Lorentz–Berthelot mixing rule.
//!
//! # Laws
//!
//! ```text
//! Lennard-Jones   U = 4ε[(σ/r)¹² − (σ/r)⁶]          F = 24ε[2(σ/r)¹² − (σ/r)⁶] / r
//! Morse           U = D(1 − e)² − D,  e = e^{−a(r − r_e)}   F = −2 D a e (1 − e)
//! Coulomb         U = k q_i q_j / r                  F = k q_i q_j / r²
//! Yukawa          U = k q_i q_j e^{−r/λ} / r         F = k q_i q_j e^{−r/λ} (1/r² + 1/(λ r))
//! ```
//!
//! `F = −dU/dr` is the radial force: positive is repulsive. The force on
//! particle `i` from `j` is `F(r) r̂` with `r̂ = (x_i − x_j)/r`
//! ([`PairPotential::force_on_first`]); `j` receives the negative.
//!
//! - Lennard-Jones: minimum `U = −ε` at `r_min = 2^{1/6} σ`, `U(σ) = 0`,
//!   curvature `U''(r_min) = 72 ε / r_min² ≈ 57.146 ε/σ²`.
//! - Morse is written with its minimum at `−D` (`U(r_e) = −D`, `U → 0` as
//!   `r → ∞`), so it has the same zero at infinity as the others and can be
//!   truncated the same way. The harmonic force constant at `r_e` is
//!   `2 D a²`.
//! - Coulomb uses [`COULOMB_CONSTANT`] `k = 8.99e9 V·m/C`, the same constant
//!   and sign convention as [`crate::electromagnetic::EmSource::PointCharge`]:
//!   the force on `i` equals `q_i E_j(x_i)` with `E_j` the field of charge
//!   `j`. `Coulomb::with_constant` takes `k` explicitly (reduced units).
//! - Yukawa is the Debye–Hückel screened Coulomb interaction of charges in an
//!   electrolyte or plasma with screening length `λ`; `λ → ∞` recovers
//!   Coulomb. It is included because it is a law of the same form
//!   (parameterised by `λ` alone) and its long-range part decays
//!   exponentially, which makes a cutoff meaningful; the bare Coulomb sum
//!   with a cutoff is not convergent in a periodic system (Ewald / PPPM
//!   summation is not provided).
//!
//! # Cutoff and shift
//!
//! [`Truncated`] cuts a potential at `r_c` (zero for `r ≥ r_c`) with an
//! explicit [`ShiftMode`]:
//!
//! ```text
//! None         U(r)                                F(r)            (U jumps at r_c)
//! EnergyShift  U(r) − U(r_c)                       F(r)            (F jumps at r_c)
//! ForceShift   U(r) − U(r_c) + (r − r_c) F(r_c)    F(r) − F(r_c)   (both vanish at r_c)
//! ```
//!
//! (`ForceShift` is the shifted-force potential, `U'(r_c) = −F(r_c)`.)
//!
//! # Degenerate input
//!
//! Every evaluation takes `r > 0`; `r ≤ 0` (coincident particles) is
//! [`PairPotentialError::NonPositiveDistance`]: the direction of the force is
//! undefined there and Lennard-Jones / Coulomb diverge. A value outside the
//! `Fix128` range (`|x| ≥ 2⁶³`, e.g. `(σ/r)¹²` for `r < σ/1500`) is
//! [`PairPotentialError::Overflow`] instead of a wrapped number. Parameters
//! are validated by the constructors (`σ > 0`, `ε ≥ 0`, `D ≥ 0`, `a > 0`,
//! `r_e > 0`, `k > 0`, `λ > 0`, `r_c > 0`); `ε = 0` / `D = 0` / zero charges
//! give a potential that is zero everywhere.
//!
//! # Determinism
//!
//! `Fix128` throughout (no `std` needed). `exp` is
//! [`crate::math_util::exp_fix`], extended above its saturation at 20 by
//! halving the argument and squaring the result (the same steps `exp_fix`
//! itself takes, so the extension is continuous at 20).
//!
//! Specific molecules' `ε`, `σ`, charges or force-field tables are not part
//! of this module: every parameter is passed in.

use crate::math::{Fix128, Vec3Fix};
use crate::math_util::exp_fix;

/// Coulomb constant `k = 1/(4π ε₀) ≈ 8.99e9 V·m/C`, the value used by
/// [`crate::electromagnetic::EmSource::PointCharge`].
pub const COULOMB_CONSTANT: Fix128 = Fix128::from_int(8_990_000_000);

/// Why a pair potential could not be built or evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairPotentialError {
    /// Distance `r ≤ 0` (coincident particles).
    NonPositiveDistance,
    /// Lennard-Jones `σ ≤ 0`.
    NonPositiveSigma,
    /// Lennard-Jones `ε < 0`.
    NegativeEpsilon,
    /// Morse well depth `D < 0`.
    NegativeWellDepth,
    /// Morse width parameter `a ≤ 0`.
    NonPositiveWidth,
    /// Morse equilibrium distance `r_e ≤ 0`.
    NonPositiveEquilibriumDistance,
    /// Coulomb constant `k ≤ 0`.
    NonPositiveCoulombConstant,
    /// Yukawa screening length `λ ≤ 0`.
    NonPositiveScreeningLength,
    /// Cutoff radius `r_c ≤ 0`.
    NonPositiveCutoff,
    /// The value is outside the `Fix128` range.
    Overflow,
}

impl core::fmt::Display for PairPotentialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::NonPositiveDistance => "pair distance must be positive",
            Self::NonPositiveSigma => "Lennard-Jones sigma must be positive",
            Self::NegativeEpsilon => "Lennard-Jones epsilon must not be negative",
            Self::NegativeWellDepth => "Morse well depth must not be negative",
            Self::NonPositiveWidth => "Morse width parameter must be positive",
            Self::NonPositiveEquilibriumDistance => "Morse equilibrium distance must be positive",
            Self::NonPositiveCoulombConstant => "Coulomb constant must be positive",
            Self::NonPositiveScreeningLength => "screening length must be positive",
            Self::NonPositiveCutoff => "cutoff radius must be positive",
            Self::Overflow => "pair potential value is outside the fixed-point range",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PairPotentialError {}

/// A central pair potential `U(r)` and its radial force `F(r) = −dU/dr`.
pub trait PairPotential {
    /// Potential energy `U(r)` at distance `r > 0`.
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError>;

    /// Radial force `F(r) = −dU/dr` at `r > 0` (positive = repulsive).
    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError>;

    /// Force on the first particle of a pair separated by `d = x_i − x_j`:
    /// `F(|d|) d / |d|`. The second particle receives the negative.
    fn force_on_first(&self, d: Vec3Fix) -> Result<Vec3Fix, PairPotentialError> {
        let r = d.length();
        let f = self.force(r)?;
        Ok(d * (f / r))
    }
}

fn mul(a: Fix128, b: Fix128) -> Result<Fix128, PairPotentialError> {
    a.checked_mul(b).ok_or(PairPotentialError::Overflow)
}

fn positive_distance(r: Fix128) -> Result<(), PairPotentialError> {
    if r > Fix128::ZERO {
        Ok(())
    } else {
        Err(PairPotentialError::NonPositiveDistance)
    }
}

/// `1 / r` and `1 / r²` for `r > 0`, with the range checked: `1/r` itself
/// stays in range for every positive `Fix128` (`1/2⁻⁶⁴ = 2⁶⁴` does not, so
/// the smallest distances are an overflow).
fn inverse(r: Fix128) -> Result<Fix128, PairPotentialError> {
    // 1 / r ≥ 2⁶³ ⇔ r ≤ 2⁻⁶³
    if r <= Fix128::from_raw(0, 2) {
        return Err(PairPotentialError::Overflow);
    }
    Ok(Fix128::ONE / r)
}

/// `e^x` for every `x`: [`exp_fix`] below its saturation at 20, above that
/// by halving `x` and squaring the result back (overflow is an error).
fn exp_extended(x: Fix128) -> Result<Fix128, PairPotentialError> {
    let limit = Fix128::from_int(20);
    let mut y = x;
    let mut squarings = 0u32;
    while y >= limit {
        y = y.half();
        squarings += 1;
    }
    let mut e = exp_fix(y);
    for _ in 0..squarings {
        e = mul(e, e)?;
    }
    Ok(e)
}

// ---------------------------------------------------------------------------
// Lennard-Jones
// ---------------------------------------------------------------------------

/// Lennard-Jones 12-6 potential `U = 4ε[(σ/r)¹² − (σ/r)⁶]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LennardJones {
    epsilon: Fix128,
    sigma: Fix128,
}

impl LennardJones {
    /// Well depth `ε ≥ 0` and zero-crossing distance `σ > 0`.
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NonPositiveSigma`],
    /// [`PairPotentialError::NegativeEpsilon`].
    pub fn new(epsilon: Fix128, sigma: Fix128) -> Result<Self, PairPotentialError> {
        if sigma <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveSigma);
        }
        if epsilon.is_negative() {
            return Err(PairPotentialError::NegativeEpsilon);
        }
        Ok(Self { epsilon, sigma })
    }

    /// Well depth `ε`.
    #[must_use]
    pub const fn epsilon(&self) -> Fix128 {
        self.epsilon
    }

    /// Zero-crossing distance `σ`.
    #[must_use]
    pub const fn sigma(&self) -> Fix128 {
        self.sigma
    }

    /// `((σ/r)⁶, (σ/r)¹²)`.
    fn powers(&self, r: Fix128) -> Result<(Fix128, Fix128), PairPotentialError> {
        positive_distance(r)?;
        let s = mul(self.sigma, inverse(r)?)?;
        let s2 = mul(s, s)?;
        let s6 = mul(mul(s2, s2)?, s2)?;
        let s12 = mul(s6, s6)?;
        Ok((s6, s12))
    }
}

impl PairPotential for LennardJones {
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let (s6, s12) = self.powers(r)?;
        mul(Fix128::from_int(4), mul(self.epsilon, s12 - s6)?)
    }

    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let (s6, s12) = self.powers(r)?;
        let bracket = s12.double() - s6;
        let f = mul(mul(Fix128::from_int(24), self.epsilon)?, bracket)?;
        mul(f, inverse(r)?)
    }
}

/// Lorentz–Berthelot mixing rule for Lennard-Jones parameters of unlike
/// species: `σ_ij = (σ_i + σ_j)/2`, `ε_ij = √(ε_i ε_j)`.
#[must_use]
pub fn lorentz_berthelot(a: &LennardJones, b: &LennardJones) -> LennardJones {
    LennardJones {
        sigma: (a.sigma + b.sigma).half(),
        epsilon: (a.epsilon * b.epsilon).sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Morse
// ---------------------------------------------------------------------------

/// Morse potential `U = D(1 − e^{−a(r − r_e)})² − D` (minimum `−D` at `r_e`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Morse {
    well_depth: Fix128,
    width: Fix128,
    equilibrium_distance: Fix128,
}

impl Morse {
    /// Well depth `D ≥ 0`, width parameter `a > 0` (1/length) and
    /// equilibrium distance `r_e > 0`.
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NegativeWellDepth`],
    /// [`PairPotentialError::NonPositiveWidth`],
    /// [`PairPotentialError::NonPositiveEquilibriumDistance`].
    pub fn new(
        well_depth: Fix128,
        width: Fix128,
        equilibrium_distance: Fix128,
    ) -> Result<Self, PairPotentialError> {
        if well_depth.is_negative() {
            return Err(PairPotentialError::NegativeWellDepth);
        }
        if width <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveWidth);
        }
        if equilibrium_distance <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveEquilibriumDistance);
        }
        Ok(Self {
            well_depth,
            width,
            equilibrium_distance,
        })
    }

    /// Harmonic force constant at the minimum, `2 D a²`.
    #[must_use]
    pub fn harmonic_force_constant(&self) -> Fix128 {
        (self.well_depth * self.width * self.width).double()
    }

    /// `e^{−a(r − r_e)}`.
    fn decay(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        positive_distance(r)?;
        exp_extended(mul(self.width, self.equilibrium_distance - r)?)
    }
}

impl PairPotential for Morse {
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let one_minus = Fix128::ONE - self.decay(r)?;
        Ok(mul(self.well_depth, mul(one_minus, one_minus)?)? - self.well_depth)
    }

    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let e = self.decay(r)?;
        let da2 = (self.well_depth * self.width).double();
        Ok(-mul(da2, mul(e, Fix128::ONE - e)?)?)
    }
}

// ---------------------------------------------------------------------------
// Coulomb / Yukawa
// ---------------------------------------------------------------------------

/// Coulomb interaction `U = k q_i q_j / r` of two point charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coulomb {
    /// `k q_i q_j`, formed as `(k q_i) q_j` so that small SI charges keep
    /// their relative precision (`q_i q_j` alone is ~1e-12 C², only ~1e-8
    /// relative in `Fix128`).
    coupling: Fix128,
}

impl Coulomb {
    /// Charges `q_i`, `q_j` in coulombs with [`COULOMB_CONSTANT`].
    #[must_use]
    pub fn new(charge_i: Fix128, charge_j: Fix128) -> Self {
        Self {
            coupling: COULOMB_CONSTANT * charge_i * charge_j,
        }
    }

    /// Charges with an explicit constant `k > 0` (e.g. `k = 1` in reduced
    /// units).
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NonPositiveCoulombConstant`].
    pub fn with_constant(
        coulomb_constant: Fix128,
        charge_i: Fix128,
        charge_j: Fix128,
    ) -> Result<Self, PairPotentialError> {
        if coulomb_constant <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveCoulombConstant);
        }
        Ok(Self {
            coupling: coulomb_constant * charge_i * charge_j,
        })
    }
}

impl PairPotential for Coulomb {
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        positive_distance(r)?;
        mul(self.coupling, inverse(r)?)
    }

    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        positive_distance(r)?;
        let inv = inverse(r)?;
        mul(mul(self.coupling, inv)?, inv)
    }
}

/// Screened Coulomb (Yukawa / Debye–Hückel) interaction
/// `U = k q_i q_j e^{−r/λ} / r`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yukawa {
    coulomb: Coulomb,
    screening_length: Fix128,
}

impl Yukawa {
    /// Charges in coulombs with [`COULOMB_CONSTANT`] and screening length
    /// `λ > 0`.
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NonPositiveScreeningLength`].
    pub fn new(
        charge_i: Fix128,
        charge_j: Fix128,
        screening_length: Fix128,
    ) -> Result<Self, PairPotentialError> {
        Self::from_coulomb(Coulomb::new(charge_i, charge_j), screening_length)
    }

    /// Charges with an explicit constant `k > 0` and screening length `λ > 0`.
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NonPositiveCoulombConstant`],
    /// [`PairPotentialError::NonPositiveScreeningLength`].
    pub fn with_constant(
        coulomb_constant: Fix128,
        charge_i: Fix128,
        charge_j: Fix128,
        screening_length: Fix128,
    ) -> Result<Self, PairPotentialError> {
        Self::from_coulomb(
            Coulomb::with_constant(coulomb_constant, charge_i, charge_j)?,
            screening_length,
        )
    }

    fn from_coulomb(
        coulomb: Coulomb,
        screening_length: Fix128,
    ) -> Result<Self, PairPotentialError> {
        if screening_length <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveScreeningLength);
        }
        Ok(Self {
            coulomb,
            screening_length,
        })
    }

    fn screening(&self, r: Fix128) -> Fix128 {
        exp_fix(-(r / self.screening_length))
    }
}

impl PairPotential for Yukawa {
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let u = self.coulomb.energy(r)?;
        mul(u, self.screening(r))
    }

    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        let f = self.coulomb.force(r)?;
        // F = k q q e^{−r/λ} (1/r²)(1 + r/λ)
        let factor = Fix128::ONE + r / self.screening_length;
        mul(mul(f, factor)?, self.screening(r))
    }
}

// ---------------------------------------------------------------------------
// Cutoff
// ---------------------------------------------------------------------------

/// How a [`Truncated`] potential is made to vanish at the cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftMode {
    /// Plain truncation: `U(r)`, `F(r)` inside, both jump to 0 at `r_c`.
    None,
    /// `U(r) − U(r_c)`; the force is unchanged and jumps at `r_c`.
    EnergyShift,
    /// Shifted force: `U(r) − U(r_c) + (r − r_c) F(r_c)` and `F(r) − F(r_c)`;
    /// both vanish at `r_c`.
    ForceShift,
}

/// A pair potential cut at `r_c`: zero for `r ≥ r_c`, shifted inside by
/// [`ShiftMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncated<P> {
    potential: P,
    cutoff: Fix128,
    mode: ShiftMode,
    energy_at_cutoff: Fix128,
    force_at_cutoff: Fix128,
}

impl<P: PairPotential> Truncated<P> {
    /// Cut `potential` at `cutoff > 0` with `mode`.
    ///
    /// # Errors
    ///
    /// [`PairPotentialError::NonPositiveCutoff`], or the error of evaluating
    /// the potential at the cutoff.
    pub fn new(potential: P, cutoff: Fix128, mode: ShiftMode) -> Result<Self, PairPotentialError> {
        if cutoff <= Fix128::ZERO {
            return Err(PairPotentialError::NonPositiveCutoff);
        }
        let energy_at_cutoff = potential.energy(cutoff)?;
        let force_at_cutoff = potential.force(cutoff)?;
        Ok(Self {
            potential,
            cutoff,
            mode,
            energy_at_cutoff,
            force_at_cutoff,
        })
    }

    /// Cutoff radius `r_c`.
    #[must_use]
    pub const fn cutoff(&self) -> Fix128 {
        self.cutoff
    }

    /// Shift mode.
    #[must_use]
    pub const fn mode(&self) -> ShiftMode {
        self.mode
    }

    /// The uncut potential.
    #[must_use]
    pub const fn inner(&self) -> &P {
        &self.potential
    }
}

impl<P: PairPotential> PairPotential for Truncated<P> {
    fn energy(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        positive_distance(r)?;
        if r >= self.cutoff {
            return Ok(Fix128::ZERO);
        }
        let u = self.potential.energy(r)?;
        Ok(match self.mode {
            ShiftMode::None => u,
            ShiftMode::EnergyShift => u - self.energy_at_cutoff,
            ShiftMode::ForceShift => {
                u - self.energy_at_cutoff + mul(r - self.cutoff, self.force_at_cutoff)?
            }
        })
    }

    fn force(&self, r: Fix128) -> Result<Fix128, PairPotentialError> {
        positive_distance(r)?;
        if r >= self.cutoff {
            return Ok(Fix128::ZERO);
        }
        let f = self.potential.force(r)?;
        Ok(match self.mode {
            ShiftMode::None | ShiftMode::EnergyShift => f,
            ShiftMode::ForceShift => f - self.force_at_cutoff,
        })
    }
}
