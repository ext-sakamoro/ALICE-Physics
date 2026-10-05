//! U.S. Standard Atmosphere 1976 (ISA) for the lower two layers.
//!
//! Temperature, pressure, density and speed of sound as functions of
//! altitude, for the troposphere and the lower stratosphere:
//!
//! ```text
//! layer 0, 0 ≤ H ≤ 11 000 m      T = T₀ − L·H
//!                                p = p₀ · (T / T₀)^(g₀ M₀ / (R* L))
//! layer 1, 11 000 < H ≤ 20 000 m T = T₁₁                     (isothermal)
//!                                p = p₁₁ · exp(−g₀ M₀ (H − 11 000) / (R* T₁₁))
//! every layer                    ρ = p M₀ / (R* T),  a = √(γ R* T / M₀)
//! ```
//!
//! with the 1976 constants `T₀ = 288.15 K`, `p₀ = 101 325 Pa`,
//! `L = 0.0065 K/m`, `g₀ = 9.806 65 m/s²`, `R* = 8.314 32 J/(mol·K)`,
//! `M₀ = 0.028 964 4 kg/mol`, `γ = 1.4`. `T₁₁` and `p₁₁` are the layer-0
//! values at `H = 11 000 m` (216.65 K and about 22 632 Pa), so `T` and `p`
//! are continuous at the tropopause.
//!
//! # Altitude
//!
//! `H` is the **geopotential** altitude, the variable the 1976 layer
//! formulas are written in. [`Isa1976::geopotential_altitude_m`] converts a
//! geometric altitude `Z` with `H = r₀ Z / (r₀ + Z)`, `r₀ = 6 356 766 m`;
//! [`Isa1976::at_geometric_altitude`] does that conversion first. The two
//! differ by 0.3 % at 20 km.
//!
//! # Range
//!
//! `0 ≤ H ≤ 20 000 m`. Outside it the call returns
//! [`IsaError::AltitudeOutOfRange`]; it does not clamp, because a clamped
//! value would be a plausible-looking density at the wrong altitude. The
//! layers above 20 km (with a positive lapse rate) and the table below sea
//! level are not modelled.
//!
//! # Determinism
//!
//! Everything is `Fix128`: the power is [`Fix128::powf_pos`] and the
//! exponential [`Fix128::exp`] (relative error below about `1e-6`), so the
//! result is bit-identical on every target and available under `no_std`.

use crate::math::Fix128;

/// Effective Earth radius `r₀` of the 1976 standard (m).
const EARTH_RADIUS_M: i64 = 6_356_766;
/// Base of layer 1 (geopotential m).
const TROPOPAUSE_M: i64 = 11_000;

/// The 1976 constants as `Fix128` (built from exact decimal ratios).
struct Constants {
    t0: Fix128,
    p0: Fix128,
    lapse: Fix128,
    g0: Fix128,
    gas_constant: Fix128,
    molar_mass: Fix128,
    gamma: Fix128,
}

impl Constants {
    fn new() -> Self {
        Self {
            t0: Fix128::from_ratio(28_815, 100),
            p0: Fix128::from_int(101_325),
            lapse: Fix128::from_ratio(65, 10_000),
            g0: Fix128::from_ratio(980_665, 100_000),
            gas_constant: Fix128::from_ratio(831_432, 100_000),
            molar_mass: Fix128::from_ratio(289_644, 10_000_000),
            gamma: Fix128::from_ratio(14, 10),
        }
    }

    /// `g₀ M₀ / (R* L)` ≈ 5.255 88.
    fn troposphere_exponent(&self) -> Fix128 {
        self.g0 * self.molar_mass / (self.gas_constant * self.lapse)
    }

    /// `(T₁₁, p₁₁)`, the layer-0 values at the tropopause.
    fn tropopause(&self) -> (Fix128, Fix128) {
        let t11 = self.t0 - self.lapse * Fix128::from_int(TROPOPAUSE_M);
        let p11 = self.p0 * (t11 / self.t0).powf_pos(self.troposphere_exponent());
        (t11, p11)
    }
}

/// The U.S. Standard Atmosphere 1976, layers 0 and 1 (`0 ≤ H ≤ 20 km`).
///
/// A namespace for the layer formulas; it holds no state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Isa1976;

/// Thermodynamic state of the standard atmosphere at one altitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtmosphereState {
    /// Absolute temperature (K).
    pub temperature_k: Fix128,
    /// Static pressure (Pa).
    pub pressure_pa: Fix128,
    /// Density (kg/m³).
    pub density_kg_m3: Fix128,
    /// Speed of sound (m/s).
    pub speed_of_sound_m_s: Fix128,
}

/// Why [`Isa1976`] could not evaluate an altitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaError {
    /// The geopotential altitude is below 0 m or above 20 000 m.
    AltitudeOutOfRange {
        /// The rejected geopotential altitude (m).
        geopotential_altitude_m: Fix128,
    },
}

impl core::fmt::Display for IsaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AltitudeOutOfRange {
                geopotential_altitude_m,
            } => write!(
                f,
                "geopotential altitude {} m is outside the modelled 0..=20000 m",
                geopotential_altitude_m.to_f64()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for IsaError {}

impl Isa1976 {
    /// Highest geopotential altitude the model covers (m).
    pub const MAX_GEOPOTENTIAL_ALTITUDE_M: i64 = 20_000;

    /// Geopotential altitude `H = r₀ Z / (r₀ + Z)` (m) of a geometric altitude
    /// `Z` (m), with the 1976 effective Earth radius `r₀ = 6 356 766 m`.
    ///
    /// `Z = −r₀` (a zero denominator) returns `ZERO`, the `Fix128` division
    /// convention; the result is then rejected by the range check.
    #[must_use]
    pub fn geopotential_altitude_m(geometric_altitude_m: Fix128) -> Fix128 {
        let r0 = Fix128::from_int(EARTH_RADIUS_M);
        r0 * geometric_altitude_m / (r0 + geometric_altitude_m)
    }

    /// State at a geopotential altitude `H` (m).
    ///
    /// # Errors
    ///
    /// [`IsaError::AltitudeOutOfRange`] when `H < 0` or `H > 20 000 m`.
    pub fn at_geopotential_altitude(
        geopotential_altitude_m: Fix128,
    ) -> Result<AtmosphereState, IsaError> {
        let h = geopotential_altitude_m;
        if h.is_negative() || h > Fix128::from_int(Self::MAX_GEOPOTENTIAL_ALTITUDE_M) {
            return Err(IsaError::AltitudeOutOfRange {
                geopotential_altitude_m: h,
            });
        }
        let c = Constants::new();
        let tropopause = Fix128::from_int(TROPOPAUSE_M);
        let (temperature_k, pressure_pa) = if h <= tropopause {
            let t = c.t0 - c.lapse * h;
            (t, c.p0 * (t / c.t0).powf_pos(c.troposphere_exponent()))
        } else {
            let (t11, p11) = c.tropopause();
            let decay = (c.g0 * c.molar_mass * (h - tropopause)) / (c.gas_constant * t11);
            (t11, p11 * (-decay).exp())
        };
        let density_kg_m3 = pressure_pa * c.molar_mass / (c.gas_constant * temperature_k);
        let speed_of_sound_m_s = (c.gamma * c.gas_constant * temperature_k / c.molar_mass).sqrt();
        Ok(AtmosphereState {
            temperature_k,
            pressure_pa,
            density_kg_m3,
            speed_of_sound_m_s,
        })
    }

    /// State at a geometric altitude `Z` (m): converts with
    /// [`Self::geopotential_altitude_m`], then [`Self::at_geopotential_altitude`].
    ///
    /// # Errors
    ///
    /// [`IsaError::AltitudeOutOfRange`] when `Z < 0` or the geopotential
    /// altitude is above 20 000 m (geometric `Z` above about 20 063 m).
    pub fn at_geometric_altitude(
        geometric_altitude_m: Fix128,
    ) -> Result<AtmosphereState, IsaError> {
        let h = Self::geopotential_altitude_m(geometric_altitude_m);
        // Checked here, not only through `h`: at Z = −r₀ the conversion has a
        // zero denominator and returns 0, and below −r₀ it turns positive.
        if geometric_altitude_m.is_negative() {
            return Err(IsaError::AltitudeOutOfRange {
                geopotential_altitude_m: h,
            });
        }
        Self::at_geopotential_altitude(h)
    }
}
