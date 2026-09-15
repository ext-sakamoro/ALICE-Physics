//! Non-Newtonian Fluid Constitutive Models
//!
//! Phase F1 of the ALICE-Physics completeness project. Real fluids of
//! engineering interest (molten polymers during extrusion, blood, ketchup,
//! cornstarch slurry, drilling mud, silicone gels) do not obey Newton's
//! linear law τ = μ·γ̇. This module provides three widely-used constitutive
//! models plus one composite (Herschel-Bulkley) for use as viscosity
//! functions in the fluid solvers (`fluid.rs`, `eulerian_grid.rs`).
//!
//! # Models
//!
//! - **Power-law (Ostwald-de Waele)**: `τ = K · γ̇ⁿ`. Simplest, captures
//!   shear-thinning (`n < 1`) and shear-thickening (`n > 1`).
//! - **Carreau**: `η(γ̇) = η_∞ + (η_0 − η_∞) · (1 + (λ·γ̇)²)^((n-1)/2)`.
//!   Reproduces both the Newtonian plateau at low shear and the power-law
//!   region at high shear; used for most polymer melts.
//! - **Bingham plastic**: `τ = τ_y + μ_p·γ̇` when `τ > τ_y`, else no flow.
//!   For yield-stress fluids (drilling mud, toothpaste).
//! - **Herschel-Bulkley**: `τ = τ_y + K·γ̇ⁿ` — Bingham + power-law hybrid.
//!
//! # Integer exponent restriction
//!
//! The power-law and Carreau structs store their exponent as an integer so
//! that `Default`/`Eq` stay trivial: [`Carreau::half_exponent`] is `(n−1)/2`
//! rounded to an integer, so `-1` is `n = −1` (steep thinning), `0` is
//! Newtonian and `1` is `n = 3` (thickening). For the classical polymer-melt
//! `n ≈ 0.4` use [`Carreau::viscosity_with_index`], which evaluates the
//! fractional exponent exactly with [`Fix128::powf_pos`]; the integer field
//! is kept for the fast path and for the `Eq`-derived `Default`.
//!
//! # References
//!
//! - Bird, Stewart, Lightfoot, *Transport Phenomena* 2nd ed. Ch. 8.
//! - Chhabra & Richardson, *Non-Newtonian Flow and Applied Rheology* 2nd ed.
//! - Bingham, "An investigation of the laws of plastic flow", Bulletin
//!   NBS 13 (1917).

use crate::math::Fix128;

// ============================================================================
// Power-law
// ============================================================================

/// Power-law parameters `τ = K · γ̇ⁿ`.
///
/// Integer `n_int` is the exponent (1 = Newtonian, 2 = mild shear-thickening,
/// etc.).  For strong shear-thinning (n < 1) use `PowerLaw::shear_thinning`
/// which internally uses `1 / γ̇^m` with integer `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowerLaw {
    /// Consistency index `K` (Pa·s^n).
    pub k: Fix128,
    /// Integer exponent (see doc).
    pub n_int: u32,
    /// Whether to invert the exponent (shear-thinning n < 1).
    pub thinning: bool,
}

impl PowerLaw {
    /// Newtonian degenerate case (`τ = μ·γ̇`).
    #[must_use]
    pub fn newtonian(mu_pa_s: Fix128) -> Self {
        Self {
            k: mu_pa_s,
            n_int: 1,
            thinning: false,
        }
    }

    /// Shear-thickening: `τ = K · γ̇ⁿ` with integer `n ≥ 2`.
    #[must_use]
    pub fn shear_thickening(k: Fix128, n_int: u32) -> Self {
        Self {
            k,
            n_int: n_int.max(2),
            thinning: false,
        }
    }

    /// Shear-thinning: `τ = K · γ̇^(1/n)` with integer `n ≥ 2` (Ostwald–de Waele
    /// with flow index `1/n < 1`; evaluated with [`Fix128::powf_pos`]).
    ///
    /// Before 1.2.0 the stress was computed as `K · γ̇ / γ̇ⁿ = K · γ̇^(1−n)`, a
    /// *decreasing* flow curve (negative differential viscosity), not a
    /// shear-thinning fluid whose stress still grows with shear rate
    /// (`tests/engineering_oracles_fluid.rs`).
    #[must_use]
    pub fn shear_thinning(k: Fix128, n_int: u32) -> Self {
        Self {
            k,
            n_int: n_int.max(2),
            thinning: true,
        }
    }

    /// Shear stress at a given shear rate (Pa).
    #[must_use]
    pub fn stress(&self, shear_rate_per_s: Fix128) -> Fix128 {
        if shear_rate_per_s <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        // Integer power
        let mut p = Fix128::ONE;
        for _ in 0..self.n_int {
            p = p * shear_rate_per_s;
        }
        if self.thinning {
            // τ = K · γ̇^(1/n)
            let inv_n = Fix128::ONE / Fix128::from_int(i64::from(self.n_int));
            self.k * shear_rate_per_s.powf_pos(inv_n)
        } else {
            self.k * p
        }
    }

    /// Apparent viscosity `η = τ / γ̇` (Pa·s).
    #[must_use]
    pub fn apparent_viscosity(&self, shear_rate_per_s: Fix128) -> Fix128 {
        if shear_rate_per_s <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        self.stress(shear_rate_per_s) / shear_rate_per_s
    }
}

// ============================================================================
// Carreau
// ============================================================================

/// Carreau viscosity model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carreau {
    /// Zero-shear viscosity η_0 (Pa·s).
    pub eta_zero: Fix128,
    /// Infinite-shear viscosity η_∞ (Pa·s).
    pub eta_inf: Fix128,
    /// Time constant λ (s).
    pub lambda: Fix128,
    /// Exponent difference `(n-1)/2` rounded to an integer: `0` is Newtonian,
    /// `-1` is `n = −1` (steep shear-thinning), `1` is `n = 3`
    /// (shear-thickening). Fractional indices such as the polymer-melt
    /// `n = 0.4` are not representable here; use
    /// [`Carreau::viscosity_with_index`] for those.
    pub half_exponent: i32,
}

impl Carreau {
    /// Apparent viscosity at shear rate `γ̇` (Pa·s) with the exact fractional
    /// flow index `n` instead of the integer [`Carreau::half_exponent`]:
    /// `η = η_∞ + (η_0 − η_∞) (1 + (λγ̇)²)^((n−1)/2)`.
    ///
    /// `n < 1` is shear-thinning (the factor is `1 / inside^((1−n)/2)`),
    /// `n > 1` shear-thickening, `n = 1` Newtonian. Deterministic: the power
    /// is [`Fix128::powf_pos`] (24 fractional bits of the exponent).
    #[must_use]
    pub fn viscosity_with_index(&self, shear_rate_per_s: Fix128, n: Fix128) -> Fix128 {
        let l_gamma = self.lambda * shear_rate_per_s;
        let inside = Fix128::ONE + l_gamma * l_gamma;
        let half = Fix128::from_int(2);
        if n < Fix128::ONE {
            let factor = inside.powf_pos((Fix128::ONE - n) / half);
            if factor.is_zero() {
                return self.eta_inf;
            }
            self.eta_inf + (self.eta_zero - self.eta_inf) / factor
        } else {
            let factor = inside.powf_pos((n - Fix128::ONE) / half);
            self.eta_inf + (self.eta_zero - self.eta_inf) * factor
        }
    }

    /// Apparent viscosity at shear rate `γ̇` (Pa·s) using the integer
    /// [`Carreau::half_exponent`] (repeated multiplication, no root).
    #[must_use]
    pub fn viscosity(&self, shear_rate_per_s: Fix128) -> Fix128 {
        let l_gamma = self.lambda * shear_rate_per_s;
        let inside = Fix128::ONE + l_gamma * l_gamma;
        // Compute inside^half_exponent by repeated multiplication / division
        let mut factor = Fix128::ONE;
        let abs_exp = self.half_exponent.unsigned_abs();
        for _ in 0..abs_exp {
            factor = factor * inside;
        }
        if self.half_exponent < 0 {
            if factor.is_zero() {
                return self.eta_inf;
            }
            self.eta_inf + (self.eta_zero - self.eta_inf) / factor
        } else {
            self.eta_inf + (self.eta_zero - self.eta_inf) * factor
        }
    }
}

// ============================================================================
// Bingham plastic
// ============================================================================

/// Bingham plastic parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bingham {
    /// Yield stress τ_y (Pa).
    pub yield_stress: Fix128,
    /// Plastic viscosity μ_p (Pa·s).
    pub plastic_viscosity: Fix128,
}

impl Bingham {
    /// Shear stress at a given rate (Pa). Below yield, returns 0 (no flow).
    #[must_use]
    pub fn stress(&self, shear_rate_per_s: Fix128) -> Fix128 {
        if shear_rate_per_s <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        self.yield_stress + self.plastic_viscosity * shear_rate_per_s
    }

    /// Whether the fluid flows given an applied shear stress.
    #[must_use]
    pub fn flows_under_stress(&self, applied_stress_pa: Fix128) -> bool {
        applied_stress_pa > self.yield_stress
    }
}

// ============================================================================
// Herschel-Bulkley
// ============================================================================

/// Herschel-Bulkley model `τ = τ_y + K · γ̇ⁿ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HerschelBulkley {
    /// Yield stress τ_y (Pa).
    pub yield_stress: Fix128,
    /// Consistency K (Pa·s^n).
    pub k: Fix128,
    /// Integer exponent.
    pub n_int: u32,
}

impl HerschelBulkley {
    /// Shear stress (Pa).
    #[must_use]
    pub fn stress(&self, shear_rate_per_s: Fix128) -> Fix128 {
        if shear_rate_per_s <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        let mut p = Fix128::ONE;
        for _ in 0..self.n_int {
            p = p * shear_rate_per_s;
        }
        self.yield_stress + self.k * p
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn newtonian_stress_linear() {
        let n = PowerLaw::newtonian(Fix128::from_int(2));
        // τ = 2 · γ̇ → at γ̇=5, τ=10
        let t = n.stress(Fix128::from_int(5));
        assert_eq!(t, Fix128::from_int(10));
    }

    #[test]
    fn newtonian_viscosity_constant() {
        let n = PowerLaw::newtonian(Fix128::from_int(3));
        let v1 = n.apparent_viscosity(Fix128::from_int(1));
        let v2 = n.apparent_viscosity(Fix128::from_int(100));
        assert_eq!(v1, v2);
        assert_eq!(v1, Fix128::from_int(3));
    }

    #[test]
    fn shear_thickening_viscosity_grows_with_rate() {
        let n = PowerLaw::shear_thickening(Fix128::from_int(1), 2);
        let v_low = n.apparent_viscosity(Fix128::from_int(1));
        let v_high = n.apparent_viscosity(Fix128::from_int(10));
        assert!(v_high > v_low);
    }

    #[test]
    fn shear_thinning_viscosity_drops_with_rate() {
        let n = PowerLaw::shear_thinning(Fix128::from_int(10), 2);
        let v_low = n.apparent_viscosity(Fix128::from_int(1));
        let v_high = n.apparent_viscosity(Fix128::from_int(10));
        assert!(v_high < v_low);
    }

    #[test]
    fn power_law_zero_rate_zero_stress() {
        let n = PowerLaw::newtonian(Fix128::from_int(1));
        assert_eq!(n.stress(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn carreau_low_shear_returns_eta_zero() {
        // At γ̇ = 0, inside = 1, factor = 1 → η = η_∞ + (η_0 − η_∞) = η_0
        let c = Carreau {
            eta_zero: Fix128::from_int(100),
            eta_inf: Fix128::from_int(1),
            lambda: Fix128::from_int(1),
            half_exponent: -1,
        };
        assert_eq!(c.viscosity(Fix128::ZERO), Fix128::from_int(100));
    }

    #[test]
    fn carreau_high_shear_approaches_eta_inf() {
        // Very high γ̇ makes inside huge; for half_exponent negative,
        // (η_0 − η_∞)/factor → 0 → viscosity → η_∞.
        let c = Carreau {
            eta_zero: Fix128::from_int(100),
            eta_inf: Fix128::from_int(1),
            lambda: Fix128::from_int(1),
            half_exponent: -1,
        };
        let v = c.viscosity(Fix128::from_int(1000));
        // Should be close to 1 (η_∞)
        assert!(v < Fix128::from_int(2));
        assert!(v >= Fix128::from_int(1));
    }

    #[test]
    fn bingham_no_flow_below_yield() {
        let b = Bingham {
            yield_stress: Fix128::from_int(10),
            plastic_viscosity: Fix128::from_int(1),
        };
        assert!(!b.flows_under_stress(Fix128::from_int(5)));
        assert!(b.flows_under_stress(Fix128::from_int(20)));
    }

    #[test]
    fn bingham_stress_above_yield() {
        let b = Bingham {
            yield_stress: Fix128::from_int(10),
            plastic_viscosity: Fix128::from_int(1),
        };
        // At γ̇ = 5 → τ = 10 + 5 = 15
        assert_eq!(b.stress(Fix128::from_int(5)), Fix128::from_int(15));
    }

    #[test]
    fn bingham_zero_rate_zero_stress() {
        let b = Bingham {
            yield_stress: Fix128::from_int(10),
            plastic_viscosity: Fix128::from_int(1),
        };
        assert_eq!(b.stress(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn herschel_bulkley_reduces_to_bingham_when_n_1() {
        let hb = HerschelBulkley {
            yield_stress: Fix128::from_int(10),
            k: Fix128::from_int(2),
            n_int: 1,
        };
        // τ = 10 + 2·γ̇ → at γ̇=5, τ = 20
        assert_eq!(hb.stress(Fix128::from_int(5)), Fix128::from_int(20));
    }

    #[test]
    fn herschel_bulkley_stronger_shear_thickening() {
        let hb1 = HerschelBulkley {
            yield_stress: Fix128::from_int(5),
            k: Fix128::from_int(1),
            n_int: 1,
        };
        let hb2 = HerschelBulkley {
            yield_stress: Fix128::from_int(5),
            k: Fix128::from_int(1),
            n_int: 2,
        };
        // At γ̇ = 3: hb1 = 5+3=8, hb2 = 5+9=14
        assert!(hb2.stress(Fix128::from_int(3)) > hb1.stress(Fix128::from_int(3)));
    }

    #[test]
    fn carreau_half_exponent_zero_constant_viscosity() {
        // half_exponent = 0 → factor = 1 → η = η_∞ + (η_0 - η_∞) = η_0
        let c = Carreau {
            eta_zero: Fix128::from_int(50),
            eta_inf: Fix128::from_int(5),
            lambda: Fix128::from_int(1),
            half_exponent: 0,
        };
        assert_eq!(c.viscosity(Fix128::from_int(10)), Fix128::from_int(50));
    }
}
