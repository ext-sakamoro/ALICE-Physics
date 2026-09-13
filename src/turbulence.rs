//! Turbulence Closure Models — Smagorinsky LES + k-ε + k-ω
//!
//! Phase G1 of the ALICE-Physics completeness project. Provides the three
//! most common Reynolds-Averaged (RANS) and Large Eddy Simulation (LES)
//! turbulence closures — each computes an effective **eddy viscosity**
//! `ν_t` (m²/s) that augments the molecular viscosity in the momentum
//! equations, plus the transport equations for `k`, `ε`, `ω` where used.
//!
//! # Closures
//!
//! - **Smagorinsky** (LES): `ν_t = (C_s · Δ)² · |S̄|` where `|S̄|` is the
//!   magnitude of the resolved strain rate. Zero-equation SGS model.
//! - **k-ε** (Launder & Spalding 1974): `ν_t = C_μ · k² / ε`. Requires
//!   two transport equations. Standard for industrial CFD.
//! - **k-ω** (Wilcox 1988): `ν_t = k / ω`. Better wall behaviour than
//!   k-ε; used in aerospace and turbomachinery.
//!
//! # Constants
//!
//! Universal fit-to-experiment values (Pope, *Turbulent Flows* Table 10.1):
//! - `C_s = 0.17`, `C_μ = 0.09`, `σ_k = 1.0`, `σ_ε = 1.3`,
//!   `C_{ε1} = 1.44`, `C_{ε2} = 1.92`, `β* = 0.09`, `β = 3/40`.
//!
//! # References
//!
//! - Smagorinsky, "General circulation experiments with the primitive
//!   equations", Monthly Weather Review 91, 1963.
//! - Launder & Spalding, "The numerical computation of turbulent flows",
//!   Comp. Meth. Appl. Mech. Eng. 3, 1974 (standard k-ε).
//! - Wilcox, "Reassessment of the scale-determining equation for advanced
//!   turbulence models", AIAA J. 26, 1988 (k-ω).
//! - Pope, *Turbulent Flows*, Cambridge 2000.
//!
//! # Integration status
//!
//! Only the Smagorinsky path (`smagorinsky_eddy_viscosity`,
//! `strain_rate_magnitude`, `SMAGORINSKY_CS`) is currently wired into
//! `cfd_solver.rs`. The k-ε / k-ω transport state, dynamic Smagorinsky
//! coefficient, and wall functions are reserved crate-internal API
//! awaiting downstream integration; module-level `#![allow(dead_code)]`
//! documents this state.

#![allow(dead_code)] // Reserved RANS/wall-function API — awaiting cfd_solver integration

use crate::math::Fix128;

// ============================================================================
// Constants
// ============================================================================

/// Smagorinsky constant `C_s`.
pub const SMAGORINSKY_CS: Fix128 = Fix128 {
    hi: 0,
    lo: 0x2B85_1EB8_51EB_851F, // ≈ 0.17
};

/// k-ε model constants (Launder & Spalding 1974, crate-internal).
pub(crate) const KE_C_MU: Fix128 = Fix128 {
    hi: 0,
    lo: 0x1707_5F6F_D21F_F2E5, // ≈ 0.09
};
/// k-ε turbulent Prandtl number for k (crate-internal).
pub(crate) const KE_SIGMA_K: Fix128 = Fix128::ONE;
/// k-ε turbulent Prandtl number for ε (≈ 1.3, crate-internal).
pub(crate) const KE_SIGMA_EPS: Fix128 = Fix128 {
    hi: 1,
    lo: 0x4CCC_CCCC_CCCC_CCCD,
};
/// k-ε constant C_{ε1} (≈ 1.44, crate-internal).
pub(crate) const KE_C1_EPS: Fix128 = Fix128 {
    hi: 1,
    lo: 0x70A3_D70A_3D70_A3D7,
};
/// k-ε constant C_{ε2} (≈ 1.92, crate-internal).
pub(crate) const KE_C2_EPS: Fix128 = Fix128 {
    hi: 1,
    lo: 0xEB85_1EB8_51EB_851F,
};

/// k-ω model constant β* (= 0.09, crate-internal).
pub(crate) const KW_BETA_STAR: Fix128 = KE_C_MU;
/// k-ω model constant β (= 3/40 = 0.075, crate-internal).
pub(crate) const KW_BETA: Fix128 = Fix128 {
    hi: 0,
    lo: 0x1333_3333_3333_3333,
};

// ============================================================================
// Smagorinsky
// ============================================================================

/// Smagorinsky eddy viscosity: `ν_t = (C_s · Δ)² · |S̄|`.
///
/// - `filter_width_m`: cell size Δ.
/// - `strain_rate_magnitude`: `|S̄| = √(2·S_ij·S_ij)`.
#[must_use]
pub fn smagorinsky_eddy_viscosity(filter_width_m: Fix128, strain_rate_magnitude: Fix128) -> Fix128 {
    let cs_delta = SMAGORINSKY_CS * filter_width_m;
    cs_delta * cs_delta * strain_rate_magnitude
}

/// Strain-rate magnitude from a symmetric 3×3 strain tensor S_ij:
/// `|S̄| = √(2·(S_11² + S_22² + S_33² + 2·(S_12² + S_13² + S_23²)))`.
#[must_use]
pub fn strain_rate_magnitude(
    s11: Fix128,
    s22: Fix128,
    s33: Fix128,
    s12: Fix128,
    s13: Fix128,
    s23: Fix128,
) -> Fix128 {
    let diag = s11 * s11 + s22 * s22 + s33 * s33;
    let off = Fix128::from_int(2) * (s12 * s12 + s13 * s13 + s23 * s23);
    (Fix128::from_int(2) * (diag + off)).sqrt()
}

// ============================================================================
// k-ε
// ============================================================================

/// k-ε turbulence state at a single point (crate-internal, not yet wired to CFD solver).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct KEpsilonState {
    /// Turbulent kinetic energy `k` (m²/s²).
    pub(crate) k: Fix128,
    /// Turbulent dissipation rate `ε` (m²/s³).
    pub(crate) epsilon: Fix128,
}

#[allow(dead_code)] // Reserved RANS state — unused pending k-ε integration into cfd_solver
impl KEpsilonState {
    /// Eddy viscosity `ν_t = C_μ · k² / ε` (m²/s).
    #[must_use]
    pub(crate) fn eddy_viscosity(&self) -> Fix128 {
        if self.epsilon.is_zero() {
            return Fix128::ZERO;
        }
        KE_C_MU * self.k * self.k / self.epsilon
    }

    /// One explicit Euler step of the k transport equation with production
    /// term `P_k` (m²/s³) and no diffusion or convection (point model):
    ///
    /// `dk/dt = P_k − ε`
    pub(crate) fn advance_k(&mut self, production_k: Fix128, dt_s: Fix128) {
        self.k = self.k + (production_k - self.epsilon) * dt_s;
        if self.k < Fix128::ZERO {
            self.k = Fix128::ZERO;
        }
    }

    /// One explicit Euler step of the ε transport equation (point model):
    ///
    /// `dε/dt = (ε/k) · (C_{ε1}·P_k − C_{ε2}·ε)`
    pub(crate) fn advance_epsilon(&mut self, production_k: Fix128, dt_s: Fix128) {
        if self.k.is_zero() {
            return;
        }
        let factor = self.epsilon / self.k;
        let d_eps = factor * (KE_C1_EPS * production_k - KE_C2_EPS * self.epsilon);
        self.epsilon = self.epsilon + d_eps * dt_s;
        if self.epsilon < Fix128::ZERO {
            self.epsilon = Fix128::ZERO;
        }
    }
}

// ============================================================================
// k-ω
// ============================================================================

/// k-ω turbulence state at a single point (crate-internal, not yet wired to CFD solver).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct KOmegaState {
    /// Turbulent kinetic energy `k` (m²/s²).
    pub(crate) k: Fix128,
    /// Specific dissipation rate `ω = ε/(β*·k)` (1/s).
    pub(crate) omega: Fix128,
}

#[allow(dead_code)] // Reserved RANS state — unused pending k-ω integration into cfd_solver
impl KOmegaState {
    /// Eddy viscosity `ν_t = k / ω` (m²/s).
    #[must_use]
    pub(crate) fn eddy_viscosity(&self) -> Fix128 {
        if self.omega.is_zero() {
            return Fix128::ZERO;
        }
        self.k / self.omega
    }

    /// Convert k-ε state to k-ω via `ω = ε / (β* · k)`.
    #[must_use]
    pub(crate) fn from_k_epsilon(state: &KEpsilonState) -> Self {
        let omega = if state.k.is_zero() {
            Fix128::ZERO
        } else {
            state.epsilon / (KW_BETA_STAR * state.k)
        };
        Self { k: state.k, omega }
    }
}

// ============================================================================
// Dynamic Smagorinsky (Session 3 I10 upgrade)
// ============================================================================

/// Compute a **dynamic** Smagorinsky coefficient `C_s(x,t)` from local flow
/// state using a Germano-style estimator. Returns a clamped value in
/// `[0.05, 0.25]` to avoid unphysical excursions.
///
/// Simplified form suitable for cell-local flows without a full test-filter
/// grid: use the ratio of test-filter strain to grid-filter strain as a
/// proxy for the Germano identity. `strain_grid` is the resolved-scale
/// magnitude, `strain_test` is an approximate `2·Δ` filter (average of
/// neighbouring cells; supplied by caller).
///
/// `C_s² = ⟨L·M⟩ / (2·⟨M·M⟩ + ε)` where `L` and `M` are the Leonard /
/// mixed tensors; here we approximate the isotropic invariants.
#[must_use]
pub(crate) fn dynamic_smagorinsky_cs(strain_grid: Fix128, strain_test: Fix128) -> Fix128 {
    let min_cs = Fix128::from_ratio(5, 100);
    let max_cs = Fix128::from_ratio(25, 100);
    if strain_grid.is_zero() || strain_test.is_zero() {
        return SMAGORINSKY_CS;
    }
    // ratio ≈ (strain_test / strain_grid); high ratio → less SGS energy
    let ratio = strain_test / strain_grid;
    // Simple mapping: C_s ∝ ratio, clamped
    let unclamped = SMAGORINSKY_CS * ratio;
    if unclamped < min_cs {
        min_cs
    } else if unclamped > max_cs {
        max_cs
    } else {
        unclamped
    }
}

// ============================================================================
// Wall functions (Session 3 I8 upgrade)
// ============================================================================

/// von Kármán constant κ (≈ 0.41) for log-law wall function (crate-internal).
pub(crate) const VON_KARMAN: Fix128 = Fix128 {
    hi: 0,
    lo: 0x68F5_C28F_5C28_F5C3,
};
/// Additive log-law constant B (≈ 5.5) for smooth walls (crate-internal).
pub(crate) const LOG_LAW_B: Fix128 = Fix128 {
    hi: 5,
    lo: 0x8000_0000_0000_0000,
};

/// y+ boundary between the viscous sublayer and the log-law region (crate-internal).
pub(crate) const Y_PLUS_TRANSITION: Fix128 = Fix128 { hi: 11, lo: 0 };

/// Non-dimensional wall distance `y+ = ρ·u_τ·y / μ`.
#[must_use]
pub(crate) fn y_plus(
    density: Fix128,
    u_tau: Fix128,
    y_m: Fix128,
    dynamic_viscosity_pas: Fix128,
) -> Fix128 {
    if dynamic_viscosity_pas.is_zero() {
        return Fix128::ZERO;
    }
    density * u_tau * y_m / dynamic_viscosity_pas
}

/// Deterministic natural log via range reduction to `[1, 2)` + atanh series.
#[must_use]
fn ln_fix(x: Fix128) -> Fix128 {
    if x <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    let ln2 = Fix128 {
        hi: 0,
        lo: 0xB172_17F7_D1CF_79AC,
    };
    let mut m = x;
    let mut k: i64 = 0;
    while m >= Fix128::from_int(2) {
        m = m.half();
        k += 1;
    }
    while m < Fix128::ONE {
        m = m.double();
        k -= 1;
    }
    let t = (m - Fix128::ONE) / (m + Fix128::ONE);
    let t2 = t * t;
    let mut term = t;
    let mut sum = Fix128::ZERO;
    for n in 0..16u32 {
        let denom = Fix128::from_int(i64::from(2 * n + 1));
        sum = sum + term / denom;
        term = term * t2;
    }
    Fix128::from_int(k) * ln2 + sum.double()
}

/// Dimensionless velocity `u+ = u / u_τ` from the universal wall profile.
/// - `y+ < 11.63` → `u+ = y+` (viscous sublayer).
/// - Otherwise    → `u+ = (1/κ)·ln(y+) + B`  (log law).
#[must_use]
pub(crate) fn u_plus(y_plus_val: Fix128) -> Fix128 {
    if y_plus_val <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    if y_plus_val < Y_PLUS_TRANSITION {
        return y_plus_val;
    }
    let ln_yp = ln_fix(y_plus_val);
    ln_yp / VON_KARMAN + LOG_LAW_B
}

/// Wall-consistent turbulent kinetic energy and dissipation in the log
/// region. Given friction velocity `u_τ` (m/s) and wall distance `y` (m):
///
///   k = u_τ² / √C_μ
///   ε = u_τ³ / (κ · y)
#[must_use]
pub(crate) fn wall_k_epsilon(u_tau_m_per_s: Fix128, y_m: Fix128) -> (Fix128, Fix128) {
    if y_m.is_zero() || u_tau_m_per_s.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let sqrt_cmu = KE_C_MU.sqrt();
    let k = if sqrt_cmu.is_zero() {
        Fix128::ZERO
    } else {
        u_tau_m_per_s * u_tau_m_per_s / sqrt_cmu
    };
    let u_tau_cube = u_tau_m_per_s * u_tau_m_per_s * u_tau_m_per_s;
    let eps = u_tau_cube / (VON_KARMAN * y_m);
    (k, eps)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn smagorinsky_cs_constant() {
        // Should be ≈ 0.17
        assert!(approx_eq(
            SMAGORINSKY_CS,
            Fix128::from_ratio(17, 100),
            Fix128::from_ratio(1, 1000)
        ));
    }

    #[test]
    fn ke_c_mu_constant() {
        // Should be ≈ 0.09
        assert!(approx_eq(
            KE_C_MU,
            Fix128::from_ratio(9, 100),
            Fix128::from_ratio(1, 1000)
        ));
    }

    #[test]
    fn smagorinsky_zero_strain_zero_viscosity() {
        let nu = smagorinsky_eddy_viscosity(Fix128::from_ratio(1, 100), Fix128::ZERO);
        assert_eq!(nu, Fix128::ZERO);
    }

    #[test]
    fn smagorinsky_scales_with_delta_squared() {
        let s = Fix128::from_int(10);
        let nu1 = smagorinsky_eddy_viscosity(Fix128::from_ratio(1, 100), s);
        let nu2 = smagorinsky_eddy_viscosity(Fix128::from_ratio(2, 100), s);
        // Doubling Δ → 4× ν_t
        let ratio = nu2 / nu1;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn strain_rate_isotropic_diagonal() {
        // Pure diagonal S_11 = 1: |S̄| = √(2·1) = √2
        let s = strain_rate_magnitude(
            Fix128::ONE,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
        );
        let expected = Fix128::from_int(2).sqrt();
        assert!(approx_eq(s, expected, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn strain_rate_zero_tensor() {
        let s = strain_rate_magnitude(
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
        );
        assert_eq!(s, Fix128::ZERO);
    }

    #[test]
    fn ke_eddy_viscosity_formula() {
        let st = KEpsilonState {
            k: Fix128::ONE,
            epsilon: Fix128::from_ratio(9, 100),
        };
        // ν_t = 0.09 · 1 / 0.09 = 1
        let nu = st.eddy_viscosity();
        assert!(approx_eq(nu, Fix128::ONE, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn ke_zero_epsilon_zero_viscosity() {
        let st = KEpsilonState {
            k: Fix128::ONE,
            epsilon: Fix128::ZERO,
        };
        assert_eq!(st.eddy_viscosity(), Fix128::ZERO);
    }

    #[test]
    fn ke_k_transport_reduces_k_no_production() {
        let mut st = KEpsilonState {
            k: Fix128::from_int(10),
            epsilon: Fix128::from_int(1),
        };
        // No production, ε=1 → dk = -1·dt = -0.1 at dt=0.1
        st.advance_k(Fix128::ZERO, Fix128::from_ratio(1, 10));
        assert!(st.k < Fix128::from_int(10));
        assert!(st.k > Fix128::from_int(9));
    }

    #[test]
    fn ke_k_never_goes_negative() {
        let mut st = KEpsilonState {
            k: Fix128::ONE,
            epsilon: Fix128::from_int(100),
        };
        st.advance_k(Fix128::ZERO, Fix128::ONE);
        assert!(st.k >= Fix128::ZERO);
    }

    #[test]
    fn ke_epsilon_transport_zero_k_stays() {
        let mut st = KEpsilonState {
            k: Fix128::ZERO,
            epsilon: Fix128::from_int(1),
        };
        let eps_before = st.epsilon;
        st.advance_epsilon(Fix128::ONE, Fix128::ONE);
        assert_eq!(st.epsilon, eps_before);
    }

    #[test]
    fn kw_from_ke_conversion() {
        // ε = 0.09, k = 1 → ω = 0.09 / (0.09·1) = 1
        let ke = KEpsilonState {
            k: Fix128::ONE,
            epsilon: Fix128::from_ratio(9, 100),
        };
        let kw = KOmegaState::from_k_epsilon(&ke);
        assert_eq!(kw.k, ke.k);
        assert!(approx_eq(kw.omega, Fix128::ONE, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn kw_eddy_viscosity_reduces_when_omega_high() {
        // Higher ω → smaller ν_t
        let low = KOmegaState {
            k: Fix128::ONE,
            omega: Fix128::ONE,
        };
        let high = KOmegaState {
            k: Fix128::ONE,
            omega: Fix128::from_int(10),
        };
        assert!(low.eddy_viscosity() > high.eddy_viscosity());
    }

    #[test]
    fn kw_zero_omega_zero_viscosity() {
        let st = KOmegaState {
            k: Fix128::ONE,
            omega: Fix128::ZERO,
        };
        assert_eq!(st.eddy_viscosity(), Fix128::ZERO);
    }

    #[test]
    fn von_karman_and_b_constants() {
        assert!(approx_eq(
            VON_KARMAN,
            Fix128::from_ratio(41, 100),
            Fix128::from_ratio(1, 100)
        ));
        assert!(approx_eq(
            LOG_LAW_B,
            Fix128::from_ratio(55, 10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn y_plus_zero_viscosity_returns_zero() {
        let v = y_plus(
            Fix128::from_int(1000),
            Fix128::ONE,
            Fix128::from_ratio(1, 100),
            Fix128::ZERO,
        );
        assert_eq!(v, Fix128::ZERO);
    }

    #[test]
    fn u_plus_sublayer_linear() {
        // y+ = 5 → u+ = 5 (viscous sublayer)
        let up = u_plus(Fix128::from_int(5));
        assert_eq!(up, Fix128::from_int(5));
    }

    #[test]
    fn u_plus_log_region_increases_slowly() {
        // y+ = 100 → u+ = (1/0.41)·ln(100) + 5.5 ≈ 2.44·4.605 + 5.5 ≈ 16.7
        let up = u_plus(Fix128::from_int(100));
        assert!(approx_eq(
            up,
            Fix128::from_ratio(167, 10),
            Fix128::from_int(1)
        ));
    }

    #[test]
    fn u_plus_zero_returns_zero() {
        assert_eq!(u_plus(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn wall_k_epsilon_positive() {
        let (k, eps) = wall_k_epsilon(Fix128::from_ratio(5, 10), Fix128::from_ratio(1, 100));
        assert!(k > Fix128::ZERO);
        assert!(eps > Fix128::ZERO);
    }

    #[test]
    fn dynamic_cs_default_when_strain_zero() {
        let c = dynamic_smagorinsky_cs(Fix128::ZERO, Fix128::from_int(1));
        assert_eq!(c, SMAGORINSKY_CS);
    }

    #[test]
    fn dynamic_cs_clamped_to_range() {
        // High ratio should clamp at 0.25
        let c = dynamic_smagorinsky_cs(Fix128::ONE, Fix128::from_int(100));
        assert!(c <= Fix128::from_ratio(25, 100));
        // Low ratio should clamp at 0.05
        let c2 = dynamic_smagorinsky_cs(Fix128::from_int(100), Fix128::ONE);
        assert!(c2 >= Fix128::from_ratio(5, 100));
    }

    #[test]
    fn wall_k_epsilon_zero_y_returns_zero() {
        let (k, eps) = wall_k_epsilon(Fix128::ONE, Fix128::ZERO);
        assert_eq!(k, Fix128::ZERO);
        assert_eq!(eps, Fix128::ZERO);
    }
}
