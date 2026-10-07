//! Composite laminate failure criteria for orthotropic plies.
//!
//! Complements [`crate::laminate`] (ABD stiffness matrix, plane-stress
//! analysis) with the four in-plane failure-envelope evaluations that
//! structural engineers reach for when qualifying a CFRP / GFRP
//! layup:
//!
//! - **Tsai–Wu** — smooth interaction polynomial. Widely used because
//!   it produces a single failure index across all stress states.
//! - **Tsai–Hill** — Hoffman / von Mises analogue for orthotropic
//!   materials. Distinguishes tensile / compressive regimes via `X`
//!   and `Y` selection.
//! - **Hashin** — distinguishes four physical failure modes (fibre
//!   tension / compression, matrix tension / compression). Useful for
//!   progressive-damage post-processing.
//! - **Puck** — inter-fibre failure (IFF) mode A / B / C classifier
//!   with fibre-mode piggy-backed onto Hashin.
//!
//! All criteria return a scalar **failure index** `FI`:
//!
//! ```text
//! FI < 1  →  the ply is safe (margin = 1 − FI).
//! FI ≥ 1  →  the ply has failed under the criterion.
//! ```
//!
//! Only plane-stress (`σ₁, σ₂, τ₁₂` in the ply's principal directions)
//! is modelled; through-thickness `σ₃ / τ₁₃ / τ₂₃` is future work.

use crate::math::Fix128;

/// Material-level strength constants for a single ply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaminateStrengths {
    /// Longitudinal (fibre-direction) tensile strength `Xt` (MPa).
    pub xt: Fix128,
    /// Longitudinal compressive strength `Xc` (MPa, positive value).
    pub xc: Fix128,
    /// Transverse tensile strength `Yt` (MPa).
    pub yt: Fix128,
    /// Transverse compressive strength `Yc` (MPa, positive value).
    pub yc: Fix128,
    /// In-plane shear strength `S` (MPa).
    pub s: Fix128,
}

/// Error returned by [`LaminateStrengths::try_new`] when a strength constant is
/// not strictly positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidStrengthError {
    /// Name of the offending field (`"xt"`, `"xc"`, `"yt"`, `"yc"` or `"s"`).
    pub field: &'static str,
}

impl core::fmt::Display for InvalidStrengthError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "laminate strength `{}` must be strictly positive",
            self.field
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InvalidStrengthError {}

/// Failure index reported for a ply whose strength is zero or negative while the
/// matching stress component is loaded: the largest representable value, the
/// same saturation value `Fix128` division returns on overflow.
const FAILED_INDEX: Fix128 = Fix128::from_raw(i64::MAX, u64::MAX);

/// `a · c / b²`, or `None` when it does not fit Fix128. Computed as written
/// while `a · c` and `b²` fit, otherwise as `(a / b)(c / b)`, so a stress far
/// beyond its strength saturates instead of wrapping its square past `2⁶³`.
fn mul_over_sq(a: Fix128, c: Fix128, b: Fix128) -> Option<Fix128> {
    match (a.checked_mul(c), b.checked_mul(b)) {
        (Some(ac), Some(bb)) => Some(ac / bb),
        _ => (a / b).checked_mul(c / b),
    }
}

/// `a² / b²` (see [`mul_over_sq`]).
fn sq_over(a: Fix128, b: Fix128) -> Option<Fix128> {
    mul_over_sq(a, a, b)
}

/// `Σ terms`, or [`FAILED_INDEX`] when a term is missing (did not fit) or the
/// sum leaves Fix128: the criteria are positive quadratic forms for strengths
/// of a real ply, so a term too large to represent means an index far above 1.
/// (Tsai-Hill's form is indefinite when `Y > 2X` and only semi-definite at
/// `Y = 2X`, where it is `(σ₁ − σ₂/2)²/X²` and vanishes along `σ₁ = σ₂/2`: from
/// `Y ≥ 2X` the cross term can cancel an unrepresentable square, and a stress
/// whose exact index is below 1 is reported failed — e.g. `X = 1, Y = 3,
/// σ₂ = 2³³, σ₁ ≈ 1.09·10⁹`, exact index 0.5, cross term 9.4·10¹⁸ out of range,
/// or `X = 1, Y = 2, σ = (2.5·10⁹, 5·10⁹)`, exact index 0. The answer errs to
/// the safe side.)
fn index_sum(terms: &[Option<Fix128>]) -> Fix128 {
    let mut acc: i128 = 0;
    for t in terms {
        let Some(t) = t else {
            return FAILED_INDEX;
        };
        let raw = (i128::from(t.hi) << 64) | i128::from(t.lo);
        match acc.checked_add(raw) {
            Some(v) => acc = v,
            None => return FAILED_INDEX,
        }
    }
    Fix128::from_raw((acc >> 64) as i64, acc as u64)
}

impl LaminateStrengths {
    /// Validated constructor: every strength must be strictly positive.
    ///
    /// # Claims
    ///
    /// - Returns `Ok` only if `xt`, `xc`, `yt`, `yc` and `s` are all `> 0` (MPa).
    /// - Returns `Err` naming the first offending field in the order
    ///   `xt, xc, yt, yc, s` otherwise.
    ///
    /// # Errors
    ///
    /// [`InvalidStrengthError`] when any strength is zero or negative.
    pub fn try_new(
        xt: Fix128,
        xc: Fix128,
        yt: Fix128,
        yc: Fix128,
        s: Fix128,
    ) -> Result<Self, InvalidStrengthError> {
        for (field, v) in [("xt", xt), ("xc", xc), ("yt", yt), ("yc", yc), ("s", s)] {
            if v <= Fix128::ZERO {
                return Err(InvalidStrengthError { field });
            }
        }
        Ok(Self { xt, xc, yt, yc, s })
    }

    /// Which stress components act on a non-positive strength.
    ///
    /// Returns `(fibre, matrix, shear)`: a component is flagged when its stress
    /// is non-zero and a strength it is divided by is `<= 0` (closed form
    /// `FI -> infinity`, the ply is failed).
    fn degenerate(&self, stress: StressState) -> (bool, bool, bool) {
        let bad = |v: Fix128| v <= Fix128::ZERO;
        (
            !stress.sigma_1.is_zero() && (bad(self.xt) || bad(self.xc)),
            !stress.sigma_2.is_zero() && (bad(self.yt) || bad(self.yc)),
            !stress.tau_12.is_zero() && bad(self.s),
        )
    }

    /// Representative uni-directional CFRP (T300 / 5208 equivalent).
    #[must_use]
    pub fn cfrp_ud() -> Self {
        Self {
            xt: Fix128::from_int(1500),
            xc: Fix128::from_int(1500),
            yt: Fix128::from_int(40),
            yc: Fix128::from_int(246),
            s: Fix128::from_int(68),
        }
    }

    /// Representative E-glass / epoxy uni-directional laminate.
    #[must_use]
    pub fn gfrp_ud() -> Self {
        Self {
            xt: Fix128::from_int(1080),
            xc: Fix128::from_int(620),
            yt: Fix128::from_int(39),
            yc: Fix128::from_int(128),
            s: Fix128::from_int(89),
        }
    }
}

/// In-plane stress state expressed in the ply's principal coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StressState {
    /// Longitudinal (fibre direction) normal stress (MPa).
    pub sigma_1: Fix128,
    /// Transverse normal stress (MPa).
    pub sigma_2: Fix128,
    /// In-plane shear stress (MPa).
    pub tau_12: Fix128,
}

impl StressState {
    /// Zero stress state.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            sigma_1: Fix128::ZERO,
            sigma_2: Fix128::ZERO,
            tau_12: Fix128::ZERO,
        }
    }
}

/// Failure mode identifier returned by Hashin / Puck classifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureMode {
    /// No failure predicted.
    Safe,
    /// Fibre-direction tensile failure.
    FibreTension,
    /// Fibre-direction compressive failure (Euler / kink).
    FibreCompression,
    /// Matrix / transverse tensile failure.
    MatrixTension,
    /// Matrix / transverse compressive failure.
    MatrixCompression,
    /// Inter-fibre failure mode A (matrix cracking, transverse tension +
    /// shear).
    InterFibreA,
    /// Inter-fibre failure mode B (matrix cracking, small transverse
    /// compression + high shear).
    InterFibreB,
    /// Inter-fibre failure mode C (fibre-parallel matrix cracking, large
    /// transverse compression).
    InterFibreC,
}

/// Available in-plane failure criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureCriterion {
    /// Tsai–Wu smooth polynomial.
    TsaiWu,
    /// Tsai–Hill quadratic.
    TsaiHill,
    /// Hashin four-mode.
    Hashin,
    /// Puck fibre + IFF classifier.
    Puck,
}

/// Tsai–Wu failure index.
///
/// ```text
/// FI = F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2·F12·σ1·σ2
/// ```
///
/// with `F1 = 1/Xt − 1/Xc`, `F2 = 1/Yt − 1/Yc`, `F11 = 1/(Xt · Xc)`,
/// `F22 = 1/(Yt · Yc)`, `F66 = 1/S²`, and `F12 = -0.5 · √(F11 · F22)`
/// (the standard geometric-mean coupling term).
///
/// # Claims
///
/// - Unit: dimensionless; stresses and strengths in MPa.
/// - `FI >= 1` means failed, `FI < 1` means safe.
/// - A strength `<= 0` whose stress component is non-zero returns the largest
///   representable `Fix128` (failed), never `0`.
/// - Zero stress on a zero strength contributes nothing (no load, no failure).
#[must_use]
pub fn tsai_wu_failure_index(strengths: LaminateStrengths, stress: StressState) -> Fix128 {
    let (df, dm, ds) = strengths.degenerate(stress);
    if df || dm || ds {
        return FAILED_INDEX;
    }
    let s = strengths;
    let f1 = Fix128::ONE / s.xt - Fix128::ONE / s.xc;
    let f2 = Fix128::ONE / s.yt - Fix128::ONE / s.yc;
    let f11 = Fix128::ONE / (s.xt * s.xc);
    let f22 = Fix128::ONE / (s.yt * s.yc);
    let f66 = Fix128::ONE / (s.s * s.s);
    let f12 = -Fix128::from_ratio(1, 2) * (f11 * f22).sqrt();

    let StressState {
        sigma_1,
        sigma_2,
        tau_12,
    } = stress;

    // the products in the order written, each checked: a stress far beyond
    // the strengths would wrap a squared term past 2^63
    let term = |k: Fix128, a: Fix128, b: Fix128| k.checked_mul(a).and_then(|t| t.checked_mul(b));
    index_sum(&[
        f1.checked_mul(sigma_1),
        f2.checked_mul(sigma_2),
        term(f11, sigma_1, sigma_1),
        term(f22, sigma_2, sigma_2),
        term(f66, tau_12, tau_12),
        term(Fix128::from_int(2) * f12, sigma_1, sigma_2),
    ])
}

/// Tsai–Hill failure index (orthotropic von Mises).
///
/// Uses `X = Xt` when `σ1 ≥ 0`, else `X = Xc`; likewise for `Y`.
///
/// # Claims
///
/// - Unit: dimensionless; stresses and strengths in MPa.
/// - `FI >= 1` means failed, `FI < 1` means safe.
/// - A strength `<= 0` whose stress component is non-zero returns the largest
///   representable `Fix128` (failed), never `0`.
#[must_use]
pub fn tsai_hill_failure_index(strengths: LaminateStrengths, stress: StressState) -> Fix128 {
    let (df, dm, ds) = strengths.degenerate(stress);
    if df || dm || ds {
        return FAILED_INDEX;
    }
    let StressState {
        sigma_1,
        sigma_2,
        tau_12,
    } = stress;
    let x = if sigma_1 >= Fix128::ZERO {
        strengths.xt
    } else {
        strengths.xc
    };
    let y = if sigma_2 >= Fix128::ZERO {
        strengths.yt
    } else {
        strengths.yc
    };
    let s = strengths.s;

    let cross = mul_over_sq(sigma_1, sigma_2, x).map(|c| -c);
    index_sum(&[
        sq_over(sigma_1, x),
        cross,
        sq_over(sigma_2, y),
        sq_over(tau_12, s),
    ])
}

/// Hashin failure mode classifier for the in-plane 2-D form.
///
/// # Claims
///
/// - Returns `Safe` only when no criterion reaches `FI >= 1`.
/// - A strength `<= 0` whose stress component is non-zero is a failed ply:
///   fibre stress picks `FibreTension` / `FibreCompression` by the sign of
///   `sigma_1`; otherwise transverse or shear load picks `MatrixTension` /
///   `MatrixCompression` by the sign of `sigma_2`. Never `Safe`.
#[must_use]
pub fn hashin_failure_mode(strengths: LaminateStrengths, stress: StressState) -> FailureMode {
    let s = strengths;
    let StressState {
        sigma_1,
        sigma_2,
        tau_12,
    } = stress;
    let (df, dm, ds) = s.degenerate(stress);
    if df {
        return if sigma_1 >= Fix128::ZERO {
            FailureMode::FibreTension
        } else {
            FailureMode::FibreCompression
        };
    }
    if dm || ds {
        return if sigma_2 >= Fix128::ZERO {
            FailureMode::MatrixTension
        } else {
            FailureMode::MatrixCompression
        };
    }

    // Fibre tension: σ1 ≥ 0.
    if sigma_1 >= Fix128::ZERO {
        let fi = index_sum(&[sq_over(sigma_1, s.xt), sq_over(tau_12, s.s)]);
        if fi >= Fix128::ONE {
            return FailureMode::FibreTension;
        }
    } else {
        // Fibre compression: σ1 < 0.
        let fi = index_sum(&[sq_over(sigma_1, s.xc)]);
        if fi >= Fix128::ONE {
            return FailureMode::FibreCompression;
        }
    }

    // Matrix tension: σ2 ≥ 0.
    if sigma_2 >= Fix128::ZERO {
        let fi = index_sum(&[sq_over(sigma_2, s.yt), sq_over(tau_12, s.s)]);
        if fi >= Fix128::ONE {
            return FailureMode::MatrixTension;
        }
    } else {
        // Matrix compression: σ2 < 0.
        let yc_over_2s = s.yc / (Fix128::from_int(2) * s.s);
        let coeff = (yc_over_2s * yc_over_2s - Fix128::ONE) * sigma_2 / s.yc;
        let ratio = sigma_2 / (Fix128::from_int(2) * s.s);
        let fi = index_sum(&[ratio.checked_mul(ratio), Some(coeff), sq_over(tau_12, s.s)]);
        if fi >= Fix128::ONE {
            return FailureMode::MatrixCompression;
        }
    }

    FailureMode::Safe
}

/// Puck fibre-mode + inter-fibre-failure classifier (simplified 2-D).
///
/// Fibre modes reuse Hashin (identical closed form). IFF modes are
/// selected by the transverse stress and shear ratio; the returned
/// `FailureMode::InterFibreA / B / C` variants convey the physical
/// mode class expected under progressive damage.
///
/// # Claims
///
/// - Returns `Safe` only when neither a fibre mode nor an inter-fibre mode is
///   predicted.
/// - A strength `<= 0` whose stress component is non-zero is a failed ply:
///   fibre stress gives the Hashin fibre mode, `sigma_2 >= 0` gives
///   `InterFibreA`, shear on a zero `S` with `sigma_2 < 0` gives
///   `InterFibreB`, otherwise `InterFibreC`. Never `Safe`.
#[must_use]
pub fn puck_failure_mode(strengths: LaminateStrengths, stress: StressState) -> FailureMode {
    // Reuse Hashin for fibre modes.
    if let m @ (FailureMode::FibreTension | FailureMode::FibreCompression) =
        hashin_failure_mode(strengths, stress)
    {
        return m;
    }

    let (_, dm, ds) = strengths.degenerate(stress);
    if dm || ds {
        return if stress.sigma_2 >= Fix128::ZERO {
            FailureMode::InterFibreA
        } else if ds {
            FailureMode::InterFibreB
        } else {
            FailureMode::InterFibreC
        };
    }

    let StressState {
        sigma_2, tau_12, ..
    } = stress;
    let s = strengths;
    let ratio_normal = if sigma_2 >= Fix128::ZERO {
        sigma_2 / s.yt
    } else {
        sigma_2.abs() / s.yc
    };
    let ratio_shear = tau_12.abs() / s.s;

    let fi = index_sum(&[
        ratio_normal.checked_mul(ratio_normal),
        ratio_shear.checked_mul(ratio_shear),
    ]);
    if fi < Fix128::ONE {
        return FailureMode::Safe;
    }

    if sigma_2 >= Fix128::ZERO {
        FailureMode::InterFibreA
    } else if ratio_shear > ratio_normal {
        FailureMode::InterFibreB
    } else {
        FailureMode::InterFibreC
    }
}

/// Convenience dispatch — evaluate the requested criterion and return
/// the failure index (Tsai–Wu / Tsai–Hill) or a normalised 1.0 marker
/// (Hashin / Puck: 1.0 = failure detected, 0.0 = safe).
#[must_use]
pub fn failure_index(
    criterion: FailureCriterion,
    strengths: LaminateStrengths,
    stress: StressState,
) -> Fix128 {
    match criterion {
        FailureCriterion::TsaiWu => tsai_wu_failure_index(strengths, stress),
        FailureCriterion::TsaiHill => tsai_hill_failure_index(strengths, stress),
        FailureCriterion::Hashin => {
            if hashin_failure_mode(strengths, stress) == FailureMode::Safe {
                Fix128::ZERO
            } else {
                Fix128::ONE
            }
        }
        FailureCriterion::Puck => {
            if puck_failure_mode(strengths, stress) == FailureMode::Safe {
                Fix128::ZERO
            } else {
                Fix128::ONE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stress(s1: i64, s2: i64, t12: i64) -> StressState {
        StressState {
            sigma_1: Fix128::from_int(s1),
            sigma_2: Fix128::from_int(s2),
            tau_12: Fix128::from_int(t12),
        }
    }

    #[test]
    fn tsai_wu_zero_stress_is_safe() {
        let fi = tsai_wu_failure_index(LaminateStrengths::cfrp_ud(), StressState::zero());
        assert!(fi < Fix128::ONE);
    }

    #[test]
    fn tsai_wu_fibre_tension_at_xt_reaches_unity() {
        let strengths = LaminateStrengths::cfrp_ud();
        let fi = tsai_wu_failure_index(strengths, stress(1500, 0, 0));
        // At σ1 = Xt, FI should be close to 1.
        let diff = fi - Fix128::ONE;
        let mag = if diff < Fix128::ZERO { -diff } else { diff };
        assert!(mag < Fix128::from_ratio(1, 100), "FI {fi:?} not near 1");
    }

    #[test]
    fn tsai_hill_safe_below_all_limits() {
        let fi = tsai_hill_failure_index(LaminateStrengths::cfrp_ud(), stress(100, 5, 5));
        assert!(fi < Fix128::ONE);
    }

    #[test]
    fn tsai_hill_flags_pure_shear_at_s() {
        let strengths = LaminateStrengths::cfrp_ud();
        let fi = tsai_hill_failure_index(strengths, stress(0, 0, 68));
        // τ12 = S ⇒ FI = 1.
        let diff = fi - Fix128::ONE;
        let mag = if diff < Fix128::ZERO { -diff } else { diff };
        assert!(mag < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn hashin_reports_fibre_tension() {
        let mode = hashin_failure_mode(LaminateStrengths::cfrp_ud(), stress(2000, 0, 0));
        assert_eq!(mode, FailureMode::FibreTension);
    }

    #[test]
    fn hashin_reports_fibre_compression() {
        let mode = hashin_failure_mode(LaminateStrengths::cfrp_ud(), stress(-2000, 0, 0));
        assert_eq!(mode, FailureMode::FibreCompression);
    }

    #[test]
    fn hashin_reports_matrix_tension() {
        let mode = hashin_failure_mode(LaminateStrengths::cfrp_ud(), stress(0, 100, 0));
        assert_eq!(mode, FailureMode::MatrixTension);
    }

    #[test]
    fn hashin_reports_matrix_compression() {
        let mode = hashin_failure_mode(LaminateStrengths::cfrp_ud(), stress(0, -500, 0));
        assert_eq!(mode, FailureMode::MatrixCompression);
    }

    #[test]
    fn hashin_safe_under_low_stress() {
        let mode = hashin_failure_mode(LaminateStrengths::cfrp_ud(), stress(500, 10, 20));
        assert_eq!(mode, FailureMode::Safe);
    }

    #[test]
    fn puck_reports_interfibre_a() {
        let mode = puck_failure_mode(LaminateStrengths::cfrp_ud(), stress(0, 100, 0));
        // High σ2 tension + low shear → mode A.
        assert_eq!(mode, FailureMode::InterFibreA);
    }

    #[test]
    fn puck_reports_interfibre_c_under_compression() {
        let mode = puck_failure_mode(LaminateStrengths::cfrp_ud(), stress(0, -400, 20));
        assert_eq!(mode, FailureMode::InterFibreC);
    }

    #[test]
    fn dispatch_returns_expected_shape() {
        let strengths = LaminateStrengths::cfrp_ud();
        let s = stress(500, 20, 20);
        let fi = failure_index(FailureCriterion::TsaiWu, strengths, s);
        assert!(fi >= Fix128::ZERO);
        let hashin_fi = failure_index(FailureCriterion::Hashin, strengths, s);
        assert!(hashin_fi == Fix128::ZERO || hashin_fi == Fix128::ONE);
    }

    #[test]
    fn presets_cfrp_and_gfrp_are_distinct() {
        let cfrp = LaminateStrengths::cfrp_ud();
        let gfrp = LaminateStrengths::gfrp_ud();
        assert_ne!(cfrp.xt, gfrp.xt);
        assert_ne!(cfrp.yt, gfrp.yt);
    }
}
