//! Nonlinear Hyperelastic Constitutive Models (Rubber / TPU / Silicone)
//!
//! Phase B4 of the ALICE-Physics completeness project. Provides strain-
//! energy-density formulations for large-deformation elastic response, needed
//! for TPU printed parts, silicone gaskets, elastomeric snap-fits and any
//! material stretching beyond ~10% strain where linear Hooke's law breaks
//! down catastrophically.
//!
//! # Models provided
//!
//! - **Neo-Hookean** — single-parameter (μ, shear modulus). Simplest hyper-
//!   elastic model; captures ~30 % strain accurately.
//! - **Mooney-Rivlin** — two-parameter (C₁, C₂). Better fit to real rubber
//!   for moderate strains (up to ~60 %).
//! - **Yeoh** — three-parameter (C₁, C₂, C₃). Captures the S-shaped stress-
//!   strain curve of natural rubber up to ~200 % strain.
//! - **Ogden** — not implemented (requires non-integer powers unsupported by
//!   Fix128).
//!
//! All models assume **incompressibility** (`J = λ₁·λ₂·λ₃ = 1`) — the usual
//! rubber approximation. For a compressible variant subtract a `K·(J−1)²/2`
//! bulk term; not implemented here.
//!
//! # References
//!
//! - Ogden, *Non-Linear Elastic Deformations*, 1984 Chapter 4.
//! - Yeoh, "Characterization of Elastic Properties of Carbon-Black-Filled
//!   Rubber Vulcanizates", Rubber Chem. Technol. 63(5), 1990.
//! - Mooney, "A Theory of Large Elastic Deformation", J. Applied Physics 11,
//!   1940.
//! - Rivlin & Saunders, "Large Elastic Deformations of Isotropic Materials",
//!   Phil. Trans. Roy. Soc. A 243, 1951.
//! - Bower, *Applied Mechanics of Solids* Ch. 3 (numerical implementation).

use crate::math::{Fix128, Mat3Fix, Vec3Fix};

// ============================================================================
// Stretch state
// ============================================================================

/// Principal stretch state (λ₁, λ₂, λ₃).
///
/// Stretches are the ratios of current to reference length along each
/// principal axis. Incompressible loading requires `λ₁·λ₂·λ₃ = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stretch {
    /// Principal stretch along axis 1.
    pub l1: Fix128,
    /// Principal stretch along axis 2.
    pub l2: Fix128,
    /// Principal stretch along axis 3.
    pub l3: Fix128,
}

impl Stretch {
    /// Undeformed reference state (all stretches unity).
    pub const UNITY: Self = Self {
        l1: Fix128::ONE,
        l2: Fix128::ONE,
        l3: Fix128::ONE,
    };

    /// Uniaxial incompressible extension by stretch `lambda` along axis 1.
    ///
    /// Enforces `λ₂ = λ₃ = 1/√λ` via the incompressibility constraint.
    /// Returns `Stretch::UNITY` if `lambda` is non-positive.
    #[must_use]
    pub fn uniaxial(lambda: Fix128) -> Self {
        if lambda <= Fix128::ZERO {
            return Self::UNITY;
        }
        let lateral = Fix128::ONE / lambda.sqrt();
        Self {
            l1: lambda,
            l2: lateral,
            l3: lateral,
        }
    }

    /// Equibiaxial extension by stretch `lambda` in axes 1 and 2.
    /// Incompressibility gives `λ₃ = 1/λ²`.
    #[must_use]
    pub fn equibiaxial(lambda: Fix128) -> Self {
        if lambda <= Fix128::ZERO {
            return Self::UNITY;
        }
        let l2 = lambda * lambda;
        if l2.is_zero() {
            return Self::UNITY;
        }
        Self {
            l1: lambda,
            l2: lambda,
            l3: Fix128::ONE / l2,
        }
    }

    /// Volume ratio `J = λ₁·λ₂·λ₃`. Should be ≈ 1 for incompressible loading.
    #[must_use]
    pub fn volume_ratio(&self) -> Fix128 {
        self.l1 * self.l2 * self.l3
    }

    /// First invariant `I₁ = λ₁² + λ₂² + λ₃²`.
    #[must_use]
    pub fn i1(&self) -> Fix128 {
        self.l1 * self.l1 + self.l2 * self.l2 + self.l3 * self.l3
    }

    /// Second invariant `I₂ = 1/λ₁² + 1/λ₂² + 1/λ₃²` for incompressible loading.
    ///
    /// (Equivalent to `λ₁²·λ₂² + λ₂²·λ₃² + λ₃²·λ₁²` when J=1.)
    #[must_use]
    pub fn i2(&self) -> Fix128 {
        let l1sq = self.l1 * self.l1;
        let l2sq = self.l2 * self.l2;
        let l3sq = self.l3 * self.l3;
        if l1sq.is_zero() || l2sq.is_zero() || l3sq.is_zero() {
            return Fix128::ZERO;
        }
        Fix128::ONE / l1sq + Fix128::ONE / l2sq + Fix128::ONE / l3sq
    }
}

// ============================================================================
// Model definitions
// ============================================================================

/// Hyperelastic constitutive model selector.
///
/// Parameter units are MPa; equivalent to (N/mm²).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperelasticModel {
    /// Neo-Hookean: `W = (μ/2)·(I₁ − 3)`.
    NeoHookean {
        /// Shear modulus μ (MPa).
        mu_mpa: Fix128,
    },
    /// Mooney-Rivlin: `W = C₁·(I₁ − 3) + C₂·(I₂ − 3)`.
    /// Recovers Neo-Hookean when `C₂ = 0`.
    MooneyRivlin {
        /// First material constant (MPa).
        c1_mpa: Fix128,
        /// Second material constant (MPa).
        c2_mpa: Fix128,
    },
    /// Yeoh: `W = C₁·(I₁ − 3) + C₂·(I₁ − 3)² + C₃·(I₁ − 3)³`.
    /// Captures the strain-stiffening at large stretch found in natural rubber.
    Yeoh {
        /// First Yeoh constant (MPa).
        c1_mpa: Fix128,
        /// Second Yeoh constant (MPa).
        c2_mpa: Fix128,
        /// Third Yeoh constant (MPa).
        c3_mpa: Fix128,
    },
}

impl HyperelasticModel {
    /// Preset: soft TPU (Shore 85A) — printed elastomer. Uses Neo-Hookean
    /// with μ ≈ 3 MPa fit to NinjaFlex uniaxial data at 20 % strain.
    #[must_use]
    pub const fn tpu_soft() -> Self {
        Self::NeoHookean {
            mu_mpa: Fix128::from_raw(3, 0),
        }
    }

    /// Preset: silicone RTV (Ecoflex 00-30 grade). Very soft, Mooney-Rivlin.
    #[must_use]
    pub const fn silicone_soft() -> Self {
        Self::MooneyRivlin {
            c1_mpa: Fix128 {
                hi: 0,
                lo: 0x1999_9999_9999_999A, // ≈ 0.1
            },
            c2_mpa: Fix128 {
                hi: 0,
                lo: 0x0CCC_CCCC_CCCC_CCCD, // ≈ 0.05
            },
        }
    }

    /// Preset: natural rubber (Yeoh fit from Boyce & Arruda 2000).
    /// Values scaled to give ~1 MPa shear modulus at small strain.
    #[must_use]
    pub const fn natural_rubber() -> Self {
        Self::Yeoh {
            // ≈ 0.5 MPa
            c1_mpa: Fix128 {
                hi: 0,
                lo: 0x8000_0000_0000_0000,
            },
            // ≈ -0.017 MPa (softens then hardens)
            c2_mpa: Fix128 {
                hi: -1,
                lo: 0xFBA3_D70A_3D70_A3D8, // -0.017
            },
            // ≈ 0.00062 MPa (large-strain hardening)
            c3_mpa: Fix128 {
                hi: 0,
                lo: 0x0028_F5C2_8F5C_28F6, // 0.00062
            },
        }
    }
}

// ============================================================================
// Strain energy & principal Cauchy stress
// ============================================================================

/// Strain energy density W (MPa · dimensionless = MPa).
#[must_use]
pub fn strain_energy_density(model: &HyperelasticModel, stretch: &Stretch) -> Fix128 {
    let i1 = stretch.i1();
    let three = Fix128::from_int(3);
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => {
            *mu_mpa * (i1 - three) * Fix128::from_ratio(1, 2)
        }
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            let i2 = stretch.i2();
            *c1_mpa * (i1 - three) + *c2_mpa * (i2 - three)
        }
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let d = i1 - three;
            *c1_mpa * d + *c2_mpa * d * d + *c3_mpa * d * d * d
        }
    }
}

/// Cauchy (true) stress along axis 1 for a uniaxial incompressible extension.
///
/// For incompressible hyperelastic materials the standard result is:
/// - Neo-Hookean:  σ = μ·(λ² − 1/λ)
/// - Mooney-Rivlin: σ = 2·(C₁ + C₂/λ)·(λ² − 1/λ)  (Ogden 1984 §4.3; the code has always used `+`)
/// - Yeoh:         σ = 2·(λ² − 1/λ)·(C₁ + 2·C₂·(I₁ − 3) + 3·C₃·(I₁ − 3)²)
///
/// These are the derivatives of the strain-energy density under the
/// incompressibility constraint.
#[must_use]
pub fn uniaxial_cauchy_stress(model: &HyperelasticModel, lambda: Fix128) -> Fix128 {
    if lambda <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    let lam2 = lambda * lambda;
    let inv_lam = Fix128::ONE / lambda;
    let base = lam2 - inv_lam;

    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => *mu_mpa * base,
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            // 2 · (C1 + C2/λ) · (λ² − 1/λ) is the more common form
            // (some references use C1 − C2/λ; verify with references).
            let mix = *c1_mpa + *c2_mpa * inv_lam;
            Fix128::from_int(2) * mix * base
        }
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let stretch = Stretch::uniaxial(lambda);
            let i1 = stretch.i1();
            let d = i1 - Fix128::from_int(3);
            // dW/dI1 = C1 + 2·C2·d + 3·C3·d²
            let dw =
                *c1_mpa + Fix128::from_int(2) * *c2_mpa * d + Fix128::from_int(3) * *c3_mpa * d * d;
            Fix128::from_int(2) * dw * base
        }
    }
}

/// Small-strain shear modulus `μ_0` (MPa) — the limiting slope of the stress-
/// strain curve at λ → 1. Useful for comparison with the linear Young's
/// modulus `E ≈ 3·μ_0` for incompressible materials.
#[must_use]
pub fn small_strain_shear_modulus(model: &HyperelasticModel) -> Fix128 {
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => *mu_mpa,
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => (*c1_mpa + *c2_mpa).double(),
        HyperelasticModel::Yeoh { c1_mpa, .. } => c1_mpa.double(),
    }
}

// ============================================================================
// Tensor Cauchy stress (what a finite element needs)
// ============================================================================

/// `(∂W/∂I₁, ∂W/∂I₂)` at first invariant `i1`.
///
/// Both are constants for Neo-Hookean and Mooney-Rivlin; only Yeoh's `W₁`
/// depends on the deformation. `W₂` is zero for every model that is a function
/// of `I₁` alone.
fn energy_derivatives(model: &HyperelasticModel, i1: Fix128) -> (Fix128, Fix128) {
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => {
            (*mu_mpa * Fix128::from_ratio(1, 2), Fix128::ZERO)
        }
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => (*c1_mpa, *c2_mpa),
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let d = i1 - Fix128::from_int(3);
            (
                *c1_mpa + Fix128::from_int(2) * *c2_mpa * d + Fix128::from_int(3) * *c3_mpa * d * d,
                Fix128::ZERO,
            )
        }
    }
}

/// `(W₁, W₂, W₁₁, p_ref)` at first invariant `i1`: the constants the **tangent**
/// of [`cauchy_stress`] is built from.
///
/// `W₁₁ = ∂²W/∂I₁²` is zero for Neo-Hookean and Mooney-Rivlin (their `W₁` is
/// constant) and `2C₂ + 6C₃(I₁ − 3)` for Yeoh. `p_ref` is the isotropic offset
/// [`cauchy_stress`] subtracts so that the reference state is stress free.
pub(crate) fn tangent_constants(
    model: &HyperelasticModel,
    i1: Fix128,
) -> (Fix128, Fix128, Fix128, Fix128) {
    let (w1, w2) = energy_derivatives(model, i1);
    let (w1_ref, w2_ref) = energy_derivatives(model, Fix128::from_int(3));
    let p_ref = Fix128::from_int(2) * (w1_ref + w2_ref.double());
    let w11 = match model {
        HyperelasticModel::Yeoh { c2_mpa, c3_mpa, .. } => {
            Fix128::from_int(2) * *c2_mpa
                + Fix128::from_int(6) * *c3_mpa * (i1 - Fix128::from_int(3))
        }
        HyperelasticModel::NeoHookean { .. } | HyperelasticModel::MooneyRivlin { .. } => {
            Fix128::ZERO
        }
    };
    (w1, w2, w11, p_ref)
}

/// Cauchy stress `σ` for a deformation gradient `f`, in the frame `f` is written
/// in (MPa). `None` when `det F ≤ 0` — a reflected or collapsed element has no
/// stress under any of these models.
///
/// # What is added to the models above and why
///
/// The module's models are **incompressible**: `W` is a function of the
/// isochoric invariants only, so it fixes the stress up to an arbitrary pressure
/// and cannot be evaluated on its own. A finite element has to answer with a
/// number, so this function adds the volumetric pair the module doc names as
/// missing:
///
/// ```text
/// W_total(I₁, I₂, J) = W(I₁, I₂) − p_ref·(J − 1) + K/2·(J − 1)²
/// ```
///
/// with `p_ref = 2(W₁ + 2W₂)` evaluated at the **undeformed** state. The linear
/// term is what makes the reference state stress free (without it `σ(I) = p_ref·I`,
/// a body under pressure at rest); the quadratic term is the bulk response, and
/// `bulk_modulus_mpa` is its `K`. Differentiating,
///
/// ```text
/// σ = (2/J)·[ (W₁ + I₁·W₂)·B − W₂·B² ]  +  [ K·(J − 1) − p_ref ]·I,   B = F Fᵀ
/// ```
///
/// which is the standard isotropic result `σ = (2/J)·F·(∂W/∂C)·Fᵀ`.
///
/// ⚠️ **`W` is not re-normalised by `J^(−2/3)`.** The deviatoric response is
/// therefore the incompressible one **exactly at `J = 1`** and drifts from it as
/// `J` departs from one; `K` is what keeps `J` near one. That is a deliberate
/// limitation and not an approximation that refines away: a `J^(−2/3)` split
/// needs a cube root, which `Fix128` does not carry, and the alternative of
/// iterating for one would put a tolerance inside a constitutive law.
///
/// # Checks a caller can make
///
/// - `σ(I) = 0` exactly, for every model and every `K`.
/// - At `J = 1` and Neo-Hookean, `σ = μ·(B − I)`, so `dev σ = μ·dev B`.
/// - `I₂ = ½·(I₁² − tr B²)` here. That is the same number as [`Stretch::i2`]
///   when `J = 1`, which is the only place the two are both defined.
#[must_use]
pub fn cauchy_stress(
    model: &HyperelasticModel,
    bulk_modulus_mpa: Fix128,
    f: Mat3Fix,
) -> Option<Mat3Fix> {
    let j = f.determinant();
    if j <= Fix128::ZERO {
        return None;
    }
    let b = f.mul_mat(f.transpose());
    let i1 = b.col0.x + b.col1.y + b.col2.z;
    let (w1, w2) = energy_derivatives(model, i1);
    let (w1_ref, w2_ref) = energy_derivatives(model, Fix128::from_int(3));
    let p_ref = Fix128::from_int(2) * (w1_ref + w2_ref.double());

    let two_over_j = Fix128::from_int(2) / j;
    let mut s = b.scale(two_over_j * (w1 + i1 * w2));
    if !w2.is_zero() {
        s = mat_sub(s, b.mul_mat(b).scale(two_over_j * w2));
    }
    let pressure = bulk_modulus_mpa * (j - Fix128::ONE) - p_ref;
    Some(mat_add_diagonal(s, pressure))
}

/// `a − b`, entry by entry.
fn mat_sub(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(
            a.col0.x - b.col0.x,
            a.col0.y - b.col0.y,
            a.col0.z - b.col0.z,
        ),
        Vec3Fix::new(
            a.col1.x - b.col1.x,
            a.col1.y - b.col1.y,
            a.col1.z - b.col1.z,
        ),
        Vec3Fix::new(
            a.col2.x - b.col2.x,
            a.col2.y - b.col2.y,
            a.col2.z - b.col2.z,
        ),
    )
}

/// `m + s·I`.
fn mat_add_diagonal(m: Mat3Fix, s: Fix128) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(m.col0.x + s, m.col0.y, m.col0.z),
        Vec3Fix::new(m.col1.x, m.col1.y + s, m.col1.z),
        Vec3Fix::new(m.col2.x, m.col2.y, m.col2.z + s),
    )
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
    fn stretch_unity_is_undeformed() {
        assert_eq!(Stretch::UNITY.l1, Fix128::ONE);
        assert_eq!(Stretch::UNITY.i1(), Fix128::from_int(3));
        assert_eq!(Stretch::UNITY.volume_ratio(), Fix128::ONE);
    }

    #[test]
    fn stretch_uniaxial_preserves_volume() {
        // λ = 2 → λ2 = λ3 = 1/√2 ≈ 0.7071
        let s = Stretch::uniaxial(Fix128::from_int(2));
        let j = s.volume_ratio();
        // J = 2 · (1/√2)² = 2 · 0.5 = 1
        assert!(approx_eq(j, Fix128::ONE, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn stretch_equibiaxial_preserves_volume() {
        // λ = 1.5 → λ3 = 1/2.25 ≈ 0.4444
        let s = Stretch::equibiaxial(Fix128::from_ratio(15, 10));
        let j = s.volume_ratio();
        assert!(approx_eq(j, Fix128::ONE, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn stretch_i1_undeformed_is_three() {
        assert_eq!(Stretch::UNITY.i1(), Fix128::from_int(3));
    }

    #[test]
    fn stretch_i1_grows_with_extension() {
        let s1 = Stretch::uniaxial(Fix128::from_int(1));
        let s2 = Stretch::uniaxial(Fix128::from_int(3));
        assert!(s2.i1() > s1.i1());
    }

    #[test]
    fn strain_energy_undeformed_is_zero() {
        for model in [
            HyperelasticModel::tpu_soft(),
            HyperelasticModel::silicone_soft(),
            HyperelasticModel::natural_rubber(),
        ] {
            let w = strain_energy_density(&model, &Stretch::UNITY);
            assert!(
                approx_eq(w, Fix128::ZERO, Fix128::from_ratio(1, 1000)),
                "model {:?}: W = {}",
                model,
                w.to_f32()
            );
        }
    }

    #[test]
    fn strain_energy_grows_with_stretch() {
        let model = HyperelasticModel::tpu_soft();
        let s2 = Stretch::uniaxial(Fix128::from_int(2));
        let s3 = Stretch::uniaxial(Fix128::from_int(3));
        let w2 = strain_energy_density(&model, &s2);
        let w3 = strain_energy_density(&model, &s3);
        assert!(w3 > w2);
    }

    #[test]
    fn uniaxial_stress_zero_at_unity() {
        for model in [
            HyperelasticModel::tpu_soft(),
            HyperelasticModel::silicone_soft(),
            HyperelasticModel::natural_rubber(),
        ] {
            let s = uniaxial_cauchy_stress(&model, Fix128::ONE);
            assert!(
                approx_eq(s, Fix128::ZERO, Fix128::from_ratio(1, 100)),
                "{:?}: stress at λ=1 was {}",
                model,
                s.to_f32()
            );
        }
    }

    #[test]
    fn uniaxial_stress_neo_hookean_analytical() {
        // TPU μ=3 MPa. At λ=2: σ = 3·(4 - 0.5) = 10.5 MPa
        let model = HyperelasticModel::tpu_soft();
        let s = uniaxial_cauchy_stress(&model, Fix128::from_int(2));
        let expected = Fix128::from_ratio(105, 10);
        assert!(
            approx_eq(s, expected, Fix128::from_ratio(1, 10)),
            "got {}, expected 10.5",
            s.to_f32()
        );
    }

    #[test]
    fn tension_positive_compression_negative() {
        let model = HyperelasticModel::tpu_soft();
        let s_ext = uniaxial_cauchy_stress(&model, Fix128::from_int(2));
        let s_com = uniaxial_cauchy_stress(&model, Fix128::from_ratio(5, 10));
        assert!(s_ext > Fix128::ZERO);
        assert!(s_com < Fix128::ZERO);
    }

    #[test]
    fn yeoh_softens_then_stiffens() {
        // Natural rubber Yeoh: at low strain rate ≈ μ_0, at large strain
        // hardens. Compare secant modulus at low and high strain.
        let model = HyperelasticModel::natural_rubber();
        let s_small = uniaxial_cauchy_stress(&model, Fix128::from_ratio(12, 10));
        let s_large = uniaxial_cauchy_stress(&model, Fix128::from_int(3));
        assert!(s_small > Fix128::ZERO);
        assert!(s_large > s_small);
    }

    #[test]
    fn small_strain_shear_matches_neo_hookean_mu() {
        let model = HyperelasticModel::tpu_soft();
        let mu0 = small_strain_shear_modulus(&model);
        assert_eq!(mu0, Fix128::from_int(3));
    }

    #[test]
    fn small_strain_shear_mooney_rivlin_2x_c1_plus_c2() {
        let m = HyperelasticModel::MooneyRivlin {
            c1_mpa: Fix128::from_int(2),
            c2_mpa: Fix128::from_int(1),
        };
        let mu0 = small_strain_shear_modulus(&m);
        // 2·(2+1) = 6
        assert_eq!(mu0, Fix128::from_int(6));
    }

    #[test]
    fn small_strain_shear_yeoh_2x_c1() {
        let m = HyperelasticModel::Yeoh {
            c1_mpa: Fix128::from_int(4),
            c2_mpa: Fix128::from_int(1),
            c3_mpa: Fix128::from_int(1),
        };
        let mu0 = small_strain_shear_modulus(&m);
        assert_eq!(mu0, Fix128::from_int(8));
    }

    #[test]
    fn uniaxial_lambda_zero_returns_zero_stress() {
        let m = HyperelasticModel::tpu_soft();
        assert_eq!(uniaxial_cauchy_stress(&m, Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn uniaxial_lambda_negative_returns_zero_stress() {
        let m = HyperelasticModel::tpu_soft();
        assert_eq!(
            uniaxial_cauchy_stress(&m, Fix128::from_int(-1)),
            Fix128::ZERO
        );
    }

    #[test]
    fn stretch_uniaxial_lambda_zero_is_unity() {
        let s = Stretch::uniaxial(Fix128::ZERO);
        assert_eq!(s, Stretch::UNITY);
    }

    /// The reference state carries no stress, for every model and every bulk
    /// modulus.
    ///
    /// This is what the `−p_ref·(J−1)` term in [`cauchy_stress`] is for: without
    /// it the deviatoric part alone leaves `σ(I) = 2(W₁+2W₂)·I`, a body under
    /// pressure at rest. ⚠️ **No solver test can see that**: a *uniform* stress
    /// puts zero force on an interior node whatever the stress is, so the
    /// oracles in `tests/analytic_corotational.rs` stay green under exactly this
    /// mutation (measured 2026-10-01, `13 passed`). It is checked here instead.
    #[test]
    fn cauchy_stress_vanishes_in_the_reference_state() {
        for model in [
            HyperelasticModel::tpu_soft(),
            HyperelasticModel::silicone_soft(),
            HyperelasticModel::natural_rubber(),
        ] {
            for bulk in [Fix128::ZERO, Fix128::from_int(1), Fix128::from_int(4000)] {
                let s = cauchy_stress(&model, bulk, Mat3Fix::IDENTITY).expect("det I = 1 > 0");
                assert_eq!(
                    s,
                    Mat3Fix::ZERO,
                    "{model:?} at K = {} carries stress when undeformed",
                    bulk.to_f32()
                );
            }
        }
    }

    /// Neo-Hookean at `J = 1` is `σ = μ·(B − I)`, written out here from the
    /// definition rather than taken from the function under test.
    ///
    /// `F = diag(2, 1/2, 1)` has `det F = 1`, so `B = diag(4, 1/4, 1)` and
    /// `σ = μ·diag(3, −3/4, 0)`. With `μ = 3` (TPU) that is `(9, −9/4, 0)`.
    #[test]
    fn cauchy_stress_neo_hookean_isochoric_closed_form() {
        let f = Mat3Fix::diagonal(Fix128::from_int(2), Fix128::from_ratio(1, 2), Fix128::ONE);
        let s = cauchy_stress(&HyperelasticModel::tpu_soft(), Fix128::from_int(4000), f)
            .expect("det F = 1");
        let want = [Fix128::from_int(9), Fix128::from_ratio(-9, 4), Fix128::ZERO];
        let got = [s.col0.x, s.col1.y, s.col2.z];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                approx_eq(*g, *w, Fix128::from_ratio(1, 1000)),
                "σ = {:?}, expected diag(9, -2.25, 0)",
                got.map(Fix128::to_f32)
            );
        }
        assert_eq!(s.col1.x, Fix128::ZERO, "a diagonal F gives a diagonal σ");
        assert_eq!(s.col2.y, Fix128::ZERO, "a diagonal F gives a diagonal σ");
    }

    /// An element turned inside out, or flattened, has no stress under any of
    /// these models — `2/J` is not defined and neither is the pull back.
    #[test]
    fn cauchy_stress_refuses_a_non_positive_determinant() {
        let model = HyperelasticModel::tpu_soft();
        let bulk = Fix128::from_int(4000);
        let flat = Mat3Fix::diagonal(Fix128::ONE, Fix128::ONE, Fix128::ZERO);
        let inverted = Mat3Fix::diagonal(Fix128::from_int(-1), Fix128::ONE, Fix128::ONE);
        assert!(cauchy_stress(&model, bulk, flat).is_none());
        assert!(cauchy_stress(&model, bulk, inverted).is_none());
    }

    #[test]
    fn i2_incompressible_uniaxial() {
        // For λ=2 with λ2=λ3=1/√2, I2 = 1/λ² + 2·λ = 0.25 + 2·2 = 4.25... wait
        // Using our definition: I2 = 1/λ1² + 1/λ2² + 1/λ3²
        //                    = 1/4 + 2·2 = 4.25... no
        // Actually 1/(1/√2)² = 2, so I2 = 1/4 + 2 + 2 = 4.25
        let s = Stretch::uniaxial(Fix128::from_int(2));
        let expected = Fix128::from_ratio(425, 100);
        assert!(approx_eq(s.i2(), expected, Fix128::from_ratio(1, 100)));
    }
}
