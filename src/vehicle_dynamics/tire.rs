//! Tyre force models: slip in, contact-patch force out.
//!
//! # Conventions (shared with [`super::DynamicVehicle`])
//!
//! - Wheel frame: `x` = wheel heading projected onto the ground plane,
//!   `y` = `normal × x` (points to the wheel's right when the normal is up).
//! - `v_x`, `v_y`: contact-point velocity components in that frame.
//! - Slip ratio `κ = (ω r − v_x) / max(|v_x|, v_floor)`: positive while
//!   driving (wheel surface faster than the road), `−1` for a locked wheel
//!   rolling forward.
//! - Lateral slip `tan α = v_y / max(|v_x|, v_floor)`: positive when the
//!   contact point slides to the right. The lateral force opposes it.
//! - Output force components are in the same frame: `longitudinal` along `x`,
//!   `lateral` along `y`, in newtons.
//!
//! Every model saturates on the friction ellipse with semi-axes
//! `μ_x F_z` and `μ_y F_z` taken from [`TireInput::grip`].

use crate::anisotropic_friction::AnisotropicFriction;
use crate::math::Fix128;

/// Slip state and load of one contact patch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireInput {
    /// Slip ratio `κ` (dimensionless, see module conventions).
    pub slip_ratio: Fix128,
    /// Lateral slip `tan α` (dimensionless, see module conventions).
    pub slip_tan_alpha: Fix128,
    /// Normal load `F_z` (N). Non-positive load gives a zero force.
    pub normal_load: Fix128,
    /// Road grip at this contact (already scaled by surface and weather).
    /// `*_static` is the peak (adhesion) coefficient, `*_kinetic` the
    /// full-sliding coefficient.
    pub grip: AnisotropicFriction,
}

/// Contact-patch force in the wheel frame (N).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireForce {
    /// Force along the wheel heading.
    pub longitudinal: Fix128,
    /// Force along the wheel's lateral axis.
    pub lateral: Fix128,
}

/// Brush (Fiala-type) model with combined slip.
///
/// Contract:
/// - slope at zero slip: `∂F_x/∂κ = longitudinal_stiffness`,
///   `∂F_y/∂tan α = −cornering_stiffness` (both independent of `F_z`)
/// - full sliding (locked wheel, `κ = −1`, `tan α = 0`): `F_x = −μ_x,kinetic F_z`
///   exactly
/// - `|F|` never leaves the friction ellipse built from the static coefficients
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrushTire {
    /// `C_κ` (N per unit slip ratio).
    pub longitudinal_stiffness: Fix128,
    /// `C_α` (N per unit `tan α`).
    pub cornering_stiffness: Fix128,
}

/// Pacejka Magic Formula (pure-slip curves, combined by the friction ellipse).
///
/// `F = D sin(C atan(B s − E (B s − atan(B s))))` with `D = μ_static F_z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MagicFormulaTire {
    /// Longitudinal stiffness factor `B_x`.
    pub b_x: Fix128,
    /// Longitudinal shape factor `C_x`.
    pub c_x: Fix128,
    /// Longitudinal curvature factor `E_x`.
    pub e_x: Fix128,
    /// Lateral stiffness factor `B_y` (input is `α` in radians).
    pub b_y: Fix128,
    /// Lateral shape factor `C_y`.
    pub c_y: Fix128,
    /// Lateral curvature factor `E_y`.
    pub e_y: Fix128,
}

/// Selectable tyre model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireModel {
    /// Brush / Fiala model.
    Brush(BrushTire),
    /// Pacejka Magic Formula.
    MagicFormula(MagicFormulaTire),
}

impl BrushTire {
    /// Passenger-car tyre: `C_κ = 80 kN`, `C_α = 60 kN`.
    #[must_use]
    pub fn passenger_car() -> Self {
        Self {
            longitudinal_stiffness: Fix128::from_int(80_000),
            cornering_stiffness: Fix128::from_int(60_000),
        }
    }

    /// Brush force (see [`TireModel::force`] for the law).
    fn force(&self, input: &TireInput) -> TireForce {
        let fz = input.normal_load;
        if fz <= Fix128::ZERO {
            return TireForce::default();
        }
        let (mu_xs, mu_xk) = axis_grip(
            input.grip.longitudinal_static,
            input.grip.longitudinal_kinetic,
        );
        let (mu_ys, mu_yk) = axis_grip(input.grip.transverse_static, input.grip.transverse_kinetic);
        let x_on = mu_xs > Fix128::ZERO;
        let y_on = mu_ys > Fix128::ZERO;
        if !x_on && !y_on {
            return TireForce::default();
        }
        let kappa = input.slip_ratio;
        let tan_a = input.slip_tan_alpha;
        let one_k = Fix128::ONE + kappa;
        // Slip vector in ellipse-normalised units, multiplied through by
        // μ_xs μ_ys (and by 1+κ) so that no coefficient is divided by:
        //   b = (C_κ κ μ_ys, C_α tanα μ_xs),  d = 3 F_z (1+κ) μ_xs μ_ys,  λ = |b| / d.
        // An axis without grip drops out (its component is 0, its μ factor 1).
        let wx = if x_on { mu_xs } else { Fix128::ONE };
        let wy = if y_on { mu_ys } else { Fix128::ONE };
        let bx = if x_on {
            self.longitudinal_stiffness * kappa * wy
        } else {
            Fix128::ZERO
        };
        let by = if y_on {
            self.cornering_stiffness * tan_a * wx
        } else {
            Fix128::ZERO
        };
        let d = (Fix128::from_int(3) * fz * one_k * wx * wy).max(Fix128::ZERO);
        let (m, ux, uy) = magnitude_and_direction(bx, by);

        if m < d {
            // adhesion (λ < 1, hence d > 0 and 1+κ > 0)
            let lam = m / d;
            let shape = Fix128::ONE - lam + lam * lam / Fix128::from_int(3);
            let longitudinal = if x_on {
                self.longitudinal_stiffness * kappa / one_k * shape
            } else {
                Fix128::ZERO
            };
            let lateral = if y_on {
                -(self.cornering_stiffness * tan_a / one_k * shape)
            } else {
                Fix128::ZERO
            };
            TireForce {
                longitudinal,
                lateral,
            }
        } else {
            // full sliding: 1/λ = d / m (0 when 1+κ ≤ 0)
            let inv_lam = d / m;
            let mu_x = mu_xk + (mu_xs - mu_xk) * inv_lam;
            let mu_y = mu_yk + (mu_ys - mu_yk) * inv_lam;
            TireForce {
                longitudinal: fz * mu_x * ux,
                lateral: -(fz * mu_y * uy),
            }
        }
    }
}

impl MagicFormulaTire {
    /// Commonly quoted passenger-car coefficients
    /// (`B_x 10, C_x 1.9, E_x 0.97, B_y 8, C_y 1.3, E_y −1`).
    #[must_use]
    pub fn passenger_car() -> Self {
        Self {
            b_x: Fix128::from_int(10),
            c_x: Fix128::from_ratio(19, 10),
            e_x: Fix128::from_ratio(97, 100),
            b_y: Fix128::from_int(8),
            c_y: Fix128::from_ratio(13, 10),
            e_y: Fix128::from_int(-1),
        }
    }

    /// Magic Formula force (see [`TireModel::force`] for the law).
    fn force(&self, input: &TireInput) -> TireForce {
        let fz = input.normal_load;
        // F_z ≤ 0 or a non-positive static coefficient gives D ≤ 0, which zeroes
        // that axis below (and keeps it out of the ellipse scaling)
        let d_x = input.grip.longitudinal_static * fz;
        let d_y = input.grip.transverse_static * fz;
        let mut nx = if d_x > Fix128::ZERO {
            mf_curve(self.b_x, self.c_x, self.e_x, input.slip_ratio)
        } else {
            Fix128::ZERO
        };
        let tan_a = input.slip_tan_alpha;
        let mut ny = if d_y > Fix128::ZERO && !tan_a.is_zero() {
            -mf_curve(self.b_y, self.c_y, self.e_y, tan_a.atan())
        } else {
            Fix128::ZERO
        };
        let n2 = nx * nx + ny * ny;
        if n2 > Fix128::ONE {
            let n = n2.sqrt();
            nx = nx / n;
            ny = ny / n;
        }
        TireForce {
            longitudinal: d_x * nx,
            lateral: d_y * ny,
        }
    }
}

/// `(μ_static, μ_kinetic)` of one axis: static clamped at 0, kinetic clamped
/// into `[0, μ_static]` (keeps the sliding force inside the static ellipse).
fn axis_grip(stat: Fix128, kin: Fix128) -> (Fix128, Fix128) {
    let s = stat.max(Fix128::ZERO);
    (s, kin.max(Fix128::ZERO).min(s))
}

/// `(|v|, v_x/|v|, v_y/|v|)` without squaring the raw components (scaled by
/// the larger one, so large slips cannot overflow). A zero component yields
/// an exactly `±1` unit component; `v = 0` yields all zeros.
fn magnitude_and_direction(a: Fix128, b: Fix128) -> (Fix128, Fix128, Fix128) {
    let large = a.abs().max(b.abs());
    if large.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    }
    let ra = a / large;
    let rb = b / large;
    let n = (ra * ra + rb * rb).sqrt();
    (large * n, ra / n, rb / n)
}

/// Normalised Pacejka curve `sin(C atan(B s − E (B s − atan(B s))))`;
/// exactly 0 at `s = 0`.
fn mf_curve(b: Fix128, c: Fix128, e: Fix128, s: Fix128) -> Fix128 {
    if s.is_zero() {
        return Fix128::ZERO;
    }
    let bs = b * s;
    let inner = bs - e * (bs - bs.atan());
    (c * inner.atan()).sin()
}

impl TireModel {
    /// Contact-patch force for one wheel.
    ///
    /// Both models return exactly zero force for `F_z ≤ 0`, for zero static
    /// grip on both axes, and for `κ = tan α = 0`. A non-positive static
    /// coefficient on one axis removes that axis (its force is 0) and the
    /// other axis follows its one-dimensional law. Negative kinetic
    /// coefficients are treated as 0 and kinetic coefficients above the
    /// static one are clamped to it.
    ///
    /// # Brush
    ///
    /// Theoretical slip `σ = (κ, tan α) / (1 + κ)`, normalised combined slip
    ///
    /// ```text
    /// λ = sqrt((C_κ σ_x / μ_x,s)² + (C_α σ_y / μ_y,s)²) / (3 F_z)
    /// ```
    ///
    /// - adhesion (`1 + κ > 0`, `λ < 1`), the Fiala polynomial along the
    ///   combined-slip direction:
    ///   `F_x = C_κ σ_x (1 − λ + λ²/3)`, `F_y = −C_α σ_y (1 − λ + λ²/3)`;
    ///   in ellipse units `|F| = 1 − (1 − λ)³ ≤ 1`, slope at zero slip
    ///   `C_κ` / `−C_α` for every load.
    /// - full sliding (`λ ≥ 1`): the force points along
    ///   `u = (C_κ κ / μ_x,s, C_α tan α / μ_y,s) / |…|` (opposing the
    ///   slide, continuous with adhesion) with the static → kinetic
    ///   transition
    ///
    ///   ```text
    ///   μ_i(λ) = μ_i,k + (μ_i,s − μ_i,k) / λ        F = F_z (μ_x u_x, −μ_y u_y)
    ///   ```
    ///
    ///   i.e. the full static coefficient at the adhesion limit `λ = 1`
    ///   (force continuous there) decaying to the kinetic coefficient as the
    ///   sliding grows. A locked or backward-spinning wheel (`1 + κ ≤ 0`) is
    ///   the limit `λ = ∞`: pure kinetic, so `κ = −1, tan α = 0` gives
    ///   `F_x = −μ_x,k F_z` exactly. Since `μ_i(λ) ≤ μ_i,s`, the force stays
    ///   inside the static ellipse.
    ///
    /// Evaluated as `b = (C_κ κ μ_y,s, C_α tan α μ_x,s)`,
    /// `d = 3 F_z (1 + κ) μ_x,s μ_y,s`, `λ = |b| / d` — no division by a
    /// coefficient and no squaring of large slips.
    ///
    /// # Magic Formula
    ///
    /// Pure-slip curves (`D_x = μ_x,s F_z`, `D_y = μ_y,s F_z`, `α = atan(tan α)`)
    ///
    /// ```text
    /// f_x = sin(C_x atan(B_x κ − E_x (B_x κ − atan(B_x κ))))
    /// f_y = −sin(C_y atan(B_y α − E_y (B_y α − atan(B_y α))))
    /// ```
    ///
    /// combined by the friction ellipse: if `n = sqrt(f_x² + f_y²) > 1` both
    /// are divided by `n` (radial projection onto the ellipse), otherwise
    /// they are used as is; `F = (D_x f_x, D_y f_y)`. The kinetic
    /// coefficients are not used — the curve's own post-peak fall-off is the
    /// sliding behaviour.
    #[must_use]
    pub fn force(&self, input: &TireInput) -> TireForce {
        match self {
            Self::Brush(b) => b.force(input),
            Self::MagicFormula(m) => m.force(input),
        }
    }

    /// Small-slip longitudinal slope `∂F_x/∂κ` at load `normal_load` and grip
    /// `mu_peak` (used by the vehicle's implicit wheel-spin update).
    ///
    /// Brush: `C_κ`. Magic Formula: `B_x C_x μ_peak F_z`. Zero when
    /// `normal_load ≤ 0` or `mu_peak ≤ 0` (the force is zero there).
    #[must_use]
    pub fn longitudinal_slope(&self, normal_load: Fix128, mu_peak: Fix128) -> Fix128 {
        if normal_load <= Fix128::ZERO || mu_peak <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        match self {
            Self::Brush(b) => b.longitudinal_stiffness,
            Self::MagicFormula(m) => m.b_x * m.c_x * mu_peak * normal_load,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Oracles. Every expected value is an f64 closed form written out here
    //! (Fiala polynomial, Pacejka formula, the brush law of this module's doc)
    //! and evaluated with platform libm — never by calling the code under test.
    #![allow(clippy::disallowed_methods)] // f64 libm closed forms are the oracle on purpose

    use super::*;

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    fn brush() -> TireModel {
        TireModel::Brush(BrushTire::passenger_car())
    }

    fn mf() -> TireModel {
        TireModel::MagicFormula(MagicFormulaTire::passenger_car())
    }

    fn input(kappa: f64, tan_alpha: f64, fz: f64, grip: AnisotropicFriction) -> TireInput {
        TireInput {
            slip_ratio: fx(kappa),
            slip_tan_alpha: fx(tan_alpha),
            normal_load: fx(fz),
            grip,
        }
    }

    /// Grip with exactly representable coefficients (binary fractions).
    fn exact_grip() -> AnisotropicFriction {
        AnisotropicFriction {
            longitudinal_static: Fix128::from_ratio(5, 4),
            longitudinal_kinetic: Fix128::from_ratio(3, 4),
            transverse_static: Fix128::ONE,
            transverse_kinetic: Fix128::from_ratio(1, 2),
            slip_threshold_m_s: Fix128::from_ratio(1, 20),
        }
    }

    fn zero_grip() -> AnisotropicFriction {
        AnisotropicFriction {
            longitudinal_static: Fix128::ZERO,
            longitudinal_kinetic: Fix128::ZERO,
            transverse_static: Fix128::ZERO,
            transverse_kinetic: Fix128::ZERO,
            slip_threshold_m_s: Fix128::ZERO,
        }
    }

    /// μ_kinetic > μ_static (physically inconsistent input; the kinetic
    /// coefficient is clamped to the static one).
    fn inverted_grip() -> AnisotropicFriction {
        AnisotropicFriction {
            longitudinal_static: Fix128::from_ratio(1, 2),
            longitudinal_kinetic: Fix128::from_ratio(9, 10),
            transverse_static: Fix128::from_ratio(4, 10),
            transverse_kinetic: Fix128::from_ratio(8, 10),
            slip_threshold_m_s: Fix128::from_ratio(1, 20),
        }
    }

    fn grips() -> [AnisotropicFriction; 5] {
        [
            AnisotropicFriction::tyre_asphalt(),
            AnisotropicFriction::ski_snow(),
            AnisotropicFriction::skate_ice(),
            exact_grip(),
            inverted_grip(),
        ]
    }

    const KAPPAS: [f64; 15] = [
        -1.0e6, -2.0, -1.0, -0.99, -0.5, -0.1, -0.01, 0.0, 0.003, 0.05, 0.2, 1.0, 5.0, 40.0, 1.0e6,
    ];
    const TANS: [f64; 13] = [
        -1.0e6, -3.0, -0.5, -0.1, -0.01, 0.0, 0.004, 0.08, 0.3, 1.0, 10.0, 200.0, 1.0e6,
    ];
    const LOADS: [f64; 3] = [100.0, 4000.0, 20_000.0];

    /// `(μ_static, μ_kinetic)` of one axis as the module doc defines it:
    /// static clamped at 0, kinetic clamped into `[0, μ_static]`.
    fn axis(s: Fix128, k: Fix128) -> (f64, f64) {
        let s = s.to_f64().max(0.0);
        (s, k.to_f64().clamp(0.0, s))
    }

    /// Brush law of the module doc, written out in f64 (both axes with
    /// positive static grip).
    ///
    /// `σ = (κ, tan α)/(1+κ)`, `λ = |(C_κ σ_x/μ_xs, C_α σ_y/μ_ys)| / (3 F_z)`;
    /// adhesion (`1+κ > 0`, `λ < 1`): `F = (C_κ σ_x, −C_α σ_y)(1 − λ + λ²/3)`;
    /// sliding: direction `u = (C_κ κ/μ_xs, C_α tan α/μ_ys)/|…|`,
    /// `μ_i = μ_ik + (μ_is − μ_ik)/λ` (`1/λ = 0` when `1+κ ≤ 0`),
    /// `F = F_z (μ_x u_x, −μ_y u_y)`.
    #[allow(clippy::too_many_arguments)]
    fn brush_ref(
        ck: f64,
        ca: f64,
        kappa: f64,
        t: f64,
        fz: f64,
        g: &AnisotropicFriction,
    ) -> (f64, f64) {
        let (mxs, mxk) = axis(g.longitudinal_static, g.longitudinal_kinetic);
        let (mys, myk) = axis(g.transverse_static, g.transverse_kinetic);
        let one_k = 1.0 + kappa;
        if one_k > 0.0 {
            let sx = kappa / one_k;
            let sy = t / one_k;
            let lam = (ck * sx / mxs).hypot(ca * sy / mys) / (3.0 * fz);
            if lam < 1.0 {
                let shape = 1.0 - lam + lam * lam / 3.0;
                return (ck * sx * shape, -ca * sy * shape);
            }
            let ax = ck * kappa / mxs;
            let ay = ca * t / mys;
            let m = ax.hypot(ay);
            let mx = mxk + (mxs - mxk) / lam;
            let my = myk + (mys - myk) / lam;
            (fz * mx * ax / m, -fz * my * ay / m)
        } else {
            let ax = ck * kappa / mxs;
            let ay = ca * t / mys;
            let m = ax.hypot(ay);
            (fz * mxk * ax / m, -fz * myk * ay / m)
        }
    }

    /// Pacejka pure-slip curve normalised by `D`, f64 / libm.
    fn mf_curve_ref(b: f64, c: f64, e: f64, s: f64) -> f64 {
        let bs = b * s;
        (c * (bs - e * (bs - bs.atan())).atan()).sin()
    }

    /// Magic Formula + friction-ellipse scaling of the module doc, f64 / libm.
    fn mf_ref(
        t: &MagicFormulaTire,
        kappa: f64,
        tan_a: f64,
        fz: f64,
        g: &AnisotropicFriction,
    ) -> (f64, f64) {
        let dx = g.longitudinal_static.to_f64().max(0.0) * fz;
        let dy = g.transverse_static.to_f64().max(0.0) * fz;
        let mut nx = if dx > 0.0 {
            mf_curve_ref(t.b_x.to_f64(), t.c_x.to_f64(), t.e_x.to_f64(), kappa)
        } else {
            0.0
        };
        let mut ny = if dy > 0.0 {
            -mf_curve_ref(t.b_y.to_f64(), t.c_y.to_f64(), t.e_y.to_f64(), tan_a.atan())
        } else {
            0.0
        };
        let n = nx.hypot(ny);
        if n > 1.0 {
            nx /= n;
            ny /= n;
        }
        (dx * nx, dy * ny)
    }

    /// Error bound of one normalised MF curve evaluated with the Fix128
    /// CORDIC functions.
    ///
    /// Pinned accuracies (`crate::math` tests): `atan` ≤ 1e-12
    /// (`atan_matches_f64_within_1e12`), `sin` ≤ 1e-11
    /// (`sin_cos_large_and_negative_angles_reduce_correctly`). With
    /// `|d atan| ≤ 1`, `|d sin| ≤ 1`, an error `ε_a` on `atan(B s)` moves the
    /// outer argument by `|E| ε_a`, the outer `atan` adds `ε_a`, the factor
    /// `C` multiplies both, and `sin` adds `ε_s`. On the lateral axis the
    /// input `α = atan(tan α)` carries another `ε_a`, amplified by
    /// `B (1 + |E|)` before the outer `atan`. Fixed-point rounding (2^-64)
    /// and the f64 reference (~1e-16) are far below and covered by the final
    /// factor 2.
    fn mf_curve_tol(b: f64, c: f64, e: f64, lateral: bool) -> f64 {
        const EPS_ATAN: f64 = 1e-12;
        const EPS_SIN: f64 = 1e-11;
        let alpha_term = if lateral {
            b * (1.0 + e.abs()) * EPS_ATAN
        } else {
            0.0
        };
        2.0 * (EPS_SIN + c * (EPS_ATAN * (1.0 + e.abs()) + alpha_term))
    }

    fn ellipse_norm(f: TireForce, fz: f64, g: &AnisotropicFriction) -> f64 {
        let mx = g.longitudinal_static.to_f64();
        let my = g.transverse_static.to_f64();
        let x = f.longitudinal.to_f64() / (mx * fz);
        let y = f.lateral.to_f64() / (my * fz);
        x * x + y * y
    }

    // ---------------------------------------------------------------- brush

    #[test]
    fn brush_passenger_car_matches_doc() {
        let b = BrushTire::passenger_car();
        assert_eq!(b.longitudinal_stiffness, Fix128::from_int(80_000));
        assert_eq!(b.cornering_stiffness, Fix128::from_int(60_000));
    }

    #[test]
    fn brush_small_slip_slopes_are_c_kappa_and_minus_c_alpha_for_any_load() {
        // F_x = C_κ κ/(1+κ) (1 − λ + λ²/3) ⇒ |F_x/κ − C_κ| ≤ C_κ (|κ|/(1−|κ|) + λ)
        let (ck, ca) = (80_000.0, 60_000.0);
        let g = AnisotropicFriction::tyre_asphalt();
        for fz in [500.0, 2000.0, 8000.0] {
            for s in [1e-7, -1e-7, 1e-6] {
                let f = brush().force(&input(s, 0.0, fz, g));
                let lam = ck * s.abs() / (3.0 * 1.1 * fz);
                // + 1e-9 C_κ: f64 slack, the bound is attained to first order
                let tol = ck * (s.abs() / (1.0 - s.abs()) + lam + 1e-9);
                let slope = f.longitudinal.to_f64() / s;
                assert!(
                    (slope - ck).abs() <= tol,
                    "dFx/dκ {slope} at Fz {fz}, κ {s}"
                );
                assert!(f.lateral.is_zero());

                let f = brush().force(&input(0.0, s, fz, g));
                let lam = ca * s.abs() / (3.0 * 0.9 * fz);
                let slope = f.lateral.to_f64() / s;
                // |F_y/t + C_α| = C_α (λ − λ²/3) ≤ C_α λ (+ f64 slack)
                assert!(
                    (slope + ca).abs() <= ca * (lam + 1e-9),
                    "dFy/dtanα {slope} at Fz {fz}"
                );
                assert!(f.longitudinal.is_zero());
            }
        }
    }

    #[test]
    fn brush_locked_wheel_is_exactly_minus_mu_kinetic_fz() {
        // κ = −1, tan α = 0: F_x = −μ_x,kinetic F_z exactly (3/4 · 4000 = 3000)
        let g = exact_grip();
        let f = brush().force(&TireInput {
            slip_ratio: Fix128::NEG_ONE,
            slip_tan_alpha: Fix128::ZERO,
            normal_load: Fix128::from_int(4000),
            grip: g,
        });
        assert_eq!(f.longitudinal, Fix128::from_int(-3000));
        assert_eq!(f.lateral, Fix128::ZERO);

        // non-dyadic coefficient: still the bit-exact Fix128 product
        let g = AnisotropicFriction::tyre_asphalt();
        let fz = Fix128::from_ratio(73_291, 17);
        let f = brush().force(&TireInput {
            slip_ratio: Fix128::NEG_ONE,
            slip_tan_alpha: Fix128::ZERO,
            normal_load: fz,
            grip: g,
        });
        assert_eq!(f.longitudinal, -(fz * g.longitudinal_kinetic));
        assert_eq!(f.lateral, Fix128::ZERO);
    }

    #[test]
    fn brush_stays_inside_static_friction_ellipse() {
        for g in grips() {
            for fz in LOADS {
                for k in KAPPAS {
                    for t in TANS {
                        let f = brush().force(&input(k, t, fz, g));
                        let n = ellipse_norm(f, fz, &g);
                        assert!(
                            n <= 1.0 + 1e-12,
                            "outside ellipse: {n} (κ {k}, tanα {t}, Fz {fz})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn brush_pure_lateral_matches_fiala_closed_form() {
        // Fiala: |tanα| < 3 μ_s F_z / C_α:
        //   F_y = −C_α t + C_α² |t| t /(3 μ_s F_z) − C_α³ t³ /(27 μ_s² F_z²)
        // beyond: F_y = −sign(t) F_z (μ_k + (μ_s − μ_k) · 3 μ_s F_z /(C_α |t|))
        let ca = 60_000.0;
        let g = AnisotropicFriction::tyre_asphalt();
        let (ms, mk) = (0.9, 0.7);
        for fz in LOADS {
            for t in [
                -2.0f64, -0.3, -0.05, -0.004, 0.001, 0.02, 0.1, 0.25, 1.0, 50.0,
            ] {
                let limit = 3.0 * ms * fz / ca;
                let expect = if t.abs() < limit {
                    -ca * t + ca * ca * t.abs() * t / (3.0 * ms * fz)
                        - ca * ca * ca * t * t * t / (27.0 * ms * ms * fz * fz)
                } else {
                    -t.signum() * fz * (mk + (ms - mk) * 3.0 * ms * fz / (ca * t.abs()))
                };
                let f = brush().force(&input(0.0, t, fz, g));
                assert!(
                    (f.lateral.to_f64() - expect).abs() <= 1e-9 * ms * fz,
                    "Fy {} vs Fiala {expect} (tanα {t}, Fz {fz})",
                    f.lateral.to_f64()
                );
                assert!(f.longitudinal.is_zero());
            }
        }
    }

    #[test]
    fn brush_combined_slip_matches_closed_form() {
        let (ck, ca) = (80_000.0, 60_000.0);
        for g in grips() {
            for fz in LOADS {
                for k in KAPPAS {
                    for t in TANS {
                        if k == 0.0 && t == 0.0 {
                            continue;
                        }
                        let (ex, ey) = brush_ref(ck, ca, k, t, fz, &g);
                        let f = brush().force(&input(k, t, fz, g));
                        let tol = 1e-9
                            * fz
                            * g.longitudinal_static
                                .to_f64()
                                .max(g.transverse_static.to_f64());
                        assert!(
                            (f.longitudinal.to_f64() - ex).abs() <= tol
                                && (f.lateral.to_f64() - ey).abs() <= tol,
                            "({}, {}) vs ({ex}, {ey}) at κ {k}, tanα {t}, Fz {fz}",
                            f.longitudinal.to_f64(),
                            f.lateral.to_f64()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn brush_is_continuous_at_the_adhesion_limit() {
        // pure longitudinal: λ = 1 at σ_x = 3 μ_s F_z / C_κ; both sides tend to μ_s F_z
        let g = exact_grip();
        let fz = 4000.0;
        let sigma = 3.0 * 1.25 * fz / 80_000.0;
        let k_lim = sigma / (1.0 - sigma);
        for k in [k_lim * (1.0 - 1e-9), k_lim * (1.0 + 1e-9)] {
            let f = brush().force(&input(k, 0.0, fz, g));
            assert!(
                (f.longitudinal.to_f64() - 1.25 * fz).abs() < 1e-4,
                "{}",
                f.longitudinal.to_f64()
            );
        }
    }

    // ---------------------------------------------------------- magic formula

    #[test]
    fn mf_passenger_car_matches_doc() {
        let m = MagicFormulaTire::passenger_car();
        assert_eq!(m.b_x, Fix128::from_int(10));
        assert_eq!(m.c_x, Fix128::from_ratio(19, 10));
        assert_eq!(m.e_x, Fix128::from_ratio(97, 100));
        assert_eq!(m.b_y, Fix128::from_int(8));
        assert_eq!(m.c_y, Fix128::from_ratio(13, 10));
        assert_eq!(m.e_y, Fix128::from_int(-1));
    }

    #[test]
    fn mf_matches_pacejka_formula_on_slip_grid() {
        let p = MagicFormulaTire::passenger_car();
        let tx = mf_curve_tol(10.0, 1.9, 0.97, false);
        let ty = mf_curve_tol(8.0, 1.3, -1.0, true);
        for g in grips() {
            for fz in LOADS {
                for k in KAPPAS {
                    for t in TANS {
                        let (ex, ey) = mf_ref(&p, k, t, fz, &g);
                        let f = mf().force(&input(k, t, fz, g));
                        let dx = g.longitudinal_static.to_f64() * fz;
                        let dy = g.transverse_static.to_f64() * fz;
                        // ellipse scaling: f/n with n ≤ √2 and |∂(f/n)/∂f| ≤ 2/n ⇒ ×2
                        let (ax, ay) = (2.0 * (tx + ty) * dx, 2.0 * (tx + ty) * dy);
                        assert!(
                            (f.longitudinal.to_f64() - ex).abs() <= ax
                                && (f.lateral.to_f64() - ey).abs() <= ay,
                            "({}, {}) vs ({ex}, {ey}) at κ {k}, tanα {t}, Fz {fz}",
                            f.longitudinal.to_f64(),
                            f.lateral.to_f64()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mf_pure_slip_equals_d_sin_formula() {
        // pure slip: no ellipse scaling (one component is 0), F = D · curve
        let g = AnisotropicFriction::tyre_asphalt();
        let fz = 3500.0;
        for k in [-0.7, -0.08, 0.02, 0.12, 0.6] {
            let f = mf().force(&input(k, 0.0, fz, g));
            let d = 1.1 * fz;
            let expect = d * mf_curve_ref(10.0, 1.9, 0.97, k);
            assert!(
                (f.longitudinal.to_f64() - expect).abs()
                    <= d * mf_curve_tol(10.0, 1.9, 0.97, false)
            );
            assert!(f.lateral.is_zero());
        }
        for t in [-0.7, -0.05, 0.01, 0.15, 2.0] {
            let f = mf().force(&input(0.0, t, fz, g));
            let d = 0.9 * fz;
            let expect = -d * mf_curve_ref(8.0, 1.3, -1.0, t.atan());
            assert!((f.lateral.to_f64() - expect).abs() <= d * mf_curve_tol(8.0, 1.3, -1.0, true));
            assert!(f.longitudinal.is_zero());
        }
    }

    #[test]
    fn mf_stays_inside_static_friction_ellipse() {
        for g in grips() {
            for fz in LOADS {
                for k in KAPPAS {
                    for t in TANS {
                        let f = mf().force(&input(k, t, fz, g));
                        let n = ellipse_norm(f, fz, &g);
                        assert!(
                            n <= 1.0 + 1e-12,
                            "outside ellipse: {n} (κ {k}, tanα {t}, Fz {fz})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mf_combined_slip_is_scaled_onto_the_ellipse() {
        // κ = 0.2, tanα = 0.3: both pure curves are near 1 ⇒ n > 1, result lies on the ellipse
        let g = AnisotropicFriction::tyre_asphalt();
        let nx = mf_curve_ref(10.0, 1.9, 0.97, 0.2);
        let ny = mf_curve_ref(8.0, 1.3, -1.0, 0.3f64.atan());
        assert!(nx.hypot(ny) > 1.0);
        let f = mf().force(&input(0.2, 0.3, 4000.0, g));
        let n = ellipse_norm(f, 4000.0, &g);
        assert!((n - 1.0).abs() < 1e-9, "{n}");
    }

    // -------------------------------------------------------------- signs

    #[test]
    fn signs_follow_the_slip_conventions_for_both_models() {
        for model in [brush(), mf()] {
            for g in grips() {
                for fz in LOADS {
                    for k in [0.001, 0.1, 1.0, 1.0e6] {
                        for t in [-0.3, 0.0, 0.3] {
                            let f = model.force(&input(k, t, fz, g));
                            assert!(f.longitudinal > Fix128::ZERO, "κ {k} > 0 ⇒ Fx > 0");
                            let f = model.force(&input(-k.min(0.5), t, fz, g));
                            assert!(f.longitudinal < Fix128::ZERO, "κ < 0 ⇒ Fx < 0");
                        }
                    }
                    for t in [0.001, 0.1, 1.0, 1.0e6] {
                        for k in [-0.5, 0.0, 0.5] {
                            let f = model.force(&input(k, t, fz, g));
                            assert!(f.lateral < Fix128::ZERO, "tanα {t} > 0 ⇒ Fy < 0");
                            let f = model.force(&input(k, -t, fz, g));
                            assert!(f.lateral > Fix128::ZERO, "tanα < 0 ⇒ Fy > 0");
                        }
                    }
                }
            }
        }
    }

    // --------------------------------------------------- degenerate inputs

    #[test]
    fn non_positive_load_gives_exactly_zero_force() {
        // doc: "Non-positive load gives a zero force"
        for model in [brush(), mf()] {
            for fz in [0.0, -1.0, -5000.0] {
                for (k, t) in [(0.1, 0.2), (-1.0, 0.0), (1.0e6, -1.0e6)] {
                    let f = model.force(&input(k, t, fz, AnisotropicFriction::tyre_asphalt()));
                    assert_eq!(f, TireForce::default(), "Fz {fz}");
                }
                assert_eq!(model.longitudinal_slope(fx(fz), fx(1.1)), Fix128::ZERO);
            }
        }
    }

    #[test]
    fn zero_grip_gives_exactly_zero_force_and_slope() {
        // doc: friction ellipse with zero semi-axes ⇒ F = 0
        for model in [brush(), mf()] {
            for (k, t) in [(0.1, 0.2), (-1.0, 0.0), (1.0e6, -1.0e6), (0.0, 0.3)] {
                let f = model.force(&input(k, t, 4000.0, zero_grip()));
                assert_eq!(f, TireForce::default());
            }
            assert_eq!(
                model.longitudinal_slope(fx(4000.0), Fix128::ZERO),
                Fix128::ZERO
            );
        }
    }

    #[test]
    fn zero_grip_on_one_axis_keeps_the_other_axis_as_pure_slip() {
        // μ_x = 0: F_x = 0, F_y follows the 1-D law of the transverse axis (with σ_y = tanα/(1+κ))
        let g = AnisotropicFriction {
            longitudinal_static: Fix128::ZERO,
            longitudinal_kinetic: Fix128::ZERO,
            ..AnisotropicFriction::tyre_asphalt()
        };
        let (ca, ms, fz) = (60_000.0, 0.9, 4000.0);
        for (k, t) in [(0.0, 0.01), (0.3, 0.01), (0.0, 0.5)] {
            let f = brush().force(&input(k, t, fz, g));
            assert!(f.longitudinal.is_zero());
            let s = t / (1.0 + k);
            let lam = ca * s / (3.0 * ms * fz);
            let expect = if lam < 1.0 {
                -ca * s * (1.0 - lam + lam * lam / 3.0)
            } else {
                -fz * (0.7 + 0.2 / lam)
            };
            assert!(
                (f.lateral.to_f64() - expect).abs() < 1e-9 * fz,
                "{} vs {expect}",
                f.lateral.to_f64()
            );

            let f = mf().force(&input(k, t, fz, g));
            assert!(f.longitudinal.is_zero());
            let expect = -ms * fz * mf_curve_ref(8.0, 1.3, -1.0, t.atan());
            assert!(
                (f.lateral.to_f64() - expect).abs() <= ms * fz * mf_curve_tol(8.0, 1.3, -1.0, true)
            );
        }
    }

    #[test]
    fn huge_slip_saturates_to_the_documented_values() {
        let g = AnisotropicFriction::tyre_asphalt();
        let fz = 4000.0;
        let (ck, ca) = (80_000.0, 60_000.0);
        // brush, κ = +1e6: sliding, λ = C_κ σ_x /(3 μ_s F_z), F_x = F_z (μ_k + (μ_s − μ_k)/λ)
        let sx = 1.0e6 / (1.0 + 1.0e6);
        let lam = ck * sx / (3.0 * 1.1 * fz);
        let f = brush().force(&input(1.0e6, 0.0, fz, g));
        assert!((f.longitudinal.to_f64() - fz * (0.9 + 0.2 / lam)).abs() < 1e-9 * fz);
        assert!(f.lateral.is_zero());
        // brush, κ = −1e6 (1+κ < 0): pure kinetic, exactly −μ_k F_z
        let f = brush().force(&TireInput {
            slip_ratio: Fix128::from_int(-1_000_000),
            slip_tan_alpha: Fix128::ZERO,
            normal_load: Fix128::from_int(4000),
            grip: g,
        });
        assert_eq!(
            f.longitudinal,
            -(Fix128::from_int(4000) * g.longitudinal_kinetic)
        );
        // brush, tanα = ±1e6: F_y = ∓F_z (μ_k + (μ_s − μ_k)/λ), λ = C_α 1e6 /(3 μ_s F_z)
        let lam = ca * 1.0e6 / (3.0 * 0.9 * fz);
        for s in [1.0, -1.0] {
            let f = brush().force(&input(0.0, s * 1.0e6, fz, g));
            assert!((f.lateral.to_f64() + s * fz * (0.7 + 0.2 / lam)).abs() < 1e-9 * fz);
        }
        // MF, κ = ±1e6: F_x = ±D sin(C atan(B κ (1−E) + E atan(B κ)))
        for s in [1.0, -1.0] {
            let f = mf().force(&input(s * 1.0e6, 0.0, fz, g));
            let expect = 1.1 * fz * mf_curve_ref(10.0, 1.9, 0.97, s * 1.0e6);
            assert!(
                (f.longitudinal.to_f64() - expect).abs()
                    <= 1.1 * fz * mf_curve_tol(10.0, 1.9, 0.97, false)
            );
            // MF, tanα = ±1e6 ⇒ α → ±π/2
            let f = mf().force(&input(0.0, s * 1.0e6, fz, g));
            let expect = -0.9 * fz * mf_curve_ref(8.0, 1.3, -1.0, (s * 1.0e6f64).atan());
            assert!(
                (f.lateral.to_f64() - expect).abs()
                    <= 0.9 * fz * mf_curve_tol(8.0, 1.3, -1.0, true)
            );
        }
    }

    #[test]
    fn zero_slip_gives_exactly_zero_force() {
        for model in [brush(), mf()] {
            for g in grips() {
                for fz in LOADS {
                    let f = model.force(&input(0.0, 0.0, fz, g));
                    assert_eq!(f, TireForce::default());
                }
            }
        }
    }

    // ------------------------------------------------------ longitudinal slope

    #[test]
    fn longitudinal_slope_is_the_zero_slip_derivative() {
        // brush: C_κ (load and grip independent); MF: d/dκ D sin(C atan(Bκ − E(Bκ − atan Bκ)))|₀ = B C D
        assert_eq!(
            brush().longitudinal_slope(Fix128::from_int(4000), fx(1.1)),
            Fix128::from_int(80_000)
        );
        assert_eq!(
            brush().longitudinal_slope(Fix128::from_int(200), fx(0.1)),
            Fix128::from_int(80_000)
        );
        let s = mf().longitudinal_slope(Fix128::from_int(4000), Fix128::from_ratio(11, 10));
        assert!((s.to_f64() - 10.0 * 1.9 * 1.1 * 4000.0).abs() < 1e-9);
        // consistent with a central difference of the MF force (h = 1e-6, error O(h²)·|f'''|)
        let g = AnisotropicFriction {
            longitudinal_static: Fix128::from_ratio(11, 10),
            ..AnisotropicFriction::tyre_asphalt()
        };
        let h = 1e-6;
        let fp = mf().force(&input(h, 0.0, 4000.0, g)).longitudinal.to_f64();
        let fm = mf().force(&input(-h, 0.0, 4000.0, g)).longitudinal.to_f64();
        assert!(((fp - fm) / (2.0 * h) - s.to_f64()).abs() / s.to_f64() < 1e-6);
    }
}
