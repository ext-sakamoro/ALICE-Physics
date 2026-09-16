//! 1-D acoustic wave equation solver.
//!
//! Solves the linear wave equation
//!
//! ```text
//! ∂²u / ∂t² = c² · ∂² u / ∂x²
//! ```
//!
//! on a uniform 1-D grid with reflective (Neumann, zero-gradient) end
//! conditions using an explicit leap-frog scheme:
//!
//! ```text
//! u_new[i] = 2·u[i] − u_old[i] + C² · (u[i+1] − 2·u[i] + u[i−1])
//! ```
//!
//! where `C = c · dt / dx` is the Courant number (`C ≤ 1` for
//! stability). The module is deliberately compact — it exposes the
//! kernel + a stability helper + material presets, and callers write
//! their own driver loop for room-scale audio simulation.
//!
//! # Scope
//!
//! - 1-D uniform grid, reflective ends. 2-D / 3-D solvers, absorbing
//!   boundaries, spectral / DG discretisations, and coupled fluid–
//!   structure interaction are future work.
//! - Small-amplitude (linear) acoustics; nonlinear compression is out
//!   of scope.

/// One integration step of the 1-D wave equation. `current` and
/// `previous` must be the same length; the result is written back to
/// `next` (also same length).
///
/// The Courant number `C = c · dt / dx` should not exceed `1` for
/// stability; the module provides [`stable_dt`] to compute the CFL
/// upper bound.
///
/// # Panics
///
/// Panics if the three arrays have mismatched lengths.
pub fn leapfrog_step(current: &[f32], previous: &[f32], next: &mut [f32], courant: f32) {
    assert_eq!(current.len(), previous.len(), "length mismatch");
    assert_eq!(current.len(), next.len(), "length mismatch");
    let n = current.len();
    if n < 3 {
        for (dst, &src) in next.iter_mut().zip(current.iter()) {
            *dst = src;
        }
        return;
    }
    let c2 = courant * courant;
    // Interior update.
    for i in 1..n - 1 {
        let laplacian = current[i + 1] - 2.0 * current[i] + current[i - 1];
        next[i] = 2.0 * current[i] - previous[i] + c2 * laplacian;
    }
    // Neumann (zero-gradient) reflection at both ends.
    next[0] = next[1];
    next[n - 1] = next[n - 2];
}

/// CFL upper bound on the time step: `dt < dx / c`.
///
/// # Panics
///
/// Panics if `dx <= 0` or `wave_speed_m_s <= 0`.
#[must_use]
pub fn stable_dt(dx_m: f32, wave_speed_m_s: f32) -> f32 {
    assert!(dx_m > 0.0, "dx must be positive");
    assert!(wave_speed_m_s > 0.0, "wave_speed must be positive");
    dx_m / wave_speed_m_s
}

/// Common wave-speed presets (m/s).
pub mod speeds {
    /// Sound in dry air at 20 °C.
    pub const AIR_20C: f32 = 343.0;
    /// Sound in fresh water at 25 °C.
    pub const WATER_25C: f32 = 1_497.0;
    /// Compressional wave in structural steel.
    pub const STEEL_LONGITUDINAL: f32 = 5_960.0;
    /// Compressional wave in typical cast concrete.
    pub const CONCRETE_LONGITUDINAL: f32 = 3_650.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_field_stays_uniform() {
        let current = vec![1.0_f32; 64];
        let previous = current.clone();
        let mut next = vec![0.0_f32; 64];
        let dt = stable_dt(0.01, speeds::AIR_20C) * 0.5;
        let courant = speeds::AIR_20C * dt / 0.01;
        leapfrog_step(&current, &previous, &mut next, courant);
        for &v in &next {
            assert!((v - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn peak_propagates_outward() {
        let n = 64;
        let mut current = vec![0.0_f32; n];
        current[n / 2] = 1.0;
        let previous = current.clone();
        let mut next = vec![0.0_f32; n];
        leapfrog_step(&current, &previous, &mut next, 0.5);
        // Neighbours must have picked up some value.
        assert!(next[n / 2 - 1] > 0.0);
        assert!(next[n / 2 + 1] > 0.0);
    }

    #[test]
    fn stable_dt_returns_positive_value() {
        let dt = stable_dt(0.01, speeds::WATER_25C);
        assert!(dt > 0.0);
        assert!(dt.is_finite());
    }

    #[test]
    #[should_panic(expected = "dx must be positive")]
    fn stable_dt_panics_on_zero_dx() {
        let _ = stable_dt(0.0, speeds::AIR_20C);
    }

    #[test]
    fn short_array_returns_unchanged_copy() {
        let current = vec![1.0_f32, 2.0];
        let previous = vec![0.5_f32, 1.5];
        let mut next = vec![0.0_f32, 0.0];
        leapfrog_step(&current, &previous, &mut next, 0.9);
        assert_eq!(next, current);
    }

    #[test]
    fn presets_have_expected_ordering() {
        // Gas < liquid < porous solid < dense solid.
        let ascending = [
            speeds::AIR_20C,
            speeds::WATER_25C,
            speeds::CONCRETE_LONGITUDINAL,
            speeds::STEEL_LONGITUDINAL,
        ];
        assert!(
            ascending.windows(2).all(|w| w[0] < w[1]),
            "speed presets out of order: {ascending:?}"
        );
    }

    #[test]
    fn reflective_ends_mirror_interior() {
        let n = 8;
        let mut current = vec![0.0_f32; n];
        current[3] = 1.0;
        current[4] = 1.0;
        let previous = current.clone();
        let mut next = vec![0.0_f32; n];
        leapfrog_step(&current, &previous, &mut next, 0.5);
        assert_eq!(next[0], next[1]);
        assert_eq!(next[n - 1], next[n - 2]);
    }

    /// Hand-computed leap-frog step on five dyadic cells with `C = 1/2`
    /// (`C² = 1/4`), every value exact in f32:
    ///
    /// ```text
    /// u   = [1, 2, 4, 6, 3]      u_old = [1/2, 1, 3/2, 2, 5/2]
    /// i=1: Δ = 4 − 2·2 + 1 = 1    u' = 2·2 − 1   + 1/4·1    = 3.25
    /// i=2: Δ = 6 − 2·4 + 2 = 0    u' = 2·4 − 3/2 + 0        = 6.5
    /// i=3: Δ = 3 − 2·6 + 4 = −5   u' = 2·6 − 2   + 1/4·(−5) = 8.75
    /// ends: u'[0] = u'[1] = 3.25, u'[4] = u'[3] = 8.75
    /// ```
    ///
    /// `C = 1/2` separates `C·C = 1/4` from `C + C = 1` and `C / C = 1`
    /// (at i=1 those give 4 instead of 3.25); `u[i] ∉ {0, 1, √2}` separates
    /// `2·u[i]` from `2/u[i]` (i=1: 1 − 1 + 1/4 = 0.25 instead of 3.25).
    #[test]
    fn leapfrog_step_interior_matches_hand_computation() {
        let current = [1.0_f32, 2.0, 4.0, 6.0, 3.0];
        let previous = [0.5_f32, 1.0, 1.5, 2.0, 2.5];
        let mut next = [0.0_f32; 5];
        leapfrog_step(&current, &previous, &mut next, 0.5);
        assert_eq!(next, [3.25, 3.25, 6.5, 8.75, 8.75]);
    }

    /// `n < 3` is the copy path; `n == 3` is exactly at the threshold and must
    /// be integrated (one interior cell, both ends mirrored):
    ///
    /// ```text
    /// u = [0, 4, 0], u_old = [0, 1, 0], C = 1/2
    /// i=1: Δ = 0 − 8 + 0 = −8    u' = 8 − 1 + 1/4·(−8) = 5
    /// ends: [5, 5, 5]
    /// ```
    ///
    /// The `n <= 3` mutant would return `[0, 4, 0]` unchanged.
    #[test]
    fn three_cells_are_integrated_not_copied() {
        let current = [0.0_f32, 4.0, 0.0];
        let previous = [0.0_f32, 1.0, 0.0];
        let mut next = [0.0_f32; 3];
        leapfrog_step(&current, &previous, &mut next, 0.5);
        assert_eq!(next, [5.0, 5.0, 5.0]);
        // one below the threshold: verbatim copy, previous ignored
        let current2 = [0.0_f32, 4.0];
        let previous2 = [0.0_f32, 1.0];
        let mut next2 = [7.0_f32; 2];
        leapfrog_step(&current2, &previous2, &mut next2, 0.5);
        assert_eq!(next2, current2);
    }

    /// `dt_max = dx / c` with exactly representable quotients. Operands are
    /// chosen so `dx % c` and `dx · c` both differ: `0.5 / 4 = 0.125` (mod
    /// 0.5, product 2), `3 / 1.5 = 2` (mod 0, product 4.5), and with the
    /// air preset `686 / 343 = 2` (mod 0, product 235 298).
    #[test]
    fn stable_dt_is_dx_over_wave_speed() {
        assert_eq!(stable_dt(0.5, 4.0), 0.125);
        assert_eq!(stable_dt(3.0, 1.5), 2.0);
        assert_eq!(stable_dt(686.0, speeds::AIR_20C), 2.0);
        assert_eq!(stable_dt(2994.0, speeds::WATER_25C), 2.0);
    }
}
