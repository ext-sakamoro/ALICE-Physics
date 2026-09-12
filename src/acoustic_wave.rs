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
        assert!(speeds::AIR_20C < speeds::WATER_25C);
        assert!(speeds::WATER_25C < speeds::STEEL_LONGITUDINAL);
        assert!(speeds::CONCRETE_LONGITUDINAL < speeds::STEEL_LONGITUDINAL);
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
}
