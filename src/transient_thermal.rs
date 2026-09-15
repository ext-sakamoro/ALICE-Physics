//! Transient thermal conduction with temperature-dependent material
//! properties.
//!
//! Complements [`crate::thermal`] (constant-diffusivity SDF-surface
//! thermal modifier) with a **1-D transient heat-equation** helper that
//! respects the temperature-dependence of thermal conductivity `k(T)`,
//! specific heat `cp(T)`, and density `ρ(T)`. The three combined give
//! the **local thermal diffusivity**
//!
//! ```text
//! α(T) = k(T) / (ρ(T) · cp(T))    [m² / s]
//! ```
//!
//! which drives the explicit-Euler update
//!
//! ```text
//! T_new[i] = T_old[i] + dt · α(T[i]) · (T[i+1] − 2·T[i] + T[i−1]) / dx²
//! ```
//!
//! # Scope
//!
//! Ships:
//!
//! - Property evaluation (`conductivity_at`, `specific_heat_at`,
//!   `density_at`, `diffusivity_at`, `heat_capacity_at`).
//! - Four material presets (`steel_1018`, `aluminum_6061`,
//!   `pla_polymer`, `titanium_ti6al4v`) fitted to open literature over
//!   the 273 – 1000 K range for the metals and 293 – 473 K for PLA.
//! - `transient_step_1d` — explicit-Euler advance on a 1-D grid with
//!   Neumann (zero-flux) end conditions.
//! - `stable_dt_1d` — CFL upper bound `dt < dx² / (2 · max α)`.
//!
//! Deferred to future work: implicit / Crank–Nicolson integration,
//! 3-D grid variant (callers can wire a per-cell diffusivity into
//! [`crate::sim_field::ScalarField3D::diffuse`] using
//! `diffusivity_at`), phase-change hysteresis (see
//! [`crate::phase_change`]), radiation-only boundary conditions.

/// Single-property temperature dependence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureDependence {
    /// Constant value across the temperature range.
    Constant(f32),
    /// Polynomial `c0 + c1·T + c2·T²` (T in kelvin).
    Polynomial {
        /// Constant term.
        c0: f32,
        /// Linear coefficient.
        c1: f32,
        /// Quadratic coefficient.
        c2: f32,
    },
    /// Linear reference-point model `ref_value · (1 + coeff · (T − ref_temp))`.
    /// Convenient for materials whose property is quoted with a single
    /// temperature coefficient.
    Linear {
        /// Value at the reference temperature.
        ref_value: f32,
        /// Reference temperature (K).
        ref_temp: f32,
        /// Fractional change per kelvin.
        coeff: f32,
    },
}

impl TemperatureDependence {
    /// Evaluate the underlying property at `temperature` (K).
    #[must_use]
    pub fn evaluate(&self, temperature: f32) -> f32 {
        match *self {
            Self::Constant(v) => v,
            Self::Polynomial { c0, c1, c2 } => {
                c0 + c1 * temperature + c2 * temperature * temperature
            }
            Self::Linear {
                ref_value,
                ref_temp,
                coeff,
            } => ref_value * (1.0 + coeff * (temperature - ref_temp)),
        }
    }
}

/// Thermal material with temperature-dependent conductivity, specific
/// heat, and density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalMaterial {
    /// Stable identifier (`"steel_1018"`, `"aluminum_6061"`, …).
    pub name: &'static str,
    /// Thermal conductivity `k(T)` in `W / (m · K)`.
    pub conductivity: TemperatureDependence,
    /// Specific heat `cp(T)` in `J / (kg · K)`.
    pub specific_heat: TemperatureDependence,
    /// Density `ρ(T)` in `kg / m³`.
    pub density: TemperatureDependence,
    /// Reference temperature (K) used by `Linear` models. Callers may
    /// also use this as the default ambient state.
    pub reference_temperature: f32,
}

impl ThermalMaterial {
    /// Thermal conductivity at `temperature` (K).
    #[must_use]
    pub fn conductivity_at(&self, temperature: f32) -> f32 {
        self.conductivity.evaluate(temperature)
    }

    /// Specific heat at `temperature` (K).
    #[must_use]
    pub fn specific_heat_at(&self, temperature: f32) -> f32 {
        self.specific_heat.evaluate(temperature)
    }

    /// Density at `temperature` (K).
    #[must_use]
    pub fn density_at(&self, temperature: f32) -> f32 {
        self.density.evaluate(temperature)
    }

    /// Volumetric heat capacity `ρ(T) · cp(T)` in `J / (m³ · K)`.
    #[must_use]
    pub fn heat_capacity_at(&self, temperature: f32) -> f32 {
        self.density_at(temperature) * self.specific_heat_at(temperature)
    }

    /// Thermal diffusivity `α(T) = k(T) / (ρ(T) · cp(T))` in `m² / s`.
    ///
    /// Returns `0.0` if the volumetric heat capacity is not finite or
    /// non-positive (guards against pathological polynomial evaluations
    /// outside the calibrated range).
    #[must_use]
    pub fn diffusivity_at(&self, temperature: f32) -> f32 {
        let denom = self.heat_capacity_at(temperature);
        if !denom.is_finite() || denom <= 0.0 {
            return 0.0;
        }
        self.conductivity_at(temperature) / denom
    }

    /// AISI 1018 low-carbon steel over ~ 273 – 1000 K.
    ///
    /// Property fits from Incropera / DeWitt tables. Conductivity
    /// drops with temperature; specific heat rises linearly; density
    /// decreases slightly with thermal expansion.
    #[must_use]
    pub fn steel_1018() -> Self {
        Self {
            name: "steel_1018",
            // k ≈ 60 − 0.03·T   [W / (m · K)]
            conductivity: TemperatureDependence::Polynomial {
                c0: 60.0,
                c1: -0.03,
                c2: 0.0,
            },
            // cp ≈ 380 + 0.3·T  [J / (kg · K)]
            specific_heat: TemperatureDependence::Polynomial {
                c0: 380.0,
                c1: 0.3,
                c2: 0.0,
            },
            // ρ: linear thermal expansion, α_L ≈ 12e-6 / K.
            density: TemperatureDependence::Linear {
                ref_value: 7870.0,
                ref_temp: 293.15,
                coeff: -3.6e-5, // -3 · α_L (volumetric)
            },
            reference_temperature: 293.15,
        }
    }

    /// AA 6061-T6 aluminium alloy over ~ 273 – 800 K.
    #[must_use]
    pub fn aluminum_6061() -> Self {
        Self {
            name: "aluminum_6061",
            conductivity: TemperatureDependence::Polynomial {
                c0: 155.0,
                c1: 0.04,
                c2: 0.0,
            },
            specific_heat: TemperatureDependence::Polynomial {
                c0: 780.0,
                c1: 0.4,
                c2: 0.0,
            },
            density: TemperatureDependence::Linear {
                ref_value: 2700.0,
                ref_temp: 293.15,
                coeff: -6.9e-5,
            },
            reference_temperature: 293.15,
        }
    }

    /// PLA polymer over ~ 293 – 473 K (below glass transition +
    /// nozzle). The polynomial pieces are simple linear fits;
    /// callers who need the glass-transition drop should combine
    /// this with [`crate::phase_change`].
    #[must_use]
    pub fn pla_polymer() -> Self {
        Self {
            name: "pla_polymer",
            conductivity: TemperatureDependence::Polynomial {
                c0: 0.13,
                c1: 1.0e-4,
                c2: 0.0,
            },
            specific_heat: TemperatureDependence::Polynomial {
                c0: 1200.0,
                c1: 2.0,
                c2: 0.0,
            },
            density: TemperatureDependence::Linear {
                ref_value: 1240.0,
                ref_temp: 293.15,
                coeff: -1.8e-4,
            },
            reference_temperature: 293.15,
        }
    }

    /// Titanium alloy Ti-6Al-4V over ~ 300 – 900 K.
    #[must_use]
    pub fn titanium_ti6al4v() -> Self {
        Self {
            name: "titanium_ti6al4v",
            conductivity: TemperatureDependence::Polynomial {
                c0: 6.7,
                c1: 0.0011,
                c2: 0.0,
            },
            specific_heat: TemperatureDependence::Polynomial {
                c0: 546.0,
                c1: 0.16,
                c2: 0.0,
            },
            density: TemperatureDependence::Linear {
                ref_value: 4430.0,
                ref_temp: 293.15,
                coeff: -2.7e-5,
            },
            reference_temperature: 293.15,
        }
    }
}

/// Explicit-Euler advance of a 1-D temperature array by `dt`, with
/// Neumann (zero-flux) boundary conditions at both ends.
///
/// Uses the *local* diffusivity `α(T_i)` in each interior cell — this
/// is what makes the update transient-material-aware. Callers who want
/// a stable choice of `dt` should compare against [`stable_dt_1d`].
///
/// # Panics
///
/// Panics if `dx <= 0.0`. Zero-length or single-cell arrays are handled
/// gracefully (they are copied without changes).
pub fn transient_step_1d(temperatures: &mut [f32], material: &ThermalMaterial, dx: f32, dt: f32) {
    assert!(dx > 0.0, "dx must be positive");
    if temperatures.len() < 3 {
        return;
    }
    let n = temperatures.len();
    let inv_dx_squared = 1.0 / (dx * dx);
    let mut updated = Vec::with_capacity(n);
    // Zero-flux (Neumann) faces at the outer edges of cells 0 and n−1: the
    // virtual neighbour outside the rod equals the boundary cell itself, so
    // every cell in the slice is a physical cell of length `dx` and the rod is
    // `n·dx` long — the same finite-volume convention as
    // [`crank_nicolson_step_1d`]. Before 1.2.0 this function instead copied
    // cells 1 / n−2 into cells 0 / n−1 after the update, which made the two
    // end cells ghost cells (rod length `(n−2)·dx`) and gave a different
    // decay rate from the Crank–Nicolson step on the same array; the cosine
    // eigenmode oracle (`tests/engineering_oracles.rs`) exposed the mismatch.
    for i in 0..n {
        let left = if i == 0 {
            temperatures[0]
        } else {
            temperatures[i - 1]
        };
        let right = if i == n - 1 {
            temperatures[n - 1]
        } else {
            temperatures[i + 1]
        };
        let alpha = material.diffusivity_at(temperatures[i]);
        let laplacian = (right - 2.0 * temperatures[i] + left) * inv_dx_squared;
        updated.push(temperatures[i] + dt * alpha * laplacian);
    }
    temperatures.copy_from_slice(&updated);
}

/// Advance one time step with **Crank–Nicolson** (implicit-trapezoidal)
/// integration.
///
/// The linear system
///
/// ```text
/// (I − r/2 · L) T_new = (I + r/2 · L) T_old
/// ```
///
/// where `r = α(T_i) · dt / dx²` and `L` is the 1-D Laplacian
/// stencil, is solved by the Thomas algorithm (tridiagonal LU) in
/// `O(N)` time. Compared to [`transient_step_1d`] this is
/// **unconditionally A-stable** — arbitrary `dt` cannot amplify the
/// solution — and second-order accurate in time.
///
/// The material properties are frozen at the beginning-of-step
/// temperatures (linearised C–N); nonlinear iteration is left as
/// future work.
///
/// # Boundary conditions
///
/// Neumann (zero-flux) at both ends, matching [`transient_step_1d`].
///
/// # Panics
///
/// Panics if `dx <= 0.0` or `dt <= 0.0`. Arrays with fewer than three
/// cells are returned unmodified.
pub fn crank_nicolson_step_1d(
    temperatures: &mut [f32],
    material: &ThermalMaterial,
    dx: f32,
    dt: f32,
) {
    assert!(dx > 0.0, "dx must be positive");
    assert!(dt > 0.0, "dt must be positive");
    let n = temperatures.len();
    if n < 3 {
        return;
    }
    let inv_dx_squared = 1.0 / (dx * dx);

    // Per-cell r = α(T) · dt / dx² frozen at t^n.
    let mut r = vec![0.0_f32; n];
    for i in 0..n {
        r[i] = material.diffusivity_at(temperatures[i]) * dt * inv_dx_squared;
    }

    // RHS: (I + r/2 · L) T_old with Neumann ghost mirrors.
    let mut rhs = vec![0.0_f32; n];
    for i in 0..n {
        let left = if i == 0 {
            temperatures[i]
        } else {
            temperatures[i - 1]
        };
        let right = if i == n - 1 {
            temperatures[i]
        } else {
            temperatures[i + 1]
        };
        let laplacian = left - 2.0 * temperatures[i] + right;
        rhs[i] = temperatures[i] + 0.5 * r[i] * laplacian;
    }

    // Tridiagonal LHS. Neumann BC folds the missing ghost into the
    // diagonal by removing one −r/2 term at the end row.
    let mut sub = vec![0.0_f32; n];
    let mut diag = vec![0.0_f32; n];
    let mut sup = vec![0.0_f32; n];
    for i in 0..n {
        let half_r = 0.5 * r[i];
        let neumann_left = i == 0;
        let neumann_right = i == n - 1;
        sub[i] = if neumann_left { 0.0 } else { -half_r };
        sup[i] = if neumann_right { 0.0 } else { -half_r };
        // Diag: 1 + r − r/2·(neumann count) — the ghost mirror on the
        // implicit side absorbs one off-band coefficient into the diagonal.
        let ghost_fold = f32::from(u8::from(neumann_left) + u8::from(neumann_right));
        diag[i] = 1.0 + r[i] - half_r * ghost_fold;
    }

    thomas_solve_in_place(&mut sub, &mut diag, &mut sup, &mut rhs);
    temperatures.copy_from_slice(&rhs);
}

/// Thomas algorithm — solve a tridiagonal system in place, storing the
/// solution vector into `d`.
fn thomas_solve_in_place(a: &mut [f32], b: &mut [f32], c: &mut [f32], d: &mut [f32]) {
    let n = d.len();
    if n == 0 {
        return;
    }
    // Forward sweep.
    for i in 1..n {
        let m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    // Back substitution.
    d[n - 1] /= b[n - 1];
    for i in (0..n - 1).rev() {
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
    }
}

/// Nonlinear Crank–Nicolson step with **Picard iteration** on the
/// temperature-dependent material properties.
///
/// The linearised [`crank_nicolson_step_1d`] freezes `α(T)` at the
/// beginning-of-step temperatures. When `α` varies strongly across
/// the temperature range this introduces a first-order error in each
/// step. This nonlinear variant refines that estimate by iterating:
///
/// 1. Predict `T*` with `α` evaluated at the current guess.
/// 2. Re-evaluate `α(T*)` and repeat until the change between successive
///    iterates falls below `tolerance` (L∞) or `max_iterations` is hit.
///
/// A `tolerance` of `1e-4` and `max_iterations` of `5` are reasonable
/// defaults for engineering-grade transients.
///
/// Returns the number of Picard iterations actually performed (0 if
/// the array had fewer than three cells).
///
/// # Panics
///
/// Panics if `dx <= 0.0` or `dt <= 0.0`.
pub fn crank_nicolson_step_1d_nonlinear(
    temperatures: &mut [f32],
    material: &ThermalMaterial,
    dx: f32,
    dt: f32,
    tolerance: f32,
    max_iterations: u32,
) -> u32 {
    assert!(dx > 0.0, "dx must be positive");
    assert!(dt > 0.0, "dt must be positive");
    let n = temperatures.len();
    if n < 3 {
        return 0;
    }
    let initial: Vec<f32> = temperatures.to_vec();
    let mut current: Vec<f32> = temperatures.to_vec();
    let mut iterations = 0_u32;
    for _ in 0..max_iterations {
        iterations += 1;
        // Averaged temperature T̄ = ½(T_n + T_k) for the α evaluation —
        // trapezoidal Picard, matches the C–N mid-step assumption.
        let mut t_avg = vec![0.0_f32; n];
        for i in 0..n {
            t_avg[i] = 0.5 * (initial[i] + current[i]);
        }
        // Advance from initial using α(T̄); write result into a fresh
        // working buffer so we can compare against `current`.
        let mut next: Vec<f32> = initial.clone();
        crank_nicolson_step_1d_with_alpha(&mut next, material, &t_avg, dx, dt);
        // Convergence check in L∞.
        let mut max_delta = 0.0_f32;
        for i in 0..n {
            let d = (next[i] - current[i]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
        current = next;
        if max_delta < tolerance {
            break;
        }
    }
    temperatures.copy_from_slice(&current);
    iterations
}

/// Crank–Nicolson advance with a caller-supplied diffusivity sample
/// point per cell — factored out of `crank_nicolson_step_1d` and reused
/// by the Picard driver above.
fn crank_nicolson_step_1d_with_alpha(
    temperatures: &mut [f32],
    material: &ThermalMaterial,
    alpha_sample_temps: &[f32],
    dx: f32,
    dt: f32,
) {
    let n = temperatures.len();
    debug_assert_eq!(alpha_sample_temps.len(), n);
    if n < 3 {
        return;
    }
    let inv_dx_squared = 1.0 / (dx * dx);
    let mut r = vec![0.0_f32; n];
    for i in 0..n {
        r[i] = material.diffusivity_at(alpha_sample_temps[i]) * dt * inv_dx_squared;
    }
    let mut rhs = vec![0.0_f32; n];
    for i in 0..n {
        let left = if i == 0 {
            temperatures[i]
        } else {
            temperatures[i - 1]
        };
        let right = if i == n - 1 {
            temperatures[i]
        } else {
            temperatures[i + 1]
        };
        let laplacian = left - 2.0 * temperatures[i] + right;
        rhs[i] = temperatures[i] + 0.5 * r[i] * laplacian;
    }
    let mut sub = vec![0.0_f32; n];
    let mut diag = vec![0.0_f32; n];
    let mut sup = vec![0.0_f32; n];
    for i in 0..n {
        let half_r = 0.5 * r[i];
        let neumann_left = i == 0;
        let neumann_right = i == n - 1;
        sub[i] = if neumann_left { 0.0 } else { -half_r };
        sup[i] = if neumann_right { 0.0 } else { -half_r };
        let ghost_fold = f32::from(u8::from(neumann_left) + u8::from(neumann_right));
        diag[i] = 1.0 + r[i] - half_r * ghost_fold;
    }
    thomas_solve_in_place(&mut sub, &mut diag, &mut sup, &mut rhs);
    temperatures.copy_from_slice(&rhs);
}

/// 3-D explicit-Euler transient thermal advance on a Cartesian grid.
///
/// The temperature field is stored row-major with the ordering
/// `t[i + nx·(j + ny·k)]`. All six external faces use Neumann
/// (zero-flux) boundary conditions, matching [`transient_step_1d`].
/// Per-cell diffusivity `α(T)` respects the local temperature-dependent
/// material properties, driving the standard 7-point Laplacian.
///
/// # Panics
///
/// Panics if `dx <= 0.0` or `t.len() != nx·ny·nz`. Arrays with fewer
/// than three cells in any dimension are returned unmodified.
pub fn transient_step_3d(
    t: &mut [f32],
    nx: usize,
    ny: usize,
    nz: usize,
    material: &ThermalMaterial,
    dx: f32,
    dt: f32,
) {
    assert!(dx > 0.0, "dx must be positive");
    assert_eq!(t.len(), nx * ny * nz, "temperature slice length mismatch");
    if nx < 3 || ny < 3 || nz < 3 {
        return;
    }
    let inv_dx_squared = 1.0 / (dx * dx);
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut next: Vec<f32> = t.to_vec();
    // Zero-flux faces on all six sides: the virtual neighbour outside the
    // block equals the face cell itself (same convention as the 1-D steps
    // since 1.2.0; every cell is physical).
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = t[idx(i, j, k)];
                let xm = if i == 0 { c } else { t[idx(i - 1, j, k)] };
                let xp = if i == nx - 1 { c } else { t[idx(i + 1, j, k)] };
                let ym = if j == 0 { c } else { t[idx(i, j - 1, k)] };
                let yp = if j == ny - 1 { c } else { t[idx(i, j + 1, k)] };
                let zm = if k == 0 { c } else { t[idx(i, j, k - 1)] };
                let zp = if k == nz - 1 { c } else { t[idx(i, j, k + 1)] };
                let alpha = material.diffusivity_at(c);
                let laplacian = (xp + xm + yp + ym + zp + zm - 6.0 * c) * inv_dx_squared;
                next[idx(i, j, k)] = c + dt * alpha * laplacian;
            }
        }
    }
    t.copy_from_slice(&next);
}

/// CFL upper bound on the time step for [`transient_step_3d`].
///
/// The 3-D explicit stencil has a stricter bound than 1-D: `dt ≤ dx² /
/// (6 · max α)` for stability (compare to `dx²/(2 α)` in 1-D).
#[must_use]
pub fn stable_dt_3d(t: &[f32], material: &ThermalMaterial, dx: f32) -> f32 {
    assert!(dx > 0.0, "dx must be positive");
    let mut max_alpha = 0.0_f32;
    for &v in t {
        let alpha = material.diffusivity_at(v);
        if alpha > max_alpha {
            max_alpha = alpha;
        }
    }
    if max_alpha <= 0.0 || !max_alpha.is_finite() {
        return f32::INFINITY;
    }
    dx * dx / (6.0 * max_alpha)
}

/// CFL upper bound on the time step for [`transient_step_1d`].
///
/// Uses the maximum diffusivity across the array so a single choice of
/// `dt` remains stable even where the temperature-dependent properties
/// spike locally.
///
/// # Panics
///
/// Panics if `dx <= 0.0`.
#[must_use]
pub fn stable_dt_1d(temperatures: &[f32], material: &ThermalMaterial, dx: f32) -> f32 {
    assert!(dx > 0.0, "dx must be positive");
    let mut max_alpha = 0.0_f32;
    for &t in temperatures {
        let alpha = material.diffusivity_at(t);
        if alpha > max_alpha {
            max_alpha = alpha;
        }
    }
    if max_alpha <= 0.0 || !max_alpha.is_finite() {
        return f32::INFINITY;
    }
    dx * dx / (2.0 * max_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_dependence_returns_value_directly() {
        let d = TemperatureDependence::Constant(42.5);
        assert!((d.evaluate(300.0) - 42.5).abs() < 1.0e-6);
        assert!((d.evaluate(1000.0) - 42.5).abs() < 1.0e-6);
    }

    #[test]
    fn polynomial_matches_hand_calculation() {
        let d = TemperatureDependence::Polynomial {
            c0: 1.0,
            c1: 2.0,
            c2: 3.0,
        };
        // 1 + 2·2 + 3·4 = 17
        assert!((d.evaluate(2.0) - 17.0).abs() < 1.0e-5);
    }

    #[test]
    fn linear_expansion_around_reference() {
        let d = TemperatureDependence::Linear {
            ref_value: 100.0,
            ref_temp: 300.0,
            coeff: 0.01,
        };
        // At ref_temp, value == ref_value.
        assert!((d.evaluate(300.0) - 100.0).abs() < 1.0e-5);
        // Δ = 100 * (1 + 0.01 * (400 - 300)) = 100 * 2 = 200.
        assert!((d.evaluate(400.0) - 200.0).abs() < 1.0e-3);
    }

    #[test]
    fn steel_diffusivity_positive_and_finite() {
        let m = ThermalMaterial::steel_1018();
        let alpha = m.diffusivity_at(500.0);
        assert!(alpha.is_finite());
        assert!(alpha > 0.0);
    }

    #[test]
    fn aluminum_diffusivity_greater_than_steel() {
        let steel = ThermalMaterial::steel_1018();
        let al = ThermalMaterial::aluminum_6061();
        let ts = steel.diffusivity_at(500.0);
        let ta = al.diffusivity_at(500.0);
        assert!(
            ta > ts,
            "aluminum should conduct heat faster (α_al={ta}, α_steel={ts})"
        );
    }

    #[test]
    fn pla_diffusivity_orders_of_magnitude_below_metals() {
        let steel = ThermalMaterial::steel_1018();
        let pla = ThermalMaterial::pla_polymer();
        let ts = steel.diffusivity_at(400.0);
        let tp = pla.diffusivity_at(400.0);
        assert!(
            tp * 100.0 < ts,
            "polymer α should be ≪ steel (α_pla={tp}, α_steel={ts})"
        );
    }

    #[test]
    fn heat_capacity_matches_rho_times_cp() {
        let m = ThermalMaterial::steel_1018();
        let t = 500.0;
        let expected = m.density_at(t) * m.specific_heat_at(t);
        let actual = m.heat_capacity_at(t);
        assert!((expected - actual).abs() < 1.0e-3);
    }

    #[test]
    fn diffusivity_zero_when_heat_capacity_nonpositive() {
        // Construct a pathological material with zero density.
        let m = ThermalMaterial {
            name: "bad",
            conductivity: TemperatureDependence::Constant(10.0),
            specific_heat: TemperatureDependence::Constant(100.0),
            density: TemperatureDependence::Constant(0.0),
            reference_temperature: 293.15,
        };
        assert_eq!(m.diffusivity_at(300.0), 0.0);
    }

    #[test]
    fn presets_have_distinct_names() {
        assert_eq!(ThermalMaterial::steel_1018().name, "steel_1018");
        assert_eq!(ThermalMaterial::aluminum_6061().name, "aluminum_6061");
        assert_eq!(ThermalMaterial::pla_polymer().name, "pla_polymer");
        assert_eq!(ThermalMaterial::titanium_ti6al4v().name, "titanium_ti6al4v");
    }

    #[test]
    fn transient_step_leaves_uniform_field_unchanged() {
        let material = ThermalMaterial::steel_1018();
        let mut temperatures = vec![500.0_f32; 32];
        let dt = stable_dt_1d(&temperatures, &material, 0.001) * 0.5;
        transient_step_1d(&mut temperatures, &material, 0.001, dt);
        for &t in &temperatures {
            assert!((t - 500.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn transient_step_smooths_peak_toward_neighbours() {
        let material = ThermalMaterial::aluminum_6061();
        let mut temperatures = vec![300.0_f32; 32];
        temperatures[16] = 500.0;
        let dx = 0.001;
        let dt = stable_dt_1d(&temperatures, &material, dx) * 0.5;
        let initial_peak = temperatures[16];
        for _ in 0..100 {
            transient_step_1d(&mut temperatures, &material, dx, dt);
        }
        // Peak decayed; neighbouring cells rose above 300 K.
        assert!(temperatures[16] < initial_peak);
        assert!(temperatures[15] > 300.0);
        assert!(temperatures[17] > 300.0);
    }

    #[test]
    fn transient_step_conserves_total_energy_approx() {
        let material = ThermalMaterial::aluminum_6061();
        let mut temperatures = vec![300.0_f32; 64];
        for (i, t) in temperatures.iter_mut().enumerate() {
            *t += crate::det_math::sin((i as f32) * 0.05) * 50.0;
        }
        let dx = 0.001;
        let dt = stable_dt_1d(&temperatures, &material, dx) * 0.5;
        let initial_sum: f32 = temperatures.iter().sum();
        for _ in 0..50 {
            transient_step_1d(&mut temperatures, &material, dx, dt);
        }
        let final_sum: f32 = temperatures.iter().sum();
        let rel_error = ((final_sum - initial_sum) / initial_sum).abs();
        // Zero-flux BC + explicit diffusion should conserve to within
        // ~1% given the coarse fixed-precision integrator.
        assert!(rel_error < 1.0e-2, "energy drift {rel_error}");
    }

    #[test]
    fn stable_dt_returns_positive_for_valid_material() {
        let material = ThermalMaterial::steel_1018();
        let temperatures = vec![300.0_f32; 32];
        let dt = stable_dt_1d(&temperatures, &material, 0.001);
        assert!(dt > 0.0);
        assert!(dt.is_finite());
    }

    #[test]
    fn stable_dt_infinite_when_diffusivity_zero() {
        let material = ThermalMaterial {
            name: "inert",
            conductivity: TemperatureDependence::Constant(0.0),
            specific_heat: TemperatureDependence::Constant(1000.0),
            density: TemperatureDependence::Constant(1000.0),
            reference_temperature: 293.15,
        };
        let temperatures = vec![300.0_f32; 8];
        assert!(stable_dt_1d(&temperatures, &material, 0.001).is_infinite());
    }

    #[test]
    #[should_panic(expected = "dx must be positive")]
    fn stable_dt_panics_on_nonpositive_dx() {
        let material = ThermalMaterial::steel_1018();
        let temperatures = vec![300.0_f32; 4];
        let _ = stable_dt_1d(&temperatures, &material, 0.0);
    }

    // ---- Crank–Nicolson tests -------------------------------------------

    /// Constant-property material for isolating the C–N scheme from
    /// temperature-dependent nonlinearity.
    fn constant_material() -> ThermalMaterial {
        ThermalMaterial {
            name: "test_constant",
            conductivity: TemperatureDependence::Constant(50.0),
            specific_heat: TemperatureDependence::Constant(500.0),
            density: TemperatureDependence::Constant(7800.0),
            reference_temperature: 300.0,
        }
    }

    #[test]
    fn crank_nicolson_leaves_uniform_field_unchanged() {
        let material = constant_material();
        let mut t = vec![400.0_f32; 8];
        crank_nicolson_step_1d(&mut t, &material, 0.001, 0.01);
        for &v in &t {
            assert!((v - 400.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn crank_nicolson_smooths_peak_like_explicit() {
        let material = constant_material();
        let mut initial = vec![300.0_f32; 9];
        initial[4] = 500.0;
        let mut t_impl = initial.clone();
        let mut t_expl = initial;
        let dx = 0.001;
        let dt = stable_dt_1d(&t_expl, &material, dx) * 0.5;
        for _ in 0..50 {
            transient_step_1d(&mut t_expl, &material, dx, dt);
            crank_nicolson_step_1d(&mut t_impl, &material, dx, dt);
        }
        // Both schemes must diffuse the peak; C–N should stay bounded
        // by the initial extremum (500 K) and remain above the baseline.
        for &v in &t_impl {
            assert!((300.0 - 1.0e-2..=500.0 + 1.0e-2).contains(&v));
        }
        // At the peak cell, C–N and explicit agree to within ~15 K under
        // this well-below-CFL step (small phase shift from trapezoidal
        // vs forward-Euler dissipation).
        let diff = (t_impl[4] - t_expl[4]).abs();
        assert!(
            diff < 15.0,
            "peak diff {diff}, impl={} expl={}",
            t_impl[4],
            t_expl[4]
        );
    }

    #[test]
    fn crank_nicolson_is_stable_beyond_cfl() {
        // Explicit Euler diverges beyond the CFL bound; C–N must stay
        // finite and bounded in amplitude relative to the initial peak
        // (the CN amplification factor lies in [-1, 1]).
        let material = constant_material();
        let mut t = vec![300.0_f32; 10];
        t[5] = 800.0;
        let dx = 0.001;
        let cfl = stable_dt_1d(&t, &material, dx);
        // 3× CFL is unstable for forward Euler but well-behaved for CN.
        let dt = 3.0 * cfl;
        let peak_initial = 800.0_f32;
        for _ in 0..20 {
            crank_nicolson_step_1d(&mut t, &material, dx, dt);
            for &v in &t {
                assert!(v.is_finite(), "temperature not finite: {v}");
                // Amplitude never exceeds the initial max.
                assert!(v <= peak_initial + 1.0, "overshoot: {v}");
            }
        }
    }

    #[test]
    fn crank_nicolson_conserves_total_energy_approx() {
        // Constant-property material: the discrete Neumann Laplacian
        // has zero row sum on the interior; the two boundary rows are
        // biased by the ghost-fold. Energy drift is bounded by that
        // boundary term only.
        let material = constant_material();
        let mut t = vec![300.0_f32; 11];
        t[5] = 900.0;
        let dx = 0.001;
        let dt = stable_dt_1d(&t, &material, dx);
        let sum_initial: f32 = t.iter().sum();
        for _ in 0..100 {
            crank_nicolson_step_1d(&mut t, &material, dx, dt);
        }
        let sum_final: f32 = t.iter().sum();
        let delta_rel = (sum_final - sum_initial).abs() / sum_initial;
        assert!(delta_rel < 5.0e-3, "energy drift {delta_rel}");
    }

    #[test]
    #[should_panic(expected = "dt must be positive")]
    fn crank_nicolson_panics_on_nonpositive_dt() {
        let material = ThermalMaterial::steel_1018();
        let mut t = vec![300.0_f32; 6];
        crank_nicolson_step_1d(&mut t, &material, 0.001, 0.0);
    }

    // ---- Nonlinear Crank–Nicolson tests --------------------------------

    #[test]
    fn nonlinear_crank_nicolson_converges_on_constant_material() {
        // Constant α ⇒ Picard should converge in 1 iteration (T̄ never
        // changes α, so the next iterate equals the first).
        let material = constant_material();
        let mut t = vec![300.0_f32; 11];
        t[5] = 700.0;
        let iters = crank_nicolson_step_1d_nonlinear(&mut t, &material, 0.001, 0.001, 1.0e-4, 5);
        assert!(iters <= 2, "expected fast convergence, got {iters}");
        for &v in &t {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn nonlinear_crank_nicolson_matches_linear_on_constant_material() {
        // Under constant α, nonlinear should produce the same result
        // as the linearised C–N (Picard converges after one iterate).
        let material = constant_material();
        let dx = 0.001_f32;
        let dt = 0.001_f32;
        let mut initial = vec![300.0_f32; 11];
        initial[5] = 700.0;
        let mut t_lin = initial.clone();
        let mut t_nl = initial;
        crank_nicolson_step_1d(&mut t_lin, &material, dx, dt);
        let _ = crank_nicolson_step_1d_nonlinear(&mut t_nl, &material, dx, dt, 1.0e-5, 10);
        for i in 0..t_lin.len() {
            assert!(
                (t_lin[i] - t_nl[i]).abs() < 1.0e-3,
                "linear vs nonlinear diverge at {i}: {} vs {}",
                t_lin[i],
                t_nl[i]
            );
        }
    }

    // ---- 3-D transient thermal tests -----------------------------------

    fn make_uniform_field_3d(n: usize, value: f32) -> Vec<f32> {
        vec![value; n * n * n]
    }

    #[test]
    fn transient_step_3d_preserves_uniform_field() {
        let material = ThermalMaterial::steel_1018();
        let mut field = make_uniform_field_3d(5, 400.0);
        transient_step_3d(&mut field, 5, 5, 5, &material, 0.001, 0.001);
        for &v in &field {
            assert!((v - 400.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn transient_step_3d_smooths_central_peak() {
        let material = ThermalMaterial::steel_1018();
        let n = 5;
        let mut field = make_uniform_field_3d(n, 300.0);
        let ctr = 2;
        let idx = |i, j, k| i + n * (j + n * k);
        field[idx(ctr, ctr, ctr)] = 900.0;
        let dt = stable_dt_3d(&field, &material, 0.001) * 0.5;
        for _ in 0..20 {
            transient_step_3d(&mut field, n, n, n, &material, 0.001, dt);
        }
        // Peak decreased, neighbours warmed above ambient.
        assert!(field[idx(ctr, ctr, ctr)] < 900.0);
        assert!(field[idx(ctr + 1, ctr, ctr)] > 300.0);
    }

    #[test]
    fn stable_dt_3d_returns_positive_bound() {
        let material = ThermalMaterial::steel_1018();
        let field = make_uniform_field_3d(4, 400.0);
        let dt = stable_dt_3d(&field, &material, 0.001);
        assert!(dt > 0.0 && dt.is_finite());
    }

    #[test]
    fn transient_step_3d_zero_flux_faces_conserve_energy() {
        // 1.2.0: every cell is physical and the outer faces carry zero flux, so
        // the total energy (sum of temperatures at constant properties) is
        // conserved exactly up to f32 rounding, and the hot cell's heat spreads
        // to its 6 neighbours only (a face cell is *not* a copy of its
        // neighbour any more — that was the pre-1.2.0 ghost-cell convention).
        let material = ThermalMaterial {
            name: "const",
            conductivity: TemperatureDependence::Constant(40.0),
            specific_heat: TemperatureDependence::Constant(500.0),
            density: TemperatureDependence::Constant(8000.0),
            reference_temperature: 293.15,
        };
        let n = 4;
        let mut field = make_uniform_field_3d(n, 300.0);
        let idx = |i, j, k| i + n * (j + n * k);
        field[idx(1, 1, 1)] = 800.0;
        let before: f64 = field.iter().map(|&v| f64::from(v)).sum();
        let dt = stable_dt_3d(&field, &material, 0.001) * 0.1;
        transient_step_3d(&mut field, n, n, n, &material, 0.001, dt);
        let after: f64 = field.iter().map(|&v| f64::from(v)).sum();
        assert!((after - before).abs() < 1e-2, "energy {before} → {after}");
        assert!(field[idx(1, 1, 1)] < 800.0);
        assert!(field[idx(0, 1, 1)] > 300.0 && field[idx(2, 1, 1)] > 300.0);
        // a face cell not adjacent to the hot cell is untouched
        assert_eq!(field[idx(0, 0, 0)], 300.0);
        assert_eq!(field[idx(3, 3, 3)], 300.0);
    }

    #[test]
    fn nonlinear_crank_nicolson_stable_on_temp_dependent_material() {
        // Strongly temperature-dependent steel: nonlinear iterate must
        // remain bounded and finite even with a coarse tolerance.
        let material = ThermalMaterial::steel_1018();
        let mut t = vec![300.0_f32; 11];
        t[5] = 800.0;
        let dt = stable_dt_1d(&t, &material, 0.001);
        let iters = crank_nicolson_step_1d_nonlinear(&mut t, &material, 0.001, dt, 1.0e-4, 5);
        assert!(iters >= 1);
        for &v in &t {
            assert!(v.is_finite());
            assert!((250.0..=850.0).contains(&v), "out of envelope: {v}");
        }
    }
}
