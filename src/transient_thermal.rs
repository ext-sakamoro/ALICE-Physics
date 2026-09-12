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
    // Neumann BC: T_new[0] mirrors T_new[1], T_new[n-1] mirrors T_new[n-2].
    // We compute interior cells first, then reapply BC.
    updated.push(temperatures[0]);
    for i in 1..n - 1 {
        let alpha = material.diffusivity_at(temperatures[i]);
        let laplacian =
            (temperatures[i + 1] - 2.0 * temperatures[i] + temperatures[i - 1]) * inv_dx_squared;
        updated.push(temperatures[i] + dt * alpha * laplacian);
    }
    updated.push(temperatures[n - 1]);
    // Zero-flux BCs: mirror the interior cells.
    updated[0] = updated[1];
    updated[n - 1] = updated[n - 2];
    temperatures.copy_from_slice(&updated);
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
            *t += ((i as f32) * 0.05).sin() * 50.0;
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
}
