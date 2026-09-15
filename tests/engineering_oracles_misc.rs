//! Engineering-module oracles, group D (contact / fracture / topology /
//! 2-D dynamics / netcode).
//!
//! "Deterministic" and "correct" are different properties. Every test here
//! compares a module against a textbook closed form or an independent f64
//! reference, and states the tolerance and where it comes from (f32 SDF
//! evaluation, `Fix128` truncation, a rounded constant in the module, …).
//!
//! Modules that are empirical models with no closed form (the FDM layer
//! strength lookup, the delta-compression threshold) get invariant tests
//! only (limits, symmetry, exact round-trip, conservation) and say so in
//! their doc comment (`validation: invariants only`). `warp_risk` is an
//! empirical fit and is deliberately not covered.
//!
//! Tests marked `#[ignore = "src bug: …"]` assert the physically correct
//! value that the module currently does not produce; the numbers are in
//! the accompanying report. Nothing here touches `src/`.

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::fluid_netcode::{FluidDelta, FluidSnapshot};
use alice_physics::fracture::{Crack, FractureConfig, FractureModifier};
use alice_physics::kinematic_loop::{four_bar_linkage, LoopClosureConstraint};
use alice_physics::layer_adhesion::{EffectiveStrength, PrintOrientation};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::physics2d::{
    BodyType2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};
use alice_physics::rolling_contact::{
    basquin_cycles_to_failure, hertzian_sphere_sphere, materials, rolling_contact_life_cycles,
};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
use alice_physics::sdf_destruction::{
    destruction_from_explosion, destruction_from_projectile, DestructibleSdf, DestructionShape,
};
use alice_physics::sdf_force::{compute_sdf_force, SdfForceType};
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::soft_body_cut::{cut_cloth, cut_deformable, CutPlane};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

use core::f64::consts::PI;

// ============================================================================
// Helpers
// ============================================================================

/// Relative error `|a − b| / |b|` (b ≠ 0).
fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// `|a − b| ≤ 2⁻⁴⁸ ≈ 3.6e-15`: agreement within the truncation noise of the
/// I64F64 format (products of non-dyadic ratios carry a few thousand ulp
/// depending on evaluation order).
fn fix_close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() <= Fix128::from_raw(0, 1 << 16)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn assert_vec_close(got: Vec3Fix, want: (f64, f64, f64), tol: f64, what: &str) {
    let (gx, gy, gz) = (got.x.to_f64(), got.y.to_f64(), got.z.to_f64());
    let err = ((gx - want.0).powi(2) + (gy - want.1).powi(2) + (gz - want.2).powi(2)).sqrt();
    assert!(
        err <= tol,
        "{what}: got ({gx}, {gy}, {gz}), want {want:?} (abs err {err:.3e} > {tol:.1e})"
    );
}

/// Unit sphere SDF `φ = |x| − 1`, exact gradient `x / |x|`.
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            if len < 1e-10 {
                (0.0, 1.0, 0.0)
            } else {
                (x / len, y / len, z / len)
            }
        },
    )
}

/// Half-space `φ = y` (solid below the plane `y = 0`).
fn ground_plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

/// Midpoint-rule volume of `{φ < 0}` inside an axis-aligned box, `n` cells
/// per axis. The indicator-function quadrature error is `O(h)` per surface
/// cell but averages out over a smooth surface; at `h = 0.02` it is well
/// below 0.5 % of the removed volume (checked against the sphere below).
fn solid_volume(field: &dyn SdfField, min: (f32, f32, f32), max: (f32, f32, f32), n: usize) -> f64 {
    let h = (
        (max.0 - min.0) / n as f32,
        (max.1 - min.1) / n as f32,
        (max.2 - min.2) / n as f32,
    );
    let mut count = 0u64;
    for k in 0..n {
        let z = min.2 + (k as f32 + 0.5) * h.2;
        for j in 0..n {
            let y = min.1 + (j as f32 + 0.5) * h.1;
            for i in 0..n {
                let x = min.0 + (i as f32 + 0.5) * h.0;
                if field.distance(x, y, z) < 0.0 {
                    count += 1;
                }
            }
        }
    }
    count as f64 * (h.0 as f64) * (h.1 as f64) * (h.2 as f64)
}

// ============================================================================
// rolling_contact — Johnson, *Contact Mechanics* (1985) §4.2 / Basquin 1910
// ============================================================================

/// Hertz point contact (Johnson §4.2, eqs. 4.22–4.24): with
/// `1/R* = 1/R₁ + 1/R₂`, `1/E* = (1−ν₁²)/E₁ + (1−ν₂²)/E₂`,
/// `a = (3 P R* / 4 E*)^{1/3}`, `p₀ = 3 P / (2 π a²)`, and the pressure
/// distribution `p(r) = p₀ √(1 − r²/a²)` integrates back to `P`
/// (`∫ p dA = 2π p₀ a² / 3`). Depth of maximum shear `z = 0.48 a` for
/// ν = 0.3 (Johnson Table 4.1). Module is f32 with a deterministic cbrt,
/// hence 1e-5 on `a` and 2e-5 on `p₀ ∝ a⁻²`.
#[test]
fn rolling_contact_hertz_sphere_sphere_matches_johnson() {
    let (p, r1, r2, e, nu) = (100.0f64, 5.0e-3f64, 5.0e-3f64, 210.0e9f64, 0.30f64);
    let c = hertzian_sphere_sphere(
        p as f32, r1 as f32, r2 as f32, e as f32, e as f32, nu as f32, nu as f32,
    );

    let r_star = 1.0 / (1.0 / r1 + 1.0 / r2); // 2.5 mm
    let e_star = 1.0 / (2.0 * (1.0 - nu * nu) / e); // 115.38 GPa
    let a = (3.0 * p * r_star / (4.0 * e_star)).cbrt(); // 0.1176 mm
    let p0 = 3.0 * p / (2.0 * PI * a * a); // 3.45 GPa
    assert!(
        rel_err(c.contact_radius_m as f64, a) < 1e-5,
        "a = {} m, Johnson (4.22) gives {a}",
        c.contact_radius_m
    );
    assert!(
        rel_err(c.peak_pressure_pa as f64, p0) < 2e-5,
        "p₀ = {} Pa, Johnson (4.24) gives {p0}",
        c.peak_pressure_pa
    );
    // Load balance: the Hertz pressure ellipsoid carries exactly P.
    let load_from_pressure = 2.0 * PI * (c.peak_pressure_pa as f64) * a * a / 3.0;
    assert!(
        rel_err(load_from_pressure, p) < 2e-5,
        "∫ p dA = {load_from_pressure} N, applied {p} N"
    );
    assert!(
        rel_err(c.max_shear_depth_m as f64, 0.48 * a) < 1e-5,
        "z(τ_max) = {} m, Johnson Table 4.1 gives 0.48 a = {}",
        c.max_shear_depth_m,
        0.48 * a
    );
    // Scaling law a ∝ P^{1/3}: eight times the load doubles the patch.
    let c8 = hertzian_sphere_sphere(
        8.0 * p as f32,
        r1 as f32,
        r2 as f32,
        e as f32,
        e as f32,
        nu as f32,
        nu as f32,
    );
    assert!(
        rel_err(c8.contact_radius_m as f64, 2.0 * c.contact_radius_m as f64) < 1e-5,
        "a(8P)/a(P) = {}, want 2",
        c8.contact_radius_m / c.contact_radius_m
    );
}

/// Sphere on a flat (Johnson §4.2 with `R₂ → ∞`, `R* = R₁`): the module
/// documents `f32::INFINITY` for the flat body. Oracle: the same closed
/// form with `1/R₂ = 0`.
#[test]
fn rolling_contact_sphere_on_flat_uses_sphere_radius_as_r_star() {
    let (p, r1, e, nu) = (250.0f64, 6.0e-3f64, 210.0e9f64, 0.30f64);
    let c = hertzian_sphere_sphere(
        p as f32,
        r1 as f32,
        f32::INFINITY,
        e as f32,
        e as f32,
        nu as f32,
        nu as f32,
    );
    let e_star = e / (2.0 * (1.0 - nu * nu));
    let a = (3.0 * p * r1 / (4.0 * e_star)).cbrt();
    assert!(
        rel_err(c.contact_radius_m as f64, a) < 1e-5,
        "a = {} m, flat-body closed form {a}",
        c.contact_radius_m
    );
    assert!(c.peak_pressure_pa.is_finite() && c.peak_pressure_pa > 0.0);
}

/// Basquin 1910 S–N power law `N_f = C σ⁻ᵐ`: σ = 2 GPa, C = 5e34, m = 3
/// gives exactly `5e34 / 8e27 = 6.25e6` cycles; doubling the stress divides
/// the life by `2ᵐ = 8`. Below the endurance limit the life is infinite
/// (Wöhler knee). The f32 `powf` goes through `exp(m ln σ)` with
/// `ln σ ≈ 21`, so ~1e-5 relative is the attainable precision; 1e-4 used.
#[test]
fn rolling_contact_basquin_life_matches_power_law() {
    let n = basquin_cycles_to_failure(2.0e9, 5.0e34, 3.0, 1.5e9);
    assert!(
        rel_err(n as f64, 6.25e6) < 1e-4,
        "N(2 GPa) = {n}, Basquin gives 6.25e6"
    );
    let n2 = basquin_cycles_to_failure(4.0e9, 5.0e34, 3.0, 1.5e9);
    assert!(
        rel_err((n / n2) as f64, 8.0) < 2e-4,
        "N(σ)/N(2σ) = {}, want 2³ = 8",
        n / n2
    );
    assert!(basquin_cycles_to_failure(1.5e9, 5.0e34, 3.0, 1.5e9).is_infinite());
    assert!(basquin_cycles_to_failure(0.0, 5.0e34, 3.0, 1.5e9).is_infinite());

    // Composition: life(P) = Basquin(p₀(P)) with the preset bearing steel.
    let (e, nu, c, m, sigma_e) = materials::bearing_steel_52100();
    let contact = hertzian_sphere_sphere(5000.0, 6.0e-3, 6.0e-3, e, e, nu, nu);
    let direct = basquin_cycles_to_failure(contact.peak_pressure_pa, c, m, sigma_e);
    let combined = rolling_contact_life_cycles(5000.0, 6.0e-3, 6.0e-3, e, e, nu, nu, c, m, sigma_e);
    assert!(
        rel_err(combined as f64, direct as f64) < 1e-6,
        "combined {combined} vs direct {direct}"
    );
    let p0 = contact.peak_pressure_pa as f64;
    let expected = c as f64 * p0.powf(-(m as f64));
    assert!(
        rel_err(combined as f64, expected) < 1e-4,
        "life = {combined}, C p₀⁻ᵐ = {expected}"
    );
}

// ============================================================================
// fracture — kinematic crack growth + capsule CSG + stress relaxation
// ============================================================================

/// Constant-velocity crack growth `L(t) = min(v t, L_max)` (the module's
/// kinematic law; the arrest length plays the role of the Griffith
/// critical length). With `Config::default()` (`v = 5`, `L_max = 3`) and a
/// dyadic `dt` every step is exact in f32, so the tip position
/// `start + d̂ L` is exact too. The arrest time `t* = L_max / v = 0.6 s`
/// must not depend on `dt`.
#[test]
fn fracture_crack_length_grows_linearly_and_arrests_at_max_length() {
    let cfg = FractureConfig::default();
    let seed = Crack {
        start: (0.25, -0.5, 0.75),
        end: (0.25, -0.5, 0.75),
        direction: (0.6, 0.0, 0.8),
        length: 0.0,
        active: true,
    };
    let (v, l_max) = (cfg.propagation_speed as f64, cfg.max_crack_length as f64);

    for &dt in &[1.0f32 / 16.0, 1.0 / 32.0, 1.0 / 64.0] {
        let mut m = FractureModifier::new(cfg, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        m.cracks.push(seed);
        let mut t = 0.0f64;
        let mut arrested_at = None;
        for step in 1..=64 {
            m.update(dt);
            t = step as f64 * dt as f64;
            let c = m.cracks[0];
            let want = (v * t).min(l_max);
            assert!(
                (c.length as f64 - want).abs() < 1e-6,
                "dt {dt}: L({t}) = {}, want min(v t, L_max) = {want}",
                c.length
            );
            assert!(
                (c.end.0 as f64 - (0.25 + 0.6 * want)).abs() < 1e-6
                    && (c.end.1 as f64 + 0.5).abs() < 1e-6
                    && (c.end.2 as f64 - (0.75 + 0.8 * want)).abs() < 1e-6,
                "dt {dt}: tip {:?}, want start + d̂ L",
                c.end
            );
            assert_eq!(c.active, v * t < l_max, "dt {dt}: active flag at t = {t}");
            if !c.active && arrested_at.is_none() {
                arrested_at = Some(t);
            }
        }
        let t_star = arrested_at.expect("crack must arrest within 64 steps");
        assert!(
            t_star >= l_max / v && t_star < l_max / v + dt as f64 + 1e-9,
            "dt {dt}: arrested at {t_star}, want first step ≥ L_max / v = {} (t = {t})",
            l_max / v
        );
    }
}

/// A finished crack is a capsule of half-width `w` around its segment, and
/// the modifier applies the CSG difference `max(φ, −φ_capsule)` (Quilez,
/// *Distance functions*, "opSubtraction"). Closed form for a segment on the
/// x axis from −1 to 1: `φ_capsule(p) = dist(p, segment) − w`, where
/// `dist = √(y² + z²)` beside the segment and `√((|x| − 1)² + y² + z²)`
/// past its ends. Points deep inside the body therefore read the distance
/// to the crack wall, `w − dist`, and points whose original distance is
/// ≥ 2 w are left untouched.
#[test]
fn fracture_capsule_subtraction_matches_segment_distance_closed_form() {
    let cfg = FractureConfig::default(); // crack_width = 0.02
    let w = cfg.crack_width as f64;
    let mut m = FractureModifier::new(cfg, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
    m.cracks.push(Crack {
        start: (-1.0, 0.0, 0.0),
        end: (1.0, 0.0, 0.0),
        direction: (1.0, 0.0, 0.0),
        length: 2.0,
        active: false,
    });

    let seg_dist = |x: f64, y: f64, z: f64| -> f64 {
        let ax = (x.abs() - 1.0).max(0.0);
        (ax * ax + y * y + z * z).sqrt()
    };
    let deep = -0.5f32; // original SDF value: deep inside the body
    for &(x, y, z) in &[
        (0.0, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.5, 0.0, 0.25),
        (-0.75, 0.1, -0.1),
        (1.4, 0.0, 0.0),
        (-1.3, 0.2, 0.2),
    ] {
        let got = m.modify_distance(x as f32, y as f32, z as f32, deep) as f64;
        let want = (deep as f64).max(w - seg_dist(x, y, z));
        assert!(
            (got - want).abs() < 1e-6,
            "φ' at ({x}, {y}, {z}) = {got}, want max(φ, w − dist) = {want}"
        );
    }
    // Outside the body by more than 2 w: untouched even on the crack line.
    let outside = m.modify_distance(0.0, 0.0, 0.0, 0.05);
    assert!(
        (outside - 0.05).abs() < 1e-7,
        "outside point changed: {outside}"
    );
    // A modifier with no cracks is the identity.
    let empty = FractureModifier::new(cfg, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
    assert_eq!(empty.modify_distance(0.1, 0.2, 0.3, -0.42), -0.42);
}

/// Stress relaxation: each update scales the field by `exp(−λ dt)` after a
/// zero-flux (mirror) Laplacian smoothing, so the total stress obeys
/// `S(t) = S₀ exp(−λ t)` exactly and the peak strictly decreases (maximum
/// principle for the heat equation, Carslaw & Jaeger §1.9). `λ = 0.1` from
/// `Config::default()`; the amplitude stays below the toughness so no
/// crack is seeded. f32 sums over 512 cells for 60 steps: 1e-4.
#[test]
fn fracture_stress_field_total_decays_exponentially_and_peak_diffuses() {
    let cfg = FractureConfig::default();
    let mut m = FractureModifier::new(cfg, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
    m.apply_stress_at(0.0, 0.0, 0.0, 10.0, 1.5);
    let total = |m: &FractureModifier| m.stress.data.iter().map(|&s| s as f64).sum::<f64>();
    let s0 = total(&m);
    let peak0 = m.stress.max_value();
    assert!(s0 > 0.0 && peak0 > 0.0);

    let dt = 1.0f32 / 60.0;
    for _ in 0..60 {
        m.update(dt);
    }
    let want = s0 * (-(cfg.stress_decay as f64) * 60.0 * dt as f64).exp();
    assert!(
        rel_err(total(&m), want) < 1e-4,
        "S(1 s) = {}, S₀ exp(−λ t) = {want}",
        total(&m)
    );
    assert!(
        m.stress.max_value() < peak0 * (-(cfg.stress_decay as f64) * 60.0 * dt as f64).exp() as f32,
        "peak {} must fall faster than pure decay {}",
        m.stress.max_value(),
        peak0
    );
    assert_eq!(m.active_crack_count(), 0, "10 < toughness 50: no seed");
}

// ============================================================================
// layer_adhesion — validation: invariants only (empirical FDM lookup)
// ============================================================================

/// `validation: invariants only`. The layer-bond factors (`σ_z = r σ_y`,
/// `τ_xy = 0.6 σ_y`, `τ_z = τ_xy (1 + r)/2`) are empirical FDM fits with no
/// closed form (von Mises would give `τ = σ_y/√3 = 0.577 σ_y`, Tresca
/// `0.5 σ_y`; the module's 0.6 is neither and is documented as "typical
/// polymer"). What *is* checkable: an isotropic material (`r = 1`) must
/// collapse to a single normal and a single shear allowable; `r < 1`
/// orders the allowables Z < XY and across-layer < within-layer; and every
/// factor of safety is `allowable / |σ|` (exact in `Fix128` for integer
/// loads), sign-blind, and `min_fos` is the component minimum.
#[test]
fn layer_adhesion_isotropic_limit_ordering_and_fos_identity() {
    let make = |r_num: i64, r_den: i64| MaterialProperties {
        id: 0,
        name: "test",
        category: MaterialCategory::Fdm,
        youngs_modulus_gpa: r(35, 10),
        yield_strength_mpa: Fix128::from_int(40),
        tensile_strength_mpa: Fix128::from_int(50),
        density_g_cm3: r(124, 100),
        print_temp_c: Fix128::from_int(200),
        glass_transition_c: Fix128::from_int(60),
        bridging_distance_mm: Fix128::from_int(20),
        shrinkage_ratio: r(2, 1000),
        anisotropy_z_ratio: r(r_num, r_den),
    };

    // Isotropic limit: r = 1 → no direction is special.
    let iso = EffectiveStrength::for_material(&make(1, 1), PrintOrientation::XYFlat);
    assert_eq!(iso.normal_x_mpa, iso.normal_y_mpa);
    assert_eq!(iso.normal_x_mpa, iso.normal_z_mpa);
    assert_eq!(iso.shear_xy_mpa, iso.shear_xz_mpa);
    assert_eq!(iso.shear_xy_mpa, iso.shear_yz_mpa);
    assert_eq!(iso.normal_x_mpa, Fix128::from_int(40));

    // Anisotropic: r = 3/4 → σ_z = 30, τ_xy = 24, τ_z = 24·7/8 = 21.
    let ani = EffectiveStrength::for_material(&make(3, 4), PrintOrientation::XYFlat);
    assert!(
        fix_close(ani.normal_z_mpa, Fix128::from_int(30)),
        "{:?}",
        ani.normal_z_mpa
    );
    assert!(
        fix_close(ani.shear_xy_mpa, Fix128::from_int(24)),
        "{:?}",
        ani.shear_xy_mpa
    );
    assert!(
        fix_close(ani.shear_xz_mpa, Fix128::from_int(21)),
        "{:?}",
        ani.shear_xz_mpa
    );
    assert!(ani.normal_z_mpa < ani.normal_x_mpa);
    assert!(ani.shear_xz_mpa < ani.shear_xy_mpa);
    assert!(
        ani.shear_xz_mpa < ani.normal_z_mpa,
        "shear allowable below tension"
    );

    // FoS = allowable / |σ|: 30 / 12 = 2.5 exactly; sign-blind.
    let fos_z = ani.fos_normal_z(Fix128::from_int(12));
    assert!(fix_close(fos_z, r(5, 2)), "FoS_z = {}", fos_z.to_f64());
    assert_eq!(ani.fos_normal_z(Fix128::from_int(-12)), fos_z);
    assert!(fix_close(ani.fos_normal_x(Fix128::from_int(16)), r(5, 2)));
    assert!(fix_close(ani.fos_shear_xy(Fix128::from_int(48)), r(1, 2)));
    assert!(fix_close(
        ani.fos_shear_xz(Fix128::from_int(7)),
        Fix128::from_int(3)
    ));

    // min_fos picks the smallest component (Z at 12 MPa here → 2.5, vs 4 and 6).
    let stress = [
        (Fix128::from_int(10), ani.normal_x_mpa),
        (Fix128::from_int(4), ani.normal_y_mpa),
        (Fix128::from_int(12), ani.normal_z_mpa),
        (Fix128::from_int(-6), ani.shear_xy_mpa),
        (Fix128::ZERO, ani.shear_yz_mpa),
        (Fix128::ZERO, ani.shear_xz_mpa),
    ];
    assert_eq!(ani.min_fos(&stress), fos_z);
    // Zero load in every component → the "infinite" sentinel, larger than any FoS.
    let none = [(Fix128::ZERO, ani.normal_x_mpa); 6];
    assert!(ani.min_fos(&none) > Fix128::from_int(1 << 40));
}

// ============================================================================
// kinematic_loop — mass-weighted projection (Müller et al. 2007 PBD §3.3)
// ============================================================================

/// One rigid projection pass of a coincident-point constraint moves the
/// bodies by `Δx_a = −w_a r`, `Δx_b = +w_b r` with `w = m⁻¹ / Σ m⁻¹`
/// (Müller et al., *Position Based Dynamics*, 2007, eq. 9–10), so the
/// residual vanishes exactly, the centre of mass is unchanged
/// (`m_a Δx_a + m_b Δx_b = 0`), and the displacement ratio is
/// `|Δx_a| / |Δx_b| = m_b / m_a`. Inverse masses 1 and 3 (masses 1 and ⅓)
/// give dyadic weights (1/4, 3/4): every value is exact in `Fix128`, so the
/// assertions are `==`.
#[test]
fn kinematic_loop_rigid_pass_closes_exactly_and_preserves_centre_of_mass() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let pa = v3(0.5, -1.0, 2.0);
    let pb = v3(4.5, 3.0, -2.0);
    let a = w.add_body(RigidBody::new(pa, Fix128::from_int(1)));
    let mut light = RigidBody::new(pb, Fix128::ONE);
    light.inv_mass = Fix128::from_int(3); // m_b = 1/3 exactly (1/3 itself is not dyadic)
    let b = w.add_body(light);
    let mut c = LoopClosureConstraint::centre_to_centre(a, b);
    c.local_anchor_a = v3(0.25, 0.0, 0.0);
    c.local_anchor_b = v3(0.0, -0.5, 0.0);

    let r0 = c.residual(&w);
    assert_eq!(r0, (pa + c.local_anchor_a) - (pb + c.local_anchor_b));

    c.apply(&mut w);
    assert_eq!(c.residual(&w), Vec3Fix::ZERO, "rigid closure must be exact");
    let (na, nb) = (w.bodies[a].position, w.bodies[b].position);
    assert_eq!(na, pa - r0 * r(1, 4), "Δx_a = −w_a r, w_a = 1/4");
    assert_eq!(nb, pb + r0 * r(3, 4), "Δx_b = +w_b r, w_b = 3/4");
    // m_a Δx_a + m_b Δx_b = 0 with m_a = 1, m_b = 1/3 ⇔ Δx_b = −3 Δx_a.
    assert_eq!(
        nb - pb,
        -((na - pa) * Fix128::from_int(3)),
        "centre of mass must not move"
    );
    // A static partner absorbs nothing: the dynamic body takes the whole residual.
    let mut w2 = PhysicsWorld::new(SolverConfig::default());
    let s = w2.add_body(RigidBody::new_static(pb));
    let d = w2.add_body(RigidBody::new(pa, Fix128::from_int(7)));
    let c2 = LoopClosureConstraint::centre_to_centre(d, s);
    c2.apply(&mut w2);
    assert_eq!(w2.bodies[s].position, pb);
    assert_eq!(w2.bodies[d].position, pb);
}

/// With compliance `c` the pass removes the fraction `1/(1 + c)` of the
/// residual, so repeated passes contract geometrically:
/// `r_n = r₀ (c / (1 + c))ⁿ` (a fixed-point iteration with ratio
/// `c/(1+c) < 1`). `c = 1` halves the residual each pass (exact dyadic),
/// `c = 3` leaves 3/4 of it.
#[test]
fn kinematic_loop_compliant_residual_contracts_geometrically() {
    for &(c_num, ratio) in &[(1i64, r(1, 2)), (3, r(3, 4))] {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = w.add_body(RigidBody::new(v3(2.0, -4.0, 8.0), Fix128::ONE));
        let mut c = LoopClosureConstraint::centre_to_centre(a, b);
        c.compliance = Fix128::from_int(c_num);
        let r0 = c.residual(&w);
        let mut expected = r0;
        for pass in 1..=6 {
            c.apply(&mut w);
            expected = expected * ratio;
            let got = c.residual(&w);
            assert!(
                fix_close(got.x, expected.x)
                    && fix_close(got.y, expected.y)
                    && fix_close(got.z, expected.z),
                "c = {c_num}, pass {pass}: residual {got:?}, want r₀ (c/(1+c))ⁿ = {expected:?}"
            );
        }
    }
}

/// A four-bar linkage keeps its link lengths: with the solver running,
/// `|x_crank − x_ground| = L_crank` and `|x_coupler − x_crank| = L_coupler`
/// (Norton, *Design of Machinery*, §4.5 — the loop-closure equation
/// `r₂ + r₃ = r₁ + r₄` is written in link *lengths*). The helper places the
/// bodies at the right initial spacing, so the lengths must survive a few
/// zero-gravity frames.
///
/// Before 1.2.0 the three joints were zero-anchor ball joints and the
/// solver collapsed the mechanism onto the ground point.
#[test]
fn kinematic_loop_four_bar_preserves_link_lengths_under_solver() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let (l_crank, l_coupler, l_rocker, l_ground) = (1.0, 2.0, 1.5, 2.5);
    let linkage = four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(l_crank),
        Fix128::from_f64(l_coupler),
        Fix128::from_f64(l_rocker),
        Fix128::from_f64(l_ground),
        Fix128::ONE,
    );
    for _ in 0..30 {
        w.step(r(1, 60));
        linkage.closure.apply(&mut w);
    }
    let dist = |i: usize, j: usize| {
        (w.bodies[i].position - w.bodies[j].position)
            .length()
            .to_f64()
    };
    let crank = dist(linkage.crank, linkage.ground);
    let coupler = dist(linkage.coupler, linkage.crank);
    assert!(
        (crank - l_crank).abs() < 1e-3,
        "|crank − ground| = {crank}, want L_crank = {l_crank}"
    );
    assert!(
        (coupler - l_coupler).abs() < 1e-3,
        "|coupler − crank| = {coupler}, want L_coupler = {l_coupler}"
    );
}

/// The links the four-bar helper registers are rigid distance constraints
/// between consecutive pins (ground↔crank, crank↔coupler, coupler↔rocker)
/// carrying the link lengths, the ground and rocker pins are static, and
/// the closure ties the rocker pin to the ground anchor `(r₁, 0, 0)`. Pinned
/// as an invariant so the chain order cannot silently change.
#[test]
fn kinematic_loop_four_bar_registers_a_ground_crank_coupler_rocker_chain() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let l = four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::from_int(2),
        Fix128::ONE,
        Fix128::from_int(3),
        Fix128::ONE,
    );
    let pairs = [
        (l.ground, l.crank, 1.0),
        (l.crank, l.coupler, 2.0),
        (l.coupler, l.rocker, 1.0),
    ];
    for (k, &(a, b, len)) in pairs.iter().enumerate() {
        let c = &w.distance_constraints[l.joints[k]];
        assert_eq!((c.body_a, c.body_b), (a, b), "link {k}");
        assert_eq!(c.target_distance.to_f64(), len, "link {k} length");
        assert_eq!(c.compliance, Fix128::ZERO, "rigid link {k}");
    }
    assert!(w.bodies[l.ground].is_static());
    assert!(w.bodies[l.rocker].is_static(), "O4 is a ground pin");
    assert!(!w.bodies[l.crank].is_static());
    assert!(!w.bodies[l.coupler].is_static());
    assert_eq!(l.closure.body_a, l.rocker);
    assert_eq!(l.closure.body_b, l.ground);
    assert_eq!(
        l.closure.local_anchor_b,
        Vec3Fix::new(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO)
    );
    assert_eq!(l.closure.residual(&w), Vec3Fix::ZERO);
}

// ============================================================================
// sdf_force — F = −k φ ∇φ on an analytic sphere
// ============================================================================

/// `Attract` is the harmonic potential `U = ½ k φ²`, `F = −∇U = −k φ ∇φ`.
/// For a sphere collider of world radius `R` at `c` (scale `s` on a unit
/// sphere: `φ = |x − c| − s`, `∇φ = (x − c)/|x − c|`) the closed form is
/// `F = −k (|x − c| − R) (x − c)/|x − c|`, capped at `max_force`. The SDF
/// is evaluated in f32 and converted with `from_f32`, hence 1e-5 absolute
/// on O(10) forces. A rotation of the collider must not change anything
/// (sphere symmetry).
#[test]
fn sdf_force_attract_matches_negative_gradient_of_harmonic_potential() {
    let centre = (1.0, -2.0, 0.5);
    let scale = 2.0;
    let k = 3.0;
    let sdf = SdfCollider::new_static(
        Box::new(unit_sphere()),
        v3(centre.0, centre.1, centre.2),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI),
    )
    .with_scale(Fix128::from_f64(scale));
    let attract = SdfForceType::Attract {
        strength: Fix128::from_f64(k),
        max_force: Fix128::from_int(100),
    };
    for &p in &[
        (4.0, -2.0, 0.5),  // outside on +x: φ = 1
        (1.0, 1.0, 0.5),   // outside on +y: φ = 1
        (1.0, -2.0, -2.5), // outside on −z: φ = 1
        (2.0, -1.0, 1.5),  // inside, off-axis: |d| = √3 → φ = √3 − 2
        (1.5, -2.0, 0.5),  // inside on +x: φ = −1.5
        (3.0, 0.0, 2.5),   // outside, off-axis: |d| = √12
    ] {
        let d = (p.0 - centre.0, p.1 - centre.1, p.2 - centre.2);
        let len = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
        let phi = len - scale;
        let mag = -k * phi / len;
        let want = (mag * d.0, mag * d.1, mag * d.2);
        let body = RigidBody::new(v3(p.0, p.1, p.2), Fix128::ONE);
        let f = compute_sdf_force(&body, &sdf, &attract);
        assert_vec_close(f, want, 1e-5, &format!("F = −k φ ∇φ at {p:?}"));
    }
    // Cap: far away the magnitude saturates at max_force along −∇φ.
    let far = RigidBody::new(v3(1.0 + 60.0, -2.0, 0.5), Fix128::ONE);
    let f = compute_sdf_force(&far, &sdf, &attract);
    assert_vec_close(f, (-100.0, 0.0, 0.0), 1e-5, "|F| capped at max_force");
}

/// `Contain`: outside the surface `F = −k φ ∇φ − c v` (linear restoring
/// force plus viscous damping), inside only `−c v`. `Repel`: the documented
/// quadratic falloff `F = k (1 − |φ|/ρ)² sign(φ) ∇φ` — away from the
/// *surface* on both sides (toward the centre when inside) — zero beyond
/// `ρ`, `k/4` at half range; the module takes `sign(0) = −1` on the
/// measure-zero surface itself, which is not pinned. `SdfVortex`:
/// `F = k (1 − |φ|/ρ) (â × ∇φ)/|â × ∇φ|`, perpendicular to both the axis and
/// the normal.
#[test]
fn sdf_force_contain_repel_and_vortex_match_their_closed_forms() {
    let sdf = SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);
    let (k, c) = (4.0, 0.5);
    let contain = SdfForceType::Contain {
        strength: Fix128::from_f64(k),
        damping: Fix128::from_f64(c),
    };
    let mut body = RigidBody::new(v3(0.0, 3.0, 0.0), Fix128::ONE); // φ = 2, n̂ = +y
    body.velocity = v3(1.0, -2.0, 4.0);
    let f = compute_sdf_force(&body, &sdf, &contain);
    assert_vec_close(
        f,
        (-0.5, -8.0 + 1.0, -2.0),
        1e-5,
        "contain outside: −k φ n̂ − c v",
    );
    body.position = v3(0.0, 0.25, 0.0); // inside
    let f_in = compute_sdf_force(&body, &sdf, &contain);
    assert_vec_close(f_in, (-0.5, 1.0, -2.0), 1e-9, "contain inside: −c v only");

    let rho = 2.0;
    let repel = SdfForceType::Repel {
        strength: Fix128::from_f64(k),
        range: Fix128::from_f64(rho),
    };
    for &(x, want_fx) in &[
        (1.5, k * 0.5625),  // φ = 0.5, outside → +x
        (2.0, k * 0.25),    // φ = 1 = ρ/2
        (0.5, -k * 0.5625), // φ = −0.5, inside → away from the surface = −x
        (3.5, 0.0),         // beyond ρ
        (-2.0, -k * 0.25),  // φ = 1 on the −x side → −x
    ] {
        let b = RigidBody::new(v3(x, 0.0, 0.0), Fix128::ONE);
        let f = compute_sdf_force(&b, &sdf, &repel);
        assert_vec_close(f, (want_fx, 0.0, 0.0), 1e-5, &format!("repel at x = {x}"));
    }

    let vortex = SdfForceType::SdfVortex {
        axis: Vec3Fix::UNIT_Y,
        strength: Fix128::from_f64(k),
        influence_distance: Fix128::from_f64(rho),
    };
    let b = RigidBody::new(v3(1.5, 0.0, 0.0), Fix128::ONE); // φ = 0.5, n̂ = +x
    let f = compute_sdf_force(&b, &sdf, &vortex);
    // ŷ × x̂ = −ẑ, falloff 1 − 0.5/2 = 3/4 → (0, 0, −3)
    assert_vec_close(f, (0.0, 0.0, -3.0), 1e-5, "vortex = k f (â × n̂)");
    assert!(fix_close(f.dot(Vec3Fix::UNIT_Y), Fix128::ZERO), "⟂ axis");
    assert!(fix_close(f.dot(Vec3Fix::UNIT_X), Fix128::ZERO), "⟂ normal");
}

// ============================================================================
// anisotropic_friction — orthotropic Coulomb (Zmitrowicz 1981)
// ============================================================================

/// Along a principal axis every anisotropic friction law reduces to Coulomb:
/// `F = −μ N t̂` with `μ = μ_static` for `|v| ≤ v_slip` and `μ_kinetic`
/// above (the module's stick/slip switch is inclusive at the threshold).
/// Exact in `Fix128` up to the non-dyadic preset coefficients.
#[test]
fn anisotropic_friction_principal_axes_reduce_to_coulomb() {
    let m = AnisotropicFriction::tyre_asphalt();
    let (tl, tt) = (Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z);
    let n = Fix128::from_int(250);
    let cases = [
        (v3(5.0, 0.0, 0.0), -m.longitudinal_kinetic, Fix128::ZERO),
        (v3(-5.0, 0.0, 0.0), m.longitudinal_kinetic, Fix128::ZERO),
        (v3(0.0, 0.0, 5.0), Fix128::ZERO, -m.transverse_kinetic),
        (v3(0.0, 0.0, -0.01), Fix128::ZERO, m.transverse_static),
        // Exactly at the slip threshold: static (inclusive).
        (
            Vec3Fix::new(m.slip_threshold_m_s, Fix128::ZERO, Fix128::ZERO),
            -m.longitudinal_static,
            Fix128::ZERO,
        ),
        // One ulp above: kinetic.
        (
            Vec3Fix::new(
                m.slip_threshold_m_s + Fix128::from_raw(0, 1),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            -m.longitudinal_kinetic,
            Fix128::ZERO,
        ),
    ];
    for (v, mu_x, mu_z) in cases {
        let f = m.friction_force(n, tl, tt, v);
        assert!(
            fix_close(f.x, n * mu_x) && fix_close(f.y, Fix128::ZERO) && fix_close(f.z, n * mu_z),
            "v = {v:?}: F = {f:?}, want (−μ N) along the slip axis = ({:?}, 0, {:?})",
            n * mu_x,
            n * mu_z
        );
    }
    // Velocity components normal to the surface are ignored; N ≤ 0 → no force.
    let f = m.friction_force(n, tl, tt, v3(5.0, 7.0, 0.0));
    assert!(fix_close(f.y, Fix128::ZERO));
    assert_eq!(
        m.friction_force(Fix128::ZERO, tl, tt, v3(5.0, 0.0, 0.0)),
        Vec3Fix::ZERO
    );
    assert_eq!(
        m.friction_force(-n, tl, tt, v3(5.0, 0.0, 0.0)),
        Vec3Fix::ZERO
    );
}

/// Isotropic limit of the orthotropic friction ellipse (Zmitrowicz, *A
/// theoretical model of anisotropic dry friction*, Wear 73 (1981); the
/// same "friction ellipse" is used by Bullet's and PhysX's anisotropic
/// friction): with `μ_long = μ_trans = μ` the ellipse is a circle and the
/// law must reduce to Coulomb, `F = −μ N v̂`, for *every* slip direction —
/// magnitude `μ N`, direction opposite to the slip.
///
/// The module sums `sign(v_i) μ_i t̂_i` per axis (`src/anisotropic_friction.rs:100–102`),
/// a box law: for slip along `(0.6, 0, 0.8)` it returns `−μ N (1, 0, 1)`,
/// magnitude `√2 μ N` (41 % too high) at 45° instead of along `−v̂`.
#[test]

fn anisotropic_friction_isotropic_limit_is_coulomb_for_oblique_slip() {
    let mu = r(8, 10);
    let m = AnisotropicFriction {
        longitudinal_static: mu,
        longitudinal_kinetic: mu,
        transverse_static: mu,
        transverse_kinetic: mu,
        slip_threshold_m_s: r(5, 100),
    };
    let n = Fix128::from_int(100);
    let v = v3(6.0, 0.0, 8.0); // |v| = 10, v̂ = (0.6, 0, 0.8)
    let f = m.friction_force(n, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, v);
    let want = (-0.8 * 100.0 * 0.6, 0.0, -0.8 * 100.0 * 0.8);
    assert_vec_close(f, want, 1e-9, "isotropic μ: F = −μ N v̂");
    assert!(
        (f.length().to_f64() - 80.0).abs() < 1e-9,
        "|F| = {}, Coulomb bound μ N = 80",
        f.length().to_f64()
    );
}

// ============================================================================
// physics2d — free fall, rotation, momentum, inertia, impulse–momentum
// ============================================================================

fn circle(mass: i64, radius: f64, pos: (f64, f64)) -> RigidBody2D {
    RigidBody2D::new_dynamic(
        Vec2Fix::new(Fix128::from_f64(pos.0), Fix128::from_f64(pos.1)),
        Fix128::from_int(mass),
        Shape2D::Circle {
            radius: Fix128::from_f64(radius),
        },
    )
}

fn cfg2d(gravity_y: i64, damping: Fix128, substeps: usize) -> PhysicsConfig2D {
    PhysicsConfig2D {
        gravity: Vec2Fix::new(Fix128::ZERO, Fix128::from_int(gravity_y)),
        damping,
        substeps,
        ..PhysicsConfig2D::default()
    }
}

/// Free fall `y = −½ g t²` (Newton). The substepped symplectic Euler scheme
/// is exactly `y_N = −½ g t² − ½ g t h` (`h` = substep), so the deviation
/// from the closed form is bounded by `½ g t h` and must shrink as the
/// substep count grows. With `Config::default()` (4 substeps, 0.99 frame
/// damping) the discrete closed form is
/// `y_{n+1} = y_n + v_n dt + g dt² (s+1)/(2s)`, `v_{n+1} = (v_n + g dt) d`
/// (same scheme as the 3-D solver, `tests/analytic_physics.rs` #2).
#[test]
fn physics2d_free_fall_matches_newton_and_default_config_discrete_form() {
    let (g, dt, frames) = (10.0, 1.0 / 60.0, 60);
    let mut prev_dev = f64::INFINITY;
    for &s in &[1usize, 4, 16] {
        let mut w = PhysicsWorld2D::new(cfg2d(-10, Fix128::ONE, s));
        let b = w.add_body(circle(1, 0.5, (0.0, 0.0)));
        for _ in 0..frames {
            w.step(Fix128::from_f64(dt));
        }
        let t = frames as f64 * dt;
        let y = w.bodies[b].position.y.to_f64();
        let exact = -0.5 * g * t * t;
        let bound = 0.5 * g * t * dt / s as f64;
        let dev = (y - exact).abs();
        assert!(
            dev <= bound + 1e-9,
            "s = {s}: y = {y}, −½ g t² = {exact}, |Δ| = {dev} > ½ g t h = {bound}"
        );
        assert!(
            dev < prev_dev,
            "s = {s}: error {dev} did not shrink from {prev_dev}"
        );
        prev_dev = dev;
        // Velocity is exact for symplectic Euler: v = g t.
        let v = w.bodies[b].velocity.y.to_f64();
        assert!(
            (v + g * t).abs() < 1e-9,
            "s = {s}: v = {v}, want −g t = {}",
            -g * t
        );
    }

    let mut w = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let b = w.add_body(circle(1, 0.5, (0.0, 0.0)));
    for _ in 0..frames {
        w.step(Fix128::from_f64(dt));
    }
    let (s, d) = (4.0, 0.99);
    let (mut y_ref, mut v) = (0.0, 0.0);
    for _ in 0..frames {
        y_ref -= v * dt + g * dt * dt * (s + 1.0) / (2.0 * s);
        v = (v + g * dt) * d;
    }
    let y = w.bodies[b].position.y.to_f64();
    assert!(
        (y - y_ref).abs() < 1e-6,
        "default config: y = {y}, discrete closed form {y_ref} (undamped −5)"
    );
}

/// Uniform rotation `θ = ω t` (no torque). The angle is integrated linearly
/// in every substep and the angular velocity is recovered exactly, so with
/// unit damping `θ_N = ω t` to `Fix128` truncation for any substep count.
/// With `Config::default()` the 0.99 frame damping gives the geometric
/// series `θ_N = ω₀ dt (1 − dᴺ)/(1 − d)`.
#[test]
fn physics2d_rotation_is_theta_equals_omega_t() {
    let (omega, dt, frames) = (1.5, 1.0 / 60.0, 90);
    for &s in &[1usize, 4, 16] {
        let mut w = PhysicsWorld2D::new(cfg2d(0, Fix128::ONE, s));
        let mut b = circle(2, 1.0, (0.0, 0.0));
        b.angular_velocity = Fix128::from_f64(omega);
        let i = w.add_body(b);
        for _ in 0..frames {
            w.step(Fix128::from_f64(dt));
        }
        let theta = w.bodies[i].angle.to_f64();
        let want = omega * frames as f64 * dt; // 2.25 rad
        assert!(
            (theta - want).abs() < 1e-12,
            "s = {s}: θ = {theta}, ω t = {want}"
        );
        assert!((w.bodies[i].angular_velocity.to_f64() - omega).abs() < 1e-12);
    }
    let mut w = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let mut b = circle(2, 1.0, (0.0, 0.0));
    b.angular_velocity = Fix128::from_f64(omega);
    let i = w.add_body(b);
    for _ in 0..frames {
        w.step(Fix128::from_f64(dt));
    }
    let d: f64 = 0.99;
    let want = omega * dt * (1.0 - d.powi(frames)) / (1.0 - d);
    assert!(
        (w.bodies[i].angle.to_f64() - want).abs() < 1e-9,
        "default config: θ = {}, geometric series {want}",
        w.bodies[i].angle.to_f64()
    );
}

/// Two circles in a head-on collision: the mass-weighted positional
/// projection moves the bodies by `−w_a d n̂` and `+w_b d n̂`, which keeps
/// `m_a x_a + m_b x_b` and hence the total momentum `m_a v_a + m_b v_b`
/// (velocities are position differences) exactly — Newton's third law in
/// impulse form. Masses 1 and 3 give dyadic weights. After the collision
/// the bodies must not overlap (`|x_b − x_a| ≥ r_a + r_b`).
#[test]
fn physics2d_head_on_circle_collision_conserves_momentum() {
    for &s in &[1usize, 4, 8] {
        let mut w = PhysicsWorld2D::new(cfg2d(0, Fix128::ONE, s));
        let mut a = circle(1, 0.5, (-1.0, 0.0));
        a.velocity = Vec2Fix::from_int(2, 0);
        let mut b = circle(3, 0.5, (1.0, 0.0));
        b.velocity = Vec2Fix::from_int(-1, 0);
        let ia = w.add_body(a);
        let ib = w.add_body(b);
        let p0 = -1.0; // 1·2 + 3·(−1)
        let mut collided = false;
        for frame in 0..90 {
            w.step(r(1, 60));
            let (ba, bb) = (&w.bodies[ia], &w.bodies[ib]);
            let p = ba.velocity.x.to_f64() + 3.0 * bb.velocity.x.to_f64();
            assert!(
                (p - p0).abs() < 1e-9,
                "s = {s}, frame {frame}: Σ m v = {p}, want {p0}"
            );
            assert!(
                ba.velocity.y.is_zero() && bb.velocity.y.is_zero(),
                "head-on collision must stay on the x axis"
            );
            let gap = bb.position.x.to_f64() - ba.position.x.to_f64();
            if ba.velocity.x.to_f64() < 2.0 - 1e-9 {
                collided = true;
            }
            if collided {
                assert!(
                    gap >= 1.0 - 1e-6,
                    "s = {s}, frame {frame}: overlap, gap = {gap}"
                );
            }
        }
        assert!(collided, "s = {s}: the circles never met");
        assert!(
            w.bodies[ia].velocity.x < w.bodies[ib].velocity.x,
            "separating"
        );
    }
}

/// Newton's law of restitution `v_sep = e v_app` (Meriam & Kraige,
/// *Dynamics*, §3/12) with the bodies' default `e = 0.5`: equal circles
/// approaching at 2 m/s must separate at 1 m/s. Energy cannot be created:
/// `v_sep ≤ v_app` always.
///
/// `RigidBody2D::restitution` (and `friction`) are never read by the solver
/// (`src/physics2d.rs:303–306` vs `solve_contacts_xpbd` `:1118`), and the
/// position projection is re-applied `iterations` times per substep on the
/// same contact without a multiplier (`:1137–1146`), so the separation
/// speed is set by the solver settings instead of by `e`.
#[test]
fn physics2d_head_on_collision_obeys_newton_restitution() {
    let mut w = PhysicsWorld2D::new(cfg2d(0, Fix128::ONE, 4));
    let mut a = circle(1, 0.5, (-1.0, 0.0));
    a.velocity = Vec2Fix::from_int(1, 0);
    let mut b = circle(1, 0.5, (1.0, 0.0));
    b.velocity = Vec2Fix::from_int(-1, 0);
    let ia = w.add_body(a);
    let ib = w.add_body(b);
    let e = w.bodies[ia].restitution.to_f64(); // 0.5 default
    for _ in 0..120 {
        w.step(r(1, 60));
    }
    let v_sep = w.bodies[ib].velocity.x.to_f64() - w.bodies[ia].velocity.x.to_f64();
    assert!(
        v_sep <= 2.0 + 1e-9,
        "separation {v_sep} m/s exceeds approach 2 m/s: energy created"
    );
    assert!(
        (v_sep - e * 2.0).abs() < 0.05,
        "v_sep = {v_sep}, Newton e·v_app = {}",
        e * 2.0
    );
    // precision parameters must not change the outcome: iterations 1 / 2 / 8
    // and substeps 1 / 4 / 16 give the same separation speed (a single
    // contact is resolved in one iteration once the penetration is
    // re-evaluated; the velocity pass sees the same approach speed)
    for &(iterations, substeps) in &[(1usize, 4usize), (2, 4), (8, 1), (8, 16)] {
        let mut w = PhysicsWorld2D::new(PhysicsConfig2D {
            iterations,
            ..cfg2d(0, Fix128::ONE, substeps)
        });
        let mut a = circle(1, 0.5, (-1.0, 0.0));
        a.velocity = Vec2Fix::from_int(1, 0);
        let mut b = circle(1, 0.5, (1.0, 0.0));
        b.velocity = Vec2Fix::from_int(-1, 0);
        let ia = w.add_body(a);
        let ib = w.add_body(b);
        for _ in 0..120 {
            w.step(r(1, 60));
        }
        let got = w.bodies[ib].velocity.x.to_f64() - w.bodies[ia].velocity.x.to_f64();
        assert!(
            (got - v_sep).abs() < 1e-6,
            "iterations {iterations} substeps {substeps}: v_sep {got} vs {v_sep}"
        );
    }
}

/// Coulomb sliding friction: a non-rotating body sliding on a static edge
/// decelerates at `μ g` (Meriam & Kraige §6/3), `v(t) = v₀ − μ g t`, and
/// stops (never reverses) at `t = v₀/(μ g)`. The pair coefficient is the
/// geometric mean `√(μ_a μ_b)`; both bodies carry `μ` so it is `μ`.
#[test]
fn physics2d_sliding_friction_decelerates_at_mu_g() {
    let g = 10.0;
    let mu = 0.5;
    for &substeps in &[1usize, 4, 16] {
        let mut w = PhysicsWorld2D::new(cfg2d(-10, Fix128::ONE, substeps));
        let mut ground = RigidBody2D::new_static(
            Vec2Fix::ZERO,
            Shape2D::Edge {
                start: Vec2Fix::from_int(-50, 0),
                end: Vec2Fix::from_int(50, 0),
            },
        );
        ground.friction = Fix128::from_f64(mu);
        w.add_body(ground);
        // resting exactly on the edge, sliding at 2 m/s, rotation locked
        let mut slider = circle(1, 0.5, (0.0, 0.5));
        slider.velocity = Vec2Fix::from_int(2, 0);
        slider.friction = Fix128::from_f64(mu);
        slider.restitution = Fix128::ZERO;
        slider.inv_inertia = Fix128::ZERO;
        let i = w.add_body(slider);
        let dt = 1.0 / 60.0;
        for frame in 1..=12 {
            w.step(r(1, 60));
            let t = frame as f64 * dt;
            let v = w.bodies[i].velocity.x.to_f64();
            // one substep of friction lag at most: μ g h
            assert!(
                (v - (2.0 - mu * g * t)).abs() <= mu * g * dt / substeps as f64 + 1e-6,
                "substeps {substeps} t = {t}: v = {v}, want v₀ − μ g t = {}",
                2.0 - mu * g * t
            );
        }
        // stops at t = 0.4 s and stays stopped
        for _ in 0..30 {
            w.step(r(1, 60));
        }
        let v = w.bodies[i].velocity.x.to_f64();
        assert!(v.abs() < 1e-6, "substeps {substeps}: v after stop = {v}");
        assert!(
            (w.bodies[i].position.y.to_f64() - 0.5).abs() < 1e-3,
            "rests on the edge"
        );
    }
}

/// Moments of inertia about the centroid (Meriam & Kraige, Appendix B):
/// disc `I = ½ m r²`; rectangle `I = m (w² + h²)/12` (the polygon formula
/// must reproduce it for a 2 × 2 square: `2m/3`); thin rod `I = m L²/12`.
/// Impulse–momentum (§3/9): an impulse `J` at lever arm `r` changes
/// `Δv = J/m` and `Δω = (r × J)/I`.
#[test]
fn physics2d_inertia_and_impulse_momentum_closed_forms() {
    let disc = circle(4, 1.5, (0.0, 0.0));
    assert!(
        fix_close(
            disc.inv_inertia,
            Fix128::ONE / (Fix128::from_int(4) * r(225, 100) * r(1, 2))
        ),
        "disc 1/I = {}, want 1/(½ m r²) = {}",
        disc.inv_inertia.to_f64(),
        1.0 / (0.5 * 4.0 * 2.25)
    );
    let square = RigidBody2D::new_dynamic(
        Vec2Fix::ZERO,
        Fix128::from_int(3),
        Shape2D::Polygon {
            vertices: vec![
                Vec2Fix::from_int(-1, -1),
                Vec2Fix::from_int(1, -1),
                Vec2Fix::from_int(1, 1),
                Vec2Fix::from_int(-1, 1),
            ],
        },
    );
    assert!(
        fix_close(square.inv_inertia, r(1, 2)),
        "square 1/I = {}, want 1/(m (w²+h²)/12) = 1/2",
        square.inv_inertia.to_f64()
    );
    let rod = RigidBody2D::new_dynamic(
        Vec2Fix::ZERO,
        Fix128::from_int(6),
        Shape2D::Edge {
            start: Vec2Fix::from_int(-2, 0),
            end: Vec2Fix::from_int(2, 0),
        },
    );
    assert!(
        fix_close(rod.inv_inertia, r(1, 8)),
        "rod 1/I = {}, want 12/(m L²) = 1/8",
        rod.inv_inertia.to_f64()
    );

    // J = (0, 3) at r = (2, 0) on a disc m = 2, r = 1 (I = 1): Δv = (0, 1.5), Δω = 6.
    let mut body = circle(2, 1.0, (5.0, 5.0));
    body.apply_impulse_at_point(Vec2Fix::from_int(0, 3), Vec2Fix::from_int(7, 5));
    assert_eq!(body.velocity, Vec2Fix::new(Fix128::ZERO, r(3, 2)));
    assert_eq!(body.angular_velocity, Fix128::from_int(6));
    // Static bodies ignore impulses.
    let mut wall = RigidBody2D::new_static(
        Vec2Fix::ZERO,
        Shape2D::Circle {
            radius: Fix128::ONE,
        },
    );
    wall.apply_impulse(Vec2Fix::from_int(9, 9));
    assert_eq!(wall.velocity, Vec2Fix::ZERO);
    assert_eq!(wall.body_type, BodyType2D::Static);
}

// ============================================================================
// sdf_destruction — CSG difference, volume accounting, bore geometry
// ============================================================================

/// CSG difference `max(φ_A, −φ_B)` (Quilez, "opSubtraction"). For a sphere
/// crater of radius `R` centred on a flat ground (`φ = y`), on the vertical
/// axis below the surface the result is the distance to the crater wall,
/// `R − |y|`, and the removed material is exactly the hemisphere
/// `⅔ π R³` (Archimedes). Midpoint quadrature at `h = 0.02` on the box
/// `[−1,1]×[−1,0]×[−1,1]`; the sphere-indicator quadrature error at this
/// resolution is ≈ 1e-3 relative, so 0.5 % of the removed volume is used.
/// Smooth subtraction blends *outward* (it is `−smooth_union(−A, B)`), so
/// it can only remove more material than the sharp cut.
#[test]
fn sdf_destruction_sphere_crater_removes_a_hemisphere() {
    let radius = 1.0f32;
    let mut sharp = DestructibleSdf::new(Box::new(ground_plane()));
    let (min, max) = ((-1.0, -1.0, -1.0), (1.0, 0.0, 1.0));
    let n = 100;
    let v_before = solid_volume(&sharp, min, max, n);
    assert!((v_before - 4.0).abs() < 1e-5, "box volume {v_before}"); // f32 cell products

    sharp.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, radius));
    for &y in &[-0.1f32, -0.5, -0.9] {
        let d = sharp.distance(0.0, y, 0.0);
        assert!(
            (d - (radius + y)).abs() < 1e-6,
            "φ'(0, {y}, 0) = {d}, want R − |y| = {}",
            radius + y
        );
    }
    let d_deep = sharp.distance(0.0, -1.5, 0.0);
    assert!(
        (d_deep + 0.5).abs() < 1e-6,
        "below the crater: {d_deep}, want −0.5"
    );
    let d_far = sharp.distance(3.0, -0.2, 0.0);
    assert!(
        (d_far + 0.2).abs() < 1e-6,
        "away from the crater: {d_far}, want −0.2"
    );

    let hemisphere = 2.0 * PI * (radius as f64).powi(3) / 3.0;
    let v_after = solid_volume(&sharp, min, max, n);
    assert!(
        (v_before - v_after - hemisphere).abs() < 0.005 * hemisphere,
        "removed {} m³, hemisphere ⅔πR³ = {hemisphere}",
        v_before - v_after
    );

    let mut smooth = DestructibleSdf::new(Box::new(ground_plane()));
    smooth.apply_destruction(destruction_from_explosion(Vec3Fix::ZERO, radius, 0.2));
    let v_smooth = solid_volume(&smooth, min, max, n);
    assert!(
        v_smooth < v_after,
        "smooth cut left {v_smooth} m³ ≥ sharp {v_after} m³"
    );
    for &(x, y, z) in &[
        (0.0, -0.5, 0.0),
        (0.5, -0.1, 0.3),
        (0.9, -0.3, 0.0),
        (1.2, -0.05, 0.0),
    ] {
        let (a, b) = (sharp.distance(x, y, z), smooth.distance(x, y, z));
        assert!(
            b >= a - 1e-6,
            "smooth φ' {b} < sharp {a} at ({x}, {y}, {z})"
        );
    }
    // Far from the blend band (|φ_A + φ_B| ≥ k) the two cuts coincide.
    let (a, b) = (
        sharp.distance(0.0, -0.5, 0.0),
        smooth.distance(0.0, -0.5, 0.0),
    );
    assert!(
        (a - b).abs() < 1e-6,
        "outside the band: sharp {a}, smooth {b}"
    );
    // The undamaged field is restored by reset.
    sharp.reset();
    assert!((solid_volume(&sharp, min, max, n) - 4.0).abs() < 1e-5);
}

/// Projectile bore: a cylinder of radius `R` and length `D` starting at the
/// entry point along `d̂`. In an everywhere-solid field (`φ = −10`) the cut
/// equals `−φ_cyl`, so along the bore `φ' = min(R − ρ, D/2 − |s − D/2|)`
/// with `s` the axial and `ρ` the radial coordinate (Quilez, "sdCappedCylinder",
/// interior branch). Checked for an axis-aligned and a 45° direction; the
/// quaternion built in `Fix128` reaches f32 at ~1e-6.
#[test]
fn sdf_destruction_projectile_bore_matches_capped_cylinder_closed_form() {
    let solid = ClosureSdf::new(|_, _, _| -10.0, |_, _, _| (0.0, 1.0, 0.0));
    let (radius, depth) = (0.2f32, 2.0f32);
    let entry = v3(0.5, -1.0, 0.25);
    let inv_sqrt2 = core::f64::consts::FRAC_1_SQRT_2;
    for &(dir, perp) in &[
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((inv_sqrt2, inv_sqrt2, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    ] {
        let mut dsdf = DestructibleSdf::new(Box::new(solid_clone(&solid)));
        dsdf.apply_destruction(destruction_from_projectile(
            entry,
            v3(dir.0, dir.1, dir.2),
            radius,
            depth,
        ));
        for &(s, rho) in &[
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.1),
            (1.7, 0.15),
            (2.0, 0.0),
            (1.0, 0.19),
        ] {
            let p = (
                entry.x.to_f64() + dir.0 * s + perp.0 * rho,
                entry.y.to_f64() + dir.1 * s + perp.1 * rho,
                entry.z.to_f64() + dir.2 * s + perp.2 * rho,
            );
            let got = dsdf.distance(p.0 as f32, p.1 as f32, p.2 as f32) as f64;
            let half = depth as f64 / 2.0;
            let want = (radius as f64 - rho).min(half - (s - half).abs());
            assert!(
                (got - want).abs() < 2e-6,
                "dir {dir:?}, s = {s}, ρ = {rho}: φ' = {got}, want −φ_cyl = {want}"
            );
        }
        // Outside the bore radius the solid is untouched.
        let q = (
            entry.x.to_f64() + dir.0 + perp.0 * 0.5,
            entry.y.to_f64() + dir.1 + perp.1 * 0.5,
            entry.z.to_f64() + dir.2 + perp.2 * 0.5,
        );
        let d_out = dsdf.distance(q.0 as f32, q.1 as f32, q.2 as f32);
        assert!(
            (d_out - (-(0.5 - radius))).abs() < 2e-6,
            "beside the bore: {d_out}, want −(ρ − R) = {}",
            -(0.5 - radius)
        );
    }
}

/// `ClosureSdf` is not `Clone`; rebuild the constant solid field per case.
fn solid_clone(_: &ClosureSdf) -> ClosureSdf {
    ClosureSdf::new(|_, _, _| -10.0, |_, _, _| (0.0, 1.0, 0.0))
}

// ============================================================================
// soft_body_cut — validation: invariants only (exact topology bookkeeping)
// ============================================================================

/// `validation: invariants only` (a plane cut has no "answer" beyond
/// bookkeeping). What must hold exactly: the two sides partition the
/// particle set; a closed convex ring crosses a plane exactly twice
/// (Jordan curve / convexity); a cubic lattice cut between two layers
/// loses exactly one edge per column; every new particle lies on the plane
/// (`n·(p − p₀) = 0`, the interpolation `t = dᵢ/(dᵢ − dⱼ)` is exact for
/// dyadic distances); and one intersection is created per removed edge.
#[test]
fn soft_body_cut_partition_crossing_count_and_intersections_on_plane() {
    // 3 × 3 × 3 lattice, unit spacing, 54 axis-aligned edges.
    let mut particles = Vec::new();
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                particles.push(Vec3Fix::from_int(i, j, k));
            }
        }
    }
    let id = |i: i64, j: i64, k: i64| (i + 3 * j + 9 * k) as usize;
    let mut edges = Vec::new();
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..3 {
                if i < 2 {
                    edges.push((id(i, j, k), id(i + 1, j, k)));
                }
                if j < 2 {
                    edges.push((id(i, j, k), id(i, j + 1, k)));
                }
                if k < 2 {
                    edges.push((id(i, j, k), id(i, j, k + 1)));
                }
            }
        }
    }
    assert_eq!(edges.len(), 54);
    let plane = CutPlane {
        point: v3(0.0, 0.5, 0.0),
        normal: Vec3Fix::UNIT_Y,
    };
    let cut = cut_deformable(&particles, &edges, &plane);
    assert_eq!(cut.side_a_particles.len(), 18, "layers j = 1, 2");
    assert_eq!(cut.side_b_particles.len(), 9, "layer j = 0");
    let mut all: Vec<usize> = cut
        .side_a_particles
        .iter()
        .chain(cut.side_b_particles.iter())
        .copied()
        .collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all, (0..27).collect::<Vec<_>>(), "sides partition the set");
    assert_eq!(
        cut.removed_constraints.len(),
        9,
        "one vertical edge per column"
    );
    assert_eq!(cut.new_particles.len(), cut.removed_constraints.len());
    for p in &cut.new_particles {
        assert_eq!(
            p.y,
            r(1, 2),
            "intersection must lie on y = 1/2 exactly: {p:?}"
        );
    }
    for &(i, j) in &cut.removed_constraints {
        let (di, dj) = (particles[i].y - r(1, 2), particles[j].y - r(1, 2));
        assert!(
            di.is_negative() != dj.is_negative(),
            "edge ({i}, {j}) does not cross"
        );
    }
    // The surviving edge count is what a topological cut must leave: 54 − 9.
    let kept = edges
        .iter()
        .filter(|e| !cut.removed_constraints.contains(e))
        .count();
    assert_eq!(kept, 45);

    // Convex octagon ring, radius 1: any plane through it crosses exactly 2 edges,
    // and the crossing points satisfy n·(p − p₀) = 0.
    let ring: Vec<Vec3Fix> = (0..8)
        .map(|k| {
            let a = k as f64 * PI / 4.0;
            v3(a.cos(), 0.0, a.sin())
        })
        .collect();
    let ring_edges: Vec<(usize, usize)> = (0..8).map(|k| (k, (k + 1) % 8)).collect();
    let oblique = CutPlane {
        point: v3(0.3, 0.0, 0.1),
        normal: v3(0.6, 0.0, 0.8),
    };
    let rc = cut_cloth(&ring, &ring_edges, &oblique);
    assert_eq!(
        rc.removed_constraints.len(),
        2,
        "{:?}",
        rc.removed_constraints
    );
    assert_eq!(rc.new_particles.len(), 2);
    for p in &rc.new_particles {
        let d = (*p - oblique.point).dot(oblique.normal);
        assert!(
            fix_close(d, Fix128::ZERO),
            "intersection off the plane by {}",
            d.to_f64()
        );
        assert!(
            (p.x.to_f64().powi(2) + p.z.to_f64().powi(2)).sqrt() < 1.0,
            "inside the ring"
        );
    }
    assert_eq!(rc.side_a_particles.len() + rc.side_b_particles.len(), 8);

    // Scaling the (unnormalised) normal changes nothing: t = dᵢ/(dᵢ − dⱼ) is homogeneous.
    let scaled = CutPlane {
        point: oblique.point,
        normal: oblique.normal * Fix128::from_int(3),
    };
    let rs = cut_cloth(&ring, &ring_edges, &scaled);
    assert_eq!(rs.side_a_particles, rc.side_a_particles);
    assert_eq!(rs.removed_constraints, rc.removed_constraints);
    for (p, q) in rs.new_particles.iter().zip(&rc.new_particles) {
        assert!(fix_close(p.x, q.x) && fix_close(p.z, q.z), "{p:?} vs {q:?}");
    }
    // Empty cut: nothing crosses, both lists empty.
    let far = CutPlane {
        point: v3(5.0, 0.0, 0.0),
        normal: Vec3Fix::UNIT_X,
    };
    let rf = cut_cloth(&ring, &ring_edges, &far);
    assert!(rf.removed_constraints.is_empty() && rf.new_particles.is_empty());
    assert_eq!(rf.side_b_particles.len(), 8);
}

// ============================================================================
// fluid_netcode — validation: invariants only (exact round-trip, FNV-1a)
// ============================================================================

/// `validation: invariants only`. Serialisation must be a bijection on the
/// full `Fix128` range: extreme `hi`/`lo` words, negative fractions, and
/// the most negative value come back bit-identical, `size_bytes` is
/// `4 + 2·48 N + 8 + 8`, a truncated buffer is rejected, and the checksum
/// of an empty state is the published FNV-1a 64-bit offset basis
/// `0xcbf29ce484222325` (Fowler–Noll–Vo; one flipped `lo` bit changes it).
#[test]
fn fluid_netcode_snapshot_round_trip_is_bit_exact_over_the_full_range() {
    let positions = vec![
        Vec3Fix::new(
            Fix128::from_raw(i64::MAX, u64::MAX),
            Fix128::from_raw(i64::MIN, 0),
            Fix128::from_raw(-1, 1),
        ),
        Vec3Fix::new(r(-7, 3), r(1, 3), Fix128::from_raw(0, 1)),
        Vec3Fix::from_int(0, 0, 0),
    ];
    let velocities = vec![
        Vec3Fix::new(Fix128::PI, -Fix128::TWO_PI, Fix128::HALF_PI),
        Vec3Fix::new(r(-1, 1_000_000_007), Fix128::NEG_ONE, r(123_456_789, 1000)),
        Vec3Fix::from_int(-9, 8, -7),
    ];
    let snap = FluidSnapshot::capture(&positions, &velocities, 77);
    assert_eq!(snap.particle_count, 3);
    assert_eq!(snap.frame, 77);
    assert_eq!(snap.size_bytes(), 4 + 2 * 48 * 3 + 8 + 8);
    let (p, v) = snap.restore().expect("restore");
    assert_eq!(p, positions, "positions must round-trip bit-exactly");
    assert_eq!(v, velocities, "velocities must round-trip bit-exactly");
    assert!(snap.verify(&positions, &velocities));

    // One ulp in the last component flips verification.
    let mut nudged = velocities.clone();
    nudged[2].z = nudged[2].z + Fix128::from_raw(0, 1);
    assert!(!snap.verify(&positions, &nudged));
    assert_ne!(
        FluidSnapshot::capture(&positions, &nudged, 77).checksum,
        snap.checksum
    );

    // Truncated payload → None; empty state → FNV offset basis.
    let mut cut = snap.clone();
    cut.positions.truncate(cut.positions.len() - 1);
    assert!(cut.restore().is_none());
    let empty = FluidSnapshot::capture(&[], &[], 0);
    assert_eq!(empty.checksum, 0xcbf2_9ce4_8422_2325);
    assert_eq!(empty.size_bytes(), 20);
    let (p0, v0) = empty.restore().expect("empty restore");
    assert!(p0.is_empty() && v0.is_empty());
}

/// Delta compression is lossless at threshold 0 for any change of at least
/// `2⁻³²` per component (`|Δ|² > 0` survives the `Fix128` product) —
/// `base ∘ apply(delta) == new` bit for bit and the delta's checksum equals
/// the full snapshot's — and with a threshold `τ` the reconstruction error
/// per particle is bounded by `τ` in both position and velocity (particles
/// that moved by ≤ τ are deliberately not shipped). `changed_indices` is
/// strictly ascending with one position/velocity per entry. Sub-`2⁻³²`
/// changes are covered by the ignored test below.
#[test]
fn fluid_netcode_delta_is_lossless_at_zero_threshold_and_tau_bounded_otherwise() {
    let n = 16;
    let old_pos: Vec<Vec3Fix> = (0..n).map(|i| Vec3Fix::from_int(i, 2 * i, -i)).collect();
    let old_vel: Vec<Vec3Fix> = (0..n).map(|i| v3(0.1 * i as f64, 0.0, 1.0)).collect();
    let mut new_pos = old_pos.clone();
    let mut new_vel = old_vel.clone();
    // Even particles: large move; particle 3: 2⁻²⁰; particle 5: velocity only, 0.05.
    for i in (0..n as usize).step_by(2) {
        new_pos[i] = new_pos[i] + v3(0.5, -0.25, 0.0);
    }
    new_pos[3].x = new_pos[3].x + Fix128::from_raw(0, 1 << 44);
    new_vel[5] = new_vel[5] + v3(0.0, 0.05, 0.0);

    let d0 = FluidDelta::compute(&old_pos, &old_vel, &new_pos, &new_vel, Fix128::ZERO, 10, 11);
    assert_eq!(d0.changed_count(), 8 + 2, "8 even + #3 + #5");
    assert_eq!(d0.positions.len(), d0.changed_count());
    assert_eq!(d0.velocities.len(), d0.changed_count());
    assert!(
        d0.changed_indices.windows(2).all(|w| w[0] < w[1]),
        "ascending"
    );
    let (mut bp, mut bv) = (old_pos.clone(), old_vel.clone());
    d0.apply(&mut bp, &mut bv);
    assert_eq!(bp, new_pos, "lossless positions at τ = 0");
    assert_eq!(bv, new_vel, "lossless velocities at τ = 0");
    assert_eq!(
        d0.checksum,
        FluidSnapshot::capture(&new_pos, &new_vel, 11).checksum
    );
    assert!(FluidSnapshot::capture(&new_pos, &new_vel, 11).verify(&bp, &bv));
    assert!((d0.compression_ratio(n as usize) - 10.0 / 16.0).abs() < 1e-7);

    let tau = r(1, 10);
    let dt = FluidDelta::compute(&old_pos, &old_vel, &new_pos, &new_vel, tau, 10, 11);
    assert_eq!(
        dt.changed_count(),
        8,
        "only the even particles exceed τ = 0.1"
    );
    let (mut bp, mut bv) = (old_pos.clone(), old_vel.clone());
    dt.apply(&mut bp, &mut bv);
    for i in 0..n as usize {
        let ep = (bp[i] - new_pos[i]).length();
        let ev = (bv[i] - new_vel[i]).length();
        assert!(
            ep <= tau && ev <= tau,
            "particle {i}: error ({ep:?}, {ev:?}) > τ"
        );
    }
    assert_eq!(bp[3], old_pos[3], "sub-threshold move not shipped");
    assert_eq!(bv[5], old_vel[5], "sub-threshold velocity not shipped");
    assert!((dt.compression_ratio(n as usize) - 0.5).abs() < 1e-7);
    assert_eq!(dt.base_frame, 10);
    assert_eq!(dt.frame, 11);
    // No change at all → empty delta, ratio 0; empty world → ratio 1 by convention.
    let none = FluidDelta::compute(&old_pos, &old_vel, &old_pos, &old_vel, Fix128::ZERO, 0, 1);
    assert_eq!(none.changed_count(), 0);
    assert_eq!(none.compression_ratio(n as usize), 0.0);
    assert_eq!(none.compression_ratio(0), 1.0);
}

/// Lockstep determinism needs the delta to be lossless at threshold 0 for
/// *every* representable change: a single last-place-unit difference in
/// one component must be shipped, otherwise the receiver's state diverges
/// while the delta's own checksum (computed from the sender's full state)
/// says it should match.
///
/// `FluidDelta::compute` compares `length_squared() > τ²`
/// (`src/fluid_netcode.rs:127–130`); the `Fix128` product of a
/// difference below `2⁻³²` truncates to exactly 0, so the change is
/// silently dropped and `apply` reproduces the *old* value.
#[test]
fn fluid_netcode_delta_ships_one_ulp_changes_at_zero_threshold() {
    let old_pos = vec![Vec3Fix::from_int(1, 2, 3); 4];
    let old_vel = vec![Vec3Fix::ZERO; 4];
    let mut new_pos = old_pos.clone();
    new_pos[2].y = new_pos[2].y + Fix128::from_raw(0, 1);
    let delta = FluidDelta::compute(&old_pos, &old_vel, &new_pos, &old_vel, Fix128::ZERO, 0, 1);
    assert_eq!(delta.changed_count(), 1, "the 1-ulp change must be shipped");
    let (mut bp, mut bv) = (old_pos, old_vel.clone());
    delta.apply(&mut bp, &mut bv);
    assert_eq!(
        bp, new_pos,
        "receiver must reproduce the sender's state bit-exactly"
    );
    assert!(FluidSnapshot::capture(&new_pos, &old_vel, 1).verify(&bp, &bv));
}
