//! Absolute temperature vs temperature rise in a thermal eigenstrain
//!
//! A clamped PLA cube is heated to a uniform **absolute** 65 K while its
//! stress-free reference is 25 K. The eigenstrain `ε_th = α ΔT I` is specified
//! on the rise, so the field that drives it has to be `65 − 25 = 40` K. This
//! example routes the same absolute field in twice — once naming the real
//! reference, once naming zero — and prints what the second one costs.
//!
//! The two numbers differ by the stress of the reference temperature itself,
//! `E α T_ref / (1 − 2ν) = 291.666667` MPa, which is not a small correction: it
//! is 62% of the right answer. `coupled_field::TemperatureRise` exists so that
//! the reference has to be named in the call rather than assumed, because a
//! bare `CoupledField` of temperatures and a bare `CoupledField` of rises are
//! the same type on the same grid, and both temperature owners in this crate
//! (`thermal::ThermalModifier`, `phase_change::PhaseChangeModifier`) fill
//! theirs absolutely.
//!
//! ```bash
//! cargo run --example thermoelastic_rise --features std
//! ```

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    solve_with_eigenstrain, BoundaryConditions, ElasticMaterial, SolverConfig, ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Young's modulus, MPa (PLA).
const E_MPA: f64 = 3500.0;
/// Poisson's ratio.
const NU: f64 = 0.35;
/// Linear expansion coefficient, K⁻¹.
const ALPHA_PER_K: f64 = 1.0 / 1000.0;
/// Side of the cube, mm.
const SIDE: f64 = 4.0;
/// The stress-free reference the mesh was built at, K.
const REFERENCE_K: i64 = 25;
/// The absolute temperature the body is held at, K.
const ABSOLUTE_K: i64 = 65;

/// `σ = −E α ΔT / (1 − 2ν)`, MPa — every normal component of the fully
/// suppressed stress.
///
/// Clamping every boundary node of a body under a uniform eigenstrain admits
/// `u ≡ 0`, so `ε = 0` and `σ = −(3λ + 2μ) α ΔT I` with
/// `3λ + 2μ = E/(1 − 2ν)`, and no shear.
fn suppressed_stress_mpa(delta_t_k: f64) -> f64 {
    -E_MPA * (ALPHA_PER_K * delta_t_k) / (1.0 - 2.0 * NU)
}

/// Node index within an `(n+1)³` lattice.
fn node(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// A cube of side `n·h`, split into Kuhn 6-tet cells.
fn kuhn_cube(n: usize, h: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push([
                    (i as f64 * h) as f32,
                    (j as f64 * h) as f32,
                    (k as f64 * h) as f32,
                ]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(n, i, j, k);
                    for (slot, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[slot + 1] = node(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// Solve the clamped cube with the eigenstrain read from `rise`, and report the
/// worst normal stress component together with the worst shear.
fn worst_stress(mesh: &SdfTetMesh, bc: &BoundaryConditions, rise: &TemperatureRise) -> (f64, f64) {
    let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100))
        .expect("E > 0 and ν in (-1, 0.5)");
    let out = solve_with_eigenstrain(
        mesh,
        &material,
        bc,
        &SolverConfig::default(),
        Some(ThermalExpansion::from_rise(
            rise,
            Fix128::from_ratio(1, 1000),
        )),
    )
    .expect("a clamped cube under a uniform eigenstrain is well posed");

    let mut normal = 0.0_f64;
    let mut shear = 0.0_f64;
    for s in &out.element_stress {
        for c in [s.xx, s.yy, s.zz] {
            if c.to_f64().abs() > normal.abs() {
                normal = c.to_f64();
            }
        }
        for c in [s.xy, s.yz, s.zx] {
            shear = shear.max(c.to_f64().abs());
        }
    }
    (normal, shear)
}

fn main() {
    let n = 2usize;
    let mesh = kuhn_cube(n, SIDE / n as f64);

    // Clamp every boundary node, leaving the single interior node free. A
    // uniform eigenstrain then has `u ≡ 0` as its exact solution, so the stress
    // is the fully suppressed one and the closed form applies element by
    // element.
    let mut bc = BoundaryConditions::new();
    let last = SIDE as f32;
    for (v, p) in mesh.vertices.iter().enumerate() {
        let on_face = p
            .iter()
            .any(|c| *c <= 0.0 || *c >= last - (SIDE * 1e-9) as f32);
        if on_face {
            bc.fix(u32::try_from(v).expect("fits"));
        }
    }

    // One absolute temperature field, exactly as a temperature owner would
    // publish it: `ambient_temperature` copied into every cell.
    let absolute = CoupledField::try_new_filled(
        3,
        3,
        3,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
        ),
        Fix128::from_int(ABSOLUTE_K),
    )
    .expect("a 3³ grid over the cube is valid");

    // The right way: name the reference the mesh was built at, so the field the
    // eigenstrain reads is the rise.
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
    let (right, right_shear) = worst_stress(&mesh, &bc, &rise);

    // The mistake, in the only shape the type still permits: naming a zero
    // reference for a field that is not measured from zero. Nothing about the
    // call is ill-formed — the field is a valid `CoupledField` on the right
    // grid — so this is the error the type turns from invisible into stated.
    let as_if_rise = TemperatureRise::from_absolute(&absolute, Fix128::ZERO);
    let (wrong, _) = worst_stress(&mesh, &bc, &as_if_rise);

    let delta_t = (ABSOLUTE_K - REFERENCE_K) as f64;
    println!("absolute temperature   T     = {ABSOLUTE_K} K");
    println!("stress-free reference  T_ref = {REFERENCE_K} K");
    println!("rise the law wants     ΔT    = {delta_t} K");
    println!();
    println!(
        "from_absolute(field, {REFERENCE_K:>2})  σ = {right:>12.6} MPa   (closed form {:.6})",
        suppressed_stress_mpa(delta_t)
    );
    println!(
        "from_absolute(field,  0)  σ = {wrong:>12.6} MPa   (closed form {:.6})",
        suppressed_stress_mpa(ABSOLUTE_K as f64)
    );
    println!(
        "cost of not subtracting     = {:>12.6} MPa   (E α T_ref / (1 − 2ν) = {:.6})",
        (wrong - right).abs(),
        suppressed_stress_mpa(REFERENCE_K as f64).abs()
    );
    println!(
        "                            = {:.1}% of the right answer",
        100.0 * (wrong - right).abs() / right.abs()
    );
    println!();
    println!("worst shear (hydrostatic scene, so this is round-off) = {right_shear:.3e} MPa");
}
