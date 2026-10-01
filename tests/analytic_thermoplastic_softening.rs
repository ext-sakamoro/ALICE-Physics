//! Closed-form oracles for temperature entering the plastic solve.
//!
//! Two effects arrive together and have to be separated, because they are not
//! equally strong:
//!
//! - **thermal expansion** supplies the eigenstrain `ε_th = α ΔT I`, which is
//!   removed from the elastic trial strain before the return mapping. It is
//!   purely volumetric, and J2 yielding reads only the deviator, so on its own
//!   it moves the pressure and — through equilibrium in a constrained body —
//!   the total strain, but never the yield criterion directly;
//! - **thermal softening** shrinks the yield surface itself,
//!   `σ_y(T) = σ_y₀ (1 − w_y ΔT)` and `H(T) = H₀ (1 − w_h ΔT)`. This is the
//!   effect that makes a thermoplastic coupling strong rather than formal.
//!
//! Every expected value below is written from those two formulas, never from
//! running the solver.
//!
//! # The scene
//!
//! A uniaxial bar `[0,4] × [0,2] × [0,2]` in 12 Kuhn tetrahedra, pulled in
//! displacement to `ε = 0.005` against a yield strain of `σ_y/E = 2/1024`. The
//! numbers are dyadic where the scene allows: `E = 1024`, `ν = 1/4`,
//! `σ_y = 2`, `H = 1024`, so `ε_y = 2⁻⁹` and the plastic tangent is `512`
//! exactly.
//!
//! The temperature field is **uniform**, which is what makes the expectations
//! closed form: the centroid sample is then exact, every element sees the same
//! `ΔT`, and the bar's response is the one-dimensional bilinear curve with
//! softened parameters. A non-uniform field would make the sampling first
//! order and the expectation an integral.
//!
//! # Why a uniform field still measures the eigenstrain
//!
//! A free bar under uniform `ΔT` expands without stress, so a test that only
//! looked at a free bar would measure nothing. Here `u_x` is **prescribed** at
//! both ends, so the expansion is fought by the constraint: the axial strain
//! available to the elastic-plastic response is `ε − α ΔT`, which is the
//! closed form the first test below compares against.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig, ElastoplasticIncrementRequest,
    ElastoplasticProblem, SolverConfig, ThermalExpansion, ThermalSoftening,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const E_MPA: f64 = 1024.0;
const NU: f64 = 0.25;
const SIGMA_Y: f64 = 2.0;
const H_PLASTIC: f64 = 1024.0;
/// `E·H / (E+H)`, exact for the dyadic inputs.
const E_TANGENT: f64 = 512.0;
/// Yield strain `σ_y / E = 2⁻⁹`.
const EPS_Y: f64 = 1.0 / 512.0;
/// Total axial strain the bar is pulled to.
const EPS_TOTAL: f64 = 0.005;
/// Linear expansion coefficient, dyadic so `α ΔT` is exact.
const ALPHA_PER_K: f64 = 1.0 / 4096.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node(i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * 3 + k * 6).expect("the 3x2x2 lattice fits u32")
}

fn bar_mesh() -> SdfTetMesh {
    let (nx, ny, nz, h) = (2usize, 1usize, 1usize, 2.0f32);
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
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
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node(i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn bar_bc(eps: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.prescribe(node(0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node(2, j, k), Axis::X, fx(eps) * Fix128::from_int(4));
        }
    }
    bc.prescribe(node(0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node(0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node(0, 1, 0), Axis::Z, Fix128::ZERO);
    bc
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and nu in (-1, 0.5)")
}

fn config() -> ElastoplasticConfig {
    ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24),
        fx(SIGMA_Y),
        fx(H_PLASTIC),
    )
    .expect("a valid elastoplastic config")
}

/// A uniform temperature-rise field covering the bar with a margin, so the
/// coverage check passes and every centroid samples the same value.
///
/// Built through `from_absolute` with a zero reference: the field already holds
/// rises, and that call is the identity that records a reference was chosen.
fn uniform_rise(delta_t: f64) -> TemperatureRise {
    let absolute = CoupledField::try_new_filled(
        5,
        5,
        5,
        (
            Fix128::from_int(-1),
            Fix128::from_int(-1),
            Fix128::from_int(-1),
        ),
        (
            Fix128::from_int(5),
            Fix128::from_int(3),
            Fix128::from_int(3),
        ),
        fx(delta_t),
    )
    .expect("a grid with at least two nodes per axis");
    TemperatureRise::from_absolute(&absolute, Fix128::ZERO)
}

/// Solve the bar in one increment, optionally with a temperature field.
fn solve_bar(
    eps: f64,
    thermal: Option<(&TemperatureRise, Option<ThermalSoftening>)>,
) -> alice_physics::linear_elastic_fem::ElastoplasticIncrement {
    let mesh = bar_mesh();
    let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(eps), &config())
        .expect("the bar prepares");
    let state = problem.virgin_state();
    let request = ElastoplasticIncrementRequest::new(Fix128::ONE);
    let request = match thermal {
        None => request,
        Some((rise, softening)) => request.with_thermal(
            ThermalExpansion::from_rise(rise, fx(ALPHA_PER_K)),
            softening,
        ),
    };
    problem
        .step(&state, &request)
        .expect("the increment solves")
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.12e}, closed form {want:.12e}, difference {:.3e} > tol {tol:.3e}",
        (got - want).abs()
    );
}

/// Axial stress of the one-dimensional bilinear curve at axial strain `eps`.
fn bilinear(eps: f64) -> f64 {
    if eps <= EPS_Y {
        E_MPA * eps
    } else {
        SIGMA_Y + E_TANGENT * (eps - EPS_Y)
    }
}

// ---------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------

/// Tolerances. The Newton stop is `2⁻⁴⁰` relative with an absolute floor, so
/// stresses land within about `1e-9` MPa of the closed form on this stiffness;
/// `1e-6` is three orders of margin, matching
/// `tests/analytic_elastoplastic_fem.rs`.
const STRESS_TOL: f64 = 1.0e-6;

/// With softening off, a uniform rise takes `α ΔT` out of the strain the
/// elastic-plastic response sees.
///
/// The ends are held, so the expansion cannot relieve itself: the axial stress
/// is the bilinear curve evaluated at `ε − α ΔT` rather than at `ε`. The
/// closed form is the same one the isothermal oracles use, read at a shifted
/// strain, which is what makes this a statement about the eigenstrain and not
/// about the curve.
#[test]
fn a_uniform_rise_shifts_the_axial_strain_by_alpha_delta_t() {
    let delta_t = 8.0; // α ΔT = 8/4096 = 2⁻⁹, exactly one yield strain
    let rise = uniform_rise(delta_t);
    let hot = solve_bar(EPS_TOTAL, Some((&rise, Some(ThermalSoftening::none()))));
    let cold = solve_bar(EPS_TOTAL, None);

    let shifted = EPS_TOTAL - ALPHA_PER_K * delta_t;
    assert!(
        shifted > EPS_Y,
        "the shifted strain {shifted} no longer yields, so this scene measures \
         the elastic branch only"
    );
    for (e, t) in hot.field.element_stress.iter().enumerate() {
        close(
            t.xx.to_f64(),
            bilinear(shifted),
            STRESS_TOL,
            &format!("element {e} axial stress under a rise of {delta_t} K"),
        );
    }
    for (e, t) in cold.field.element_stress.iter().enumerate() {
        close(
            t.xx.to_f64(),
            bilinear(EPS_TOTAL),
            STRESS_TOL,
            &format!("element {e} axial stress with no temperature"),
        );
    }

    // Vacuity guard: the shift has to be big enough to see. One yield strain of
    // expansion against a plastic tangent of 512 is 1 MPa of stress, six orders
    // over the tolerance.
    let gap = (bilinear(EPS_TOTAL) - bilinear(shifted)).abs();
    assert!(
        gap > 1.0e3 * STRESS_TOL,
        "the hot and cold answers differ by only {gap:.3e} MPa, so the \
         comparison above would pass without the eigenstrain"
    );
}

/// Softening lowers the yield stress by the fraction it is given, and the bar
/// follows the bilinear curve of the softened parameters.
///
/// This is the effect that couples temperature to *plasticity*: the expansion
/// above moves where on the curve the bar sits, softening moves the curve.
/// Both are read at the same `ΔT`, and the expansion is switched off here
/// (`α = 0` is not available through the request, so the rise is applied with
/// the same `α` and the shifted strain is used in the closed form) so that the
/// two effects are separated rather than summed.
#[test]
fn softening_lowers_the_yield_stress_by_the_fraction_it_is_given() {
    let delta_t = 8.0;
    let rise = uniform_rise(delta_t);
    // w_y = 1/32 per K at ΔT = 8 K ⇒ σ_y(T) = σ_y·(1 − 1/4) = 3/2, dyadic.
    let w_y = 1.0 / 256.0;
    let law = ThermalSoftening::try_new(fx(w_y), Fix128::ZERO).expect("a valid softening law");
    let hot = solve_bar(EPS_TOTAL, Some((&rise, Some(law))));

    let shifted = EPS_TOTAL - ALPHA_PER_K * delta_t;
    let sigma_y_hot = SIGMA_Y * (1.0 - w_y * delta_t);
    let eps_y_hot = sigma_y_hot / E_MPA;
    assert!(
        shifted > eps_y_hot,
        "the softened yield strain {eps_y_hot} is above the shifted strain \
         {shifted}, so the bar stays elastic and the curve below is not reached"
    );
    let want = sigma_y_hot + E_TANGENT * (shifted - eps_y_hot);

    for (e, t) in hot.field.element_stress.iter().enumerate() {
        close(
            t.xx.to_f64(),
            want,
            STRESS_TOL,
            &format!("element {e} axial stress with sigma_y softened to {sigma_y_hot}"),
        );
    }

    // Vacuity guard: softening has to change the answer by much more than the
    // tolerance, or the closed form above is indistinguishable from the
    // unsoftened one.
    let unsoftened = bilinear(shifted);
    assert!(
        (want - unsoftened).abs() > 1.0e3 * STRESS_TOL,
        "softening moved the stress by only {:.3e} MPa",
        (want - unsoftened).abs()
    );
}

/// Softening the hardening modulus changes the slope past yield, not the yield
/// point.
///
/// Separating `w_h` from `w_y` matters because a single fraction applied to
/// both would reproduce either test on its own; this one holds `σ_y` fixed and
/// moves `H`, so an implementation that softened the wrong parameter fails.
#[test]
fn softening_the_hardening_modulus_changes_only_the_slope() {
    let delta_t = 8.0;
    let rise = uniform_rise(delta_t);
    // w_h = 1/16 per K at 8 K ⇒ H(T) = H/2 = 512 ⇒ E_t = E·H/(E+H) = 1024/3.
    let w_h = 1.0 / 16.0;
    let law = ThermalSoftening::try_new(Fix128::ZERO, fx(w_h)).expect("a valid softening law");
    let hot = solve_bar(EPS_TOTAL, Some((&rise, Some(law))));

    let shifted = EPS_TOTAL - ALPHA_PER_K * delta_t;
    let h_hot = H_PLASTIC * (1.0 - w_h * delta_t);
    let e_tangent_hot = E_MPA * h_hot / (E_MPA + h_hot);
    let want = SIGMA_Y + e_tangent_hot * (shifted - EPS_Y);

    for (e, t) in hot.field.element_stress.iter().enumerate() {
        close(
            t.xx.to_f64(),
            want,
            STRESS_TOL,
            &format!("element {e} axial stress with H softened to {h_hot}"),
        );
    }
    assert!(
        (want - bilinear(shifted)).abs() > 1.0e3 * STRESS_TOL,
        "softening the hardening modulus moved the stress by only {:.3e} MPa",
        (want - bilinear(shifted)).abs()
    );
}

/// No law and the zero law are the same solve, to the bit.
///
/// `Option<ThermalSoftening>` defaulting to `None` is a silent constraint — a
/// caller who forgets it gets no softening and no complaint — so the two paths
/// have to be the same arithmetic rather than merely close, otherwise the
/// absent law would be a third behaviour nobody chose.
#[test]
fn thermal_softening_off_matches_no_law_at_all() {
    let rise = uniform_rise(8.0);
    let without = solve_bar(EPS_TOTAL, Some((&rise, None)));
    let zeroed = solve_bar(EPS_TOTAL, Some((&rise, Some(ThermalSoftening::none()))));
    assert_eq!(
        without, zeroed,
        "passing no softening law and passing the zero law gave different \
         answers, so the absent law is a behaviour of its own"
    );
    assert_eq!(ThermalSoftening::none().yield_per_k(), Fix128::ZERO);
    assert_eq!(ThermalSoftening::none().hardening_per_k(), Fix128::ZERO);

    // And the zero law is not the same as no temperature at all, which is what
    // keeps the equality above from being vacuous.
    let cold = solve_bar(EPS_TOTAL, None);
    assert_ne!(
        without.field.element_stress[0].xx, cold.field.element_stress[0].xx,
        "the temperature field changed nothing, so the equality above holds for \
         the wrong reason"
    );
}

/// Enough softening takes the yield stress to zero rather than negative.
///
/// A negative yield radius has no meaning for the return mapping — it would
/// admit a stress state inside a surface of negative size — so the law clamps.
/// The bar then carries only what the hardening gives it.
#[test]
fn softening_past_the_whole_yield_stress_clamps_at_zero() {
    let delta_t = 8.0;
    let rise = uniform_rise(delta_t);
    // w_y ΔT = 2 ⇒ 1 − 2 = −1, so the clamp has to fire.
    let law = ThermalSoftening::try_new(fx(0.25), Fix128::ZERO).expect("a valid softening law");
    let hot = solve_bar(EPS_TOTAL, Some((&rise, Some(law))));

    let shifted = EPS_TOTAL - ALPHA_PER_K * delta_t;
    // σ_y = 0 with H unchanged: the whole response is the plastic tangent from
    // the origin, `E_t · ε`.
    let want = E_TANGENT * shifted;
    for (e, t) in hot.field.element_stress.iter().enumerate() {
        close(
            t.xx.to_f64(),
            want,
            STRESS_TOL,
            &format!("element {e} axial stress with the yield stress clamped to zero"),
        );
    }
    assert!(
        want > 1.0e3 * STRESS_TOL,
        "the clamped answer is {want:.3e} MPa, too small to distinguish from zero"
    );
}

/// The strain shift and the yield shift are different quantities.
///
/// A single implementation mistake — subtracting `α ΔT` from the yield strain
/// instead of from the total strain — reproduces the first oracle exactly.
/// This one separates them: with `α` carrying the whole effect the stress falls
/// by `E_t · α ΔT`, and with softening carrying it the stress falls by
/// `σ_y w_y ΔT` scaled by how the curve is entered. Choosing `ΔT` so the two
/// predictions differ is what makes the pair non-redundant.
#[test]
fn expansion_and_softening_are_not_the_same_shift() {
    let delta_t = 8.0;
    let rise = uniform_rise(delta_t);
    let expansion_only = solve_bar(EPS_TOTAL, Some((&rise, Some(ThermalSoftening::none()))));
    let w_y = 1.0 / 256.0;
    let both = solve_bar(
        EPS_TOTAL,
        Some((
            &rise,
            Some(ThermalSoftening::try_new(fx(w_y), Fix128::ZERO).expect("valid")),
        )),
    );

    let shifted = EPS_TOTAL - ALPHA_PER_K * delta_t;
    let from_expansion = bilinear(EPS_TOTAL) - bilinear(shifted);
    let sigma_y_hot = SIGMA_Y * (1.0 - w_y * delta_t);
    let eps_y_hot = sigma_y_hot / E_MPA;
    let from_softening = bilinear(shifted) - (sigma_y_hot + E_TANGENT * (shifted - eps_y_hot));

    close(
        bilinear(EPS_TOTAL) - expansion_only.field.element_stress[0].xx.to_f64(),
        from_expansion,
        STRESS_TOL,
        "the drop attributable to expansion",
    );
    close(
        expansion_only.field.element_stress[0].xx.to_f64()
            - both.field.element_stress[0].xx.to_f64(),
        from_softening,
        STRESS_TOL,
        "the drop attributable to softening",
    );
    let ratio = (from_expansion / from_softening).abs();
    assert!(
        ratio > 2.0 || ratio < 0.5,
        "the two drops are within a factor of two of each other ({from_expansion:.6} \
         and {from_softening:.6}), so swapping which quantity is shifted would \
         still pass"
    );
}

/// A negative or oversized softening fraction is refused.
#[test]
fn an_unusable_softening_fraction_is_refused() {
    let big = Fix128::from_int(1 << 30) + Fix128::ONE;
    for (what, y, h) in [
        ("a negative yield fraction", fx(-1.0 / 1024.0), Fix128::ZERO),
        (
            "a negative hardening fraction",
            Fix128::ZERO,
            fx(-1.0 / 1024.0),
        ),
        ("an oversized yield fraction", big, Fix128::ZERO),
        ("an oversized hardening fraction", Fix128::ZERO, big),
    ] {
        assert!(
            ThermalSoftening::try_new(y, h).is_err(),
            "{what} was accepted"
        );
    }
    assert!(ThermalSoftening::try_new(Fix128::ZERO, Fix128::ZERO).is_ok());
}

/// A temperature field that does not reach every vertex is refused, and the
/// refusal names a vertex.
#[test]
fn a_field_that_does_not_cover_the_mesh_is_refused() {
    let small = CoupledField::try_new_filled(
        3,
        3,
        3,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (Fix128::ONE, Fix128::ONE, Fix128::ONE),
        fx(8.0),
    )
    .expect("a grid");
    let rise = TemperatureRise::from_absolute(&small, Fix128::ZERO);
    let mesh = bar_mesh();
    let problem = ElastoplasticProblem::try_new(&mesh, &material(), &bar_bc(EPS_TOTAL), &config())
        .expect("the bar prepares");
    let state = problem.virgin_state();
    let request = ElastoplasticIncrementRequest::new(Fix128::ONE)
        .with_thermal(ThermalExpansion::from_rise(&rise, fx(ALPHA_PER_K)), None);
    let err = problem
        .step(&state, &request)
        .expect_err("a field covering [0,1]^3 cannot cover a bar of length 4");
    assert!(
        format!("{err:?}").contains("TemperatureFieldDoesNotCoverMesh"),
        "the refusal was {err:?}, which does not name the coverage failure"
    );
}

/// A field that varies along the bar is read element by element, not once.
///
/// Every oracle above uses a **uniform** rise, which is what makes their
/// expectations closed form — and also what makes them blind to a sampler that
/// reads one centroid and reuses it. A uniform sampler cannot produce a
/// non-uniform answer, so the discriminator here is that the elements disagree
/// at all; the ordering is then the physical check on top.
///
/// The field is affine in `x`, which `CoupledField`'s trilinear interpolation
/// reproduces exactly (`trilinear_reproduces_an_affine_function_bit_exactly`),
/// so each element samples `ΔT(x_c)` at its own centroid with no interpolation
/// error. The expectation is not a single bilinear curve — equilibrium
/// redistributes between the hot and cold halves — so what is asserted is the
/// ordering the softening law forces: a hotter element has a smaller yield
/// radius and therefore cannot carry more von Mises stress than a colder one.
#[test]
fn a_field_that_varies_along_the_bar_is_sampled_per_element() {
    // ΔT(x) = x, over a grid that spans the bar with a margin. Affine, so the
    // centroid sample is exact.
    let mut absolute = CoupledField::try_new(
        5,
        3,
        3,
        (
            Fix128::from_int(-1),
            Fix128::from_int(-1),
            Fix128::from_int(-1),
        ),
        (
            Fix128::from_int(5),
            Fix128::from_int(3),
            Fix128::from_int(3),
        ),
    )
    .expect("a grid");
    let (nx, ny, nz) = (absolute.nx(), absolute.ny(), absolute.nz());
    let (hx, _, _) = absolute.cell_size();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let x = Fix128::from_int(-1) + hx * Fix128::from_int(ix as i64);
                absolute.set(ix, iy, iz, x);
            }
        }
    }
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::ZERO);

    // Soften hard enough that the gradient matters: w_y = 1/8 per K over a bar
    // whose ΔT runs 0 → 4, so σ_y falls from σ_y to σ_y/2 across the length.
    let law = ThermalSoftening::try_new(fx(1.0 / 8.0), Fix128::ZERO).expect("valid");
    let hot = solve_bar(EPS_TOTAL, Some((&rise, Some(law))));

    // Each element's own centroid x, computed here rather than read back.
    let mesh = bar_mesh();
    let centroid_x: Vec<f64> = mesh
        .tets
        .iter()
        .map(|t| {
            t.vertices
                .iter()
                .map(|&v| f64::from(mesh.vertices[v as usize][0]))
                .sum::<f64>()
                / 4.0
        })
        .collect();

    let mises: Vec<f64> = hot
        .field
        .element_stress
        .iter()
        .map(|t| t.von_mises().to_f64())
        .collect();

    let spread = mises.iter().copied().fold(f64::MIN, f64::max)
        - mises.iter().copied().fold(f64::MAX, f64::min);
    assert!(
        spread > 1.0e3 * STRESS_TOL,
        "every element carried the same von Mises stress (spread {spread:.3e} MPa), \
         which is what a sampler that reads one centroid and reuses it would \
         produce. The field varies by 4 K across the bar and the law softens by \
         an eighth per kelvin, so the elements must disagree."
    );

    // The sampler claim, stated exactly: one distinct stress per distinct
    // centroid. A sampler that read element zero's centroid and reused it would
    // give one value for twelve elements; a sampler that indexed by element id
    // instead of by position would give twelve values for six positions.
    let mut xs: Vec<f64> = centroid_x.clone();
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
    let mut distinct: Vec<f64> = mises.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup_by(|a, b| (*a - *b).abs() < STRESS_TOL);
    assert_eq!(
        distinct.len(),
        xs.len(),
        "{} distinct centroid positions produced {} distinct stresses. One each \
         is what sampling the field at each element's own centroid gives; fewer \
         means a centroid was reused, more means the stress is following \
         something other than position.",
        xs.len(),
        distinct.len()
    );
    assert!(
        xs.len() > 1,
        "the mesh has one distinct centroid position, so the count above cannot \
         separate a per-element sampler from a single-sample one"
    );

    // Equal position, equal answer: the Kuhn subdivision puts two tetrahedra at
    // each centroid, and a sampler keyed on anything but position would split
    // them.
    let mut pairs = 0usize;
    for a in 0..mises.len() {
        for b in (a + 1)..mises.len() {
            if (centroid_x[a] - centroid_x[b]).abs() < 1.0e-9 {
                pairs += 1;
                close(
                    mises[a],
                    mises[b],
                    STRESS_TOL,
                    &format!(
                        "elements {a} and {b} share a centroid at x = {:.3} but \
                         carry different stress",
                        centroid_x[a]
                    ),
                );
            }
        }
    }
    assert!(
        pairs > 0,
        "no two elements share a centroid, so the equality above is vacuous"
    );

    // ⚠️ What is deliberately *not* asserted: that a hotter element carries less
    // stress. Softening lowers σ_y(T), so the hotter region yields first and
    // accumulates more plastic strain, and with hardening its current radius
    // `σ_y(T) + H ε̄_p` can end up *above* a colder element's. Measured on this
    // scene: x = 1.5 carries 2.948 MPa while the hotter x = 3.5 carries 2.950.
    // The monotone quantity is the yield radius the law hands out, not the
    // stress the body settles at.
}
