//! Oracles for the force on a magnetisable body in the field of permanent magnets
//! (`electromagnetic::{dipole_sample, induced_dipole_force, induced_dipole_energy}`).
//!
//! oracle (the law, closed form): a magnet is a point dipole `m`, whose field is
//! `B = μ0/4π · (3(m·r̂)r̂ − m) / r³`. A soft ferromagnetic sphere of volume `V` in it takes the
//! moment `min(χ_eff·V·|B|/μ0, M_s·V)·B̂`, so its energy is `U = −χ_eff·V·|B|²/(2μ0)` below the
//! saturation field `B_s = μ0·M_s/χ_eff` and `U = −M_s·V·|B| + M_s·V·B_s/2` above it (equal at
//! `B_s`, as are their gradients), and the force is `F = −∇U`. On the axis `|B| = μ0/4π·2|m|/r³`,
//! so the force falls off as `1/r⁷` below saturation (`∇|B|²`) and as `1/r⁴` above it (`∇|B|`);
//! on the equator `|B|` is half the axial value at the same distance, so the linear force is a
//! quarter of the axial one, and both point back toward the magnet. With several magnets the
//! fields add first (`B = ΣBᵢ`) and the response is applied to the sum.
//!
//! oracle (independent): `F = −∇U` checked by a central difference of the energy, with its
//! tolerance derived from the step (truncation `h²/6·|∂³U|` plus rounding `2·δU/(2h)`).

use alice_physics_core::electromagnetic::{
    dipole_sample, induced_dipole_energy, induced_dipole_force, DipoleSample, MagneticResponse,
};
use alice_physics_core::{Fix128, Vec3Fix};

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// Steel: χ_eff = 3 (a sphere of large χ), M_s = 1.7e6 A/m
fn steel() -> MagneticResponse {
    MagneticResponse {
        chi_eff: Fix128::from_int(3),
        saturation_magnetization: Fix128::from_int(1_700_000),
    }
}

/// A 1 cm steel ball: V = 4/3·π·(0.005 m)³ ≈ 5.236e-7 m³
fn ball_volume() -> Fix128 {
    fx(5_236, 10_000_000_000)
}

/// One fixed-point step is 2⁻⁶⁴; a chain of a few hundred operations stays below 2¹⁰ of them
const ULP_CHAIN: f64 = 1024.0 / 18_446_744_073_709_551_616.0;

fn sample_at(moment_z: Fix128, point: Vec3Fix) -> DipoleSample {
    dipole_sample(
        Vec3Fix::ZERO,
        v(Fix128::ZERO, Fix128::ZERO, moment_z),
        point,
    )
}

fn force_at(moment_z: Fix128, point: Vec3Fix) -> Vec3Fix {
    induced_dipole_force(&sample_at(moment_z, point), &steel(), ball_volume())
}

fn rel_close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(a.abs())
}

#[test]
fn below_saturation_the_axial_force_falls_off_as_r_to_the_seventh() {
    // a 1 A·m² magnet: |B| on the axis at 0.1 m is 2e-4 T, far below B_s = μ0·M_s/3 ≈ 0.71 T
    // (and the force, about 1e-6 N, far above the fixed-point resolution)
    let m = Fix128::ONE;
    let r = fx(1, 10);
    let near = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r));
    let far = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r + r));
    assert!(
        near.z < Fix128::ZERO && far.z < Fix128::ZERO,
        "pulled toward the magnet"
    );
    assert_eq!(near.x, Fix128::ZERO);
    assert_eq!(near.y, Fix128::ZERO);
    let ratio = near.z.to_f64() / far.z.to_f64();
    let tol = ULP_CHAIN / far.z.to_f64().abs();
    assert!(
        rel_close(ratio, 128.0, tol.max(1e-9)),
        "F(r)/F(2r) = {ratio}, want 2^7"
    );
}

#[test]
fn above_saturation_the_axial_force_falls_off_as_r_to_the_fourth() {
    // a strong magnet (50 A·m²): on the axis |B| = μ0/4π·2m/r³ stays above B_s ≈ 0.71 T for
    // r < (1e-7·100/0.71)^(1/3) ≈ 0.024 m, so both 0.01 m and 0.02 m are saturated
    let m = Fix128::from_int(50);
    let r = fx(1, 100);
    let near = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r));
    let far = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r + r));
    assert!(near.z < Fix128::ZERO && far.z < Fix128::ZERO);
    let ratio = near.z.to_f64() / far.z.to_f64();
    assert!(
        rel_close(ratio, 16.0, 1e-9),
        "F(r)/F(2r) = {ratio}, want 2^4"
    );
}

#[test]
fn on_the_equator_the_linear_force_is_a_quarter_of_the_axial_one_and_points_inward() {
    let m = Fix128::ONE;
    let r = fx(1, 10);
    let axial = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r));
    let equatorial = force_at(m, v(r, Fix128::ZERO, Fix128::ZERO));
    assert!(equatorial.x < Fix128::ZERO, "pulled toward the magnet");
    assert_eq!(
        (equatorial.y, equatorial.z),
        (Fix128::ZERO, Fix128::ZERO),
        "radial"
    );
    let ratio = axial.z.to_f64() / equatorial.x.to_f64();
    let tol = ULP_CHAIN / equatorial.x.to_f64().abs();
    assert!(
        rel_close(ratio, 4.0, tol.max(1e-12)),
        "axial / equatorial = {ratio}"
    );
}

#[test]
fn saturated_the_axial_force_is_twice_the_equatorial_one_both_radial_and_attractive() {
    // above B_s the force is M_s·V·∇|B|: on the axis |B| = 2k|m|/r³ (∇|B| = 6k|m|/r⁴), on the
    // equator k|m|/r³ (3k|m|/r⁴), so the ratio is 2; an induced dipole seeks the stronger field,
    // so at both points the force points straight back to the magnet
    let m = Fix128::from_int(20);
    let r = fx(1, 100);
    let axial = force_at(m, v(Fix128::ZERO, Fix128::ZERO, r));
    let equatorial = force_at(m, v(r, Fix128::ZERO, Fix128::ZERO));
    assert!(
        axial.z < Fix128::ZERO && equatorial.x < Fix128::ZERO,
        "attractive"
    );
    assert_eq!(
        (axial.x, axial.y),
        (Fix128::ZERO, Fix128::ZERO),
        "radial on the axis"
    );
    assert_eq!(
        (equatorial.y, equatorial.z),
        (Fix128::ZERO, Fix128::ZERO),
        "radial on the equator"
    );
    let ratio = axial.z.to_f64() / equatorial.x.to_f64();
    assert!(
        rel_close(ratio, 2.0, 1e-9),
        "saturated axial / equatorial = {ratio}"
    );
    // and both points are saturated (B above B_s ≈ 0.71 T): the equatorial |B| is 2 T
    let b_eq = sample_at(m, v(r, Fix128::ZERO, Fix128::ZERO))
        .field
        .length()
        .to_f64();
    assert!(
        b_eq > 0.72,
        "the equatorial point must be saturated, |B| = {b_eq}"
    );
}

#[test]
fn a_body_that_does_not_magnetise_feels_exactly_no_force() {
    let rubber = MagneticResponse {
        chi_eff: Fix128::ZERO,
        saturation_magnetization: Fix128::ZERO,
    };
    let s = sample_at(Fix128::from_int(5), v(fx(3, 100), fx(1, 100), fx(7, 100)));
    assert_eq!(
        induced_dipole_force(&s, &rubber, ball_volume()),
        Vec3Fix::ZERO
    );
    assert_eq!(
        induced_dipole_energy(s.field, &rubber, ball_volume()),
        Fix128::ZERO
    );
}

#[test]
fn energy_and_force_are_continuous_across_saturation() {
    // walk outward along the axis through B = B_s; the energy must not jump between neighbours
    let m = Fix128::from_int(2);
    let mut prev: Option<(f64, f64)> = None;
    let mut crossed = false;
    let mut started_above = false;
    let bs = (fx(12_566_370_614, 10_000_000_000_000_000) * Fix128::from_int(1_700_000)
        / Fix128::from_int(3))
    .to_f64();
    for k in 0..400 {
        let r = fx(5, 1000) + fx(k, 20_000);
        let s = sample_at(m, v(Fix128::ZERO, Fix128::ZERO, r));
        let b = s.field.length().to_f64();
        let u = induced_dipole_energy(s.field, &steel(), ball_volume()).to_f64();
        let f = induced_dipole_force(&s, &steel(), ball_volume()).z.to_f64();
        if let Some((pu, pf)) = prev {
            // neighbours dr = 50 µm apart: a continuous energy moves by at most max|F|·dr, and a
            // continuous force falling off at most as r⁻⁷ by at most 7·dr/r of itself (a jump
            // at B_s would exceed both)
            let dr = 5e-5;
            let fmax = pf.abs().max(f.abs());
            let rf = r.to_f64();
            assert!(
                (u - pu).abs() <= fmax * dr * 1.01 + 1e-15,
                "U jumps at r = {rf}"
            );
            assert!(
                (f - pf).abs() <= fmax * 7.0 * dr / (rf - dr) * 1.01,
                "F jumps at r = {rf}"
            );
        }
        if k == 0 {
            started_above = b > bs;
        }
        if b < bs {
            crossed = true;
        }
        prev = Some((u, f));
    }
    assert!(
        started_above && crossed,
        "the walk must start above B_s and pass it"
    );
}

/// −∂U/∂x by a central difference, with its tolerance derived from the step:
/// truncation h²/6·|∂³U| with |∂³U| ≤ 9·8·7·|U|/r³ (U ∝ r⁻⁶ is the steepest case), and the
/// rounding of the two energies divided by 2h
fn check_force_is_minus_grad_energy(magnets: &[(Vec3Fix, Vec3Fix)], p: Vec3Fix) {
    let total = |q: Vec3Fix| -> DipoleSample {
        let mut s: Option<DipoleSample> = None;
        for &(pos, mom) in magnets {
            let d = dipole_sample(pos, mom, q);
            s = Some(match s {
                None => d,
                Some(acc) => acc + d,
            });
        }
        s.expect("at least one magnet")
    };
    let f = induced_dipole_force(&total(p), &steel(), ball_volume());
    let h = fx(1, 100_000);
    let u_at = |q: Vec3Fix| induced_dipole_energy(total(q).field, &steel(), ball_volume()).to_f64();
    let u0 = u_at(p).abs();
    let r_min = magnets
        .iter()
        .map(|&(pos, _)| (p - pos).length().to_f64())
        .fold(f64::INFINITY, f64::min);
    let hf = h.to_f64();
    let trunc = hf * hf / 6.0 * 504.0 * u0 / (r_min * r_min * r_min);
    let round = 2.0 * (ULP_CHAIN + u0 * 1e-15) / (2.0 * hf);
    let tol = trunc + round;
    for (axis, got) in [(0, f.x), (1, f.y), (2, f.z)] {
        let e = |sign: i64| {
            let d = h * Fix128::from_int(sign);
            match axis {
                0 => v(p.x + d, p.y, p.z),
                1 => v(p.x, p.y + d, p.z),
                _ => v(p.x, p.y, p.z + d),
            }
        };
        let fd = -(u_at(e(1)) - u_at(e(-1))) / (2.0 * hf);
        assert!(
            (got.to_f64() - fd).abs() <= tol,
            "axis {axis}: F = {}, −∂U = {fd}, tol {tol}",
            got.to_f64()
        );
    }
}

#[test]
fn the_force_is_minus_the_gradient_of_the_energy_off_axis() {
    let one = [(Vec3Fix::ZERO, v(fx(1, 3), fx(2, 3), Fix128::ONE))];
    check_force_is_minus_grad_energy(&one, v(fx(7, 100), fx(-4, 100), fx(9, 100)));
    // saturated: a strong magnet close by
    let strong = [(
        Vec3Fix::ZERO,
        v(Fix128::ZERO, Fix128::from_int(20), Fix128::from_int(5)),
    )];
    check_force_is_minus_grad_energy(&strong, v(fx(1, 100), fx(12, 1000), fx(-6, 1000)));
}

#[test]
fn two_magnets_add_their_fields_before_the_response() {
    let a = (
        v(fx(-1, 10), Fix128::ZERO, Fix128::ZERO),
        v(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    );
    let b = (
        v(fx(1, 10), fx(2, 100), Fix128::ZERO),
        v(Fix128::ZERO, Fix128::from_int(2), Fix128::ONE),
    );
    let p = v(fx(1, 100), fx(3, 100), fx(2, 100));
    check_force_is_minus_grad_energy(&[a, b], p);
    // and the response is not additive: the force of the summed field is not the sum of forces
    let sa = dipole_sample(a.0, a.1, p);
    let sb = dipole_sample(b.0, b.1, p);
    let together = induced_dipole_force(&(sa + sb), &steel(), ball_volume());
    let apart = induced_dipole_force(&sa, &steel(), ball_volume())
        + induced_dipole_force(&sb, &steel(), ball_volume());
    assert_ne!(together, apart);
}

#[test]
fn opposed_magnets_cancel_at_the_midpoint() {
    // on the axis each dipole's field is μ0/4π·2m/r³ along its own moment, so two equal and
    // opposite moments at the same distance on either side cancel at the midpoint: exactly in
    // the law, to the rounding of the mirrored terms in fixed point (a product of a negative
    // number truncates on the other side of zero, one unit apart at most per term)
    let d = fx(1, 10);
    let m = Fix128::ONE;
    let a = (
        v(Fix128::ZERO, Fix128::ZERO, -d),
        v(Fix128::ZERO, Fix128::ZERO, m),
    );
    let b = (
        v(Fix128::ZERO, Fix128::ZERO, d),
        v(Fix128::ZERO, Fix128::ZERO, -m),
    );
    let sa = dipole_sample(a.0, a.1, Vec3Fix::ZERO);
    let s = sa + dipole_sample(b.0, b.1, Vec3Fix::ZERO);
    // the residual field: the fixed-point rounding of the two fields
    let eps_b = ULP_CHAIN;
    assert!(
        s.field.length().to_f64() <= eps_b,
        "the fields cancel to rounding: {} T left of {} T",
        s.field.length().to_f64(),
        sa.field.length().to_f64()
    );
    // the residual force, propagated from the residual field: F = (χ_eff·V/μ0)·JᵀB is linear in
    // B, so |F| ≤ (χ_eff·V/μ0)·‖J‖_F·ε_B, plus the rounding of the product itself
    let frob = |j: &[Vec3Fix; 3]| -> f64 {
        j.iter()
            .map(|c| {
                let (x, y, z) = (c.x.to_f64(), c.y.to_f64(), c.z.to_f64());
                x * x + y * y + z * z
            })
            .sum::<f64>()
            .sqrt()
    };
    let mu0 = 12_566_370_614.0 / 1e22;
    let c = 3.0 * ball_volume().to_f64() / mu0;
    let bound = c * frob(&s.jacobian) * eps_b + ULP_CHAIN;
    let together = induced_dipole_force(&s, &steel(), ball_volume())
        .length()
        .to_f64();
    assert!(
        together <= bound,
        "the pull cancels: {together} N left, bound {bound} N"
    );
    // each alone pulls the ball toward itself
    let fa = induced_dipole_force(&sa, &steel(), ball_volume());
    assert!(fa.z < Fix128::ZERO);

    // at the midpoint a force summed per magnet cancels too (by symmetry), so the per-magnet sum is
    // told apart a little off it: there the summed field gives a force that differs from the
    // per-magnet sum by far more than the bound (the cross terms Jaᵀ·Bb + Jbᵀ·Ba)
    let off = v(Fix128::ZERO, Fix128::ZERO, fx(1, 100));
    let (pa, pb) = (dipole_sample(a.0, a.1, off), dipole_sample(b.0, b.1, off));
    let summed = induced_dipole_force(&(pa + pb), &steel(), ball_volume());
    let per_magnet = induced_dipole_force(&pa, &steel(), ball_volume())
        + induced_dipole_force(&pb, &steel(), ball_volume());
    let gap = (summed - per_magnet).length().to_f64();
    let bound_off = c * frob(&(pa + pb).jacobian) * eps_b + ULP_CHAIN;
    assert!(
        gap > 1e3 * bound_off,
        "the per-magnet sum must miss by orders of magnitude: gap {gap} N, bound {bound_off} N"
    );
}
