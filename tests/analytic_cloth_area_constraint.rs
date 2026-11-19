//! Oracles for `Cloth::set_area_constraint` (every triangle keeps its rest area).
//!
//! Law: `C = A − A₀` per triangle, `A = |(x₁ − x₀) × (x₂ − x₀)| / 2`, solved by XPBD
//! inside the cloth's iterations with the stretch and bending constraints.
//!
//! Closed forms used here:
//! - a sheet whose triangles keep their area keeps its total area `A₀`; a strip of
//!   width `W₀` pulled from `L₀` to `L` has mean width `A₀ / L`, and away from its
//!   clamped ends the width `W₀·L₀/L` (in-plane Poisson ratio 1: the 2-D area is the
//!   incompressible quantity);
//! - the constraint is workless and internal: its gradients sum to zero, so with equal
//!   masses it moves no linear momentum, and in a released tethered sheet it cannot add
//!   energy (tether + kinetic never exceeds its start and decays to rest);
//! - a small mode tangent to the constraint (the area-preserving stretch
//!   `x → (1 + a) x`, `z → z / (1 + a)`) oscillates with the tether's own period and
//!   decay: every particle has the same `ω`, and the constraint only acts at second
//!   order in `a`.

use alice_physics_core::det_math::exp64;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::Cloth;

/// A flat sheet in the `xz` plane, `w × h` m with `nx × nz` particles of `mass` kg, no
/// gravity, no frame damping, soft edges and bending (`stretch`, `bend` compliances).
fn sheet(w: f64, h: f64, nx: usize, nz: usize, mass: f64, stretch: f64) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::new(
            Fix128::from_f64(-w / 2.0),
            Fix128::ZERO,
            Fix128::from_f64(-h / 2.0),
        ),
        Fix128::from_f64(w),
        Fix128::from_f64(h),
        nx,
        nz,
        Fix128::from_f64(mass),
    );
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    c.config.stretch_compliance = Fix128::from_f64(stretch);
    c.config.bend_compliance = Fix128::from_f64(1.0);
    c.config.self_collision = false;
    c
}

fn x(c: &Cloth, i: usize) -> f64 {
    c.positions[i].x.to_f64()
}

fn z(c: &Cloth, i: usize) -> f64 {
    c.positions[i].z.to_f64()
}

/// Pull a strip `aspect` m long and 1 m wide to 1.25 times its length by its two clamped
/// end columns (moved over 60 frames, then held 600 frames with damping), edges of
/// compliance `stretch`, and return `(width at the middle column, total area, rest area,
/// bound)` where `bound` is the area error the competing edges can leave (below).
fn pulled_strip(aspect: usize, stretch: f64, area_on: bool) -> (f64, f64, f64, f64) {
    let per_m = 8;
    let nx = aspect * per_m + 1;
    let nz = per_m + 1;
    let l0 = aspect as f64;
    let mass = 0.01;
    let mut c = sheet(l0, 1.0, nx, nz, mass, stretch);
    c.config.iterations = 20;
    c.config.damping = Fix128::from_f64(0.9);
    if area_on {
        c.set_area_constraint(Fix128::ZERO);
    }
    let a0 = c.total_area().to_f64();
    let ends: Vec<(usize, f64)> = (0..nz)
        .flat_map(|j| [(j * nx, -1.0), (j * nx + nx - 1, 1.0)])
        .collect();
    for &(i, _) in &ends {
        c.pin(i);
    }
    let lambda = 1.25;
    for f in 1..=660 {
        let s = 1.0 + (lambda - 1.0) * (f.min(60) as f64 / 60.0);
        for &(i, side) in &ends {
            c.positions[i].x = Fix128::from_f64(side * l0 / 2.0 * s);
        }
        c.step(Fix128::from_ratio(1, 60));
    }
    // Gauss-Seidel with a rigid area constraint and compliant edges settles where each
    // edge pass undoes at most the fraction `h² (w_i + w_j) / α` of its strain `λ − 1`
    // before the area pass restores the triangles (`h` the substep).
    let h = 1.0 / (60.0 * c.config.substeps as f64);
    let bound = (lambda - 1.0) * h * h * (2.0 / mass) / stretch;
    let mid = nx / 2;
    let width = z(&c, (nz - 1) * nx + mid) - z(&c, mid);
    (width, c.total_area().to_f64(), a0, bound)
}

#[test]
fn a_pulled_strip_keeps_its_area_and_narrows_toward_the_incompressible_width() {
    let closed = 1.0 / 1.25;
    let mut errs = Vec::new();
    for aspect in [4, 8] {
        let (w, a, a0, bound) = pulled_strip(aspect, 1.0, true);
        let rel_area = (a - a0).abs() / a0;
        let err = (closed - w) / closed;
        eprintln!(
            "aspect {aspect}: mid width {w:.6} vs W0/λ {closed:.6} (rel {err:.2e}); area rel {rel_area:.2e} (bound {bound:.2e})"
        );
        assert!(
            rel_area <= bound,
            "area kept within the edges' share (aspect {aspect}): {rel_area:e} > {bound:e}"
        );
        // the clamped ends hold the width at W0, so with the area kept the middle is
        // narrower than the mean A/L = W0/λ
        assert!(err > 0.0, "middle below W0/λ (aspect {aspect}): {w}");
        errs.push(err);
    }
    // the ends' extra area is spread over the length: the deficit falls like 1/L
    assert!(
        errs[1] < 0.6 * errs[0],
        "width deficit falls with length: {errs:?}"
    );
}

#[test]
fn without_the_area_constraint_the_pulled_strip_does_not_keep_its_area() {
    let (w, a, a0, bound) = pulled_strip(4, 1.0, false);
    let rel_area = (a - a0).abs() / a0;
    eprintln!("off: mid width {w:.6}, area rel {rel_area:.2e}");
    assert!(
        rel_area > 10.0 * bound,
        "off, the edges alone let the area change: {rel_area:e}"
    );
}

/// Tether + kinetic energy of the sheet about its rest shape.
fn energy(c: &Cloth, rest: &[Vec3Fix], k: f64, m: f64) -> f64 {
    let mut e = 0.0;
    for i in 0..c.positions.len() {
        let d = c.positions[i] - rest[i];
        let v = c.velocities[i];
        e += 0.5 * k * (d.dot(d)).to_f64() + 0.5 * m * (v.dot(v)).to_f64();
    }
    e
}

/// A tethered sheet started on an area-preserving stretch `(1 + a, 1 / (1 + a))` about
/// its centre, released at rest.
fn released(a: f64, omega: f64, zeta: f64) -> (Cloth, Vec<Vec3Fix>, f64, f64) {
    released_with(a, omega, zeta, true, 1e9)
}

/// `stretch` is the edges' compliance: at `1e9` their pull per pass is `h² w / α ≈ 2e-12`
/// and the tether and the area constraint are the only forces; at `1e3` (a cloth's own
/// soft edges) the edges stiffen with the iterations and shorten the period by itself.
fn released_with(
    a: f64,
    omega: f64,
    zeta: f64,
    area: bool,
    stretch: f64,
) -> (Cloth, Vec<Vec3Fix>, f64, f64) {
    let m = 0.01;
    let mut c = sheet(1.0, 1.0, 9, 9, m, stretch);
    let rest = c.positions.clone();
    let k = m * omega * omega;
    let damp = 2.0 * zeta * m * omega;
    c.set_rest_tether(Fix128::from_f64(k), Fix128::from_f64(damp));
    if area {
        c.set_area_constraint(Fix128::ZERO);
    }
    for p in &mut c.positions {
        p.x = Fix128::from_f64(p.x.to_f64() * (1.0 + a));
        p.z = Fix128::from_f64(p.z.to_f64() / (1.0 + a));
    }
    c.prev_positions = c.positions.clone();
    (c, rest, k, m)
}

/// Mean period (corner zero crossings, two apart) and amplitude ratio over one period.
fn period_and_decay(area: bool, stretch: f64, omega: f64, zeta: f64) -> (f64, f64, usize) {
    let (mut c, rest, _, _) = released_with(0.01, omega, zeta, area, stretch);
    let corner = 0; // (-0.5·(1+a), -0.5/(1+a))
    let x0 = rest[corner].x.to_f64();
    let dt = 1.0 / 240.0;
    let mut prev = x(&c, corner) - x0;
    let mut crossings = Vec::new();
    let mut peaks: Vec<f64> = Vec::new();
    let mut prev_prev = prev;
    for f in 1..=960 {
        c.step(Fix128::from_ratio(1, 240));
        let e = x(&c, corner) - x0;
        if (prev < 0.0) != (e < 0.0) {
            let t = (f as f64 - 1.0 + prev / (prev - e)) * dt;
            crossings.push(t);
        }
        if prev.abs() > prev_prev.abs() && prev.abs() >= e.abs() {
            peaks.push(prev.abs());
        }
        prev_prev = prev;
        prev = e;
    }
    let periods: Vec<f64> = crossings.windows(3).map(|w| w[2] - w[0]).collect();
    let mean = periods.iter().sum::<f64>() / periods.len() as f64;
    (mean, peaks[2] / peaks[0], periods.len())
}

#[test]
fn a_small_area_preserving_mode_oscillates_with_the_tethers_period_and_decay() {
    let (omega, zeta) = (2.0 * core::f64::consts::PI, 0.1);
    let (t_on, d_on, n) = period_and_decay(true, 1e9, omega, zeta);
    let (t_off, d_off, _) = period_and_decay(false, 1e9, omega, zeta);
    let (t_on_soft, d_on_soft, _) = period_and_decay(true, 1e3, omega, zeta);
    let (t_off_soft, d_off_soft, _) = period_and_decay(false, 1e3, omega, zeta);
    let wd = omega * (1.0 - zeta * zeta).sqrt();
    let t_closed = 2.0 * core::f64::consts::PI / wd;
    let decay_closed = exp64(-zeta * omega * t_closed);
    eprintln!(
        "period {t_on:.6} (off {t_off:.6}) vs {t_closed:.6} ({n} periods); decay per period {d_on:.6} (off {d_off:.6}) vs {decay_closed:.6}"
    );
    eprintln!(
        "soft edges: period {t_on_soft:.6} (off {t_off_soft:.6}); decay {d_on_soft:.6} (off {d_off_soft:.6})"
    );
    assert!(n >= 4);
    // The mode is tangent to the constraint: the area it breaks is O(a²) of the triangle,
    // so the correction is O(a²·x) against an amplitude O(a·x) — relative O(a) = 1e-2 on
    // the amplitude (and so on the decay ratio), while the period, a phase, moves only at
    // O(a²) = 1e-4. With the cloth's own edges the same holds.
    let a = 0.01;
    assert!(
        (t_on_soft - t_off_soft).abs() / t_off_soft < 1e-4,
        "soft edges: same period as without"
    );
    assert!(
        (d_on_soft - d_off_soft).abs() / d_off_soft < a,
        "soft edges: same decay as without"
    );
    assert!(
        (t_on - t_off).abs() / t_off < 1e-4,
        "same period as without"
    );
    assert!((d_on - d_off).abs() / d_off < a, "same decay as without");
    assert!((t_on - t_closed).abs() / t_closed < 1e-3, "period");
    assert!((d_on - decay_closed).abs() / decay_closed < 1e-2, "decay");
}

#[test]
fn a_released_tethered_sheet_gains_no_energy_and_settles() {
    let (mut c, rest, k, m) = released(0.2, 2.0 * core::f64::consts::PI, 0.1);
    // 1200 steps of 1/240 s = 5 s
    let mut e_prev = energy(&c, &rest, k, m);
    let e0 = e_prev;
    let mut worst_rise = 0.0_f64;
    for _ in 0..1200 {
        c.step(Fix128::from_ratio(1, 240));
        let e = energy(&c, &rest, k, m);
        worst_rise = worst_rise.max(e - e_prev);
        assert!(e <= e0 * (1.0 + 1e-9), "never above the start: {e} > {e0}");
        e_prev = e;
    }
    // Each particle is a tethered damped oscillator; `V = E + ζω m x·v` obeys
    // `dV/dt = −2ζω V` exactly and `(1 − ζ) E ≤ V ≤ (1 + ζ) E`, so a start at rest
    // (`V₀ = E₀`) bounds `E(t)` to `E₀ e^{−2ζωt} / (1 ± ζ)`.
    let (omega, zeta) = (2.0 * core::f64::consts::PI, 0.1);
    let decayed = e0 * exp64(-2.0 * zeta * omega * 5.0);
    eprintln!(
        "start {e0:.6e}, end {e_prev:.6e} (envelope {decayed:.6e}), worst single-step rise {worst_rise:.3e}"
    );
    assert!(
        e_prev <= decayed / (1.0 - zeta) && e_prev >= decayed / (1.0 + zeta),
        "settles on the damped envelope: {e_prev} vs {decayed}"
    );
}

#[test]
fn the_area_constraint_moves_no_linear_momentum() {
    let m = 0.01;
    let mut c = sheet(1.0, 1.0, 9, 9, m, 1e3);
    c.set_area_constraint(Fix128::ZERO);
    // a non-uniform velocity field that changes the areas
    for (i, v) in c.velocities.iter_mut().enumerate() {
        let s = Fix128::from_ratio(((i * 37) % 11) as i64 - 5, 10);
        *v = Vec3Fix::new(s, Fix128::ZERO, -s.half());
    }
    let total = |c: &Cloth| {
        c.velocities
            .iter()
            .fold(Vec3Fix::ZERO, |sum, v| sum + *v * Fix128::from_f64(m))
    };
    let p0 = total(&c);
    let mut worst = 0.0_f64;
    for _ in 0..120 {
        c.step(Fix128::from_ratio(1, 60));
        let d = total(&c) - p0;
        worst = worst.max(d.length().to_f64());
    }
    eprintln!("worst |ΔP| {worst:.3e}");
    assert!(worst < 1e-12, "momentum kept to rounding: {worst:e}");
}

#[test]
fn off_is_the_cloth_without_the_constraint_bit_for_bit() {
    let run = |toggle: bool| {
        let mut c = sheet(1.0, 1.0, 7, 7, 0.01, 1e-4);
        c.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
        c.pin(0);
        c.pin(6);
        if toggle {
            c.set_area_constraint(Fix128::ZERO);
            c.clear_area_constraint();
        }
        for _ in 0..60 {
            c.step(Fix128::from_ratio(1, 60));
        }
        c.state_digest()
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn the_area_constraint_is_part_of_the_state() {
    let mut a = sheet(1.0, 1.0, 5, 5, 0.01, 1e-4);
    let off = a.state_digest();
    a.set_area_constraint(Fix128::ZERO);
    let zero = a.state_digest();
    a.set_area_constraint(Fix128::from_f64(1e-3));
    let soft = a.state_digest();
    assert_ne!(off, zero);
    assert_ne!(zero, soft);
    assert_eq!(a.area_constraint(), Some(Fix128::from_f64(1e-3)));
}
