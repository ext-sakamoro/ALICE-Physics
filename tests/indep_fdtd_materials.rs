//! Independent oracles for the material path of `maxwell_fdtd`
//! (`ε_r`, `μ_r`, `σ` per cell).
//!
//! The existing material oracles (`analytic_fdtd_materials.rs`) use: a vacuum
//! map hashed against pre-change goldens, a uniform cavity checked against
//! the projected three-term recurrence, the averaging rules sample by sample,
//! and a `272 × 32` waveguide driven around `ω = 0.6` and evaluated at
//! `ω ∈ {0.5, 0.6, 0.7}` with media `ε = 4`, `μ = 4`, `ε = μ = 2` and
//! `(ε, σ) = (2, 1/2)`, the reflection separated by subtracting a reference
//! run, and the Courant bound at `S = 9/16`.
//!
//! None of those inputs or derivations is reused here:
//!
//! - **phase velocity**: oscillation frequency of a TM cavity mode measured from
//!   zero crossings, in a medium with `ε = 9/8`, `μ = 2` (`n = 3/2`), in all
//!   three orientations, against the discrete dispersion relation derived in
//!   this file and against `c/n`;
//! - **reflection**: from a dense medium (`ε = 9/4`) into vacuum (`r > 0`), at
//!   `ω = 0.45`, separated by a two-probe forward/backward decomposition in
//!   the frequency domain, against a reflection coefficient derived in this
//!   file from the node equation at the interface, and against Fresnel;
//! - **conductive loss**: per-cell attenuation in `ε = 3/2`, `σ = 1/8` at
//!   `ω = 0.45` from the probe ratio, against the discrete closed form derived
//!   here and the continuum `Im k`, `k² = εω² − iωσ − k_y²`;
//! - **energy**: the discrete leap-frog energy
//!   `W = Σ ε E^n·E^{n+1} + Σ μ |H^{n+½}|²` (weights computed here from the
//!   documented averaging rules) is conserved in a lossless inhomogeneous
//!   lattice and loses exactly `(S/2) Σ σ E^n·(E^{n+1} + 2E^n + E^{n−1})` per
//!   step with conductivity;
//! - **Courant**: `S = 3/4` (above the vacuum limit) accepted and stable in a
//!   slow medium, refused with one vacuum cell, and unstable without a map.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    Component, Material, MaterialError, MaterialMap, YeeGrid, COURANT_3D,
};
use std::f64::consts::PI;

const S: f64 = 0.5625;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

// ----------------------------------------------------------------------------
// small complex arithmetic
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    fn exp_i(phi: f64) -> Self {
        Self::new(phi.cos(), phi.sin())
    }
    fn sqrt(self) -> Self {
        let r = self.abs().sqrt();
        let h = self.arg() / 2.0;
        Self::new(r * h.cos(), r * h.sin())
    }
    fn powi(self, n: i32) -> Self {
        let mut out = Self::new(1.0, 0.0);
        let base = if n >= 0 {
            self
        } else {
            Self::new(1.0, 0.0).div(self)
        };
        for _ in 0..n.unsigned_abs() {
            out = out.mul(base);
        }
        out
    }
}

// ----------------------------------------------------------------------------
// discrete closed forms (derived here, see each comment)
// ----------------------------------------------------------------------------

/// Plane-wave time dependence `z^n`, `z = e^{iωS}`. Eliminating `H` from
/// `μ(H^{n+½} − H^{n−½}) = −S ∇×E^n` and
/// `E^{n+1} = C_a E^n + C_b ∇×H^{n+½}` gives, for a mode with discrete
/// curl-curl eigenvalue `Λ = Λ_x + Λ_y`,
/// `Λ = −(μ / (C_b S)) · 2i sin(ωS/2) · (z^{½} − C_a z^{−½})`.
/// `Λ_x = 2 − p − 1/p` for a spatial factor `p` per cell along x.
/// Returns the forward root `p` (|p| < 1, or `p = e^{−ik}`, `k > 0`).
fn forward_root(eps: f64, mu: f64, sigma: f64, w: f64, lambda_y: f64) -> C {
    let a = sigma * S / (2.0 * eps);
    let ca = (1.0 - a) / (1.0 + a);
    let cb = (S / eps) / (1.0 + a);
    let half = C::exp_i(w * S / 2.0);
    let half_inv = C::exp_i(-w * S / 2.0);
    let two_i_s = C::new(0.0, 2.0 * (w * S / 2.0).sin());
    let lam = two_i_s
        .mul(half.sub(half_inv.scale(ca)))
        .scale(-mu / (cb * S));
    let lam_x = lam.sub(C::new(lambda_y, 0.0));
    // p + 1/p = c, c = 2 − Λ_x.
    let c = C::new(2.0, 0.0).sub(lam_x);
    let disc = c.mul(c).sub(C::new(4.0, 0.0)).sqrt();
    let p1 = c.add(disc).scale(0.5);
    let p2 = c.sub(disc).scale(0.5);
    let (m1, m2) = (p1.abs(), p2.abs());
    if (m1 - 1.0).abs() < 1e-12 && (m2 - 1.0).abs() < 1e-12 {
        // Lossless propagating: e^{−ik} with k > 0 has negative imaginary part.
        if p1.im < 0.0 {
            p1
        } else {
            p2
        }
    } else if m1 < m2 {
        p1
    } else {
        p2
    }
}

/// Discrete reflection at node `I0` of a lossless `μ = 1` chain whose cells
/// left of `I0` have `ε₁` and right of it `ε₂`. The node `I0` itself takes the
/// arithmetic mean `ε_J = (ε₁ + ε₂)/2` of the four cells around it (doc
/// rule). With `τ² = 4 sin²(ωS/2)/S²` the node equation is
/// `E_{i+1} + E_{i−1} = (2 + Λ_y − ε_i τ²) E_i`. Writing the field left as
/// `A p₁^{i−I0} + B p₁^{−(i−I0)}` and right as `(A + B) p₂^{i−I0}`, the equation
/// at `I0` gives `r = B/A = −(p₂ − Q + 1/p₁)/(p₂ − Q + p₁)`,
/// `Q = 2 + Λ_y − ε_J τ²`.
fn r_interface(eps1: f64, eps2: f64, w: f64, lambda_y: f64) -> C {
    let p1 = forward_root(eps1, 1.0, 0.0, w, lambda_y);
    let p2 = forward_root(eps2, 1.0, 0.0, w, lambda_y);
    let tau2 = 4.0 * (w * S / 2.0).sin().powi(2) / (S * S);
    let qv = C::new(2.0 + lambda_y - 0.5 * (eps1 + eps2) * tau2, 0.0);
    let num = p2.sub(qv).add(C::new(1.0, 0.0).div(p1));
    let den = p2.sub(qv).add(p1);
    C::new(0.0, 0.0).sub(num.div(den))
}

// ----------------------------------------------------------------------------
// quasi-2-D waveguide driver (Ez, lowest transverse mode)
// ----------------------------------------------------------------------------

const NY: usize = 32;
const IS: usize = 10;
const TAU: f64 = 16.0;
const W0: f64 = 0.5;
/// Evaluation frequency (not one used by the existing oracles).
const WE: f64 = 0.45;

fn lambda_y() -> f64 {
    4.0 * (PI / (2.0 * NY as f64)).sin().powi(2)
}

fn pulse(n: usize) -> f64 {
    let t = n as f64 * S - 4.0 * TAU;
    (-(t / TAU) * (t / TAU)).exp() * (W0 * t).sin()
}

/// Ez(i, NY/2) at each probe after every step.
fn waveguide(map: &MaterialMap, nx: usize, steps: usize, probes: &[usize]) -> Vec<Vec<f64>> {
    let mut g = YeeGrid::new(nx, NY, 1, COURANT_3D)
        .with_materials(map)
        .expect("valid map");
    let prof: Vec<f64> = (0..=NY)
        .map(|j| (PI * j as f64 / NY as f64).sin())
        .collect();
    let mut out = vec![Vec::with_capacity(steps); probes.len()];
    for n in 0..steps {
        let a = pulse(n);
        for j in 1..NY {
            g.set_current(Component::Ez, IS, j, 0, Fix128::from_f64(a * prof[j]));
        }
        g.step();
        for (p, &i) in probes.iter().enumerate() {
            out[p].push(g.get(Component::Ez, i, NY / 2, 0).to_f64());
        }
    }
    out
}

/// `Σ x_n e^{−iωS n}`: the `e^{+iωt}` component.
fn dft(x: &[f64], w: f64) -> C {
    x.iter().enumerate().fold(C::new(0.0, 0.0), |acc, (n, &v)| {
        acc.add(C::exp_i(-w * S * n as f64).scale(v))
    })
}

fn tail_ratio(x: &[f64]) -> f64 {
    let peak = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tail = x[x.len() - 20..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    tail / peak
}

/// Reflection from a dense dielectric (`ε₁ = 9/4`, source side) into vacuum.
///
/// Two probes `P1 = 130`, `P2 = 137` in medium 1 (interface node `I0 = 200`,
/// lattice `340 × 32`) see
/// `E(P) = F p₁^{P−I0} + R p₁^{−(P−I0)}` at each frequency; solving the 2×2
/// system gives `r = R/F` directly at the interface node without a
/// reference run. Geometry (group velocities ≈ 0.66 / 0.98): the run ends
/// after both pulses have left the probes and before the echo from the far
/// PEC wall or the `x = 0` wall can return to them (checked: the tail at the
/// probes is below `2e-5` of the peak; measured envelope floor ≈ 6e-6 between
/// steps 1120 and 1240, the first echo arrives after step 1240).
#[test]
fn dense_to_vacuum_reflection_matches_interface_node_equation() {
    let (nx, i0, p1, p2, steps) = (340usize, 200usize, 130usize, 137usize, 1240usize);
    let eps1 = 2.25;
    let mut map = MaterialMap::vacuum(nx, NY, 1);
    map.fill([0, 0, 0], [i0, NY, 1], Material::dielectric(q(9, 4)));
    let rec = waveguide(&map, nx, steps, &[p1, p2]);
    for r in &rec {
        assert!(
            tail_ratio(r) < 2e-5,
            "probe not quiet at the end: {}",
            tail_ratio(r)
        );
    }
    let ly = lambda_y();
    let pf = forward_root(eps1, 1.0, 0.0, WE, ly);
    assert!((pf.abs() - 1.0).abs() < 1e-12, "medium 1 propagates");
    let e1 = dft(&rec[0], WE);
    let e2 = dft(&rec[1], WE);
    let (d1, d2) = (p1 as i32 - i0 as i32, p2 as i32 - i0 as i32);
    let (a1, b1) = (pf.powi(d1), pf.powi(-d1));
    let (a2, b2) = (pf.powi(d2), pf.powi(-d2));
    // [a1 b1; a2 b2] [F; R] = [e1; e2]
    let det = a1.mul(b2).sub(b1.mul(a2));
    let f = e1.mul(b2).sub(b1.mul(e2)).div(det);
    let rr = a1.mul(e2).sub(e1.mul(a2)).div(det);
    let measured = rr.div(f);

    let predicted = r_interface(eps1, 1.0, WE, ly);
    let err = measured.sub(predicted).abs();
    assert!(
        err < 2e-5,
        "r measured {measured:?} vs discrete {predicted:?} (|Δ| = {err:e})"
    );

    // Continuum, same waveguide mode (s-polarised, μ equal):
    // r = (k₁ − k₂)/(k₁ + k₂), k = √(εω² − k_y²), k_y = π/NY.
    let ky2 = (PI / NY as f64).powi(2);
    let k1 = (eps1 * WE * WE - ky2).sqrt();
    let k2 = (WE * WE - ky2).sqrt();
    let r_cont = (k1 - k2) / (k1 + k2);
    // Medium 1 is resolved by ≈ 9.4 cells per wavelength, so the lattice
    // shifts r by several percent (measured |r| = 0.1902 vs 0.2066). That gap
    // must be the discretisation error of the closed form, i.e. shrink by
    // ≈ 4 when ω and k_y are both halved (continuum r unchanged).
    assert!(measured.re > 0.0, "dense → rare reflects with r > 0");
    let gap = |w: f64, ly: f64| (r_interface(eps1, 1.0, w, ly).abs() - r_cont).abs();
    let ly_fine = 4.0 * (PI / (4.0 * NY as f64)).sin().powi(2);
    let (g1, g2) = (gap(WE, ly), gap(WE / 2.0, ly_fine));
    assert!(
        (measured.abs() - r_cont).abs() < g1 + 1e-4,
        "|r| {} vs continuum {r_cont}, closed-form gap {g1}",
        measured.abs()
    );
    assert!(
        (3.5..4.5).contains(&(g1 / g2)),
        "lattice gap order: {g1:e} / {g2:e} = {}",
        g1 / g2
    );
    // And the plane-wave limit (n₁ − n₂)/(n₁ + n₂) = 1/5 is close by.
    assert!((r_cont - 0.2).abs() < 0.01);
}

/// Attenuation in a conductive dielectric (`ε = 3/2`, `σ = 1/8`): with no
/// interface the probes see a forward wave only, so
/// `E(P2)/E(P1) = p^{P2−P1}` at each frequency. The measured complex ratio
/// against `p^{12}` from [`forward_root`]; the attenuation constant `−ln|p|` also against the
/// continuum `−Im k`, `k² = εω² − iωσ − k_y²` (`e^{iωt}` convention).
#[test]
fn conductive_medium_attenuates_by_the_discrete_closed_form() {
    let (nx, p1, p2, steps) = (170usize, 40usize, 52usize, 700usize);
    let (eps, sigma) = (1.5, 0.125);
    let mut map = MaterialMap::vacuum(nx, NY, 1);
    map.fill(
        [0, 0, 0],
        [nx, NY, 1],
        Material::new(q(3, 2), Fix128::ONE, q(1, 8)),
    );
    let rec = waveguide(&map, nx, steps, &[p1, p2]);
    for r in &rec {
        assert!(tail_ratio(r) < 1e-6, "probe not quiet: {}", tail_ratio(r));
    }
    let ratio = dft(&rec[1], WE).div(dft(&rec[0], WE));
    let d = (p2 - p1) as f64;
    let alpha_measured = -ratio.abs().ln() / d;

    let p = forward_root(eps, 1.0, sigma, WE, lambda_y());
    let alpha_disc = -p.abs().ln();
    assert!(
        (alpha_measured - alpha_disc).abs() < 1e-5 * alpha_disc.max(1e-3) + 1e-7,
        "α measured {alpha_measured} vs discrete {alpha_disc}"
    );
    // Amplitude and phase together: the complex ratio against `p^{12}`.
    let want = p.powi((p2 - p1) as i32);
    assert!(
        ratio.sub(want).abs() < 1e-5 * want.abs(),
        "E(P2)/E(P1) = {ratio:?} vs p^12 = {want:?}"
    );

    let ky2 = (PI / NY as f64).powi(2);
    let k = C::new(eps * WE * WE - ky2, -WE * sigma).sqrt();
    let alpha_cont = -k.im;
    assert!(alpha_cont > 0.0);
    // The lattice shifts α by ≈ 3 % at ≈ 11 cells per wavelength; that shift
    // must be the closed form's second-order discretisation error: halving
    // ω, k_y and σ halves the continuum α and should shrink the gap by ≈ 4.
    let ly_fine = 4.0 * (PI / (4.0 * NY as f64)).sin().powi(2);
    let alpha_fine = -forward_root(eps, 1.0, sigma / 2.0, WE / 2.0, ly_fine)
        .abs()
        .ln()
        * 2.0;
    let (g1, g2) = (
        (alpha_disc - alpha_cont).abs(),
        (alpha_fine - alpha_cont).abs(),
    );
    assert!(
        (alpha_measured - alpha_cont).abs() < g1 + 1e-6,
        "α measured {alpha_measured} vs continuum {alpha_cont}"
    );
    assert!(g1 < 0.05 * alpha_cont);
    assert!(
        (3.5..4.5).contains(&(g1 / g2)),
        "lattice gap order: {g1:e} / {g2:e} = {}",
        g1 / g2
    );
    // The loss is real: the same medium without σ does not attenuate.
    let p0 = forward_root(eps, 1.0, 0.0, WE, lambda_y());
    assert!((p0.abs() - 1.0).abs() < 1e-12);
    assert!(alpha_disc > 0.04);
}

// ----------------------------------------------------------------------------
// phase velocity from a cavity mode
// ----------------------------------------------------------------------------

/// Orientation: the field component and how (u, v) map onto lattice indices.
#[derive(Clone, Copy)]
enum Orient {
    /// Ez on (x, y).
    Z,
    /// Ex on (y, z).
    X,
    /// Ey on (z, x).
    Y,
}

const NU: usize = 16;
const NV: usize = 8;

fn cavity_dims(o: Orient) -> (usize, usize, usize) {
    match o {
        Orient::Z => (NU, NV, 1),
        Orient::X => (1, NU, NV),
        Orient::Y => (NV, 1, NU),
    }
}

fn cavity_sample(o: Orient, u: usize, v: usize) -> (Component, usize, usize, usize) {
    match o {
        Orient::Z => (Component::Ez, u, v, 0),
        Orient::X => (Component::Ex, 0, u, v),
        Orient::Y => (Component::Ey, v, 0, u),
    }
}

/// Phase per step `θ = ωS` from zero crossings of `E` at one probe over
/// `steps` steps: linear interpolation of each crossing, then a least-squares
/// line through crossing index vs time; half period `π/θ`.
fn theta_from_crossings(x: &[f64]) -> f64 {
    let mut times = Vec::new();
    for n in 1..x.len() {
        let (a, b) = (x[n - 1], x[n]);
        if a == 0.0 || (a > 0.0) != (b > 0.0) {
            times.push((n - 1) as f64 + a / (a - b));
        }
    }
    let m = times.len() as f64;
    let mean_i = (m - 1.0) / 2.0;
    let mean_t = times.iter().sum::<f64>() / m;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, t) in times.iter().enumerate() {
        sxy += (i as f64 - mean_i) * (t - mean_t);
        sxx += (i as f64 - mean_i).powi(2);
    }
    PI / (sxy / sxx)
}

fn cavity_theta(o: Orient, material: Material, steps: usize) -> f64 {
    let (nx, ny, nz) = cavity_dims(o);
    let mut map = MaterialMap::vacuum(nx, ny, nz);
    map.fill([0, 0, 0], [nx, ny, nz], material);
    let mut g = YeeGrid::new(nx, ny, nz, COURANT_3D)
        .with_materials(&map)
        .expect("map");
    let (m, n) = (2.0, 1.0);
    for u in 0..=NU {
        for v in 0..=NV {
            let val = (PI * m * u as f64 / NU as f64).sin() * (PI * n * v as f64 / NV as f64).sin();
            let (c, i, j, k) = cavity_sample(o, u, v);
            g.set(c, i, j, k, Fix128::from_f64(val));
        }
    }
    let (c, i, j, k) = cavity_sample(o, NU / 8, NV / 2);
    let mut series = Vec::with_capacity(steps);
    for _ in 0..steps {
        g.step();
        series.push(g.get(c, i, j, k).to_f64());
    }
    theta_from_crossings(&series)
}

/// TM₂₁ mode of a `16 × 8` cavity, one cell thick, in all three orientations.
/// Discrete dispersion (derived above with `σ = 0`):
/// `sin(θ/2) = (S/n) · √(sin²(2π/32) + sin²(π/16))`, `n² = εμ`. The medium
/// `ε = 9/8`, `μ = 2` (`n = 3/2`) exercises both the `E` and the `H`
/// coefficient; the measured phase velocity relative to the vacuum run is
/// `1/n` up to the lattice dispersion (`0.13 %` at this resolution).
#[test]
fn cavity_mode_frequency_gives_phase_velocity_c_over_n() {
    let steps = 2400;
    let lam = (PI * 2.0 / (2.0 * NU as f64)).sin().powi(2) + (PI / (2.0 * NV as f64)).sin().powi(2);
    let n_idx = 1.5;
    let theta_vac = 2.0 * (S * lam.sqrt()).asin();
    let theta_med = 2.0 * (S * lam.sqrt() / n_idx).asin();
    for o in [Orient::Z, Orient::X, Orient::Y] {
        let tv = cavity_theta(o, Material::VACUUM, steps);
        let tm = cavity_theta(o, Material::new(q(9, 8), q(2, 1), Fix128::ZERO), steps);
        assert!(
            (tv / theta_vac - 1.0).abs() < 2e-5,
            "vacuum θ {tv} vs {theta_vac}"
        );
        assert!(
            (tm / theta_med - 1.0).abs() < 2e-5,
            "medium θ {tm} vs {theta_med}"
        );
        // Exact discrete relation between the two runs.
        assert!(((tm / 2.0).sin() * n_idx / (tv / 2.0).sin() - 1.0).abs() < 4e-5);
        // Phase velocity ratio (same k) = ω_m/ω_v → 1/n.
        assert!(
            (tm / tv * n_idx - 1.0).abs() < 3e-3,
            "v_p ratio {} vs 1/n",
            tm / tv
        );
    }
}

// ----------------------------------------------------------------------------
// discrete energy
// ----------------------------------------------------------------------------

const ALL_E: [Component; 3] = [Component::Ex, Component::Ey, Component::Ez];
const ALL_H: [Component; 3] = [Component::Hx, Component::Hy, Component::Hz];

fn axis(c: Component) -> usize {
    match c {
        Component::Ex | Component::Hx => 0,
        Component::Ey | Component::Hy => 1,
        Component::Ez | Component::Hz => 2,
    }
}

/// Cell ranges touching a sample along each axis, from the doc rule: an `E`
/// edge lies in one cell along its own axis and between up to two along the
/// others; an `H` face separates up to two cells along its normal and lies in
/// one along the others.
fn touching(c: Component, pos: [usize; 3], dims: [usize; 3]) -> Vec<[usize; 3]> {
    let electric = matches!(c, Component::Ex | Component::Ey | Component::Ez);
    let a = axis(c);
    let mut lists: Vec<Vec<usize>> = Vec::new();
    for d in 0..3 {
        let own = d == a;
        if own == electric {
            lists.push(vec![pos[d]]);
        } else {
            let mut l = Vec::new();
            if pos[d] >= 1 {
                l.push(pos[d] - 1);
            }
            if pos[d] < dims[d] {
                l.push(pos[d]);
            }
            lists.push(l);
        }
    }
    let mut out = Vec::new();
    for &i in &lists[0] {
        for &j in &lists[1] {
            for &k in &lists[2] {
                out.push([i, j, k]);
            }
        }
    }
    out
}

/// Per-sample weights in f64: `(ε, σ)` arithmetic means for `E`, harmonic
/// mean of `μ` for `H`.
struct Weights {
    e: Vec<Vec<(f64, f64)>>,
    h: Vec<Vec<f64>>,
}

fn weights(g: &YeeGrid, map: &MaterialMap) -> Weights {
    let (nx, ny, nz) = g.dims();
    let dims = [nx, ny, nz];
    let mut e = Vec::new();
    for c in ALL_E {
        let (a, b, d) = g.component_dims(c);
        let mut w = Vec::with_capacity(a * b * d);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    let cells = touching(c, [i, j, k], dims);
                    let m = cells.len() as f64;
                    let eps = cells
                        .iter()
                        .map(|p| map.get(p[0], p[1], p[2]).eps_r.to_f64())
                        .sum::<f64>()
                        / m;
                    let sig = cells
                        .iter()
                        .map(|p| map.get(p[0], p[1], p[2]).sigma.to_f64())
                        .sum::<f64>()
                        / m;
                    w.push((eps, sig));
                }
            }
        }
        e.push(w);
    }
    let mut h = Vec::new();
    for c in ALL_H {
        let (a, b, d) = g.component_dims(c);
        let mut w = Vec::with_capacity(a * b * d);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    let cells = touching(c, [i, j, k], dims);
                    let m = cells.len() as f64;
                    let inv = cells
                        .iter()
                        .map(|p| 1.0 / map.get(p[0], p[1], p[2]).mu_r.to_f64())
                        .sum::<f64>();
                    w.push(m / inv);
                }
            }
        }
        h.push(w);
    }
    Weights { e, h }
}

fn snapshot(g: &YeeGrid, comps: [Component; 3]) -> Vec<Vec<f64>> {
    comps
        .iter()
        .map(|&c| {
            let (a, b, d) = g.component_dims(c);
            let mut v = Vec::with_capacity(a * b * d);
            for i in 0..a {
                for j in 0..b {
                    for k in 0..d {
                        v.push(g.get(c, i, j, k).to_f64());
                    }
                }
            }
            v
        })
        .collect()
}

/// A 9×8×7 lattice with a dielectric slab, a magnetic block overlapping it,
/// and (optionally) a conductive region, seeded with a fixed pseudo-random
/// field on every updated sample.
fn energy_scene(lossy: bool) -> (YeeGrid, MaterialMap) {
    let (nx, ny, nz) = (9, 8, 7);
    let mut map = MaterialMap::vacuum(nx, ny, nz);
    map.fill([2, 0, 0], [6, 8, 7], Material::dielectric(q(5, 2)));
    map.fill(
        [4, 3, 1],
        [9, 7, 5],
        Material::new(q(5, 2), q(7, 4), Fix128::ZERO),
    );
    if lossy {
        map.fill(
            [0, 2, 2],
            [5, 6, 6],
            Material::new(q(3, 2), Fix128::ONE, q(3, 4)),
        );
    }
    let mut g = YeeGrid::new(nx, ny, nz, COURANT_3D)
        .with_materials(&map)
        .expect("map");
    let mut state = 0x00DD_BA11_u64;
    let (gx, gy, gz) = (nx, ny, nz);
    for c in [
        Component::Ex,
        Component::Ey,
        Component::Ez,
        Component::Hx,
        Component::Hy,
        Component::Hz,
    ] {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    // Tangential E on a PEC wall stays zero.
                    let on_wall = match c {
                        Component::Ex => j == 0 || j == gy || k == 0 || k == gz,
                        Component::Ey => i == 0 || i == gx || k == 0 || k == gz,
                        Component::Ez => i == 0 || i == gx || j == 0 || j == gy,
                        _ => false,
                    };
                    if on_wall {
                        continue;
                    }
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let v = ((state >> 40) % 2001) as i64 - 1000;
                    g.set(c, i, j, k, q(v, 1024));
                }
            }
        }
    }
    (g, map)
}

fn weighted_dot(a: &[Vec<f64>], b: &[Vec<f64>], w: &dyn Fn(usize, usize) -> f64) -> f64 {
    let mut s = 0.0;
    for c in 0..3 {
        for (n, (x, y)) in a[c].iter().zip(&b[c]).enumerate() {
            s += w(c, n) * x * y;
        }
    }
    s
}

/// `W^{n+½} = Σ ε E^n·E^{n+1} + Σ μ |H^{n+½}|²` is invariant under the
/// lossless leap-frog (the `E` and `H` curls are negative adjoints on the
/// Yee lattice, so the cross terms telescope) for any positive per-sample
/// weights — provided the weights the update uses are the ones in `W`.
/// Weights here come from the documented averaging rules, not from the
/// lattice. Also: `W` is positive (the scene is below the Courant bound).
#[test]
fn lossless_inhomogeneous_lattice_conserves_discrete_energy() {
    let (mut g, map) = energy_scene(false);
    let w = weights(&g, &map);
    let mut e_prev = snapshot(&g, ALL_E);
    let mut first = None;
    let mut worst: f64 = 0.0;
    for _ in 0..400 {
        g.step();
        let e_now = snapshot(&g, ALL_E);
        let h = snapshot(&g, ALL_H);
        let we = weighted_dot(&e_prev, &e_now, &|c, n| w.e[c][n].0);
        let wh = weighted_dot(&h, &h, &|c, n| w.h[c][n]);
        let total = we + wh;
        assert!(total > 0.0);
        let w0 = *first.get_or_insert(total);
        worst = worst.max((total - w0).abs() / w0);
        e_prev = e_now;
    }
    assert!(worst < 1e-12, "relative energy drift {worst:e}");
}

/// With conductivity the same `W` changes per step by exactly
/// `−(S/2) Σ σ E^n·(E^{n+1} + 2E^n + E^{n−1})` (from
/// `ε(E^{n+1} − E^n) = S ∇×H − (σS/2)(E^{n+1} + E^n)`), and over every
/// 10-step window it does not increase.
#[test]
fn conductive_lattice_loses_energy_by_the_dissipation_identity() {
    let (mut g, map) = energy_scene(true);
    let w = weights(&g, &map);
    let mut e_m1 = snapshot(&g, ALL_E);
    g.step();
    let mut e_0 = snapshot(&g, ALL_E);
    let mut h_prev = snapshot(&g, ALL_H);
    let energy = |ea: &[Vec<f64>], eb: &[Vec<f64>], h: &[Vec<f64>]| {
        weighted_dot(ea, eb, &|c, n| w.e[c][n].0) + weighted_dot(h, h, &|c, n| w.h[c][n])
    };
    let mut w_prev = energy(&e_m1, &e_0, &h_prev);
    let w_start = w_prev;
    let mut history = vec![w_prev];
    for _ in 0..300 {
        g.step();
        let e_1 = snapshot(&g, ALL_E);
        let h = snapshot(&g, ALL_H);
        let w_now = energy(&e_0, &e_1, &h);
        let mut diss = 0.0;
        for c in 0..3 {
            for n in 0..e_0[c].len() {
                diss += w.e[c][n].1 * e_0[c][n] * (e_1[c][n] + 2.0 * e_0[c][n] + e_m1[c][n]);
            }
        }
        let predicted = -(S / 2.0) * diss;
        let actual = w_now - w_prev;
        assert!(
            (actual - predicted).abs() <= 1e-10 * w_start,
            "ΔW {actual:e} vs identity {predicted:e}"
        );
        history.push(w_now);
        e_m1 = e_0;
        e_0 = e_1;
        h_prev = h;
        w_prev = w_now;
    }
    let _ = h_prev;
    for win in history.chunks(10).collect::<Vec<_>>().windows(2) {
        assert!(win[1][0] <= win[0][0] * (1.0 + 1e-12), "energy rose");
    }
    let end = *history.last().expect("non-empty");
    assert!(end < 0.5 * w_start, "loss visible: {end} vs {w_start}");
}

// ----------------------------------------------------------------------------
// Courant bound with materials
// ----------------------------------------------------------------------------

/// `S = 3/4` exceeds the vacuum limit `1/√3`; `3S² = 27/16`.
///
/// - every cell `ε = 2`: `27/16 ≤ 2` ⇒ accepted, and the energy `W` of
///   [`lossless_inhomogeneous_lattice_conserves_discrete_energy`] stays
///   constant over 1500 steps (stable);
/// - `ε = 3/2, μ = 9/8`: `εμ = 27/16` exactly ⇒ accepted (equality), and
///   with `μ` lowered by `2⁻¹⁰` ⇒ refused;
/// - one vacuum cell in an `ε = 2` lattice ⇒ `CourantViolated` with
///   `ε_min = μ_min = 1`;
/// - no map at all (`YeeGrid::new` does not check): the same seed grows by
///   more than `10⁶` (unstable).
#[test]
fn courant_bound_admits_s_above_vacuum_limit_only_in_slow_media() {
    let s = q(3, 4);
    let (nx, ny, nz) = (6, 5, 4);
    let seed = |g: &mut YeeGrid| {
        // A checkerboard on interior Ez: the highest spatial frequency, which
        // is what an instability amplifies first.
        for i in 1..nx {
            for j in 1..ny {
                for k in 0..nz {
                    let v = if (i + j + k) % 2 == 0 { 1 } else { -1 };
                    g.set(Component::Ez, i, j, k, Fix128::from_int(v));
                }
            }
        }
    };

    let mut slow = MaterialMap::vacuum(nx, ny, nz);
    slow.fill([0, 0, 0], [nx, ny, nz], Material::dielectric(q(2, 1)));
    let mut g = YeeGrid::new(nx, ny, nz, s)
        .with_materials(&slow)
        .expect("3S² = 27/16 ≤ 2");
    seed(&mut g);
    let w = weights(&g, &slow);
    let mut e_prev = snapshot(&g, ALL_E);
    let mut w0 = None;
    for _ in 0..1500 {
        g.step();
        let e_now = snapshot(&g, ALL_E);
        let h = snapshot(&g, ALL_H);
        let total = weighted_dot(&e_prev, &e_now, &|c, n| w.e[c][n].0)
            + weighted_dot(&h, &h, &|c, n| w.h[c][n]);
        let base = *w0.get_or_insert(total);
        assert!((total - base).abs() < 1e-11 * base, "unstable at ε = 2");
        e_prev = e_now;
    }
    assert!(g.max_abs_field().to_f64() < 10.0);

    let mut eq = MaterialMap::vacuum(nx, ny, nz);
    eq.fill(
        [0, 0, 0],
        [nx, ny, nz],
        Material::new(q(3, 2), q(9, 8), Fix128::ZERO),
    );
    assert!(YeeGrid::new(nx, ny, nz, s).with_materials(&eq).is_ok());
    // μ lowered by 2⁻¹⁰: εμ = 27/16 − 3/2048 < 3S², refused.
    let mut below = MaterialMap::vacuum(nx, ny, nz);
    below.fill(
        [0, 0, 0],
        [nx, ny, nz],
        Material::new(q(3, 2), q(9, 8) - q(1, 1024), Fix128::ZERO),
    );
    assert!(matches!(
        YeeGrid::new(nx, ny, nz, s).with_materials(&below),
        Err(MaterialError::CourantViolated { .. })
    ));

    let mut holed = slow.clone();
    holed.set(3, 2, 1, Material::VACUUM);
    match YeeGrid::new(nx, ny, nz, s).with_materials(&holed) {
        Err(MaterialError::CourantViolated {
            courant,
            eps_min,
            mu_min,
        }) => {
            assert_eq!(courant, s);
            assert_eq!(eps_min, Fix128::ONE);
            assert_eq!(mu_min, Fix128::ONE);
        }
        other => panic!("expected CourantViolated, got {other:?}"),
    }

    let mut bare = YeeGrid::new(nx, ny, nz, s);
    seed(&mut bare);
    for _ in 0..300 {
        bare.step();
    }
    assert!(
        bare.max_abs_field().to_f64() > 1e6,
        "vacuum at S = 3/4 should blow up, max |f| = {}",
        bare.max_abs_field().to_f64()
    );
}
