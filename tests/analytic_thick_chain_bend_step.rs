//! Oracles for the substep of a thick chain with the bending law, near the wall of the law.
//!
//! A substep is the minimum of
//!
//! ```text
//! Ψ(x) = Σ_i m_i |x_i − x̃_i|² / 2 + Σ_c [ C_c(x)² + γ (C_c(x) − C_c(x_start))² ] / (2 α_c / h²)
//! ```
//!
//! (`h²` times the incremental potential of the step: `x̃` the particles after the velocity
//! step, `C_c` the links' stretch and the joints' bending constraints, `α_c` their
//! compliances, `γ = τ / h` the Kelvin-Voigt rate), and the passes of a substep are steps
//! towards it. What is asked here, of the passes themselves:
//!
//! - `Ψ` does not rise from one pass to the next;
//! - after 8 and after 16 passes the chain is within a stated distance of the minimum.
//!
//! The minimum is found here in `f64`, apart from the library: the constraints written
//! from the law's statement (the stretch `λ − 1/λ`; the joint `κ S`, `κ = 2 tan(θ/2) / l̄`,
//! `S² = Σ (A/2) r0² (1 + λ⁻⁴ g(κ r0 / λ))`, `g(a) = (artanh(a)/a − 1)/a²`), their slopes
//! by the complex step (exact to rounding, no formula of the library's read), and Newton's
//! method with the second slopes by differences of that gradient, each step shortened
//! until `Ψ` falls. The reference is accepted only with a gradient at rounding.
//!
//! The scenes are arcs of constant curvature let go from rest with the root clamped:
//! `a = κ r0 / λ` is 0.86 and 0.67 of the wall at 1, where the law's shape steepens. The
//! distances asked for are four times what a step of the same kind takes in `f64` (the
//! matrix without the second slopes, the first of the step, its half, quarter, eighth and
//! sixteenth that lowers `Ψ` by a ten-thousandth of what the slope promises): one scene
//! (the softer chain at a sixteenth of the step) closes slowly with that matrix, 5·10⁻⁷ m
//! left after 16 passes, and is asked that much. With a retardation time of 0.05 s the
//! damping's own stiffness (`γ` of 48 to 96 here) leaves the constraints' second slopes
//! little to say: 10⁻⁹ m is asked after 8 passes.

#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    reason = "analytic reference: the minimum is found in f64, never by the code under test; its solve indexes rows and columns as the algebra is written"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainBend, ThickChainConfig, ThickChainFit, ThickChainStretch,
};

const LINKS: usize = 12;
const LINK: f64 = 0.01;
const RADIUS: f64 = 0.02;
const DENSITY: f64 = 1000.0;
const AREA: f64 = 2.0 * RADIUS * LINK;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

// ---- complex numbers for the complex step ----

#[derive(Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    fn add(self, o: Self) -> Self {
        Self { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, im: self.im - o.im }
    }
    fn mul(self, o: Self) -> Self {
        Self { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn div(self, o: Self) -> Self {
        let n = o.re * o.re + o.im * o.im;
        Self {
            re: (self.re * o.re + self.im * o.im) / n,
            im: (self.im * o.re - self.re * o.im) / n,
        }
    }
    fn scale(self, k: f64) -> Self {
        Self { re: self.re * k, im: self.im * k }
    }
    fn sqrt(self) -> Self {
        let r = self.re.hypot(self.im);
        let re = ((r + self.re) / 2.0).sqrt();
        Self { re, im: self.im / (2.0 * re) }
    }
    fn ln(self) -> Self {
        Self { re: self.re.hypot(self.im).ln(), im: self.im.atan2(self.re) }
    }
}

/// `g(a) = (artanh(a)/a − 1)/a²`, `artanh(a) = ln((1 + a)/(1 − a)) / 2`; its series near 0
fn shape(a: Cx) -> Cx {
    let a2 = a.mul(a);
    if a.re.abs() < 1.0e-3 {
        return Cx::real(1.0 / 3.0).add(a2.scale(1.0 / 5.0));
    }
    let one = Cx::real(1.0);
    let artanh = one.add(a).div(one.sub(a)).ln().scale(0.5);
    artanh.div(a).sub(one).div(a2)
}

/// every constraint of the chain at `x` (two coordinates a particle): link 0's stretch,
/// the joint at particle 1, link 1's stretch, …
fn constraints(x: &[Cx]) -> Vec<Cx> {
    let d: Vec<[Cx; 2]> = (0..LINKS)
        .map(|k| [x[2 * k + 2].sub(x[2 * k]), x[2 * k + 3].sub(x[2 * k + 1])])
        .collect();
    let len: Vec<Cx> = d.iter().map(|v| v[0].mul(v[0]).add(v[1].mul(v[1])).sqrt()).collect();
    let mut out = Vec::with_capacity(2 * LINKS - 1);
    for at in 0..2 * LINKS - 1 {
        if at % 2 == 0 {
            let ratio = len[at / 2].scale(1.0 / LINK);
            out.push(ratio.sub(Cx::real(1.0).div(ratio)));
        } else {
            let j = at / 2 + 1;
            let (da, db, la, lb) = (d[j - 1], d[j], len[j - 1], len[j]);
            let cross = da[0].mul(db[1]).sub(da[1].mul(db[0]));
            let open = la.mul(lb).add(da[0].mul(db[0])).add(da[1].mul(db[1]));
            let kappa = cross.div(open).scale(2.0).div(la.add(lb).scale(0.5));
            let mut total = Cx::real(0.0);
            for l in [la, lb] {
                let back = Cx::real(LINK).div(l);
                let back2 = back.mul(back);
                let g = shape(kappa.mul(back).scale(RADIUS));
                total = total.add(
                    Cx::real(1.0).add(back2.mul(back2).mul(g)).scale(AREA / 2.0 * RADIUS * RADIUS),
                );
            }
            out.push(kappa.mul(total.sqrt()));
        }
    }
    out
}

/// the largest `|a|` of the joints at `x`; infinite when a joint is folded flat
fn nearest_the_wall(x: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for j in 1..LINKS {
        let da = [x[2 * j] - x[2 * j - 2], x[2 * j + 1] - x[2 * j - 1]];
        let db = [x[2 * j + 2] - x[2 * j], x[2 * j + 3] - x[2 * j + 1]];
        let (la, lb) = (da[0].hypot(da[1]), db[0].hypot(db[1]));
        let open = la * lb + da[0] * db[0] + da[1] * db[1];
        if open <= 0.0 {
            return f64::INFINITY;
        }
        let kappa = 2.0 * (da[0] * db[1] - da[1] * db[0]) / open / ((la + lb) / 2.0);
        worst = worst.max((kappa * RADIUS * LINK / la).abs()).max((kappa * RADIUS * LINK / lb).abs());
    }
    worst
}

/// one substep's problem: the chain's shear modulus, the substep, the retardation time,
/// where the particles are after the velocity step and where they started
struct Problem {
    shear: f64,
    h: f64,
    tau: f64,
    start: Vec<f64>,
}

impl Problem {
    /// `α_c / h²` of every constraint
    fn softness(&self) -> Vec<f64> {
        (0..2 * LINKS - 1)
            .map(|at| {
                if at % 2 == 0 {
                    1.0 / (self.shear * AREA * self.h * self.h)
                } else {
                    1.0 / (self.shear * self.h * self.h)
                }
            })
            .collect()
    }

    fn mass(i: usize) -> f64 {
        DENSITY * AREA * if i == 0 || i == LINKS { 0.5 } else { 1.0 }
    }

    fn values(x: &[f64]) -> Vec<f64> {
        let at: Vec<Cx> = x.iter().map(|&v| Cx::real(v)).collect();
        constraints(&at).iter().map(|c| c.re).collect()
    }

    /// `Ψ`; infinite past the wall of the law
    fn psi(&self, x: &[f64]) -> f64 {
        if nearest_the_wall(x) >= 1.0 {
            return f64::INFINITY;
        }
        let (now, first) = (Self::values(x), Self::values(&self.start));
        let rate = self.tau / self.h;
        let mut sum = 0.0;
        for f in 4..x.len() {
            sum += Self::mass(f / 2) * (x[f] - self.start[f]).powi(2) / 2.0;
        }
        for (c, soft) in self.softness().iter().enumerate() {
            sum += (now[c] * now[c] + rate * (now[c] - first[c]).powi(2)) / (2.0 * soft);
        }
        sum
    }

    /// `∇Ψ` on the free coordinates (particles 2 on), the slopes by the complex step
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        const STEP: f64 = 1.0e-30;
        let (now, first) = (Self::values(x), Self::values(&self.start));
        let soft = self.softness();
        let rate = self.tau / self.h;
        (4..x.len())
            .map(|f| {
                let mut at: Vec<Cx> = x.iter().map(|&v| Cx::real(v)).collect();
                at[f].im = STEP;
                let slope = constraints(&at);
                let mut g = Self::mass(f / 2) * (x[f] - self.start[f]);
                for c in 0..now.len() {
                    g += slope[c].im / STEP * (now[c] + rate * (now[c] - first[c])) / soft[c];
                }
                g
            })
            .collect()
    }

    /// the minimum of `Ψ` from the start: Newton's method, the second slopes by central
    /// differences of the gradient; a step that does not lower `Ψ` is halved, and where no
    /// part of it does (the second slopes are not positive far from the minimum) the
    /// matrix is weighted towards the masses until one does
    fn minimum(&self) -> Vec<f64> {
        let mut x = self.start.clone();
        let n = x.len() - 4;
        let mut weight = 0.0f64;
        for _ in 0..400 {
            let g = self.gradient(&x);
            let mut h = vec![vec![0.0f64; n]; n];
            for j in 0..n {
                let (mut up, mut down) = (x.clone(), x.clone());
                up[4 + j] += 1.0e-8;
                down[4 + j] -= 1.0e-8;
                let (gu, gd) = (self.gradient(&up), self.gradient(&down));
                for i in 0..n {
                    h[i][j] = (gu[i] - gd[i]) / 2.0e-8;
                }
            }
            for (i, row) in h.iter_mut().enumerate() {
                row[i] += weight * Self::mass(2 + i / 2);
            }
            let step = solve(&h, &g.iter().map(|v| -v).collect::<Vec<_>>());
            let here = self.psi(&x);
            let mut t = 1.0;
            let mut moved = false;
            for _ in 0..12 {
                let mut trial = x.clone();
                for i in 0..n {
                    trial[4 + i] += t * step[i];
                }
                if self.psi(&trial) < here {
                    x = trial;
                    moved = true;
                    break;
                }
                t /= 2.0;
            }
            if moved {
                weight /= 4.0;
            } else if weight > 1.0e12 {
                break;
            } else {
                weight = (weight * 10.0).max(1.0e-3);
            }
        }
        x
    }
}

/// Gaussian elimination with partial pivoting
fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &v)| {
            let mut row = row.clone();
            row.push(v);
            row
        })
        .collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).expect("a row");
        m.swap(c, p);
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            for k in c..=n {
                m[r][k] -= f * m[c][k];
            }
        }
    }
    let mut out = vec![0.0; n];
    for c in (0..n).rev() {
        let mut v = m[c][n];
        for k in c + 1..n {
            v -= m[c][k] * out[k];
        }
        out[c] = v / m[c][c];
    }
    out
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// an arc of constant curvature from a clamped root along `x`
fn arc(radius: f64) -> Vec<f64> {
    let mut p = vec![0.0, 0.0, LINK, 0.0];
    let mut angle = 0.0f64;
    for _ in 2..=LINKS {
        angle += LINK / radius;
        let (x, y) = (p[p.len() - 2], p[p.len() - 1]);
        p.push(x + LINK * angle.cos());
        p.push(y + LINK * angle.sin());
    }
    p
}

/// where the library leaves the chain after one substep of `passes` passes from `problem`'s
/// start at rest
fn after(problem: &Problem, per: i64, passes: u32) -> Result<Vec<f64>, String> {
    let config = ThickChainConfig {
        substeps: 1,
        iterations: passes,
        gravity: [Fix128::ZERO, Fix128::ZERO],
        stretch: ThickChainStretch::NeoHookean { shear: fx(problem.shear) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        // one substep, not its parts: a refused substep is reported, not split
        splits: 0,
        relaxation: fx(problem.tau),
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let rest: Vec<[Fix128; 2]> =
        (0..=LINKS).map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO]).collect();
    let mut chain = ThickChain::new(config, &rest, fx(RADIUS), fx(DENSITY))
        .map_err(|e| format!("{e:?}"))?
        .bending(ThickChainBend::Section)
        .map_err(|e| format!("{e:?}"))?;
    chain.pin(0);
    chain.pin(1);
    for i in 0..=LINKS {
        chain.set_position(i, [fx(problem.start[2 * i]), fx(problem.start[2 * i + 1])]);
    }
    chain.step(Fix128::from_ratio(1, 240 * per)).map_err(|e| format!("{e:?}"))?;
    Ok(chain.positions().iter().flat_map(|p| [p[0].to_f64(), p[1].to_f64()]).collect())
}

/// (the arc's radius, the shear modulus, the substeps to 1/240 s, the retardation time,
/// the distance from the minimum asked after 8 passes and after 16, m)
const SCENES: [(f64, f64, i64, f64, f64, f64); 8] = [
    (0.0235, 6667.0, 1, 0.0, 3.0e-5, 5.2e-8),
    (0.0235, 6667.0, 4, 0.0, 7.0e-6, 1.7e-8),
    (0.0235, 6667.0, 16, 0.0, 1.9e-7, 2.3e-10),
    (0.0235, 1000.0, 1, 0.0, 9.8e-6, 4.6e-8),
    (0.0235, 1000.0, 8, 0.0, 6.1e-6, 2.2e-6),
    (0.03, 6667.0, 1, 0.0, 1.7e-5, 2.1e-6),
    (0.0235, 6667.0, 4, 0.05, 1.0e-9, 1.0e-9),
    (0.0235, 1000.0, 8, 0.05, 1.0e-9, 1.0e-9),
];

fn problem(scene: &(f64, f64, i64, f64, f64, f64)) -> Problem {
    Problem { shear: scene.1, h: 1.0 / 240.0 / scene.2 as f64, tau: scene.3, start: arc(scene.0) }
}

#[test]
fn the_reference_minimum_is_a_minimum_and_the_scenes_are_near_the_wall() {
    for scene in &SCENES {
        let p = problem(scene);
        let near = nearest_the_wall(&p.start);
        assert!(near > 0.65 && near < 0.9, "the scene starts at a = {near}");
        let x = p.minimum();
        let (first, left) = (norm(&p.gradient(&p.start)), norm(&p.gradient(&x)));
        // the second slopes of `Ψ` are the masses and more at a minimum, so the gradient left
        // over the lightest mass bounds the distance from it: a tenth of the least asked
        let within = left / (DENSITY * AREA / 2.0);
        assert!(
            left < 1.0e-6 * first && within < scene.5 / 10.0,
            "{scene:?}: the gradient {first:e} is left at {left:e}, {within:e} m from the minimum"
        );
        assert!(p.psi(&x) < p.psi(&p.start), "{scene:?}: nothing gained");
        let far = x.iter().zip(&p.start).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(far > 1.0e-5, "{scene:?}: the minimum is {far:e} m from the start, nothing to close");
    }
}

#[test]
fn near_the_wall_no_pass_of_a_substep_raises_its_potential() {
    let mut raised = Vec::new();
    for scene in &SCENES {
        let p = problem(scene);
        let mut before = p.psi(&p.start);
        for passes in 1..=16u32 {
            let x = match after(&p, scene.2, passes) {
                Ok(x) => x,
                Err(e) => {
                    raised.push(format!("{scene:?}: {passes} passes refused: {e}"));
                    break;
                }
            };
            let now = p.psi(&x);
            println!("potential {scene:?} passes {passes}: {now:.12e}");
            if now > before * (1.0 + 1.0e-12) {
                raised.push(format!("{scene:?}: pass {passes} raised the potential from {before:e} to {now:e}"));
            }
            before = now;
        }
    }
    assert!(raised.is_empty(), "{} rises:\n{}", raised.len(), raised.join("\n"));
}

#[test]
fn near_the_wall_the_passes_of_a_substep_close_on_its_minimum() {
    let mut far = Vec::new();
    for scene in &SCENES {
        let p = problem(scene);
        let want = p.minimum();
        for (passes, most) in [(8u32, scene.4), (16, scene.5)] {
            let x = match after(&p, scene.2, passes) {
                Ok(x) => x,
                Err(e) => {
                    far.push(format!("{scene:?}: {passes} passes refused: {e}"));
                    continue;
                }
            };
            let off = x.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            println!("distance {scene:?} passes {passes}: {off:.3e} of {most:.1e}");
            if off.is_nan() || off > most {
                far.push(format!("{scene:?}: {off:e} m from the minimum after {passes} passes, {most:e} asked"));
            }
        }
    }
    assert!(far.is_empty(), "{} too far:\n{}", far.len(), far.join("\n"));
}
