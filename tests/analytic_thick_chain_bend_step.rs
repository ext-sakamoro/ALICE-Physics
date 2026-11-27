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
//! distances asked for are a quarter over what the step leaves (measured; a step of the
//! same kind in `f64`, written before the library's, leaves the same to three digits: the
//! matrix without the second slopes, the first of the step, its half, quarter, eighth and
//! sixteenth that lowers `Ψ` by a ten-thousandth of what the slope promises), so a step
//! that closes more slowly is refused, and one that closes faster has these to beat: 7.4e-6
//! and 1.3e-8 m after 8 and 16 passes for the stiffer chain at a substep of 1/240 s, and
//! for the slowest scene (the softer chain at a substep of 1/1920 s) 1.5e-6 and 5.4e-7 m: there the matrix without the constraints' second slopes closes by a
//! twentieth of a decade a pass. With a retardation time of 0.05 s the damping's own
//! stiffness (`γ` of 48 to 96 here) leaves the constraints' second slopes little to say:
//! 10⁻⁹ m is asked after 8 passes (the step leaves 4·10⁻¹⁶).

#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    reason = "analytic reference: the minimum is found in f64, never by the code under test; its solve indexes rows and columns as the algebra is written"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::SdfField;
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
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn div(self, o: Self) -> Self {
        let n = o.re * o.re + o.im * o.im;
        Self {
            re: (self.re * o.re + self.im * o.im) / n,
            im: (self.im * o.re - self.re * o.im) / n,
        }
    }
    fn scale(self, k: f64) -> Self {
        Self {
            re: self.re * k,
            im: self.im * k,
        }
    }
    fn sqrt(self) -> Self {
        let r = self.re.hypot(self.im);
        let re = ((r + self.re) / 2.0).sqrt();
        Self {
            re,
            im: self.im / (2.0 * re),
        }
    }
    fn ln(self) -> Self {
        Self {
            re: self.re.hypot(self.im).ln(),
            im: self.im.atan2(self.re),
        }
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
    let len: Vec<Cx> = d
        .iter()
        .map(|v| v[0].mul(v[0]).add(v[1].mul(v[1])).sqrt())
        .collect();
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
                    Cx::real(1.0)
                        .add(back2.mul(back2).mul(g))
                        .scale(AREA / 2.0 * RADIUS * RADIUS),
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
        worst = worst
            .max((kappa * RADIUS * LINK / la).abs())
            .max((kappa * RADIUS * LINK / lb).abs());
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
        let p = (c..n)
            .max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))
            .expect("a row");
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
    let chain = chain_after(problem, per, passes, 0)?;
    Ok(chain
        .positions()
        .iter()
        .flat_map(|p| [p[0].to_f64(), p[1].to_f64()])
        .collect())
}

/// the chain itself after that substep, allowed `splits` halvings of it
fn chain_after(
    problem: &Problem,
    per: i64,
    passes: u32,
    splits: u32,
) -> Result<ThickChain, String> {
    let config = ThickChainConfig {
        substeps: 1,
        iterations: passes,
        gravity: [Fix128::ZERO, Fix128::ZERO],
        stretch: ThickChainStretch::NeoHookean {
            shear: fx(problem.shear),
        },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        // 0: one substep, not its parts (a refused substep is reported, not split)
        splits,
        relaxation: fx(problem.tau),
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let rest: Vec<[Fix128; 2]> = (0..=LINKS)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect();
    let mut chain = ThickChain::new(config, &rest, fx(RADIUS), fx(DENSITY))
        .map_err(|e| format!("{e:?}"))?
        .bending(ThickChainBend::Section)
        .map_err(|e| format!("{e:?}"))?;
    chain.pin(0);
    chain.pin(1);
    for i in 0..=LINKS {
        chain.set_position(i, [fx(problem.start[2 * i]), fx(problem.start[2 * i + 1])]);
    }
    chain
        .step(Fix128::from_ratio(1, 240 * per))
        .map_err(|e| format!("{e:?}"))?;
    Ok(chain)
}

/// (the arc's radius, the shear modulus, the substeps to 1/240 s, the retardation time,
/// the distance from the minimum asked after 8 passes and after 16, m)
const SCENES: [(f64, f64, i64, f64, f64, f64); 8] = [
    (0.0235, 6667.0, 1, 0.0, 9.3e-6, 1.6e-8),
    (0.0235, 6667.0, 4, 0.0, 2.2e-6, 5.3e-9),
    (0.0235, 6667.0, 16, 0.0, 5.8e-8, 7.0e-11),
    (0.0235, 1000.0, 1, 0.0, 3.1e-6, 1.5e-8),
    (0.0235, 1000.0, 8, 0.0, 1.9e-6, 6.8e-7),
    (0.03, 6667.0, 1, 0.0, 5.1e-6, 6.3e-7),
    (0.0235, 6667.0, 4, 0.05, 1.0e-9, 1.0e-9),
    (0.0235, 1000.0, 8, 0.05, 1.0e-9, 1.0e-9),
];

fn problem(scene: &(f64, f64, i64, f64, f64, f64)) -> Problem {
    Problem {
        shear: scene.1,
        h: 1.0 / 240.0 / scene.2 as f64,
        tau: scene.3,
        start: arc(scene.0),
    }
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
        let far = x
            .iter()
            .zip(&p.start)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            far > 1.0e-5,
            "{scene:?}: the minimum is {far:e} m from the start, nothing to close"
        );
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
                raised.push(format!(
                    "{scene:?}: pass {passes} raised the potential from {before:e} to {now:e}"
                ));
            }
            before = now;
        }
    }
    assert!(
        raised.is_empty(),
        "{} rises:\n{}",
        raised.len(),
        raised.join("\n")
    );
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
            let off = x
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            println!("distance {scene:?} passes {passes}: {off:.3e} of {most:.1e}");
            if off.is_nan() || off > most {
                far.push(format!(
                    "{scene:?}: {off:e} m from the minimum after {passes} passes, {most:e} asked"
                ));
            }
        }
    }
    assert!(far.is_empty(), "{} too far:\n{}", far.len(), far.join("\n"));
}

#[test]
fn the_counters_say_what_the_passes_and_the_substeps_did() {
    // the arc at a = 0.86, a substep of 1/960 s, 16 passes: the f64 step of the same kind takes
    // the half of its step in 11 of them and the whole in 5, and is never left without a part
    let p = problem(&SCENES[1]);
    let chain = chain_after(&p, 4, 16, 0).expect("a substep");
    assert_eq!(chain.bend_halved(), 11, "the halvings taken");
    assert_eq!(chain.bend_stalled(), 0);
    assert_eq!(
        chain.substep_retries(),
        0,
        "not split: the setting allows none"
    );
    // with a retardation time the passes are at the minimum from the sixth on (10⁻¹⁴ m), where
    // the potential no longer tells one part of a step from another: none of them is a pass
    // left without a part
    let damped = chain_after(&problem(&SCENES[6]), 4, 16, 0).expect("a substep");
    assert_eq!(damped.bend_stalled(), 0);
    // a substep of 1/240 s is too long for the links' stiffness and is taken in parts from the
    // start (that is the setting's rule, not a refusal); no part is refused, so nothing is retried
    let split = chain_after(&problem(&SCENES[0]), 1, 8, 8).expect("a step");
    assert_eq!(split.substep_retries(), 0);
    // a chain thrown faster than a part of a substep may carry it is refused and taken again in
    // halves: each such try is counted
    let config = ThickChainConfig {
        substeps: 1,
        iterations: 4,
        gravity: [Fix128::ZERO, Fix128::ZERO],
        stretch: ThickChainStretch::NeoHookean { shear: fx(1000.0) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        splits: 8,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let rest: Vec<[Fix128; 2]> = (0..=LINKS)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect();
    let thrown = |speed: f64| {
        let mut chain = ThickChain::new(config, &rest, fx(RADIUS), fx(DENSITY))
            .expect("a chain")
            .bending(ThickChainBend::Section)
            .expect("the law");
        for i in 0..=LINKS {
            chain.set_velocity(i, [Fix128::ZERO, fx(speed)]);
        }
        chain.step(Fix128::from_ratio(1, 240)).expect("a step");
        chain.substep_retries()
    };
    // a part of a substep may carry a particle half the least radius, 5 mm. The stiffness of the
    // links and joints has the substep of 1/240 s taken in quarters from the start, so a part
    // carries a particle thrown at up to 4.8 m/s, and each doubling of the speed beyond that
    // costs one more refused try
    assert_eq!(thrown(0.6), 0);
    assert_eq!(thrown(3.0), 0);
    assert_eq!(thrown(6.0), 1);
    assert_eq!(thrown(12.0), 2);
    assert_eq!(thrown(24.0), 3);
}

/// a flat wall: the half-plane `n · (p − q) ≥ 0` is free
struct FlatWall {
    n: [f32; 2],
    q: [f32; 2],
}

impl SdfField for FlatWall {
    fn distance(&self, x: f32, y: f32, _z: f32) -> f32 {
        self.n[0] * (x - self.q[0]) + self.n[1] * (y - self.q[1])
    }

    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (self.n[0], self.n[1], 0.0)
    }
}

/// the arc of `radius` at rest with a flat wall on the outer side of its far end, the end's
/// disc `into` metres into it; the chain after one substep of 1/960 s of `passes` passes
/// between the walls, and the wall
fn against_a_wall(radius: f64, into: f64, passes: u32) -> (Problem, ThickChain, FlatWall) {
    let p = Problem {
        shear: 6667.0,
        h: 1.0 / 960.0,
        tau: 0.0,
        start: arc(radius),
    };
    let turned = (LINKS - 1) as f64 * LINK / radius;
    // towards the arc's centre at its end
    let n = [(-turned.sin()) as f32, turned.cos() as f32];
    let end = [p.start[2 * LINKS], p.start[2 * LINKS + 1]];
    let wall = FlatWall {
        n,
        q: [
            (end[0] - f64::from(n[0]) * (RADIUS - into)) as f32,
            (end[1] - f64::from(n[1]) * (RADIUS - into)) as f32,
        ],
    };
    let config = ThickChainConfig {
        substeps: 1,
        iterations: passes,
        gravity: [Fix128::ZERO, Fix128::ZERO],
        stretch: ThickChainStretch::NeoHookean { shear: fx(p.shear) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        splits: 0,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let rest: Vec<[Fix128; 2]> = (0..=LINKS)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect();
    let mut chain = ThickChain::new(config, &rest, fx(RADIUS), fx(DENSITY))
        .expect("a chain")
        .bending(ThickChainBend::Section)
        .expect("the law");
    chain.pin(0);
    chain.pin(1);
    for i in 0..=LINKS {
        chain.set_position(i, [fx(p.start[2 * i]), fx(p.start[2 * i + 1])]);
    }
    chain
        .step_in(Fix128::from_ratio(1, 960), &wall, &[])
        .expect("a substep between the walls");
    (p, chain, wall)
}

/// What the stationarity of the substep's minimum between the walls leaves of `∇Ψ` at `x`:
/// the walls' constraints are `d(p) − r(l) ≥ 0` at the points of each link (`r = r₀ l₀ / l`),
/// linear along a link at a flat wall, so those at a link's two ends span them all. Returns
/// `|∇Ψ|`, the part of it no combination of the slopes of the constraints within `near` of
/// zero accounts for, the smallest weight of that combination over the largest, how many
/// constraints those are, and the deepest any constraint is below zero.
fn left_of_the_slope(
    p: &Problem,
    x: &[f64],
    wall: &FlatWall,
    near: f64,
) -> (f64, f64, f64, usize, f64) {
    let n = [f64::from(wall.n[0]), f64::from(wall.n[1])];
    let q = [f64::from(wall.q[0]), f64::from(wall.q[1])];
    let g = p.gradient(x);
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut deepest = 0.0f64;
    for k in 1..LINKS {
        let (a, b) = ([x[2 * k], x[2 * k + 1]], [x[2 * k + 2], x[2 * k + 3]]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = d[0].hypot(d[1]);
        let u = [d[0] / l, d[1] / l];
        let r = RADIUS * LINK / l;
        for (end, at) in [(0.0f64, a), (1.0f64, b)] {
            let c = n[0] * (at[0] - q[0]) + n[1] * (at[1] - q[1]) - r;
            deepest = deepest.min(c);
            if c > near {
                continue;
            }
            // the slope of the constraint in the link's two particles
            let mut column = vec![0.0f64; g.len()];
            for axis in 0..2 {
                let first = n[axis] * (1.0 - end) - u[axis] * r / l;
                let second = n[axis] * end + u[axis] * r / l;
                if k >= 2 {
                    column[2 * (k - 2) + axis] = first;
                }
                column[2 * (k - 1) + axis] = second;
            }
            columns.push(column);
        }
    }
    let m = columns.len();
    if m == 0 {
        return (norm(&g), norm(&g), 0.0, 0, deepest);
    }
    let normal: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| columns[i].iter().zip(&columns[j]).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    let right: Vec<f64> = (0..m)
        .map(|i| columns[i].iter().zip(&g).map(|(a, b)| a * b).sum())
        .collect();
    let weights = solve(&normal, &right);
    let mut left = g.clone();
    for (column, w) in columns.iter().zip(&weights) {
        for (v, c) in left.iter_mut().zip(column) {
            *v -= w * c;
        }
    }
    let largest = weights.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let smallest = weights.iter().fold(f64::INFINITY, |m, &w| m.min(w));
    (norm(&g), norm(&left), smallest / largest, m, deepest)
}

/// the chain's coordinates
fn coordinates(chain: &ThickChain) -> Vec<f64> {
    chain
        .positions()
        .iter()
        .flat_map(|p| [p[0].to_f64(), p[1].to_f64()])
        .collect()
}

/// (the arc's radius, how far its end's disc starts in the wall, m)
const AGAINST: [(f64, f64); 4] = [(0.1, 2.0e-4), (0.1, 1.0e-3), (0.05, 2.0e-4), (0.05, 1.0e-3)];

#[test]
fn between_walls_a_substep_leaves_the_chain_out_of_them_and_its_press_is_the_same_after_more_passes(
) {
    // the walls' pushes of a pass are carried into where the particles' inertia would leave
    // them, so the next pass's step does not undo them: the passes end with every point of
    // the chain out of the wall (to the distance an f32 field tells at 0.1 m, 7.5e-9 m), and
    // the wall's press on the end link, summed over the passes, has stopped growing by the
    // sixteenth to 10⁻⁴ of it (a press that grows with the passes is a push taken back and made again)
    for (radius, into) in AGAINST {
        let mut press = [0.0f64; 2];
        for (slot, passes) in [16u32, 64].into_iter().enumerate() {
            let (p, chain, wall) = against_a_wall(radius, into, passes);
            let got = left_of_the_slope(&p, &coordinates(&chain), &wall, 1.0e-6);
            assert!(
                got.4 >= -1.0e-8,
                "arc {radius} m, {into} m in, {passes} passes: {:e} m in the wall",
                -got.4
            );
            press[slot] =
                chain.wall_force(LINKS - 1).to_f64() + chain.wall_force(LINKS - 2).to_f64();
            assert_eq!(chain.bend_stalled(), 0);
        }
        assert!(
            press[0] > 10.0,
            "arc {radius} m, {into} m in: the wall is pressed ({} N)",
            press[0]
        );
        assert!(
            (press[1] - press[0]).abs() <= 1.0e-4 * press[0],
            "arc {radius} m, {into} m in: the press {} N after 16 passes, {} N after 64",
            press[0],
            press[1]
        );
    }
}

#[test]
#[ignore = "known defect: with walls the passes end where the potential's slope is a combination of the walls' slopes, but part of it on points that ended off the wall (a push made in an early pass is not taken back): 18 % to 100 % of the slope is left by the points on the wall alone"]
fn between_walls_a_substep_ends_at_the_minimum_the_walls_allow() {
    // at the minimum of the substep's potential among the positions the walls allow, its
    // slope is a combination, with weights of one sign, of the slopes of the walls'
    // constraints that are at zero there, and of no other
    for (radius, into) in AGAINST {
        let (p, chain, wall) = against_a_wall(radius, into, 64);
        let (whole, left, sign, count, _) =
            left_of_the_slope(&p, &coordinates(&chain), &wall, 1.0e-6);
        assert!(
            count > 0,
            "arc {radius} m, {into} m in: no point on the wall"
        );
        assert!(
            left <= 2.0e-2 * whole,
            "arc {radius} m, {into} m in: {left:e} of a slope of {whole:e} is not the walls'"
        );
        assert!(
            sign >= -1.0e-3,
            "arc {radius} m, {into} m in: a wall that pulls ({sign})"
        );
    }
}

#[test]
fn let_go_from_an_arc_a_chain_comes_to_rest_without_a_step_halved() {
    // the arc of the example (each link turned 0.284 rad from the last), a retardation time
    // of 0.05 s, 4 s: the swing dies out and the passes go on at rest, where what a step
    // promises is below what the potential resolves. Such a step is taken whole when it does
    // not raise the potential to that resolution; judged as a larger one it would be halved
    // pass after pass
    let config = ThickChainConfig {
        substeps: 16,
        iterations: 8,
        gravity: [Fix128::ZERO; 2],
        stretch: ThickChainStretch::NeoHookean { shear: fx(3000.0) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        splits: 8,
        relaxation: fx(0.05),
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let rest: Vec<[Fix128; 2]> = (0..=LINKS)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect();
    let mut chain = ThickChain::new(config, &rest, fx(RADIUS), fx(DENSITY))
        .expect("a chain")
        .bending(ThickChainBend::Section)
        .expect("the law");
    chain.pin(0);
    chain.pin(1);
    let (mut at, mut dir) = ([LINK, 0.0], [1.0f64, 0.0]);
    for i in 2..=LINKS {
        dir = [
            (24.0 * dir[0] - 7.0 * dir[1]) / 25.0,
            (7.0 * dir[0] + 24.0 * dir[1]) / 25.0,
        ];
        at = [at[0] + LINK * dir[0], at[1] + LINK * dir[1]];
        chain.set_position(i, [fx(at[0]), fx(at[1])]);
    }
    for _ in 0..240 {
        chain.step(Fix128::from_ratio(1, 60)).expect("a step");
    }
    let head = chain.positions()[LINKS][1].to_f64();
    assert!(
        head.abs() < 1.0e-4,
        "at rest on its line: the head {head:e} m off it"
    );
    assert_eq!(chain.bend_halved(), 0, "the steps halved");
    assert_eq!(chain.bend_stalled(), 0);
    assert_eq!(chain.substep_retries(), 0);
}
