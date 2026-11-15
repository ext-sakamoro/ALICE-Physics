//! The backward Euler step of [`Mpm2d`]: the grid velocities of the step minimise the
//! incremental potential, so the step length is set by accuracy, not by the elastic wave's
//! cell crossing time (the equations, the approximations and the tests are in the MODEL
//! section of [`super`]).

use super::{det, kirchhoff, mul, rotation, Mpm2d, Mpm2dError, Mpm2dModel, Stencil, M2, V2};
use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// How [`Mpm2d::step_implicit`] solves the step's system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplicitSolve {
    /// The step is solved when the residual force's mass-weighted RMS velocity
    /// (`sqrt(Σ |r_i|² / m_i / Σ m_i)`) is at most this, m/s
    pub tolerance: Fix128,
    /// Newton iterations allowed (counting every round of the ground's contact set)
    pub max_newton: u32,
    /// Conjugate gradient iterations allowed per Newton iteration
    pub max_cg: u32,
    /// A Newton direction is solved until the conjugate gradient's residual is at most this
    /// fraction of the Newton residual (same measure)
    pub cg_ratio: Fix128,
}

impl Default for ImplicitSolve {
    /// 1e-9 m/s, 40 Newton iterations, 4000 conjugate gradient iterations, ratio 1/100
    fn default() -> Self {
        Self {
            tolerance: Fix128::from_ratio(1, 1_000_000_000),
            max_newton: 40,
            max_cg: 4000,
            cg_ratio: Fix128::from_ratio(1, 100),
        }
    }
}

/// What one implicit step cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImplicitReport {
    /// Newton iterations (each one gradient, one direction, one line search)
    pub newton: u32,
    /// Conjugate gradient iterations over all Newton iterations (each one Hessian product)
    pub cg: u32,
    /// Halvings of the Newton step in the line searches
    pub halvings: u32,
}

/// One particle's part of the step's system: its stencil (node index, weight, node minus
/// particle) and the terms that do not depend on the unknown grid velocities.
struct Part {
    nodes: [(usize, Fix128, V2); 9],
    /// the deformation gradient at the step's start
    f0: M2,
    /// rest area × the stiffness factor `k` of the step (1 elastic; `1 − g + g γ` with a
    /// Maxwell branch, the branch's response to this step's strain)
    a0k: Fix128,
    /// rest area × the stress that does not depend on this step's strain (the branch's held
    /// part, rotated with the start rotation), Pa·m²
    konst: M2,
}

/// The step's system: the grid's masses and explicit velocities, which degrees of freedom are
/// held by the ground, and the particles' parts.
struct System {
    mass: Vec<Fix128>,
    vstar: Vec<V2>,
    held: Vec<[bool; 2]>,
    parts: Vec<Part>,
    dt: Fix128,
    /// `4 / Δx²`
    c4: Fix128,
    model: Mpm2dModel,
    mu: Fix128,
    lambda: Fix128,
}

fn tr(m: M2) -> M2 {
    [[m[0][0], m[1][0]], [m[0][1], m[1][1]]]
}

fn add(a: M2, b: M2) -> M2 {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

fn scale(s: Fix128, m: M2) -> M2 {
    [[s * m[0][0], s * m[0][1]], [s * m[1][0], s * m[1][1]]]
}

/// The first Piola–Kirchhoff stress `P` of `model` (`J F⁻ᵀ` is the cofactor matrix of `F`).
fn first_pk(model: Mpm2dModel, f: M2, mu: Fix128, lambda: Fix128) -> M2 {
    let j = det(f);
    let cof = [[f[1][1], -f[1][0]], [-f[0][1], f[0][0]]];
    if model == Mpm2dModel::NeoHookean {
        // μ (F − F⁻ᵀ) + λ ln J F⁻ᵀ, F⁻ᵀ = cof / J
        let lj = j.ln();
        let mut p = [[Fix128::ZERO; 2]; 2];
        for i in 0..2 {
            for k in 0..2 {
                let fit = cof[i][k] / j;
                p[i][k] = mu * (f[i][k] - fit) + lambda * lj * fit;
            }
        }
        return p;
    }
    // 2μ (F − R) + λ (J − 1) J F⁻ᵀ
    let r = rotation(f);
    let two_mu = Fix128::from_int(2) * mu;
    let l = lambda * (j - Fix128::ONE);
    let mut p = [[Fix128::ZERO; 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            p[i][k] = two_mu * (f[i][k] - r[i][k]) + l * cof[i][k];
        }
    }
    p
}

/// The energy density `ψ(F)` of `model` (per unit rest area).
pub(super) fn psi(model: Mpm2dModel, f: M2, mu: Fix128, lambda: Fix128) -> Fix128 {
    let two = Fix128::from_int(2);
    if model == Mpm2dModel::NeoHookean {
        let t = f[0][0] * f[0][0] + f[0][1] * f[0][1] + f[1][0] * f[1][0] + f[1][1] * f[1][1];
        let lj = det(f).ln();
        return mu * (t - two) / two - mu * lj + lambda * lj * lj / two;
    }
    let r = rotation(f);
    let mut dev = Fix128::ZERO;
    for (fr, rr) in f.iter().zip(&r) {
        for (&a, &b) in fr.iter().zip(rr) {
            let d = a - b;
            dev = dev + d * d;
        }
    }
    let j1 = det(f) - Fix128::ONE;
    mu * dev + lambda * j1 * j1 / two
}

impl System {
    /// `C = (4/Δx²) Σ_i w v_i (x_i − x_p)ᵀ` of a particle under the grid velocities `v`
    fn affine(&self, part: &Part, v: &[V2]) -> M2 {
        let mut b = [[Fix128::ZERO; 2]; 2];
        for &(idx, w, d) in &part.nodes {
            let vi = v[idx];
            for r in 0..2 {
                for q in 0..2 {
                    b[r][q] = b[r][q] + w * vi[r] * d[q];
                }
            }
        }
        scale(self.c4, b)
    }

    /// `F = (I + Δt C) F₀`
    fn deformation(&self, part: &Part, c: M2) -> M2 {
        let dt = self.dt;
        let g = [
            [Fix128::ONE + dt * c[0][0], dt * c[0][1]],
            [dt * c[1][0], Fix128::ONE + dt * c[1][1]],
        ];
        mul(g, part.f0)
    }

    /// The incremental potential `Σ_i ½ m_i |v_i − v*_i|² + Σ_p (A⁰ k ψ(F_p) + Δt K_p : C_p)`,
    /// or `None` when a particle's `det F ≤ 0` under `v`.
    fn energy(&self, v: &[V2]) -> Option<Fix128> {
        let half = Fix128::from_ratio(1, 2);
        let mut e = Fix128::ZERO;
        for ((m, vi), vs) in self.mass.iter().zip(v).zip(&self.vstar) {
            if *m == Fix128::ZERO {
                continue;
            }
            let d = [vi[0] - vs[0], vi[1] - vs[1]];
            e = e + half * *m * (d[0] * d[0] + d[1] * d[1]);
        }
        for part in &self.parts {
            let c = self.affine(part, v);
            let f = self.deformation(part, c);
            if det(f) <= Fix128::ZERO {
                return None;
            }
            e = e + part.a0k * psi(self.model, f, self.mu, self.lambda);
            let k = part.konst;
            e = e + self.dt
                * (k[0][0] * c[0][0] + k[0][1] * c[0][1] + k[1][0] * c[1][0] + k[1][1] * c[1][1]);
        }
        Some(e)
    }

    /// The gradient of [`Self::energy`] at `v`, the held degrees of freedom left as they are
    /// (the caller reads them for the ground's contact set) and `None` for a particle with
    /// `det F ≤ 0`.
    fn gradient(&self, v: &[V2]) -> Option<Vec<V2>> {
        let mut g: Vec<V2> = self
            .mass
            .iter()
            .zip(v)
            .zip(&self.vstar)
            .map(|((m, vi), vs)| [*m * (vi[0] - vs[0]), *m * (vi[1] - vs[1])])
            .collect();
        for part in &self.parts {
            let c = self.affine(part, v);
            let f = self.deformation(part, c);
            if det(f) <= Fix128::ZERO {
                return None;
            }
            // Δt (4/Δx²) (A⁰ k P F₀ᵀ + K) per node, times w (x_i − x_p)
            let p = first_pk(self.model, f, self.mu, self.lambda);
            let s = add(scale(part.a0k, mul(p, tr(part.f0))), part.konst);
            let s = scale(self.dt * self.c4, s);
            for &(idx, w, d) in &part.nodes {
                g[idx][0] = g[idx][0] + w * (s[0][0] * d[0] + s[0][1] * d[1]);
                g[idx][1] = g[idx][1] + w * (s[1][0] * d[0] + s[1][1] * d[1]);
            }
        }
        Some(g)
    }

    /// `H x` with `H` the mass plus the isotropic linear stiffness of every particle (`μ`, `λ`
    /// times `A⁰ k Δt²`): positive definite, an approximation of the true Hessian that the
    /// line search and the stop on the true gradient make exact in the limit. Held degrees of
    /// freedom give 0.
    fn hessian_times(&self, x: &[V2]) -> Vec<V2> {
        let mut y: Vec<V2> = self
            .mass
            .iter()
            .zip(x)
            .map(|(m, xi)| [*m * xi[0], *m * xi[1]])
            .collect();
        for part in &self.parts {
            let dc = self.affine(part, x);
            let t = dc[0][0] + dc[1][1];
            let sym = add(dc, tr(dc));
            let l = add(
                scale(self.mu, sym),
                [
                    [self.lambda * t, Fix128::ZERO],
                    [Fix128::ZERO, self.lambda * t],
                ],
            );
            let s = scale(part.a0k * self.dt * self.dt * self.c4, l);
            for &(idx, w, d) in &part.nodes {
                y[idx][0] = y[idx][0] + w * (s[0][0] * d[0] + s[0][1] * d[1]);
                y[idx][1] = y[idx][1] + w * (s[1][0] * d[0] + s[1][1] * d[1]);
            }
        }
        self.clear_held(&mut y);
        y
    }

    /// Every particle's deformation gradient under the grid velocities `v`
    fn deformations(&self, v: &[V2]) -> Vec<M2> {
        self.parts
            .iter()
            .map(|part| self.deformation(part, self.affine(part, v)))
            .collect()
    }

    /// `H x` with `H` the Hessian of [`Self::energy`] at the deformations `fs`: the mass plus,
    /// per particle, the derivative of its stress along the deformation `x` makes, by the
    /// central difference `(P(F + h δF) − P(F − h δF)) / 2h` with `h |δF| = 2⁻²⁰` (truncation
    /// of the order of `2⁻⁴⁰` relative). A particle whose `F ± h δF` is not invertible takes the
    /// isotropic stiffness of [`Self::hessian_times`]. Held degrees of freedom give 0.
    fn hessian_times_exact(&self, x: &[V2], fs: &[M2]) -> Vec<V2> {
        let mut y: Vec<V2> = self
            .mass
            .iter()
            .zip(x)
            .map(|(m, xi)| [*m * xi[0], *m * xi[1]])
            .collect();
        let two = Fix128::from_int(2);
        for (part, &f) in self.parts.iter().zip(fs) {
            let dc = self.affine(part, x);
            let df = scale(self.dt, mul(dc, part.f0));
            let mut big = Fix128::ZERO;
            for row in df {
                for e in row {
                    if e.abs() > big {
                        big = e.abs();
                    }
                }
            }
            if big == Fix128::ZERO {
                continue;
            }
            let h = Fix128::from_ratio(1, 1 << 20) / big;
            let up = add(f, scale(h, df));
            let dn = add(f, scale(-h, df));
            let dp = if det(up) > Fix128::ZERO && det(dn) > Fix128::ZERO {
                let pu = first_pk(self.model, up, self.mu, self.lambda);
                let pd = first_pk(self.model, dn, self.mu, self.lambda);
                let inv = Fix128::ONE / (two * h);
                mul(scale(inv, add(pu, scale(-Fix128::ONE, pd))), tr(part.f0))
            } else {
                let t = dc[0][0] + dc[1][1];
                let l = add(
                    scale(self.mu, add(dc, tr(dc))),
                    [
                        [self.lambda * t, Fix128::ZERO],
                        [Fix128::ZERO, self.lambda * t],
                    ],
                );
                scale(self.dt, l)
            };
            let s = scale(part.a0k * self.dt * self.c4, dp);
            for &(idx, w, d) in &part.nodes {
                y[idx][0] = y[idx][0] + w * (s[0][0] * d[0] + s[0][1] * d[1]);
                y[idx][1] = y[idx][1] + w * (s[1][0] * d[0] + s[1][1] * d[1]);
            }
        }
        self.clear_held(&mut y);
        y
    }

    /// The diagonal of [`Self::hessian_times`]'s matrix (the Jacobi preconditioner).
    fn diagonal(&self) -> Vec<V2> {
        let mut diag: Vec<V2> = self.mass.iter().map(|m| [*m, *m]).collect();
        for part in &self.parts {
            let s = part.a0k * self.dt * self.dt * self.c4 * self.c4;
            for &(idx, w, d) in &part.nodes {
                let dd = d[0] * d[0] + d[1] * d[1];
                for a in 0..2 {
                    let da = d[a] * d[a];
                    diag[idx][a] =
                        diag[idx][a] + s * w * w * (self.mu * (dd + da) + self.lambda * da);
                }
            }
        }
        diag
    }

    fn clear_held(&self, x: &mut [V2]) {
        for (xi, h) in x.iter_mut().zip(&self.held) {
            for a in 0..2 {
                if h[a] {
                    xi[a] = Fix128::ZERO;
                }
            }
        }
    }

    /// The mass-weighted RMS velocity of a force field over the free degrees of freedom:
    /// `sqrt(Σ_i |r_i|² / m_i / Σ_i m_i)`, m/s (the speed whose kinetic energy, spread over
    /// the grid's mass, equals the field's `½ rᵀ M⁻¹ r`); nodes of small mass count by their
    /// mass, so the rounding of their forces does not set the floor.
    fn residual(&self, r: &[V2]) -> Fix128 {
        let mut num = Fix128::ZERO;
        let mut total = Fix128::ZERO;
        for ((ri, m), h) in r.iter().zip(&self.mass).zip(&self.held) {
            if *m == Fix128::ZERO {
                continue;
            }
            total = total + *m;
            for a in 0..2 {
                if !h[a] {
                    num = num + ri[a] * ri[a] / *m;
                }
            }
        }
        if total == Fix128::ZERO {
            return Fix128::ZERO;
        }
        (num / total).sqrt()
    }
}

/// Solve `H d = r` by the conjugate gradient with the Jacobi preconditioner `diag`, `H` the
/// exact Hessian at the deformations `exact` or the isotropic one (`None`), until the
/// residual is at most `goal`. Returns the solution and whether a direction of non-positive
/// curvature stopped it.
fn conjugate_gradient(
    sys: &System,
    diag: &[V2],
    rhs: &[V2],
    goal: Fix128,
    exact: Option<&[M2]>,
    max: u32,
    report: &mut ImplicitReport,
) -> (Vec<V2>, bool) {
    let precond = |r: &[V2]| -> Vec<V2> {
        r.iter()
            .zip(diag)
            .zip(&sys.held)
            .map(|((x, dg), h)| {
                let mut z = [Fix128::ZERO; 2];
                for a in 0..2 {
                    if !h[a] && dg[a] != Fix128::ZERO {
                        z[a] = x[a] / dg[a];
                    }
                }
                z
            })
            .collect()
    };
    let mut d = vec![[Fix128::ZERO; 2]; rhs.len()];
    let mut r = rhs.to_vec();
    let mut z = precond(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..max {
        if sys.residual(&r) <= goal {
            break;
        }
        report.cg += 1;
        let hp = match exact {
            Some(fs) => sys.hessian_times_exact(&p, fs),
            None => sys.hessian_times(&p),
        };
        let php = dot(&p, &hp);
        if php <= Fix128::ZERO {
            return (d, true);
        }
        let alpha = rz / php;
        for ((di, ri), (pi, hpi)) in d.iter_mut().zip(r.iter_mut()).zip(p.iter().zip(&hp)) {
            for a in 0..2 {
                di[a] = di[a] + alpha * pi[a];
                ri[a] = ri[a] - alpha * hpi[a];
            }
        }
        z = precond(&r);
        let rz_next = dot(&r, &z);
        if rz == Fix128::ZERO {
            break;
        }
        let beta = rz_next / rz;
        rz = rz_next;
        for (pi, zi) in p.iter_mut().zip(&z) {
            for a in 0..2 {
                pi[a] = zi[a] + beta * pi[a];
            }
        }
    }
    (d, false)
}

/// `(up, down)`, powers of two with `max |x| · up / down` in `[1/2, 1)` (both 1 for zero)
fn power_of_two_scale(x: &[V2]) -> (Fix128, Fix128) {
    let mut m = Fix128::ZERO;
    for xi in x {
        for a in xi {
            if a.abs() > m {
                m = a.abs();
            }
        }
    }
    let (mut up, mut down) = (Fix128::ONE, Fix128::ONE);
    if m == Fix128::ZERO {
        return (up, down);
    }
    let two = Fix128::from_int(2);
    let half = Fix128::from_ratio(1, 2);
    while m * up / down < half && up < Fix128::from_int(1 << 60) {
        up = up * two;
    }
    while m * up / down >= Fix128::ONE && down < Fix128::from_int(1 << 60) {
        down = down * two;
    }
    (up, down)
}

fn dot(a: &[V2], b: &[V2]) -> Fix128 {
    a.iter()
        .zip(b)
        .fold(Fix128::ZERO, |s, (x, y)| s + x[0] * y[0] + x[1] * y[1])
}

impl Mpm2d {
    /// One backward Euler step of [`super::Mpm2dConfig::dt`]: the grid velocities minimise the
    /// incremental potential (inertia, gravity, external loads, the elastic energy of the
    /// deformation the step makes, the Maxwell branch's response), solved by Newton
    /// iterations with a conjugate gradient direction and a line search, and the ground as
    /// a contact set on its nodes; then the same grid to particle transfer as [`Self::step`].
    /// Stable for any step; the step is chosen for accuracy (it damps the modes it does not
    /// resolve, see the MODEL section of [`super`]).
    ///
    /// # Errors
    ///
    /// [`Mpm2dError::OutOfGrid`] as [`Self::step`]; [`Mpm2dError::NotConverged`] when the
    /// solve does not reach `solve.tolerance` within `solve.max_newton` iterations or a line
    /// search finds no decrease; [`Mpm2dError::Invalid`] for a load on no particle. A refused
    /// step changes nothing.
    pub fn step_implicit(&mut self, solve: &ImplicitSolve) -> Result<ImplicitReport, Mpm2dError> {
        self.step_implicit_with_loads(solve, &[])
    }

    /// [`Self::step_implicit`] with external forces on particles, as
    /// [`Self::step_with_loads`].
    ///
    /// # Errors
    ///
    /// As [`Self::step_implicit`].
    pub fn step_implicit_with_loads(
        &mut self,
        solve: &ImplicitSolve,
        loads: &[(usize, V2)],
    ) -> Result<ImplicitReport, Mpm2dError> {
        let mut force = vec![[Fix128::ZERO; 2]; self.particles.len()];
        for &(k, f) in loads {
            let at = force
                .get_mut(k)
                .ok_or(Mpm2dError::Invalid("loads: no such particle"))?;
            at[0] = at[0] + f[0];
            at[1] = at[1] + f[1];
        }
        let cfg = self.config;
        let (nx, ny) = (cfg.nodes[0], cfg.nodes[1]);
        let inv_dx = Fix128::ONE / cfg.cell;
        let c4 = Fix128::from_int(4) * inv_dx * inv_dx;
        let (mu, lambda) = self.lame();
        let dt = cfg.dt;
        let mut stencils = Vec::with_capacity(self.particles.len());
        for (k, p) in self.particles.iter().enumerate() {
            let st =
                Stencil::new(&cfg, inv_dx, p.x).ok_or(Mpm2dError::OutOfGrid { particle: k })?;
            stencils.push(st);
        }
        // the Maxwell branch over this step: h' = e h + γ (τ̂' − τ̂_ref), so the stress
        // (1 − g) τ̂' + g h' = (1 − g + g γ) τ̂' + g (e h − γ τ̂_ref)
        let decay = self
            .material
            .maxwell
            .filter(|m| m.fraction > Fix128::ZERO)
            .map(|m| {
                let e = (-(dt / m.relaxation_time)).exp();
                (e, (Fix128::ONE - e) * m.relaxation_time / dt, m)
            });
        // particle to grid without the stress: mass, APIC momentum, load impulse
        let mut mass = vec![Fix128::ZERO; nx * ny];
        let mut mom = vec![[Fix128::ZERO; 2]; nx * ny];
        let mut parts = Vec::with_capacity(self.particles.len());
        for ((p, st), load) in self.particles.iter().zip(&stencils).zip(&force) {
            let mv = [
                p.mass * p.v[0] + dt * load[0],
                p.mass * p.v[1] + dt * load[1],
            ];
            let mut nodes = [(0usize, Fix128::ZERO, [Fix128::ZERO; 2]); 9];
            for a in 0..3 {
                for b in 0..3 {
                    let w = st.w[0][a] * st.w[1][b];
                    let d = st.dpos(&cfg, a, b);
                    let idx = (st.base[0] + a) * ny + st.base[1] + b;
                    mass[idx] = mass[idx] + w * p.mass;
                    let mc = [
                        p.mass * (p.c[0][0] * d[0] + p.c[0][1] * d[1]),
                        p.mass * (p.c[1][0] * d[0] + p.c[1][1] * d[1]),
                    ];
                    mom[idx][0] = mom[idx][0] + w * (mv[0] + mc[0]);
                    mom[idx][1] = mom[idx][1] + w * (mv[1] + mc[1]);
                    nodes[3 * a + b] = (idx, w, d);
                }
            }
            let (a0k, konst) = match decay {
                Some((e, gamma, m)) => {
                    let g = m.fraction;
                    let k = Fix128::ONE - g + g * gamma;
                    let r = rotation(p.f);
                    let held: M2 = core::array::from_fn(|i| {
                        core::array::from_fn(|j| {
                            g * (e * p.branch[i][j] - gamma * p.branch_ref[i][j])
                        })
                    });
                    (p.area0 * k, scale(p.area0, mul(mul(r, held), tr(r))))
                }
                None => (p.area0, [[Fix128::ZERO; 2]; 2]),
            };
            parts.push(Part {
                nodes,
                f0: p.f,
                a0k,
                konst,
            });
        }
        let mut vstar = vec![[Fix128::ZERO; 2]; nx * ny];
        for (idx, vs) in vstar.iter_mut().enumerate() {
            if mass[idx] == Fix128::ZERO {
                continue;
            }
            *vs = [
                mom[idx][0] / mass[idx] + dt * cfg.gravity[0],
                mom[idx][1] / mass[idx] + dt * cfg.gravity[1],
            ];
        }
        // the ground: a node at or below it may not move down; start with the nodes the
        // explicit velocity would move down held at 0
        let on_ground: Vec<bool> = (0..nx * ny)
            .map(|idx| {
                let j = idx % ny;
                cfg.ground
                    .is_some_and(|g| cfg.origin[1] + cfg.cell * Fix128::from_int(j as i64) <= g)
                    && mass[idx] != Fix128::ZERO
            })
            .collect();
        let held: Vec<[bool; 2]> = (0..nx * ny)
            .map(|idx| {
                [
                    mass[idx] == Fix128::ZERO,
                    mass[idx] == Fix128::ZERO || (on_ground[idx] && vstar[idx][1] < Fix128::ZERO),
                ]
            })
            .collect();
        let mut sys = System {
            mass,
            vstar,
            held,
            parts,
            dt,
            c4,
            model: self.material.model,
            mu,
            lambda,
        };
        let v = solve_system(&mut sys, &on_ground, solve)?;
        let (v, report) = v;
        // grid to particle, as the explicit step
        let mut next = Vec::with_capacity(self.particles.len());
        for (k, (p, part)) in self.particles.iter().zip(&sys.parts).enumerate() {
            let mut vel = [Fix128::ZERO; 2];
            for &(idx, w, _) in &part.nodes {
                vel[0] = vel[0] + w * v[idx][0];
                vel[1] = vel[1] + w * v[idx][1];
            }
            let c = sys.affine(part, &v);
            let f = sys.deformation(part, c);
            if det(f) <= Fix128::ZERO {
                return Err(Mpm2dError::Inverted { particle: k });
            }
            let (branch, branch_ref) = match decay {
                Some((e, gamma, _)) => {
                    let r = rotation(f);
                    let hat = mul(mul(tr(r), kirchhoff(self.material.model, f, mu, lambda)), r);
                    let mut h = [[Fix128::ZERO; 2]; 2];
                    for i in 0..2 {
                        for j in 0..2 {
                            h[i][j] = e * p.branch[i][j] + gamma * (hat[i][j] - p.branch_ref[i][j]);
                        }
                    }
                    (h, hat)
                }
                None => (p.branch, p.branch_ref),
            };
            next.push(super::MpmParticle2 {
                x: [p.x[0] + dt * vel[0], p.x[1] + dt * vel[1]],
                v: vel,
                c,
                f,
                branch,
                branch_ref,
                ..*p
            });
        }
        self.particles = next;
        Ok(report)
    }
}

/// Newton's method on the system, the ground's contact set updated between solves: a held
/// ground degree of freedom whose gradient pulls it up is released, a free one the solve moves
/// down is held at 0, and the solve runs again until the set does not change.
fn solve_system(
    sys: &mut System,
    on_ground: &[bool],
    solve: &ImplicitSolve,
) -> Result<(Vec<V2>, ImplicitReport), Mpm2dError> {
    let mut report = ImplicitReport::default();
    let mut v = sys.vstar.clone();
    sys.clear_held(&mut v);
    let diag = sys.diagonal();
    let armijo = Fix128::from_ratio(1, 10_000);
    let min_alpha = Fix128::from_ratio(1, 1 << 30);
    loop {
        // Newton on the current contact set
        loop {
            let mut g = sys.gradient(&v).ok_or(Mpm2dError::NotConverged {
                newton: report.newton,
            })?;
            sys.clear_held(&mut g);
            let res = sys.residual(&g);
            if res <= solve.tolerance {
                break;
            }
            if report.newton >= solve.max_newton {
                return Err(Mpm2dError::NotConverged {
                    newton: report.newton,
                });
            }
            report.newton += 1;
            // H d = −g by the preconditioned conjugate gradient
            // the right-hand side scaled by a power of two to the order of 1 (the system is
            // linear, so the solution scales back exactly): near convergence the forces are so
            // small that their products would fall to `Q64.64`'s resolution
            let (up, down) = power_of_two_scale(&g);
            let goal = solve.cg_ratio * res * up / down;
            let mut d;
            let r: Vec<V2> = g
                .iter()
                .map(|x| [-(x[0] * up / down), -(x[1] * up / down)])
                .collect();
            // the exact Hessian first (each particle's stress derivative along the direction);
            // where it is not positive along a direction, the isotropic one
            let fs = sys.deformations(&v);
            let (dd, broke) =
                conjugate_gradient(sys, &diag, &r, goal, Some(&fs), solve.max_cg, &mut report);
            d = if broke {
                conjugate_gradient(sys, &diag, &r, goal, None, solve.max_cg, &mut report).0
            } else {
                dd
            };
            for di in &mut d {
                di[0] = di[0] * down / up;
                di[1] = di[1] * down / up;
            }
            // a direction that does not descend (a direction of zero length is one): no step
            let slope = dot(&g, &d);
            if slope >= Fix128::ZERO {
                return Err(Mpm2dError::NotConverged {
                    newton: report.newton,
                });
            }
            let e0 = sys.energy(&v).ok_or(Mpm2dError::NotConverged {
                newton: report.newton,
            })?;
            let mut alpha = Fix128::ONE;
            loop {
                let trial: Vec<V2> = v
                    .iter()
                    .zip(&d)
                    .map(|(x, y)| [x[0] + alpha * y[0], x[1] + alpha * y[1]])
                    .collect();
                if let Some(e) = sys.energy(&trial) {
                    if e <= e0 + armijo * alpha * slope {
                        v = trial;
                        break;
                    }
                    // near convergence the decrease falls below the rounding of the energy's
                    // sum: the full step is taken when it lowers the residual of the true
                    // gradient
                    if alpha == Fix128::ONE {
                        if let Some(mut gt) = sys.gradient(&trial) {
                            sys.clear_held(&mut gt);
                            if sys.residual(&gt) < res {
                                v = trial;
                                break;
                            }
                        }
                    }
                }
                alpha = alpha / Fix128::from_int(2);
                report.halvings += 1;
                if alpha < min_alpha {
                    return Err(Mpm2dError::NotConverged {
                        newton: report.newton,
                    });
                }
            }
        }
        // the ground's contact set
        let g = sys.gradient(&v).ok_or(Mpm2dError::NotConverged {
            newton: report.newton,
        })?;
        let mut changed = false;
        for idx in 0..v.len() {
            if !on_ground[idx] {
                continue;
            }
            if sys.held[idx][1] && g[idx][1] < Fix128::ZERO {
                // held at 0, but the potential falls as it moves up: let it go
                sys.held[idx][1] = false;
                changed = true;
            } else if !sys.held[idx][1] && v[idx][1] < Fix128::ZERO {
                sys.held[idx][1] = true;
                v[idx][1] = Fix128::ZERO;
                changed = true;
            }
        }
        if !changed {
            return Ok((v, report));
        }
        if report.newton >= solve.max_newton {
            return Err(Mpm2dError::NotConverged {
                newton: report.newton,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `first_pk` is the derivative of `psi`: a central difference of the energy along each
    /// entry of `F` matches `P` to the difference's truncation, for both materials.
    #[test]
    fn the_stress_is_the_derivative_of_the_energy() {
        let f = [
            [Fix128::from_ratio(11, 10), Fix128::from_ratio(1, 20)],
            [Fix128::from_ratio(-1, 30), Fix128::from_ratio(9, 10)],
        ];
        let (mu, lambda) = (Fix128::from_int(1000), Fix128::from_int(1500));
        let h = Fix128::from_ratio(1, 1 << 24);
        for model in [Mpm2dModel::FixedCorotated, Mpm2dModel::NeoHookean] {
            let p = first_pk(model, f, mu, lambda);
            for i in 0..2 {
                for k in 0..2 {
                    let mut up = f;
                    up[i][k] = up[i][k] + h;
                    let mut dn = f;
                    dn[i][k] = dn[i][k] - h;
                    let fd = (psi(model, up, mu, lambda) - psi(model, dn, mu, lambda))
                        / (Fix128::from_int(2) * h);
                    let err = (fd - p[i][k]).abs();
                    assert!(err < Fix128::from_ratio(1, 1000), "{model:?} P[{i}][{k}]");
                }
            }
        }
    }
}
