//! 2D material point method for soft solids (the mochi): particles carry mass, velocity, an
//! affine velocity and the deformation gradient; a regular grid of nodes carries the momentum
//! and the stress force each step.
//!
//! The method is the moving least squares material point method (MLS-MPM, Hu et al. 2018,
//! "A moving least squares material point method with displacement discontinuity and
//! two-way rigid body coupling", ACM TOG 37(4)) with quadratic B-spline weights and the
//! affine particle-in-cell transfer (APIC, Jiang et al. 2015), in plane strain, in `Q64.64`
//! throughout, so a run is bit for bit the same on every machine.
//!

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A 2D vector `[x, y]`
pub type V2 = [Fix128; 2];
/// A 2×2 matrix, row major: `m[r][c]`
pub type M2 = [[Fix128; 2]; 2];

/// The grid and the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpm2dConfig {
    /// The position of node `(0, 0)`, m
    pub origin: V2,
    /// The node spacing, m
    pub cell: Fix128,
    /// Nodes along x and y
    pub nodes: [usize; 2],
    /// Gravity, m/s²
    pub gravity: V2,
    /// The step, s
    pub dt: Fix128,
    /// A frictionless ground at this height, m: the nodes at or below it cannot move down
    pub ground: Option<Fix128>,
}

/// An isotropic elastic material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpm2dMaterial {
    /// Young's modulus, Pa
    pub youngs: Fix128,
    /// Poisson's ratio, in `[0, 0.5)`
    pub poisson: Fix128,
    /// Density, kg/m³ (per unit thickness: masses are kg per metre of depth)
    pub density: Fix128,
}

/// One material point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmParticle2 {
    /// Position, m
    pub x: V2,
    /// Velocity, m/s
    pub v: V2,
    /// Affine velocity (the APIC matrix), 1/s
    pub c: M2,
    /// Deformation gradient
    pub f: M2,
    /// Mass per metre of depth, kg/m
    pub mass: Fix128,
    /// Area at rest, m²
    pub area0: Fix128,
}

/// Why a world cannot be made or stepped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mpm2dError {
    /// A configuration or material value out of its range (the field named)
    Invalid(&'static str),
    /// A particle's 3×3 stencil leaves the grid
    OutOfGrid {
        /// The particle index
        particle: usize,
    },
    /// A particle's deformation gradient has `det F <= 0`
    Inverted {
        /// The particle index
        particle: usize,
    },
}

/// A 2D MPM world: one material, its particles and the grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mpm2d {
    config: Mpm2dConfig,
    material: Mpm2dMaterial,
    particles: Vec<MpmParticle2>,
}

/// The 3×3 quadratic B-spline stencil of one particle: the lower-left node and the weights
/// along each axis.
struct Stencil {
    base: [usize; 2],
    /// the particle's position in cell units from `base`, in `[0.5, 1.5)`
    frac: V2,
    w: [[Fix128; 3]; 2],
}

impl Stencil {
    fn new(cfg: &Mpm2dConfig, inv_dx: Fix128, x: V2) -> Option<Self> {
        let half = Fix128::from_ratio(1, 2);
        let mut base = [0usize; 2];
        let mut frac = [Fix128::ZERO; 2];
        let mut w = [[Fix128::ZERO; 3]; 2];
        for k in 0..2 {
            let g = (x[k] - cfg.origin[k]) * inv_dx;
            let b = (g - half).hi;
            if b < 0 || (b as u64).checked_add(3)? > cfg.nodes[k] as u64 {
                return None;
            }
            base[k] = b as usize;
            let f = g - Fix128::from_int(b);
            frac[k] = f;
            let a = Fix128::from_ratio(3, 2) - f;
            let m = f - Fix128::ONE;
            let c = f - half;
            w[k] = [half * a * a, Fix128::from_ratio(3, 4) - m * m, half * c * c];
        }
        Some(Self { base, frac, w })
    }

    /// node `(base + (a, b))` minus the particle, m
    fn dpos(&self, cfg: &Mpm2dConfig, a: usize, b: usize) -> V2 {
        [
            (Fix128::from_int(a as i64) - self.frac[0]) * cfg.cell,
            (Fix128::from_int(b as i64) - self.frac[1]) * cfg.cell,
        ]
    }
}

fn mul(a: M2, b: M2) -> M2 {
    let mut m = [[Fix128::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    m
}

fn det(f: M2) -> Fix128 {
    f[0][0] * f[1][1] - f[0][1] * f[1][0]
}

/// The rotation of the polar decomposition `F = R S` of a 2×2 `F` with `det F > 0`:
/// `R = [[c, −s], [s, c]]`, `(c, s) = (F₀₀ + F₁₁, F₁₀ − F₀₁) / |…|`.
fn rotation(f: M2) -> M2 {
    let c = f[0][0] + f[1][1];
    let s = f[1][0] - f[0][1];
    let n = (c * c + s * s).sqrt();
    if n == Fix128::ZERO {
        return identity();
    }
    let (c, s) = (c / n, s / n);
    [[c, -s], [s, c]]
}

/// The Kirchhoff stress `τ = P Fᵀ` of the fixed co-rotated model,
/// `P = 2μ (F − R) + λ (J − 1) J F⁻ᵀ` (`J F⁻ᵀ` is the cofactor matrix of `F`).
fn kirchhoff(f: M2, mu: Fix128, lambda: Fix128) -> M2 {
    let r = rotation(f);
    let j = det(f);
    let cof = [[f[1][1], -f[1][0]], [-f[0][1], f[0][0]]];
    let two_mu = Fix128::from_int(2) * mu;
    let l = lambda * (j - Fix128::ONE);
    let mut p = [[Fix128::ZERO; 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            p[i][k] = two_mu * (f[i][k] - r[i][k]) + l * cof[i][k];
        }
    }
    let ft = [[f[0][0], f[1][0]], [f[0][1], f[1][1]]];
    mul(p, ft)
}

const fn identity() -> M2 {
    [[Fix128::ONE, Fix128::ZERO], [Fix128::ZERO, Fix128::ONE]]
}

impl Mpm2d {
    /// A world with no particle. Refused: a cell or step `<= 0`, fewer than 3 nodes along an
    /// axis, `E <= 0`, `ν` outside `[0, 0.5)`, density `<= 0`.
    pub fn new(config: Mpm2dConfig, material: Mpm2dMaterial) -> Result<Self, Mpm2dError> {
        if config.cell <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("cell"));
        }
        if config.dt <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("dt"));
        }
        if config.nodes[0] < 3 || config.nodes[1] < 3 {
            return Err(Mpm2dError::Invalid("nodes"));
        }
        if material.youngs <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("youngs"));
        }
        if material.poisson < Fix128::ZERO || material.poisson >= Fix128::from_ratio(1, 2) {
            return Err(Mpm2dError::Invalid("poisson"));
        }
        if material.density <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("density"));
        }
        Ok(Self {
            config,
            material,
            particles: Vec::new(),
        })
    }

    /// The configuration
    #[must_use]
    pub const fn config(&self) -> &Mpm2dConfig {
        &self.config
    }

    /// The material
    #[must_use]
    pub const fn material(&self) -> &Mpm2dMaterial {
        &self.material
    }

    /// The particles, in the order they were added
    #[must_use]
    pub fn particles(&self) -> &[MpmParticle2] {
        &self.particles
    }

    /// Add a particle at rest shape (`F = I`, `C = 0`) of area `area0` at `x` moving at `v`;
    /// its mass is `density × area0`. Returns its index.
    pub fn add_particle(&mut self, x: V2, v: V2, area0: Fix128) -> usize {
        self.particles.push(MpmParticle2 {
            x,
            v,
            c: [[Fix128::ZERO; 2]; 2],
            f: identity(),
            mass: self.material.density * area0,
            area0,
        });
        self.particles.len() - 1
    }

    /// Seed `per_axis × per_axis` particles in every cell whose sample points lie inside the
    /// signed distance `sdf` (negative inside), at rest; returns the number added.
    pub fn seed<F: Fn(V2) -> Fix128>(&mut self, sdf: F, per_axis: usize) -> usize {
        if per_axis == 0 {
            return 0;
        }
        let k = Fix128::from_int(per_axis as i64);
        let h = self.config.cell / k;
        let area = h * h;
        let before = self.particles.len();
        for i in 0..self.config.nodes[0] - 1 {
            for j in 0..self.config.nodes[1] - 1 {
                for a in 0..per_axis {
                    for b in 0..per_axis {
                        let x = [
                            self.config.origin[0]
                                + self.config.cell * Fix128::from_int(i as i64)
                                + h * Fix128::from_int(a as i64)
                                + h / Fix128::from_int(2),
                            self.config.origin[1]
                                + self.config.cell * Fix128::from_int(j as i64)
                                + h * Fix128::from_int(b as i64)
                                + h / Fix128::from_int(2),
                        ];
                        if sdf(x) < Fix128::ZERO {
                            self.add_particle(x, [Fix128::ZERO; 2], area);
                        }
                    }
                }
            }
        }
        self.particles.len() - before
    }

    /// Advance one step of [`Mpm2dConfig::dt`]: particle to grid (mass, APIC momentum, MLS
    /// stress force), grid update (gravity, ground), grid to particle (velocity, affine
    /// velocity, position, deformation gradient).
    pub fn step(&mut self) -> Result<(), Mpm2dError> {
        let cfg = self.config;
        let (nx, ny) = (cfg.nodes[0], cfg.nodes[1]);
        let inv_dx = Fix128::ONE / cfg.cell;
        let four_inv_dx2 = Fix128::from_int(4) * inv_dx * inv_dx;
        let (mu, lambda) = self.lame();
        let dt = cfg.dt;
        // every particle's stencil first: a step either runs whole or not at all
        let mut stencils = Vec::with_capacity(self.particles.len());
        for (k, p) in self.particles.iter().enumerate() {
            let st =
                Stencil::new(&cfg, inv_dx, p.x).ok_or(Mpm2dError::OutOfGrid { particle: k })?;
            stencils.push(st);
        }
        // particle to grid: mass, APIC momentum and the MLS stress force
        let mut gm = vec![Fix128::ZERO; nx * ny];
        let mut gv = vec![[Fix128::ZERO; 2]; nx * ny];
        for (p, st) in self.particles.iter().zip(&stencils) {
            let tau = kirchhoff(p.f, mu, lambda);
            // affine = −dt · area0 · 4/Δx² · τ + m C
            let s = -(dt * p.area0 * four_inv_dx2);
            let aff = [
                [
                    s * tau[0][0] + p.mass * p.c[0][0],
                    s * tau[0][1] + p.mass * p.c[0][1],
                ],
                [
                    s * tau[1][0] + p.mass * p.c[1][0],
                    s * tau[1][1] + p.mass * p.c[1][1],
                ],
            ];
            let mv = [p.mass * p.v[0], p.mass * p.v[1]];
            for a in 0..3 {
                for b in 0..3 {
                    let w = st.w[0][a] * st.w[1][b];
                    let d = st.dpos(&cfg, a, b);
                    let idx = (st.base[0] + a) * ny + st.base[1] + b;
                    gm[idx] = gm[idx] + w * p.mass;
                    gv[idx][0] = gv[idx][0] + w * (mv[0] + aff[0][0] * d[0] + aff[0][1] * d[1]);
                    gv[idx][1] = gv[idx][1] + w * (mv[1] + aff[1][0] * d[0] + aff[1][1] * d[1]);
                }
            }
        }
        // grid: momentum to velocity, gravity, the ground
        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                if gm[idx] == Fix128::ZERO {
                    continue;
                }
                let mut v = [gv[idx][0] / gm[idx], gv[idx][1] / gm[idx]];
                v[0] = v[0] + dt * cfg.gravity[0];
                v[1] = v[1] + dt * cfg.gravity[1];
                if let Some(g) = cfg.ground {
                    let y = cfg.origin[1] + cfg.cell * Fix128::from_int(j as i64);
                    if y <= g && v[1] < Fix128::ZERO {
                        v[1] = Fix128::ZERO;
                    }
                }
                gv[idx] = v;
            }
        }
        // grid to particle: velocity, affine velocity, position, deformation gradient
        for (k, (p, st)) in self.particles.iter_mut().zip(&stencils).enumerate() {
            let mut v = [Fix128::ZERO; 2];
            let mut b = [[Fix128::ZERO; 2]; 2];
            for a in 0..3 {
                for c in 0..3 {
                    let w = st.w[0][a] * st.w[1][c];
                    let d = st.dpos(&cfg, a, c);
                    let gvi = gv[(st.base[0] + a) * ny + st.base[1] + c];
                    v[0] = v[0] + w * gvi[0];
                    v[1] = v[1] + w * gvi[1];
                    for r in 0..2 {
                        for q in 0..2 {
                            b[r][q] = b[r][q] + w * gvi[r] * d[q];
                        }
                    }
                }
            }
            let c = [
                [four_inv_dx2 * b[0][0], four_inv_dx2 * b[0][1]],
                [four_inv_dx2 * b[1][0], four_inv_dx2 * b[1][1]],
            ];
            // F ← (I + dt C) F
            let g = [
                [Fix128::ONE + dt * c[0][0], dt * c[0][1]],
                [dt * c[1][0], Fix128::ONE + dt * c[1][1]],
            ];
            let f = mul(g, p.f);
            if det(f) <= Fix128::ZERO {
                return Err(Mpm2dError::Inverted { particle: k });
            }
            p.v = v;
            p.c = c;
            p.x = [p.x[0] + dt * v[0], p.x[1] + dt * v[1]];
            p.f = f;
        }
        Ok(())
    }

    /// The Lamé parameters `(μ, λ)` of the material (plane strain: `λ = E ν / ((1 + ν)(1 − 2ν))`)
    fn lame(&self) -> (Fix128, Fix128) {
        let (e, nu) = (self.material.youngs, self.material.poisson);
        let mu = e / (Fix128::from_int(2) * (Fix128::ONE + nu));
        let lambda = e * nu / ((Fix128::ONE + nu) * (Fix128::ONE - Fix128::from_int(2) * nu));
        (mu, lambda)
    }

    /// Total mass, kg/m
    #[must_use]
    pub fn mass(&self) -> Fix128 {
        self.particles.iter().fold(Fix128::ZERO, |a, p| a + p.mass)
    }

    /// Total linear momentum, kg/s per metre of depth
    #[must_use]
    pub fn momentum(&self) -> V2 {
        self.particles.iter().fold([Fix128::ZERO; 2], |a, p| {
            [a[0] + p.mass * p.v[0], a[1] + p.mass * p.v[1]]
        })
    }

    /// Total angular momentum about the origin, kg·m²/s per metre of depth, including the
    /// affine part of APIC: `m (x × v) + m D (C₁₀ − C₀₁)` with `D = cell² / 4` for quadratic
    /// B-splines (the momentum the particle hands to the grid).
    #[must_use]
    pub fn angular_momentum(&self) -> Fix128 {
        let d = self.config.cell * self.config.cell / Fix128::from_int(4);
        self.particles.iter().fold(Fix128::ZERO, |a, p| {
            a + p.mass * (p.x[0] * p.v[1] - p.x[1] * p.v[0]) + p.mass * d * (p.c[1][0] - p.c[0][1])
        })
    }

    /// Kinetic energy, J per metre of depth
    #[must_use]
    pub fn kinetic_energy(&self) -> Fix128 {
        self.particles.iter().fold(Fix128::ZERO, |a, p| {
            a + p.mass * (p.v[0] * p.v[0] + p.v[1] * p.v[1]) / Fix128::from_int(2)
        })
    }

    /// Stored elastic energy, J per metre of depth
    #[must_use]
    pub fn elastic_energy(&self) -> Fix128 {
        let (mu, lambda) = self.lame();
        self.particles.iter().fold(Fix128::ZERO, |acc, p| {
            let r = rotation(p.f);
            let mut dev = Fix128::ZERO;
            for (fr, rr) in p.f.iter().zip(&r) {
                for (&a, &b) in fr.iter().zip(rr) {
                    let d = a - b;
                    dev = dev + d * d;
                }
            }
            let j1 = det(p.f) - Fix128::ONE;
            acc + p.area0 * (mu * dev + lambda * j1 * j1 / Fix128::from_int(2))
        })
    }

    /// SHA-256 of the configuration, the material and every particle's state, in order
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        let mut put = |x: Fix128| {
            h.update(x.hi.to_le_bytes());
            h.update(x.lo.to_le_bytes());
        };
        let c = &self.config;
        put(c.origin[0]);
        put(c.origin[1]);
        put(c.cell);
        put(Fix128::from_int(c.nodes[0] as i64));
        put(Fix128::from_int(c.nodes[1] as i64));
        put(c.gravity[0]);
        put(c.gravity[1]);
        put(c.dt);
        put(c.ground.unwrap_or(Fix128::ZERO));
        put(Fix128::from_int(i64::from(c.ground.is_some())));
        put(self.material.youngs);
        put(self.material.poisson);
        put(self.material.density);
        put(Fix128::from_int(self.particles.len() as i64));
        for p in &self.particles {
            for x in [p.x[0], p.x[1], p.v[0], p.v[1], p.mass, p.area0] {
                put(x);
            }
            for m in [p.c, p.f] {
                for row in m {
                    for x in row {
                        put(x);
                    }
                }
            }
        }
        h.finalize().into()
    }
}
