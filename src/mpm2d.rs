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
//! Status: the public surface and the oracles exist; [`Mpm2d::step`], [`Mpm2d::seed`],
//! [`Mpm2d::elastic_energy`] and [`Mpm2d::state_digest`] fail fast until implemented.

use crate::math::Fix128;

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
        let _ = (sdf, per_axis);
        todo!("STUB: Mpm2d::seed (2D MPM mochi, regular lattice from the SDF)")
    }

    /// Advance one step of [`Mpm2dConfig::dt`]: particle to grid (mass, APIC momentum, MLS
    /// stress force), grid update (gravity, ground), grid to particle (velocity, affine
    /// velocity, position, deformation gradient).
    pub fn step(&mut self) -> Result<(), Mpm2dError> {
        todo!("STUB: Mpm2d::step (MLS-MPM P2G / grid / G2P)")
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
        todo!("STUB: Mpm2d::elastic_energy (the constitutive model's energy density × area0)")
    }

    /// SHA-256 of the configuration, the material and every particle's state, in order
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        todo!("STUB: Mpm2d::state_digest")
    }
}
