//! Source-free Maxwell field solver on a Yee (FDTD) lattice, in normalised units.
//!
//! This is the field solver that [`crate::electromagnetic`] is not:
//! that module applies `F = q(E + v×B)` to a rigid body from a *prescribed*
//! analytic source, and says so in its own header. Here the field itself is a
//! state variable that is marched in time by the curl equations.
//!
//! # Units — normalised, not SI (this is forced, not a preference)
//!
//! `c = ε₀ = μ₀ = 1` and `Δx = Δy = Δz = 1`, so a time step is the Courant
//! number `S = c·Δt/Δx` and nothing else. Lengths are counted in cells and
//! frequencies come out in units of `c/Δx`.
//!
//! ⚠️ **SI units cannot be used here.** [`Fix128`] is Q64.64, so its resolution
//! is the *absolute* constant `2⁻⁶⁴ ≈ 5.421e-20` — there is no exponent. The
//! individual constants survive (`ε₀ = 8.854e-12` keeps 28 significant bits,
//! `μ₀` keeps 45), but their **product does not**: `ε₀·μ₀` is `1/c² ≈
//! 1.113e-17`, which is only 205 ULP, so it retains **8 significant bits** and
//! lands 1.2e-3 away from the true value; a light speed recovered from it as
//! `1/√(ε₀μ₀)` is off by 6.0e-4 (measured 2026-09-29). The failure also scales
//! with the cell size a caller picks — an SI Courant consistency check
//! (`Δt/(ε₀Δx) · Δt/(μ₀Δx)` against `S²`) drifts from 3.4e-9 at `Δx = 1 m` to
//! 2.0e-3 at `Δx = 1 µm` — so it is not something an API can guard against.
//!
//! Callers that want SI convert at the boundary: multiply lengths by `Δx`,
//! times by `Δx/c`, and scale `E`/`H` by the wave impedance `Z₀`.
//!
//! # Lattice
//!
//! Standard Yee staggering on an `nx × ny × nz` cell block. `E` lives on cell
//! edges and `H` on cell faces, which is what makes the discrete `∇·(∇×·)`
//! telescope:
//!
//! ```text
//! Ex at (i+½, j,   k  )   nx   × ny+1 × nz+1
//! Ey at (i,   j+½, k  )   nx+1 × ny   × nz+1
//! Ez at (i,   j,   k+½)   nx+1 × ny+1 × nz
//! Hx at (i,   j+½, k+½)   nx+1 × ny   × nz
//! Hy at (i+½, j,   k+½)   nx   × ny+1 × nz
//! Hz at (i+½, j+½, k  )   nx   × ny   × nz+1
//! ```
//!
//! This is the same staggering idea as [`crate::eulerian_grid::MacGrid`]
//! (face-centred vectors, centre-centred scalar) and shares its index
//! arithmetic; the difference is which components sit on edges versus faces.
//!
//! # Boundary
//!
//! Perfect electric conductor (PEC) on all six walls: the tangential `E` on a
//! wall is held at zero and is never written. A PEC box is a closed resonant
//! cavity, which is what gives the closed-form oracle its modes. Absorbing
//! boundaries (PML) are future work.
//!
//! # Time stepping
//!
//! Leap-frog, `E` on integer steps and `H` on half steps:
//!
//! ```text
//! H^{n+½} = H^{n−½} − S · (∇×E)^n
//! E^{n+1} = E^{n}   + S · (∇×H)^{n+½}
//! ```
//!
//! Eliminating `H` gives `E^{n+1} = 2E^n − E^{n−1} − S²·(∇×∇×E)^n`, which is
//! the form the oracle exploits: the discrete curl-curl operator on a PEC box
//! is diagonalised exactly by the sine modes, so a projected amplitude obeys a
//! scalar three-term recurrence whose coefficient is a closed form.
//!
//! # Stability
//!
//! The 3-D Courant limit is `S ≤ 1/√3 = 0.577350269…`. [`COURANT_3D`] is
//! `9/16 = 0.5625`, the largest dyadic rational under that bound with a
//! denominator up to 2⁸ apart from `73/128` and `147/256`; a dyadic `S` keeps
//! the update coefficient exactly representable in Q64.64.
//!
//! # Determinism
//!
//! Every operation is [`Fix128`] add / sub / mul. No transcendental is reached
//! for, so nothing here can touch a platform `libm`. The lattice is swept in
//! index order, so two runs on different targets produce bit-identical fields.
//!
//! ⚠️ `Fix128` multiplication **truncates**, so it does not distribute over a
//! sum: `(Σf)·S` and `Σ(f·S)` were measured to differ by 1–3 ULP. The textbook
//! claim that a Yee lattice keeps `∇·B` at exactly zero is a statement about
//! continuum arithmetic and **does not survive truncation** — see
//! [`YeeGrid::div_b`], whose contract is a small bounded residual, not zero.
//!
//! # Scope
//!
//! Source free: no charge density, no current density, no material tensors, no
//! absorbing boundary. Gauss's law and charge continuity need `ρ` and `J` and
//! are future work.

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Courant number `9/16 = 0.5625`, the recommended step for a 3-D lattice.
///
/// Under the 3-D limit `1/√3 = 0.577350269…` with 2.6% of margin, and dyadic,
/// so `S` and `S²` are both exact in Q64.64. Raw value `9·2⁶⁰`.
pub const COURANT_3D: Fix128 = Fix128::from_raw(0, 10_376_293_541_461_622_784);

/// Which of the six staggered field components an index refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    /// `Ex`, on edges along x.
    Ex,
    /// `Ey`, on edges along y.
    Ey,
    /// `Ez`, on edges along z.
    Ez,
    /// `Hx`, on faces normal to x.
    Hx,
    /// `Hy`, on faces normal to y.
    Hy,
    /// `Hz`, on faces normal to z.
    Hz,
}

/// A Yee lattice of source-free electromagnetic field, marched by [`YeeGrid::step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YeeGrid {
    nx: usize,
    ny: usize,
    nz: usize,
    courant: Fix128,
    ex: Vec<Fix128>,
    ey: Vec<Fix128>,
    ez: Vec<Fix128>,
    hx: Vec<Fix128>,
    hy: Vec<Fix128>,
    hz: Vec<Fix128>,
}

impl YeeGrid {
    /// Allocate an `nx × ny × nz` cell lattice with all six components zero.
    ///
    /// `courant` is `S = c·Δt/Δx`; see [`COURANT_3D`] and [`cfl_limit_3d`].
    /// The constructor does not reject an unstable `S`, because a caller may
    /// legitimately want to observe the instability — [`cfl_limit_3d`] is the
    /// value to compare against.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, courant: Fix128) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "lattice must have a cell in each axis"
        );
        Self {
            nx,
            ny,
            nz,
            courant,
            ex: vec![Fix128::ZERO; nx * (ny + 1) * (nz + 1)],
            ey: vec![Fix128::ZERO; (nx + 1) * ny * (nz + 1)],
            ez: vec![Fix128::ZERO; (nx + 1) * (ny + 1) * nz],
            hx: vec![Fix128::ZERO; (nx + 1) * ny * nz],
            hy: vec![Fix128::ZERO; nx * (ny + 1) * nz],
            hz: vec![Fix128::ZERO; nx * ny * (nz + 1)],
        }
    }

    /// Cell counts along each axis.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// The Courant number this lattice steps with.
    #[must_use]
    pub const fn courant(&self) -> Fix128 {
        self.courant
    }

    /// Sample count of `component` along each axis (see the module header).
    #[must_use]
    pub const fn component_dims(&self, component: Component) -> (usize, usize, usize) {
        match component {
            Component::Ex => (self.nx, self.ny + 1, self.nz + 1),
            Component::Ey => (self.nx + 1, self.ny, self.nz + 1),
            Component::Ez => (self.nx + 1, self.ny + 1, self.nz),
            Component::Hx => (self.nx + 1, self.ny, self.nz),
            Component::Hy => (self.nx, self.ny + 1, self.nz),
            Component::Hz => (self.nx, self.ny, self.nz + 1),
        }
    }

    #[inline]
    fn offset(&self, component: Component, i: usize, j: usize, k: usize) -> usize {
        let (ni, nj, nk) = self.component_dims(component);
        assert!(i < ni && j < nj && k < nk, "field index out of range");
        (i * nj + j) * nk + k
    }

    #[inline]
    const fn slot(&self, component: Component) -> &Vec<Fix128> {
        match component {
            Component::Ex => &self.ex,
            Component::Ey => &self.ey,
            Component::Ez => &self.ez,
            Component::Hx => &self.hx,
            Component::Hy => &self.hy,
            Component::Hz => &self.hz,
        }
    }

    #[inline]
    fn slot_mut(&mut self, component: Component) -> &mut Vec<Fix128> {
        match component {
            Component::Ex => &mut self.ex,
            Component::Ey => &mut self.ey,
            Component::Ez => &mut self.ez,
            Component::Hx => &mut self.hx,
            Component::Hy => &mut self.hy,
            Component::Hz => &mut self.hz,
        }
    }

    /// Read one field sample.
    ///
    /// # Panics
    ///
    /// Panics if the index is outside [`YeeGrid::component_dims`].
    #[must_use]
    pub fn get(&self, component: Component, i: usize, j: usize, k: usize) -> Fix128 {
        self.slot(component)[self.offset(component, i, j, k)]
    }

    /// Write one field sample, for setting up an initial condition.
    ///
    /// # Panics
    ///
    /// Panics if the index is outside [`YeeGrid::component_dims`].
    pub fn set(&mut self, component: Component, i: usize, j: usize, k: usize, value: Fix128) {
        let at = self.offset(component, i, j, k);
        self.slot_mut(component)[at] = value;
    }

    /// Advance the field by one time step (`H` half step, then `E` full step).
    ///
    /// The lattice is swept in index order, so the result is bit-identical on
    /// every target.
    pub fn step(&mut self) {
        let Self {
            nx,
            ny,
            nz,
            courant,
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
        } = self;
        let (nx, ny, nz) = (*nx, *ny, *nz);
        let s = *courant;

        // Strides: index (i, j, k) of a component with sample counts
        // (ni, nj, nk) lives at (i·nj + j)·nk + k.
        let (ex_i, ex_j) = ((ny + 1) * (nz + 1), nz + 1);
        let (ey_i, ey_j) = (ny * (nz + 1), nz + 1);
        let (ez_i, ez_j) = ((ny + 1) * nz, nz);
        let (hx_i, hx_j) = (ny * nz, nz);
        let (hy_i, hy_j) = ((ny + 1) * nz, nz);
        let (hz_i, hz_j) = (ny * (nz + 1), nz + 1);

        // H^{n+1/2} = H^{n-1/2} - S·(∇×E)^n. Every index is in range for the
        // whole H lattice, so there is no boundary case here.
        // (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z
        for i in 0..=nx {
            for j in 0..ny {
                for k in 0..nz {
                    let curl = (ez[i * ez_i + (j + 1) * ez_j + k] - ez[i * ez_i + j * ez_j + k])
                        - (ey[i * ey_i + j * ey_j + (k + 1)] - ey[i * ey_i + j * ey_j + k]);
                    let at = i * hx_i + j * hx_j + k;
                    hx[at] = hx[at] - s * curl;
                }
            }
        }
        // (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x
        for i in 0..nx {
            for j in 0..=ny {
                for k in 0..nz {
                    let curl = (ex[i * ex_i + j * ex_j + (k + 1)] - ex[i * ex_i + j * ex_j + k])
                        - (ez[(i + 1) * ez_i + j * ez_j + k] - ez[i * ez_i + j * ez_j + k]);
                    let at = i * hy_i + j * hy_j + k;
                    hy[at] = hy[at] - s * curl;
                }
            }
        }
        // (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..=nz {
                    let curl = (ey[(i + 1) * ey_i + j * ey_j + k] - ey[i * ey_i + j * ey_j + k])
                        - (ex[i * ex_i + (j + 1) * ex_j + k] - ex[i * ex_i + j * ex_j + k]);
                    let at = i * hz_i + j * hz_j + k;
                    hz[at] = hz[at] - s * curl;
                }
            }
        }

        // E^{n+1} = E^n + S·(∇×H)^{n+1/2}, interior only: the tangential E on a
        // PEC wall is held at zero and is never written.
        // (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z
        for i in 0..nx {
            for j in 1..ny {
                for k in 1..nz {
                    let curl = (hz[i * hz_i + j * hz_j + k] - hz[i * hz_i + (j - 1) * hz_j + k])
                        - (hy[i * hy_i + j * hy_j + k] - hy[i * hy_i + j * hy_j + (k - 1)]);
                    let at = i * ex_i + j * ex_j + k;
                    ex[at] = ex[at] + s * curl;
                }
            }
        }
        // (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
        for i in 1..nx {
            for j in 0..ny {
                for k in 1..nz {
                    let curl = (hx[i * hx_i + j * hx_j + k] - hx[i * hx_i + j * hx_j + (k - 1)])
                        - (hz[i * hz_i + j * hz_j + k] - hz[(i - 1) * hz_i + j * hz_j + k]);
                    let at = i * ey_i + j * ey_j + k;
                    ey[at] = ey[at] + s * curl;
                }
            }
        }
        // (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
        for i in 1..nx {
            for j in 1..ny {
                for k in 0..nz {
                    let curl = (hy[i * hy_i + j * hy_j + k] - hy[(i - 1) * hy_i + j * hy_j + k])
                        - (hx[i * hx_i + j * hx_j + k] - hx[i * hx_i + (j - 1) * hx_j + k]);
                    let at = i * ez_i + j * ez_j + k;
                    ez[at] = ez[at] + s * curl;
                }
            }
        }
    }

    /// Discrete `∇·B` on cell `(i, j, k)`, summed over the cell's six faces.
    ///
    /// ⚠️ The contract is **a small bounded residual, not zero**. On a Yee
    /// lattice `∇·(∇×E)` telescopes to zero in exact arithmetic, but `Fix128`
    /// multiplication truncates and therefore does not distribute over the
    /// face sum (measured 1–3 ULP), so the residual grows slowly with the step
    /// count instead of staying at the zero bit pattern.
    ///
    /// # Panics
    ///
    /// Panics if the cell index is outside the lattice.
    #[must_use]
    pub fn div_b(&self, i: usize, j: usize, k: usize) -> Fix128 {
        assert!(
            i < self.nx && j < self.ny && k < self.nz,
            "cell index out of range"
        );
        (self.get(Component::Hx, i + 1, j, k) - self.get(Component::Hx, i, j, k))
            + (self.get(Component::Hy, i, j + 1, k) - self.get(Component::Hy, i, j, k))
            + (self.get(Component::Hz, i, j, k + 1) - self.get(Component::Hz, i, j, k))
    }

    /// Largest `|∇·B|` over every cell.
    #[must_use]
    pub fn max_abs_div_b(&self) -> Fix128 {
        let mut worst = Fix128::ZERO;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let d = self.div_b(i, j, k).abs();
                    if d > worst {
                        worst = d;
                    }
                }
            }
        }
        worst
    }

    /// Largest `|value|` over every sample of every component.
    ///
    /// Intended as a stability probe: below the Courant limit this stays
    /// bounded, above it grows without bound.
    #[must_use]
    pub fn max_abs_field(&self) -> Fix128 {
        [&self.ex, &self.ey, &self.ez, &self.hx, &self.hy, &self.hz]
            .iter()
            .flat_map(|v| v.iter())
            .fold(Fix128::ZERO, |worst, &v| {
                let a = v.abs();
                if a > worst {
                    a
                } else {
                    worst
                }
            })
    }
}

/// The 3-D Courant limit `1/√3 = 0.577350269…`, as a [`Fix128`].
///
/// A lattice is stable when its Courant number does not exceed this. The 2-D
/// limit (`1/√2`) and the 1-D limit (`1`) are looser, so a solver that has to
/// work in three dimensions uses this one.
#[must_use]
pub fn cfl_limit_3d() -> Fix128 {
    Fix128::ONE / Fix128::from_int(3).sqrt()
}
