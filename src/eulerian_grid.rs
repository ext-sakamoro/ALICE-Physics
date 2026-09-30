//! Eulerian Fluid Grid (MAC + FLIP/PIC)
//!
//! Phase F4 of the ALICE-Physics completeness project. Provides a
//! **staggered marker-and-cell (MAC) grid** for grid-based fluid solvers,
//! plus the classical **FLIP** (Fluid-Implicit Particle) and **PIC**
//! (Particle-In-Cell) particle ↔ grid transfer operators.
//!
//! # MAC layout
//!
//! Velocity components live on the corresponding cell faces:
//! - `u[i,j,k]` on the X-face between cells `i-1` and `i` → shape
//!   `(nx+1) × ny × nz`
//! - `v[i,j,k]` on the Y-face → `nx × (ny+1) × nz`
//! - `w[i,j,k]` on the Z-face → `nx × ny × (nz+1)`
//! - Pressure `p[i,j,k]` on the cell centre → `nx × ny × nz`
//!
//! This staggering makes divergence and gradient consistent (avoids the
//! famous "checkerboard" pressure oscillation of collocated grids).
//!
//! # Transfer operators
//!
//! - **P2G** scatters particle velocities to the grid using trilinear
//!   weights. Additive accumulation, divide by weight-sum at the end.
//! - **G2P** samples grid velocities at particle positions via trilinear
//!   interpolation.
//! - **FLIP** update: `v_p ← v_p + G2P(u_new − u_old)`. Momentum-preserving.
//! - **PIC** update: `v_p ← G2P(u_new)`. More damped, less noise.
//!
//! # Pressure projection
//!
//! Solves `∇²p = ρ/dt · ∇·u*` via Jacobi iterations (simple, robust).
//! After the pressure is found, subtract `dt/ρ · ∇p` from the intermediate
//! velocities to project onto the divergence-free space.
//!
//! # References
//!
//! - Harlow & Welch, "Numerical calculation of time-dependent viscous
//!   incompressible flow of fluid with free surface", Phys. Fluids 8, 1965
//!   (original MAC).
//! - Brackbill & Ruppel, "FLIP: A method for adaptively zoned, particle-in-
//!   cell calculations of fluid flows in two dimensions", J. Comp. Phys. 65,
//!   1986.
//! - Zhu & Bridson, "Animating sand as a fluid", ACM Trans. Graph. 24, 2005.
//!
//! # Integration status
//!
//! Only the red-black Gauss-Seidel projection (`project_pressure`),
//! trilinear sampling helpers (`sample_u/v/w_range/trilinear`), and
//! `g2p_velocity` are currently wired into `cfd_solver.rs`. The
//! Jacobi + BiCGStab pressure variants and P2G scatter operators
//! are reserved crate-internal API awaiting downstream integration.
//!
//! # Face mask — walls inside the projection
//!
//! `u_solid` / `v_solid` / `w_solid` mark which faces are walls
//! ([`MacGrid::set_closed_box_walls`] does the six boundary layers of a
//! sealed box). The mask is part of the Poisson problem, not a post-pass:
//!
//! - a **solid** face carries no flux, so it leaves both the off-diagonal
//!   coupling and the diagonal count — the homogeneous Neumann condition
//!   `∂p/∂n = 0` — and its normal velocity is held at zero;
//! - a face that is **not** solid always counts toward the diagonal; on the
//!   domain boundary its neighbour is the exterior `p = 0` (open / free
//!   surface), which is the behaviour of a grid with no mask set.
//!
//! Two consequences that the old fixed `1/6` divisor got wrong:
//!
//! - the diagonal is the number of open faces, so a slab with walled `z`
//!   sides is a genuine 2-D Poisson problem at any `nz` instead of a screened
//!   one that barely moves the field;
//! - the velocity correction sweeps every face including the boundary layer,
//!   so the rim cells have the degree of freedom they need and their
//!   divergence is removed with the rest.
//!
//! # Flow boundaries — inflow and outflow
//!
//! [`FaceBc`] names what a face is, and [`MacGrid::set_u_bc`] and friends set
//! it. The four conditions differ in exactly two places, the pressure stencil
//! and [`MacGrid::enforce_face_boundaries`]:
//!
//! | condition | pressure | normal velocity |
//! |---|---|---|
//! | [`FaceBc::Fluid`] | interior coupling, exterior `p = 0` on the rim | free |
//! | [`FaceBc::Wall`] | homogeneous Neumann (face dropped) | held at 0, no-slip ghost for the viscous term |
//! | [`FaceBc::SlipWall`] | homogeneous Neumann (face dropped) | held at 0, zero-gradient tangential ghost |
//! | [`FaceBc::Inflow`] | homogeneous Neumann (face dropped) | held at the prescribed value |
//! | [`FaceBc::Outflow`] | exterior `p = 0`, i.e. Dirichlet | zero-gradient extrapolation from the interior |
//!
//! A prescribed normal velocity is a velocity Dirichlet condition, and a
//! Dirichlet velocity is a Neumann pressure: the projection has no freedom
//! left on that face, so it must drop out of the stencil exactly like a wall.
//! The flux the inflow pushes in has to leave somewhere, and the only faces
//! that can carry it are the ones whose pressure is Dirichlet — the outflow
//! and plain fluid rim faces. **A domain with an inflow and no such face has
//! no solution**: the pure-Neumann Poisson problem is only solvable when the
//! net boundary flux is zero, and the relaxation will drift instead of
//! converging.

// Reserved algorithm variants (Jacobi / BiCGStab pressure, P2G scatter) are
// pub(crate) but currently unused outside their own unit tests — awaiting
// cfd_solver integration.
#![allow(dead_code)]

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::BTreeMap;

// ============================================================================
// Face boundary conditions
// ============================================================================

/// What a single MAC face is: interior fluid, a wall, or a place where the
/// flow enters or leaves the domain.
///
/// See the module header for the table of what each one does to the pressure
/// stencil and to the normal velocity. Set with [`MacGrid::set_u_bc`] /
/// [`MacGrid::set_v_bc`] / [`MacGrid::set_w_bc`], read back with
/// [`MacGrid::u_bc`] / [`MacGrid::v_bc`] / [`MacGrid::w_bc`].
///
/// [`FaceBc::Inflow`] and [`FaceBc::Outflow`] are only meaningful on the
/// domain boundary; on an interior face the neighbouring cell exists and the
/// "exterior" the condition refers to does not.
///
/// The enum is `#[non_exhaustive]` — a convective outflow and a periodic pair
/// are the obvious next entries — so a downstream `match` needs a wildcard
/// arm. Building the variants is unaffected: the variants themselves are
/// exhaustive, and `tests/analytic_cfd_flow_bc.rs` constructs every one of
/// them from outside the crate so that stays true.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FaceBc {
    /// Ordinary fluid face. On the domain boundary this is the open
    /// condition: the exterior pressure is `p = 0` and nothing constrains
    /// the velocity. This is what every face of a fresh [`MacGrid`] is.
    #[default]
    Fluid,
    /// No-slip wall moving at `velocity` (the zero vector for a wall at
    /// rest; the lid of a lid-driven cavity is a wall with a tangential
    /// velocity). No flux through the face, and the viscous term mirrors
    /// the tangential components with the no-slip ghost `2 u_wall − u_in`
    /// instead of a zero gradient.
    Wall {
        /// Velocity of the wall itself. The component normal to the face is
        /// ignored — a wall that moved into the fluid would not be a wall.
        velocity: Vec3Fix,
    },
    /// Free-slip wall / symmetry plane: no flux, but no tangential shear
    /// either, so the viscous term keeps the zero-gradient mirror.
    ///
    /// This is what the `z` faces of a quasi-2-D run want. Marking them
    /// [`FaceBc::Wall`] instead makes the sheet no-slip against two plates
    /// one cell apart, which damps the in-plane flow by `4 ν dt / dx²` per
    /// step rather than modelling a two-dimensional problem.
    SlipWall,
    /// Prescribed normal velocity (velocity Dirichlet). The face carries no
    /// pressure degree of freedom — homogeneous Neumann, exactly like a wall
    /// — but its normal velocity is held at `normal_velocity` instead of at
    /// zero, every step, after advection and diffusion have had their say.
    Inflow {
        /// Signed velocity along the face normal (`+x` for an X-face), so a
        /// positive value on the `i = 0` layer pushes fluid into the domain
        /// and a positive value on `i = nx` pulls it out.
        normal_velocity: Fix128,
    },
    /// Outflow: the exterior pressure is the Dirichlet `p = 0` and the normal
    /// velocity is extrapolated with zero gradient from the interior face
    /// next to it before the projection runs.
    ///
    /// ⚠️ On the domain boundary the *pressure* side of this is what
    /// [`FaceBc::Fluid`] already did, so **at a converged pressure solve an
    /// outflow face and a plain open rim face agree**; the zero-gradient
    /// extrapolation is a better starting iterate, not a different condition.
    /// Measured on a 12×8 duct: the two fields differ by `4.8e-5` with 200
    /// Gauss-Seidel sweeps and by `1.3e-2` with one, and at one sweep the
    /// extrapolated run is the one closer to closing its mass balance
    /// (`tests/analytic_cfd_flow_bc.rs`). The variant is here because the
    /// caller's intent — *this* is where the flow leaves — cannot be read off
    /// a face that merely was not marked, and because a convective outflow
    /// has somewhere to go.
    Outflow,
}

impl FaceBc {
    /// Does the face block flow (either kind of wall)?
    #[must_use]
    pub fn is_wall(self) -> bool {
        matches!(self, Self::Wall { .. } | Self::SlipWall)
    }

    /// Is the normal velocity of this face prescribed by the caller, rather
    /// than held at zero (a wall) or solved for (fluid, outflow)?
    #[must_use]
    pub fn is_inflow(self) -> bool {
        matches!(self, Self::Inflow { .. })
    }

    /// Does the face carry no pressure degree of freedom?
    ///
    /// True for both walls and for [`FaceBc::Inflow`]: all three fix the
    /// normal velocity, so the Poisson problem must drop the face from both
    /// the off-diagonal coupling and the diagonal count.
    ///
    /// This is the predicate the Poisson stencil is built from, not a
    /// restatement of it — the mask and the velocity correction both route
    /// through here so that the documented rule and the solved problem cannot
    /// drift. They did once: while this method merely described the rule and
    /// the mask spelled it out again, dropping [`FaceBc::Inflow`] from here
    /// changed nothing a duct could measure.
    #[must_use]
    pub fn blocks_pressure(self) -> bool {
        self.is_wall() || self.is_inflow()
    }

    /// Velocity of the no-slip wall this face is, if it is one.
    ///
    /// `None` for [`FaceBc::SlipWall`] — a symmetry plane has no velocity to
    /// impose — and for everything that is not a wall.
    #[must_use]
    pub fn no_slip_velocity(self) -> Option<Vec3Fix> {
        match self {
            Self::Wall { velocity } => Some(velocity),
            _ => None,
        }
    }
}

/// Read a face condition out of the sparse map, falling back to the dense
/// solid flag and then to [`FaceBc::Fluid`].
fn read_face_bc(map: &BTreeMap<usize, FaceBc>, solid: &[bool], ix: usize) -> FaceBc {
    if let Some(bc) = map.get(&ix) {
        return *bc;
    }
    if solid[ix] {
        FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        }
    } else {
        FaceBc::Fluid
    }
}

/// Store a face condition, keeping the map to the faces that are *not* plain
/// fluid and not a wall at rest — those two are what the dense solid flag
/// already says, and leaving them out keeps the map empty for the common
/// closed box.
fn write_face_bc(map: &mut BTreeMap<usize, FaceBc>, ix: usize, bc: FaceBc) {
    let default_wall = FaceBc::Wall {
        velocity: Vec3Fix::ZERO,
    };
    if bc == FaceBc::Fluid || bc == default_wall {
        map.remove(&ix);
    } else {
        map.insert(ix, bc);
    }
}

/// The condition `set_u_solid` and friends stand for.
fn wall_or_fluid(solid: bool) -> FaceBc {
    if solid {
        FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        }
    } else {
        FaceBc::Fluid
    }
}

/// The wall separating two same-component faces, if *every* face in between
/// is a no-slip wall.
///
/// `None` entries are faces that do not exist (the pair straddles the domain
/// edge), which do not veto. Anything that is not a [`FaceBc::Wall`] does
/// veto: a half-open gap is not a wall, and a [`FaceBc::SlipWall`] is asking
/// for the zero-gradient mirror rather than a no-slip ghost.
fn wall_between(a: Option<FaceBc>, b: Option<FaceBc>) -> Option<Vec3Fix> {
    let side = |bc: Option<FaceBc>| match bc {
        None => Some(None),
        Some(FaceBc::Wall { velocity }) => Some(Some(velocity)),
        Some(_) => None,
    };
    match (side(a)?, side(b)?) {
        (Some(p), Some(q)) => Some(Vec3Fix::new(
            (p.x + q.x).half(),
            (p.y + q.y).half(),
            (p.z + q.z).half(),
        )),
        (Some(p), None) | (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

// ============================================================================
// MAC grid
// ============================================================================

/// Staggered marker-and-cell grid holding face-based velocity components and
/// cell-centred pressure.
#[derive(Clone, Debug)]
pub struct MacGrid {
    /// Cells in X.
    pub nx: usize,
    /// Cells in Y.
    pub ny: usize,
    /// Cells in Z.
    pub nz: usize,
    /// Cell spacing (m).
    pub dx: Fix128,
    /// X-velocity on X-faces `[(nx+1) · ny · nz]`.
    pub u: Vec<Fix128>,
    /// Y-velocity on Y-faces `[nx · (ny+1) · nz]`.
    pub v: Vec<Fix128>,
    /// Z-velocity on Z-faces `[nx · ny · (nz+1)]`.
    pub w: Vec<Fix128>,
    /// Cell-centred pressure `[nx · ny · nz]`.
    pub pressure: Vec<Fix128>,
    /// Solid flag per X-face `[(nx+1) · ny · nz]`; see [`MacGrid::set_u_solid`].
    pub u_solid: Vec<bool>,
    /// Solid flag per Y-face `[nx · (ny+1) · nz]`; see [`MacGrid::set_v_solid`].
    pub v_solid: Vec<bool>,
    /// Solid flag per Z-face `[nx · ny · (nz+1)]`; see [`MacGrid::set_w_solid`].
    pub w_solid: Vec<bool>,
    /// X-faces whose condition is neither plain fluid nor a wall at rest —
    /// a moving wall, a symmetry plane, an inflow or an outflow — keyed by
    /// face index. Empty for a closed box, which the dense `u_solid` flag
    /// already describes. Private so that it cannot drift from `u_solid`:
    /// [`MacGrid::set_u_bc`] writes both.
    u_face_bc: BTreeMap<usize, FaceBc>,
    /// Y-face conditions; see `u_face_bc`.
    v_face_bc: BTreeMap<usize, FaceBc>,
    /// Z-face conditions; see `u_face_bc`.
    w_face_bc: BTreeMap<usize, FaceBc>,
}

impl MacGrid {
    /// Empty grid initialised to zero everywhere, with every face fluid
    /// (no walls). Call [`MacGrid::set_closed_box_walls`] to turn the six
    /// domain-boundary face layers into no-through-flow walls.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Fix128) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            u: vec![Fix128::ZERO; (nx + 1) * ny * nz],
            v: vec![Fix128::ZERO; nx * (ny + 1) * nz],
            w: vec![Fix128::ZERO; nx * ny * (nz + 1)],
            pressure: vec![Fix128::ZERO; nx * ny * nz],
            u_solid: vec![false; (nx + 1) * ny * nz],
            v_solid: vec![false; nx * (ny + 1) * nz],
            w_solid: vec![false; nx * ny * (nz + 1)],
            u_face_bc: BTreeMap::new(),
            v_face_bc: BTreeMap::new(),
            w_face_bc: BTreeMap::new(),
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn idx_u(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }
    #[inline]
    #[must_use]
    pub(crate) fn idx_v(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }
    #[inline]
    #[must_use]
    pub(crate) fn idx_w(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }
    #[inline]
    #[must_use]
    fn idx_c(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Read `u` at face `(i, j, k)`. Out of range returns 0.
    #[must_use]
    pub fn u(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.u[self.idx_u(i, j, k)]
    }
    /// Read `v`.
    #[must_use]
    pub fn v(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.v[self.idx_v(i, j, k)]
    }
    /// Read `w`.
    #[must_use]
    pub fn w(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return Fix128::ZERO;
        }
        self.w[self.idx_w(i, j, k)]
    }
    /// Read pressure.
    #[must_use]
    pub fn pressure(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return Fix128::ZERO;
        }
        self.pressure[self.idx_c(i, j, k)]
    }

    /// Cell-centred velocity by averaging adjacent faces (used for output /
    /// downstream Lagrangian modules).
    #[must_use]
    pub fn cell_velocity(&self, i: usize, j: usize, k: usize) -> (Fix128, Fix128, Fix128) {
        let u_c = (self.u(i, j, k) + self.u(i + 1, j, k)).half();
        let v_c = (self.v(i, j, k) + self.v(i, j + 1, k)).half();
        let w_c = (self.w(i, j, k) + self.w(i, j, k + 1)).half();
        (u_c, v_c, w_c)
    }

    /// Is the X-face `(i, j, k)` a wall? Out of range returns `false`
    /// (there is no such face).
    #[must_use]
    pub fn is_u_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return false;
        }
        self.u_solid[self.idx_u(i, j, k)]
    }
    /// Is the Y-face `(i, j, k)` a wall?
    #[must_use]
    pub fn is_v_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return false;
        }
        self.v_solid[self.idx_v(i, j, k)]
    }
    /// Is the Z-face `(i, j, k)` a wall?
    #[must_use]
    pub fn is_w_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return false;
        }
        self.w_solid[self.idx_w(i, j, k)]
    }

    /// Mark the X-face `(i, j, k)` as a wall (`true`) or as fluid (`false`).
    ///
    /// A wall face carries no flux: the pressure projection holds its normal
    /// velocity at zero and drops it from the Poisson stencil, which is the
    /// homogeneous Neumann condition `∂p/∂n = 0` on that face. A face left
    /// `false` on the domain boundary keeps the open condition (exterior
    /// pressure `p = 0`). Out-of-range indices are ignored.
    ///
    /// Shorthand for [`MacGrid::set_u_bc`] with `FaceBc::Wall { velocity:
    /// Vec3Fix::ZERO }` / [`FaceBc::Fluid`], so it also clears any inflow,
    /// outflow or wall velocity previously set on that face.
    pub fn set_u_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        self.set_u_bc(i, j, k, wall_or_fluid(solid));
    }
    /// Mark the Y-face `(i, j, k)` as a wall; see [`MacGrid::set_u_solid`].
    pub fn set_v_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        self.set_v_bc(i, j, k, wall_or_fluid(solid));
    }
    /// Mark the Z-face `(i, j, k)` as a wall; see [`MacGrid::set_u_solid`].
    pub fn set_w_solid(&mut self, i: usize, j: usize, k: usize, solid: bool) {
        self.set_w_bc(i, j, k, wall_or_fluid(solid));
    }

    /// Boundary condition of the X-face `(i, j, k)`. Out of range returns
    /// [`FaceBc::Fluid`] (there is no such face).
    #[must_use]
    pub fn u_bc(&self, i: usize, j: usize, k: usize) -> FaceBc {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return FaceBc::Fluid;
        }
        read_face_bc(&self.u_face_bc, &self.u_solid, self.idx_u(i, j, k))
    }
    /// Boundary condition of the Y-face `(i, j, k)`; see [`MacGrid::u_bc`].
    #[must_use]
    pub fn v_bc(&self, i: usize, j: usize, k: usize) -> FaceBc {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return FaceBc::Fluid;
        }
        read_face_bc(&self.v_face_bc, &self.v_solid, self.idx_v(i, j, k))
    }
    /// Boundary condition of the Z-face `(i, j, k)`; see [`MacGrid::u_bc`].
    #[must_use]
    pub fn w_bc(&self, i: usize, j: usize, k: usize) -> FaceBc {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return FaceBc::Fluid;
        }
        read_face_bc(&self.w_face_bc, &self.w_solid, self.idx_w(i, j, k))
    }

    /// Set the boundary condition of the X-face `(i, j, k)`.
    ///
    /// Out-of-range indices are ignored. See [`FaceBc`] and the module
    /// header for what each condition does; the prescribed velocity of an
    /// [`FaceBc::Inflow`] is reimposed by
    /// [`MacGrid::enforce_face_boundaries`], which the projection calls.
    pub fn set_u_bc(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        if i > self.nx || j >= self.ny || k >= self.nz {
            return;
        }
        let ix = self.idx_u(i, j, k);
        self.u_solid[ix] = bc.is_wall();
        write_face_bc(&mut self.u_face_bc, ix, bc);
    }
    /// Set the boundary condition of the Y-face `(i, j, k)`; see
    /// [`MacGrid::set_u_bc`].
    pub fn set_v_bc(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        if i >= self.nx || j > self.ny || k >= self.nz {
            return;
        }
        let ix = self.idx_v(i, j, k);
        self.v_solid[ix] = bc.is_wall();
        write_face_bc(&mut self.v_face_bc, ix, bc);
    }
    /// Set the boundary condition of the Z-face `(i, j, k)`; see
    /// [`MacGrid::set_u_bc`].
    pub fn set_w_bc(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        if i >= self.nx || j >= self.ny || k > self.nz {
            return;
        }
        let ix = self.idx_w(i, j, k);
        self.w_solid[ix] = bc.is_wall();
        write_face_bc(&mut self.w_face_bc, ix, bc);
    }

    /// Turn the six domain-boundary face layers into walls — the closed box
    /// used by the lid-driven cavity and by any sealed container.
    ///
    /// Marks `u` at `i = 0, nx`, `v` at `j = 0, ny` and `w` at `k = 0, nz`.
    /// Interior faces are left untouched, so this composes with obstacle
    /// masks set through [`MacGrid::set_u_solid`] and friends.
    ///
    /// # The walls are no-slip
    ///
    /// ⚠️ **Changed behaviour.** These are [`FaceBc::Wall`], so the viscous
    /// term of [`crate::cfd_solver::CfdSolver::step`] mirrors them with the
    /// no-slip ghost. Before the no-slip ghost existed they behaved as
    /// free-slip, which for a sealed container of viscous fluid was the wrong
    /// condition, not a milder one. A caller that genuinely wants free slip
    /// has to say so, by overwriting the layer with [`FaceBc::SlipWall`]
    /// through [`MacGrid::set_u_bc`] and friends afterwards.
    ///
    /// # An axis one cell thick becomes a symmetry plane
    ///
    /// ⚠️ **An axis with a single cell gets [`FaceBc::SlipWall`] instead**,
    /// because a box one cell thick cannot resolve a boundary layer: no-slip
    /// on both of its faces makes every ghost of the in-plane components
    /// `−u_in`, which damps the flow by `4 ν dt / dx²` per step and leaves a
    /// field that looks like a broken solver rather than like a thin box.
    /// Measured on the `16 × 16 × 1` lid-driven cavity of
    /// `tests/analytic_cfd_wall_bc.rs`: the centreline peak falls from 86.0 %
    /// of the Ghia (1982) reference to **4.2 %**.
    ///
    /// A single cell along an axis always means "this problem is
    /// two-dimensional", so the symmetry plane is what the caller meant. The
    /// faces are still walls for the pressure — they have to be, or the
    /// Poisson problem degenerates into a screened one — they just exert no
    /// shear. A caller who really wants a no-slip sheet one cell thick can
    /// set [`FaceBc::Wall`] on that layer explicitly.
    pub fn set_closed_box_walls(&mut self) {
        let wall = FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        };
        let along = |n: usize| if n == 1 { FaceBc::SlipWall } else { wall };
        let (bc_x, bc_y, bc_z) = (along(self.nx), along(self.ny), along(self.nz));
        for k in 0..self.nz {
            for j in 0..self.ny {
                self.set_u_bc(0, j, k, bc_x);
                self.set_u_bc(self.nx, j, k, bc_x);
            }
        }
        for k in 0..self.nz {
            for i in 0..self.nx {
                self.set_v_bc(i, 0, k, bc_y);
                self.set_v_bc(i, self.ny, k, bc_y);
            }
        }
        for j in 0..self.ny {
            for i in 0..self.nx {
                self.set_w_bc(i, j, 0, bc_z);
                self.set_w_bc(i, j, self.nz, bc_z);
            }
        }
    }

    /// Zero the normal velocity on every face marked solid.
    ///
    /// Walls only. [`MacGrid::enforce_face_boundaries`] is the superset that
    /// also reimposes inflow values and extrapolates outflow faces, and is
    /// what the projection and [`crate::cfd_solver::CfdSolver::step`] call.
    pub fn enforce_solid_faces(&mut self) {
        for (val, solid) in self.u.iter_mut().zip(self.u_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
        for (val, solid) in self.v.iter_mut().zip(self.v_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
        for (val, solid) in self.w.iter_mut().zip(self.w_solid.iter()) {
            if *solid {
                *val = Fix128::ZERO;
            }
        }
    }

    /// Impose every face condition on the normal velocities.
    ///
    /// Walls go to zero (what [`MacGrid::enforce_solid_faces`] does on its
    /// own), inflow faces go back to their prescribed value, and outflow
    /// faces take the value of the face one cell inside the domain. The
    /// projection calls this before it builds the divergence right-hand
    /// side, and [`crate::cfd_solver::CfdSolver::step`] calls it before
    /// advection and before the viscous term, so a wall never feeds a
    /// stale value into either.
    pub fn enforce_face_boundaries(&mut self) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..=self.nx {
                    let ix = self.idx_u(i, j, k);
                    match self.u_bc(i, j, k) {
                        FaceBc::Fluid => {}
                        FaceBc::Inflow { normal_velocity } => self.u[ix] = normal_velocity,
                        FaceBc::Outflow => {
                            let inner = if i > 0 { i - 1 } else { i + 1 };
                            self.u[ix] = self.u(inner, j, k);
                        }
                        _ => self.u[ix] = Fix128::ZERO,
                    }
                }
            }
        }
        for k in 0..self.nz {
            for j in 0..=self.ny {
                for i in 0..self.nx {
                    let ix = self.idx_v(i, j, k);
                    match self.v_bc(i, j, k) {
                        FaceBc::Fluid => {}
                        FaceBc::Inflow { normal_velocity } => self.v[ix] = normal_velocity,
                        FaceBc::Outflow => {
                            let inner = if j > 0 { j - 1 } else { j + 1 };
                            self.v[ix] = self.v(i, inner, k);
                        }
                        _ => self.v[ix] = Fix128::ZERO,
                    }
                }
            }
        }
        for k in 0..=self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let ix = self.idx_w(i, j, k);
                    match self.w_bc(i, j, k) {
                        FaceBc::Fluid => {}
                        FaceBc::Inflow { normal_velocity } => self.w[ix] = normal_velocity,
                        FaceBc::Outflow => {
                            let inner = if k > 0 { k - 1 } else { k + 1 };
                            self.w[ix] = self.w(i, j, inner);
                        }
                        _ => self.w[ix] = Fix128::ZERO,
                    }
                }
            }
        }
    }

    /// Does the X-face `(i, j, k)` carry no pressure degree of freedom?
    ///
    /// Reads the dense wall flag first and only consults the sparse map when
    /// the face is not a wall — and not at all while the map is empty, which
    /// is the closed box and the no-boundary default.
    #[inline]
    #[must_use]
    pub(crate) fn u_blocks_pressure(&self, i: usize, j: usize, k: usize) -> bool {
        self.is_u_solid(i, j, k)
            || (!self.u_face_bc.is_empty() && self.u_bc(i, j, k).blocks_pressure())
    }
    /// See [`MacGrid::u_blocks_pressure`].
    #[inline]
    #[must_use]
    pub(crate) fn v_blocks_pressure(&self, i: usize, j: usize, k: usize) -> bool {
        self.is_v_solid(i, j, k)
            || (!self.v_face_bc.is_empty() && self.v_bc(i, j, k).blocks_pressure())
    }
    /// See [`MacGrid::u_blocks_pressure`].
    #[inline]
    #[must_use]
    pub(crate) fn w_blocks_pressure(&self, i: usize, j: usize, k: usize) -> bool {
        self.is_w_solid(i, j, k)
            || (!self.w_face_bc.is_empty() && self.w_bc(i, j, k).blocks_pressure())
    }

    /// Is the X-face `(i, j, k)` an inflow, i.e. is its normal velocity
    /// prescribed rather than solved for?
    #[inline]
    #[must_use]
    fn u_is_inflow(&self, i: usize, j: usize, k: usize) -> bool {
        !self.u_face_bc.is_empty() && self.u_bc(i, j, k).is_inflow()
    }
    #[inline]
    #[must_use]
    fn v_is_inflow(&self, i: usize, j: usize, k: usize) -> bool {
        !self.v_face_bc.is_empty() && self.v_bc(i, j, k).is_inflow()
    }
    #[inline]
    #[must_use]
    fn w_is_inflow(&self, i: usize, j: usize, k: usize) -> bool {
        !self.w_face_bc.is_empty() && self.w_bc(i, j, k).is_inflow()
    }

    /// The no-slip wall standing between the X-face `(i, j, k)` and its
    /// neighbour one cell away in `y` (`forward` selects `+y`), if there is
    /// one.
    ///
    /// The two faces sit either side of the plane `y = j_f · dx` at
    /// `x = i · dx`, which is the corner shared by the Y-faces `(i−1, j_f,
    /// k)` and `(i, j_f, k)`; both have to be no-slip walls for the mirror
    /// to cross a wall. A viscous term uses the returned velocity `u_wall`
    /// as the no-slip ghost `2 u_wall − u_in`; `None` means the ordinary
    /// neighbour (or, on the domain edge, the zero-gradient mirror).
    #[must_use]
    pub fn u_wall_across_y(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let jf = if forward { j + 1 } else { j };
        wall_between(
            (i > 0).then(|| self.v_bc(i - 1, jf, k)),
            (i < self.nx).then(|| self.v_bc(i, jf, k)),
        )
    }
    /// See [`MacGrid::u_wall_across_y`]; the separating faces are Z-faces.
    #[must_use]
    pub fn u_wall_across_z(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let kf = if forward { k + 1 } else { k };
        wall_between(
            (i > 0).then(|| self.w_bc(i - 1, j, kf)),
            (i < self.nx).then(|| self.w_bc(i, j, kf)),
        )
    }
    /// See [`MacGrid::u_wall_across_y`]; a Y-face mirrored along `x`.
    #[must_use]
    pub fn v_wall_across_x(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let if_ = if forward { i + 1 } else { i };
        wall_between(
            (j > 0).then(|| self.u_bc(if_, j - 1, k)),
            (j < self.ny).then(|| self.u_bc(if_, j, k)),
        )
    }
    /// See [`MacGrid::u_wall_across_y`]; a Y-face mirrored along `z`.
    #[must_use]
    pub fn v_wall_across_z(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let kf = if forward { k + 1 } else { k };
        wall_between(
            (j > 0).then(|| self.w_bc(i, j - 1, kf)),
            (j < self.ny).then(|| self.w_bc(i, j, kf)),
        )
    }
    /// See [`MacGrid::u_wall_across_y`]; a Z-face mirrored along `x`.
    #[must_use]
    pub fn w_wall_across_x(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let if_ = if forward { i + 1 } else { i };
        wall_between(
            (k > 0).then(|| self.u_bc(if_, j, k - 1)),
            (k < self.nz).then(|| self.u_bc(if_, j, k)),
        )
    }
    /// See [`MacGrid::u_wall_across_y`]; a Z-face mirrored along `y`.
    #[must_use]
    pub fn w_wall_across_y(&self, i: usize, j: usize, k: usize, forward: bool) -> Option<Vec3Fix> {
        let jf = if forward { j + 1 } else { j };
        wall_between(
            (k > 0).then(|| self.v_bc(i, jf, k - 1)),
            (k < self.nz).then(|| self.v_bc(i, jf, k)),
        )
    }

    /// Divergence at cell (i, j, k). Positive = fluid expanding.
    #[must_use]
    pub fn divergence(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if self.dx.is_zero() {
            return Fix128::ZERO;
        }
        let du = self.u(i + 1, j, k) - self.u(i, j, k);
        let dv = self.v(i, j + 1, k) - self.v(i, j, k);
        let dw = self.w(i, j, k + 1) - self.w(i, j, k);
        (du + dv + dw) / self.dx
    }
}

// ============================================================================
// Poisson stencil under the face mask
// ============================================================================

/// Which faces of each cell take part in the pressure solve.
///
/// `open[c]` is ordered `[-x, +x, -y, +y, -z, +z]`. A face whose normal
/// velocity is fixed — either kind of wall, or an [`FaceBc::Inflow`] — is
/// closed: it contributes neither an off-diagonal coupling nor a count to
/// the diagonal, which is the homogeneous Neumann condition. An open face
/// always counts toward the diagonal; when it sits on the domain boundary
/// its neighbour pressure is the exterior `p = 0`, which is the Dirichlet
/// side of [`FaceBc::Outflow`] and of a plain [`FaceBc::Fluid`] rim.
///
/// Owned rather than borrowed so the solvers can keep mutating the grid.
struct PoissonMask {
    nx: usize,
    ny: usize,
    nz: usize,
    open: Vec<[bool; 6]>,
}

impl PoissonMask {
    fn from_grid(grid: &MacGrid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut open = vec![[true; 6]; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    open[i + nx * (j + ny * k)] = [
                        !grid.u_blocks_pressure(i, j, k),
                        !grid.u_blocks_pressure(i + 1, j, k),
                        !grid.v_blocks_pressure(i, j, k),
                        !grid.v_blocks_pressure(i, j + 1, k),
                        !grid.w_blocks_pressure(i, j, k),
                        !grid.w_blocks_pressure(i, j, k + 1),
                    ];
                }
            }
        }
        Self { nx, ny, nz, open }
    }

    /// Number of open faces of cell `c`; `−degree` is the matrix diagonal.
    #[inline]
    fn degree(&self, c: usize) -> i64 {
        self.open[c].iter().filter(|&&o| o).count() as i64
    }

    /// Sum of the neighbour pressures reachable through the open faces of
    /// cell `(i, j, k)`. Faces open onto the exterior contribute `p = 0`.
    #[inline]
    fn neighbour_sum(&self, p: &[Fix128], i: usize, j: usize, k: usize) -> Fix128 {
        let c = i + self.nx * (j + self.ny * k);
        let o = self.open[c];
        let mut acc = Fix128::ZERO;
        if o[0] && i > 0 {
            acc = acc + p[c - 1];
        }
        if o[1] && i + 1 < self.nx {
            acc = acc + p[c + 1];
        }
        if o[2] && j > 0 {
            acc = acc + p[c - self.nx];
        }
        if o[3] && j + 1 < self.ny {
            acc = acc + p[c + self.nx];
        }
        if o[4] && k > 0 {
            acc = acc + p[c - self.nx * self.ny];
        }
        if o[5] && k + 1 < self.nz {
            acc = acc + p[c + self.nx * self.ny];
        }
        acc
    }

    /// `out = A p` with `(A p)_c = −degree(c)·p_c + Σ_open p_neighbour`.
    fn apply(&self, p: &[Fix128], out: &mut [Fix128]) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let c = i + self.nx * (j + self.ny * k);
                    out[c] =
                        p[c] * Fix128::from_int(-self.degree(c)) + self.neighbour_sum(p, i, j, k);
                }
            }
        }
    }
}

/// `1 / degree(c)` per cell, zero where a cell is sealed on all six faces.
///
/// A sealed cell has no flux across any face, so its divergence — and with it
/// its right-hand side — is identically zero and the relaxation drives its
/// pressure to zero. That is the correct answer: the cell is decoupled from
/// the rest of the field and its pressure has no gradient to produce.
fn inverse_degrees(mask: &PoissonMask, n: usize) -> Vec<Fix128> {
    let mut inv = vec![Fix128::ZERO; n];
    for (c, slot) in inv.iter_mut().enumerate() {
        let deg = mask.degree(c);
        if deg > 0 {
            *slot = Fix128::from_ratio(1, deg);
        }
    }
    inv
}

/// Subtract `coeff · ∇p` from the face velocities and hold the walls at zero.
///
/// The sweep covers **every** face, boundary layer included: the Poisson
/// operator counts a non-solid boundary face against the exterior `p = 0`, so
/// skipping it leaves the rim cells with no degree of freedom and their
/// divergence grows instead of vanishing.
///
/// An [`FaceBc::Inflow`] face is skipped rather than corrected: its normal
/// velocity is prescribed, the stencil dropped it for exactly that reason,
/// and the value it must keep is the one
/// [`MacGrid::enforce_face_boundaries`] put there.
fn subtract_pressure_gradient(grid: &mut MacGrid, coeff: Fix128) {
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..=grid.nx {
                let ix = grid.idx_u(i, j, k);
                if grid.u_solid[ix] {
                    grid.u[ix] = Fix128::ZERO;
                    continue;
                }
                if grid.u_is_inflow(i, j, k) {
                    continue;
                }
                let hi = if i < grid.nx {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if i > 0 {
                    grid.pressure(i - 1, j, k)
                } else {
                    Fix128::ZERO
                };
                grid.u[ix] = grid.u[ix] - coeff * (hi - lo);
            }
        }
    }
    for k in 0..grid.nz {
        for j in 0..=grid.ny {
            for i in 0..grid.nx {
                let ix = grid.idx_v(i, j, k);
                if grid.v_solid[ix] {
                    grid.v[ix] = Fix128::ZERO;
                    continue;
                }
                if grid.v_is_inflow(i, j, k) {
                    continue;
                }
                let hi = if j < grid.ny {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if j > 0 {
                    grid.pressure(i, j - 1, k)
                } else {
                    Fix128::ZERO
                };
                grid.v[ix] = grid.v[ix] - coeff * (hi - lo);
            }
        }
    }
    for k in 0..=grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let ix = grid.idx_w(i, j, k);
                if grid.w_solid[ix] {
                    grid.w[ix] = Fix128::ZERO;
                    continue;
                }
                if grid.w_is_inflow(i, j, k) {
                    continue;
                }
                let hi = if k < grid.nz {
                    grid.pressure(i, j, k)
                } else {
                    Fix128::ZERO
                };
                let lo = if k > 0 {
                    grid.pressure(i, j, k - 1)
                } else {
                    Fix128::ZERO
                };
                grid.w[ix] = grid.w[ix] - coeff * (hi - lo);
            }
        }
    }
}

/// Build `rhs = ρ dx²/dt · ∇·u` after the walls have been enforced.
fn poisson_rhs(grid: &MacGrid, scale: Fix128) -> Vec<Fix128> {
    let mut rhs = vec![Fix128::ZERO; grid.nx * grid.ny * grid.nz];
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                rhs[i + grid.nx * (j + grid.ny * k)] = grid.divergence(i, j, k) * scale;
            }
        }
    }
    rhs
}

// ============================================================================
// Pressure projection (Jacobi)
// ============================================================================

/// Solve `∇²p = (ρ / dt) · ∇·u*` using Jacobi iteration, then subtract
/// `(dt / ρ) · ∇p` from the intermediate velocity to enforce
/// incompressibility.
///
/// - `iterations`: 20-100 typical for Jacobi; more expensive but stable.
/// - `density`: fluid density (kg/m³).
///
/// This is O(n · iterations); acceptable for grids up to ~64³. For larger
/// problems replace with multigrid.
/// Session 3 I9 upgrade: alias to `project_pressure_red_black_gs`, which
/// converges ~2× faster than Jacobi while retaining the same API. Older
/// callers see no behavioural change; a call with the same `iterations` now
/// yields a strictly smaller residual.
pub fn project_pressure(grid: &mut MacGrid, dt_s: Fix128, density_kg_m3: Fix128, iterations: u32) {
    project_pressure_red_black_gs(grid, dt_s, density_kg_m3, iterations);
}

/// Red-black Gauss-Seidel variant of `project_pressure` (Session 3 I9).
///
/// Alternates two sweeps per iteration: "red" cells where `i+j+k` is even and
/// "black" cells where it is odd. Immediately-updated pressures propagate
/// during each sweep, giving ~2× the convergence rate of Jacobi at the
/// same computational cost.
///
/// # Why a sweep is order-independent (and therefore parallel)
///
/// A sweep writes only cells of one colour, and the masked 7-point Laplacian
/// reads exactly the six face neighbours — each of which differs from the
/// centre in a single index, so each has the *opposite* parity. The cells read
/// and the cells written by one sweep are disjoint, so the cells of a colour
/// may be visited in any order, split across any number of workers, for a
/// result that is identical bit for bit. `Fix128` addition is a group
/// operation mod 2¹²⁸ (`tests/reduction_order_independence.rs`), so no rounding
/// enters through the partitioning either.
///
/// `red_black_sweep_is_independent_of_visit_order` pins that property against a
/// control (`colour_blind_gauss_seidel_drifts_from_red_black`) that does drift,
/// so the licence to reorder is measured rather than asserted.
///
/// # Why this sweep is nevertheless left sequential
///
/// Order-independence permits threading but does not pay for it here. Two
/// rayon forms were measured on 8 cores at 128³ (2.1 M cells) against the
/// 388.6 ms sequential baseline, both bit-identical and both slower:
/// per-cell `par_iter_mut` 935.4 ms, row-chunked `par_chunks_mut` 508.9 ms
/// (minimum of three runs each). The kernel is a 7-point stencil over a 33 MiB
/// working set, so it is memory-bandwidth-bound; adding threads on a single
/// SoC does not add bandwidth, while double-buffering to satisfy the borrow
/// checker adds roughly 40% more traffic. The gain has to come from more
/// memory systems — domain decomposition across ranks — and the property pinned
/// above is exactly what makes that split safe.
pub(crate) fn project_pressure_red_black_gs(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);

    for _ in 0..iterations {
        // Two-colour sweep (colour ∈ {0, 1})
        for colour in 0..2u32 {
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        if ((i + j + k) as u32 % 2) != colour {
                            continue;
                        }
                        let idx = i + grid.nx * (j + grid.ny * k);
                        let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                        grid.pressure[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                    }
                }
            }
        }
    }

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
}

/// Legacy Jacobi implementation, kept for benchmarking (Session 3 I9, crate-internal).
pub(crate) fn project_pressure_jacobi(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);

    // Jacobi iterations for `A p = rhs`, `A` the masked 7-point Laplacian.
    let mut p_new = grid.pressure.clone();
    for _ in 0..iterations {
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let idx = i + grid.nx * (j + grid.ny * k);
                    let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                    p_new[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                }
            }
        }
        grid.pressure.clone_from(&p_new);
    }

    // Velocity correction: u ← u − (dt/ρ)·∇p
    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
}

/// Preconditioned **BiCGStab** pressure solver (van der Vorst 1992).
///
/// Solves the discrete Poisson system `A p = b` for the MAC-grid
/// pressure, where `A` is the 7-point Laplacian restricted by the face mask:
/// a solid face drops out entirely (homogeneous Neumann) and an open face on
/// the domain boundary couples to the exterior `p = 0`. The RHS
/// `b = ρ dx² / dt · ∇·u` matches the Jacobi and red-black Gauss–Seidel
/// variants, and the velocity correction stage at the end is identical.
///
/// # Preconditioner
///
/// Diagonal Jacobi preconditioner `M = diag(A)`, read from the **same**
/// stencil the operator uses: `−(number of open faces)`, which is `−6` in the
/// interior and drops by one per walled face. Before the face mask landed the
/// operator used a fixed `−6` while the preconditioner counted missing
/// neighbours, so the two disagreed at every boundary cell — harmless for the
/// answer, but it slowed the iteration and the doc described the
/// preconditioner as if it were the operator.
///
/// # Convergence
///
/// Compared to red-black Gauss–Seidel, BiCGStab converges in roughly
/// `O(√N)` iterations vs `O(N)` for Jacobi/GS on 3-D Poisson, at the
/// price of ~7 dot products per iteration. Under Fix128 the dot
/// products dominate cost at small grid sizes; net wins appear as
/// grid size grows.
///
/// The iteration stops early when `‖r‖_∞ < tolerance` or after
/// `max_iterations` (whichever comes first).
pub(crate) fn project_pressure_bicgstab(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    max_iterations: u32,
    tolerance: Fix128,
) -> BicgstabStats {
    let default_stats = BicgstabStats {
        iterations: 0,
        final_residual: Fix128::ZERO,
        converged: true,
    };
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return default_stats;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let n = grid.nx * grid.ny * grid.nz;

    // Build RHS and cache the per-cell diagonal from the operator's own mask.
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let mut diag = vec![Fix128::ZERO; n];
    for (c, slot) in diag.iter_mut().enumerate() {
        *slot = Fix128::from_int(-mask.degree(c)); // A[c,c] = −degree(c)
    }

    // x = grid.pressure; solve A x = b with A = -Laplacian sign convention:
    // r0 = b − A x0
    let mut x = grid.pressure.clone();
    let mut r = vec![Fix128::ZERO; n];
    mask.apply(&x, &mut r);
    for i in 0..n {
        r[i] = rhs[i] - r[i];
    }
    let r_hat = r.clone();
    let mut p_vec = r.clone();
    let mut rho_prev = dot(&r_hat, &r);
    let mut alpha = Fix128::ONE;
    let mut omega = Fix128::ONE;
    let mut v_vec = vec![Fix128::ZERO; n];
    let mut y_vec = vec![Fix128::ZERO; n];
    let mut z_vec = vec![Fix128::ZERO; n];
    let mut s_vec = vec![Fix128::ZERO; n];
    let mut t_vec = vec![Fix128::ZERO; n];
    let mut iterations = 0_u32;
    let mut residual = linf_norm(&r);
    let mut converged = residual < tolerance;

    while iterations < max_iterations && !converged {
        iterations += 1;
        let rho = dot(&r_hat, &r);
        if rho.is_zero() {
            break;
        }
        if iterations > 1 {
            let beta = (rho / rho_prev) * (alpha / omega);
            // p = r + β (p - ω v)
            for i in 0..n {
                p_vec[i] = r[i] + beta * (p_vec[i] - omega * v_vec[i]);
            }
        }
        // y = M^{-1} p
        for i in 0..n {
            y_vec[i] = if diag[i].is_zero() {
                p_vec[i]
            } else {
                p_vec[i] / diag[i]
            };
        }
        mask.apply(&y_vec, &mut v_vec);
        let denom = dot(&r_hat, &v_vec);
        if denom.is_zero() {
            break;
        }
        alpha = rho / denom;
        // s = r - α v
        for i in 0..n {
            s_vec[i] = r[i] - alpha * v_vec[i];
        }
        let s_norm = linf_norm(&s_vec);
        if s_norm < tolerance {
            for i in 0..n {
                x[i] = x[i] + alpha * y_vec[i];
            }
            residual = s_norm;
            converged = true;
            break;
        }
        // z = M^{-1} s
        for i in 0..n {
            z_vec[i] = if diag[i].is_zero() {
                s_vec[i]
            } else {
                s_vec[i] / diag[i]
            };
        }
        mask.apply(&z_vec, &mut t_vec);
        let tt = dot(&t_vec, &t_vec);
        if tt.is_zero() {
            break;
        }
        omega = dot(&t_vec, &s_vec) / tt;
        // x = x + α y + ω z
        for i in 0..n {
            x[i] = x[i] + alpha * y_vec[i] + omega * z_vec[i];
        }
        // r = s - ω t
        for i in 0..n {
            r[i] = s_vec[i] - omega * t_vec[i];
        }
        residual = linf_norm(&r);
        converged = residual < tolerance;
        rho_prev = rho;
    }

    grid.pressure = x;

    // Velocity correction (same convention as the Jacobi variant).
    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
    BicgstabStats {
        iterations,
        final_residual: residual,
        converged,
    }
}

/// Diagnostic bundle returned by [`project_pressure_bicgstab`] (crate-internal).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BicgstabStats {
    /// Iterations actually performed.
    pub(crate) iterations: u32,
    /// Final `‖r‖_∞`.
    pub(crate) final_residual: Fix128,
    /// True if the iteration terminated below `tolerance`.
    pub(crate) converged: bool,
}

fn dot(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}

fn linf_norm(v: &[Fix128]) -> Fix128 {
    let mut best = Fix128::ZERO;
    for &x in v {
        let ax = x.abs();
        if ax > best {
            best = ax;
        }
    }
    best
}

// ============================================================================
// Trilinear P2G / G2P (Session 3 I2 upgrade)
// ============================================================================

/// Split a world coordinate `p` (m) into `(base_index, frac)` for a
/// specified face-grid offset `axis_offset` (in cell units, 0 or 0.5).
fn split(p_over_dx: Fix128, axis_offset: Fix128) -> (usize, Fix128) {
    // Adjust for staggering (subtract offset so origin aligns with face 0).
    let shifted = p_over_dx - axis_offset;
    // Fix128 is two's-complement I64F64: `hi` is already floor(shifted) and
    // `lo` the non-negative fractional part in [0, 1), for negative values
    // too. Anything left of face 0 clamps to the first face.
    if shifted.hi < 0 {
        (0, Fix128::ZERO)
    } else {
        (
            shifted.hi as usize,
            Fix128 {
                hi: 0,
                lo: shifted.lo,
            },
        )
    }
}

/// Trilinear-interpolate the u-face grid at a world point.
///
/// The u-face is staggered by `+0` on X, `+0.5` on Y, `+0.5` on Z relative
/// to the cell corner grid.
pub(crate) fn sample_u_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::ZERO);
    let (j, v) = split(fy, Fix128::from_ratio(1, 2));
    let (k, w) = split(fz, Fix128::from_ratio(1, 2));

    let i0 = i.min(grid.nx);
    let i1 = (i + 1).min(grid.nx);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);

    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;

    let c00 = grid.u(i0, j0, k0) * om_u + grid.u(i1, j0, k0) * u;
    let c10 = grid.u(i0, j1, k0) * om_u + grid.u(i1, j1, k0) * u;
    let c01 = grid.u(i0, j0, k1) * om_u + grid.u(i1, j0, k1) * u;
    let c11 = grid.u(i0, j1, k1) * om_u + grid.u(i1, j1, k1) * u;

    let c0 = c00 * om_v + c10 * v;
    let c1 = c01 * om_v + c11 * v;
    c0 * om_w + c1 * w
}

pub(crate) fn sample_v_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::from_ratio(1, 2));
    let (j, v) = split(fy, Fix128::ZERO);
    let (k, w) = split(fz, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny);
    let j1 = (j + 1).min(grid.ny);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let c00 = grid.v(i0, j0, k0) * om_v + grid.v(i0, j1, k0) * v;
    let c10 = grid.v(i1, j0, k0) * om_v + grid.v(i1, j1, k0) * v;
    let c01 = grid.v(i0, j0, k1) * om_v + grid.v(i0, j1, k1) * v;
    let c11 = grid.v(i1, j0, k1) * om_v + grid.v(i1, j1, k1) * v;
    let c0 = c00 * om_u + c10 * u;
    let c1 = c01 * om_u + c11 * u;
    c0 * om_w + c1 * w
}

pub(crate) fn sample_w_trilinear(grid: &MacGrid, pos_m: Vec3Fix) -> Fix128 {
    if grid.dx.is_zero() {
        return Fix128::ZERO;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let fx = pos_m.x * inv_dx;
    let fy = pos_m.y * inv_dx;
    let fz = pos_m.z * inv_dx;
    let (i, u) = split(fx, Fix128::from_ratio(1, 2));
    let (j, v) = split(fy, Fix128::from_ratio(1, 2));
    let (k, w) = split(fz, Fix128::ZERO);
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz);
    let k1 = (k + 1).min(grid.nz);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let c00 = grid.w(i0, j0, k0) * om_w + grid.w(i0, j0, k1) * w;
    let c10 = grid.w(i1, j0, k0) * om_w + grid.w(i1, j0, k1) * w;
    let c01 = grid.w(i0, j1, k0) * om_w + grid.w(i0, j1, k1) * w;
    let c11 = grid.w(i1, j1, k0) * om_w + grid.w(i1, j1, k1) * w;
    let c0 = c00 * om_u + c10 * u;
    let c1 = c01 * om_u + c11 * u;
    c0 * om_v + c1 * v
}

/// Return `(min, max)` of the 8 u-face corner values surrounding `pos_m`.
///
/// Used by MacCormack to clamp corrector results into the pre-advection
/// local range (Fedkiw's monotonicity guard).
pub(crate) fn sample_u_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::ZERO);
    let (j, _) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, _) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx);
    let i1 = (i + 1).min(grid.nx);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    corner_range(&[
        grid.u(i0, j0, k0),
        grid.u(i1, j0, k0),
        grid.u(i0, j1, k0),
        grid.u(i1, j1, k0),
        grid.u(i0, j0, k1),
        grid.u(i1, j0, k1),
        grid.u(i0, j1, k1),
        grid.u(i1, j1, k1),
    ])
}

/// Return `(min, max)` of the 8 v-face corner values surrounding `pos_m`.
pub(crate) fn sample_v_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, _) = split(pos_m.y * inv_dx, Fix128::ZERO);
    let (k, _) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny);
    let j1 = (j + 1).min(grid.ny);
    let k0 = k.min(grid.nz - 1);
    let k1 = (k + 1).min(grid.nz - 1);
    corner_range(&[
        grid.v(i0, j0, k0),
        grid.v(i1, j0, k0),
        grid.v(i0, j1, k0),
        grid.v(i1, j1, k0),
        grid.v(i0, j0, k1),
        grid.v(i1, j0, k1),
        grid.v(i0, j1, k1),
        grid.v(i1, j1, k1),
    ])
}

/// Return `(min, max)` of the 8 w-face corner values surrounding `pos_m`.
pub(crate) fn sample_w_range(grid: &MacGrid, pos_m: Vec3Fix) -> (Fix128, Fix128) {
    if grid.dx.is_zero() {
        return (Fix128::ZERO, Fix128::ZERO);
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, _) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, _) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, _) = split(pos_m.z * inv_dx, Fix128::ZERO);
    let i0 = i.min(grid.nx - 1);
    let i1 = (i + 1).min(grid.nx - 1);
    let j0 = j.min(grid.ny - 1);
    let j1 = (j + 1).min(grid.ny - 1);
    let k0 = k.min(grid.nz);
    let k1 = (k + 1).min(grid.nz);
    corner_range(&[
        grid.w(i0, j0, k0),
        grid.w(i1, j0, k0),
        grid.w(i0, j1, k0),
        grid.w(i1, j1, k0),
        grid.w(i0, j0, k1),
        grid.w(i1, j0, k1),
        grid.w(i0, j1, k1),
        grid.w(i1, j1, k1),
    ])
}

fn corner_range(corners: &[Fix128; 8]) -> (Fix128, Fix128) {
    let mut lo = corners[0];
    let mut hi = corners[0];
    for &c in &corners[1..] {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    (lo, hi)
}

/// Grid-to-particle: trilinear-sample velocity at world position `pos_m`.
///
/// Correct MAC-grid staggering is applied per component; velocities read
/// from their own face grids. Session 3 I2 upgrade from nearest-cell.
#[must_use]
pub fn g2p_velocity(grid: &MacGrid, pos_m: Vec3Fix) -> Vec3Fix {
    if grid.dx.is_zero() {
        return Vec3Fix::default();
    }
    Vec3Fix::new(
        sample_u_trilinear(grid, pos_m),
        sample_v_trilinear(grid, pos_m),
        sample_w_trilinear(grid, pos_m),
    )
}

/// Scatter one particle's velocity onto the 8 nearest u-face grid nodes
/// using trilinear weights (Session 3 I2 upgrade). The caller must
/// separately track per-cell weight sums if quantitative velocity means
/// are required — this routine only accumulates weighted deposits.
fn deposit_u_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vx: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::ZERO);
    let (j, v) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, w) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci <= grid.nx && cj < grid.ny && ck < grid.nz {
            let ix = grid.idx_u(ci, cj, ck);
            grid.u[ix] = grid.u[ix] + weight * vx;
        }
    }
}

fn deposit_v_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vy: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, v) = split(pos_m.y * inv_dx, Fix128::ZERO);
    let (k, w) = split(pos_m.z * inv_dx, Fix128::from_ratio(1, 2));
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci < grid.nx && cj <= grid.ny && ck < grid.nz {
            let ix = grid.idx_v(ci, cj, ck);
            grid.v[ix] = grid.v[ix] + weight * vy;
        }
    }
}

fn deposit_w_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vz: Fix128) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let (i, u) = split(pos_m.x * inv_dx, Fix128::from_ratio(1, 2));
    let (j, v) = split(pos_m.y * inv_dx, Fix128::from_ratio(1, 2));
    let (k, w) = split(pos_m.z * inv_dx, Fix128::ZERO);
    let om_u = Fix128::ONE - u;
    let om_v = Fix128::ONE - v;
    let om_w = Fix128::ONE - w;
    let corners = [
        (i, j, k, om_u * om_v * om_w),
        (i + 1, j, k, u * om_v * om_w),
        (i, j + 1, k, om_u * v * om_w),
        (i + 1, j + 1, k, u * v * om_w),
        (i, j, k + 1, om_u * om_v * w),
        (i + 1, j, k + 1, u * om_v * w),
        (i, j + 1, k + 1, om_u * v * w),
        (i + 1, j + 1, k + 1, u * v * w),
    ];
    for (ci, cj, ck, weight) in corners {
        if ci < grid.nx && cj < grid.ny && ck <= grid.nz {
            let ix = grid.idx_w(ci, cj, ck);
            grid.w[ix] = grid.w[ix] + weight * vz;
        }
    }
}

/// Particle-to-grid: trilinear scatter of one particle's velocity across
/// the 8 nearest face nodes for each of u/v/w. Session 3 I2 upgrade;
/// the earlier `p2g_nearest` implementation is retained below for callers
/// that need the simpler (less accurate) variant.
pub(crate) fn p2g_trilinear(grid: &mut MacGrid, pos_m: Vec3Fix, vel_m_per_s: Vec3Fix) {
    deposit_u_trilinear(grid, pos_m, vel_m_per_s.x);
    deposit_v_trilinear(grid, pos_m, vel_m_per_s.y);
    deposit_w_trilinear(grid, pos_m, vel_m_per_s.z);
}

/// Particle-to-grid: scatter one particle's velocity `vel_m_per_s` at
/// position `pos_m` onto the closest cell centre (nearest-neighbour).
///
/// Trilinear scatter is the "true" P2G but requires a companion weight
/// grid; this simplified version is sufficient for coarse PIC tests.
pub(crate) fn p2g_nearest(grid: &mut MacGrid, pos_m: Vec3Fix, vel_m_per_s: Vec3Fix) {
    if grid.dx.is_zero() {
        return;
    }
    let inv_dx = Fix128::ONE / grid.dx;
    let ix = (pos_m.x * inv_dx).hi.max(0) as usize;
    let iy = (pos_m.y * inv_dx).hi.max(0) as usize;
    let iz = (pos_m.z * inv_dx).hi.max(0) as usize;
    if ix >= grid.nx || iy >= grid.ny || iz >= grid.nz {
        return;
    }
    // Deposit x-vel onto the two neighbouring x-faces (average)
    let u_lo = grid.idx_u(ix, iy, iz);
    let u_hi = grid.idx_u(ix + 1, iy, iz);
    let half = Fix128::from_ratio(1, 2);
    grid.u[u_lo] = grid.u[u_lo] + vel_m_per_s.x * half;
    grid.u[u_hi] = grid.u[u_hi] + vel_m_per_s.x * half;
    let v_lo = grid.idx_v(ix, iy, iz);
    let v_hi = grid.idx_v(ix, iy + 1, iz);
    grid.v[v_lo] = grid.v[v_lo] + vel_m_per_s.y * half;
    grid.v[v_hi] = grid.v[v_hi] + vel_m_per_s.y * half;
    let w_lo = grid.idx_w(ix, iy, iz);
    let w_hi = grid.idx_w(ix, iy, iz + 1);
    grid.w[w_lo] = grid.w[w_lo] + vel_m_per_s.z * half;
    grid.w[w_hi] = grid.w[w_hi] + vel_m_per_s.z * half;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_grid_dimensions() {
        let g = MacGrid::new(4, 3, 2, Fix128::ONE);
        assert_eq!(g.u.len(), 5 * 3 * 2);
        assert_eq!(g.v.len(), 4 * 4 * 2);
        assert_eq!(g.w.len(), 4 * 3 * 3);
        assert_eq!(g.pressure.len(), 24);
    }

    #[test]
    fn mac_grid_zero_initialised() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.u(0, 0, 0), Fix128::ZERO);
        assert_eq!(g.v(1, 1, 1), Fix128::ZERO);
        assert_eq!(g.pressure(0, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn mac_grid_out_of_range_returns_zero() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.u(10, 0, 0), Fix128::ZERO);
        assert_eq!(g.pressure(5, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn cell_velocity_averages_faces() {
        let mut g = MacGrid::new(2, 1, 1, Fix128::ONE);
        // Set u(0, 0, 0) = 2, u(1, 0, 0) = 4 → avg = 3
        let i0 = g.idx_u(0, 0, 0);
        let i1 = g.idx_u(1, 0, 0);
        g.u[i0] = Fix128::from_int(2);
        g.u[i1] = Fix128::from_int(4);
        let (uc, _, _) = g.cell_velocity(0, 0, 0);
        assert_eq!(uc, Fix128::from_int(3));
    }

    #[test]
    fn divergence_zero_for_still_grid() {
        let g = MacGrid::new(3, 3, 3, Fix128::ONE);
        assert_eq!(g.divergence(1, 1, 1), Fix128::ZERO);
    }

    #[test]
    fn divergence_positive_for_expanding_flow() {
        let mut g = MacGrid::new(3, 3, 3, Fix128::ONE);
        // Set u(2, 1, 1) = 1, u(1, 1, 1) = 0 → du = 1 > 0
        let ix = g.idx_u(2, 1, 1);
        g.u[ix] = Fix128::ONE;
        let d = g.divergence(1, 1, 1);
        assert!(d > Fix128::ZERO);
    }

    #[test]
    fn project_zero_flow_stays_zero() {
        let mut g = MacGrid::new(3, 3, 3, Fix128::ONE);
        project_pressure(
            &mut g,
            Fix128::from_ratio(1, 60),
            Fix128::from_int(1000),
            10,
        );
        // All zeros → nothing to do; divergence remains 0 everywhere
        assert_eq!(g.divergence(1, 1, 1), Fix128::ZERO);
    }

    #[test]
    fn project_reduces_divergence() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        // Create a divergent velocity: u increasing across grid
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..=4 {
                    let ix = g.idx_u(i, j, k);
                    g.u[ix] = Fix128::from_int(i as i64);
                }
            }
        }
        let div_before = g.divergence(2, 2, 2);
        project_pressure(
            &mut g,
            Fix128::from_ratio(1, 60),
            Fix128::from_int(1000),
            50,
        );
        let div_after = g.divergence(2, 2, 2);
        // Projection should reduce |divergence|
        assert!(div_after.abs() < div_before.abs());
    }

    #[test]
    fn p2g_scatters_x_velocity() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let vel = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        p2g_nearest(&mut g, pos, vel);
        // Cell (1,1,1) faces should now hold 2 each (half of 4)
        assert_eq!(g.u(1, 1, 1), Fix128::from_int(2));
        assert_eq!(g.u(2, 1, 1), Fix128::from_int(2));
    }

    #[test]
    fn g2p_reads_cell_average_velocity() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let i0 = g.idx_u(1, 1, 1);
        let i1 = g.idx_u(2, 1, 1);
        g.u[i0] = Fix128::from_int(2);
        g.u[i1] = Fix128::from_int(4);
        // Cell centre u = 3
        let vel = g2p_velocity(
            &g,
            Vec3Fix::new(
                Fix128::from_ratio(15, 10),
                Fix128::from_ratio(15, 10),
                Fix128::from_ratio(15, 10),
            ),
        );
        assert_eq!(vel.x, Fix128::from_int(3));
    }

    #[test]
    fn p2g_out_of_range_ignored() {
        let mut g = MacGrid::new(2, 2, 2, Fix128::ONE);
        let pos = Vec3Fix::new(
            Fix128::from_int(100),
            Fix128::from_int(100),
            Fix128::from_int(100),
        );
        let vel = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        p2g_nearest(&mut g, pos, vel);
        assert_eq!(g.u(0, 0, 0), Fix128::ZERO);
    }

    #[test]
    fn p2g_trilinear_deposits_to_8_corners() {
        // Place a particle at the exact centre of a cell (interior).
        // Trilinear weights should distribute 1/8 to each of the 8 nearest
        // u-face nodes (well, technically the 8 face nodes around the u-face
        // cell whose centre is at that offset).
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        // Particle at (1.5, 1.5, 1.5) — a cell centre.
        // u-grid offset is (0, 0.5, 0.5) → local frac (0.5, 0.0, 0.0)
        //   → distributes only in x, so u_lo · 0.5 + u_hi · 0.5
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let vel = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        p2g_trilinear(&mut g, pos, vel);
        // u(1, 1, 1) and u(2, 1, 1) should each receive 2
        assert_eq!(g.u(1, 1, 1), Fix128::from_int(2));
        assert_eq!(g.u(2, 1, 1), Fix128::from_int(2));
    }

    #[test]
    fn g2p_trilinear_between_faces_interpolates() {
        let mut g = MacGrid::new(4, 4, 4, Fix128::ONE);
        let i0 = g.idx_u(1, 1, 1);
        let i1 = g.idx_u(2, 1, 1);
        g.u[i0] = Fix128::from_int(10);
        g.u[i1] = Fix128::from_int(20);
        // Position between the two u-faces should give ~15 by linear interp
        let pos = Vec3Fix::new(
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
            Fix128::from_ratio(15, 10),
        );
        let v = g2p_velocity(&g, pos);
        assert_eq!(v.x, Fix128::from_int(15));
    }

    #[test]
    fn split_negative_position_clamps_to_zero() {
        // For a negative x coordinate the base index must clamp to 0
        // so out-of-range particles don't crash the deposit routines.
        let (base, _) = split(Fix128::from_int(-3), Fix128::ZERO);
        assert_eq!(base, 0);
    }

    // ---- BiCGStab pressure solver tests --------------------------------

    fn seed_divergent_flow(nx: usize) -> MacGrid {
        let mut g = MacGrid::new(nx, nx, nx, Fix128::ONE);
        // Linear u profile: u(i, j, k) = i so ∇·u ≠ 0.
        for i in 0..=nx {
            for j in 0..nx {
                for k in 0..nx {
                    let ix = g.idx_u(i, j, k);
                    if ix < g.u.len() {
                        g.u[ix] = Fix128::from_int(i as i64);
                    }
                }
            }
        }
        g
    }

    #[test]
    fn bicgstab_reduces_divergence_below_jacobi_iterations() {
        let mut g = seed_divergent_flow(4);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let div_before = g.divergence(2, 2, 2).abs();
        let stats = project_pressure_bicgstab(&mut g, dt, rho, 15, Fix128::from_ratio(1, 10_000));
        let div_after = g.divergence(2, 2, 2).abs();
        assert!(div_after < div_before);
        assert!(stats.iterations <= 15);
    }

    #[test]
    fn bicgstab_matches_jacobi_within_tolerance() {
        let mut g_bicg = seed_divergent_flow(4);
        let mut g_jac = g_bicg.clone();
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let _ = project_pressure_bicgstab(&mut g_bicg, dt, rho, 30, Fix128::from_ratio(1, 100_000));
        project_pressure_jacobi(&mut g_jac, dt, rho, 200);
        // Both should drive the centre divergence close to zero.
        let div_bicg = g_bicg.divergence(2, 2, 2).abs();
        let div_jac = g_jac.divergence(2, 2, 2).abs();
        assert!(div_bicg < Fix128::from_ratio(1, 10));
        assert!(div_jac < Fix128::from_ratio(1, 10));
    }

    /// Visit orders for the reference red-black sweep below.
    #[derive(Clone, Copy)]
    enum VisitOrder {
        /// `i` fastest, then `j`, then `k` — what the solver itself walks.
        Natural,
        /// Exactly reversed.
        Reverse,
        /// A deterministic stride-7 permutation, so neighbouring cells are not
        /// visited near each other in time.
        Strided,
    }

    /// Indices of the cells of one colour, in the requested visit order.
    fn colour_cells(nx: usize, ny: usize, nz: usize, colour: u32, order: VisitOrder) -> Vec<usize> {
        let mut v = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if ((i + j + k) as u32 % 2) == colour {
                        v.push(i + nx * (j + ny * k));
                    }
                }
            }
        }
        match order {
            VisitOrder::Natural => v,
            VisitOrder::Reverse => {
                v.reverse();
                v
            }
            VisitOrder::Strided => {
                let n = v.len();
                let mut out = Vec::with_capacity(n);
                let mut seen = vec![false; n];
                let mut idx = 0usize;
                for _ in 0..n {
                    while seen[idx] {
                        idx = (idx + 1) % n;
                    }
                    seen[idx] = true;
                    out.push(v[idx]);
                    idx = (idx + 7) % n;
                }
                out
            }
        }
    }

    /// Red-black sweep written from the discretisation
    /// (`p_c ← (Σ_open-neighbours p − rhs_c) / deg_c`), with the visit order as
    /// a free parameter.
    ///
    /// The property under test is *order-independence*, so the reference shares
    /// the discretisation with the solver on purpose and varies only the order.
    /// Sharing the stencil is what isolates the variable; `reference_plain_gs`
    /// below is the control that shows the comparison has teeth.
    fn reference_red_black(
        grid: &mut MacGrid,
        dt_s: Fix128,
        density: Fix128,
        iterations: u32,
        order: VisitOrder,
    ) {
        grid.enforce_face_boundaries();
        let scale = density * grid.dx * grid.dx / dt_s;
        let n = grid.nx * grid.ny * grid.nz;
        let rhs = poisson_rhs(grid, scale);
        let mask = PoissonMask::from_grid(grid);
        let inv_deg = inverse_degrees(&mask, n);
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let plan = [
            colour_cells(nx, ny, nz, 0, order),
            colour_cells(nx, ny, nz, 1, order),
        ];
        for _ in 0..iterations {
            for cells in &plan {
                for &idx in cells {
                    let i = idx % nx;
                    let j = (idx / nx) % ny;
                    let k = idx / (nx * ny);
                    let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                    grid.pressure[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                }
            }
        }
        let inv_dx = Fix128::ONE / grid.dx;
        subtract_pressure_gradient(grid, dt_s / density * inv_dx);
    }

    /// Control: the same stencil swept over **all** cells in one pass, with no
    /// colouring. Dropping the colouring is exactly what makes a cell read a
    /// neighbour that the same sweep already wrote, so this result must differ
    /// from the red-black one. If it ever stops differing, the order-independence
    /// assertions above are vacuous.
    fn reference_plain_gs(grid: &mut MacGrid, dt_s: Fix128, density: Fix128, iterations: u32) {
        grid.enforce_face_boundaries();
        let scale = density * grid.dx * grid.dx / dt_s;
        let n = grid.nx * grid.ny * grid.nz;
        let rhs = poisson_rhs(grid, scale);
        let mask = PoissonMask::from_grid(grid);
        let inv_deg = inverse_degrees(&mask, n);
        for _ in 0..iterations {
            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let idx = i + grid.nx * (j + grid.ny * k);
                        let neighbours = mask.neighbour_sum(&grid.pressure, i, j, k);
                        grid.pressure[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                    }
                }
            }
        }
        let inv_dx = Fix128::ONE / grid.dx;
        subtract_pressure_gradient(grid, dt_s / density * inv_dx);
    }

    fn grids_are_bit_equal(a: &MacGrid, b: &MacGrid) -> bool {
        a.pressure == b.pressure && a.u == b.u && a.v == b.v && a.w == b.w
    }

    /// A red-black sweep touches cells whose stencils do not overlap, so the
    /// order the colour's cells are visited in cannot change the answer — which
    /// is what licenses running the sweep on rayon under `parallel`.
    #[test]
    fn red_black_sweep_is_independent_of_visit_order() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_divergent_flow(5);

        let mut solver_grid = base.clone();
        project_pressure_red_black_gs(&mut solver_grid, dt, rho, 8);

        for order in [
            VisitOrder::Natural,
            VisitOrder::Reverse,
            VisitOrder::Strided,
        ] {
            let mut reference = base.clone();
            reference_red_black(&mut reference, dt, rho, 8, order);
            assert!(
                grids_are_bit_equal(&solver_grid, &reference),
                "red-black result changed with the visit order: the sweep is not \
                 order-independent, so the parallel path cannot be bit-identical",
            );
        }
    }

    /// Teeth for the test above: drop the colouring and the answer does move.
    #[test]
    fn colour_blind_gauss_seidel_drifts_from_red_black() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_divergent_flow(5);

        let mut red_black = base.clone();
        project_pressure_red_black_gs(&mut red_black, dt, rho, 8);

        let mut colour_blind = base.clone();
        reference_plain_gs(&mut colour_blind, dt, rho, 8);

        assert!(
            !grids_are_bit_equal(&red_black, &colour_blind),
            "a colour-blind sweep agreed with the red-black one, so the \
             order-independence test has no discriminating power",
        );
    }

    #[test]
    fn bicgstab_stats_reports_convergence_status() {
        let mut g = seed_divergent_flow(4);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let stats =
            project_pressure_bicgstab(&mut g, dt, rho, 50, Fix128::from_ratio(1, 1_000_000));
        // Either converged within budget, or iterations == 50.
        assert!(stats.iterations >= 1);
        assert!(stats.iterations <= 50);
    }

    #[test]
    fn bicgstab_no_op_on_zero_dt() {
        let mut g = seed_divergent_flow(3);
        let stats = project_pressure_bicgstab(
            &mut g,
            Fix128::ZERO,
            Fix128::from_int(1000),
            10,
            Fix128::from_ratio(1, 10_000),
        );
        assert_eq!(stats.iterations, 0);
        assert!(stats.converged);
    }

    #[test]
    fn poisson_operator_diagonal_matches_the_preconditioner() {
        // The preconditioner BiCGStab divides by must be the diagonal of the
        // operator it is preconditioning: `(A e_c)_c == −degree(c)`. Before
        // the face mask the operator used a fixed −6 while `diag` counted
        // missing neighbours, so the two disagreed on every boundary cell.
        let nx = 3;
        let ny = 3;
        let nz = 3;
        let mut grid = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
        grid.set_u_solid(1, 1, 1, true); // one interior wall face
        let mask = PoissonMask::from_grid(&grid);
        let n = nx * ny * nz;
        for c in 0..n {
            let mut e = vec![Fix128::ZERO; n];
            e[c] = Fix128::ONE;
            let mut out = vec![Fix128::ZERO; n];
            mask.apply(&e, &mut out);
            assert_eq!(
                out[c],
                Fix128::from_int(-mask.degree(c)),
                "diagonal of cell {c} disagrees with degree()"
            );
        }
    }

    #[test]
    fn poisson_operator_is_symmetric() {
        // `A` must be symmetric for BiCGStab's convergence theory to apply:
        // `(A e_a)_b == (A e_b)_a`. A one-sided face mask (marking a face
        // solid for one of its two cells only) would break it.
        let nx = 3;
        let ny = 3;
        let nz = 2;
        let mut grid = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
        grid.set_closed_box_walls();
        grid.set_v_solid(1, 1, 0, true);
        let mask = PoissonMask::from_grid(&grid);
        let n = nx * ny * nz;
        let mut columns = Vec::with_capacity(n);
        for c in 0..n {
            let mut e = vec![Fix128::ZERO; n];
            e[c] = Fix128::ONE;
            let mut out = vec![Fix128::ZERO; n];
            mask.apply(&e, &mut out);
            columns.push(out);
        }
        for (a, col_a) in columns.iter().enumerate() {
            for (b, col_b) in columns.iter().enumerate() {
                assert_eq!(col_a[b], col_b[a], "A[{b},{a}] != A[{a},{b}]");
            }
        }
    }

    #[test]
    fn closed_box_degree_drops_to_the_open_face_count() {
        // A sealed 1-cell-thick slab has its two z faces walled, so the
        // stencil is the 2-D one (degree 4), not the 3-D one with two zero
        // contributions (degree 6) the fixed 1/6 divisor assumed.
        let mut grid = MacGrid::new(2, 2, 1, Fix128::from_ratio(1, 8));
        grid.set_closed_box_walls();
        let mask = PoissonMask::from_grid(&grid);
        for c in 0..4 {
            assert_eq!(mask.degree(c), 2, "corner cell of a sealed 2x2x1 slab");
        }
        let mut open_slab = MacGrid::new(2, 2, 1, Fix128::from_ratio(1, 8));
        open_slab.set_w_solid(0, 0, 0, true);
        open_slab.set_w_solid(0, 0, 1, true);
        let open_mask = PoissonMask::from_grid(&open_slab);
        assert_eq!(
            open_mask.degree(0),
            4,
            "only the z faces are walled, the four lateral faces stay open"
        );
    }

    #[test]
    fn walls_are_held_at_zero_by_the_projection() {
        let mut grid = MacGrid::new(4, 4, 4, Fix128::from_ratio(1, 8));
        grid.u.fill(Fix128::ONE);
        grid.set_closed_box_walls();
        project_pressure(&mut grid, Fix128::from_ratio(1, 100), Fix128::ONE, 50);
        for j in 0..4 {
            for k in 0..4 {
                assert_eq!(grid.u(0, j, k), Fix128::ZERO);
                assert_eq!(grid.u(4, j, k), Fix128::ZERO);
            }
        }
    }
}
