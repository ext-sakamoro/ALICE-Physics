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
/// problems use [`project_pressure_multigrid`] (reached from a solver step by
/// `CfdSolver::step_multigrid`).
/// Session 3 I9 upgrade: alias to `project_pressure_red_black_gs`, which
/// converges ~2× faster than Jacobi while retaining the same API. Older
/// callers see no behavioural change; a call with the same `iterations` now
/// yields a strictly smaller residual.
pub fn project_pressure(grid: &mut MacGrid, dt_s: Fix128, density_kg_m3: Fix128, iterations: u32) {
    project_pressure_red_black_gs(grid, dt_s, density_kg_m3, iterations);
}

/// Number of red-black Gauss-Seidel iterations before restriction.
const MG_PRE_SMOOTH: u32 = 1;
/// Number of red-black Gauss-Seidel iterations after prolongation.
const MG_POST_SMOOTH: u32 = 1;
/// Coarse correction factor `NUM / DEN`, applied to the prolonged correction.
///
/// With summation restriction and injection the Galerkin operator is four
/// times a re-discretised one while the restricted residual is eight times an
/// average, so the geometrically consistent correction is 8/4 = 2. Measured
/// on the smooth scene (n = 8/16/32) with a W-cycle: factor 1 gives
/// 0.41/0.55/0.60, factor 3/2 gives 0.21/0.26/0.28, factor 2 gives
/// 0.294/0.283/0.294 (flat), factor 5/2 gives 0.38/0.45/0.37. 2 is kept for
/// the flat rate and because it is the derived value.
const MG_CORRECTION_SCALE_NUM: i64 = 2;
/// See [`MG_CORRECTION_SCALE_NUM`].
const MG_CORRECTION_SCALE_DEN: i64 = 1;
/// Visits of the next-coarser level per cycle: 2 makes this a W-cycle.
///
/// Piecewise-constant aggregation does not give a level-independent rate
/// with a V-cycle (1 visit): measured on the smooth scene (n = 8/16/32),
/// factor 1 gives 0.46/0.73/0.82 and factor 2 gives 0.84/1.45/2.32
/// (diverges). Two visits give 0.294/0.283/0.294.
const MG_COARSE_VISITS: u32 = 2;

/// One level of the multigrid hierarchy: the pressure operator as integer
/// face conductances.
///
/// `cond[c][f]` (`f` ordered `-x, +x, -y, +y, -z, +z`) counts the open fine
/// faces between cell `c` and the neighbour across face `f`; a face onto the
/// exterior counts toward the diagonal only (Dirichlet `p = 0`). The operator
/// is `(A p)_c = −Σ_f cond[c][f]·p_c + Σ_{f, in domain} cond[c][f]·p_nb`.
///
/// On the finest level every entry is 0 or 1 and this is exactly
/// [`PoissonMask`]. A coarse level is the Galerkin product `R A P` with
/// `P` the piecewise-constant injection and `R = Pᵀ` the sum over the
/// aggregate: couplings inside an aggregate cancel against the diagonal and
/// what is left is the sum of the fine conductances across each aggregate
/// face, so the coarse operator is again a 7-point conductance operator and
/// no re-discretisation is involved (walls survive coarsening as zeros).
struct MgLevel {
    nx: usize,
    ny: usize,
    nz: usize,
    cond: Vec<[i64; 6]>,
}

impl MgLevel {
    fn from_mask(mask: &PoissonMask) -> Self {
        Self {
            nx: mask.nx,
            ny: mask.ny,
            nz: mask.nz,
            cond: mask.open.iter().map(|o| o.map(i64::from)).collect(),
        }
    }

    fn cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn degree(&self, c: usize) -> i64 {
        self.cond[c].iter().sum()
    }

    /// Galerkin coarse level: every axis longer than 1 is halved.
    fn coarsen(&self) -> Self {
        let fx = if self.nx > 1 { 2 } else { 1 };
        let fy = if self.ny > 1 { 2 } else { 1 };
        let fz = if self.nz > 1 { 2 } else { 1 };
        let (cx, cy, cz) = (self.nx / fx, self.ny / fy, self.nz / fz);
        let mut cond = vec![[0i64; 6]; cx * cy * cz];
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let c = i + self.nx * (j + self.ny * k);
                    let (ci, cj, ck) = (i / fx, j / fy, k / fz);
                    let target = &mut cond[ci + cx * (cj + cy * ck)];
                    let f = self.cond[c];
                    // Only the outer face of an aggregate counts; the faces
                    // between two children of one aggregate cancel.
                    if i % fx == 0 {
                        target[0] += f[0];
                    }
                    if i % fx == fx - 1 {
                        target[1] += f[1];
                    }
                    if j % fy == 0 {
                        target[2] += f[2];
                    }
                    if j % fy == fy - 1 {
                        target[3] += f[3];
                    }
                    if k % fz == 0 {
                        target[4] += f[4];
                    }
                    if k % fz == fz - 1 {
                        target[5] += f[5];
                    }
                }
            }
        }
        Self {
            nx: cx,
            ny: cy,
            nz: cz,
            cond,
        }
    }

    #[inline]
    fn neighbour_sum(&self, p: &[Fix128], i: usize, j: usize, k: usize) -> Fix128 {
        let c = i + self.nx * (j + self.ny * k);
        let f = self.cond[c];
        let mut acc = Fix128::ZERO;
        if f[0] != 0 && i > 0 {
            acc = acc + p[c - 1] * Fix128::from_int(f[0]);
        }
        if f[1] != 0 && i + 1 < self.nx {
            acc = acc + p[c + 1] * Fix128::from_int(f[1]);
        }
        if f[2] != 0 && j > 0 {
            acc = acc + p[c - self.nx] * Fix128::from_int(f[2]);
        }
        if f[3] != 0 && j + 1 < self.ny {
            acc = acc + p[c + self.nx] * Fix128::from_int(f[3]);
        }
        if f[4] != 0 && k > 0 {
            acc = acc + p[c - self.nx * self.ny] * Fix128::from_int(f[4]);
        }
        if f[5] != 0 && k + 1 < self.nz {
            acc = acc + p[c + self.nx * self.ny] * Fix128::from_int(f[5]);
        }
        acc
    }

    /// `1 / degree(c)`, zero for a cell with no open face (see
    /// [`inverse_degrees`]).
    fn inverse_degrees(&self) -> Vec<Fix128> {
        (0..self.cells())
            .map(|c| {
                let deg = self.degree(c);
                if deg > 0 {
                    Fix128::from_ratio(1, deg)
                } else {
                    Fix128::ZERO
                }
            })
            .collect()
    }

    /// Red-black Gauss-Seidel on `A p = rhs`, fixed visiting order.
    fn smooth(&self, inv_deg: &[Fix128], p: &mut [Fix128], rhs: &[Fix128], iterations: u32) {
        for _ in 0..iterations {
            for colour in 0..2usize {
                for k in 0..self.nz {
                    for j in 0..self.ny {
                        for i in 0..self.nx {
                            if (i + j + k) % 2 != colour {
                                continue;
                            }
                            let c = i + self.nx * (j + self.ny * k);
                            let nb = self.neighbour_sum(p, i, j, k);
                            p[c] = (nb - rhs[c]) * inv_deg[c];
                        }
                    }
                }
            }
        }
    }

    /// `res = rhs − A p`.
    fn residual(&self, p: &[Fix128], rhs: &[Fix128], res: &mut [Fix128]) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let c = i + self.nx * (j + self.ny * k);
                    let ap =
                        p[c] * Fix128::from_int(-self.degree(c)) + self.neighbour_sum(p, i, j, k);
                    res[c] = rhs[c] - ap;
                }
            }
        }
    }
}

/// One multigrid cycle on `levels[0]` (finest first); with
/// [`MG_COARSE_VISITS`] = 2 this is a W-cycle. `ps` / `rhss` hold the
/// per-level unknown and right-hand side, parallel to `levels`.
fn mg_vcycle(
    levels: &[MgLevel],
    invs: &[Vec<Fix128>],
    ps: &mut [Vec<Fix128>],
    rhss: &mut [Vec<Fix128>],
) {
    let fine = &levels[0];
    let (Some((p, p_rest)), Some((rhs, rhs_rest))) = (ps.split_first_mut(), rhss.split_first_mut())
    else {
        return;
    };
    if levels.len() == 1 {
        // Coarsest level: a single cell (every axis has been halved to 1),
        // for which one Gauss-Seidel sweep from zero is the exact solve.
        for x in p.iter_mut() {
            *x = Fix128::ZERO;
        }
        fine.smooth(&invs[0], p, rhs, 1);
        return;
    }
    let coarse = &levels[1];
    fine.smooth(&invs[0], p, rhs, MG_PRE_SMOOTH);

    let mut res = vec![Fix128::ZERO; fine.cells()];
    fine.residual(p, rhs, &mut res);

    // Restriction: sum over the aggregate, ascending fine index.
    let (fx, fy, fz) = (
        fine.nx / coarse.nx,
        fine.ny / coarse.ny,
        fine.nz / coarse.nz,
    );
    let rc = &mut rhs_rest[0];
    for x in rc.iter_mut() {
        *x = Fix128::ZERO;
    }
    for k in 0..fine.nz {
        for j in 0..fine.ny {
            for i in 0..fine.nx {
                let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                rc[ci] = rc[ci] + res[i + fine.nx * (j + fine.ny * k)];
            }
        }
    }
    for x in p_rest[0].iter_mut() {
        *x = Fix128::ZERO;
    }

    for _ in 0..MG_COARSE_VISITS {
        mg_vcycle(&levels[1..], &invs[1..], p_rest, rhs_rest);
    }

    // Prolongation: injection into the children, scaled.
    let scale = Fix128::from_ratio(MG_CORRECTION_SCALE_NUM, MG_CORRECTION_SCALE_DEN);
    let e = &p_rest[0];
    for k in 0..fine.nz {
        for j in 0..fine.ny {
            for i in 0..fine.nx {
                let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                let c = i + fine.nx * (j + fine.ny * k);
                p[c] = p[c] + e[ci] * scale;
            }
        }
    }

    fine.smooth(&invs[0], p, rhs, MG_POST_SMOOTH);
}

/// Multigrid pressure projection: W-cycles with a Galerkin coarse operator
/// (`A_c = R A P`, assembled from the fine `PoissonMask`) in place of the
/// Gauss-Seidel sweeps of [`project_pressure`].
///
/// Solves the same masked Poisson problem as [`project_pressure`] (walls are
/// Neumann, open rim faces Dirichlet `p = 0`) and then subtracts the pressure
/// gradient, so the two are interchangeable; the difference is that the
/// per-cycle error reduction does not degrade as the grid is refined.
///
/// Each cycle is one red-black Gauss-Seidel pre-smooth, restriction of the
/// residual by summation over `2×2×2` aggregates (an axis of extent 1 is not
/// coarsened), two recursive visits of the next level on the Galerkin
/// operator (a W-cycle: a V-cycle with piecewise-constant aggregation does
/// not reach a grid-independent rate, measured 0.46 / 0.73 / 0.82 for
/// n = 8 / 16 / 32, against 0.294 / 0.283 / 0.294 for the W-cycle),
/// prolongation by injection with a correction factor of 2 and one
/// post-smooth. The existing
/// `grid.pressure` is the initial guess, as for [`project_pressure`]. Every
/// loop visits cells in a fixed order, so the result is bit-reproducible.
///
/// # Inputs that cannot be solved
///
/// The grid is returned **bit-identical** (no wall enforcement, no pressure
/// write) when any extent is not a power of two (including 0), when
/// `dx`, `dt_s` or `density_kg_m3` is zero, or when `cycles == 0`. The
/// function has no error channel, so this early return is the explicit
/// contract rather than a silent fallback.
///
/// A fully sealed domain is the singular pure-Neumann problem. It is not
/// special-cased: a sealed cell has zero degree and gets zero pressure, and
/// the constant null space of a connected sealed region is left to the
/// smoother exactly as in [`project_pressure`], so the two solvers behave
/// alike there.
pub fn project_pressure_multigrid(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
) {
    let pow2 = |n: usize| n.is_power_of_two();
    if cycles == 0
        || !(pow2(grid.nx) && pow2(grid.ny) && pow2(grid.nz))
        || grid.dx.is_zero()
        || density_kg_m3.is_zero()
        || dt_s.is_zero()
    {
        return;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let rhs0 = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);

    let mut levels = vec![MgLevel::from_mask(&mask)];
    while let Some(next) = levels
        .last()
        .filter(|l| l.cells() > 1)
        .map(MgLevel::coarsen)
    {
        levels.push(next);
    }
    let invs: Vec<Vec<Fix128>> = levels.iter().map(MgLevel::inverse_degrees).collect();
    let mut ps: Vec<Vec<Fix128>> = levels
        .iter()
        .map(|l| vec![Fix128::ZERO; l.cells()])
        .collect();
    let mut rhss: Vec<Vec<Fix128>> = levels
        .iter()
        .map(|l| vec![Fix128::ZERO; l.cells()])
        .collect();
    ps[0].copy_from_slice(&grid.pressure);
    rhss[0].copy_from_slice(&rhs0);

    for _ in 0..cycles {
        mg_vcycle(&levels, &invs, &mut ps, &mut rhss);
    }
    grid.pressure.copy_from_slice(&ps[0]);

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
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

/// When a slab decomposition sends its boundary layers to the neighbouring
/// ranks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HaloSchedule {
    /// After every colour sweep. A red cell on a slab boundary is read by a
    /// black cell in the neighbouring slab during the very next sweep, so this
    /// is the schedule that reproduces the monolithic solve.
    EverySweep,
    /// Only after a full red+black iteration, so the second sweep of each
    /// iteration reads a stale halo. Kept as the control for
    /// `a_halo_exchanged_once_per_iteration_diverges_from_the_monolithic_solve`:
    /// if this ever stopped diverging, the bit-equality test above would be
    /// passing for reasons unrelated to the exchange.
    EveryIteration,
}

/// The `z` layers owned by `rank` when `nz` layers are split across `ranks`
/// contiguous slabs. Ranks may be empty when `ranks > nz`.
fn slab_bounds(nz: usize, ranks: usize, rank: usize) -> (usize, usize) {
    (rank * nz / ranks, (rank + 1) * nz / ranks)
}

/// The rank owning layer `k`, or `None` if no rank does.
fn slab_owner(bounds: &[(usize, usize)], k: usize) -> Option<usize> {
    bounds.iter().position(|&(k0, k1)| k0 <= k && k < k1)
}

/// Overwrite every layer a rank may not read — outside its owned range widened
/// by one halo layer — with a value far from any physical pressure.
///
/// This is what gives the decomposition test teeth: a stencil that reached past
/// the halo would pull the sentinel into the result instead of silently reading
/// a correct value that some other rank happened to leave in a shared buffer.
fn poison_beyond_halo(
    buf: &mut [Fix128],
    nz: usize,
    plane: usize,
    (k0, k1): (usize, usize),
    sentinel: Fix128,
) {
    let lo = k0.saturating_sub(1);
    let hi = (k1 + 1).min(nz);
    for k in 0..nz {
        if k >= lo && k < hi {
            continue;
        }
        let base = k * plane;
        for slot in &mut buf[base..base + plane] {
            *slot = sentinel;
        }
    }
}

/// Carries one `z` layer of a slab decomposition's pressure field from the
/// rank that owns it to a rank that needs it as a halo.
///
/// A layer is the `plane = nx · ny` values a 7-point stencil needs from across
/// a slab boundary, so it is also the unit a halo exchange moves: one layer,
/// one direction, one pair of ranks.
///
/// # Why the slabs belong to the transport
///
/// The per-rank buffers are reached through [`RankTransport::slab_mut`] rather
/// than held by the solver, because *where a neighbour's layer lives* is
/// exactly what a backend changes. In one process every slab is a `Vec` this
/// address space can read, so a delivery is a copy; under MPI a neighbour's
/// slab is in another address space and a delivery is a send/receive pair. A
/// solver that indexed its neighbour's buffer directly would be describing an
/// interface only the in-process backend could ever implement.
///
/// # Why a type parameter and not `dyn RankTransport`
///
/// A build selects its one backend at compile time (the MPI one will arrive
/// behind a feature, since it cannot coexist with the `no_std` build), so there
/// is nothing to choose at run time; a type parameter keeps the exchange
/// monomorphised inside the iteration loop and needs no `alloc::boxed::Box`,
/// which a `dyn` receiver would drag into `no_std`.
pub(crate) trait RankTransport {
    /// The full-length buffer rank `rank` sweeps its owned layers into.
    ///
    /// Must be `nx · ny · nz` values long: this stage keeps whole buffers so a
    /// halo that is too narrow fails as a mismatch rather than as a deadlock
    /// (see [`project_pressure_decomposed`]).
    fn slab_mut(&mut self, rank: usize) -> &mut [Fix128];

    /// Place layer `layer`, as rank `src` currently holds it, into rank `dst`'s
    /// copy of that layer.
    ///
    /// Only the named layer of `dst` may change; `src` is left alone.
    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize);
}

/// [`RankTransport`] for ranks sharing one address space: the slabs sit side by
/// side and a delivery is a copy between two of them.
pub(crate) struct LocalTransport {
    plane: usize,
    slabs: Vec<Vec<Fix128>>,
}

impl LocalTransport {
    /// `ranks` buffers of `nz · plane` values each; the solver fills them.
    pub(crate) fn new(ranks: usize, nz: usize, plane: usize) -> Self {
        Self {
            plane,
            slabs: vec![vec![Fix128::ZERO; nz * plane]; ranks],
        }
    }
}

impl RankTransport for LocalTransport {
    fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
        &mut self.slabs[rank]
    }

    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
        let base = layer * self.plane;
        // Through a temporary, which is also what a wire transport does with a
        // message: the slabs are separate allocations, so one cannot be
        // borrowed for reading while the other is borrowed for writing.
        let incoming = self.slabs[src][base..base + self.plane].to_vec();
        self.slabs[dst][base..base + self.plane].copy_from_slice(&incoming);
    }
}

/// Deliver one boundary layer in each direction to every rank that needs one.
///
/// Only halo layers are written, and owned layers are never touched, so the
/// order the ranks are serviced in cannot matter.
fn exchange_slab_halos<T: RankTransport>(transport: &mut T, bounds: &[(usize, usize)], nz: usize) {
    for (r, &(k0, k1)) in bounds.iter().enumerate() {
        if k0 == k1 {
            continue; // empty rank: nothing owned, nothing to surround
        }
        for layer in [k0.checked_sub(1), (k1 < nz).then_some(k1)]
            .into_iter()
            .flatten()
        {
            let Some(src) = slab_owner(bounds, layer) else {
                continue;
            };
            transport.deliver_layer(src, r, layer);
        }
    }
}

/// The red-black pressure projection run as `ranks` contiguous `z` slabs that
/// exchange a single halo layer, rather than as one monolithic sweep.
///
/// This is the decomposition that lets the solve span more than one memory
/// system, which the scale probe showed is the only way past both walls at the
/// hundreds-of-millions-of-elements target: a 1e8-body state does not fit in one
/// machine's RAM, and the stencil is bandwidth-bound so extra threads on one SoC
/// buy nothing.
///
/// The result is identical bit for bit to [`project_pressure_red_black_gs`] for
/// every rank count, because a colour sweep's reads and writes are disjoint (see
/// that function's notes) and `Fix128` addition is a group operation mod 2¹²⁸.
/// `slab_decomposition_reproduces_the_monolithic_pressure_solve` pins that.
///
/// Each rank still holds a full-size buffer here, with everything beyond its
/// halo poisoned, because the point of this stage is to fix the *decomposition*
/// — which layers a rank may read, and when they must arrive — without also
/// committing to slab-local storage. Keeping the buffers whole is what lets the
/// exchange schedule be tested in-process, where a wrong halo width fails as a
/// mismatch rather than as a deadlock.
///
/// The halo itself goes through [`RankTransport`], here the in-process
/// [`LocalTransport`]; [`project_pressure_decomposed_over`] takes any other
/// implementation, which is where an MPI backend attaches.
pub(crate) fn project_pressure_decomposed(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
    ranks: usize,
    schedule: HaloSchedule,
) {
    let mut transport = LocalTransport::new(ranks, grid.nz, grid.nx * grid.ny);
    project_pressure_decomposed_over(
        grid,
        dt_s,
        density_kg_m3,
        iterations,
        ranks,
        schedule,
        &mut transport,
    );
}

/// [`project_pressure_decomposed`] over a caller-supplied [`RankTransport`].
///
/// The solve owns the sweep, the ownership map and the poisoning; the transport
/// owns only the slabs and the deliveries between them. Splitting it here is
/// what makes the seam measurable:
/// `a_second_rank_transport_reproduces_the_in_process_one_bit_for_bit` runs the
/// same decomposition over a second implementation and gets the same bits, and
/// `a_transport_that_never_delivers_does_not_reproduce_the_solve` shows that
/// agreement is carried by `deliver_layer` rather than by both runs having
/// started from the same field.
fn project_pressure_decomposed_over<T: RankTransport>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
    ranks: usize,
    schedule: HaloSchedule,
    transport: &mut T,
) {
    if grid.dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() || ranks == 0 {
        return;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let n = nx * ny * nz;
    let plane = nx * ny;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);

    // Far outside the pressures this solve produces, so a stray read shows up as
    // a mismatch of many units rather than one in the last place.
    let sentinel = Fix128::from_int(1_000_000);
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    for (r, &b) in bounds.iter().enumerate() {
        let buf = transport.slab_mut(r);
        assert_eq!(
            buf.len(),
            n,
            "transport handed rank {r} a slab of {} cells instead of the {n} this stage needs",
            buf.len(),
        );
        buf.copy_from_slice(&grid.pressure);
        poison_beyond_halo(buf, nz, plane, b, sentinel);
    }

    for _ in 0..iterations {
        for colour in 0..2u32 {
            for (r, &(k0, k1)) in bounds.iter().enumerate() {
                let buf = transport.slab_mut(r);
                for k in k0..k1 {
                    for j in 0..ny {
                        for i in 0..nx {
                            if ((i + j + k) as u32 % 2) != colour {
                                continue;
                            }
                            let idx = i + nx * (j + ny * k);
                            let neighbours = mask.neighbour_sum(buf, i, j, k);
                            buf[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                        }
                    }
                }
            }
            if schedule == HaloSchedule::EverySweep {
                exchange_slab_halos(transport, &bounds, nz);
            }
        }
        if schedule == HaloSchedule::EveryIteration {
            exchange_slab_halos(transport, &bounds, nz);
        }
    }

    // Gather: every layer is owned by exactly one rank.
    for (r, &(k0, k1)) in bounds.iter().enumerate() {
        let buf = transport.slab_mut(r);
        for k in k0..k1 {
            let base = k * plane;
            grid.pressure[base..base + plane].copy_from_slice(&buf[base..base + plane]);
        }
    }

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
}

/// Bring every layer to rank 0, so that one rank ends up holding the whole
/// field even when no rank can read another's memory.
///
/// The order is fixed by `bounds` alone, which is what lets a transport whose
/// ranks live in different processes perform only its own half of each delivery
/// and still stay in step: every rank walks this same sequence.
fn gather_slabs_to_root<T: RankTransport>(transport: &mut T, bounds: &[(usize, usize)]) {
    for (r, &(k0, k1)) in bounds.iter().enumerate().skip(1) {
        for layer in k0..k1 {
            transport.deliver_layer(r, 0, layer);
        }
    }
}

/// One rank's half of [`project_pressure_decomposed`], for transports whose
/// ranks do not share an address space.
///
/// # Why a second driver
///
/// [`project_pressure_decomposed_over`] drives every rank from one place: it
/// sweeps rank 0, then rank 1, and so on, asking the transport for each slab in
/// turn. Reaching every slab from one call stack is only possible where every
/// slab is in this process, so that driver cannot be the one a cross-address-
/// space backend runs under. This function is the same decomposition seen from
/// inside a single rank: it asks for `my_rank`'s slab and never any other, and
/// every rank runs it concurrently over its own copy of `grid`.
///
/// `slab_mut` keeps its signature, so the in-process tests that drive all ranks
/// from one place keep working; what changes is which ranks a driver asks for.
/// A transport that owns only its own slab — [`SocketTransport`] — can therefore
/// assert that the rank it is handed is its own, and that assertion is what
/// makes "this driver is rank-local" a checked property rather than a comment.
///
/// # Why the schedule is still global
///
/// The exchange is walked in full on every rank — every `(src, dst, layer)` of
/// [`exchange_slab_halos`] and then of [`gather_slabs_to_root`], in the same
/// order — and the transport performs only the half it is party to. Deliveries
/// are thus matched by position in a sequence both ranks agree on, with no
/// handshake and no message header, and for any one delivery exactly one rank
/// writes while exactly one reads. Nothing in the schedule has both ranks
/// writing at once, which is what keeps a blocking stream from deadlocking
/// regardless of how much it will buffer.
///
/// # What each rank is left holding
///
/// Every rank must start from the same `grid` (the field is the input to the
/// decomposition, not something a rank derives). The gather lands on rank 0, so
/// rank 0 is the only rank that writes `grid` back; the others leave their copy
/// untouched, which `cross_process_rank1_worker` asserts from inside the one
/// process that is actually a non-root rank.
///
/// Each rank still holds a full-length buffer, for the reason
/// [`project_pressure_decomposed`] gives: a halo that is too narrow then fails
/// as a mismatch rather than as a deadlock. Slab-local storage is a separate
/// change.
// One more parameter than `project_pressure_decomposed_over`, whose list this
// deliberately mirrors so the two drivers stay comparable; `my_rank` is the
// whole difference between them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pressure_decomposed_on_rank<T: RankTransport>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
    ranks: usize,
    schedule: HaloSchedule,
    my_rank: usize,
    transport: &mut T,
) {
    if grid.dx.is_zero()
        || density_kg_m3.is_zero()
        || dt_s.is_zero()
        || ranks == 0
        || my_rank >= ranks
    {
        return;
    }
    grid.enforce_face_boundaries();
    let scale = density_kg_m3 * grid.dx * grid.dx / dt_s;
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let n = nx * ny * nz;
    let plane = nx * ny;
    let rhs = poisson_rhs(grid, scale);
    let mask = PoissonMask::from_grid(grid);
    let inv_deg = inverse_degrees(&mask, n);

    // Same sentinel as the single-process driver: far outside the pressures this
    // solve produces, so a read past the halo shows up as a mismatch of many
    // units rather than one in the last place.
    let sentinel = Fix128::from_int(1_000_000);
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    let (k0, k1) = bounds[my_rank];

    {
        let buf = transport.slab_mut(my_rank);
        assert_eq!(
            buf.len(),
            n,
            "transport handed rank {my_rank} a slab of {} cells instead of the {n} this stage needs",
            buf.len(),
        );
        buf.copy_from_slice(&grid.pressure);
        poison_beyond_halo(buf, nz, plane, (k0, k1), sentinel);
    }

    for _ in 0..iterations {
        for colour in 0..2u32 {
            {
                let buf = transport.slab_mut(my_rank);
                for k in k0..k1 {
                    for j in 0..ny {
                        for i in 0..nx {
                            if ((i + j + k) as u32 % 2) != colour {
                                continue;
                            }
                            let idx = i + nx * (j + ny * k);
                            let neighbours = mask.neighbour_sum(buf, i, j, k);
                            buf[idx] = (neighbours - rhs[idx]) * inv_deg[idx];
                        }
                    }
                }
            }
            if schedule == HaloSchedule::EverySweep {
                exchange_slab_halos(transport, &bounds, nz);
            }
        }
        if schedule == HaloSchedule::EveryIteration {
            exchange_slab_halos(transport, &bounds, nz);
        }
    }

    gather_slabs_to_root(transport, &bounds);

    if my_rank == 0 {
        {
            let buf = transport.slab_mut(0);
            grid.pressure.copy_from_slice(buf);
        }
        let inv_dx = Fix128::ONE / grid.dx;
        subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
    }
}

/// Bytes one [`Fix128`] occupies on the wire: `hi` then `lo`, little-endian.
#[cfg(feature = "std")]
const WIRE_BYTES_PER_CELL: usize = 16;

/// [`RankTransport`] for ranks in *different address spaces*: a rank holds only
/// its own slab, and a delivery is a message over a byte stream to the peer
/// that owns — or needs — the layer.
///
/// # What this backend settles that [`LocalTransport`] cannot
///
/// `LocalTransport` can always reach a neighbour's `Vec`, so it cannot tell a
/// rank-local driver from one that quietly reads foreign memory: both work. This
/// one has no foreign memory to reach, and [`RankTransport::slab_mut`] asserts
/// that the only rank it can serve is its own, so a driver that asked for a
/// neighbour's slab aborts here instead of returning a plausible answer. Running
/// the solve over it from two processes is what shows the trait is implementable
/// across a boundary an MPI backend would also have to cross.
///
/// # Wire format
///
/// `S` is any paired byte stream — `TcpStream`, `UnixStream`, a pipe — because
/// the only thing this type fixes is the encoding: a layer is `plane` `Fix128`
/// values, each 16 little-endian bytes (`hi`, then `lo`). The encoding is
/// explicit rather than a memory image, so the bytes one rank writes are the
/// bytes its peer reads on every target the crate builds for.
///
/// # Lockstep
///
/// Deliveries carry no header and are matched by their position in the schedule
/// both ranks walk (see [`project_pressure_decomposed_on_rank`]). A delivery
/// this rank is not party to is a no-op, which is how the two sequences stay
/// aligned.
#[cfg(feature = "std")]
pub(crate) struct SocketTransport<S> {
    /// The rank this process is; the only one [`RankTransport::slab_mut`] serves.
    my_rank: usize,
    /// `nx · ny`, the cells in one `z` layer and so in one message.
    plane: usize,
    /// This rank's full-length slab.
    buf: Vec<Fix128>,
    /// `links[r]` is the stream to rank `r`; this rank's own entry is `None`.
    links: Vec<Option<S>>,
    /// Encode/decode scratch for one layer, sized `plane · 16`.
    wire: Vec<u8>,
}

#[cfg(feature = "std")]
impl<S> SocketTransport<S> {
    /// A rank holding `cells` values, with `links[r]` the stream to rank `r`.
    pub(crate) fn new(my_rank: usize, cells: usize, plane: usize, links: Vec<Option<S>>) -> Self {
        Self {
            my_rank,
            plane,
            buf: vec![Fix128::ZERO; cells],
            links,
            wire: vec![0u8; plane * WIRE_BYTES_PER_CELL],
        }
    }
}

#[cfg(feature = "std")]
impl<S: std::io::Read + std::io::Write> RankTransport for SocketTransport<S> {
    fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
        assert_eq!(
            rank, self.my_rank,
            "rank {} was asked for rank {rank}'s slab, which is in another address \
             space: the driver is not rank-local",
            self.my_rank,
        );
        &mut self.buf
    }

    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
        if src == dst {
            return;
        }
        let base = layer * self.plane;

        if self.my_rank == src {
            for (cell, chunk) in self.buf[base..base + self.plane]
                .iter()
                .zip(self.wire.chunks_exact_mut(WIRE_BYTES_PER_CELL))
            {
                chunk[..8].copy_from_slice(&cell.hi.to_le_bytes());
                chunk[8..].copy_from_slice(&cell.lo.to_le_bytes());
            }
            let link = self.links[dst]
                .as_mut()
                .expect("no stream to the rank this delivery is addressed to");
            // A failed halo delivery leaves the ranks disagreeing about the
            // field, which no later step can repair, so this is fatal by design.
            link.write_all(&self.wire)
                .expect("send a halo layer to the peer rank");
            link.flush().expect("flush a halo layer to the peer rank");
        } else if self.my_rank == dst {
            let link = self.links[src]
                .as_mut()
                .expect("no stream from the rank this delivery comes from");
            link.read_exact(&mut self.wire)
                .expect("receive a halo layer from the peer rank");
            for (cell, chunk) in self.buf[base..base + self.plane]
                .iter_mut()
                .zip(self.wire.chunks_exact(WIRE_BYTES_PER_CELL))
            {
                let hi =
                    i64::from_le_bytes(chunk[..8].try_into().expect("8 bytes of a 16-byte cell"));
                let lo =
                    u64::from_le_bytes(chunk[8..].try_into().expect("8 bytes of a 16-byte cell"));
                *cell = Fix128::from_raw(hi, lo);
            }
        }
        // Neither end: another pair's delivery, counted but not performed, which
        // is what keeps every rank's position in the schedule the same.
    }
}

// ============================================================================
// Slab-local storage: the working set one rank actually needs
// ============================================================================
//
// Everything above this point gives every rank a *full-length* buffer. That is
// deliberate for fixing the decomposition — a halo narrower than the stencil
// then fails as a mismatch against the monolithic solve rather than as a
// deadlock — but it also means the whole field is resident on every rank, and
// at the hundreds-of-millions-of-cells target the field is the thing that does
// not fit. Measured on a 464³ grid (99,897,344 cells), the red-black path
// allocates seven full-length arrays:
//
//   pressure 1.49 GiB, the three face arrays 4.48 GiB, `PoissonMask::open`
//   0.56 GiB, `inverse_degrees` 1.49 GiB, `poisson_rhs` 1.49 GiB
//
// which is 9.50 GiB before the per-rank slab buffer, on a 16 GiB machine.
// Localising only the pressure leaves 8.01 GiB, so the whole working set has to
// shrink together or none of it does.
//
// # Why the index is rebased, and how a halo mistake surfaces
//
// There are two ways to make a rank's allocation small. Keeping the global
// index `i + nx·(j + ny·k)` and allocating less memory turns a read past the
// halo into an out-of-bounds panic, which is the strongest possible report —
// but it only works for the rank whose band starts at layer 0, because every
// other band would have to be addressed from an offset, and an offset *is* a
// rebase. So the index is rebased, and the rebase is the hazard the brief for
// this stage names: a global layer mapped to the wrong local row reads a cell
// that exists, and a halo one layer too narrow then returns a plausible number
// instead of failing.
//
// The answer is to let exactly one type own the conversion and to make it
// return `Option`: [`SlabStorage::layer`], [`SlabStorage::layer_mut`] and
// [`SlabStorage::sweep_window`] are the only places a global `z` layer becomes
// an offset, and a layer that is neither owned nor held as a halo is `None`
// there. The sweep and the gradient subtraction ask for the layers an *open*
// face obliges them to read, so a halo narrower than the stencil aborts at the
// read with [`HALO_MISSING_BELOW`] / [`HALO_MISSING_ABOVE`] — at the position
// of the mistake, with no dependence on a later value comparison.
// `a_halo_narrower_than_the_stencil_aborts_instead_of_returning_a_number` pins
// that, and the whole full-length path above is untouched, so the poison-based
// oracles keep running next to this one rather than being replaced by it.

/// A rank was asked for the layer below one of its owned cells and does not
/// hold it.
///
/// Named once because the sweep and the gradient subtraction can both reach it,
/// and because `#[should_panic]` matches on the text.
pub(crate) const HALO_MISSING_BELOW: &str =
    "slab halo too narrow: an open -z face of an owned cell reads the layer below the slab, \
     which is not resident";

/// The `+z` counterpart of [`HALO_MISSING_BELOW`].
pub(crate) const HALO_MISSING_ABOVE: &str =
    "slab halo too narrow: an open +z face of an owned cell reads the layer above the slab, \
     which is not resident";

/// A delivery named a layer the sending rank does not hold.
pub(crate) const DELIVERY_SOURCE_LACKS_LAYER: &str =
    "slab delivery: the sending rank does not hold the layer it was asked to send";

/// A delivery named a layer the receiving rank has no room for — the halo it
/// would land in was never allocated.
pub(crate) const DELIVERY_DESTINATION_LACKS_LAYER: &str =
    "slab delivery: the receiving rank has no room for the layer it was asked to receive, \
     so its halo is narrower than the exchange schedule";

/// Cells in one `z` layer of the cell-centred field.
#[inline]
fn cell_plane(nx: usize, ny: usize) -> usize {
    nx * ny
}

/// Cells in one `z` layer of the X-face field.
#[inline]
fn u_plane(nx: usize, ny: usize) -> usize {
    (nx + 1) * ny
}

/// Cells in one `z` layer of the Y-face field.
#[inline]
fn v_plane(nx: usize, ny: usize) -> usize {
    nx * (ny + 1)
}

/// The `z` layers of a cell-centred field that one rank keeps, and nothing
/// else: the layers it owns, widened by its halo.
///
/// The band is `lo..hi` of the global `0..nz`, stored layer-major, so a rank
/// owning 58 of 464 layers holds 60 layers rather than 464. Global layers are
/// turned into offsets here and nowhere else (see the section header above);
/// every accessor reports a layer outside the band as `None` rather than
/// clamping, wrapping or returning a neighbouring row.
pub(crate) struct SlabStorage {
    /// `nx · ny`.
    plane: usize,
    /// First resident layer.
    lo: usize,
    /// One past the last resident layer.
    hi: usize,
    /// Layers `lo..hi`, `plane` values each.
    cells: Vec<Fix128>,
}

impl SlabStorage {
    /// The band a rank owning `k0..k1` of `nz` layers needs with a halo of
    /// `halo` layers, zero-initialised.
    ///
    /// A rank that owns nothing holds nothing: it has no cell to sweep and so
    /// no neighbour to read.
    pub(crate) fn for_slab(plane: usize, nz: usize, (k0, k1): (usize, usize), halo: usize) -> Self {
        let (lo, hi) = if k0 == k1 {
            (k0, k0)
        } else {
            (k0.saturating_sub(halo), (k1 + halo).min(nz))
        };
        Self {
            plane,
            lo,
            hi,
            cells: vec![Fix128::ZERO; (hi - lo) * plane],
        }
    }

    /// The resident band, as `lo..hi` in global layer numbers.
    pub(crate) fn resident(&self) -> (usize, usize) {
        (self.lo, self.hi)
    }

    /// Layer `k`, or `None` when this rank does not hold it.
    pub(crate) fn layer(&self, k: usize) -> Option<&[Fix128]> {
        let row = k.checked_sub(self.lo)?;
        let base = row.checked_mul(self.plane)?;
        self.cells.get(base..base + self.plane)
    }

    /// Layer `k` for writing, or `None` when this rank does not hold it.
    pub(crate) fn layer_mut(&mut self, k: usize) -> Option<&mut [Fix128]> {
        let row = k.checked_sub(self.lo)?;
        let base = row.checked_mul(self.plane)?;
        self.cells.get_mut(base..base + self.plane)
    }

    /// Layer `k` for writing together with its two `z` neighbours for reading.
    ///
    /// A red-black sweep writes one colour and reads the opposite one, so the
    /// centre layer is both written and read; the two neighbours are only read.
    /// A neighbour that is not resident comes back as `None`, which is what the
    /// sweep turns into [`HALO_MISSING_BELOW`] / [`HALO_MISSING_ABOVE`] at the
    /// first open face that needs it.
    ///
    /// # Panics
    ///
    /// When `k` is not resident at all: a sweep may only visit layers its own
    /// rank owns, so that is a mistake in the driver rather than in the halo
    /// width, and the two deserve different reports.
    pub(crate) fn sweep_window(&mut self, k: usize) -> SweepWindow<'_> {
        let row = k.checked_sub(self.lo).filter(|r| self.lo + r < self.hi);
        let row = row.unwrap_or_else(|| {
            panic!(
                "a sweep asked for layer {k}, which rank's slab does not hold (resident {}..{})",
                self.lo, self.hi,
            )
        });
        let plane = self.plane;
        let (lower, rest) = self.cells.split_at_mut(row * plane);
        let (centre, upper) = rest.split_at_mut(plane);
        SweepWindow {
            below: (row > 0).then(|| &lower[(row - 1) * plane..row * plane]),
            centre,
            above: (!upper.is_empty()).then(|| &upper[..plane]),
        }
    }

    /// Bytes this rank's pressure band actually allocated.
    pub(crate) fn bytes(&self) -> SlabBytes {
        SlabBytes {
            pressure: self.cells.capacity() * size_of::<Fix128>(),
            ..SlabBytes::ZERO
        }
    }
}

/// One owned layer and its two `z` neighbours; see
/// [`SlabStorage::sweep_window`].
pub(crate) struct SweepWindow<'a> {
    /// Layer `k − 1`, if resident.
    below: Option<&'a [Fix128]>,
    /// Layer `k`.
    centre: &'a mut [Fix128],
    /// Layer `k + 1`, if resident.
    above: Option<&'a [Fix128]>,
}

/// What the pressure solve needs to know about one MAC face.
///
/// Carried per face rather than recomputed, because the conditions live in
/// `MacGrid`'s sparse maps and a rank holding slab-local storage has no
/// `MacGrid` to consult.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FaceFlags {
    /// The normal velocity is pinned to zero (`MacGrid::u_solid` and friends).
    solid: bool,
    /// The face carries no pressure degree of freedom
    /// ([`MacGrid::u_blocks_pressure`]).
    blocks_pressure: bool,
    /// The normal velocity is prescribed, so the projection leaves it alone.
    inflow: bool,
}

impl FaceFlags {
    fn of_u(grid: &MacGrid, i: usize, j: usize, k: usize) -> Self {
        Self {
            solid: grid.u_solid[grid.idx_u(i, j, k)],
            blocks_pressure: grid.u_blocks_pressure(i, j, k),
            inflow: grid.u_is_inflow(i, j, k),
        }
    }
    fn of_v(grid: &MacGrid, i: usize, j: usize, k: usize) -> Self {
        Self {
            solid: grid.v_solid[grid.idx_v(i, j, k)],
            blocks_pressure: grid.v_blocks_pressure(i, j, k),
            inflow: grid.v_is_inflow(i, j, k),
        }
    }
    fn of_w(grid: &MacGrid, i: usize, j: usize, k: usize) -> Self {
        Self {
            solid: grid.w_solid[grid.idx_w(i, j, k)],
            blocks_pressure: grid.w_blocks_pressure(i, j, k),
            inflow: grid.w_is_inflow(i, j, k),
        }
    }
}

/// One rank's share of the staggered velocity field and of the face
/// conditions, sized by the layers it owns instead of by the whole domain.
///
/// # Which faces a rank holds
///
/// X- and Y-faces belong to a single cell layer, so a rank owning `k0..k1`
/// holds and updates those for `k0..k1`. A Z-face sits *between* two layers:
/// the face at `k` needs the pressures at `k − 1` and `k`, so the rank owning
/// `k0..k1` holds `k0..=k1` — one more than it owns — and updates `k0..k1`,
/// leaving the face at `k1` to the rank that owns the layer above it. The rank
/// whose band ends at `nz` also updates the face at `nz`, which no layer sits
/// above. Every Z-face is therefore written exactly once across the ranks, and
/// the extra layer each rank holds is what its own divergence needs.
///
/// # Precondition
///
/// The velocities arrive with the face conditions already imposed
/// ([`MacGrid::enforce_face_boundaries`]). The monolithic solve calls that
/// itself, and calling it twice is not always the same as calling it once (an
/// outflow face copies its inward neighbour, which may itself have been
/// rewritten), so the enforcement happens once, before the field is split.
pub(crate) struct SlabFaces {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: Fix128,
    /// First owned layer.
    k0: usize,
    /// One past the last owned layer.
    k1: usize,
    /// X-face velocities of layers `k0..k1`, `(nx + 1) · ny` each.
    u: Vec<Fix128>,
    u_flags: Vec<FaceFlags>,
    /// Y-face velocities of layers `k0..k1`, `nx · (ny + 1)` each.
    v: Vec<Fix128>,
    v_flags: Vec<FaceFlags>,
    /// Z-face velocities of layers `k0..=k1`, `nx · ny` each; empty when the
    /// rank owns nothing.
    w: Vec<Fix128>,
    w_flags: Vec<FaceFlags>,
}

impl SlabFaces {
    /// Zero velocities and plain fluid faces over the layers `k0..k1`.
    pub(crate) fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: Fix128,
        (k0, k1): (usize, usize),
    ) -> Self {
        let owned = k1 - k0;
        let w_layers = if owned == 0 { 0 } else { owned + 1 };
        Self {
            nx,
            ny,
            nz,
            dx,
            k0,
            k1,
            u: vec![Fix128::ZERO; owned * u_plane(nx, ny)],
            u_flags: vec![FaceFlags::default(); owned * u_plane(nx, ny)],
            v: vec![Fix128::ZERO; owned * v_plane(nx, ny)],
            v_flags: vec![FaceFlags::default(); owned * v_plane(nx, ny)],
            w: vec![Fix128::ZERO; w_layers * cell_plane(nx, ny)],
            w_flags: vec![FaceFlags::default(); w_layers * cell_plane(nx, ny)],
        }
    }

    /// The layers `k0..k1` of `grid`, copied out face by face.
    ///
    /// The whole grid is resident here by construction — this is the
    /// constructor an in-process decomposition and the tests use. A rank with
    /// no `MacGrid` builds the same thing through [`SlabFaces::new`] plus the
    /// `*_layer_mut` accessors, which is what the hundred-million-cell
    /// measurement does.
    pub(crate) fn from_grid(grid: &MacGrid, (k0, k1): (usize, usize)) -> Self {
        let mut faces = Self::new(grid.nx, grid.ny, grid.nz, grid.dx, (k0, k1));
        if k0 == k1 {
            return faces;
        }
        let (nx, ny) = (grid.nx, grid.ny);
        for k in k0..k1 {
            let row = (k - k0) * u_plane(nx, ny);
            for j in 0..ny {
                for i in 0..=nx {
                    let at = row + i + (nx + 1) * j;
                    faces.u[at] = grid.u[grid.idx_u(i, j, k)];
                    faces.u_flags[at] = FaceFlags::of_u(grid, i, j, k);
                }
            }
            let row = (k - k0) * v_plane(nx, ny);
            for j in 0..=ny {
                for i in 0..nx {
                    let at = row + i + nx * j;
                    faces.v[at] = grid.v[grid.idx_v(i, j, k)];
                    faces.v_flags[at] = FaceFlags::of_v(grid, i, j, k);
                }
            }
        }
        for k in k0..=k1 {
            let row = (k - k0) * cell_plane(nx, ny);
            for j in 0..ny {
                for i in 0..nx {
                    let at = row + i + nx * j;
                    faces.w[at] = grid.w[grid.idx_w(i, j, k)];
                    faces.w_flags[at] = FaceFlags::of_w(grid, i, j, k);
                }
            }
        }
        faces
    }

    /// The owned layers, as `k0..k1`.
    pub(crate) fn owned(&self) -> (usize, usize) {
        (self.k0, self.k1)
    }

    /// Offset of layer `k` inside an owned-layer array of `plane` values.
    ///
    /// # Panics
    ///
    /// When `k` is not an owned layer. Unlike the pressure band there is no
    /// halo here, so there is no legitimate caller for a layer the rank does
    /// not own.
    fn owned_row(&self, k: usize, plane: usize) -> usize {
        assert!(
            k >= self.k0 && k < self.k1,
            "layer {k} is not owned by this slab (owns {}..{})",
            self.k0,
            self.k1,
        );
        (k - self.k0) * plane
    }

    /// Offset of Z-face layer `k`, which runs one past the owned layers.
    fn w_row(&self, k: usize) -> usize {
        assert!(
            k >= self.k0 && k <= self.k1 && self.k0 != self.k1,
            "Z-face layer {k} is not held by this slab (holds {}..={})",
            self.k0,
            self.k1,
        );
        (k - self.k0) * cell_plane(self.nx, self.ny)
    }

    fn u_layer(&self, k: usize) -> (&[Fix128], &[FaceFlags]) {
        let plane = u_plane(self.nx, self.ny);
        let row = self.owned_row(k, plane);
        (&self.u[row..row + plane], &self.u_flags[row..row + plane])
    }

    fn v_layer(&self, k: usize) -> (&[Fix128], &[FaceFlags]) {
        let plane = v_plane(self.nx, self.ny);
        let row = self.owned_row(k, plane);
        (&self.v[row..row + plane], &self.v_flags[row..row + plane])
    }

    fn w_layer(&self, k: usize) -> (&[Fix128], &[FaceFlags]) {
        let plane = cell_plane(self.nx, self.ny);
        let row = self.w_row(k);
        (&self.w[row..row + plane], &self.w_flags[row..row + plane])
    }

    /// X-face velocities and conditions of owned layer `k`, for writing.
    pub(crate) fn u_layer_mut(&mut self, k: usize) -> (&mut [Fix128], &mut [FaceFlags]) {
        let plane = u_plane(self.nx, self.ny);
        let row = self.owned_row(k, plane);
        (
            &mut self.u[row..row + plane],
            &mut self.u_flags[row..row + plane],
        )
    }

    /// Y-face velocities and conditions of owned layer `k`, for writing.
    pub(crate) fn v_layer_mut(&mut self, k: usize) -> (&mut [Fix128], &mut [FaceFlags]) {
        let plane = v_plane(self.nx, self.ny);
        let row = self.owned_row(k, plane);
        (
            &mut self.v[row..row + plane],
            &mut self.v_flags[row..row + plane],
        )
    }

    /// Z-face velocities and conditions of held layer `k`, for writing.
    pub(crate) fn w_layer_mut(&mut self, k: usize) -> (&mut [Fix128], &mut [FaceFlags]) {
        let plane = cell_plane(self.nx, self.ny);
        let row = self.w_row(k);
        (
            &mut self.w[row..row + plane],
            &mut self.w_flags[row..row + plane],
        )
    }

    /// The Z-face layers this rank updates: its owned layers, plus the domain's
    /// top face when its band ends there.
    fn w_written(&self) -> core::ops::Range<usize> {
        if self.k0 == self.k1 {
            return self.k0..self.k0;
        }
        let end = if self.k1 == self.nz {
            self.nz + 1
        } else {
            self.k1
        };
        self.k0..end
    }

    /// Bytes this rank's face arrays actually allocated.
    pub(crate) fn bytes(&self) -> SlabBytes {
        let values = self.u.capacity() + self.v.capacity() + self.w.capacity();
        let flags = self.u_flags.capacity() + self.v_flags.capacity() + self.w_flags.capacity();
        SlabBytes {
            faces: values * size_of::<Fix128>(),
            face_flags: flags * size_of::<FaceFlags>(),
            ..SlabBytes::ZERO
        }
    }
}

/// The per-cell data a rank's sweep reads, over its owned layers only.
///
/// The monolithic solve builds the same three things full-length
/// ([`PoissonMask`], [`inverse_degrees`], [`poisson_rhs`]); together they are
/// 3.54 of the 9.50 GiB a 464³ grid needs, so localising the pressure without
/// localising these would not fit either.
pub(crate) struct SlabStencil {
    k0: usize,
    k1: usize,
    plane: usize,
    /// Which of the six faces of each owned cell take part, ordered as
    /// [`PoissonMask::open`].
    open: Vec<[bool; 6]>,
    /// `1 / degree`, zero for a sealed cell, as [`inverse_degrees`].
    inv_deg: Vec<Fix128>,
    /// `ρ dx²/dt · ∇·u`, as [`poisson_rhs`].
    rhs: Vec<Fix128>,
}

impl SlabStencil {
    /// Build the owned-layer stencil from this rank's faces.
    ///
    /// Each value is computed by the same expression the full-length path uses,
    /// in the same order: `Fix128` multiplication is not associative, so
    /// "equivalent" arithmetic is not good enough for a bit-exact claim.
    ///
    /// # Panics
    ///
    /// When `dx` is zero, which the drivers reject before they get here.
    pub(crate) fn build(faces: &SlabFaces, scale: Fix128) -> Self {
        assert!(
            !faces.dx.is_zero(),
            "a slab stencil needs a non-zero cell spacing",
        );
        let (nx, ny) = (faces.nx, faces.ny);
        let (k0, k1) = faces.owned();
        let plane = cell_plane(nx, ny);
        let cells = (k1 - k0) * plane;
        let mut open = Vec::with_capacity(cells);
        let mut inv_deg = Vec::with_capacity(cells);
        let mut rhs = Vec::with_capacity(cells);
        // Row stride inside one X-face layer: the X-faces of a layer are
        // `(nx + 1) · ny`, laid out `i + (nx + 1) · j`.
        let u_row = nx + 1;

        for k in k0..k1 {
            let (u, u_flags) = faces.u_layer(k);
            let (v, v_flags) = faces.v_layer(k);
            let (w_lo, w_lo_flags) = faces.w_layer(k);
            let (w_hi, w_hi_flags) = faces.w_layer(k + 1);
            for j in 0..ny {
                for i in 0..nx {
                    let o = [
                        !u_flags[i + u_row * j].blocks_pressure,
                        !u_flags[i + 1 + u_row * j].blocks_pressure,
                        !v_flags[i + nx * j].blocks_pressure,
                        !v_flags[i + nx * (j + 1)].blocks_pressure,
                        !w_lo_flags[i + nx * j].blocks_pressure,
                        !w_hi_flags[i + nx * j].blocks_pressure,
                    ];
                    let degree = o.iter().filter(|&&face| face).count() as i64;
                    inv_deg.push(if degree > 0 {
                        Fix128::from_ratio(1, degree)
                    } else {
                        Fix128::ZERO
                    });
                    let du = u[i + 1 + u_row * j] - u[i + u_row * j];
                    let dv = v[i + nx * (j + 1)] - v[i + nx * j];
                    let dw = w_hi[i + nx * j] - w_lo[i + nx * j];
                    rhs.push((du + dv + dw) / faces.dx * scale);
                    open.push(o);
                }
            }
        }

        Self {
            k0,
            k1,
            plane,
            open,
            inv_deg,
            rhs,
        }
    }

    /// Offset of owned layer `k` inside the per-cell arrays.
    fn row(&self, k: usize) -> usize {
        assert!(
            k >= self.k0 && k < self.k1,
            "layer {k} is not owned by this stencil (owns {}..{})",
            self.k0,
            self.k1,
        );
        (k - self.k0) * self.plane
    }

    /// Bytes this rank's stencil actually allocated.
    pub(crate) fn bytes(&self) -> SlabBytes {
        SlabBytes {
            open: self.open.capacity() * size_of::<[bool; 6]>(),
            inverse_degrees: self.inv_deg.capacity() * size_of::<Fix128>(),
            rhs: self.rhs.capacity() * size_of::<Fix128>(),
            ..SlabBytes::ZERO
        }
    }
}

/// What one rank's slab-local working set allocated, read back from the
/// containers rather than recomputed from the dimensions.
///
/// Recomputing would be a second expression for the same thing, free to drift
/// from the allocations it claims to describe — which is how
/// `examples/hpc_scale_probe.rs` came to under-report the full-length path by
/// 1.59x (it counted `MacGrid` and not the mask, the inverse degrees or the
/// right-hand side).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlabBytes {
    /// The resident pressure band.
    pub(crate) pressure: usize,
    /// The three face-velocity arrays.
    pub(crate) faces: usize,
    /// The three face-condition arrays.
    pub(crate) face_flags: usize,
    /// The open-face mask.
    pub(crate) open: usize,
    /// The reciprocal diagonal.
    pub(crate) inverse_degrees: usize,
    /// The Poisson right-hand side.
    pub(crate) rhs: usize,
}

impl SlabBytes {
    /// All fields zero; the identity of [`SlabBytes::add`].
    pub(crate) const ZERO: Self = Self {
        pressure: 0,
        faces: 0,
        face_flags: 0,
        open: 0,
        inverse_degrees: 0,
        rhs: 0,
    };

    /// Field-wise sum, for adding up the parts of one rank's working set.
    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            pressure: self.pressure + other.pressure,
            faces: self.faces + other.faces,
            face_flags: self.face_flags + other.face_flags,
            open: self.open + other.open,
            inverse_degrees: self.inverse_degrees + other.inverse_degrees,
            rhs: self.rhs + other.rhs,
        }
    }

    /// Bytes in the whole working set.
    pub(crate) fn total(self) -> usize {
        self.pressure + self.faces + self.face_flags + self.open + self.inverse_degrees + self.rhs
    }
}

/// [`RankTransport`]'s counterpart for ranks whose storage is slab-local.
///
/// # Why a second trait rather than a second implementation
///
/// [`RankTransport::slab_mut`] returns one contiguous `&mut [Fix128]` that both
/// full-length drivers require to be `nx · ny · nz` values long, and index with
/// the global cell index. A slab-local rank has no such buffer: that is the
/// whole point. Widening `RankTransport` to admit one would change the meaning
/// of a method four existing implementations and nine existing tests depend on,
/// so the slab-local side gets its own trait and the full-length side keeps
/// running unchanged beside it. The deliveries are the same `(src, dst, layer)`
/// sequence either way, which
/// `both_exchange_walks_deliver_the_same_layers_in_the_same_order` pins.
pub(crate) trait SlabTransport {
    /// Rank `rank`'s slab-local pressure band.
    fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage;

    /// Place layer `layer`, as rank `src` holds it, into rank `dst`'s halo.
    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize);
}

/// [`SlabTransport`] for ranks sharing one address space: every rank's band is
/// a `Vec` here, and a delivery is a copy through a one-layer buffer — which is
/// also the shape a wire transport has, so the schedule does not change when
/// the backend does.
pub(crate) struct LocalSlabTransport {
    slabs: Vec<SlabStorage>,
    /// One layer of scratch, so a delivery never borrows two slabs at once.
    staging: Vec<Fix128>,
}

impl LocalSlabTransport {
    /// One band per rank, each covering its owned layers widened by `halo`, and
    /// each initialised from `field` — the field the decomposition starts from.
    ///
    /// The initial halo matters: the first colour sweep of a rank reads its
    /// neighbour's boundary layer before any exchange has happened, and the
    /// monolithic solve reads the same initial values at that point. Starting a
    /// halo at zero instead would not be caught by a field that starts at zero,
    /// which is why `slab_local_storage_reproduces_the_monolithic_pressure_solve`
    /// includes a case whose initial pressure is not zero.
    pub(crate) fn from_field(
        bounds: &[(usize, usize)],
        nz: usize,
        plane: usize,
        halo: usize,
        field: &[Fix128],
    ) -> Self {
        assert_eq!(
            field.len(),
            nz * plane,
            "the field a slab decomposition starts from must cover the whole domain",
        );
        let mut slabs = Vec::with_capacity(bounds.len());
        for &b in bounds {
            let mut slab = SlabStorage::for_slab(plane, nz, b, halo);
            let (lo, hi) = slab.resident();
            for k in lo..hi {
                let layer = slab
                    .layer_mut(k)
                    .expect("a layer inside the band this slab just reported");
                layer.copy_from_slice(&field[k * plane..(k + 1) * plane]);
            }
            slabs.push(slab);
        }
        Self {
            slabs,
            staging: vec![Fix128::ZERO; plane],
        }
    }

    /// Rank `rank`'s band, for reading back the answer.
    pub(crate) fn slab(&self, rank: usize) -> &SlabStorage {
        &self.slabs[rank]
    }

    /// Bytes every rank's band allocated here, rank by rank.
    pub(crate) fn bytes(&self) -> Vec<SlabBytes> {
        self.slabs.iter().map(SlabStorage::bytes).collect()
    }
}

impl SlabTransport for LocalSlabTransport {
    fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage {
        &mut self.slabs[rank]
    }

    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
        if src == dst {
            return;
        }
        let from = self.slabs[src]
            .layer(layer)
            .expect(DELIVERY_SOURCE_LACKS_LAYER);
        self.staging.copy_from_slice(from);
        let into = self.slabs[dst]
            .layer_mut(layer)
            .expect(DELIVERY_DESTINATION_LACKS_LAYER);
        into.copy_from_slice(&self.staging);
    }
}

/// [`exchange_slab_halos`] for a [`SlabTransport`]: the same `(src, dst, layer)`
/// sequence, over storage that holds only a band.
///
/// The walk is written out again rather than shared, because sharing it would
/// mean moving `deliver_layer` to a supertrait of [`RankTransport`] and editing
/// every existing implementation of it. The two walks agreeing is pinned by
/// `both_exchange_walks_deliver_the_same_layers_in_the_same_order`, which
/// compares the recorded sequences instead of trusting that two copies of a
/// loop stay the same.
fn exchange_slab_halos_local<T: SlabTransport>(
    transport: &mut T,
    bounds: &[(usize, usize)],
    nz: usize,
) {
    for (r, &(k0, k1)) in bounds.iter().enumerate() {
        if k0 == k1 {
            continue; // empty rank: nothing owned, nothing to surround
        }
        for layer in [k0.checked_sub(1), (k1 < nz).then_some(k1)]
            .into_iter()
            .flatten()
        {
            let Some(src) = slab_owner(bounds, layer) else {
                continue;
            };
            transport.deliver_layer(src, r, layer);
        }
    }
}

/// One red-black colour sweep over the layers a rank owns, reading its band and
/// nothing else.
///
/// The stencil is [`PoissonMask::neighbour_sum`]'s, in the same order and with
/// the same guards, but addressed layer by layer: the four in-plane neighbours
/// come out of the centre layer, and the two `z` neighbours out of the window's
/// `below` / `above`. Those two are the only reads that can leave the band, and
/// an open face obliges them, so a halo narrower than the stencil aborts here.
///
/// # Panics
///
/// [`HALO_MISSING_BELOW`] / [`HALO_MISSING_ABOVE`] when an open `z` face of an
/// owned cell needs a layer the rank does not hold.
fn sweep_slab_colour(
    storage: &mut SlabStorage,
    faces: &SlabFaces,
    stencil: &SlabStencil,
    colour: u32,
) {
    let (nx, ny, nz) = (faces.nx, faces.ny, faces.nz);
    let (k0, k1) = faces.owned();
    for k in k0..k1 {
        let row = stencil.row(k);
        let SweepWindow {
            below,
            centre,
            above,
        } = storage.sweep_window(k);
        for j in 0..ny {
            for i in 0..nx {
                if ((i + j + k) as u32 % 2) != colour {
                    continue;
                }
                let here = i + nx * j;
                let cell = row + here;
                let o = stencil.open[cell];
                let mut acc = Fix128::ZERO;
                if o[0] && i > 0 {
                    acc = acc + centre[here - 1];
                }
                if o[1] && i + 1 < nx {
                    acc = acc + centre[here + 1];
                }
                if o[2] && j > 0 {
                    acc = acc + centre[here - nx];
                }
                if o[3] && j + 1 < ny {
                    acc = acc + centre[here + nx];
                }
                if o[4] && k > 0 {
                    acc = acc + below.expect(HALO_MISSING_BELOW)[here];
                }
                if o[5] && k + 1 < nz {
                    acc = acc + above.expect(HALO_MISSING_ABOVE)[here];
                }
                centre[here] = (acc - stencil.rhs[cell]) * stencil.inv_deg[cell];
            }
        }
    }
}

/// [`subtract_pressure_gradient`] over the faces one rank owns, reading its
/// band and nothing else.
///
/// # Panics
///
/// [`HALO_MISSING_BELOW`] when the Z-face at the bottom of the band needs the
/// pressure one layer below it and the rank does not hold that layer.
fn subtract_slab_pressure_gradient(faces: &mut SlabFaces, storage: &SlabStorage, coeff: Fix128) {
    let (nx, ny, nz) = (faces.nx, faces.ny, faces.nz);
    let (k0, k1) = faces.owned();
    // Row stride inside one X-face layer; see `SlabStencil::build`.
    let u_row = nx + 1;

    for k in k0..k1 {
        let here = storage
            .layer(k)
            .expect("a rank's own layer, which its band holds by construction");
        let (u, u_flags) = faces.u_layer_mut(k);
        for j in 0..ny {
            for i in 0..=nx {
                let at = i + u_row * j;
                if u_flags[at].solid {
                    u[at] = Fix128::ZERO;
                    continue;
                }
                if u_flags[at].inflow {
                    continue;
                }
                let hi = if i < nx {
                    here[i + nx * j]
                } else {
                    Fix128::ZERO
                };
                let lo = if i > 0 {
                    here[i - 1 + nx * j]
                } else {
                    Fix128::ZERO
                };
                u[at] = u[at] - coeff * (hi - lo);
            }
        }
        let (v, v_flags) = faces.v_layer_mut(k);
        for j in 0..=ny {
            for i in 0..nx {
                let at = i + nx * j;
                if v_flags[at].solid {
                    v[at] = Fix128::ZERO;
                    continue;
                }
                if v_flags[at].inflow {
                    continue;
                }
                let hi = if j < ny {
                    here[i + nx * j]
                } else {
                    Fix128::ZERO
                };
                let lo = if j > 0 {
                    here[i + nx * (j - 1)]
                } else {
                    Fix128::ZERO
                };
                v[at] = v[at] - coeff * (hi - lo);
            }
        }
    }

    for k in faces.w_written() {
        let above = if k < nz {
            Some(
                storage
                    .layer(k)
                    .expect("a rank's own layer, which its band holds by construction"),
            )
        } else {
            None
        };
        let below = if k > 0 {
            Some(storage.layer(k - 1).expect(HALO_MISSING_BELOW))
        } else {
            None
        };
        let (w, w_flags) = faces.w_layer_mut(k);
        for j in 0..ny {
            for i in 0..nx {
                let at = i + nx * j;
                if w_flags[at].solid {
                    w[at] = Fix128::ZERO;
                    continue;
                }
                if w_flags[at].inflow {
                    continue;
                }
                let hi = above.map_or(Fix128::ZERO, |layer| layer[at]);
                let lo = below.map_or(Fix128::ZERO, |layer| layer[at]);
                w[at] = w[at] - coeff * (hi - lo);
            }
        }
    }
}

/// The red-black pressure projection over slab-local storage: every rank holds
/// its own layers plus one halo layer, and nothing else.
///
/// Bit-identical to [`project_pressure_red_black_gs`] — not within a tolerance:
/// `Fix128` addition is a group operation mod 2¹²⁸, every value is computed by
/// the same expression in the same order, and a correct decomposition therefore
/// has no error to bound.
/// `slab_local_storage_reproduces_the_monolithic_pressure_solve` pins it.
///
/// # What the caller supplies
///
/// `faces` is one [`SlabFaces`] per rank, in rank order, with the face
/// conditions already imposed (see that type's precondition). `transport` holds
/// one band per rank, already initialised from the field the solve starts from.
/// On return each rank's band holds the final pressure over the layers it owns,
/// and its `faces` hold the corrected velocities over the faces it owns; no
/// rank holds the whole field at any point, which is the difference from
/// [`project_pressure_decomposed_over`].
///
/// A degenerate `dx`, density or step leaves everything untouched, as the
/// full-length drivers do.
pub(crate) fn project_pressure_slab_local_over<T: SlabTransport>(
    faces: &mut [SlabFaces],
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
    schedule: HaloSchedule,
    transport: &mut T,
) {
    let Some(first) = faces.first() else {
        return;
    };
    let (nx, ny, nz, dx) = (first.nx, first.ny, first.nz, first.dx);
    if dx.is_zero() || density_kg_m3.is_zero() || dt_s.is_zero() {
        return;
    }
    let ranks = faces.len();
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    for (r, rank_faces) in faces.iter().enumerate() {
        assert_eq!(
            rank_faces.owned(),
            bounds[r],
            "rank {r} was handed the faces of layers {:?} but the decomposition gives it {:?}",
            rank_faces.owned(),
            bounds[r],
        );
        assert_eq!(
            (rank_faces.nx, rank_faces.ny, rank_faces.nz),
            (nx, ny, nz),
            "rank {r}'s faces describe a different grid from rank 0's",
        );
    }

    let scale = density_kg_m3 * dx * dx / dt_s;
    let stencils: Vec<SlabStencil> = faces
        .iter()
        .map(|rank_faces| SlabStencil::build(rank_faces, scale))
        .collect();

    for _ in 0..iterations {
        for colour in 0..2u32 {
            for r in 0..ranks {
                sweep_slab_colour(transport.slab_mut(r), &faces[r], &stencils[r], colour);
            }
            if schedule == HaloSchedule::EverySweep {
                exchange_slab_halos_local(transport, &bounds, nz);
            }
        }
        if schedule == HaloSchedule::EveryIteration {
            exchange_slab_halos_local(transport, &bounds, nz);
        }
    }

    let inv_dx = Fix128::ONE / dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;
    for (r, rank_faces) in faces.iter_mut().enumerate() {
        subtract_slab_pressure_gradient(rank_faces, transport.slab_mut(r), coeff);
    }
}

// ============================================================================
// Crossing a process boundary with slab-local storage
// ============================================================================
//
// [`project_pressure_slab_local_over`] above shrinks what one rank holds, but it
// still drives every rank from one process: it asks the transport for rank `r`'s
// band for every `r`, so a transport holding one band cannot serve it. What was
// therefore still unmeasured is the thing the hundred-million-cell target needs
// — several processes, each holding one band, agreeing with the single-process
// solve to the bit.
//
// Two crossings have to be made for that, and they are not the same shape.
//
// ## The pressure halo, once per colour sweep
//
// That is the schedule [`exchange_slab_halos_local`] already walks, and it is
// written against [`SlabTransport`], so a rank-local driver reuses it unchanged
// and the only new thing is a transport whose `slab_mut` serves its own rank and
// refuses every other — [`SlabSocketTransport`]. The walk being shared is why
// there is no second sequence to keep in step with the first here;
// `both_exchange_walks_deliver_the_same_layers_in_the_same_order` continues to
// pin the one pair of walks that does exist.
//
// ## The face conditions, once before the solve
//
// [`MacGrid::enforce_face_boundaries`] is a *prefix propagation along `z`*: a
// Z-face marked [`FaceBc::Outflow`] copies the face one cell inside the domain,
// which for the first Z-face of a rank is a layer the rank below owns — and
// copies it *after* that rank has imposed its own conditions, because the
// monolithic loop runs in ascending `k` and reads values it has already
// rewritten. So the enforcement serialises across the ranks: one layer per slab
// boundary, received before a rank enforces and sent after it has. That one
// layer is the halo this stage had to add, and [`PlaneChannel`] is what carries
// it.
//
// The X- and Y-face conditions need no halo: an outflow X-face copies a face in
// the same cell layer, so the propagation there runs inside a rank's band.
//
// ### What that halo is for, since it looks removable
//
// Measured: with the Z-outflow on the *domain boundary* alone — which is where
// [`FaceBc::Outflow`]'s own documentation says the condition is meaningful — no
// rank ever reads outside its band, and the enforcement runs to completion with
// nothing handed over. The reason is structural: the first Z-face a rank writes
// is the one at `k0`, which for `k0 > 0` is an interior layer, while the faces at
// `0` and `nz` are each held by a rank that also holds the layer they read. So
// the halo is load-bearing only when an *interior* Z-face is marked
// [`FaceBc::Outflow`] — a configuration the crate calls meaningless and
// nonetheless accepts.
//
// It stays, because "meaningless" is a promise asked of the caller and not
// something the API refuses, and a decomposition is required to agree with the
// single-process solve over everything the API accepts. The two facts are pinned
// as a pair so that this paragraph cannot rot into a reason to delete the halo:
// `a_z_outflow_on_the_domain_boundary_alone_needs_no_face_halo` fixes that the
// boundary case needs nothing, and
// `an_interior_z_outflow_is_what_makes_the_face_halo_load_bearing` fixes that the
// interior case aborts without it. Removing the halo leaves the second one red.
//
// ## Why the top Z-face layer is recomputed instead of carried back
//
// A rank owning `k0..k1` holds Z-faces `k0..=k1` — one more than it writes,
// because its own divergence needs the face above its last cell — and the extra
// one is written by the rank above. That looks like a second crossing in the
// opposite direction: the rank would have to learn the *enforced* value of layer
// `k1` from its neighbour. It does not. The condition on layer `k1` is one the
// rank already holds, and the only value that condition can read is layer
// `k1 − 1`, which the rank has just enforced itself. So each rank enforces
// `k0..=k1` — the whole band it holds, its own top layer included — and the two
// ranks sharing a layer compute the same value out of the same expression over
// the same input. One redundant plane of arithmetic per boundary buys the
// absence of a second message, and
// `both_ranks_sharing_a_z_face_layer_enforce_it_to_the_same_value` pins that the
// two agree rather than assuming it.

/// A rank was asked to impose the conditions on its first Z-face, that face is
/// an outflow, and the layer below it — which the rank below owns — was not
/// handed over.
///
/// Named once because `#[should_panic]` matches on the text.
pub(crate) const SLAB_FACE_HALO_MISSING: &str =
    "slab face conditions: the first Z-face of this rank is an outflow and copies the face \
     one layer below the slab, which the rank below has to hand over";

/// The face conditions on the faces one rank holds, kept the way `MacGrid`
/// keeps them.
///
/// [`FaceFlags`] carries the three predicates the *solve* needs, which is all a
/// stencil ever asks for. Imposing the conditions needs more than predicates:
/// the prescribed velocity of an [`FaceBc::Inflow`] is a value, and
/// [`FaceBc::Outflow`] is not distinguishable from plain fluid by any of the
/// three. Carrying the condition densely would cost more than the field it
/// describes — a [`FaceBc`] is wider than the [`Fix128`] velocity on the face —
/// so it is carried the way [`MacGrid`] carries it: sparsely, for the faces that
/// are neither plain fluid nor a wall at rest, with the dense solid flag of
/// [`FaceFlags`] standing for the rest. The precedence between the two is
/// `read_face_bc`'s, because that is the function
/// [`MacGrid::enforce_face_boundaries`] reads its conditions through.
///
/// Keyed by the index inside the rank's own face array and kept ascending in it,
/// which is the order [`enforce_slab_face_boundaries`] walks the faces in and
/// the order the monolithic enforcement walks the whole domain in. That order is
/// load-bearing: an outflow face reads a neighbour that may itself be an outflow
/// face, and whether the neighbour has already been rewritten is part of the
/// answer.
pub(crate) struct SlabFaceConditions {
    nx: usize,
    ny: usize,
    /// First owned cell layer.
    k0: usize,
    /// One past the last owned cell layer.
    k1: usize,
    /// `(index into the rank's X-face array, condition)`, ascending.
    u: Vec<(usize, FaceBc)>,
    /// Y-faces; see `u`.
    v: Vec<(usize, FaceBc)>,
    /// Z-faces of layers `k0..=k1`; see `u`.
    w: Vec<(usize, FaceBc)>,
}

impl SlabFaceConditions {
    /// No conditions at all: every face plain fluid, every wall described by the
    /// dense flag alone.
    ///
    /// The constructor a rank with no [`MacGrid`] starts from; it then names the
    /// faces that differ with [`SlabFaceConditions::set_u`] and its siblings.
    pub(crate) fn new(nx: usize, ny: usize, (k0, k1): (usize, usize)) -> Self {
        Self {
            nx,
            ny,
            k0,
            k1,
            u: Vec::new(),
            v: Vec::new(),
            w: Vec::new(),
        }
    }

    /// The conditions on the faces of layers `k0..k1` (and the Z-faces
    /// `k0..=k1`), copied out of `grid`'s sparse maps.
    ///
    /// The maps are keyed by the global face index, and one `z` layer of each
    /// face field is a contiguous run of those keys, so a rank's entries are one
    /// range of the map — taken in ascending order, which is the order the
    /// enforcement wants them in.
    pub(crate) fn from_grid(grid: &MacGrid, (k0, k1): (usize, usize)) -> Self {
        let mut out = Self::new(grid.nx, grid.ny, (k0, k1));
        if k0 == k1 {
            return out;
        }
        let (up, vp, cp) = (
            u_plane(grid.nx, grid.ny),
            v_plane(grid.nx, grid.ny),
            cell_plane(grid.nx, grid.ny),
        );
        for (&ix, &bc) in grid.u_face_bc.range(k0 * up..k1 * up) {
            out.u.push((ix - k0 * up, bc));
        }
        for (&ix, &bc) in grid.v_face_bc.range(k0 * vp..k1 * vp) {
            out.v.push((ix - k0 * vp, bc));
        }
        for (&ix, &bc) in grid.w_face_bc.range(k0 * cp..(k1 + 1) * cp) {
            out.w.push((ix - k0 * cp, bc));
        }
        out
    }

    /// Name the condition on the X-face `(i, j, k)` of this rank.
    ///
    /// Keeps the same entries `MacGrid` would: plain fluid and a wall at rest
    /// are what the dense flag already says, so naming either of those removes
    /// the entry instead of storing it.
    ///
    /// # Panics
    ///
    /// When `k` is not an owned layer, or `(i, j)` is not a face of one.
    pub(crate) fn set_u(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        assert!(i <= self.nx && j < self.ny, "({i}, {j}) is not an X-face");
        let at = self.owned_base(k, u_plane(self.nx, self.ny)) + i + (self.nx + 1) * j;
        Self::write(&mut self.u, at, bc);
    }

    /// See [`SlabFaceConditions::set_u`].
    pub(crate) fn set_v(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        assert!(i < self.nx && j <= self.ny, "({i}, {j}) is not a Y-face");
        let at = self.owned_base(k, v_plane(self.nx, self.ny)) + i + self.nx * j;
        Self::write(&mut self.v, at, bc);
    }

    /// See [`SlabFaceConditions::set_u`]. Z-faces run one past the owned
    /// layers, so `k` may be `k1`.
    pub(crate) fn set_w(&mut self, i: usize, j: usize, k: usize, bc: FaceBc) {
        assert!(i < self.nx && j < self.ny, "({i}, {j}) is not a Z-face");
        assert!(
            k >= self.k0 && k <= self.k1 && self.k0 != self.k1,
            "Z-face layer {k} is not held by this slab (holds {}..={})",
            self.k0,
            self.k1,
        );
        let at = (k - self.k0) * cell_plane(self.nx, self.ny) + i + self.nx * j;
        Self::write(&mut self.w, at, bc);
    }

    /// Offset of owned layer `k` inside an array of `plane` values per layer.
    fn owned_base(&self, k: usize, plane: usize) -> usize {
        assert!(
            k >= self.k0 && k < self.k1,
            "layer {k} is not owned by this slab (owns {}..{})",
            self.k0,
            self.k1,
        );
        (k - self.k0) * plane
    }

    /// Insert, replace or remove one entry, keeping the list ascending.
    fn write(entries: &mut Vec<(usize, FaceBc)>, at: usize, bc: FaceBc) {
        let default_wall = FaceBc::Wall {
            velocity: Vec3Fix::ZERO,
        };
        let slot = entries.partition_point(|&(other, _)| other < at);
        let present = entries.get(slot).is_some_and(|&(other, _)| other == at);
        if bc == FaceBc::Fluid || bc == default_wall {
            if present {
                entries.remove(slot);
            }
        } else if present {
            entries[slot].1 = bc;
        } else {
            entries.insert(slot, (at, bc));
        }
    }
}

/// The entries of a [`SlabFaceConditions`] list that belong to one layer.
///
/// The list is ascending in the index and one layer is a contiguous run of
/// indices, so the layer's entries are a sub-slice found without scanning the
/// rest.
fn layer_conditions(entries: &[(usize, FaceBc)], base: usize, plane: usize) -> &[(usize, FaceBc)] {
    let lo = entries.partition_point(|&(at, _)| at < base);
    let hi = entries.partition_point(|&(at, _)| at < base + plane);
    &entries[lo..hi]
}

/// The condition on face `at`, for a walk whose `at` only increases.
///
/// `next` is the walk's position in `entries`; faces with no entry fall back to
/// the dense solid flag exactly as `read_face_bc` does.
#[inline]
fn condition_at(entries: &[(usize, FaceBc)], next: &mut usize, at: usize, solid: bool) -> FaceBc {
    while *next < entries.len() && entries[*next].0 < at {
        *next += 1;
    }
    match entries.get(*next) {
        Some(&(other, bc)) if other == at => bc,
        _ => wall_or_fluid(solid),
    }
}

/// Impose the face conditions on the faces one rank holds — what
/// [`MacGrid::enforce_face_boundaries`] does, restricted to a band.
///
/// `below_w` is the Z-face layer `k0 − 1` *after* the rank below imposed its own
/// conditions, or `None` for a rank whose band starts at layer 0. Supplying it
/// is the caller's job; [`enforce_slab_face_boundaries_on_rank`] receives it
/// over a [`PlaneChannel`].
///
/// Every value is written by the same expression in the same order as the
/// monolithic enforcement, so the two agree to the bit rather than within a
/// tolerance — which `distributed_face_enforcement_matches_the_monolithic_one`
/// pins. The order is the reason the conditions are walked face by face instead
/// of entry by entry: a face whose only condition is the dense solid flag has no
/// entry, and an outflow face reads a neighbour whose own turn may be earlier or
/// later than its own.
///
/// # Panics
///
/// [`SLAB_FACE_HALO_MISSING`] when the first Z-face this rank writes is an
/// outflow and `below_w` is `None`. When `cond` describes a different band or a
/// different grid from `faces`.
pub(crate) fn enforce_slab_face_boundaries(
    faces: &mut SlabFaces,
    cond: &SlabFaceConditions,
    below_w: Option<&[Fix128]>,
) {
    let (k0, k1) = faces.owned();
    assert_eq!(
        (cond.k0, cond.k1),
        (k0, k1),
        "the conditions describe layers {:?} and the faces {:?}",
        (cond.k0, cond.k1),
        (k0, k1),
    );
    assert_eq!(
        (cond.nx, cond.ny),
        (faces.nx, faces.ny),
        "the conditions describe a different grid from the faces",
    );
    if k0 == k1 {
        return; // owns nothing, so holds no face to impose anything on
    }
    let (nx, ny) = (faces.nx, faces.ny);
    let u_row = nx + 1;

    // X-faces, `MacGrid::enforce_face_boundaries`'s first loop over this rank's
    // layers. An outflow X-face copies a face of the same layer, so this stays
    // inside the band.
    let up = u_plane(nx, ny);
    for k in k0..k1 {
        let base = (k - k0) * up;
        let entries = layer_conditions(&cond.u, base, up);
        let mut next = 0usize;
        let (u, flags) = faces.u_layer_mut(k);
        for j in 0..ny {
            for i in 0..=nx {
                let at = i + u_row * j;
                match condition_at(entries, &mut next, base + at, flags[at].solid) {
                    FaceBc::Fluid => {}
                    FaceBc::Inflow { normal_velocity } => u[at] = normal_velocity,
                    FaceBc::Outflow => {
                        let inner = if i > 0 { i - 1 } else { i + 1 };
                        u[at] = if inner <= nx {
                            u[inner + u_row * j]
                        } else {
                            Fix128::ZERO
                        };
                    }
                    _ => u[at] = Fix128::ZERO,
                }
            }
        }
    }

    // Y-faces, the second loop. Also layer-local.
    let vp = v_plane(nx, ny);
    for k in k0..k1 {
        let base = (k - k0) * vp;
        let entries = layer_conditions(&cond.v, base, vp);
        let mut next = 0usize;
        let (v, flags) = faces.v_layer_mut(k);
        for j in 0..=ny {
            for i in 0..nx {
                let at = i + nx * j;
                match condition_at(entries, &mut next, base + at, flags[at].solid) {
                    FaceBc::Fluid => {}
                    FaceBc::Inflow { normal_velocity } => v[at] = normal_velocity,
                    FaceBc::Outflow => {
                        let inner = if j > 0 { j - 1 } else { j + 1 };
                        v[at] = if inner <= ny {
                            v[i + nx * inner]
                        } else {
                            Fix128::ZERO
                        };
                    }
                    _ => v[at] = Fix128::ZERO,
                }
            }
        }
    }

    // Z-faces, the third loop, over every layer the rank holds — its own top
    // layer included, which the rank above also enforces (see the section
    // header). This is the loop that leaves the band: the face at `k0` reads the
    // face at `k0 − 1`.
    let cp = cell_plane(nx, ny);
    let mut source = Vec::new();
    for k in k0..=k1 {
        let base = (k - k0) * cp;
        let entries = layer_conditions(&cond.w, base, cp);
        // The read is set up before the layer is borrowed for writing, and only
        // when an outflow face in this layer obliges it: the source is another
        // layer of the same array, or the one the rank below handed over.
        if entries.iter().any(|&(_, bc)| bc == FaceBc::Outflow) {
            if source.is_empty() {
                source = vec![Fix128::ZERO; cp];
            }
            let inner = if k > 0 { k - 1 } else { k + 1 };
            if inner < k0 {
                source.copy_from_slice(below_w.expect(SLAB_FACE_HALO_MISSING));
            } else {
                assert!(
                    inner <= k1,
                    "the face below Z-face layer {k} is layer {inner}, which is above the \
                     band {k0}..={k1}: only a rank owning nothing could reach this, and one \
                     has already returned",
                );
                source.copy_from_slice(faces.w_layer(inner).0);
            }
        }
        let mut next = 0usize;
        let (w, flags) = faces.w_layer_mut(k);
        for j in 0..ny {
            for i in 0..nx {
                let at = i + nx * j;
                match condition_at(entries, &mut next, base + at, flags[at].solid) {
                    FaceBc::Fluid => {}
                    FaceBc::Inflow { normal_velocity } => w[at] = normal_velocity,
                    FaceBc::Outflow => w[at] = source[at],
                    _ => w[at] = Fix128::ZERO,
                }
            }
        }
    }
}

/// Carries one `nx · ny` plane of face velocities from the rank that owns it to
/// the rank that reads it.
///
/// Separate from [`SlabTransport`] because it moves a different thing at a
/// different time: the pressure halo crosses once per colour sweep and its
/// destination is a band the transport owns, while this crosses once per solve
/// and its destination is a buffer the caller owns. One backend implements both
/// over the same links ([`SlabSocketTransport`]), which is how a rank ends up
/// with one object and two schedules rather than two connections.
///
/// A plane carries no header; the two ends are matched by their position in the
/// pipeline the enforcement walks, which is the same arrangement
/// [`SocketTransport`] relies on for the pressure halo.
pub(crate) trait PlaneChannel {
    /// Hand rank `dst` the plane of Z-face layer `layer`.
    fn send_plane(&mut self, dst: usize, layer: usize, plane: &[Fix128]);

    /// Take the plane of Z-face layer `layer` from rank `src` into `plane`.
    fn recv_plane(&mut self, src: usize, layer: usize, plane: &mut [Fix128]);
}

/// [`PlaneChannel`] for ranks sharing one address space: the plane is copied
/// through a one-layer buffer, which is the shape the wire backend has too.
///
/// The pipeline the enforcement walks never has two planes in flight — a rank
/// sends only after the rank below has been received from — so one buffer is
/// enough, and the assertions here say so rather than leaving a second send to
/// overwrite the first unnoticed.
pub(crate) struct LocalPlaneChannel {
    /// The plane in flight.
    staged: Vec<Fix128>,
    /// Which Z-face layer `staged` holds, or `None` when nothing is in flight.
    layer: Option<usize>,
}

impl LocalPlaneChannel {
    /// A channel for planes of `plane` values, with nothing in flight.
    pub(crate) fn new(plane: usize) -> Self {
        Self {
            staged: vec![Fix128::ZERO; plane],
            layer: None,
        }
    }

    /// Has every plane that was sent been received?
    pub(crate) fn is_drained(&self) -> bool {
        self.layer.is_none()
    }
}

impl PlaneChannel for LocalPlaneChannel {
    fn send_plane(&mut self, _dst: usize, layer: usize, plane: &[Fix128]) {
        assert!(
            self.layer.is_none(),
            "layer {layer} was offered while layer {:?} was still in flight",
            self.layer,
        );
        self.staged.copy_from_slice(plane);
        self.layer = Some(layer);
    }

    fn recv_plane(&mut self, _src: usize, layer: usize, plane: &mut [Fix128]) {
        let staged = self
            .layer
            .take()
            .expect("a rank asked for a plane that no rank had offered");
        assert_eq!(
            staged, layer,
            "a rank asked for layer {layer} and layer {staged} was in flight",
        );
        plane.copy_from_slice(&self.staged);
    }
}

/// Impose the face conditions on one rank's faces, receiving the one layer the
/// enforcement reads from across the boundary below and sending on the one layer
/// the rank above will read.
///
/// The crossing happens whether or not this rank's first Z-face is an outflow:
/// the schedule is a function of the decomposition alone, so two ranks cannot
/// come to different conclusions about whether a message is on the wire. Ranks
/// owning nothing take part in neither half, and the rank below a gap is the one
/// that *owns* the layer rather than the one with the previous index, which is
/// how a decomposition with empty ranks still forms a chain.
///
/// # Panics
///
/// When `faces` does not describe the layers the decomposition gives `my_rank`.
pub(crate) fn enforce_slab_face_boundaries_on_rank<C: PlaneChannel>(
    faces: &mut SlabFaces,
    cond: &SlabFaceConditions,
    ranks: usize,
    my_rank: usize,
    channel: &mut C,
) {
    if ranks == 0 || my_rank >= ranks {
        return;
    }
    let nz = faces.nz;
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    let (k0, k1) = bounds[my_rank];
    assert_eq!(
        faces.owned(),
        (k0, k1),
        "rank {my_rank} was handed the faces of layers {:?} but the decomposition gives it {:?}",
        faces.owned(),
        (k0, k1),
    );
    if k0 == k1 {
        return; // owns nothing: nothing to impose, nothing to pass on
    }
    let plane = cell_plane(faces.nx, faces.ny);

    let mut below = Vec::new();
    if k0 > 0 {
        let src = slab_owner(&bounds, k0 - 1)
            .expect("the layer below an owned layer is owned by some rank");
        below = vec![Fix128::ZERO; plane];
        channel.recv_plane(src, k0 - 1, &mut below);
    }
    let below_w = if k0 > 0 { Some(&below[..]) } else { None };

    enforce_slab_face_boundaries(faces, cond, below_w);

    if let Some(dst) = slab_owner(&bounds, k1) {
        let (w, _) = faces.w_layer(k1 - 1);
        channel.send_plane(dst, k1 - 1, w);
    }
}

/// [`enforce_slab_face_boundaries_on_rank`] for every rank of an in-process
/// decomposition, in the order the pipeline requires.
///
/// Drives the rank-local function rather than restating the pipeline, so there
/// is one description of the order and not two.
///
/// # Panics
///
/// When a plane is left in flight at the end, which would mean a rank sent one
/// that no rank was going to read.
pub(crate) fn enforce_slab_face_boundaries_over(
    faces: &mut [SlabFaces],
    cond: &[SlabFaceConditions],
) {
    let Some(first) = faces.first() else {
        return;
    };
    let plane = cell_plane(first.nx, first.ny);
    let ranks = faces.len();
    assert_eq!(
        cond.len(),
        ranks,
        "{} sets of conditions for {ranks} ranks",
        cond.len(),
    );
    let mut channel = LocalPlaneChannel::new(plane);
    for (r, rank_faces) in faces.iter_mut().enumerate() {
        enforce_slab_face_boundaries_on_rank(rank_faces, &cond[r], ranks, r, &mut channel);
    }
    assert!(
        channel.is_drained(),
        "a Z-face layer was passed on to a rank that never read it",
    );
}

/// One [`Fix128`] on the wire: `hi` then `lo`, little-endian — the encoding
/// [`SocketTransport`] fixes, reused so that the two transports put the same
/// bytes on the same wire.
///
/// `the_two_wire_encodings_agree` pins that they do, by comparing what one
/// transport writes with what this function produces.
#[cfg(feature = "std")]
fn encode_plane(values: &[Fix128], wire: &mut [u8]) {
    for (cell, chunk) in values
        .iter()
        .zip(wire.chunks_exact_mut(WIRE_BYTES_PER_CELL))
    {
        chunk[..8].copy_from_slice(&cell.hi.to_le_bytes());
        chunk[8..].copy_from_slice(&cell.lo.to_le_bytes());
    }
}

/// Inverse of [`encode_plane`].
#[cfg(feature = "std")]
fn decode_plane(wire: &[u8], values: &mut [Fix128]) {
    for (cell, chunk) in values
        .iter_mut()
        .zip(wire.chunks_exact(WIRE_BYTES_PER_CELL))
    {
        let hi = i64::from_le_bytes(chunk[..8].try_into().expect("8 bytes of a 16-byte cell"));
        let lo = u64::from_le_bytes(chunk[8..].try_into().expect("8 bytes of a 16-byte cell"));
        *cell = Fix128::from_raw(hi, lo);
    }
}

/// [`SlabTransport`] and [`PlaneChannel`] for ranks in *different address
/// spaces*, each holding one band and nothing else.
///
/// # What this settles that [`LocalSlabTransport`] cannot
///
/// `LocalSlabTransport` holds every rank's band, so it cannot tell a rank-local
/// driver from one that reads a neighbour's storage: both work. This one has no
/// neighbour's storage to read, and [`SlabTransport::slab_mut`] serves its own
/// rank alone, so a driver that asked for another rank's band aborts here
/// instead of returning a plausible answer. It is also the first transport for
/// which a halo layer is bytes on a wire while the band it lands in is *not* the
/// whole field — the combination the hundred-million-cell target needs and that
/// neither [`SocketTransport`] (whole field) nor `LocalSlabTransport` (one
/// address space) reaches on its own.
///
/// `S` is any paired byte stream, because the only thing fixed here is the
/// encoding ([`encode_plane`]).
///
/// # Lockstep
///
/// Deliveries carry no header and are matched by their position in the schedule
/// every rank walks. A delivery this rank is not party to is a no-op, which is
/// what keeps the positions the same; the same holds for a rank that owns
/// nothing and therefore takes part in no delivery at all.
#[cfg(feature = "std")]
pub(crate) struct SlabSocketTransport<S> {
    /// The rank this process is; the only one `slab_mut` serves.
    my_rank: usize,
    /// `nx · ny`, the values in one layer and so in one message.
    plane: usize,
    /// This rank's band: its owned layers plus its halo, and nothing else.
    slab: SlabStorage,
    /// `links[r]` is the stream to rank `r`; this rank's own entry is `None`.
    links: Vec<Option<S>>,
    /// Encode/decode scratch for one layer, sized `plane · 16`.
    wire: Vec<u8>,
}

#[cfg(feature = "std")]
impl<S> SlabSocketTransport<S> {
    /// A rank holding `slab`, with `links[r]` the stream to rank `r`.
    pub(crate) fn new(
        my_rank: usize,
        plane: usize,
        slab: SlabStorage,
        links: Vec<Option<S>>,
    ) -> Self {
        Self {
            my_rank,
            plane,
            slab,
            links,
            wire: vec![0u8; plane * WIRE_BYTES_PER_CELL],
        }
    }

    /// This rank's band, for reading back the layers it owns.
    pub(crate) fn slab(&self) -> &SlabStorage {
        &self.slab
    }
}

#[cfg(feature = "std")]
impl<S: std::io::Read + std::io::Write> SlabTransport for SlabSocketTransport<S> {
    fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage {
        assert_eq!(
            rank, self.my_rank,
            "rank {} was asked for rank {rank}'s band, which is in another address space: \
             the driver is not rank-local",
            self.my_rank,
        );
        &mut self.slab
    }

    fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
        if src == dst {
            return;
        }
        if self.my_rank == src {
            let from = self.slab.layer(layer).expect(DELIVERY_SOURCE_LACKS_LAYER);
            encode_plane(from, &mut self.wire);
            let link = self.links[dst]
                .as_mut()
                .expect("no stream to the rank this delivery is addressed to");
            // A failed halo delivery leaves the ranks disagreeing about the
            // field, which no later step can repair, so this is fatal by design.
            link.write_all(&self.wire)
                .expect("send a halo layer to the peer rank");
            link.flush().expect("flush a halo layer to the peer rank");
        } else if self.my_rank == dst {
            let link = self.links[src]
                .as_mut()
                .expect("no stream from the rank this delivery comes from");
            link.read_exact(&mut self.wire)
                .expect("receive a halo layer from the peer rank");
            let into = self
                .slab
                .layer_mut(layer)
                .expect(DELIVERY_DESTINATION_LACKS_LAYER);
            decode_plane(&self.wire, into);
        }
        // Neither end: another pair's delivery, counted but not performed.
    }
}

#[cfg(feature = "std")]
impl<S: std::io::Read + std::io::Write> PlaneChannel for SlabSocketTransport<S> {
    fn send_plane(&mut self, dst: usize, layer: usize, plane: &[Fix128]) {
        assert_eq!(
            plane.len(),
            self.plane,
            "layer {layer} is {} values and a plane on this wire is {}",
            plane.len(),
            self.plane,
        );
        encode_plane(plane, &mut self.wire);
        let link = self.links[dst]
            .as_mut()
            .expect("no stream to the rank this plane is addressed to");
        link.write_all(&self.wire)
            .expect("send a Z-face layer to the peer rank");
        link.flush().expect("flush a Z-face layer to the peer rank");
    }

    fn recv_plane(&mut self, src: usize, layer: usize, plane: &mut [Fix128]) {
        assert_eq!(
            plane.len(),
            self.plane,
            "layer {layer} is {} values and a plane on this wire is {}",
            plane.len(),
            self.plane,
        );
        let link = self.links[src]
            .as_mut()
            .expect("no stream from the rank this plane comes from");
        link.read_exact(&mut self.wire)
            .expect("receive a Z-face layer from the peer rank");
        decode_plane(&self.wire, plane);
    }
}

/// The red-black pressure projection as *one rank* runs it: this rank's band,
/// this rank's faces, and a transport that can reach no other rank's storage.
///
/// [`project_pressure_slab_local_over`] is the same solve driven from one
/// process, and it asks the transport for every rank's band in turn, which a
/// rank-local transport refuses. This is the form a distributed run takes: every
/// rank calls it with its own `my_rank`, the sweeps happen concurrently, and the
/// exchange between two sweeps is what puts them back in step. The schedule walk
/// is [`exchange_slab_halos_local`], shared with the in-process driver, so there
/// is no second sequence that could drift from the first.
///
/// Bit-identical to [`project_pressure_red_black_gs`] over the layers this rank
/// owns — not within a tolerance, for the reason the in-process driver gives.
/// `three_processes_reproduce_the_monolithic_slab_local_solve` pins it across
/// real process boundaries.
///
/// # What the caller supplies
///
/// `faces` holds this rank's face velocities with the conditions already imposed
/// (by [`enforce_slab_face_boundaries_on_rank`], which is the distributed form of
/// that step). `transport` holds this rank's band, already initialised from the
/// field the solve starts from — its halo layers included, because the first
/// colour sweep reads them before any exchange has happened. On return the band
/// holds the final pressure over the layers this rank owns and `faces` holds the
/// corrected velocities over the faces it writes. Nothing gathers: no rank holds
/// the whole field at any point, which is the difference from
/// [`project_pressure_decomposed_on_rank`].
///
/// A degenerate `dx`, density or step leaves everything untouched, as the other
/// drivers do.
///
/// # Panics
///
/// When `faces` does not describe the layers the decomposition gives `my_rank`.
// One more parameter than `project_pressure_slab_local_over`, whose list this
// deliberately mirrors so the two drivers stay comparable; `my_rank` is the
// whole difference between them, as it is between the two full-length drivers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pressure_slab_local_on_rank<T: SlabTransport>(
    faces: &mut SlabFaces,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    iterations: u32,
    ranks: usize,
    schedule: HaloSchedule,
    my_rank: usize,
    transport: &mut T,
) {
    if faces.dx.is_zero()
        || density_kg_m3.is_zero()
        || dt_s.is_zero()
        || ranks == 0
        || my_rank >= ranks
    {
        return;
    }
    let nz = faces.nz;
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    assert_eq!(
        faces.owned(),
        bounds[my_rank],
        "rank {my_rank} was handed the faces of layers {:?} but the decomposition gives it {:?}",
        faces.owned(),
        bounds[my_rank],
    );

    let scale = density_kg_m3 * faces.dx * faces.dx / dt_s;
    let stencil = SlabStencil::build(faces, scale);

    for _ in 0..iterations {
        for colour in 0..2u32 {
            sweep_slab_colour(transport.slab_mut(my_rank), faces, &stencil, colour);
            if schedule == HaloSchedule::EverySweep {
                exchange_slab_halos_local(transport, &bounds, nz);
            }
        }
        if schedule == HaloSchedule::EveryIteration {
            exchange_slab_halos_local(transport, &bounds, nz);
        }
    }

    let inv_dx = Fix128::ONE / faces.dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;
    subtract_slab_pressure_gradient(faces, transport.slab_mut(my_rank), coeff);
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

/// Particle-to-grid with weight normalisation: every face of `grid` that at
/// least one particle reaches ends up holding the trilinear-weighted **mean**
/// of the particle velocities, so a uniform particle velocity gives a uniform
/// face velocity. Faces no particle reaches keep their previous value.
///
/// `particles` is a slice of `(position_m, velocity_m_per_s)`.
/// A particle with any negative coordinate, or one beyond the far side, lies
/// outside the domain and contributes nothing.
///
/// `p2g_trilinear` only accumulates `weight * v` and keeps no weight sum, so
/// it cannot produce a mean. This entry point runs it twice on scratch grids,
/// once with the velocities and once with unit velocity, so the second pass
/// leaves the per-face weight sum and the quotient is the mean. Reusing the
/// deposit routines keeps the stagger offsets and the out-of-range rule in
/// one place.
pub fn p2g_normalized(grid: &mut MacGrid, particles: &[(Vec3Fix, Vec3Fix)]) {
    if grid.dx.is_zero() {
        return;
    }
    let mut num = MacGrid::new(grid.nx, grid.ny, grid.nz, grid.dx);
    let mut den = MacGrid::new(grid.nx, grid.ny, grid.nz, grid.dx);
    let one = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    for &(pos, vel) in particles {
        // A particle left of the domain is dropped, as one beyond the far side
        // already is. `split` clamps a negative coordinate to face 0 with full
        // weight, which `g2p_velocity` and the advection back-trace rely on, but
        // here it would overwrite the corner face with the velocity of a particle
        // that is nowhere near it.
        if pos.x < Fix128::ZERO || pos.y < Fix128::ZERO || pos.z < Fix128::ZERO {
            continue;
        }
        p2g_trilinear(&mut num, pos, vel);
        p2g_trilinear(&mut den, pos, one);
    }
    let divide = |dst: &mut [Fix128], n: &[Fix128], d: &[Fix128]| {
        for ((out, &n), &d) in dst.iter_mut().zip(n).zip(d) {
            if !d.is_zero() {
                *out = n / d;
            }
        }
    };
    divide(&mut grid.u, &num.u, &den.u);
    divide(&mut grid.v, &num.v, &den.v);
    divide(&mut grid.w, &num.w, &den.w);
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

    /// Splitting the domain into `z` slabs that exchange one halo layer after
    /// every colour sweep reproduces the monolithic solve exactly — for rank
    /// counts that divide the depth and for ones that do not, including a split
    /// fine enough to leave a rank with nothing to own.
    ///
    /// Exactness, not a tolerance: `Fix128` addition is a group operation mod
    /// 2¹²⁸, so a decomposition that is correct at all is correct to the bit.
    #[test]
    fn slab_decomposition_reproduces_the_monolithic_pressure_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);

        // (grid size, ranks): even splits, uneven splits, and `ranks > nz`
        // (which leaves rank 0 owning nothing).
        for &(n, ranks) in &[
            (8usize, 1usize),
            (8, 2),
            (8, 4),
            (8, 8),
            (7, 2),
            (7, 3),
            (7, 4),
            (5, 4),
            (3, 4),
        ] {
            let base = seed_divergent_flow(n);

            let mut monolithic = base.clone();
            project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

            let mut split = base.clone();
            project_pressure_decomposed(&mut split, dt, rho, 6, ranks, HaloSchedule::EverySweep);

            assert!(
                grids_are_bit_equal(&monolithic, &split),
                "{n}³ grid split across {ranks} slabs did not reproduce the \
                 monolithic solve: either one halo layer is not enough for the \
                 7-point stencil, or a rank read past its halo",
            );
        }
    }

    /// Teeth for the test above: delay the exchange by one sweep and the slabs
    /// stop agreeing with the monolithic solve.
    ///
    /// Without this, `slab_decomposition_reproduces_the_monolithic_pressure_solve`
    /// could pass on a decomposition that never actually depended on the
    /// exchange arriving on time.
    #[test]
    fn a_halo_exchanged_once_per_iteration_diverges_from_the_monolithic_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_divergent_flow(8);

        let mut monolithic = base.clone();
        project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

        let mut stale = base.clone();
        project_pressure_decomposed(&mut stale, dt, rho, 6, 4, HaloSchedule::EveryIteration);

        assert!(
            !grids_are_bit_equal(&monolithic, &stale),
            "a halo one sweep out of date still reproduced the monolithic solve, \
             so the bit-equality test above is not actually testing the exchange",
        );
    }

    /// A second [`RankTransport`] with a different internal shape: every slab
    /// in one flat rank-major allocation, and a delivery staged through an
    /// explicit mailbox — post, then collect — the way a message-passing
    /// backend is obliged to stage one, instead of copied straight across.
    struct StagedTransport {
        cells: usize,
        plane: usize,
        flat: Vec<Fix128>,
        mailbox: Vec<Fix128>,
    }

    impl StagedTransport {
        fn new(ranks: usize, cells: usize, plane: usize) -> Self {
            Self {
                cells,
                plane,
                flat: vec![Fix128::ZERO; ranks * cells],
                mailbox: vec![Fix128::ZERO; plane],
            }
        }
    }

    impl RankTransport for StagedTransport {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
            let base = rank * self.cells;
            &mut self.flat[base..base + self.cells]
        }

        fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
            let off = layer * self.plane;
            let from = src * self.cells + off;
            self.mailbox
                .copy_from_slice(&self.flat[from..from + self.plane]);
            let to = dst * self.cells + off;
            self.flat[to..to + self.plane].copy_from_slice(&self.mailbox);
        }
    }

    /// Control for the test below. Same storage, but a delivery never arrives.
    ///
    /// Not a stub standing in for unwritten code: the empty body *is* the
    /// measurement, and the test that uses it asserts the solve comes out
    /// wrong.
    struct UndeliveredTransport(LocalTransport);

    impl RankTransport for UndeliveredTransport {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
            self.0.slab_mut(rank)
        }

        fn deliver_layer(&mut self, _src: usize, _dst: usize, _layer: usize) {}
    }

    /// The transport is a seam, not an implementation detail: a second
    /// `RankTransport` with a different internal layout and an explicit staging
    /// step produces the same pressure field, bit for bit, as the in-process
    /// one — which is what lets an MPI backend be substituted without
    /// re-pinning any physics.
    ///
    /// Exactness rather than a tolerance, for the same reason as the
    /// monolithic comparison: `Fix128` addition is a group operation mod 2¹²⁸,
    /// so a transport that is right at all is right to the bit.
    #[test]
    fn a_second_rank_transport_reproduces_the_in_process_one_bit_for_bit() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);

        // Even splits, uneven splits, and a split with an empty rank.
        for &(n, ranks) in &[(8usize, 2usize), (8, 4), (8, 8), (7, 3), (5, 4), (3, 4)] {
            let base = seed_divergent_flow(n);

            let mut via_local = base.clone();
            project_pressure_decomposed(
                &mut via_local,
                dt,
                rho,
                6,
                ranks,
                HaloSchedule::EverySweep,
            );

            let mut via_staged = base.clone();
            let mut staged = StagedTransport::new(ranks, n * n * n, n * n);
            project_pressure_decomposed_over(
                &mut via_staged,
                dt,
                rho,
                6,
                ranks,
                HaloSchedule::EverySweep,
                &mut staged,
            );

            assert!(
                grids_are_bit_equal(&via_local, &via_staged),
                "{n}³ grid over {ranks} slabs: the staged transport disagreed with \
                 the in-process one, so the solve depends on how a layer is \
                 carried and not only on which layer arrives when",
            );
        }
    }

    /// Teeth for the test above: with the deliveries removed the slabs keep the
    /// poison in their halos, so the answer must move.
    ///
    /// Without this, `a_second_rank_transport_reproduces_the_in_process_one_bit_for_bit`
    /// could be comparing two runs that agree merely because both started from
    /// the same field, with `deliver_layer` carrying nothing.
    #[test]
    fn a_transport_that_never_delivers_does_not_reproduce_the_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_divergent_flow(8);

        let mut monolithic = base.clone();
        project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

        let mut undelivered = base.clone();
        let mut transport = UndeliveredTransport(LocalTransport::new(4, 8, 8 * 8));
        project_pressure_decomposed_over(
            &mut undelivered,
            dt,
            rho,
            6,
            4,
            HaloSchedule::EverySweep,
            &mut transport,
        );

        assert!(
            !grids_are_bit_equal(&monolithic, &undelivered),
            "slabs that never received a halo still reproduced the monolithic \
             solve, so the transport tests above are not measuring the delivery",
        );
    }

    // ---- A-3.2b: the same decomposition across a process boundary ----------
    //
    // Everything above runs the ranks inside one process, so it pins that two
    // *implementations* of `RankTransport` agree. What it cannot pin is whether
    // the trait can leave an address space at all: `LocalTransport` and
    // `StagedTransport` can both reach every slab, so a driver that read a
    // neighbour's memory directly would still pass. The tests below run rank 0
    // here and rank 1 in a second process — this binary, re-executed — with a
    // loopback stream as the only thing between them, and compare the result
    // with the single-process solve.
    //
    // Loopback TCP rather than `UnixStream`: the CI matrix includes
    // `windows-latest`, and `std::os::unix` would compile out there, leaving the
    // oracle silently absent on one of the five platforms it most needs to hold
    // on. `std::net` is the same on all of them and adds no dependency.

    /// Environment variables the parent sets when it re-executes this binary as
    /// rank 1; the presence of `XPROC_ROLE` is what tells
    /// `cross_process_rank1_worker` that it is the child.
    #[cfg(feature = "std")]
    const XPROC_ROLE: &str = "ALICE_PHYSICS_XPROC_ROLE";
    /// Loopback port rank 0 is listening on.
    #[cfg(feature = "std")]
    const XPROC_PORT: &str = "ALICE_PHYSICS_XPROC_PORT";
    /// Edge length of the cubic grid both ranks seed.
    #[cfg(feature = "std")]
    const XPROC_SIZE: &str = "ALICE_PHYSICS_XPROC_SIZE";
    /// Which fault, if any, to apply to the crossing on *both* ranks.
    #[cfg(feature = "std")]
    const XPROC_FAULT: &str = "ALICE_PHYSICS_XPROC_FAULT";

    /// libtest name of the child entry point, passed to the re-executed binary
    /// as `--exact`.
    ///
    /// Renaming `cross_process_rank1_worker` without updating this would make
    /// libtest match nothing and exit 0, so the parent checks for a child that
    /// exits before connecting and says so rather than waiting out its deadline.
    #[cfg(feature = "std")]
    const XPROC_WORKER: &str = "eulerian_grid::tests::cross_process_rank1_worker";

    /// Iterations both ranks run; the same count the in-process oracles use.
    #[cfg(feature = "std")]
    const XPROC_ITERATIONS: u32 = 6;

    /// Step both ranks use. Must match on the two ranks: it is part of the
    /// problem, not of the decomposition.
    #[cfg(feature = "std")]
    fn xproc_dt() -> Fix128 {
        Fix128::from_ratio(1, 100)
    }

    /// Density both ranks use.
    #[cfg(feature = "std")]
    fn xproc_rho() -> Fix128 {
        Fix128::from_int(1000)
    }

    /// A fault applied to the crossing itself, identically on both ranks.
    #[cfg(feature = "std")]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CrossFault {
        /// Deliver what the schedule asked for.
        None,
        /// Deliver the layer below the requested one.
        ShiftOneLayer,
        /// Skip the first delivery of the run.
        DropFirstDelivery,
    }

    #[cfg(feature = "std")]
    impl CrossFault {
        /// Name passed to the child through the environment.
        fn as_str(self) -> &'static str {
            match self {
                Self::None => "none",
                Self::ShiftOneLayer => "shift",
                Self::DropFirstDelivery => "drop",
            }
        }

        /// Inverse of [`Self::as_str`]; an unknown name is a harness bug.
        fn parse(name: &str) -> Self {
            match name {
                "none" => Self::None,
                "shift" => Self::ShiftOneLayer,
                "drop" => Self::DropFirstDelivery,
                other => panic!("unknown cross-process fault `{other}`"),
            }
        }
    }

    /// Wraps [`SocketTransport`] to damage the crossing, and nothing else.
    ///
    /// Both ranks apply the same fault to the same delivery, because both count
    /// deliveries over the one global schedule. That is deliberate: a fault on
    /// one side only would desynchronise the stream, and the run would hang
    /// instead of producing a wrong answer — and a hang says nothing about
    /// whether the bit-equality test has teeth.
    ///
    /// Not a stub standing in for unwritten code: the damage *is* the
    /// measurement, and the tests that use it assert the two-process solve comes
    /// out different from the single-process one.
    #[cfg(feature = "std")]
    struct FaultyCrossing<S> {
        inner: SocketTransport<S>,
        fault: CrossFault,
        seen: usize,
    }

    #[cfg(feature = "std")]
    impl<S: std::io::Read + std::io::Write> RankTransport for FaultyCrossing<S> {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
            self.inner.slab_mut(rank)
        }

        fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
            let index = self.seen;
            self.seen += 1;
            match self.fault {
                CrossFault::None => self.inner.deliver_layer(src, dst, layer),
                CrossFault::ShiftOneLayer => {
                    self.inner.deliver_layer(src, dst, layer.saturating_sub(1));
                }
                CrossFault::DropFirstDelivery => {
                    if index > 0 {
                        self.inner.deliver_layer(src, dst, layer);
                    }
                }
            }
        }
    }

    /// Run `my_rank`'s half of the two-rank solve over `link`, the stream to the
    /// other rank. Shared by both processes so neither can drift from the other
    /// in step count, iteration count or fluid parameters.
    #[cfg(feature = "std")]
    fn solve_as_rank(
        grid: &mut MacGrid,
        n: usize,
        my_rank: usize,
        fault: CrossFault,
        link: std::net::TcpStream,
    ) {
        use std::time::Duration;

        // Without these, a desynchronised stream hangs the suite instead of
        // failing it.
        link.set_read_timeout(Some(Duration::from_secs(30)))
            .expect("set a read timeout on the peer link");
        link.set_write_timeout(Some(Duration::from_secs(30)))
            .expect("set a write timeout on the peer link");

        let mut links: Vec<Option<std::net::TcpStream>> = vec![None, None];
        links[1 - my_rank] = Some(link);
        let socket = SocketTransport::new(my_rank, n * n * n, n * n, links);

        if fault == CrossFault::None {
            let mut transport = socket;
            project_pressure_decomposed_on_rank(
                grid,
                xproc_dt(),
                xproc_rho(),
                XPROC_ITERATIONS,
                2,
                HaloSchedule::EverySweep,
                my_rank,
                &mut transport,
            );
        } else {
            let mut transport = FaultyCrossing {
                inner: socket,
                fault,
                seen: 0,
            };
            project_pressure_decomposed_on_rank(
                grid,
                xproc_dt(),
                xproc_rho(),
                XPROC_ITERATIONS,
                2,
                HaloSchedule::EverySweep,
                my_rank,
                &mut transport,
            );
        }
    }

    /// Rank 1's entry point, reached only when this binary has been re-executed
    /// with [`XPROC_ROLE`] set.
    ///
    /// On an ordinary `cargo test` run the variable is absent and this returns
    /// at once: nobody has asked for rank 1, and the tests below are what ask.
    /// It is a `#[test]` rather than a `main` because the crate publishes no
    /// binary a test could spawn, and re-executing the test binary needs a name
    /// libtest will dispatch to.
    #[cfg(feature = "std")]
    #[test]
    fn cross_process_rank1_worker() {
        let Ok(role) = std::env::var(XPROC_ROLE) else {
            return;
        };
        assert_eq!(role, "rank1", "unknown cross-process role `{role}`");

        let n: usize = std::env::var(XPROC_SIZE)
            .expect("grid size from the parent")
            .parse()
            .expect("grid size is a number");
        let port: u16 = std::env::var(XPROC_PORT)
            .expect("loopback port from the parent")
            .parse()
            .expect("loopback port is a number");
        let fault =
            CrossFault::parse(&std::env::var(XPROC_FAULT).expect("fault mode from the parent"));

        let link = std::net::TcpStream::connect(("127.0.0.1", port))
            .expect("connect to rank 0 on the loopback port it is listening on");

        let mut grid = seed_divergent_flow(n);
        solve_as_rank(&mut grid, n, 1, fault, link);

        // `a_rank_that_is_not_the_root_assembles_nothing`: the gather lands on
        // rank 0, so a non-root rank must not have written a field of its own.
        // Checked here because this is the only process that is rank 1.
        assert!(
            grid.pressure.iter().all(|p| *p == Fix128::ZERO),
            "rank 1 assembled a pressure field into its own grid: the gather is \
             supposed to land on rank 0 alone",
        );
    }

    /// Solve `n³` across two processes — rank 0 here, rank 1 re-executed — and
    /// return rank 0's grid.
    #[cfg(feature = "std")]
    fn pressure_field_from_two_processes(n: usize, fault: CrossFault) -> MacGrid {
        use std::io::ErrorKind;
        use std::net::TcpListener;
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        let listener =
            TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port for rank 1");
        let port = listener
            .local_addr()
            .expect("the bound loopback address")
            .port();
        let exe = std::env::current_exe().expect("path of this test binary");

        let mut child = Command::new(exe)
            .args(["--exact", XPROC_WORKER, "--test-threads=1"])
            .env(XPROC_ROLE, "rank1")
            .env(XPROC_PORT, port.to_string())
            .env(XPROC_SIZE, n.to_string())
            .env(XPROC_FAULT, fault.as_str())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("re-execute this test binary as rank 1");

        listener
            .set_nonblocking(true)
            .expect("poll for rank 1 instead of blocking on it");
        let deadline = Instant::now() + Duration::from_secs(60);
        let link = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => panic!("accepting rank 1 failed: {e}"),
            }
            // A child that exits without connecting is not a slow child. The
            // likeliest cause is that `XPROC_WORKER` no longer names a test, in
            // which case libtest ran nothing and exited 0 — say so rather than
            // sitting out the deadline and reporting a timeout.
            if let Some(status) = child.try_wait().expect("poll rank 1") {
                panic!(
                    "rank 1 exited ({status}) without connecting: does \
                     `{XPROC_WORKER}` still name a test?"
                );
            }
            assert!(
                Instant::now() < deadline,
                "rank 1 did not connect within 60 s",
            );
            std::thread::sleep(Duration::from_millis(2));
        };
        link.set_nonblocking(false)
            .expect("switch the accepted stream back to blocking");

        let mut grid = seed_divergent_flow(n);
        solve_as_rank(&mut grid, n, 0, fault, link);

        let status = child.wait().expect("wait for rank 1 to exit");
        assert!(status.success(), "rank 1 exited with {status}");
        grid
    }

    /// Two processes, one loopback stream between them, and the result is the
    /// single-process solve bit for bit — for a rank count that divides the
    /// depth and for one that does not.
    ///
    /// This is what the in-process transport tests could not reach. There,
    /// agreement between two `RankTransport` implementations is compatible with
    /// a decomposition that only works because every slab happens to be
    /// addressable; here rank 1's slab is in another process, every halo layer
    /// is 16 bytes per cell on a wire, and `slab_mut` on either side refuses to
    /// hand out the other rank's buffer.
    ///
    /// Exactness, not a tolerance: `Fix128` addition is a group operation mod
    /// 2¹²⁸, so a distribution that is right at all is right to the bit.
    #[cfg(feature = "std")]
    #[test]
    fn two_processes_reproduce_the_single_process_pressure_solve() {
        for &n in &[8usize, 7] {
            let mut monolithic = seed_divergent_flow(n);
            project_pressure_red_black_gs(
                &mut monolithic,
                xproc_dt(),
                xproc_rho(),
                XPROC_ITERATIONS,
            );

            let distributed = pressure_field_from_two_processes(n, CrossFault::None);

            assert_eq!(
                monolithic.pressure, distributed.pressure,
                "{n}³ over two processes produced a different pressure field \
                 than the single-process solve",
            );
            assert!(
                grids_are_bit_equal(&monolithic, &distributed),
                "{n}³ over two processes matched on pressure but not on the \
                 projected velocities",
            );
        }
    }

    /// Teeth for the test above, on the crossing itself: ship the layer below
    /// the one the schedule asked for — on both ranks, so the stream stays in
    /// step — and the answer must move.
    ///
    /// Without this, `two_processes_reproduce_the_single_process_pressure_solve`
    /// could be passing because both processes seed the same field and the
    /// stream contributes nothing.
    #[cfg(feature = "std")]
    #[test]
    fn a_cross_process_halo_shifted_by_one_layer_does_not_reproduce_the_solve() {
        let n = 8usize;
        let mut monolithic = seed_divergent_flow(n);
        project_pressure_red_black_gs(&mut monolithic, xproc_dt(), xproc_rho(), XPROC_ITERATIONS);

        let shifted = pressure_field_from_two_processes(n, CrossFault::ShiftOneLayer);

        assert_ne!(
            monolithic.pressure, shifted.pressure,
            "halos delivered one layer off still reproduced the single-process \
             solve, so the two-process test is not measuring what crosses the \
             boundary",
        );
    }

    /// Teeth of the other kind: drop one delivery — the first, on both ranks —
    /// and the answer must move. The dropped layer is a halo of rank 0, so its
    /// sentinel survives into the next sweep.
    ///
    /// Dropped symmetrically on purpose: dropping only the send would leave the
    /// receiver waiting, and a timeout would not distinguish a transport that
    /// carries the wrong thing from one that carries nothing.
    #[cfg(feature = "std")]
    #[test]
    fn a_cross_process_delivery_dropped_once_does_not_reproduce_the_solve() {
        let n = 8usize;
        let mut monolithic = seed_divergent_flow(n);
        project_pressure_red_black_gs(&mut monolithic, xproc_dt(), xproc_rho(), XPROC_ITERATIONS);

        let dropped = pressure_field_from_two_processes(n, CrossFault::DropFirstDelivery);

        assert_ne!(
            monolithic.pressure, dropped.pressure,
            "a run missing one halo delivery still reproduced the single-process \
             solve, so the two-process test is not measuring the delivery",
        );
    }

    /// The rank-local driver on one rank is the monolithic solve: with `ranks =
    /// 1` there is nothing to exchange and nothing to gather, so the only thing
    /// left is the sweep.
    ///
    /// Pins the driver itself, separately from the stream: if this and the
    /// two-process test fail together the sweep is at fault, and if only the
    /// two-process one fails the crossing is.
    #[test]
    fn the_rank_local_driver_on_a_single_rank_matches_the_monolithic_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let n = 8usize;

        let mut monolithic = seed_divergent_flow(n);
        project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

        let mut alone = seed_divergent_flow(n);
        let mut transport = LocalTransport::new(1, n, n * n);
        project_pressure_decomposed_on_rank(
            &mut alone,
            dt,
            rho,
            6,
            1,
            HaloSchedule::EverySweep,
            0,
            &mut transport,
        );

        assert!(
            grids_are_bit_equal(&monolithic, &alone),
            "the rank-local driver on a single rank did not reproduce the \
             monolithic solve",
        );
    }

    // ---- A-3.2c: three ranks, where a delivery can belong to neither end ----
    //
    // Two ranks cannot separate "every rank walks the whole schedule" from
    // "each rank walks only its own deliveries": with two ranks every delivery
    // names both of them, so `SocketTransport`'s third branch — perform
    // nothing, because this rank is neither sender nor receiver — is never
    // reached, and a transport that mishandled another pair's delivery would
    // have gone unnoticed. Three ranks are the fewest that put a rank outside a
    // delivery: rank 0 is a bystander to the halo rank 2 trades with rank 1.
    //
    // The topology below is read off `exchange_slab_halos` rather than assumed.
    // A rank asks only for the layer under its first and the layer over its
    // last, so halos couple neighbouring slabs and ranks 0 and 2 never trade
    // one; `gather_slabs_to_root` has the opposite shape, every rank shipping
    // its own layers to rank 0, so the 0 <-> 2 link is needed for the gather
    // even though no halo crosses it. Both are pinned by
    // `the_halo_exchange_couples_only_neighbouring_slabs`.
    //
    // Deadlock, which would make a failure say nothing: each delivery names one
    // sender and one receiver, and all ranks walk the same sequence, so for any
    // position exactly one end writes and one reads. A rank can only run ahead
    // of its peers across deliveries it is not party to, and a blocked write is
    // released by the receiver reaching that same position, so blocking only
    // ever propagates to earlier positions — of which there are finitely many.
    // Read and write timeouts stay in place as the backstop.

    /// Records the `(src, dst, layer)` sequence a schedule walks, so the shape
    /// of an exchange can be asserted without running a solve.
    ///
    /// Not a stand-in for a transport: the walkers never ask it for a slab, and
    /// `slab_mut` says so rather than handing back a buffer that would make a
    /// mistaken call look successful.
    struct DeliveryRecorder {
        calls: Vec<(usize, usize, usize)>,
    }

    impl RankTransport for DeliveryRecorder {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
            unreachable!("a schedule walk asked the recorder for rank {rank}'s slab")
        }

        fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
            self.calls.push((src, dst, layer));
        }
    }

    /// The halo sequence and then the gather sequence for `nz` layers split
    /// across `ranks` slabs.
    #[allow(clippy::type_complexity)]
    fn recorded_schedule(
        nz: usize,
        ranks: usize,
    ) -> (Vec<(usize, usize, usize)>, Vec<(usize, usize, usize)>) {
        let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
        let mut halo = DeliveryRecorder { calls: Vec::new() };
        exchange_slab_halos(&mut halo, &bounds, nz);
        let mut gather = DeliveryRecorder { calls: Vec::new() };
        gather_slabs_to_root(&mut gather, &bounds);
        (halo.calls, gather.calls)
    }

    /// A rank is a bystander to some delivery only once there are more than two
    /// ranks.
    ///
    /// This is why the two-process harness above leaves `SocketTransport`'s
    /// no-op branch unvisited, and why the three-process one below reaches it.
    /// Stated over the schedule itself so it holds for every decomposition, not
    /// only the sizes the process tests can afford to run.
    #[test]
    fn only_more_than_two_ranks_give_a_rank_deliveries_it_is_not_party_to() {
        for ranks in 1usize..=6 {
            for nz in [ranks, 2 * ranks + 1, 3 * ranks] {
                let (halo, gather) = recorded_schedule(nz, ranks);
                let bystanding = |r: usize| {
                    halo.iter()
                        .chain(gather.iter())
                        .filter(|&&(src, dst, _)| src != r && dst != r)
                        .count()
                };
                if ranks <= 2 {
                    for r in 0..ranks {
                        assert_eq!(
                            bystanding(r),
                            0,
                            "{nz} layers over {ranks} ranks gave rank {r} a delivery it \
                             is not party to, so the two-rank harness does exercise the \
                             no-op branch after all",
                        );
                    }
                } else {
                    assert!(
                        bystanding(0) > 0,
                        "{nz} layers over {ranks} ranks left rank 0 party to every \
                         delivery, so running it as rank 0 of a multi-process solve \
                         would not reach the no-op branch",
                    );
                }
            }
        }
    }

    /// Halos couple neighbouring slabs only — a delivery's two ranks have no
    /// rank owning layers between them — while every gather delivery lands on
    /// rank 0.
    ///
    /// The two shapes differ, which is what decides the links a run needs: with
    /// three ranks no halo crosses between ranks 0 and 2, yet the gather does,
    /// so that link has to exist anyway.
    #[test]
    fn the_halo_exchange_couples_only_neighbouring_slabs() {
        for ranks in 1usize..=6 {
            for nz in [1, ranks, 2 * ranks + 1, 3 * ranks] {
                let bounds: Vec<(usize, usize)> =
                    (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
                let (halo, gather) = recorded_schedule(nz, ranks);

                for &(src, dst, _) in &halo {
                    assert_ne!(src, dst, "a halo delivery had one rank at both ends");
                    let (lo, hi) = (src.min(dst), src.max(dst));
                    for (r, &(k0, k1)) in bounds.iter().enumerate().take(hi).skip(lo + 1) {
                        assert_eq!(
                            k0, k1,
                            "the halo delivery {src} -> {dst} reached over rank {r}, \
                             which owns layers {k0}..{k1}",
                        );
                    }
                }

                for &(src, dst, layer) in &gather {
                    assert_eq!(dst, 0, "a gather delivery landed on rank {dst}, not root");
                    assert_ne!(src, 0, "the root was asked to gather from itself");
                    assert_eq!(
                        slab_owner(&bounds, layer),
                        Some(src),
                        "rank {src} gathered layer {layer}, which it does not own",
                    );
                }
            }
        }

        // Three slabs: the two ends never trade a halo, but the gather crosses
        // them, so a three-rank run still needs that link.
        let (halo, gather) = recorded_schedule(9, 3);
        assert!(
            halo.iter()
                .all(|&(src, dst, _)| (src, dst) != (0, 2) && (src, dst) != (2, 0)),
            "a halo crossed directly between the first and last of three slabs",
        );
        assert!(
            gather.iter().any(|&(src, dst, _)| (src, dst) == (2, 0)),
            "the gather never crossed from the last of three slabs to the root",
        );
    }

    /// Environment rank 0 sets when it re-executes this binary as rank 1 or
    /// rank 2 of a three-rank solve; the presence of `XPROC3_RANK` is what tells
    /// `cross_process_three_rank_worker` that it is a child.
    #[cfg(feature = "std")]
    const XPROC3_RANK: &str = "ALICE_PHYSICS_XPROC3_RANK";
    /// Loopback port on which rank 0 is waiting for this child.
    #[cfg(feature = "std")]
    const XPROC3_PORT: &str = "ALICE_PHYSICS_XPROC3_PORT";
    /// Edge length of the cubic grid all three ranks seed.
    #[cfg(feature = "std")]
    const XPROC3_SIZE: &str = "ALICE_PHYSICS_XPROC3_SIZE";
    /// How the ranks treat the deliveries they are not party to.
    #[cfg(feature = "std")]
    const XPROC3_FAULT: &str = "ALICE_PHYSICS_XPROC3_FAULT";

    /// libtest name of the three-rank child entry point, passed to the
    /// re-executed binary as `--exact`.
    ///
    /// Same hazard as `XPROC_WORKER`, and handled the same way: a filter that
    /// matches nothing makes libtest run no test and exit 0, so a child that
    /// exits before connecting is reported against this constant by name
    /// instead of being waited out as a timeout.
    #[cfg(feature = "std")]
    const XPROC3_WORKER: &str = "eulerian_grid::tests::cross_process_three_rank_worker";

    /// Iterations all three ranks run.
    #[cfg(feature = "std")]
    const XPROC3_ITERATIONS: u32 = 6;

    /// What a rank does with a delivery it is neither the sender nor the
    /// receiver of.
    #[cfg(feature = "std")]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum BystanderMode {
        /// Perform nothing and move on, which is what keeps the ranks'
        /// positions in the schedule the same.
        Skip,
        /// Perform it anyway, as though this rank were the sender.
        Deliver,
    }

    #[cfg(feature = "std")]
    impl BystanderMode {
        /// Name passed to the children through the environment.
        fn as_str(self) -> &'static str {
            match self {
                Self::Skip => "skip",
                Self::Deliver => "deliver",
            }
        }

        /// Inverse of [`Self::as_str`]; an unknown name is a harness bug.
        fn parse(name: &str) -> Self {
            match name {
                "skip" => Self::Skip,
                "deliver" => Self::Deliver,
                other => panic!("unknown bystander mode `{other}`"),
            }
        }
    }

    /// Wraps [`SocketTransport`] so that a rank sends on the deliveries it is
    /// not party to, and changes nothing else.
    ///
    /// The control for the three-process oracle. Deliveries a rank is party to
    /// pass through untouched, so whatever the run does differently comes from
    /// the third-party ones alone.
    ///
    /// # Why the damage cannot be a quiet wrong answer
    ///
    /// A purely local misstep on a third-party delivery would not show up at
    /// all: such a delivery always names a layer outside this rank's halo, and
    /// everything out there is already the sentinel, so writing to it changes
    /// nothing a sweep reads. What the no-op branch protects is therefore not
    /// the slab but the *alignment of the streams* — and a bystander that acts
    /// adds an endpoint no peer is matched to, which is exactly a desynchronised
    /// stream. Measured: the peers finish their own walk and close with those
    /// extra layers still unread, the close becomes a reset, and the run ends in
    /// `ConnectionReset` rather than in a wrong field. The control therefore
    /// accepts either outcome; what it does not accept is reproducing the solve.
    /// It cannot hang: every stream carries a 30 s timeout each way.
    ///
    /// Not a stub standing in for unwritten code: the extra send is the
    /// measurement.
    #[cfg(feature = "std")]
    struct BystanderDelivers<S> {
        inner: SocketTransport<S>,
    }

    #[cfg(feature = "std")]
    impl<S: std::io::Read + std::io::Write> RankTransport for BystanderDelivers<S> {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
            self.inner.slab_mut(rank)
        }

        fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
            let mine = self.inner.my_rank;
            if mine == src || mine == dst {
                self.inner.deliver_layer(src, dst, layer);
            } else {
                // Claim it: the inner transport then takes its sending branch
                // and puts a layer on the wire to `dst`, where the real
                // sender's layer now queues behind it.
                self.inner.deliver_layer(mine, dst, layer);
            }
        }
    }

    /// 30 s each way, so a desynchronised stream fails the suite instead of
    /// hanging it.
    #[cfg(feature = "std")]
    fn bound_peer_link(link: &std::net::TcpStream) {
        use std::time::Duration;

        link.set_read_timeout(Some(Duration::from_secs(30)))
            .expect("set a read timeout on a peer link");
        link.set_write_timeout(Some(Duration::from_secs(30)))
            .expect("set a write timeout on a peer link");
    }

    /// Accept one connection, giving up after 60 s rather than hanging when the
    /// peer rank never arrives.
    #[cfg(feature = "std")]
    fn accept_before_deadline(listener: &std::net::TcpListener, who: &str) -> std::net::TcpStream {
        use std::io::ErrorKind;
        use std::time::{Duration, Instant};

        listener
            .set_nonblocking(true)
            .expect("poll for a peer rank instead of blocking on it");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_nonblocking(false)
                        .expect("switch an accepted stream back to blocking");
                    bound_peer_link(&stream);
                    return stream;
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(e) => panic!("accepting {who} failed: {e}"),
            }
            assert!(
                Instant::now() < deadline,
                "{who} did not connect within 60 s"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// The child ranks rank 0 spawned, reaped on the way out even when an
    /// assertion unwinds, so a failing run leaves no process holding a loopback
    /// port.
    #[cfg(feature = "std")]
    struct SpawnedRanks {
        kids: Vec<std::process::Child>,
    }

    #[cfg(feature = "std")]
    impl Drop for SpawnedRanks {
        fn drop(&mut self) {
            for kid in &mut self.kids {
                let _ = kid.kill();
                let _ = kid.wait();
            }
        }
    }

    #[cfg(feature = "std")]
    impl SpawnedRanks {
        /// Accept one rank's connection while watching *every* child.
        ///
        /// All children are polled on each turn, not just the one this listener
        /// belongs to: a second child that exited early would otherwise leave
        /// this loop running to its deadline, and "timed out" is all the run
        /// would be able to say about it.
        fn accept(&mut self, listener: &std::net::TcpListener, rank: usize) -> std::net::TcpStream {
            use std::io::ErrorKind;
            use std::time::{Duration, Instant};

            listener
                .set_nonblocking(true)
                .expect("poll for a child rank instead of blocking on it");
            let deadline = Instant::now() + Duration::from_secs(60);
            let link = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                    Err(e) => panic!("accepting rank {rank} failed: {e}"),
                }
                for (slot, kid) in self.kids.iter_mut().enumerate() {
                    if let Some(status) = kid.try_wait().expect("poll a child rank") {
                        panic!(
                            "rank {} exited ({status}) before rank {rank} connected: does \
                             `{XPROC3_WORKER}` still name a test?",
                            slot + 1,
                        );
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "rank {rank} did not connect within 60 s",
                );
                std::thread::sleep(Duration::from_millis(2));
            };
            link.set_nonblocking(false)
                .expect("switch the accepted stream back to blocking");
            bound_peer_link(&link);
            link
        }

        /// Wait for every rank and require a clean exit.
        fn join(&mut self) {
            for (slot, kid) in self.kids.iter_mut().enumerate() {
                let status = kid.wait().expect("wait for a child rank to exit");
                assert!(status.success(), "rank {} exited with {status}", slot + 1);
            }
        }
    }

    /// Run `my_rank`'s half of the three-rank solve over `links`, where
    /// `links[r]` is the stream to rank `r` and this rank's own entry is `None`.
    ///
    /// Shared by all three processes so none can drift from the others in
    /// iteration count, schedule or fluid parameters.
    #[cfg(feature = "std")]
    fn solve_as_rank_of_three(
        grid: &mut MacGrid,
        n: usize,
        my_rank: usize,
        mode: BystanderMode,
        links: Vec<Option<std::net::TcpStream>>,
    ) {
        let socket = SocketTransport::new(my_rank, n * n * n, n * n, links);
        match mode {
            BystanderMode::Skip => {
                let mut transport = socket;
                project_pressure_decomposed_on_rank(
                    grid,
                    xproc_dt(),
                    xproc_rho(),
                    XPROC3_ITERATIONS,
                    3,
                    HaloSchedule::EverySweep,
                    my_rank,
                    &mut transport,
                );
            }
            BystanderMode::Deliver => {
                let mut transport = BystanderDelivers { inner: socket };
                project_pressure_decomposed_on_rank(
                    grid,
                    xproc_dt(),
                    xproc_rho(),
                    XPROC3_ITERATIONS,
                    3,
                    HaloSchedule::EverySweep,
                    my_rank,
                    &mut transport,
                );
            }
        }
    }

    /// Rank 1 and rank 2 of the three-rank solve, reached only when this binary
    /// has been re-executed with [`XPROC3_RANK`] set.
    ///
    /// One entry point for both: they run the same code and differ only in the
    /// rank they are told they are, and in which of them hosts the link between
    /// them. On an ordinary `cargo test` run the variable is absent and this
    /// returns at once — the tests below are what ask for a child rank.
    #[cfg(feature = "std")]
    #[test]
    fn cross_process_three_rank_worker() {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        let Ok(rank) = std::env::var(XPROC3_RANK) else {
            return;
        };
        let my_rank: usize = rank.parse().expect("the rank is a number");
        assert!(
            my_rank == 1 || my_rank == 2,
            "rank {my_rank} is not a child rank of the three-rank harness",
        );

        let n: usize = std::env::var(XPROC3_SIZE)
            .expect("grid size from rank 0")
            .parse()
            .expect("grid size is a number");
        let port: u16 = std::env::var(XPROC3_PORT)
            .expect("loopback port from rank 0")
            .parse()
            .expect("loopback port is a number");
        let mode =
            BystanderMode::parse(&std::env::var(XPROC3_FAULT).expect("bystander mode from rank 0"));

        let mut root = TcpStream::connect(("127.0.0.1", port))
            .expect("connect to rank 0 on the loopback port it is listening on");
        bound_peer_link(&root);

        // The link between the two children cannot be handed down by rank 0,
        // which is not one of its ends. Rank 1 binds it and sends the port up,
        // rank 0 passes the number on to rank 2, and rank 2 dials it: no fixed
        // port and nothing racing to rebind one.
        let mut links: Vec<Option<TcpStream>> = vec![None, None, None];
        if my_rank == 1 {
            let peer_listener =
                TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port for rank 2");
            let peer_port = peer_listener
                .local_addr()
                .expect("the bound loopback address")
                .port();
            root.write_all(&peer_port.to_le_bytes())
                .expect("tell rank 0 where rank 2 should dial");
            root.flush().expect("flush the peer port to rank 0");
            links[2] = Some(accept_before_deadline(&peer_listener, "rank 2"));
            links[0] = Some(root);
        } else {
            let mut peer_port = [0u8; 2];
            root.read_exact(&mut peer_port)
                .expect("learn from rank 0 where rank 1 is listening");
            let peer = TcpStream::connect(("127.0.0.1", u16::from_le_bytes(peer_port)))
                .expect("connect to rank 1 on the port it is listening on");
            bound_peer_link(&peer);
            links[1] = Some(peer);
            links[0] = Some(root);
        }

        let mut grid = seed_divergent_flow(n);
        solve_as_rank_of_three(&mut grid, n, my_rank, mode, links);

        // The gather lands on rank 0, so neither child may have assembled a
        // field of its own. Checked here because these are the only processes
        // that are not the root.
        assert!(
            grid.pressure.iter().all(|p| *p == Fix128::ZERO),
            "rank {my_rank} assembled a pressure field into its own grid: the \
             gather is supposed to land on rank 0 alone",
        );
    }

    /// Solve `n³` across three processes — rank 0 here, ranks 1 and 2
    /// re-executed — and return rank 0's grid.
    #[cfg(feature = "std")]
    fn pressure_field_from_three_processes(n: usize, mode: BystanderMode) -> MacGrid {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::process::{Command, Stdio};

        let listeners: Vec<TcpListener> = (1usize..=2)
            .map(|rank| {
                TcpListener::bind(("127.0.0.1", 0))
                    .unwrap_or_else(|e| panic!("bind a loopback port for rank {rank}: {e}"))
            })
            .collect();
        let exe = std::env::current_exe().expect("path of this test binary");

        let mut ranks = SpawnedRanks {
            kids: listeners
                .iter()
                .enumerate()
                .map(|(slot, listener)| {
                    let port = listener
                        .local_addr()
                        .expect("the bound loopback address")
                        .port();
                    Command::new(&exe)
                        .args(["--exact", XPROC3_WORKER, "--test-threads=1"])
                        .env(XPROC3_RANK, (slot + 1).to_string())
                        .env(XPROC3_PORT, port.to_string())
                        .env(XPROC3_SIZE, n.to_string())
                        .env(XPROC3_FAULT, mode.as_str())
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .unwrap_or_else(|e| {
                            panic!("re-execute this test binary as rank {}: {e}", slot + 1)
                        })
                })
                .collect(),
        };

        let mut to_rank1 = ranks.accept(&listeners[0], 1);
        let mut to_rank2 = ranks.accept(&listeners[1], 2);

        // Relay rank 1's listening port to rank 2. Rank 0 is not an end of that
        // link and only carries the number.
        let mut peer_port = [0u8; 2];
        to_rank1
            .read_exact(&mut peer_port)
            .expect("learn from rank 1 where rank 2 should dial");
        to_rank2
            .write_all(&peer_port)
            .expect("tell rank 2 where rank 1 is listening");
        to_rank2.flush().expect("flush the peer port to rank 2");

        let mut grid = seed_divergent_flow(n);
        solve_as_rank_of_three(
            &mut grid,
            n,
            0,
            mode,
            vec![None, Some(to_rank1), Some(to_rank2)],
        );

        ranks.join();
        grid
    }

    /// Three processes — two loopback streams out of rank 0 and one between the
    /// other two — reproduce the single-process solve bit for bit, for a rank
    /// count that divides the depth and for one that does not.
    ///
    /// What three ranks settle that two could not: rank 0 is not party to the
    /// halo rank 2 trades with rank 1, so `SocketTransport` reaches its third
    /// branch — perform nothing, stay in step — for the first time here.
    ///
    /// Exactness, not a tolerance: `Fix128` addition is a group operation mod
    /// 2¹²⁸, so a distribution that is right at all is right to the bit.
    #[cfg(feature = "std")]
    #[test]
    fn three_processes_reproduce_the_single_process_pressure_solve() {
        for &n in &[9usize, 8] {
            let mut monolithic = seed_divergent_flow(n);
            project_pressure_red_black_gs(
                &mut monolithic,
                xproc_dt(),
                xproc_rho(),
                XPROC3_ITERATIONS,
            );

            let distributed = pressure_field_from_three_processes(n, BystanderMode::Skip);

            assert_eq!(
                monolithic.pressure, distributed.pressure,
                "{n}³ over three processes produced a different pressure field \
                 than the single-process solve",
            );
            assert!(
                grids_are_bit_equal(&monolithic, &distributed),
                "{n}³ over three processes matched on pressure but not on the \
                 projected velocities",
            );
        }
    }

    /// Teeth for the test above, aimed at the branch three ranks exist to
    /// reach: at the same size, skipping the deliveries a rank is not party to
    /// reproduces the single-process solve and performing them does not.
    ///
    /// Without this, `three_processes_reproduce_the_single_process_pressure_solve`
    /// would show only that three ranks can agree, not that declining another
    /// pair's delivery is what makes them agree — which is the one property two
    /// ranks could not express at all, since with two ranks every delivery names
    /// both of them (`only_more_than_two_ranks_give_a_rank_deliveries_it_is_not_party_to`).
    ///
    /// Both runs are the same grid, the same iteration count and the same
    /// schedule, so the mode is the only difference between them. "Does not
    /// reproduce" covers a differing field and a failed run alike, for the
    /// reason [`BystanderDelivers`] gives: acting as a bystander desynchronises
    /// the streams instead of merely miscomputing, and which of the two a given
    /// platform reports is not the property being pinned.
    #[cfg(feature = "std")]
    #[test]
    fn a_rank_that_performs_deliveries_it_is_not_party_to_does_not_reproduce_the_solve() {
        let n = 6usize;
        let mut monolithic = seed_divergent_flow(n);
        project_pressure_red_black_gs(&mut monolithic, xproc_dt(), xproc_rho(), XPROC3_ITERATIONS);

        let skipped = pressure_field_from_three_processes(n, BystanderMode::Skip);
        assert_eq!(
            monolithic.pressure, skipped.pressure,
            "{n}³ over three processes did not reproduce the single-process \
             solve, so the comparison below is not isolating the mode",
        );

        // The faulted run is expected to end in a reset rather than a wrong
        // field, so it is run where an unwind can be read as an outcome. The
        // panic message it prints belongs to this test passing.
        let meddled = std::panic::catch_unwind(|| {
            pressure_field_from_three_processes(n, BystanderMode::Deliver)
        });

        match meddled {
            Ok(field) => assert_ne!(
                monolithic.pressure, field.pressure,
                "ranks that performed the deliveries they are not party to still \
                 reproduced the single-process solve, so the three-process test \
                 is not measuring the bystander branch",
            ),
            Err(_) => {
                // The exchange came apart, which is the other way of not
                // reproducing the solve.
            }
        }
    }

    /// The slabs must tile the depth exactly: every layer owned once, none twice.
    #[test]
    fn slab_bounds_partition_every_layer_exactly_once() {
        for nz in 1usize..=16 {
            for ranks in 1usize..=20 {
                let bounds: Vec<(usize, usize)> =
                    (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
                for k in 0..nz {
                    let owners = bounds.iter().filter(|&&(k0, k1)| k0 <= k && k < k1).count();
                    assert_eq!(
                        owners, 1,
                        "layer {k} of {nz} has {owners} owners across {ranks} ranks",
                    );
                }
                let owned: usize = bounds.iter().map(|&(k0, k1)| k1 - k0).sum();
                assert_eq!(owned, nz, "{ranks} ranks over {nz} layers own {owned}");
            }
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

    // ========================================================================
    // Slab-local storage
    // ========================================================================

    /// Rank counts that divide the depth, ones that do not, and ones fine
    /// enough to leave a rank owning nothing — the same list the full-length
    /// decomposition oracle uses, so the two are comparable.
    const SLAB_CASES: [(usize, usize); 9] = [
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
        (7, 2),
        (7, 3),
        (7, 4),
        (5, 4),
        (3, 4),
    ];

    /// A divergent field, optionally boxed in by walls and optionally starting
    /// from a non-zero pressure.
    ///
    /// The walls make the Poisson mask non-trivial (without them every face is
    /// open and every cell has degree six, so a localised mask could be wrong
    /// in ways nothing would notice). The initial pressure makes the *initial*
    /// halo load-bearing: a rank's first sweep reads its neighbour's boundary
    /// layer before any exchange has happened, and a field starting at zero
    /// cannot tell a correctly initialised halo from one left at zero.
    fn seed_slab_case(n: usize, walls: bool, initial_pressure: bool) -> MacGrid {
        let mut grid = seed_divergent_flow(n);
        if walls {
            grid.set_closed_box_walls();
        }
        if initial_pressure {
            for (c, slot) in grid.pressure.iter_mut().enumerate() {
                *slot = Fix128::from_ratio((c % 5) as i64 - 2, 7);
            }
        }
        grid
    }

    /// Far from any pressure or velocity this solve produces, so a cell or face
    /// that no rank wrote shows up as a mismatch of many units rather than as a
    /// plausible number left over from the template.
    const UNWRITTEN: Fix128 = Fix128::from_int(1_000_000);

    /// Put the ranks' owned layers back together into one grid, so the result
    /// can be compared with the monolithic solve.
    ///
    /// Only the tests do this: the point of slab-local storage is that no rank
    /// holds the whole field, and this function is the measurement apparatus,
    /// not part of the decomposition.
    fn assemble_slabs(
        template: &MacGrid,
        faces: &[SlabFaces],
        transport: &LocalSlabTransport,
    ) -> MacGrid {
        let mut out = template.clone();
        out.pressure.fill(UNWRITTEN);
        out.u.fill(UNWRITTEN);
        out.v.fill(UNWRITTEN);
        out.w.fill(UNWRITTEN);
        let (nx, ny) = (out.nx, out.ny);
        let plane = cell_plane(nx, ny);

        for (r, rank_faces) in faces.iter().enumerate() {
            let (k0, k1) = rank_faces.owned();
            for k in k0..k1 {
                let layer = transport
                    .slab(r)
                    .layer(k)
                    .expect("a rank's own layer is resident in its band");
                out.pressure[k * plane..(k + 1) * plane].copy_from_slice(layer);

                let (u, _) = rank_faces.u_layer(k);
                for j in 0..ny {
                    for i in 0..=nx {
                        let ix = out.idx_u(i, j, k);
                        out.u[ix] = u[i + (nx + 1) * j];
                    }
                }
                let (v, _) = rank_faces.v_layer(k);
                for j in 0..=ny {
                    for i in 0..nx {
                        let ix = out.idx_v(i, j, k);
                        out.v[ix] = v[i + nx * j];
                    }
                }
            }
            for k in rank_faces.w_written() {
                let (w, _) = rank_faces.w_layer(k);
                for j in 0..ny {
                    for i in 0..nx {
                        let ix = out.idx_w(i, j, k);
                        out.w[ix] = w[i + nx * j];
                    }
                }
            }
        }
        out
    }

    /// Run the slab-local projection over `ranks` slabs with a halo of `halo`
    /// layers, and reassemble the answer.
    fn solve_slab_local(
        base: &MacGrid,
        dt_s: Fix128,
        density: Fix128,
        iterations: u32,
        ranks: usize,
        halo: usize,
        schedule: HaloSchedule,
    ) -> MacGrid {
        // Imposed once, before the split: the monolithic solve does it inside
        // itself, and an outflow face copying its inward neighbour is not
        // idempotent, so the enforcement must not happen twice.
        let mut enforced = base.clone();
        enforced.enforce_face_boundaries();

        let nz = enforced.nz;
        let plane = cell_plane(enforced.nx, enforced.ny);
        let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
        let mut faces: Vec<SlabFaces> = bounds
            .iter()
            .map(|&b| SlabFaces::from_grid(&enforced, b))
            .collect();
        let mut transport =
            LocalSlabTransport::from_field(&bounds, nz, plane, halo, &enforced.pressure);

        project_pressure_slab_local_over(
            &mut faces,
            dt_s,
            density,
            iterations,
            schedule,
            &mut transport,
        );
        assemble_slabs(base, &faces, &transport)
    }

    /// Slab-local storage — every rank holding only its own layers plus one
    /// halo layer, with no full-length array anywhere — reproduces the
    /// monolithic solve bit for bit.
    ///
    /// Exactness, not a tolerance, for the reason the full-length oracle gives:
    /// `Fix128` addition is a group operation mod 2¹²⁸, so a decomposition that
    /// is correct at all is correct to the bit. The variants with walls and
    /// with a non-zero initial pressure are there because a localised mask and
    /// a localised initial halo are two more things that can be wrong without
    /// the plainest case noticing (see `seed_slab_case`).
    #[test]
    fn slab_local_storage_reproduces_the_monolithic_pressure_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);

        for &(n, ranks) in &SLAB_CASES {
            for &(walls, initial_pressure) in
                &[(false, false), (true, false), (true, true), (false, true)]
            {
                let base = seed_slab_case(n, walls, initial_pressure);

                let mut monolithic = base.clone();
                project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

                let assembled =
                    solve_slab_local(&base, dt, rho, 6, ranks, 1, HaloSchedule::EverySweep);

                assert!(
                    grids_are_bit_equal(&monolithic, &assembled),
                    "{n}³ over {ranks} slab-local ranks (walls {walls}, initial pressure \
                     {initial_pressure}) did not reproduce the monolithic solve: either a \
                     localised array is indexed wrongly, a face is written by the wrong rank, \
                     or the halo arrives with the wrong contents",
                );
            }
        }
    }

    /// Teeth for the test above, in the same shape the full-length path uses:
    /// delay the exchange by one sweep and the slabs stop agreeing.
    ///
    /// Without this, the bit-equality above could be passing on a decomposition
    /// that never depended on the halo arriving at all.
    #[test]
    fn a_stale_slab_halo_does_not_reproduce_the_monolithic_solve() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_slab_case(8, true, true);

        let mut monolithic = base.clone();
        project_pressure_red_black_gs(&mut monolithic, dt, rho, 6);

        let stale = solve_slab_local(&base, dt, rho, 6, 4, 1, HaloSchedule::EveryIteration);

        assert!(
            !grids_are_bit_equal(&monolithic, &stale),
            "a halo one sweep out of date still reproduced the monolithic solve, so the \
             bit-equality test above is not actually testing the exchange",
        );
    }

    /// The point of localising the storage: a halo narrower than the stencil is
    /// reported at the read, by a layer that is not there, rather than by a
    /// number that comes out wrong later.
    ///
    /// This is what the full-length path cannot do. There, every rank holds the
    /// whole field, so a stencil reaching past its halo finds *something* — a
    /// sentinel if the buffer was poisoned, and a perfectly good value if it was
    /// not — and the mistake only surfaces when the final field is compared.
    /// Here the layer does not exist, and [`SlabStorage`] is the one place a
    /// global layer becomes an offset, so the read cannot land on a different
    /// cell by accident.
    #[test]
    #[should_panic(expected = "slab halo too narrow")]
    fn a_halo_narrower_than_the_stencil_aborts_instead_of_returning_a_number() {
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let base = seed_slab_case(8, false, false);
        // Halo 0: rank 1 owns layers 4..8 and its cells at layer 4 have an open
        // `-z` face onto layer 3, which no longer exists in its band.
        let _ = solve_slab_local(&base, dt, rho, 6, 2, 0, HaloSchedule::EverySweep);
    }

    /// The exchange is guarded at the other end too: a delivery into a rank
    /// that has no room for the layer aborts rather than being dropped.
    ///
    /// The sweep reaches its own guard first in a full solve, so this exercises
    /// the delivery path directly. Both guards matter: a transport that
    /// silently discarded an undeliverable layer would turn a configuration
    /// error into a wrong answer.
    #[test]
    #[should_panic(expected = "no room for the layer it was asked to receive")]
    fn a_slab_delivery_into_a_rank_without_room_aborts() {
        let nz = 8usize;
        let plane = 4usize;
        let bounds: Vec<(usize, usize)> = (0..2).map(|r| slab_bounds(nz, 2, r)).collect();
        let field = vec![Fix128::ZERO; nz * plane];
        let mut transport = LocalSlabTransport::from_field(&bounds, nz, plane, 0, &field);
        exchange_slab_halos_local(&mut transport, &bounds, nz);
    }

    /// Records the `(src, dst, layer)` sequence a slab-local schedule walks.
    ///
    /// Not a stand-in for a transport: the walk never asks it for a slab, and
    /// `slab_mut` says so rather than handing back storage that would make a
    /// mistaken call look successful.
    struct SlabDeliveryRecorder {
        calls: Vec<(usize, usize, usize)>,
    }

    impl SlabTransport for SlabDeliveryRecorder {
        fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage {
            unreachable!("a schedule walk asked the recorder for rank {rank}'s slab")
        }

        fn deliver_layer(&mut self, src: usize, dst: usize, layer: usize) {
            self.calls.push((src, dst, layer));
        }
    }

    /// The two exchange walks — the full-length one and the slab-local one —
    /// deliver the same layers between the same ranks in the same order.
    ///
    /// They are two copies of one loop, which is the price of not moving
    /// `deliver_layer` to a supertrait of [`RankTransport`] and editing every
    /// existing implementation. This is what keeps the copies honest: if either
    /// walk is changed alone, the recorded sequences stop matching.
    #[test]
    fn both_exchange_walks_deliver_the_same_layers_in_the_same_order() {
        for ranks in 1usize..=6 {
            for nz in [ranks, 2 * ranks + 1, 3 * ranks, 1] {
                let bounds: Vec<(usize, usize)> =
                    (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
                let (full_length, _) = recorded_schedule(nz, ranks);
                let mut local = SlabDeliveryRecorder { calls: Vec::new() };
                exchange_slab_halos_local(&mut local, &bounds, nz);
                assert_eq!(
                    local.calls, full_length,
                    "{nz} layers over {ranks} ranks: the slab-local exchange walks a \
                     different sequence from the full-length one",
                );
            }
        }
    }

    /// Every cell layer is swept by exactly one rank and every Z-face layer is
    /// updated by exactly one rank, for every depth and rank count — including
    /// the splits that leave a rank owning nothing.
    ///
    /// The Z-faces are the ones that can go wrong: a face sits *between* two
    /// cell layers, so each rank holds one more Z-face layer than it owns and
    /// has to leave the top one to its neighbour, except at the top of the
    /// domain where there is no neighbour. Written over the partition rather
    /// than inferred from a solve, so it holds for decompositions no test
    /// could afford to run.
    #[test]
    fn every_face_layer_is_updated_by_exactly_one_rank() {
        for nz in 1usize..=16 {
            for ranks in 1usize..=20 {
                let mut cells = vec![0usize; nz];
                let mut z_faces = vec![0usize; nz + 1];
                for r in 0..ranks {
                    let b = slab_bounds(nz, ranks, r);
                    let faces = SlabFaces::new(2, 2, nz, Fix128::ONE, b);
                    for owned in &mut cells[b.0..b.1] {
                        *owned += 1;
                    }
                    for k in faces.w_written() {
                        z_faces[k] += 1;
                    }
                }
                assert!(
                    cells.iter().all(|&c| c == 1),
                    "{nz} layers over {ranks} ranks: cell layers swept {cells:?} times",
                );
                assert!(
                    z_faces.iter().all(|&c| c == 1),
                    "{nz} layers over {ranks} ranks: Z-face layers updated {z_faces:?} times",
                );
            }
        }
    }

    /// The byte accounting reports the containers that exist, so it cannot
    /// drift from them the way a formula can.
    ///
    /// Checked against the layer counts the decomposition implies, and checked
    /// to be non-zero: an instrument that measured nothing would otherwise
    /// satisfy every comparison made with it.
    #[test]
    fn the_measured_working_set_is_the_one_the_layer_counts_imply() {
        let n = 6usize;
        let ranks = 3usize;
        let grid = seed_slab_case(n, true, false);
        let plane = cell_plane(n, n);
        let scale = Fix128::ONE;

        for r in 0..ranks {
            let b = slab_bounds(n, ranks, r);
            let faces = SlabFaces::from_grid(&grid, b);
            let storage = SlabStorage::for_slab(plane, n, b, 1);
            let stencil = SlabStencil::build(&faces, scale);
            let bytes = storage.bytes().add(faces.bytes()).add(stencil.bytes());

            let (lo, hi) = storage.resident();
            let owned = b.1 - b.0;
            assert_eq!(
                bytes.pressure,
                (hi - lo) * plane * size_of::<Fix128>(),
                "rank {r}: the pressure band is not the resident layers",
            );
            assert_eq!(
                bytes.faces,
                (owned * (u_plane(n, n) + v_plane(n, n)) + (owned + 1) * plane)
                    * size_of::<Fix128>(),
                "rank {r}: the face arrays are not the owned layers plus one Z-face layer",
            );
            assert_eq!(
                bytes.open,
                owned * plane * size_of::<[bool; 6]>(),
                "rank {r}: the mask is not the owned layers",
            );
            assert_eq!(bytes.rhs, owned * plane * size_of::<Fix128>());
            assert_eq!(bytes.inverse_degrees, owned * plane * size_of::<Fix128>());
            assert!(
                bytes.total() > 0,
                "rank {r} reported an empty working set, which would make every \
                 comparison against it vacuous",
            );
        }
    }

    /// One rank's slab-local working set is a fraction of what the same rank
    /// needs on the full-length path, measured on both sides from the
    /// containers that actually get allocated.
    ///
    /// The full-length figure is the one the scale probe under-reported: the
    /// grid is only four of the seven arrays, and the mask, the reciprocal
    /// diagonal and the right-hand side are the other three.
    #[test]
    fn the_slab_local_working_set_is_a_fraction_of_the_full_length_one() {
        let n = 64usize;
        let ranks = 8usize;
        let grid = seed_slab_case(n, true, false);
        let cells = n * n * n;
        let plane = cell_plane(n, n);

        let mask = PoissonMask::from_grid(&grid);
        let inv_deg = inverse_degrees(&mask, cells);
        let rhs = poisson_rhs(&grid, Fix128::ONE);
        let full_length_transport = LocalTransport::new(1, n, plane);
        let full_length = (grid.u.capacity()
            + grid.v.capacity()
            + grid.w.capacity()
            + grid.pressure.capacity()
            + inv_deg.capacity()
            + rhs.capacity()
            + full_length_transport.slabs[0].capacity())
            * size_of::<Fix128>()
            + mask.open.capacity() * size_of::<[bool; 6]>();

        let b = slab_bounds(n, ranks, ranks / 2);
        let faces = SlabFaces::from_grid(&grid, b);
        let storage = SlabStorage::for_slab(plane, n, b, 1);
        let stencil = SlabStencil::build(&faces, Fix128::ONE);
        let slab_local = storage
            .bytes()
            .add(faces.bytes())
            .add(stencil.bytes())
            .total();

        assert!(
            slab_local * 4 < full_length,
            "{n}³ over {ranks} ranks: slab-local storage is {slab_local} B per rank against \
             {full_length} B for the full-length path, which is less than the 4x the \
             decomposition is for",
        );
    }

    /// One rank of a hundred million cells, built and swept, with the resident
    /// bytes read back from the containers.
    ///
    /// Ignored by default because it allocates over a gigabyte and runs for
    /// tens of seconds; it is the measurement this stage exists to make, run by
    /// hand with
    /// `cargo test --release --features std -- --ignored --nocapture
    /// one_rank_of_a_hundred_million_cells`.
    ///
    /// No peer rank, so the halo keeps its initial contents: what this measures
    /// is residency and sweep cost at the target size. The answer is pinned
    /// bit-exactly by `slab_local_storage_reproduces_the_monolithic_pressure_solve`
    /// at sizes a test suite can afford.
    #[cfg(feature = "std")]
    #[test]
    #[ignore = "allocates over 1 GiB and runs for tens of seconds: the 1e8-cell residency measurement"]
    fn one_rank_of_a_hundred_million_cells_fits_in_slab_local_storage() {
        let n = 464usize; // 464³ = 99,897,344 cells
        let ranks = 8usize;
        let rank = 4usize; // interior: a halo on both sides
        let dx = Fix128::from_ratio(1, 100);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let plane = cell_plane(n, n);
        let b = slab_bounds(n, ranks, rank);
        let (k0, k1) = b;

        let t0 = std::time::Instant::now();
        let mut faces = SlabFaces::new(n, n, n, dx, b);
        for k in k0..k1 {
            let base = k * u_plane(n, n);
            let (u, _) = faces.u_layer_mut(k);
            for (i, slot) in u.iter_mut().enumerate() {
                *slot = probe_u(base + i);
            }
        }
        let mut storage = SlabStorage::for_slab(plane, n, b, 1);
        let seeded = t0.elapsed();

        let t1 = std::time::Instant::now();
        let stencil = SlabStencil::build(&faces, rho * dx * dx / dt);
        let stencil_built = t1.elapsed();

        let bytes = storage.bytes().add(faces.bytes()).add(stencil.bytes());

        let t2 = std::time::Instant::now();
        for colour in 0..2u32 {
            sweep_slab_colour(&mut storage, &faces, &stencil, colour);
        }
        let swept = t2.elapsed();

        let owned = (k1 - k0) * plane;
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "rank {rank} of {ranks}, {n}³ = {} cells, owns layers {k0}..{k1} ({owned} cells)",
            n * n * n,
        );
        println!(
            "  pressure band {:.3} GiB / faces {:.3} GiB / face conditions {:.3} GiB",
            gib(bytes.pressure),
            gib(bytes.faces),
            gib(bytes.face_flags),
        );
        println!(
            "  mask {:.3} GiB / inverse degrees {:.3} GiB / right-hand side {:.3} GiB",
            gib(bytes.open),
            gib(bytes.inverse_degrees),
            gib(bytes.rhs),
        );
        println!(
            "  total per rank {:.3} GiB, times {ranks} ranks = {:.3} GiB",
            gib(bytes.total()),
            gib(bytes.total() * ranks),
        );
        println!(
            "  seed {seeded:?} / stencil {stencil_built:?} / one iteration (both colours) \
             {swept:?} = {:.1} ns per owned cell",
            swept.as_secs_f64() * 1.0e9 / owned as f64,
        );

        assert!(
            bytes.total() < 2 * 1024 * 1024 * 1024,
            "one rank of {n}³ over {ranks} ranks needs {:.3} GiB, which is not the \
             gigabyte-class residency the decomposition is for",
            gib(bytes.total()),
        );
    }

    /// The X-face velocity the probes seed, by *global* face index, so the
    /// monolithic grid and the slab-local ranks start from the same field and
    /// their timings are comparable.
    fn probe_u(global_face: usize) -> Fix128 {
        Fix128::from_f64(((global_face % 7) as f64 - 3.0) * 0.1)
    }

    /// The whole decomposition — all eight ranks, the exchange, each rank
    /// correcting its own faces — timed against the monolithic solve on the
    /// same machine, at the same size, from the same field.
    ///
    /// This is the measurement that says whether localising the storage costs
    /// anything per cell: the slab path does more index arithmetic and reads
    /// through one more level of slicing, and a decomposition that paid for
    /// that in the kernel would show up here. The two solves are run one after
    /// the other, the monolithic one dropped before the ranks are built, so a
    /// 16 GiB machine only ever holds one of them.
    ///
    /// Ignored for the same reason as the test above. Run with
    /// `cargo test --release --features std -- --ignored --nocapture
    /// the_whole_slab_decomposition_costs`.
    #[cfg(feature = "std")]
    #[test]
    #[ignore = "allocates about 2 GiB twice over: the multi-rank timing measurement"]
    fn the_whole_slab_decomposition_costs_what_the_monolithic_solve_does_per_cell() {
        let n = 256usize;
        let ranks = 8usize;
        let dx = Fix128::from_ratio(1, 100);
        let dt = Fix128::from_ratio(1, 100);
        let rho = Fix128::from_int(1000);
        let plane = cell_plane(n, n);
        let cells = n * n * n;
        let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(n, ranks, r)).collect();
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);

        let monolithic = {
            let mut grid = MacGrid::new(n, n, n, dx);
            for (i, slot) in grid.u.iter_mut().enumerate() {
                *slot = probe_u(i);
            }
            let t = std::time::Instant::now();
            project_pressure_red_black_gs(&mut grid, dt, rho, 1);
            t.elapsed()
        };

        let t0 = std::time::Instant::now();
        let mut faces: Vec<SlabFaces> = bounds
            .iter()
            .map(|&b| {
                let mut f = SlabFaces::new(n, n, n, dx, b);
                for k in b.0..b.1 {
                    let base = k * u_plane(n, n);
                    let (u, _) = f.u_layer_mut(k);
                    for (i, slot) in u.iter_mut().enumerate() {
                        *slot = probe_u(base + i);
                    }
                }
                f
            })
            .collect();
        let mut transport = LocalSlabTransport {
            slabs: bounds
                .iter()
                .map(|&b| SlabStorage::for_slab(plane, n, b, 1))
                .collect(),
            staging: vec![Fix128::ZERO; plane],
        };
        let seeded = t0.elapsed();

        let t1 = std::time::Instant::now();
        project_pressure_slab_local_over(
            &mut faces,
            dt,
            rho,
            1,
            HaloSchedule::EverySweep,
            &mut transport,
        );
        let solved = t1.elapsed();

        let per_rank: Vec<usize> = faces
            .iter()
            .zip(transport.bytes())
            .map(|(f, storage_bytes)| {
                storage_bytes
                    .add(f.bytes())
                    .add(SlabStencil::build(f, rho * dx * dx / dt).bytes())
                    .total()
            })
            .collect();
        let total: usize = per_rank.iter().sum();
        println!(
            "{n}³ = {cells} cells, one red-black iteration from the same field:\n  \
             monolithic     {monolithic:?} = {:.1} ns/cell\n  \
             {ranks} slab-local  {solved:?} = {:.1} ns/cell (seed {seeded:?})\n  \
             working set {:.3} GiB over {ranks} ranks, largest rank {:.3} GiB",
            monolithic.as_secs_f64() * 1.0e9 / cells as f64,
            solved.as_secs_f64() * 1.0e9 / cells as f64,
            gib(total),
            gib(per_rank.iter().copied().max().unwrap_or(0)),
        );
        assert!(
            solved.as_secs_f64() < 4.0 * monolithic.as_secs_f64(),
            "the slab-local decomposition took {solved:?} against the monolithic solve's \
             {monolithic:?}, so localising the storage has moved the cost into the index \
             arithmetic instead of leaving it in the memory system",
        );
    }

    // ========================================================================
    // Slab-local storage across process boundaries
    // ========================================================================

    /// What a slab case puts in the grid before the ranks split it, and so which
    /// parts of the enforcement it exercises.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum SlabScene {
        /// Divergent flow and no conditions at all: every face plain fluid,
        /// every cell of degree six.
        Open,
        /// Closed box, so the Poisson mask is non-trivial, plus a non-zero
        /// initial pressure, so the initial halo is load-bearing.
        ClosedBox,
        /// The conditions whose enforcement crosses a slab boundary: every
        /// *interior* Z-face layer of odd index is an outflow, so the first
        /// Z-face of a rank whose band starts on an odd layer copies a layer the
        /// rank below owns. Also an X-inflow (a prescribed value, which
        /// [`FaceFlags`] cannot carry), X-outflow (propagation inside a layer),
        /// Y-walls set through the dense flag, and `SlipWall` on the top and
        /// bottom Z-faces (a map entry that is neither inflow nor outflow).
        ///
        /// The interior outflow is the point: see
        /// `a_z_outflow_on_the_domain_boundary_alone_needs_no_face_halo`.
        ZOutflow,
    }

    impl SlabScene {
        /// Name passed to a child rank through the environment.
        fn as_str(self) -> &'static str {
            match self {
                Self::Open => "open",
                Self::ClosedBox => "closed",
                Self::ZOutflow => "zoutflow",
            }
        }

        /// Inverse of [`Self::as_str`]; an unknown name is a harness bug.
        fn parse(name: &str) -> Self {
            match name {
                "open" => Self::Open,
                "closed" => Self::ClosedBox,
                "zoutflow" => Self::ZOutflow,
                other => panic!("unknown slab scene `{other}`"),
            }
        }
    }

    /// The field every rank of a case starts from — a pure function of `(n,
    /// scene)`, so processes that never speak to each other still agree about
    /// the problem.
    fn seed_slab_scene(n: usize, scene: SlabScene) -> MacGrid {
        let mut grid = seed_divergent_flow(n);
        let varied_pressure = |grid: &mut MacGrid| {
            for (c, slot) in grid.pressure.iter_mut().enumerate() {
                *slot = Fix128::from_ratio((c % 5) as i64 - 2, 7);
            }
        };
        match scene {
            SlabScene::Open => {}
            SlabScene::ClosedBox => {
                grid.set_closed_box_walls();
                varied_pressure(&mut grid);
            }
            SlabScene::ZOutflow => {
                // A Z-velocity that differs from layer to layer, so an outflow
                // face copying the wrong layer is a different number rather than
                // the same one.
                for k in 0..=n {
                    for j in 0..n {
                        for i in 0..n {
                            let ix = grid.idx_w(i, j, k);
                            grid.w[ix] =
                                Fix128::from_ratio(((k * n + i + 2 * j) % 11) as i64 - 5, 4);
                        }
                    }
                }
                varied_pressure(&mut grid);
                for j in 0..n {
                    for i in 0..n {
                        grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
                        grid.set_w_bc(i, j, n, FaceBc::SlipWall);
                        for k in (1..n).filter(|k| k % 2 == 1) {
                            grid.set_w_bc(i, j, k, FaceBc::Outflow);
                        }
                    }
                }
                for k in 0..n {
                    for j in 0..n {
                        grid.set_u_bc(
                            0,
                            j,
                            k,
                            FaceBc::Inflow {
                                normal_velocity: Fix128::from_ratio(3, 4),
                            },
                        );
                        grid.set_u_bc(n, j, k, FaceBc::Outflow);
                    }
                    for i in 0..n {
                        grid.set_v_bc(
                            i,
                            0,
                            k,
                            FaceBc::Wall {
                                velocity: Vec3Fix::ZERO,
                            },
                        );
                        grid.set_v_bc(
                            i,
                            n,
                            k,
                            FaceBc::Wall {
                                velocity: Vec3Fix::ZERO,
                            },
                        );
                    }
                }
            }
        }
        grid
    }

    /// How many ranks of this decomposition have a first Z-face that is an
    /// outflow reading a layer the rank below owns, under
    /// [`SlabScene::ZOutflow`]'s rule.
    ///
    /// Zero would mean the crossing this stage exists to make is not being made,
    /// so the oracles assert it is not zero rather than trusting the arithmetic
    /// to work out.
    fn z_outflow_crossings(nz: usize, ranks: usize) -> usize {
        (0..ranks)
            .map(|r| slab_bounds(nz, ranks, r))
            .filter(|&(k0, k1)| k0 != k1 && k0 > 0 && k0 < nz && k0 % 2 == 1)
            .count()
    }

    /// One rank's band, filled from the field the solve starts from — its halo
    /// layers included, because the first colour sweep reads them.
    ///
    /// Apparatus: in a real run a rank would generate its own band, and the
    /// initial halo would be the one exchange this does not model.
    fn band_for_rank(
        plane: usize,
        nz: usize,
        b: (usize, usize),
        halo: usize,
        field: &[Fix128],
    ) -> SlabStorage {
        let mut slab = SlabStorage::for_slab(plane, nz, b, halo);
        let (lo, hi) = slab.resident();
        for k in lo..hi {
            slab.layer_mut(k)
                .expect("a layer inside the band this slab just reported")
                .copy_from_slice(&field[k * plane..(k + 1) * plane]);
        }
        slab
    }

    /// Put the ranks' faces back together into one grid, so the conditions they
    /// imposed can be compared with the monolithic ones.
    ///
    /// Apparatus, not part of the decomposition — the role `assemble_slabs`
    /// plays for the solved field.
    fn assemble_slab_faces(template: &MacGrid, faces: &[SlabFaces]) -> MacGrid {
        let mut out = template.clone();
        out.u.fill(UNWRITTEN);
        out.v.fill(UNWRITTEN);
        out.w.fill(UNWRITTEN);
        let (nx, ny) = (out.nx, out.ny);
        for rank_faces in faces {
            let (k0, k1) = rank_faces.owned();
            for k in k0..k1 {
                let (u, _) = rank_faces.u_layer(k);
                for j in 0..ny {
                    for i in 0..=nx {
                        let ix = out.idx_u(i, j, k);
                        out.u[ix] = u[i + (nx + 1) * j];
                    }
                }
                let (v, _) = rank_faces.v_layer(k);
                for j in 0..=ny {
                    for i in 0..nx {
                        let ix = out.idx_v(i, j, k);
                        out.v[ix] = v[i + nx * j];
                    }
                }
            }
            for k in rank_faces.w_written() {
                let (w, _) = rank_faces.w_layer(k);
                for j in 0..ny {
                    for i in 0..nx {
                        let ix = out.idx_w(i, j, k);
                        out.w[ix] = w[i + nx * j];
                    }
                }
            }
        }
        out
    }

    /// Split `base` across `ranks` slabs and impose the face conditions rank by
    /// rank through the pipeline, in one process.
    fn enforce_in_slabs(base: &MacGrid, ranks: usize) -> Vec<SlabFaces> {
        let nz = base.nz;
        let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
        let mut faces: Vec<SlabFaces> = bounds
            .iter()
            .map(|&b| SlabFaces::from_grid(base, b))
            .collect();
        let cond: Vec<SlabFaceConditions> = bounds
            .iter()
            .map(|&b| SlabFaceConditions::from_grid(base, b))
            .collect();
        enforce_slab_face_boundaries_over(&mut faces, &cond);
        faces
    }

    /// Imposing the face conditions rank by rank, with one Z-face layer crossing
    /// each slab boundary, gives the field `MacGrid::enforce_face_boundaries`
    /// gives — bit for bit, over every rank count the decomposition oracle uses.
    ///
    /// Exactness, not a tolerance: every value is written by the same expression
    /// in the same order as the monolithic enforcement, so a distribution that
    /// is right at all is right to the bit.
    ///
    /// The scenes are not interchangeable. `Open` has no conditions and would
    /// pass over an enforcement that did nothing; `ClosedBox` adds walls the
    /// *dense* flag carries; `ZOutflow` adds the three things only the sparse
    /// conditions carry — a prescribed inflow value, an outflow, and a
    /// `SlipWall` — and is the only one whose enforcement leaves a rank's band.
    #[test]
    fn distributed_face_enforcement_matches_the_monolithic_one() {
        let mut crossings = 0usize;
        for &(n, ranks) in &SLAB_CASES {
            crossings += z_outflow_crossings(n, ranks);
            for scene in [SlabScene::Open, SlabScene::ClosedBox, SlabScene::ZOutflow] {
                let base = seed_slab_scene(n, scene);
                let mut monolithic = base.clone();
                monolithic.enforce_face_boundaries();

                let faces = enforce_in_slabs(&base, ranks);
                let assembled = assemble_slab_faces(&base, &faces);

                assert_eq!(
                    assembled.u, monolithic.u,
                    "{n}³ over {ranks} ranks ({scene:?}): the X-face conditions the ranks \
                     imposed differ from the monolithic ones",
                );
                assert_eq!(
                    assembled.v, monolithic.v,
                    "{n}³ over {ranks} ranks ({scene:?}): the Y-face conditions the ranks \
                     imposed differ from the monolithic ones",
                );
                assert_eq!(
                    assembled.w, monolithic.w,
                    "{n}³ over {ranks} ranks ({scene:?}): the Z-face conditions the ranks \
                     imposed differ from the monolithic ones — the layer crossing a slab \
                     boundary is the one an outflow face reads",
                );
            }
        }
        assert!(
            crossings > 0,
            "no rank of any case had a first Z-face that reads across a slab boundary, so \
             the crossing this stage adds was never exercised",
        );
    }

    /// A rank holds one Z-face layer it does not write — the one above its last
    /// cell, which its own divergence needs — and recomputes the conditions on
    /// it rather than taking the neighbour's answer over a second message. The
    /// two answers are the same one.
    ///
    /// This is the claim that licenses the enforcement having only an upward
    /// crossing. Without it the rank's right-hand side would be built from a
    /// layer nobody had imposed the conditions on.
    #[test]
    fn both_ranks_sharing_a_z_face_layer_enforce_it_to_the_same_value() {
        let mut shared = 0usize;
        for &(n, ranks) in &SLAB_CASES {
            let base = seed_slab_scene(n, SlabScene::ZOutflow);
            let mut monolithic = base.clone();
            monolithic.enforce_face_boundaries();

            for (r, rank_faces) in enforce_in_slabs(&base, ranks).iter().enumerate() {
                let (k0, k1) = rank_faces.owned();
                if k0 == k1 || k1 == n {
                    continue; // owns nothing, or its top layer is the domain's
                }
                let (w, _) = rank_faces.w_layer(k1);
                for j in 0..n {
                    for i in 0..n {
                        assert_eq!(
                            w[i + n * j],
                            monolithic.w[monolithic.idx_w(i, j, k1)],
                            "{n}³ over {ranks} ranks: rank {r} recomputed the conditions on \
                             Z-face layer {k1}, which rank above it writes, and got a \
                             different value at ({i}, {j})",
                        );
                    }
                }
                shared += 1;
            }
        }
        assert!(
            shared > 0,
            "no rank held a Z-face layer it does not write, so the recomputation this test \
             exists to check never happened",
        );
    }

    /// The crossing is reachable only through an *interior* Z-outflow: with the
    /// outflow on the domain boundary alone, no rank ever reads a layer outside
    /// its band, and the enforcement runs with nothing handed over.
    ///
    /// Measured, not reasoned: the reason is that the first Z-face a rank writes
    /// is the one at `k0`, which for `k0 > 0` is an interior layer, while the
    /// boundary faces at `0` and `nz` are each held by a rank that also holds the
    /// layer they read. [`FaceBc::Outflow`]'s own documentation says the
    /// condition is "only meaningful on the domain boundary", so this test and
    /// the one below record together that the halo the distributed enforcement
    /// carries is load-bearing only for a configuration the crate calls
    /// meaningless — and that it is load-bearing there.
    #[test]
    fn a_z_outflow_on_the_domain_boundary_alone_needs_no_face_halo() {
        // Counted, not merely observed: "no rank needed it" is indistinguishable
        // from "no rank was asked" without the number of ranks that ran.
        let mut ran = 0usize;
        let mut outflow_faces = 0usize;
        for &(n, ranks) in &SLAB_CASES {
            let mut base = seed_divergent_flow(n);
            for j in 0..n {
                for i in 0..n {
                    base.set_w_bc(i, j, 0, FaceBc::Outflow);
                    base.set_w_bc(i, j, n, FaceBc::Outflow);
                }
            }
            let mut monolithic = base.clone();
            monolithic.enforce_face_boundaries();

            let bounds: Vec<(usize, usize)> =
                (0..ranks).map(|r| slab_bounds(n, ranks, r)).collect();
            let mut faces: Vec<SlabFaces> = bounds
                .iter()
                .map(|&b| SlabFaces::from_grid(&base, b))
                .collect();
            for (r, rank_faces) in faces.iter_mut().enumerate() {
                let cond = SlabFaceConditions::from_grid(&base, bounds[r]);
                outflow_faces += cond
                    .w
                    .iter()
                    .filter(|&&(_, bc)| bc == FaceBc::Outflow)
                    .count();
                // `None`: nothing crosses the boundary below.
                enforce_slab_face_boundaries(rank_faces, &cond, None);
                ran += 1;
            }
            let assembled = assemble_slab_faces(&base, &faces);
            assert_eq!(
                assembled.w, monolithic.w,
                "{n}³ over {ranks} ranks: a Z-outflow on the domain boundary alone did not \
                 reproduce the monolithic enforcement without a halo",
            );
        }
        let expected_ranks: usize = SLAB_CASES.iter().map(|&(_, ranks)| ranks).sum();
        assert_eq!(
            ran, expected_ranks,
            "only {ran} of {expected_ranks} ranks ran, so the ones that did not are not \
             evidence of anything",
        );
        assert!(
            outflow_faces > 0,
            "the ranks held no outflow Z-face between them, so running without a halo \
             proves nothing",
        );
        println!(
            "boundary-only Z-outflow: {ran} ranks over {} cases imposed {outflow_faces} \
             outflow Z-faces with nothing handed over",
            SLAB_CASES.len(),
        );
    }

    /// The other half of the pair above: put an outflow on an interior Z-face and
    /// the enforcement aborts without the layer below, at the read, naming the
    /// rank below as the one that has to hand it over.
    #[test]
    #[should_panic(expected = "which the rank below has to hand over")]
    fn an_interior_z_outflow_is_what_makes_the_face_halo_load_bearing() {
        let n = 8usize;
        let base = seed_slab_scene(n, SlabScene::ZOutflow);
        // Rank 1 of 8 owns layer 1 only, and layer 1 is an outflow under
        // `ZOutflow`'s rule, so its first written Z-face reads layer 0.
        let b = slab_bounds(n, 8, 1);
        assert_eq!(
            b,
            (1, 2),
            "the rank this test aims at owns a different layer"
        );
        let mut faces = SlabFaces::from_grid(&base, b);
        let cond = SlabFaceConditions::from_grid(&base, b);
        enforce_slab_face_boundaries(&mut faces, &cond, None);
    }

    /// Naming the conditions face by face, as a rank with no [`MacGrid`] must,
    /// gives the same lists as reading them out of a grid.
    ///
    /// Without this the only way to build a [`SlabFaceConditions`] that anything
    /// checks would be from a resident whole grid, which is the thing slab-local
    /// storage exists to avoid.
    #[test]
    fn conditions_named_face_by_face_match_the_ones_read_from_a_grid() {
        let n = 6usize;
        let ranks = 3usize;
        let base = seed_slab_scene(n, SlabScene::ZOutflow);
        for r in 0..ranks {
            let b = slab_bounds(n, ranks, r);
            let from_grid = SlabFaceConditions::from_grid(&base, b);
            let mut by_hand = SlabFaceConditions::new(n, n, b);
            if b.0 != b.1 {
                for k in b.0..b.1 {
                    for j in 0..n {
                        for i in 0..=n {
                            by_hand.set_u(i, j, k, base.u_bc(i, j, k));
                        }
                    }
                    for j in 0..=n {
                        for i in 0..n {
                            by_hand.set_v(i, j, k, base.v_bc(i, j, k));
                        }
                    }
                }
                for k in b.0..=b.1 {
                    for j in 0..n {
                        for i in 0..n {
                            by_hand.set_w(i, j, k, base.w_bc(i, j, k));
                        }
                    }
                }
            }
            assert_eq!(
                from_grid.u, by_hand.u,
                "rank {r}: the X-face conditions differ"
            );
            assert_eq!(
                from_grid.v, by_hand.v,
                "rank {r}: the Y-face conditions differ"
            );
            assert_eq!(
                from_grid.w, by_hand.w,
                "rank {r}: the Z-face conditions differ"
            );
            assert!(
                !from_grid.w.is_empty() || b.0 == b.1,
                "rank {r} read no Z-face conditions out of a scene that sets them on every \
                 layer, so the comparison above is vacuous",
            );
        }
    }

    /// Records which planes a face-condition pipeline carries, and between which
    /// ranks.
    ///
    /// Not a stand-in for a channel: it moves nothing, and the test that uses it
    /// compares the sequence against one derived from the partition rather than
    /// from a run.
    struct PlaneRecorder {
        plane: usize,
        /// `(dst, layer)` per send, in order.
        sent: Vec<(usize, usize)>,
        /// `(src, layer)` per receive, in order.
        received: Vec<(usize, usize)>,
    }

    impl PlaneChannel for PlaneRecorder {
        fn send_plane(&mut self, dst: usize, layer: usize, plane: &[Fix128]) {
            assert_eq!(
                plane.len(),
                self.plane,
                "a plane of the wrong size was offered"
            );
            self.sent.push((dst, layer));
        }

        fn recv_plane(&mut self, src: usize, layer: usize, plane: &mut [Fix128]) {
            assert_eq!(
                plane.len(),
                self.plane,
                "a plane of the wrong size was asked for"
            );
            self.received.push((src, layer));
        }
    }

    /// The face-condition pipeline carries exactly one Z-face layer across each
    /// boundary between consecutive *non-empty* ranks, in ascending order, and
    /// the layer it carries is the last one the sending rank writes.
    ///
    /// Derived from the partition, not from a solve, so it holds for
    /// decompositions no test could afford to run — and it is what says a
    /// decomposition with empty ranks still forms one chain rather than breaking
    /// into pieces at the gaps.
    #[test]
    fn the_face_condition_pipeline_carries_one_plane_per_slab_boundary() {
        for ranks in 1usize..=6 {
            for nz in [ranks, 2 * ranks + 1, 3 * ranks, 1] {
                let bounds: Vec<(usize, usize)> =
                    (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
                let busy: Vec<usize> = (0..ranks).filter(|&r| bounds[r].0 != bounds[r].1).collect();
                let expected: Vec<(usize, usize, usize)> = busy
                    .windows(2)
                    .map(|pair| (pair[0], pair[1], bounds[pair[0]].1 - 1))
                    .collect();

                let plane = cell_plane(2, 2);
                let mut recorder = PlaneRecorder {
                    plane,
                    sent: Vec::new(),
                    received: Vec::new(),
                };
                for (r, &b) in bounds.iter().enumerate() {
                    let mut faces = SlabFaces::new(2, 2, nz, Fix128::ONE, b);
                    let cond = SlabFaceConditions::new(2, 2, b);
                    enforce_slab_face_boundaries_on_rank(
                        &mut faces,
                        &cond,
                        ranks,
                        r,
                        &mut recorder,
                    );
                }

                assert_eq!(
                    recorder.sent.len(),
                    recorder.received.len(),
                    "{nz} layers over {ranks} ranks: {} planes were offered and {} asked for",
                    recorder.sent.len(),
                    recorder.received.len(),
                );
                let walked: Vec<(usize, usize, usize)> = recorder
                    .received
                    .iter()
                    .zip(&recorder.sent)
                    .map(|(&(src, layer), &(dst, sent_layer))| {
                        assert_eq!(
                            layer, sent_layer,
                            "{nz} layers over {ranks} ranks: layer {sent_layer} was offered \
                             and layer {layer} asked for",
                        );
                        (src, dst, layer)
                    })
                    .collect();
                assert_eq!(
                    walked, expected,
                    "{nz} layers over {ranks} ranks: the pipeline carried a different \
                     sequence of planes than the partition implies",
                );
            }
        }
    }

    /// The two transports put the same bytes on the wire for the same values.
    ///
    /// [`SlabSocketTransport`] encodes through [`encode_plane`] and
    /// [`SocketTransport`] has the loop written out inside `deliver_layer`; this
    /// compares one against the other instead of trusting two copies of an
    /// encoding to stay the same. The decode is checked to be the encode's
    /// inverse in the same place, so a symmetric change to both halves of
    /// [`encode_plane`] / [`decode_plane`] is at least visible against the other
    /// transport.
    #[cfg(feature = "std")]
    #[test]
    fn the_two_wire_encodings_agree() {
        use std::io::Cursor;

        let plane = 5usize;
        let values: Vec<Fix128> = (0..plane)
            .map(|c| Fix128::from_ratio(c as i64 * 7 - 11, 3))
            .collect();
        let mut by_hand = vec![0u8; plane * WIRE_BYTES_PER_CELL];
        encode_plane(&values, &mut by_hand);

        let mut transport: SocketTransport<Cursor<Vec<u8>>> = SocketTransport::new(
            0,
            plane * 2,
            plane,
            vec![None, Some(Cursor::new(Vec::new()))],
        );
        transport.slab_mut(0)[plane..].copy_from_slice(&values);
        transport.deliver_layer(0, 1, 1);
        let on_the_wire = transport.links[1]
            .as_ref()
            .expect("the link the delivery went out on")
            .get_ref()
            .clone();

        assert_eq!(
            on_the_wire, by_hand,
            "the slab transport's encoding differs from the full-length transport's, so the \
             two cannot be peers on one wire",
        );
        assert!(
            !by_hand.iter().all(|&b| b == 0),
            "the encoding produced only zero bytes, so the comparison above is vacuous",
        );

        let mut back = vec![Fix128::ZERO; plane];
        decode_plane(&by_hand, &mut back);
        assert_eq!(back, values, "decoding is not the inverse of encoding");
    }

    // ---- several processes, each holding one band ---------------------------
    //
    // The in-process oracles above run every rank from one address space, where
    // `LocalSlabTransport` can always reach a neighbour's band. That is enough to
    // fix the decomposition and not enough to fix the *distribution*: a driver
    // that quietly read another rank's storage would pass every one of them. Here
    // each rank is a process, its band is the only one it has, and a halo layer
    // is bytes on a loopback stream.
    //
    // Ranks connect as a full mesh, because the schedule addresses the rank that
    // *owns* a layer rather than the rank with the adjacent index, and with empty
    // ranks those are not the same. Rank 0 brokers the port table: every child
    // binds a listener, tells rank 0 its port, and rank 0 hands the table back,
    // after which each rank dials every lower rank and announces which rank it
    // is. Nothing races for a fixed port and no rank has to be told the topology.

    /// Which rank a re-executed test binary is.
    #[cfg(feature = "std")]
    const XSLAB_RANK: &str = "ALICE_PHYSICS_XSLAB_RANK";
    /// How many ranks the decomposition has.
    #[cfg(feature = "std")]
    const XSLAB_RANKS: &str = "ALICE_PHYSICS_XSLAB_RANKS";
    /// Loopback port rank 0 brokers the port table on.
    #[cfg(feature = "std")]
    const XSLAB_BROKER: &str = "ALICE_PHYSICS_XSLAB_BROKER";
    /// Edge length of the cubic grid every rank seeds.
    #[cfg(feature = "std")]
    const XSLAB_SIZE: &str = "ALICE_PHYSICS_XSLAB_SIZE";
    /// Which [`SlabScene`] every rank seeds.
    #[cfg(feature = "std")]
    const XSLAB_SCENE: &str = "ALICE_PHYSICS_XSLAB_SCENE";
    /// Which fault, if any, every rank applies.
    #[cfg(feature = "std")]
    const XSLAB_FAULT: &str = "ALICE_PHYSICS_XSLAB_FAULT";

    /// libtest name of the child entry point, passed to the re-executed binary as
    /// `--exact`.
    ///
    /// Renaming `cross_process_slab_rank_worker` without updating this would make
    /// libtest match nothing and exit 0, so the accept loops below report a child
    /// that exits before connecting instead of waiting out their deadline.
    #[cfg(feature = "std")]
    const XSLAB_WORKER: &str = "eulerian_grid::tests::cross_process_slab_rank_worker";

    /// Iterations every rank runs.
    #[cfg(feature = "std")]
    const XSLAB_ITERATIONS: u32 = 6;

    /// A fault applied to a crossing, identically on every rank so the streams
    /// stay in step and the run produces a wrong answer rather than a hang.
    #[cfg(feature = "std")]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum SlabCrossFault {
        /// Carry what the schedule asks for.
        None,
        /// Perform no pressure halo delivery at all.
        NoPressureHalo,
        /// Receive the Z-face plane and then impose the conditions as though it
        /// had been zero.
        BlankFaceHalo,
    }

    #[cfg(feature = "std")]
    impl SlabCrossFault {
        /// Name passed to a child rank through the environment.
        fn as_str(self) -> &'static str {
            match self {
                Self::None => "none",
                Self::NoPressureHalo => "nohalo",
                Self::BlankFaceHalo => "blankface",
            }
        }

        /// Inverse of [`Self::as_str`]; an unknown name is a harness bug.
        fn parse(name: &str) -> Self {
            match name {
                "none" => Self::None,
                "nohalo" => Self::NoPressureHalo,
                "blankface" => Self::BlankFaceHalo,
                other => panic!("unknown slab cross-process fault `{other}`"),
            }
        }
    }

    /// Drops every pressure halo delivery, and changes nothing else.
    ///
    /// Not a stub standing in for unwritten code: the omission is the
    /// measurement, and the test that uses it asserts the run stops agreeing
    /// with the single-process solve. Dropped on every rank, so no stream is left
    /// with an unread layer.
    #[cfg(feature = "std")]
    struct DroppedHalo<T> {
        inner: T,
    }

    #[cfg(feature = "std")]
    impl<T: SlabTransport> SlabTransport for DroppedHalo<T> {
        fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage {
            self.inner.slab_mut(rank)
        }

        fn deliver_layer(&mut self, _src: usize, _dst: usize, _layer: usize) {}
    }

    /// Receives the Z-face plane the pipeline carries and hands the caller zeros
    /// instead, and changes nothing else.
    ///
    /// Receiving it first is deliberate: dropping the receive would desynchronise
    /// the stream, and a timeout says nothing about whether the plane's contents
    /// mattered.
    #[cfg(feature = "std")]
    struct BlankedPlane<C> {
        inner: C,
        scratch: Vec<Fix128>,
    }

    #[cfg(feature = "std")]
    impl<C: PlaneChannel> PlaneChannel for BlankedPlane<C> {
        fn send_plane(&mut self, dst: usize, layer: usize, plane: &[Fix128]) {
            self.inner.send_plane(dst, layer, plane);
        }

        fn recv_plane(&mut self, src: usize, layer: usize, plane: &mut [Fix128]) {
            self.inner.recv_plane(src, layer, &mut self.scratch);
            plane.fill(Fix128::ZERO);
        }
    }

    /// What one rank found when it compared the layers it owns against the
    /// single-process solve: mismatching pressure cells, mismatching face
    /// values, the first layer that differs (or `u32::MAX`), and how many cells
    /// it compared.
    ///
    /// The last field is there because a comparison of nothing satisfies every
    /// assertion made with it; the parent adds the counts up and requires the
    /// whole domain.
    #[cfg(feature = "std")]
    fn compare_band(reference: &MacGrid, faces: &SlabFaces, slab: &SlabStorage) -> [u32; 4] {
        let (nx, ny) = (reference.nx, reference.ny);
        let plane = cell_plane(nx, ny);
        let (k0, k1) = faces.owned();
        let mut bad_cells = 0u32;
        let mut bad_faces = 0u32;
        let mut first_bad = u32::MAX;
        let mut compared = 0u32;
        let mut note = |layer: usize| {
            first_bad = first_bad.min(layer as u32);
        };

        for k in k0..k1 {
            let mine = slab.layer(k).expect("a rank's own layer is resident");
            let theirs = &reference.pressure[k * plane..(k + 1) * plane];
            for (a, b) in mine.iter().zip(theirs) {
                compared += 1;
                if a != b {
                    bad_cells += 1;
                    note(k);
                }
            }
            let (u, _) = faces.u_layer(k);
            for j in 0..ny {
                for i in 0..=nx {
                    if u[i + (nx + 1) * j] != reference.u[reference.idx_u(i, j, k)] {
                        bad_faces += 1;
                        note(k);
                    }
                }
            }
            let (v, _) = faces.v_layer(k);
            for j in 0..=ny {
                for i in 0..nx {
                    if v[i + nx * j] != reference.v[reference.idx_v(i, j, k)] {
                        bad_faces += 1;
                        note(k);
                    }
                }
            }
        }
        for k in faces.w_written() {
            let (w, _) = faces.w_layer(k);
            for j in 0..ny {
                for i in 0..nx {
                    if w[i + nx * j] != reference.w[reference.idx_w(i, j, k)] {
                        bad_faces += 1;
                        note(k);
                    }
                }
            }
        }
        [bad_cells, bad_faces, first_bad, compared]
    }

    /// Run `my_rank`'s share of the slab-local solve over `links`, where
    /// `links[r]` is the stream to rank `r` and this rank's own entry is `None`,
    /// and report how it compares with the single-process solve.
    ///
    /// Shared by every process so none can drift from the others in iteration
    /// count, schedule, scene or fluid parameters. The rank builds its own faces
    /// from the *unenforced* field and imposes the conditions itself: that step
    /// is part of what is being distributed, not something the harness does for
    /// it. The reference solve each rank computes for the comparison is apparatus
    /// — the distributed path never holds a full-length array.
    #[cfg(feature = "std")]
    fn solve_slab_as_rank(
        n: usize,
        ranks: usize,
        my_rank: usize,
        scene: SlabScene,
        fault: SlabCrossFault,
        links: Vec<Option<std::net::TcpStream>>,
    ) -> [u32; 4] {
        let (dt, rho) = (xproc_dt(), xproc_rho());
        let base = seed_slab_scene(n, scene);
        let b = slab_bounds(n, ranks, my_rank);
        let plane = cell_plane(n, n);

        let mut faces = SlabFaces::from_grid(&base, b);
        let cond = SlabFaceConditions::from_grid(&base, b);
        let band = band_for_rank(plane, n, b, 1, &base.pressure);
        let mut socket = SlabSocketTransport::new(my_rank, plane, band, links);

        if fault == SlabCrossFault::BlankFaceHalo {
            let mut channel = BlankedPlane {
                inner: socket,
                scratch: vec![Fix128::ZERO; plane],
            };
            enforce_slab_face_boundaries_on_rank(&mut faces, &cond, ranks, my_rank, &mut channel);
            socket = channel.inner;
        } else {
            enforce_slab_face_boundaries_on_rank(&mut faces, &cond, ranks, my_rank, &mut socket);
        }

        if fault == SlabCrossFault::NoPressureHalo {
            let mut dropped = DroppedHalo { inner: socket };
            project_pressure_slab_local_on_rank(
                &mut faces,
                dt,
                rho,
                XSLAB_ITERATIONS,
                ranks,
                HaloSchedule::EverySweep,
                my_rank,
                &mut dropped,
            );
            socket = dropped.inner;
        } else {
            project_pressure_slab_local_on_rank(
                &mut faces,
                dt,
                rho,
                XSLAB_ITERATIONS,
                ranks,
                HaloSchedule::EverySweep,
                my_rank,
                &mut socket,
            );
        }

        let mut reference = base.clone();
        project_pressure_red_black_gs(&mut reference, dt, rho, XSLAB_ITERATIONS);
        compare_band(&reference, &faces, socket.slab())
    }

    /// Dial every lower rank and accept every higher one, so that `links[r]` is
    /// the stream to rank `r`.
    ///
    /// Dialling first cannot deadlock: every rank's listener is bound before its
    /// port reaches the table, so a connection completes into the backlog whether
    /// or not the peer has reached its own accept loop, and the rank number that
    /// follows it is two bytes.
    #[cfg(feature = "std")]
    fn join_slab_mesh(
        my_rank: usize,
        ranks: usize,
        ports: &[u16],
        listener: &std::net::TcpListener,
    ) -> Vec<Option<std::net::TcpStream>> {
        use std::io::{Read, Write};
        use std::net::TcpStream;

        let mut links: Vec<Option<TcpStream>> = (0..ranks).map(|_| None).collect();
        for lower in 0..my_rank {
            let mut link = TcpStream::connect(("127.0.0.1", ports[lower]))
                .unwrap_or_else(|e| panic!("rank {my_rank} dialling rank {lower}: {e}"));
            bound_peer_link(&link);
            link.write_all(&(my_rank as u16).to_le_bytes())
                .expect("announce which rank is dialling");
            link.flush().expect("flush the rank announcement");
            links[lower] = Some(link);
        }
        for _ in my_rank + 1..ranks {
            let mut link = accept_before_deadline(listener, "a higher rank");
            let mut who = [0u8; 2];
            link.read_exact(&mut who)
                .expect("learn which rank dialled in");
            let peer = usize::from(u16::from_le_bytes(who));
            assert!(
                peer > my_rank && peer < ranks && links[peer].is_none(),
                "rank {my_rank} was dialled by rank {peer}, which is not a higher rank it \
                 is still waiting for",
            );
            links[peer] = Some(link);
        }
        links
    }

    /// Ranks 1 and up of the slab-local solve, reached only when this binary has
    /// been re-executed with [`XSLAB_RANK`] set.
    ///
    /// One entry point for all of them: they run the same code and differ only in
    /// the rank they are told they are. On an ordinary `cargo test` run the
    /// variable is absent and this returns at once — the tests below are what ask
    /// for a child rank.
    #[cfg(feature = "std")]
    #[test]
    fn cross_process_slab_rank_worker() {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};

        let Ok(rank) = std::env::var(XSLAB_RANK) else {
            return;
        };
        let my_rank: usize = rank.parse().expect("the rank is a number");
        let ranks: usize = std::env::var(XSLAB_RANKS)
            .expect("rank count from rank 0")
            .parse()
            .expect("the rank count is a number");
        assert!(
            my_rank > 0 && my_rank < ranks,
            "rank {my_rank} is not a child rank of a {ranks}-rank harness",
        );
        let n: usize = std::env::var(XSLAB_SIZE)
            .expect("grid size from rank 0")
            .parse()
            .expect("grid size is a number");
        let broker: u16 = std::env::var(XSLAB_BROKER)
            .expect("broker port from rank 0")
            .parse()
            .expect("broker port is a number");
        let scene = SlabScene::parse(&std::env::var(XSLAB_SCENE).expect("scene from rank 0"));
        let fault = SlabCrossFault::parse(&std::env::var(XSLAB_FAULT).expect("fault from rank 0"));

        // Bound before the port is announced, so a peer that dials the moment it
        // sees the table finds a listener rather than a refusal.
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port");
        let my_port = listener
            .local_addr()
            .expect("the bound loopback address")
            .port();

        let mut to_root = TcpStream::connect(("127.0.0.1", broker))
            .expect("connect to rank 0 on the port it brokers on");
        bound_peer_link(&to_root);
        let mut hello = [0u8; 4];
        hello[..2].copy_from_slice(&(my_rank as u16).to_le_bytes());
        hello[2..].copy_from_slice(&my_port.to_le_bytes());
        to_root.write_all(&hello).expect("announce rank and port");
        to_root.flush().expect("flush the announcement");

        let mut table = vec![0u8; 2 * ranks];
        to_root
            .read_exact(&mut table)
            .expect("learn the port of every rank from rank 0");
        let ports: Vec<u16> = table
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();

        let links = join_slab_mesh(my_rank, ranks, &ports, &listener);
        let report = solve_slab_as_rank(n, ranks, my_rank, scene, fault, links);

        let mut bytes = [0u8; 16];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(report) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        to_root.write_all(&bytes).expect("report to rank 0");
        to_root.flush().expect("flush the report");
    }

    /// The child ranks rank 0 spawned, reaped on the way out even when an
    /// assertion unwinds, so a failing run leaves no process holding a port.
    #[cfg(feature = "std")]
    struct SlabRanks {
        kids: Vec<std::process::Child>,
    }

    #[cfg(feature = "std")]
    impl Drop for SlabRanks {
        fn drop(&mut self) {
            for kid in &mut self.kids {
                let _ = kid.kill();
                let _ = kid.wait();
            }
        }
    }

    #[cfg(feature = "std")]
    impl SlabRanks {
        /// Accept one connection while watching every child.
        ///
        /// All children are polled on each turn: one that exited early would
        /// otherwise leave this loop running to its deadline, and "timed out" is
        /// all the run would be able to say about it.
        fn accept(&mut self, listener: &std::net::TcpListener) -> std::net::TcpStream {
            use std::io::ErrorKind;
            use std::time::{Duration, Instant};

            listener
                .set_nonblocking(true)
                .expect("poll for a child rank instead of blocking on it");
            let deadline = Instant::now() + Duration::from_secs(60);
            let link = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                    Err(e) => panic!("accepting a child rank failed: {e}"),
                }
                for (slot, kid) in self.kids.iter_mut().enumerate() {
                    if let Some(status) = kid.try_wait().expect("poll a child rank") {
                        panic!(
                            "rank {} exited ({status}) before connecting: does \
                             `{XSLAB_WORKER}` still name a test?",
                            slot + 1,
                        );
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "a child rank did not connect within 60 s",
                );
                std::thread::sleep(Duration::from_millis(2));
            };
            link.set_nonblocking(false)
                .expect("switch the accepted stream back to blocking");
            bound_peer_link(&link);
            link
        }

        /// Wait for every rank and require a clean exit.
        fn join(&mut self) {
            for (slot, kid) in self.kids.iter_mut().enumerate() {
                let status = kid.wait().expect("wait for a child rank to exit");
                assert!(status.success(), "rank {} exited with {status}", slot + 1);
            }
        }
    }

    /// Solve `n³` across `ranks` processes — rank 0 here, the rest re-executed —
    /// and return every rank's comparison against the single-process solve.
    #[cfg(feature = "std")]
    fn slab_reports_across_processes(
        n: usize,
        ranks: usize,
        scene: SlabScene,
        fault: SlabCrossFault,
    ) -> Vec<[u32; 4]> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::process::{Command, Stdio};

        assert!(ranks >= 2, "a cross-process case needs at least two ranks");
        let broker = TcpListener::bind(("127.0.0.1", 0)).expect("bind the broker port");
        let broker_port = broker
            .local_addr()
            .expect("the bound loopback address")
            .port();
        // Bound before any child is spawned, for the reason the worker gives.
        let mine = TcpListener::bind(("127.0.0.1", 0)).expect("bind rank 0's data port");
        let my_port = mine.local_addr().expect("the bound address").port();
        let exe = std::env::current_exe().expect("path of this test binary");

        let mut kids = SlabRanks {
            kids: (1..ranks)
                .map(|rank| {
                    Command::new(&exe)
                        .args(["--exact", XSLAB_WORKER, "--test-threads=1"])
                        .env(XSLAB_RANK, rank.to_string())
                        .env(XSLAB_RANKS, ranks.to_string())
                        .env(XSLAB_BROKER, broker_port.to_string())
                        .env(XSLAB_SIZE, n.to_string())
                        .env(XSLAB_SCENE, scene.as_str())
                        .env(XSLAB_FAULT, fault.as_str())
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .unwrap_or_else(|e| {
                            panic!("re-execute this test binary as rank {rank}: {e}")
                        })
                })
                .collect(),
        };

        // Collect every child's port, then hand the whole table back. A child
        // cannot be told the table before every child has announced itself, which
        // is why this is two passes and not one.
        let mut brokered: Vec<Option<std::net::TcpStream>> = (0..ranks).map(|_| None).collect();
        let mut ports = vec![0u16; ranks];
        ports[0] = my_port;
        for _ in 1..ranks {
            let mut link = kids.accept(&broker);
            let mut hello = [0u8; 4];
            link.read_exact(&mut hello)
                .expect("a child rank's announcement");
            let rank = usize::from(u16::from_le_bytes([hello[0], hello[1]]));
            assert!(
                rank > 0 && rank < ranks && brokered[rank].is_none(),
                "rank {rank} announced itself twice or is not a child rank",
            );
            ports[rank] = u16::from_le_bytes([hello[2], hello[3]]);
            brokered[rank] = Some(link);
        }
        let table: Vec<u8> = ports.iter().flat_map(|p| p.to_le_bytes()).collect();
        for link in brokered.iter_mut().flatten() {
            link.write_all(&table).expect("hand a child the port table");
            link.flush().expect("flush the port table");
        }

        let links = join_slab_mesh(0, ranks, &ports, &mine);
        let mut reports = vec![[0u32; 4]; ranks];
        reports[0] = solve_slab_as_rank(n, ranks, 0, scene, fault, links);

        // Every rank has finished its own half of every delivery by the time rank
        // 0 is through its schedule, so the children are on their way out; their
        // reports are already in rank 0's receive buffers and survive the close.
        kids.join();
        for (rank, link) in brokered.iter_mut().enumerate() {
            let Some(link) = link else { continue };
            let mut bytes = [0u8; 16];
            link.read_exact(&mut bytes)
                .unwrap_or_else(|e| panic!("rank {rank}'s report: {e}"));
            for (slot, chunk) in reports[rank].iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = u32::from_le_bytes(chunk.try_into().expect("4 bytes of a report"));
            }
        }
        reports
    }

    /// Every rank of a `ranks`-process run agrees with the single-process solve
    /// over the layers it owns, and between them they cover the whole domain.
    #[cfg(feature = "std")]
    fn assert_slab_processes_agree(n: usize, ranks: usize, scene: SlabScene) {
        let reports = slab_reports_across_processes(n, ranks, scene, SlabCrossFault::None);
        for (rank, report) in reports.iter().enumerate() {
            assert_eq!(
                *report,
                [0, 0, u32::MAX, report[3]],
                "{n}³ over {ranks} processes ({scene:?}): rank {rank} found {} pressure \
                 cells and {} face values differing from the single-process solve, first at \
                 layer {}",
                report[0],
                report[1],
                report[2],
            );
        }
        let compared: u32 = reports.iter().map(|report| report[3]).sum();
        assert_eq!(
            compared as usize,
            n * n * n,
            "{n}³ over {ranks} processes ({scene:?}): the ranks compared {compared} cells \
             between them instead of the whole domain, so the agreement above is partly \
             vacuous",
        );
    }

    /// Three processes, each holding one band and one halo layer, reproduce the
    /// single-process solve bit for bit — over a rank count that divides the
    /// depth and one that does not, and over the scene whose face conditions
    /// cross a slab boundary.
    ///
    /// What this settles that the in-process oracles cannot: there every rank's
    /// band is a `Vec` in one address space, so a driver that read a neighbour's
    /// storage would also pass. Here a neighbour's band is in another process,
    /// every halo layer is sixteen bytes per cell on a loopback stream, and
    /// `SlabSocketTransport::slab_mut` refuses to hand out any band but its own.
    ///
    /// Exactness, not a tolerance: `Fix128` addition is a group operation mod
    /// 2¹²⁸, so a distribution that is right at all is right to the bit.
    #[cfg(feature = "std")]
    #[test]
    fn three_processes_reproduce_the_monolithic_slab_local_solve() {
        for &(n, scene) in &[
            (9usize, SlabScene::ClosedBox),
            (9, SlabScene::ZOutflow),
            (8, SlabScene::ZOutflow),
        ] {
            if scene == SlabScene::ZOutflow {
                assert!(
                    z_outflow_crossings(n, 3) > 0,
                    "{n}³ over 3 ranks has no Z-face layer crossing a slab boundary, so this \
                     case does not exercise the face-condition pipeline",
                );
            }
            assert_slab_processes_agree(n, 3, scene);
        }
    }

    /// Four processes over three layers, so one rank owns nothing — and the rank
    /// that owns nothing is rank 0, the one brokering the run.
    ///
    /// An empty rank takes part in no delivery, which means it must still walk
    /// the whole schedule to keep its position in it, and the chain the face
    /// conditions travel along has to skip it rather than break at it.
    #[cfg(feature = "std")]
    #[test]
    fn four_processes_one_of_which_owns_nothing_reproduce_the_monolithic_slab_local_solve() {
        let bounds: Vec<(usize, usize)> = (0..4).map(|r| slab_bounds(3, 4, r)).collect();
        assert!(
            bounds.iter().any(|&(k0, k1)| k0 == k1),
            "this case was chosen because a rank owns nothing and none does: {bounds:?}",
        );
        assert!(
            z_outflow_crossings(3, 4) > 0,
            "3³ over 4 ranks has no Z-face layer crossing a slab boundary",
        );
        assert_slab_processes_agree(3, 4, SlabScene::ZOutflow);
    }

    /// Teeth for the two tests above, aimed at the pressure halo: drop every
    /// delivery — on every rank, so no stream is left with an unread layer — and
    /// the ranks must stop agreeing with the single-process solve.
    ///
    /// Without this, agreement could be coming from every rank seeding the same
    /// field and the streams contributing nothing.
    #[cfg(feature = "std")]
    #[test]
    fn a_cross_process_slab_run_without_the_pressure_halo_does_not_reproduce_the_solve() {
        let reports = slab_reports_across_processes(
            8,
            3,
            SlabScene::ZOutflow,
            SlabCrossFault::NoPressureHalo,
        );
        assert!(
            reports.iter().any(|r| r[0] > 0),
            "three processes that exchanged no halo at all still reproduced the \
             single-process pressure field, so the cross-process oracle is not measuring the \
             exchange: {reports:?}",
        );
    }

    /// Teeth aimed at the other crossing: receive the Z-face plane the pipeline
    /// carries and impose the conditions as though it had been zero, and the
    /// ranks must stop agreeing.
    ///
    /// This is the one thing two ranks sharing an address space cannot show: in
    /// `distributed_face_enforcement_matches_the_monolithic_one` the plane is a
    /// copy within one process, so the test cannot distinguish a pipeline that
    /// carries the layer from one that happens to find it already correct.
    #[cfg(feature = "std")]
    #[test]
    fn a_cross_process_slab_run_without_the_z_face_layer_does_not_reproduce_the_solve() {
        assert!(
            z_outflow_crossings(8, 3) > 0,
            "8³ over 3 ranks has no Z-face layer crossing a slab boundary, so blanking it \
             could not change anything",
        );
        let reports =
            slab_reports_across_processes(8, 3, SlabScene::ZOutflow, SlabCrossFault::BlankFaceHalo);
        assert!(
            reports.iter().any(|r| r[0] > 0 || r[1] > 0),
            "three processes that imposed their face conditions without the layer below \
             still reproduced the single-process solve, so the face-condition pipeline is \
             not measured: {reports:?}",
        );
    }

    /// A rank-local driver asks its transport for its own band and no other, and
    /// a transport holding one band says so when a driver asks for a neighbour's.
    ///
    /// The guard that makes the cross-process oracles mean what they say: without
    /// it a driver walking every rank would read whatever `slab_mut` returned and
    /// the run would look distributed.
    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "the driver is not rank-local")]
    fn a_slab_transport_refuses_to_hand_out_another_ranks_band() {
        use std::io::Cursor;

        let n = 4usize;
        let plane = cell_plane(n, n);
        let b = slab_bounds(n, 2, 0);
        let band = SlabStorage::for_slab(plane, n, b, 1);
        let links: Vec<Option<Cursor<Vec<u8>>>> = vec![None, Some(Cursor::new(Vec::new()))];
        let mut transport = SlabSocketTransport::new(0, plane, band, links);
        let _ = transport.slab_mut(1);
    }
}
