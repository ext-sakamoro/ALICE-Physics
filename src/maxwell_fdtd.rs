//! Maxwell field solver on a Yee (FDTD) lattice, in normalised units.
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
//! cavity, which is what gives the closed-form oracle its modes.
//!
//! [`Absorber::GradedPml`] lines the walls with a Bérenger split-field PML,
//! still PEC-backed, so an outgoing wave is attenuated on the way in and again
//! on the way back out. ⚠️ The split update runs **only** on samples whose
//! conductivity is non-zero on one of their two loss axes; everywhere else the
//! loss-free update runs unchanged, which is what keeps a lattice without an
//! absorber bit-identical. `Fix128` multiplication truncates, so
//! `S·a + S·b` and `S·(a + b)` differ by 1 ULP in 47.0% of random pairs
//! (measured 2026-09-30) — splitting the interior too would have broken every
//! exact cavity oracle.
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
//! # Sources, Gauss's law and charge continuity
//!
//! A lattice starts source free and stays that way — and stays exactly as
//! cheap — until [`YeeGrid::set_current`] or [`YeeGrid::set_charge`] is called,
//! which is when the `J` and `ρ` storage is allocated. With sources present,
//! Ampère's law carries the conduction term:
//!
//! ```text
//! E^{n+1} = E^{n} + S · (∇×H)^{n+½} − S · J
//! ρ^{n+1} = ρ^{n} − S · (∇·J)
//! ```
//!
//! `J` sits on edges alongside `E`; `ρ` sits on the **interior nodes**, which
//! is where the node divergence `∇·E` has all six of the edges it needs. The
//! two Gauss laws are then:
//!
//! ```text
//! ∇·E = ρ     ([`YeeGrid::div_e`], [`YeeGrid::gauss_residual`])
//! ∇·B = 0     ([`YeeGrid::div_b`])
//! ```
//!
//! Neither is imposed. Both are **consequences** of the update, and that is
//! what makes them worth testing: `∇·(∇×H)` telescopes to zero on this lattice,
//! so `∇·E − ρ` changes by exactly `S·(∇·J) − S·(∇·J) = 0` per step. The
//! cancellation is what a test can pin, and it only happens if the same
//! discrete divergence is used on both sides — the charge-conserving-deposition
//! condition that particle-in-cell codes have to satisfy for the same reason.
//!
//! ⚠️ The cancellation is **bit-exact only when `S·J` is exactly representable**
//! (for instance dyadic `S` with integer `J`). Otherwise the `E` update
//! truncates six products and the `ρ` update truncates one product of their
//! sum, and those disagree by 1–3 ULP per step — the same truncation that keeps
//! `∇·B` off the zero bit pattern.
//!
//! # Scope
//!
//! No material tensors and no bidirectional coupling to the thermal or
//! piezoelectric solvers.

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
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

/// What absorbs at the walls, chosen once at construction.
///
/// The variant decides whether a lattice allocates split-field storage at all;
/// [`Absorber::None`] allocates nothing and leaves [`YeeGrid::step`] on exactly
/// the arithmetic it performs without this type existing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Absorber {
    /// Nothing: all six walls are plain PEC and the lattice is a closed cavity.
    None,
    /// Bérenger split-field PML, `depth[axis]` cells thick on both walls of
    /// that axis.
    ///
    /// The conductivity is graded as `σ(u) = σ_max·(u/depth)³` with `u` the
    /// distance into the layer, which is the usual cubic profile; the grading
    /// is what keeps the layer from reflecting off its own front face.
    /// [`theoretical_pml_reflection`] gives the continuum round-trip
    /// reflection one axis's `(depth, sigma_max)` pair is aiming at.
    ///
    /// ⚠️ **Three depths rather than one, and this is a testability
    /// requirement — do not collapse it back to a scalar.** Each split half
    /// field is damped by the conductivity of *one named axis*. With a single
    /// depth all three axes carry the same profile, so swapping two of those
    /// names changes the arithmetic on no sample at all and the wiring becomes
    /// **unverifiable in principle**. A destruction test measured this on
    /// 2026-09-30: reading `Exy`'s coefficient from the `z` profile instead of
    /// the `y` profile left every oracle in `tests/analytic_maxwell_fdtd.rs`
    /// green, and it still did after the absorption measurement was moved to an
    /// anisotropic lattice — a field damped by the wrong axis is still a damped
    /// field, so every scalar measurement passes. What catches it is
    /// `relabelling_the_axes_relabels_the_solution_exactly`, and that oracle
    /// needs a configuration the cyclic permutation does **not** map to itself.
    /// Three depths are what make such a configuration expressible.
    ///
    /// Being able to line only two walls (a quasi-2-D run) is a convenience
    /// this happens to buy; it is not the reason.
    GradedPml {
        /// Layer thickness in cells for the x, y and z axes. All zero is the
        /// same as [`Absorber::None`]. ⚠️ Keeping these independent is what
        /// makes the split wiring testable — see the variant's own note.
        depth: [usize; 3],
        /// Conductivity at the PEC-backed outer face, in normalised units.
        sigma_max: Fix128,
    },
    /// A uniform matched conductivity `σ = σ* ` over the whole lattice.
    ///
    /// Not an absorbing boundary — a lossy medium. It shares the split-field
    /// code path with [`Absorber::GradedPml`], which is what lets the lossy
    /// update be pinned against a closed form: with one `σ` everywhere the
    /// projected amplitude of a cavity mode obeys the three-term recurrence
    /// `q^{n+1} = (2c − b²λ)·q^n − c²·q^{n−1}` exactly, where `c` and `b` are
    /// the coefficients from [`loss_coefficients`].
    Uniform {
        /// Conductivity, in normalised units.
        sigma: Fix128,
    },
}

/// The pair `(ca, cb)` that a lossy leap-frog update uses for one half field.
///
/// `f^{n+1} = ca·f^n + cb·d`, from the semi-implicit (exponentially stable)
/// discretisation of `∂f/∂t + σ·f = d`:
///
/// ```text
/// a  = σ·S/2
/// ca = (1 − a)/(1 + a)
/// cb = S/(1 + a)
/// ```
///
/// `σ = 0` returns `(1, S)` exactly, which is the loss-free update.
///
/// ⚠️ Both coefficients are `O(1)`, so Q64.64 keeps 54–64 significant bits of
/// them across `σ ∈ [10⁻⁴, 128]` — measured against exact rational references
/// with a gap of **0 ULP** (2026-09-30). The fixed-point range problem that
/// forces normalised units on this module is a problem for `ε₀·μ₀ ≈ 1.1e-17`,
/// which is 205 ULP; it does not reach coefficients near one.
#[must_use]
pub fn loss_coefficients(sigma: Fix128, courant: Fix128) -> (Fix128, Fix128) {
    if sigma.is_zero() {
        return (Fix128::ONE, courant);
    }
    let a = (sigma * courant).half();
    let denom = Fix128::ONE + a;
    ((Fix128::ONE - a) / denom, courant / denom)
}

/// The cubic conductivity profile, `σ(u) = σ_max·(u/depth)³`.
///
/// `into2` and `depth2` are **doubled** distances so that a half-integer
/// coordinate stays an exact rational: `into2` is twice the distance from the
/// inner edge of the layer and `depth2` is twice its thickness. `into2 = 0` is
/// the inner edge (no loss) and `into2 = depth2` is the PEC-backed outer face.
///
/// ⚠️ One home on purpose. Both the lattice construction and
/// [`theoretical_pml_reflection`] need this profile, and a formula written twice
/// is a formula that gets fixed once.
fn cubic_graded_sigma(into2: i64, depth2: i64, sigma_max: Fix128) -> Fix128 {
    if depth2 <= 0 || into2 <= 0 {
        return Fix128::ZERO;
    }
    let t = Fix128::from_ratio(into2.min(depth2), depth2);
    sigma_max * t * t * t
}

/// Continuum round-trip reflection of a cubic-graded PEC-backed PML layer.
///
/// `R = exp(−2·∫₀^d σ(u) du)` with the integral taken by the midpoint rule over
/// the `depth` cells, which is where the conductivity samples actually sit. The
/// factor two is the round trip: the wave crosses the layer, reflects off the
/// PEC behind it and crosses back.
///
/// ⚠️ This is the **continuum** figure. The discrete lattice reflects more than
/// this, because a conductivity that steps from cell to cell is itself an
/// impedance discontinuity; the value here is the floor a measurement is
/// compared against, not a number the solver is expected to reach.
#[must_use]
pub fn theoretical_pml_reflection(depth: usize, sigma_max: Fix128) -> Fix128 {
    if depth == 0 {
        return Fix128::ONE;
    }
    let d2 = 2 * depth as i64;
    let mut integral = Fix128::ZERO;
    for cell in 0..depth as i64 {
        // Cell centres measured from the inner edge, doubled: `½, 1½, …`.
        integral = integral + cubic_graded_sigma(2 * cell + 1, d2, sigma_max);
    }
    (Fix128::from_int(-2) * integral).exp()
}

/// Split half fields and the per-axis loss coefficients that drive them.
///
/// ⚠️ The twelve arrays are **full lattice size**, not PML size: a lattice with
/// a four-cell layer still allocates twelve arrays as large as its six primary
/// ones. The layer is thin but the storage is not proportional to it. Only the
/// coefficient vectors are `O(n)`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Splits {
    exy: Vec<Fix128>,
    exz: Vec<Fix128>,
    eyz: Vec<Fix128>,
    eyx: Vec<Fix128>,
    ezx: Vec<Fix128>,
    ezy: Vec<Fix128>,
    hxy: Vec<Fix128>,
    hxz: Vec<Fix128>,
    hyz: Vec<Fix128>,
    hyx: Vec<Fix128>,
    hzx: Vec<Fix128>,
    hzy: Vec<Fix128>,
    /// `[axis][coord]` at integer coordinates `0..=n`, where `E` samples sit.
    ca_e: [Vec<Fix128>; 3],
    cb_e: [Vec<Fix128>; 3],
    /// `[axis][coord]` at half-integer coordinates `½..n−½`, where `H` sits.
    ca_h: [Vec<Fix128>; 3],
    cb_h: [Vec<Fix128>; 3],
    lossy_e: [Vec<bool>; 3],
    lossy_h: [Vec<bool>; 3],
}

impl Splits {
    /// The two half fields of `component`, in the order the update writes them.
    fn halves_mut(&mut self, component: Component) -> (&mut Vec<Fix128>, &mut Vec<Fix128>) {
        match component {
            Component::Ex => (&mut self.exy, &mut self.exz),
            Component::Ey => (&mut self.eyz, &mut self.eyx),
            Component::Ez => (&mut self.ezx, &mut self.ezy),
            Component::Hx => (&mut self.hxy, &mut self.hxz),
            Component::Hy => (&mut self.hyz, &mut self.hyx),
            Component::Hz => (&mut self.hzx, &mut self.hzy),
        }
    }
}

/// Current density on edges and charge density on interior nodes.
///
/// Allocated lazily: a lattice that is never given a source never pays for
/// this, and [`YeeGrid::step`] skips the source pass entirely, so a source-free
/// run performs exactly the arithmetic it performed before sources existed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Sources {
    jx: Vec<Fix128>,
    jy: Vec<Fix128>,
    jz: Vec<Fix128>,
    /// `(nx−1) × (ny−1) × (nz−1)`, indexed by node `(i, j, k)` minus one.
    rho: Vec<Fix128>,
}

/// A Yee lattice of electromagnetic field, marched by [`YeeGrid::step`].
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
    sources: Option<Box<Sources>>,
    absorber: Option<Box<Splits>>,
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
            sources: None,
            absorber: None,
        }
    }

    /// Allocate a lattice with an [`Absorber`] at the walls.
    ///
    /// [`Absorber::None`] is exactly [`YeeGrid::new`]: no split-field storage
    /// is allocated and [`YeeGrid::step`] takes the same branch it takes for a
    /// lattice that has never heard of a PML, so a source-free run is
    /// bit-identical to one from before absorbers existed.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero, or if a PML layer is thicker than half
    /// the lattice on some axis — the two layers would overlap and the
    /// conductivity profile would stop being monotone.
    #[must_use]
    pub fn new_with_absorber(
        nx: usize,
        ny: usize,
        nz: usize,
        courant: Fix128,
        absorber: Absorber,
    ) -> Self {
        let mut grid = Self::new(nx, ny, nz, courant);
        let dims = [nx, ny, nz];
        let sigma_at = |axis: usize, coord2: i64| -> Fix128 {
            let n2 = 2 * dims[axis] as i64;
            match absorber {
                Absorber::None => Fix128::ZERO,
                Absorber::Uniform { sigma } => sigma,
                Absorber::GradedPml { depth, sigma_max } => {
                    let d2 = 2 * depth[axis] as i64;
                    let into2 = (d2 - coord2).max(coord2 - (n2 - d2));
                    cubic_graded_sigma(into2, d2, sigma_max)
                }
            }
        };
        if let Absorber::GradedPml { depth, .. } = absorber {
            assert!(
                (0..3).all(|a| 2 * depth[a] <= dims[a]),
                "a PML layer must fit twice into its own axis"
            );
        }
        let inert = match absorber {
            Absorber::None => true,
            Absorber::GradedPml { depth, .. } => depth == [0, 0, 0],
            Absorber::Uniform { .. } => false,
        };
        if inert {
            return grid;
        }

        let mut ca_e = [Vec::new(), Vec::new(), Vec::new()];
        let mut cb_e = [Vec::new(), Vec::new(), Vec::new()];
        let mut ca_h = [Vec::new(), Vec::new(), Vec::new()];
        let mut cb_h = [Vec::new(), Vec::new(), Vec::new()];
        let mut lossy_e = [Vec::new(), Vec::new(), Vec::new()];
        let mut lossy_h = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..3 {
            let n = dims[axis];
            for coord in 0..=n {
                // `E` sits at integer coordinates along its two loss axes.
                let sigma = sigma_at(axis, 2 * coord as i64);
                let (ca, cb) = loss_coefficients(sigma, courant);
                ca_e[axis].push(ca);
                cb_e[axis].push(cb);
                lossy_e[axis].push(!sigma.is_zero());
            }
            for coord in 0..n {
                // `H` sits at half-integer coordinates along its two loss axes.
                let sigma = sigma_at(axis, 2 * coord as i64 + 1);
                let (ca, cb) = loss_coefficients(sigma, courant);
                ca_h[axis].push(ca);
                cb_h[axis].push(cb);
                lossy_h[axis].push(!sigma.is_zero());
            }
        }
        grid.absorber = Some(Box::new(Splits {
            exy: vec![Fix128::ZERO; grid.ex.len()],
            exz: vec![Fix128::ZERO; grid.ex.len()],
            eyz: vec![Fix128::ZERO; grid.ey.len()],
            eyx: vec![Fix128::ZERO; grid.ey.len()],
            ezx: vec![Fix128::ZERO; grid.ez.len()],
            ezy: vec![Fix128::ZERO; grid.ez.len()],
            hxy: vec![Fix128::ZERO; grid.hx.len()],
            hxz: vec![Fix128::ZERO; grid.hx.len()],
            hyz: vec![Fix128::ZERO; grid.hy.len()],
            hyx: vec![Fix128::ZERO; grid.hy.len()],
            hzx: vec![Fix128::ZERO; grid.hz.len()],
            hzy: vec![Fix128::ZERO; grid.hz.len()],
            ca_e,
            cb_e,
            ca_h,
            cb_h,
            lossy_e,
            lossy_h,
        }));
        grid
    }

    /// Whether this sample is marched by the lossy split-field update.
    ///
    /// A sample is split when the conductivity is non-zero on at least one of
    /// its **two** loss axes, which for `Ex` are `y` and `z`. That is not a
    /// simplification: in a PML slab normal to `x`, the normal component `Ex`
    /// is the one the layer does not attenuate, so it stays on the lossless
    /// update even though it sits geometrically inside the layer.
    ///
    /// Tests use this to keep the exact invariants — `∇·B = 0` in particular —
    /// on the part of the lattice where they are supposed to hold. ⚠️ A split
    /// half field is book-keeping, not a physical field, so `∇·B` does not
    /// telescope inside the layer and is **not expected to**.
    ///
    /// # Panics
    ///
    /// Panics if the index is outside [`YeeGrid::component_dims`].
    #[must_use]
    pub fn is_absorbing(&self, component: Component, i: usize, j: usize, k: usize) -> bool {
        let _ = self.offset(component, i, j, k);
        let Some(sp) = self.absorber.as_ref() else {
            return false;
        };
        let (axis_a, ca, axis_b, cb) = match component {
            Component::Ex => (1, j, 2, k),
            Component::Ey => (2, k, 0, i),
            Component::Ez => (0, i, 1, j),
            Component::Hx => (1, j, 2, k),
            Component::Hy => (2, k, 0, i),
            Component::Hz => (0, i, 1, j),
        };
        match component {
            Component::Ex | Component::Ey | Component::Ez => {
                sp.lossy_e[axis_a][ca] || sp.lossy_e[axis_b][cb]
            }
            Component::Hx | Component::Hy | Component::Hz => {
                sp.lossy_h[axis_a][ca] || sp.lossy_h[axis_b][cb]
            }
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
    /// # Split samples
    ///
    /// Only the **sum** of a split sample's two half fields is physical, so any
    /// pair that adds up to `value` is a valid state and a convention has to be
    /// picked. This one puts the whole value in the half the update writes
    /// first (`Exy` for `Ex`, and so on) and zeros the other, because that is
    /// the choice under which `set` then [`YeeGrid::get`] returns the value
    /// unchanged, `set(ZERO)` genuinely empties the sample, and calling `set`
    /// twice is idempotent. Splitting it evenly would satisfy none of the
    /// three, since the halves decay at different rates from the next step on.
    pub fn set(&mut self, component: Component, i: usize, j: usize, k: usize, value: Fix128) {
        let at = self.offset(component, i, j, k);
        self.slot_mut(component)[at] = value;
        if let Some(sp) = self.absorber.as_deref_mut() {
            let (first, second) = sp.halves_mut(component);
            first[at] = value;
            second[at] = Fix128::ZERO;
        }
    }

    // -- sources: J on edges, ρ on interior nodes ---------------------------

    /// Number of **interior** nodes along each axis, `(nx−1, ny−1, nz−1)`.
    ///
    /// A node carries `ρ` and the node divergence `∇·E`, both of which need the
    /// six edges around the node, so only nodes strictly inside the lattice
    /// have them. Node `(i, j, k)` is interior when `1 ≤ i < nx` and likewise
    /// for `j` and `k`; this returns how many such `i`, `j`, `k` there are, so
    /// a lattice thinner than two cells on any axis has none.
    #[must_use]
    pub const fn interior_node_dims(&self) -> (usize, usize, usize) {
        (
            self.nx.saturating_sub(1),
            self.ny.saturating_sub(1),
            self.nz.saturating_sub(1),
        )
    }

    #[inline]
    fn node_offset(&self, i: usize, j: usize, k: usize) -> usize {
        let (ni, nj, nk) = self.interior_node_dims();
        assert!(
            i >= 1 && j >= 1 && k >= 1 && i <= ni && j <= nj && k <= nk,
            "node index is not an interior node"
        );
        ((i - 1) * nj + (j - 1)) * nk + (k - 1)
    }

    /// Whether `component` is written by the `E` update at `(i, j, k)`.
    ///
    /// The tangential `E` on a PEC wall is held at zero, so the samples outside
    /// these ranges are never updated. A current placed on one of them would be
    /// discarded silently, which is why [`YeeGrid::set_current`] rejects it.
    #[inline]
    const fn is_updated_edge(&self, component: Component, i: usize, j: usize, k: usize) -> bool {
        match component {
            Component::Ex => i < self.nx && j >= 1 && j < self.ny && k >= 1 && k < self.nz,
            Component::Ey => i >= 1 && i < self.nx && j < self.ny && k >= 1 && k < self.nz,
            Component::Ez => i >= 1 && i < self.nx && j >= 1 && j < self.ny && k < self.nz,
            Component::Hx | Component::Hy | Component::Hz => false,
        }
    }

    fn sources_mut(&mut self) -> &mut Sources {
        let (ni, nj, nk) = self.interior_node_dims();
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        self.sources.get_or_insert_with(|| {
            Box::new(Sources {
                jx: vec![Fix128::ZERO; nx * (ny + 1) * (nz + 1)],
                jy: vec![Fix128::ZERO; (nx + 1) * ny * (nz + 1)],
                jz: vec![Fix128::ZERO; (nx + 1) * (ny + 1) * nz],
                rho: vec![Fix128::ZERO; ni * nj * nk],
            })
        })
    }

    /// Read the current density on an edge; zero if the lattice has no sources.
    ///
    /// # Panics
    ///
    /// Panics if `component` is a magnetic one, or if the index is outside
    /// [`YeeGrid::component_dims`].
    #[must_use]
    pub fn current(&self, component: Component, i: usize, j: usize, k: usize) -> Fix128 {
        assert!(
            matches!(component, Component::Ex | Component::Ey | Component::Ez),
            "current density lives on electric edges, not {component:?}"
        );
        let at = self.offset(component, i, j, k);
        self.sources
            .as_ref()
            .map_or(Fix128::ZERO, |src| match component {
                Component::Ex => src.jx[at],
                Component::Ey => src.jy[at],
                _ => src.jz[at],
            })
    }

    /// Set the current density on an edge, allocating source storage on demand.
    ///
    /// # Panics
    ///
    /// Panics if `component` is a magnetic one, if the index is outside
    /// [`YeeGrid::component_dims`], or if the edge is one the `E` update never
    /// writes (the tangential edges on a PEC wall). The last case would
    /// otherwise be a current that drives nothing and that `∇·J` never sees,
    /// which breaks charge continuity without any visible symptom.
    pub fn set_current(
        &mut self,
        component: Component,
        i: usize,
        j: usize,
        k: usize,
        value: Fix128,
    ) {
        assert!(
            matches!(component, Component::Ex | Component::Ey | Component::Ez),
            "current density lives on electric edges, not {component:?}"
        );
        let at = self.offset(component, i, j, k);
        assert!(
            self.is_updated_edge(component, i, j, k),
            "{component:?}[{i}][{j}][{k}] is tangential to a PEC wall, so a current there would be discarded"
        );
        assert!(
            !self.is_absorbing(component, i, j, k),
            "{component:?}[{i}][{j}][{k}] is a split sample: its value is rebuilt from the two half fields every step, so a current there would be discarded"
        );
        let src = self.sources_mut();
        match component {
            Component::Ex => src.jx[at] = value,
            Component::Ey => src.jy[at] = value,
            Component::Ez => src.jz[at] = value,
            Component::Hx | Component::Hy | Component::Hz => unreachable!(),
        }
    }

    /// Read the charge density on an interior node; zero if there are no sources.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j, k)` is not an interior node — see
    /// [`YeeGrid::interior_node_dims`].
    #[must_use]
    pub fn charge(&self, i: usize, j: usize, k: usize) -> Fix128 {
        let at = self.node_offset(i, j, k);
        self.sources
            .as_ref()
            .map_or(Fix128::ZERO, |src| src.rho[at])
    }

    /// Set the charge density on an interior node, allocating on demand.
    ///
    /// ⚠️ This does **not** adjust `E` to match. Gauss's law is a consequence of
    /// the update rather than a constraint it imposes, so a caller that wants
    /// `∇·E = ρ` to hold has to set up an initial condition in which it does;
    /// [`YeeGrid::gauss_residual`] reports how far off it is.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j, k)` is not an interior node.
    pub fn set_charge(&mut self, i: usize, j: usize, k: usize, value: Fix128) {
        let at = self.node_offset(i, j, k);
        self.sources_mut().rho[at] = value;
    }

    /// Discrete `∇·E` at an interior node, summed over the six edges that meet there.
    ///
    /// In normalised units `D = E`, so this is the left-hand side of Gauss's
    /// law. Every edge it reads is one the `E` update writes, which is what
    /// makes the residual in [`YeeGrid::gauss_residual`] a statement about the
    /// update rather than about the boundary.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j, k)` is not an interior node.
    #[must_use]
    pub fn div_e(&self, i: usize, j: usize, k: usize) -> Fix128 {
        let _ = self.node_offset(i, j, k);
        (self.get(Component::Ex, i, j, k) - self.get(Component::Ex, i - 1, j, k))
            + (self.get(Component::Ey, i, j, k) - self.get(Component::Ey, i, j - 1, k))
            + (self.get(Component::Ez, i, j, k) - self.get(Component::Ez, i, j, k - 1))
    }

    /// Discrete `∇·J` at an interior node, on the same stencil as [`YeeGrid::div_e`].
    ///
    /// ⚠️ Using the *same* stencil is the whole point: `ρ` is marched by
    /// `−S·∇·J` with this operator, so the charge it deposits is exactly the
    /// charge that the `−S·J` term in Ampère's law removes from `∇·E`. A
    /// different stencil on either side would leave a residual that grows with
    /// the current rather than with the step count.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j, k)` is not an interior node.
    #[must_use]
    pub fn div_j(&self, i: usize, j: usize, k: usize) -> Fix128 {
        let _ = self.node_offset(i, j, k);
        let Some(src) = self.sources.as_ref() else {
            return Fix128::ZERO;
        };
        let jx = |i: usize, j: usize, k: usize| src.jx[self.offset(Component::Ex, i, j, k)];
        let jy = |i: usize, j: usize, k: usize| src.jy[self.offset(Component::Ey, i, j, k)];
        let jz = |i: usize, j: usize, k: usize| src.jz[self.offset(Component::Ez, i, j, k)];
        (jx(i, j, k) - jx(i - 1, j, k))
            + (jy(i, j, k) - jy(i, j - 1, k))
            + (jz(i, j, k) - jz(i, j, k - 1))
    }

    /// `∇·E − ρ` at an interior node: how far Gauss's law is from holding.
    ///
    /// # Panics
    ///
    /// Panics if `(i, j, k)` is not an interior node.
    #[must_use]
    pub fn gauss_residual(&self, i: usize, j: usize, k: usize) -> Fix128 {
        self.div_e(i, j, k) - self.charge(i, j, k)
    }

    /// Largest `|∇·E − ρ|` over every interior node.
    #[must_use]
    pub fn max_abs_gauss_residual(&self) -> Fix128 {
        let (ni, nj, nk) = self.interior_node_dims();
        let mut worst = Fix128::ZERO;
        for i in 1..=ni {
            for j in 1..=nj {
                for k in 1..=nk {
                    let r = self.gauss_residual(i, j, k).abs();
                    if r > worst {
                        worst = r;
                    }
                }
            }
        }
        worst
    }

    /// Sum of `ρ` over every interior node.
    ///
    /// Addition in [`Fix128`] does not round, so this sum is exact whatever the
    /// summands are; the only inexactness a conservation test can see is in how
    /// `ρ` itself was marched.
    #[must_use]
    pub fn total_charge(&self) -> Fix128 {
        self.sources.as_ref().map_or(Fix128::ZERO, |src| {
            src.rho.iter().fold(Fix128::ZERO, |acc, &r| acc + r)
        })
    }

    /// Advance the field by one time step (`H` half step, then `E` full step).
    ///
    /// The lattice is swept in index order, so the result is bit-identical on
    /// every target.
    ///
    /// With sources present a third pass applies `−S·J` to `E` and `−S·∇·J` to
    /// `ρ`. It is a separate pass rather than a term folded into the `E` loops
    /// because [`Fix128`] addition does not round: `(E + S·curl) − S·J` and
    /// `E + (S·curl − S·J)` are the same bit pattern, so splitting it costs
    /// nothing and leaves the source-free loops literally untouched.
    pub fn step(&mut self) {
        self.step_fields();
        if self.sources.is_some() {
            self.apply_sources();
        }
    }

    /// The source term: `E −= S·J` on updated edges, `ρ −= S·(∇·J)` on nodes.
    ///
    /// ⚠️ The edge ranges are the same ones the `E` update writes, so a PEC
    /// wall stays at zero. [`YeeGrid::set_current`] rejects the other edges, so
    /// every `J` sample that exists here is both applied to an `E` sample and
    /// read by some `∇·J` — which is what makes the two cancel.
    fn apply_sources(&mut self) {
        let s = self.courant;
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // ρ first: it reads J only, and the E pass below does not touch J, so
        // the order is immaterial — but reading ρ's input before E changes
        // keeps the two halves independent to a reader as well.
        let (ni, nj, nk) = self.interior_node_dims();
        for i in 1..=ni {
            for j in 1..=nj {
                for k in 1..=nk {
                    let d = self.div_j(i, j, k);
                    let at = self.node_offset(i, j, k);
                    let src = self.sources_mut();
                    src.rho[at] = src.rho[at] - s * d;
                }
            }
        }

        for i in 0..nx {
            for j in 1..ny {
                for k in 1..nz {
                    let at = self.offset(Component::Ex, i, j, k);
                    let j_here = self.sources.as_ref().map_or(Fix128::ZERO, |v| v.jx[at]);
                    self.ex[at] = self.ex[at] - s * j_here;
                }
            }
        }
        for i in 1..nx {
            for j in 0..ny {
                for k in 1..nz {
                    let at = self.offset(Component::Ey, i, j, k);
                    let j_here = self.sources.as_ref().map_or(Fix128::ZERO, |v| v.jy[at]);
                    self.ey[at] = self.ey[at] - s * j_here;
                }
            }
        }
        for i in 1..nx {
            for j in 1..ny {
                for k in 0..nz {
                    let at = self.offset(Component::Ez, i, j, k);
                    let j_here = self.sources.as_ref().map_or(Fix128::ZERO, |v| v.jz[at]);
                    self.ez[at] = self.ez[at] - s * j_here;
                }
            }
        }
    }

    /// The curl pair of updates, with the split-field branch folded in.
    ///
    /// ⚠️ Each curl appears **once**. The lossless arm is the expression this
    /// module has always used, character for character, which is what keeps a
    /// lattice without an absorber bit-identical: [`Fix128`] multiplication
    /// truncates toward −∞, so `f − S·(a − b)` and `f + S·((−a) + b)` are *not*
    /// the same bit pattern, and neither is `S·a + S·b` the same as `S·(a+b)`
    /// (measured: 47.0% of random pairs differ, always by 1 ULP, 2026-09-30).
    /// Rewriting the lossless arm to look more like the lossy one would quietly
    /// break the exact cavity oracles.
    fn step_fields(&mut self) {
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
            sources: _,
            absorber,
        } = self;
        let (nx, ny, nz) = (*nx, *ny, *nz);
        let s = *courant;
        let mut ab = absorber.as_deref_mut();

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
        // (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z; Hxy carries -∂Ez/∂y, Hxz carries +∂Ey/∂z.
        for i in 0..=nx {
            for j in 0..ny {
                for k in 0..nz {
                    let a = ez[i * ez_i + (j + 1) * ez_j + k] - ez[i * ez_i + j * ez_j + k];
                    let b = ey[i * ey_i + j * ey_j + (k + 1)] - ey[i * ey_i + j * ey_j + k];
                    let curl = a - b;
                    let at = i * hx_i + j * hx_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_h[1][j] || sp.lossy_h[2][k] => {
                            sp.hxy[at] = sp.ca_h[1][j] * sp.hxy[at] - sp.cb_h[1][j] * a;
                            sp.hxz[at] = sp.ca_h[2][k] * sp.hxz[at] + sp.cb_h[2][k] * b;
                            hx[at] = sp.hxy[at] + sp.hxz[at];
                        }
                        _ => hx[at] = hx[at] - s * curl,
                    }
                }
            }
        }
        // (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x; Hyz carries -∂Ex/∂z, Hyx carries +∂Ez/∂x.
        for i in 0..nx {
            for j in 0..=ny {
                for k in 0..nz {
                    let a = ex[i * ex_i + j * ex_j + (k + 1)] - ex[i * ex_i + j * ex_j + k];
                    let b = ez[(i + 1) * ez_i + j * ez_j + k] - ez[i * ez_i + j * ez_j + k];
                    let curl = a - b;
                    let at = i * hy_i + j * hy_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_h[2][k] || sp.lossy_h[0][i] => {
                            sp.hyz[at] = sp.ca_h[2][k] * sp.hyz[at] - sp.cb_h[2][k] * a;
                            sp.hyx[at] = sp.ca_h[0][i] * sp.hyx[at] + sp.cb_h[0][i] * b;
                            hy[at] = sp.hyz[at] + sp.hyx[at];
                        }
                        _ => hy[at] = hy[at] - s * curl,
                    }
                }
            }
        }
        // (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y; Hzx carries -∂Ey/∂x, Hzy carries +∂Ex/∂y.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..=nz {
                    let a = ey[(i + 1) * ey_i + j * ey_j + k] - ey[i * ey_i + j * ey_j + k];
                    let b = ex[i * ex_i + (j + 1) * ex_j + k] - ex[i * ex_i + j * ex_j + k];
                    let curl = a - b;
                    let at = i * hz_i + j * hz_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_h[0][i] || sp.lossy_h[1][j] => {
                            sp.hzx[at] = sp.ca_h[0][i] * sp.hzx[at] - sp.cb_h[0][i] * a;
                            sp.hzy[at] = sp.ca_h[1][j] * sp.hzy[at] + sp.cb_h[1][j] * b;
                            hz[at] = sp.hzx[at] + sp.hzy[at];
                        }
                        _ => hz[at] = hz[at] - s * curl,
                    }
                }
            }
        }

        // E^{n+1} = E^n + S·(∇×H)^{n+1/2}, interior only: the tangential E on a
        // PEC wall is held at zero and is never written.
        // (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z; Exy carries +∂Hz/∂y, Exz carries -∂Hy/∂z.
        for i in 0..nx {
            for j in 1..ny {
                for k in 1..nz {
                    let a = hz[i * hz_i + j * hz_j + k] - hz[i * hz_i + (j - 1) * hz_j + k];
                    let b = hy[i * hy_i + j * hy_j + k] - hy[i * hy_i + j * hy_j + (k - 1)];
                    let curl = a - b;
                    let at = i * ex_i + j * ex_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_e[1][j] || sp.lossy_e[2][k] => {
                            sp.exy[at] = sp.ca_e[1][j] * sp.exy[at] + sp.cb_e[1][j] * a;
                            sp.exz[at] = sp.ca_e[2][k] * sp.exz[at] - sp.cb_e[2][k] * b;
                            ex[at] = sp.exy[at] + sp.exz[at];
                        }
                        _ => ex[at] = ex[at] + s * curl,
                    }
                }
            }
        }
        // (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x; Eyz carries +∂Hx/∂z, Eyx carries -∂Hz/∂x.
        for i in 1..nx {
            for j in 0..ny {
                for k in 1..nz {
                    let a = hx[i * hx_i + j * hx_j + k] - hx[i * hx_i + j * hx_j + (k - 1)];
                    let b = hz[i * hz_i + j * hz_j + k] - hz[(i - 1) * hz_i + j * hz_j + k];
                    let curl = a - b;
                    let at = i * ey_i + j * ey_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_e[2][k] || sp.lossy_e[0][i] => {
                            sp.eyz[at] = sp.ca_e[2][k] * sp.eyz[at] + sp.cb_e[2][k] * a;
                            sp.eyx[at] = sp.ca_e[0][i] * sp.eyx[at] - sp.cb_e[0][i] * b;
                            ey[at] = sp.eyz[at] + sp.eyx[at];
                        }
                        _ => ey[at] = ey[at] + s * curl,
                    }
                }
            }
        }
        // (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y; Ezx carries +∂Hy/∂x, Ezy carries -∂Hx/∂y.
        for i in 1..nx {
            for j in 1..ny {
                for k in 0..nz {
                    let a = hy[i * hy_i + j * hy_j + k] - hy[(i - 1) * hy_i + j * hy_j + k];
                    let b = hx[i * hx_i + j * hx_j + k] - hx[i * hx_i + (j - 1) * hx_j + k];
                    let curl = a - b;
                    let at = i * ez_i + j * ez_j + k;
                    match ab.as_deref_mut() {
                        Some(sp) if sp.lossy_e[0][i] || sp.lossy_e[1][j] => {
                            sp.ezx[at] = sp.ca_e[0][i] * sp.ezx[at] + sp.cb_e[0][i] * a;
                            sp.ezy[at] = sp.ca_e[1][j] * sp.ezy[at] - sp.cb_e[1][j] * b;
                            ez[at] = sp.ezx[at] + sp.ezy[at];
                        }
                        _ => ez[at] = ez[at] + s * curl,
                    }
                }
            }
        }
    }

    /// Discrete `∇·B` on cell `(i, j, k)`, summed over the cell's six faces.
    ///
    /// ⚠️ The contract is **a small bounded residual, not zero, and the residual
    /// is not a defect.** On a Yee lattice `∇·(∇×E)` telescopes to zero, and it
    /// would do so exactly if the arithmetic were exact — the identity is
    /// structural, not approximate. What breaks it is that [`Fix128`]
    /// multiplication **truncates**, so `Σ(face·S)` is not `(Σface)·S`
    /// (measured 1–3 ULP per application). The residual therefore accumulates
    /// at roughly one ULP per step and is a property of Q64.64, not of the
    /// update: measured 466 ULP after 500 steps on a PEC lattice, and 31 ULP
    /// after 200 steps over the loss-free cells of a PML lattice (2026-09-30).
    /// ⚠️ A reader who expects the textbook "exactly zero" will read those
    /// numbers as a bug and go looking for one.
    ///
    /// ⚠️ Inside an absorbing layer the residual is **not** bounded like that
    /// and is not meant to be: a split half field is book-keeping rather than a
    /// component of `B`, and the two halves of a face carry different decay
    /// coefficients, so the face sum has nothing to telescope. Measured on the
    /// same lattice, the whole-lattice maximum is 8e-4 — about 1.5e16 ULP,
    /// fourteen orders above the loss-free cells. That gap is why
    /// `div_b_stays_bounded_outside_the_layer` restricts itself with
    /// [`YeeGrid::is_absorbing`], and why it also asserts that the unrestricted
    /// maximum is larger: without that second assertion the restriction would
    /// be indistinguishable from a decoration that quietly hides a real
    /// regression.
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
