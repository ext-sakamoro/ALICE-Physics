//! The multigrid pressure projection run as `z` slabs that exchange halo layers.
//!
//! [`project_pressure_multigrid_decomposed`] is
//! [`project_pressure_multigrid`](super::project_pressure_multigrid) split across
//! `ranks` contiguous slabs. It is the stage that carries the multigrid's
//! iteration count (a handful of cycles, flat in the grid size) onto the slab
//! decomposition that [`super::project_pressure_decomposed`] gave the
//! Gauss-Seidel solve, whose iteration count grows with the grid.
//!
//! # What is exact, and why
//!
//! The result is **bit-identical** to the single-process solve for every rank
//! count, which is the oracle (`the_decomposed_cycle_reproduces_the_single_process_one`).
//! Three facts carry it:
//!
//! - a red-black colour sweep reads only the opposite colour, so a sweep may be
//!   split across ranks (see `project_pressure_red_black_gs`), and one halo layer
//!   suffices for the 7-point stencil;
//! - restriction (sum over a `2×2×2` aggregate) and prolongation (injection)
//!   move a layer `k` to layer `k / 2` and back, so they are rank-local when each
//!   rank's slab bounds are a multiple of two at every level that is
//!   distributed — [`layout`] chooses bounds that are;
//! - `Fix128` addition is a group operation mod 2¹²⁸, so the order partial sums
//!   are formed in cannot change a result.
//!
//! # Levels with fewer layers than ranks
//!
//! Coarsening halves `nz`, so below some level a rank would own no layer. From
//! that level on the cycle is **agglomerated**: the residual of the last
//! distributed level is gathered to rank 0, rank 0 runs the remaining levels with
//! the single-process [`super::mg_vcycle`], and the prolonged correction is sent
//! back to the layers' owners. Those levels hold at most `1/8` of the cells of
//! the last distributed one, so the serial part does not dominate; a cheaper
//! coarse solve is a later stage.
//!
//! # Storage
//!
//! Each rank holds, for every distributed level, its owned layers plus one halo
//! layer of the pressure ([`SlabStorage`]), and for its owned layers only the
//! conductances, inverse degrees, right-hand side and residual. Reading a layer
//! outside the band is a panic with a message, not a stale value, so a halo that
//! is too narrow shows up as the abort it is. Rank 0 additionally holds every
//! layer of the last distributed level while it gathers the residual and while it
//! sends the correction back.
//!
//! # What this stage does not do
//!
//! The setup (conductances, inverse degrees, right-hand side, the agglomerated
//! hierarchy) is still built for the whole grid and sliced per rank; a driver that
//! runs one rank per process, and a setup built from a band, are the next stage.

use super::{
    exchange_slab_halos_local, mg_vcycle, poisson_rhs, subtract_pressure_gradient,
    subtract_slab_pressure_gradient, HaloSchedule, LocalSlabTransport, MacGrid, MgLevel,
    PoissonMask, SlabBytes, SlabFaces, SlabStencil, SlabStorage, SlabTransport, SweepWindow,
    MG_COARSE_VISITS, MG_CORRECTION_SCALE_DEN, MG_CORRECTION_SCALE_NUM, MG_POST_SMOOTH,
    MG_PRE_SMOOTH,
};
use crate::eulerian_grid::slab_bounds;
use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Halo width a 7-point stencil needs.
pub(super) const HALO: usize = 1;

const BELOW_MISSING: &str =
    "an open z face of an owned cell needs the layer below, which this rank's band does not hold";
const ABOVE_MISSING: &str =
    "an open z face of an owned cell needs the layer above, which this rank's band does not hold";
const OWNED_LAYER_MISSING: &str = "a rank's own layer is outside its own band";

/// How a slab decomposition of the finest level carries down the hierarchy.
#[derive(Debug, PartialEq, Eq)]
struct Layout {
    /// The last level that is distributed; levels below it are agglomerated.
    last: usize,
    /// `bounds[l][r]`: the layers of level `l` that rank `r` owns, for
    /// `l ≤ last`.
    bounds: Vec<Vec<(usize, usize)>>,
}

/// Slab bounds for every distributed level, given each level's `nz` (finest
/// first, non-increasing, each either equal to or half of the one above).
///
/// The last distributed level is the deepest one that still has a layer for
/// every rank. The finest bounds are a multiple of `2^h` (`h` = the number of
/// halvings down to that level), so every level down to it gets bounds that are
/// whole layers: `bounds[l+1] = bounds[l] / (nz[l] / nz[l+1])` is exact.
///
/// When even the finest level has fewer layers than ranks the finest level is
/// split as the Gauss-Seidel decomposition splits it (some ranks empty) and
/// everything below it is agglomerated.
fn layout(nz: &[usize], ranks: usize) -> Layout {
    let last = (0..nz.len()).rev().find(|&l| nz[l] >= ranks).unwrap_or(0);
    let halvings = (0..last).filter(|&l| nz[l] != nz[l + 1]).count();
    let unit = 1usize << halvings;
    let mut bounds = vec![(0..ranks)
        .map(|r| {
            let (k0, k1) = slab_bounds(nz[0] / unit, ranks, r);
            (k0 * unit, k1 * unit)
        })
        .collect::<Vec<_>>()];
    for l in 0..last {
        let factor = nz[l] / nz[l + 1];
        let next = bounds[l]
            .iter()
            .map(|&(k0, k1)| (k0 / factor, k1 / factor))
            .collect();
        bounds.push(next);
    }
    Layout { last, bounds }
}

/// The layers each distributed level's pressure band ended up holding, rank by
/// rank: what `resident` reports after the solve. Tests read it to check that a
/// rank holds its owned layers plus one halo layer and no more.
pub(crate) type Residency = Vec<Vec<(usize, usize)>>;

/// [`super::project_pressure_multigrid`] run as `ranks` contiguous `z` slabs over
/// the in-process [`LocalSlabTransport`], each rank holding only its band.
///
/// Same early-return contract as the single-process solve (a bit-identical grid
/// for a non-power-of-two extent, a zero `dx` / `dt_s` / density, or
/// `cycles == 0`), and also for `ranks == 0`.
pub(crate) fn project_pressure_multigrid_decomposed(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
) {
    project_pressure_multigrid_decomposed_over(
        grid,
        dt_s,
        density_kg_m3,
        cycles,
        ranks,
        schedule,
        local_slab_transport,
    );
}

/// The default transport factory: a [`LocalSlabTransport`] over `bounds`, each
/// band starting from `field` (zero when there is none).
fn local_slab_transport(
    bounds: &[(usize, usize)],
    nz: usize,
    plane: usize,
    halo: usize,
    field: Option<&[Fix128]>,
) -> LocalSlabTransport {
    match field {
        Some(f) => LocalSlabTransport::from_field(bounds, nz, plane, halo, f),
        None => {
            LocalSlabTransport::from_field(bounds, nz, plane, halo, &vec![Fix128::ZERO; nz * plane])
        }
    }
}

/// [`project_pressure_multigrid_decomposed`] over a caller-supplied
/// [`SlabTransport`]. `make(bounds, nz, plane, halo, field)` builds the transport
/// for one field: a band per rank covering its `bounds` widened by `halo`, filled
/// from `field` (zero when `None`). The solve needs one per distributed level for
/// the pressure, and two more at the last distributed level for the agglomeration
/// (where rank 0 holds every layer).
///
/// Returns the band every rank's pressure transport held, level by level.
///
/// # Setup is still global
///
/// The conductances, inverse degrees and right-hand side are built for the whole
/// grid and each rank is handed the slice for its owned layers; what a rank keeps
/// for the solve itself — pressure, right-hand side, residual — is band-local.
/// [`project_pressure_multigrid_banded_on_rank`] is the form that builds the
/// setup from a band, so that no rank ever sees the whole grid.
pub(crate) fn project_pressure_multigrid_decomposed_over<T, F>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    make: F,
) -> Residency
where
    T: SlabTransport,
    F: FnMut(&[(usize, usize)], usize, usize, usize, Option<&[Fix128]>) -> T,
{
    solve_decomposed(grid, dt_s, density_kg_m3, cycles, ranks, schedule, make)
}

/// The decomposed solve, driving every rank from this one address space.
fn solve_decomposed<T, F>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    mut make: F,
) -> Residency
where
    T: SlabTransport,
    F: FnMut(&[(usize, usize)], usize, usize, usize, Option<&[Fix128]>) -> T,
{
    let pow2 = |n: usize| n.is_power_of_two();
    if cycles == 0
        || ranks == 0
        || !(pow2(grid.nx) && pow2(grid.ny) && pow2(grid.nz))
        || grid.dx.is_zero()
        || density_kg_m3.is_zero()
        || dt_s.is_zero()
    {
        return Vec::new();
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
    let nzs: Vec<usize> = levels.iter().map(|l| l.nz).collect();
    let Layout { last, bounds } = layout(&nzs, ranks);
    let plane = |l: usize| levels[l].nx * levels[l].ny;

    // Per rank and level, only the owned layers.
    let active: Vec<usize> = (0..ranks).collect();
    let local: Vec<Vec<Local>> = (0..=last)
        .map(|l| {
            let p = plane(l);
            bounds[l]
                .iter()
                .map(|&(k0, k1)| {
                    let (a, b) = (k0 * p, k1 * p);
                    Local {
                        cond: levels[l].cond[a..b].to_vec(),
                        inv: invs[l][a..b].to_vec(),
                        rhs: if l == 0 {
                            rhs0[a..b].to_vec()
                        } else {
                            vec![Fix128::ZERO; b - a]
                        },
                        res: vec![Fix128::ZERO; b - a],
                    }
                })
                .collect()
        })
        .collect();

    // Rank 0 holds every layer of the last level's residual and correction; the
    // others hold only their own.
    let root_bounds: Vec<(usize, usize)> = bounds[last]
        .iter()
        .enumerate()
        .map(|(r, &b)| if r == 0 { (0, levels[last].nz) } else { b })
        .collect();

    let root_bounds0: Vec<(usize, usize)> = bounds[0]
        .iter()
        .enumerate()
        .map(|(r, &b)| if r == 0 { (0, levels[0].nz) } else { b })
        .collect();

    let mut solve = Decomposed {
        active: active.clone(),
        pressure: (0..=last)
            .map(|l| {
                let field = (l == 0).then_some(grid.pressure.as_slice());
                make(&bounds[l], levels[l].nz, plane(l), HALO, field)
            })
            .collect(),
        gather: make(&root_bounds, levels[last].nz, plane(last), 0, None),
        finish: Some(make(&root_bounds0, levels[0].nz, plane(0), 0, None)),
        correction: make(&root_bounds, levels[last].nz, plane(last), 0, None),
        coarse_p: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        coarse_rhs: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        coarse_invs: invs[last + 1..].to_vec(),
        local,
        levels: &levels,
        bounds: &bounds,
        last,
        ranks,
        schedule,
    };

    for _ in 0..cycles {
        solve.cycle(0);
    }

    let plane0 = plane(0);
    solve.finish_to_root(&bounds[0]);
    let residency: Residency = solve
        .pressure
        .iter_mut()
        .map(|t| active.iter().map(|&r| t.slab_mut(r).resident()).collect())
        .collect();
    // Rank 0 holds the whole field, so the gradient is taken from its copy.
    let root = solve
        .finish
        .as_mut()
        .expect("the grid path always builds the finish transport")
        .slab_mut(0);
    for k in 0..levels[0].nz {
        grid.pressure[k * plane0..(k + 1) * plane0]
            .copy_from_slice(root.layer(k).expect(OWNED_LAYER_MISSING));
    }

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
    residency
}

/// The `(nx, ny, nz)` of every level of the hierarchy, finest first, down to a
/// single cell — the shape [`MgLevel::coarsen`] produces, without its
/// conductances.
fn level_dims(nx: usize, ny: usize, nz: usize) -> Vec<(usize, usize, usize)> {
    let mut dims = vec![(nx, ny, nz)];
    while let Some(&(x, y, z)) = dims.last().filter(|&&(x, y, z)| x * y * z > 1) {
        let half = |n: usize| if n > 1 { n / 2 } else { 1 };
        dims.push((half(x), half(y), half(z)));
    }
    dims
}

/// The layers each rank owns of the finest level under the decomposition the
/// multigrid solve uses — what the faces handed to
/// [`project_pressure_multigrid_banded_on_rank`] have to describe.
///
/// These are not the Gauss-Seidel decomposition's `slab_bounds`: the finest
/// bounds are a multiple of `2^h` so that the transfers stay rank-local (see
/// [`layout`]). `None` when an extent is not a power of two.
pub(crate) fn multigrid_slab_bounds(
    nx: usize,
    ny: usize,
    nz: usize,
    ranks: usize,
) -> Option<Vec<(usize, usize)>> {
    if ranks == 0 || !(nx.is_power_of_two() && ny.is_power_of_two() && nz.is_power_of_two()) {
        return None;
    }
    let nzs: Vec<usize> = level_dims(nx, ny, nz).iter().map(|d| d.2).collect();
    Some(layout(&nzs, ranks).bounds.swap_remove(0))
}

/// One rank's half of the decomposed multigrid solve, for a rank that holds **no
/// `MacGrid`**: its faces and its pressure band are all it has. The schedule is
/// [`project_pressure_multigrid_decomposed_over`]'s, narrowed to `my_rank`: the
/// exchange, the gather and the correction are walked in full on every rank, in
/// the same order, and the transport performs only the half this rank is party
/// to.
///
/// Everything the solve needs is built from `faces` — the right-hand side, the
/// conductances and the inverse degrees through [`SlabStencil`] (the same
/// expressions the full-grid path uses), the coarser levels by coarsening the
/// band, which is exact because the slab bounds are a multiple of two at every
/// distributed level. The one thing a rank cannot build alone is the hierarchy
/// below the last distributed level, which rank 0 solves: every rank sends it its
/// conductances at that level (six small integers per cell, carried as `Fix128`
/// through the same transport the residual uses) and rank 0 coarsens from there.
///
/// # What the caller supplies
///
/// `faces` describes the layers [`multigrid_slab_bounds`] gives `my_rank`, with the
/// face conditions already imposed (the precondition of [`SlabFaces`]). `pressure`
/// holds the starting field over the owned layers and one halo layer either side,
/// because the first sweep reads the halo before any exchange. On return it holds
/// the final pressure over the same layers and `faces` the corrected velocities;
/// nothing is gathered, so no rank ever holds the whole field.
///
/// A degenerate `dx`, density, step, cycle count or extent leaves everything
/// untouched, as the other drivers do.
///
/// Returns what the finest-level stencil this rank built allocated
/// ([`SlabStencil::bytes`]), and [`SlabBytes::ZERO`] when it returned before
/// building one. The coarser levels and the transports are not in it.
///
/// # Panics
///
/// When `faces` does not describe the layers the decomposition gives `my_rank`, or
/// `pressure` does not hold the band the first sweep reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pressure_multigrid_banded_on_rank<T, F>(
    faces: &mut SlabFaces,
    pressure: &mut SlabStorage,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    my_rank: usize,
    mut make: F,
) -> SlabBytes
where
    T: SlabTransport,
    F: FnMut(&[(usize, usize)], usize, usize, usize, Option<&[Fix128]>) -> T,
{
    let (nx, ny, nz) = (faces.nx, faces.ny, faces.nz);
    let pow2 = |n: usize| n.is_power_of_two();
    if cycles == 0
        || ranks == 0
        || my_rank >= ranks
        || !(pow2(nx) && pow2(ny) && pow2(nz))
        || faces.dx.is_zero()
        || density_kg_m3.is_zero()
        || dt_s.is_zero()
    {
        return SlabBytes::ZERO;
    }
    let dims = level_dims(nx, ny, nz);
    let nzs: Vec<usize> = dims.iter().map(|d| d.2).collect();
    let Layout { last, bounds } = layout(&nzs, ranks);
    assert_eq!(
        faces.owned(),
        bounds[0][my_rank],
        "rank {my_rank} was handed the faces of layers {:?} but the multigrid decomposition gives \
         it {:?}",
        faces.owned(),
        bounds[0][my_rank],
    );
    let plane = |l: usize| dims[l].0 * dims[l].1;

    let scale = density_kg_m3 * faces.dx * faces.dx / dt_s;
    let stencil = SlabStencil::build(faces, scale);

    // The band at every distributed level: conductances by coarsening the band.
    let mut band = vec![MgLevel {
        nx,
        ny,
        nz: stencil.k1 - stencil.k0,
        cond: stencil.open.iter().map(|o| o.map(i64::from)).collect(),
    }];
    for l in 0..last {
        let coarse = band[l].coarsen();
        band.push(coarse);
    }

    let mut local: Vec<Vec<Local>> = Vec::with_capacity(last + 1);
    for (l, b) in band.iter().enumerate() {
        let cells = b.cells();
        let mut per_rank: Vec<Local> = (0..ranks)
            .map(|_| Local {
                cond: Vec::new(),
                inv: Vec::new(),
                rhs: Vec::new(),
                res: Vec::new(),
            })
            .collect();
        per_rank[my_rank] = Local {
            cond: b.cond.clone(),
            inv: b.inverse_degrees(),
            rhs: if l == 0 {
                stencil.rhs.clone()
            } else {
                vec![Fix128::ZERO; cells]
            },
            res: vec![Fix128::ZERO; cells],
        };
        local.push(per_rank);
    }

    // Rank 0 holds every layer of the last distributed level (residual,
    // correction, conductances); the others hold only their own.
    let root_bounds: Vec<(usize, usize)> = bounds[last]
        .iter()
        .enumerate()
        .map(|(r, &b)| if r == 0 { (0, dims[last].2) } else { b })
        .collect();

    // The levels, with conductances only where this rank needs them: none for the
    // distributed levels (the band has them), the whole level for rank 0 below.
    let mut levels: Vec<MgLevel> = dims
        .iter()
        .map(|&(x, y, z)| MgLevel {
            nx: x,
            ny: y,
            nz: z,
            cond: Vec::new(),
        })
        .collect();
    let mut cond_gather = make(&root_bounds, dims[last].2, 6 * plane(last), 0, None);
    if last + 1 < levels.len() {
        let (k0, k1) = bounds[last][my_rank];
        let own = &local[last][my_rank].cond;
        let p = plane(last);
        let storage = cond_gather.slab_mut(my_rank);
        for k in k0..k1 {
            let layer = storage.layer_mut(k).expect(OWNED_LAYER_MISSING);
            for c in 0..p {
                for (f, &v) in own[(k - k0) * p + c].iter().enumerate() {
                    layer[6 * c + f] = Fix128::from_int(v);
                }
            }
        }
        for (r, &(a, b)) in bounds[last].iter().enumerate().skip(1) {
            for layer in a..b {
                cond_gather.deliver_layer(r, 0, layer);
            }
        }
        if my_rank == 0 {
            let root = cond_gather.slab_mut(0);
            let mut cond = Vec::with_capacity(dims[last].2 * p);
            for k in 0..dims[last].2 {
                let layer = root.layer(k).expect(OWNED_LAYER_MISSING);
                for c in 0..p {
                    let mut f = [0i64; 6];
                    for (slot, v) in f.iter_mut().zip(&layer[6 * c..6 * c + 6]) {
                        *slot = v.hi;
                    }
                    cond.push(f);
                }
            }
            levels[last].cond = cond;
            for l in last..levels.len() - 1 {
                levels[l + 1] = levels[l].coarsen();
            }
        }
    }
    let coarse_invs: Vec<Vec<Fix128>> = if my_rank == 0 {
        levels[last + 1..]
            .iter()
            .map(MgLevel::inverse_degrees)
            .collect()
    } else {
        Vec::new()
    };

    let mut solve = Decomposed {
        active: vec![my_rank],
        finish: None,
        pressure: (0..=last)
            .map(|l| make(&bounds[l], dims[l].2, plane(l), HALO, None))
            .collect(),
        gather: make(&root_bounds, dims[last].2, plane(last), 0, None),
        correction: make(&root_bounds, dims[last].2, plane(last), 0, None),
        coarse_p: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        coarse_rhs: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        coarse_invs,
        local,
        levels: &levels,
        bounds: &bounds,
        last,
        ranks,
        schedule,
    };

    // The starting field: the owned layers and the halo, from the caller's band.
    {
        let start = solve.pressure[0].slab_mut(my_rank);
        let (lo, hi) = start.resident();
        for k in lo..hi {
            start
                .layer_mut(k)
                .expect(OWNED_LAYER_MISSING)
                .copy_from_slice(pressure.layer(k).expect(
                    "the starting pressure must cover the owned layers and one halo layer",
                ));
        }
    }

    for _ in 0..cycles {
        solve.cycle(0);
    }

    let inv_dx = Fix128::ONE / faces.dx;
    let coeff = dt_s / density_kg_m3 * inv_dx;
    let done = solve.pressure[0].slab_mut(my_rank);
    subtract_slab_pressure_gradient(faces, done, coeff);
    let (lo, hi) = done.resident();
    for k in lo..hi {
        pressure
            .layer_mut(k)
            .expect("the band the starting field was read from")
            .copy_from_slice(done.layer(k).expect(OWNED_LAYER_MISSING));
    }
    stencil.bytes()
}

/// What one rank keeps for one level besides its pressure band, for its owned
/// layers only.
struct Local {
    cond: Vec<[i64; 6]>,
    inv: Vec<Fix128>,
    rhs: Vec<Fix128>,
    res: Vec<Fix128>,
}

/// `Σ cond·p` over the in-domain neighbours of cell `(i, j)` of layer `k`: the
/// stencil of [`MgLevel::neighbour_sum`], addressed layer by layer. The two `z`
/// reads are the only ones that can leave the band, and an open face obliges
/// them, so a halo narrower than the stencil aborts here.
fn neighbour_sum_layers(
    level: &MgLevel,
    f: [i64; 6],
    (below, centre, above): (Option<&[Fix128]>, &[Fix128], Option<&[Fix128]>),
    (i, j, k): (usize, usize, usize),
) -> Fix128 {
    let c = i + level.nx * j;
    let mut acc = Fix128::ZERO;
    if f[0] != 0 && i > 0 {
        acc = acc + centre[c - 1] * Fix128::from_int(f[0]);
    }
    if f[1] != 0 && i + 1 < level.nx {
        acc = acc + centre[c + 1] * Fix128::from_int(f[1]);
    }
    if f[2] != 0 && j > 0 {
        acc = acc + centre[c - level.nx] * Fix128::from_int(f[2]);
    }
    if f[3] != 0 && j + 1 < level.ny {
        acc = acc + centre[c + level.nx] * Fix128::from_int(f[3]);
    }
    if f[4] != 0 && k > 0 {
        acc = acc + below.expect(BELOW_MISSING)[c] * Fix128::from_int(f[4]);
    }
    if f[5] != 0 && k + 1 < level.nz {
        acc = acc + above.expect(ABOVE_MISSING)[c] * Fix128::from_int(f[5]);
    }
    acc
}

/// The state of one decomposed solve.
struct Decomposed<'a, T: SlabTransport> {
    levels: &'a [MgLevel],
    bounds: &'a [Vec<(usize, usize)>],
    /// The last distributed level.
    last: usize,
    ranks: usize,
    schedule: HaloSchedule,
    /// The ranks this driver runs: every rank in one process, or just this one.
    active: Vec<usize>,
    /// The finished pressure, gathered to rank 0 for the gradient.
    finish: Option<T>,
    /// One transport per distributed level, carrying that level's pressure band.
    pressure: Vec<T>,
    /// `local[l][r]`: what rank `r` keeps for level `l`.
    local: Vec<Vec<Local>>,
    /// The last level's residual, gathered to rank 0.
    gather: T,
    /// The coarse correction, sent from rank 0 to the owners.
    correction: T,
    /// Unknown and right-hand side of the agglomerated levels below `last`.
    coarse_p: Vec<Vec<Fix128>>,
    coarse_rhs: Vec<Vec<Fix128>>,
    /// Inverse degrees of the agglomerated levels.
    coarse_invs: Vec<Vec<Fix128>>,
}

impl<T: SlabTransport> Decomposed<'_, T> {
    fn exchange(&mut self, l: usize) {
        exchange_slab_halos_local(&mut self.pressure[l], &self.bounds[l], self.levels[l].nz);
    }

    /// Red-black Gauss-Seidel on level `l`, the colour order and the halo
    /// schedule of [`super::project_pressure_decomposed`].
    fn smooth(&mut self, l: usize, iterations: u32) {
        let level = &self.levels[l];
        let active = self.active.clone();
        for _ in 0..iterations {
            for colour in 0..2usize {
                for &r in &active {
                    let (k0, k1) = self.bounds[l][r];
                    let loc = &self.local[l][r];
                    let storage = self.pressure[l].slab_mut(r);
                    let plane = level.nx * level.ny;
                    for k in k0..k1 {
                        let SweepWindow {
                            below,
                            centre,
                            above,
                        } = storage.sweep_window(k);
                        for j in 0..level.ny {
                            for i in 0..level.nx {
                                if (i + j + k) % 2 != colour {
                                    continue;
                                }
                                let lc = (k - k0) * plane + i + level.nx * j;
                                let nb = neighbour_sum_layers(
                                    level,
                                    loc.cond[lc],
                                    (below, centre, above),
                                    (i, j, k),
                                );
                                centre[i + level.nx * j] = (nb - loc.rhs[lc]) * loc.inv[lc];
                            }
                        }
                    }
                }
                if self.schedule == HaloSchedule::EverySweep {
                    self.exchange(l);
                }
            }
            if self.schedule == HaloSchedule::EveryIteration {
                self.exchange(l);
            }
        }
    }

    /// `res = rhs − A p` on every rank's own layers of level `l`.
    fn compute_residual(&mut self, l: usize) {
        let level = &self.levels[l];
        let plane = level.nx * level.ny;
        for &r in &self.active {
            let (k0, k1) = self.bounds[l][r];
            let storage: &SlabStorage = self.pressure[l].slab_mut(r);
            let loc = &mut self.local[l][r];
            for k in k0..k1 {
                let centre = storage.layer(k).expect(OWNED_LAYER_MISSING);
                let below = k.checked_sub(1).and_then(|b| storage.layer(b));
                let above = storage.layer(k + 1);
                for j in 0..level.ny {
                    for i in 0..level.nx {
                        let lc = (k - k0) * plane + i + level.nx * j;
                        let f = loc.cond[lc];
                        let degree: i64 = f.iter().sum();
                        let ap = centre[i + level.nx * j] * Fix128::from_int(-degree)
                            + neighbour_sum_layers(level, f, (below, centre, above), (i, j, k));
                        loc.res[lc] = loc.rhs[lc] - ap;
                    }
                }
            }
        }
    }

    /// Zero level `l`'s unknown on every rank: owned layers and halo.
    fn zero_pressure(&mut self, l: usize) {
        for &r in &self.active {
            let storage = self.pressure[l].slab_mut(r);
            let (lo, hi) = storage.resident();
            for k in lo..hi {
                storage
                    .layer_mut(k)
                    .expect(OWNED_LAYER_MISSING)
                    .fill(Fix128::ZERO);
            }
        }
    }

    /// One cycle on level `l`, mirroring [`super::mg_vcycle`] step for step.
    fn cycle(&mut self, l: usize) {
        if l == self.last && l + 1 == self.levels.len() {
            // Coarsest level: one cell, for which one sweep from zero is exact.
            self.zero_pressure(l);
            self.smooth(l, 1);
            return;
        }
        self.smooth(l, MG_PRE_SMOOTH);
        self.compute_residual(l);
        if l == self.last {
            self.agglomerated_correction(l);
        } else {
            self.restrict(l);
            self.zero_pressure(l + 1);
            for _ in 0..MG_COARSE_VISITS {
                self.cycle(l + 1);
            }
            self.prolong(l);
        }
        self.exchange(l);
        self.smooth(l, MG_POST_SMOOTH);
    }

    /// Sum of the residual over each `2×2×2` aggregate into level `l + 1`'s
    /// right-hand side, on the layers each rank owns at `l + 1`.
    fn restrict(&mut self, l: usize) {
        let (fine, coarse) = (&self.levels[l], &self.levels[l + 1]);
        let (fx, fy, fz) = (
            fine.nx / coarse.nx,
            fine.ny / coarse.ny,
            fine.nz / coarse.nz,
        );
        let (plane_f, plane_c) = (fine.nx * fine.ny, coarse.nx * coarse.ny);
        let (lower, upper) = self.local.split_at_mut(l + 1);
        for &r in &self.active {
            let (k0, k1) = self.bounds[l][r];
            let c0 = self.bounds[l + 1][r].0;
            let res = &lower[l][r].res;
            let rc = &mut upper[0][r].rhs;
            rc.fill(Fix128::ZERO);
            for k in k0..k1 {
                for j in 0..fine.ny {
                    for i in 0..fine.nx {
                        let ci = (k / fz - c0) * plane_c + i / fx + coarse.nx * (j / fy);
                        rc[ci] = rc[ci] + res[(k - k0) * plane_f + i + fine.nx * j];
                    }
                }
            }
        }
    }

    /// Add the prolonged, scaled correction of level `l + 1` to level `l`.
    fn prolong(&mut self, l: usize) {
        let (fine, coarse) = (&self.levels[l], &self.levels[l + 1]);
        let (fx, fy, fz) = (
            fine.nx / coarse.nx,
            fine.ny / coarse.ny,
            fine.nz / coarse.nz,
        );
        let scale = Fix128::from_ratio(MG_CORRECTION_SCALE_NUM, MG_CORRECTION_SCALE_DEN);
        let (lower, upper) = self.pressure.split_at_mut(l + 1);
        for &r in &self.active {
            let (k0, k1) = self.bounds[l][r];
            let e = upper[0].slab_mut(r);
            let p = lower[l].slab_mut(r);
            for k in k0..k1 {
                let ecoarse = e.layer(k / fz).expect(OWNED_LAYER_MISSING);
                let pl = p.layer_mut(k).expect(OWNED_LAYER_MISSING);
                for j in 0..fine.ny {
                    for i in 0..fine.nx {
                        let ci = i / fx + coarse.nx * (j / fy);
                        let c = i + fine.nx * j;
                        pl[c] = pl[c] + ecoarse[ci] * scale;
                    }
                }
            }
        }
    }

    /// The last distributed level's coarse correction: gather the residual to
    /// rank 0, run the remaining levels there, send each owner its layers of the
    /// prolonged correction and add it.
    ///
    /// The two walks — every rank's layers to rank 0, then rank 0's to every rank —
    /// are taken in full by every driver; the transport performs the half it is
    /// party to. Only rank 0's driver forms the coarse right-hand side, solves, and
    /// writes the correction.
    fn agglomerated_correction(&mut self, l: usize) {
        let level = &self.levels[l];
        let plane = level.nx * level.ny;
        for &r in &self.active {
            let (k0, k1) = self.bounds[l][r];
            let storage = self.gather.slab_mut(r);
            for k in k0..k1 {
                storage
                    .layer_mut(k)
                    .expect(OWNED_LAYER_MISSING)
                    .copy_from_slice(&self.local[l][r].res[(k - k0) * plane..(k - k0 + 1) * plane]);
            }
        }
        for (r, &(k0, k1)) in self.bounds[l].iter().enumerate().skip(1) {
            for layer in k0..k1 {
                self.gather.deliver_layer(r, 0, layer);
            }
        }

        if self.active.contains(&0) {
            let coarse = &self.levels[l + 1];
            let (fx, fy, fz) = (
                level.nx / coarse.nx,
                level.ny / coarse.ny,
                level.nz / coarse.nz,
            );
            let rc = &mut self.coarse_rhs[0];
            rc.fill(Fix128::ZERO);
            let root = self.gather.slab_mut(0);
            for k in 0..level.nz {
                let res = root.layer(k).expect(OWNED_LAYER_MISSING);
                for j in 0..level.ny {
                    for i in 0..level.nx {
                        let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                        rc[ci] = rc[ci] + res[i + level.nx * j];
                    }
                }
            }
            self.coarse_p[0].fill(Fix128::ZERO);
            for _ in 0..MG_COARSE_VISITS {
                mg_vcycle(
                    &self.levels[l + 1..],
                    &self.coarse_invs,
                    &mut self.coarse_p,
                    &mut self.coarse_rhs,
                );
            }

            let scale = Fix128::from_ratio(MG_CORRECTION_SCALE_NUM, MG_CORRECTION_SCALE_DEN);
            let out = self.correction.slab_mut(0);
            for k in 0..level.nz {
                let layer = out.layer_mut(k).expect(OWNED_LAYER_MISSING);
                for j in 0..level.ny {
                    for i in 0..level.nx {
                        let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                        layer[i + level.nx * j] = self.coarse_p[0][ci] * scale;
                    }
                }
            }
        }
        for (r, &(k0, k1)) in self.bounds[l].iter().enumerate().skip(1) {
            for layer in k0..k1 {
                self.correction.deliver_layer(0, r, layer);
            }
        }
        for &r in &self.active {
            let (k0, k1) = self.bounds[l][r];
            let corr = self.correction.slab_mut(r);
            let p = self.pressure[l].slab_mut(r);
            for k in k0..k1 {
                let add = corr.layer(k).expect(OWNED_LAYER_MISSING);
                let pl = p.layer_mut(k).expect(OWNED_LAYER_MISSING);
                for (x, a) in pl.iter_mut().zip(add) {
                    *x = *x + *a;
                }
            }
        }
    }

    /// Bring every rank's owned layers of the finished pressure to rank 0.
    fn finish_to_root(&mut self, bounds0: &[(usize, usize)]) {
        let Some(finish) = self.finish.as_mut() else {
            return;
        };
        for &r in &self.active {
            let (k0, k1) = bounds0[r];
            let from = self.pressure[0].slab_mut(r);
            let to = finish.slab_mut(r);
            for k in k0..k1 {
                to.layer_mut(k)
                    .expect(OWNED_LAYER_MISSING)
                    .copy_from_slice(from.layer(k).expect(OWNED_LAYER_MISSING));
            }
        }
        for (r, &(k0, k1)) in bounds0.iter().enumerate().skip(1) {
            for layer in k0..k1 {
                finish.deliver_layer(r, 0, layer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eulerian_grid::project_pressure_multigrid;

    fn fx(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    /// Which conditions a scene carries.
    #[derive(Clone, Copy, Debug)]
    enum Scene {
        /// Divergent flow, every face plain fluid.
        Open,
        /// Closed box, an interior wall, and a varied initial pressure — so the
        /// coarse conductances are not uniform and the initial halo matters.
        Walled,
    }

    /// The seed field's three components, as functions of the face — one
    /// definition, so a rank that builds only its own faces gets the numbers the
    /// whole-grid seed has.
    fn seed_u(i: usize, j: usize, k: usize) -> Fix128 {
        fx(((i * 3 + j + 2 * k) % 7) as i64 - 3, 4)
    }
    fn seed_v(i: usize, j: usize, k: usize) -> Fix128 {
        fx(((i + j * 5 + k) % 5) as i64 - 2, 8)
    }
    fn seed_w(i: usize, j: usize, k: usize) -> Fix128 {
        fx(((i + 2 * j + k * 3) % 11) as i64 - 5, 4)
    }

    /// A divergent field in all three axes (so a wrong layer or axis is a
    /// different number), on an `nx × ny × nz` grid.
    fn seed(nx: usize, ny: usize, nz: usize, scene: Scene) -> MacGrid {
        let mut g = MacGrid::new(nx, ny, nz, Fix128::ONE);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let ix = g.idx_u(i, j, k);
                    g.u[ix] = seed_u(i, j, k);
                }
            }
        }
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let ix = g.idx_v(i, j, k);
                    g.v[ix] = seed_v(i, j, k);
                }
            }
        }
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let ix = g.idx_w(i, j, k);
                    g.w[ix] = seed_w(i, j, k);
                }
            }
        }
        if let Scene::Walled = scene {
            g.set_closed_box_walls();
            // An interior wall that is not aligned with an aggregate boundary.
            if nx > 2 && ny > 1 && nz > 1 {
                for k in 0..nz {
                    g.set_u_solid(nx / 2 + 1, ny / 2, k, true);
                }
            }
            for (c, slot) in g.pressure.iter_mut().enumerate() {
                *slot = fx((c % 5) as i64 - 2, 7);
            }
        }
        g
    }

    fn bit_equal(a: &MacGrid, b: &MacGrid) -> bool {
        a.pressure == b.pressure && a.u == b.u && a.v == b.v && a.w == b.w
    }

    const DT: (i64, i64) = (1, 100);
    const RHO: i64 = 1000;

    fn solve_single(base: &MacGrid, cycles: u32) -> MacGrid {
        let mut g = base.clone();
        project_pressure_multigrid(&mut g, fx(DT.0, DT.1), Fix128::from_int(RHO), cycles);
        g
    }

    fn solve_split(base: &MacGrid, cycles: u32, ranks: usize, schedule: HaloSchedule) -> MacGrid {
        let mut g = base.clone();
        project_pressure_multigrid_decomposed(
            &mut g,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            cycles,
            ranks,
            schedule,
        );
        g
    }

    /// Grids (cubes, anisotropic, a single layer, fewer layers than ranks) × rank
    /// counts (even, odd, more than layers), open and walled.
    fn cases() -> Vec<((usize, usize, usize), usize)> {
        let mut out = Vec::new();
        for &ranks in &[1usize, 2, 3, 4, 8] {
            out.push(((8, 8, 8), ranks));
        }
        for &ranks in &[2usize, 4, 8] {
            out.push(((16, 8, 4), ranks));
        }
        for &ranks in &[3usize, 5, 8, 16] {
            out.push(((4, 4, 16), ranks));
        }
        out.push(((8, 8, 1), 1));
        out.push(((8, 8, 1), 2));
        out.push(((16, 16, 16), 4));
        out
    }

    /// The oracle: the decomposed cycle lands on the single-process answer to the
    /// bit, for every grid, rank count and scene. Exactness, not a tolerance.
    #[test]
    fn the_decomposed_cycle_reproduces_the_single_process_one() {
        for scene in [Scene::Open, Scene::Walled] {
            for ((nx, ny, nz), ranks) in cases() {
                let base = seed(nx, ny, nz, scene);
                let want = solve_single(&base, 2);
                assert!(
                    !bit_equal(&base, &want),
                    "{nx}x{ny}x{nz} {scene:?}: the single-process solve left the grid alone, \
                     so equality below would say nothing"
                );
                let got = solve_split(&base, 2, ranks, HaloSchedule::EverySweep);
                assert!(
                    bit_equal(&want, &got),
                    "{nx}x{ny}x{nz} over {ranks} ranks, {scene:?}: the decomposed multigrid \
                     cycle is not the single-process one"
                );
            }
        }
    }

    /// Teeth: a halo exchanged once per iteration instead of once per colour
    /// sweep leaves the second sweep reading a stale layer, and the answer moves.
    #[test]
    fn a_stale_halo_does_not_reproduce_the_single_process_cycle() {
        for scene in [Scene::Open, Scene::Walled] {
            let base = seed(8, 8, 8, scene);
            let want = solve_single(&base, 2);
            let stale = solve_split(&base, 2, 4, HaloSchedule::EveryIteration);
            assert!(
                !bit_equal(&want, &stale),
                "{scene:?}: a stale halo still reproduced the single-process cycle, so the \
                 oracle above does not depend on the exchange arriving on time"
            );
        }
    }

    /// A transport that accepts every delivery and moves nothing.
    struct Mute(LocalSlabTransport);

    impl SlabTransport for Mute {
        fn slab_mut(&mut self, rank: usize) -> &mut SlabStorage {
            self.0.slab_mut(rank)
        }
        fn deliver_layer(&mut self, _src: usize, _dst: usize, _layer: usize) {}
    }

    /// Teeth: with no delivery at all neither the halos nor the agglomeration
    /// carry anything, so the answer must not be the single-process one.
    #[test]
    fn a_transport_that_never_delivers_does_not_reproduce_the_single_process_cycle() {
        let base = seed(8, 8, 8, Scene::Walled);
        let want = solve_single(&base, 2);
        let mut got = base.clone();
        project_pressure_multigrid_decomposed_over(
            &mut got,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            2,
            4,
            HaloSchedule::EverySweep,
            |bounds, nz, plane, halo, field| {
                Mute(local_slab_transport(bounds, nz, plane, halo, field))
            },
        );
        assert!(!bit_equal(&want, &got));
    }

    /// Teeth for the storage: a band with no halo cannot supply the layer an open
    /// `z` face reads, and the solve aborts instead of reading something stale.
    #[test]
    #[should_panic(expected = "this rank's band does not hold")]
    fn a_band_without_a_halo_aborts_instead_of_reading_a_stale_layer() {
        let mut g = seed(8, 8, 8, Scene::Open);
        project_pressure_multigrid_decomposed_over(
            &mut g,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            1,
            2,
            HaloSchedule::EverySweep,
            |bounds, nz, plane, _halo, field| local_slab_transport(bounds, nz, plane, 0, field),
        );
    }

    /// What each rank holds: owned layers plus one halo layer, clipped to the
    /// domain, at every distributed level — and so, with more than one rank, less
    /// than the whole level.
    #[test]
    fn a_rank_holds_its_owned_layers_and_one_halo_layer_and_no_more() {
        for ((nx, ny, nz), ranks) in cases() {
            let mut g = seed(nx, ny, nz, Scene::Walled);
            let residency = project_pressure_multigrid_decomposed_over(
                &mut g,
                fx(DT.0, DT.1),
                Fix128::from_int(RHO),
                1,
                ranks,
                HaloSchedule::EverySweep,
                local_slab_transport,
            );
            // the coarsening of nx, ny can end the hierarchy before nz does; the
            // levels the solve distributes are the ones `layout` keeps
            let mut nzs = vec![nz];
            while nzs.len() < residency.len() {
                let n = *nzs.last().expect("non-empty");
                nzs.push(if n > 1 { n / 2 } else { 1 });
            }
            let lay = layout(&nzs, ranks);
            for (l, per_rank) in residency.iter().enumerate() {
                for (r, &(lo, hi)) in per_rank.iter().enumerate() {
                    let (k0, k1) = lay.bounds[l][r];
                    let want = if k0 == k1 {
                        (k0, k0)
                    } else {
                        (k0.saturating_sub(1), (k1 + 1).min(nzs[l]))
                    };
                    assert_eq!(
                        (lo, hi),
                        want,
                        "{nx}x{ny}x{nz} over {ranks}, level {l}, rank {r}"
                    );
                }
            }
            if ranks > 1 && nz >= ranks {
                let held: usize = residency[0].iter().map(|&(lo, hi)| hi - lo).sum();
                assert!(
                    held < ranks * nz,
                    "{nx}x{ny}x{nz} over {ranks}: every rank still holds the whole field"
                );
                assert!(residency[0].iter().all(|&(lo, hi)| hi - lo < nz));
            }
        }
    }

    /// A grid the single-process solve refuses is refused here, bit for bit.
    #[test]
    fn inputs_the_single_process_solve_refuses_are_left_untouched() {
        let dt = fx(DT.0, DT.1);
        let rho = Fix128::from_int(RHO);
        for (nx, ny, nz, cycles, ranks) in [
            (6usize, 8usize, 8usize, 2u32, 2usize), // non power of two
            (8, 8, 8, 0, 2),                        // no cycles
            (8, 8, 8, 2, 0),                        // no ranks
        ] {
            let base = seed(nx, ny, nz, Scene::Open);
            let mut got = base.clone();
            project_pressure_multigrid_decomposed(
                &mut got,
                dt,
                rho,
                cycles,
                ranks,
                HaloSchedule::EverySweep,
            );
            assert!(
                bit_equal(&base, &got),
                "{nx}x{ny}x{nz} cycles {cycles} ranks {ranks}"
            );
        }
        let mut zero_dt = seed(8, 8, 8, Scene::Open);
        let before = zero_dt.clone();
        project_pressure_multigrid_decomposed(
            &mut zero_dt,
            Fix128::ZERO,
            rho,
            2,
            2,
            HaloSchedule::EverySweep,
        );
        assert!(bit_equal(&before, &zero_dt));
    }

    fn halving(nz0: usize) -> Vec<usize> {
        let mut v = vec![nz0];
        while *v.last().expect("non-empty") > 1 {
            let n = *v.last().expect("non-empty") / 2;
            v.push(n);
        }
        // x and y keep halving after z has stopped: two more levels with nz = 1.
        v.extend([1, 1]);
        v
    }

    /// The layout: bounds partition every distributed level exactly once, each
    /// level's bounds are the one above's divided by the layer factor (so the
    /// transfers are rank-local), and no rank is empty while a level has layers
    /// for all.
    #[test]
    fn the_layout_partitions_each_distributed_level_and_keeps_the_transfers_local() {
        for nz0 in [1usize, 2, 4, 8, 16, 64] {
            let nz = halving(nz0);
            for ranks in 1..=9usize {
                let lay = layout(&nz, ranks);
                for (l, level_bounds) in lay.bounds.iter().enumerate() {
                    assert_eq!(level_bounds.len(), ranks);
                    let mut covered = vec![0usize; nz[l]];
                    for &(k0, k1) in level_bounds {
                        assert!(k0 <= k1 && k1 <= nz[l]);
                        for c in &mut covered[k0..k1] {
                            *c += 1;
                        }
                    }
                    assert!(
                        covered.iter().all(|&c| c == 1),
                        "nz0 {nz0} ranks {ranks} level {l}: bounds {level_bounds:?} do not \
                         partition {} layers",
                        nz[l]
                    );
                    if l > 0 {
                        let factor = nz[l - 1] / nz[l];
                        for (r, &(k0, k1)) in level_bounds.iter().enumerate() {
                            let (f0, f1) = lay.bounds[l - 1][r];
                            assert_eq!((f0, f1), (k0 * factor, k1 * factor));
                        }
                    }
                }
                if nz[0] >= ranks {
                    assert!(
                        lay.bounds[lay.last].iter().all(|&(k0, k1)| k1 > k0),
                        "nz0 {nz0} ranks {ranks}: the last distributed level leaves a rank empty"
                    );
                    // The deepest such level: the next one has fewer layers than ranks.
                    assert!(lay.last + 1 == nz.len() || nz[lay.last + 1] < ranks);
                } else {
                    assert_eq!(lay.last, 0);
                }
            }
        }
    }

    /// Loopback streams between every pair of ranks: `links[a][b]` is rank `a`'s
    /// end of the stream to rank `b`. Reads time out, so a schedule that
    /// deadlocks fails instead of hanging.
    #[cfg(feature = "std")]
    fn socket_mesh(ranks: usize) -> Vec<Vec<Option<std::net::TcpStream>>> {
        use std::net::{TcpListener, TcpStream};
        use std::time::Duration;
        let mut links: Vec<Vec<Option<TcpStream>>> = (0..ranks)
            .map(|_| (0..ranks).map(|_| None).collect())
            .collect();
        let pairs = (0..ranks).flat_map(|a| (a + 1..ranks).map(move |b| (a, b)));
        for (a, b) in pairs {
            let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
            let addr = listener.local_addr().expect("address");
            let dialled = TcpStream::connect(addr).expect("dial");
            let (accepted, _) = listener.accept().expect("accept");
            for end in [&dialled, &accepted] {
                end.set_nodelay(true).expect("nodelay");
                end.set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("timeout");
            }
            links[a][b] = Some(accepted);
            links[b][a] = Some(dialled);
        }
        links
    }

    /// The factory a socket rank hands the solve: `(bounds, nz, plane, halo, field)`
    /// to this rank's transport.
    #[cfg(feature = "std")]
    type SocketMake<'a> = Box<
        dyn FnMut(
                &[(usize, usize)],
                usize,
                usize,
                usize,
                Option<&[Fix128]>,
            ) -> crate::eulerian_grid::SlabSocketTransport<std::net::TcpStream>
            + 'a,
    >;

    /// A transport factory for rank `my_rank`: a band over `bounds[my_rank]`
    /// filled from `field`, over a clone of each stream (every field shares the
    /// stream to a peer; deliveries are matched by position in the schedule).
    #[cfg(feature = "std")]
    fn socket_factory(my_rank: usize, links: &[Option<std::net::TcpStream>]) -> SocketMake<'_> {
        Box::new(move |bounds, nz, plane, halo, field| {
            let mut slab = SlabStorage::for_slab(plane, nz, bounds[my_rank], halo);
            if let Some(f) = field {
                let (lo, hi) = slab.resident();
                for k in lo..hi {
                    slab.layer_mut(k)
                        .expect("a layer inside the band")
                        .copy_from_slice(&f[k * plane..(k + 1) * plane]);
                }
            }
            let own = links
                .iter()
                .map(|l| l.as_ref().map(|s| s.try_clone().expect("clone stream")))
                .collect();
            crate::eulerian_grid::SlabSocketTransport::new(my_rank, plane, slab, own)
        })
    }

    /// Teeth: a driver that runs every rank cannot be handed a transport that
    /// holds one rank's band — the transport refuses to serve another's.
    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "the driver is not rank-local")]
    fn a_driver_that_runs_every_rank_is_refused_by_a_rank_local_transport() {
        let mut g = seed(8, 8, 8, Scene::Open);
        let links: Vec<Option<std::net::TcpStream>> = (0..2).map(|_| None).collect();
        project_pressure_multigrid_decomposed_over(
            &mut g,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            1,
            2,
            HaloSchedule::EverySweep,
            socket_factory(0, &links),
        );
    }

    /// The result a banded rank must reproduce, read off `want` (the
    /// single-process answer): its owned pressure layers and the face velocities
    /// it writes — X and Y faces of owned layers, Z faces `k0..k1` (and `nz` for
    /// the top rank).
    #[cfg(feature = "std")]
    fn banded_matches(
        want: &MacGrid,
        faces: &SlabFaces,
        pressure: &SlabStorage,
        what: &str,
    ) -> Result<(), String> {
        let (nx, ny, nz) = (want.nx, want.ny, want.nz);
        let (k0, k1) = faces.owned();
        for k in k0..k1 {
            let p = pressure.layer(k).expect("owned layer");
            if p != &want.pressure[k * nx * ny..(k + 1) * nx * ny] {
                return Err(format!("{what}: pressure layer {k}"));
            }
            let (u, _) = faces.u_layer(k);
            for j in 0..ny {
                for i in 0..=nx {
                    if u[i + (nx + 1) * j] != want.u[want.idx_u(i, j, k)] {
                        return Err(format!("{what}: u({i},{j},{k})"));
                    }
                }
            }
            let (v, _) = faces.v_layer(k);
            for j in 0..=ny {
                for i in 0..nx {
                    if v[i + nx * j] != want.v[want.idx_v(i, j, k)] {
                        return Err(format!("{what}: v({i},{j},{k})"));
                    }
                }
            }
            let top = if k1 == nz && k + 1 == k1 {
                Some(nz)
            } else {
                None
            };
            for kk in std::iter::once(k).chain(top) {
                let (w, _) = faces.w_layer(kk);
                for j in 0..ny {
                    for i in 0..nx {
                        if w[i + nx * j] != want.w[want.idx_w(i, j, kk)] {
                            return Err(format!("{what}: w({i},{j},{kk})"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run the banded driver as one thread per rank over loopback sockets and
    /// hand back each rank's faces and pressure band.
    #[cfg(feature = "std")]
    fn run_banded(
        base: &MacGrid,
        cycles: u32,
        ranks: usize,
        schedule: HaloSchedule,
    ) -> Vec<(SlabFaces, SlabStorage)> {
        let (nx, ny, nz) = (base.nx, base.ny, base.nz);
        let bounds = multigrid_slab_bounds(nx, ny, nz, ranks).expect("power-of-two grid");
        // The precondition of `SlabFaces`: conditions already imposed.
        let mut enforced = base.clone();
        enforced.enforce_face_boundaries();
        let mut mesh = socket_mesh(ranks);
        std::thread::scope(|sc| {
            let handles: Vec<_> = mesh
                .iter_mut()
                .enumerate()
                .map(|(rank, links)| {
                    let mut faces = SlabFaces::from_grid(&enforced, bounds[rank]);
                    let mut band = SlabStorage::for_slab(nx * ny, nz, bounds[rank], HALO);
                    let (lo, hi) = band.resident();
                    for k in lo..hi {
                        band.layer_mut(k)
                            .expect("resident")
                            .copy_from_slice(&base.pressure[k * nx * ny..(k + 1) * nx * ny]);
                    }
                    let links = &*links;
                    sc.spawn(move || {
                        project_pressure_multigrid_banded_on_rank(
                            &mut faces,
                            &mut band,
                            fx(DT.0, DT.1),
                            Fix128::from_int(RHO),
                            cycles,
                            ranks,
                            schedule,
                            rank,
                            socket_factory(rank, links),
                        );
                        (faces, band)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a rank panicked"))
                .collect()
        })
    }

    /// The oracle for stage 4a: a rank holding only its faces and its band, over
    /// sockets, reproduces the single-process answer to the bit — pressure on its
    /// owned layers and every face velocity it writes.
    #[cfg(feature = "std")]
    #[test]
    fn a_rank_built_from_its_band_alone_reproduces_the_single_process_cycle() {
        type Dims = (usize, usize, usize);
        let cases: &[(Dims, &[usize])] = &[
            ((8, 8, 8), &[2, 3, 4, 8]),
            ((16, 8, 4), &[2, 4, 8]),
            ((4, 4, 16), &[3, 5]),
            ((8, 8, 1), &[2]),
            ((16, 16, 16), &[4]),
        ];
        for scene in [Scene::Open, Scene::Walled] {
            for &((nx, ny, nz), rank_counts) in cases {
                for &ranks in rank_counts {
                    let base = seed(nx, ny, nz, scene);
                    let want = solve_single(&base, 2);
                    assert!(!bit_equal(&base, &want));
                    let got = run_banded(&base, 2, ranks, HaloSchedule::EverySweep);
                    for (rank, (faces, band)) in got.iter().enumerate() {
                        let what = format!("{nx}x{ny}x{nz} over {ranks}, {scene:?}, rank {rank}");
                        if let Err(e) = banded_matches(&want, faces, band, &what) {
                            panic!("{e}: not the single-process answer");
                        }
                    }
                }
            }
        }
    }

    /// Teeth: the stale-halo schedule moves the banded answer too.
    #[cfg(feature = "std")]
    #[test]
    fn a_stale_halo_does_not_reproduce_the_single_process_cycle_when_banded() {
        let base = seed(8, 8, 8, Scene::Walled);
        let want = solve_single(&base, 2);
        let got = run_banded(&base, 2, 4, HaloSchedule::EveryIteration);
        assert!(got.iter().enumerate().any(|(r, (f, b))| banded_matches(
            &want,
            f,
            b,
            &format!("rank {r}")
        )
        .is_err()));
    }

    /// Faces for the wrong layers are refused: the multigrid decomposition is not
    /// the Gauss-Seidel one.
    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "the multigrid decomposition gives")]
    fn faces_for_another_decomposition_are_refused() {
        let base = seed(8, 8, 8, Scene::Open);
        let mut enforced = base.clone();
        enforced.enforce_face_boundaries();
        let links: Vec<Option<std::net::TcpStream>> = (0..2).map(|_| None).collect();
        // rank 0 of 3 owns layers 0..2 under the Gauss-Seidel split of 8 layers
        // and 0..2 under the multigrid one too, so use rank 1: 2..5 against 2..4.
        let mut faces = SlabFaces::from_grid(&enforced, (2, 5));
        let mut band = SlabStorage::for_slab(64, 8, (2, 5), HALO);
        project_pressure_multigrid_banded_on_rank(
            &mut faces,
            &mut band,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            1,
            3,
            HaloSchedule::EverySweep,
            1,
            socket_factory(1, &links),
        );
    }

    /// The bounds the banded driver asks for are the layout's, and a
    /// non-power-of-two grid has none.
    #[test]
    fn the_banded_bounds_are_the_layouts_and_refuse_a_non_power_of_two() {
        assert_eq!(multigrid_slab_bounds(6, 8, 8, 2), None);
        assert_eq!(multigrid_slab_bounds(8, 8, 8, 0), None);
        let b = multigrid_slab_bounds(8, 8, 8, 3).expect("power of two");
        let nzs: Vec<usize> = level_dims(8, 8, 8).iter().map(|d| d.2).collect();
        assert_eq!(b, layout(&nzs, 3).bounds[0]);
        // 8 layers over 3 ranks is 2 / 3 / 3 in the Gauss-Seidel split and a
        // multiple of the 2-halving unit here.
        assert!(b.iter().all(|&(k0, k1)| k0 % 2 == 0 && k1 % 2 == 0));
    }

    // ------------------------------------------------------------------
    // One process per rank
    // ------------------------------------------------------------------

    /// An order-independent summary of the fields a solve produces: for each of
    /// pressure, `u`, `v`, `w`, the wrapping sum over entries of `value` times
    /// `index + 1`, once on the integer half and once on the fractional half. `Fix128`
    /// addition is a group operation, so the per-rank summaries add up to the
    /// whole-grid one whatever order the ranks finish in.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Fold([u64; 8]);

    impl Fold {
        fn add(&mut self, field: usize, index: usize, v: Fix128) {
            let w = index as u64 + 1;
            self.0[2 * field] = self.0[2 * field].wrapping_add((v.hi as u64).wrapping_mul(w));
            self.0[2 * field + 1] = self.0[2 * field + 1].wrapping_add(v.lo.wrapping_mul(w));
        }

        fn merge(&mut self, other: &Fold) {
            for (a, b) in self.0.iter_mut().zip(other.0) {
                *a = a.wrapping_add(b);
            }
        }
    }

    /// Every entry of a whole grid, in vector order.
    fn fold_grid(g: &MacGrid) -> Fold {
        let mut f = Fold::default();
        for (field, values) in [&g.pressure, &g.u, &g.v, &g.w].into_iter().enumerate() {
            for (index, &v) in values.iter().enumerate() {
                f.add(field, index, v);
            }
        }
        f
    }

    /// What one rank wrote: its owned pressure layers and the face velocities it
    /// writes (X and Y faces of owned layers, Z faces `k0..k1`, and `nz` for the
    /// top rank), at the indices the whole grid would give them.
    #[cfg(feature = "std")]
    fn fold_band(faces: &SlabFaces, pressure: &SlabStorage) -> Fold {
        let (nx, ny, nz) = (faces.nx, faces.ny, faces.nz);
        let (k0, k1) = faces.owned();
        let mut f = Fold::default();
        for k in k0..k1 {
            for (c, &v) in pressure.layer(k).expect("owned").iter().enumerate() {
                f.add(0, k * nx * ny + c, v);
            }
            for (t, &v) in faces.u_layer(k).0.iter().enumerate() {
                f.add(1, k * (nx + 1) * ny + t, v);
            }
            for (t, &v) in faces.v_layer(k).0.iter().enumerate() {
                f.add(2, k * nx * (ny + 1) + t, v);
            }
        }
        let top = (k1 == nz && k1 > k0).then_some(nz);
        for k in (k0..k1).chain(top) {
            for (t, &v) in faces.w_layer(k).0.iter().enumerate() {
                f.add(3, k * nx * ny + t, v);
            }
        }
        f
    }

    #[cfg(feature = "std")]
    const MGD_RANK: &str = "ALICE_PHYSICS_MGD_RANK";
    #[cfg(feature = "std")]
    const MGD_RANKS: &str = "ALICE_PHYSICS_MGD_RANKS";
    #[cfg(feature = "std")]
    const MGD_SIDE: &str = "ALICE_PHYSICS_MGD_SIDE";
    #[cfg(feature = "std")]
    const MGD_DEPTH: &str = "ALICE_PHYSICS_MGD_DEPTH";
    #[cfg(feature = "std")]
    const MGD_CYCLES: &str = "ALICE_PHYSICS_MGD_CYCLES";
    #[cfg(feature = "std")]
    const MGD_BROKER: &str = "ALICE_PHYSICS_MGD_BROKER";
    #[cfg(feature = "std")]
    const MGD_TIMEOUT: &str = "ALICE_PHYSICS_MGD_TIMEOUT_SECS";
    #[cfg(feature = "std")]
    const MGD_WORKER: &str =
        "eulerian_grid::multigrid_decomposed::tests::a_banded_process_rank_worker";

    #[cfg(feature = "std")]
    fn mgd_env<V: std::str::FromStr>(name: &str) -> V {
        std::env::var(name)
            .unwrap_or_else(|_| panic!("{name} from the parent process"))
            .parse()
            .unwrap_or_else(|_| panic!("{name} is not valid"))
    }

    /// Dial every lower rank and accept every higher one, so `links[r]` is the
    /// stream to rank `r`. Every listener is bound before its port reaches the
    /// table, so a dial completes into the backlog whether or not the peer has
    /// reached its accept loop.
    #[cfg(feature = "std")]
    fn join_mesh(
        my_rank: usize,
        ports: &[u16],
        listener: &std::net::TcpListener,
        timeout: std::time::Duration,
    ) -> Vec<Option<std::net::TcpStream>> {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        let ranks = ports.len();
        let mut links: Vec<Option<TcpStream>> = (0..ranks).map(|_| None).collect();
        for (lower, &port) in ports.iter().enumerate().take(my_rank) {
            let mut link = TcpStream::connect(("127.0.0.1", port)).expect("dial a lower rank");
            link.write_all(&(my_rank as u16).to_le_bytes())
                .expect("announce the dialling rank");
            links[lower] = Some(link);
        }
        for _ in my_rank + 1..ranks {
            let (mut link, _) = listener.accept().expect("accept a higher rank");
            let mut who = [0u8; 2];
            link.read_exact(&mut who).expect("learn who dialled");
            links[usize::from(u16::from_le_bytes(who))] = Some(link);
        }
        for link in links.iter().flatten() {
            link.set_nodelay(true).expect("nodelay");
            link.set_read_timeout(Some(timeout)).expect("read timeout");
            link.set_write_timeout(Some(timeout))
                .expect("write timeout");
        }
        links
    }

    /// A rank of the process-per-rank run: reached only when this test binary has
    /// been re-executed with `MGD_RANK` set; an ordinary `cargo test` finds it
    /// absent and returns.
    ///
    /// The rank builds **its own faces** straight from the seed formulas — no
    /// `MacGrid` exists in this process — runs the banded driver over sockets to
    /// the other ranks, and prints one line: its solve time and the fold of what
    /// it wrote.
    #[cfg(feature = "std")]
    #[test]
    fn a_banded_process_rank_worker() {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::time::{Duration, Instant};

        let Ok(rank) = std::env::var(MGD_RANK) else {
            return;
        };
        let rank: usize = rank.parse().expect("the rank is a number");
        let ranks: usize = mgd_env(MGD_RANKS);
        let n: usize = mgd_env(MGD_SIDE);
        let depth: usize = mgd_env(MGD_DEPTH);
        let cycles: u32 = mgd_env(MGD_CYCLES);
        let broker: u16 = mgd_env(MGD_BROKER);
        let timeout = Duration::from_secs(mgd_env(MGD_TIMEOUT));

        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind a loopback port");
        let port = listener.local_addr().expect("address").port();
        let mut to_parent = TcpStream::connect(("127.0.0.1", broker)).expect("reach the broker");
        let mut hello = [0u8; 4];
        hello[..2].copy_from_slice(&(rank as u16).to_le_bytes());
        hello[2..].copy_from_slice(&port.to_le_bytes());
        to_parent.write_all(&hello).expect("announce rank and port");
        let mut table = vec![0u8; 2 * ranks];
        to_parent.read_exact(&mut table).expect("the port table");
        let ports: Vec<u16> = table
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let links = join_mesh(rank, &ports, &listener, timeout);

        let bounds = multigrid_slab_bounds(n, n, depth, ranks).expect("power-of-two grid");
        let (k0, k1) = bounds[rank];
        let mut faces = SlabFaces::new(n, n, depth, Fix128::ONE, (k0, k1));
        for k in k0..k1 {
            let (u, _) = faces.u_layer_mut(k);
            for j in 0..n {
                for i in 0..=n {
                    u[i + (n + 1) * j] = seed_u(i, j, k);
                }
            }
            let (v, _) = faces.v_layer_mut(k);
            for j in 0..=n {
                for i in 0..n {
                    v[i + n * j] = seed_v(i, j, k);
                }
            }
        }
        if k1 > k0 {
            for k in k0..=k1 {
                let (w, _) = faces.w_layer_mut(k);
                for j in 0..n {
                    for i in 0..n {
                        w[i + n * j] = seed_w(i, j, k);
                    }
                }
            }
        }
        let mut band = SlabStorage::for_slab(n * n, depth, (k0, k1), HALO);

        let started = Instant::now();
        project_pressure_multigrid_banded_on_rank(
            &mut faces,
            &mut band,
            fx(DT.0, DT.1),
            Fix128::from_int(RHO),
            cycles,
            ranks,
            HaloSchedule::EverySweep,
            rank,
            socket_factory(rank, &links),
        );
        let secs = started.elapsed().as_secs_f64();
        let fold = fold_band(&faces, &band);
        let words: Vec<String> = fold.0.iter().map(|w| format!("{w:016x}")).collect();
        println!(
            "MGD-RESULT rank={rank} secs={secs:.3} fold={}",
            words.join(",")
        );
    }

    /// What a child rank reported.
    #[cfg(feature = "std")]
    #[derive(Debug)]
    struct RankReport {
        rank: usize,
        secs: f64,
        fold: Fold,
        /// Peak resident set, bytes, from `/usr/bin/time -l` when asked for.
        peak_rss: Option<u64>,
    }

    /// Solve `n³` across `ranks` re-executed processes of this test binary and
    /// return their reports. This process only brokers the ports and reaps the
    /// children; it holds no field.
    #[cfg(feature = "std")]
    fn run_processes(
        n: usize,
        depth: usize,
        ranks: usize,
        cycles: u32,
        timeout_secs: u64,
        measure_rss: bool,
    ) -> Vec<RankReport> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::process::{Command, Stdio};

        let broker = TcpListener::bind(("127.0.0.1", 0)).expect("bind the broker");
        let broker_port = broker.local_addr().expect("address").port();
        let exe = std::env::current_exe().expect("path of this test binary");
        let kids: Vec<_> = (0..ranks)
            .map(|rank| {
                let mut cmd = if measure_rss {
                    let mut c = Command::new("/usr/bin/time");
                    c.arg("-l").arg(&exe);
                    c
                } else {
                    Command::new(&exe)
                };
                cmd.args(["--exact", MGD_WORKER, "--test-threads=1", "--nocapture"])
                    .env(MGD_RANK, rank.to_string())
                    .env(MGD_RANKS, ranks.to_string())
                    .env(MGD_SIDE, n.to_string())
                    .env(MGD_DEPTH, depth.to_string())
                    .env(MGD_CYCLES, cycles.to_string())
                    .env(MGD_BROKER, broker_port.to_string())
                    .env(MGD_TIMEOUT, timeout_secs.to_string())
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                cmd.spawn()
                    .unwrap_or_else(|e| panic!("re-execute this test binary as rank {rank}: {e}"))
            })
            .collect();

        let mut brokered: Vec<Option<std::net::TcpStream>> = (0..ranks).map(|_| None).collect();
        let mut ports = vec![0u16; ranks];
        for _ in 0..ranks {
            let (mut link, _) = broker.accept().expect("a rank's announcement");
            let mut hello = [0u8; 4];
            link.read_exact(&mut hello).expect("announcement");
            let rank = usize::from(u16::from_le_bytes([hello[0], hello[1]]));
            ports[rank] = u16::from_le_bytes([hello[2], hello[3]]);
            brokered[rank] = Some(link);
        }
        let table: Vec<u8> = ports.iter().flat_map(|p| p.to_le_bytes()).collect();
        for link in brokered.iter_mut().flatten() {
            link.write_all(&table).expect("hand a rank the port table");
        }

        let mut reports = Vec::new();
        for (rank, kid) in kids.into_iter().enumerate() {
            let out = kid.wait_with_output().expect("reap a rank");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success(),
                "rank {rank} failed ({}):\n{stdout}\n{stderr}",
                out.status
            );
            // The harness prints `test <name> ... ` before the test's own output on
            // the same line, so the result is found by its tag, not by line start.
            let line = stdout
                .lines()
                .find_map(|l| l.find("MGD-RESULT").map(|at| &l[at..]))
                .unwrap_or_else(|| panic!("rank {rank} printed no result:\n{stdout}\n{stderr}"));
            let field = |key: &str| {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(key))
                    .unwrap_or_else(|| panic!("`{key}` missing from `{line}`"))
            };
            let mut fold = Fold::default();
            for (slot, word) in fold.0.iter_mut().zip(field("fold=").split(',')) {
                *slot = u64::from_str_radix(word, 16).expect("a hex word");
            }
            let peak_rss = stderr.lines().find_map(|l| {
                l.trim()
                    .strip_suffix("maximum resident set size")
                    .and_then(|v| v.trim().parse().ok())
            });
            reports.push(RankReport {
                rank,
                secs: field("secs=").parse().expect("seconds"),
                fold,
                peak_rss,
            });
        }
        reports
    }

    fn merged(reports: &[RankReport]) -> Fold {
        let mut total = Fold::default();
        for r in reports {
            total.merge(&r.fold);
        }
        total
    }

    /// The oracle across real process boundaries: every rank is its own process,
    /// holds no `MacGrid`, and all the ranks together wrote exactly what the
    /// single-process solve writes — pressure and every face velocity, summed with
    /// an index weight so a value in the wrong place is a different sum.
    #[cfg(feature = "std")]
    #[test]
    fn ranks_in_separate_processes_reproduce_the_single_process_cycle() {
        for &(n, ranks) in &[(16usize, 2usize), (16, 3), (32, 4)] {
            let want = fold_grid(&solve_single(&seed(n, n, n, Scene::Open), 2));
            let got = merged(&run_processes(n, n, ranks, 2, 120, false));
            assert_eq!(
                got, want,
                "{n}³ over {ranks} processes is not the single-process answer"
            );
        }
    }

    /// Teeth: the same run with one more cycle is not the two-cycle answer, so the
    /// comparison above can tell a different field from the right one.
    #[cfg(feature = "std")]
    #[test]
    fn the_process_comparison_tells_a_different_cycle_count_apart() {
        let n = 16;
        let two = fold_grid(&solve_single(&seed(n, n, n, Scene::Open), 2));
        let three = merged(&run_processes(n, n, 2, 3, 120, false));
        assert_ne!(three, two);
    }

    /// Manual measurement, not part of the suite: `n³` over `ranks` processes,
    /// timed and with each rank's peak resident set.
    ///
    /// ```text
    /// ALICE_PHYSICS_MGD_N=256 ALICE_PHYSICS_MGD_DEPTH=256 ALICE_PHYSICS_MGD_RANKS=8 ALICE_PHYSICS_MGD_CYCLES=6 \
    ///   ALICE_PHYSICS_MGD_REF=1 cargo test --release --lib \
    ///   banded_processes_timed -- --ignored --nocapture
    /// ```
    ///
    /// `ALICE_PHYSICS_MGD_REF=1` also solves the single-process reference (a whole
    /// `MacGrid`, so only for sizes that fit) and compares.
    #[cfg(feature = "std")]
    #[test]
    #[ignore = "manual measurement: set ALICE_PHYSICS_MGD_N / _RANKS / _CYCLES"]
    fn banded_processes_timed() {
        let get = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let n = get("ALICE_PHYSICS_MGD_N", 64);
        let depth = get("ALICE_PHYSICS_MGD_DEPTH", n);
        let ranks = get("ALICE_PHYSICS_MGD_RANKS", 4);
        let cycles = get("ALICE_PHYSICS_MGD_CYCLES", 6) as u32;
        let reference = std::env::var("ALICE_PHYSICS_MGD_REF").is_ok();
        let want = reference.then(|| {
            let started = std::time::Instant::now();
            let f = fold_grid(&solve_single(&seed(n, n, depth, Scene::Open), cycles));
            eprintln!(
                "reference (single process, {cycles} cycles): {:.1} s",
                started.elapsed().as_secs_f64()
            );
            f
        });
        let started = std::time::Instant::now();
        let reports = run_processes(n, depth, ranks, cycles, 7200, true);
        eprintln!(
            "{n}x{n}x{depth} = {} cells over {ranks} processes, {cycles} cycles: wall {:.1} s",
            n * n * depth,
            started.elapsed().as_secs_f64()
        );
        for r in &reports {
            eprintln!(
                "  rank {}: solve {:.1} s, peak RSS {}",
                r.rank,
                r.secs,
                r.peak_rss.map_or("n/a".to_string(), |b| format!(
                    "{:.2} GiB",
                    b as f64 / 1073741824.0
                ))
            );
        }
        if let Some(want) = want {
            assert_eq!(merged(&reports), want, "not the single-process answer");
            eprintln!("  bit-identical to the single-process solve");
        }
    }
}
