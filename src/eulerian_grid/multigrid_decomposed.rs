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
    PoissonMask, SlabFaces, SlabStencil, SlabStorage, SlabTransport, SweepWindow, MG_COARSE_VISITS,
    MG_CORRECTION_SCALE_DEN, MG_CORRECTION_SCALE_NUM, MG_POST_SMOOTH, MG_PRE_SMOOTH,
};
use crate::eulerian_grid::slab_bounds;
use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Halo width a 7-point stencil needs.
const HALO: usize = 1;

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
// ALLOW-UNWIRED: stage 2 of the distributed multigrid — the in-process driver the
// oracle runs; the rank-per-process driver (stage 3) is its caller.
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
/// Building the setup from a band, so that no rank ever sees the whole grid, goes
/// with the rank-per-process driver.
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
    solve_decomposed(
        grid,
        dt_s,
        density_kg_m3,
        cycles,
        ranks,
        schedule,
        None,
        make,
    )
}

/// One rank's half of [`project_pressure_multigrid_decomposed_over`], for
/// transports whose ranks do not share an address space.
///
/// The solve is the same code with the set of ranks it drives narrowed from every
/// rank to `my_rank`: that rank sweeps, restricts and prolongs its own layers and
/// asks its transports for its own band and no other. The exchange, the gather and
/// the correction are still walked in full on every rank, in the same order, and
/// the transport performs only the half it is party to; so deliveries are matched
/// by position in a sequence every rank agrees on. Rank 0 also runs the
/// agglomerated levels, and is the only rank that writes `grid` back.
///
/// Every rank must start from the same `grid`.
// ALLOW-UNWIRED: stage 3 of the distributed multigrid — the rank-local driver a
// process-per-rank harness calls; the oracle runs it over threads and sockets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_pressure_multigrid_decomposed_on_rank<T, F>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    my_rank: usize,
    make: F,
) -> Residency
where
    T: SlabTransport,
    F: FnMut(&[(usize, usize)], usize, usize, usize, Option<&[Fix128]>) -> T,
{
    if my_rank >= ranks {
        return Vec::new();
    }
    solve_decomposed(
        grid,
        dt_s,
        density_kg_m3,
        cycles,
        ranks,
        schedule,
        Some(my_rank),
        make,
    )
}

/// The decomposed solve for the ranks it drives: every rank when `only` is
/// `None`, otherwise just that one.
#[allow(clippy::too_many_arguments)]
fn solve_decomposed<T, F>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    only: Option<usize>,
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
    let active: Vec<usize> = only.map_or_else(|| (0..ranks).collect(), |r| vec![r]);
    let local: Vec<Vec<Local>> = (0..=last)
        .map(|l| {
            let p = plane(l);
            bounds[l]
                .iter()
                .enumerate()
                .map(|(r, &(k0, k1))| {
                    if !active.contains(&r) {
                        return Local {
                            cond: Vec::new(),
                            inv: Vec::new(),
                            rhs: Vec::new(),
                            res: Vec::new(),
                        };
                    }
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
    // Only rank 0 holds the whole field and so only rank 0 can take the gradient.
    if !active.contains(&0) {
        return residency;
    }
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
// ALLOW-UNWIRED: stage 4a of the distributed multigrid — read by the process-per-rank
// harness (stage 4b) and by the oracle, which build each rank's faces from it.
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

/// [`project_pressure_multigrid_decomposed_on_rank`] for a rank that holds **no
/// `MacGrid`**: its faces and its pressure band are all it has.
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
/// # Panics
///
/// When `faces` does not describe the layers the decomposition gives `my_rank`, or
/// `pressure` does not hold the band the first sweep reads.
// ALLOW-UNWIRED: stage 4a of the distributed multigrid — the banded driver the
// process-per-rank harness (stage 4b) calls; the oracle runs it over threads.
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
) where
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
        return;
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

    /// A divergent field in all three axes (so a wrong layer or axis is a
    /// different number), on an `nx × ny × nz` grid.
    fn seed(nx: usize, ny: usize, nz: usize, scene: Scene) -> MacGrid {
        let mut g = MacGrid::new(nx, ny, nz, Fix128::ONE);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let ix = g.idx_u(i, j, k);
                    g.u[ix] = fx(((i * 3 + j + 2 * k) % 7) as i64 - 3, 4);
                }
            }
        }
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let ix = g.idx_v(i, j, k);
                    g.v[ix] = fx(((i + j * 5 + k) % 5) as i64 - 2, 8);
                }
            }
        }
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let ix = g.idx_w(i, j, k);
                    g.w[ix] = fx(((i + 2 * j + k * 3) % 11) as i64 - 5, 4);
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

    /// The rank-local driver, run as one thread per rank over loopback sockets,
    /// lands on the single-process answer to the bit; rank 0 is the only one that
    /// writes its grid back, and the others leave theirs as they found it.
    #[cfg(feature = "std")]
    #[test]
    fn the_rank_local_driver_over_sockets_reproduces_the_single_process_cycle() {
        type Dims = (usize, usize, usize);
        let cases: &[(Dims, &[usize])] = &[
            ((8, 8, 8), &[2, 3, 4, 8]),
            ((16, 8, 4), &[2, 4, 8]),
            ((4, 4, 16), &[3, 5]),
            ((8, 8, 1), &[2]),
        ];
        for scene in [Scene::Open, Scene::Walled] {
            for &((nx, ny, nz), rank_counts) in cases {
                for &ranks in rank_counts {
                    let base = seed(nx, ny, nz, scene);
                    let want = solve_single(&base, 2);
                    assert!(!bit_equal(&base, &want));
                    let mut mesh = socket_mesh(ranks);
                    let results: Vec<MacGrid> = std::thread::scope(|sc| {
                        let handles: Vec<_> = mesh
                            .iter_mut()
                            .enumerate()
                            .map(|(rank, links)| {
                                let mut grid = base.clone();
                                let links = &*links;
                                sc.spawn(move || {
                                    project_pressure_multigrid_decomposed_on_rank(
                                        &mut grid,
                                        fx(DT.0, DT.1),
                                        Fix128::from_int(RHO),
                                        2,
                                        ranks,
                                        HaloSchedule::EverySweep,
                                        rank,
                                        socket_factory(rank, links),
                                    );
                                    grid
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| h.join().expect("a rank panicked"))
                            .collect()
                    });
                    assert!(
                        bit_equal(&want, &results[0]),
                        "{nx}x{ny}x{nz} over {ranks} ranks, {scene:?}: rank 0 over sockets is \
                         not the single-process answer"
                    );
                    // The solve enforces the face conditions on every rank's grid
                    // before anything else; past that, a non-root rank leaves its
                    // copy alone.
                    let mut untouched = base.clone();
                    untouched.enforce_face_boundaries();
                    for (rank, grid) in results.iter().enumerate().skip(1) {
                        assert!(
                            bit_equal(&untouched, grid),
                            "{nx}x{ny}x{nz} over {ranks} ranks: rank {rank} wrote its grid back"
                        );
                    }
                }
            }
        }
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
}
