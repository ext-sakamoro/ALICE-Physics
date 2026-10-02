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
    exchange_slab_halos_local, mg_vcycle, poisson_rhs, subtract_pressure_gradient, HaloSchedule,
    LocalSlabTransport, MacGrid, MgLevel, PoissonMask, SlabStorage, SlabTransport, SweepWindow,
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

    let mut solve = Decomposed {
        pressure: (0..=last)
            .map(|l| {
                let field = (l == 0).then_some(grid.pressure.as_slice());
                make(&bounds[l], levels[l].nz, plane(l), HALO, field)
            })
            .collect(),
        gather: make(&root_bounds, levels[last].nz, plane(last), 0, None),
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
    for (r, &(k0, k1)) in bounds[0].iter().enumerate() {
        let storage = solve.pressure[0].slab_mut(r);
        for k in k0..k1 {
            grid.pressure[k * plane0..(k + 1) * plane0]
                .copy_from_slice(storage.layer(k).expect(OWNED_LAYER_MISSING));
        }
    }
    let residency: Residency = solve
        .pressure
        .iter_mut()
        .map(|t| (0..ranks).map(|r| t.slab_mut(r).resident()).collect())
        .collect();

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
    residency
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
        for _ in 0..iterations {
            for colour in 0..2usize {
                for r in 0..self.ranks {
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
        for r in 0..self.ranks {
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
        for r in 0..self.ranks {
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
        for r in 0..self.ranks {
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
        for r in 0..self.ranks {
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
    fn agglomerated_correction(&mut self, l: usize) {
        let level = &self.levels[l];
        let plane = level.nx * level.ny;
        for r in 0..self.ranks {
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
        for (r, &(k0, k1)) in self.bounds[l].iter().enumerate().skip(1) {
            for layer in k0..k1 {
                self.correction.deliver_layer(0, r, layer);
            }
        }
        for r in 0..self.ranks {
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
}
