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
//! # What this stage does not do
//!
//! Every rank holds a full-length buffer per level, with everything beyond its
//! halo overwritten by a sentinel, so a stencil that reaches past the halo fails
//! as a mismatch rather than reading a value some other rank left behind. Slab
//! local storage and a driver that runs one rank per process are separate
//! changes, as they were for the Gauss-Seidel solve.

use super::{
    exchange_slab_halos, gather_slabs_to_root, mg_vcycle, poison_beyond_halo, poisson_rhs,
    slab_bounds, subtract_pressure_gradient, HaloSchedule, LocalTransport, MacGrid, MgLevel,
    PoissonMask, RankTransport, MG_COARSE_VISITS, MG_CORRECTION_SCALE_DEN, MG_CORRECTION_SCALE_NUM,
    MG_POST_SMOOTH, MG_PRE_SMOOTH,
};
use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Far outside any pressure this solve produces; see
/// [`super::project_pressure_decomposed`].
fn sentinel() -> Fix128 {
    Fix128::from_int(1_000_000)
}

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

/// [`super::project_pressure_multigrid`] run as `ranks` contiguous `z` slabs over
/// the in-process [`LocalTransport`].
///
/// Same early-return contract as the single-process solve (a bit-identical grid
/// for a non-power-of-two extent, a zero `dx` / `dt_s` / density, or
/// `cycles == 0`), and also for `ranks == 0`.
// ALLOW-UNWIRED: stage 1 of the distributed multigrid — the in-process driver the
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
        |nz, plane| LocalTransport::new(ranks, nz, plane),
    );
}

/// [`project_pressure_multigrid_decomposed`] over a caller-supplied
/// [`RankTransport`]. `make(nz, plane)` builds the transport for one field of `nz`
/// layers of `plane` values: the solve needs one per distributed level for the
/// pressure and two more at the last distributed level for the agglomeration.
pub(crate) fn project_pressure_multigrid_decomposed_over<T, F>(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    cycles: u32,
    ranks: usize,
    schedule: HaloSchedule,
    mut make: F,
) where
    T: RankTransport,
    F: FnMut(usize, usize) -> T,
{
    let pow2 = |n: usize| n.is_power_of_two();
    if cycles == 0
        || ranks == 0
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
    let nzs: Vec<usize> = levels.iter().map(|l| l.nz).collect();
    let Layout { last, bounds } = layout(&nzs, ranks);

    let mut solve = Decomposed {
        pressure: (0..=last)
            .map(|l| make(levels[l].nz, levels[l].nx * levels[l].ny))
            .collect(),
        rhs: (0..=last)
            .map(|l| vec![vec![Fix128::ZERO; levels[l].cells()]; ranks])
            .collect(),
        residual: vec![Vec::new(); ranks],
        gather: make(levels[last].nz, levels[last].nx * levels[last].ny),
        correction: make(levels[last].nz, levels[last].nx * levels[last].ny),
        coarse_p: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        coarse_rhs: levels[last + 1..]
            .iter()
            .map(|l| vec![Fix128::ZERO; l.cells()])
            .collect(),
        levels: &levels,
        invs: &invs,
        bounds: &bounds,
        last,
        ranks,
        schedule,
    };

    let n0 = levels[0].cells();
    let plane0 = levels[0].nx * levels[0].ny;
    for (r, &own) in bounds[0].iter().enumerate() {
        solve.rhs[0][r].copy_from_slice(&rhs0);
        let buf = solve.pressure[0].slab_mut(r);
        assert_eq!(
            buf.len(),
            n0,
            "transport handed rank {r} a slab of the wrong size"
        );
        buf.copy_from_slice(&grid.pressure);
        poison_beyond_halo(buf, levels[0].nz, plane0, own, sentinel());
    }

    for _ in 0..cycles {
        solve.cycle(0);
    }

    for (r, &(k0, k1)) in bounds[0].iter().enumerate() {
        let buf = solve.pressure[0].slab_mut(r);
        grid.pressure[k0 * plane0..k1 * plane0].copy_from_slice(&buf[k0 * plane0..k1 * plane0]);
    }

    let inv_dx = Fix128::ONE / grid.dx;
    subtract_pressure_gradient(grid, dt_s / density_kg_m3 * inv_dx);
}

/// The state of one decomposed solve.
struct Decomposed<'a, T: RankTransport> {
    levels: &'a [MgLevel],
    invs: &'a [Vec<Fix128>],
    bounds: &'a [Vec<(usize, usize)>],
    /// The last distributed level.
    last: usize,
    ranks: usize,
    schedule: HaloSchedule,
    /// One transport per distributed level, carrying that level's pressure.
    pressure: Vec<T>,
    /// `rhs[l][r]`: rank `r`'s right-hand side at level `l`; only its own layers
    /// are ever read, so it never travels.
    rhs: Vec<Vec<Vec<Fix128>>>,
    /// Scratch for a level's residual, one buffer per rank.
    residual: Vec<Vec<Fix128>>,
    /// The last level's residual, gathered to rank 0.
    gather: T,
    /// The coarse correction, sent from rank 0 to the owners.
    correction: T,
    /// Unknown and right-hand side of the agglomerated levels below `last`.
    coarse_p: Vec<Vec<Fix128>>,
    coarse_rhs: Vec<Vec<Fix128>>,
}

impl<T: RankTransport> Decomposed<'_, T> {
    fn exchange(&mut self, l: usize) {
        exchange_slab_halos(&mut self.pressure[l], &self.bounds[l], self.levels[l].nz);
    }

    /// Red-black Gauss-Seidel on level `l`, the colour order and the halo
    /// schedule of [`super::project_pressure_decomposed`].
    fn smooth(&mut self, l: usize, iterations: u32) {
        let level = &self.levels[l];
        for _ in 0..iterations {
            for colour in 0..2usize {
                for r in 0..self.ranks {
                    let (k0, k1) = self.bounds[l][r];
                    let buf = self.pressure[l].slab_mut(r);
                    let rhs = &self.rhs[l][r];
                    for k in k0..k1 {
                        for j in 0..level.ny {
                            for i in 0..level.nx {
                                if (i + j + k) % 2 != colour {
                                    continue;
                                }
                                let c = i + level.nx * (j + level.ny * k);
                                let nb = level.neighbour_sum(buf, i, j, k);
                                buf[c] = (nb - rhs[c]) * self.invs[l][c];
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
        for r in 0..self.ranks {
            let (k0, k1) = self.bounds[l][r];
            let mut res = vec![Fix128::ZERO; level.cells()];
            let buf = self.pressure[l].slab_mut(r);
            let rhs = &self.rhs[l][r];
            for k in k0..k1 {
                for j in 0..level.ny {
                    for i in 0..level.nx {
                        let c = i + level.nx * (j + level.ny * k);
                        let ap = buf[c] * Fix128::from_int(-level.degree(c))
                            + level.neighbour_sum(buf, i, j, k);
                        res[c] = rhs[c] - ap;
                    }
                }
            }
            self.residual[r] = res;
        }
    }

    /// Zero level `l`'s unknown on every rank: owned layers and halo are zero,
    /// everything beyond is the sentinel again.
    fn zero_pressure(&mut self, l: usize) {
        let level = &self.levels[l];
        let plane = level.nx * level.ny;
        for r in 0..self.ranks {
            let buf = self.pressure[l].slab_mut(r);
            buf.fill(Fix128::ZERO);
            poison_beyond_halo(buf, level.nz, plane, self.bounds[l][r], sentinel());
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
        for r in 0..self.ranks {
            let (k0, k1) = self.bounds[l][r];
            let rc = &mut self.rhs[l + 1][r];
            rc.fill(Fix128::ZERO);
            for k in k0..k1 {
                for j in 0..fine.ny {
                    for i in 0..fine.nx {
                        let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                        rc[ci] = rc[ci] + self.residual[r][i + fine.nx * (j + fine.ny * k)];
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
        for r in 0..self.ranks {
            let (k0, k1) = self.bounds[l][r];
            let e = self.pressure[l + 1].slab_mut(r).to_vec();
            let p = self.pressure[l].slab_mut(r);
            for k in k0..k1 {
                for j in 0..fine.ny {
                    for i in 0..fine.nx {
                        let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                        let c = i + fine.nx * (j + fine.ny * k);
                        p[c] = p[c] + e[ci] * scale;
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
            let buf = self.gather.slab_mut(r);
            buf.fill(sentinel());
            buf[k0 * plane..k1 * plane].copy_from_slice(&self.residual[r][k0 * plane..k1 * plane]);
        }
        gather_slabs_to_root(&mut self.gather, &self.bounds[l]);

        let coarse = &self.levels[l + 1];
        let (fx, fy, fz) = (
            level.nx / coarse.nx,
            level.ny / coarse.ny,
            level.nz / coarse.nz,
        );
        let res = self.gather.slab_mut(0).to_vec();
        let rc = &mut self.coarse_rhs[0];
        rc.fill(Fix128::ZERO);
        for k in 0..level.nz {
            for j in 0..level.ny {
                for i in 0..level.nx {
                    let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                    rc[ci] = rc[ci] + res[i + level.nx * (j + level.ny * k)];
                }
            }
        }
        self.coarse_p[0].fill(Fix128::ZERO);
        for _ in 0..MG_COARSE_VISITS {
            mg_vcycle(
                &self.levels[l + 1..],
                &self.invs[l + 1..],
                &mut self.coarse_p,
                &mut self.coarse_rhs,
            );
        }

        let scale = Fix128::from_ratio(MG_CORRECTION_SCALE_NUM, MG_CORRECTION_SCALE_DEN);
        let out = self.correction.slab_mut(0);
        for k in 0..level.nz {
            for j in 0..level.ny {
                for i in 0..level.nx {
                    let ci = i / fx + coarse.nx * (j / fy + coarse.ny * (k / fz));
                    out[i + level.nx * (j + level.ny * k)] = self.coarse_p[0][ci] * scale;
                }
            }
        }
        for r in 1..self.ranks {
            let (k0, k1) = self.bounds[l][r];
            for layer in k0..k1 {
                self.correction.deliver_layer(0, r, layer);
            }
        }
        for r in 0..self.ranks {
            let (k0, k1) = self.bounds[l][r];
            let corr = self.correction.slab_mut(r).to_vec();
            let p = self.pressure[l].slab_mut(r);
            for c in k0 * plane..k1 * plane {
                p[c] = p[c] + corr[c];
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
    struct Mute(LocalTransport);

    impl RankTransport for Mute {
        fn slab_mut(&mut self, rank: usize) -> &mut [Fix128] {
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
            |nz, plane| Mute(LocalTransport::new(4, nz, plane)),
        );
        assert!(!bit_equal(&want, &got));
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
