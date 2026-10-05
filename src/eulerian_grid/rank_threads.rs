//! The slab-decomposed pressure projection with every rank on its own thread.
//!
//! [`project_pressure_distributed`] is the public entry to the rank-local
//! drivers: each rank runs the same schedule as a rank in a separate process
//! would, holds its own state, and reaches its neighbours only through byte
//! messages. The ranks share an address space here, but no rank reads another
//! rank's memory — the transports are the cross-address-space ones
//! ([`SocketTransport`] / [`SlabSocketTransport`]), and the stream under them is
//! an in-memory channel ([`MemoryLink`]) instead of a socket. A driver that asked
//! a transport for another rank's slab therefore aborts here exactly as it would
//! across processes.
//!
//! # Face conditions are imposed rank by rank
//!
//! The slab solvers impose the face conditions after the split, each rank on
//! its own faces (`enforce_slab_face_boundaries_on_rank`), with one Z-face layer
//! handed up across each slab boundary over the same links the pressure halo
//! uses. The step takes two passes per axis, as
//! [`MacGrid::enforce_face_boundaries`] does — every non-outflow condition before
//! any outflow face reads its inward neighbour — so the faces it produces are
//! the single-process ones to the bit, including an outflow face at the low end
//! of an axis next to an inflow or a wall.

use super::{
    enforce_slab_face_boundaries_in_chain, enforce_slab_face_boundaries_on_rank,
    multigrid_decomposed::{
        multigrid_slab_bounds, project_pressure_multigrid_banded_on_rank, HALO,
    },
    project_pressure_decomposed_on_rank, project_pressure_slab_local_on_rank, slab_bounds,
    write_band_back, HaloSchedule, MacGrid, SlabFaceConditions, SlabFaces, SlabSocketTransport,
    SlabStorage, SocketTransport,
};
use crate::cfd_solver::{PressureSolver, PressureSolverError};
use crate::math::Fix128;

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};

/// One end of an in-memory byte stream between two ranks.
///
/// Writes never block (the channel is unbounded) and reads block until the peer
/// has written, which is the behaviour the lockstep transports assume of a
/// stream: for every delivery exactly one rank writes and exactly one reads. When
/// the peer's end is gone — the peer finished or panicked — a read that still
/// wants bytes sees end-of-stream, so a rank whose peer failed fails too instead
/// of waiting forever.
///
/// Clones share the stream, as `TcpStream::try_clone` does: the multigrid solve
/// builds several transports per rank over the same link to a peer, and
/// deliveries are matched by their position in the schedule, not by handle.
///
/// Every byte written through any link of one mesh is added to the mesh's
/// traffic counter, which is how the tests see that a step crossed the wire.
#[derive(Clone)]
pub(super) struct MemoryLink(Arc<Mutex<LinkEnd>>);

struct LinkEnd {
    tx: Sender<Vec<u8>>,
    rx: Receiver<Vec<u8>>,
    /// The message being read, and how far into it.
    pending: Vec<u8>,
    at: usize,
    /// Bytes written through every link of the mesh this end belongs to.
    traffic: Arc<AtomicUsize>,
}

impl MemoryLink {
    /// The two ends of one stream, counting what is written into `traffic`.
    fn pair(traffic: &Arc<AtomicUsize>) -> (Self, Self) {
        let (tx_ab, rx_ab) = channel();
        let (tx_ba, rx_ba) = channel();
        let end = |tx, rx| {
            Self(Arc::new(Mutex::new(LinkEnd {
                tx,
                rx,
                pending: Vec::new(),
                at: 0,
                traffic: Arc::clone(traffic),
            })))
        };
        (end(tx_ab, rx_ba), end(tx_ba, rx_ab))
    }
}

impl io::Read for MemoryLink {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut end = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if end.at == end.pending.len() {
            match end.rx.recv() {
                Ok(message) => {
                    end.pending = message;
                    end.at = 0;
                }
                Err(_) => return Ok(0), // the peer's end is gone: end of stream
            }
        }
        let n = buf.len().min(end.pending.len() - end.at);
        let at = end.at;
        buf[..n].copy_from_slice(&end.pending[at..at + n]);
        end.at += n;
        Ok(n)
    }
}

impl io::Write for MemoryLink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let end = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        end.tx
            .send(buf.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "the peer rank has gone"))?;
        end.traffic.fetch_add(buf.len(), Ordering::Relaxed);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `links[a][b]` is rank `a`'s end of the stream to rank `b`; `links[a][a]` is
/// `None`. Every link counts what is written through it into `traffic`.
fn mesh(ranks: usize, traffic: &Arc<AtomicUsize>) -> Vec<Vec<Option<MemoryLink>>> {
    let mut links: Vec<Vec<Option<MemoryLink>>> = (0..ranks)
        .map(|_| (0..ranks).map(|_| None).collect())
        .collect();
    for (a, b) in (0..ranks).flat_map(|a| (a + 1..ranks).map(move |b| (a, b))) {
        let (to_b, to_a) = MemoryLink::pair(traffic);
        links[a][b] = Some(to_b);
        links[b][a] = Some(to_a);
    }
    links
}

/// Rank `rank`'s band over `bounds[rank]` widened by `halo`, filled from `field`
/// over every layer it holds (the first sweep reads the halo before any
/// exchange).
fn band_from_field(
    bounds: &[(usize, usize)],
    rank: usize,
    nz: usize,
    plane: usize,
    halo: usize,
    field: Option<&[Fix128]>,
) -> SlabStorage {
    let mut band = SlabStorage::for_slab(plane, nz, bounds[rank], halo);
    if let Some(f) = field {
        let (lo, hi) = band.resident();
        for k in lo..hi {
            band.layer_mut(k)
                .expect("a layer inside the band this slab just reported")
                .copy_from_slice(&f[k * plane..(k + 1) * plane]);
        }
    }
    band
}

/// Run `rank_main(rank, links)` for every rank on its own thread and return the
/// results in rank order, with the bytes the ranks wrote to one another. A rank
/// that panicked is re-raised here, after every rank has stopped (the others see
/// their streams to it close).
fn run_ranks<R, F>(ranks: usize, rank_main: F) -> (Vec<R>, usize)
where
    R: Send,
    F: Fn(usize, Vec<Option<MemoryLink>>) -> R + Sync,
{
    let rank_main = &rank_main;
    let traffic = Arc::new(AtomicUsize::new(0));
    let out = std::thread::scope(|scope| {
        let handles: Vec<_> = mesh(ranks, &traffic)
            .into_iter()
            .enumerate()
            .map(|(rank, links)| scope.spawn(move || rank_main(rank, links)))
            .collect();
        let mut out = Vec::with_capacity(ranks);
        let mut failure = None;
        for handle in handles {
            match handle.join() {
                Ok(r) => out.push(r),
                Err(payload) => {
                    failure.get_or_insert(payload);
                }
            }
        }
        if let Some(payload) = failure {
            std::panic::resume_unwind(payload);
        }
        out
    });
    (out, traffic.load(Ordering::Relaxed))
}

/// The pressure projection of a slab-decomposed `solver`, run with one thread per
/// rank, each rank holding only what its process would hold in a distributed run
/// and exchanging halo layers with its neighbours as byte messages.
///
/// The answer is **bit-identical** to the single-process projection for every
/// rank count, including counts that do not divide `nz` and counts above it (a
/// surplus rank owns no layer and only walks the schedule):
///
/// | `solver` | each rank holds | identical to |
/// |---|---|---|
/// | [`PressureSolver::DecomposedGs`] | a whole copy of `grid`, sweeping its own layers; the result is gathered to rank 0 | [`super::project_pressure`] with `sweeps` |
/// | [`PressureSolver::BandedGs`] | its faces and its pressure band (owned layers plus one halo layer) | [`super::project_pressure`] with `sweeps` |
/// | [`PressureSolver::DecomposedMultigrid`] | its faces and its band at every distributed level; the coarsest levels are gathered to rank 0 | [`super::project_pressure_multigrid`] with `cycles` |
///
/// It costs `ranks` threads and `ranks · (ranks − 1) / 2` in-memory links for the
/// duration of the call, so a rank count far above `nz` buys nothing but
/// overhead. It needs a target with threads.
///
/// # Design
///
/// - One public entry point, everything else crate-internal. The transports,
///   slab bounds and per-rank solves stay `pub(crate)` so their shape can still
///   change when the rank transport is generalised; callers depend only on
///   "same answer as the single-process solver, for any rank count".
/// - Ranks talk only through byte messages on the same transports the
///   multi-process path uses (here over private in-memory links), and a rank
///   that asks for another rank's slab aborts. A rank therefore cannot read
///   state it would not have in a separate process, which is what makes the
///   bit identity a statement about the distributed algorithm and not about
///   shared memory.
/// - The solver is chosen through the existing [`PressureSolver`] enum
///   (`#[non_exhaustive]`), so a new decomposed solver is an added variant, not
///   a new function.
/// - The slab solvers impose the face conditions after the split, each rank on
///   its own faces, with one Z-face layer handed up across each slab boundary
///   (the in-process `BandedGs` path does the same). That step reproduces
///   [`MacGrid::enforce_face_boundaries`] to the bit, including an outflow face
///   at the low end of an axis next to an inflow or a wall, so no rank needs the
///   whole grid to start from enforced faces.
/// - The per-rank working set is available through
///   [`project_pressure_distributed_with_report`], an addition next to this
///   function rather than a change to its signature.
///
/// # Errors
///
/// The inputs [`crate::cfd_solver::CfdSolver::step_with_pressure_solver`] refuses,
/// with `grid` left untouched: [`PressureSolverError::ZeroTimeStep`],
/// [`PressureSolverError::ZeroDensity`], [`PressureSolverError::ZeroSpacing`],
/// [`PressureSolverError::ZeroIterations`], [`PressureSolverError::ZeroRanks`],
/// [`PressureSolverError::MultigridNeedsPowerOfTwoExtents`] for
/// `DecomposedMultigrid`, and [`PressureSolverError::NotDecomposed`] for a solver
/// that is not a slab decomposition.
///
/// # Panics
///
/// Re-raises a panic of any rank, which would be an internal invariant of the
/// decomposition failing.
pub fn project_pressure_distributed(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    solver: PressureSolver,
) -> Result<(), PressureSolverError> {
    run(grid, dt_s, density_kg_m3, solver).map(|_| ())
}

/// What each rank of a [`project_pressure_distributed_with_report`] call held.
///
/// A rank of a slab solver ([`PressureSolver::BandedGs`],
/// [`PressureSolver::DecomposedMultigrid`]) holds:
///
/// | part | bytes |
/// |---|---|
/// | face velocities | 16 per face it holds: the X- and Y-faces of its owned layers and the Z-faces of those layers plus the one above |
/// | face conditions | 3 per face it holds |
/// | pressure band | 16 per cell of its owned layers widened by the one-layer halo on each side, clipped to the grid |
/// | open-face mask | 6 per owned cell |
/// | inverse degrees | 16 per owned cell |
/// | right-hand side | 16 per owned cell |
///
/// read back from the containers' allocations, not recomputed from the
/// dimensions. A rank that owns no layer holds nothing. Not counted: the
/// transports' one-layer wire buffers and, for `DecomposedMultigrid`, the
/// coarser levels and the coarse solve rank 0 runs below them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedProjectionReport {
    /// Bytes of each rank's slab-local working set, in rank order (the parts in
    /// the table above). Empty for [`PressureSolver::DecomposedGs`], whose ranks
    /// each hold a whole copy of the grid rather than a slab.
    pub rank_bytes: Vec<usize>,
}

impl DistributedProjectionReport {
    /// The largest rank's working set: what one process of a distributed run
    /// has to fit. `None` when no rank held a slab.
    #[must_use]
    pub fn max_rank_bytes(&self) -> Option<usize> {
        self.rank_bytes.iter().copied().max()
    }

    /// The working sets of all ranks added up. `None` when no rank held a slab.
    #[must_use]
    pub fn total_bytes(&self) -> Option<usize> {
        (!self.rank_bytes.is_empty()).then(|| self.rank_bytes.iter().sum())
    }
}

/// [`project_pressure_distributed`], also returning what each rank held.
///
/// Same answer, same refusals (with `grid` left untouched), same panics; the
/// report is described on [`DistributedProjectionReport`].
///
/// # Errors
///
/// Those of [`project_pressure_distributed`].
///
/// # Panics
///
/// Those of [`project_pressure_distributed`].
pub fn project_pressure_distributed_with_report(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    solver: PressureSolver,
) -> Result<DistributedProjectionReport, PressureSolverError> {
    run(grid, dt_s, density_kg_m3, solver).map(|ran| ran.report)
}

/// One call of the driver: the report, and the bytes the ranks wrote to one
/// another (which the tests read).
struct Ran {
    report: DistributedProjectionReport,
    wire_bytes: usize,
}

fn run(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density_kg_m3: Fix128,
    solver: PressureSolver,
) -> Result<Ran, PressureSolverError> {
    if dt_s.is_zero() {
        return Err(PressureSolverError::ZeroTimeStep);
    }
    if density_kg_m3.is_zero() {
        return Err(PressureSolverError::ZeroDensity);
    }
    if grid.dx.is_zero() {
        return Err(PressureSolverError::ZeroSpacing);
    }
    match solver {
        PressureSolver::DecomposedGs { sweeps: 0, .. }
        | PressureSolver::BandedGs { sweeps: 0, .. }
        | PressureSolver::DecomposedMultigrid { cycles: 0, .. } => {
            Err(PressureSolverError::ZeroIterations)
        }
        PressureSolver::DecomposedGs { ranks: 0, .. }
        | PressureSolver::BandedGs { ranks: 0, .. }
        | PressureSolver::DecomposedMultigrid { ranks: 0, .. } => {
            Err(PressureSolverError::ZeroRanks)
        }
        PressureSolver::DecomposedGs { ranks, sweeps } => {
            Ok(decomposed_gs(grid, dt_s, density_kg_m3, sweeps, ranks))
        }
        PressureSolver::BandedGs { ranks, sweeps } => {
            Ok(banded_gs(grid, dt_s, density_kg_m3, sweeps, ranks))
        }
        PressureSolver::DecomposedMultigrid { ranks, cycles } => {
            let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
            let Some(bounds) = multigrid_slab_bounds(nx, ny, nz, ranks) else {
                return Err(PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                    extents: (nx, ny, nz),
                });
            };
            Ok(banded_multigrid(grid, dt_s, density_kg_m3, cycles, &bounds))
        }
        _ => Err(PressureSolverError::NotDecomposed),
    }
}

/// Every rank holds a copy of the grid and a full-length buffer; the pressure is
/// gathered to rank 0, whose grid is the answer.
fn decomposed_gs(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density: Fix128,
    sweeps: u32,
    ranks: usize,
) -> Ran {
    let start: &MacGrid = grid;
    let n = start.nx * start.ny * start.nz;
    let plane = start.nx * start.ny;
    let (mut grids, wire_bytes) = run_ranks(ranks, |rank, links| {
        let mut own = start.clone();
        let mut transport = SocketTransport::new(rank, n, plane, links);
        project_pressure_decomposed_on_rank(
            &mut own,
            dt_s,
            density,
            sweeps,
            ranks,
            HaloSchedule::EverySweep,
            rank,
            &mut transport,
        );
        own
    });
    *grid = grids.swap_remove(0);
    Ran {
        report: DistributedProjectionReport {
            rank_bytes: Vec::new(),
        },
        wire_bytes,
    }
}

/// Every rank holds its faces and its pressure band and nothing else, and
/// imposes the face conditions on its own faces before the solve.
fn banded_gs(grid: &mut MacGrid, dt_s: Fix128, density: Fix128, sweeps: u32, ranks: usize) -> Ran {
    let (nz, plane) = (grid.nz, grid.nx * grid.ny);
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    let start: &MacGrid = grid;
    let (solved, wire_bytes) = run_ranks(ranks, |rank, links| {
        let mut faces = SlabFaces::from_grid(start, bounds[rank]);
        let conditions = SlabFaceConditions::from_grid(start, bounds[rank]);
        let band = band_from_field(&bounds, rank, nz, plane, 1, Some(&start.pressure));
        let mut transport = SlabSocketTransport::new(rank, plane, band, links);
        enforce_slab_face_boundaries_on_rank(&mut faces, &conditions, ranks, rank, &mut transport);
        let stencil = project_pressure_slab_local_on_rank(
            &mut faces,
            dt_s,
            density,
            sweeps,
            ranks,
            HaloSchedule::EverySweep,
            rank,
            &mut transport,
        );
        let held = faces.bytes().add(transport.slab().bytes()).add(stencil);
        (faces, transport, held)
    });
    for (faces, transport, _) in &solved {
        write_band_back(grid, faces, transport.slab());
    }
    Ran {
        report: DistributedProjectionReport {
            rank_bytes: solved.iter().map(|(_, _, held)| held.total()).collect(),
        },
        wire_bytes,
    }
}

/// Every rank holds its faces and its band at every distributed level, and
/// imposes the face conditions on its own faces before the solve; `bounds` is
/// the multigrid decomposition of the finest level.
fn banded_multigrid(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density: Fix128,
    cycles: u32,
    bounds: &[(usize, usize)],
) -> Ran {
    let (nz, plane) = (grid.nz, grid.nx * grid.ny);
    let ranks = bounds.len();
    let start: &MacGrid = grid;
    let (solved, wire_bytes) = run_ranks(ranks, |rank, links| {
        let mut faces = SlabFaces::from_grid(start, bounds[rank]);
        let conditions = SlabFaceConditions::from_grid(start, bounds[rank]);
        // The face step needs a channel and no band: a transport holding none
        // carries the one Z-face layer over the same links.
        let (k0, _) = bounds[rank];
        let mut channel = SlabSocketTransport::new(
            rank,
            plane,
            SlabStorage::for_slab(plane, nz, (k0, k0), 0),
            links.clone(),
        );
        enforce_slab_face_boundaries_in_chain(&mut faces, &conditions, bounds, rank, &mut channel);
        let mut band = band_from_field(bounds, rank, nz, plane, HALO, Some(&start.pressure));
        let stencil = project_pressure_multigrid_banded_on_rank(
            &mut faces,
            &mut band,
            dt_s,
            density,
            cycles,
            ranks,
            HaloSchedule::EverySweep,
            rank,
            |level_bounds: &[(usize, usize)], nz, plane, halo, field: Option<&[Fix128]>| {
                let slab = band_from_field(level_bounds, rank, nz, plane, halo, field);
                SlabSocketTransport::new(rank, plane, slab, links.clone())
            },
        );
        let held = faces.bytes().add(band.bytes()).add(stencil);
        (faces, band, held)
    });
    for (faces, band, _) in &solved {
        write_band_back(grid, faces, band);
    }
    Ran {
        report: DistributedProjectionReport {
            rank_bytes: solved.iter().map(|(_, _, held)| held.total()).collect(),
        },
        wire_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eulerian_grid::FaceBc;

    /// A grid whose faces all differ, with outflows at both ends of every axis
    /// and an inflow next to each low-end one.
    fn outflow_grid(nx: usize, ny: usize, nz: usize) -> MacGrid {
        let mut g = MacGrid::new(nx, ny, nz, Fix128::from_ratio(1, 8));
        for (c, x) in g.u.iter_mut().enumerate() {
            *x = Fix128::from_ratio(((c * 7) % 13) as i64 - 6, 4);
        }
        for (c, x) in g.v.iter_mut().enumerate() {
            *x = Fix128::from_ratio(((c * 5) % 11) as i64 - 5, 4);
        }
        for (c, x) in g.w.iter_mut().enumerate() {
            *x = Fix128::from_ratio(((c * 3) % 17) as i64 - 8, 4);
        }
        let inflow = FaceBc::Inflow {
            normal_velocity: Fix128::from_ratio(3, 4),
        };
        for k in 0..nz {
            for j in 0..ny {
                g.set_u_bc(0, j, k, FaceBc::Outflow);
                g.set_u_bc(1, j, k, inflow);
                g.set_u_bc(nx, j, k, FaceBc::Outflow);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                g.set_w_bc(i, j, 0, FaceBc::Outflow);
                g.set_w_bc(i, j, 1, inflow);
                g.set_w_bc(i, j, nz, FaceBc::Outflow);
            }
        }
        g
    }

    /// The slab solvers impose the face conditions on each rank's own faces and
    /// hand one Z-face layer up across each slab boundary: with a zero time step
    /// (which the public entry refuses, and which leaves the solve itself a
    /// no-op) the only bytes on the wire are those layers, one per pair of
    /// neighbouring ranks that own something, and the faces written back are the
    /// single-process enforcement's.
    ///
    /// This is what tells the rank-local step apart from imposing the conditions
    /// on the whole grid before the split: both give the same faces, which is
    /// the point of the fix, so only the crossing distinguishes them.
    #[test]
    fn the_slab_solvers_impose_the_face_conditions_rank_by_rank() {
        let (nx, ny) = (4usize, 4usize);
        let plane_bytes = nx * ny * 16;
        for nz in [8usize, 16] {
            let base = outflow_grid(nx, ny, nz);
            let mut want = base.clone();
            want.enforce_face_boundaries();
            assert_ne!(want.u, base.u, "the conditions must change the faces");
            for ranks in [1usize, 2, 3, 4, 8, nz + 3] {
                let owning = (0..ranks)
                    .filter(|&r| {
                        let (k0, k1) = slab_bounds(nz, ranks, r);
                        k0 != k1
                    })
                    .count();
                let mut g = base.clone();
                let ran = banded_gs(&mut g, Fix128::ZERO, Fix128::ONE, 1, ranks);
                assert_eq!(
                    ran.wire_bytes,
                    (owning - 1) * plane_bytes,
                    "BandedGs nz {nz} over {ranks} ranks",
                );
                assert_eq!((&g.u, &g.v, &g.w), (&want.u, &want.v, &want.w));
                assert_eq!(g.pressure, base.pressure);

                let Some(bounds) = multigrid_slab_bounds(nx, ny, nz, ranks) else {
                    panic!("a power-of-two grid has a multigrid decomposition");
                };
                let owning = bounds.iter().filter(|&&(k0, k1)| k0 != k1).count();
                let mut g = base.clone();
                let ran = banded_multigrid(&mut g, Fix128::ZERO, Fix128::ONE, 1, &bounds);
                assert_eq!(
                    ran.wire_bytes,
                    (owning - 1) * plane_bytes,
                    "DecomposedMultigrid nz {nz} over {ranks} ranks",
                );
                assert_eq!((&g.u, &g.v, &g.w), (&want.u, &want.v, &want.w));
            }
        }
    }
}
