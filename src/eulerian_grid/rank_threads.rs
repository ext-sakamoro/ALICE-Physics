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
//! # Why the face conditions are imposed before the split
//!
//! Every solver imposes the face conditions on the whole grid first, as the
//! single-process solves do, and only then hands each rank its faces. The
//! slab-local form of that step (`enforce_slab_face_boundaries_on_rank`) is a
//! single ascending pass, while [`MacGrid::enforce_face_boundaries`] imposes
//! every non-outflow condition before any outflow face reads its inward
//! neighbour; the two disagree when an outflow face at the low end of an axis
//! sits next to an inflow or wall face. Routing the public entry through it
//! would publish that disagreement, so the in-process entry does not.

use super::{
    multigrid_decomposed::{
        multigrid_slab_bounds, project_pressure_multigrid_banded_on_rank, HALO,
    },
    project_pressure_decomposed_on_rank, project_pressure_slab_local_on_rank, slab_bounds,
    write_band_back, HaloSchedule, MacGrid, SlabFaces, SlabSocketTransport, SlabStorage,
    SocketTransport,
};
use crate::cfd_solver::{PressureSolver, PressureSolverError};
use crate::math::Fix128;

use std::io;
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
#[derive(Clone)]
pub(super) struct MemoryLink(Arc<Mutex<LinkEnd>>);

struct LinkEnd {
    tx: Sender<Vec<u8>>,
    rx: Receiver<Vec<u8>>,
    /// The message being read, and how far into it.
    pending: Vec<u8>,
    at: usize,
}

impl MemoryLink {
    /// The two ends of one stream.
    fn pair() -> (Self, Self) {
        let (tx_ab, rx_ab) = channel();
        let (tx_ba, rx_ba) = channel();
        let end = |tx, rx| {
            Self(Arc::new(Mutex::new(LinkEnd {
                tx,
                rx,
                pending: Vec::new(),
                at: 0,
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
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `links[a][b]` is rank `a`'s end of the stream to rank `b`; `links[a][a]` is
/// `None`.
fn mesh(ranks: usize) -> Vec<Vec<Option<MemoryLink>>> {
    let mut links: Vec<Vec<Option<MemoryLink>>> = (0..ranks)
        .map(|_| (0..ranks).map(|_| None).collect())
        .collect();
    let pairs = (0..ranks).flat_map(|a| (a + 1..ranks).map(move |b| (a, b)));
    for (a, b) in pairs {
        let (to_b, to_a) = MemoryLink::pair();
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
/// results in rank order. A rank that panicked is re-raised here, after every
/// rank has stopped (the others see their streams to it close).
fn run_ranks<R, F>(ranks: usize, rank_main: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize, Vec<Option<MemoryLink>>) -> R + Sync,
{
    let rank_main = &rank_main;
    std::thread::scope(|scope| {
        let handles: Vec<_> = mesh(ranks)
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
    })
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
            decomposed_gs(grid, dt_s, density_kg_m3, sweeps, ranks);
            Ok(())
        }
        PressureSolver::BandedGs { ranks, sweeps } => {
            banded_gs(grid, dt_s, density_kg_m3, sweeps, ranks);
            Ok(())
        }
        PressureSolver::DecomposedMultigrid { ranks, cycles } => {
            let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
            let Some(bounds) = multigrid_slab_bounds(nx, ny, nz, ranks) else {
                return Err(PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                    extents: (nx, ny, nz),
                });
            };
            banded_multigrid(grid, dt_s, density_kg_m3, cycles, &bounds);
            Ok(())
        }
        _ => Err(PressureSolverError::NotDecomposed),
    }
}

/// Every rank holds a copy of the grid and a full-length buffer; the pressure is
/// gathered to rank 0, whose grid is the answer.
fn decomposed_gs(grid: &mut MacGrid, dt_s: Fix128, density: Fix128, sweeps: u32, ranks: usize) {
    let start: &MacGrid = grid;
    let n = start.nx * start.ny * start.nz;
    let plane = start.nx * start.ny;
    let mut grids = run_ranks(ranks, |rank, links| {
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
}

/// Every rank holds its faces and its pressure band and nothing else.
fn banded_gs(grid: &mut MacGrid, dt_s: Fix128, density: Fix128, sweeps: u32, ranks: usize) {
    grid.enforce_face_boundaries();
    let (nz, plane) = (grid.nz, grid.nx * grid.ny);
    let bounds: Vec<(usize, usize)> = (0..ranks).map(|r| slab_bounds(nz, ranks, r)).collect();
    let start: &MacGrid = grid;
    let solved = run_ranks(ranks, |rank, links| {
        let mut faces = SlabFaces::from_grid(start, bounds[rank]);
        let band = band_from_field(&bounds, rank, nz, plane, 1, Some(&start.pressure));
        let mut transport = SlabSocketTransport::new(rank, plane, band, links);
        project_pressure_slab_local_on_rank(
            &mut faces,
            dt_s,
            density,
            sweeps,
            ranks,
            HaloSchedule::EverySweep,
            rank,
            &mut transport,
        );
        (faces, transport)
    });
    for (faces, transport) in &solved {
        write_band_back(grid, faces, transport.slab());
    }
}

/// Every rank holds its faces and its band at every distributed level; `bounds`
/// is the multigrid decomposition of the finest level.
fn banded_multigrid(
    grid: &mut MacGrid,
    dt_s: Fix128,
    density: Fix128,
    cycles: u32,
    bounds: &[(usize, usize)],
) {
    grid.enforce_face_boundaries();
    let (nz, plane) = (grid.nz, grid.nx * grid.ny);
    let ranks = bounds.len();
    let start: &MacGrid = grid;
    let solved = run_ranks(ranks, |rank, links| {
        let mut faces = SlabFaces::from_grid(start, bounds[rank]);
        let mut band = band_from_field(bounds, rank, nz, plane, HALO, Some(&start.pressure));
        project_pressure_multigrid_banded_on_rank(
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
        (faces, band)
    });
    for (faces, band) in &solved {
        write_band_back(grid, faces, band);
    }
}
