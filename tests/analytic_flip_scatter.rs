//! Oracles for the selectable particle-to-grid stencil:
//! `eulerian_grid::ParticleScatter`, `p2g_normalized_with` and
//! `CfdSolver::step_flip_with`.
//!
//! The nearest stencil deposits a particle on the six faces of the cell that
//! contains it with weight 1/2 each, and the normalisation turns that into a
//! closed form that does not depend on where inside the cell the particle is:
//!
//! ```text
//! face (i, j, k) along x  =  mean of v_x over the particles in cells
//!                            (i − 1, j, k) and (i, j, k)
//! ```
//!
//! With `Σ ½·v ÷ Σ ½` taken exactly (the accumulator keeps the full product)
//! the 1/2 cancels, so the face is `trunc(Σ v_raw / count)` on the raw
//! `Fix128` integers. Every expected value below is computed from that
//! definition on the test's own bookkeeping of which cell holds which
//! particle, never from the code under test.
//!
//! The trilinear stencil puts a particle at a cell centre on exactly the same
//! two faces with the same two halves, so at cell centres the stencils agree
//! bit for bit; off centre they do not, and the pair of particles in
//! `off_centre_particles_separate_the_two_stencils` has both answers in
//! closed form. A uniform velocity is reproduced bit for bit by both, so a
//! uniform cloud makes `step_flip_with` independent of the stencil, which is
//! how the solver-level wiring is pinned without a second physics oracle.
//!
//! # Degenerate input
//!
//! The nearest stencil never clamps a stray into the domain: negative
//! coordinates, coordinates beyond the far face and coordinates whose
//! quotient by `dx` does not fit `Fix128` deposit nothing, and a particle
//! exactly on the far face belongs to the last cell. The refusals of
//! `step_flip` hold unchanged for `step_flip_with`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{p2g_normalized, p2g_normalized_with, MacGrid, ParticleScatter};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

type Particle = (Vec3Fix, Vec3Fix);

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn raw(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

fn from_raw_i128(r: i128) -> Fix128 {
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

fn faces(g: &MacGrid) -> (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>) {
    (g.u.clone(), g.v.clone(), g.w.clone())
}

/// A deterministic dyadic velocity for cell `(i, j, k)`: distinct between
/// neighbours along every axis, with halves and quarters so a wrong weight
/// cannot hide behind integers.
fn cell_velocity(i: usize, j: usize, k: usize) -> Vec3Fix {
    let (i, j, k) = (i as i64, j as i64, k as i64);
    v3(
        q(4 * i - 2 * j + k, 4),
        q(-3 * i + j * j - 2 * k, 2),
        q(i * j - 5 * k + 7, 8),
    )
}

/// Mean over the given particle velocities on the raw integers, truncated
/// toward zero, as the exact accumulator defines it; `None` if there are none.
fn mean_raw(vals: &[Fix128]) -> Option<Fix128> {
    if vals.is_empty() {
        return None;
    }
    let sum: i128 = vals.iter().map(|&v| raw(v)).sum();
    Some(from_raw_i128(sum / vals.len() as i128))
}

/// Face means of one axis, `None` where no particle reaches the face.
type FaceMeans = Vec<Option<Fix128>>;

/// The nearest-stencil closed form for every face of `grid`, from the
/// test's own cell bookkeeping: `cells[i][j][k]` lists the velocities of the
/// particles the test put into cell `(i, j, k)`.
fn nearest_closed_form(
    grid: &MacGrid,
    cells: &[Vec<Vec<Vec<Vec3Fix>>>],
) -> (FaceMeans, FaceMeans, FaceMeans) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    // The mean over the particles of the two cells either side of a face,
    // `None` for a cell that does not exist.
    let across = |a: Option<(usize, usize, usize)>,
                  b: Option<(usize, usize, usize)>,
                  pick: fn(&Vec3Fix) -> Fix128| {
        let mut vals = Vec::new();
        for (i, j, k) in a.into_iter().chain(b) {
            vals.extend(cells[i][j][k].iter().map(pick));
        }
        mean_raw(&vals)
    };
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut w = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let lo = (i > 0).then(|| (i - 1, j, k));
                let hi = (i < nx).then_some((i, j, k));
                u.push(across(lo, hi, |p| p.x));
            }
        }
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                let lo = (j > 0).then(|| (i, j - 1, k));
                let hi = (j < ny).then_some((i, j, k));
                v.push(across(lo, hi, |p| p.y));
            }
        }
    }
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                let lo = (k > 0).then(|| (i, j, k - 1));
                let hi = (k < nz).then_some((i, j, k));
                w.push(across(lo, hi, |p| p.z));
            }
        }
    }
    (u, v, w)
}

fn assert_faces_match(got: &[Fix128], want: &[Option<Fix128>], untouched: &[Fix128], axis: &str) {
    assert_eq!(got.len(), want.len());
    for (ix, (g, w)) in got.iter().zip(want).enumerate() {
        match w {
            Some(w) => assert_eq!(g, w, "{axis} face {ix}: nearest mean"),
            None => assert_eq!(
                g, &untouched[ix],
                "{axis} face {ix}: no particle reaches it, must keep its value"
            ),
        }
    }
}

fn empty_cells(nx: usize, ny: usize, nz: usize) -> Vec<Vec<Vec<Vec<Vec3Fix>>>> {
    vec![vec![vec![Vec::new(); nz]; ny]; nx]
}

// ===========================================================================
// Oracle 1 — cell centres: the two stencils coincide bit for bit, and both
// equal the closed form
// ===========================================================================

#[test]
fn cell_centre_particles_give_bit_identical_faces_under_both_stencils() {
    let n = 4usize;
    let dx = q(1, 4);
    let mut cells = empty_cells(n, n, n);
    let mut ps: Vec<Particle> = Vec::new();
    let mut put = |i: usize, j: usize, k: usize, vel: Vec3Fix| cells[i][j][k].push(vel);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let c = |a: usize| (int(a as i64) + q(1, 2)) * dx;
                let vel = cell_velocity(i, j, k);
                ps.push((v3(c(i), c(j), c(k)), vel));
                put(i, j, k, vel);
            }
        }
    }
    let mut tri = MacGrid::new(n, n, n, dx);
    let mut near = MacGrid::new(n, n, n, dx);
    p2g_normalized_with(&mut tri, &ps, ParticleScatter::Trilinear);
    p2g_normalized_with(&mut near, &ps, ParticleScatter::Nearest);
    assert_eq!(
        faces(&tri),
        faces(&near),
        "cell centres: stencils must agree"
    );

    let (eu, ev, ew) = nearest_closed_form(&near, &cells);
    let zero = vec![Fix128::ZERO; near.u.len().max(near.v.len()).max(near.w.len())];
    assert_faces_match(&near.u, &eu, &zero, "u");
    assert_faces_match(&near.v, &ev, &zero, "v");
    assert_faces_match(&near.w, &ew, &zero, "w");
    // Not vacuous: an interior face averages two different cells.
    assert_ne!(near.u(1, 0, 0), near.u(2, 0, 0));
}

// ===========================================================================
// Oracle 2 — off-centre particles, several per cell: the face is the mean
// over the two cells sharing it, independent of sub-cell position
// ===========================================================================

#[test]
fn nearest_gives_the_mean_over_the_two_cells_sharing_a_face_wherever_the_particles_sit() {
    let (nx, ny, nz) = (3usize, 2usize, 2usize);
    let dx = q(1, 4);
    // Three particles per cell at sub-cell offsets that are nowhere near the
    // centre, with their own velocities (not the cell's), so the trilinear
    // weights would differ from 1/2 everywhere.
    let offsets = [
        (q(1, 16), q(13, 16), q(5, 32)),
        (q(27, 32), q(3, 16), q(29, 32)),
        (q(7, 8), q(7, 8), q(1, 64)),
    ];
    let mut cells = empty_cells(nx, ny, nz);
    let mut ps: Vec<Particle> = Vec::new();
    let mut put = |i: usize, j: usize, k: usize, vel: Vec3Fix| cells[i][j][k].push(vel);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for (m, (ox, oy, oz)) in offsets.iter().enumerate() {
                    let pos = v3(
                        (int(i as i64) + *ox) * dx,
                        (int(j as i64) + *oy) * dx,
                        (int(k as i64) + *oz) * dx,
                    );
                    let base = cell_velocity(i, j, k);
                    let m = m as i64;
                    let vel = v3(base.x + q(m, 3), base.y - q(5 * m, 7), base.z + int(m * m));
                    ps.push((pos, vel));
                    put(i, j, k, vel);
                }
            }
        }
    }
    let mut near = MacGrid::new(nx, ny, nz, dx);
    let sentinel = q(-99, 1);
    near.u.fill(sentinel);
    near.v.fill(sentinel);
    near.w.fill(sentinel);
    p2g_normalized_with(&mut near, &ps, ParticleScatter::Nearest);
    let (eu, ev, ew) = nearest_closed_form(&near, &cells);
    let keep = vec![sentinel; near.u.len().max(near.v.len()).max(near.w.len())];
    assert_faces_match(&near.u, &eu, &keep, "u");
    assert_faces_match(&near.v, &ev, &keep, "v");
    assert_faces_match(&near.w, &ew, &keep, "w");

    // The same cloud with every particle moved to its cell centre gives the
    // same faces: position inside the cell is invisible to this stencil.
    let centred: Vec<Particle> = ps
        .iter()
        .map(|&(p, v)| {
            let c = |x: Fix128| (Fix128::from_int((x / dx).hi) + q(1, 2)) * dx;
            (v3(c(p.x), c(p.y), c(p.z)), v)
        })
        .collect();
    let mut again = MacGrid::new(nx, ny, nz, dx);
    p2g_normalized_with(&mut again, &centred, ParticleScatter::Nearest);
    assert_eq!(faces(&again), faces(&near), "sub-cell position leaked in");

    // And the trilinear stencil does see the positions.
    let mut tri = MacGrid::new(nx, ny, nz, dx);
    p2g_normalized_with(&mut tri, &ps, ParticleScatter::Trilinear);
    assert_ne!(tri.u, near.u, "off-centre cloud: stencils must differ");
}

// ===========================================================================
// Oracle 3 — two particles, both answers in closed form
// ===========================================================================

/// Cell `(0, 0, 0)` of a `2 × 1 × 1` grid, `dx = 1`. Particle A at `x = 1/4`
/// with `v_x = 1`, particle B at `x = 3/4` with `v_x = 3`, both at
/// `y = z = 1/2` (on the u-face stagger, so the y / z weights are 1 and 0).
///
/// Nearest: both faces hold the mean `(1 + 3) / 2 = 2`.
/// Trilinear: face 0 gets `3/4 · 1 + 1/4 · 3 = 3/2` over weight `1`, face 1
/// gets `1/4 · 1 + 3/4 · 3 = 5/2` over weight `1`.
#[test]
fn off_centre_particles_separate_the_two_stencils() {
    let ps = [
        (v3(q(1, 4), q(1, 2), q(1, 2)), v3(int(1), int(2), int(-4))),
        (v3(q(3, 4), q(1, 2), q(1, 2)), v3(int(3), int(2), int(-4))),
    ];
    let mut near = MacGrid::new(2, 1, 1, Fix128::ONE);
    let mut tri = MacGrid::new(2, 1, 1, Fix128::ONE);
    p2g_normalized_with(&mut near, &ps, ParticleScatter::Nearest);
    p2g_normalized_with(&mut tri, &ps, ParticleScatter::Trilinear);
    assert_eq!(near.u(0, 0, 0), int(2));
    assert_eq!(near.u(1, 0, 0), int(2));
    assert_eq!(tri.u(0, 0, 0), q(3, 2));
    assert_eq!(tri.u(1, 0, 0), q(5, 2));
    // The components the two particles share are uniform, so every face a
    // stencil reaches holds that value; the stencils differ in *which* faces
    // they reach. Nearest touches only the faces of cell 0; trilinear, with B
    // a quarter cell past the centre, also lands on the Y and Z faces of
    // cell 1 (weight 1/4, over the stagger of those faces), which then hold
    // the uniform value too.
    for j in 0..=1 {
        assert_eq!(near.v(0, j, 0), int(2));
        assert_eq!(
            near.v(1, j, 0),
            Fix128::ZERO,
            "nearest must not reach cell 1"
        );
        assert_eq!(tri.v(0, j, 0), int(2));
        assert_eq!(
            tri.v(1, j, 0),
            int(2),
            "trilinear reaches cell 1 from x = 3/4"
        );
    }
    for k in 0..=1 {
        assert_eq!(near.w(0, 0, k), int(-4));
        assert_eq!(near.w(1, 0, k), Fix128::ZERO);
        assert_eq!(tri.w(0, 0, k), int(-4));
        assert_eq!(tri.w(1, 0, k), int(-4));
    }
}

// ===========================================================================
// Oracle 4 — a uniform velocity is reproduced bit for bit, any count, any
// position, including the far face
// ===========================================================================

#[test]
fn a_uniform_velocity_is_reproduced_bit_for_bit_by_nearest() {
    let n = 3usize;
    let dx = q(1, 8);
    let u0 = v3(q(7, 3), int(-5), q(-1, 9));
    let mut ps: Vec<Particle> = Vec::new();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for _ in 0..5 {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let f = |s: u64| Fix128::from_raw(0, s); // in [0, 1)
                    let pos = v3(
                        (int(i as i64) + f(seed)) * dx,
                        (int(j as i64) + f(seed.rotate_left(21))) * dx,
                        (int(k as i64) + f(seed.rotate_left(42))) * dx,
                    );
                    ps.push((pos, u0));
                }
            }
        }
    }
    // Particles exactly on the far faces belong to the last cells.
    let far = int(n as i64) * dx;
    ps.push((v3(far, far, far), u0));
    ps.push((v3(far, q(1, 16), q(1, 16)), u0));
    let mut g = MacGrid::new(n, n, n, dx);
    p2g_normalized_with(&mut g, &ps, ParticleScatter::Nearest);
    for (ix, u) in g.u.iter().enumerate() {
        assert_eq!(*u, u0.x, "u face {ix}");
    }
    for (ix, v) in g.v.iter().enumerate() {
        assert_eq!(*v, u0.y, "v face {ix}");
    }
    for (ix, w) in g.w.iter().enumerate() {
        assert_eq!(*w, u0.z, "w face {ix}");
    }
}

// ===========================================================================
// Oracle 5 — strays deposit nothing; the far face belongs to the last cell
// ===========================================================================

#[test]
fn nearest_drops_strays_and_keeps_a_particle_on_the_far_face() {
    let dx = q(1, 2); // domain [0, 1] on each axis of a 2 × 2 × 2 grid
    let ulp = Fix128::from_raw(0, 1);
    let sentinel = int(7);
    let fresh = || {
        let mut g = MacGrid::new(2, 2, 2, dx);
        g.u.fill(sentinel);
        g.v.fill(sentinel);
        g.w.fill(sentinel);
        g
    };

    // On the far x face: cell 1 along x, so u(1) and u(2) of row (0, 0).
    let mut g = fresh();
    p2g_normalized_with(
        &mut g,
        &[(v3(int(1), q(1, 4), q(1, 4)), v3(int(5), int(6), int(8)))],
        ParticleScatter::Nearest,
    );
    assert_eq!(g.u(1, 0, 0), int(5));
    assert_eq!(g.u(2, 0, 0), int(5));
    assert_eq!(
        g.u(0, 0, 0),
        sentinel,
        "the face of the other cell is untouched"
    );
    assert_eq!(g.v(1, 0, 0), int(6));
    assert_eq!(g.v(1, 1, 0), int(6));
    assert_eq!(g.w(1, 0, 0), int(8));
    assert_eq!(g.w(1, 0, 1), int(8));

    // Strays: one ulp beyond the far face, negative, and a coordinate whose
    // quotient by dx overflows. Each leaves the grid bit-identical.
    let strays: [Particle; 7] = [
        (
            v3(int(1) + ulp, q(1, 4), q(1, 4)),
            v3(int(9), int(9), int(9)),
        ),
        (
            v3(q(1, 4), int(1) + ulp, q(1, 4)),
            v3(int(9), int(9), int(9)),
        ),
        (
            v3(q(1, 4), q(1, 4), int(1) + ulp),
            v3(int(9), int(9), int(9)),
        ),
        (
            v3(Fix128::ZERO - ulp, q(1, 4), q(1, 4)),
            v3(int(9), int(9), int(9)),
        ),
        (v3(q(1, 4), q(-3, 2), q(1, 4)), v3(int(9), int(9), int(9))),
        (
            v3(Fix128::from_raw(i64::MAX, 0), q(1, 4), q(1, 4)),
            v3(int(9), int(9), int(9)),
        ),
        (
            v3(q(1, 4), q(1, 4), Fix128::from_raw(i64::MIN, 0)),
            v3(int(9), int(9), int(9)),
        ),
    ];
    for (idx, stray) in strays.iter().enumerate() {
        let mut g = fresh();
        let before = faces(&g);
        let res = catch_unwind(AssertUnwindSafe(|| {
            p2g_normalized_with(&mut g, &[*stray], ParticleScatter::Nearest);
        }));
        assert!(res.is_ok(), "stray {idx} made the scatter panic");
        assert_eq!(faces(&g), before, "stray {idx} deposited something");
    }

    // A quotient that wraps to a small non-negative value: `2^62 · 4 = 2^64`
    // is `0` modulo 2^64 on the integer part, which an unchecked product
    // would read as cell 0.
    let mut fine = MacGrid::new(2, 2, 2, q(1, 4));
    fine.u.fill(sentinel);
    let before = faces(&fine);
    p2g_normalized_with(
        &mut fine,
        &[(
            v3(Fix128::from_raw(1 << 62, 0), q(1, 8), q(1, 8)),
            v3(int(9), int(9), int(9)),
        )],
        ParticleScatter::Nearest,
    );
    assert_eq!(
        faces(&fine),
        before,
        "a wrapping quotient deposited into cell 0"
    );
}

// ===========================================================================
// Oracles 6-7 — the solver entry: Trilinear reproduces `step_flip`, Nearest
// differs on an off-centre cloud and agrees on a uniform one
// ===========================================================================

/// Eight particles per cell at `(2a + 1) / 4` (off centre on every axis),
/// each with its own velocity.
fn sheared_lattice(n: usize) -> Vec<Particle> {
    let m = 2 * n as i64;
    let mut out = Vec::new();
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let pos = v3(q(2 * a + 1, 4), q(2 * b + 1, 4), q(2 * c + 1, 4));
                out.push((pos, v3(pos.y, Fix128::ZERO - pos.x, pos.z * q(1, 2))));
            }
        }
    }
    out
}

fn sealed(n: usize) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    s.grid.set_closed_box_walls();
    s.jacobi_iterations = 20;
    s.gravity = Vec3Fix::ZERO;
    s
}

#[test]
fn step_flip_with_trilinear_is_step_flip_bit_for_bit_and_nearest_differs_off_centre() {
    let n = 3usize;
    let (dt, r) = (q(1, 64), q(1, 2));

    let mut plain = sealed(n);
    let mut ps_plain = sheared_lattice(n);
    plain.step_flip(&mut ps_plain, dt, r);

    let mut tri = sealed(n);
    let mut ps_tri = sheared_lattice(n);
    tri.step_flip_with(&mut ps_tri, dt, r, ParticleScatter::Trilinear);
    assert_eq!(ps_tri, ps_plain, "Trilinear must reproduce step_flip");
    assert_eq!(faces(&tri.grid), faces(&plain.grid));
    assert_eq!(tri.grid.pressure, plain.grid.pressure);

    let mut near = sealed(n);
    let mut ps_near = sheared_lattice(n);
    near.step_flip_with(&mut ps_near, dt, r, ParticleScatter::Nearest);
    assert_ne!(
        ps_near, ps_plain,
        "an off-centre sheared cloud must come out different under Nearest"
    );
    assert_eq!(near.step_count, 1);
    // Still a projected field: every wall face is zero under both.
    for j in 0..n {
        for k in 0..n {
            assert_eq!(near.grid.u(0, j, k), Fix128::ZERO);
            assert_eq!(near.grid.u(n, j, k), Fix128::ZERO);
        }
    }
}

#[test]
fn a_uniform_cloud_makes_step_flip_with_independent_of_the_stencil() {
    let n = 3usize;
    let u0 = v3(q(3, 4), q(-1, 8), q(5, 16));
    let cloud = || -> Vec<Particle> {
        sheared_lattice(n)
            .into_iter()
            .map(|(p, _)| (p, u0))
            .collect()
    };
    let (dt, r) = (q(1, 32), q(3, 4));
    let mut a = sealed(n);
    let mut pa = cloud();
    a.step_flip_with(&mut pa, dt, r, ParticleScatter::Trilinear);
    let mut b = sealed(n);
    let mut pb = cloud();
    b.step_flip_with(&mut pb, dt, r, ParticleScatter::Nearest);
    assert_eq!(pa, pb, "uniform cloud: particles must agree bit for bit");
    assert_eq!(faces(&a.grid), faces(&b.grid));
    assert_eq!(a.grid.pressure, b.grid.pressure);
    // Not vacuous: the step moved the particles.
    assert_ne!(pa, cloud());
}

// ===========================================================================
// Panic / degenerate input
// ===========================================================================

fn assert_untouched(mut s: CfdSolver, mut ps: Vec<Particle>, dt: Fix128, r: Fix128, what: &str) {
    let grid_before = faces(&s.grid);
    let p_before = ps.clone();
    let count = s.step_count;
    let res = catch_unwind(AssertUnwindSafe(|| {
        s.step_flip_with(&mut ps, dt, r, ParticleScatter::Nearest);
    }));
    assert!(res.is_ok(), "{what}: panicked");
    assert_eq!(faces(&s.grid), grid_before, "{what}: grid changed");
    assert_eq!(ps, p_before, "{what}: particles changed");
    assert_eq!(s.step_count, count, "{what}: step_count changed");
}

#[test]
fn step_flip_with_nearest_keeps_every_refusal_of_step_flip() {
    let cloud = sheared_lattice(2);
    assert_untouched(sealed(2), Vec::new(), q(1, 16), q(1, 2), "empty list");
    assert_untouched(sealed(2), cloud.clone(), Fix128::ZERO, q(1, 2), "dt = 0");
    assert_untouched(sealed(2), cloud.clone(), q(-1, 16), q(1, 2), "dt < 0");
    assert_untouched(
        CfdSolver::new(2, 2, 2, Fix128::ZERO),
        cloud.clone(),
        q(1, 16),
        q(1, 2),
        "dx = 0",
    );
    assert_untouched(
        CfdSolver::new(0, 2, 2, Fix128::ONE),
        cloud.clone(),
        q(1, 16),
        q(1, 2),
        "nx = 0",
    );
    let mut rho0 = sealed(2);
    rho0.density_kg_m3 = Fix128::ZERO;
    assert_untouched(rho0, cloud.clone(), q(1, 16), q(1, 2), "rho = 0");
    for r in [q(-1, 4), q(5, 4), int(100)] {
        assert_untouched(sealed(2), cloud.clone(), q(1, 16), r, "flip_ratio");
    }
    let only_strays = vec![
        (v3(int(-1), q(1, 2), q(1, 2)), v3(int(1), int(1), int(1))),
        (v3(q(1, 2), int(3), q(1, 2)), v3(int(1), int(1), int(1))),
    ];
    assert_untouched(sealed(2), only_strays, q(1, 16), q(1, 2), "only strays");
}

#[test]
fn p2g_normalized_with_nearest_refuses_zero_dx_empty_input_and_zero_sized_grids() {
    let cloud = sheared_lattice(2);
    let sentinel = int(3);

    let mut g = MacGrid::new(2, 2, 2, Fix128::ZERO);
    g.u.fill(sentinel);
    let before = faces(&g);
    p2g_normalized_with(&mut g, &cloud, ParticleScatter::Nearest);
    assert_eq!(faces(&g), before, "dx = 0 must leave the grid untouched");

    let mut g = MacGrid::new(2, 2, 2, Fix128::ONE);
    g.v.fill(sentinel);
    let before = faces(&g);
    p2g_normalized_with(&mut g, &[], ParticleScatter::Nearest);
    assert_eq!(
        faces(&g),
        before,
        "empty input must leave the grid untouched"
    );

    for (nx, ny, nz) in [(0, 2, 2), (2, 0, 2), (2, 2, 0), (0, 0, 0)] {
        let mut g = MacGrid::new(nx, ny, nz, Fix128::ONE);
        let before = faces(&g);
        let res = catch_unwind(AssertUnwindSafe(|| {
            p2g_normalized_with(&mut g, &cloud, ParticleScatter::Nearest);
        }));
        assert!(res.is_ok(), "zero-sized grid ({nx}, {ny}, {nz}) panicked");
        assert_eq!(faces(&g), before);
    }

    // The default stencil is the trilinear one, and the two-argument entry
    // is that stencil, bit for bit.
    assert_eq!(ParticleScatter::default(), ParticleScatter::Trilinear);
    let mut a = MacGrid::new(2, 2, 2, Fix128::ONE);
    let mut b = MacGrid::new(2, 2, 2, Fix128::ONE);
    p2g_normalized(&mut a, &cloud);
    p2g_normalized_with(&mut b, &cloud, ParticleScatter::default());
    assert_eq!(faces(&a), faces(&b));
}
