//! The two particle-to-grid stencils of the FLIP / PIC step
//!
//! A sheared particle cloud in a sealed box is stepped once with
//! `step_flip_with` under each `ParticleScatter`. The trilinear stencil
//! reproduces `step_flip`; the nearest stencil gives every particle of a cell
//! the same weight on the cell's faces, so the transferred field no longer
//! depends on where inside its cell a particle sits. A uniform cloud comes
//! out identical under both, which the last line checks.
//!
//! ```bash
//! cargo run --example flip_scatter --features std
//! ```

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{p2g_normalized, p2g_normalized_with, MacGrid, ParticleScatter};
use alice_physics::math::{Fix128, Vec3Fix};

type Particle = (Vec3Fix, Vec3Fix);

fn cloud(n: usize, shear: bool) -> Vec<Particle> {
    let m = 2 * n as i64;
    let mut out = Vec::new();
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let pos = Vec3Fix::new(
                    Fix128::from_ratio(2 * a + 1, 4),
                    Fix128::from_ratio(2 * b + 1, 4),
                    Fix128::from_ratio(2 * c + 1, 4),
                );
                let vel = if shear {
                    Vec3Fix::new(pos.y, Fix128::ZERO - pos.x, Fix128::ZERO)
                } else {
                    Vec3Fix::new(Fix128::from_ratio(3, 4), Fix128::ZERO, Fix128::ZERO)
                };
                out.push((pos, vel));
            }
        }
    }
    out
}

fn solver(n: usize) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    s.grid.set_closed_box_walls();
    s.gravity = Vec3Fix::ZERO;
    s.jacobi_iterations = 20;
    s
}

fn max_abs_diff(a: &[Particle], b: &[Particle]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            [(p.1.x, q.1.x), (p.1.y, q.1.y), (p.1.z, q.1.z)]
                .into_iter()
                .map(|(x, y)| (x.to_f64() - y.to_f64()).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn main() {
    let n = 3usize;
    let (dt, r) = (Fix128::from_ratio(1, 64), Fix128::from_ratio(1, 2));

    // The transfer on its own, before any step: the two-argument entry is
    // the trilinear stencil.
    let sheared = cloud(n, true);
    let mut g_default = MacGrid::new(n, n, n, Fix128::ONE);
    let mut g_tri = MacGrid::new(n, n, n, Fix128::ONE);
    let mut g_near = MacGrid::new(n, n, n, Fix128::ONE);
    p2g_normalized(&mut g_default, &sheared);
    p2g_normalized_with(&mut g_tri, &sheared, ParticleScatter::Trilinear);
    p2g_normalized_with(&mut g_near, &sheared, ParticleScatter::Nearest);
    let max_face_gap = g_tri
        .u
        .iter()
        .zip(&g_near.u)
        .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
        .fold(0.0, f64::max);
    println!(
        "[flip_scatter] transfer only: p2g_normalized == Trilinear: {}, max |u_tri - u_near| over the faces = {max_face_gap:.4}",
        g_default.u == g_tri.u && g_default.v == g_tri.v && g_default.w == g_tri.w
    );

    let mut plain = solver(n);
    let mut ps_plain = cloud(n, true);
    plain.step_flip(&mut ps_plain, dt, r);

    let mut tri = solver(n);
    let mut ps_tri = cloud(n, true);
    tri.step_flip_with(&mut ps_tri, dt, r, ParticleScatter::Trilinear);

    let mut near = solver(n);
    let mut ps_near = cloud(n, true);
    near.step_flip_with(&mut ps_near, dt, r, ParticleScatter::Nearest);

    println!(
        "[flip_scatter] sheared cloud, {} particles, dt = {}, flip ratio = {}",
        ps_plain.len(),
        dt.to_f64(),
        r.to_f64()
    );
    println!(
        "[flip_scatter] Trilinear vs step_flip: bit-identical = {}",
        ps_tri == ps_plain && tri.grid.u == plain.grid.u
    );
    println!(
        "[flip_scatter] Nearest vs Trilinear: max |dv| = {:.3e} (particle velocities differ: {})",
        max_abs_diff(&ps_near, &ps_tri),
        ps_near != ps_tri
    );
    println!(
        "[flip_scatter] grid u(1,1,1) after the step: trilinear {:.6}, nearest {:.6}",
        tri.grid.u(1, 1, 1).to_f64(),
        near.grid.u(1, 1, 1).to_f64()
    );

    let mut ua = solver(n);
    let mut pa = cloud(n, false);
    ua.step_flip_with(&mut pa, dt, r, ParticleScatter::Trilinear);
    let mut ub = solver(n);
    let mut pb = cloud(n, false);
    ub.step_flip_with(&mut pb, dt, r, ParticleScatter::Nearest);
    println!(
        "[flip_scatter] uniform cloud: the two stencils agree bit for bit = {}",
        pa == pb && ua.grid.u == ub.grid.u
    );
}
