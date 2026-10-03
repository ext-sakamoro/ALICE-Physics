//! Oracles for `sdf_sph`: kernels, density, the spatial hash and the
//! hashed-vs-naive equivalence the module doc claims.
//!
//! Closed forms (Muller, Charypar, Gross 2003):
//! - `W_poly6(r, h) = 315 / (64 pi h^9) (h^2 - r^2)^3`
//! - `|grad W_spiky| = 45 / (pi h^6) (h - r)^2`
//! - `lap W_visc = 45 / (pi h^6) (h - r)`
//! - density of an isolated particle `rho = m W_poly6(0, h) = 315 m / (64 pi h^3)`
//!
//! NOTE: the sign / density normalisation of the pressure force is not
//! pinned here; see the Backlog entry `sdf_sph pressure force`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_sph::{
    poly6, spiky_grad, viscosity_lap, SphConfig, SphParticle, SphSolver, SphSpatialHash,
};

const PI: f64 = core::f64::consts::PI;

fn far_field() -> ClosureSdf {
    ClosureSdf::new(|_x, _y, _z| 1.0e6, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn close(a: f32, b: f64, rel: f64) -> bool {
    ((a as f64) - b).abs() <= rel * b.abs().max(1e-12)
}

#[test]
fn kernels_match_closed_forms() {
    let h = 0.1_f64;
    for r in [0.0_f64, 0.02, 0.05, 0.09] {
        let p6 = 315.0 / (64.0 * PI * h.powi(9)) * (h * h - r * r).powi(3);
        assert!(close(poly6(r as f32, h as f32), p6, 2e-4), "poly6 r={r}");
        let sp = 45.0 / (PI * h.powi(6)) * (h - r).powi(2);
        assert!(
            close(spiky_grad(r as f32, h as f32), sp, 2e-4),
            "spiky r={r}"
        );
        let vl = 45.0 / (PI * h.powi(6)) * (h - r);
        assert!(
            close(viscosity_lap(r as f32, h as f32), vl, 2e-4),
            "visc r={r}"
        );
    }
    // Compact support: exactly zero at and beyond h.
    for r in [0.1_f32, 0.2] {
        assert_eq!(poly6(r, 0.1), 0.0);
        assert_eq!(spiky_grad(r, 0.1), 0.0);
        assert_eq!(viscosity_lap(r, 0.1), 0.0);
    }
}

#[test]
fn poly6_integrates_to_one() {
    // Volume integral 4 pi int_0^h W r^2 dr = 1 (midpoint rule, 4000 shells).
    let h = 0.1_f64;
    let n = 4000;
    let mut s = 0.0_f64;
    for i in 0..n {
        let r = (i as f64 + 0.5) * h / n as f64;
        s += poly6(r as f32, h as f32) as f64 * r * r * (h / n as f64);
    }
    assert!((4.0 * PI * s - 1.0).abs() < 2e-3, "{}", 4.0 * PI * s);
}

#[test]
fn isolated_particle_density_and_pressure() {
    let cfg = SphConfig::water_like();
    let f = far_field();
    let mut s = SphSolver::new(vec![SphParticle::at_rest([0.0, 0.0, 0.0])], cfg, &f);
    s.step(0.0);
    let rho = 315.0 * cfg.particle_mass as f64 / (64.0 * PI * (cfg.kernel_radius as f64).powi(3));
    assert!(
        close(s.particles[0].density, rho, 2e-4),
        "rho {}",
        s.particles[0].density
    );
    // rho = 250 < rest 1000: pressure clamps to 0.
    assert_eq!(s.particles[0].pressure, 0.0);
    // Overpressured: rest density below the sampled one -> p = k (rho - rho0).
    let mut cfg2 = cfg;
    cfg2.rest_density = 100.0;
    let mut s2 = SphSolver::new(vec![SphParticle::at_rest([0.0, 0.0, 0.0])], cfg2, &f);
    s2.step(0.0);
    assert!(close(
        s2.particles[0].pressure,
        cfg2.gas_stiffness as f64 * (rho - 100.0),
        2e-3
    ));
}

#[test]
fn gravity_and_boundary_repulsion_on_an_isolated_particle() {
    // Plane y = 0 with the fluid above. A particle at y = 0.01 (d = 0.01 < range 0.02):
    // a_y = g + strength * (range - d) = -9.81 + 500 * 0.01 = -4.81.
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let cfg = SphConfig::water_like();
    let mut s = SphSolver::new(vec![SphParticle::at_rest([0.0, 0.01, 0.0])], cfg, &plane);
    s.step(0.001);
    assert!(
        close(s.particles[0].velocity[1], -4.81e-3, 2e-3),
        "{}",
        s.particles[0].velocity[1]
    );
    // Position is advanced by the *new* velocity (symplectic Euler).
    assert!(close(
        s.particles[0].position[1] - 0.01,
        -4.81e-3 * 1e-3,
        5e-3
    ));
    // The hashed path applies the same boundary force.
    let mut s = SphSolver::new(vec![SphParticle::at_rest([0.0, 0.01, 0.0])], cfg, &plane);
    s.step_hashed(0.001);
    assert!(
        close(s.particles[0].velocity[1], -4.81e-3, 2e-3),
        "{}",
        s.particles[0].velocity[1]
    );
    // Outside the repel range only gravity acts.
    let mut s = SphSolver::new(vec![SphParticle::at_rest([0.0, 0.5, 0.0])], cfg, &plane);
    s.step(0.001);
    assert!(close(s.particles[0].velocity[1], -9.81e-3, 1e-4));
    assert_eq!(s.particles[0].velocity[0], 0.0);
    assert_eq!(s.particles[0].velocity[2], 0.0);
}

#[test]
fn spatial_hash_cells_and_neighbourhood() {
    let h = 0.1_f32;
    let ps = vec![
        SphParticle::at_rest([0.05, 0.05, 0.05]),  // cell (0,0,0)
        SphParticle::at_rest([0.06, 0.07, 0.01]),  // cell (0,0,0)
        SphParticle::at_rest([0.15, 0.05, 0.05]),  // cell (1,0,0)
        SphParticle::at_rest([-0.01, 0.05, 0.05]), // cell (-1,0,0): floor, not truncation
        SphParticle::at_rest([0.55, 0.55, 0.55]),  // cell (5,5,5)
    ];
    let hash = SphSpatialHash::build(&ps, h);
    assert_eq!(hash.cell_size(), h);
    assert_eq!(hash.populated_cell_count(), 4);
    // Neighbours of a probe in cell (0,0,0): cells -1..=1 on each axis -> particles 0..=3.
    let mut seen = Vec::new();
    hash.for_each_neighbour([0.05, 0.05, 0.05], |j| seen.push(j));
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3]);
    // A probe far from everything sees nobody.
    let mut none = 0;
    hash.for_each_neighbour([5.0, 5.0, 5.0], |_| none += 1);
    assert_eq!(none, 0);
    // The probe in cell (5,5,5) sees only itself.
    let mut only = Vec::new();
    hash.for_each_neighbour([0.55, 0.55, 0.55], |j| only.push(j));
    assert_eq!(only, vec![4]);
    // Empty and degenerate radius.
    assert_eq!(SphSpatialHash::build(&[], h).populated_cell_count(), 0);
    assert!((SphSpatialHash::build(&ps, 0.0).cell_size() - 1.0e-6).abs() < 1e-12);
    assert_eq!(SphSpatialHash::build(&ps, -1.0).cell_size(), 1.0e-6);
}

#[test]
fn neighbourhood_covers_the_whole_kernel_sphere() {
    // Every particle within h of the probe must be visited (cell size = h, 3x3x3 cells).
    let h = 0.1_f32;
    let mut ps = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                ps.push(SphParticle::at_rest([
                    i as f32 * 0.037 - 0.1,
                    j as f32 * 0.041 - 0.1,
                    k as f32 * 0.029 - 0.1,
                ]));
            }
        }
    }
    let hash = SphSpatialHash::build(&ps, h);
    for probe in [[0.0, 0.0, 0.0], [0.043, 0.031, -0.02], [-0.099, 0.099, 0.0]] {
        let mut visited = vec![false; ps.len()];
        hash.for_each_neighbour(probe, |j| visited[j] = true);
        for (j, p) in ps.iter().enumerate() {
            let d2: f32 = (0..3).map(|a| (p.position[a] - probe[a]).powi(2)).sum();
            if d2 < h * h {
                assert!(visited[j], "particle {j} within h of {probe:?} not visited");
            }
        }
    }
}

#[test]
fn hashed_step_equals_the_naive_step() {
    let f = ClosureSdf::new(|_x, y, _z| y + 0.05, |_x, _y, _z| (0.0, 1.0, 0.0));
    let cfg = SphConfig::water_like();
    let mut ps = Vec::new();
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..5 {
                let mut p =
                    SphParticle::at_rest([i as f32 * 0.03, j as f32 * 0.03, k as f32 * 0.03]);
                p.velocity = [1.0 * (i as f32) - 2.0, 0.5 * (j as f32), 0.7 * (k as f32)];
                ps.push(p);
            }
        }
    }
    let mut naive = SphSolver::new(ps.clone(), cfg, &f);
    let mut hashed = SphSolver::new(ps, cfg, &f);
    for _ in 0..3 {
        naive.step(1.0e-4);
        hashed.step_hashed(1.0e-4);
    }
    for (a, b) in naive.particles.iter().zip(&hashed.particles) {
        for ax in 0..3 {
            assert!(
                (a.position[ax] - b.position[ax]).abs() < 1e-6,
                "{:?} vs {:?}",
                a.position,
                b.position
            );
            assert!(
                (a.velocity[ax] - b.velocity[ax]).abs() <= 1e-3 * a.velocity[ax].abs().max(1.0),
                "{:?} vs {:?}",
                a.velocity,
                b.velocity
            );
        }
        assert!((a.density - b.density).abs() <= 1e-3 * a.density.max(1.0));
    }
}

// ---------------------------------------------------------------------------
// Pressure and viscosity forces (Muller 2003, symmetric form), with the
// acceleration a_i = f_i / rho_i:
//   a_i^p = sum_j m (p_i + p_j) / (2 rho_j rho_i) |grad W_spiky(r)| r_hat_ij
//           (r_hat_ij from j to i: over-pressure pushes i away from j)
//   a_i^v = mu sum_j m (v_j - v_i) / (rho_j rho_i) lap W_visc(r)
// ---------------------------------------------------------------------------

fn rho_pair(m: f64, h: f64, r: f64) -> f64 {
    let w = |r: f64| 315.0 / (64.0 * PI * h.powi(9)) * (h * h - r * r).powi(3);
    m * (w(0.0) + w(r))
}

fn run_pair(hashed: bool, v0: f32, v1: f32) -> (SphSolver<'static, ClosureSdf>, f64) {
    // leak a far field so the solver can borrow it for 'static in the test
    let f: &'static ClosureSdf = Box::leak(Box::new(far_field()));
    let mut cfg = SphConfig::water_like();
    cfg.gravity = [0.0; 3];
    cfg.viscosity = 0.0;
    cfg.rest_density = 1.0; // force over-pressure
    let mut a = SphParticle::at_rest([0.0, 0.0, 0.0]);
    let mut b = SphParticle::at_rest([0.02, 0.0, 0.0]);
    a.velocity = [v0, 0.0, 0.0];
    b.velocity = [v1, 0.0, 0.0];
    let mut s = SphSolver::new(vec![a, b], cfg, f);
    let dt = 1.0e-4;
    if hashed {
        s.step_hashed(dt);
    } else {
        s.step(dt);
    }
    (s, dt as f64)
}

#[test]
fn over_pressure_pushes_two_particles_apart_with_the_closed_form_magnitude() {
    let (m, h, r, k, rho0) = (0.02_f64, 0.05_f64, 0.02_f64, 20.0_f64, 1.0_f64);
    let rho = rho_pair(m, h, r);
    let p = k * (rho - rho0);
    let grad = 45.0 / (PI * h.powi(6)) * (h - r).powi(2);
    let want = m * (p + p) / (2.0 * rho * rho) * grad; // |a|, along +-x
    for hashed in [false, true] {
        let (s, dt) = run_pair(hashed, 0.0, 0.0);
        let a0 = s.particles[0].velocity[0] as f64 / dt;
        let a1 = s.particles[1].velocity[0] as f64 / dt;
        assert!(
            a0 < 0.0 && a1 > 0.0,
            "hashed={hashed}: must separate, a0={a0} a1={a1}"
        );
        assert!(
            ((-a0) - want).abs() <= 1e-3 * want,
            "hashed={hashed}: |a0| {} vs {want}",
            -a0
        );
        assert!(
            (a1 - want).abs() <= 1e-3 * want,
            "hashed={hashed}: a1 {a1} vs {want}"
        );
        assert!(s.particles[0].velocity[1] == 0.0 && s.particles[0].velocity[2] == 0.0);
    }
}

#[test]
fn pressure_and_viscosity_conserve_momentum_for_an_asymmetric_cluster() {
    let f = far_field();
    let mut cfg = SphConfig::water_like();
    cfg.gravity = [0.0; 3];
    cfg.rest_density = 1.0;
    cfg.viscosity = 0.3;
    let pos = [
        [0.0, 0.0, 0.0],
        [0.02, 0.005, 0.0],
        [0.01, 0.025, 0.01],
        [-0.015, 0.01, 0.02],
        [0.03, -0.01, 0.015],
    ];
    let vel = [
        [1.0, 0.0, 0.0],
        [-0.5, 0.2, 0.0],
        [0.0, -1.0, 0.3],
        [0.7, 0.1, -0.2],
        [0.0, 0.0, 0.9],
    ];
    for hashed in [false, true] {
        let ps: Vec<SphParticle> = pos
            .iter()
            .zip(&vel)
            .map(|(p, v)| {
                let mut q = SphParticle::at_rest(*p);
                q.velocity = *v;
                q
            })
            .collect();
        let mut s = SphSolver::new(ps, cfg, &f);
        let dt = 1.0e-5;
        if hashed {
            s.step_hashed(dt);
        } else {
            s.step(dt);
        }
        // sum of velocity changes (uniform mass) must vanish
        let mut sum = [0.0_f64; 3];
        let mut mag = 0.0_f64;
        for (q, v) in s.particles.iter().zip(&vel) {
            for a in 0..3 {
                let d = (q.velocity[a] - v[a]) as f64 / dt as f64;
                sum[a] += d;
                mag += d.abs();
            }
        }
        assert!(mag > 1.0, "forces must be non-trivial: {mag}");
        for a in 0..3 {
            assert!(
                sum[a].abs() < 2e-4 * mag,
                "hashed={hashed}: momentum drift {sum:?} vs {mag}"
            );
        }
    }
}

#[test]
fn viscosity_reduces_the_relative_velocity_with_the_closed_form_magnitude() {
    let (m, h, r, mu) = (0.02_f64, 0.05_f64, 0.02_f64, 0.5_f64);
    let rho = rho_pair(m, h, r);
    let lap = 45.0 / (PI * h.powi(6)) * (h - r);
    // pressure off (rest density above the sampled density -> clamped to 0)
    let want = mu * m * 1.0 / (rho * rho) * lap; // |a| for relative speed 1
    for (hashed, axis) in [
        (false, 1),
        (true, 1),
        (false, 0),
        (true, 0),
        (false, 2),
        (true, 2),
    ] {
        let f: &'static ClosureSdf = Box::leak(Box::new(far_field()));
        let mut cfg = SphConfig::water_like();
        cfg.gravity = [0.0; 3];
        cfg.viscosity = mu as f32;
        let mut a = SphParticle::at_rest([0.0, 0.0, 0.0]);
        let b = SphParticle::at_rest([0.02, 0.0, 0.0]);
        a.velocity[axis] = 1.0;
        let mut s = SphSolver::new(vec![a, b], cfg, f);
        let dt = 1.0e-5_f32;
        if hashed {
            s.step_hashed(dt)
        } else {
            s.step(dt)
        }
        let da = (s.particles[0].velocity[axis] as f64 - 1.0) / dt as f64;
        let db = s.particles[1].velocity[axis] as f64 / dt as f64;
        assert!(
            da < 0.0 && db > 0.0,
            "hashed={hashed}: relative velocity must shrink: {da} {db}"
        );
        assert!(
            (-da - want).abs() <= 2e-3 * want,
            "hashed={hashed}: {} vs {want}",
            -da
        );
        assert!((db - want).abs() <= 2e-3 * want);
    }
}
