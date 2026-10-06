//! Audit S1-4 oracles for `cfd_solver` (axis A: claims the module makes that no
//! existing oracle pins).
//!
//! Every expected value is a closed form or a symmetry, not read back from the
//! implementation:
//!
//! ```text
//! free fall            u = g dt per face component
//! Boussinesq           v = g_y dt (1 - beta (T_face - T_ref))
//! hydrostatics         p(j+1) - p(j) = rho g_y dx,   v = 0
//! linear transport     phi_new(x) = phi(x - c dt)    (trilinear sampling is exact on a linear field)
//! Young-Laplace line   sum_i f_x dx = 2 sigma / R    (Brackbill 1992), sphere symmetry x <-> y <-> z
//! ```
//!
//! Tests marked `#[ignore = "known defect: AUD-..."]` are red on the audited
//! base: the oracle is right and the implementation is not.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::{
    AdvectionScheme, CfdSolver, PressureSolver, StepError, StepOptions, WallModel,
};
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{initialize_level_set_sphere, Grid3d};

fn q(a: i64, b: i64) -> Fix128 {
    Fix128::from_ratio(a, b)
}

fn int(a: i64) -> Fix128 {
    Fix128::from_int(a)
}

/// A solver with no gravity, no viscosity, no surface tension and a projection
/// that does nothing (`step_multigrid(dt, 0)` falls back to Gauss-Seidel with
/// `jacobi_iterations = 0` sweeps), so that one `step` leaves exactly what the
/// advection and the body forces made.
fn bare(nx: usize, ny: usize, nz: usize, dx: Fix128) -> CfdSolver {
    let mut s = CfdSolver::new(nx, ny, nz, dx);
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.surface_tension_n_m = Fix128::ZERO;
    s.jacobi_iterations = 0;
    s.reinit_every_n_steps = 0;
    s
}

fn close_box(s: &mut CfdSolver) {
    let (nx, ny, nz) = (s.grid.nx, s.grid.ny, s.grid.nz);
    for k in 0..nz {
        for j in 0..ny {
            s.grid.set_u_bc(0, j, k, FaceBc::SlipWall);
            s.grid.set_u_bc(nx, j, k, FaceBc::SlipWall);
        }
    }
    for k in 0..nz {
        for i in 0..nx {
            s.grid.set_v_bc(i, 0, k, FaceBc::SlipWall);
            s.grid.set_v_bc(i, ny, k, FaceBc::SlipWall);
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            s.grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            s.grid.set_w_bc(i, j, nz, FaceBc::SlipWall);
        }
    }
}

// ---------------------------------------------------------------- gravity

#[test]
fn free_fall_gives_every_face_component_g_dt() {
    // gravity in all three axes: u = g_x dt, v = g_y dt, w = g_z dt on every face
    let mut s = bare(4, 5, 6, int(1));
    s.gravity = Vec3Fix::new(q(3, 10), q(-981, 100), q(-7, 5));
    let dt = q(1, 50);
    s.step_multigrid(dt, 0);
    let (gx, gy, gz) = (
        s.gravity.x * dt, // closed form is the product itself
        s.gravity.y * dt,
        s.gravity.z * dt,
    );
    assert!(s
        .grid
        .u
        .iter()
        .all(|&u| (u - gx).abs() < q(1, 1_000_000_000)));
    assert!(s
        .grid
        .v
        .iter()
        .all(|&v| (v - gy).abs() < q(1, 1_000_000_000)));
    assert!(s
        .grid
        .w
        .iter()
        .all(|&w| (w - gz).abs() < q(1, 1_000_000_000)));
    // and the value is the textbook one, -9.81 * 0.02 = -0.1962 (independent of the product above)
    assert!((s.grid.v[0].to_f64() + 0.1962).abs() < 1e-12);
}

#[test]
fn hydrostatic_column_in_a_closed_box_has_dp_equal_rho_g_dx() {
    // closed box, gravity only: the projection must cancel g dt exactly and leave
    // p(j+1) - p(j) = rho g_y dx (pressure rises downward, g_y < 0)
    let n = 6;
    let dx = q(1, 4);
    let mut s = CfdSolver::new(n, n, n, dx);
    s.dynamic_viscosity_pas = Fix128::ZERO;
    close_box(&mut s);
    let dt = q(1, 100);
    let opts = StepOptions::new(PressureSolver::BiCgStab {
        max_iterations: 400,
        tolerance: q(1, 1_000_000_000),
    });
    s.step_with_options(dt, &opts).expect("steps");
    let mut worst_v = 0.0f64;
    for &v in &s.grid.v {
        worst_v = worst_v.max(v.to_f64().abs());
    }
    assert!(worst_v < 1e-6, "residual vertical velocity {worst_v}");
    let rho_g_dx = 1000.0 * 9.81 * dx.to_f64();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n - 1 {
                let lo = s.grid.pressure[i + n * (j + n * k)].to_f64();
                let hi = s.grid.pressure[i + n * (j + 1 + n * k)].to_f64();
                let dp = lo - hi; // lower cell is at higher pressure
                assert!(
                    (dp - rho_g_dx).abs() < 1e-3 * rho_g_dx,
                    "dp = {dp}, rho g dx = {rho_g_dx} at ({i},{j},{k})"
                );
            }
        }
    }
}

// ------------------------------------------------------------- Boussinesq

fn temperature_solver(dx: Fix128, profile: &dyn Fn(usize) -> Fix128) -> CfdSolver {
    let (nx, ny, nz) = (3, 6, 3);
    let mut s = bare(nx, ny, nz, dx);
    s.gravity = Vec3Fix::new(Fix128::ZERO, q(-981, 100), Fix128::ZERO);
    s.reference_temp_k = int(293);
    let mut t = Grid3d::new(nx, ny, nz, dx, int(293));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t.set(i, j, k, profile(j));
            }
        }
    }
    s.temperature = Some(t);
    s
}

#[test]
fn boussinesq_uniform_hot_fluid_falls_slower_by_the_factor_1_minus_beta_dt() {
    // v = g_y dt (1 - beta (T - T_ref)) on every y face, boundary faces included
    let mut s = temperature_solver(int(1), &|_| int(303));
    let beta = s.beta_per_k.to_f64(); // the default fluid's expansion coefficient
    let dt = q(1, 50);
    s.step_multigrid(dt, 0);
    let expect = -9.81 * 0.02 * (1.0 - beta * 10.0);
    for (ix, &v) in s.grid.v.iter().enumerate() {
        assert!(
            (v.to_f64() - expect).abs() < 1e-12,
            "face {ix}: {} vs {expect}",
            v.to_f64()
        );
    }
}

#[test]
fn boussinesq_face_force_is_the_mean_of_the_two_cells_sharing_the_face() {
    // T(cell j) = 293 + G (j + 1/2) dx, so the face j sits at T = 293 + G j dx exactly
    // (the mean of the two cells is exact for a linear profile)
    let dx = int(1);
    let g_per_m = 50.0; // K per metre
    let mut s = temperature_solver(dx, &|j| int(293) + q(50 * (2 * j as i64 + 1), 2));
    let beta = s.beta_per_k.to_f64();
    let dt = q(1, 50);
    s.step_multigrid(dt, 0);
    let nx = s.grid.nx;
    for j in 1..s.grid.ny {
        let t_face = g_per_m * j as f64; // + 293, relative to T_ref = 293
        let expect = -9.81 * 0.02 * (1.0 - beta * t_face);
        let got = s.grid.v[nx * (j)].to_f64(); // face (0, j, 0)
        assert!(
            (got - expect).abs() < 1e-9,
            "face {j}: got {got}, closed form {expect}"
        );
    }
}

// ------------------------------------------------- linear-field transport

/// `phi = a * index` advected by a uniform `u = c` over `dt` with `c dt / dx = 1/2`:
/// trilinear sampling and the MacCormack / BFECC corrections reproduce a linear
/// field exactly, so `phi_new(i) = a (i - 1/2)` away from the rim (two cells in: BFECC's compensator carries the clamped rim value one cell inward).
fn linear_transport(scheme: AdvectionScheme) {
    let (nx, ny, nz) = (8usize, 3usize, 3usize);
    let dx = int(1);
    let mut s = bare(nx, ny, nz, dx);
    s.advection_scheme = scheme;
    s.reference_temp_k = int(300);
    let mut t = Grid3d::new(nx, ny, nz, dx, int(300));
    let mut ls = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t.set(i, j, k, int(300) + int(10 * i as i64));
                ls.set(i, j, k, int(2 * i as i64 - 7));
            }
        }
    }
    s.temperature = Some(t);
    s.level_set = Some(ls);
    // u = 1 everywhere; v(i,j,k) = 4 (i + 1/2) is linear in x (a face at x = i + 1/2)
    for u in s.grid.u.iter_mut() {
        *u = int(1);
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                let ix = i + nx * (j + (ny + 1) * k);
                s.grid.v[ix] = int(4) * (int(i as i64) + q(1, 2));
            }
        }
    }
    s.step_multigrid(q(1, 2), 0);
    for k in 0..nz {
        for j in 0..ny {
            for i in 2..nx - 3 {
                let want_t = 300.0 + 10.0 * (i as f64 - 0.5);
                let got_t = s.temperature.as_ref().unwrap().get(i, j, k).to_f64();
                assert!(
                    (got_t - want_t).abs() < 1e-9,
                    "{scheme:?} T({i},{j},{k}) = {got_t}, want {want_t}"
                );
                let want_l = 2.0 * (i as f64 - 0.5) - 7.0;
                let got_l = s.level_set.as_ref().unwrap().get(i, j, k).to_f64();
                assert!(
                    (got_l - want_l).abs() < 1e-9,
                    "{scheme:?} phi({i},{j},{k}) = {got_l}, want {want_l}"
                );
            }
        }
    }
    // the v field is linear in x too: v_new(i) = 4 ((i + 1/2) - 1/2) = 4 i
    for k in 0..nz {
        for j in 0..=ny {
            for i in 2..nx - 3 {
                let ix = i + nx * (j + (ny + 1) * k);
                let got = s.grid.v[ix].to_f64();
                let want = 4.0 * i as f64;
                assert!(
                    (got - want).abs() < 1e-9,
                    "{scheme:?} v({i},{j},{k}) = {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn semi_lagrangian_translates_a_linear_field_exactly() {
    linear_transport(AdvectionScheme::SemiLagrangian);
}

#[test]
fn maccormack_translates_a_linear_field_exactly() {
    linear_transport(AdvectionScheme::MacCormack);
}

#[test]
fn bfecc_translates_a_linear_field_exactly() {
    linear_transport(AdvectionScheme::Bfecc);
}

#[test]
fn maccormack_does_not_overshoot_a_step_profile_although_its_doc_says_unlimited() {
    // `AdvectionScheme::MacCormack` is documented "(second-order, unlimited)" and
    // "No monotone flux limiter is applied ... can produce local over/under-shoots".
    // The code clamps to the back-traced 8-corner range (Fedkiw limiter): the
    // result stays inside the pre-advection range. This pins the *code* (green);
    // the doc is the thing that is wrong (AUD-A-S1W4-003).
    let (nx, ny, nz) = (10usize, 3usize, 3usize);
    let mut s = bare(nx, ny, nz, int(1));
    s.advection_scheme = AdvectionScheme::MacCormack;
    let mut t = Grid3d::new(nx, ny, nz, int(1), int(300));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t.set(i, j, k, if i < 5 { int(400) } else { int(300) });
            }
        }
    }
    s.temperature = Some(t);
    for u in s.grid.u.iter_mut() {
        *u = q(7, 10);
    }
    s.step_multigrid(q(1, 1), 0);
    for &v in &s.temperature.as_ref().unwrap().data {
        assert!(v >= int(300) && v <= int(400), "overshoot {}", v.to_f64());
    }
}

// --------------------------------------------- self-advection (inviscid Burgers)

/// `c(x) = a x` along its own axis advects itself: for the linear field the
/// semi-Lagrangian back-trace is exact, `c_new(x) = c(x - dt c(x)) = a x (1 - q)`
/// with `q = a dt`. MacCormack is `a x (1 - q + q^2 / 2)` and BFECC
/// `a x (1 - q) (1 + q^2 / 2)` (both worked out in the doc of this test, away
/// from the clamped rim and with the limiter inactive).
fn burgers_linear(scheme: AdvectionScheme, axis: usize) {
    let n = 8usize;
    let mut s = bare(n, n, n, int(1));
    s.advection_scheme = scheme;
    let a = q(1, 10);
    let dt = q(1, 2);
    let qq = 0.05f64;
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let ix = i + (n + 1) * (j + n * k);
                if axis == 0 {
                    s.grid.u[ix] = a * int(i as i64);
                }
            }
        }
    }
    for k in 0..n {
        for j in 0..=n {
            for i in 0..n {
                let ix = i + n * (j + (n + 1) * k);
                if axis == 1 {
                    s.grid.v[ix] = a * int(j as i64);
                }
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                let ix = i + n * (j + n * k);
                if axis == 2 {
                    s.grid.w[ix] = a * int(k as i64);
                }
            }
        }
    }
    s.step_multigrid(dt, 0);
    let factor = match scheme {
        AdvectionScheme::SemiLagrangian => 1.0 - qq,
        AdvectionScheme::MacCormack => 1.0 - qq + 0.5 * qq * qq,
        AdvectionScheme::Bfecc => (1.0 - qq) * (1.0 + 0.5 * qq * qq),
    };
    let top = if scheme == AdvectionScheme::SemiLagrangian {
        n
    } else {
        n - 1
    };
    for c in 0..=top {
        let want = 0.1 * c as f64 * factor;
        for t in 0..n {
            for r in 0..n {
                let got = match axis {
                    0 => s.grid.u(c, r, t),
                    1 => s.grid.v(r, c, t),
                    _ => s.grid.w(r, t, c),
                }
                .to_f64();
                assert!(
                    (got - want).abs() < 1e-9,
                    "{scheme:?} axis {axis} face {c}: {got} vs closed form {want}"
                );
            }
        }
    }
    // the other two components stay zero
    let others: Vec<f64> = match axis {
        0 => s
            .grid
            .v
            .iter()
            .chain(&s.grid.w)
            .map(|v| v.to_f64())
            .collect(),
        1 => s
            .grid
            .u
            .iter()
            .chain(&s.grid.w)
            .map(|v| v.to_f64())
            .collect(),
        _ => s
            .grid
            .u
            .iter()
            .chain(&s.grid.v)
            .map(|v| v.to_f64())
            .collect(),
    };
    assert!(
        others.iter().all(|&v| v == 0.0),
        "{scheme:?} axis {axis}: cross-talk"
    );
}

#[test]
fn semi_lagrangian_self_advection_of_a_linear_field_is_a_x_times_one_minus_q_on_every_axis() {
    for axis in 0..3 {
        burgers_linear(AdvectionScheme::SemiLagrangian, axis);
    }
}

#[test]
fn maccormack_self_advection_of_a_linear_field_matches_the_second_order_closed_form() {
    for axis in 0..3 {
        burgers_linear(AdvectionScheme::MacCormack, axis);
    }
}

#[test]
fn bfecc_self_advection_of_a_linear_field_matches_the_closed_form() {
    for axis in 0..3 {
        burgers_linear(AdvectionScheme::Bfecc, axis);
    }
}

/// a step profile in each velocity component, advected by itself: the limited
/// schemes (MacCormack, BFECC) stay inside the pre-advection range
fn step_profile_bounded(scheme: AdvectionScheme, axis: usize) {
    let n = 10usize;
    let mut s = bare(n, n, n, int(1));
    s.advection_scheme = scheme;
    let hi = int(1);
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let ix = i + (n + 1) * (j + n * k);
                if axis == 0 && i < 5 {
                    s.grid.u[ix] = hi;
                }
            }
        }
    }
    for k in 0..n {
        for j in 0..=n {
            for i in 0..n {
                let ix = i + n * (j + (n + 1) * k);
                if axis == 1 && j < 5 {
                    s.grid.v[ix] = hi;
                }
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                let ix = i + n * (j + n * k);
                if axis == 2 && k < 5 {
                    s.grid.w[ix] = hi;
                }
            }
        }
    }
    s.step_multigrid(q(7, 10), 0);
    let all: Vec<Fix128> = s
        .grid
        .u
        .iter()
        .chain(&s.grid.v)
        .chain(&s.grid.w)
        .copied()
        .collect();
    for v in all {
        assert!(
            v >= Fix128::ZERO && v <= hi,
            "{scheme:?} axis {axis}: {} left [0, 1]",
            v.to_f64()
        );
    }
}

#[test]
fn the_limited_schemes_keep_a_velocity_step_inside_its_initial_range_on_every_axis() {
    for scheme in [AdvectionScheme::MacCormack, AdvectionScheme::Bfecc] {
        for axis in 0..3 {
            step_profile_bounded(scheme, axis);
        }
    }
}

/// `T = i^3` (cell index) carried by `u = 1` over `dt = 1/2` (half a cell). The
/// documented formulas applied to the cubic in exact rational arithmetic give,
/// for cell `i`: MacCormack `3, 61/4, 85/2, 363/4, 166` (i = 2..6; not exact for a
/// cubic) and BFECC `(i - 1/2)^3` (exact for a cubic). The two schemes differ here,
/// which they do not on a linear or a quadratic field.
fn cubic_transport(scheme: AdvectionScheme) -> Vec<f64> {
    let (nx, ny, nz) = (10usize, 3usize, 3usize);
    let mut s = bare(nx, ny, nz, int(1));
    s.advection_scheme = scheme;
    let mut t = Grid3d::new(nx, ny, nz, int(1), Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t.set(i, j, k, int((i * i * i) as i64));
            }
        }
    }
    s.temperature = Some(t);
    s.beta_per_k = Fix128::ZERO;
    for u in s.grid.u.iter_mut() {
        *u = int(1);
    }
    s.step_multigrid(q(1, 2), 0);
    (2..=6)
        .map(|i| s.temperature.as_ref().unwrap().get(i, 1, 1).to_f64())
        .collect()
}

#[test]
fn maccormack_temperature_on_a_cubic_matches_the_documented_formulas() {
    let want = [3.0, 61.0 / 4.0, 85.0 / 2.0, 363.0 / 4.0, 166.0];
    let got = cubic_transport(AdvectionScheme::MacCormack);
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-9, "MacCormack {got:?} vs {want:?}");
    }
}

#[test]
fn bfecc_temperature_on_a_cubic_is_exact() {
    let got = cubic_transport(AdvectionScheme::Bfecc);
    for (n, g) in got.iter().enumerate() {
        let i = (n + 2) as f64;
        let w = (i - 0.5) * (i - 0.5) * (i - 0.5);
        assert!((g - w).abs() < 1e-9, "BFECC {got:?}");
    }
}

#[test]
fn semi_lagrangian_temperature_on_a_cubic_is_the_linear_interpolation() {
    // half a cell back: T_new(i) = (i^3 + (i - 1)^3) / 2
    let got = cubic_transport(AdvectionScheme::SemiLagrangian);
    for (n, g) in got.iter().enumerate() {
        let i = (n + 2) as f64;
        let w = (i * i * i + (i - 1.0) * (i - 1.0) * (i - 1.0)) / 2.0;
        assert!((g - w).abs() < 1e-9, "SL {got:?}");
    }
}

// ----------------------------------------------- wall model: summary envelope

#[test]
fn wall_summary_envelope_separates_a_fast_and_a_slow_wall_and_counts_every_face() {
    // channel between a wall at rest (relative speed 2) and a wall moving at 1.5
    // (relative speed 0.5), u = 2 everywhere; both in the log layer
    // (y+ = u_tau y_p / nu with y_p = 1/8, nu = 1e-6)
    let (nx, ny, nz) = (3usize, 4usize, 3usize);
    let dx = q(1, 4);
    let mut s = bare(nx, ny, nz, dx);
    s.dynamic_viscosity_pas = q(1, 1000);
    for k in 0..nz {
        for i in 0..nx {
            s.grid.set_v_bc(
                i,
                0,
                k,
                FaceBc::Wall {
                    velocity: Vec3Fix::ZERO,
                },
            );
            s.grid.set_v_bc(
                i,
                ny,
                k,
                FaceBc::Wall {
                    velocity: Vec3Fix::new(q(3, 2), Fix128::ZERO, Fix128::ZERO),
                },
            );
        }
    }
    for u in s.grid.u.iter_mut() {
        *u = int(2);
    }
    let opts = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 })
        .with_wall_model(WallModel::log_law());
    let report = s.step_with_options(q(1, 1000), &opts).expect("steps");
    let w = report.wall.expect("a wall model was enabled");
    // u faces next to a wall: (nx + 1) nz per wall; w faces next to a wall: nx (nz + 1) per wall (at rest)
    assert_eq!(w.faces, 2 * (nx + 1) * nz + 2 * nx * (nz + 1));
    assert_eq!(w.resting_faces, 2 * nx * (nz + 1));
    // the log law u / u_tau = ln(y+) / kappa + B holds at both ends of the envelope
    let nu = 1e-6;
    let y_p = 0.125;
    for (name, u_tau, y_plus, u_rel) in [
        ("slow", w.u_tau_min.to_f64(), w.y_plus_min.to_f64(), 0.5),
        ("fast", w.u_tau_max.to_f64(), w.y_plus_max.to_f64(), 2.0),
    ] {
        assert!(u_tau > 0.0, "{name}: u_tau {u_tau}");
        assert!(
            (y_plus - u_tau * y_p / nu).abs() < 1e-6 * y_plus,
            "{name}: y+ {y_plus} vs u_tau y / nu"
        );
        let u_plus = y_plus.ln() / 0.41 + 5.5;
        assert!(
            (u_tau * u_plus - u_rel).abs() < 1e-5 * u_rel,
            "{name}: u_tau {u_tau}, log law gives {}",
            u_tau * u_plus
        );
    }
    assert!(w.u_tau_min < w.u_tau_max && w.y_plus_min < w.y_plus_max);
}

// ----------------------------------------------- step_flip: boundary particles

#[test]
fn step_flip_lets_a_particle_on_the_far_boundary_take_part() {
    // doc: "one on the boundary does take part". A lattice that includes the planes
    // x, y, z = 0 and N dx: in free fall every particle, boundary ones included,
    // ends at u0 + g dt.
    let n = 3usize;
    let mut s = CfdSolver::new(n, n, n, int(1));
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.gravity = Vec3Fix::new(Fix128::ZERO, int(-8), Fix128::ZERO);
    let u0 = Vec3Fix::new(q(3, 2), int(1), q(-1, 2));
    let dt = q(1, 16);
    let mut ps = Vec::new();
    for a in 0..=2 * n as i64 {
        for b in 0..=2 * n as i64 {
            for c in 0..=2 * n as i64 {
                ps.push((Vec3Fix::new(q(a, 2), q(b, 2), q(c, 2)), u0));
            }
        }
    }
    s.step_flip(&mut ps, dt, q(1, 2));
    let want = Vec3Fix::new(u0.x, u0.y - int(8) * dt, u0.z);
    for (pos, vel) in &ps {
        assert!(
            (vel.x - want.x).abs() < q(1, 1_000_000_000)
                && (vel.y - want.y).abs() < q(1, 1_000_000_000)
                && (vel.z - want.z).abs() < q(1, 1_000_000_000),
            "particle {:?}: {:?} vs {:?}",
            (pos.x.to_f64(), pos.y.to_f64(), pos.z.to_f64()),
            (vel.x.to_f64(), vel.y.to_f64(), vel.z.to_f64()),
            (want.x.to_f64(), want.y.to_f64(), want.z.to_f64())
        );
    }
}

// ------------------------------------------------------------------ CSF

fn drop_solver() -> (CfdSolver, f64, f64) {
    // sphere of radius 8 dx in a 24^3 grid, dx = 1/16 (R = 0.5 m), as in
    // tests/analytic_csf_wiring.rs
    let n = 24usize;
    let dx = q(1, 16);
    let mut s = bare(n, n, n, dx);
    s.surface_tension_n_m = q(72, 1000);
    let mut ls = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    let c = int(12) * dx;
    initialize_level_set_sphere(&mut ls, c, c, c, int(8) * dx);
    s.level_set = Some(ls);
    (s, 0.5, 0.072)
}

fn csf_x_line(s: &mut CfdSolver) -> f64 {
    s.step_multigrid(q(1, 1000), 0);
    let dx = 1.0 / 16.0;
    let mut sum = 0.0;
    for i in 12..=24 {
        sum += s.grid.u(i, 12, 12).to_f64();
    }
    sum * 1000.0 / 0.001 * dx
}

#[test]
fn csf_x_line_impulse_points_inward_with_the_young_laplace_magnitude_to_15_percent() {
    // sum over the u faces right of the centre of rho/dt * u dx = - 2 sigma / R
    // (the band integral of f . n_out = sigma kappa, force toward the centre)
    let (mut s, r, sigma) = drop_solver();
    let line = csf_x_line(&mut s);
    let want = -2.0 * sigma / r;
    assert!(
        line < 0.0,
        "the force must point toward the centre of curvature"
    );
    assert!(
        (line - want).abs() < 0.15 * want.abs(),
        "line integral {line}, Young-Laplace {want}"
    );
}

#[test]
fn csf_x_line_impulse_matches_young_laplace_to_2_percent() {
    let (mut s, r, sigma) = drop_solver();
    let line = csf_x_line(&mut s);
    let want = -2.0 * sigma / r;
    assert!(
        (line - want).abs() < 0.02 * want.abs(),
        "line integral {line}, Young-Laplace {want}"
    );
}

#[test]
fn csf_y_and_z_line_impulses_equal_the_x_one_by_sphere_symmetry() {
    let (mut s, r, sigma) = drop_solver();
    s.step_multigrid(q(1, 1000), 0);
    let dx = 1.0 / 16.0;
    let want = -2.0 * sigma / r;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for a in 12..=24 {
        sy += s.grid.v(12, a, 12).to_f64();
        sz += s.grid.w(12, 12, a).to_f64();
    }
    let ly = sy * 1000.0 / 0.001 * dx;
    let lz = sz * 1000.0 / 0.001 * dx;
    assert!(
        (ly - want).abs() < 0.05 * want.abs(),
        "y line {ly}, want {want}"
    );
    assert!(
        (lz - want).abs() < 0.05 * want.abs(),
        "z line {lz}, want {want}"
    );
}

// -------------------------------------------------- reinitialisation cadence

#[test]
// AUD-A-S1W4-004
fn reinit_every_two_steps_reinitialises_at_the_end_of_the_second_step() {
    // a plane of slope 2 (not a signed distance) at rest; fast sweeping restores slope 1
    let (nx, ny, nz) = (7usize, 3usize, 3usize);
    let dx = int(1);
    let mut s = bare(nx, ny, nz, dx);
    s.reinit_every_n_steps = 2;
    let mut ls = Grid3d::new(nx, ny, nz, dx, Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                ls.set(i, j, k, int(2 * (i as i64 - 3)));
            }
        }
    }
    let before = ls.data.clone();
    s.level_set = Some(ls);
    s.step_multigrid(q(1, 100), 0);
    s.step_multigrid(q(1, 100), 0);
    assert_ne!(
        s.level_set.as_ref().unwrap().data,
        before,
        "after 2 steps with reinit_every_n_steps = 2 the level set is still the un-reinitialised one"
    );
}

#[test]
fn reinit_zero_means_never() {
    let (nx, ny, nz) = (7usize, 3usize, 3usize);
    let mut s = bare(nx, ny, nz, int(1));
    s.reinit_every_n_steps = 0;
    let mut ls = Grid3d::new(nx, ny, nz, int(1), Fix128::ZERO);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                ls.set(i, j, k, int(2 * (i as i64 - 3)));
            }
        }
    }
    let before = ls.data.clone();
    s.level_set = Some(ls);
    for _ in 0..6 {
        s.step_multigrid(q(1, 100), 0);
    }
    assert_eq!(s.level_set.as_ref().unwrap().data, before);
}

// --------------------------------------------------------- explicit diffusion

#[test]
// AUD-A-S1W4-005
fn step_with_options_refuses_or_bounds_an_unstable_explicit_diffusion_number() {
    let (nx, ny, nz) = (4usize, 8usize, 4usize);
    let mut s = bare(nx, ny, nz, int(1));
    s.density_kg_m3 = int(1);
    s.dynamic_viscosity_pas = q(6, 10); // nu dt / dx^2 = 0.6 with dt = 1
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let ix = i + (nx + 1) * (j + ny * k);
                s.grid.u[ix] = if j % 2 == 0 { int(1) } else { int(-1) };
            }
        }
    }
    let opts = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 });
    let before = s.grid.u.clone();
    match s.step_with_options(int(1), &opts) {
        Err(StepError::DiffusionUnstable { diffusion_number }) => {
            assert_eq!(diffusion_number, q(6, 10));
            assert_eq!(s.grid.u, before, "refused before anything changed");
        }
        Err(e) => panic!("unexpected refusal {e:?}"),
        Ok(_) => {
            let peak = s
                .grid
                .u
                .iter()
                .fold(0.0f64, |a, &u| a.max(u.to_f64().abs()));
            assert!(peak <= 1.0 + 1e-9, "peak grew to {peak}");
        }
    }
}

// ------------------------------------------------------------ degenerate grids

fn zero_extent_step(turb: bool, temp: bool, ls: bool) -> bool {
    // run in a thread so a hang (usize wrap-around loop) becomes a timeout, not a stuck suite
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(|| {
            let mut s = CfdSolver::new(0, 0, 0, int(1));
            s.use_turbulence = turb;
            if temp {
                s.temperature = Some(Grid3d::new(0, 0, 0, int(1), int(293)));
            }
            if ls {
                s.level_set = Some(Grid3d::new(0, 0, 0, int(1), Fix128::ZERO));
            }
            s.step(q(1, 100));
        });
        let _ = tx.send(r.is_ok());
    });
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or(false)
}

#[test]
fn a_zero_extent_grid_steps_without_panic() {
    assert!(zero_extent_step(false, false, false));
}

#[test]
fn a_zero_extent_grid_with_a_level_set_steps_without_panic() {
    assert!(zero_extent_step(false, false, true));
}

#[test]
fn a_zero_extent_grid_with_temperature_steps_without_panic() {
    assert!(zero_extent_step(false, true, false));
}

#[test]
fn a_zero_extent_grid_with_use_turbulence_steps_without_panic() {
    assert!(zero_extent_step(true, false, false));
}
