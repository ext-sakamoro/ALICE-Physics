#![no_main]
use alice_physics::cfd_solver::{AdvectionScheme, CfdSolver};
use alice_physics::math::Fix128;
use alice_physics::multiphase::Grid3d;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct CfdInput {
    /// Grid size: 4..=8 in each dimension (bounded to keep the fuzz corpus small).
    nx_raw: u8,
    ny_raw: u8,
    nz_raw: u8,
    /// Which advection scheme to exercise (mapped from 0..=2).
    scheme_raw: u8,
    /// Velocity magnitude on u-faces (i8, mapped to Fix128 as unit velocity components).
    u_seeds: Vec<i8>,
    /// Temperature seed values for the scalar field (i8, mapped to Fix128).
    temp_seeds: Vec<i8>,
    /// Number of simulation steps (capped at 8 to keep each fuzz iteration bounded).
    step_count: u8,
    /// dt in 1/N seconds (N in 30..=240 range for a realistic CFL window).
    dt_denom: u8,
}

// Fuzz the integrated CFD solver: build a small MAC grid + temperature scalar,
// seed both with arbitrary values, pick an advection scheme, and step. Must
// never panic — the solver must remain numerically robust even under seeded
// discontinuities and unusual grid sizes.
fuzz_target!(|input: CfdInput| {
    let nx = ((input.nx_raw % 5) + 4) as usize; // 4..=8
    let ny = ((input.ny_raw % 5) + 4) as usize;
    let nz = ((input.nz_raw % 5) + 4) as usize;

    let mut solver = CfdSolver::new(nx, ny, nz, Fix128::ONE);
    solver.advection_scheme = match input.scheme_raw % 3 {
        0 => AdvectionScheme::SemiLagrangian,
        1 => AdvectionScheme::MacCormack,
        _ => AdvectionScheme::Bfecc,
    };
    // Trim gravity to zero so pure advection is exercised.
    solver.gravity = alice_physics::math::Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);

    // Seed u-faces with arbitrary values (bounded via saturating_mul so extreme
    // ratios don't destabilise the CFL window catastrophically — the solver
    // itself must remain non-panicking regardless).
    for (idx, seed) in input.u_seeds.iter().enumerate().take(solver.grid.u.len()) {
        solver.grid.u[idx] = Fix128::from_ratio(*seed as i64, 10);
    }

    // Seed a temperature scalar field.
    let mut temp = Grid3d::new(nx, ny, nz, Fix128::ONE, Fix128::ZERO);
    for (idx, seed) in input.temp_seeds.iter().enumerate().take(temp.data.len()) {
        temp.data[idx] = Fix128::from_int(*seed as i64);
    }
    solver.temperature = Some(temp);

    let denom = (input.dt_denom % 211) as i64 + 30; // 30..=240
    let dt = Fix128::from_ratio(1, denom);
    let steps = (input.step_count as usize).min(8);

    for _ in 0..steps {
        solver.step(dt);
    }
});
