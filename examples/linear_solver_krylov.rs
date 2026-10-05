//! General Krylov solvers on `Fix128`: GMRES(m), BiCGStab, preconditioning and
//! the block equilibration a coupled system must go through
//!
//! Three scenes:
//!
//! 1. **Steady advection–diffusion** `−u'' + Pe·u' = 0`, `u(0) = 0`,
//!    `u(1) = 1`, central differences on 32 cells. The operator is
//!    non-symmetric, so the crate's conjugate gradients do not apply. The table
//!    compares GMRES (full and restarted) and BiCGStab, with and without a
//!    Jacobi preconditioner, against the discrete closed form
//!    `u_j = (1 − ρ^j)/(1 − ρ^N)`, `ρ = (1 + P)/(1 − P)`.
//! 2. **Block Jacobi** on a block-diagonal matrix: the exact inverse, so one
//!    iteration.
//! 3. **A PLA thermo-elastic block pair** (`E·h = 3.5e6`, `k·h = 1.3e-4`): the
//!    explicitly unscaled entry refuses it because the thermal block would
//!    vanish from every inner product, and the equilibrated solve returns both
//!    blocks.
//!
//! Run with `cargo run --example linear_solver_krylov`.

use alice_physics::linear_solver::{
    bicgstab, gmres, solve_equilibrated, BlockEquilibration, BlockJacobiPreconditioner, BlockScale,
    BreakdownKind, DenseMatrix, FnOperator, IdentityPreconditioner, JacobiPreconditioner,
    KrylovConfig, KrylovConfigFault, KrylovMethod, KrylovSolution, LinearOperator,
    LinearSolverError, BREAKDOWN_RELATIVE,
};
use alice_physics::math::Fix128;

const CELLS: usize = 32;

fn tolerance() -> Fix128 {
    // 2⁻⁴⁰
    Fix128::from_raw(0, 1 << 24)
}

fn advection_diffusion(pe: i64) -> Result<(DenseMatrix, Vec<Fix128>), LinearSolverError> {
    let n = CELLS - 1;
    let p = Fix128::from_ratio(pe, 2 * CELLS as i64);
    let mut entries = vec![Fix128::ZERO; n * n];
    for i in 0..n {
        entries[i * n + i] = Fix128::from_int(2);
        if i > 0 {
            entries[i * n + i - 1] = Fix128::NEG_ONE - p;
        }
        if i + 1 < n {
            entries[i * n + i + 1] = Fix128::NEG_ONE + p;
        }
    }
    let mut b = vec![Fix128::ZERO; n];
    b[n - 1] = Fix128::ONE - p;
    Ok((DenseMatrix::try_new(n, entries)?, b))
}

fn closed_form(pe: i64) -> Vec<f64> {
    let p = pe as f64 / (2 * CELLS) as f64;
    let rho = (1.0 + p) / (1.0 - p);
    let pow = |k: usize| (0..k).fold(1.0f64, |acc, _| acc * rho);
    (1..CELLS)
        .map(|j| (1.0 - pow(j)) / (1.0 - pow(CELLS)))
        .collect()
}

fn max_error(x: &[Fix128], reference: &[f64]) -> f64 {
    x.iter()
        .zip(reference)
        .map(|(a, b)| (a.to_f64() - b).abs())
        .fold(0.0, f64::max)
}

fn row(name: &str, run: Result<KrylovSolution, LinearSolverError>, reference: &[f64]) {
    match run {
        Ok(sol) => println!(
            "  {name:<18} {:>5} {:>8} {:>12.3e} {:>12.3e}",
            sol.stats.iterations,
            sol.stats.restarts,
            (sol.stats.residual_norm / sol.stats.rhs_norm).to_f64(),
            max_error(&sol.x, reference)
        ),
        Err(e) => println!("  {name:<18} {e}"),
    }
}

fn main() -> Result<(), LinearSolverError> {
    println!("1. advection-diffusion, {CELLS} cells, relative tolerance 2^-40");
    println!(
        "  {:<18} {:>5} {:>8} {:>12} {:>12}",
        "method", "iters", "restarts", "rel resid", "max |u-u_h|"
    );
    for pe in [1i64, 10, 40, 60] {
        println!(
            " Pe = {pe} (cell Peclet {:.4})",
            pe as f64 / (2 * CELLS) as f64
        );
        let (a, b) = advection_diffusion(pe)?;
        let n = a.dim();
        let reference = closed_form(pe);
        let identity = IdentityPreconditioner::new(n);
        let jacobi = JacobiPreconditioner::from_dense(&a)?;
        let full = KrylovConfig::try_new(500, tolerance())?.with_restart(n)?;
        let restarted = KrylovConfig::try_new(5000, tolerance())?.with_restart(8)?;
        let bicg = KrylovConfig::try_new(500, tolerance())?.with_stagnation_window(64)?;
        row("GMRES full", gmres(&a, &identity, &b, &full), &reference);
        row("GMRES(8)", gmres(&a, &identity, &b, &restarted), &reference);
        row("GMRES + Jacobi", gmres(&a, &jacobi, &b, &full), &reference);
        row("BiCGStab", bicgstab(&a, &identity, &b, &bicg), &reference);
        row(
            "BiCGStab + Jacobi",
            bicgstab(&a, &jacobi, &b, &bicg),
            &reference,
        );
    }

    // The same operator as a closure, stopped early by an absolute tolerance.
    let (a, b) = advection_diffusion(10)?;
    let op = FnOperator::new(a.dim(), |x: &[Fix128], y: &mut [Fix128]| a.apply(x, y));
    let loose = KrylovConfig::try_new(500, tolerance())?
        .with_absolute_tolerance(Fix128::from_ratio(1, 1000))?;
    let early = gmres(&op, &IdentityPreconditioner::new(op.dim()), &b, &loose)?;
    println!(
        "  closure operator, absolute tolerance 1e-3: {} iterations, residual {:.3e}, entry (0, 0) = {}",
        early.stats.iterations,
        early.stats.residual_norm.to_f64(),
        a.get(0, 0)
    );

    println!("\n2. block Jacobi on a block-diagonal matrix (blocks 3, 4, 2)");
    let sizes = [3usize, 4, 2];
    let n: usize = sizes.iter().sum();
    let mut block_of = Vec::new();
    for (k, &s) in sizes.iter().enumerate() {
        block_of.extend(std::iter::repeat_n(k, s));
    }
    let mut entries = vec![Fix128::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            if block_of[i] == block_of[j] {
                let scale = if i == j { 6 } else { 1 };
                entries[i * n + j] = Fix128::from_int(scale * (block_of[i] as i64 + 1))
                    + Fix128::from_ratio(i as i64 - j as i64, 4);
            }
        }
    }
    let blocky = DenseMatrix::try_new(n, entries)?;
    let rhs: Vec<Fix128> = (0..n)
        .map(|i| Fix128::from_ratio(i as i64 + 1, 3))
        .collect();
    let block_jacobi = BlockJacobiPreconditioner::from_dense(&blocky, &sizes)?;
    let config = KrylovConfig::try_new(100, tolerance())?;
    let plain = gmres(&blocky, &IdentityPreconditioner::new(n), &rhs, &config)?;
    let one = gmres(&blocky, &block_jacobi, &rhs, &config)?;
    println!(
        "  GMRES iterations: plain {}, block Jacobi {}",
        plain.stats.iterations, one.stats.iterations
    );

    println!("\n3. PLA thermo-elastic block pair (E h = 3.5e6, k h = 1.3e-4)");
    let magnitudes = [Fix128::from_f64(3.5e6), Fix128::from_f64(1.3e-4)];
    match BlockEquilibration::unscaled(&[4, 4], &magnitudes) {
        Err(LinearSolverError::BlockBelowProductFloor {
            block,
            relative_magnitude,
        }) => println!(
            "  unscaled: refused, block {block} is {:.3e} of the largest (floor 2^-32 = 2.33e-10)",
            relative_magnitude.to_f64()
        ),
        other => println!("  unscaled: {other:?}"),
    }
    let eq = BlockEquilibration::try_new(&[4, 4], &magnitudes)?;
    for block in 0..eq.block_count() {
        match eq.scale(block) {
            Some(BlockScale::Down(s)) => println!("  block {block}: d = 2^-{}", s.exponent()),
            Some(BlockScale::Up(s)) => println!("  block {block}: d = 2^{}", s.exponent()),
            None => {}
        }
    }
    let m = eq.dim();
    let nb = m / 2;
    let alpha = Fix128::from_raw(0, 1 << 50); // 2⁻¹⁴ /K
    let mut entries = vec![Fix128::ZERO; m * m];
    for i in 0..nb {
        for (offset, magnitude) in [(0, magnitudes[0]), (nb, magnitudes[1])] {
            entries[(offset + i) * m + offset + i] = magnitude * Fix128::from_int(2);
            if i > 0 {
                entries[(offset + i) * m + offset + i - 1] = -magnitude;
            }
            if i + 1 < nb {
                entries[(offset + i) * m + offset + i + 1] = -magnitude;
            }
        }
        entries[i * m + nb + i] = -(magnitudes[0] * alpha);
    }
    let coupled = DenseMatrix::try_new(m, entries)?;
    let truth: Vec<Fix128> = (0..m)
        .map(|i| Fix128::from_ratio(i as i64 % nb as i64 + 1, 4))
        .collect();
    let mut b = vec![Fix128::ZERO; m];
    coupled.apply(&truth, &mut b);
    let diagonal: Vec<Fix128> = (0..m).map(|i| coupled.get(i, i)).collect();
    let jacobi = JacobiPreconditioner::from_diagonal(eq.equilibrate_diagonal(&diagonal)?)?;
    let config = KrylovConfig::try_new(200, tolerance())?.with_restart(m)?;
    for method in [KrylovMethod::Gmres, KrylovMethod::BiCgStab] {
        let sol = solve_equilibrated(&coupled, &eq, &jacobi, &b, method, &config)?;
        let err = |range: std::ops::Range<usize>| {
            sol.x[range.clone()]
                .iter()
                .zip(&truth[range])
                .map(|(x, t)| (x.to_f64() - t.to_f64()).abs())
                .fold(0.0, f64::max)
        };
        println!(
            "  {method:?}: {} iterations, max error u {:.2e}, T {:.2e}",
            sol.stats.iterations,
            err(0..nb),
            err(nb..m)
        );
    }

    // Failure verdicts are values, not panics.
    let rotation = DenseMatrix::try_new(
        2,
        vec![Fix128::ZERO, Fix128::ONE, Fix128::NEG_ONE, Fix128::ZERO],
    )?;
    let swap = DenseMatrix::try_new(
        2,
        vec![Fix128::ZERO, Fix128::ONE, Fix128::ONE, Fix128::ZERO],
    )?;
    let e0 = [Fix128::ONE, Fix128::ZERO];
    let gmres1 = KrylovConfig::try_new(100, tolerance())?.with_restart(1)?;
    println!(
        "\nfailure verdicts (breakdown threshold {:.2e} relative):",
        BREAKDOWN_RELATIVE.to_f64()
    );
    println!(
        "  GMRES(1) on a rotation: {:?}",
        gmres(&rotation, &IdentityPreconditioner::new(2), &e0, &gmres1).err()
    );
    let swap_result = bicgstab(&swap, &IdentityPreconditioner::new(2), &e0, &config);
    if let Err(LinearSolverError::Breakdown {
        kind: BreakdownKind::DirectionOrthogonal,
        ..
    }) = swap_result
    {
        println!("  BiCGStab on [[0,1],[1,0]]: breakdown, (r~, A p) = 0 at the first step");
    }
    if let Err(LinearSolverError::InvalidConfig(KrylovConfigFault::ZeroRestart)) =
        KrylovConfig::try_new(10, tolerance())?.with_restart(0)
    {
        println!("  restart 0: refused");
    }
    Ok(())
}
