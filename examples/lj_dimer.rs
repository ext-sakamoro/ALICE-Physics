//! A Lennard-Jones dimer and a small periodic LJ gas integrated with velocity
//! Verlet, plus the other pair laws (Morse, Coulomb, Yukawa) and the
//! Lorentz–Berthelot rule, each checked against its closed form.
//!
//! ```text
//! U_LJ = 4ε[(σ/r)¹² − (σ/r)⁶],  r_min = 2^{1/6} σ,  k = U''(r_min) = 72 ε / r_min²
//! dimer period (velocity Verlet):  T_d = 2π h / arccos(1 − (ωh)²/2),  ω = √(k/μ)
//! ```
//!
//! All quantities are in reduced units (`ε = σ = k_B = 1`) except the
//! Coulomb line, which uses SI charges and the crate's Coulomb constant.
//!
//! ```bash
//! cargo run --release --example lj_dimer
//! ```

// f64 `powf` / `acos` / `exp` compute closed-form references, not state.
#![allow(clippy::disallowed_methods)]

use alice_physics::electromagnetic::EmSource;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{
    pair_forces_all_pairs, pair_forces_cell_list, MdError, PeriodicBox, VelocityVerlet,
};
use alice_physics::pair_potential::{
    lorentz_berthelot, Coulomb, LennardJones, Morse, PairPotential, PairPotentialError, ShiftMode,
    Truncated, Yukawa, COULOMB_CONSTANT,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn main() -> Result<(), MdError> {
    let lj = LennardJones::new(Fix128::ONE, Fix128::ONE).map_err(pot_err)?;
    let r_min = 2f64.powf(1.0 / 6.0);
    println!(
        "LJ (ε = {}, σ = {}): U(r_min) = {:.12} (closed form −1), F(r_min) = {:.2e}",
        lj.epsilon(),
        lj.sigma(),
        lj.energy(fx(r_min)).map_err(pot_err)?.to_f64(),
        lj.force(fx(r_min)).map_err(pot_err)?.to_f64(),
    );

    // --- dimer -----------------------------------------------------------
    let (m1, m2) = (1.0, 3.0);
    let mu = m1 * m2 / (m1 + m2);
    let k = 72.0 / (r_min * r_min);
    let h = 1e-3;
    let omega = (k / mu).sqrt();
    let t_d = 2.0 * std::f64::consts::PI * h / (1.0 - (omega * h).powi(2) / 2.0).acos();
    let potential = Truncated::new(lj, fx(2.5), ShiftMode::EnergyShift).map_err(pot_err)?;
    let a = 1e-4;
    let mut dimer = VelocityVerlet::new(
        potential,
        PeriodicBox::cubic(Fix128::from_int(10))?,
        vec![
            Vec3Fix::new(fx(4.0), fx(5.0), fx(5.0)),
            Vec3Fix::new(fx(4.0 + r_min + a), fx(5.0), fx(5.0)),
        ],
        vec![Vec3Fix::ZERO; 2],
        vec![fx(m1), fx(m2)],
    )?;
    let separation = |md: &VelocityVerlet<LennardJones>| {
        let p = md.positions();
        md.periodic_box()
            .minimum_image(p[1] - p[0])
            .length()
            .to_f64()
            - r_min
    };
    let mut prev = separation(&dimer);
    let mut crossings = Vec::new();
    for step in 1..=8000 {
        dimer.step(fx(h))?;
        let cur = separation(&dimer);
        if prev < 0.0 && cur >= 0.0 {
            crossings.push((step as f64 - 1.0 + (-prev / (cur - prev))) * h);
        }
        prev = cur;
    }
    let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    println!(
        "dimer (μ = {mu}, k = {k:.4}, cutoff {} {:?}): period {period:.9} vs T_d {t_d:.9} (rel {:.1e}, anharmonic shift ≈ 1.8e-7)",
        dimer.potential().cutoff(),
        dimer.potential().mode(),
        (period - t_d) / t_d
    );
    assert!((period - t_d).abs() < 1e-6 * t_d);
    assert_eq!(dimer.masses().len(), 2);
    assert_eq!(*dimer.potential().inner(), lj);

    // --- periodic gas: NVE, cell list = all pairs -------------------------
    let bx = PeriodicBox::new(Vec3Fix::new(fx(5.4), fx(5.4), fx(5.4)))?;
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    for i in 0..27usize {
        let (x, y, z) = (i % 3, (i / 3) % 3, i / 9);
        let c = |b: usize, j: f64| fx((b as f64 + 0.5) * 1.8 + 0.07 * j);
        pos.push(Vec3Fix::new(c(x, 1.0), c(y, -0.5), c(z, 0.3)));
        let s = if i % 2 == 0 { 1.0 } else { -1.0 };
        vel.push(Vec3Fix::new(fx(0.4 * s), fx(-0.3 * s), fx(0.2 * s)));
    }
    // 27 is odd: cancel the momentum of the extra +s particle
    vel[0] = Vec3Fix::ZERO;
    let shifted = Truncated::new(lj, fx(2.5), ShiftMode::ForceShift).map_err(pot_err)?;
    let cell = pair_forces_cell_list(&shifted, &bx, &pos)?;
    let all = pair_forces_all_pairs(&shifted, &bx, &pos)?;
    assert_eq!(cell, all);
    let mut gas = VelocityVerlet::new(shifted, bx, pos, vel, vec![Fix128::ONE; 27])?;
    let e0 = gas.total_energy();
    let p0 = gas.momentum();
    for _ in 0..200 {
        gas.step(fx(0.002))?;
    }
    println!(
        "gas (27 particles, box {}): E {:.6} -> {:.6}, U = {:.4}, K = {:.4}, T = {:.4}, |ΔP| = {:.1e}, |ΣF| = {}",
        bx.lengths().x,
        e0.to_f64(),
        gas.total_energy().to_f64(),
        gas.potential_energy().to_f64(),
        gas.kinetic_energy().to_f64(),
        gas.instantaneous_temperature(Fix128::ONE)?.to_f64(),
        (gas.momentum() - p0).length().to_f64(),
        gas.forces().iter().fold(Vec3Fix::ZERO, |s, f| s + *f).length()
    );
    let wrapped = bx.wrap(Vec3Fix::new(fx(-0.4), fx(6.0), fx(1.0)));
    assert!(gas.velocities().len() == 27 && wrapped.x > Fix128::ZERO);

    // --- other laws --------------------------------------------------------
    let morse = Morse::new(fx(2.5), fx(1.8), fx(1.2)).map_err(pot_err)?;
    println!(
        "Morse: U(r_e) = {} (closed form −D = −2.5), 2Da² = {:.4}",
        morse.energy(fx(1.2)).map_err(pot_err)?,
        morse.harmonic_force_constant().to_f64()
    );

    let (qi, qj) = (fx(1.5e-6), fx(-0.75e-6));
    let (xi, xj) = (
        Vec3Fix::new(fx(0.4), fx(-0.2), fx(1.1)),
        Vec3Fix::new(fx(-0.3), fx(0.5), fx(0.2)),
    );
    let pair = Coulomb::new(qi, qj)
        .force_on_first(xi - xj)
        .map_err(pot_err)?;
    let field = EmSource::PointCharge {
        position: xj,
        charge_c: qj,
    }
    .sample(xi)
    .0 * qi;
    println!(
        "Coulomb (k = {COULOMB_CONSTANT}): pair force x {:.6e} vs q_i E_j x {:.6e}",
        pair.x.to_f64(),
        field.x.to_f64()
    );

    let reduced = Coulomb::with_constant(Fix128::ONE, Fix128::ONE, Fix128::ONE).map_err(pot_err)?;
    let screened =
        Yukawa::with_constant(Fix128::ONE, Fix128::ONE, Fix128::ONE, fx(0.7)).map_err(pot_err)?;
    let si_screened = Yukawa::new(qi, qj, fx(1e-3)).map_err(pot_err)?;
    println!(
        "Yukawa (λ = 0.7): U(1) = {:.9} (closed form e^(−1/0.7) = {:.9}), Coulomb U(1) = {}, SI U(1 mm) = {:.3e}",
        screened.energy(Fix128::ONE).map_err(pot_err)?.to_f64(),
        (-1.0f64 / 0.7).exp(),
        reduced.energy(Fix128::ONE).map_err(pot_err)?,
        si_screened.energy(fx(1e-3)).map_err(pot_err)?.to_f64()
    );

    let mixed = lorentz_berthelot(
        &LennardJones::new(Fix128::ONE, Fix128::ONE).map_err(pot_err)?,
        &LennardJones::new(Fix128::from_int(4), Fix128::from_int(2)).map_err(pot_err)?,
    );
    println!(
        "Lorentz–Berthelot: σ = {} (closed form 1.5), ε = {} (closed form 2)",
        mixed.sigma(),
        mixed.epsilon()
    );
    Ok(())
}

fn pot_err(error: PairPotentialError) -> MdError {
    println!("pair potential error: {error}");
    MdError::Potential { i: 0, j: 0, error }
}
