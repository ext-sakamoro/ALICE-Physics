//! Closed-form oracle for first-order exponential decay of a reactant
//! (COV-CHEM-102). `ScalarField3D::decay` multiplies every grid cell by
//! `exp(-rate * dt)` each call, which is exactly isothermal first-order
//! kinetics `dC/dt = -k C` -> `C(t) = C0 exp(-k t)` (Turns ch.4) when `rate`
//! is read as the rate constant `k`; nothing compared it to that closed
//! form or checked the half-life `ln(2) / k` before this test (see
//! docs/coverage/chem.toml COV-CHEM-102 evidence, which also notes
//! `src/smoke_fire.rs`'s Arrhenius reaction rate is second order in the
//! fuel and oxidizer densities, not the simple first-order law this row
//! asks for, so `ScalarField3D::decay` rather than `smoke_fire.rs` is the
//! matching implementation).

use alice_physics::det_math;
use alice_physics::ScalarField3D;

fn make_field(c0: f32) -> ScalarField3D {
    let mut field = ScalarField3D::new(2, 2, 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    field.data.fill(c0);
    field
}

#[test]
fn concentration_follows_first_order_closed_form_c0_exp_minus_kt() {
    let c0 = 10.0_f32;
    let k = 0.5_f32; // rate constant, 1/s
    let dt = 0.001_f32;
    let mut field = make_field(c0);

    let checkpoints_s = [1.0_f32, 2.0, 4.0, 8.0];
    let mut t = 0.0_f32;
    let mut next_checkpoint = 0;
    let total_steps = (checkpoints_s[checkpoints_s.len() - 1] / dt).round() as usize;

    for step in 0..=total_steps {
        if next_checkpoint < checkpoints_s.len()
            && (t - checkpoints_s[next_checkpoint]).abs() < dt / 2.0
        {
            let expected = c0 * det_math::exp(-k * t);
            for &c in &field.data {
                let rel_err = (c - expected).abs() / c0;
                assert!(
                    rel_err < 1e-3,
                    "t={t}: concentration {c} expected {expected} (rel err {rel_err})"
                );
            }
            next_checkpoint += 1;
        }
        if step < total_steps {
            field.decay(k, dt);
            t += dt;
        }
    }
    assert_eq!(
        next_checkpoint,
        checkpoints_s.len(),
        "not all checkpoints were hit"
    );
}

#[test]
fn half_life_matches_ln2_over_k() {
    let c0 = 10.0_f32;
    let k = 0.25_f32;
    let dt = 0.0005_f32;
    let half_life = det_math::ln(2.0) / k;

    let mut field = make_field(c0);
    let mut t = 0.0_f32;
    while t < half_life {
        field.decay(k, dt);
        t += dt;
    }

    let got = field.data[0];
    let rel_err = (got - c0 / 2.0).abs() / c0;
    assert!(
        rel_err < 1e-3,
        "concentration at t=half_life={half_life}: got {got} expected {} (rel err {rel_err})",
        c0 / 2.0
    );
}
