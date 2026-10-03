//! Material presets and failure-criterion evaluation of
//! `alice_physics::laminate_failure`.
//!
//! Drives the two strength presets (`LaminateStrengths::cfrp_ud` /
//! `gfrp_ud`) through all four in-plane failure criteria
//! (`FailureCriterion::{TsaiWu, TsaiHill, Hashin, Puck}`) at the zero stress
//! state (`StressState::zero`) and at a representative loaded state, both
//! via the `failure_index` dispatcher and via the four criterion functions
//! directly (`tsai_wu_failure_index`, `tsai_hill_failure_index`,
//! `hashin_failure_mode`, `puck_failure_mode`).
//!
//! ⚠️ **Why this example exists.** `scripts/wiring_guard.py` reported all
//! ten items above (`FailureCriterion`, `FailureMode`, `cfrp_ud`,
//! `failure_index`, `gfrp_ud`, `hashin_failure_mode`, `puck_failure_mode`,
//! `tsai_hill_failure_index`, `tsai_wu_failure_index`, `zero`) as unwired:
//! `tests/engineering_oracles_solid.rs` already exercises
//! `tsai_wu_failure_index` / `tsai_hill_failure_index` /
//! `hashin_failure_mode` / `puck_failure_mode` against the Jones (1999) /
//! Hashin (1980) closed forms, but tests do not count as production callers
//! for the wiring guard, and nothing in `src/` / `examples/` / `benches/`
//! called any of the ten before this file existed — in particular the
//! `failure_index` dispatcher and `StressState::zero` had no caller
//! anywhere. This example is that caller, and
//! `tests/analytic_laminate_failure_wiring.rs` holds the closed-form oracles
//! for the dispatcher, the presets, `zero` and two degenerate-input cases
//! (zero strengths, extreme/overflowing stress) that the existing test file
//! does not cover.
//!
//! ```text
//! cargo run --example laminate_failure_criteria --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example laminate_failure_criteria --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::laminate_failure::{
        failure_index, hashin_failure_mode, puck_failure_mode, tsai_hill_failure_index,
        tsai_wu_failure_index, FailureCriterion, FailureMode, LaminateStrengths, StressState,
    };
    use alice_physics::math::Fix128;

    let presets: [(&str, LaminateStrengths); 2] = [
        ("cfrp_ud", LaminateStrengths::cfrp_ud()),
        ("gfrp_ud", LaminateStrengths::gfrp_ud()),
    ];

    for (name, s) in presets {
        println!(
            "[laminate_failure] preset {name}: Xt={:.3} Xc={:.3} Yt={:.3} Yc={:.3} S={:.3} MPa",
            s.xt.to_f64(),
            s.xc.to_f64(),
            s.yt.to_f64(),
            s.yc.to_f64(),
            s.s.to_f64()
        );

        let zero = StressState::zero();
        println!(
            "[laminate_failure]   zero stress ({:.1},{:.1},{:.1}): TsaiWu={:.6} TsaiHill={:.6} Hashin={:.6} Puck={:.6}",
            zero.sigma_1.to_f64(),
            zero.sigma_2.to_f64(),
            zero.tau_12.to_f64(),
            failure_index(FailureCriterion::TsaiWu, s, zero).to_f64(),
            failure_index(FailureCriterion::TsaiHill, s, zero).to_f64(),
            failure_index(FailureCriterion::Hashin, s, zero).to_f64(),
            failure_index(FailureCriterion::Puck, s, zero).to_f64(),
        );

        // A representative loaded state: half the fibre-tension strength
        // plus a third of the transverse-tension strength and a quarter of
        // the shear strength, biaxial + shear all at once so every term of
        // every criterion's formula is exercised (unlike zero stress).
        let loaded = StressState {
            sigma_1: s.xt.half(),
            sigma_2: s.yt / Fix128::from_int(3),
            tau_12: s.s / Fix128::from_int(4),
        };
        let fi_tsai_wu = tsai_wu_failure_index(s, loaded);
        let fi_tsai_hill = tsai_hill_failure_index(s, loaded);
        let mode_hashin = hashin_failure_mode(s, loaded);
        let mode_puck = puck_failure_mode(s, loaded);
        println!(
            "[laminate_failure]   loaded stress ({:.3},{:.3},{:.3}): TsaiWu={:.6} TsaiHill={:.6} Hashin={:?} Puck={:?}",
            loaded.sigma_1.to_f64(),
            loaded.sigma_2.to_f64(),
            loaded.tau_12.to_f64(),
            fi_tsai_wu.to_f64(),
            fi_tsai_hill.to_f64(),
            mode_hashin,
            mode_puck,
        );

        // Dispatcher must agree with the direct criterion calls above.
        let dispatch_tsai_wu = failure_index(FailureCriterion::TsaiWu, s, loaded);
        let dispatch_tsai_hill = failure_index(FailureCriterion::TsaiHill, s, loaded);
        let dispatch_hashin = failure_index(FailureCriterion::Hashin, s, loaded);
        let dispatch_puck = failure_index(FailureCriterion::Puck, s, loaded);
        let hashin_as_index = if mode_hashin == FailureMode::Safe {
            Fix128::ZERO
        } else {
            Fix128::ONE
        };
        let puck_as_index = if mode_puck == FailureMode::Safe {
            Fix128::ZERO
        } else {
            Fix128::ONE
        };
        println!(
            "[laminate_failure]   dispatcher vs direct: TsaiWu match={} TsaiHill match={} Hashin match={} Puck match={}",
            dispatch_tsai_wu == fi_tsai_wu,
            dispatch_tsai_hill == fi_tsai_hill,
            dispatch_hashin == hashin_as_index,
            dispatch_puck == puck_as_index,
        );
    }
}
