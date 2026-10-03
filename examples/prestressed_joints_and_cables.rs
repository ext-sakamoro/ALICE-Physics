//! Bolted-joint preload (Motosh / VDI 2230) and pretensioned-cable stiffness
//! — the production entry point for every public item in `prestressed`
//! (`preload_from_torque`, `recommended_preload_n`, `bolt_load_fraction`,
//! `bolt_peak_tension`, `separation_load_n`, `cable_pretension_n`,
//! `tensioned_cable_stiffness_n_per_mm`).
//!
//! Each section prints the closed-form value next to the function's output.
//!
//! ```bash
//! cargo run --example prestressed_joints_and_cables --features std
//! ```
//!
//! # Closed forms (derived in the module doc, `src/prestressed.rs`)
//!
//! * **Preload from torque** (Motosh): `F_i = T / (K · d)`.
//! * **Recommended preload**: `F_i = S_p · A_t · fraction` (Shigley §8-8).
//! * **Bolt load fraction**: `C = k_b / (k_b + k_m)`.
//! * **Peak bolt tension**: `F_b = F_i + C · P_ext`.
//! * **Separation load**: `P_sep = F_i / (1 − C)`.
//! * **Cable pretension**: `T ≈ w · L² / (8 · s)` (parabolic approximation).
//! * **Tensioned cable stiffness**: `k = 8 · T / L`.

use alice_physics::math::Fix128;
use alice_physics::prestressed::{
    bolt_load_fraction, bolt_peak_tension, cable_pretension_n, preload_from_torque,
    recommended_preload_n, separation_load_n, tensioned_cable_stiffness_n_per_mm,
};

fn main() {
    // ------------------------------------------------------------------
    // A known M8-class bolt: T = 10 N·m, K = 0.2 (dry steel-on-steel),
    // d = 8 mm, S_p = 800 MPa, A_t = 20 mm², installation fraction 0.75.
    // ------------------------------------------------------------------
    let torque_nm = Fix128::from_int(10);
    let nut_factor_k = Fix128::from_ratio(2, 10);
    let diameter_mm = Fix128::from_int(8);
    let proof_strength_mpa = Fix128::from_int(800);
    let tensile_stress_area_mm2 = Fix128::from_int(20);
    let fraction = Fix128::from_ratio(75, 100);

    let preload = preload_from_torque(torque_nm, nut_factor_k, diameter_mm);
    println!(
        "[prestressed] preload_from_torque(T=10, K=0.2, d=8mm) = {} N  (closed form: 10/(0.2*0.008) = 6250 N)",
        preload.to_f64()
    );

    let recommended = recommended_preload_n(proof_strength_mpa, tensile_stress_area_mm2, fraction);
    println!(
        "[prestressed] recommended_preload_n(Sp=800, At=20, 0.75) = {} N  (closed form: 800*20*0.75 = 12000 N)",
        recommended.to_f64()
    );

    // Joint stiffness: a stiff bolt (k_b = 500,000 N/mm) through a softer
    // clamped flange stack (k_m = 300,000 N/mm).
    let k_bolt = Fix128::from_int(500_000);
    let k_member = Fix128::from_int(300_000);
    let c = bolt_load_fraction(k_bolt, k_member);
    println!(
        "[prestressed] bolt_load_fraction(500000, 300000) = {}  (closed form: 500000/800000 = 0.625)",
        c.to_f64()
    );

    // Peak bolt tension under a 4000 N external working load.
    let p_ext = Fix128::from_int(4000);
    let f_b = bolt_peak_tension(preload, c, p_ext);
    println!(
        "[prestressed] bolt_peak_tension(F_i=6250, C=0.625, P_ext=4000) = {} N  (closed form: 6250+0.625*4000 = 8750 N)",
        f_b.to_f64()
    );

    // Degenerate: zero external load must read back the preload exactly.
    let f_b_at_rest = bolt_peak_tension(preload, c, Fix128::ZERO);
    println!(
        "[prestressed] bolt_peak_tension(.., P_ext=0) = {} N  (must equal preload exactly: {})",
        f_b_at_rest.to_f64(),
        f_b_at_rest == preload
    );

    let p_sep = separation_load_n(preload, c);
    println!(
        "[prestressed] separation_load_n(F_i=6250, C=0.625) = {} N  (closed form: 6250/0.375 = 16666.67 N)",
        p_sep.to_f64()
    );

    // ------------------------------------------------------------------
    // A known suspension cable: w = 0.02 N/mm, L = 10,000 mm, s = 200 mm.
    // ------------------------------------------------------------------
    let w_n_per_mm = Fix128::from_ratio(2, 100);
    let span_mm = Fix128::from_int(10_000);
    let sag_mm = Fix128::from_int(200);

    let tension = cable_pretension_n(w_n_per_mm, span_mm, sag_mm);
    println!(
        "[prestressed] cable_pretension_n(w=0.02, L=10000, s=200) = {} N  (closed form: 0.02*10000^2/(8*200) = 1250 N)",
        tension.to_f64()
    );

    let stiffness = tensioned_cable_stiffness_n_per_mm(tension, span_mm);
    println!(
        "[prestressed] tensioned_cable_stiffness_n_per_mm(T=1250, L=10000) = {} N/mm  (closed form: 8*1250/10000 = 1 N/mm)",
        stiffness.to_f64()
    );

    // Degenerate: zero sag is a degenerate parabola — the module returns a
    // large sentinel (i64::MAX >> 8) rather than Err or a panic.
    let infinite_tension = cable_pretension_n(w_n_per_mm, span_mm, Fix128::ZERO);
    let sentinel = Fix128::from_int(i64::MAX >> 8);
    println!(
        "[prestressed] cable_pretension_n(.., s=0) = {} N  (sentinel i64::MAX>>8 = {})",
        infinite_tension.to_f64(),
        infinite_tension == sentinel
    );
}
