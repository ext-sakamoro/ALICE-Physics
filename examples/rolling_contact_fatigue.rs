//! Hertzian rolling-contact stress + Basquin fatigue life of
//! `alice_physics::rolling_contact`.
//!
//! Drives the three material presets (`materials::bearing_steel_52100`,
//! `materials::gear_steel_8620`, `materials::silicon_nitride`) through
//! `hertzian_sphere_sphere` (self-mated ball-on-ball contact), then feeds
//! the resulting peak pressure into `basquin_cycles_to_failure` both
//! directly and via the `rolling_contact_life_cycles` convenience wrapper,
//! and separately exercises a ball-on-flat-rail contact
//! (`radius_2_m = f32::INFINITY`, the degenerate-radius usage documented on
//! `hertzian_sphere_sphere`).
//!
//! ⚠️ **Why this example exists.** `scripts/wiring_guard.py` reported all
//! seven items above (`HertzianContact`, `basquin_cycles_to_failure`,
//! `bearing_steel_52100`, `gear_steel_8620`, `hertzian_sphere_sphere`,
//! `rolling_contact_life_cycles`, `silicon_nitride`) as unwired: the
//! module's own `#[cfg(test)]` block exercises the formulas thoroughly, but
//! tests do not count as production callers for the wiring guard, and
//! nothing in `src/` / `examples/` / `benches/` called any of the seven
//! before this file existed. This example is that caller, and
//! `tests/analytic_rolling_contact_wiring.rs` holds the closed-form oracles
//! and degenerate-input panic tests for the cases the module's own unit
//! tests do not cover (flat-rail / infinite-radius contact, the
//! `rolling_contact_life_cycles` wiring itself, and overflow behaviour).
//!
//! ```text
//! cargo run --example rolling_contact_fatigue --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example rolling_contact_fatigue --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::rolling_contact::{
        basquin_cycles_to_failure, hertzian_sphere_sphere, materials, rolling_contact_life_cycles,
        HertzianContact,
    };

    /// `(E [Pa], nu, Basquin C, Basquin m, endurance limit [Pa])`, same
    /// shape as `materials::*`'s return type.
    type MaterialPreset = (f32, f32, f32, f32, f32);

    // Self-mated ball-on-ball contact for each preset: identical 8 mm
    // radius balls of the same material pressed together at 1000 N.
    let presets: [(&str, MaterialPreset); 3] = [
        ("bearing_steel_52100", materials::bearing_steel_52100()),
        ("gear_steel_8620", materials::gear_steel_8620()),
        ("silicon_nitride", materials::silicon_nitride()),
    ];

    let load_n = 1000.0_f32;
    let radius_m = 8.0e-3_f32;

    for (name, (e, nu, basquin_c, basquin_m, endurance_pa)) in presets {
        let contact: HertzianContact =
            hertzian_sphere_sphere(load_n, radius_m, radius_m, e, e, nu, nu);
        println!(
            "[rolling_contact] preset {name}: E={e:.3e} Pa nu={nu:.3} C={basquin_c:.3e} m={basquin_m:.3} sigma_e={endurance_pa:.3e} Pa"
        );
        println!(
            "[rolling_contact]   contact (P={load_n} N, R1=R2={radius_m:.1e} m): a={:.6e} m p_max={:.6e} Pa z_shear={:.6e} m",
            contact.contact_radius_m, contact.peak_pressure_pa, contact.max_shear_depth_m,
        );

        // Direct Basquin call on the contact's peak pressure.
        let direct_cycles =
            basquin_cycles_to_failure(contact.peak_pressure_pa, basquin_c, basquin_m, endurance_pa);
        // Convenience wrapper: must compose the same two steps.
        let combined_cycles = rolling_contact_life_cycles(
            load_n,
            radius_m,
            radius_m,
            e,
            e,
            nu,
            nu,
            basquin_c,
            basquin_m,
            endurance_pa,
        );
        println!(
            "[rolling_contact]   life: direct={direct_cycles:.6e} cycles combined={combined_cycles:.6e} cycles match={}",
            direct_cycles == combined_cycles,
        );
    }

    // Ball-on-flat-rail contact: radius_2_m = INFINITY collapses the
    // effective radius to radius_1_m (documented usage on
    // `hertzian_sphere_sphere`). Gear steel ball on a gear steel rail.
    let (gear_e, gear_nu, gear_c, gear_m, gear_se) = materials::gear_steel_8620();
    let rail_contact = hertzian_sphere_sphere(
        300.0,
        5.0e-3,
        f32::INFINITY,
        gear_e,
        gear_e,
        gear_nu,
        gear_nu,
    );
    println!(
        "[rolling_contact] ball-on-flat-rail (gear_steel_8620, P=300 N, R1=5e-3 m, R2=inf): a={:.6e} m p_max={:.6e} Pa z_shear={:.6e} m",
        rail_contact.contact_radius_m, rail_contact.peak_pressure_pa, rail_contact.max_shear_depth_m,
    );
    let rail_cycles =
        basquin_cycles_to_failure(rail_contact.peak_pressure_pa, gear_c, gear_m, gear_se);
    println!("[rolling_contact]   flat-rail life: {rail_cycles:.6e} cycles");
}
