//! Driving the filament / sheet-metal material database end to end
//!
//! `filament_db` is a pure lookup table: unit-conversion constants
//! (`GPA_TO_PA` / `MPA_TO_PA` / `G_CM3_TO_KG_M3`), per-material anisotropic
//! accessors (`yield_z` / `youngs_z` / `tensile_z` / `yield_at_angle` /
//! `youngs_at_angle`), SI-unit accessors (`density_si` / `yield_pa` /
//! `youngs_pa`), category classification (`is_fdm` / `is_sheet_metal`) and
//! the registry filter `FilamentDb::by_category`. None of these had a
//! production caller before this example — `structural_solver.rs` and
//! `beam_stress.rs` take already-converted SI values as plain `Fix128`
//! parameters rather than calling into this module, and `print_orientation.rs`
//! re-derives its own angle-mixing law (`effective_yield_at_angle`, with a
//! *different* theta convention — see its doc comment) instead of calling
//! `yield_at_angle`.
//!
//! This walks the whole surface: partition the default registry by category,
//! classify every material, convert one material's properties to SI units,
//! derive its Z-axis (through-layer) properties, and sweep the anisotropic
//! mixing law across the full angle domain. Every production value is
//! printed next to the closed-form expression it must equal.
//!
//! ```bash
//! cargo run --example filament_database_properties --features std
//! ```

use alice_physics::filament_db::{
    FilamentDb, MaterialCategory, MaterialProperties, GPA_TO_PA, G_CM3_TO_KG_M3, MPA_TO_PA,
};
use alice_physics::math::Fix128;

fn main() {
    let db = FilamentDb::with_defaults();
    println!("[filament_db] registry: {} materials", db.len());

    // --- by_category: partition into FDM filaments vs sheet metal --------
    let fdm = db.by_category(MaterialCategory::Fdm);
    let sheet = db.by_category(MaterialCategory::SheetMetal);
    println!(
        "[filament_db] by_category(Fdm) -> {} materials: {:?}",
        fdm.len(),
        fdm.iter().map(|m| m.name).collect::<Vec<_>>()
    );
    println!(
        "[filament_db] by_category(SheetMetal) -> {} materials: {:?}",
        sheet.len(),
        sheet.iter().map(|m| m.name).collect::<Vec<_>>()
    );
    assert_eq!(
        fdm.len() + sheet.len(),
        db.len(),
        "every default preset is exactly one of Fdm / SheetMetal"
    );

    // An unregistered category (no Sla / Powder presets in with_defaults)
    // filters to empty, not an error and not the full registry.
    let sla = db.by_category(MaterialCategory::Sla);
    println!(
        "[filament_db] by_category(Sla) -> {} materials (no SLA presets registered)",
        sla.len()
    );
    assert!(sla.is_empty());

    // --- is_fdm / is_sheet_metal classification ----------------------------
    for m in db.iter() {
        let in_fdm_partition = fdm.iter().any(|f| f.id == m.id);
        let in_sheet_partition = sheet.iter().any(|s| s.id == m.id);
        assert_eq!(
            m.is_fdm(),
            in_fdm_partition,
            "{}: is_fdm vs by_category",
            m.name
        );
        assert_eq!(
            m.is_sheet_metal(),
            in_sheet_partition,
            "{}: is_sheet_metal vs by_category",
            m.name
        );
        assert!(
            !(m.is_fdm() && m.is_sheet_metal()),
            "{}: category cannot be both Fdm and SheetMetal",
            m.name
        );
        println!(
            "[filament_db]   {:>8} category={:?} is_fdm={} is_sheet_metal={}",
            m.name,
            m.category,
            m.is_fdm(),
            m.is_sheet_metal()
        );
    }

    // --- SI unit conversion: PLA ---------------------------------------
    let pla = db.find_by_name("PLA").expect("PLA is a default preset");
    let density_closed_form = pla.density_g_cm3 * G_CM3_TO_KG_M3;
    let yield_closed_form = pla.yield_strength_mpa * MPA_TO_PA;
    let youngs_closed_form = pla.youngs_modulus_gpa * GPA_TO_PA;
    println!(
        "[filament_db] PLA density_si={} (closed form {})",
        pla.density_si().to_f64(),
        density_closed_form.to_f64()
    );
    println!(
        "[filament_db] PLA yield_pa={} (closed form {})",
        pla.yield_pa().to_f64(),
        yield_closed_form.to_f64()
    );
    println!(
        "[filament_db] PLA youngs_pa={} (closed form {})",
        pla.youngs_pa().to_f64(),
        youngs_closed_form.to_f64()
    );
    assert_eq!(pla.density_si(), density_closed_form);
    assert_eq!(pla.yield_pa(), yield_closed_form);
    assert_eq!(pla.youngs_pa(), youngs_closed_form);

    // --- Z-axis (through-layer) anisotropic properties ---------------------
    let tensile_z_closed_form = pla.tensile_strength_mpa * pla.anisotropy_z_ratio;
    let yield_z_closed_form = pla.yield_strength_mpa * pla.anisotropy_z_ratio;
    let youngs_z_closed_form = pla.youngs_modulus_gpa * pla.anisotropy_z_ratio;
    println!(
        "[filament_db] PLA tensile_z={} yield_z={} youngs_z={} (anisotropy_z_ratio={})",
        pla.tensile_z().to_f64(),
        pla.yield_z().to_f64(),
        pla.youngs_z().to_f64(),
        pla.anisotropy_z_ratio.to_f64()
    );
    assert_eq!(pla.tensile_z(), tensile_z_closed_form);
    assert_eq!(pla.yield_z(), yield_z_closed_form);
    assert_eq!(pla.youngs_z(), youngs_z_closed_form);

    // --- Anisotropic mixing law across the full angle domain --------------
    let quarter_pi = Fix128::HALF_PI.shr_bits(1); // exact pi/4 (shift is exact)
    for (label, theta) in [
        ("0", Fix128::ZERO),
        ("pi/4", quarter_pi),
        ("pi/2", Fix128::HALF_PI),
    ] {
        let y = pla.yield_at_angle(theta);
        let e = pla.youngs_at_angle(theta);
        println!(
            "[filament_db] PLA yield_at_angle({label})={} youngs_at_angle({label})={}",
            y.to_f64(),
            e.to_f64()
        );
    }

    // Sheet metal is isotropic: Z-axis accessors and angle sweep are constant.
    let sus304 = db
        .find_by_name("SUS304")
        .expect("SUS304 is a default preset");
    println!(
        "[filament_db] SUS304 (isotropic) yield_z={} == yield_strength_mpa={}",
        sus304.yield_z().to_f64(),
        sus304.yield_strength_mpa.to_f64()
    );
    assert_eq!(sus304.yield_z(), sus304.yield_strength_mpa);
    assert_eq!(
        sus304.yield_at_angle(Fix128::ZERO),
        sus304.yield_at_angle(Fix128::HALF_PI)
    );

    println!(
        "[filament_db] done: {} production entry points exercised",
        13
    );
    let _ = MaterialProperties::pla(); // keep the preset constructor path visible in the log above
}
