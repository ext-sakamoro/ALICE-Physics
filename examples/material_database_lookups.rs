//! Material Database Lookups Example
//!
//! Production entry point for the lookup side of the two material databases
//! and the anisotropic constructors that read them: `FilamentDb::get` /
//! `FilamentDb::is_empty` (`src/filament_db.rs`), `MaterialTable::len` /
//! `MaterialTable::is_empty` (`src/material.rs`), and
//! `OrthotropicElasticity::from_fdm_material` /
//! `AnisotropicStrength::from_fdm_material` (`src/anisotropic.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - a database holds what was registered: IDs count up from zero, `get(id)`
//!   returns the entry with that ID and `None` past the end
//! - a `MaterialTable` starts with one default material and grows by one per
//!   registration
//! - the FDM lift to an orthotropic solid keeps the in-plane modulus
//!   (`E_L = E_T = 1000 E_GPa` MPa), scales the stacking axis by the Z ratio
//!   `r` (`E_Z = r E_L`), takes the in-plane shear modulus from
//!   `G = E / (2 (1 + 0.35))` and scales it by `r` across the layers
//! - the strength lift is symmetric in tension and compression, `X_Z = r σ_y`,
//!   `S_LT = 0.6 σ_y` and `S_LZ = S_TZ = r S_LT`
//!
//! For PLA (3.5 GPa, σ_y = 50 MPa, r = 0.65) that is `E_Z = 2275` MPa,
//! `G_LT = 1296.296…` MPa, `X_Z = 32.5` MPa and `S_LZ = 19.5` MPa.
//!
//! Run with: `cargo run --example material_database_lookups`

use alice_physics::anisotropic::{AnisotropicStrength, OrthotropicElasticity};
use alice_physics::filament_db::{FilamentDb, MaterialProperties};
use alice_physics::material::{MaterialTable, PhysicsMaterial};
use alice_physics::math::Fix128;

fn close(got: Fix128, want: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= 1e-9 * want.abs().max(1.0),
        "{what}: got {g:.12}, closed form {want:.12}"
    );
}

fn filament_db() {
    let mut db = FilamentDb::new();
    assert!(db.is_empty(), "a new database is empty");
    assert!(db.get(0).is_none(), "nothing to look up yet");

    let pla = db.register(MaterialProperties::pla());
    assert!(!db.is_empty(), "one material registered");
    let tpu = db.register(MaterialProperties::tpu());
    assert_eq!((pla, tpu), (0, 1), "IDs count up from zero");
    let got = db.get(tpu).expect("TPU was registered");
    assert_eq!((got.id, got.name), (1, "TPU"), "get(1) is the second entry");
    assert!(db.get(2).is_none(), "get past the end is None");

    let defaults = FilamentDb::with_defaults();
    assert!(!defaults.is_empty(), "the presets are registered");
    for id in 0..10 {
        let m = defaults.get(id).expect("ten presets, IDs 0-9");
        assert_eq!(m.id, id, "get({id}) returns the entry with that ID");
    }
    assert!(defaults.get(10).is_none(), "ten presets, no eleventh");
    println!(
        "FilamentDb: {} presets, get(0) = {}, get(9) = {}",
        defaults.len(),
        defaults.get(0).map_or("-", |m| m.name),
        defaults.get(9).map_or("-", |m| m.name)
    );
}

fn material_table() {
    let mut table = MaterialTable::new();
    assert!(!table.is_empty(), "the default material is always there");
    assert_eq!(table.len(), 1, "one default material");
    let metal = table.register_metal();
    let rubber = table.register_rubber();
    let custom = table.register(PhysicsMaterial::new(
        0,
        Fix128::from_ratio(1, 2),
        Fix128::from_ratio(1, 4),
    ));
    assert_eq!((metal, rubber, custom), (1, 2, 3), "IDs follow the default");
    assert_eq!(table.len(), 4, "default + three registrations");
    assert!(!table.is_empty());
    println!("MaterialTable: {} materials", table.len());
}

fn anisotropic_lift() {
    let db = FilamentDb::with_defaults();
    for m in db.iter().filter(|m| m.is_fdm()) {
        let e_gpa = m.youngs_modulus_gpa.to_f64();
        let sy = m.yield_strength_mpa.to_f64();
        let r = m.anisotropy_z_ratio.to_f64();

        let e = OrthotropicElasticity::from_fdm_material(m);
        let e_l = 1000.0 * e_gpa;
        let g = e_l / (2.0 * (1.0 + 0.35));
        close(e.e_l_mpa, e_l, "E_L");
        close(e.e_t_mpa, e_l, "E_T");
        close(e.e_z_mpa, r * e_l, "E_Z");
        close(e.g_lt_mpa, g, "G_LT");
        close(e.g_lz_mpa, r * g, "G_LZ");
        close(e.g_tz_mpa, r * g, "G_TZ");
        close(e.nu_lt, 0.35, "ν_LT");
        close(e.nu_lz, 0.30, "ν_LZ");

        let s = AnisotropicStrength::from_fdm_material(m);
        for (got, what) in [
            (s.x_l_tension_mpa, "X_L tension"),
            (s.x_l_compression_mpa, "X_L compression"),
            (s.x_t_tension_mpa, "X_T tension"),
            (s.x_t_compression_mpa, "X_T compression"),
        ] {
            close(got, sy, what);
        }
        close(s.x_z_tension_mpa, r * sy, "X_Z tension");
        close(s.x_z_compression_mpa, r * sy, "X_Z compression");
        close(s.s_lt_mpa, 0.6 * sy, "S_LT");
        close(s.s_lz_mpa, r * 0.6 * sy, "S_LZ");
        close(s.s_tz_mpa, r * 0.6 * sy, "S_TZ");
        println!(
            "{:<8} E_Z {:>8.2} MPa  G_LT {:>9.3} MPa  X_Z {:>6.2} MPa  S_LZ {:>6.2} MPa",
            m.name,
            e.e_z_mpa.to_f64(),
            e.g_lt_mpa.to_f64(),
            s.x_z_tension_mpa.to_f64(),
            s.s_lz_mpa.to_f64()
        );
    }

    // The PLA numbers quoted in the header.
    let pla = db.get(0).expect("PLA is preset 0");
    let e = OrthotropicElasticity::from_fdm_material(pla);
    let s = AnisotropicStrength::from_fdm_material(pla);
    close(e.e_z_mpa, 2275.0, "PLA E_Z");
    close(e.g_lt_mpa, 3500.0 / 2.7, "PLA G_LT");
    close(s.x_z_tension_mpa, 32.5, "PLA X_Z");
    close(s.s_lz_mpa, 19.5, "PLA S_LZ");
}

fn main() {
    filament_db();
    material_table();
    anisotropic_lift();
    println!("all closed forms hold");
}
