//! Audit S1-5 oracles for `alice_physics::filament_db`.
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::*;
use alice_physics::math::Fix128;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn near(got: Fix128, want: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= 1e-12 * (1.0 + want.abs()),
        "{what}: got {g} want {want}"
    );
}

#[test]
fn unit_constants_exact() {
    assert_eq!(GPA_TO_PA, Fix128::from_int(1_000_000_000));
    assert_eq!(MPA_TO_PA, Fix128::from_int(1_000_000));
    assert_eq!(G_CM3_TO_KG_M3, Fix128::from_int(1000));
}

/// (name, E GPa, yield MPa, tensile MPa, rho g/cm3, print C, Tg C, bridge mm, shrink, aniso)
type Row = (&'static str, f64, f64, f64, f64, f64, f64, f64, f64, f64);
const TABLE: [Row; 10] = [
    ("PLA", 3.5, 50.0, 60.0, 1.24, 200.0, 60.0, 20.0, 0.002, 0.65),
    (
        "PETG", 2.0, 50.0, 53.0, 1.27, 235.0, 80.0, 15.0, 0.004, 0.70,
    ),
    (
        "ABS", 2.3, 40.0, 40.0, 1.04, 240.0, 100.0, 12.0, 0.008, 0.68,
    ),
    ("PC", 2.4, 65.0, 70.0, 1.20, 270.0, 145.0, 15.0, 0.006, 0.65),
    (
        "TPU", 0.02, 30.0, 50.0, 1.20, 220.0, -30.0, 5.0, 0.015, 0.90,
    ),
    (
        "Nylon", 1.5, 40.0, 60.0, 1.05, 260.0, 50.0, 10.0, 0.015, 0.75,
    ),
    (
        "CF-Nylon", 10.0, 85.0, 100.0, 1.15, 270.0, 60.0, 20.0, 0.005, 0.50,
    ),
    (
        "PEEK", 4.0, 95.0, 100.0, 1.32, 400.0, 143.0, 25.0, 0.012, 0.70,
    ),
    ("SUS304", 200.0, 215.0, 505.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ("A5052", 70.0, 90.0, 230.0, 2.68, 0.0, 0.0, 0.0, 0.0, 1.0),
];

#[test]
fn preset_table_regression_pin_and_registration_order() {
    let db = FilamentDb::with_defaults();
    assert_eq!(db.len(), 10);
    for (i, row) in TABLE.iter().enumerate() {
        let m = db.get(i as FilamentId).unwrap();
        assert_eq!(m.name, row.0);
        assert_eq!(m.id as usize, i, "{}", row.0);
        near(m.youngs_modulus_gpa, row.1, "E");
        near(m.yield_strength_mpa, row.2, "yield");
        near(m.tensile_strength_mpa, row.3, "tensile");
        near(m.density_g_cm3, row.4, "rho");
        near(m.print_temp_c, row.5, "print");
        near(m.glass_transition_c, row.6, "Tg");
        near(m.bridging_distance_mm, row.7, "bridge");
        near(m.shrinkage_ratio, row.8, "shrink");
        near(m.anisotropy_z_ratio, row.9, "aniso");
        let by_name = db.find_by_name(row.0).unwrap();
        assert_eq!(by_name.id as usize, i);
    }
    // direct constructors agree with the registered copies except for id
    for (ctor, row) in [
        (MaterialProperties::pla as fn() -> MaterialProperties, "PLA"),
        (MaterialProperties::petg, "PETG"),
        (MaterialProperties::abs, "ABS"),
        (MaterialProperties::pc, "PC"),
        (MaterialProperties::tpu, "TPU"),
        (MaterialProperties::nylon, "Nylon"),
        (MaterialProperties::cf_nylon, "CF-Nylon"),
        (MaterialProperties::peek, "PEEK"),
        (MaterialProperties::sus304, "SUS304"),
        (MaterialProperties::a5052, "A5052"),
    ] {
        let direct = ctor();
        let reg = db.find_by_name(row).unwrap();
        let mut d2 = direct;
        d2.id = reg.id;
        assert_eq!(&d2, reg, "{row}");
    }
}

/// Physical sanity of the data set (documented ranges in the struct docs).
#[test]
fn preset_physical_invariants_and_documented_ranges() {
    let db = FilamentDb::with_defaults();
    for m in db.iter() {
        assert!(
            m.yield_strength_mpa <= m.tensile_strength_mpa,
            "{}: yield<=tensile",
            m.name
        );
        assert!(m.youngs_modulus_gpa > Fix128::ZERO);
        assert!(m.density_g_cm3 > Fix128::ZERO);
        assert!(m.anisotropy_z_ratio > Fix128::ZERO && m.anisotropy_z_ratio <= Fix128::ONE);
        assert!(m.shrinkage_ratio >= Fix128::ZERO && m.shrinkage_ratio < f(0.05));
        if m.is_fdm() {
            assert!(
                m.glass_transition_c < m.print_temp_c,
                "{}: Tg < print temp",
                m.name
            );
        } else {
            assert!(
                m.print_temp_c.is_zero()
                    && m.bridging_distance_mm.is_zero()
                    && m.glass_transition_c.is_zero()
            );
        }
    }
    // struct doc: typical 0.60-0.80 for PLA/ABS, 0.90+ for TPU, 0.50 for CF-Nylon, 1.0 for metals
    let g = |n: &str| db.find_by_name(n).unwrap().anisotropy_z_ratio.to_f64();
    for n in ["PLA", "ABS"] {
        assert!((0.60..=0.80).contains(&g(n)), "{n}");
    }
    assert!(g("TPU") >= 0.90);
    assert!((g("CF-Nylon") - 0.5).abs() < 1e-12);
    assert!(g("SUS304") == 1.0 && g("A5052") == 1.0);
}

#[test]
fn z_axis_and_si_conversions_closed_forms() {
    let db = FilamentDb::with_defaults();
    for m in db.iter() {
        let r = m.anisotropy_z_ratio.to_f64();
        near(m.youngs_z(), m.youngs_modulus_gpa.to_f64() * r, "E_z");
        near(m.yield_z(), m.yield_strength_mpa.to_f64() * r, "yield_z");
        near(
            m.tensile_z(),
            m.tensile_strength_mpa.to_f64() * r,
            "tensile_z",
        );
        near(m.youngs_pa(), m.youngs_modulus_gpa.to_f64() * 1e9, "E Pa");
        near(
            m.yield_pa(),
            m.yield_strength_mpa.to_f64() * 1e6,
            "yield Pa",
        );
        near(m.density_si(), m.density_g_cm3.to_f64() * 1000.0, "rho SI");
    }
    // exact for dyadic data
    let pla = MaterialProperties::pla();
    assert_eq!(pla.youngs_pa(), Fix128::from_int(3_500_000_000));
    assert_eq!(pla.yield_pa(), Fix128::from_int(50_000_000));
    let sus = MaterialProperties::sus304();
    assert_eq!(sus.density_si(), Fix128::from_int(8000));
    assert_eq!(sus.youngs_z(), sus.youngs_modulus_gpa);
}

/// density_si() is documented as kg/m3 of a datasheet value in g/cm3: PLA 1.24 -> 1240.
#[test]
#[ignore = "known defect: AUD-A-S1W5-011: from_ratio floors, so density_si() of PLA is 1239.99999999999995 (raw lo=u64::MAX-839), PETG/ABS/PC/TPU/Nylon/CF-Nylon/PEEK/A5052 likewise; floor()/integer use gives 1239 not 1240"]
fn density_si_of_decimal_datasheet_values_is_exact() {
    for (m, want) in [
        (MaterialProperties::pla(), 1240),
        (MaterialProperties::petg(), 1270),
        (MaterialProperties::abs(), 1040),
        (MaterialProperties::pc(), 1200),
        (MaterialProperties::tpu(), 1200),
        (MaterialProperties::nylon(), 1050),
        (MaterialProperties::cf_nylon(), 1150),
        (MaterialProperties::peek(), 1320),
        (MaterialProperties::a5052(), 2680),
    ] {
        assert_eq!(m.density_si(), Fix128::from_int(want), "{}", m.name);
    }
}

#[test]
fn angle_models_closed_form_and_symmetry() {
    let pla = MaterialProperties::pla();
    let (e, ez) = (pla.youngs_modulus_gpa.to_f64(), pla.youngs_z().to_f64());
    let (y, yz) = (pla.yield_strength_mpa.to_f64(), pla.yield_z().to_f64());
    for k in 0..16 {
        let th = k as f64 * 0.2 - 1.6;
        let (s, c) = (th.sin(), th.cos());
        // Reuss mixing (AUD-A-S1W5-012; this test used to pin the Voigt form)
        near(
            pla.youngs_at_angle(f(th)),
            1.0 / (c * c / e + s * s / ez),
            "E(theta)",
        );
        near(
            pla.yield_at_angle(f(th)),
            y * c * c + yz * s * s,
            "yield(theta)",
        );
        // even in theta, periodic in pi
        let a = pla.youngs_at_angle(f(th)).to_f64();
        let b = pla.youngs_at_angle(f(-th)).to_f64();
        let p = pla.youngs_at_angle(f(th + std::f64::consts::PI)).to_f64();
        assert!((a - b).abs() < 1e-9 && (a - p).abs() < 1e-9);
        // bounded by the axis values
        assert!(a <= e + 1e-9 && a >= ez - 1e-9);
    }
    near(pla.youngs_at_angle(Fix128::ZERO), e, "E(0)");
    near(pla.youngs_at_angle(Fix128::HALF_PI), ez, "E(pi/2)");
    near(pla.yield_at_angle(Fix128::ZERO), y, "yield(0)");
    near(pla.yield_at_angle(Fix128::HALF_PI), yz, "yield(pi/2)");
    // isotropic material: constant
    let sus = MaterialProperties::sus304();
    near(sus.youngs_at_angle(f(0.7)), 200.0, "iso E");
}

/// Doc: "Reuss-like lower bound". Reuss (iso-stress) mixing is the harmonic form
/// 1/E = cos^2/E_xy + sin^2/E_z; the implementation is the arithmetic (Voigt) mixture, an
/// UPPER bound.
#[test]
// AUD-A-S1W5-012
fn youngs_at_angle_is_a_lower_bound_reuss_form() {
    let pla = MaterialProperties::pla();
    let (e, ez) = (pla.youngs_modulus_gpa.to_f64(), pla.youngs_z().to_f64());
    let th = std::f64::consts::FRAC_PI_4;
    let reuss = 1.0 / (th.cos().powi(2) / e + th.sin().powi(2) / ez);
    near(pla.youngs_at_angle(f(th)), reuss, "Reuss");
}

#[test]
fn db_container_semantics() {
    let mut db = FilamentDb::new();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);
    assert!(db.get(0).is_none());
    assert!(db.find_by_name("PLA").is_none());
    assert!(db.by_category(MaterialCategory::Fdm).is_empty());
    let mut a = MaterialProperties::pla();
    a.id = 77; // register must overwrite the id
    assert_eq!(db.register(a), 0);
    let mut dup = MaterialProperties::petg();
    dup.name = "PLA";
    assert_eq!(db.register(dup), 1);
    assert_eq!(db.get(0).unwrap().id, 0);
    assert_eq!(db.get(1).unwrap().id, 1);
    // duplicate names: first match wins
    assert_eq!(db.find_by_name("PLA").unwrap().id, 0);
    assert!(db.get(2).is_none());
    assert_eq!(db.len(), 2);
    assert!(!db.is_empty());
    // category filter preserves insertion order; custom categories honoured
    let mut sla = MaterialProperties::pla();
    sla.name = "Resin";
    sla.category = MaterialCategory::Sla;
    db.register(sla);
    let fdm: Vec<_> = db
        .by_category(MaterialCategory::Fdm)
        .iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(fdm, vec![0, 1]);
    let r: Vec<_> = db
        .by_category(MaterialCategory::Sla)
        .iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(r, vec![2]);
    assert!(db.by_category(MaterialCategory::Powder).is_empty());
    assert!(!db.get(2).unwrap().is_fdm() && !db.get(2).unwrap().is_sheet_metal());
    let names: Vec<_> = db.iter().map(|m| m.name).collect();
    assert_eq!(names, vec!["PLA", "PLA", "Resin"]);
    assert!(
        MaterialProperties::sus304().is_sheet_metal() && !MaterialProperties::sus304().is_fdm()
    );
}

/// Number of distinct `FilamentId = u16` values: the closed form the capacity
/// boundary is derived from (AUD-A-S1W5-013).
const FILAMENT_ID_COUNT: usize = u16::MAX as usize + 1;

/// Fill an empty database up to the last free id. Ids start at 0 and are handed out
/// in order, so registration `i` (0-based) must get id `i`. Each material carries
/// its own index in `print_temp_c` so a lookup can be checked against it.
fn full_db() -> FilamentDb {
    let mut db = FilamentDb::new();
    for i in 0..FILAMENT_ID_COUNT {
        let mut m = MaterialProperties::pla();
        m.print_temp_c = Fix128::from_int(i as i64);
        let id = db
            .try_register(m)
            .unwrap_or_else(|e| panic!("registration {i} of {FILAMENT_ID_COUNT} failed: {e}"));
        assert_eq!(usize::from(id), i, "registration {i} got id {id}");
    }
    assert_eq!(db.len(), FILAMENT_ID_COUNT);
    db
}

/// AUD-A-S1W5-013: with `u16` ids the database holds at most 65,536 materials.
/// Registration 65,537 is refused instead of wrapping to id 0, and every earlier
/// id still finds the material registered under it.
#[test]
fn filament_db_try_register_refuses_past_the_u16_id_range() {
    let mut db = full_db();
    let mut last = MaterialProperties::pla();
    last.name = "LAST";
    assert_eq!(
        db.try_register(last),
        Err(alice_physics::PhysicsError::CapacityExceeded {
            resource: "filament materials",
            limit: FILAMENT_ID_COUNT,
        })
    );
    assert_eq!(
        db.len(),
        FILAMENT_ID_COUNT,
        "a refused registration must not grow the database"
    );
    assert!(
        db.find_by_name("LAST").is_none(),
        "the refused material must not be stored"
    );
    for i in 0..FILAMENT_ID_COUNT {
        let id = i as u16;
        let m = db.get(id).unwrap_or_else(|| panic!("id {id} is missing"));
        assert_eq!(
            m.id, id,
            "id {id} reads back a material carrying id {}",
            m.id
        );
        assert_eq!(
            m.print_temp_c,
            Fix128::from_int(i as i64),
            "id {id} does not find its material"
        );
    }
}

/// AUD-A-S1W5-013: `register` panics on registration 65,537 instead of silently
/// returning a wrapped id that aliases the first material.
#[test]
#[should_panic(expected = "FilamentDb::register")]
fn register_panics_when_every_filament_id_is_in_use() {
    let mut db = full_db();
    let _ = db.register(MaterialProperties::pla());
}

/// A direction without stiffness (anisotropy_z_ratio = 0, E_z = 0): any load
/// with a component along Z sees no stiffness (Reuss: the compliance is
/// infinite), while a pure in-plane load keeps E_xy
#[test]
fn a_zero_z_modulus_gives_zero_stiffness_off_the_plane() {
    let mut m = MaterialProperties::pla();
    m.anisotropy_z_ratio = Fix128::ZERO;
    // cos^2(0) from the CORDIC is a few ulps below 1
    near(
        m.youngs_at_angle(Fix128::ZERO),
        m.youngs_modulus_gpa.to_f64(),
        "E(0)",
    );
    assert_eq!(m.youngs_at_angle(f(0.3)), Fix128::ZERO);
    assert_eq!(m.youngs_at_angle(Fix128::HALF_PI), Fix128::ZERO);
}
