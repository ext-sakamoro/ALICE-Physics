//! Audit oracles for layer_adhesion
//!
//! 期待値は (σ_y, z) の整数比から手計算した有理数で、実装を呼んで写していない
//! 注: `Fix128` の非 dyadic 比は数千 ulp の誤差を持つので 2^-48 を許容にする

use alice_physics::filament_db::MaterialProperties;
use alice_physics::layer_adhesion::{EffectiveStrength, PrintOrientation};
use alice_physics::math::Fix128;

fn close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() <= Fix128::from_raw(0, 1 << 16)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// (name, material, σ_y, z_num, z_den) filament_db の literal から転記
fn presets() -> Vec<(&'static str, MaterialProperties, i64, i64, i64)> {
    vec![
        ("PLA", MaterialProperties::pla(), 50, 65, 100),
        ("PETG", MaterialProperties::petg(), 50, 70, 100),
        ("ABS", MaterialProperties::abs(), 40, 68, 100),
        ("PC", MaterialProperties::pc(), 65, 65, 100),
        ("TPU", MaterialProperties::tpu(), 30, 90, 100),
        ("Nylon", MaterialProperties::nylon(), 40, 75, 100),
        ("CF-Nylon", MaterialProperties::cf_nylon(), 85, 50, 100),
        ("PEEK", MaterialProperties::peek(), 95, 70, 100),
        ("SUS304", MaterialProperties::sus304(), 215, 1, 1),
        ("A5052", MaterialProperties::a5052(), 90, 1, 1),
    ]
}

#[test]
fn allowables_all_presets_match_closed_form() {
    for (name, m, sigma, zn, zd) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        // σ_z = σ z, τ_xy = 0.6 σ, τ_across = τ_xy (1+z)/2
        let want_z = r(sigma * zn, zd);
        let want_txy = r(sigma * 6, 10);
        // τ_across = 0.6 σ (zd+zn) / (2 zd) = 3 σ (zd+zn) / (10 zd)
        let want_tz = r(3 * sigma * (zd + zn), 10 * zd);
        assert_eq!(s.normal_x_mpa, Fix128::from_int(sigma), "{name} normal_x");
        assert_eq!(s.normal_y_mpa, Fix128::from_int(sigma), "{name} normal_y");
        assert!(close(s.normal_z_mpa, want_z), "{name} normal_z");
        assert!(close(s.shear_xy_mpa, want_txy), "{name} shear_xy");
        assert!(close(s.shear_yz_mpa, want_tz), "{name} shear_yz");
        assert!(close(s.shear_xz_mpa, want_tz), "{name} shear_xz");
    }
}

#[test]
fn across_layer_shear_lies_between_z_tension_ratio_and_in_layer_shear() {
    // doc: across-layer shear は "intermediate" (z 張力と層内せん断の間)
    // 0 < z <= 1 の材料では τ_xy*z <= τ_across <= τ_xy
    for (name, m, ..) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        assert!(s.shear_yz_mpa <= s.shear_xy_mpa, "{name}");
        assert!(
            s.shear_yz_mpa >= s.shear_xy_mpa * m.anisotropy_z_ratio,
            "{name}"
        );
        assert!(s.normal_z_mpa <= s.normal_x_mpa, "{name}");
    }
}

#[test]
fn fos_four_components_all_presets_closed_form() {
    for (name, m, sigma, zn, zd) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        let a = Fix128::from_int(-7);
        assert!(close(s.fos_normal_x(a), r(sigma, 7)), "{name} x");
        assert!(close(s.fos_normal_z(a), r(sigma * zn, 7 * zd)), "{name} z");
        assert!(close(s.fos_shear_xy(a), r(sigma * 6, 70)), "{name} xy");
        assert!(
            close(s.fos_shear_xz(a), r(3 * sigma * (zd + zn), 70 * zd)),
            "{name} xz"
        );
    }
}

#[test]
fn min_fos_is_independent_of_self_and_uses_supplied_allowables() {
    // doc: "Minimum FoS across all 6 components" だが allowable は引数の組から取る
    // 期待値: min_i |allow_i / applied_i| を手計算
    let s = EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
    let other =
        EffectiveStrength::for_material(&MaterialProperties::peek(), PrintOrientation::XYFlat);
    let pairs = [
        (Fix128::from_int(10), Fix128::from_int(100)), // 10
        (Fix128::from_int(-20), Fix128::from_int(40)), // 2
        (Fix128::from_int(8), Fix128::from_int(8)),    // 1  <- worst
        (Fix128::ZERO, Fix128::from_int(1)),           // sentinel
        (Fix128::from_int(5), Fix128::from_int(60)),   // 12
        (Fix128::from_int(-1), Fix128::from_int(3)),   // 3
    ];
    assert_eq!(s.min_fos(&pairs), Fix128::ONE);
    assert_eq!(other.min_fos(&pairs), Fix128::ONE);
}

#[test]
#[ignore = "known defect: AUD-A-S1W5-030 (downstream: component_fos): for applied = 2^-64 or 2^-63 the quotient allowable / |applied| exceeds the Fix128 range, Fix128::div truncates it and the FoS comes back as 0 (PLA 50 MPa / 2^-64 -> 0), while applied == 0 returns the large sentinel, so 0 and 2^-64 give opposite answers"]
fn fos_is_never_below_one_when_applied_is_below_allowable() {
    // FoS = allowable/|applied| >= 1 iff |applied| <= allowable (閉形式)
    // 極小の applied (2^-64 .. 2^-40) でも商が i64 を超える範囲で符号が反転しないこと
    let s = EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
    for shift in [0u32, 1, 8, 16, 31, 32, 33, 40] {
        let applied = Fix128::from_raw(0, 1u64 << shift);
        for (label, fos) in [
            ("x", s.fos_normal_x(applied)),
            ("z", s.fos_normal_z(applied)),
            ("xy", s.fos_shear_xy(applied)),
            ("xz", s.fos_shear_xz(applied)),
        ] {
            assert!(
                fos >= Fix128::ONE,
                "{label}: applied = 2^-{} gives FoS {:?} (< 1) although applied << allowable",
                64 - shift,
                fos
            );
        }
    }
}

#[test]
fn fos_of_extreme_negative_applied_does_not_panic_and_is_positive() {
    let s = EffectiveStrength::for_material(&MaterialProperties::pla(), PrintOrientation::XYFlat);
    let fos = s.fos_normal_x(Fix128::from_raw(i64::MIN, 0));
    assert!(
        fos > Fix128::ZERO,
        "FoS must be positive for a finite applied stress, got {fos:?}"
    );
}

#[test]
fn fos_methods_read_their_own_pub_field_not_a_sibling() {
    // pub field を直接組んだ値で x/y と xz/yz を区別する (for_material 経由だと x==y, xz==yz で区別不能)
    let s = EffectiveStrength {
        normal_x_mpa: Fix128::from_int(10),
        normal_y_mpa: Fix128::from_int(20),
        normal_z_mpa: Fix128::from_int(30),
        shear_xy_mpa: Fix128::from_int(40),
        shear_yz_mpa: Fix128::from_int(50),
        shear_xz_mpa: Fix128::from_int(60),
    };
    let a = Fix128::from_int(5);
    assert_eq!(s.fos_normal_x(a), Fix128::from_int(2));
    assert_eq!(s.fos_normal_z(a), Fix128::from_int(6));
    assert_eq!(s.fos_shear_xy(a), Fix128::from_int(8));
    assert_eq!(s.fos_shear_xz(a), Fix128::from_int(12));
}
