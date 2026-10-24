//! Oracle: `YeeGrid::write_state` / `YeeGrid::read_state` の往復
//!
//! - **往復の bit 一致**: PML + 源 + 材料を持つ格子を途中まで回し、書き出して
//!   読み戻した格子が元の格子と `==` で等しく、そこから N step 回した後も
//!   6 成分すべてが bit 一致する 書き出しを 2 回しても同じ byte 列になる
//! - **歯の確認**: 同じ格子を `get` / `set` で作り直すと、PML の split 半場が
//!   失われて次の step から bit が分かれる (この比較が分かれない scene では
//!   往復試験は split 半場を見ていないことになる)
//! - **拒否**: 版違い / 0 の辺 / 未定義の option bit / flag が 0・1 以外 /
//!   途中で切れた byte 列 (全 prefix) / 余り byte、を `YeeStateError` で返し
//!   panic も過大な確保もしない

use alice_physics_core::math::Fix128;
use alice_physics_core::maxwell_fdtd::{
    Absorber, Component, Material, MaterialMap, YeeGrid, YeeStateError, COURANT_3D,
    YEE_STATE_VERSION,
};

const COMPONENTS: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

fn q(num: i64, den: i64) -> Fix128 {
    Fix128::from_int(num) / Fix128::from_int(den)
}

/// 12 × 10 × 8、PML (軸ごとに厚さが違う)、PML の内側に誘電体 + 損失の板、源 3 本
fn full_grid() -> YeeGrid {
    let mut map = MaterialMap::vacuum(12, 10, 8);
    map.fill(
        [5, 4, 3],
        [7, 6, 5],
        Material::new(q(9, 4), Fix128::ONE, q(1, 8)),
    );
    let mut g = YeeGrid::new_with_absorber(
        12,
        10,
        8,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 2, 2],
            sigma_max: Fix128::from_int(3),
        },
    )
    .with_materials(&map)
    .unwrap();
    g.set(Component::Ez, 4, 5, 4, Fix128::ONE);
    g.set(Component::Hx, 6, 4, 4, q(-3, 7));
    g.set_current(Component::Ex, 4, 4, 4, q(1, 3));
    g.set_current(Component::Ey, 8, 5, 3, q(-2, 5));
    g.set_charge(3, 3, 3, q(5, 11));
    g
}

fn assert_fields_equal(a: &YeeGrid, b: &YeeGrid, what: &str) {
    for c in COMPONENTS {
        let (ni, nj, nk) = a.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert_eq!(
                        a.get(c, i, j, k),
                        b.get(c, i, j, k),
                        "{what}: {c:?}({i},{j},{k})"
                    );
                }
            }
        }
    }
}

fn first_difference(a: &YeeGrid, b: &YeeGrid) -> Option<(Component, usize, usize, usize)> {
    for c in COMPONENTS {
        let (ni, nj, nk) = a.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    if a.get(c, i, j, k) != b.get(c, i, j, k) {
                        return Some((c, i, j, k));
                    }
                }
            }
        }
    }
    None
}

fn bytes_of(g: &YeeGrid) -> Vec<u8> {
    let mut out = Vec::new();
    g.write_state(&mut out);
    out
}

#[test]
fn a_pml_lattice_with_sources_and_materials_round_trips_bit_for_bit() {
    let mut a = full_grid();
    for _ in 0..40 {
        a.step();
    }
    let bytes = bytes_of(&a);
    let mut b = YeeGrid::read_state(&bytes).unwrap();
    assert_eq!(a, b, "the read lattice differs from the written one");
    assert_eq!(bytes_of(&b), bytes, "writing again gives other bytes");
    for _ in 0..120 {
        a.step();
        b.step();
    }
    assert_eq!(a, b, "the lattices drifted apart after the round trip");
    assert_fields_equal(&a, &b, "after 120 steps");
    assert!(
        a.max_abs_field() > Fix128::ZERO,
        "the scene went to zero, so the comparison measured nothing"
    );
}

#[test]
fn rebuilding_through_get_and_set_loses_the_split_halves() {
    let mut a = full_grid();
    for _ in 0..40 {
        a.step();
    }
    let mut map = MaterialMap::vacuum(12, 10, 8);
    map.fill(
        [5, 4, 3],
        [7, 6, 5],
        Material::new(q(9, 4), Fix128::ONE, q(1, 8)),
    );
    let mut b = YeeGrid::new_with_absorber(
        12,
        10,
        8,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 2, 2],
            sigma_max: Fix128::from_int(3),
        },
    )
    .with_materials(&map)
    .unwrap();
    for c in COMPONENTS {
        let (ni, nj, nk) = a.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    b.set(c, i, j, k, a.get(c, i, j, k));
                }
            }
        }
    }
    b.set_current(Component::Ex, 4, 4, 4, q(1, 3));
    b.set_current(Component::Ey, 8, 5, 3, q(-2, 5));
    b.set_charge(3, 3, 3, a.charge(3, 3, 3));
    assert_eq!(
        first_difference(&a, &b),
        None,
        "the visible fields differ before stepping"
    );
    for _ in 0..20 {
        a.step();
        b.step();
    }
    assert!(
        first_difference(&a, &b).is_some(),
        "get/set rebuilt a PML lattice exactly, so the round trip test does not see the split halves"
    );
}

#[test]
fn every_option_combination_round_trips() {
    for flags in 0u8..8 {
        let sources = flags & 1 != 0;
        let pml = flags & 2 != 0;
        let materials = flags & 4 != 0;
        let mut g = if pml {
            YeeGrid::new_with_absorber(
                8,
                8,
                8,
                COURANT_3D,
                Absorber::GradedPml {
                    depth: [2, 2, 2],
                    sigma_max: Fix128::from_int(2),
                },
            )
        } else {
            YeeGrid::new(8, 8, 8, COURANT_3D)
        };
        if materials {
            let mut map = MaterialMap::vacuum(8, 8, 8);
            map.fill(
                [3, 3, 3],
                [5, 5, 5],
                Material::dielectric(Fix128::from_int(2)),
            );
            g = g.with_materials(&map).unwrap();
        }
        g.set(Component::Ez, 4, 4, 3, Fix128::ONE);
        if sources {
            g.set_current(Component::Ex, 3, 4, 4, q(1, 4));
        }
        for _ in 0..10 {
            g.step();
        }
        let bytes = bytes_of(&g);
        assert_eq!(bytes[44], flags, "option byte for case {flags}");
        let mut r = YeeGrid::read_state(&bytes).unwrap();
        assert_eq!(r, g, "case {flags}");
        for _ in 0..10 {
            g.step();
            r.step();
        }
        assert_eq!(r, g, "case {flags} after stepping");
    }
}

#[test]
fn the_header_is_version_dimensions_courant_flags() {
    let g = YeeGrid::new(3, 4, 5, COURANT_3D);
    let bytes = bytes_of(&g);
    assert_eq!(&bytes[0..4], &YEE_STATE_VERSION.to_le_bytes());
    assert_eq!(&bytes[4..12], &3u64.to_le_bytes());
    assert_eq!(&bytes[12..20], &4u64.to_le_bytes());
    assert_eq!(&bytes[20..28], &5u64.to_le_bytes());
    assert_eq!(&bytes[28..36], &COURANT_3D.hi.to_le_bytes());
    assert_eq!(&bytes[36..44], &COURANT_3D.lo.to_le_bytes());
    assert_eq!(bytes[44], 0, "a bare lattice carries no options");
    let samples = 3 * 5 * 6 + 4 * 4 * 6 + 4 * 5 * 5 + 4 * 4 * 5 + 3 * 5 * 5 + 3 * 4 * 6;
    assert_eq!(bytes.len(), 4 + 24 + 16 + 1 + 16 * samples);
}

#[test]
fn malformed_bytes_are_refused() {
    let mut g = YeeGrid::new_with_absorber(
        2,
        2,
        2,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [1, 1, 1],
            sigma_max: Fix128::ONE,
        },
    );
    g.set_current(Component::Ex, 0, 1, 1, Fix128::ONE);
    let good = bytes_of(&g);
    assert!(YeeGrid::read_state(&good).is_ok());

    for cut in 0..good.len() {
        assert!(
            matches!(
                YeeGrid::read_state(&good[..cut]),
                Err(YeeStateError::Truncated { .. })
            ),
            "a prefix of {cut} bytes was not refused as truncated"
        );
    }

    let mut extra = good.clone();
    extra.push(0);
    assert_eq!(
        YeeGrid::read_state(&extra),
        Err(YeeStateError::Trailing { extra: 1 })
    );

    let mut version = good.clone();
    version[0] ^= 1;
    assert!(matches!(
        YeeGrid::read_state(&version),
        Err(YeeStateError::Version { .. })
    ));

    let mut zero = good.clone();
    zero[12..20].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(YeeGrid::read_state(&zero), Err(YeeStateError::Dimensions));

    let mut huge = good.clone();
    for at in [4, 12, 20] {
        huge[at..at + 8].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    }
    assert_eq!(YeeGrid::read_state(&huge), Err(YeeStateError::Dimensions));

    let mut big = good.clone();
    big[4..12].copy_from_slice(&(1u64 << 20).to_le_bytes());
    assert!(matches!(
        YeeGrid::read_state(&big),
        Err(YeeStateError::Truncated { .. })
    ));

    let mut flags = good.clone();
    flags[44] |= 8;
    assert_eq!(
        YeeGrid::read_state(&flags),
        Err(YeeStateError::Flags { found: flags[44] })
    );

    // the last bytes of this state are the z axis's lossy flags at the H coordinates
    let mut flag_byte = good.clone();
    let last = flag_byte.len() - 1;
    assert!(flag_byte[last] <= 1);
    flag_byte[last] = 2;
    assert_eq!(
        YeeGrid::read_state(&flag_byte),
        Err(YeeStateError::Bool { at: last })
    );
}
