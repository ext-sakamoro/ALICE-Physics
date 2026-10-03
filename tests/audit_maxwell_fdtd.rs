//! Audit oracle for `maxwell_fdtd`: hand-derived one-step update, accessor shapes, guards,
//! closed forms of the pml helpers. Expected values are derived by hand from Faraday / Ampere on
//! the Yee lattice, not read back from the solver.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    cfl_limit_3d, loss_coefficients, theoretical_pml_reflection, Absorber, Component, YeeGrid,
    COURANT_3D,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn half() -> Fix128 {
    Fix128::from_raw(0, 1u64 << 63)
}
fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
const ALL: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

fn panics<F: FnOnce()>(f: F) -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f)).is_err();
    std::panic::set_hook(prev);
    r
}

#[test]
fn courant_constant_and_cfl_limit() {
    assert_eq!(COURANT_3D, Fix128::from_raw(0, 9u64 << 60));
    assert_eq!(COURANT_3D.to_f64(), 0.5625);
    // S^2 is exact: 81/256
    assert_eq!(COURANT_3D * COURANT_3D, Fix128::from_raw(0, 81u64 << 56));
    let lim = cfl_limit_3d().to_f64();
    assert!((lim - 1.0 / 3.0_f64.sqrt()).abs() < 1e-15, "lim={lim}");
    let margin = 1.0 - 0.5625 / lim;
    assert!(
        margin > 0.025 && margin < 0.027,
        "margin {margin} (doc says 2.6%)"
    );
}

#[test]
fn loss_coefficients_closed_form_and_stability() {
    let s = COURANT_3D;
    assert_eq!(loss_coefficients(Fix128::ZERO, s), (Fix128::ONE, s));
    for sigma in [1e-4, 0.01, 0.5, 2.0, 20.0, 128.0] {
        let (ca, cb) = loss_coefficients(fx(sigma), s);
        let a = sigma * 0.5625 / 2.0;
        assert!(
            (ca.to_f64() - (1.0 - a) / (1.0 + a)).abs() < 1e-14,
            "ca sigma={sigma}"
        );
        assert!(
            (cb.to_f64() - 0.5625 / (1.0 + a)).abs() < 1e-14,
            "cb sigma={sigma}"
        );
        // unconditional stability of the semi-implicit factor, and cb in (0, S)
        assert!(ca.to_f64().abs() < 1.0);
        assert!(cb > Fix128::ZERO && cb < s);
    }
    // sign: a = sigma S / 2 > 1 makes ca negative but still of modulus below 1
    let (ca, _) = loss_coefficients(fx(10.0), s);
    assert!(ca < Fix128::ZERO && ca > Fix128::NEG_ONE);
}

#[test]
fn theoretical_reflection_closed_form() {
    assert_eq!(theoretical_pml_reflection(0, fx(5.0)), Fix128::ONE);
    // depth 1: sigma at the cell centre is sigma_max/8 -> R = exp(-sigma_max/4)
    for sm in [0.0, 1.0, 4.0, 12.0] {
        let r = theoretical_pml_reflection(1, fx(sm)).to_f64();
        // Fix128::exp is documented to ~1e-6 relative (measured 2.4e-8 here), so compare at 1e-6
        let want = (-sm / 4.0_f64).exp();
        assert!((r - want).abs() < 1e-6 * want, "sm={sm} r={r} want={want}");
    }
    // depth d: R = exp(-2 sum_c sm ((2c+1)/(2d))^3)
    for (d, sm) in [(2usize, 3.0), (4, 2.0), (8, 1.0)] {
        let mut sum = 0.0;
        for c in 0..d {
            let t = (2 * c + 1) as f64 / (2 * d) as f64;
            sum += sm * t * t * t;
        }
        let r = theoretical_pml_reflection(d, fx(sm)).to_f64();
        let want = (-2.0 * sum).exp();
        assert!(
            (r - want).abs() < 1e-6 * want + 1e-18,
            "d={d} r={r} want={want}"
        );
    }
    // monotone: deeper / stronger layers reflect less
    assert!(theoretical_pml_reflection(8, fx(4.0)) < theoretical_pml_reflection(4, fx(4.0)));
    assert!(theoretical_pml_reflection(4, fx(8.0)) < theoretical_pml_reflection(4, fx(4.0)));
}

#[test]
fn shapes_and_accessors() {
    let g = YeeGrid::new(3, 4, 5, half());
    assert_eq!(g.dims(), (3, 4, 5));
    assert_eq!(g.courant(), half());
    assert_eq!(g.component_dims(Component::Ex), (3, 5, 6));
    assert_eq!(g.component_dims(Component::Ey), (4, 4, 6));
    assert_eq!(g.component_dims(Component::Ez), (4, 5, 5));
    assert_eq!(g.component_dims(Component::Hx), (4, 4, 5));
    assert_eq!(g.component_dims(Component::Hy), (3, 5, 5));
    assert_eq!(g.component_dims(Component::Hz), (3, 4, 6));
    assert_eq!(g.interior_node_dims(), (2, 3, 4));
    assert_eq!(
        YeeGrid::new(1, 1, 1, half()).interior_node_dims(),
        (0, 0, 0)
    );
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        assert_eq!(g.get(c, a - 1, b - 1, d - 1), Fix128::ZERO);
        assert!(panics(|| {
            let _ = g.get(c, a, 0, 0);
        }));
        assert!(panics(|| {
            let _ = g.get(c, 0, b, 0);
        }));
        assert!(panics(|| {
            let _ = g.get(c, 0, 0, d);
        }));
    }
    assert!(panics(|| {
        let _ = YeeGrid::new(0, 2, 2, half());
    }));
    assert!(panics(|| {
        let _ = YeeGrid::new(2, 0, 2, half());
    }));
    assert!(panics(|| {
        let _ = YeeGrid::new(2, 2, 0, half());
    }));
}

#[test]
fn set_get_round_trip_is_per_component_and_per_index() {
    let mut g = YeeGrid::new(3, 3, 3, half());
    let mut tag = 1i64;
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    g.set(c, i, j, k, Fix128::from_int(tag));
                    tag += 1;
                }
            }
        }
    }
    let mut tag = 1i64;
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    assert_eq!(
                        g.get(c, i, j, k),
                        Fix128::from_int(tag),
                        "{c:?} ({i},{j},{k})"
                    );
                    tag += 1;
                }
            }
        }
    }
}

/// Hand-derived: 2x2x2 lattice, S = 1/2, Ex(0,1,1) = 1, everything else 0.
///  Faraday  H -= S curl E:  Hy(0,1,0) = -S, Hy(0,1,1) = +S, Hz(0,0,1) = +S, Hz(0,1,1) = -S
///  Ampere   E += S curl H:  Ex(0,1,1) = 1 - 4 S^2, Ey(1,0,1) = S^2, Ey(1,1,1) = -S^2,
///                           Ez(1,1,0) = S^2, Ez(1,1,1) = -S^2
#[test]
fn one_step_matches_the_hand_derived_update() {
    let mut g = YeeGrid::new(2, 2, 2, half());
    g.set(Component::Ex, 0, 1, 1, Fix128::ONE);
    g.step();
    let q = Fix128::from_raw(0, 1u64 << 62); // 1/4
    let expect = |c: Component, i, j, k| -> Fix128 {
        match (c, i, j, k) {
            (Component::Hy, 0, 1, 0) => -half(),
            (Component::Hy, 0, 1, 1) => half(),
            (Component::Hz, 0, 0, 1) => half(),
            (Component::Hz, 0, 1, 1) => -half(),
            (Component::Ex, 0, 1, 1) => Fix128::ZERO, // 1 - 4 S^2
            (Component::Ey, 1, 0, 1) => q,
            (Component::Ey, 1, 1, 1) => -q,
            (Component::Ez, 1, 1, 0) => q,
            (Component::Ez, 1, 1, 1) => -q,
            _ => Fix128::ZERO,
        }
    };
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    assert_eq!(g.get(c, i, j, k), expect(c, i, j, k), "{c:?} ({i},{j},{k})");
                }
            }
        }
    }
}

#[test]
fn div_b_and_max_abs_helpers_hand_values() {
    let mut g = YeeGrid::new(2, 2, 2, half());
    // cell (0,1,0): Hx(1,1,0)-Hx(0,1,0) + Hy(0,2,0)-Hy(0,1,0) + Hz(0,1,1)-Hz(0,1,0)
    g.set(Component::Hx, 1, 1, 0, Fix128::from_int(5));
    g.set(Component::Hx, 0, 1, 0, Fix128::from_int(2));
    g.set(Component::Hy, 0, 2, 0, Fix128::from_int(-7));
    g.set(Component::Hy, 0, 1, 0, Fix128::from_int(1));
    g.set(Component::Hz, 0, 1, 1, Fix128::from_int(11));
    g.set(Component::Hz, 0, 1, 0, Fix128::from_int(3));
    assert_eq!(
        g.div_b(0, 1, 0),
        Fix128::from_int((5 - 2) + (-7 - 1) + (11 - 3))
    );
    assert_eq!(g.max_abs_div_b(), Fix128::from_int(11)); // cell (0,1,1): 0 + 0 + (0 - 11)
    assert!(panics(|| {
        let _ = g.div_b(2, 0, 0);
    }));
    assert!(panics(|| {
        let _ = g.div_b(0, 2, 0);
    }));
    assert!(panics(|| {
        let _ = g.div_b(0, 0, 2);
    }));
    // max_abs_field sees every component and takes |.|
    let mut h = YeeGrid::new(2, 2, 2, half());
    for (n, c) in ALL.iter().enumerate() {
        h.set(*c, 0, 0, 0, Fix128::from_int(-(n as i64) - 1));
        assert_eq!(
            h.max_abs_field(),
            Fix128::from_int(n as i64 + 1),
            "after {c:?}"
        );
    }
}

#[test]
fn charge_current_and_gauss_accessors() {
    let mut g = YeeGrid::new(3, 3, 3, half());
    assert_eq!(g.total_charge(), Fix128::ZERO);
    assert_eq!(g.charge(1, 1, 1), Fix128::ZERO);
    assert_eq!(g.div_j(1, 1, 1), Fix128::ZERO);
    g.set_charge(1, 1, 1, Fix128::from_int(2));
    g.set_charge(2, 2, 2, Fix128::from_int(-5));
    assert_eq!(g.charge(1, 1, 1), Fix128::from_int(2));
    assert_eq!(g.charge(2, 2, 2), Fix128::from_int(-5));
    assert_eq!(g.charge(1, 2, 1), Fix128::ZERO);
    assert_eq!(g.total_charge(), Fix128::from_int(-3));
    // set_charge does not touch E, so the Gauss residual is -rho
    assert_eq!(g.gauss_residual(1, 1, 1), Fix128::from_int(-2));
    assert_eq!(g.max_abs_gauss_residual(), Fix128::from_int(5));
    // div_e on the six edges that meet at the node
    g.set(Component::Ex, 1, 1, 1, Fix128::from_int(3));
    g.set(Component::Ex, 0, 1, 1, Fix128::from_int(1));
    g.set(Component::Ey, 1, 1, 1, Fix128::from_int(10));
    g.set(Component::Ey, 1, 0, 1, Fix128::from_int(4));
    g.set(Component::Ez, 1, 1, 1, Fix128::from_int(-6));
    g.set(Component::Ez, 1, 1, 0, Fix128::from_int(-2));
    assert_eq!(
        g.div_e(1, 1, 1),
        Fix128::from_int((3 - 1) + (10 - 4) + (-6 + 2))
    );
    // div_j on the same stencil
    g.set_current(Component::Ex, 1, 1, 1, Fix128::from_int(8));
    g.set_current(Component::Ex, 0, 1, 1, Fix128::from_int(3));
    g.set_current(Component::Ey, 1, 1, 1, Fix128::from_int(2));
    g.set_current(Component::Ez, 1, 1, 1, Fix128::from_int(-1));
    assert_eq!(g.current(Component::Ex, 1, 1, 1), Fix128::from_int(8));
    assert_eq!(g.div_j(1, 1, 1), Fix128::from_int((8 - 3) + 2 + (-1)));
    // interior only
    for (i, j, k) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
    ] {
        assert!(
            panics(|| {
                let _ = g.charge(i, j, k);
            }),
            "charge({i},{j},{k})"
        );
    }
    // magnetic components have no current, PEC-tangential edges refuse one
    assert!(panics(|| {
        let _ = g.current(Component::Hx, 0, 0, 0);
    }));
    assert!(panics(|| g.set_current(
        Component::Hz,
        0,
        0,
        0,
        Fix128::ONE
    )));
    assert!(panics(|| g.set_current(
        Component::Ex,
        1,
        0,
        1,
        Fix128::ONE
    )));
    assert!(panics(|| g.set_current(
        Component::Ey,
        0,
        1,
        1,
        Fix128::ONE
    )));
    assert!(panics(|| g.set_current(
        Component::Ez,
        1,
        1,
        3,
        Fix128::ONE
    )));
}

/// One step with a current: E -= S J on the written edge; rho -= S div J on the node.
#[test]
fn source_step_deposits_minus_s_j_and_minus_s_div_j() {
    let mut g = YeeGrid::new(3, 3, 3, half());
    g.set_current(Component::Ex, 1, 1, 1, Fix128::from_int(4));
    g.step();
    // E_x(1,1,1) = -S J = -2 (no H yet when the curl was evaluated)
    assert_eq!(g.get(Component::Ex, 1, 1, 1), Fix128::from_int(-2));
    // div J at node (1,1,1) = +4, at node (2,1,1) = -4 (Ex(1,..) - ... enters with the other sign)
    assert_eq!(g.charge(1, 1, 1), Fix128::from_int(-2));
    assert_eq!(g.charge(2, 1, 1), Fix128::from_int(2));
    assert_eq!(g.total_charge(), Fix128::ZERO);
}

#[test]
fn absorber_none_and_zero_depth_equal_a_plain_lattice() {
    let a = YeeGrid::new(4, 4, 4, COURANT_3D);
    assert_eq!(
        YeeGrid::new_with_absorber(4, 4, 4, COURANT_3D, Absorber::None),
        a
    );
    assert_eq!(
        YeeGrid::new_with_absorber(
            4,
            4,
            4,
            COURANT_3D,
            Absorber::GradedPml {
                depth: [0, 0, 0],
                sigma_max: fx(5.0)
            }
        ),
        a
    );
    assert_ne!(
        YeeGrid::new_with_absorber(4, 4, 4, COURANT_3D, Absorber::Uniform { sigma: fx(0.5) }),
        a
    );
}

#[test]
fn pml_depth_limit_is_half_the_lattice() {
    let ok = catch_unwind(AssertUnwindSafe(|| {
        YeeGrid::new_with_absorber(
            4,
            6,
            8,
            half(),
            Absorber::GradedPml {
                depth: [2, 3, 4],
                sigma_max: fx(2.0),
            },
        )
    }));
    assert!(ok.is_ok());
    for d in [[3, 3, 4], [2, 4, 4], [2, 3, 5]] {
        assert!(
            panics(|| {
                let _ = YeeGrid::new_with_absorber(
                    4,
                    6,
                    8,
                    half(),
                    Absorber::GradedPml {
                        depth: d,
                        sigma_max: fx(2.0),
                    },
                );
            }),
            "depth {d:?}"
        );
    }
}

/// A slab normal to x (depth [2,0,0], 6 cells): Ex/Hx (normal components) stay lossless inside it,
/// the tangential ones are absorbing; samples outside the layer are lossless.
#[test]
fn is_absorbing_follows_the_two_loss_axes() {
    let g = YeeGrid::new_with_absorber(
        8,
        4,
        4,
        half(),
        Absorber::GradedPml {
            depth: [2, 0, 0],
            sigma_max: fx(4.0),
        },
    );
    // Ey/Ez/Hy/Hz depend on x: absorbing where the x conductivity is non-zero
    let (ey_i, _, _) = g.component_dims(Component::Ey);
    for i in 0..ey_i {
        // E sits at integer x; sigma != 0 for i in {0,1} (below depth=2 from the left) and {7,8}
        let want = i <= 1 || i >= 7;
        assert_eq!(g.is_absorbing(Component::Ey, i, 1, 1), want, "Ey i={i}");
        assert_eq!(g.is_absorbing(Component::Ez, i, 1, 1), want, "Ez i={i}");
    }
    // H sits at half-integer x: cell index i in {0,1} and {6,7}
    for i in 0..8 {
        let want = i <= 1 || i >= 6;
        assert_eq!(g.is_absorbing(Component::Hy, i, 1, 1), want, "Hy i={i}");
        assert_eq!(g.is_absorbing(Component::Hz, i, 1, 1), want, "Hz i={i}");
    }
    // normal components never absorb in an x slab
    for i in 0..8 {
        assert!(!g.is_absorbing(Component::Ex, i, 1, 1), "Ex i={i}");
    }
    for i in 0..=8 {
        assert!(!g.is_absorbing(Component::Hx, i, 1, 1), "Hx i={i}");
    }
    // no absorber: nothing absorbs
    let p = YeeGrid::new(8, 4, 4, half());
    assert!(!p.is_absorbing(Component::Ey, 0, 1, 1));
    // out-of-range index panics
    assert!(panics(|| {
        let _ = g.is_absorbing(Component::Ex, 8, 0, 0);
    }));
}

#[test]
fn uniform_absorber_marks_every_sample_and_blocks_currents_there() {
    let mut g = YeeGrid::new_with_absorber(3, 3, 3, half(), Absorber::Uniform { sigma: fx(0.25) });
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    assert!(g.is_absorbing(c, i, j, k), "{c:?}");
                }
            }
        }
    }
    assert!(panics(|| g.set_current(
        Component::Ex,
        1,
        1,
        1,
        Fix128::ONE
    )));
}

#[test]
fn uniform_loss_decays_a_mode_like_ca_per_step_and_lossless_does_not() {
    // a single Ex sample on a 2x2x2 lattice with uniform sigma: the field must shrink, and with
    // sigma = 0 (Uniform) the update is the lossless one (bit-identical to a plain lattice).
    let mut lossy =
        YeeGrid::new_with_absorber(2, 2, 2, half(), Absorber::Uniform { sigma: fx(1.0) });
    let mut plain = YeeGrid::new(2, 2, 2, half());
    let mut zero = YeeGrid::new_with_absorber(
        2,
        2,
        2,
        half(),
        Absorber::Uniform {
            sigma: Fix128::ZERO,
        },
    );
    for g in [&mut lossy, &mut plain, &mut zero] {
        g.set(Component::Ex, 0, 1, 1, Fix128::ONE);
    }
    for _ in 0..10 {
        lossy.step();
        plain.step();
        zero.step();
    }
    for c in ALL {
        let (a, b, d) = plain.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    assert_eq!(zero.get(c, i, j, k), plain.get(c, i, j, k), "{c:?}");
                }
            }
        }
    }
    assert!(lossy.max_abs_field() < plain.max_abs_field());
}

/// Stability: below the CFL limit the field stays bounded; above it grows without bound.
#[test]
fn courant_below_and_above_the_limit() {
    let seed = |g: &mut YeeGrid| {
        // deterministic non-symmetric seed
        let mut x = 12345u64;
        for c in [Component::Ex, Component::Ey, Component::Ez] {
            let (a, b, d) = g.component_dims(c);
            for i in 0..a {
                for j in 0..b {
                    for k in 0..d {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if g.is_absorbing(c, i, j, k) {
                            continue;
                        }
                        let v = ((x >> 33) % 64) as i64 - 32;
                        let interior = match c {
                            Component::Ex => j >= 1 && j < b - 1 && k >= 1 && k < d - 1,
                            Component::Ey => i >= 1 && i < a - 1 && k >= 1 && k < d - 1,
                            _ => i >= 1 && i < a - 1 && j >= 1 && j < b - 1,
                        };
                        if interior {
                            g.set(
                                c,
                                i,
                                j,
                                k,
                                Fix128::from_raw(v, 0)
                                    .half()
                                    .half()
                                    .half()
                                    .half()
                                    .half()
                                    .half(),
                            );
                        }
                    }
                }
            }
        }
    };
    let mut ok = YeeGrid::new(6, 6, 6, COURANT_3D);
    seed(&mut ok);
    let m0 = ok.max_abs_field().to_f64();
    let mut worst: f64 = 0.0;
    for _ in 0..400 {
        ok.step();
        worst = worst.max(ok.max_abs_field().to_f64());
    }
    assert!(
        worst < 4.0 * m0.max(0.5),
        "stable run grew: {worst} from {m0}"
    );
    let mut bad = YeeGrid::new(6, 6, 6, fx(0.65));
    seed(&mut bad);
    for _ in 0..400 {
        bad.step();
        if bad.max_abs_field().to_f64() > 1e6 {
            break;
        }
    }
    assert!(
        bad.max_abs_field().to_f64() > 1e3,
        "unstable run did not diverge: {}",
        bad.max_abs_field().to_f64()
    );
}
