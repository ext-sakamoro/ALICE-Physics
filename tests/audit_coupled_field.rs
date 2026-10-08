//! Audit S1-5 oracles for `alice_physics::coupled_field`.
//! Expectations come from closed forms (trilinear interpolation, discrete heat equation,
//! exponential relaxation, arithmetic / weighted means), not from the implementation.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::*;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sim_field::ScalarField3D;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn p(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}
fn t3(a: f64, b: f64, c: f64) -> (Fix128, Fix128, Fix128) {
    (f(a), f(b), f(c))
}
fn grid(nx: usize, ny: usize, nz: usize, lo: f64, hi: f64) -> CoupledField {
    CoupledField::try_new(nx, ny, nz, t3(lo, lo, lo), t3(hi, hi, hi)).unwrap()
}
fn close(a: Fix128, b: f64, tol: f64, what: &str) {
    assert!(
        (a.to_f64() - b).abs() <= tol,
        "{what}: {} vs {}",
        a.to_f64(),
        b
    );
}

#[test]
fn construction_errors_getters_and_layout() {
    for (nx, ny, nz) in [(0, 2, 2), (2, 0, 2), (2, 2, 0)] {
        let e =
            CoupledField::try_new(nx, ny, nz, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap_err();
        assert_eq!(
            e,
            CoupledFieldError::EmptyGrid {
                resolution: (nx, ny, nz)
            }
        );
    }
    let g = CoupledField::try_new_filled(3, 4, 5, t3(-1.0, 0.0, 2.0), t3(1.0, 3.0, 10.0), f(7.5))
        .unwrap();
    assert_eq!((g.nx(), g.ny(), g.nz(), g.cell_count()), (3, 4, 5, 60));
    assert_eq!(g.as_slice().len(), 60);
    assert!(g.as_slice().iter().all(|&v| v == f(7.5)));
    assert_eq!(g.min(), t3(-1.0, 0.0, 2.0));
    assert_eq!(g.max(), t3(1.0, 3.0, 10.0));
    let c = g.cell_size();
    close(c.0, 1.0, 0.0, "cell x");
    close(c.1, 1.0, 1e-18, "cell y");
    close(c.2, 2.0, 0.0, "cell z");
    // row-major data[iz*ny*nx + iy*nx + ix]
    assert_eq!(g.index(2, 3, 4), 4 * 12 + 3 * 3 + 2);
    assert_eq!(g.index(1, 0, 0), 1);
    assert_eq!(g.index(0, 1, 0), 3);
    assert_eq!(g.index(0, 0, 1), 12);
    // degenerate axis: cell size one, never zero
    let d = CoupledField::try_new(1, 3, 1, t3(0.0, 0.0, 0.0), t3(5.0, 1.0, 5.0)).unwrap();
    assert_eq!(d.cell_size().0, Fix128::ONE);
    assert_eq!(d.cell_size().2, Fix128::ONE);
}

/// Doc (try_new_filled): "the stored spacing is never zero". With n > 1 and max == min the
/// spacing is zero, and every later `/ cell` returns ZERO silently (sample reads node 0 everywhere).
#[test]
fn degenerate_bounds_with_more_than_one_node_are_rejected_or_have_nonzero_cell() {
    let r = CoupledField::try_new(4, 4, 4, t3(0.0, 0.0, 0.0), t3(0.0, 1.0, 1.0));
    match r {
        Err(_) => {}
        Ok(g) => assert!(!g.cell_size().0.is_zero(), "cell.x = 0 with nx = 4"),
    }
}

#[test]
fn get_set_add_clamp_ignore_and_contains() {
    let mut g = grid(3, 3, 3, 0.0, 2.0);
    g.set(1, 2, 0, f(4.0));
    g.add(1, 2, 0, f(0.5));
    assert_eq!(g.get(1, 2, 0), f(4.5));
    // out-of-range writes are no-ops (no panic, no wrap into a neighbour)
    let before: Vec<_> = g.as_slice().to_vec();
    g.set(3, 0, 0, f(9.0));
    g.set(0, 3, 0, f(9.0));
    g.set(0, 0, 3, f(9.0));
    g.add(3, 3, 3, f(9.0));
    g.add(usize::MAX, 0, 0, f(9.0));
    assert_eq!(g.as_slice(), &before[..]);
    // get clamps each index independently
    g.set(2, 2, 2, f(-3.0));
    assert_eq!(g.get(99, 99, 99), f(-3.0));
    assert_eq!(g.get(2, 99, 0), g.get(2, 2, 0));
    // contains: closed box
    assert!(
        g.contains(p(0.0, 0.0, 0.0))
            && g.contains(p(2.0, 2.0, 2.0))
            && g.contains(p(1.0, 1.0, 1.0))
    );
    for out in [
        p(-1e-9, 1.0, 1.0),
        p(2.0 + 1e-9, 1.0, 1.0),
        p(1.0, -1e-9, 1.0),
        p(1.0, 2.0 + 1e-9, 1.0),
        p(1.0, 1.0, -1e-9),
        p(1.0, 1.0, 2.0 + 1e-9),
    ] {
        assert!(!g.contains(out));
    }
}

#[test]
fn trilinear_sample_matches_independent_formula_on_nonaffine_data() {
    let mut g = CoupledField::try_new(3, 4, 2, t3(-1.0, 0.0, 2.0), t3(1.0, 3.0, 4.0)).unwrap();
    let data = |i: usize, j: usize, k: usize| ((i * 7 + j * 3 + k * 11) % 13) as f64 * 0.5 - 2.0;
    for k in 0..2 {
        for j in 0..4 {
            for i in 0..3 {
                g.set(i, j, k, f(data(i, j, k)));
            }
        }
    }
    // nodes exactly
    for k in 0..2 {
        for j in 0..4 {
            for i in 0..3 {
                let pt = p(-1.0 + i as f64, j as f64, 2.0 + 2.0 * k as f64);
                close(g.sample(pt), data(i, j, k), 1e-15, "node");
            }
        }
    }
    // interior points via explicit weights
    for &(x, y, z) in &[
        (-0.3, 0.4, 2.7),
        (0.45, 2.2, 3.1),
        (0.9, 2.95, 2.05),
        (-0.99, 0.01, 3.99),
    ] {
        let (gx, gy, gz): (f64, f64, f64) = ((x + 1.0) / 1.0, y / 1.0, (z - 2.0) / 2.0);
        let (i0, j0, k0) = (
            gx.floor() as usize,
            gy.floor() as usize,
            gz.floor() as usize,
        );
        let (fx, fy, fz) = (gx - i0 as f64, gy - j0 as f64, gz - k0 as f64);
        let mut want = 0.0;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let w = (if di == 1 { fx } else { 1.0 - fx })
                        * (if dj == 1 { fy } else { 1.0 - fy })
                        * (if dk == 1 { fz } else { 1.0 - fz });
                    want += w * data((i0 + di).min(2), (j0 + dj).min(3), (k0 + dk).min(1));
                }
            }
        }
        close(g.sample(p(x, y, z)), want, 1e-12, "interior");
    }
    // outside clamps to the nearest boundary (face, edge, corner)
    close(
        g.sample(p(-50.0, 0.0, 2.0)),
        data(0, 0, 0),
        1e-15,
        "clamp corner",
    );
    close(
        g.sample(p(50.0, 50.0, 50.0)),
        data(2, 3, 1),
        1e-15,
        "clamp far corner",
    );
    close(
        g.sample(p(0.0, -9.0, 3.0)),
        g.sample(p(0.0, 0.0, 3.0)).to_f64(),
        1e-15,
        "clamp y",
    );
}

#[test]
fn gradient_affine_interior_and_boundary_degradation() {
    let mut g = grid(5, 5, 5, 0.0, 4.0);
    let (a, b, c) = (2.0, -3.0, 0.5);
    for k in 0..5 {
        for j in 0..5 {
            for i in 0..5 {
                g.set(i, j, k, f(a * i as f64 + b * j as f64 + c * k as f64 + 1.0));
            }
        }
    }
    let gr = g.gradient(p(1.7, 2.2, 1.3));
    close(gr.x, a, 1e-12, "gx");
    close(gr.y, b, 1e-12, "gy");
    close(gr.z, c, 1e-12, "gz");
    // on the min face: the lower arm is clamped, the estimate is half the slope (documented)
    let gb = g.gradient(p(0.0, 2.0, 2.0));
    close(gb.x, a / 2.0, 1e-12, "boundary gx");
    close(gb.y, b, 1e-12, "tangential gy");
    // 1-node axis: zero gradient there
    let mut d = CoupledField::try_new(1, 3, 1, t3(0.0, 0.0, 0.0), t3(1.0, 2.0, 1.0)).unwrap();
    d.set(0, 1, 0, f(5.0));
    close(d.gradient(p(0.0, 1.0, 0.0)).x, 0.0, 0.0, "degenerate gx");
}

#[test]
fn splat_weights_closed_form() {
    let mut g = grid(3, 3, 3, 0.0, 2.0);
    g.splat(p(1.0, 1.0, 1.0), f(8.0));
    assert_eq!(g.get(1, 1, 1), f(8.0), "on a node");
    assert_eq!(g.sum(), f(8.0));
    g.clear();
    g.splat(p(0.5, 0.5, 0.5), f(8.0));
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                assert_eq!(g.get(i, j, k), f(1.0), "1/8 each");
            }
        }
    }
    g.clear();
    g.splat(p(0.25, 0.0, 0.0), f(4.0));
    assert_eq!(g.get(0, 0, 0), f(3.0));
    assert_eq!(g.get(1, 0, 0), f(1.0));
    // outside: lands on the nearest boundary node, total preserved, negative value ok
    g.clear();
    g.splat(p(-9.0, 7.0, 0.0), f(-2.0));
    assert_eq!(g.get(0, 2, 0), f(-2.0));
    assert_eq!(g.sum(), f(-2.0));
    // splat accumulates
    g.splat(p(-9.0, 7.0, 0.0), f(5.0));
    assert_eq!(g.get(0, 2, 0), f(3.0));
}

/// diffuse: closed forms on tiny grids (reflective ghost node), explicit Euler.
#[test]
fn diffuse_closed_forms_and_invariants() {
    // constant field is a fixed point
    let mut c = CoupledField::try_new_filled(4, 3, 2, t3(0.0, 0.0, 0.0), t3(3.0, 2.0, 1.0), f(7.0))
        .unwrap();
    c.diffuse(f(0.01), f(1.0));
    assert!(c.as_slice().iter().all(|&v| v == f(7.0)));
    // 1-D, 3 nodes h=1: T=[0,1,0], r*dt = 0.1: node0: T0 + 0.1*(2T1-2T0)=0.2; node1: 1+0.1*(0-2+0)=0.8
    let mut g = CoupledField::try_new(3, 1, 1, t3(0.0, 0.0, 0.0), t3(2.0, 1.0, 1.0)).unwrap();
    g.set(1, 0, 0, f(1.0));
    g.diffuse(f(0.1), f(1.0));
    close(g.get(0, 0, 0), 0.2, 1e-12, "n0");
    close(g.get(1, 0, 0), 0.8, 1e-12, "n1");
    close(g.get(2, 0, 0), 0.2, 1e-12, "n2");
    // rate and dt enter only as the product; h scales as 1/h^2
    let mut g2 = CoupledField::try_new(3, 1, 1, t3(0.0, 0.0, 0.0), t3(4.0, 1.0, 1.0)).unwrap(); // h = 2
    g2.set(1, 0, 0, f(1.0));
    g2.diffuse(f(0.4), f(0.5)); // r dt = 0.2, /h^2 = 0.05
    close(g2.get(1, 0, 0), 1.0 + 0.05 * (-2.0), 1e-12, "h=2 center");
    close(g2.get(0, 0, 0), 0.05 * 2.0, 1e-12, "h=2 edge");
    // 3-D: separable sum of the 3 axis Laplacians for a single hot interior node
    let mut h = grid(3, 3, 3, 0.0, 2.0);
    h.set(1, 1, 1, f(1.0));
    h.diffuse(f(0.05), f(1.0));
    close(
        h.get(1, 1, 1),
        1.0 - 6.0 * 0.05,
        1e-12,
        "centre loses 6 r dt",
    );
    close(
        h.get(0, 1, 1),
        0.1,
        1e-12,
        "face neighbour (a boundary node) gains 2 r dt through the mirror ghost",
    );
    close(h.get(0, 0, 1), 0.0, 1e-15, "edge neighbour untouched");
    // 2-node axis: mirror of the single neighbour on both sides
    let mut m = CoupledField::try_new(2, 1, 1, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap();
    m.set(0, 0, 0, f(1.0));
    m.diffuse(f(0.1), f(1.0));
    close(
        m.get(0, 0, 0),
        1.0 + 0.1 * (2.0 * 0.0 - 2.0 * 1.0),
        1e-12,
        "n=2 node0",
    );
    close(m.get(1, 0, 0), 0.1 * 2.0, 1e-12, "n=2 node1");
}

/// Doc: explicit Euler is stable iff rate*dt*(2/hx^2+2/hy^2+2/hz^2) <= 1. The Neumann sawtooth
/// mode T_i = (-1)^i has eigenvalue -4/h^2, so the amplification factor is |1 - 4 r dt / h^2|.
#[test]
fn diffuse_stability_threshold_matches_documented_bound() {
    let run = |rdt: f64| -> f64 {
        let mut g = CoupledField::try_new(8, 1, 1, t3(0.0, 0.0, 0.0), t3(7.0, 1.0, 1.0)).unwrap();
        for i in 0..8 {
            g.set(i, 0, 0, f(if i % 2 == 0 { 1.0 } else { -1.0 }));
        }
        for _ in 0..40 {
            g.diffuse(f(rdt), f(1.0));
        }
        g.get(0, 0, 0).to_f64().abs()
    };
    // bound for h=1, 1-D: r dt * 2 <= 1 -> r dt <= 0.5
    let stable = run(0.49);
    let unstable = run(0.51);
    assert!(
        (stable - 0.96f64.powi(40)).abs() < 1e-9,
        "stable amplitude {stable}"
    );
    assert!(
        (unstable - 1.04f64.powi(40)).abs() / 1.04f64.powi(40) < 1e-9,
        "unstable amplitude {unstable}"
    );
    assert!(stable < 1.0 && unstable > 1.0);
}

#[test]
fn diffuse_conserves_dual_weighted_sum() {
    // weights 2^-b (b = number of axes on which the node is an end) are conserved by the mirror ghost
    let mut g = grid(5, 4, 3, 0.0, 4.0);
    for k in 0..3 {
        for j in 0..4 {
            for i in 0..5 {
                g.set(i, j, k, f(((i * 5 + j * 3 + k * 7) % 11) as f64 * 0.25));
            }
        }
    }
    let wsum = |g: &CoupledField| -> f64 {
        let mut s = 0.0;
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    let b = (i == 0 || i == 4) as i32
                        + (j == 0 || j == 3) as i32
                        + (k == 0 || k == 2) as i32;
                    s += g.get(i, j, k).to_f64() * 0.5f64.powi(b);
                }
            }
        }
        s
    };
    let w0 = wsum(&g);
    for _ in 0..25 {
        g.diffuse(f(0.01), f(2.0));
    }
    assert!((wsum(&g) - w0).abs() < 1e-9, "{} vs {}", wsum(&g), w0);
}

#[test]
fn decay_clamp_fill_clear_sum_max() {
    let mut g = grid(2, 2, 2, 0.0, 1.0);
    for (n, v) in g.as_mut_slice().iter_mut().enumerate() {
        *v = f(n as f64 - 3.0); // -3..4
    }
    assert_eq!(g.sum(), f(4.0));
    assert_eq!(g.max_value(), f(4.0));
    let mut neg = grid(2, 1, 1, 0.0, 1.0);
    neg.set(0, 0, 0, f(-5.0));
    neg.set(1, 0, 0, f(-2.0));
    assert_eq!(
        neg.max_value(),
        f(-2.0),
        "max of an all-negative field is not ZERO"
    );
    // decay_toward closed form
    let mut d = g.clone();
    d.decay_toward(f(10.0), f(0.5), f(2.0)); // factor e^-1
    let e1 = (-1.0f64).exp();
    for (n, v) in d.as_slice().iter().enumerate() {
        let want = (n as f64 - 3.0 - 10.0) * e1 + 10.0;
        assert!(
            (v.to_f64() - want).abs() < 1e-5 * want.abs().max(1.0),
            "n={n}: {} vs {want}",
            v.to_f64()
        );
    }
    let mut z = g.clone();
    z.decay(f(0.5), f(2.0));
    for (n, v) in z.as_slice().iter().enumerate() {
        let want = (n as f64 - 3.0) * e1;
        assert!((v.to_f64() - want).abs() < 1e-5 * want.abs().max(1.0));
    }
    // rate 0 / dt 0: identity
    let mut id = g.clone();
    id.decay(f(0.0), f(5.0));
    assert_eq!(id.as_slice(), g.as_slice());
    id.decay(f(5.0), f(0.0));
    assert_eq!(id.as_slice(), g.as_slice());
    // negative rate*dt grows away from target
    let mut gr = g.clone();
    gr.decay_toward(Fix128::ZERO, f(-0.5), f(2.0));
    assert!(gr.get(1, 1, 1).to_f64() > 4.0 * 2.7);
    // clamp
    let mut c = g.clone();
    c.clamp(f(-1.0), f(2.0));
    let want = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    for (v, w) in c.as_slice().iter().zip(want) {
        assert_eq!(v.to_f64(), w);
    }
    c.fill(f(3.5));
    assert!(c.as_slice().iter().all(|&v| v == f(3.5)));
    c.clear();
    assert_eq!(c.sum(), Fix128::ZERO);
}

#[test]
fn channel_arithmetic_and_grid_checks() {
    let mut a = grid(2, 2, 2, 0.0, 1.0);
    let mut b = grid(2, 2, 2, 0.0, 1.0);
    a.fill(f(1.0));
    b.fill(f(5.0));
    assert!(a.same_grid_as(&b));
    a.add_assign(&b).unwrap();
    assert!(a.as_slice().iter().all(|&v| v == f(6.0)));
    // blend: weight 0 untouched, 1 replaces, 0.25 interpolates
    let mut c = grid(2, 2, 2, 0.0, 1.0);
    c.fill(f(1.0));
    c.blend_from(&b, f(0.0)).unwrap();
    assert!(c.as_slice().iter().all(|&v| v == f(1.0)));
    c.blend_from(&b, f(0.25)).unwrap();
    assert!(c.as_slice().iter().all(|&v| v == f(2.0)));
    c.blend_from(&b, Fix128::ONE).unwrap();
    assert!(c.as_slice().iter().all(|&v| v == f(5.0)));
    // scale_div: zero is a no-op
    c.scale_div(Fix128::ZERO);
    assert!(c.as_slice().iter().all(|&v| v == f(5.0)));
    c.scale_div(f(4.0));
    assert!(c.as_slice().iter().all(|&v| v == f(1.25)));
    // mismatches reported with the right variant and leave `self` untouched
    let other_res = grid(3, 2, 2, 0.0, 1.0);
    let other_box = grid(2, 2, 2, 0.0, 2.0);
    let before = c.as_slice().to_vec();
    assert_eq!(
        c.add_assign(&other_res).unwrap_err(),
        CoupledFieldError::ResolutionMismatch {
            channel: (2, 2, 2),
            other: (3, 2, 2)
        }
    );
    assert_eq!(
        c.add_assign(&other_box).unwrap_err(),
        CoupledFieldError::BoundsMismatch
    );
    assert_eq!(
        c.blend_from(&other_box, f(0.5)).unwrap_err(),
        CoupledFieldError::BoundsMismatch
    );
    assert!(c.blend_from(&other_res, f(0.5)).is_err());
    assert_eq!(c.as_slice(), &before[..]);
    assert!(!c.same_grid_as(&other_res) && !c.same_grid_as(&other_box));
    // one differing min only
    let shifted = CoupledField::try_new(2, 2, 2, t3(0.5, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap();
    assert!(!c.same_grid_as(&shifted));
    // error Display strings are distinct and non-empty
    let msgs = [
        CoupledFieldError::EmptyGrid {
            resolution: (0, 1, 1),
        }
        .to_string(),
        CoupledFieldError::BoundsMismatch.to_string(),
        CoupledFieldError::NoParticipants.to_string(),
        CoupledFieldError::ResolutionMismatch {
            channel: (1, 1, 1),
            other: (2, 2, 2),
        }
        .to_string(),
    ];
    for i in 0..4 {
        assert!(!msgs[i].is_empty());
        for j in 0..i {
            assert_ne!(msgs[i], msgs[j]);
        }
    }
    assert!(msgs[0].contains("0x1x1"));
}

#[test]
fn f32_bridge_roundtrip_and_mismatch() {
    let mut s = ScalarField3D::new(3, 2, 2, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0));
    for (n, v) in s.data.iter_mut().enumerate() {
        *v = (n as f32) * 0.37 - 1.1;
    }
    let mut c = CoupledField::try_matching(&s).unwrap();
    assert_eq!((c.nx(), c.ny(), c.nz()), (3, 2, 2));
    c.copy_from_f32(&s).unwrap();
    for (a, b) in c.as_slice().iter().zip(s.data.iter()) {
        assert_eq!(a.to_f32(), *b, "exact f32 -> Fix128 -> f32");
    }
    let mut back = ScalarField3D::new(3, 2, 2, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0));
    c.write_to_f32(&mut back).unwrap();
    assert_eq!(back.data, s.data);
    let mut wrong_res = ScalarField3D::new(2, 2, 2, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0));
    assert!(matches!(
        c.copy_from_f32(&wrong_res),
        Err(CoupledFieldError::ResolutionMismatch { .. })
    ));
    assert!(matches!(
        c.write_to_f32(&mut wrong_res),
        Err(CoupledFieldError::ResolutionMismatch { .. })
    ));
    let mut wrong_box = ScalarField3D::new(3, 2, 2, (0.0, 0.0, 0.0), (2.5, 1.0, 1.0));
    assert_eq!(
        c.copy_from_f32(&wrong_box).unwrap_err(),
        CoupledFieldError::BoundsMismatch
    );
    assert_eq!(
        c.write_to_f32(&mut wrong_box).unwrap_err(),
        CoupledFieldError::BoundsMismatch
    );
}

// ---------------------------------------------------------------------------------------
// reconcile_*: participants
// ---------------------------------------------------------------------------------------

struct Part {
    field: CoupledField,
    fail_adopt: bool,
}
impl Part {
    fn new(vals: &[f64]) -> Self {
        let mut field =
            CoupledField::try_new(vals.len(), 1, 1, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap();
        for (i, v) in vals.iter().enumerate() {
            field.set(i, 0, 0, f(*v));
        }
        Self {
            field,
            fail_adopt: false,
        }
    }
}
impl CoupledScalar for Part {
    fn coupled_name(&self) -> &'static str {
        "temperature"
    }
    fn coupled_channel(&self) -> Result<CoupledField, CoupledFieldError> {
        CoupledField::try_new(self.field.nx(), 1, 1, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0))
    }
    fn publish(&self, out: &mut CoupledField) -> Result<(), CoupledFieldError> {
        out.as_mut_slice().copy_from_slice(self.field.as_slice());
        Ok(())
    }
    fn adopt(&mut self, src: &CoupledField) -> Result<(), CoupledFieldError> {
        if self.fail_adopt {
            return Err(CoupledFieldError::BoundsMismatch);
        }
        self.field.as_mut_slice().copy_from_slice(src.as_slice());
        Ok(())
    }
}
fn chan(n: usize) -> CoupledField {
    CoupledField::try_new(n, 1, 1, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap()
}
fn vals(p: &Part) -> Vec<f64> {
    p.field.as_slice().iter().map(|v| v.to_f64()).collect()
}

#[test]
fn reconcile_mean_closed_form_and_order_independence() {
    let mut a = Part::new(&[1.0, 10.0, -4.0]);
    let mut b = Part::new(&[3.0, 20.0, 4.0]);
    let mut c = Part::new(&[8.0, 0.0, 6.0]);
    let mut ch = chan(3);
    reconcile_mean(&mut [&mut a, &mut b, &mut c], &mut ch).unwrap();
    let want = [4.0, 10.0, 2.0];
    for p in [&a, &b, &c] {
        for (g, w) in vals(p).iter().zip(want) {
            assert!((g - w).abs() < 1e-15, "{g} vs {w}");
        }
    }
    // reversed order: bit-identical
    let mut a2 = Part::new(&[1.0, 10.0, -4.0]);
    let mut b2 = Part::new(&[3.0, 20.0, 4.0]);
    let mut c2 = Part::new(&[8.0, 0.0, 6.0]);
    let mut ch2 = chan(3);
    reconcile_mean(&mut [&mut c2, &mut b2, &mut a2], &mut ch2).unwrap();
    assert_eq!(ch.as_slice(), ch2.as_slice());
    // channel holds the agreed field; single participant is the identity
    let mut s = Part::new(&[2.5, 7.0]);
    let mut chs = chan(2);
    reconcile_mean(&mut [&mut s], &mut chs).unwrap();
    assert_eq!(vals(&s), vec![2.5, 7.0]);
}

#[test]
fn reconcile_errors() {
    let mut ch = chan(2);
    assert_eq!(
        reconcile_mean(&mut [], &mut ch).unwrap_err(),
        CoupledFieldError::NoParticipants
    );
    let mut a = Part::new(&[1.0, 2.0]);
    let mut wrong = chan(3);
    assert!(reconcile_mean(&mut [&mut a], &mut wrong).is_err());
    assert_eq!(vals(&a), vec![1.0, 2.0], "no adoption on error");
    assert_eq!(
        reconcile_weighted(&mut [], &[], &mut ch).unwrap_err(),
        CoupledFieldError::NoParticipants
    );
    assert_eq!(
        reconcile_weighted(&mut [&mut a], &[1, 2], &mut ch).unwrap_err(),
        CoupledFieldError::NoParticipants
    );
    assert_eq!(
        reconcile_weighted(&mut [&mut a], &[0], &mut ch).unwrap_err(),
        CoupledFieldError::NoParticipants
    );
    assert!(reconcile_weighted(&mut [&mut a], &[1], &mut wrong).is_err());
    assert_eq!(vals(&a), vec![1.0, 2.0]);
}

#[test]
fn reconcile_weighted_closed_form_equal_weights_and_abstention() {
    let mut a = Part::new(&[1.0, 10.0]);
    let mut b = Part::new(&[5.0, 2.0]);
    let mut ch = chan(2);
    reconcile_weighted(&mut [&mut a, &mut b], &[1, 3], &mut ch).unwrap();
    // (1*1 + 3*5)/4 = 4 ; (10 + 3*2)/4 = 4
    assert_eq!(vals(&a), vec![4.0, 4.0]);
    assert_eq!(vals(&b), vec![4.0, 4.0]);
    // abstaining participant (weight 0) is ignored by the mean and still adopts
    let mut a = Part::new(&[1.0, 10.0]);
    let mut b = Part::new(&[5.0, 2.0]);
    let mut c = Part::new(&[1000.0, -1000.0]);
    reconcile_weighted(&mut [&mut a, &mut b, &mut c], &[2, 2, 0], &mut ch).unwrap();
    assert_eq!(vals(&c), vec![3.0, 6.0]);
    // equal weights bit-identical to the plain mean
    let mut x1 = Part::new(&[0.1, 0.7, 1.0 / 3.0]);
    let mut y1 = Part::new(&[0.2, 0.3, 2.0 / 3.0]);
    let mut z1 = Part::new(&[0.9, 0.35, 0.123]);
    let mut c1 = chan(3);
    reconcile_mean(&mut [&mut x1, &mut y1, &mut z1], &mut c1).unwrap();
    let mut x2 = Part::new(&[0.1, 0.7, 1.0 / 3.0]);
    let mut y2 = Part::new(&[0.2, 0.3, 2.0 / 3.0]);
    let mut z2 = Part::new(&[0.9, 0.35, 0.123]);
    let mut c2 = chan(3);
    reconcile_weighted(&mut [&mut x2, &mut y2, &mut z2], &[7, 7, 7], &mut c2).unwrap();
    assert_eq!(c1.as_slice(), c2.as_slice());
    // weight order independence
    let mut x3 = Part::new(&[0.1, 0.7, 1.0 / 3.0]);
    let mut y3 = Part::new(&[0.2, 0.3, 2.0 / 3.0]);
    let mut c3 = chan(3);
    let mut x4 = Part::new(&[0.1, 0.7, 1.0 / 3.0]);
    let mut y4 = Part::new(&[0.2, 0.3, 2.0 / 3.0]);
    let mut c4 = chan(3);
    reconcile_weighted(&mut [&mut x3, &mut y3], &[2, 5], &mut c3).unwrap();
    reconcile_weighted(&mut [&mut y4, &mut x4], &[5, 2], &mut c4).unwrap();
    assert_eq!(c3.as_slice(), c4.as_slice());
}

/// reconcile_weighted doc: "On `Err` no participant has adopted anything."
#[test]
#[ignore = "known defect: AUD-A-S1W5-017: reconcile_weighted (and reconcile_mean) adopt participant by participant; if participant k's adopt() fails, participants 0..k have already adopted, contradicting 'On Err no participant has adopted anything'"]
fn reconcile_weighted_is_atomic_on_error() {
    let mut a = Part::new(&[1.0, 2.0]);
    let mut b = Part::new(&[3.0, 4.0]);
    b.fail_adopt = true;
    let mut ch = chan(2);
    let r = reconcile_weighted(&mut [&mut a, &mut b], &[1, 1], &mut ch);
    assert!(r.is_err());
    assert_eq!(
        vals(&a),
        vec![1.0, 2.0],
        "participant 0 adopted before participant 1 failed"
    );
}

/// Out-of-range index along ONE axis only (the other two in range) must not wrap into the
/// neighbouring row / slab (`index()` is row-major, so ix == nx aliases (0, iy + 1, iz)).
#[test]
fn set_and_add_with_single_out_of_range_axis_do_not_alias_a_neighbour() {
    let mut g = grid(3, 3, 3, 0.0, 2.0);
    let before: Vec<_> = g.as_slice().to_vec();
    g.set(3, 0, 0, f(9.0));
    g.add(3, 0, 0, f(9.0));
    g.set(0, 3, 0, f(9.0));
    g.add(0, 3, 0, f(9.0));
    g.set(0, 0, 3, f(9.0));
    g.add(0, 0, 3, f(9.0));
    assert_eq!(g.as_slice(), &before[..]);
}

/// Anisotropic spacing (hx, hy, hz) = (0.5, 2, 1): every axis must use its own cell size in
/// world_to_grid, gradient and diffuse.
fn aniso() -> CoupledField {
    // nx=5 over [0,2] -> 0.5 ; ny=3 over [0,4] -> 2 ; nz=3 over [0,8] -> 4
    CoupledField::try_new(5, 3, 3, t3(0.0, 0.0, 0.0), t3(2.0, 4.0, 8.0)).unwrap()
}

#[test]
fn anisotropic_spacing_sample_gradient_and_diffuse_use_per_axis_cells() {
    let (a, b, c) = (3.0, -2.0, 0.75); // slopes per world unit
    let mut g = aniso();
    for k in 0..3 {
        for j in 0..3 {
            for i in 0..5 {
                g.set(
                    i,
                    j,
                    k,
                    f(a * 0.5 * i as f64 + b * 2.0 * j as f64 + c * 4.0 * k as f64 + 1.0),
                );
            }
        }
    }
    // sample reproduces the affine function at arbitrary world points
    for &(x, y, z) in &[
        (0.3, 1.1, 2.7),
        (1.77, 3.2, 6.4),
        (0.0, 0.0, 0.0),
        (2.0, 4.0, 8.0),
    ] {
        close(
            g.sample(p(x, y, z)),
            a * x + b * y + c * z + 1.0,
            1e-12,
            "affine sample",
        );
    }
    // gradient = (a, b, c) in world units
    let gr = g.gradient(p(1.1, 1.9, 3.3));
    close(gr.x, a, 1e-12, "gx");
    close(gr.y, b, 1e-12, "gy");
    close(gr.z, c, 1e-12, "gz");
    // diffuse: delta at an interior node; neighbours gain r dt / h^2 along each axis
    let mut d = aniso();
    d.set(2, 1, 1, f(1.0));
    let rdt = 0.01;
    d.diffuse(f(rdt), f(1.0));
    let (hx2, hy2, hz2) = (0.25, 4.0, 16.0);
    close(
        d.get(2, 1, 1),
        1.0 - rdt * (2.0 / hx2 + 2.0 / hy2 + 2.0 / hz2),
        1e-12,
        "centre",
    );
    close(d.get(1, 1, 1), rdt / hx2, 1e-12, "x neighbour");
    close(
        d.get(2, 0, 1),
        rdt / hy2 * 2.0,
        1e-12,
        "y neighbour (mirror node: 2x)",
    );
    close(
        d.get(2, 1, 0),
        rdt / hz2 * 2.0,
        1e-12,
        "z neighbour at the boundary k=0 (mirror image of the hot node: 2x)",
    );
    close(
        d.get(2, 1, 2),
        rdt / hz2 * 2.0,
        1e-12,
        "z neighbour at the boundary k=2",
    );
}

#[test]
fn grid_mismatch_on_each_single_axis_is_rejected() {
    let mut c = grid(2, 3, 4, 0.0, 1.0);
    for other in [
        grid(3, 3, 4, 0.0, 1.0),
        grid(2, 4, 4, 0.0, 1.0),
        grid(2, 3, 5, 0.0, 1.0),
    ] {
        assert!(matches!(
            c.add_assign(&other),
            Err(CoupledFieldError::ResolutionMismatch { .. })
        ));
        assert!(matches!(
            c.blend_from(&other, f(0.5)),
            Err(CoupledFieldError::ResolutionMismatch { .. })
        ));
        assert!(!c.same_grid_as(&other));
    }
    // only the max corner differs / only one axis of min differs
    let a = CoupledField::try_new(2, 2, 2, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap();
    let mut b = a.clone();
    for other in [
        CoupledField::try_new(2, 2, 2, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 2.0)).unwrap(),
        CoupledField::try_new(2, 2, 2, t3(0.0, 0.5, 0.0), t3(1.0, 1.0, 1.0)).unwrap(),
    ] {
        assert_eq!(
            b.add_assign(&other).unwrap_err(),
            CoupledFieldError::BoundsMismatch
        );
        assert!(!a.same_grid_as(&other));
    }
    // f32 bridge: single-axis resolution mismatch
    let s = ScalarField3D::new(2, 2, 3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    let mut c2 = CoupledField::try_new(2, 2, 2, t3(0.0, 0.0, 0.0), t3(1.0, 1.0, 1.0)).unwrap();
    assert!(matches!(
        c2.copy_from_f32(&s),
        Err(CoupledFieldError::ResolutionMismatch { .. })
    ));
}
