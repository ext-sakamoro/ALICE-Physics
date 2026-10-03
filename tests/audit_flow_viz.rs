//! Audit oracles for `flow_viz`.
//!
//! Closed forms (hand-derived, inputs dyadic so `Fix128` holds them exactly):
//!
//! ```text
//! cell centre  c_i = min + (i + 1/2) * (max - min)/res         (per axis)
//! arrow        = (c, v/|v|, |v| * scale) where v = mean of the particle velocities
//!                within distance < dx of c   (dx = (max.x-min.x)/res)
//! streamline   p_{n+1} = p_n + dt * sum_i w_i v_i / sum_i w_i,   w_i = 1 - |p_n - x_i| / 1
//! ```
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::flow_viz::{
    generate_flow_arrows, generate_streamlines, FlowArrow, FlowVizConfig, Streamlines,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}
fn cfg(res: usize, lo: Vec3Fix, hi: Vec3Fix, scale: f64) -> FlowVizConfig {
    FlowVizConfig {
        grid_resolution: res,
        bounds_min: lo,
        bounds_max: hi,
        arrow_scale: fx(scale),
    }
}
fn cube(res: usize, scale: f64) -> FlowVizConfig {
    cfg(res, v3(-2.0, -2.0, -2.0), v3(2.0, 2.0, 2.0), scale)
}

/// One particle at the centre of cell (1,0,0): exactly one arrow, at the cell
/// centre, direction v/|v| = (0.6, 0.8, 0), magnitude 5 * scale. The 6 face
/// neighbours are at distance exactly dx = 2 and the test is strict `<`, so
/// they are excluded; this pins the strictness.
#[test]
fn single_particle_gives_single_arrow_with_closed_form() {
    let arrows = generate_flow_arrows(&[v3(1.0, -1.0, -1.0)], &[v3(3.0, 4.0, 0.0)], &cube(2, 2.0));
    assert_eq!(arrows.len(), 1);
    let a: &FlowArrow = &arrows[0];
    assert_eq!(a.position, v3(1.0, -1.0, -1.0));
    let d = f3(a.direction);
    assert!(
        near(d[0], 0.6, 1e-15) && near(d[1], 0.8, 1e-15) && near(d[2], 0.0, 1e-15),
        "{d:?}"
    );
    assert!(
        near(a.magnitude.to_f64(), 10.0, 1e-15),
        "{}",
        a.magnitude.to_f64()
    );
}

/// Cell centres are `min + (i + 1/2) * d` on each axis and the output order is
/// z-outer, y, x-inner. Particles with a common velocity at every cell centre of
/// a 3x3x3 grid over [0,3]^3 (dx = 1 -> radius 1, neighbours at distance 1 are
/// excluded) give 27 arrows at the 27 centres in that order.
#[test]
fn arrow_centres_and_order_on_a_3x3x3_grid() {
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                pos.push(v3(ix as f64 + 0.5, iy as f64 + 0.5, iz as f64 + 0.5));
                vel.push(v3(0.0, 0.0, 2.0));
            }
        }
    }
    let c = cfg(3, v3(0.0, 0.0, 0.0), v3(3.0, 3.0, 3.0), 1.0);
    let arrows = generate_flow_arrows(&pos, &vel, &c);
    assert_eq!(arrows.len(), 27);
    let mut k = 0;
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                assert_eq!(arrows[k].position, pos[k], "arrow {k} ({ix},{iy},{iz})");
                assert_eq!(arrows[k].direction, v3(0.0, 0.0, 1.0));
                assert_eq!(arrows[k].magnitude, fx(2.0));
                k += 1;
            }
        }
    }
}

/// Mean of two velocities in the same cell: (2,0,0) and (0,2,0) average to
/// (1,1,0): magnitude sqrt 2, direction (1/sqrt2, 1/sqrt2, 0).
#[test]
fn arrow_velocity_is_the_mean_of_nearby_particles() {
    let p = [v3(1.0, -1.0, -1.0), v3(1.25, -1.0, -1.0)];
    let v = [v3(2.0, 0.0, 0.0), v3(0.0, 2.0, 0.0)];
    let arrows = generate_flow_arrows(&p, &v, &cube(2, 1.0));
    let a = arrows
        .iter()
        .find(|a| a.position == v3(1.0, -1.0, -1.0))
        .expect("arrow");
    let s = 0.5_f64.sqrt();
    let d = f3(a.direction);
    assert!(near(d[0], s, 1e-15) && near(d[1], s, 1e-15), "{d:?}");
    assert!(near(a.magnitude.to_f64(), 2.0_f64.sqrt(), 1e-15));
}

/// Opposite velocities cancel: mean 0 -> the cell produces no arrow.
#[test]
fn cancelling_velocities_produce_no_arrow_in_that_cell() {
    let p = [v3(1.0, -1.0, -1.0), v3(1.0, -1.0, -1.0)];
    let v = [v3(2.0, 0.0, 0.0), v3(-2.0, 0.0, 0.0)];
    let arrows = generate_flow_arrows(&p, &v, &cube(2, 1.0));
    assert!(arrows.is_empty(), "{arrows:?}");
}

/// Direction is a unit vector and the magnitude equals |v|*scale across 11
/// decades of speed (relative 1e-10; Fix128 resolves |v|^2 to 2^-64, so the
/// relative error of the direction grows as 1/|v|^2 for tiny speeds).
#[test]
fn direction_is_unit_and_magnitude_scales_over_speed_range() {
    for e in [-5, -3, -1, 0, 2, 4, 6] {
        let s = 10f64.powi(e);
        let arrows = generate_flow_arrows(
            &[v3(1.0, -1.0, -1.0)],
            &[v3(3.0 * s, 4.0 * s, 12.0 * s)],
            &cube(2, 0.5),
        );
        assert_eq!(arrows.len(), 1, "speed scale 1e{e}");
        let d = f3(arrows[0].direction);
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-10, "1e{e}: |d| = {n}");
        let m = arrows[0].magnitude.to_f64();
        assert!(((m - 6.5 * s) / (6.5 * s)).abs() < 1e-10, "1e{e}: m = {m}");
        assert!(near(d[0], 3.0 / 13.0, 1e-10) && near(d[2], 12.0 / 13.0, 1e-10));
    }
}

/// Degenerate configs do not panic: res 0 acts as res 1, empty velocities,
/// empty positions, zero-volume bounds (no arrows).
#[test]
fn degenerate_configs_are_safe() {
    let p = [v3(0.0, 0.0, 0.0)];
    let v = [v3(1.0, 0.0, 0.0)];
    let one = generate_flow_arrows(
        &p,
        &v,
        &cfg(0, v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0), 1.0),
    );
    let ref1 = generate_flow_arrows(
        &p,
        &v,
        &cfg(1, v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0), 1.0),
    );
    assert_eq!(one.len(), ref1.len());
    assert!(generate_flow_arrows(&[], &[], &cube(4, 1.0)).is_empty());
    let flat = cfg(2, v3(1.0, 1.0, 1.0), v3(1.0, 1.0, 1.0), 1.0);
    assert!(generate_flow_arrows(&[v3(1.0, 1.0, 1.0)], &v, &flat).is_empty());
}

/// AUD-A-S4W1-007 (known defect): when `fluid_velocities` is shorter than
/// `fluid_positions`, a particle without a velocity is still COUNTED in the
/// average (`count += 1`) but contributes no velocity, so it dilutes the mean
/// toward zero. `interpolate_velocity` (streamlines) skips such a particle
/// entirely. Two particles in one cell, velocity given for the first only:
/// expected mean = v0 (magnitude 2), actual = v0/2 (magnitude 1).
#[test]
#[ignore = "known defect: AUD-A-S4W1-007: generate_flow_arrows counts position-only particles in the mean (magnitude 1 instead of 2 for 2 particles / 1 velocity); streamlines skip them"]
fn missing_velocity_does_not_dilute_the_mean() {
    let p = [v3(1.0, -1.0, -1.0), v3(1.0, -1.0, -1.0)];
    let v = [v3(2.0, 0.0, 0.0)];
    let arrows = generate_flow_arrows(&p, &v, &cube(2, 1.0));
    let a = arrows
        .iter()
        .find(|a| a.position == v3(1.0, -1.0, -1.0))
        .expect("arrow");
    assert!(
        near(a.magnitude.to_f64(), 2.0, 1e-15),
        "magnitude {}",
        a.magnitude.to_f64()
    );
}

/// AUD-A-S4W1-008 (known defect): the averaging radius is `dx` (x cell size)
/// for every axis. With non-cubic cells (here dx = 1, dy = 100) a particle
/// INSIDE the cell but more than dx from its centre along y is ignored, so a
/// tall cell holding a moving particle gets no arrow. For cubic cells every
/// interior point is within sqrt(3)/2*dx < dx of the centre, so the property
/// "a particle inside a cell contributes to that cell" holds there only.
#[test]
#[ignore = "known defect: AUD-A-S4W1-008: averaging radius uses dx for all axes; particle inside a dy=100 cell, 5 from its centre, gives no arrow"]
fn particle_inside_a_non_cubic_cell_contributes() {
    let c = cfg(1, v3(0.0, 0.0, 0.0), v3(1.0, 100.0, 1.0), 1.0);
    let arrows = generate_flow_arrows(&[v3(0.5, 55.0, 0.5)], &[v3(1.0, 0.0, 0.0)], &c);
    assert_eq!(arrows.len(), 1);
}

fn uniform_cloud(spacing: f64, n: i64, v: Vec3Fix) -> (Vec<Vec3Fix>, Vec<Vec3Fix>) {
    let mut p = Vec::new();
    let mut vv = Vec::new();
    for i in 0..n {
        for j in -1..=1 {
            for k in -1..=1 {
                p.push(v3(
                    i as f64 * spacing,
                    j as f64 * spacing,
                    k as f64 * spacing,
                ));
                vv.push(v);
            }
        }
    }
    (p, vv)
}

fn check_layout(s: &Streamlines, seeds: usize) {
    assert_eq!(s.len(), seeds);
    assert_eq!(s.offsets.len(), seeds + 1);
    assert_eq!(s.offsets[0], 0);
    assert_eq!(*s.offsets.last().expect("non-empty"), s.points.len());
    assert!(
        s.offsets.windows(2).all(|w| w[0] < w[1]),
        "each line has >= 1 point"
    );
    assert_eq!(s.is_empty(), seeds == 0);
}

/// Uniform flow: p_n = seed + n v dt (to Fix128 rounding), offsets layout
/// invariants, each line starts at its seed.
#[test]
fn uniform_flow_traces_straight_euler_lines() {
    let vel = v3(1.0, 0.5, 0.0);
    let (p, v) = uniform_cloud(0.25, 40, vel);
    let seeds = [v3(1.0, 0.0, 0.0), v3(2.0, 0.25, 0.0), v3(3.0, -0.25, 0.0)];
    let dt = fx(0.125);
    let s = generate_streamlines(&p, &v, &seeds, 6, dt);
    check_layout(&s, 3);
    for (k, seed) in seeds.iter().enumerate() {
        let line = s.line(k);
        assert_eq!(line.len(), 7);
        assert_eq!(line[0], *seed);
        let sd = f3(*seed);
        for (n, pt) in line.iter().enumerate() {
            let q = f3(*pt);
            let n = n as f64;
            assert!(
                near(q[0], sd[0] + n * 0.125, 1e-15),
                "line {k} pt {n}: {q:?}"
            );
            assert!(
                near(q[1], sd[1] + n * 0.0625, 1e-15),
                "line {k} pt {n}: {q:?}"
            );
        }
    }
}

/// Inverse-distance weighting by hand: particles at (0.25,0,0) v=(1,0,0) and
/// (-0.5,0,0) v=(0,2,0); at the origin w = 0.75 and 0.5, so
/// v = (0.75*(1,0,0) + 0.5*(0,2,0))/1.25 = (0.6, 0.8, 0).
#[test]
fn idw_weights_match_the_hand_derivation() {
    let p = [v3(0.25, 0.0, 0.0), v3(-0.5, 0.0, 0.0)];
    let v = [v3(1.0, 0.0, 0.0), v3(0.0, 2.0, 0.0)];
    let s = generate_streamlines(&p, &v, &[Vec3Fix::ZERO], 1, fx(0.5));
    let q = f3(s.line(0)[1]);
    assert!(
        near(q[0], 0.3, 1e-15) && near(q[1], 0.4, 1e-15) && near(q[2], 0.0, 1e-15),
        "{q:?}"
    );
}

/// A particle at distance exactly 1 (the influence radius) has weight 0: the
/// seed sees no velocity and the line stops at the seed. A particle at 0.5 moves it.
#[test]
fn influence_radius_boundary() {
    let v = [v3(2.0, 0.0, 0.0)];
    let at1 = generate_streamlines(&[v3(1.0, 0.0, 0.0)], &v, &[Vec3Fix::ZERO], 3, fx(0.5));
    assert_eq!(at1.line(0).len(), 1);
    let at_half = generate_streamlines(&[v3(0.5, 0.0, 0.0)], &v, &[Vec3Fix::ZERO], 1, fx(0.5));
    assert_eq!(at_half.line(0).len(), 2);
    assert_eq!(at_half.line(0)[1], v3(1.0, 0.0, 0.0));
}

/// Empty inputs: no seeds -> no lines; no particles / dt = 0 -> each line is its seed only.
#[test]
fn degenerate_streamline_inputs() {
    let (p, v) = uniform_cloud(0.25, 8, v3(1.0, 0.0, 0.0));
    let none = generate_streamlines(&p, &v, &[], 5, fx(0.5));
    check_layout(&none, 0);
    let seeds = [v3(0.5, 0.0, 0.0), v3(1.0, 0.0, 0.0)];
    for s in [
        generate_streamlines(&[], &[], &seeds, 5, fx(0.5)),
        generate_streamlines(&p, &v, &seeds, 5, Fix128::ZERO),
        generate_streamlines(&p, &v, &seeds, 0, fx(0.5)),
    ] {
        check_layout(&s, 2);
        assert_eq!(s.points, seeds.to_vec());
    }
}

/// Backward tracing: dt < 0 reverses the line (closed form seed - n v |dt|).
#[test]
fn negative_dt_traces_upstream() {
    let (p, v) = uniform_cloud(0.25, 40, v3(1.0, 0.0, 0.0));
    let s = generate_streamlines(&p, &v, &[v3(5.0, 0.0, 0.0)], 4, fx(-0.25));
    let line = s.line(0);
    assert_eq!(line.len(), 5);
    for (n, pt) in line.iter().enumerate() {
        assert!(near(pt.x.to_f64(), 5.0 - 0.25 * n as f64, 1e-15));
    }
}

/// AUD-A-S4W1-006 (known defect): `generate_streamlines` hard-codes an influence
/// radius of 1.0 (metre) with no parameter, so tracing is not scale-covariant.
/// The same uniform flow sampled on a 10x coarser particle grid (positions,
/// velocities, seeds all x10, dt unchanged) must give the same line scaled by
/// 10; here the 5.0-spaced particles are all farther than the radius from the
/// seed, so the line ends at the seed.
#[test]
#[ignore = "known defect: AUD-A-S4W1-006: influence radius hard-coded to 1.0; uniform flow with 2.5 m particle spacing yields a 1-point line (expected 5 points)"]
fn streamline_tracing_is_scale_covariant() {
    // Dense reference: spacing 0.25 (inside the radius), uniform v = (1,0,0).
    let (pd, vd) = uniform_cloud(0.25, 40, v3(1.0, 0.0, 0.0));
    let dense = generate_streamlines(&pd, &vd, &[v3(1.125, 0.0, 0.0)], 4, fx(0.25));
    assert_eq!(dense.line(0).len(), 5);
    // Same flow at 10x scale: spacing 2.5, v = (10,0,0), seed x10; dt unchanged.
    let (ps, vs) = uniform_cloud(2.5, 40, v3(10.0, 0.0, 0.0));
    let coarse = generate_streamlines(&ps, &vs, &[v3(11.25, 0.0, 0.0)], 4, fx(0.25));
    assert_eq!(coarse.line(0).len(), dense.line(0).len());
}

/// Mismatched lengths must not panic (positions longer than velocities): the
/// surplus particle sits in range of a cell and of the seed.
#[test]
fn mismatched_lengths_do_not_panic() {
    let p = [v3(1.0, -1.0, -1.0), v3(1.0, -1.0, -1.0)];
    let v = [v3(2.0, 0.0, 0.0)];
    let arrows = generate_flow_arrows(&p, &v, &cube(2, 1.0));
    assert!(!arrows.is_empty());
    let s = generate_streamlines(&p, &v, &[v3(1.0, -1.0, -1.0)], 2, fx(0.25));
    assert!(s.line(0).len() >= 2);
}

/// Search radius is dx, not sqrt(dx): with dx = 2 a particle at distance 1.5
/// from a cell centre is inside (1.5 < 2) while 1.5^2 = 2.25 > dx would have
/// excluded it under radius^2 = dx. Cell (1,0,0) centre is (1,-1,-1).
#[test]
fn search_radius_is_the_cell_size_not_its_square_root() {
    let arrows = generate_flow_arrows(&[v3(2.5, -1.0, -1.0)], &[v3(0.0, 1.0, 0.0)], &cube(2, 1.0));
    assert!(arrows.iter().any(|a| a.position == v3(1.0, -1.0, -1.0)));
}
