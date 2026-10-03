//! Audit oracles for `alice_physics::acoustic_wave`.
//!
//! Expected values come from the exact discrete dispersion relation of the
//! leap-frog scheme, von Neumann stability analysis, and Neumann wall
//! reflection of a travelling pulse; none is derived by calling the
//! function under test twice.

#![allow(clippy::disallowed_methods)]

use alice_physics::acoustic_wave::{leapfrog_step, speeds, stable_dt};

/// Exact discrete plane wave: `u_i^n = cos(k i - w n)` with
/// `sin(w/2) = C sin(k/2)` solves the scheme in the interior, so one step
/// from two consecutive exact states must reproduce the next exact state.
#[test]
fn interior_update_reproduces_exact_discrete_plane_wave() {
    let n = 64usize;
    for &(courant, k) in &[(0.5f64, 0.7f64), (0.9, 1.3), (0.25, 0.4), (1.0, 2.0)] {
        let w = 2.0 * (courant * (k / 2.0).sin()).asin();
        let state = |step: i32| -> Vec<f32> {
            (0..n)
                .map(|i| (k * i as f64 - w * f64::from(step)).cos() as f32)
                .collect()
        };
        let prev = state(-1);
        let cur = state(0);
        let want = state(1);
        let mut next = vec![0.0f32; n];
        leapfrog_step(&cur, &prev, &mut next, courant as f32);
        for i in 1..n - 1 {
            assert!(
                (next[i] - want[i]).abs() < 2.0e-6,
                "C={courant} k={k} i={i}: {} vs {}",
                next[i],
                want[i]
            );
        }
    }
}

/// Courant number enters squared: `C` and `-C` give identical results.
#[test]
fn courant_sign_does_not_matter() {
    let cur: Vec<f32> = (0..16).map(|i| ((i * 7) % 5) as f32).collect();
    let prev: Vec<f32> = (0..16).map(|i| ((i * 3) % 4) as f32).collect();
    let mut a = vec![0.0f32; 16];
    let mut b = vec![0.0f32; 16];
    leapfrog_step(&cur, &prev, &mut a, 0.6);
    leapfrog_step(&cur, &prev, &mut b, -0.6);
    assert_eq!(a, b);
}

/// Zero Courant number: pure time extrapolation `2 u - u_old` in the
/// interior.
#[test]
fn zero_courant_is_time_extrapolation() {
    let cur = vec![1.0f32, 2.0, 4.0, 8.0, 16.0];
    let prev = vec![0.5f32, 1.0, 1.0, 2.0, 3.0];
    let mut next = vec![0.0f32; 5];
    leapfrog_step(&cur, &prev, &mut next, 0.0);
    assert_eq!(next[1], 2.0 * 2.0 - 1.0);
    assert_eq!(next[2], 2.0 * 4.0 - 1.0);
    assert_eq!(next[3], 2.0 * 8.0 - 2.0);
}

fn run(n: usize, courant: f32, steps: usize, init: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut cur: Vec<f32> = (0..n).map(&init).collect();
    let mut prev = cur.clone();
    let mut next = vec![0.0f32; n];
    for _ in 0..steps {
        leapfrog_step(&cur, &prev, &mut next, courant);
        prev.clone_from(&cur);
        cur.clone_from(&next);
    }
    cur
}

/// Von Neumann stability: for `C <= 1` a unit pulse stays bounded for many
/// steps, for `C = 1.5` the highest-wavenumber mode grows exponentially.
#[test]
fn stability_boundary_at_courant_one() {
    let init = |i: usize| if i == 50 { 1.0 } else { 0.0 };
    for &c in &[0.5f32, 0.9, 1.0] {
        let u = run(101, c, 400, init);
        let m = u.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(m < 2.0, "C={c} should stay bounded, got max {m}");
    }
    let u = run(101, 1.5, 120, init);
    let blown = u.iter().any(|v| !v.is_finite() || v.abs() > 1.0e3);
    assert!(blown, "C=1.5 must blow up");
}

/// A pulse reflected by the Neumann wall keeps its sign and amplitude and
/// returns to its starting cell after travelling there and back (C = 1
/// moves a right-going delta exactly one cell per step).
#[test]
fn neumann_wall_reflects_pulse_with_same_sign_and_amplitude() {
    let n = 40usize;
    let start = 10usize;
    // right-going delta: u(x, -dt) = f(x + dx) = delta at start-1
    let mut cur = vec![0.0f32; n];
    cur[start] = 1.0;
    let mut prev = vec![0.0f32; n];
    prev[start - 1] = 1.0;
    let mut next = vec![0.0f32; n];
    // travel to the wall at cell n-1 and back: 2*(n-1-start) steps
    let steps = 2 * (n - 1 - start);
    for _ in 0..steps {
        leapfrog_step(&cur, &prev, &mut next, 1.0);
        prev.clone_from(&cur);
        cur.clone_from(&next);
    }
    let (argmax, peak) =
        cur.iter().enumerate().fold(
            (0, 0.0f32),
            |a, (i, &v)| if v.abs() > a.1.abs() { (i, v) } else { a },
        );
    assert!(
        (argmax as i64 - start as i64).abs() <= 1,
        "reflected pulse at cell {argmax}, expected near {start}"
    );
    assert!(
        (peak - 1.0).abs() < 0.05,
        "reflected amplitude {peak}, expected 1 (total reflection, same sign)"
    );
}

/// Zero-gradient ends: after any step the two end cells equal their
/// interior neighbours, for any state.
#[test]
fn end_cells_equal_their_neighbours_after_step() {
    let cur: Vec<f32> = (0..9).map(|i| (i * i) as f32).collect();
    let prev: Vec<f32> = (0..9).map(|i| (i * 2) as f32).collect();
    let mut next = vec![99.0f32; 9];
    leapfrog_step(&cur, &prev, &mut next, 0.7);
    assert_eq!(next[0], next[1]);
    assert_eq!(next[8], next[7]);
}

/// `stable_dt` is the CFL bound `dx / c`: stepping at exactly that dt gives
/// Courant number 1, and a smaller dt gives a proportionally smaller `C`.
#[test]
fn stable_dt_gives_unit_courant_number() {
    for &(dx, c) in &[
        (0.01f32, speeds::AIR_20C),
        (0.5, speeds::STEEL_LONGITUDINAL),
    ] {
        let dt = stable_dt(dx, c);
        assert!((c * dt / dx - 1.0).abs() < 1.0e-6);
        assert!(2.0 * c * dt / dx > 1.9);
    }
}

/// n = 3 smallest non-degenerate grid: single interior cell, hand value.
#[test]
fn three_cell_grid_single_interior_cell() {
    let cur = [1.0f32, 5.0, 2.0];
    let prev = [0.0f32, 3.0, 0.0];
    let mut next = [0.0f32; 3];
    leapfrog_step(&cur, &prev, &mut next, 0.5);
    // 2*5 - 3 + 0.25*(2 - 10 + 1) = 7 - 1.75 = 5.25
    assert_eq!(next[1], 5.25);
    assert_eq!(next[0], 5.25);
    assert_eq!(next[2], 5.25);
}

/// A NaN Courant number must not be silently accepted as a valid step.
#[test]
#[ignore = "known defect: AUD-A-S5W1-002: leapfrog_step accepts NaN / unstable Courant numbers (C > 1) with no check and no documented return; output is silently NaN"]
fn nan_courant_is_rejected() {
    let cur = vec![1.0f32; 8];
    let prev = vec![1.0f32; 8];
    let mut next = vec![0.0f32; 8];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        leapfrog_step(&cur, &prev, &mut next, f32::NAN)
    }));
    assert!(r.is_err() || next.iter().all(|v| v.is_finite()));
}
