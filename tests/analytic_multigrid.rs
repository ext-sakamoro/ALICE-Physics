//! Oracles for `eulerian_grid::project_pressure_multigrid` (W-cycle with a
//! Galerkin coarse operator `A_c = R A P`, piecewise-constant aggregation,
//! coarse correction x2).
//!
//! History: the oracles were written BEFORE the implementation (red against a
//! `todo!()` stub) and validated against the existing red-black Gauss-Seidel
//! solver (`project_pressure`) by the ignored diagnostic
//! `gs_satisfies_exact_solution_oracle`, so a red means "the solver is wrong"
//! and not "the oracle cannot be met".
//!
//! Why a W-cycle: with piecewise-constant aggregation a V-cycle does not
//! reach a grid-independent rate. Measured on the smooth scene (n = 8/16/32):
//! V + correction x1 = 0.46 / 0.73 / 0.82, V + correction x2 = 0.84 / 1.45 /
//! 2.32 (diverges), W + correction x2 = 0.294 / 0.283 / 0.294 (flat). The
//! correction factor 2 follows from the Galerkin product: summation
//! restriction and injection make `A_c` four times a re-discretised
//! operator while the restricted residual is eight times an average, so the
//! geometrically consistent correction is 8/4 = 2.
//!
//! Oracles:
//! 1. `exact_solution_is_reproduced`: Dirichlet box, `dx = dt = rho = 1`,
//!    `p(i,j,k) = f(i) f(j) f(k)` with `f(i) = (i+1)(n-i)`, face velocity
//!    `u(i,j,k) = p(i,j,k) - p(i-1,j,k)` (out of range p = 0). Then
//!    `div u == A p` exactly (the 2nd difference is exact on quadratics and
//!    `f` vanishes at the ghost cells `-1` and `n`), so a converged projection
//!    removes the whole velocity field.
//! 2. `contraction_rate_is_grid_independent`: per-cycle reduction of
//!    `max|div|` must not worsen with `n`. Red-black GS rates at the same
//!    counts are measured in the same test as the control group.
//! 3. `walls_keep_the_rate`: interior baffle walls; the rate must stay close
//!    to the wall-free rate (this is what the Galerkin operator is for; a
//!    re-discretised coarse operator that ignores the baffle would not).
//! 4. `result_is_bit_deterministic`.
//! 5. `degenerate_inputs_do_not_panic`.
//!
//! Design memo for oracle 5: the function returns `()`, so there is no `Err`
//! channel. The contract these tests pin is therefore "never panics; inputs
//! it cannot solve (extent not a power of two, `dx == 0`, `cycles == 0`)
//! return early leaving the grid bit-identical" (confirmed by the user). A
//! fully sealed box is the singular pure-Neumann problem; it is not
//! special-cased (no mean-pressure pinning) so that it behaves like the
//! existing Gauss-Seidel solver, and the test only pins "no panic, divergence
//! does not grow, wall faces stay zero".
//!
//! Tolerances are named consts fixed from measurements with the margin stated
//! on each; `mg_rate_report` (ignored) reprints the numbers.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::eulerian_grid::{project_pressure, project_pressure_multigrid, FaceBc, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

/// Residual velocity after convergence, relative to the initial `max|u|`.
///
/// Measured (`mg_rate_report`) after `EXACT_CYCLES = 24`: n=8 2.04e-12,
/// n=16 1.22e-11 (n=32 3.28e-11); the bound is about 80x above the worst
/// tested size. The residual is a convergence residual, not a Fix128 floor
/// (32 cycles reach 3e-15).
const EXACT_RESIDUAL_REL: f64 = 1e-9;
/// Cycles used by oracle 1 ("sufficiently converged"). Measured per-cycle
/// reduction on this field is 0.32-0.34, so 24 cycles are 1e-11 level.
const EXACT_CYCLES: u32 = 24;
/// Upper bound of the geometric-mean per-cycle reduction of `max|div|`.
///
/// Measured over smooth / mixed / perforated / cavity scenes at n = 8/16/32:
/// 0.2816 to 0.3273 (worst: cavity n=8). The bound is 0.35, 7% above the worst
/// measurement and below the 0.360 / 0.376 of mutation M5. The red-black
/// GS control on the smooth scene is 0.924 / 0.981 / 0.995 at the same counts.
const RATE_MAX: f64 = 0.35;
/// Allowed spread of that rate between the smallest and largest `n`.
///
/// Measured spread: smooth 0.011 (0.2834 to 0.2943), mixed 0.012. The bound
/// is 0.05, 4x the measurement; the V-cycle that was rejected spread 0.36.
const RATE_SPREAD_MAX: f64 = 0.05;
/// Allowed `rate_with_walls - rate_without_walls`.
///
/// Measured against the open scene (0.2932 / 0.2821 / 0.2924 for n = 8/16/32):
/// perforated wall +0.0 / +0.013 / +0.013, cavity +0.034 / +0.006 / +0.017. The
/// bound is 0.05, 47% above the worst measurement. A coarse operator that
/// treats partly blocked faces as open (mutation M5) measured cavity
/// 0.360 / 0.335 / 0.376, i.e. +0.067 / +0.053 / +0.083, which this bound
/// rejects. Limit of this oracle: a coarse operator that ignores the walls
/// altogether does not slow the rate on these scenes (cavity 0.300 / 0.280 /
/// 0.293), so it is only caught by `result_matches_pinned_bits`.
const WALL_RATE_EXCESS_MAX: f64 = 0.05;
/// Below this `max|div|` a ratio is dominated by Fix128 truncation, not by
/// the solver; the measurement stops there.
const DIV_FLOOR: f64 = 1e-12;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_ratio(n, 1)
}

fn iu(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + (g.nx + 1) * (j + g.ny * k)
}
fn iv(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + g.nx * (j + (g.ny + 1) * k)
}
fn iw(g: &MacGrid, i: usize, j: usize, k: usize) -> usize {
    i + g.nx * (j + g.ny * k)
}

/// Exact Dirichlet-box pressure `f(i) f(j) f(k)`, zero outside the box.
fn p_exact(n: usize, i: i64, j: i64, k: i64) -> i64 {
    let n = n as i64;
    let f = |a: i64| {
        if a < 0 || a >= n {
            0
        } else {
            (a + 1) * (n - a)
        }
    };
    f(i) * f(j) * f(k)
}

/// Face velocities `= grad p_exact` (dx = 1), as in oracle 1.
fn exact_scene(n: usize) -> MacGrid {
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let (a, b, c) = (i as i64, j as i64, k as i64);
                let ix = iu(&g, i, j, k);
                g.u[ix] = int(p_exact(n, a, b, c) - p_exact(n, a - 1, b, c));
            }
        }
    }
    for k in 0..n {
        for j in 0..=n {
            for i in 0..n {
                let (a, b, c) = (i as i64, j as i64, k as i64);
                let ix = iv(&g, i, j, k);
                g.v[ix] = int(p_exact(n, a, b, c) - p_exact(n, a, b - 1, c));
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                let (a, b, c) = (i as i64, j as i64, k as i64);
                let ix = iw(&g, i, j, k);
                g.w[ix] = int(p_exact(n, a, b, c) - p_exact(n, a, b, c - 1));
            }
        }
    }
    g
}

/// Deterministic LCG noise in `{-8..=8} / 8` (dyadic).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> Fix128 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        q(((self.0 >> 33) % 17) as i64 - 8, 8)
    }
}

/// Smooth exact-gradient field scaled by `128 / n^5` (dyadic, ~O(8)) plus
/// full-spectrum noise, so both smooth and rough error components exist.
fn mixed_scene(n: usize, seed: u64) -> MacGrid {
    let mut g = exact_scene(n);
    let scale = q(128, (n as i64).pow(5));
    let mut rng = Lcg(seed);
    for x in g.u.iter_mut().chain(g.v.iter_mut()).chain(g.w.iter_mut()) {
        *x = *x * scale + rng.next();
    }
    g
}

/// `mixed_scene` plus a perforated wall: the interior plane `i = n/2` is a
/// wall except at the faces with `j` and `k` both even, and a short interior
/// Y-wall.
///
/// Every 2x2 aggregate face of that plane holds exactly one open fine face,
/// so its Galerkin conductance is 1 of a possible 4. A baffle aligned with the
/// aggregates would be all-or-nothing on the coarse level, and a coarse
/// operator that treats a partly blocked face as open (a re-discretisation)
/// could not be told from the Galerkin one.
fn walled_scene(n: usize, seed: u64) -> MacGrid {
    let mut g = mixed_scene(n, seed);
    let zero = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    for k in 0..n {
        for j in 0..n {
            if j % 2 == 0 && k % 2 == 0 {
                continue;
            }
            g.set_u_bc(n / 2, j, k, FaceBc::Wall { velocity: zero });
        }
    }
    for k in 0..n / 2 {
        g.set_v_bc(n / 4, n / 2, k, FaceBc::Wall { velocity: zero });
    }
    g
}

/// `mixed_scene` plus a sealed cavity: the cells `[n/4, 3n/4)^3` are enclosed
/// by walls on all six sides except one fine face (the +x wall at its
/// lowest `(j, k)`). The cavity talks to the Dirichlet exterior through that
/// single face, so its pressure is set by a conductance of 1 where an
/// aggregate face could hold 4. A coarse operator that ignores the walls, or
/// that treats a partly blocked face as open, gets the cavity wrong by
/// construction.
fn cavity_scene(n: usize, seed: u64) -> MacGrid {
    let mut g = mixed_scene(n, seed);
    let zero = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    let wall = FaceBc::Wall { velocity: zero };
    let (a, b) = (n / 4, 3 * n / 4);
    for p in a..b {
        for q in a..b {
            g.set_u_bc(a, p, q, wall);
            if (p, q) != (a, a) {
                g.set_u_bc(b, p, q, wall);
            }
            g.set_v_bc(p, a, q, wall);
            g.set_v_bc(p, b, q, wall);
            g.set_w_bc(p, q, a, wall);
            g.set_w_bc(p, q, b, wall);
        }
    }
    g
}

/// Noise-only field on an arbitrary extent (dx = 1).
fn noise_scene(nx: usize, ny: usize, nz: usize) -> MacGrid {
    let mut g = MacGrid::new(nx, ny, nz, Fix128::ONE);
    let mut rng = Lcg(99);
    for x in g.u.iter_mut().chain(g.v.iter_mut()).chain(g.w.iter_mut()) {
        *x = rng.next();
    }
    g
}

fn max_abs_u(g: &MacGrid) -> f64 {
    g.u.iter()
        .chain(g.v.iter())
        .chain(g.w.iter())
        .map(|x| x.to_f64().abs())
        .fold(0.0, f64::max)
}

fn max_div(g: &MacGrid) -> f64 {
    let mut m = 0.0f64;
    for k in 0..g.nz {
        for j in 0..g.ny {
            for i in 0..g.nx {
                m = m.max(g.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    m
}

fn same_state(a: &MacGrid, b: &MacGrid) -> bool {
    a.u == b.u && a.v == b.v && a.w == b.w && a.pressure == b.pressure
}

/// `max|div|` after `r` solver steps, for r = 0..=4 (r = 0 is the input).
/// `mg` selects multigrid cycles or red-black GS iterations.
fn residual_history(scene: &dyn Fn() -> MacGrid, mg: bool) -> Vec<f64> {
    let mut out = vec![max_div(&scene())];
    for r in 1..=4u32 {
        let mut g = scene();
        if mg {
            project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, r);
        } else {
            project_pressure(&mut g, Fix128::ONE, Fix128::ONE, r);
        }
        out.push(max_div(&g));
    }
    out
}

/// Geometric-mean per-step reduction `(r4/r1)^(1/3)`. `None` when the
/// residual hit the Fix128 floor before the measurement window ended.
fn mean_rate(h: &[f64]) -> Option<f64> {
    if h[4] < DIV_FLOOR || h[1] < DIV_FLOOR {
        return None;
    }
    // Cube root by bisection: `f64::powf` is a disallowed method in this
    // crate (platform libm); the monotone `x^3` needs only multiplication.
    let ratio = h[4] / h[1];
    let (mut lo, mut hi) = (0.0f64, ratio.max(1.0));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid * mid * mid < ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

#[test]
fn exact_solution_is_reproduced() {
    for n in [8usize, 16] {
        let mut g = exact_scene(n);
        let before = max_abs_u(&g);
        assert!(before > 1.0, "scene must start with a nonzero field");
        project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, EXACT_CYCLES);
        let after = max_abs_u(&g);
        assert!(
            after <= before * EXACT_RESIDUAL_REL,
            "n={n}: max|u| {before} -> {after}, expected <= {EXACT_RESIDUAL_REL} of initial"
        );
    }
}

#[test]
fn contraction_rate_is_grid_independent() {
    // `smooth` is the discriminating scene: measured GS rates there are
    // 0.924 / 0.981 / 0.995 for n = 8 / 16 / 32 (1 - O(1/n^2)). The noisy
    // `mixed` scene is dominated by high-frequency error that GS damps well,
    // so its GS rates (0.912 / 0.946 / 0.895) barely show the n dependence;
    // it stays as the second scene for the "rough error" side.
    type Scene = fn(usize) -> MacGrid;
    let scenes: [(&str, Scene); 2] = [
        ("smooth", smooth_scene),
        ("mixed", |n| mixed_scene(n, 0xA11CE)),
    ];
    for (name, make) in scenes {
        let mut mg_rates = Vec::new();
        for n in [8usize, 16, 32] {
            let scene = move || make(n);
            // Control group first: it is measurable today.
            let gs = residual_history(&scene, false);
            eprintln!(
                "[mg-oracle b] {name} GS n={n}: max|div| r0..r4 = {gs:?}, mean rate = {:?}",
                mean_rate(&gs)
            );
            let mg = residual_history(&scene, true);
            eprintln!(
                "[mg-oracle b] {name} MG n={n}: max|div| r0..r4 = {mg:?}, mean rate = {:?}",
                mean_rate(&mg)
            );
            let rate = mean_rate(&mg).unwrap_or(0.0);
            assert!(rate <= RATE_MAX, "{name} n={n}: rate {rate} > {RATE_MAX}");
            mg_rates.push(rate);
        }
        let lo = mg_rates.iter().cloned().fold(f64::MAX, f64::min);
        let hi = mg_rates.iter().cloned().fold(0.0, f64::max);
        assert!(
            hi - lo <= RATE_SPREAD_MAX,
            "{name}: rate depends on n: {mg_rates:?}"
        );
    }
}

#[test]
fn walls_keep_the_rate() {
    type Scene = fn(usize, u64) -> MacGrid;
    let scenes: [(&str, Scene); 2] = [("perforated", walled_scene), ("cavity", cavity_scene)];
    for n in [8usize, 16, 32] {
        let open = residual_history(&move || mixed_scene(n, 0xBEEF), true);
        let ro = mean_rate(&open).unwrap_or(0.0);
        for (name, make) in scenes {
            let walled = residual_history(&move || make(n, 0xBEEF), true);
            eprintln!("[mg-oracle c] n={n} {name}: open {open:?} walled {walled:?}");
            let rw = mean_rate(&walled).unwrap_or(0.0);
            assert!(rw <= RATE_MAX, "n={n} {name}: rate {rw} > {RATE_MAX}");
            assert!(
                rw - ro <= WALL_RATE_EXCESS_MAX,
                "n={n} {name}: walls degrade the rate {ro} -> {rw}"
            );
        }
    }
}

#[test]
fn result_is_bit_deterministic() {
    for scene in [mixed_scene(16, 7), walled_scene(16, 7)] {
        let mut a = scene.clone();
        let mut b = scene.clone();
        project_pressure_multigrid(&mut a, q(1, 100), Fix128::ONE, 3);
        project_pressure_multigrid(&mut b, q(1, 100), Fix128::ONE, 3);
        assert!(same_state(&a, &b), "two runs differ");
        assert!(!same_state(&a, &scene), "solver must change the field");
    }
}

/// FNV-1a over the raw `hi` / `lo` words of pressure and the three face
/// arrays, in storage order.
fn checksum(g: &MacGrid) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in g
        .pressure
        .iter()
        .chain(g.u.iter())
        .chain(g.v.iter())
        .chain(g.w.iter())
    {
        for word in [x.hi as u64, x.lo] {
            for b in word.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

/// Two runs agreeing says nothing about the visiting order: a fixed but
/// different order is also reproducible. This pins the bits of one run so
/// that a change of visiting order, smoother count or correction factor is
/// loud. Re-pin only on purpose, together with the doc of the changed
/// parameter.
const GOLDEN_WALLED_16: u64 = 0xd3379b243c508c3b;
const GOLDEN_MIXED_8: u64 = 0xca146c4786121f08;

#[test]
fn result_matches_pinned_bits() {
    let mut a = walled_scene(16, 7);
    project_pressure_multigrid(&mut a, q(1, 100), Fix128::ONE, 3);
    let mut b = mixed_scene(8, 7);
    project_pressure_multigrid(&mut b, q(1, 100), Fix128::ONE, 3);
    eprintln!(
        "[mg-golden] walled16 = {:#x} mixed8 = {:#x}",
        checksum(&a),
        checksum(&b)
    );
    assert_eq!(checksum(&a), GOLDEN_WALLED_16, "walled 16 bits changed");
    assert_eq!(checksum(&b), GOLDEN_MIXED_8, "mixed 8 bits changed");
}

fn no_panic<F: FnOnce()>(what: &str, f: F) {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    assert!(r.is_ok(), "{what}: panicked");
}

#[test]
fn degenerate_inputs_do_not_panic() {
    // Extent not a power of two: early return, grid untouched.
    no_panic("6x6x6", || {
        let mut g = noise_scene(6, 6, 6);
        let before = g.clone();
        project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, 3);
        assert!(
            same_state(&g, &before),
            "non-2^k grid must be left untouched"
        );
    });
    // One cell thick (1 = 2^0 is a power of two): must solve, not panic.
    no_panic("8x8x1", || {
        let mut g = noise_scene(8, 8, 1);
        let before = max_div(&g);
        project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, 3);
        assert!(max_div(&g) <= before, "divergence must not grow");
    });
    // dx = 0, cycles = 0, dt = 0, density = 0: untouched. A wall face carrying a
    // nonzero velocity makes "untouched" observable, because the solver
    // would zero it when it enforces the walls.
    let walled_noise = || {
        let mut g = noise_scene(8, 8, 8);
        let zero = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        g.set_u_bc(4, 3, 3, FaceBc::Wall { velocity: zero });
        let ix = iu(&g, 4, 3, 3);
        g.u[ix] = Fix128::ONE;
        g
    };
    for (what, dx, dt, rho, cycles) in [
        ("dx=0", Fix128::ZERO, Fix128::ONE, Fix128::ONE, 3u32),
        ("cycles=0", Fix128::ONE, Fix128::ONE, Fix128::ONE, 0),
        ("dt=0", Fix128::ONE, Fix128::ZERO, Fix128::ONE, 3),
        ("density=0", Fix128::ONE, Fix128::ONE, Fix128::ZERO, 3),
    ] {
        no_panic(what, || {
            let mut g = walled_noise();
            g.dx = dx;
            let before = g.clone();
            project_pressure_multigrid(&mut g, dt, rho, cycles);
            assert!(
                same_state(&g, &before),
                "{what} must leave the grid untouched"
            );
        });
    }
    // Huge dt / density: a panic is not acceptable, the result is not
    // pinned (see `huge_values_report`).
    no_panic("huge dt", || {
        let mut g = noise_scene(8, 8, 8);
        project_pressure_multigrid(&mut g, int(1 << 40), Fix128::ONE, 3);
    });
    no_panic("huge density", || {
        let mut g = noise_scene(8, 8, 8);
        project_pressure_multigrid(&mut g, Fix128::ONE, int(1 << 40), 3);
    });
    // Fully closed box: singular Poisson problem (constant null space).
    no_panic("closed box", || {
        let mut g = noise_scene(8, 8, 8);
        g.set_closed_box_walls();
        g.enforce_face_boundaries();
        let before = max_div(&g);
        project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, 4);
        assert!(max_div(&g) <= before, "divergence must not grow");
        for k in 0..8 {
            for j in 0..8 {
                assert_eq!(g.u(0, j, k), Fix128::ZERO, "wall face moved");
                assert_eq!(g.u(8, j, k), Fix128::ZERO, "wall face moved");
            }
        }
    });
}

/// Validity check of oracle 1 against the existing solver: if red-black GS
/// with enough sweeps cannot satisfy it, the oracle itself is wrong.
#[test]
#[ignore = "diagnostic: oracle validity against the existing GS"]
fn gs_satisfies_exact_solution_oracle() {
    for (n, iters) in [(8usize, 400u32), (16, 1500)] {
        let mut g = exact_scene(n);
        let before = max_abs_u(&g);
        project_pressure(&mut g, Fix128::ONE, Fix128::ONE, iters);
        let after = max_abs_u(&g);
        eprintln!(
            "[mg-oracle a/GS] n={n} iters={iters}: max|u| {before} -> {after} (rel {:e})",
            after / before
        );
        assert!(
            after <= before * EXACT_RESIDUAL_REL,
            "n={n}: GS rel {:e}",
            after / before
        );
    }
}

/// Control group of oracle 2: red-black GS rates for the same scenes and
/// counts. GS is expected to worsen with `n`; this prints the numbers.
#[test]
#[ignore = "diagnostic: GS control rates for oracle 2"]
fn gs_control_rates() {
    for n in [8usize, 16, 32] {
        let h = residual_history(&move || mixed_scene(n, 0xA11CE), false);
        let step: Vec<f64> = (1..5).map(|r| h[r] / h[r - 1]).collect();
        eprintln!(
            "[mg-oracle b/GS] n={n}: max|div| r0..r4 = {h:?} step ratios = {step:?} mean rate (r4/r1)^(1/3) = {:?}",
            mean_rate(&h)
        );
    }
    // Smooth-only variant (no noise): the error GS damps worst.
    for n in [8usize, 16, 32] {
        let h = residual_history(&move || smooth_scene(n), false);
        let step: Vec<f64> = (1..5).map(|r| h[r] / h[r - 1]).collect();
        eprintln!(
            "[mg-oracle b/GS smooth-only] n={n}: r0..r4 = {h:?} step ratios = {step:?} mean rate = {:?}",
            mean_rate(&h)
        );
    }
}

/// Smallest iteration count after which `solve` leaves `max|u|` of the exact
/// scene at or below `target` of the initial value: doubling, then bisection
/// (the residual is monotone in the count to the precision that matters
/// here).
fn iterations_to_reach(n: usize, target: f64, solve: &dyn Fn(&mut MacGrid, u32)) -> u32 {
    let reached = |count: u32| {
        let mut g = exact_scene(n);
        let before = max_abs_u(&g);
        solve(&mut g, count);
        max_abs_u(&g) <= before * target
    };
    let mut hi = 1u32;
    while !reached(hi) {
        hi *= 2;
        assert!(hi < 1 << 20, "target not reached");
    }
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if reached(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Cost comparison on oracle 1: cycles of the multigrid solver against
/// iterations of red-black GS for the same `EXACT_RESIDUAL_REL`. Run with
/// `--release`: the GS side at n = 32 is slow in a debug build.
#[test]
#[ignore = "diagnostic: cycles vs GS iterations to the same precision (use --release)"]
fn iterations_to_same_precision() {
    for n in [8usize, 16, 32] {
        let mg = iterations_to_reach(n, EXACT_RESIDUAL_REL, &|g, c| {
            project_pressure_multigrid(g, Fix128::ONE, Fix128::ONE, c)
        });
        let gs = iterations_to_reach(n, EXACT_RESIDUAL_REL, &|g, c| {
            project_pressure(g, Fix128::ONE, Fix128::ONE, c)
        });
        eprintln!(
            "[mg-iters] n={n}: target {EXACT_RESIDUAL_REL:e}: multigrid {mg} cycles, GS {gs} iterations (ratio {:.1})",
            f64::from(gs) / f64::from(mg)
        );
    }
}

/// Multigrid rates for every scene and size without asserting: the numbers
/// the thresholds are fixed from.
#[test]
#[ignore = "diagnostic: multigrid rates used to fix the thresholds"]
fn mg_rate_report() {
    for n in [8usize, 16, 32] {
        let mut line = String::new();
        for cycles in [4u32, 8, 12, 16, 20, 24, 32] {
            let mut g = exact_scene(n);
            let before = max_abs_u(&g);
            project_pressure_multigrid(&mut g, Fix128::ONE, Fix128::ONE, cycles);
            line.push_str(&format!(" {cycles}:{:.2e}", max_abs_u(&g) / before));
        }
        eprintln!("[mg-report] exact n={n} rel max|u| by cycles:{line}");
    }
    for n in [8usize, 16, 32] {
        for (name, h) in [
            ("smooth", residual_history(&move || smooth_scene(n), true)),
            (
                "mixed",
                residual_history(&move || mixed_scene(n, 0xA11CE), true),
            ),
            (
                "mixed-open",
                residual_history(&move || mixed_scene(n, 0xBEEF), true),
            ),
            (
                "walled",
                residual_history(&move || walled_scene(n, 0xBEEF), true),
            ),
            (
                "cavity",
                residual_history(&move || cavity_scene(n, 0xBEEF), true),
            ),
        ] {
            eprintln!(
                "[mg-report] n={n} {name}: rate = {:?} r0..r4 = {h:?}",
                mean_rate(&h)
            );
        }
    }
}

/// `exact_scene` scaled to O(8), without noise.
fn smooth_scene(n: usize) -> MacGrid {
    let mut g = exact_scene(n);
    let scale = q(128, (n as i64).pow(5));
    for x in g.u.iter_mut().chain(g.v.iter_mut()).chain(g.w.iter_mut()) {
        *x = *x * scale;
    }
    g
}
