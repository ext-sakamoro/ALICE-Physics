//! Sources, Gauss's law, the absorber and the Courant bookkeeping of the Yee
//! lattice, each printed next to the closed form it is supposed to reproduce.
//!
//! Everything is in the module's normalised units (`c = ε₀ = μ₀ = Δx = 1`), so
//! the Courant number *is* the time step and lengths are counted in cells.
//! The module header explains why SI is not an option in Q64.64.
//!
//! Four things are worth watching in the output.
//!
//! **Gauss's law is a consequence, not a constraint.** `∇·E − ρ` is read
//! off the lattice by [`YeeGrid::gauss_residual`]; the solver never imposes
//! it. With a consistent initial placement (`ρ` set to the node divergence of
//! a seeded `E`) and a dyadic Courant number the residual stays at the zero
//! bit pattern step after step, because `∇·(∇×H)` telescopes exactly while the
//! arithmetic is exact. An *inconsistent* placement is not corrected either:
//! its residual is a constant of the motion.
//!
//! **A current that ends deposits charge at `∓S·J` per step.** The two ends
//! cancel, so [`YeeGrid::total_charge`] stays exactly zero while
//! [`YeeGrid::charge`] at each end grows linearly.
//!
//! **`∇·B` is a structural identity of the staggered curl, and the bits show
//! it only while the arithmetic is exact.** With `S = 1/2` and an integer
//! seed every product is a shift, so [`YeeGrid::max_abs_div_b`] is the zero
//! bit pattern for tens of steps; with `S = 9/16` the truncation of
//! `Fix128` multiplication breaks distributivity and a residual of a few ULP
//! per step appears. The identity is exact; the arithmetic is not.
//!
//! **The absorber is a layer, not a medium.** [`YeeGrid::is_absorbing`] maps
//! which samples take the split-field update — the normal component of a slab
//! is left alone — and [`theoretical_pml_reflection`] is the continuum floor
//! the layer aims at, which the table prints next to the same formula
//! evaluated through `det_math::exp64`.
//!
//! ```bash
//! cargo run --example maxwell_sources_and_absorber --features std
//! ```

use alice_physics::det_math::{exp64, sqrt64};
use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    cfl_limit_3d, loss_coefficients, theoretical_pml_reflection, Absorber, Component, YeeGrid,
    COURANT_3D,
};

const TAG: &str = "[maxwell_fdtd]";

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn raw(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

/// `S·√(1/Δx² + 1/Δy² + 1/Δz²)` with unit cells: the Courant number of a
/// step against the 3-D stability bound, by hand.
fn courant_bookkeeping() {
    let three = int(3);
    let sqrt3 = three.sqrt();
    let limit = cfl_limit_3d();
    println!(
        "{TAG} courant: S = COURANT_3D = {COURANT_3D} (9/16 = {})",
        q(9, 16)
    );
    println!(
        "{TAG} courant: cfl_limit_3d = {limit}, 1/sqrt(3) via det_math::sqrt64 = {}",
        1.0 / sqrt64(3.0)
    );
    println!(
        "{TAG} courant: S*sqrt(3) = {} (< 1 is stable), limit*sqrt(3) = {} (= 1 by definition, {} ULP off)",
        COURANT_3D * sqrt3,
        limit * sqrt3,
        raw(limit * sqrt3 - Fix128::ONE)
    );
    let just_over = q(37, 64);
    println!(
        "{TAG} courant: 37/64*sqrt(3) = {} (> 1, the next dyadic up is unstable)",
        just_over * sqrt3
    );

    let grid = YeeGrid::new(3, 5, 7, q(3, 8));
    let (nx, ny, nz) = grid.dims();
    println!(
        "{TAG} courant: dims = ({nx}, {ny}, {nz}) [expect (3, 5, 7)], courant = {} [expect 3/8 = {}]",
        grid.courant(),
        q(3, 8)
    );
}

/// `max |field|` over a hand-seeded lattice is the largest seeded magnitude.
fn field_probe() {
    let mut grid = YeeGrid::new(4, 4, 4, COURANT_3D);
    println!(
        "{TAG} probe: fresh lattice max |field| = {} [expect 0]",
        grid.max_abs_field()
    );
    grid.set(Component::Ex, 1, 1, 1, q(3, 7));
    grid.set(Component::Hy, 2, 2, 2, q(-11, 5));
    grid.set(Component::Ez, 1, 2, 1, q(1, 2));
    println!(
        "{TAG} probe: seeded 3/7, -11/5, 1/2 -> max |field| = {} [expect 11/5 = {}]",
        grid.max_abs_field(),
        q(11, 5)
    );
}

/// `∇·E = ρ` on a 4×4×4 cavity with a seeded edge and a consistent charge.
fn gauss_law_with_a_placed_charge() {
    let s = q(1, 2);
    let mut grid = YeeGrid::new(4, 4, 4, s);
    let e = int(3);
    // One Ex edge at (1, 2, 2) enters div E with +e at node (1, 2, 2) and
    // -e at node (2, 2, 2); every other interior node sees zero.
    grid.set(Component::Ex, 1, 2, 2, e);
    println!(
        "{TAG} gauss: div_e(1,2,2) = {} [expect +3], div_e(2,2,2) = {} [expect -3], div_e(3,2,2) = {} [expect 0]",
        grid.div_e(1, 2, 2),
        grid.div_e(2, 2, 2),
        grid.div_e(3, 2, 2)
    );
    println!(
        "{TAG} gauss: before placing charge, max |div E - rho| = {} [expect 3]",
        grid.max_abs_gauss_residual()
    );
    grid.set_charge(1, 2, 2, e);
    grid.set_charge(2, 2, 2, -e);
    println!(
        "{TAG} gauss: charge(1,2,2) = {}, charge(2,2,2) = {}, total_charge = {} [expect 0]",
        grid.charge(1, 2, 2),
        grid.charge(2, 2, 2),
        grid.total_charge()
    );
    println!(
        "{TAG} gauss: after placing charge, gauss_residual(1,2,2) = {}, (2,2,2) = {}, max = {} [expect 0, 0, 0]",
        grid.gauss_residual(1, 2, 2),
        grid.gauss_residual(2, 2, 2),
        grid.max_abs_gauss_residual()
    );
    for n in 1..=30 {
        grid.step();
        if n % 10 == 0 {
            println!(
                "{TAG} gauss: step {n:2}: max |div E - rho| = {} ULP [expect 0 while S*field is dyadic], max |field| = {}",
                raw(grid.max_abs_gauss_residual()),
                grid.max_abs_field()
            );
        }
    }

    // An inconsistent placement is not corrected: the residual is a constant
    // of the motion, so an extra 5 at (2, 2, 2) is carried unchanged.
    let mut off = YeeGrid::new(4, 4, 4, s);
    off.set(Component::Ex, 1, 2, 2, e);
    off.set_charge(2, 2, 2, int(5));
    let before = off.gauss_residual(2, 2, 2);
    for _ in 0..30 {
        off.step();
    }
    println!(
        "{TAG} gauss: inconsistent rho: residual(2,2,2) = {before} before and {} after 30 steps [expect -3 - 5 = -8 both]",
        off.gauss_residual(2, 2, 2)
    );
}

/// A current on one edge that ends inside the lattice.
fn a_current_deposits_charge() {
    let s = q(1, 2);
    let j = int(3);
    let mut grid = YeeGrid::new(4, 4, 4, s);
    grid.set_current(Component::Ex, 1, 2, 2, j);
    println!(
        "{TAG} current: current(Ex,1,2,2) = {} [expect 3], current(Ey,2,1,2) = {} [expect 0]",
        grid.current(Component::Ex, 1, 2, 2),
        grid.current(Component::Ey, 2, 1, 2)
    );
    for n in 1..=20i64 {
        grid.step();
        if n % 5 == 0 {
            println!(
                "{TAG} current: step {n:2}: charge(1,2,2) = {} [expect -n*S*J = {}], charge(2,2,2) = {} [expect {}], total = {} [expect 0], max |div E - rho| = {} ULP [expect 0]",
                grid.charge(1, 2, 2),
                -(s * j * int(n)),
                grid.charge(2, 2, 2),
                s * j * int(n),
                grid.total_charge(),
                raw(grid.max_abs_gauss_residual())
            );
        }
    }
}

/// `∇·B` by hand on one cell, then under the update at two Courant numbers.
fn div_b_is_structural() {
    let mut grid = YeeGrid::new(4, 4, 4, q(1, 2));
    grid.set(Component::Hx, 1, 0, 0, int(5));
    grid.set(Component::Hx, 0, 0, 0, int(2));
    grid.set(Component::Hy, 0, 1, 0, int(-1));
    grid.set(Component::Hz, 0, 0, 1, int(4));
    println!(
        "{TAG} div_b: hand stencil (5-2) + (-1-0) + (4-0) = 6, div_b(0,0,0) = {}, max_abs_div_b = {}",
        grid.div_b(0, 0, 0),
        grid.max_abs_div_b()
    );

    for (label, s) in [
        ("S = 1/2 (dyadic, integer seed)", q(1, 2)),
        ("S = 9/16", COURANT_3D),
    ] {
        let mut grid = YeeGrid::new(4, 4, 4, s);
        // The (a = 4, m = 2) ⊗ (a = 4, m = 2) cavity mode from the oracle table.
        let v = [0i64, 1, 0, -1, 0];
        for (i, &vi) in v.iter().enumerate() {
            for (j, &vj) in v.iter().enumerate() {
                for k in 0..4 {
                    grid.set(Component::Ez, i, j, k, int(vi * vj));
                }
            }
        }
        let mut worst = 0i128;
        for _ in 0..40 {
            grid.step();
            worst = worst.max(raw(grid.max_abs_div_b()));
        }
        println!(
            "{TAG} div_b: {label}: worst max |div B| over 40 steps = {worst} ULP, max |H| after = {}",
            [Component::Hx, Component::Hy, Component::Hz]
                .iter()
                .map(|&c| {
                    let (ni, nj, nk) = grid.component_dims(c);
                    let mut m = Fix128::ZERO;
                    for i in 0..ni {
                        for j in 0..nj {
                            for k in 0..nk {
                                m = m.max(grid.get(c, i, j, k).abs());
                            }
                        }
                    }
                    m
                })
                .fold(Fix128::ZERO, Fix128::max)
        );
    }
}

/// The PML: which samples it damps, the coefficients it damps them with, the
/// continuum reflection it aims at, and what is left in the box afterwards.
fn absorber() {
    let n = 12usize;
    let depth = [3usize, 2, 4];
    let sigma_max = int(4);
    let pml = Absorber::GradedPml { depth, sigma_max };
    let grid = YeeGrid::new_with_absorber(n, n, n, COURANT_3D, pml);
    let mid = n / 2;
    println!(
        "{TAG} absorber: {pml:?} on {n}^3: is_absorbing(Ex,0,mid,mid) = {} [expect false: normal component of the x slab], (Ey,0,mid,mid) = {} [expect true], (Ez,mid,mid,mid) = {} [expect false: centre]",
        grid.is_absorbing(Component::Ex, 0, mid, mid),
        grid.is_absorbing(Component::Ey, 0, mid, mid),
        grid.is_absorbing(Component::Ez, mid, mid, mid)
    );
    let mut lossless_cells = 0usize;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let faces = [
                    grid.is_absorbing(Component::Hx, i, j, k),
                    grid.is_absorbing(Component::Hx, i + 1, j, k),
                    grid.is_absorbing(Component::Hy, i, j, k),
                    grid.is_absorbing(Component::Hy, i, j + 1, k),
                    grid.is_absorbing(Component::Hz, i, j, k),
                    grid.is_absorbing(Component::Hz, i, j, k + 1),
                ];
                if faces.iter().all(|&f| !f) {
                    lossless_cells += 1;
                }
            }
        }
    }
    println!(
        "{TAG} absorber: loss-free core = {lossless_cells} cells [expect prod(n - 2*depth) = {}]",
        (n - 2 * depth[0]) * (n - 2 * depth[1]) * (n - 2 * depth[2])
    );

    // Coefficients at the outer face, where sigma = sigma_max: a = sigma*S/2 = 4*9/32 = 9/8.
    let (ca, cb) = loss_coefficients(sigma_max, COURANT_3D);
    println!(
        "{TAG} absorber: loss_coefficients(4, 9/16) = ({ca}, {cb}) [expect ca = (1-9/8)/(1+9/8) = -1/17 = {}, cb = (9/16)/(17/8) = 9/34 = {}]",
        q(-1, 17),
        q(9, 34)
    );
    let (ca0, cb0) = loss_coefficients(Fix128::ZERO, COURANT_3D);
    println!("{TAG} absorber: loss_coefficients(0, S) = ({ca0}, {cb0}) [expect (1, S)]");

    // Continuum reflection: R = exp(-2 * sigma_max * sum over cell centres of t^3).
    for (d, sm, num, den) in [(4usize, 4i32, -31i32, 4i32), (3, 2, -17, 6), (1, 4, -1, 1)] {
        // depth 4: t = 1/8, 3/8, 5/8, 7/8, sum t^3 = 31/32, exponent -2*4*31/32 = -31/4
        // depth 3: t = 1/6, 3/6, 5/6,      sum t^3 = 17/24, exponent -2*2*17/24 = -17/6
        // depth 1: t = 1/2,                sum t^3 = 1/8,   exponent -2*4/8 = -1
        let r = theoretical_pml_reflection(d, int(i64::from(sm)));
        let exponent = f64::from(num) / f64::from(den);
        println!(
            "{TAG} absorber: theoretical_pml_reflection({d}, {sm}) = {:.9} [closed form exp({num}/{den}) via det_math::exp64 = {:.9}, via Fix128::exp = {:.9}, gap {} ULP]",
            r.to_f64(),
            exp64(exponent),
            q(i64::from(num), i64::from(den)).exp().to_f64(),
            raw(r - q(i64::from(num), i64::from(den)).exp())
        );
    }
    println!(
        "{TAG} absorber: theoretical_pml_reflection(0, 4) = {} [expect 1: no layer reflects everything]",
        theoretical_pml_reflection(0, int(4))
    );

    // What is left in the box: PEC versus PML versus a uniform lossy medium.
    let seed = |g: &mut YeeGrid| {
        let (i, j, k) = (mid - 1, mid - 1, mid);
        g.set(Component::Ex, i, j, k, Fix128::ONE);
        g.set(Component::Ey, i + 1, j, k, Fix128::ONE);
        g.set(Component::Ex, i, j + 1, k, -Fix128::ONE);
        g.set(Component::Ey, i, j, k, -Fix128::ONE);
    };
    let mut pec = YeeGrid::new_with_absorber(n, n, n, COURANT_3D, Absorber::None);
    let mut lined = YeeGrid::new_with_absorber(
        n,
        n,
        n,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 3, 3],
            sigma_max,
        },
    );
    let mut lossy =
        YeeGrid::new_with_absorber(n, n, n, COURANT_3D, Absorber::Uniform { sigma: int(2) });
    for g in [&mut pec, &mut lined, &mut lossy] {
        seed(g);
    }
    println!(
        "{TAG} absorber: divergence-free seed: max |div E - rho| at step 0 = {} [expect 0, or the seed has a component no absorber can remove]",
        lined.max_abs_gauss_residual()
    );
    for step in 1..=150 {
        for g in [&mut pec, &mut lined, &mut lossy] {
            g.step();
        }
        if step == 25 || step % 50 == 0 {
            println!(
                "{TAG} absorber: step {step:3}: max |field| PEC = {}, PML = {}, uniform sigma=2 = {}",
                pec.max_abs_field(),
                lined.max_abs_field(),
                lossy.max_abs_field()
            );
        }
    }
}

fn main() {
    courant_bookkeeping();
    field_probe();
    gauss_law_with_a_placed_charge();
    a_current_deposits_charge();
    div_b_is_structural();
    absorber();
}
