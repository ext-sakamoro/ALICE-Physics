//! Closed-form oracles of the walls of the 2D material point method
//! ([`alice_physics_core::mpm2d::Mpm2d::set_walls`]): a signed distance field sampled on the
//! grid's nodes, with Coulomb friction.
//!
//! 1. Parity: a flat frictionless wall along a row of nodes is the ground. The wall rule with
//!    normal `(0, 1)` and `μ = 0` removes the downward velocity of the nodes at or below the
//!    row and nothing else, so the particles are bit for bit those of the `ground` run.
//! 2. Rest: a block set on a flat wall with friction does not drift sideways. The load is
//!    symmetric about the block's middle only up to the rounding, so the bound is the rounding
//!    budget of oracle 2 of `analytic_mpm2d.rs` (per step, per particle, per stencil node, 16
//!    roundings of `2⁻⁶⁴`), as a displacement.
//! 3. Slope, below the friction angle (`tan θ < μ`): the block holds. The wall's nodes are
//!    stopped, so the block is a shear column fixed at its base under `ρ g sin θ`: the static
//!    displacement of its centre of mass along the slope is `ρ g sin θ H² / (3 G)` (the mean of
//!    `u(y) = (ρ g sin θ / G)(H y − y² / 2)`), a load applied at once overshoots to twice that,
//!    and the base holds up to one cell above the wall, so once settled it moves no more than
//!    `2 ρ g sin θ (H + Δx)² / (3 G)`. From the start it may also slip while the wall's nodes
//!    take up its weight: the base is found to the stencil's reach, `1.5 Δx` along the normal,
//!    `1.5 Δx tan θ` along the slope.
//! 4. Slope, above the friction angle: once settled the block accelerates at
//!    `a = g (sin θ − μ cos θ)` (Coulomb), for `μ = 0.5`, `0.25` and `0`. Measured by a second
//!    difference of the travel at three times `h` apart, after the first 0.1 s: a block let go
//!    at once is not rigid (its weight reaches the wall as a wave, and until then little
//!    friction acts; measured, the travel is ahead of the rigid one by a fixed speed of about
//!    0.03 m/s from 0.05 s on), so the travel itself is not compared. Each sample is off the
//!    rigid motion by at most the shear wobble of 3 and `μ` times the same along the normal
//!    (`2 ρ g cos θ (H + Δx)² / (3 M)`, `M = λ + 2 G`), and the second difference sums four.
//! 5. No passage: a block thrown at a wall never has a particle deeper in it than `1.5 Δx`: a
//!    particle that deep has all nine nodes of its stencil inside the wall, and those do not
//!    move inwards. The block leaves the wall again (no adhesion).
//! 6. Invalid walls are refused, and the walls are part of the state: digest, snapshot,
//!    replay.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{
    ImplicitSolve, Mpm2d, Mpm2dConfig, Mpm2dError, Mpm2dMaterial, Mpm2dModel, V2,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.01;
const E: f64 = 1.0e5;
const NU: f64 = 0.3;
const RHO: f64 = 1000.0;
const G: f64 = 9.81;
const MU: f64 = 0.5;
/// the block: 8 cells along the wall, 4 cells off it
const W: f64 = 0.08;
const H: f64 = 0.04;
const NODES: usize = 64;

fn shear_modulus() -> f64 {
    E / (2.0 * (1.0 + NU))
}

/// the P-wave modulus `λ + 2 G` in plane strain
fn p_modulus() -> f64 {
    E * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
}

fn dt() -> f64 {
    CELL / (E / ((1.0 - NU * NU) * RHO)).sqrt() / 3.0
}

fn world(gravity: [f64; 2], ground: Option<f64>) -> Mpm2d {
    Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes: [NODES, NODES],
            gravity: [fx(gravity[0]), fx(gravity[1])],
            dt: fx(dt()),
            ground: ground.map(fx),
        },
        Mpm2dMaterial {
            youngs: fx(E),
            poisson: fx(NU),
            density: fx(RHO),
            model: Mpm2dModel::FixedCorotated,
            maxwell: None,
        },
    )
    .expect("valid world")
}

/// 2 × 2 particles per cell over the rectangle with corner `corner`, `W` along the unit
/// vector `t` and `H` along `n`, moving at `v`
fn block(world: &mut Mpm2d, corner: [f64; 2], t: [f64; 2], n: [f64; 2], v: [f64; 2]) {
    let (na, nb) = ((W / CELL).round() as usize * 2, (H / CELL).round() as usize * 2);
    for i in 0..na {
        for j in 0..nb {
            let (s, h) = ((i as f64 + 0.5) * CELL / 2.0, (j as f64 + 0.5) * CELL / 2.0);
            world.add_particle(
                [fx(corner[0] + s * t[0] + h * n[0]), fx(corner[1] + s * t[1] + h * n[1])],
                [fx(v[0]), fx(v[1])],
                fx(CELL * CELL / 4.0),
            );
        }
    }
}

/// the half plane through `p0` with outward unit normal `n`: distance and normal at `x`
fn half_plane(p0: [f64; 2], n: [f64; 2]) -> impl Fn(V2) -> (Fix128, V2) {
    let (p, n) = ([fx(p0[0]), fx(p0[1])], [fx(n[0]), fx(n[1])]);
    move |x: V2| ((x[0] - p[0]) * n[0] + (x[1] - p[1]) * n[1], n)
}

fn centre_of_mass(w: &Mpm2d) -> [f64; 2] {
    let m = w.mass().to_f64();
    let mut c = [0.0; 2];
    for p in w.particles() {
        c[0] += p.mass.to_f64() * p.x[0].to_f64();
        c[1] += p.mass.to_f64() * p.x[1].to_f64();
    }
    [c[0] / m, c[1] / m]
}

#[test]
fn a_flat_frictionless_wall_is_the_ground() {
    let row = 0.1;
    let mut ground = world([0.0, -G], Some(row));
    let mut wall = world([0.0, -G], None);
    wall.set_walls(half_plane([0.0, row], [0.0, 1.0]), Fix128::ZERO)
        .expect("valid walls");
    for w in [&mut ground, &mut wall] {
        block(w, [0.2, row + 0.02], [1.0, 0.0], [0.0, 1.0], [0.3, 0.0]);
    }
    for step in 0..600 {
        ground.step().expect("ground step");
        wall.step().expect("wall step");
        assert_eq!(ground.particles(), wall.particles(), "step {step}");
    }
    // the block has landed and is sliding: the rule was exercised
    let lowest = wall
        .particles()
        .iter()
        .map(|p| p.x[1].to_f64())
        .fold(f64::INFINITY, f64::min);
    // (a particle stops within its stencil's reach of the row, 1.5 cells, and the block
    // rebounds from there; free fall would be `g t` by now)
    assert!(lowest < row + 2.0 * CELL, "the block reached the wall: {lowest}");
    let vy = wall.momentum()[1].to_f64() / wall.mass().to_f64();
    assert!(vy > -G * 600.0 * dt() / 2.0, "the wall stopped the fall: {vy}");
}

#[test]
fn a_block_rests_on_a_flat_wall_without_drift() {
    let row = 0.1;
    let mut w = world([0.0, -G], None);
    w.set_walls(half_plane([0.0, row], [0.0, 1.0]), fx(MU))
        .expect("valid walls");
    block(&mut w, [0.28, row], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]);
    let start = centre_of_mass(&w);
    let steps = 2000usize;
    for _ in 0..steps {
        w.step().expect("step");
    }
    let end = centre_of_mass(&w);
    // rounding budget as a displacement: steps² dt per step of velocity error, each step's
    // error 16 roundings per particle per stencil node, scaled by the speed scale g dt
    let per_step = 16.0 * 9.0 * w.particles().len() as f64 * 2f64.powi(-64);
    let budget = per_step * (steps as f64).powi(2) * dt() * (G * dt()).max(1.0);
    assert!(
        (end[0] - start[0]).abs() <= budget,
        "sideways drift {} over the rounding budget {budget}",
        end[0] - start[0]
    );
    // it stands on the wall: within the static compression below the start, doubled for the
    // load applied at once, and one cell for the base
    let sink = 2.0 * RHO * G * (H + CELL).powi(2) / (3.0 * p_modulus()) + CELL;
    assert!(start[1] - end[1] <= sink && end[1] <= start[1] + 1e-12, "height {end:?}");
}

/// the block on a slope of angle `theta`: the travel of the centre of mass down the slope and
/// along the normal, m, at each of `times` (ascending, s)
fn slope_run(theta: f64, mu: f64, times: &[f64]) -> Vec<(f64, f64)> {
    let (s, c) = theta.sin_cos();
    let (t, n) = ([c, s], [-s, c]);
    let p0 = [0.32, 0.2];
    let mut w = world([0.0, -G], None);
    w.set_walls(half_plane(p0, n), fx(mu)).expect("valid walls");
    // the block's lower edge on the wall, its middle 0.06 m up the slope from `p0`
    let corner = [p0[0] + (0.06 - W / 2.0) * t[0], p0[1] + (0.06 - W / 2.0) * t[1]];
    block(&mut w, corner, t, n, [0.0, 0.0]);
    let start = centre_of_mass(&w);
    let mut done = 0usize;
    times
        .iter()
        .map(|time| {
            let until = (time / dt()).round() as usize;
            for _ in done..until {
                w.step().expect("step");
            }
            done = until;
            let end = centre_of_mass(&w);
            let d = [end[0] - start[0], end[1] - start[1]];
            (-(d[0] * t[0] + d[1] * t[1]), d[0] * n[0] + d[1] * n[1])
        })
        .collect()
}

/// twice the static travel of the centre of mass of the block as a shear column under
/// `ρ g sin θ`, its base up to a cell off the wall
fn shear_wobble(theta: f64) -> f64 {
    2.0 * RHO * G * theta.sin() * (H + CELL).powi(2) / (3.0 * shear_modulus())
}

/// the same along the normal: the block as a bar under `ρ g cos θ`
fn normal_wobble(theta: f64) -> f64 {
    2.0 * RHO * G * theta.cos() * (H + CELL).powi(2) / (3.0 * p_modulus())
}

/// the acceleration down the slope from three samples `h` apart (a second difference), and
/// what the block's wobble can add to it: each sample is off the rigid travel by at most the
/// shear wobble and `mu` times the normal wobble, and the second difference sums four of them
fn settled_acceleration(theta: f64, mu: f64) -> (f64, f64) {
    let h = 0.05;
    let s = slope_run(theta, mu, &[0.1, 0.1 + h, 0.1 + 2.0 * h]);
    let a = (s[2].0 - 2.0 * s[1].0 + s[0].0) / (h * h);
    (a, 4.0 * (shear_wobble(theta) + mu * normal_wobble(theta)) / (h * h))
}

#[test]
fn a_block_below_the_friction_angle_holds() {
    let theta = 20f64.to_radians();
    assert!(theta.tan() < MU);
    let s = slope_run(theta, MU, &[0.2, 0.4]);
    // settled: it does not creep
    let creep = s[1].0 - s[0].0;
    assert!(
        creep.abs() <= shear_wobble(theta),
        "crept {creep} m in 0.2 s, the shear column allows {}",
        shear_wobble(theta)
    );
    // from the start: the shear column, and the base found to the stencil's reach (1.5 cells
    // along the normal, so `1.5 Δx tan θ` along the slope before the wall's nodes carry it)
    let whole = shear_wobble(theta) + 1.5 * CELL * theta.tan();
    assert!(s[1].0.abs() <= whole, "moved {} m, allowed {whole}", s[1].0);
}

#[test]
fn a_block_above_the_friction_angle_slides_by_coulomb() {
    let theta = 33f64.to_radians();
    assert!(theta.tan() > MU);
    let expected = G * (theta.sin() - MU * theta.cos());
    let (a, bound) = settled_acceleration(theta, MU);
    assert!(
        (a - expected).abs() <= bound,
        "accelerates at {a} m/s², Coulomb gives {expected} ± {bound}"
    );
    // friction matters in this scene: `g sin θ` is further than the bound
    assert!(G * theta.sin() - expected > 2.0 * bound, "the scene separates μ from no friction");
}

#[test]
fn friction_is_read_from_the_walls() {
    // the same slope with no friction, and with half of it
    let theta = 33f64.to_radians();
    for mu in [0.0, MU / 2.0] {
        let expected = G * (theta.sin() - mu * theta.cos());
        let (a, bound) = settled_acceleration(theta, mu);
        assert!(
            (a - expected).abs() <= bound,
            "μ {mu}: accelerates at {a} m/s², Coulomb gives {expected} ± {bound}"
        );
    }
}

#[test]
fn no_particle_passes_a_wall_and_the_block_leaves_it() {
    // a wall on the right at x = 0.4, the block thrown at it at 1 m/s (the bar wave is
    // 10.5 m/s), no gravity
    let (at, speed) = (0.4, 1.0);
    let mut w = world([0.0, 0.0], None);
    w.set_walls(half_plane([at, 0.0], [-1.0, 0.0]), fx(MU))
        .expect("valid walls");
    block(&mut w, [at - 0.05 - H, 0.28], [0.0, 1.0], [1.0, 0.0], [speed, 0.0]);
    let mut deepest = f64::INFINITY;
    let mut touched = false;
    for _ in 0..(0.3 / dt()).round() as usize {
        w.step().expect("step");
        for p in w.particles() {
            let d = at - p.x[0].to_f64();
            deepest = deepest.min(d);
            touched |= d < CELL;
        }
    }
    assert!(touched, "the block reached the wall");
    assert!(deepest >= -1.5 * CELL, "a particle went {deepest} m into the wall");
    let v = w.momentum()[0].to_f64() / w.mass().to_f64();
    assert!(v < 0.0, "the block left the wall: normal speed {v}");
    assert!(v.abs() <= speed, "it left no faster than it came: {v}");
}

#[test]
fn invalid_walls_are_refused_and_change_nothing() {
    let mut w = world([0.0, -G], None);
    block(&mut w, [0.28, 0.2], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]);
    let before = w.clone();
    let flat = half_plane([0.0, 0.1], [0.0, 1.0]);
    assert_eq!(
        w.set_walls(&flat, fx(-0.1)),
        Err(Mpm2dError::Invalid("walls: friction"))
    );
    assert_eq!(
        w.set_walls(|_x: V2| (Fix128::ZERO, [Fix128::ZERO, Fix128::ZERO]), fx(MU)),
        Err(Mpm2dError::Invalid("walls: normal"))
    );
    assert_eq!(w, before);
    // a normal of any length is taken as a direction
    let mut long = w.clone();
    long.set_walls(|x: V2| (x[1] - fx(0.1), [Fix128::ZERO, fx(3.0)]), fx(MU))
        .expect("valid walls");
    w.set_walls(&flat, fx(MU)).expect("valid walls");
    assert_eq!(w, long);
    // the backward Euler step does not know the walls: refused, nothing changed
    let held = w.clone();
    assert_eq!(
        w.step_implicit(&ImplicitSolve::default()).map(|_| ()),
        Err(Mpm2dError::Invalid("implicit: walls"))
    );
    assert_eq!(w, held);
    w.clear_walls();
    assert_eq!(w, before);
}

#[test]
fn the_walls_are_part_of_the_state() {
    let build = |mu: f64| {
        let mut w = world([0.0, -G], None);
        w.set_walls(half_plane([0.0, 0.1], [0.0, 1.0]), fx(mu))
            .expect("valid walls");
        block(&mut w, [0.28, 0.1], [1.0, 0.0], [0.0, 1.0], [0.4, 0.0]);
        w
    };
    let (mut a, mut b, other) = (build(MU), build(MU), build(0.25));
    assert_ne!(a.state_digest(), other.state_digest(), "μ is in the digest");
    let mut bare = a.clone();
    bare.clear_walls();
    assert_ne!(a.state_digest(), bare.state_digest(), "the walls are in the digest");
    for _ in 0..200 {
        a.step().expect("step");
        b.step().expect("step");
    }
    assert_eq!(a.state_digest(), b.state_digest());
    let mut back = Mpm2d::restore(&a.snapshot()).expect("restores");
    assert_eq!(back, a);
    for _ in 0..200 {
        a.step().expect("step");
        back.step().expect("step");
    }
    assert_eq!(a.state_digest(), back.state_digest());
    // a world without walls keeps the first layout: version 1, byte for byte
    assert_eq!(bare.snapshot()[4..6], 1u16.to_le_bytes());
    assert_eq!(a.snapshot()[4..6], 2u16.to_le_bytes());
}

/// Measurement (not an oracle): the slope runs sampled in time.
#[test]
#[ignore = "measurement: prints the travel down the slope and along the normal over time"]
fn measure_the_slope_runs_in_time() {
    let times = [0.05, 0.1, 0.15, 0.2];
    for (deg, mu) in [(20.0f64, MU), (33.0, MU), (33.0, MU / 2.0), (33.0, 0.0)] {
        let theta = deg.to_radians();
        let a = (G * (theta.sin() - mu * theta.cos())).max(0.0);
        for (time, (down, normal)) in times.iter().zip(slope_run(theta, mu, &times)) {
            println!(
                "deg {deg} μ {mu} t {time}: down {down:.6} (rigid {:.6}), normal {normal:.6}",
                a * time * time / 2.0
            );
        }
        if a > 0.0 {
            println!("  settled acceleration {:?}, Coulomb {a}", settled_acceleration(theta, mu));
        }
    }
}
