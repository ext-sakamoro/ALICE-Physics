//! Independent oracles for `nbody` (direct-sum mutual gravity and the
//! velocity-Verlet / leapfrog integrator), standalone and through
//! `PhysicsWorld`.
//!
//! Expected values come from closed forms evaluated here in `f64`:
//! Newton's inverse-square law and the Plummer-softened force
//! `G·m·d/(d² + ε²)^{3/2}`, the circular two-body orbit `ω = √(G(m₁+m₂)/d³)`,
//! the leapfrog's discrete rotation angle `θ = 2·asin(ω·dt/2)` (derivation in
//! `circular_two_body_bound`), and the Chenciner–Montgomery figure-eight
//! three-body orbit with Simó's published initial conditions and period. No
//! expected value is produced by calling the code under test; where two
//! runs of the code are compared (order of accuracy, world vs standalone,
//! determinism) the comparison is a property, not a value.

#![allow(clippy::disallowed_methods)]

use alice_physics::nbody::{kinetic_energy, total_momentum, DirectSum, NBodyError, VelocityVerlet};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use core::f64::consts::PI;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// Sign-magnitude conversion: `Fix128::to_f64` has an absolute (not
/// relative) error of `2⁻⁵³` on negative values.
fn f64_of(x: Fix128) -> f64 {
    if x.is_negative() {
        -(-x).to_f64()
    } else {
        x.to_f64()
    }
}

fn v3(v: Vec3Fix) -> [f64; 3] {
    [f64_of(v.x), f64_of(v.y), f64_of(v.z)]
}

fn vx(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn law(g: f64, eps: f64) -> DirectSum {
    DirectSum::new(fx(g), fx(eps)).unwrap()
}

// ---------------------------------------------------------------------------
// The force law
// ---------------------------------------------------------------------------

/// oracle: Newton's law of gravitation, `a₁ = G·m₂/d²` toward body 2 and
/// `a₂ = G·m₁/d²` toward body 1, for `G = 6.674·10⁻¹¹`-free test values
/// `G = 2.5, m = (3, 0.5), d = 4` (along a skew axis). Relative `1e-15`.
#[test]
fn two_body_acceleration_is_inverse_square() {
    let (g, m1, m2, d) = (2.5, 3.0, 0.5, 4.0);
    let axis = [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0];
    let p1 = vx(1.0, 1.0, 1.0);
    let p2 = vx(1.0 + d * axis[0], 1.0 + d * axis[1], 1.0 + d * axis[2]);
    let mut acc = Vec::new();
    law(g, 0.0)
        .accelerations(&[p1, p2], &[fx(m1), fx(m2)], &mut acc)
        .unwrap();
    let (a1, a2) = (v3(acc[0]), v3(acc[1]));
    for k in 0..3 {
        let want1 = g * m2 / (d * d) * axis[k];
        let want2 = -g * m1 / (d * d) * axis[k];
        assert!(
            (a1[k] - want1).abs() < 1e-15,
            "a1[{k}] = {} vs {want1}",
            a1[k]
        );
        assert!(
            (a2[k] - want2).abs() < 1e-15,
            "a2[{k}] = {} vs {want2}",
            a2[k]
        );
    }
}

/// oracle: Plummer softening, `a = G·m·d/(d² + ε²)^{3/2}`: `G = 1, m = 2,
/// d = 3, ε = 4` gives `2·3/125 = 0.048` exactly; and the softened potential
/// `−G·m₁·m₂/√(d² + ε²) = −1·1·2/5 = −0.4`.
#[test]
fn plummer_softened_force_and_potential() {
    let l = law(1.0, 4.0);
    let pos = [Vec3Fix::ZERO, vx(3.0, 0.0, 0.0)];
    let mass = [fx(1.0), fx(2.0)];
    let mut acc = Vec::new();
    l.accelerations(&pos, &mass, &mut acc).unwrap();
    assert!(
        (f64_of(acc[0].x) - 0.048).abs() < 1e-17,
        "a = {}",
        f64_of(acc[0].x)
    );
    assert!((f64_of(acc[1].x) + 0.024).abs() < 1e-17);
    let pe = f64_of(l.potential_energy(&pos, &mass).unwrap());
    assert!((pe + 0.4).abs() < 1e-17, "U = {pe}");
}

/// oracle: superposition — the acceleration of a test body at the centre of
/// a square of four equal masses is zero, and off-centre it is the sum of
/// the four inverse-square terms evaluated in `f64`.
#[test]
fn accelerations_superpose() {
    let corners = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
    for probe in [(0.0, 0.0), (0.3, -0.2)] {
        let mut pos: Vec<Vec3Fix> = corners.iter().map(|&(x, y)| vx(x, y, 0.0)).collect();
        pos.push(vx(probe.0, probe.1, 0.0));
        let mut mass = vec![fx(1.5); 4];
        mass.push(Fix128::ZERO);
        let mut acc = Vec::new();
        law(1.0, 0.0).accelerations(&pos, &mass, &mut acc).unwrap();
        let mut want = [0.0, 0.0];
        for &(x, y) in &corners {
            let (dx, dy) = (x - probe.0, y - probe.1);
            let r3 = (dx * dx + dy * dy).powf(1.5);
            want[0] += 1.5 * dx / r3;
            want[1] += 1.5 * dy / r3;
        }
        let got = v3(acc[4]);
        assert!(
            (got[0] - want[0]).abs() < 1e-15 && (got[1] - want[1]).abs() < 1e-15,
            "{got:?} vs {want:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Integration: two-body circular orbit (closed form) and order of accuracy
// ---------------------------------------------------------------------------

/// Two equal masses `m = 1/2` at `±(1/2, 0, 0)`, `G = 1`: separation `d = 1`,
/// `μ = G(m₁+m₂) = 1`, `ω = 1`, circular speed of each body `ω·d/2 = 1/2`,
/// period `2π`.
fn circular_pair() -> (Vec<Vec3Fix>, Vec<Vec3Fix>, Vec<Fix128>) {
    let half = Fix128::from_ratio(1, 2);
    (
        vec![
            Vec3Fix::new(half, Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(-half, Fix128::ZERO, Fix128::ZERO),
        ],
        vec![
            Vec3Fix::new(Fix128::ZERO, half, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, -half, Fix128::ZERO),
        ],
        vec![half, half],
    )
}

/// Closed form of the leapfrog on the circular two-body orbit with
/// `h = ω·dt` (`d = 1`, `ω = 1`).
///
/// The recursion `x_{n+1} − 2x_n + x_{n−1} = dt²·a(x_n)` has an exact
/// circular solution rotating by `θ = 2·asin(h/2)` per step
/// (`2(1 − cos θ) = h²`): frequency high by `h²/24`, tangential speed
/// `d·sin θ/dt = d·ω·(1 − h²/8)`. Started with the exact speed `ω·d`, the
/// orbit is `h²/8` too fast for the discrete circle, so it is an ellipse with
/// periapsis at the start and `e = 2·δv/v = h²/4`:
///
/// - the separation stays in `[d, d(1+2e)] = [1, 1 + h²/2]`;
/// - its semi-major axis is longer by `δa/a = 2·δv/v = h²/4`, so its mean
///   motion is slower by `(3/2)·δa/a = 3h²/8`; with the `+h²/24` of the
///   discrete rotation the net phase lag is `h²/3`, i.e. `2π·h²/3` after one
///   period, and the separation misses its start by `d·2π·h²/3`.
///
/// Both are leading-order terms (`O(h⁴)` corrections), so they are asserted
/// to 2 %.
struct LeapfrogCircle {
    /// `|sep(2π) − sep(0)|`
    closure: f64,
    /// `max |sep| − 1`
    apoapsis_excess: f64,
    /// `1 − min |sep|`
    periapsis_deficit: f64,
}

fn leapfrog_circle_closed_form(h: f64) -> (f64, f64) {
    (2.0 * PI * h * h / 3.0, h * h / 2.0)
}

fn sep_norm(a: Vec3Fix, b: Vec3Fix) -> f64 {
    dist(v3(a - b), [0.0; 3])
}

fn run_pair_one_period(steps: i64) -> LeapfrogCircle {
    let (mut pos, mut vel, mass) = circular_pair();
    let dt = Fix128::TWO_PI / Fix128::from_int(steps);
    let l = law(1.0, 0.0);
    let mut vv = VelocityVerlet::new();
    let (mut rmax, mut rmin) = (1.0_f64, 1.0_f64);
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
        let r = sep_norm(pos[0], pos[1]);
        rmax = rmax.max(r);
        rmin = rmin.min(r);
    }
    LeapfrogCircle {
        closure: dist(v3(pos[0] - pos[1]), [1.0, 0.0, 0.0]),
        apoapsis_excess: rmax - 1.0,
        periapsis_deficit: 1.0 - rmin,
    }
}

fn assert_leapfrog_circle(got: &LeapfrogCircle, h: f64, ctx: &str) {
    let (closure, excess) = leapfrog_circle_closed_form(h);
    assert!(
        (got.closure / closure - 1.0).abs() < 0.02,
        "{ctx}: closure {:e} vs {closure:e}",
        got.closure
    );
    assert!(
        (got.apoapsis_excess / excess - 1.0).abs() < 0.02,
        "{ctx}: max |sep| − 1 = {:e} vs {excess:e}",
        got.apoapsis_excess
    );
    // periapsis is the start: the separation never drops below 1 (beyond rounding)
    assert!(
        got.periapsis_deficit < 1e-9,
        "{ctx}: 1 − min |sep| = {:e}",
        got.periapsis_deficit
    );
}

/// oracle: one period of the circular pair matches the leapfrog closed form
/// (closure `2π·h²/3`, separation in `[1, 1 + h²/2]`) for `h = 2π/200` and
/// `2π/1000`.
#[test]
fn circular_two_body_matches_leapfrog_closed_form() {
    for steps in [200_i64, 1000] {
        let h = 2.0 * PI / steps as f64;
        assert_leapfrog_circle(&run_pair_one_period(steps), h, &format!("N={steps}"));
    }
}

/// oracle: second order — halving `dt` divides the one-period closure error
/// by `≈ 4` (accepted `[3, 5]`).
#[test]
fn leapfrog_is_second_order() {
    let e1 = run_pair_one_period(400).closure;
    let e2 = run_pair_one_period(800).closure;
    let ratio = e1 / e2;
    assert!(
        (3.0..5.0).contains(&ratio),
        "error ratio {ratio} ({e1:e} / {e2:e})"
    );
}

/// oracle: a symplectic integrator keeps the energy error bounded — on an
/// `e = 0.5` orbit over 20 periods the largest relative energy error in the
/// last period is no larger than 1.2× the largest in the first (a secular
/// drift would grow it ~20×), and halving `dt` divides it by `[3, 5]`
/// (second order). Total momentum stays at its initial value (zero) within
/// `1e-15` (pairwise-antisymmetric forces; only fixed-point rounding).
#[test]
fn energy_error_is_bounded_and_momentum_is_conserved() {
    // Two bodies m = (0.8, 0.2), G = 1, μ = 1, a = 1, e = 0.5 started at
    // apoapsis r = a(1+e) = 1.5 with relative speed √(μ(2/r − 1/a)) = √(1/3).
    let run = |steps_per_period: i64| -> (f64, f64, f64) {
        let (m1, m2) = (0.8, 0.2);
        let r = 1.5;
        let v = (1.0_f64 / 3.0).sqrt();
        let mut pos = vec![vx(-m2 * r, 0.0, 0.0), vx(m1 * r, 0.0, 0.0)];
        let mut vel = vec![vx(0.0, -m2 * v, 0.0), vx(0.0, m1 * v, 0.0)];
        let mass = vec![fx(m1), fx(m2)];
        let l = law(1.0, 0.0);
        let energy = |pos: &[Vec3Fix], vel: &[Vec3Fix]| {
            f64_of(kinetic_energy(vel, &mass).unwrap())
                + f64_of(l.potential_energy(pos, &mass).unwrap())
        };
        let e0 = energy(&pos, &vel);
        let p0 = v3(total_momentum(&vel, &mass).unwrap());
        let dt = Fix128::TWO_PI / Fix128::from_int(steps_per_period);
        let mut vv = VelocityVerlet::new();
        let (mut first, mut last, mut pmax) = (0.0_f64, 0.0_f64, 0.0_f64);
        let periods = 20;
        for k in 0..periods * steps_per_period {
            vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
            let de = ((energy(&pos, &vel) - e0) / e0).abs();
            if k < steps_per_period {
                first = first.max(de);
            }
            if k >= (periods - 1) * steps_per_period {
                last = last.max(de);
            }
            pmax = pmax.max(dist(v3(total_momentum(&vel, &mass).unwrap()), p0));
        }
        (first, last, pmax)
    };
    let (first, last, pmax) = run(500);
    assert!(
        last <= 1.2 * first,
        "energy error grew: first period {first:e}, last {last:e}"
    );
    assert!(pmax < 1e-15, "momentum drift {pmax:e}");
    let (first2, _, _) = run(1000);
    let ratio = first / first2;
    assert!((3.0..5.0).contains(&ratio), "energy error ratio {ratio}");
}

// ---------------------------------------------------------------------------
// Three bodies: the figure-eight choreography
// ---------------------------------------------------------------------------

/// oracle: the Chenciner–Montgomery figure-eight (Ann. Math. 152, 2000) with
/// Simó's initial conditions (G = m = 1): `x₁ = −x₂ = (0.970 004 36,
/// −0.243 087 53)`, `x₃ = 0`, `v₃ = (−0.932 407 37, −0.864 731 46)`,
/// `v₁ = v₂ = −v₃/2`, period `T = 6.325 913 98`. After one period every body
/// is back at its start.
///
/// Tolerance `1e-6`: the published data carry 8 decimals, so the orbit they
/// describe is the choreography up to `≈ 5·10⁻⁹` in state and the period is
/// known to `5·10⁻⁹`; with speeds `≲ 1.4` and the orbit's mild (linearly
/// stable) sensitivity the return error from the data alone is `≲ 10⁻⁷`.
/// The leapfrog at `dt = T/20 000` adds `O(dt²) ≈ 10⁻⁷`. Measured return
/// error `2.4·10⁻⁷` (body 3); the bound keeps a factor 4 over it.
#[test]
fn figure_eight_three_body_orbit_is_periodic() {
    let (x1, y1) = (0.970_004_36, -0.243_087_53);
    let (vx3, vy3) = (-0.932_407_37, -0.864_731_46);
    let p0 = vec![vx(x1, y1, 0.0), vx(-x1, -y1, 0.0), Vec3Fix::ZERO];
    let mut pos = p0.clone();
    let mut vel = vec![
        vx(-vx3 / 2.0, -vy3 / 2.0, 0.0),
        vx(-vx3 / 2.0, -vy3 / 2.0, 0.0),
        vx(vx3, vy3, 0.0),
    ];
    let mass = vec![Fix128::ONE; 3];
    let steps = 20_000;
    let dt = fx(6.325_913_98) / Fix128::from_int(steps);
    let l = law(1.0, 0.0);
    let mut vv = VelocityVerlet::new();
    let mut max_excursion = 0.0_f64;
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
        max_excursion = max_excursion.max(dist(v3(pos[2]), [0.0; 3]));
    }
    for k in 0..3 {
        let d = dist(v3(pos[k]), v3(p0[k]));
        assert!(d < 1e-6, "body {k}: return error {d:e}");
    }
    // the orbit actually went somewhere (body 3 crosses to ±0.97)
    assert!(max_excursion > 0.9, "max excursion {max_excursion}");
}

// ---------------------------------------------------------------------------
// Through PhysicsWorld
// ---------------------------------------------------------------------------

fn orbit_world(substeps: usize) -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let (pos, vel, mass) = circular_pair();
    for k in 0..2 {
        let mut b = RigidBody::new_dynamic(pos[k], mass[k]);
        b.velocity = vel[k];
        world.add_body(b);
    }
    world
}

/// oracle (world entry): `DirectSum::step_world` — half kick, `world.step`,
/// half kick — on the circular pair matches the same leapfrog closed form as
/// the standalone integrator (closure `2π·h²/3`, separation in
/// `[1, 1 + h²/2]`) for 1 and 8 substeps, and agrees with the standalone
/// `VelocityVerlet` to `1e-11`. The world drifts in `substeps` pieces and
/// re-derives the velocity as `Δx/h_sub`, which rounds it by up to
/// `substeps·2⁻⁶⁴/dt ≈ 7·10⁻¹⁷` per frame; such a speed error shifts the mean
/// motion, so the phase difference grows like `(3/2)·N²·δv·dt ≈ 7·10⁻¹³` over
/// `N = 1000` frames (measured `2.3·10⁻¹²` with 8 substeps); the bound keeps
/// a factor 4.
#[test]
fn step_world_follows_the_leapfrog_closed_form() {
    let steps = 1000_i64;
    let dt = Fix128::TWO_PI / Fix128::from_int(steps);
    let h = 2.0 * PI / steps as f64;
    let (mut pos, mut vel, mass) = circular_pair();
    let l = law(1.0, 0.0);
    let mut vv = VelocityVerlet::new();
    for _ in 0..steps {
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    for substeps in [1, 8] {
        let mut world = orbit_world(substeps);
        let (mut rmax, mut rmin) = (1.0_f64, 1.0_f64);
        for _ in 0..steps {
            l.step_world(&mut world, dt);
            let r = sep_norm(world.bodies[0].position, world.bodies[1].position);
            rmax = rmax.max(r);
            rmin = rmin.min(r);
        }
        let got = LeapfrogCircle {
            closure: dist(
                v3(world.bodies[0].position - world.bodies[1].position),
                [1.0, 0.0, 0.0],
            ),
            apoapsis_excess: rmax - 1.0,
            periapsis_deficit: 1.0 - rmin,
        };
        assert_leapfrog_circle(&got, h, &format!("substeps={substeps}"));
        for (k, (body, p)) in world.bodies.iter().zip(&pos).enumerate() {
            let d = dist(v3(body.position), v3(*p));
            assert!(
                d < 1e-11,
                "substeps={substeps} body {k}: world vs standalone {d:e}"
            );
        }
    }
}

/// characterization (frame-head splitting): applying the whole kick before
/// `world.step` (`kick_bodies(dt)` then `step`, the way `force_fields` enter
/// the world) is symplectic Euler. Its positions are those of the leapfrog
/// started with the extra radial velocity `a₀·dt/2 = ω²d·dt/2`, which gives
/// the circular pair an eccentricity `e = h/2`: the separation swings
/// `±d·h/2` — **first order** in `dt` (asserted to 5 % at `h = 2π/400` and
/// `2π/800`), against `step_world`'s `h²/2`. This pins why `step_world`
/// splits the kick in two.
#[test]
fn frame_head_kick_is_first_order() {
    for steps in [400_i64, 800] {
        let h = 2.0 * PI / steps as f64;
        let dt = Fix128::TWO_PI / Fix128::from_int(steps);
        let l = law(1.0, 0.0);
        let mut world = orbit_world(8);
        let (mut rmax, mut rmin) = (1.0_f64, 1.0_f64);
        for _ in 0..steps {
            l.kick_bodies(&mut world.bodies, dt);
            world.step(dt);
            let r = sep_norm(world.bodies[0].position, world.bodies[1].position);
            rmax = rmax.max(r);
            rmin = rmin.min(r);
        }
        for (name, dev) in [("max |sep| − 1", rmax - 1.0), ("1 − min |sep|", 1.0 - rmin)] {
            assert!(
                (dev / (h / 2.0) - 1.0).abs() < 0.05,
                "N={steps}: {name} = {dev:e}, h/2 = {:e}",
                h / 2.0
            );
        }
    }
}

/// oracle (world entry, sleeping): two bodies released from rest attract
/// slowly — `a = G·m/d² = 2.5·10⁻⁵`, so for 10 s their speed stays below the
/// default sleep threshold `0.01` and the world would put them to sleep after
/// 60 frames. `step_world` wakes the bodies it kicks, so they keep falling:
/// the separation follows the standalone integrator (`1e-11`, see
/// `step_world_follows_the_leapfrog_closed_form`) and the closed form of the
/// start of a radial free fall (series of `d̈ = −μ/d²` about rest, `μ = G(m₁+m₂)`):
/// `d(t) = d₀ − μt²/(2d₀²) − μ²t⁴/(12d₀⁵) + O(μ³t⁶/d₀⁸)`, here
/// `2 − 2.5·10⁻³ − 1.04·10⁻⁶`; tolerance `1e-7` (the next term is
/// `≲ 3·10⁻⁹` and the leapfrog's `O(dt²)` error is below it).
#[test]
fn step_world_keeps_slow_bodies_awake() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let start = [vx(-1.0, 0.0, 0.0), vx(1.0, 0.0, 0.0)];
    for p in start {
        world.add_body(RigidBody::new_dynamic(p, fx(1e-4)));
    }
    let l = law(1.0, 0.0);
    let mut pos = start.to_vec();
    let mut vel = vec![Vec3Fix::ZERO; 2];
    let mass = vec![fx(1e-4); 2];
    let mut vv = VelocityVerlet::new();
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..600 {
        l.step_world(&mut world, dt);
        vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    }
    assert!(!world.is_sleeping(0) && !world.is_sleeping(1));
    let sep = sep_norm(world.bodies[0].position, world.bodies[1].position);
    let t = 10.0;
    let (mu, d0) = (2e-4_f64, 2.0_f64);
    let want = d0 - mu * t * t / (2.0 * d0 * d0) - mu * mu * t.powi(4) / (12.0 * d0.powi(5));
    assert!((sep - want).abs() < 1e-7, "separation {sep} vs {want}");
    assert!((sep - sep_norm(pos[0], pos[1])).abs() < 1e-11);
}

/// The acceleration cache of `VelocityVerlet` is reused only for the
/// positions it was computed at: moving a body between steps gives exactly
/// the result of a fresh integrator started from the moved state.
#[test]
fn velocity_verlet_recomputes_after_external_change() {
    let l = law(1.0, 0.0);
    let (mut pos, mut vel, mass) = circular_pair();
    let dt = fx(0.01);
    let mut vv = VelocityVerlet::new();
    vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    pos[0] = pos[0] + vx(0.1, 0.0, 0.0);
    let (mut pos2, mut vel2) = (pos.clone(), vel.clone());
    vv.step(&l, &mut pos, &mut vel, &mass, dt).unwrap();
    VelocityVerlet::new()
        .step(&l, &mut pos2, &mut vel2, &mass, dt)
        .unwrap();
    assert_eq!((pos, vel), (pos2, vel2));
}

/// Static and kinematic bodies neither feel nor exert mutual gravity
/// (`inv_mass = 0` has no finite mass); a lone dynamic body next to a static
/// one is not accelerated.
#[test]
fn kick_bodies_ignores_non_dynamic_bodies() {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_dynamic(vx(1.0, 0.0, 0.0), fx(1.0)),
    ];
    law(1.0, 0.0).kick_bodies(&mut bodies, fx(0.1));
    assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
    assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);
    // with a second dynamic body the pull appears: Δv = G·m/d²·dt = 0.1
    bodies.push(RigidBody::new_dynamic(vx(2.0, 0.0, 0.0), fx(1.0)));
    law(1.0, 0.0).kick_bodies(&mut bodies, fx(0.1));
    assert!((f64_of(bodies[1].velocity.x) - 0.1).abs() < 1e-17);
    assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);
}

/// Determinism: two identical runs give bit-identical states.
#[test]
fn integration_is_bit_reproducible() {
    let run = || {
        let mut world = orbit_world(8);
        let l = law(1.0, Fix128::from_ratio(1, 100).to_f64());
        for _ in 0..300 {
            l.step_world(&mut world, Fix128::from_ratio(1, 50));
        }
        (world.bodies[0].position, world.bodies[1].velocity)
    };
    assert_eq!(run(), run());
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

/// `G ≤ 0` → `Err(NonPositiveGravitationalConstant)`, `ε < 0` →
/// `Err(NegativeSoftening)`; `ε = 0` is allowed.
#[test]
fn degenerate_law_parameters_are_rejected() {
    assert_eq!(
        DirectSum::new(Fix128::ZERO, Fix128::ZERO).err(),
        Some(NBodyError::NonPositiveGravitationalConstant)
    );
    assert_eq!(
        DirectSum::new(fx(-1.0), Fix128::ZERO).err(),
        Some(NBodyError::NonPositiveGravitationalConstant)
    );
    assert_eq!(
        DirectSum::new(Fix128::ONE, fx(-0.1)).err(),
        Some(NBodyError::NegativeSoftening)
    );
    assert!(DirectSum::new(Fix128::ONE, Fix128::ZERO).is_ok());
}

/// `N = 0` → no accelerations, zero energies and momentum; `N = 1` → zero
/// acceleration and the body drifts at constant velocity.
#[test]
fn empty_and_single_body_systems() {
    let l = law(1.0, 0.0);
    let mut acc = vec![Vec3Fix::UNIT_X; 3];
    l.accelerations(&[], &[], &mut acc).unwrap();
    assert!(acc.is_empty());
    assert_eq!(l.potential_energy(&[], &[]).unwrap(), Fix128::ZERO);
    assert_eq!(total_momentum(&[], &[]).unwrap(), Vec3Fix::ZERO);
    assert_eq!(kinetic_energy(&[], &[]).unwrap(), Fix128::ZERO);
    let mut vv = VelocityVerlet::new();
    vv.step(&l, &mut [], &mut [], &[], Fix128::ONE).unwrap();

    let mut pos = [vx(1.0, 2.0, 3.0)];
    let mut vel = [vx(0.5, 0.0, -1.0)];
    l.accelerations(&pos, &[fx(5.0)], &mut acc).unwrap();
    assert_eq!(acc, vec![Vec3Fix::ZERO]);
    vv.step(&l, &mut pos, &mut vel, &[fx(5.0)], fx(2.0))
        .unwrap();
    assert_eq!(pos[0], vx(2.0, 2.0, 1.0));
    assert_eq!(vel[0], vx(0.5, 0.0, -1.0));
}

/// Two bodies at the same position: with `ε = 0` the pair has no direction
/// and is skipped (acceleration zero, potential term zero) rather than
/// returning an infinite or wrapped value; with `ε > 0` the softened force at
/// `d = 0` is zero by symmetry and the potential is `−G·m₁·m₂/ε`.
#[test]
fn coincident_bodies() {
    let pos = [vx(1.0, 1.0, 1.0), vx(1.0, 1.0, 1.0)];
    let mass = [fx(2.0), fx(3.0)];
    let mut acc = Vec::new();
    law(1.0, 0.0).accelerations(&pos, &mass, &mut acc).unwrap();
    assert_eq!(acc, vec![Vec3Fix::ZERO, Vec3Fix::ZERO]);
    assert_eq!(
        law(1.0, 0.0).potential_energy(&pos, &mass).unwrap(),
        Fix128::ZERO
    );
    law(1.0, 0.5).accelerations(&pos, &mass, &mut acc).unwrap();
    assert_eq!(acc, vec![Vec3Fix::ZERO, Vec3Fix::ZERO]);
    let pe = f64_of(law(1.0, 0.5).potential_energy(&pos, &mass).unwrap());
    assert!((pe + 12.0).abs() < 1e-15, "U = {pe}");
}

/// Slice lengths that disagree → `Err(LengthMismatch)` (nothing written);
/// a negative mass → `Err(NegativeMass)`.
#[test]
fn mismatched_lengths_and_negative_mass_are_rejected() {
    let l = law(1.0, 0.0);
    let mut acc = Vec::new();
    let two = [Vec3Fix::ZERO, Vec3Fix::UNIT_X];
    assert_eq!(
        l.accelerations(&two, &[Fix128::ONE], &mut acc),
        Err(NBodyError::LengthMismatch)
    );
    assert_eq!(
        l.potential_energy(&two, &[Fix128::ONE]),
        Err(NBodyError::LengthMismatch)
    );
    assert_eq!(
        kinetic_energy(&two, &[Fix128::ONE]),
        Err(NBodyError::LengthMismatch)
    );
    assert_eq!(
        total_momentum(&two, &[Fix128::ONE]),
        Err(NBodyError::LengthMismatch)
    );
    let mut pos = two;
    let mut vel = [Vec3Fix::ZERO];
    let mut vv = VelocityVerlet::new();
    assert_eq!(
        vv.step(
            &l,
            &mut pos,
            &mut vel,
            &[Fix128::ONE, Fix128::ONE],
            Fix128::ONE
        ),
        Err(NBodyError::LengthMismatch)
    );
    assert_eq!(pos, two, "positions untouched on error");
    assert_eq!(
        l.accelerations(&two, &[Fix128::ONE, fx(-1.0)], &mut acc),
        Err(NBodyError::NegativeMass)
    );
}
