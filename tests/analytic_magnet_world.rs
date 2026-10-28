//! Oracles for magnets in the world: `ForceField::InducedDipole` acting on the bodies of one
//! material through `PhysicsWorld::step`.
//!
//! oracle (the law, through the world's entry point): a magnet field pulls the bodies whose
//! material it names with `electromagnetic::induced_dipole_force` of the summed field of every
//! magnet naming that material, the body's volume being its mass over the field's density; the
//! velocity changes by that force times the inverse mass and the step. Bodies of other materials
//! feel nothing: their trail is the trail of a world without the magnet, to the bit.
//!
//! oracle (statics): below a magnet on its axis, gravity and the magnetic pull balance at the
//! distance `r*` where `F(r*) = m·g`; the balance is unstable (Earnshaw): a body placed slightly
//! farther falls away, one placed slightly nearer is drawn in.
//!
//! oracle (energy): the force is conservative, so in a flyby without gravity, damping or contact
//! the energy `½mv² + U` drifts only by the integrator's error, which halves when the step halves.
//!
//! oracle (state): a magnet and a magnetised body are part of the world's snapshot; restoring at
//! mid-run continues the same trail to the bit, and two runs are identical.

use alice_physics_core::electromagnetic::{
    dipole_sample, induced_dipole_energy, induced_dipole_force, MagneticResponse,
};
use alice_physics_core::{
    Fix128, ForceField, ForceFieldInstance, MaterialId, PhysicsConfig, PhysicsWorld, RigidBody,
    Vec3Fix,
};

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn steel_response() -> MagneticResponse {
    MagneticResponse {
        chi_eff: Fix128::from_int(3),
        saturation_magnetization: Fix128::from_int(1_700_000),
    }
}

/// Steel, 7850 kg/m³
fn steel_density() -> Fix128 {
    Fix128::from_int(7850)
}

/// A 1 cm steel ball: 4/3·π·(0.005 m)³·7850 kg/m³ ≈ 4.11e-3 kg
fn ball_mass() -> Fix128 {
    fx(4_110, 1_000_000)
}

fn magnet(position: Vec3Fix, moment: Vec3Fix, material: MaterialId) -> ForceFieldInstance {
    ForceFieldInstance::new(ForceField::InducedDipole {
        position,
        moment,
        radius: fx(1, 100),
        material,
        response: steel_response(),
        density: steel_density(),
    })
}

fn config(gravity: bool) -> PhysicsConfig {
    let mut c = PhysicsConfig::default();
    if !gravity {
        c.gravity = Vec3Fix::ZERO;
    }
    c.damping = Fix128::ONE;
    c
}

/// A world with one dynamic ball at `at` with velocity `vel`, made of a registered metal
/// material; returns (world, ball, metal material)
fn world_with_ball(gravity: bool, at: Vec3Fix, vel: Vec3Fix) -> (PhysicsWorld, usize, MaterialId) {
    let mut w = PhysicsWorld::new(config(gravity));
    let metal = w.material_table.register_metal();
    let mut body = RigidBody::new_dynamic(at, ball_mass());
    body.velocity = vel;
    let b = w.add_body(body);
    w.set_body_material(b, metal);
    (w, b, metal)
}

fn dt() -> Fix128 {
    fx(1, 240)
}

#[test]
fn one_step_changes_the_velocity_by_the_summed_field_law() {
    let p = Vec3Fix::new(fx(1, 100), fx(-6, 100), fx(2, 100));
    let (mut w, b, metal) = world_with_ball(false, p, Vec3Fix::ZERO);
    let ma = (
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::ZERO, fx(3, 10), Fix128::ZERO),
    );
    let mb = (
        Vec3Fix::new(fx(5, 100), fx(-10, 100), Fix128::ZERO),
        Vec3Fix::new(fx(1, 10), Fix128::ZERO, fx(1, 10)),
    );
    w.try_add_force_field(magnet(ma.0, ma.1, metal)).unwrap();
    w.try_add_force_field(magnet(mb.0, mb.1, metal)).unwrap();
    let inv_mass = w.bodies[b].inv_mass;
    let volume = Fix128::ONE / (inv_mass * steel_density());
    let s = dipole_sample(ma.0, ma.1, p) + dipole_sample(mb.0, mb.1, p);
    let f = induced_dipole_force(&s, &steel_response(), volume);
    assert_ne!(f, Vec3Fix::ZERO);
    w.step(dt());
    // the force enters before the step's integration as `v + F·inv_mass·dt`: the same step of a
    // world without the magnets, whose ball starts with exactly that velocity, ends in the same
    // bits (a force summed per magnet instead of per field would not)
    let (mut kicked, kb, _) = world_with_ball(false, p, Vec3Fix::ZERO + f * inv_mass * dt());
    kicked.step(dt());
    assert_eq!(
        (w.bodies[b].position, w.bodies[b].velocity),
        (kicked.bodies[kb].position, kicked.bodies[kb].velocity)
    );
    let apart = induced_dipole_force(&dipole_sample(ma.0, ma.1, p), &steel_response(), volume)
        + induced_dipole_force(&dipole_sample(mb.0, mb.1, p), &steel_response(), volume);
    assert_ne!(
        apart, f,
        "the per-magnet sum differs, so the comparison above has teeth"
    );
}

#[test]
fn each_body_is_clamped_at_its_own_contact_distance() {
    // a magnet of radius 1 cm; a ball of radius 5 mm 1 cm below it and a ball of radius 15 mm
    // 2 cm above it (3 cm apart, so they do not touch): both are inside their contact distance
    // (1.5 cm and 2.5 cm), so each is pulled with the law at its own contact distance
    let magnet_pos = Vec3Fix::ZERO;
    let moment = Vec3Fix::new(Fix128::ZERO, fx(1, 10), Fix128::ZERO);
    let mut w = PhysicsWorld::new(config(false));
    let metal = w.material_table.register_metal();
    let at_small = Vec3Fix::new(Fix128::ZERO, -fx(1, 100), Fix128::ZERO);
    let at_large = Vec3Fix::new(Fix128::ZERO, fx(2, 100), Fix128::ZERO);
    let (r_small, r_large) = (fx(5, 1000), fx(15, 1000));
    let small = w.add_body_with_radius(RigidBody::new_dynamic(at_small, ball_mass()), r_small);
    let large = w.add_body_with_radius(RigidBody::new_dynamic(at_large, ball_mass()), r_large);
    w.set_body_material(small, metal);
    w.set_body_material(large, metal);
    w.try_add_force_field(magnet(magnet_pos, moment, metal))
        .unwrap();
    let inv_mass = w.bodies[small].inv_mass;
    let volume = Fix128::ONE / (inv_mass * steel_density());
    let law_at = |p: Vec3Fix| {
        induced_dipole_force(
            &dipole_sample(magnet_pos, moment, p),
            &steel_response(),
            volume,
        )
    };
    // the clamp's point: the contact distance along the body's direction from the magnet
    let clamped = |at: Vec3Fix, radius: Fix128| {
        let (dir, _) = (at - magnet_pos).normalize_with_length();
        magnet_pos + dir * (fx(1, 100) + radius)
    };
    let f_small = law_at(clamped(at_small, r_small));
    let f_large = law_at(clamped(at_large, r_large));
    assert_ne!(
        f_small,
        law_at(at_small),
        "the clamp changes the small ball's force"
    );
    assert_ne!(
        f_large,
        law_at(at_large),
        "the clamp changes the large ball's force"
    );
    w.step(dt());
    for (b, at, radius, f) in [
        (small, at_small, r_small, f_small),
        (large, at_large, r_large, f_large),
    ] {
        // the same ball alone, without the magnet, starting with exactly the kick of the law at
        // its contact distance
        let mut kicked = PhysicsWorld::new(config(false));
        let mut body = RigidBody::new_dynamic(at, ball_mass());
        body.velocity = Vec3Fix::ZERO + f * inv_mass * dt();
        let k = kicked.add_body_with_radius(body, radius);
        kicked.step(dt());
        assert_eq!(
            (w.bodies[b].position, w.bodies[b].velocity),
            (kicked.bodies[k].position, kicked.bodies[k].velocity),
            "body {b}"
        );
    }
}

#[test]
fn a_body_of_another_material_feels_nothing() {
    let at = Vec3Fix::new(Fix128::ZERO, fx(-5, 100), Fix128::ZERO);
    let run = |with_magnet: bool| -> Vec<Vec3Fix> {
        let mut w = PhysicsWorld::new(config(true));
        let metal = w.material_table.register_metal();
        let rubber = w.material_table.register_rubber();
        let b = w.add_body(RigidBody::new_dynamic(at, ball_mass()));
        w.set_body_material(b, rubber);
        if with_magnet {
            w.try_add_force_field(magnet(
                Vec3Fix::ZERO,
                Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
                metal,
            ))
            .unwrap();
        }
        (0..120)
            .map(|_| {
                w.step(dt());
                w.bodies[b].position
            })
            .collect()
    };
    assert_eq!(run(true), run(false));
}

/// The distance below a magnet of moment `m` (along +y, at the origin) where the pull on a steel
/// ball balances gravity, by bisection on the law
fn balance_distance(m: Fix128) -> Fix128 {
    let weight = ball_mass() * Fix128::from_int(10);
    let volume = ball_mass() / steel_density();
    let pull = |r: Fix128| {
        let s = dipole_sample(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, m, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, -r, Fix128::ZERO),
        );
        induced_dipole_force(&s, &steel_response(), volume).y
    };
    let (mut lo, mut hi) = (fx(1, 100), Fix128::ONE);
    assert!(
        pull(lo) > weight && pull(hi) < weight,
        "the balance is between 1 cm and 1 m"
    );
    for _ in 0..80 {
        let mid = (lo + hi) / Fix128::from_int(2);
        if pull(mid) > weight {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[test]
fn below_the_magnet_gravity_balances_at_r_star_and_the_balance_is_unstable() {
    let m = fx(1, 2);
    let r_star = balance_distance(m);
    // (displacement after each of 60 steps, + up toward the magnet; and the velocity after the first)
    let trail = |r: Fix128| -> (Vec<Fix128>, Fix128) {
        let at = Vec3Fix::new(Fix128::ZERO, -r, Fix128::ZERO);
        let (mut w, b, metal) = world_with_ball(true, at, Vec3Fix::ZERO);
        w.try_add_force_field(magnet(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, m, Fix128::ZERO),
            metal,
        ))
        .unwrap();
        let mut moved = Vec::new();
        let mut v1 = Fix128::ZERO;
        for k in 0..60 {
            w.step(dt());
            if k == 0 {
                v1 = w.bodies[b].velocity.y;
            }
            moved.extend([w.bodies[b].position.y + r]);
        }
        (moved, v1)
    };
    let d = fx(1, 1000);
    // the balance is read in the first step's velocity: the magnet's kick at the start of the step
    // (F/m·dt) and the gravity of its substeps (g·dt) cancel. Not in its displacement: the step
    // applies the force field as a kick before its substeps and gravity within them, so a constant
    // field balancing gravity still moves the body by g·dt²·(1 − (n+1)/(2n)) per step (n substeps;
    // a known mismatch between the force-field kick and the substepped gravity, to be removed by
    // integrating force fields per substep, after which this oracle tightens to the position:
    // no displacement at r*), and the balance is unstable with rate √(7g/r*).
    let g_dt = Fix128::from_int(10) * dt();
    let (_, v_star) = trail(r_star);
    assert!(
        v_star.abs() < g_dt / Fix128::from_int(1000),
        "at r*: velocity {v_star:?} after one step, gravity alone gives {g_dt:?}"
    );
    let farther = trail(r_star + d).0[59];
    let nearer = trail(r_star - d).0[59];
    assert!(
        farther < -fx(1, 10_000),
        "1 mm farther: falls away, moved {farther:?}"
    );
    assert!(
        nearer > fx(1, 10_000),
        "1 mm nearer: drawn in, moved {nearer:?}"
    );
}

/// The energy drift of a flyby past the magnet without gravity or damping, with step `h`
fn flyby_drift(h: Fix128, steps: usize) -> f64 {
    let at = Vec3Fix::new(fx(-15, 100), fx(4, 100), Fix128::ZERO);
    let vel = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let (mut w, b, metal) = world_with_ball(false, at, vel);
    let m = (
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::ZERO, fx(1, 10), Fix128::ZERO),
    );
    w.try_add_force_field(magnet(m.0, m.1, metal)).unwrap();
    let volume = ball_mass() / steel_density();
    let energy = |w: &PhysicsWorld| -> f64 {
        let body = &w.bodies[b];
        let kinetic = body.velocity.length_squared() * ball_mass() / Fix128::from_int(2);
        let field = dipole_sample(m.0, m.1, body.position).field;
        (kinetic + induced_dipole_energy(field, &steel_response(), volume)).to_f64()
    };
    let e0 = energy(&w);
    let mut worst = 0.0f64;
    for _ in 0..steps {
        w.step(h);
        worst = worst.max((energy(&w) - e0).abs());
    }
    worst
}

#[test]
fn the_flyby_energy_drift_is_the_integrators_and_halves_with_the_step() {
    let coarse = flyby_drift(fx(1, 1000), 300);
    let fine = flyby_drift(fx(1, 2000), 600);
    assert!(
        coarse > 0.0,
        "the flyby must pass close enough to exchange energy"
    );
    // a first-order integrator: half the step, about half the drift (0.6 leaves room for the
    // higher-order terms near closest approach)
    assert!(fine < 0.6 * coarse, "drift {coarse} at h, {fine} at h/2");
}

fn trail_from(w: &mut PhysicsWorld, b: usize, steps: usize) -> Vec<(Vec3Fix, Vec3Fix)> {
    (0..steps)
        .map(|_| {
            w.step(dt());
            (w.bodies[b].position, w.bodies[b].velocity)
        })
        .collect()
}

#[test]
fn a_magnet_and_a_magnetised_body_restore_mid_run_to_the_same_trail() {
    let build = || {
        let at = Vec3Fix::new(fx(3, 100), fx(-9, 100), fx(1, 100));
        let (mut w, b, metal) = world_with_ball(true, at, Vec3Fix::ZERO);
        w.try_add_force_field(magnet(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, fx(4, 10), Fix128::ZERO),
            metal,
        ))
        .unwrap();
        (w, b)
    };
    let (mut a, b) = build();
    let first = trail_from(&mut a, b, 40);
    let blob = a.snapshot_world();
    let rest_a = trail_from(&mut a, b, 40);

    let (mut replay, _) = build();
    let mut whole = trail_from(&mut replay, b, 80);
    assert_eq!(
        whole.drain(..40).collect::<Vec<_>>(),
        first,
        "two runs agree"
    );
    assert_eq!(whole, rest_a);

    let mut restored = PhysicsWorld::new(PhysicsConfig::default());
    restored
        .restore_world(&blob)
        .expect("the snapshot restores");
    assert_eq!(
        restored.snapshot_world(),
        blob,
        "the restored world re-encodes the same"
    );
    assert_eq!(
        trail_from(&mut restored, b, 40),
        rest_a,
        "restored at mid-run, same trail"
    );
}

#[test]
fn two_magnets_naming_one_material_must_carry_the_same_response() {
    let (mut w, _, metal) = world_with_ball(true, Vec3Fix::ZERO, Vec3Fix::ZERO);
    w.try_add_force_field(magnet(Vec3Fix::ZERO, Vec3Fix::from_int(0, 1, 0), metal))
        .unwrap();
    let mut other = magnet(
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(0, 1, 0),
        metal,
    );
    if let ForceField::InducedDipole { response, .. } = &mut other.field {
        response.chi_eff = Fix128::from_int(2);
    }
    assert!(
        w.try_add_force_field(other.clone()).is_err(),
        "a second response for one material"
    );
    assert_eq!(w.force_fields.len(), 1, "the refused field is not added");

    // a snapshot holding two disagreeing fields (put in through the public field, not the API)
    // is refused on restore
    let good = w.snapshot_world();
    w.force_fields.extend([other]);
    let bad = w.snapshot_world();
    let mut target = PhysicsWorld::new(PhysicsConfig::default());
    assert!(
        target.restore_world(&bad).is_err(),
        "restore refuses the disagreeing fields"
    );
    assert!(target.restore_world(&good).is_ok());
}
