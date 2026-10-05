//! Scene snapshot through JSON: `CURRENT_SCENE_VERSION`, `save_scene_json`, `load_scene_json`,
//! `PhysicsConfig::new`.
//!
//! A `PhysicsWorld` is captured into a `PhysicsScene` (raw `hi` / `lo` limbs, so the file is
//! bit exact), written as JSON, read back, and the bodies are rebuilt: every position, velocity and
//! mass limb must match the original bit for bit. The scene carries a non-default solver
//! configuration built with `PhysicsConfig::new` from the world's own settings (substeps,
//! iterations, gravity `(0, -9.81, 0)`, damping 0.95); the documented contract is that the raw
//! values are stored as given, so each field must decode back to the value that went in.
//!
//! ```bash
//! cargo run --release --example scene_snapshot_roundtrip --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::scene_io::{
    load_scene_json, save_scene_json, PhysicsConfig, PhysicsScene, SerializedBody,
    CURRENT_SCENE_VERSION,
};
use alice_physics::solver::{PhysicsConfig as WorldConfig, PhysicsWorld, RigidBody};

fn limbs3(v: Vec3Fix) -> [i64; 6] {
    [
        v.x.hi,
        v.x.lo as i64,
        v.y.hi,
        v.y.lo as i64,
        v.z.hi,
        v.z.lo as i64,
    ]
}
fn limb(v: Fix128) -> [i64; 2] {
    [v.hi, v.lo as i64]
}
fn from_limbs3(r: &[i64; 6]) -> Vec3Fix {
    let f = |h: i64, l: i64| Fix128 {
        hi: h,
        lo: l as u64,
    };
    Vec3Fix::new(f(r[0], r[1]), f(r[2], r[3]), f(r[4], r[5]))
}

fn main() -> std::io::Result<()> {
    let mut world = PhysicsWorld::new(WorldConfig::default());
    world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(
            Fix128::from_ratio(1, 3),
            Fix128::from_int(5),
            Fix128::from_ratio(-2, 7),
        ),
        Fix128::from_ratio(5, 2),
    ));
    for _ in 0..30 {
        world.step(Fix128::from_ratio(1, 60));
    }

    let q = QuatFix::IDENTITY;
    let bodies: Vec<SerializedBody> = world
        .bodies
        .iter()
        .map(|b| SerializedBody {
            position: limbs3(b.position),
            velocity: limbs3(b.velocity),
            rotation: [
                q.x.hi,
                q.x.lo as i64,
                q.y.hi,
                q.y.lo as i64,
                q.z.hi,
                q.z.lo as i64,
                q.w.hi,
                q.w.lo as i64,
            ],
            mass: if b.inv_mass.is_zero() {
                [0, 0]
            } else {
                limb(Fix128::ONE / b.inv_mass)
            },
            body_type: u8::from(b.inv_mass.is_zero()),
        })
        .collect();
    let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-981, 100), Fix128::ZERO);
    let damping = Fix128::from_ratio(95, 100);
    let config = PhysicsConfig::new(
        world.config.substeps as u32,
        world.config.iterations as u32 + 1,
        limbs3(gravity),
        limb(damping),
    );
    assert_eq!(config.substeps, world.config.substeps as u32);
    assert_eq!(config.iterations, world.config.iterations as u32 + 1);
    assert_eq!(from_limbs3(&config.gravity), gravity);
    assert_eq!(config.damping, limb(damping));
    assert_ne!(config, PhysicsConfig::default(), "a custom configuration");
    let scene = PhysicsScene::new(bodies, Vec::new(), config, CURRENT_SCENE_VERSION);

    let path = std::env::temp_dir().join("alice_physics_scene_snapshot.json");
    save_scene_json(&scene, &path)?;
    let loaded = load_scene_json(&path)?;
    std::fs::remove_file(&path)?;

    assert_eq!(loaded, scene, "JSON round trip must be exact");
    assert_eq!(loaded.version, CURRENT_SCENE_VERSION);
    assert_eq!(loaded.config.substeps, world.config.substeps as u32);
    assert_eq!(loaded.config.iterations, world.config.iterations as u32 + 1);
    assert_eq!(from_limbs3(&loaded.config.gravity), gravity);
    assert_eq!(loaded.config.damping, limb(damping));
    for (b, s) in world.bodies.iter().zip(&loaded.bodies) {
        assert_eq!(from_limbs3(&s.position), b.position);
        assert_eq!(from_limbs3(&s.velocity), b.velocity);
    }
    println!(
        "scene v{}: {} bodies restored bit-exact; dynamic body y = {:.6}",
        loaded.version,
        loaded.bodies.len(),
        from_limbs3(&loaded.bodies[1].position).y.to_f64()
    );
    Ok(())
}
