//! World-scale step cost with most bodies asleep.
//!
//! `N` spheres (radius 0.5, 2 m apart) lie on a grid; a fraction of them is put
//! to sleep at rest, the others drift upward at 1 m/s far above the grid, so the
//! awake set stays awake and never touches the sleeping one (weightless world,
//! no damping). One iteration is one `PhysicsWorld::step` (8 substeps).
//!
//! Each case runs with the sleep skip on (the default) and off
//! (`set_sleep_skip(false)`, every stage visits every body as before 2.x).
//! The per-stage work of one step is printed once per case from
//! `PhysicsWorld::stage_work`.
//!
//! Run: `cargo bench --bench world_scale`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use alice_physics::sleeping::SleepState;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world(n: usize, sleep_percent: usize, skip: bool) -> PhysicsWorld {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_sleep_skip(skip);
    let awake = n * (100 - sleep_percent) / 100;
    let side = (n as f64).sqrt().ceil() as i64;
    let radius = Fix128::from_ratio(1, 2);
    for i in 0..n {
        let x = (i as i64 % side) * 2;
        let z = (i as i64 / side) * 2;
        let y = if i < awake { 100 } else { 0 };
        let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE);
        if i < awake {
            body.velocity = Vec3Fix::from_int(0, 1, 0);
        }
        w.add_body_with_radius(body, radius);
    }
    for i in awake..n {
        w.islands.sleep_data[i].state = SleepState::Sleeping;
        w.islands.sleep_data[i].idle_frames = 100;
    }
    // The first step parks the sleeping bodies and fills the tree.
    w.step(dt());
    w.step(dt());
    w
}

fn bench_case(c: &mut Criterion, n: usize, sleep_percent: usize, skip: bool) {
    let label = format!(
        "world_scale/n{n}_sleep{sleep_percent}_{}",
        if skip { "skip_on" } else { "skip_off" }
    );
    let mut w = world(n, sleep_percent, skip);
    w.step(dt());
    let work = w.stage_work();
    eprintln!(
        "{label}: integrated {} broadphase_primitives {} broadphase_pairs {} \
         sleep_evaluated {} sleep_scanned {} parked {}",
        work.integrated,
        work.broadphase_primitives,
        work.broadphase_pairs,
        work.sleep_evaluated,
        work.sleep_scanned,
        work.parked
    );
    let mut group = c.benchmark_group("world_scale");
    group.sample_size(10);
    group.bench_function(&label, |b| b.iter(|| w.step(black_box(dt()))));
    group.finish();
    assert_eq!(w.sleep_skip(), skip);
}

fn bench_all(c: &mut Criterion) {
    for n in [10_000, 100_000] {
        for sleep_percent in [0, 90, 99] {
            for skip in [true, false] {
                bench_case(c, n, sleep_percent, skip);
            }
        }
    }
}

criterion_group!(world_scale, bench_all);
criterion_main!(world_scale);
