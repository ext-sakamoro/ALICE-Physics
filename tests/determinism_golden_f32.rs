//! Cross-platform golden hashes for the `f32` / `f64` field modules.
//!
//! `determinism_golden.rs` pins the Fix128 core. This file pins every module
//! whose arithmetic runs in IEEE `f32` / `f64`: their outputs are only
//! bit-exact across platforms because (a) `+ - * / sqrt` are IEEE-exact on
//! every target Rust supports and (b) all transcendentals go through
//! `alice_physics::det_math` (enforced by `clippy.toml` `disallowed-methods`).
//!
//! Each scenario feeds fixed inputs through a module's public entry point,
//! serialises the outputs bit-for-bit and compares the SHA-256 against a
//! constant recorded on macOS aarch64. CI runs this on macOS x86, Linux
//! x86 / ARM, Windows and `wasm32-wasip1`; a mismatch on any of them means a
//! platform-dependent operation crept in.
//!
//! Updating a golden (only after an intentional algorithm change): run the
//! failing test, copy the "actual" hex into the constant, and note the change
//! in CHANGELOG.

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Byte sink
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Sink(Vec<u8>);

impl Sink {
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn fix(&mut self, v: Fix128) {
        self.0.extend_from_slice(&v.hi.to_le_bytes());
        self.0.extend_from_slice(&v.lo.to_le_bytes());
    }
    fn vec3(&mut self, v: Vec3Fix) {
        self.fix(v.x);
        self.fix(v.y);
        self.fix(v.z);
    }
    fn f32x3(&mut self, v: (f32, f32, f32)) {
        self.f32(v.0);
        self.f32(v.1);
        self.f32(v.2);
    }
    fn arr3(&mut self, v: [f32; 3]) {
        self.f32(v[0]);
        self.f32(v[1]);
        self.f32(v[2]);
    }
    fn finish(self) -> String {
        let hash = Sha256::digest(&self.0);
        hash.iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn assert_golden(scenario: &str, actual: &str, expected: &str) {
    assert_eq!(
        actual, expected,
        "\n\nGolden hash mismatch for f32 scenario `{scenario}`.\n\
         actual:   {actual}\n\
         expected: {expected}\n\n\
         If the module's algorithm changed on purpose, update the GOLDEN_*\n\
         constant in tests/determinism_golden_f32.rs and record the change in\n\
         CHANGELOG. If it did not, a platform-dependent float operation was\n\
         introduced (see clippy.toml disallowed-methods / det_math).\n"
    );
}

/// Unit sphere as an SDF field. `sqrt` only: IEEE-exact everywhere.
fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            if len > 0.0 {
                (x / len, y / len, z / len)
            } else {
                (0.0, 1.0, 0.0)
            }
        },
    )
}

/// Deterministic pseudo-random f32 in [0, 1) from an integer index.
fn prand(i: u32) -> f32 {
    let mut h = i.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h >> 8) as f32 / 16_777_216.0
}

// ---------------------------------------------------------------------------
// 1. sim_field — ScalarField3D / VectorField3D (exp decay)
// ---------------------------------------------------------------------------

const GOLDEN_SIM_FIELD: &str = "4b1bf9a7eb709e37caf7bc0aa4499435bea0a6be693b29e6257daa31e1c60085";

#[test]
fn golden_sim_field() {
    use alice_physics::sim_field::{ScalarField3D, VectorField3D};
    let mut f = ScalarField3D::new(8, 8, 8, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    for i in 0..12u32 {
        let p = (
            prand(i) * 2.0 - 1.0,
            prand(i + 100) * 2.0 - 1.0,
            prand(i + 200) * 2.0 - 1.0,
        );
        f.splat(p.0, p.1, p.2, 1.0 + prand(i + 300), 0.4);
    }
    for _ in 0..4 {
        f.diffuse(0.016, 0.3);
        f.decay(0.5, 0.016);
        f.decay_toward(0.1, 0.2, 0.016);
    }
    let mut s = Sink::default();
    for iz in 0..8 {
        for iy in 0..8 {
            for ix in 0..8 {
                s.f32(f.get(ix, iy, iz));
            }
        }
    }
    s.f32(f.sample(0.13, -0.42, 0.77));
    s.f32x3(f.gradient(0.13, -0.42, 0.77));
    s.f32(f.max_value());
    let mut v = VectorField3D::new(4, 4, 4, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    v.splat(0.2, 0.1, -0.3, 1.0, 2.0, -0.5, 0.6);
    s.f32x3(v.sample(0.25, 0.0, -0.25));
    assert_golden("sim_field", &s.finish(), GOLDEN_SIM_FIELD);
}

// ---------------------------------------------------------------------------
// 2. fracture — crack pattern generation (sin/cos hash directions)
// ---------------------------------------------------------------------------

const GOLDEN_FRACTURE: &str = "3a45ff9e27d7a88bce3a46a87a1b3d79beebb368c29ac2bd44c3e543c6281dd2";

#[test]
fn golden_fracture() {
    use alice_physics::fracture::{FractureConfig, FractureModifier};
    use alice_physics::sim_modifier::PhysicsModifier;
    let cfg = FractureConfig {
        fracture_toughness: 0.5,
        ..FractureConfig::default()
    };
    let mut m = FractureModifier::new(cfg, 12, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    for i in 0..10u32 {
        m.apply_stress_at(
            prand(i) - 0.5,
            prand(i + 7) - 0.5,
            prand(i + 13) - 0.5,
            2.0,
            0.5,
        );
        m.update(0.016);
    }
    let mut s = Sink::default();
    s.usize(m.active_crack_count());
    for c in &m.cracks {
        s.f32x3(c.start);
        s.f32x3(c.end);
        s.f32x3(c.direction);
        s.f32(c.length);
        s.bool(c.active);
    }
    for i in 0..20u32 {
        let (x, y, z) = (
            prand(i + 40) - 0.5,
            prand(i + 41) - 0.5,
            prand(i + 42) - 0.5,
        );
        s.f32(m.stress_at(x, y, z));
        s.f32(m.modify_distance(x, y, z, 0.1));
    }
    assert_golden("fracture", &s.finish(), GOLDEN_FRACTURE);
}

// ---------------------------------------------------------------------------
// 3. rolling_contact — Hertz (cbrt) + Basquin (powf)
// ---------------------------------------------------------------------------

const GOLDEN_ROLLING_CONTACT: &str =
    "10b437547fa48e649bc13f8b60935f986577f58463c054f20327109dd39c0470";

#[test]
fn golden_rolling_contact() {
    use alice_physics::rolling_contact::{
        basquin_cycles_to_failure, hertzian_sphere_sphere, rolling_contact_life_cycles,
    };
    let mut s = Sink::default();
    for i in 0..16u32 {
        let load = 100.0 + 5000.0 * prand(i);
        let c = hertzian_sphere_sphere(load, 0.01, 0.02, 2.1e11, 2.1e11, 0.3, 0.3);
        s.f32(c.contact_radius_m);
        s.f32(c.peak_pressure_pa);
        s.f32(c.max_shear_depth_m);
        s.f32(basquin_cycles_to_failure(
            1.0e8 + 1.0e9 * prand(i + 3),
            1.0e12,
            3.0,
            2.0e8,
        ));
        s.f32(rolling_contact_life_cycles(
            load, 0.01, 0.02, 2.1e11, 2.1e11, 0.3, 0.3, 1.0e12, 3.0, 2.0e8,
        ));
    }
    assert_golden("rolling_contact", &s.finish(), GOLDEN_ROLLING_CONTACT);
}

// ---------------------------------------------------------------------------
// 4. sdf_destruction — cylinder carve (hypot) + impact helpers
// ---------------------------------------------------------------------------

const GOLDEN_SDF_DESTRUCTION: &str =
    "8072eb0449c1a0f89e5c73d79c76319db9d25960135b0c55411d1ddbfa52f692";

#[test]
fn golden_sdf_destruction() {
    use alice_physics::sdf_destruction::{
        destruction_from_impact, destruction_from_projectile, DestructibleSdf, DestructionShape,
        DestructionType,
    };
    let mut d = DestructibleSdf::new(Box::new(unit_sphere()));
    d.apply_destruction(DestructionShape {
        center: Vec3Fix::from_f32(0.5, 0.2, 0.1),
        rotation: QuatFix::IDENTITY,
        shape: DestructionType::Cylinder {
            radius: 0.3,
            half_height: 0.4,
        },
        smooth_factor: 0.05,
    });
    d.apply_destruction(DestructionShape {
        center: Vec3Fix::from_f32(-0.4, -0.3, 0.2),
        rotation: QuatFix::IDENTITY,
        shape: DestructionType::Box {
            half_extents: (0.2, 0.25, 0.3),
        },
        smooth_factor: 0.0,
    });
    let contact = Contact {
        point_a: Vec3Fix::from_f32(0.9, 0.1, 0.0),
        point_b: Vec3Fix::from_f32(0.9, 0.1, 0.0),
        normal: Vec3Fix::UNIT_X,
        depth: Fix128::from_ratio(1, 20),
    };
    d.apply_destruction(destruction_from_impact(
        &contact,
        Fix128::from_ratio(7, 2),
        0.1,
        0.05,
        0.5,
    ));
    d.apply_destruction(destruction_from_projectile(
        Vec3Fix::from_f32(0.0, 0.8, 0.0),
        Vec3Fix::from_f32(0.0, -1.0, 0.0),
        0.15,
        0.6,
    ));
    let mut s = Sink::default();
    s.usize(d.destruction_count());
    for i in 0..64u32 {
        let (x, y, z) = (
            prand(i) * 2.4 - 1.2,
            prand(i + 1) * 2.4 - 1.2,
            prand(i + 2) * 2.4 - 1.2,
        );
        s.f32(d.distance(x, y, z));
        s.f32x3(d.normal(x, y, z));
    }
    assert_golden("sdf_destruction", &s.finish(), GOLDEN_SDF_DESTRUCTION);
}

// ---------------------------------------------------------------------------
// 5. privacy — Laplace (ln) + randomized response (exp), fixed seeds
// ---------------------------------------------------------------------------

const GOLDEN_PRIVACY: &str = "77fa76a6a8b314f7321712495ccc22b7f13bc559f67a143a029a9b9600b40be7";

#[test]
fn golden_privacy() {
    use alice_physics::privacy::{LaplaceNoise, RandomizedResponse, XorShift64};
    let mut s = Sink::default();
    let mut lap = LaplaceNoise::with_seed(1.0, 0.5, 0x1234_5678_9abc_def0);
    for i in 0..64 {
        s.f64(lap.sample());
        s.f64(lap.privatize(f64::from(i)));
        s.u64(lap.privatize_int(i64::from(i) * 10) as u64);
    }
    // `new(epsilon)` seeds from OS entropy (and traps on wasm); the seeded
    // constructor keeps the scenario reproducible. The exp64 in `new` is
    // covered by det_math's own bit pins.
    let mut rr = RandomizedResponse::with_probability(0.668_187_8, 0x0bad_5eed_cafe_f00d);
    for i in 0..64u8 {
        s.bool(rr.privatize(i % 3 == 0));
        s.0.push(rr.privatize_bit(i & 1));
    }
    let mut rng = XorShift64::new(42);
    for _ in 0..16 {
        s.f64(rng.next_f64());
        s.f64(rng.next_f64_range(-3.0, 7.0));
    }
    assert_golden("privacy", &s.finish(), GOLDEN_PRIVACY);
}

// ---------------------------------------------------------------------------
// 6. sketch — HyperLogLog (ln), DDSketch (ln/powf), CountMin (exp confidence)
// ---------------------------------------------------------------------------

const GOLDEN_SKETCH: &str = "2339dc43634fa99c1d25d6242d1590370f90b321ab35c0a30ec60732ffc7812d";

#[test]
fn golden_sketch() {
    use alice_physics::sketch::{CountMinSketch1024x5, DDSketch512, HyperLogLog12};
    let mut s = Sink::default();
    let mut hll = HyperLogLog12::new();
    for i in 0..5000u64 {
        hll.insert(&(i * 7919));
    }
    s.f64(hll.cardinality());
    let mut dd = DDSketch512::new(0.02);
    for i in 0..2000u32 {
        dd.insert(f64::from(0.001 + 9.999 * prand(i)));
    }
    for q in [0.01, 0.1, 0.5, 0.9, 0.99] {
        s.f64(dd.quantile(q));
    }
    s.f64(dd.mean());
    let mut cm = CountMinSketch1024x5::new();
    for i in 0..3000u64 {
        cm.insert(&(i % 97));
    }
    for k in [0u64, 13, 42, 96, 1000] {
        s.u64(cm.estimate(&k));
    }
    s.f64(cm.confidence());
    assert_golden("sketch", &s.finish(), GOLDEN_SKETCH);
}

// ---------------------------------------------------------------------------
// 7. sdf_sph — kernels (powi) + solver step against sphere boundary
// ---------------------------------------------------------------------------

const GOLDEN_SDF_SPH: &str = "2cb82127d65107e5d9e9df92590a303507064b680214d458da3b4760639960d4";

#[test]
fn golden_sdf_sph() {
    use alice_physics::sdf_sph::{
        poly6, spiky_grad, viscosity_lap, SphConfig, SphParticle, SphSolver,
    };
    let mut s = Sink::default();
    for i in 0..16u32 {
        let r = 0.05 * prand(i);
        s.f32(poly6(r, 0.05));
        s.f32(spiky_grad(r, 0.05));
        s.f32(viscosity_lap(r, 0.05));
    }
    let boundary = ClosureSdf::new(
        |x, y, z| 0.5 - (x * x + y * y + z * z).sqrt(), // inside-out sphere: fluid stays inside
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            if len > 0.0 {
                (-x / len, -y / len, -z / len)
            } else {
                (0.0, 1.0, 0.0)
            }
        },
    );
    let particles: Vec<SphParticle> = (0..64u32)
        .map(|i| {
            SphParticle::at_rest([
                0.3 * (prand(i) - 0.5),
                0.3 * (prand(i + 64) - 0.5),
                0.3 * (prand(i + 128) - 0.5),
            ])
        })
        .collect();
    let mut solver = SphSolver::new(particles, SphConfig::water_like(), &boundary);
    for _ in 0..5 {
        solver.step(0.004);
    }
    for p in &solver.particles {
        s.arr3(p.position);
        s.arr3(p.velocity);
    }
    assert_golden("sdf_sph", &s.finish(), GOLDEN_SDF_SPH);
}

// ---------------------------------------------------------------------------
// 8. thin_wall — thickness analysis over sampled surface points
// ---------------------------------------------------------------------------

const GOLDEN_THIN_WALL: &str = "d735baa3573b193582e0b05450a05dfcb39f508694060e4575bace521fb604fa";

#[test]
fn golden_thin_wall() {
    use alice_physics::thin_wall::{analyze_thickness, sample_surface_points, ThinWallConfig};
    // Thin shell: |r - 1| - 0.02 → 40 mm wall if units are mm... keep unitless
    let shell = ClosureSdf::new(
        |x, y, z| ((x * x + y * y + z * z).sqrt() - 1.0).abs() - 0.02,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            let sign = if len >= 1.0 { 1.0 } else { -1.0 };
            (sign * x / len, sign * y / len, sign * z / len)
        },
    );
    let pts = sample_surface_points(
        &shell,
        Vec3Fix::from_f32(-1.2, -1.2, -1.2),
        Vec3Fix::from_f32(1.2, 1.2, 1.2),
        Fix128::from_ratio(3, 10),
    );
    let report = analyze_thickness(&shell, &pts, &ThinWallConfig::default());
    let mut s = Sink::default();
    s.usize(pts.len());
    for p in pts.iter().take(32) {
        s.vec3(*p);
    }
    s.usize(report.sampled_count);
    s.usize(report.unbounded_count);
    s.fix(report.min_thickness_seen);
    s.fix(report.max_thickness_seen);
    s.usize(report.regions.len());
    for r in report.regions.iter().take(16) {
        s.vec3(r.position);
        s.f32x3(r.outward_normal);
        s.fix(r.thickness_mm);
    }
    assert_golden("thin_wall", &s.finish(), GOLDEN_THIN_WALL);
}

// ---------------------------------------------------------------------------
// 9-12. SDF collision family: collider / manifold / ccd / adaptive
// ---------------------------------------------------------------------------

const GOLDEN_SDF_COLLISION: &str =
    "ea75865ae60f57919d58c4ad093738b50a378a5d4ca68d64e2ee3e742099d4d3";

#[test]
fn golden_sdf_collision_family() {
    use alice_physics::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
    use alice_physics::sdf_ccd::{sphere_trace_sdf, SdfCcdConfig};
    use alice_physics::sdf_collider::{
        collide_aabb_sdf, collide_capsule_sdf, collide_point_sdf, collide_sphere_sdf,
    };
    use alice_physics::sdf_manifold::{generate_sdf_manifold, ManifoldConfig};
    let collider = SdfCollider::new_static(
        Box::new(unit_sphere()),
        Vec3Fix::from_f32(0.1, -0.2, 0.05),
        QuatFix::IDENTITY,
    );
    let mut s = Sink::default();
    let push = |s: &mut Sink, c: Option<Contact>| {
        s.bool(c.is_some());
        if let Some(c) = c {
            s.vec3(c.point_a);
            s.vec3(c.point_b);
            s.vec3(c.normal);
            s.fix(c.depth);
        }
    };
    for i in 0..16u32 {
        let p = Vec3Fix::from_f32(
            prand(i) * 2.0 - 1.0,
            prand(i + 9) * 2.0 - 1.0,
            prand(i + 17) * 2.0 - 1.0,
        );
        push(&mut s, collide_point_sdf(p, &collider));
        push(
            &mut s,
            collide_sphere_sdf(p, Fix128::from_ratio(1, 4), &collider),
        );
        push(
            &mut s,
            collide_capsule_sdf(
                p,
                p + Vec3Fix::from_f32(0.0, 0.3, 0.0),
                Fix128::from_ratio(1, 10),
                &collider,
            ),
        );
        push(
            &mut s,
            collide_aabb_sdf(
                p - Vec3Fix::from_f32(0.2, 0.2, 0.2),
                p + Vec3Fix::from_f32(0.2, 0.2, 0.2),
                &collider,
            ),
        );
    }
    let manifold = generate_sdf_manifold(
        Vec3Fix::from_f32(0.9, 0.2, 0.0),
        Fix128::from_ratio(3, 10),
        &collider,
        &ManifoldConfig::default(),
    );
    s.usize(manifold.len());
    if let Some(c) = manifold.deepest() {
        s.vec3(c.point_a);
        s.vec3(c.normal);
        s.fix(c.depth);
    }
    for i in 0..8u32 {
        let start = Vec3Fix::from_f32(-3.0, prand(i) - 0.5, prand(i + 5) - 0.5);
        let toi = sphere_trace_sdf(
            start,
            Vec3Fix::from_f32(5.0, 0.0, 0.0),
            Fix128::from_ratio(1, 10),
            &collider,
            &SdfCcdConfig::default(),
        );
        s.bool(toi.is_some());
        if let Some(t) = toi {
            s.fix(t.t);
            s.vec3(t.point);
            s.vec3(t.normal);
        }
    }
    let mut adaptive = AdaptiveSdfEvaluator::new(4, AdaptiveConfig::default());
    for frame in 0..6u32 {
        adaptive.begin_frame();
        for b in 0..4usize {
            let p = Vec3Fix::from_f32(0.5 + 0.05 * frame as f32, 0.1 * b as f32, 0.2);
            let (d, n) = adaptive.evaluate(b, p, &collider);
            s.f32(d);
            s.f32x3(n);
        }
    }
    assert_golden("sdf_collision_family", &s.finish(), GOLDEN_SDF_COLLISION);
}

// ---------------------------------------------------------------------------
// 13-16. SDF soft-body / character family: fem_mesh / character / wind / gpu batch
// ---------------------------------------------------------------------------

const GOLDEN_SDF_SOFT: &str = "fe673fb6d5ad25c6322228f1f39711f0edd61b810a61d36bdda8c1640e45550a";

#[test]
fn golden_sdf_soft_family() {
    use alice_physics::gpu_sdf::{GpuDispatchConfig, GpuSdfBatch};
    use alice_physics::sdf_character::SdfCharacter;
    use alice_physics::sdf_fem_mesh::generate;
    use alice_physics::sdf_wind_field::SdfWindField;
    let sphere = unit_sphere();
    let mut s = Sink::default();

    let mut mesh = generate(&sphere, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0.5);
    s.usize(mesh.vertex_count());
    s.usize(mesh.tet_count());
    s.f32(mesh.max_edge_length());
    let passes = mesh.refine_by_max_edge_length(0.6, 2);
    s.u64(u64::from(passes));
    s.usize(mesh.vertex_count());
    for v in mesh.vertices.iter().take(48) {
        s.arr3(*v);
    }

    // Character walking over the sphere: ground SDF = inverted half-space + sphere
    let ground = ClosureSdf::new(
        |x, y, z| {
            let sphere = (x * x + y * y + z * z).sqrt() - 1.0;
            let plane = y + 1.5;
            sphere.min(plane)
        },
        |x, y, z| {
            let sphere = (x * x + y * y + z * z).sqrt() - 1.0;
            let plane = y + 1.5;
            if sphere < plane {
                let len = (x * x + y * y + z * z).sqrt();
                (x / len, y / len, z / len)
            } else {
                (0.0, 1.0, 0.0)
            }
        },
    );
    let mut ch = SdfCharacter::new([-2.0, 0.2, 0.0], 0.3, 1.0);
    for _ in 0..12 {
        let out = ch.move_and_slide(&ground, [0.25, -0.1, 0.02]);
        ch.position = out.position;
        s.arr3(out.position);
        s.bool(out.converged);
        s.usize(out.iterations);
        s.bool(ch.is_grounded(&ground));
    }

    let wind = SdfWindField::new(&sphere, [1.0, 0.0, 0.0], 8.0);
    for i in 0..16u32 {
        s.arr3(wind.sample([
            prand(i) * 6.0 - 3.0,
            prand(i + 2) * 2.0 - 1.0,
            prand(i + 4) * 2.0 - 1.0,
        ]));
    }

    let mut batch = GpuSdfBatch::new(GpuDispatchConfig::default());
    for i in 0..8u32 {
        batch.add_query(
            i as usize,
            Vec3Fix::from_f32(prand(i) * 2.0 - 1.0, prand(i + 1) * 2.0 - 1.0, 0.0),
            Fix128::from_ratio(1, 5),
        );
    }
    s.usize(batch.query_count());
    s.u64(u64::from(batch.num_workgroups()));
    s.0.extend_from_slice(batch.query_bytes());
    batch.prepare_output();
    // Fill results deterministically as a GPU would, then extract.
    for (i, b) in batch.result_bytes_mut().iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31);
    }
    for c in batch.extract_contacts(0.2) {
        s.usize(c.body_index);
        s.f32(c.penetration);
        s.f32x3(c.normal);
        s.f32(c.distance);
    }
    assert_golden("sdf_soft_family", &s.finish(), GOLDEN_SDF_SOFT);
}

// ---------------------------------------------------------------------------
// 17-22. Modifier family: thermal / transient_thermal / phase_change / erosion /
//        pressure / sim_modifier (composed)
// ---------------------------------------------------------------------------

const GOLDEN_MODIFIERS: &str = "d252f5a987e2d904bb53095fa9c1a30ae9f3ff993ddd214ae6ead319b5c41b42";

#[test]
fn golden_modifier_family() {
    use alice_physics::erosion::{ErosionConfig, ErosionModifier};
    use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
    use alice_physics::pressure::{PressureConfig, PressureModifier};
    use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier};
    use alice_physics::thermal::{ThermalConfig, ThermalModifier};
    use alice_physics::transient_thermal::{
        crank_nicolson_step_1d, stable_dt_1d, transient_step_1d, ThermalMaterial,
    };
    let bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    let mut s = Sink::default();

    let mut th = ThermalModifier::new(ThermalConfig::default(), 8, bounds.0, bounds.1);
    th.add_heat_point(0.3, 0.2, 0.0, 500.0, 0.4);
    let mut pc = PhaseChangeModifier::new(PhaseChangeConfig::default(), 8, bounds.0, bounds.1);
    let mut er = ErosionModifier::new(ErosionConfig::default(), 8, bounds.0, bounds.1);
    let mut pr = PressureModifier::new(PressureConfig::default(), 8, bounds.0, bounds.1);
    for i in 0..6u32 {
        let (x, y, z) = (prand(i) - 0.5, prand(i + 3) - 0.5, prand(i + 6) - 0.5);
        th.apply_heat_at(x, y, z, 300.0, 0.3);
        pc.apply_heat_at(x, y, z, 800.0, 0.3);
        er.set_exposure_at(x, y, z, 1.0, 0.3);
        pr.apply_pressure_at(x, y, z, 50.0, 0.3);
        pr.apply_impact(-x, -y, -z, 5.0, 0.2);
        th.update(0.016);
        pc.update(0.016);
        er.update(0.016);
        pr.update(0.016);
    }
    for i in 0..24u32 {
        let (x, y, z) = (
            prand(i + 50) - 0.5,
            prand(i + 51) - 0.5,
            prand(i + 52) - 0.5,
        );
        s.f32(th.temperature_at(x, y, z));
        s.f32(th.modify_distance(x, y, z, 0.05));
        s.f32(pc.temperature_at(x, y, z));
        s.0.push(pc.phase_at(x, y, z) as u8);
        s.f32(pc.modify_distance(x, y, z, 0.05));
        s.f32(er.erosion_at(x, y, z));
        s.f32(er.modify_distance(x, y, z, 0.05));
        s.f32(pr.pressure_at(x, y, z));
        s.f32(pr.deformation_at(x, y, z));
        s.f32(pr.modify_distance(x, y, z, 0.05));
    }

    // Composition through ModifiedSdf
    let mut composed = ModifiedSdf::new(Box::new(unit_sphere()));
    let mut th2 = ThermalModifier::new(ThermalConfig::default(), 6, bounds.0, bounds.1);
    th2.apply_heat_at(0.9, 0.0, 0.0, 2000.0, 0.5);
    composed.add_modifier(Box::new(th2));
    composed.add_modifier(Box::new(PressureModifier::new(
        PressureConfig::default(),
        6,
        bounds.0,
        bounds.1,
    )));
    for _ in 0..3 {
        composed.update(0.016);
    }
    s.usize(composed.modifier_count());
    for i in 0..16u32 {
        let (x, y, z) = (
            prand(i + 80) * 2.0 - 1.0,
            prand(i + 81) * 2.0 - 1.0,
            prand(i + 82) * 2.0 - 1.0,
        );
        s.f32(composed.distance(x, y, z));
        s.f32x3(composed.normal(x, y, z));
    }

    // 1-D transient conduction (sin only appears in tests; production is + - * /)
    let mat = ThermalMaterial::aluminum_6061();
    let mut t: Vec<f32> = (0..32).map(|i| 300.0 + 200.0 * prand(i)).collect();
    let dt = stable_dt_1d(&t, &mat, 0.01) * 0.9;
    s.f32(dt);
    for _ in 0..10 {
        transient_step_1d(&mut t, &mat, 0.01, dt);
    }
    let mut t2 = t.clone();
    for _ in 0..10 {
        crank_nicolson_step_1d(&mut t2, &mat, 0.01, dt * 4.0);
    }
    for v in t.iter().chain(t2.iter()) {
        s.f32(*v);
    }
    s.f32(mat.conductivity_at(450.0));
    s.f32(mat.diffusivity_at(450.0));
    assert_golden("modifier_family", &s.finish(), GOLDEN_MODIFIERS);
}

// ---------------------------------------------------------------------------
// 23-26. Engineering scalars: aeroelasticity / piezoelectric / acoustic_wave /
//        convex_decompose
// ---------------------------------------------------------------------------

const GOLDEN_ENGINEERING: &str = "b7e637efa9a6a0f3ced6b94239afd71653f597d90c175a62fea474cf7438d397";

#[test]
fn golden_engineering_family() {
    use alice_physics::acoustic_wave::{leapfrog_step, stable_dt};
    use alice_physics::aeroelasticity::{viv_step, VivParameters, VivState};
    use alice_physics::convex_decompose::{decompose_sdf, DecomposeConfig};
    use alice_physics::piezoelectric::PiezoElement;
    let mut s = Sink::default();

    let mut st = VivState {
        displacement_m: 0.001,
        velocity_m_s: 0.0,
        wake_q: 0.1,
        wake_qdot: 0.0,
    };
    let params = VivParameters::facchinetti_reference();
    for _ in 0..200 {
        viv_step(&mut st, &params, 0.002);
    }
    s.f32(st.displacement_m);
    s.f32(st.velocity_m_s);
    s.f32(st.wake_q);
    s.f32(st.wake_qdot);

    for el in [
        PiezoElement::pzt_5a(1.0e-4, 1.0e-3),
        PiezoElement::quartz(1.0e-4, 1.0e-3),
        PiezoElement::pvdf(1.0e-4, 1.0e-3),
    ] {
        s.f32(el.permittivity());
        s.f32(el.voltage_from_force(12.5));
        s.f32(el.force_from_voltage(3.3));
        s.f32(el.strain_under_stress(2.0e6));
    }

    let dt = stable_dt(0.01, 343.0);
    s.f32(dt);
    let courant = 343.0 * dt / 0.01;
    let mut prev: Vec<f32> = (0..64).map(|i| prand(i) * 0.1).collect();
    let mut cur = prev.clone();
    let mut next = vec![0.0f32; 64];
    for _ in 0..40 {
        leapfrog_step(&cur, &prev, &mut next, courant);
        prev.copy_from_slice(&cur);
        cur.copy_from_slice(&next);
    }
    for v in &cur {
        s.f32(*v);
    }

    let res = decompose_sdf(
        &unit_sphere(),
        Vec3Fix::from_f32(-1.2, -1.2, -1.2),
        Vec3Fix::from_f32(1.2, 1.2, 1.2),
        &DecomposeConfig {
            resolution: 12,
            max_hulls: 4,
            ..DecomposeConfig::default()
        },
    );
    s.usize(res.hulls.len());
    for (h, (c, v)) in res
        .hulls
        .iter()
        .zip(res.centers.iter().zip(res.volumes.iter()))
    {
        s.usize(h.vertices.len());
        s.vec3(*c);
        s.fix(*v);
    }
    assert_golden("engineering_family", &s.finish(), GOLDEN_ENGINEERING);
}

// ---------------------------------------------------------------------------
// 27-30. Statistics / state / netcode: anomaly / character_state / fluid_netcode
//        (db_bridge is pass-through I/O over alice-db and has no arithmetic)
// ---------------------------------------------------------------------------

const GOLDEN_STATS_STATE: &str = "ed0629e3c73c081c875c19fe34bed6590210d9c0af2d898c5d8876daf0f13c6f";

#[test]
fn golden_stats_state_family() {
    use alice_physics::anomaly::{MadDetector, StreamingMedian};
    use alice_physics::character_state::{transition, CharacterState, CharacterStateContext};
    use alice_physics::fluid_netcode::{FluidDelta, FluidSnapshot};
    let mut s = Sink::default();

    let mut med = StreamingMedian::new();
    let mut mad = MadDetector::new(3.5);
    for i in 0..256u32 {
        let v = f64::from(prand(i)) * 10.0 + if i % 50 == 0 { 100.0 } else { 0.0 };
        med.push(v);
        mad.observe(v);
        if i % 32 == 31 {
            s.f64(med.median());
            s.f64(mad.median());
            s.f64(mad.mad());
            s.f64(mad.anomaly_score(v));
            s.bool(mad.is_anomaly(v));
        }
    }

    let mut state = CharacterState::Grounded;
    for i in 0..32u32 {
        let ctx = CharacterStateContext {
            is_grounded: i % 4 != 1,
            slope_radians: prand(i) * 1.2,
            in_water: i % 7 == 0,
            crouch_requested: i % 5 == 0,
            jump_pressed: i % 6 == 0,
            ..CharacterStateContext::standing()
        };
        state = transition(state, ctx);
        s.0.push(state as u8);
    }

    let pos: Vec<Vec3Fix> = (0..16u32)
        .map(|i| Vec3Fix::from_f32(prand(i), prand(i + 1), prand(i + 2)))
        .collect();
    let vel: Vec<Vec3Fix> = (0..16u32)
        .map(|i| Vec3Fix::from_f32(prand(i + 3) - 0.5, 0.0, prand(i + 4) - 0.5))
        .collect();
    let snap = FluidSnapshot::capture(&pos, &vel, 7);
    s.usize(snap.size_bytes());
    s.bool(snap.verify(&pos, &vel));
    let mut pos2 = pos.clone();
    pos2[3] = pos2[3] + Vec3Fix::from_f32(0.5, 0.0, 0.0);
    pos2[9] = pos2[9] + Vec3Fix::from_f32(0.0, 0.0, 0.001);
    let delta = FluidDelta::compute(&pos, &vel, &pos2, &vel, Fix128::from_ratio(1, 100), 7, 8);
    s.usize(delta.changed_count());
    s.f32(delta.compression_ratio(16));
    let (mut rp, mut rv) = snap.restore().expect("snapshot restores");
    delta.apply(&mut rp, &mut rv);
    for p in &rp {
        s.vec3(*p);
    }
    assert_golden("stats_state_family", &s.finish(), GOLDEN_STATS_STATE);
}
