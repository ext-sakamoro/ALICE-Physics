//! Goldens of the Law outputs: every src file that calls the Law max / min /
//! clamp (`det_math::{max, min, clamp}`, ALICE-Law G2) has a golden here, and
//! each golden is an entry of `PHYSICS_SEMANTICS_PINS`, so a change of any of
//! those results moves `PHYSICS_SEMANTICS_ID`.
//!
//! Each test drives public entry points with fixed inputs (signed zeros and
//! NaN where an input reaches a Law call, ordinary values otherwise), collects
//! the bits of the outputs, and compares their SHA-256 with a constant. Where
//! a NaN result's sign is not fixed by the Law (an `inf - inf` or a NaN after
//! further arithmetic), the record holds whether the value is NaN, not its bits.
//!
//! [`LAW_OUTPUT_PINS`] names the entry, the test and the src files it covers.
//! `law_output_pins_cover_every_law_caller` reads `src/` (comments, string
//! literals and `#[cfg(test)]` items blanked), lists the files that call the
//! Law max / min / clamp in any spelling (`det_math::max(`,
//! `crate::det_math::clamp(`, `alice_det_math::metric::clamp(`, an alias of
//! the module, or a bare `max(` after importing it), and fails when one of
//! them is not covered by a row, when a row covers a file that does not call
//! them, or when it found no caller at all.
//! `tests/physics_semantics_id.rs` reads the same table and checks that each
//! row's entry in `PHYSICS_SEMANTICS_PINS` is the constant its test asserts.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;

use alice_physics::anomaly::CompositeDetector;
use alice_physics::collider::Contact;
use alice_physics::convex_decompose::{decompose_sdf, DecomposeConfig};
use alice_physics::erosion::{ErosionConfig, ErosionModifier};
use alice_physics::fracture::{Crack, FractureConfig, FractureModifier};
use alice_physics::heatmap::{heatmap_to_rgba, Heatmap};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::privacy::PrivacyBudget;
use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::{ClosureSdf, SdfField, SdfUnion};
use alice_physics::sdf_destruction::{
    destruction_from_impact, DestructibleSdf, DestructionShape, DestructionType,
};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics::sdf_sph::{SphConfig, SphParticle, SphSolver, SphSpatialHash};
use alice_physics::sdf_wind_field::SdfWindField;
use alice_physics::shape::Shape;
use alice_physics::sim_field::ScalarField3D;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::sketch::{DDSketch, Mergeable};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics::thermal::{ThermalConfig, ThermalModifier};
use alice_physics::thin_wall::{measure_thickness_at, ThinWallConfig};
use alice_physics::transient_thermal::{crank_nicolson_step_1d_nonlinear, ThermalMaterial};
use alice_physics::world_participant::{ObservationSink, Participant};
use sha2::{Digest, Sha256};

/// A golden of the Law outputs: (entry of `PHYSICS_SEMANTICS_PINS`, test fn
/// of this file, the src files it covers, comma-separated).
const LAW_OUTPUT_PINS: &[(&str, &str, &str)] = &[
    (
        "law output: anomaly",
        "law_output_anomaly",
        "src/anomaly.rs",
    ),
    (
        "law output: heatmap",
        "law_output_heatmap",
        "src/heatmap.rs",
    ),
    (
        "law output: privacy",
        "law_output_privacy",
        "src/privacy.rs",
    ),
    ("law output: shape", "law_output_shape", "src/shape.rs"),
    ("law output: sketch", "law_output_sketch", "src/sketch.rs"),
    (
        "law output: convex_decompose",
        "law_output_convex_decompose",
        "src/convex_decompose.rs",
    ),
    (
        "law output: erosion",
        "law_output_erosion",
        "src/erosion.rs",
    ),
    (
        "law output: fracture",
        "law_output_fracture",
        "src/fracture.rs",
    ),
    (
        "law output: phase_change",
        "law_output_phase_change",
        "src/phase_change.rs",
    ),
    (
        "law output: pressure",
        "law_output_pressure",
        "src/pressure.rs",
    ),
    (
        "law output: sdf_character",
        "law_output_sdf_character",
        "src/sdf_character.rs",
    ),
    (
        "law output: sdf_collider",
        "law_output_sdf_collider",
        "src/sdf_collider.rs",
    ),
    (
        "law output: sdf_destruction",
        "law_output_sdf_destruction",
        "src/sdf_destruction.rs",
    ),
    (
        "law output: sdf_fem_mesh",
        "law_output_sdf_fem_mesh",
        "src/sdf_fem_mesh.rs",
    ),
    (
        "law output: sdf_sph",
        "law_output_sdf_sph",
        "src/sdf_sph.rs",
    ),
    (
        "law output: sdf_wind_field",
        "law_output_sdf_wind_field",
        "src/sdf_wind_field.rs",
    ),
    (
        "law output: sim_field",
        "law_output_sim_field",
        "src/sim_field.rs",
    ),
    (
        "law output: sim_modifier observe",
        "law_output_sim_modifier_observe",
        "src/sim_modifier.rs",
    ),
    (
        "law output: thermal",
        "law_output_thermal",
        "src/thermal.rs",
    ),
    (
        "law output: thin_wall",
        "law_output_thin_wall",
        "src/thin_wall.rs",
    ),
    (
        "law output: transient_thermal",
        "law_output_transient_thermal",
        "src/transient_thermal.rs",
    ),
    (
        "law output: tgs oriented 6dof hooks",
        "law_output_tgs_oriented_6dof_hooks",
        "src/solver_tgs_hooks_6dof_oriented.rs",
    ),
];

// ── Recording ───────────────────────────────────────────────────────────────

/// The bytes the goldens hash: each value as its bits, little-endian.
#[derive(Default)]
struct Record(Vec<u8>);

impl Record {
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn f32s(&mut self, vs: &[f32]) {
        for &v in vs {
            self.f32(v);
        }
    }
    /// Whether `v` is NaN, for a result whose NaN sign the Law leaves open,
    /// else its bits
    fn f32_or_nan(&mut self, v: f32) {
        if v.is_nan() {
            self.0.extend_from_slice(b"NaN!");
        } else {
            self.f32(v);
        }
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
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
    fn hex(&self) -> String {
        Sha256::digest(&self.0)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// A field whose distance and normal are fixed values
struct Constant(f32, (f32, f32, f32));

impl SdfField for Constant {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        black_box(self.0)
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        black_box(self.1)
    }
}

fn sphere(r: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - r,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            if l > 0.0 {
                (x / l, y / l, z / l)
            } else {
                (0.0, 1.0, 0.0)
            }
        },
    )
}

const UNIT: (f32, f32, f32) = (0.0, 0.0, 0.0);
const BOX: (f32, f32, f32) = (2.0, 2.0, 2.0);

// ── The goldens ─────────────────────────────────────────────────────────────

/// The largest of the three detector scores, with NaN scores
#[test]
fn law_output_anomaly() {
    const ANOMALY_GOLDEN: &str = "ba0f9ef3fde0df57aa55af0ecd88608875d326ad8b9ae88dd98ca257954375e1";
    let mut r = Record::default();
    for history in [&[1.0f64, 2.0, 3.0][..], &[1.0, 1.0, 1.0, 5.0], &[1.0]] {
        for query in [f64::NAN, f64::from_bits(0x7ff8_0000_0000_0001), 2.5, 50.0] {
            let mut c = CompositeDetector::new();
            for &v in history {
                c.observe(v);
            }
            r.u64(c.anomaly_score(black_box(query)).to_bits());
        }
    }
    assert_eq!(r.hex(), ANOMALY_GOLDEN);
}

/// The colour index cap `min(t * 4, 3.999)` of the heatmap palette
#[test]
fn law_output_heatmap() {
    const HEATMAP_GOLDEN: &str = "32d50d9108e2fce2cc0d4b55f82987e70625d97f15173cd37142b765ab4f8be7";
    let mut r = Record::default();
    let map = Heatmap {
        data: [0, 1, 2, 3, 4, 7, -2].map(Fix128::from_int).to_vec(),
        width: 7,
        height: 1,
        min_value: Fix128::ZERO,
        max_value: Fix128::from_int(4),
    };
    for px in heatmap_to_rgba(&map) {
        r.0.extend_from_slice(&px);
    }
    assert_eq!(r.hex(), HEATMAP_GOLDEN);
}

/// The remaining budget `max(max - spent, 0)` from `-0`, `+0` and NaN
#[test]
fn law_output_privacy() {
    const PRIVACY_GOLDEN: &str = "9c8f5949b5346420ca73e4ac029ddd07da9c4e74d0c08ddd4a0e1d4c8c46bff8";
    let mut r = Record::default();
    for max in [-0.0f64, 0.0, f64::NAN, 1.5] {
        r.u64(PrivacyBudget::new(black_box(max)).remaining().to_bits());
    }
    assert_eq!(r.hex(), PRIVACY_GOLDEN);
}

/// The longest dimension of a shape, which decides whether its mass is
/// representable: the long side in each operand position
#[test]
fn law_output_shape() {
    const SHAPE_GOLDEN: &str = "d3a0cb188c4331c280ac3bf3678570c850c7ac03563c8c44830569560596f36a";
    let mut r = Record::default();
    let fx = Fix128::from_int;
    let mut shapes = Vec::new();
    for long in [3000i64, 3200] {
        shapes.extend([
            Shape::Box {
                half_extents: Vec3Fix::from_int(long, 1, 1),
            },
            Shape::Box {
                half_extents: Vec3Fix::from_int(1, long, 1),
            },
            Shape::Box {
                half_extents: Vec3Fix::from_int(1, 1, long),
            },
            Shape::Cylinder {
                radius: fx(long),
                half_height: fx(1),
            },
            Shape::Cone {
                radius: fx(1),
                half_height: fx(long),
            },
            Shape::Ellipsoid {
                radii: Vec3Fix::from_int(1, long, 1),
            },
            Shape::Wedge {
                width: fx(1),
                height: fx(1),
                depth: fx(long),
            },
            Shape::Wedge {
                width: fx(long),
                height: fx(1),
                depth: fx(1),
            },
        ]);
    }
    for s in &shapes {
        match s.mass_and_inertia(Fix128::ONE) {
            Ok((mass, inertia)) => {
                r.u32(1);
                r.fix(mass);
                r.vec3(inertia);
            }
            Err(e) => {
                r.u32(0);
                r.0.extend_from_slice(format!("{e:?}").as_bytes());
            }
        }
    }
    assert_eq!(r.hex(), SHAPE_GOLDEN);
}

/// The running minimum and maximum of a sketch, on insert and on merge,
/// with signed zeros in both orders
#[test]
fn law_output_sketch() {
    const SKETCH_GOLDEN: &str = "83275eec34cb7e18cf3c5db2582b25047ba6cbc254fd08036db12fb4b9291b31";
    let mut r = Record::default();
    for values in [
        &[0.0f64, -0.0][..],
        &[-0.0, 0.0],
        &[-0.0],
        &[2.5, -1.0, 0.0],
    ] {
        let mut s = DDSketch::new(0.02);
        for &v in values {
            s.insert(black_box(v));
        }
        r.u64(s.min().to_bits());
        r.u64(s.max().to_bits());
    }
    for (a, b) in [(0.0f64, -0.0f64), (-0.0, 0.0)] {
        let mut x = DDSketch::new(0.02);
        x.insert(black_box(a));
        let mut y = DDSketch::new(0.02);
        y.insert(black_box(b));
        x.merge(&y);
        r.u64(x.min().to_bits());
        r.u64(x.max().to_bits());
    }
    assert_eq!(r.hex(), SKETCH_GOLDEN);
}

/// `decompose_sdf` of a concave solid (the floor of the cell range of the
/// concavity)
#[test]
fn law_output_convex_decompose() {
    const CONVEX_DECOMPOSE_GOLDEN: &str =
        "504fba138f56a4bc99a3141402bea5d794fd1e3b69f0224c415e50d95817350d";
    let mut r = Record::default();
    let config = DecomposeConfig {
        resolution: 8,
        ..DecomposeConfig::default()
    };
    // two spheres joined by a thin waist: concave, so the concavity (the cell
    // range of the hull's bounding box) decides the cut
    let dumbbell = ClosureSdf::new(
        |x, y, z| {
            let a = ((x - 1.0) * (x - 1.0) + y * y + z * z).sqrt() - 0.8;
            let b = ((x + 1.0) * (x + 1.0) + y * y + z * z).sqrt() - 0.8;
            let waist = (y * y + z * z).sqrt() - 0.2;
            a.min(b).min(waist.max(x.abs() - 1.0))
        },
        |_, _, _| (0.0, 1.0, 0.0),
    );
    let result = decompose_sdf(
        &dumbbell,
        Vec3Fix::from_int(-2, -2, -2),
        Vec3Fix::from_int(2, 2, 2),
        &config,
    );
    r.u32(u32::try_from(result.hulls.len()).unwrap());
    for hull in &result.hulls {
        r.u32(u32::try_from(hull.vertices.len()).unwrap());
        for &v in &hull.vertices {
            r.vec3(v);
        }
    }
    assert_eq!(r.hex(), CONVEX_DECOMPOSE_GOLDEN);
}

/// The erosion depth cap `min(rate * dt + depth, max_depth)`: a `-0` pair
/// and a NaN depth
#[test]
fn law_output_erosion() {
    const EROSION_GOLDEN: &str = "3d4bbcb2c3107120bb33a7b91821dbeda2c564a13dd3c640e25c30bd38042746";
    let mut r = Record::default();
    for (rate, depth) in [
        (-0.0f32, -0.0f32),
        (0.5, f32::NAN),
        (f32::NAN, 0.25),
        (0.5, 0.25),
    ] {
        let config = ErosionConfig {
            rate: black_box(rate),
            max_depth: 0.0,
            smoothing: 0.0,
            ..ErosionConfig::default()
        };
        let mut m = ErosionModifier::new(config, 2, UNIT, BOX);
        m.exposure.data.fill(1.0);
        m.erosion_depth.data.fill(black_box(depth));
        m.update(0.1);
        r.f32s(&m.erosion_depth.data);
    }
    assert_eq!(r.hex(), EROSION_GOLDEN);
}

fn crack_modifier(length: f32, speed: f32, max_len: f32) -> FractureModifier {
    let config = FractureConfig {
        propagation_speed: black_box(speed),
        max_crack_length: black_box(max_len),
        crack_width: 1.0,
        ..FractureConfig::default()
    };
    let mut m = FractureModifier::new(config, 2, UNIT, BOX);
    m.cracks.push(Crack {
        start: (0.0, 0.0, 0.0),
        end: (1.0, 0.0, 0.0),
        direction: (1.0, 0.0, 0.0),
        length: black_box(length),
        active: true,
    });
    m
}

/// The crack length cap and the crack carving `max(d, -crack_dist)`
#[test]
fn law_output_fracture() {
    const FRACTURE_GOLDEN: &str =
        "6547e36692326d189c321022b065b2567490da1fc3994bcd11ccc1d4bf684f5f";
    let mut r = Record::default();
    for (length, speed, max_len) in [
        (-0.0f32, -0.0f32, 0.0f32),
        (f32::NAN, 0.5, 2.0),
        (1.0, 0.5, 1.25),
    ] {
        let mut m = crack_modifier(length, speed, max_len);
        m.update(0.1);
        r.f32(m.cracks[0].length);
    }
    let m = crack_modifier(1.0, 0.0, 1.0);
    for (x, d) in [
        (0.5f32, 0.0f32),
        (0.5, -0.0),
        (f32::NAN, 0.75),
        (0.5, f32::NAN),
        (0.5, 0.25),
    ] {
        r.f32_or_nan(m.modify_distance(black_box(x), 1.0, 0.0, black_box(d)));
    }
    assert_eq!(r.hex(), FRACTURE_GOLDEN);
}

fn phase_modifier(config: PhaseChangeConfig) -> PhaseChangeModifier {
    PhaseChangeModifier::new(
        PhaseChangeConfig {
            diffusion_rate: 0.0,
            cooling_rate: 0.0,
            ..config
        },
        2,
        UNIT,
        BOX,
    )
}

fn record_phase(r: &mut Record, m: &PhaseChangeModifier) {
    for field in [&m.temperature, &m.phase, &m.latent_heat, &m.sdf_offset] {
        for &v in &field.data {
            r.f32_or_nan(v);
        }
    }
}

/// The latent heat bounds and the offset caps of the phase transitions
#[test]
fn law_output_phase_change() {
    const PHASE_CHANGE_GOLDEN: &str =
        "298c35d5de20f49859c56389809987bcfbf3c66a8ef9ec9e3efd8e5bb623c677";
    let mut r = Record::default();
    let base = PhaseChangeConfig::default();
    // a solid above the melting point, with a -0 or NaN latent heat bound
    for lh_fusion in [-0.0f32, f32::NAN, 50.0] {
        let mut m = phase_modifier(PhaseChangeConfig {
            latent_heat_fusion: black_box(lh_fusion),
            ..base
        });
        m.phase.data.fill(0.0);
        m.temperature.data.fill(base.melt_temperature + 50.0);
        m.latent_heat.data.fill(0.0);
        m.update(0.1);
        record_phase(&mut r, &m);
    }
    // a liquid above boiling with a NaN vaporization heat, and NaN stored heat
    for (lh_vap, stored) in [(f32::NAN, 0.0f32), (100.0, f32::NAN)] {
        let mut m = phase_modifier(PhaseChangeConfig {
            latent_heat_vaporization: black_box(lh_vap),
            ..base
        });
        m.phase.data.fill(1.0);
        m.temperature.data.fill(base.boil_temperature + 50.0);
        m.latent_heat
            .data
            .fill(black_box(base.latent_heat_fusion + stored));
        m.update(0.1);
        record_phase(&mut r, &m);
    }
    // a gas and a liquid offset at a zero cap, from -0 rates and offsets
    for phase in [2.0f32, 1.0] {
        let mut m = phase_modifier(PhaseChangeConfig {
            gas_expansion_rate: -0.0,
            gas_dissipation_rate: -0.0,
            liquid_flow_speed: -0.0,
            max_offset: 0.0,
            ..base
        });
        m.phase.data.fill(phase);
        m.temperature.data.fill(if phase == 2.0 {
            base.boil_temperature + 10.0
        } else {
            base.melt_temperature
        });
        m.latent_heat.data.fill(if phase == 2.0 {
            base.latent_heat_fusion + base.latent_heat_vaporization
        } else {
            base.latent_heat_fusion
        });
        m.sdf_offset.data.fill(black_box(-0.0));
        m.update(0.1);
        record_phase(&mut r, &m);
    }
    assert_eq!(r.hex(), PHASE_CHANGE_GOLDEN);
}

/// The impact dent cap and the yield cap, with `-0` and NaN operands
#[test]
fn law_output_pressure() {
    const PRESSURE_GOLDEN: &str =
        "ced6e512e72ddfdda18be4db0eee969765011b3b38d9e54c268b053c1177635f";
    let mut r = Record::default();
    for impulse in [-0.0f32, f32::NAN, 1.0e30, 0.5] {
        let mut m = PressureModifier::new(
            PressureConfig {
                max_deformation: 0.0,
                ..PressureConfig::default()
            },
            2,
            UNIT,
            BOX,
        );
        m.deformation.data.fill(-0.0);
        m.apply_impact(0.5, 0.5, 0.5, black_box(impulse), 1.0);
        r.f32s(&m.deformation.data);
    }
    for (rate, def) in [(-0.0f32, -0.0f32), (0.5, f32::NAN), (0.5, 0.25)] {
        let mut m = PressureModifier::new(
            PressureConfig {
                deformation_rate: black_box(rate),
                max_deformation: 0.5,
                yield_threshold: 10.0,
                ..PressureConfig::default()
            },
            2,
            UNIT,
            BOX,
        );
        m.pressure.data.fill(20.0);
        m.deformation.data.fill(black_box(def));
        m.update(0.1);
        for &v in &m.deformation.data {
            r.f32_or_nan(v);
        }
    }
    assert_eq!(r.hex(), PRESSURE_GOLDEN);
}

/// The push cap `min(push, max_push)` and the slope clamp
#[test]
fn law_output_sdf_character() {
    const SDF_CHARACTER_GOLDEN: &str =
        "b17e24f1544ea96ab55ed1b6949cb4fc2f3beabd8966f9a4981c8993c2174fa0";
    let mut r = Record::default();
    for (distance, max_push) in [(0.5f32, -0.0f32), (f32::NAN, 0.25), (0.5, 0.25)] {
        let mut c = SdfCharacter::new([-0.0; 3], 1.0, 2.0);
        c.skin_width = -0.5;
        c.max_push = black_box(max_push);
        c.max_iterations = 1;
        let out = c.move_and_slide(&Constant(distance, (0.0, 1.0, 0.0)), [-0.0; 3]);
        for v in out.position {
            r.f32_or_nan(v);
        }
    }
    for normal in [
        (0.0f32, 1.0f32, 0.0f32),
        (0.0, 1.5, 0.0),
        (f32::NAN, 1.0, 0.0),
    ] {
        let c = SdfCharacter::new([0.0; 3], 0.5, 2.0);
        let ctx = c.locomotion_context(&Constant(0.0, normal), false, false, false, 0.8);
        r.f32_or_nan(ctx.slope_radians);
    }
    assert_eq!(r.hex(), SDF_CHARACTER_GOLDEN);
}

/// The union of two distances and the finite-difference normal step
#[test]
fn law_output_sdf_collider() {
    const SDF_COLLIDER_GOLDEN: &str =
        "b6feffa57022484b7a84e13fec2faa7290e264f273d895c01dac963986e4db1e";
    let mut r = Record::default();
    for (a, b) in [
        (0.0f32, -0.0f32),
        (-0.0, 0.0),
        (f32::NAN, 1.5),
        (1.5, f32::NAN),
        (0.5, 0.25),
    ] {
        r.f32(
            SdfUnion::new(Constant(a, (0.0, 1.0, 0.0)), Constant(b, (0.0, 1.0, 0.0)))
                .distance(0.5, 0.25, 0.125),
        );
    }
    let s = sphere(1.0);
    for p in [(2.0f32, 0.5f32, 0.25f32), (1.0e5, 0.0, 1.0)] {
        let n = s.normal(p.0, p.1, p.2);
        r.f32s(&[n.0, n.1, n.2]);
    }
    assert_eq!(r.hex(), SDF_COLLIDER_GOLDEN);
}

/// The carving max, the shape bounds, the smooth blend clamp and the impact
/// radius clamp
#[test]
fn law_output_sdf_destruction() {
    const SDF_DESTRUCTION_GOLDEN: &str =
        "684e6d18da512f696f3620a1540adce95165550a37a97dbde8de60b30557a717";
    let mut r = Record::default();
    let custom = |v: f32| DestructionShape {
        center: Vec3Fix::ZERO,
        rotation: QuatFix::IDENTITY,
        shape: DestructionType::Custom {
            sdf: Arc::new(move |_, _, _| black_box(v)),
        },
        smooth_factor: 0.0,
    };
    for (original, carved) in [
        (0.0f32, 0.0f32),
        (-0.0, 0.0),
        (f32::NAN, 0.5),
        (0.5, f32::NAN),
    ] {
        let mut d = DestructibleSdf::new(Box::new(Constant(original, (0.0, 1.0, 0.0))));
        d.apply_destruction(custom(carved));
        r.f32_or_nan(d.distance(0.0, 0.0, 0.0));
    }
    for shape in [
        DestructionShape::cube(Vec3Fix::ZERO, (f32::NAN, 1.0, 1.0)),
        DestructionShape::cylinder(Vec3Fix::ZERO, f32::NAN, 1.0),
        DestructionShape::sphere(Vec3Fix::ZERO, 1.0).with_smoothing(0.5),
        DestructionShape::cube(Vec3Fix::ZERO, (0.5, 0.75, 1.0)),
    ] {
        let mut d = DestructibleSdf::new(Box::new(Constant(-5.0, (0.0, 1.0, 0.0))));
        d.apply_destruction(shape);
        r.f32_or_nan(d.distance(0.25, 0.0, 0.0));
    }
    let contact = Contact {
        depth: Fix128::ZERO,
        normal: Vec3Fix::from_int(0, 1, 0),
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    };
    for (scale, lo) in [
        (-0.0f32, 0.0f32),
        (f32::NAN, 0.0),
        (0.5, f32::NAN),
        (0.5, 0.25),
    ] {
        match destruction_from_impact(&contact, Fix128::from_int(2), black_box(scale), lo, 4.0)
            .shape
        {
            DestructionType::Sphere { radius } => r.f32(radius),
            _ => panic!("an impact makes a sphere"),
        }
    }
    assert_eq!(r.hex(), SDF_DESTRUCTION_GOLDEN);
}

/// The longest edge of a mesh, with a NaN vertex
#[test]
fn law_output_sdf_fem_mesh() {
    const SDF_FEM_MESH_GOLDEN: &str =
        "641feebc7a04d2f8a0c5367d2f687db315f1bb2c37a0b24d4d01bc7b800354f3";
    let mut r = Record::default();
    for corner in [1.0f32, f32::NAN] {
        let mesh = SdfTetMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, black_box(corner)],
            ],
            tets: vec![Tetrahedron {
                vertices: [0, 1, 2, 3],
            }],
        };
        r.f32(mesh.max_edge_length());
    }
    assert_eq!(r.hex(), SDF_FEM_MESH_GOLDEN);
}

/// The pressure floor `max(k * excess, 0)` on both density paths, and the
/// hash cell size floor
#[test]
fn law_output_sdf_sph() {
    const SDF_SPH_GOLDEN: &str = "d20c7444cc5a18c5548881c46015bca1768e13e15857ce94ab791a32c473c051";
    let mut r = Record::default();
    let far = Constant(100.0, (0.0, 1.0, 0.0));
    for (stiffness, rest) in [(-0.0f32, 0.0f32), (f32::NAN, 0.0), (2.0, 1000.0)] {
        for hashed in [false, true] {
            let config = SphConfig {
                gas_stiffness: black_box(stiffness),
                rest_density: rest,
                ..SphConfig::water_like()
            };
            let particles = vec![
                SphParticle::at_rest([0.0, 0.0, 0.0]),
                SphParticle::at_rest([0.01, 0.0, 0.0]),
            ];
            let mut s = SphSolver::new(particles, config, &far);
            if hashed {
                s.step_hashed(0.001);
            } else {
                s.step(0.001);
            }
            for p in &s.particles {
                r.f32_or_nan(p.pressure);
                r.f32_or_nan(p.density);
            }
        }
    }
    for k in [f32::NAN, -0.0, 0.5] {
        r.f32(SphSpatialHash::build(&[], black_box(k)).cell_size());
    }
    assert_eq!(r.hex(), SDF_SPH_GOLDEN);
}

/// The decay `clamp(max(d, 0) / scale, 0, 1)` of the wind near a surface
#[test]
fn law_output_sdf_wind_field() {
    const SDF_WIND_FIELD_GOLDEN: &str =
        "41b4e95ba70ff419c2f1c2083df0999e6e34b110e2fb430431dd578900f483a8";
    let mut r = Record::default();
    for (d, scale) in [(-0.0f32, 5.0f32), (0.0, 5.0), (1.0, f32::NAN), (2.5, 5.0)] {
        let field = Constant(d, (0.0, 1.0, 0.0));
        let mut w = SdfWindField::new(&field, [1.0, 0.0, 0.0], 1.0);
        w.decay_scale_m = black_box(scale);
        for v in w.sample([0.0, 0.0, 0.0]) {
            r.f32_or_nan(v);
        }
    }
    assert_eq!(r.hex(), SDF_WIND_FIELD_GOLDEN);
}

/// The grid coordinate clamp, the value clamp and the maximum of a field
#[test]
fn law_output_sim_field() {
    const SIM_FIELD_GOLDEN: &str =
        "98ea603b87edf5f774011a75b426b7545561e06d0870e72bc2275f0cfe41768d";
    let mut r = Record::default();
    let inputs = [
        -0.0f32,
        0.0,
        f32::NAN,
        f32::from_bits(0x7fc0_0001),
        -0.5,
        0.5,
        1.5,
    ];
    let mut f = ScalarField3D::new(inputs.len(), 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    f.data.copy_from_slice(&inputs);
    r.f32(f.max_value());
    for x in [-0.0f32, f32::NAN, 0.5, 2.0] {
        r.f32_or_nan(f.sample(black_box(x), 0.5, 0.5));
    }
    f.clamp(black_box(0.0), black_box(1.0));
    r.f32s(&f.data);
    assert_eq!(r.hex(), SIM_FIELD_GOLDEN);
}

/// The largest cell a participant reports (NaN cells skipped)
#[test]
fn law_output_sim_modifier_observe() {
    const SIM_MODIFIER_OBSERVE_GOLDEN: &str =
        "28068d73ca7e51dc188a25c39562709bf56620c522ae317b7c42b013cc5022e4";
    let mut r = Record::default();
    let mut m = ThermalModifier::new(ThermalConfig::default(), 2, UNIT, BOX);
    m.temperature
        .data
        .copy_from_slice(&[1.0, f32::NAN, -0.0, 3.5, 0.0, -2.0, 2.0, 0.5]);
    let mut out = ObservationSink::new();
    m.observe(&mut out);
    for &(channel, value) in out.values() {
        r.u32(channel);
        r.fix(value);
    }
    assert_eq!(r.hex(), SIM_MODIFIER_OBSERVE_GOLDEN);
}

/// The freezing factor of a cold cell
#[test]
fn law_output_thermal() {
    const THERMAL_GOLDEN: &str = "9c50b43dc262b5c078215b91c01c4914a18f13fbc05092a987ac2cd28a492040";
    let mut r = Record::default();
    for (temp, rate) in [(-20.0f32, 0.5f32), (-200.0, 0.5), (-20.0, f32::INFINITY)] {
        let mut m = ThermalModifier::new(
            ThermalConfig {
                freeze_rate: black_box(rate),
                ..ThermalConfig::default()
            },
            2,
            UNIT,
            BOX,
        );
        m.temperature.data.fill(black_box(temp));
        r.f32_or_nan(m.modify_distance(0.5, 0.5, 0.5, 0.25));
    }
    assert_eq!(r.hex(), THERMAL_GOLDEN);
}

/// The step floor of the wall march, with a NaN floor and a NaN field inside
#[test]
fn law_output_thin_wall() {
    const THIN_WALL_GOLDEN: &str =
        "5fd41dcdab079b2632a3e36f1f093949d451504b043eedd11230b27e1bad68f8";
    let mut r = Record::default();
    let slab = ClosureSdf::new(|_, y, _| y.abs() - 1.0, |_, y, _| (0.0, y.signum(), 0.0));
    let hollow = ClosureSdf::new(
        |_, y, _| {
            if y.abs() < 1.0 {
                f32::NAN
            } else {
                y.abs() - 1.0
            }
        },
        |_, y, _| (0.0, y.signum(), 0.0),
    );
    for (field, step) in [(&slab, 0.01f32), (&slab, f32::NAN), (&hollow, 0.05)] {
        let config = ThinWallConfig {
            min_step_mm: black_box(step),
            ..ThinWallConfig::default()
        };
        match measure_thickness_at(field, Vec3Fix::from_int(0, 1, 0), (0.0, 1.0, 0.0), &config) {
            Some(t) => {
                r.u32(1);
                r.fix(t);
            }
            None => r.u32(0),
        }
    }
    assert_eq!(r.hex(), THIN_WALL_GOLDEN);
}

/// The largest change of a nonlinear Crank-Nicolson step (NaN cells skipped)
#[test]
fn law_output_transient_thermal() {
    const TRANSIENT_THERMAL_GOLDEN: &str =
        "5c8159bac8ee768397456149e0ed151cf7b5975a3f3a0f13981b32bb975175b3";
    let mut r = Record::default();
    let material = ThermalMaterial::steel_1018();
    for cells in [
        vec![300.0f32, 400.0, 500.0, 400.0, 300.0],
        vec![300.0, f32::NAN, 500.0, 400.0, 300.0],
        vec![f32::NAN; 5],
    ] {
        let mut t = cells.clone();
        r.u32(crank_nicolson_step_1d_nonlinear(
            &mut t, &material, 0.01, 0.1, 1.0e-4, 50,
        ));
        for v in t {
            r.f32_or_nan(v);
        }
    }
    assert_eq!(r.hex(), TRANSIENT_THERMAL_GOLDEN);
}

/// A ball sliding on the ground under the TGS backend: its friction impulse
/// exceeds the cone, so the oriented 6-DOF hook scales it (the magnitude
/// floor of that scale)
#[test]
fn law_output_tgs_oriented_6dof_hooks() {
    const TGS_ORIENTED_6DOF_HOOKS_GOLDEN: &str =
        "79144576c7e17440faaaa14e5e1496c501bd5e929e83ebbdaece82ad87f115ff";
    let mut r = Record::default();
    let mut w = PhysicsWorld::new(SolverConfig {
        substeps: 4,
        solver_backend: SolverBackend::Tgs,
        ..SolverConfig::default()
    });
    let big = Fix128::from_int(10_000);
    w.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::new(Fix128::ZERO, -big, Fix128::ZERO)),
        big,
    );
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_f32(0.0, 0.49, 0.0), Fix128::ONE),
        Fix128::from_f32(0.5),
    );
    w.bodies[ball].velocity = Vec3Fix::from_int(3, 0, 0);
    for _ in 0..30 {
        w.step(Fix128::from_ratio(1, 60));
    }
    for body in &w.bodies {
        r.vec3(body.position);
        r.vec3(body.velocity);
    }
    assert_eq!(r.hex(), TGS_ORIENTED_6DOF_HOOKS_GOLDEN);
}

// ── Coverage ────────────────────────────────────────────────────────────────

fn read(rel: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
        .replace("\r\n", "\n")
}

/// Comments, string and char literals blanked (newlines kept)
fn blank_literals(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for c in &mut out[from..to] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    };
    while i < b.len() {
        match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                let end = src[i..].find('\n').map_or(b.len(), |n| i + n);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let end = src[i + 2..].find("*/").map_or(b.len(), |n| i + 2 + n + 2);
                blank(&mut out, i, end);
                i = end;
            }
            b'"' => {
                let mut j = i + 1;
                while j < b.len() && b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                blank(&mut out, i, (j + 1).min(b.len()));
                i = j + 1;
            }
            b'\'' if b.get(i + 2) == Some(&b'\'') || (b.get(i + 1) == Some(&b'\\')) => {
                let end = src[i + 1..].find('\'').map_or(b.len(), |n| i + 1 + n + 1);
                blank(&mut out, i, end);
                i = end;
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("blanking keeps UTF-8 boundaries of ASCII bytes")
}

/// `#[cfg(test)]` items blanked (the attribute through the item's closing brace)
fn blank_test_items(code: &str) -> String {
    let mut out = code.to_string();
    let mut from = 0;
    while let Some(n) = out[from..].find("#[cfg(test)]") {
        let start = from + n;
        let bytes = out.as_bytes();
        let mut depth = 0i32;
        let mut end = None;
        for (k, &c) in bytes.iter().enumerate().skip(start) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(k + 1);
                        break;
                    }
                }
                b';' if depth == 0 => {
                    end = Some(k + 1);
                    break;
                }
                _ => {}
            }
        }
        let end = end.unwrap_or(out.len());
        let blanked: String = out[start..end]
            .chars()
            .map(|c| if c == '\n' { '\n' } else { ' ' })
            .collect();
        out.replace_range(start..end, &blanked);
        from = end;
    }
    out
}

const LAW_FNS: [&str; 5] = ["max", "min", "clamp", "max64", "min64"];

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The Law max / min / clamp calls in one file's code (literals, comments and
/// test items already blanked), as `(line, spelling)`
fn law_calls(code: &str) -> Vec<(usize, String)> {
    // the module paths that name the Law functions here: the crate's
    // re-export, alice-det-math itself and its metric module, and any alias
    // `use <one of them> as <name>;`
    let mut modules: BTreeSet<String> = [
        "det_math",
        "crate::det_math",
        "alice_det_math",
        "alice_det_math::metric",
        "crate::det_math::metric",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let mut bare = false;
    for stmt in code.split(';') {
        let s: String = stmt.split_whitespace().collect::<Vec<_>>().join(" ");
        let Some(rest) = s.find("use ").map(|k| &s[k + 4..]) else {
            continue;
        };
        if let Some((path, alias)) = rest.split_once(" as ") {
            if modules.contains(path.trim()) {
                modules.insert(alias.trim().to_string());
            }
        }
        for m in modules.clone() {
            if let Some(list) = rest.strip_prefix(&format!("{m}::")) {
                let names: Vec<&str> = list
                    .trim_matches(|c| c == '{' || c == '}' || c == ' ')
                    .split(',')
                    .map(str::trim)
                    .collect();
                if names
                    .iter()
                    .any(|n| *n == "*" || LAW_FNS.contains(&n.split(" as ").next().unwrap_or("")))
                {
                    bare = true;
                }
            }
        }
    }
    let b = code.as_bytes();
    let mut calls = Vec::new();
    for f in LAW_FNS {
        let mut from = 0;
        while let Some(n) = code[from..].find(f) {
            let k = from + n;
            from = k + f.len();
            if (k > 0 && is_ident(b[k - 1])) || b.get(from).is_some_and(|&c| is_ident(c)) {
                continue;
            }
            // a qualified path counts called or passed as a value
            // (`.reduce(det_math::max)`); a bare name only when called
            let called = code[from..].trim_start().starts_with('(');
            let before = &code[..k];
            // a path in a `use` item is an import, not a use of the function
            let stmt = &before[before.rfind([';', '{', '}']).map_or(0, |p| p + 1)..];
            if stmt.split_whitespace().any(|w| w == "use") {
                continue;
            }
            let qualified = modules.iter().any(|m| {
                before.ends_with(&format!("{m}::"))
                    && !before[..before.len() - m.len() - 2]
                        .ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ':')
            });
            let bare_call = bare
                && called
                && !before.trim_end().ends_with('.')
                && !before.ends_with("::")
                && !before.trim_end().ends_with("fn");
            if qualified || bare_call {
                let line = code[..k].matches('\n').count() + 1;
                let start = before
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
                    .map_or(0, |p| p + 1);
                calls.push((line, code[start..from].to_string()));
            }
        }
    }
    calls.sort();
    calls
}

fn src_files(dir: &Path, out: &mut Vec<String>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            src_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let rel = p
                .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
}

/// Files under `src/` that call the Law max / min / clamp, with their calls
fn law_callers() -> BTreeMap<String, Vec<(usize, String)>> {
    let mut files = Vec::new();
    src_files(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    assert!(!files.is_empty(), "no src file read (compared nothing)");
    files
        .into_iter()
        .filter_map(|f| {
            let calls = law_calls(&blank_test_items(&blank_literals(&read(&f))));
            (!calls.is_empty()).then_some((f, calls))
        })
        .collect()
}

fn covered(rows: &[(&str, &str, &str)]) -> BTreeMap<String, Vec<String>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, _, files) in rows {
        for f in files.split(',') {
            map.entry(f.trim().to_string())
                .or_default()
                .push((*name).to_string());
        }
    }
    map
}

fn coverage_problems(
    callers: &BTreeMap<String, Vec<(usize, String)>>,
    rows: &[(&str, &str, &str)],
) -> Vec<String> {
    let cov = covered(rows);
    let mut problems = Vec::new();
    for (file, calls) in callers {
        if !cov.contains_key(file) {
            problems.push(format!(
                "{file} calls the Law max / min / clamp ({}) but no LAW_OUTPUT_PINS row covers it",
                calls
                    .iter()
                    .map(|(l, s)| format!("{s} at line {l}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    for (file, names) in &cov {
        if !callers.contains_key(file) {
            problems.push(format!(
                "{names:?} cover {file}, which calls no Law max / min / clamp"
            ));
        }
    }
    problems
}

#[test]
fn law_output_pins_cover_every_law_caller() {
    let callers = law_callers();
    assert!(
        !callers.is_empty(),
        "no Law max / min / clamp caller found (compared nothing)"
    );
    let problems = coverage_problems(&callers, LAW_OUTPUT_PINS);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // each row names a test of this file
    let me = read("tests/determinism_law_outputs.rs");
    for (_, test, _) in LAW_OUTPUT_PINS {
        assert!(
            me.contains(&format!("fn {test}()")),
            "{test} is not a test of this file"
        );
    }
    println!(
        "{} Law caller files, {} calls, all covered by {} rows",
        callers.len(),
        callers.values().map(Vec::len).sum::<usize>(),
        LAW_OUTPUT_PINS.len()
    );
}

/// The scan finds each spelling of a call and nothing else
#[test]
fn the_caller_scan_finds_every_spelling() {
    let found = |src: &str| law_calls(&blank_test_items(&blank_literals(src)));
    for src in [
        "fn f(a: f32) -> f32 { crate::det_math::max(a, 0.0) }",
        "fn f(a: f32) -> f32 { det_math::clamp(a, 0.0, 1.0) }",
        "fn f(a: f32) -> f32 { alice_det_math::metric::clamp(a, 0.0, 1.0) }",
        "fn f(a: f32) -> f32 { alice_det_math::min(a, 0.0) }",
        "use crate::det_math as dm;\nfn f(a: f32) -> f32 { dm::min(a, 0.0) }",
        "use crate::det_math::{max, sqrt};\nfn f(a: f32) -> f32 { max(a, 0.0) }",
        "use crate::det_math::*;\nfn f(a: f32) -> f32 { clamp(a, 0.0, 1.0) }",
        "use alice_det_math::metric::clamp;\nfn f(a: f32) -> f32 { clamp (a, 0.0, 1.0) }",
        "fn f(a: f64) -> f64 { crate::det_math::max64(a, 0.0) }",
        "fn f(v: &[f32]) -> Option<f32> { v.iter().copied().reduce(crate::det_math::max) }",
    ] {
        assert_eq!(found(src).len(), 1, "{src}");
    }
    for src in [
        "fn f(a: f32) -> f32 { a.max(0.0) }",
        "fn f(a: f32) -> f32 { f32::max(a, 0.0) }",
        "fn f(a: f32) -> f32 { crate::det_math::sqrt(a) }",
        "fn max(a: f32) -> f32 { a }",
        "fn f(a: f32) -> f32 { other::max(a, 0.0) }",
        "fn f() { let s = \"det_math::max(1.0, 2.0)\"; }",
        "// det_math::max(a, 0.0)\nfn f() {}",
        "#[cfg(test)]\nmod tests { fn g(a: f32) -> f32 { crate::det_math::max(a, 0.0) } }",
        "fn f(a: f32) -> f32 { max(a, 0.0) }",
        "use crate::det_math::max;\nfn f(v: &[f32]) -> usize { let max = 3; max }",
    ] {
        assert!(found(src).is_empty(), "{src}: {:?}", found(src));
    }
}

/// A caller with no row, and a row with no caller, are both reported
#[test]
fn coverage_reports_a_missing_and_a_stale_row() {
    let mut callers = BTreeMap::new();
    callers.insert(
        "src/a.rs".to_string(),
        vec![(3, "det_math::max".to_string())],
    );
    callers.insert(
        "src/b.rs".to_string(),
        vec![(9, "det_math::clamp".to_string())],
    );
    let rows = [
        ("law output: a", "t", "src/a.rs"),
        ("law output: c", "t", "src/c.rs"),
    ];
    let p = coverage_problems(&callers, &rows);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].contains("src/b.rs"), "{p:?}");
    assert!(p[1].contains("src/c.rs"), "{p:?}");
}
