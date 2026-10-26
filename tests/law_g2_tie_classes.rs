//! ALICE-Law G2: one tie test per class of scalar max / min site
//!
//! Each test goes through a public entry point of its class and checks the
//! result's bits for the operand pairs the Law fixes: maximumNumber /
//! minimumNumber give `+0` for the max and `-0` for the min of an equal pair
//! of zeros in either order, and the other operand when one operand is NaN.
//! A clamp follows alice-det-math's clamp instead (the owner of the clamp
//! Law): it compares, so `-0` against a `+0` bound stays `-0`, and a NaN in
//! any argument gives the canonical quiet NaN. The classes are listed in
//! `alice-law-conformance.toml` under `[G2]`.
//!
//! Where an entry point fixes one operand (`max(x, +0)`, `max(x, eps)`), only
//! the order it computes exists, and the test states which pair it feeds.

use alice_physics_core::pressure::{PressureConfig, PressureModifier};
use alice_physics_core::privacy::PrivacyBudget;
use alice_physics_core::sdf_collider::{SdfField, SdfUnion};
use alice_physics_core::sdf_sph::SphSpatialHash;
use alice_physics_core::sim_field::ScalarField3D;
use std::hint::black_box;

const POS: u32 = 0x0000_0000;
const NEG: u32 = 0x8000_0000;
const POS64: u64 = 0;
const NEG64: u64 = 0x8000_0000_0000_0000;
const CANONICAL_NAN: u32 = 0x7fc0_0000;
const CANONICAL_NAN64: u64 = 0x7ff8_0000_0000_0000;

/// A field whose distance is a fixed value
struct Constant(f32);

impl SdfField for Constant {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        black_box(self.0)
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

fn union_distance(a: f32, b: f32) -> u32 {
    SdfUnion::new(Constant(a), Constant(b))
        .distance(0.5, 0.25, 0.125)
        .to_bits()
}

/// Class: SDF distance combination (the union is the min of two distances)
#[test]
fn sdf_distance_combination_follows_minimum_number() {
    assert_eq!(union_distance(0.0, -0.0), NEG);
    assert_eq!(union_distance(-0.0, 0.0), NEG);
    // a NaN distance yields the other operand, in both orders
    assert_eq!(union_distance(f32::NAN, 1.5), 1.5f32.to_bits());
    assert_eq!(union_distance(1.5, f32::NAN), 1.5f32.to_bits());
}

/// Class: nonnegative part `max(x, +0)` (the remaining privacy budget)
#[test]
fn nonnegative_part_follows_maximum_number() {
    // -0 - +0 = -0, so this feeds the pair (-0, +0)
    assert_eq!(
        PrivacyBudget::new(black_box(-0.0)).remaining().to_bits(),
        POS64
    );
    assert_eq!(
        PrivacyBudget::new(black_box(0.0)).remaining().to_bits(),
        POS64
    );
    // NaN - 0 = NaN: maximumNumber yields the +0 operand
    assert_eq!(
        PrivacyBudget::new(black_box(f64::NAN))
            .remaining()
            .to_bits(),
        POS64
    );
}

fn deformation_after_impact(impulse: f32) -> u32 {
    let mut m = PressureModifier::new(
        PressureConfig::default(),
        4,
        (0.0, 0.0, 0.0),
        (3.0, 3.0, 3.0),
    );
    m.apply_impact(1.0, 1.0, 1.0, black_box(impulse), 1.5);
    m.deformation_at(1.0, 1.0, 1.0).to_bits()
}

/// Class: upper cap `min(x, cap)` (an impact's dent is capped at max_deformation)
///
/// A signed zero dent is absorbed by the field's `+0 + -0 = +0` before it can
/// be read, so the NaN operand is the observable tie here: minimumNumber
/// yields the cap, the same dent as an impulse far above it.
#[test]
fn upper_cap_follows_minimum_number() {
    let capped = deformation_after_impact(1.0e30);
    assert_eq!(deformation_after_impact(f32::NAN), capped);
    assert_ne!(
        f32::from_bits(capped),
        0.0,
        "the impact must leave a dent to compare"
    );
}

/// Class: positive floor `max(x, eps)` with eps > 0 (the spatial hash cell size)
///
/// A zero tie needs eps = +0, which this entry point does not have; the NaN
/// operand yields eps.
#[test]
fn positive_floor_follows_maximum_number() {
    let eps = SphSpatialHash::build(&[], black_box(1.0e-9)).cell_size();
    assert_eq!(
        SphSpatialHash::build(&[], black_box(f32::NAN))
            .cell_size()
            .to_bits(),
        eps.to_bits()
    );
    assert_eq!(
        SphSpatialHash::build(&[], black_box(-0.0))
            .cell_size()
            .to_bits(),
        eps.to_bits()
    );
    assert_eq!(eps.to_bits(), 1.0e-6f32.to_bits());
}

fn ramp_field() -> ScalarField3D {
    let mut f = ScalarField3D::new(3, 3, 3, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0));
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                f.set(x, y, z, 1.0 + (x + 3 * y + 9 * z) as f32);
            }
        }
    }
    f
}

/// Class: grid coordinate clamp `min(max(g, +0), n)` (field sampling)
///
/// A NaN coordinate clamps to the low edge (maximumNumber(NaN, +0) = +0),
/// so it reads the same bits as the low edge itself, and so does -0.
#[test]
fn grid_coordinate_clamp_follows_maximum_number() {
    let f = ramp_field();
    let edge = f.sample(0.0, 1.0, 1.0).to_bits();
    assert_eq!(f.sample(black_box(-0.0), 1.0, 1.0).to_bits(), edge);
    assert_eq!(f.sample(black_box(f32::NAN), 1.0, 1.0).to_bits(), edge);
    assert_eq!(
        f.sample(1.0, black_box(f32::NAN), 1.0).to_bits(),
        f.sample(1.0, 0.0, 1.0).to_bits()
    );
}

/// Class: reduction (the maximum of a field)
#[test]
fn reduction_follows_maximum_number() {
    for (a, b) in [(0.0f32, -0.0f32), (-0.0, 0.0)] {
        let mut f = ScalarField3D::new(2, 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        f.set(0, 0, 0, black_box(a));
        f.set(1, 0, 0, black_box(b));
        assert_eq!(f.max_value().to_bits(), POS, "({a:?}, {b:?})");
    }
    for (a, b) in [(f32::NAN, -2.5f32), (-2.5, f32::NAN)] {
        let mut f = ScalarField3D::new(2, 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        f.set(0, 0, 0, black_box(a));
        f.set(1, 0, 0, black_box(b));
        assert_eq!(
            f.max_value().to_bits(),
            (-2.5f32).to_bits(),
            "({a:?}, {b:?})"
        );
    }
}

/// Class: clamp with a zero bound
///
/// A clamp follows alice-det-math's clamp (the owner of the clamp Law). Since alice-det-math
/// 0.5.2 at 39507d1b that is `min(max(x, low), high)` with the G2 max and min, so `-0`
/// against a `+0` low bound gives `+0` (it compared before, and kept `-0`), and a NaN `x`
/// gives the canonical quiet NaN whatever its payload or sign.
#[test]
fn class7_clamp_with_a_zero_bound_follows_det_math_clamp() {
    let inputs = [
        -0.0f32,
        0.0,
        f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0000),
        f32::from_bits(0x7f80_0001),
        -0.5,
        0.5,
        1.5,
    ];
    let mut f = ScalarField3D::new(inputs.len(), 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
    for (i, &v) in inputs.iter().enumerate() {
        f.set(i, 0, 0, black_box(v));
    }
    f.clamp(black_box(0.0), black_box(1.0));
    let got: Vec<u32> = (0..inputs.len())
        .map(|i| f.get(i, 0, 0).to_bits())
        .collect();
    let want = [
        POS,
        POS,
        CANONICAL_NAN,
        CANONICAL_NAN,
        CANONICAL_NAN,
        CANONICAL_NAN,
        POS,
        0.5f32.to_bits(),
        1.0f32.to_bits(),
    ];
    assert_eq!(got, want, "{got:08x?}");
}

/// Class: clamp with a zero bound, through the std `f64::clamp` of a
/// probability. alice-det-math has no 64-bit clamp yet, so this site still
/// uses the std clamp: it agrees with the det-math rule for `-0` and for the
/// canonical NaN, and a non-canonical NaN payload is not checked here (the
/// std clamp passes it through)
#[test]
#[allow(
    deprecated,
    reason = "RandomizedResponse::with_probability is the public entry point of this f64 clamp site"
)]
fn class7_f64_clamp_agrees_with_det_math_0_5_rule_where_it_can() {
    use alice_physics_core::privacy::RandomizedResponse;
    let p = |x: f64| {
        RandomizedResponse::with_probability(black_box(x), 1)
            .p_true()
            .to_bits()
    };
    assert_eq!(p(-0.0), NEG64);
    assert_eq!(p(0.0), POS64);
    assert_eq!(p(f64::NAN), CANONICAL_NAN64);
    assert_eq!(p(1.5), 1.0f64.to_bits());
}

/// A field that answers a scripted sequence of distances, one per call, with
/// the normal `+y`
struct Scripted {
    distances: Vec<f32>,
    next: std::sync::atomic::AtomicUsize,
}

impl SdfField for Scripted {
    fn distance(&self, _x: f32, _y: f32, _z: f32) -> f32 {
        let i = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        black_box(self.distances[i])
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// Moves a character through `distances` (the loop samples, then the final
/// sample of a run that did not converge) and returns the selected distance
/// bits and the `y` of the selected position: sample `k` is taken at `y = k`
/// (each step is capped at 1 and the radius 10 is never reached)
fn select(distances: &[f32]) -> (u32, f32) {
    use alice_physics_core::sdf_character::SdfCharacter;
    let mut c = SdfCharacter::new([0.0, 0.0, 0.0], 10.0, 2.0);
    c.max_iterations = distances.len() - 1;
    c.skin_width = 0.0;
    c.max_push = 1.0;
    let field = Scripted {
        distances: distances.to_vec(),
        next: std::sync::atomic::AtomicUsize::new(0),
    };
    let out = c.move_and_slide(&field, [0.0, 0.0, 0.0]);
    assert!(!out.converged);
    assert_eq!(
        field.next.load(std::sync::atomic::Ordering::Relaxed),
        distances.len(),
        "every scripted distance was sampled"
    );
    (out.best_distance.to_bits(), out.best_position[1])
}

/// Class: selection (argmax with a payload), `SdfCharacter`'s best sample,
/// both the loop sample and the final sample of a run that did not converge.
/// The rule (alice-law-conformance.toml [G2]): the first strict maximum is
/// kept, a NaN distance is never selected, and a tie keeps the first, so of
/// `-0` and `+0` the earlier one stays
#[test]
fn selection_keeps_the_first_strict_maximum_and_never_a_nan() {
    let nan = f32::NAN;
    let cases: [(&[f32], u32, f32); 8] = [
        (&[0.5, 0.5, 0.5], 0.5f32.to_bits(), 0.0),
        (&[-0.0, 0.0, -1.0], NEG, 0.0),
        (&[0.0, -0.0, -1.0], POS, 0.0),
        (&[nan, -1.0, -2.0], (-1.0f32).to_bits(), 1.0),
        (&[-2.0, nan, -1.0], (-1.0f32).to_bits(), 2.0),
        (&[-1.0, -3.0, -1.0], (-1.0f32).to_bits(), 0.0),
        (&[-3.0, -2.0, -1.0], (-1.0f32).to_bits(), 2.0),
        (&[nan, nan, nan], f32::NEG_INFINITY.to_bits(), 0.0),
    ];
    let mut checked = 0;
    for (distances, bits, y) in cases {
        assert_eq!(select(distances), (bits, y), "{distances:?}");
        checked += 1;
    }
    assert_eq!(checked, 8);
}
