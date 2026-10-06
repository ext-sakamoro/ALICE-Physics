//! Audit oracles for `alice_physics::sdf_wind_field`.
//!
//! Expected values come from the module's documented model
//! `wind = direction * speed * clamp(d / decay_scale, 0, 1)` evaluated by
//! hand for planar and spherical SDFs, never from the sampler itself.

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_wind_field::SdfWindField;

/// Signed distance to the plane x = 0 (inside for x < 0).
fn wall() -> ClosureSdf {
    ClosureSdf::new(|x, _y, _z| x, |_x, _y, _z| (1.0, 0.0, 0.0))
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn sample_matches_documented_formula_on_a_grid() {
    let sdf = wall();
    let dir = [0.3_f32, -0.5, 0.7]; // non-unit, all three components distinct
    for &speed in &[1.0_f32, 8.0, -4.0] {
        for &decay in &[0.5_f32, 5.0, 12.0] {
            for &d in &[0.0_f32, 0.25, 1.0, 2.5, 5.0, 7.0, 40.0] {
                let mut f = SdfWindField::new(&sdf, dir, speed);
                f.decay_scale_m = decay;
                let v = f.sample([d, 3.0, -2.0]);
                let shelter = (d / decay).clamp(0.0, 1.0);
                for k in 0..3 {
                    let want = dir[k] * speed * shelter;
                    assert!(
                        close(v[k], want, 1.0e-6),
                        "axis {k} speed {speed} decay {decay} d {d}: got {} want {want}",
                        v[k]
                    );
                }
            }
        }
    }
}

#[test]
fn new_stores_arguments_and_defaults_decay_scale_to_five_metres() {
    let sdf = wall();
    let f = SdfWindField::new(&sdf, [0.0, 1.0, 0.0], 3.0);
    assert_eq!(f.base_direction, [0.0, 1.0, 0.0]);
    assert_eq!(f.base_speed_m_s, 3.0);
    assert_eq!(f.decay_scale_m, 5.0);
    // Default ramp: half the base wind at d = 2.5 m.
    let v = f.sample([2.5, 0.0, 0.0]);
    assert!(close(v[1], 1.5, 1.0e-6), "got {}", v[1]);
}

#[test]
fn wind_is_monotone_in_distance_and_bounded_by_base_wind() {
    let sdf = wall();
    let f = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 9.0);
    let mut prev = f.sample([-3.0, 0.0, 0.0])[0];
    let mut d = -3.0_f32;
    while d < 30.0 {
        let w = f.sample([d, 0.0, 0.0])[0];
        assert!(w >= prev - 1.0e-6, "not monotone at d={d}");
        assert!(
            (0.0..=9.0 + 1.0e-5).contains(&w),
            "out of [0, base] at d={d}: {w}"
        );
        prev = w;
        d += 0.37;
    }
}

#[test]
fn ramp_is_scale_invariant_in_distance_and_decay() {
    let sdf = wall();
    let mut a = SdfWindField::new(&sdf, [0.0, 0.0, 1.0], 2.0);
    let mut b = SdfWindField::new(&sdf, [0.0, 0.0, 1.0], 2.0);
    a.decay_scale_m = 4.0;
    b.decay_scale_m = 8.0;
    let va = a.sample([1.0, 0.0, 0.0])[2];
    let vb = b.sample([2.0, 0.0, 0.0])[2];
    assert!(close(va, vb, 1.0e-6), "{va} vs {vb}");
}

#[test]
fn wind_depends_on_position_only_through_the_sdf() {
    let sdf = wall();
    let f = SdfWindField::new(&sdf, [0.0, 0.0, 1.0], 6.0);
    let a = f.sample([2.0, 100.0, -50.0]);
    let b = f.sample([2.0, -7.0, 33.0]);
    assert_eq!(a, b);
}

#[test]
fn works_through_a_trait_object() {
    let sdf = wall();
    let dynref: &dyn SdfField = &sdf;
    let f = SdfWindField::new(dynref, [1.0, 0.0, 0.0], 5.0);
    assert!(close(f.sample([5.0, 0.0, 0.0])[0], 5.0, 1.0e-6));
}

#[test]
fn spherical_obstacle_shelter_on_the_axes() {
    // Unit-sphere SDF: d = |p| - 1. Wind +x, decay 2: at p = (2,0,0) d = 1 -> 0.5.
    let sphere = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1.0e-9);
            (x / l, y / l, z / l)
        },
    );
    let mut f = SdfWindField::new(&sphere, [1.0, 0.0, 0.0], 10.0);
    f.decay_scale_m = 2.0;
    assert!(close(f.sample([2.0, 0.0, 0.0])[0], 5.0, 1.0e-5));
    assert!(close(f.sample([0.0, 2.0, 0.0])[0], 5.0, 1.0e-5));
    assert_eq!(f.sample([0.5, 0.0, 0.0])[0], 0.0); // inside
    assert!(close(f.sample([5.0, 0.0, 0.0])[0], 10.0, 1.0e-5));
}

/// With `decay_scale_m <= 0` the sampler returns the full base wind even at
/// points inside the obstacle, while any positive decay (including 1e-30)
/// gives zero wind there. The limit decay -> 0+ and the module's statement
/// that inside points get no wind both say zero.
#[test]
// AUD-A-S6W1-001
fn zero_decay_scale_still_blocks_wind_inside_the_obstacle() {
    let sdf = wall();
    let mut tiny = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 10.0);
    tiny.decay_scale_m = 1.0e-30;
    assert_eq!(tiny.sample([-1.0, 0.0, 0.0])[0], 0.0);
    let mut zero = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 10.0);
    zero.decay_scale_m = 0.0;
    assert_eq!(zero.sample([-1.0, 0.0, 0.0])[0], 0.0);
}

/// A NaN distance from the SDF is turned into zero wind (`f32::max(NaN, 0.0)`
/// returns 0.0), so a broken SDF reads as still air instead of surfacing.
#[test]
// AUD-A-S6W1-002
fn nan_sdf_distance_is_not_masked_as_still_air() {
    let bad = ClosureSdf::new(|_x, _y, _z| f32::NAN, |_x, _y, _z| (0.0, 1.0, 0.0));
    let f = SdfWindField::new(&bad, [1.0, 0.0, 0.0], 10.0);
    let v = f.sample([0.0, 0.0, 0.0]);
    assert!(v[0].is_nan(), "got {}", v[0]);
}

/// Negative decay is documented nowhere; it takes the same branch as zero.
#[test]
fn negative_decay_scale_takes_the_full_wind_branch_outside() {
    let sdf = wall();
    let mut f = SdfWindField::new(&sdf, [1.0, 0.0, 0.0], 4.0);
    f.decay_scale_m = -3.0;
    assert!(close(f.sample([1.0, 0.0, 0.0])[0], 4.0, 1.0e-6));
}

/// On the obstacle surface (d = 0) with `decay_scale_m = 0` the ramp must not
/// divide 0 by 0: the result stays finite.
#[test]
fn zero_decay_scale_on_the_surface_is_finite() {
    let sdf = wall();
    let mut f = SdfWindField::new(&sdf, [1.0, 0.5, -0.25], 6.0);
    f.decay_scale_m = 0.0;
    let v = f.sample([0.0, 0.0, 0.0]);
    assert!(v.iter().all(|c| c.is_finite()), "{v:?}");
    // the ramp's limit at d = 0 is clamp(0 / scale) = 0 for every scale > 0
    assert_eq!(v, [0.0, 0.0, 0.0]);
}
