//! Audit oracles for `sim_modifier`.
//!
//! Closed forms (independent of the crate):
//!
//! * a chain applies its modifiers in push order: `d -> m_n(... m_1(d))`;
//! * for the unit-sphere field `d = |p| - 1` and a modifier `d' = d + c`, the
//!   zero set moves to radius `1 - c` ("positive offset = surface recedes");
//! * the normal of a modified field is the unit gradient of the modified
//!   distance; for `d' = |p| - 1 + k x` it is `(p / |p| + (k, 0, 0))`
//!   normalised.
#![allow(
    clippy::disallowed_methods,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
use std::sync::{Arc, Mutex};

fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

struct Affine {
    scale: f32,
    offset: f32,
    active: bool,
    log: Arc<Mutex<Vec<f32>>>,
}

impl Affine {
    fn new(scale: f32, offset: f32) -> Self {
        Self {
            scale,
            offset,
            active: true,
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl PhysicsModifier for Affine {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d * self.scale + self.offset
    }
    fn update(&mut self, dt: f32) {
        self.log.lock().unwrap().push(dt);
    }
    fn name(&self) -> &str {
        "affine"
    }
    fn is_active(&self) -> bool {
        self.active
    }
}

struct Ramp {
    k: f32,
}
impl PhysicsModifier for Ramp {
    fn modify_distance(&self, x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d + self.k * x
    }
    fn update(&mut self, _dt: f32) {}
    fn name(&self) -> &str {
        "ramp"
    }
}

struct Recorder {
    seen: Arc<Mutex<Vec<(f32, f32, f32, f32)>>>,
}
impl PhysicsModifier for Recorder {
    fn modify_distance(&self, x: f32, y: f32, z: f32, d: f32) -> f32 {
        self.seen.lock().unwrap().push((x, y, z, d));
        d + 10.0
    }
    fn update(&mut self, _dt: f32) {}
    fn name(&self) -> &str {
        "rec"
    }
}

#[test]
fn chain_applies_modifiers_in_push_order() {
    // y = 3: (3*2)+1 = 7 when scale comes first; (3+1)*2 = 8 when offset first.
    let a = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(Affine::new(2.0, 0.0)))
        .with_modifier(Box::new(Affine::new(1.0, 1.0)));
    assert_eq!(a.distance(0.0, 3.0, 0.0), 7.0);
    let b = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(Affine::new(1.0, 1.0)))
        .with_modifier(Box::new(Affine::new(2.0, 0.0)));
    assert_eq!(b.distance(0.0, 3.0, 0.0), 8.0);
}

#[test]
fn each_modifier_receives_the_point_and_the_distance_handed_down_the_chain() {
    let seen1 = Arc::new(Mutex::new(Vec::new()));
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let m = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(Recorder {
            seen: seen1.clone(),
        }))
        .with_modifier(Box::new(Recorder {
            seen: seen2.clone(),
        }));
    let d = m.distance(1.5, 2.5, -3.5);
    assert_eq!(d, 22.5);
    assert_eq!(seen1.lock().unwrap().as_slice(), &[(1.5, 2.5, -3.5, 2.5)]);
    assert_eq!(seen2.lock().unwrap().as_slice(), &[(1.5, 2.5, -3.5, 12.5)]);
}

#[test]
fn positive_offset_makes_the_surface_recede() {
    let m =
        ModifiedSdf::new(Box::new(unit_sphere())).with_modifier(Box::new(Affine::new(1.0, 0.1)));
    assert!(
        m.distance(0.9, 0.0, 0.0).abs() < 1e-6,
        "zero set at radius 0.9"
    );
    assert!(
        m.distance(0.95, 0.0, 0.0) > 0.0,
        "formerly inside now outside"
    );
    let g =
        ModifiedSdf::new(Box::new(unit_sphere())).with_modifier(Box::new(Affine::new(1.0, -0.1)));
    assert!(
        g.distance(1.1, 0.0, 0.0).abs() < 1e-6,
        "negative offset expands to 1.1"
    );
}

#[test]
fn inactive_modifier_is_skipped_by_the_chain() {
    let mut m = Affine::new(1.0, 100.0);
    m.active = false;
    let chain = ModifiedSdf::new(Box::new(ground())).with_modifier(Box::new(m));
    assert_eq!(chain.distance(0.0, 3.0, 0.0), 3.0);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-007: SingleModifiedSdf ignores PhysicsModifier::is_active (an inactive +100 modifier still shifts the distance 3 -> 103; the chain wrapper returns 3)"]
fn inactive_modifier_is_skipped_by_the_single_wrapper() {
    let mut m = Affine::new(1.0, 100.0);
    m.active = false;
    let single = SingleModifiedSdf::new(Box::new(ground()), m);
    assert_eq!(single.distance(0.0, 3.0, 0.0), 3.0);
}

#[test]
fn update_reaches_every_modifier_once_with_the_given_dt_even_if_inactive() {
    let la = Affine::new(1.0, 0.0);
    let log_a = la.log.clone();
    let mut lb = Affine::new(1.0, 0.0);
    lb.active = false;
    let log_b = lb.log.clone();
    let mut chain = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(la))
        .with_modifier(Box::new(lb));
    chain.update(0.25);
    chain.update(0.5);
    assert_eq!(log_a.lock().unwrap().as_slice(), &[0.25, 0.5]);
    assert_eq!(log_b.lock().unwrap().as_slice(), &[0.25, 0.5]);
}

#[test]
fn single_wrapper_forwards_update_and_exposes_the_modifier() {
    let m = Affine::new(1.0, 0.0);
    let log = m.log.clone();
    let mut s = SingleModifiedSdf::new(Box::new(ground()), m);
    s.update(0.125);
    assert_eq!(log.lock().unwrap().as_slice(), &[0.125]);
    s.modifier.offset = 4.0;
    assert_eq!(s.distance(0.0, 1.0, 0.0), 5.0);
}

fn ramp_normal_expected(p: [f64; 3], k: f64) -> [f64; 3] {
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let g = [p[0] / l + k, p[1] / l, p[2] / l];
    let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    [g[0] / gl, g[1] / gl, g[2] / gl]
}

#[test]
fn normal_is_the_unit_gradient_of_the_modified_distance() {
    let k = 0.3_f32;
    let chain = ModifiedSdf::new(Box::new(unit_sphere())).with_modifier(Box::new(Ramp { k }));
    let single = SingleModifiedSdf::new(Box::new(unit_sphere()), Ramp { k });
    for p in [
        [1.0_f32, 0.5, -0.25],
        [-0.7, 1.2, 0.9],
        [0.1, -0.4, 1.7],
        [2.0, 2.0, 2.0],
    ] {
        let want = ramp_normal_expected(
            [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])],
            f64::from(k),
        );
        for (name, n) in [
            ("chain", chain.normal(p[0], p[1], p[2])),
            ("single", single.normal(p[0], p[1], p[2])),
        ] {
            let len = f64::from(n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "{name} not unit: {len}");
            assert!((f64::from(n.0) - want[0]).abs() < 2e-3, "{name} nx {p:?}");
            assert!((f64::from(n.1) - want[1]).abs() < 2e-3, "{name} ny {p:?}");
            assert!((f64::from(n.2) - want[2]).abs() < 2e-3, "{name} nz {p:?}");
        }
    }
}

#[test]
fn distance_and_normal_agree_with_the_separate_queries() {
    let chain = ModifiedSdf::new(Box::new(unit_sphere())).with_modifier(Box::new(Ramp { k: 0.3 }));
    let single = SingleModifiedSdf::new(Box::new(unit_sphere()), Ramp { k: 0.3 });
    let p = (0.8_f32, -0.6, 1.1);
    assert_eq!(
        chain.distance_and_normal(p.0, p.1, p.2),
        (chain.distance(p.0, p.1, p.2), chain.normal(p.0, p.1, p.2))
    );
    assert_eq!(
        single.distance_and_normal(p.0, p.1, p.2),
        (single.distance(p.0, p.1, p.2), single.normal(p.0, p.1, p.2))
    );
}

#[test]
fn normal_of_an_unmodified_chain_is_the_original_outward_normal() {
    let chain = ModifiedSdf::new(Box::new(unit_sphere()));
    let n = chain.normal(0.0, 2.0, 0.0);
    assert!(
        n.0.abs() < 1e-3 && (n.1 - 1.0).abs() < 1e-3 && n.2.abs() < 1e-3,
        "{n:?}"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-008: the normal's finite-difference step is the absolute constant 0.001, so far from the origin the f32 step is lost (unit sphere at (1e4,5e3,0): normal (1,0,0) instead of (0.894,0.447,0); at (1e5,0,0) it is (0,1,0))"]
fn normal_is_accurate_far_from_the_origin() {
    let chain =
        ModifiedSdf::new(Box::new(unit_sphere())).with_modifier(Box::new(Affine::new(1.0, 0.0)));
    let n = chain.normal(1.0e4, 5.0e3, 0.0);
    let s5 = 5.0_f32.sqrt();
    assert!(
        (n.0 - 2.0 / s5).abs() < 1e-2 && (n.1 - 1.0 / s5).abs() < 1e-2 && n.2.abs() < 1e-2,
        "{n:?}"
    );
}

#[test]
fn wrappers_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<ModifiedSdf>();
    check::<SingleModifiedSdf<Affine>>();
}

#[test]
fn modifier_mut_gives_access_to_the_named_modifier() {
    let mut chain = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(Affine::new(1.0, 0.0)))
        .with_modifier(Box::new(Ramp { k: 0.0 }));
    assert_eq!(
        chain
            .modifier_mut(0)
            .map(|m| m.name().to_string())
            .as_deref(),
        Some("affine")
    );
    assert_eq!(
        chain
            .modifier_mut(1)
            .map(|m| m.name().to_string())
            .as_deref(),
        Some("ramp")
    );
    assert!(chain.modifier_mut(2).is_none());
}

struct Cubic {
    c: f32,
}
impl PhysicsModifier for Cubic {
    fn modify_distance(&self, x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d + self.c * x * x * x
    }
    fn update(&mut self, _dt: f32) {}
    fn name(&self) -> &str {
        "cubic"
    }
}

#[test]
fn normal_finite_difference_step_is_small_enough_for_a_curved_field() {
    // d = y + c x^3: exact gradient (3 c x^2, 1, 0); a central difference of step e adds c e^2 to
    // the x component, so only a small step reproduces the exact direction to 2e-4.
    let c = 100.0_f32;
    let x = 0.1_f32;
    let gx = 3.0 * f64::from(c) * f64::from(x) * f64::from(x);
    let l = (gx * gx + 1.0).sqrt();
    let chain = ModifiedSdf::new(Box::new(ground())).with_modifier(Box::new(Cubic { c }));
    let single = SingleModifiedSdf::new(Box::new(ground()), Cubic { c });
    for (name, n) in [
        ("chain", chain.normal(x, 0.0, 0.0)),
        ("single", single.normal(x, 0.0, 0.0)),
    ] {
        assert!(
            (f64::from(n.0) - gx / l).abs() < 2e-4,
            "{name} nx {} vs {}",
            n.0,
            gx / l
        );
        assert!((f64::from(n.1) - 1.0 / l).abs() < 2e-4, "{name} ny");
    }
}

#[test]
fn normal_direction_does_not_depend_on_a_small_slope_magnitude() {
    // d = 0.1 x: gradient length 0.1, direction +X.
    let gentle = ClosureSdf::new(|x, _y, _z| 0.1 * x, |_x, _y, _z| (1.0, 0.0, 0.0));
    let chain = ModifiedSdf::new(Box::new(gentle)).with_modifier(Box::new(Affine::new(1.0, 0.0)));
    let n = chain.normal(0.3, 0.2, 0.1);
    assert!(
        (n.0 - 1.0).abs() < 1e-3 && n.1.abs() < 1e-3 && n.2.abs() < 1e-3,
        "{n:?}"
    );
    let gentle2 = ClosureSdf::new(|x, _y, _z| 0.1 * x, |_x, _y, _z| (1.0, 0.0, 0.0));
    let single = SingleModifiedSdf::new(Box::new(gentle2), Affine::new(1.0, 0.0));
    let n = single.normal(0.3, 0.2, 0.1);
    assert!(
        (n.0 - 1.0).abs() < 1e-3 && n.1.abs() < 1e-3 && n.2.abs() < 1e-3,
        "{n:?}"
    );
}

#[test]
fn normal_of_a_constant_field_is_the_up_axis_fallback() {
    let flat = ClosureSdf::new(|_x, _y, _z| 5.0, |_x, _y, _z| (0.0, 0.0, 1.0));
    let chain = ModifiedSdf::new(Box::new(flat)).with_modifier(Box::new(Affine::new(1.0, 0.0)));
    assert_eq!(chain.normal(1.0, 2.0, 3.0), (0.0, 1.0, 0.0));
    let flat2 = ClosureSdf::new(|_x, _y, _z| 5.0, |_x, _y, _z| (0.0, 0.0, 1.0));
    let single = SingleModifiedSdf::new(Box::new(flat2), Affine::new(1.0, 0.0));
    assert_eq!(single.normal(1.0, 2.0, 3.0), (0.0, 1.0, 0.0));
}

#[test]
fn add_modifier_appends_to_the_end_of_the_chain() {
    // scale 2 first then +1: y = 3 -> 7; the reverse order would give 8.
    let mut chain = ModifiedSdf::new(Box::new(ground()));
    chain.add_modifier(Box::new(Affine::new(2.0, 0.0)));
    chain.add_modifier(Box::new(Affine::new(1.0, 1.0)));
    assert_eq!(chain.distance(0.0, 3.0, 0.0), 7.0);
}

#[test]
fn modifier_count_counts_installed_modifiers_including_inactive_ones() {
    // The doc says "Number of active modifiers" but the count is the chain length: an inactive
    // modifier still counts (AUD-A-S5W3-021, documentation wording).
    let mut inactive = Affine::new(1.0, 0.0);
    inactive.active = false;
    let chain = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(Affine::new(1.0, 0.0)))
        .with_modifier(Box::new(inactive));
    assert_eq!(chain.modifier_count(), 2);
}
