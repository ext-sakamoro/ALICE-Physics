//! Oracle: the overflow flag of a pair whose collision radii sum past the `Fix128`
//! range is decided by the pair alone, not by the broad-phase kind or the sleep skip.
//!
//! # What is measured
//!
//! The narrow-phase raises the sticky overflow flag (`PhysicsWorld::overflow_detected`,
//! one byte of `serialize_state`) when the sum of a pair's collision radii leaves the
//! `Fix128` range. Which pairs reach the narrow-phase depends on the broad-phase: the
//! Bvh hands over a superset of the overlapping boxes (quantised leaves), the dynamic
//! tree a superset by its margin, the hybrid broad-phase the overlapping boxes but
//! never two static bodies, and with the sleep skip on (Bvh only) a pair of two parked
//! bodies never reaches the narrow-phase. For a body with a collider the broad-phase
//! box is the collider's tight box clipped to the sphere cube, so a radius sum past the
//! range does not mean the boxes meet.
//!
//! The rule every world must follow: the flag reports range excess in arithmetic the
//! step performs. The radius sum is used for a pair whose broad-phase boxes overlap
//! (closed intervals) and that can interact: not filtered out, not two static bodies,
//! not two sleeping bodies. A box that itself leaves the range raises the flag where
//! it is built, independently of any pair.
//!
//! Every expectation here is written from that rule and the scene's coordinates (the
//! box of a unit ellipsoid at `x` is `x ± 1` on the x axis; a sphere cube is `x ± r`),
//! never from a run of the code. Each scene runs in 6 worlds (3 kinds × skip off / on),
//! asserts the flag value of the rule in every world and the same `serialize_state` in
//! all 6.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepState;
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

const WORLDS: [(Broadphase, bool); 6] = [
    (Broadphase::Bvh, false),
    (Broadphase::Bvh, true),
    (Broadphase::DynamicTree, false),
    (Broadphase::DynamicTree, true),
    (Broadphase::Hybrid, false),
    (Broadphase::Hybrid, true),
];

/// Gaps between the boxes `[x0 - 1, x0 + 1]` and `[x0 + 1 + gap, x0 + 3 + gap]`:
/// 0 is the closed boundary (the boxes touch), negative gaps overlap.
fn gaps() -> [Fix128; 6] {
    [
        Fix128::from_ratio(-1, 4),
        Fix128::ZERO,
        Fix128::from_ratio(1, 4),
        Fix128::from_ratio(3, 4),
        Fix128::from_ratio(5, 4),
        Fix128::from_int(3),
    ]
}

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

/// Two of these sum to `2^63`, past the range; a cube `x ± 2^62` stays inside it for
/// `|x| <= 2^62`.
fn huge() -> Fix128 {
    Fix128::from_int(1 << 62)
}

/// How a body is made
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// dynamic with a unit-sphere ellipsoid collider (box `x ± 1`)
    Collider,
    /// dynamic with an ellipsoid collider of radii (1, 0.9, 0.8) (box `x ± 1` on x)
    FlatCollider,
    /// dynamic without a collider (box `x ± r`)
    Plain,
    /// dynamic sensor without a collider
    Sensor,
    /// static with a unit-sphere ellipsoid collider
    StaticCollider,
    /// static without a collider
    StaticPlain,
}

#[derive(Clone, Copy, Debug)]
struct Body {
    kind: Kind,
    x: Fix128,
    radius: Fix128,
    asleep: bool,
    group: u32,
}

fn body(kind: Kind, x: Fix128, asleep: bool) -> Body {
    Body {
        kind,
        x,
        radius: huge(),
        asleep,
        group: 0,
    }
}

fn add(w: &mut PhysicsWorld, b: Body) -> usize {
    let radii = match b.kind {
        Kind::FlatCollider => Some(Vec3Fix::new(
            Fix128::ONE,
            Fix128::from_ratio(9, 10),
            Fix128::from_ratio(4, 5),
        )),
        Kind::Collider | Kind::StaticCollider => {
            Some(Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE))
        }
        Kind::Plain | Kind::Sensor | Kind::StaticPlain => None,
    };
    let mut rigid = match b.kind {
        Kind::StaticCollider | Kind::StaticPlain => RigidBody::new_static(x_at(b.x)),
        _ => RigidBody::new_dynamic(x_at(b.x), Fix128::ONE),
    };
    rigid.is_sensor = b.kind == Kind::Sensor;
    let i = w.add_body_with_radius(rigid, b.radius);
    if let Some(radii) = radii {
        assert!(w.set_body_shape(i, &Shape::Ellipsoid { radii }));
        w.set_body_collision_radius(i, b.radius);
    }
    if b.group != 0 {
        w.set_body_filter(
            i,
            CollisionFilter {
                group: b.group,
                ..CollisionFilter::DEFAULT
            },
        );
    }
    if b.asleep {
        w.islands.sleep_data[i].state = SleepState::Sleeping;
        w.islands.sleep_data[i].idle_frames = 100;
    }
    i
}

fn build(kind: Broadphase, skip: bool, bodies: &[Body]) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    for b in bodies {
        add(&mut w, *b);
    }
    w
}

fn stepped(kind: Broadphase, skip: bool, bodies: &[Body], steps: usize) -> PhysicsWorld {
    let mut w = build(kind, skip, bodies);
    for _ in 0..steps {
        w.step(Fix128::from_ratio(1, 60));
    }
    w
}

/// Every world of `bodies` after 3 steps: flag `want` and one `serialize_state`.
/// Returns the descriptions of the worlds that differ.
fn check(label: &str, bodies: &[Body], want: bool) -> Vec<String> {
    let reference = stepped(Broadphase::Bvh, false, bodies, 3).serialize_state();
    let mut differing = Vec::new();
    for (kind, skip) in WORLDS {
        let w = stepped(kind, skip, bodies, 3);
        if w.overflow_detected() != want || w.serialize_state() != reference {
            differing.extend([format!(
                "{label} {kind:?} skip={skip}: flag {} (want {want}), same state {}",
                w.overflow_detected(),
                w.serialize_state() == reference
            )]);
        }
    }
    differing
}

fn assert_none(what: &str, compared: usize, differing: &[String]) {
    assert!(compared > 0, "{what}: nothing compared");
    assert!(
        differing.is_empty(),
        "{what}: {} of {compared} worlds differ:\n{}",
        differing.len(),
        differing.join("\n")
    );
}

/// A gap sweep of the pair `a` at `x0` and `b` at `x0 + 2 + gap`, with the flag
/// expected exactly where `rule(gap)` holds.
fn sweep(what: &str, a: Body, b: Body, x0: Fix128, rule: impl Fn(Fix128) -> bool) {
    let mut differing = Vec::new();
    let mut compared = 0;
    let (mut raised, mut lowered) = (0, 0);
    for gap in gaps() {
        let want = rule(gap);
        if want {
            raised += 1;
        } else {
            lowered += 1;
        }
        let pair = [
            Body { x: x0, ..a },
            Body {
                x: x0 + Fix128::from_int(2) + gap,
                ..b
            },
        ];
        differing.extend(check(&format!("{what} gap={gap:?}"), &pair, want));
        compared += WORLDS.len();
    }
    assert_eq!(compared, 36, "{what}: 6 gaps x 6 worlds");
    assert!(
        raised > 0 && lowered > 0,
        "{what}: the sweep raises and lowers"
    );
    assert_none(what, compared, &differing);
}

fn overlap(gap: Fix128) -> bool {
    gap <= Fix128::ZERO
}

/// Oracle: two awake collider bodies. Absolute flags: gap -1/4 and 0 raise it (the
/// closed boxes meet), 1/4 and beyond do not, in every world.
#[test]
fn awake_collider_pair_raises_the_flag_exactly_when_the_boxes_meet() {
    let a = body(Kind::Collider, Fix128::ZERO, false);
    sweep("awake collider", a, a, Fix128::ZERO, overlap);
}

/// Oracle: as above with the second ellipsoid flattened to radii (1, 0.9, 0.8); its
/// box on x is still `x ± 1`.
#[test]
fn awake_flat_collider_pair_raises_the_flag_exactly_when_the_boxes_meet() {
    let a = body(Kind::Collider, Fix128::ZERO, false);
    let b = body(Kind::FlatCollider, Fix128::ZERO, false);
    sweep("awake flat collider", a, b, Fix128::ZERO, overlap);
}

/// Oracle: one collider body asleep, the other awake: the pair can interact, so the
/// flag follows the box overlap.
#[test]
fn one_sleeping_collider_pair_raises_the_flag_exactly_when_the_boxes_meet() {
    let a = body(Kind::Collider, Fix128::ZERO, true);
    let b = body(Kind::Collider, Fix128::ZERO, false);
    sweep("one sleeping collider", a, b, Fix128::ZERO, overlap);
}

/// Oracle: a static collider body and a sleeping dynamic collider body (parked with
/// the skip on): a static body is not asleep, so the pair can interact and the flag
/// follows the box overlap.
#[test]
fn static_and_sleeping_collider_pair_raises_the_flag_exactly_when_the_boxes_meet() {
    let a = body(Kind::StaticCollider, Fix128::ZERO, false);
    let b = body(Kind::Collider, Fix128::ZERO, true);
    sweep("static and sleeping collider", a, b, Fix128::ZERO, overlap);
}

/// Oracle: two sleeping collider bodies form no radius sum at any gap.
#[test]
fn sleeping_collider_pair_raises_no_flag() {
    let a = body(Kind::Collider, Fix128::ZERO, true);
    let mut differing = Vec::new();
    for gap in gaps() {
        let pair = [
            a,
            Body {
                x: Fix128::from_int(2) + gap,
                ..a
            },
        ];
        differing.extend(check(
            &format!("sleeping collider gap={gap:?}"),
            &pair,
            false,
        ));
    }
    assert_none("sleeping collider", 36, &differing);
}

/// Oracle: an awake plain body (cube `± 2^62`) next to an awake collider body: the cube
/// covers the collider box at every gap here, so the flag is raised at every gap.
#[test]
fn awake_plain_and_collider_pair_raises_the_flag_at_every_gap() {
    let a = body(Kind::Plain, Fix128::ZERO, false);
    let b = body(Kind::Collider, Fix128::ZERO, false);
    let mut differing = Vec::new();
    for gap in gaps() {
        let pair = [
            a,
            Body {
                x: Fix128::from_int(2) + gap,
                ..b
            },
        ];
        differing.extend(check(
            &format!("plain and collider gap={gap:?}"),
            &pair,
            true,
        ));
    }
    assert_none("plain and collider", 36, &differing);
}

/// Oracle: two static bodies never interact, with or without colliders, so their
/// radius sum raises the flag in no world, whether their boxes overlap or not.
#[test]
fn static_pair_raises_no_flag() {
    let mut differing = Vec::new();
    for kind in [Kind::StaticPlain, Kind::StaticCollider] {
        for gap in gaps() {
            let pair = [
                body(kind, Fix128::ZERO, false),
                body(kind, Fix128::from_int(2) + gap, false),
            ];
            differing.extend(check(&format!("{kind:?} pair gap={gap:?}"), &pair, false));
        }
    }
    assert_none("static pair", 72, &differing);
}

/// Oracle: two awake bodies in the same collision group (the filter rejects the pair)
/// with overlapping cubes raise no flag; the same pair without the group does.
#[test]
fn filtered_pair_raises_no_flag_and_the_same_pair_unfiltered_does() {
    let far = Fix128::from_int(1 << 40);
    let grouped = Body {
        group: 7,
        ..body(Kind::Plain, Fix128::ZERO, false)
    };
    let filtered = [grouped, Body { x: far, ..grouped }];
    let mut differing = check("filtered", &filtered, false);
    let open = [
        body(Kind::Plain, Fix128::ZERO, false),
        body(Kind::Plain, far, false),
    ];
    differing.extend(check("unfiltered", &open, true));
    assert_none("filtered pair", 12, &differing);
}

/// Oracle: two awake sensor bodies use the radius sum (the sensor path reports a
/// trigger from the same contact), so their overlapping cubes raise the flag.
#[test]
fn sensor_pair_raises_the_flag() {
    let far = Fix128::from_int(1 << 40);
    let pair = [
        body(Kind::Sensor, Fix128::ZERO, false),
        body(Kind::Sensor, far, false),
    ];
    assert_none("sensor pair", 6, &check("sensor", &pair, true));
}

/// Oracle: two sleeping plain bodies whose radius sum alone leaves the range (both
/// cubes inside it) raise no flag. With the second body's radius the largest value
/// (`Fix128::from_raw(i64::MAX, u64::MAX)`), its cube `2^40 + r` leaves the range: that
/// box raises the flag where it is built (staging, or the insert into the tree of
/// parked bodies), independently of the pair, in every world.
#[test]
fn sleeping_pair_raises_the_flag_only_through_a_box_that_leaves_the_range() {
    let far = Fix128::from_int(1 << 40);
    let sum_only = [
        body(Kind::Plain, Fix128::ZERO, true),
        body(Kind::Plain, far, true),
    ];
    let mut differing = check("sum only", &sum_only, false);
    let cube_out = [
        body(Kind::Plain, Fix128::ZERO, true),
        Body {
            radius: Fix128::from_raw(i64::MAX, u64::MAX),
            ..body(Kind::Plain, far, true)
        },
    ];
    differing.extend(check("cube out", &cube_out, true));
    assert_none("sleeping pair", 12, &differing);
}

/// Oracle: the sleeping pair whose radius sum leaves the range raises no flag while
/// both sleep and raises it on the step one of them is woken, in every world.
#[test]
fn a_sleeping_pair_raises_the_flag_on_the_step_it_wakes() {
    let far = Fix128::from_int(1 << 40);
    let pair = [
        body(Kind::Plain, Fix128::ZERO, true),
        body(Kind::Plain, far, true),
    ];
    let mut differing = Vec::new();
    let mut reference: Option<Vec<u8>> = None;
    for (kind, skip) in WORLDS {
        let mut w = stepped(kind, skip, &pair, 3);
        if w.overflow_detected() {
            differing.extend([format!("{kind:?} skip={skip}: flag raised before the wake")]);
        }
        w.wake_body(0);
        w.step(Fix128::from_ratio(1, 60));
        let state = w.serialize_state();
        let same = reference.get_or_insert_with(|| state.clone()) == &state;
        if !w.overflow_detected() || !same {
            differing.extend([format!(
                "{kind:?} skip={skip}: flag {} after the wake step, same state {same}",
                w.overflow_detected()
            )]);
        }
    }
    assert_none("wake", 6, &differing);
}

/// Oracle: the closed box test survives the hybrid broad-phase's clamped cell
/// coordinates. Ten small awake bodies near the origin (radius 1/16) make the median
/// box extent 1/8, so the hybrid grid's cell is below one unit and a coordinate near
/// `2^62` (raw `2^126`) shifted by fewer than 63 bits passes `i64::MAX` and is clamped.
/// The collider pair sits at `x0 = 2^62 - 2^61`, where its cubes (`± 2^62`) stay inside
/// the range, so no box leaves it and only the pair's radius sum can raise the flag:
/// gap -1/4 and 0 raise it, 1/4 and beyond do not, in every world.
#[test]
fn clamped_hybrid_cells_keep_every_closed_meeting_pair() {
    let x0 = Fix128::from_int((1 << 62) - (1 << 61));
    let small = |k: i64| Body {
        radius: Fix128::from_ratio(1, 16),
        ..body(Kind::Plain, Fix128::from_int(k), false)
    };
    let mut differing = Vec::new();
    let mut compared = 0;
    for gap in gaps() {
        let mut bodies: Vec<Body> = (0..10).map(|k| small(4 * k)).collect();
        bodies.extend([
            body(Kind::Collider, x0, false),
            body(Kind::Collider, x0 + Fix128::from_int(2) + gap, false),
        ]);
        differing.extend(check(
            &format!("clamped gap={gap:?}"),
            &bodies,
            overlap(gap),
        ));
        compared += WORLDS.len();
    }
    assert_none("clamped hybrid cells", compared, &differing);
}
