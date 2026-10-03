//! Rope pin constraints, body-following, length query, and SDF collision:
//! `Rope::{add_pin, pin_start, pin_end, update_pin_targets, current_length,
//! step_with_sdf}`. All six had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/rope.rs::*`, 6 items). This
//! is their production entry point.
//!
//! `step_with_sdf` has the exact same name on `Cloth`, `Deformable`, and
//! `Fluid` (`src/cloth.rs`, `src/deformable.rs`, `src/fluid.rs`) -- all four
//! are *independently* unwired as of this file; wiring `Rope::step_with_sdf`
//! here does not wire the other three (they are a different `self` type
//! each, with their own separate implementations and separate baseline
//! entries).
//!
//! # Closed forms (every expected value is derived here, independently of
//! the `Rope` method under test)
//!
//! * **`add_pin`**: appends exactly one entry to `pins` and zeroes
//!   `inv_masses[particle_index]`. Checked by comparing `pins.len()` and
//!   `inv_masses[i]` *before* and *after* the call (snapshot diff, not a
//!   call to `add_pin` itself).
//! * **`pin_start`/`pin_end`**: equivalent to `add_pin` with
//!   `target = positions[0]` / `positions[last]` *captured before the
//!   call*, `body_index: None`. Then, run under gravity for many frames:
//!   since `substep`'s pin-application step
//!   (`self.positions[pin.particle_index] = pin.target`) runs
//!   unconditionally every substep, and `solve_distance_constraints` never
//!   moves a particle whose `inv_mass` is zero, the pinned particle's
//!   position must stay *bit-exactly* equal to the captured target
//!   forever -- not merely "close".
//! * **`update_pin_targets`**: for a pin with `body_index = Some(idx)`,
//!   `target' = body_positions[idx] + body_rotations[idx].rotate_vec(local_offset)`.
//!   The 180-degree-about-Z identity `rotate_vec(v) = (-v.x, -v.y, v.z)`
//!   (hand-derived Hamilton product, reused verbatim from
//!   `examples/compound_shapes.rs` / `tests/analytic_multi_world_wiring.rs`)
//!   gives a second, non-identity, independently-checkable case. A pin
//!   with `body_index: None` must be left untouched by the call (the loop
//!   only visits `Some` entries).
//! * **`current_length`**: sum of `(positions[i+1] - positions[i]).length()`
//!   over all segments. Checked against a hand-built "staircase" of 3-4-5
//!   right-triangle displacement vectors (`(3,4,0)` / `(3,-4,0)`
//!   alternating), each of exact length 5 (`5*5 = 25`, a perfect square, so
//!   `Fix128::sqrt` -- itself an exact-floor digit recurrence, see
//!   `src/math.rs` doc comment on `Fix128::sqrt` -- returns exactly `5`,
//!   not merely "close to 5"). Expected total: `5 * num_segments`, summed
//!   here as plain integer arithmetic, never by calling `current_length`.
//! * **`step_with_sdf`**: driven against the same `y`-plane `ClosureSdf`
//!   already used as a *production* SDF elsewhere in this crate
//!   (`src/sdf_character.rs`, `src/sdf_ccd.rs`, `src/sdf_force.rs`:
//!   `ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))`), placed
//!   at the world origin with identity rotation and unit scale. With that
//!   placement, `SdfCollider::world_to_local` is the identity map, so the
//!   plane's local-space distance *is* the particle's world-space `y`
//!   coordinate -- a one-line closed form, not something borrowed from the
//!   function under test. A horizontal rope with `gravity = ZERO` and
//!   `substeps = 1`, built lying flat at `y = -0.5` (an exact value in both
//!   `f32` and `Fix128`), has every pairwise `delta` with `y = 0` exactly
//!   (identical `y` on every particle), so `solve_distance_constraints`'s
//!   correction vector (which is `delta` scaled) can only move particles
//!   along `x`; `y` is untouched by steps 1-3 of `substep`. Step 4
//!   (`resolve_sdf_collisions`) then pushes every particle from `y = -0.5`
//!   to exactly `y = -0.5 + 0.5 = 0` (`depth = Fix128::from_f32(0.5)`,
//!   `0.5` exact in both formats, `normal = (0, 1, 0)` exact for this
//!   plane). So after exactly one `step_with_sdf(dt, &[plane])` call, every
//!   *free* particle's `y` must be exactly `Fix128::ZERO` -- a bit-exact
//!   prediction, not a tolerance bound. A second scene (gravity restored to
//!   the default, a unit-sphere `ClosureSdf`) checks the weaker
//!   non-penetration invariant `sphere_dist(p) >= -tolerance` for every
//!   particle on every frame, mirroring the non-penetration style of
//!   `Cloth::step_with_sdf`'s own test (`src/cloth.rs`
//!   `step_with_sdf_drapes_cloth_over_unit_sphere_without_penetration`).
//!
//! ```bash
//! cargo run --example rope_pin_constraints --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rope::{PinConstraint, Rope};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

/// 180-degree rotation about the world Z axis, as an exact unit quaternion.
/// Reused verbatim from `examples/compound_shapes.rs` (same derivation,
/// independent of `rope.rs`): `rotate_vec(q, v) = (-v.x, -v.y, v.z)`.
const ROT_180_Z: QuatFix = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

fn rot_180_z(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(-v.x, -v.y, v.z)
}

/// Ground plane at `y = 0`, placed at the world origin with identity
/// rotation and unit scale -- the same closed-form plane already used as a
/// *production* SDF in `src/sdf_character.rs` / `src/sdf_ccd.rs` /
/// `src/sdf_force.rs`.
fn ground_plane() -> SdfCollider {
    let field = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// Unit sphere at the world origin (same closed form as `src/rope.rs`'s own
/// `unit_sphere_collider` test helper, rederived here independently since
/// this file must not call into that `#[cfg(test)]`-only helper).
fn unit_sphere() -> SdfCollider {
    let field = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            if len > 1e-6 {
                (x / len, y / len, z / len)
            } else {
                (0.0, 1.0, 0.0)
            }
        },
    );
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn sphere_dist(p: Vec3Fix) -> f32 {
    let (x, y, z) = p.to_f32();
    (x * x + y * y + z * z).sqrt() - 1.0
}

fn main() {
    // ------------------------------------------------------------------
    // add_pin: snapshot inv_masses/pins before and after; closed form is
    // "exactly one new pin, exactly one zeroed inv_mass, every other
    // particle's inv_mass unchanged".
    // ------------------------------------------------------------------
    let mut rope = Rope::new(
        Vec3Fix::ZERO,
        Vec3Fix::from_int(8, 0, 0),
        4,
        Fix128::from_ratio(1, 10),
    );
    let pins_before = rope.pins.len();
    let inv_mass_2_before = rope.inv_masses[2];
    let inv_mass_0_before = rope.inv_masses[0];
    assert!(
        !inv_mass_2_before.is_zero(),
        "particle 2 must start unpinned"
    );

    let pin_target = Vec3Fix::from_int(100, 200, 300);
    rope.add_pin(PinConstraint {
        particle_index: 2,
        target: pin_target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });

    assert_eq!(
        rope.pins.len(),
        pins_before + 1,
        "add_pin must append exactly one pin"
    );
    assert!(
        rope.inv_masses[2].is_zero(),
        "add_pin must zero the pinned particle's inv_mass"
    );
    assert_eq!(
        rope.inv_masses[0], inv_mass_0_before,
        "add_pin must not touch any other particle's inv_mass"
    );
    assert_eq!(rope.pins[pins_before].target, pin_target);
    assert_eq!(rope.pins[pins_before].particle_index, 2);
    println!(
        "[rope] add_pin(particle=2, target={:?}) -> pins.len()={} inv_masses[2]={:?} (expected 0)",
        pin_target.to_f32(),
        rope.pins.len(),
        rope.inv_masses[2].to_f32()
    );

    // ------------------------------------------------------------------
    // pin_start: target must be positions[0] *as captured before the
    // call*. Then under gravity for many steps, positions[0] must stay
    // bit-exactly equal to that captured target.
    // ------------------------------------------------------------------
    let mut rope_a = Rope::new(
        Vec3Fix::from_int(0, 5, 0),
        Vec3Fix::from_int(10, 5, 0),
        6,
        Fix128::ONE,
    );
    let start_captured = rope_a.positions[0];
    rope_a.pin_start();
    assert_eq!(rope_a.pins.len(), 1, "pin_start must add exactly one pin");
    assert_eq!(rope_a.pins[0].particle_index, 0);
    assert_eq!(rope_a.pins[0].target, start_captured);
    assert_eq!(rope_a.pins[0].body_index, None);
    assert!(rope_a.inv_masses[0].is_zero());

    rope_a.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-9), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..30 {
        rope_a.step(dt);
        assert_eq!(
            rope_a.positions[0], start_captured,
            "pin_start's target must hold particle 0 bit-exactly at frame {frame}"
        );
    }
    println!(
        "[rope] pin_start() -> target={:?}, held exactly across 30 gravity steps",
        start_captured.to_f32()
    );

    // ------------------------------------------------------------------
    // pin_end: same closed form, last particle.
    // ------------------------------------------------------------------
    let mut rope_b = Rope::new(
        Vec3Fix::from_int(0, 5, 0),
        Vec3Fix::from_int(10, 5, 0),
        6,
        Fix128::ONE,
    );
    let last_index = rope_b.particle_count() - 1;
    let end_captured = rope_b.positions[last_index];
    rope_b.pin_end();
    assert_eq!(rope_b.pins.len(), 1, "pin_end must add exactly one pin");
    assert_eq!(rope_b.pins[0].particle_index, last_index);
    assert_eq!(rope_b.pins[0].target, end_captured);
    assert!(rope_b.inv_masses[last_index].is_zero());

    rope_b.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-9), Fix128::ZERO);
    for frame in 0..30 {
        rope_b.step(dt);
        assert_eq!(
            rope_b.positions[last_index], end_captured,
            "pin_end's target must hold the last particle bit-exactly at frame {frame}"
        );
    }
    println!(
        "[rope] pin_end() -> particle={} target={:?}, held exactly across 30 gravity steps",
        last_index,
        end_captured.to_f32()
    );

    // ------------------------------------------------------------------
    // update_pin_targets: body-following pin tracks
    // body_pos + body_rot.rotate_vec(local_offset); a body_index=None pin
    // must be left untouched.
    // ------------------------------------------------------------------
    let mut rope_c = Rope::new(Vec3Fix::ZERO, Vec3Fix::from_int(4, 0, 0), 4, Fix128::ONE);
    let offset = Vec3Fix::from_int(1, 2, 3);
    rope_c.add_pin(PinConstraint {
        particle_index: 0,
        target: Vec3Fix::ZERO,
        body_index: Some(0),
        local_offset: offset,
    });
    let static_target = Vec3Fix::from_int(42, 42, 42);
    let static_index = rope_c.particle_count() - 1;
    rope_c.add_pin(PinConstraint {
        particle_index: static_index,
        target: static_target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });

    let body_pos = [Vec3Fix::from_int(10, 0, 0)];
    let body_rot_identity = [QuatFix::IDENTITY];
    rope_c.update_pin_targets(&body_pos, &body_rot_identity);
    let expected_identity = body_pos[0] + offset;
    assert_eq!(
        rope_c.pins[0].target, expected_identity,
        "update_pin_targets (identity rotation) must equal body_pos + local_offset"
    );
    assert_eq!(
        rope_c.pins[1].target, static_target,
        "a body_index=None pin must be untouched by update_pin_targets"
    );
    println!(
        "[rope] update_pin_targets(identity) -> pins[0].target={:?} (expected {:?}); pins[1] (body_index=None) unchanged at {:?}",
        rope_c.pins[0].target.to_f32(),
        expected_identity.to_f32(),
        rope_c.pins[1].target.to_f32()
    );

    let body_rot_180 = [ROT_180_Z];
    rope_c.update_pin_targets(&body_pos, &body_rot_180);
    let expected_180 = body_pos[0] + rot_180_z(offset);
    assert_eq!(
        rope_c.pins[0].target, expected_180,
        "update_pin_targets (180deg about Z) must equal body_pos + rot_180_z(local_offset)"
    );
    println!(
        "[rope] update_pin_targets(180deg about Z) -> pins[0].target={:?} (expected {:?})",
        rope_c.pins[0].target.to_f32(),
        expected_180.to_f32()
    );

    // ------------------------------------------------------------------
    // current_length: hand-built 3-4-5-triangle staircase, every segment
    // exactly length 5 (25 is a perfect square, so Fix128::sqrt's exact
    // digit recurrence returns exactly 5). Expected total: 5 * num_segments,
    // computed here with plain integer arithmetic.
    // ------------------------------------------------------------------
    let num_segments = 5usize;
    let mut rope_d = Rope::new(
        Vec3Fix::ZERO,
        Vec3Fix::from_int(num_segments as i64, 0, 0),
        num_segments,
        Fix128::ONE,
    );
    let mut staircase = Vec::with_capacity(num_segments + 1);
    let mut cursor = Vec3Fix::ZERO;
    staircase.push(cursor);
    for i in 0..num_segments {
        let step = if i % 2 == 0 {
            Vec3Fix::from_int(3, 4, 0)
        } else {
            Vec3Fix::from_int(3, -4, 0)
        };
        cursor = cursor + step;
        staircase.push(cursor);
    }
    assert_eq!(staircase.len(), rope_d.positions.len());
    rope_d.positions = staircase;

    let expected_length = Fix128::from_int(5 * num_segments as i64);
    let got_length = rope_d.current_length();
    assert_eq!(
        got_length, expected_length,
        "current_length over a 3-4-5-triangle staircase must equal 5*num_segments exactly"
    );
    println!(
        "[rope] current_length() over {num_segments} 3-4-5 segments = {:?} (expected {:?})",
        got_length.to_f32(),
        expected_length.to_f32()
    );

    // ------------------------------------------------------------------
    // step_with_sdf (exact plane case): gravity=ZERO, substeps=1, flat
    // horizontal rope at y=-0.5. Closed form: every free particle's y must
    // become exactly 0 after one call (see module doc comment above for
    // the full derivation of why steps 1-3 of substep cannot touch y).
    // ------------------------------------------------------------------
    let mut rope_e = Rope::new(
        Vec3Fix::from_f32(-2.0, -0.5, 0.0),
        Vec3Fix::from_f32(2.0, -0.5, 0.0),
        4,
        Fix128::ONE,
    );
    rope_e.config.gravity = Vec3Fix::ZERO;
    rope_e.config.substeps = 1;
    let plane = [ground_plane()];

    for p in &rope_e.positions {
        assert_eq!(
            p.y,
            Fix128::from_f32(-0.5),
            "fixture must start exactly at y=-0.5"
        );
    }
    rope_e.step_with_sdf(dt, &plane);
    for (i, p) in rope_e.positions.iter().enumerate() {
        assert_eq!(
            p.y,
            Fix128::ZERO,
            "particle {i} must land exactly on y=0 after one step_with_sdf against the plane"
        );
    }
    println!(
        "[rope] step_with_sdf(plane, gravity=0, 1 substep): all {} particles moved from y=-0.5 to exactly y=0",
        rope_e.particle_count()
    );

    // ------------------------------------------------------------------
    // step_with_sdf (non-penetration invariant, sphere + gravity):
    // mirrors Cloth::step_with_sdf's own non-penetration test style.
    // sphere_dist is a one-line closed form independent of the function
    // under test; the invariant checked is "never penetrates", not an
    // exact value.
    // ------------------------------------------------------------------
    let mut rope_f = Rope::new(
        Vec3Fix::from_int(-1, 2, 0),
        Vec3Fix::from_int(1, 2, 0),
        8,
        Fix128::ONE,
    );
    let sphere = [unit_sphere()];
    let mut min_dist = f32::MAX;
    for frame in 0..60 {
        rope_f.step_with_sdf(dt, &sphere);
        for (i, p) in rope_f.positions.iter().enumerate() {
            let d = sphere_dist(*p);
            min_dist = min_dist.min(d);
            assert!(
                d >= -1e-3,
                "frame {frame} particle {i} penetrated the unit sphere: dist {d}"
            );
        }
    }
    println!(
        "[rope] step_with_sdf(unit sphere, gravity, 60 frames): no particle penetrated (min dist {min_dist:.4})"
    );

    println!("[rope] all 6 wiring items (add_pin, pin_start, pin_end, update_pin_targets, current_length, step_with_sdf) exercised and matched hand-derived closed forms");
}
