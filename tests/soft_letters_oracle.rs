//! Oracle: soft letters — a soft body built from a glyph outline, living inside
//! [`PhysicsWorld`] and inside its state digest.
//!
//! # What is measured
//!
//! A glyph outline (planar polygons: outer contours and holes, even-odd fill) becomes
//! a lattice of cloth particles at a chosen spacing, tethered to its rest shape. The
//! body is stepped by the world, can be grabbed at several points, thrown by releasing a
//! grab, recovers its shape and falls asleep at the exact fixed point. Every test here
//! compares bits, not tolerances, unless a geometric bound is the law being tested.
//!
//! The scripts are in the `x–z` plane with `y` the normal: the letters are flat, as on a
//! page.

use alice_physics_core::cloth::{Cloth, ClothConfig, GrabId, OutlineDesc, OutlineError};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, SolverConfig, WorldSnapshotError};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt() -> Fix128 {
    r(1, 60)
}

/// A block letter E in the unit square: spine `x ∈ [0, 0.2]`, three arms 0.2 thick.
fn letter_e() -> Vec<Vec<[Fix128; 2]>> {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    vec![vec![
        p(0, 0),
        p(10, 0),
        p(10, 2),
        p(2, 2),
        p(2, 4),
        p(10, 4),
        p(10, 6),
        p(2, 6),
        p(2, 8),
        p(10, 8),
        p(10, 10),
        p(0, 10),
    ]]
}

/// A ring O: the unit square with the hole `[0.3, 0.7]²`.
fn letter_o() -> Vec<Vec<[Fix128; 2]>> {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    vec![
        vec![p(0, 0), p(10, 0), p(10, 10), p(0, 10)],
        vec![p(3, 3), p(3, 7), p(7, 7), p(7, 3)],
    ]
}

/// A dotted i: stem `[0.4, 0.6] × [0, 0.6]` and dot `[0.4, 0.6] × [0.8, 1.0]`.
fn letter_i() -> Vec<Vec<[Fix128; 2]>> {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    vec![
        vec![p(4, 0), p(6, 0), p(6, 6), p(4, 6)],
        vec![p(4, 8), p(6, 8), p(6, 10), p(4, 10)],
    ]
}

fn config() -> ClothConfig {
    ClothConfig {
        iterations: 4,
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: r(985, 1000),
        stretch_compliance: r(1, 100),
        bend_compliance: r(1, 4),
        ..ClothConfig::default()
    }
}

/// The site's mochi: `ω = 2π / 0.5`, `ζ = 0.42`, unit mass.
fn tether() -> (Fix128, Fix128) {
    let omega = Fix128::from_f32(core::f32::consts::TAU / 0.5);
    (omega * omega, omega * Fix128::from_f32(2.0 * 0.42))
}

fn desc(contours: Vec<Vec<[Fix128; 2]>>, x_offset: Fix128, spacing: Fix128) -> OutlineDesc {
    OutlineDesc {
        origin: Vec3Fix::new(x_offset, Fix128::ZERO, Fix128::ZERO),
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours,
        spacing,
        mass_per_particle: Fix128::ONE,
        config: config(),
        tether: Some(tether()),
    }
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    })
}

/// E at the origin and O at `x = 1.5`, spacing 0.1.
fn two_letters() -> PhysicsWorld {
    let mut w = world();
    w.add_cloth(Cloth::from_outline(&desc(letter_e(), Fix128::ZERO, r(1, 10))).expect("E"));
    w.add_cloth(Cloth::from_outline(&desc(letter_o(), r(15, 10), r(1, 10))).expect("O"));
    w
}

fn at(x: f32, z: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, 0.0, z)
}

fn step(w: &mut PhysicsWorld, n: usize) {
    for _ in 0..n {
        w.step(dt());
    }
}

type State = (Vec<Vec3Fix>, Vec<Vec3Fix>, Vec<Vec3Fix>);

fn state(w: &PhysicsWorld, b: usize) -> State {
    let c = &w.cloths[b];
    (
        c.positions.clone(),
        c.prev_positions.clone(),
        c.velocities.clone(),
    )
}

fn dist(a: Vec3Fix, b: Vec3Fix) -> f32 {
    (a - b).length().to_f32()
}

fn max_dev(w: &PhysicsWorld, b: usize) -> f32 {
    let c = &w.cloths[b];
    c.positions
        .iter()
        .zip(c.rest_positions())
        .map(|(p, r)| dist(*p, *r))
        .fold(0.0, f32::max)
}

/// Connected components of the triangle graph.
fn components(n: usize, triangles: &[[usize; 3]]) -> usize {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for t in triangles {
        for k in 1..3 {
            let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[k]));
            if a != b {
                parent[a] = b;
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}

/// Distance from a point to a closed polygon's boundary (f32, for bounds only).
fn boundary_distance(p: [f32; 2], contours: &[Vec<[Fix128; 2]>]) -> f32 {
    let mut best = f32::MAX;
    for c in contours {
        let n = c.len();
        for i in 0..n {
            let a = [c[i][0].to_f32(), c[i][1].to_f32()];
            let b = [c[(i + 1) % n][0].to_f32(), c[(i + 1) % n][1].to_f32()];
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len2 = ex * ex + ey * ey;
            let t = (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2).clamp(0.0, 1.0);
            let (qx, qy) = (a[0] + t * ex, a[1] + t * ey);
            let (dx, dy) = (p[0] - qx, p[1] - qy);
            best = best.min((dx * dx + dy * dy).sqrt());
        }
    }
    best
}

/// Even-odd fill test (f32, for bounds only).
fn inside(p: [f32; 2], contours: &[Vec<[Fix128; 2]>]) -> bool {
    let mut odd = false;
    for c in contours {
        let n = c.len();
        for i in 0..n {
            let a = [c[i][0].to_f32(), c[i][1].to_f32()];
            let b = [c[(i + 1) % n][0].to_f32(), c[(i + 1) % n][1].to_f32()];
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                if p[0] < x {
                    odd = !odd;
                }
            }
        }
    }
    odd
}

// ---------------------------------------------------------------------------
// O1: outline → masked lattice
// ---------------------------------------------------------------------------

/// oracle: every particle of the E is inside the fill or within spacing/2 of its
/// outline, the E is one connected piece, and two builds agree bit for bit.
#[test]
fn outline_e_is_one_connected_lattice_inside_the_fill() {
    let w = two_letters();
    let e = &w.cloths[0];
    let contours = letter_e();
    let n = e.particle_count();
    assert!(n >= 40, "too few particles for a 0.1 lattice on the E: {n}");
    for p in e.positions.iter() {
        let q = [p.x.to_f32(), p.z.to_f32()];
        assert_eq!(p.y, Fix128::ZERO);
        assert!(
            inside(q, &contours) || boundary_distance(q, &contours) <= 0.05 + 1e-6,
            "particle {q:?} is outside the E"
        );
    }
    assert_eq!(
        components(n, e.triangles.as_slice()),
        1,
        "E is not one piece"
    );
    assert_eq!(e.rest_positions(), e.positions.as_slice());
    let w2 = two_letters();
    assert_eq!(w2.cloths[0].positions, e.positions);
    assert_eq!(w2.cloths[0].triangles, e.triangles);
}

/// oracle: the O keeps its hole (no particle deeper than spacing/2 inside it) and is
/// one ring; the dotted i is two pieces in one body.
#[test]
fn outline_hole_is_empty_and_pieces_are_counted() {
    let w = two_letters();
    let o = &w.cloths[1];
    for p in o.positions.iter() {
        let (x, z) = (p.x.to_f32() - 1.5, p.z.to_f32());
        assert!(
            !(x > 0.35 && x < 0.65 && z > 0.35 && z < 0.65),
            "particle inside the hole: ({x}, {z})"
        );
    }
    assert_eq!(
        components(o.particle_count(), &o.triangles),
        1,
        "O is not one ring"
    );

    let mut w = world();
    let i = w.add_cloth(Cloth::from_outline(&desc(letter_i(), Fix128::ZERO, r(1, 10))).expect("i"));
    let body = &w.cloths[i];
    assert_eq!(
        components(body.particle_count(), &body.triangles),
        2,
        "i is stem + dot"
    );
}

/// oracle: an outline too small for the spacing yields no triangle and is refused,
/// never a silent empty body.
#[test]
fn outline_smaller_than_the_spacing_is_refused() {
    let p = |x: i64, y: i64| [r(x, 1000), r(y, 1000)];
    let tiny = vec![vec![p(0, 0), p(10, 0), p(10, 10), p(0, 10)]];
    let w = world();
    let err = Cloth::from_outline(&desc(tiny, Fix128::ZERO, Fix128::ONE)).unwrap_err();
    assert_eq!(err, OutlineError::NoTriangles);
    assert!(w.cloths.is_empty());
    let err = Cloth::from_outline(&desc(letter_e(), Fix128::ZERO, Fix128::ZERO)).unwrap_err();
    assert_eq!(err, OutlineError::SpacingNotPositive);
    let err = Cloth::from_outline(&desc(Vec::new(), Fix128::ZERO, r(1, 10))).unwrap_err();
    assert_eq!(err, OutlineError::EmptyOutline);
}

// ---------------------------------------------------------------------------
// O2 / O9: rest, plane
// ---------------------------------------------------------------------------

/// oracle: at rest nothing moves, bit for bit, and the body is asleep after the first
/// step (the step was the identity).
#[test]
fn rest_is_still_and_asleep() {
    let mut w = two_letters();
    let before = (state(&w, 0), state(&w, 1));
    assert!(!w.cloths[0].is_asleep());
    step(&mut w, 1);
    assert!(w.cloths[0].is_asleep() && w.cloths[1].is_asleep());
    step(&mut w, 239);
    assert_eq!((state(&w, 0), state(&w, 1)), before);
}

// ---------------------------------------------------------------------------
// O3 / O4: grabs
// ---------------------------------------------------------------------------

/// oracle: a grabbed particle follows the pointer, the letter stretches locally (the
/// neighbour follows more than the far corner), and the other letter is untouched.
#[test]
fn grab_stretches_the_letter_locally() {
    let mut w = two_letters();
    let rest0 = w.cloths[0].positions.clone();
    let o_before = state(&w, 1);
    let id = w
        .cloth_grab(0, at(0.0, 0.0), r(34, 100))
        .expect("corner within reach");
    let g = w.cloths[0].grab_particle(id).expect("grab is listed");
    assert_eq!(rest0[g], at(0.0, 0.0));
    let target = at(-0.9, -0.5);
    for _ in 0..90 {
        assert!(w.cloth_move_grab(id, target));
        step(&mut w, 1);
    }
    let p = w.cloths[0].positions.clone();
    assert_eq!(p[g], target, "grabbed particle is held at the pointer");
    let far = (0..rest0.len())
        .max_by(|&a, &b| dist(rest0[a], rest0[g]).total_cmp(&dist(rest0[b], rest0[g])))
        .unwrap();
    let near = (0..rest0.len())
        .filter(|&i| i != g)
        .min_by(|&a, &b| dist(rest0[a], rest0[g]).total_cmp(&dist(rest0[b], rest0[g])))
        .unwrap();
    let (near_move, far_move) = (dist(p[near], rest0[near]), dist(p[far], rest0[far]));
    assert!(near_move > 0.3, "neighbour did not follow: {near_move}");
    assert!(
        far_move < near_move * 0.6,
        "letter moved rigidly: far {far_move} near {near_move}"
    );
    assert_eq!(state(&w, 1), o_before, "the O moved");
    assert!(!w.cloths[0].is_asleep());
}

/// oracle: two grabs on one letter hold two particles; releasing one keeps the other;
/// a grab cannot take a particle already held (it takes the next nearest) and fails
/// out of reach or on a missing body.
#[test]
fn two_grabs_hold_two_particles() {
    let mut w = two_letters();
    let a = w.cloth_grab(0, at(0.0, 0.0), r(34, 100)).expect("a");
    let b = w
        .cloth_grab(0, at(0.0, 0.0), r(34, 100))
        .expect("b: next nearest");
    let (pa, pb) = (
        w.cloths[0].grab_particle(a).unwrap(),
        w.cloths[0].grab_particle(b).unwrap(),
    );
    assert_ne!(pa, pb);
    assert_eq!(w.cloths[0].grabs().len(), 2);
    let (ta, tb) = (at(-0.5, -0.5), at(0.5, -0.5));
    for _ in 0..30 {
        w.cloth_move_grab(a, ta);
        w.cloth_move_grab(b, tb);
        step(&mut w, 1);
    }
    let c = &w.cloths[0];
    assert_eq!(c.positions[pa], ta);
    assert_eq!(c.positions[pb], tb);
    assert!(w.cloth_release(a));
    assert!(!w.cloth_release(a), "released twice");
    step(&mut w, 30);
    let c = &w.cloths[0];
    assert_eq!(c.positions[pb], tb, "b is still held");
    assert_ne!(c.positions[pa], ta, "a is free again");
    assert_eq!(c.inv_masses[pa], Fix128::ONE, "a's mass is back");
    assert_eq!(c.inv_masses[pb], Fix128::ZERO);
    assert_eq!(
        w.cloth_grab(0, at(50.0, 50.0), r(34, 100)),
        None,
        "out of reach"
    );
    assert_eq!(
        w.cloth_grab(7, at(0.0, 0.0), r(34, 100)),
        None,
        "no such body"
    );
    assert!(!w.cloth_move_grab(GrabId(999), ta));
}

// ---------------------------------------------------------------------------
// O5: throw
// ---------------------------------------------------------------------------

/// oracle: releasing a grab gives the particle the pointer's velocity over the last
/// step, `(target applied at the last step − target applied at the step before) / dt`,
/// bit for bit; a grab released before any step throws nothing; the explicit form sets
/// exactly the given velocity.
#[test]
fn release_throws_with_the_pointers_velocity() {
    let mut w = two_letters();
    let id = w.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    let g = w.cloths[0].grab_particle(id).unwrap();
    assert!(w.cloth_release(id));
    assert_eq!(
        w.cloths[0].velocities[g],
        Vec3Fix::ZERO,
        "no step: no throw"
    );

    let id = w.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    let g = w.cloths[0].grab_particle(id).unwrap();
    let d = Vec3Fix::new(r(3, 100), Fix128::ZERO, r(-2, 100));
    let mut target = at(0.0, 0.0);
    for _ in 0..10 {
        target = target + d;
        w.cloth_move_grab(id, target);
        step(&mut w, 1);
    }
    assert!(w.cloth_release(id));
    let v = w.cloths[0].velocities[g];
    assert_eq!(
        v,
        d / dt(),
        "throw velocity is the pointer's last-step velocity"
    );
    assert_eq!(w.cloths[0].inv_masses[g], Fix128::ONE);
    // the same script released without a throw: the thrown particle is ahead of it
    // along the throw after one step (the tether and the neighbours pull both back)
    let mut still = two_letters();
    let id2 = still.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    let mut t2 = at(0.0, 0.0);
    for _ in 0..10 {
        t2 = t2 + d;
        still.cloth_move_grab(id2, t2);
        step(&mut still, 1);
    }
    assert!(still.cloth_release_with_velocity(id2, Vec3Fix::ZERO));
    step(&mut w, 1);
    step(&mut still, 1);
    let (p1, q1) = (w.cloths[0].positions[g], still.cloths[0].positions[g]);
    assert!(
        p1.x > q1.x && p1.z < q1.z,
        "the throw did not carry the particle: {p1:?} vs {q1:?}"
    );

    let id = w.cloth_grab(1, at(1.5, 0.0), r(34, 100)).unwrap();
    let g = w.cloths[1].grab_particle(id).unwrap();
    step(&mut w, 3);
    let v = Vec3Fix::from_f32(40.0, 0.0, -25.0);
    assert!(w.cloth_release_with_velocity(id, v));
    assert_eq!(w.cloths[1].velocities[g], v);
}

// ---------------------------------------------------------------------------
// O6 / O7 / O9: recovery, sleep
// ---------------------------------------------------------------------------

fn throw_script(w: &mut PhysicsWorld) {
    let id = w.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    for _ in 0..30 {
        w.cloth_move_grab(id, at(-0.9, -0.5));
        step(w, 1);
    }
    w.cloth_release_with_velocity(id, Vec3Fix::from_f32(40.0, 0.0, -25.0));
}

/// oracle: after a hard throw the letter comes back to its rest shape, reaches the
/// exact fixed point within 600 frames and sleeps there; a sleeping step changes no
/// bit; a grab wakes it; every position stays in the plane throughout.
#[test]
fn thrown_letter_recovers_sleeps_and_wakes() {
    let mut w = two_letters();
    throw_script(&mut w);
    assert!(max_dev(&w, 0) > 0.5, "the throw did not deform the letter");
    let mut fixed_at = None;
    let mut prev = state(&w, 0);
    for f in 1..=600u32 {
        step(&mut w, 1);
        let s = state(&w, 0);
        assert!(
            s.0.iter().all(|p| p.y == Fix128::ZERO),
            "left the plane at frame {f}"
        );
        if s == prev {
            fixed_at = Some(f);
            break;
        }
        prev = s;
    }
    let fixed_at = fixed_at.expect("no exact fixed point within 600 frames");
    assert!(
        w.cloths[0].is_asleep(),
        "not asleep at the fixed point (frame {fixed_at})"
    );
    assert!(
        max_dev(&w, 0) < 1e-3,
        "did not return to rest: {}",
        max_dev(&w, 0)
    );
    let s = state(&w, 0);
    step(&mut w, 60);
    assert_eq!(state(&w, 0), s, "a sleeping step changed a bit");
    assert!(w.cloths[0].is_asleep());
    let id = w.cloth_grab(0, at(1.0, 1.0), r(34, 100)).unwrap();
    w.cloth_move_grab(id, at(1.3, 1.2));
    step(&mut w, 1);
    assert!(!w.cloths[0].is_asleep(), "a grab did not wake the letter");
    assert_ne!(state(&w, 0), s);
}

/// oracle: sleeping is invisible to the law — with sleep off the same script gives the
/// same bits every frame and the same digest.
#[test]
fn sleep_changes_no_bit_and_no_digest() {
    let mut a = two_letters();
    let mut b = two_letters();
    b.cloths[0].set_sleep(false);
    b.cloths[1].set_sleep(false);
    assert!(a.cloths[0].sleep_enabled() && !b.cloths[0].sleep_enabled());
    throw_script(&mut a);
    throw_script(&mut b);
    for f in 0..600 {
        step(&mut a, 1);
        step(&mut b, 1);
        assert_eq!(state(&a, 0), state(&b, 0), "frame {f}");
    }
    assert!(a.cloths[0].is_asleep() && !b.cloths[0].is_asleep());
    assert_eq!(a.state_digest(), b.state_digest());
}

// ---------------------------------------------------------------------------
// O8 / O10 / O11: digest, snapshot, rollback blob
// ---------------------------------------------------------------------------

/// oracle: the state digest covers the letters — it changes when a letter is grabbed
/// and moved, two worlds built the same agree, and a snapshot restores to the same
/// digest and the same bits 100 frames later.
#[test]
fn digest_covers_cloths_and_snapshot_round_trips() {
    let mut a = two_letters();
    let empty = world().state_digest();
    assert_ne!(a.state_digest(), empty, "letters are not in the digest");
    assert_eq!(a.state_digest(), two_letters().state_digest());
    let d0 = a.state_digest();
    let id = a.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    a.cloth_move_grab(id, at(-0.3, -0.2));
    step(&mut a, 5);
    assert_ne!(
        a.state_digest(),
        d0,
        "a grabbed letter did not change the digest"
    );

    let blob = a.snapshot_world();
    let mut b = world();
    b.restore_world(&blob).expect("restore");
    assert_eq!(b.state_digest(), a.state_digest());
    assert_eq!(b.cloths.len(), 2);
    assert_eq!(b.cloths[0].grabs().len(), 1, "the grab is state");
    for f in 0..100 {
        a.cloth_move_grab(id, at(-0.3, -0.2));
        b.cloth_move_grab(id, at(-0.3, -0.2));
        step(&mut a, 1);
        step(&mut b, 1);
        assert_eq!(state(&a, 0), state(&b, 0), "frame {f}");
    }
    assert!(a.cloth_release(id) && b.cloth_release(id));
    step(&mut a, 600);
    step(&mut b, 600);
    assert_eq!(a.state_digest(), b.state_digest());
    assert!(a.cloths[0].is_asleep() && b.cloths[0].is_asleep());
}

/// Frozen digest of the throw script at frame 600 (E and O, spacing 0.1). Every CI OS
/// and the wasm lane must agree with it; it moves only with an announced law change.
const THROW_DIGEST_600: &str = "303aeac1e640bb64fc26460550d4851ea84a4b905127683ccb3147d52528bb8e";

/// oracle: the throw script's digest at frame 600 is the frozen value on every target.
#[test]
fn throw_script_digest_is_frozen() {
    let mut w = two_letters();
    throw_script(&mut w);
    step(&mut w, 600);
    let hex: String = w
        .state_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, THROW_DIGEST_600);
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    h
}

/// Rewrite a snapshot: set its version, replace its payload, recompute the checksum.
fn reframe(blob: &[u8], version: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = blob[..80].to_vec();
    out[4..6].copy_from_slice(&version.to_le_bytes());
    out[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    let h = fnv1a(&out);
    out.extend_from_slice(&h.to_le_bytes());
    out
}

/// oracle: a blob of the version before soft bodies (no section) restores with no
/// letters; a blob cut inside the soft-body section is refused as truncated.
#[test]
fn previous_version_has_no_letters_and_a_cut_section_is_refused() {
    let w = world();
    let blob = w.snapshot_world();
    let payload = &blob[80..blob.len() - 8];
    // the section of an empty world is its count (u64 = 0) and the next grab id
    // (u32 = 0) at the payload's end
    assert_eq!(&payload[payload.len() - 12..], &[0u8; 12]);
    let old = reframe(
        &blob,
        PhysicsWorld::WORLD_SNAPSHOT_VERSION - 1,
        &payload[..payload.len() - 12],
    );
    let mut t = two_letters();
    t.restore_world(&old).expect("previous version restores");
    assert!(
        t.cloths.is_empty(),
        "a previous-version blob carries no letters"
    );
    assert_eq!(t.state_digest(), world().state_digest());

    let w = two_letters();
    let blob = w.snapshot_world();
    let payload = &blob[80..blob.len() - 8];
    let cut = reframe(
        &blob,
        PhysicsWorld::WORLD_SNAPSHOT_VERSION,
        &payload[..payload.len() - 16],
    );
    let mut t = world();
    assert_eq!(t.restore_world(&cut), Err(WorldSnapshotError::Truncated));
}

/// oracle: the rollback blob (`serialize_state`) carries the letters' motion state: it
/// restores into a world of the same letters bit for bit and the two worlds then agree
/// for 50 frames; a world of other letters refuses it.
#[test]
fn rollback_blob_carries_the_letters() {
    let mut a = two_letters();
    let id = a.cloth_grab(0, at(0.0, 0.0), r(34, 100)).unwrap();
    for _ in 0..20 {
        a.cloth_move_grab(id, at(-0.6, -0.4));
        step(&mut a, 1);
    }
    a.cloth_release(id);
    step(&mut a, 7);
    let blob = a.serialize_state();
    let mut b = two_letters();
    assert!(b.deserialize_state(&blob), "same letters: accepted");
    assert_eq!(state(&b, 0), state(&a, 0));
    assert_eq!(state(&b, 1), state(&a, 1));
    assert_eq!(b.cloths[0].inv_masses, a.cloths[0].inv_masses);
    for f in 0..50 {
        step(&mut a, 1);
        step(&mut b, 1);
        assert_eq!(state(&a, 0), state(&b, 0), "frame {f}");
    }
    let mut other = world();
    other.add_cloth(Cloth::from_outline(&desc(letter_i(), Fix128::ZERO, r(1, 10))).unwrap());
    assert!(!other.deserialize_state(&blob), "other letters: refused");
    assert!(!world().deserialize_state(&blob), "no letters: refused");
    // the same letters with another tether: same particle counts, other structure
    let mut stiffer = world();
    let (k, c) = tether();
    let mut e = desc(letter_e(), Fix128::ZERO, r(1, 10));
    e.tether = Some((k + k, c));
    stiffer.add_cloth(Cloth::from_outline(&e).unwrap());
    stiffer.add_cloth(Cloth::from_outline(&desc(letter_o(), r(15, 10), r(1, 10))).unwrap());
    assert_eq!(
        stiffer.cloths[0].particle_count(),
        a.cloths[0].particle_count()
    );
    let before = state(&stiffer, 0);
    assert!(
        !stiffer.deserialize_state(&blob),
        "another structure: refused"
    );
    assert_eq!(
        state(&stiffer, 0),
        before,
        "a refused blob changed the letters"
    );
}
