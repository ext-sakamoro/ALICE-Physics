//! Audit oracles for `fluid_netcode` (`FluidSnapshot`, `FluidDelta`).
//!
//! The wire layout (per component: `hi` as i64 little-endian then `lo` as u64
//! little-endian, 48 bytes per vector) and the FNV-1a 64-bit checksum are
//! re-derived here from their definitions, so the checks do not reuse the
//! module's own serialiser.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::fluid_netcode::{FluidDelta, FluidSnapshot};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn ulp() -> Fix128 {
    Fix128::from_raw(0, 1)
}

fn le_bytes(vs: &[Vec3Fix]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in vs {
        for c in [v.x, v.y, v.z] {
            out.extend_from_slice(&c.hi.to_le_bytes());
            out.extend_from_slice(&c.lo.to_le_bytes());
        }
    }
    out
}

/// FNV-1a 64: offset basis 0xcbf29ce484222325, prime 0x100000001b3.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn sample(n: usize) -> (Vec<Vec3Fix>, Vec<Vec3Fix>) {
    let p = (0..n)
        .map(|i| v3(i as f64 * 0.5, -(i as f64), 2.0 + i as f64 * 0.125))
        .collect();
    let v = (0..n).map(|i| v3(0.25 * i as f64, 1.0, -0.5)).collect();
    (p, v)
}

#[test]
fn serialized_layout_is_hi_then_lo_little_endian_per_component() {
    let (p, v) = sample(5);
    let s = FluidSnapshot::capture(&p, &v, 3);
    assert_eq!(s.particle_count, 5);
    assert_eq!(s.positions.len(), 5 * 3 * 16);
    assert_eq!(s.velocities.len(), 5 * 3 * 16);
    assert_eq!(s.positions, le_bytes(&p));
    assert_eq!(s.velocities, le_bytes(&v));
    assert_eq!(s.frame, 3);
}

#[test]
fn checksum_is_fnv1a_of_positions_then_velocities() {
    let (p, v) = sample(4);
    let s = FluidSnapshot::capture(&p, &v, 0);
    let mut all = le_bytes(&p);
    all.extend(le_bytes(&v));
    assert_eq!(s.checksum, fnv1a(&all));
    // known vector: FNV-1a of "a" is 0xaf63dc4c8601ec8c (sanity of the reference itself)
    assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
}

#[test]
fn checksum_depends_on_every_byte_of_the_state() {
    let (p, v) = sample(3);
    let base = FluidSnapshot::capture(&p, &v, 0).checksum;
    for i in 0..3 {
        for comp in 0..3 {
            let mut q = p.clone();
            match comp {
                0 => q[i].x = q[i].x + ulp(),
                1 => q[i].y = q[i].y + ulp(),
                _ => q[i].z = q[i].z + ulp(),
            }
            assert_ne!(
                FluidSnapshot::capture(&q, &v, 0).checksum,
                base,
                "pos {i}.{comp}"
            );
            let mut w = v.clone();
            match comp {
                0 => w[i].x = w[i].x + ulp(),
                1 => w[i].y = w[i].y + ulp(),
                _ => w[i].z = w[i].z + ulp(),
            }
            assert_ne!(
                FluidSnapshot::capture(&p, &w, 0).checksum,
                base,
                "vel {i}.{comp}"
            );
        }
    }
}

#[test]
fn checksum_does_not_depend_on_the_frame_number() {
    let (p, v) = sample(2);
    assert_eq!(
        FluidSnapshot::capture(&p, &v, 1).checksum,
        FluidSnapshot::capture(&p, &v, 9999).checksum
    );
}

#[test]
fn swapping_positions_and_velocities_changes_the_checksum() {
    let (p, v) = sample(3);
    assert_ne!(
        FluidSnapshot::capture(&p, &v, 0).checksum,
        FluidSnapshot::capture(&v, &p, 0).checksum
    );
}

#[test]
fn size_bytes_counts_count_buffers_checksum_and_frame() {
    for n in [0usize, 1, 7] {
        let (p, v) = sample(n);
        let s = FluidSnapshot::capture(&p, &v, 0);
        assert_eq!(s.size_bytes(), 4 + 96 * n + 8 + 8);
    }
}

#[test]
fn restore_returns_what_was_captured_bit_for_bit_including_extremes() {
    let p = vec![
        Vec3Fix::new(
            Fix128::from_raw(i64::MIN, 0),
            Fix128::from_raw(i64::MAX, u64::MAX),
            Fix128::from_raw(-1, 1),
        ),
        v3(0.1, -0.7, 1e9),
    ];
    let v = vec![Vec3Fix::ZERO, Vec3Fix::new(Fix128::PI, -Fix128::PI, ulp())];
    let (rp, rv) = FluidSnapshot::capture(&p, &v, 1).restore().unwrap();
    assert_eq!((rp, rv), (p, v));
}

#[test]
fn restore_with_a_larger_particle_count_than_the_buffers_is_refused() {
    let (p, v) = sample(3);
    let mut s = FluidSnapshot::capture(&p, &v, 0);
    s.particle_count = 4;
    assert!(s.restore().is_none());
    // a count of zero with data present restores nothing, without error
    s.particle_count = 0;
    let (a, b) = s.restore().unwrap();
    assert!(a.is_empty() && b.is_empty());
}

#[test]
fn restore_with_only_the_velocity_buffer_short_is_refused() {
    let (p, v) = sample(3);
    let mut s = FluidSnapshot::capture(&p, &v, 0);
    s.velocities.pop();
    assert!(s.restore().is_none());
}

#[test]
fn verify_is_true_only_for_the_captured_arrays() {
    let (p, v) = sample(4);
    let s = FluidSnapshot::capture(&p, &v, 0);
    assert!(s.verify(&p, &v));
    assert!(!s.verify(&p[..3], &v[..3]));
    assert!(!s.verify(&v, &p));
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-012: the checksum hashes positions then velocities as one byte stream with no length, so unequal-length inputs collide: capture(&[a,b], &[c]) and capture(&[a], &[b,c]) have the same checksum and verify() accepts the other split"]
fn checksum_distinguishes_where_the_position_stream_ends() {
    let a = v3(1.0, 2.0, 3.0);
    let b = v3(4.0, 5.0, 6.0);
    let c = v3(7.0, 8.0, 9.0);
    let s1 = FluidSnapshot::capture(&[a, b], &[c], 0);
    let s2 = FluidSnapshot::capture(&[a], &[b, c], 0);
    assert_ne!(s1.checksum, s2.checksum);
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-013: capture accepts position and velocity slices of different length; with fewer positions than velocities the surplus velocities are silently dropped on restore (capture(&[a], &[b, c]).restore() returns one velocity) while verify() still passes"]
fn capture_with_mismatched_lengths_must_not_silently_lose_velocities() {
    let a = v3(1.0, 0.0, 0.0);
    let b = v3(2.0, 0.0, 0.0);
    let c = v3(3.0, 0.0, 0.0);
    let s = FluidSnapshot::capture(&[a], &[b, c], 0);
    match s.restore() {
        None => {}
        Some((_, rv)) => assert_eq!(rv.len(), 2, "restored velocities {}", rv.len()),
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-014: restore() does not check the snapshot checksum, so a snapshot whose buffer was altered after capture (checksum unchanged) restores corrupted state as Some(..) (design decision: verify() is a separate call, module doc lists checksum validation as a purpose)"]
fn restore_rejects_a_snapshot_whose_buffer_no_longer_matches_its_checksum() {
    let (p, v) = sample(3);
    let mut s = FluidSnapshot::capture(&p, &v, 0);
    s.positions[0] ^= 0x01;
    assert!(s.restore().is_none());
}

// ----------------------------------------------------------------------------
// FluidDelta
// ----------------------------------------------------------------------------

fn delta(
    old: &(Vec<Vec3Fix>, Vec<Vec3Fix>),
    new: &(Vec<Vec3Fix>, Vec<Vec3Fix>),
    tau: Fix128,
) -> FluidDelta {
    FluidDelta::compute(&old.0, &old.1, &new.0, &new.1, tau, 4, 5)
}

#[test]
fn threshold_is_strict_and_per_component_for_positions() {
    let old = sample(1);
    let tau = fx(0.25);
    // every component moves by exactly tau: not shipped
    let mut new = old.clone();
    new.0[0] = old.0[0] + v3(0.25, -0.25, 0.25);
    assert_eq!(delta(&old, &new, tau).changed_count(), 0);
    // one component one ulp over: shipped
    new.0[0] = old.0[0] + Vec3Fix::new(Fix128::ZERO, -(tau + ulp()), Fix128::ZERO);
    assert_eq!(delta(&old, &new, tau).changed_count(), 1);
    // the norm (0.25*sqrt3 > tau) is irrelevant, only components count
    new.0[0] = old.0[0] + v3(0.25, 0.25, 0.25);
    assert_eq!(delta(&old, &new, tau).changed_count(), 0);
}

#[test]
fn velocity_only_change_is_shipped_with_both_arrays_for_that_particle() {
    let old = sample(3);
    let mut new = old.clone();
    new.1[1] = old.1[1] + v3(0.0, 0.0, 2.0);
    let d = delta(&old, &new, Fix128::ZERO);
    assert_eq!(d.changed_indices, vec![1]);
    assert_eq!(d.positions, vec![new.0[1]]);
    assert_eq!(d.velocities, vec![new.1[1]]);
}

#[test]
fn delta_metadata_and_checksum_describe_the_full_new_state() {
    let old = sample(4);
    let mut new = old.clone();
    new.0[2] = old.0[2] + v3(1.0, 0.0, 0.0);
    let d = delta(&old, &new, Fix128::ZERO);
    assert_eq!((d.base_frame, d.frame), (4, 5));
    let mut all = le_bytes(&new.0);
    all.extend(le_bytes(&new.1));
    assert_eq!(d.checksum, fnv1a(&all));
}

#[test]
fn apply_at_zero_threshold_reproduces_the_new_state_exactly() {
    let old = sample(9);
    let mut new = old.clone();
    for i in [0usize, 4, 8] {
        new.0[i] = new.0[i] + v3(0.1, 0.2, -0.3);
        new.1[i] = new.1[i] + Vec3Fix::new(ulp(), Fix128::ZERO, Fix128::ZERO);
    }
    let d = delta(&old, &new, Fix128::ZERO);
    let (mut bp, mut bv) = old.clone();
    d.apply(&mut bp, &mut bv);
    assert_eq!((bp, bv), new);
}

#[test]
fn apply_error_is_bounded_by_tau_per_component() {
    let old = sample(12);
    let tau = fx(0.125);
    let mut new = old.clone();
    for i in 0..12usize {
        let k = (i % 4) as f64 * 0.09; // 0, .09, .18, .27 : some below, some above tau
        new.0[i] = old.0[i] + v3(k, -k, k / 2.0);
    }
    let d = delta(&old, &new, tau);
    let (mut bp, mut bv) = old.clone();
    d.apply(&mut bp, &mut bv);
    for i in 0..12 {
        for (a, b) in [
            (bp[i].x, new.0[i].x),
            (bp[i].y, new.0[i].y),
            (bp[i].z, new.0[i].z),
        ] {
            assert!((a - b).abs() <= tau, "particle {i}");
        }
    }
    assert!(d.changed_count() > 0 && d.changed_count() < 12);
}

#[test]
fn apply_ignores_indices_beyond_the_base_arrays() {
    let d = FluidDelta {
        frame: 1,
        base_frame: 0,
        changed_indices: vec![0, 7],
        positions: vec![v3(1.0, 1.0, 1.0), v3(9.0, 9.0, 9.0)],
        velocities: vec![v3(2.0, 2.0, 2.0), v3(8.0, 8.0, 8.0)],
        checksum: 0,
    };
    let mut p = vec![Vec3Fix::ZERO; 2];
    let mut v = vec![Vec3Fix::ZERO; 2];
    d.apply(&mut p, &mut v);
    assert_eq!(p, vec![v3(1.0, 1.0, 1.0), Vec3Fix::ZERO]);
    assert_eq!(v, vec![v3(2.0, 2.0, 2.0), Vec3Fix::ZERO]);
}

#[test]
fn compression_ratio_is_changed_over_total() {
    let old = sample(8);
    let mut new = old.clone();
    for i in 0..2 {
        new.0[i] = old.0[i] + v3(1.0, 0.0, 0.0);
    }
    let d = delta(&old, &new, Fix128::ZERO);
    assert_eq!(d.changed_count(), 2);
    assert_eq!(d.compression_ratio(8), 0.25);
    assert_eq!(d.compression_ratio(2), 1.0);
    assert_eq!(d.compression_ratio(0), 1.0);
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-015: FluidDelta::compute indexes velocities with the position-array bound and panics (index out of bounds) when the velocity slices are shorter than the position slices instead of treating the particle sets as the common prefix"]
fn compute_with_shorter_velocity_slices_must_not_panic() {
    let old = sample(3);
    let new = sample(3);
    let r = catch_unwind(AssertUnwindSafe(|| {
        FluidDelta::compute(&old.0, &old.1[..1], &new.0, &new.1[..1], Fix128::ZERO, 0, 1)
    }));
    assert!(r.is_ok(), "panicked");
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-016: apply() bounds-checks only base_positions; a base_velocities slice shorter than base_positions panics (index out of bounds) after the positions were already overwritten"]
fn apply_with_a_short_velocity_base_must_not_panic() {
    let old = sample(3);
    let mut new = old.clone();
    new.0[2] = old.0[2] + v3(1.0, 0.0, 0.0);
    let d = delta(&old, &new, Fix128::ZERO);
    let mut p = old.0.clone();
    let mut v = old.1[..1].to_vec();
    let r = catch_unwind(AssertUnwindSafe(|| d.apply(&mut p, &mut v)));
    assert!(r.is_ok(), "panicked");
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-017: a position difference of exactly Fix128::MIN (hi = i64::MIN, lo = 0) has abs() == MIN (negative), so `abs() > threshold` is false and the change is not shipped at threshold 0; the receiver keeps the old value (extreme range only)"]
fn a_position_jump_of_the_most_negative_representable_difference_is_shipped() {
    let mut old = sample(1);
    old.0[0].x = Fix128::ZERO;
    let mut new = old.clone();
    new.0[0].x = Fix128::from_raw(i64::MIN, 0);
    let d = delta(&old, &new, Fix128::ZERO);
    assert_eq!(d.changed_count(), 1);
}

#[test]
fn apply_ignores_an_index_equal_to_the_base_length() {
    let d = FluidDelta {
        frame: 1,
        base_frame: 0,
        changed_indices: vec![2],
        positions: vec![v3(9.0, 9.0, 9.0)],
        velocities: vec![v3(8.0, 8.0, 8.0)],
        checksum: 0,
    };
    let mut p = vec![Vec3Fix::ZERO; 2];
    let mut v = vec![Vec3Fix::ZERO; 2];
    d.apply(&mut p, &mut v);
    assert_eq!(p, vec![Vec3Fix::ZERO; 2]);
}
