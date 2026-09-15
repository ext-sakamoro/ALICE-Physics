//! Continuous Collision Detection (CCD)
//!
//! Prevents fast-moving objects from tunneling through thin geometry.
//!
//! # Algorithms
//!
//! - **Conservative Advancement**: Iteratively advance along trajectory
//! - **Sphere-Sphere TOI**: Exact time of impact for two moving spheres
//! - **Sphere-Plane TOI**: Exact time of impact for sphere vs infinite plane

use crate::collider::AABB;
use crate::math::{Fix128, Vec3Fix};

/// Time of Impact result
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TOI {
    /// Time of first impact [0, 1], where 0=start, 1=end of timestep
    pub t: Fix128,
    /// Contact point at time of impact (world space)
    pub point: Vec3Fix,
    /// Contact normal at time of impact
    pub normal: Vec3Fix,
}

/// CCD configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CcdConfig {
    /// Maximum iterations for conservative advancement
    pub max_iterations: usize,
    /// Convergence threshold (stop when remaining gap < this)
    pub tolerance: Fix128,
    /// Minimum velocity magnitude to trigger CCD
    pub velocity_threshold: Fix128,
}

impl Default for CcdConfig {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            tolerance: Fix128::from_ratio(1, 1000),
            velocity_threshold: Fix128::from_int(5),
        }
    }
}

/// Check if a body needs CCD based on its velocity
#[inline]
#[must_use]
pub fn needs_ccd(velocity: Vec3Fix, radius: Fix128, dt: Fix128, config: &CcdConfig) -> bool {
    let speed = velocity.length();
    // CCD needed if displacement > half the body's radius
    speed * dt > radius.half() && speed > config.velocity_threshold
}

/// Exact TOI for two moving spheres
///
/// Sphere A moves from `center_a` by `vel_a * t`
/// Sphere B moves from `center_b` by `vel_b * t`
/// Returns t in [0, 1] or None if no collision in this timestep.
#[must_use]
pub fn sphere_sphere_toi(
    center_a: Vec3Fix,
    radius_a: Fix128,
    vel_a: Vec3Fix,
    center_b: Vec3Fix,
    radius_b: Fix128,
    vel_b: Vec3Fix,
) -> Option<TOI> {
    let rel_pos = center_b - center_a;
    let rel_vel = vel_b - vel_a;
    let combined_r = radius_a + radius_b;

    // Quadratic: |rel_pos + rel_vel * t|^2 = combined_r^2
    let a = rel_vel.dot(rel_vel);
    let b = rel_pos.dot(rel_vel).double();
    let c = rel_pos.dot(rel_pos) - combined_r * combined_r;

    // Already overlapping
    if c <= Fix128::ZERO {
        let normal = rel_pos.normalize();
        return Some(TOI {
            t: Fix128::ZERO,
            point: center_a + normal * radius_a,
            normal,
        });
    }

    // No relative motion
    if a.is_zero() {
        return None;
    }

    let discriminant = b * b - Fix128::from_int(4) * a * c;
    if discriminant < Fix128::ZERO {
        return None;
    }

    let sqrt_d = discriminant.sqrt();
    let two_a = a.double();
    let t = (-b - sqrt_d) / two_a;

    if t >= Fix128::ZERO && t <= Fix128::ONE {
        let pos_a = center_a + vel_a * t;
        let pos_b = center_b + vel_b * t;
        let normal = (pos_b - pos_a).normalize();
        let point = pos_a + normal * radius_a;
        Some(TOI { t, point, normal })
    } else {
        None
    }
}

/// Exact TOI for a moving sphere against a static plane
///
/// Plane: dot(normal, p) = offset. Two-sided: a sphere on the back side
/// (`dist < −r`) moving towards the plane hits it at
/// `t = (−r − dist) / v·n`, and the returned normal points towards the
/// sphere's side (`−n` from behind). Before 1.2.0 the back side used the
/// front-side formula and reported the *exit* time (when the far surface
/// crossed the plane) with the front normal.
#[must_use]
pub fn sphere_plane_toi(
    center: Vec3Fix,
    radius: Fix128,
    velocity: Vec3Fix,
    plane_normal: Vec3Fix,
    plane_offset: Fix128,
) -> Option<TOI> {
    let dist = center.dot(plane_normal) - plane_offset;
    let vel_toward = velocity.dot(plane_normal);

    // Moving away from plane
    if vel_toward >= Fix128::ZERO && dist > radius {
        return None;
    }

    // +1 in front of the plane, −1 behind: the contact normal and the
    // touching surface point are on the sphere's side
    let side = if dist.is_negative() {
        -Fix128::ONE
    } else {
        Fix128::ONE
    };

    // Already penetrating
    if dist.abs() <= radius {
        let point = center - plane_normal * dist;
        return Some(TOI {
            t: Fix128::ZERO,
            point,
            normal: plane_normal * side,
        });
    }

    if vel_toward.is_zero() {
        return None;
    }

    // t when the near surface of the sphere touches the plane
    let t = (side * radius - dist) / vel_toward;

    if t >= Fix128::ZERO && t <= Fix128::ONE {
        let point = center + velocity * t - plane_normal * (side * radius);
        Some(TOI {
            t,
            point,
            normal: plane_normal * side,
        })
    } else {
        None
    }
}

/// Conservative advancement for convex shapes
///
/// Iteratively advances along trajectory, using distance function
/// to determine safe step sizes.
pub fn conservative_advancement<F>(
    start_pos: Vec3Fix,
    displacement: Vec3Fix,
    radius: Fix128,
    distance_fn: F,
    config: &CcdConfig,
) -> Option<TOI>
where
    F: Fn(Vec3Fix) -> (Fix128, Vec3Fix), // (signed_distance, normal)
{
    let mut t = Fix128::ZERO;

    for _ in 0..config.max_iterations {
        let pos = start_pos + displacement * t;
        let (dist, normal) = distance_fn(pos);

        // Account for sphere radius
        let gap = dist - radius;

        if gap <= config.tolerance {
            let point = pos - normal * dist;
            return Some(TOI { t, point, normal });
        }

        // Safe advance: we can move at most `gap` along the trajectory
        let speed = displacement.length();
        if speed.is_zero() {
            return None;
        }

        let dt = gap / speed;
        t = t + dt;

        if t > Fix128::ONE {
            return None; // No collision this timestep
        }
    }

    None
}

/// Swept AABB test (moving AABB vs static AABB)
///
/// Returns t in [0, 1] for first overlap, or None.
#[must_use]
pub fn swept_aabb(moving: &AABB, velocity: Vec3Fix, target: &AABB) -> Option<Fix128> {
    let mut t_enter = Fix128::from_int(-1000000);
    let mut t_exit = Fix128::from_int(1000000);

    // X axis
    if let Some((te, tx)) = slab_test(
        moving.min.x,
        moving.max.x,
        target.min.x,
        target.max.x,
        velocity.x,
    ) {
        t_enter = if te > t_enter { te } else { t_enter };
        t_exit = if tx < t_exit { tx } else { t_exit };
        if t_enter > t_exit {
            return None;
        }
    } else {
        return None;
    }

    // Y axis
    if let Some((te, tx)) = slab_test(
        moving.min.y,
        moving.max.y,
        target.min.y,
        target.max.y,
        velocity.y,
    ) {
        t_enter = if te > t_enter { te } else { t_enter };
        t_exit = if tx < t_exit { tx } else { t_exit };
        if t_enter > t_exit {
            return None;
        }
    } else {
        return None;
    }

    // Z axis
    if let Some((te, tx)) = slab_test(
        moving.min.z,
        moving.max.z,
        target.min.z,
        target.max.z,
        velocity.z,
    ) {
        t_enter = if te > t_enter { te } else { t_enter };
        t_exit = if tx < t_exit { tx } else { t_exit };
        if t_enter > t_exit {
            return None;
        }
    } else {
        return None;
    }

    if t_enter >= Fix128::ZERO && t_enter <= Fix128::ONE {
        Some(t_enter)
    } else if t_enter < Fix128::ZERO && t_exit >= Fix128::ZERO {
        Some(Fix128::ZERO) // Already overlapping
    } else {
        None
    }
}

/// Slab test for one axis of swept AABB
fn slab_test(
    a_min: Fix128,
    a_max: Fix128,
    b_min: Fix128,
    b_max: Fix128,
    vel: Fix128,
) -> Option<(Fix128, Fix128)> {
    if vel.is_zero() {
        // Static on this axis - check overlap
        if a_max < b_min || a_min > b_max {
            return None;
        }
        return Some((Fix128::from_int(-1000000), Fix128::from_int(1000000)));
    }

    let inv_vel = Fix128::ONE / vel;
    let mut t0 = (b_min - a_max) * inv_vel;
    let mut t1 = (b_max - a_min) * inv_vel;

    if t0 > t1 {
        core::mem::swap(&mut t0, &mut t1);
    }

    Some((t0, t1))
}

/// Speculative contact for CCD integration with the solver
///
/// Instead of rewinding time to TOI, creates a contact constraint
/// at the predicted collision point with a negative depth (gap).
/// Adaptive TOI-aware sub-step count for a pair of moving bodies
/// (Phase F 11.1 CCD 統合 skeleton).
///
/// Combines the sub-stepping policy from
/// `crate::solver_tgs::adaptive_substeps_for_ccd` (crate-internal since
/// v0.14.0-preview.8) with the
/// speculative contact TOI so that fast-moving pairs receive extra
/// sub-steps proportional to their closing speed. This prevents
/// tunneling through thin walls (see
/// `deterministic-physics-lockstep-discipline` skill §11.1 CCD).
///
/// # Status
/// Skeleton — the TOI-derived scaling policy is scheduled for the
/// follow-up commit. The current implementation returns a
/// conservative constant so downstream integrations can start
/// compiling against the stable signature.
///
/// # Determinism
/// Skill §1 経路 2 — no floating-point comparison, closed-form
/// clamp, deterministic sub-step count.
#[must_use]
// 1.0.0 で公開済の signature (crates.io)、引数 struct 化は semver major = 2.0 で実施
#[allow(clippy::too_many_arguments)]
pub fn adaptive_toi_substeps(
    pos_a: Vec3Fix,
    vel_a: Vec3Fix,
    radius_a: Fix128,
    pos_b: Vec3Fix,
    vel_b: Vec3Fix,
    radius_b: Fix128,
    dt: Fix128,
    max_substeps: u32,
) -> u32 {
    // Body of Phase F 11.1: use the existing sphere-swept
    // `speculative_contact` as an oracle for TOI prediction. When the
    // pair is on a collision course over `dt`, we return
    // `max_substeps` to give the caller headroom for TOI-aware CCD
    // handling; otherwise we return `1` so a single sub-step is used.
    //
    // # Determinism
    // - `speculative_contact` is a pure Fix128 function of its inputs.
    // - No floating-point comparison, no CORDIC / rounding on the
    //   branch predicate.
    // - `max_substeps` and the constant `1` are compile-time integers.
    // - The clamp `min(max_substeps)` matches the skeleton contract.
    match speculative_contact(pos_a, vel_a, radius_a, pos_b, vel_b, radius_b, dt) {
        Some(_) => max_substeps.max(1),
        None => 1,
    }
}

/// The solver then prevents penetration by maintaining the gap.
#[must_use]
pub fn speculative_contact(
    pos_a: Vec3Fix,
    vel_a: Vec3Fix,
    radius_a: Fix128,
    pos_b: Vec3Fix,
    vel_b: Vec3Fix,
    radius_b: Fix128,
    dt: Fix128,
) -> Option<crate::collider::Contact> {
    let rel_pos = pos_b - pos_a;
    let dist = rel_pos.length();
    let combined_r = radius_a + radius_b;

    if dist.is_zero() {
        return None;
    }

    let normal = rel_pos / dist;
    let gap = dist - combined_r;

    // Already overlapping — regular contact
    if gap <= Fix128::ZERO {
        return Some(crate::collider::Contact {
            depth: -gap,
            normal,
            point_a: pos_a + normal * radius_a,
            point_b: pos_b - normal * radius_b,
        });
    }

    // Check if closing velocity will breach the gap within dt
    let rel_vel = vel_b - vel_a;
    let closing_speed = -rel_vel.dot(normal);

    if closing_speed <= Fix128::ZERO {
        return None; // Moving apart
    }

    let predicted_gap = gap - closing_speed * dt;

    if predicted_gap < Fix128::ZERO {
        // Speculative contact: depth = predicted penetration
        Some(crate::collider::Contact {
            depth: -predicted_gap,
            normal,
            point_a: pos_a + normal * radius_a,
            point_b: pos_b - normal * radius_b,
        })
    } else {
        None
    }
}

/// TOI for a moving capsule against a static plane
///
/// Capsule defined by two endpoints and radius, moving with given velocity.
/// Plane in Hessian normal form: dot(normal, p) = offset.
#[must_use]
pub fn capsule_plane_toi(
    cap_a: Vec3Fix,
    cap_b: Vec3Fix,
    radius: Fix128,
    velocity: Vec3Fix,
    plane_normal: Vec3Fix,
    plane_offset: Fix128,
) -> Option<TOI> {
    // Closest capsule endpoint to the plane determines first contact
    let dist_a = cap_a.dot(plane_normal) - plane_offset;
    let dist_b = cap_b.dot(plane_normal) - plane_offset;
    // Use the endpoint closer to the plane
    let (center, _dist) = if dist_a < dist_b {
        (cap_a, dist_a)
    } else {
        (cap_b, dist_b)
    };
    sphere_plane_toi(center, radius, velocity, plane_normal, plane_offset)
}

/// TOI for a moving sphere against a static capsule
///
/// Reduces to sphere-sphere TOI against the closest point on capsule segment.
#[must_use]
pub fn sphere_capsule_toi(
    sphere_center: Vec3Fix,
    sphere_radius: Fix128,
    sphere_vel: Vec3Fix,
    cap_a: Vec3Fix,
    cap_b: Vec3Fix,
    cap_radius: Fix128,
) -> Option<TOI> {
    // Find closest point on capsule segment to sphere center
    let ab = cap_b - cap_a;
    let ab_len_sq = ab.dot(ab);
    let closest = if ab_len_sq.is_zero() {
        cap_a
    } else {
        let t = (sphere_center - cap_a).dot(ab) / ab_len_sq;
        let t_clamped = if t < Fix128::ZERO {
            Fix128::ZERO
        } else if t > Fix128::ONE {
            Fix128::ONE
        } else {
            t
        };
        cap_a + ab * t_clamped
    };
    sphere_sphere_toi(
        sphere_center,
        sphere_radius,
        sphere_vel,
        closest,
        cap_radius,
        Vec3Fix::ZERO, // Static capsule
    )
}

/// TOI for a moving AABB against a static plane
///
/// Finds the AABB vertex closest to the plane and computes point-plane TOI.
#[must_use]
pub fn aabb_plane_toi(
    aabb: &AABB,
    velocity: Vec3Fix,
    plane_normal: Vec3Fix,
    plane_offset: Fix128,
) -> Option<TOI> {
    // Find the AABB vertex most in the direction of -plane_normal (closest to plane)
    let support_x = if plane_normal.x < Fix128::ZERO {
        aabb.max.x
    } else {
        aabb.min.x
    };
    let support_y = if plane_normal.y < Fix128::ZERO {
        aabb.max.y
    } else {
        aabb.min.y
    };
    let support_z = if plane_normal.z < Fix128::ZERO {
        aabb.max.z
    } else {
        aabb.min.z
    };
    let support = Vec3Fix::new(support_x, support_y, support_z);

    let dist = support.dot(plane_normal) - plane_offset;
    let vel_toward = velocity.dot(plane_normal);

    // Moving away from plane
    if vel_toward >= Fix128::ZERO && dist > Fix128::ZERO {
        return None;
    }

    // Already penetrating
    if dist <= Fix128::ZERO {
        return Some(TOI {
            t: Fix128::ZERO,
            point: support,
            normal: plane_normal,
        });
    }

    if vel_toward.is_zero() {
        return None;
    }

    // Time when support vertex touches plane
    let t = -dist / vel_toward;

    if t >= Fix128::ZERO && t <= Fix128::ONE {
        let point = support + velocity * t;
        Some(TOI {
            t,
            point,
            normal: plane_normal,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_sphere_toi() {
        // Two spheres moving toward each other
        let toi = sphere_sphere_toi(
            Vec3Fix::from_int(-5, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(10, 0, 0),
            Vec3Fix::from_int(5, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(-10, 0, 0),
        );

        let toi = toi.expect("Should find collision");
        // Distance = 10, combined_r = 2, relative_speed = 20
        // t = (10 - 2) / 20 = 0.4
        let expected = Fix128::from_ratio(4, 10);
        let error = (toi.t - expected).abs();
        assert!(
            error < Fix128::from_ratio(1, 10),
            "TOI should be ~0.4, got {:?}",
            toi.t
        );
    }

    #[test]
    fn test_sphere_sphere_miss() {
        // Spheres moving apart
        let toi = sphere_sphere_toi(
            Vec3Fix::from_int(-5, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(-10, 0, 0),
            Vec3Fix::from_int(5, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(10, 0, 0),
        );
        assert!(toi.is_none());
    }

    #[test]
    fn test_sphere_plane_toi() {
        // Sphere falling toward ground plane at y=0
        let toi = sphere_plane_toi(
            Vec3Fix::from_int(0, 10, 0),
            Fix128::ONE,
            Vec3Fix::from_int(0, -20, 0),
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        );

        let toi = toi.expect("Should find collision");
        // dist = 10, radius = 1, vel_toward = -20
        // t = (1 - 10) / (-20) = 0.45
        let expected = Fix128::from_ratio(45, 100);
        let error = (toi.t - expected).abs();
        assert!(error < Fix128::from_ratio(1, 10), "TOI should be ~0.45");
    }

    #[test]
    fn test_needs_ccd() {
        let config = CcdConfig::default();
        let fast = Vec3Fix::from_int(100, 0, 0);
        let slow = Vec3Fix::from_int(1, 0, 0);
        let dt = Fix128::from_ratio(1, 60);
        let radius = Fix128::ONE;

        assert!(needs_ccd(fast, radius, dt, &config), "Fast body needs CCD");
        assert!(
            !needs_ccd(slow, radius, dt, &config),
            "Slow body doesn't need CCD"
        );
    }

    #[test]
    fn test_swept_aabb() {
        let moving = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
        let target = AABB::new(Vec3Fix::from_int(5, -1, -1), Vec3Fix::from_int(7, 1, 1));
        let velocity = Vec3Fix::from_int(10, 0, 0);

        let t = swept_aabb(&moving, velocity, &target);
        assert!(t.is_some(), "Should detect swept collision");
        let t = t.unwrap();
        // Moving AABB right edge at x=1, target left edge at x=5
        // distance = 4, speed = 10, t = 0.4
        let expected = Fix128::from_ratio(4, 10);
        let error = (t - expected).abs();
        assert!(error < Fix128::from_ratio(1, 10), "TOI should be ~0.4");
    }

    #[test]
    fn test_conservative_advancement() {
        let config = CcdConfig::default();
        let start = Vec3Fix::from_int(-10, 0, 0);
        let displacement = Vec3Fix::from_int(20, 0, 0);
        let radius = Fix128::ONE;

        // Distance function: sphere at origin with radius 2
        let toi = conservative_advancement(
            start,
            displacement,
            radius,
            |pos| {
                let dist = pos.length() - Fix128::from_int(2);
                let normal = pos.normalize();
                (dist, normal)
            },
            &config,
        );

        assert!(
            toi.is_some(),
            "Should find collision via conservative advancement"
        );
    }

    #[test]
    fn test_speculative_contact_closing() {
        let dt = Fix128::from_ratio(1, 60);
        // gap=4, closing_speed=300, predicted_gap = 4 - 300/60 = -1 < 0
        let contact = speculative_contact(
            Vec3Fix::from_int(-3, 0, 0),
            Vec3Fix::from_int(300, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(3, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ONE,
            dt,
        );
        assert!(
            contact.is_some(),
            "Fast-closing spheres should generate speculative contact"
        );
    }

    #[test]
    fn test_speculative_contact_separating() {
        let dt = Fix128::from_ratio(1, 60);
        let contact = speculative_contact(
            Vec3Fix::from_int(-3, 0, 0),
            Vec3Fix::from_int(-10, 0, 0), // moving away
            Fix128::ONE,
            Vec3Fix::from_int(3, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ONE,
            dt,
        );
        assert!(
            contact.is_none(),
            "Separating spheres should not generate contact"
        );
    }

    #[test]
    fn test_capsule_plane_toi() {
        let toi = capsule_plane_toi(
            Vec3Fix::from_int(0, 10, -1),
            Vec3Fix::from_int(0, 10, 1),
            Fix128::ONE,
            Vec3Fix::from_int(0, -20, 0),
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        );
        let toi = toi.expect("Should find capsule-plane collision");
        assert!(toi.t > Fix128::ZERO && toi.t < Fix128::ONE);
    }

    #[test]
    fn test_sphere_capsule_toi() {
        let toi = sphere_capsule_toi(
            Vec3Fix::from_int(-10, 0, 0),
            Fix128::ONE,
            Vec3Fix::from_int(20, 0, 0),
            Vec3Fix::from_int(5, -2, 0),
            Vec3Fix::from_int(5, 2, 0),
            Fix128::ONE,
        );
        let toi = toi.expect("Should find sphere-capsule collision");
        assert!(toi.t > Fix128::ZERO && toi.t < Fix128::ONE);
    }

    #[test]
    fn test_aabb_plane_toi() {
        let aabb = AABB::new(Vec3Fix::from_int(-1, 8, -1), Vec3Fix::from_int(1, 10, 1));
        let toi = aabb_plane_toi(
            &aabb,
            Vec3Fix::from_int(0, -20, 0),
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        );
        let toi = toi.expect("Should find AABB-plane collision");
        assert!(toi.t > Fix128::ZERO && toi.t < Fix128::ONE);
    }

    #[test]
    fn test_aabb_plane_miss() {
        let aabb = AABB::new(Vec3Fix::from_int(-1, 8, -1), Vec3Fix::from_int(1, 10, 1));
        let toi = aabb_plane_toi(
            &aabb,
            Vec3Fix::from_int(0, 20, 0), // Moving away
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        );
        assert!(toi.is_none());
    }

    // ---- mutation-score tests (2026-09-15) ----------------------------

    fn fi(n: i64) -> Fix128 {
        Fix128::from_int(n)
    }

    fn r(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    #[test]
    fn needs_ccd_requires_both_displacement_and_speed_thresholds() {
        let cfg = CcdConfig::default(); // velocity_threshold 5
                                        // speed 8, dt 1/4 → 変位 2 > r/2 (r=2 → 1) かつ 8 > 5 → true
        assert!(needs_ccd(v3i(8, 0, 0), fi(2), r(1, 4), &cfg));
        // 変位不足: speed 8, dt 1/16 → 0.5 == r/2 → `>` false
        assert!(!needs_ccd(v3i(8, 0, 0), fi(2), r(1, 16), &cfg));
        // 速度不足: speed 4 (< 5) だが変位 4 > 1
        assert!(!needs_ccd(v3i(4, 0, 0), fi(2), fi(1), &cfg));
        // 速度ちょうど 5 は `>` false
        assert!(!needs_ccd(v3i(5, 0, 0), fi(2), fi(1), &cfg));
        assert!(needs_ccd(v3i(0, 0, 6), fi(2), fi(1), &cfg));
    }

    #[test]
    fn sphere_sphere_toi_closed_form() {
        // A at 0 r=1 速度 (8,0,0)、B at (10,0,0) r=1 静止 → 接触は距離 2 → t = (10-2)/8 = 1
        let toi = sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, 0, 0),
            fi(1),
            Vec3Fix::ZERO,
        )
        .expect("hit at t=1");
        assert_eq!(toi.t, Fix128::ONE);
        assert_eq!(toi.normal, v3i(1, 0, 0));
        assert_eq!(toi.point, v3i(9, 0, 0)); // pos_a(8) + n * r_a
                                             // 相対速度: B も逆向き (-8) → t = 8/16 = 1/2、point = 4 + 1
        let t2 = sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, 0, 0),
            fi(1),
            v3i(-8, 0, 0),
        )
        .expect("t=1/2");
        assert_eq!(t2.t, r(1, 2));
        assert_eq!(t2.point, v3i(5, 0, 0));
        // 半径が効く: r_b = 3 → 距離 4 で接触 → t = 6/8 = 3/4
        let t3 = sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, 0, 0),
            fi(3),
            Vec3Fix::ZERO,
        )
        .expect("t=3/4");
        assert_eq!(t3.t, r(3, 4));
        // 届かない (t > 1)、離れていく、横にずれて外れる、静止
        assert!(sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(4, 0, 0),
            v3i(10, 0, 0),
            fi(1),
            Vec3Fix::ZERO
        )
        .is_none());
        assert!(sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(-8, 0, 0),
            v3i(10, 0, 0),
            fi(1),
            Vec3Fix::ZERO
        )
        .is_none());
        assert!(sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, 5, 0),
            fi(1),
            Vec3Fix::ZERO
        )
        .is_none());
        assert!(sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            Vec3Fix::ZERO,
            v3i(10, 0, 0),
            fi(1),
            Vec3Fix::ZERO
        )
        .is_none());
        // 既に重なっている → t = 0、normal は a→b
        let ov = sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            Vec3Fix::ZERO,
            v3i(1, 0, 0),
            fi(1),
            Vec3Fix::ZERO,
        )
        .expect("overlap");
        assert_eq!(ov.t, Fix128::ZERO);
        assert_eq!(ov.normal, v3i(1, 0, 0));
        assert_eq!(ov.point, v3i(1, 0, 0));
        // 接線 (disc == 0): B at (10, 2, 0) r=1、A r=1 → 距離 2 の接線経路 → t = 10/8 > 1 → None、速度 16 → t = 10/16
        let tan = sphere_sphere_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(16, 0, 0),
            v3i(10, 2, 0),
            fi(1),
            Vec3Fix::ZERO,
        )
        .expect("tangent");
        assert_eq!(tan.t, r(5, 8));
    }

    #[test]
    fn sphere_plane_toi_closed_form() {
        // 平面 y = 0 (normal +y、offset 0)、球 center (0,5,0) r=1 速度 (0,-8,0) → t = (1-5)/(-8) = 1/2
        let toi = sphere_plane_toi(
            v3i(0, 5, 0),
            fi(1),
            v3i(0, -8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO,
        )
        .expect("hit");
        assert_eq!(toi.t, r(1, 2));
        assert_eq!(toi.point, v3i(0, 0, 0)); // center + v t - n r = (0,1,0) - (0,1,0)
        assert_eq!(toi.normal, v3i(0, 1, 0));
        // offset が効く: 平面 y = 2 → dist 3 → t = (1-3)/(-8) = 1/4、point = (0, 2, 0)
        let off =
            sphere_plane_toi(v3i(0, 5, 0), fi(1), v3i(0, -8, 0), v3i(0, 1, 0), fi(2)).expect("hit");
        assert_eq!(off.t, r(1, 4));
        assert_eq!(off.point, v3i(0, 2, 0));
        // 離れていく / 平行 / 届かない
        assert!(sphere_plane_toi(
            v3i(0, 5, 0),
            fi(1),
            v3i(0, 8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none());
        assert!(sphere_plane_toi(
            v3i(0, 5, 0),
            fi(1),
            v3i(8, 0, 0),
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none());
        assert!(sphere_plane_toi(
            v3i(0, 5, 0),
            fi(1),
            v3i(0, -2, 0),
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none());
        // 既に貫通 (|dist| <= r): dist 1 == r → t 0、point は投影点
        let pen = sphere_plane_toi(
            v3i(3, 1, 0),
            fi(1),
            v3i(0, 8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO,
        )
        .expect("touching");
        assert_eq!(pen.t, Fix128::ZERO);
        assert_eq!(pen.point, v3i(3, 0, 0));
        // 裏側から: center (0,-5,0)、速度 (0,8,0) → dist -5、近い面 (y = -4) が
        // 平面に触れる t = (-1 + 5)/8 = 1/2 (1.2.0 以前は遠い面の 3/4 = 通過時刻を返していた)
        // normal は球側 (-y)、接触点は平面上
        let back = sphere_plane_toi(
            v3i(0, -5, 0),
            fi(1),
            v3i(0, 8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO,
        )
        .expect("from behind");
        assert_eq!(back.t, r(1, 2));
        assert_eq!(back.normal, v3i(0, -1, 0));
        assert_eq!(back.point, Vec3Fix::ZERO);
        // 裏側で離れる向き → None
        assert!(sphere_plane_toi(
            v3i(0, -5, 0),
            fi(1),
            v3i(0, -8, 0),
            v3i(0, 1, 0),
            Fix128::ZERO
        )
        .is_none());
    }

    #[test]
    fn conservative_advancement_converges_on_a_plane() {
        let cfg = CcdConfig {
            max_iterations: 32,
            tolerance: r(1, 1000),
            velocity_threshold: fi(5),
        };
        // 距離関数: 平面 x = 10 (signed distance = 10 - x、normal -x 向き)
        let plane = |p: Vec3Fix| (fi(10) - p.x, v3i(-1, 0, 0));
        // start 0、変位 (16,0,0)、r=1 → 接触は x = 9 → t = 9/16 (gap は幾何級数で 0 に収束)
        let toi = conservative_advancement(Vec3Fix::ZERO, v3i(16, 0, 0), fi(1), plane, &cfg)
            .expect("hit");
        assert!((toi.t - r(9, 16)).abs() < r(1, 1000), "t {:?}", toi.t);
        assert_eq!(toi.normal, v3i(-1, 0, 0));
        // 届かない (変位 4 → x=4、gap 5)、速度 0
        assert!(
            conservative_advancement(Vec3Fix::ZERO, v3i(4, 0, 0), fi(1), plane, &cfg).is_none()
        );
        assert!(
            conservative_advancement(Vec3Fix::ZERO, Vec3Fix::ZERO, fi(1), plane, &cfg).is_none()
        );
        // 既に tolerance 内 (start x = 9) → t = 0
        let now = conservative_advancement(v3i(9, 0, 0), v3i(16, 0, 0), fi(1), plane, &cfg)
            .expect("immediate");
        assert_eq!(now.t, Fix128::ZERO);
        // max_iterations 1 では収束せず None
        let one = CcdConfig {
            max_iterations: 1,
            ..cfg
        };
        assert!(
            conservative_advancement(Vec3Fix::ZERO, v3i(16, 0, 0), fi(1), plane, &one).is_none()
        );
    }

    #[test]
    fn swept_aabb_entry_time_per_axis_and_overlap_cases() {
        let m = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        let target = AABB::new(v3i(5, -1, -1), v3i(7, 1, 1));
        // x で 4 離れ、速度 8 → t = 1/2
        assert_eq!(swept_aabb(&m, v3i(8, 0, 0), &target), Some(r(1, 2)));
        // 速度 2 → t = 2 > 1 → None
        assert_eq!(swept_aabb(&m, v3i(2, 0, 0), &target), None);
        // 逆向き → None、y でずれて通過しない → None
        assert_eq!(swept_aabb(&m, v3i(-8, 0, 0), &target), None);
        assert_eq!(
            swept_aabb(&m, v3i(8, 0, 0), &AABB::new(v3i(5, 3, -1), v3i(7, 5, 1))),
            None
        );
        // 斜め: y でも 4 離れた target、速度 (8, 8, 0) → x,y とも t 1/2 で進入 → 1/2
        assert_eq!(
            swept_aabb(&m, v3i(8, 8, 0), &AABB::new(v3i(5, 5, -1), v3i(7, 7, 1))),
            Some(r(1, 2))
        );
        // 各軸の enter は max 側: x は 1/2、y は 1/4 で進入 → 1/2
        assert_eq!(
            swept_aabb(&m, v3i(8, 8, 0), &AABB::new(v3i(5, 3, -1), v3i(7, 5, 1))),
            Some(r(1, 2))
        );
        // y 軸と z 軸単独
        assert_eq!(
            swept_aabb(&m, v3i(0, 8, 0), &AABB::new(v3i(-1, 5, -1), v3i(1, 7, 1))),
            Some(r(1, 2))
        );
        assert_eq!(
            swept_aabb(
                &m,
                v3i(0, 0, -8),
                &AABB::new(v3i(-1, -1, -7), v3i(1, 1, -5))
            ),
            Some(r(1, 2))
        );
        // 既に重なっている → 0、静止で重ならない → None、静止で重なる → 0
        assert_eq!(
            swept_aabb(&m, v3i(8, 0, 0), &AABB::new(Vec3Fix::ZERO, v3i(2, 2, 2))),
            Some(Fix128::ZERO)
        );
        assert_eq!(swept_aabb(&m, Vec3Fix::ZERO, &target), None);
        assert_eq!(
            swept_aabb(&m, Vec3Fix::ZERO, &AABB::new(Vec3Fix::ZERO, v3i(2, 2, 2))),
            Some(Fix128::ZERO)
        );
        // 速度ちょうど 4 → t = 1 (`<=` 境界)
        assert_eq!(swept_aabb(&m, v3i(4, 0, 0), &target), Some(Fix128::ONE));
    }

    #[test]
    fn slab_test_orders_times_and_handles_static_axis() {
        // a [-1,1] → b [5,7]、vel 8: t0 = (5-1)/8 = 1/2、t1 = (7+1)/8 = 1
        assert_eq!(
            slab_test(fi(-1), fi(1), fi(5), fi(7), fi(8)),
            Some((r(1, 2), fi(1)))
        );
        // 負の速度では swap される: a [5,7] → b [-1,1]、vel -8: t0 = (-1-7)/-8 = 1、t1 = (1-5)/-8 = 1/2 → (1/2, 1)
        assert_eq!(
            slab_test(fi(5), fi(7), fi(-1), fi(1), fi(-8)),
            Some((r(1, 2), fi(1)))
        );
        // 静止: 重なりなし → None、重なり / 接触 → 無限区間
        assert_eq!(slab_test(fi(-1), fi(1), fi(5), fi(7), Fix128::ZERO), None);
        assert_eq!(
            slab_test(fi(-1), fi(1), fi(0), fi(7), Fix128::ZERO),
            Some((fi(-1000000), fi(1000000)))
        );
        assert_eq!(
            slab_test(fi(-1), fi(1), fi(1), fi(7), Fix128::ZERO),
            Some((fi(-1000000), fi(1000000)))
        );
        assert_eq!(
            slab_test(fi(-1), fi(1), fi(-7), fi(-1), Fix128::ZERO),
            Some((fi(-1000000), fi(1000000)))
        );
    }

    #[test]
    fn speculative_contact_predicts_gap_breach() {
        // A at 0 r=1、B at (10,0,0) r=1、gap 8、closing 8*dt: dt 1/2 → predicted 4 > 0 → None
        assert!(speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            r(1, 2)
        )
        .is_none());
        // dt 2 → predicted -8 → depth 8
        let c = speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            fi(2),
        )
        .expect("breach");
        assert_eq!(c.depth, fi(8));
        assert_eq!(c.normal, v3i(1, 0, 0));
        assert_eq!(c.point_a, v3i(1, 0, 0));
        assert_eq!(c.point_b, v3i(9, 0, 0));
        // predicted == 0 ちょうど (dt 1) は `<` false → None
        assert!(speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            fi(1)
        )
        .is_none());
        // 離れる向き → None、既に重なり → depth = -gap
        assert!(speculative_contact(
            Vec3Fix::ZERO,
            v3i(-8, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            fi(9)
        )
        .is_none());
        let ov = speculative_contact(
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fi(1),
            v3i(1, 0, 0),
            Vec3Fix::ZERO,
            fi(1),
            fi(1),
        )
        .expect("overlap");
        assert_eq!(ov.depth, fi(1));
        assert_eq!(ov.point_b, Vec3Fix::ZERO);
        // 同一点 → None、B の速度も効く (B が -8 → closing 16、dt 1 → predicted -8)
        assert!(speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(1),
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fi(1),
            fi(1)
        )
        .is_none());
        let both = speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(1),
            v3i(10, 0, 0),
            v3i(-8, 0, 0),
            fi(1),
            fi(1),
        )
        .expect("both");
        assert_eq!(both.depth, fi(8));
        // adaptive_toi_substeps: 衝突コースなら max、そうでなければ 1、max 0 は 1 に clamp
        assert_eq!(
            adaptive_toi_substeps(
                Vec3Fix::ZERO,
                v3i(8, 0, 0),
                fi(1),
                v3i(10, 0, 0),
                Vec3Fix::ZERO,
                fi(1),
                fi(2),
                6
            ),
            6
        );
        assert_eq!(
            adaptive_toi_substeps(
                Vec3Fix::ZERO,
                v3i(8, 0, 0),
                fi(1),
                v3i(10, 0, 0),
                Vec3Fix::ZERO,
                fi(1),
                r(1, 2),
                6
            ),
            1
        );
        assert_eq!(
            adaptive_toi_substeps(
                Vec3Fix::ZERO,
                v3i(8, 0, 0),
                fi(1),
                v3i(10, 0, 0),
                Vec3Fix::ZERO,
                fi(1),
                fi(2),
                0
            ),
            1
        );
    }

    #[test]
    fn capsule_and_aabb_plane_toi_use_closest_feature() {
        let n = v3i(0, 1, 0);
        // capsule 端点 a (0,5,0) / b (0,9,0) r=1、速度 (0,-8,0) → 近い方 a → t = (1-5)/-8 = 1/2
        let cap = capsule_plane_toi(
            v3i(0, 5, 0),
            v3i(0, 9, 0),
            fi(1),
            v3i(0, -8, 0),
            n,
            Fix128::ZERO,
        )
        .expect("hit");
        assert_eq!(cap.t, r(1, 2));
        // 端点を入れ替えても同じ (b が近い場合の分岐)
        let cap2 = capsule_plane_toi(
            v3i(0, 9, 0),
            v3i(0, 5, 0),
            fi(1),
            v3i(0, -8, 0),
            n,
            Fix128::ZERO,
        )
        .expect("hit");
        assert_eq!(cap2.t, r(1, 2));
        // sphere_capsule: 球 (0,0,0) r=1 速度 (8,0,0)、capsule 線分 (10,-5,0)-(10,5,0) r=1 → 最近点 (10,0,0) → t = 1
        let sc = sphere_capsule_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, -5, 0),
            v3i(10, 5, 0),
            fi(1),
        )
        .expect("hit");
        assert_eq!(sc.t, Fix128::ONE);
        // clamp: 球が線分の延長上 (y = 9) → 最近点は端 (10,5,0) → 距離 √(100+16)... 横方向 4 のずれで外れる (速度 8 では届かない)
        assert!(sphere_capsule_toi(
            v3i(0, 9, 0),
            fi(1),
            v3i(8, 0, 0),
            v3i(10, -5, 0),
            v3i(10, 5, 0),
            fi(1)
        )
        .is_none());
        // 退化 capsule (a == b) は球扱い
        let deg = sphere_capsule_toi(
            Vec3Fix::ZERO,
            fi(1),
            v3i(8, 0, 0),
            v3i(10, 0, 0),
            v3i(10, 0, 0),
            fi(1),
        )
        .expect("sphere");
        assert_eq!(deg.t, Fix128::ONE);
        // aabb_plane: 箱 [1,3]³ 速度 (0,-8,0)、平面 y=0 → 最下点 y=1 → t = 1/8、point は support 頂点 + 移動
        let ab = aabb_plane_toi(
            &AABB::new(v3i(1, 1, 1), v3i(3, 3, 3)),
            v3i(0, -8, 0),
            n,
            Fix128::ZERO,
        )
        .expect("hit");
        assert_eq!(ab.t, r(1, 8));
        assert_eq!(ab.normal, n);
        // 法線が -y なら support は max 側 (y=3)、平面 y = 5 (offset -5 for normal -y): dist = -3 + 5 = 2、速度 (0,8,0) → vel_toward -8 → t = 2/8 = 1/4
        let ab2 = aabb_plane_toi(
            &AABB::new(v3i(1, 1, 1), v3i(3, 3, 3)),
            v3i(0, 8, 0),
            v3i(0, -1, 0),
            fi(-5),
        )
        .expect("hit");
        assert_eq!(ab2.t, r(1, 4));
        // 離れる / 既に貫通 (t 0) / 平行
        assert!(aabb_plane_toi(
            &AABB::new(v3i(1, 1, 1), v3i(3, 3, 3)),
            v3i(0, 8, 0),
            n,
            Fix128::ZERO
        )
        .is_none());
        let pen = aabb_plane_toi(
            &AABB::new(v3i(1, -1, 1), v3i(3, 3, 3)),
            v3i(0, 8, 0),
            n,
            Fix128::ZERO,
        )
        .expect("penetrating");
        assert_eq!(pen.t, Fix128::ZERO);
        assert_eq!(pen.point, v3i(1, -1, 1));
        assert!(aabb_plane_toi(
            &AABB::new(v3i(1, 1, 1), v3i(3, 3, 3)),
            v3i(8, 0, 0),
            n,
            Fix128::ZERO
        )
        .is_none());
    }

    // ---- mutation-score tests, round 2 (2026-09-15) ---------------------
    //
    // Each test below names the surviving cargo-mutants mutant it kills
    // (line numbers refer to the non-test code above). All inputs are chosen
    // so that the arithmetic is exact in I64F64 (powers of two, perfect
    // squares), so equality assertions are used unless noted otherwise.

    /// Kills L52 `>` -> `>=` (displacement test in `needs_ccd`).
    ///
    /// speed 8, dt 1/16 -> displacement 0.5; radius 1 -> half 0.5.
    /// The contract is strict (`displacement > radius/2`), so exactly on the
    /// boundary CCD is NOT needed even though speed 8 > threshold 5.
    #[test]
    fn needs_ccd_displacement_boundary_is_strict() {
        let cfg = CcdConfig::default();
        assert!(!needs_ccd(v3i(8, 0, 0), fi(1), r(1, 16), &cfg));
        // One step past the boundary (displacement 1.0 > 0.5) -> needed.
        assert!(needs_ccd(v3i(8, 0, 0), fi(1), r(1, 8), &cfg));
    }

    /// Kills L83 `*` -> `/` (contact point of the already-overlapping branch
    /// in `sphere_sphere_toi`).
    ///
    /// A at (3,0,0) r=2, B at (4,0,0) r=1: c = 1 - 9 < 0 -> overlap,
    /// normal = +x, point = center_a + normal * 2 = (5,0,0)
    /// (normal / 2 would give (3.5,0,0)).
    #[test]
    fn sphere_sphere_toi_overlap_point_scales_with_radius_a() {
        let toi = sphere_sphere_toi(
            v3i(3, 0, 0),
            fi(2),
            Vec3Fix::ZERO,
            v3i(4, 0, 0),
            fi(1),
            Vec3Fix::ZERO,
        )
        .expect("overlapping");
        assert_eq!(toi.t, Fix128::ZERO);
        assert_eq!(toi.normal, v3i(1, 0, 0));
        assert_eq!(toi.point, v3i(5, 0, 0));
    }

    /// Kills L104 `+` -> `-` (pos_b), L105 `-` -> `+` (normal direction) and
    /// L106 `*` -> `/` (point) in `sphere_sphere_toi`.
    ///
    /// Oblique, non-collinear impact so that every mutant changes the result:
    /// A: center (0,1,0), r 3/2, vel (8,0,0); B: center (8,7,0), r 1/2,
    /// vel (0,-8,0).
    /// rel_pos = (8,6,0), rel_vel = (-8,-8,0), combined_r = 2
    /// a = 128, b = 2*(-64-48) = -224, c = 100 - 4 = 96
    /// disc = 224^2 - 4*128*96 = 50176 - 49152 = 1024, sqrt = 32
    /// t = (224 - 32) / 256 = 3/4 (exact)
    /// pos_a = (6,1,0), pos_b = (8,1,0) -> normal = (1,0,0)
    /// point = pos_a + normal * 3/2 = (15/2, 1, 0)
    /// - L104 mutant: pos_b = (8,13,0) -> normal = normalize(2,12,0) != +x
    /// - L105 mutant: normal = normalize(14,2,0) != +x
    /// - L106 mutant: point = (6 + 2/3, 1, 0)
    #[test]
    fn sphere_sphere_toi_oblique_impact_closed_form() {
        let toi = sphere_sphere_toi(
            v3i(0, 1, 0),
            r(3, 2),
            v3i(8, 0, 0),
            v3i(8, 7, 0),
            r(1, 2),
            v3i(0, -8, 0),
        )
        .expect("hit at t=3/4");
        assert_eq!(toi.t, r(3, 4));
        assert_eq!(toi.normal, v3i(1, 0, 0));
        assert_eq!(toi.point, Vec3Fix::new(r(15, 2), fi(1), Fix128::ZERO));
    }

    /// Kills L134 `*` -> `/` (penetrating-branch point) and L150 `*` -> `/`
    /// (surface point, `plane_normal * radius`) in `sphere_plane_toi`.
    ///
    /// Penetrating: center (3, 5/2, 0), r 1, plane y = 2 -> dist 1/2 <= r,
    /// point = center - n * dist = (3, 2, 0)  (n / dist gives (3, 1/2, 0)).
    /// Surface: center (0,5,0), r 2, vel (0,-8,0), plane y = 0:
    /// t = (2 - 5) / (-8) = 3/8, point = (0,5,0) + (0,-3,0) - (0,2,0) = 0
    /// (n / r gives (0, 3/2, 0)).
    #[test]
    fn sphere_plane_toi_points_scale_with_dist_and_radius() {
        let n = v3i(0, 1, 0);
        let pen = sphere_plane_toi(
            Vec3Fix::new(fi(3), r(5, 2), Fix128::ZERO),
            fi(1),
            v3i(0, 8, 0),
            n,
            fi(2),
        )
        .expect("penetrating");
        assert_eq!(pen.t, Fix128::ZERO);
        assert_eq!(pen.point, v3i(3, 2, 0));

        let hit = sphere_plane_toi(v3i(0, 5, 0), fi(2), v3i(0, -8, 0), n, Fix128::ZERO)
            .expect("surface hit");
        assert_eq!(hit.t, r(3, 8));
        assert_eq!(hit.point, Vec3Fix::ZERO);
    }

    /// Kills L185 `-` -> `+`, L185 `*` -> `/` (contact point) and
    /// L198 `>` -> `>=` (t == 1 must still be checked) in
    /// `conservative_advancement`.
    ///
    /// Distance function: plane x = 10, signed distance 10 - x, normal -x.
    /// (1) start (8,0,0), r 2: dist 2, gap 0 -> immediate hit,
    ///     point = pos - normal * dist = (8,0,0) - (-2,0,0) = (10,0,0)
    ///     (`+` gives (6,0,0); normal / dist gives (8.5,0,0)).
    /// (2) start 0, displacement (9,0,0), r 1: iteration 0 has dist 10,
    ///     gap 9, speed 9 -> dt = 1 -> t = 1 exactly. `t > 1` is false so
    ///     iteration 1 evaluates pos (9,0,0): dist 1, gap 0 -> Some(t = 1).
    ///     With `>=` the function returns None at t == 1.
    #[test]
    fn conservative_advancement_contact_point_and_t_equal_one() {
        let cfg = CcdConfig::default();
        let plane = |p: Vec3Fix| (fi(10) - p.x, v3i(-1, 0, 0));

        let now = conservative_advancement(v3i(8, 0, 0), v3i(16, 0, 0), fi(2), plane, &cfg)
            .expect("immediate");
        assert_eq!(now.t, Fix128::ZERO);
        assert_eq!(now.point, v3i(10, 0, 0));

        let edge = conservative_advancement(Vec3Fix::ZERO, v3i(9, 0, 0), fi(1), plane, &cfg)
            .expect("touch exactly at t = 1");
        assert_eq!(edge.t, Fix128::ONE);
        assert_eq!(edge.point, v3i(10, 0, 0));
    }

    /// Kills the `t_enter > t_exit` mutants in `swept_aabb`:
    /// L224 `>` -> `==` / `>=` (x), L241 `>` -> `==` / `>=` (y),
    /// L258 `>` -> `==` / `>=` (z).
    ///
    /// Closed intervals: a corner-to-corner touch (`t_enter == t_exit`) is a
    /// hit, not a miss.
    /// - x axis: after the first slab `t_enter <= t_exit` always holds
    ///   (`slab_test` sorts), so equality needs zero-thickness boxes:
    ///   a = [0,0], b = [4,4], v 8 -> t0 = t1 = 1/2 -> Some(1/2).
    /// - y axis: m [-1,1]^3, v (8,8,0), target x [5,7] y [9,11]:
    ///   x (1/2, 1), y (1, 3/2) -> t_enter = t_exit = 1 -> Some(1).
    /// - z axis: same with v (8,0,8), target x [5,7] z [9,11] -> Some(1).
    #[test]
    fn swept_aabb_corner_touch_counts_as_hit() {
        let thin_m = AABB::new(v3i(0, -1, -1), v3i(0, 1, 1));
        let thin_t = AABB::new(v3i(4, -1, -1), v3i(4, 1, 1));
        assert_eq!(swept_aabb(&thin_m, v3i(8, 0, 0), &thin_t), Some(r(1, 2)));

        let m = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        assert_eq!(
            swept_aabb(&m, v3i(8, 8, 0), &AABB::new(v3i(5, 9, -1), v3i(7, 11, 1))),
            Some(Fix128::ONE)
        );
        assert_eq!(
            swept_aabb(&m, v3i(8, 0, 8), &AABB::new(v3i(5, -1, 9), v3i(7, 1, 11))),
            Some(Fix128::ONE)
        );
    }

    /// Kills L240 `<` -> `==` (y exit) and L257 `<` -> `==` (z exit) in
    /// `swept_aabb`: the exit time must be lowered by a later axis.
    ///
    /// m [-1,1]^3, v (8,8,0), target x [5,7] y [1,2]:
    /// x (1/2, 1), y (0, 3/8) -> t_enter 1/2 > t_exit 3/8 -> None
    /// (the box leaves the y slab before entering the x slab).
    /// With the mutant t_exit stays 1 and the call returns Some(1/2).
    #[test]
    fn swept_aabb_exit_before_enter_on_later_axis_is_miss() {
        let m = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        assert_eq!(
            swept_aabb(&m, v3i(8, 8, 0), &AABB::new(v3i(5, 1, -1), v3i(7, 2, 1))),
            None
        );
        assert_eq!(
            swept_aabb(&m, v3i(8, 0, 8), &AABB::new(v3i(5, -1, 1), v3i(7, 1, 2))),
            None
        );
    }

    /// Kills L383 `+` -> `-`, L383 `*` -> `/`, L384 `*` -> `/` (overlap
    /// branch) and L403 / L404 `*` -> `/` (speculative branch) in
    /// `speculative_contact`.
    ///
    /// Overlap: A (2,0,0) r 2, B (3,0,0) r 1/2: dist 1, gap -3/2,
    /// point_a = (2,0,0) + (1,0,0)*2 = (4,0,0), point_b = (3,0,0) - 1/2 = (5/2,0,0).
    /// Speculative: A 0 r 2 vel (8,0,0), B (10,0,0) r 1/2, dt 1:
    /// gap 15/2, closing 8 -> predicted -1/2 -> depth 1/2,
    /// point_a = (2,0,0), point_b = (19/2,0,0).
    #[test]
    fn speculative_contact_points_scale_with_radii() {
        let ov = speculative_contact(
            v3i(2, 0, 0),
            Vec3Fix::ZERO,
            fi(2),
            v3i(3, 0, 0),
            Vec3Fix::ZERO,
            r(1, 2),
            fi(1),
        )
        .expect("overlap");
        assert_eq!(ov.depth, r(3, 2));
        assert_eq!(ov.point_a, v3i(4, 0, 0));
        assert_eq!(
            ov.point_b,
            Vec3Fix::new(r(5, 2), Fix128::ZERO, Fix128::ZERO)
        );

        let sp = speculative_contact(
            Vec3Fix::ZERO,
            v3i(8, 0, 0),
            fi(2),
            v3i(10, 0, 0),
            Vec3Fix::ZERO,
            r(1, 2),
            fi(1),
        )
        .expect("speculative");
        assert_eq!(sp.depth, r(1, 2));
        assert_eq!(sp.point_a, v3i(2, 0, 0));
        assert_eq!(
            sp.point_b,
            Vec3Fix::new(r(19, 2), Fix128::ZERO, Fix128::ZERO)
        );
    }

    /// Kills L425 `-` -> `+` (dist_a), L426 `-` -> `+` (dist_b) and
    /// L428 `<` -> `<=` (tie-break) in `capsule_plane_toi`.
    ///
    /// Plane y = 2 (offset 2), r 1, vel (0,-8,0).
    /// - a y=5, b y=7: dist_a 3 < dist_b 5 -> endpoint a -> t = (1-3)/(-8) = 1/4.
    ///   L425 mutant: dist_a = 7 -> picks b -> t = 1/2.
    /// - a y=7, b y=5: dist_b 3 -> endpoint b -> t = 1/4.
    ///   L426 mutant: dist_b = 7 -> picks a -> t = 1/2.
    /// - Tie (capsule parallel to the plane): a (0,9,-1), b (0,9,1), r 1,
    ///   vel (0,-16,0), plane y = 0: `dist_a < dist_b` is false, so endpoint b
    ///   is used -> t = 1/2, point = (0,9,1) + (0,-8,0) - (0,1,0) = (0,0,1).
    ///   `<=` would pick a and report (0,0,-1).
    #[test]
    fn capsule_plane_toi_offset_and_tie_break() {
        let n = v3i(0, 1, 0);
        let a_near = capsule_plane_toi(v3i(0, 5, 0), v3i(0, 7, 0), fi(1), v3i(0, -8, 0), n, fi(2))
            .expect("hit");
        assert_eq!(a_near.t, r(1, 4));
        assert_eq!(a_near.point, v3i(0, 2, 0));

        let b_near = capsule_plane_toi(v3i(0, 7, 0), v3i(0, 5, 0), fi(1), v3i(0, -8, 0), n, fi(2))
            .expect("hit");
        assert_eq!(b_near.t, r(1, 4));
        assert_eq!(b_near.point, v3i(0, 2, 0));

        let tie = capsule_plane_toi(
            v3i(0, 9, -1),
            v3i(0, 9, 1),
            fi(1),
            v3i(0, -16, 0),
            n,
            Fix128::ZERO,
        )
        .expect("hit");
        assert_eq!(tie.t, r(1, 2));
        assert_eq!(tie.point, v3i(0, 0, 1));
    }

    /// Kills L455 `<` -> `==` (lower clamp of the segment parameter) in
    /// `sphere_capsule_toi`.
    ///
    /// Sphere (0,-9,0) r 1 vel (8,0,0); capsule (10,-5,0)-(10,5,0) r 1.
    /// t = ((-10,-4,0)·(0,10,0)) / 100 = -2/5 -> clamped to 0 -> closest
    /// point (10,-5,0); the sphere passes 4 below it (combined r 2) -> None.
    /// Without the clamp the closest point would be (10,-9,0) and the sphere
    /// would hit it at t = 1.
    #[test]
    fn sphere_capsule_toi_clamps_below_segment_start() {
        assert!(sphere_capsule_toi(
            v3i(0, -9, 0),
            fi(1),
            v3i(8, 0, 0),
            v3i(10, -5, 0),
            v3i(10, 5, 0),
            fi(1)
        )
        .is_none());
        // Control: level with the segment start it does hit at t = 1.
        let hit = sphere_capsule_toi(
            v3i(0, -5, 0),
            fi(1),
            v3i(8, 0, 0),
            v3i(10, -5, 0),
            v3i(10, 5, 0),
            fi(1),
        )
        .expect("hit");
        assert_eq!(hit.t, Fix128::ONE);
    }

    /// Kills L485 `<` -> `>` (support x), L490 `<` -> `<=` (support y) and
    /// L495 `<` -> `>` (support z) in `aabb_plane_toi`.
    ///
    /// AABB [1,3]^3.
    /// - Plane x = 10 with normal -x (offset -10), vel (8,0,0): support
    ///   (max.x, min.y, min.z) = (3,1,1), dist = -3 + 10 = 7, vel_toward -8,
    ///   t = 7/8, point = (3,1,1) + (7,0,0) = (10,1,1).
    ///   L485 mutant picks min.x -> dist 9 -> t 9/8 -> None.
    ///   L490 mutant picks max.y (normal.y == 0) -> point (10,3,1).
    /// - Plane x = -5 with normal +x (offset -5), vel (-8,0,0): support
    ///   min.x = 1, dist 6, t = 3/4 (mutant: max.x 3 -> dist 8 -> t 1).
    /// - Plane z = 10 with normal -z (offset -10), vel (0,0,8): support
    ///   max.z = 3, t = 7/8, point (1,1,10) (L495 mutant -> None).
    #[test]
    fn aabb_plane_toi_support_vertex_follows_normal_sign() {
        let bx = AABB::new(v3i(1, 1, 1), v3i(3, 3, 3));

        let neg_x = aabb_plane_toi(&bx, v3i(8, 0, 0), v3i(-1, 0, 0), fi(-10)).expect("hit");
        assert_eq!(neg_x.t, r(7, 8));
        assert_eq!(neg_x.point, v3i(10, 1, 1));

        let pos_x = aabb_plane_toi(&bx, v3i(-8, 0, 0), v3i(1, 0, 0), fi(-5)).expect("hit");
        assert_eq!(pos_x.t, r(3, 4));
        assert_eq!(pos_x.point, v3i(-5, 1, 1));

        let neg_z = aabb_plane_toi(&bx, v3i(0, 0, 8), v3i(0, 0, -1), fi(-10)).expect("hit");
        assert_eq!(neg_z.t, r(7, 8));
        assert_eq!(neg_z.point, v3i(1, 1, 10));
    }

    /// Kills L506 `>` -> `==` / `>=` (resting contact while separating),
    /// L526 `&&` -> `||` (t > 1 must be a miss) and L527 `+` -> `-`,
    /// `*` -> `/` (contact point) in `aabb_plane_toi`.
    ///
    /// - AABB resting on y = 0 (min.y = 0) moving away (0,8,0): dist == 0 is
    ///   not "moving away from the plane", it is the penetrating branch ->
    ///   Some(t = 0, point = support (1,0,1)).
    /// - AABB [1,3]^3, vel (0,-1/2,0): t = 1 / (1/2) = 2 > 1 -> None.
    /// - AABB [1,3]^3, vel (0,-8,0): t = 1/8,
    ///   point = (1,1,1) + (0,-8,0) * 1/8 = (1,0,1)
    ///   (`-` gives (1,2,1); velocity / t gives (1,-63,1)).
    #[test]
    fn aabb_plane_toi_resting_contact_and_contact_point() {
        let n = v3i(0, 1, 0);
        let resting = aabb_plane_toi(
            &AABB::new(v3i(1, 0, 1), v3i(3, 2, 3)),
            v3i(0, 8, 0),
            n,
            Fix128::ZERO,
        )
        .expect("resting contact");
        assert_eq!(resting.t, Fix128::ZERO);
        assert_eq!(resting.point, v3i(1, 0, 1));

        let bx = AABB::new(v3i(1, 1, 1), v3i(3, 3, 3));
        assert!(aabb_plane_toi(
            &bx,
            Vec3Fix::new(Fix128::ZERO, r(-1, 2), Fix128::ZERO),
            n,
            Fix128::ZERO
        )
        .is_none());

        let hit = aabb_plane_toi(&bx, v3i(0, -8, 0), n, Fix128::ZERO).expect("hit");
        assert_eq!(hit.t, r(1, 8));
        assert_eq!(hit.point, v3i(1, 0, 1));
    }
}
