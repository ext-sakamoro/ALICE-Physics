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
#[derive(Clone, Copy, Debug)]
pub struct TOI {
    /// Time of first impact [0, 1], where 0=start, 1=end of timestep
    pub t: Fix128,
    /// Contact point at time of impact (world space)
    pub point: Vec3Fix,
    /// Contact normal at time of impact
    pub normal: Vec3Fix,
}

/// CCD configuration
#[derive(Clone, Copy, Debug)]
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
/// Plane: dot(normal, p) = offset
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

    // Already penetrating
    if dist.abs() <= radius {
        let point = center - plane_normal * dist;
        return Some(TOI {
            t: Fix128::ZERO,
            point,
            normal: plane_normal,
        });
    }

    if vel_toward.is_zero() {
        return None;
    }

    // t when sphere surface touches plane
    let t = (radius - dist) / vel_toward;

    if t >= Fix128::ZERO && t <= Fix128::ONE {
        let point = center + velocity * t - plane_normal * radius;
        Some(TOI {
            t,
            point,
            normal: plane_normal,
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
/// The solver then prevents penetration by maintaining the gap.
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
}
