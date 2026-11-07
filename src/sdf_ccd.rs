//! SDF Sphere Tracing CCD (Continuous Collision Detection)
//!
//! Uses sphere tracing (ray marching) along the SDF to detect
//! time of impact for arbitrary SDF shapes. Works with concave,
//! CSG, and fractal geometry — anything expressible as an SDF.
//!
//! # MODEL — sphere tracing against a signed distance field
//!
//! - **方程式 (equations)**: time of impact `t ∈ [0, 1]` of a sphere of
//!   radius `r` translating by `Δx` against a field `φ` placed by a
//!   frame of uniform scale `k`: at `x(t) = x₀ + Δx t`,
//!   `gap = k φ(x_local) − r`; a hit when `gap ≤ tol`, else
//!   `t ← t + s · gap / |Δx|` with the safety factor `s` (default `0.9`,
//!   `tol = 10⁻³`, 64 iterations). The hit's point is `x − n k φ` with
//!   `n` the field's world normal there. The ray march is the same with
//!   `r = 0`, `t` a distance along the unit direction, a hit when
//!   `k φ < tol` (strict). The batch and `sdf_ccd_hits` sweep every
//!   non-static, non-sensor body faster than the velocity threshold
//!   (`5` m/s, equality swept) along `v dt`, skip an SDF attached to the
//!   body itself and sort the hits by `t` (stable).
//! - **単位 (units)**: positions, `Δx`, `r` and `t` are `Fix128` (m and
//!   dimensionless); the field, `k`, `gap`, `tol`, `s` and each step are
//!   `f32` before the step returns to `Fix128`. Bit identity across
//!   platforms is claimed only for fields whose arithmetic is `+ − × ÷ √`
//!   or goes through the crate's deterministic math (the golden below);
//!   a user field with other transcendentals is not claimed.
//! - **保存則 (conservation)**: none applies — a query on straight-line
//!   motion; no body is moved and no momentum or energy is exchanged
//!   (the world step's SDF response is the discrete push-out).
//! - **適用範囲 (range)**: `φ` must be 1-Lipschitz (a true signed
//!   distance or a lower bound of it): the safe step relies on no surface
//!   within `gap`; a field that overestimates the distance is stepped
//!   through and the hit is missed, undetected. A hit's `t` is never past
//!   the true impact and short of it by at most `tol / (|Δx| cos θ)` at
//!   approach angle `θ`; a grazing approach can exhaust the 64 iterations
//!   and answer `None`. The absolute `tol` in `f32` loses meaning once
//!   `|x|` exceeds about `2¹³` m (the `f32` spacing there is `2⁻¹⁰`);
//!   not measured. Outside the range nothing is clamped: `t` past `1`, a zero
//!   `Δx` (also when starting inside) or an exhausted budget is `None`.
//! - **近似 (approximations)**: a translating sphere, no rotation and no
//!   shape; uniform scale only; an attached SDF is swept at the pose of
//!   the last step, not relative to its body's motion; the step stops
//!   short of the surface by up to `tol`; no convergence sweep over `tol`,
//!   `s` or the budget exists yet.
//! - **検証 (validation)**: closed-form impact against a plane and a unit
//!   sphere, `t` in `[t* − tol / |Δx|, t*]`:
//!   `tests/analytic_sdf_ccd_wiring.rs::
//!   plane_toi_matches_closed_form_and_never_overshoots` (`t* = 0.45`),
//!   `::oblique_sphere_toi_matches_quadratic` (`t* = (5 − √2) / 10`),
//!   `::scaled_and_translated_collider_toi`,
//!   `::rotated_plane_reports_world_normal`,
//!   `::misses_and_degenerate_inputs`,
//!   `::step_safety_scales_iteration_count_not_the_answer`,
//!   `::ray_march_hits_at_closed_form_distance`,
//!   `::ray_march_with_a_non_unit_direction_does_not_tunnel`,
//!   `::ray_march_step_is_scaled_by_step_safety`,
//!   `::contact_exactly_at_tolerance_is_a_hit_at_t_zero`,
//!   `::world_sweep_reports_fast_bodies_earliest_first`,
//!   `::batch_filters_and_sorts_like_the_world_sweep`,
//!   `::velocity_threshold_gates_on_the_bodys_speed_not_the_displacement`;
//!   the borrowed-field entry point,
//!   `tests/analytic_sdf_ccd_borrowed_field.rs::
//!   borrowed_field_is_bit_identical_to_collider_entry_point` (30 cases,
//!   `Option<TOI>` equal bit for bit),
//!   `::closed_form_toi_against_unit_sphere` (`r = 0.5, 0.25, 0`),
//!   `::closed_form_toi_through_translated_scaled_frame`,
//!   `::degenerate_inputs`; the finite-difference normal through the
//!   trace, `tests/analytic_sdf_distance_query.rs::
//!   ccd_and_contacts_match_analytic_normal_query` (within `10⁻²`); the
//!   cross-platform bits,
//!   `tests/determinism_golden_f32.rs::golden_sdf_collision_family`.
//!   validation missing: no tunnelling of a fast body through a thin SDF
//!   wall inside the world step (the trace is a query; the step's SDF
//!   response is discrete);
//!   validation missing: a field that is not 1-Lipschitz (the overshoot
//!   is described, not tested);
//!   validation missing: convergence over `tol`, `s` and the iteration
//!   budget (no sweep yet);
//!   validation missing: a moving or rotating attached SDF swept with
//!   relative motion.
//! - **段 (tiers)**: L0 only — the trace is the reference; no lighter
//!   tier exists.
//!
//! # Algorithm
//!
//! 1. Start at body position at t=0
//! 2. Evaluate SDF distance at current position
//! 3. Advance along trajectory by `distance - radius` (safe step)
//! 4. Repeat until contact or t > 1
//!
//! This is a generalization of conservative advancement that directly
//! uses the SDF's Lipschitz property for optimal step sizes.
//!
//! Author: Moroya Sakamoto

#[cfg(feature = "std")]
use crate::ccd::TOI;
use crate::math::Fix128;
#[cfg(feature = "std")]
use crate::math::Vec3Fix;
#[cfg(feature = "std")]
use crate::sdf_collider::{SdfCollider, SdfFrame, SdfQuery};

// ============================================================================
// SDF CCD Configuration
// ============================================================================

/// Configuration for SDF sphere tracing CCD
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCcdConfig {
    /// Maximum sphere tracing iterations
    pub max_iterations: usize,
    /// Contact tolerance: `sphere_trace_sdf` stops when `gap <= tolerance`,
    /// `ray_march_sdf` when `distance < tolerance`
    pub tolerance: f32,
    /// Minimum velocity magnitude to trigger CCD
    pub velocity_threshold: Fix128,
    /// Safety margin multiplier for step size (< 1.0 for conservative)
    pub step_safety: f32,
}

impl Default for SdfCcdConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            tolerance: 0.001,
            velocity_threshold: Fix128::from_int(5),
            step_safety: 0.9,
        }
    }
}

// ============================================================================
// SDF Sphere Tracing CCD
// ============================================================================

/// Sphere trace a moving sphere against an SDF collider.
///
/// Returns time of impact in [0, 1] where:
/// - 0 = start of timestep (already penetrating)
/// - 1 = end of timestep
///
/// The sphere moves from `start` to `start + displacement` over [0, 1].
/// This is [`sphere_trace_sdf_field`] with the collider's field and
/// [`SdfCollider::frame`]; the two return the same result bit for bit.
#[cfg(feature = "std")]
#[must_use]
pub fn sphere_trace_sdf(
    start: Vec3Fix,
    displacement: Vec3Fix,
    radius: Fix128,
    sdf: &SdfCollider,
    config: &SdfCcdConfig,
) -> Option<TOI> {
    sphere_trace_sdf_field(
        start,
        displacement,
        radius,
        &*sdf.field,
        &sdf.frame(),
        config,
    )
}

/// Sphere trace a moving sphere against a borrowed distance field.
///
/// The same computation as [`sphere_trace_sdf`], for a field that is not
/// owned by an [`SdfCollider`]: `field` is any [`SdfQuery`] (every
/// [`SdfField`](crate::sdf_collider::SdfField), or borrowed closures through
/// [`ClosureSdfQuery`](crate::sdf_collider::ClosureSdfQuery)) and need not be
/// `'static`, `Send` or `Sync`. `frame` places the field in the world
/// ([`SdfFrame::IDENTITY`] when the field is already in world coordinates).
///
/// The sphere of `radius` moves from `start` to `start + displacement` over
/// `t` in [0, 1]. Each iteration evaluates the gap `distance * scale - radius`
/// at the current position: when `gap <= config.tolerance` the hit is
/// returned (with the world normal of the field there and the point `pos -
/// normal * distance`), otherwise `t` advances by `gap * step_safety /
/// |displacement|`. A hit's `t` is therefore at most the true time of impact
/// and short of it by about `tolerance / |displacement|`.
///
/// Degenerate input:
/// - zero `displacement`: `None` (also when the sphere starts inside)
/// - zero `radius`: the trace of the centre point
/// - `config.max_iterations == 0`: `None` (no evaluation is made)
/// - `t` passing 1, or the iteration budget running out: `None`
#[cfg(feature = "std")]
#[must_use]
pub fn sphere_trace_sdf_field<F: SdfQuery + ?Sized>(
    start: Vec3Fix,
    displacement: Vec3Fix,
    radius: Fix128,
    field: &F,
    frame: &SdfFrame,
    config: &SdfCcdConfig,
) -> Option<TOI> {
    let radius_f32 = radius.to_f32();
    let scale = frame.scale_f32();
    let disp_len = displacement.length();

    if disp_len.is_zero() {
        return None;
    }

    let inv_disp_len = Fix128::ONE / disp_len;
    let mut t = Fix128::ZERO;

    for _ in 0..config.max_iterations {
        // Current position along trajectory
        let pos = start + displacement * t;

        // SDF distance query in local space
        let (lx, ly, lz) = frame.world_to_local(pos);
        let dist = field.query_distance(lx, ly, lz) * scale;

        // Gap = distance to surface minus sphere radius
        let gap = dist - radius_f32;

        if gap <= config.tolerance {
            // Contact found
            let (nx, ny, nz) = field.query_normal(lx, ly, lz);
            let normal = frame.local_normal_to_world(nx, ny, nz);
            let point = pos - normal * Fix128::from_f32(dist);

            return Some(TOI { t, point, normal });
        }

        // Safe advance: the SDF guarantees no surface within `gap` distance
        let safe_step = Fix128::from_f32(gap * config.step_safety);
        let dt = safe_step * inv_disp_len;
        t = t + dt;

        if t > Fix128::ONE {
            return None; // No collision this timestep
        }
    }

    None // Max iterations reached without convergence
}

/// Sphere trace a moving point (zero radius) against an SDF.
///
/// Simpler variant for raycasting against SDF geometry. `direction` need not
/// be normalized; the returned `t` and `max_distance` are distances along it.
#[cfg(feature = "std")]
#[must_use]
pub fn ray_march_sdf(
    origin: Vec3Fix,
    direction: Vec3Fix,
    max_distance: Fix128,
    sdf: &SdfCollider,
    config: &SdfCcdConfig,
) -> Option<TOI> {
    let len_sq = direction.length_squared();
    if len_sq.is_zero() {
        return None;
    }
    // `t` is a distance (it is advanced by the SDF value and compared with
    // `max_distance`), so the direction has to be a unit vector: stepping
    // `dist` along `origin + direction * t` with `|direction| > 1` jumps
    // `|direction| * dist` and tunnels through the surface. A unit direction
    // (the common case) is used as given, bit for bit.
    let direction = if len_sq == Fix128::ONE {
        direction
    } else {
        direction.normalize()
    };
    let scale = sdf.scale_f32;
    let mut t = Fix128::ZERO;

    for _ in 0..config.max_iterations {
        let pos = origin + direction * t;

        let (lx, ly, lz) = sdf.world_to_local(pos);
        let dist = sdf.field.distance(lx, ly, lz) * scale;

        if dist < config.tolerance {
            let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
            let normal = sdf.local_normal_to_world(nx, ny, nz);
            let point = pos;

            return Some(TOI { t, point, normal });
        }

        t = t + Fix128::from_f32(dist * config.step_safety);

        if t > max_distance {
            return None;
        }
    }

    None
}

/// Batch sphere trace: test multiple bodies against multiple SDF colliders.
///
/// Returns (`body_index`, `sdf_index`, TOI) tuples for all detected impacts.
#[cfg(feature = "std")]
#[must_use]
pub fn batch_sphere_trace_sdf(
    bodies: &[crate::solver::RigidBody],
    displacements: &[Vec3Fix],
    radius: Fix128,
    sdf_colliders: &[SdfCollider],
    config: &SdfCcdConfig,
) -> Vec<(usize, usize, TOI)> {
    let mut results = Vec::new();

    for (body_idx, (body, disp)) in bodies.iter().zip(displacements.iter()).enumerate() {
        if body.is_static() {
            continue;
        }

        // Check if body is moving fast enough for CCD
        let speed = body.velocity.length();
        if speed < config.velocity_threshold {
            continue;
        }

        for (sdf_idx, sdf) in sdf_colliders.iter().enumerate() {
            if sdf.body_index == body_idx {
                continue;
            }

            if let Some(toi) = sphere_trace_sdf(body.position, *disp, radius, sdf, config) {
                results.push((body_idx, sdf_idx, toi));
            }
        }
    }

    // Sort by earliest TOI for deterministic processing
    results.sort_by_key(|a| a.2.t);
    results
}

#[cfg(feature = "std")]
impl crate::solver::PhysicsWorld {
    /// Sweep the world's moving bodies against its SDF colliders.
    ///
    /// A body is swept as a sphere of [`PhysicsWorld::sdf_collision_radius`](crate::solver::PhysicsWorld::sdf_collision_radius)
    /// (the radius [`PhysicsWorld::sdf_contacts`](crate::solver::PhysicsWorld::sdf_contacts) gives a body without a shape)
    /// along `velocity * dt`; this is the continuous counterpart of the
    /// discrete push-out in [`PhysicsWorld::step`](crate::solver::PhysicsWorld::step), to be asked before a step to
    /// learn which fast bodies would tunnel through a thin SDF. Static bodies,
    /// sensors and bodies slower than `config.velocity_threshold` are skipped,
    /// and an SDF attached to the body itself is never tested against it. Hits
    /// come back as in [`batch_sphere_trace_sdf`], earliest time of impact first.
    /// An SDF attached to a body is swept at the pose it was given when it was
    /// added or at the end of the last step (see
    /// [`PhysicsWorld::sdf_contacts`](crate::solver::PhysicsWorld::sdf_contacts)).
    ///
    /// `dt` is the frame step.
    #[must_use]
    pub fn sdf_ccd_hits(&self, dt: Fix128, config: &SdfCcdConfig) -> Vec<(usize, usize, TOI)> {
        let displacements: Vec<Vec3Fix> = self
            .bodies
            .iter()
            .map(|b| {
                if b.is_sensor {
                    Vec3Fix::ZERO
                } else {
                    b.velocity * dt
                }
            })
            .collect();
        batch_sphere_trace_sdf(
            &self.bodies,
            &displacements,
            self.sdf_collision_radius,
            &self.sdf_colliders,
            config,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::sdf_collider::{ClosureSdf, SdfCollider};

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        )
    }

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    #[test]
    fn test_sphere_trace_hit() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let start = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
        let displacement = Vec3Fix::from_f32(10.0, 0.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_some(), "Should detect collision with sphere");

        let toi = result.unwrap();
        let t = toi.t.to_f32();
        // Start at -5, sphere at 0, radius 1 + body radius 0.5 = contact at x = -1.5
        // t = (5 - 1.5) / 10 = 0.35
        assert!(t > 0.2 && t < 0.5, "TOI should be ~0.35, got {t}");
    }

    #[test]
    fn test_sphere_trace_miss() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Moving parallel to sphere, missing it
        let start = Vec3Fix::from_f32(-5.0, 5.0, 0.0);
        let displacement = Vec3Fix::from_f32(10.0, 0.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_none(), "Should miss the sphere");
    }

    #[test]
    fn test_sphere_trace_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Falling sphere toward ground
        let start = Vec3Fix::from_f32(0.0, 5.0, 0.0);
        let displacement = Vec3Fix::from_f32(0.0, -10.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_some(), "Should detect ground collision");

        let toi = result.unwrap();
        let t = toi.t.to_f32();
        // Start at y=5, ground at y=0, radius=0.5, contact at y=0.5
        // t = (5 - 0.5) / 10 = 0.45
        assert!(t > 0.3 && t < 0.6, "TOI should be ~0.45, got {t}");
    }

    #[test]
    fn test_ray_march() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let origin = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
        let direction = Vec3Fix::UNIT_X;
        let max_dist = Fix128::from_int(20);
        let config = SdfCcdConfig::default();

        let result = ray_march_sdf(origin, direction, max_dist, &sdf, &config);
        assert!(result.is_some(), "Ray should hit sphere");
    }

    #[test]
    fn batch_sphere_trace_sdf_filters_bodies_and_sorts_by_closed_form_toi() {
        use crate::solver::RigidBody;
        let colliders = [
            // 0: 地面 y = 0
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY),
            // 1: 単位球 (原点)
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY),
            // 2: 単位球を body 3 に attach (body 3 に対しては self-collision skip)
            SdfCollider::new_dynamic(Box::new(unit_sphere()), 3),
        ];
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default(); // velocity_threshold 5、tolerance 0.001

        let fast_down = Vec3Fix::from_int(0, -10, 0);
        let fast_right = Vec3Fix::from_int(10, 0, 0);
        let mut bodies = vec![
            // 0: (0, 5, 0) → 下へ 10: 球 top 接触 y = 1.5 → t 0.35 (sdf 1 と 2)、地面 y = 0.5 → t 0.45
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE)
                .with_velocity(fast_down),
            // 1: (-5, 5, 0) → 右へ 10: 球も地面も miss
            RigidBody::new_dynamic(Vec3Fix::from_int(-5, 5, 0), Fix128::ONE)
                .with_velocity(fast_right),
            // 2: static は速くても skip
            RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)).with_velocity(fast_down),
            // 3: (-7, 0, 0) → 右へ 10: 地面は既に貫入 (gap -0.5) → t 0、球 x = -1.5 → t 0.55、sdf 2 は self skip
            RigidBody::new_dynamic(Vec3Fix::from_int(-7, 0, 0), Fix128::ONE)
                .with_velocity(fast_right),
            // 4: body 0 と同じ軌道だが速度 1 < threshold 5 → skip
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE)
                .with_velocity(Vec3Fix::from_int(0, -1, 0)),
        ];
        let displacements = [fast_down, fast_right, fast_down, fast_right, fast_down];

        let hits = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &config);
        let keys: Vec<(usize, usize)> = hits.iter().map(|(b, s, _)| (*b, *s)).collect();
        // t 昇順、同値 (body 0 の sdf 1 / 2 は同じ field) は stable sort で挿入順
        assert_eq!(keys, vec![(3, 0), (0, 1), (0, 2), (0, 0), (3, 1)]);
        let expected_t = [0.0_f32, 0.35, 0.35, 0.45, 0.55];
        for (k, want) in expected_t.iter().enumerate() {
            let t = hits[k].2.t.to_f32();
            assert!((t - want).abs() < 1e-3, "hit {k} t {t}, want {want}");
        }
        // 法線: 地面 +Y、球 top +Y、球の -x 側 -X
        let n = |k: usize| hits[k].2.normal.to_f32();
        for k in [0usize, 1, 2, 3] {
            let (nx, ny, nz) = n(k);
            assert!(
                nx.abs() < 1e-4 && (ny - 1.0).abs() < 1e-4 && nz.abs() < 1e-4,
                "hit {k}"
            );
        }
        let (nx, ny, nz) = n(4);
        assert!((nx + 1.0).abs() < 1e-4 && ny.abs() < 1e-4 && nz.abs() < 1e-4);
        // 接触点は SDF 表面上: 地面 hit の y ≈ 0、球 hit は |p| ≈ 1
        let (_, py, _) = hits[3].2.point.to_f32();
        assert!(py.abs() < 2e-3, "ground contact point y {py}");
        for k in [1usize, 2, 4] {
            let (px, py, pz) = hits[k].2.point.to_f32();
            let r = (px * px + py * py + pz * pz).sqrt();
            assert!((r - 1.0).abs() < 2e-3, "hit {k} contact point radius {r}");
        }

        // velocity_threshold を 0 にすると body 4 も (body 0 と同じ 3 hit で) 拾われる
        let lenient = SdfCcdConfig {
            velocity_threshold: Fix128::ZERO,
            ..SdfCcdConfig::default()
        };
        let all = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &lenient);
        assert_eq!(all.len(), 8);
        assert_eq!(all.iter().filter(|(b, _, _)| *b == 4).count(), 3);
        // zip: displacement が足りない body は評価されない
        let short =
            batch_sphere_trace_sdf(&bodies, &displacements[..1], radius, &colliders, &config);
        assert_eq!(short.len(), 3);
        assert!(short.iter().all(|(b, _, _)| *b == 0));
        // body 3 を static にすると (3, *) が消える
        bodies[3].inv_mass = Fix128::ZERO;
        let no3 = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &config);
        assert_eq!(no3.len(), 3);
        assert!(no3.iter().all(|(b, _, _)| *b == 0));
    }
}
