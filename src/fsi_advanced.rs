//! Advanced Fluid-Structure Interaction (Deformable / Articulation Coupling)
//:
//! Phase G3 of the ALICE-Physics completeness project. Extends the existing
//! `cloth_fluid` two-way coupling module to work with **deformable** solids
//! (`deformable.rs`) and **articulated** bodies (`articulation.rs`).
//!
//! # Scope
//!
//! Provides the two building-block force computations that any FSI coupling
//! needs — pushed one abstraction level up so that arbitrary solid types
//! (rigid, deformable, articulated) can be handled uniformly by supplying a
//! set of "sample points" (world-space positions with local drag areas /
//! displaced volumes).
//!
//! - **Quadratic drag**: `F_d = -½·ρ·C_d·A·|v_rel|·v_rel`
//! - **Buoyancy**: `F_b = ρ · V · g_up`
//! - **Aggregate torque** from an offset point.
//!
//! Bring your own fluid sampler (level-set / VOF / PIC velocity field). The
//! coupling is one-way by default; call [`react_back_pressure`] to close the
//! loop by depositing the reaction force back into the fluid grid.
//!
//! # References
//!
//! - Bao et al., "An immersed boundary method with divergence-free velocity
//!   interpolation and force spreading", J. Comp. Phys. 347, 2017.
//! - Peskin, "The immersed boundary method", Acta Numerica 11, 2002.

use crate::math::{Fix128, Vec3Fix};

// ============================================================================
// Sample point
// ============================================================================

/// A single sample point used to couple a solid to a fluid.
///
/// Solids are represented as a bag of sample points; the API doesn't care
/// whether they come from rigid-body surface samples, deformable-body
/// nodes, or articulation link centres. Each sample carries its local
/// drag area (m²) and displaced volume (m³).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolidSample {
    /// World-space position (m).
    pub position: Vec3Fix,
    /// Velocity of the solid at this point (m/s).
    pub velocity: Vec3Fix,
    /// Effective drag area normal to the flow (m²).
    pub area_m2: Fix128,
    /// Displaced fluid volume attributed to this sample (m³).
    pub volume_m3: Fix128,
}

// ============================================================================
// Drag
// ============================================================================

/// Quadratic drag force at one sample: `F_d = -½·ρ·C_d·A·|v_rel|·v_rel`.
#[must_use]
pub fn drag_force(
    sample: &SolidSample,
    fluid_velocity: Vec3Fix,
    fluid_density: Fix128,
    drag_coefficient: Fix128,
) -> Vec3Fix {
    let v_rel = Vec3Fix::new(
        sample.velocity.x - fluid_velocity.x,
        sample.velocity.y - fluid_velocity.y,
        sample.velocity.z - fluid_velocity.z,
    );
    let mag_sq = v_rel.x * v_rel.x + v_rel.y * v_rel.y + v_rel.z * v_rel.z;
    if mag_sq.is_zero() {
        return Vec3Fix::default();
    }
    let mag = mag_sq.sqrt();
    // F = -0.5·ρ·C_d·A·|v_rel|·v_rel  →  a scalar prefactor times v_rel
    let prefactor =
        Fix128::from_ratio(-1, 2) * fluid_density * drag_coefficient * sample.area_m2 * mag;
    Vec3Fix::new(
        prefactor * v_rel.x,
        prefactor * v_rel.y,
        prefactor * v_rel.z,
    )
}

// ============================================================================
// Buoyancy
// ============================================================================

/// Buoyancy force `F_b = ρ · V · |g|` along +Y (world "up" convention).
#[must_use]
pub fn buoyancy_force(
    sample: &SolidSample,
    fluid_density: Fix128,
    gravity_m_per_s2: Fix128,
) -> Vec3Fix {
    let mag = fluid_density * sample.volume_m3 * gravity_m_per_s2;
    Vec3Fix::new(Fix128::ZERO, mag, Fix128::ZERO)
}

// ============================================================================
// Aggregation
// ============================================================================

/// Aggregate the drag + buoyancy forces of all samples into one net force.
/// Returns `(F_net, torque_net)` about the body's `reference_point`.
#[must_use]
pub fn aggregate_forces(
    samples: &[SolidSample],
    fluid_velocity_sampler: impl Fn(Vec3Fix) -> Vec3Fix,
    fluid_density: Fix128,
    drag_coefficient: Fix128,
    gravity_m_per_s2: Fix128,
    reference_point: Vec3Fix,
) -> (Vec3Fix, Vec3Fix) {
    let mut f_sum = Vec3Fix::default();
    let mut t_sum = Vec3Fix::default();
    for s in samples {
        let v_fluid = fluid_velocity_sampler(s.position);
        let f_d = drag_force(s, v_fluid, fluid_density, drag_coefficient);
        let f_b = buoyancy_force(s, fluid_density, gravity_m_per_s2);
        let f_total = Vec3Fix::new(f_d.x + f_b.x, f_d.y + f_b.y, f_d.z + f_b.z);
        f_sum = Vec3Fix::new(
            f_sum.x + f_total.x,
            f_sum.y + f_total.y,
            f_sum.z + f_total.z,
        );
        // torque = r × F where r = s.position − reference_point
        let rx = s.position.x - reference_point.x;
        let ry = s.position.y - reference_point.y;
        let rz = s.position.z - reference_point.z;
        let tx = ry * f_total.z - rz * f_total.y;
        let ty = rz * f_total.x - rx * f_total.z;
        let tz = rx * f_total.y - ry * f_total.x;
        t_sum = Vec3Fix::new(t_sum.x + tx, t_sum.y + ty, t_sum.z + tz);
    }
    (f_sum, t_sum)
}

/// Deposit the reaction force `-F_solid` into a fluid grid at the position
/// of `sample`. The caller supplies a mutable closure that maps a world
/// position + force back into whatever data structure it uses (usually the
/// `EulerianGrid` P2G routine). Placeholder for the true immersed-boundary
/// scatter operator; kept minimal to avoid coupling this module to any
/// specific grid representation.
pub fn react_back_pressure(
    samples: &[SolidSample],
    solid_force_by_sample: &[Vec3Fix],
    mut deposit: impl FnMut(Vec3Fix, Vec3Fix),
) {
    for (s, f) in samples.iter().zip(solid_force_by_sample.iter()) {
        let reaction = Vec3Fix::new(Fix128::ZERO - f.x, Fix128::ZERO - f.y, Fix128::ZERO - f.z);
        deposit(s.position, reaction);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn drag_zero_relative_velocity_zero_force() {
        let s = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ONE,
            volume_m3: Fix128::ZERO,
        };
        let f = drag_force(&s, Vec3Fix::default(), Fix128::from_int(1000), Fix128::ONE);
        assert_eq!(f, Vec3Fix::default());
    }

    #[test]
    fn drag_opposes_solid_motion() {
        // Solid moving +X in still fluid → drag along -X
        let s = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
            area_m2: Fix128::ONE,
            volume_m3: Fix128::ZERO,
        };
        let f = drag_force(&s, Vec3Fix::default(), Fix128::from_int(1000), Fix128::ONE);
        assert!(f.x < Fix128::ZERO);
    }

    #[test]
    fn drag_quadratic_in_speed() {
        let s1 = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
            area_m2: Fix128::ONE,
            volume_m3: Fix128::ZERO,
        };
        let s2 = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            area_m2: Fix128::ONE,
            volume_m3: Fix128::ZERO,
        };
        let f1 = drag_force(&s1, Vec3Fix::default(), Fix128::from_int(1000), Fix128::ONE);
        let f2 = drag_force(&s2, Vec3Fix::default(), Fix128::from_int(1000), Fix128::ONE);
        // 2× speed → 4× drag
        let ratio = f2.x / f1.x;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn buoyancy_scales_with_volume() {
        let s1 = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::ONE,
        };
        let s2 = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::from_int(2),
        };
        let f1 = buoyancy_force(&s1, Fix128::from_int(1000), Fix128::from_int(10));
        let f2 = buoyancy_force(&s2, Fix128::from_int(1000), Fix128::from_int(10));
        assert!(f2.y > f1.y);
    }

    #[test]
    fn buoyancy_always_up() {
        let s = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::ONE,
        };
        let f = buoyancy_force(&s, Fix128::from_int(1000), Fix128::from_int(10));
        assert_eq!(f.x, Fix128::ZERO);
        assert!(f.y > Fix128::ZERO);
        assert_eq!(f.z, Fix128::ZERO);
    }

    #[test]
    fn aggregate_empty_samples_zero_force() {
        let (f, t) = aggregate_forces(
            &[],
            |_| Vec3Fix::default(),
            Fix128::from_int(1000),
            Fix128::ONE,
            Fix128::from_int(10),
            Vec3Fix::default(),
        );
        assert_eq!(f, Vec3Fix::default());
        assert_eq!(t, Vec3Fix::default());
    }

    #[test]
    fn aggregate_single_sample_buoyancy_only() {
        // Sample at rest, buoyant → force = ρ·V·g in +Y
        let samples = vec![SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::ONE,
        }];
        let (f, _) = aggregate_forces(
            &samples,
            |_| Vec3Fix::default(),
            Fix128::from_int(1000),
            Fix128::ONE,
            Fix128::from_int(10),
            Vec3Fix::default(),
        );
        // F_y = 1000 · 1 · 10 = 10000
        assert_eq!(f.y, Fix128::from_int(10_000));
    }

    #[test]
    fn aggregate_torque_offset_sample() {
        // Sample at (1, 0, 0), buoyancy +Y = 10 → torque = r × F = (0, 0, -10)
        // Wait: r = (1,0,0), F = (0,10,0), τ = (0·0-0·10, 0·0-1·0, 1·10-0·0) = (0, 0, 10)
        let samples = vec![SolidSample {
            position: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::from_ratio(1, 100),
        }];
        let (_, t) = aggregate_forces(
            &samples,
            |_| Vec3Fix::default(),
            Fix128::from_int(1000),
            Fix128::ONE,
            Fix128::ONE,
            Vec3Fix::default(),
        );
        // buoyancy F.y = 1000·0.01·1 = 10 → τ.z = 1·10 = 10
        assert!(approx_eq(
            t.z,
            Fix128::from_int(10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn react_back_pressure_calls_closure_negatively() {
        let sample = SolidSample {
            position: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ZERO,
            volume_m3: Fix128::ZERO,
        };
        let force = Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO);
        let mut received_pos: Option<Vec3Fix> = None;
        let mut received_force: Option<Vec3Fix> = None;
        react_back_pressure(&[sample], &[force], |p, f| {
            received_pos = Some(p);
            received_force = Some(f);
        });
        assert_eq!(received_pos.unwrap(), sample.position);
        assert_eq!(received_force.unwrap().x, Fix128::from_int(-5));
    }

    #[test]
    fn drag_uses_fluid_velocity() {
        // Fluid moving +X at 5, solid stationary → v_rel = -5X → drag +X
        let s = SolidSample {
            position: Vec3Fix::default(),
            velocity: Vec3Fix::default(),
            area_m2: Fix128::ONE,
            volume_m3: Fix128::ZERO,
        };
        let vf = Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO);
        let f = drag_force(&s, vf, Fix128::from_int(1000), Fix128::ONE);
        assert!(f.x > Fix128::ZERO);
    }
}
