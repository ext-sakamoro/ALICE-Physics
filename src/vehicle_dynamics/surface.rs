//! Road geometry (where is the ground) and road condition (how much grip).

use crate::anisotropic_friction::AnisotropicFriction;
use crate::heightfield::HeightField;
use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfField;
use crate::trimesh::TriMesh;

/// Result of a wheel probe against the road.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundHit {
    /// Distance along the probe direction from the probe origin (m).
    pub distance: Fix128,
    /// World-space contact point.
    pub point: Vec3Fix,
    /// Unit surface normal at the contact, pointing out of the road.
    pub normal: Vec3Fix,
}

/// Road geometry seen by the wheels.
///
/// Contract: `probe(origin, dir, max_dist)` returns the first road point on
/// the segment `origin + t dir`, `0 ≤ t ≤ max_dist` (`dir` is unit length),
/// or `None`. An origin already below the surface returns `distance = 0`
/// with the point projected onto the surface (a bottomed-out wheel keeps its
/// contact instead of falling through).
pub trait RoadSurface {
    /// Probe the road along `dir` from `origin`.
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit>;
}

/// Infinite horizontal plane `y = height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlatGround {
    /// Plane height (m).
    pub height: Fix128,
}

/// Infinite plane through `point` with unit `normal` (slopes, banked roads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InclinedPlane {
    /// Any point on the plane.
    pub point: Vec3Fix,
    /// Unit normal pointing out of the road.
    pub normal: Vec3Fix,
}

/// Height-field terrain.
#[derive(Clone, Copy, Debug)]
pub struct HeightFieldRoad<'a> {
    /// The terrain.
    pub field: &'a HeightField,
}

/// Triangle-mesh road.
#[derive(Clone, Copy)]
pub struct TriMeshRoad<'a> {
    /// The mesh (normals of the hit face are used as the road normal).
    pub mesh: &'a TriMesh,
}

/// Signed-distance-field road (sphere-traced).
pub struct SdfRoad<'a> {
    /// The field; the road is its zero level set.
    pub field: &'a dyn SdfField,
    /// Hit tolerance (m).
    pub tolerance: Fix128,
    /// Maximum tracing steps.
    pub max_steps: u32,
}

/// Weather state of the road surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weather {
    /// Dry surface.
    Dry,
    /// Wet surface with a water film (`water_depth_mm` controls hydroplaning).
    Wet {
        /// Water film depth (mm).
        water_depth_mm: Fix128,
    },
    /// Packed snow.
    Snow,
    /// Ice.
    Ice,
}

/// Road material + weather + rolling resistance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoadCondition {
    /// Grip of the dry road material.
    pub material: AnisotropicFriction,
    /// Current weather.
    pub weather: Weather,
    /// Rolling-resistance coefficient `C_rr` (force `C_rr F_z` against rolling).
    pub rolling_resistance: Fix128,
}

impl RoadCondition {
    /// Dry asphalt, `C_rr = 0.012`.
    #[must_use]
    pub fn dry_asphalt() -> Self {
        todo!("STUB: RoadCondition::dry_asphalt")
    }

    /// Grip multiplier of the weather alone (dry = 1). The table and its
    /// source are documented on the implementation.
    #[must_use]
    pub fn weather_factor(&self) -> Fix128 {
        todo!("STUB: RoadCondition::weather_factor")
    }

    /// Effective grip at a contact moving at `speed` (m/s) with tyre
    /// inflation `tyre_pressure_kpa`: material × weather × hydroplaning loss.
    #[must_use]
    pub fn grip(&self, speed: Fix128, tyre_pressure_kpa: Fix128) -> AnisotropicFriction {
        let _ = (speed, tyre_pressure_kpa);
        todo!("STUB: RoadCondition::grip")
    }
}

/// Speed (m/s) at which a tyre at `tyre_pressure_kpa` starts to hydroplane on
/// a flooded road (Horne: `V[km/h] = 6.35 √p[kPa]`).
#[must_use]
pub fn hydroplaning_onset_speed(tyre_pressure_kpa: Fix128) -> Fix128 {
    let _ = tyre_pressure_kpa;
    todo!("STUB: hydroplaning_onset_speed")
}

impl RoadSurface for FlatGround {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let _ = (origin, dir, max_dist);
        todo!("STUB: FlatGround::probe")
    }
}

impl RoadSurface for InclinedPlane {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let _ = (origin, dir, max_dist);
        todo!("STUB: InclinedPlane::probe")
    }
}

impl RoadSurface for HeightFieldRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let _ = (origin, dir, max_dist);
        todo!("STUB: HeightFieldRoad::probe")
    }
}

impl RoadSurface for TriMeshRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let _ = (origin, dir, max_dist);
        todo!("STUB: TriMeshRoad::probe")
    }
}

impl RoadSurface for SdfRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let _ = (origin, dir, max_dist);
        todo!("STUB: SdfRoad::probe")
    }
}
