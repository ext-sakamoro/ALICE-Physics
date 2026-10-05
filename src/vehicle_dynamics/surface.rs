//! Road geometry (where is the ground) and road condition (how much grip).

use crate::anisotropic_friction::AnisotropicFriction;
use crate::heightfield::HeightField;
use crate::math::{Fix128, Vec3Fix};
use crate::raycast::Ray;
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
///
/// Winding contract: every triangle is given so that its outward normal
/// (the side the wheels drive on) is the counter-clockwise normal
/// `(v1 − v0) × (v2 − v0)`. The reported normal is always this outward
/// normal, never flipped towards the probe.
///
/// Inside / below test: the probe line is traced both ways. If the nearest
/// face behind the origin (along `−dir`, up to the mesh bounds) or the first
/// face ahead of it (along `dir`, up to `max_dist`) is reached from its back
/// side, the origin lies on the inner side of that face and the probe returns
/// `distance = 0` with the point on that face (projection along the probe
/// line) and the face's outward normal. Consequences of the contract:
///
/// - a mesh wound the other way round is a road whose drivable side faces
///   down: probing it from above returns `distance = 0` with a downward
///   normal, probing it from below finds no road
/// - a single-sided surface above the wheel (an open bridge deck wound
///   upwards) is taken as the road the wheel is under; model overhead
///   structures as closed meshes
#[derive(Clone, Copy)]
pub struct TriMeshRoad<'a> {
    /// The mesh (normals of the hit face are used as the road normal).
    pub mesh: &'a TriMesh,
}

/// Signed-distance-field road (sphere-traced).
///
/// Sphere tracing: `t ← t + d(origin + t dir)` until `d ≤ tolerance` (hit),
/// `t > max_dist` or `max_steps` steps (no hit). The `f32` conversion happens
/// only at the [`SdfField`] call; the march itself is `Fix128`. A non-finite
/// distance from the field is treated as no hit. The returned point is the
/// last sample projected along the field normal (`p − n d`), the distance is
/// the ray parameter of that sample (within `tolerance / cos θ` of the true
/// crossing, `θ` the incidence angle). An origin with `d ≤ 0` returns
/// `distance = 0` and `origin − n d`. A degenerate (zero) field normal falls
/// back to `−dir`.
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

/// Water depth (mm) from which a wet road is treated as flooded and
/// hydroplaning can occur. Horne's onset formula is for a flooded surface,
/// with water deeper than the tyre tread can drain; 2.5 mm (≈ 0.1 in) is the
/// order of magnitude usually quoted for that. Model choice of this module.
pub const HYDROPLANING_MIN_DEPTH_MM: Fix128 = Fix128 {
    hi: 2,
    lo: 0x8000_0000_0000_0000,
};

/// Lowest hydroplaning grip factor (model floor, see [`RoadCondition::grip`]).
fn hydroplaning_floor() -> Fix128 {
    Fix128::from_ratio(1, 10)
}

impl RoadCondition {
    /// Dry asphalt, `C_rr = 0.012`.
    #[must_use]
    pub fn dry_asphalt() -> Self {
        Self {
            material: AnisotropicFriction::tyre_asphalt(),
            weather: Weather::Dry,
            rolling_resistance: Fix128::from_ratio(12, 1000),
        }
    }

    /// Grip multiplier of the weather alone (dry = 1).
    ///
    /// | weather | factor |
    /// |---|---|
    /// | dry | 1 |
    /// | wet | 7/10 |
    /// | packed snow | 6/25 (0.24) |
    /// | ice | 3/25 (0.12) |
    ///
    /// Ratios of the typical peak tyre–road coefficients in J. Y. Wong,
    /// *Theory of Ground Vehicles* (4th ed., 2008), Table 1.3: asphalt dry
    /// 0.8–0.9, asphalt wet 0.5–0.7, hard-packed snow 0.2, ice 0.1, each
    /// divided by the dry mid value 0.85 and rounded (wet 0.71 → 0.7,
    // LIMITATION(COV-MBD-046): One factor scales static and kinetic coefficients alike
    /// snow 0.235 → 0.24, ice 0.118 → 0.12). One factor scales static and
    /// kinetic coefficients alike (the table's sliding/peak ratios differ by
    /// a few per cent between surfaces; that difference is not modelled).
    #[must_use]
    pub fn weather_factor(&self) -> Fix128 {
        match self.weather {
            Weather::Dry => Fix128::ONE,
            Weather::Wet { .. } => Fix128::from_ratio(7, 10),
            Weather::Snow => Fix128::from_ratio(6, 25),
            Weather::Ice => Fix128::from_ratio(3, 25),
        }
    }

    /// Effective grip at a contact moving at `speed` (m/s) with tyre
    /// inflation `tyre_pressure_kpa`: material × weather × hydroplaning loss.
    ///
    // LIMITATION(COV-MBD-047): Hydroplaning loss (a model, not a measured law)
    /// Hydroplaning loss (a model, not a measured law): on a wet road with
    /// `water_depth_mm ≥` [`HYDROPLANING_MIN_DEPTH_MM`] and
    /// `|speed| > V_p =` [`hydroplaning_onset_speed`]`(p)`, the factor is
    /// `max(1/10, (V_p / |speed|)²)`; otherwise 1. Rationale: the
    /// hydrodynamic lift grows with `V²` and equals the wheel load at `V_p`,
    /// so the share of the load still carried by dry contact falls like
    /// `(V_p / V)²`; the floor keeps a residual viscous grip. The factor is
    /// continuous at `V_p`. A negative depth counts as no standing water.
    /// All four coefficients (static and kinetic) are scaled; the slip
    /// threshold is unchanged.
    #[must_use]
    pub fn grip(&self, speed: Fix128, tyre_pressure_kpa: Fix128) -> AnisotropicFriction {
        let mut k = self.weather_factor();
        if let Weather::Wet { water_depth_mm } = self.weather {
            if water_depth_mm >= HYDROPLANING_MIN_DEPTH_MM {
                let v = speed.abs();
                let vp = hydroplaning_onset_speed(tyre_pressure_kpa);
                if v > vp {
                    let q = vp / v;
                    let loss = (q * q).max(hydroplaning_floor());
                    k = k * loss;
                }
            }
        }
        let m = self.material;
        AnisotropicFriction {
            longitudinal_static: m.longitudinal_static * k,
            longitudinal_kinetic: m.longitudinal_kinetic * k,
            transverse_static: m.transverse_static * k,
            transverse_kinetic: m.transverse_kinetic * k,
            slip_threshold_m_s: m.slip_threshold_m_s,
        }
    }
}

/// Speed (m/s) at which a tyre at `tyre_pressure_kpa` starts to hydroplane on
/// a flooded road (Horne: `V[km/h] = 6.35 √p[kPa]`, i.e. `V[m/s] =
/// (6.35 / 3.6) √p`; W. B. Horne, U. T. Joyner, NASA TN D-2056 / SAE 650145).
/// A non-positive pressure returns 0.
#[must_use]
pub fn hydroplaning_onset_speed(tyre_pressure_kpa: Fix128) -> Fix128 {
    if tyre_pressure_kpa <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    Fix128::from_ratio(635, 360) * tyre_pressure_kpa.sqrt()
}

/// Shared plane probe: plane through `p` with unit normal `n`.
fn probe_plane(
    p: Vec3Fix,
    n: Vec3Fix,
    origin: Vec3Fix,
    dir: Vec3Fix,
    max_dist: Fix128,
) -> Option<GroundHit> {
    if max_dist < Fix128::ZERO {
        return None;
    }
    let s = (origin - p).dot(n);
    if s <= Fix128::ZERO {
        // On or below the plane: project along the normal.
        return Some(GroundHit {
            distance: Fix128::ZERO,
            point: origin - n * s,
            normal: n,
        });
    }
    let approach = -dir.dot(n);
    if approach <= Fix128::ZERO {
        return None;
    }
    // `s > approach · max_dist` ⇔ `t > max_dist`, tested without dividing so
    // a nearly parallel `dir` cannot overflow `t`.
    if s > approach * max_dist {
        return None;
    }
    let t = s / approach;
    Some(GroundHit {
        distance: t,
        point: origin + dir * t,
        normal: n,
    })
}

impl RoadSurface for FlatGround {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        let p = Vec3Fix::new(Fix128::ZERO, self.height, Fix128::ZERO);
        let mut hit = probe_plane(p, Vec3Fix::UNIT_Y, origin, dir, max_dist)?;
        // The contact lies on y = height exactly.
        hit.point.y = self.height;
        Some(hit)
    }
}

impl RoadSurface for InclinedPlane {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        probe_plane(self.point, self.normal, origin, dir, max_dist)
    }
}

/// March step of the height-field probe, in grid spacings.
const HF_STEPS_PER_CELL: i64 = 4;
/// Bisection iterations of the height-field probe.
const HF_BISECT_ITERS: u32 = 64;

impl HeightFieldRoad<'_> {
    /// Grid XZ extent `(x_min, x_max, z_min, z_max)`, or `None` when the field
    /// has no surface (fewer than 2×2 points or non-positive spacing).
    fn extent(&self) -> Option<(Fix128, Fix128, Fix128, Fix128)> {
        let f = self.field;
        if f.width < 2 || f.depth < 2 || f.spacing <= Fix128::ZERO {
            return None;
        }
        let w = f.spacing * Fix128::from_int(i64::from(f.width - 1));
        let d = f.spacing * Fix128::from_int(i64::from(f.depth - 1));
        Some((f.origin.x, f.origin.x + w, f.origin.z, f.origin.z + d))
    }

    /// Surface normal from `sample_height` differences of half a spacing,
    /// clamped to the grid (one-sided at the border, where
    /// `HeightField::sample_normal` would read clamped samples and halve the
    /// slope).
    fn normal_at(&self, x: Fix128, z: Fix128, ext: (Fix128, Fix128, Fix128, Fix128)) -> Vec3Fix {
        let f = self.field;
        let eps = f.spacing.half();
        let x0 = (x - eps).max(ext.0);
        let x1 = (x + eps).min(ext.1);
        let z0 = (z - eps).max(ext.2);
        let z1 = (z + eps).min(ext.3);
        let dhdx = (f.sample_height(x1, z) - f.sample_height(x0, z)) / (x1 - x0);
        let dhdz = (f.sample_height(x, z1) - f.sample_height(x, z0)) / (z1 - z0);
        Vec3Fix::new(-dhdx, Fix128::ONE, -dhdz).normalize()
    }

    fn hit_at(
        &self,
        origin: Vec3Fix,
        dir: Vec3Fix,
        t: Fix128,
        ext: (Fix128, Fix128, Fix128, Fix128),
    ) -> GroundHit {
        let p = origin + dir * t;
        let y = self.field.sample_height(p.x, p.z);
        GroundHit {
            distance: t,
            point: Vec3Fix::new(p.x, y, p.z),
            normal: self.normal_at(p.x, p.z, ext),
        }
    }
}

/// Clip `[t0, t1]` to `lo ≤ o + t d ≤ hi`; `None` when empty.
fn clip_slab(
    t0: Fix128,
    t1: Fix128,
    o: Fix128,
    d: Fix128,
    lo: Fix128,
    hi: Fix128,
) -> Option<(Fix128, Fix128)> {
    if d.is_zero() {
        return if o < lo || o > hi {
            None
        } else {
            Some((t0, t1))
        };
    }
    let ta = (lo - o) / d;
    let tb = (hi - o) / d;
    let (near, far) = if ta <= tb { (ta, tb) } else { (tb, ta) };
    let a = t0.max(near);
    let b = t1.min(far);
    if a > b {
        None
    } else {
        Some((a, b))
    }
}

/// Height-field road.
///
/// The surface is `y = HeightField::sample_height(x, z)` over the grid's XZ
/// extent `[origin.x, origin.x + (width−1)·spacing] × [origin.z, …]`; it has
/// no side walls and nothing outside the extent. `origin.y` is handled
/// exactly as `sample_height` handles it (no offset of its own). Fields with
/// fewer than 2×2 points or `spacing ≤ 0` have no surface.
///
/// Method: the probe segment is clipped to the extent, then marched in steps
/// of `spacing / 4` looking for `f(t) = y(t) − h(x(t), z(t))` turning from
/// positive to `≤ 0`; the bracketing step is bisected 64 times (the step
/// shrinks below the `Fix128` resolution). Two crossings closer than one step
/// (a ridge thinner than a quarter cell along the ray) can be missed. A ray
/// entering the extent already below the surface finds no road. At most
/// `4·(width + depth)` steps per probe. The normal comes from
/// half-spacing differences of `sample_height`, one-sided at the border.
impl RoadSurface for HeightFieldRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        if max_dist < Fix128::ZERO {
            return None;
        }
        let ext = self.extent()?;
        let f = self.field;
        let gap = |t: Fix128| {
            let p = origin + dir * t;
            p.y - f.sample_height(p.x, p.z)
        };
        let inside =
            origin.x >= ext.0 && origin.x <= ext.1 && origin.z >= ext.2 && origin.z <= ext.3;
        if inside && gap(Fix128::ZERO) <= Fix128::ZERO {
            return Some(self.hit_at(origin, dir, Fix128::ZERO, ext));
        }
        let (t0, t1) = clip_slab(Fix128::ZERO, max_dist, origin.x, dir.x, ext.0, ext.1)?;
        let (t0, t1) = clip_slab(t0, t1, origin.z, dir.z, ext.2, ext.3)?;
        let step = f.spacing / Fix128::from_int(HF_STEPS_PER_CELL);
        let mut a = t0;
        if gap(a) <= Fix128::ZERO {
            return None;
        }
        while a < t1 {
            let b = (a + step).min(t1);
            if gap(b) <= Fix128::ZERO {
                let (mut lo, mut hi) = (a, b);
                for _ in 0..HF_BISECT_ITERS {
                    let mid = lo + (hi - lo).half();
                    if gap(mid) <= Fix128::ZERO {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return Some(self.hit_at(origin, dir, hi, ext));
            }
            a = b;
        }
        None
    }
}

impl TriMeshRoad<'_> {
    /// Outward (winding) unit normal of triangle `i`.
    fn outward(&self, i: usize) -> Vec3Fix {
        self.mesh.triangles[i].unit_normal()
    }
}

impl RoadSurface for TriMeshRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        if max_dist < Fix128::ZERO || self.mesh.triangles.is_empty() {
            return None;
        }
        // Behind the origin, as far as the mesh reaches (per-axis bound on the
        // distance to the farthest bounds corner, no sqrt needed).
        let b = self.mesh.bounds;
        let reach = |o: Fix128, lo: Fix128, hi: Fix128| (o - lo).abs().max((hi - o).abs());
        let back_len = reach(origin.x, b.min.x, b.max.x)
            + reach(origin.y, b.min.y, b.max.y)
            + reach(origin.z, b.min.z, b.max.z);
        let back = Ray {
            origin,
            direction: -dir,
        };
        if let Some(h) = self.mesh.raycast(&back, back_len) {
            let n = self.outward(h.body_index);
            if n.dot(dir) < Fix128::ZERO {
                // The face behind us faces along −dir: we are under it.
                return Some(GroundHit {
                    distance: Fix128::ZERO,
                    point: h.point,
                    normal: n,
                });
            }
        }
        let fwd = Ray {
            origin,
            direction: dir,
        };
        let h = self.mesh.raycast(&fwd, max_dist)?;
        let n = self.outward(h.body_index);
        if n.dot(dir) > Fix128::ZERO {
            // Reached from the back: the origin is on the inner side.
            return Some(GroundHit {
                distance: Fix128::ZERO,
                point: h.point,
                normal: n,
            });
        }
        Some(GroundHit {
            distance: h.t,
            point: h.point,
            normal: n,
        })
    }
}

impl SdfRoad<'_> {
    /// Field distance and unit normal at `p`; `None` for a non-finite distance.
    fn sample(&self, p: Vec3Fix, dir: Vec3Fix) -> Option<(Fix128, Vec3Fix)> {
        let (x, y, z) = p.to_f32();
        let (d, (nx, ny, nz)) = self.field.distance_and_normal(x, y, z);
        if !d.is_finite() || !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return None;
        }
        let n = Vec3Fix::from_f32(nx, ny, nz)
            .try_normalize()
            .unwrap_or(-dir);
        Some((Fix128::from_f32(d), n))
    }
}

impl RoadSurface for SdfRoad<'_> {
    fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
        if max_dist < Fix128::ZERO {
            return None;
        }
        let (d0, n0) = self.sample(origin, dir)?;
        if d0 <= Fix128::ZERO {
            return Some(GroundHit {
                distance: Fix128::ZERO,
                point: origin - n0 * d0,
                normal: n0,
            });
        }
        let tol = self.tolerance.max(Fix128::ZERO);
        let mut t = Fix128::ZERO;
        for _ in 0..self.max_steps {
            let p = origin + dir * t;
            let (d, n) = self.sample(p, dir)?;
            if d <= tol {
                return Some(GroundHit {
                    distance: t,
                    point: p - n * d,
                    normal: n,
                });
            }
            t = t + d;
            if t > max_dist {
                return None;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    //! Expected values are closed forms written out in each test (or an
    //! independent f64 evaluation of the same closed form); none of them is
    //! produced by the functions under test.
    use super::*;
    use crate::sdf_collider::ClosureSdf;
    use crate::trimesh::Triangle;

    fn fx(n: i64) -> Fix128 {
        Fix128::from_int(n)
    }
    fn r(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }
    fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
        Vec3Fix::new(x, y, z)
    }
    fn down() -> Vec3Fix {
        Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO)
    }
    fn close(a: Fix128, b: f64, tol: f64) -> bool {
        (a.to_f64() - b).abs() <= tol
    }
    fn close_v(a: Vec3Fix, b: [f64; 3], tol: f64) -> bool {
        close(a.x, b[0], tol) && close(a.y, b[1], tol) && close(a.z, b[2], tol)
    }

    // ---------- FlatGround / InclinedPlane ----------

    #[test]
    fn flat_vertical_probe_is_height_difference() {
        // y0 − h = 5 − 1 = 4
        let g = FlatGround { height: fx(1) };
        let hit = g.probe(v(fx(3), fx(5), fx(-2)), down(), fx(10)).unwrap();
        assert_eq!(hit.distance, fx(4));
        assert_eq!(hit.point, v(fx(3), fx(1), fx(-2)));
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y);
    }

    #[test]
    fn flat_oblique_probe_matches_closed_form() {
        // dir = (3/5, −4/5, 0): t = (5 − 1) / (4/5) = 5, point = (3, 1, 0)
        let g = FlatGround { height: fx(1) };
        let dir = v(r(3, 5), r(-4, 5), Fix128::ZERO);
        let hit = g.probe(v(fx(0), fx(5), fx(0)), dir, fx(10)).unwrap();
        assert!(close(hit.distance, 5.0, 1e-15), "{}", hit.distance);
        assert!(close_v(hit.point, [3.0, 1.0, 0.0], 1e-15));
    }

    #[test]
    fn flat_origin_below_returns_zero_and_projects() {
        // origin y = −1 below y = 1: distance 0, point (2, 1, 7)
        let g = FlatGround { height: fx(1) };
        let hit = g.probe(v(fx(2), fx(-1), fx(7)), down(), fx(10)).unwrap();
        assert_eq!(hit.distance, Fix128::ZERO);
        assert_eq!(hit.point, v(fx(2), fx(1), fx(7)));
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y);
    }

    #[test]
    fn flat_out_of_range_and_upward_miss() {
        let g = FlatGround { height: fx(1) };
        // needs 4, only 3 allowed
        assert_eq!(g.probe(v(fx(0), fx(5), fx(0)), down(), fx(3)), None);
        // pointing away from the plane
        assert_eq!(
            g.probe(v(fx(0), fx(5), fx(0)), Vec3Fix::UNIT_Y, fx(100)),
            None
        );
    }

    #[test]
    fn degenerate_inputs_on_plane() {
        let g = FlatGround { height: fx(1) };
        // max_dist = 0, origin above: segment is the origin only, not on the road → None
        assert_eq!(g.probe(v(fx(0), fx(5), fx(0)), down(), Fix128::ZERO), None);
        // max_dist = 0, origin below: bottomed-out contact, distance 0
        let hit = g
            .probe(v(fx(0), fx(0), fx(0)), down(), Fix128::ZERO)
            .unwrap();
        assert_eq!(hit.distance, Fix128::ZERO);
        // horizontal dir (parallel to the plane) from above → None
        assert_eq!(
            g.probe(v(fx(0), fx(5), fx(0)), Vec3Fix::UNIT_X, fx(100)),
            None
        );
        // negative max_dist → None even below
        assert_eq!(g.probe(v(fx(0), fx(0), fx(0)), down(), fx(-1)), None);
        // origin exactly on the plane → distance 0 at the origin
        let on = g
            .probe(v(fx(4), fx(1), fx(4)), Vec3Fix::UNIT_X, fx(1))
            .unwrap();
        assert_eq!(on.distance, Fix128::ZERO);
        assert_eq!(on.point, v(fx(4), fx(1), fx(4)));
    }

    /// Slope `y = (3/4) x` through the origin: normal (−3/5, 4/5, 0).
    fn slope() -> InclinedPlane {
        InclinedPlane {
            point: Vec3Fix::ZERO,
            normal: v(r(-3, 5), r(4, 5), Fix128::ZERO),
        }
    }

    #[test]
    fn incline_vertical_probe_matches_closed_form() {
        // vertical drop to y = 0.75 x: t = y0 − 0.75 x0 = 10 − 3.75 = 6.25
        let hit = slope()
            .probe(v(fx(5), fx(10), fx(1)), down(), fx(20))
            .unwrap();
        assert!(close(hit.distance, 6.25, 1e-15), "{}", hit.distance);
        assert!(close_v(hit.point, [5.0, 3.75, 1.0], 1e-15));
        assert!(close_v(hit.normal, [-0.6, 0.8, 0.0], 1e-15));
    }

    #[test]
    fn incline_origin_below_projects_along_normal() {
        // origin (4,0,0) below y = 3: s = o·n = −2.4, projection o − s n = (2.56, 1.92, 0)
        let hit = slope()
            .probe(v(fx(4), fx(0), fx(0)), down(), fx(5))
            .unwrap();
        assert_eq!(hit.distance, Fix128::ZERO);
        assert!(close_v(hit.point, [2.56, 1.92, 0.0], 1e-15));
        // dir parallel to the slope from above → None
        let along = v(r(4, 5), r(3, 5), Fix128::ZERO);
        assert_eq!(slope().probe(v(fx(0), fx(1), fx(0)), along, fx(100)), None);
    }

    // ---------- HeightFieldRoad ----------

    /// Field sampling `h(x, z) = a x + b z + c` at its grid points.
    fn linear_field(
        a: Fix128,
        b: Fix128,
        c: Fix128,
        n: u32,
        spacing: Fix128,
        origin: Vec3Fix,
    ) -> HeightField {
        let mut hs = Vec::new();
        for iz in 0..n {
            for ix in 0..n {
                let x = origin.x + spacing * fx(i64::from(ix));
                let z = origin.z + spacing * fx(i64::from(iz));
                hs.push(a * x + b * z + c);
            }
        }
        HeightField::new(hs, n, n, spacing, origin)
    }

    /// Closed form for a ray against `y = a x + b z + c`.
    fn linear_t(o: [f64; 3], d: [f64; 3], a: f64, b: f64, c: f64) -> f64 {
        (o[1] - a * o[0] - b * o[2] - c) / (a * d[0] + b * d[2] - d[1])
    }

    #[test]
    fn heightfield_flat_matches_flat_ground() {
        let field = HeightField::flat(5, 5, fx(1), Vec3Fix::ZERO, fx(1));
        let road = HeightFieldRoad { field: &field };
        let o = v(r(23, 10), fx(5), r(17, 10));
        let hit = road.probe(o, down(), fx(10)).unwrap();
        // y0 − h = 4
        assert!(close(hit.distance, 4.0, 1e-15), "{}", hit.distance);
        assert!(close_v(hit.point, [2.3, 1.0, 1.7], 1e-15));
        assert!(close_v(hit.normal, [0.0, 1.0, 0.0], 1e-15));
    }

    #[test]
    fn heightfield_linear_oblique_matches_closed_form() {
        // h = x/2 + z/4 + 1 on a 9×9 grid, spacing 1/2 (exact under bilinear)
        let field = linear_field(r(1, 2), r(1, 4), fx(1), 9, r(1, 2), Vec3Fix::ZERO);
        let road = HeightFieldRoad { field: &field };
        let dir = v(r(3, 5), r(-4, 5), Fix128::ZERO);
        let o = [0.5, 6.0, 2.0];
        let hit = road.probe(v(r(1, 2), fx(6), fx(2)), dir, fx(20)).unwrap();
        let t = linear_t(o, [0.6, -0.8, 0.0], 0.5, 0.25, 1.0);
        assert!(close(hit.distance, t, 1e-12), "{} vs {t}", hit.distance);
        let p = [o[0] + 0.6 * t, o[1] - 0.8 * t, o[2]];
        assert!(close_v(hit.point, p, 1e-12));
        // normal (−a, 1, −b)/|·|
        let n = (0.25f64 + 1.0 + 0.0625).sqrt();
        assert!(close_v(hit.normal, [-0.5 / n, 1.0 / n, -0.25 / n], 1e-12));
    }

    #[test]
    fn heightfield_normal_at_grid_edge_keeps_full_slope() {
        // probe 0.1 inside the +x edge (x_max = 4): central difference with
        // eps = spacing/2 would read a clamped (flat) sample beyond the edge
        let field = linear_field(r(1, 2), r(1, 4), fx(1), 9, r(1, 2), Vec3Fix::ZERO);
        let road = HeightFieldRoad { field: &field };
        let hit = road
            .probe(v(r(39, 10), fx(10), fx(2)), down(), fx(20))
            .unwrap();
        let n = (0.25f64 + 1.0 + 0.0625).sqrt();
        assert!(
            close_v(hit.normal, [-0.5 / n, 1.0 / n, -0.25 / n], 1e-12),
            "{}",
            hit.normal
        );
    }

    #[test]
    fn heightfield_xz_origin_offset_matches_closed_form() {
        // same plane in world coordinates, grid min corner at (10, 0, −4)
        let origin = v(fx(10), Fix128::ZERO, fx(-4));
        let field = linear_field(r(1, 2), r(1, 4), fx(1), 9, r(1, 2), origin);
        let road = HeightFieldRoad { field: &field };
        let hit = road
            .probe(v(fx(12), fx(20), fx(-3)), down(), fx(30))
            .unwrap();
        // h(12, −3) = 6 − 0.75 + 1 = 6.25 → t = 13.75
        assert!(close(hit.distance, 13.75, 1e-12), "{}", hit.distance);
        // outside the offset grid's XZ extent [10, 14] × [−4, 0] → None
        assert_eq!(road.probe(v(fx(2), fx(20), fx(-3)), down(), fx(30)), None);
    }

    #[test]
    fn heightfield_origin_y_follows_sample_height() {
        // AUD-A-S4W3-006: HeightField ignores origin.y. The road follows the
        // field's own `sample_height` (whichever convention is settled), so
        // the expected surface height is read from that dependency.
        let field = HeightField::flat(5, 5, fx(1), v(fx(0), fx(7), fx(0)), fx(1));
        let road = HeightFieldRoad { field: &field };
        let surface_y = field.sample_height(fx(2), fx(2));
        let hit = road.probe(v(fx(2), fx(20), fx(2)), down(), fx(30)).unwrap();
        assert!(close(hit.distance, (fx(20) - surface_y).to_f64(), 1e-15));
        assert!(close(hit.point.y, surface_y.to_f64(), 1e-15));
    }

    #[test]
    fn heightfield_below_and_degenerate() {
        let field = HeightField::flat(5, 5, fx(1), Vec3Fix::ZERO, fx(1));
        let road = HeightFieldRoad { field: &field };
        // below: distance 0, projected vertically to y = 1
        let hit = road.probe(v(fx(2), fx(0), fx(2)), down(), fx(5)).unwrap();
        assert_eq!(hit.distance, Fix128::ZERO);
        assert!(close_v(hit.point, [2.0, 1.0, 2.0], 1e-15));
        // max_dist = 0 from above → None
        assert_eq!(
            road.probe(v(fx(2), fx(3), fx(2)), down(), Fix128::ZERO),
            None
        );
        // horizontal ray above the field → None
        assert_eq!(
            road.probe(v(fx(0), fx(3), fx(2)), Vec3Fix::UNIT_X, fx(10)),
            None
        );
        // empty field and zero spacing have no surface → None
        let empty = HeightField::new(Vec::new(), 0, 0, fx(1), Vec3Fix::ZERO);
        assert_eq!(
            HeightFieldRoad { field: &empty }.probe(v(fx(0), fx(3), fx(0)), down(), fx(10)),
            None
        );
        let flat0 = HeightField::flat(5, 5, Fix128::ZERO, Vec3Fix::ZERO, fx(1));
        assert_eq!(
            HeightFieldRoad { field: &flat0 }.probe(v(fx(0), fx(3), fx(0)), down(), fx(10)),
            None
        );
    }

    // ---------- TriMeshRoad ----------

    /// Square [−5, 5]² at y = 2, wound counter-clockwise seen from above.
    fn quad() -> TriMesh {
        let y = fx(2);
        let a = v(fx(-5), y, fx(-5));
        let b = v(fx(-5), y, fx(5));
        let c = v(fx(5), y, fx(5));
        let d = v(fx(5), y, fx(-5));
        TriMesh::from_triangles(vec![Triangle::new(a, b, c), Triangle::new(a, c, d)])
    }

    #[test]
    fn trimesh_quad_matches_closed_form() {
        let mesh = quad();
        let road = TriMeshRoad { mesh: &mesh };
        // dir (0, −4/5, 3/5) from (1, 6, −2): t = (6 − 2)/(4/5) = 5, point (1, 2, 1)
        let dir = v(Fix128::ZERO, r(-4, 5), r(3, 5));
        let hit = road.probe(v(fx(1), fx(6), fx(-2)), dir, fx(10)).unwrap();
        assert!(close(hit.distance, 5.0, 1e-15), "{}", hit.distance);
        assert!(close_v(hit.point, [1.0, 2.0, 1.0], 1e-15));
        assert!(close_v(hit.normal, [0.0, 1.0, 0.0], 1e-15));
        // too short / off the mesh → None
        assert_eq!(road.probe(v(fx(1), fx(6), fx(-2)), dir, fx(4)), None);
        assert_eq!(road.probe(v(fx(9), fx(6), fx(0)), down(), fx(10)), None);
    }

    #[test]
    fn trimesh_origin_below_returns_zero_with_outward_normal() {
        let mesh = quad();
        let road = TriMeshRoad { mesh: &mesh };
        // origin under the road: distance 0, point on the road above it, normal
        // the outward (winding) normal +y, not the one facing the reverse ray
        let hit = road.probe(v(fx(1), fx(0), fx(1)), down(), fx(10)).unwrap();
        assert_eq!(hit.distance, Fix128::ZERO);
        assert!(close_v(hit.point, [1.0, 2.0, 1.0], 1e-15));
        assert!(close_v(hit.normal, [0.0, 1.0, 0.0], 1e-15));
        // empty mesh → None
        let empty = TriMesh::from_triangles(Vec::new());
        assert_eq!(
            TriMeshRoad { mesh: &empty }.probe(v(fx(0), fx(3), fx(0)), down(), fx(10)),
            None
        );
        // horizontal ray above the quad → None
        assert_eq!(
            road.probe(v(fx(0), fx(3), fx(0)), Vec3Fix::UNIT_X, fx(10)),
            None
        );
    }

    #[test]
    fn trimesh_reversed_winding_is_a_road_facing_down() {
        // Same square wound clockwise from above: outward normal −y (winding
        // contract on TriMeshRoad). From above the origin is on the inner
        // side: distance 0, point on the face along the probe, normal −y.
        // From below there is no road under the origin.
        let y = fx(2);
        let a = v(fx(-5), y, fx(-5));
        let b = v(fx(-5), y, fx(5));
        let c = v(fx(5), y, fx(5));
        let d = v(fx(5), y, fx(-5));
        let mesh = TriMesh::from_triangles(vec![Triangle::new(a, c, b), Triangle::new(a, d, c)]);
        let road = TriMeshRoad { mesh: &mesh };
        let above = road.probe(v(fx(1), fx(6), fx(1)), down(), fx(10)).unwrap();
        assert_eq!(above.distance, Fix128::ZERO);
        assert!(close_v(above.point, [1.0, 2.0, 1.0], 1e-15));
        assert!(close_v(above.normal, [0.0, -1.0, 0.0], 1e-15));
        assert_eq!(road.probe(v(fx(1), fx(0), fx(1)), down(), fx(10)), None);
    }

    // ---------- SdfRoad ----------

    fn plane_sdf() -> ClosureSdf {
        // y = 1
        ClosureSdf::new(|_, y, _| y - 1.0, |_, _, _| (0.0, 1.0, 0.0))
    }
    fn sphere_sdf() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )
    }

    #[test]
    fn sdf_plane_matches_closed_form() {
        let f = plane_sdf();
        let road = SdfRoad {
            field: &f,
            tolerance: r(1, 10_000),
            max_steps: 64,
        };
        let hit = road.probe(v(fx(0), fx(4), fx(0)), down(), fx(10)).unwrap();
        // 4 − 1 = 3 (f32-exact)
        assert!(close(hit.distance, 3.0, 1e-4), "{}", hit.distance);
        assert!(close_v(hit.point, [0.0, 1.0, 0.0], 1e-6));
        assert!(close_v(hit.normal, [0.0, 1.0, 0.0], 1e-12));
        // below the plane → distance 0, projected to y = 1
        let below = road.probe(v(fx(2), fx(-1), fx(0)), down(), fx(10)).unwrap();
        assert_eq!(below.distance, Fix128::ZERO);
        assert!(close_v(below.point, [2.0, 1.0, 0.0], 1e-6));
    }

    #[test]
    fn sdf_sphere_matches_closed_form() {
        let f = sphere_sdf();
        let tol = 1e-4;
        let road = SdfRoad {
            field: &f,
            tolerance: r(1, 10_000),
            max_steps: 256,
        };
        let dir = v(Fix128::ZERO, Fix128::ZERO, Fix128::NEG_ONE);
        let hit = road.probe(v(r(1, 2), fx(0), fx(5)), dir, fx(10)).unwrap();
        // chord: z_hit = √(1 − 0.25) → t = 5 − √0.75; the trace stops within
        // tol of the surface, i.e. ≤ tol / cos(incidence) = tol / 0.866 along the ray
        let t = 5.0 - 0.75f64.sqrt();
        assert!(close(hit.distance, t, 2.0 * tol), "{} vs {t}", hit.distance);
        assert!(close_v(hit.normal, [0.5, 0.0, 0.75f64.sqrt()], 1e-3));
        // inside the sphere: distance 0, projected onto it along the normal
        let inside = road.probe(v(fx(0), fx(0), r(1, 2)), dir, fx(10)).unwrap();
        assert_eq!(inside.distance, Fix128::ZERO);
        assert!(close_v(inside.point, [0.0, 0.0, 1.0], 1e-6));
        // miss, too short, zero steps
        assert_eq!(road.probe(v(fx(3), fx(0), fx(5)), dir, fx(10)), None);
        assert_eq!(road.probe(v(r(1, 2), fx(0), fx(5)), dir, fx(4)), None);
        let none_steps = SdfRoad {
            field: &f,
            tolerance: r(1, 10_000),
            max_steps: 0,
        };
        assert_eq!(
            none_steps.probe(v(r(1, 2), fx(0), fx(5)), dir, fx(10)),
            None
        );
    }

    #[test]
    fn sdf_non_finite_distance_is_no_hit() {
        let f = ClosureSdf::new(|_, _, _| f32::NAN, |_, _, _| (0.0, 1.0, 0.0));
        let road = SdfRoad {
            field: &f,
            tolerance: r(1, 10_000),
            max_steps: 16,
        };
        assert_eq!(road.probe(v(fx(0), fx(4), fx(0)), down(), fx(10)), None);
    }

    // ---------- Weather / grip ----------

    fn assert_scaled(g: AnisotropicFriction, base: AnisotropicFriction, k: Fix128) {
        assert_eq!(g.longitudinal_static, base.longitudinal_static * k);
        assert_eq!(g.longitudinal_kinetic, base.longitudinal_kinetic * k);
        assert_eq!(g.transverse_static, base.transverse_static * k);
        assert_eq!(g.transverse_kinetic, base.transverse_kinetic * k);
        assert_eq!(g.slip_threshold_m_s, base.slip_threshold_m_s);
    }

    #[test]
    fn dry_asphalt_values() {
        let c = RoadCondition::dry_asphalt();
        assert_eq!(c.material, AnisotropicFriction::tyre_asphalt());
        assert_eq!(c.weather, Weather::Dry);
        assert_eq!(c.rolling_resistance, r(12, 1000));
    }

    #[test]
    fn weather_table_ratios_are_exact() {
        // table (documented on weather_factor): dry 1, wet 7/10, snow 6/25, ice 3/25
        let base = RoadCondition::dry_asphalt();
        let speed = fx(10);
        let p = fx(220);
        let cases = [
            (Weather::Dry, Fix128::ONE),
            (
                Weather::Wet {
                    water_depth_mm: fx(1),
                },
                r(7, 10),
            ),
            (Weather::Snow, r(6, 25)),
            (Weather::Ice, r(3, 25)),
        ];
        for (w, k) in cases {
            let c = RoadCondition { weather: w, ..base };
            assert_eq!(c.weather_factor(), k, "{w:?}");
            assert_scaled(c.grip(speed, p), base.material, k);
        }
    }

    #[test]
    fn horne_onset_speed_matches_f64() {
        for p in [100i64, 180, 220, 250, 900] {
            let got = hydroplaning_onset_speed(fx(p));
            let want = 6.35 * (p as f64).sqrt() / 3.6;
            assert!(close(got, want, 1e-12), "p={p}: {got} vs {want}");
        }
        // 0 and negative pressure: onset speed 0
        assert_eq!(hydroplaning_onset_speed(Fix128::ZERO), Fix128::ZERO);
        assert_eq!(hydroplaning_onset_speed(fx(-50)), Fix128::ZERO);
    }

    #[test]
    fn wet_grip_below_onset_is_speed_independent() {
        let p = fx(220);
        let vp = hydroplaning_onset_speed(p);
        let c = RoadCondition {
            weather: Weather::Wet {
                water_depth_mm: fx(5),
            },
            ..RoadCondition::dry_asphalt()
        };
        let g0 = c.grip(Fix128::ZERO, p);
        assert_eq!(c.grip(fx(5), p), g0);
        assert_eq!(c.grip(vp * r(9, 10), p), g0);
        assert_eq!(c.grip(vp, p), g0);
        assert_scaled(g0, c.material, r(7, 10));
    }

    #[test]
    fn hydroplaning_above_onset_follows_model() {
        // model: factor (V_p / V)², floored at 1/10, deep water only (≥ 2.5 mm)
        let p = fx(220);
        let vp = hydroplaning_onset_speed(p);
        let deep = RoadCondition {
            weather: Weather::Wet {
                water_depth_mm: r(5, 2),
            },
            ..RoadCondition::dry_asphalt()
        };
        let g = deep.grip(vp * fx(2), p);
        let want = 1.1 * 0.7 * 0.25;
        assert!(
            close(g.longitudinal_static, want, 1e-12),
            "{}",
            g.longitudinal_static
        );
        let g_far = deep.grip(vp * fx(10), p);
        assert!(
            close(g_far.longitudinal_static, 1.1 * 0.7 * 0.1, 1e-12),
            "{}",
            g_far.longitudinal_static
        );
        // negative speed is the same as its magnitude
        assert_eq!(deep.grip(-(vp * fx(2)), p), g);
        // shallow film (2.4 mm) → no hydroplaning loss
        let shallow = RoadCondition {
            weather: Weather::Wet {
                water_depth_mm: r(24, 10),
            },
            ..deep
        };
        assert_scaled(shallow.grip(vp * fx(2), p), shallow.material, r(7, 10));
        // negative depth → treated as no standing water
        let neg = RoadCondition {
            weather: Weather::Wet {
                water_depth_mm: fx(-3),
            },
            ..deep
        };
        assert_scaled(neg.grip(vp * fx(2), p), neg.material, r(7, 10));
        // pressure 0: onset 0, any motion is above onset → floor 1/10; at rest no loss
        assert!(close(
            deep.grip(fx(1), Fix128::ZERO).longitudinal_static,
            1.1 * 0.7 * 0.1,
            1e-12
        ));
        assert_scaled(
            deep.grip(Fix128::ZERO, Fix128::ZERO),
            deep.material,
            r(7, 10),
        );
    }
}
