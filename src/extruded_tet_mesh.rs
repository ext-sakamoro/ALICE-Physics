//! Tetrahedral mesh of a rounded extruded outline: the solid
//! `{ length(max(w, 0)) + min(max(w.x, w.y), 0) ≤ skin }` with
//! `w = (d(x, y) + skin, |z| - (h - skin))`, `d` the signed distance of a
//! polygonal outline (holes allowed). It is the outline eroded by `skin`,
//! extruded to half-depth `h - skin`, and rounded by `skin`: flat top and bottom
//! equal to the eroded outline, side walls on the outline itself (its convex
//! corners stay sharp, as the distance to the edges says), and a quarter-round
//! of radius `skin` between them.
//!
//! # Construction
//!
//! 1. Erode the outline by `skin` ([`crate::polygon_offset`]); every vertex of
//!    the eroded outline knows the direction the solid's slices grow from it:
//!    the edge normal, the miter at a convex corner, the direction to the
//!    centre on a reflex arc.
//! 2. Split the eroded outline's edges to at most [`RoundedSlab::boundary_step`]
//!    and triangulate it ([`crate::polygon_cdt`]).
//! 3. Core: every triangle times every layer of `z` levels is a prism.
//! 4. Rim: a fixed triangulated cross-section in `(r, z)` (a strip `r ∈ [0,
//!    skin]` along the side wall, a fan for each quarter-round) is placed at
//!    every boundary vertex of the triangulation, along its growth direction,
//!    and swept from vertex to vertex: prisms again.
//! 5. Every prism becomes three tetrahedra by the smallest-index diagonal rule
//!    ([`crate::sdf_fem_mesh`]), so prisms that share a face split it the
//!    same way and the mesh is conforming. A prism that collapses (on a reflex
//!    arc all the stations meet at the arc's centre at `r = skin`) loses its
//!    degenerate tetrahedra.
//!
//! # How close to the solid
//!
//! Flat faces and side walls are exact up to the triangulation's grid. The
//! eroded outline is chords where it has arcs (reflex corners, sagitta
//! `skin (1 - cos 22.5°)`), and vertices within half that sagitta of the chord
//! of their neighbours are dropped; the two add, so a boundary vertex can be up
//! to 1.5 sagitta off the solid, inside or outside (0.31 mm and 1.4 mm at a
//! 12 mm skin). The round is inscribed by two chords per quarter.
//!
//! **Bends tighter than the skin.** Where the outline turns convexly with a
//! radius under the skin (the bowl of an `a`, the arm of an `r`), the erosion
//! drops edges and the two neighbours meet at a corner. The slices of the solid
//! between that eroded corner and the wall bring the bend back gradually; the
//! rim here sweeps the eroded corner's miter instead, an approximation held to
//! the same 1.5-sagitta bound on the vertices (measured on the logo letters, one
//! test per letter). A corner whose miter is longer than [`MAX_MITER`] skins
//! is refused ([`ExtrudeError::SharpCorner`]).

use std::collections::BTreeMap;

use crate::polygon_cdt::{triangulate, CdtConfig, CdtError};
use crate::polygon_offset::{erode, OffsetConfig, OffsetError, OffsetNormal};
use crate::sdf_fem_mesh::{prism_to_tets, push_tet, SdfTetMesh};

/// Chords per quarter-round, and the angular resolution of the arcs at reflex
/// corners (45° each).
///
/// Measured on an L at skin 12 mm (worst dihedral angle): 2 chords 14.64°,
/// 3 chords 5.76°, 4 chords 2.21°. Around a reflex corner the round's slices
/// shrink to the corner's vertical edge, so rim vertices near that edge come
/// closer together the finer the arcs are; past two chords the elements there
/// fall under the dihedral floor. Two chords leave a sagitta of
/// `skin (1 - cos 22.5°)`, 7.6 % of the skin.
pub const ROUND_SEGMENTS: usize = 2;

/// Longest miter, as a multiple of the skin, that a convex corner of the eroded
/// outline may have: `1 / sin(15°)`, a corner of 30°. The sharpest logo corner
/// is 53.1° (2.24).
pub const MAX_MITER: f64 = 3.863_703_305_156_273;

/// Most layers between the two rounds.
pub const MAX_LAYERS: f64 = 4096.0;

/// Most points on the densified eroded outline.
pub const MAX_BOUNDARY_POINTS: f64 = 200_000.0;

/// Parameters of [`mesh_rounded_slab`] (input units, e.g. metres).
#[derive(Debug, Clone, Copy)]
pub struct RoundedSlab {
    /// Radius of the round, and depth of the erosion.
    pub skin: f64,
    /// Half the thickness of the slab.
    pub half_depth: f64,
    /// Longest boundary edge of the eroded outline.
    pub boundary_step: f64,
    /// Longest edge inside the outline.
    pub max_edge: f64,
    /// Target thickness of a layer between the two rounds.
    pub layer: f64,
    /// Grid of the triangulation.
    pub grid: f64,
}

/// Why a mesh was refused.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExtrudeError {
    /// A parameter is out of range.
    BadConfig,
    /// The erosion failed.
    Offset(OffsetError),
    /// The triangulation failed.
    Cdt(CdtError),
    /// A boundary vertex of the triangulation could not be traced back to the
    /// eroded outline.
    LostBoundary,
    /// A convex corner of the eroded outline sharper than [`MAX_MITER`] allows
    /// (a spike, or a bend of radius under the skin that the erosion closed).
    SharpCorner {
        /// The corner of the eroded outline.
        at: [f64; 2],
    },
    /// The settings ask for more than [`MAX_LAYERS`] layers or
    /// [`MAX_BOUNDARY_POINTS`] boundary points.
    TooLarge,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len(a: [f64; 2]) -> f64 {
    crate::det_math::sqrt64(dot(a, a))
}

/// A densified segment of the eroded outline: its ends, the centre of the
/// reflex arc it is a chord of (if it is one), and its outward normal.
type SegGrowth = ([f64; 2], [f64; 2], Option<[f64; 2]>, [f64; 2]);

/// How a boundary point's slice grows outward with `r ∈ [0, skin]`.
#[derive(Clone, Copy)]
enum Growth {
    /// `p + r g` (edge normal, or the miter at a convex corner).
    Along([f64; 2]),
    /// On a reflex arc: `p + r u`, `u` toward `centre`; at `r = skin` the point
    /// is `centre` itself.
    Arc { u: [f64; 2], centre: [f64; 2] },
}

/// Mesh the rounded slab of `contours`.
///
/// # Errors
///
/// See [`ExtrudeError`].
pub fn mesh_rounded_slab(
    contours: &[Vec<[f64; 2]>],
    cfg: &RoundedSlab,
) -> Result<SdfTetMesh, ExtrudeError> {
    let s = cfg.skin;
    let h = cfg.half_depth;
    let finite = [s, h, cfg.boundary_step, cfg.max_edge, cfg.layer, cfg.grid]
        .iter()
        .all(|x| x.is_finite());
    if !(finite
        && s > 0.0
        && h > s
        && cfg.boundary_step > 0.0
        && cfg.max_edge > 0.0
        && cfg.layer > 0.0
        && cfg.grid > 0.0)
    {
        return Err(ExtrudeError::BadConfig);
    }
    let k = ROUND_SEGMENTS;
    // the arcs at reflex corners share the round's angular resolution
    let alpha = std::f64::consts::FRAC_PI_2 / k as f64;
    let sagitta = s * (1.0 - cos_small(alpha / 2.0));
    let eroded = erode(contours, s, &OffsetConfig { sagitta }).map_err(ExtrudeError::Offset)?;

    // 2. densify the eroded outline; remember each point's growth
    let mut budget = 0.0f64;
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut growth_at: Vec<([f64; 2], Growth)> = Vec::new();
    // and, per densified segment, the growth of a point inserted on it
    let mut seg_growth: Vec<SegGrowth> = Vec::new();
    for c in &eroded.contours {
        let c = &simplify(c, 0.5 * sagitta);
        let mut ring = Vec::new();
        let n = c.len();
        for i in 0..n {
            let (v, w) = (c[i], c[(i + 1) % n]);
            let g = match v.normal {
                OffsetNormal::Edge(nrm) => Growth::Along(nrm),
                OffsetNormal::Corner(a, b) => {
                    let d = 1.0 + dot(a, b);
                    let m = [(a[0] + b[0]) / d, (a[1] + b[1]) / d];
                    // the miter reaches `skin / sin(θ/2)`; past MAX_MITER the
                    // corner is a spike, or the two lines met only because the
                    // erosion dropped a bend of radius under the skin between
                    // them, and the slices there are not a miter: refuse
                    if len(m).partial_cmp(&MAX_MITER) != Some(std::cmp::Ordering::Less) {
                        return Err(ExtrudeError::SharpCorner { at: v.p });
                    }
                    Growth::Along(m)
                }
                OffsetNormal::Arc { centre, normal } => Growth::Arc { u: normal, centre },
            };
            growth_at.push((v.p, g));
            ring.push(v.p);
            let e = sub(w.p, v.p);
            let l = len(e);
            // outward normal of the segment: the inside is on the left
            let out = [e[1] / l, -e[0] / l];
            let on_arc = matches!((v.normal, w.normal), (OffsetNormal::Arc { centre: a, .. }, OffsetNormal::Arc { centre: b, .. }) if a == b);
            // next to a sharp convex corner the rim reaches out along the miter,
            // `skin / sin(θ/2)`; a boundary point closer to the corner than that
            // makes the swept elements there skewed, so the step along such an
            // edge is at least the reach
            let reach = |n: OffsetNormal| match n {
                OffsetNormal::Corner(a, b) => {
                    s * len([a[0] + b[0], a[1] + b[1]]) / (1.0 + dot(a, b))
                }
                _ => 0.0,
            };
            let step = crate::det_math::max64(
                crate::det_math::max64(cfg.boundary_step, reach(v.normal)),
                reach(w.normal),
            );
            let pieces = if on_arc {
                1
            } else {
                let p = crate::det_math::max64((l / step).ceil(), 1.0);
                budget += p;
                if budget > MAX_BOUNDARY_POINTS {
                    return Err(ExtrudeError::TooLarge);
                }
                p as usize
            };
            for j in 1..pieces {
                let t = j as f64 / pieces as f64;
                let p = [v.p[0] + t * e[0], v.p[1] + t * e[1]];
                growth_at.push((p, Growth::Along(out)));
                ring.push(p);
            }
            // a chord of a reflex arc keeps the arc's centre
            let arc = match (v.normal, w.normal) {
                (OffsetNormal::Arc { centre: a, .. }, OffsetNormal::Arc { centre: b, .. })
                    if a == b =>
                {
                    Some(a)
                }
                _ => None,
            };
            seg_growth.push((v.p, w.p, arc, out));
        }
        rings.push(ring);
    }
    let cdt = triangulate(
        &rings,
        &CdtConfig {
            grid: cfg.grid,
            min_angle_deg: 20.0,
            max_edge: cfg.max_edge,
            max_points: 200_000,
        },
    )
    .map_err(ExtrudeError::Cdt)?;

    // growth of every boundary vertex of the triangulation
    let quant = |p: [f64; 2]| -> [i64; 2] {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "inside the triangulation's checked grid range"
        )]
        let q = [
            crate::det_math::round64(p[0] / cfg.grid) as i64,
            crate::det_math::round64(p[1] / cfg.grid) as i64,
        ];
        q
    };
    let mut by_grid: BTreeMap<[i64; 2], Growth> = BTreeMap::new();
    for (p, g) in &growth_at {
        by_grid.entry(quant(*p)).or_insert(*g);
    }
    let mut growth: BTreeMap<u32, Growth> = BTreeMap::new();
    for seg in &cdt.segments {
        for &v in seg {
            if growth.contains_key(&v) {
                continue;
            }
            let g = if let Some(g) = by_grid.get(&cdt.points[v as usize]) {
                *g
            } else {
                // a split point of the triangulation: on one densified segment
                let p = cdt.point(v);
                let best = seg_growth
                    .iter()
                    .map(|(a, b, arc, out)| (seg_dist(p, *a, *b), *arc, *out))
                    .min_by(|x, y| x.0.total_cmp(&y.0))
                    .ok_or(ExtrudeError::LostBoundary)?;
                // a split point is within half a grid diagonal of its input
                // segment, whose ends were rounded by as much: √2 grid steps
                if best.0 > std::f64::consts::SQRT_2 * cfg.grid * (1.0 + 1.0e-9) {
                    return Err(ExtrudeError::LostBoundary);
                }
                match best.1 {
                    Some(centre) => {
                        let d = sub(centre, p);
                        let l = len(d);
                        Growth::Arc {
                            u: [d[0] / l, d[1] / l],
                            centre,
                        }
                    }
                    None => Growth::Along(best.2),
                }
            };
            growth.insert(v, g);
        }
    }

    // 3. z levels: -h, -(h - s), middle layers, h - s, h
    let hp = h - s;
    let mid_f = crate::det_math::max64((2.0 * hp / cfg.layer).ceil(), 1.0);
    if mid_f > MAX_LAYERS {
        return Err(ExtrudeError::TooLarge);
    }
    let mid = mid_f as usize;
    let mut zs = vec![-h];
    for j in 0..=mid {
        zs.push(-hp + 2.0 * hp * j as f64 / mid as f64);
    }
    zs.push(h);
    let nz = zs.len();

    let mut mesh = SdfTetMesh::default();
    let np = cdt.points.len();
    // core vertex (point i, level j)
    for &z in &zs {
        for i in 0..np {
            let p = cdt.point(i as u32);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "vertices are stored in f32"
            )]
            mesh.vertices.push([p[0] as f32, p[1] as f32, z as f32]);
        }
    }
    let core = |i: u32, j: usize| -> u32 { (j * np) as u32 + i };
    for t in &cdt.triangles {
        for j in 0..nz - 1 {
            for tet in prism_to_tets(t.map(|i| core(i, j)), t.map(|i| core(i, j + 1))) {
                push_tet(&mut mesh, tet);
            }
        }
    }

    // 4. rim: cross-section nodes (r, z, core level if r = 0)
    #[derive(Clone, Copy)]
    struct Node {
        r: f64,
        z: f64,
        level: Option<usize>,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut node_at_level = vec![usize::MAX; nz];
    for (j, &z) in zs.iter().enumerate() {
        node_at_level[j] = nodes.len();
        nodes.push(Node {
            r: 0.0,
            z,
            level: Some(j),
        });
    }
    // wall nodes r = s at the middle levels 1 ..= nz-2
    let mut wall = vec![usize::MAX; nz];
    for j in 1..nz - 1 {
        wall[j] = nodes.len();
        nodes.push(Node {
            r: s,
            z: zs[j],
            level: None,
        });
    }
    // quarter-round fans, top (sign +1, centre level nz-2) and bottom (sign -1, centre level 1)
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for j in 1..nz - 2 {
        let (a, b, c, d) = (node_at_level[j], wall[j], wall[j + 1], node_at_level[j + 1]);
        tris.push([a, b, c]);
        tris.push([a, c, d]);
    }
    for (sign, centre_level, edge_level) in [(1.0f64, nz - 2, nz - 1), (-1.0, 1usize, 0usize)] {
        let centre = node_at_level[centre_level];
        let mut prev = wall[centre_level];
        for q in 1..=k {
            let phi = std::f64::consts::FRAC_PI_2 * q as f64 / k as f64;
            let cur = if q == k {
                node_at_level[edge_level]
            } else {
                let (c, sn) = cos_sin(phi);
                nodes.push(Node {
                    r: s * c,
                    z: zs[centre_level] + sign * s * sn,
                    level: None,
                });
                nodes.len() - 1
            };
            tris.push([centre, prev, cur]);
            prev = cur;
        }
    }

    // rim vertices per boundary vertex; vertices at r = s on a reflex arc are
    // the arc's centre, shared by all its stations
    let mut rim: BTreeMap<(u32, usize), u32> = BTreeMap::new();
    let mut centres: BTreeMap<([u64; 2], usize), u32> = BTreeMap::new();
    let mut vertex = |mesh: &mut SdfTetMesh, b: u32, node: usize| -> u32 {
        let nd = nodes[node];
        if let Some(j) = nd.level {
            return core(b, j);
        }
        if let Some(&v) = rim.get(&(b, node)) {
            return v;
        }
        let p = cdt.point(b);
        let (q, shared) = match growth[&b] {
            Growth::Along(g) => ([p[0] + nd.r * g[0], p[1] + nd.r * g[1]], None),
            Growth::Arc { u, centre } => {
                if nd.r == s {
                    (centre, Some([centre[0].to_bits(), centre[1].to_bits()]))
                } else {
                    ([p[0] + nd.r * u[0], p[1] + nd.r * u[1]], None)
                }
            }
        };
        if let Some(key) = shared {
            if let Some(&v) = centres.get(&(key, node)) {
                rim.insert((b, node), v);
                return v;
            }
        }
        let v = mesh.vertices.len() as u32;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "vertices are stored in f32"
        )]
        mesh.vertices.push([q[0] as f32, q[1] as f32, nd.z as f32]);
        rim.insert((b, node), v);
        if let Some(key) = shared {
            centres.insert((key, node), v);
        }
        v
    };
    // boundary edges of the triangulation, swept
    let mut count: BTreeMap<(u32, u32), (u32, u32, usize)> = BTreeMap::new();
    for t in &cdt.triangles {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let en = count.entry((a.min(b), a.max(b))).or_insert((a, b, 0));
            en.2 += 1;
        }
    }
    for (_, (a, b, n)) in count {
        if n != 1 {
            continue;
        }
        for tri in &tris {
            let bottom = tri.map(|nd| vertex(&mut mesh, a, nd));
            let top = tri.map(|nd| vertex(&mut mesh, b, nd));
            for tet in prism_to_tets(bottom, top) {
                push_tet(&mut mesh, tet);
            }
        }
    }
    Ok(mesh)
}

/// Drop vertices of an eroded contour that lie within `tol` of the chord
/// joining their neighbours, one at a time in index order, until none does.
///
/// A flattened curve erodes into edges far shorter than the skin (a 3 mm edge
/// on a convex bend comes out about 1 mm long), and a boundary point every
/// millimetre under layers of a centimetre makes slivers. Dropping a vertex
/// moves the boundary by at most `tol`, the same sagitta the arcs already
/// carry.
fn simplify(
    c: &[crate::polygon_offset::OffsetVertex],
    tol: f64,
) -> Vec<crate::polygon_offset::OffsetVertex> {
    // indices into `c` still kept; a drop is allowed only if every original
    // vertex between the new neighbours stays within `tol` of their chord, so
    // drops do not add up
    let n = c.len();
    let mut keep: Vec<usize> = (0..n).collect();
    loop {
        let m = keep.len();
        if m <= 3 {
            break;
        }
        let drop = (0..m).find(|&k| {
            let (ia, ib) = (keep[(k + m - 1) % m], keep[(k + 1) % m]);
            let (a, b) = (c[ia].p, c[ib].p);
            let mut j = (ia + 1) % n;
            while j != ib {
                if seg_dist(c[j].p, a, b) > tol {
                    return false;
                }
                j = (j + 1) % n;
            }
            true
        });
        match drop {
            Some(k) => {
                keep.remove(k);
            }
            None => break,
        }
    }
    keep.into_iter().map(|i| c[i]).collect()
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let e = sub(b, a);
    let t = crate::det_math::raw_clamp64(dot(sub(p, a), e) / dot(e, e), 0.0, 1.0);
    len(sub(p, [a[0] + t * e[0], a[1] + t * e[1]]))
}

/// `cos` of a small angle (at most a quarter turn) by its Taylor series.
fn cos_small(x: f64) -> f64 {
    cos_sin(x).0
}

/// `(cos x, sin x)` for `x` in `[0, π/2]` by Taylor series (no platform libm).
fn cos_sin(x: f64) -> (f64, f64) {
    let (mut c, mut s) = (1.0, x);
    let (mut tc, mut ts) = (1.0, x);
    for k in 1..14 {
        tc *= -x * x / f64::from((2 * k - 1) * (2 * k));
        ts *= -x * x / f64::from((2 * k) * (2 * k + 1));
        c += tc;
        s += ts;
    }
    (c, s)
}
