//! Boundary conformity for [`crate::sdf_fem_mesh::generate_marching_tets`]:
//! boundary vertices onto the surface, and vertices onto the surface's creases
//! and corners.
//!
//! # Why the lattice warp is not enough
//!
//! Marching tetrahedra place a boundary vertex where the field changes sign
//! along a lattice edge, by interpolating the field linearly. Two things are
//! then wrong at a sharp feature (the rim of a box, where the field is the
//! maximum of two planes):
//!
//! - **The vertex is off the surface.** Along an edge that straddles the rim the
//!   field is not linear, so its interpolated zero is not a zero of the field.
//! - **No vertex is on the rim.** Every vertex lies on one lattice edge, and a
//!   rim crosses lattice edges only where it happens to; between those points the
//!   boundary cuts across the rim as a chamfer. Measured on a `0.2 m` cube at
//!   `0.04 m` cells: half a cell off every rim, `0.64` of each face left and
//!   `8 %` of the volume missing.
//!
//! The lattice warp ([`crate::sdf_fem_mesh`], Labelle and Shewchuk's
//! *Isosurface Stuffing*) fixes neither: it is built for smooth surfaces and
//! rounds sharp ones by design. Moving each vertex onto the surface along the
//! gradient fixes the first and not the second, because it puts every vertex on
//! one face or the other and still none on the rim.
//!
//! # What this pass does
//!
//! 1. **Project** every boundary vertex onto the zero set by Newton steps on the
//!    field (`p ← p − d ∇d / |∇d|²`).
//! 2. **Split creases.** A boundary edge whose two ends lie on different faces
//!    of the surface (tangent planes more than [`FEATURE_ANGLE_COS`] apart, with
//!    no face in common) is split, and the new vertex is placed where the tangent
//!    planes of the two faces meet, nearest the edge's midpoint. This is the
//!    quadric-error placement of dual contouring (Ju, Losasso, Schaefer and
//!    Warren, *Dual Contouring of Hermite Data*, 2002) applied to one crease at a
//!    time, and the same move as Attene et al.'s *EdgeSharpener* (2003) on a
//!    tetrahedral rather than a triangle mesh.
//! 3. **Split corners.** A boundary triangle whose three vertices pairwise share
//!    a face but have no face in common touches three faces, and is split at the
//!    point where the three tangent planes meet.
//!
//! The tangent planes are not taken at the vertices themselves: a vertex near a
//! crease has a finite-difference gradient that mixes both faces. Each plane is
//! re-anchored at a point [`ANCHOR_CELLS`] away from the current estimate,
//! inside its own face, and the meeting point is solved again
//! ([`REFINE_ITERS`] times). On flat faces that changes nothing; on a curved
//! face it moves the crease vertex onto the curved crease, and on a smooth
//! surface (no crease at all) it makes the re-anchored planes nearly parallel,
//! which is what rejects a false crease that a coarse lattice suggested.
//!
//! # Element quality and validity
//!
//! A split is made at the midpoint (always valid), and the new vertex is then
//! moved to its target with the same guard as a projection: no incident element
//! may invert, and none may end below a minimum dihedral angle of
//! [`MIN_SIN_DIHEDRAL`] (or below what it had, if it had less). A move that fails
//! is retried at half the distance, down to an eighth, and otherwise not made.
//!
//! # Determinism
//!
//! Vertices, edges and faces are visited in index order, maps are ordered, and
//! the arithmetic is IEEE add, multiply, divide and square root (no fused
//! multiply-add, no transcendental), so the output is the same on every target.

use std::collections::{BTreeMap, BTreeSet};

use crate::sdf_collider::SdfField;
use crate::sdf_fem_mesh::{split_children, SdfTetMesh, Tetrahedron};

/// Two tangent planes belong to the same face when their normals are within
/// 20° (`cos 20°`). Creases shallower than this are left to the lattice.
const FEATURE_ANGLE_COS: f64 = 0.939_692_620_785_908_4;

/// The Gram determinant a set of crease planes must keep after re-anchoring:
/// `sin² 20°` for two planes, the same floor on `det(N)²` for three.
const MIN_GRAM_DET: f64 = 0.116_977_778_440_511;

/// `sin 8.5°`: the dihedral floor a move may not take an element under. The
/// quality gate in `tests/mesh_quality.rs` is 8°.
const MIN_SIN_DIHEDRAL: f64 = 0.147_809_411_129_436_3;
fn floor_sin() -> f64 { std::env::var("SFB_FLOOR").ok().and_then(|x| x.parse().ok()).unwrap_or(MIN_SIN_DIHEDRAL) }

/// `min sin(dihedral)` under which a tetrahedron on one face counts as
/// collapsed. Coplanar points rounded to f32 leave `~1e-7`; a real element
/// with a dihedral angle under `1e-4` rad does not survive the quality gate.
const COLLAPSED_SIN: f64 = 1.0e-4;

/// Finite-difference step for Newton projection, as a fraction of `cell`.
const NEWTON_STEP_CELLS: f32 = 1.0e-3;

/// Finite-difference step for a tangent plane at an anchor, as a fraction of
/// `cell`. Wider than the Newton step for accuracy; the anchor is
/// [`ANCHOR_CELLS`] from the crease so the stencil stays on its face.
const PLANE_STEP_CELLS: f32 = 0.1;

/// Distance from the crease estimate to a plane's anchor, in cells.
const ANCHOR_CELLS: f64 = 0.25;

/// Re-anchoring rounds per crease or corner.
const REFINE_ITERS: usize = 3;

/// Newton steps per projection.
const NEWTON_ITERS: usize = 8;

/// A crease or corner point is accepted only on the surface to within this
/// fraction of a cell.
const ON_SURFACE_CELLS: f64 = 1.0e-3;

/// Rounds of smoothing around moves blocked by element shape.
const SMOOTH_ROUNDS: usize = 4;

/// Passes of crease and corner splitting.
const FEATURE_PASSES: usize = 3;

#[derive(Clone, Copy, Debug)]
struct Plane {
    n: [f64; 3],
    p: [f64; 3],
    /// Re-anchored inside its face by [`feature_point`]. A plane that is not
    /// is the raw gradient at the vertex, which mixes the faces of a crease the
    /// vertex sits on; it is dropped once the crease's own planes are known.
    refined: bool,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add_scaled(a: [f64; 3], s: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn to64(p: [f32; 3]) -> [f64; 3] {
    p.map(f64::from)
}
#[allow(
    clippy::cast_possible_truncation,
    reason = "vertices are stored in f32; rounding the solved f64 point to f32 is the intended last step"
)]
fn to32(p: [f64; 3]) -> [f32; 3] {
    p.map(|x| x as f32)
}

fn field<F: SdfField + ?Sized>(sdf: &F, p: [f64; 3]) -> f64 {
    let q = to32(p);
    f64::from(sdf.distance(q[0], q[1], q[2]))
}

/// Central-difference gradient of the field, in f64 from f32 samples.
fn gradient<F: SdfField + ?Sized>(sdf: &F, p: [f64; 3], h: f32) -> [f64; 3] {
    let q = to32(p);
    let mut g = [0.0; 3];
    for k in 0..3 {
        let mut a = q;
        let mut b = q;
        a[k] += h;
        b[k] -= h;
        let span = f64::from(a[k]) - f64::from(b[k]);
        if span > 0.0 {
            g[k] = (f64::from(sdf.distance(a[0], a[1], a[2]))
                - f64::from(sdf.distance(b[0], b[1], b[2])))
                / span;
        }
    }
    g
}

fn unit(g: [f64; 3]) -> Option<[f64; 3]> {
    let l = norm(g);
    (l > 0.0 && l.is_finite()).then(|| g.map(|x| x / l))
}

/// Newton projection onto the zero set; `None` when the gradient vanishes or
/// the result is not finite.
fn project<F: SdfField + ?Sized>(sdf: &F, p: [f64; 3], h: f32) -> Option<[f64; 3]> {
    let mut x = p;
    for _ in 0..NEWTON_ITERS {
        let d = field(sdf, x);
        if d == 0.0 {
            break;
        }
        let g = gradient(sdf, x, h);
        let gg = dot(g, g);
        if !(gg > 1.0e-12) {
            return None;
        }
        let next = to64(to32(add_scaled(x, -d / gg, g)));
        if next == x {
            break;
        }
        x = next;
    }
    x.iter().all(|c| c.is_finite()).then_some(x)
}

/// Solve the 2×2 or 3×3 symmetric system `g α = r` by Cramer's rule.
fn solve_small(g: &[[f64; 3]; 3], r: [f64; 3], k: usize) -> Option<([f64; 3], f64)> {
    match k {
        2 => {
            let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
            if !(det.abs() > 0.0) {
                return None;
            }
            Some((
                [
                    (r[0] * g[1][1] - g[0][1] * r[1]) / det,
                    (g[0][0] * r[1] - r[0] * g[1][0]) / det,
                    0.0,
                ],
                det,
            ))
        }
        3 => {
            let c0 = cross(g[1], g[2]);
            let det = dot(g[0], c0);
            if !(det.abs() > 0.0) {
                return None;
            }
            // Cramer: replace column j with r. g is symmetric, so rows = columns.
            let col = |j: usize| -> f64 {
                let mut m = *g;
                for (i, row) in m.iter_mut().enumerate() {
                    row[j] = r[i];
                }
                dot(m[0], cross(m[1], m[2]))
            };
            Some(([col(0) / det, col(1) / det, col(2) / det], det))
        }
        _ => None,
    }
}

/// The point on every plane nearest `m` (least-norm correction of `m`), with
/// the Gram determinant of the unit normals.
fn meet(planes: &[Plane], m: [f64; 3]) -> Option<([f64; 3], f64)> {
    let k = planes.len();
    if !(2..=3).contains(&k) {
        return None;
    }
    let mut g = [[0.0; 3]; 3];
    let mut r = [0.0; 3];
    for i in 0..k {
        for j in 0..k {
            g[i][j] = dot(planes[i].n, planes[j].n);
        }
        r[i] = dot(planes[i].n, sub(planes[i].p, m));
    }
    let (alpha, det) = solve_small(&g, r, k)?;
    let mut x = m;
    for i in 0..k {
        x = add_scaled(x, alpha[i], planes[i].n);
    }
    x.iter().all(|c| c.is_finite()).then_some((x, det))
}

/// Re-anchor each plane inside its own face near the current estimate and
/// solve again. `owners[i]` is a point known to lie on face `i`.
fn feature_point<F: SdfField + ?Sized>(
    sdf: &F,
    planes: &[Plane],
    owners: &[[f64; 3]],
    m: [f64; 3],
    cell: f32,
) -> Option<([f64; 3], Vec<Plane>)> {
    let c = f64::from(cell);
    let h_newton = NEWTON_STEP_CELLS * cell;
    let h_plane = PLANE_STEP_CELLS * cell;
    let mut planes = planes.to_vec();
    let (mut x, _) = meet(&planes, m)?;
    for _ in 0..REFINE_ITERS {
        let mut next = Vec::with_capacity(planes.len());
        for (i, (plane, owner)) in planes.iter().zip(owners).enumerate() {
            // Step into face i straight away from the other faces: minus the
            // other normals' components in this plane. That is the face's side
            // at a convex crease; at a concave one the face lies the other way,
            // which the vertex the plane came from tells.
            let mut away = [0.0; 3];
            for (j, other) in planes.iter().enumerate() {
                if j != i {
                    let in_plane = add_scaled(other.n, -dot(other.n, plane.n), plane.n);
                    away = sub(away, in_plane);
                }
            }
            let toward_owner = sub(*owner, x);
            let dir = if dot(away, toward_owner) < 0.0 {
                away.map(|a| -a)
            } else {
                away
            };
            let dir = unit(dir)?;
            let anchor = project(sdf, add_scaled(x, ANCHOR_CELLS * c, dir), h_newton)?;
            let n = unit(gradient(sdf, anchor, h_plane))?;
            next.push(Plane {
                n,
                p: anchor,
                refined: true,
            });
        }
        let (y, det) = meet(&next, x)?;
        if !(det >= MIN_GRAM_DET) {
            return None;
        }
        planes = next;
        x = y;
    }
    (field(sdf, x).abs() <= ON_SURFACE_CELLS * c).then_some((x, planes))
}

/// Where a lattice corner should warp to if it lies near a crease or a corner
/// of the surface: the meeting point of the tangent planes of the distinct
/// faces met by the corner's incident crossings (two faces: the crease point
/// nearest the corner; three: the corner point). `None` when the crossings
/// see one face only, or the meeting point is not on the surface.
pub(crate) fn crease_target<F: SdfField + ?Sized>(
    sdf: &F,
    corner: [f32; 3],
    crossings: &[[f32; 3]],
    cell: f32,
) -> Option<[f32; 3]> {
    let h = NEWTON_STEP_CELLS * cell;
    let mut planes: Vec<Plane> = Vec::new();
    for c in crossings {
        let Some(q) = project(sdf, to64(*c), h) else {
            continue;
        };
        let Some(n) = unit(gradient(sdf, q, h)) else {
            continue;
        };
        let pl = Plane {
            n,
            p: q,
            refined: false,
        };
        if !planes.iter().any(|x| dot(x.n, pl.n) >= FEATURE_ANGLE_COS) {
            planes.push(pl);
        }
    }
    if planes.len() < 2 {
        return None;
    }
    planes.truncate(3);
    let owners: Vec<[f64; 3]> = planes.iter().map(|p| p.p).collect();
    let (x, _) = feature_point(sdf, &planes, &owners, to64(corner), cell)?;
    Some(to32(x))
}

/// `min sin(dihedral)` over the six edges of a tetrahedron, and its signed
/// volume ×6. `sin θ_e = 3 V |e| / (2 A₁ A₂)`.
fn tet_quality(p: [[f64; 3]; 4]) -> (f64, f64) {
    let vol6 = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])));
    // face opposite vertex i, as twice its area
    let area2 = |i: usize| -> f64 {
        let o: Vec<usize> = (0..4).filter(|&k| k != i).collect();
        norm(cross(sub(p[o[1]], p[o[0]]), sub(p[o[2]], p[o[0]])))
    };
    let a = [area2(0), area2(1), area2(2), area2(3)];
    let mut worst = f64::INFINITY;
    for i in 0..4 {
        for j in (i + 1)..4 {
            // edge (i, j) is shared by the faces opposite the other two vertices
            let o: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
            let e = norm(sub(p[j], p[i]));
            let denom = a[o[0]] * a[o[1]];
            // V = vol6 / 6, A = area2 / 2: 3 V |e| / (2 A A) = vol6 |e| / (area2 area2)
            let s = if denom > 0.0 {
                vol6.abs() * e / denom
            } else {
                0.0
            };
            worst = worst.min(s);
        }
    }
    (worst, vol6)
}

struct Work<'a, F: SdfField + ?Sized> {
    mesh: &'a mut SdfTetMesh,
    sdf: &'a F,
    cell: f32,
    /// Tangent planes per vertex; empty for interior vertices.
    planes: Vec<Vec<Plane>>,
    /// Tetrahedra incident to each vertex.
    incident: Vec<Vec<u32>>,
    /// Vertices of proposals whose vertex could not reach its target; the
    /// elements around them are refined before the next pass.
    blocked: BTreeSet<u32>,
}

impl<F: SdfField + ?Sized> Work<'_, F> {
    fn global_worst(&self) -> f64 {
        (0..self.mesh.tets.len() as u32).map(|t| tet_quality(self.tet_points(t)).0).fold(f64::INFINITY, f64::min)
    }

    fn rebuild_incidence(&mut self) {
        self.incident = vec![Vec::new(); self.mesh.vertices.len()];
        for (t, tet) in self.mesh.tets.iter().enumerate() {
            for v in tet.vertices {
                self.incident[v as usize].push(t as u32);
            }
        }
    }

    fn tet_points(&self, t: u32) -> [[f64; 3]; 4] {
        self.mesh.tets[t as usize]
            .vertices
            .map(|v| to64(self.mesh.vertices[v as usize]))
    }

    /// Move every vertex to its target. A move can be blocked only by a
    /// neighbour that has not moved yet (two new vertices on one crease, each
    /// flattening the other's elements while it waits), so whole moves are
    /// swept until a sweep makes no progress, and only then is each remaining
    /// vertex moved as far as it can go. Returns how many reached their target.
    fn move_all(&mut self, moves: &[(u32, [f64; 3])]) -> usize {
        let mut pending: Vec<(u32, [f64; 3])> = moves.to_vec();
        let mut done = 0;
        let rounds = std::env::var("SFB_ROUNDS").ok().and_then(|x| x.parse().ok()).unwrap_or(SMOOTH_ROUNDS);
        for round in 0..=rounds {
            loop {
                let before = pending.len();
                pending.retain(|&(v, x)| !self.guarded_move(v, x, true));
                done += before - pending.len();
                if pending.len() == before || pending.is_empty() {
                    break;
                }
            }
            if pending.is_empty() {
                break;
            }
            // two new vertices on one crease can each block the other: move
            // what is left together
            let moved = self.batch_move(&pending);
            if moved > 0 {
                done += moved;
                pending.retain(|&(v, x)| to64(self.mesh.vertices[v as usize]) != to64(to32(x)));
                continue;
            }
            if round == rounds {
                break;
            }
            // what is still blocked is blocked by element shape: give the
            // neighbourhood room and try again
            let around: Vec<u32> = pending.iter().map(|&(v, _)| v).collect();
            if self.smooth_around(&around) == 0 {
                break;
            }
        }
        for &(v, x) in &pending {
            self.guarded_move(v, x, false);
        }
        done
    }

    /// Move every vertex of `moves` at once, then put back the ones whose
    /// elements end inverted or under the floor, until what remains is valid.
    /// Returns the number left moved.
    fn batch_move(&mut self, moves: &[(u32, [f64; 3])]) -> usize {
        let mut tets: BTreeSet<u32> = BTreeSet::new();
        for &(v, _) in moves {
            tets.extend(self.incident[v as usize].iter().copied());
        }
        let q0: BTreeMap<u32, f64> = tets
            .iter()
            .map(|&t| (t, tet_quality(self.tet_points(t)).0))
            .collect();
        let from: Vec<[f32; 3]> = moves.iter().map(|&(v, _)| self.mesh.vertices[v as usize]).collect();
        let mut moved: Vec<bool> = vec![true; moves.len()];
        for &(v, x) in moves {
            self.mesh.vertices[v as usize] = to32(x);
        }
        loop {
            let mut bad: BTreeSet<u32> = BTreeSet::new();
            for (&t, &q_before) in &q0 {
                let (q, vol6) = tet_quality(self.tet_points(t));
                if self.collapses_onto_face(t, q) {
                    continue;
                }
                if !(vol6 > 0.0) || q < q_before.min(floor_sin()) {
                    bad.extend(self.mesh.tets[t as usize].vertices);
                }
            }
            let mut reverted = false;
            for (i, &(v, _)) in moves.iter().enumerate() {
                if moved[i] && bad.contains(&v) {
                    self.mesh.vertices[v as usize] = from[i];
                    moved[i] = false;
                    reverted = true;
                }
            }
            if !reverted {
                break;
            }
        }
        moved.iter().filter(|&&m| m).count()
    }

    /// Smooth the two rings of vertices around `seeds`, so that a blocked move
    /// has room. An interior vertex goes toward the centroid of its neighbours;
    /// a vertex on exactly one face goes toward the centroid of its neighbours on
    /// that face and back onto the surface. Vertices on a crease or a corner,
    /// and vertices of a face the bounding box cut, do not move. A move is kept
    /// only if it raises the worst element around the vertex. Returns the
    /// number of moves kept.
    fn smooth_around(&mut self, seeds: &[u32]) -> usize {
        let mut boundary = vec![false; self.mesh.vertices.len()];
        for f in boundary_faces(self.mesh) {
            for v in f {
                boundary[v as usize] = true;
            }
        }
        let ring = |w: &Self, vs: &BTreeSet<u32>| -> BTreeSet<u32> {
            let mut out = vs.clone();
            for &v in vs {
                for &t in &w.incident[v as usize] {
                    out.extend(w.mesh.tets[t as usize].vertices);
                }
            }
            out
        };
        let seeds: BTreeSet<u32> = seeds.iter().copied().collect();
        let region = ring(self, &ring(self, &seeds));
        let h_newton = NEWTON_STEP_CELLS * self.cell;
        let mut kept = 0;
        for &u in &region {
            if seeds.contains(&u) {
                continue;
            }
            let planes = &self.planes[u as usize];
            let surface = match planes.len() {
                0 if !boundary[u as usize] => false,
                1 => true,
                _ => continue,
            };
            let mut nbrs: BTreeSet<u32> = BTreeSet::new();
            for &t in &self.incident[u as usize] {
                nbrs.extend(self.mesh.tets[t as usize].vertices);
            }
            nbrs.remove(&u);
            let face = surface.then(|| self.planes[u as usize][0]);
            let pts: Vec<[f64; 3]> = nbrs
                .iter()
                .filter(|&&n| match face {
                    None => true,
                    Some(f) => self.planes[n as usize].iter().any(|q| Self::same_face(&f, q)),
                })
                .map(|&n| to64(self.mesh.vertices[n as usize]))
                .collect();
            if pts.is_empty() {
                continue;
            }
            let k = pts.len() as f64;
            let mut c = [0.0; 3];
            for p in &pts {
                for i in 0..3 {
                    c[i] += p[i] / k;
                }
            }
            let target = if surface {
                match project(self.sdf, c, h_newton) {
                    Some(q) => q,
                    None => continue,
                }
            } else {
                c
            };
            let from = to64(self.mesh.vertices[u as usize]);
            let worst = |w: &Self| -> (f64, bool) {
                let mut q = f64::INFINITY;
                let mut valid = true;
                for &t in &w.incident[u as usize] {
                    let (qt, vol6) = tet_quality(w.tet_points(t));
                    q = q.min(qt);
                    valid &= vol6 > 0.0;
                }
                (q, valid)
            };
            let (q0, _) = worst(self);
            let mut moved = false;
            for frac in [1.0f64, 0.5, 0.25] {
                self.mesh.vertices[u as usize] = to32(add_scaled(from, frac, sub(target, from)));
                let (q, valid) = worst(self);
                if valid && q > q0 {
                    moved = true;
                    break;
                }
            }
            if moved {
                kept += 1;
                if surface {
                    let at = to64(self.mesh.vertices[u as usize]);
                    self.planes[u as usize][0].p = at;
                }
            } else {
                self.mesh.vertices[u as usize] = to32(from);
            }
        }
        kept
    }

    /// Move vertex `v` toward `target` as far as the guard allows (the whole
    /// way, then a half, a quarter, an eighth). Returns whether it reached the
    /// target.
    fn guarded_move(&mut self, v: u32, target: [f64; 3], whole_only: bool) -> bool {
        let from = to64(self.mesh.vertices[v as usize]);
        let before: Vec<(u32, f64)> = self.incident[v as usize]
            .iter()
            .map(|&t| (t, tet_quality(self.tet_points(t)).0))
            .collect();
        let fracs: &[f64] = if whole_only {
            &[1.0]
        } else {
            &[1.0, 0.5, 0.25, 0.125]
        };
        for (step, &frac) in fracs.iter().enumerate() {
            let to = to32(add_scaled(from, frac, sub(target, from)));
            self.mesh.vertices[v as usize] = to;
            let mut inv = 0;
            let mut low = 0;
            let mut qmin = f64::INFINITY;
            for &(t, q0) in &before {
                let (q, vol6) = tet_quality(self.tet_points(t));
                if self.collapses_onto_face(t, q) {
                    continue;
                }
                if !(vol6 > 0.0) {
                    inv += 1;
                    if step == 0 && std::env::var_os("SFB_DEBUG").is_some() {
                        let vs = self.mesh.tets[t as usize].vertices;
                        let np: Vec<usize> = vs.iter().map(|&u| self.planes[u as usize].len()).collect();
                        let ps: Vec<[f32;3]> = vs.iter().map(|&u| self.mesh.vertices[u as usize]).collect();
                        eprintln!("    inv t{t} vol6 {vol6:.2e} q {q:.2e} planes {np:?} v{v} from {from:?} {vs:?} {ps:?}");
                    }
                } else if q < q0.min(floor_sin()) {
                    low += 1;
                    qmin = qmin.min(q);
                    if step == 0 && std::env::var_os("SFB_DEBUG").is_some() {
                        let vs = self.mesh.tets[t as usize].vertices;
                        let np: Vec<usize> = vs.iter().map(|&u| self.planes[u as usize].len()).collect();
                        let ps: Vec<[f32;3]> = vs.iter().map(|&u| self.mesh.vertices[u as usize]).collect();
                        eprintln!("    low t{t} q {q:.2e} q0 {q0:.3} planes {np:?} share {} {ps:?}", self.shares_face(&vs));
                    }
                }
            }
            let ok = inv == 0 && low == 0;
            if step == 0 && !ok && std::env::var_os("SFB_DEBUG").is_some() {
                eprintln!("  move v{v} full: {inv} inverted, {low} low (min sin {qmin:.3})");
            }
            if ok {
                return step == 0;
            }
        }
        self.mesh.vertices[v as usize] = to32(from);
        false
    }

    /// A tetrahedron all four of whose vertices lie on one face of the surface
    /// and which has gone flat: the region it filled is the sliver between a
    /// chamfer and the rim the chamfer cut off, and with its vertices on the rim
    /// it encloses nothing. It is removed after the pass ([`Self::drop_collapsed`]).
    fn collapses_onto_face(&self, t: u32, q: f64) -> bool {
        q < COLLAPSED_SIN && self.shares_face(&self.mesh.tets[t as usize].vertices)
    }

    /// Remove the tetrahedra [`Self::collapses_onto_face`] accepts.
    fn drop_collapsed(&mut self) -> usize {
        let keep: Vec<bool> = (0..self.mesh.tets.len())
            .map(|t| {
                let (q, _) = tet_quality(self.tet_points(t as u32));
                !self.collapses_onto_face(t as u32, q)
            })
            .collect();
        let before = self.mesh.tets.len();
        let mut i = 0;
        self.mesh.tets.retain(|_| {
            let k = keep[i];
            i += 1;
            k
        });
        let dropped = before - self.mesh.tets.len();
        if dropped > 0 {
            self.rebuild_incidence();
        }
        dropped
    }

    /// Add the faces of `planes` that vertex `v` does not have yet.
    /// Returns whether the vertex's faces changed.
    fn merge_planes(&mut self, v: u32, planes: &[Plane]) -> bool {
        let before = self.planes[v as usize].len();
        self.planes[v as usize].retain(|q| q.refined);
        let mut changed = self.planes[v as usize].len() != before;
        for pl in planes {
            if !self.planes[v as usize].iter().any(|q| Self::same_face(q, pl)) {
                self.planes[v as usize].push(*pl);
                changed = true;
            }
        }
        changed
    }

    fn same_face(a: &Plane, b: &Plane) -> bool {
        dot(a.n, b.n) >= FEATURE_ANGLE_COS
    }

    /// The distinct faces among `sets`, first occurrence kept, with the
    /// position of the vertex each came from.
    fn distinct(&self, verts: &[u32]) -> (Vec<Plane>, Vec<[f64; 3]>) {
        let mut out: Vec<Plane> = Vec::new();
        let mut owners = Vec::new();
        for &v in verts {
            for pl in &self.planes[v as usize] {
                if !out.iter().any(|q| Self::same_face(q, pl)) {
                    out.push(*pl);
                    owners.push(pl.p);
                }
            }
        }
        (out, owners)
    }

    fn shares_face(&self, verts: &[u32]) -> bool {
        // some face of the first vertex is a face of every other vertex
        self.planes[verts[0] as usize].iter().any(|pl| {
            verts[1..]
                .iter()
                .all(|&v| self.planes[v as usize].iter().any(|q| Self::same_face(pl, q)))
        })
    }

    fn surface_point(&self, v: u32) -> bool {
        !self.planes[v as usize].is_empty()
    }
}

/// Make the boundary of a marching-tets mesh conform to the field: vertices on
/// the surface, and on its creases and corners. `on_surface[i]` says whether
/// vertex `i` was placed by the zero level set (and so belongs on it).
pub(crate) fn conform_boundary<F: SdfField + ?Sized>(
    mesh: &mut SdfTetMesh,
    sdf: &F,
    cell: f32,
    on_surface: Vec<bool>,
) {
    if mesh.tets.is_empty() {
        return;
    }
    if std::env::var_os("SFB_OFF").is_some() {
        return;
    }
    let h_newton = NEWTON_STEP_CELLS * cell;
    let mut w = Work {
        planes: vec![Vec::new(); mesh.vertices.len()],
        incident: Vec::new(),
        blocked: BTreeSet::new(),
        mesh,
        sdf,
        cell,
    };
    w.rebuild_incidence();
    let dbg_t = std::time::Instant::now();
    let dbg = std::env::var_os("SFB_TIME").is_some();
    if dbg { eprintln!("T start worst {:.4}", w.global_worst()); }

    // 1. projection
    let mut moves = Vec::new();
    for (v, &surface) in on_surface.iter().enumerate() {
        if !surface {
            continue;
        }
        let p = to64(w.mesh.vertices[v]);
        if let Some(q) = project(sdf, p, h_newton) {
            if q != p {
                moves.push((v as u32, q));
            }
        }
    }
    w.move_all(&moves);
    for (v, &surface) in on_surface.iter().enumerate() {
        if !surface {
            continue;
        }
        let at = to64(w.mesh.vertices[v]);
        if let Some(n) = unit(gradient(sdf, at, h_newton)) {
            w.planes[v].push(Plane {
                n,
                p: at,
                refined: false,
            });
        }
    }

    if dbg { eprintln!("T verts {} tets {} project {:?} worst {:.4}", w.mesh.vertices.len(), w.mesh.tets.len(), dbg_t.elapsed(), w.global_worst()); }
    for _ in 0..FEATURE_PASSES {
        let creases = split_creases(&mut w);
        if dbg { eprintln!("T creases {creases} {:?} worst {:.4}", dbg_t.elapsed(), w.global_worst()); }
        let corners = split_corners(&mut w);
        if dbg { eprintln!("T corners {corners} {:?} worst {:.4}", dbg_t.elapsed(), w.global_worst()); }
        let dropped = w.drop_collapsed();
        if std::env::var_os("SFB_DEBUG").is_some() { eprintln!("dropped {dropped}"); }
        // refine_guarded measured 2026-11-08: thin box at a half-cell offset
        // -0.46 % -> -0.04 %, but concave scenes 4x the elements and up to
        // +1.8 % volume; not called until that is understood
        if creases + corners == 0 {
            break;
        }
    }
}

fn boundary_faces(mesh: &SdfTetMesh) -> Vec<[u32; 3]> {
    // The marching-tets mesh is manifold by construction; a refusal here would
    // mean the mesher itself is broken, and the pass then does nothing.
    let mut faces = mesh.boundary_faces().unwrap_or_default();
    for f in &mut faces {
        f.sort_unstable();
    }
    faces.sort_unstable();
    faces
}

/// One pass of crease splitting. Returns the number of edges split.
fn split_creases<F: SdfField + ?Sized>(w: &mut Work<'_, F>) -> usize {
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for f in boundary_faces(w.mesh) {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    // Decide every split from the mesh as it stands, then apply. A vertex found
    // to sit on a crease already takes the crease's planes, and the decisions
    // are made again, because its edges along the crease are no longer creases.
    let mut relabelled = 0;
    let targets = loop {
        let mut targets: BTreeMap<(u32, u32), ([f64; 3], Vec<Plane>)> = BTreeMap::new();
        let mut relabel: Vec<(u32, Vec<Plane>)> = Vec::new();
        for &(a, b) in &edges {
            if !w.surface_point(a) || !w.surface_point(b) || w.shares_face(&[a, b]) {
                continue;
            }
            let (planes, owners) = w.distinct(&[a, b]);
            let (planes, owners) = if planes.len() > 3 {
                (planes[..3].to_vec(), owners[..3].to_vec())
            } else {
                (planes, owners)
            };
            let pa = to64(w.mesh.vertices[a as usize]);
            let pb = to64(w.mesh.vertices[b as usize]);
            let m = add_scaled(pa, 0.5, sub(pb, pa));
            let Some((x, planes)) = feature_point(w.sdf, &planes, &owners, m, w.cell) else {
                continue;
            };
            // the meeting point is within reach of the edge it replaces: no
            // further from its midpoint than the edge is long (a crease cut by
            // a long chamfer edge sits up to that far away)
            if norm(sub(x, m)) > f64::from(w.cell).max(norm(sub(pb, pa))) {
                continue;
            }
            // An end already on the crease (a lattice corner the rim passes
            // through) is the crease vertex: it takes the planes and nothing is split.
            let tol = ON_SURFACE_CELLS * f64::from(w.cell);
            // An edge that already lies along the crease needs no vertex: both ends
            // are on it.
            let ab = sub(pb, pa);
            let s_ab = crate::det_math::raw_clamp64(dot(sub(x, pa), ab) / dot(ab, ab), 0.0, 1.0);
            if norm(sub(x, add_scaled(pa, s_ab, ab))) <= tol
                && norm(sub(x, pa)) > tol
                && norm(sub(x, pb)) > tol
            {
                relabel.push((a, planes.clone()));
                relabel.push((b, planes));
                continue;
            }
            if norm(sub(x, pa)) <= tol || norm(sub(x, pb)) <= tol {
                let v = if norm(sub(x, pa)) <= norm(sub(x, pb)) { a } else { b };
                relabel.push((v, planes));
                continue;
            }
            // a target on an existing vertex near the edge (a corner the mesh
        // already has) is that vertex
        let near = [a, b]
            .iter()
            .flat_map(|&e| w.incident[e as usize].iter())
            .flat_map(|&t| w.mesh.tets[t as usize].vertices)
            .find(|&u| norm(sub(to64(w.mesh.vertices[u as usize]), x)) <= tol);
        if let Some(u) = near {
            if w.surface_point(u) {
                relabel.push((u, planes));
            }
            continue;
        }
        targets.insert((a, b), (x, planes));
        }
        if std::env::var_os("SFB_DEBUG").is_some() { eprintln!("creases: {} candidate edges, {} targets", edges.len(), targets.len()); }
        if relabel.is_empty() {
            break targets;
        }
        let mut changed = false;
        for (v, planes) in relabel {
            if w.merge_planes(v, &planes) {
                relabelled += 1;
                changed = true;
            }
        }
        // a relabel that changed nothing would be decided again the same way
        if !changed {
            break targets;
        }
    };
    if targets.is_empty() {
        return relabelled;
    }
    let proposals: Vec<((u32, u32), [f64; 3], Vec<Plane>)> =
        targets.into_iter().map(|(k, (x, pl))| (k, x, pl)).collect();
    let n = place(w, proposals, split_rec);
    if std::env::var_os("SFB_DEBUG").is_some() {
        eprintln!("creases committed {n}");
    }
    n + relabelled
}

/// Split a tetrahedron on every marked edge it holds, smallest edge first, so
/// that two tetrahedra sharing a face split it the same way.
fn split_rec(vs: [u32; 4], mid: &BTreeMap<(u32, u32), u32>, out: &mut Vec<Tetrahedron>) {
    let mut best: Option<((u32, u32), u32)> = None;
    for i in 0..4 {
        for j in (i + 1)..4 {
            let key = (vs[i].min(vs[j]), vs[i].max(vs[j]));
            if let Some(&m) = mid.get(&key) {
                if best.is_none_or(|(k, _)| key < k) {
                    best = Some((key, m));
                }
            }
        }
    }
    match best {
        None => out.push(Tetrahedron { vertices: vs }),
        Some(((a, b), m)) => {
            for child in split_children(vs, a, b, m) {
                split_rec(child, mid, out);
            }
        }
    }
}

/// One pass of corner splitting. Returns the number of faces split.
fn split_corners<F: SdfField + ?Sized>(w: &mut Work<'_, F>) -> usize {
    let faces = boundary_faces(w.mesh);
    let mut targets: Vec<([u32; 3], [f64; 3], Vec<Plane>)> = Vec::new();
    let mut relabel: Vec<(u32, Vec<Plane>)> = Vec::new();
    let mut taken: Vec<[f64; 3]> = Vec::new();
    for f in faces {
        if f.iter().any(|&v| !w.surface_point(v)) || w.shares_face(&f) {
            continue;
        }
        let pairwise = w.shares_face(&[f[0], f[1]])
            && w.shares_face(&[f[1], f[2]])
            && w.shares_face(&[f[0], f[2]]);
        if !pairwise {
            continue;
        }
        let (planes, owners) = w.distinct(&f);
        if planes.len() < 3 {
            continue;
        }
        let dbg = std::env::var_os("SFB_DEBUG").is_some();
        if dbg { eprintln!("corner cand {:?} planes {}", f.map(|v| w.mesh.vertices[v as usize]), planes.len()); }
        let pts = f.map(|v| to64(w.mesh.vertices[v as usize]));
        let c = [0, 1, 2].map(|k| (pts[0][k] + pts[1][k] + pts[2][k]) / 3.0);
        let Some((x, planes)) =
            feature_point(w.sdf, &planes[..3], &owners[..3], c, w.cell)
        else {
            if dbg { eprintln!("  feature_point failed"); }
            continue;
        };
        if dbg { eprintln!("  x {x:?}"); }
        let reach = (0..3)
            .map(|k| norm(sub(pts[k], pts[(k + 1) % 3])))
            .fold(f64::from(w.cell), f64::max);
        if norm(sub(x, c)) > reach {
            continue;
        }
        // one vertex per corner: a second face reaching for the same point is
        // left for the next pass, when the first one's vertex is in the mesh
        let tol = ON_SURFACE_CELLS * f64::from(w.cell);
        let exists = taken.iter().any(|t| norm(sub(*t, x)) <= tol)
            || (0..w.mesh.vertices.len())
                .any(|v| w.planes[v].len() >= 3 && norm(sub(to64(w.mesh.vertices[v]), x)) <= tol);
        if exists {
            continue;
        }
        if let Some(&v) = f.iter().find(|&&v| norm(sub(to64(w.mesh.vertices[v as usize]), x)) <= tol) {
            relabel.push((v, planes));
            continue;
        }
        taken.push(x);
        targets.push((f, x, planes));
    }
    let mut relabelled = 0;
    for (v, planes) in relabel {
        if w.merge_planes(v, &planes) {
            relabelled += 1;
        }
    }
    if targets.is_empty() {
        return relabelled;
    }
    let n = place(w, targets, split_face);
    if std::env::var_os("SFB_DEBUG").is_some() {
        eprintln!("corners committed {n}");
    }
    n + relabelled
}

/// Split a tetrahedron holding a marked face into three, one per face vertex
/// replaced by the new one (same winding).
fn split_face(vs: [u32; 4], mid: &BTreeMap<[u32; 3], u32>, out: &mut Vec<Tetrahedron>) {
    for skip in 0..4 {
        let mut f = [0u32; 3];
        let mut k = 0;
        for (i, &v) in vs.iter().enumerate() {
            if i != skip {
                f[k] = v;
                k += 1;
            }
        }
        f.sort_unstable();
        if let Some(&m) = mid.get(&f) {
            for i in 0..4 {
                if i != skip {
                    let mut child = vs;
                    child[i] = m;
                    out.push(Tetrahedron { vertices: child });
                }
            }
            return;
        }
    }
    out.push(Tetrahedron { vertices: vs });
}

/// Insert the proposed vertices: split at a provisional point (the centroid of
/// the split simplex, always valid), move every new vertex to its target as
/// far as the guard and local smoothing allow, then keep only the vertices
/// that reached their target and re-split the original elements on those
/// alone through [`commit`], which re-checks every child. Returns the number
/// of vertices kept.
fn place<F, K>(
    w: &mut Work<'_, F>,
    proposals: Vec<(K, [f64; 3], Vec<Plane>)>,
    split: fn([u32; 4], &BTreeMap<K, u32>, &mut Vec<Tetrahedron>),
) -> usize
where
    F: SdfField + ?Sized,
    K: Ord + Copy + Simplex,
{
    if proposals.is_empty() {
        return 0;
    }
    let base = w.mesh.vertices.len();
    let old = w.mesh.tets.clone();
    let snapshot = w.mesh.vertices.clone();
    let mut mid: BTreeMap<K, u32> = BTreeMap::new();
    for (k, _, planes) in &proposals {
        let c = k.centroid(w);
        mid.insert(*k, w.mesh.vertices.len() as u32);
        w.mesh.vertices.push(to32(c));
        w.planes.push(planes.clone());
    }
    let mut split_tets = Vec::with_capacity(old.len() + 4 * proposals.len());
    for tet in &old {
        split(tet.vertices, &mid, &mut split_tets);
    }
    w.mesh.tets = split_tets;
    w.rebuild_incidence();
    let moves: Vec<(u32, [f64; 3])> = proposals.iter().map(|(k, x, _)| (mid[k], *x)).collect();
    w.move_all(&moves);
    let (reached, missed): (Vec<_>, Vec<_>) = proposals
        .into_iter()
        .partition(|(k, x, _)| w.mesh.vertices[mid[k] as usize] == to32(*x));
    for (k, _, _) in &missed {
        w.blocked.extend(k.verts());
    }
    if std::env::var_os("SFB_DEBUG").is_some() {
        eprintln!("place: proposed {} reached {}", mid.len(), reached.len());
    }
    commit(w, base, old, &snapshot, reached, split)
}

/// The simplex a proposal splits: its centroid is the provisional position.
trait Simplex {
    fn centroid<F: SdfField + ?Sized>(&self, w: &Work<'_, F>) -> [f64; 3];
    fn verts(&self) -> Vec<u32>;
}

impl Simplex for (u32, u32) {
    fn centroid<F: SdfField + ?Sized>(&self, w: &Work<'_, F>) -> [f64; 3] {
        let a = to64(w.mesh.vertices[self.0 as usize]);
        let b = to64(w.mesh.vertices[self.1 as usize]);
        add_scaled(a, 0.5, sub(b, a))
    }
    fn verts(&self) -> Vec<u32> {
        vec![self.0, self.1]
    }
}

impl Simplex for [u32; 3] {
    fn centroid<F: SdfField + ?Sized>(&self, w: &Work<'_, F>) -> [f64; 3] {
        let p = self.map(|v| to64(w.mesh.vertices[v as usize]));
        [0, 1, 2].map(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0)
    }
    fn verts(&self) -> Vec<u32> {
        self.to_vec()
    }
}

/// Make room around blocked proposals: bisect the longest edge of each element
/// around them, interior edges only, through [`commit`], so a bisection is
/// kept only if none of its children inverts or falls under the floor (or
/// under its parent). Returns the number of edges split.
#[allow(dead_code, reason = "measured and parked, see the pass loop")]
fn refine_guarded<F: SdfField + ?Sized>(w: &mut Work<'_, F>) -> usize {
    if w.blocked.is_empty() {
        return 0;
    }
    let mut boundary_edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    for f in boundary_faces(w.mesh) {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])] {
            boundary_edges.insert((a.min(b), a.max(b)));
        }
    }
    let mut proposals: BTreeMap<(u32, u32), [f64; 3]> = BTreeMap::new();
    for tet in &w.mesh.tets {
        if !tet.vertices.iter().any(|v| w.blocked.contains(v)) {
            continue;
        }
        let mut longest: Option<((u32, u32), f64)> = None;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let (a, b) = (tet.vertices[i], tet.vertices[j]);
                let key = (a.min(b), a.max(b));
                let l = norm(sub(to64(w.mesh.vertices[b as usize]), to64(w.mesh.vertices[a as usize])));
                if longest.is_none_or(|(k, m)| l > m || (l == m && key < k)) {
                    longest = Some((key, l));
                }
            }
        }
        if let Some((key, _)) = longest {
            if !boundary_edges.contains(&key) {
                let pa = to64(w.mesh.vertices[key.0 as usize]);
                let pb = to64(w.mesh.vertices[key.1 as usize]);
                proposals.insert(key, add_scaled(pa, 0.5, sub(pb, pa)));
            }
        }
    }
    w.blocked.clear();
    let base = w.mesh.vertices.len();
    let old = w.mesh.tets.clone();
    let snapshot = w.mesh.vertices.clone();
    let list = proposals.into_iter().map(|(k, x)| (k, x, Vec::new())).collect();
    commit(w, base, old, &snapshot, list, split_rec)
}

/// Insert a vertex per proposal at its target, split the tetrahedra that hold
/// it, and check every child: none may invert, and none may fall under the
/// dihedral floor (or under its parent, if the parent was already lower),
/// except a sliver that collapses onto one face of the surface (dropped
/// after the pass). A proposal whose vertex sits in a failing child is
/// withdrawn and the rest are tried again from the unsplit mesh, so every
/// accepted vertex is placed exactly at its target and no element is ever
/// made worse than the floor. Returns the number of vertices inserted.
fn commit<F, K>(
    w: &mut Work<'_, F>,
    base: usize,
    old: Vec<Tetrahedron>,
    snapshot: &[[f32; 3]],
    mut proposals: Vec<(K, [f64; 3], Vec<Plane>)>,
    split: fn([u32; 4], &BTreeMap<K, u32>, &mut Vec<Tetrahedron>),
) -> usize
where
    F: SdfField + ?Sized,
    K: Ord + Copy,
{
    // the reference every element is held to: its quality before this step
    let q_ref: Vec<f64> = old
        .iter()
        .map(|t| tet_quality(t.vertices.map(|v| to64(snapshot[v as usize]))).0)
        .collect();
    loop {
        w.mesh.vertices.truncate(base);
        w.planes.truncate(base);
        let mut mid: BTreeMap<K, u32> = BTreeMap::new();
        for (k, x, planes) in &proposals {
            mid.insert(*k, w.mesh.vertices.len() as u32);
            w.mesh.vertices.push(to32(*x));
            w.planes.push(planes.clone());
        }
        let mut out: Vec<Tetrahedron> = Vec::with_capacity(old.len() + 4 * proposals.len());
        let mut bad: BTreeSet<u32> = BTreeSet::new();
        let mut restored = false;
        for (tet, &q0) in old.iter().zip(&q_ref) {
            let start = out.len();
            split(tet.vertices, &mid, &mut out);
            let floor = q0.min(floor_sin());
            for child in &out[start..] {
                let p = child.vertices.map(|v| to64(w.mesh.vertices[v as usize]));
                let (q, vol6) = tet_quality(p);
                let collapsed = q < COLLAPSED_SIN && w.shares_face(&child.vertices);
                if collapsed || (vol6 > 0.0 && q >= floor) {
                    continue;
                }
                let new: Vec<u32> = child.vertices.iter().copied().filter(|&v| v as usize >= base).collect();
                if new.is_empty() {
                    // an element no proposal split, made worse by smoothing:
                    // put its vertices back
                    for v in child.vertices {
                        if w.mesh.vertices[v as usize] != snapshot[v as usize] {
                            w.mesh.vertices[v as usize] = snapshot[v as usize];
                            restored = true;
                        }
                    }
                } else {
                    bad.extend(new);
                }
            }
        }
        if bad.is_empty() && !restored {
            w.mesh.tets = out;
            w.rebuild_incidence();
            return proposals.len();
        }
        let mut i = 0;
        proposals.retain(|_| {
            let keep = !bad.contains(&((base + i) as u32));
            i += 1;
            keep
        });
    }
}
