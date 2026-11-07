//! Reflex creases: the elements the linear cut bridges across a cavity's
//! inside edge.
//!
//! Along a reflex crease (two faces meeting at an inside edge, the field being
//! `max` of two signed distances) the zero set of the field's linear
//! interpolant on a lattice element cuts across the corner from the cavity
//! side, so the mesh holds a wedge of the cavity. Measured on a cube with a
//! cubic cavity at cell 0.25: 158 such elements with the warp off, 25 at the
//! warp fraction 0.45. The wedge's faces are shared between elements, so no
//! per-element re-cut and no vertex move removes it (the interior vertices
//! lie on the far side of the crease).
//!
//! Three steps, each a local topological change that keeps the mesh
//! conforming, in index order:
//!
//! 1. **Insert** the crease point (where the tangent planes of the faces met
//!    by an outside element's surface vertices meet) into every element whose
//!    closure holds it, by replacing in turn each vertex whose barycentric
//!    weight is positive (a 1-4 split inside an element, 2 x 3 on a face,
//!    `n` x 2 on an edge).
//! 2. **Remove chords**: a boundary edge between two surface vertices whose
//!    midpoint is outside is replaced, with the ring of elements around it, by
//!    a fan triangulation of the ring polygon joined to each end (edge removal
//!    on an open ring, after Klingner and Shewchuk, *Aggressive Tetrahedral
//!    Mesh Improvement*, 2007). The new boundary runs through the ring's end
//!    vertices instead of across the cavity.
//! 3. **Carve** elements whose four vertices are on the surface and whose
//!    centroid is outside, when the boundary stays a manifold.
//!
//! Each change is kept only if no new element inverts, none reaches outside
//! and none falls under the dihedral floor (or under the worst element it
//! replaces, if that was lower).

use super::*;

/// Rounds of insert / remove / carve.
const REFLEX_ROUNDS: usize = 4;

/// Barycentric weights under this count as zero when locating a point.
const BARY_EPS: f64 = 1.0e-9;

/// Counts of what [`repair_reflex`] did.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ReflexStats {
    pub(crate) inserted: usize,
    pub(crate) chords_removed: usize,
    pub(crate) carved: usize,
    pub(crate) insert_refused: usize,
    pub(crate) chord_refused: usize,
}

pub(super) fn repair_reflex<F: SdfField + ?Sized>(w: &mut Work<'_, F>) -> ReflexStats {
    let mut st = ReflexStats::default();
    let saved = (w.mesh.clone(), w.planes.clone());
    let floor_before = w.global_worst().min(floor_sin());
    for _ in 0..REFLEX_ROUNDS {
        let before = (st.inserted, st.chords_removed, st.carved);
        insert_crease_points(w, &mut st);
        remove_chords(w, &mut st);
        carve(w, &mut st);
        if before == (st.inserted, st.chords_removed, st.carved) {
            break;
        }
    }
    if w.global_worst() < floor_before {
        if std::env::var_os("SFB_REFLEX").is_some() {
            eprintln!("REFLEX reverted: worst {:.4} under {floor_before:.4} (outside left {})", w.global_worst(), w.outside_count());
        }
        *w.mesh = saved.0;
        w.planes = saved.1;
        w.rebuild_incidence();
    }
    st
}

fn worst_of(w: &Work<'_, impl SdfField + ?Sized>, tets: &[[u32; 4]]) -> f64 {
    tets.iter()
        .map(|t| tet_quality(t.map(|v| to64(w.mesh.vertices[v as usize]))).0)
        .fold(f64::INFINITY, f64::min)
}

/// Volume (x6) of the elements whose centroid is outside.
fn outside_vol6<F: SdfField + ?Sized>(w: &Work<'_, F>, tets: &[[u32; 4]]) -> f64 {
    tets.iter()
        .filter(|t| w.reaches_outside(t))
        .map(|t| tet_quality(t.map(|v| to64(w.mesh.vertices[v as usize]))).1.abs())
        .sum()
}

/// Why a change was refused, for the counts.
pub(crate) static REFUSE: [std::sync::atomic::AtomicUsize; 3] = [const { std::sync::atomic::AtomicUsize::new(0) }; 3];

/// New elements are acceptable against the ones they replace: none
/// degenerate, none under the floor (or under the worst replaced element),
/// and no more volume outside than before.
fn acceptable<F: SdfField + ?Sized>(w: &Work<'_, F>, new: &[[u32; 4]], old: &[[u32; 4]]) -> bool {
    acceptable_with(w, new, old, true)
}

fn acceptable_with<F: SdfField + ?Sized>(w: &Work<'_, F>, new: &[[u32; 4]], old: &[[u32; 4]], floor: bool) -> bool {
    let old_worst = if floor { worst_of(w, old) } else { 0.0 };
    for t in new {
        let (q, vol6) = tet_quality(t.map(|v| to64(w.mesh.vertices[v as usize])));
        if vol6 == 0.0 {
            REFUSE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        if q < old_worst.min(floor_sin()) {
            REFUSE[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
    }
    if outside_vol6(w, new) > outside_vol6(w, old) {
        REFUSE[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    true
}

fn insert_crease_points<F: SdfField + ?Sized>(w: &mut Work<'_, F>, st: &mut ReflexStats) {
    let tol = ON_SURFACE_CELLS * f64::from(w.cell);
    let outside: Vec<[u32; 4]> = w
        .mesh
        .tets
        .iter()
        .map(|t| t.vertices)
        .filter(|vs| w.reaches_outside(vs))
        .collect();
    let mut targets: Vec<([f64; 3], Vec<Plane>)> = Vec::new();
    for vs in outside {
        let surf: Vec<u32> = vs.iter().copied().filter(|&v| w.surface_point(v)).collect();
        if surf.len() < 2 {
            continue;
        }
        let (planes, owners) = w.distinct(&surf);
        if planes.len() < 2 {
            continue;
        }
        let k = planes.len().min(3);
        let p = vs.map(|v| to64(w.mesh.vertices[v as usize]));
        let c = [0, 1, 2].map(|i| (p[0][i] + p[1][i] + p[2][i] + p[3][i]) / 4.0);
        let Some((x, pl)) = feature_point(w.sdf, &planes[..k], &owners[..k], c, w.cell) else {
            continue;
        };
        if targets.iter().any(|(y, _)| norm(sub(*y, x)) <= tol)
            || w.mesh.vertices.iter().any(|v| norm(sub(to64(*v), x)) <= tol)
        {
            continue;
        }
        targets.push((x, pl));
    }
    for (x, planes) in targets {
        if insert_point(w, x, planes) {
            st.inserted += 1;
        } else {
            st.insert_refused += 1;
        }
    }
}

/// Insert `x` into every element whose closure holds it. Returns whether it
/// was inserted.
fn insert_point<F: SdfField + ?Sized>(w: &mut Work<'_, F>, x: [f64; 3], planes: Vec<Plane>) -> bool {
    let xf = to32(x);
    let mut hits: Vec<(usize, [f64; 4])> = Vec::new();
    for (i, t) in w.mesh.tets.iter().enumerate() {
        let p = t.vertices.map(|v| to64(w.mesh.vertices[v as usize]));
        let total = dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])));
        if total == 0.0 {
            continue;
        }
        let mut b = [0.0; 4];
        for k in 0..4 {
            let mut q = p;
            q[k] = to64(xf);
            b[k] = dot(sub(q[1], q[0]), cross(sub(q[2], q[0]), sub(q[3], q[0]))) / total;
        }
        if b.iter().all(|&y| y >= -BARY_EPS) {
            hits.push((i, b));
        }
    }
    if hits.is_empty() {
        return false;
    }
    let v = w.mesh.vertices.len() as u32;
    w.mesh.vertices.push(xf);
    w.planes.push(planes);
    let mut new: Vec<[u32; 4]> = Vec::new();
    let mut old: Vec<[u32; 4]> = Vec::new();
    for &(i, b) in &hits {
        let vs = w.mesh.tets[i].vertices;
        old.push(vs);
        for k in 0..4 {
            if b[k] > BARY_EPS {
                let mut child = vs;
                child[k] = v;
                new.push(child);
            }
        }
    }
    // an inserted crease point sits on faces of the surface, so its star
    // split has thin children by construction; they are what the chord
    // removal and the carving then take apart, and the pass as a whole is
    // held to the floor at the end
    if !acceptable_with(w, &new, &old, false) {
        w.mesh.vertices.pop();
        w.planes.pop();
        return false;
    }
    let drop: BTreeSet<usize> = hits.iter().map(|&(i, _)| i).collect();
    let mut i = 0;
    w.mesh.tets.retain(|_| {
        let keep = !drop.contains(&i);
        i += 1;
        keep
    });
    for t in new {
        crate::sdf_fem_mesh::push_tet_pub(w.mesh, t);
    }
    w.rebuild_incidence();
    true
}

fn remove_chords<F: SdfField + ?Sized>(w: &mut Work<'_, F>, st: &mut ReflexStats) {
    let tol = OUTSIDE_CELLS * f64::from(w.cell);
    let mut chords: Vec<(u32, u32)> = Vec::new();
    for f in boundary_faces(w.mesh) {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])] {
            let (a, b) = (a.min(b), a.max(b));
            if chords.contains(&(a, b)) || !w.surface_point(a) || !w.surface_point(b) {
                continue;
            }
            let pa = to64(w.mesh.vertices[a as usize]);
            let pb = to64(w.mesh.vertices[b as usize]);
            if field(w.sdf, add_scaled(pa, 0.5, sub(pb, pa))) > tol {
                chords.push((a, b));
            }
        }
    }
    chords.sort_unstable();
    for (a, b) in chords {
        if remove_edge(w, a, b) {
            st.chords_removed += 1;
        } else {
            st.chord_refused += 1;
        }
    }
}

/// Edge removal on the open ring of a boundary edge. Returns whether the edge
/// was removed.
fn remove_edge<F: SdfField + ?Sized>(w: &mut Work<'_, F>, a: u32, b: u32) -> bool {
    let ring_tets: Vec<usize> = w.incident[a as usize]
        .iter()
        .map(|&t| t as usize)
        .filter(|&t| w.mesh.tets[t].vertices.contains(&b))
        .collect();
    if ring_tets.is_empty() {
        return false;
    }
    // the other two vertices of each element around the edge
    let pairs: Vec<(u32, u32)> = ring_tets
        .iter()
        .map(|&t| {
            let o: Vec<u32> = w.mesh.tets[t].vertices.iter().copied().filter(|&v| v != a && v != b).collect();
            (o[0], o[1])
        })
        .collect();
    // order them into a path r0 .. r(n-1): an open ring has two ends seen once
    let mut count: BTreeMap<u32, usize> = BTreeMap::new();
    for &(x, y) in &pairs {
        *count.entry(x).or_default() += 1;
        *count.entry(y).or_default() += 1;
    }
    let ends: Vec<u32> = count.iter().filter(|(_, &c)| c == 1).map(|(&v, _)| v).collect();
    if ends.len() != 2 || count.values().any(|&c| c > 2) {
        return false;
    }
    let mut path = vec![ends[0]];
    let mut used = vec![false; pairs.len()];
    while path.len() <= pairs.len() {
        let last = *path.last().expect("non-empty");
        let Some(k) = (0..pairs.len()).find(|&k| !used[k] && (pairs[k].0 == last || pairs[k].1 == last)) else {
            break;
        };
        used[k] = true;
        path.push(if pairs[k].0 == last { pairs[k].1 } else { pairs[k].0 });
    }
    if path.len() != pairs.len() + 1 || path.len() < 3 {
        return false;
    }
    let old: Vec<[u32; 4]> = ring_tets.iter().map(|&t| w.mesh.tets[t].vertices).collect();
    let n = path.len();
    let pos = |v: u32| to64(w.mesh.vertices[v as usize]);
    let mut best: Option<(f64, Vec<[u32; 4]>)> = None;
    for apex in 0..n {
        let mut new = Vec::new();
        let mut ok = true;
        for j in 1..n - 1 {
            let (r1, r2) = (path[(apex + j) % n], path[(apex + j + 1) % n]);
            let tri = [path[apex], r1, r2];
            let pa = pos(a);
            let pb = pos(b);
            let o = |p: [f64; 3]| {
                let t = tri.map(pos);
                dot(sub(t[1], t[0]), cross(sub(t[2], t[0]), sub(p, t[0])))
            };
            let (sa, sb) = (o(pa), o(pb));
            if !(sa * sb < 0.0) {
                ok = false;
                break;
            }
            new.push([tri[0], tri[1], tri[2], a]);
            new.push([tri[0], tri[1], tri[2], b]);
        }
        if !ok || !acceptable(w, &new, &old) {
            continue;
        }
        let q = worst_of(w, &new);
        if best.as_ref().is_none_or(|(bq, _)| q > *bq) {
            best = Some((q, new));
        }
    }
    let Some((_, new)) = best else {
        return false;
    };
    let drop: BTreeSet<usize> = ring_tets.into_iter().collect();
    let mut i = 0;
    w.mesh.tets.retain(|_| {
        let keep = !drop.contains(&i);
        i += 1;
        keep
    });
    for t in new {
        crate::sdf_fem_mesh::push_tet_pub(w.mesh, t);
    }
    w.rebuild_incidence();
    true
}

fn carve<F: SdfField + ?Sized>(w: &mut Work<'_, F>, st: &mut ReflexStats) {
    let mut i = 0;
    while i < w.mesh.tets.len() {
        let vs = w.mesh.tets[i].vertices;
        if vs.iter().all(|&v| w.surface_point(v)) && w.reaches_outside(&vs) {
            let removed = w.mesh.tets.remove(i);
            if w.mesh.boundary_faces().is_ok() {
                st.carved += 1;
                continue;
            }
            w.mesh.tets.insert(i, removed);
        }
        i += 1;
    }
    w.rebuild_incidence();
}
