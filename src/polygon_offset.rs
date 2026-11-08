//! Inward offset (erosion) of a polygonal domain with holes: the boundary of
//! the set of points of the domain at least `delta` from its boundary.
//!
//! Each edge is shifted inward by `delta`. At a convex vertex the two shifted
//! edges meet at a point (erosion keeps a convex corner sharp); at a reflex
//! vertex the eroded boundary is an arc of radius `delta` about the vertex,
//! flattened into chords whose sagitta stays under [`OffsetConfig::sagitta`]
//! (subdivided by bisecting the end normals, so no trigonometry and equal
//! angles). An edge whose shifted segment comes out reversed (a run of short
//! edges around a convex bend of radius under `delta`) is dropped and its
//! neighbours are intersected instead, until none is reversed.
//!
//! Each output vertex carries its outward normal(s) with respect to the
//! eroded domain, which is what a rounded rim of radius `delta` swept back
//! out from it needs: an edge point has the edge's normal, a convex corner the
//! fan between its two edges' normals, an arc point the direction to its
//! centre.
//!
//! # Refusals
//!
//! A result that would not be a valid domain is an error, never a returned
//! polygon: a contour that erodes away entirely, two shifted lines left
//! parallel after a drop, and output contours that intersect themselves or
//! each other (a stroke thinner than `2 delta`).

use crate::det_math::{acos64, sqrt64};

/// Parameters of [`erode`].
#[derive(Debug, Clone, Copy)]
pub struct OffsetConfig {
    /// Largest distance between an arc and its chords (input units).
    pub sagitta: f64,
}

/// Why an erosion was refused.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OffsetError {
    /// `delta` or the sagitta is not positive and finite.
    BadConfig,
    /// A contour has fewer than three distinct points or no area.
    DegenerateContour {
        /// Index of the contour.
        contour: usize,
    },
    /// Every edge of a contour reversed: the contour erodes away.
    Vanished {
        /// Index of the contour.
        contour: usize,
    },
    /// Two shifted lines left adjacent by a drop are parallel.
    Parallel {
        /// Index of the contour.
        contour: usize,
    },
    /// Output contours cross themselves or each other.
    SelfIntersection,
}

/// The outward normal(s) of an eroded vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OffsetNormal {
    /// On a shifted edge: one normal.
    Edge([f64; 2]),
    /// A convex corner: the fan from the first to the second normal
    /// (turning counter-clockwise, less than half a turn).
    Corner([f64; 2], [f64; 2]),
    /// On an arc about `centre`: the normal points at the centre.
    Arc {
        /// The reflex vertex of the input the arc is about.
        centre: [f64; 2],
        /// Unit vector from the point to the centre.
        normal: [f64; 2],
    },
}

/// A vertex of an eroded contour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffsetVertex {
    /// Position.
    pub p: [f64; 2],
    /// Outward normal(s) of the eroded domain at `p`.
    pub normal: OffsetNormal,
}

/// The eroded domain: contours oriented with the inside on their left (the
/// outer one counter-clockwise, holes clockwise).
#[derive(Debug, Clone, PartialEq)]
pub struct Eroded {
    /// One per input contour, in input order.
    pub contours: Vec<Vec<OffsetVertex>>,
}

impl Eroded {
    /// The contours as plain points.
    #[must_use]
    pub fn points(&self) -> Vec<Vec<[f64; 2]>> {
        self.contours
            .iter()
            .map(|c| c.iter().map(|v| v.p).collect())
            .collect()
    }
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn unit(a: [f64; 2]) -> [f64; 2] {
    let l = sqrt64(dot(a, a));
    [a[0] / l, a[1] / l]
}
fn area2(c: &[[f64; 2]]) -> f64 {
    (0..c.len())
        .map(|i| cross(c[i], c[(i + 1) % c.len()]))
        .sum()
}

/// A shifted edge: a point on it and its direction.
#[derive(Clone, Copy)]
struct Line {
    p: [f64; 2],
    d: [f64; 2],
    /// Inward unit normal of the original edge.
    n: [f64; 2],
    /// Index of the original edge (its start vertex).
    edge: usize,
}

fn intersect(a: &Line, b: &Line) -> Option<[f64; 2]> {
    let den = cross(a.d, b.d);
    if den == 0.0 {
        return None;
    }
    let t = cross(sub(b.p, a.p), b.d) / den;
    Some([a.p[0] + t * a.d[0], a.p[1] + t * a.d[1]])
}

/// Erode `contours` by `delta`.
///
/// # Errors
///
/// See [`OffsetError`].
pub fn erode(
    contours: &[Vec<[f64; 2]>],
    delta: f64,
    cfg: &OffsetConfig,
) -> Result<Eroded, OffsetError> {
    if !(delta > 0.0 && delta.is_finite() && cfg.sagitta > 0.0 && cfg.sagitta.is_finite()) {
        return Err(OffsetError::BadConfig);
    }
    // clean and orient: the largest contour's sign decides; inside on the left
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    for (ci, c) in contours.iter().enumerate() {
        let mut r: Vec<[f64; 2]> = Vec::new();
        for &p in c {
            if r.last() != Some(&p) {
                r.push(p);
            }
        }
        while r.len() > 1 && r.first() == r.last() {
            r.pop();
        }
        if r.len() < 3 || area2(&r) == 0.0 {
            return Err(OffsetError::DegenerateContour { contour: ci });
        }
        rings.push(r);
    }
    let Some(outer) = (0..rings.len()).max_by(|&a, &b| {
        area2(&rings[a])
            .abs()
            .total_cmp(&area2(&rings[b]).abs())
            .then(b.cmp(&a))
    }) else {
        return Ok(Eroded {
            contours: Vec::new(),
        });
    };
    if area2(&rings[outer]) < 0.0 {
        for r in &mut rings {
            r.reverse();
        }
    }
    let mut out = Vec::with_capacity(rings.len());
    for (ci, r) in rings.iter().enumerate() {
        let ring = erode_ring(r, ci, delta, cfg)?;
        // every eroded point is `delta` from the boundary, and the contour
        // keeps its orientation: otherwise the drops left a contour that is
        // not the erosion (it eroded away and folded over)
        let pts: Vec<[f64; 2]> = ring.iter().map(|v| v.p).collect();
        let floor = delta * (1.0 - 1.0e-9);
        let own = std::slice::from_ref(r);
        if pts.iter().any(|&p| boundary_distance(own, p) < floor) || area2(&pts) * area2(r) <= 0.0 {
            return Err(OffsetError::Vanished { contour: ci });
        }
        // closer than `delta` to another contour: the wall between them is
        // thinner than `2 delta`
        if pts.iter().any(|&p| boundary_distance(&rings, p) < floor) {
            return Err(OffsetError::SelfIntersection);
        }
        out.push(ring);
    }
    let eroded = Eroded { contours: out };
    if crosses(&eroded.points()) {
        return Err(OffsetError::SelfIntersection);
    }
    Ok(eroded)
}

fn erode_ring(
    r: &[[f64; 2]],
    ci: usize,
    delta: f64,
    cfg: &OffsetConfig,
) -> Result<Vec<OffsetVertex>, OffsetError> {
    let n = r.len();
    let lines: Vec<Line> = (0..n)
        .map(|i| {
            let d = sub(r[(i + 1) % n], r[i]);
            let u = unit(d);
            let nrm = [-u[1], u[0]];
            Line {
                p: [r[i][0] + delta * nrm[0], r[i][1] + delta * nrm[1]],
                d,
                n: nrm,
                edge: i,
            }
        })
        .collect();
    // active edges, in order; drop reversed ones until none is
    let mut active: Vec<Line> = lines;
    loop {
        if active.len() < 3 {
            return Err(OffsetError::Vanished { contour: ci });
        }
        let m = active.len();
        let mut reversed: Option<usize> = None;
        for k in 0..m {
            let (prev, cur, next) = (&active[(k + m - 1) % m], &active[k], &active[(k + 1) % m]);
            // two neighbours left parallel by drops: the strip between them
            // has eroded away
            let s = start_of(prev, cur, r, delta, ci)
                .map_err(|_| OffsetError::Vanished { contour: ci })?;
            let e = start_of(cur, next, r, delta, ci)
                .map_err(|_| OffsetError::Vanished { contour: ci })?;
            let (s, e) = (s.1, e.0);
            if dot(sub(e, s), cur.d) < 0.0 {
                reversed = Some(k);
                break;
            }
        }
        match reversed {
            Some(k) => {
                active.remove(k);
            }
            None => break,
        }
    }
    // emit
    let m = active.len();
    let mut verts: Vec<OffsetVertex> = Vec::new();
    for k in 0..m {
        let (prev, cur) = (&active[(k + m - 1) % m], &active[k]);
        let adjacent = (prev.edge + 1) % r.len() == cur.edge;
        let turn = cross(prev.d, cur.d);
        if adjacent && turn < 0.0 {
            // reflex vertex: arc about it, from the previous normal to this one
            let v = r[cur.edge];
            let arc = arc_normals(prev.n, cur.n, delta, cfg.sagitta);
            for nrm in arc {
                let p = [v[0] + delta * nrm[0], v[1] + delta * nrm[1]];
                verts.push(OffsetVertex {
                    p,
                    normal: OffsetNormal::Arc {
                        centre: v,
                        normal: [-nrm[0], -nrm[1]],
                    },
                });
            }
        } else if adjacent && turn == 0.0 {
            let v = r[cur.edge];
            let p = [v[0] + delta * cur.n[0], v[1] + delta * cur.n[1]];
            verts.push(OffsetVertex {
                p,
                normal: OffsetNormal::Edge([-cur.n[0], -cur.n[1]]),
            });
        } else {
            let p = intersect(prev, cur).ok_or(OffsetError::Parallel { contour: ci })?;
            verts.push(OffsetVertex {
                p,
                normal: OffsetNormal::Corner([-prev.n[0], -prev.n[1]], [-cur.n[0], -cur.n[1]]),
            });
        }
    }
    Ok(verts)
}

/// The end of the shifted `prev` and the start of the shifted `cur` at the
/// corner between them.
fn start_of(
    prev: &Line,
    cur: &Line,
    r: &[[f64; 2]],
    delta: f64,
    ci: usize,
) -> Result<([f64; 2], [f64; 2]), OffsetError> {
    let adjacent = (prev.edge + 1) % r.len() == cur.edge;
    if adjacent && cross(prev.d, cur.d) <= 0.0 {
        let v = r[cur.edge];
        Ok((
            [v[0] + delta * prev.n[0], v[1] + delta * prev.n[1]],
            [v[0] + delta * cur.n[0], v[1] + delta * cur.n[1]],
        ))
    } else {
        let p = intersect(prev, cur).ok_or(OffsetError::Parallel { contour: ci })?;
        Ok((p, p))
    }
}

/// Unit normals from `a` to `b` (turning clockwise through less than half a
/// turn) at equal angles, fine enough that the chords of radius `delta` stay
/// within `sagitta`; both ends included, except that an arc one chord covers
/// is given by its middle normal alone.
fn arc_normals(a: [f64; 2], b: [f64; 2], delta: f64, sagitta: f64) -> Vec<[f64; 2]> {
    let phi = acos64(crate::det_math::raw_clamp64(dot(a, b), -1.0, 1.0));
    // a chord over angle t has sagitta delta (1 - cos(t / 2))
    let max_t = if sagitta >= delta {
        std::f64::consts::PI
    } else {
        2.0 * acos64(1.0 - sagitta / delta)
    };
    // an arc one chord covers is its midpoint alone: its two ends would sit a
    // fraction of a chord apart (a slight reflex bend of a flattened curve),
    // and the midpoint is on the arc within the same sagitta
    if phi <= max_t {
        return vec![unit([a[0] + b[0], a[1] + b[1]])];
    }
    let mut out = vec![a, b];
    let mut t = phi;
    while t > max_t {
        let mut next = Vec::with_capacity(out.len() * 2);
        for w in out.windows(2) {
            next.push(w[0]);
            next.push(unit([w[0][0] + w[1][0], w[0][1] + w[1][1]]));
        }
        next.push(*out.last().expect("non-empty"));
        out = next;
        t *= 0.5;
    }
    out
}

/// Distance from `p` to the nearest segment of the rings.
fn boundary_distance(rings: &[Vec<[f64; 2]>], p: [f64; 2]) -> f64 {
    let mut best = f64::INFINITY;
    for r in rings {
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            let e = sub(b, a);
            let s = crate::det_math::raw_clamp64(dot(sub(p, a), e) / dot(e, e), 0.0, 1.0);
            let q = [a[0] + s * e[0], a[1] + s * e[1]];
            best = best.min(sqrt64(dot(sub(p, q), sub(p, q))));
        }
    }
    best
}

/// Whether any two non-adjacent segments of the contours cross or touch.
fn crosses(contours: &[Vec<[f64; 2]>]) -> bool {
    // (start, end, contour, index in contour, contour length)
    type Seg = ([f64; 2], [f64; 2], usize, usize, usize);
    let mut segs: Vec<Seg> = Vec::new();
    for (ci, c) in contours.iter().enumerate() {
        for i in 0..c.len() {
            segs.push((c[i], c[(i + 1) % c.len()], ci, i, c.len()));
        }
    }
    for x in 0..segs.len() {
        for y in (x + 1)..segs.len() {
            let (a, b, ca, ia, na) = segs[x];
            let (c, d, cb, ib, _) = segs[y];
            if ca == cb && ((ia + 1) % na == ib || (ib + 1) % na == ia || ia == ib) {
                continue;
            }
            let o1 = cross(sub(b, a), sub(c, a));
            let o2 = cross(sub(b, a), sub(d, a));
            let o3 = cross(sub(d, c), sub(a, c));
            let o4 = cross(sub(d, c), sub(b, c));
            if o1 * o2 <= 0.0 && o3 * o4 <= 0.0 && !(o1 == 0.0 && o2 == 0.0) {
                return true;
            }
        }
    }
    false
}
