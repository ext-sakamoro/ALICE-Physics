//! Inward offset (erosion) of a polygonal domain with holes: the boundary of
//! the set of points of the domain at least `delta` from its boundary.
//!
//! Each edge is shifted inward by `delta`. At a convex vertex the two shifted
//! edges meet at a point (erosion keeps a convex corner sharp); at a reflex
//! vertex the eroded boundary is an arc of radius `delta` about the vertex,
//! stood for by the polygon of its tangents at equal angles (subdivided by
//! bisecting the normals, no trigonometry), which lies outside the circle by at
//! most [`OffsetConfig::sagitta`]. Every junction is then the intersection of
//! two lines. A line whose stretch between its two junctions comes out reversed
//! (an edge eaten by its convex corners, or the part of an arc the erosion cuts
//! off) is dropped and its neighbours are intersected instead, until none is.
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
//! parallel and apart after a drop, and output contours that intersect
//! themselves or each other (a stroke thinner than `2 delta`). So is input
//! that is not a domain: no contours, a coordinate that is not finite, a
//! contour with fewer than three points, no area or a half turn (a zero-width
//! spike or slit), and a contour whose winding disagrees with its nesting.
//!
//! # Winding
//!
//! The contour with the largest absolute area sets the outside's winding. A
//! contour inside an even number of others runs the same way (an island, like
//! the dot of an `i`), one inside an odd number runs the other way (a hole).
//! Two lines left on one line by drops (a notch shorter than the offset on a
//! straight edge) are merged back into one.

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
    /// No contours.
    NoContours,
    /// A coordinate is not finite.
    NonFinite {
        /// Index of the contour.
        contour: usize,
    },
    /// A contour's winding disagrees with how deep it is nested: a contour
    /// inside an odd number of others is a hole and must run against the
    /// outermost one, a contour inside an even number runs with it.
    WindingMismatch {
        /// Index of the contour.
        contour: usize,
    },
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
    /// Inward unit normal (of the edge, or of the arc at the tangent point).
    n: [f64; 2],
    /// Position in the ring's element sequence.
    seq: usize,
    /// Position of the last element this line stands for (the same as `seq`
    /// unless collinear neighbours were merged into it).
    last_seq: usize,
    /// The reflex vertex whose arc the junction at this line's end belongs
    /// to, if it does.
    end_reflex: Option<[f64; 2]>,
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
    if contours.is_empty() {
        return Err(OffsetError::NoContours);
    }
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    for (ci, c) in contours.iter().enumerate() {
        if c.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
            return Err(OffsetError::NonFinite { contour: ci });
        }
        let mut r: Vec<[f64; 2]> = Vec::new();
        for &p in c {
            if r.last() != Some(&p) {
                r.push(p);
            }
        }
        while r.len() > 1 && r.first() == r.last() {
            r.pop();
        }
        // a half turn (two edges back to back) is a zero-width spike or slit:
        // degenerate, it has no erosion of its own
        let n = r.len();
        let half_turn = (0..n).any(|i| {
            let e = sub(r[i], r[(i + n - 1) % n]);
            let f = sub(r[(i + 1) % n], r[i]);
            cross(e, f) == 0.0 && dot(e, f) < 0.0
        });
        if r.len() < 3 || area2(&r) == 0.0 || half_turn {
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
        return Err(OffsetError::NoContours);
    };
    if area2(&rings[outer]) < 0.0 {
        for r in &mut rings {
            r.reverse();
        }
    }
    // winding against nesting: inside an even number of others, inside on the
    // left (counter-clockwise); inside an odd number, a hole (clockwise)
    for (ci, r) in rings.iter().enumerate() {
        let depth = rings
            .iter()
            .enumerate()
            .filter(|(cj, other)| *cj != ci && point_in_ring(r[0], other))
            .count();
        if (area2(r) > 0.0) != (depth % 2 == 0) {
            return Err(OffsetError::WindingMismatch { contour: ci });
        }
    }
    let mut out = Vec::with_capacity(rings.len());
    for (ci, r) in rings.iter().enumerate() {
        let ring = erode_ring(r, ci, delta, cfg)?;
        // every eroded point is `delta` from the boundary, and the contour
        // keeps its orientation: otherwise the drops left a contour that is
        // not the erosion (it eroded away and folded over)
        let pts: Vec<[f64; 2]> = ring.iter().map(|v| v.p).collect();
        // `delta` less f64 rounding of the shifted points (relative 1e-9,
        // far above the ~1e-16 of the arithmetic, far below any real shortfall)
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
    let edge_line = |i: usize| -> Line {
        let d = sub(r[(i + 1) % n], r[i]);
        let u = unit(d);
        let nrm = [-u[1], u[0]];
        Line {
            p: [r[i][0] + delta * nrm[0], r[i][1] + delta * nrm[1]],
            d,
            n: nrm,
            seq: 0,
            last_seq: 0,
            end_reflex: None,
        }
    };
    // the element sequence: for each vertex, the tangents of its arc (if it
    // is reflex), then the edge leaving it. An arc is the polygon of its
    // tangents, outside the circle by at most the sagitta, so every junction is
    // a line intersection and an arc that the erosion cuts short is trimmed by
    // the same drops as the edges.
    let mut elems: Vec<Line> = Vec::new();
    let mut first_reflex: Option<[f64; 2]> = None;
    for (i, &v) in r.iter().enumerate() {
        let (prev, cur) = (edge_line((i + n - 1) % n), edge_line(i));
        if cross(prev.d, cur.d) < 0.0 {
            match elems.last_mut() {
                Some(last) => last.end_reflex = Some(v),
                None => first_reflex = Some(v),
            }
            for nrm in tangent_normals(prev.n, cur.n, delta, cfg.sagitta) {
                elems.push(Line {
                    p: [v[0] + delta * nrm[0], v[1] + delta * nrm[1]],
                    d: [nrm[1], -nrm[0]],
                    n: nrm,
                    seq: 0,
                    last_seq: 0,
                    end_reflex: Some(v),
                });
            }
        }
        elems.push(cur);
    }
    if let (Some(c), Some(last)) = (first_reflex, elems.last_mut()) {
        last.end_reflex = Some(c);
    }
    let total = elems.len();
    for (k, e) in elems.iter_mut().enumerate() {
        e.seq = k;
        e.last_seq = k;
    }
    let adjacent = |a: &Line, b: &Line| (a.last_seq + 1) % total == b.seq;
    let junction = |a: &Line, b: &Line| -> Result<[f64; 2], OffsetError> {
        match intersect(a, b) {
            Some(p) => Ok(p),
            None if adjacent(a, b) && collinear(a, b) => Ok(b.p),
            None => Err(parallel_error(a, b, ci)),
        }
    };
    // drop reversed lines until none is
    let mut active: Vec<Line> = elems;
    loop {
        if active.len() < 3 {
            return Err(OffsetError::Vanished { contour: ci });
        }
        let m = active.len();
        // neighbours left on one line by drops are one line again
        if let Some(k) = (0..m).find(|&k| {
            let (a, b) = (&active[k], &active[(k + 1) % m]);
            !adjacent(a, b) && collinear(a, b)
        }) {
            let next = (k + 1) % m;
            active[k].last_seq = active[next].last_seq;
            active[k].end_reflex = active[next].end_reflex;
            active.remove(next);
            continue;
        }
        let mut reversed: Option<usize> = None;
        for k in 0..m {
            let (prev, cur, next) = (&active[(k + m - 1) % m], &active[k], &active[(k + 1) % m]);
            let s = junction(prev, cur)?;
            let e = junction(cur, next)?;
            // reversed, or shrunk to a point (its two junctions meet)
            if dot(sub(e, s), cur.d) <= 0.0 {
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
    let m = active.len();
    let mut verts: Vec<OffsetVertex> = Vec::with_capacity(m);
    for k in 0..m {
        let (prev, cur) = (&active[(k + m - 1) % m], &active[k]);
        let p = junction(prev, cur)?;
        let normal = match (adjacent(prev, cur), prev.end_reflex) {
            (true, Some(c)) => OffsetNormal::Arc {
                centre: c,
                normal: unit(sub(c, p)),
            },
            (true, None) if cross(prev.d, cur.d) == 0.0 => {
                OffsetNormal::Edge([-cur.n[0], -cur.n[1]])
            }
            _ => OffsetNormal::Corner([-prev.n[0], -prev.n[1]], [-cur.n[0], -cur.n[1]]),
        };
        verts.push(OffsetVertex { p, normal });
    }
    Ok(verts)
}

/// The inward normals of the tangents that stand for the arc from `a` to `b`
/// (turning clockwise through less than half a turn), ends excluded: at equal
/// angles, fine enough that two neighbouring tangents meet within `sagitta` of
/// the circle (a junction over angle `t` sits `delta (1/cos(t/2) - 1)`
/// outside it). None when the two edges' own lines already meet within it.
fn tangent_normals(a: [f64; 2], b: [f64; 2], delta: f64, sagitta: f64) -> Vec<[f64; 2]> {
    let phi = acos64(crate::det_math::raw_clamp64(dot(a, b), -1.0, 1.0));
    let max_t = 2.0 * acos64(delta / (delta + sagitta));
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
    out[1..out.len() - 1].to_vec()
}

/// Whether two shifted lines are the same line, running the same way.
fn collinear(a: &Line, b: &Line) -> bool {
    let scale = 1.0 + dot(a.p, a.p) + dot(b.p, b.p);
    let la = sqrt64(dot(a.d, a.d));
    cross(a.d, b.d) == 0.0
        && dot(a.d, b.d) > 0.0
        && (cross(a.d, sub(b.p, a.p)) / la).abs() <= 1.0e-12 * scale
}

fn parallel_error(a: &Line, b: &Line, ci: usize) -> OffsetError {
    if dot(a.d, b.d) < 0.0 {
        OffsetError::Vanished { contour: ci }
    } else {
        OffsetError::Parallel { contour: ci }
    }
}

/// Point-in-polygon by crossing parity.
fn point_in_ring(p: [f64; 2], r: &[[f64; 2]]) -> bool {
    let n = r.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (r[i], r[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        {
            inside = !inside;
        }
    }
    inside
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
            best = crate::det_math::min64(best, sqrt64(dot(sub(p, q), sub(p, q))));
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
