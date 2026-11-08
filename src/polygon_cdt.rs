//! Quality triangulation of a polygonal domain with holes: a conforming
//! Delaunay triangulation refined to a minimum angle (Ruppert, *A Delaunay
//! Refinement Algorithm for Quality 2-Dimensional Mesh Generation*, 1995;
//! Shewchuk, *Delaunay Refinement Algorithms for Triangular Mesh Generation*,
//! 2002).
//!
//! # Exactness and determinism
//!
//! Every point lives on an integer grid of spacing [`CdtConfig::grid`]. The two
//! predicates that decide topology, orientation and in-circle, are evaluated in
//! `i128` on those integers, so they are exact: no tolerance, no platform
//! dependence, the same triangulation on every target. Quality decisions
//! (which triangle is too skinny, where its circumcentre is) use `f64`
//! arithmetic on the same integers, which is deterministic too (IEEE add,
//! multiply, divide, square root; no fused multiply-add, no transcendental).
//! A point the refinement creates (a segment midpoint, a circumcentre) is
//! rounded to the grid, so a split boundary segment can sit up to half a grid
//! step off the input line; with the default 1 µm grid that is 0.5 µm.
//!
//! # Domain
//!
//! The input is a set of closed contours. The contour with the largest
//! absolute area sets the outside's winding; contours of the opposite winding
//! are holes. A triangle belongs to the domain when its centroid is inside an
//! odd number of contours, which is the same rule for nested simple contours.
//! Contours must be simple and must not cross each other; crossing input is
//! not detected.
//!
//! # Termination
//!
//! Ruppert's algorithm terminates for minimum angles up to about 20.7° when no
//! two input segments meet at less than 60°. Glyph outlines do have sharper
//! corners, so a triangle whose shortest edge joins two input segments that
//! meet at less than 60° is not refined (it cannot be, without endless
//! splitting near that corner), and [`CdtConfig::max_points`] bounds the work;
//! running out of it is an error, not a silently worse mesh.

use std::collections::{BTreeMap, BTreeSet};

/// Parameters of [`triangulate`].
#[derive(Debug, Clone, Copy)]
pub struct CdtConfig {
    /// Grid spacing every point is rounded to (input units).
    pub grid: f64,
    /// Minimum angle to refine to, in degrees; at most 20.7.
    pub min_angle_deg: f64,
    /// Longest edge allowed, in input units (`f64::INFINITY` for none).
    pub max_edge: f64,
    /// Upper bound on the number of points, input included.
    pub max_points: usize,
}

impl Default for CdtConfig {
    fn default() -> Self {
        Self {
            grid: 1.0e-6,
            min_angle_deg: 20.0,
            max_edge: f64::INFINITY,
            max_points: 200_000,
        }
    }
}

/// Why a triangulation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CdtError {
    /// A contour has fewer than three distinct grid points.
    DegenerateContour {
        /// Index of the contour.
        contour: usize,
    },
    /// No contour encloses any area.
    NoArea,
    /// A parameter is out of range (grid not positive and finite, angle not in
    /// `(0, 20.7]`, edge bound not positive).
    BadConfig,
    /// An input coordinate does not fit the grid's integer range.
    OutOfRange,
    /// The point budget ran out before the mesh met the quality bounds.
    Budget {
        /// Points at the time.
        points: usize,
    },
}

/// A triangulation of the domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Cdt {
    /// The grid every point is on.
    pub grid: f64,
    /// Points, in grid units.
    pub points: Vec<[i64; 2]>,
    /// Triangles of the domain, counter-clockwise, by point index.
    pub triangles: Vec<[u32; 3]>,
    /// The boundary subsegments (input segments after splitting).
    pub segments: Vec<[u32; 2]>,
}

impl Cdt {
    /// A point in input units.
    #[must_use]
    pub fn point(&self, i: u32) -> [f64; 2] {
        let p = self.points[i as usize];
        [p[0] as f64 * self.grid, p[1] as f64 * self.grid]
    }

    /// Twice the area of the domain in grid units, exact.
    #[must_use]
    pub fn area2_grid(&self) -> i128 {
        self.triangles
            .iter()
            .map(|t| {
                orient(
                    self.points[t[0] as usize],
                    self.points[t[1] as usize],
                    self.points[t[2] as usize],
                )
            })
            .sum()
    }
}

/// Twice the signed area of `abc`, exact.
#[must_use]
pub fn orient(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i128 {
    let (ax, ay) = (i128::from(b[0] - a[0]), i128::from(b[1] - a[1]));
    let (bx, by) = (i128::from(c[0] - a[0]), i128::from(c[1] - a[1]));
    ax * by - ay * bx
}

/// Positive when `d` is strictly inside the circle through the
/// counter-clockwise triangle `abc`, exact.
#[must_use]
pub fn incircle(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> i128 {
    let r = |p: [i64; 2]| (i128::from(p[0] - d[0]), i128::from(p[1] - d[1]));
    let (ax, ay) = r(a);
    let (bx, by) = r(b);
    let (cx, cy) = r(c);
    let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
    ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx)
}

/// Largest grid coordinate accepted: the in-circle determinant of points a
/// few times this far apart stays inside `i128`.
const MAX_COORD: i64 = 1 << 28;

/// Twice the signed area of a contour, exact.
fn contour_area2(c: &[[i64; 2]]) -> i128 {
    let n = c.len();
    (0..n)
        .map(|i| {
            let (p, q) = (c[i], c[(i + 1) % n]);
            i128::from(p[0]) * i128::from(q[1]) - i128::from(q[0]) * i128::from(p[1])
        })
        .sum()
}

/// Triangulate the domain bounded by `contours` (input units).
///
/// # Errors
///
/// See [`CdtError`].
pub fn triangulate(contours: &[Vec<[f64; 2]>], cfg: &CdtConfig) -> Result<Cdt, CdtError> {
    if !(cfg.grid > 0.0 && cfg.grid.is_finite())
        || !(cfg.min_angle_deg > 0.0 && cfg.min_angle_deg <= 20.7)
        || cfg.max_edge.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return Err(CdtError::BadConfig);
    }
    // quantise, drop repeated points
    let mut rings: Vec<Vec<[i64; 2]>> = Vec::new();
    for (ci, c) in contours.iter().enumerate() {
        let mut ring: Vec<[i64; 2]> = Vec::new();
        for p in c {
            let q = [p[0] / cfg.grid, p[1] / cfg.grid];
            if !(q[0].abs() < MAX_COORD as f64 && q[1].abs() < MAX_COORD as f64) {
                return Err(CdtError::OutOfRange);
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "bounded by MAX_COORD just above"
            )]
            let g = [q[0].round() as i64, q[1].round() as i64];
            if ring.last() != Some(&g) {
                ring.push(g);
            }
        }
        while ring.len() > 1 && ring.first() == ring.last() {
            ring.pop();
        }
        if ring.len() < 3 || contour_area2(&ring) == 0 {
            return Err(CdtError::DegenerateContour { contour: ci });
        }
        rings.push(ring);
    }
    if rings.is_empty() {
        return Err(CdtError::NoArea);
    }
    let mut m = Mesh::new(cfg);
    let mut segs: BTreeSet<(u32, u32)> = BTreeSet::new();
    for ring in &rings {
        let ids: Vec<u32> = ring.iter().map(|&p| m.insert(p)).collect();
        for i in 0..ids.len() {
            let (a, b) = (ids[i], ids[(i + 1) % ids.len()]);
            if a != b {
                segs.insert((a.min(b), a.max(b)));
            }
        }
    }
    m.segs = segs;
    m.rings = rings.clone();
    m.recover_segments()?;
    m.refine()?;
    let triangles = m.domain_triangles(&rings);
    if triangles.is_empty() {
        return Err(CdtError::NoArea);
    }
    let mut segments = m.segs_out();
    segments.sort_unstable();
    Ok(Cdt {
        grid: cfg.grid,
        points: m.pts[3..].to_vec(),
        triangles: triangles.into_iter().map(|t| t.map(|v| v - 3)).collect(),
        segments,
    })
}

/// The working triangulation: points 0..3 are a super-triangle.
struct Mesh {
    cfg: CdtConfig,
    pts: Vec<[i64; 2]>,
    tris: Vec<Option<[u32; 3]>>,
    segs: BTreeSet<(u32, u32)>,
    /// The input contours, for telling the domain's triangles from the rest.
    rings: Vec<Vec<[i64; 2]>>,
}

impl Mesh {
    fn new(cfg: &CdtConfig) -> Self {
        let s = MAX_COORD * 4;
        let pts = vec![[-s, -s], [s, -s], [0, s]];
        Self {
            cfg: *cfg,
            pts,
            tris: vec![Some([0, 1, 2])],
            segs: BTreeSet::new(),
            rings: Vec::new(),
        }
    }

    fn segs_out(&self) -> Vec<[u32; 2]> {
        self.segs.iter().map(|&(a, b)| [a - 3, b - 3]).collect()
    }

    /// Bowyer-Watson insertion; returns the index of the point (an existing
    /// one if the grid point is already there).
    fn insert(&mut self, p: [i64; 2]) -> u32 {
        if let Some(i) = self.pts.iter().position(|&q| q == p) {
            return i as u32;
        }
        let v = self.pts.len() as u32;
        self.pts.push(p);
        let mut cavity: Vec<usize> = Vec::new();
        for (i, t) in self.tris.iter().enumerate() {
            if let Some(t) = t {
                let (a, b, c) = (
                    self.pts[t[0] as usize],
                    self.pts[t[1] as usize],
                    self.pts[t[2] as usize],
                );
                if incircle(a, b, c, p) > 0 {
                    cavity.push(i);
                }
            }
        }
        if cavity.is_empty() {
            // p on a circumcircle of every candidate: it lies on an edge or
            // cocircular; take the triangles whose closure holds it
            for (i, t) in self.tris.iter().enumerate() {
                if let Some(t) = t {
                    let (a, b, c) = (
                        self.pts[t[0] as usize],
                        self.pts[t[1] as usize],
                        self.pts[t[2] as usize],
                    );
                    if orient(a, b, p) >= 0 && orient(b, c, p) >= 0 && orient(c, a, p) >= 0 {
                        cavity.push(i);
                    }
                }
            }
        }
        let mut count: BTreeMap<(u32, u32), (u32, u32, usize)> = BTreeMap::new();
        for &i in &cavity {
            let t = self.tris[i].expect("live");
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = (a.min(b), a.max(b));
                let e = count.entry(key).or_insert((a, b, 0));
                e.2 += 1;
            }
            self.tris[i] = None;
        }
        for (_, (a, b, n)) in count {
            if n == 1 && orient(self.pts[a as usize], self.pts[b as usize], p) > 0 {
                self.tris.push(Some([a, b, v]));
            }
        }
        // compact occasionally
        if self.tris.len() > 4 * self.pts.len() + 64 {
            self.tris.retain(Option::is_some);
        }
        v
    }

    fn edges(&self) -> BTreeSet<(u32, u32)> {
        let mut e = BTreeSet::new();
        for t in self.tris.iter().flatten() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                e.insert((a.min(b), a.max(b)));
            }
        }
        e
    }

    fn budget(&self) -> Result<(), CdtError> {
        if self.pts.len() - 3 > self.cfg.max_points {
            Err(CdtError::Budget {
                points: self.pts.len() - 3,
            })
        } else {
            Ok(())
        }
    }

    /// Split segment `(a, b)` at its grid midpoint. Returns `false` when the
    /// segment is too short to split.
    fn split(&mut self, a: u32, b: u32) -> Result<bool, CdtError> {
        let (pa, pb) = (self.pts[a as usize], self.pts[b as usize]);
        let mid = [(pa[0] + pb[0]).div_euclid(2), (pa[1] + pb[1]).div_euclid(2)];
        if mid == pa || mid == pb {
            return Ok(false);
        }
        self.segs.remove(&(a.min(b), a.max(b)));
        let m = self.insert(mid);
        self.segs.insert((a.min(m), a.max(m)));
        self.segs.insert((m.min(b), m.max(b)));
        self.budget()?;
        Ok(true)
    }

    /// Conforming recovery: split every segment that is not an edge.
    fn recover_segments(&mut self) -> Result<(), CdtError> {
        loop {
            let edges = self.edges();
            let missing: Vec<(u32, u32)> = self
                .segs
                .iter()
                .copied()
                .filter(|s| !edges.contains(s))
                .collect();
            if missing.is_empty() {
                return Ok(());
            }
            let mut progress = false;
            for (a, b) in missing {
                if self.segs.contains(&(a, b)) && !self.edges().contains(&(a, b)) {
                    progress |= self.split(a, b)?;
                }
            }
            if !progress {
                return Err(CdtError::Budget {
                    points: self.pts.len() - 3,
                });
            }
        }
    }

    /// Whether `p` is strictly inside the diametral circle of segment `(a, b)`.
    fn encroaches(&self, a: u32, b: u32, p: [i64; 2]) -> bool {
        let (pa, pb) = (self.pts[a as usize], self.pts[b as usize]);
        let d = |u: [i64; 2], w: [i64; 2]| (i128::from(u[0] - w[0]), i128::from(u[1] - w[1]));
        let (x1, y1) = d(pa, p);
        let (x2, y2) = d(pb, p);
        x1 * x2 + y1 * y2 < 0
    }

    /// The input segments meeting at vertex `v`, other ends.
    fn seg_neighbours(&self, v: u32) -> Vec<u32> {
        self.segs
            .iter()
            .filter_map(|&(a, b)| {
                if a == v {
                    Some(b)
                } else if b == v {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    fn refine(&mut self) -> Result<(), CdtError> {
        let sin_min = sin_deg(self.cfg.min_angle_deg);
        // circumradius / shortest edge above this means an angle below the minimum
        let ratio_max = 1.0 / (2.0 * sin_min);
        let max_edge_g = self.cfg.max_edge / self.cfg.grid;
        loop {
            // 1. encroached segments
            let mut split_any = false;
            let segs: Vec<(u32, u32)> = self.segs.iter().copied().collect();
            for (a, b) in segs {
                if !self.segs.contains(&(a, b)) {
                    continue;
                }
                let encroached = self.tris.iter().flatten().any(|t| {
                    t.contains(&a) && t.contains(&b) && {
                        let o = t
                            .iter()
                            .copied()
                            .find(|&v| v != a && v != b)
                            .expect("third");
                        o >= 3 && self.encroaches(a, b, self.pts[o as usize])
                    }
                });
                if encroached && self.split(a, b)? {
                    split_any = true;
                }
            }
            if split_any {
                continue;
            }
            // 2. the worst bad triangle inside the hull of the input
            let mut worst: Option<(f64, [u32; 3])> = None;
            for t in self.tris.iter().flatten() {
                if t.iter().any(|&v| v < 3) {
                    continue;
                }
                let p = t.map(|v| self.pts[v as usize]);
                // only the domain is refined: a triangle in a hole or a notch
                // of the outline can be as skinny as it likes
                let c3 = [p[0][0] + p[1][0] + p[2][0], p[0][1] + p[1][1] + p[2][1]];
                if self.rings.iter().filter(|r| point_in_ring3(c3, r)).count() % 2 == 0 {
                    continue;
                }
                let l = [dist2(p[0], p[1]), dist2(p[1], p[2]), dist2(p[2], p[0])];
                let lmin = l[0].min(l[1]).min(l[2]);
                let lmax = l[0].max(l[1]).max(l[2]);
                let area2 = orient(p[0], p[1], p[2]) as f64;
                if area2 <= 0.0 {
                    continue;
                }
                // R = abc / (2 * area2) ; R / lmin
                let r = (l[0] * l[1] * l[2]).sqrt() / (2.0 * area2);
                let ratio = r / lmin.sqrt();
                let too_long = lmax.sqrt() > max_edge_g;
                if (ratio > ratio_max && !self.small_input_angle(t, &l)) || too_long {
                    let score = if too_long { f64::INFINITY } else { ratio };
                    if worst
                        .as_ref()
                        .is_none_or(|(s, w)| score > *s || (score == *s && t < w))
                    {
                        worst = Some((score, *t));
                    }
                }
            }
            let Some((_, t)) = worst else {
                return Ok(());
            };
            let c = circumcentre(t.map(|v| self.pts[v as usize]));
            let segs: Vec<(u32, u32)> = self.segs.iter().copied().collect();
            let hit: Vec<(u32, u32)> = segs
                .into_iter()
                .filter(|&(a, b)| self.encroaches(a, b, c))
                .collect();
            if hit.is_empty() {
                if self.pts.contains(&c) {
                    return Err(CdtError::Budget {
                        points: self.pts.len() - 3,
                    });
                }
                self.insert(c);
                self.budget()?;
            } else {
                let mut any = false;
                for (a, b) in hit {
                    any |= self.split(a, b)?;
                }
                if !any {
                    return Err(CdtError::Budget {
                        points: self.pts.len() - 3,
                    });
                }
            }
            self.recover_segments()?;
        }
    }

    /// The triangle's shortest edge joins two input segments meeting at less
    /// than 60° at a shared vertex: such a corner cannot be refined away.
    fn small_input_angle(&self, t: &[u32; 3], l: &[f64; 3]) -> bool {
        let k = if l[0] <= l[1] && l[0] <= l[2] {
            0
        } else if l[1] <= l[2] {
            1
        } else {
            2
        };
        let (a, b) = (t[k], t[(k + 1) % 3]);
        let apex = t[(k + 2) % 3];
        // the corner is at the apex when both its edges to a and b lie on segments
        let n = self.seg_neighbours(apex);
        if n.contains(&a) && n.contains(&b) {
            let pa = self.pts[a as usize];
            let pb = self.pts[b as usize];
            let pc = self.pts[apex as usize];
            let u = [(pa[0] - pc[0]) as f64, (pa[1] - pc[1]) as f64];
            let w = [(pb[0] - pc[0]) as f64, (pb[1] - pc[1]) as f64];
            let cos = (u[0] * w[0] + u[1] * w[1])
                / ((u[0] * u[0] + u[1] * u[1]).sqrt() * (w[0] * w[0] + w[1] * w[1]).sqrt());
            // cos 60° = 0.5
            return cos > 0.5;
        }
        false
    }

    /// Triangles inside the domain: centroid inside an odd number of rings.
    fn domain_triangles(&self, rings: &[Vec<[i64; 2]>]) -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        for t in self.tris.iter().flatten() {
            if t.iter().any(|&v| v < 3) {
                continue;
            }
            let p = t.map(|v| self.pts[v as usize]);
            // centroid times 3, compared against rings scaled by 3: exact
            let c3 = [p[0][0] + p[1][0] + p[2][0], p[0][1] + p[1][1] + p[2][1]];
            let inside = rings.iter().filter(|r| point_in_ring3(c3, r)).count() % 2 == 1;
            if inside {
                out.push(*t);
            }
        }
        out.sort_unstable();
        out
    }
}

fn dist2(a: [i64; 2], b: [i64; 2]) -> f64 {
    let (x, y) = ((a[0] - b[0]) as f64, (a[1] - b[1]) as f64);
    x * x + y * y
}

/// Circumcentre rounded to the grid.
fn circumcentre(p: [[i64; 2]; 3]) -> [i64; 2] {
    let (ax, ay) = (p[0][0] as f64, p[0][1] as f64);
    let (bx, by) = ((p[1][0] - p[0][0]) as f64, (p[1][1] - p[0][1]) as f64);
    let (cx, cy) = ((p[2][0] - p[0][0]) as f64, (p[2][1] - p[0][1]) as f64);
    let d = 2.0 * (bx * cy - by * cx);
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (cy * b2 - by * c2) / d;
    let uy = (bx * c2 - cx * b2) / d;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "inside the super-triangle, far below i64 range"
    )]
    let r = [(ax + ux).round() as i64, (ay + uy).round() as i64];
    r
}

/// Point-in-ring by crossing parity, with the point given times 3 and the
/// ring scaled by 3 on the fly: exact.
fn point_in_ring3(c3: [i64; 2], ring: &[[i64; 2]]) -> bool {
    let n = ring.len();
    let mut inside = false;
    for i in 0..n {
        let a = [ring[i][0] * 3, ring[i][1] * 3];
        let b = [ring[(i + 1) % n][0] * 3, ring[(i + 1) % n][1] * 3];
        if (a[1] > c3[1]) != (b[1] > c3[1]) {
            // x of the crossing compared with c3.x, without division
            let lhs = i128::from(c3[0] - a[0]) * i128::from(b[1] - a[1]);
            let rhs = i128::from(b[0] - a[0]) * i128::from(c3[1] - a[1]);
            let left = if b[1] > a[1] { lhs < rhs } else { lhs > rhs };
            if left {
                inside = !inside;
            }
        }
    }
    inside
}

/// `sin` of an angle in degrees by its Taylor series (the crate does not use
/// the platform libm; this is only evaluated on the configured minimum angle).
fn sin_deg(deg: f64) -> f64 {
    let x = deg * std::f64::consts::PI / 180.0;
    let mut term = x;
    let mut sum = x;
    for k in 1..12 {
        term *= -x * x / f64::from((2 * k) * (2 * k + 1));
        sum += term;
    }
    sum
}
