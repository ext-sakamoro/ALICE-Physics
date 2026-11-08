//! The area of a polygon (holes allowed) eroded by `d`, measured without the
//! code under test and without the closed form: an exact length per scan row,
//! integrated over the rows.
//!
//! On a row `y`, the eroded set is the polygon's inside minus every point
//! within `d` of a boundary segment. Each segment's capsule (the points within
//! `d` of it) meets the row in one interval, computed in closed form as the
//! union of its two end discs and its side strip. So the eroded length of the
//! row is exact up to f64: the inside intervals (crossing parity) minus the
//! union of the capsule intervals.
//!
//! The rows are integrated by the midpoint rule, piece by piece between the
//! heights where the length can jump or kink (vertex heights, and those plus
//! and minus `d`), so inside a piece it is smooth and the rule's error falls as
//! `h²`; the area is taken at `n` and `2n` rows and the difference of the two is
//! returned as the error bound of the finer one (three times the Richardson
//! estimate `|A_h - A_{h/2}| / 3`).
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: in f64, from the polygon only"
)]
#![allow(dead_code, reason = "shared by several test binaries, each uses part")]

fn row_inside(c: &[Vec<[f64; 2]>], y: f64) -> Vec<(f64, f64)> {
    let mut xs: Vec<f64> = Vec::new();
    for r in c {
        for i in 0..r.len() {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            if (a[1] > y) != (b[1] > y) {
                xs.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    xs.chunks(2)
        .filter(|p| p.len() == 2)
        .map(|p| (p[0], p[1]))
        .collect()
}

/// The interval of `x` on row `y` within `d` of segment `ab`, if any.
fn capsule(a: [f64; 2], b: [f64; 2], d: f64, y: f64) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for p in [a, b] {
        let dy = y - p[1];
        if dy.abs() <= d {
            let w = (d * d - dy * dy).sqrt();
            lo = lo.min(p[0] - w);
            hi = hi.max(p[0] + w);
        }
    }
    // side strip: 0 <= u.(p - a) <= len and |n.(p - a)| <= d, both linear in x
    let e = [b[0] - a[0], b[1] - a[1]];
    let len = (e[0] * e[0] + e[1] * e[1]).sqrt();
    let u = [e[0] / len, e[1] / len];
    let n = [-u[1], u[0]];
    let mut s_lo = f64::NEG_INFINITY;
    let mut s_hi = f64::INFINITY;
    for (k, l, h) in [(u, 0.0, len), (n, -d, d)] {
        // k.x * x + k.y * (y - a.y) - k.x * a.x in [l, h]
        let c = k[1] * (y - a[1]) - k[0] * a[0];
        if k[0] == 0.0 {
            if c < l || c > h {
                return if lo <= hi { Some((lo, hi)) } else { None };
            }
        } else {
            let (x1, x2) = ((l - c) / k[0], (h - c) / k[0]);
            s_lo = s_lo.max(x1.min(x2));
            s_hi = s_hi.min(x1.max(x2));
        }
    }
    if s_lo <= s_hi {
        lo = lo.min(s_lo);
        hi = hi.max(s_hi);
    }
    if lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

fn row_length(c: &[Vec<[f64; 2]>], d: f64, y: f64) -> f64 {
    let inside = row_inside(c, y);
    let mut caps: Vec<(f64, f64)> = Vec::new();
    if d > 0.0 {
        for r in c {
            for i in 0..r.len() {
                if let Some(iv) = capsule(r[i], r[(i + 1) % r.len()], d, y) {
                    caps.push(iv);
                }
            }
        }
    }
    caps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (l, h) in caps {
        match merged.last_mut() {
            Some(last) if l <= last.1 => last.1 = last.1.max(h),
            _ => merged.push((l, h)),
        }
    }
    let mut total = 0.0;
    for (a, b) in inside {
        let mut len = b - a;
        for &(l, h) in &merged {
            let (l, h) = (l.max(a), h.min(b));
            if l < h {
                len -= h - l;
            }
        }
        total += len;
    }
    total
}

/// Midpoint rule between the heights where a row's length can jump or kink:
/// every vertex height, and every vertex height plus and minus `d` (where an
/// end disc starts or ends). Inside each piece the length is smooth.
fn integrate(c: &[Vec<[f64; 2]>], d: f64, n: usize) -> f64 {
    let mut ys: Vec<f64> = Vec::new();
    for r in c {
        for p in r {
            ys.push(p[1]);
            if d > 0.0 {
                ys.push(p[1] - d);
                ys.push(p[1] + d);
            }
        }
    }
    ys.sort_by(f64::total_cmp);
    ys.dedup();
    let span = ys[ys.len() - 1] - ys[0];
    let step = span / n as f64;
    let mut area = 0.0;
    for w in ys.windows(2) {
        let (y0, y1) = (w[0], w[1]);
        if y1 <= y0 {
            continue;
        }
        let k = ((y1 - y0) / step).ceil().max(2.0) as usize;
        let h = (y1 - y0) / k as f64;
        area += (0..k)
            .map(|i| row_length(c, d, y0 + (i as f64 + 0.5) * h))
            .sum::<f64>()
            * h;
    }
    area
}

/// `(area, error bound)` of the polygon eroded by `d`, at `n` and `2n` rows.
pub fn eroded_area(c: &[Vec<[f64; 2]>], d: f64, n: usize) -> (f64, f64) {
    let coarse = integrate(c, d, n);
    let fine = integrate(c, d, 2 * n);
    (fine, (fine - coarse).abs() + 1e-15)
}
