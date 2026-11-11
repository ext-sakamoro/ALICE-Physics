//! The position-based (XPBD) contact as a point constraint on rigid bodies.
//!
//! # MODEL
//!
//! Müller, Macklin, Chentanez, Jeschke, Kim, "Detailed Rigid Body Simulation
//! with Extended Position Based Dynamics", SCA 2020, §3.5 (contacts and static
//! friction) and §3.6 (velocity level).
//!
//! - **Anchors.** A contact is detected once per substep with the point `p_a`
//!   on body A, `p_b` on B, normal `n` (from B towards A, so moving A along `n`
//!   relieves it) and depth `d₀ = n · (p_b − p_a)`. Its arms are taken at the
//!   poses of detection, `r = p − x` (`x` the centre of mass).
//! - **Current depth.** Each body has turned by `Δθ` since detection (the sum
//!   of the rotation vectors the contact corrections of this substep gave it);
//!   its arm is `r + Δθ × r` (first order in the turn of one substep) and its
//!   point `x + r + Δθ × r`. The depth now is
//!   `d = d₀ − n · ((p_a' − p_a) − (p_b' − p_b))`, so a correction that
//!   another contact (or a joint) made to the same body is seen. The turn is
//!   built from the corrections, which do not depend on the orientation the body
//!   had, so an isotropic body's motion does not either.
//! - **Normal correction** (eq. 2–5 with zero compliance): for `d > 0`,
//!   `Δλ = d / w` with the generalised inverse mass
//!   `w = m_a⁻¹ + m_b⁻¹ + (r_a × n)ᵀ I_a⁻¹ (r_a × n) + (r_b × n)ᵀ I_b⁻¹ (r_b × n)`;
//!   A moves by `+n Δλ / m_a` and turns by `+I_a⁻¹ (r_a × n) Δλ`, B by the
//!   negatives. The contact point moves apart by exactly `d`; the separation
//!   applied (`d` summed over iterations) is the contact's `cached_lambda`.
//! - **Static friction** (eq. 26–27): the relative displacement of the contact
//!   points over the substep, `Δp = Δp_a − Δp_b` with
//!   `Δp_body = (x − x_prev) + (ω h + Δθ) × r` (`ω` the substep's predicted
//!   angular velocity, which the position solve does not change), has a
//!   tangential part `Δp_t`; while `|Δp_t| < μ λ_n` it is cancelled the same
//!   way with `w` along `Δp_t / |Δp_t|`.
//! - **Velocity level** (eq. 29–34): restitution and dynamic friction act on the
//!   relative velocity of the contact points `v + ω × r`, through the same `w`
//!   (an impulse `j` changes the point's relative velocity by `w j`).
//!
//! Units: metres, seconds, kilograms (as the rest of the solver). Conserved: the
//! corrections are equal and opposite at the contact point (linear and angular
//! momentum of the pair about it). Valid for rigid bodies with diagonal inertia
//! in the body frame; the turn is first order in one substep's rotation.
//!
//! Verified by `tests/analytic_static_contact_response.rs`
//! (`a_sliding_ball_rolls_at_five_sevenths_of_v0_under_xpbd`,
//! `a_domino_falls_over_only_past_atan_w_over_h_under_xpbd`) and
//! `tests/analytic_resting_box_manifold.rs`.

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::{ContactConstraint, RigidBody};

/// The arms of a contact at the poses it was detected at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContactAnchor {
    /// `p_a − x_a` at detection.
    pub(crate) r_a: Vec3Fix,
    /// `p_b − x_b` at detection.
    pub(crate) r_b: Vec3Fix,
}

impl ContactAnchor {
    /// The anchor of `c` between bodies `a` and `b` at their present positions.
    pub(crate) fn at(c: &ContactConstraint, a: &RigidBody, b: &RigidBody) -> Self {
        Self {
            r_a: c.contact.point_a - a.position,
            r_b: c.contact.point_b - b.position,
        }
    }
}

/// What the substep's contact corrections have turned a body by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ContactTurn {
    /// The sum of the rotation vectors applied (world frame).
    pub(crate) turn: Vec3Fix,
    /// The body's rotation before the first of them (`None`: none applied).
    pub(crate) start: Option<QuatFix>,
    /// The body's rotation right after the last of them.
    pub(crate) end: Option<QuatFix>,
}

impl ContactTurn {
    /// The angular velocity of a body whose rotation went from `predicted` to
    /// `rotation` over the substep `1 / inv_dt`, when nothing but the contact
    /// corrections turned it after the prediction: `omega + turn / h`, built from
    /// the corrections themselves (no quaternion difference, whose rounding
    /// would depend on the orientation the body had). `None` otherwise.
    pub(crate) fn angular_velocity(
        &self,
        predicted: QuatFix,
        rotation: QuatFix,
        omega: Vec3Fix,
        inv_dt: Fix128,
    ) -> Option<Vec3Fix> {
        (self.start == Some(predicted) && self.end == Some(rotation))
            .then(|| omega + self.turn * inv_dt)
    }
}

/// A body's share of a correction: a displacement and a rotation vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BodyDelta {
    /// Added to the position.
    pub(crate) dx: Vec3Fix,
    /// The rotation (axis × angle, world frame) applied to the orientation and
    /// added to the body's turn.
    pub(crate) dtheta: Vec3Fix,
}

impl BodyDelta {
    const ZERO: Self = Self {
        dx: Vec3Fix::ZERO,
        dtheta: Vec3Fix::ZERO,
    };
}

/// A correction of one contact: what it gives each body and the separation it
/// applied along the normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PointCorrection {
    /// Body A's share.
    pub(crate) a: BodyDelta,
    /// Body B's share.
    pub(crate) b: BodyDelta,
    /// The separation applied along the normal (0 for friction).
    pub(crate) separation: Fix128,
}

/// The arm `r` turned by `turn` (first order): `r + turn × r`.
#[inline]
pub(crate) fn arm(r: Vec3Fix, turn: Vec3Fix) -> Vec3Fix {
    r + turn.cross(r)
}

/// `(r × axis)ᵀ I⁻¹ (r × axis)` of `body` (zero for a body without inverse mass).
#[inline]
pub(crate) fn angular_w(body: &RigidBody, r: Vec3Fix, axis: Vec3Fix) -> Fix128 {
    if body.inv_mass.is_zero() {
        return Fix128::ZERO;
    }
    let rn = r.cross(axis);
    rn.dot(body.world_inv_inertia_apply(rn))
}

/// The generalised inverse mass of a point correction along `axis` at arms
/// `r_a`, `r_b`.
#[inline]
pub(crate) fn point_w(
    a: &RigidBody,
    b: &RigidBody,
    r_a: Vec3Fix,
    r_b: Vec3Fix,
    axis: Vec3Fix,
) -> Fix128 {
    a.inv_mass + b.inv_mass + angular_w(a, r_a, axis) + angular_w(b, r_b, axis)
}

/// What an impulse-like multiplier `lambda` along `axis` at arm `r` gives `body`
/// (A takes `+lambda`, B `−lambda`).
#[inline]
fn share(body: &RigidBody, r: Vec3Fix, axis: Vec3Fix, lambda: Fix128) -> BodyDelta {
    if body.inv_mass.is_zero() {
        return BodyDelta::ZERO;
    }
    BodyDelta {
        dx: axis * (lambda * body.inv_mass),
        dtheta: body.world_inv_inertia_apply(r.cross(axis)) * lambda,
    }
}

/// The contact's depth at the bodies' present poses (see the module MODEL).
pub(crate) fn current_depth(
    c: &ContactConstraint,
    anchor: ContactAnchor,
    a: &RigidBody,
    b: &RigidBody,
    turn_a: Vec3Fix,
    turn_b: Vec3Fix,
) -> Fix128 {
    let moved_a = a.position + arm(anchor.r_a, turn_a) - c.contact.point_a;
    let moved_b = b.position + arm(anchor.r_b, turn_b) - c.contact.point_b;
    c.contact.depth - c.contact.normal.dot(moved_a - moved_b)
}

/// The normal correction of contact `c` (see the module MODEL), or `None` when
/// it is not penetrating now or neither body can move.
pub(crate) fn normal_correction(
    c: &ContactConstraint,
    anchor: ContactAnchor,
    a: &RigidBody,
    b: &RigidBody,
    turn_a: Vec3Fix,
    turn_b: Vec3Fix,
) -> Option<PointCorrection> {
    let d = current_depth(c, anchor, a, b, turn_a, turn_b);
    if d <= Fix128::ZERO {
        return None;
    }
    let (r_a, r_b) = (arm(anchor.r_a, turn_a), arm(anchor.r_b, turn_b));
    let n = c.contact.normal;
    let w = point_w(a, b, r_a, r_b, n);
    if w <= Fix128::ZERO {
        return None;
    }
    let lambda = d / w;
    Some(PointCorrection {
        a: share(a, r_a, n, lambda),
        b: share(b, r_b, n, -lambda),
        separation: d,
    })
}

/// The static friction correction of contact `c` over a substep of width `h`
/// (see the module MODEL), or `None` when the contact slides (or does not move
/// tangentially, or carries no normal separation).
pub(crate) fn static_friction(
    c: &ContactConstraint,
    anchor: ContactAnchor,
    a: &RigidBody,
    b: &RigidBody,
    turn_a: Vec3Fix,
    turn_b: Vec3Fix,
    h: Fix128,
) -> Option<PointCorrection> {
    let limit = c.friction * c.cached_lambda;
    if limit <= Fix128::ZERO {
        return None;
    }
    let (r_a, r_b) = (arm(anchor.r_a, turn_a), arm(anchor.r_b, turn_b));
    let disp = |body: &RigidBody, r: Vec3Fix, turn: Vec3Fix| {
        if body.inv_mass.is_zero() {
            return Vec3Fix::ZERO;
        }
        (body.position - body.prev_position) + (body.angular_velocity * h + turn).cross(r)
    };
    let n = c.contact.normal;
    let dp = disp(a, r_a, turn_a) - disp(b, r_b, turn_b);
    let dp_t = dp - n * dp.dot(n);
    let len_sq = dp_t.length_squared();
    if len_sq.is_zero() || len_sq >= limit * limit {
        return None;
    }
    let (t, len) = dp_t.normalize_with_length();
    let w = point_w(a, b, r_a, r_b, t);
    if w <= Fix128::ZERO {
        return None;
    }
    let lambda = len / w;
    // A's point moved by `+dp_t` relative to B's: A is moved back along `−t`.
    Some(PointCorrection {
        a: share(a, r_a, t, -lambda),
        b: share(b, r_b, t, lambda),
        separation: Fix128::ZERO,
    })
}

/// Apply a body's share: move it, turn it (exact angle about the rotation
/// vector's axis) and add the rotation to its turn.
pub(crate) fn apply(body: &mut RigidBody, turn: &mut ContactTurn, delta: BodyDelta) {
    if body.inv_mass.is_zero() {
        return;
    }
    body.position = body.position + delta.dx;
    let (axis, angle) = delta.dtheta.normalize_with_length();
    if !angle.is_zero() {
        if turn.start.is_none() {
            turn.start = Some(body.rotation);
        }
        body.rotation = crate::joint::rotate_by_angle(body.rotation, axis, angle);
        turn.turn = turn.turn + delta.dtheta;
        turn.end = Some(body.rotation);
    }
}

/// The relative velocity of the contact points (`v + ω × r` of A minus B's).
#[inline]
pub(crate) fn point_velocity(a: &RigidBody, b: &RigidBody, r_a: Vec3Fix, r_b: Vec3Fix) -> Vec3Fix {
    (a.velocity + a.angular_velocity.cross(r_a)) - (b.velocity + b.angular_velocity.cross(r_b))
}

/// What a velocity impulse `j` along `axis` at arm `r` gives `body`: a change of
/// linear and angular velocity (A takes `+j`, B `−j`).
#[inline]
pub(crate) fn velocity_share(
    body: &RigidBody,
    r: Vec3Fix,
    axis: Vec3Fix,
    j: Fix128,
) -> (Vec3Fix, Vec3Fix) {
    let d = share(body, r, axis, j);
    (d.dx, d.dtheta)
}

/// The most points one block solve takes ([`block_normal_correction`]).
pub(crate) const BLOCK_MAX: usize = 4;

/// The normal corrections of up to [`BLOCK_MAX`] contacts between the same two
/// bodies, solved together (the block solver of Catto, Box2D, generalised from
/// two points to four, at the position level of Müller et al. 2020).
///
/// # MODEL
///
/// With current depths `d_j` and the coupling
/// `A_jk = n_j · n_k (m_a⁻¹ + m_b⁻¹) + (r_aj × n_j) · I_a⁻¹ (r_ak × n_k)
///         + (r_bj × n_j) · I_b⁻¹ (r_bk × n_k)`
/// (how much a unit multiplier at point `k` relieves point `j`), the
/// multipliers `λ ≥ 0` solve the complementarity problem: on an active set `S`,
/// `A_S λ_S = d_S`, and every point outside it is left out of penetration,
/// `d_j − Σ_k A_jk λ_k ≤ 0`. Solving the points one after another instead
/// (Gauss–Seidel) overshoots when they are coupled through the rotation: one
/// iteration over the four corners of a face pushed the centre of mass past the
/// floor and the excess became velocity.
///
/// The active sets are tried in a fixed order (more points first, then by the
/// bit pattern of the set), each solved by Gaussian elimination with partial
/// pivoting (exact comparisons, no tolerance); the first that satisfies both
/// conditions is taken. `None` when none does (the caller falls back to one
/// point at a time).
///
/// Returns each body's share and the separation each point's own multiplier
/// applied (`λ_k A_kk`, the unit of `cached_lambda`).
pub(crate) fn block_normal_correction(
    points: &[(ContactConstraint, ContactAnchor)],
    a: &RigidBody,
    b: &RigidBody,
    turn_a: Vec3Fix,
    turn_b: Vec3Fix,
) -> Option<(BodyDelta, BodyDelta, [Fix128; BLOCK_MAX])> {
    let m = points.len();
    if m == 0 || m > BLOCK_MAX {
        return None;
    }
    let mut d = [Fix128::ZERO; BLOCK_MAX];
    let mut n = [Vec3Fix::ZERO; BLOCK_MAX];
    let mut ra = [Vec3Fix::ZERO; BLOCK_MAX];
    let mut rb = [Vec3Fix::ZERO; BLOCK_MAX];
    for (k, (c, anchor)) in points.iter().enumerate() {
        d[k] = current_depth(c, *anchor, a, b, turn_a, turn_b);
        n[k] = c.contact.normal;
        ra[k] = arm(anchor.r_a, turn_a);
        rb[k] = arm(anchor.r_b, turn_b);
    }
    if d[..m].iter().all(|x| *x <= Fix128::ZERO) {
        return None;
    }
    let big_a = coupling(a, b, &ra, &rb, &n, m);
    let lambda = block_lcp(&big_a, &d, m)?;
    {
        let set: Vec<usize> = (0..m).filter(|&k| lambda[k] > Fix128::ZERO).collect();
        let mut da = BodyDelta::ZERO;
        let mut db = BodyDelta::ZERO;
        let mut sep = [Fix128::ZERO; BLOCK_MAX];
        for &k in &set {
            let sa = share(a, ra[k], n[k], lambda[k]);
            let sb = share(b, rb[k], n[k], -lambda[k]);
            da = BodyDelta {
                dx: da.dx + sa.dx,
                dtheta: da.dtheta + sa.dtheta,
            };
            db = BodyDelta {
                dx: db.dx + sb.dx,
                dtheta: db.dtheta + sb.dtheta,
            };
            sep[k] = lambda[k] * big_a[k][k];
        }
        Some((da, db, sep))
    }
}

/// The coupling `A_jk` of up to [`BLOCK_MAX`] contact points between two bodies:
/// how much a unit multiplier (or impulse) along `n_k` at point `k` moves point
/// `j` along `n_j`:
/// `n_j · n_k (m_a⁻¹ + m_b⁻¹) + (r_aj × n_j) · I_a⁻¹ (r_ak × n_k)
///  + (r_bj × n_j) · I_b⁻¹ (r_bk × n_k)`.
pub(crate) fn coupling(
    a: &RigidBody,
    b: &RigidBody,
    ra: &[Vec3Fix; BLOCK_MAX],
    rb: &[Vec3Fix; BLOCK_MAX],
    n: &[Vec3Fix; BLOCK_MAX],
    m: usize,
) -> [[Fix128; BLOCK_MAX]; BLOCK_MAX] {
    let lin = a.inv_mass + b.inv_mass;
    let ang = |body: &RigidBody, r: &[Vec3Fix; BLOCK_MAX], j: usize, k: usize| {
        if body.inv_mass.is_zero() {
            return Fix128::ZERO;
        }
        r[j].cross(n[j]).dot(body.world_inv_inertia_apply(r[k].cross(n[k])))
    };
    let mut big_a = [[Fix128::ZERO; BLOCK_MAX]; BLOCK_MAX];
    for j in 0..m {
        for k in 0..m {
            big_a[j][k] = n[j].dot(n[k]) * lin + ang(a, ra, j, k) + ang(b, rb, j, k);
        }
    }
    big_a
}

/// The complementarity problem `λ ≥ 0`, `w = d − A λ ≤ 0`, `λ_k w_k = 0` on the
/// first `m` points: the active sets tried in a fixed order (more points first,
/// then ascending bit pattern), each solved by [`solve_small`]; the first with
/// `λ_S ≥ 0` that leaves every other point at `d_j − Σ A_jk λ_k ≤ 0` is taken.
/// `None` when none does.
pub(crate) fn block_lcp(
    big_a: &[[Fix128; BLOCK_MAX]; BLOCK_MAX],
    d: &[Fix128; BLOCK_MAX],
    m: usize,
) -> Option<[Fix128; BLOCK_MAX]> {
    let mut masks: Vec<u32> = (1..(1u32 << m)).collect();
    masks.sort_by_key(|s| (core::cmp::Reverse(s.count_ones()), *s));
    for mask in masks {
        let set: Vec<usize> = (0..m).filter(|k| mask & (1 << k) != 0).collect();
        let Some(sol) = solve_small(big_a, d, &set) else {
            continue;
        };
        let mut lambda = [Fix128::ZERO; BLOCK_MAX];
        for (i, &k) in set.iter().enumerate() {
            lambda[k] = sol[i];
        }
        if set.iter().any(|&k| lambda[k] < Fix128::ZERO) {
            continue;
        }
        let left_out_ok = (0..m).filter(|j| mask & (1 << j) == 0).all(|j| {
            let mut relieved = Fix128::ZERO;
            for &k in &set {
                relieved = relieved + big_a[j][k] * lambda[k];
            }
            d[j] - relieved <= Fix128::ZERO
        });
        if left_out_ok {
            return Some(lambda);
        }
    }
    None
}

/// `A_S x = d_S` for the rows and columns of `set`, by Gaussian elimination with
/// partial pivoting (the largest magnitude, first on ties); `None` for a zero
/// pivot.
fn solve_small(
    big_a: &[[Fix128; BLOCK_MAX]; BLOCK_MAX],
    d: &[Fix128; BLOCK_MAX],
    set: &[usize],
) -> Option<[Fix128; BLOCK_MAX]> {
    let s = set.len();
    let mut m = [[Fix128::ZERO; BLOCK_MAX + 1]; BLOCK_MAX];
    for (i, &j) in set.iter().enumerate() {
        for (c, &k) in set.iter().enumerate() {
            m[i][c] = big_a[j][k];
        }
        m[i][s] = d[j];
    }
    for col in 0..s {
        let mut piv = col;
        for row in col + 1..s {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        if m[piv][col].is_zero() {
            return None;
        }
        m.swap(col, piv);
        let pivot_row = m[col];
        for row in m.iter_mut().take(s).skip(col + 1) {
            let f = row[col] / pivot_row[col];
            for (cell, p) in row.iter_mut().zip(pivot_row.iter()).take(s + 1).skip(col) {
                *cell = *cell - f * *p;
            }
        }
    }
    let mut x = [Fix128::ZERO; BLOCK_MAX];
    for row in (0..s).rev() {
        let mut acc = m[row][s];
        for c in row + 1..s {
            acc = acc - m[row][c] * x[c];
        }
        x[row] = acc / m[row][row];
    }
    Some(x)
}
