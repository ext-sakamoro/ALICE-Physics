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

// ---------------------------------------------------------------------------
// Odd arithmetic: `Fix128`'s product floors toward −∞, so `(−a)·b` is one unit
// below `−(a·b)`. The shares of mirror-image contact points (a face's corners)
// must cancel exactly when summed, or the remainder (a few units) spins a body
// that should rest, and a stiff contact turns that tilt into uneven loads. The
// contact math therefore uses products that are odd by construction.
// ---------------------------------------------------------------------------

/// `a · b` rounded symmetrically about zero: `(−a)·b = −(a·b)` exactly.
#[inline]
pub(crate) fn mul_odd(a: Fix128, b: Fix128) -> Fix128 {
    let p = a.abs() * b.abs();
    if a.is_negative() != b.is_negative() {
        -p
    } else {
        p
    }
}

/// `v · s` componentwise with [`mul_odd`].
#[inline]
pub(crate) fn scale_odd(v: Vec3Fix, s: Fix128) -> Vec3Fix {
    Vec3Fix::new(mul_odd(v.x, s), mul_odd(v.y, s), mul_odd(v.z, s))
}

/// `u × v` with [`mul_odd`] products.
#[inline]
pub(crate) fn cross_odd(u: Vec3Fix, v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(
        mul_odd(u.y, v.z) - mul_odd(u.z, v.y),
        mul_odd(u.z, v.x) - mul_odd(u.x, v.z),
        mul_odd(u.x, v.y) - mul_odd(u.y, v.x),
    )
}

/// `u · v` with [`mul_odd`] products.
#[inline]
pub(crate) fn dot_odd(u: Vec3Fix, v: Vec3Fix) -> Fix128 {
    mul_odd(u.x, v.x) + mul_odd(u.y, v.y) + mul_odd(u.z, v.z)
}

/// The body's world inverse inertia applied to `v`, made odd:
/// `(I⁻¹ v − I⁻¹ (−v)) / 2` (the quotient truncates symmetrically), so the
/// mirror image of `v` gets exactly the negated result.
#[inline]
pub(crate) fn inv_inertia_odd(body: &RigidBody, v: Vec3Fix) -> Vec3Fix {
    let p = body.world_inv_inertia_apply(v);
    let q = body.world_inv_inertia_apply(-v);
    let two = Fix128::from_int(2);
    Vec3Fix::new((p.x - q.x) / two, (p.y - q.y) / two, (p.z - q.z) / two)
}

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
    /// The change of the contact's normal multiplier (0 for friction).
    pub(crate) dlambda: Fix128,
}

/// The arm `r` turned by `turn` (first order): `r + turn × r`.
#[inline]
pub(crate) fn arm(r: Vec3Fix, turn: Vec3Fix) -> Vec3Fix {
    r + cross_odd(turn, r)
}

/// `(r × axis)ᵀ I⁻¹ (r × axis)` of `body` (zero for a body without inverse mass).
#[inline]
pub(crate) fn angular_w(body: &RigidBody, r: Vec3Fix, axis: Vec3Fix) -> Fix128 {
    if body.inv_mass.is_zero() {
        return Fix128::ZERO;
    }
    let rn = cross_odd(r, axis);
    dot_odd(rn, inv_inertia_odd(body, rn))
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
        dx: scale_odd(axis, mul_odd(lambda, body.inv_mass)),
        dtheta: scale_odd(inv_inertia_odd(body, cross_odd(r, axis)), lambda),
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
    c.contact.depth - dot_odd(c.contact.normal, moved_a - moved_b)
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
    h: Fix128,
) -> Option<PointCorrection> {
    let d = current_depth(c, anchor, a, b, turn_a, turn_b);
    if d <= Fix128::ZERO && c.cached_lambda <= Fix128::ZERO {
        return None;
    }
    let (r_a, r_b) = (arm(anchor.r_a, turn_a), arm(anchor.r_b, turn_b));
    let n = c.contact.normal;
    let w = point_w(a, b, r_a, r_b, n);
    let alpha = compliance_over_h2(c.compliance, h);
    if w + alpha <= Fix128::ZERO {
        return None;
    }
    // Macklin et al. 2016 eq. (18): Δλ = (d − α̃ λ) / (w + α̃), the total kept
    // non-negative (a contact only pushes).
    let dlambda = ((d - alpha * c.cached_lambda) / (w + alpha)).max(-c.cached_lambda);
    if dlambda.is_zero() {
        return None;
    }
    Some(PointCorrection {
        a: share(a, r_a, n, dlambda),
        b: share(b, r_b, n, -dlambda),
        dlambda,
    })
}

/// `α̃ = α / h²` (zero for a zero substep width).
#[inline]
pub(crate) fn compliance_over_h2(alpha: Fix128, h: Fix128) -> Fix128 {
    let h2 = h * h;
    if h2.is_zero() {
        Fix128::ZERO
    } else {
        alpha / h2
    }
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
    if c.friction * c.cached_lambda <= Fix128::ZERO {
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
    if dp_t.length_squared().is_zero() {
        return None;
    }
    let (t, len) = dp_t.normalize_with_length();
    let w = point_w(a, b, r_a, r_b, t);
    if w <= Fix128::ZERO {
        return None;
    }
    // Static while the multiplier that holds the point, `λ_t = |Δp_t| / w`, stays
    // inside the cone `μ λ_n` (eq. (27)).
    let lambda = len / w;
    if lambda >= c.friction * c.cached_lambda {
        return None;
    }
    // A's point moved by `+dp_t` relative to B's: A is moved back along `−t`.
    Some(PointCorrection {
        a: share(a, r_a, t, -lambda),
        b: share(b, r_b, t, lambda),
        dlambda: Fix128::ZERO,
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
    (a.velocity + cross_odd(a.angular_velocity, r_a))
        - (b.velocity + cross_odd(b.angular_velocity, r_b))
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
/// The regularisation of a rigid push-out block, `2⁻⁵⁰` of its largest
/// diagonal entry: it only makes the multipliers unique (a face's corners
/// constrain three motions); the bodies' motion does not depend on it (see
/// the unit test `the_rigid_push_out_does_not_depend_on_the_regularisation`).
pub(crate) const RIGID_REGULARIZATION: Fix128 = Fix128 { hi: 0, lo: 1 << 14 };

/// The rigid push-out of points `0..m` (arms `ra` on `a`, `rb` on `b`,
/// normals `n` from `b` to `a`) by the overlaps `d`: the least-constraint
/// motion that leaves no point overlapping (Gauss's principle in the bodies'
/// mass metric), `λ` from the point-space block `(A + ε I) λ = d` with the
/// sign conditions (`block_lcp`), `ε = RIGID_REGULARIZATION · max Aₖₖ`, and
/// the motions `W Jᵀ λ`: equal and opposite impulses on the two bodies. `None`
/// when no active set is feasible.
#[allow(
    clippy::too_many_arguments,
    reason = "the two bodies, the points' arms and normals, the overlaps and the point count are the push-out's whole input"
)]
pub(crate) fn rigid_push_out(
    a: &RigidBody,
    b: &RigidBody,
    ra: &[Vec3Fix; BLOCK_MAX],
    rb: &[Vec3Fix; BLOCK_MAX],
    n: &[Vec3Fix; BLOCK_MAX],
    d: &[Fix128; BLOCK_MAX],
    m: usize,
    scale: Fix128,
) -> Option<(BodyDelta, BodyDelta)> {
    let mut big_a = coupling(a, b, ra, rb, n, m);
    let mut top = Fix128::ZERO;
    for (k, row) in big_a.iter().enumerate().take(m) {
        top = top.max(row[k]);
    }
    let eps = mul_odd(top, scale);
    for (k, row) in big_a.iter_mut().enumerate().take(m) {
        row[k] = row[k] + eps;
    }
    // `λ` is linear in `d`: a small push is solved scaled up by an exact power
    // of two (to `|d| ≥ 1/2`) and scaled back, so the block keeps its
    // precision (`Fix128` resolves `2⁻⁶⁴` absolutely, and a face's block is
    // regularised only by `2⁻⁵⁰` of its scale).
    let mut biggest = Fix128::ZERO;
    for x in d.iter().take(m) {
        biggest = biggest.max(x.abs());
    }
    let mut shift = 0u32;
    while !biggest.is_zero() && biggest < Fix128::from_ratio(1, 2) && shift < 62 {
        biggest = biggest + biggest;
        shift += 1;
    }
    let up = Fix128::from_int(1i64 << shift);
    let mut scaled = [Fix128::ZERO; BLOCK_MAX];
    for k in 0..m {
        scaled[k] = d[k] * up;
    }
    let mut lambda = block_lcp(&big_a, &scaled, m)?;
    for x in lambda.iter_mut().take(m) {
        *x = *x / up;
    }
    let (mut da, mut db) = (BodyDelta::ZERO, BodyDelta::ZERO);
    for k in 0..m {
        let (pa, pb) = (
            share(a, ra[k], n[k], lambda[k]),
            share(b, rb[k], n[k], -lambda[k]),
        );
        da.dx = da.dx + pa.dx;
        da.dtheta = da.dtheta + pa.dtheta;
        db.dx = db.dx + pb.dx;
        db.dtheta = db.dtheta + pb.dtheta;
    }
    Some((da, db))
}

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
/// Returns each body's share and each point's change of multiplier (`Δλ_k`,
/// added to its `cached_lambda`).
pub(crate) fn block_normal_correction(
    points: &[(ContactConstraint, ContactAnchor)],
    a: &RigidBody,
    b: &RigidBody,
    turn_a: Vec3Fix,
    turn_b: Vec3Fix,
    h: Fix128,
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
    let held: [Fix128; BLOCK_MAX] =
        core::array::from_fn(|k| points.get(k).map_or(Fix128::ZERO, |p| p.0.cached_lambda));
    if d[..m].iter().all(|x| *x <= Fix128::ZERO) && held[..m].iter().all(|x| *x <= Fix128::ZERO) {
        return None;
    }
    // With compliance (Macklin et al. 2016) the totals `λ' = λ + Δλ ≥ 0` solve
    // `(A + α̃ I) λ' = d + A λ` on the active set: `(A + α̃ I) Δλ = d − α̃ λ`
    // rewritten for the totals, so the complementarity is on what each point
    // pushes in all. `α̃ > 0` makes the matrix regular even when the points
    // constrain fewer motions than there are points (a face's four corners).
    let big_a = coupling(a, b, &ra, &rb, &n, m);
    let mut stiff = big_a;
    let mut rhs = d;
    for j in 0..m {
        stiff[j][j] = stiff[j][j] + compliance_over_h2(points[j].0.compliance, h);
        for k in 0..m {
            rhs[j] = rhs[j] + big_a[j][k] * held[k];
        }
    }
    let alphas: [Fix128; BLOCK_MAX] = core::array::from_fn(|k| {
        points
            .get(k)
            .map_or(Fix128::ZERO, |p| compliance_over_h2(p.0.compliance, h))
    });
    let total = match body_lcp(a, b, &ra, &rb, &n, &alphas, &rhs, m) {
        Some(t) => t,
        None => block_lcp(&stiff, &rhs, m)?,
    };
    let lambda: [Fix128; BLOCK_MAX] = core::array::from_fn(|k| total[k] - held[k]);
    {
        let set: Vec<usize> = (0..m).filter(|&k| !lambda[k].is_zero()).collect();
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
            sep[k] = lambda[k];
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
        dot_odd(
            cross_odd(r[j], n[j]),
            inv_inertia_odd(body, cross_odd(r[k], n[k])),
        )
    };
    let mut big_a = [[Fix128::ZERO; BLOCK_MAX]; BLOCK_MAX];
    for j in 0..m {
        for k in 0..m {
            big_a[j][k] = mul_odd(dot_odd(n[j], n[k]), lin) + ang(a, ra, j, k) + ang(b, rb, j, k);
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

/// The compliant complementarity problem of a block, `(A + D) λ = r`,
/// `D = diag(α̃_k) > 0`, solved in the bodies' coordinates instead of the
/// points' (Woodbury): with `J` the points' rows (body A `[n, r_a × n]`, body B
/// `[−n, −(r_b × n)]`, a body without inverse mass left out) and `W` the bodies'
/// inverse mass (`m⁻¹ I`, `I⁻¹` in the world frame), the bodies' generalised
/// impulse `y = Jᵀ λ` solves `(I + Jᵀ D⁻¹ J W) y = Jᵀ D⁻¹ r` (6 unknowns per body
/// that moves) and each point's multiplier is `λ_k = (r_k − J_k W y) / α̃_k`.
///
/// Why: the multipliers of a face's corners are fixed only through the
/// compliance (four corners constrain three motions), so eliminating in the
/// points' coordinates rounds them unevenly (the pivot order favours some), and
/// the uneven multipliers tilt the body; in the bodies' coordinates a symmetric
/// face makes the coupling between force and tilt cancel exactly, the tilt part
/// of `y` is exactly zero, and every corner gets the same `λ` by construction.
///
/// The active sets are tried in [`block_lcp`]'s order; the body system is
/// eliminated in a fixed order (row by row, no pivoting: its matrix is the
/// identity plus a positive semi-definite product, so the pivots stay at least
/// 1 in exact arithmetic). `None` when a point has `α̃ = 0` (the caller then
/// solves in the points' coordinates, [`block_lcp`]) or no active set fits.
#[allow(
    clippy::too_many_arguments,
    reason = "the two bodies, the points' arms and normals, their compliances, the right-hand side and the point count are the block's whole input"
)]
pub(crate) fn body_lcp(
    a: &RigidBody,
    b: &RigidBody,
    ra: &[Vec3Fix; BLOCK_MAX],
    rb: &[Vec3Fix; BLOCK_MAX],
    n: &[Vec3Fix; BLOCK_MAX],
    alpha: &[Fix128; BLOCK_MAX],
    r: &[Fix128; BLOCK_MAX],
    m: usize,
) -> Option<[Fix128; BLOCK_MAX]> {
    if alpha[..m].iter().any(|x| *x <= Fix128::ZERO) {
        return None;
    }
    // Bodies that move, each with 6 unknowns (linear, angular).
    let moving: Vec<(&RigidBody, bool)> = [(a, true), (b, false)]
        .into_iter()
        .filter(|(body, _)| !body.inv_mass.is_zero())
        .collect();
    let dof = 6 * moving.len();
    if dof == 0 {
        return None;
    }
    // Rows J_k (length dof) and W applied to a dof vector.
    let row = |k: usize| -> Vec<Fix128> {
        let mut out = Vec::with_capacity(dof);
        for (_, is_a) in &moving {
            let (lin, ang) = if *is_a {
                (n[k], cross_odd(ra[k], n[k]))
            } else {
                (-n[k], -cross_odd(rb[k], n[k]))
            };
            out.extend_from_slice(&[lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]);
        }
        out
    };
    let apply_w = |v: &[Fix128]| -> Vec<Fix128> {
        let mut out = Vec::with_capacity(dof);
        for (i, (body, _)) in moving.iter().enumerate() {
            let o = 6 * i;
            out.extend_from_slice(&[
                mul_odd(v[o], body.inv_mass),
                mul_odd(v[o + 1], body.inv_mass),
                mul_odd(v[o + 2], body.inv_mass),
            ]);
            let w = inv_inertia_odd(body, Vec3Fix::new(v[o + 3], v[o + 4], v[o + 5]));
            out.extend_from_slice(&[w.x, w.y, w.z]);
        }
        out
    };
    let rows: Vec<Vec<Fix128>> = (0..m).map(row).collect();
    let jw: Vec<Vec<Fix128>> = rows.iter().map(|j| apply_w(j)).collect();
    let dot = |x: &[Fix128], y: &[Fix128]| {
        let mut s = Fix128::ZERO;
        for (p, q) in x.iter().zip(y) {
            s = s + mul_odd(*p, *q);
        }
        s
    };
    let mut masks: Vec<u32> = (1..(1u32 << m)).collect();
    masks.sort_by_key(|s| (core::cmp::Reverse(s.count_ones()), *s));
    for mask in masks {
        let set: Vec<usize> = (0..m).filter(|k| mask & (1 << k) != 0).collect();
        // K = I + Σ J_kᵀ (J_k W) / α̃_k, rhs = Σ J_kᵀ r_k / α̃_k.
        let mut kmat = vec![vec![Fix128::ZERO; dof + 1]; dof];
        for (i, krow) in kmat.iter_mut().enumerate() {
            krow[i] = Fix128::ONE;
        }
        for &k in &set {
            let inv = Fix128::ONE / alpha[k];
            for i in 0..dof {
                let ji = mul_odd(rows[k][i], inv);
                if ji.is_zero() {
                    continue;
                }
                for c in 0..dof {
                    kmat[i][c] = kmat[i][c] + mul_odd(ji, jw[k][c]);
                }
                kmat[i][dof] = kmat[i][dof] + mul_odd(ji, r[k]);
            }
        }
        // Fixed-order elimination (no pivoting).
        let mut singular = false;
        for col in 0..dof {
            let p = kmat[col][col];
            if p.is_zero() {
                singular = true;
                break;
            }
            let pivot_row = kmat[col].clone();
            for krow in kmat.iter_mut().skip(col + 1) {
                let f = krow[col] / p;
                if f.is_zero() {
                    continue;
                }
                for (cell, q) in krow.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *cell = *cell - mul_odd(f, *q);
                }
            }
        }
        if singular {
            continue;
        }
        let mut y = vec![Fix128::ZERO; dof];
        for i in (0..dof).rev() {
            let mut acc = kmat[i][dof];
            for c in i + 1..dof {
                acc = acc - mul_odd(kmat[i][c], y[c]);
            }
            y[i] = acc / kmat[i][i];
        }
        let mut lambda = [Fix128::ZERO; BLOCK_MAX];
        let mut ok = true;
        for j in 0..m {
            let left = r[j] - dot(&jw[j], &y);
            if mask & (1 << j) != 0 {
                lambda[j] = left / alpha[j];
                if lambda[j] < Fix128::ZERO {
                    ok = false;
                    break;
                }
            } else if left > Fix128::ZERO {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(lambda);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    /// The motion `(Δx, Δθ)` a rigid push-out gives a 0.9 × 0.12 × 0.6 m box
    /// (500 kg/m³) whose bottom face's four corners overlap a floor by
    /// different depths, with the block regularised by `alpha` (relative to
    /// its largest diagonal entry).
    fn push_out(alpha: Fix128) -> (Vec3Fix, Vec3Fix) {
        let (hx, hy, hz) = (0.45, 0.06, 0.3);
        let m = 500.0 * 8.0 * hx * hy * hz;
        let i = |a: f64, b: f64| m * (4.0 * a * a + 4.0 * b * b) / 12.0;
        let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, fx(m));
        body.inv_inertia = Vec3Fix::new(
            fx(1.0 / i(hy, hz)),
            fx(1.0 / i(hx, hz)),
            fx(1.0 / i(hx, hy)),
        );
        let still = RigidBody::new_static(Vec3Fix::ZERO);
        let up = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let mut ra = [Vec3Fix::ZERO; BLOCK_MAX];
        let mut d = [Fix128::ZERO; BLOCK_MAX];
        for (k, (sx, sz, depth)) in [
            (1.0, 1.0, 3e-3),
            (-1.0, 1.0, 2e-3),
            (-1.0, -1.0, 1e-3),
            (1.0, -1.0, 2e-3),
        ]
        .into_iter()
        .enumerate()
        {
            ra[k] = Vec3Fix::new(fx(sx * hx), fx(-hy), fx(sz * hz));
            d[k] = fx(depth);
        }
        let (da, _) = rigid_push_out(
            &body,
            &still,
            &ra,
            &[Vec3Fix::ZERO; BLOCK_MAX],
            &[up; BLOCK_MAX],
            &d,
            BLOCK_MAX,
            alpha,
        )
        .expect("the block is solvable");
        (da.dx, da.dtheta)
    }

    /// oracle (rigid statics of a face): the four overlaps lie on a plane
    /// (`3 + 1 = 2 + 2` mm), so the rigid push-out that leaves every corner
    /// just touching is the plane's lift and tilt, unique although the four
    /// multipliers are not (four points, three motions). The regulariser only
    /// picks the multipliers; over `2⁻⁵⁵, 2⁻⁵⁰, 2⁻⁴⁵` (three orders of
    /// magnitude about the stated one) the motion agrees to `1e-12` m / rad
    /// and every corner is out to `1e-12` m.
    #[test]
    fn the_rigid_push_out_does_not_depend_on_the_regularisation() {
        let at = |shift: u32| Fix128 {
            hi: 0,
            lo: 1u64 << shift,
        };
        let reference = push_out(RIGID_REGULARIZATION);
        for alpha in [at(9), at(14), at(19)] {
            let (dx, dth) = push_out(alpha);
            let gap = (dx - reference.0)
                .length()
                .to_f64()
                .max((dth - reference.1).length().to_f64());
            assert!(
                gap < 1e-12,
                "motion moves by {gap:.3e} with the regulariser"
            );
            for (sx, sz, depth) in [
                (1.0, 1.0, 3e-3),
                (-1.0, 1.0, 2e-3),
                (-1.0, -1.0, 1e-3),
                (1.0, -1.0, 2e-3),
            ] {
                let r = Vec3Fix::new(fx(sx * 0.45), fx(-0.06), fx(sz * 0.3));
                let lift = (dx + dth.cross(r)).y.to_f64();
                assert!(
                    (lift - depth).abs() < 1e-12,
                    "corner ({sx},{sz}) lifted {lift:.6e}, want {depth}"
                );
            }
        }
        // The material compliance (1e-10 m/N over a 1/240 s substep squared)
        // is not a rigid push-out: it leaves a measurable share of the overlap.
        // relative to the block's largest entry (`1/m + …`, about 0.04 1/kg)
        let h = fx(1.0 / 240.0);
        let soft = compliance_over_h2(fx(1e-10), h) / fx(0.04);
        let (dx, _) = push_out(soft);
        assert!(
            (dx.y.to_f64() - 2e-3).abs() > 1e-9,
            "the material compliance pushes out the whole overlap"
        );
    }
}
