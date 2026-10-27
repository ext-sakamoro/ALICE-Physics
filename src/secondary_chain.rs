//! Secondary motion chains (hair, tails, ribbons, skirts): particles at the tips of bones that
//! follow an animated pose with inertia, gravity and a pull back to the pose
//!
//! A chain is a tree of particles, each the tip of a bone hanging from its parent particle (or,
//! for a root, from an anchor that the animation moves). Each step reads the animated position of
//! every particle (where the pose alone would put it) and the anchor's translation since the last
//! step, then for every particle in order, parents first:
//!
//! 1. inertia: `x = p + momentum · (p − q) + immobile · anchor_delta`
//! 2. gravity: `x += gravity_vector · gravity · (1 − falloff · max(0, rest_dir · g_dir)) · dt²`
//! 3. pull to the pose: the target is the simulated parent plus the animated link
//!    `t = sim_parent + (anim_i − anim_parent)`, `x += (t − x) · pull`
//! 4. stiffness: the target turned with the parent link's rotation from its pose,
//!    `x += (t' − x) · stiffness`
//! 5. `q = p`, `p = x`, then the constraints, `iterations` times, in priority order: the rest
//!    length to the parent, the angle limit, the capsule colliders; and the rest length once more
//!
//! Every projection leaves a particle that already satisfies it untouched, so a chain at its pose
//! with no gravity stays there bit for bit. The arithmetic is `Fix128` (the trigonometry is
//! CORDIC, no `alice-det-math`), so a run is the same on every platform.

use crate::collider::Capsule;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::world_shape_query::closest_on_segment;

/// How a particle's direction from its parent is limited, around its rest direction (the pose's
/// link turned with the parent link's rotation from its pose)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainLimit {
    /// No limit
    None,
    /// Within `max` radians of the rest direction (a cone)
    Angle {
        /// The cone's half angle (0..=π)
        max: Fix128,
    },
    /// In the plane through the rest direction perpendicular to `axis`, within `max` of the rest
    /// direction
    Hinge {
        /// The hinge axis in the bone's own frame (turned each step by the bone's animated frame
        /// and the parent link's turn from its pose)
        axis: Vec3Fix,
        /// The largest swing in the plane (0..=π)
        max: Fix128,
    },
    /// Within an ellipse of angles: `(a_x / max_x)² + (a_z / max_z)² <= 1`, `a_x` and `a_z`
    /// the angles from the rest direction toward `x_axis` and toward `rest × x_axis`
    Polar {
        /// The reference axis for `max_x` in the bone's own frame (turned like the hinge axis)
        x_axis: Vec3Fix,
        /// The largest angle toward `x_axis` (> 0)
        max_x: Fix128,
        /// The largest angle toward the other axis (> 0)
        max_z: Fix128,
    },
}

/// The settings of one particle (already scaled by its chain's curves)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainParams {
    /// Pull back to the pose per step (0..=1)
    pub pull: Fix128,
    /// The share of last step's motion kept (0..=1)
    pub momentum: Fix128,
    /// Pull toward the pose turned with the parent (0..=1)
    pub stiffness: Fix128,
    /// The share of the gravity vector applied (0..=1)
    pub gravity: Fix128,
    /// How much less gravity acts when the rest direction already points along it (0..=1)
    pub gravity_falloff: Fix128,
    /// The share of the anchor's translation the particle takes at once (0..=1)
    pub immobile: Fix128,
    /// The angle limit
    pub limit: ChainLimit,
    /// The particle's radius against the colliders (>= 0)
    pub radius: Fix128,
    /// The colliders (bit k = the step's `colliders[k]`) that the authored rest attaches this link
    /// inside: they do not act on it (see [`SecondaryChain::attached_inside`])
    pub attached_inside: u64,
}

/// A particle of a chain
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainParticle {
    /// The parent particle's index (earlier in the chain), or `None` for one hanging from the
    /// anchor
    pub parent: Option<usize>,
    /// The bone's rest length (the distance to the parent particle, or to the anchor's point)
    pub rest_length: Fix128,
    /// Its settings
    pub params: ChainParams,
    /// Position now
    pub p: Vec3Fix,
    /// Position one step ago
    pub q: Vec3Fix,
}

/// A chain's global settings
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainConfig {
    /// Gravity (acceleration, world units per second squared)
    pub gravity: Vec3Fix,
    /// Constraint passes per step (>= 1)
    pub iterations: u32,
}

/// What one step reads: the anchor, the animated pose and the colliders now
#[derive(Clone, Copy, Debug)]
pub struct ChainInput<'a> {
    /// Where each root particle's anchor point is now
    pub anchor: Vec3Fix,
    /// The animated anchor point (for the pose link of a root)
    pub anim_anchor: Vec3Fix,
    /// The animated position of every particle
    pub anim: &'a [Vec3Fix],
    /// The animated world rotation of every particle's bone (the frame its limit axes are given
    /// in)
    pub anim_frames: &'a [QuatFix],
    /// The anchor's translation since the last step
    pub anchor_delta: Vec3Fix,
    /// The capsules now
    pub colliders: &'a [Capsule],
}

/// What the chain refuses
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainError {
    /// A parent index not before the particle
    BadParent(usize),
    /// An input of the wrong length
    BadInput(&'static str),
    /// A setting out of its range (the particle's index and the setting)
    BadParam(usize, &'static str),
    /// No constraint pass
    NoIterations,
    /// More colliders than a mask holds (64)
    TooManyColliders(usize),
}

impl core::fmt::Display for ChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadParent(i) => write!(f, "particle {i}: its parent is not an earlier particle"),
            Self::BadInput(w) => write!(f, "input of the wrong length: {w}"),
            Self::BadParam(i, w) => write!(f, "particle {i}: {w} out of range"),
            Self::NoIterations => write!(f, "no constraint pass"),
            Self::TooManyColliders(n) => write!(f, "{n} colliders, a mask holds 64"),
        }
    }
}

/// A secondary motion chain
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecondaryChain {
    particles: Vec<ChainParticle>,
    config: ChainConfig,
    /// per particle, per collider: the radius at most (the rest clearance), after `fit_to_rest`
    radius_caps: Vec<Vec<Fix128>>,
    /// the number of colliders `fit_to_rest` fitted to, which every later step must pass
    fitted: Option<usize>,
}

fn unit01(x: Fix128) -> bool {
    !x.is_negative() && x <= Fix128::ONE
}

/// The direction `from` turned toward `to` by `angle` (both unit, not opposite)
fn turn_toward(from: Vec3Fix, to: Vec3Fix, angle: Fix128) -> Vec3Fix {
    let axis = from.cross(to);
    if axis == Vec3Fix::ZERO {
        return from;
    }
    QuatFix::from_axis_angle(axis, angle).rotate_vec(from)
}

impl SecondaryChain {
    /// A chain of these particles (parents before children), at rest where they stand
    ///
    /// # Errors
    ///
    /// [`ChainError`] for a parent not before its particle, a setting out of its range or no
    /// constraint pass
    pub fn new(particles: Vec<ChainParticle>, config: ChainConfig) -> Result<Self, ChainError> {
        if config.iterations == 0 {
            return Err(ChainError::NoIterations);
        }
        for (i, pt) in particles.iter().enumerate() {
            if pt.parent.is_some_and(|p| p >= i) {
                return Err(ChainError::BadParent(i));
            }
            let c = &pt.params;
            for (v, w) in [
                (c.pull, "pull"),
                (c.momentum, "momentum"),
                (c.stiffness, "stiffness"),
                (c.gravity, "gravity"),
                (c.gravity_falloff, "gravity falloff"),
                (c.immobile, "immobile"),
            ] {
                if !unit01(v) {
                    return Err(ChainError::BadParam(i, w));
                }
            }
            if c.radius.is_negative() || pt.rest_length.is_negative() {
                return Err(ChainError::BadParam(i, "radius or rest length"));
            }
            let bad_angle = |a: Fix128| a.is_negative() || a > Fix128::PI;
            let ok = match c.limit {
                ChainLimit::None => true,
                ChainLimit::Angle { max } | ChainLimit::Hinge { max, .. } => !bad_angle(max),
                ChainLimit::Polar { max_x, max_z, .. } => {
                    !bad_angle(max_x) && !bad_angle(max_z) && !max_x.is_zero() && !max_z.is_zero()
                }
            };
            if !ok {
                return Err(ChainError::BadParam(i, "limit angle"));
            }
        }
        Ok(Self {
            particles,
            config,
            radius_caps: Vec::new(),
            fitted: None,
        })
    }

    /// Which colliders the authored rest attaches each link inside: bit k of entry i is set when
    /// `colliders[k]` holds the rest position of particle i's parent end (its parent's rest
    /// position, or `rest_anchor` for a root) strictly within the capsule's own radius. The rest is
    /// the pose the character was built in; masks are configuration derived from it, so a changed
    /// rest skeleton needs them computed again
    ///
    /// # Errors
    ///
    /// [`ChainError::TooManyColliders`] for more than 64 colliders, [`ChainError::BadInput`]
    /// when `rest` is not one position per particle
    pub fn attached_inside(
        particles: &[ChainParticle],
        rest: &[Vec3Fix],
        rest_anchor: Vec3Fix,
        colliders: &[Capsule],
    ) -> Result<Vec<u64>, ChainError> {
        if colliders.len() > 64 {
            return Err(ChainError::TooManyColliders(colliders.len()));
        }
        if rest.len() != particles.len() {
            return Err(ChainError::BadInput("rest"));
        }
        Ok(particles
            .iter()
            .zip(rest)
            .map(|(pt, _)| {
                let end = pt
                    .parent
                    .and_then(|p| rest.get(p).copied())
                    .unwrap_or(rest_anchor);
                colliders.iter().enumerate().fold(0u64, |m, (k, cap)| {
                    let inside =
                        (end - closest_on_segment(cap.a, cap.b, end)).length() < cap.radius;
                    if inside {
                        m | (1u64 << k)
                    } else {
                        m
                    }
                })
            })
            .collect())
    }

    /// Fits the chain, standing at its rest, to the colliders as the rest places them: the rest is
    /// collision free. Each particle's mask gets the colliders that hold its parent end (as
    /// [`SecondaryChain::attached_inside`]) or its own centre; against every other collider its
    /// radius is at most its rest clearance (its rest distance to the capsule's surface). The
    /// colliders are the step's list in the same order: every later step must pass as many
    /// colliders (it refuses another count; the order it cannot check), and a changed rest needs
    /// the fit again. The fitted caps are state of this value only: the chain is not part of a
    /// world snapshot or of a digest, so a chain built again (or read back from elsewhere) is
    /// fitted again
    ///
    /// # Errors
    ///
    /// [`ChainError::TooManyColliders`] for more than 64 colliders
    pub fn fit_to_rest(
        &mut self,
        rest_anchor: Vec3Fix,
        rest_colliders: &[Capsule],
    ) -> Result<(), ChainError> {
        let rest: Vec<Vec3Fix> = self.particles.iter().map(|p| p.p).collect();
        let masks = Self::attached_inside(&self.particles, &rest, rest_anchor, rest_colliders)?;
        self.radius_caps = Vec::with_capacity(self.particles.len());
        for (pt, m) in self.particles.iter_mut().zip(masks) {
            let mut mask = m;
            let mut caps = Vec::with_capacity(rest_colliders.len());
            for (k, cap) in rest_colliders.iter().enumerate() {
                let d = (pt.p - closest_on_segment(cap.a, cap.b, pt.p)).length();
                let mut r = pt.params.radius;
                if d <= cap.radius {
                    // the rest puts the particle's centre inside: it lives in that body
                    mask |= 1u64 << k;
                } else if d - cap.radius < r {
                    r = d - cap.radius;
                }
                caps.extend([r]);
            }
            pt.params.attached_inside = mask;
            self.radius_caps.extend([caps]);
        }
        self.fitted = Some(rest_colliders.len());
        Ok(())
    }

    /// The particles
    #[must_use]
    pub fn particles(&self) -> &[ChainParticle] {
        &self.particles
    }

    /// One step of `dt` under `input` (the anchor, the animated pose and the colliders now)
    ///
    /// # Errors
    ///
    /// [`ChainError::BadInput`] when `input.anim` is not one position per particle,
    /// `input.anim_frames` not one rotation per particle, or (after [`SecondaryChain::fit_to_rest`])
    /// `input.colliders` not as many as the fit's; [`ChainError::TooManyColliders`] for more than
    /// 64 colliders (a mask holds 64)
    #[allow(
        clippy::too_many_lines,
        reason = "the five stages and the constraints, in order"
    )]
    pub fn step(&mut self, dt: Fix128, input: &ChainInput<'_>) -> Result<(), ChainError> {
        let ChainInput {
            anchor,
            anim_anchor,
            anim,
            anim_frames,
            anchor_delta,
            colliders,
        } = *input;
        if anim.len() != self.particles.len() {
            return Err(ChainError::BadInput("anim"));
        }
        if anim_frames.len() != self.particles.len() {
            return Err(ChainError::BadInput("anim_frames"));
        }
        if colliders.len() > 64 {
            return Err(ChainError::TooManyColliders(colliders.len()));
        }
        if self.fitted.is_some_and(|n| n != colliders.len()) {
            return Err(ChainError::BadInput("colliders"));
        }
        let g = self.config.gravity;
        let g_dir = g.try_normalize().unwrap_or(Vec3Fix::ZERO);
        let dt2 = dt * dt;
        for i in 0..self.particles.len() {
            let pt = self.particles[i];
            let c = pt.params;
            let (sim_parent, anim_parent) = match pt.parent {
                Some(p) => (self.particles[p].p, anim[p]),
                None => (anchor, anim_anchor),
            };
            let link = anim[i] - anim_parent;
            // the parent link's rotation from its pose (identity for a root)
            let turn = match pt
                .parent
                .and_then(|p| self.particles[p].parent.map(|g2| (p, g2)))
            {
                Some((p, gp)) => QuatFix::from_to(
                    anim[p] - anim[gp],
                    self.particles[p].p - self.particles[gp].p,
                ),
                None => match pt.parent {
                    Some(p) => {
                        QuatFix::from_to(anim[p] - anim_anchor, self.particles[p].p - anchor)
                    }
                    None => QuatFix::IDENTITY,
                },
            };
            let rest_dir = if turn == QuatFix::IDENTITY {
                link
            } else {
                turn.rotate_vec(link)
            };
            // 1. inertia
            let mut x = pt.p;
            if !c.momentum.is_zero() {
                x = x + (pt.p - pt.q) * c.momentum;
            }
            if !c.immobile.is_zero() {
                x = x + anchor_delta * c.immobile;
            }
            // 2. gravity
            if !c.gravity.is_zero() && g_dir != Vec3Fix::ZERO {
                let along = rest_dir
                    .try_normalize()
                    .map_or(Fix128::ZERO, |d| d.dot(g_dir));
                let along = if along.is_negative() {
                    Fix128::ZERO
                } else {
                    along
                };
                let share = Fix128::ONE - c.gravity_falloff * along;
                x = x + g * (c.gravity * share * dt2);
            }
            // 3. pull to the pose
            let t = sim_parent + link;
            if !c.pull.is_zero() {
                x = x + (t - x) * c.pull;
            }
            // 4. stiffness
            if !c.stiffness.is_zero() {
                let t2 = sim_parent + rest_dir;
                x = x + (t2 - x) * c.stiffness;
            }
            self.particles[i].q = pt.p;
            self.particles[i].p = x;
            // 5. constraints: rest length > limit > colliders, then the rest length again
            for _ in 0..self.config.iterations {
                self.keep_length(i, sim_parent);
                self.keep_limit(i, sim_parent, rest_dir, turn.mul(anim_frames[i]));
                self.keep_out(i, colliders);
            }
            self.keep_length(i, sim_parent);
        }
        Ok(())
    }

    fn keep_length(&mut self, i: usize, parent: Vec3Fix) {
        let pt = self.particles[i];
        let d = pt.p - parent;
        let (dir, len) = d.normalize_with_length();
        if len == pt.rest_length || dir == Vec3Fix::ZERO {
            return;
        }
        self.particles[i].p = parent + dir * pt.rest_length;
    }

    /// `frame` turns the limit's axes from the bone's frame to the world: the bone's animated frame,
    /// then the parent link's turn from its pose
    fn keep_limit(&mut self, i: usize, parent: Vec3Fix, rest_dir: Vec3Fix, frame: QuatFix) {
        let pt = self.particles[i];
        let Some(r) = rest_dir.try_normalize() else {
            return;
        };
        let (dir, len) = (pt.p - parent).normalize_with_length();
        if dir == Vec3Fix::ZERO {
            return;
        }
        let limited = match pt.params.limit {
            ChainLimit::None => return,
            ChainLimit::Angle { max } => {
                if dir.dot(r) >= max.cos() {
                    return;
                }
                turn_toward(r, dir, max)
            }
            ChainLimit::Hinge { axis, max } => {
                let Some(ax) = frame.rotate_vec(axis).try_normalize() else {
                    return;
                };
                // in the plane perpendicular to the axis
                let inplane = dir - ax * dir.dot(ax);
                let Some(inplane) = inplane.try_normalize() else {
                    return;
                };
                let swing = if inplane.dot(r) >= max.cos() {
                    inplane
                } else {
                    turn_toward(r, inplane, max)
                };
                if swing == dir {
                    return;
                }
                swing
            }
            ChainLimit::Polar {
                x_axis,
                max_x,
                max_z,
            } => {
                let x_axis = frame.rotate_vec(x_axis);
                let xa = (x_axis - r * x_axis.dot(r)).try_normalize();
                let Some(xa) = xa else { return };
                let za = r.cross(xa);
                let ax = Fix128::atan2(dir.dot(xa), dir.dot(r));
                let az = Fix128::atan2(dir.dot(za), dir.dot(r));
                let e = (ax / max_x) * (ax / max_x) + (az / max_z) * (az / max_z);
                if e <= Fix128::ONE {
                    return;
                }
                let k = Fix128::ONE / e.sqrt();
                let (ax, az) = (ax * k, az * k);
                // the direction at angles (ax, az): turn about za by ax, then about the turned xa
                let total = (ax * ax + az * az).sqrt();
                let toward = (xa * ax + za * az).try_normalize().unwrap_or(xa);
                turn_toward(r, toward, total)
            }
        };
        self.particles[i].p = parent + limited * len;
    }

    /// Out of every capsule, except the ones the authored rest attaches this link inside (its
    /// mask): a bone attached inside a body is not constrained by that body
    fn keep_out(&mut self, i: usize, colliders: &[Capsule]) {
        let mask = self.particles[i].params.attached_inside;
        for (k, cap) in colliders.iter().enumerate() {
            // at most 64 colliders: `step` refuses more
            if mask & (1u64 << k) != 0 {
                continue;
            }
            // at most the rest clearance against this collider (after `fit_to_rest`)
            let radius = self
                .radius_caps
                .get(i)
                .and_then(|c| c.get(k))
                .copied()
                .unwrap_or(self.particles[i].params.radius);
            let p = self.particles[i].p;
            let c = closest_on_segment(cap.a, cap.b, p);
            let reach = radius + cap.radius;
            let (n, d) = (p - c).normalize_with_length();
            if d >= reach || n == Vec3Fix::ZERO {
                continue;
            }
            self.particles[i].p = c + n * reach;
        }
    }
}

/// A key of a curve (a Hermite keyframe, not weighted): time, value and the
/// slopes in and out
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CurveKey {
    /// Where the key is
    pub time: Fix128,
    /// Its value
    pub value: Fix128,
    /// The slope coming in
    pub in_slope: Fix128,
    /// The slope going out
    pub out_slope: Fix128,
}

/// A curve of keys in time order: cubic Hermite between keys, held at the first and last values
/// outside them; no key is 1 everywhere (a curve that does not scale)
#[must_use]
pub fn curve_value(keys: &[CurveKey], t: Fix128) -> Fix128 {
    let Some(first) = keys.first() else {
        return Fix128::ONE;
    };
    let last = keys[keys.len() - 1];
    if t <= first.time {
        return first.value;
    }
    if t >= last.time {
        return last.value;
    }
    for w in keys.windows(2) {
        let (k0, k1) = (w[0], w[1]);
        if t <= k1.time {
            let span = k1.time - k0.time;
            if span.is_zero() {
                return k1.value;
            }
            let s = (t - k0.time) / span;
            let s2 = s * s;
            let s3 = s2 * s;
            let two = Fix128::from_int(2);
            let three = Fix128::from_int(3);
            let h00 = two * s3 - three * s2 + Fix128::ONE;
            let h10 = s3 - two * s2 + s;
            let h01 = three * s2 - two * s3;
            let h11 = s3 - s2;
            return h00 * k0.value
                + h10 * span * k0.out_slope
                + h01 * k1.value
                + h11 * span * k1.in_slope;
        }
    }
    last.value
}
