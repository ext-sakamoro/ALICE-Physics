//! Limits on how a layered touch playback moves its bodies: how often a body's centre turns back within a second,
//! and how far it goes from rest. A playback beyond them is found by [`check`](crate::touch_limits::check); [`within_limits`](crate::touch_limits::within_limits) weakens only the
//! top layer's drags until the whole playback passes.
//!
//! # MODEL — reversals with hysteresis and a displacement bound on body centres, and weakening by scaling drags
//!
//! **Equations.** A body's centre is the mean of its particles `c = (1/N) Σ xᵢ`
//! ([`crate::touch_layers::TouchBodies::centre`]), its rest centre the mean of its rest shape. Per axis, a reversal
//! is counted when the centre, having moved one way, comes back at least `min_swing` from the last extreme
//! (`Swing::feed`: hysteresis, so a jitter smaller than `min_swing` is not counted); the first direction is taken
//! when the centre first moves `min_swing` from rest. The count in a window is the most reversals whose steps lie
//! within `steps_per_second` of each other (`Swing::worst_window`); a body violates the rate when that count on any
//! axis exceeds `max_reversals_per_second`, and the distance when `|c − c_rest|² > d_max²` at any step while it is
//! active ([`check`](crate::touch_limits::check)). Weakening ([`within_limits`](crate::touch_limits::within_limits)): every drag `to` of the top layer becomes `a + s (to − a)`, `a`
//! the grab point of its span (`scaled`), for `s` in 3/4, 1/2, 3/8, 1/4, 1/8, 1/16, 1/32, the first `s` at which the
//! whole layered playback passes; the lower layers are never changed.
//!
//! **Units.** Positions, `min_swing` and `d_max` m; steps counts, `steps_per_second` 1/s; `s` dimensionless. Every
//! quantity is `Fix128` (Q64.64).
//!
//! **Conservation.** Not applicable: the limits observe a playback and change no state; weakening changes the
//! input (the top layer's drags), not the law.
//!
//! **Range.** `steps_per_second ≥ 1`, `min_swing ≥ 0`, `d_max ≥ 0`, else [`crate::touch_layers::LayerError::BadLimits`]
//! ([`TouchLimits::new`](crate::touch_limits::TouchLimits::new)). With `min_swing = 0` every change of direction counts. Only active bodies are observed
//! (a body at rest is at its rest centre). When the lower layers alone are beyond the limits, or 1/32 does not pass,
//! [`within_limits`](crate::touch_limits::within_limits) returns `None` (weakening the top layer cannot help). A drag outside any span (no grab before
//! it) is left as it is.
//!
//! **Approximations.** The centre stands for the body (a body that deforms without moving its centre is not
//! counted); reversals are per axis, not along the path; the rate window is a sliding count of reversal steps;
//! weakening is a fixed ladder, not a search for the largest passing scale; scaled drags go back on the 2⁻¹⁶ grid.
//!
//! **Validation.** `tests/analytic_touch_limits.rs::a_shake_faster_than_the_limit_is_found_and_weakened_until_it_passes`
//! `::a_pull_beyond_the_distance_is_weakened_until_it_passes`
//! `::a_safe_log_passes_unchanged_at_scale_one`
//! `::a_lower_layer_beyond_the_limits_cannot_be_fixed_by_weakening_the_top`
//! `::weakening_leaves_a_lower_layer_on_the_same_body_at_full_strength`
//! `::a_jitter_under_the_swing_is_not_a_reversal`
//! `::reversals_are_counted_in_any_one_second_window`
//! `::limits_out_of_range_are_refused`
//!
//! **Tiers.** L0 only.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::cloth::Cloth;
use crate::input_log::InputLog;
use crate::math::Fix128;
use crate::touch::{Hand, Touch};
use crate::touch_layers::{mean, replay_with, LayerError, PlaybackConfig};

/// How a playback may move its bodies
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TouchLimits {
    max_reversals_per_second: usize,
    max_displacement: Fix128,
    min_swing: Fix128,
    steps_per_second: u64,
}

impl TouchLimits {
    /// At most `max_reversals_per_second` reversals of a body centre per axis in any `steps_per_second` steps (a
    /// reversal counts once the centre comes back `min_swing` m from its last extreme), and no centre farther than
    /// `max_displacement` m from rest.
    ///
    /// # Errors
    ///
    /// [`LayerError::BadLimits`] for `steps_per_second = 0` or a negative distance or swing.
    pub fn new(
        max_reversals_per_second: usize,
        max_displacement: Fix128,
        min_swing: Fix128,
        steps_per_second: u64,
    ) -> Result<Self, LayerError> {
        if steps_per_second == 0 || max_displacement.is_negative() || min_swing.is_negative() {
            return Err(LayerError::BadLimits);
        }
        Ok(Self {
            max_reversals_per_second,
            max_displacement,
            min_swing,
            steps_per_second,
        })
    }
}

/// What a playback did beyond the limits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Violation {
    /// A body's centre turned back too often within one second
    TooFast {
        /// The body
        body: usize,
        /// Reversals in its worst one-second window, on its worst axis
        reversals: usize,
    },
    /// A body's centre went too far from rest
    TooFar {
        /// The body
        body: usize,
    },
}

/// One axis of one body's centre: the steps of its reversals
struct Swing {
    extreme: Fix128,
    rising: Option<bool>,
    reversals: Vec<u64>,
}

impl Swing {
    fn new(x: Fix128) -> Self {
        Self {
            extreme: x,
            rising: None,
            reversals: Vec::new(),
        }
    }

    fn feed(&mut self, step: u64, x: Fix128, min_swing: Fix128) {
        match self.rising {
            None => {
                if (x - self.extreme).abs() >= min_swing && x != self.extreme {
                    self.rising = Some(x > self.extreme);
                    self.extreme = x;
                }
            }
            Some(true) if x > self.extreme => self.extreme = x,
            Some(false) if x < self.extreme => self.extreme = x,
            Some(up) => {
                if (self.extreme - x).abs() >= min_swing && x != self.extreme {
                    self.reversals.push(step);
                    self.rising = Some(!up);
                    self.extreme = x;
                }
            }
        }
    }

    fn worst_window(&self, steps_per_second: u64) -> usize {
        let mut worst = 0;
        let mut first = 0;
        for (i, &s) in self.reversals.iter().enumerate() {
            while s - self.reversals[first] >= steps_per_second {
                first += 1;
            }
            worst = worst.max(i + 1 - first);
        }
        worst
    }
}

/// Whether a centre `d` from rest is farther than the reach (`|d|² > d_max²`; exactly `d_max` is within)
fn beyond_reach(d: crate::math::Vec3Fix, reach_squared: Fix128) -> bool {
    d.dot(d) > reach_squared
}

/// Every way the layered playback of `layers` (lowest first, the top last) on `cloths` goes beyond `limits`, body
/// by body (empty when it is within them).
///
/// # Errors
///
/// As [`crate::touch_layers::replay`].
pub fn check(
    cloths: &[Cloth],
    hand: &Hand,
    layers: &[InputLog<Touch>],
    config: PlaybackConfig,
    limits: &TouchLimits,
) -> Result<Vec<Violation>, LayerError> {
    let rest: Vec<_> = cloths
        .iter()
        .map(|c| mean(c.rest_positions()).unwrap_or(crate::math::Vec3Fix::ZERO))
        .collect();
    let mut swings: Vec<[Swing; 3]> = rest
        .iter()
        .map(|r| [Swing::new(r.x), Swing::new(r.y), Swing::new(r.z)])
        .collect();
    let mut far = vec![false; cloths.len()];
    let reach_squared = limits.max_displacement * limits.max_displacement;
    replay_with(cloths, hand, layers, config, |step, bodies| {
        for &b in bodies.active() {
            let Some(c) = bodies.centre(b) else { continue };
            swings[b][0].feed(step, c.x, limits.min_swing);
            swings[b][1].feed(step, c.y, limits.min_swing);
            swings[b][2].feed(step, c.z, limits.min_swing);
            if beyond_reach(c - rest[b], reach_squared) {
                far[b] = true;
            }
        }
    })?;
    let mut out = Vec::new();
    for (b, axes) in swings.iter().enumerate() {
        let reversals = axes
            .iter()
            .map(|s| s.worst_window(limits.steps_per_second))
            .max()
            .unwrap_or(0);
        if reversals > limits.max_reversals_per_second {
            out.push(Violation::TooFast { body: b, reversals });
        }
        if far[b] {
            out.push(Violation::TooFar { body: b });
        }
    }
    Ok(out)
}

/// `log` with every drag brought toward the point its span was grabbed at by `scale`
fn scaled(log: &InputLog<Touch>, scale: Fix128) -> InputLog<Touch> {
    let mut out = InputLog::new();
    let mut anchor = None;
    for (step, touch) in log.entries() {
        let t = match *touch {
            Touch::Grab { at, .. } => {
                anchor = Some(at);
                *touch
            }
            Touch::Drag { to } => match anchor {
                Some(a) => Touch::drag(a + (to - a) * scale),
                None => *touch,
            },
            Touch::Release => {
                anchor = None;
                *touch
            }
        };
        // the steps are those of a log already in order, so recording cannot fail
        let _ = out.record(*step, t);
    }
    out
}

/// The scales tried, strongest first
const LADDER: [(i64, i64); 7] = [(3, 4), (1, 2), (3, 8), (1, 4), (1, 8), (1, 16), (1, 32)];

/// The top layer of `layers` (the last) weakened until the whole playback is within `limits`: every top-layer drag
/// is brought toward its span's grab point by 3/4, then 1/2, 3/8, 1/4, 1/8, 1/16, 1/32, the lower layers left at
/// full strength. Returns the top layer to send and the scale used (1 when it already passes), or `None` when the
/// lower layers alone go beyond the limits or 1/32 does not pass.
///
/// # Errors
///
/// As [`check`].
pub fn within_limits(
    cloths: &[Cloth],
    hand: &Hand,
    layers: &[InputLog<Touch>],
    config: PlaybackConfig,
    limits: &TouchLimits,
) -> Result<Option<(InputLog<Touch>, Fix128)>, LayerError> {
    let mut all = layers.to_vec();
    let top = all.pop().unwrap_or_default();
    all.push(top.clone());
    if check(cloths, hand, &all, config, limits)?.is_empty() {
        return Ok(Some((top, Fix128::ONE)));
    }
    if let Some(last) = all.last_mut() {
        *last = InputLog::new();
    }
    if !check(cloths, hand, &all, config, limits)?.is_empty() {
        return Ok(None);
    }
    for (num, den) in LADDER {
        let s = Fix128::from_ratio(num, den);
        let weaker = scaled(&top, s);
        if let Some(last) = all.last_mut() {
            last.clone_from(&weaker);
        }
        if check(cloths, hand, &all, config, limits)?.is_empty() {
            return Ok(Some((weaker, s)));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    //! The two thresholds at their boundaries: a swing of exactly `min_swing` counts and one ulp less does not
    //! (the first direction and a reversal alike); a centre exactly `d_max` from rest is within and one ulp farther
    //! is beyond. Oracle: the MODEL section's `≥ min_swing` and `|c − c_rest|² > d_max²`, with values chosen so
    //! every product is exact in `Fix128`.

    use super::{beyond_reach, Swing};
    use crate::math::{Fix128, Vec3Fix};

    const ULP: Fix128 = Fix128::from_raw(0, 1);

    /// The steps of the reversals of a centre going 0 → `up` → `up − back` with hysteresis `min_swing`
    fn reversals(up: Fix128, back: Fix128, min_swing: Fix128) -> Vec<u64> {
        let mut s = Swing::new(Fix128::ZERO);
        s.feed(1, up, min_swing);
        s.feed(2, up - back, min_swing);
        s.reversals
    }

    #[test]
    fn a_swing_of_exactly_min_swing_counts_and_one_ulp_less_does_not() {
        let m = Fix128::from_ratio(1, 32);
        // a reversal: out by m, back by exactly m / by m − ulp
        assert_eq!(reversals(m, m, m), vec![2]);
        assert_eq!(reversals(m, m - ULP, m), Vec::<u64>::new());
        // the first direction: out by exactly m is taken (so the way back counts), by m − ulp is not
        assert_eq!(reversals(m - ULP, m, m), Vec::<u64>::new());
        let mut s = Swing::new(Fix128::ZERO);
        s.feed(1, m, m);
        assert_eq!(s.rising, Some(true));
        let mut s = Swing::new(Fix128::ZERO);
        s.feed(1, m - ULP, m);
        assert_eq!(s.rising, None);
    }

    #[test]
    fn a_centre_exactly_d_max_from_rest_is_within_and_one_ulp_farther_is_beyond() {
        let d_max = Fix128::from_ratio(1, 2);
        let reach_squared = d_max * d_max;
        let at = |x: Fix128| Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO);
        assert!(!beyond_reach(at(d_max), reach_squared));
        assert!(!beyond_reach(at(-d_max), reach_squared));
        assert!(beyond_reach(at(d_max + ULP), reach_squared));
        assert!(!beyond_reach(Vec3Fix::ZERO, Fix128::ZERO));
        // 2⁻³² squared is one ulp (2⁻⁶⁴), the least distance the square can tell from zero
        assert!(beyond_reach(at(Fix128::from_raw(0, 1 << 32)), Fix128::ZERO));
    }
}
