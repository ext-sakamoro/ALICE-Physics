//! The metric a distance is measured in, in fixed point.
//!
//! The simulation is Euclidean by default and stays that way bit for bit.
//! What this module adds is the ability to say that some distance — the
//! surface of a field, the clearance a broadphase pads with — was measured
//! in a *different* norm, and to convert that statement into the Euclidean
//! numbers the collision pipeline works in.
//!
//! A metric here is a non-negative combination of the three basis norms,
//! `g(v) = w₁‖v‖₁ + w₂‖v‖₂ + w∞‖v‖∞`. Negative weights dent the unit ball
//! inwards, the triangle inequality fails, and it is no longer a metric, so
//! [`MetricWeights::new`] refuses them.
//!
//! # Two different numbers
//!
//! Conflating these is the mistake this module exists to prevent.
//!
//! * [`axis_extent`](MetricWeights::axis_extent) — how wide the ball
//!   `{x : g(x) ≤ r}` is along an axis. Exactly `r / (w₁ + w₂ + w∞)`: every
//!   basis norm satisfies `N(h) ≥ (w₁+w₂+w∞)·|hₓ|`, so `|hₓ|/g(h) ≤ 1/Σw`
//!   with equality at `h = e₁`. A cube-metric ball of radius `r` *is* the
//!   cube `[−r, r]³`; its box is not grown at all.
//! * [`euclidean_radius`](MetricWeights::euclidean_radius) — how far a
//!   *clearance* of `r`, measured in this metric, can reach in Euclidean
//!   space. `r / min`, which is `√3·r` for the cube metric. A broadphase
//!   margin is a clearance, so this is the one it needs; padding by the
//!   plain `r` there lets pairs through.
//!
//! # Closed forms
//!
//! Both extremes of `g` over the Euclidean unit sphere are exact. `g` is
//! positively homogeneous, and sign / permutation symmetry reduce `g − w₂`
//! to the linear functional `⟨u, h⟩` with `u = (w₁+w∞, w₁, w₁)` over the
//! descending non-negative cone:
//!
//! ```text
//! lipschitz = ‖u‖₂ + w₂ = √((w₁ + w∞)² + 2w₁²) + w₂
//! minimum   = min_{k∈{1,2,3}} (k·w₁ + w∞)/√k + w₂
//! ```
//!
//! This is a fixed-point port of `alice_det_math::metric`, which derives the
//! same forms in `f32` and checks them against a brute-force sweep;
//! `tests/analytic_metric_broadphase.rs` pins the port against that crate so
//! the two cannot drift.

use crate::math::{Fix128, Vec3Fix};

/// √2 in fixed point, to the precision `Fix128` carries.
fn sqrt_2() -> Fix128 {
    Fix128::from_int(2).sqrt()
}

/// √3 in fixed point, to the precision `Fix128` carries.
fn sqrt_3() -> Fix128 {
    Fix128::from_int(3).sqrt()
}

/// Why a weight triple is not a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetricError {
    /// A weight was negative: the unit ball stops being convex and the
    /// triangle inequality fails.
    NotConvex,
    /// Every weight was zero: there is no norm to speak of.
    Degenerate,
}

impl core::fmt::Display for MetricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::NotConvex => "metric weight is negative (unit ball not convex)",
            Self::Degenerate => "all metric weights are zero",
        };
        f.write_str(s)
    }
}

/// A norm built as a non-negative combination of `‖·‖₁`, `‖·‖₂`, `‖·‖∞`.
///
/// The weights are private so that a triple which is not a metric cannot be
/// constructed; [`new`](Self::new) is the only way in besides the three
/// basis constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricWeights {
    l1: Fix128,
    l2: Fix128,
    linf: Fix128,
}

impl Default for MetricWeights {
    /// Euclidean — the only default that leaves the simulation unchanged.
    fn default() -> Self {
        Self::L2
    }
}

impl MetricWeights {
    /// Pure `‖·‖₁`: the unit ball is an octahedron.
    pub const L1: Self = Self {
        l1: Fix128::ONE,
        l2: Fix128::ZERO,
        linf: Fix128::ZERO,
    };
    /// Pure `‖·‖₂`: the unit ball is a sphere; the simulation is unchanged.
    pub const L2: Self = Self {
        l1: Fix128::ZERO,
        l2: Fix128::ONE,
        linf: Fix128::ZERO,
    };
    /// Pure `‖·‖∞`: the unit ball is a cube.
    pub const LINF: Self = Self {
        l1: Fix128::ZERO,
        l2: Fix128::ZERO,
        linf: Fix128::ONE,
    };

    /// Builds a metric from three weights.
    ///
    /// # Errors
    ///
    /// [`MetricError::NotConvex`] if any weight is negative,
    /// [`MetricError::Degenerate`] if all three are zero.
    pub fn new(l1: Fix128, l2: Fix128, linf: Fix128) -> Result<Self, MetricError> {
        if l1 < Fix128::ZERO || l2 < Fix128::ZERO || linf < Fix128::ZERO {
            return Err(MetricError::NotConvex);
        }
        if l1 + l2 + linf <= Fix128::ZERO {
            return Err(MetricError::Degenerate);
        }
        Ok(Self { l1, l2, linf })
    }

    /// The three weights, in the order `(l1, l2, linf)`.
    #[must_use]
    pub const fn weights(self) -> (Fix128, Fix128, Fix128) {
        (self.l1, self.l2, self.linf)
    }

    /// True when this is the plain Euclidean metric.
    ///
    /// Every metric-aware path checks this first and returns its input
    /// untouched, so enabling the feature cannot move a bit in a Euclidean
    /// simulation.
    #[must_use]
    pub fn is_euclidean(self) -> bool {
        self.l1 == Fix128::ZERO && self.linf == Fix128::ZERO && self.l2 == Fix128::ONE
    }

    /// `g(v) = w₁‖v‖₁ + w₂‖v‖₂ + w∞‖v‖∞`, summed left to right.
    #[must_use]
    pub fn norm(self, v: Vec3Fix) -> Fix128 {
        let ax = v.x.abs();
        let ay = v.y.abs();
        let az = v.z.abs();
        let l1 = ax + ay + az;
        let l2 = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        let m = if ax > ay { ax } else { ay };
        let linf = if m > az { m } else { az };
        self.l1 * l1 + self.l2 * l2 + self.linf * linf
    }

    /// `max_{‖h‖₂=1} g(h)` — the most the metric can report for a unit
    /// Euclidean step.
    #[must_use]
    pub fn lipschitz(self) -> Fix128 {
        let s = self.l1 + self.linf;
        (s * s + Fix128::from_int(2) * self.l1 * self.l1).sqrt() + self.l2
    }

    /// `min_{‖h‖₂=1} g(h)` — the least it can report.
    ///
    /// Strictly positive for any value built by [`new`](Self::new).
    #[must_use]
    pub fn minimum(self) -> Fix128 {
        let k1 = self.l1 + self.linf;
        let k2 = (Fix128::from_int(2) * self.l1 + self.linf) / sqrt_2();
        let k3 = (Fix128::from_int(3) * self.l1 + self.linf) / sqrt_3();
        let m = if k1 < k2 { k1 } else { k2 };
        let m = if m < k3 { m } else { k3 };
        m + self.l2
    }

    /// The Euclidean radius a clearance of `r`, measured in this metric, can
    /// reach — `r / minimum()`, and `√3·r` for [`LINF`](Self::LINF).
    ///
    /// Returns `r` unchanged for the Euclidean metric, bit for bit.
    #[must_use]
    pub fn euclidean_radius(self, r: Fix128) -> Fix128 {
        if self.is_euclidean() {
            return r;
        }
        r / self.minimum()
    }

    /// The half-width along each axis of the ball `{x : g(x) ≤ r}` —
    /// `r / (w₁ + w₂ + w∞)`, exactly.
    ///
    /// Returns `r` unchanged for the Euclidean metric, bit for bit.
    #[must_use]
    pub fn axis_extent(self, r: Fix128) -> Fix128 {
        if self.is_euclidean() {
            return r;
        }
        r / (self.l1 + self.l2 + self.linf)
    }
}
