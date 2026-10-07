//! A homogeneous medium that exchanges momentum with rigid bodies through a
//! linear drag, as a [`Participant`] of the world's substep loop.
//!
//! [`DragMedium`] owns a medium of mass `M` moving with one velocity `u`
//! (a well-stirred fluid, an air mass, a conveyor of granular material seen as
//! a whole) and a list of coupled bodies with a drag coefficient `c_i ≥ 0`
//! each. In every substep of width `h` it stages on each coupled dynamic body
//! `i` the drag force
//!
//! ```text
//! F_i = c_i (u − v_i)
//! ```
//!
//! and takes the equal and opposite momentum itself. Both sides use the
//! velocities at the start of the substep, so the exchange is the explicit
//! (forward Euler) step of
//!
//! ```text
//! m_i dv_i/dt = c_i (u − v_i),      M du/dt = −Σ_i c_i (u − v_i).
//! ```
//!
//! For one body the relative velocity `w = v − u` then follows
//! `w_{n+1} = (1 − c h (1/m + 1/M)) w_n`, the discrete form of
//! `w(t) = w(0) e^{−c (1/m + 1/M) t}`. The explicit step is stable while
//! `c_i h (1/m_i + 1/M) < 2` for every body; it keeps every velocity inside
//! the range spanned by the initial velocities (no overshoot) while
//! `c_i h / m_i ≤ 1` and `Σ_i c_i h / M ≤ 1`. These are conditions on the
//! world's substep width, which the participant does not see before the
//! first step, so they are documented, not checked.
//!
//! # Momentum bookkeeping
//!
//! The medium holds its momentum `P = M u`, not `u`: `u = P / M` is derived
//! from it in every substep. The reaction it takes for body `i` is computed
//! with the same factorisation the world uses to apply the staged force,
//!
//! ```text
//! dv_i = (F_i · inv_mass_i) · h        (the world: v_i += dv_i)
//! P   -= dv_i / inv_mass_i
//! ```
//!
//! so the momentum the body receives and the momentum the medium gives up
//! are the same number. With `inv_mass_i = 2^(−k_i)`, `k_i ≥ 0` (masses `1`,
//! `2`, `4`, …) the division is exact, and `Σ_i v_i / inv_mass_i + P` is the
//! same bit pattern before and after the exchange. `M` can be any positive
//! value: it only enters `u`, never the bookkeeping.
//!
//! Other integer masses `m` are exact while the change stays small. The
//! world's `inv_mass = 1 / m` is truncated to `(2⁶⁴ − r) / m · 2⁻⁶⁴` with
//! `r = 2⁶⁴ mod m` (`r = 1` for `m = 3, 5`, `4` for `6`, `2` for `7`, `0` for
//! a power of two), and the [`Fix128`] division truncates as well, so for a
//! component `dv = D · 2⁻⁶⁴` of one substep's change the reaction is
//!
//! ```text
//! dv / inv_mass = (D m + ⌊D m r / (2⁶⁴ − r)⌋) · 2⁻⁶⁴,
//! ```
//!
//! which is the momentum the body receives, `m · dv`, exactly when
//! `(D m + 1) r < 2⁶⁴`, i.e. about `|dv| · m · r < 1` per component and
//! substep (`|dv| < 1/3` for `m = 3`, `< 1/24` for `m = 6`). Above it the
//! medium books `⌊D m r / (2⁶⁴ − r)⌋` raw units too many per component and
//! substep, and `P + Σ m_i v_i` drifts by that much. Measured: mass 3,
//! `c = 3`, relative velocity 100, `M = 4`, `dt = 1/60` with one substep
//! (`|dv| ≈ 5/3`) drifts by 4 raw units per frame; masses 5, 6, 7 with
//! `c = 1` and relative velocity 1 stay exact.
//!
//! Whether that sum stays the same bit pattern over a whole step also depends
//! on what the world does with the body velocities after the exchange:
//!
//! * **TGS** keeps a free body's velocity as it was integrated: the sum is
//!   conserved bit for bit in every substep (measured for `dt` `1/64` and
//!   `1/60`, `3`, `4` and `8` substeps).
//! * **XPBD** derives every body velocity again from the position change,
//!   `v ← ((x + v·h) − x) · (1/h)`, which drops low bits of `v` even for a
//!   body no constraint touched. That is momentum the world loses, not the
//!   medium; per body and substep it is at most
//!   `2⁻⁶⁴ (1/h + |v| h + 2)` per component (truncation of `v·h`, rounding
//!   of `1/h` and of the product), and the sum over a run stays within
//!   `n · Σ_i m_i · 2⁻⁶⁴ (1/h + V h + 2)` for `n` substeps and `|v| ≤ V`. The
//!   sum is then conserved to that bound, measured at about `1e-13` over
//!   240 frames; it becomes bit-exact when XPBD keeps the predicted velocity
//!   of the bodies its constraints did not move.
//!
//! # Assumptions
//!
//! The bookkeeping above is exact only under these conditions; outside them
//! the medium still applies the drag, but the sum may change.
//!
//! * **No other force on the coupled bodies.** The world adds the forces of
//!   every participant before it multiplies by `inv_mass · h`; with a second
//!   participant pushing the same body the rounding of the sum differs from
//!   the rounding of the medium's share.
//! * **Masses.** Powers of two `≥ 1`, or other integer masses under the
//!   condition above.
//! * **No damping, no gravity along the measured axis.** Frame damping and
//!   gravity change the body momentum by themselves; a contact changes it
//!   along its normal.
//! * **Range.** A product or sum out of the range of [`Fix128`] is a fault
//!   ([`ParticipantFault::OutOfRange`]), checked before anything is staged,
//!   including the body's new velocity `v_i + dv_i`, so the world never
//!   refuses a force the medium already paid for.
//!
//! Static and kinematic bodies, and a dynamic body with `inv_mass = 0`, get
//! no force and give no reaction: they are skipped.
//!
//! # Sleeping bodies
//!
//! Sleeping does not need to be turned off. The world wakes a sleeping or
//! parked body whenever the change a participant force makes to it is
//! non-zero ([`crate::world_participant::wakes_parked_body`]), so every drag
//! force the medium books reaches its body. A body moving with the medium
//! (`v_i = u`, drag exactly zero) falls asleep as without the medium; a body
//! that still differs from `u` by any amount the drag resolves stays awake.
//! The cost is that coupled bodies relaxing toward `u` keep stepping instead
//! of sleeping. Measured in release (1000 free bodies of masses 1 to 8 all
//! coupled, `M = 1000`, `c = 1/2`, `dt = 1/60`, 4 substeps, 300 frames after
//! the first sleep, best of 5, three runs on a loaded machine): the bodies
//! fall asleep once every 60 frames and are woken in the next substep
//! (asleep in 1.7 % of the frames), and the step costs as much as with sleep
//! turned off, XPBD 4.5 / 4.4, 10.4 / 7.3 and 12.7 / 10.4 ms per frame (sleep
//! on / off, the waking of all parked bodies every 60 frames costs 2–43 %),
//! TGS 5.3 / 5.1, 6.1 / 5.8 and 5.5 / 5.9 ms. The same bodies without the
//! medium sleep in every frame and step in 0.02 ms (XPBD) and 0.5 ms (TGS).
//!
//! # Small medium masses
//!
//! [`DragMedium::new`] keeps `P = M · u` rounded down to a multiple of
//! `2⁻⁶⁴` (the [`Fix128`] product) and the medium moves with `P / M`, so the
//! velocity it starts with differs from `u` by less than `2⁻⁶⁴ / M + 2⁻⁶⁴`
//! per component. That is negligible for `M` of order one, but for a tiny
//! mass it is not: `M = 2⁻⁶⁴` with `u = (0.3, −0.7, 1)` gives
//! `P = (0, −2⁻⁶⁴, 2⁻⁶⁴)` and the velocity `(0, −1, 1)`. Read the starting
//! velocity back with [`DragMedium::velocity`] where it matters.
//!
//! # Snapshot payload
//!
//! Little endian, 60 bytes: `version: u32` (1), `digest: u64`, then the
//! medium momentum `P` as three [`Fix128`] (`hi: i64`, `lo: u64`). The digest
//! is FNV-1a 64 over the configuration (the mass `M`, then every coupling's
//! body index and coefficient in order); [`Participant::check_state`] refuses
//! a payload of another length ([`StateError::Length`]), version or
//! configuration ([`StateError::InvalidValue`]). Only `P` is state.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Snapshot tag of [`DragMedium`]: the ASCII code `MEDM`, big endian.
pub const DRAG_MEDIUM_KIND: ParticipantKind = ParticipantKind::new(u32::from_be_bytes(*b"MEDM"));

/// Observation channel of [`DragMedium`]: `x` of the medium velocity `P / M`
/// (`y` and `z` follow as `+1`, `+2`). Not reported when the quotient is out
/// of range.
pub const MEDIUM_OBS_VELOCITY: u32 = 0;

/// Observation channel of [`DragMedium`]: `x` of the medium momentum `P`
/// (`y` and `z` follow as `+1`, `+2`).
pub const MEDIUM_OBS_MOMENTUM: u32 = 3;

/// Payload layout version written by [`DragMedium`].
const MEDIUM_STATE_VERSION: u32 = 1;

/// `version: u32`, `digest: u64`, three `Fix128`.
const MEDIUM_STATE_LEN: usize = 4 + 8 + 3 * 16;

/// Why a [`DragMedium`] or one of its couplings was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DragMediumError {
    /// The medium mass is zero or negative.
    NonPositiveMass,
    /// The drag coefficient of `body` is negative.
    NegativeCoefficient {
        /// Body index of the refused coupling.
        body: usize,
    },
    /// `body` is already coupled (one coupling per body keeps the reaction
    /// of each force exactly the momentum the body receives).
    DuplicateBody {
        /// Body index of the refused coupling.
        body: usize,
    },
    /// `M · u` is out of the range of [`Fix128`].
    MomentumOutOfRange,
}

impl core::fmt::Display for DragMediumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveMass => write!(f, "medium mass must be positive"),
            Self::NegativeCoefficient { body } => {
                write!(f, "drag coefficient of body {body} is negative")
            }
            Self::DuplicateBody { body } => write!(f, "body {body} is already coupled"),
            Self::MomentumOutOfRange => write!(f, "medium momentum out of range"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DragMediumError {}

/// One coupled body and its drag coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DragCoupling {
    /// Body index in the world.
    pub body: usize,
    /// Drag coefficient `c ≥ 0` (force per relative velocity, kg/s).
    pub coefficient: Fix128,
}

/// A homogeneous medium coupled to rigid bodies by linear drag; see the
/// module documentation.
#[derive(Clone, Debug)]
pub struct DragMedium {
    mass: Fix128,
    momentum: Vec3Fix,
    couplings: Vec<DragCoupling>,
    digest: u64,
}

fn wide(f: Fix128) -> i128 {
    (i128::from(f.hi) << 64) | i128::from(f.lo)
}

fn narrow(raw: i128) -> Fix128 {
    Fix128 {
        hi: (raw >> 64) as i64,
        lo: raw as u64,
    }
}

fn checked_add(a: Fix128, b: Fix128) -> Option<Fix128> {
    wide(a).checked_add(wide(b)).map(narrow)
}

fn checked_sub(a: Fix128, b: Fix128) -> Option<Fix128> {
    wide(a).checked_sub(wide(b)).map(narrow)
}

fn checked_add_vec(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        checked_add(a.x, b.x)?,
        checked_add(a.y, b.y)?,
        checked_add(a.z, b.z)?,
    ))
}

fn checked_sub_vec(a: Vec3Fix, b: Vec3Fix) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        checked_sub(a.x, b.x)?,
        checked_sub(a.y, b.y)?,
        checked_sub(a.z, b.z)?,
    ))
}

/// `a / b`, or `None` for `b = 0` or a quotient out of range. The division
/// itself ([`Fix128`]'s `/`) does not report a wrap, so the quotient is
/// multiplied back: a quotient in range reproduces `a` to within `|b|·2⁻⁶⁴`
/// plus the rounding of the product, a wrapped one misses it by far more.
fn checked_quotient(a: Fix128, b: Fix128) -> Option<Fix128> {
    if b.is_zero() {
        return None;
    }
    let q = a / b;
    let back = q.checked_mul(b)?;
    let miss = wide(back).checked_sub(wide(a))?.unsigned_abs();
    let allowed = (wide(b).unsigned_abs() >> 64) + 2;
    (miss <= allowed).then_some(q)
}

fn checked_quotient_vec(a: Vec3Fix, b: Fix128) -> Option<Vec3Fix> {
    Some(Vec3Fix::new(
        checked_quotient(a.x, b)?,
        checked_quotient(a.y, b)?,
        checked_quotient(a.z, b)?,
    ))
}

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv_fix(h: u64, f: Fix128) -> u64 {
    fnv(fnv(h, &f.hi.to_le_bytes()), &f.lo.to_le_bytes())
}

fn digest(mass: Fix128, couplings: &[DragCoupling]) -> u64 {
    let mut h = fnv_fix(0xcbf2_9ce4_8422_2325, mass);
    for c in couplings {
        h = fnv(h, &(c.body as u64).to_le_bytes());
        h = fnv_fix(h, c.coefficient);
    }
    h
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn get_fix(bytes: &[u8], at: usize) -> Fix128 {
    let mut hi = [0u8; 8];
    let mut lo = [0u8; 8];
    hi.copy_from_slice(&bytes[at..at + 8]);
    lo.copy_from_slice(&bytes[at + 8..at + 16]);
    Fix128 {
        hi: i64::from_le_bytes(hi),
        lo: u64::from_le_bytes(lo),
    }
}

impl DragMedium {
    /// A medium of mass `mass` moving with `velocity`, coupled to no body.
    /// The state is the momentum `mass · velocity` rounded to [`Fix128`], so
    /// for a very small `mass` the medium's velocity ([`Self::velocity`]) can
    /// differ from `velocity` (module documentation, small medium masses).
    ///
    /// # Errors
    ///
    /// [`DragMediumError::NonPositiveMass`] for `mass ≤ 0`,
    /// [`DragMediumError::MomentumOutOfRange`] when `mass · velocity` is out
    /// of range.
    pub fn new(mass: Fix128, velocity: Vec3Fix) -> Result<Self, DragMediumError> {
        if mass <= Fix128::ZERO {
            return Err(DragMediumError::NonPositiveMass);
        }
        let momentum = velocity
            .checked_scale(mass)
            .ok_or(DragMediumError::MomentumOutOfRange)?;
        Ok(Self {
            mass,
            momentum,
            couplings: Vec::new(),
            digest: digest(mass, &[]),
        })
    }

    /// Couple body `body` with drag coefficient `coefficient`. Couplings are
    /// applied in the order they were added. Whether `body` exists is only
    /// known in the world: a missing body is a fault at the first substep
    /// ([`ParticipantFault::InvalidState`]).
    ///
    /// # Errors
    ///
    /// [`DragMediumError::NegativeCoefficient`] for `coefficient < 0`,
    /// [`DragMediumError::DuplicateBody`] for a body already coupled. The
    /// medium is unchanged.
    pub fn couple(&mut self, body: usize, coefficient: Fix128) -> Result<(), DragMediumError> {
        if coefficient < Fix128::ZERO {
            return Err(DragMediumError::NegativeCoefficient { body });
        }
        if self.couplings.iter().any(|c| c.body == body) {
            return Err(DragMediumError::DuplicateBody { body });
        }
        self.couplings.push(DragCoupling { body, coefficient });
        self.digest = digest(self.mass, &self.couplings);
        Ok(())
    }

    /// The medium mass `M`.
    #[must_use]
    pub const fn mass(&self) -> Fix128 {
        self.mass
    }

    /// The medium momentum `P` (the state).
    #[must_use]
    pub const fn momentum(&self) -> Vec3Fix {
        self.momentum
    }

    /// The medium velocity `P / M`, `None` when out of range.
    #[must_use]
    pub fn velocity(&self) -> Option<Vec3Fix> {
        checked_quotient_vec(self.momentum, self.mass)
    }

    /// The couplings, in the order they are applied.
    #[must_use]
    pub fn couplings(&self) -> &[DragCoupling] {
        &self.couplings
    }

    /// `P + Σ v_i / inv_mass_i` over the coupled bodies the medium exchanges
    /// momentum with (dynamic, `inv_mass ≠ 0`), the quantity the module
    /// documentation conserves. `None` for a coupled index past `bodies` or a
    /// value out of range.
    #[must_use]
    pub fn total_momentum(&self, bodies: &[RigidBody]) -> Option<Vec3Fix> {
        let mut total = self.momentum;
        for c in &self.couplings {
            let body = bodies.get(c.body)?;
            if !body.is_dynamic() || body.inv_mass.is_zero() {
                continue;
            }
            total = checked_add_vec(total, checked_quotient_vec(body.velocity, body.inv_mass)?)?;
        }
        Some(total)
    }

    /// The medium momentum after one exchange of width `h` with `bodies`,
    /// staging the drag forces through `stage`. Nothing of `self` changes.
    fn exchange(
        &self,
        bodies: &[RigidBody],
        h: Fix128,
        mut stage: impl FnMut(usize, Vec3Fix) -> Result<(), ParticipantFault>,
    ) -> Result<Vec3Fix, ParticipantFault> {
        let oor = ParticipantFault::OutOfRange;
        let u = checked_quotient_vec(self.momentum, self.mass).ok_or(oor)?;
        let mut momentum = self.momentum;
        let mut staged: Vec<(usize, Vec3Fix)> = Vec::with_capacity(self.couplings.len());
        for c in &self.couplings {
            let body = bodies.get(c.body).ok_or(ParticipantFault::InvalidState)?;
            if !body.is_dynamic() || body.inv_mass.is_zero() {
                continue;
            }
            let force = checked_sub_vec(u, body.velocity)
                .and_then(|w| w.checked_scale(c.coefficient))
                .ok_or(oor)?;
            // The world's factorisation (`apply_participant_forces`).
            let dv = force
                .checked_scale(body.inv_mass)
                .and_then(|a| a.checked_scale(h))
                .ok_or(oor)?;
            checked_add_vec(body.velocity, dv).ok_or(oor)?;
            let given = checked_quotient_vec(dv, body.inv_mass).ok_or(oor)?;
            momentum = checked_sub_vec(momentum, given).ok_or(oor)?;
            staged.push((c.body, force));
        }
        for (body, force) in staged {
            stage(body, force)?;
        }
        Ok(momentum)
    }
}

impl Participant for DragMedium {
    fn kind(&self) -> ParticipantKind {
        DRAG_MEDIUM_KIND
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let bodies = ctx.bodies().to_vec();
        let momentum = self.exchange(&bodies, h, |body, force| {
            ctx.add_force(body, force)
                .map_err(|_| ParticipantFault::InvalidState)
        })?;
        self.momentum = momentum;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        if let Some(u) = self.velocity() {
            out.push(MEDIUM_OBS_VELOCITY, u.x);
            out.push(MEDIUM_OBS_VELOCITY + 1, u.y);
            out.push(MEDIUM_OBS_VELOCITY + 2, u.z);
        }
        out.push(MEDIUM_OBS_MOMENTUM, self.momentum.x);
        out.push(MEDIUM_OBS_MOMENTUM + 1, self.momentum.y);
        out.push(MEDIUM_OBS_MOMENTUM + 2, self.momentum.z);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MEDIUM_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        put_fix(out, self.momentum.x);
        put_fix(out, self.momentum.y);
        put_fix(out, self.momentum.z);
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() != MEDIUM_STATE_LEN {
            return Err(StateError::Length {
                expected: MEDIUM_STATE_LEN,
                found: bytes.len(),
            });
        }
        let mut version = [0u8; 4];
        version.copy_from_slice(&bytes[0..4]);
        let mut d = [0u8; 8];
        d.copy_from_slice(&bytes[4..12]);
        if u32::from_le_bytes(version) != MEDIUM_STATE_VERSION
            || u64::from_le_bytes(d) != self.digest
        {
            return Err(StateError::InvalidValue);
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.momentum = Vec3Fix::new(get_fix(bytes, 12), get_fix(bytes, 28), get_fix(bytes, 44));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotient_refuses_zero_and_wrapped_results() {
        assert_eq!(checked_quotient(Fix128::ONE, Fix128::ZERO), None);
        let tiny = Fix128::from_raw(0, 1);
        assert_eq!(checked_quotient(Fix128::from_int(1 << 40), tiny), None);
        assert_eq!(
            checked_quotient(Fix128::from_int(6), Fix128::from_int(2)),
            Some(Fix128::from_int(3))
        );
        assert_eq!(
            checked_quotient(Fix128::from_int(-6), Fix128::from_ratio(1, 4)),
            Some(Fix128::from_int(-24))
        );
    }

    #[test]
    fn digest_depends_on_mass_and_couplings() {
        let mut a = DragMedium::new(Fix128::ONE, Vec3Fix::ZERO).expect("medium");
        let b = DragMedium::new(Fix128::from_int(2), Vec3Fix::ZERO).expect("medium");
        assert_ne!(a.digest, b.digest);
        let before = a.digest;
        a.couple(0, Fix128::ONE).expect("couple");
        assert_ne!(a.digest, before);
    }
}
