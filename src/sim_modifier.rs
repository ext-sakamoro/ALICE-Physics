//! Simulation Modifier Core System
//!
//! Trait and wrapper for physics-driven SDF modifiers.
//! `ModifiedSdf` chains multiple modifiers that read simulation
//! fields (temperature, pressure, stress) and alter the SDF
//! distance at evaluation time.
//!
//! # Pattern
//!
//! Each modifier:
//! 1. Owns its simulation field data (`ScalarField3D`)
//! 2. Has `update(dt)` to advance simulation state
//! 3. `modify_distance(x, y, z, dist)` alters SDF distance
//!
//! # World participants
//!
//! [`crate::thermal::ThermalModifier`], [`crate::phase_change::PhaseChangeModifier`],
//! [`crate::pressure::PressureModifier`], [`crate::fracture::FractureModifier`]
//! and [`crate::erosion::ErosionModifier`] implement
//! [`crate::world_participant::Participant`], so they advance with world
//! time when registered with the world (or driven by
//! [`crate::world_participant::run_substep`]) and are carried in snapshots.
//!
//! * **Time.** [`crate::world_participant::StepRule::FollowSubstep`]: each
//!   substep calls `update(h)` once, with the substep width `h` converted by
//!   [`crate::math::Fix128::to_f32`] (a fixed rounding, the same on every
//!   platform). A width `h ≤ 0` never reaches the modifier: the world refuses
//!   it in the step rule check before any participant runs, so the modifier
//!   has no path of its own for it. `substep` never returns `Err`; the f32
//!   state is not checked for non-finite values.
//! * **State payload** (version 1, little endian): `version: u32`, then every
//!   field of the configuration in declaration order (`f32` as its bits,
//!   `usize` as `u64`, `ErosionType` as `u8`: `Wind` 0, `Water` 1,
//!   `Chemical` 2, `Ablation` 3), `enabled: u8` (0 or 1), each
//!   `ScalarField3D` as `nx`, `ny`, `nz` (`u64`), the min and max corners
//!   (3 `f32` each) and the cells, then the lists (`count: u64` and items):
//!   the heat sources of the thermal modifier (tag `u8` `Point` 0: x, y, z,
//!   power, radius / `Volume` 1: min, max, power) and the cracks of the
//!   fracture modifier (start, end, direction, length, `active: u8`). The
//!   configuration is part of the state: a restore brings it back. A field is
//!   rebuilt from its sizes and bounds, so its cell size is recomputed from
//!   the stored bounds. `check_state` refuses another version, a byte that
//!   is not a valid bool or tag, a size whose cell count overflows, missing
//!   and trailing bytes, without changing the modifier.
//! * **Observations** are listed on each `Participant` impl; field values are
//!   converted with [`crate::math::Fix128::from_f32`] (exact for finite
//!   values).
//! * **Not coupled yet.** The participant only advances in world time. It
//!   stages no force on any body, reads no contact or body state (pressure
//!   loads, erosion exposure and fracture stress are still applied by the caller
//!   through the inherent methods), declares no ports and exchanges no shared
//!   field (the thermal and phase-change temperatures are reconciled only
//!   through [`crate::coupled_field`]). A modifier registered with the world
//!   is owned by it, so it is not at the same time in a [`ModifiedSdf`]
//!   chain; the changed geometry does not reach the world's SDF colliders.
//!
//! # Example
//!
//! ```
//! use alice_physics::sdf_collider::ClosureSdf;
//! use alice_physics::sim_modifier::ModifiedSdf;
//! use alice_physics::thermal::{ThermalModifier, ThermalConfig};
//! use alice_physics::pressure::{PressureModifier, PressureConfig};
//! use alice_physics::sdf_collider::SdfField;
//!
//! let sphere_sdf = ClosureSdf::new(
//!     |x, y, z| (x*x + y*y + z*z).sqrt() - 1.0,
//!     |x, y, z| { let l = (x*x + y*y + z*z).sqrt().max(1e-6); (x/l, y/l, z/l) },
//! );
//! let bounds = (-2.0, -2.0, -2.0);
//! let bounds_max = (2.0, 2.0, 2.0);
//! let thermal = ThermalModifier::new(ThermalConfig::default(), 4, bounds, bounds_max);
//! let pressure = PressureModifier::new(PressureConfig::default(), 4, bounds, bounds_max);
//!
//! let modified = ModifiedSdf::new(Box::new(sphere_sdf))
//!     .with_modifier(Box::new(thermal))
//!     .with_modifier(Box::new(pressure));
//!
//! let dist = modified.distance(0.0, 0.0, 0.0);
//! ```
//!
//! Author: Moroya Sakamoto

use crate::math::Fix128;
use crate::sdf_collider::{fd_normal, SdfField, FD_NORMAL_BASE_EPS};
use crate::sim_field::ScalarField3D;
use crate::world_participant::{ObservationSink, StateError};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// PhysicsModifier Trait
// ============================================================================

/// Trait for simulation-driven SDF modifiers
pub trait PhysicsModifier: Send + Sync {
    /// Modify the SDF distance at a world-space point.
    ///
    /// Returns the modified distance. Positive offset = surface recedes
    /// (material removed). Negative offset = surface expands.
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32;

    /// Advance the simulation state by dt seconds.
    fn update(&mut self, dt: f32);

    /// Name of this modifier (for debugging)
    fn name(&self) -> &str;

    /// Whether this modifier has any active effect
    fn is_active(&self) -> bool {
        true
    }
}

// ============================================================================
// ModifiedSdf
// ============================================================================

/// SDF wrapper that applies a chain of physics modifiers.
///
/// Implements `SdfField`, so it can be used anywhere an SDF is expected.
pub struct ModifiedSdf {
    /// Original SDF field
    original: Box<dyn SdfField>,
    /// Chain of modifiers (applied in order)
    modifiers: Vec<Box<dyn PhysicsModifier>>,
    /// Epsilon for normal computation
    normal_eps: f32,
}

impl ModifiedSdf {
    /// Create a new modified SDF wrapping the original field
    #[must_use]
    pub fn new(original: Box<dyn SdfField>) -> Self {
        Self {
            original,
            modifiers: Vec::new(),
            normal_eps: FD_NORMAL_BASE_EPS,
        }
    }

    /// Add a modifier to the chain
    #[must_use]
    pub fn with_modifier(mut self, modifier: Box<dyn PhysicsModifier>) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Add a modifier (mutable)
    pub fn add_modifier(&mut self, modifier: Box<dyn PhysicsModifier>) {
        self.modifiers.push(modifier);
    }

    /// Remove all modifiers
    pub fn clear_modifiers(&mut self) {
        self.modifiers.clear();
    }

    /// Number of active modifiers
    #[must_use]
    pub fn modifier_count(&self) -> usize {
        self.modifiers.len()
    }

    /// Update all modifier simulations
    pub fn update(&mut self, dt: f32) {
        for m in &mut self.modifiers {
            m.update(dt);
        }
    }

    /// Get mutable access to a modifier by index
    #[allow(clippy::option_if_let_else)]
    pub fn modifier_mut(&mut self, index: usize) -> Option<&mut dyn PhysicsModifier> {
        match self.modifiers.get_mut(index) {
            Some(m) => Some(m.as_mut()),
            None => None,
        }
    }

    /// Evaluate the modified distance at a point
    #[inline]
    fn eval_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut d = self.original.distance(x, y, z);
        for m in &self.modifiers {
            if m.is_active() {
                d = m.modify_distance(x, y, z, d);
            }
        }
        d
    }
}

impl SdfField for ModifiedSdf {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval_distance(x, y, z)
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        fd_normal(
            |a, b, c| self.eval_distance(a, b, c),
            self.normal_eps,
            x,
            y,
            z,
        )
    }

    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let d = self.eval_distance(x, y, z);
        let n = self.normal(x, y, z);
        (d, n)
    }
}

// ============================================================================
// SingleModifiedSdf (lightweight single-modifier wrapper)
// ============================================================================

/// Lightweight SDF wrapper for a single modifier (avoids Vec overhead).
///
/// Evaluates as a [`ModifiedSdf`] holding that one modifier: the distance is
/// `modifier.modify_distance(p, original(p))` while
/// [`PhysicsModifier::is_active`] is true and `original(p)` otherwise, and
/// [`Self::update`] advances the modifier whether or not it is active.
pub struct SingleModifiedSdf<M: PhysicsModifier> {
    /// Original SDF
    pub original: Box<dyn SdfField>,
    /// The modifier
    pub modifier: M,
    /// Epsilon for normals
    normal_eps: f32,
}

impl<M: PhysicsModifier> SingleModifiedSdf<M> {
    /// Create wrapper with a single modifier
    pub fn new(original: Box<dyn SdfField>, modifier: M) -> Self {
        Self {
            original,
            modifier,
            normal_eps: FD_NORMAL_BASE_EPS,
        }
    }

    /// Update the modifier simulation
    pub fn update(&mut self, dt: f32) {
        self.modifier.update(dt);
    }

    /// The original distance, passed through the modifier when it is active
    /// (an inactive modifier is skipped, as in [`ModifiedSdf`]).
    #[inline]
    fn eval_distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let d = self.original.distance(x, y, z);
        if self.modifier.is_active() {
            self.modifier.modify_distance(x, y, z, d)
        } else {
            d
        }
    }
}

impl<M: PhysicsModifier> SdfField for SingleModifiedSdf<M> {
    #[inline]
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.eval_distance(x, y, z)
    }

    #[inline]
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        fd_normal(
            |a, b, c| self.eval_distance(a, b, c),
            self.normal_eps,
            x,
            y,
            z,
        )
    }

    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        let d = self.eval_distance(x, y, z);
        let n = self.normal(x, y, z);
        (d, n)
    }
}

// ============================================================================
// World participation (state payload and observations)
// ============================================================================

/// Version written in front of every modifier state payload.
pub(crate) const MODIFIER_STATE_VERSION: u32 = 1;

/// Appends a modifier state payload, little endian, version first.
pub(crate) struct StateWriter<'a> {
    out: &'a mut Vec<u8>,
}

impl<'a> StateWriter<'a> {
    /// Starts a payload in `out` with [`MODIFIER_STATE_VERSION`].
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        out.extend_from_slice(&MODIFIER_STATE_VERSION.to_le_bytes());
        Self { out }
    }

    pub(crate) fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub(crate) fn bool(&mut self, v: bool) {
        self.out.push(u8::from(v));
    }

    pub(crate) fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    pub(crate) fn f32(&mut self, v: f32) {
        self.out.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub(crate) fn vec3(&mut self, v: (f32, f32, f32)) {
        self.f32(v.0);
        self.f32(v.1);
        self.f32(v.2);
    }

    /// `nx`, `ny`, `nz` as u64, the bounds, then every cell.
    pub(crate) fn field(&mut self, f: &ScalarField3D) {
        self.usize(f.nx);
        self.usize(f.ny);
        self.usize(f.nz);
        self.vec3(f.min);
        self.vec3(f.max);
        for &v in &f.data {
            self.f32(v);
        }
    }
}

/// Reads a payload written by [`StateWriter`], refusing anything it cannot
/// apply. Nothing is allocated before the bytes it describes are known to be
/// present.
pub(crate) struct StateReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> StateReader<'a> {
    /// Reads the version; any other than [`MODIFIER_STATE_VERSION`] is
    /// [`StateError::InvalidValue`].
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self, StateError> {
        let mut r = Self { bytes, at: 0 };
        let mut v = [0_u8; 4];
        v.copy_from_slice(r.take(4)?);
        if u32::from_le_bytes(v) != MODIFIER_STATE_VERSION {
            return Err(StateError::InvalidValue);
        }
        Ok(r)
    }

    fn need(&self, n: usize) -> Result<(), StateError> {
        let end = self.at.checked_add(n).ok_or(StateError::InvalidValue)?;
        if end > self.bytes.len() {
            return Err(StateError::Length {
                expected: end,
                found: self.bytes.len(),
            });
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StateError> {
        self.need(n)?;
        let s = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, StateError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn bool(&mut self) -> Result<bool, StateError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(StateError::InvalidValue),
        }
    }

    pub(crate) fn u64(&mut self) -> Result<u64, StateError> {
        let mut v = [0_u8; 8];
        v.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(v))
    }

    pub(crate) fn usize(&mut self) -> Result<usize, StateError> {
        usize::try_from(self.u64()?).map_err(|_| StateError::InvalidValue)
    }

    pub(crate) fn f32(&mut self) -> Result<f32, StateError> {
        let mut v = [0_u8; 4];
        v.copy_from_slice(self.take(4)?);
        Ok(f32::from_bits(u32::from_le_bytes(v)))
    }

    pub(crate) fn vec3(&mut self) -> Result<(f32, f32, f32), StateError> {
        Ok((self.f32()?, self.f32()?, self.f32()?))
    }

    /// A count of items, each at least `min_item_bytes` long, checked against
    /// the bytes left before the caller allocates for it.
    pub(crate) fn count(&mut self, min_item_bytes: usize) -> Result<usize, StateError> {
        let n = self.usize()?;
        self.need(
            n.checked_mul(min_item_bytes)
                .ok_or(StateError::InvalidValue)?,
        )?;
        Ok(n)
    }

    /// A field written by [`StateWriter::field`], rebuilt with
    /// [`ScalarField3D::new`] (cell sizes follow from the stored bounds).
    pub(crate) fn field(&mut self) -> Result<ScalarField3D, StateError> {
        let nx = self.usize()?;
        let ny = self.usize()?;
        let nz = self.usize()?;
        let min = self.vec3()?;
        let max = self.vec3()?;
        let cells = nx
            .checked_mul(ny)
            .and_then(|c| c.checked_mul(nz))
            .ok_or(StateError::InvalidValue)?;
        self.need(cells.checked_mul(4).ok_or(StateError::InvalidValue)?)?;
        let mut f = ScalarField3D::new(nx, ny, nz, min, max);
        for v in &mut f.data {
            *v = self.f32()?;
        }
        Ok(f)
    }

    /// Refuses trailing bytes.
    pub(crate) fn finish(self) -> Result<(), StateError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: self.at,
                found: self.bytes.len(),
            })
        }
    }
}

/// Pushes the largest cell of `f` on `channel`; nothing when `f` has no
/// cells, NaN cells are skipped.
pub(crate) fn observe_max(out: &mut ObservationSink, channel: u32, f: &ScalarField3D) {
    let max = f
        .data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .reduce(f32::max);
    if let Some(m) = max {
        out.push(channel, Fix128::from_f32(m));
    }
}

/// Pushes the sum of the cells of `f` on `channel`, each cell converted to
/// [`Fix128`] first (exact for finite cells, so the sum does not depend on
/// float rounding).
pub(crate) fn observe_sum(out: &mut ObservationSink, channel: u32, f: &ScalarField3D) {
    let sum = f
        .data
        .iter()
        .fold(Fix128::ZERO, |acc, &v| acc + Fix128::from_f32(v));
    out.push(channel, sum);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    /// Simple modifier that uniformly expands the SDF
    struct ExpandModifier {
        amount: f32,
    }

    impl PhysicsModifier for ExpandModifier {
        fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
            d - self.amount
        }
        fn update(&mut self, _dt: f32) {}
        fn name(&self) -> &'static str {
            "expand"
        }
    }

    #[test]
    fn test_modified_sdf_no_modifiers() {
        let sphere = ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere));
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 1.0).abs() < 0.01,
            "No modifiers should pass through, got {d}"
        );
    }

    #[test]
    fn test_modified_sdf_expand() {
        let sphere = ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere))
            .with_modifier(Box::new(ExpandModifier { amount: 0.5 }));

        // Original: distance at (2,0,0) = 1.0
        // After expand by 0.5: distance = 0.5
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 0.5).abs() < 0.01,
            "Expand should reduce distance, got {d}"
        );
    }

    #[test]
    fn test_modifier_chain() {
        let sphere = ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = ModifiedSdf::new(Box::new(sphere))
            .with_modifier(Box::new(ExpandModifier { amount: 0.3 }))
            .with_modifier(Box::new(ExpandModifier { amount: 0.2 }));

        // Total expand = 0.5
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 0.5).abs() < 0.01,
            "Chain should sum expansions, got {d}"
        );
    }

    #[test]
    fn test_single_modified_sdf() {
        let sphere = ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let modified = SingleModifiedSdf::new(Box::new(sphere), ExpandModifier { amount: 0.5 });

        let d = modified.distance(2.0, 0.0, 0.0);
        assert!((d - 0.5).abs() < 0.01, "Single modifier expand, got {d}");
    }

    /// 地面 (distance = y): 数値誤差なしで offset を観測できる
    fn ground() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    /// update(dt) で amount が dt ずつ増える (状態を持つ) modifier
    struct GrowModifier {
        amount: f32,
    }

    impl PhysicsModifier for GrowModifier {
        fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
            d - self.amount
        }
        fn update(&mut self, dt: f32) {
            self.amount += dt;
        }
        fn name(&self) -> &'static str {
            "grow"
        }
    }

    #[test]
    fn clear_modifiers_restores_original_field_and_allows_re_adding() {
        let mut modified = ModifiedSdf::new(Box::new(ground()))
            .with_modifier(Box::new(ExpandModifier { amount: 0.25 }))
            .with_modifier(Box::new(ExpandModifier { amount: 0.5 }));
        assert_eq!(modified.modifier_count(), 2);
        // y = 2 → 2 - 0.25 - 0.5 = 1.25
        assert!((modified.distance(0.0, 2.0, 0.0) - 1.25).abs() < 1e-6);

        modified.clear_modifiers();
        assert_eq!(modified.modifier_count(), 0);
        assert!(modified.modifier_mut(0).is_none());
        // 元の field そのまま
        assert!((modified.distance(0.0, 2.0, 0.0) - 2.0).abs() < 1e-6);
        let (nx, ny, nz) = modified.normal(0.0, 2.0, 0.0);
        assert!(nx.abs() < 1e-6 && (ny - 1.0).abs() < 1e-6 && nz.abs() < 1e-6);
        // 空でもう一度 clear しても問題なし、再追加で再び効く
        modified.clear_modifiers();
        modified.add_modifier(Box::new(ExpandModifier { amount: 1.0 }));
        assert_eq!(modified.modifier_count(), 1);
        assert!((modified.distance(0.0, 2.0, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn modifier_mut_gives_in_place_access_by_index() {
        let mut modified = ModifiedSdf::new(Box::new(ground()))
            .with_modifier(Box::new(GrowModifier { amount: 0.25 }))
            .with_modifier(Box::new(ExpandModifier { amount: 0.5 }));
        // y = 2 → 2 - 0.25 - 0.5 = 1.25
        assert!((modified.distance(0.0, 2.0, 0.0) - 1.25).abs() < 1e-6);

        // index 順に対応する modifier が返る
        assert_eq!(modified.modifier_mut(0).map(|m| m.name()), Some("grow"));
        assert_eq!(modified.modifier_mut(1).map(|m| m.name()), Some("expand"));
        assert!(modified.modifier_mut(2).is_none());

        // 返された &mut を通した update は field に反映される: grow 0.25 → 0.75
        match modified.modifier_mut(0) {
            Some(m) => m.update(0.5),
            None => panic!("modifier 0 must exist"),
        }
        assert!((modified.distance(0.0, 2.0, 0.0) - 0.75).abs() < 1e-6);
        // expand (index 1) は状態を持たないので update しても不変
        match modified.modifier_mut(1) {
            Some(m) => m.update(0.5),
            None => panic!("modifier 1 must exist"),
        }
        assert!((modified.distance(0.0, 2.0, 0.0) - 0.75).abs() < 1e-6);
        // ModifiedSdf::update(dt) は全 modifier を進める = modifier_mut(0).update と同じ効果
        modified.update(0.25);
        assert!((modified.distance(0.0, 2.0, 0.0) - 0.5).abs() < 1e-6);
        assert_eq!(modified.modifier_count(), 2);
    }

    /// A modifier that would move the surface by `amount` but reports itself
    /// inactive, so both wrappers must skip it.
    struct InactiveModifier {
        amount: f32,
        updates: u32,
    }

    impl PhysicsModifier for InactiveModifier {
        fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
            d - self.amount
        }
        fn update(&mut self, _dt: f32) {
            self.updates += 1;
        }
        fn name(&self) -> &'static str {
            "inactive"
        }
        fn is_active(&self) -> bool {
            false
        }
    }

    /// oracle: on the ground `distance = y`, an expansion by `a` gives
    /// `distance = y − a` and normal `(0, 1, 0)`; an inactive modifier leaves
    /// `y`; `SingleModifiedSdf::update` advances the modifier (`GrowModifier`
    /// grows by `dt`) whether or not it is active.
    #[test]
    fn distance_and_normal_of_both_wrappers_on_the_ground() {
        let multi = ModifiedSdf::new(Box::new(ground()))
            .with_modifier(Box::new(ExpandModifier { amount: 0.5 }))
            .with_modifier(Box::new(InactiveModifier {
                amount: 9.0,
                updates: 0,
            }));
        let (d, n) = multi.distance_and_normal(0.0, 2.0, 0.0);
        assert!((d - 1.5).abs() < 1e-6, "{d}");
        assert!(
            n.0.abs() < 1e-6 && (n.1 - 1.0).abs() < 1e-6 && n.2.abs() < 1e-6,
            "{n:?}"
        );

        let mut grow = SingleModifiedSdf::new(Box::new(ground()), GrowModifier { amount: 0.25 });
        grow.update(0.5);
        let (d, n) = grow.distance_and_normal(0.0, 2.0, 0.0);
        assert!((d - 1.25).abs() < 1e-6, "{d}");
        assert!(
            n.0.abs() < 1e-6 && (n.1 - 1.0).abs() < 1e-6 && n.2.abs() < 1e-6,
            "{n:?}"
        );
        let n = grow.normal(1.0, 3.0, -1.0);
        assert!((n.1 - 1.0).abs() < 1e-6, "{n:?}");

        let mut off = SingleModifiedSdf::new(
            Box::new(ground()),
            InactiveModifier {
                amount: 9.0,
                updates: 0,
            },
        );
        off.update(0.5);
        off.update(0.5);
        assert_eq!(off.modifier.updates, 2);
        assert!((off.distance(0.0, 2.0, 0.0) - 2.0).abs() < 1e-6);
    }

    /// oracle: the payload is the version `1` as 4 little-endian bytes, then
    /// each item little endian: a `u8` as itself, a `bool` as 0/1, a `u64` /
    /// `usize` as 8 bytes, an `f32` as its 4 bit bytes, a vector as three
    /// `f32`s, a field as `nx, ny, nz` (u64), `min`, `max`, then the cells.
    #[test]
    fn state_writer_byte_layout() {
        let mut out = Vec::new();
        let mut w = StateWriter::new(&mut out);
        w.u8(7);
        w.bool(true);
        w.bool(false);
        w.u64(0x0102_0304_0506_0708);
        w.usize(5);
        w.f32(1.5);
        w.vec3((1.0, -2.0, 0.5));
        let mut f = ScalarField3D::new(2, 1, 1, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0));
        f.data = vec![0.25, -4.0];
        w.field(&f);

        let mut want: Vec<u8> = vec![
            1, 0, 0, 0, 7, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0,
        ];
        for v in [1.5_f32, 1.0, -2.0, 0.5] {
            want.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for n in [2_u64, 1, 1] {
            want.extend_from_slice(&n.to_le_bytes());
        }
        for v in [0.0_f32, 0.0, 0.0, 2.0, 1.0, 1.0, 0.25, -4.0] {
            want.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        assert_eq!(out, want);

        // read back what was written
        let mut r = StateReader::new(&out).expect("version 1");
        assert_eq!(r.u8(), Ok(7));
        assert_eq!(r.bool(), Ok(true));
        assert_eq!(r.bool(), Ok(false));
        assert_eq!(r.u64(), Ok(0x0102_0304_0506_0708));
        assert_eq!(r.usize(), Ok(5));
        assert_eq!(r.f32(), Ok(1.5));
        assert_eq!(r.vec3(), Ok((1.0, -2.0, 0.5)));
        let g = r.field().expect("field");
        assert_eq!(
            (g.nx, g.ny, g.nz, g.min, g.max),
            (2, 1, 1, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        );
        assert_eq!(g.data, vec![0.25, -4.0]);
        assert_eq!(r.finish(), Ok(()));
    }

    /// oracle: the reader refuses a version other than 1, a `bool` byte other
    /// than 0/1, bytes that end early (`Length { expected: end, found: len }`),
    /// a count whose items cannot fit, a field whose cell count overflows, and
    /// trailing bytes (`Length { expected: read, found: len }`).
    #[test]
    fn state_reader_refusals() {
        assert_eq!(
            StateReader::new(&[2, 0, 0, 0]).err(),
            Some(StateError::InvalidValue)
        );
        assert_eq!(
            StateReader::new(&[1, 0]).err(),
            Some(StateError::Length {
                expected: 4,
                found: 2
            })
        );
        let mut r = StateReader::new(&[1, 0, 0, 0, 2]).expect("version");
        assert_eq!(r.bool(), Err(StateError::InvalidValue));
        let mut r = StateReader::new(&[1, 0, 0, 0, 9, 9]).expect("version");
        assert_eq!(
            r.f32(),
            Err(StateError::Length {
                expected: 8,
                found: 6
            })
        );

        // a count of 3 items of 4 bytes with only 4 bytes left
        let mut bytes = vec![1, 0, 0, 0];
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        let mut r = StateReader::new(&bytes).expect("version");
        assert_eq!(
            r.count(4),
            Err(StateError::Length {
                expected: 24,
                found: 16
            })
        );
        // a count whose byte size overflows usize
        let mut bytes = vec![1, 0, 0, 0];
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut r = StateReader::new(&bytes).expect("version");
        assert_eq!(r.count(2), Err(StateError::InvalidValue));
        let mut r = StateReader::new(&bytes[..]).expect("version");
        assert_eq!(r.count(1), Err(StateError::InvalidValue));

        // a field of u64::MAX × 2 × 1 cells overflows the cell count
        let mut bytes = vec![1, 0, 0, 0];
        for n in [u64::MAX, 2, 1] {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&[0; 24]);
        let mut r = StateReader::new(&bytes).expect("version");
        assert_eq!(r.field().err(), Some(StateError::InvalidValue));

        // trailing byte after one u8
        let mut r = StateReader::new(&[1, 0, 0, 0, 5, 6]).expect("version");
        assert_eq!(r.u8(), Ok(5));
        assert_eq!(
            r.finish(),
            Err(StateError::Length {
                expected: 5,
                found: 6
            })
        );
    }

    /// oracle: `observe_max` pushes the largest non-NaN cell (3 of
    /// `[1, NaN, 3, −2]`) and nothing for a field without cells;
    /// `observe_sum` pushes `1 + 3 − 2 + 0.5 = 2.5` exactly.
    #[test]
    fn observe_max_and_sum_closed_form() {
        let mut f = ScalarField3D::new(4, 1, 1, (0.0, 0.0, 0.0), (4.0, 1.0, 1.0));
        f.data = vec![1.0, f32::NAN, 3.0, -2.0];
        let mut out = ObservationSink::new();
        observe_max(&mut out, 4, &f);
        assert_eq!(out.values(), &[(4, Fix128::from_int(3))]);

        let mut g = ScalarField3D::new(4, 1, 1, (0.0, 0.0, 0.0), (4.0, 1.0, 1.0));
        g.data = vec![1.0, 0.5, 3.0, -2.0];
        let mut out = ObservationSink::new();
        observe_sum(&mut out, 1, &g);
        assert_eq!(out.values(), &[(1, Fix128::from_ratio(5, 2))]);

        let mut empty = ScalarField3D::new(1, 1, 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        empty.data.clear();
        let mut out = ObservationSink::new();
        observe_max(&mut out, 0, &empty);
        assert!(out.values().is_empty());
    }
}
