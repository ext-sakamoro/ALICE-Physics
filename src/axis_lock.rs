//! # MODEL
//!
//! - **Equations**: a holonomic constraint `xᵢ = cᵢ` on each locked world
//!   position axis `i` and `ωⱼ = 0` on each locked world rotation axis `j`,
//!   enforced by projection: at the start of every substep the locked
//!   components of the linear and angular velocity are set to zero, and at the
//!   end the locked position components are put back to their value at the
//!   start of the substep. For the orientation, the substep's rotation
//!   `Δ = q · q₀*` has the vector components along the locked axes removed and
//!   is normalized again (`q = normalize(Δ') · q₀`); with all three rotation
//!   axes locked `q = q₀`. The velocities are derived from the projected pose,
//!   and the locked components are set to zero once more.
//! - **Units**: SI. Positions m, velocities m/s and rad/s; the mask has no unit.
//! - **Conservation**: a lock is an external constraint: it removes momentum
//!   and kinetic energy along the locked axes (the reaction is not applied to
//!   anything else), and conserves both along the free axes for a body that is
//!   otherwise free.
// LIMITATION(COV-RIGID-122): With one locked rotation axis the removal of that component of `Δ` is exact to first order in the substep's rotation angle.
//! - **Range**: dynamic bodies only (static and kinematic bodies move as they
//!   do without a lock). Exact for any number of locked position axes and for
//!   zero, two or three locked rotation axes (two locked leave the rotation
//!   about the free axis, a twist). With one locked rotation axis the removal of
//!   that component of `Δ` is exact to first order in the substep's rotation
//!   angle.
//! - **Approximations**: a projection, not a solved constraint: a contact or a
//!   joint that pushes along a locked axis moves the body and is undone at the
//!   end of the substep, so the locked body acts as infinitely heavy along that
//!   axis only after the projection, and the other body receives the share of
//!   the correction its inverse mass gives it in each iteration. The locked axes
//!   are world axes.
//! - **Tiers**: L0 only (the projection is the model; a constraint solved per
//!   axis inside the solve, scaling the inverse mass and inertia per axis, is
//!   another model and is not built).
//! - **Validation** (`tests/analytic_axis_lock.rs`):
//!   `a_locked_position_axis_stays_exact_and_the_free_axes_follow_the_twin`,
//!   `locked_rotation_axes_leave_a_turn_about_the_free_axis`,
//!   `contacts_and_joints_do_not_move_a_locked_axis`,
//!   `remove_body_keeps_the_lock_table_aligned`,
//!   `the_snapshot_and_law_id_carry_locks_only_when_there_are_some`,
//!   `the_mask_is_six_bits`.
//! - validation missing: one locked rotation axis against an exact constrained
//!   rotation (the first-order removal is not compared with a closed form)
//!
//! Per-body axis locks: the world axes along which a dynamic body may not move
//! or turn, for planar games (lock z and the x / y rotations), sliders on rails
//! and turntables. Set with [`crate::solver::PhysicsWorld::set_body_axis_lock`].

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::RigidBody;

/// A set of locked world axes: three position axes and three rotation axes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AxisLock(u8);

impl AxisLock {
    /// Nothing locked.
    pub const NONE: Self = Self(0);
    /// The x position.
    pub const LINEAR_X: Self = Self(1);
    /// The y position.
    pub const LINEAR_Y: Self = Self(1 << 1);
    /// The z position.
    pub const LINEAR_Z: Self = Self(1 << 2);
    /// The rotation about x.
    pub const ANGULAR_X: Self = Self(1 << 3);
    /// The rotation about y.
    pub const ANGULAR_Y: Self = Self(1 << 4);
    /// The rotation about z.
    pub const ANGULAR_Z: Self = Self(1 << 5);
    /// The three position axes.
    pub const LINEAR: Self = Self(0b000_111);
    /// The three rotation axes.
    pub const ANGULAR: Self = Self(0b111_000);
    /// Every axis.
    pub const ALL: Self = Self(0b111_111);

    /// The set with these bits (`LINEAR_X` = 1 … `ANGULAR_Z` = 32), `None` when
    /// a bit above the sixth is set.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// The bits of the set.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every axis of `other` is in this set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether nothing is locked.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// The locked position axes as `[x, y, z]`.
    const fn linear(self) -> [bool; 3] {
        [self.0 & 1 != 0, self.0 & 2 != 0, self.0 & 4 != 0]
    }

    /// The locked rotation axes as `[x, y, z]`.
    const fn angular(self) -> [bool; 3] {
        [self.0 & 8 != 0, self.0 & 16 != 0, self.0 & 32 != 0]
    }
}

impl core::ops::BitOr for AxisLock {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for AxisLock {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// `v` with the components of the locked axes set to zero.
fn zero_axes(v: Vec3Fix, locked: [bool; 3]) -> Vec3Fix {
    Vec3Fix::new(
        if locked[0] { Fix128::ZERO } else { v.x },
        if locked[1] { Fix128::ZERO } else { v.y },
        if locked[2] { Fix128::ZERO } else { v.z },
    )
}

/// The start-of-substep pose of a locked body, kept to project its end pose.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LockedStart {
    pub(crate) body: usize,
    pub(crate) lock: AxisLock,
    pub(crate) position: Vec3Fix,
    pub(crate) rotation: QuatFix,
}

/// Set the locked components of the body's velocities to zero.
pub(crate) fn project_velocity(lock: AxisLock, body: &mut RigidBody) {
    body.velocity = zero_axes(body.velocity, lock.linear());
    body.angular_velocity = zero_axes(body.angular_velocity, lock.angular());
}

/// Put the locked components of the body's pose back to `start`.
///
/// A coordinate already at its start value and a substep rotation with no
/// component along a locked axis are left as they are, bit for bit.
pub(crate) fn project_pose(start: &LockedStart, body: &mut RigidBody) {
    let lin = start.lock.linear();
    let p = body.position;
    body.position = Vec3Fix::new(
        if lin[0] { start.position.x } else { p.x },
        if lin[1] { start.position.y } else { p.y },
        if lin[2] { start.position.z } else { p.z },
    );
    let ang = start.lock.angular();
    if ang == [true; 3] {
        body.rotation = start.rotation;
    } else if ang != [false; 3] {
        let delta = body.rotation.mul(start.rotation.conjugate());
        let v = Vec3Fix::new(delta.x, delta.y, delta.z);
        let kept = zero_axes(v, ang);
        if kept != v {
            let turn = QuatFix::new(kept.x, kept.y, kept.z, delta.w).normalize();
            body.rotation = turn.mul(start.rotation);
        }
    }
}
