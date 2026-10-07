//! Whole-world snapshot: every piece of [`PhysicsWorld`] state that
//! [`PhysicsWorld::step`] reads, in one versioned blob with one checksum.
//!
// LIMITATION(COV-ENGINE-011): keeps only the per-body motion and sleep state and expects the caller to rebuild joints, colliders, force fields, materials and so on by replay.
//! [`PhysicsWorld::serialize_state`] (rollback netcode) keeps only the
//! per-body motion and sleep state and expects the caller to rebuild
//! joints, colliders, force fields, materials and so on by replay. This
//! module is the other end: [`PhysicsWorld::snapshot_world`] writes the whole
//! world, and [`PhysicsWorld::from_world_snapshot`] /
//! [`PhysicsWorld::restore_world`] bring it back so that every later `step`
//! is bit-identical to the original's (branching search, rollback without a
//! replay log, persistence).
//!
//! The field-by-field coverage, the format and the errors are documented on
//! [`PhysicsWorld::snapshot_world`] and [`WorldSnapshotError`].

use super::{fnv1a_fold, ConstraintBatch, ContactConstraint, DistanceConstraint, PhysicsWorld};
use super::{BodyType, Broadphase, RigidBody, SolverBackend, SolverConfig};
use crate::body_collider::BodyCollider;
use crate::collider::{Capsule, Contact, ConvexHull, Sphere, AABB};
use crate::compound::{CompoundChild, CompoundShape, ShapeRef};
use crate::contact_cache::{BodyPairKey, CachedContactPoint, ContactCache, ContactManifold};
use crate::event::{ContactEvent, ContactEventType, EventCollector, TriggerEvent};
use crate::filter::CollisionFilter;
use crate::force::{ForceField, ForceFieldInstance};
use crate::joint::{
    BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, SliderJoint,
    SpringJoint,
};
use crate::material::{
    CombineRule, MaterialTable, PairOverride, PhysicsMaterial, MATERIAL_ID_CAPACITY,
};
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::shape::Shape;
use crate::sleeping::{IslandManager, SleepConfig, SleepData, SleepState};
use crate::static_collider::StaticCollider;
use crate::world_participant::WorldFault;

#[cfg(not(feature = "std"))]
use alloc::collections::{BTreeMap, BTreeSet};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::{BTreeMap, BTreeSet};

/// Why [`PhysicsWorld::restore_world`] / [`PhysicsWorld::from_world_snapshot`]
/// rejected a blob. The target world is unchanged in every case.
///
/// # Which input gives which error
///
/// | input | error |
/// |---|---|
/// | fewer than 24 bytes, or shorter than the declared payload | [`Self::Truncated`] |
/// | wrong magic | [`Self::BadMagic`] |
/// | other version | [`Self::UnsupportedVersion`] |
/// | reserved bytes not 0 | [`Self::ReservedNotZero`] |
/// | longer than the declared payload | [`Self::TrailingBytes`] |
/// | checksum differs | [`Self::ChecksumMismatch`] |
/// | unknown enum tag / non-boolean byte / unrepresentable value inside the payload | [`Self::InvalidValue`] |
/// | material table with 0 or more than 65,536 materials, or a material whose id is not its index | [`Self::InvalidValue`] (`section: "material_table"`) |
/// | a joint / constraint / contact / SDF collider / batch / proxy index past its target | [`Self::DanglingIndex`] |
/// | target world holds a different number of SDF fields | [`Self::SdfFieldCountMismatch`] |
/// | target world holds a different number of hooks / modifiers / bridges | [`Self::CallbackCountMismatch`] |
/// | target world holds other participants (count, kinds, order) | [`Self::ParticipantMismatch`] |
/// | a participant of the target world refuses its payload | [`Self::ParticipantState`] |
/// | the target world declares other shared fields | [`Self::FieldState`] |
/// | unknown fault code / participant fault tag, field mode or layout tag | [`Self::InvalidValue`] |
///
/// On every error the target world is left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorldSnapshotError {
    /// The blob ends before the header or the declared payload does.
    Truncated,
    /// The first 4 bytes are not [`PhysicsWorld::WORLD_SNAPSHOT_MAGIC`].
    BadMagic,
    /// The version is not [`PhysicsWorld::WORLD_SNAPSHOT_VERSION`].
    UnsupportedVersion {
        /// Version stored in the blob.
        found: u16,
        /// Version this build reads.
        supported: u16,
    },
    /// The reserved header bytes are not 0.
    ReservedNotZero,
    /// Bytes remain after the checksum, or after the last payload section.
    TrailingBytes {
        /// Number of unread bytes.
        extra: usize,
    },
    /// The stored checksum differs from the one computed over the blob.
    ChecksumMismatch {
        /// Checksum stored in the blob.
        stored: u64,
        /// Checksum of the bytes as received.
        computed: u64,
    },
    /// A value inside the payload is not valid for its field (unknown enum
    /// tag, a boolean byte other than 0 / 1, a count that does not fit
    /// `usize`, invalid metric weights, a material table holding 0 or more
    /// than 65,536 materials or a material whose id is not its index).
    InvalidValue {
        /// Section the value was read in.
        section: &'static str,
    },
    /// An index refers past the end of what it indexes (for example a joint
    /// naming a body the snapshot does not hold).
    DanglingIndex {
        /// Section holding the index.
        section: &'static str,
        /// The index.
        index: usize,
        /// Length of what it indexes.
        len: usize,
    },
    /// The snapshot holds a different number of SDF colliders than the target
    /// world holds SDF fields (fields are code and are not in the blob).
    SdfFieldCountMismatch {
        /// SDF colliders in the snapshot.
        snapshot: usize,
        /// SDF colliders in the target world.
        world: usize,
    },
    /// The snapshot was taken with a different number of pre-solve hooks,
    /// contact modifiers or installed GPU bridges than the target world has.
    CallbackCountMismatch {
        /// `"pre_solve_hooks"`, `"contact_modifiers"` or `"gpu_solver_bridge"`.
        section: &'static str,
        /// Count in the snapshot.
        snapshot: usize,
        /// Count in the target world.
        world: usize,
    },
    /// The target world's participants differ from the snapshot's in number,
    /// kinds or order (participants are code and are not in the blob; the
    /// target world must already hold the same ones).
    ParticipantMismatch(crate::world_participant::ParticipantMismatch),
    /// Participant `index` of the target world refused its payload
    /// ([`crate::world_participant::Participant::check_state`]). Every
    /// payload is checked before any is read.
    ParticipantState {
        /// Registration index of the participant.
        index: usize,
        /// Why it refused.
        error: crate::world_participant::StateError,
    },
    /// The snapshot's shared fields do not match the target world's
    /// declared fields (ids, modes, layouts;
    /// [`crate::world_participant::FieldBoard::check_values`]).
    FieldState(crate::world_participant::StateError),
}

impl core::fmt::Display for WorldSnapshotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "world snapshot is truncated"),
            Self::BadMagic => write!(f, "not a world snapshot (bad magic)"),
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "world snapshot version {found} is not supported (expected {supported})"
            ),
            Self::ReservedNotZero => write!(f, "world snapshot reserved header bytes are not 0"),
            Self::TrailingBytes { extra } => {
                write!(f, "world snapshot has {extra} trailing bytes")
            }
            Self::ChecksumMismatch { stored, computed } => write!(
                f,
                "world snapshot checksum mismatch (stored {stored:#018x}, computed {computed:#018x})"
            ),
            Self::InvalidValue { section } => {
                write!(f, "world snapshot has an invalid value in {section}")
            }
            Self::DanglingIndex { section, index, len } => write!(
                f,
                "world snapshot {section} refers to index {index} of {len}"
            ),
            Self::SdfFieldCountMismatch { snapshot, world } => write!(
                f,
                "world snapshot has {snapshot} SDF colliders, target world holds {world} SDF fields"
            ),
            Self::CallbackCountMismatch {
                section,
                snapshot,
                world,
            } => write!(
                f,
                "world snapshot was taken with {snapshot} {section}, target world has {world}"
            ),
            Self::ParticipantMismatch(m) => write!(
                f,
                "world snapshot participants differ from the target world's: {m:?}"
            ),
            Self::ParticipantState { index, error } => write!(
                f,
                "participant {index} of the target world refused its snapshot payload: {error:?}"
            ),
            Self::FieldState(error) => write!(
                f,
                "world snapshot fields differ from the target world's: {error:?}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WorldSnapshotError {}

type Res<T> = Result<T, WorldSnapshotError>;

const HEADER_LEN: usize = 16;
const CHECKSUM_LEN: usize = 8;

// ── Writer ────────────────────────────────────────────────────────────────

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn fix(&mut self, v: Fix128) {
        self.0.extend_from_slice(&v.hi.to_le_bytes());
        self.0.extend_from_slice(&v.lo.to_le_bytes());
    }
    fn vec3(&mut self, v: Vec3Fix) {
        self.fix(v.x);
        self.fix(v.y);
        self.fix(v.z);
    }
    fn quat(&mut self, q: QuatFix) {
        self.fix(q.x);
        self.fix(q.y);
        self.fix(q.z);
        self.fix(q.w);
    }
    fn opt_fix(&mut self, v: Option<Fix128>) {
        match v {
            Some(x) => {
                self.u8(1);
                self.fix(x);
            }
            None => self.u8(0),
        }
    }
    fn aabb(&mut self, b: AABB) {
        self.vec3(b.min);
        self.vec3(b.max);
    }
}

// ── Reader ────────────────────────────────────────────────────────────────

struct R<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(WorldSnapshotError::Truncated)?;
        if end > self.data.len() {
            return Err(WorldSnapshotError::Truncated);
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Res<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self, section: &'static str) -> Res<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(WorldSnapshotError::InvalidValue { section }),
        }
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn i32(&mut self) -> Res<i32> {
        Ok(i32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn usize(&mut self, section: &'static str) -> Res<usize> {
        usize::try_from(self.u64()?).map_err(|_| WorldSnapshotError::InvalidValue { section })
    }
    /// A element count: each element takes at least `min_size` bytes, so a
    /// count larger than the rest of the blob allows is truncation (and is
    /// rejected before anything is allocated for it).
    fn len(&mut self, section: &'static str, min_size: usize) -> Res<usize> {
        let n = self.usize(section)?;
        let remaining = self.data.len() - self.pos;
        if n.checked_mul(min_size.max(1))
            .is_none_or(|need| need > remaining)
        {
            return Err(WorldSnapshotError::Truncated);
        }
        Ok(n)
    }
    fn f32(&mut self) -> Res<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn fix(&mut self) -> Res<Fix128> {
        let hi = i64::from_le_bytes(self.arr()?);
        let lo = u64::from_le_bytes(self.arr()?);
        Ok(Fix128 { hi, lo })
    }
    fn vec3(&mut self) -> Res<Vec3Fix> {
        Ok(Vec3Fix::new(self.fix()?, self.fix()?, self.fix()?))
    }
    fn quat(&mut self) -> Res<QuatFix> {
        Ok(QuatFix {
            x: self.fix()?,
            y: self.fix()?,
            z: self.fix()?,
            w: self.fix()?,
        })
    }
    fn opt_fix(&mut self, section: &'static str) -> Res<Option<Fix128>> {
        Ok(if self.bool(section)? {
            Some(self.fix()?)
        } else {
            None
        })
    }
    fn aabb(&mut self) -> Res<AABB> {
        Ok(AABB {
            min: self.vec3()?,
            max: self.vec3()?,
        })
    }
}

const fn invalid(section: &'static str) -> WorldSnapshotError {
    WorldSnapshotError::InvalidValue { section }
}

// ── Leaf types ────────────────────────────────────────────────────────────

fn w_config(w: &mut W, c: &SolverConfig) {
    w.usize(c.substeps);
    w.usize(c.iterations);
    w.vec3(c.gravity);
    w.fix(c.damping);
    w.fix(c.warm_start_factor);
    w.u8(match c.solver_backend {
        SolverBackend::Xpbd => 0,
        SolverBackend::Tgs => 1,
    });
}

fn r_config(r: &mut R<'_>) -> Res<SolverConfig> {
    const S: &str = "config";
    Ok(SolverConfig {
        substeps: r.usize(S)?,
        iterations: r.usize(S)?,
        gravity: r.vec3()?,
        damping: r.fix()?,
        warm_start_factor: r.fix()?,
        solver_backend: match r.u8()? {
            0 => SolverBackend::Xpbd,
            1 => SolverBackend::Tgs,
            _ => return Err(invalid(S)),
        },
    })
}

fn w_body(w: &mut W, b: &RigidBody) {
    w.vec3(b.position);
    w.vec3(b.velocity);
    w.fix(b.inv_mass);
    w.vec3(b.inv_inertia);
    w.vec3(b.prev_position);
    w.quat(b.rotation);
    w.vec3(b.angular_velocity);
    w.quat(b.prev_rotation);
    w.fix(b.restitution);
    w.fix(b.friction);
    w.fix(b.gravity_scale);
    w.fix(b.linear_damping);
    w.fix(b.angular_damping);
    w.bool(b.is_sensor);
    w.u8(b.body_type as u8);
    match b.kinematic_target {
        Some((p, q)) => {
            w.u8(1);
            w.vec3(p);
            w.quat(q);
        }
        None => w.u8(0),
    }
}

fn r_body(r: &mut R<'_>) -> Res<RigidBody> {
    const S: &str = "bodies";
    Ok(RigidBody {
        position: r.vec3()?,
        velocity: r.vec3()?,
        inv_mass: r.fix()?,
        inv_inertia: r.vec3()?,
        prev_position: r.vec3()?,
        rotation: r.quat()?,
        angular_velocity: r.vec3()?,
        prev_rotation: r.quat()?,
        restitution: r.fix()?,
        friction: r.fix()?,
        gravity_scale: r.fix()?,
        linear_damping: r.fix()?,
        angular_damping: r.fix()?,
        is_sensor: r.bool(S)?,
        body_type: match r.u8()? {
            0 => BodyType::Dynamic,
            1 => BodyType::Static,
            2 => BodyType::Kinematic,
            _ => return Err(invalid(S)),
        },
        kinematic_target: if r.bool(S)? {
            Some((r.vec3()?, r.quat()?))
        } else {
            None
        },
    })
}

fn w_contact(w: &mut W, c: &Contact) {
    w.fix(c.depth);
    w.vec3(c.normal);
    w.vec3(c.point_a);
    w.vec3(c.point_b);
}

fn r_contact(r: &mut R<'_>) -> Res<Contact> {
    Ok(Contact {
        depth: r.fix()?,
        normal: r.vec3()?,
        point_a: r.vec3()?,
        point_b: r.vec3()?,
    })
}

fn w_joint(w: &mut W, j: &Joint) {
    match j {
        Joint::Ball(j) => {
            w.u8(0);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.fix(j.compliance);
            w.opt_fix(j.break_force);
        }
        Joint::Hinge(j) => {
            w.u8(1);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.vec3(j.local_axis_a);
            w.vec3(j.local_axis_b);
            w.opt_fix(j.angle_min);
            w.opt_fix(j.angle_max);
            w.fix(j.compliance);
            w.fix(j.angular_compliance);
            w.opt_fix(j.break_force);
        }
        Joint::Fixed(j) => {
            w.u8(2);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.quat(j.relative_rotation);
            w.fix(j.compliance);
            w.fix(j.angular_compliance);
            w.opt_fix(j.break_force);
        }
        Joint::Slider(j) => {
            w.u8(3);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_axis);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.opt_fix(j.limit_min);
            w.opt_fix(j.limit_max);
            w.fix(j.compliance);
            w.opt_fix(j.break_force);
        }
        Joint::Spring(j) => {
            w.u8(4);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.fix(j.rest_length);
            w.fix(j.stiffness);
            w.fix(j.damping);
            w.opt_fix(j.break_force);
        }
        Joint::D6(j) => {
            w.u8(5);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.quat(j.local_frame_a);
            w.quat(j.local_frame_b);
            for m in [
                j.linear_x,
                j.linear_y,
                j.linear_z,
                j.angular_x,
                j.angular_y,
                j.angular_z,
            ] {
                w.u8(match m {
                    D6Motion::Locked => 0,
                    D6Motion::Free => 1,
                    D6Motion::Limited => 2,
                });
            }
            w.vec3(j.linear_limit_min);
            w.vec3(j.linear_limit_max);
            w.vec3(j.angular_limit_min);
            w.vec3(j.angular_limit_max);
            w.fix(j.compliance);
            w.fix(j.angular_compliance);
            w.opt_fix(j.break_force);
        }
        Joint::ConeTwist(j) => {
            w.u8(6);
            w.usize(j.body_a);
            w.usize(j.body_b);
            w.vec3(j.local_anchor_a);
            w.vec3(j.local_anchor_b);
            w.vec3(j.twist_axis_a);
            w.vec3(j.twist_axis_b);
            w.fix(j.cone_limit);
            w.fix(j.twist_limit);
            w.fix(j.compliance);
            w.fix(j.angular_compliance);
            w.opt_fix(j.break_force);
        }
    }
}

fn r_d6_motion(r: &mut R<'_>) -> Res<D6Motion> {
    match r.u8()? {
        0 => Ok(D6Motion::Locked),
        1 => Ok(D6Motion::Free),
        2 => Ok(D6Motion::Limited),
        _ => Err(invalid("joints")),
    }
}

fn r_joint(r: &mut R<'_>) -> Res<Joint> {
    const S: &str = "joints";
    Ok(match r.u8()? {
        0 => Joint::Ball(BallJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        1 => Joint::Hinge(HingeJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            local_axis_a: r.vec3()?,
            local_axis_b: r.vec3()?,
            angle_min: r.opt_fix(S)?,
            angle_max: r.opt_fix(S)?,
            compliance: r.fix()?,
            angular_compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        2 => Joint::Fixed(FixedJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            relative_rotation: r.quat()?,
            compliance: r.fix()?,
            angular_compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        3 => Joint::Slider(SliderJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_axis: r.vec3()?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            limit_min: r.opt_fix(S)?,
            limit_max: r.opt_fix(S)?,
            compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        4 => Joint::Spring(SpringJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            rest_length: r.fix()?,
            stiffness: r.fix()?,
            damping: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        5 => Joint::D6(D6Joint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            local_frame_a: r.quat()?,
            local_frame_b: r.quat()?,
            linear_x: r_d6_motion(r)?,
            linear_y: r_d6_motion(r)?,
            linear_z: r_d6_motion(r)?,
            angular_x: r_d6_motion(r)?,
            angular_y: r_d6_motion(r)?,
            angular_z: r_d6_motion(r)?,
            linear_limit_min: r.vec3()?,
            linear_limit_max: r.vec3()?,
            angular_limit_min: r.vec3()?,
            angular_limit_max: r.vec3()?,
            compliance: r.fix()?,
            angular_compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        6 => Joint::ConeTwist(ConeTwistJoint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            twist_axis_a: r.vec3()?,
            twist_axis_b: r.vec3()?,
            cone_limit: r.fix()?,
            twist_limit: r.fix()?,
            compliance: r.fix()?,
            angular_compliance: r.fix()?,
            break_force: r.opt_fix(S)?,
        }),
        _ => return Err(invalid(S)),
    })
}

fn w_force_field(w: &mut W, f: &ForceFieldInstance) {
    match f.field {
        ForceField::Directional {
            direction,
            strength,
        } => {
            w.u8(0);
            w.vec3(direction);
            w.fix(strength);
        }
        ForceField::Point {
            center,
            strength,
            repulsive,
            max_force,
        } => {
            w.u8(1);
            w.vec3(center);
            w.fix(strength);
            w.bool(repulsive);
            w.fix(max_force);
        }
        ForceField::Drag { coefficient } => {
            w.u8(2);
            w.fix(coefficient);
        }
        ForceField::Buoyancy {
            surface_y,
            density,
            drag,
        } => {
            w.u8(3);
            w.fix(surface_y);
            w.fix(density);
            w.fix(drag);
        }
        ForceField::Vortex {
            center,
            axis,
            strength,
            falloff_radius,
        } => {
            w.u8(4);
            w.vec3(center);
            w.vec3(axis);
            w.fix(strength);
            w.fix(falloff_radius);
        }
        ForceField::Explosion {
            center,
            strength,
            radius,
            falloff_power,
        } => {
            w.u8(5);
            w.vec3(center);
            w.fix(strength);
            w.fix(radius);
            w.fix(falloff_power);
        }
        ForceField::Magnetic {
            position,
            moment,
            strength,
        } => {
            w.u8(6);
            w.vec3(position);
            w.vec3(moment);
            w.fix(strength);
        }
    }
    match &f.affected_bodies {
        Some(list) => {
            w.u8(1);
            w.usize(list.len());
            for &i in list {
                w.usize(i);
            }
        }
        None => w.u8(0),
    }
    w.bool(f.enabled);
}

fn r_force_field(r: &mut R<'_>) -> Res<ForceFieldInstance> {
    const S: &str = "force_fields";
    let field = match r.u8()? {
        0 => ForceField::Directional {
            direction: r.vec3()?,
            strength: r.fix()?,
        },
        1 => ForceField::Point {
            center: r.vec3()?,
            strength: r.fix()?,
            repulsive: r.bool(S)?,
            max_force: r.fix()?,
        },
        2 => ForceField::Drag {
            coefficient: r.fix()?,
        },
        3 => ForceField::Buoyancy {
            surface_y: r.fix()?,
            density: r.fix()?,
            drag: r.fix()?,
        },
        4 => ForceField::Vortex {
            center: r.vec3()?,
            axis: r.vec3()?,
            strength: r.fix()?,
            falloff_radius: r.fix()?,
        },
        5 => ForceField::Explosion {
            center: r.vec3()?,
            strength: r.fix()?,
            radius: r.fix()?,
            falloff_power: r.fix()?,
        },
        6 => ForceField::Magnetic {
            position: r.vec3()?,
            moment: r.vec3()?,
            strength: r.fix()?,
        },
        _ => return Err(invalid(S)),
    };
    let affected_bodies = if r.bool(S)? {
        let n = r.len(S, 8)?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(r.usize(S)?);
        }
        Some(v)
    } else {
        None
    };
    Ok(ForceFieldInstance {
        field,
        affected_bodies,
        enabled: r.bool(S)?,
    })
}

fn w_static_collider(w: &mut W, c: &StaticCollider) {
    match c {
        StaticCollider::Plane(p) => {
            w.u8(0);
            w.vec3(p.normal);
            w.fix(p.offset);
        }
        StaticCollider::HeightField(h) => {
            w.u8(1);
            w.usize(h.heights.len());
            for &v in &h.heights {
                w.fix(v);
            }
            w.u32(h.width);
            w.u32(h.depth);
            w.fix(h.spacing);
            w.vec3(h.origin);
        }
        StaticCollider::TriMesh(m) => {
            w.u8(2);
            w.usize(m.triangles.len());
            for t in &m.triangles {
                w.vec3(t.v0);
                w.vec3(t.v1);
                w.vec3(t.v2);
            }
            w.usize(m.bvh.nodes.len());
            for n in &m.bvh.nodes {
                for v in n.aabb_min {
                    w.i32(v);
                }
                w.u32(n.first_child_or_prim);
                for v in n.aabb_max {
                    w.i32(v);
                }
                w.u32(n.prim_count_escape);
            }
            w.usize(m.bvh.primitives.len());
            for &p in &m.bvh.primitives {
                w.u32(p);
            }
            w.aabb(m.bvh.bounds);
            w.aabb(m.bounds);
        }
    }
}

fn r_static_collider(r: &mut R<'_>) -> Res<StaticCollider> {
    const S: &str = "static_colliders";
    Ok(match r.u8()? {
        0 => StaticCollider::Plane(crate::plane_collider::PlaneCollider {
            normal: r.vec3()?,
            offset: r.fix()?,
        }),
        1 => {
            let n = r.len(S, 16)?;
            let mut heights = Vec::with_capacity(n);
            for _ in 0..n {
                heights.push(r.fix()?);
            }
            StaticCollider::HeightField(crate::heightfield::HeightField {
                heights,
                width: r.u32()?,
                depth: r.u32()?,
                spacing: r.fix()?,
                origin: r.vec3()?,
            })
        }
        2 => {
            let n = r.len(S, 144)?;
            let mut triangles = Vec::with_capacity(n);
            for _ in 0..n {
                triangles.push(crate::trimesh::Triangle {
                    v0: r.vec3()?,
                    v1: r.vec3()?,
                    v2: r.vec3()?,
                });
            }
            let n = r.len(S, 32)?;
            let mut nodes = Vec::with_capacity(n);
            for _ in 0..n {
                let aabb_min = [r.i32()?, r.i32()?, r.i32()?];
                let first_child_or_prim = r.u32()?;
                let aabb_max = [r.i32()?, r.i32()?, r.i32()?];
                let prim_count_escape = r.u32()?;
                nodes.push(crate::bvh::BvhNode {
                    aabb_min,
                    first_child_or_prim,
                    aabb_max,
                    prim_count_escape,
                });
            }
            let n = r.len(S, 4)?;
            let mut primitives = Vec::with_capacity(n);
            for _ in 0..n {
                primitives.push(r.u32()?);
            }
            let bvh = crate::bvh::LinearBvh {
                nodes,
                primitives,
                bounds: r.aabb()?,
            };
            StaticCollider::TriMesh(crate::trimesh::TriMesh {
                triangles,
                bvh,
                bounds: r.aabb()?,
            })
        }
        _ => return Err(invalid(S)),
    })
}

fn w_shape(w: &mut W, s: &Shape) {
    match *s {
        Shape::Box { half_extents } => {
            w.u8(0);
            w.vec3(half_extents);
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            w.u8(1);
            w.fix(radius);
            w.fix(half_height);
        }
        Shape::Cone {
            radius,
            half_height,
        } => {
            w.u8(2);
            w.fix(radius);
            w.fix(half_height);
        }
        Shape::Ellipsoid { radii } => {
            w.u8(3);
            w.vec3(radii);
        }
        Shape::Wedge {
            width,
            height,
            depth,
        } => {
            w.u8(4);
            w.fix(width);
            w.fix(height);
            w.fix(depth);
        }
        Shape::Torus {
            major_radius,
            minor_radius,
        } => {
            w.u8(5);
            w.fix(major_radius);
            w.fix(minor_radius);
        }
    }
}

fn r_shape(r: &mut R<'_>) -> Res<Shape> {
    Ok(match r.u8()? {
        0 => Shape::Box {
            half_extents: r.vec3()?,
        },
        1 => Shape::Cylinder {
            radius: r.fix()?,
            half_height: r.fix()?,
        },
        2 => Shape::Cone {
            radius: r.fix()?,
            half_height: r.fix()?,
        },
        3 => Shape::Ellipsoid { radii: r.vec3()? },
        4 => Shape::Wedge {
            width: r.fix()?,
            height: r.fix()?,
            depth: r.fix()?,
        },
        5 => Shape::Torus {
            major_radius: r.fix()?,
            minor_radius: r.fix()?,
        },
        _ => return Err(invalid("body_colliders")),
    })
}

fn w_body_collider(w: &mut W, c: Option<&BodyCollider>) {
    match c {
        None => w.u8(0),
        Some(BodyCollider::Shape(s)) => {
            w.u8(1);
            w_shape(w, s);
        }
        Some(BodyCollider::Compound(c)) => {
            w.u8(2);
            w.usize(c.children.len());
            for ch in &c.children {
                match &ch.shape {
                    ShapeRef::Sphere(s) => {
                        w.u8(0);
                        w.vec3(s.center);
                        w.fix(s.radius);
                    }
                    ShapeRef::Capsule(s) => {
                        w.u8(1);
                        w.vec3(s.a);
                        w.vec3(s.b);
                        w.fix(s.radius);
                    }
                    ShapeRef::Box(b) => {
                        w.u8(2);
                        w.vec3(b.center);
                        w.vec3(b.half_extents);
                        w.quat(b.rotation);
                    }
                    ShapeRef::ConvexHull(h) => {
                        w.u8(3);
                        w.usize(h.vertices.len());
                        for &v in &h.vertices {
                            w.vec3(v);
                        }
                    }
                }
                w.vec3(ch.local_position);
                w.quat(ch.local_rotation);
            }
            w.aabb(c.cached_aabb);
            w.bool(c.dirty);
        }
    }
}

fn r_body_collider(r: &mut R<'_>) -> Res<Option<BodyCollider>> {
    const S: &str = "body_colliders";
    Ok(match r.u8()? {
        0 => None,
        1 => Some(BodyCollider::Shape(r_shape(r)?)),
        2 => {
            let n = r.len(S, 1)?;
            let mut children = Vec::with_capacity(n);
            for _ in 0..n {
                let shape = match r.u8()? {
                    0 => ShapeRef::Sphere(Sphere {
                        center: r.vec3()?,
                        radius: r.fix()?,
                    }),
                    1 => ShapeRef::Capsule(Capsule {
                        a: r.vec3()?,
                        b: r.vec3()?,
                        radius: r.fix()?,
                    }),
                    2 => ShapeRef::Box(crate::box_collider::OrientedBox {
                        center: r.vec3()?,
                        half_extents: r.vec3()?,
                        rotation: r.quat()?,
                    }),
                    3 => {
                        let m = r.len(S, 48)?;
                        let mut vertices = Vec::with_capacity(m);
                        for _ in 0..m {
                            vertices.push(r.vec3()?);
                        }
                        ShapeRef::ConvexHull(ConvexHull { vertices })
                    }
                    _ => return Err(invalid(S)),
                };
                children.push(CompoundChild {
                    shape,
                    local_position: r.vec3()?,
                    local_rotation: r.quat()?,
                });
            }
            let mut c = CompoundShape::new();
            c.children = children;
            c.cached_aabb = r.aabb()?;
            c.dirty = r.bool(S)?;
            Some(BodyCollider::Compound(c))
        }
        _ => return Err(invalid(S)),
    })
}

fn w_contact_cache(w: &mut W, c: &ContactCache) {
    w.usize(c.manifolds.len());
    for m in &c.manifolds {
        w.u32(m.pair.body_a);
        w.u32(m.pair.body_b);
        w.usize(m.points.len());
        for p in &m.points {
            w.vec3(p.local_point_a);
            w.vec3(p.local_point_b);
            w.vec3(p.normal);
            w.fix(p.depth);
            w.fix(p.lambda_n);
            w.fix(p.lambda_t1);
            w.fix(p.lambda_t2);
            w.u32(p.age);
        }
        w.vec3(m.normal);
        w.fix(m.friction);
        w.fix(m.restitution);
        w.u32(m.stale_frames);
    }
    w.u32(c.max_stale_frames);
    w.fix(c.warm_start_factor);
}

fn r_contact_cache(r: &mut R<'_>) -> Res<ContactCache> {
    const S: &str = "contact_cache";
    let n = r.len(S, 8)?;
    let mut manifolds = Vec::with_capacity(n);
    for _ in 0..n {
        let pair = BodyPairKey {
            body_a: r.u32()?,
            body_b: r.u32()?,
        };
        let k = r.len(S, 212)?;
        let mut points = Vec::with_capacity(k);
        for _ in 0..k {
            points.push(CachedContactPoint {
                local_point_a: r.vec3()?,
                local_point_b: r.vec3()?,
                normal: r.vec3()?,
                depth: r.fix()?,
                lambda_n: r.fix()?,
                lambda_t1: r.fix()?,
                lambda_t2: r.fix()?,
                age: r.u32()?,
            });
        }
        manifolds.push(ContactManifold {
            pair,
            points,
            normal: r.vec3()?,
            friction: r.fix()?,
            restitution: r.fix()?,
            stale_frames: r.u32()?,
        });
    }
    let mut cache = ContactCache::new();
    cache.manifolds = manifolds;
    cache.max_stale_frames = r.u32()?;
    cache.warm_start_factor = r.fix()?;
    // Rebuilt (not saved): the same index `ContactCache::end_frame` builds.
    #[cfg(feature = "std")]
    {
        cache.pair_index.clear();
        for (idx, m) in cache.manifolds.iter().enumerate() {
            cache.pair_index.insert(m.pair, idx);
        }
    }
    Ok(cache)
}

const fn combine_tag(c: CombineRule) -> u8 {
    match c {
        CombineRule::Average => 0,
        CombineRule::Min => 1,
        CombineRule::Max => 2,
        CombineRule::Multiply => 3,
    }
}

fn r_combine(r: &mut R<'_>) -> Res<CombineRule> {
    match r.u8()? {
        0 => Ok(CombineRule::Average),
        1 => Ok(CombineRule::Min),
        2 => Ok(CombineRule::Max),
        3 => Ok(CombineRule::Multiply),
        _ => Err(invalid("material_table")),
    }
}

fn w_material_table(w: &mut W, t: &MaterialTable) {
    w.usize(t.materials.len());
    for m in &t.materials {
        w.u16(m.id);
        w.fix(m.static_friction);
        w.fix(m.dynamic_friction);
        w.fix(m.restitution);
        w.u8(combine_tag(m.friction_combine));
        w.u8(combine_tag(m.restitution_combine));
    }
    w.usize(t.pair_overrides.len());
    for p in &t.pair_overrides {
        w.u16(p.mat_a);
        w.u16(p.mat_b);
        w.fix(p.friction);
        w.fix(p.restitution);
    }
    w.u8(combine_tag(t.default_friction_combine));
    w.u8(combine_tag(t.default_restitution_combine));
}

/// Reads the material table and rejects any table the `MaterialTable` API
/// cannot build: `MaterialTable::new` puts the default material at id 0 and
/// `try_register` assigns `id = len` up to 65,536 ids, so a valid table holds
/// `1..=65_536` materials with `materials[i].id == i` (`get` indexes by
/// position and falls back to entry 0, so an empty table panics there).
fn r_material_table(r: &mut R<'_>) -> Res<MaterialTable> {
    const S: &str = "material_table";
    let n = r.len(S, 52)?;
    // `n > 65_536` is also caught by the id check below (an id is a `u16`);
    // checking it here rejects before reading the entries.
    if n == 0 || n > MATERIAL_ID_CAPACITY {
        return Err(invalid(S));
    }
    let mut materials = Vec::with_capacity(n);
    for i in 0..n {
        let id = r.u16()?;
        if usize::from(id) != i {
            return Err(invalid(S));
        }
        materials.push(PhysicsMaterial {
            id,
            static_friction: r.fix()?,
            dynamic_friction: r.fix()?,
            restitution: r.fix()?,
            friction_combine: r_combine(r)?,
            restitution_combine: r_combine(r)?,
        });
    }
    let n = r.len(S, 36)?;
    let mut pair_overrides = Vec::with_capacity(n);
    for _ in 0..n {
        pair_overrides.push(PairOverride {
            mat_a: r.u16()?,
            mat_b: r.u16()?,
            friction: r.fix()?,
            restitution: r.fix()?,
        });
    }
    let mut t = MaterialTable::new();
    t.materials = materials;
    t.pair_overrides = pair_overrides;
    t.default_friction_combine = r_combine(r)?;
    t.default_restitution_combine = r_combine(r)?;
    Ok(t)
}

fn w_pair(w: &mut W, p: (usize, usize)) {
    w.usize(p.0);
    w.usize(p.1);
}

fn r_pair(r: &mut R<'_>) -> Res<(usize, usize)> {
    Ok((r.usize("events")?, r.usize("events")?))
}

fn w_events(w: &mut W, e: &EventCollector) {
    w.usize(e.contact_events.len());
    for c in &e.contact_events {
        w.usize(c.body_a);
        w.usize(c.body_b);
        w.u8(match c.event_type {
            ContactEventType::Begin => 0,
            ContactEventType::Persist => 1,
            ContactEventType::End => 2,
        });
        w.vec3(c.normal);
        w.vec3(c.point);
        w.fix(c.depth);
        w.fix(c.relative_velocity);
    }
    w.usize(e.trigger_events.len());
    for t in &e.trigger_events {
        w.usize(t.trigger_body);
        w.usize(t.other_body);
        w.bool(t.entered);
    }
    w.usize(e.prev_pairs.len());
    for &p in &e.prev_pairs {
        w_pair(w, p);
    }
    w.usize(e.curr_pairs.len());
    for &p in &e.curr_pairs {
        w_pair(w, p);
    }
    w.usize(e.prev_triggers.len());
    for &(k, v) in &e.prev_triggers {
        w_pair(w, k);
        w_pair(w, v);
    }
    w.usize(e.curr_triggers.len());
    for (&k, &v) in &e.curr_triggers {
        w_pair(w, k);
        w_pair(w, v);
    }
}

fn r_events(r: &mut R<'_>) -> Res<EventCollector> {
    const S: &str = "events";
    let mut e = EventCollector::new();
    let n = r.len(S, 145)?;
    for _ in 0..n {
        e.contact_events.push(ContactEvent {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            event_type: match r.u8()? {
                0 => ContactEventType::Begin,
                1 => ContactEventType::Persist,
                2 => ContactEventType::End,
                _ => return Err(invalid(S)),
            },
            normal: r.vec3()?,
            point: r.vec3()?,
            depth: r.fix()?,
            relative_velocity: r.fix()?,
        });
    }
    let n = r.len(S, 17)?;
    for _ in 0..n {
        e.trigger_events.push(TriggerEvent {
            trigger_body: r.usize(S)?,
            other_body: r.usize(S)?,
            entered: r.bool(S)?,
        });
    }
    let n = r.len(S, 16)?;
    for _ in 0..n {
        e.prev_pairs.push(r_pair(r)?);
    }
    let n = r.len(S, 16)?;
    let mut curr_pairs = BTreeSet::new();
    for _ in 0..n {
        curr_pairs.insert(r_pair(r)?);
    }
    e.curr_pairs = curr_pairs;
    let n = r.len(S, 32)?;
    for _ in 0..n {
        let k = r_pair(r)?;
        let v = r_pair(r)?;
        e.prev_triggers.push((k, v));
    }
    let n = r.len(S, 32)?;
    let mut curr_triggers = BTreeMap::new();
    for _ in 0..n {
        let k = r_pair(r)?;
        let v = r_pair(r)?;
        curr_triggers.insert(k, v);
    }
    e.curr_triggers = curr_triggers;
    Ok(e)
}

fn w_tree(w: &mut W, t: &crate::dynamic_bvh::DynamicAabbTree) {
    w.usize(t.nodes.len());
    for n in &t.nodes {
        w.aabb(n.aabb);
        w.u32(n.parent);
        w.u32(n.left);
        w.u32(n.right);
        w.i32(n.height);
        w.u32(n.user_data);
        w.bool(n.is_leaf);
    }
    w.usize(t.free_list.len());
    for &f in &t.free_list {
        w.u32(f);
    }
    w.u32(t.root);
    w.fix(t.margin);
    let (l1, l2, linf) = t.metric.weights();
    w.fix(l1);
    w.fix(l2);
    w.fix(linf);
}

fn r_tree(r: &mut R<'_>) -> Res<crate::dynamic_bvh::DynamicAabbTree> {
    const S: &str = "broadphase_tree";
    let mut t = crate::dynamic_bvh::DynamicAabbTree::new();
    let n = r.len(S, 117)?;
    for _ in 0..n {
        t.nodes.push(crate::dynamic_bvh::DynamicNode {
            aabb: r.aabb()?,
            parent: r.u32()?,
            left: r.u32()?,
            right: r.u32()?,
            height: r.i32()?,
            user_data: r.u32()?,
            is_leaf: r.bool(S)?,
        });
    }
    let n = r.len(S, 4)?;
    for _ in 0..n {
        t.free_list.push(r.u32()?);
    }
    t.root = r.u32()?;
    t.margin = r.fix()?;
    let (l1, l2, linf) = (r.fix()?, r.fix()?, r.fix()?);
    t.metric = crate::metric::MetricWeights::new(l1, l2, linf).map_err(|_| invalid(S))?;
    Ok(t)
}

// ── Whole world ───────────────────────────────────────────────────────────

/// Everything decoded from a blob, before it is moved into a world.
struct Decoded {
    config: SolverConfig,
    bodies: Vec<RigidBody>,
    distance_constraints: Vec<DistanceConstraint>,
    contact_constraints: Vec<ContactConstraint>,
    /// position, rotation, scale, body index, inv_rotation, scale_f32, inv_scale_f32
    sdf_poses: Vec<(Vec3Fix, QuatFix, Fix128, usize, QuatFix, f32, f32)>,
    sdf_collision_radius: Fix128,
    static_colliders: Vec<StaticCollider>,
    constraint_batches: Vec<ConstraintBatch>,
    batches_dirty: bool,
    batch_static_bodies: Vec<bool>,
    contact_cache: ContactCache,
    material_table: MaterialTable,
    body_materials: Vec<u16>,
    pre_solve_hooks: usize,
    contact_modifiers: usize,
    gpu_solver_bridge: usize,
    joints: Vec<Joint>,
    force_fields: Vec<ForceFieldInstance>,
    events: EventCollector,
    sleep_config: SleepConfig,
    sleep_data: Vec<SleepData>,
    body_collision_radii: Vec<Option<Fix128>>,
    broadphase: Broadphase,
    broadphase_tree: crate::dynamic_bvh::DynamicAabbTree,
    broadphase_proxies: Vec<Option<u32>>,
    body_colliders: Vec<Option<BodyCollider>>,
    body_filters: Vec<CollisionFilter>,
    overflow_detected: bool,
    /// (entries sorted by id, hits, misses)
    tgs_cache: (Vec<(u64, [Fix128; 3])>, u64, u64),
    /// (kind, payload) per participant, registration order (version 2)
    participants: Vec<(u32, Vec<u8>)>,
    /// The recorded fault (version 2)
    fault: Option<WorldFault>,
    /// The bytes of `FieldBoard::write_values` (version 2; an empty board for
    /// version 1)
    fields: Vec<u8>,
    /// The continuous collision setting (version 3; off for versions 1 and 2)
    ccd: super::WorldCcdConfig,
}

/// Encoded tag of a static SDF collider's body index ([`crate::sdf_collider::SDF_STATIC`]),
/// kept apart from body indices so the blob is the same on 32- and 64-bit targets.
const SDF_STATIC_TAG: u8 = 0;

impl PhysicsWorld {
    /// Magic of a [`Self::snapshot_world`] blob (`b"APWS"`, distinct from
    /// [`Self::STATE_MAGIC`] of the rollback blob).
    pub const WORLD_SNAPSHOT_MAGIC: [u8; 4] = *b"APWS";

    /// Format version of a [`Self::snapshot_world`] blob. Version 1 blobs
    /// (no `participants`, `fault` or `fields` section) are still read, as a
    /// world without participants, fault or fields, and version 1 and 2 blobs
    /// (no `continuous_collision` section) as a world with continuous
    /// collision off; a blob of any other version is rejected with
    /// [`WorldSnapshotError::UnsupportedVersion`].
    pub const WORLD_SNAPSHOT_VERSION: u16 = 3;

    /// Write every piece of state [`Self::step`] reads into one versioned blob
    /// with a checksum.
    ///
    /// Restore it with [`Self::from_world_snapshot`] (into a new world) or
    /// [`Self::restore_world`] (into an existing one, keeping its SDF fields
    /// and callbacks). Every later `step` of the restored world is
    /// bit-identical to the original's.
    ///
    /// # Examples
    ///
    /// ```
    /// use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
    ///
    /// let mut world = PhysicsWorld::new(PhysicsConfig::default());
    /// world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE));
    /// let dt = Fix128::from_ratio(1, 60);
    /// world.step_n(10, dt);
    ///
    /// let blob = world.snapshot_world();
    /// let mut branch = PhysicsWorld::from_world_snapshot(&blob).unwrap();
    /// world.step_n(30, dt);
    /// branch.step_n(30, dt);
    /// assert_eq!(world.snapshot_world(), branch.snapshot_world());
    /// ```
    ///
    /// # Field coverage
    ///
    /// Every field of [`PhysicsWorld`], classified. "Saved" fields are written
    /// byte-exact; "rebuilt" fields are derived on restore from saved ones;
    /// "not covered" fields cannot be written as data, and restore instead
    /// requires the target world to already hold the same number of them.
    ///
    /// | field | class | how / why |
    /// |---|---|---|
    /// | `config` | saved | all of [`SolverConfig`] |
    /// | `bodies` | saved | every [`RigidBody`] field, including `prev_*`, damping, `kinematic_target` |
    /// | `distance_constraints` | saved | including `cached_lambda` |
    /// | `contact_constraints` | saved | including `cached_lambda` |
    /// | `sdf_colliders` (pose) | saved | `position` / `rotation` / `scale` / `body_index` and the cached `inv_rotation` / `scale_f32` / `inv_scale_f32` as stored (not recomputed) |
    /// | `sdf_colliders` (`field`) | not covered | `Box<dyn SdfField>` is code; the target world keeps its own fields, and the count must match ([`WorldSnapshotError::SdfFieldCountMismatch`]) |
    /// | `sdf_collision_radius` | saved | |
    /// | `static_colliders` | saved | plane / height field / triangle mesh including its BVH nodes as stored |
    /// | `constraint_batches` / `batches_dirty` / `batch_static_bodies` | saved | as stored, so a stale-but-clean batch set is reproduced rather than recomputed |
    /// | `contact_cache` (manifolds, `max_stale_frames`, `warm_start_factor`) | saved | |
    /// | `contact_cache` (`pair_index`, std) | rebuilt | from the manifolds in order, the same way `ContactCache::end_frame` rebuilds it |
    /// | `material_table` / `body_materials` | saved | materials, pair overrides, default combine rules |
    /// | `pre_solve_hooks` / `contact_modifiers` (std) | not covered | closures / trait objects; the count is recorded and must match ([`WorldSnapshotError::CallbackCountMismatch`]) |
    /// | `gpu_solver_bridge` (feature `gpu-solver-bridge`) | not covered | trait object; "installed or not" is recorded and must match |
    /// | `joints` | saved | all seven kinds, every field |
    /// | `joint_motors` / `joint_motors_3d` | not covered | the restore target keeps its own motors (add them with [`PhysicsWorld::add_joint_motor`] / [`PhysicsWorld::add_joint_motor_3d`] before restoring), as the format predates them; a motor's gains and targets that change between frames are saved next to the snapshot |
    /// | `force_fields` | saved | field, `affected_bodies`, `enabled` |
    /// | `events` | saved | this frame's events and the previous / current pair and trigger sets that decide begin / persist / end |
    /// | `islands` (`sleep_data`, `config`) | saved | |
    /// | `islands` (union-find `parent` / `rank`) | rebuilt | `IslandManager::new` plus a union of every joint's bodies, the same as the start of every `step` |
    /// | `body_collision_radii` / `body_filters` | saved | as stored, including their own lengths |
    /// | `broadphase` / `broadphase_tree` / `broadphase_proxies` | saved | the persistent tree node by node (pair order follows the tree layout) |
    /// | `broadphase_hybrid` | rebuilt | empty; its pairs are a pure function of the bodies staged each substep, so a fresh one gives the same pairs |
    /// | `body_colliders` | saved | shape or compound, including the compound's cached AABB and dirty flag |
    /// | `overflow_detected` | saved | sticky flag |
    /// | `kinematic_substeps_left` | rebuilt | always 0 outside `step` (reset at the end of every step), and a snapshot is only taken between steps |
    /// | `tgs_impulse_cache` (std) | saved | entries sorted by id, hit / miss counters; the live set is rebuilt empty (the `sweep` at the end of every step empties it) |
    /// | `participants` (std) | saved / not covered | each participant's kind and its [`crate::world_participant::Participant::write_state`] payload are saved; the participant itself is code, so the target world must hold the same kinds in the same order ([`WorldSnapshotError::ParticipantMismatch`]), and every payload is checked before any is read ([`WorldSnapshotError::ParticipantState`]) |
    /// | `participant_plan` (std) | not covered | derived from the target world's own participants and fields |
    /// | `fault` | saved | the recorded fault, so a restored branch is still faulted |
    /// | `fields` | saved | the committed values once ([`crate::world_participant::FieldBoard::write_values`]); the target world must declare the same fields ([`WorldSnapshotError::FieldState`]) |
    /// | `ccd` | saved | [`PhysicsWorld::continuous_collision`] (version 3) |
    ///
    /// Not in the table because they are not [`PhysicsWorld`] fields:
    /// motors the caller applies itself with [`crate::motor::apply_motors`]
    /// and [`crate::character::CharacterController`] are owned by the caller and
    /// are applied from outside `step`, and cloth / fluid / FEM / vehicle state
    /// lives in its own types. Save them next to the world snapshot.
    ///
    // LIMITATION(COV-ENGINE-012): Known difference: the union-find of `islands` is rebuilt from the joints.
    /// Known difference: the union-find of `islands` is rebuilt from the joints. After a
    /// `remove_joint` between two steps, the original still carries that union
    /// until the next `step` resets it, while the restored world does not; the
    /// two differ only for [`PhysicsWorld::wake_body`] called before that next
    /// step. The structure is private to [`crate::sleeping`] and has no
    /// non-mutating accessor, so it cannot be copied from `&self`.
    ///
    /// # Format (version 3)
    ///
    /// | range | content |
    /// |---|---|
    /// | `[0..4)` | magic [`PhysicsWorld::WORLD_SNAPSHOT_MAGIC`] (`b"APWS"`) |
    /// | `[4..6)` | version u16 = [`PhysicsWorld::WORLD_SNAPSHOT_VERSION`] |
    /// | `[6..8)` | reserved u16 = 0 |
    /// | `[8..16)` | payload length u64 |
    /// | `[16..16+len)` | payload (the sections in the table above, little endian) |
    /// | last 8 | FNV-1a 64 of every preceding byte |
    ///
    /// Version 2 appends three sections to the version 1 payload, after the
    /// TGS cache, in this order:
    ///
    /// | section | content |
    /// |---|---|
    /// | `participants` | `count: u64`, then per participant `kind: u32`, `payload_len: u64`, payload |
    /// | `fault` | `u8` code: `0` none, `1` participant (`index: u64`, `kind: u32`, fault tag `u8`), `2` rigid overflow, `3` force out of range (`body: u64`), `4` field out of range (`field: u32`, `index: u64`) |
    /// | `fields` | [`crate::world_participant::FieldBoard::write_values`] |
    ///
    /// Version 3 appends one section after `fields`:
    ///
    /// | section | content |
    /// |---|---|
    /// | `continuous_collision` | `enabled: u8` (`0` / `1`), `motion_threshold` ([`PhysicsWorld::continuous_collision`]) |
    ///
    /// The version is raised for it because the reader rejects trailing
    /// bytes: a section appended to a version 2 blob could not be told apart
    /// from a corrupt one.
    #[must_use]
    pub fn snapshot_world(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(&Self::WORLD_SNAPSHOT_MAGIC);
        w.u16(Self::WORLD_SNAPSHOT_VERSION);
        w.u16(0);
        w.u64(0); // payload length, patched below

        w_config(&mut w, &self.config);

        w.usize(self.bodies.len());
        for b in &self.bodies {
            w_body(&mut w, b);
        }

        w.usize(self.distance_constraints.len());
        for c in &self.distance_constraints {
            w.usize(c.body_a);
            w.usize(c.body_b);
            w.vec3(c.local_anchor_a);
            w.vec3(c.local_anchor_b);
            w.fix(c.target_distance);
            w.fix(c.compliance);
            w.fix(c.cached_lambda);
        }

        w.usize(self.contact_constraints.len());
        for c in &self.contact_constraints {
            w.usize(c.body_a);
            w.usize(c.body_b);
            w_contact(&mut w, &c.contact);
            w.fix(c.friction);
            w.fix(c.restitution);
            w.fix(c.cached_lambda);
        }

        w.usize(self.sdf_colliders.len());
        for s in &self.sdf_colliders {
            w.vec3(s.position);
            w.quat(s.rotation);
            w.fix(s.scale);
            if s.body_index == crate::sdf_collider::SDF_STATIC {
                w.u8(SDF_STATIC_TAG);
            } else {
                w.u8(1);
                w.usize(s.body_index);
            }
            w.quat(s.inv_rotation);
            w.f32(s.scale_f32);
            w.f32(s.inv_scale_f32);
        }
        w.fix(self.sdf_collision_radius);

        w.usize(self.static_colliders.len());
        for c in &self.static_colliders {
            w_static_collider(&mut w, c);
        }

        w.usize(self.constraint_batches.len());
        for b in &self.constraint_batches {
            w.usize(b.distance_indices.len());
            for &i in &b.distance_indices {
                w.usize(i);
            }
            w.usize(b.contact_indices.len());
            for &i in &b.contact_indices {
                w.usize(i);
            }
        }
        w.bool(self.batches_dirty);
        w.usize(self.batch_static_bodies.len());
        for &s in &self.batch_static_bodies {
            w.bool(s);
        }

        w_contact_cache(&mut w, &self.contact_cache);
        w_material_table(&mut w, &self.material_table);
        w.usize(self.body_materials.len());
        for &m in &self.body_materials {
            w.u16(m);
        }

        // Callbacks are code: only their count is recorded.
        #[cfg(feature = "std")]
        {
            w.usize(self.pre_solve_hooks.len());
            w.usize(self.contact_modifiers.len());
        }
        #[cfg(not(feature = "std"))]
        {
            w.usize(0);
            w.usize(0);
        }
        #[cfg(feature = "gpu-solver-bridge")]
        w.usize(usize::from(self.gpu_solver_bridge.is_some()));
        #[cfg(not(feature = "gpu-solver-bridge"))]
        w.usize(0);

        w.usize(self.joints.len());
        for j in &self.joints {
            w_joint(&mut w, j);
        }

        w.usize(self.force_fields.len());
        for f in &self.force_fields {
            w_force_field(&mut w, f);
        }

        w_events(&mut w, &self.events);

        w.fix(self.islands.config.linear_threshold);
        w.fix(self.islands.config.angular_threshold);
        w.u32(self.islands.config.frames_to_sleep);
        w.usize(self.islands.sleep_data.len());
        for sd in &self.islands.sleep_data {
            w.u8(match sd.state {
                SleepState::Awake => 0,
                SleepState::Sleeping => 1,
            });
            w.u32(sd.idle_frames);
        }

        w.usize(self.body_collision_radii.len());
        for &r in &self.body_collision_radii {
            w.opt_fix(r);
        }

        w.u8(match self.broadphase {
            Broadphase::Bvh => 0,
            Broadphase::DynamicTree => 1,
            Broadphase::Hybrid => 2,
        });
        w_tree(&mut w, &self.broadphase_tree);
        w.usize(self.broadphase_proxies.len());
        for &p in &self.broadphase_proxies {
            match p {
                Some(id) => {
                    w.u8(1);
                    w.u32(id);
                }
                None => w.u8(0),
            }
        }

        w.usize(self.body_colliders.len());
        for c in &self.body_colliders {
            w_body_collider(&mut w, c.as_ref());
        }

        w.usize(self.body_filters.len());
        for f in &self.body_filters {
            w.u32(f.layer);
            w.u32(f.mask);
            w.u32(f.group);
        }

        w.bool(self.overflow_detected);

        #[cfg(feature = "std")]
        {
            let cache = &self.tgs_impulse_cache;
            let mut entries: Vec<_> = cache.entries.iter().map(|(&k, &v)| (k, v)).collect();
            entries.sort_unstable_by_key(|e| e.0);
            w.usize(entries.len());
            for (id, imp) in entries {
                w.u64(id);
                w.fix(imp.normal);
                w.fix(imp.tangent1);
                w.fix(imp.tangent2);
            }
            w.u64(cache.hits);
            w.u64(cache.misses);
        }
        #[cfg(not(feature = "std"))]
        {
            w.usize(0);
            w.u64(0);
            w.u64(0);
        }

        #[cfg(feature = "std")]
        let participants = {
            let mut out = Vec::new();
            self.write_participants(&mut out);
            out
        };
        #[cfg(not(feature = "std"))]
        let participants: Vec<(u32, Vec<u8>)> = Vec::new();
        w.usize(participants.len());
        for (kind, payload) in &participants {
            w.u32(*kind);
            w.usize(payload.len());
            w.0.extend_from_slice(payload);
        }
        w_fault(&mut w, self.fault());
        self.fields().write_values(&mut w.0);
        w.bool(self.ccd.is_enabled());
        w.fix(self.ccd.motion_threshold());

        let mut data = w.0;
        let payload_len = (data.len() - HEADER_LEN) as u64;
        data[8..16].copy_from_slice(&payload_len.to_le_bytes());
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        fnv1a_fold(&mut hash, &data);
        data.extend_from_slice(&hash.to_le_bytes());
        data
    }

    /// Build a new world from a [`Self::snapshot_world`] blob.
    ///
    /// The new world holds no SDF fields, hooks, contact modifiers or GPU
    /// bridge, so a snapshot taken with any of those is rejected
    /// ([`WorldSnapshotError::SdfFieldCountMismatch`] /
    /// [`WorldSnapshotError::CallbackCountMismatch`]); use
    /// [`Self::restore_world`] on a world that holds them instead.
    ///
    /// # Errors
    ///
    /// See [`WorldSnapshotError`] for which input gives which error.
    pub fn from_world_snapshot(data: &[u8]) -> Result<Self, WorldSnapshotError> {
        let mut world = Self::new(SolverConfig::default());
        world.restore_world(data)?;
        Ok(world)
    }

    /// Replace this world's state with a [`Self::snapshot_world`] blob.
    ///
    /// Everything in the blob replaces the current state, including the
    /// [`SolverConfig`] and the body population. What is not data stays:
    /// the SDF fields of `self.sdf_colliders` (paired with the blob's poses in
    /// order), the pre-solve hooks, contact modifiers and the installed GPU
    /// bridge; their counts must equal the snapshot's.
    ///
    /// # Errors
    ///
    /// See [`WorldSnapshotError`] for which input gives which error. On error `self` is unchanged.
    pub fn restore_world(&mut self, data: &[u8]) -> Result<(), WorldSnapshotError> {
        let d = decode(data)?;
        self.check_attachable(&d)?;
        self.check_participants(&d.participants)?;
        self.fields()
            .check_values(&d.fields)
            .map_err(WorldSnapshotError::FieldState)?;
        self.install(d);
        Ok(())
    }

    fn check_attachable(&self, d: &Decoded) -> Res<()> {
        if d.sdf_poses.len() != self.sdf_colliders.len() {
            return Err(WorldSnapshotError::SdfFieldCountMismatch {
                snapshot: d.sdf_poses.len(),
                world: self.sdf_colliders.len(),
            });
        }
        #[cfg(feature = "std")]
        let (hooks, modifiers) = (self.pre_solve_hooks.len(), self.contact_modifiers.len());
        #[cfg(not(feature = "std"))]
        let (hooks, modifiers) = (0usize, 0usize);
        #[cfg(feature = "gpu-solver-bridge")]
        let bridge = usize::from(self.gpu_solver_bridge.is_some());
        #[cfg(not(feature = "gpu-solver-bridge"))]
        let bridge = 0usize;
        for (section, snapshot, world) in [
            ("pre_solve_hooks", d.pre_solve_hooks, hooks),
            ("contact_modifiers", d.contact_modifiers, modifiers),
            ("gpu_solver_bridge", d.gpu_solver_bridge, bridge),
        ] {
            if snapshot != world {
                return Err(WorldSnapshotError::CallbackCountMismatch {
                    section,
                    snapshot,
                    world,
                });
            }
        }
        Ok(())
    }

    fn install(&mut self, d: Decoded) {
        self.config = d.config;
        self.bodies = d.bodies;
        self.distance_constraints = d.distance_constraints;
        self.contact_constraints = d.contact_constraints;
        for (s, p) in self.sdf_colliders.iter_mut().zip(d.sdf_poses) {
            s.position = p.0;
            s.rotation = p.1;
            s.scale = p.2;
            s.body_index = p.3;
            s.inv_rotation = p.4;
            s.scale_f32 = p.5;
            s.inv_scale_f32 = p.6;
        }
        self.sdf_collision_radius = d.sdf_collision_radius;
        self.static_colliders = d.static_colliders;
        self.constraint_batches = d.constraint_batches;
        self.batches_dirty = d.batches_dirty;
        self.batch_static_bodies = d.batch_static_bodies;
        self.contact_cache = d.contact_cache;
        self.material_table = d.material_table;
        self.body_materials = d.body_materials;
        self.joints = d.joints;
        self.force_fields = d.force_fields;
        self.events = d.events;

        // Rebuilt: the union-find, as at the start of every `step`.
        let mut islands = IslandManager::new(d.sleep_data.len(), d.sleep_config);
        for j in &self.joints {
            let (a, b) = j.bodies();
            if a < d.sleep_data.len() && b < d.sleep_data.len() {
                islands.union(a, b);
            }
        }
        islands.sleep_data = d.sleep_data;
        self.islands = islands;

        self.body_collision_radii = d.body_collision_radii;
        self.broadphase = d.broadphase;
        self.broadphase_tree = d.broadphase_tree;
        self.broadphase_proxies = d.broadphase_proxies;
        self.broadphase_hybrid = crate::bvh::BroadphaseHybrid::new();
        self.body_colliders = d.body_colliders;
        self.body_filters = d.body_filters;
        self.overflow_detected = d.overflow_detected;
        self.kinematic_substeps_left = 0;

        #[cfg(feature = "std")]
        {
            let (entries, hits, misses) = d.tgs_cache;
            let mut cache = crate::solver_tgs::ImpulseCache::new();
            for (id, [n, t1, t2]) in entries {
                cache.entries.insert(
                    id,
                    crate::solver_tgs::CachedImpulse {
                        normal: n,
                        tangent1: t1,
                        tangent2: t2,
                    },
                );
            }
            // Rebuilt: the live set is empty between steps (`sweep` clears it).
            cache.hits = hits;
            cache.misses = misses;
            self.tgs_impulse_cache = cache;
        }
        #[cfg(not(feature = "std"))]
        let _ = d.tgs_cache; // the TGS backend falls back to XPBD without std

        // Checked by `restore_world` before anything was installed.
        self.read_participants(&d.participants);
        self.set_fault(d.fault);
        self.fields_mut().read_values(&d.fields);
        self.ccd = d.ccd;
    }
}

fn decode(data: &[u8]) -> Res<Decoded> {
    if data.len() < HEADER_LEN + CHECKSUM_LEN {
        return Err(WorldSnapshotError::Truncated);
    }
    if data[0..4] != PhysicsWorld::WORLD_SNAPSHOT_MAGIC {
        return Err(WorldSnapshotError::BadMagic);
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if !(1..=PhysicsWorld::WORLD_SNAPSHOT_VERSION).contains(&version) {
        return Err(WorldSnapshotError::UnsupportedVersion {
            found: version,
            supported: PhysicsWorld::WORLD_SNAPSHOT_VERSION,
        });
    }
    if data[6] != 0 || data[7] != 0 {
        return Err(WorldSnapshotError::ReservedNotZero);
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[8..16]);
    let payload_len = u64::from_le_bytes(len_bytes);
    let available = (data.len() - HEADER_LEN - CHECKSUM_LEN) as u64;
    if payload_len > available {
        return Err(WorldSnapshotError::Truncated);
    }
    if payload_len < available {
        return Err(WorldSnapshotError::TrailingBytes {
            extra: (available - payload_len) as usize,
        });
    }
    let body_end = data.len() - CHECKSUM_LEN;
    let mut stored = [0u8; 8];
    stored.copy_from_slice(&data[body_end..]);
    let stored = u64::from_le_bytes(stored);
    let mut computed: u64 = 0xcbf2_9ce4_8422_2325;
    fnv1a_fold(&mut computed, &data[..body_end]);
    if stored != computed {
        return Err(WorldSnapshotError::ChecksumMismatch { stored, computed });
    }

    let mut r = R {
        data: &data[..body_end],
        pos: HEADER_LEN,
    };
    let d = decode_payload(&mut r, version)?;
    if r.pos != body_end {
        return Err(WorldSnapshotError::TrailingBytes {
            extra: body_end - r.pos,
        });
    }
    validate(&d)?;
    Ok(d)
}

fn decode_payload(r: &mut R<'_>, version: u16) -> Res<Decoded> {
    let config = r_config(r)?;

    let n = r.len("bodies", 1)?;
    let mut bodies = Vec::with_capacity(n);
    for _ in 0..n {
        bodies.push(r_body(r)?);
    }

    let n = r.len("distance_constraints", 160)?;
    let mut distance_constraints = Vec::with_capacity(n);
    for _ in 0..n {
        const S: &str = "distance_constraints";
        distance_constraints.push(DistanceConstraint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            local_anchor_a: r.vec3()?,
            local_anchor_b: r.vec3()?,
            target_distance: r.fix()?,
            compliance: r.fix()?,
            cached_lambda: r.fix()?,
        });
    }

    let n = r.len("contact_constraints", 208)?;
    let mut contact_constraints = Vec::with_capacity(n);
    for _ in 0..n {
        const S: &str = "contact_constraints";
        contact_constraints.push(ContactConstraint {
            body_a: r.usize(S)?,
            body_b: r.usize(S)?,
            contact: r_contact(r)?,
            friction: r.fix()?,
            restitution: r.fix()?,
            cached_lambda: r.fix()?,
        });
    }

    let n = r.len("sdf_colliders", 193)?;
    let mut sdf_poses = Vec::with_capacity(n);
    for _ in 0..n {
        const S: &str = "sdf_colliders";
        let position = r.vec3()?;
        let rotation = r.quat()?;
        let scale = r.fix()?;
        let body_index = match r.u8()? {
            SDF_STATIC_TAG => crate::sdf_collider::SDF_STATIC,
            1 => r.usize(S)?,
            _ => return Err(invalid(S)),
        };
        let inv_rotation = r.quat()?;
        let scale_f32 = r.f32()?;
        let inv_scale_f32 = r.f32()?;
        sdf_poses.push((
            position,
            rotation,
            scale,
            body_index,
            inv_rotation,
            scale_f32,
            inv_scale_f32,
        ));
    }
    let sdf_collision_radius = r.fix()?;

    let n = r.len("static_colliders", 1)?;
    let mut static_colliders = Vec::with_capacity(n);
    for _ in 0..n {
        static_colliders.push(r_static_collider(r)?);
    }

    let n = r.len("constraint_batches", 16)?;
    let mut constraint_batches = Vec::with_capacity(n);
    for _ in 0..n {
        const S: &str = "constraint_batches";
        let k = r.len(S, 8)?;
        let mut distance_indices = Vec::with_capacity(k);
        for _ in 0..k {
            distance_indices.push(r.usize(S)?);
        }
        let k = r.len(S, 8)?;
        let mut contact_indices = Vec::with_capacity(k);
        for _ in 0..k {
            contact_indices.push(r.usize(S)?);
        }
        constraint_batches.push(ConstraintBatch {
            distance_indices,
            contact_indices,
        });
    }
    let batches_dirty = r.bool("constraint_batches")?;
    let n = r.len("batch_static_bodies", 1)?;
    let mut batch_static_bodies = Vec::with_capacity(n);
    for _ in 0..n {
        batch_static_bodies.push(r.bool("batch_static_bodies")?);
    }

    let contact_cache = r_contact_cache(r)?;
    let material_table = r_material_table(r)?;
    let n = r.len("body_materials", 2)?;
    let mut body_materials = Vec::with_capacity(n);
    for _ in 0..n {
        body_materials.push(r.u16()?);
    }

    let pre_solve_hooks = r.usize("pre_solve_hooks")?;
    let contact_modifiers = r.usize("contact_modifiers")?;
    let gpu_solver_bridge = r.usize("gpu_solver_bridge")?;

    let n = r.len("joints", 1)?;
    let mut joints = Vec::with_capacity(n);
    for _ in 0..n {
        joints.push(r_joint(r)?);
    }

    let n = r.len("force_fields", 1)?;
    let mut force_fields = Vec::with_capacity(n);
    for _ in 0..n {
        force_fields.push(r_force_field(r)?);
    }

    let events = r_events(r)?;

    let sleep_config = SleepConfig {
        linear_threshold: r.fix()?,
        angular_threshold: r.fix()?,
        frames_to_sleep: r.u32()?,
    };
    let n = r.len("islands", 5)?;
    let mut sleep_data = Vec::with_capacity(n);
    for _ in 0..n {
        let state = match r.u8()? {
            0 => SleepState::Awake,
            1 => SleepState::Sleeping,
            _ => return Err(invalid("islands")),
        };
        sleep_data.push(SleepData {
            state,
            idle_frames: r.u32()?,
        });
    }

    let n = r.len("body_collision_radii", 1)?;
    let mut body_collision_radii = Vec::with_capacity(n);
    for _ in 0..n {
        body_collision_radii.push(r.opt_fix("body_collision_radii")?);
    }

    let broadphase = match r.u8()? {
        0 => Broadphase::Bvh,
        1 => Broadphase::DynamicTree,
        2 => Broadphase::Hybrid,
        _ => return Err(invalid("broadphase")),
    };
    let broadphase_tree = r_tree(r)?;
    let n = r.len("broadphase_proxies", 1)?;
    let mut broadphase_proxies = Vec::with_capacity(n);
    for _ in 0..n {
        broadphase_proxies.push(if r.bool("broadphase_proxies")? {
            Some(r.u32()?)
        } else {
            None
        });
    }

    let n = r.len("body_colliders", 1)?;
    let mut body_colliders = Vec::with_capacity(n);
    for _ in 0..n {
        body_colliders.push(r_body_collider(r)?);
    }

    let n = r.len("body_filters", 12)?;
    let mut body_filters = Vec::with_capacity(n);
    for _ in 0..n {
        body_filters.push(CollisionFilter {
            layer: r.u32()?,
            mask: r.u32()?,
            group: r.u32()?,
        });
    }

    let overflow_detected = r.bool("overflow_detected")?;

    let n = r.len("tgs_impulse_cache", 56)?;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        entries.push((r.u64()?, [r.fix()?, r.fix()?, r.fix()?]));
    }
    let tgs_cache = (entries, r.u64()?, r.u64()?);

    let (participants, fault, fields) = if version >= 2 {
        let n = r.len("participants", 12)?;
        let mut participants = Vec::with_capacity(n);
        for _ in 0..n {
            let kind = r.u32()?;
            let len = r.len("participants", 1)?;
            participants.push((kind, r.take(len)?.to_vec()));
        }
        (participants, r_fault(r)?, r_fields(r)?)
    } else {
        let mut empty = Vec::new();
        crate::world_participant::FieldBoard::new().write_values(&mut empty);
        (Vec::new(), None, empty)
    };
    let ccd = if version >= 3 {
        super::WorldCcdConfig::new()
            .with_enabled(r.bool("continuous_collision")?)
            .with_motion_threshold(r.fix()?)
    } else {
        super::WorldCcdConfig::new()
    };

    Ok(Decoded {
        config,
        bodies,
        distance_constraints,
        contact_constraints,
        sdf_poses,
        sdf_collision_radius,
        static_colliders,
        constraint_batches,
        batches_dirty,
        batch_static_bodies,
        contact_cache,
        material_table,
        body_materials,
        pre_solve_hooks,
        contact_modifiers,
        gpu_solver_bridge,
        joints,
        force_fields,
        events,
        sleep_config,
        sleep_data,
        body_collision_radii,
        broadphase,
        broadphase_tree,
        broadphase_proxies,
        body_colliders,
        body_filters,
        overflow_detected,
        tgs_cache,
        participants,
        fault,
        fields,
        ccd,
    })
}

fn w_fault(w: &mut W, fault: Option<WorldFault>) {
    match fault {
        None => w.u8(0),
        Some(WorldFault::Participant { index, kind, fault }) => {
            w.u8(1);
            w.usize(index);
            w.u32(kind.get());
            w.u8(fault.tag());
        }
        Some(WorldFault::RigidOverflow) => w.u8(2),
        Some(WorldFault::ForceOutOfRange { body }) => {
            w.u8(3);
            w.usize(body);
        }
        Some(WorldFault::FieldOutOfRange { field, index }) => {
            w.u8(4);
            w.u32(field.get());
            w.usize(index);
        }
    }
}

fn r_fault(r: &mut R<'_>) -> Res<Option<WorldFault>> {
    const SECTION: &str = "fault";
    Ok(match r.u8()? {
        0 => None,
        1 => {
            let index = r.usize(SECTION)?;
            let kind = crate::world_participant::ParticipantKind::new(r.u32()?);
            let fault = crate::world_participant::ParticipantFault::from_tag(r.u8()?)
                .ok_or(WorldSnapshotError::InvalidValue { section: SECTION })?;
            Some(WorldFault::Participant { index, kind, fault })
        }
        2 => Some(WorldFault::RigidOverflow),
        3 => Some(WorldFault::ForceOutOfRange {
            body: r.usize(SECTION)?,
        }),
        4 => Some(WorldFault::FieldOutOfRange {
            field: crate::world_participant::PortId::new(r.u32()?),
            index: r.usize(SECTION)?,
        }),
        _ => return Err(WorldSnapshotError::InvalidValue { section: SECTION }),
    })
}

/// The bytes of the `fields` section (`FieldBoard::write_values`), walked to
/// find where it ends; whether they fit the target world is checked on
/// restore (`FieldBoard::check_values`).
fn r_fields(r: &mut R<'_>) -> Res<Vec<u8>> {
    const SECTION: &str = "fields";
    let start = r.pos;
    let n = r.len(SECTION, 4 + 1 + 1 + 8)?;
    for _ in 0..n {
        r.u32()?;
        if r.u8()? > 1 {
            return Err(WorldSnapshotError::InvalidValue { section: SECTION });
        }
        let samples = match r.u8()? {
            0 => r.usize(SECTION)?,
            1 => {
                let origin = r.vec3()?;
                let cell = r.fix()?;
                let dims = [r.usize(SECTION)?, r.usize(SECTION)?, r.usize(SECTION)?];
                crate::world_participant::FieldLayout::Grid { origin, cell, dims }
                    .samples()
                    .ok_or(WorldSnapshotError::InvalidValue { section: SECTION })?
            }
            _ => return Err(WorldSnapshotError::InvalidValue { section: SECTION }),
        };
        let bytes = samples
            .checked_mul(16)
            .ok_or(WorldSnapshotError::Truncated)?;
        r.take(bytes)?;
    }
    Ok(r.data[start..r.pos].to_vec())
}

/// Reject indices that `step` would follow past the end of a `Vec`.
fn validate(d: &Decoded) -> Res<()> {
    let n = d.bodies.len();
    let check = |section: &'static str, index: usize, len: usize| -> Res<()> {
        if index < len {
            Ok(())
        } else {
            Err(WorldSnapshotError::DanglingIndex {
                section,
                index,
                len,
            })
        }
    };
    for j in &d.joints {
        let (a, b) = j.bodies();
        check("joints", a, n)?;
        check("joints", b, n)?;
    }
    for c in &d.distance_constraints {
        check("distance_constraints", c.body_a, n)?;
        check("distance_constraints", c.body_b, n)?;
    }
    for c in &d.contact_constraints {
        check("contact_constraints", c.body_a, n)?;
        check("contact_constraints", c.body_b, n)?;
    }
    for p in &d.sdf_poses {
        if p.3 != crate::sdf_collider::SDF_STATIC {
            check("sdf_colliders", p.3, n)?;
        }
    }
    for b in &d.constraint_batches {
        for &i in &b.distance_indices {
            check("constraint_batches", i, d.distance_constraints.len())?;
        }
        for &i in &b.contact_indices {
            check("constraint_batches", i, d.contact_constraints.len())?;
        }
    }
    for &p in d.broadphase_proxies.iter().flatten() {
        check(
            "broadphase_proxies",
            p as usize,
            d.broadphase_tree.nodes.len(),
        )?;
    }
    Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    // Oracles for the whole-world snapshot.
    //
    // The comparator ([`digest`]) reads every [`PhysicsWorld`] field directly
    // (through `Debug` or field by field) and never goes through the snapshot
    // encoder, so a field the encoder drops on both sides still shows up as a
    // difference between the original and the restored world.

    use super::super::*;
    use super::WorldSnapshotError;
    use crate::character::CharacterController;
    use crate::collider::Contact;
    use crate::compound::CompoundShape;
    use crate::force::{ForceField, ForceFieldInstance};
    use crate::heightfield::HeightField;
    use crate::joint::{
        BallJoint, ConeTwistJoint, D6Joint, FixedJoint, HingeJoint, Joint, SliderJoint, SpringJoint,
    };
    use crate::motor::{apply_motors, JointMotor, PdController};
    use crate::plane_collider::PlaneCollider;
    use crate::sdf_collider::{ClosureSdf, SdfCollider};
    use crate::shape::Shape;
    use crate::static_collider::StaticCollider;
    use crate::trimesh::TriMesh;

    fn fx(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    /// Every field of `w`, rendered without the snapshot encoder.
    fn digest(w: &mut PhysicsWorld) -> Vec<(&'static str, String)> {
        let mut d = Vec::new();
        d.push(("config", format!("{:?}", w.config)));
        d.push(("bodies", format!("{:?}", w.bodies)));
        d.push((
            "distance_constraints",
            format!("{:?}", w.distance_constraints),
        ));
        d.push((
            "contact_constraints",
            format!("{:?}", w.contact_constraints),
        ));
        let sdf: Vec<String> = w
            .sdf_colliders
            .iter()
            .map(|s| {
                format!(
                    "{:?} {:?} {:?} {} {:?} {} {} field@probe={}",
                    s.position,
                    s.rotation,
                    s.scale,
                    s.body_index,
                    s.inv_rotation,
                    s.scale_f32.to_bits(),
                    s.inv_scale_f32.to_bits(),
                    s.field.distance(0.3, 0.2, 0.1).to_bits()
                )
            })
            .collect();
        d.push(("sdf_colliders", format!("{sdf:?}")));
        d.push((
            "sdf_collision_radius",
            format!("{:?}", w.sdf_collision_radius),
        ));
        let statics: Vec<String> = w
            .static_colliders
            .iter()
            .map(|c| match c {
                StaticCollider::Plane(p) => format!("plane {p:?}"),
                StaticCollider::HeightField(h) => format!("hf {h:?}"),
                StaticCollider::TriMesh(m) => format!(
                    "mesh {:?} {:?} {:?} {:?} {:?}",
                    m.triangles, m.bvh.nodes, m.bvh.primitives, m.bvh.bounds, m.bounds
                ),
            })
            .collect();
        d.push(("static_colliders", format!("{statics:?}")));
        let batches: Vec<_> = w
            .constraint_batches
            .iter()
            .map(|b| (b.distance_indices.clone(), b.contact_indices.clone()))
            .collect();
        d.push(("constraint_batches", format!("{batches:?}")));
        d.push(("batches_dirty", format!("{}", w.batches_dirty)));
        d.push((
            "batch_static_bodies",
            format!("{:?}", w.batch_static_bodies),
        ));
        let index_probe: Vec<_> = w
            .contact_cache
            .manifolds
            .iter()
            .map(|m| w.contact_cache.find(&m.pair).map(|f| f.stale_frames))
            .collect();
        d.push((
            "contact_cache",
            format!(
                "{:?} {} {:?}",
                w.contact_cache.manifolds,
                w.contact_cache.max_stale_frames,
                w.contact_cache.warm_start_factor
            ),
        ));
        d.push(("contact_cache.pair_index", format!("{index_probe:?}")));
        d.push((
            "material_table",
            format!(
                "{:?} {:?} {:?} {:?}",
                w.material_table.materials,
                w.material_table.pair_overrides,
                w.material_table.default_friction_combine,
                w.material_table.default_restitution_combine
            ),
        ));
        d.push(("body_materials", format!("{:?}", w.body_materials)));
        d.push((
            "callbacks",
            format!("{} {}", w.pre_solve_hooks.len(), w.contact_modifiers.len()),
        ));
        d.push(("joints", format!("{:?}", w.joints)));
        d.push(("force_fields", format!("{:?}", w.force_fields)));
        let e = &w.events;
        d.push((
            "events",
            format!(
                "{:?} {:?} {:?} {:?} {:?} {:?}",
                e.contact_events,
                e.trigger_events,
                e.prev_pairs,
                e.curr_pairs,
                e.prev_triggers,
                e.curr_triggers
            ),
        ));
        d.push(("islands.config", format!("{:?}", w.islands.config)));
        d.push(("islands.sleep_data", format!("{:?}", w.islands.sleep_data)));
        let n = w.islands.sleep_data.len();
        let mut partition = Vec::new();
        for i in 0..n {
            for j in 0..n {
                partition.push(w.islands.find(i) == w.islands.find(j));
            }
        }
        d.push(("islands.partition", format!("{partition:?}")));
        d.push((
            "body_collision_radii",
            format!("{:?}", w.body_collision_radii),
        ));
        d.push(("broadphase", format!("{:?}", w.broadphase)));
        let t = &w.broadphase_tree;
        d.push((
            "broadphase_tree",
            format!(
                "{:?} {:?} {} {:?} {:?}",
                t.nodes, t.free_list, t.root, t.margin, t.metric
            ),
        ));
        d.push(("broadphase_proxies", format!("{:?}", w.broadphase_proxies)));
        d.push(("body_colliders", format!("{:?}", w.body_colliders)));
        d.push(("body_filters", format!("{:?}", w.body_filters)));
        d.push(("overflow_detected", format!("{}", w.overflow_detected)));
        d.push((
            "kinematic_substeps_left",
            format!("{}", w.kinematic_substeps_left),
        ));
        let c = &w.tgs_impulse_cache;
        let mut entries: Vec<_> = c.entries.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_unstable_by_key(|e| e.0);
        let mut live: Vec<_> = c.live.keys().copied().collect();
        live.sort_unstable();
        d.push((
            "tgs_impulse_cache",
            format!("{entries:?} {live:?} {} {}", c.hits, c.misses),
        ));
        d
    }

    /// Fields whose digest differs between `a` and `b`.
    fn differing(a: &mut PhysicsWorld, b: &mut PhysicsWorld) -> Vec<&'static str> {
        digest(a)
            .into_iter()
            .zip(digest(b))
            .filter(|(x, y)| x.1 != y.1)
            .map(|(x, _)| x.0)
            .collect()
    }

    fn ball_sdf() -> Box<dyn crate::sdf_collider::SdfField> {
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt().max(1e-6);
                (x / l, y / l, z / l)
            },
        ))
    }

    /// The SDF fields a restore target must already hold (code, not data).
    fn target_world() -> PhysicsWorld {
        let mut t = PhysicsWorld::new(SolverConfig::default());
        // attached to body 0 at the default pose: every pose field differs from
        // the scene's static, rotated, scaled collider
        t.add_sdf_collider(SdfCollider::new_dynamic(ball_sdf(), 0));
        t
    }

    /// Caller-owned state stepped next to the world (not part of the snapshot).
    #[derive(Clone)]
    struct Driver {
        motors: Vec<JointMotor>,
        character: (Vec3Fix, Vec3Fix, bool, Option<usize>, Vec3Fix),
    }

    impl Driver {
        fn new() -> Self {
            let mut pd = PdController::new(fx(20, 1), fx(2, 1), fx(50, 1));
            pd.set_position_target(fx(3, 2));
            Self {
                motors: vec![JointMotor::new(1, pd)],
                character: (v(1, 1, 3), Vec3Fix::ZERO, false, None, Vec3Fix::ZERO),
            }
        }

        /// One frame: motors, a character that pushes bodies, then `step`.
        fn frame(&mut self, w: &mut PhysicsWorld, dt: Fix128) {
            apply_motors(&self.motors, &w.joints, &mut w.bodies, dt);
            let (p, vel, g, gb, pv) = self.character;
            let mut ch = CharacterController::new_default(p);
            ch.velocity = vel;
            ch.grounded = g;
            ch.ground_body_index = gb;
            ch.platform_velocity = pv;
            ch.move_and_slide(
                Vec3Fix::new(fx(1, 20), Fix128::ZERO, Fix128::ZERO),
                &w.bodies,
                &w.sdf_colliders,
            );
            for push in ch.compute_push_impulses(&w.bodies, fx(1, 2)) {
                let b = &mut w.bodies[push.body_index];
                if !b.is_static() {
                    b.velocity = b.velocity + push.impulse * b.inv_mass;
                }
            }
            self.character = (
                ch.position,
                ch.velocity,
                ch.grounded,
                ch.ground_body_index,
                ch.platform_velocity,
            );
            w.step(dt);
        }
    }

    /// A scene that exercises every saved field: joints of all seven kinds,
    /// distance constraint, force fields with an affected-body list, static
    /// plane / height field / triangle mesh, SDF collider, shaped and compound
    /// bodies, materials with a pair override, filters, a sensor (trigger
    /// events), a kinematic body with a target, and a resting body that falls
    /// asleep.
    fn scene(backend: SolverBackend, broadphase: Broadphase) -> PhysicsWorld {
        let config = SolverConfig {
            solver_backend: backend,
            ..SolverConfig::default()
        };
        let mut w = PhysicsWorld::new(config);
        w.set_broadphase(broadphase);
        w.set_sdf_collision_radius(fx(3, 5));
        // a non-default fat margin, so the restored tree must carry it
        w.broadphase_tree.margin = fx(3, 8);
        w.set_sleep_config(SleepConfig {
            linear_threshold: fx(1, 2),
            angular_threshold: fx(1, 2),
            frames_to_sleep: 3,
        });

        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            v(0, 1, 0),
            Fix128::ZERO,
        )));
        let heights: Vec<Fix128> = (0..16).map(|i| fx(i % 4, 8)).collect();
        w.add_static_collider(StaticCollider::HeightField(HeightField::new(
            heights,
            4,
            4,
            Fix128::ONE,
            v(6, 0, -2),
        )));
        w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
            &[v(-10, 0, -10), v(-6, 1, -10), v(-10, 0, -6), v(-6, 1, -6)],
            &[0, 1, 2, 1, 3, 2],
        )));
        w.add_sdf_collider(
            SdfCollider::new_static(
                ball_sdf(),
                v(-4, 0, 0),
                QuatFix::from_axis_angle(v(0, 1, 0), fx(1, 3)),
            )
            .with_scale(fx(3, 2)),
        );

        // 0: rests on the plane and falls asleep
        w.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(Fix128::ZERO, fx(1, 2), Fix128::ZERO),
                Fix128::ONE,
            ),
            fx(1, 2),
        );
        // 1: anchor, 2..=8: one joint of each kind to the anchor
        let anchor = w.add_body(RigidBody::new_static(v(3, 6, 0)));
        let mut ids = Vec::new();
        for k in 0..7 {
            let id = w.add_body_with_radius(
                RigidBody::new_dynamic(v(3 + k, 5, 2 * k), Fix128::ONE),
                fx(1, 4),
            );
            ids.push(id);
        }
        let up = v(0, 1, 0);
        let z = Vec3Fix::ZERO;
        let mut d6 = D6Joint::new(anchor, ids[5], z, z);
        d6.linear_x = crate::joint::D6Motion::Limited;
        w.add_joint(Joint::Ball(BallJoint::new(anchor, ids[0], z, v(-1, 1, 0))));
        let mut hinge = HingeJoint::new(anchor, ids[1], z, v(-1, 0, 0), up, up);
        hinge.angle_min = Some(fx(-1, 2));
        hinge.angle_max = Some(fx(1, 2));
        w.add_joint(Joint::Hinge(hinge));
        w.add_joint(Joint::Fixed(FixedJoint::new(
            anchor,
            ids[2],
            z,
            v(0, 1, 0),
            QuatFix::IDENTITY,
        )));
        let mut slider = SliderJoint::new(anchor, ids[3], v(1, 0, 0), z, z);
        slider.limit_min = Some(fx(-1, 1));
        slider.break_force = Some(fx(1000, 1));
        w.add_joint(Joint::Slider(slider));
        w.add_joint(Joint::Spring(SpringJoint::new(
            anchor,
            ids[4],
            z,
            z,
            fx(3, 2),
            fx(40, 1),
            fx(1, 2),
        )));
        w.add_joint(Joint::D6(d6));
        w.add_joint(Joint::ConeTwist(ConeTwistJoint::new(
            anchor, ids[6], z, z, up, up,
        )));
        w.add_distance_constraint(DistanceConstraint {
            body_a: ids[0],
            body_b: ids[1],
            local_anchor_a: z,
            local_anchor_b: z,
            target_distance: fx(3, 2),
            compliance: fx(1, 1000),
            cached_lambda: Fix128::ZERO,
        });

        // shaped + compound bodies above the height field, a body above the SDF
        let boxed = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: Vec3Fix::new(fx(1, 2), fx(1, 4), fx(1, 2)),
                },
                Fix128::ONE,
                v(7, 2, 0),
            )
            .unwrap();
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            crate::collider::Sphere::new(Vec3Fix::ZERO, fx(1, 2)),
            v(-1, 0, 0) * fx(1, 2),
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            crate::collider::Sphere::new(Vec3Fix::ZERO, fx(1, 2)),
            v(1, 0, 0) * fx(1, 2),
            QuatFix::IDENTITY,
        );
        let comp = w
            .add_compound_body(&compound, Fix128::ONE, v(8, 3, 1))
            .unwrap();
        // compute the cached AABB, so `dirty` is false and the cache is non-zero
        if let Some(crate::body_collider::BodyCollider::Compound(c)) = &mut w.body_colliders[comp] {
            let _ = c.compute_aabb();
            assert!(!c.dirty);
        }
        let over_sdf =
            w.add_body_with_radius(RigidBody::new_dynamic(v(-4, 3, 0), Fix128::ONE), fx(1, 2));
        // sensor around the swinging bodies, kinematic body with a target
        let sensor = w.add_body_with_radius(RigidBody::new_sensor(v(3, 5, 0)), fx(2, 1));
        let mut kin = RigidBody::new_kinematic(v(0, 2, -3));
        kin.kinematic_target = Some((v(2, 2, -3), QuatFix::IDENTITY));
        let kin = w.add_body_with_radius(kin, fx(1, 2));

        let rubber = w.material_table.register_rubber();
        let ice = w.material_table.register_ice();
        w.material_table
            .set_pair_override(rubber, ice, fx(1, 7), fx(2, 7));
        w.set_body_material(boxed, rubber);
        w.set_body_material(comp, ice);
        w.set_body_filter(over_sdf, CollisionFilter::new(2, 0xFFFF_FFFF));

        w.add_force_field(
            ForceFieldInstance::new(ForceField::Directional {
                direction: v(1, 0, 0),
                strength: fx(3, 1),
            })
            .with_affected_bodies(vec![boxed, comp]),
        );
        w.add_force_field(ForceFieldInstance::new(ForceField::Drag {
            coefficient: fx(1, 10),
        }));
        w.add_force_field(ForceFieldInstance::new(ForceField::Vortex {
            center: v(3, 5, 0),
            axis: up,
            strength: fx(1, 2),
            falloff_radius: fx(20, 1),
        }));
        let mut blast = ForceFieldInstance::new(ForceField::Explosion {
            center: v(0, 0, 0),
            strength: fx(500, 1),
            radius: fx(50, 1),
            falloff_power: fx(2, 1),
        });
        blast.enabled = false;
        w.add_force_field(blast);
        let _ = (sensor, kin);
        w
    }

    fn dt() -> Fix128 {
        fx(1, 60)
    }

    /// Run `k` frames, add a manual contact (so `contact_constraints` and
    /// `contact_cache` are non-empty in the snapshot), snapshot, restore into a
    /// target holding the same SDF field, then run `m` more frames on both.
    /// Returns the field names that differed at any checkpoint.
    fn round_trip(
        backend: SolverBackend,
        broadphase: Broadphase,
        k: usize,
        m: usize,
        overflow: bool,
    ) -> Vec<&'static str> {
        let mut a = scene(backend, broadphase);
        let mut drv = Driver::new();
        for _ in 0..k {
            drv.frame(&mut a, dt());
        }
        a.add_contact(ContactConstraint::new(
            2,
            3,
            Contact {
                depth: fx(1, 100),
                normal: v(0, 1, 0),
                point_a: v(3, 5, 0),
                point_b: v(4, 5, 2),
            },
        ));
        a.overflow_detected = overflow;
        // populate the colour batches (dirty = false) so they differ from a
        // fresh world's (empty, dirty = true)
        a.rebuild_batches();
        assert!(!a.constraint_batches.is_empty() && !a.batches_dirty);
        assert!(!a.contact_constraints.is_empty() && !a.contact_cache.manifolds.is_empty());
        let blob = a.snapshot_world();
        let mut b = target_world();
        b.restore_world(&blob).expect("restore");
        let mut drv_b = drv.clone();

        let mut diff = differing(&mut a, &mut b);
        for _ in 0..m {
            drv.frame(&mut a, dt());
            drv_b.frame(&mut b, dt());
            for f in differing(&mut a, &mut b) {
                if !diff.contains(&f) {
                    diff.push(f);
                }
            }
        }
        if a.snapshot_world() != b.snapshot_world() && !diff.contains(&"blob") {
            diff.push("blob");
        }
        diff
    }

    #[test]
    fn round_trip_is_bit_identical_for_every_field_and_every_later_step() {
        for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
            for bp in [Broadphase::Bvh, Broadphase::DynamicTree] {
                for (k, m) in [(0, 0), (0, 5), (1, 1), (7, 3), (20, 15)] {
                    let diff = round_trip(backend, bp, k, m, k % 2 == 1);
                    assert!(diff.is_empty(), "{backend:?} {bp:?} k={k} m={m}: {diff:?}");
                }
            }
        }
    }

    /// Controls: the round trip above is only meaningful if the scene really
    /// holds a sleeping body, events, contact-cache entries, tree nodes, TGS
    /// warm-start entries and a non-trivial island partition at snapshot time.
    #[test]
    fn round_trip_scene_exercises_the_covered_state() {
        let mut a = scene(SolverBackend::Xpbd, Broadphase::DynamicTree);
        let mut drv = Driver::new();
        for _ in 0..20 {
            drv.frame(&mut a, dt());
        }
        assert!(a
            .islands
            .sleep_data
            .iter()
            .any(|s| s.state == SleepState::Sleeping && a.bodies.len() > 1));
        assert!(a.is_sleeping(0), "body 0 rests on the plane and sleeps");
        assert!(!a.events.curr_pairs.is_empty() || !a.events.prev_pairs.is_empty());
        assert!(
            !a.events.curr_triggers.is_empty(),
            "sensor overlaps the swinging bodies"
        );
        assert!(!a.broadphase_tree.nodes.is_empty());
        assert!(a.broadphase_proxies.iter().any(Option::is_some));
        assert_ne!(a.islands.find(1), a.islands.find(0));
        assert_eq!(a.islands.find(1), a.islands.find(2));
        assert!(a
            .body_colliders
            .iter()
            .any(|c| matches!(c, Some(crate::body_collider::BodyCollider::Compound(_)))));

        let mut t = scene(SolverBackend::Tgs, Broadphase::Bvh);
        let mut drv = Driver::new();
        for _ in 0..7 {
            drv.frame(&mut t, dt());
        }
        assert!(!t.tgs_impulse_cache.entries.is_empty());
        assert!(t.tgs_impulse_cache.hits + t.tgs_impulse_cache.misses > 0);
        // the live set is empty between steps, which is why it is rebuilt empty
        assert!(t.tgs_impulse_cache.live.is_empty());
    }

    #[test]
    fn step_n_equals_step_called_n_times() {
        for n in [0usize, 1, 2, 9] {
            let mut a = scene(SolverBackend::Xpbd, Broadphase::Bvh);
            let mut b = scene(SolverBackend::Xpbd, Broadphase::Bvh);
            a.step_n(n, dt());
            for _ in 0..n {
                b.step(dt());
            }
            assert!(differing(&mut a, &mut b).is_empty(), "n={n}");
            assert_eq!(a.snapshot_world(), b.snapshot_world());
        }
        // n = 0 leaves the world unchanged
        let mut a = scene(SolverBackend::Xpbd, Broadphase::Bvh);
        let before = a.snapshot_world();
        a.step_n(0, dt());
        assert_eq!(a.snapshot_world(), before);
        // and so does a non-positive dt for any n
        a.step_n(5, Fix128::ZERO);
        assert_eq!(a.snapshot_world(), before);
    }

    /// The target keeps its SDF fields and callbacks; the counts must match.
    #[test]
    fn restore_requires_the_same_number_of_non_data_parts() {
        let mut a = scene(SolverBackend::Xpbd, Broadphase::Bvh);
        // differs from the target's config, so a partial install would show
        a.config.substeps = 3;
        let blob = a.snapshot_world();

        // a fresh world has no SDF field
        assert!(matches!(
            PhysicsWorld::from_world_snapshot(&blob),
            Err(WorldSnapshotError::SdfFieldCountMismatch {
                snapshot: 1,
                world: 0
            })
        ));

        // a pre-solve hook on the target that the source did not have
        let mut t = target_world();
        t.add_pre_solve_hook(Box::new(|_, _, _| true));
        let before = digest(&mut t);
        assert_eq!(
            t.restore_world(&blob),
            Err(WorldSnapshotError::CallbackCountMismatch {
                section: "pre_solve_hooks",
                snapshot: 0,
                world: 1,
            })
        );
        assert_eq!(
            digest(&mut t),
            before,
            "rejected restore leaves the target untouched"
        );
    }

    // ---- coverage: codec variants and reader boundaries (2026-10-07) ----

    /// A world holding every `Shape` variant, a compound with every child
    /// kind, every force field kind, a `Multiply` material and the hybrid
    /// broadphase (none of which the round-trip scene uses).
    fn variant_world() -> PhysicsWorld {
        use crate::body_collider::BodyCollider;
        use crate::collider::{Capsule, ConvexHull, Sphere};
        use crate::material::{CombineRule, PhysicsMaterial};
        let mut w = PhysicsWorld::new(SolverConfig::default());
        w.set_broadphase(Broadphase::Hybrid);
        let shapes = [
            Shape::Box {
                half_extents: Vec3Fix::new(fx(1, 2), fx(1, 4), fx(1, 2)),
            },
            Shape::Cylinder {
                radius: fx(1, 2),
                half_height: fx(3, 4),
            },
            Shape::Cone {
                radius: fx(1, 2),
                half_height: fx(1, 1),
            },
            Shape::Ellipsoid {
                radii: Vec3Fix::new(fx(1, 2), fx(3, 4), fx(1, 4)),
            },
            Shape::Wedge {
                width: fx(1, 1),
                height: fx(1, 2),
                depth: fx(3, 2),
            },
            Shape::Torus {
                major_radius: fx(1, 1),
                minor_radius: fx(1, 4),
            },
        ];
        for (k, shape) in shapes.iter().enumerate() {
            let i = w.add_body_with_radius(
                RigidBody::new_dynamic(v(3 * k as i64, 4, 0), Fix128::ONE),
                fx(3, 2),
            );
            w.body_colliders[i] = Some(BodyCollider::Shape(*shape));
        }
        let mut c = CompoundShape::new();
        c.add_sphere(
            Sphere::new(Vec3Fix::ZERO, fx(1, 4)),
            v(1, 0, 0),
            QuatFix::IDENTITY,
        );
        c.add_capsule(
            Capsule::new(v(0, -1, 0), v(0, 1, 0), fx(1, 4)),
            v(-1, 0, 0),
            QuatFix::IDENTITY,
        );
        c.add_box(
            crate::box_collider::OrientedBox {
                center: Vec3Fix::ZERO,
                half_extents: Vec3Fix::new(fx(1, 4), fx(1, 4), fx(1, 4)),
                rotation: QuatFix::IDENTITY,
            },
            v(0, 0, 1),
            QuatFix::from_axis_angle(v(0, 1, 0), fx(1, 3)),
        );
        c.add_convex_hull(
            ConvexHull::new(vec![v(0, 0, 0), v(1, 0, 0), v(0, 1, 0), v(0, 0, 1)]),
            v(0, 0, -1),
            QuatFix::IDENTITY,
        );
        let comp =
            w.add_body_with_radius(RigidBody::new_dynamic(v(0, 8, 4), Fix128::ONE), fx(2, 1));
        w.body_colliders[comp] = Some(BodyCollider::Compound(c));

        let mut mat = PhysicsMaterial::new(0, fx(1, 3), fx(1, 5));
        mat.friction_combine = CombineRule::Multiply;
        mat.restitution_combine = CombineRule::Multiply;
        let id = w.material_table.register(mat);
        w.material_table.default_friction_combine = CombineRule::Multiply;
        w.set_body_material(comp, id);

        w.add_force_field(
            ForceFieldInstance::new(ForceField::Point {
                center: v(0, 10, 0),
                strength: fx(5, 1),
                repulsive: true,
                max_force: fx(50, 1),
            })
            .with_affected_bodies(vec![0, comp]),
        );
        w.add_force_field(ForceFieldInstance::new(ForceField::Buoyancy {
            surface_y: fx(2, 1),
            density: fx(1000, 1),
            drag: fx(1, 2),
        }));
        w.add_force_field(ForceFieldInstance::new(ForceField::Magnetic {
            position: v(0, 0, 0),
            moment: v(0, 1, 0),
            strength: fx(1, 10),
        }));
        w
    }

    /// Every codec variant survives a round trip: the restored world equals
    /// the original field by field, now and after the same later steps.
    #[test]
    fn round_trip_covers_every_shape_child_and_force_field_kind() {
        let mut a = variant_world();
        let blob = a.snapshot_world();
        let mut b = PhysicsWorld::from_world_snapshot(&blob).expect("restore");
        assert!(differing(&mut a, &mut b).is_empty());
        for _ in 0..5 {
            a.step(dt());
            b.step(dt());
        }
        assert_eq!(differing(&mut a, &mut b), Vec::<&str>::new());
        assert_eq!(a.snapshot_world(), b.snapshot_world());
    }

    /// A dynamic SDF collider (attached to a body, not static) round-trips
    /// into a target holding the same field.
    #[test]
    fn round_trip_keeps_a_body_attached_sdf_collider() {
        let mut a = target_world();
        a.add_body_with_radius(RigidBody::new_dynamic(v(0, 3, 0), Fix128::ONE), fx(1, 2));
        let blob = a.snapshot_world();
        let mut b = target_world();
        b.restore_world(&blob).expect("restore");
        assert!(differing(&mut a, &mut b).is_empty());
    }

    /// Every error renders a message naming its values.
    #[test]
    fn error_display_names_the_values() {
        use crate::world_participant::{ParticipantMismatch, StateError};
        let cases: [(WorldSnapshotError, &[&str]); 13] = [
            (WorldSnapshotError::Truncated, &["truncated"]),
            (WorldSnapshotError::BadMagic, &["magic"]),
            (
                WorldSnapshotError::UnsupportedVersion {
                    found: 77,
                    supported: 88,
                },
                &["77", "88"],
            ),
            (WorldSnapshotError::ReservedNotZero, &["reserved"]),
            (WorldSnapshotError::TrailingBytes { extra: 5 }, &["5"]),
            (
                WorldSnapshotError::ChecksumMismatch {
                    stored: 0x10,
                    computed: 0x20,
                },
                &["0x0000000000000010", "0x0000000000000020"],
            ),
            (
                WorldSnapshotError::InvalidValue { section: "joints" },
                &["joints"],
            ),
            (
                WorldSnapshotError::DanglingIndex {
                    section: "contacts",
                    index: 9,
                    len: 4,
                },
                &["contacts", "9", "4"],
            ),
            (
                WorldSnapshotError::SdfFieldCountMismatch {
                    snapshot: 2,
                    world: 3,
                },
                &["2", "3"],
            ),
            (
                WorldSnapshotError::CallbackCountMismatch {
                    section: "contact_modifiers",
                    snapshot: 6,
                    world: 7,
                },
                &["contact_modifiers", "6", "7"],
            ),
            (
                WorldSnapshotError::ParticipantMismatch(ParticipantMismatch::Count {
                    snapshot: 11,
                    world: 12,
                }),
                &["11", "12"],
            ),
            (
                WorldSnapshotError::ParticipantState {
                    index: 13,
                    error: StateError::InvalidValue,
                },
                &["13", "InvalidValue"],
            ),
            (
                WorldSnapshotError::FieldState(StateError::Length {
                    expected: 14,
                    found: 15,
                }),
                &["14", "15"],
            ),
        ];
        for (e, parts) in cases {
            let text = e.to_string();
            for p in parts {
                assert!(text.contains(p), "{e:?} -> {text:?} lacks {p:?}");
            }
        }
    }

    /// Header checks that do not depend on the format version: a blob
    /// shorter than a header, a wrong magic, non-zero reserved bytes, a
    /// declared payload longer than the blob, extra bytes after the
    /// checksum and a corrupted payload byte are all rejected, and the
    /// target world is left untouched each time.
    #[test]
    fn restore_rejects_malformed_headers_and_corruption() {
        let src = variant_world();
        let blob = src.snapshot_world();
        let mut t = PhysicsWorld::new(SolverConfig::default());
        let before = digest(&mut t);

        assert_eq!(
            t.restore_world(&blob[..10]),
            Err(WorldSnapshotError::Truncated)
        );

        let mut bad = blob.clone();
        bad[0] ^= 0xFF;
        assert_eq!(t.restore_world(&bad), Err(WorldSnapshotError::BadMagic));

        let mut bad = blob.clone();
        bad[6] = 1;
        assert_eq!(
            t.restore_world(&bad),
            Err(WorldSnapshotError::ReservedNotZero)
        );

        let mut bad = blob.clone();
        bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(t.restore_world(&bad), Err(WorldSnapshotError::Truncated));

        let mut bad = blob.clone();
        bad.extend_from_slice(&[0, 0, 0]);
        assert_eq!(
            t.restore_world(&bad),
            Err(WorldSnapshotError::TrailingBytes { extra: 3 })
        );

        let mut bad = blob.clone();
        bad[20] ^= 0x01;
        assert!(matches!(
            t.restore_world(&bad),
            Err(WorldSnapshotError::ChecksumMismatch { stored, computed }) if stored != computed
        ));

        assert_eq!(digest(&mut t), before);
        // control: the untouched blob restores
        assert!(t.restore_world(&blob).is_ok());
    }

    fn reader(data: &[u8]) -> super::R<'_> {
        super::R { data, pos: 0 }
    }

    /// The leaf readers reject an unknown tag in their own section.
    #[test]
    fn leaf_readers_reject_unknown_tags() {
        let inv = |section| Some(WorldSnapshotError::InvalidValue { section });
        assert_eq!(
            super::r_shape(&mut reader(&[6])).err(),
            inv("body_colliders")
        );
        assert_eq!(
            super::r_body_collider(&mut reader(&[3])).err(),
            inv("body_colliders")
        );
        // a compound with one child of unknown kind 4
        let mut w = super::W(Vec::new());
        w.u8(2);
        w.usize(1);
        w.u8(4);
        assert_eq!(
            super::r_body_collider(&mut reader(&w.0)).err(),
            inv("body_colliders")
        );
        assert_eq!(
            super::r_force_field(&mut reader(&[7])).err(),
            inv("force_fields")
        );
        assert_eq!(super::r_d6_motion(&mut reader(&[3])).err(), inv("joints"));
        assert_eq!(
            super::r_combine(&mut reader(&[4])).err(),
            inv("material_table")
        );
        assert_eq!(super::r_fault(&mut reader(&[5])).err(), inv("fault"));
        // config: a valid encoding with its backend tag replaced
        let mut w = super::W(Vec::new());
        super::w_config(&mut w, &SolverConfig::default());
        *w.0.last_mut().expect("backend tag") = 2;
        assert_eq!(super::r_config(&mut reader(&w.0)).err(), inv("config"));
        // empty input is truncation, not a value error
        assert_eq!(
            super::r_shape(&mut reader(&[])).err(),
            Some(WorldSnapshotError::Truncated)
        );
    }

    /// The fault codec round-trips every variant.
    #[test]
    fn fault_codec_round_trips_every_variant() {
        use crate::world_participant::{ParticipantFault, ParticipantKind, PortId, WorldFault};
        let faults = [
            None,
            Some(WorldFault::Participant {
                index: 3,
                kind: ParticipantKind::new(0xABCD),
                fault: ParticipantFault::InvalidState,
            }),
            Some(WorldFault::RigidOverflow),
            Some(WorldFault::ForceOutOfRange { body: 17 }),
            Some(WorldFault::FieldOutOfRange {
                field: PortId::new(9),
                index: 21,
            }),
        ];
        for f in faults {
            let mut w = super::W(Vec::new());
            super::w_fault(&mut w, f);
            let mut r = reader(&w.0);
            assert_eq!(super::r_fault(&mut r), Ok(f));
            assert_eq!(r.pos, w.0.len(), "{f:?} reads exactly what it wrote");
        }
        // a participant fault with an unknown fault tag
        let mut w = super::W(Vec::new());
        w.u8(1);
        w.usize(0);
        w.u32(1);
        w.u8(0xEE);
        assert_eq!(
            super::r_fault(&mut reader(&w.0)),
            Err(WorldSnapshotError::InvalidValue { section: "fault" })
        );
    }
}
