//! Content identifier of the law a [`PhysicsWorld`]
//! is configured with: [`PhysicsWorld::law_id`].
//!
//! Two worlds that start from the same state and receive the same inputs
//! produce the same bits when they follow the same law. The law has two
//! parts, each named by a 32-byte identifier:
//!
//! * the **arithmetic and the step implementation** of a build, named by
//!   [`PHYSICS_SEMANTICS_ID`](crate::semantics::PHYSICS_SEMANTICS_ID);
//! * the **rules a given world is configured with**, named by
//!   `PhysicsWorld::law_id`, which hashes them together with a semantics
//!   identifier.

/// Domain separation for world law identifiers, published so that an
/// independent implementation can reproduce one byte for byte.
pub const LAW_ID_DOMAIN: &[u8] = b"alice-physics/law-id/v1";

/// Identifies the encoding of a world law identifier. A change to the
/// encoding needs a new tag, since it invalidates every identifier published
/// under the old one.
pub const WORLD_LAW_KIND: &[u8] = b"physics-world/v1";

use crate::force::ForceField;
use crate::material::CombineRule;
use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsWorld, SolverBackend};
use sha2::{Digest, Sha256};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The canonical byte encoding of a world's law, written into a hasher.
struct Encoder(Sha256);

impl Encoder {
    /// A length prefix: the count as a big-endian `u64`.
    fn len(&mut self, n: usize) {
        self.0.update((n as u64).to_be_bytes());
    }
    /// A length-prefixed byte string (also used for the names of enum values).
    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.0.update(b);
    }
    fn u32(&mut self, v: u32) {
        self.0.update(v.to_be_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.update(v.to_be_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.update([u8::from(v)]);
    }
    /// The raw bits of a `Fix128`, high word then low word, big-endian.
    fn fix(&mut self, v: Fix128) {
        self.0.update(v.hi.to_be_bytes());
        self.0.update(v.lo.to_be_bytes());
    }
    fn vec3(&mut self, v: Vec3Fix) {
        self.fix(v.x);
        self.fix(v.y);
        self.fix(v.z);
    }
    fn combine(&mut self, rule: CombineRule) {
        self.bytes(match rule {
            CombineRule::Average => b"average",
            CombineRule::Min => b"min",
            CombineRule::Max => b"max",
            CombineRule::Multiply => b"multiply",
        });
    }
    fn field(&mut self, field: &ForceField) {
        match *field {
            ForceField::Directional {
                direction,
                strength,
            } => {
                self.bytes(b"directional");
                self.vec3(direction);
                self.fix(strength);
            }
            ForceField::Point {
                center,
                strength,
                repulsive,
                max_force,
            } => {
                self.bytes(b"point");
                self.vec3(center);
                self.fix(strength);
                self.bool(repulsive);
                self.fix(max_force);
            }
            ForceField::Drag { coefficient } => {
                self.bytes(b"drag");
                self.fix(coefficient);
            }
            ForceField::Buoyancy {
                surface_y,
                density,
                drag,
            } => {
                self.bytes(b"buoyancy");
                self.fix(surface_y);
                self.fix(density);
                self.fix(drag);
            }
            ForceField::Vortex {
                center,
                axis,
                strength,
                falloff_radius,
            } => {
                self.bytes(b"vortex");
                self.vec3(center);
                self.vec3(axis);
                self.fix(strength);
                self.fix(falloff_radius);
            }
            ForceField::Explosion {
                center,
                strength,
                radius,
                falloff_power,
            } => {
                self.bytes(b"explosion");
                self.vec3(center);
                self.fix(strength);
                self.fix(radius);
                self.fix(falloff_power);
            }
            ForceField::Magnetic {
                position,
                moment,
                strength,
            } => {
                self.bytes(b"magnetic");
                self.vec3(position);
                self.vec3(moment);
                self.fix(strength);
            }
        }
    }
}

impl PhysicsWorld {
    /// Identifier of the law this world steps by, hashed with `semantics_id`
    /// (pass [`PHYSICS_SEMANTICS_ID`](crate::semantics::PHYSICS_SEMANTICS_ID)
    /// for this build).
    ///
    /// # What it covers
    ///
    /// The rules a step reads, not the state it acts on: the solver config,
    /// the contact cache settings, `sdf_collision_radius`, continuous
    /// collision, the sleep thresholds, the material table, the force fields
    /// and the registered participants. The encoding below lists every field.
    ///
    /// Not covered:
    ///
    /// * the state: bodies, constraints and joints, including joint motors
    ///   (their gains and targets live on the joint, so they are state);
    /// * the broadphase kind and the sleep skip, since every choice gives the
    ///   same bits;
    /// * the caller-supplied code a world can hold: pre-solve hooks, contact
    ///   modifiers, SDF colliders and an installed GPU solver bridge.
    ///
    /// `config.warm_start_factor` is covered although the CPU step does not
    /// read it: it is passed to an installed GPU solver bridge, whose result
    /// may depend on it.
    ///
    /// # Guarantee, and its limit
    ///
    /// Two worlds with the same identifier, the same state and the same
    /// inputs step to the same bits, **provided the hooks, modifiers, SDF
    /// colliders and bridge they hold also behave the same**, since those are
    /// outside the identifier. A participant contributes only its kind, step
    /// rule and ports: two participants of one kind that differ in their
    /// coefficients get the same identifier until participants expose their
    /// coefficients. The converse does not hold either: worlds whose laws step
    /// identically may still differ here (for example two force fields
    /// declared in the other order), so this is not a deduplication key.
    ///
    /// # Encoding
    ///
    /// SHA-256 over the concatenation below, which an independent
    /// implementation can reproduce byte for byte (the layout of the law
    /// identifiers of `alice-zip`). Primitive encodings:
    ///
    /// | Value | Bytes |
    /// |-------|-------|
    /// | length, count, `usize` | `u64` big-endian |
    /// | `u32` | 4 bytes big-endian |
    /// | `MaterialId` (`u16`) | widened to `u64`, big-endian |
    /// | `bool` | one byte, `0` or `1` |
    /// | `Fix128` | raw high word (`i64`) then low word (`u64`), big-endian |
    /// | `Vec3Fix` | `x`, `y`, `z` as `Fix128` |
    /// | byte string, enum value | length, then the bytes; an enum value is written as the tag given in the table |
    /// | list | count, then the items |
    /// | `Option<T>` | `bool` (`1` for `Some`), then `T` when present |
    ///
    /// Fields, in order:
    ///
    /// | # | Field | Encoding |
    /// |---|-------|----------|
    /// | 1 | [`LAW_ID_DOMAIN`] | byte string |
    /// | 2 | [`WORLD_LAW_KIND`] | byte string |
    /// | 3 | `semantics_id` | the 32 bytes, no length |
    /// | 4 | `config.substeps`, `config.iterations` | `u64`, `u64` |
    /// | 5 | `config.gravity`, `config.damping` | `Vec3Fix`, `Fix128` |
    /// | 6 | `config.solver_backend` | tag `xpbd` or `tgs` |
    /// | 7 | `config.warm_start_factor` | `Fix128` |
    /// | 8 | contact cache `warm_start_factor`, `max_stale_frames` | `Fix128`, `u32` |
    /// | 9 | `sdf_collision_radius` | `Fix128` |
    /// | 10 | continuous collision: enabled, motion threshold | `bool`, `Fix128` |
    /// | 11 | sleep `linear_threshold`, `angular_threshold`, `frames_to_sleep` | `Fix128`, `Fix128`, `u32` |
    /// | 12 | default friction and restitution combine rules | tag `average`, `min`, `max` or `multiply`, twice |
    /// | 13 | materials, in id order | list of: id (`MaterialId`), `static_friction`, `dynamic_friction`, `restitution` (`Fix128` each), friction combine, restitution combine (tags) |
    /// | 14 | pair overrides, sorted by `(mat_a, mat_b)` with `mat_a <= mat_b` | list of: `mat_a`, `mat_b` (`MaterialId`), `friction`, `restitution` (`Fix128`) |
    /// | 15 | force fields, in the order they were added | list of: `enabled` (`bool`), the bodies it is limited to (`Option` of a list of `u64`, sorted ascending without duplicates), then the field |
    /// | 16 | participants, in registration order | list of: kind (`u32`), step rule, ports |
    ///
    /// A field is its tag followed by its parameters in declaration order:
    /// `directional` (direction `Vec3Fix`, strength), `point` (center
    /// `Vec3Fix`, strength, repulsive `bool`, max_force), `drag`
    /// (coefficient), `buoyancy` (surface_y, density, drag), `vortex` (center
    /// `Vec3Fix`, axis `Vec3Fix`, strength, falloff_radius), `explosion`
    /// (center `Vec3Fix`, strength, radius, falloff_power), `magnetic`
    /// (position `Vec3Fix`, moment `Vec3Fix`, strength); every parameter not
    /// marked otherwise is a `Fix128`. A step rule is the tag
    /// `follow-substep`, `fixed` followed by its `dt` (`Fix128`), or
    /// `subcycle`. A port is its id (`u32`) followed by the tag `read`,
    /// `write` or `read-committed`.
    ///
    /// Sorting the pair overrides and the body lists keeps the identifier
    /// independent of the order they were registered in, which the step does
    /// not depend on either. Force fields and participants keep their order,
    /// since the step applies them in that order.
    #[must_use]
    pub fn law_id(&self, semantics_id: &[u8; 32]) -> [u8; 32] {
        let mut e = Encoder(Sha256::new());
        e.bytes(LAW_ID_DOMAIN);
        e.bytes(WORLD_LAW_KIND);
        e.0.update(semantics_id);

        let c = &self.config;
        e.u64(c.substeps as u64);
        e.u64(c.iterations as u64);
        e.vec3(c.gravity);
        e.fix(c.damping);
        e.bytes(match c.solver_backend {
            SolverBackend::Xpbd => b"xpbd",
            SolverBackend::Tgs => b"tgs",
        });
        e.fix(c.warm_start_factor);

        e.fix(self.contact_cache.warm_start_factor);
        e.u32(self.contact_cache.max_stale_frames);
        e.fix(self.sdf_collision_radius);

        let ccd = self.continuous_collision();
        e.bool(ccd.is_enabled());
        e.fix(ccd.motion_threshold());

        let sleep = self.islands.config;
        e.fix(sleep.linear_threshold);
        e.fix(sleep.angular_threshold);
        e.u32(sleep.frames_to_sleep);

        let m = &self.material_table;
        e.combine(m.default_friction_combine);
        e.combine(m.default_restitution_combine);
        e.len(m.materials.len());
        for mat in &m.materials {
            e.u64(u64::from(mat.id));
            e.fix(mat.static_friction);
            e.fix(mat.dynamic_friction);
            e.fix(mat.restitution);
            e.combine(mat.friction_combine);
            e.combine(mat.restitution_combine);
        }
        let mut overrides: Vec<_> = m.pair_overrides.iter().collect();
        overrides.sort_by_key(|p| (p.mat_a, p.mat_b));
        e.len(overrides.len());
        for p in overrides {
            e.u64(u64::from(p.mat_a));
            e.u64(u64::from(p.mat_b));
            e.fix(p.friction);
            e.fix(p.restitution);
        }

        e.len(self.force_fields.len());
        for f in &self.force_fields {
            e.bool(f.enabled);
            match &f.affected_bodies {
                None => e.bool(false),
                Some(bodies) => {
                    let mut bodies = bodies.clone();
                    bodies.sort_unstable();
                    bodies.dedup();
                    e.bool(true);
                    e.len(bodies.len());
                    for b in bodies {
                        e.u64(b as u64);
                    }
                }
            }
            e.field(&f.field);
        }

        self.encode_participants(&mut e);
        e.0.finalize().into()
    }

    #[cfg(feature = "std")]
    fn encode_participants(&self, e: &mut Encoder) {
        use crate::world_participant::{PortAccess, StepRule};
        let participants = self.participant_law_parts();
        e.len(participants.len());
        for (kind, step_rule, ports) in &participants {
            e.u32(kind.get());
            match *step_rule {
                StepRule::FollowSubstep => e.bytes(b"follow-substep"),
                StepRule::Fixed(dt) => {
                    e.bytes(b"fixed");
                    e.fix(dt);
                }
                StepRule::Subcycle => e.bytes(b"subcycle"),
            }
            e.len(ports.len());
            for port in ports {
                e.u32(port.id().get());
                e.bytes(match port.access() {
                    PortAccess::Read => b"read",
                    PortAccess::Write => b"write",
                    PortAccess::ReadCommitted => b"read-committed",
                });
            }
        }
    }

    /// Without `std` no participant can be registered: the list is empty.
    #[cfg(not(feature = "std"))]
    fn encode_participants(&self, e: &mut Encoder) {
        e.len(0);
    }
}
