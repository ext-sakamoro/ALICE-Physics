//! The documented `law_id` encoding written out independently of `src/law_id.rs`, shared by
//! `tests/law_id_world.rs` (which checks it against production for every covered field) and by
//! the fixture tests that recompute a header `law_id` under an earlier writer's rules.
//!
//! Row 11 (the sleep rule) is folded as the writer encoded it: a `u32` step count up to the
//! semantics identifier `e90000da…` (`frames_to_sleep`), a `Fix128` time since `aa8c72e9…`
//! (`time_to_sleep`); [`Laws::legacy_sleep_steps`] selects the former.

#![allow(
    dead_code,
    reason = "shared by several test binaries, each of which uses a part of the encoder"
)]

use alice_physics_core::coupling_medium::DragMedium;
use alice_physics_core::electromagnetic::MagneticResponse;
use alice_physics_core::force::{ForceField, ForceFieldInstance};
use alice_physics_core::law_id::{LAW_ID_DOMAIN, WORLD_LAW_KIND};
use alice_physics_core::material::{CombineRule, MaterialId, PhysicsMaterial};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, PortAccess, StateError,
    StepRule, SubstepCtx,
};
use alice_physics_core::{SleepConfig, SolverBackend, WorldCcdConfig};
use sha2::{Digest, Sha256};

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

/// The covered rules of a world, as the test chooses them.
pub struct Laws {
    pub config: PhysicsConfig,
    contact_warm_start: Fix128,
    pub contact_stale_frames: u32,
    sdf_radius: Fix128,
    pub ccd: Option<Fix128>,
    sleep: SleepConfig,
    /// `Some(n)`: fold row 11 as the writers up to `e90000da…` did, a `u32` count of idle
    /// steps; `None`: as this build does, `sleep.time_to_sleep` as `Fix128`.
    pub legacy_sleep_steps: Option<u32>,
    /// `true`: fold as the writers before 4.0 did, without the contact rows (the speculative
    /// switch and each material's contact compliance); `false`: as this build does.
    pub legacy_without_contact_rows: bool,
    default_friction_combine: CombineRule,
    pub default_restitution_combine: CombineRule,
    /// Materials registered after the default one.
    pub materials: Vec<PhysicsMaterial>,
    /// `(mat_a, mat_b, friction, restitution)` in the order they are set.
    pub pair_overrides: Vec<(MaterialId, MaterialId, Fix128, Fix128)>,
    force_fields: Vec<ForceFieldInstance>,
    pub drag_participants: Vec<Fix128>,
    /// Participants registered after the drag media, each declaring this
    /// semantics identifier.
    pub declaring_participants: Vec<[u8; 32]>,
}

pub const KIND_DECLARES: ParticipantKind = ParticipantKind::new(0x4445_434c);

/// A participant with no state that declares a semantics identifier.
pub struct Declares([u8; 32]);

impl Participant for Declares {
    fn kind(&self) -> ParticipantKind {
        KIND_DECLARES
    }
    fn semantics_id(&self) -> Option<[u8; 32]> {
        Some(self.0)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

impl Laws {
    pub fn defaults() -> Self {
        Self {
            config: PhysicsConfig::default(),
            contact_warm_start: fx(8, 10),
            contact_stale_frames: 3,
            sdf_radius: PhysicsWorld::new(PhysicsConfig::default()).sdf_collision_radius,
            ccd: None,
            sleep: SleepConfig::default(),
            legacy_sleep_steps: None,
            legacy_without_contact_rows: false,
            default_friction_combine: CombineRule::Average,
            default_restitution_combine: CombineRule::Average,
            materials: Vec::new(),
            pair_overrides: Vec::new(),
            force_fields: Vec::new(),
            drag_participants: Vec::new(),
            declaring_participants: Vec::new(),
        }
    }

    /// Every covered field away from its default.
    pub fn everything() -> Self {
        let config = PhysicsConfig {
            substeps: 5,
            iterations: 7,
            gravity: v3(1, -3, 2),
            damping_rate: fx(3, 2),
            solver_backend: SolverBackend::Tgs,
            warm_start_factor: fx(1, 3),
            speculative_contacts: false,
        };
        let mut limited = ForceFieldInstance::new(ForceField::Drag {
            coefficient: fx(3, 10),
        });
        // unsorted with a repeat: encoded as {0, 2}
        limited.affected_bodies = Some(vec![2, 0, 2]);
        let mut disabled = ForceFieldInstance::new(ForceField::Directional {
            direction: v3(0, 0, 1),
            strength: fx(1, 7),
        });
        disabled.enabled = false;
        Self {
            config,
            contact_warm_start: fx(3, 5),
            contact_stale_frames: 9,
            sdf_radius: fx(5, 4),
            ccd: Some(fx(1, 9)),
            sleep: SleepConfig {
                linear_threshold: fx(1, 11),
                angular_threshold: fx(1, 13),
                time_to_sleep: Fix128::from_ratio(17, 60),
            },
            legacy_sleep_steps: None,
            legacy_without_contact_rows: false,
            default_friction_combine: CombineRule::Max,
            default_restitution_combine: CombineRule::Multiply,
            materials: vec![
                PhysicsMaterial {
                    id: 0,
                    static_friction: fx(9, 10),
                    dynamic_friction: fx(4, 10),
                    restitution: fx(2, 10),
                    friction_combine: CombineRule::Min,
                    restitution_combine: CombineRule::Max,
                    contact_compliance: fx(1, 1_000_000_000),
                },
                PhysicsMaterial {
                    id: 0,
                    static_friction: fx(1, 10),
                    dynamic_friction: fx(1, 20),
                    restitution: fx(7, 10),
                    friction_combine: CombineRule::Multiply,
                    restitution_combine: CombineRule::Average,
                    contact_compliance: alice_physics_core::material::DEFAULT_CONTACT_COMPLIANCE,
                },
            ],
            // set out of order and with the second pair given as (2, 1)
            pair_overrides: vec![(2, 1, fx(1, 4), fx(3, 4)), (0, 1, fx(2, 3), fx(1, 6))],
            force_fields: vec![
                ForceFieldInstance::new(ForceField::Directional {
                    direction: v3(1, 2, 3),
                    strength: fx(5, 2),
                }),
                ForceFieldInstance::new(ForceField::Point {
                    center: v3(4, 5, 6),
                    strength: fx(3, 2),
                    repulsive: true,
                    max_force: fx(40, 1),
                }),
                limited,
                ForceFieldInstance::new(ForceField::Buoyancy {
                    surface_y: fx(-2, 1),
                    density: fx(11, 10),
                    drag: fx(1, 5),
                }),
                ForceFieldInstance::new(ForceField::Vortex {
                    center: v3(7, 8, 9),
                    axis: v3(0, 1, 0),
                    strength: fx(6, 5),
                    falloff_radius: fx(12, 1),
                }),
                ForceFieldInstance::new(ForceField::Explosion {
                    center: v3(-1, -2, -3),
                    strength: fx(30, 1),
                    radius: fx(8, 1),
                    falloff_power: fx(2, 1),
                }),
                ForceFieldInstance::new(ForceField::Magnetic {
                    position: v3(3, 1, 4),
                    moment: v3(1, 5, 9),
                    strength: fx(2, 7),
                }),
                ForceFieldInstance::new(ForceField::InducedDipole {
                    position: v3(2, 7, 1),
                    moment: v3(8, 2, 8),
                    radius: fx(1, 50),
                    material: 1,
                    response: MagneticResponse {
                        chi_eff: fx(5, 2),
                        saturation_magnetization: fx(1_600_000, 1),
                    },
                    density: fx(7800, 1),
                }),
                disabled,
            ],
            drag_participants: vec![fx(1, 2)],
            declaring_participants: vec![[0x5d; 32]],
        }
    }

    pub fn build(&self) -> PhysicsWorld {
        let mut w = PhysicsWorld::new(self.config);
        w.contact_cache.warm_start_factor = self.contact_warm_start;
        w.contact_cache.max_stale_frames = self.contact_stale_frames;
        w.sdf_collision_radius = self.sdf_radius;
        if let Some(threshold) = self.ccd {
            w.set_continuous_collision(WorldCcdConfig::on().with_motion_threshold(threshold));
        }
        w.set_sleep_config(self.sleep);
        let t = &mut w.material_table;
        t.default_friction_combine = self.default_friction_combine;
        t.default_restitution_combine = self.default_restitution_combine;
        for m in &self.materials {
            t.register(*m);
        }
        for &(a, b, f, r) in &self.pair_overrides {
            t.set_pair_override(a, b, f, r);
        }
        for f in &self.force_fields {
            w.add_force_field(f.clone());
        }
        for &c in &self.drag_participants {
            let m = DragMedium::new(c, Vec3Fix::ZERO).expect("medium");
            w.add_participant(Box::new(m)).expect("register");
        }
        for &id in &self.declaring_participants {
            w.add_participant(Box::new(Declares(id))).expect("register");
        }
        w
    }
}

// ── The documented encoding, written out independently ──────────────────────

pub struct Spec(Vec<u8>);

impl Spec {
    fn len(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_be_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.0.extend_from_slice(b);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn fix(&mut self, v: Fix128) {
        self.0.extend_from_slice(&v.hi.to_be_bytes());
        self.0.extend_from_slice(&v.lo.to_be_bytes());
    }
    fn vec3(&mut self, v: Vec3Fix) {
        self.fix(v.x);
        self.fix(v.y);
        self.fix(v.z);
    }
    fn rule(&mut self, r: CombineRule) {
        self.bytes(match r {
            CombineRule::Average => b"average",
            CombineRule::Min => b"min",
            CombineRule::Max => b"max",
            CombineRule::Multiply => b"multiply",
        });
    }
    fn material(&mut self, id: MaterialId, m: &PhysicsMaterial, with_compliance: bool) {
        self.u64(u64::from(id));
        self.fix(m.static_friction);
        self.fix(m.dynamic_friction);
        self.fix(m.restitution);
        self.rule(m.friction_combine);
        self.rule(m.restitution_combine);
        if with_compliance {
            self.fix(m.contact_compliance);
        }
    }
}

/// The documented encoding of `laws`, hashed.
pub fn spec(laws: &Laws, semantics: &[u8; 32]) -> [u8; 32] {
    Sha256::digest(spec_bytes(laws, semantics)).into()
}

/// The documented encoding of `laws`, before hashing.
pub fn spec_bytes(laws: &Laws, semantics: &[u8; 32]) -> Vec<u8> {
    let c = &laws.config;
    let mut s = Spec(Vec::new());
    s.bytes(LAW_ID_DOMAIN);
    s.bytes(WORLD_LAW_KIND);
    s.0.extend_from_slice(semantics);
    s.u64(c.substeps as u64);
    s.u64(c.iterations as u64);
    s.vec3(c.gravity);
    s.fix(c.damping_rate);
    s.bytes(match c.solver_backend {
        SolverBackend::Xpbd => b"xpbd",
        SolverBackend::Tgs => b"tgs",
        other => panic!("no documented encoding for {other:?}"),
    });
    s.fix(c.warm_start_factor);
    if !laws.legacy_without_contact_rows {
        s.bool(c.speculative_contacts);
    }
    s.fix(laws.contact_warm_start);
    s.u32(laws.contact_stale_frames);
    s.fix(laws.sdf_radius);
    s.bool(laws.ccd.is_some());
    s.fix(laws.ccd.unwrap_or(Fix128::ONE));
    s.fix(laws.sleep.linear_threshold);
    s.fix(laws.sleep.angular_threshold);
    match laws.legacy_sleep_steps {
        Some(steps) => s.u32(steps),
        None => s.fix(laws.sleep.time_to_sleep),
    }

    s.rule(laws.default_friction_combine);
    s.rule(laws.default_restitution_combine);
    // the default material is id 0; registered ones follow in order
    s.len(1 + laws.materials.len());
    s.material(
        0,
        &PhysicsMaterial::default(),
        !laws.legacy_without_contact_rows,
    );
    for (i, m) in laws.materials.iter().enumerate() {
        s.material(
            MaterialId::try_from(i + 1).expect("id"),
            m,
            !laws.legacy_without_contact_rows,
        );
    }
    // pair overrides: ordered within the pair, sorted, the last value of a pair wins
    let mut pairs: Vec<(MaterialId, MaterialId, Fix128, Fix128)> = Vec::new();
    for &(a, b, f, r) in &laws.pair_overrides {
        let (a, b) = (a.min(b), a.max(b));
        pairs.retain(|p| (p.0, p.1) != (a, b));
        pairs.push((a, b, f, r));
    }
    pairs.sort_by_key(|p| (p.0, p.1));
    s.len(pairs.len());
    for (a, b, f, r) in pairs {
        s.u64(u64::from(a));
        s.u64(u64::from(b));
        s.fix(f);
        s.fix(r);
    }

    s.len(laws.force_fields.len());
    for f in &laws.force_fields {
        s.bool(f.enabled);
        match &f.affected_bodies {
            None => s.bool(false),
            Some(list) => {
                let mut list = list.clone();
                list.sort_unstable();
                list.dedup();
                s.bool(true);
                s.len(list.len());
                for b in list {
                    s.u64(b as u64);
                }
            }
        }
        match f.field {
            ForceField::Directional {
                direction,
                strength,
            } => {
                s.bytes(b"directional");
                s.vec3(direction);
                s.fix(strength);
            }
            ForceField::Point {
                center,
                strength,
                repulsive,
                max_force,
            } => {
                s.bytes(b"point");
                s.vec3(center);
                s.fix(strength);
                s.bool(repulsive);
                s.fix(max_force);
            }
            ForceField::Drag { coefficient } => {
                s.bytes(b"drag");
                s.fix(coefficient);
            }
            ForceField::Buoyancy {
                surface_y,
                density,
                drag,
            } => {
                s.bytes(b"buoyancy");
                s.fix(surface_y);
                s.fix(density);
                s.fix(drag);
            }
            ForceField::Vortex {
                center,
                axis,
                strength,
                falloff_radius,
            } => {
                s.bytes(b"vortex");
                s.vec3(center);
                s.vec3(axis);
                s.fix(strength);
                s.fix(falloff_radius);
            }
            ForceField::Explosion {
                center,
                strength,
                radius,
                falloff_power,
            } => {
                s.bytes(b"explosion");
                s.vec3(center);
                s.fix(strength);
                s.fix(radius);
                s.fix(falloff_power);
            }
            ForceField::Magnetic {
                position,
                moment,
                strength,
            } => {
                s.bytes(b"magnetic");
                s.vec3(position);
                s.vec3(moment);
                s.fix(strength);
            }
            ForceField::InducedDipole {
                position,
                moment,
                radius,
                material,
                response,
                density,
            } => {
                s.bytes(b"induced-dipole");
                s.vec3(position);
                s.vec3(moment);
                s.fix(radius);
                s.u64(u64::from(material));
                s.fix(response.chi_eff);
                s.fix(response.saturation_magnetization);
                s.fix(density);
            }
            other => panic!("no documented encoding for {other:?}"),
        }
    }

    // participants: kind, step rule and ports as the participant declares them
    s.len(laws.drag_participants.len() + laws.declaring_participants.len());
    for &coefficient in &laws.drag_participants {
        let p = DragMedium::new(coefficient, Vec3Fix::ZERO).expect("medium");
        s.u32(p.kind().get());
        match p.step_rule() {
            StepRule::FollowSubstep => s.bytes(b"follow-substep"),
            StepRule::Fixed(dt) => {
                s.bytes(b"fixed");
                s.fix(dt);
            }
            StepRule::Subcycle => s.bytes(b"subcycle"),
            other => panic!("no documented encoding for {other:?}"),
        }
        s.len(p.ports().len());
        for port in p.ports() {
            s.u32(port.id().get());
            s.bytes(match port.access() {
                PortAccess::Read => b"read",
                PortAccess::Write => b"write",
                PortAccess::ReadCommitted => b"read-committed",
                other => panic!("no documented encoding for {other:?}"),
            });
        }
    }
    // a declaring participant: its kind, follow-substep, no ports
    for _ in &laws.declaring_participants {
        s.u32(KIND_DECLARES.get());
        s.bytes(b"follow-substep");
        s.len(0);
    }
    // row 17: only when a participant declares a semantics identifier
    if !laws.declaring_participants.is_empty() {
        s.bytes(b"participant-semantics");
        s.len(laws.declaring_participants.len());
        for (i, id) in laws.declaring_participants.iter().enumerate() {
            s.u64((laws.drag_participants.len() + i) as u64);
            s.0.extend_from_slice(id);
        }
    }
    s.0
}
