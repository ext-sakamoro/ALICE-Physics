//! `PhysicsWorld::law_id`: what it covers, what it leaves out, and the
//! encoding its documentation states.
//!
//! oracle: the encoding table in the documentation of `PhysicsWorld::law_id`,
//! written out independently here (`spec`). It encodes the values the test
//! sets (a `Laws` description), not values read back from the world, and is
//! checked against the default world and against a world that sets every
//! covered field to a value other than its default (all seven force field
//! kinds, a limited and a disabled field, pair overrides, a material whose
//! frictions differ, continuous collision, a participant). One golden value,
//! taken with an all-zero semantics identifier, pins the encoding by itself.

#![cfg(feature = "std")]

use alice_physics::coupling_medium::DragMedium;
use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::law_id::{LAW_ID_DOMAIN, WORLD_LAW_KIND};
use alice_physics::material::{CombineRule, MaterialId, PhysicsMaterial};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::semantics::PHYSICS_SEMANTICS_ID;
use alice_physics::solver::{Broadphase, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{Participant, PortAccess, StepRule};
use alice_physics::{SleepConfig, SolverBackend, WorldCcdConfig};
use sha2::{Digest, Sha256};

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn id(w: &PhysicsWorld) -> [u8; 32] {
    w.law_id(&PHYSICS_SEMANTICS_ID)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

// ── What a test sets, and how it builds the world ───────────────────────────

/// The covered rules of a world, as the test chooses them.
struct Laws {
    config: PhysicsConfig,
    contact_warm_start: Fix128,
    contact_stale_frames: u32,
    sdf_radius: Fix128,
    ccd: Option<Fix128>,
    sleep: SleepConfig,
    default_friction_combine: CombineRule,
    default_restitution_combine: CombineRule,
    /// Materials registered after the default one.
    materials: Vec<PhysicsMaterial>,
    /// `(mat_a, mat_b, friction, restitution)` in the order they are set.
    pair_overrides: Vec<(MaterialId, MaterialId, Fix128, Fix128)>,
    force_fields: Vec<ForceFieldInstance>,
    drag_participants: Vec<Fix128>,
}

impl Laws {
    fn defaults() -> Self {
        Self {
            config: PhysicsConfig::default(),
            contact_warm_start: fx(8, 10),
            contact_stale_frames: 3,
            sdf_radius: world().sdf_collision_radius,
            ccd: None,
            sleep: SleepConfig::default(),
            default_friction_combine: CombineRule::Average,
            default_restitution_combine: CombineRule::Average,
            materials: Vec::new(),
            pair_overrides: Vec::new(),
            force_fields: Vec::new(),
            drag_participants: Vec::new(),
        }
    }

    /// Every covered field away from its default.
    fn everything() -> Self {
        let config = PhysicsConfig {
            substeps: 5,
            iterations: 7,
            gravity: v3(1, -3, 2),
            damping: fx(97, 100),
            solver_backend: SolverBackend::Tgs,
            warm_start_factor: fx(1, 3),
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
                frames_to_sleep: 17,
            },
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
                },
                PhysicsMaterial {
                    id: 0,
                    static_friction: fx(1, 10),
                    dynamic_friction: fx(1, 20),
                    restitution: fx(7, 10),
                    friction_combine: CombineRule::Multiply,
                    restitution_combine: CombineRule::Average,
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
                disabled,
            ],
            drag_participants: vec![fx(1, 2)],
        }
    }

    fn build(&self) -> PhysicsWorld {
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
        w
    }
}

// ── The documented encoding, written out independently ──────────────────────

struct Spec(Vec<u8>);

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
    fn material(&mut self, id: MaterialId, m: &PhysicsMaterial) {
        self.u64(u64::from(id));
        self.fix(m.static_friction);
        self.fix(m.dynamic_friction);
        self.fix(m.restitution);
        self.rule(m.friction_combine);
        self.rule(m.restitution_combine);
    }
}

/// The documented encoding of `laws`.
fn spec(laws: &Laws, semantics: &[u8; 32]) -> [u8; 32] {
    let c = &laws.config;
    let mut s = Spec(Vec::new());
    s.bytes(LAW_ID_DOMAIN);
    s.bytes(WORLD_LAW_KIND);
    s.0.extend_from_slice(semantics);
    s.u64(c.substeps as u64);
    s.u64(c.iterations as u64);
    s.vec3(c.gravity);
    s.fix(c.damping);
    s.bytes(match c.solver_backend {
        SolverBackend::Xpbd => b"xpbd",
        SolverBackend::Tgs => b"tgs",
        other => panic!("no documented encoding for {other:?}"),
    });
    s.fix(c.warm_start_factor);
    s.fix(laws.contact_warm_start);
    s.u32(laws.contact_stale_frames);
    s.fix(laws.sdf_radius);
    s.bool(laws.ccd.is_some());
    s.fix(laws.ccd.unwrap_or(Fix128::ONE));
    s.fix(laws.sleep.linear_threshold);
    s.fix(laws.sleep.angular_threshold);
    s.u32(laws.sleep.frames_to_sleep);

    s.rule(laws.default_friction_combine);
    s.rule(laws.default_restitution_combine);
    // the default material is id 0; registered ones follow in order
    s.len(1 + laws.materials.len());
    s.material(0, &PhysicsMaterial::default());
    for (i, m) in laws.materials.iter().enumerate() {
        s.material(MaterialId::try_from(i + 1).expect("id"), m);
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
        }
    }

    // participants: kind, step rule and ports as the participant declares them
    s.len(laws.drag_participants.len());
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
    Sha256::digest(&s.0).into()
}

#[test]
fn the_default_world_is_encoded_as_documented() {
    let laws = Laws::defaults();
    for semantics in [PHYSICS_SEMANTICS_ID, [0; 32], [7; 32]] {
        assert_eq!(world().law_id(&semantics), spec(&laws, &semantics));
    }
}

#[test]
fn a_world_that_sets_every_covered_field_is_encoded_as_documented() {
    let laws = Laws::everything();
    let w = laws.build();
    assert_eq!(id(&w), spec(&laws, &PHYSICS_SEMANTICS_ID));
    assert_ne!(id(&w), id(&world()));
}

/// The law identifier of the default world, with an all-zero semantics
/// identifier so that it pins the encoding alone: it moves when the encoding
/// or a default of a covered field moves, not when `PHYSICS_SEMANTICS_ID`
/// does (that path is covered by `the_default_world_is_encoded_as_documented`).
const GOLDEN_DEFAULT_LAW_ID: &str =
    "95e0102b9a68da440561011710aad4983b68edb8ad7f338b7f8da6da1aad53f1";

#[test]
fn golden_default_world_law_id() {
    assert_eq!(hex(&world().law_id(&[0; 32])), GOLDEN_DEFAULT_LAW_ID);
}

#[test]
fn a_different_semantics_id_gives_a_different_law_id() {
    let w = world();
    let mut other = PHYSICS_SEMANTICS_ID;
    other[0] ^= 1;
    assert_ne!(w.law_id(&PHYSICS_SEMANTICS_ID), w.law_id(&other));
}

// ── Every covered field moves the identifier ────────────────────────────────

fn changed(edit: impl FnOnce(&mut PhysicsWorld)) -> [u8; 32] {
    let mut w = world();
    edit(&mut w);
    id(&w)
}

/// A named change to a default world.
type Edit = Box<dyn Fn(&mut PhysicsWorld)>;

#[test]
fn every_covered_field_moves_the_law_id() {
    let base = id(&world());
    let edits: Vec<(&str, Edit)> = vec![
        ("substeps", Box::new(|w| w.config.substeps += 1)),
        ("iterations", Box::new(|w| w.config.iterations += 1)),
        ("gravity", Box::new(|w| w.config.gravity.x = Fix128::ONE)),
        ("damping", Box::new(|w| w.config.damping = Fix128::ONE)),
        (
            "backend",
            Box::new(|w| w.config.solver_backend = SolverBackend::Tgs),
        ),
        (
            "config warm start",
            Box::new(|w| w.config.warm_start_factor = Fix128::ZERO),
        ),
        (
            "contact warm start",
            Box::new(|w| w.contact_cache.warm_start_factor = fx(1, 2)),
        ),
        (
            "contact stale frames",
            Box::new(|w| w.contact_cache.max_stale_frames += 1),
        ),
        (
            "sdf collision radius",
            Box::new(|w| w.sdf_collision_radius = Fix128::from_int(3)),
        ),
        (
            "ccd on",
            Box::new(|w| w.set_continuous_collision(WorldCcdConfig::on())),
        ),
        (
            "ccd threshold",
            Box::new(|w| {
                w.set_continuous_collision(
                    WorldCcdConfig::new().with_motion_threshold(Fix128::from_int(2)),
                )
            }),
        ),
        (
            "sleep frames",
            Box::new(|w| {
                w.set_sleep_config(SleepConfig {
                    frames_to_sleep: 7,
                    ..SleepConfig::default()
                })
            }),
        ),
        (
            "sleep linear threshold",
            Box::new(|w| {
                w.set_sleep_config(SleepConfig {
                    linear_threshold: Fix128::ONE,
                    ..SleepConfig::default()
                })
            }),
        ),
        (
            "sleep angular threshold",
            Box::new(|w| {
                w.set_sleep_config(SleepConfig {
                    angular_threshold: Fix128::ONE,
                    ..SleepConfig::default()
                })
            }),
        ),
        (
            "material",
            Box::new(|w| {
                w.material_table.register_rubber();
            }),
        ),
        (
            "friction combine",
            Box::new(|w| w.material_table.default_friction_combine = CombineRule::Max),
        ),
        (
            "restitution combine",
            Box::new(|w| w.material_table.default_restitution_combine = CombineRule::Min),
        ),
        (
            "pair override",
            Box::new(|w| {
                w.material_table
                    .set_pair_override(0, 0, Fix128::ONE, Fix128::ZERO)
            }),
        ),
        (
            "force field",
            Box::new(|w| {
                w.add_force_field(ForceFieldInstance::new(ForceField::Drag {
                    coefficient: Fix128::ONE,
                }));
            }),
        ),
        (
            "participant",
            Box::new(|w| {
                let m = DragMedium::new(Fix128::ONE, Vec3Fix::ZERO).expect("medium");
                w.add_participant(Box::new(m)).expect("register");
            }),
        ),
    ];
    let mut seen = vec![base];
    for (name, edit) in &edits {
        let mut w = world();
        edit(&mut w);
        let got = id(&w);
        assert!(
            !seen.contains(&got),
            "{name} does not move the law id (or collides)"
        );
        seen.push(got);
    }
}

#[test]
fn force_field_parameters_and_scope_move_the_law_id() {
    let drag = |c: i64| {
        ForceFieldInstance::new(ForceField::Drag {
            coefficient: Fix128::from_int(c),
        })
    };
    let a = changed(|w| {
        w.add_force_field(drag(1));
    });
    let b = changed(|w| {
        w.add_force_field(drag(2));
    });
    let c = changed(|w| {
        let mut f = drag(1);
        f.enabled = false;
        w.add_force_field(f);
    });
    let d = changed(|w| {
        let mut f = drag(1);
        f.affected_bodies = Some(vec![0]);
        w.add_force_field(f);
    });
    let d1 = changed(|w| {
        let mut f = drag(1);
        f.affected_bodies = Some(vec![1]);
        w.add_force_field(f);
    });
    let e = changed(|w| {
        w.add_force_field(ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::from_int(0, 1, 0),
            strength: Fix128::ONE,
        }));
    });
    let all = [a, b, c, d, d1, e];
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i], all[j], "force fields {i} and {j} share a law id");
        }
    }
}

// ── Orders the step does not depend on are not part of the identifier ───────

#[test]
fn pair_override_and_scope_order_do_not_move_the_law_id() {
    let pairs = |order: &[(MaterialId, MaterialId)]| {
        changed(|w| {
            w.material_table.register_ice();
            w.material_table.register_rubber();
            for &(a, b) in order {
                let v = Fix128::from_int(i64::from(a.min(b)) * 10 + i64::from(a.max(b)));
                w.material_table.set_pair_override(a, b, v, v);
            }
        })
    };
    assert_eq!(pairs(&[(0, 1), (1, 2)]), pairs(&[(2, 1), (1, 0)]));

    let scoped = |bodies: Vec<usize>| {
        changed(|w| {
            let mut f = ForceFieldInstance::new(ForceField::Drag {
                coefficient: Fix128::ONE,
            });
            f.affected_bodies = Some(bodies);
            w.add_force_field(f);
        })
    };
    assert_eq!(scoped(vec![0, 2]), scoped(vec![2, 0, 0]));
    assert_ne!(scoped(vec![0, 2]), scoped(vec![0, 1]));
}

#[test]
fn force_field_order_is_part_of_the_law_id() {
    let fields = |first: i64, second: i64| {
        changed(|w| {
            for c in [first, second] {
                w.add_force_field(ForceFieldInstance::new(ForceField::Drag {
                    coefficient: Fix128::from_int(c),
                }));
            }
        })
    };
    assert_ne!(fields(1, 2), fields(2, 1));
}

// ── What is left out ────────────────────────────────────────────────────────

#[test]
fn state_broadphase_and_sleep_skip_are_left_out() {
    let base = id(&world());
    // the state the law acts on
    assert_eq!(
        changed(|w| {
            w.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(1, 2, 3),
                Fix128::ONE,
            ));
        }),
        base
    );
    // every broadphase gives the same bits
    for kind in [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid] {
        assert_eq!(changed(|w| w.set_broadphase(kind)), base, "{kind:?}");
    }
    // the sleep skip gives the same bits (measured: on and off step a world
    // that falls asleep and is woken to identical bodies over 200 steps)
    assert_eq!(changed(|w| w.set_sleep_skip(false)), base);
}

#[test]
fn the_same_world_built_twice_has_one_law_id() {
    let laws = Laws::everything();
    assert_eq!(id(&laws.build()), id(&laws.build()));
}
