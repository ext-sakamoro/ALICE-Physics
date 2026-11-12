//! `PhysicsWorld::law_id`: what it covers, what it leaves out, and the
//! encoding its documentation states.
//!
//! oracle: the encoding table in the documentation of `PhysicsWorld::law_id`,
//! written out independently here (`spec`). It encodes the values the test
//! sets (a `Laws` description), not values read back from the world, and is
//! checked against the default world and against a world that sets every
//! covered field to a value other than its default (all eight force field
//! kinds, a limited and a disabled field, pair overrides, a material whose
//! frictions differ, continuous collision, a participant, a participant that
//! declares a semantics identifier). One golden value, taken with an all-zero
//! semantics identifier, pins the encoding by itself. A decoder written from the
//! same table (`rows_1_to_16_end`) shows that rows 1 to 16 end at a point their
//! own lengths and counts fix, so row 17 (written only when a participant
//! declares an identifier) cannot be confused with them.

#![cfg(feature = "std")]

use alice_physics_core::coupling_medium::DragMedium;
use alice_physics_core::force::{ForceField, ForceFieldInstance};
use alice_physics_core::material::{CombineRule, MaterialId, PhysicsMaterial};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::semantics::PHYSICS_SEMANTICS_ID;
use alice_physics_core::solver::{Broadphase, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::{SleepConfig, SolverBackend, WorldCcdConfig};

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

// ── What a test sets, and how it builds the world ───────────────────────────

#[path = "support/law_rows.rs"]
mod law_rows;
use law_rows::{spec, spec_bytes, Laws};

// ── Decoding the documented rows ────────────────────────────────────────────

/// A cursor over an encoding, reading the primitives of the documented table.
struct Rows<'a>(&'a [u8], usize);

impl Rows<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let out = &self.0[self.1..self.1 + n];
        self.1 += n;
        out
    }
    fn u64(&mut self) -> u64 {
        u64::from_be_bytes(self.take(8).try_into().expect("8 bytes"))
    }
    fn len(&mut self) -> usize {
        usize::try_from(self.u64()).expect("length")
    }
    fn bytes(&mut self) -> Vec<u8> {
        let n = self.len();
        self.take(n).to_vec()
    }
    fn skip(&mut self, n: usize) {
        self.take(n);
    }
}

const FIX: usize = 16;
const VEC3: usize = 3 * FIX;

/// The offset where row 16 ends, read from the documented table alone: every
/// row's length follows from its own lengths, counts and tags.
fn rows_1_to_16_end(b: &[u8]) -> usize {
    let mut r = Rows(b, 0);
    r.bytes(); // 1 domain
    r.bytes(); // 2 kind
    r.skip(32); // 3 semantics
    r.skip(8 + 8); // 4 substeps, iterations
    r.skip(VEC3 + FIX); // 5 gravity, damping
    r.bytes(); // 6 backend
    r.skip(FIX + 1); // 7 warm start, speculative contacts
    r.skip(FIX + 4); // 8 contact cache
    r.skip(FIX); // 9 sdf radius
    r.skip(1 + FIX); // 10 continuous collision
    r.skip(FIX + FIX + FIX); // 11 sleep (thresholds and the time to sleep)
    r.bytes(); // 12 friction combine
    r.bytes(); //    restitution combine
    for _ in 0..r.len() {
        // 13 material: id, three Fix128, two tags, contact compliance
        r.skip(8 + 3 * FIX);
        r.bytes();
        r.bytes();
        r.skip(FIX);
    }
    for _ in 0..r.len() {
        r.skip(8 + 8 + FIX + FIX); // 14 pair override
    }
    for _ in 0..r.len() {
        // 15 force field: enabled, optional body list, tag, parameters
        r.skip(1);
        if r.take(1)[0] == 1 {
            let n = r.len();
            r.skip(8 * n);
        }
        let params = match r.bytes().as_slice() {
            b"directional" => VEC3 + FIX,
            b"point" => VEC3 + FIX + 1 + FIX,
            b"drag" => FIX,
            b"buoyancy" => 3 * FIX,
            b"vortex" => 2 * VEC3 + 2 * FIX,
            b"explosion" => VEC3 + 3 * FIX,
            b"magnetic" => 2 * VEC3 + FIX,
            b"induced-dipole" => 2 * VEC3 + FIX + 8 + 3 * FIX,
            other => panic!("no documented field tag {other:?}"),
        };
        r.skip(params);
    }
    for _ in 0..r.len() {
        // 16 participant: kind, step rule, ports
        r.skip(4);
        if r.bytes().as_slice() == b"fixed" {
            r.skip(FIX);
        }
        for _ in 0..r.len() {
            r.skip(4);
            r.bytes();
        }
    }
    r.1
}

/// Rows 1 to 16 end at a point fixed by their own contents: the encoding ends there when no
/// participant declares a semantics identifier, and row 17 follows otherwise and reads to the
/// end, so an encoding with row 17 is never an encoding without it
#[test]
fn rows_1_to_16_end_at_a_fixed_point_and_row_17_follows() {
    let mut without = Laws::everything();
    without.declaring_participants.clear();
    for (name, laws) in [
        ("defaults", Laws::defaults()),
        ("everything without row 17", without),
    ] {
        let b = spec_bytes(&laws, &PHYSICS_SEMANTICS_ID);
        assert_eq!(rows_1_to_16_end(&b), b.len(), "{name}");
    }

    let mut laws = Laws::everything();
    laws.declaring_participants.push([0x6e; 32]);
    let b = spec_bytes(&laws, &PHYSICS_SEMANTICS_ID);
    let mut r = Rows(&b, rows_1_to_16_end(&b));
    assert_eq!(r.bytes(), b"participant-semantics");
    assert_eq!(r.len(), 2);
    let drags = laws.drag_participants.len() as u64;
    for (i, id) in laws.declaring_participants.iter().enumerate() {
        assert_eq!(r.u64(), drags + i as u64);
        assert_eq!(r.take(32), id);
    }
    assert_eq!(r.1, b.len(), "row 17 reads to the end");
    assert_eq!(
        laws.build().law_id(&PHYSICS_SEMANTICS_ID),
        spec(&laws, &PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(
        alice_physics_core::law_id::PARTICIPANT_SEMANTICS_ROW,
        b"participant-semantics"
    );
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
/// Re-pinned (from `95e0102b…`) when the world damping field became a rate
/// per second with default 0, where it had been a retention per step with
/// default 0.99: a default of a covered field moved.
const GOLDEN_DEFAULT_LAW_ID: &str =
    "19259542111e7ca54786d4e3a59f04d7146b5d421a31af828a7b790371bc33c3";

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
        ("damping", Box::new(|w| w.config.damping_rate = Fix128::ONE)),
        (
            "backend",
            Box::new(|w| w.config.solver_backend = SolverBackend::Tgs),
        ),
        (
            "config warm start",
            Box::new(|w| w.config.warm_start_factor = Fix128::ZERO),
        ),
        (
            "speculative contacts",
            Box::new(|w| w.config.speculative_contacts = false),
        ),
        (
            "material contact compliance",
            Box::new(|w| {
                w.material_table.register(PhysicsMaterial {
                    contact_compliance: fx(1, 1_000_000_000),
                    ..PhysicsMaterial::default()
                });
            }),
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
                    time_to_sleep: Fix128::from_ratio(7, 60),
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
