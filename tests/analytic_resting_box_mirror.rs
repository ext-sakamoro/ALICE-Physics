//! A box set down flat on a flat floor does not start to turn: the shares of its mirror-image
//! contact points cancel to the bit, so its angular velocity stays exactly zero and its
//! orientation exactly the identity.
//!
//! oracle (statics and symmetry): the box and the floor are symmetric under x → −x and z → −z
//! about the box's centre; every corner's contact share has a mirror partner with the opposite
//! arm, so the summed torque is zero exactly when the product is odd ((−a)·b = −(a·b)). With
//! `Fix128`'s product floored toward −∞ the x and z sums kept a few units from the first tick
//! (measured: up to 457 raw units of ω on a plane and on a height field, the rotation drifting
//! off the identity); with the product rounded to nearest, ties to even, both stay exactly zero
//! for the 60 ticks without any local odd helper.
//!
//! Floors: a plane and a flat height field (both flat and symmetric under the reflections).
//! XPBD, 240 Hz, 60 ticks, the default substeps. Compared: 2 floors × 60 ticks, 0 compared fails.
use alice_physics_core::heightfield::HeightField;
use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::solver::SolverConfig;
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::{PhysicsWorld, PlaneCollider, Shape, SolverBackend};

const G: f64 = 9.81;
const HALF: (f64, f64, f64) = (0.5, 0.25, 0.375);

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

#[derive(Clone, Copy, Debug)]
enum Floor {
    Plane,
    HeightField,
}

fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

#[test]
fn a_box_set_down_flat_stays_exactly_unturned_on_flat_floors() {
    let mut compared = 0u32;
    for floor in [Floor::Plane, Floor::HeightField] {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: v3(0.0, -G, 0.0),
            solver_backend: SolverBackend::Xpbd,
            ..SolverConfig::default()
        });
        match floor {
            Floor::Plane => {
                w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                    v3(0.0, 1.0, 0.0),
                    Fix128::ZERO,
                )));
            }
            Floor::HeightField => {
                w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
                    41,
                    41,
                    fx(1.0),
                    v3(-20.0, 0.0, -20.0),
                    Fix128::ZERO,
                )));
            }
        }
        let m = w
            .material_table
            .register(PhysicsMaterial::new(0, fx(0.4), Fix128::ZERO));
        w.material_table
            .set_pair_override(m, DEFAULT_MATERIAL, fx(0.4), Fix128::ZERO);
        let shape = Shape::Box {
            half_extents: v3(HALF.0, HALF.1, HALF.2),
        };
        let b = w
            .add_shaped_body(&shape, fx(500.0), v3(0.0, HALF.1, 0.0))
            .expect("box");
        w.set_body_material(b, m);
        for tick in 0..60 {
            w.step(fx(1.0 / 240.0));
            let body = w.get_body(b).expect("box");
            let om = body.angular_velocity;
            assert_eq!(
                (raw(om.x), raw(om.z)),
                (0, 0),
                "{floor:?} tick {tick}: the box turns (ω raw x, z)"
            );
            assert_eq!(
                body.rotation,
                QuatFix::IDENTITY,
                "{floor:?} tick {tick}: the box tilted"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 120);
}
