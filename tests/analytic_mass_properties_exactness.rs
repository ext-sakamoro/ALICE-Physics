//! Mass properties that have an exact closed form must come out exact.
//!
//! `I = m/12·(…)` and `I = 2/5·m·r²` were written with `from_ratio(1, 12)` and
//! `from_ratio(2, 5)`: a constant that is not a dyadic rational is rounded once in
//! Q64.64, and multiplying it by the mass keeps that error (`12 · (1/12)` is not 1).
//! So a box of mass 12 did not have `I = h² + d²` exactly. Dividing at the end rounds
//! once, correctly. Each case is chosen so that the exact answer is a number Q64.64
//! holds (a small integer), so the check is bit equality, not a tolerance.
//!
//! Author: Moroya Sakamoto

use alice_physics::mass_properties::{box_mass_properties, cylinder_mass_properties};
use alice_physics::math::{Fix128, Vec3Fix};

fn int(v: i64) -> Fix128 {
    Fix128::from_int(v)
}

/// A box with half-extents (3, 2, 1) at density 1/4: mass = 6·4·2 · 1/4 = 12, and
/// `Ixx = 12/12·(4² + 2²) = 20`, `Iyy = 12/12·(6² + 2²) = 40`, `Izz = 12/12·(6² + 4²) = 52`.
#[test]
fn a_box_of_mass_twelve_has_integer_inertia() {
    let p = box_mass_properties(
        Vec3Fix::new(int(3), int(2), int(1)),
        Fix128::from_ratio(1, 4),
    );
    assert_eq!(p.mass, int(12), "mass");
    assert_eq!(p.inertia_tensor.col0.x, int(20), "Ixx");
    assert_eq!(p.inertia_tensor.col1.y, int(40), "Iyy");
    assert_eq!(p.inertia_tensor.col2.z, int(52), "Izz");
}

/// A cube of side 2 (half-extent 1) at density 3/2: mass 12, `I = 12/12·(4 + 4) = 8` on every axis.
#[test]
fn a_cube_of_mass_twelve_has_inertia_eight() {
    let p = box_mass_properties(
        Vec3Fix::new(int(1), int(1), int(1)),
        Fix128::from_ratio(3, 2),
    );
    assert_eq!(p.mass, int(12));
    for (name, v) in [
        ("Ixx", p.inertia_tensor.col0.x),
        ("Iyy", p.inertia_tensor.col1.y),
        ("Izz", p.inertia_tensor.col2.z),
    ] {
        assert_eq!(v, int(8), "{name}");
    }
}

use alice_physics::mass_properties::{capsule_mass_properties, sphere_mass_properties};

/// A cylinder (radius r, half-height h) has `Ixx = m/12·(3r² + (2h)²)`. Radius 2 and
/// half-height 1.5 give `3·4 + 9 = 21`; a mass of 12 would need `π`, so the check is
/// against the closed form built from the reported mass: `Ixx = mass·21/12`, and the
/// product `Ixx · 12 / mass` must be exactly 21 (the division rounds once, the old
/// `mass · (1/12)` rounded twice).
#[test]
fn a_cylinders_transverse_inertia_is_the_mass_over_twelve_times_its_shape_factor() {
    let (r, hh) = (int(2), Fix128::from_ratio(3, 2));
    let p = cylinder_mass_properties(r, hh, Fix128::ONE);
    // mass·(3r² + h²)/12 with h = 2·hh, written the same way: bit equal
    let h = hh * int(2);
    let want = p.mass * (int(3) * r * r + h * h) / int(12);
    assert_eq!(p.inertia_tensor.col0.x, want, "Ixx");
    assert_eq!(p.inertia_tensor.col2.z, want, "Izz equals Ixx");
}

/// A sphere has `I = 2/5·m·r²`: radius 1, the exact form `2·m/5`.
#[test]
fn a_sphere_has_two_fifths_m_r_squared() {
    let p = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
    let want = int(2) * p.mass / int(5);
    assert_eq!(p.inertia_tensor.col0.x, want);
    assert_eq!(p.inertia_tensor.col1.y, want);
    assert_eq!(p.inertia_tensor.col2.z, want);
}

/// A capsule with no cylinder part (half-height 0) is a sphere: the same inertia, bit for bit.
#[test]
fn a_capsule_with_no_cylinder_is_a_sphere() {
    let c = capsule_mass_properties(int(1), Fix128::ZERO, Fix128::ONE);
    let s = sphere_mass_properties(int(1), Fix128::ONE);
    assert_eq!(c.mass, s.mass);
    assert_eq!(c.inertia_tensor.col1.y, s.inertia_tensor.col1.y, "axial");
}

/// A long capsule (half-height 3, radius 1): the cylinder part dominates, so its
/// transverse inertia `m_c/12·(3r² + (2h)²)` must be the exact quotient. The
/// capsule's `Ixx` is that plus the (separately exact) sphere part; the cylinder part
/// is recovered by subtracting the sphere's own contribution, which has a closed form
/// built from the reported capsule mass.
#[test]
fn a_long_capsule_has_the_exact_cylinder_transverse_inertia() {
    use alice_physics::mass_properties::capsule_mass_properties;
    let (r, hh) = (int(1), int(3));
    let c = capsule_mass_properties(r, hh, Fix128::ONE);
    // cylinder mass = π r² (2 hh) ; sphere mass = 4/3 π r³ ; total mass is reported
    let cyl_mass = Fix128::PI * r * r * (hh * int(2));
    let cyl_ixx = cyl_mass * (int(3) * r * r + (hh * int(2)) * (hh * int(2))) / int(12);
    let sph_mass = c.mass - cyl_mass;
    let hemi_offset = hh + Fix128::from_ratio(3, 8) * r;
    let hemi_own = Fix128::from_ratio(83, 320) * sph_mass * r * r;
    let want = cyl_ixx + hemi_own + sph_mass * hemi_offset * hemi_offset;
    assert_eq!(
        c.inertia_tensor.col0.x, want,
        "Ixx of the capsule, composed from exact parts"
    );
}
