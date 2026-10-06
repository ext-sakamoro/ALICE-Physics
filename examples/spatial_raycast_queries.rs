//! Raycast wiring: the remaining unwired queries in `src/raycast.rs`.
//!
//! `ray_sphere` / `ray_plane` in this file already had production callers
//! (`PhysicsWorld::raycast` and others). This example is the production
//! entry point for the other 9 pub items, which had none:
//!
//! - `ray_aabb` (slab method) / `ray_capsule` (infinite cylinder clipped by
//!   two hemisphere caps)
//! - `raycast_aabbs` / `raycast_spheres` (nearest-hit over a candidate list)
//! - `raycast_all_aabbs` / `raycast_all_spheres` (every hit, sorted by t)
//! - `raycast_any_aabbs` / `raycast_any_spheres` (early-out boolean test)
//! - `sweep_sphere` (moving-sphere shape cast against a static target)
//!
//! Every expected value below is derived by hand from the primitive types
//! (`Ray`, `Vec3Fix`, `Fix128`, `AABB`, `Sphere`, `Capsule`) and basic
//! geometry (slab intersection, quadratic ray/sphere roots, Pythagorean
//! combined-radius TOI) -- never by calling the function under test for the
//! expected side. This is distinct from `examples/spatial_queries.rs`, which
//! wires the *different* `src/query.rs` module (`overlap_*` / `sphere_cast` /
//! `capsule_cast` / `batch_*`); there is no name or purpose collision between
//! the two files.
//!
//! ```bash
//! cargo run --example spatial_raycast_queries --features std
//! ```

use alice_physics::collider::{Capsule, Sphere, AABB};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::{
    ray_aabb, ray_capsule, raycast_aabbs, raycast_all_aabbs, raycast_all_spheres,
    raycast_any_aabbs, raycast_any_spheres, raycast_spheres, sweep_sphere, Ray, RayHit,
};

fn main() {
    ray_aabb_section();
    ray_capsule_section();
    raycast_aabbs_section();
    raycast_spheres_section();
    sweep_sphere_section();
    extreme_magnitude_section();

    println!("[raycast] all 9 previously-unwired raycast.rs items exercised");
}

// ---------------------------------------------------------------------------
// ray_aabb
// ---------------------------------------------------------------------------

fn ray_aabb_section() {
    // AABB centered at the origin, half-extents (2,3,4): min=(-2,-3,-4),
    // max=(2,3,4). Ray from (-10,1,2) along +X (already unit).
    //   x slab: t0 = (-2 - -10)/1 = 8, t1 = (2 - -10)/1 = 12 -> (8,12)
    //   y: dir.y=0, origin.y=1 in [-3,3] -> pass-through (no update)
    //   z: dir.z=0, origin.z=2 in [-4,4] -> pass-through (no update)
    // t_min=8 >= 0 so t=8. point = (-10+8,1,2) = (-2,1,2).
    // normal: local=(-2,1,2), half=(2,3,4): dx=|(-2).abs()-2|=0,
    // dy=|1-3|=2, dz=|2-4|=2 -> dx<dy && dx<dz -> -X face (local.x<0).
    let aabb = AABB::new(Vec3Fix::from_int(-2, -3, -4), Vec3Fix::from_int(2, 3, 4));
    let ray = Ray::new(Vec3Fix::from_int(-10, 1, 2), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, Fix128::from_int(100)).expect("ray_aabb should hit");
    assert_eq!(hit.t, Fix128::from_int(8));
    assert_eq!(hit.point, Vec3Fix::from_int(-2, 1, 2));
    assert_eq!(hit.normal, Vec3Fix::from_int(-1, 0, 0));
    println!(
        "[raycast] ray_aabb: t={} point={:?}",
        hit.t.to_f64(),
        v3f(hit.point)
    );

    // Degenerate: ray origin INSIDE the AABB -> must report the exit point,
    // not t=0. Same AABB, origin=(0,1,2) (inside): x slab t0=-2, t1=2 ->
    // (t_min,t_max)=(-2,2); t_min<0 so t=t_max=2. point=(2,1,2).
    let inside_ray = Ray::new(Vec3Fix::from_int(0, 1, 2), Vec3Fix::UNIT_X);
    let inside_hit =
        ray_aabb(&inside_ray, &aabb, Fix128::from_int(100)).expect("origin inside must still hit");
    assert_eq!(
        inside_hit.t,
        Fix128::from_int(2),
        "must report the exit point, not t=0"
    );
    assert_eq!(inside_hit.point, Vec3Fix::from_int(2, 1, 2));
    assert_eq!(inside_hit.normal, Vec3Fix::from_int(1, 0, 0));
    println!(
        "[raycast] ray_aabb (origin inside): exit t={}",
        inside_hit.t.to_f64()
    );

    // Degenerate: ray parallel to a slab face (dir.z=0) with origin OUTSIDE
    // that slab's range (z=5 vs [-2,2]) -> must miss regardless of x/y.
    let aabb2 = AABB::new(Vec3Fix::from_int(-2, -2, -2), Vec3Fix::from_int(2, 2, 2));
    let parallel_miss = Ray::new(Vec3Fix::from_int(-10, 0, 5), Vec3Fix::UNIT_X);
    assert!(
        ray_aabb(&parallel_miss, &aabb2, Fix128::from_int(100)).is_none(),
        "ray parallel to z-slab with origin outside z-range must miss"
    );
    println!("[raycast] ray_aabb (parallel to slab, out of range): miss as expected");

    // max_t = 0 boundary: ray origin exactly on the surface, pointing
    // inward -> t=0 is a valid hit (closed interval [0, max_t]).
    let aabb3 = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
    let touch_ray = Ray::new(Vec3Fix::from_int(-1, 0, 0), Vec3Fix::UNIT_X);
    let touch_hit =
        ray_aabb(&touch_ray, &aabb3, Fix128::ZERO).expect("t=0 is inside the closed interval");
    assert_eq!(touch_hit.t, Fix128::ZERO);
    assert_eq!(touch_hit.point, Vec3Fix::from_int(-1, 0, 0));
    // max_t = 0 with no touch: origin strictly outside -> None.
    let far_ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
    assert!(ray_aabb(&far_ray, &aabb3, Fix128::ZERO).is_none());
    println!("[raycast] ray_aabb (max_t=0): touch=Some(t=0), no-touch=None");

    // Degenerate: non-normalized direction + the X-slab min/max swap path
    // (negative direction forces t0 > t1 before the swap). Ray::new
    // normalizes (-2,0,0) to (-1,0,0) internally.
    //   x slab (dir=-1): inv_d=-1, t0=(-1-5)*-1=6, t1=(1-5)*-1=4 ->
    //   swap -> (t_min,t_max)=(4,6). t=4. point=(5-4,0,0)=(1,0,0).
    let swap_ray = Ray::new(Vec3Fix::from_int(5, 0, 0), Vec3Fix::from_int(-2, 0, 0));
    let swap_hit = ray_aabb(&swap_ray, &aabb3, Fix128::from_int(100))
        .expect("non-normalized negative direction must still hit via the min/max swap");
    assert_eq!(swap_hit.t, Fix128::from_int(4));
    assert_eq!(swap_hit.point, Vec3Fix::from_int(1, 0, 0));
    assert_eq!(swap_hit.normal, Vec3Fix::from_int(1, 0, 0));
    println!(
        "[raycast] ray_aabb (non-normalized dir, slab swap): t={}",
        swap_hit.t.to_f64()
    );

    // Degenerate: corner tie-break. origin=(-5,1,0) on the y=max boundary
    // exactly, dir=+X. x slab -> t_min=4. y: dir.y=0, origin.y=1==max.y=1
    // passes (boundary inclusive). z: origin.z=0 in range, passes.
    // point=(-1,1,0): local=(-1,1,0), half=(1,1,1): dx=0, dy=0, dz=1.
    // dx<dy is false (tie) -> falls to dy<dz (0<1, true) -> +Y face.
    let corner_ray = Ray::new(Vec3Fix::from_int(-5, 1, 0), Vec3Fix::UNIT_X);
    let corner_hit =
        ray_aabb(&corner_ray, &aabb3, Fix128::from_int(100)).expect("corner ray should hit");
    assert_eq!(corner_hit.t, Fix128::from_int(4));
    assert_eq!(
        corner_hit.normal,
        Vec3Fix::from_int(0, 1, 0),
        "dx==dy tie resolves to +Y face"
    );
    println!(
        "[raycast] ray_aabb (corner tie dx==dy): normal={:?}",
        v3f(corner_hit.normal)
    );

    // Degenerate: Y-slab and Z-slab min/max swap (mirrors the X-slab swap
    // above, along the other two axes). AABB [-1,1]^3:
    //   -Y: origin (0,5,0), dir (0,-1,0): t0=(-1-5)*-1=6, t1=(1-5)*-1=4
    //       -> swap -> t=4, point=(0,1,0).
    //   -Z: origin (0,0,5), dir (0,0,-1): same algebra -> t=4, point=(0,0,1).
    let y_swap_ray = Ray::new(Vec3Fix::from_int(0, 5, 0), Vec3Fix::from_int(0, -1, 0));
    let y_swap_hit =
        ray_aabb(&y_swap_ray, &aabb3, Fix128::from_int(100)).expect("Y-slab swap must hit");
    assert_eq!(y_swap_hit.t, Fix128::from_int(4));
    assert_eq!(y_swap_hit.point, Vec3Fix::from_int(0, 1, 0));

    let z_swap_ray = Ray::new(Vec3Fix::from_int(0, 0, 5), Vec3Fix::from_int(0, 0, -1));
    let z_swap_hit =
        ray_aabb(&z_swap_ray, &aabb3, Fix128::from_int(100)).expect("Z-slab swap must hit");
    assert_eq!(z_swap_hit.t, Fix128::from_int(4));
    assert_eq!(z_swap_hit.point, Vec3Fix::from_int(0, 0, 1));
    println!("[raycast] ray_aabb (Y/Z-slab swap): both t=4, confirming all 3 axes");
}

// ---------------------------------------------------------------------------
// ray_capsule
// ---------------------------------------------------------------------------

fn ray_capsule_section() {
    // Vertical capsule (0,-2,0)-(0,2,0) radius 1, ray along +X at y=0,z=0:
    // standard infinite-cylinder intersection, distance from axis line to
    // the ray's line is 0, so entry is at x=-radius=-1: t = -1-(-5) = 4.
    let capsule = Capsule::new(
        Vec3Fix::from_int(0, -2, 0),
        Vec3Fix::from_int(0, 2, 0),
        Fix128::ONE,
    );
    let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, Fix128::from_int(100)).expect("should hit cylinder side");
    assert_eq!(hit.t, Fix128::from_int(4));
    assert_eq!(hit.point, Vec3Fix::from_int(-1, 0, 0));
    println!(
        "[raycast] ray_capsule (cylinder side): t={}",
        hit.t.to_f64()
    );

    // Degenerate: zero-length capsule (a==b) collapses to a sphere of the
    // same radius. Sphere center (0,0,0) r=2, ray from (-10,0,0): entry at
    // x=-2 -> t = -2-(-10) = 8.
    let point_capsule = Capsule::new(
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(0, 0, 0),
        Fix128::from_int(2),
    );
    let point_ray = Ray::new(Vec3Fix::from_int(-10, 0, 0), Vec3Fix::UNIT_X);
    let point_hit =
        ray_capsule(&point_ray, &point_capsule, Fix128::from_int(100)).expect("degenerate==sphere");
    assert_eq!(point_hit.t, Fix128::from_int(8));
    println!(
        "[raycast] ray_capsule (zero-length == sphere): t={}",
        point_hit.t.to_f64()
    );

    // Miss everything: ray offset far enough off-axis (y=5, radius 1) to
    // clear both the cylinder (perpendicular distance 5 > 1) and both
    // hemisphere caps (distance to each cap center >= sqrt(5^2+3^2) > 1).
    let miss_ray = Ray::new(Vec3Fix::from_int(-5, 5, 0), Vec3Fix::UNIT_X);
    assert!(ray_capsule(&miss_ray, &capsule, Fix128::from_int(100)).is_none());
    println!("[raycast] ray_capsule: total miss as expected");

    // Degenerate: proj < 0 (below the cylinder's finite segment) forces the
    // fallthrough to the hemisphere-cap test. Ray at y=-3 (the bottom cap's
    // south pole height) is tangent to the bottom cap sphere (disc=0) and
    // the cylinder quadratic's proj comes out to -0.25 (out of [0,1]).
    //   oc=(-10,-1,0), b=-10, c=101-1=100, disc=100-100=0, sqrt=0,
    //   t=10-0=10, point=(0,-3,0) -- exactly the south pole.
    let south_ray = Ray::new(Vec3Fix::from_int(-10, -3, 0), Vec3Fix::UNIT_X);
    let south_hit =
        ray_capsule(&south_ray, &capsule, Fix128::from_int(100)).expect("south-pole cap fallback");
    assert_eq!(south_hit.t, Fix128::from_int(10));
    assert_eq!(south_hit.point, Vec3Fix::from_int(0, -3, 0));
    assert_eq!(south_hit.normal, Vec3Fix::from_int(0, -1, 0));
    println!(
        "[raycast] ray_capsule (proj<0, bottom cap fallback): t={}",
        south_hit.t.to_f64()
    );

    // Mirror: proj > 1, top cap's north pole.
    let north_ray = Ray::new(Vec3Fix::from_int(-10, 3, 0), Vec3Fix::UNIT_X);
    let north_hit =
        ray_capsule(&north_ray, &capsule, Fix128::from_int(100)).expect("north-pole cap fallback");
    assert_eq!(north_hit.t, Fix128::from_int(10));
    assert_eq!(north_hit.point, Vec3Fix::from_int(0, 3, 0));
    assert_eq!(north_hit.normal, Vec3Fix::from_int(0, 1, 0));
    println!(
        "[raycast] ray_capsule (proj>1, top cap fallback): t={}",
        north_hit.t.to_f64()
    );
}

// ---------------------------------------------------------------------------
// raycast_aabbs / raycast_all_aabbs / raycast_any_aabbs
// ---------------------------------------------------------------------------

// Deliberately NOT in ascending-t order (18, 4, miss, 12, miss): this is
// the scene a missing `sort_by_key` mutation in `raycast_all_aabbs` would
// fail to catch if the list were already sorted by construction.
fn aabb_scene() -> Vec<(AABB, usize)> {
    vec![
        (
            AABB::new(Vec3Fix::from_int(8, -1, -1), Vec3Fix::from_int(9, 1, 1)),
            2,
        ), // t=18
        (
            AABB::new(Vec3Fix::from_int(-6, -1, -1), Vec3Fix::from_int(-4, 1, 1)),
            0,
        ), // t=4
        (
            AABB::new(Vec3Fix::from_int(0, 5, -1), Vec3Fix::from_int(1, 6, 1)),
            3,
        ), // off-axis miss
        (
            AABB::new(Vec3Fix::from_int(2, -1, -1), Vec3Fix::from_int(4, 1, 1)),
            1,
        ), // t=12
        (
            AABB::new(Vec3Fix::from_int(-20, -1, -1), Vec3Fix::from_int(-15, 1, 1)),
            4,
        ), // behind, miss
    ]
}

fn raycast_aabbs_section() {
    let ray = Ray::new(Vec3Fix::from_int(-10, 0, 0), Vec3Fix::UNIT_X);
    let max_t = Fix128::from_int(100);
    let aabbs = aabb_scene();

    let nearest = raycast_aabbs(&ray, &aabbs, max_t).expect("nearest AABB hit");
    assert_eq!(
        nearest,
        RayHit {
            t: Fix128::from_int(4),
            point: Vec3Fix::from_int(-6, 0, 0),
            normal: Vec3Fix::from_int(-1, 0, 0),
            body_index: 0,
        }
    );
    // Order invariance.
    let mut reversed = aabbs.clone();
    reversed.reverse();
    assert_eq!(raycast_aabbs(&ray, &reversed, max_t), Some(nearest));
    println!(
        "[raycast] raycast_aabbs: nearest body={} t={}",
        nearest.body_index,
        nearest.t.to_f64()
    );

    let all = raycast_all_aabbs(&ray, &aabbs, max_t);
    assert_eq!(
        all.iter().map(|h| h.body_index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(all[0].t, Fix128::from_int(4));
    assert_eq!(all[1].t, Fix128::from_int(12));
    assert_eq!(all[2].t, Fix128::from_int(18));
    println!(
        "[raycast] raycast_all_aabbs: {} hits sorted by t = {:?}",
        all.len(),
        all.iter().map(|h| h.t.to_f64()).collect::<Vec<_>>()
    );

    assert!(raycast_any_aabbs(&ray, &aabbs, max_t));
    let miss_only: Vec<(AABB, usize)> = aabbs
        .iter()
        .copied()
        .filter(|(_, idx)| *idx == 3 || *idx == 4)
        .collect();
    assert!(!raycast_any_aabbs(&ray, &miss_only, max_t));
    println!("[raycast] raycast_any_aabbs: true (full scene), false (miss-only subset)");

    // Degenerate: empty candidate list.
    assert!(raycast_aabbs(&ray, &[], max_t).is_none());
    assert!(raycast_all_aabbs(&ray, &[], max_t).is_empty());
    assert!(!raycast_any_aabbs(&ray, &[], max_t));
    println!("[raycast] raycast_*_aabbs: empty list -> None/empty/false");

    // max_t narrowing: 3 < 4 -> None; 5 -> only the nearest (t=4) qualifies.
    assert!(raycast_aabbs(&ray, &aabbs, Fix128::from_int(3)).is_none());
    assert_eq!(
        raycast_aabbs(&ray, &aabbs, Fix128::from_int(5)).map(|h| h.body_index),
        Some(0)
    );
    println!("[raycast] raycast_aabbs: max_t narrowing confirmed");
}

// ---------------------------------------------------------------------------
// raycast_spheres / raycast_all_spheres / raycast_any_spheres
// ---------------------------------------------------------------------------

// Deliberately NOT in ascending-t order (13, miss, 5, miss): same rationale
// as `aabb_scene` above, for `raycast_all_spheres`'s sort.
fn sphere_scene() -> Vec<(Sphere, usize)> {
    vec![
        (
            Sphere::new(Vec3Fix::from_int(5, 0, 0), Fix128::from_int(2)),
            1,
        ), // t=13
        (Sphere::new(Vec3Fix::from_int(0, 4, 0), Fix128::ONE), 2), // off-axis miss
        (Sphere::new(Vec3Fix::from_int(-4, 0, 0), Fix128::ONE), 0), // t=5
        (Sphere::new(Vec3Fix::from_int(-20, 0, 0), Fix128::ONE), 3), // behind, miss
    ]
}

fn raycast_spheres_section() {
    let ray = Ray::new(Vec3Fix::from_int(-10, 0, 0), Vec3Fix::UNIT_X);
    let max_t = Fix128::from_int(100);
    let spheres = sphere_scene();

    let nearest = raycast_spheres(&ray, &spheres, max_t).expect("nearest sphere hit");
    assert_eq!(
        nearest,
        RayHit {
            t: Fix128::from_int(5),
            point: Vec3Fix::from_int(-5, 0, 0),
            normal: Vec3Fix::from_int(-1, 0, 0),
            body_index: 0,
        }
    );
    let mut reversed = spheres.clone();
    reversed.reverse();
    assert_eq!(raycast_spheres(&ray, &reversed, max_t), Some(nearest));
    println!(
        "[raycast] raycast_spheres: nearest body={} t={}",
        nearest.body_index,
        nearest.t.to_f64()
    );

    let all = raycast_all_spheres(&ray, &spheres, max_t);
    assert_eq!(
        all.iter().map(|h| h.body_index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(all[0].t, Fix128::from_int(5));
    assert_eq!(all[1].t, Fix128::from_int(13));
    println!(
        "[raycast] raycast_all_spheres: {} hits sorted by t = {:?}",
        all.len(),
        all.iter().map(|h| h.t.to_f64()).collect::<Vec<_>>()
    );

    assert!(raycast_any_spheres(&ray, &spheres, max_t));
    let miss_only: Vec<(Sphere, usize)> = spheres
        .iter()
        .copied()
        .filter(|(_, idx)| *idx == 2 || *idx == 3)
        .collect();
    assert!(!raycast_any_spheres(&ray, &miss_only, max_t));
    println!("[raycast] raycast_any_spheres: true (full scene), false (miss-only subset)");

    // Degenerate: empty candidate list.
    assert!(raycast_spheres(&ray, &[], max_t).is_none());
    assert!(raycast_all_spheres(&ray, &[], max_t).is_empty());
    assert!(!raycast_any_spheres(&ray, &[], max_t));
    println!("[raycast] raycast_*_spheres: empty list -> None/empty/false");

    // max_t narrowing: 4 < 5 -> None; 5 -> boundary-inclusive hit.
    assert!(raycast_spheres(&ray, &spheres, Fix128::from_int(4)).is_none());
    assert_eq!(
        raycast_spheres(&ray, &spheres, Fix128::from_int(5)).map(|h| h.body_index),
        Some(0)
    );
    println!("[raycast] raycast_spheres: max_t narrowing confirmed");

    // Degenerate: ray origin exactly at a sphere's center -> must use the
    // far (exit) root, not t=0. oc=(0,0,0), disc=r^2, t_far=r=3.
    let center_sphere = vec![(
        Sphere::new(Vec3Fix::from_int(-10, 0, 0), Fix128::from_int(3)),
        0usize,
    )];
    let center_ray = Ray::new(Vec3Fix::from_int(-10, 0, 0), Vec3Fix::UNIT_X);
    let center_hit = raycast_spheres(&center_ray, &center_sphere, max_t).expect("origin==center");
    assert_eq!(center_hit.t, Fix128::from_int(3));
    assert_eq!(center_hit.point, Vec3Fix::from_int(-7, 0, 0));
    println!(
        "[raycast] raycast_spheres (origin==center): far-exit t={}",
        center_hit.t.to_f64()
    );

    // Degenerate: non-normalized direction along +Y, magnitude 5.
    // Ray::new normalizes to (0,1,0). Sphere (0,10,0) r=2, origin (0,0,0):
    // oc=(0,-10,0), b=-10, c=100-4=96, disc=4, sqrt=2, t=10-2=8.
    let y_sphere = vec![(
        Sphere::new(Vec3Fix::from_int(0, 10, 0), Fix128::from_int(2)),
        0usize,
    )];
    let y_ray = Ray::new(Vec3Fix::ZERO, Vec3Fix::from_int(0, 5, 0));
    let y_hit = raycast_spheres(&y_ray, &y_sphere, max_t).expect("non-normalized +Y direction");
    assert_eq!(y_hit.t, Fix128::from_int(8));
    assert_eq!(y_hit.point, Vec3Fix::from_int(0, 8, 0));
    println!(
        "[raycast] raycast_spheres (non-normalized direction): t={}",
        y_hit.t.to_f64()
    );
}

// ---------------------------------------------------------------------------
// sweep_sphere
// ---------------------------------------------------------------------------

fn sweep_sphere_section() {
    // Moving sphere center (-10,0,0) r=1, direction +X, static target
    // center (0,0,0) r=2. Equivalent to a ray vs the expanded sphere
    // (center (0,0,0), radius 1+2=3): distance 10, so surfaces touch at
    // center-separation == 3, i.e. t = 10-3 = 7.
    let moving = Sphere::new(Vec3Fix::from_int(-10, 0, 0), Fix128::ONE);
    let target = Sphere::new(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(2));
    let hit = sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, Fix128::from_int(100))
        .expect("sweep_sphere should hit");
    assert_eq!(hit.t, Fix128::from_int(7));
    assert_eq!(hit.point, Vec3Fix::from_int(-3, 0, 0));
    println!(
        "[raycast] sweep_sphere: t={} point={:?}",
        hit.t.to_f64(),
        v3f(hit.point)
    );

    // max_t = 0: the real contact is at t=7 > 0, so no hit within budget.
    assert!(sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, Fix128::ZERO).is_none());
    println!("[raycast] sweep_sphere (max_t=0): None as expected (contact is at t=7)");

    // Degenerate: spheres already coincident at the start (overlapping).
    // Expanded sphere center (0,0,0) r=2 (1+1), ray from (0,0,0): oc=(0,0,0).
    // A sweep that starts overlapping is an initial overlap: t=0 at the start
    // centre, with the fallback normal -direction=(-1,0,0) for coincident
    // centres. The exit root t=2 is never reported as a contact.
    let coincident_a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let coincident_b = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let coincident_hit = sweep_sphere(
        &coincident_a,
        Vec3Fix::UNIT_X,
        &coincident_b,
        Fix128::from_int(100),
    )
    .expect("already-overlapping spheres report the initial overlap");
    assert_eq!(
        coincident_hit.t,
        Fix128::ZERO,
        "initial overlap, not the exit t=2"
    );
    assert_eq!(coincident_hit.point, Vec3Fix::ZERO);
    assert_eq!(coincident_hit.normal, Vec3Fix::from_int(-1, 0, 0));
    println!(
        "[raycast] sweep_sphere (already overlapping): initial overlap t={} normal={:?}",
        coincident_hit.t.to_f64(),
        v3f(coincident_hit.normal)
    );

    // Degenerate: zero direction. `Ray::new` documents falling back to
    // UNIT_X when the given direction has zero length, so this must equal
    // the original +X scenario exactly.
    let zero_dir_hit = sweep_sphere(&moving, Vec3Fix::ZERO, &target, Fix128::from_int(100))
        .expect("zero direction falls back to Ray::new's UNIT_X default");
    assert_eq!(
        zero_dir_hit, hit,
        "zero direction must equal the explicit UNIT_X case"
    );
    println!(
        "[raycast] sweep_sphere (zero direction): falls back to UNIT_X, t={}",
        zero_dir_hit.t.to_f64()
    );
}

// ---------------------------------------------------------------------------
// Extreme Fix128 magnitudes: must wrap (mod 2^128 group), never panic.
// ---------------------------------------------------------------------------

fn extreme_magnitude_section() {
    let extreme = Vec3Fix::new(
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX,
        },
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let extreme_aabb = AABB::new(extreme, extreme);
    let extreme_sphere = Sphere::new(extreme, Fix128::ONE);
    let extreme_capsule = Capsule::new(extreme, Vec3Fix::ZERO, Fix128::ONE);
    let ray = Ray::new(Vec3Fix::ZERO, Vec3Fix::UNIT_X);
    let max_t = Fix128::from_int(100);

    let outcome = std::panic::catch_unwind(|| {
        let _ = ray_aabb(&ray, &extreme_aabb, max_t);
        let _ = ray_capsule(&ray, &extreme_capsule, max_t);
        let _ = raycast_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_all_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_any_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = raycast_all_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = raycast_any_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = sweep_sphere(&extreme_sphere, Vec3Fix::UNIT_X, &extreme_sphere, max_t);
    });
    assert!(
        outcome.is_ok(),
        "extreme Fix128 magnitudes must wrap, not panic"
    );
    println!("[raycast] extreme Fix128 magnitudes: all 9 items wrap without panicking");
}

fn v3f(v: Vec3Fix) -> (f64, f64, f64) {
    (v.x.to_f64(), v.y.to_f64(), v.z.to_f64())
}
