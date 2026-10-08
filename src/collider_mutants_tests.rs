//! Oracle tests for specific missed mutants in `collider.rs`, kept in a
//! separate file (loaded via `#[path]` from `collider.rs`) to keep this
//! crate's diff in `collider.rs` itself to the one `mod` declaration line —
//! `collider.rs` is concurrently being changed for GJK unification work, and
//! these tests do not touch anything in or near that area (`add_face` is
//! EPA's own polytope-face constructor, not GJK's simplex-evolution code).

use super::*;

fn p(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_int(x),
        Fix128::from_int(y),
        Fix128::from_int(z),
    )
}

fn interior_of(quad: [Vec3Fix; 4]) -> Vec3Fix {
    let sum = quad[0] + quad[1] + quad[2] + quad[3];
    sum * Fix128::from_ratio(1, 4)
}

/// Kills `replace < with <=` / `replace < with ==` at `collider.rs`'s
/// `add_face` orientation check (`(a - interior).dot(normal) < Fix128::ZERO`).
///
/// The comparison is normally never exactly zero for a non-degenerate
/// tetrahedron: `interior` is the centroid of the whole tetrahedron, which
/// is strictly inside it, so it cannot sit exactly on any one face's plane,
/// and `<` vs `<=`/`==` only differ at that exact zero. This builds the one
/// case where it genuinely is zero: a *flat* (zero-volume, all 4 points
/// coplanar) initial tetrahedron, whose centroid then lies exactly in that
/// shared plane for every face built from it. `add_face` does not itself
/// reject a flat initial simplex (only a degenerate *triangle*, caught
/// separately by the `cross.length_squared().is_zero()` check above it), so
/// this is a real input, not a contrived one — e.g. `epa`'s own
/// `initial_simplex.len() < 4` guard does not check volume either.
///
/// With all four points at z = 0, the centroid is at z = 0 too, so `a -
/// interior` has no z component while the raw `cross(ab, ac)` here is purely
/// in z — their dot product is exactly `0`, not merely close to it, so this
/// does not depend on any rounding.
#[test]
fn add_face_orientation_check_is_exact_at_the_flat_tetrahedron_boundary() {
    let vertices = [p(0, 0, 0), p(4, 0, 0), p(1, 3, 0), p(2, -2, 0)];
    let interior = interior_of(vertices);
    let mut faces: Vec<EpaFace> = Vec::new();

    add_face(&mut faces, &vertices, interior, 0, 1, 2);

    assert_eq!(
        faces.len(),
        1,
        "ab x ac is non-zero here, so the face is kept"
    );
    // The raw cross(ab, ac) for (0,0,0),(4,0,0),(1,3,0) is (0,0,12); at the
    // exact-zero boundary `<` does not flip it, so the stored normal is the
    // unflipped (0,0,1), not its negation.
    assert_eq!(
        faces[0].normal,
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE)
    );
}

/// Kills `delete -` in `add_face` (the `-normal` in its orientation
/// flip branch): a case where the flip is genuinely required, confirming
/// the stored normal is actually negated, not merely passed through.
///
/// Same `a, b, c` as above, but with a `d` lifted off the shared plane
/// (still a face of a real tetrahedron, just no longer the degenerate
/// case), which moves the centroid to `(a - interior).dot(normal) = -24`
/// (strictly negative, not a boundary case): the flip branch must run.
#[test]
fn add_face_flips_the_normal_away_from_the_interior() {
    let vertices = [p(0, 0, 0), p(4, 0, 0), p(1, 3, 0), p(2, -2, 8)];
    let interior = interior_of(vertices);
    let mut faces: Vec<EpaFace> = Vec::new();

    add_face(&mut faces, &vertices, interior, 0, 1, 2);

    assert_eq!(faces.len(), 1);
    // Raw cross(ab, ac) is still (0,0,12) (a, b, c unchanged); with the
    // flip applied the stored normal must be its negation, (0,0,-1).
    assert_eq!(
        faces[0].normal,
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, -Fix128::ONE)
    );
}
