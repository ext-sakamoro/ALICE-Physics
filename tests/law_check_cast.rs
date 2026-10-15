//! `LawCheck` stays a fieldless enum, so a caller outside the crate can turn a
//! variant into an integer with `as` (3.0.1 allowed it; a variant with fields
//! would make every such cast a compile error, whatever `#[non_exhaustive]`
//! says). This file is compiled as another crate, so it is that caller.
//!
//! oracle: the discriminants of the 3.0.1 variants (`Verified` 0,
//! `SemanticsVerified` 1, `Unpinned` 2) do not move, and the variants added
//! later follow them in declaration order.

use alice_physics::LawCheck;

#[test]
fn every_variant_casts_to_its_discriminant() {
    let cast = [
        LawCheck::Verified as isize,
        LawCheck::SemanticsVerified as isize,
        LawCheck::Unpinned as isize,
        LawCheck::VerifiedUnderPrevious as isize,
        LawCheck::SemanticsPrevious as isize,
    ];
    assert_eq!(cast, [0, 1, 2, 3, 4]);
}
