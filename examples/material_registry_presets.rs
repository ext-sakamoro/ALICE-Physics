//! Material registry presets: concrete / ice / metal / rubber / wood, pairwise
//! combine rules, the per-pair override, and the two `PhysicsMaterial`
//! builder methods.
//!
//! `register_concrete` / `register_ice` / `register_metal` / `register_rubber`
//! / `register_wood` are preset constructors with hardcoded friction and
//! restitution coefficients, and (for ice / rubber) a non-default
//! `CombineRule`. The module doc (`src/material.rs:1-11`) does not cite an
//! external handbook, so the "documented" values exercised below are the
//! constants fixed in the `register_*` bodies themselves
//! (`src/material.rs:271-312`), hand-copied here rather than read back from a
//! call to the function under test. `with_combine_rules` and
//! `with_static_friction` are `PhysicsMaterial` builder methods;
//! `set_pair_override` lets one specific material pair bypass the
//! combine-rule arithmetic entirely.
//!
//! `scripts/wiring_guard.py` reported all eight as unwired: `src/solver.rs`
//! calls `register_metal` / `register_rubber` / `set_pair_override`, but only
//! from its own `#[cfg(test)] mod tests` (starting `src/solver.rs:4100`),
//! which the guard does not count as production. This example is that
//! production caller; `tests/analytic_material_wiring.rs` holds the
//! closed-form oracles and degenerate-input checks for the same eight items.
//!
//! ```bash
//! cargo run --example material_registry_presets --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example material_registry_presets --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::material::{CombineRule, MaterialTable, PhysicsMaterial, DEFAULT_MATERIAL};
    use alice_physics::math::Fix128;

    let r = Fix128::from_ratio;
    // Fix128::Average::apply is exact (sum + bit-shift), so the only slop is
    // the independent rounding of the hand-written ratio literal itself —
    // same tolerance the existing src/material.rs unit tests use.
    let close = |a: Fix128, b: Fix128| (a - b).abs() < Fix128::from_raw(0, 1 << 8);

    let mut table = MaterialTable::new();
    let metal = table.register_metal();
    let wood = table.register_wood();
    let rubber = table.register_rubber();
    let ice = table.register_ice();
    let concrete = table.register_concrete();

    println!("[material] presets (id, friction=static=dynamic at registration, restitution, friction_combine, restitution_combine):");
    for (name, id) in [
        ("metal", metal),
        ("wood", wood),
        ("rubber", rubber),
        ("ice", ice),
        ("concrete", concrete),
    ] {
        let m = table.get(id);
        println!(
            "[material]   {name:8} id={id} friction={:.4} restitution={:.4} combine=({:?},{:?})",
            m.dynamic_friction.to_f64(),
            m.restitution.to_f64(),
            m.friction_combine,
            m.restitution_combine
        );
    }

    // Documented constants (src/material.rs:271-312) — hand-copied, not
    // derived by calling register_* and reading the result back.
    let expect = [
        (
            "metal",
            metal,
            r(4, 10),
            r(1, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
        (
            "wood",
            wood,
            r(5, 10),
            r(3, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
        (
            "rubber",
            rubber,
            r(8, 10),
            r(8, 10),
            CombineRule::Max,
            CombineRule::Max,
        ),
        (
            "ice",
            ice,
            r(5, 100),
            r(1, 10),
            CombineRule::Min,
            CombineRule::Min,
        ),
        (
            "concrete",
            concrete,
            r(6, 10),
            r(2, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
    ];
    for (name, id, friction, restitution, fc, rc) in expect {
        let m = table.get(id);
        assert_eq!(m.dynamic_friction, friction, "{name} dynamic_friction");
        assert_eq!(
            m.static_friction, friction,
            "{name} static==dynamic at registration"
        );
        assert_eq!(m.restitution, restitution, "{name} restitution");
        assert_eq!(m.friction_combine, fc, "{name} friction_combine");
        assert_eq!(m.restitution_combine, rc, "{name} restitution_combine");
    }

    // Pairwise combine, closed form by hand. CombineRule priority (highest
    // wins, ties keep the first operand's rule which is identical anyway
    // when tied): Max(3) > Multiply(2) > Average(1) > Min(0).
    let wc = table.combine(wood, concrete); // Average/Average vs Average/Average -> Average
    assert!(
        close(wc.friction, r(55, 100)),
        "wood x concrete friction {wc:?}"
    );
    assert!(
        close(wc.restitution, r(25, 100)),
        "wood x concrete restitution {wc:?}"
    );
    println!(
        "[material] wood x concrete -> friction={:.4} restitution={:.4} (closed form: avg(0.5,0.6)=0.55, avg(0.3,0.2)=0.25)",
        wc.friction.to_f64(),
        wc.restitution.to_f64()
    );

    let ir = table.combine(ice, rubber); // ice's Min(0) loses to rubber's Max(3)
    assert_eq!(
        ir.friction,
        r(8, 10),
        "ice x rubber friction (Max wins) {ir:?}"
    );
    assert_eq!(
        ir.restitution,
        r(8, 10),
        "ice x rubber restitution (Max wins) {ir:?}"
    );
    println!(
        "[material] ice x rubber    -> friction={:.4} restitution={:.4} (closed form: max(0.05,0.8)=0.8, max(0.1,0.8)=0.8)",
        ir.friction.to_f64(),
        ir.restitution.to_f64()
    );

    // set_pair_override takes precedence over the combine-rule arithmetic
    // above (metal x rubber would otherwise be Max -> 0.8/0.8). The call is
    // made with the pair reversed (rubber, metal) to exercise the sort
    // normalization inside set_pair_override itself, then queried in the
    // other order (metal, rubber).
    table.set_pair_override(rubber, metal, r(15, 100), r(5, 100));
    let mr = table.combine(metal, rubber);
    assert_eq!(
        mr.friction,
        r(15, 100),
        "metal x rubber override friction {mr:?}"
    );
    assert_eq!(
        mr.restitution,
        r(5, 100),
        "metal x rubber override restitution {mr:?}"
    );
    println!(
        "[material] metal x rubber  -> friction={:.4} restitution={:.4} (pair override 0.15/0.05, bypasses the Max combine rule)",
        mr.friction.to_f64(),
        mr.restitution.to_f64()
    );

    // with_static_friction only changes static_friction; combine() reads
    // dynamic_friction, so pair results must be bit-identical with or
    // without the builder call.
    let base = PhysicsMaterial::new(0, r(3, 10), r(2, 10));
    let sticky = base.with_static_friction(r(9, 10));
    assert_eq!(
        sticky.static_friction,
        r(9, 10),
        "with_static_friction sets static_friction"
    );
    assert_eq!(
        sticky.dynamic_friction,
        r(3, 10),
        "with_static_friction leaves dynamic_friction untouched"
    );
    let plain_id = table.register(base);
    let sticky_id = table.register(sticky);
    let plain_pair = table.combine(plain_id, DEFAULT_MATERIAL);
    let sticky_pair = table.combine(sticky_id, DEFAULT_MATERIAL);
    assert_eq!(
        plain_pair, sticky_pair,
        "static_friction is invisible to combine()"
    );
    println!(
        "[material] with_static_friction: static 0.3 -> 0.9, dynamic unchanged, pair result identical ({:.4}/{:.4})",
        plain_pair.friction.to_f64(),
        plain_pair.restitution.to_f64()
    );

    // with_combine_rules on a from-scratch material: its Min friction rule
    // (priority 0) loses to wood's Average (priority 1); its Max restitution
    // rule (priority 3) beats wood's Average (priority 1).
    let custom = PhysicsMaterial::new(0, r(9, 10), r(9, 10))
        .with_combine_rules(CombineRule::Min, CombineRule::Max);
    assert_eq!(
        custom.friction_combine,
        CombineRule::Min,
        "with_combine_rules sets friction_combine"
    );
    assert_eq!(
        custom.restitution_combine,
        CombineRule::Max,
        "with_combine_rules sets restitution_combine"
    );
    let custom_id = table.register(custom);
    let cw = table.combine(custom_id, wood);
    assert!(
        close(cw.friction, r(70, 100)),
        "custom x wood friction (Average wins) {cw:?}"
    );
    assert_eq!(
        cw.restitution,
        r(9, 10),
        "custom x wood restitution (Max wins) {cw:?}"
    );
    println!(
        "[material] custom(Min,Max) x wood -> friction={:.4} restitution={:.4} (closed form: avg(0.9,0.5)=0.7, max(0.9,0.3)=0.9)",
        cw.friction.to_f64(),
        cw.restitution.to_f64()
    );

    println!("[material] all closed-form checks passed");
}
