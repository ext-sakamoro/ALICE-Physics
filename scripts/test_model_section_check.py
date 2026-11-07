#!/usr/bin/env python3
"""scripts/model_section_check.py の oracle.

検査器の空振り (比較 0 件で green) が最も危ないので、fixture の crate で
(1) 7 field と実在する試験を持つ module が通る (bullet 形と段落形の両方)
(2) field の欠落 / 空 / 知らない label で落ちる
(3) 名指した試験 file の不在 / fn の不在 / `#[ignore]` / file より前の `::fn` / 全 fn ignore で落ちる
(4) feature gate のある試験 file を ci.yml のどの `cargo test` も走らせない時に落ち、揃えれば通る
(5) 段の field: L1 を挙げて誤差の試験を名指さない時に落ちる、L0 が無い時に落ちる
(5b) 既知の欠陥の field (任意の 8 つ目): 名指した試験が実在し `#[ignore = "known defect: …"]` で docs/oracle-status.md に
     載る時だけ通る、検証の field に既知の欠陥を書くと落ちる (red の oracle は検証ではない)
(6) src の分類: 未分類 / 両方に載る / 一覧にあって不在 で落ちる
(7) ratchet: baseline に無い欠落 / 節が付いたのに残る行 / module でない行 で落ちる、--write-baseline で揃う
(8) 比較 0 件: module 0 / 節 0 / 試験 0 / 引用 0 で落ちる
(9) 方程式 / 保存則 / 適用範囲 / 近似 の各文 (`.` `;` `。` で区切る) が src で解決する code 参照 (backtick の `fn` / 型 / `path.rs:line`) を 1 つ以上引く、無い / rename された / 行が無い / 数式だけ で落ちる 単位の field は対象外
を固定する 期待値は fixture の構造から決め、検査器を呼んで作らない

run: python3 scripts/test_model_section_check.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_section_check as msc  # noqa: E402

CI_DEFAULT = textwrap.dedent("""\
    jobs:
      test:
        steps:
          - run: cargo test
          - run: cargo test --lib --features "simd"
    """)

SECTION_PARAGRAPH = textwrap.dedent("""\
    //! Falling bodies.
    //!
    //! # MODEL — free fall
    //!
    //! **Equations.** `v' = g`, `x' = v`, symplectic Euler per step (`drop`).
    //!
    //! **Units.** SI: m, s, m/s².
    //!
    //! **Conservation.** Energy `½ m v² + m g h` is conserved by the exact solution
    //! ([`drop`]); the integrator drifts by O(h) (`src/fall.rs:1`).
    //!
    //! **Valid range.** Any `h > 0` (`fall::drop`); nothing is clamped (`drop()`).
    //!
    //! **Approximations.** No drag (`drop`).
    //!
    //! **Validation.** `tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),
    //! `::drop_replays_bit_for_bit`.
    //!
    //! **Tiers.** L0 only.
    """)

SECTION_BULLETS = textwrap.dedent("""\
    //! Falling bodies.
    //!
    //! # MODEL — free fall
    //!
    //! - **方程式 (equations)**: `v' = g`, `x' = v` (`drop`).
    //! - **単位 (units)**: SI.
    //! - **保存則 (conservation)**: energy up to O(h)
    //!   (`drop`, `a_body_falls_g_h_squared_over_two`).
    //! - **適用範囲 (range)**: any `h > 0` (`drop`).
    //! - **近似 (approximations)**: no drag (`drop`).
    //! - **検証 (verification)**: `tests/analytic_fall.rs::
    //!   a_body_falls_g_h_squared_over_two` (closed form) and
    //!   `::drop_replays_bit_for_bit`.
    //! - **段 (tiers)**: L0 only.
    """)

TEST_FALL = textwrap.dedent("""\
    #![cfg(feature = "std")]
    use fixture::fall;

    #[test]
    fn a_body_falls_g_h_squared_over_two() {
        assert!(fall::drop(1.0) > 0.0);
    }

    #[test]
    fn drop_replays_bit_for_bit() {
        assert_eq!(fall::drop(1.0), fall::drop(1.0));
    }

    #[test]
    #[ignore = "src gap: no drag yet"]
    fn a_body_with_drag_reaches_terminal_velocity() {}

    #[test]
    #[ignore = "known defect: FALL-001: the floor contact restitution is one bit too high"]
    fn a_dropped_body_comes_to_rest_on_the_floor() {}
    """)

ORACLE_STATUS = textwrap.dedent("""\
    # Oracle Status

    ## Red by design (1)

    - `a_dropped_body_comes_to_rest_on_the_floor` (analytic_fall.rs) — known defect: FALL-001: the floor contact restitution is one bit too high
    """)

KNOWN_DEFECTS_LINE = "//!\n//! **Known defects.** `tests/analytic_fall.rs::a_dropped_body_comes_to_rest_on_the_floor` (FALL-001).\n"

BODY = "\npub fn drop(t: f64) -> f64 { 0.5 * 9.81 * t * t }\n"


def crate(files: dict[str, str], ci: str = CI_DEFAULT, default_features: str = '"std"') -> Path:
    root = Path(tempfile.mkdtemp(prefix="model-section-fixture-"))
    unittest.addModuleCleanup(shutil.rmtree, root, True)
    base = {
        "Cargo.toml": f'[package]\nname = "fixture"\n\n[features]\ndefault = [{default_features}]\nstd = []\nsimd = []\n',
        ".github/workflows/ci.yml": ci,
        "src/lib.rs": "pub mod fall;\n",
        "scripts/model-modules.txt": "C\tsrc/fall.rs\n",
        "scripts/model-exempt.txt": "src/lib.rs\tcrate root\n",
        "scripts/model-section-baseline.txt": "",
        "tests/analytic_fall.rs": TEST_FALL,
        "docs/oracle-status.md": ORACLE_STATUS,
        "src/fall.rs": SECTION_PARAGRAPH + BODY,
    }
    base.update(files)
    for rel, body in base.items():
        if body is None:
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return root


def run(root: Path, write_baseline: bool = False) -> msc.Report:
    return msc.run(str(root), write_baseline)


class Citations(unittest.TestCase):
    """(9) every sentence of equations / conservation / range / approximations cites code
    that resolves; the citation count is compared"""

    def test_sentence_without_a_citation_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("No drag (`drop`).", "No drag.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("field `approximations`, sentence 1 cites no code" in f for f in rep.findings), rep.findings)

    def test_each_clause_after_a_semicolon_is_a_sentence(self) -> None:
        sec = SECTION_PARAGRAPH.replace("nothing is clamped (`drop()`).", "nothing is clamped.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("field `range`, sentence 2 cites no code" in f for f in rep.findings), rep.findings)

    def test_renamed_fn_fails(self) -> None:
        body = BODY.replace("pub fn drop(", "pub fn fall(")
        test = TEST_FALL.replace("fall::drop(", "fall::fall(")
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + body, "tests/analytic_fall.rs": test}))
        self.assertTrue(any("sentence 1 cites `drop`, which is not defined in src/" in f for f in rep.findings), rep.findings)

    def test_path_line_past_the_file_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("`src/fall.rs:1`", "`src/fall.rs:9999`")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("cites `src/fall.rs:9999`, which is not defined" in f for f in rep.findings), rep.findings)

    def test_math_in_backticks_is_no_citation(self) -> None:
        sec = SECTION_PARAGRAPH.replace("No drag (`drop`).", "No drag on `v`.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("sentence 1 cites `v`, which is not defined" in f for f in rep.findings), rep.findings)

    def test_units_need_no_citation(self) -> None:
        rep = run(crate({}))
        self.assertEqual(rep.findings, [])

    def test_counts_sentences_and_citations(self) -> None:
        rep = run(crate({}))
        # equations 1, conservation 2, range 2, approximations 1
        self.assertEqual(rep.sentences, 6)
        self.assertEqual(rep.citations_resolved, 6)


class Passes(unittest.TestCase):
    def test_paragraph_form_passes_and_counts(self) -> None:
        rep = run(crate({}))
        self.assertEqual(rep.findings, [])
        self.assertEqual((rep.modules, rep.with_model, rep.exempt, rep.baseline), (1, 1, 1, 0))
        self.assertEqual(rep.tests_resolved, 2)

    def test_bullet_form_passes(self) -> None:
        rep = run(crate({"src/fall.rs": SECTION_BULLETS + BODY}))
        self.assertEqual(rep.findings, [])
        # two `::fn` references and one bare fn name in the conservation field is not counted
        # (only validation and tiers resolve tests)
        self.assertEqual(rep.tests_resolved, 2)

    def test_bare_fn_name_in_validation_counts(self) -> None:
        sec = SECTION_PARAGRAPH.replace("`::drop_replays_bit_for_bit`", "and `drop_replays_bit_for_bit` (replay)")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.tests_resolved, 2)

    def test_type_path_in_validation_is_not_a_test(self) -> None:
        sec = SECTION_PARAGRAPH.replace("(closed form)", "(closed form, through [`fall::drop`] and `Fall::drop`)")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertEqual(rep.findings, [])

    def test_bare_file_counts_its_live_tests(self) -> None:
        sec = SECTION_PARAGRAPH.replace(
            "`tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),\n//! `::drop_replays_bit_for_bit`.",
            "`tests/analytic_fall.rs` (closed form and replay).")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.tests_resolved, 2)  # the ignored one is not live


class Fields(unittest.TestCase):
    def test_each_missing_field_fails(self) -> None:
        for label in ("**Equations.**", "**Units.**", "**Conservation.**", "**Valid range.**",
                      "**Approximations.**", "**Validation.**", "**Tiers.**"):
            sec = SECTION_PARAGRAPH.replace(label, "**Notes.**")
            rep = run(crate({"src/fall.rs": sec + BODY}))
            self.assertTrue(any("lacks the field" in f for f in rep.findings), (label, rep.findings))
            self.assertTrue(any("no field: **Notes.**" in f for f in rep.findings), (label, rep.findings))

    def test_empty_field_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("**Approximations.** No drag (`drop`).", "**Approximations.**")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("`approximations` is empty" in f for f in rep.findings), rep.findings)

    def test_section_ends_at_next_heading(self) -> None:
        sec = SECTION_PARAGRAPH.replace("//! **Tiers.** L0 only.\n", "//! # Notes\n//!\n//! **Tiers.** L0 only.\n")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("lacks the field `tiers`" in f for f in rep.findings), rep.findings)

    def test_no_model_heading_is_a_missing_section(self) -> None:
        sec = SECTION_PARAGRAPH.replace("# MODEL — free fall", "# Model notes")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("no MODEL section" in f for f in rep.findings), rep.findings)


class Tests(unittest.TestCase):
    def test_missing_test_file_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("tests/analytic_fall.rs", "tests/analytic_fall_v2.rs")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("tests/analytic_fall_v2.rs, which does not exist" in f for f in rep.findings), rep.findings)

    def test_missing_fn_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("drop_replays_bit_for_bit", "drop_replays_twice")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("::drop_replays_twice, which is not a fn" in f for f in rep.findings), rep.findings)

    def test_ignored_fn_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("drop_replays_bit_for_bit", "a_body_with_drag_reaches_terminal_velocity")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("#[ignore]d" in f for f in rep.findings), rep.findings)

    def test_multi_line_ignore_attribute_is_seen(self) -> None:
        t = TEST_FALL.replace('#[ignore = "src gap: no drag yet"]', '#[ignore = \\\n    "src gap: no drag yet"\n]')
        sec = SECTION_PARAGRAPH.replace("drop_replays_bit_for_bit", "a_body_with_drag_reaches_terminal_velocity")
        rep = run(crate({"src/fall.rs": sec + BODY, "tests/analytic_fall.rs": t}))
        self.assertTrue(any("#[ignore]d" in f for f in rep.findings), rep.findings)

    def test_fn_before_any_file_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace(
            "`tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),\n//! `::drop_replays_bit_for_bit`.",
            "`::a_body_falls_g_h_squared_over_two` and `tests/analytic_fall.rs::drop_replays_bit_for_bit`.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("with no tests/<file>.rs before it" in f for f in rep.findings), rep.findings)

    def test_bare_name_that_is_no_fn_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("`::drop_replays_bit_for_bit`", "`drop_replays_bit_for_bit_v9`")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("`drop_replays_bit_for_bit_v9`, which is not a fn" in f for f in rep.findings), rep.findings)

    def test_file_with_only_ignored_tests_fails(self) -> None:
        t = TEST_FALL.replace("#[test]\nfn a_body", "#[test]\n#[ignore]\nfn a_body").replace(
            "#[test]\nfn drop_", "#[test]\n#[ignore]\nfn drop_")
        self.assertEqual(t.count("#[ignore"), 4)
        sec = SECTION_PARAGRAPH.replace(
            "`tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),\n//! `::drop_replays_bit_for_bit`.",
            "`tests/analytic_fall.rs`.")
        rep = run(crate({"src/fall.rs": sec + BODY, "tests/analytic_fall.rs": t}))
        self.assertTrue(any("whose tests are all #[ignore]d" in f for f in rep.findings), rep.findings)

    def test_no_test_named_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace(
            "`tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),\n//! `::drop_replays_bit_for_bit`.",
            "checked by hand.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("`validation` names no test" in f for f in rep.findings), rep.findings)


class CiFeatures(unittest.TestCase):
    def test_feature_gate_not_run_by_ci_fails(self) -> None:
        t = TEST_FALL.replace('#![cfg(feature = "std")]', '#![cfg(all(feature = "std", feature = "simd"))]')
        rep = run(crate({"tests/analytic_fall.rs": t}))
        self.assertTrue(any("needs features ['simd', 'std'] that no cargo test run" in f for f in rep.findings), rep.findings)

    def test_feature_gate_run_by_a_features_command_passes(self) -> None:
        t = TEST_FALL.replace('#![cfg(feature = "std")]', '#![cfg(all(feature = "std", feature = "simd"))]')
        ci = CI_DEFAULT + '      - run: cargo test --features "simd"\n'
        rep = run(crate({"tests/analytic_fall.rs": t}, ci=ci))
        self.assertEqual(rep.findings, [])

    def test_test_filter_selects_only_its_file(self) -> None:
        t = TEST_FALL.replace('#![cfg(feature = "std")]', '#![cfg(feature = "simd")]')
        ci = CI_DEFAULT + '      - run: cargo test --test other_file --features "simd"\n'
        rep = run(crate({"tests/analytic_fall.rs": t}, ci=ci))
        self.assertTrue(any("no cargo test run" in f for f in rep.findings), rep.findings)
        ci2 = CI_DEFAULT + '      - run: cargo test --test analytic_fall --features "simd"\n'
        rep2 = run(crate({"tests/analytic_fall.rs": t}, ci=ci2))
        self.assertEqual(rep2.findings, [])

    def test_no_default_features_drops_std(self) -> None:
        ci = '      - run: cargo test --no-default-features\n'
        rep = run(crate({}, ci=ci))
        self.assertTrue(any("needs features ['std']" in f for f in rep.findings), rep.findings)

    def test_not_feature_is_not_required(self) -> None:
        t = TEST_FALL.replace('#![cfg(feature = "std")]', '#![cfg(not(feature = "simd"))]')
        rep = run(crate({"tests/analytic_fall.rs": t}))
        self.assertEqual(rep.findings, [])

    def test_ci_without_any_test_run_fails(self) -> None:
        rep = run(crate({}, ci="jobs:\n  test:\n    steps:\n      - run: cargo test --lib\n"))
        self.assertTrue(any("has no cargo test run that builds tests/" in f for f in rep.findings), rep.findings)


class Tiers(unittest.TestCase):
    def test_lighter_tier_without_error_test_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("**Tiers.** L0 only.", "**Tiers.** L0 (3D); L1 drops the depth axis.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("lists a lighter tier than L0 (L1) but names no error test" in f for f in rep.findings), rep.findings)

    def test_lighter_tier_with_error_test_passes(self) -> None:
        sec = SECTION_PARAGRAPH.replace(
            "**Tiers.** L0 only.",
            "**Tiers.** L0 (3D); L1 drops the depth axis, error vs L0 in `tests/analytic_fall.rs::drop_replays_bit_for_bit`.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.tests_resolved, 3)

    def test_tiers_without_l0_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("**Tiers.** L0 only.", "**Tiers.** L1 only.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("has no `L0`" in f for f in rep.findings), rep.findings)

    def test_tiers_without_any_tier_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("**Tiers.** L0 only.", "**Tiers.** one precision.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("names no tier" in f for f in rep.findings), rep.findings)


class KnownDefects(unittest.TestCase):
    def test_known_defect_field_passes_when_ignored_with_reason_and_listed(self) -> None:
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + KNOWN_DEFECTS_LINE + BODY}))
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.tests_resolved, 3)

    def test_japanese_bullet_label_is_the_field(self) -> None:
        sec = SECTION_BULLETS + "//! - **既知の欠陥 (known defects)**: `tests/analytic_fall.rs::\n//!   a_dropped_body_comes_to_rest_on_the_floor` (FALL-001).\n"
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertEqual(rep.findings, [])

    def test_bare_fn_name_resolves_across_test_files(self) -> None:
        line = "//!\n//! **Known defects.** `a_dropped_body_comes_to_rest_on_the_floor` (FALL-001).\n"
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + line + BODY}))
        self.assertEqual(rep.findings, [])

    def test_known_defect_named_as_validation_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace("`::drop_replays_bit_for_bit`", "`::a_dropped_body_comes_to_rest_on_the_floor`")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertTrue(any("validation names tests/analytic_fall.rs::a_dropped_body_comes_to_rest_on_the_floor, which is #[ignore]d" in f
                            for f in rep.findings), rep.findings)

    def test_fixed_defect_must_leave_the_list(self) -> None:
        t = TEST_FALL.replace('#[ignore = "known defect: FALL-001: the floor contact restitution is one bit too high"]\n', "")
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + KNOWN_DEFECTS_LINE + BODY, "tests/analytic_fall.rs": t}))
        self.assertTrue(any("which is not #[ignore]d (a known defect is a red oracle)" in f for f in rep.findings), rep.findings)

    def test_reason_must_start_with_known_defect(self) -> None:
        t = TEST_FALL.replace('"known defect: FALL-001:', '"src gap: FALL-001:')
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + KNOWN_DEFECTS_LINE + BODY, "tests/analytic_fall.rs": t}))
        self.assertTrue(any('reason does not start with "known defect:"' in f for f in rep.findings), rep.findings)

    def test_must_be_listed_in_oracle_status(self) -> None:
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + KNOWN_DEFECTS_LINE + BODY, "docs/oracle-status.md": "# Oracle Status\n"}))
        self.assertTrue(any("which is not listed in docs/oracle-status.md" in f for f in rep.findings), rep.findings)

    def test_unknown_fn_fails(self) -> None:
        line = KNOWN_DEFECTS_LINE.replace("comes_to_rest_on_the_floor", "comes_to_rest_on_the_ceiling")
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + line + BODY}))
        self.assertTrue(any("known defects names tests/analytic_fall.rs::a_dropped_body_comes_to_rest_on_the_ceiling, which is not a fn" in f
                            for f in rep.findings), rep.findings)

    def test_empty_known_defects_field_fails(self) -> None:
        line = "//!\n//! **Known defects.** none at the moment.\n"
        rep = run(crate({"src/fall.rs": SECTION_PARAGRAPH + line + BODY}))
        self.assertTrue(any("`known defects` names no ignored oracle" in f for f in rep.findings), rep.findings)

    def test_known_defects_is_not_required(self) -> None:
        rep = run(crate({}))
        self.assertEqual(rep.findings, [])


class Classification(unittest.TestCase):
    def test_unclassified_src_file_fails(self) -> None:
        rep = run(crate({"src/wind.rs": "//! wind\npub fn w() {}\n"}))
        self.assertTrue(any("src/wind.rs is neither in" in f for f in rep.findings), rep.findings)

    def test_file_in_both_lists_fails(self) -> None:
        rep = run(crate({"scripts/model-exempt.txt": "src/lib.rs\tcrate root\nsrc/fall.rs\tno reason at all here\n"}))
        self.assertTrue(any("src/fall.rs is both a model module and exempt" in f for f in rep.findings), rep.findings)

    def test_listed_but_absent_fails(self) -> None:
        rep = run(crate({"scripts/model-modules.txt": "C\tsrc/fall.rs\nC\tsrc/gone.rs\n",
                         "scripts/model-exempt.txt": "src/lib.rs\tcrate root\nsrc/also_gone.rs\tnever existed\n"}))
        self.assertTrue(any("model-modules.txt: src/gone.rs does not exist" in f for f in rep.findings), rep.findings)
        self.assertTrue(any("model-exempt.txt: src/also_gone.rs does not exist" in f for f in rep.findings), rep.findings)

    def test_bad_group_and_short_reason_fail(self) -> None:
        rep = run(crate({"scripts/model-modules.txt": "Z\tsrc/fall.rs\n",
                         "scripts/model-exempt.txt": "src/lib.rs\troot\n"}))
        self.assertTrue(any("has group 'Z'" in f for f in rep.findings), rep.findings)
        self.assertTrue(any("src/lib.rs has no reason" in f for f in rep.findings), rep.findings)

    def test_malformed_list_line_exits(self) -> None:
        with self.assertRaises(SystemExit):
            run(crate({"scripts/model-modules.txt": "C src/fall.rs\n"}))


class Ratchet(unittest.TestCase):
    def test_missing_section_not_in_baseline_fails(self) -> None:
        rep = run(crate({"src/fall.rs": "//! Falling bodies.\n" + BODY}))
        self.assertTrue(any("src/fall.rs: no MODEL section (a C-group physics module) and not in" in f
                            for f in rep.findings), rep.findings)

    def test_missing_section_in_baseline_passes(self) -> None:
        rep = run(crate({"src/fall.rs": "//! Falling bodies.\n" + BODY,
                         "src/wind.rs": SECTION_PARAGRAPH.replace("Falling", "Wind") + BODY,
                         "src/lib.rs": "pub mod fall;\npub mod wind;\n",
                         "scripts/model-modules.txt": "C\tsrc/fall.rs\nB\tsrc/wind.rs\n",
                         "scripts/model-section-baseline.txt": "src/fall.rs\n"}))
        self.assertEqual(rep.findings, [])
        self.assertEqual((rep.modules, rep.with_model, rep.baseline), (2, 1, 1))

    def test_stale_baseline_line_fails(self) -> None:
        rep = run(crate({"scripts/model-section-baseline.txt": "src/fall.rs\n"}))
        self.assertTrue(any("src/fall.rs now has a MODEL section, remove the line" in f for f in rep.findings), rep.findings)

    def test_baseline_line_that_is_no_module_fails(self) -> None:
        rep = run(crate({"scripts/model-section-baseline.txt": "src/lib.rs\n"}))
        self.assertTrue(any("src/lib.rs is not a model module" in f for f in rep.findings), rep.findings)

    def test_write_baseline_regenerates(self) -> None:
        root = crate({"src/fall.rs": "//! Falling bodies.\n" + BODY,
                      "src/wind.rs": SECTION_PARAGRAPH.replace("Falling", "Wind") + BODY,
                      "src/lib.rs": "pub mod fall;\npub mod wind;\n",
                      "scripts/model-modules.txt": "C\tsrc/fall.rs\nB\tsrc/wind.rs\n",
                      "scripts/model-section-baseline.txt": "src/wind.rs\n"})
        rep = run(root, write_baseline=True)
        self.assertEqual(rep.findings, [])
        lines = [l for l in (root / "scripts/model-section-baseline.txt").read_text().splitlines() if not l.startswith("#")]
        self.assertEqual(lines, ["src/fall.rs"])
        self.assertEqual(run(root).findings, [])


class ComparedZero(unittest.TestCase):
    def test_no_module_fails(self) -> None:
        rep = run(crate({"scripts/model-modules.txt": "",
                         "scripts/model-exempt.txt": "src/lib.rs\tcrate root\nsrc/fall.rs\tnothing in it\n"}))
        self.assertIn("no model module was classified (compared 0)", rep.findings)

    def test_no_section_fails(self) -> None:
        rep = run(crate({"src/fall.rs": "//! Falling bodies.\n" + BODY,
                         "scripts/model-section-baseline.txt": "src/fall.rs\n"}))
        self.assertIn("no MODEL section was read (compared 0 sections)", rep.findings)

    def test_no_test_resolved_fails(self) -> None:
        sec = SECTION_PARAGRAPH.replace(
            "`tests/analytic_fall.rs::a_body_falls_g_h_squared_over_two` (closed form),\n//! `::drop_replays_bit_for_bit`.",
            "`tests/analytic_fall_v2.rs`.")
        rep = run(crate({"src/fall.rs": sec + BODY}))
        self.assertIn("no test named by a MODEL section was resolved (compared 0 tests)", rep.findings)


class Cli(unittest.TestCase):
    def test_main_prints_counts_and_exit_codes(self) -> None:
        import io
        from contextlib import redirect_stdout
        root = crate({})
        out = io.StringIO()
        with redirect_stdout(out):
            rc = msc.main(["--root", str(root)])
        self.assertEqual(rc, 0)
        self.assertIn('MODEL-SECTION-CHECK: {"modules": 1, "with_model": 1', out.getvalue())
        root2 = crate({"src/wind.rs": "//! wind\npub fn w() {}\n"})
        out2 = io.StringIO()
        with redirect_stdout(out2):
            rc2 = msc.main(["--root", str(root2)])
        self.assertEqual(rc2, 1)
        self.assertIn("model-section-check: src/wind.rs is neither in", out2.getvalue())


if __name__ == "__main__":
    unittest.main()
