"""Oracles of scripts/preflight_ci_parity.py: the commands of scripts/preflight.sh and of the
landing workflows are compared both ways after normalising, the lists of one-sided commands
are checked against what the two sides run, and a side with no command fails.

run: python3 scripts/test_preflight_ci_parity.py
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("parity", os.path.join(HERE, "preflight_ci_parity.py"))
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)

CI = """name: CI
on:
  push:
    branches: [main, 'ci/**']
jobs:
  t:
    runs-on: ubuntu-latest
    steps:
      - name: a
        run: cargo test
      - name: b
        run: cargo test --features "parallel"
      - name: c
        run: 'cargo test --lib --features "ffi" "ffi::"'
      - name: d
        run: cargo test --lib --features "std,simd"
      - name: e
        run: python scripts/x.py --check
"""
PF = """NATIVE='std,simd'
cargo test --no-fail-fast
cargo test --no-fail-fast --features "parallel"
cargo test --lib --features "ffi" "ffi::"
cargo test --lib --features "$NATIVE"
python3 scripts/x.py --check
"""
EVIDENCE = '[workflows]\n"ci.yml" = "landing"\n"weekly.yml" = "schedule-only"\n'


def tree(ci: str = CI, pf: str = PF, evidence: str = EVIDENCE, more: dict[str, str] | None = None) -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, ".github", "workflows"))
    os.makedirs(os.path.join(root, "scripts"))
    with open(os.path.join(root, ".github", "workflows", "ci.yml"), "w", encoding="utf-8") as f:
        f.write(ci)
    for name, text in (more or {}).items():
        with open(os.path.join(root, ".github", "workflows", name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(root, ".github", "ci-evidence.toml"), "w", encoding="utf-8") as f:
        f.write(evidence)
    with open(os.path.join(root, "scripts", "preflight.sh"), "w", encoding="utf-8") as f:
        f.write(pf)
    return root


def run(root: str, exempt=None, local_only=None, families=None, for_cost=()):
    return parity.check(root, exempt=exempt or {}, local_only=local_only or {}, families=families or {},
                        for_cost=for_cost)


def step(run_text: str, extra: str = "") -> str:
    return f"      - name: x\n{extra}        run: {run_text}\n"


class Normalise(unittest.TestCase):
    def test_matching_commands_pass_after_normalising(self):
        errors, n = run(tree())
        self.assertEqual(errors, [])
        self.assertEqual((n["ci"], n["preflight"]), (5, 5))

    def test_a_variable_inside_a_longer_word_is_expanded(self):
        errors, _ = run(tree(CI + step('cargo clippy --features "std,simd,neural" -- -D warnings'),
                             PF + 'cargo clippy --features "$NATIVE,neural" -- -D warnings\n'))
        self.assertEqual(errors, [])

    def test_option_order_before_the_double_dash_does_not_matter(self):
        errors, _ = run(tree(CI + step('cargo clippy --features "a" --all-targets -- -D warnings'),
                             PF + 'cargo clippy --all-targets --features "a" -- -D warnings\n'))
        self.assertEqual(errors, [])
        errors, _ = run(tree(CI + step("cargo clippy --lib -- -D warnings -A x"),
                             PF + "cargo clippy --lib -- -A x -D warnings\n"))
        self.assertEqual(len(errors), 2, "after ` -- ` the order is the command's (lint order matters)")

    def test_a_different_feature_set_is_not_a_match(self):
        errors, _ = run(tree(pf=PF.replace("NATIVE='std,simd'", "NATIVE='std,simd,parallel'")))
        self.assertTrue(any("std,simd" in e for e in errors), errors)

    def test_python_and_bash_spellings_are_one_command(self):
        errors, _ = run(tree(CI + step("bash scripts/run.sh --a"), PF + "scripts/run.sh --a\n"))
        self.assertEqual(errors, [])

    def test_redirections_pipes_and_conditions_are_not_part_of_the_command(self):
        ci = CI + step("|\n          cargo test --lib 2>&1 | tee log.txt\n          if ! cargo build --lib > /dev/null; then exit 1; fi")
        errors, _ = run(tree(ci, PF + "cargo test --lib\ncargo build --lib\n"))
        self.assertEqual(errors, [])
        # the right side of a pipe reads output, it is not a command of the step
        errors, _ = run(tree(CI + step("cargo test --lib | scripts/filter.sh"), PF + "cargo test --lib\n"))
        self.assertEqual(errors, [])

    def test_block_scalars_and_continuations(self):
        ci = CI + "      - name: f\n        run: >-\n          cargo build\n          --lib\n          --features q\n"
        ci += "      - name: g\n        run: |\n          cargo check \\\n            --lib\n"
        errors, _ = run(tree(ci, PF + 'cargo build --lib --features "q"\ncargo check --lib\n'))
        self.assertEqual(errors, [])

    def test_comments_are_not_commands(self):
        ci = CI + "      # run: cargo test --features nothing\n" + step("|\n          # cargo test --features none\n          echo hi")
        self.assertEqual(run(tree(ci))[0], [])
        errors, _ = run(tree(pf=PF + "# cargo build --release\ncargo test --lib  # cargo build --x\n",
                             ci=CI + step("cargo test --lib")))
        self.assertEqual(errors, [])


class Environment(unittest.TestCase):
    def test_a_build_assignment_is_part_of_the_command(self):
        ci = CI + '      - name: e\n        run: RUSTFLAGS="-C target-cpu=native" cargo test --lib\n'
        errors, n = run(tree(ci, PF + "cargo test --lib\n"))
        self.assertTrue(any("RUSTFLAGS" in e for e in errors), errors)
        # the same assignments, quoted differently and in another order, match
        ci2 = CI + step("CLIPPY_CONF_DIR=a RUSTFLAGS='-Cx' cargo test --lib")
        errors, _ = run(tree(ci2, PF + 'RUSTFLAGS="-Cx" CLIPPY_CONF_DIR=a cargo test --lib\n'))
        self.assertEqual(errors, [])

    def test_the_env_of_the_step_job_and_workflow_counts(self):
        everywhere = PF.replace("cargo test", 'RUSTDOCFLAGS="-Dwarnings" cargo test').replace(
            "python3 scripts/x.py", 'RUSTDOCFLAGS="-Dwarnings" python3 scripts/x.py')
        cases = (  # (workflow, preflight's other commands: the job / workflow env reaches them too)
            (CI + step("cargo doc --lib", "        env:\n          RUSTDOCFLAGS: \"-Dwarnings\"\n"), PF),
            (CI.replace("    runs-on: ubuntu-latest\n", "    runs-on: ubuntu-latest\n    env:\n      RUSTDOCFLAGS: -Dwarnings\n")
             + step("cargo doc --lib"), everywhere),
            (CI.replace("jobs:\n", "env:\n  RUSTDOCFLAGS: -Dwarnings\njobs:\n") + step("cargo doc --lib"), everywhere))
        for ci, pf in cases:
            errors, _ = run(tree(ci, pf + 'RUSTDOCFLAGS="-Dwarnings" cargo doc --lib\n'))
            self.assertEqual(errors, [], ci)
            errors, _ = run(tree(ci, pf + "cargo doc --lib\n"))
            self.assertTrue(any("RUSTDOCFLAGS" in e for e in errors), (ci, errors))

    def test_an_assignment_that_changes_no_build_is_not_compared(self):
        ci = CI + step("cargo build --lib", "        env:\n          CARGO_TERM_COLOR: never\n")
        self.assertEqual(run(tree(ci, PF + "cargo build --lib\n"))[0], [])

    def test_the_directory_is_part_of_the_command(self):
        ci = CI + step("cargo build --bins", "        working-directory: fuzz\n")
        errors, _ = run(tree(ci, PF + "(cd fuzz && cargo build --bins)\n"))
        self.assertEqual(errors, [])
        errors, _ = run(tree(ci, PF + "cargo build --bins\n"))
        self.assertEqual(len(errors), 2, errors)
        ci = CI + step("|\n          cd fuzz\n          cargo build --bins")
        self.assertEqual(run(tree(ci, PF + "(cd fuzz && cargo build --bins)\n"))[0], [])


class Sides(unittest.TestCase):
    def test_a_ci_command_missing_from_preflight_fails(self):
        errors, _ = run(tree(pf=PF.replace('cargo test --no-fail-fast --features "parallel"\n', "")))
        self.assertEqual(len(errors), 1)
        self.assertIn("ci.yml / t runs", errors[0])
        self.assertIn('--features "parallel"', errors[0])

    def test_a_preflight_command_missing_from_ci_fails(self):
        errors, _ = run(tree(pf=PF + "cargo clippy --lib -- -D warnings\n"))
        self.assertEqual(len(errors), 1)
        self.assertIn("no landing workflow does", errors[0])

    def test_only_landing_workflows_are_compared(self):
        weekly = "on:\n  schedule:\n    - cron: '0 0 * * 0'\njobs:\n  w:\n    steps:\n" + step("cargo mutants")
        self.assertEqual(run(tree(more={"weekly.yml": weekly}))[0], [], "a schedule-only command is not a landing one")
        errors, _ = run(tree(pf=PF + "cargo mutants\n", more={"weekly.yml": weekly}))
        self.assertEqual(len(errors), 1, "a schedule-only run is no CI counterpart")
        ev = EVIDENCE.replace('"schedule-only"', '"landing"')
        self.assertEqual(run(tree(pf=PF + "cargo mutants\n", evidence=ev, more={"weekly.yml": weekly}))[0], [])

    def test_a_side_with_no_command_fails(self):
        errors, n = run(tree(ci="jobs: {}\n"))
        self.assertEqual(n["ci"], 0)
        self.assertTrue(any("compared nothing" in e for e in errors), errors)
        errors, n = run(tree(pf="echo nothing\n"))
        self.assertEqual(n["preflight"], 0)
        self.assertTrue(any("preflight.sh has no compared command" in e for e in errors), errors)


class Lists(unittest.TestCase):
    def test_exempt_covers_a_ci_command_and_goes_stale(self):
        pf = PF.replace("python3 scripts/x.py --check\n", "")
        self.assertEqual(run(tree(pf=pf), exempt={"python3 scripts/x.py --check": "needs the CI context x"})[0], [])
        errors, _ = run(tree(), exempt={"python3 scripts/x.py --check": "needs the CI context x"})
        self.assertTrue(any("runs it now (a stale entry)" in e for e in errors), errors)
        errors, _ = run(tree(), exempt={"cargo gone": "needs the CI context x"})
        self.assertTrue(any("no landing workflow runs it (a stale entry)" in e for e in errors), errors)
        errors, _ = run(tree(pf=pf), exempt={"python3 scripts/x.py --check": "ci"})
        self.assertTrue(any("needs a reason" in e for e in errors), errors)

    def test_a_command_left_out_for_its_cost_must_run_on_a_ci_push(self):
        heavy = "cargo hack build --feature-powerset --lib"
        pf = PF
        ex = {heavy: "cost; CI runs it on every ci/** push"}
        for cond, ok in (("", True), ("    if: github.event_name != 'schedule'\n", True),
                         ("    if: needs.changes.outputs.run == 'true'\n", True),
                         ("    if: github.event_name == 'schedule'\n", False),
                         ("    if: ${{ github.ref == 'refs/heads/main' }}\n", False)):
            ci = CI + f"  heavy:\n{cond}    runs-on: x\n    steps:\n" + step(heavy)
            errors, _ = run(tree(ci, pf), exempt=ex, for_cost=(heavy,))
            self.assertEqual(errors == [], ok, (cond, errors))
        ci = CI + "  heavy:\n    steps:\n" + step(heavy)
        errors, _ = run(tree(ci, pf), exempt={}, for_cost=(heavy,))
        self.assertTrue(any("not in EXEMPT" in e for e in errors), errors)
        errors, _ = run(tree(pf=pf), exempt=ex, for_cost=(heavy,))
        self.assertTrue(any("no landing workflow runs it" in e for e in errors), "gone from CI: the exemption is stale")

    def test_a_family_covers_whole_commands_only_and_goes_stale(self):
        pf = PF.replace("python3 scripts/x.py --check\n", "")
        self.assertEqual(run(tree(pf=pf), families={r"python3 scripts/x\.py .*": "plumbing of the CI run"})[0], [])
        errors, _ = run(tree(pf=pf), families={r"python3 scripts/x\.py": "plumbing of the CI run"})
        self.assertEqual(len(errors), 2, "a pattern matches the whole command (so it is also stale here)")
        errors, _ = run(tree(), families={r"cargo gone .*": "plumbing of the CI run"})
        self.assertTrue(any("matches no landing-workflow command" in e for e in errors), errors)

    def test_local_only_names_and_checks_its_ci_counterpart(self):
        pf = PF + "cargo check --lib\n"
        ok = {"cargo check --lib": ("non-gating", "the quick form of a CI test", (r"cargo test --features \S+ --lib",))}
        errors, n = run(tree(pf=pf), local_only=ok)
        self.assertEqual(errors, [])
        self.assertEqual(n["gating_local_only"], 0)
        gone = {"cargo check --lib": ("non-gating", "the quick form of a CI test", (r"cargo test --release",))}
        errors, _ = run(tree(pf=pf), local_only=gone)
        self.assertTrue(any("the claimed counterpart" in e for e in errors), errors)
        none = {"cargo check --lib": ("non-gating", "the quick form of a CI test", ())}
        self.assertTrue(any("names the CI commands" in e for e in run(tree(pf=pf), local_only=none)[0]))
        bad = {"cargo check --lib": ("maybe", "the quick form of a CI test", (r"cargo test",))}
        self.assertTrue(any("gating or non-gating" in e for e in run(tree(pf=pf), local_only=bad)[0]))
        errors, _ = run(tree(), local_only=ok)
        self.assertTrue(any("does not run it (a stale entry)" in e for e in errors), errors)
        ci = CI + step("cargo check --lib")
        errors, _ = run(tree(ci, pf), local_only=ok)
        self.assertTrue(any("a landing workflow runs it now" in e for e in errors), errors)

    def test_a_gating_local_only_command_is_counted_and_fails_the_opt_in(self):
        pf = PF + "cargo check --lib\n"
        gating = {"cargo check --lib": ("gating", "no landing workflow checks this yet", ())}
        errors, n = run(tree(pf=pf), local_only=gating)
        self.assertEqual((errors, n["gating_local_only"]), ([], 1))
        with_counterpart = {"cargo check --lib": ("gating", "no landing workflow checks this yet", (r"cargo test",))}
        self.assertTrue(any("names no CI command" in e for e in run(tree(pf=pf), local_only=with_counterpart)[0]))
        root = tree(pf=pf)
        saved = (parity.ROOT, parity.LOCAL_ONLY, parity.EXEMPT, parity.EXEMPT_FAMILIES)
        try:
            parity.LOCAL_ONLY, parity.EXEMPT, parity.EXEMPT_FAMILIES = gating, {}, {}
            parity.check.__defaults__ = (root, None, None, None, ())
            self.assertEqual(parity.main([]), 0)
            self.assertEqual(parity.main(["--opt-in"]), 1)
        finally:
            parity.ROOT, parity.LOCAL_ONLY, parity.EXEMPT, parity.EXEMPT_FAMILIES = saved
            parity.check.__defaults__ = (saved[0], None, None, None, None)


class ThisRepository(unittest.TestCase):
    def test_this_repository(self):
        errors, n = parity.check()
        self.assertEqual(errors, [])
        self.assertGreater(n["ci"], 100)
        self.assertGreater(n["preflight"], 100)
        self.assertEqual(n["gating_local_only"], 0, "a gate CI lacks blocks landing by CI evidence")


if __name__ == "__main__":
    unittest.main()
