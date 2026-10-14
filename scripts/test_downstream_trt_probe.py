#!/usr/bin/env python3
"""Tests for scripts/downstream_trt_probe.sh: the GPU probe of downstream_check.sh.

The function is sourced into bash with a stand-in `cargo` that either fails to build
(the message cargo gives for a dependency whose rust-version is newer than the
toolchain) or builds and prints / does not print an adapter. A probe that does not
build must be reported as that build failure with the compiler's message, never as a
missing adapter; a probe that builds and prints nothing must still be a missing
adapter. test_mutant_* turn the script into its wrong forms and require the checks to
notice, so they cannot pass on a script that merges the two.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "downstream_trt_probe.sh")

MSRV = (
    "error: rustc 1.92.0 is not supported by the following package:\n"
    "  alice-ml@0.3.0 requires rustc 1.98.1\n"
)

COLOURED = (
    "\x1b[1m\x1b[91merror\x1b[0m\x1b[1m: rustc 1.92.0 is not supported by the following package:\x1b[0m\n"
    "  alice-ml@0.3.0 requires rustc 1.98.1\n"
)
WARN_ONLY = "\x1b[1m\x1b[93mwarning\x1b[0m: unused manifest key\n  --> x\n"
NO_MATCH = "linker `cc` not found\nsome line\n"

# FAKE_BUILD: ok | fail ; FAKE_RUN: adapter | empty | exit1
DRIVER = r"""
set -euo pipefail
. "$PROBE"
cargo() {
    case "$1" in
        build)
            case " $* " in *" --color never "*) ;; *) [ -z "${REQUIRE_NO_COLOR:-}" ] || { echo "probe built with colour" >&2; return 102; } ;; esac
            if [ "$FAKE_BUILD" = fail ]; then printf '%s' "$FAKE_BUILD_ERR" >&2; return 101; fi
            return 0 ;;
        run)
            case "$FAKE_RUN" in
                adapter) echo "Mesa llvmpipe (Vulkan)"; return 0 ;;
                empty) return 0 ;;
                exit1) echo "no adapter found" >&2; return 1 ;;
            esac ;;
    esac
}
failed=(); summary=()
trt_gpu_probe
echo "RC_FAILED=${failed[*]-}"
echo "RC_SUMMARY=${summary[*]-}"
"""


def run_probe(build: str, run: str, script: str = PROBE, build_err: str = MSRV,
              probe_env: str | None = None) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "logs"))
        os.makedirs(os.path.join(root, "ALICE-TRT"))
        env = dict(
            os.environ,
            PROBE=script,
            ROOT=root,
            TRT=os.path.join(root, "ALICE-TRT"),
            FAKE_BUILD=build,
            FAKE_RUN=run,
            FAKE_BUILD_ERR=build_err,
        )
        env.pop("TRT_GPU_PROBE", None)
        if probe_env is not None:
            env["TRT_GPU_PROBE"] = probe_env
        p = subprocess.run(["bash", "-c", DRIVER], env=env, capture_output=True, text=True, check=False)
        return p.stdout, p.stderr


def failures(script: str = PROBE) -> list[str]:
    """What the script gets wrong (empty list: nothing)."""
    wrong = []
    out, err = run_probe("fail", "adapter", script)
    if "RC_FAILED=trt (GPU probe build failed)" not in out:
        wrong.append(f"build error not its own failure: {out!r}")
    if "alice-ml@0.3.0 requires rustc 1.98.1" not in err or "rustc 1.92.0 is not supported" not in err:
        wrong.append(f"build error message lost: {err!r}")
    if "no GPU adapter" in err or "no GPU adapter" in out:
        wrong.append("build error reported as a missing adapter")
    for build_err in (COLOURED, NO_MATCH):
        out, err = run_probe("fail", "adapter", script, build_err=build_err)
        if "RC_FAILED=trt (GPU probe build failed)" not in out:
            wrong.append(f"build error {build_err!r}: script did not report it: {out!r}")
        tail = err.split("the adapter was not asked")[-1]
        if (": error: rustc 1.92.0" if build_err is COLOURED else "linker") not in tail:
            wrong.append(f"build error {build_err!r}: message not extracted: {err!r}")
    os.environ["REQUIRE_NO_COLOR"] = "1"
    try:
        out, _ = run_probe("ok", "adapter", script)
    finally:
        del os.environ["REQUIRE_NO_COLOR"]
    if "RC_FAILED=\n" not in out:
        wrong.append("probe built with colour")
    for run in ("empty", "exit1"):
        out, err = run_probe("ok", run, script)
        if "RC_FAILED=trt (no GPU adapter)" not in out:
            wrong.append(f"{run}: not reported as no adapter: {out!r}")
        if "ALICE-TRT finds no GPU adapter" not in err:
            wrong.append(f"{run}: no-adapter message missing: {err!r}")
        if "failed to build" in err:
            wrong.append(f"{run}: reported as a build failure")
    out, err = run_probe("ok", "adapter", script)
    if "RC_FAILED=\n" not in out:
        wrong.append(f"adapter found but failed: {out!r}")
    if "trt: GPU adapter: Mesa llvmpipe (Vulkan)" not in out:
        wrong.append(f"adapter not reported: {out!r}")
    return wrong


class ProbeTest(unittest.TestCase):
    def test_coloured_error_is_extracted(self) -> None:
        out, err = run_probe("fail", "adapter", build_err=COLOURED)
        self.assertIn("RC_FAILED=trt (GPU probe build failed)", out)
        self.assertIn("failed to build, so the adapter was not asked: error: rustc 1.92.0 is not supported", err)
        self.assertIn("alice-ml@0.3.0 requires rustc 1.98.1", err.split("not asked:")[1])
        self.assertNotIn("RC_FAILED=trt (no GPU adapter)", out)

    def test_warning_fallback(self) -> None:
        out, err = run_probe("fail", "adapter", build_err=WARN_ONLY)
        self.assertIn("RC_FAILED=trt (GPU probe build failed)", out)
        self.assertIn("not asked: warning: unused manifest key", err)

    def test_no_match_falls_back_to_last_lines_and_does_not_abort(self) -> None:
        out, err = run_probe("fail", "adapter", build_err=NO_MATCH)
        self.assertIn("RC_FAILED=trt (GPU probe build failed)", out)  # the script went on to report
        self.assertIn("linker `cc` not found", err.split("not asked:")[1])

    def test_skip_path(self) -> None:
        out, err = run_probe("fail", "empty", probe_env="0")
        self.assertIn("RC_FAILED=\n", out)
        self.assertIn("not probed (TRT_GPU_PROBE=0)", out)
        self.assertIn("GPU probe skipped", out)

    def test_probe_built_without_colour(self) -> None:
        # the stand-in cargo rejects a build without `--color never`
        with tempfile.TemporaryDirectory():
            os.environ["REQUIRE_NO_COLOR"] = "1"
            try:
                out, _ = run_probe("ok", "adapter")
            finally:
                del os.environ["REQUIRE_NO_COLOR"]
        self.assertIn("RC_FAILED=\n", out)

    def test_script_is_correct(self) -> None:
        self.assertEqual(failures(), [])

    def mutated(self, old: str, new: str, *more: str) -> list[str]:
        with open(PROBE, encoding="utf-8") as f:
            text = f.read()
        self.assertIn(old, text, "mutation target missing: the test is out of date")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.sh")
            with open(path, "w", encoding="utf-8") as f:
                text = text.replace(old, new, 1)
                pairs = list(zip(more[::2], more[1::2]))
                for o, n in pairs:
                    self.assertIn(o, text, "mutation target missing: the test is out of date")
                    text = text.replace(o, n, 1)
                f.write(text)
            return failures(path)

    def test_mutant_build_error_reported_as_no_adapter(self) -> None:
        # the old behaviour: the build and the run are one step, any failure is "no adapter"
        self.assertNotEqual(self.mutated('failed+=("trt (GPU probe build failed)")',
                                         'failed+=("trt (no GPU adapter)")'), [])

    def test_mutant_build_error_ignored(self) -> None:
        self.assertNotEqual(self.mutated('if ! (cd "$TRT" && cargo build', 'if (cd "$TRT" && cargo build'), [])

    def test_mutant_message_dropped(self) -> None:
        self.assertNotEqual(self.mutated('asked: ${msg:-no compiler output}', 'asked',
                                         '    cat "$build_log" >&2\n', ""), [])

    def test_mutant_colour_not_stripped(self) -> None:
        self.assertNotEqual(self.mutated('s/${esc}\\[[0-9;]*[A-Za-z]//g', 's/NOTHING//g'), [])

    def test_mutant_extraction_fatal(self) -> None:
        self.assertNotEqual(self.mutated("grep -m1 -A1 '^error' | tr '\\n' ' ' || true)",
                                         "grep -m1 -A1 '^error' | tr '\\n' ' ')"), [])

    def test_mutant_no_colour_flag_dropped(self) -> None:
        self.assertNotEqual(self.mutated('cargo build -q --color never', 'cargo build -q'), [])

    def test_mutant_empty_output_passes(self) -> None:
        self.assertNotEqual(self.mutated('&& [ -n "$out" ]', ""), [])

    def test_mutant_build_failure_not_red(self) -> None:
        self.assertNotEqual(self.mutated('failed+=("trt (GPU probe build failed)")\n', ""), [])


if __name__ == "__main__":
    sys.exit(unittest.main())
