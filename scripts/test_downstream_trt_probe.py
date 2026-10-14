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

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "downstream_trt_probe.sh")


def posix(path: str) -> str:
    """A path in the form bash accepts on every platform (Git for Windows bash takes C:/x)."""
    return path.replace("\\", "/")


def usable_bash(candidate: str) -> bool:
    """True when `candidate` is a bash that runs a script and prints plain text.

    Judged on the raw bytes: the WSL launcher that Windows puts first on PATH answers in
    UTF-16 ("W\\0i\\0n\\0...") or fails, and must not be taken for a shell."""
    try:
        p = subprocess.run([candidate, "-c", "echo probe-ok"], capture_output=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return plain_ok(p.returncode, p.stdout)


def plain_ok(returncode: int, stdout: bytes) -> bool:
    return returncode == 0 and stdout.strip() == b"probe-ok"


def bash_candidates() -> list[str]:
    found: list[str] = []
    if os.environ.get("ALICE_TEST_BASH"):
        found.append(os.environ["ALICE_TEST_BASH"])
    if os.name == "nt":
        # Git for Windows first: <git root>/bin/bash.exe, from `git --exec-path` (<root>/mingw64/libexec/git-core)
        git = shutil.which("git")
        roots = []
        if git:
            roots.append(os.path.dirname(os.path.dirname(git)))
            try:
                ep = subprocess.run([git, "--exec-path"], capture_output=True, text=True, check=False).stdout.strip()
                if ep:
                    roots += [os.path.normpath(os.path.join(ep, *[".."] * n)) for n in (2, 3)]
            except OSError:
                pass
        roots += [os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Git")]
        for r in roots:
            found += [os.path.join(r, "bin", "bash.exe"), os.path.join(r, "usr", "bin", "bash.exe")]
    which = shutil.which("bash")
    if which:
        found.append(which)
    return [c for c in dict.fromkeys(found) if os.path.exists(c) or c == which]


def resolve_bash(candidates: list[str] | None = None) -> str:
    """The first usable bash; raises (the tests fail) when there is none, never skips."""
    cands = bash_candidates() if candidates is None else candidates
    for c in cands:
        if usable_bash(c):
            return c
    raise RuntimeError(f"no usable bash among {cands!r}; set ALICE_TEST_BASH")


_TMP = tempfile.mkdtemp(prefix="trt-probe-test-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


def lf_copy(path: str) -> str:
    """The script with LF line endings (a Windows checkout may have CRLF, which bash reads as part of every word)."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read().replace("\r\n", "\n")
    out = os.path.join(_TMP, f"{len(os.listdir(_TMP))}-{os.path.basename(path)}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out

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
            PROBE=posix(lf_copy(script)),
            ROOT=posix(root),
            TRT=posix(os.path.join(root, "ALICE-TRT")),
            FAKE_BUILD=build,
            FAKE_RUN=run,
            FAKE_BUILD_ERR=build_err,
        )
        env.pop("TRT_GPU_PROBE", None)
        if probe_env is not None:
            env["TRT_GPU_PROBE"] = probe_env
        p = subprocess.run([resolve_bash(), "-c", DRIVER], env=env, capture_output=True, check=False)
        return p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


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
    def test_bash_resolves(self) -> None:
        self.assertTrue(usable_bash(resolve_bash()))

    def test_resolver_rejects_utf16_launcher(self) -> None:
        # what the WSL launcher prints, and an exit-0 answer with other text
        wsl = "Windows Subsystem for Linux has no installed distributions.".encode("utf-16-le")
        self.assertFalse(plain_ok(0, wsl))
        self.assertFalse(plain_ok(0, "p\0r\0o\0b\0e\0-\0o\0k\0\n\0".encode()))
        self.assertFalse(plain_ok(1, b"probe-ok\n"))
        self.assertTrue(plain_ok(0, b"probe-ok\r\n"))
        # and a stand-in executable that is not a shell is not taken for one
        self.assertFalse(usable_bash(os.path.join(_TMP, "does-not-exist")))
        with self.assertRaises(RuntimeError):
            resolve_bash([os.path.join(_TMP, "does-not-exist")])

    def test_no_bash_fails_loudly(self) -> None:
        with self.assertRaises(RuntimeError):
            resolve_bash([])

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
