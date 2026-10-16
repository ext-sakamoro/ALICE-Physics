#!/usr/bin/env python3
"""Every allow states a reason worth reading, and no wide allow hides a clock read.

Whether an allow has a reason at all is not judged here: Cargo.toml sets
`[lints.clippy] allow_attributes_without_reason = "deny"`, so clippy refuses a
reasonless `allow` in every form (`#[allow]`, `#![allow]`, inside `cfg_attr`,
`expect`, spaced paths, rustc lints). This script checks what the lint does not:

static (no cargo):
  * the manifest keeps that lint at "deny", and no `[lints]` table sets a
    group (`all`, `style`, `pedantic`, `warnings`, …) at all: a group entry
    could lower the level of every lint in it;
  * no allow / expect names a lint group: an allow of `clippy::style` or
    `warnings` silences the libm and clock ban whatever its reason says;
  * every reason is at least MIN_REASON characters, is not a placeholder and
    is not just the name of the lint it allows (an allow whose cfg_attr is off
    in this build is not seen by clippy, so a missing reason is reported too);
  * an allow of `clippy::disallowed_methods` (the platform libm and the wall
    clock, see clippy.toml) starts its reason with one of KINDS and a colon;
  * in src/, an allow of `clippy::disallowed_methods` outside a
    `#[cfg(test)]` module is production code and fails, except the counted
    entries of PRODUCTION (each with its reason).

--clippy (needs cargo, run where clippy runs):
  * a canary crate with this manifest's `[lints]` table gets one
    `allow_attributes_without_reason` diagnostic per reasonless form and none
    on the forms that carry a reason, so a level clippy does not apply fails;
  * clippy (`--force-warn clippy::disallowed_methods`, every target, the full
    native feature set) lists every clock read, aliases and macros included;
    each must be covered only by an allow on its own statement, not by one on
    a whole file, module, function or impl;
  * a run that finds no clock read, or a canary with no diagnostic, fails
    (it would have compared nothing).

    python3 scripts/allow_reason_guard.py            static checks
    python3 scripts/allow_reason_guard.py --clippy   static + canary + clock reads
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("src", "tests", "examples", "benches")
LINT = "clippy::disallowed_methods"
REASON_LINT = "allow_attributes_without_reason"
MIN_REASON = 25
MIN_KIND_REASON = 30
KINDS = (
    "analytic reference",
    "independent reference",
    "timing harness",
    "measurement table",
    "measured quantity",
    "test input",
    "deprecated API",
)
PLACEHOLDERS = {"todo", "fixme", "n/a", "na", "tbd", "allow", "allowed", "oracle", "reference", "-", "x", "needed"}
# (file) -> (count, reason): allows of the ban in non-test src code
PRODUCTION: dict[str, tuple[int, str]] = {
    "src/privacy.rs": (2, "the deprecated clock-seeded XorShift64::from_entropy, documented as not private"),
}
# lint groups: rustc's `warnings` and clippy's groups (an allow of one
# silences every lint in it, the ban included)
GROUPS = {
    "warnings", "clippy", "clippy::all", "clippy::style", "clippy::complexity", "clippy::correctness",
    "clippy::suspicious", "clippy::perf", "clippy::pedantic", "clippy::nursery", "clippy::restriction",
    "clippy::cargo",
}
CLOCK_METHODS = (
    "std::time::Instant::now", "std::time::SystemTime::now",
    "std::time::Instant::elapsed", "std::time::SystemTime::elapsed",
)
FEATURES = "std,simd,parallel,ffi,gpu-solver-bridge,neural,replay,analytics,privacy-crypto"

ATTR = re.compile(r"#\s*(!?)\s*\[")
SILENCE = re.compile(r"\b(?:allow|expect)\(")
REASON = re.compile(r'\breason\s*=\s*"((?:\\.|[^"\\])*)"')
CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test\b|all\s*\(\s*test\b)")


def _strip(text: str, keep_strings: bool) -> str:
    """Comments (line and nested block) replaced by spaces, newlines kept;
    string and char literals kept or blanked. One left-to-right pass, so a
    quote inside a comment or a `//` inside a string is read as what it is."""
    out = list(text)
    n, i = len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        prev_ident = i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")
        raw = re.match(r'b?r(#*)"', text[i:i + 260]) if c in "br" and not prev_ident else None
        if raw:
            close = '"' + raw.group(1)
            j = text.find(close, i + raw.end())
            j = n if j == -1 else j + len(close)
            if not keep_strings:
                blank(i, j)
            i = j
            continue
        if c == '"' or (c == "b" and not prev_ident and text.startswith('b"', i)):
            j = i + (2 if c == "b" else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            if not keep_strings:
                blank(i, j)
            i = j
            continue
        if c == "'":
            ch = re.match(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", text[i:i + 16])
            if ch:
                if not keep_strings:
                    blank(i, i + ch.end())
                i += ch.end()
                continue
        i += 1
    return "".join(out)


def blank_comments(text: str) -> str:
    """Comments replaced by spaces; strings kept (they hold the reasons)."""
    return _strip(text, keep_strings=True)


def blank_all(text: str) -> str:
    """Comments and string / char literals replaced by spaces."""
    return _strip(text, keep_strings=False)


def matching(code: str, i: int, open_c: str, close_c: str) -> int:
    """Index of the bracket closing the one at `i` (code with strings blanked)."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] == open_c:
            depth += 1
        elif code[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    return len(code) - 1


def line_of(code: str, i: int) -> int:
    return code.count("\n", 0, i) + 1


def test_ranges(code: str) -> list[tuple[int, int]]:
    """Character ranges of `#[cfg(test)]` modules (code fully blanked)."""
    out = []
    for m in CFG_TEST.finditer(code):
        rest = code[m.end():]
        mm = re.match(r"[^\n]*\)\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{", rest)
        if mm:
            start = m.end() + mm.end() - 1
            out.append((m.start(), matching(code, start, "{", "}")))
    return out


def scope(code: str, attr_start: int, attr_end: int, inner: bool) -> tuple[int, int, bool]:
    """(start, end, wide) of what the attribute covers: the enclosing block for
    an inner attribute, else the next item; `wide` if that is a file, module,
    function or impl rather than a statement."""
    if inner:
        depth, j = 0, attr_start
        while j > 0:
            j -= 1
            if code[j] == "}":
                depth += 1
            elif code[j] == "{":
                if depth == 0:
                    return j, matching(code, j, "{", "}"), True
                depth -= 1
        return 0, len(code), True
    after = code[attr_end:]
    m = re.match(r"\s*(?:#\s*!?\s*\[[^\]]*\]\s*)*", after)
    head = after[m.end():]
    wide = bool(re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+)*(?:fn|mod|impl|trait)\b", head))
    k = attr_end + m.end()
    brace = code.find("{", k)
    semi = code.find(";", k)
    if brace == -1 or (semi != -1 and semi < brace and not wide):
        return k, semi if semi != -1 else len(code), wide
    return k, matching(code, brace, "{", "}"), wide


def judge_reason(reason: str | None, lints: list[str]) -> str | None:
    """Why `reason` is not acceptable for an allow of `lints`, or None."""
    if reason is None:
        return "no `reason = \"…\"`"
    text = reason.strip()
    low = text.lower().rstrip(".")
    if len(text) < MIN_REASON:
        return f"reason shorter than {MIN_REASON} characters: {text!r}"
    if low in PLACEHOLDERS or low in {l.lower() for l in lints} or low.split("::")[-1] in {
        l.lower().split("::")[-1] for l in lints
    }:
        return f"reason is a placeholder or the lint's own name: {text!r}"
    if LINT not in lints:
        return None
    if len(text) < MIN_KIND_REASON:
        return f"reason for {LINT} shorter than {MIN_KIND_REASON} characters: {text!r}"
    kind = next((k for k in KINDS if low.startswith(k.lower() + ":")), None)
    if kind is None:
        return f"reason for {LINT} does not start with a kind ({', '.join(KINDS)}) and a colon: {text!r}"
    rest = low[len(kind) + 1:].strip()
    if rest in PLACEHOLDERS or len(rest) < 8:
        return f"reason names no concrete reference after the kind: {text!r}"
    return None


def silenced(norm: str) -> list[str]:
    """Lint names of every allow( / expect( in an attribute (whitespace removed)."""
    out = []
    for m in SILENCE.finditer(norm):
        close = matching(norm, m.end() - 1, "(", ")")
        out += [t for t in norm[m.end():close].split(",") if t and "=" not in t]
    return out


def attributes(root: Path) -> list[dict]:
    """Every allow / expect attribute under DIRS, with its lints, reason and scope."""
    found = []
    files = sorted(p for d in DIRS if (root / d).is_dir() for p in (root / d).rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        attr_code = blank_comments(text)
        code = blank_all(text)
        tests = test_ranges(code) if rel.startswith("src/") else []
        for m in ATTR.finditer(code):
            bracket = m.end() - 1
            close = matching(code, bracket, "[", "]")
            lints = silenced(re.sub(r"\s+", "", code[bracket:close + 1]))
            if not lints:
                continue
            r = REASON.search(attr_code[bracket:close + 1])
            s, e, wide = scope(code, m.start(), close + 1, m.group(1) == "!")
            found.append({
                "file": rel, "line": line_of(code, m.start()), "lints": lints,
                "reason": r.group(1) if r else None, "start_line": line_of(code, s),
                "end_line": line_of(code, e), "wide": wide,
                "production": rel.startswith("src/") and not any(a <= m.start() <= b for a, b in tests),
            })
    return found


def check_manifest(manifest: dict) -> list[str]:
    problems = []
    lints = manifest.get("lints", {})
    level = lints.get("clippy", {}).get(REASON_LINT)
    level = level.get("level") if isinstance(level, dict) else level
    if level not in ("deny", "forbid"):
        problems.append(f"Cargo.toml [lints.clippy] {REASON_LINT} is {level!r}, not \"deny\"")
    for tool, table in lints.items():
        for name in table if isinstance(table, dict) else ():
            full = name if tool == "rust" else f"{tool}::{name}"
            if full in GROUPS or name in {g.split("::")[-1] for g in GROUPS if g.startswith(tool + "::")}:
                problems.append(f"Cargo.toml [lints.{tool}] sets the group {name}: list the lints instead")
    return problems


def check(root: Path, production: dict[str, tuple[int, str]] | None = None) -> tuple[list[str], int]:
    """Static problems, and the number of attributes read."""
    production = PRODUCTION if production is None else production
    problems: list[str] = []
    manifest = root / "Cargo.toml"
    if manifest.exists():
        problems += check_manifest(tomllib.loads(manifest.read_text(encoding="utf-8")))
    attrs = attributes(root)
    prod_counts: dict[str, int] = {}
    for a in attrs:
        where = f"{a['file']}:{a['line']}"
        groups = sorted(set(a["lints"]) & GROUPS)
        if groups:
            problems.append(f"{where}: an allow / expect of {', '.join(groups)} also silences {LINT}: "
                            "name the lints it is for instead")
        why = judge_reason(a["reason"], a["lints"])
        if why:
            problems.append(f"{where}: {why}")
        if LINT in a["lints"] and a["production"]:
            prod_counts[a["file"]] = prod_counts.get(a["file"], 0) + 1
    for rel, n in sorted(prod_counts.items()):
        entry = production.get(rel)
        if entry is None:
            problems.append(f"{rel}: {n} allow(s) of {LINT} in production code (outside #[cfg(test)])")
        elif entry[0] != n:
            problems.append(f"{rel}: {n} production allow(s), PRODUCTION says {entry[0]}")
    for rel, (n, why) in sorted(production.items()):
        if not why.strip():
            problems.append(f"PRODUCTION entry {rel} has no reason")
        if rel not in prod_counts:
            problems.append(f"PRODUCTION entry {rel} ({n}) matches nothing: remove it")
    return problems, len(attrs)


def clock_problems(attrs: list[dict], hits: list[tuple[str, int, str]]) -> list[str]:
    """A clock read (file, line, method) inside the scope of a wide allow of the ban."""
    problems = []
    for file, line, method in hits:
        for a in attrs:
            if a["file"] == file and LINT in a["lints"] and a["wide"] and a["start_line"] <= line <= a["end_line"]:
                problems.append(f"{file}:{a['line']}: this allow covers a whole file / module / function and "
                                f"a clock read ({method}, line {line}): give the clock statement an allow of its own")
    return problems


def clock_hits(messages: list[dict]) -> list[tuple[str, int, str]]:
    """(file, line, method) of every clock read in clippy's JSON messages."""
    out = set()
    for m in messages:
        if m.get("reason") != "compiler-message":
            continue
        msg = m["message"]
        if (msg.get("code") or {}).get("code") != LINT:
            continue
        method = next((c for c in CLOCK_METHODS if f"`{c}`" in msg["message"]), None)
        if method is None:
            continue
        for sp in msg["spans"]:
            if not sp.get("is_primary"):
                continue
            sp = call_site(sp)
            if not sp["file_name"].startswith("/"):
                out.add((sp["file_name"], sp["line_start"], method))
    return sorted(out)


def call_site(span: dict) -> dict:
    """The outermost call site of a span produced by a macro expansion.

    A clock read written in a macro body is reported at the body (the macro
    definition), but the lint level that silences it is the one at the place the
    macro is called; follow `expansion.span` out to that place."""
    while span.get("expansion") and span["expansion"].get("span"):
        span = span["expansion"]["span"]
    return span


def clock_check(root: Path, args: list[str]) -> tuple[list[str], list[tuple[str, int, str]]]:
    """Problems and clock reads of `cargo clippy <args>` in `root`."""
    rc, msgs = run_json(["cargo", "clippy", *args, "--message-format=json", "--", "--force-warn", LINT], root)
    problems = [] if rc == 0 else [f"clippy for the clock reads exited {rc}"]
    hits = clock_hits(msgs)
    if not hits:
        problems.append("clippy reported no clock read (compared nothing)")
    return problems + clock_problems(attributes(root), hits), hits


# the clock canary: each function named wide_* reads the clock under a function
# allow and must be reported; `statement` reads it under an allow on the statement
CLOCK_CANARY = """\
#![allow(missing_docs, reason = "canary crate, not documented anywhere")]
use std::time::Instant as Clk;
macro_rules! clk {
    () => {
        std::time::Instant::now()
    };
}
#[allow(clippy::disallowed_methods, reason = "timing harness: canary, a function allow over a direct read")]
pub fn wide_direct() -> std::time::Instant {
    std::time::Instant::now()
}
#[allow(clippy::disallowed_methods, reason = "timing harness: canary, a function allow over an aliased read")]
pub fn wide_alias() -> Clk {
    Clk::now()
}
#[allow(clippy::disallowed_methods, reason = "timing harness: canary, a function allow over a macro read")]
pub fn wide_macro() -> std::time::Instant {
    clk!()
}
pub fn statement() -> std::time::Instant {
    #[allow(clippy::disallowed_methods, reason = "timing harness: canary, an allow on the read itself")]
    let t = std::time::Instant::now();
    t
}
"""


def judge_clock_canary(problems: list[str], hits: list[tuple[str, int, str]], source: str) -> list[str]:
    """Errors: every wide_* allow reported once, nothing else, and a read per function."""
    lines = source.split("\n")
    wide = {i for i, l in enumerate(lines, 1) if l.startswith("#[allow(clippy::disallowed_methods")
            and lines[i].startswith("pub fn wide_")}
    flagged = set()
    errors = []
    for p in problems:
        m = re.match(r"src/lib\.rs:(\d+): this allow covers", p)
        if m:
            flagged.add(int(m.group(1)))
        else:
            errors.append(f"clock canary: unexpected problem: {p}")
    for line in sorted(wide - flagged):
        errors.append(f"clock canary: the function allow on line {line} covers a clock read but was not reported")
    for line in sorted(flagged - wide):
        errors.append(f"clock canary: the allow on line {line} was reported but covers no read outside its statement")
    if len(hits) != len(wide) + 1:
        errors.append(f"clock canary: {len(hits)} clock reads, expected {len(wide) + 1}")
    if not wide:
        errors.append("clock canary: no function allow in the source (compared nothing)")
    return errors


def run_clock_canary(root: Path) -> list[str]:
    """The clock check on a crate with this repository's clippy.toml."""
    tmp = Path(tempfile.mkdtemp(prefix="allow-reason-clock-canary-"))
    (tmp / "src").mkdir()
    (tmp / "Cargo.toml").write_text('[package]\nname = "allow-reason-clock-canary"\nversion = "0.0.0"\n'
                                    'edition = "2021"\npublish = false\n\n[workspace]\n', encoding="utf-8")
    (tmp / "clippy.toml").write_text((root / "clippy.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp / "src" / "lib.rs").write_text(CLOCK_CANARY, encoding="utf-8")
    target = os.environ.get("CARGO_TARGET_DIR") or str(Path(tempfile.gettempdir()) / "allow-reason-canary-target")
    found, hits = clock_check(tmp, ["--target-dir", str(Path(target) / "allow-reason-clock-canary")])
    return judge_clock_canary(found, hits, CLOCK_CANARY)


def run_json(cmd: list[str], cwd: Path) -> tuple[int, list[dict]]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    msgs = []
    for line in r.stdout.splitlines():
        try:
            msgs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return r.returncode, msgs


# the canary: each FORM line must get an allow_attributes_without_reason
# diagnostic, each CLEAN line none
CANARY_FORMS = [
    "#[allow(dead_code)] fn form_plain() {}",
    "#[cfg_attr(all(), allow(dead_code))] fn form_cfg_attr() {}",
    "#[expect(dead_code)] fn form_expect() {}",
    "#[allow(clippy :: needless_range_loop)] fn form_spaced() {}",
    "#[allow(clippy::style)] fn form_group() {}",
    "#[allow(deprecated)] fn form_rustc() {}",
    "mod form_inner { #![allow(dead_code)] fn f() {} }",
]
CANARY_CLEAN = [
    '#[allow(dead_code, reason = "canary: an allow that states its reason passes")] fn clean_plain() {}',
    '#[expect(dead_code, reason = "canary: an expect that states its reason passes")] fn clean_expect() {}',
]


def canary_source() -> str:
    return "\n".join(["#![allow(missing_docs, reason = \"canary crate, not documented\")]"]
                     + CANARY_FORMS + CANARY_CLEAN) + "\n"


def lints_tables(manifest: str) -> str:
    """The `[lints]` / `[lints.*]` tables of a manifest, verbatim."""
    out, keep = [], False
    for line in manifest.split("\n"):
        head = re.match(r"\s*\[([^\]]+)\]\s*$", line)
        if head:
            keep = head.group(1).strip() == "lints" or head.group(1).strip().startswith("lints.")
        if keep:
            out.append(line)
    return "\n".join(out)


def run_canary(root: Path) -> list[str]:
    """Clippy on a crate that has this manifest's [lints] table."""
    lines = ['[package]', 'name = "allow-reason-canary"', 'version = "0.0.0"', 'edition = "2021"',
             'publish = false', "", "[workspace]", "", lints_tables((root / "Cargo.toml").read_text(encoding="utf-8"))]
    tmp = Path(tempfile.mkdtemp(prefix="allow-reason-canary-"))
    (tmp / "src").mkdir()
    (tmp / "Cargo.toml").write_text("\n".join(lines), encoding="utf-8")
    source = canary_source()
    (tmp / "src" / "lib.rs").write_text(source, encoding="utf-8")
    target = os.environ.get("CARGO_TARGET_DIR") or str(Path(tempfile.gettempdir()) / "allow-reason-canary-target")
    rc, msgs = run_json(["cargo", "clippy", "--quiet", "--message-format=json",
                         "--target-dir", str(Path(target) / "allow-reason-canary")], tmp)
    return judge_canary(rc, msgs, source)


def judge_canary(rc: int, msgs: list[dict], source: str) -> list[str]:
    src = source.split("\n")
    flagged = set()
    for m in msgs:
        if m.get("reason") != "compiler-message":
            continue
        code = (m["message"].get("code") or {}).get("code", "")
        if REASON_LINT in code:
            for sp in m["message"]["spans"]:
                if sp.get("is_primary"):
                    flagged.add(sp["line_start"])
    errors = []
    if rc == 0:
        errors.append("clippy exited 0 on the canary: the manifest does not deny reasonless allows")
    for form in CANARY_FORMS:
        if src.index(form) + 1 not in flagged:
            errors.append(f"canary: no {REASON_LINT} diagnostic on `{form}`")
    for form in CANARY_CLEAN:
        if src.index(form) + 1 in flagged:
            errors.append(f"canary: `{form}` states a reason but was flagged")
    if not flagged:
        errors.append("canary: no diagnostic at all (compared nothing)")
    return errors


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, attrs = check(ROOT)
    summary = f"allow_reason_guard: {attrs} allow / expect attributes"
    if attrs == 0:
        problems.append("compared nothing (no allow / expect attribute under src/ tests/ examples/ benches/)")
    if "--clippy" in argv:
        problems += run_canary(ROOT)
        problems += run_clock_canary(ROOT)
        found, hits = clock_check(ROOT, ["--all-targets", "--features", FEATURES])
        problems += found
        summary += f", canaries run, {len(hits)} clock reads"
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    print(f"{summary}, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
