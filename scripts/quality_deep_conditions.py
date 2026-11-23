#!/usr/bin/env python3
"""Which jobs of quality-deep.yml run for which trigger.

The workflow serves four kinds of run: the nightly schedule, the weekly
schedule, a plain dispatch and a scoped dispatch (inputs.scope set). Adding
inputs for the scoped kind must not change what the other kinds run, and a
scoped run must run the scoped jobs only.

Two rules, both read from the job-level `if` and `needs` of the file:

1. For every unscoped trigger, each job runs exactly when it ran under the
   conditions pinned in BEFORE (the conditions before the scoped inputs grew).
   A job that BEFORE does not know must be listed in SCOPED_ONLY and must be
   skipped.
2. For a scoped dispatch, the jobs that run are exactly SCOPED.

Zero comparisons is a failure.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ".github/workflows/quality-deep.yml"

NIGHTLY = "30 17 * * 1-5"
WEEKLY = "30 17 * * 6"
_GATED = "inputs.diff_base == '' && (github.event.schedule != '30 17 * * 1-5')"
_GATED_NO_PR = ("inputs.diff_base == '' && (github.event_name != 'pull_request' && "
                "github.event.schedule != '30 17 * * 1-5')")
_WEEKLY_ONLY = ("inputs.diff_base == '' && (github.event.schedule == '30 17 * * 6' || "
                "(github.event_name == 'workflow_dispatch' && inputs.scope == ''))")

# job -> (if, needs) before the scoped inputs grew
BEFORE: dict[str, tuple[str, tuple[str, ...]]] = {
    "mutants-scoped": ("github.event_name == 'workflow_dispatch' && inputs.scope != ''", ()),
    "mutants-plan": ("github.event_name != 'workflow_dispatch' || inputs.scope == ''", ()),
    "mutants-accounted-age": ("needs.mutants-plan.outputs.mode == 'weekly'", ("mutants-plan",)),
    "mutants": ("needs.mutants-plan.outputs.jobs != '0'", ("mutants-plan",)),
    "mutants-ratchet": ("always() && " + _WEEKLY_ONLY, ("mutants",)),
    "loom": (_GATED, ()),
    "miri": (_GATED, ()),
    "ignored-tests": (_GATED_NO_PR, ()),
    "opt0-suite": (_GATED_NO_PR, ()),
    "laminar-plume": (_WEEKLY_ONLY, ()),
    "bench-regress": (_GATED, ()),
    "coverage-full": (_WEEKLY_ONLY, ()),
}
SCOPED_ONLY = frozenset({"scoped-shards"})           # jobs BEFORE does not know
SCOPED = frozenset({"scoped-shards", "mutants-scoped"})  # what a scoped dispatch runs


class Unreadable(Exception):
    """An expression this reader does not understand (never guessed)."""


_TOKEN = re.compile(r"\s*(?:(&&|\|\||==|!=|!|\(|\))|'((?:[^']|'')*)'|([A-Za-z_][A-Za-z0-9_.\-]*))")


def tokens(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    expr = expr.strip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            raise Unreadable(f"cannot read {expr[pos:]!r}")
        if m.group(1):
            out.append(("op", m.group(1)))
        elif m.group(2) is not None:
            out.append(("str", m.group(2).replace("''", "'")))
        else:
            out.append(("name", m.group(3)))
        pos = m.end()
    return out


def truthy(v) -> bool:
    return bool(v)


def same(a, b) -> bool:
    """GitHub's loose equality for the two types used here (string, bool); a missing value is ''."""
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    raise Unreadable(f"comparison of {type(a).__name__} with {type(b).__name__}")


class Parser:
    def __init__(self, toks: list[tuple[str, str]], ctx: dict[str, str]):
        self.t, self.i, self.ctx = toks, 0, ctx

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        v = self.conj()
        while self.peek() == ("op", "||"):
            self.take()
            r = self.conj()
            v = v if truthy(v) else r
        return v

    def conj(self):
        v = self.cmp()
        while self.peek() == ("op", "&&"):
            self.take()
            r = self.cmp()
            v = r if truthy(v) else v
        return v

    def cmp(self):
        v = self.unary()
        kind, op = self.peek()
        if kind == "op" and op in ("==", "!="):
            self.take()
            r = self.unary()
            return same(v, r) if op == "==" else not same(v, r)
        return v

    def unary(self):
        kind, val = self.take()
        if kind == "op" and val == "!":
            return not truthy(self.unary())
        if kind == "op" and val == "(":
            v = self.expr()
            if self.take() != ("op", ")"):
                raise Unreadable("unbalanced parenthesis")
            return v
        if kind == "str":
            return val
        if kind == "name":
            if self.peek() == ("op", "("):
                if val != "always":
                    raise Unreadable(f"function {val}()")
                self.take()
                if self.take() != ("op", ")"):
                    raise Unreadable("always() takes no argument")
                return True
            head = val.split(".", 1)[0]
            if head not in ("github", "inputs", "needs"):
                raise Unreadable(f"context {val}")
            return self.ctx.get(val, "")
        raise Unreadable(f"unexpected {val!r}")


def evaluate(expr: str, ctx: dict[str, str]) -> bool:
    p = Parser(tokens(expr), ctx)
    v = p.expr()
    if p.i != len(p.t):
        raise Unreadable(f"trailing {p.t[p.i:]}")
    return truthy(v)


def strip_comment(value: str) -> str:
    """Drop a trailing YAML comment (a # outside quotes, after a space) and a ${{ }} wrapper."""
    quoted = False
    for i, ch in enumerate(value):
        if ch == "'":
            quoted = not quoted
        elif ch == "#" and not quoted and i and value[i - 1].isspace():
            value = value[:i]
            break
    value = value.strip()
    m = re.fullmatch(r"\$\{\{(.*)\}\}", value, re.S)
    return (m.group(1) if m else value).strip()


def job_table(text: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    """job -> (if, needs) of the workflow text; a job with no `if` has ''."""
    m = re.search(r"^jobs:\s*$", text, re.M)
    if not m:
        raise Unreadable("no jobs: block")
    out: dict[str, list] = {}
    cur = None
    for line in text[m.end():].splitlines():
        if re.match(r"^\S", line):
            break
        j = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", line)
        if j:
            cur = j.group(1)
            out[cur] = ["", ()]
            continue
        if cur is None:
            continue
        c = re.match(r"^    if:\s*(.+)$", line)
        if c:
            out[cur][0] = strip_comment(c.group(1))
        n = re.match(r"^    needs:\s*(.+)$", line)
        if n:
            raw = strip_comment(n.group(1)).strip("[]")
            out[cur][1] = tuple(x.strip() for x in raw.split(",") if x.strip())
    return {k: (v[0], v[1]) for k, v in out.items()}


def ran(table: dict[str, tuple[str, tuple[str, ...]]], ctx: dict[str, str]) -> dict[str, bool]:
    """job -> runs under ctx. A job whose needed job is skipped is skipped unless its `if` calls always()."""
    done: dict[str, bool] = {}

    def run(job: str) -> bool:
        if job not in done:
            cond, needs = table[job]
            for n in needs:
                if n not in table:
                    raise Unreadable(f"{job} needs the unknown job {n}")
            held = all(run(n) for n in needs) or "always()" in cond
            done[job] = held and (evaluate(cond, ctx) if cond else True)
        return done[job]

    for job in table:
        run(job)
    return done


def _ctx(event: str, schedule: str = "", **inputs: str) -> dict[str, str]:
    ctx = {"github.event_name": event, "github.event.schedule": schedule}
    ctx.update({f"inputs.{k}": v for k, v in inputs.items()})
    return ctx


_PLAIN = {"scope": "", "shards": "", "features": "", "test_tool": "", "test_filter": "", "label": ""}
UNSCOPED = {
    "nightly schedule": _ctx("schedule", NIGHTLY),
    "weekly schedule": _ctx("schedule", WEEKLY),
    "plain dispatch": _ctx("workflow_dispatch", **_PLAIN, diff_base=""),
    "dispatch with a diff base": _ctx("workflow_dispatch", **_PLAIN, diff_base="0123abc"),
    "pull request": _ctx("pull_request"),
}
SCOPED_RUNS = {
    "scoped dispatch": _ctx("workflow_dispatch", **{**_PLAIN, "scope": "--file src/x.rs"}, diff_base=""),
    "scoped dispatch in shards": _ctx("workflow_dispatch", **{**_PLAIN, "scope": "--file src/x.rs", "shards": "8",
                                                              "features": "default", "test_tool": "nextest"},
                                      diff_base=""),
}
# the plan job's outputs are not known from the file: every combination is compared
PLAN_OUTPUTS = [{"needs.mutants-plan.outputs.mode": m, "needs.mutants-plan.outputs.jobs": j}
                for m, j in itertools.product(("weekly", "in-diff"), ("0", "4"))]


def check(text: str) -> tuple[list[str], int]:
    """(errors, comparisons made)."""
    try:
        now = job_table(text)
    except Unreadable as e:
        return [f"{WORKFLOW}: {e}"], 0
    errors: list[str] = []
    compared = 0
    for job in BEFORE:
        if job not in now:
            errors.append(f"{WORKFLOW}: job {job} is gone (BEFORE still pins it)")
    for job in now:
        if job not in BEFORE and job not in SCOPED_ONLY:
            errors.append(f"{WORKFLOW}: job {job} is neither pinned in BEFORE nor listed in SCOPED_ONLY")
    if errors:
        return errors, 0
    try:
        for (name, base), plan in itertools.product(UNSCOPED.items(), PLAN_OUTPUTS):
            ctx = {**base, **plan}
            was, is_ = ran(BEFORE, ctx), ran(now, ctx)
            for job in now:
                want = was[job] if job in BEFORE else False
                compared += 1
                if is_[job] != want:
                    errors.append(f"{WORKFLOW}: {name}: job {job} {'runs' if is_[job] else 'is skipped'}, "
                                  f"expected {'run' if want else 'skip'} ({sorted(plan.values())})")
        for (name, base), plan in itertools.product(SCOPED_RUNS.items(), PLAN_OUTPUTS):
            is_ = ran(now, {**base, **plan})
            for job in now:
                compared += 1
                if is_[job] != (job in SCOPED):
                    errors.append(f"{WORKFLOW}: {name}: job {job} {'runs' if is_[job] else 'is skipped'}, "
                                  f"a scoped run is {sorted(SCOPED)} only")
    except Unreadable as e:
        return [f"{WORKFLOW}: {e}"], compared
    if compared == 0:
        errors.append(f"{WORKFLOW}: 0 job conditions compared")
    return sorted(set(errors)), compared


def main(argv: list[str] | None = None) -> int:
    root = Path((argv or sys.argv[1:] or [str(ROOT)])[0])
    path = root / WORKFLOW
    if not path.is_file():
        print(f"quality_deep_conditions: FAIL {WORKFLOW} is missing")
        return 1
    errors, compared = check(path.read_text(encoding="utf-8"))
    for e in errors:
        print(f"quality_deep_conditions: {e}")
    if errors:
        print(f"quality_deep_conditions: FAIL {len(errors)} finding(s), {compared} comparison(s)")
        return 1
    print(f"quality_deep_conditions: OK {compared} job condition(s) compared over "
          f"{len(UNSCOPED)} unscoped and {len(SCOPED_RUNS)} scoped trigger(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
