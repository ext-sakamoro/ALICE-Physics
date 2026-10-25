#!/usr/bin/env python3
"""docs/CI_TRIGGERS.md and .github/ci-evidence.toml agree with the workflows.

For every workflow in .github/workflows the triggers are read from its `on:` block
(as text, no YAML library on the runners) and compared with its row in the table of
docs/CI_TRIGGERS.md; the evidence class of the row is compared with
.github/ci-evidence.toml. The checks:

  * the workflows, the rows of the table and the classes of the evidence file name
    the same set of workflows;
  * every cell of a row equals what the workflow file says (push branches or tags,
    path gate, pull_request, schedule, dispatch, behaviour on a ci/** push, class);
  * the class fits the triggers: landing and after-landing run on ci/**, schedule-only
    has no push trigger, tag-only is pushed by tags only;
  * every declared leg (a matrix value of a job of a landing workflow) names the events it
    runs on, and they equal the events for which the job's matrix (a literal list, or a
    `fromJSON(...)` expression of github.event_name and github.ref, evaluated here) holds
    the value; a leg that runs on a main push but not on a ci/** push must be declared;
    the second table of the document lists the same legs;
  * every landing and after-landing workflow has `workflow_dispatch` (a run dropped by
    the one-pending-run-per-group rule is re-created on the ci/<id> SHA with
    `gh workflow run`), and the dispatchable list of the document names exactly the
    workflows that have it;
  * a path-gated workflow has the guard's `gate` job (`changed_paths.py gate`, `if:
    always()`, needs every other job), and a workflow without a path gate has none (ci.yml's
    `gate` job is the nightly Mac Intel decision, a different job);
  * the landing-waits table equals what scripts/land.py derives and waits on, and the
    after-landing-wait table equals what `land.py --wait` reads on main, and the document
    names every ledger land.py counts as a document when main moves;
  * a workflow that runs on ci/** cannot cancel a main run: its concurrency group
    includes the ref, or `cancel-in-progress` is false.

Fails (exit 1) on a finding, and when no row was compared.

  python3 scripts/ci_triggers_check.py            # exit 1 on a finding
  python3 scripts/ci_triggers_check.py --root DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    print("error: ci_triggers_check.py needs Python 3.11 or newer (tomllib)", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths as cp  # noqa: E402
from ci_load_check import jobs, scalar, top_block  # noqa: E402

DOC = "docs/CI_TRIGGERS.md"
EVIDENCE = ".github/ci-evidence.toml"
CLASSES = ("landing", "after-landing", "schedule-only", "tag-only")
COLUMNS = ("workflow", "push", "path gate", "pull_request", "schedule", "dispatch", "ci/** push", "class")
LEG_COLUMNS = ("workflow", "job", "matrix", "class", "events")
WF_TABLE = "ci-triggers:workflows"
LEG_TABLE = "ci-triggers:after-landing-legs"
# the events a matrix leg is evaluated for: (label, github.event_name, github.ref, trigger key)
EVENTS = (
    ("push main", "push", "refs/heads/main", "push"),
    ("push ci/**", "push", "refs/heads/ci/x", "push"),
    ("pull_request", "pull_request", "refs/pull/1/merge", "pull_request"),
    ("workflow_dispatch", "workflow_dispatch", "refs/heads/ci/x", "workflow_dispatch"),
    ("schedule", "schedule", "refs/heads/main", "schedule"),
)
# leg classes: started on the ci/<id> SHA and read after the landing, or not started on
# ci/** at all (main push and schedule only) and read on main after the landing
LEG_CLASSES = ("after-landing", "main-after-landing")
# job outputs a matrix expression may read: each is evaluated as 'false' for the leg's
# `events` and as 'true' for its own key (a leg the output adds to an event declares the
# events it then runs on; an output not listed here makes the matrix unknown before the run)
CONDITION_OUTPUTS = {"needs.gate.outputs.landing_scripts": "events_when_landing_scripts"}


def event_ctx(ev: str, ref: str, true_output: str | None = None) -> dict[str, str]:
    ctx = {"github.event_name": ev, "github.ref": ref}
    ctx.update({o: "true" if o == true_output else "false" for o in CONDITION_OUTPUTS})
    return ctx
DISPATCH_TABLE = "ci-triggers:dispatchable"
DISPATCH_COLUMNS = ("workflow",)
WAIT_TABLE = "ci-triggers:landing-waits"
WAIT_COLUMNS = ("workflow", "class", "waits for")
AFTER_TABLE = "ci-triggers:after-landing-wait"
AFTER_COLUMNS = ("item", "class")


class ExprError(ValueError):
    pass


def _tokens(s: str) -> list[str]:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "'":
            j = s.find("'", i + 1)
            while j != -1 and s[j + 1:j + 2] == "'":  # '' is an escaped quote
                j = s.find("'", j + 2)
            if j == -1:
                raise ExprError(f"unterminated string in {s!r}")
            out.append(s[i:j + 1])
            i = j + 1
        elif s.startswith(("&&", "||", "==", "!="), i):
            out.append(s[i:i + 2])
            i += 2
        elif c in "()!,":
            out.append(c)
            i += 1
        else:
            m = re.match(r"[A-Za-z_][A-Za-z0-9_.\-]*", s[i:])
            if not m:
                raise ExprError(f"unexpected {c!r} in {s!r}")
            out.append(m.group(0))
            i += len(m.group(0))
    return out


def evaluate(expr: str, ctx: dict[str, str]):
    """a GitHub expression of `&&`, `||`, `==`, `!=`, `!`, parentheses, string literals,
    `fromJSON(...)` and the context names in ctx (anything else is an error, so a matrix the
    evaluator cannot read is reported instead of guessed)"""
    toks = _tokens(expr)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(want=None):
        nonlocal pos
        tok = peek()
        if tok is None or (want is not None and tok != want):
            raise ExprError(f"expected {want or 'a token'} in {expr!r}")
        pos += 1
        return tok

    def primary():
        tok = take()
        if tok == "(":
            v = or_()
            take(")")
            return v
        if tok == "!":
            return not primary()
        if tok.startswith("'"):
            return tok[1:-1].replace("''", "'")
        if tok == "fromJSON":
            take("(")
            v = or_()
            take(")")
            try:
                return json.loads(v)
            except (TypeError, ValueError) as e:
                raise ExprError(f"fromJSON of {v!r}: {e}") from e
        if tok in ("true", "false"):
            return tok == "true"
        if tok in ctx:
            return ctx[tok]
        raise ExprError(f"unknown name {tok!r} in {expr!r}")

    def cmp_():
        v = primary()
        while peek() in ("==", "!="):
            op = take()
            w = primary()
            v = (v == w) if op == "==" else (v != w)
        return v

    def and_():
        v = cmp_()
        while peek() == "&&":
            take()
            w = cmp_()
            v = w if v else v
        return v

    def or_():
        v = and_()
        while peek() == "||":
            take()
            w = and_()
            v = v if v else w
        return v

    out = or_()
    if pos != len(toks):
        raise ExprError(f"trailing {toks[pos]!r} in {expr!r}")
    return out


def matrix_values(job: str, key: str, ctx: dict[str, str]) -> list[str] | None:
    """the values of matrix `key` in a job block for an event (None: the key is absent)"""
    m = re.search(rf"^\s+{re.escape(key)}:\s*(.+?)\s*$", job.split("matrix:", 1)[1] if "matrix:" in job else "", re.M)
    if not m:
        return None
    v = m.group(1)
    if v.startswith("${{") and v.endswith("}}"):
        out = evaluate(v[3:-2], ctx)
    elif v.startswith("["):
        out = [x.strip().strip("'\"") for x in v.strip("[]").split(",") if x.strip()]
    else:
        raise ExprError(f"matrix {key}: cannot read {v!r}")
    if not isinstance(out, list):
        raise ExprError(f"matrix {key}: {v!r} is not a list")
    return [str(x) for x in out]


def leg_events(text: str, job: str, key: str, value: str,
               true_output: str | None = None) -> tuple[list[str], list[str]]:
    """(events whose run includes the leg, events the workflow is triggered by); every
    CONDITION_OUTPUTS output is 'false' except `true_output`"""
    on = cp.triggers(cp.load_yaml(text))
    triggered = [label for label, _ev, _ref, trig in EVENTS if trig in on]
    present = []
    for label, ev, ref, _trig in EVENTS:
        if label not in triggered:
            continue
        vals = matrix_values(job, key, event_ctx(ev, ref, true_output))
        if vals is not None and value in vals:
            present.append(label)
    return present, triggered


def derive(name: str, text: str) -> dict[str, str]:
    """the cells of a workflow's row, from its file"""
    on = cp.triggers(cp.load_yaml(text))
    push = on.get("push") if "push" in on else None
    if "push" not in on:
        push_cell, on_ci = "-", False
    elif isinstance(push, dict) and push.get("branches") is not None:
        branches = cp.name_list(push["branches"], f"{name}: on.push.branches")
        push_cell, on_ci = ", ".join(branches), "ci/**" in branches
    elif isinstance(push, dict) and push.get("tags") is not None:
        tags = cp.name_list(push["tags"], f"{name}: on.push.tags")
        push_cell, on_ci = "tags " + ", ".join(tags), False
    else:
        push_cell, on_ci = "every branch", True
    rel = f".github/paths/{name.rsplit('.', 1)[0]}.txt"
    gate = rel if f"--patterns-file {rel}" in text else "-"
    if "pull_request" not in on:
        pr_cell = "-"
    else:
        pr = on.get("pull_request")
        pr = pr if isinstance(pr, dict) else {}
        pr_branches = cp.name_list(pr["branches"], f"{name}: on.pull_request.branches") if pr.get("branches") is not None else []
        pr_cell = ", ".join(pr_branches) or "every branch"
        if pr.get("paths") is not None:
            pr_cell += " (paths)"
    has = lambda key: key in on  # noqa: E731
    ci_cell = "-" if not on_ci else ("runs when the paths match" if gate != "-" else "runs")
    return {
        "workflow": name,
        "push": push_cell,
        "path gate": gate,
        "pull_request": pr_cell,
        "schedule": "yes" if has("schedule") else "-",
        "dispatch": "yes" if has("workflow_dispatch") else "-",
        "ci/** push": ci_cell,
    }


def table(doc: str, marker: str, columns: tuple[str, ...]) -> tuple[list[dict[str, str]], list[str]]:
    """the rows of the table between `<!-- marker:begin -->` and `<!-- marker:end -->`"""
    m = re.search(rf"<!-- {re.escape(marker)}:begin -->\n(.*?)<!-- {re.escape(marker)}:end -->", doc, re.S)
    if not m:
        return [], [f"{DOC}: no table between <!-- {marker}:begin --> and <!-- {marker}:end -->"]
    lines = [l for l in m.group(1).splitlines() if l.strip().startswith("|")]
    errors: list[str] = []
    if len(lines) < 2:
        return [], [f"{DOC}: table {marker} has no header"]
    head = [c.strip().lower() for c in lines[0].strip().strip("|").split("|")]
    if tuple(head) != columns:
        return [], [f"{DOC}: table {marker} columns {head} != {list(columns)}"]
    rows = []
    for l in lines[2:]:
        cells = [c.strip().replace("`", "") for c in l.strip().strip("|").split("|")]
        if len(cells) != len(columns):
            errors.append(f"{DOC}: row {l.strip()!r} has {len(cells)} cells, expected {len(columns)}")
            continue
        rows.append(dict(zip(columns, cells)))
    return rows, errors


def guard_gate_errors(name: str, text: str, gated: bool) -> list[str]:
    """the changes-guard `gate` job (name `Gate`) of a path-gated workflow: it runs
    `changed_paths.py gate`, always, and needs `changes` and every other job; ci.yml's own
    `gate` job (the nightly Mac Intel decision) is a different job and must not be read as it"""
    js = jobs(text)
    g = js.get("gate")
    runs_guard = g is not None and "changed_paths.py gate" in g
    if not gated:
        if runs_guard:
            return [f"{name}: a `gate` job runs changed_paths.py gate, but the workflow has no path gate"]
        return []
    if g is None or not runs_guard:
        return [f"{name}: path-gated, but no `gate` job running scripts/changed_paths.py gate"]
    errors = []
    if not re.search(r"^\s+if:\s*always\(\)\s*$", g, re.M):
        errors.append(f"{name}: the guard's gate job is not `if: always()`")
    m = re.search(r"^\s+needs:\s*\[(.*)\]\s*$", g, re.M)
    needs = {x.strip() for x in m.group(1).split(",")} if m else set()
    missing = sorted(set(js) - {"gate"} - needs)
    if missing:
        errors.append(f"{name}: the guard's gate job does not need {missing}")
    return errors


def concurrency_errors(name: str, text: str) -> list[str]:
    conc = top_block(text, "concurrency")
    group, cancel = scalar(conc, "group"), scalar(conc, "cancel-in-progress")
    if group is None or "github.ref" in group or cancel in (None, "false"):
        return []
    return [f"{name}: runs on ci/** with concurrency group {group} shared by every ref and "
            f"cancel-in-progress {cancel}: a ci/** push can cancel the main run"]


def check(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    wf_dir = root / ".github" / "workflows"
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(wf_dir.glob("*.yml"))}
    try:
        ev = tomllib.loads((root / EVIDENCE).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"{EVIDENCE}: {e}"], 0
    classes = ev.get("workflows", {})
    legs = ev.get("legs", [])
    try:
        doc = (root / DOC).read_text(encoding="utf-8")
    except OSError as e:
        return [f"{DOC}: {e}"], 0
    rows, errs = table(doc, WF_TABLE, COLUMNS)
    errors += errs
    leg_rows, errs = table(doc, LEG_TABLE, LEG_COLUMNS)
    errors += errs
    dispatch_rows, errs = table(doc, DISPATCH_TABLE, DISPATCH_COLUMNS)
    errors += errs
    wait_rows, errs = table(doc, WAIT_TABLE, WAIT_COLUMNS)
    errors += errs
    after_rows, errs = table(doc, AFTER_TABLE, AFTER_COLUMNS)
    errors += errs

    names = set(files)
    for label, got in ((EVIDENCE, set(classes)), (DOC, {r["workflow"] for r in rows})):
        for n in sorted(names - got):
            errors.append(f"{label}: no entry for workflow {n}")
        for n in sorted(got - names):
            errors.append(f"{label}: entry for {n}, which is not a workflow")
    seen = set()
    for r in rows:
        if r["workflow"] in seen:
            errors.append(f"{DOC}: workflow {r['workflow']} listed twice")
        seen.add(r["workflow"])

    compared = 0
    for r in rows:
        name = r["workflow"]
        if name not in files:
            continue
        compared += 1
        want = derive(name, files[name])
        cls = classes.get(name)
        want["class"] = cls if isinstance(cls, str) else "?"
        for col in COLUMNS[1:]:
            if r[col] != want[col]:
                errors.append(f"{DOC}: {name}: `{col}` is {r[col]!r}, the workflow says {want[col]!r}")
        if cls not in CLASSES:
            errors.append(f"{EVIDENCE}: {name}: class {cls!r} is not one of {', '.join(CLASSES)}")
        elif cls in ("landing", "after-landing") and want["ci/** push"] == "-":
            errors.append(f"{name}: class {cls} but it does not run on a ci/** push")
        elif cls == "schedule-only" and want["push"] != "-":
            errors.append(f"{name}: class schedule-only but it triggers on push")
        elif cls == "tag-only" and not want["push"].startswith("tags "):
            errors.append(f"{name}: class tag-only but its push trigger is {want['push']!r}")
        if want["ci/** push"] != "-":
            errors += concurrency_errors(name, files[name])
        errors += guard_gate_errors(name, files[name], want["path gate"] != "-")
        if cls in ("landing", "after-landing") and want["dispatch"] != "yes":
            errors.append(f"{name}: class {cls} but no workflow_dispatch (a dropped run cannot be "
                          "re-created on the ci/<id> SHA)")

    want_legs = set()
    for leg in legs:
        w, j, key, mval = leg.get("workflow"), leg.get("job"), leg.get("key", "os"), leg.get("matrix")
        lcls, events = leg.get("class"), leg.get("events")
        label = f"{w}/{j}/{mval}"
        if not isinstance(events, list):
            errors.append(f"{EVIDENCE}: leg {label}: `events` is not a list")
            events = []
        want_legs.add((w, j, mval, lcls, ", ".join(events)))
        if classes.get(w) != "landing":
            errors.append(f"{EVIDENCE}: leg {label} of {w!r}, which is not a landing workflow")
            continue
        if lcls not in LEG_CLASSES:
            errors.append(f"{EVIDENCE}: leg {label}: class {lcls!r} is not one of {', '.join(LEG_CLASSES)}")
        js = jobs(files.get(w, ""))
        if j not in js:
            errors.append(f"{EVIDENCE}: leg {label}: no job {j!r}")
            continue
        try:
            present, triggered = leg_events(files[w], js[j], key, str(mval))
        except ExprError as e:
            errors.append(f"{w}: job {j}: {e}")
            continue
        if not present:
            errors.append(f"{EVIDENCE}: leg {label}: matrix value {mval!r} not in the job for any event")
            continue
        unknown = [e for e in events if e not in [x[0] for x in EVENTS]]
        if unknown:
            errors.append(f"{EVIDENCE}: leg {label}: unknown events {unknown}")
        if events != present:
            errors.append(f"{EVIDENCE}: leg {label}: declared events {events} but the workflow runs it on {present}")
        for output, field in CONDITION_OUTPUTS.items():
            when = leg.get(field)
            if output not in js[j]:
                if when is not None:
                    errors.append(f"{EVIDENCE}: leg {label}: `{field}` but the job does not read {output}")
                continue
            try:
                present_when, _ = leg_events(files[w], js[j], key, str(mval), output)
            except ExprError as e:
                errors.append(f"{w}: job {j}: {e}")
                continue
            if when is None and present_when != present:
                errors.append(f"{EVIDENCE}: leg {label}: with {output} true the workflow runs it on "
                              f"{present_when}; declare `{field}`")
            elif when is not None and when != present_when:
                errors.append(f"{EVIDENCE}: leg {label}: declared `{field}` {when} but with {output} true "
                              f"the workflow runs it on {present_when}")
        if lcls == "after-landing" and "push ci/**" not in present:
            errors.append(f"{EVIDENCE}: leg {label}: class after-landing but it is not started on a ci/** push")
        if lcls == "main-after-landing" and ("push ci/**" in present or "push main" not in present):
            errors.append(f"{EVIDENCE}: leg {label}: class main-after-landing needs a main push run "
                          "and no ci/** push run")
    # a leg of a landing workflow that runs on main but not on ci/** must be declared:
    # otherwise the landing reads a ci/** run that silently lacks it
    declared = {(lg.get("workflow"), lg.get("job"), str(lg.get("matrix"))) for lg in legs}
    for w, cls in classes.items():
        if cls != "landing" or w not in files:
            continue
        for j, block in jobs(files[w]).items():
            for key, expr in re.findall(r"^\s+([A-Za-z_][\w-]*):\s*(\$\{\{.*fromJSON.*)$", block, re.M):
                # a matrix computed from job outputs or inputs is not known before the run
                # (the CONDITION_OUTPUTS are evaluated as 'false': a leg one of them adds on
                # a ci/** push is still missing from the ordinary ci/** run)
                names = re.findall(r"\b(?:needs|steps|inputs|matrix|vars|env|secrets)\.[\w.\-]+", expr)
                if any(n not in CONDITION_OUTPUTS for n in names):
                    continue
                try:
                    main_vals = matrix_values(block, key, event_ctx("push", "refs/heads/main")) or []
                    ci_vals = matrix_values(block, key, event_ctx("push", "refs/heads/ci/x")) or []
                except ExprError as e:
                    errors.append(f"{w}: job {j}: {e}")
                    continue
                for v in main_vals:
                    if v not in ci_vals and (w, j, v) not in declared:
                        errors.append(f"{w}: job {j}: leg {key}={v} runs on a main push but not on a ci/** "
                                      f"push and is not declared in {EVIDENCE}")
    doc_legs = {tuple(r[c] for c in LEG_COLUMNS) for r in leg_rows}
    for leg in sorted(want_legs - doc_legs, key=str):
        errors.append(f"{DOC}: leg {leg} missing from the table")
    for leg in sorted(doc_legs - want_legs, key=str):
        errors.append(f"{DOC}: leg {leg} is not in {EVIDENCE}")

    have = {n for n, text in files.items() if derive(n, text)["dispatch"] == "yes"}
    listed = [r["workflow"] for r in dispatch_rows]
    for n in sorted(have - set(listed)):
        errors.append(f"{DOC}: dispatchable list misses {n}")
    for n in sorted(set(listed) - have):
        errors.append(f"{DOC}: dispatchable list names {n}, which has no workflow_dispatch")
    if len(listed) != len(set(listed)):
        errors.append(f"{DOC}: dispatchable list names a workflow twice")

    # the landing-waits table equals what scripts/land.py derives (land.expected_workflows)
    import land
    try:
        want_wait = {(w["file"], w["cls"], land.waits_for(w)) for w in land.expected_workflows(root)}
    except land.LandError as e:
        errors.append(f"scripts/land.py cannot derive the landing waits: {e}")
        want_wait = None
    if want_wait is not None:
        got_wait = {tuple(r[c] for c in WAIT_COLUMNS) for r in wait_rows}
        for row in sorted(want_wait - got_wait):
            errors.append(f"{DOC}: landing-waits row {row} missing (scripts/land.py waits on it)")
        for row in sorted(got_wait - want_wait):
            errors.append(f"{DOC}: landing-waits row {row} is not what scripts/land.py waits on")

    # the after-landing table equals what `land.py --wait` reads (land.after_landing_items)
    try:
        want_after = {(land.after_item_label(i), i["cls"]) for i in land.after_landing_items(root)}
    except land.LandError as e:
        errors.append(f"scripts/land.py cannot derive the after-landing set: {e}")
        want_after = None
    if want_after is not None:
        got_after = {tuple(r[c] for c in AFTER_COLUMNS) for r in after_rows}
        for row in sorted(want_after - got_after):
            errors.append(f"{DOC}: after-landing-wait row {row} missing (land.py --wait reads it)")
        for row in sorted(got_after - want_after):
            errors.append(f"{DOC}: after-landing-wait row {row} is not what land.py --wait reads")

    # the ledgers land.py counts as documents (land.LOCAL_LEDGERS) are named in the document
    for ledger in sorted(land.LOCAL_LEDGERS):
        if f"`{ledger}`" not in doc:
            errors.append(f"{DOC}: does not name `{ledger}`, which scripts/land.py counts as a document")

    if compared == 0:
        errors.append("no workflow row compared (compared nothing)")
    return errors, compared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    errors, n = check(Path(args.root))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: workflow rows {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
