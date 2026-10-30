#!/usr/bin/env python3
"""The reading of a reader map, shared by the repositories that keep one (ALICE-Law's conformance map, the whole-tree
map of the crates): whether a script or a workflow's run step names a file, so that a check can say which file a check
on a ci/** branch reads.

A file is named only by its whole repository-relative path as one token (`token`), and only where it is used: in a
shell script or a run step outside its comments (`without_shell_comments`), in a Python script in a string value, not
a comment or a docstring, with `/` chains and `Path(...)` calls joined (`python_strings`), or through a path name a
module it imports binds (`uses_imported_path`). A script counts as run only when a `run:` step names it
(`run_texts`), not when a `paths:` list or a comment does.

This file is shared byte for byte (ALICE-Law's conformance/shared-files.toml); each repository's map script imports it
and adds its own scope and rules.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

RUN_SCRIPT = re.compile(r"scripts/([A-Za-z0-9_.-]+\.(?:py|sh))")


def on_ci_branches(text: str) -> bool:
    """`ci/**` in the `branches` of the `on.push` block (the text between `on:` and `jobs:`)"""
    m = re.search(r"(?ms)^on:\s*$(.*?)^jobs:", text)
    if not m:
        return False
    block = m.group(1)
    push = re.search(r"(?ms)^  push:\s*$(.*?)(?=^  [a-z_]+:\s*$|\Z)", block)
    return bool(push and re.search(r"""branches:.*["']ci/\*\*["']""", push.group(1)))


RUN_KEY = re.compile(r"^(\s*)(?:- )?run:\s*(.*)$")


def run_texts(text: str) -> list[str]:
    """the text of every `run:` step of a workflow: a single-line value, or a block scalar (`|` / `>`) and the
    lines indented under it (a script named elsewhere, as in `paths:`, is not run)"""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = RUN_KEY.match(lines[i])
        i += 1
        if not m:
            continue
        indent = len(m.group(1)) + (2 if lines[i - 1].lstrip().startswith("- ") else 0)
        value = m.group(2).strip()
        if value and value[0] not in "|>":
            out.append(value)
            continue
        block = []
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
            block.append(lines[i])
            i += 1
        out.append("\n".join(block))
    return out


def without_shell_comments(text: str) -> str:
    """`text` with its shell comments taken out: a line whose first non-blank character is `#`, and from a `#` that
    starts a word (at the line start or after a blank) to the end of the line, outside single and double quotes.
    Limits: a quote is open until the same quote closes it on that line (a backslash-escaped quote, and a string that
    spans lines, are not followed), and a here-document's body is read like any other line."""
    out = []
    for line in text.splitlines():
        quote = ""
        cut = len(line)
        for i, ch in enumerate(line):
            if quote:
                if ch == quote:
                    quote = ""
            elif ch in "'\"":
                quote = ch
            elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def token(file: str) -> re.Pattern[str]:
    """`file`'s repository-relative path as one token of a string (no path character before or after it)"""
    return re.compile(r"(?<![\w./-])" + re.escape(file) + r"(?![\w.-])")


def python_strings(source: str) -> list[str]:
    """the strings a Python script uses as values: its string constants (f-string parts included) except the
    docstrings, and each `/` chain of them joined with `/` (`ROOT / "conformance" / "a.toml"` and
    `Path("conformance") / "a.toml"` give "conformance/a.toml"); comments are not part of the syntax tree. A script
    that does not parse uses nothing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return strings_in(tree)


PATH_CALLS = {"Path", "PurePath", "PurePosixPath"}


def path_call(n: ast.AST) -> bool:
    """a call that builds a path from its arguments (`Path("conformance", "a.toml")`, `pathlib.Path(…)`):
    its string arguments are parts of a `/` chain; another call's are not"""
    if not isinstance(n, ast.Call):
        return False
    f = n.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    return name in PATH_CALLS


def strings_in(tree: ast.AST) -> list[str]:
    """[`python_strings`] of a syntax tree (a whole module, or one assignment's value)"""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    out = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]

    def chain(n: ast.AST) -> list[ast.AST]:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return chain(n.left) + chain(n.right)
        if path_call(n):
            return [p for a in n.args for p in chain(a)]
        return [n]
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) or path_call(n):
            run: list[str] = []
            for part in chain(n) + [None]:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    run.append(part.value)
                else:
                    if len(run) > 1:
                        out.append("/".join(run))
                    run = []
    return out


def path_names(module: Path, want: str) -> set[str]:
    """the module-level names a scripts/ module binds to a value that uses the path `want`
    (`HOLDER_BASELINE = … / "conformance" / "holder-baseline.toml"`)"""
    if not module.is_file():
        return set()
    try:
        tree = ast.parse(module.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    found = set()
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(token(want).search(v) for v in strings_in(n.value)):
                found |= {t.id for t in targets if isinstance(t, ast.Name)}
    return found


def uses_imported_path(script: Path, tree: ast.AST, want: str) -> bool:
    """whether `script` uses a path name of a scripts/ module it imports: `alias.NAME` after `import m as alias`, or
    `from m import NAME`; importing a module for something else does not read its file"""
    aliases: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names = path_names(script.parent / f"{n.module.split('.')[0]}.py", want)
            if any(a.name in names for a in n.names):
                return True
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
            if n.attr in path_names(script.parent / f"{aliases[n.value.id]}.py", want):
                return True
    return False



def names_path(script: Path, want: str) -> bool:
    """whether `script` uses the path `want`: for Python, in a string value (not a comment or a docstring) or through
    a path name it imports; for a shell script, outside its comments"""
    source = script.read_text(encoding="utf-8", errors="replace")
    if script.suffix == ".py":
        if any(token(want).search(v) for v in python_strings(source)):
            return True
        try:
            return uses_imported_path(script, ast.parse(source), want)
        except SyntaxError:
            return False
    return bool(token(want).search(without_shell_comments(source)))
