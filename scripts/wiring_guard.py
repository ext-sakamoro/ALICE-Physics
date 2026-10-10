#!/usr/bin/env python3
"""配線ガード: 実装したが production から呼ばれていない item を CI で止める検査器.

検査 A (dead_code): `#[allow(dead_code)]` / `#![allow(dead_code)]` は
    `// ALLOW-DEAD: <12 字以上の理由>` を直前に置くか baseline に載せる
検査 B (unwired): `src/` の `pub` / `pub(crate)` な fn / struct / enum / const / static /
    trait / type / union で、test と自身の宣言と `use` 行を除いた production code
    (src / examples / benches / fuzz / bindings 等) から「生きている文脈」で参照されないものは未配線.
    `// ALLOW-UNWIRED: <12 字以上の理由>` を直前に置くか baseline に載せる
    生きている文脈とは、根 (src 外のコード / src/bin / module 直下 / trait 本体の member /
    src 外の型か型引数そのものへの trait impl の member /
    `ALLOW-UNWIRED` 付き / no_mangle 等の exempt 属性付き / `fn main` / macro_rules) か、
    根から到達できる item の本体 (宣言から波括弧の対応までの範囲) のこと
    src の型 T への trait impl (`impl Trait for T`) は T に結び付いた文脈で、T が生きた時に
    impl の見出しと本体 (member を含む) が生きる (`impl From<Foo> for Bar` は Bar が未配線なら Foo を配線しない)
    不動点反復で求めるので、未配線の item の本体からしか参照されない item (private helper を含む) も未配線になる
    (報告するのは従来どおり `pub` / `pub(crate)` のみ)
    参照と数えるのは束縛位置以外の出現: `let` / `for` / closure / fn 引数の束縛、`name:` (フィールド・引数)、
    `.name` (フィールド参照)、構造体の field shorthand、同じ fn 内で束縛された local 名の後続の使用は数えない
    free fn は `.name(` のメソッド呼び出しでは配線済にならず、メソッドは `.name(` か `::name` でだけ配線済になる
検査 C: 検査対象が 0 件なら fail (検査器が空振りして green になるのを防ぐ)
検査 D: src の波括弧が閉じていない file は unbalanced_braces で fail (本体の範囲を切れない)
検査 E: `// ALLOW-UNWIRED: wiring debt <key>` の key は docs/wiring-debt.md (公開の台帳) の行と
    2 方向で一致する 台帳に無い key / どの marker も使わない台帳の行 / kebab-case でない key は fail

Cargo workspace: root の `Cargo.toml` の `[workspace] members` (glob 可、`exclude` 対応) を展開し、
各 member の `src/` (と root 自身の `src/`) を定義の走査対象にする 参照 corpus は repo 全体なので
member 間の呼び出しは配線済になる violation の key は repo root からの相対パス

baseline (`scripts/wiring-baseline.txt`) は既存の違反を記録するラチェットで、
新規の違反だけが fail する 解消された entry が残っていても fail (stale_baseline).

限界: 名前で数えるので、(1) 別 file の同名 item は区別しない (同名の free fn が別 module にあれば一方の呼び出しで両方配線済)
(2) match 腕のパターン束縛や macro 内の束縛は束縛と認識しない (配線済側に倒れる)
(3) 型 T 自身の `impl T` / `impl<..> Trait for T` の見出しと本体の中の T は T の配線に数えない
    (同じ file の別の item が T を使う場合はその item が生きている時だけ数える) impl の対象の型は
    見出しの最後の識別子で決めるので、tuple や型 alias 越しの impl は対象外として扱う inherent impl
    (`impl T`) の見出しと本体の中の別の型・関数の参照は従来どおり impl を囲む文脈の参照として数え、
    trait impl のそれは T が生きている時だけ数える T は名前で決めるので、別 file の同名の型が生きても
    生きる (限界 (1) と同じ)
(4) T が src の型である trait impl の member は T が生きた時に生きる (どの trait method が呼ばれるかは
    見ない: T が生きていれば dispatch されうるとみなす) T が src 外の型 (u32 / Vec<..>) か見出しの
    型引数そのもの (`impl<T> Trait for T`) の trait impl と、trait 本体の member は常に根とみなす
(5) macro_rules 内の参照は常に根とみなす 偽陽性より偽陰性を選んでいる
"""
from __future__ import annotations

import argparse
import re
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

MIN_REASON = 12
SKIP_DIRS = {"target", ".git", "tests", "node_modules", "scratchpad"}


def _skipped(part: str) -> bool:
    """A directory the scan does not enter: build / VCS / test dirs, and any
    hidden directory (local tool state lives in dot directories)."""
    return part in SKIP_DIRS or (part.startswith(".") and part not in (".", ".."))
EXEMPT_ATTRS = re.compile(r"no_mangle|export_name|wasm_bindgen|pyfunction|pyclass|pymethods|napi|uniffi")
DEF_RE = re.compile(
    r"\bpub(?:\([^)]*\))?\s+"
    r"(?:(?:const|unsafe|async|extern(?:\s+\"[^\"]*\")?|default)\s+)*"
    r"(fn|struct|enum|const|static|trait|type|union)\s+([A-Za-z_]\w*)"
)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DECL_RE = re.compile(r"\b(?:fn|struct|enum|const|static|trait|type|mod|union)\s+([A-Za-z_][A-Za-z0-9_]*)")
USE_RE = re.compile(r"\b(?:pub(?:\([^)]*\))?\s+)?use\s+[^;]*;")
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:all\s*\(\s*)?test\b[^\]]*\]")
ALLOW_DEAD_RE = re.compile(r"#\s*!?\s*\[[^\]]*\ballow\s*\([^)]*\bdead_code\b")
MARKER_RE = re.compile(r"//\s*ALLOW-(UNWIRED|DEAD)\s*:(.*)$")
# `// ALLOW-UNWIRED: wiring debt <key> ...` names a tracked debt; the key resolves in
# DEBT_LEDGER, a public table, so the marker never points at a record outside the repo
DEBT_RE = re.compile(r"//\s*ALLOW-UNWIRED\s*:\s*wiring debt\s+(\S+)")
DEBT_KEY_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
DEBT_LEDGER = "docs/wiring-debt.md"
LEDGER_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")


@dataclass(frozen=True)
class Violation:
    kind: str
    key: str
    message: str


def strip_rust(src: str) -> str:
    """コメントと文字列 / 文字 literal を空白に置換する (改行と長さは保つ)."""
    out = list(src)
    n = len(src)
    i = 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', src[i : i + 40]) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'r(#*)"', src[i:])
            close = '"' + m.group(1)
            j = src.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
        elif c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                j = src.find("'", i + 2)
                j = n if j < 0 else j + 1
                blank(i, j)
                i = j
            elif i + 2 < n and src[i + 2] == "'":
                blank(i, i + 3)
                i += 3
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def blank_span(code: str, a: int, b: int) -> str:
    return code[:a] + "".join("\n" if ch == "\n" else " " for ch in code[a:b]) + code[b:]


def remove_cfg_test(code: str) -> str:
    """`#[cfg(test)]` が付いた item (mod / fn / use) を丸ごと空白にする."""
    while True:
        m = CFG_TEST_RE.search(code)
        if not m:
            return code
        j = m.end()
        n = len(code)
        while j < n and code[j] not in "{;":
            j += 1
        if j >= n:
            return blank_span(code, m.start(), n)
        if code[j] == ";":
            code = blank_span(code, m.start(), j + 1)
            continue
        depth, k = 0, j
        while k < n:
            if code[k] == "{":
                depth += 1
            elif code[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        code = blank_span(code, m.start(), min(k + 1, n))


ITEM_RE = re.compile(
    r"(?<!['\w])(fn|struct|enum|trait|union|type|static|const)\s+(?!(?:fn|unsafe|async|extern)\b)([A-Za-z_]\w*)"
)
IMPL_RE = re.compile(r"(?:^|(?<=[;{}\]]))\s*(?:unsafe\s+)?impl\b")
MACRO_RE = re.compile(r"\bmacro_rules\s*!\s*([A-Za-z_]\w*)")
ROOT_ATTRS = re.compile(r"#\s*\[\s*(?:[\w:]*::)?(?:test|bench)\b")
STRUCT_RE = re.compile(r"[(\[{;)\]}]")
BRACE_RE = re.compile(r"[{}]")
LET_RE = re.compile(r"\blet\b")
FOR_RE = re.compile(r"\bfor\b([^{};]*?)\bin\b")
CLOSURE_RE = re.compile(r"\|([^|;{}]*)\|")
PREV_WORD_RE = re.compile(r"([A-Za-z_]\w*)\s*$")
HEADER_CAP = 20000
TYPE_KINDS = ("struct", "enum", "union", "type", "trait")


class Node:
    """item (fn / struct / enum / trait / impl / macro_rules ...) の宣言から本体の終わりまでの範囲."""

    __slots__ = ("idx", "rel", "kind", "name", "name_pos", "start", "end", "body_start", "parent", "cls", "root", "exempt", "in_src",
                 "self_ty", "own_root", "tied", "type_params")

    def __init__(self, idx, rel, kind, name, name_pos, start, end, body_start, in_src):
        self.idx, self.rel, self.kind, self.name, self.name_pos = idx, rel, kind, name, name_pos
        self.start, self.end, self.body_start, self.in_src = start, end, body_start, in_src
        self.parent = None
        self.cls = "item"
        self.root = False
        self.exempt = False
        self.self_ty = None  # impl の対象の型名 (`impl<..> Trait for T` / `impl T` の T)
        self.own_root = False  # 親によらない根の理由 (exempt 属性 / test 属性 / main / macro)
        self.tied = False  # src の型 T の trait impl: 自分が文脈になり、T が生きた時に生きる
        self.type_params: set[str] = set()  # impl の見出しの `<..>` で宣言された型引数


def _prev_nonspace(code: str, pos: int) -> int:
    j = pos - 1
    while j >= 0 and code[j] in " \t\r\n":
        j -= 1
    return j


LIFETIME_RE = re.compile(r"'[A-Za-z_]\w*")
WHERE_RE = re.compile(r"\bwhere\b")
FOR_KW_RE = re.compile(r"\bfor\b")
SELF_TY_SKIP = {"mut", "dyn", "const", "crate", "self", "super", "unsafe"}


def _drop_angle(text: str) -> str:
    """`<...>` の中身を (入れ子ごと) 落とす `->` の `>` は数えない."""
    out: list[str] = []
    depth = 0
    for ch in text.replace("->", "  "):
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def impl_self_type(header: str) -> str | None:
    """`impl` の見出し (`impl` の直後から本体の `{` まで) から対象の型の名前を取る
    (`impl<T> Foo<T>` / `impl Trait for &'a mut a::Foo<T> where ..` の Foo) tuple 等は None."""
    h = _drop_angle(WHERE_RE.split(LIFETIME_RE.sub(" ", header))[0])
    h = FOR_KW_RE.split(h)[-1].strip()
    if not h or h[0] == "(":
        return None
    ids = [w for w in IDENT_RE.findall(h) if w not in SELF_TY_SKIP]
    return ids[-1] if ids else None


def impl_type_params(header: str) -> set[str]:
    """`impl<T: Copy, 'a, const N: usize>` の型引数名 {T, N} (見出しが `<` で始まらなければ空)."""
    h = header.lstrip()
    if not h.startswith("<"):
        return set()
    depth, parts, cur = 0, [], []
    for ch in h:
        if ch == "<":
            depth += 1
            if depth == 1:
                continue
        elif ch == ">":
            depth -= 1
            if depth == 0:
                break
        if depth == 1 and ch == ",":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    out: set[str] = set()
    for part in parts:
        part = part.strip()
        if not part or part.startswith("'"):
            continue
        m = re.match(r"(?:const\s+)?([A-Za-z_]\w*)", part)
        if m:
            out.add(m.group(1))
    return out


def _find_end(code: str, i: int, kind: str, match: dict[int, int]) -> tuple[int, int]:
    """item の (end, body_start) 見つからなければ上限で打ち切る (巨大 / 壊れた file で止まらない)."""
    n = len(code)
    limit = min(n, i + HEADER_CAP)
    depth = 0
    if kind in ("const", "static", "type"):
        for m in STRUCT_RE.finditer(code, i, limit):
            ch = m.group()
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    return m.start(), m.start()
                depth -= 1
            elif depth == 0:  # ';'
                return m.end(), m.end()
        return limit, limit
    for m in STRUCT_RE.finditer(code, i, limit):
        ch = m.group()
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "{" and depth == 0:
            return match.get(m.start(), n - 1) + 1, m.start()
        elif ch == ";" and depth == 0:
            return m.end(), m.end()
    return limit, limit


def parse_nodes(code: str, rel: str, in_src: bool, first_idx: int) -> list[Node]:
    """波括弧の対応で各 item の本体の範囲を切り出し、親子を付ける."""
    match: dict[int, int] = {}
    stack: list[int] = []
    for m in BRACE_RE.finditer(code):
        if m.group() == "{":
            stack.append(m.start())
        elif stack:
            match[stack.pop()] = m.start()
    cands: list[tuple[int, str, str | None, int]] = []  # (start, kind, name, name_pos)
    for m in ITEM_RE.finditer(code):
        kind = m.group(1)
        if kind in ("const", "static", "type"):
            j = _prev_nonspace(code, m.start())
            if j >= 0 and code[j] in "<,":
                continue  # `<const N: usize>` 等の generic 引数
        cands.append((m.start(), kind, m.group(2), m.start(2)))
    for m in IMPL_RE.finditer(code):
        cands.append((m.end() - 4, "impl", None, -1))
    for m in MACRO_RE.finditer(code):
        cands.append((m.start(), "macro", m.group(1), m.start(1)))
    cands.sort(key=lambda c: (c[0], c[1]))
    nodes: list[Node] = []
    for start, kind, name, name_pos in cands:
        scan_from = name_pos if name_pos >= 0 else start + 4
        end, body = _find_end(code, scan_from, "fn" if kind in ("impl", "macro") else kind, match)
        self_ty = impl_self_type(code[start + 4 : body]) if kind == "impl" else None
        tparams = impl_type_params(code[start + 4 : body]) if kind == "impl" else set()
        if kind == "impl" and re.search(r"\bfor\b(?!\s*<)", code[start + 4 : body]):
            kind = "impl_trait"
        nd = Node(first_idx + len(nodes), rel, kind, name, name_pos, start, end, body, in_src)
        nd.self_ty = self_ty
        nd.type_params = tparams
        nodes.append(nd)
    nodes.sort(key=lambda x: (x.start, -x.end))
    st: list[Node] = []
    for nd in nodes:
        while st and st[-1].end <= nd.start:
            st.pop()
        nd.parent = st[-1] if st else None
        st.append(nd)
    return nodes


def _bind_positions(code: str, names: set[str]) -> set[int]:
    """let / for / closure の pattern 内で、名前を束縛している識別子の位置."""
    pos: set[int] = set()
    n = len(code)

    def add(a: int, b: int) -> None:
        for m in IDENT_RE.finditer(code, a, b):
            nm = m.group()
            if nm not in names or not (nm[0].islower() or nm[0] == "_"):
                continue
            pj = _prev_nonspace(code, m.start())
            if pj >= 0 and (code[pj] == "." or (code[pj] == ":" and pj >= 1 and code[pj - 1] == ":")):
                continue
            k = m.end()
            while k < n and code[k] in " \t\r\n":
                k += 1
            if code.startswith(("(", "::", "{", "!"), k):
                continue
            pos.add(m.start())

    for m in LET_RE.finditer(code):
        i = m.end()
        depth, j, end = 0, i, min(n, i + 400)
        stop = end
        while j < end:
            ch = code[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth < 0:
                    stop = j
                    break
            elif depth == 0:
                if ch == ";":
                    stop = j
                    break
                if ch == ":":
                    if code.startswith("::", j):
                        j += 2
                        continue
                    stop = j
                    break
                if ch == "=" and code[j + 1 : j + 2] not in ("=", ">") and code[j - 1 : j] not in ("<", ">", "!", "="):
                    stop = j
                    break
            j += 1
        add(i, stop)
    for m in FOR_RE.finditer(code):
        add(m.start(1), m.end(1))
    for m in CLOSURE_RE.finditer(code):
        if not m.group(1).strip():
            continue
        j = _prev_nonspace(code, m.start())
        wm = PREV_WORD_RE.search(code, max(0, j - 8), j + 1) if j >= 0 else None
        if j < 0 or code[j] in "(,={;:" or (wm and wm.group(1) in ("move", "return")):
            add(m.start(1), m.end(1))
    return pos


def _is_struct_literal_field(code: str, s: int) -> bool:
    """`Foo { a, b }` の shorthand field か (直近の未対応の `{` の直前が大文字始まりの識別子)."""
    depth, j = 0, s - 1
    lim = max(0, s - 5000)
    while j >= lim:
        ch = code[j]
        if ch in ")]}":
            depth += 1
        elif ch in "([{":
            if depth == 0:
                if ch != "{":
                    return False
                m = PREV_WORD_RE.search(code, max(0, j - 80), j)
                return bool(m) and m.group(1)[0].isupper()
            depth -= 1
        j -= 1
    return False


def _compat(node: Node, dot: bool, path: bool) -> bool:
    """この出現の形が、その node を指しうるか (free fn は `.name(` に呼ばれない / method は `.name` か `::name`)."""
    if node.cls == "free":
        return not dot
    if node.cls == "method":
        return dot or path
    return True


def collect_refs(code: str, nodes: list[Node], names: set[str], decl_pos: set[int]) -> list[tuple[int, str, bool, bool]]:
    """束縛位置を除いた参照 (文脈 node の idx / 根なら -1, 名前, `.` の後か, `::` の後か) を列挙する."""
    n = len(code)
    bind_pos = _bind_positions(code, names)
    refs: list[tuple[int, str, bool, bool]] = []
    bound: dict[tuple[int, str], int] = {}
    st: list[Node] = []
    ni = 0
    for m in IDENT_RE.finditer(code):
        name = m.group()
        if name not in names:
            continue
        s, e = m.start(), m.end()
        if s in decl_pos:
            continue
        while ni < len(nodes) and nodes[ni].start <= s:
            while st and st[-1].end <= nodes[ni].start:
                st.pop()
            st.append(nodes[ni])
            ni += 1
        while st and st[-1].end <= s:
            st.pop()
        cur = st[-1] if st else None
        own = cur
        while own is not None and own.self_ty != name:
            own = own.parent
        if own is not None:
            continue  # T 自身の `impl T` / `impl Trait for T` の見出しと本体の T は T の配線にならない
        fn = cur
        while fn is not None and fn.kind != "fn":
            fn = fn.parent
        ctx = cur
        while ctx is not None and ctx.kind in ("impl", "impl_trait") and not ctx.tied:
            ctx = ctx.parent
        cid = -1 if ctx is None or ctx.root else ctx.idx
        j = _prev_nonspace(code, s)
        pc = code[j] if j >= 0 else ""
        dot = pc == "." and code[j - 1 : j] != "."
        path = pc == ":" and code[j - 1 : j] == ":"
        k = e
        while k < n and code[k] in " \t\r\n":
            k += 1
        nc = code[k : k + 1]
        call = nc == "(" or (code.startswith("::", k) and code[k + 2 : k + 12].lstrip()[:1] == "<")
        if name[0].islower() or name[0] == "_":
            if s in bind_pos:
                if fn is not None:
                    bound.setdefault((fn.idx, name), s)
                continue
            if nc == ":" and code[k + 1 : k + 2] != ":":
                if fn is not None and fn.start <= s < fn.body_start:
                    bound.setdefault((fn.idx, name), s)  # fn 引数
                continue
            if pc and (pc.isalnum() or pc == "_"):
                wm = PREV_WORD_RE.search(code, max(0, j - 8), j + 1)
                if wm and wm.group(1) in ("mut", "ref"):
                    if fn is not None:
                        bound.setdefault((fn.idx, name), s)
                    continue
            if dot and not call:
                continue  # field access
            if pc in ("{", ",") and nc in (",", "}") and _is_struct_literal_field(code, s):
                continue
            if not call and not path and not dot and fn is not None and bound.get((fn.idx, name), n + 1) < s:
                continue  # 同じ fn 内で束縛された local の使用
        refs.append((cid, name, dot, path))
    return refs


def workspace_src_dirs(root: Path) -> list[Path]:
    """root の `src/` と、`[workspace] members` (glob 可、exclude 対応) の各 member の `src/`."""
    dirs: list[Path] = []
    if (root / "src").is_dir():
        dirs.append(root / "src")
    cargo = root / "Cargo.toml"
    if cargo.is_file():
        text = "\n".join(re.sub(r"#.*$", "", ln) for ln in cargo.read_text(encoding="utf-8", errors="replace").split("\n"))
        sec = re.search(r"^\[workspace\][ \t]*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        if sec:

            def strs(key: str) -> list[str]:
                m = re.search(rf"^\s*{key}\s*=\s*\[(.*?)\]", sec.group(1), re.M | re.S)
                return [a or b for a, b in re.findall(r'"([^"]*)"|\'([^\']*)\'', m.group(1))] if m else []

            def safe(pat: str) -> bool:
                return bool(pat) and not pat.startswith(("/", "\\")) and ".." not in Path(pat).parts

            excl = [e.rstrip("/") for e in strs("exclude") if safe(e)]
            for pat in strs("members"):
                if not safe(pat):
                    continue
                for d in sorted(root.glob(pat.rstrip("/"))):
                    if not d.is_dir():
                        continue
                    rel = d.relative_to(root)
                    if any(_skipped(part) for part in rel.parts) or any(rel.as_posix() == e or rel.match(e) for e in excl):
                        continue
                    if (d / "src").is_dir() and (d / "src") not in dirs:
                        dirs.append(d / "src")
    return dirs


def rs_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.rs"):
        if any(_skipped(part) for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return sorted(out)


def lineno(code: str, pos: int) -> int:
    return code.count("\n", 0, pos)


def parse_baseline(text: str) -> tuple[set[str], dict[str, int]]:
    unwired: set[str] = set()
    dead: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "unwired" and len(parts) == 2:
            unwired.add(parts[1])
        elif parts[0] == "dead_code" and len(parts) == 3:
            dead[parts[1]] = int(parts[2])
    return unwired, dead


def check(root: Path, baseline_text: str = "") -> list[Violation]:
    root = Path(root)
    base_unwired, base_dead = parse_baseline(baseline_text)
    vs: list[Violation] = []

    src_dirs = workspace_src_dirs(root)
    files = []
    stripped: dict[Path, str] = {}
    raw: dict[Path, list[str]] = {}
    for p in rs_files(root):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            code = remove_cfg_test(strip_rust(text))
        except Exception as exc:  # 読めない / 解析できない file は明示的な違反にする
            vs.append(Violation("unreadable", p.relative_to(root).as_posix(), f"{type(exc).__name__}: {exc}"))
            continue
        files.append(p)
        raw[p] = text.split("\n")
        stripped[p] = code

    def src_of(p: Path) -> Path | None:
        for d in src_dirs:
            if d in p.parents:
                return d
        return None

    # 参照の走査用: use 文を空白にした code (長さと改行は保つ)
    plain = {p: USE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), c) for p, c in stripped.items()}

    # item の本体の範囲 (src 外は全部「根」の文脈)
    nodes: list[Node] = []
    file_nodes: dict[Path, list[Node]] = {}
    node_at: dict[tuple[Path, int], Node] = {}
    for p in files:
        sd = src_of(p)
        rel = p.relative_to(root).as_posix()
        in_src = sd is not None and (p.relative_to(sd).parts[0] != "bin")
        if sd is not None and plain[p].count("{") != plain[p].count("}"):
            vs.append(Violation("unbalanced_braces", rel, "波括弧が閉じていない (本体の範囲を切り出せない)"))
        try:
            fn_nodes = parse_nodes(plain[p], rel, in_src, len(nodes))
        except Exception as exc:
            vs.append(Violation("unreadable", rel, f"{type(exc).__name__}: {exc}"))
            fn_nodes = []
        # parse_nodes は start 順に並べ替えるので idx を振り直す
        for nd in fn_nodes:
            nd.idx = len(nodes)
            nodes.append(nd)
            if nd.name_pos >= 0:
                node_at[(p, nd.name_pos)] = nd
        file_nodes[p] = fn_nodes
        code = stripped[p]
        nl = [m.start() for m in re.finditer("\n", code)]
        for nd in fn_nodes:  # 親が先に来る順
            if not in_src:
                nd.root = True
                continue
            ln = bisect_left(nl, nd.start)
            ws = nl[ln - 4] + 1 if ln - 3 > 0 else 0
            we = nl[ln] if ln < len(nl) else len(code)
            window = code[ws:we]
            nd.exempt = bool(EXEMPT_ATTRS.search(window)) or (nd.parent is not None and nd.parent.exempt)
            nd.own_root = (
                nd.exempt
                or nd.kind == "macro"
                or (nd.kind == "fn" and nd.name == "main")
                or bool(ROOT_ATTRS.search(window))
            )
            nd.root = (
                nd.own_root
                or (nd.parent is not None and nd.parent.kind in ("trait", "impl_trait"))
                or (nd.parent is not None and nd.parent.root and nd.parent.kind in ("macro",))
            )
            if nd.kind == "fn":
                nd.cls = "method" if nd.parent is not None and nd.parent.kind == "impl" else "free"

    # 2 パス目: src の型 T の trait impl (`impl Trait for T`) は T に結び付ける 見出しと本体の
    # 参照はその impl を文脈にし、member は根にしない (T が生きた時に impl ごと生きる)
    # src に無い型 (u32 / Vec<..>) や見出しの型引数そのもの (`impl<T> Tr for T`) は従来どおり member が根
    src_types = {nd.name for nd in nodes if nd.in_src and nd.name and nd.kind in TYPE_KINDS}
    tied_by_type: dict[str, list[int]] = {}
    for nd in nodes:
        if (nd.kind == "impl_trait" and nd.in_src and not nd.root and nd.self_ty in src_types
                and nd.self_ty not in nd.type_params):
            nd.tied = True
            tied_by_type.setdefault(nd.self_ty, []).append(nd.idx)
    for nd in nodes:
        if nd.parent is not None and nd.parent.tied:
            nd.root = nd.own_root
            tied_by_type[nd.parent.self_ty].append(nd.idx)

    defs: list[tuple[str, str, int, Path, int]] = []  # (name, key, line, path, name_pos)
    exempt: set[str] = set()
    for p in files:
        if src_of(p) is None:
            continue
        code = stripped[p]
        lines = code.split("\n")
        rel = p.relative_to(root).as_posix()
        for m in DEF_RE.finditer(code):
            ln = lineno(code, m.start())
            above = " ".join(lines[max(0, ln - 4) : ln + 1])
            key = f"{rel}::{m.group(2)}"
            nd = node_at.get((p, m.start(2)))
            if EXEMPT_ATTRS.search(above) or (nd is not None and nd.exempt):
                exempt.add(key)
            defs.append((m.group(2), key, ln, p, m.start(2)))

    if not src_dirs or not defs:
        return vs + [Violation("empty_scan", str(root), "検査対象の pub item が 0 件 (src/ が無いか、検査器が何も見ていない)")]

    # --- 参照の収集と到達可能性 (不動点反復) ---
    names = {nd.name for nd in nodes if nd.in_src and nd.name}
    refs_by_ctx: dict[int, set[tuple[str, bool, bool]]] = {}
    refs_by_name: dict[str, list[tuple[int, bool, bool]]] = {}
    for p in files:
        decl_pos = {m.start(1) for m in DECL_RE.finditer(plain[p])} | {nd.name_pos for nd in file_nodes[p] if nd.name_pos >= 0}
        for cid, name, dot, path in collect_refs(plain[p], file_nodes[p], names, decl_pos):
            refs_by_ctx.setdefault(cid, set()).add((name, dot, path))
            refs_by_name.setdefault(name, []).append((cid, dot, path))
    by_name: dict[str, list[Node]] = {}
    for nd in nodes:
        if nd.in_src and nd.name and nd.kind != "macro":
            by_name.setdefault(nd.name, []).append(nd)
    static_roots = {nd.idx for nd in nodes if nd.root}

    def compute_live(extra_roots: set[int]) -> set[int]:
        live = set(static_roots) | extra_roots

        def tie(idxs: list[int], out: list[int]) -> None:
            """生きた型の trait impl (と その member) を生かす."""
            for i in idxs:
                nd = nodes[i]
                if nd.kind in TYPE_KINDS and nd.in_src:
                    for t in tied_by_type.get(nd.name, ()):
                        if t not in live:
                            live.add(t)
                            out.append(t)

        frontier = [-1] + sorted(live)
        tie(sorted(live), frontier)
        while frontier:
            nxt: list[int] = []
            for ctx in frontier:
                for name, dot, path in refs_by_ctx.get(ctx, ()):
                    for m in by_name.get(name, ()):
                        if m.idx not in live and m.idx != ctx and _compat(m, dot, path):
                            live.add(m.idx)
                            nxt.append(m.idx)
            tie(list(nxt), nxt)
            frontier = nxt
        return live

    def referenced_from_dead(nd: Node) -> bool:
        return any(c != nd.idx and _compat(nd, d, pa) for c, d, pa in refs_by_name.get(nd.name, ()))

    # --- markers ---
    unwired_markers: dict[str, tuple[Path, int, str]] = {}  # key -> (file, line, reason)
    dead_markers: list[tuple[Path, int, str]] = []
    for p in files:
        code_lines = stripped[p].split("\n")
        rel = p.relative_to(root).as_posix()
        for i, rawline in enumerate(raw[p]):
            m = MARKER_RE.search(rawline)
            if not m:
                continue
            kind, reason = m.group(1), m.group(2).strip()
            if len(reason) < MIN_REASON:
                vs.append(Violation("bad_marker", f"{rel}:{i + 1}", f"ALLOW-{kind} の理由が {MIN_REASON} 字未満: {reason!r}"))
                continue
            if code_lines[i].strip():
                target = i  # 同一行にコードがあればその行
            else:
                j = i + 1
                target = None
                while j < len(code_lines):
                    c = code_lines[j]
                    if not c.strip():
                        j += 1
                        continue
                    if kind == "DEAD" and ALLOW_DEAD_RE.search(c):
                        target = j
                        break
                    if c.lstrip().startswith("#"):
                        # skip the whole attribute: `#[deprecated(` ... `)]` spans lines
                        depth = c.count("[") - c.count("]")
                        j += 1
                        while depth > 0 and j < len(code_lines):
                            depth += code_lines[j].count("[") - code_lines[j].count("]")
                            j += 1
                        continue
                    target = j
                    break
            if kind == "UNWIRED":
                names = [(dm.group(2)) for dm in DEF_RE.finditer(code_lines[target])] if target is not None else []
                if not names:
                    vs.append(Violation("stale_marker", f"{rel}:{i + 1}", "ALLOW-UNWIRED の直後に pub item が無い"))
                for nm in names:
                    unwired_markers[f"{rel}::{nm}"] = (p, i + 1, reason)
            else:
                if target is None or not ALLOW_DEAD_RE.search(code_lines[target]):
                    vs.append(Violation("stale_marker", f"{rel}:{i + 1}", "ALLOW-DEAD の直後に allow(dead_code) が無い"))
                else:
                    dead_markers.append((p, target, reason))

    # --- 検査 B: unwired ---
    def_nodes: dict[str, list[Node]] = {}
    for name, key, ln, p, npos in defs:
        nd = node_at.get((p, npos))
        if nd is not None:
            def_nodes.setdefault(key, []).append(nd)
    marker_nodes = {nd.idx for k in unwired_markers for nd in def_nodes.get(k, ())}
    live_all = compute_live(marker_nodes)

    def naturally_wired(key: str) -> bool:
        """marker 自身を根にしない場合に配線済か (marker 付きの item の stale 判定用)."""
        mine = {nd.idx for nd in def_nodes[key]}
        if mine & static_roots:
            return True
        rest = marker_nodes - mine
        outer = [(c, d, pa) for nd in def_nodes[key] for c, d, pa in refs_by_name.get(nd.name, ()) if c not in mine and _compat(nd, d, pa)]
        if not any(c == -1 or c in live_all for c, _d, _pa in outer):
            return False
        if any(c == -1 for c, _d, _pa in outer):
            return True
        return bool(mine & compute_live(rest))

    current_unwired: set[str] = set()
    for name, key, ln, p, npos in defs:
        if key in exempt or key not in def_nodes:
            continue
        if key in unwired_markers:
            if naturally_wired(key):
                continue
        elif any(nd.idx in live_all for nd in def_nodes[key]):
            continue
        current_unwired.add(key)
        if key in unwired_markers or key in base_unwired:
            continue
        nd = def_nodes[key][0]
        if referenced_from_dead(nd):
            msg = f"{name}: 未配線の item の本体 (または自己再帰) からしか参照されていない"
        else:
            msg = f"{name}: production code から 1 度も参照されていない (test / doc / use のみ)"
        vs.append(Violation("unwired", key, msg))
    for key, (p, ln, _r) in unwired_markers.items():
        if key not in current_unwired:
            vs.append(Violation("stale_marker", f"{key}", f"ALLOW-UNWIRED があるが配線済 ({p.name}:{ln}) マーカーを消す"))
    for key in sorted(base_unwired - current_unwired):
        vs.append(Violation("stale_baseline", key, "baseline にあるが既に配線済 / 消えている 行を消す"))

    # --- 検査 A: dead_code ---
    marked_lines = {(p, ln) for p, ln, _ in dead_markers}
    actual_unmarked: dict[str, int] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        for i, c in enumerate(stripped[p].split("\n")):
            if ALLOW_DEAD_RE.search(c) and (p, i) not in marked_lines:
                actual_unmarked[rel] = actual_unmarked.get(rel, 0) + 1
    for rel, n in sorted(actual_unmarked.items()):
        allowed = base_dead.get(rel, 0)
        if n > allowed:
            vs.append(Violation("dead_code", rel, f"理由マーカーの無い allow(dead_code) が {n} 件 (baseline {allowed})"))
    for rel, allowed in sorted(base_dead.items()):
        n = actual_unmarked.get(rel, 0)
        if allowed > n:
            vs.append(Violation("stale_baseline", rel, f"baseline の dead_code {allowed} 件に対し実際は {n} 件 数を減らす"))
    vs += check_debt_ledger(root, files, raw)
    return vs


def check_debt_ledger(root: Path, files: list[Path], raw: dict[Path, list[str]]) -> list[Violation]:
    """検査 E: `wiring debt <key>` の key は docs/wiring-debt.md の行と 2 方向で一致する
    (marker の key が台帳に無い / 台帳の key をどの marker も使っていない / key の書式違反 はどれも fail)"""
    vs: list[Violation] = []
    used: dict[str, str] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(raw[p]):
            m = DEBT_RE.search(line)
            if not m:
                continue
            key = m.group(1).rstrip(",.;:")
            if not DEBT_KEY_RE.fullmatch(key):
                vs.append(Violation("bad_debt_key", f"{rel}:{i + 1}",
                                    f"wiring debt の key {key!r} は小文字・数字・ハイフンの kebab-case にする"))
                continue
            used.setdefault(key, f"{rel}:{i + 1}")
    ledger = root / DEBT_LEDGER
    if not ledger.is_file():
        if used:
            vs.append(Violation("debt_ledger", DEBT_LEDGER, f"wiring debt の marker が {len(used)} 種あるが台帳が無い"))
        return vs
    rows: dict[str, int] = {}
    for n, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        m = LEDGER_ROW_RE.match(line)
        if m:
            if m.group(1) in rows:
                vs.append(Violation("debt_ledger", f"{DEBT_LEDGER}:{n}", f"key {m.group(1)!r} が 2 行ある"))
            rows.setdefault(m.group(1), n)
    if not rows:
        vs.append(Violation("debt_ledger", DEBT_LEDGER, "台帳に key の行が 1 つも無い (比較 0 件)"))
    for key, where in sorted(used.items()):
        if key not in rows:
            vs.append(Violation("debt_unlisted", where, f"wiring debt {key} が {DEBT_LEDGER} に無い"))
    for key, n in sorted(rows.items()):
        if key not in used:
            vs.append(Violation("debt_stale", f"{DEBT_LEDGER}:{n}", f"台帳の {key} をどの marker も使っていない (解消済なら行を消す)"))
    return vs


def build_baseline(root: Path) -> str:
    vs = check(root, "")
    lines = ["# wiring-guard baseline: 既存の違反 (ラチェット) 新規は fail、解消したら行を消す"]
    dead: dict[str, int] = {}
    for v in vs:
        if v.kind == "unwired":
            lines.append(f"unwired {v.key}")
        elif v.kind == "dead_code":
            dead[v.key] = int(re.search(r"(\d+) 件", v.message).group(1))
    lines += [f"dead_code {k} {n}" for k, n in sorted(dead.items())]
    return "\n".join([lines[0]] + sorted(lines[1:])) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--baseline", default="scripts/wiring-baseline.txt")
    ap.add_argument("--update-baseline", action="store_true", help="現在の違反を baseline として書き出す (レビュー必須)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    bpath = root / a.baseline
    if a.update_baseline:
        bpath.write_text(build_baseline(root), encoding="utf-8")
        print(f"wrote {bpath}")
        return 0
    text = bpath.read_text(encoding="utf-8") if bpath.exists() else ""
    vs = check(root, text)
    if vs:
        for v in vs:
            print(f"{v.kind}: {v.key}: {v.message}", file=sys.stderr)
        print(f"wiring-guard: {len(vs)} violation(s)", file=sys.stderr)
        return 1
    print("wiring-guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
