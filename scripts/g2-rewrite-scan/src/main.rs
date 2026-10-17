//! Syntax-tree scan for a max, min or clamp written as a comparison (ALICE-Law G2).
//!
//! clippy's disallowed-methods sees method calls; a comparison is not one. This
//! tool parses every `.rs` file under the given directories and reports:
//!
//! * `expr`: a selection between the two operands of a comparison, written as
//!   `if <cond> { .. } else { .. }` or `match <cond> { true => .., false => .. }`,
//!   where one branch yields one operand and the other branch the other. A branch
//!   yields the tail expression of its block (any statements may come first), the
//!   value of a `return` in it, or, through `else if`, every arm of the chain
//! * `return`: `if <cond> { ..; return a; }` followed, in the same block, by
//!   statements that yield `b` (the early-return form of the same selection)
//! * `assign`: `if X op Y { .. X = Y; .. }`, the assignment in either branch
//! * `raw`: a function whose name starts with `raw_`, the named helpers that
//!   call a platform operation on purpose
//! * `macro`: every `macro_rules!`; its rule bodies (with `$name` read as `name`,
//!   as statements or as items) are scanned like source as well, so a selection
//!   inside one is reported; a body that parses as neither is named in the row
//!   (`<name> (a body does not parse)`), so an unread body cannot pass silently
//!
//! A condition is `<`, `>`, `<=` or `>=` of two expressions, the methods `lt`,
//! `gt`, `le`, `ge`, under parentheses and `!`, or a variable bound to one with
//! `let` earlier in an enclosing block. Operands and branch values are compared
//! as token streams after removing parentheses, `*`, `&`, `.clone()` and
//! `.to_owned()`, so formatting, line breaks, comments, references and copies
//! do not hide a site. Out of scope: selections through `partial_cmp`,
//! `total_cmp` or `Ordering` (ordering rules, not a max or min value).
//!
//! Items under `#[cfg(test)]` (or a `cfg(all(test, ..))`) are skipped. Each
//! finding is one line: `file<TAB>enclosing fn<TAB>shape<TAB>operands<TAB>line<TAB>spans`,
//! the operands as `a op b` in normalized token form, the spans as
//! `l:c-l:c|l:c-l:c` (line 1-based, column 0-based, end exclusive) of the two
//! operands, empty for `raw`, `macro` and a site inside a macro body. A summary
//! line `scanned <files> <findings>` goes to stderr. Exit 2 when no file was
//! read or a file does not parse.
//!
//! usage: g2-rewrite-scan <dir>...

use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{BinOp, Block, Expr, ExprIf, ExprMatch, Lit, Pat, Stmt, UnOp};

/// The expression with parentheses, `*`, `&`, `.clone()` and `.to_owned()` removed
fn strip(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Paren(p) => e = &p.expr,
            Expr::Group(g) => e = &g.expr,
            Expr::Reference(r) => e = &r.expr,
            Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => e = &u.expr,
            Expr::MethodCall(m)
                if m.args.is_empty() && (m.method == "clone" || m.method == "to_owned") =>
            {
                e = &m.receiver
            }
            _ => return e,
        }
    }
}

fn norm(e: &Expr) -> String {
    strip(e).to_token_stream().to_string()
}

fn span_of(e: &Expr) -> String {
    let ts = e.to_token_stream();
    let first = ts.clone().into_iter().next().map(|t| t.span());
    let last = ts.into_iter().last().map(|t| t.span());
    match (first, last) {
        (Some(f), Some(l)) => {
            let (s, t) = (f.start(), l.end());
            format!("{}:{}-{}:{}", s.line, s.column, t.line, t.column)
        }
        _ => "0:0-0:0".to_string(),
    }
}

#[derive(Clone)]
struct Cmp {
    l: Expr,
    op: &'static str,
    r: Expr,
}

fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if !a.path().is_ident("cfg") {
            return false;
        }
        let s = a.meta.to_token_stream().to_string().replace(' ', "");
        let Some(inner) = s.strip_prefix("cfg(").and_then(|x| x.strip_suffix(')')) else {
            return false;
        };
        if inner == "test" {
            return true;
        }
        inner.starts_with("all(")
            && !inner.contains("not(")
            && !inner.contains("any(")
            && inner[4..inner.len() - 1].split(',').any(|p| p == "test")
    })
}

/// The values a block can yield: its tail expression and every `return` among its statements
fn block_values(stmts: &[Stmt]) -> Option<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for (i, s) in stmts.iter().enumerate() {
        match s {
            Stmt::Expr(Expr::Return(r), _) => {
                out.insert(norm(r.expr.as_ref()?));
            }
            Stmt::Expr(e, None) if i + 1 == stmts.len() => {
                out.extend(values(e)?);
            }
            _ => {}
        }
    }
    (!out.is_empty()).then_some(out)
}

/// The values an expression in value position can yield (through blocks and `else if` arms)
fn values(e: &Expr) -> Option<BTreeSet<String>> {
    match strip(e) {
        Expr::Block(b) => block_values(&b.block.stmts),
        Expr::If(i) => {
            let mut v = block_values(&i.then_branch.stmts)?;
            v.extend(values(&i.else_branch.as_ref()?.1)?);
            Some(v)
        }
        Expr::Return(r) => Some(BTreeSet::from([norm(r.expr.as_ref()?)])),
        other => Some(BTreeSet::from([norm(other)])),
    }
}

/// The values the `return` statements of a block yield (the then-block of an early return)
fn returns(stmts: &[Stmt]) -> BTreeSet<String> {
    stmts
        .iter()
        .filter_map(|s| match s {
            Stmt::Expr(Expr::Return(r), _) => r.expr.as_ref().map(|x| norm(x)),
            _ => None,
        })
        .collect()
}

fn selects(cmp: &Cmp, a: &BTreeSet<String>, b: &BTreeSet<String>) -> bool {
    let (ln, rn) = (norm(&cmp.l), norm(&cmp.r));
    if a.len() != 1 || b.len() != 1 || ln == rn {
        return false;
    }
    let (x, y) = (a.iter().next().unwrap(), b.iter().next().unwrap());
    x != y && ((*x == ln && *y == rn) || (*x == rn && *y == ln))
}

fn assigns(stmts: &[Stmt], cmp: &Cmp) -> bool {
    let (ln, rn) = (norm(&cmp.l), norm(&cmp.r));
    stmts.iter().any(|s| match s {
        Stmt::Expr(Expr::Assign(a), _) => {
            let (an, bn) = (norm(&a.left), norm(&a.right));
            (an == ln && bn == rn) || (an == rn && bn == ln)
        }
        _ => false,
    })
}

struct Scan<'a> {
    file: &'a str,
    current_fn: String,
    in_macro: bool,
    binds: Vec<HashMap<String, Cmp>>,
    out: Vec<String>,
}

impl Scan<'_> {
    fn emit(&mut self, shape: &str, operands: &str, line: usize, spans: &str) {
        let spans = if self.in_macro { "" } else { spans };
        self.out.push(format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.file, self.current_fn, shape, operands, line, spans
        ));
    }

    fn emit_cmp(&mut self, shape: &str, cmp: &Cmp, line: usize) {
        let operands = format!("{} {} {}", norm(&cmp.l), cmp.op, norm(&cmp.r));
        let spans = format!("{}|{}", span_of(&cmp.l), span_of(&cmp.r));
        self.emit(shape, &operands, line, &spans);
    }

    /// The comparison a condition is, through parentheses, `!`, the `lt`/`gt`/`le`/`ge`
    /// methods and a `let`-bound variable
    fn cmp_of(&self, mut e: &Expr) -> Option<Cmp> {
        loop {
            match e {
                Expr::Paren(p) => e = &p.expr,
                Expr::Group(g) => e = &g.expr,
                Expr::Unary(u) if matches!(u.op, UnOp::Not(_)) => e = &u.expr,
                Expr::Binary(b) => {
                    let op = match b.op {
                        BinOp::Lt(_) => "<",
                        BinOp::Gt(_) => ">",
                        BinOp::Le(_) => "<=",
                        BinOp::Ge(_) => ">=",
                        _ => return None,
                    };
                    return Some(Cmp {
                        l: (*b.left).clone(),
                        op,
                        r: (*b.right).clone(),
                    });
                }
                Expr::MethodCall(m) if m.args.len() == 1 => {
                    let op = match m.method.to_string().as_str() {
                        "lt" => "<",
                        "gt" => ">",
                        "le" => "<=",
                        "ge" => ">=",
                        _ => return None,
                    };
                    return Some(Cmp {
                        l: (*m.receiver).clone(),
                        op,
                        r: m.args[0].clone(),
                    });
                }
                Expr::Path(p) if p.path.segments.len() == 1 && p.qself.is_none() => {
                    let name = p.path.segments[0].ident.to_string();
                    return self.binds.iter().rev().find_map(|s| s.get(&name)).cloned();
                }
                _ => return None,
            }
        }
    }

    fn enter_fn(
        &mut self,
        name: String,
        attrs: &[syn::Attribute],
        line: usize,
        body: impl FnOnce(&mut Self),
    ) {
        if is_cfg_test(attrs) {
            return;
        }
        if name.starts_with("raw_") {
            let saved = std::mem::replace(&mut self.current_fn, name.clone());
            self.emit("raw", &name, line, "");
            self.current_fn = saved;
        }
        let saved = std::mem::replace(&mut self.current_fn, name);
        body(self);
        self.current_fn = saved;
    }

    /// Scans every rule body; false when a body does not parse as a block (it was not read)
    fn scan_macro_body(&mut self, name: &str, tokens: TokenStream) -> bool {
        let mut all_parsed = true;
        let tts: Vec<TokenTree> = tokens.into_iter().collect();
        let mut i = 0;
        while i < tts.len() {
            // a rule: (pattern) => {body}
            if let (
                Some(TokenTree::Punct(eq)),
                Some(TokenTree::Punct(gt)),
                Some(TokenTree::Group(body)),
            ) = (tts.get(i), tts.get(i + 1), tts.get(i + 2))
            {
                if eq.as_char() == '=' && gt.as_char() == '>' {
                    let flat = demeta(body.stream());
                    let block: TokenStream =
                        TokenTree::Group(Group::new(Delimiter::Brace, flat)).into();
                    if let Ok(b) = syn::parse2::<Block>(block) {
                        let saved =
                            std::mem::replace(&mut self.current_fn, format!("macro {name}"));
                        let was = std::mem::replace(&mut self.in_macro, true);
                        self.visit_block(&b);
                        self.in_macro = was;
                        self.current_fn = saved;
                    } else if let Ok(f) = syn::parse2::<syn::File>(demeta(body.stream())) {
                        // a body of items (functions, impls) rather than statements
                        let saved =
                            std::mem::replace(&mut self.current_fn, format!("macro {name}"));
                        let was = std::mem::replace(&mut self.in_macro, true);
                        self.visit_file(&f);
                        self.in_macro = was;
                        self.current_fn = saved;
                    } else {
                        all_parsed = false;
                    }
                    i += 3;
                    continue;
                }
            }
            i += 1;
        }
        all_parsed
    }
}

/// A macro body with `$name` read as `name` and each `$( .. ) sep? *|+|?` repetition read once
fn demeta(ts: TokenStream) -> TokenStream {
    let tts: Vec<TokenTree> = ts.into_iter().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < tts.len() {
        match (&tts[i], tts.get(i + 1)) {
            (TokenTree::Punct(p), Some(TokenTree::Ident(id))) if p.as_char() == '$' => {
                out.push(TokenTree::Ident(id.clone()));
                i += 2;
            }
            (TokenTree::Punct(p), Some(TokenTree::Group(g))) if p.as_char() == '$' => {
                out.extend(demeta(g.stream()));
                i += 2;
                // optional separator, then the repetition operator
                for _ in 0..2 {
                    if let Some(TokenTree::Punct(q)) = tts.get(i) {
                        if matches!(q.as_char(), '*' | '+' | '?') {
                            i += 1;
                            break;
                        }
                        if q.as_char() != '$' {
                            i += 1;
                            continue;
                        }
                    }
                    break;
                }
            }
            (TokenTree::Group(g), _) => {
                let mut ng = Group::new(g.delimiter(), demeta(g.stream()));
                ng.set_span(g.span());
                out.push(TokenTree::Group(ng));
                i += 1;
            }
            (t, _) => {
                out.push(t.clone());
                i += 1;
            }
        }
    }
    out.into_iter().collect()
}

impl<'ast> Visit<'ast> for Scan<'_> {
    fn visit_item_mod(&mut self, m: &'ast syn::ItemMod) {
        if !is_cfg_test(&m.attrs) {
            visit::visit_item_mod(self, m);
        }
    }
    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        if !is_cfg_test(&i.attrs) {
            visit::visit_item_impl(self, i);
        }
    }
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        let line = f.sig.ident.span().start().line;
        self.enter_fn(f.sig.ident.to_string(), &f.attrs, line, |s| {
            visit::visit_item_fn(s, f)
        });
    }
    fn visit_impl_item_fn(&mut self, f: &'ast syn::ImplItemFn) {
        let line = f.sig.ident.span().start().line;
        self.enter_fn(f.sig.ident.to_string(), &f.attrs, line, |s| {
            visit::visit_impl_item_fn(s, f)
        });
    }
    fn visit_trait_item_fn(&mut self, f: &'ast syn::TraitItemFn) {
        let line = f.sig.ident.span().start().line;
        self.enter_fn(f.sig.ident.to_string(), &f.attrs, line, |s| {
            visit::visit_trait_item_fn(s, f)
        });
    }
    fn visit_item_macro(&mut self, m: &'ast syn::ItemMacro) {
        if is_cfg_test(&m.attrs) {
            return;
        }
        if m.mac.path.is_ident("macro_rules") {
            let name = m
                .ident
                .as_ref()
                .map_or_else(|| "<unnamed>".to_string(), ToString::to_string);
            let line = m.mac.path.segments[0].ident.span().start().line;
            // the row is emitted after the scan so it can say whether every body was read
            let mark = self.out.len();
            let parsed = self.scan_macro_body(&name, m.mac.tokens.clone());
            let label = if parsed {
                name.clone()
            } else {
                format!("{name} (a body does not parse)")
            };
            let saved = std::mem::replace(&mut self.current_fn, "<module>".to_string());
            self.emit("macro", &label, line, "");
            self.current_fn = saved;
            let row = self.out.pop().unwrap();
            self.out.insert(mark, row);
        }
    }
    fn visit_block(&mut self, b: &'ast Block) {
        self.binds.push(HashMap::new());
        for (i, s) in b.stmts.iter().enumerate() {
            // early return: `if c { ..; return a; }` then the rest of the block yields `b`
            if let Stmt::Expr(Expr::If(e), _) = s {
                if e.else_branch.is_none() {
                    if let Some(cmp) = self.cmp_of(&e.cond) {
                        let then = returns(&e.then_branch.stmts);
                        if let Some(rest) = block_values(&b.stmts[i + 1..]) {
                            if selects(&cmp, &then, &rest) {
                                self.emit_cmp("return", &cmp, e.if_token.span.start().line);
                            }
                        }
                    }
                }
            }
            self.visit_stmt(s);
            if let Stmt::Local(l) = s {
                if let (Pat::Ident(pi), Some(init)) = (&l.pat, &l.init) {
                    if let Some(cmp) = self.cmp_of(&init.expr) {
                        self.binds
                            .last_mut()
                            .unwrap()
                            .insert(pi.ident.to_string(), cmp);
                    }
                }
            }
        }
        self.binds.pop();
    }
    fn visit_expr_if(&mut self, e: &'ast ExprIf) {
        if let Some(cmp) = self.cmp_of(&e.cond) {
            let line = e.if_token.span.start().line;
            let then = block_values(&e.then_branch.stmts);
            let other = e.else_branch.as_ref().and_then(|(_, x)| values(x));
            if let (Some(a), Some(b)) = (&then, &other) {
                if selects(&cmp, a, b) {
                    self.emit_cmp("expr", &cmp, line);
                    visit::visit_expr_if(self, e);
                    return;
                }
            }
            let else_stmts = e.else_branch.as_ref().and_then(|(_, x)| match strip(x) {
                Expr::Block(b) => Some(&b.block.stmts),
                _ => None,
            });
            if assigns(&e.then_branch.stmts, &cmp) || else_stmts.is_some_and(|s| assigns(s, &cmp)) {
                self.emit_cmp("assign", &cmp, line);
            }
        }
        visit::visit_expr_if(self, e);
    }
    fn visit_expr_match(&mut self, m: &'ast ExprMatch) {
        if let Some(cmp) = self.cmp_of(&m.expr) {
            let (mut t, mut f) = (BTreeSet::new(), BTreeSet::new());
            let mut ok = true;
            for arm in &m.arms {
                let target = match &arm.pat {
                    Pat::Lit(l) => match &l.lit {
                        Lit::Bool(b) if b.value => &mut t,
                        Lit::Bool(_) => &mut f,
                        _ => {
                            ok = false;
                            break;
                        }
                    },
                    Pat::Wild(_) => &mut f,
                    _ => {
                        ok = false;
                        break;
                    }
                };
                match values(&arm.body) {
                    Some(v) => target.extend(v),
                    None => ok = false,
                }
            }
            if ok && selects(&cmp, &t, &f) {
                self.emit_cmp("expr", &cmp, m.match_token.span.start().line);
            }
        }
        visit::visit_expr_match(self, m);
    }
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn main() {
    let dirs: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    let mut files = Vec::new();
    for d in &dirs {
        rs_files(d, &mut files);
    }
    if files.is_empty() {
        eprintln!("g2-rewrite-scan: no .rs file under {dirs:?} (0 scanned)");
        std::process::exit(2);
    }
    let mut findings = 0usize;
    for path in &files {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("g2-rewrite-scan: {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        let ast = match syn::parse_file(&text) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("g2-rewrite-scan: {}: does not parse: {e}", path.display());
                std::process::exit(2);
            }
        };
        let rel = path.to_string_lossy().replace('\\', "/");
        let mut s = Scan {
            file: &rel,
            current_fn: "<module>".to_string(),
            in_macro: false,
            binds: Vec::new(),
            out: Vec::new(),
        };
        s.visit_file(&ast);
        findings += s.out.len();
        for line in s.out {
            println!("{line}");
        }
    }
    eprintln!("scanned {} {}", files.len(), findings);
}
