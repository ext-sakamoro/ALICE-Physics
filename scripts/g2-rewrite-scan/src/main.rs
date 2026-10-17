//! Syntax-tree scan for a max, min or clamp written as a comparison (ALICE-Law G2).
//!
//! clippy's disallowed-methods sees method calls; a comparison is not one. This
//! tool parses every `.rs` file under the given directories and reports:
//!
//! * `expr`: `if <cond> { e1 } else { e2 }` where `<cond>` (through parentheses
//!   and `!`) is `<`, `>`, `<=` or `>=` of two expressions and `{e1, e2}` are
//!   those two expressions (compared as token streams, so formatting, line
//!   breaks and comments do not matter, and an operand may be any expression)
//! * `assign`: `if X op Y { ... X = Y; ... }` (either operand assigned the
//!   other, anywhere among the statements of the block)
//! * `raw`: a function whose name starts with `raw_`, the named helpers that
//!   call a platform operation on purpose
//!
//! Items under `#[cfg(test)]` are skipped. Each finding is one line:
//! `file<TAB>enclosing fn<TAB>shape<TAB>operands<TAB>line<TAB>spans`, the
//! operands as `a op b` in normalized token form, the spans as
//! `l:c-l:c|l:c-l:c` (line 1-based, column 0-based, end exclusive) of the two
//! operands in the condition. A summary line `scanned <files> <findings>`
//! goes to stderr. Exit 2 when no file was read or a file does not parse.
//!
//! usage: g2-rewrite-scan <dir>...

use quote::ToTokens;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{BinOp, Block, Expr, ExprIf, Stmt, UnOp};

fn norm(e: &Expr) -> String {
    e.to_token_stream().to_string()
}

/// The comparison under parentheses and `!`
fn comparison(mut e: &Expr) -> Option<(&Expr, &'static str, &Expr)> {
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
                return Some((&b.left, op, &b.right));
            }
            _ => return None,
        }
    }
}

/// The single expression a block evaluates to (`{ e }`)
fn sole_expr(b: &Block) -> Option<&Expr> {
    match b.stmts.as_slice() {
        [Stmt::Expr(e, None)] => Some(e),
        _ => None,
    }
}

fn else_expr(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::Block(b) => sole_expr(&b.block),
        _ => None,
    }
}

fn span_of(e: &Expr) -> String {
    let mut tokens = e.to_token_stream().into_iter();
    let first = tokens.next().map(|t| t.span());
    let last = e.to_token_stream().into_iter().last().map(|t| t.span());
    match (first, last) {
        (Some(f), Some(l)) => {
            let (s, t) = (f.start(), l.end());
            format!("{}:{}-{}:{}", s.line, s.column, t.line, t.column)
        }
        _ => "0:0-0:0".to_string(),
    }
}

/// `#[cfg(test)]`, or a `cfg` whose predicate requires `test` (`all(test, ..)`);
/// a predicate with `not(..)` or `any(..)` is not taken as test-only
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

struct Scan<'a> {
    file: &'a str,
    current_fn: String,
    out: Vec<String>,
}

impl Scan<'_> {
    fn emit(&mut self, shape: &str, operands: &str, line: usize, spans: &str) {
        self.out.push(format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.file, self.current_fn, shape, operands, line, spans
        ));
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
    fn visit_expr_if(&mut self, e: &'ast ExprIf) {
        if let Some((l, op, r)) = comparison(&e.cond) {
            let (ln, rn) = (norm(l), norm(r));
            let operands = format!("{ln} {op} {rn}");
            let line = e.if_token.span.start().line;
            let spans = format!("{}|{}", span_of(l), span_of(r));
            let mut matched = false;
            if let (Some(t), Some((_, el))) = (sole_expr(&e.then_branch), e.else_branch.as_ref()) {
                if let Some(f) = else_expr(el) {
                    let (tn, fnn) = (norm(t), norm(f));
                    if tn != fnn && ((tn == ln && fnn == rn) || (tn == rn && fnn == ln)) {
                        self.emit("expr", &operands, line, &spans);
                        matched = true;
                    }
                }
            }
            if !matched {
                let assigned = e.then_branch.stmts.iter().any(|s| match s {
                    Stmt::Expr(Expr::Assign(a), _) => {
                        let (an, bn) = (norm(&a.left), norm(&a.right));
                        (an == ln && bn == rn) || (an == rn && bn == ln)
                    }
                    _ => false,
                });
                if assigned {
                    self.emit("assign", &operands, line, &spans);
                }
            }
        }
        visit::visit_expr_if(self, e);
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
