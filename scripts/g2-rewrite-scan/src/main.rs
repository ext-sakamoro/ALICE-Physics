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
//! Not reported (known holes, so this is a partial inventory, not a
//! completeness claim): a tuple sort2 (`if a < b { (a, b) } else { (b, a) }`),
//! the swap form (`if hi < lo { std::mem::swap(&mut lo, &mut hi) }`), a
//! compound condition (`a < b && ok`), a helper predicate, `(a < b).then(..)`,
//! an array indexed by a comparison and `PartialOrd::lt(&a, &b)`.
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
//!
//! `g2-rewrite-scan --comparisons <dir>...` instead lists every ordering comparison in the
//! non-test code, whatever its context (the inventory scripts/g2_comparisons.py types with
//! the compiler): a binary `<`, `>`, `<=` or `>=`; a method call `lt`, `gt`, `le`, `ge`,
//! `partial_cmp`, `total_cmp`, `max_by` / `min_by` closures included because their bodies are
//! visited; a path call `PartialOrd::lt(&a, &b)` (any path ending in one of those names with
//! two arguments); and the same inside the arguments of a macro invocation that parse as
//! expressions (`assert!`, `debug_assert!`, `format!` ...). One line per comparison:
//! `file<TAB>enclosing fn<TAB>kind<TAB>operands<TAB>line<TAB>probe span<TAB>context`, where kind is
//! `bin`, `method`, `path` or `macro-arg`, the operands are `a op b` in normalized token
//! form, and the probe span `l:c-l:c` is the operand whose type the probe asks for (the left
//! one, or the right one when the left is a literal); empty inside a `macro_rules!` body (no
//! source position to edit). The context is the innermost construct the comparison sits in
//! (`if condition`, `while condition`, `match guard`, `match scrutinee`, `boolean operand`,
//! `closure body`, `let binding`, `index`, `return value`, `assertion`, `macro argument`
//! or `expression`). A summary `scanned <files> <comparisons>` goes to stderr.

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

/// Every ordering comparison of the non-test code (the `--comparisons` mode)
struct Comparisons<'a> {
    file: &'a str,
    current_fn: String,
    in_macro: bool,
    /// the construct the comparison sits in, innermost last
    ctx: Vec<&'static str>,
    /// the variables the enclosing function assigns to (`x = ..`)
    assigned: BTreeSet<String>,
    /// the names bound from an assigned variable (`match best { Some((_, b)) => .. }`,
    /// `best.is_none_or(|b| ..)`, `if let Some(b) = best`) in the scope being visited
    incumbent: Vec<String>,
    out: Vec<String>,
}

/// The context of a comparison with an operand bound from a variable its function assigns
/// to: the running best of an argmin / argmax, whatever condition shape the result feeds
const INCUMBENT: &str = "incumbent (compares with a binding of an assigned best)";

/// The single-name variables a block assigns to (`x = ..`, also inside closures)
fn assigned_names(b: &Block) -> BTreeSet<String> {
    struct V(BTreeSet<String>);
    impl<'ast> Visit<'ast> for V {
        fn visit_expr_assign(&mut self, a: &'ast syn::ExprAssign) {
            if let Expr::Path(p) = strip(&a.left) {
                if let Some(i) = p.path.get_ident() {
                    self.0.insert(i.to_string());
                }
            }
            visit::visit_expr_assign(self, a);
        }
    }
    let mut v = V(BTreeSet::new());
    v.visit_block(b);
    v.0
}

/// The names a pattern binds
fn pat_names(p: &syn::Pat) -> Vec<String> {
    struct V(Vec<String>);
    impl<'ast> Visit<'ast> for V {
        fn visit_pat_ident(&mut self, i: &'ast syn::PatIdent) {
            self.0.push(i.ident.to_string());
            visit::visit_pat_ident(self, i);
        }
    }
    let mut v = V(Vec::new());
    v.visit_pat(p);
    v.0
}

/// The single-name paths an expression reads
fn expr_names(e: &Expr) -> BTreeSet<String> {
    struct V(BTreeSet<String>);
    impl<'ast> Visit<'ast> for V {
        fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
            if let Some(i) = p.path.get_ident() {
                self.0.insert(i.to_string());
            }
        }
    }
    let mut v = V(BTreeSet::new());
    v.visit_expr(e);
    v.0
}

/// The variable an expression names, under parentheses, references and `.as_ref()`
fn named_var(e: &Expr) -> Option<String> {
    match strip(e) {
        Expr::Path(p) => p.path.get_ident().map(ToString::to_string),
        Expr::MethodCall(m)
            if m.args.is_empty()
                && matches!(m.method.to_string().as_str(), "as_ref" | "as_mut") =>
        {
            named_var(&m.receiver)
        }
        _ => None,
    }
}

const ORDER_METHODS: [&str; 6] = ["lt", "gt", "le", "ge", "partial_cmp", "total_cmp"];

fn is_literal(e: &Expr) -> bool {
    match strip(e) {
        Expr::Lit(_) => true,
        Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => is_literal(&u.expr),
        _ => false,
    }
}

impl Comparisons<'_> {
    fn emit(&mut self, kind: &str, l: &Expr, op: &str, r: &Expr, line: usize) {
        let probe = if self.in_macro {
            String::new()
        } else if is_literal(l) && !is_literal(r) {
            span_of(r)
        } else {
            span_of(l)
        };
        let mut ctx = self.ctx.last().copied().unwrap_or("expression");
        if !self.incumbent.is_empty() {
            let names: BTreeSet<String> = expr_names(l).union(&expr_names(r)).cloned().collect();
            if self.incumbent.iter().any(|n| names.contains(n)) {
                ctx = INCUMBENT;
            }
        }
        self.out.push(format!(
            "{}\t{}\t{}\t{} {} {}\t{}\t{}\t{}",
            self.file,
            self.current_fn,
            kind,
            norm(l),
            op,
            norm(r),
            line,
            probe,
            ctx
        ));
    }

    /// Visit `f` with `c` as the innermost context
    fn within(&mut self, c: &'static str, f: impl FnOnce(&mut Self)) {
        self.ctx.push(c);
        f(self);
        self.ctx.pop();
    }

    /// Visit `f` with `names` bound from an incumbent
    fn binding(&mut self, names: Vec<String>, f: impl FnOnce(&mut Self)) {
        let n = names.len();
        self.incumbent.extend(names);
        f(self);
        self.incumbent.truncate(self.incumbent.len() - n);
    }

    /// Whether `e` names a variable the enclosing function assigns to
    fn is_assigned(&self, e: &Expr) -> bool {
        named_var(e).is_some_and(|v| self.assigned.contains(&v))
    }

    fn enter(
        &mut self,
        name: String,
        attrs: &[syn::Attribute],
        block: Option<&Block>,
        body: impl FnOnce(&mut Self),
    ) {
        if is_cfg_test(attrs) {
            return;
        }
        let saved = std::mem::replace(&mut self.current_fn, name);
        let assigned = block.map(assigned_names).unwrap_or_default();
        let saved_assigned = std::mem::replace(&mut self.assigned, assigned);
        let saved_incumbent = std::mem::take(&mut self.incumbent);
        body(self);
        self.current_fn = saved;
        self.assigned = saved_assigned;
        self.incumbent = saved_incumbent;
    }

    /// The arguments of a macro invocation that parse as comma-separated expressions; the
    /// tokens keep their source positions, so the probe can edit them
    fn visit_macro_args(&mut self, tokens: &TokenStream) {
        let parser = syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated;
        if let Ok(args) = syn::parse::Parser::parse2(parser, tokens.clone()) {
            for a in &args {
                self.visit_expr(a);
            }
        }
    }
}

impl<'ast> Visit<'ast> for Comparisons<'_> {
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
        self.enter(f.sig.ident.to_string(), &f.attrs, Some(&f.block), |s| {
            visit::visit_item_fn(s, f)
        });
    }
    fn visit_impl_item_fn(&mut self, f: &'ast syn::ImplItemFn) {
        self.enter(f.sig.ident.to_string(), &f.attrs, Some(&f.block), |s| {
            visit::visit_impl_item_fn(s, f)
        });
    }
    fn visit_trait_item_fn(&mut self, f: &'ast syn::TraitItemFn) {
        self.enter(f.sig.ident.to_string(), &f.attrs, f.default.as_ref(), |s| {
            visit::visit_trait_item_fn(s, f)
        });
    }
    fn visit_item_const(&mut self, c: &'ast syn::ItemConst) {
        self.enter(format!("const {}", c.ident), &c.attrs, None, |s| {
            visit::visit_item_const(s, c)
        });
    }
    fn visit_item_static(&mut self, c: &'ast syn::ItemStatic) {
        self.enter(format!("static {}", c.ident), &c.attrs, None, |s| {
            visit::visit_item_static(s, c)
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
            // the rule bodies, read as in the rewrite scan; their comparisons have no position
            let tts: Vec<TokenTree> = m.mac.tokens.clone().into_iter().collect();
            let mut i = 0;
            while i < tts.len() {
                if let (
                    Some(TokenTree::Punct(eq)),
                    Some(TokenTree::Punct(gt)),
                    Some(TokenTree::Group(body)),
                ) = (tts.get(i), tts.get(i + 1), tts.get(i + 2))
                {
                    if eq.as_char() == '=' && gt.as_char() == '>' {
                        let flat = demeta(body.stream());
                        let block: TokenStream =
                            TokenTree::Group(Group::new(Delimiter::Brace, flat.clone())).into();
                        let saved =
                            std::mem::replace(&mut self.current_fn, format!("macro {name}"));
                        let was = std::mem::replace(&mut self.in_macro, true);
                        if let Ok(b) = syn::parse2::<Block>(block) {
                            self.visit_block(&b);
                        } else if let Ok(f) = syn::parse2::<syn::File>(flat) {
                            self.visit_file(&f);
                        } else {
                            self.out.push(format!(
                                "{}\tmacro {name}\tmacro-unparsed\t-\t{}\t",
                                self.file,
                                m.mac.path.segments[0].ident.span().start().line
                            ));
                        }
                        self.in_macro = was;
                        self.current_fn = saved;
                        i += 3;
                        continue;
                    }
                }
                i += 1;
            }
            return;
        }
        self.visit_macro_args(&m.mac.tokens);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let name = m
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let c = if name.contains("assert") {
            "assertion"
        } else {
            "macro argument"
        };
        self.within(c, |s| s.visit_macro_args(&m.tokens));
    }
    fn visit_expr_if(&mut self, e: &'ast ExprIf) {
        self.within("if condition", |s| s.visit_expr(&e.cond));
        // `if let Some(b) = best { .. }`: the then branch sees names bound from the incumbent
        let names = match &*e.cond {
            Expr::Let(l) if self.is_assigned(&l.expr) => pat_names(&l.pat),
            _ => Vec::new(),
        };
        self.within("expression", |s| {
            s.binding(names, |s| s.visit_block(&e.then_branch));
            if let Some((_, x)) = &e.else_branch {
                s.visit_expr(x);
            }
        });
    }
    fn visit_expr_while(&mut self, e: &'ast syn::ExprWhile) {
        self.within("while condition", |s| s.visit_expr(&e.cond));
        self.within("expression", |s| s.visit_block(&e.body));
    }
    fn visit_arm(&mut self, a: &'ast syn::Arm) {
        if let Some((_, g)) = &a.guard {
            self.within("match guard", |s| s.visit_expr(g));
        }
        self.within("expression", |s| s.visit_expr(&a.body));
    }
    fn visit_expr_match(&mut self, m: &'ast ExprMatch) {
        self.within("match scrutinee", |s| s.visit_expr(&m.expr));
        // `match best { Some((_, b)) => .. }`: each arm sees the names its pattern binds
        let from_incumbent = self.is_assigned(&m.expr);
        for a in &m.arms {
            let names = if from_incumbent {
                pat_names(&a.pat)
            } else {
                Vec::new()
            };
            self.binding(names, |s| s.visit_arm(a));
        }
    }
    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        self.within("closure body", |s| visit::visit_expr_closure(s, c));
    }
    fn visit_local(&mut self, l: &'ast syn::Local) {
        self.within("let binding", |s| visit::visit_local(s, l));
    }
    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        self.visit_expr(&i.expr);
        self.within("index", |s| s.visit_expr(&i.index));
    }
    fn visit_expr_return(&mut self, r: &'ast syn::ExprReturn) {
        self.within("return value", |s| visit::visit_expr_return(s, r));
    }
    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) {
            self.within("boolean operand", |s| visit::visit_expr_binary(s, b));
            return;
        }
        let op = match b.op {
            BinOp::Lt(_) => Some("<"),
            BinOp::Gt(_) => Some(">"),
            BinOp::Le(_) => Some("<="),
            BinOp::Ge(_) => Some(">="),
            _ => None,
        };
        if let Some(op) = op {
            let line = span_of(&b.left)
                .split(':')
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            self.emit("bin", &b.left, op, &b.right, line);
        }
        visit::visit_expr_binary(self, b);
    }
    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        let name = m.method.to_string();
        if m.args.len() == 1 && ORDER_METHODS.contains(&name.as_str()) {
            let line = m.method.span().start().line;
            self.emit("method", &m.receiver, &name, &m.args[0], line);
        }
        // `best.is_none_or(|b| ..)`: the closure's parameters are bound from the incumbent
        if self.is_assigned(&m.receiver) {
            self.visit_expr(&m.receiver);
            for a in &m.args {
                let names = match strip(a) {
                    Expr::Closure(c) => c.inputs.iter().flat_map(pat_names).collect(),
                    _ => Vec::new(),
                };
                self.binding(names, |s| s.visit_expr(a));
            }
            return;
        }
        visit::visit_expr_method_call(self, m);
    }
    fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
        if let Expr::Path(p) = &*c.func {
            let last = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if c.args.len() == 2
                && p.path.segments.len() >= 2
                && ORDER_METHODS.contains(&last.as_str())
            {
                let line = p.path.segments[0].ident.span().start().line;
                self.emit("path", &c.args[0], &last, &c.args[1], line);
            }
        }
        visit::visit_expr_call(self, c);
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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let comparisons = args.first().is_some_and(|a| a == "--comparisons");
    if comparisons {
        args.remove(0);
    }
    let dirs: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();
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
        if comparisons {
            let mut c = Comparisons {
                file: &rel,
                current_fn: "<module>".to_string(),
                in_macro: false,
                ctx: Vec::new(),
                assigned: BTreeSet::new(),
                incumbent: Vec::new(),
                out: Vec::new(),
            };
            c.visit_file(&ast);
            findings += c.out.len();
            for line in c.out {
                println!("{line}");
            }
            continue;
        }
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
