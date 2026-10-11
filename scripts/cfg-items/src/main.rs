//! Lists the `#[cfg(...)]`-gated nodes of Rust source files with their line spans.
//!
//! `scripts/mutants_in_diff_plan.py` needs to know which features a changed line
//! needs to be compiled. A line scanner loses the attribute when the header of the
//! item spans several lines (a long signature, a `where` clause, a brace on its own
//! line, a multi-line `#[cfg(all(...))]`), so the spans come from `syn` instead.
//!
//! Output: one line per gated node, tab separated:
//! `file  start_line  end_line  kind  name  cfg_tokens`
//! The span starts at the first attribute of the node and ends at its last token.
//! `kind` is `file` for an inner `#![cfg(...)]`, `mod` for a module (with its name),
//! and `item` for every other node (items, impl / trait / foreign items, fields,
//! variants, match arms, `let` statements, expressions, struct literal fields).
//! A file that does not parse is an error (exit 1): the planner then fails instead
//! of guessing.
//!
//! `cfg-items --mutants-skips FILE...` (scripts/mutants_skip_check.py) prints,
//! tab separated, one line per:
//! - `skip  file  line  name  inner  route`: an attribute that is `mutants::skip`
//!   (`::` first allowed) or a `cfg_attr(<pred>, ...)` holding one at any depth.
//!   `inner` is 1 for `#![...]`. `route` is the `cfg(...)` tokens of the node and of
//!   every node around it, joined by U+001E.
//! - `mod  file  line  name  path  inline  route`: a `mod x;` (no body). `path` is its
//!   `#[path = "..."]` value or empty, `inline` the names of the inline `mod a { }`
//!   it sits in, joined by `/`.
//! - `include  file  line  literal  route`: an `include!(...)`; `literal` is empty when
//!   the argument is not one string literal.
//! - `pathinline  file  line  name  path`: a `#[path]` on an inline `mod x { }`, which
//!   moves where the modules declared inside it are looked up (not followed: the
//!   checker refuses it).
//! - `tokens  file  count`: the `mutants :: skip` token runs anywhere in the file
//!   (attributes, macro bodies, all token trees), for the checker to compare with the
//!   skips it was given.

use proc_macro2::Span;
use std::fmt::Write as _;
use std::process::ExitCode;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, ForeignItem, ImplItem, Item, Meta, TraitItem};

struct Collector<'a> {
    file: &'a str,
    out: String,
}

fn cfgs(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("cfg"))
        .filter_map(|a| match &a.meta {
            Meta::List(list) => Some(list.tokens.to_string()),
            _ => None,
        })
        .collect()
}

impl Collector<'_> {
    fn record(&mut self, attrs: &[Attribute], span: Span, kind: &str, name: &str) {
        let (start, end) = (span.start().line, span.end().line);
        for cfg in cfgs(attrs) {
            let _ = writeln!(self.out, "{}\t{start}\t{end}\t{kind}\t{name}\t{cfg}", self.file);
        }
    }
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(i) => &i.attrs,
        ImplItem::Fn(i) => &i.attrs,
        ImplItem::Type(i) => &i.attrs,
        ImplItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn trait_item_attrs(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(i) => &i.attrs,
        TraitItem::Fn(i) => &i.attrs,
        TraitItem::Type(i) => &i.attrs,
        TraitItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn foreign_item_attrs(item: &ForeignItem) -> &[Attribute] {
    match item {
        ForeignItem::Fn(i) => &i.attrs,
        ForeignItem::Static(i) => &i.attrs,
        ForeignItem::Type(i) => &i.attrs,
        ForeignItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn expr_attrs(e: &Expr) -> &[Attribute] {
    macro_rules! attrs {
        ($($v:ident),*) => {
            match e { $(Expr::$v(x) => &x.attrs,)* _ => &[] }
        };
    }
    attrs!(
        Array, Assign, Async, Await, Binary, Block, Break, Call, Cast, Closure, Const, Continue,
        Field, ForLoop, Group, If, Index, Infer, Let, Lit, Loop, Macro, Match, MethodCall, Paren,
        Path, Range, RawAddr, Reference, Repeat, Return, Struct, Try, TryBlock, Tuple, Unary,
        Unsafe, While, Yield
    )
}

impl<'ast> Visit<'ast> for Collector<'_> {
    fn visit_file(&mut self, f: &'ast syn::File) {
        let last = f.items.last().map_or(1, |i| i.span().end().line);
        for cfg in cfgs(&f.attrs) {
            let _ = writeln!(self.out, "{}\t1\t{last}\tfile\t\t{cfg}", self.file);
        }
        visit::visit_file(self, f);
    }
    fn visit_item(&mut self, i: &'ast Item) {
        match i {
            Item::Mod(m) => self.record(&m.attrs, i.span(), "mod", &m.ident.to_string()),
            _ => self.record(item_attrs(i), i.span(), "item", ""),
        }
        visit::visit_item(self, i);
    }
    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        self.record(impl_item_attrs(i), i.span(), "item", "");
        visit::visit_impl_item(self, i);
    }
    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        self.record(trait_item_attrs(i), i.span(), "item", "");
        visit::visit_trait_item(self, i);
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) {
        self.record(foreign_item_attrs(i), i.span(), "item", "");
        visit::visit_foreign_item(self, i);
    }
    fn visit_field(&mut self, f: &'ast syn::Field) {
        self.record(&f.attrs, f.span(), "item", "");
        visit::visit_field(self, f);
    }
    fn visit_variant(&mut self, v: &'ast syn::Variant) {
        self.record(&v.attrs, v.span(), "item", "");
        visit::visit_variant(self, v);
    }
    fn visit_arm(&mut self, a: &'ast syn::Arm) {
        self.record(&a.attrs, a.span(), "item", "");
        visit::visit_arm(self, a);
    }
    fn visit_local(&mut self, l: &'ast syn::Local) {
        self.record(&l.attrs, l.span(), "item", "");
        visit::visit_local(self, l);
    }
    fn visit_stmt_macro(&mut self, m: &'ast syn::StmtMacro) {
        self.record(&m.attrs, m.span(), "item", "");
        visit::visit_stmt_macro(self, m);
    }
    fn visit_field_value(&mut self, f: &'ast syn::FieldValue) {
        self.record(&f.attrs, f.span(), "item", "");
        visit::visit_field_value(self, f);
    }
    fn visit_expr(&mut self, e: &'ast Expr) {
        self.record(expr_attrs(e), e.span(), "item", "");
        visit::visit_expr(self, e);
    }
}

/// `--mutants-skips`: see the module doc.
mod skips {
    use proc_macro2::{TokenStream, TokenTree};
    use std::fmt::Write as _;
    use syn::punctuated::Punctuated;
    use syn::spanned::Spanned;
    use syn::visit::{self, Visit};
    use syn::{AttrStyle, Attribute, Expr, Item, Lit, Macro, Meta, Token};

    const SEP: char = '\u{1e}';

    pub fn is_skip(meta: &Meta) -> bool {
        let path = meta.path();
        let names: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
        if matches!(meta, Meta::Path(_)) && names == ["mutants", "skip"] {
            return true;
        }
        if let Meta::List(list) = meta {
            if path.is_ident("cfg_attr") {
                if let Ok(args) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated) {
                    return args.iter().skip(1).any(is_skip);
                }
            }
        }
        false
    }

    /// `mutants :: skip` runs in a token stream, at any depth.
    pub fn count_tokens(ts: TokenStream) -> usize {
        let tts: Vec<TokenTree> = ts.into_iter().collect();
        let mut n = 0;
        for (k, tt) in tts.iter().enumerate() {
            match tt {
                TokenTree::Group(g) => n += count_tokens(g.stream()),
                TokenTree::Ident(i) if i == "mutants" => {
                    let colon = |j: usize| matches!(tts.get(j), Some(TokenTree::Punct(p)) if p.as_char() == ':');
                    let skip = matches!(tts.get(k + 3), Some(TokenTree::Ident(s)) if s == "skip");
                    if colon(k + 1) && colon(k + 2) && skip {
                        n += 1;
                    }
                }
                _ => {}
            }
        }
        n
    }

    fn cfgs(attrs: &[Attribute]) -> Vec<String> {
        attrs
            .iter()
            .filter(|a| a.path().is_ident("cfg"))
            .filter_map(|a| match &a.meta {
                Meta::List(list) => Some(list.tokens.to_string()),
                _ => None,
            })
            .collect()
    }

    pub struct Collector<'a> {
        pub file: &'a str,
        pub out: String,
        route: Vec<String>,
        inline: Vec<String>,
    }

    impl<'a> Collector<'a> {
        pub fn new(file: &'a str) -> Self {
            Collector { file, out: String::new(), route: Vec::new(), inline: Vec::new() }
        }

        fn route_with(&self, own: &[Attribute]) -> String {
            let mut r = self.route.clone();
            r.extend(cfgs(own));
            r.join(&SEP.to_string())
        }

        fn skips(&mut self, attrs: &[Attribute], name: &str) {
            for a in attrs.iter().filter(|a| is_skip(&a.meta)) {
                let inner = u8::from(matches!(a.style, AttrStyle::Inner(_)));
                let line = a.span().start().line;
                let route = self.route_with(attrs);
                let _ = writeln!(self.out, "skip\t{}\t{line}\t{name}\t{inner}\t{route}", self.file);
            }
        }

        fn include(&mut self, mac: &Macro, attrs: &[Attribute]) {
            if !mac.path.is_ident("include") {
                return;
            }
            let literal = match mac.parse_body::<Lit>() {
                Ok(Lit::Str(s)) => s.value(),
                _ => String::new(),
            };
            let line = mac.span().start().line;
            let route = self.route_with(attrs);
            let _ = writeln!(self.out, "include\t{}\t{line}\t{literal}\t{route}", self.file);
        }

        fn enter(&mut self, attrs: &[Attribute], name: &str) -> usize {
            self.skips(attrs, name);
            let before = self.route.len();
            self.route.extend(cfgs(attrs));
            before
        }
    }

    fn path_attr(attrs: &[Attribute]) -> String {
        attrs
            .iter()
            .filter(|a| a.path().is_ident("path"))
            .find_map(|a| match &a.meta {
                Meta::NameValue(nv) => match &nv.value {
                    Expr::Lit(l) => match &l.lit {
                        Lit::Str(s) => Some(s.value()),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or_default()
    }

    impl<'ast> Visit<'ast> for Collector<'_> {
        fn visit_file(&mut self, f: &'ast syn::File) {
            let before = self.enter(&f.attrs, "file");
            visit::visit_file(self, f);
            self.route.truncate(before);
        }
        fn visit_item(&mut self, i: &'ast Item) {
            let (attrs, name): (&[Attribute], String) = match i {
                Item::Fn(f) => (&f.attrs, f.sig.ident.to_string()),
                Item::Mod(m) => (&m.attrs, m.ident.to_string()),
                Item::Impl(x) => (&x.attrs, "impl".to_string()),
                Item::Trait(x) => (&x.attrs, x.ident.to_string()),
                Item::Const(x) => (&x.attrs, x.ident.to_string()),
                Item::Static(x) => (&x.attrs, x.ident.to_string()),
                Item::Struct(x) => (&x.attrs, x.ident.to_string()),
                Item::Enum(x) => (&x.attrs, x.ident.to_string()),
                Item::Macro(x) => (&x.attrs, "macro".to_string()),
                Item::ForeignMod(x) => (&x.attrs, "extern".to_string()),
                Item::Use(x) => (&x.attrs, "use".to_string()),
                Item::Type(x) => (&x.attrs, x.ident.to_string()),
                Item::Union(x) => (&x.attrs, x.ident.to_string()),
                Item::ExternCrate(x) => (&x.attrs, x.ident.to_string()),
                Item::TraitAlias(x) => (&x.attrs, x.ident.to_string()),
                _ => (&[], String::new()),
            };
            let before = self.enter(attrs, &name);
            match i {
                Item::Mod(m) if m.content.is_none() => {
                    let line = m.mod_token.span.start().line;
                    let route = self.route.join(&SEP.to_string());
                    let _ = writeln!(
                        self.out,
                        "mod\t{}\t{line}\t{}\t{}\t{}\t{route}",
                        self.file,
                        m.ident,
                        path_attr(&m.attrs),
                        self.inline.join("/")
                    );
                }
                Item::Mod(m) => {
                    let path = path_attr(&m.attrs);
                    if !path.is_empty() {
                        let line = m.mod_token.span.start().line;
                        let _ = writeln!(self.out, "pathinline\t{}\t{line}\t{}\t{path}", self.file, m.ident);
                    }
                    self.inline.push(m.ident.to_string());
                    visit::visit_item(self, i);
                    self.inline.pop();
                    self.route.truncate(before);
                    return;
                }
                Item::Macro(m) => self.include(&m.mac, &m.attrs),
                _ => {}
            }
            visit::visit_item(self, i);
            self.route.truncate(before);
        }
        fn visit_impl_item(&mut self, i: &'ast syn::ImplItem) {
            let (attrs, name): (&[Attribute], String) = match i {
                syn::ImplItem::Fn(f) => (&f.attrs, f.sig.ident.to_string()),
                syn::ImplItem::Const(c) => (&c.attrs, c.ident.to_string()),
                syn::ImplItem::Type(t) => (&t.attrs, t.ident.to_string()),
                syn::ImplItem::Macro(m) => (&m.attrs, "macro".to_string()),
                _ => (&[], String::new()),
            };
            let before = self.enter(attrs, &name);
            visit::visit_impl_item(self, i);
            self.route.truncate(before);
        }
        fn visit_trait_item(&mut self, i: &'ast syn::TraitItem) {
            let (attrs, name): (&[Attribute], String) = match i {
                syn::TraitItem::Fn(f) => (&f.attrs, f.sig.ident.to_string()),
                syn::TraitItem::Const(c) => (&c.attrs, c.ident.to_string()),
                syn::TraitItem::Type(t) => (&t.attrs, t.ident.to_string()),
                syn::TraitItem::Macro(m) => (&m.attrs, "macro".to_string()),
                _ => (&[], String::new()),
            };
            let before = self.enter(attrs, &name);
            visit::visit_trait_item(self, i);
            self.route.truncate(before);
        }
        fn visit_foreign_item(&mut self, i: &'ast syn::ForeignItem) {
            let attrs: &[Attribute] = match i {
                syn::ForeignItem::Fn(f) => &f.attrs,
                syn::ForeignItem::Static(s) => &s.attrs,
                syn::ForeignItem::Type(t) => &t.attrs,
                syn::ForeignItem::Macro(m) => &m.attrs,
                _ => &[],
            };
            let before = self.enter(attrs, "foreign");
            visit::visit_foreign_item(self, i);
            self.route.truncate(before);
        }
        fn visit_stmt_macro(&mut self, m: &'ast syn::StmtMacro) {
            self.skips(&m.attrs, "?");
            self.include(&m.mac, &m.attrs);
            visit::visit_stmt_macro(self, m);
        }
        fn visit_expr_macro(&mut self, m: &'ast syn::ExprMacro) {
            self.skips(&m.attrs, "?");
            self.include(&m.mac, &m.attrs);
            visit::visit_expr_macro(self, m);
        }
        fn visit_local(&mut self, l: &'ast syn::Local) {
            self.skips(&l.attrs, "?");
            visit::visit_local(self, l);
        }
        fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
            self.skips(&c.attrs, "?");
            visit::visit_expr_closure(self, c);
        }
        fn visit_field(&mut self, f: &'ast syn::Field) {
            self.skips(&f.attrs, "?");
            visit::visit_field(self, f);
        }
        fn visit_variant(&mut self, v: &'ast syn::Variant) {
            self.skips(&v.attrs, "?");
            visit::visit_variant(self, v);
        }
        fn visit_arm(&mut self, a: &'ast syn::Arm) {
            self.skips(&a.attrs, "?");
            visit::visit_arm(self, a);
        }
        fn visit_field_value(&mut self, f: &'ast syn::FieldValue) {
            self.skips(&f.attrs, "?");
            visit::visit_field_value(self, f);
        }
    }

    /// The whole `--mutants-skips` output for one parsed file.
    pub fn report(path: &str, src: &str, file: &syn::File) -> String {
        let mut c = Collector::new(path);
        c.visit_file(file);
        let tokens = src.parse::<TokenStream>().map(count_tokens).unwrap_or(usize::MAX);
        let _ = writeln!(c.out, "tokens\t{path}\t{tokens}");
        c.out
    }
}

fn main() -> ExitCode {
    let mut failed = false;
    let mut out = String::new();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mutants_skips = args.first().is_some_and(|a| a == "--mutants-skips");
    if mutants_skips {
        args.remove(0);
    }
    for path in args {
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cfg-items: {path}: {e}");
                failed = true;
                continue;
            }
        };
        match syn::parse_file(&src) {
            Ok(file) if mutants_skips => out.push_str(&skips::report(&path, &src, &file)),
            Ok(file) => {
                let mut c = Collector { file: &path, out: String::new() };
                c.visit_file(&file);
                out.push_str(&c.out);
            }
            Err(e) => {
                let at = e.span().start();
                eprintln!("cfg-items: {path}:{}:{}: {e}", at.line, at.column + 1);
                failed = true;
            }
        }
    }
    print!("{out}");
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
