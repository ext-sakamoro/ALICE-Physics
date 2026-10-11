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

fn main() -> ExitCode {
    let mut failed = false;
    let mut out = String::new();
    for path in std::env::args().skip(1) {
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cfg-items: {path}: {e}");
                failed = true;
                continue;
            }
        };
        match syn::parse_file(&src) {
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
