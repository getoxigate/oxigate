// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 OxiGate contributors
//! Provider usage-site scan.
//!
//! Each provider wire shape turns its upstream usage payload into the domain `Usage` in exactly
//! one place — a projection taking the lane's accounting contract as a parameter. The
//! OpenAI-shaped lanes are the exception by design: their wire schema *is* the domain schema, so
//! they deserialize straight into `Usage` and normalize it once. This scan keeps new construction
//! and deserialization sites from appearing under `src/providers/` unnoticed, by holding each
//! file's production region to an exact count of each kind, recorded in [`ALLOWANCES`].
//!
//! It polices *sites*, not behaviour. It cannot see through `serde` into a struct field, so it
//! cannot prove a parsed usage was normalized; the providers' reachability tests prove that, and
//! only for the entry points they exercise.
//!
//! What counts, matched on a path's final segment so `ConverseUsage`, `TokenUsage` and friends
//! never match while `crate::domain::chat::Usage::default()` does. Inside an impl whose self type
//! is `Usage` or `ChatResponse`, `Self` is read as that type:
//!
//! * **Construction** — a `Usage { .. }` literal, or `Usage::default` called or passed as a
//!   function.
//! * **Deserialization** — `Usage` or `ChatResponse` in a parse position: a `let` ascribed with
//!   either (inference carries the ascription back into any unannotated parse that feeds it, so
//!   only `= None` is exempt), a turbofish naming either, or a field naming either on a locally
//!   declared type that derives `Deserialize` — directly, or through a `cfg_attr` that is not
//!   test-only. Signatures are not counted; they neither construct nor parse.
//! * **Rename** — always a failure, never an allowance. `syn` reads names without resolving them,
//!   so `use … Usage as U;` or `type U = Usage;` would hide every later site from the matchers.
//!   The `Deserialize` derive is matched by name too, so `use serde::Deserialize as De;` is refused
//!   for the same reason: `#[derive(De)]` would hide its type's fields.
//!
//! Nothing else is a site. `syn` sees syntax, not types, so whatever reaches `Usage` only through
//! inference goes unseen, and some of it carries real token counts:
//!
//! * a parse whose target type is inferred rather than written at the parse — from the function's
//!   return type, the field or argument the result is assigned or passed to
//!   (`chunk.usage = serde_json::from_value(v).ok()`), or a closure parameter's annotation;
//! * a `Usage` produced without naming it — `Default::default()` or `.unwrap_or_default()` in a
//!   `Usage` position, `std::mem::take`, or a call to any function that returns one;
//! * a type whose `Deserialize` is implemented by hand rather than derived.
//!
//! Macro bodies are parsed and scanned like any other code whenever they are ordinary Rust, which
//! is what matters here: every streaming lane runs inside `async_stream::stream! { .. }`. A body
//! that is not — a `tracing` field list, `json!` — is scanned for constructions by token shape
//! only.
//!
//! Test-only code is not counted, wherever the `cfg` sits — on an item, a statement, an
//! expression, a match arm or a field. A `cfg` predicate is evaluated in three-valued logic — `test` is
//! false, every other atom unknown — and only a definitely-false predicate removes code from the
//! production region. `cfg(any(test, feature = "x"))` and `cfg(not(feature = "x"))` both compile
//! into some production build, so both are scanned. The same rule applies to out-of-line module
//! files: a file is not scanned only when every `mod` declaration reaching it is test-only.

use anyhow::{Context, Result, bail};
use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
use syn::{
    AngleBracketedGenericArguments, Arm, Attribute, Block, Expr, ExprMethodCall, ExprPath,
    ExprStruct, FieldValue, Fields, GenericArgument, ImplItem, Item, ItemEnum, ItemStruct,
    ItemType, ItemUse, Local, LocalInit, Macro, Meta, Pat, PathArguments, StmtMacro, Token,
    TraitItem, Type, UseTree,
    ext::IdentExt,
    punctuated::Punctuated,
    spanned::Spanned,
    visit::{self, Visit},
};

/// The domain types whose parse positions are counted and whose renaming is refused.
const DOMAIN_TYPES: [&str; 2] = ["Usage", "ChatResponse"];

/// How many sites of each kind a file's production region holds.
struct Allowance {
    /// Path under `src/providers/`.
    file: &'static str,
    construction: usize,
    deserialization: usize,
    /// What each allowed site is, so a stale entry can be told from a live one.
    reason: &'static str,
}

/// Exact per-file counts; every file not listed is held to zero.
///
/// Exact rather than a ceiling, in both directions. A count above its allowance is a new site that
/// should have gone through the wire shape's projection. A count below it is either a site that
/// was removed — so the entry is stale and would silently admit a replacement — or a matcher that
/// stopped seeing the syntax in use, which would otherwise pass the gate vacuously.
const ALLOWANCES: &[Allowance] = &[
    Allowance {
        file: "anthropic/translate.rs",
        construction: 1,
        deserialization: 0,
        reason: "the wire shape's projection",
    },
    Allowance {
        file: "bedrock/translate.rs",
        construction: 1,
        deserialization: 0,
        reason: "the wire shape's projection",
    },
    Allowance {
        file: "gemini/translate.rs",
        construction: 1,
        deserialization: 0,
        reason: "the wire shape's projection",
    },
    Allowance {
        file: "azure/mod.rs",
        construction: 1,
        deserialization: 1,
        reason: "missing-usage fallback; the lane's response DTO",
    },
    Allowance {
        file: "openai_compat/mod.rs",
        construction: 1,
        deserialization: 1,
        reason: "missing-usage fallback; the lane's response DTO",
    },
    Allowance {
        file: "openai/mod.rs",
        construction: 0,
        deserialization: 1,
        reason: "the lane's response parse",
    },
    Allowance {
        file: "openai_compat/sse.rs",
        construction: 0,
        deserialization: 1,
        reason: "the streaming usage parse",
    },
];

/// Runs the scan over `src/providers/` and fails on any count that differs from its allowance.
pub fn run() -> Result<()> {
    let providers = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/providers");
    let excluded = test_only_modules(&providers.join("mod.rs"), &|path: &Path| {
        fs::read_to_string(path).ok()
    });

    let mut files = Vec::new();
    rust_files(&providers, &mut files)?;
    files.sort();

    let mut report = Vec::new();
    let mut scanned = BTreeSet::new();
    for path in files {
        let rel = path
            .strip_prefix(&providers)
            .with_context(|| format!("{} is outside the provider tree", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        let sites = if excluded.contains(&path) {
            Vec::new()
        } else {
            let src = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            scan_source(&src).with_context(|| format!("failed to parse src/providers/{rel}"))?
        };
        report.extend(violations(&rel, &sites));
        scanned.insert(rel);
    }
    // An allowance for a file that no longer exists is stale in the same way as one above its
    // file's count.
    for allowance in ALLOWANCES.iter().filter(|a| !scanned.contains(a.file)) {
        report.extend(violations(allowance.file, &[]));
    }

    if report.is_empty() {
        println!("xtask usage-scan: provider usage sites match their allowance");
        return Ok(());
    }
    for line in &report {
        eprintln!("{line}");
    }
    bail!("usage-scan: provider usage sites differ from their allowance")
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rule {
    Construction,
    Deserialization,
    Rename,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::Construction => "construction",
            Rule::Deserialization => "deserialization",
            Rule::Rename => "rename",
        }
    }
}

/// One counted site in a file's production region.
#[derive(Debug)]
struct Site {
    line: usize,
    rule: Rule,
    detail: &'static str,
}

/// Compares one file's sites against its allowance and describes every mismatch, naming the
/// file, the line and the rule.
fn violations(file: &str, sites: &[Site]) -> Vec<String> {
    let allowance = ALLOWANCES.iter().find(|a| a.file == file);
    let reason = allowance.map_or("no allowance", |a| a.reason);
    let mut report = Vec::new();

    for (rule, allowed) in [
        (Rule::Construction, allowance.map_or(0, |a| a.construction)),
        (
            Rule::Deserialization,
            allowance.map_or(0, |a| a.deserialization),
        ),
    ] {
        let found: Vec<&Site> = sites.iter().filter(|s| s.rule == rule).collect();
        if found.len() == allowed {
            continue;
        }
        let advice = if found.len() > allowed {
            "reuse the wire shape's existing projection or parse instead of adding a site"
        } else {
            "a site was removed, so lower the allowance to match — unless it was rewritten into a \
             form the scan cannot see, which is a site still in place: restore a counted form"
        };
        report.push(format!(
            "src/providers/{file}: {} {} site(s), allowance {allowed} ({reason}) — {advice}",
            found.len(),
            rule.name(),
        ));
        for site in found {
            report.push(format!(
                "  src/providers/{file}:{}: {}: {}",
                site.line,
                rule.name(),
                site.detail
            ));
        }
    }

    for site in sites.iter().filter(|s| s.rule == Rule::Rename) {
        report.push(format!(
            "src/providers/{file}:{}: rename: {} — refer to the type by its own name",
            site.line, site.detail
        ));
    }
    report
}

// ---------------------------------------------------------------------------
// Source scan
// ---------------------------------------------------------------------------

/// Parses one file and returns the sites in its production region, in line order.
fn scan_source(src: &str) -> Result<Vec<Site>> {
    let file = syn::parse_file(src)?;
    if is_test_only(&file.attrs) {
        return Ok(Vec::new());
    }
    let mut scanner = Scanner::default();
    scanner.visit_file(&file);
    scanner.sites.sort_by_key(|s| s.line);
    Ok(scanner.sites)
}

#[derive(Default)]
struct Scanner {
    sites: Vec<Site>,
    /// The domain type `Self` stands for, inside an impl whose self type is one.
    self_type: Option<&'static str>,
}

impl Scanner {
    fn push(&mut self, span: Span, rule: Rule, detail: &'static str) {
        self.sites.push(Site {
            line: span.start().line,
            rule,
            detail,
        });
    }

    /// Whether `ident` names the domain `Usage`, reading `Self` as the enclosing impl's self type.
    fn is_usage(&self, ident: &syn::Ident) -> bool {
        ident == "Usage" || (ident == "Self" && self.self_type == Some("Usage"))
    }

    fn is_usage_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Path(p) if p.qself.is_none()
            && p.path.segments.last().is_some_and(|s| self.is_usage(&s.ident)))
    }

    /// Fields of a `Deserialize` type that name a domain type are parse positions.
    fn count_parsed_fields(&mut self, fields: &Fields) {
        for field in fields.iter().filter(|f| !is_test_only(&f.attrs)) {
            if names_domain_type(&field.ty, self.self_type) {
                self.push(
                    field.ty.span(),
                    Rule::Deserialization,
                    "field of a `Deserialize` type names the domain type",
                );
            }
        }
    }

    fn check_use_tree(&mut self, tree: &UseTree) {
        match tree {
            UseTree::Path(path) => self.check_use_tree(&path.tree),
            UseTree::Group(group) => group.items.iter().for_each(|t| self.check_use_tree(t)),
            UseTree::Rename(rename) if is_domain_ident(&rename.ident) => self.push(
                rename.ident.span(),
                Rule::Rename,
                "`use … as …` renames the domain type",
            ),
            // `as _` binds no name, so no derive can use it.
            UseTree::Rename(rename) if rename.ident == "Deserialize" && rename.rename != "_" => {
                self.push(
                    rename.ident.span(),
                    Rule::Rename,
                    "`use … as …` renames the `Deserialize` derive",
                )
            }
            _ => {}
        }
    }

    /// The fallback for a macro body that does not parse as Rust: constructions are found by
    /// token shape — `Usage` followed by a brace group, or by `::default`.
    fn scan_macro_tokens(&mut self, tokens: TokenStream) {
        let trees: Vec<TokenTree> = tokens.into_iter().collect();
        for (i, tree) in trees.iter().enumerate() {
            match tree {
                TokenTree::Group(group) => self.scan_macro_tokens(group.stream()),
                TokenTree::Ident(ident) if self.is_usage(ident) => {
                    let literal = matches!(
                        trees.get(i + 1),
                        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace
                    );
                    let default = matches!(
                        (trees.get(i + 1), trees.get(i + 2), trees.get(i + 3)),
                        (Some(TokenTree::Punct(a)), Some(TokenTree::Punct(b)), Some(TokenTree::Ident(d)))
                            if a.as_char() == ':' && b.as_char() == ':' && d == "default"
                    );
                    if literal || default {
                        self.push(
                            ident.span(),
                            Rule::Construction,
                            "`Usage` constructed inside a macro",
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for Scanner {
    fn visit_item(&mut self, item: &'ast Item) {
        if is_test_only(item_attrs(item)) {
            return;
        }
        // `Self` reaches an impl's own items and no further: any nested item — even one declared
        // inside a method body — is a scope where the enclosing impl's `Self` is not in reach.
        let self_type = match item {
            Item::Impl(imp) => domain_type_of(&imp.self_ty),
            _ => None,
        };
        let outer = std::mem::replace(&mut self.self_type, self_type);
        visit::visit_item(self, item);
        self.self_type = outer;
    }

    fn visit_impl_item(&mut self, item: &'ast ImplItem) {
        let attrs: &[Attribute] = match item {
            ImplItem::Const(i) => &i.attrs,
            ImplItem::Fn(i) => &i.attrs,
            ImplItem::Type(i) => &i.attrs,
            ImplItem::Macro(i) => &i.attrs,
            _ => &[],
        };
        if !is_test_only(attrs) {
            visit::visit_impl_item(self, item);
        }
    }

    fn visit_trait_item(&mut self, item: &'ast TraitItem) {
        let attrs: &[Attribute] = match item {
            TraitItem::Const(i) => &i.attrs,
            TraitItem::Fn(i) => &i.attrs,
            TraitItem::Type(i) => &i.attrs,
            TraitItem::Macro(i) => &i.attrs,
            _ => &[],
        };
        if !is_test_only(attrs) {
            visit::visit_trait_item(self, item);
        }
    }

    fn visit_stmt_macro(&mut self, stmt: &'ast StmtMacro) {
        if !is_test_only(&stmt.attrs) {
            visit::visit_stmt_macro(self, stmt);
        }
    }

    /// Covers an attributed expression statement and every other position an expression can
    /// carry a `cfg` — array elements, call arguments, a block.
    fn visit_expr(&mut self, expr: &'ast Expr) {
        if !is_test_only(expr_attrs(expr)) {
            visit::visit_expr(self, expr);
        }
    }

    fn visit_arm(&mut self, arm: &'ast Arm) {
        if !is_test_only(&arm.attrs) {
            visit::visit_arm(self, arm);
        }
    }

    fn visit_field_value(&mut self, field: &'ast FieldValue) {
        if !is_test_only(&field.attrs) {
            visit::visit_field_value(self, field);
        }
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if is_test_only(&local.attrs) {
            return;
        }
        // Inference flows backwards, so an ascription steers any earlier unannotated parse whose
        // result it receives — `let usage: Usage = parsed;` — and is counted whatever its
        // initializer, or with none at all. The one exception is `= None`, which receives
        // nothing.
        if let Pat::Type(typed) = &local.pat
            && names_domain_type(&typed.ty, self.self_type)
            && !local.init.as_ref().is_some_and(is_none)
        {
            self.push(
                typed.ty.span(),
                Rule::Deserialization,
                "`let` ascribed with the domain type",
            );
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_struct(&mut self, expr: &'ast ExprStruct) {
        if let Some(last) = expr.path.segments.last()
            && self.is_usage(&last.ident)
        {
            self.push(
                last.ident.span(),
                Rule::Construction,
                "`Usage { .. }` literal",
            );
        }
        visit::visit_expr_struct(self, expr);
    }

    fn visit_expr_path(&mut self, expr: &'ast ExprPath) {
        let segments = &expr.path.segments;
        let is_default = segments.last().is_some_and(|s| s.ident == "default");
        let owner = segments.len().checked_sub(2).map(|i| &segments[i].ident);

        if is_default && expr.qself.is_none() && owner.is_some_and(|o| self.is_usage(o)) {
            let span = owner.map_or_else(|| expr.span(), |o| o.span());
            self.push(span, Rule::Construction, "`Usage::default`");
        } else if is_default
            && expr
                .qself
                .as_ref()
                .is_some_and(|q| self.is_usage_type(&q.ty))
        {
            self.push(
                expr.span(),
                Rule::Construction,
                "`<Usage as Default>::default`",
            );
        } else if expr
            .qself
            .as_ref()
            .is_some_and(|q| names_domain_type(&q.ty, self.self_type))
            || segments.iter().any(|s| match &s.arguments {
                PathArguments::AngleBracketed(args) => {
                    generics_name_domain_type(args, self.self_type)
                }
                _ => false,
            })
        {
            self.push(
                expr.span(),
                Rule::Deserialization,
                "path generics name the domain type",
            );
        }
        visit::visit_expr_path(self, expr);
    }

    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if let Some(turbofish) = &call.turbofish
            && generics_name_domain_type(turbofish, self.self_type)
        {
            self.push(
                turbofish.span(),
                Rule::Deserialization,
                "turbofish names the domain type",
            );
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_item_struct(&mut self, item: &'ast ItemStruct) {
        if derives_deserialize(&item.attrs) {
            self.count_parsed_fields(&item.fields);
        }
        visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast ItemEnum) {
        if derives_deserialize(&item.attrs) {
            for variant in item.variants.iter().filter(|v| !is_test_only(&v.attrs)) {
                self.count_parsed_fields(&variant.fields);
            }
        }
        visit::visit_item_enum(self, item);
    }

    fn visit_item_use(&mut self, item: &'ast ItemUse) {
        self.check_use_tree(&item.tree);
    }

    fn visit_item_type(&mut self, item: &'ast ItemType) {
        if names_domain_type(&item.ty, self.self_type) {
            self.push(
                item.ident.span(),
                Rule::Rename,
                "type alias of the domain type",
            );
        }
        visit::visit_item_type(self, item);
    }

    /// `syn` leaves macro bodies unparsed, and every streaming lane's body lives inside one —
    /// `async_stream::stream! { .. }`. A body that parses as statements or as comma-separated
    /// expressions is scanned like any other code; only one that is neither, such as a `tracing`
    /// field list, falls back to token shape, which finds constructions alone.
    fn visit_macro(&mut self, mac: &'ast Macro) {
        if let Ok(stmts) = mac.parse_body_with(Block::parse_within) {
            stmts.iter().for_each(|stmt| self.visit_stmt(stmt));
        } else if let Ok(exprs) =
            mac.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
        {
            exprs.iter().for_each(|expr| self.visit_expr(expr));
        } else {
            self.scan_macro_tokens(mac.tokens.clone());
        }
        visit::visit_macro(self, mac);
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
        // Anything unrecognized is scanned: an unknown item is never assumed test-only.
        _ => &[],
    }
}

/// An expression's outer attributes; `syn` has no accessor for them.
fn expr_attrs(expr: &Expr) -> &[Attribute] {
    match expr {
        Expr::Array(e) => &e.attrs,
        Expr::Assign(e) => &e.attrs,
        Expr::Async(e) => &e.attrs,
        Expr::Await(e) => &e.attrs,
        Expr::Binary(e) => &e.attrs,
        Expr::Block(e) => &e.attrs,
        Expr::Break(e) => &e.attrs,
        Expr::Call(e) => &e.attrs,
        Expr::Cast(e) => &e.attrs,
        Expr::Closure(e) => &e.attrs,
        Expr::Const(e) => &e.attrs,
        Expr::Continue(e) => &e.attrs,
        Expr::Field(e) => &e.attrs,
        Expr::ForLoop(e) => &e.attrs,
        Expr::Group(e) => &e.attrs,
        Expr::If(e) => &e.attrs,
        Expr::Index(e) => &e.attrs,
        Expr::Infer(e) => &e.attrs,
        Expr::Let(e) => &e.attrs,
        Expr::Lit(e) => &e.attrs,
        Expr::Loop(e) => &e.attrs,
        Expr::Macro(e) => &e.attrs,
        Expr::Match(e) => &e.attrs,
        Expr::MethodCall(e) => &e.attrs,
        Expr::Paren(e) => &e.attrs,
        Expr::Path(e) => &e.attrs,
        Expr::Range(e) => &e.attrs,
        Expr::RawAddr(e) => &e.attrs,
        Expr::Reference(e) => &e.attrs,
        Expr::Repeat(e) => &e.attrs,
        Expr::Return(e) => &e.attrs,
        Expr::Struct(e) => &e.attrs,
        Expr::Try(e) => &e.attrs,
        Expr::TryBlock(e) => &e.attrs,
        Expr::Tuple(e) => &e.attrs,
        Expr::Unary(e) => &e.attrs,
        Expr::Unsafe(e) => &e.attrs,
        Expr::While(e) => &e.attrs,
        Expr::Yield(e) => &e.attrs,
        // `Verbatim` carries no attributes, and an unrecognized expression is never assumed
        // test-only.
        _ => &[],
    }
}

fn is_domain_ident(ident: &syn::Ident) -> bool {
    DOMAIN_TYPES.iter().any(|name| ident == name)
}

/// The domain type an impl's self type names, if it names one.
fn domain_type_of(ty: &Type) -> Option<&'static str> {
    let Type::Path(p) = ty else { return None };
    let last = p.path.segments.last().filter(|_| p.qself.is_none())?;
    DOMAIN_TYPES.into_iter().find(|name| last.ident == name)
}

/// Whether any path inside `ty` ends in a domain type — `Usage`, `Option<Usage>`,
/// `Result<ChatResponse, E>`, `crate::domain::chat::Usage` — or is `Self` inside an impl for one.
fn names_domain_type(ty: &Type, self_type: Option<&'static str>) -> bool {
    struct Names {
        found: bool,
        self_is_domain: bool,
    }
    impl<'ast> Visit<'ast> for Names {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            if path.segments.last().is_some_and(|s| {
                is_domain_ident(&s.ident) || (self.self_is_domain && s.ident == "Self")
            }) {
                self.found = true;
            }
            visit::visit_path(self, path);
        }
    }
    let mut names = Names {
        found: false,
        self_is_domain: self_type.is_some(),
    };
    names.visit_type(ty);
    names.found
}

fn generics_name_domain_type(
    args: &AngleBracketedGenericArguments,
    self_type: Option<&'static str>,
) -> bool {
    args.args.iter().any(|arg| match arg {
        GenericArgument::Type(ty) => names_domain_type(ty, self_type),
        GenericArgument::AssocType(assoc) => names_domain_type(&assoc.ty, self_type),
        _ => false,
    })
}

/// Whether a `let` initializer is exactly `None`, bare or path-qualified.
fn is_none(init: &LocalInit) -> bool {
    init.diverge.is_none()
        && matches!(&*init.expr, Expr::Path(p) if p.qself.is_none()
            && p.path.segments.last().is_some_and(|s| s.ident == "None"))
}

fn derives_deserialize(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| meta_derives_deserialize(&a.meta))
}

/// A `derive` naming `Deserialize`, written directly or inside a `cfg_attr` — nested to any
/// depth — whose predicate is not definitely test-only.
fn meta_derives_deserialize(meta: &Meta) -> bool {
    let Meta::List(list) = meta else {
        return false;
    };
    if list.path.is_ident("derive") {
        list.parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated)
            .is_ok_and(|paths| {
                paths.iter().any(|path| {
                    path.segments
                        .last()
                        .is_some_and(|s| s.ident == "Deserialize")
                })
            })
    } else if list.path.is_ident("cfg_attr") {
        list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .is_ok_and(|args| {
                let mut args = args.iter();
                args.next()
                    .is_some_and(|predicate| eval_cfg(predicate) != Tri::False)
                    && args.any(meta_derives_deserialize)
            })
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// cfg evaluation
// ---------------------------------------------------------------------------

/// Kleene three-valued truth: `Unknown` is a predicate that holds in some build and not others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    fn negate(self) -> Tri {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }

    fn and(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::False, _) | (_, Tri::False) => Tri::False,
            (Tri::True, Tri::True) => Tri::True,
            _ => Tri::Unknown,
        }
    }

    fn or(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::True, _) | (_, Tri::True) => Tri::True,
            (Tri::False, Tri::False) => Tri::False,
            _ => Tri::Unknown,
        }
    }
}

/// Evaluates a `cfg` predicate for a production build: `test` is false, every other atom —
/// features, targets, anything unrecognized — is unknown and never assumed either way.
fn eval_cfg(meta: &Meta) -> Tri {
    match meta {
        Meta::Path(path) if path.is_ident("test") => Tri::False,
        Meta::List(list) => {
            let Ok(args) = list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                return Tri::Unknown;
            };
            let mut values = args.iter().map(eval_cfg);
            if list.path.is_ident("not") {
                match (values.next(), values.next()) {
                    (Some(value), None) => value.negate(),
                    _ => Tri::Unknown,
                }
            } else if list.path.is_ident("all") {
                values.fold(Tri::True, Tri::and)
            } else if list.path.is_ident("any") {
                values.fold(Tri::False, Tri::or)
            } else {
                Tri::Unknown
            }
        }
        _ => Tri::Unknown,
    }
}

/// Whether a `cfg` on these attributes rules the item out of every production build. Only a
/// definitely-false predicate does; unknown means scanned.
fn is_test_only(attrs: &[Attribute]) -> bool {
    attrs.iter().filter(|a| a.path().is_ident("cfg")).any(|a| {
        a.parse_args::<Meta>()
            .is_ok_and(|m| eval_cfg(&m) == Tri::False)
    })
}

// ---------------------------------------------------------------------------
// Module tree
// ---------------------------------------------------------------------------

/// Files that only a test build compiles, found by following `mod name;` declarations down from
/// `root`. A file declared more than once is excluded only when no declaration reaches it in a
/// production build. A file not reached this way stays scanned, as does one behind `#[path]`,
/// which is not followed — the failure mode is a loud false positive, never a silent exclusion.
fn test_only_modules(root: &Path, read: &dyn Fn(&Path) -> Option<String>) -> BTreeSet<PathBuf> {
    let mut reached = Reached::default();
    walk_module_file(root, false, read, &mut reached);
    reached
        .test
        .difference(&reached.production)
        .cloned()
        .collect()
}

/// Every module file reached, split by whether the route to it compiles in production.
#[derive(Default)]
struct Reached {
    production: BTreeSet<PathBuf>,
    test: BTreeSet<PathBuf>,
}

fn walk_module_file(
    file: &Path,
    test_only: bool,
    read: &dyn Fn(&Path) -> Option<String>,
    reached: &mut Reached,
) {
    let Some(ast) = read(file).and_then(|src| syn::parse_file(&src).ok()) else {
        return;
    };
    let test_only = test_only || is_test_only(&ast.attrs);
    let routes = if test_only {
        &mut reached.test
    } else {
        &mut reached.production
    };
    // A file walked once per kind of route is enough: its children inherit that kind.
    if !routes.insert(file.to_path_buf()) {
        return;
    }
    walk_module_items(&ast.items, &module_dir(file), test_only, read, reached);
}

fn walk_module_items(
    items: &[Item],
    dir: &Path,
    test_only: bool,
    read: &dyn Fn(&Path) -> Option<String>,
    reached: &mut Reached,
) {
    for item in items {
        let Item::Mod(module) = item else { continue };
        if module.attrs.iter().any(|a| a.path().is_ident("path")) {
            continue;
        }
        let test_only = test_only || is_test_only(&module.attrs);
        let name = module.ident.unraw().to_string();
        match &module.content {
            Some((_, inner)) => {
                walk_module_items(inner, &dir.join(&name), test_only, read, reached)
            }
            None => {
                let flat = dir.join(format!("{name}.rs"));
                let file = if read(&flat).is_some() {
                    flat
                } else {
                    dir.join(&name).join("mod.rs")
                };
                walk_module_file(&file, test_only, read, reached);
            }
        }
    }
}

/// The directory a file's out-of-line child modules live in.
fn module_dir(file: &Path) -> PathBuf {
    let parent = file.parent().unwrap_or_else(|| Path::new(""));
    match file.file_stem().and_then(|stem| stem.to_str()) {
        Some("mod" | "lib" | "main") | None => parent.to_path_buf(),
        Some(stem) => parent.join(stem),
    }
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sites(src: &str) -> Vec<Site> {
        scan_source(src).expect("fixture parses")
    }

    fn count(src: &str, rule: Rule) -> usize {
        sites(src).iter().filter(|s| s.rule == rule).count()
    }

    fn site(line: usize, rule: Rule) -> Site {
        Site {
            line,
            rule,
            detail: "fixture",
        }
    }

    // --- construction ---------------------------------------------------------------------

    #[test]
    fn a_usage_literal_is_a_construction() {
        let src = "fn f() -> Usage { Usage { prompt_tokens: 1, ..Default::default() } }";
        assert_eq!(count(src, Rule::Construction), 1);
    }

    #[test]
    fn usage_default_is_a_construction_however_it_is_reached() {
        let src = r#"
            fn a() -> Usage { Usage::default() }
            fn b() -> Usage { crate::domain::chat::Usage::default() }
            fn c(u: Option<Usage>) -> Usage { u.unwrap_or_else(Usage::default) }
            fn d() -> Usage { <Usage as Default>::default() }
        "#;
        assert_eq!(count(src, Rule::Construction), 4);
    }

    #[test]
    fn self_in_an_impl_for_a_domain_type_is_that_type() {
        let src = r#"
            impl From<Wire> for Usage {
                fn from(w: Wire) -> Self { Self { prompt_tokens: w.a, ..Self::default() } }
            }
            impl crate::domain::chat::Usage {
                fn empty() -> Self { <Self as Default>::default() }
                fn parse(s: &str) -> Self { serde_json::from_str::<Self>(s).unwrap() }
            }
            impl ChatResponse {
                fn parse(s: &str) -> Self { let r: Self = serde_json::from_str(s).unwrap(); r }
            }
        "#;
        assert_eq!(count(src, Rule::Construction), 3);
        assert_eq!(count(src, Rule::Deserialization), 2);
    }

    #[test]
    fn self_anywhere_else_is_not_the_domain_usage() {
        // A nested item starts its own scope: the inner `Self` is `Wire`, not the outer `Usage`.
        let src = r#"
            impl ConverseUsage { fn empty() -> Self { Self::default() } }
            impl From<Wire> for Usage {
                fn from(w: Wire) -> Self {
                    impl Default for Wire { fn default() -> Self { Self { a: 0 } } }
                    todo!()
                }
            }
            impl ChatResponse { fn empty() -> Self { Self::default() } }
        "#;
        assert!(sites(src).is_empty());
    }

    #[test]
    fn a_literal_split_across_lines_is_reported_at_its_type_name() {
        let src = "fn f() -> Usage {\n    let u = Usage\n    {\n        ..Default::default()\n    };\n    u\n}\n";
        let found = sites(src);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].rule, found[0].line), (Rule::Construction, 2));
    }

    #[test]
    fn a_construction_inside_a_macro_is_still_a_construction() {
        let src = r#"
            fn f() -> Vec<Usage> {
                vec![Usage { ..Default::default() }, Usage::default()]
            }
        "#;
        assert_eq!(count(src, Rule::Construction), 2);
    }

    #[test]
    fn a_macro_body_that_is_not_rust_is_still_scanned_for_constructions() {
        let src = r#"
            fn f() {
                warn!(provider = %name, usage = ?Usage::default(), "no usage");
                let v = json!({ "usage": Usage { prompt_tokens: 1 } });
            }
        "#;
        assert_eq!(count(src, Rule::Construction), 2);
    }

    #[test]
    fn a_stream_body_is_scanned_like_any_other_code() {
        let src = r#"
            fn f(s: &str, x: Value) -> ChatCompletionStream {
                Box::pin(async_stream::stream! {
                    let mut last_usage: Option<Usage> = None;
                    let u: Usage = serde_json::from_str(s).unwrap();
                    let v = serde_json::from_value::<Usage>(x);
                    yield Usage::default();
                })
            }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 2);
        assert_eq!(count(src, Rule::Construction), 1);
    }

    #[test]
    fn similarly_named_types_are_not_the_domain_usage() {
        let src = r#"
            fn f() {
                let a = ConverseUsage { input_tokens: 1 };
                let b = AnthropicUsage { input_tokens: Some(1) };
                let c = TokenUsage { input_tokens: 1, ..Default::default() };
                let d = EmbeddingUsage { prompt_tokens: 1, total_tokens: 1 };
                let e = UsageMetadata::default();
                let g = vec![ConverseUsage { input_tokens: 1 }, UsageMetadata::default()];
                let h: UsageMetadata = serde_json::from_str(s).unwrap();
            }
        "#;
        assert!(sites(src).is_empty());
    }

    #[test]
    fn signatures_and_patterns_neither_construct_nor_parse() {
        let src = r#"
            fn f(u: &Usage, r: Option<ChatResponse>) -> Result<Usage, ()> {
                let Usage { prompt_tokens, .. } = u;
                let g = |x: Usage| x;
                Err(())
            }
        "#;
        assert!(sites(src).is_empty());
    }

    // --- deserialization ------------------------------------------------------------------

    #[test]
    fn a_type_ascribed_parse_is_a_deserialization() {
        let src = r#"
            async fn f(resp: Response, v: Value, s: &str) -> Result<(), E> {
                let mut chat_resp: ChatResponse = resp.json().await?;
                let mut u: Usage = serde_json::from_value(v).ok()?;
                let o: Option<crate::domain::chat::Usage> = serde_json::from_str(s).ok();
                Ok(())
            }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 3);
    }

    #[test]
    fn an_ascription_initialized_to_none_is_not_a_parse() {
        let src = r#"
            fn f() {
                let mut last_usage: Option<Usage> = None;
                let r: Option<ChatResponse> = Option::None;
            }
        "#;
        assert!(sites(src).is_empty());
    }

    #[test]
    fn an_ascription_that_steers_an_earlier_parse_is_a_deserialization() {
        // Inference runs backwards from the ascription into the call on the line above.
        let src = r#"
            fn f(body: &str) -> Result<(), E> {
                let parsed = serde_json::from_str(body)?;
                let usage: Usage = parsed;
                Ok(())
            }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 1);
    }

    #[test]
    fn an_ascription_assigned_later_is_a_deserialization() {
        let src = "fn f(s: &str) { let u: Usage; u = serde_json::from_str(s).unwrap(); }";
        assert_eq!(count(src, Rule::Deserialization), 1);
    }

    #[test]
    fn a_turbofish_naming_the_domain_type_is_a_deserialization() {
        let src = r#"
            async fn f(resp: Response, s: &str, b: &[u8]) {
                let a = serde_json::from_str::<Usage>(s);
                let c = resp.json::<ChatResponse>().await;
                let d = serde_json::from_slice::<Option<Usage>>(b);
            }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 3);
    }

    #[test]
    fn a_domain_typed_field_of_a_local_deserialize_type_is_a_deserialization() {
        let src = r#"
            fn parse(bytes: &[u8]) {
                #[derive(Deserialize)]
                struct LaneResponse {
                    pub id: Option<String>,
                    #[serde(default)]
                    pub usage: Option<Usage>,
                }
            }
            #[derive(Debug, serde::Deserialize)]
            struct Envelope(ChatResponse);
            #[derive(Deserialize)]
            enum Frame { Done { usage: Usage }, Text(String) }
            #[derive(Serialize)]
            struct Outbound { usage: Usage }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 3);
    }

    #[test]
    fn a_deserialize_derive_behind_cfg_attr_counts_unless_test_only() {
        let src = r#"
            #[cfg_attr(feature = "x", derive(Deserialize))]
            struct A { usage: Option<Usage> }
            #[cfg_attr(not(test), derive(Debug, serde::Deserialize))]
            struct B { usage: Usage }
            #[cfg_attr(feature = "x", cfg_attr(unix, derive(Deserialize)))]
            struct C { usage: Usage }
            #[cfg_attr(test, derive(Deserialize))]
            struct D { usage: Usage }
            #[cfg_attr(feature = "x", derive(Serialize), serde(rename_all = "camelCase"))]
            struct E { usage: Usage }
        "#;
        assert_eq!(count(src, Rule::Deserialization), 3);
    }

    // --- renames --------------------------------------------------------------------------

    #[test]
    fn renaming_a_domain_type_is_refused_in_every_import_form() {
        let src = r#"
            use crate::domain::chat::Usage as U;
            use crate::domain::chat::{Usage as U2, Role};
            pub use crate::domain::chat::Usage as U3;
            use crate::domain::chat::ChatResponse as R;
            use crate::domain::{chat::{Role, Usage as U4}};
            fn f() { use crate::domain::chat::Usage as U5; }
        "#;
        assert_eq!(count(src, Rule::Rename), 6);
    }

    #[test]
    fn a_type_alias_naming_a_domain_type_is_refused() {
        let src = r#"
            type U = Usage;
            type R = ChatResponse;
            type O = Option<crate::domain::chat::Usage>;
        "#;
        assert_eq!(count(src, Rule::Rename), 3);
    }

    #[test]
    fn renaming_the_deserialize_derive_fails_the_scan() {
        // `derives_deserialize` matches the derive by name, so an alias would hide this field.
        let src = "use serde::Deserialize as De;\n\n#[derive(De)]\nstruct WireResponse {\n    usage: Option<Usage>,\n}\n";
        let report = violations("health.rs", &sites(src)).join("\n");
        assert!(
            report.contains("src/providers/health.rs:1: rename"),
            "{report}"
        );

        let more = r#"
            use serde::{Deserialize as De2, Serialize};
            pub use serde::de::Deserialize as De3;
        "#;
        assert_eq!(count(more, Rule::Rename), 2);
    }

    #[test]
    fn an_anonymous_deserialize_import_is_not_a_rename() {
        // `as _` brings the trait's methods into scope and names nothing a derive could use.
        assert!(sites("use serde::Deserialize as _;").is_empty());
    }

    #[test]
    fn glob_plain_and_module_imports_are_not_renames() {
        let src = r#"
            use crate::domain::chat::*;
            use crate::domain::chat::{ChatResponse, Usage};
            use crate::domain::chat as wire;
            type Meta = UsageMetadata;
        "#;
        assert!(sites(src).is_empty());
    }

    // --- cfg ------------------------------------------------------------------------------

    #[test]
    fn cfg_predicates_evaluate_in_three_valued_logic() {
        let table = [
            ("test", Tri::False),
            ("all(test, feature = \"x\")", Tri::False),
            ("any(test, feature = \"x\")", Tri::Unknown),
            ("any(test, not(feature = \"x\"))", Tri::Unknown),
            ("not(test)", Tri::True),
            ("not(feature = \"x\")", Tri::Unknown),
            ("all(not(test), feature = \"x\")", Tri::Unknown),
            ("all(not(test), not(test))", Tri::True),
            ("any()", Tri::False),
        ];
        for (predicate, expected) in table {
            let meta: syn::Meta = syn::parse_str(predicate).expect("predicate parses");
            assert_eq!(eval_cfg(&meta), expected, "cfg({predicate})");
        }
    }

    #[test]
    fn production_reachable_cfg_forms_are_scanned() {
        for predicate in [
            "any(test, feature = \"x\")",
            "any(test, not(feature = \"x\"))",
            "not(test)",
            "not(feature = \"x\")",
        ] {
            let on_fn = format!("#[cfg({predicate})] fn f() -> Usage {{ Usage::default() }}");
            assert_eq!(
                count(&on_fn, Rule::Construction),
                1,
                "fn under cfg({predicate})"
            );
            let on_mod =
                format!("#[cfg({predicate})] mod m {{ fn f() -> Usage {{ Usage::default() }} }}");
            assert_eq!(
                count(&on_mod, Rule::Construction),
                1,
                "mod under cfg({predicate})"
            );
        }
    }

    #[test]
    fn test_only_code_is_not_scanned() {
        let src = r#"
            #[cfg(test)]
            mod tests {
                use crate::domain::chat::Usage as U;
                fn f() -> Usage { Usage::default() }
            }
            #[cfg(all(test, feature = "x"))]
            fn g() -> Usage { Usage { ..Default::default() } }
            impl Lane {
                #[cfg(test)]
                fn h() -> Usage { Usage::default() }
            }
            fn k(s: &str) {
                #[cfg(test)]
                let u: Usage = serde_json::from_str(s).unwrap();
            }
            #[derive(Deserialize)]
            struct Response {
                #[cfg(test)]
                usage: Option<Usage>,
            }
        "#;
        assert!(sites(src).is_empty());
    }

    #[test]
    fn test_only_statements_and_expressions_are_not_scanned() {
        let src = r#"
            fn f(u: &mut Vec<Usage>, k: u8, s: &str) {
                #[cfg(test)]
                assert_eq!(Usage::default(), Usage::default());
                #[cfg(test)]
                u.push(Usage::default());
                #[cfg(test)]
                {
                    let x: Usage = serde_json::from_str(s).unwrap();
                }
                let m = match k {
                    #[cfg(test)]
                    0 => Usage::default(),
                    _ => todo!(),
                };
                let w = Wrapper {
                    #[cfg(test)]
                    usage: Usage::default(),
                    n: 1,
                };
                #[cfg(not(test))]
                u.push(Usage::default());
            }
        "#;
        let found = sites(src);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((found[0].rule, found[0].line), (Rule::Construction, 22));
    }

    #[test]
    fn production_code_after_a_mid_file_test_module_is_still_scanned() {
        let src = "#[cfg(test)]\npub(crate) mod fixture {\n    fn f() -> Usage { Usage::default() }\n}\n\nfn production() -> Usage {\n    Usage::default()\n}\n\n#[cfg(test)]\nmod tests {}\n";
        let found = sites(src);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].rule, found[0].line), (Rule::Construction, 7));
    }

    // --- out-of-line modules --------------------------------------------------------------

    #[test]
    fn files_declared_under_a_test_only_module_are_excluded_with_their_children() {
        let files: HashMap<PathBuf, &str> = [
            ("p/mod.rs", "pub mod router; #[cfg(test)] mod parity;"),
            ("p/parity.rs", ""),
            (
                "p/router/mod.rs",
                "mod streaming; #[cfg(test)] mod tests; #[cfg(any(test, feature = \"x\"))] mod hooks; #[cfg(test)] mod inline { mod deep; }",
            ),
            ("p/router/streaming.rs", ""),
            ("p/router/tests.rs", "mod helpers;"),
            ("p/router/tests/helpers.rs", ""),
            ("p/router/hooks.rs", ""),
            ("p/router/inline/deep.rs", ""),
        ]
        .into_iter()
        .map(|(path, src)| (PathBuf::from(path), src))
        .collect();
        let read = |path: &Path| files.get(path).map(|src| src.to_string());

        let excluded = test_only_modules(Path::new("p/mod.rs"), &read);

        let expected: BTreeSet<PathBuf> = [
            "p/parity.rs",
            "p/router/tests.rs",
            "p/router/tests/helpers.rs",
            "p/router/inline/deep.rs",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(excluded, expected);
    }

    #[test]
    fn a_file_with_any_production_route_is_not_excluded() {
        // Each file is declared twice, once per build; the order of the two must not matter.
        let files: HashMap<PathBuf, &str> = [
            (
                "p/mod.rs",
                "#[cfg(test)] mod shared; #[cfg(not(test))] mod shared; \
                 #[cfg(not(test))] mod other; #[cfg(test)] mod other; \
                 #[cfg(test)] mod only;",
            ),
            ("p/shared.rs", "mod child;"),
            ("p/shared/child.rs", ""),
            ("p/other.rs", ""),
            ("p/only.rs", ""),
        ]
        .into_iter()
        .map(|(path, src)| (PathBuf::from(path), src))
        .collect();
        let read = |path: &Path| files.get(path).map(|src| src.to_string());

        let excluded = test_only_modules(Path::new("p/mod.rs"), &read);

        assert_eq!(excluded, BTreeSet::from([PathBuf::from("p/only.rs")]));
    }

    // --- allowance ------------------------------------------------------------------------

    #[test]
    fn counts_matching_the_allowance_pass() {
        assert!(violations("anthropic/translate.rs", &[site(10, Rule::Construction)]).is_empty());
        assert!(
            violations(
                "azure/mod.rs",
                &[
                    site(10, Rule::Deserialization),
                    site(20, Rule::Construction)
                ]
            )
            .is_empty()
        );
        assert!(violations("health.rs", &[]).is_empty());
    }

    #[test]
    fn a_second_construction_in_an_allowed_file_fails() {
        let report = violations(
            "anthropic/translate.rs",
            &[site(10, Rule::Construction), site(20, Rule::Construction)],
        );
        let text = report.join("\n");
        assert!(
            text.contains("src/providers/anthropic/translate.rs:10"),
            "{text}"
        );
        assert!(
            text.contains("src/providers/anthropic/translate.rs:20"),
            "{text}"
        );
        assert!(text.contains("construction"), "{text}");
    }

    #[test]
    fn any_site_in_a_file_with_no_allowance_fails() {
        let construction = violations("health.rs", &[site(5, Rule::Construction)]).join("\n");
        assert!(
            construction.contains("src/providers/health.rs:5"),
            "{construction}"
        );
        let parse = violations("router/mod.rs", &[site(9, Rule::Deserialization)]).join("\n");
        assert!(parse.contains("src/providers/router/mod.rs:9"), "{parse}");
        assert!(parse.contains("deserialization"), "{parse}");
    }

    #[test]
    fn a_count_below_its_allowance_fails_as_stale() {
        let text = violations("azure/mod.rs", &[site(10, Rule::Deserialization)]).join("\n");
        assert!(text.contains("src/providers/azure/mod.rs"), "{text}");
        assert!(text.contains("allowance 1"), "{text}");
    }

    #[test]
    fn a_rename_fails_even_in_a_file_with_an_allowance() {
        let text = violations(
            "anthropic/translate.rs",
            &[site(10, Rule::Construction), site(3, Rule::Rename)],
        )
        .join("\n");
        assert!(
            text.contains("src/providers/anthropic/translate.rs:3"),
            "{text}"
        );
        assert!(text.contains("rename"), "{text}");
    }
}
