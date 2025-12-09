// NOTE: this representation disallows A[..., d] shapes

use std::{
    collections::HashMap,
    ops::{Add, Mul, Sub},
};

use anyhow::{Result, anyhow};
use std::str::Chars;

pub fn parse_dimvar(s: &str) -> Result<DimVar> {
    let mut parser = Parser::new(s);
    let out = parser.parse_expr()?;
    if parser.peek().is_some() {
        return Err(anyhow!("trailing garbage in dim expression: {}", s));
    }
    Ok(out)
}

struct Parser<'a> {
    it: Chars<'a>,
    look: Option<char>,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        let mut it = s.chars();
        let look = it.next();
        Self { it, look }
    }

    fn bump(&mut self) {
        self.look = self.it.next();
    }

    fn peek(&self) -> Option<char> {
        self.look
    }

    fn eat(&mut self, c: char) -> bool {
        if self.look == Some(c) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<DimVar> {
        let mut node = self.parse_term()?;
        loop {
            if self.eat('+') {
                let rhs = self.parse_term()?;
                node = node + rhs;
            } else if self.eat('-') {
                let rhs = self.parse_term()?;
                node = node - rhs;
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_term(&mut self) -> Result<DimVar> {
        let mut node = self.parse_factor()?;
        loop {
            if self.eat('*') {
                let rhs = self.parse_factor()?;
                node = node * rhs;
            } else {
                break;
            }
        }
        Ok(node)
    }

    fn parse_factor(&mut self) -> Result<DimVar> {
        match self.peek() {
            Some(c) if c.is_ascii_digit() || c == '-' => self.parse_int(),
            Some(c) if c.is_ascii_alphabetic() => self.parse_ident(),
            _ => Err(anyhow!("unexpected char {:?}", self.look)),
        }
    }

    fn parse_int(&mut self) -> Result<DimVar> {
        let mut s = String::new();
        if self.eat('-') {
            s.push('-');
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        let n: i64 = s.parse()?;
        Ok(DimVar::from(n))
    }

    fn parse_ident(&mut self) -> Result<DimVar> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Ok(DimVar::new(DimKind::Named(s)))
    }
}

#[derive(Debug, Clone, Hash)]
pub enum DimKind {
    Named(String),
    Concrete(i64),
    Add {
        left: Box<DimVar>,
        right: Box<DimVar>,
    },
    Mul {
        left: Box<DimVar>,
        right: Box<DimVar>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalDimVar(Vec<Term>);

impl CanonicalDimVar {
    /// Sort terms and merge those with the same variables
    fn normalize(mut terms: Vec<Term>) -> Self {
        terms.sort();

        let mut result: Vec<Term> = Vec::with_capacity(terms.len());
        for term in terms {
            if let Some(last) = result.last_mut() {
                if last.variables == term.variables {
                    last.constant += term.constant;
                    continue;
                }
            }
            result.push(term);
        }

        // Remove zero terms (but keep at least one if all are zero)
        result.retain(|t| t.constant != 0);
        if result.is_empty() {
            result.push(Term::new(0, vec![]));
        }

        CanonicalDimVar(result)
    }

    fn add(&self, other: &CanonicalDimVar) -> CanonicalDimVar {
        let mut terms = self.0.clone();
        terms.extend(other.0.iter().cloned());
        CanonicalDimVar::normalize(terms)
    }

    fn mul(&self, other: &CanonicalDimVar) -> CanonicalDimVar {
        let mut terms = Vec::with_capacity(self.0.len() * other.0.len());
        for a in &self.0 {
            for b in &other.0 {
                terms.push(a.mul(b));
            }
        }
        CanonicalDimVar::normalize(terms)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Term {
    constant: i64,
    variables: Vec<NamedPow>,
}

impl Term {
    pub fn new(constant: i64, variables: Vec<NamedPow>) -> Self {
        Self {
            constant,
            variables,
        }
    }

    fn mul(&self, other: &Term) -> Term {
        Term {
            constant: self.constant * other.constant,
            variables: mul_named_pows(&self.variables, &other.variables),
        }
    }
}

impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sort by variables only (not constant), so terms with same vars are adjacent
        self.variables.cmp(&other.variables)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NamedPow {
    variable: String,
    pow: u32,
}

impl NamedPow {
    pub fn new(variable: String, pow: u32) -> Self {
        Self { variable, pow }
    }
}

/// Multiply two sorted NamedPow vectors, combining powers for same variable
fn mul_named_pows(a: &[NamedPow], b: &[NamedPow]) -> Vec<NamedPow> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].variable.cmp(&b[j].variable) {
            std::cmp::Ordering::Less => {
                result.push(a[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j].clone());
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(NamedPow::new(a[i].variable.clone(), a[i].pow + b[j].pow));
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

impl DimVar {
    pub fn is_one(&self) -> bool {
        if let DimKind::Concrete(c1) = self.kind() {
            c1 == 1
        } else {
            false
        }
    }

    /// transforms a DimVar into its canonical polynomial form
    pub fn canonical(&self) -> CanonicalDimVar {
        match self.kind() {
            DimKind::Named(n) => CanonicalDimVar(vec![Term::new(1, vec![NamedPow::new(n, 1)])]),
            DimKind::Concrete(c) => CanonicalDimVar(vec![Term::new(c, vec![])]),
            DimKind::Add { left, right } => {
                let left = left.canonical();
                let right = right.canonical();
                return left.add(&right);
            }
            DimKind::Mul { left, right } => {
                let left = left.canonical();
                let right = right.canonical();
                return left.mul(&right);
            }
        }
    }

    pub fn substitute(&self, map: &HashMap<String, DimVar>) -> Result<DimVar> {
        Ok(match self.kind() {
            DimKind::Named(name) => map
                .get(&name)
                .cloned()
                .ok_or_else(|| anyhow!("can't resolve callee dim var {}", name))?,
            DimKind::Mul { left, right } => left.substitute(map)? * right.substitute(map)?,
            DimKind::Add { left, right } => left.substitute(map)? + right.substitute(map)?,
            DimKind::Concrete(_) => self.clone(),
        })
    }
}

impl Add for DimVar {
    type Output = DimVar;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.kind(), rhs.kind()) {
            (DimKind::Concrete(c1), DimKind::Concrete(c2)) => {
                let kind = DimKind::Concrete(c1 + c2);
                DimVar::new(kind)
            }
            _ => {
                let kind = DimKind::Add {
                    left: Box::new(self),
                    right: Box::new(rhs),
                };
                DimVar::new(kind)
            }
        }
    }
}

impl Sub for DimVar {
    type Output = DimVar;

    fn sub(self, rhs: Self) -> Self::Output {
        let negative_rhs = DimVar::new(DimKind::Concrete(-1)) * rhs;
        self + negative_rhs
    }
}

impl Mul for DimVar {
    type Output = DimVar;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.kind(), rhs.kind()) {
            (DimKind::Concrete(c1), DimKind::Concrete(c2)) => {
                let kind = DimKind::Concrete(c1 * c2);
                DimVar::new(kind)
            }
            _ => {
                let kind = DimKind::Mul {
                    left: Box::new(self),
                    right: Box::new(rhs),
                };
                DimVar::new(kind)
            }
        }
    }
}

impl PartialEq for DimVar {
    fn eq(&self, other: &Self) -> bool {
        self.canonical() == other.canonical()
    }
}

impl Eq for DimVar {}

#[derive(Debug, Clone, Hash)]
pub struct DimVar {
    pub kind: DimKind,
    // TODO: more info potent around this, related to origin of this dimvar which would be
    // useful for debugging
}

impl DimVar {
    pub fn new(kind: DimKind) -> Self {
        Self { kind }
    }

    pub fn kind(&self) -> DimKind {
        self.kind.clone()
    }
}

impl From<i64> for DimVar {
    fn from(value: i64) -> Self {
        Self {
            kind: DimKind::Concrete(value),
        }
    }
}
