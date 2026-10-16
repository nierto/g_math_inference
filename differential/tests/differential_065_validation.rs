#![allow(dead_code, unused_macros, unused_imports)]
//! Differential gate: bit-identity with the published 0.6.7 (the scalar surface is unchanged since 0.6.5).
//!
//! `g_math` 0.6.7 from crates.io is a dependency of this crate (`g_math_065`).
//! It reads `GMATH_PROFILE` / `GMATH_FRAC_BITS`; the new type is derived from
//! its storage word and split, so in each build the old `FixedPoint` and the
//! new one are the same format, and every operation is run on both with the
//! same raw inputs. `build.rs` mirrors the profile as `published_profile`
//! for the paths that exist on some formats only. Required: the same
//! integers, the same `Err` variants, and a panic exactly where the old
//! crate panicked. Run once per profile and split.
//!
//! Inputs are raw integers of every magnitude: a random bit length, random
//! bits below it, a random sign, plus the extremes and the values around
//! zero and one.

use g_math::fixed_point::{Fixed, FixedNum, Storage, WideInt};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---- the format of the published crate's build, mirrored in this crate -----
//
// The published crate still reads `GMATH_PROFILE` / `GMATH_FRAC_BITS`; this
// crate has no profile. The new type is chosen from the old crate's own
// storage word and split, so one build of this test compares the two crates
// at whatever format the old one was built for.

type OldWord = g_math_065::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
const OLD_FRAC: u32 = g_math_065::fixed_point::frac_config::FRAC_BITS;

/// The new crate's word for an old crate's word, and the raw in both as bytes.
trait Mirror: Copy {
    type New: Storage;
    fn mirror_to_be(self) -> Vec<u8>;
    fn mirror_from_be(bytes: &[u8]) -> Self;
}
macro_rules! mirror_native {
    ($t:ty) => {
        impl Mirror for $t {
            type New = $t;
            fn mirror_to_be(self) -> Vec<u8> { self.to_be_bytes().to_vec() }
            fn mirror_from_be(bytes: &[u8]) -> Self { <$t>::from_be_bytes(bytes.try_into().expect("word bytes")) }
        }
    };
}
mirror_native!(i32);
mirror_native!(i64);
mirror_native!(i128);
macro_rules! mirror_wide {
    ($old:ty, $new:ty, $words:expr) => {
        impl Mirror for $old {
            type New = $new;
            fn mirror_to_be(self) -> Vec<u8> {
                let mut be = Vec::with_capacity($words * 8);
                for w in self.words.iter().rev() { be.extend_from_slice(&w.to_be_bytes()); }
                be
            }
            fn mirror_from_be(bytes: &[u8]) -> Self {
                let mut words = [0u64; $words];
                for (i, chunk) in bytes.rchunks(8).enumerate() { words[i] = u64::from_be_bytes(chunk.try_into().expect("word bytes")); }
                Self { words }
            }
        }
    };
}
mirror_wide!(g_math_065::fixed_point::I256, g_math::fixed_point::I256, 4);
mirror_wide!(g_math_065::fixed_point::I512, g_math::fixed_point::I512, 8);

type New = Fixed<<OldWord as Mirror>::New, OLD_FRAC>;
use g_math_065::fixed_point::FixedPoint as Old;

// ---- the same raw integer in both crates --------------------------------

mod raw {
    use super::{Mirror, New, Old, OldWord, Storage};
    /// A raw of the new crate's word.
    pub type Raw = <OldWord as Mirror>::New;
    pub const WORDS: usize = (<Raw as super::WideInt>::BITS / 64) as usize + (<Raw as super::WideInt>::BITS % 64 != 0) as usize;
    /// Little-endian 64-bit words into a raw (the words beyond the width are ignored).
    pub fn from_words(w: &[u64]) -> Raw {
        let bytes = (<Raw as super::WideInt>::BITS / 8) as usize;
        let mut be = vec![0u8; bytes];
        for (i, x) in w.iter().enumerate() {
            for k in 0..8 {
                let pos = i * 8 + k;
                if pos < bytes { be[bytes - 1 - pos] = (x >> (8 * k)) as u8; }
            }
        }
        <Raw as Storage>::from_be_slice(&be)
    }
    pub fn old(r: Raw) -> Old { Old::from_raw(OldWord::mirror_from_be(&<Raw as Storage>::to_be_vec(r))) }
    pub fn new(r: Raw) -> New { New::from_raw(r) }
    pub fn old_raw(v: Old) -> Raw { <Raw as Storage>::from_be_slice(&v.raw().mirror_to_be()) }
    pub fn new_raw(v: New) -> Raw { v.raw() }
}

use raw::{Raw, WORDS};

#[allow(dead_code)]
const BITS: u32 = <Raw as WideInt>::BITS;
const WIDTH: u32 = BITS;
const FRAC: u32 = <New as FixedNum>::FRAC_BITS;

// ---- inputs ---------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd) ^ (x >> 29)
    }
    /// A raw of a random magnitude: `bits` significant bits, random sign.
    fn raw(&mut self, max_bits: u32) -> Raw {
        let bits = (self.next() % (max_bits as u64 + 1)) as u32;
        let mut w = [0u64; 8];
        for i in 0..WORDS {
            w[i] = self.next();
        }
        // keep `bits` low bits, force the top one
        for i in 0..WORDS as u32 {
            let lo = i * 64;
            if bits <= lo {
                w[i as usize] = 0;
            } else if bits < lo + 64 {
                w[i as usize] &= (1u64 << (bits - lo)) - 1;
                w[i as usize] |= 1u64 << (bits - lo - 1);
            }
        }
        if self.next() & 1 == 1 {
            // two's complement negation over the storage width
            let mut carry = 1u64;
            for x in w.iter_mut().take(WORDS) {
                let (v, c) = (!*x).overflowing_add(carry);
                *x = v;
                carry = c as u64;
            }
        }
        if WIDTH < 64 {
            // sign-extend the low word's WIDTH bits
            w[0] = ((w[0] as i64) << (64 - WIDTH) >> (64 - WIDTH)) as u64;
        }
        raw::from_words(&w)
    }
}

/// Cases per function, scaled down where one evaluation is slow.
fn count(narrow: usize) -> usize {
    let scale = std::env::var("GMATH_DIFF_SCALE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let base = match WIDTH {
        32 | 64 => narrow,
        128 => narrow / 4,
        256 => narrow / 20,
        _ => narrow / 60,
    };
    (base * scale).max(24)
}

/// Raws to try: every magnitude up to `max_bits`, then the special values.
fn inputs(seed: u64, n: usize, max_bits: u32) -> Vec<Raw> {
    let mut rng = Rng(seed);
    let mut v: Vec<Raw> = (0..n).map(|_| rng.raw(max_bits)).collect();
    v.extend(specials());
    v
}

fn specials() -> Vec<Raw> {
    let mut out = Vec::new();
    let mut push = |f: &dyn Fn(&mut [u64; 8])| {
        let mut w = [0u64; 8];
        f(&mut w);
        if WIDTH < 64 {
            w[0] = ((w[0] as i64) << (64 - WIDTH) >> (64 - WIDTH)) as u64;
        }
        out.push(raw::from_words(&w));
    };
    let set = |w: &mut [u64; 8], bit: u32| w[(bit / 64) as usize] |= 1u64 << (bit % 64);
    push(&|_| {}); // 0
    push(&|w| w[0] = 1); // one unit
    push(&|w| w.iter_mut().take(WORDS).for_each(|x| *x = u64::MAX)); // minus one unit
    push(&|w| set(w, FRAC)); // 1.0
    push(&|w| { set(w, FRAC); w[0] |= 1 }); // 1.0 + one unit
    push(&|w| set(w, FRAC - 1)); // 0.5
    push(&|w| set(w, FRAC + 1)); // 2.0
    push(&|w| { // -1.0
        for b in FRAC..WIDTH { set(w, b) }
    });
    push(&|w| { // -0.5
        for b in FRAC - 1..WIDTH { set(w, b) }
    });
    push(&|w| set(w, WIDTH - 1)); // minimum
    push(&|w| { // maximum
        for b in 0..WIDTH - 1 { set(w, b) }
    });
    push(&|w| { // minimum + one unit
        set(w, WIDTH - 1);
        w[0] |= 1
    });
    out
}

// ---- comparing outcomes -----------------------------------------------------

/// What a call did: a value, an error (by its Debug text, the variants are
/// distinct types in the two crates), or a panic.
#[derive(PartialEq, Debug)]
enum Outcome<T> {
    Value(T),
    Error(String),
    Panic,
}

thread_local! {
    /// Set while a call that may legitimately panic is being compared.
    static EXPECTING_PANICS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn quiet<T>(f: impl FnOnce() -> T) -> Option<T> {
    EXPECTING_PANICS.with(|e| e.set(true));
    let result = catch_unwind(AssertUnwindSafe(f)).ok();
    EXPECTING_PANICS.with(|e| e.set(false));
    result
}

fn value<T>(f: impl FnOnce() -> T) -> Outcome<T> {
    quiet(f).map_or(Outcome::Panic, Outcome::Value)
}

fn fallible<T, E: std::fmt::Debug>(f: impl FnOnce() -> Result<T, E>) -> Outcome<T> {
    match quiet(f) {
        None => Outcome::Panic,
        Some(Ok(v)) => Outcome::Value(v),
        Some(Err(e)) => Outcome::Error(format!("{e:?}")),
    }
}

struct Tally {
    name: &'static str,
    cases: usize,
    values: usize,
    errors: usize,
    panics: usize,
    /// Differences where 0.6.7 was not correctly rounded and the new crate
    /// equals the sure path (transcendentals), or where a composition moved
    /// by at most `NEAR_UNITS` because a transcendental inside it now rounds
    /// correctly (geometry). Every other difference fails the gate.
    corrected: usize,
    worst: u128,
}

/// The largest move of a geometric composition allowed when the
/// transcendentals inside it became correctly rounded.
const NEAR_UNITS: u128 = 4;

/// Distance in storage units between two results.
trait Units {
    fn units(&self, other: &Self) -> u128;
}
impl Units for Raw {
    fn units(&self, other: &Self) -> u128 {
        let d = match self.checked_sub_wide(*other) { Some(d) => d, None => return u128::MAX };
        let m = if d.is_negative() { -d } else { d };
        let n = m.bit_length();
        if n > 127 { return u128::MAX; }
        (0..n).fold(0u128, |acc, k| if m.bit(k) { acc | (1u128 << k) } else { acc })
    }
}
impl Units for usize {
    fn units(&self, other: &Self) -> u128 { if self == other { 0 } else { u128::MAX } }
}
impl<A: Units, B: Units> Units for (A, B) {
    fn units(&self, other: &Self) -> u128 { self.0.units(&other.0).max(self.1.units(&other.1)) }
}
impl<A: Units, B: Units, C: Units> Units for (A, B, C) {
    fn units(&self, other: &Self) -> u128 { self.0.units(&other.0).max(self.1.units(&other.1)).max(self.2.units(&other.2)) }
}
impl<T: Units> Units for Vec<T> {
    fn units(&self, other: &Self) -> u128 {
        if self.len() != other.len() { return u128::MAX; }
        self.iter().zip(other).map(|(a, b)| a.units(b)).max().unwrap_or(0)
    }
}

impl Tally {
    fn new(name: &'static str) -> Self {
        Tally { name, cases: 0, values: 0, errors: 0, panics: 0, corrected: 0, worst: 0 }
    }
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, old: Outcome<T>, new: Outcome<T>, input: &dyn std::fmt::Debug) {
        assert_eq!(old, new, "{} differs from 0.6.5 on input {:?}", self.name, input);
        self.cases += 1;
        match old {
            Outcome::Value(_) => self.values += 1,
            Outcome::Error(_) => self.errors += 1,
            Outcome::Panic => self.panics += 1,
        }
    }
    fn report(&self) {
        println!(
            "{:<22} {:>6} cases: {:>6} values, {:>5} errors, {:>5} panics, {} corrected (worst {} units), 0 other differences",
            self.name, self.cases, self.values, self.errors, self.panics, self.corrected, self.worst
        );
    }
    fn check_oracle<T: PartialEq + std::fmt::Debug + Units>(&mut self, old: Outcome<T>, new: Outcome<T>, oracle: impl FnOnce() -> Option<T>, input: &dyn std::fmt::Debug) {
        if old == new { return self.check(old, new, input); }
        let want = oracle();
        let ok = match (&new, &want) {
            (Outcome::Value(v), Some(w)) => *v == *w,
            (Outcome::Value(_), None) => false,
            (_, None) => true,
            (_, Some(_)) => false,
        };
        assert!(ok, "{} differs from 0.6.7 on input {:?} and is not the correctly rounded value: old {:?}, new {:?}, sure {:?}", self.name, input, old, new, want);
        if let (Outcome::Value(o), Outcome::Value(n)) = (&old, &new) { self.worst = self.worst.max(o.units(n)); }
        self.cases += 1;
        self.corrected += 1;
        if let Outcome::Value(_) = new { self.values += 1 } else { self.errors += 1 }
    }
    /// As `check`, where a geometric composition may move by at most
    /// `NEAR_UNITS` (a transcendental inside it now rounds correctly).
    fn check_near<T: PartialEq + std::fmt::Debug + Units>(&mut self, old: Outcome<T>, new: Outcome<T>, input: &dyn std::fmt::Debug) {
        if old == new { return self.check(old, new, input); }
        let u = match (&old, &new) { (Outcome::Value(o), Outcome::Value(n)) => o.units(n), _ => u128::MAX };
        assert!(u <= NEAR_UNITS, "{} differs from 0.6.7 by {} units on input {:?}: old {:?}, new {:?}", self.name, u, input, old, new);
        self.worst = self.worst.max(u);
        self.cases += 1;
        self.corrected += 1;
        self.values += 1;
    }
    /// As `report`, for inputs chosen inside the function's domain: a gate
    /// that only ever compared errors or panics would prove nothing.
    fn report_in_domain(&self) {
        self.report();
        assert!(self.values * 2 >= self.cases, "{}: too few in-range cases", self.name);
    }
}

/// Keep the expected panics out of the test output; report every other one.
fn silence_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING_PANICS.with(|e| e.get()) {
                default(info);
            }
        }));
    });
}

// ---- the inference module (width-types phase 4d) ----------------------------

#[test]
fn inference_kernels_are_bit_identical_to_0_6_5() {
    use g_math_inference as nt;
    use g_math_065::tq19 as ot;
    silence_panics();
    let n = count(40);
    let mut rng = Rng(0x7019_0001);
    let t = |name: &'static str| Tally::new(name);
    #[allow(unused_mut)] // the row-scaled forms exist on the 32- and 64-bit words only
    let (mut dots, mut dense, mut planar, mut hybrid, mut rowscaled, mut bits) = (t("dots"), t("TQ19Matrix"), t("PlanarTQ19"), t("HybridTQ19"), t("RowScaledTQ19"), t("bits/quantize"));
    let weights = |rng: &mut Rng, k: usize| (0..k).map(|_| ((rng.next() % 59_049) as i32 - 29_524) as i16).collect::<Vec<i16>>();
    let trits = |rng: &mut Rng, k: usize| (0..k).map(|_| (rng.next() % 3) as i8 - 1).collect::<Vec<i8>>();
    // the published crate takes raw words, the tree takes Fixed slices over the same words
    let old_acts = |r: &[Raw]| r.iter().map(|&x| raw::old(x).raw()).collect::<Vec<_>>();
    let old_res = |v| raw::old_raw(Old::from_raw(v));
    let old_vec_res = |v: Vec<_>| v.into_iter().map(|x| raw::old_raw(Old::from_raw(x))).collect::<Vec<Raw>>();
    let new_acts = |r: &[Raw]| r.iter().map(|&x| raw::new(x)).collect::<Vec<_>>();
    let new_raws_of = |v: Vec<New>| v.iter().map(|&x| raw::new_raw(x)).collect::<Vec<Raw>>();
    for _ in 0..n {
        let cols = 1 + (rng.next() % 40) as usize;
        let rows = 1 + (rng.next() % 6) as usize;
        let a = (0..cols).map(|_| rng.raw(FRAC + 4)).collect::<Vec<Raw>>();
        let (oa, na) = (old_acts(&a), new_acts(&a));
        let w = weights(&mut rng, cols);
        let tr = trits(&mut rng, cols);
        dots.check(value(|| old_res(ot::tq19_dot(&w, &oa))), value(|| raw::new_raw(nt::tq19_dot(&w, &na))), &(&w, &a));
        dots.check(value(|| old_res(ot::trit_dot(&tr, &oa))), value(|| raw::new_raw(nt::trit_dot(&tr, &na))), &(&tr, &a));
        let packed: Vec<u8> = tr.chunks(5).map(|c| c.iter().fold(0u8, |acc, &t| acc * 3 + (t + 1) as u8)).collect();
        let s = rng.raw(FRAC + 1);
        dots.check(value(|| old_res(ot::packed_trit_dot(&packed, cols, &oa, raw::old(s).raw()))), value(|| raw::new_raw(nt::packed_trit_dot(&packed, cols, &na, raw::new(s)))), &(&tr, &a, s));
        #[cfg(any(published_profile = "q16_16", published_profile = "q32_32"))]
        dots.check(value(|| format!("{:?}", ot::tq19_dot_q2f(&w, &oa))), value(|| format!("{:?}", nt::tq19_dot_q2f(&w, &na))), &(&w, &a));
        // the dense matrix and its repacked forms
        let data = weights(&mut rng, rows * cols);
        let (om, nm) = (ot::TQ19Matrix::new(rows, cols, data.clone()), nt::TQ19Matrix::new(rows, cols, data.clone()));
        let b = (0..3).map(|_| (0..cols).map(|_| rng.raw(FRAC + 4)).collect::<Vec<Raw>>()).collect::<Vec<_>>();
        let ob: Vec<Vec<_>> = b.iter().map(|r| old_acts(r)).collect();
        let nb: Vec<Vec<New>> = b.iter().map(|r| new_acts(r)).collect();
        let obr: Vec<&[_]> = ob.iter().map(|v| v.as_slice()).collect();
        let nbr: Vec<&[New]> = nb.iter().map(|v| v.as_slice()).collect();
        dense.check(value(|| old_vec_res(om.matvec(&oa))), value(|| new_raws_of(nm.matvec(&na))), &(&data, &a));
        dense.check(value(|| old_vec_res(om.matvec_par(&oa))), value(|| new_raws_of(nm.matvec_par(&na))), &(&data, &a));
        dense.check(value(|| om.matvec_batch(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nm.matvec_batch(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        dense.check(value(|| om.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nm.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        #[cfg(any(published_profile = "q16_16", published_profile = "q32_32"))]
        {
            dense.check(value(|| format!("{:?}", om.matvec_q2f(&oa))), value(|| format!("{:?}", nm.matvec_q2f(&na))), &(&data, &a));
            dense.check(value(|| format!("{:?}", om.matvec_q2f_batch_par(&obr))), value(|| format!("{:?}", nm.matvec_q2f_batch_par(&nbr))), &(&data, &b));
        }
        let (op, np_) = (ot::PlanarTQ19::from_tq19(&om), nt::PlanarTQ19::from_tq19(&nm));
        planar.check(value(|| old_vec_res(op.matvec(&oa))), value(|| new_raws_of(np_.matvec(&na))), &(&data, &a));
        planar.check(value(|| op.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| np_.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        let (oh, nh) = (ot::HybridTQ19::from_tq19(&om), nt::HybridTQ19::from_tq19(&nm));
        hybrid.check(value(|| old_vec_res(oh.matvec(&oa))), value(|| new_raws_of(nh.matvec(&na))), &(&data, &a));
        hybrid.check(value(|| oh.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nh.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        #[cfg(any(published_profile = "q16_16", published_profile = "q32_32"))]
        {
            planar.check(value(|| format!("{:?}", op.matvec_q2f(&oa))), value(|| format!("{:?}", np_.matvec_q2f(&na))), &(&data, &a));
            hybrid.check(value(|| format!("{:?}", oh.matvec_q2f(&oa))), value(|| format!("{:?}", nh.matvec_q2f(&na))), &(&data, &a));
            let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 30)).collect();
            let (or, nr) = (ot::RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone()), nt::RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone()));
            rowscaled.check(value(|| old_vec_res(or.matvec(&oa))), value(|| new_raws_of(nr.matvec(&na))), &(&data, &a, &scales));
            rowscaled.check(value(|| or.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nr.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b, &scales));
            rowscaled.check(value(|| format!("{:?}", or.matvec_q2f(&oa))), value(|| format!("{:?}", nr.matvec_q2f(&na))), &(&data, &a, &scales));
            rowscaled.check(value(|| format!("{:?}", or.matvec_q2f_batch_par(&obr))), value(|| format!("{:?}", nr.matvec_q2f_batch_par(&nbr))), &(&data, &b, &scales));
        }
        #[cfg(published_profile = "q16_16")]
        {
            let codes: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
            let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 28)).collect();
            let (o5, n5) = (ot::RowScaledTQ5::from_parts(rows, cols, codes.clone(), scales.clone()), nt::RowScaledTQ5::from_parts(rows, cols, codes.clone(), scales.clone()));
            let x: Vec<i32> = a.iter().map(|&r| (r as i64 >> 2) as i32).collect();
            let nx: Vec<New> = x.iter().map(|&v| raw::new(v as Raw)).collect();
            rowscaled.check(value(|| old_vec_res(o5.matvec(&x))), value(|| new_raws_of(n5.matvec(&nx))), &(&codes, &x));
            rowscaled.check(value(|| o5.matvec_q2f(&x)), value(|| n5.matvec_q2f(&nx)), &(&codes, &x));
            let xb: Vec<&[i32]> = vec![&x, &x];
            let nxb: Vec<&[New]> = vec![&nx, &nx];
            rowscaled.check(value(|| o5.matvec_batch_par(&xb).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| n5.matvec_batch_par(&nxb).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&codes, &x));
        }
        // the half-precision decoders and the quantisers
        let hb = (rng.next() & 0xFFFF) as u16;
        for kind in [ot::bits::HalfKind::BFloat16, ot::bits::HalfKind::Binary16] {
            let nkind = match kind { ot::bits::HalfKind::BFloat16 => nt::bits::HalfKind::BFloat16, _ => nt::bits::HalfKind::Binary16 };
            bits.check(value(|| ot::bits::to_tq19_raw(hb, kind).map_err(|e| format!("{e:?}"))), value(|| nt::bits::to_tq19_raw(hb, nkind).map_err(|e| format!("{e:?}"))), &hb);
            #[cfg(any(published_profile = "q16_16", published_profile = "q32_32"))]
            bits.check(value(|| ot::bits::to_fixed(hb, kind).map(raw::old_raw).map_err(|e| format!("{e:?}"))), value(|| nt::bits::to_fixed::<_, FRAC>(hb, nkind).map(raw::new_raw).map_err(|e| format!("{e:?}"))), &hb);
        }
        let wb: Vec<u8> = (0..rows * cols * 2).map(|_| (rng.next() & 0xFF) as u8).collect();
        let (ow, nw) = (ot::bits::WeightBits::from_le_bytes(rows, cols, ot::bits::HalfKind::BFloat16, &wb).unwrap(), nt::bits::WeightBits::from_le_bytes(rows, cols, nt::bits::HalfKind::BFloat16, &wb).unwrap());
        bits.check(value(|| ot::quantize::quantize_tq19(&ow).map(|m| m.data().to_vec())), value(|| nt::quantize::quantize_tq19(&nw).map(|m| m.data().to_vec())), &wb);
        #[cfg(any(published_profile = "q16_16", published_profile = "q32_32"))]
        bits.check(value(|| ot::quantize::quantize_tq19_rowscaled(&ow).map(|m| (m.data().to_vec(), m.scales_q32().to_vec()))), value(|| nt::quantize::quantize_tq19_rowscaled(&nw).map(|m| (m.data().to_vec(), m.scales_q32().to_vec()))), &wb);
    }
    for t in [&dots, &dense, &planar, &hybrid, &rowscaled, &bits] {
        t.report();
    }
}
