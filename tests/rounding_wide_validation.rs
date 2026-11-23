//! Rounding modes and wide activations against exact references.
//!
//! - `Rounding::Nearest` on RowScaledTQ5, RowScaledTQ19, TQ19Matrix,
//!   PlanarTQ19, HybridTQ19: every form (matvec, par, batch, batch_into,
//!   q2f) equals the exact value rounded once to nearest (ties toward
//!   +infinity), computed here from the stored weights at 512 bits.
//!   (NvFp4Matrix / MxFp8Matrix nearest: against Fraction references in
//!   blockscaled_validation.rs.)
//! - `Rounding::Default` unchanged (the existing suites) and, here, the
//!   wide-output nearest consistent with the narrow nearest.
//! - `matvec_wide_act` (NvFp4Matrix, MxFp8Matrix, RowScaledTQ5) at Q32.32 and
//!   Q64.64: equal to the exact value rounded once by the matrix's mode
//!   (512-bit references from the stored weights), and equal to `matvec`
//!   when the activations are 32-bit values widened to the larger word.
//! - RowScaledTQ19 and the TQ1.9 forms at Q64.64 through their generic
//!   `matvec` (the i128 word now has the wide epilogues).
//!
//! Integer arithmetic only.

use g_math::fixed_point::{Fixed, I512};
use g_math_inference::{HybridTQ19, MxFp8Matrix, NvFp4Matrix, PlanarTQ19, Rounding, RowScaledTQ19, RowScaledTQ5, TQ19Matrix};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn signed(&mut self, bits: u32) -> i128 {
        let r = self.next();
        let m = ((r >> 1) as i128) & ((1i128 << bits.min(62)) - 1);
        let m = if bits > 62 { (m << (bits - 62)) | (self.next() as i128 & ((1i128 << (bits - 62)) - 1)) } else { m };
        if r & 1 == 1 { -m } else { m }
    }
}

const SCALE: i128 = 19683;

fn i(v: i128) -> I512 {
    I512::from(v)
}

/// floor(n / d), d > 0
fn floor_div(n: I512, d: I512) -> I512 {
    let (q, r) = (n / d, n % d);
    if !r.is_zero() && r.is_negative() { q - i(1) } else { q }
}

/// n / d rounded by mode: `floor_or_trunc` gives the Default rule of the form.
fn round(n: I512, d: I512, mode: Rounding, trunc_default: bool) -> I512 {
    match mode {
        Rounding::Nearest => floor_div((n << 1usize) + d, d << 1usize),
        Rounding::Default if trunc_default => n / d,
        Rounding::Default => floor_div(n, d),
    }
}

fn val(v: I512) -> i128 {
    assert!(v.fits_in_i128());
    v.as_i128()
}

fn opt(v: I512) -> Option<i128> {
    v.fits_in_i128().then(|| v.as_i128())
}

// ------------------------------------------------------------- nearest

#[test]
fn tq19_forms_round_to_nearest_once() {
    let mut rng = Rng(0x7E19);
    for &(rows, cols) in &[(5usize, 37usize), (16, 256), (3, 1030)] {
        let data: Vec<i16> = (0..rows * cols).map(|_| ((rng.next() % 59049) as i32 - 29524) as i16).collect();
        let dense = TQ19Matrix::new(rows, cols, data.clone()).with_rounding(Rounding::Nearest);
        let planar = PlanarTQ19::from_tq19(&dense);
        let hybrid = HybridTQ19::from_tq19(&dense);
        assert_eq!(planar.rounding(), Rounding::Nearest);
        type X = Fixed<i32, 10>;
        let xs: Vec<Vec<X>> = (0..3).map(|_| (0..cols).map(|_| X::from_raw(rng.signed(20) as i32)).collect()).collect();
        for x in &xs {
            let acc = |r: usize| (0..cols).map(|c| data[r * cols + c] as i128 * x[c].raw() as i128).sum::<i128>();
            let want: Vec<i32> = (0..rows).map(|r| val(round(i(acc(r)), i(SCALE), Rounding::Nearest, true)) as i32).collect();
            let want_q2f: Vec<i64> = (0..rows).map(|r| val(round(i(acc(r)) << 10usize, i(SCALE), Rounding::Nearest, true)) as i64).collect();
            let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw()).collect::<Vec<i32>>();
            for (name, got) in [
                ("dense", raw(dense.matvec(x))),
                ("dense_par", raw(dense.matvec_par(x))),
                ("planar", raw(planar.matvec(x))),
                ("planar_par", raw(planar.matvec_par(x))),
                ("hybrid", raw(hybrid.matvec(x))),
                ("hybrid_par", raw(hybrid.matvec_par(x))),
            ] {
                assert_eq!(got, want, "{name} {rows}x{cols}");
            }
            assert_eq!(dense.matvec_q2f(x), want_q2f);
            assert_eq!(planar.matvec_q2f(x), want_q2f);
            assert_eq!(hybrid.matvec_q2f_par(x), want_q2f);
        }
        let refs: Vec<&[X]> = xs.iter().map(|v| &v[..]).collect();
        let single: Vec<Vec<X>> = xs.iter().map(|x| dense.matvec(x)).collect();
        assert_eq!(dense.matvec_batch_par(&refs), single);
        assert_eq!(dense.matvec_batch(&refs), single);
        assert_eq!(planar.matvec_batch_par(&refs), single);
        assert_eq!(hybrid.matvec_batch_par(&refs), single);
        let mut out = vec![X::from_raw(0); refs.len() * rows];
        hybrid.matvec_batch_par_into(&refs, &mut out);
        assert_eq!(out, single.concat());
        let q2f_single: Vec<Vec<i64>> = xs.iter().map(|x| dense.matvec_q2f(x)).collect();
        assert_eq!(dense.matvec_q2f_batch_par(&refs), q2f_single);
    }
}

#[test]
fn row_scaled_forms_round_to_nearest_once() {
    let mut rng = Rng(0x5CA1);
    for &(rows, cols) in &[(4usize, 33usize), (9, 512), (2, 2304)] {
        type X = Fixed<i32, 10>;
        let xs: Vec<Vec<X>> = (0..3).map(|_| (0..cols).map(|_| X::from_raw(rng.signed(17) as i32)).collect()).collect();
        let refs: Vec<&[X]> = xs.iter().map(|v| &v[..]).collect();
        // RowScaledTQ19: exact acc * s / (SCALE 2^32), once
        let d19: Vec<i16> = (0..rows * cols).map(|_| ((rng.next() % 59049) as i32 - 29524) as i16).collect();
        let s19: Vec<u64> = (0..rows).map(|_| (rng.next() >> 34) + 1).collect();
        let m19 = RowScaledTQ19::from_parts(rows, cols, d19.clone(), s19.clone()).with_rounding(Rounding::Nearest);
        // RowScaledTQ5: exact acc * s / 2^32, once
        let d5: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
        let s5: Vec<u64> = (0..rows).map(|_| (rng.next() >> 36) + 1).collect();
        let m5 = RowScaledTQ5::from_parts(rows, cols, d5.clone(), s5.clone()).with_rounding(Rounding::Nearest);
        for x in &xs {
            let a19 = |r: usize| (0..cols).map(|c| d19[r * cols + c] as i128 * x[c].raw() as i128).sum::<i128>();
            let a5 = |r: usize| (0..cols).map(|c| d5[r * cols + c] as i128 * x[c].raw() as i128).sum::<i128>();
            let w19: Vec<i32> = (0..rows).map(|r| val(round(i(a19(r)) * i(s19[r] as i128), i(SCALE) << 32usize, Rounding::Nearest, false)) as i32).collect();
            let w19q: Vec<i64> = (0..rows).map(|r| val(round((i(a19(r)) * i(s19[r] as i128)) << 10usize, i(SCALE) << 32usize, Rounding::Nearest, false)) as i64).collect();
            let w5: Vec<i32> = (0..rows).map(|r| val(round(i(a5(r)) * i(s5[r] as i128), i(1) << 32usize, Rounding::Nearest, false)) as i32).collect();
            let w5q: Vec<i64> = (0..rows).map(|r| val(round((i(a5(r)) * i(s5[r] as i128)) << 10usize, i(1) << 32usize, Rounding::Nearest, false)) as i64).collect();
            let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw()).collect::<Vec<i32>>();
            assert_eq!(raw(m19.matvec(x)), w19, "RowScaledTQ19 {rows}x{cols}");
            assert_eq!(raw(m19.matvec_par(x)), w19);
            assert_eq!(m19.matvec_q2f(x), w19q);
            assert_eq!(m19.matvec_q2f_par(x), w19q);
            assert_eq!(raw(m5.matvec(x)), w5, "RowScaledTQ5 {rows}x{cols}");
            assert_eq!(raw(m5.matvec_par(x)), w5);
            assert_eq!(m5.matvec_q2f(x), w5q);
            assert_eq!(m5.matvec_q2f_par(x), w5q);
        }
        let single19: Vec<Vec<X>> = xs.iter().map(|x| m19.matvec(x)).collect();
        assert_eq!(m19.matvec_batch_par(&refs), single19);
        let mut out = vec![X::from_raw(0); refs.len() * rows];
        m19.matvec_batch_par_into(&refs, &mut out);
        assert_eq!(out, single19.concat());
        assert_eq!(m19.matvec_q2f_batch_par(&refs), xs.iter().map(|x| m19.matvec_q2f(x)).collect::<Vec<_>>());
        let single5: Vec<Vec<X>> = xs.iter().map(|x| m5.matvec(x)).collect();
        assert_eq!(m5.matvec_batch_par(&refs), single5);
        let mut out = vec![X::from_raw(0); refs.len() * rows];
        m5.matvec_batch_par_into(&refs, &mut out);
        assert_eq!(out, single5.concat());
    }
}

// ---------------------------------------------------- wide activations

/// Block-scaled and TQ5 matrices for the wide checks, with an exact
/// reference `(numerator per weight, power of two, odd denominator)`.
fn nv(rng: &mut Rng, rows: usize, cols: usize) -> NvFp4Matrix {
    let codes = (0..rows * cols / 2).map(|_| rng.next() as u8).collect();
    let scales = (0..rows * cols / 16).map(|_| loop { let b = ((rng.next() % 5 + 11) << 3 | rng.next() % 8) as u8; if b & 0x7F != 0x7F { break b; } }).collect();
    NvFp4Matrix::from_parts(rows, cols, codes, scales, 0x3EA0_0000 | (rng.next() as u32 & 0x7_FFFF), 0x4528_0000).unwrap()
}

fn mx(rng: &mut Rng, rows: usize, cols: usize, spread: u64) -> MxFp8Matrix {
    let elems = (0..rows * cols).map(|_| loop { let b = rng.next() as u8; if b & 0x7F != 0x7F { break b; } }).collect();
    let scales = (0..rows * cols / 32).map(|_| (110 + rng.next() % (spread + 1)) as u8).collect();
    MxFp8Matrix::from_parts(rows, cols, elems, scales).unwrap()
}

fn nv_ref(m: &NvFp4Matrix, x: &[i128], mode: Rounding) -> Vec<Option<i128>> {
    (0..m.rows())
        .map(|r| {
            let (nums, e, den) = m.dequant_row_exact(r);
            let s = nums.iter().zip(x).fold(i(0), |a, (&n, &v)| a + i(n as i128) * i(v));
            let (n, d) = if e >= 0 { (s << e as usize, i(den as i128)) } else { (s, i(den as i128) << (-e) as usize) };
            opt(round(n, d, mode, false))
        })
        .collect()
}

fn mx_ref(m: &MxFp8Matrix, x: &[i128], mode: Rounding) -> Vec<Option<i128>> {
    (0..m.rows())
        .map(|r| {
            let w = m.dequant_row_exact(r);
            let emin = w.iter().map(|p| p.1).min().unwrap();
            let s = w.iter().zip(x).fold(i(0), |a, (&(wi, e), &v)| a + ((i(wi as i128) * i(v)) << (e - emin) as usize));
            opt(if emin >= 0 { s << emin as usize } else { round(s, i(1) << (-emin) as usize, mode, false) })
        })
        .collect()
}

fn tq5_ref(data: &[i8], scales: &[u64], cols: usize, x: &[i128], mode: Rounding) -> Vec<Option<i128>> {
    scales
        .iter()
        .enumerate()
        .map(|(r, &s)| {
            let acc = (0..cols).fold(i(0), |a, c| a + i(data[r * cols + c] as i128) * i(x[c]));
            opt(round(acc * i(s as i128), i(1) << 32usize, mode, false))
        })
        .collect()
}

macro_rules! wide_suite {
    ($name:ident, $W:ty, $F:literal, $bits:expr) => {
        #[test]
        fn $name() {
            type X = Fixed<$W, $F>;
            let mut rng = Rng(0x31DE + $F);
            let (rows, cols) = (24usize, 256usize);
            let nvm = nv(&mut rng, rows, cols);
            let mxm = mx(&mut rng, rows, cols, 4);
            let mx_wide = mx(&mut rng, rows, cols, 70); // beyond the i128 scalar spread: the I512 path
            let d5: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
            let s5: Vec<u64> = (0..rows).map(|_| (rng.next() >> 33) + 1).collect();
            let tq5 = RowScaledTQ5::from_parts(rows, cols, d5.clone(), s5.clone());
            let mut checked = 0;
            for kind in 0..4 {
                // activations from small to most of the word (results must fit it)
                let bits = [12u32, 30, $bits / 2, $bits - 24][kind];
                let x: Vec<i128> = (0..cols).map(|_| rng.signed(bits)).collect();
                let xs: Vec<X> = x.iter().map(|&v| X::from_raw(v as $W)).collect();
                for mode in [Rounding::Default, Rounding::Nearest] {
                    let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw() as i128).collect::<Vec<i128>>();
                    let (a, b, c, d) = (nvm.clone().with_rounding(mode), mxm.clone().with_rounding(mode), mx_wide.clone().with_rounding(mode), tq5.clone().with_rounding(mode));
                    let run = |f: &dyn Fn() -> Vec<i128>| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok();
                    let fits = |v: Vec<Option<i128>>| v.into_iter().map(|y| y.filter(|&y| <$W>::try_from(y).is_ok())).collect::<Option<Vec<i128>>>();
                    assert_eq!(run(&|| raw(a.matvec_wide_act(&xs))), fits(nv_ref(&a, &x, mode)), "NVFP4 kind {kind} {mode:?}");
                    assert_eq!(run(&|| raw(a.matvec_wide_act_par(&xs))), fits(nv_ref(&a, &x, mode)));
                    assert_eq!(run(&|| raw(b.matvec_wide_act(&xs))), fits(mx_ref(&b, &x, mode)), "MXFP8 kind {kind} {mode:?}");
                    assert_eq!(run(&|| raw(c.matvec_wide_act_par(&xs))), fits(mx_ref(&c, &x, mode)), "MXFP8 wide spread kind {kind} {mode:?}");
                    assert_eq!(run(&|| raw(d.matvec_wide_act(&xs))), fits(tq5_ref(&d5, &s5, cols, &x, mode)), "TQ5 kind {kind} {mode:?}");
                    assert_eq!(run(&|| raw(d.matvec_wide_act_par(&xs))), fits(tq5_ref(&d5, &s5, cols, &x, mode)));
                    checked += 6 * rows;
                }
            }
            // 32-bit activations widened to the larger word: the same values as matvec
            for mode in [Rounding::Default, Rounding::Nearest] {
                let x32: Vec<Fixed<i32, $F>> = (0..cols).map(|_| Fixed::from_raw(rng.signed(16) as i32)).collect();
                let xw: Vec<X> = x32.iter().map(|v| X::from_raw(v.raw() as $W)).collect();
                let (a, b, d) = (nvm.clone().with_rounding(mode), mxm.clone().with_rounding(mode), tq5.clone().with_rounding(mode));
                let as_w = |v: Vec<Fixed<i32, $F>>| v.into_iter().map(|y| y.raw() as i128).collect::<Vec<i128>>();
                let wide = |v: Vec<X>| v.into_iter().map(|y| y.raw() as i128).collect::<Vec<i128>>();
                assert_eq!(wide(a.matvec_wide_act(&xw)), as_w(a.matvec(&x32)), "NVFP4 widened {mode:?}");
                assert_eq!(wide(b.matvec_wide_act(&xw)), as_w(b.matvec(&x32)), "MXFP8 widened {mode:?}");
                assert_eq!(wide(d.matvec_wide_act(&xw)), as_w(d.matvec(&x32)), "TQ5 widened {mode:?}");
            }
            eprintln!("{}: {checked} wide-activation outputs equal the exact references", stringify!($name));
        }
    };
}

wide_suite!(wide_activations_q32_32, i64, 32, 64);
wide_suite!(wide_activations_q64_64, i128, 64, 128);

#[test]
fn tq19_and_row_scaled_tq19_at_q64_64() {
    type X = Fixed<i128, 64>;
    let mut rng = Rng(0x0640);
    let (rows, cols) = (7usize, 300usize);
    let data: Vec<i16> = (0..rows * cols).map(|_| ((rng.next() % 59049) as i32 - 29524) as i16).collect();
    let scales: Vec<u64> = (0..rows).map(|_| (rng.next() >> 34) + 1).collect();
    let x: Vec<i128> = (0..cols).map(|_| rng.signed(90)).collect();
    let xs: Vec<X> = x.iter().map(|&v| X::from_raw(v)).collect();
    let acc = |r: usize| (0..cols).fold(i(0), |a, c| a + i(data[r * cols + c] as i128) * i(x[c]));
    for mode in [Rounding::Default, Rounding::Nearest] {
        let dense = TQ19Matrix::new(rows, cols, data.clone()).with_rounding(mode);
        let want: Vec<i128> = (0..rows).map(|r| val(round(acc(r), i(SCALE), mode, true))).collect();
        let got: Vec<i128> = dense.matvec(&xs).into_iter().map(|y| y.raw()).collect();
        assert_eq!(got, want, "TQ19 Q64.64 {mode:?}");
        let hybrid = HybridTQ19::from_tq19(&dense);
        assert_eq!(hybrid.matvec(&xs).into_iter().map(|y| y.raw()).collect::<Vec<_>>(), want);
        let m19 = RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone()).with_rounding(mode);
        let got19: Vec<i128> = m19.matvec(&xs).into_iter().map(|y| y.raw()).collect();
        let want19: Vec<i128> = match mode {
            // the released rule: the dot rounded toward zero, then floor of the scale
            Rounding::Default => (0..rows).map(|r| val(round(i(val(acc(r) / i(SCALE))) * i(scales[r] as i128), i(1) << 32usize, mode, false))).collect(),
            Rounding::Nearest => (0..rows).map(|r| val(round(acc(r) * i(scales[r] as i128), i(SCALE) << 32usize, mode, false))).collect(),
        };
        assert_eq!(got19, want19, "RowScaledTQ19 Q64.64 {mode:?}");
    }
}
