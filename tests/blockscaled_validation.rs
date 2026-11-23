//! Block-scaled kernels (NVFP4, MXFP8) against exact references.
//!
//! 1. The crate's decoders against Fraction values from the format
//!    definitions (scripts/generate_blockscaled_refs.py), every pattern.
//! 2. Every form against those references, row by row (a one-row matrix per
//!    row pins value or panic for each row; the full matrix panics iff a row
//!    leaves the word).
//! 3. A random differential at checkpoint shapes against an in-test exact
//!    reference built only on the generated tables and hardware division:
//!    every form, batches 1..9 and 64, 1 to 8 threads.
//!
//! No float anywhere.

#[path = "data/blockscaled_refs.rs"]
mod refs;

use g_math::fixed_point::Fixed;
use g_math_inference::bits::{decompose_e4m3, decompose_f32, e2m1_x2, ue8m0_exponent};
use g_math_inference::{MxFp8Matrix, NvFp4Matrix, Rounding};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn silence() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // GMI_LOUD=1 keeps panic messages (to read a failing assertion)
    if std::env::var_os("GMI_LOUD").is_none() {
        ONCE.call_once(|| std::panic::set_hook(Box::new(|_| {})));
    }
}

fn run<T>(f: impl FnOnce() -> T) -> Option<T> {
    catch_unwind(AssertUnwindSafe(f)).ok()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------------------------------------------------------------- decoders

#[test]
fn decoders_equal_the_format_definitions() {
    for c in 0..16u8 {
        assert_eq!(e2m1_x2(c) as i64, refs::E2M1_TIMES_2[c as usize], "E2M1 code {c}");
        assert_eq!(e2m1_x2(c | 0xF0), e2m1_x2(c), "E2M1 reads the low nibble only");
    }
    for b in 0..=255u8 {
        match (decompose_e4m3(b), refs::E4M3_TIMES_512[b as usize]) {
            (Ok((neg, m, e)), Some(v)) => {
                let mag = (m as i64) << (e + 9);
                assert_eq!(if neg { -mag } else { mag }, v, "E4M3 {b:#04x}");
            }
            (Err(_), None) => {}
            (got, want) => panic!("E4M3 {b:#04x}: {got:?} vs {want:?}"),
        }
        match ue8m0_exponent(b) {
            Ok(e) => assert_eq!(e, b as i32 - 127),
            Err(_) => assert_eq!(b, 0xFF),
        }
    }
    for &(bits, num, pow) in refs::F32_EXACT.iter() {
        let (neg, m, e) = decompose_f32(bits).unwrap();
        // value = num * 2^pow; ours = +-m * 2^e
        let ours = if neg { -(m as i128) } else { m as i128 };
        if num == 0 {
            assert_eq!(m, 0, "{bits:#x}");
            continue;
        }
        let (a, b) = if e >= pow { (ours << (e - pow), num) } else { (ours, num << (pow - e)) };
        assert_eq!(a, b, "f32 {bits:#x}");
    }
    assert!(decompose_f32(0x7F80_0000).is_err() && decompose_f32(0x7FC0_0000).is_err() && decompose_f32(0xFF80_0000).is_err());
}

// ------------------------------------------------------ reference cases

fn fits32(v: Option<i128>) -> Option<i32> {
    v.filter(|&v| v >= i32::MIN as i128 && v <= i32::MAX as i128).map(|v| v as i32)
}
fn fits64(v: Option<i128>) -> Option<i64> {
    v.filter(|&v| v >= i64::MIN as i128 && v <= i64::MAX as i128).map(|v| v as i64)
}

/// The six forms of one matrix and activation against per-row expectations.
macro_rules! check_forms {
    ($m:expr, $f:literal, $x:expr, $out:expr, $q2f:expr, $what:expr) => {{
        type X = Fixed<i32, $f>;
        let m = $m;
        let xs: Vec<X> = $x.iter().map(|&v| X::from_raw(v)).collect();
        let narrow: Option<Vec<i32>> = $out.iter().map(|&v| fits32(v)).collect();
        let wide: Option<Vec<i64>> = $q2f.iter().map(|&v| fits64(v)).collect();
        let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw()).collect::<Vec<i32>>();
        assert_eq!(run(|| raw(m.matvec(&xs))), narrow, "{} matvec", $what);
        assert_eq!(run(|| raw(m.matvec_par(&xs))), narrow, "{} matvec_par", $what);
        assert_eq!(run(|| m.matvec_q2f(&xs)), wide, "{} matvec_q2f", $what);
        assert_eq!(run(|| m.matvec_q2f_par(&xs)), wide, "{} matvec_q2f_par", $what);
        for b in [1usize, 2, 5] {
            let batch: Vec<&[X]> = (0..b).map(|_| &xs[..]).collect();
            let want = narrow.clone().map(|v| vec![v; b]);
            assert_eq!(run(|| m.matvec_batch_par(&batch).into_iter().map(raw).collect::<Vec<_>>()), want, "{} batch {b}", $what);
            let flat = run(|| {
                let mut o = vec![X::from_raw(0); b * m.rows()];
                m.matvec_batch_par_into(&batch, &mut o);
                raw(o)
            });
            assert_eq!(flat, want.map(|v| v.concat()), "{} batch_into {b}", $what);
        }
    }};
}

#[test]
fn nvfp4_equals_fraction_references_row_by_row() {
    silence();
    let (mut rows, mut over) = (0, 0);
    for (ci, c) in refs::NV_CASES.iter().enumerate() {
        let what = format!("NV case {ci} ({}) {}x{}", c.mode, c.rows, c.cols);
        let m = NvFp4Matrix::from_parts(c.rows, c.cols, c.codes.to_vec(), c.scales.to_vec(), c.amax, c.den).unwrap();
        match c.frac {
            10 => check_forms!(&m, 10, c.x, c.out, c.q2f, what),
            _ => check_forms!(&m, 16, c.x, c.out, c.q2f, what),
        }
        // the same matrix rounding to nearest (ties toward +infinity)
        let mn = m.clone().with_rounding(Rounding::Nearest);
        let wn = format!("{what} nearest");
        match c.frac {
            10 => check_forms!(&mn, 10, c.x, c.near, c.q2f_near, wn),
            _ => check_forms!(&mn, 16, c.x, c.near, c.q2f_near, wn),
        }
        // one row at a time: each row's own value or panic
        for r in 0..c.rows {
            let (h, s) = (c.cols / 2, c.cols / 16);
            let one = NvFp4Matrix::from_parts(1, c.cols, c.codes[r * h..(r + 1) * h].to_vec(), c.scales[r * s..(r + 1) * s].to_vec(), c.amax, c.den).unwrap();
            match c.frac {
                10 => check_forms!(&one, 10, c.x, &c.out[r..r + 1], &c.q2f[r..r + 1], format!("{what} row {r}")),
                _ => check_forms!(&one, 16, c.x, &c.out[r..r + 1], &c.q2f[r..r + 1], format!("{what} row {r}")),
            }
            rows += 1;
            over += fits32(c.out[r]).is_none() as usize;
        }
    }
    eprintln!("NVFP4: {} cases, {rows} rows, {over} rows panic as expected", refs::NV_CASES.len());
    assert!(over * 4 < rows);
}

#[test]
fn mxfp8_equals_fraction_references_row_by_row() {
    silence();
    let (mut rows, mut over) = (0, 0);
    for (ci, c) in refs::MX_CASES.iter().enumerate() {
        let what = format!("MX case {ci} ({}) {}x{}", c.mode, c.rows, c.cols);
        let m = MxFp8Matrix::from_parts(c.rows, c.cols, c.elems.to_vec(), c.scales.to_vec()).unwrap();
        match c.frac {
            10 => check_forms!(&m, 10, c.x, c.out, c.q2f, what),
            _ => check_forms!(&m, 16, c.x, c.out, c.q2f, what),
        }
        let mn = m.clone().with_rounding(Rounding::Nearest);
        let wn = format!("{what} nearest");
        match c.frac {
            10 => check_forms!(&mn, 10, c.x, c.near, c.q2f_near, wn),
            _ => check_forms!(&mn, 16, c.x, c.near, c.q2f_near, wn),
        }
        for r in 0..c.rows {
            let s = c.cols / 32;
            let one = MxFp8Matrix::from_parts(1, c.cols, c.elems[r * c.cols..(r + 1) * c.cols].to_vec(), c.scales[r * s..(r + 1) * s].to_vec()).unwrap();
            match c.frac {
                10 => check_forms!(&one, 10, c.x, &c.out[r..r + 1], &c.q2f[r..r + 1], format!("{what} row {r}")),
                _ => check_forms!(&one, 16, c.x, &c.out[r..r + 1], &c.q2f[r..r + 1], format!("{what} row {r}")),
            }
            rows += 1;
            over += fits32(c.out[r]).is_none() as usize;
        }
    }
    eprintln!("MXFP8: {} cases, {rows} rows, {over} rows panic as expected", refs::MX_CASES.len());
    assert!(over * 4 < rows);
}

#[test]
fn construction_refuses_nan_and_zero_denominator() {
    let codes = vec![0x21u8; 8];
    assert!(NvFp4Matrix::from_parts(1, 16, codes.clone(), vec![0x7F], 0x3F80_0000, 0x4528_0000).is_err());
    assert!(NvFp4Matrix::from_parts(1, 16, codes.clone(), vec![0xFF], 0x3F80_0000, 0x4528_0000).is_err());
    assert!(NvFp4Matrix::from_parts(1, 16, codes.clone(), vec![0x38], 0x7F80_0000, 0x4528_0000).is_err());
    assert!(NvFp4Matrix::from_parts(1, 16, codes.clone(), vec![0x38], 0x3F80_0000, 0x7FC0_0000).is_err());
    assert!(NvFp4Matrix::from_parts(1, 16, codes.clone(), vec![0x38], 0x3F80_0000, 0x0000_0000).is_err());
    assert!(NvFp4Matrix::from_parts(1, 16, codes, vec![0x38], 0x3F80_0000, 0x8000_0000).is_err());
    assert!(MxFp8Matrix::from_parts(1, 32, vec![0x7F; 32], vec![127]).is_err());
    assert!(MxFp8Matrix::from_parts(1, 32, vec![0x38; 32], vec![255]).is_err());
    assert!(MxFp8Matrix::from_parts(1, 32, vec![0xFF; 32], vec![127]).is_err());
}

// ------------------------------------------------- random differential

/// Exact reference, built on the generated tables only: `T = sum (2 E2M1) (2^9 E4M3) x`,
/// output `floor(T * amax / den / 2^10 * 2^extra)`.
fn nv_reference(c: &NvParts, x: &[i32], extra: i32) -> Vec<i128> {
    let f32x = |bits: u32| -> (i128, i32) {
        let (e, m) = ((bits >> 23) & 0xFF, (bits & 0x7F_FFFF) as i128);
        let (m, p) = if e == 0 { (m, -149) } else { (m | 0x80_0000, e as i32 - 150) };
        (if bits >> 31 == 1 { -m } else { m }, p)
    };
    let (na, pa) = f32x(c.amax);
    let (nd, pd) = f32x(c.den);
    let k = pa - pd - 10 + extra;
    (0..c.rows)
        .map(|r| {
            let mut t: i128 = 0;
            for col in 0..c.cols {
                let byte = c.codes[(r * c.cols + col) / 2];
                let code = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                let w = refs::E2M1_TIMES_2[code as usize] as i128 * refs::E4M3_TIMES_512[c.scales[(r * c.cols + col) / 16] as usize].unwrap() as i128;
                t += w * x[col] as i128;
            }
            let (mut n, mut d) = (t * na, nd);
            if d < 0 {
                n = -n;
                d = -d;
            }
            if k >= 0 {
                (n << k).div_euclid(d)
            } else {
                n.div_euclid(d).div_euclid(1i128 << (-k))
            }
        })
        .collect()
}

/// `T = sum_b 2^(s_b - smin) sum (2^9 E4M3) x`, output `floor(T * 2^(smin - 136 + extra))`.
fn mx_reference(c: &MxParts, x: &[i32], extra: i32) -> Vec<i128> {
    let smin = *c.scales.iter().min().unwrap() as i32;
    let k = smin - 136 + extra;
    (0..c.rows)
        .map(|r| {
            let mut t: i128 = 0;
            for b in 0..c.cols / 32 {
                let mut bs: i128 = 0;
                for j in 0..32 {
                    let col = b * 32 + j;
                    bs += refs::E4M3_TIMES_512[c.elems[r * c.cols + col] as usize].unwrap() as i128 * x[col] as i128;
                }
                t += bs << (c.scales[r * c.cols / 32 + b] as i32 - smin);
            }
            if k >= 0 { t << k } else { t >> (-k) }
        })
        .collect()
}

struct NvParts {
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    scales: Vec<u8>,
    amax: u32,
    den: u32,
}
struct MxParts {
    rows: usize,
    cols: usize,
    elems: Vec<u8>,
    scales: Vec<u8>,
}

fn activation(rng: &mut Rng, n: usize, kind: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            let bits = match kind {
                0 => 14,
                1 => if r % 10 == 0 { 24 } else { 14 },
                2 => 24,
                _ => if r % 50 == 0 { 31 } else { 20 },
            };
            let m = ((r >> 8) & ((1u64 << bits) - 1)) as i64;
            (if r & 1 == 1 { -m } else { m }).clamp(i32::MIN as i64, i32::MAX as i64) as i32
        })
        .collect()
}

/// Every form against the reference, inside pools of 1, 2, 3 and 8 threads.
fn differential<M: Sync>(
    m: &M,
    rows: usize,
    acts: &[Vec<i32>],
    reference: &dyn Fn(&[i32], i32) -> Vec<i128>,
    forms: &(dyn Fn(&M, &[Vec<i32>]) -> (Vec<Option<Vec<i32>>>, Option<Vec<Vec<i32>>>, Option<Vec<i32>>, Vec<Option<Vec<i64>>>) + Sync),
    what: &str,
) -> (usize, usize) {
    let mut cases = (0, 0);
    let want: Vec<Option<Vec<i32>>> = acts.iter().map(|x| reference(x, 0).into_iter().map(|v| fits32(Some(v))).collect()).collect();
    let want_q: Vec<Option<Vec<i64>>> = acts.iter().map(|x| reference(x, 10).into_iter().map(|v| fits64(Some(v))).collect()).collect();
    let want_batch: Option<Vec<Vec<i32>>> = want.iter().cloned().collect();
    for threads in [1usize, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let (single, batch, flat, q2f) = pool.install(|| forms(m, acts));
        assert_eq!(single, want, "{what} matvec/matvec_par, {threads} threads");
        assert_eq!(batch, want_batch, "{what} matvec_batch_par, {threads} threads");
        assert_eq!(flat, want_batch.clone().map(|v| v.concat()), "{what} matvec_batch_par_into, {threads} threads");
        assert_eq!(q2f, want_q, "{what} matvec_q2f(_par), {threads} threads");
        cases.0 += acts.len() * rows;
    }
    cases.1 = want.iter().filter(|w| w.is_none()).count();
    let _ = rows;
    cases
}

macro_rules! forms_q22_10 {
    () => {
        |m, acts: &[Vec<i32>]| {
            type X = Fixed<i32, 10>;
            let xs: Vec<Vec<X>> = acts.iter().map(|a| a.iter().map(|&v| X::from_raw(v)).collect()).collect();
            let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw()).collect::<Vec<i32>>();
            let single: Vec<Option<Vec<i32>>> = xs
                .iter()
                .map(|x| {
                    let a = run(|| raw(m.matvec(x)));
                    let b = run(|| raw(m.matvec_par(x)));
                    assert_eq!(a, b, "matvec vs matvec_par");
                    a
                })
                .collect();
            let refs: Vec<&[X]> = xs.iter().map(|v| &v[..]).collect();
            let batch = run(|| m.matvec_batch_par(&refs).into_iter().map(raw).collect::<Vec<_>>());
            let flat = run(|| {
                let mut o = vec![X::from_raw(0); refs.len() * m.rows()];
                m.matvec_batch_par_into(&refs, &mut o);
                raw(o)
            });
            let q2f: Vec<Option<Vec<i64>>> = xs
                .iter()
                .map(|x| {
                    let a = run(|| m.matvec_q2f(x));
                    let b = run(|| m.matvec_q2f_par(x));
                    assert_eq!(a, b, "q2f vs q2f_par");
                    a
                })
                .collect();
            (single, batch, flat, q2f)
        }
    };
}

#[test]
fn nvfp4_random_differential_at_checkpoint_shapes() {
    silence();
    let mut rng = Rng(0x4E56_4650_3400);
    let shapes: &[(usize, usize, usize)] = &[(1, 16, 9), (3, 32, 5), (4, 48, 64), (7, 160, 8), (13, 2304, 9), (64, 1152, 64), (130, 256, 7), (2304, 2304, 3), (2304, 1152, 64)];
    let (mut total, mut simd, mut scalar) = (0usize, 0usize, 0usize);
    for (si, &(rows, cols, nb)) in shapes.iter().enumerate() {
        for mode in 0..2 {
            let scales: Vec<u8> = (0..rows * cols / 16)
                .map(|_| loop {
                    let b = if mode == 0 { ((rng.below(5) + 11) << 3 | rng.below(8)) as u8 } else { rng.below(256) as u8 };
                    if b & 0x7F != 0x7F {
                        break b;
                    }
                })
                .collect();
            let p = NvParts { rows, cols, codes: (0..rows * cols / 2).map(|_| rng.next() as u8).collect(), scales, amax: 0x3EA0_0000 | (rng.next() as u32 & 0x7_FFFF), den: 0x4528_0000 };
            let m = NvFp4Matrix::from_parts(rows, cols, p.codes.clone(), p.scales.clone(), p.amax, p.den).unwrap();
            if m.simd_eligible() { simd += 1 } else { scalar += 1 }
            let acts: Vec<Vec<i32>> = (0..nb).map(|k| activation(&mut rng, cols, (k as u64 + si as u64) % 4)).collect();
            let (n, _) = differential(&m, rows, &acts, &|x, e| nv_reference(&p, x, e), &forms_q22_10!(), &format!("NV {rows}x{cols} mode {mode}"));
            total += n;
        }
    }
    eprintln!("NVFP4 differential: {total} row outputs equal the exact reference ({simd} SIMD-eligible matrices, {scalar} scalar)");
    assert!(simd > 0 && scalar > 0);
}

#[test]
fn mxfp8_random_differential_at_checkpoint_shapes() {
    silence();
    let mut rng = Rng(0x4D58_4650_3800);
    let shapes: &[(usize, usize, usize)] = &[(1, 32, 9), (3, 64, 5), (4, 96, 64), (7, 160, 8), (13, 2304, 9), (64, 1152, 64), (130, 256, 7), (2304, 2304, 3), (2304, 1152, 64)];
    let (mut total, mut simd, mut scalar) = (0usize, 0usize, 0usize);
    for (si, &(rows, cols, nb)) in shapes.iter().enumerate() {
        // scale spreads 1, 4, 5, 6, 7 reach the 3-, 4-, 5-, 8- and 15-class SIMD forms; 8 and 30 the scalar path
        let spreads: &[u64] = if rows * cols > 1 << 20 { &[1, 4, 30] } else { &[1, 4, 5, 6, 7, 8, 12, 13, 30] };
        for (mode, &spread) in spreads.iter().enumerate() {
            let mut scales: Vec<u8> = (0..rows * cols / 32).map(|_| (100 + rng.below(spread + 1)) as u8).collect();
            // pin the full spread: one block at each end
            scales[0] = 100;
            let last = scales.len() - 1;
            scales[last] = (100 + spread) as u8;
            let p = MxParts {
                rows,
                cols,
                elems: (0..rows * cols).map(|_| loop { let b = rng.next() as u8; if b & 0x7F != 0x7F && b & 0x7F != 0 { break b; } }).collect(),
                scales,
            };
            let m = MxFp8Matrix::from_parts(rows, cols, p.elems.clone(), p.scales.clone()).unwrap();
            let actual = p.scales.iter().max().unwrap() - p.scales.iter().min().unwrap();
            assert_eq!(m.simd_eligible(), actual <= 7, "spread {actual}");
            if m.simd_eligible() { simd += 1 } else { scalar += 1 }
            let acts: Vec<Vec<i32>> = (0..nb).map(|k| activation(&mut rng, cols, (k as u64 + si as u64) % 4)).collect();
            let (n, _) = differential(&m, rows, &acts, &|x, e| mx_reference(&p, x, e), &forms_q22_10!(), &format!("MX {rows}x{cols} mode {mode}"));
            total += n;
        }
    }
    eprintln!("MXFP8 differential: {total} row outputs equal the exact reference ({simd} SIMD-eligible matrices, {scalar} scalar)");
    assert!(simd > 0 && scalar > 0);
}

// ------------------------------------------------- real checkpoint slices

/// Real expert bytes (not in the repository): with `GMI_REAL_SLICES=<dir>`
/// holding the slices and `real_refs.txt` from
/// scripts/generate_blockscaled_real_refs.py, every form equals the exact
/// outputs. Skipped without the variable.
#[test]
fn real_checkpoint_slices() {
    let Ok(dir) = std::env::var("GMI_REAL_SLICES") else {
        eprintln!("real_checkpoint_slices: GMI_REAL_SLICES not set, skipped");
        return;
    };
    silence();
    let rd = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap();
    let text = std::fs::read_to_string(format!("{dir}/real_refs.txt")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let nums = |l: &str| l.split_whitespace().skip(1).map(|v| v.parse::<i128>().unwrap()).collect::<Vec<i128>>();
    let mut checked = 0;
    for chunk in lines.chunks(4) {
        let head: Vec<&str> = chunk[0].split_whitespace().collect();
        let (rows, cols): (usize, usize) = (head[1].parse().unwrap(), head[2].parse().unwrap());
        let x: Vec<i32> = nums(chunk[1]).into_iter().map(|v| v as i32).collect();
        let out: Vec<Option<i128>> = nums(chunk[2]).into_iter().map(Some).collect();
        let q2f: Vec<Option<i128>> = nums(chunk[3]).into_iter().map(Some).collect();
        if head[0] == "NV" {
            let m = NvFp4Matrix::from_parts(
                rows,
                cols,
                rd("0.mlp.experts.w13_weight.e0.bin")[..rows * cols / 2].to_vec(),
                rd("0.mlp.experts.w13_weight_scale.e0.bin")[..rows * cols / 16].to_vec(),
                head[3].parse().unwrap(),
                head[4].parse().unwrap(),
            )
            .unwrap();
            assert!(m.simd_eligible());
            check_forms!(&m, 10, x, out, q2f, "real NVFP4");
        } else {
            let m = MxFp8Matrix::from_parts(rows, cols, rd("20.mlp.experts.w13_weight.e0.bin")[..rows * cols].to_vec(), rd("20.mlp.experts.w13_weight_scale.e0.bin")[..rows * cols / 32].to_vec()).unwrap();
            assert!(m.simd_eligible());
            check_forms!(&m, 10, x, out, q2f, "real MXFP8");
        }
        checked += rows;
    }
    eprintln!("real checkpoint slices: {checked} row outputs equal the exact Fraction values (x 4 forms + batches)");
    assert!(checked > 0);
}
