//! Gate for `Ptq1Matrix` (PTQ1_0 ternary blocks, optional Walsh-Hadamard
//! input transform).
//!
//! 1. Exact references (scripts/generate_ptq1_refs.py, fractions from the
//!    definitions): floor, nearest and 2F outputs of every form.
//! 2. A large random matrix (parallel paths) against an independent integer
//!    reference written here (llama.cpp's decode loop, i128 sums, the
//!    transform as a plain butterfly), batches 1..9 and 64, 1/2/3/8 threads,
//!    single == batched bit for bit.
//! 3. The construction errors.

#[path = "data/ptq1_refs.rs"]
mod refs;

use g_math::fixed_point::{Fixed, FixedNum};
use g_math_inference::{Ptq1Matrix, Rounding, PTQ1_BLOCK_BYTES};

fn check_case<const F: u32>(c: &refs::Case) -> usize
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    let mut m = Ptq1Matrix::from_parts(c.rows, c.cols, c.blocks.to_vec()).unwrap();
    if c.hadamard != 0 {
        m = m.with_hadamard(c.hadamard, c.signs.map(|s| s.to_vec())).unwrap();
    }
    let near = m.clone().with_rounding(Rounding::Nearest);
    let xs: Vec<Vec<Fixed<i32, F>>> = c.acts.iter().map(|a| a.iter().map(|&v| Fixed::from_raw(v)).collect()).collect();
    let mut checked = 0;
    for (k, x) in xs.iter().enumerate() {
        let want = c.outs[k];
        if want.iter().any(|w| w.is_none()) {
            // an output out of range: every form panics
            assert!(std::panic::catch_unwind(|| m.matvec(x)).is_err() || std::panic::catch_unwind(|| near.matvec(x)).is_err());
            continue;
        }
        let fl: Vec<i64> = m.matvec(x).iter().map(|v| v.raw() as i64).collect();
        let fp: Vec<i64> = m.matvec_par(x).iter().map(|v| v.raw() as i64).collect();
        let nr: Vec<i64> = near.matvec(x).iter().map(|v| v.raw() as i64).collect();
        let q2: Vec<i64> = m.matvec_q2f(x);
        let q2p: Vec<i64> = m.matvec_q2f_par(x);
        for r in 0..c.rows {
            let (wf, wn, wq) = want[r].unwrap();
            assert_eq!(fl[r], wf, "{}x{} h {} act {k} row {r}: floor", c.rows, c.cols, c.hadamard);
            assert_eq!(fp[r], wf);
            assert_eq!(nr[r], wn, "{}x{} h {} act {k} row {r}: nearest", c.rows, c.cols, c.hadamard);
            assert_eq!(q2[r], wq, "{}x{} h {} act {k} row {r}: q2f", c.rows, c.cols, c.hadamard);
            assert_eq!(q2p[r], wq);
            checked += 3;
        }
    }
    // the batched forms over every in-range activation at once
    let ok: Vec<usize> = (0..xs.len()).filter(|&k| c.outs[k].iter().all(|w| w.is_some())).collect();
    let batch: Vec<&[Fixed<i32, F>]> = ok.iter().map(|&k| &xs[k][..]).collect();
    let b = m.matvec_batch_par(&batch);
    let bn = near.matvec_batch_par(&batch);
    let bq = m.matvec_q2f_batch_par(&batch);
    for (i, &k) in ok.iter().enumerate() {
        for r in 0..c.rows {
            let (wf, wn, wq) = c.outs[k][r].unwrap();
            assert_eq!(b[i][r].raw() as i64, wf);
            assert_eq!(bn[i][r].raw() as i64, wn);
            assert_eq!(bq[i][r], wq);
        }
    }
    checked
}

#[test]
fn equals_the_exact_fraction_references() {
    let mut checked = 0;
    for c in refs::CASES {
        checked += match c.frac {
            10 => check_case::<10>(c),
            16 => check_case::<16>(c),
            f => panic!("no case at F = {f}"),
        };
    }
    assert!(checked >= 240, "only {checked} reference outputs checked");
}

// ---------------------------------------------------------------------------
// an independent integer reference
// ---------------------------------------------------------------------------

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 17
}

/// The trits of one 28-byte block: llama.cpp's loop.
fn block_trits(b: &[u8]) -> Vec<i64> {
    let pow3 = [1u8, 3, 9, 27, 81];
    let mut y = Vec::with_capacity(128);
    for (start, c) in [(0usize, 16usize), (16, 8)] {
        for n in 0..5 {
            for m in 0..c {
                y.push(((((b[start + m].wrapping_mul(pow3[n])) as u16 * 3) >> 8) as i64) - 1);
            }
        }
    }
    for n in 0..4 {
        for h in 0..2 {
            y.push(((((b[24 + h].wrapping_mul(pow3[n])) as u16 * 3) >> 8) as i64) - 1);
        }
    }
    y
}

/// A binary16 scale times 2^24, as an integer.
fn scale_s24(h: u16) -> i128 {
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as i128;
    let v = if e == 0 { m } else { (1024 + m) << (e - 1) };
    if h & 0x8000 != 0 { -v } else { v }
}

/// `floor(sum_c w(r, c) u_c * 2^extra)` per row, i128 throughout.
fn reference(blocks: &[u8], rows: usize, cols: usize, had: Option<(usize, &[i8])>, x: &[i32], extra: u32) -> Vec<i128> {
    let mut u: Vec<i128> = x.iter().map(|&v| v as i128).collect();
    let mut shift = 24u32;
    if let Some((n, signs)) = had {
        for (v, &s) in u.iter_mut().zip(signs) {
            *v *= s as i128;
        }
        for seg in u.chunks_mut(n) {
            let mut h = 1;
            while h < n {
                for i in (0..n).step_by(2 * h) {
                    for j in i..i + h {
                        let (a, b) = (seg[j], seg[j + h]);
                        seg[j] = a + b;
                        seg[j + h] = a - b;
                    }
                }
                h *= 2;
            }
        }
        shift += n.trailing_zeros() / 2;
    }
    let nb = cols / 128;
    (0..rows)
        .map(|r| {
            let mut t = 0i128;
            for b in 0..nb {
                let blk = &blocks[(r * nb + b) * PTQ1_BLOCK_BYTES..(r * nb + b + 1) * PTQ1_BLOCK_BYTES];
                let tr = block_trits(blk);
                let s: i128 = (0..128).map(|j| tr[j] as i128 * u[b * 128 + j]).sum();
                t += scale_s24(u16::from_le_bytes([blk[26], blk[27]])) * s;
            }
            (t << extra) >> shift
        })
        .collect()
}

fn random_blocks(seed: &mut u64, rows: usize, cols: usize, max_exp: u64) -> Vec<u8> {
    let mut blocks = vec![0u8; rows * cols / 128 * PTQ1_BLOCK_BYTES];
    for b in blocks.chunks_exact_mut(PTQ1_BLOCK_BYTES) {
        for v in b.iter_mut().take(26) {
            *v = lcg(seed) as u8;
        }
        let r = lcg(seed);
        let h = ((r & 1) << 15) as u16 | (((r >> 1) % max_exp) << 10) as u16 | ((r >> 8) & 1023) as u16;
        b[26..28].copy_from_slice(&h.to_le_bytes());
    }
    blocks
}

#[test]
fn large_random_matrices_every_batch_and_thread_count() {
    let mut seed = 0x7E57_1125u64;
    // 2304 x 2048: 1 MiB of blocks, so the row tasks run in parallel
    for &(rows, cols, had, max_exp, bits) in &[
        (2304usize, 2048usize, Some(1024usize), 8u64, 22u32),
        (2304, 2048, None, 10, 28),
        (1031, 1536, Some(64), 6, 24),
    ] {
        let blocks = random_blocks(&mut seed, rows, cols, max_exp);
        let signs: Vec<i8> = (0..cols).map(|_| if lcg(&mut seed) & 1 == 0 { 1 } else { -1 }).collect();
        let mut m = Ptq1Matrix::from_parts(rows, cols, blocks.clone()).unwrap();
        if let Some(h) = had {
            m = m.with_hadamard(h, Some(signs.clone())).unwrap();
        }
        let xs: Vec<Vec<Fixed<i32, 10>>> = (0..64)
            .map(|_| (0..cols).map(|_| Fixed::from_raw(((lcg(&mut seed) as i64 % (1i64 << bits)) - (1i64 << (bits - 1))) as i32)).collect())
            .collect();
        let raw0: Vec<i32> = xs[0].iter().map(|v| v.raw()).collect();
        let want = reference(&blocks, rows, cols, had.map(|h| (h, &signs[..])), &raw0, 0);
        let want_q = reference(&blocks, rows, cols, had.map(|h| (h, &signs[..])), &raw0, 10);
        let single = m.matvec(&xs[0]);
        for r in 0..rows {
            assert_eq!(single[r].raw() as i128, want[r], "{rows}x{cols} row {r}");
        }
        let q = m.matvec_q2f_par(&xs[0]);
        for r in 0..rows {
            assert_eq!(q[r] as i128, want_q[r]);
        }
        let singles: Vec<Vec<Fixed<i32, 10>>> = xs.iter().map(|x| m.matvec_par(x)).collect();
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                for n in (1..=9).chain([64]) {
                    let batch: Vec<&[Fixed<i32, 10>]> = xs[..n].iter().map(|v| &v[..]).collect();
                    let out = m.matvec_batch_par(&batch);
                    for k in 0..n {
                        assert_eq!(out[k], singles[k], "{rows}x{cols} threads {threads} batch {n} vector {k}");
                    }
                }
            });
        }
        let batch: Vec<&[Fixed<i32, 10>]> = xs[..8].iter().map(|v| &v[..]).collect();
        let bq = m.matvec_q2f_batch_par(&batch);
        for k in 0..8 {
            assert_eq!(bq[k], m.matvec_q2f(&xs[k]));
        }
    }
}

#[test]
fn construction_errors() {
    let blk = vec![0u8; PTQ1_BLOCK_BYTES];
    assert!(Ptq1Matrix::from_parts(1, 100, blk.clone()).is_err(), "cols not a multiple of 128");
    assert!(Ptq1Matrix::from_parts(2, 128, blk.clone()).is_err(), "byte count");
    let mut inf = blk.clone();
    inf[26..28].copy_from_slice(&0x7C00u16.to_le_bytes());
    assert!(Ptq1Matrix::from_parts(1, 128, inf).is_err(), "infinite scale");
    let mut nan = blk.clone();
    nan[26..28].copy_from_slice(&0xFE00u16.to_le_bytes());
    assert!(Ptq1Matrix::from_parts(1, 128, nan).is_err(), "NaN scale");
    let m = Ptq1Matrix::from_parts(1, 1024, vec![0u8; 8 * PTQ1_BLOCK_BYTES]).unwrap();
    assert!(m.clone().with_hadamard(512, None).is_err(), "odd power of two");
    assert!(m.clone().with_hadamard(8192, None).is_err(), "too large");
    assert!(m.clone().with_hadamard(1024, Some(vec![1; 1023])).is_err(), "signs length");
    assert!(m.clone().with_hadamard(1024, Some(vec![0; 1024])).is_err(), "signs values");
    assert!(m.clone().with_hadamard(1024, Some(vec![-1; 1024])).is_ok());
    let m3 = Ptq1Matrix::from_parts(1, 384, vec![0u8; 3 * PTQ1_BLOCK_BYTES]).unwrap();
    assert!(m3.with_hadamard(256, None).is_err(), "block does not divide cols");
}

/// Worst case for the 32-bit lanes: every trit +1 (or the row's sign), the
/// row's block scales four binades apart with full mantissas (folded weights
/// 2047 * 2^4), every activation limb at its largest magnitude; plus rows
/// whose small blocks fall outside the fold (added per block). Against the
/// independent reference, batches 1..8.
#[test]
fn adversarial_lanes_and_outlier_blocks() {
    let (rows, cols) = (6usize, 2048usize);
    let nb = cols / 128;
    let mut blocks = vec![0u8; rows * nb * PTQ1_BLOCK_BYTES];
    for r in 0..rows {
        for b in 0..nb {
            let blk = &mut blocks[(r * nb + b) * PTQ1_BLOCK_BYTES..(r * nb + b + 1) * PTQ1_BLOCK_BYTES];
            // all +1 trits: 242 -> 255 for five-trit bytes, 80 * 3 -> 253 for four-trit bytes; row 1 all -1 (byte 0)
            let (five, four) = if r == 1 { (0u8, 0u8) } else { (255u8, 253u8) };
            blk[..24].fill(five);
            blk[24..26].fill(four);
            // mantissa all ones; exponent field 10 or 14 (four binades), rows 4..6 also far smaller blocks
            let e: u16 = match (r, b % 4) {
                (r, 0) if r >= 4 => 2, // 12 binades below: outside the fold
                (_, 1) => 14,
                _ => 10,
            };
            let h = (e << 10) | 1023 | if r == 5 && b % 3 == 0 { 0x8000 } else { 0 };
            blk[26..28].copy_from_slice(&h.to_le_bytes());
        }
    }
    let m = Ptq1Matrix::from_parts(rows, cols, blocks.clone()).unwrap();
    let mut checked = 0;
    for &mag in &[(1i64 << 31) - 1, (1 << 22) - 1, (1 << 11) - 1, 1 << 15, -(1i64 << 31)] {
        let xs: Vec<Vec<Fixed<i32, 10>>> = (0..8)
            .map(|k| (0..cols).map(|c| Fixed::from_raw(if k % 2 == 1 && c % 5 == 0 { (-mag).clamp(i32::MIN as i64, i32::MAX as i64) } else { mag.clamp(i32::MIN as i64, i32::MAX as i64) } as i32)).collect())
            .collect();
        for n in 1..=8 {
            let batch: Vec<&[Fixed<i32, 10>]> = xs[..n].iter().map(|v| &v[..]).collect();
            let out = m.matvec_q2f_batch_par(&batch);
            for k in 0..n {
                let raw: Vec<i32> = xs[k].iter().map(|v| v.raw()).collect();
                let want = reference(&blocks, rows, cols, None, &raw, 10);
                for r in 0..rows {
                    assert_eq!(out[k][r] as i128, want[r], "mag {mag} batch {n} vector {k} row {r}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 5 * 36 * 6);
}
