//! Gate for the batched RowScaledTQ19 kernel and the row-subset forms.
//!
//! - every form == mpmath references (scripts/generate_rowscaled_batch_refs.py:
//!   the released rounding rules, 80 digits, integer cross-check), both
//!   roundings, batches of nine, column tails, Q22.10 and Q16.16;
//! - `matvec_batch_par`, `matvec_batch_par_into`, `matvec_q2f_batch_par` ==
//!   `matvec` / `matvec_q2f` bit for bit, batches 1..9, 16 and 64, 1/2/3/8
//!   threads, both roundings, shapes with a column tail (not a multiple of 8);
//! - the batched sums == an independent i128 reference (rounding `Nearest`,
//!   whose value is the exact sum rounded once);
//! - `matvec_rows` / `matvec_rows_q2f` == the same rows of `matvec` /
//!   `matvec_q2f`, duplicates and any order; out of range panics.

#[path = "data/rowscaled_batch_refs.rs"]
mod refs;

use g_math::fixed_point::{Fixed, FixedNum};
use g_math_inference::{Rounding, RowScaledTQ19};

fn check_refs<const F: u32>(c: &refs::Case) -> usize
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    let base = RowScaledTQ19::from_parts(c.rows, c.cols, c.data.to_vec(), c.scales.to_vec());
    let xs: Vec<Vec<Fixed<i32, F>>> = c.acts.iter().map(|a| a.iter().map(|&v| Fixed::from_raw(v)).collect()).collect();
    let batch: Vec<&[Fixed<i32, F>]> = xs.iter().map(|v| &v[..]).collect();
    let mut n = 0;
    for (rounding, pick) in [(Rounding::Default, 0usize), (Rounding::Nearest, 1)] {
        let m = base.clone().with_rounding(rounding);
        let out = m.matvec_batch_par(&batch);
        let outq = m.matvec_q2f_batch_par(&batch);
        let all: Vec<usize> = (0..c.rows).rev().collect();
        for (k, x) in xs.iter().enumerate() {
            let single = m.matvec(x);
            let sub = m.matvec_rows(&all, x);
            let subq = m.matvec_rows_q2f(&all, x);
            for r in 0..c.rows {
                let (dm, dq, nm, nq) = c.outs[k][r];
                let (wm, wq) = if pick == 0 { (dm, dq) } else { (nm, nq) };
                assert_eq!(out[k][r].raw(), wm, "{}x{} {rounding:?} vector {k} row {r}", c.rows, c.cols);
                assert_eq!(outq[k][r], wq, "{}x{} {rounding:?} q2f vector {k} row {r}", c.rows, c.cols);
                assert_eq!(single[r].raw(), wm);
                assert_eq!(sub[c.rows - 1 - r].raw(), wm);
                assert_eq!(subq[c.rows - 1 - r], wq);
                n += 2;
            }
        }
    }
    n
}

#[test]
fn every_form_equals_the_mpmath_references() {
    let mut n = 0;
    for c in refs::CASES {
        n += match c.frac {
            10 => check_refs::<10>(c),
            16 => check_refs::<16>(c),
            f => panic!("no case at F = {f}"),
        };
    }
    assert_eq!(n, 2 * 2 * 9 * (12 + 5 + 7));
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 17
}

fn matrix(seed: &mut u64, rows: usize, cols: usize) -> RowScaledTQ19 {
    let data: Vec<i16> = (0..rows * cols).map(|_| ((lcg(seed) % 59049) as i64 - 29524) as i16).collect();
    // relative scales from 1/16 to 2, in Q32.32
    let scales: Vec<u64> = (0..rows).map(|_| (1u64 << 28) + lcg(seed) % (1u64 << 33)).collect();
    RowScaledTQ19::from_parts(rows, cols, data, scales)
}

fn acts<const F: u32>(seed: &mut u64, cols: usize, bits: u32) -> Vec<Fixed<i32, F>>
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    (0..cols).map(|_| Fixed::from_raw(((lcg(seed) as i64 % (1i64 << bits)) - (1i64 << (bits - 1))) as i32)).collect()
}

/// `floor(sum raw * x * s / (3^9 * 2^32) * 2^extra + 1/2)`: rounding Nearest, i128 / exact.
fn nearest_ref(m: &RowScaledTQ19, r: usize, x: &[i32], extra: u32) -> i128 {
    let row = &m.data()[r * m.cols()..(r + 1) * m.cols()];
    let acc: i128 = row.iter().zip(x).map(|(&w, &a)| w as i128 * a as i128).sum();
    // acc * s * 2^extra / (19683 * 2^32), nearest ties toward +inf: floor((2 n + d) / 2 d)
    let n = acc * m.scales_q32()[r] as i128 * (1i128 << extra);
    let d = 19683i128 << 32;
    (2 * n + d).div_euclid(2 * d)
}

#[test]
fn batched_equals_single_every_batch_thread_and_rounding() {
    let mut seed = 0xBA7C_0001u64;
    for &(rows, cols) in &[(1024usize, 2304usize), (384, 2304), (517, 1029), (3, 7)] {
        let base = matrix(&mut seed, rows, cols);
        let xs: Vec<Vec<Fixed<i32, 10>>> = (0..64).map(|_| acts::<10>(&mut seed, cols, 20)).collect();
        for rounding in [Rounding::Default, Rounding::Nearest] {
            let m = base.clone().with_rounding(rounding);
            let singles: Vec<Vec<Fixed<i32, 10>>> = xs.iter().map(|x| m.matvec(x)).collect();
            let singles_q: Vec<Vec<i64>> = xs.iter().map(|x| m.matvec_q2f(x)).collect();
            for threads in [1usize, 2, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                pool.install(|| {
                    for n in (1..=9).chain([16, 64]) {
                        let batch: Vec<&[Fixed<i32, 10>]> = xs[..n].iter().map(|v| &v[..]).collect();
                        let out = m.matvec_batch_par(&batch);
                        let outq = m.matvec_q2f_batch_par(&batch);
                        let mut flat = vec![Fixed::from_raw(0); n * rows];
                        m.matvec_batch_par_into(&batch, &mut flat);
                        for k in 0..n {
                            assert_eq!(out[k], singles[k], "{rows}x{cols} {rounding:?} threads {threads} batch {n} vector {k}");
                            assert_eq!(outq[k], singles_q[k], "{rows}x{cols} {rounding:?} q2f threads {threads} batch {n} vector {k}");
                            assert_eq!(&flat[k * rows..(k + 1) * rows], &singles[k][..]);
                        }
                    }
                });
            }
            if rounding == Rounding::Nearest {
                let batch: Vec<&[Fixed<i32, 10>]> = xs[..8].iter().map(|v| &v[..]).collect();
                let out = m.matvec_batch_par(&batch);
                let outq = m.matvec_q2f_batch_par(&batch);
                for k in 0..8 {
                    let raw: Vec<i32> = xs[k].iter().map(|v| v.raw()).collect();
                    for r in (0..rows).step_by(37) {
                        assert_eq!(out[k][r].raw() as i128, nearest_ref(&m, r, &raw, 0), "nearest {rows}x{cols} k {k} r {r}");
                        assert_eq!(outq[k][r] as i128, nearest_ref(&m, r, &raw, 10), "nearest q2f {rows}x{cols} k {k} r {r}");
                    }
                }
            }
        }
    }
}

#[test]
fn row_subsets_equal_the_rows_of_matvec() {
    let mut seed = 0x5B5E_7001u64;
    for &(rows, cols) in &[(2000usize, 2304usize), (5, 13)] {
        let base = matrix(&mut seed, rows, cols);
        for rounding in [Rounding::Default, Rounding::Nearest] {
            let m = base.clone().with_rounding(rounding);
            let x = acts::<10>(&mut seed, cols, 20);
            let full = m.matvec(&x);
            let fullq = m.matvec_q2f(&x);
            let pick: Vec<usize> = (0..24).map(|i| if i % 7 == 3 { rows - 1 } else { (lcg(&mut seed) as usize) % rows }).chain([0, 0]).collect();
            let sub = m.matvec_rows(&pick, &x);
            let subq = m.matvec_rows_q2f(&pick, &x);
            for (i, &r) in pick.iter().enumerate() {
                assert_eq!(sub[i], full[r], "{rounding:?} row {r}");
                assert_eq!(subq[i], fullq[r], "{rounding:?} q2f row {r}");
            }
            assert!(m.matvec_rows(&[], &x).is_empty());
        }
        let x = acts::<10>(&mut seed, cols, 20);
        assert!(std::panic::catch_unwind(|| base.matvec_rows(&[rows], &x)).is_err(), "row out of range");
    }
}

#[test]
fn wider_words_keep_the_generic_path() {
    // i64 activations: the batch forms still equal matvec (generic path)
    let mut seed = 0x64_64u64;
    let m = matrix(&mut seed, 64, 200);
    let xs: Vec<Vec<Fixed<i64, 32>>> = (0..5)
        .map(|_| (0..200).map(|_| Fixed::from_raw((lcg(&mut seed) as i64 % (1i64 << 40)) - (1i64 << 39))).collect())
        .collect();
    let batch: Vec<&[Fixed<i64, 32>]> = xs.iter().map(|v| &v[..]).collect();
    let out = m.matvec_batch_par(&batch);
    for k in 0..5 {
        assert_eq!(out[k], m.matvec(&xs[k]));
    }
}

/// Worst case for the batched kernel's 32-bit lanes: every weight at
/// +-29524 and every activation limb at its largest magnitude with the
/// weight's sign, so each lane grows by the bound every step. Batched sums
/// against an exact i128 reference at every limb count.
#[test]
fn adversarial_lanes_reach_the_bound() {
    let (rows, cols) = (6usize, 2304usize);
    let mut seed = 0xAD7E_0001u64;
    let data: Vec<i16> = (0..rows * cols).map(|i| if (i / cols) % 2 == 0 { 29524 } else if lcg(&mut seed) & 1 == 0 { 29524 } else { -29524 }).collect();
    let m = RowScaledTQ19::from_parts(rows, cols, data.clone(), vec![1u64 << 18; rows]).with_rounding(Rounding::Nearest);
    // magnitudes: one limb (2047, 2048), two (2^14, 2^21 - 1), three (2^21, 2^25 - 1, 2^31 - 1, -2^31)
    let mut checked = 0;
    for &mag in &[2047i64, 2048, 1 << 14, (1 << 21) - 1, 1 << 21, (1 << 25) - 1, (1 << 31) - 1] {
        let xs: Vec<Vec<Fixed<i32, 10>>> = (0..8)
            .map(|k| {
                (0..cols)
                    .map(|c| {
                        // row 0's signs (all +) for even vectors, row 1's for odd ones; -2^31 once
                        let s = if k % 2 == 0 { 1 } else { data[cols + c].signum() as i64 };
                        let v = if mag == (1 << 31) - 1 && k == 3 && c == 0 { i32::MIN as i64 } else { s * mag };
                        Fixed::from_raw(v.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                    })
                    .collect()
            })
            .collect();
        for n in 1..=8 {
            let batch: Vec<&[Fixed<i32, 10>]> = xs[..n].iter().map(|v| &v[..]).collect();
            let Ok(out) = std::panic::catch_unwind(|| m.matvec_batch_par(&batch)) else {
                // an output beyond i32: the single form must panic too
                assert!(xs[..n].iter().any(|x| std::panic::catch_unwind(|| m.matvec(x)).is_err()));
                continue;
            };
            for k in 0..n {
                let raw: Vec<i32> = xs[k].iter().map(|v| v.raw()).collect();
                for r in 0..rows {
                    assert_eq!(out[k][r].raw() as i128, nearest_ref(&m, r, &raw, 0), "mag {mag} batch {n} vector {k} row {r}");
                    checked += 1;
                }
            }
        }
    }
    // every magnitude in range: 7 x (1 + .. + 8) vectors x 6 rows
    assert_eq!(checked, 7 * 36 * 6);
}
