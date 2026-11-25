//! Gate for `I16Matrix::matvec_i64(_par)`: exact integer row sums.
//!
//! 1. References (scripts/generate_i16_gemv_refs.py: integers, confirmed by
//!    mpmath) in every lane regime: one flush per row, a flush every step,
//!    -32768 in both operands (the scalar path), column tails, zero vectors.
//! 2. A random differential at a draft-head shape (20000 x 256) against an
//!    in-test i64 loop, 1/2/3/8 threads, including the extremes.

#[path = "data/i16_gemv_refs.rs"]
mod refs;

use g_math_inference::I16Matrix;

#[test]
fn equals_the_references() {
    let mut n = 0;
    for &(rows, cols, f, t, sums) in refs::CASES {
        let m = I16Matrix::from_parts(rows, cols, f.to_vec());
        assert_eq!(m.matvec_i64(t), sums, "{rows}x{cols}");
        assert_eq!(m.matvec_i64_par(t), sums, "{rows}x{cols} par");
        n += rows;
    }
    assert_eq!(n, 30);
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 17
}

#[test]
fn random_draft_head_shape_every_thread_count() {
    let mut seed = 0x1600_0256u64;
    let (rows, cols) = (20000usize, 256usize);
    for &(fl, tl) in &[(1i64 << 9, 1i64 << 15), (32767, 32767), (32768, 32768)] {
        let rnd = |s: &mut u64, lim: i64| -> i16 { ((lcg(s) as i64 % (2 * lim)) - lim).clamp(-32768, 32767) as i16 };
        let f: Vec<i16> = (0..rows * cols).map(|_| rnd(&mut seed, fl)).collect();
        let t: Vec<i16> = (0..cols).map(|_| rnd(&mut seed, tl)).collect();
        let want: Vec<i64> = (0..rows).map(|r| (0..cols).map(|j| f[r * cols + j] as i64 * t[j] as i64).sum()).collect();
        let m = I16Matrix::from_parts(rows, cols, f);
        assert_eq!(m.matvec_i64(&t), want);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            assert_eq!(pool.install(|| m.matvec_i64_par(&t)), want, "limits {fl}/{tl}, {threads} threads");
        }
    }
}
