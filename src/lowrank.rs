//! A dense matrix of 16-bit integers with an exact integer matvec: low-rank
//! folds of a large matrix (`W_a . W_b` at a common power-of-two scale) used
//! as a cheap ranking proxy.
//!
//! `y[r] = sum_j F[r][j] * t[j]`, exact in i64 for every i16 input. The AVX2
//! kernel multiplies 16 pairs per instruction into 32-bit lanes and moves the
//! lanes to 64 bits before they can overflow: after `K` steps, `K` derived
//! from the largest `|F|` (at construction) and the largest `|t|` (per call)
//! so that `K * 2 * max|F| * max|t| < 2^31`. Only when both hold -32768 can a
//! single pair overflow 32 bits; that call takes the exact scalar path.

use crate::tq5::{avx2_available, MIN_TASK_BYTES, SERIAL_BELOW_BYTES};
use rayon::prelude::*;

/// A row-major matrix of i16 entries (see the module documentation).
#[derive(Clone, Debug)]
pub struct I16Matrix {
    rows: usize,
    cols: usize,
    data: Vec<i16>,
    /// The largest `|F|` (32768 for -32768).
    max_abs: u32,
}

impl I16Matrix {
    /// A matrix from its entries, row-major.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn from_parts(rows: usize, cols: usize, data: Vec<i16>) -> Self {
        assert_eq!(data.len(), rows * cols, "I16Matrix: data length mismatch");
        let max_abs = data.iter().map(|v| v.unsigned_abs() as u32).max().unwrap_or(0);
        Self { rows, cols, data, max_abs }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn data(&self) -> &[i16] {
        &self.data
    }
    /// Bytes held (2 per entry).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2
    }

    /// `y[r] = sum_j F[r][j] * t[j]`, exact.
    ///
    /// # Panics
    /// Panics if `t.len() != cols`.
    pub fn matvec_i64(&self, t: &[i16]) -> Vec<i64> {
        self.run(t, false)
    }

    /// [`Self::matvec_i64`] with the rows in parallel: the same results.
    pub fn matvec_i64_par(&self, t: &[i16]) -> Vec<i64> {
        self.run(t, true)
    }

    /// Pair steps (16 columns each) a 32-bit lane takes without overflow, 0
    /// when one pair can overflow.
    fn steps(&self, t: &[i16]) -> usize {
        let mt = t.iter().map(|v| v.unsigned_abs() as u64).max().unwrap_or(0);
        let pair = 2 * self.max_abs as u64 * mt;
        if pair == 0 {
            return usize::MAX;
        }
        (((1u64 << 31) - 1) / pair) as usize
    }

    fn run(&self, t: &[i16], par: bool) -> Vec<i64> {
        assert_eq!(t.len(), self.cols, "I16Matrix::matvec_i64: length mismatch");
        let cols = self.cols;
        let k = self.steps(t);
        let simd = cfg!(target_arch = "x86_64") && avx2_available() && k > 0;
        let row = |r: usize| -> i64 {
            let f = &self.data[r * cols..(r + 1) * cols];
            #[cfg(target_arch = "x86_64")]
            if simd {
                // SAFETY: AVX2 checked; f and t have cols elements; k steps never overflow a lane
                return unsafe { dot_avx2(f, t, k) };
            }
            let _ = simd;
            f.iter().zip(t).map(|(&a, &b)| a as i64 * b as i64).sum()
        };
        if par && self.rows * cols * 2 >= SERIAL_BELOW_BYTES {
            (0..self.rows).into_par_iter().with_min_len((MIN_TASK_BYTES / (2 * cols).max(1)).max(1)).map(row).collect()
        } else {
            (0..self.rows).map(row).collect()
        }
    }
}

/// `sum f[j] * t[j]` with 16-column pair steps, the 32-bit lanes moved to 64
/// bits every `k` steps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(f: &[i16], t: &[i16], k: usize) -> i64 {
    use std::arch::x86_64::*;
    let n = f.len();
    let steps = n / 16;
    let mut wide = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_si256();
    let mut since = 0usize;
    for s in 0..steps {
        let a = _mm256_loadu_si256(f.as_ptr().add(16 * s) as *const __m256i);
        let b = _mm256_loadu_si256(t.as_ptr().add(16 * s) as *const __m256i);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
        since += 1;
        if since == k {
            wide = _mm256_add_epi64(wide, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)));
            wide = _mm256_add_epi64(wide, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)));
            acc = _mm256_setzero_si256();
            since = 0;
        }
    }
    wide = _mm256_add_epi64(wide, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)));
    wide = _mm256_add_epi64(wide, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)));
    let lo = _mm256_castsi256_si128(wide);
    let hi = _mm256_extracti128_si256(wide, 1);
    let s2 = _mm_add_epi64(lo, hi);
    let mut sum = _mm_cvtsi128_si64(_mm_add_epi64(s2, _mm_srli_si128(s2, 8)));
    for j in steps * 16..n {
        sum += f[j] as i64 * t[j] as i64;
    }
    sum
}
