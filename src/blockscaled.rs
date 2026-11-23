//! Block-scaled weights as stored in modern checkpoints, computed on directly
//! and exactly in integers: NVFP4 and MXFP8.
//!
//! - [`NvFp4Matrix`]: four-bit E2M1 elements (two per byte, the low nibble is
//!   the even column), one E4M3 scale per 16 consecutive columns of a row,
//!   one factor `amax / den` per matrix (two binary32 patterns).
//!   `value(r, c) = E2M1(code) * E4M3(scale) * amax / den`.
//! - [`MxFp8Matrix`]: E4M3 elements, one UE8M0 scale (`2^(s - 127)`) per 32
//!   consecutive columns of a row. `value(r, c) = E4M3(elem) * 2^(s - 127)`.
//!
//! No float type and no float rounding on any path. Every pattern is read as
//! an integer and a power of two ([`crate::bits`]); `amax / den` is kept as an
//! exact fraction; each row sum is an exact integer; each output is rounded
//! once, toward minus infinity, as [`crate::RowScaledTQ5`] rounds:
//!
//! `y[r] = floor(sum_c value(r, c) * x[c])` at the activations' format, and
//! the wide form at twice its fractional bits. A result that does not fit
//! panics; nothing wraps.
//!
//! Every path (scalar, AVX2, the batched register tile, any thread count)
//! forms the same integer row sum before the one rounding, so all forms agree
//! bit for bit. The AVX2 paths need the block multipliers to fit 16 bits,
//! which holds for every checkpoint seen (block scales within a few powers of
//! two of each other); any other matrix takes the exact scalar path.

use crate::bits::{decompose_e4m3, decompose_f32, ue8m0_exponent, E2M1_X2};
use crate::exact::{FloorDiv, Rounding};
use crate::tq5::{avx2_available, SplitActivation, MIN_TASK_BYTES, SERIAL_BELOW_BYTES};
use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::{Fixed, FixedNum, Storage, WideInt, I512};
use rayon::prelude::*;

/// Activation-half vectors up to which a batch decodes each row once in
/// registers (no scratch): the speculative-checking sizes.
const DIRECT_MAX_VECS: usize = 8;

/// Rows per register tile and activations per full tile of the batched NVFP4 kernel.
const TILE_R: usize = 3;
const TILE_T: usize = 4;

// ---------------------------------------------------------------------------
// exact finishing: floor(s * num * 2^shift / den), one rounding
// ---------------------------------------------------------------------------

/// The matrix's factor applied to one exact row sum `s`:
/// `floor(s * num * 2^(shift + extra) / den)`. `None` when the result does not
/// fit 128 bits (then it does not fit any output word either).
#[derive(Clone, Copy, Debug)]
struct Finish {
    num: i128,
    shift: i32,
    div: FloorDiv,
}

impl Finish {
    #[inline(always)]
    fn apply(&self, s: i128, extra: i32) -> Option<i128> {
        let n = s.checked_mul(self.num)?;
        let k = self.shift + extra;
        if k >= 0 {
            if n == 0 {
                return Some(0);
            }
            // n << k must keep a sign bit: |n| < 2^(127 - k)
            if k >= 127 || (n.unsigned_abs().leading_zeros() as i32) <= k {
                return None;
            }
            Some(self.div.floor(n << k))
        } else {
            // floor(floor(n / den) / 2^-k) = floor(n / (den * 2^-k)), and >> floors
            Some(self.div.floor(n) >> ((-k).min(127) as u32))
        }
    }
}

/// `m * 2^e` with the trailing zero bits of `m` moved into `e`.
fn strip(m: u64, e: i32) -> (u64, i32) {
    if m == 0 {
        (0, 0)
    } else {
        let t = m.trailing_zeros();
        (m >> t, e + t as i32)
    }
}

/// An output under a rounding mode from a floor-rounding finish `f(extra)`:
/// `Nearest` is `floor(2x)` one bit finer, then `(+1) >> 1`
/// (`floor(x + 1/2)`, ties toward +infinity).
#[inline(always)]
fn rounded(r: Rounding, extra: i32, f: impl Fn(i32) -> Option<i128>) -> Option<i128> {
    match r {
        Rounding::Default => f(extra),
        Rounding::Nearest => f(extra + 1).and_then(|v| v.checked_add(1)).map(|v| v >> 1),
    }
}

/// An exact row sum: native when it fits, 512 bits otherwise (MXFP8 with
/// block scales far apart).
#[derive(Clone, Copy)]
enum RowSum {
    N(i128),
    W(I512),
}

#[inline]
fn narrow_i32(v: Option<i128>, what: &str) -> i32 {
    match v {
        Some(v) if v >= i32::MIN as i128 && v <= i32::MAX as i128 => v as i32,
        _ => panic!("{what}: output exceeds storage range"),
    }
}

#[inline]
fn narrow_i64(v: Option<i128>, what: &str) -> i64 {
    match v {
        Some(v) if v >= i64::MIN as i128 && v <= i64::MAX as i128 => v as i64,
        _ => panic!("{what}: q2f output exceeds compute range"),
    }
}

// ---------------------------------------------------------------------------
// NVFP4
// ---------------------------------------------------------------------------

/// Whether an activation's high halves are few enough to add as a scalar
/// correction (at most one per `div` columns) rather than multiply densely.
/// A correction costs a weight decode; a dense high half one more
/// multiply-add per 16 columns, so the crossover is near one per 128 columns
/// (NVFP4) or 64 (MXFP8, whose dense form costs more per column).
#[inline]
fn sparse(xs: &SplitActivation, cols: usize, div: usize) -> bool {
    xs.hi_nz.len() * div <= cols
}

/// A row-major NVFP4 matrix: E2M1 codes, an E4M3 scale per 16 columns, one
/// `amax / den` factor, at the checkpoint's 4.5 bits per weight.
///
/// Exact form: each scale byte maps to a signed integer multiplier
/// `m << (e - e_min)` (`e_min` the smallest exponent of a nonzero scale in the
/// matrix), so weight `(r, c) = E2M1_X2[code] * mult[scale] * 2^(e_min - 1) *
/// amax / den`, and a row sum is `sum_blocks mult * sum_16 (E2M1_X2 * x)`.
///
/// Storage: the codes are regrouped once at construction (same bytes): the
/// 16 bytes of block pair `(2p, 2p + 1)` hold column `k` of block `2p` in the
/// low nibble of byte `k` and of block `2p + 1` in its high nibble, so one
/// load and a mask give sixteen codes in column order.
/// [`NvFp4Matrix::to_codes`] returns the checkpoint layout.
#[derive(Clone)]
pub struct NvFp4Matrix {
    rows: usize,
    cols: usize,
    /// blocks per row (`cols / 16`) and block pairs per row (rounded up)
    nb: usize,
    pairs: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    amax_bits: u32,
    den_bits: u32,
    mult: Box<[i32; 256]>,
    /// `mult` in both 16-bit halves of a word, when the SIMD bounds hold.
    mult32: Option<Box<[u32; 256]>>,
    /// Block pairs a 32-bit SIMD lane accumulates before it is widened.
    pairs_per_flush: usize,
    finish: Finish,
    rounding: Rounding,
}

impl NvFp4Matrix {
    /// From the checkpoint's parts: `codes` (`rows * cols / 2` bytes, low
    /// nibble = even column), `scales` (`rows * cols / 16` E4M3 bytes), and
    /// the binary32 patterns of `amax` and the denominator.
    ///
    /// `Err(InvalidInput)` for a NaN scale or a non-finite factor,
    /// `Err(DivisionByZero)` for a zero denominator.
    ///
    /// # Panics
    /// Panics if `cols` is not a multiple of 16, is `2^24` or more, or a
    /// length does not match.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
        amax_bits: u32,
        den_bits: u32,
    ) -> Result<Self, OverflowDetected> {
        assert!(cols % 16 == 0, "NvFp4Matrix: cols must be a multiple of 16");
        // row sums: each term below 2^57 (block multiplier < 2^18), < 2^20 blocks: below 2^77
        assert!(cols < 1 << 24, "NvFp4Matrix: 2^24 columns or more");
        assert_eq!(codes.len(), rows * cols / 2, "NvFp4Matrix: codes length mismatch");
        assert_eq!(scales.len(), rows * cols / 16, "NvFp4Matrix: scales length mismatch");

        let mut present = [false; 256];
        for &b in &scales {
            present[b as usize] = true;
        }
        let mut emin = i32::MAX;
        for b in 0..256 {
            if present[b] {
                let (_, m, e) = decompose_e4m3(b as u8)?;
                if m != 0 {
                    emin = emin.min(e);
                }
            }
        }
        if emin == i32::MAX {
            emin = -9; // every scale is zero: any exponent will do
        }
        let mut mult = Box::new([0i32; 256]);
        for b in 0..256 {
            if present[b] {
                let (neg, m, e) = decompose_e4m3(b as u8)?;
                if m != 0 {
                    // e - emin <= 5 - (-9) = 14, so |mult| <= 15 * 2^14
                    let v = (m as i32) << (e - emin);
                    mult[b] = if neg { -v } else { v };
                }
            }
        }
        // SIMD: weights E2M1_X2 * mult in i16, and a lane gains at most
        // 2 * wmax * 2^15 per block, two blocks per pair, before widening
        let wmax = 12 * mult.iter().map(|v| v.unsigned_abs() as i64).max().unwrap_or(0);
        let pairs_per_flush = if wmax == 0 { 1 << 20 } else { ((i32::MAX as i64) / (4 * wmax * 32768)) as usize };
        let mult32 = (pairs_per_flush >= 1).then(|| {
            Box::new(std::array::from_fn(|b| {
                let h = mult[b] as i16 as u16 as u32;
                h | (h << 16)
            }))
        });

        let (na, ma, ea) = decompose_f32(amax_bits)?;
        let (nd, md, ed) = decompose_f32(den_bits)?;
        if md == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        let (ma, ea) = strip(ma, ea);
        let (md, ed) = strip(md, ed);
        let num = if na != nd { -(ma as i128) } else { ma as i128 };
        // E2M1 = code_x2 / 2 and E4M3 = |mult| * 2^emin, so value = code_x2 * mult * 2^(emin - 1) * amax / den
        let finish = Finish { num, shift: emin - 1 + ea - ed, div: FloorDiv::new(md) };

        let nb = cols / 16;
        let pairs = nb.div_ceil(2);
        let mut packed = vec![0u8; rows * pairs * 16];
        for r in 0..rows {
            for c in 0..cols {
                let byte = codes[(r * cols + c) / 2];
                let code = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let b = c / 16;
                packed[r * pairs * 16 + (b / 2) * 16 + c % 16] |= code << (4 * (b % 2));
            }
        }
        Ok(Self { rows, cols, nb, pairs, packed, scales, amax_bits, den_bits, mult, mult32, pairs_per_flush, finish, rounding: Rounding::Default })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// The codes in the checkpoint layout (`rows * cols / 2` bytes, low nibble = even column).
    /// The same matrix rounding its outputs by `rounding` (see [`Rounding`]).
    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }
    /// The rounding mode applied to every output.
    pub fn rounding(&self) -> Rounding {
        self.rounding
    }
    pub fn to_codes(&self) -> Vec<u8> {
        let mut codes = vec![0u8; self.rows * self.cols / 2];
        for r in 0..self.rows {
            for c in 0..self.cols {
                codes[(r * self.cols + c) / 2] |= self.code(r, c) << (4 * (c % 2));
            }
        }
        codes
    }
    /// The E4M3 block scales as given.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    /// The binary32 patterns of `amax` and the denominator as given.
    pub fn factor_bits(&self) -> (u32, u32) {
        (self.amax_bits, self.den_bits)
    }
    /// Bytes held: codes and scales (4.5 bits per weight; 8 more bytes per
    /// row when a row has an odd number of blocks) plus the 8-byte factor.
    pub fn size_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() + 8
    }
    /// Whether the AVX2 kernels apply to this matrix (block weights fit 16
    /// bits). The results are the same either way.
    pub fn simd_eligible(&self) -> bool {
        self.mult32.is_some()
    }

    /// The exact weights of row `r`: `value(r, c) = nums[c] * 2^exp / den`,
    /// returned as `(nums, exp, den)`. For gates and debugging, not the hot path.
    pub fn dequant_row_exact(&self, r: usize) -> (Vec<i64>, i32, u64) {
        let nums = (0..self.cols).map(|c| self.weight(r, c) as i64 * self.finish.num as i64).collect();
        (nums, self.finish.shift, self.finish.div.d)
    }

    #[inline(always)]
    fn code(&self, r: usize, c: usize) -> u8 {
        let b = c / 16;
        (self.packed[r * self.pairs * 16 + (b / 2) * 16 + c % 16] >> (4 * (b % 2))) & 0x0F
    }

    /// `E2M1_X2[code] * mult[scale]` at `(r, c)`.
    #[inline(always)]
    fn weight(&self, r: usize, c: usize) -> i32 {
        E2M1_X2[self.code(r, c) as usize] as i32 * self.mult[self.scales[r * self.nb + c / 16] as usize]
    }

    #[inline(always)]
    fn row_ptrs(&self, r: usize) -> (*const u8, *const u8) {
        debug_assert!(r < self.rows);
        // SAFETY: r < rows; both offsets are within the vectors
        unsafe { (self.packed.as_ptr().add(r * self.pairs * 16), self.scales.as_ptr().add(r * self.nb)) }
    }

    /// The exact row sum `sum_c E2M1_X2 * mult * x` in plain integer arithmetic.
    fn sum_scalar(&self, r: usize, x: &[i32]) -> i128 {
        let packed = &self.packed[r * self.pairs * 16..(r + 1) * self.pairs * 16];
        let scales = &self.scales[r * self.nb..(r + 1) * self.nb];
        let mut s = 0i128;
        for (b, &sb) in scales.iter().enumerate() {
            let m = self.mult[sb as usize];
            if m == 0 {
                continue;
            }
            let (pb, sh, xb) = (&packed[(b / 2) * 16..(b / 2) * 16 + 16], 4 * (b % 2), &x[b * 16..b * 16 + 16]);
            // |code_x2| <= 12, |x| <= 2^31: a block sum is below 2^39
            let mut acc = 0i64;
            for k in 0..16 {
                acc += E2M1_X2[((pb[k] >> sh) & 0x0F) as usize] as i64 * xb[k] as i64;
            }
            s += acc as i128 * m as i128;
        }
        s
    }
}

/// The eight 32-bit lanes summed as i64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32(v: std::arch::x86_64::__m256i) -> i64 {
    use std::arch::x86_64::*;
    let a = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    let b = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    hsum_i64(_mm256_add_epi64(a, b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i64(v: std::arch::x86_64::__m256i) -> i64 {
    use std::arch::x86_64::*;
    let mut l = [0i64; 4];
    _mm256_storeu_si256(l.as_mut_ptr() as *mut __m256i, v);
    l[0] + l[1] + l[2] + l[3]
}

/// Sixteen weights of one NVFP4 block in column order as i16: the codes of
/// the pair at `pair` (low nibbles when `HI` is false), times the block's
/// multiplier (`mult32` holds it in both halves of a word: one broadcast load).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn nvfp4_weights<const HI: bool>(pair: *const u8, mult32: *const u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    let mask = _mm_set1_epi8(0x0F);
    // SAFETY (caller): 16 readable bytes at `pair`, a readable word at `mult32`
    let bytes = _mm_loadu_si128(pair as *const __m128i);
    let nib = if HI { _mm_and_si128(_mm_srli_epi16(bytes, 4), mask) } else { _mm_and_si128(bytes, mask) };
    let c = _mm_shuffle_epi8(table, nib);
    _mm256_mullo_epi16(_mm256_cvtepi8_epi16(c), _mm256_set1_epi32(std::ptr::read_unaligned(mult32) as i32))
}

/// Register tile: `sum(weight * v)` for `R` NVFP4 rows x `T` vectors of 16-bit
/// activation halves. Each decoded weight vector feeds `T` vectors; `R * T`
/// accumulators + `R` weights + one activation fit the 16 vector registers
/// at 3 x 4. A lane takes two multiply-adds per block pair and is widened
/// every `pairs_per_flush` pairs, so the sums are the same integers in any
/// tile shape.
///
/// # Safety
/// AVX2; every row pointer addresses `nb` blocks of the packed layout and
/// `nb` scale bytes, every vector `16 * nb` halves.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nvfp4_tile<const R: usize, const T: usize>(
    packed: [*const u8; R],
    scales: [*const u8; R],
    nb: usize,
    mult32: *const u32,
    vecs: [*const i16; T],
    pairs_per_flush: usize,
) -> [[i64; T]; R] {
    use std::arch::x86_64::*;
    let mut total = [[0i64; T]; R];
    let full = nb / 2;
    let mut p = 0usize;
    while p < full {
        let end = (p + pairs_per_flush).min(full);
        let mut acc = [[_mm256_setzero_si256(); T]; R];
        while p < end {
            let b = 2 * p;
            let mut w = [_mm256_setzero_si256(); R];
            for r in 0..R {
                w[r] = nvfp4_weights::<false>(packed[r].add(p * 16), mult32.add(*scales[r].add(b) as usize));
            }
            for t in 0..T {
                let xv = _mm256_loadu_si256(vecs[t].add(b * 16) as *const __m256i);
                for r in 0..R {
                    acc[r][t] = _mm256_add_epi32(acc[r][t], _mm256_madd_epi16(w[r], xv));
                }
            }
            for r in 0..R {
                w[r] = nvfp4_weights::<true>(packed[r].add(p * 16), mult32.add(*scales[r].add(b + 1) as usize));
            }
            for t in 0..T {
                let xv = _mm256_loadu_si256(vecs[t].add(b * 16 + 16) as *const __m256i);
                for r in 0..R {
                    acc[r][t] = _mm256_add_epi32(acc[r][t], _mm256_madd_epi16(w[r], xv));
                }
            }
            p += 1;
        }
        for r in 0..R {
            for t in 0..T {
                total[r][t] += hsum_i32(acc[r][t]);
            }
        }
    }
    if nb % 2 == 1 {
        // the last block of an odd row: low nibbles of the last pair
        let b = nb - 1;
        for r in 0..R {
            let w = nvfp4_weights::<false>(packed[r].add(full * 16), mult32.add(*scales[r].add(b) as usize));
            for t in 0..T {
                let xv = _mm256_loadu_si256(vecs[t].add(b * 16) as *const __m256i);
                total[r][t] += hsum_i32(_mm256_madd_epi16(w, xv));
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// MXFP8
// ---------------------------------------------------------------------------

/// E4M3 element as an exact integer: `value = ELEM[bits] * 2^-9`
/// (`|.| <= 15 * 2^14`); NaN patterns map to 0 and are refused at construction.
const fn elem_table() -> [i32; 256] {
    let mut t = [0i32; 256];
    let mut b = 0;
    while b < 256 {
        let e = ((b >> 3) & 0x0F) as i32;
        let m = (b & 0x07) as i32;
        let v = if e == 0 { m } else { (8 + m) << (e - 1) };
        t[b] = if b & 0x7F == 0x7F { 0 } else if b & 0x80 != 0 { -v } else { v };
        b += 1;
    }
    t
}
static ELEM: [i32; 256] = elem_table();

/// The largest scale spread (in powers of two) for the i128 scalar path:
/// block sums below 2^54, shifted, over fewer than 2^19 blocks, stay below 2^126.
const MX_I128_SPREAD: i32 = 52;

/// A row-major MXFP8 matrix: E4M3 elements and a UE8M0 scale per 32 columns,
/// as stored (8.25 bits per weight).
///
/// Exact form: element `w = ELEM[bits]` (an integer), block shift
/// `d = s - s_min` (`s_min` the smallest scale of a block with a nonzero
/// element), weight `= w * 2^d * 2^(s_min - 136)`.
#[derive(Clone)]
pub struct MxFp8Matrix {
    rows: usize,
    cols: usize,
    elems: Vec<u8>,
    scales: Vec<u8>,
    smin: i32,
    spread: i32,
    /// the SIMD form, for block scales within 2^7 of each other
    simd: Option<MxSimd>,
    /// the batched form, for block scales within 2^12 of each other: each
    /// weight `ELEM * 2^d = wh * 2^B + wl` in two 16-bit limbs, as
    /// `(B, tile steps per 32-bit lane widening)`
    limbs: Option<(u32, usize)>,
    rounding: Rounding,
}

impl MxFp8Matrix {
    /// From the checkpoint's parts: `elems` (`rows * cols` E4M3 bytes) and
    /// `scales` (`rows * cols / 32` UE8M0 bytes, one per row and 32 columns).
    ///
    /// `Err(InvalidInput)` for a NaN element or scale.
    ///
    /// # Panics
    /// Panics if `cols` is not a multiple of 32, is `2^24` or more, or a
    /// length does not match.
    pub fn from_parts(rows: usize, cols: usize, elems: Vec<u8>, scales: Vec<u8>) -> Result<Self, OverflowDetected> {
        assert!(cols % 32 == 0, "MxFp8Matrix: cols must be a multiple of 32");
        assert!(cols < 1 << 24, "MxFp8Matrix: 2^24 columns or more");
        assert_eq!(elems.len(), rows * cols, "MxFp8Matrix: elems length mismatch");
        assert_eq!(scales.len(), rows * cols / 32, "MxFp8Matrix: scales length mismatch");
        if elems.iter().any(|&b| b & 0x7F == 0x7F) {
            return Err(OverflowDetected::InvalidInput);
        }
        let (mut smin, mut smax) = (i32::MAX, i32::MIN);
        for (k, &s) in scales.iter().enumerate() {
            let e = ue8m0_exponent(s)? + 127;
            if elems[k * 32..k * 32 + 32].iter().any(|&b| b & 0x7F != 0) {
                smin = smin.min(e);
                smax = smax.max(e);
            }
        }
        if smin == i32::MAX {
            smin = 0;
            smax = 0;
        }
        let spread = smax - smin;
        Ok(Self { rows, cols, elems, scales, smin, spread, simd: MxSimd::for_spread(spread), limbs: mx_limbs(spread), rounding: Rounding::Default })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// The same matrix rounding its outputs by `rounding` (see [`Rounding`]).
    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }
    /// The rounding mode applied to every output.
    pub fn rounding(&self) -> Rounding {
        self.rounding
    }
    /// The E4M3 elements as given.
    pub fn elems(&self) -> &[u8] {
        &self.elems
    }
    /// The UE8M0 block scales as given.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    /// Bytes of elements and scales (8.25 bits per weight).
    pub fn size_bytes(&self) -> usize {
        self.elems.len() + self.scales.len()
    }
    /// Whether the AVX2 kernels apply (block scales of nonzero blocks within
    /// 2^7 of each other). The results are the same either way.
    pub fn simd_eligible(&self) -> bool {
        self.simd.is_some()
    }

    /// The exact weights of row `r` as `(w, e)` pairs, `value = w * 2^e`.
    /// For gates and debugging, not the hot path.
    pub fn dequant_row_exact(&self, r: usize) -> Vec<(i32, i32)> {
        let (elems, scales) = self.row(r);
        (0..self.cols).map(|c| (ELEM[elems[c] as usize], scales[c / 32] as i32 - 136)).collect()
    }

    #[inline(always)]
    fn row(&self, r: usize) -> (&[u8], &[u8]) {
        let c32 = self.cols / 32;
        (&self.elems[r * self.cols..(r + 1) * self.cols], &self.scales[r * c32..(r + 1) * c32])
    }

    /// Block shift `s - s_min`, or `None` for a block whose elements are all
    /// zero and whose scale lies outside `[s_min, s_min + spread]` (it adds 0).
    #[inline(always)]
    fn shift(&self, s: u8) -> Option<u32> {
        let d = s as i32 - self.smin;
        (0..=self.spread).contains(&d).then_some(d as u32)
    }

    /// `sum_c ELEM * 2^d * x` exactly, in i128 (spread <= 52) or I512.
    fn sum_scalar_i128(&self, r: usize, x: &[i32]) -> i128 {
        let (elems, scales) = self.row(r);
        let mut s = 0i128;
        for (b, &sb) in scales.iter().enumerate() {
            let Some(d) = self.shift(sb) else { continue };
            // |ELEM| < 2^18, |x| <= 2^31: a block sum is below 2^54
            let acc: i64 = elems[b * 32..b * 32 + 32]
                .iter()
                .zip(&x[b * 32..b * 32 + 32])
                .map(|(&e, &v)| ELEM[e as usize] as i64 * v as i64)
                .sum();
            s += (acc as i128) << d;
        }
        s
    }

    fn sum_scalar_i512(&self, r: usize, x: &[i32]) -> I512 {
        let (elems, scales) = self.row(r);
        let mut s = I512::from(0i128);
        for (b, &sb) in scales.iter().enumerate() {
            let Some(d) = self.shift(sb) else { continue };
            let acc: i64 = elems[b * 32..b * 32 + 32]
                .iter()
                .zip(&x[b * 32..b * 32 + 32])
                .map(|(&e, &v)| ELEM[e as usize] as i64 * v as i64)
                .sum();
            // below 2^54 << 254, over < 2^19 blocks: below 2^327
            s = s + (I512::from(acc as i128) << d as usize);
        }
        s
    }

    /// `floor(s * 2^(s_min - 136 + extra))`.
    #[inline(always)]
    fn finish(&self, s: i128, extra: i32) -> Option<i128> {
        let k = self.smin - 136 + extra;
        if k >= 0 {
            if s == 0 {
                return Some(0);
            }
            if k >= 127 || (s.unsigned_abs().leading_zeros() as i32) <= k {
                return None;
            }
            Some(s << k)
        } else {
            Some(s >> ((-k).min(127) as u32))
        }
    }

    fn finish_i512(&self, s: I512, extra: i32) -> Option<i128> {
        let k = self.smin - 136 + extra;
        // |s| < 2^327 and k <= 254 - 136 + 64: the left shift stays below 2^509
        let v = if k >= 0 { s << k as usize } else { s >> (-k) as usize };
        v.fits_in_i128().then(|| v.as_i128())
    }
}

/// The SIMD form of an MXFP8 matrix. An element's integer value is
/// `s_m * 2^t` (`s_m <= 15`, `t = max(e, 1) - 1 <= 14`); its exponent class
/// is `c = t / k` and within the class `s_m * 2^(t - c k)`. With the block
/// shift `d` folded in, `s_m * 2^(t - c k + d) <= 15 * 2^7` fits 16 bits for
/// every class, so a 16-bit multiply-add per class does the work, and the
/// class sums combine as `sum_c S_c * 2^(c k)` once per flush.
#[derive(Clone, Debug)]
struct MxSimd {
    /// number of classes and their width in exponents
    classes: usize,
    k: u32,
    /// per block shift d: 2^(t - c k + d) by exponent field, as a 16-entry byte table in both 128-bit lanes
    pow: Vec<[u8; 32]>,
    /// class by exponent field, the same layout
    cls: [u8; 32],
}

/// Class counts with SIMD kernels, and the widest block shift each allows.
const MX_CLASSES: [usize; 5] = [3, 4, 5, 8, 15];
/// Steps of 16 elements a 32-bit lane takes before it is widened: each adds
/// at most 2 * 1920 * 2^15 < 2^27, so 16 steps stay below 2^31.
const MX_FLUSH_STEPS: usize = 16;

impl MxSimd {
    /// The fewest classes whose largest table entry `2^(k - 1 + spread)` fits a byte.
    fn for_spread(spread: i32) -> Option<Self> {
        MX_CLASSES.iter().find_map(|&classes| {
            let k = 15u32.div_ceil(classes as u32);
            (spread >= 0 && k as i32 - 1 + spread <= 7).then(|| {
                let mut cls = [0u8; 32];
                let pow = (0..=spread as u32)
                    .map(|d| {
                        let mut p = [0u8; 32];
                        for e in 0..16u32 {
                            let t = e.max(1) - 1;
                            let c = t / k;
                            p[e as usize] = 1u8 << (t - c * k + d);
                            p[e as usize + 16] = p[e as usize];
                            cls[e as usize] = c as u8;
                            cls[e as usize + 16] = c as u8;
                        }
                        p
                    })
                    .collect();
                Self { classes, k, pow, cls }
            })
        })
    }
}

/// Sixteen MXFP8 elements as 16-bit words: `(s_m * pow[e], sign applied; class)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mx_words(src: *const u8, pow: std::arch::x86_64::__m256i, cls: std::arch::x86_64::__m256i) -> (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i) {
    use std::arch::x86_64::*;
    // SAFETY (caller): 16 readable bytes at `src`
    let g = _mm256_cvtepi8_epi16(_mm_loadu_si128(src as *const __m128i));
    let e = _mm256_and_si256(_mm256_srli_epi16(g, 3), _mm256_set1_epi16(15));
    let hid = _mm256_andnot_si256(_mm256_cmpeq_epi16(e, _mm256_setzero_si256()), _mm256_set1_epi16(8));
    let sm = _mm256_or_si256(_mm256_and_si256(g, _mm256_set1_epi16(7)), hid);
    // table index in the low byte; 0x80 in the high byte selects 0
    let idx = _mm256_or_si256(e, _mm256_set1_epi16(0x8000u16 as i16));
    let p = _mm256_shuffle_epi8(pow, idx);
    let c = _mm256_shuffle_epi8(cls, idx);
    // a negative pattern negates; the zero pattern is the value 0 anyway
    (_mm256_sign_epi16(_mm256_mullo_epi16(sm, p), g), c)
}

/// One MXFP8 row against `H` activation halves: the exact sums
/// `sum_c ELEM * 2^d * half` (block shift relative to `s_min`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mxfp8_row<const C: usize, const H: usize>(m: &MxFp8Matrix, sd: &MxSimd, r: usize, halves: [*const i16; H]) -> [i128; H] {
    use std::arch::x86_64::*;
    let (elems, scales) = m.row(r);
    let cls = _mm256_loadu_si256(sd.cls.as_ptr() as *const __m256i);
    let mut total = [0i128; H];
    let mut acc = [[_mm256_setzero_si256(); H]; C];
    let mut steps = 0usize;
    for (b, &sb) in scales.iter().enumerate() {
        let Some(d) = m.shift(sb) else { continue };
        let pow = _mm256_loadu_si256(sd.pow[d as usize].as_ptr() as *const __m256i);
        for half in 0..2 {
            let j = b * 32 + half * 16;
            let (w, c) = mx_words(elems.as_ptr().add(j), pow, cls);
            let mut wc = [_mm256_setzero_si256(); C];
            for (k, wk) in wc.iter_mut().enumerate() {
                *wk = _mm256_and_si256(w, _mm256_cmpeq_epi16(c, _mm256_set1_epi16(k as i16)));
            }
            for h in 0..H {
                let xv = _mm256_loadu_si256(halves[h].add(j) as *const __m256i);
                for k in 0..C {
                    acc[k][h] = _mm256_add_epi32(acc[k][h], _mm256_madd_epi16(wc[k], xv));
                }
            }
        }
        steps += 2;
        if steps >= MX_FLUSH_STEPS {
            for k in 0..C {
                for h in 0..H {
                    total[h] += (hsum_i32(acc[k][h]) as i128) << (sd.k * k as u32);
                    acc[k][h] = _mm256_setzero_si256();
                }
            }
            steps = 0;
        }
    }
    for k in 0..C {
        for h in 0..H {
            total[h] += (hsum_i32(acc[k][h]) as i128) << (sd.k * k as u32);
        }
    }
    total
}

/// The limb split for the batched MXFP8 tile: `|ELEM * 2^d| < 15 * 2^(14 + d)`,
/// so with `B >= 3 + spread` the high limb `w >> B` and the low limb
/// `w & (2^B - 1)` both fit 16 bits. `B` balances the two; the period keeps a
/// 32-bit lane below 2^31 (one multiply-add per step adds at most
/// `2 * max_limb * 2^15`). `None` past a spread of 12.
fn mx_limbs(spread: i32) -> Option<(u32, usize)> {
    if !(0..=12).contains(&spread) {
        return None;
    }
    let b = (9 + (spread as u32).div_ceil(2)).clamp(3 + spread as u32, 15);
    let wl = (1i64 << b) - 1;
    let wh = 15i64 << (14 + spread as u32 - b);
    assert!(wl <= i16::MAX as i64 && wh <= i16::MAX as i64);
    let period = (i32::MAX as i64 / (2 * wl.max(wh) * 32768)) as usize;
    (period >= 1).then_some((b, period))
}

/// MXFP8 row `r` decoded for the batched tile into two rows of `cols`
/// 16-bit limbs: `out[..cols]` the low limbs, `out[cols..]` the high limbs of
/// `ELEM * 2^d` (block shift `d`; zero for out-of-range zero blocks).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mxfp8_decode_limbs(m: &MxFp8Matrix, b_bits: u32, r: usize, out: &mut [i16]) {
    use std::arch::x86_64::*;
    let (elems, scales) = m.row(r);
    let cols = m.cols;
    assert!(out.len() >= 2 * cols);
    let o = out.as_mut_ptr();
    let mask = _mm256_set1_epi32((1i32 << b_bits) - 1);
    let cnt = _mm_cvtsi32_si128(b_bits as i32);
    let one = _mm256_set1_epi32(1);
    for (b, &sb) in scales.iter().enumerate() {
        let Some(d) = m.shift(sb) else {
            for off in [b * 32, b * 32 + 16, cols + b * 32, cols + b * 32 + 16] {
                _mm256_storeu_si256(o.add(off) as *mut __m256i, _mm256_setzero_si256());
            }
            continue;
        };
        let dv = _mm256_set1_epi32(d as i32);
        for q in 0..2 {
            let j = b * 32 + q * 16;
            let mut lo = [_mm256_setzero_si256(); 2];
            let mut hi = [_mm256_setzero_si256(); 2];
            for h in 0..2 {
                // SAFETY: 8 element bytes at j + 8h < cols
                let g = _mm256_cvtepi8_epi32(_mm_loadl_epi64(elems.as_ptr().add(j + 8 * h) as *const __m128i));
                let e = _mm256_and_si256(_mm256_srli_epi32(g, 3), _mm256_set1_epi32(15));
                let sub = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
                let sm = _mm256_or_si256(_mm256_and_si256(g, _mm256_set1_epi32(7)), _mm256_andnot_si256(sub, _mm256_set1_epi32(8)));
                let t = _mm256_sub_epi32(_mm256_max_epu32(e, one), one);
                // ELEM * 2^d = s_m << (t + d), the pattern's sign applied (a zero byte is 0 anyway)
                let w = _mm256_sign_epi32(_mm256_sllv_epi32(sm, _mm256_add_epi32(t, dv)), g);
                lo[h] = _mm256_and_si256(w, mask);
                hi[h] = _mm256_sra_epi32(w, cnt);
            }
            // packs interleaves 64-bit quarters as (a0-3, b0-3, a4-7, b4-7): reorder to column order
            let pl = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo[0], lo[1]), 0b11_01_10_00);
            let ph = _mm256_permute4x64_epi64(_mm256_packs_epi32(hi[0], hi[1]), 0b11_01_10_00);
            _mm256_storeu_si256(o.add(j) as *mut __m256i, pl);
            _mm256_storeu_si256(o.add(cols + j) as *mut __m256i, ph);
        }
    }
}

/// Register tile over decoded 16-bit weights: `sum(w * v)` for `R` weight
/// rows x `T` activation-half vectors of `n` elements (`n` a multiple of 16).
/// `R * T + R + 1 <= 16` vector registers at 3 x 4; each lane takes one
/// multiply-add per step and is widened every `period` steps (chosen by the
/// caller from the weight bound), so the sums are exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i16_tile<const R: usize, const T: usize>(w: [*const i16; R], v: [*const i16; T], n: usize, period: usize) -> [[i64; T]; R] {
    use std::arch::x86_64::*;
    let steps = n / 16;
    let mut total = [[0i64; T]; R];
    let mut s = 0usize;
    while s < steps {
        let end = (s + period).min(steps);
        let mut acc = [[_mm256_setzero_si256(); T]; R];
        while s < end {
            let mut wv = [_mm256_setzero_si256(); R];
            for r in 0..R {
                wv[r] = _mm256_loadu_si256(w[r].add(s * 16) as *const __m256i);
            }
            for t in 0..T {
                let xv = _mm256_loadu_si256(v[t].add(s * 16) as *const __m256i);
                for r in 0..R {
                    acc[r][t] = _mm256_add_epi32(acc[r][t], _mm256_madd_epi16(wv[r], xv));
                }
            }
            s += 1;
        }
        for r in 0..R {
            for t in 0..T {
                total[r][t] += hsum_i32(acc[r][t]);
            }
        }
    }
    total
}

/// [`i16_tile`] for 1..=3 weight rows and 1..=4 vectors given as slices.
///
/// # Safety
/// AVX2; every weight row and vector has at least `n` elements.
#[cfg(target_arch = "x86_64")]
unsafe fn i16_tile_dyn(w: &[*const i16], v: &[&[i16]], n: usize, period: usize) -> [[i64; TILE_T]; TILE_R] {
    let mut out = [[0i64; TILE_T]; TILE_R];
    macro_rules! go {
        ($r:literal, $t:literal) => {{
            let wa: [*const i16; $r] = std::array::from_fn(|k| w[k]);
            let va: [*const i16; $t] = std::array::from_fn(|k| v[k].as_ptr());
            let a = i16_tile::<$r, $t>(wa, va, n, period);
            for r in 0..$r {
                out[r][..$t].copy_from_slice(&a[r]);
            }
        }};
    }
    match (w.len(), v.len()) {
        (3, 4) => go!(3, 4),
        (3, 3) => go!(3, 3),
        (3, 2) => go!(3, 2),
        (3, _) => go!(3, 1),
        (2, 4) => go!(2, 4),
        (2, 3) => go!(2, 3),
        (2, 2) => go!(2, 2),
        (2, _) => go!(2, 1),
        (_, 4) => go!(1, 4),
        (_, 3) => go!(1, 3),
        (_, 2) => go!(1, 2),
        (_, _) => go!(1, 1),
    }
    out
}

/// NVFP4 row `r` decoded for the batched tile: `cols` words `E2M1_X2 * mult`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nvfp4_decode_row(m: &NvFp4Matrix, r: usize, out: &mut [i16]) {
    use std::arch::x86_64::*;
    assert!(out.len() >= m.cols);
    let (packed, scales) = m.row_ptrs(r);
    let m32 = m.mult32.as_deref().unwrap().as_ptr();
    let o = out.as_mut_ptr();
    for p in 0..m.nb / 2 {
        let b = 2 * p;
        _mm256_storeu_si256(o.add(b * 16) as *mut __m256i, nvfp4_weights::<false>(packed.add(p * 16), m32.add(*scales.add(b) as usize)));
        _mm256_storeu_si256(o.add(b * 16 + 16) as *mut __m256i, nvfp4_weights::<true>(packed.add(p * 16), m32.add(*scales.add(b + 1) as usize)));
    }
    if m.nb % 2 == 1 {
        let b = m.nb - 1;
        _mm256_storeu_si256(o.add(b * 16) as *mut __m256i, nvfp4_weights::<false>(packed.add((m.nb / 2) * 16), m32.add(*scales.add(b) as usize)));
    }
}

// ---------------------------------------------------------------------------
// shared drivers
// ---------------------------------------------------------------------------

/// What the drivers need of a block-scaled matrix: the exact output of one
/// row, and for the batched tile a decoding of rows into 16-bit weights.
trait BlockRows: Sync {
    const NAME: &'static str;
    /// See [`sparse`].
    const SPARSE_DIV: usize;
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn bytes_per_row(&self) -> usize;
    /// `floor(sum_c value(r, c) * x[c] * 2^extra)`, `None` past 128 bits.
    fn row_out(&self, r: usize, x: &[i32], split: Option<&SplitActivation>, extra: i32) -> Option<i128>;
    /// The output from an exact row sum.
    fn finish_sum(&self, s: i128, extra: i32) -> Option<i128>;
    /// The exact row sum at 512 bits (the wide-activation path).
    fn row_sum_wide(&self, r: usize, x: &[i32], split: Option<&SplitActivation>) -> I512;
    /// A wide exact sum to the output scale under the matrix's rounding.
    fn finish_wide(&self, total: I512) -> I512;
    /// Whether [`BlockRows::row_vecs`] may run (AVX2, SIMD-eligible matrix).
    fn direct_ready(&self) -> bool {
        false
    }
    /// Exact sums of row `r` against 1 to `DIRECT_MAX_VECS` vectors of 16-bit
    /// activation halves, each block decoded once in registers (no scratch).
    ///
    /// # Safety
    /// Only when [`BlockRows::direct_ready`] holds; vectors have `cols` elements.
    unsafe fn row_vecs(&self, _r: usize, _vecs: &[&[i16]]) -> [i128; DIRECT_MAX_VECS] {
        unreachable!()
    }
    /// The exact contribution of the listed high halves to row `r`, `<< 16`.
    fn hi_corr(&self, r: usize, xs: &SplitActivation) -> i128;
    /// Whether the batched tile may run (AVX2, SIMD-eligible matrix).
    fn batch_ready(&self) -> bool {
        false
    }
    /// 16-bit weight rows per matrix row in the batched form, and the power
    /// of two of row `c`'s sum.
    fn batch_rows(&self) -> usize {
        1
    }
    fn batch_shift(&self, _c: usize) -> u32 {
        0
    }
    /// Tile steps (16 columns) a 32-bit lane takes before it is widened.
    fn tile_period(&self) -> usize {
        1
    }
    /// Row `r` as `batch_rows()` rows of `cols` 16-bit weights.
    ///
    /// # Safety
    /// Only when [`BlockRows::batch_ready`] holds.
    unsafe fn decode_row(&self, _r: usize, _out: &mut [i16]) {
        unreachable!()
    }
}

impl BlockRows for NvFp4Matrix {
    const NAME: &'static str = "NvFp4Matrix";
    const SPARSE_DIV: usize = 128;
    fn n_rows(&self) -> usize {
        self.rows
    }
    fn n_cols(&self) -> usize {
        self.cols
    }
    fn bytes_per_row(&self) -> usize {
        self.pairs * 16 + self.nb
    }
    #[inline]
    fn row_out(&self, r: usize, x: &[i32], split: Option<&SplitActivation>, extra: i32) -> Option<i128> {
        let s = self.nv_row_sum(r, x, split);
        rounded(self.rounding, extra, |e| self.finish.apply(s, e))
    }
    fn row_sum_wide(&self, r: usize, x: &[i32], split: Option<&SplitActivation>) -> I512 {
        I512::from(self.nv_row_sum(r, x, split))
    }
    fn finish_wide(&self, total: I512) -> I512 {
        // floor / nearest of total * num * 2^shift / den
        let n = total * I512::from(self.finish.num);
        let (n, d) = if self.finish.shift >= 0 {
            (n << self.finish.shift as usize, I512::from(self.finish.div.d as i128))
        } else {
            (n, I512::from(self.finish.div.d as i128) << (-self.finish.shift) as usize)
        };
        crate::exact::round_div_i512(n, d, self.rounding)
    }
    fn finish_sum(&self, s: i128, extra: i32) -> Option<i128> {
        rounded(self.rounding, extra, |e| self.finish.apply(s, e))
    }
    fn direct_ready(&self) -> bool {
        self.batch_ready()
    }
    unsafe fn row_vecs(&self, r: usize, vecs: &[&[i16]]) -> [i128; DIRECT_MAX_VECS] {
        let mut out = [0i128; DIRECT_MAX_VECS];
        #[cfg(target_arch = "x86_64")]
        {
            // one weight vector + T accumulators + one activation: T <= 8 fits the 16 registers
            assert!(r < self.rows && (1..=DIRECT_MAX_VECS).contains(&vecs.len()) && vecs.iter().all(|v| v.len() >= self.cols));
            let m32 = self.mult32.as_deref().unwrap().as_ptr();
            let (pk, sc) = self.row_ptrs(r);
            macro_rules! go {
                ($t:literal) => {{
                    let vs: [*const i16; $t] = std::array::from_fn(|k| vecs[k].as_ptr());
                    let [a] = nvfp4_tile::<1, $t>([pk], [sc], self.nb, m32, vs, self.pairs_per_flush);
                    for k in 0..$t {
                        out[k] = a[k] as i128;
                    }
                }};
            }
            match vecs.len() {
                8 => go!(8),
                7 => go!(7),
                6 => go!(6),
                5 => go!(5),
                4 => go!(4),
                3 => go!(3),
                2 => go!(2),
                _ => go!(1),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (r, vecs);
        out
    }
    fn hi_corr(&self, r: usize, xs: &SplitActivation) -> i128 {
        self.hi_correction(r, xs)
    }
    fn batch_ready(&self) -> bool {
        cfg!(target_arch = "x86_64") && self.mult32.is_some() && avx2_available()
    }
    fn tile_period(&self) -> usize {
        // a step adds at most 2 * wmax * 2^15 per lane; two steps per block pair
        2 * self.pairs_per_flush
    }
    unsafe fn decode_row(&self, r: usize, out: &mut [i16]) {
        #[cfg(target_arch = "x86_64")]
        nvfp4_decode_row(self, r, out);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (r, out);
    }
}

impl NvFp4Matrix {
    /// The exact row sum `sum E2M1_X2 * mult * x` (before the factor).
    #[inline]
    fn nv_row_sum(&self, r: usize, x: &[i32], split: Option<&SplitActivation>) -> i128 {
        #[cfg(target_arch = "x86_64")]
        if let (Some(xs), Some(m32)) = (split, self.mult32.as_deref()) {
            if avx2_available() {
                let (pk, sc) = self.row_ptrs(r);
                // SAFETY: AVX2 checked; the row has nb blocks, the halves cols elements
                let s = unsafe {
                    if sparse(xs, self.cols, Self::SPARSE_DIV) {
                        let [[lo]] = nvfp4_tile::<1, 1>([pk], [sc], self.nb, m32.as_ptr(), [xs.lo.as_ptr()], self.pairs_per_flush);
                        lo as i128 + self.hi_correction(r, xs)
                    } else {
                        let [[lo, hi]] = nvfp4_tile::<1, 2>([pk], [sc], self.nb, m32.as_ptr(), [xs.lo.as_ptr(), xs.hi.as_ptr()], self.pairs_per_flush);
                        lo as i128 + ((hi as i128) << 16)
                    }
                };
                return s;
            }
        }
        self.sum_scalar(r, x)
    }

    fn hi_correction(&self, r: usize, xs: &SplitActivation) -> i128 {
        let mut s = 0i64;
        for &(i, h) in &xs.hi_nz {
            // |weight| < 2^22, |h| < 2^15, fewer than 2^24 terms
            s += self.weight(r, i as usize) as i64 * h as i64;
        }
        (s as i128) << 16
    }
}

impl BlockRows for MxFp8Matrix {
    const NAME: &'static str = "MxFp8Matrix";
    const SPARSE_DIV: usize = 64;
    fn n_rows(&self) -> usize {
        self.rows
    }
    fn n_cols(&self) -> usize {
        self.cols
    }
    fn bytes_per_row(&self) -> usize {
        self.cols + self.cols / 32
    }
    #[inline]
    fn row_out(&self, r: usize, x: &[i32], split: Option<&SplitActivation>, extra: i32) -> Option<i128> {
        match self.mx_row_sum(r, x, split) {
            RowSum::N(s) => rounded(self.rounding, extra, |e| self.finish(s, e)),
            RowSum::W(s) => rounded(self.rounding, extra, |e| self.finish_i512(s, e)),
        }
    }
    fn row_sum_wide(&self, r: usize, x: &[i32], split: Option<&SplitActivation>) -> I512 {
        match self.mx_row_sum(r, x, split) {
            RowSum::N(s) => I512::from(s),
            RowSum::W(s) => s,
        }
    }
    fn finish_wide(&self, total: I512) -> I512 {
        crate::exact::round_shift_i512(total, self.smin - 136, self.rounding)
    }
    fn finish_sum(&self, s: i128, extra: i32) -> Option<i128> {
        rounded(self.rounding, extra, |e| self.finish(s, e))
    }
    // no direct path: the class kernel's cost grows with classes x vectors;
    // the two-limb scratch tile is faster from two activations (measured)
    fn hi_corr(&self, r: usize, xs: &SplitActivation) -> i128 {
        self.hi_correction(r, xs)
    }
    fn batch_ready(&self) -> bool {
        cfg!(target_arch = "x86_64") && self.limbs.is_some() && avx2_available()
    }
    fn batch_rows(&self) -> usize {
        2
    }
    fn batch_shift(&self, c: usize) -> u32 {
        c as u32 * self.limbs.map_or(0, |(b, _)| b)
    }
    fn tile_period(&self) -> usize {
        self.limbs.map_or(1, |(_, p)| p)
    }
    unsafe fn decode_row(&self, r: usize, out: &mut [i16]) {
        #[cfg(target_arch = "x86_64")]
        mxfp8_decode_limbs(self, self.limbs.unwrap().0, r, out);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (r, out);
    }
}

impl MxFp8Matrix {
    /// The exact row sum `sum ELEM * 2^d * x` (before `2^(s_min - 136)`).
    #[inline]
    fn mx_row_sum(&self, r: usize, x: &[i32], split: Option<&SplitActivation>) -> RowSum {
        #[cfg(target_arch = "x86_64")]
        if let (Some(xs), Some(sd)) = (split, self.simd.as_ref()) {
            if avx2_available() {
                let (lo, hi) = (xs.lo.as_ptr(), xs.hi.as_ptr());
                let sparse = sparse(xs, self.cols, Self::SPARSE_DIV);
                macro_rules! run {
                    ($c:literal) => {
                        if sparse {
                            let [s] = mxfp8_row::<$c, 1>(self, sd, r, [lo]);
                            s + self.hi_correction(r, xs)
                        } else {
                            let [l, h] = mxfp8_row::<$c, 2>(self, sd, r, [lo, hi]);
                            l + (h << 16)
                        }
                    };
                }
                // SAFETY: AVX2 checked; the halves have cols elements
                let s = unsafe {
                    match sd.classes {
                        3 => run!(3),
                        4 => run!(4),
                        5 => run!(5),
                        8 => run!(8),
                        _ => run!(15),
                    }
                };
                return RowSum::N(s);
            }
        }
        if self.spread <= MX_I128_SPREAD {
            RowSum::N(self.sum_scalar_i128(r, x))
        } else {
            RowSum::W(self.sum_scalar_i512(r, x))
        }
    }

    fn hi_correction(&self, r: usize, xs: &SplitActivation) -> i128 {
        let (elems, scales) = self.row(r);
        let mut s = 0i128;
        for &(i, h) in &xs.hi_nz {
            let i = i as usize;
            if let Some(d) = self.shift(scales[i / 32]) {
                s += ((ELEM[elems[i] as usize] as i64 * h as i64) as i128) << d;
            }
        }
        s << 16
    }
}

fn check_len<M: BlockRows>(m: &M, x: &[i32], what: &str) {
    assert_eq!(x.len(), m.n_cols(), "{}::{what}: activation length mismatch", M::NAME);
}

fn serial_is_faster<M: BlockRows>(m: &M) -> bool {
    m.n_rows() * m.bytes_per_row() < SERIAL_BELOW_BYTES
}

fn min_rows_per_task<M: BlockRows>(m: &M) -> usize {
    (MIN_TASK_BYTES / m.bytes_per_row().max(1)).max(1)
}

fn matvec_raw<M: BlockRows>(m: &M, x: &[i32]) -> Vec<i32> {
    check_len(m, x, "matvec");
    let split = SplitActivation::new(x);
    (0..m.n_rows()).map(|r| narrow_i32(m.row_out(r, x, split.as_ref(), 0), M::NAME)).collect()
}

fn matvec_par_raw<M: BlockRows>(m: &M, x: &[i32]) -> Vec<i32> {
    check_len(m, x, "matvec_par");
    if serial_is_faster(m) {
        return matvec_raw(m, x);
    }
    let split = SplitActivation::new(x);
    (0..m.n_rows())
        .into_par_iter()
        .with_min_len(min_rows_per_task(m))
        .map(|r| narrow_i32(m.row_out(r, x, split.as_ref(), 0), M::NAME))
        .collect()
}

fn matvec_q2f_raw<M: BlockRows>(m: &M, x: &[i32], frac: u32) -> Vec<i64> {
    check_len(m, x, "matvec_q2f");
    let split = SplitActivation::new(x);
    (0..m.n_rows()).map(|r| narrow_i64(m.row_out(r, x, split.as_ref(), frac as i32), M::NAME)).collect()
}

fn matvec_q2f_par_raw<M: BlockRows>(m: &M, x: &[i32], frac: u32) -> Vec<i64> {
    check_len(m, x, "matvec_q2f_par");
    if serial_is_faster(m) {
        return matvec_q2f_raw(m, x, frac);
    }
    let split = SplitActivation::new(x);
    (0..m.n_rows())
        .into_par_iter()
        .with_min_len(min_rows_per_task(m))
        .map(|r| narrow_i64(m.row_out(r, x, split.as_ref(), frac as i32), M::NAME))
        .collect()
}

thread_local! {
    /// Decoded 16-bit weights of the rows a batched task works on.
    static SCRATCH: std::cell::RefCell<Vec<i16>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Batched matvec into `out[b * rows + r]`.
///
/// When the matrix allows it, each task decodes its rows once into 16-bit
/// weights (L1-resident scratch) and runs a register tile over the
/// activations' 16-bit halves: every splittable activation contributes its
/// low half, and its high half too unless at most one element in eight has
/// one (then a correction over the listed high halves, read from the decoded
/// weights). Vectors go in chunks that stay in L2. Each (activation, row)
/// sum is then combined exactly and finished once. Every other (activation,
/// row) takes the per-row path, a row read once per eight activations. Every
/// path forms the same exact row sum.
fn matvec_batch_par_into_raw<M: BlockRows>(m: &M, batch: &[&[i32]], out: &mut [i32]) {
    for x in batch {
        check_len(m, x, "matvec_batch_par");
    }
    let (rows, cols) = (m.n_rows(), m.n_cols());
    let what = M::NAME;
    let sink = crate::ops::BatchOut::new(out, rows, batch.len(), "matvec_batch_par_into");
    let splits: Vec<Option<SplitActivation>> = batch.iter().map(|x| SplitActivation::new(x)).collect();
    // A few activations (speculative checks, drafting): decode each row once
    // in registers for all of them, no scratch; the same exact sums.
    let direct: Vec<usize> = (0..batch.len()).filter(|&k| splits[k].is_some()).collect();
    let direct_vecs: usize = direct.iter().map(|&k| 1 + !sparse(splits[k].as_ref().unwrap(), cols, M::SPARSE_DIV) as usize).sum();
    if m.direct_ready() && rows > 0 && direct_vecs <= DIRECT_MAX_VECS && direct_vecs > 0 {
        let mut vecs: Vec<&[i16]> = Vec::with_capacity(direct_vecs);
        let mut slot: Vec<(usize, usize, Option<usize>)> = Vec::new(); // (token, lo, hi)
        for &k in &direct {
            let xs = splits[k].as_ref().unwrap();
            let lo = vecs.len();
            vecs.push(&xs.lo);
            let hi = (!sparse(xs, cols, M::SPARSE_DIV)).then(|| {
                vecs.push(&xs.hi);
                vecs.len() - 1
            });
            slot.push((k, lo, hi));
        }
        (0..rows).into_par_iter().with_min_len(min_rows_per_task(m)).for_each(|r| {
            // SAFETY: direct_ready; every vector has cols elements
            let sums = unsafe { m.row_vecs(r, &vecs) };
            for &(k, lo, hi) in &slot {
                let s = sums[lo] + hi.map_or_else(|| m.hi_corr(r, splits[k].as_ref().unwrap()), |h| sums[h] << 16);
                // SAFETY: each row belongs to one task
                unsafe { sink.write(k, r, narrow_i32(m.finish_sum(s, 0), what)) };
            }
        });
        for &k in (0..batch.len()).filter(|&k| splits[k].is_none()).collect::<Vec<_>>().iter() {
            (0..rows).into_par_iter().with_min_len(min_rows_per_task(m)).for_each(|r| {
                // SAFETY: each row belongs to one task
                unsafe { sink.write(k, r, narrow_i32(m.row_out(r, batch[k], None, 0), what)) };
            });
        }
        return;
    }
    let tiled = m.batch_ready() && rows > 0;
    // vectors through the tile; for each, the activation whose sparse high halves it must correct
    let mut vecs: Vec<&[i16]> = Vec::new();
    let mut corr: Vec<Option<usize>> = Vec::new();
    let slots: Vec<Option<(usize, Option<usize>)>> = splits
        .iter()
        .enumerate()
        .map(|(k, s)| match (tiled, s) {
            (true, Some(xs)) => {
                let sparse = sparse(xs, cols, M::SPARSE_DIV);
                let lo = vecs.len();
                vecs.push(&xs.lo);
                corr.push(sparse.then_some(k));
                let hi = (!sparse).then(|| {
                    vecs.push(&xs.hi);
                    corr.push(None);
                    vecs.len() - 1
                });
                Some((lo, hi))
            }
            _ => None,
        })
        .collect();
    if !vecs.is_empty() {
        #[cfg(target_arch = "x86_64")]
        {
            let nv = vecs.len();
            let mut sums = vec![0i128; nv * rows];
            {
                let partial = crate::ops::BatchOut::new(&mut sums, rows, nv, "matvec_batch_par_into sums");
                let classes = m.batch_rows();
                let per_task = (TILE_R / classes).max(1);
                let tasks = rows.div_ceil(per_task);
                let period = m.tile_period();
                // vectors per chunk: 2 bytes per element, within ~256 KiB, a multiple of TILE_T
                let chunk = ((256 * 1024) / (2 * cols.max(1))).clamp(TILE_T, 64) / TILE_T * TILE_T;
                for (ci, vchunk) in vecs.chunks(chunk).enumerate() {
                    (0..tasks).into_par_iter().for_each(|g| {
                        let r0 = g * per_task;
                        let nr = per_task.min(rows - r0);
                        SCRATCH.with(|cell| {
                            let mut scratch = cell.borrow_mut();
                            let width = classes * cols;
                            if scratch.len() < nr * width {
                                scratch.resize(nr * width, 0);
                            }
                            for i in 0..nr {
                                // SAFETY: batch_ready; the slice holds classes * cols words
                                unsafe { m.decode_row(r0 + i, &mut scratch[i * width..(i + 1) * width]) };
                            }
                            let mut part = vec![0i128; vchunk.len() * nr];
                            // weight rows (row, class) in tiles of TILE_R
                            let wrows: Vec<(usize, usize)> = (0..nr).flat_map(|i| (0..classes).map(move |c| (i, c))).collect();
                            for wt in wrows.chunks(TILE_R) {
                                let wp: Vec<*const i16> = wt.iter().map(|&(i, c)| scratch[i * width + c * cols..].as_ptr()).collect();
                                for (bi, blk) in vchunk.chunks(TILE_T).enumerate() {
                                    // SAFETY: AVX2 (batch_ready); rows and vectors hold cols elements
                                    let acc = unsafe { i16_tile_dyn(&wp, blk, cols, period) };
                                    for t in 0..blk.len() {
                                        for (wi, &(i, c)) in wt.iter().enumerate() {
                                            part[(bi * TILE_T + t) * nr + i] += (acc[wi][t] as i128) << m.batch_shift(c);
                                        }
                                    }
                                }
                            }
                            for (vl, _) in vchunk.iter().enumerate() {
                                if let Some(k) = corr[ci * chunk + vl] {
                                    let xs = splits[k].as_ref().unwrap();
                                    for i in 0..nr {
                                        let mut cs = 0i128;
                                        for &(j, h) in &xs.hi_nz {
                                            let mut w = 0i64;
                                            for c in 0..classes {
                                                w += (scratch[i * width + c * cols + j as usize] as i64) << m.batch_shift(c);
                                            }
                                            cs += w as i128 * h as i128;
                                        }
                                        part[vl * nr + i] += cs << 16;
                                    }
                                }
                                for i in 0..nr {
                                    // SAFETY: each (vector, row) belongs to one task
                                    unsafe { partial.write(ci * chunk + vl, r0 + i, part[vl * nr + i]) };
                                }
                            }
                        });
                    });
                }
            }
            let sums = &sums;
            (0..rows).into_par_iter().with_min_len(64).for_each(|r| {
                for (k, slot) in slots.iter().enumerate() {
                    if let Some((lo, hi)) = *slot {
                        let s = sums[lo * rows + r] + hi.map_or(0, |h| sums[h * rows + r] << 16);
                        // SAFETY: each row belongs to one task
                        unsafe { sink.write(k, r, narrow_i32(m.finish_sum(s, 0), what)) };
                    }
                }
            });
        }
    }
    let rest: Vec<usize> = (0..batch.len()).filter(|&k| slots[k].is_none()).collect();
    for tile in rest.chunks(8) {
        (0..rows).into_par_iter().with_min_len(min_rows_per_task(m)).for_each(|r| {
            for &k in tile {
                // SAFETY: each row belongs to one task, each token to one tile
                unsafe { sink.write(k, r, narrow_i32(m.row_out(r, batch[k], splits[k].as_ref(), 0), what)) };
            }
        });
    }
}

/// Activations of a wider word (Q32.32, Q64.64): each balanced 30-bit limb
/// of the activations through the exact 32-bit row sums, the limb sums
/// combined at 512 bits, one rounding at the activations' format.
fn matvec_wide_act_raw<M: BlockRows, W: Storage + Into<i128>>(m: &M, x: &[W], par: bool) -> Vec<W> {
    assert_eq!(x.len(), m.n_cols(), "{}::matvec_wide_act: activation length mismatch", M::NAME);
    let limbs = crate::exact::limbs30(x, <W as WideInt>::BITS);
    let splits: Vec<Option<SplitActivation>> = limbs.iter().map(|(_, l)| SplitActivation::new(l)).collect();
    let row = |r: usize| -> W {
        let mut total = I512::from(0i128);
        for ((k, l), sp) in limbs.iter().zip(&splits) {
            total = total + (m.row_sum_wide(r, l, sp.as_ref()) << (30 * *k as usize));
        }
        crate::exact::narrow_wide::<W>(m.finish_wide(total), M::NAME)
    };
    if par && !serial_is_faster(m) {
        (0..m.n_rows()).into_par_iter().with_min_len(min_rows_per_task(m)).map(row).collect()
    } else {
        (0..m.n_rows()).map(row).collect()
    }
}

macro_rules! public_forms {
    ($t:ty) => {
        impl $t {
            /// `y[r] = floor(sum_c value(r, c) * x[c])` at the activations' format.
            ///
            /// # Panics
            /// Panics on a length mismatch or when a result leaves the storage range.
            pub fn matvec<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<Fixed<i32, F>>
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                matvec_raw(self, Fixed::raw_slice(x)).into_iter().map(Fixed::from_raw).collect()
            }
            /// [`Self::matvec`] with the rows in parallel: the same results.
            pub fn matvec_par<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<Fixed<i32, F>>
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                matvec_par_raw(self, Fixed::raw_slice(x)).into_iter().map(Fixed::from_raw).collect()
            }
            /// A batch of activations, one result vector each: the same results as [`Self::matvec`].
            pub fn matvec_batch_par<const F: u32>(&self, batch: &[&[Fixed<i32, F>]]) -> Vec<Vec<Fixed<i32, F>>>
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                let rows = self.rows;
                let mut flat = vec![Fixed::from_raw(0); batch.len() * rows];
                self.matvec_batch_par_into(batch, &mut flat);
                if rows == 0 {
                    return vec![Vec::new(); batch.len()];
                }
                flat.chunks_exact(rows).map(<[Fixed<i32, F>]>::to_vec).collect()
            }
            /// [`Self::matvec_batch_par`] into a caller buffer of `batch.len() * rows`
            /// values, activation-major (`out[b * rows + r]`).
            pub fn matvec_batch_par_into<const F: u32>(&self, batch: &[&[Fixed<i32, F>]], out: &mut [Fixed<i32, F>])
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                let raw: Vec<&[i32]> = batch.iter().map(|x| Fixed::raw_slice(x)).collect();
                matvec_batch_par_into_raw(self, &raw, Fixed::raw_slice_mut(out))
            }
            /// Wide output at `2F` fractional bits: `floor(sum_c value(r, c) * x[c] * 2^F)`.
            pub fn matvec_q2f<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<i64>
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                matvec_q2f_raw(self, Fixed::raw_slice(x), F)
            }
            /// `matvec` for activations of a wider word (i64 or i128 storage,
            /// e.g. Q32.32 or Q64.64, at any split): the exact value
            /// `sum_c value(r, c) * x[c]` rounded once at the activations'
            /// format by the matrix's rounding mode. The activations run as
            /// exact 30-bit limbs through the 32-bit kernels; a reference
            /// path (a few times a 32-bit matvec), not a hot one.
            pub fn matvec_wide_act<W: Storage + Into<i128>, const F: u32>(&self, x: &[Fixed<W, F>]) -> Vec<Fixed<W, F>>
            where
                Fixed<W, F>: FixedNum<Storage = W>,
            {
                matvec_wide_act_raw(self, Fixed::raw_slice(x), false).into_iter().map(Fixed::from_raw).collect()
            }
            /// [`Self::matvec_wide_act`] with the rows in parallel: the same results.
            pub fn matvec_wide_act_par<W: Storage + Into<i128>, const F: u32>(&self, x: &[Fixed<W, F>]) -> Vec<Fixed<W, F>>
            where
                Fixed<W, F>: FixedNum<Storage = W>,
            {
                matvec_wide_act_raw(self, Fixed::raw_slice(x), true).into_iter().map(Fixed::from_raw).collect()
            }
            /// [`Self::matvec_q2f`] with the rows in parallel: the same results.
            pub fn matvec_q2f_par<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<i64>
            where
                Fixed<i32, F>: FixedNum<Storage = i32>,
            {
                matvec_q2f_par_raw(self, Fixed::raw_slice(x), F)
            }
        }
    };
}
public_forms!(NvFp4Matrix);
public_forms!(MxFp8Matrix);

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn floor_div_matches_div_euclid() {
        let mut rng = Rng(0xD1F);
        let divisors = [1u64, 2, 3, 7, 21, 2688, (1 << 24) - 1, 1 << 40, u64::MAX, (1 << 63) + 5];
        for &d in &divisors {
            let f = FloorDiv::new(d);
            for i in 0..20000 {
                let n: i128 = match i % 4 {
                    0 => ((rng.next() as u128) << 64 | rng.next() as u128) as i128,
                    1 => (rng.next() as i64) as i128,
                    2 => [i128::MIN + 1, i128::MAX, -1, 0, 1, i128::MIN / 3][i % 6],
                    _ => (((rng.next() as u128) << 64 | rng.next() as u128) as i128) >> (rng.next() % 120),
                };
                assert_eq!(f.floor(n), n.div_euclid(d as i128), "n {n} d {d}");
            }
            // i128::MIN is the one value whose magnitude reaches 2^127
            if d > 1 {
                assert_eq!(f.floor(i128::MIN), i128::MIN.div_euclid(d as i128));
            }
        }
    }

    #[test]
    fn elem_table_is_decompose_e4m3() {
        for b in 0..=255u8 {
            match decompose_e4m3(b) {
                Ok((neg, m, e)) => {
                    let v = (m as i64) << (e + 9);
                    assert_eq!(ELEM[b as usize] as i64, if neg { -v } else { v }, "byte {b:#x}");
                }
                Err(_) => assert_eq!(b & 0x7F, 0x7F),
            }
        }
    }
}
