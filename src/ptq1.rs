//! Block-scaled base-3 ternary weights in the PTQ1_0 GGUF block layout,
//! exact in integers, with an optional Walsh-Hadamard input transform.
//!
//! A block holds 128 weights in 28 bytes: 24 bytes of five trits each, 2
//! bytes of four trits each, then the block scale as IEEE binary16 bits
//! (little-endian). Trit `n` of a byte `q` is `((q * 3^n mod 256) * 3) >> 8`
//! minus one, a pure integer function of the byte (every byte decodes,
//! including the 13 a canonical encoder never writes). Element order within a
//! block: bytes 0..16 give elements `16 n + m`, bytes 16..24 give `80 + 8 n +
//! m`, bytes 24..26 give `120 + 2 n + h`. Every finite binary16 value is an
//! integer multiple of `2^-24`, so a weight is exactly `t * S * 2^-24` with
//! `t` in {-1, 0, 1} and `S` an integer of at most 41 bits.
//!
//! With a transform the matrix computes `W (H D x) / sqrt(n)`: `D` an
//! optional diagonal of signs, `H` the Walsh-Hadamard matrix applied to each
//! consecutive block of `n` columns. The transform runs on the exact
//! integers and the `1 / sqrt(n)` (an exact power of two: `n` is a power of
//! four) is folded into the one final rounding.
//!
//! Every path (scalar, AVX2, single, batched, any thread count) forms the
//! same exact integer row sum and rounds it once, so all forms agree bit for
//! bit.

use crate::exact::Rounding;
use crate::tq5::{avx2_available, MIN_TASK_BYTES, SERIAL_BELOW_BYTES};
use g_math::fixed_point::imperative::hadamard::walsh_hadamard_blocks_in_place;
use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::{Fixed, FixedNum};
use rayon::prelude::*;

/// Weights per block.
pub const PTQ1_BLOCK: usize = 128;
/// Bytes per block: 26 bytes of trits and a 2-byte binary16 scale.
pub const PTQ1_BLOCK_BYTES: usize = 28;

/// Largest Walsh-Hadamard block (keeps every transformed activation within
/// 44 bits, three 15-bit limbs).
const MAX_HADAMARD_BLOCK: usize = 4096;
/// Columns per panel of the batched kernel (a row panel and four
/// activations' limbs stay in L1 / L2).
const PANEL: usize = 2048;
/// Binades of block scale folded into the 16-bit weights of a row: blocks
/// down to 2^-4 of the row's largest scale. Smaller nonzero scales are added
/// per block, exactly.
const FOLD_SPAN: u32 = 4;

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// Trit `n` of byte `q`, in {-1, 0, 1}.
#[inline(always)]
fn digit(q: u8, n: usize) -> i8 {
    ((((q.wrapping_mul(POW3[n])) as u16 * 3) >> 8) as i8) - 1
}

/// The 128 trits of one block, in element order.
fn decode_block_scalar(b: &[u8], out: &mut [i8; PTQ1_BLOCK]) {
    for n in 0..5 {
        for m in 0..16 {
            out[16 * n + m] = digit(b[m], n);
        }
    }
    for n in 0..5 {
        for m in 0..8 {
            out[80 + 8 * n + m] = digit(b[16 + m], n);
        }
    }
    for n in 0..4 {
        for h in 0..2 {
            out[120 + 2 * n + h] = digit(b[24 + h], n);
        }
    }
}

/// A finite binary16 scale `S` (value `S * 2^-24`) as `(M, x)`, `S = M * 2^x`,
/// `|M| < 2^11`.
#[inline(always)]
fn scale_parts(h: u16) -> (i64, u32) {
    let e = ((h >> 10) & 31) as u32;
    let m = (h & 1023) as i64;
    let (mag, x) = if e == 0 { (m, 0) } else { (1024 | m, e - 1) };
    (if h & 0x8000 != 0 { -mag } else { mag }, x)
}

/// A binary16 value as `S` with value `S * 2^-24`; `None` for infinities and NaNs.
#[inline(always)]
fn half_s24(h: u16) -> Option<i64> {
    let e = (h >> 10) & 31;
    let m = (h & 1023) as i64;
    if e == 31 {
        return None;
    }
    let mag = if e == 0 { m } else { (1024 | m) << (e - 1) };
    Some(if h & 0x8000 != 0 { -mag } else { mag })
}

#[derive(Clone, Debug)]
struct Hadamard {
    block: usize,
    /// log2(block) / 2: the result is divided by 2^half_log.
    half_log: u32,
    signs: Option<Vec<i8>>,
}

/// A row-major ternary matrix in PTQ1_0 blocks of 128 weights (see the
/// module documentation), with an optional Walsh-Hadamard input transform.
///
/// `matvec` returns `floor(sum_c w(r, c) * u_c)` at the activations' format,
/// `u = x` or `u = H D x / sqrt(n)`; `Rounding::Nearest` rounds the same exact
/// value to nearest (ties toward +infinity).
#[derive(Clone, Debug)]
pub struct Ptq1Matrix {
    rows: usize,
    cols: usize,
    blocks: Vec<u8>,
    hadamard: Option<Hadamard>,
    rounding: Rounding,
    fold: Fold,
}

/// How the block scales fold into 16-bit weights (computed at construction).
/// Row `r`'s weight in a block with scale `M * 2^x` is `t * M * 2^(x - x_ref[r])`
/// when `x_ref[r] <= x` (at most `2047 * 2^FOLD_SPAN`); the row's other
/// nonzero blocks (`outliers`) are added per block.
#[derive(Clone, Debug)]
struct Fold {
    x_ref: Vec<u8>,
    /// Row `r`'s outlier blocks are `outliers[start[r]..start[r + 1]]`.
    start: Vec<u32>,
    outliers: Vec<u32>,
    /// The largest folded weight magnitude (bounds the 32-bit lanes).
    max_w: u32,
}

impl Fold {
    fn new(rows: usize, cols: usize, blocks: &[u8]) -> Self {
        let nb = cols / PTQ1_BLOCK;
        let (mut x_ref, mut start, mut outliers) = (Vec::with_capacity(rows), Vec::with_capacity(rows + 1), Vec::new());
        let mut max_w = 0i64;
        start.push(0u32);
        for r in 0..rows {
            let parts: Vec<(i64, u32)> = (0..nb)
                .map(|b| {
                    let i = (r * nb + b) * PTQ1_BLOCK_BYTES;
                    scale_parts(u16::from_le_bytes([blocks[i + 26], blocks[i + 27]]))
                })
                .collect();
            let nonzero = || parts.iter().filter(|p| p.0 != 0);
            let x_max = nonzero().map(|p| p.1).max().unwrap_or(0);
            let x_min = nonzero().map(|p| p.1).min().unwrap_or(0);
            let xr = x_min.max(x_max.saturating_sub(FOLD_SPAN));
            for (b, &(m, x)) in parts.iter().enumerate() {
                if m == 0 {
                    continue;
                }
                if x < xr {
                    outliers.push(b as u32);
                } else {
                    max_w = max_w.max(m.abs() << (x - xr));
                }
            }
            x_ref.push(xr as u8);
            start.push(outliers.len() as u32);
        }
        Fold { x_ref, start, outliers, max_w: max_w as u32 }
    }

    /// The folded weight factor of a block of row `r`: `M * 2^(x - x_ref)`, or 0.
    #[inline(always)]
    fn mult(&self, r: usize, blk: &[u8]) -> i16 {
        let (m, x) = scale_parts(u16::from_le_bytes([blk[26], blk[27]]));
        let xr = self.x_ref[r] as u32;
        if m == 0 || x < xr { 0 } else { (m << (x - xr)) as i16 }
    }
}

/// One activation vector prepared for the kernels: the exact (transformed)
/// integers, and their limbs of `lb` bits once the batch's limb count is known.
struct Prepared {
    u: Vec<i64>,
    /// `limbs * cols` words, limb-major; limb `l` carries weight `2^(lb l)`.
    split: Vec<i16>,
}

impl Ptq1Matrix {
    /// A matrix from its blocks as stored: `rows * cols / 128` blocks of 28
    /// bytes, row-major.
    ///
    /// # Errors
    /// `InvalidInput` if `cols` is not a multiple of 128, the byte count does
    /// not match, or a block scale is infinite or NaN.
    pub fn from_parts(rows: usize, cols: usize, blocks: Vec<u8>) -> Result<Self, OverflowDetected> {
        if cols % PTQ1_BLOCK != 0 || blocks.len() != rows * (cols / PTQ1_BLOCK) * PTQ1_BLOCK_BYTES {
            return Err(OverflowDetected::InvalidInput);
        }
        if blocks.chunks_exact(PTQ1_BLOCK_BYTES).any(|b| half_s24(u16::from_le_bytes([b[26], b[27]])).is_none()) {
            return Err(OverflowDetected::InvalidInput);
        }
        let fold = Fold::new(rows, cols, &blocks);
        Ok(Self { rows, cols, blocks, hadamard: None, rounding: Rounding::Default, fold })
    }

    /// The same matrix applied to `H D x / sqrt(block)`: an input transform by
    /// the Walsh-Hadamard matrix on each consecutive `block` columns, after the
    /// optional signs `D` (one per column, each +1 or -1).
    ///
    /// # Errors
    /// `InvalidInput` unless `block` is a power of four from 4 to 4096 that
    /// divides `cols`, and `signs` (if given) has `cols` entries of +1 or -1.
    pub fn with_hadamard(mut self, block: usize, signs: Option<Vec<i8>>) -> Result<Self, OverflowDetected> {
        let k = block.trailing_zeros();
        if !block.is_power_of_two() || k % 2 != 0 || !(4..=MAX_HADAMARD_BLOCK).contains(&block) || self.cols % block != 0 {
            return Err(OverflowDetected::InvalidInput);
        }
        if let Some(s) = &signs {
            if s.len() != self.cols || s.iter().any(|&v| v != 1 && v != -1) {
                return Err(OverflowDetected::InvalidInput);
            }
        }
        self.hadamard = Some(Hadamard { block, half_log: k / 2, signs });
        Ok(self)
    }

    /// The same matrix rounding its outputs by `rounding`.
    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    /// The rounding mode applied to every output.
    pub fn rounding(&self) -> Rounding {
        self.rounding
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// The blocks as stored (28 bytes per 128 weights, row-major).
    pub fn blocks(&self) -> &[u8] {
        &self.blocks
    }
    /// The Walsh-Hadamard block of the input transform, if any.
    pub fn hadamard_block(&self) -> Option<usize> {
        self.hadamard.as_ref().map(|h| h.block)
    }
    /// Bytes held: the blocks (1.75 bits per weight).
    pub fn size_bytes(&self) -> usize {
        self.blocks.len()
    }

    /// The exact weights of row `r` as integers `n` with `w = n * 2^-24`
    /// (the weights as stored, before any input transform). For gates and
    /// debugging, not the hot path.
    pub fn dequant_row_exact(&self, r: usize) -> Vec<i64> {
        let nb = self.cols / PTQ1_BLOCK;
        let mut out = Vec::with_capacity(self.cols);
        let mut t = [0i8; PTQ1_BLOCK];
        for b in 0..nb {
            let blk = self.block(r, b);
            decode_block_scalar(blk, &mut t);
            let s = self.scale(r, b);
            out.extend(t.iter().map(|&v| v as i64 * s));
        }
        out
    }

    #[inline(always)]
    fn block(&self, r: usize, b: usize) -> &[u8] {
        let i = (r * (self.cols / PTQ1_BLOCK) + b) * PTQ1_BLOCK_BYTES;
        &self.blocks[i..i + PTQ1_BLOCK_BYTES]
    }

    #[inline(always)]
    fn scale(&self, r: usize, b: usize) -> i64 {
        let blk = self.block(r, b);
        // finite: checked at construction
        half_s24(u16::from_le_bytes([blk[26], blk[27]])).unwrap_or(0)
    }

    /// The input transform of one activation vector (exact integers).
    fn prepare(&self, x: &[i32]) -> Prepared {
        let mut u: Vec<i64> = x.iter().map(|&v| v as i64).collect();
        if let Some(h) = &self.hadamard {
            if let Some(s) = &h.signs {
                for (v, &s) in u.iter_mut().zip(s) {
                    if s < 0 {
                        *v = -*v;
                    }
                }
            }
            // |x| <= 2^31 and block <= 2^12: every value fits 44 bits
            walsh_hadamard_blocks_in_place(&mut u, h.block).expect("Ptq1Matrix: transform within 44 bits");
        }
        Prepared { u, split: Vec::new() }
    }

    /// The activation limbs for a batch: `(lb, limbs, period)`. Limbs of `lb`
    /// bits (the top one in `[-2^lb, 2^lb)`) hold every prepared activation; a
    /// 32-bit lane gains at most `2 * max_w * 2^lb` per 16-column step, so it
    /// moves to 64 bits every `period` steps. Of the widths from 8 to 15 bits,
    /// the one with the least work: limbs times (1 + 0.35 / period), measured
    /// (a flush costs about a third of a step).
    fn limb_plan(&self, inputs: &[Prepared]) -> (u32, usize, usize) {
        let lo = inputs.iter().flat_map(|p| p.u.iter()).copied().min().unwrap_or(0);
        let hi = inputs.iter().flat_map(|p| p.u.iter()).copied().max().unwrap_or(0);
        let mut best: Option<(u32, usize, usize, u64)> = None;
        for lb in (8..=15u32).rev() {
            let pair = 2 * self.fold.max_w as u64 * (1u64 << lb);
            let period = if pair == 0 { usize::MAX } else { (((1u64 << 31) - 1) / pair) as usize };
            if period == 0 {
                continue;
            }
            let Some(limbs) = (1..=4usize).find(|&l| {
                let k = lb * (l as u32 - 1);
                (lo >> k) >= -(1i64 << lb) && (hi >> k) < 1i64 << lb
            }) else {
                continue;
            };
            // cost in thousandths of a step per limb
            let cost = limbs as u64 * (1000 + 350 / period.min(1 << 20) as u64);
            if best.is_none_or(|b| cost < b.3) {
                best = Some((lb, limbs, period, cost));
            }
        }
        let (lb, limbs, period, _) = best.expect("Ptq1Matrix: a limb width for 44-bit activations");
        (lb, limbs, period)
    }

    /// The limbs of one prepared activation: `u = sum_l limb_l * 2^(lb l)`.
    fn split(p: &mut Prepared, lb: u32, limbs: usize) {
        let cols = p.u.len();
        let mut split = vec![0i16; limbs * cols];
        for (c, &v0) in p.u.iter().enumerate() {
            let mut v = v0;
            for l in 0..limbs - 1 {
                split[l * cols + c] = (v & ((1 << lb) - 1)) as i16;
                v >>= lb;
            }
            split[(limbs - 1) * cols + c] = v as i16;
        }
        p.split = split;
    }

    /// The exact row sum `T` with `y = T * 2^-(24 + half_log)` (activation units),
    /// scalar: the reference every other path equals.
    fn row_sum_scalar(&self, r: usize, p: &Prepared) -> i128 {
        let nb = self.cols / PTQ1_BLOCK;
        let mut t = [0i8; PTQ1_BLOCK];
        let mut total = 0i128;
        for b in 0..nb {
            decode_block_scalar(self.block(r, b), &mut t);
            let u = &p.u[b * PTQ1_BLOCK..(b + 1) * PTQ1_BLOCK];
            let mut s = 0i64;
            for (k, &w) in t.iter().enumerate() {
                s += w as i64 * u[k];
            }
            total += self.scale(r, b) as i128 * s as i128;
        }
        total
    }

    /// `floor(T * 2^(extra - 24 - half_log))` under the matrix's rounding;
    /// `None` when it does not fit 128 bits.
    fn finish(&self, t: i128, extra: i32) -> Option<i128> {
        let base = 24 + self.hadamard.as_ref().map_or(0, |h| h.half_log) as i32;
        let floor = |e: i32| -> Option<i128> {
            let k = base - e;
            if k >= 0 {
                Some(t >> (k.min(127) as u32))
            } else {
                let k = (-k) as u32;
                if t == 0 {
                    Some(0)
                } else if k >= 127 || t.unsigned_abs().leading_zeros() <= k {
                    None
                } else {
                    Some(t << k)
                }
            }
        };
        match self.rounding {
            Rounding::Default => floor(extra),
            Rounding::Nearest => floor(extra + 1).and_then(|v| v.checked_add(1)).map(|v| v >> 1),
        }
    }

    /// Every exact row sum of every prepared activation, handed to `sink(k, r, T)`
    /// (each pair once, from one task).
    ///
    /// AVX2: each task takes a range of rows; for each chunk of activations
    /// (their limbs within L2) and each column panel it decodes a row's panel
    /// once into 16-bit folded weights and runs it against four activations at
    /// a time; the outlier blocks are added per block at the end.
    fn sums(&self, inputs: &mut [Prepared], par: bool, sink: &(dyn Fn(usize, usize, i128) + Sync)) {
        let (rows, cols) = (self.rows, self.cols);
        if rows == 0 || inputs.is_empty() {
            return;
        }
        let nb = cols / PTQ1_BLOCK;
        let simd = cfg!(target_arch = "x86_64") && avx2_available();
        let (lb, limbs, period) = if simd { self.limb_plan(inputs) } else { (0, 0, 0) };
        if simd {
            if par && inputs.len() > 1 {
                inputs.par_iter_mut().for_each(|p| Self::split(p, lb, limbs));
            } else {
                inputs.iter_mut().for_each(|p| Self::split(p, lb, limbs));
            }
        }
        let inputs: &[Prepared] = inputs;
        let bytes_per_row = nb * PTQ1_BLOCK_BYTES;
        let rows_per_task = (MIN_TASK_BYTES / bytes_per_row.max(1)).max(1);
        let tasks = rows.div_ceil(rows_per_task);
        let task = |g: usize| {
            let r0 = g * rows_per_task;
            let nr = rows_per_task.min(rows - r0);
            if !simd {
                for (k, p) in inputs.iter().enumerate() {
                    for r in r0..r0 + nr {
                        sink(k, r, self.row_sum_scalar(r, p));
                    }
                }
                return;
            }
            #[cfg(target_arch = "x86_64")]
            {
                let n = inputs.len();
                let panel = PANEL.min(cols);
                // activations per chunk: four of them for one panel in ~48 KiB, the chunk in ~192 KiB
                let chunk = ((192 * 1024) / (2 * limbs * panel)).clamp(4, 64) / 4 * 4;
                let mut acc = vec![0i128; nr * n];
                let mut scratch = vec![0i16; panel];
                let mut out = [[0i64; 4]; 4];
                for (ci, cin) in inputs.chunks(chunk).enumerate() {
                    for p0 in (0..cols).step_by(panel) {
                        let pw = panel.min(cols - p0);
                        for i in 0..nr {
                            let r = r0 + i;
                            for b in 0..pw / PTQ1_BLOCK {
                                let blk = self.block(r, p0 / PTQ1_BLOCK + b);
                                // SAFETY: AVX2 checked; the block has 28 bytes, the scratch 128 words from b
                                unsafe { decode_block_avx2(blk.as_ptr(), scratch[b * PTQ1_BLOCK..].as_mut_ptr(), self.fold.mult(r, blk)) };
                            }
                            for (qi, quad) in cin.chunks(4).enumerate() {
                                let mut ptrs = [[std::ptr::null::<i16>(); 4]; 4];
                                for (t, p) in quad.iter().enumerate() {
                                    for (j, ptr) in ptrs[t].iter_mut().enumerate().take(limbs) {
                                        *ptr = p.split[j * cols + p0..].as_ptr();
                                    }
                                }
                                // SAFETY: AVX2 checked; the scratch and every limb slice hold pw words
                                // from the panel start; the fold's period keeps every lane in range
                                unsafe { crate::simd::limb_tile_dyn(scratch.as_ptr(), &ptrs[..quad.len()], limbs, pw / 16, period, &mut out) };
                                for (t, sums) in out[..quad.len()].iter().enumerate() {
                                    let k = ci * chunk + 4 * qi + t;
                                    let s: i128 = (0..limbs).map(|j| (sums[j] as i128) << (lb * j as u32)).sum();
                                    acc[i * n + k] += s;
                                }
                            }
                        }
                    }
                }
                let mut trits = [0i8; PTQ1_BLOCK];
                for i in 0..nr {
                    let r = r0 + i;
                    let xr = self.fold.x_ref[r] as u32;
                    for v in &mut acc[i * n..(i + 1) * n] {
                        *v <<= xr;
                    }
                    let (a, z) = (self.fold.start[r] as usize, self.fold.start[r + 1] as usize);
                    for &b in &self.fold.outliers[a..z] {
                        let b = b as usize;
                        decode_block_scalar(self.block(r, b), &mut trits);
                        let s = self.scale(r, b) as i128;
                        for (k, p) in inputs.iter().enumerate() {
                            let u = &p.u[b * PTQ1_BLOCK..(b + 1) * PTQ1_BLOCK];
                            let d: i64 = trits.iter().zip(u).map(|(&t, &v)| t as i64 * v).sum();
                            acc[i * n + k] += s * d as i128;
                        }
                    }
                    for k in 0..n {
                        sink(k, r, acc[i * n + k]);
                    }
                }
            }
        };
        if par && rows * bytes_per_row >= SERIAL_BELOW_BYTES {
            (0..tasks).into_par_iter().for_each(task);
        } else {
            (0..tasks).for_each(task);
        }
    }

    fn check(&self, x: &[i32], what: &str) {
        assert_eq!(x.len(), self.cols, "Ptq1Matrix::{what}: activation length mismatch");
    }

    fn run<T: Copy + Send + Default>(&self, batch: &[&[i32]], par: bool, extra: i32, narrow: fn(Option<i128>) -> T, out: &mut [T]) {
        let mut inputs: Vec<Prepared> = if par && batch.len() > 1 {
            batch.par_iter().map(|x| self.prepare(x)).collect()
        } else {
            batch.iter().map(|x| self.prepare(x)).collect()
        };
        let sink = crate::ops::BatchOut::new(out, self.rows, batch.len(), "Ptq1Matrix");
        // SAFETY: `sums` hands every (activation, row) to exactly one call
        self.sums(&mut inputs, par, &|k, r, t| unsafe { sink.write(k, r, narrow(self.finish(t, extra))) });
    }

    fn raw_out(&self, x: &[i32], par: bool, what: &str) -> Vec<i32> {
        self.check(x, what);
        let mut out = vec![0i32; self.rows];
        self.run(&[x], par, 0, narrow_i32, &mut out);
        out
    }

    fn raw_q2f(&self, x: &[i32], frac: u32, par: bool, what: &str) -> Vec<i64> {
        self.check(x, what);
        let mut out = vec![0i64; self.rows];
        self.run(&[x], par, frac as i32, narrow_i64, &mut out);
        out
    }

    /// `y[r] = floor(sum_c w(r, c) * u_c)` at the activations' format.
    ///
    /// # Panics
    /// Panics on a length mismatch or when a result leaves the storage range.
    pub fn matvec<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<Fixed<i32, F>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        self.raw_out(Fixed::raw_slice(x), false, "matvec").into_iter().map(Fixed::from_raw).collect()
    }

    /// [`Self::matvec`] with the rows in parallel: the same results.
    pub fn matvec_par<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<Fixed<i32, F>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        self.raw_out(Fixed::raw_slice(x), true, "matvec_par").into_iter().map(Fixed::from_raw).collect()
    }

    /// Wide output at `2F` fractional bits: `floor(sum_c w(r, c) * u_c * 2^F)`.
    pub fn matvec_q2f<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<i64>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        self.raw_q2f(Fixed::raw_slice(x), F, false, "matvec_q2f")
    }

    /// [`Self::matvec_q2f`] with the rows in parallel: the same results.
    pub fn matvec_q2f_par<const F: u32>(&self, x: &[Fixed<i32, F>]) -> Vec<i64>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        self.raw_q2f(Fixed::raw_slice(x), F, true, "matvec_q2f_par")
    }

    /// A batch of activations, one result vector each: the same results as
    /// [`Self::matvec`]. Each task decodes a block of four rows once and runs
    /// it against every activation (prefill, speculative checks).
    pub fn matvec_batch_par<const F: u32>(&self, batch: &[&[Fixed<i32, F>]]) -> Vec<Vec<Fixed<i32, F>>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let mut flat = vec![Fixed::from_raw(0); batch.len() * self.rows];
        self.matvec_batch_par_into(batch, &mut flat);
        split_rows(flat, batch.len(), self.rows)
    }

    /// [`Self::matvec_batch_par`] into a caller buffer of `batch.len() * rows`
    /// values, activation-major (`out[b * rows + r]`).
    pub fn matvec_batch_par_into<const F: u32>(&self, batch: &[&[Fixed<i32, F>]], out: &mut [Fixed<i32, F>])
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let raw: Vec<&[i32]> = batch.iter().map(|x| Fixed::raw_slice(x)).collect();
        for x in &raw {
            self.check(x, "matvec_batch_par");
        }
        self.run(&raw, true, 0, narrow_i32, Fixed::raw_slice_mut(out));
    }

    /// The batched form of [`Self::matvec_q2f`]: the same results.
    pub fn matvec_q2f_batch_par<const F: u32>(&self, batch: &[&[Fixed<i32, F>]]) -> Vec<Vec<i64>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let mut flat = vec![0i64; batch.len() * self.rows];
        self.matvec_q2f_batch_par_into(batch, &mut flat);
        split_rows(flat, batch.len(), self.rows)
    }

    /// [`Self::matvec_q2f_batch_par`] into a caller buffer, activation-major.
    pub fn matvec_q2f_batch_par_into<const F: u32>(&self, batch: &[&[Fixed<i32, F>]], out: &mut [i64])
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let raw: Vec<&[i32]> = batch.iter().map(|x| Fixed::raw_slice(x)).collect();
        for x in &raw {
            self.check(x, "matvec_q2f_batch_par");
        }
        self.run(&raw, true, F as i32, narrow_i64, out);
    }

    /// The single-row scalar reference (tests).
    #[cfg(test)]
    fn reference(&self, x: &[i32], extra: i32) -> Vec<Option<i128>> {
        let p = self.prepare(x);
        (0..self.rows).map(|r| self.finish(self.row_sum_scalar(r, &p), extra)).collect()
    }
}

fn split_rows<T: Clone>(flat: Vec<T>, n: usize, rows: usize) -> Vec<Vec<T>> {
    if rows == 0 {
        return vec![Vec::new(); n];
    }
    flat.chunks_exact(rows).map(<[T]>::to_vec).collect()
}

fn narrow_i32(v: Option<i128>) -> i32 {
    match v {
        Some(v) if v >= i32::MIN as i128 && v <= i32::MAX as i128 => v as i32,
        _ => panic!("Ptq1Matrix: output exceeds storage range"),
    }
}

fn narrow_i64(v: Option<i128>) -> i64 {
    match v {
        Some(v) if v >= i64::MIN as i128 && v <= i64::MAX as i128 => v as i64,
        _ => panic!("Ptq1Matrix: q2f output exceeds compute range"),
    }
}

// ---------------------------------------------------------------------------
// AVX2: decode a block to 16-bit trits; multiply-add trits by activation limbs
// ---------------------------------------------------------------------------

/// The 128 trits of one block times `mult` (the folded scale) as i16 in element order.
///
/// With the byte in the high half of a 16-bit lane, `(q << 8) * 3^n mod 2^16`
/// is `(q * 3^n mod 256) << 8`, and its high product with 3 is
/// `(3 (q * 3^n mod 256)) >> 8`: the trit plus one, in two multiplies.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_block_avx2(b: *const u8, out: *mut i16, mult: i16) {
    use std::arch::x86_64::*;
    let three = _mm256_set1_epi16(3);
    let mv = _mm256_set1_epi16(mult);
    // weight = (t + 1 - 1) * mult, |mult| < 2^15
    let weights = |vh: __m256i, pow: __m256i| -> __m256i {
        let t1 = _mm256_mulhi_epu16(_mm256_mullo_epi16(vh, pow), three);
        _mm256_sub_epi16(_mm256_mullo_epi16(t1, mv), mv)
    };
    // bytes 0..16: digit n is elements 16 n .. 16 n + 16
    let a = _mm256_slli_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(b as *const __m128i)), 8);
    for (n, &p) in POW3.iter().enumerate() {
        _mm256_storeu_si256(out.add(16 * n) as *mut __m256i, weights(a, _mm256_set1_epi16(p as i16)));
    }
    // bytes 16..24: digits (n, n + 1) are elements 80 + 8 n .. 80 + 8 n + 16
    let bl = _mm_slli_epi16(_mm_cvtepu8_epi16(_mm_loadl_epi64(b.add(16) as *const __m128i)), 8);
    let bb = _mm256_set_m128i(bl, bl);
    for n in [0usize, 2] {
        let pow = _mm256_set_m128i(_mm_set1_epi16(POW3[n + 1] as i16), _mm_set1_epi16(POW3[n] as i16));
        _mm256_storeu_si256(out.add(80 + 8 * n) as *mut __m256i, weights(bb, pow));
    }
    // digit 4 of bytes 16..24 (elements 112..120), then bytes 24..26 (elements 120 + 2 n + h)
    let (h0, h1) = ((*b.add(24) as i16) << 8, (*b.add(25) as i16) << 8);
    let qh = _mm_set_epi16(h1, h0, h1, h0, h1, h0, h1, h0);
    let v = _mm256_set_m128i(qh, bl);
    let pow = _mm256_set_m128i(_mm_set_epi16(27, 27, 9, 9, 3, 3, 1, 1), _mm_set1_epi16(81));
    _mm256_storeu_si256(out.add(112) as *mut __m256i, weights(v, pow));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 17
    }

    /// llama.cpp's dequantize_row_ptq1_0 loop, transcribed (stages 32/16/8).
    fn llama_trits(qs: &[u8], qh: &[u8]) -> Vec<i8> {
        let pow3 = [1u8, 3, 9, 27, 81, 243];
        let mut y = Vec::new();
        let mut j = 0usize;
        for c in [32usize, 16, 8] {
            while j + c <= qs.len() {
                for n in 0..5 {
                    for m in 0..c {
                        let q = qs[j + m].wrapping_mul(pow3[n]);
                        let xi = ((q as u16 * 3) >> 8) as i16;
                        y.push((xi - 1) as i8);
                    }
                }
                j += c;
            }
        }
        for n in 0..4 {
            for h in 0..qh.len() {
                let q = qh[h].wrapping_mul(pow3[n]);
                let xi = ((q as u16 * 3) >> 8) as i16;
                y.push((xi - 1) as i8);
            }
        }
        y
    }

    #[test]
    fn decode_equals_the_reference_loop_for_every_byte() {
        let mut seed = 0x9713u64;
        let mut out = [0i8; PTQ1_BLOCK];
        let mut wide = [0i16; PTQ1_BLOCK];
        for round in 0..2000 {
            let mut b = [0u8; PTQ1_BLOCK_BYTES];
            for (i, v) in b.iter_mut().enumerate().take(26) {
                // the first 256 rounds walk every byte value through every position
                *v = if round < 256 { (round + i) as u8 } else { lcg(&mut seed) as u8 };
            }
            let want = llama_trits(&b[..24], &b[24..26]);
            decode_block_scalar(&b, &mut out);
            assert_eq!(&out[..], &want[..], "round {round}");
            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                unsafe { decode_block_avx2(b.as_ptr(), wide.as_mut_ptr(), 1) };
                assert!(wide.iter().zip(&want).all(|(a, b)| *a == *b as i16), "avx2 round {round}");
                // the folded factor at its extremes
                for mult in [-32752i16, 32752, 2047, -1023, 0] {
                    unsafe { decode_block_avx2(b.as_ptr(), wide.as_mut_ptr(), mult) };
                    assert!(wide.iter().zip(&want).all(|(a, b)| *a as i32 == *b as i32 * mult as i32), "avx2 round {round} mult {mult}");
                }
            }
        }
    }

    #[test]
    fn canonical_encoding_round_trips() {
        // the reference quantizer's packing: 5 trits most significant first, then ceil(v * 256 / 243)
        for v in 0..243u16 {
            let q = ((v * 256 + 242) / 243) as u8;
            let mut d = v;
            let mut trits = [0i8; 5];
            for n in (0..5).rev() {
                trits[n] = (d % 3) as i8 - 1;
                d /= 3;
            }
            for (n, &t) in trits.iter().enumerate() {
                assert_eq!(digit(q, n), t, "v {v} digit {n}");
            }
        }
        // the 4-trit bytes: shifted up one trit before packing
        for v in 0..81u16 {
            let q = ((v * 3 * 256 + 242) / 243) as u8;
            let mut d = v;
            let mut trits = [0i8; 4];
            for n in (0..4).rev() {
                trits[n] = (d % 3) as i8 - 1;
                d /= 3;
            }
            for (n, &t) in trits.iter().enumerate() {
                assert_eq!(digit(q, n), t, "4-trit v {v} digit {n}");
            }
        }
    }

    #[test]
    fn binary16_scales_are_exact() {
        assert_eq!(half_s24(0x3C00), Some(1 << 24)); // 1.0
        assert_eq!(half_s24(0x0001), Some(1)); // smallest subnormal 2^-24
        assert_eq!(half_s24(0x7BFF), Some(65504 << 24)); // largest finite
        assert_eq!(half_s24(0xC000), Some(-(2 << 24))); // -2.0
        assert_eq!(half_s24(0x8000), Some(0));
        assert_eq!(half_s24(0x7C00), None);
        assert_eq!(half_s24(0x7E00), None);
        // every finite pattern: S * 2^-24 equals (-1)^s (1 + m / 1024) 2^(e - 15) or the subnormal form
        for h in 0..=u16::MAX {
            let e = (h >> 10) & 31;
            if e == 31 {
                continue;
            }
            let m = (h & 1023) as i128;
            // 2 S = value * 2^25: (1024 + m) 2^e for normal values, 2 m for subnormal ones
            let want = if e == 0 { 2 * m } else { (1024 + m) << e };
            let want = if h & 0x8000 != 0 { -want } else { want };
            assert_eq!(2 * half_s24(h).unwrap() as i128, want, "h {h:#06x}");
        }
    }

    fn random_matrix(seed: &mut u64, rows: usize, cols: usize) -> Ptq1Matrix {
        let nb = rows * cols / PTQ1_BLOCK;
        let mut blocks = vec![0u8; nb * PTQ1_BLOCK_BYTES];
        for b in blocks.chunks_exact_mut(PTQ1_BLOCK_BYTES) {
            for v in b.iter_mut().take(26) {
                *v = lcg(seed) as u8;
            }
            // finite binary16 scales of either sign, exponents across the range
            let mut h = (lcg(seed) & 0xFFFF) as u16;
            if (h >> 10) & 31 == 31 {
                h &= !(1 << 14);
            }
            b[26..28].copy_from_slice(&h.to_le_bytes());
        }
        Ptq1Matrix::from_parts(rows, cols, blocks).unwrap()
    }

    fn activations(seed: &mut u64, n: usize, bits: u32) -> Vec<i32> {
        (0..n).map(|_| ((lcg(seed) as i64 % (1i64 << bits)) - (1i64 << (bits - 1))) as i32).collect()
    }

    #[test]
    fn every_path_equals_the_scalar_reference() {
        let mut seed = 0x51D5u64;
        let mut checked = 0usize;
        for &(rows, cols, had) in &[(7usize, 256usize, None), (9, 1024, Some(1024usize)), (5, 512, Some(16)), (4, 128, Some(64))] {
            for &bits in &[8u32, 14, 20, 31] {
                let mut m = random_matrix(&mut seed, rows, cols);
                // small scales so outputs fit i32
                for b in m.blocks.chunks_exact_mut(PTQ1_BLOCK_BYTES) {
                    let h = u16::from_le_bytes([b[26], b[27]]) & 0x83FF; // subnormal: S < 1024
                    b[26..28].copy_from_slice(&h.to_le_bytes());
                }
                // from the final bytes (the fold is computed at construction)
                m = Ptq1Matrix::from_parts(rows, cols, m.blocks.clone()).unwrap();
                if let Some(h) = had {
                    let signs: Vec<i8> = (0..cols).map(|_| if lcg(&mut seed) & 1 == 0 { 1 } else { -1 }).collect();
                    m = m.with_hadamard(h, Some(signs)).unwrap();
                }
                for rounding in [Rounding::Default, Rounding::Nearest] {
                    let m = m.clone().with_rounding(rounding);
                    let batch: Vec<Vec<i32>> = (0..9).map(|_| activations(&mut seed, cols, bits)).collect();
                    for extra in [0i32, 10] {
                        let want: Vec<Vec<Option<i128>>> = batch.iter().map(|x| m.reference(x, extra)).collect();
                        let fits = want.iter().flatten().all(|v| v.is_some_and(|v| if extra == 0 { i32::try_from(v).is_ok() } else { i64::try_from(v).is_ok() }));
                        if !fits {
                            continue;
                        }
                        let refs: Vec<&[i32]> = batch.iter().map(|v| &v[..]).collect();
                        checked += 1;
                        for par in [false, true] {
                            if extra == 0 {
                                let mut out = vec![0i32; refs.len() * rows];
                                m.run(&refs, par, 0, narrow_i32, &mut out);
                                for (k, w) in want.iter().enumerate() {
                                    for r in 0..rows {
                                        assert_eq!(out[k * rows + r] as i128, w[r].unwrap(), "{rows}x{cols} had {had:?} bits {bits} k {k} r {r}");
                                    }
                                }
                                // single forms
                                let one = m.raw_out(refs[0], par, "t");
                                assert_eq!(&one[..], &out[..rows]);
                            } else {
                                let mut out = vec![0i64; refs.len() * rows];
                                m.run(&refs, par, extra, narrow_i64, &mut out);
                                for (k, w) in want.iter().enumerate() {
                                    for r in 0..rows {
                                        assert_eq!(out[k * rows + r] as i128, w[r].unwrap());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // 4 shapes x 4 magnitudes x 2 roundings x 2 output widths, a few skipped for range
        assert!(checked >= 48, "only {checked} cases in range");
    }
}

