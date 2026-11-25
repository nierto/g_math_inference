//! Row-scaled TQ1.9 ("TQ1.9-R"): per-row quantization scales.
//!
//! Motivation: plain TQ19 quantizes every
//! tensor with ONE global step (1/SCALE ≈ 5.1e-5, zero-floor 2.5e-5).
//! Some BF16-trained models put substantial weight mass below that
//! floor; the measured consequence is a 0.207%/layer router top-2 flip
//! rate that compounds to ~6.4% of tokens executing a wrong expert.
//! A per-row scale adapts the step to each row's own max: measured on
//! a large MoE model, matvec output error drops ~20× and wrong-expert tokens drop
//! 6.4% → 0.22%, at unchanged 2 bytes/weight plus one i128
//! multiply-shift per output element.
//!
//! Semantics: `row_value = raw × s_row`, with `raw = round(w / s_row)`,
//! `s_row = max|w_row| / MAX_RAW`. Scales are stored RELATIVE to the
//! global step as unsigned Q32.32: `s_rel = s_row × SCALE`, so
//! `matvec_out = tq19_dot(raw_row, x) × s_rel`: the existing (SIMD)
//! `tq19_dot` is reused verbatim and the scale is applied to its result
//! with an i128 multiply and >>32 arithmetic shift. Fully deterministic;
//! truncation error ≤ ~1.5 storage units per output element.
//!
//! Words: i32 and i64 activations on native i128 scale arithmetic, i128
//! activations (Q64.64) at 512 bits.


use super::ops;
use super::MAX_RAW;
use g_math::fixed_point::{Compute, Fixed, FixedNum, WideInt};
use super::ops::{TqWord, WideOutput};
use super::exact::Rounding;

use rayon::prelude::*;

/// TQ1.9 matrix with one quantization scale per row.
#[derive(Clone)]
pub struct RowScaledTQ19 {
    rows: usize,
    cols: usize,
    /// Row-major quantized weights: `raw = round(w / s_row)`, |raw| ≤ MAX_RAW.
    data: Vec<i16>,
    /// Per-row relative scale `s_row × SCALE` in unsigned Q32.32.
    scales_q32: Vec<u64>,
    /// Output rounding (see [`Rounding`]).
    rounding: Rounding,
    /// The largest `|raw|` (bounds the batched kernel's 32-bit lanes).
    max_abs: u32,
}

impl RowScaledTQ19 {
    /// Construct from parts. `data.len()` must be `rows × cols`,
    /// `scales_q32.len()` must be `rows`.
    pub fn from_parts(rows: usize, cols: usize, data: Vec<i16>, scales_q32: Vec<u64>) -> Self {
        assert_eq!(data.len(), rows * cols, "RowScaledTQ19: data length mismatch");
        assert_eq!(scales_q32.len(), rows, "RowScaledTQ19: scales length mismatch");
        debug_assert!(data.iter().all(|&w| (w as i32).abs() <= MAX_RAW as i32));
        let max_abs = data.iter().map(|w| w.unsigned_abs() as u32).max().unwrap_or(0);
        Self { rows, cols, data, scales_q32, rounding: Rounding::Default, max_abs }
    }

    /// The same matrix rounding its outputs by `rounding`. `Nearest` rounds
    /// the exact value `sum(raw * x) * s / (SCALE * 2^32)` once; `Default`
    /// keeps the released two roundings (the dot, then the row scale).
    pub fn with_rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    /// The rounding mode applied to every output.
    pub fn rounding(&self) -> Rounding { self.rounding }

    /// One output row at storage under the matrix's rounding.
    #[inline(always)]
    fn row_out<S: WideOutput, const F: u32>(&self, row: &[i16], x: &[Fixed<S, F>], s_q32: u64) -> Fixed<S, F>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        self.out_from_acc::<S, F>(ops::tq19_dot_exact_sum(row, x), s_q32)
    }

    /// One output row at 2F fraction bits under the matrix's rounding.
    #[inline(always)]
    fn row_out_q2f<S: WideOutput, const F: u32>(&self, row: &[i16], x: &[Fixed<S, F>], s_q32: u64) -> Compute<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        self.q2f_from_acc::<S, F>(ops::tq19_dot_exact_sum(row, x), s_q32)
    }

    /// An output at storage from the row's exact sum `sum raw * x`.
    #[inline(always)]
    fn out_from_acc<S: WideOutput, const F: u32>(&self, acc: Compute<Fixed<S, F>>, s_q32: u64) -> Fixed<S, F>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        match self.rounding {
            Rounding::Default => Self::scale_row(ops::tq19_from_acc::<S, F>(acc), s_q32),
            Rounding::Nearest => {
                let v = S::rowscaled_exact(acc, s_q32, 0, Rounding::Nearest);
                Fixed::from_raw(S::narrow(v).unwrap_or_else(|| panic!("RowScaledTQ19: scaled output exceeds storage range")))
            }
        }
    }

    /// An output at 2F fraction bits from the row's exact sum.
    #[inline(always)]
    fn q2f_from_acc<S: WideOutput, const F: u32>(&self, acc: Compute<Fixed<S, F>>, s_q32: u64) -> Compute<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        match self.rounding {
            Rounding::Default => Self::scale_row_q2f::<S, F>(ops::tq19_q2f_from_acc::<S, F>(acc), s_q32),
            Rounding::Nearest => S::rowscaled_exact(acc, s_q32, F, Rounding::Nearest),
        }
    }

    /// The exact row sums of a batch of 32-bit activations by the AVX2 kernel
    /// that loads each 16-column weight chunk once for up to four activations
    /// and multiplies 16-bit pairs (the activations as 11-bit limbs), handed
    /// to `sink(b, r, sum)` (each pair once). `false` (nothing called) for
    /// other words, without AVX2, or past 2^16 columns.
    fn batch_sums_i32<S: TqWord, const F: u32>(&self, batch: &[&[Fixed<S, F>]], sink: &(dyn Fn(usize, usize, i64) + Sync)) -> bool
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        #[cfg(target_arch = "x86_64")]
        {
            use super::simd::LIMB_BITS;
            // past 2^16 columns an i64 row sum could overflow: the exact per-row path
            if batch.is_empty() || self.rows == 0 || self.cols > 1 << 16 || !crate::tq5::avx2_available() {
                return false;
            }
            let raws: Option<Vec<&[i32]>> = batch.iter().map(|x| S::as_i32_slice(Fixed::raw_slice(x))).collect();
            let Some(raws) = raws else { return false };
            let cols = self.cols;
            let steps = cols / 16;
            let head = 16 * steps;
            // the fewest limbs (one count for the batch) that hold every activation:
            // lower limbs in [0, 2^11), the top one in [-2^11, 2^11]
            let lo = raws.iter().flat_map(|a| a.iter()).copied().min().unwrap_or(0) as i64;
            let hi = raws.iter().flat_map(|a| a.iter()).copied().max().unwrap_or(0) as i64;
            let l = (1..=3usize)
                .find(|&l| {
                    let k = LIMB_BITS * (l as u32 - 1);
                    (lo >> k) >= -(1 << LIMB_BITS) && (hi >> k) <= 1 << LIMB_BITS
                })
                .expect("three 11-bit limbs hold every i32");
            let limbs: Vec<Vec<i16>> = raws
                .iter()
                .map(|a| {
                    let mut v = vec![0i16; l * head];
                    for (c, &x) in a[..head].iter().enumerate() {
                        let mut x = x as i64;
                        for j in 0..l - 1 {
                            v[j * head + c] = (x & ((1 << LIMB_BITS) - 1)) as i16;
                            x >>= LIMB_BITS;
                        }
                        v[(l - 1) * head + c] = x as i16;
                    }
                    v
                })
                .collect();
            // a lane gains at most 2 * max|w| * 2^11 per step: move it to 64 bits before 2^31
            let pair = 2 * self.max_abs as u64 * (1u64 << LIMB_BITS);
            let period = if pair == 0 { usize::MAX } else { (((1u64 << 31) - 1) / pair) as usize };
            let min_rows = (crate::tq5::MIN_TASK_BYTES / (2 * cols).max(1)).max(1);
            (0..self.rows).into_par_iter().with_min_len(min_rows).for_each(|r| {
                let row = &self.data[r * cols..(r + 1) * cols];
                let mut out = [[0i64; 4]; 4];
                for (c, chunk) in limbs.chunks(4).enumerate() {
                    let mut ptrs = [[std::ptr::null::<i16>(); 4]; 4];
                    for (i, v) in chunk.iter().enumerate() {
                        for j in 0..l {
                            ptrs[i][j] = v[j * head..].as_ptr();
                        }
                    }
                    // SAFETY: AVX2 checked; row and every limb slice hold `head` words;
                    // `period` keeps every 32-bit lane in range
                    unsafe { super::simd::limb_tile_dyn(row.as_ptr(), &ptrs[..chunk.len()], l, steps, period, &mut out) };
                    for (i, sums) in out[..chunk.len()].iter().enumerate() {
                        let b = 4 * c + i;
                        // |sum| <= 2^16 columns * 2^15 * 2^31: an i64
                        let v: i64 = (0..l).map(|j| sums[j] << (LIMB_BITS * j as u32)).sum();
                        let tail: i64 = row[head..].iter().zip(&raws[b][head..]).map(|(&w, &x)| w as i64 * x as i64).sum();
                        sink(b, r, v + tail);
                    }
                }
            });
            return true;
        }
        #[allow(unreachable_code)]
        {
            let _ = (batch, sink);
            false
        }
    }

    /// The outputs of a subset of rows: `out[i]` is row `rows[i]` of
    /// [`Self::matvec`], the same value (exact logits for a few candidates).
    ///
    /// # Panics
    /// Panics on an activation length mismatch or a row index out of range.
    pub fn matvec_rows<S: WideOutput, const F: u32>(&self, rows: &[usize], activations: &[Fixed<S, F>]) -> Vec<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_rows: activation length mismatch");
        rows.iter()
            .map(|&r| {
                assert!(r < self.rows, "RowScaledTQ19::matvec_rows: row {r} out of range");
                self.row_out(&self.data[r * self.cols..(r + 1) * self.cols], activations, self.scales_q32[r])
            })
            .collect()
    }

    /// The wide outputs of a subset of rows: `out[i]` is row `rows[i]` of
    /// [`Self::matvec_q2f`], the same value.
    ///
    /// # Panics
    /// As [`Self::matvec_rows`].
    pub fn matvec_rows_q2f<S: WideOutput, const F: u32>(&self, rows: &[usize], activations: &[Fixed<S, F>]) -> Vec<Compute<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_rows_q2f: activation length mismatch");
        rows.iter()
            .map(|&r| {
                assert!(r < self.rows, "RowScaledTQ19::matvec_rows_q2f: row {r} out of range");
                self.row_out_q2f(&self.data[r * self.cols..(r + 1) * self.cols], activations, self.scales_q32[r])
            })
            .collect()
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn data(&self) -> &[i16] { &self.data }
    pub fn scales_q32(&self) -> &[u64] { &self.scales_q32 }

    /// Bytes of weight + scale storage (2 B/weight + 8 B/row).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2 + self.scales_q32.len() * 8
    }

    #[inline(always)]
    fn scale_row<S: WideOutput, const F: u32>(dot: Fixed<S, F>, s_q32: u64) -> Fixed<S, F>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        Fixed::from_raw(S::rowscaled_scale(dot.raw(), s_q32))
    }

    /// Row-scaled matvec: `out[r] = tq19_dot(row_r, x) × s_rel[r]`.
    pub fn matvec<S: WideOutput, const F: u32>(&self, activations: &[Fixed<S, F>]) -> Vec<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec: activation length mismatch");
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                self.row_out(row, activations, self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel matvec.
    pub fn matvec_par<S: WideOutput, const F: u32>(&self, activations: &[Fixed<S, F>]) -> Vec<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_par: activation length mismatch");
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                self.row_out(row, activations, self.scales_q32[r])
            })
            .collect()
    }

    /// Wide scale application: `floor(wide_dot · s_rel / 2^32)` at full
    /// precision. Same floor rule as [`Self::scale_row`], applied to the
    /// wide dot instead of the narrowed one.
    #[inline(always)]
    fn scale_row_q2f<S: WideOutput, const F: u32>(wide_dot: Compute<Fixed<S, F>>, s_q32: u64) -> Compute<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        S::rowscaled_scale_q2f(wide_dot, s_q32)
    }

    /// Wide-output row-scaled matvec: each row at 2·FRAC_BITS precision.
    ///
    /// `floor(tq19_dot_q2f(row, x) · s_rel / 2^32)`: the wide dot keeps
    /// FRAC_BITS extra fractional bits *through* the scale multiply, so this
    /// is strictly more precise than scaling the narrowed dot. Consequence
    /// (documented deliberately): narrowing this result can differ from
    /// [`Self::matvec`] by ±1 storage LSB for non-unit scales: the narrow
    /// path scales an already-rounded dot. For `s_rel = 1.0` (`1u64 << 32`)
    /// the two agree bit-for-bit.
    pub fn matvec_q2f<S: WideOutput, const F: u32>(&self, activations: &[Fixed<S, F>]) -> Vec<Compute<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_q2f: activation length mismatch");
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                self.row_out_q2f(row, activations, self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel wide-output matvec. See [`Self::matvec_q2f`].
    pub fn matvec_q2f_par<S: WideOutput, const F: u32>(&self, activations: &[Fixed<S, F>]) -> Vec<Compute<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_q2f_par: activation length mismatch");
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                self.row_out_q2f(row, activations, self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel wide-output batch matvec. See [`Self::matvec_q2f`].
    pub fn matvec_q2f_batch_par<S: WideOutput, const F: u32>(&self, batch: &[&[Fixed<S, F>]]) -> Vec<Vec<Compute<Fixed<S, F>>>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "RowScaledTQ19::matvec_q2f_batch_par: activation[{i}] length mismatch");
        }
        let mut flat = vec![<Compute<Fixed<S, F>> as WideInt>::zero(); batch.len() * self.rows];
        {
            let sink = ops::BatchOut::new(&mut flat, self.rows, batch.len(), "RowScaledTQ19::matvec_q2f_batch_par");
            let scales = &self.scales_q32;
            // SAFETY: each (vector, row) pair is handed over once
            let fast = self.batch_sums_i32(batch, &|b, r, acc| unsafe {
                sink.write(b, r, self.q2f_from_acc::<S, F>(<Compute<Fixed<S, F>> as WideInt>::from_i64(acc), scales[r]))
            });
            if fast {
                return if self.rows == 0 { vec![Vec::new(); batch.len()] } else { flat.chunks_exact(self.rows).map(<[_]>::to_vec).collect() };
            }
        }
        let per_row: Vec<Vec<Compute<Fixed<S, F>>>> = (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                let s = self.scales_q32[r];
                batch
                    .iter()
                    .map(|x| self.row_out_q2f(row, x, s))
                    .collect()
            })
            .collect();
        (0..batch.len())
            .map(|b| per_row.iter().map(|row| row[b]).collect())
            .collect()
    }

    /// [`matvec_batch_par`](Self::matvec_batch_par) writing into a
    /// caller-provided buffer, flat and batch-major: `out[b * rows + r]` is
    /// row `r` of the result for `batch[b]`. The same values; no result
    /// vector is allocated.
    ///
    /// # Panics
    /// Panics on an activation length mismatch or if
    /// `out.len() != batch.len() * rows`.
    pub fn matvec_batch_par_into<S: WideOutput, const F: u32>(&self, batch: &[&[Fixed<S, F>]], out: &mut [Fixed<S, F>])
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "RowScaledTQ19::matvec_batch_par_into: activation[{i}] length mismatch");
        }
        let sink = ops::BatchOut::new(out, self.rows, batch.len(), "RowScaledTQ19::matvec_batch_par_into");
        let scales = &self.scales_q32;
        // SAFETY: each (vector, row) pair is handed over once
        if self.batch_sums_i32(batch, &|b, r, acc| unsafe {
            sink.write(b, r, self.out_from_acc::<S, F>(<Compute<Fixed<S, F>> as WideInt>::from_i64(acc), scales[r]))
        }) {
            return;
        }
        (0..self.rows).into_par_iter().for_each(|r| {
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            let s = self.scales_q32[r];
            for (b, x) in batch.iter().enumerate() {
                // SAFETY: this task is the only one writing row `r`.
                unsafe { sink.write(b, r, self.row_out(row, x, s)) };
            }
        });
    }

    /// Row-parallel batch matvec (row weights stay in cache across the batch).
    pub fn matvec_batch_par<S: WideOutput, const F: u32>(&self, batch: &[&[Fixed<S, F>]]) -> Vec<Vec<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
        {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "RowScaledTQ19::matvec_batch_par: activation[{i}] length mismatch");
        }
        if self.rows > 0 && !batch.is_empty() && batch.iter().all(|x| S::as_i32_slice(Fixed::raw_slice(x)).is_some()) {
            let mut flat = vec![Fixed::from_raw(<S as WideInt>::zero()); batch.len() * self.rows];
            self.matvec_batch_par_into(batch, &mut flat);
            return flat.chunks_exact(self.rows).map(<[_]>::to_vec).collect();
        }
        let per_row: Vec<Vec<Fixed<S, F>>> = (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                let s = self.scales_q32[r];
                batch
                    .iter()
                    .map(|x| self.row_out(row, x, s))
                    .collect()
            })
            .collect();
        // Transpose rows×batch → batch×rows.
        (0..batch.len())
            .map(|b| per_row.iter().map(|row| row[b]).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    macro_rules! suite {
        ($m:ident, $S:ty, $F:literal, $W:ident) => {
            mod $m {
                #![allow(dead_code, unused_imports, unreachable_code)]
                use super::super::*;
                type FixedPoint = g_math::fixed_point::Fixed<$S, $F>;
                type BinaryStorage = $S;
                type ComputeStorage = <$S as g_math::fixed_point::Storage>::Compute;
                const FRAC_BITS: u32 = $F;
                const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
                use super::super::*;
                #[allow(unused_imports)]
                use crate::{TQ19Matrix, SCALE};

                /// A row-scaled matrix whose every scale equals the global step must
                /// reproduce plain TQ19 matvec bit-for-bit (s_rel = 1.0 in Q32.32).
                #[test]
                fn unit_scale_matches_plain_tq19() {
                    let rows = 7;
                    let cols = 64;
                    let data: Vec<i16> = (0..rows * cols)
                        .map(|i| ((i as i64 * 2654435761 % 59049) - 29524) as i16)
                        .collect();
                    let tq = TQ19Matrix::new(rows, cols, data.clone());
                    let rs = RowScaledTQ19::from_parts(rows, cols, data, vec![1u64 << 32; rows]);
                    let x: Vec<FixedPoint> = (0..cols)
                        .map(|i| FixedPoint::from_raw(((i as i64 * 40503 % 2049) - 1024) as BinaryStorage))
                        .collect();
                    assert_eq!(tq.matvec(&x), rs.matvec(&x));
                }

                fn narrow_q2f(v: ComputeStorage) -> BinaryStorage {
                    (v / (1 << FRAC_BITS)) as BinaryStorage
                }

                /// Wide-output contract for the row-scaled form: with unit scales the
                /// narrow relationship is exact; with general scales the wide path keeps
                /// FRAC_BITS extra bits through the scale multiply, so narrow(q2f) may
                /// differ from matvec by at most 1 storage LSB (documented behavior).
                /// The wide value itself is pinned against an independent i128 oracle.
                #[test]
                fn q2f_unit_scale_exact_general_within_one_lsb_and_oracle() {
                    let rows = 5;
                    let cols = 97;
                    let data: Vec<i16> = (0..rows * cols)
                        .map(|i| ((i as i64 * 48271 % 59049) - 29524) as i16)
                        .collect();
                    let x: Vec<FixedPoint> = (0..cols)
                        .map(|i| FixedPoint::from_raw(((i as i64 * 69621 % 4001) - 2000) as BinaryStorage))
                        .collect();

                    // Unit scales: narrow(q2f) == matvec bit-for-bit.
                    let unit = RowScaledTQ19::from_parts(rows, cols, data.clone(), vec![1u64 << 32; rows]);
                    let narrow = unit.matvec(&x);
                    let wide = unit.matvec_q2f(&x);
                    for r in 0..rows {
                        assert_eq!(narrow_q2f(wide[r]), narrow[r].raw(), "unit-scale row {r}");
                    }

                    // General scales: oracle equality + ≤1 LSB narrow relationship.
                    let scales: Vec<u64> = vec![
                        1u64 << 31,
                        1u64 << 32,
                        (1u64 << 32) + (1u64 << 30) + 12345,
                        1u64 << 20,
                        (1u64 << 32) + 1,
                    ];
                    let rs = RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone());
                    let narrow = rs.matvec(&x);
                    let wide = rs.matvec_q2f(&x);
                    let f = FRAC_BITS;
                    for r in 0..rows {
                        // Independent oracle: exact accumulator → one truncating /SCALE
                        // at 2F precision → floor scale multiply, all in plain i128.
                        let mut acc: i128 = 0;
                        for c in 0..cols {
                            acc += data[r * cols + c] as i128 * x[c].raw() as i128;
                        }
                        let wide_dot = (acc << f) / SCALE as i128;
                        let expected = (wide_dot * scales[r] as i128) >> 32;
                        assert_eq!(wide[r] as i128, expected, "row {r} diverged from i128 oracle");
                        // Narrow relationship: within 1 storage LSB of the narrow path.
                        let diff = (narrow_q2f(wide[r]) as i128 - narrow[r].raw() as i128).abs();
                        assert!(diff <= 1, "row {r}: narrow(q2f) off by {diff} LSB");
                    }
                    assert_eq!(wide, rs.matvec_q2f_par(&x));
                    let batch: Vec<&[FixedPoint]> = vec![&x];
                    assert_eq!(rs.matvec_q2f_batch_par(&batch)[0], wide);
                }

                /// Independent oracle: recompute the matvec with plain i128 arithmetic
                /// (no tq19_dot, no SIMD): Σ raw·x accumulated at i128, then the same
                /// Q32.32 multiply-shift. Pins both the tq19_dot reuse and the scale
                /// application against a path that shares no code with the kernel.
                #[test]
                fn rowscaled_matvec_matches_i128_oracle() {
                    let rows = 5;
                    let cols = 97; // not divisible by SIMD lane widths
                    let data: Vec<i16> = (0..rows * cols)
                        .map(|i| ((i as i64 * 48271 % 59049) - 29524) as i16)
                        .collect();
                    // Scales around 1.0: 0.5×, 1×, ~1.3×, tiny, and exactly 1 LSB above 1.
                    let scales: Vec<u64> = vec![
                        1u64 << 31,
                        1u64 << 32,
                        (1u64 << 32) + (1u64 << 30) + 12345,
                        1u64 << 20,
                        (1u64 << 32) + 1,
                    ];
                    let x: Vec<FixedPoint> = (0..cols)
                        .map(|i| FixedPoint::from_raw(((i as i64 * 69621 % 4001) - 2000) as BinaryStorage))
                        .collect();
                    let rs = RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone());
                    let got = rs.matvec(&x);
                    for r in 0..rows {
                        let mut acc: i128 = 0;
                        for c in 0..cols {
                            acc += data[r * cols + c] as i128 * x[c].raw() as i128;
                        }
                        // tq19_dot spec: truncate-toward-zero division by the global
                        // SCALE, narrow, THEN the Q32.32 row-scale multiply-shift.
                        let dot = acc / SCALE as i128;
                        let expected = ((dot * scales[r] as i128) >> 32) as BinaryStorage;
                        assert_eq!(got[r], FixedPoint::from_raw(expected), "row {} diverged from i128 oracle", r);
                    }
                    // matvec_par must agree with sequential.
                    assert_eq!(got, rs.matvec_par(&x));
                }

                /// Scaling all weights up and the row scale down must approximate the
                /// same values with far finer granularity (small-weight fidelity).
                #[test]
                fn halved_scale_doubles_resolution() {
                    let cols = 32;
                    // value 3 raw units at global step, stored at s_rel = 0.5 with raw ×2
                    let data: Vec<i16> = vec![6; cols];
                    let rs = RowScaledTQ19::from_parts(1, cols, data, vec![1u64 << 31]);
                    let x: Vec<FixedPoint> = vec![FixedPoint::from_raw(SCALE as BinaryStorage); cols];
                    // per element: raw 6 × x SCALE / SCALE × 0.5 = 3 → sum = 96
                    assert_eq!(rs.matvec(&x)[0].raw(), 3 * cols as BinaryStorage);
                }
            }
        };
    }
    // the row-scaled forms exist on the 32- and 64-bit words only
    suite!(q22_10, i32, 10, w32);
    suite!(q16_16, i32, 16, w32);
    suite!(q32_32, i64, 32, w64);
}
