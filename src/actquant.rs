//! Activation quantisation for block-scaled checkpoints, evaluated exactly:
//! the rule such checkpoints declare for their expert inputs, with every
//! step done in exact integer / rational arithmetic instead of binary32
//! (the runtimes evaluate it in binary32 and disagree with each other on
//! the operation order; this is the rule itself, no float anywhere).
//!
//! NVFP4, per row of activations (`quantize_nvfp4`):
//! - `amax = max |x|`; the row factor is `amax / 1536` (`1536 = 6 * 256`).
//! - Per 16-block with `bm = max |x|` in the block, two candidate E4M3
//!   scales, nearest-even and saturating at 448: `s6 = bm * 256 / amax`
//!   (the block maximum maps to 6) and `s4 = bm * 384 / amax` (maps to 4).
//! - For each candidate, every element `x / (s * amax / 1536)` rounds to
//!   E2M1 (nearest, ties to the even code, saturating at 6); the candidate
//!   is scored by the exact sum of `|dequantised - x|`; 4 is chosen only if
//!   its sum is strictly smaller (a tie keeps 6). A zero scale gives zero
//!   codes (the value is zero either way).
//! - Value of element `i`: `E2M1(code) * E4M3(scale) * amax / 1536`.
//!
//! MXFP8, per 32-block (`quantize_mxfp8`): the shared exponent is the
//! smallest `e` with `2^e >= amax_block / 448` (no element exceeds 448),
//! clamped to the UE8M0 range; elements `x / 2^e` round to E4M3 (nearest
//! even, saturating at 448). A zero block has exponent field 0.
//!
//! The matvecs on quantised activations (`NvFp4Matrix::matvec_quantized`,
//! `MxFp8Matrix::matvec_quantized`) form the exact sum over both quantised
//! operands and round it once by the matrix's [`Rounding`](crate::Rounding).
//! They are exact reference paths (scalar): for evaluating the rule's effect
//! on a model, not yet the hot path.

use crate::bits::decompose_e4m3;
use crate::exact::round_div_i512;
use crate::{MxFp8Matrix, NvFp4Matrix};
use g_math::fixed_point::{Fixed, FixedNum, I512};

/// E4M3 (OCP "fn") value of a pattern times 2^9: an exact integer.
fn e4m3_x512(bits: u8) -> i128 {
    let (neg, m, e) = decompose_e4m3(bits).expect("E4M3 NaN");
    let v = (m as i128) << (e + 9);
    if neg { -v } else { v }
}

/// `floor(n / d)` and whether the remainder is above, at or below half
/// (n, d > 0): the pieces of a nearest-even rounding.
fn floor_and_half(n: i128, d: i128) -> (i128, std::cmp::Ordering) {
    let (q, r) = (n / d, n % d);
    (q, (2 * r).cmp(&d))
}

/// Nearest-even E4M3 pattern of the positive rational `n / d`, saturating
/// at 448 (0x7E). `n >= 0`, `d > 0`.
pub(crate) fn e4m3_rne(n: i128, d: i128) -> u8 {
    use std::cmp::Ordering::*;
    if n == 0 {
        return 0;
    }
    // compare with 448: n / d >= 448 saturates (448 is the largest finite)
    if n >= 448 * d {
        return 0x7E;
    }
    // subnormal grid m * 2^-9 below 2^-6 (= 8 * 2^-9)
    if n * 512 < 8 * d {
        let (q, h) = floor_and_half(n * 512, d);
        let m = match h {
            Greater => q + 1,
            Equal => q + (q & 1),
            Less => q,
        };
        return m as u8; // m <= 8; m = 8 is the normal 2^-6 (pattern 0x08)
    }
    // normal: find E (1..15) with 8 * 2^(E-10) <= v < 16 * 2^(E-10)
    let mut e = 1i32;
    while e < 15 && scaled_cmp(n, d, 16, e - 10) != Less {
        e += 1;
    }
    // q = v * 2^(10 - E) in [8, 16): round to an integer, ties to even
    let (num, den) = if e <= 10 { (n << (10 - e), d) } else { (n, d << (e - 10)) };
    let (q, h) = floor_and_half(num, den);
    let mut m = match h {
        Greater => q + 1,
        Equal => q + (q & 1),
        Less => q,
    };
    let mut e = e;
    if m == 16 {
        m = 8;
        e += 1;
    }
    let code = ((e as i128) << 3) | (m - 8);
    if code > 0x7E { 0x7E } else { code as u8 }
}

/// Sign of `n / d - k * 2^p` (exact).
fn scaled_cmp(n: i128, d: i128, k: i128, p: i32) -> std::cmp::Ordering {
    if p >= 0 { n.cmp(&((k * d) << p)) } else { (n << (-p)).cmp(&(k * d)) }
}

/// The E2M1 code (twice its value, signed: `E2M1_X2` form) nearest to the
/// rational `n / d` (`d > 0`), ties to the even code, saturating at 6.
fn e2m1_x2_rne(n: i128, d: i128) -> i8 {
    // thresholds in quarters of the value: 0.25 0.75 1.25 1.75 2.5 3.5 5,
    // the codes below/at them: tie -> even code (0, 1.0, 2.0, 4.0 sides)
    let a = n.unsigned_abs() as i128 * 4; // 4 |v| = a / d
    let t = |q: i128| a.cmp(&(q * d));
    use std::cmp::Ordering::*;
    let x2: i8 = if t(1) != Greater {
        0
    } else if t(3) == Less {
        1
    } else if t(5) != Greater {
        2
    } else if t(7) == Less {
        3
    } else if t(10) != Greater {
        4
    } else if t(14) == Less {
        6
    } else if t(20) != Greater {
        8
    } else {
        12
    };
    if n < 0 { -x2 } else { x2 }
}

/// Activations quantised to NVFP4 by the exact rule (see the module notes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nvfp4Activation {
    /// Twice the E2M1 value per element (`E2M1_X2` form, in [-12, 12]).
    pub codes_x2: Vec<i8>,
    /// E4M3 scale pattern per 16 elements.
    pub scales: Vec<u8>,
    /// `max |x|` as the activations' raw word (the row factor is `amax / 1536`).
    pub amax_raw: i64,
}

/// Activations quantised to MXFP8 by the exact rule (see the module notes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mxfp8Activation {
    /// E4M3 pattern per element.
    pub elems: Vec<u8>,
    /// UE8M0 exponent field per 32 elements (`2^(field - 127)`).
    pub scales: Vec<u8>,
    /// The activations' fraction bits (the elements carry `x * 2^-F`).
    pub frac: u32,
}

/// Quantise a row of activations to NVFP4 (four-over-six, exact).
///
/// # Panics
/// Panics if the length is not a multiple of 16.
pub fn quantize_nvfp4<const F: u32>(x: &[Fixed<i32, F>]) -> Nvfp4Activation
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    assert!(x.len() % 16 == 0, "quantize_nvfp4: length must be a multiple of 16");
    let xr: Vec<i128> = x.iter().map(|v| v.raw() as i128).collect();
    let amax = xr.iter().map(|v| v.abs()).max().unwrap_or(0);
    let mut codes_x2 = vec![0i8; x.len()];
    let mut scales = vec![0u8; x.len() / 16];
    if amax == 0 {
        return Nvfp4Activation { codes_x2, scales, amax_raw: 0 };
    }
    for (b, blk) in xr.chunks_exact(16).enumerate() {
        let bm = blk.iter().map(|v| v.abs()).max().unwrap();
        let mut best: Option<(i128, u8, [i8; 16])> = None;
        for target in [256i128, 384] {
            // scale = bm * (1536 / target_max) / amax, target_max 6 or 4
            let s = e4m3_rne(bm * target, amax);
            let s512 = e4m3_x512(s);
            let mut q = [0i8; 16];
            let mut err = 0i128;
            for (i, &xi) in blk.iter().enumerate() {
                // x / (s * amax / 1536) = xi * 1536 * 512 / (s512 * amax)
                if s512 != 0 {
                    q[i] = e2m1_x2_rne(xi * 1536 * 512, s512 * amax);
                }
                // |x2/2 * s512/512 * amax/1536 - xi| * (2 * 512 * 1536)
                err += (q[i] as i128 * s512 * amax - xi * 2 * 512 * 1536).abs();
            }
            // 6 first: the 4 candidate wins only if strictly smaller
            if best.as_ref().map_or(true, |b| err < b.0) {
                best = Some((err, s, q));
            }
        }
        let (_, s, q) = best.unwrap();
        scales[b] = s;
        codes_x2[b * 16..b * 16 + 16].copy_from_slice(&q);
    }
    Nvfp4Activation { codes_x2, scales, amax_raw: amax as i64 }
}

/// Quantise a row of activations to MXFP8 (ceil exponent, exact).
///
/// # Panics
/// Panics if the length is not a multiple of 32.
pub fn quantize_mxfp8<const F: u32>(x: &[Fixed<i32, F>]) -> Mxfp8Activation
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    assert!(x.len() % 32 == 0, "quantize_mxfp8: length must be a multiple of 32");
    let mut elems = vec![0u8; x.len()];
    let mut scales = vec![0u8; x.len() / 32];
    for (b, blk) in x.chunks_exact(32).enumerate() {
        let raw: Vec<i128> = blk.iter().map(|v| v.raw() as i128).collect();
        let amax = raw.iter().map(|v| v.abs()).max().unwrap();
        if amax == 0 {
            continue; // field 0, all elements 0
        }
        // value = raw * 2^-F; smallest e with 2^e >= amax 2^-F / 448
        // start just below the answer: 448 * 2^(L - 10) < 2^(L - 1) <= amax (L = bit length)
        let len = 128 - amax.leading_zeros() as i32;
        let mut e: i32 = (len - F as i32 - 10).max(-127);
        while e < 127 && scaled_cmp(amax, 448, 1, e + F as i32) == std::cmp::Ordering::Greater {
            e += 1;
        }
        scales[b] = (e + 127) as u8;
        for (i, &v) in raw.iter().enumerate() {
            // x / 2^e = v * 2^(-F - e)
            let k = -(F as i32) - e;
            let (n, d) = if k >= 0 { (v.abs() << k, 1) } else { (v.abs(), 1i128 << (-k)) };
            let code = e4m3_rne(n, d);
            elems[b * 32 + i] = if v < 0 && code != 0 { code | 0x80 } else { code };
        }
    }
    Mxfp8Activation { elems, scales, frac: F }
}

impl NvFp4Matrix {
    /// `W x` on NVFP4-quantised activations ([`quantize_nvfp4`]): the exact
    /// sum over both quantised operands, rounded once to the activations'
    /// format by the matrix's rounding mode. Scalar (a reference path).
    ///
    /// # Panics
    /// Panics on a length mismatch or a result beyond the storage range.
    pub fn matvec_quantized<const F: u32>(&self, a: &Nvfp4Activation) -> Vec<Fixed<i32, F>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        assert_eq!(a.codes_x2.len(), self.cols(), "matvec_quantized: activation length mismatch");
        let sx: Vec<i128> = a.scales.iter().map(|&s| e4m3_x512(s)).collect();
        (0..self.rows())
            .map(|r| {
                // weight = nums[c] * 2^e / den; activation = x2/2 * s512/512 * amax/1536
                let (nums, e, den) = self.dequant_row_exact(r);
                let mut s = I512::from(0i128);
                for b in 0..self.cols() / 16 {
                    let mut blk = 0i128;
                    for j in b * 16..b * 16 + 16 {
                        blk += nums[j] as i128 * a.codes_x2[j] as i128;
                    }
                    s = s + I512::from(blk) * I512::from(sx[b]);
                }
                // y raw = s * 2^e / den * amax_raw / (2 * 512 * 1536)
                let n = s * I512::from(a.amax_raw as i128);
                let d = I512::from(den as i128) * I512::from(2i128 * 512 * 1536);
                let (n, d) = if e >= 0 { (n << e as usize, d) } else { (n, d << (-e) as usize) };
                let v = round_div_i512(n, d, self.rounding());
                Fixed::from_raw(crate::exact::narrow_wide::<i32>(v, "NvFp4Matrix::matvec_quantized"))
            })
            .collect()
    }
}

impl MxFp8Matrix {
    /// `W x` on MXFP8-quantised activations ([`quantize_mxfp8`]): the exact
    /// sum over both quantised operands, rounded once by the matrix's
    /// rounding mode. Scalar (a reference path).
    ///
    /// # Panics
    /// Panics on a length or format mismatch or a result beyond the storage range.
    pub fn matvec_quantized<const F: u32>(&self, a: &Mxfp8Activation) -> Vec<Fixed<i32, F>>
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        assert_eq!(a.elems.len(), self.cols(), "matvec_quantized: activation length mismatch");
        assert_eq!(a.frac, F, "matvec_quantized: activations quantised at another format");
        (0..self.rows())
            .map(|r| {
                // weight = w * 2^ew; activation = E4M3(x) * 2^(sx - 127), as raw * 2^F
                let wts = self.dequant_row_exact(r);
                let terms: Vec<(i128, i32)> = (0..self.cols())
                    .map(|c| {
                        let (w, ew) = wts[c];
                        let xv = e4m3_x512(a.elems[c]); // E4M3 * 2^9
                        (w as i128 * xv, ew + a.scales[c / 32] as i32 - 127 - 9 + F as i32)
                    })
                    .collect();
                let emin = terms.iter().filter(|t| t.0 != 0).map(|t| t.1).min().unwrap_or(0);
                let s = terms.iter().fold(I512::from(0i128), |acc, &(v, e)| if v == 0 { acc } else { acc + (I512::from(v) << (e - emin) as usize) });
                let v = crate::exact::round_shift_i512(s, emin, self.rounding());
                Fixed::from_raw(crate::exact::narrow_wide::<i32>(v, "MxFp8Matrix::matvec_quantized"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every positive rational on a fine grid rounds to the E4M3 value the
    /// definition gives (nearest, ties to even, saturating at 448).
    #[test]
    fn e4m3_rne_is_nearest_even() {
        let vals: Vec<(u8, i128)> = (0u8..=0x7E).map(|b| (b, e4m3_x512(b))).collect();
        // v = n / 2^12 for n up to 448 * 2^12 + margin, in units of 2^-9 * 2^-3
        for n in (0..(460i128 << 12)).step_by(7) {
            let got = e4m3_rne(n, 1 << 12);
            // reference: nearest by |v - val| in units of 2^-12; ties -> even mantissa
            let v8 = n; // value * 2^12; vals in 2^-9 units -> * 8
            let mut best = (0u8, i128::MAX);
            for &(b, x512) in &vals {
                let dist = (v8 - x512 * 8).abs();
                if dist < best.1 || (dist == best.1 && b % 2 == 0) {
                    best = (b, dist);
                }
            }
            let want = if n >= 448 << 12 { 0x7E } else { best.0 };
            assert_eq!(got, want, "n {n}");
        }
    }

    #[test]
    fn e2m1_rne_thresholds() {
        // value in quarters: 0..28
        let want = [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12];
        for (q, &w) in want.iter().enumerate() {
            assert_eq!(e2m1_x2_rne(q as i128, 4), w, "value {q}/4");
            assert_eq!(e2m1_x2_rne(-(q as i128), 4), -w);
        }
    }
}
