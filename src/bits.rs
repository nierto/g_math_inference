//! Weight bit patterns to fixed point, by integer shifts only.
//!
//! Model weight files carry IEEE binary16 or bfloat16 **bit patterns**. This
//! module decodes those bits into Q-format raw values and TQ1.9 raw values
//! without a float type: a pattern is split into sign, integer mantissa and
//! binary exponent, and every conversion is a shift of that mantissa.
//!
//! - [`to_raw`]: `trunc(v * 2^frac_bits)` toward zero, at any fractional
//!   width (`to_q64_raw` is the Q64.64 form).
//! - [`to_storage_raw`] / [`to_fixed`]: the same at the build's storage
//!   format (realtime and compact profiles).
//! - [`to_tq19_raw`]: `round(v * 3^9)`, half away from zero.
//! - [`WeightBits`]: a row-major matrix of patterns as read from a file, the
//!   input of the quantisers in [`super::quantize`].
//!
//! - [`E2M1_X2`] / [`e2m1_x2`], [`decompose_e4m3`], [`ue8m0_exponent`],
//!   [`decompose_f32`]: the element and scale formats of block-scaled
//!   checkpoints (NVFP4, MXFP8) and their 32-bit factors, as integers; the
//!   kernels over them are in [`super::blockscaled`].
//! - [`f32_to_raw`], [`f32_to_storage_raw`], [`f32_to_fixed`]: binary32
//!   patterns at a fixed-point format by the same truncation as the 16-bit forms.
//!
//! Infinities and NaNs are refused with `InvalidInput`; a value outside the
//! target range is `TierOverflow`.

use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::{Fixed, FixedNum};
use g_math::fixed_point::Storage;

/// Which 16-bit float format a bit pattern is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HalfKind {
    /// IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits.
    Binary16,
    /// bfloat16: 1 sign, 8 exponent, 7 mantissa bits.
    BFloat16,
}

/// `|v| = m * 2^e` with `m` the integer mantissa (implicit bit included) and
/// the sign separately: `(negative, m, e)`. Zero gives `m = 0`; subnormals
/// have no implicit bit. `Err(InvalidInput)` for an infinity or a NaN.
#[inline]
pub fn decompose(bits: u16, kind: HalfKind) -> Result<(bool, u64, i32), OverflowDetected> {
    let sign = bits & 0x8000 != 0;
    match kind {
        HalfKind::Binary16 => {
            let exp = ((bits >> 10) & 0x1F) as i32;
            let mant = (bits & 0x3FF) as u64;
            if exp == 0x1F {
                return Err(OverflowDetected::InvalidInput);
            }
            Ok(if exp == 0 { (sign, mant, -24) } else { (sign, mant | 0x400, exp - 25) })
        }
        HalfKind::BFloat16 => {
            let exp = ((bits >> 7) & 0xFF) as i32;
            let mant = (bits & 0x7F) as u64;
            if exp == 0xFF {
                return Err(OverflowDetected::InvalidInput);
            }
            Ok(if exp == 0 { (sign, mant, -133) } else { (sign, mant | 0x80, exp - 134) })
        }
    }
}

/// `trunc(v * 2^frac_bits)` toward zero as an i128.
///
/// Exact whenever the value has no bits below `2^-frac_bits`.
/// `Err(TierOverflow)` when the result does not fit i128, `Err(InvalidInput)`
/// for a non-finite pattern.
pub fn to_raw(bits: u16, kind: HalfKind, frac_bits: u32) -> Result<i128, OverflowDetected> {
    let (sign, m, e) = decompose(bits, kind)?;
    trunc_raw(sign, m, e, frac_bits)
}

/// `trunc(+-m * 2^(e + frac_bits))` toward zero as an i128, `TierOverflow` past it.
fn trunc_raw(sign: bool, m: u64, e: i32, frac_bits: u32) -> Result<i128, OverflowDetected> {
    if m == 0 {
        return Ok(0);
    }
    let shift = e as i64 + frac_bits as i64;
    let mag: i128 = if shift >= 0 {
        let top = 64 - m.leading_zeros() as i64; // at most 24 bits (binary32)
        if top + shift > 127 {
            return Err(OverflowDetected::TierOverflow);
        }
        (m as i128) << shift
    } else if -shift >= 64 {
        0
    } else {
        (m >> (-shift) as u32) as i128
    };
    Ok(if sign { -mag } else { mag })
}

/// The value at Q64.64: `trunc(v * 2^64)`. Exact for every finite pattern of
/// either format (the smallest subnormal is `2^-133`, truncated to zero below
/// `2^-64`), `Err(TierOverflow)` from `2^63` up.
#[inline]
pub fn to_q64_raw(bits: u16, kind: HalfKind) -> Result<i128, OverflowDetected> {
    to_raw(bits, kind, 64)
}

/// The value at the build's storage format: `trunc(v * 2^FRAC_BITS)` toward
/// zero. `Err(TierOverflow)` when the magnitude exceeds the largest storage
/// value.
/// The value as a raw storage word of `F` fraction bits (truncated toward
/// zero), `TierOverflow` when its magnitude does not fit the word (so
/// -2^(W-1) is refused like +2^(W-1)).
pub fn to_storage_raw<S: Storage, const F: u32>(bits: u16, kind: HalfKind) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let raw = to_raw(bits, kind, F)?;
    match (S::from_i128_checked(raw), raw.checked_neg().and_then(S::from_i128_checked)) {
        (Some(v), Some(_)) => Ok(v),
        _ => Err(OverflowDetected::TierOverflow),
    }
}

/// The value as a `Fixed<S, F>` (truncated toward zero), `TierOverflow` past the word.
#[inline]
pub fn to_fixed<S: Storage, const F: u32>(bits: u16, kind: HalfKind) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Ok(Fixed::from_raw(to_storage_raw::<S, F>(bits, kind)?))
}

/// TQ1.9 raw value: `round(v * 3^9)`, half away from zero, unclamped (the
/// caller clamps to `MAX_RAW` / `MIN_RAW`). A value from `2^21` up, far
/// outside the TQ1.9 range, returns `i64::MAX / 4` with its sign.
pub fn to_tq19_raw(bits: u16, kind: HalfKind) -> Result<i64, OverflowDetected> {
    let (sign, m, e) = decompose(bits, kind)?;
    if m == 0 {
        return Ok(0);
    }
    let n = (m as i64) * super::SCALE as i64;
    let mag: i64 = if e >= 0 {
        if e > 20 { i64::MAX / 4 } else { n << e }
    } else {
        let k = (-e) as u32;
        if k >= 63 { 0 } else { (n + (1i64 << (k - 1))) >> k }
    };
    Ok(if sign { -mag } else { mag })
}

/// A matrix of weight bit patterns as read from a file: the float-free form
/// projections are quantised from and embeddings are decoded from.
#[derive(Clone)]
pub struct WeightBits {
    pub rows: usize,
    pub cols: usize,
    pub kind: HalfKind,
    /// Row-major, one u16 per element (the file's little-endian pair).
    pub bits: Vec<u16>,
}

impl WeightBits {
    /// From the file's little-endian bytes. `Err(InvalidInput)` when `bytes`
    /// holds fewer than `rows * cols` elements; extra bytes are ignored.
    pub fn from_le_bytes(rows: usize, cols: usize, kind: HalfKind, bytes: &[u8]) -> Result<Self, OverflowDetected> {
        let n = rows * cols;
        if bytes.len() < n * 2 {
            return Err(OverflowDetected::InvalidInput);
        }
        let bits = bytes[..n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Self { rows, cols, kind, bits })
    }

    /// Storage raw of element `i`; see [`to_storage_raw`].
        #[inline]
    pub fn storage_raw<S: Storage, const F: u32>(&self, i: usize) -> Result<S, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        to_storage_raw::<S, F>(self.bits[i], self.kind)
    }

    /// TQ1.9 raw of element `i`; see [`to_tq19_raw`].
    #[inline]
    pub fn tq19_raw(&self, i: usize) -> Result<i64, OverflowDetected> {
        to_tq19_raw(self.bits[i], self.kind)
    }

    /// `(negative, m, e)` of element `i`, for exact rational arithmetic.
    #[inline]
    pub fn decompose(&self, i: usize) -> Result<(bool, u64, i32), OverflowDetected> {
        decompose(self.bits[i], self.kind)
    }
}

/// The four-bit E2M1 element (1 sign, 2 exponent, 1 mantissa bit) times two,
/// indexed by its code: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` doubled, so every
/// value is an integer in `[-12, 12]` (three balanced trits). Code 8 is a
/// negative zero and decodes to 0.
pub const E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Twice the value of the E2M1 code in the low four bits of `code`.
#[inline]
pub fn e2m1_x2(code: u8) -> i8 {
    E2M1_X2[(code & 0x0F) as usize]
}

/// An E4M3 pattern (1 sign, 4 exponent bits with bias 7, 3 mantissa bits; no
/// infinities, `0x7F` and `0xFF` are NaN) as `(negative, m, e)` with
/// `|v| = m * 2^e`: `m = 8 + mantissa`, `e = exponent - 10` for a normal
/// pattern, `m = mantissa`, `e = -9` for a subnormal one. `m <= 15`, so the
/// largest value is `15 * 2^5 = 448`. `Err(InvalidInput)` for a NaN.
#[inline]
pub fn decompose_e4m3(bits: u8) -> Result<(bool, u64, i32), OverflowDetected> {
    if bits & 0x7F == 0x7F {
        return Err(OverflowDetected::InvalidInput);
    }
    let sign = bits & 0x80 != 0;
    let exp = ((bits >> 3) & 0x0F) as i32;
    let mant = (bits & 0x07) as u64;
    Ok(if exp == 0 { (sign, mant, -9) } else { (sign, mant | 0x08, exp - 10) })
}

/// A UE8M0 scale (an unsigned 8-bit exponent, the value `2^(bits - 127)`) as
/// its exponent. `Err(InvalidInput)` for `0xFF` (NaN).
#[inline]
pub fn ue8m0_exponent(bits: u8) -> Result<i32, OverflowDetected> {
    if bits == 0xFF {
        return Err(OverflowDetected::InvalidInput);
    }
    Ok(bits as i32 - 127)
}

/// An IEEE binary32 pattern as `(negative, m, e)` with `|v| = m * 2^e` (`m`
/// below `2^24`, the implicit bit included for a normal pattern). Zero gives
/// `m = 0`. `Err(InvalidInput)` for an infinity or a NaN.
#[inline]
pub fn decompose_f32(bits: u32) -> Result<(bool, u64, i32), OverflowDetected> {
    let sign = bits & 0x8000_0000 != 0;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x007F_FFFF) as u64;
    if exp == 0xFF {
        return Err(OverflowDetected::InvalidInput);
    }
    Ok(if exp == 0 { (sign, mant, -149) } else { (sign, mant | 0x0080_0000, exp - 150) })
}

/// A binary32 pattern at `frac_bits` fractional bits: `trunc(v * 2^frac_bits)`
/// toward zero as an i128, the rule of [`to_raw`]. Exact whenever the value
/// has no bits below `2^-frac_bits`; the exact value itself is
/// [`decompose_f32`]. `Err(TierOverflow)` when the result does not fit i128,
/// `Err(InvalidInput)` for an infinity or a NaN.
pub fn f32_to_raw(bits: u32, frac_bits: u32) -> Result<i128, OverflowDetected> {
    let (sign, m, e) = decompose_f32(bits)?;
    trunc_raw(sign, m, e, frac_bits)
}

/// A binary32 pattern as a raw storage word of `F` fraction bits (truncated
/// toward zero), `TierOverflow` when its magnitude does not fit the word, the
/// rule of [`to_storage_raw`].
pub fn f32_to_storage_raw<S: Storage, const F: u32>(bits: u32) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let raw = f32_to_raw(bits, F)?;
    match (S::from_i128_checked(raw), raw.checked_neg().and_then(S::from_i128_checked)) {
        (Some(v), Some(_)) => Ok(v),
        _ => Err(OverflowDetected::TierOverflow),
    }
}

/// A binary32 pattern as a `Fixed<S, F>` (truncated toward zero), `TierOverflow` past the word.
#[inline]
pub fn f32_to_fixed<S: Storage, const F: u32>(bits: u32) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Ok(Fixed::from_raw(f32_to_storage_raw::<S, F>(bits)?))
}
