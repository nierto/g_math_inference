//! Exact rounding shared by the kernels: the rounding mode a matrix applies
//! to its exact row values, floor division by a fixed divisor through a
//! precomputed reciprocal, and floor / nearest division of 512-bit values
//! (the wide-activation reference path). Integer arithmetic only.

use g_math::fixed_point::I512;

/// How a matrix rounds its exact row values to the output word.
///
/// - `Default`: the form's established rounding, unchanged: toward minus
///   infinity for the row-scaled and block-scaled forms (`RowScaledTQ5`,
///   `RowScaledTQ19`, `NvFp4Matrix`, `MxFp8Matrix`), toward zero for the
///   TQ1.9 forms (`TQ19Matrix`, `PlanarTQ19`, `HybridTQ19`). `RowScaledTQ19`
///   rounds its dot before the row scale and again after it.
/// - `Nearest`: the exact value rounded once to the nearest output word,
///   ties toward +infinity. No systematic bias (floor and truncation each
///   carry about half a unit per output).
///
/// Every form, its parallel, batched and wide-output (`q2f`) variants apply
/// the matrix's mode; `with_rounding` sets it. Switching to `Nearest`
/// changes outputs (a new numerics epoch for anything pinned to `Default`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Rounding {
    #[default]
    Default,
    Nearest,
}

/// Floor division of a signed 128-bit integer by a fixed positive divisor
/// below 2^64 through a precomputed reciprocal (Moller-Granlund 2-by-1
/// steps): exact, and no 128-bit hardware division per output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloorDiv {
    pub(crate) d: u64,
    dn: u64,
    v: u64,
    shift: u32,
}

impl FloorDiv {
    pub(crate) const fn new(d: u64) -> Self {
        assert!(d > 0, "FloorDiv: zero divisor");
        let shift = d.leading_zeros();
        let dn = d << shift;
        // floor((2^128 - 1) / dn) - 2^64: the quotient lies in [2^64, 2^65)
        let v = (u128::MAX / dn as u128) as u64;
        Self { d, dn, v, shift }
    }

    /// `(q, r)` of `(u1 * 2^64 + u0) / dn` for `u1 < dn`.
    #[inline(always)]
    fn div_2by1(&self, u1: u64, u0: u64) -> (u64, u64) {
        let p = (self.v as u128) * (u1 as u128);
        let p = p.wrapping_add(((u1 as u128) << 64) | u0 as u128);
        let mut q1 = ((p >> 64) as u64).wrapping_add(1);
        let q0 = p as u64;
        let mut r = u0.wrapping_sub(q1.wrapping_mul(self.dn));
        if r > q0 {
            q1 = q1.wrapping_sub(1);
            r = r.wrapping_add(self.dn);
        }
        if r >= self.dn {
            q1 = q1.wrapping_add(1);
            r -= self.dn;
        }
        (q1, r)
    }

    /// `(n / d, n % d)` for an unsigned 128-bit `n`.
    #[inline(always)]
    fn divmod_u128(&self, n: u128) -> (u128, u64) {
        let s = self.shift;
        // n << s spans three words; the top one is below 2^s <= dn
        let u2 = if s == 0 { 0 } else { (n >> (128 - s)) as u64 };
        let m = n << s;
        let (q1, r) = self.div_2by1(u2, (m >> 64) as u64);
        let (q0, r) = self.div_2by1(r, m as u64);
        (((q1 as u128) << 64) | q0 as u128, r >> s)
    }

    /// `floor(n / d)`.
    #[inline(always)]
    pub(crate) fn floor(&self, n: i128) -> i128 {
        if self.d == 1 {
            return n;
        }
        let (q, r) = self.divmod_u128(n.unsigned_abs());
        // d >= 2, so q <= 2^126 and the casts are exact
        if n >= 0 {
            q as i128
        } else {
            -(q as i128) - (r != 0) as i128
        }
    }
}

/// `floor(n / d)` for `d != 0`.
pub(crate) fn floor_div_i512(n: I512, d: I512) -> I512 {
    let (q, r) = (n / d, n % d);
    if !r.is_zero() && (r.is_negative() != d.is_negative()) {
        q - I512::from(1i128)
    } else {
        q
    }
}

/// `floor(n / d + 1/2)` (ties toward +infinity) for `d > 0`.
pub(crate) fn nearest_div_i512(n: I512, d: I512) -> I512 {
    floor_div_i512((n << 1usize) + d, d << 1usize)
}

/// `n / d` rounded by `r` with the floor rule as `Default` (`d > 0`).
pub(crate) fn round_div_i512(n: I512, d: I512, r: Rounding) -> I512 {
    match r {
        Rounding::Default => floor_div_i512(n, d),
        Rounding::Nearest => nearest_div_i512(n, d),
    }
}

/// `n * 2^k` rounded by `r` (floor as `Default`), any sign of `k`.
pub(crate) fn round_shift_i512(n: I512, k: i32, r: Rounding) -> I512 {
    if k >= 0 {
        return n << k as usize;
    }
    let s = (-k) as usize;
    match r {
        Rounding::Default => n >> s,
        Rounding::Nearest => ((n >> (s - 1)) + I512::from(1i128)) >> 1usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i512_rounding_matches_i128() {
        let mut s: u64 = 0xA5A5;
        for i in 0..200_000u64 {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let n = ((r as i64) >> (r % 40)) as i128 * if i % 3 == 0 { 1 << 40 } else { 1 };
            let d = ((r >> 33) % 100_000 + 1) as i128;
            let (ni, di) = (I512::from(n), I512::from(d));
            assert_eq!(floor_div_i512(ni, di).as_i128(), n.div_euclid(d), "floor {n}/{d}");
            assert_eq!(nearest_div_i512(ni, di).as_i128(), (2 * n + d).div_euclid(2 * d), "nearest {n}/{d}");
            let k = (r % 60) as i32 - 30;
            let want_f = if k >= 0 { n << k } else { n >> (-k) };
            let want_n = if k >= 0 { n << k } else { (n + (1i128 << (-k - 1))) >> (-k) };
            assert_eq!(round_shift_i512(ni, k, Rounding::Default).as_i128(), want_f);
            assert_eq!(round_shift_i512(ni, k, Rounding::Nearest).as_i128(), want_n, "shift {n} {k}");
        }
    }
}

/// Activations of a wider word as exact balanced 30-bit limbs:
/// `x = sum_k limb_k * 2^(30 k)` with every limb in `[-2^29, 2^29)`, so each
/// limb vector runs through the 32-bit kernels (always splittable into
/// 16-bit halves). All-zero limbs are left out: `(k, limb)` pairs.
pub(crate) fn limbs30<W: Copy + Into<i128>>(x: &[W], word_bits: u32) -> Vec<(u32, Vec<i32>)> {
    let n = word_bits.div_ceil(30);
    let mut rest: Vec<i128> = x.iter().map(|&v| v.into()).collect();
    let mut out = Vec::new();
    for k in 0..n {
        let limb: Vec<i32> = rest.iter().map(|&v| ((v << 98) >> 98) as i32).collect();
        for (r, &l) in rest.iter_mut().zip(&limb) {
            *r = (*r - l as i128) >> 30;
        }
        if limb.iter().any(|&l| l != 0) {
            out.push((k, limb));
        }
    }
    debug_assert!(rest.iter().all(|&r| r == 0));
    out
}

/// A 512-bit output as a word, or the panic every narrowing gives.
pub(crate) fn narrow_wide<W: g_math::fixed_point::WideInt>(v: I512, what: &str) -> W {
    if v.fits_in_i128() {
        if let Some(w) = W::from_i128_checked(v.as_i128()) {
            return w;
        }
    }
    panic!("{what}: output exceeds storage range")
}
