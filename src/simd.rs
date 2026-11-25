//! AVX2 SIMD acceleration for TQ1.9 operations.
//!
//! Active only on x86_64 with the realtime profile (Q16.16, i32 activations).
//!
//! - [`tq19_dot_avx2`]: 8× i32 multiply-accumulate per cycle
//! - [`trit_dot_avx2`]: 8× zero-multiply per cycle via `_mm256_sign_epi32`
//!
//! These functions are called from `ops.rs` via runtime detection
//! (`is_x86_feature_detected!("avx2")`). Scalar fallback is automatic.

// Only compile on x86_64 — ARM/RISC-V get scalar paths

use std::arch::x86_64::*;


// ============================================================================
// TQ1.9 dot product — AVX2 (realtime profile: i32 activations)
// ============================================================================

/// AVX2-accelerated TQ1.9 dot product for Q16.16 profile.
///
/// Processes 8 weight×activation pairs per iteration using
/// `_mm256_mul_epi32` (signed 32×32→64 multiply on even/odd lanes).
///
/// Returns i64 (i64 for Q16.16) accumulator before SCALE division.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn tq19_dot_avx2(
    weights: &[i16],
    activations: &[i32],
) -> i64 {
    let n = weights.len();
    let chunks = n / 8;

    let mut acc_even = _mm256_setzero_si256();
    let mut acc_odd = _mm256_setzero_si256();

    let w_ptr = weights.as_ptr();
    let a_ptr = activations.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8× i16 weights → sign-extend to 8× i32
        let w_128 = _mm_loadu_si128(w_ptr.add(offset) as *const __m128i);
        let w = _mm256_cvtepi16_epi32(w_128);

        // Load 8× i32 activations
        let a = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);

        // Even-lane multiply: positions 0,2,4,6 → 4× i64
        let prod_even = _mm256_mul_epi32(w, a);
        acc_even = _mm256_add_epi64(acc_even, prod_even);

        // Odd-lane multiply: shift right 32 to move odd→even positions
        let w_odd = _mm256_srli_epi64(w, 32);
        let a_odd = _mm256_srli_epi64(a, 32);
        let prod_odd = _mm256_mul_epi32(w_odd, a_odd);
        acc_odd = _mm256_add_epi64(acc_odd, prod_odd);
    }

    // Horizontal sum of 8× i64 accumulators
    let acc = _mm256_add_epi64(acc_even, acc_odd);
    let mut result = hsum_epi64(acc);

    // Scalar remainder
    for i in (chunks * 8)..n {
        result += (weights[i] as i64) * (activations[i] as i64);
    }

    result
}

// Stub for non-q16_16 profiles — never called, exists only so that
// the cfg(target_arch) module compiles on all profiles.

// ============================================================================
// Trit dot product — AVX2 (realtime profile: i32 activations)
// ============================================================================

/// AVX2-accelerated zero-multiply trit dot for Q16.16 profile.
///
/// Uses `_mm256_sign_epi32` to apply trit sign {-1,0,+1} to activations:
/// - trit = +1 → keep activation
/// - trit = 0  → zero
/// - trit = -1 → negate activation
///
/// Processes 8 elements per iteration. Accumulates in i64 to prevent overflow.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn trit_dot_avx2(
    trits: &[i8],
    activations: &[i32],
) -> i64 {
    let n = trits.len();
    let chunks = n / 8;

    let mut acc_lo = _mm256_setzero_si256();  // 4× i64
    let mut acc_hi = _mm256_setzero_si256();  // 4× i64
    let min = _mm256_set1_epi32(i32::MIN);
    let mut seen_min = _mm256_setzero_si256();

    let t_ptr = trits.as_ptr();
    let a_ptr = activations.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 trits (i8) → sign-extend to 8× i32
        let t_64 = _mm_loadl_epi64(t_ptr.add(offset) as *const __m128i);
        let t_i32 = _mm256_cvtepi8_epi32(t_64);

        // Load 8× i32 activations
        let a = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);

        // Apply trit sign: _mm256_sign_epi32(a, t)
        //   t > 0 → +a,  t == 0 → 0,  t < 0 → -a
        let signed = _mm256_sign_epi32(a, t_i32);

        // A 32-bit sign flip of i32::MIN wraps back to i32::MIN. A lane equal
        // to i32::MIN is the only way that can have happened: remember it and
        // redo the (rare) row exactly after the loop.
        seen_min = _mm256_or_si256(seen_min, _mm256_cmpeq_epi32(signed, min));

        // Widen 8× i32 → 2×(4× i64) and accumulate
        let lo_128 = _mm256_castsi256_si128(signed);
        let hi_128 = _mm256_extracti128_si256(signed, 1);
        let lo_64 = _mm256_cvtepi32_epi64(lo_128);
        let hi_64 = _mm256_cvtepi32_epi64(hi_128);
        acc_lo = _mm256_add_epi64(acc_lo, lo_64);
        acc_hi = _mm256_add_epi64(acc_hi, hi_64);
    }

    // Horizontal sum of 8× i64
    let acc = _mm256_add_epi64(acc_lo, acc_hi);
    if _mm256_movemask_epi8(seen_min) != 0 {
        let mut exact = 0i64;
        for i in 0..n {
            match trits[i] {
                1 => exact += activations[i] as i64,
                -1 => exact -= activations[i] as i64,
                _ => {}
            }
        }
        return exact;
    }
    let mut result = hsum_epi64(acc);

    // Scalar remainder
    for i in (chunks * 8)..n {
        match trits[i] {
            1 => result += activations[i] as i64,
            -1 => result -= activations[i] as i64,
            _ => {}
        }
    }

    result
}


// ============================================================================
// TQ1.9 rows x a few activations by 16-bit multiply-adds over 11-bit limbs
// ============================================================================

/// Bits per activation limb of the TQ1.9 batched path.
pub(crate) const LIMB_BITS: u32 = 11;

/// The per-limb sums `sum_i w[i] * limb_{t,l}[i]` for `T` activations of `L`
/// limbs each, over `16 * steps` columns of 16-bit weights `w` (limbs of `lb`
/// bits: `x = sum_l limb_l * 2^(lb l)`, the lower limbs unsigned, the top one
/// signed, every limb within +-2^lb). A 32-bit lane gains at most
/// `2 * max|w| * 2^lb` per step and moves to 64 bits every `period` steps,
/// which the caller chose so the lane cannot overflow.
///
/// # Safety
/// AVX2 available; `w` and every limb slice hold `16 * steps` elements.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn limb_tile<const T: usize, const L: usize>(w: *const i16, x: [[*const i16; L]; T], steps: usize, period: usize) -> [[i64; L]; T] {
    let mut acc = [[_mm256_setzero_si256(); L]; T];
    let mut wide = [[_mm256_setzero_si256(); L]; T];
    let mut since = 0usize;
    for s in 0..steps {
        let wv = _mm256_loadu_si256(w.add(16 * s) as *const __m256i);
        for t in 0..T {
            for l in 0..L {
                let xv = _mm256_loadu_si256(x[t][l].add(16 * s) as *const __m256i);
                acc[t][l] = _mm256_add_epi32(acc[t][l], _mm256_madd_epi16(wv, xv));
            }
        }
        since += 1;
        if since == period {
            for t in 0..T {
                for l in 0..L {
                    wide[t][l] = _mm256_add_epi64(wide[t][l], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc[t][l])));
                    wide[t][l] = _mm256_add_epi64(wide[t][l], _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc[t][l], 1)));
                    acc[t][l] = _mm256_setzero_si256();
                }
            }
            since = 0;
        }
    }
    let mut out = [[0i64; L]; T];
    for t in 0..T {
        for l in 0..L {
            let v = _mm256_add_epi64(wide[t][l], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc[t][l])));
            let v = _mm256_add_epi64(v, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc[t][l], 1)));
            out[t][l] = hsum_epi64(v);
        }
    }
    out
}

/// [`limb_tile`] for 1 to 4 activations of 1 to 4 limbs; `limbs[t]` holds the
/// activation's limb slices; `out[t][l]` receives the sums.
///
/// # Safety
/// As [`limb_tile`]; `limbs.len()` in 1..=4, `l` in 1..=4.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn limb_tile_dyn(w: *const i16, limbs: &[[*const i16; 4]], l: usize, steps: usize, period: usize, out: &mut [[i64; 4]; 4]) {
    macro_rules! go {
        ($(($t:literal, $l:literal)),*) => {
            match (limbs.len(), l) {
                $( ($t, $l) => {
                    let x: [[*const i16; $l]; $t] = std::array::from_fn(|i| std::array::from_fn(|j| limbs[i][j]));
                    let s = limb_tile::<$t, $l>(w, x, steps, period);
                    for i in 0..$t {
                        out[i][..$l].copy_from_slice(&s[i]);
                    }
                } )*
                _ => unreachable!(),
            }
        };
    }
    go!((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4))
}

// ============================================================================
// Helper: horizontal sum of 4× i64 in __m256i
// ============================================================================

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi64(v: __m256i) -> i64 {
    // v = [a, b, c, d] (4× i64)
    let hi128 = _mm256_extracti128_si256(v, 1);   // [c, d]
    let lo128 = _mm256_castsi256_si128(v);          // [a, b]
    let sum128 = _mm_add_epi64(lo128, hi128);       // [a+c, b+d]
    let hi64 = _mm_srli_si128(sum128, 8);           // [b+d, 0]
    let total = _mm_add_epi64(sum128, hi64);        // [a+b+c+d, ...]
    _mm_cvtsi128_si64(total)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_tq19_dot_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // Skip on non-AVX2 hardware
        }

        let weights: Vec<i16> = (0..16).map(|i| (i * 1000 - 8000) as i16).collect();
        let activations: Vec<i32> = (0..16).map(|i| (i * 5000 + 1000) as i32).collect();

        // Scalar reference
        let mut scalar_acc: i64 = 0;
        for i in 0..16 {
            scalar_acc += (weights[i] as i64) * (activations[i] as i64);
        }

        // SIMD
        let simd_acc = unsafe { tq19_dot_avx2(&weights, &activations) };

        assert_eq!(simd_acc, scalar_acc, "SIMD and scalar must produce identical results");
    }

    #[test]
    fn simd_trit_dot_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let trits: Vec<i8> = vec![1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 0, 1, -1, 1, -1];
        let activations: Vec<i32> = (0..16).map(|i| (i * 3000 + 500) as i32).collect();

        // Scalar reference
        let mut scalar_acc: i64 = 0;
        for i in 0..16 {
            match trits[i] {
                1 => scalar_acc += activations[i] as i64,
                -1 => scalar_acc -= activations[i] as i64,
                _ => {}
            }
        }

        let simd_acc = unsafe { trit_dot_avx2(&trits, &activations) };

        assert_eq!(simd_acc, scalar_acc, "SIMD trit dot must match scalar");
    }

    #[test]
    fn simd_handles_remainder() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        // 13 elements: 1 full chunk of 8 + 5 remainder
        let weights: Vec<i16> = (0..13).map(|i| (i * 500) as i16).collect();
        let activations: Vec<i32> = (0..13).map(|i| (i * 1000 + 100) as i32).collect();

        let mut scalar_acc: i64 = 0;
        for i in 0..13 {
            scalar_acc += (weights[i] as i64) * (activations[i] as i64);
        }

        let simd_acc = unsafe { tq19_dot_avx2(&weights, &activations) };
        assert_eq!(simd_acc, scalar_acc, "remainder handling must be correct");
    }
}
