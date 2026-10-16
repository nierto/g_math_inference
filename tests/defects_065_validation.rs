//! Defects fixed in g_math 0.6.5, tq19 part: tq19 narrowing wrapped (`as`
//! casts) where every other infallible narrowing panics; the AVX2 trit
//! kernel negated `i32::MIN` in 32 bits. Moved from g_math's
//! tests/defects_065_validation.rs with the module (0.7.0).
//!
//! References are exact integer arithmetic. No floats.

// The tq19 narrowing defects: 32-bit activations, so the 32-bit formats only.
macro_rules! suite_tq19_narrowing {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            use g_math::fixed_point::Fixed;
            type FixedPoint = Fixed<$S, $F>;
            #[allow(dead_code)]
            const FRAC_BITS: u32 = $F;
            #[allow(dead_code)]
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            use g_math_inference::{packed_trit_dot, tq19_dot, tq19_dot_q2f, trit_dot, TQ19Matrix, MAX_RAW, SCALE};

            /// 0.6.4 returned -2147483648 (the true sum is 2^31).
            #[test]
            #[should_panic(expected = "exceeds storage range")]
            fn trit_dot_past_storage_panics() {
                let _ = trit_dot(&[1, 1], FixedPoint::from_raw_slice(&[1 << 30, 1 << 30]));
            }

            /// 0.6.4 returned -436426 (the true value is 25,769,367,350).
            #[test]
            #[should_panic(expected = "exceeds storage range")]
            fn tq19_dot_past_storage_panics() {
                let _ = tq19_dot(&[MAX_RAW; 8], FixedPoint::from_raw_slice(&[i32::MAX; 8]));
            }

            #[test]
            #[should_panic(expected = "exceeds storage range")]
            fn tq19_matvec_past_storage_panics() {
                let m = TQ19Matrix::new(1, 8, vec![MAX_RAW; 8]);
                let _ = m.matvec(FixedPoint::from_raw_slice(&[i32::MAX; 8]));
            }

            /// Byte 229 decodes to trits [1, 1, 0, 0, 0].
            #[test]
            #[should_panic(expected = "exceeds storage range")]
            fn packed_trit_dot_past_storage_panics() {
                let _ = packed_trit_dot(&[229], 5, FixedPoint::from_raw_slice(&[1 << 30, 1 << 30, 0, 0, 0]), FixedPoint::from_raw(1 << FRAC_BITS));
            }

            /// The unscaled dot (2^31) exceeds storage but the scaled value (2^30)
            /// fits: one exact product, one rounding. 0.6.4 narrowed the dot first
            /// and returned -2^30.
            #[test]
            fn packed_trit_dot_scales_before_narrowing() {
                let half = 1 << (FRAC_BITS - 1);
                assert_eq!(packed_trit_dot(&[229], 5, FixedPoint::from_raw_slice(&[1 << 30, 1 << 30, 0, 0, 0]), FixedPoint::from_raw(half)).raw(), 1 << 30);
                // unchanged in range: (3 + 4) * 0.5 = 3.5 raw -> ties toward +inf
                assert_eq!(packed_trit_dot(&[229], 5, FixedPoint::from_raw_slice(&[3, 4, 0, 0, 0]), FixedPoint::from_raw(half)).raw(), 4);
                assert_eq!(packed_trit_dot(&[229], 5, FixedPoint::from_raw_slice(&[-3, -4, 0, 0, 0]), FixedPoint::from_raw(half)).raw(), -3);
            }

            /// -(i32::MIN) is 2^31. The AVX2 kernel flipped the sign in 32 bits, which
            /// wraps back to i32::MIN; with eight elements it disagreed with the
            /// scalar path by 2^32.
            #[test]
            fn trit_dot_negates_i32_min_exactly() {
                let trits = [-1i8, 1, 0, 0, 0, 0, 0, 0, 1];
                let acts = [i32::MIN, -5, 9, 9, 9, 9, 9, 9, 2];
                // 2^31 - 5 + 2 = 2147483645
                assert_eq!(trit_dot(&trits, FixedPoint::from_raw_slice(&acts)).raw(), 2147483645);
                assert_eq!(trit_dot(&trits[..2], FixedPoint::from_raw_slice(&acts[..2])).raw(), 2147483643);
            }

            fn reference(weights: &[i16], acts: &[i32]) -> i128 {
                weights.iter().zip(acts).map(|(&w, &a)| w as i128 * a as i128).sum()
            }

            /// Rows past 2^16 columns take the exactly-summed path; rows below it the
            /// unchecked one. Both equal the i128 reference.
            #[test]
            fn long_rows_match_the_exact_reference() {
                for &n in &[131usize, 65_536, 65_536 + 24, 150_000] {
                    let weights: Vec<i16> = (0..n).map(|i| ((i as i64 * 48_271 % 59_049) - 29_524) as i16).collect();
                    let acts: Vec<i32> = (0..n).map(|i| ((i as i64 * 40_503 % 8_191) - 4_095) as i32).collect();
                    let acc = reference(&weights, &acts);
                    assert_eq!(tq19_dot(&weights, FixedPoint::from_raw_slice(&acts)).raw() as i128, acc / SCALE as i128, "n = {n}");
                    assert_eq!(tq19_dot_q2f(&weights, FixedPoint::from_raw_slice(&acts)) as i128, (acc << FRAC_BITS) / SCALE as i128, "n = {n}");
                    let m = TQ19Matrix::new(1, n, weights.clone());
                    let batch: Vec<&[i32]> = vec![&acts[..], &acts[..]];
                    let expected = vec![vec![FixedPoint::from_raw((acc / SCALE as i128) as i32)]; 2];
                    assert_eq!(m.matvec_batch(&(&batch).iter().map(|v| FixedPoint::from_raw_slice(v)).collect::<Vec<_>>()), expected, "n = {n}");
                    assert_eq!(m.matvec_batch_par(&(&batch).iter().map(|v| FixedPoint::from_raw_slice(v)).collect::<Vec<_>>()), expected, "n = {n}");
                }
            }

            /// A long row whose exact sum leaves i64 panics instead of wrapping.
            #[test]
            #[should_panic(expected = "exceeds")]
            fn long_row_accumulator_overflow_panics() {
                let n = (1usize << 17) + 8;
                let _ = tq19_dot(&vec![i16::MIN; n], FixedPoint::from_raw_slice(&vec![i32::MIN; n]));
            }
        }
    };
}
suite_tq19_narrowing!(tq19_narrowing_q22_10, i32, 10);
suite_tq19_narrowing!(tq19_narrowing_q16_16, i32, 16);
