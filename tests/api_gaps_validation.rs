//! Public functions of the module that g_math's 2026-10-16 coverage audit
//! found never executed (moved here with the module): the parallel packed
//! trit matvec, the hybrid / row-scaled accessors, WeightBits::storage_raw
//! and the module constants. Exact references.

#[macro_use]
mod common;

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = g_math::fixed_point::Fixed<$S, $F>;
            fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
            fn fps(v: &[&str]) -> Vec<FixedPoint> { v.iter().map(|s| fp(s)).collect() }

            #[test]
            fn packed_trit_matvec_par_equals_the_serial_kernel() {
                // the module's encoding: 5 trits per byte, first trit * 81, digit t + 1
                #[derive(Clone, Copy)]
                enum Trit { Neg, Zero, Pos }
                fn pack_trits(t: &[Trit]) -> Vec<u8> {
                    t.chunks(5).map(|c| (0..5).fold(0u8, |b, j| b * 3 + c.get(j).map_or(1, |t| match t { Trit::Neg => 0, Trit::Zero => 1, Trit::Pos => 2 }))).collect()
                }
                // small exact case: scale 1, activations integers
                let trits = [Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos, Trit::Pos, Trit::Neg, Trit::Zero];
                let (rows, cols) = (2, 7);
                let mut both = trits.to_vec();
                both.extend(trits.iter().map(|t| match t { Trit::Pos => Trit::Neg, Trit::Neg => Trit::Pos, Trit::Zero => Trit::Zero }));
                let packed: Vec<u8> = both.chunks(cols).flat_map(|r| pack_trits(r)).collect();
                let act = fps(&["1", "2", "3", "4", "5", "6", "7"]);
                let sc = fps(&["1", "0.5"]);
                let got = g_math_inference::packed_trit_matvec_par(&packed, rows, cols, &act, &sc);
                // 1 - 2 + 4 + 5 - 6 = 2; second row negated and halved
                assert_eq!(got, fps(&["2", "-1"]));
                // larger: parallel and serial bit-identical
                let (rows, cols) = (37, 53);
                let mut s = 12345u64;
                let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u32 };
                let bpr = (cols + 4) / 5;
                let packed: Vec<u8> = (0..rows * bpr).map(|_| (next() % 243) as u8).collect();
                let act: Vec<FixedPoint> = (0..cols).map(|_| fp(&format!("{}", (next() % 16) as i32 - 8))).collect();
                let sc: Vec<FixedPoint> = (0..rows).map(|_| fp(&format!("0.{}", 1 + next() % 9))).collect();
                assert_eq!(g_math_inference::packed_trit_matvec_par(&packed, rows, cols, &act, &sc), g_math_inference::packed_trit_matvec(&packed, rows, cols, &act, &sc));
            }
        }
    };
}

for_each_format!(suite);

mod tq19_accessors {
    use g_math::fixed_point::Fixed;
    use g_math_inference::bits::{to_storage_raw, HalfKind, WeightBits};
    use g_math_inference::{HybridTQ19, RowScaledTQ19, TQ19Matrix, HYBRID_LOW_TRITS, LOW_BIAS, LOW_MOD, NUM_PLANES, POW3, SPARSE_DENSITY_PERCENT};

    #[test]
    fn hybrid_counts_its_high_corrections() {
        // a raw needs a high correction when it leaves the 7-trit low range [-1093, 1093]
        let data: Vec<i16> = vec![0, 1093, -1093, 1094, -1094, 29524, -29524, 5, 2187, -3000, 12, 0];
        let h = HybridTQ19::from_tq19(&TQ19Matrix::new(3, 4, data.clone()));
        let want = data.iter().filter(|&&r| (r as i32).abs() > LOW_BIAS as i32).count();
        assert_eq!(h.num_high_corrections(), want);
        assert_eq!((HYBRID_LOW_TRITS, LOW_MOD, LOW_BIAS), (7, 3i32.pow(7), ((3i32.pow(7) - 1) / 2) as i16));
        assert_eq!(NUM_PLANES, 10);
        for (i, p) in POW3.iter().enumerate() { assert_eq!(*p as i32, 3i32.pow(i as u32)); }
        assert_eq!(SPARSE_DENSITY_PERCENT, 5);
    }

    #[test]
    fn rowscaled_size_and_weight_bits_storage() {
        let r = RowScaledTQ19::from_parts(3, 5, vec![1i16; 15], vec![1u64 << 32; 3]);
        assert_eq!(r.size_bytes(), 15 * 2 + 3 * 8);
        let halves: [u16; 6] = [0x3f80, 0xbf80, 0x3e00, 0x4040, 0x0000, 0xc2c8]; // bf16: 1, -1, 0.125, 3, 0, -100
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let w = WeightBits::from_le_bytes(2, 3, HalfKind::BFloat16, &bytes).unwrap();
        for (i, h) in halves.iter().enumerate() {
            assert_eq!(w.storage_raw::<i32, 10>(i).unwrap(), to_storage_raw::<i32, 10>(*h, HalfKind::BFloat16).unwrap());
            assert_eq!(w.storage_raw::<i64, 32>(i).unwrap(), to_storage_raw::<i64, 32>(*h, HalfKind::BFloat16).unwrap());
        }
        assert_eq!(Fixed::<i32, 10>::from_raw(w.storage_raw::<i32, 10>(5).unwrap()), Fixed::<i32, 10>::from_str("-100"));
    }
}

