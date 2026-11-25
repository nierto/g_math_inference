//! Gate for the binary32 decoders `f32_to_raw`, `f32_to_storage_raw`,
//! `f32_to_fixed` against mpmath references (scripts/generate_f32_refs.py):
//! every exponent, subnormals, the storage-range edges, infinities and NaNs.

#[path = "data/f32_refs.rs"]
mod refs;

use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::{Fixed, FixedNum};
use g_math_inference::bits::{decompose_f32, f32_to_fixed, f32_to_raw, f32_to_storage_raw};

/// The storage form from the i128 reference: the magnitude must fit the word.
fn storage<S: TryFrom<i128>>(r: &Result<i128, OverflowDetected>) -> Result<S, OverflowDetected> {
    match r {
        Ok(v) => match (S::try_from(*v), S::try_from(-*v)) {
            (Ok(s), Ok(_)) => Ok(s),
            _ => Err(OverflowDetected::TierOverflow),
        },
        Err(e) => Err(*e),
    }
}

fn check_storage<const F: u32>(bits: u32, want: &Result<i128, OverflowDetected>)
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    assert_eq!(f32_to_storage_raw::<i32, F>(bits), storage::<i32>(want), "{bits:#010x} i32 F {F}");
    assert_eq!(f32_to_fixed::<i32, F>(bits).map(|v| v.raw()), storage::<i32>(want));
}

#[test]
fn equals_the_mpmath_references() {
    let fracs = [0u32, 10, 16, 20, 30, 64];
    for (bits, want) in refs::CASES {
        for (i, &f) in fracs.iter().enumerate() {
            assert_eq!(f32_to_raw(*bits, f), want[i], "{bits:#010x} F {f}");
        }
        check_storage::<10>(*bits, &want[1]);
        check_storage::<16>(*bits, &want[2]);
        check_storage::<20>(*bits, &want[3]);
        check_storage::<30>(*bits, &want[4]);
        // the exact dyadic agrees: trunc(m * 2^(e + 64)) == the F = 64 value whenever it fits
        if let (Ok((neg, m, e)), Ok(v)) = (decompose_f32(*bits), &want[5]) {
            let k = e + 64;
            let mag = if k >= 0 { (m as i128) << k } else if k > -64 { (m >> (-k) as u32) as i128 } else { 0 };
            assert_eq!(if neg { -mag } else { mag }, *v);
        }
    }
    assert!(refs::CASES.len() >= 150);
}
