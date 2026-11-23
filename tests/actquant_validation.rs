//! Activation quantisation (NVFP4 four-over-six, MXFP8 ceil exponent) and
//! the matvecs on quantised activations against an independent Fraction
//! implementation of the same exact rule (scripts/generate_actquant_refs.py):
//! codes, scales, row factor, and floor / nearest outputs. No floats.

#[path = "data/actquant_refs.rs"]
mod refs;

use g_math::fixed_point::Fixed;
use g_math_inference::{quantize_mxfp8, quantize_nvfp4, MxFp8Matrix, NvFp4Matrix, Rounding};

macro_rules! check {
    ($c:expr, $F:literal) => {{
        type X = Fixed<i32, $F>;
        let c = $c;
        let x: Vec<X> = c.x.iter().map(|&v| X::from_raw(v)).collect();
        let nv = quantize_nvfp4(&x);
        assert_eq!(nv.codes_x2, c.nv_codes, "NVFP4 codes");
        assert_eq!(nv.scales, c.nv_scales, "NVFP4 scales");
        assert_eq!(nv.amax_raw, c.nv_amax, "NVFP4 amax");
        let mx = quantize_mxfp8(&x);
        assert_eq!(mx.elems, c.mx_elems, "MXFP8 elements");
        assert_eq!(mx.scales, c.mx_scales, "MXFP8 scales");
        let nvm = NvFp4Matrix::from_parts(refs::ROWS, refs::COLS, refs::NV_CODES.to_vec(), refs::NV_SCALES.to_vec(), refs::NV_AMAX, refs::NV_DEN).unwrap();
        let mxm = MxFp8Matrix::from_parts(refs::ROWS, refs::COLS, refs::MX_ELEMS.to_vec(), refs::MX_SCALES.to_vec()).unwrap();
        let raw = |v: Vec<X>| v.into_iter().map(|y| y.raw() as i64).collect::<Vec<i64>>();
        assert_eq!(raw(nvm.matvec_quantized::<$F>(&nv)), c.nv_floor, "NVFP4 quantised matvec, floor");
        assert_eq!(raw(nvm.clone().with_rounding(Rounding::Nearest).matvec_quantized::<$F>(&nv)), c.nv_near, "NVFP4 quantised matvec, nearest");
        assert_eq!(raw(mxm.matvec_quantized::<$F>(&mx)), c.mx_floor, "MXFP8 quantised matvec, floor");
        assert_eq!(raw(mxm.clone().with_rounding(Rounding::Nearest).matvec_quantized::<$F>(&mx)), c.mx_near, "MXFP8 quantised matvec, nearest");
    }};
}

#[test]
fn activation_quantisation_equals_the_fraction_rule() {
    let mut n = 0;
    for c in refs::CASES {
        match c.frac {
            10 => check!(c, 10),
            _ => check!(c, 16),
        }
        n += 1;
    }
    eprintln!(
        "{n} activation rows: codes, scales and quantised matvecs equal the Fraction rule ({} blocks chose the 4 candidate, {} ties kept 6, {} zero and {} subnormal scales on nonzero blocks)",
        refs::CHOSE4, refs::TIES_KEPT_6, refs::ZERO_SCALE, refs::SUBNORMAL_SCALE
    );
    // the gate reaches both candidates and the small-scale edges
    assert!(n > 0 && refs::CHOSE4 > 0 && refs::ZERO_SCALE > 0 && refs::SUBNORMAL_SCALE > 0);
}
