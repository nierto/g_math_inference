//! The inference module at every width (g_math phase 4d), moved from
//! g_math's tests/width_types_validation.rs with the module (0.7.0).

use g_math::fixed_point::{Fixed, FixedNum, Storage, WideInt, I256, I512};

fn dyadic<S: Storage, const F: u32>(value_times_256: i32) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::from_raw(S::from_i64(value_times_256 as i64).shl_bits(F - 8))
}

// ---- the inference module at every width (phase 4d) ---------------------------

use g_math_inference::{tq19_dot, trit_dot, HybridTQ19, PlanarTQ19, TQ19Matrix, TqWord, WideOutput, SCALE};

/// Identities at every width: a weight of SCALE is 1.0, so the dot with
/// one activation returns it; a trit dot sums with signs; the dense,
/// planar and hybrid matvecs of the identity return the activations.
fn inference_identities<S: TqWord, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let q = |n: i32| dyadic::<S, F>(n); // n / 256
    let x = [q(300), q(-77), q(1024), q(5)];
    assert_eq!(tq19_dot(&[SCALE as i16], &x[..1]), x[0], "unit weight at F={F}");
    assert_eq!(trit_dot(&[1, -1, 0, 1], &x), x[0] - x[1] + x[3], "trit dot at F={F}");
    let m = TQ19Matrix::from_fn(4, 4, |r, c| if r == c { SCALE as i16 } else { 0 });
    assert_eq!(m.matvec(&x), x.to_vec(), "identity matvec at F={F}");
    assert_eq!(PlanarTQ19::from_tq19(&m).matvec(&x), x.to_vec(), "planar identity at F={F}");
    assert_eq!(HybridTQ19::from_tq19(&m).matvec_par(&x), x.to_vec(), "hybrid identity at F={F}");
}

/// The wide output narrows to the matvec bit-for-bit on the words that have it.
fn wide_output_identity<S: WideOutput, const F: u32>()
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let q = |n: i32| dyadic::<S, F>(n);
    let x = [q(300), q(-77), q(1024), q(5)];
    let m = TQ19Matrix::from_fn(3, 4, |r, c| ((r * 7 + c * 3) as i16 % 61) * 97 - 1500);
    let narrow = m.matvec(&x);
    let wide = m.matvec_q2f(&x);
    for r in 0..3 {
        let one = <<S as Storage>::Compute as g_math::fixed_point::WideInt>::one();
        let back = S::narrow(g_math::fixed_point::WideInt::divmod(wide[r], g_math::fixed_point::WideInt::shl_bits(one, F)).0).unwrap();
        assert_eq!(back, narrow[r].raw(), "q2f narrows to matvec at F={F}, row {r}");
    }
}

#[test]
fn inference_runs_at_every_width() {
    inference_identities::<i32, 10>();
    inference_identities::<i32, 16>();
    inference_identities::<i32, 24>();
    inference_identities::<i64, 32>();
    inference_identities::<i128, 64>();
    inference_identities::<I256, 128>();
    inference_identities::<I512, 256>();
    wide_output_identity::<i32, 10>();
    wide_output_identity::<i32, 16>();
    wide_output_identity::<i64, 32>();
}
