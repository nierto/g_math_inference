//! Speed-only changes return what 0.1.2 returned: every RowScaledTQ5 kernel
//! entry point, on shapes that reach every path (row counts that are and are
//! not multiples of the tile, columns that are and are not multiples of 16,
//! batches of 1 to 9 and 64, activations whose high halves are sparse, dense,
//! and too large to split), compared value for value, panics included.

use g_math::fixed_point::Fixed;
use std::panic::{catch_unwind, AssertUnwindSafe};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn silence() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::panic::set_hook(Box::new(|_| {})));
}

fn run<T: std::fmt::Debug>(f: impl FnOnce() -> T) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => format!("{v:?}"),
        Err(_) => "panic".into(),
    }
}

/// Activations of one of four kinds: small (sparse high halves), mixed,
/// dense high halves, extreme (some |x| >= 2^30: no split).
fn activation(rng: &mut Rng, n: usize, kind: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            let bits = match kind {
                0 => 14,
                1 => if r % 10 == 0 { 24 } else { 14 },
                2 => 24,
                _ => if r % 50 == 0 { 31 } else { 20 },
            };
            let m = ((r >> 8) & ((1u64 << bits) - 1)) as i64;
            let v = if r & 1 == 1 { -m } else { m };
            v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        })
        .collect()
}

macro_rules! gate {
    ($name:ident, $f:literal) => {
        #[test]
        fn $name() {
            silence();
            type X = Fixed<i32, $f>;
            let mut rng = Rng(0x0310 + $f);
            let mut cases = 0usize;
            let mut panics = 0usize;
            let shapes: &[(usize, usize)] = &[(1, 1), (3, 15), (4, 16), (5, 17), (7, 33), (8, 64), (12, 100), (13, 1024), (17, 1040), (33, 2048), (64, 4096), (130, 517)];
            for (si, &(rows, cols)) in shapes.iter().enumerate() {
                for scale_kind in 0..2u64 {
                    let data: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
                    // small scales keep results in range; large ones reach the overflow panic
                    let scales: Vec<u64> = (0..rows).map(|_| if scale_kind == 0 { (rng.next() >> 44) + 1 } else { rng.next() >> 20 }).collect();
                    let om = old::RowScaledTQ5::from_parts(rows, cols, data.clone(), scales.clone());
                    let nm = new::RowScaledTQ5::from_parts(rows, cols, data, scales);
                    for kind in 0..4u64 {
                        let batch_sizes: &[usize] = if si % 3 == 0 { &[1, 2, 3, 4, 5, 8, 9, 64] } else { &[1, 2, 3, 4, 7] };
                        for &b in batch_sizes {
                            let xs: Vec<Vec<X>> = (0..b).map(|_| activation(&mut rng, cols, kind).into_iter().map(X::from_raw).collect()).collect();
                            let refs: Vec<&[X]> = xs.iter().map(|v| &v[..]).collect();
                            let x = &xs[0][..];
                            let checks = [
                                ("matvec", run(|| om.matvec(x)), run(|| nm.matvec(x))),
                                ("matvec_par", run(|| om.matvec_par(x)), run(|| nm.matvec_par(x))),
                                ("matvec_q2f", run(|| om.matvec_q2f(x)), run(|| nm.matvec_q2f(x))),
                                ("matvec_q2f_par", run(|| om.matvec_q2f_par(x)), run(|| nm.matvec_q2f_par(x))),
                                ("matvec_batch_par", run(|| om.matvec_batch_par(&refs)), run(|| nm.matvec_batch_par(&refs))),
                                ("matvec_batch_par_into", run(|| { let mut o = vec![X::from_raw(0); b * rows]; om.matvec_batch_par_into(&refs, &mut o); o }),
                                                          run(|| { let mut o = vec![X::from_raw(0); b * rows]; nm.matvec_batch_par_into(&refs, &mut o); o })),
                            ];
                            for (what, o, n) in checks {
                                cases += 1;
                                if o == "panic" { panics += 1; }
                                assert_eq!(o, n, "{what} differs: {rows}x{cols} batch {b} kind {kind} scales {scale_kind}");
                            }
                        }
                    }
                }
            }
            eprintln!("{}: {cases} outcomes ({panics} panics), all equal to 0.1.2", stringify!($name));
            assert!(panics < cases / 2);
        }
    };
}

gate!(rowscaled_tq5_q22_10_same_as_0_1_2, 10);
gate!(rowscaled_tq5_q16_16_same_as_0_1_2, 16);
