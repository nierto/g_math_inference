//! ns per call, the published 0.1.2 ("old") against this tree ("new"), at the
//! consumer's shapes: RowScaledTQ5 matvec_par (decode, small and large
//! matrices) and matvec_batch_par (prefill 64, drafting 2-3, batch 8).
//! Median of repeated calls, rayon's global pool, inputs as in the gate.

use g_math::fixed_point::Fixed;
use std::hint::black_box;
use std::time::Instant;

type Fx = Fixed<i32, 10>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Activations like a model's at Q22.10: most below 2^14 raw (|x| < 16),
/// one in twenty up to 2^20 raw.
fn activation(rng: &mut Rng, n: usize) -> Vec<Fx> {
    (0..n)
        .map(|_| {
            let r = rng.next();
            let bits = if r % 20 == 0 { 20 } else { 14 };
            let m = ((r >> 8) & ((1 << bits) - 1)) as i32;
            Fx::from_raw(if r & 1 == 1 { -m } else { m })
        })
        .collect()
}

fn parts(rng: &mut Rng, rows: usize, cols: usize) -> (Vec<i8>, Vec<u64>) {
    let data = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
    let scales = (0..rows).map(|_| (rng.next() >> 40) + (1 << 20)).collect();
    (data, scales)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Old and new alternated call by call, so both see the same machine state;
/// the median of each.
fn time2(reps: usize, mut f: impl FnMut(), mut g: impl FnMut()) -> (f64, f64) {
    for _ in 0..3 { f(); g(); }
    let (mut a, mut b) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
    for _ in 0..reps {
        let t = Instant::now(); f(); a.push(t.elapsed().as_nanos() as f64);
        let t = Instant::now(); g(); b.push(t.elapsed().as_nanos() as f64);
    }
    (median(a), median(b))
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("crossover") { crossover(); return; }
    if std::env::args().nth(1).as_deref() == Some("blockscaled") { blockscaled(); return; }
    let mut rng = Rng(0x5EED_0001);
    println!("threads: {}", rayon::current_num_threads());
    println!("{:<34} {:>14} {:>14} {:>8}", "RowScaledTQ5", "0.1.2 ns", "now ns", "ratio");
    let report = |label: String, o: f64, n: f64| println!("{:<34} {:>14.0} {:>14.0} {:>8.3}", label, o, n, n / o);

    for &(rows, cols) in &[(32usize, 4096usize), (256, 2048), (1024, 4096), (2048, 2048), (4096, 4096), (11008, 2048)] {
        let (data, scales) = parts(&mut rng, rows, cols);
        let old_m = old::RowScaledTQ5::from_parts(rows, cols, data.clone(), scales.clone());
        let new_m = new::RowScaledTQ5::from_parts(rows, cols, data, scales);
        let x = activation(&mut rng, cols);
        let reps = if rows * cols > 4_000_000 { 60 } else { 600 };
        let (o, n) = time2(reps, || { black_box(old_m.matvec_par(black_box(&x))); }, || { black_box(new_m.matvec_par(black_box(&x))); });
        report(format!("matvec_par {rows}x{cols}"), o, n);
    }
    for &(rows, cols, batch) in &[(12288usize, 4096usize, 64usize), (2048, 11008, 64), (11008, 2048, 64), (4096, 4096, 8), (4096, 4096, 3), (4096, 4096, 2)] {
        let (data, scales) = parts(&mut rng, rows, cols);
        let old_m = old::RowScaledTQ5::from_parts(rows, cols, data.clone(), scales.clone());
        let new_m = new::RowScaledTQ5::from_parts(rows, cols, data, scales);
        let xs: Vec<Vec<Fx>> = (0..batch).map(|_| activation(&mut rng, cols)).collect();
        let refs: Vec<&[Fx]> = xs.iter().map(|v| &v[..]).collect();
        let reps = if batch >= 64 { 16 } else { 80 };
        let (o, n) = time2(reps, || { black_box(old_m.matvec_batch_par(black_box(&refs))); }, || { black_box(new_m.matvec_batch_par(black_box(&refs))); });
        report(format!("matvec_batch_par {rows}x{cols} b{batch}"), o, n);
    }
}

/// Serial `matvec` against `matvec_par` of this tree, to place the size below
/// which one thread wins.
fn crossover() {
    let mut rng = Rng(0x5EED_0002);
    println!("{:<20} {:>10} {:>12} {:>12}", "shape", "KiB", "serial ns", "par ns");
    for &(rows, cols) in &[(32usize, 2048usize), (32, 4096), (64, 4096), (128, 2048), (128, 4096), (256, 2048), (256, 4096), (512, 4096), (1024, 2048), (1024, 4096)] {
        let (data, scales) = parts(&mut rng, rows, cols);
        let m = new::RowScaledTQ5::from_parts(rows, cols, data, scales);
        let x = activation(&mut rng, cols);
        let (s, p) = time2(600, || { black_box(m.matvec(black_box(&x))); }, || { black_box(m.matvec_par(black_box(&x))); });
        println!("{:<20} {:>10} {:>12.0} {:>12.0}", format!("{rows}x{cols}"), rows * cols / 1024, s, p);
    }
}

/// NVFP4 and MXFP8 beside RowScaledTQ5 at an expert's shapes. Each format
/// cycles through 16 distinct matrices so decode streams from DRAM as a
/// model's experts do (one matrix would sit in L3); the three formats are
/// interleaved call by call. Scales as in shipped checkpoints.
fn blockscaled() {
    let mut rng = Rng(0x5EED_0B5C);
    const K: usize = 16;
    println!("threads: {}", rayon::current_num_threads());
    println!("{:<40} {:>12} {:>12} {:>12} {:>7} {:>7}", "us per call (median)", "TQ5", "NVFP4", "MXFP8", "NV/TQ5", "MX/TQ5");
    for &(rows, cols) in &[(2304usize, 2304usize), (2304, 1152)] {
        let tq: Vec<new::RowScaledTQ5> = (0..K).map(|_| { let (d, s) = parts(&mut rng, rows, cols); new::RowScaledTQ5::from_parts(rows, cols, d, s) }).collect();
        let nv: Vec<new::NvFp4Matrix> = (0..K).map(|_| {
            let codes = (0..rows * cols / 2).map(|_| rng.next() as u8).collect();
            let scales = (0..rows * cols / 16).map(|_| loop { let b = (((rng.next() % 5) + 11) << 3 | (rng.next() % 8)) as u8; if b & 0x7F != 0x7F { break b; } }).collect();
            new::NvFp4Matrix::from_parts(rows, cols, codes, scales, 0x3EA0_0000 | (rng.next() as u32 & 0x7_FFFF), 0x4528_0000).unwrap()
        }).collect();
        let mx: Vec<new::MxFp8Matrix> = (0..K).map(|_| {
            let elems = (0..rows * cols).map(|_| loop { let b = rng.next() as u8; if b & 0x7F != 0x7F { break b; } }).collect();
            let scales = (0..rows * cols / 32).map(|_| (113 + rng.next() % 5) as u8).collect();
            new::MxFp8Matrix::from_parts(rows, cols, elems, scales).unwrap()
        }).collect();
        assert!(nv.iter().all(|m| m.simd_eligible()) && mx.iter().all(|m| m.simd_eligible()));
        let bpw = |bytes: usize| bytes as f64 * 8.0 / (rows * cols) as f64;
        println!("{rows}x{cols}: bits per weight TQ5 {:.2}  NVFP4 {:.2}  MXFP8 {:.2}", bpw(tq[0].size_bytes()), bpw(nv[0].size_bytes()), bpw(mx[0].size_bytes()));
        let x = activation(&mut rng, cols);
        let xs: Vec<Vec<Fx>> = (0..64).map(|_| activation(&mut rng, cols)).collect();
        let refs: Vec<&[Fx]> = xs.iter().map(|v| &v[..]).collect();
        let modes: [(&str, usize); 3] = [("matvec (1 thread)", 40), ("matvec_par", 200), ("matvec_batch_par b64", 12)];
        for (mode, reps) in modes {
            let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..reps + 3 {
                let k = i % K;
                let t = Instant::now();
                match mode { "matvec (1 thread)" => { black_box(tq[k].matvec(black_box(&x))); } "matvec_par" => { black_box(tq[k].matvec_par(black_box(&x))); } _ => { black_box(tq[k].matvec_batch_par(black_box(&refs))); } }
                let ta = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                match mode { "matvec (1 thread)" => { black_box(nv[k].matvec(black_box(&x))); } "matvec_par" => { black_box(nv[k].matvec_par(black_box(&x))); } _ => { black_box(nv[k].matvec_batch_par(black_box(&refs))); } }
                let tb = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                match mode { "matvec (1 thread)" => { black_box(mx[k].matvec(black_box(&x))); } "matvec_par" => { black_box(mx[k].matvec_par(black_box(&x))); } _ => { black_box(mx[k].matvec_batch_par(black_box(&refs))); } }
                let tc = t.elapsed().as_nanos() as f64;
                if i >= 3 { a.push(ta); b.push(tb); c.push(tc); }
            }
            let (a, b, c) = (median(a) / 1e3, median(b) / 1e3, median(c) / 1e3);
            println!("  {:<38} {:>12.1} {:>12.1} {:>12.1} {:>7.2} {:>7.2}", mode, a, b, c, b / a, c / a);
        }
    }
}
