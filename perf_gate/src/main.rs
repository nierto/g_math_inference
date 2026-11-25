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
    if std::env::args().nth(1).as_deref() == Some("rowscaled19") { rowscaled19(); return; }
    if std::env::args().nth(1).as_deref() == Some("ptq1") { ptq1(); return; }
    if std::env::args().nth(1).as_deref() == Some("i16") { i16_gemv(); return; }
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
        // (label, batch size: 0 = matvec on one thread, 1 = matvec_par, k = matvec_batch_par with k activations)
        let modes: [(&str, usize, usize); 8] = [("matvec (1 thread)", 0, 40), ("matvec_par", 1, 200), ("batch 2", 2, 100), ("batch 3", 3, 100), ("batch 4", 4, 100), ("batch 8", 8, 60), ("batch 16", 16, 40), ("batch 64", 64, 12)];
        for (mode, bs, reps) in modes {
            let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
            let sub = &refs[..bs.max(1)];
            for i in 0..reps + 3 {
                let k = i % K;
                let t = Instant::now();
                match bs { 0 => { black_box(tq[k].matvec(black_box(&x))); } 1 => { black_box(tq[k].matvec_par(black_box(&x))); } _ => { black_box(tq[k].matvec_batch_par(black_box(sub))); } }
                let ta = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                match bs { 0 => { black_box(nv[k].matvec(black_box(&x))); } 1 => { black_box(nv[k].matvec_par(black_box(&x))); } _ => { black_box(nv[k].matvec_batch_par(black_box(sub))); } }
                let tb = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                match bs { 0 => { black_box(mx[k].matvec(black_box(&x))); } 1 => { black_box(mx[k].matvec_par(black_box(&x))); } _ => { black_box(mx[k].matvec_batch_par(black_box(sub))); } }
                let tc = t.elapsed().as_nanos() as f64;
                if i >= 3 { a.push(ta); b.push(tb); c.push(tc); }
            }
            let (a, b, c) = (median(a) / 1e3, median(b) / 1e3, median(c) / 1e3);
            println!("  {:<38} {:>12.1} {:>12.1} {:>12.1} {:>7.2} {:>7.2}", mode, a, b, c, b / a, c / a);
        }
    }
}

fn tq19_parts(rng: &mut Rng, rows: usize, cols: usize) -> (Vec<i16>, Vec<u64>) {
    let data = (0..rows * cols).map(|_| ((rng.next() % 59049) as i64 - 29524) as i16).collect();
    let scales = (0..rows).map(|_| (1u64 << 28) + rng.next() % (1u64 << 33)).collect();
    (data, scales)
}

/// RowScaledTQ19 matvec_batch_par, the published 0.1.2 against this tree, B = 1..8,
/// at the attention and LM-head shapes; K distinct matrices cycled (DRAM).
fn rowscaled19() {
    let mut rng = Rng(0x0019_B47C);
    println!("threads: {}", rayon::current_num_threads());
    println!("{:<34} {:>12} {:>12} {:>7}", "us per call (median)", "0.1.2", "new", "new/old");
    for &(name, rows, cols, k) in &[("q/gate 3072x2304", 3072usize, 2304usize, 8usize), ("k/v 1024x2304", 1024, 2304, 16), ("o 2304x3072", 2304, 3072, 8), ("wr 384x2304", 384, 2304, 32), ("LM head 200058x2304", 200058, 2304, 1)] {
        let mats: Vec<(old::RowScaledTQ19, new::RowScaledTQ19)> = (0..k)
            .map(|_| {
                let (d, s) = tq19_parts(&mut rng, rows, cols);
                (old::RowScaledTQ19::from_parts(rows, cols, d.clone(), s.clone()), new::RowScaledTQ19::from_parts(rows, cols, d, s))
            })
            .collect();
        let xs: Vec<Vec<Fx>> = (0..8).map(|_| activation(&mut rng, cols)).collect();
        let refs: Vec<&[Fx]> = xs.iter().map(|v| &v[..]).collect();
        println!("{name}:");
        let reps = if rows > 100_000 { 9 } else { 60 };
        {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for i in 0..reps + 3 {
                let (o, n) = &mats[i % k];
                let t = Instant::now();
                black_box(o.matvec_par(black_box(&xs[0])));
                let ta = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                black_box(n.matvec_par(black_box(&xs[0])));
                let tb = t.elapsed().as_nanos() as f64;
                if i >= 3 { a.push(ta); b.push(tb); }
            }
            let (a, b) = (median(a) / 1e3, median(b) / 1e3);
            println!("  {:<32} {:>12.1} {:>12.1} {:>7.2}", "matvec_par (B = 1)", a, b, b / a);
        }
        for bs in 1..=8 {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let sub = &refs[..bs];
            for i in 0..reps + 3 {
                let (o, n) = &mats[i % k];
                let t = Instant::now();
                black_box(o.matvec_batch_par(black_box(sub)));
                let ta = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                black_box(n.matvec_batch_par(black_box(sub)));
                let tb = t.elapsed().as_nanos() as f64;
                if i >= 3 { a.push(ta); b.push(tb); }
            }
            let (a, b) = (median(a) / 1e3, median(b) / 1e3);
            println!("  {:<32} {:>12.1} {:>12.1} {:>7.2}", format!("matvec_batch_par B = {bs}"), a, b, b / a);
        }
    }
}

/// Ptq1Matrix (with the 1024 Walsh-Hadamard transform and signs) beside
/// RowScaledTQ5 at a 27B model's shapes, B = 1, 8, 64, 512.
fn ptq1() {
    let mut rng = Rng(0x0F71_0001);
    println!("threads: {}", rayon::current_num_threads());
    println!("{:<34} {:>12} {:>12} {:>9}", "us per call (median)", "TQ5", "PTQ1", "PTQ1/TQ5");
    for &(rows, cols) in &[(5120usize, 5120usize), (17408, 5120), (5120, 17408)] {
        let k = if rows * cols > 50_000_000 { 2 } else { 6 };
        let tq: Vec<new::RowScaledTQ5> = (0..k).map(|_| { let (d, s) = parts(&mut rng, rows, cols); new::RowScaledTQ5::from_parts(rows, cols, d, s) }).collect();
        let pt: Vec<new::Ptq1Matrix> = (0..k)
            .map(|_| {
                let mut blocks = vec![0u8; rows * cols / 128 * 28];
                for b in blocks.chunks_exact_mut(28) {
                    for v in b.iter_mut().take(26) { *v = ((rng.next() % 243) * 256).div_ceil(243) as u8; }
                    let h: u16 = (((rng.next() % 4) + 6) << 10) as u16 | (rng.next() & 1023) as u16; // 2^-9 .. 2^-6
                    b[26..28].copy_from_slice(&h.to_le_bytes());
                }
                let signs: Vec<i8> = (0..cols).map(|_| if rng.next() & 1 == 0 { 1 } else { -1 }).collect();
                new::Ptq1Matrix::from_parts(rows, cols, blocks).unwrap().with_hadamard(1024, Some(signs)).unwrap()
            })
            .collect();
        println!("{rows}x{cols}: bytes TQ5 {:.1} MB, PTQ1 {:.1} MB", tq[0].size_bytes() as f64 / 1e6, pt[0].size_bytes() as f64 / 1e6);
        let xs: Vec<Vec<Fx>> = (0..512).map(|_| activation(&mut rng, cols)).collect();
        let refs: Vec<&[Fx]> = xs.iter().map(|v| &v[..]).collect();
        for &(bs, reps) in &[(1usize, 40usize), (8, 20), (64, 6), (512, 3)] {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let sub = &refs[..bs];
            for i in 0..reps + 1 {
                let j = i % k;
                let t = Instant::now();
                if bs == 1 { black_box(tq[j].matvec_par(black_box(&xs[0]))); } else { black_box(tq[j].matvec_batch_par(black_box(sub))); }
                let ta = t.elapsed().as_nanos() as f64;
                let t = Instant::now();
                if bs == 1 { black_box(pt[j].matvec_par(black_box(&xs[0]))); } else { black_box(pt[j].matvec_batch_par(black_box(sub))); }
                let tb = t.elapsed().as_nanos() as f64;
                if i >= 1 { a.push(ta); b.push(tb); }
            }
            let (a, b) = (median(a) / 1e3, median(b) / 1e3);
            println!("  {:<32} {:>12.1} {:>12.1} {:>9.2}", if bs == 1 { "matvec_par (B = 1)".to_string() } else { format!("matvec_batch_par B = {bs}") }, a, b, b / a);
        }
    }
}

/// I16Matrix::matvec_i64_par at the folded draft head's shape, 200058 x 256.
fn i16_gemv() {
    let mut rng = Rng(0x0016_0256);
    let (rows, cols) = (200058usize, 256usize);
    let m = new::I16Matrix::from_parts(rows, cols, (0..rows * cols).map(|_| ((rng.next() % 16385) as i32 - 8192) as i16).collect());
    let t: Vec<i16> = (0..cols).map(|_| ((rng.next() % 65535) as i32 - 32767) as i16).collect();
    let mut v = Vec::new();
    for i in 0..23 {
        let s = Instant::now();
        black_box(m.matvec_i64_par(black_box(&t)));
        if i >= 3 { v.push(s.elapsed().as_nanos() as f64); }
    }
    println!("threads: {}  I16Matrix {rows}x{cols} ({:.1} MB) matvec_i64_par: {:.0} us median", rayon::current_num_threads(), m.size_bytes() as f64 / 1e6, median(v) / 1e3);
}
