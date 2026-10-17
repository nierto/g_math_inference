# g_math_inference

Ternary weight formats and zero-multiply inference kernels over the
fixed-point formats of [g_math](https://crates.io/crates/g_math) (0.7 and
later): TQ1.9 (ten trits per weight in an i16), the five-trit row-scaled
form, the bf16 / binary16 weight decoders and the quantisers, the
compute-tier wide-output matvecs and the rayon row-parallel forms.

Until g_math 0.6.7 this was the module `g_math::tq19` behind the
`inference` feature. The code moved here unchanged apart from its paths;
its differential gate against g_math 0.6.7 shows no difference in value,
error or panic on any of the module's kernels at six formats.

Licence: proprietary, see `LICENSE`; licensing enquiries to niels@nierto.com. The versions published inside
g_math 0.4.x to 0.6.7 remain available under MIT or Apache-2.0.

## What it is

**TQ1.9** is a
decoupled, standalone format for neural-network weights: 1 integer trit + 9
fractional trits per weight (2 bytes, range ±1.5). Because weights are {−1, 0, +1}
at the trit level, a dot product needs no multiplications. `PlanarTQ19` and
`HybridTQ19` are lossless re-encodings of a `TQ19Matrix` that trade layout for
fewer weight bytes at bit-identical matvec results. `RowScaledTQ19`
(q16_16/q32_32) carries one quantization scale per row instead of the single
global step, adapting resolution to each row's own max at unchanged
2 bytes/weight.

All four forms also expose a **wide-output matvec** (`matvec_q2f`, `_par`,
`_batch_par`; q16_16/q32_32): the exact row accumulator at 2·FRAC_BITS
fractional precision with exactly one rounding, for consumers whose signal
sits below the storage rounding floor. `q2f / (1 << FRAC_BITS)` (truncating
division) reproduces the narrow `matvec` bit-for-bit for `TQ19Matrix`,
`HybridTQ19`, and `PlanarTQ19`; `RowScaledTQ19` applies its per-row scale to
the wide dot, which is strictly more precise and may differ from the narrow
path by at most 1 storage LSB for non-unit scales.

## Usage

```rust
use g_math_inference::{TQ19Matrix, PlanarTQ19};
use g_math::fixed_point::Fixed;

type Q = Fixed<i32, 16>;
// 2x3 weight matrix; raw TQ1.9 values, |raw| <= 29524. 19683 = 3^9 = weight 1.0.
let w = TQ19Matrix::new(2, 3, vec![19683, 0, -19683, 0, 19683, 19683]);
let acts: Vec<Q> = [1, 2, 3].iter().map(|&x| Q::from_int(x)).collect();

let dense = w.matvec(&acts);

// Lossless trit-plane re-encoding: same matvec, fewer weight bytes.
let planar = PlanarTQ19::from_tq19(&w);
assert_eq!(planar.matvec(&acts), dense);
```

## What's here

- **TQ1.9** (`TQ19Matrix` with `matvec` / `matvec_batch` (and rayon `_par`
  variants), the `tq19_dot` / `trit_dot` / `packed_trit_dot` kernels (AVX2 on
  x86_64 with a scalar fallback), and the `PlanarTQ19` / `HybridTQ19` compressed
  weight forms) each bit-identical to dense `TQ19Matrix::matvec` on every profile.
  A result that does not fit the storage type panics (0.6.5; it wrapped before).
- **`RowScaledTQ19`** (realtime and compact) and **`RowScaledTQ5`** (realtime,
  0.6.5): one scale per row in unsigned Q32.32, with ten trits in an i16 or
  five trits in an i8 per weight. `RowScaledTQ5` is half the bytes per weight;
  its kernels (scalar, AVX2 on 16-bit halves of the activations, a 4x4
  register tile for batches) all compute the same integer per row.
- **`bits`** and **`quantize`** (0.6.5): binary16 / bfloat16 bit
  patterns decoded to fixed point by integer shifts, and the quantisers from
  those bits to `TQ19Matrix`, `RowScaledTQ19` and `RowScaledTQ5` by exact
  rational rounding. No float type is involved.
  `matvec_batch_par_into` (0.6.7, on `TQ19Matrix`, `PlanarTQ19`,
  `HybridTQ19`, `RowScaledTQ19` and `RowScaledTQ5`) writes a batch into one
  caller-provided buffer, flat and batch-major (`out[b * rows + r]`), with
  the same integers as `matvec_batch_par` and no result vectors allocated.
  Rows may mix values of any magnitude: an element too small to reach half a
  step of its row's scale quantises to 0 (0.6.6; a gap of a little over 100 binary
  orders inside one bfloat16 row overflowed in 0.6.5).

## Behaviour & limits

- The kernels are standalone: they use only `Fixed`, its storage and compute
  words and `compute_tier` from g_math, never its routing or canonical layers.
- `TQ19Matrix` raw values must satisfy |raw| ≤ 29524 (the 10-digit balanced-ternary
  range); `PlanarTQ19::from_tq19` rejects out-of-range weights.
- `PlanarTQ19` / `HybridTQ19` are lossless: their matvec equals dense matvec
  bit-for-bit; they change only the weight byte layout.

## Versions

Each release names the g_math version it was verified against (see
CHANGELOG.md). 0.1.0 depends on `g_math = "0.7"`.

## Tests

`cargo test --release` (unit, integration, doc); the differential gate
against g_math 0.6.7: `cd differential && GMATH_PROFILE=realtime
GMATH_FRAC_BITS=10 cargo test --release` (and the other profiles).
