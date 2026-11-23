# Changelog

All notable changes to g_math_inference are documented in this file. Each
release names the g_math version it was verified against.

## [0.1.5] - 2026-11-24

Every released result is unchanged unless a matrix is switched to
`Rounding::Nearest`. Verified against g_math 0.7.3 to 0.7.5.

### Added

- **`Rounding { Default, Nearest }`** on `RowScaledTQ5`, `RowScaledTQ19`,
  `TQ19Matrix`, `PlanarTQ19`, `HybridTQ19`, `NvFp4Matrix`, `MxFp8Matrix`
  (`with_rounding`, `rounding`; `from_tq19` repacks keep the mode).
  `Default` is the released rounding (floor for the row- and block-scaled
  forms, toward zero for the TQ1.9 forms, two roundings in
  `RowScaledTQ19`). `Nearest` rounds the exact value once to nearest, ties
  toward +infinity, in every form (par, batch, batch_into, q2f): no half-unit
  bias per output. The same speed.
- **`matvec_wide_act(_par)`** on `NvFp4Matrix`, `MxFp8Matrix`,
  `RowScaledTQ5`: activations at Q32.32 or Q64.64 (any i64 / i128 split),
  the exact value rounded once at that format. The activations run as
  exact 30-bit limbs through the 32-bit kernels (all-zero limbs skipped),
  combined at 512 bits: a precision reference path. The i128 word gains the
  wide epilogues, so `RowScaledTQ19` and the TQ1.9 `q2f` forms run at
  Q64.64 through their generic methods.
- **Activation quantisation, exact** (`quantize_nvfp4`, `quantize_mxfp8`,
  `Nvfp4Activation`, `Mxfp8Activation`, and `matvec_quantized` on
  `NvFp4Matrix` / `MxFp8Matrix`): the rule block-scaled checkpoints declare
  for expert inputs (NVFP4: per-row `amax / 1536`, four-over-six block
  scales chosen by the exact sum of absolute errors, a tie keeps 6, E2M1
  and E4M3 nearest-even; MXFP8: the smallest power of two keeping every
  element within 448), evaluated exactly in integers. The runtimes evaluate
  it in binary32 and disagree on the operation order; this is the rule
  itself, no float. `matvec_quantized` forms the exact sum over both
  quantised operands, rounded once: a reference path for evaluating the
  rule's effect on a model.

### Changed

- `matvec_batch_par(_into)` on `NvFp4Matrix` with few activations (up to 8
  activation halves, the speculative-checking sizes) decodes each row once
  in registers for all of them instead of through the scratch tile: the
  same results; batch 2 at 2304 x 2304 287 -> ~240 us.

### Validation

- `tests/rounding_wide_validation.rs`: every form under `Nearest` and the
  wide paths against 512-bit references from the stored weights; widened
  32-bit activations equal to `matvec`. `tests/blockscaled_validation.rs`:
  the Fraction references gain nearest values, every form checked under
  both modes. `tests/actquant_validation.rs`: codes, scales and quantised
  matvecs equal an independent Fraction implementation of the rule (48
  rows; 137 blocks choose 4, 59 error ties keep 6, zero and subnormal block
  scales reached; flipping the tie rule fails the gate).

## [0.1.4] - 2026-11-23

Block-scaled weights computed on directly, exactly, in integers. Nothing
already shipped changes: every existing kernel returns the same integers.
Verified against g_math 0.7.3 and 0.7.4.

### Added

- `NvFp4Matrix`: four-bit E2M1 codes, an E4M3 scale per 16 columns and one
  `amax / den` factor per matrix, at the stored 4.5 bits per weight.
  `value = E2M1(code) * E4M3(scale) * amax / den`; the factor is held as an
  exact fraction (`den` is not a power of two in shipped checkpoints), so a
  row costs one exact floor division at the end, by a precomputed
  reciprocal. Codes are regrouped once at construction (same bytes) so one
  load gives sixteen codes in column order; `to_codes` returns the stored
  layout.
- `MxFp8Matrix`: E4M3 elements and a UE8M0 scale per 32 columns (8.25 bits
  per weight). Single rows run in 16-bit exponent classes with the block
  shift folded into a table; batches decode each row once into two 16-bit
  limbs per weight.
- Both: `matvec`, `matvec_par`, `matvec_batch_par(_into)`, `matvec_q2f(_par)`
  with the signatures and output rounding of `RowScaledTQ5`
  (`floor(sum value * x)`, panic past the word), `dequant_row_exact`,
  `simd_eligible`. Any matrix outside the AVX2 bounds (block scales far
  apart) takes the exact scalar path (i128, or I512 for MXFP8 scale spreads
  beyond 52).
- `bits`: `E2M1_X2` / `e2m1_x2`, `decompose_e4m3`, `ue8m0_exponent`,
  `decompose_f32`, as integers and powers of two (no float type).

### Validation

- `tests/blockscaled_validation.rs`: the decoders against Fraction values
  from the format definitions for every pattern; 229 reference rows
  (scripts/generate_blockscaled_refs.py, Fraction arithmetic, no floats) at
  subnormal, zero, negative and largest scales, UE8M0 0 and 254, amax/den of
  every kind and |x| up to 2^31, every form and each row alone (value or
  panic); a random differential at 2304 x 2304 and 2304 x 1152 with batches
  1 to 9 and 64 at 1, 2, 3 and 8 threads, every SIMD class count and the
  scalar paths: 3.3 million row outputs, all exact. With `GMI_REAL_SLICES`,
  real checkpoint slices against exact outputs
  (scripts/generate_blockscaled_real_refs.py). Mutations of the kernels
  (nibble order, class tables, limb shift and order, floor division) each
  fail the gate.

### Speed

i7-7700, 4 cores / 8 threads, one of 16 matrices per call (DRAM, as a
model's experts), us per call, against `RowScaledTQ5` at the same shape:

| 2304 x 2304 | TQ5 | NVFP4 | MXFP8 |
|---|---|---|---|
| bits per weight | 8.03 | 4.50 | 8.25 |
| matvec, 1 thread | 496 | 518 | 2180 |
| matvec_par | 210 | 149 | 529 |
| matvec_batch_par, 64 | 4673 | 5403 | 10050 |

On these 4 cores NVFP4 decode is compute-bound (the shuffle port decodes
two codes per byte); with more cores it reaches the bandwidth bound at
about 0.56 of TQ5's bytes. MXFP8's 18-bit elements need several 16-bit
multiply-adds each.

## [0.1.3] - 2026-11-22

Speed only; every result is the same integer as in 0.1.2. Verified against
g_math 0.7.2 and 0.7.3.

### Changed

- `RowScaledTQ5::matvec_batch_par` (and `_into`): a 3-row by 4-token register
  tile replaces the 4 x 4 one, whose 16 accumulators, 4 widened weight
  vectors and one activation (21) exceeded AVX2's 16 vector registers and
  spilled on every step. The 1 to 3 tokens left over in a chunk run a
  3 x 1..3 tile instead of one dot per row and token, which is what batches
  of 2 and 3 take. Batch 64: 12 to 21 percent faster; batches of 2, 3 and 8:
  10 to 22 percent (i7-7700, 8 threads).
- `RowScaledTQ5::matvec_par` and `matvec_q2f_par`: below 256 KiB of weights
  the calling thread does the work (a fork/join across the pool cost more
  than the rows), and parallel tasks take at least 64 KiB of weights each.

### Validation

- `perf_gate/` (not packaged): every RowScaledTQ5 entry point against the
  published 0.1.2 on shapes that reach every path (rows and columns on and
  off the tile and SIMD widths, batches 1 to 9 and 64, activations with
  sparse, dense and unsplittable high halves, scales that reach the overflow
  panic), at Q22.10 and Q16.16: 6912 outcomes, panics included, all equal.

## [0.1.2] - 2026-10-17

Licence only; the code is identical to 0.1.0 and 0.1.1. Verified against
g_math 0.7.0.

### Licence

- Source-available under PolyForm Noncommercial 1.0.0 (`LICENSE.md`);
  `Cargo.toml` declares `PolyForm-Noncommercial-1.0.0`. This replaces the
  proprietary `LICENSE` of 0.1.0 and 0.1.1.
- `MANIFERENCE-GRANT.md`: everyone licensed for maniference (PolyForm
  Noncommercial, PolyForm Small Business or a commercial licence) may use
  g_math_inference as part of maniference on the same terms. Use apart from
  maniference, or to build a competing inference engine, is not covered.
- `COMMERCIAL.md`: component licences at one fifth of the maniference
  price, a perpetual licence at 3x, a field-of-use limit, and a competitor
  licence by agreement.

## [0.1.1] - 2026-10-17

Licence only; the code is identical to 0.1.0. Verified against g_math 0.7.0.

### Licence

- `LICENSE` grants a free 30-day evaluation for non-production use (0.1.0
  granted no use without a licence agreement). Production use and
  redistribution still require a licence agreement; enquiries to
  niels@nierto.com.

## [0.1.0] - 2026-10-17

Verified against g_math 0.7.0.

### Added

- The ternary inference kernels, formerly the module `g_math::tq19` behind
  the `inference` feature of g_math (up to 0.6.8), as their own crate:
  `TQ19Matrix`, `PlanarTQ19`, `HybridTQ19`, `RowScaledTQ19`, `RowScaledTQ5`;
  `tq19_dot`, `tq19_dot_q2f`, `trit_dot`, `packed_trit_dot`, the matvec
  family (plain, `_par`, `_batch`, `_batch_par`, `_batch_par_into`,
  `_q2f`, `_q2f_par`, `_q2f_batch_par`) and `packed_trit_matvec(_par)`;
  `TqWord`, `WideOutput`; the weight decoders (`bits`) and the quantisers
  (`quantize`); the AVX2 kernels.
- Every result is bit-identical to g_math 0.6.7's module: the differential
  gate compares each kernel with g_math 0.6.7 at its six builds (realtime
  F = 10 and 16, compact, embedded, balanced, scientific) and finds no
  difference in value, error or panic.

### Changed (from g_math 0.6.x's module)

- The kernels take activation slices of a format, `&[Fixed<S, F>]`, instead
  of raw integers of the build profile; `Fixed::from_raw_slice` /
  `raw_slice` convert without copying. The wide outputs are `Compute<T>`.
- Paths: `g_math::tq19::X` is `g_math_inference::X`.

### Licence

Proprietary (see `LICENSE`), with a free 30-day evaluation for non-production use; licensing enquiries to niels@nierto.com. The versions of this code published inside
g_math 0.4.x to 0.6.8 remain available under MIT or Apache-2.0.
