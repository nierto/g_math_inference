# Changelog

All notable changes to g_math_inference are documented in this file. Each
release names the g_math version it was verified against.

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
