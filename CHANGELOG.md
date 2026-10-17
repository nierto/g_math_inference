# Changelog

All notable changes to g_math_inference are documented in this file. Each
release names the g_math version it was verified against.

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
