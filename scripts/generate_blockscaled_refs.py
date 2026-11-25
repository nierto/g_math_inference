#!/usr/bin/env python3
"""Exact references for the block-scaled kernels (NVFP4, MXFP8).

Every pattern is decoded from the format definitions (OCP Microscaling
spec: E2M1, E4M3 "fn", UE8M0; IEEE 754 binary32) with Fraction arithmetic,
independently of the crate's integer decoders: no float type anywhere.
Outputs are floor(sum_c value(r, c) * x[c]) at the activations' raw scale
(the F fractional bits cancel) and floor(... * 2^F) for the wide form.

Every output is computed a second time by mpmath at 120 digits, from
independent mpmath decoders (ldexp of the integer fields); the generator
asserts the decoders agree on every pattern and the outputs agree (the floor
of a value can differ only where the exact value is an integer, which the
Fraction value then shows).

Writes tests/data/blockscaled_refs.rs.
"""
from fractions import Fraction as Fr
import math
import os
from mpmath import mp, mpf, ldexp, floor as mfloor, fsum

mp.dps = 120
MP_AGREED = [0]

TWO = Fr(2)


def e2m1(code):
    s, e, m = code >> 3, (code >> 1) & 3, code & 1
    # bias 1: subnormal (e = 0) is m/2 * 2^0, normal is (1 + m/2) * 2^(e - 1)
    v = Fr(m, 2) if e == 0 else (1 + Fr(m, 2)) * TWO ** (e - 1)
    return -v if s else v


def e4m3(b):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return None  # NaN (no infinities in E4M3 "fn")
    # bias 7: subnormal (e = 0) is m/8 * 2^-6, normal is (1 + m/8) * 2^(e - 7)
    v = Fr(m, 8) * TWO ** -6 if e == 0 else (1 + Fr(m, 8)) * TWO ** (e - 7)
    return -v if s else v


def ue8m0(b):
    return None if b == 255 else TWO ** (b - 127)


def f32(bits):
    s, e, m = bits >> 31, (bits >> 23) & 255, bits & ((1 << 23) - 1)
    if e == 255:
        return None
    v = Fr(m, 1 << 23) * TWO ** -126 if e == 0 else (1 + Fr(m, 1 << 23)) * TWO ** (e - 127)
    return -v if s else v


def e2m1_m(code):
    s, e, m = code >> 3, (code >> 1) & 3, code & 1
    v = ldexp(mpf(m), -1) if e == 0 else ldexp(mpf(2 + m), e - 2)
    return -v if s else v


def e4m3_m(b):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return None
    v = ldexp(mpf(m), -9) if e == 0 else ldexp(mpf(8 + m), e - 10)
    return -v if s else v


def ue8m0_m(b):
    return None if b == 255 else ldexp(mpf(1), b - 127)


def f32_m(bits):
    s, e, m = bits >> 31, (bits >> 23) & 255, bits & ((1 << 23) - 1)
    if e == 255:
        return None
    v = ldexp(mpf(m), -149) if e == 0 else ldexp(mpf((1 << 23) + m), e - 150)
    return -v if s else v


def mpq(fr):
    return mpf(fr.numerator) / fr.denominator


# the mpmath decoders equal the Fraction decoders on every pattern
for c in range(16):
    assert e2m1_m(c) == mpq(e2m1(c))
for b in range(256):
    assert (e4m3_m(b) is None) == (e4m3(b) is None) and (e4m3(b) is None or e4m3_m(b) == mpq(e4m3(b)))
    assert (ue8m0_m(b) is None) == (ue8m0(b) is None) and (ue8m0(b) is None or ue8m0_m(b) == mpq(ue8m0(b)))


def mp_check(vals_m, x, frac, outs):
    """The four outputs again by mpmath from the mpmath values; a floor may differ
    only where the exact value is an integer."""
    n, w, nn, wn = outs
    half = mpf(1) / 2
    for r, row in enumerate(vals_m):
        sm = fsum(wv * xi for wv, xi in zip(row, x))
        for got, v in ((n[r], sm), (w[r], ldexp(sm, frac)), (nn[r], sm + half), (wn[r], ldexp(sm, frac) + half)):
            if got is None:
                continue
            f = int(mfloor(v))
            if f != got:
                # the value is (within 10^-100) the integer got + 1 or got: an exact boundary
                assert abs(v - mpf(got)) < mpf(10) ** -100 or abs(v - mpf(got + 1)) < mpf(10) ** -100, (got, v)
            MP_AGREED[0] += 1


# anchors from the spec text
assert [e2m1(c) for c in range(8)] == [0, Fr(1, 2), 1, Fr(3, 2), 2, 3, 4, 6]
assert e4m3(0x7E) == 448 and e4m3(0xFE) == -448
assert e4m3(0x01) == TWO ** -9 and e4m3(0x08) == TWO ** -6 and e4m3(0x07) == Fr(7, 8) * TWO ** -6
assert e4m3(0x7F) is None and e4m3(0xFF) is None and e4m3(0x38) == 1
assert ue8m0(127) == 1 and ue8m0(0) == TWO ** -127 and ue8m0(254) == TWO ** 127
assert f32(0x45280000) == 2688 and f32(0x3F800000) == 1 and f32(0x00000001) == TWO ** -149


class Rng:
    def __init__(self, s):
        self.s = s & ((1 << 64) - 1)

    def next(self):
        s = self.s
        s ^= s >> 12
        s ^= (s << 25) & ((1 << 64) - 1)
        s ^= s >> 27
        self.s = s
        return (s * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

    def below(self, n):
        return self.next() % n


I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def activation(rng, n, kind):
    out = []
    for _ in range(n):
        r = rng.next()
        bits = {0: 14, 1: 24 if r % 10 == 0 else 14, 2: 24, 3: 31 if r % 40 == 0 else 20, 4: 31}[kind]
        m = (r >> 8) & ((1 << bits) - 1)
        v = -m if r & 1 else m
        out.append(max(I32_MIN, min(I32_MAX, v)))
    if kind == 4 and n >= 2:
        out[0], out[1] = I32_MIN, I32_MAX
    return out


I128_MIN, I128_MAX = -(1 << 127), (1 << 127) - 1


def outputs(values_rows, x, frac):
    """Per row: floor and nearest (ties toward +inf) of sum w x and of sum w x * 2^F,
    exactly; None where beyond i128."""
    narrow, wide, nnear, wnear = [], [], [], []
    half = Fr(1, 2)
    for row in values_rows:
        s = sum((w * xi for w, xi in zip(row, x)), Fr(0))
        for out, v in ((narrow, math.floor(s)), (wide, math.floor(s * TWO ** frac)),
                       (nnear, math.floor(s + half)), (wnear, math.floor(s * TWO ** frac + half))):
            out.append(v if I128_MIN <= v <= I128_MAX else None)
    return narrow, wide, nnear, wnear


def f32_bits(neg, mant24, exp_field):
    return (neg << 31) | (exp_field << 23) | (mant24 & ((1 << 23) - 1))


def nv_scale(rng, mode):
    if mode == "real":  # exponent fields 11..15, positive, as in shipped checkpoints
        while True:
            b = (rng.below(5) + 11) << 3 | rng.below(8)
            if e4m3(b) is not None:
                return b
    if mode == "wide":  # any non-NaN pattern: signs, subnormals, zeros
        while True:
            b = rng.below(256)
            if e4m3(b) is not None:
                return b
    if mode == "edge":
        return [0x7E, 0x01, 0x00, 0x80, 0x08, 0xFE, 0x07, 0x38][rng.below(8)]
    if mode == "zero":
        return 0x00
    raise ValueError(mode)


def nv_cases(rng):
    cases = []
    factors = [
        (f32_bits(0, 0x800000 | 0x2A3F11, 101), 0x45280000),  # amax ~ 0.33, den 2688
        (f32_bits(0, 0xFFFFFF, 103), 0x45280000),
        (0x3F800000, 0x45800000),  # 1 / 4096: a power-of-two denominator
        (f32_bits(0, 0x800001, 127), 0x45400000),  # / 3072
        (f32_bits(0, 0x923457, 120), 0x4B7FFFFF),  # / 16777215
        (f32_bits(1, 0xA00000, 126), 0x45280000),  # negative amax
        (0x00000001, 0x45280000),  # subnormal amax
        (f32_bits(0, 0xC00000, 160), 0x3F800000),  # large factor: overflow panics
        (0x00000000, 0x45280000),  # zero amax
        (f32_bits(0, 0x800000, 127), 0xC5280000),  # negative den
    ]
    shapes = [(1, 16), (3, 32), (4, 48), (5, 64), (7, 128), (2, 256), (9, 80)]
    k = 0
    for mode in ["real", "wide", "edge", "zero"]:
        for fi, (amax, den) in enumerate(factors):
            if mode == "zero" and fi > 1:
                continue
            rows, cols = shapes[k % len(shapes)]
            kind = [0, 1, 2, 3, 4][k % 5]
            frac = [10, 16][k % 2]
            k += 1
            codes = [rng.below(256) for _ in range(rows * cols // 2)]
            scales = [nv_scale(rng, mode) for _ in range(rows * cols // 16)]
            x = activation(rng, cols, kind)
            fac = f32(amax) / f32(den)
            vals = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    byte = codes[(r * cols + c) // 2]
                    code = byte & 15 if c % 2 == 0 else byte >> 4
                    row.append(e2m1(code) * e4m3(scales[(r * cols + c) // 16]) * fac)
                vals.append(row)
            n, w, nn, wn = outputs(vals, x, frac)
            fac_m = f32_m(amax) / f32_m(den)
            vals_m = []
            for r in range(rows):
                rm = []
                for c in range(cols):
                    byte = codes[(r * cols + c) // 2]
                    code = byte & 15 if c % 2 == 0 else byte >> 4
                    rm.append(e2m1_m(code) * e4m3_m(scales[(r * cols + c) // 16]) * fac_m)
                vals_m.append(rm)
            mp_check(vals_m, x, frac, (n, w, nn, wn))
            cases.append(dict(rows=rows, cols=cols, a=codes, s=scales, amax=amax, den=den, frac=frac, x=x, out=n, q2f=w, near=nn, q2f_near=wn, mode=mode))
    return cases


def mx_scale(rng, mode, base):
    if mode == "real":
        return base + rng.below(5)
    if mode == "spread40":  # i128 scalar path
        return 70 + rng.below(41)
    if mode == "spread60":  # wider than the i128 path: I512
        return 50 + rng.below(61)
    if mode == "extreme":  # the smallest scales beside ordinary ones
        return [0, 1, 90, 100, 113][rng.below(5)]
    if mode == "overflow":
        return [254, 113][rng.below(2)]
    raise ValueError(mode)


def mx_elem(rng, mode):
    while True:
        b = rng.below(256)
        if mode == "subnormal" and rng.below(3) == 0:
            b = (b & 0x80) | rng.below(8)
        if e4m3(b) is not None:
            return b


def mx_cases(rng):
    cases = []
    shapes = [(1, 32), (3, 64), (4, 96), (6, 128), (2, 256), (5, 160)]
    k = 0
    for smode in ["real", "spread40", "spread60", "extreme", "overflow"]:
        for emode in ["any", "subnormal", "zeroblocks"]:
            for _ in range(1 if smode == "overflow" else 2):
                rows, cols = shapes[k % len(shapes)]
                kind = [0, 1, 2, 3, 4][k % 5]
                frac = [10, 16][k % 2]
                base = [113, 100, 127, 140][k % 4]
                k += 1
                elems = [mx_elem(rng, emode) for _ in range(rows * cols)]
                if emode == "zeroblocks":
                    for blk in range(rows * cols // 32):
                        if rng.below(2) == 0:
                            for i in range(32):
                                elems[blk * 32 + i] = [0x00, 0x80][rng.below(2)]
                scales = [mx_scale(rng, smode, base) for _ in range(rows * cols // 32)]
                if emode == "zeroblocks":
                    # an all-zero block may carry any scale, even far outside the others
                    for blk in range(rows * cols // 32):
                        if all(e & 0x7F == 0 for e in elems[blk * 32:(blk + 1) * 32]) and rng.below(2) == 0:
                            scales[blk] = [254, 0][rng.below(2)]
                x = activation(rng, cols, kind)
                vals = [[e4m3(elems[r * cols + c]) * ue8m0(scales[(r * cols + c) // 32]) for c in range(cols)] for r in range(rows)]
                n, w, nn, wn = outputs(vals, x, frac)
                vals_m = [[e4m3_m(elems[r * cols + c]) * ue8m0_m(scales[(r * cols + c) // 32]) for c in range(cols)] for r in range(rows)]
                mp_check(vals_m, x, frac, (n, w, nn, wn))
                cases.append(dict(rows=rows, cols=cols, a=elems, s=scales, frac=frac, x=x, out=n, q2f=w, near=nn, q2f_near=wn, mode=smode + "/" + emode))
    return cases


def lst(v):
    return "&[" + ", ".join(str(i) for i in v) + "]"


def opts(v):
    return "&[" + ", ".join("None" if i is None else f"Some({i})" for i in v) + "]"


def main():
    rng = Rng(0x0B10C5CA1ED)
    nv, mx = nv_cases(rng), mx_cases(rng)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "tests", "data", "blockscaled_refs.rs")
    with open(out, "w") as f:
        f.write("// Generated by scripts/generate_blockscaled_refs.py: Fraction arithmetic from the format definitions, no floats.\n")
        f.write("// Do not edit.\n\n")
        f.write("pub struct NvCase { pub rows: usize, pub cols: usize, pub codes: &'static [u8], pub scales: &'static [u8], pub amax: u32, pub den: u32, pub frac: u32, pub x: &'static [i32], pub out: &'static [Option<i128>], pub q2f: &'static [Option<i128>], pub near: &'static [Option<i128>], pub q2f_near: &'static [Option<i128>], pub mode: &'static str }\n")
        f.write("pub struct MxCase { pub rows: usize, pub cols: usize, pub elems: &'static [u8], pub scales: &'static [u8], pub frac: u32, pub x: &'static [i32], pub out: &'static [Option<i128>], pub q2f: &'static [Option<i128>], pub near: &'static [Option<i128>], pub q2f_near: &'static [Option<i128>], pub mode: &'static str }\n\n")
        # decoder tables: value * 2^10 for E2M1 (exact integers), E4M3 as (num, den) reduced, None for NaN
        f.write("pub const E2M1_TIMES_2: [i64; 16] = [" + ", ".join(str(int(e2m1(c) * 2)) for c in range(16)) + "];\n")
        f.write("/// E4M3 value * 2^9 (an exact integer for every pattern); None for NaN.\n")
        f.write("pub const E4M3_TIMES_512: [Option<i64>; 256] = [" + ", ".join("None" if e4m3(b) is None else f"Some({int(e4m3(b) * 512)})" for b in range(256)) + "];\n")
        f.write("/// binary32 patterns and their exact values as (numerator, power of two): value = num * 2^pow.\n")
        f32_samples = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x45280000, 0x4B7FFFFF, 0x7F7FFFFF, 0xC0490FDB, 0x3EAAAAAB]
        rows = []
        for b in f32_samples:
            v = f32(b)
            # v = n * 2^p exactly with n odd (or zero)
            n, d = v.numerator, v.denominator
            k = d.bit_length() - 1
            assert d == 1 << k
            p = -k
            while n != 0 and n % 2 == 0:
                n //= 2
                p += 1
            rows.append(f"({b:#010x}, {n}, {p if n else 0})")
        f.write("pub const F32_EXACT: [(u32, i128, i32); %d] = [%s];\n\n" % (len(rows), ", ".join(rows)))
        f.write("pub const NV_CASES: &[NvCase] = &[\n")
        for c in nv:
            f.write(f"    NvCase {{ rows: {c['rows']}, cols: {c['cols']}, codes: {lst(c['a'])}, scales: {lst(c['s'])}, amax: {c['amax']:#010x}, den: {c['den']:#010x}, frac: {c['frac']}, x: {lst(c['x'])}, out: {opts(c['out'])}, q2f: {opts(c['q2f'])}, near: {opts(c['near'])}, q2f_near: {opts(c['q2f_near'])}, mode: \"{c['mode']}\" }},\n")
        f.write("];\n\npub const MX_CASES: &[MxCase] = &[\n")
        for c in mx:
            f.write(f"    MxCase {{ rows: {c['rows']}, cols: {c['cols']}, elems: {lst(c['a'])}, scales: {lst(c['s'])}, frac: {c['frac']}, x: {lst(c['x'])}, out: {opts(c['out'])}, q2f: {opts(c['q2f'])}, near: {opts(c['near'])}, q2f_near: {opts(c['q2f_near'])}, mode: \"{c['mode']}\" }},\n")
        f.write("];\n")
    rows = [v for c in nv + mx for v in c["out"]]
    over = sum(v is None or not (I32_MIN <= v <= I32_MAX) for v in rows)
    print(f"mpmath agreed on {MP_AGREED[0]} outputs; {len(nv)} NVFP4 cases, {len(mx)} MXFP8 cases, {len(rows)} rows, {over} rows leave i32 (expect a panic); wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
