#!/usr/bin/env python3
"""References for activation quantisation (src/actquant.rs), Fraction arithmetic.

The rule (NVFP4 four-over-six per token, MXFP8 ceil exponent) evaluated exactly,
written independently of the Rust code: E4M3 / E2M1 nearest-even by search over
the value grids built from the format definitions; the matvecs on the quantised
operands as exact sums. No floats. Writes tests/data/actquant_refs.rs.
"""
import math
import os
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_blockscaled_refs import Rng, e2m1, e4m3, f32, ue8m0  # noqa: E402

E4M3_POS = sorted((e4m3(b), b) for b in range(0x7F))  # 0x00..0x7E: 0 .. 448
E2M1_POS = [(e2m1(c), c) for c in range(8)]  # 0 .. 6


def nearest_even(v, grid):
    """Nearest grid value to v >= 0 (saturating at the top), ties to the even pattern."""
    if v >= grid[-1][0]:
        return grid[-1][1]
    best = None
    for val, b in grid:
        d = abs(v - val)
        if best is None or d < best[0] or (d == best[0] and b % 2 == 0):
            best = (d, b)
    return best[1]


STATS = {"chose4": 0, "tie": 0, "zero_scale": 0, "subnormal_scale": 0}


def quant_nvfp4(x):
    """x: list of Fractions. Returns (codes as 2*E2M1 signed ints, E4M3 scale bytes, amax)."""
    amax = max(abs(v) for v in x)
    codes, scales = [0] * len(x), [0] * (len(x) // 16)
    if amax == 0:
        return codes, scales, amax
    g = amax / 1536
    for b in range(len(x) // 16):
        blk = x[b * 16:(b + 1) * 16]
        bm = max(abs(v) for v in blk)
        cands = []
        for k in (6, 4):
            s = nearest_even(bm / (k * g), E4M3_POS)
            sv = e4m3(s)
            q = []
            for v in blk:
                if sv == 0:
                    q.append(0)
                else:
                    c = nearest_even(abs(v) / (sv * g), E2M1_POS)
                    x2 = int(e2m1(c) * 2)
                    q.append(-x2 if v < 0 else x2)
            err = sum(abs(Fr(qi, 2) * sv * g - v) for qi, v in zip(q, blk))
            cands.append((err, s, q))
        (e6, s6, q6), (e4, s4, q4) = cands
        s, q = (s4, q4) if e4 < e6 else (s6, q6)
        STATS["chose4"] += e4 < e6
        STATS["tie"] += e4 == e6 and s4 != s6
        STATS["zero_scale"] += s == 0 and bm > 0
        STATS["subnormal_scale"] += 0 < s < 8
        scales[b] = s
        codes[b * 16:(b + 1) * 16] = q
    return codes, scales, amax


def quant_mxfp8(x):
    """x: list of Fractions. Returns (E4M3 element bytes, UE8M0 fields)."""
    elems, scales = [0] * len(x), [0] * (len(x) // 32)
    for b in range(len(x) // 32):
        blk = x[b * 32:(b + 1) * 32]
        amax = max(abs(v) for v in blk)
        if amax == 0:
            continue
        e = -127
        while e < 127 and Fr(2) ** e < amax / 448:
            e += 1
        scales[b] = e + 127
        for i, v in enumerate(blk):
            c = nearest_even(abs(v) / Fr(2) ** e, E4M3_POS)
            elems[b * 32 + i] = (c | 0x80) if v < 0 and c != 0 else c
    return elems, scales


def activation(rng, n, frac, kind):
    raws = []
    for i in range(n):
        r = rng.next()
        if kind == 0:
            m = (r >> 8) % (1 << 14)
        elif kind == 1:  # blocks of very different size: some round to a zero or subnormal scale
            m = ((r >> 8) % (1 << 30)) >> ((i // 16) % 8 * 4)
        elif kind == 2:  # zero blocks and repeated values (ties in the error sums)
            m = 0 if (i // 16) % 3 == 0 else [5, 7, 9, 512][(r >> 8) % 4] << ((i // 32) % 4)
        else:
            m = (r >> 8) % (1 << 24)
        raws.append(-m if r & 1 else m)
    return raws


def floor_nearest(v):
    return math.floor(v), math.floor(v + Fr(1, 2))


def main():
    rng = Rng(0xAC7)
    rows, cols = 4, 128
    # one NVFP4 and one MXFP8 weight matrix (real-like scales)
    codes_w = [rng.below(256) for _ in range(rows * cols // 2)]
    sc_w = []
    while len(sc_w) < rows * cols // 16:
        b = (rng.below(5) + 11) << 3 | rng.below(8)
        if e4m3(b) is not None:
            sc_w.append(b)
    amax_bits, den_bits = 0x3EA34567, 0x45280000
    fac = f32(amax_bits) / f32(den_bits)
    nv_w = [[e2m1((codes_w[(r * cols + c) // 2] >> (4 * (c % 2))) & 15) * e4m3(sc_w[(r * cols + c) // 16]) * fac for c in range(cols)] for r in range(rows)]
    el_w = []
    while len(el_w) < rows * cols:
        b = rng.below(256)
        if e4m3(b) is not None:
            el_w.append(b)
    ms_w = [113 + rng.below(4) for _ in range(rows * cols // 32)]
    mx_w = [[e4m3(el_w[r * cols + c]) * ue8m0(ms_w[(r * cols + c) // 32]) for c in range(cols)] for r in range(rows)]
    cases = []
    for t in range(48):
        frac = [10, 16][t % 2]
        kind = t % 4
        raws = activation(rng, cols, frac, kind)
        x = [Fr(v, 1 << frac) for v in raws]
        nc, ns, amax = quant_nvfp4(x)
        mc, ms = quant_mxfp8(x)
        xq_nv = [Fr(c, 2) * e4m3(ns[i // 16]) * amax / 1536 for i, c in enumerate(nc)]
        xq_mx = [e4m3(mc[i]) * ue8m0(ms[i // 32]) for i in range(cols)]
        out = []
        for W, xq in ((nv_w, xq_nv), (mx_w, xq_mx)):
            fl, nr = [], []
            for row in W:
                s = sum((w * v for w, v in zip(row, xq)), Fr(0)) * (1 << frac)
                f_, n_ = floor_nearest(s)
                fl.append(f_)
                nr.append(n_)
            out.append((fl, nr))
        cases.append(dict(frac=frac, x=raws, nc=nc, ns=ns, amax=int(amax * (1 << frac)), mc=mc, ms=ms, out=out))
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "tests", "data", "actquant_refs.rs")
    L = lambda v: "&[" + ", ".join(str(i) for i in v) + "]"
    with open(path, "w") as f:
        f.write("// Generated by scripts/generate_actquant_refs.py (Fraction arithmetic, no floats). Do not edit.\n\n")
        f.write(f"pub const ROWS: usize = {rows};\npub const COLS: usize = {cols};\n")
        f.write(f"pub const NV_CODES: &[u8] = {L(codes_w)};\npub const NV_SCALES: &[u8] = {L(sc_w)};\n")
        f.write(f"pub const NV_AMAX: u32 = {amax_bits:#010x};\npub const NV_DEN: u32 = {den_bits:#010x};\n")
        f.write(f"pub const MX_ELEMS: &[u8] = {L(el_w)};\npub const MX_SCALES: &[u8] = {L(ms_w)};\n\n")
        f.write("pub struct Case { pub frac: u32, pub x: &'static [i32], pub nv_codes: &'static [i8], pub nv_scales: &'static [u8], pub nv_amax: i64,\n")
        f.write("    pub mx_elems: &'static [u8], pub mx_scales: &'static [u8], pub nv_floor: &'static [i64], pub nv_near: &'static [i64], pub mx_floor: &'static [i64], pub mx_near: &'static [i64] }\n\n")
        f.write("pub const CASES: &[Case] = &[\n")
        for c in cases:
            (nf, nn), (mf, mn) = c["out"]
            f.write(f"    Case {{ frac: {c['frac']}, x: {L(c['x'])}, nv_codes: {L(c['nc'])}, nv_scales: {L(c['ns'])}, nv_amax: {c['amax']}, "
                    f"mx_elems: {L(c['mc'])}, mx_scales: {L(c['ms'])}, nv_floor: {L(nf)}, nv_near: {L(nn)}, mx_floor: {L(mf)}, mx_near: {L(mn)} }},\n")
        f.write("];\n")
        f.write(f"pub const CHOSE4: usize = {STATS['chose4']};\npub const ZERO_SCALE: usize = {STATS['zero_scale']};\npub const SUBNORMAL_SCALE: usize = {STATS['subnormal_scale']};\npub const TIES_KEPT_6: usize = {STATS['tie']};\n")
    print(f"{len(cases)} activation rows; wrote {os.path.normpath(path)}; {STATS}")


if __name__ == "__main__":
    main()
