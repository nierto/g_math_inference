#!/usr/bin/env python3
"""References for Ptq1Matrix (tests/ptq1_validation.rs), exact fractions.

Each case: a small matrix of PTQ1_0 blocks (random bytes, including bytes a
canonical encoder never writes, and binary16 scales of every kind), an
optional Walsh-Hadamard input transform with or without signs, activations
at Q22.10 / Q16.16 raws (some near the word's limits), and the expected
outputs: floor and nearest (ties toward +inf) at the activations' format,
floor at 2F. Everything is computed from the definitions with Fraction:
the trit decode is llama.cpp's dequantize_row_ptq1_0 loop, the scale is the
binary16 value, the transform is the matrix sum (-1)^popcount(r & c) / sqrt(n).
Every output is computed twice: with Fraction (1 / sqrt(n) as the exact power
of two) and with mpmath at 200 digits (binary16 by ldexp, 1 / sqrt(n) by
mpmath's sqrt); the generator asserts the two agree. No floats.
"""
import os
from fractions import Fraction as Fr
from mpmath import mp, mpf, sqrt, ldexp, floor as mfloor

mp.dps = 200


class Rng:
    def __init__(self, s):
        self.s = s

    def next(self):
        s = self.s
        s ^= s >> 12
        s ^= (s << 25) & ((1 << 64) - 1)
        s ^= s >> 27
        self.s = s
        return (s * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)


def trits(qs, qh):
    pow3 = [1, 3, 9, 27, 81, 243]
    y = []
    j = 0
    for c in (32, 16, 8):
        while j + c <= len(qs):
            for n in range(5):
                for m in range(c):
                    q = (qs[j + m] * pow3[n]) & 0xFF
                    y.append(((q * 3) >> 8) - 1)
            j += c
    for n in range(4):
        for h in range(len(qh)):
            q = (qh[h] * pow3[n]) & 0xFF
            y.append(((q * 3) >> 8) - 1)
    return y


def half(h):
    s = -1 if h & 0x8000 else 1
    e = (h >> 10) & 31
    m = h & 1023
    assert e != 31
    if e == 0:
        return s * Fr(m, 1 << 24)
    return s * Fr(1024 + m, 1024) * (Fr(2) ** (e - 15))


def hadamard(u, block):
    out = []
    k = block.bit_length() - 1
    assert k % 2 == 0
    norm = Fr(1, 1 << (k // 2))
    for b0 in range(0, len(u), block):
        seg = u[b0:b0 + block]
        for r in range(block):
            s = 0  # integer inputs: the sum is an integer
            for c in range(block):
                s += -seg[c] if bin(r & c).count("1") % 2 else seg[c]
            out.append(s * norm)
    return out


def floor_frac(x):
    return x.numerator // x.denominator


def main():
    rng = Rng(0x9713C0DE)
    cases = []
    # (rows, cols, hadamard block or 0, signs?, F, scale kind)
    shapes = [
        (3, 128, 0, False, 10, "mixed"),
        (3, 256, 16, True, 10, "mixed"),
        (2, 512, 256, False, 16, "small"),
        (2, 1024, 1024, True, 10, "mixed"),
        (4, 128, 4, True, 16, "small"),
        (3, 384, 64, True, 10, "tiny"),
    ]
    for rows, cols, hb, signed, frac, kind in shapes:
        nb = rows * cols // 128
        blocks = []
        for _ in range(nb):
            b = [rng.next() & 0xFF for _ in range(26)]
            r = rng.next()
            if kind == "tiny":
                h = (r & 0x83FF)  # subnormal scales, either sign
            elif kind == "small":
                h = (r & 0x8000) | (((r >> 16) % 12 + 1) << 10) | ((r >> 24) & 1023)  # 2^-14 .. 2^-3
            else:
                choice = (r >> 40) % 6
                if choice == 0:
                    h = 0x0000 if (r & 1) else 0x8000  # zero scale
                elif choice == 1:
                    h = (r & 0x83FF)  # subnormal
                else:
                    h = (r & 0x8000) | (((r >> 16) % 14 + 1) << 10) | ((r >> 24) & 1023)  # up to 2^-1
            blocks += b + [h & 0xFF, h >> 8]
        signs = [1 if rng.next() & 1 else -1 for _ in range(cols)] if signed else None
        acts = []
        for bits in (6, 14, 22, 31):
            acts.append([(rng.next() % (1 << bits)) - (1 << (bits - 1)) for _ in range(cols)])
        # values near the word's limits
        acts.append([(-(1 << 31)) if (rng.next() & 1) else ((1 << 31) - 1) for _ in range(cols)])
        outs = []
        for x in acts:
            u = [v * (signs[i] if signs else 1) for i, v in enumerate(x)]
            u = hadamard(u, hb) if hb else [Fr(v) for v in u]
            row_out = []
            for r in range(rows):
                total = Fr(0)
                for b in range(cols // 128):
                    off = (r * (cols // 128) + b) * 28
                    blk = blocks[off:off + 28]
                    t = trits(blk[:24], blk[24:26])
                    d = half(blk[26] | (blk[27] << 8))
                    s = sum(t[j] * u[b * 128 + j] for j in range(128))
                    total += d * s
                # the same value by mpmath: binary16 by ldexp, the transform with a real sqrt
                mt = mpf(0)
                xs_m = [mpf(v) * (signs[i] if signs else 1) for i, v in enumerate(x)]
                if hb:
                    um = []
                    for b0 in range(0, cols, hb):
                        seg = xs_m[b0:b0 + hb]
                        for rr in range(hb):
                            acc = mpf(0)
                            for cc in range(hb):
                                acc += -seg[cc] if bin(rr & cc).count("1") % 2 else seg[cc]
                            um.append(acc / sqrt(hb))
                else:
                    um = xs_m
                for b in range(cols // 128):
                    off = (r * (cols // 128) + b) * 28
                    blk = blocks[off:off + 28]
                    t = trits(blk[:24], blk[24:26])
                    h = blk[26] | (blk[27] << 8)
                    e, m = (h >> 10) & 31, h & 1023
                    dm = ldexp(mpf(m), -24) if e == 0 else ldexp(mpf(1024 + m), e - 25)
                    if h & 0x8000:
                        dm = -dm
                    mt += dm * sum(t[j] * um[b * 128 + j] for j in range(128))
                assert int(mfloor(mt)) == floor_frac(total), "mpmath and Fraction disagree (floor)"
                assert int(mfloor(mt + mpf(1) / 2)) == floor_frac(total + Fr(1, 2)), "mpmath and Fraction disagree (nearest)"
                assert int(mfloor(ldexp(mt, frac))) == floor_frac(total * (1 << frac)), "mpmath and Fraction disagree (2F)"
                # total is in raw units of 2^-frac
                fl = floor_frac(total)
                near = floor_frac(total + Fr(1, 2))
                q2f = floor_frac(total * (1 << frac))
                ok = all(-(1 << 31) <= v < (1 << 31) for v in (fl, near)) and -(1 << 63) <= q2f < (1 << 63)
                row_out.append((fl, near, q2f) if ok else None)
            outs.append(row_out)
        cases.append((rows, cols, hb, signs, frac, blocks, acts, outs))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "ptq1_refs.rs")
    with open(out, "w") as f:
        f.write("// Generated by scripts/generate_ptq1_refs.py (exact fractions, each value confirmed by mpmath at 200 digits). Do not edit.\n\n")
        f.write("pub struct Case {\n    pub rows: usize,\n    pub cols: usize,\n    pub hadamard: usize,\n")
        f.write("    pub signs: Option<&'static [i8]>,\n    pub frac: u32,\n    pub blocks: &'static [u8],\n")
        f.write("    pub acts: &'static [&'static [i32]],\n")
        f.write("    /// per activation, per row: (floor, nearest, floor at 2F), None when out of range\n")
        f.write("    pub outs: &'static [&'static [Option<(i64, i64, i64)>]],\n}\n\n")
        f.write("pub const CASES: &[Case] = &[\n")
        for rows, cols, hb, signs, frac, blocks, acts, outs in cases:
            f.write("    Case {\n")
            f.write(f"        rows: {rows},\n        cols: {cols},\n        hadamard: {hb},\n")
            f.write("        signs: " + ("None" if signs is None else "Some(&[" + ", ".join(map(str, signs)) + "])") + ",\n")
            f.write(f"        frac: {frac},\n")
            f.write("        blocks: &[" + ", ".join(map(str, blocks)) + "],\n")
            f.write("        acts: &[\n")
            for x in acts:
                f.write("            &[" + ", ".join(map(str, x)) + "],\n")
            f.write("        ],\n        outs: &[\n")
            for ro in outs:
                f.write("            &[" + ", ".join("None" if v is None else f"Some(({v[0]}, {v[1]}, {v[2]}))" for v in ro) + "],\n")
            f.write("        ],\n    },\n")
        f.write("];\n")
    n = sum(1 for c in cases for ro in c[7] for v in ro if v is not None)
    print(f"{len(cases)} cases, {n} outputs in range; wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
