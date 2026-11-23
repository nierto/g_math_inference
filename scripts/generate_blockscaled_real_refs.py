#!/usr/bin/env python3
"""Exact outputs for real checkpoint slices (not committed: pass their directory).

usage: generate_blockscaled_real_refs.py <dir>
<dir> holds byte slices of one NVFP4 expert (64 rows of codes, its scales, the
per-expert amax table, the denominator) and one MXFP8 expert (64 rows of
elements, its scales). Writes <dir>/real_refs.txt for the test
`real_checkpoint_slices` (env GMI_REAL_SLICES=<dir>). Fraction arithmetic from
the format definitions, no floats.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_blockscaled_refs import TWO, Fr, Rng, activation, e2m1, e4m3, f32, ue8m0  # noqa: E402
import math  # noqa: E402

ROWS = 64


def floor_rows(vals, x, frac):
    out = []
    for row in vals:
        s = sum((w * xi for w, xi in zip(row, x)), Fr(0))
        out.append((math.floor(s), math.floor(s * TWO ** frac)))
    return out


def main():
    d = sys.argv[1]
    rd = lambda n: open(os.path.join(d, n), "rb").read()
    lines = []
    rng = Rng(0x5EA1)
    # NVFP4: layer 0 w13, expert 0 (2304 columns: 1152 code bytes per row, 144 scales)
    cols = 2304
    codes = rd("0.mlp.experts.w13_weight.e0.bin")[: ROWS * cols // 2]
    scales = rd("0.mlp.experts.w13_weight_scale.e0.bin")[: ROWS * cols // 16]
    amax = struct.unpack("<I", rd("0.mlp.experts.w13_weight_amax.bin")[0:4])[0]
    den = struct.unpack("<I", rd("0.mlp.experts.w13_weight_outer_scale_denominator.bin"))[0]
    fac = f32(amax) / f32(den)
    vals = []
    for r in range(ROWS):
        row = []
        for c in range(cols):
            byte = codes[(r * cols + c) // 2]
            code = byte & 15 if c % 2 == 0 else byte >> 4
            row.append(e2m1(code) * e4m3(scales[(r * cols + c) // 16]) * fac)
        vals.append(row)
    for kind in (0, 1, 2, 3):
        x = activation(rng, cols, kind)
        o = floor_rows(vals, x, 10)
        lines.append(f"NV {ROWS} {cols} {amax} {den}")
        lines.append("x " + " ".join(map(str, x)))
        lines.append("out " + " ".join(str(a) for a, _ in o))
        lines.append("q2f " + " ".join(str(b) for _, b in o))
    # MXFP8: layer 20 w13, expert 0 (2304 columns, 72 scales per row)
    elems = rd("20.mlp.experts.w13_weight.e0.bin")[: ROWS * cols]
    mscales = rd("20.mlp.experts.w13_weight_scale.e0.bin")[: ROWS * cols // 32]
    mvals = [[e4m3(elems[r * cols + c]) * ue8m0(mscales[(r * cols + c) // 32]) for c in range(cols)] for r in range(ROWS)]
    for kind in (0, 1, 2, 3):
        x = activation(rng, cols, kind)
        o = floor_rows(mvals, x, 10)
        lines.append(f"MX {ROWS} {cols}")
        lines.append("x " + " ".join(map(str, x)))
        lines.append("out " + " ".join(str(a) for a, _ in o))
        lines.append("q2f " + " ".join(str(b) for _, b in o))
    with open(os.path.join(d, "real_refs.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(d, 'real_refs.txt')}")


if __name__ == "__main__":
    main()
