#!/usr/bin/env python3
"""Throughput on the real shapes: ms per step and amortised us per run-step for each GPU machine vs R.

    python bench.py --graph G1 --K 4 --R 100 900 7700 --T 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maxkcut import load_graph  # noqa: E402
from machines import DeviceGraph, OneHotSB, PdSB, RefIM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="G1")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--R", type=int, nargs="*", default=[100, 900, 7700])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dense-max", type=int, default=4096, help="n above this -> sparse CSR path (as run.py)")
    ap.add_argument("--methods", nargs="*",
                    default=["pdsb_fused", "pdsb", "pbsb", "clock_dsb", "onehot_dsb", "ref_im"])
    a = ap.parse_args()
    g = load_graph(a.graph)
    dg = DeviceGraph(g, a.device, dense_max=a.dense_max)
    print(f"{a.graph}: n={g.n} m={g.m} K={a.K} T={a.T}  {torch.cuda.get_device_name(a.device)}  dense={dg.dense} exact16={dg.exact16}")
    mk = {
        "pdsb_fused": lambda R: __import__("fused").FusedPdSB(dg, a.K, R, a.T, 0.75, 0.5),
        "pdsb": lambda R: PdSB(dg, a.K, R, a.T, 0.75, 0.5),
        "pbsb": lambda R: PdSB(dg, a.K, R, a.T, 0.35, 0.15, coupling="ballistic"),
        "clock_dsb": lambda R: PdSB(dg, a.K, R, a.T, 0.35, 1.0, embed="clock", wall="sphere"),
        "onehot_dsb": lambda R: OneHotSB(dg, a.K, R, a.T, 0.5, 0.5, 2.0, 0.5),
        "ref_im": lambda R: RefIM(dg, a.K, R, a.T, 200.0, 0.6),
    }
    print(f"{'method':<11} {'R':>6} {'ms/step':>9} {'us/run-step':>12} {'graph':>6}  best-cut max")
    for m in a.methods:
        for R in a.R:
            s = mk[m](R)
            o = s.run()
            ms = 1e3 * o["t_dyn"] / a.T
            print(f"{m:<11} {R:>6} {ms:>9.3f} {1e3 * ms / R:>12.4f} {str(s.graph_used):>6}  {float(o['best_cut'].max()):.0f}",
                  flush=True)
            del s, o
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
