#!/usr/bin/env python3
"""Fused-kernel PdSB (csrc/pdsb_fused.cu), the counterpart of cdsb's GEMM + fused kernel + reduce structure.

Per step: one cuBLAS FP16 GEMM  H = W @ OH  (integer counts, exact) -- or, for a sparse graph (DeviceGraph with
n > dense_max), one CSR gather kernel  H[i,r,c] = sum_j w_ij [col_j,r == c]  in FP32 (exact, no one-hot buffer)
--, one fused kernel (energy partials, best
commit, dynamics, simplex wall, new colour / one-hot), one per-replica reduce kernel (energy, best flag, t+1).
Best-seen is tracked at EVERY step on the device (the energy of the current colouring is a by-product of the
GEMM).  A chunk of steps is captured in one CUDA graph and replayed; the step counter lives on the device, so
the same graph serves every chunk.

Same dynamics as machines.PdSB(coupling="discrete", wall="simplex", embed="simplex"); selftest item 10 checks
the trajectories against it.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from machines import DeviceGraph, _rand_ball, simplex_vertices, sync

HERE = Path(__file__).resolve().parent
_EXT = None


def ext():
    """Build (first call, ~1 min) and load the extension.  Needs nvcc (CUDA_HOME) matching torch's CUDA."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
        std = "-std=c++20" if (major, minor) >= (2, 10) else "-std=c++17"
        arch = os.environ.get("PDSB_ARCH")      # e.g. "9.0" for GH200; default: the visible GPUs
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        _EXT = load(name="pdsb_fused", sources=[str(HERE / "csrc" / "pdsb_fused.cu")],
                    extra_cflags=["-O3", std], extra_cuda_cflags=["-O3", std],
                    verbose=bool(os.environ.get("PDSB_VERBOSE_BUILD")))
    return _EXT


class FusedPdSB:
    def __init__(self, dg: DeviceGraph, K: int, R: int, T: int, xi, dt, seed: int = 0, init_amp: float = 0.1,
                 a0: float = 1.0, chunk: int = 200, use_graph: bool = True, x0=None, p0=None):
        if dg.device.type != "cuda":
            raise ValueError("FusedPdSB needs a CUDA device")
        if K not in (3, 4, 5, 6, 8):
            raise ValueError("fused PdSB is instantiated for K in {3,4,5,6,8}")
        self.dg, self.K, self.R, self.T, self.a0 = dg, K, R, T, a0
        self.dev = dg.device
        self.D = K - 1
        self.Vcpu = torch.as_tensor(simplex_vertices(K), dtype=torch.float32).contiguous()
        self.V = self.Vcpu.to(self.dev)
        bc = lambda v: np.array(np.broadcast_to(np.asarray(v, dtype=np.float64), (R,)))
        self.c0 = torch.as_tensor(bc(xi) / dg.rownorm, dtype=torch.float32, device=self.dev)
        self.dt = torch.as_tensor(bc(dt), dtype=torch.float32, device=self.dev)
        self.seed, self.init_amp, self.x0, self.p0 = seed, init_amp, x0, p0
        self.chunk = max(2, chunk - chunk % 2)          # even: the ping-pong buffers return to the start
        self.use_graph = use_graph
        self.sparse = not dg.dense
        n = dg.n
        if self.sparse:                                  # CSR gather kernel, FP32 H, no one-hot buffers
            A = dg.Asp
            if A.nnz >= 2**31 or n >= 2**31:
                raise ValueError("graph too large for int32 CSR indices")
            self.indptr = torch.as_tensor(A.indptr.astype(np.int64), device=self.dev)
            self.indices = torch.as_tensor(A.indices.astype(np.int32), device=self.dev)
            self.wcsr = torch.as_tensor(A.data.astype(np.float32), device=self.dev)
            self.hdt = torch.float32
            self.W = None
        else:
            self.hdt = dg.int_dtype                      # FP16 when the integer GEMM is exact, else FP32
            self.W = dg.W16 if dg.exact16 else dg.W
        nby = (n + ext().rows_per_block() - 1) // ext().rows_per_block()
        self.x = torch.empty(n, R, self.D, device=self.dev)
        self.p = torch.empty_like(self.x)
        self.col = [torch.empty(n, R, dtype=torch.uint8, device=self.dev) for _ in range(2)]
        ohn = 0 if self.sparse else n
        self.oh = [torch.empty(ohn, R, K, dtype=self.hdt, device=self.dev) for _ in range(2)]
        self.H = torch.empty(n, R, K, dtype=self.hdt, device=self.dev)
        self.best_col = torch.zeros(n, R, dtype=torch.uint8, device=self.dev)
        self.flag = torch.zeros(R, dtype=torch.uint8, device=self.dev)
        self.part = torch.zeros(nby, R, dtype=torch.float64, device=self.dev)
        self.best_e = torch.empty(R, dtype=torch.float64, device=self.dev)
        self.tstep = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.graph_used = False

    def _init_state(self):
        gen = torch.Generator(device=self.dev).manual_seed(int(self.seed))
        n, R = self.dg.n, self.R
        self.x.copy_(_rand_ball(gen, (n, R), self.D, self.init_amp, self.dev) if self.x0 is None else self.x0)
        self.p.copy_(_rand_ball(gen, (n, R), self.D, self.init_amp, self.dev) if self.p0 is None else self.p0)
        c = torch.argmax(self.x @ self.V.T, dim=-1)
        self.col[0].copy_(c.to(torch.uint8))
        self.col[1].copy_(c.to(torch.uint8))
        if not self.sparse:
            self.oh[0].copy_(torch.nn.functional.one_hot(c, self.K).to(self.hdt))
        self.best_col.copy_(c.to(torch.uint8))
        self.flag.zero_()
        self.best_e.fill_(math.inf)
        self.tstep.zero_()

    def _step(self, cur: int):
        E = ext()
        n, R, K = self.dg.n, self.R, self.K
        if self.sparse:
            E.sparse_h(self.indptr, self.indices, self.wcsr, self.col[cur], self.H)
        else:
            torch.mm(self.W, self.oh[cur].view(n, R * K), out=self.H.view(n, R * K))
        E.step(self.H, self.x, self.p, self.col[cur], self.col[1 - cur], self.oh[1 - cur], self.best_col, self.flag,
               self.part, self.c0, self.dt, self.tstep, self.T, self.a0, self.Vcpu)
        E.reduce(self.part, self.best_e, self.flag, self.tstep)

    def _chunk(self):
        for k in range(self.chunk):
            self._step(k % 2)

    @torch.no_grad()
    def run(self) -> dict:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        T = self.T
        n_full = T // self.chunk
        graph = None
        self._init_state()
        if self.use_graph and n_full > 0:
            s = torch.cuda.Stream(device=self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                self._chunk()                       # warm-up: cuBLAS handles, kernels
            torch.cuda.current_stream(self.dev).wait_stream(s)
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._chunk()
            except Exception as e:
                print(f"[warn] fused PdSB: CUDA graph capture failed ({type(e).__name__}: {e}); eager", flush=True)
                graph = None
                torch.cuda.synchronize(self.dev)
            self._init_state()
        self.graph_used = graph is not None
        sync(self.dev)
        t0 = time.perf_counter()
        for _ in range(n_full):
            if graph is not None:
                graph.replay()
            else:
                self._chunk()
        for t in range(n_full * self.chunk, T):
            self._step(t % 2)
        sync(self.dev)
        t_dyn = time.perf_counter() - t0
        last = (T - 1) % 2 if T > 0 else 0          # buffer holding the colouring evaluated at step T-1
        fl = self.flag.bool()
        if T > 0:
            self.best_col[:, fl] = self.col[last][:, fl]
        final = self.col[T % 2].long()               # state after the last update (not yet evaluated)
        final_cut = self.dg.cut(final, self.K)
        best_cut = self.dg.total_w - self.best_e / 2.0
        imp = final_cut > best_cut
        best_cut = torch.where(imp, final_cut, best_cut)
        best_col = torch.where(imp[None, :], final, self.best_col.long())
        return {"final_col": final, "final_cut": final_cut, "best_col": best_col, "best_cut": best_cut,
                "t_dyn": t_dyn, "steps": T, "tracked_best_cut": (self.dg.total_w - self.best_e / 2.0)}
