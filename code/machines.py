#!/usr/bin/env python3
"""PyTorch implementations of every machine in the comparison (one framework, same GPU, same GEMM precision).

  PdSB      -- Potts discrete simulated bifurcation (this work).  Colour c = vertex e_c of the regular simplex
               in R^(K-1) (<e_a,e_b> = -1/(K-1)): the Potts energy is exact for every K.  dSB dynamics with
               the Ising box replaced by an inelastic wall on the simplex hull.
               Variants (ablations) through the same class:
                 coupling="ballistic"           -> PbSB (neighbours enter with their continuous position)
                 wall="sphere"                  -> PdSB-sphere (|x| <= 1 instead of the simplex hull)
                 embed="clock", wall="sphere"   -> clock-dSB (planar Potts, the class of all earlier Potts machines)
  OneHotSB  -- the same dSB integrator on N*K Ising spins with the one-hot (Lucas) encoding: isolates the
               encoding, everything else identical.
  RefIM     -- faithful re-implementation of the reference Ising machine of Frederiksen, De Prins, Bienstman,
               arXiv:2605.06425, Eq. (9): sigmoid CIM with spin-sign coupling, one-hot encoding (Eq. S1.33),
               external fields scaled by zeta, Euler-Maruyama, alpha = -50, beta = eps_beta t, dt = 1e-3,
               f_xi = 1e-4, 1e6 steps (Supp. Tables S6.23-S6.26).

Hyper-parameter grids run in ONE batch: every replica carries its own parameters (c0, dt, A, zeta, ...), so a
job is (method, graph, K, steps) and its R = n_configs x runs replicas share each GEMM.

All coupling products of discrete states are integer-valued (one-hot colours, signs, +-1 weights), so they run
as FP16 tensor-core GEMMs and are still exact (checked: integer weights, row abs-sum <= 2048).  Continuous
products (PbSB) use FP32.  Reported cuts are always recomputed exactly.

"Run" = one replica.  The metric of 2605.06425 is the maximum cut seen along each run (best-seen); it is
evaluated every `eval_every` steps inside the CUDA graph.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from maxkcut import Graph


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def simplex_vertices(K: int) -> np.ndarray:
    """(K, K-1) unit vectors with pairwise inner product -1/(K-1)."""
    if K < 2:
        raise ValueError("K >= 2")
    E = np.eye(K) - 1.0 / K
    _, _, Vt = np.linalg.svd(E)
    V = E @ Vt[: K - 1].T
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def clock_vertices(K: int) -> np.ndarray:
    if K == 2:
        return np.array([[1.0], [-1.0]])
    a = 2 * np.pi * np.arange(K) / K
    return np.stack([np.cos(a), np.sin(a)], 1)


def project_prob_simplex_sort(q: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row (last dim) onto the probability simplex (Duchi et al. 2008)."""
    K = q.shape[-1]
    u, _ = torch.sort(q, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1) - 1.0
    j = torch.arange(1, K + 1, device=q.device, dtype=q.dtype)
    rho = (u - css / j > 0).sum(-1, keepdim=True) - 1
    theta = torch.gather(css, -1, rho) / (rho + 1).to(q.dtype)
    return torch.clamp(q - theta, min=0.0)


def project_prob_simplex(q: torch.Tensor) -> torch.Tensor:
    """Same projection by Michelot's active-set iteration: elementwise only, exact after <= K rounds
    (theta never decreases, so a coordinate that leaves the active set never returns)."""
    K = q.shape[-1]
    if K > 8:
        return project_prob_simplex_sort(q)
    act = torch.ones_like(q)
    theta = None
    for _ in range(K):
        theta = ((q * act).sum(-1, keepdim=True) - 1.0) / act.sum(-1, keepdim=True)
        act = (q > theta).to(q.dtype)
    return torch.clamp(q - theta, min=0.0)


# A point less than WALL_TOL outside the hull (in barycentric coordinates) counts as inside.  Without it, a point
# lying exactly on a facet is "outside" or not depending on rounding, and the momentum-removal direction
# (x - x_proj)/|x - x_proj| is then the direction of a ~1e-9 vector, i.e. noise: fp32 and fp64 runs of the same
# algorithm differ by ~1e-2 after 3 steps.  With 1e-4 they agree to ~1e-5.  Same constant in csrc/pdsb_fused.cu.
WALL_TOL = 1e-4


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


# ----------------------------------------------------------------------------------------------
# graph on the device
# ----------------------------------------------------------------------------------------------
class DeviceGraph:
    """W on the device: dense up to dense_max vertices (FP32, plus an exact FP16 copy for integer products),
    CSR above.  Exact costs and the 1-opt polish."""

    def __init__(self, g: Graph, device="cuda", dense_max: int = 4096):
        self.g, self.n, self.device = g, g.n, torch.device(device)
        A = g.csr()
        self.Asp = A
        self.dense = g.n <= dense_max
        integer = bool(np.allclose(A.data, np.round(A.data)))
        rowabs = float(np.abs(A).sum(1).max()) if g.m else 0.0
        self.exact16 = self.dense and integer and rowabs <= 2048 and self.device.type == "cuda"
        if self.dense:
            self.W = torch.as_tensor(A.toarray(), dtype=torch.float32, device=self.device)
        else:
            self.W = torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64),
                                             torch.as_tensor(A.indices, dtype=torch.int64),
                                             torch.as_tensor(A.data), size=A.shape, dtype=torch.float32,
                                             device=self.device)
        self.W16 = self.W.half() if self.exact16 else None
        self.int_dtype = torch.float16 if self.exact16 else torch.float32
        self.deg = torch.as_tensor(g.degree_w(), dtype=torch.float32, device=self.device)
        self.q2 = np.asarray(A.multiply(A).sum(1)).ravel()
        self.rownorm = float(np.sqrt(self.q2.mean())) if g.m else 1.0
        self.total_w = g.total_w
        self._classes = None

    def mm(self, X: torch.Tensor) -> torch.Tensor:
        """FP32 W @ X, X shaped (n, ...)."""
        sh = X.shape
        return (self.W @ X.reshape(sh[0], -1)).reshape(sh)

    def mm_int(self, X: torch.Tensor) -> torch.Tensor:
        """W @ X for integer-valued X stored in self.int_dtype (FP16 tensor cores when exact)."""
        sh = X.shape
        W = self.W16 if self.exact16 else self.W
        return (W @ X.reshape(sh[0], -1)).reshape(sh)

    def same_weight(self, colors: torch.Tensor, K: int) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(colors, K).to(self.int_dtype)
        WX = self.mm_int(oh)
        return 0.5 * (oh.float() * WX.float()).sum((0, 2)).double()

    def cut(self, colors: torch.Tensor, K: int) -> torch.Tensor:
        return self.total_w - self.same_weight(colors, K)

    def cut_exact(self, colors: torch.Tensor) -> np.ndarray:
        """FP64 on the CPU from the edge list (the number that is reported)."""
        c = colors.cpu().numpy()
        e = self.g.edges
        eq = c[e[:, 0]] == c[e[:, 1]]
        return self.total_w - (self.g.w[:, None] * eq).sum(0)

    def classes(self):
        if self._classes is None:
            A = self.Asp
            order = np.argsort(-np.diff(A.indptr), kind="stable")
            col = np.full(self.n, -1, dtype=np.int64)
            for v in order:
                nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
                used = set(col[nb][col[nb] >= 0].tolist())
                c = 0
                while c in used:
                    c += 1
                col[v] = c
            self._classes = []
            for c in range(col.max() + 1 if self.n else 0):
                idx = np.flatnonzero(col == c)
                rows = A[idx]
                if self.dense:
                    Wr = torch.as_tensor(rows.toarray(), dtype=torch.float32, device=self.device)
                else:
                    Wr = torch.sparse_csr_tensor(torch.as_tensor(rows.indptr, dtype=torch.int64),
                                                 torch.as_tensor(rows.indices, dtype=torch.int64),
                                                 torch.as_tensor(rows.data), size=rows.shape,
                                                 dtype=torch.float32, device=self.device)
                self._classes.append((torch.as_tensor(idx, device=self.device), Wr))
        return self._classes

    @torch.no_grad()
    def polish(self, colors: torch.Tensor, K: int, max_sweeps: int = 1000, chunk: int = 2048):
        """Exact 1-opt, class by class (independent sets), replicas in chunks to bound memory."""
        col = colors.clone()
        n, R = col.shape
        for r0 in range(0, R, chunk):
            c = col[:, r0:r0 + chunk].contiguous()
            Rc = c.shape[1]
            for _ in range(max_sweeps):
                moved = 0
                for idx, Wr in self.classes():
                    oh = torch.nn.functional.one_hot(c, K).float().reshape(n, Rc * K)
                    F = (Wr @ oh).reshape(len(idx), Rc, K)
                    cur = c[idx]
                    fcur = torch.gather(F, -1, cur[..., None])[..., 0]
                    fbest, best = F.min(-1)
                    mv = fbest < fcur - 1e-6
                    if mv.any():
                        c[idx] = torch.where(mv, best, cur)
                        moved += int(mv.sum())
                if moved == 0:
                    break
            col[:, r0:r0 + chunk] = c
        return col


# ----------------------------------------------------------------------------------------------
# integrator: CUDA graph per chunk of `eval_every` steps, best-seen evaluation inside the graph
# ----------------------------------------------------------------------------------------------
class Machine:
    """Subclasses: state tensors, _step(s) with schedule value s (0-dim tensor), decode(), schedule(t)."""

    eval_every = 10

    def __init__(self, dg: DeviceGraph, K: int, R: int, T: int, use_graph: bool = True, eval_every: int | None = None):
        self.dg, self.K, self.R, self.T = dg, K, R, T
        self.dev = dg.device
        self.use_graph = use_graph and self.dev.type == "cuda"
        if eval_every:
            self.eval_every = int(eval_every)
        self.graph_used = False

    def schedule(self) -> torch.Tensor:          # (T,) values passed to _step
        raise NotImplementedError

    def _eval(self):
        col = self.decode()
        cut = self.dg.cut(col, self.K)
        imp = cut > self.best_cut
        self.best_cut.copy_(torch.where(imp, cut, self.best_cut))
        self.best_col.copy_(torch.where(imp[None, :], col, self.best_col))

    def _chunk(self):
        for k in range(self.C):
            self._step(self.s_buf[k])
        self._eval()

    @torch.no_grad()
    def run(self) -> dict:
        T = self.T
        self.C = max(1, min(self.eval_every, T))
        sched = self.schedule()
        self.s_buf = torch.zeros(self.C, device=self.dev, dtype=torch.float32)
        self.best_cut = torch.full((self.R,), -math.inf, dtype=torch.float64, device=self.dev)
        self.best_col = self.decode().clone()
        n_full = T // self.C
        graph = None
        if self.use_graph and n_full > 0:
            snap = [t.clone() for t in self.state()]
            s = torch.cuda.Stream(device=self.dev)
            s.wait_stream(torch.cuda.current_stream(self.dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._chunk()
            torch.cuda.current_stream(self.dev).wait_stream(s)
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._chunk()
            except Exception as e:
                print(f"[warn] CUDA graph capture failed ({type(e).__name__}: {e}); eager", flush=True)
                graph = None
                torch.cuda.synchronize(self.dev)
            for t, v in zip(self.state(), snap):
                t.copy_(v)
            self.best_cut.fill_(-math.inf)
            self.best_col.copy_(self.decode())
            del snap
        self.graph_used = graph is not None
        sync(self.dev)
        t0 = time.perf_counter()
        done = 0
        for _ in range(n_full):
            self.s_buf.copy_(sched[done:done + self.C])
            if graph is not None:
                graph.replay()
            else:
                self._chunk()
            done += self.C
        for t in range(done, T):
            self._step(sched[t])
        self._eval()
        sync(self.dev)
        t_dyn = time.perf_counter() - t0
        final_col = self.decode()
        return {"final_col": final_col, "final_cut": self.dg.cut(final_col, self.K), "best_col": self.best_col,
                "best_cut": self.best_cut, "t_dyn": t_dyn, "steps": T}


def _rand_ball(gen, shape, D, amp, device):
    v = torch.randn(*shape, D, generator=gen, device=device)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    r = amp * torch.rand(*shape, 1, generator=gen, device=device) ** (1.0 / D)
    return v * r


def _col(x, R, device):
    return torch.as_tensor(np.array(x, dtype=np.float32), device=device).reshape(1, R, 1)


# ----------------------------------------------------------------------------------------------
# PdSB and its variants
# ----------------------------------------------------------------------------------------------
class PdSB(Machine):
    def __init__(self, dg: DeviceGraph, K: int, R: int, T: int, xi, dt, embed="simplex", wall="simplex",
                 coupling="discrete", seed: int = 0, init_amp: float = 0.1, a0: float = 1.0, x0=None, p0=None,
                 **kw):
        super().__init__(dg, K, R, T, **kw)
        if wall == "simplex" and embed != "simplex" and K > 3:
            raise ValueError("simplex wall needs the simplex embedding")
        V = simplex_vertices(K) if embed == "simplex" else clock_vertices(K)
        self.embed, self.wall, self.coupling, self.a0 = embed, wall, coupling, a0
        self.V = torch.as_tensor(V, dtype=torch.float32, device=self.dev)
        self.D = self.V.shape[1]
        xi = np.broadcast_to(np.asarray(xi, dtype=np.float64), (R,))
        self.c0 = _col(xi / dg.rownorm, R, self.dev)
        self.dt = _col(np.broadcast_to(np.asarray(dt, dtype=np.float64), (R,)), R, self.dev)
        gen = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.x = _rand_ball(gen, (dg.n, R), self.D, init_amp, self.dev) if x0 is None else x0.clone().to(self.dev)
        self.p = _rand_ball(gen, (dg.n, R), self.D, init_amp, self.dev) if p0 is None else p0.clone().to(self.dev)
        if coupling == "discrete":
            self.oh = torch.nn.functional.one_hot(self.decode(), K).to(dg.int_dtype)

    def state(self):
        return [self.x, self.p] + ([self.oh] if self.coupling == "discrete" else [])

    def schedule(self):
        return self.a0 * torch.arange(max(self.T, 1), device=self.dev, dtype=torch.float32) / max(self.T, 1)

    def decode(self):
        return torch.argmax(self.x @ self.V.T, dim=-1)

    def _step(self, at):
        if self.coupling == "discrete":
            F = -(self.dg.mm_int(self.oh).float() @ self.V)          # -W e_{c_j}, via exact one-hot GEMM
        else:
            F = -self.dg.mm(self.x)
        self.p.add_(self.dt * (-(self.a0 - at) * self.x + self.c0 * F))
        self.x.add_(self.dt * self.a0 * self.p)
        self._wall()
        if self.coupling == "discrete":
            self.oh.copy_(torch.nn.functional.one_hot(self.decode(), self.K).to(self.oh.dtype))

    def _wall(self):
        x, p = self.x, self.p
        if self.wall == "sphere":
            r = x.norm(dim=-1, keepdim=True)
            out = r > 1.0
            u = x / r.clamp_min(1e-30)
            prad = (p * u).sum(-1, keepdim=True)
            p.copy_(torch.where(out, p - prad * u, p))
            x.copy_(torch.where(out, u, x))
            return
        K = self.K
        q = ((K - 1) * (x @ self.V.T) + 1.0) / K
        out = (q < -WALL_TOL).any(-1, keepdim=True)
        xp = project_prob_simplex(q) @ self.V
        d = x - xp
        nn = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        pn = torch.clamp((p * nn).sum(-1, keepdim=True), min=0.0)
        p.copy_(torch.where(out, p - pn * nn, p))
        x.copy_(torch.where(out, xp, x))


# ----------------------------------------------------------------------------------------------
# one-hot Ising encoding (Lucas graph colouring, Eq. S1.33 of 2605.06425)
# ----------------------------------------------------------------------------------------------
def onehot_fields(dg: DeviceGraph, K: int, A, B):
    """H = A sum_u (1 - sum_c x_uc)^2 + B sum_E w sum_c x_uc x_vc, x = (1+s)/2  ->  -1/2 s^T J s - h^T s:
       J[(u,c),(u,c')] = -A/2,  J[(u,c),(v,c)] = -B w_uv / 4,  h[(u,c)] = -A (K-2)/2 - B d_u / 4.
    A, B: scalars or per-replica arrays.  Returns h (n, R, 1) and per-replica rownorm of J (numpy)."""
    A = np.array(np.atleast_1d(A), dtype=np.float64)
    B = np.array(np.atleast_1d(B), dtype=np.float64)
    At = torch.as_tensor(A, dtype=torch.float32, device=dg.device).reshape(1, -1, 1)
    Bt = torch.as_tensor(B, dtype=torch.float32, device=dg.device).reshape(1, -1, 1)
    h = -At * (K - 2) / 2.0 - Bt * dg.deg.reshape(-1, 1, 1) / 4.0
    rn = np.sqrt((K - 1) * A ** 2 / 4.0 + B ** 2 * dg.q2.mean() / 16.0)
    return h, rn


def onehot_Js(dg: DeviceGraph, s: torch.Tensor, A, B):
    """J s for s (n, R, K) in dg.int_dtype (integer-valued); A, B broadcastable (1, R, 1) tensors."""
    return -(B / 4.0) * dg.mm_int(s).float() - (A / 2.0) * (s.float().sum(-1, keepdim=True) - s.float())


class OneHotSB(Machine):
    """dSB / bSB on the one-hot Ising encoding (same integrator as PdSB)."""

    def __init__(self, dg: DeviceGraph, K: int, R: int, T: int, xi, dt, A, zeta, coupling="discrete", seed=0,
                 init_amp=0.1, a0=1.0, B=1.0, **kw):
        super().__init__(dg, K, R, T, **kw)
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (R,))
        A, zeta, xi, B = bc(A), bc(zeta), bc(xi), bc(B)
        self.h, rn = onehot_fields(dg, K, A, B)
        self.A, self.B, self.zeta = _col(A, R, self.dev), _col(B, R, self.dev), _col(zeta, R, self.dev)
        self.c0 = _col(xi / rn, R, self.dev)
        self.dt = _col(bc(dt), R, self.dev)
        self.a0, self.coupling = a0, coupling
        gen = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.x = init_amp * (2 * torch.rand(dg.n, R, K, generator=gen, device=self.dev) - 1)
        self.p = init_amp * (2 * torch.rand(dg.n, R, K, generator=gen, device=self.dev) - 1)
        self.s = torch.where(self.x >= 0, 1.0, -1.0).to(dg.int_dtype)

    def state(self):
        return [self.x, self.p, self.s]

    def schedule(self):
        return self.a0 * torch.arange(max(self.T, 1), device=self.dev, dtype=torch.float32) / max(self.T, 1)

    def decode(self):
        return torch.argmax(self.x, dim=-1)

    def onehot_violation(self) -> float:
        return float(((self.x > 0).sum(-1) != 1).float().mean())

    def _step(self, at):
        if self.coupling == "discrete":
            Js = onehot_Js(self.dg, self.s, self.A, self.B)
        else:
            Js = -(self.B / 4.0) * self.dg.mm(self.x) - (self.A / 2.0) * (self.x.sum(-1, keepdim=True) - self.x)
        self.p.add_(self.dt * (-(self.a0 - at) * self.x + self.c0 * (Js + self.zeta * self.h)))
        self.x.add_(self.dt * self.a0 * self.p)
        out = self.x.abs() > 1.0
        self.p.copy_(torch.where(out, torch.zeros_like(self.p), self.p))
        self.x.copy_(torch.clamp(self.x, -1.0, 1.0))
        self.s.copy_(torch.where(self.x >= 0, 1.0, -1.0).to(self.s.dtype))


# ----------------------------------------------------------------------------------------------
# reference IM of arXiv:2605.06425 (Eq. 9), faithful
# ----------------------------------------------------------------------------------------------
class RefIM(Machine):
    """dx/dt = -x + tanh(alpha x + beta I),  I = zeta h + J sgn(x),  beta = eps_beta t;  Euler-Maruyama with noise
    f_xi dW.  Encoding: Eq. S1.33 with A = 1 and B = (B/A)n / n (the paper sweeps (B/A) n_vertices).
    eps_beta: None -> 2605.06425 default 1e-5 when steps*dt = 2000, otherwise scaled so beta ends at the same
    value 1e-5 * 2000 = 0.02 (used only for the shorter time-quality runs).  x(0) = 0 (origin).
    Colours for the cut: argmax_c x_uc (always a valid colouring; the most favourable decoding for the IM)."""

    def __init__(self, dg: DeviceGraph, K: int, R: int, T: int, BA_n, zeta, alpha=-50.0, dt=1e-3, f_xi=1e-4,
                 eps_beta=None, seed=0, A=1.0, **kw):
        super().__init__(dg, K, R, T, **kw)
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (R,))
        B = bc(BA_n) / dg.n * A
        self.h, _ = onehot_fields(dg, K, bc(A), B)
        self.A, self.B, self.zeta = _col(bc(A), R, self.dev), _col(B, R, self.dev), _col(bc(zeta), R, self.dev)
        self.alpha, self.dt_e, self.f_xi = float(alpha), float(dt), float(f_xi)
        self.eps_beta = float(eps_beta) if eps_beta is not None else 0.02 / (T * dt)
        # noise from the device's default generator: CUDA-graph safe (a private generator would have to be
        # registered with the graph); seeded per job
        if self.dev.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))
        self.x = torch.zeros(dg.n, R, K, device=self.dev)
        self.s = torch.zeros(dg.n, R, K, device=self.dev, dtype=dg.int_dtype)
        self.noise = torch.empty_like(self.x)
        self.sq = math.sqrt(self.dt_e) * self.f_xi

    def state(self):
        return [self.x, self.s]

    def schedule(self):
        return self.eps_beta * self.dt_e * torch.arange(max(self.T, 1), device=self.dev, dtype=torch.float32)

    def decode(self):
        return torch.argmax(self.x, dim=-1)

    def onehot_violation(self) -> float:
        return float(((self.x > 0).sum(-1) != 1).float().mean())

    def _step(self, beta):
        I = onehot_Js(self.dg, self.s, self.A, self.B) + self.zeta * self.h
        self.noise.normal_()
        self.x.add_(self.dt_e * (torch.tanh(self.alpha * self.x + beta * I) - self.x) + self.sq * self.noise)
        self.s.copy_(torch.sign(self.x).to(self.s.dtype))
