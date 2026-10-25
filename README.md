# PdSB — Potts Discrete Simulated Bifurcation for Max-k-Cut

Code, raw experimental records, analysis scripts and solution certificates for the paper

> **Beyond One-Hot Encoding: Native Potts Discrete Simulated Bifurcation Breaks G-set Max-k-Cut Records**
> Jiapei Zhuang and Yaocheng Chen (manuscript under review, 2026)

PdSB replaces the binary spins of discrete simulated bifurcation (dSB) by vectors in a regular simplex.
Colour *c* is a vertex **e**<sub>c</sub> of the regular simplex in R<sup>K−1</sup>; any two different colours have inner
product −1/(K−1), so the Potts energy is represented exactly for every K, with no one-hot penalty and no phase
discretisation. The box wall of dSB becomes an inelastic wall on the simplex hull, and the pump drives each state to a
vertex. K = 2 reduces to ordinary dSB.

On the G1–G5 protocol of Frederiksen, De Prins and Bienstman (arXiv:2605.06425v2), PdSB attains a smaller mean gap than
their published one-hot Ising reference on every instance for K = 3 and 4. A dedicated PdSB search exceeds seven
external published comparison values on the G-set (e.g. G81 Max-3-Cut 16498) and matches 17 others.

## Repository layout

```
code/                       algorithms, experiment driver, analysis, verification
  machines.py               PdSB (and PbSB / sphere-wall / clock variants), one-hot dSB, reference one-hot IM (PyTorch)
  fused.py, csrc/pdsb_fused.cu   fused CUDA PdSB (dense FP16 tensor-core GEMM path and sparse CSR path)
  sa_potts.py               heat-bath Potts simulated annealing (Numba, CPU)
  maxkcut.py                G-set reader/downloader, BKS for G1–G5, published values of arXiv:2605.06425
  plans.py, run.py          experiment plans E1–E4L and H1–H3; multi-GPU scheduler with per-setting resume
  analyze.py                tables, significance tests, time-to-solution (TTS99), report/figures
  hunt_report.py            report for the dedicated PdSB search (H1–H3)
  selftest.py, bench.py, check_env.sh, check_results.py
  verify_best.py, verify_cert.py   independent cut re-computation of stored colourings
  build_indexes.py          rebuilds archive_index.csv and parameter_grids.csv from the raw archives
  data/gset/                the 10 G-set graphs used (G1–G5, G22, G43, G55, G70, G81)
  results/                  main experiments E1, E1v, E2, E2L, E3, E3v, E4, E4L (+ smoke) — raw runs.jsonl,
                            certificates, best-colouring archive, analyze.py report/summary/figures
  results_initial/E4/       first large-graph snapshot (initial parameters, used in the parameter-reuse figure)
  results_pdsb/             dedicated PdSB search H1, H2, H3; best/ holds PdSB-only colourings
  README.md                 original (Chinese) experiment notes: methods, plans, run instructions
analysis/
  make_figs.py              regenerates all paper figures and numbers from the raw runs.jsonl files
  figs/                     the six figures of the paper (PDF)
  numbers.json              every derived number quoted in the paper
  parameter_robustness.csv, sa_tuning_history.csv   data behind the parameter-reuse figure
  check_geometry.py         numerical check of the mathematical statements and of the record-table values
  geometry_validation.txt   its output
certificates/               Zenodo release 2.0.0 (doi:10.5281/zenodo.23405852): 35 solutions, results.csv,
                            two standalone verifiers, graph hashes, external comparison sources, licence
archive_index.csv           role, record count, configurations, graphs and size of every runs.jsonl
parameter_grids.csv         every configuration in every archive: path, line, method, graph, K, T, seed, params
SHA256SUMS.txt              integrity list of all files
```

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`; additionally `torch` (≥ 2.5), `numba` and `ninja`
to run the solvers (`code/requirements.txt`). The fused kernel is compiled on first use with
`torch.utils.cpp_extension`; this needs `nvcc` whose CUDA major version matches `torch.version.cuda`.
The original experiments ran on 2× NVIDIA GH200 (CUDA 13.0); SA used 136 Grace CPU threads.
Exact historical package versions were not locked.

Verification and figure/number regeneration need no GPU.

## Quick start

From the repository root:

```bash
# independent verification of the 35 released certificates (standard library only)
python certificates/verify_independent.py --graph-dir code/data/gset

# re-compute the cut of every stored best colouring and every certificate
cd code
python verify_best.py --best results/best --gset data/gset
python verify_best.py --best results_pdsb/best --gset data/gset
python verify_cert.py results/*/certificates/*.npy --gset data/gset
python check_results.py                      # archive integrity; current plans.py vs recorded settings
python check_results.py --results results_pdsb
python selftest.py                           # items 1–8 on CPU, items 9–11 need a GPU
cd ..

# regenerate figures and numbers of the paper (overwrites analysis/figs, numbers.json, the two CSVs)
python analysis/make_figs.py
python analysis/check_geometry.py
python code/build_indexes.py --check         # both index tables must be identical

# regenerate the analysis report and the search report
cd code
python analyze.py results                    # -> results/report.md, summary.csv, figures/
python hunt_report.py results_pdsb --main results
```

`make_figs.py` also reruns the small CPU toy trajectory shown in the geometry figure (fixed seed).
Figures use Times New Roman when available; other fonts change only the glyphs, not the data.

## Re-running the experiments (GPU)

```bash
cd code
bash check_env.sh                                  # read-only environment check (ENV=<conda env> to choose)
python selftest.py
python run.py --plan smoke --results results_smoke # ~2 min pipeline check
python run.py --plan all --results <new_folder>    # E1 E3 E4 -> E1v E3v -> E2L E4L
python run.py --plan hunt --results <new_folder>   # dedicated PdSB search H1 -> H2 + H3
```

Write new runs to a new folder. `run.py` resumes per hyper-parameter setting, so adding grid points to the existing
folders changes the selected settings and therefore the validation/timing subsets used for the paper numbers.

| Plan | Content |
|---|---|
| E1 | G1–G5, K = 3, 4: every method over its grid (100 runs per setting) |
| E1v | best E1 setting per (method, graph, K, steps), 1000 fresh-seed runs: unbiased numbers, tests, TTS99 |
| E2 | earlier small-batch timing; kept because it enters the reference values in `make_figs.py` |
| E2L | large-batch timing (1000 runs per budget) with the E1-selected settings |
| E3 / E3v | K = 5, 6, 8 on G1–G5: tuning / 1000-run validation |
| E4 / E4L | G22, G43, G55, G70, G81, K = 3, 4: per-graph tuning / timing |
| H1–H3 | dedicated PdSB search at 10<sup>6</sup>–10<sup>7</sup> steps on 22 target instances |

## Paper figures and their sources

| Paper item | File | Raw source |
|---|---|---|
| Fig. 1 (geometry) | `analysis/figs/fig_geometry.pdf` | `make_figs.py` (`fig_geometry`, `toy_run`) |
| Fig. 2, Tables 2–3 (quality vs. one-hot reference, ablations) | `fig_onehot_comparison.pdf`, `numbers.json:onehot_comparison` | E1, E1v; published values in `code/maxkcut.py` |
| Fig. 3 (parameter reuse, SA tuning) | `fig_parameter_robustness.pdf`, the two CSVs | E1, E4, `results_initial/E4` |
| Fig. 4, Table 4 (time to target) | `fig_time_quality.pdf`, `numbers.json:e2l` | E2L (settings from E1) |
| Fig. 5 (number of colours) | `fig_vs_K.pdf`, `numbers.json:vsK` | E3, E3v |
| Fig. 6 (larger graphs) | `fig_large.pdf`, `numbers.json:e4l` | E4, E4L, H1–H3 |
| Table 5, Sec. VII (records) | `certificates/results.csv` | H1–H3, `results_pdsb/best`; sources in `certificates/verification/baseline_sources.csv` |

## Notes on interpretation

- `results/best` may contain SA colourings; the solver of each colouring is the `method` field of its `.json`.
  `results_pdsb/best` contains PdSB colourings only. The released certificates may come from different solvers.
- "BKS" / "NEW BEST" labels printed by the historical scripts are not the record criterion of the paper; the paper
  compares with the external values listed in `certificates/verification/baseline_sources.csv` (audit date 2026-10-15).
- Gaps in the figures are relative to fixed reference values, not to the latest record. TTS is an empirical,
  batch-amortised estimate and excludes tuning cost.

## Data sources and licences

- Code: MIT licence (`LICENSE`).
- `certificates/`: CC-BY-4.0 (`certificates/LICENSE`), Zenodo doi:10.5281/zenodo.23405852.
- G-set graphs: downloaded unchanged from <https://web.stanford.edu/~yyye/yyye/Gset/>; SHA-256 hashes in
  `certificates/verification/graph_sha256.json`.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`) and the certificate release:

```bibtex
@misc{chen2026records,
  author       = {Yaocheng Chen},
  title        = {Max-k-Cut solution certificates on the {G-set} benchmark},
  howpublished = {Zenodo, dataset, version 2.0.0},
  year         = {2026},
  doi          = {10.5281/zenodo.23405852}
}
```

## Contact

Yaocheng Chen (yaochengchen@usp.br) and Jiapei Zhuang (jpzhuangphy@gmail.com).
