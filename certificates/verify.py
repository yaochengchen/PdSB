#!/usr/bin/env python3
"""Independent verifier for the Max-k-Cut solutions in this repository.

Pure Python 3 (no third-party packages). For every row of results.csv it
  1. obtains the Gset graph (local folder, or download from the official
     Stanford page), and checks its SHA-256 against the value below;
  2. reads the solution file (line i = part label of vertex i, labels 0..K-1);
  3. checks that the file has exactly n labels, all in {0, ..., K-1};
  4. recomputes  cut = sum of w_ij over edges (i, j) with label_i != label_j
     and compares it with the claimed value and with the previous best-known.

Usage
  python verify.py                       # verify every row of results.csv
  python verify.py --graph-dir ./gset    # use local Gset files G1, G22, ...
  python verify.py G81 3 solutions/G81_max3cut_solution.txt   # one solution
"""
import argparse
import csv
import hashlib
import os
import sys
import urllib.request

GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/{}"

# SHA-256 of the official Gset files (web.stanford.edu/~yyye/yyye/Gset/)
GSET_SHA256 = {
    "G1": "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
    "G2": "732d57480a0151e81253a08535da539ff579f77dbc8673d2b58f4b051f6ccdac",
    "G3": "999e49b5e093408bbb9e9d8aa76a4d05d61b7f8d4704cb1e35b596a6f1ad2897",
    "G4": "df9872b8f1a5970d2d8f8345d259a6ea72480b6b00cf522a858b9d78a3804071",
    "G5": "0e69826d8ba546aa6bd20fddd7d5b9fd125c851e5d7fda227a5656ee456f27ea",
    "G22": "9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e",
    "G43": "9af5445b4b066cbf1eabe218d4e0d907cb6f211651cae557c761ec344dc37be8",
    "G55": "7537bbb613a6e6d0fe784937be68099eeb36227d85bb273299f22cb7d3ff2227",
    "G70": "0d965a2ff1444cc71d7740824b5b3a5e09fc1843cbf6d90691b52d5221987061",
    "G81": "74e69d2f5228774cedbdb86da14debf08023556f1d7693b7346ca13df7594d5a",
}

HERE = os.path.dirname(os.path.abspath(__file__))


def get_graph_bytes(name, graph_dir):
    """Return the raw bytes of Gset graph `name` (local file or download)."""
    if graph_dir:
        for cand in (name, name + ".txt", os.path.join(name, name + ".txt")):
            p = os.path.join(graph_dir, cand)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return f.read()
    cache = os.path.join(HERE, "gset_cache")
    p = os.path.join(cache, name)
    if not os.path.isfile(p):
        os.makedirs(cache, exist_ok=True)
        print(f"  downloading {name} from {GSET_URL.format(name)}")
        with urllib.request.urlopen(GSET_URL.format(name), timeout=60) as r:
            data = r.read()
        with open(p, "wb") as f:
            f.write(data)
    with open(p, "rb") as f:
        return f.read()


def parse_graph(raw):
    lines = raw.decode("ascii").split("\n")
    n, m = map(int, lines[0].split()[:2])
    edges = []
    for line in lines[1:]:
        t = line.split()
        if len(t) >= 3:
            edges.append((int(t[0]) - 1, int(t[1]) - 1, int(t[2])))
    if len(edges) != m:
        raise ValueError(f"header says m={m} but file has {len(edges)} edges")
    return n, edges


def read_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                labels.append(int(s))
    return labels


def cut_value(edges, labels):
    return sum(w for i, j, w in edges if labels[i] != labels[j])


def verify(name, K, sol_path, graph_dir, claimed=None, previous=None):
    raw = get_graph_bytes(name, graph_dir)
    sha = hashlib.sha256(raw).hexdigest()
    sha_ok = GSET_SHA256.get(name) == sha
    n, edges = parse_graph(raw)
    labels = read_labels(sol_path)
    ok = True
    msgs = []
    if not sha_ok:
        ok = False
        msgs.append(f"graph SHA-256 mismatch ({sha[:12]}...)")
    if len(labels) != n:
        ok = False
        msgs.append(f"expected {n} labels, found {len(labels)}")
    bad = [x for x in labels if not 0 <= x < K]
    if bad:
        ok = False
        msgs.append(f"{len(bad)} labels outside 0..{K - 1}")
    cut = cut_value(edges, labels) if ok else None
    if ok and claimed is not None and cut != claimed:
        ok = False
        msgs.append(f"recomputed cut {cut} != claimed {claimed}")
    parts = len(set(labels))
    line = f"{name:>4}  K={K}  n={n:<6} m={len(edges):<6} cut={cut}"
    if previous is not None and cut is not None:
        d = cut - previous
        line += f"  previous best={previous}  ({'+' if d > 0 else ''}{d})"
    line += f"  parts used={parts}  {'OK' if ok else 'FAIL: ' + '; '.join(msgs)}"
    print(line)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", nargs="?", help="Gset name, e.g. G81")
    ap.add_argument("K", nargs="?", type=int, help="number of parts")
    ap.add_argument("solution", nargs="?", help="solution file")
    ap.add_argument("--graph-dir", default=None,
                    help="folder with local Gset files (otherwise downloaded)")
    ap.add_argument("--csv", default=os.path.join(HERE, "results.csv"))
    a = ap.parse_args()

    if a.graph:
        if a.K is None or a.solution is None:
            ap.error("give graph, K and solution file together")
        sys.exit(0 if verify(a.graph, a.K, a.solution, a.graph_dir) else 1)

    all_ok = True
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            sol = os.path.join(HERE, r["solution_file"])
            with open(sol, "rb") as g:
                if hashlib.sha256(g.read()).hexdigest() != r["solution_sha256"]:
                    print(f"{r['solution_file']}: file SHA-256 differs from results.csv")
                    all_ok = False
            all_ok &= verify(r["instance"], int(r["K"]), sol, a.graph_dir,
                             claimed=int(r["cut"]),
                             previous=int(r["previous_best"]) if r["previous_best"] else None)
    print("\nALL SOLUTIONS VERIFIED" if all_ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
