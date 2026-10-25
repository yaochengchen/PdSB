#!/usr/bin/env python3
"""Independently recompute cuts, local-move gains and all-edges-cut proofs.

Python 3 standard library only. No solution-generation algorithm is included.
Usage: python verify_independent.py [--graph-dir PATH] [--report FILE.csv]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://web.stanford.edu/~yyye/yyye/Gset/{}"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def obtain_graph(name, folder):
    if folder is not None:
        for candidate in (folder / name, folder / (name + ".txt"), folder / name / (name + ".txt")):
            if candidate.is_file():
                return candidate.read_bytes()
        raise FileNotFoundError(f"No graph {name} in {folder}")
    cache = ROOT / "gset_cache" / name
    if not cache.exists():
        with urllib.request.urlopen(URL.format(name), timeout=60) as response:
            data = response.read()
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(data)
    return cache.read_bytes()


def check(row, graph, graph_hash):
    raw, n, edges = graph
    k = int(row["K"])
    solution = (ROOT / row["solution_file"]).read_bytes()
    x = [int(token) for token in solution.split()]
    if digest(raw) != graph_hash:
        raise ValueError("graph SHA-256 mismatch")
    if digest(solution) != row["solution_sha256"]:
        raise ValueError("solution SHA-256 mismatch")
    if n != int(row["n"]) or len(edges) != int(row["m"]):
        raise ValueError("graph size differs from results.csv")
    if len(x) != n or any(label < 0 or label >= k for label in x):
        raise ValueError("invalid label count or range")
    if len(set(x)) != k:
        raise ValueError("not all k parts are used")
    expected_weights = {1} if row["edge_weights"] == "+1" else {-1, 1}
    if {w for _, _, w in edges} != expected_weights:
        raise ValueError("edge weight type differs from results.csv")
    same_part_weight = 0
    total_weight = 0
    neighbor_weights = [[0] * k for _ in range(n)]
    for u, v, w in edges:
        total_weight += w
        same_part_weight += w * (x[u] == x[v])
        neighbor_weights[u][x[v]] += w
        neighbor_weights[v][x[u]] += w
    cut = total_weight - same_part_weight
    if cut != int(row["cut"]):
        raise ValueError(f"cut {cut} differs from claim {row['cut']}")
    gains = [neighbor_weights[v][x[v]] - neighbor_weights[v][target]
             for v in range(n) for target in range(k) if target != x[v]]
    largest_gain = max(gains)
    if largest_gain > 0:
        raise ValueError(f"single-vertex improvement available: +{largest_gain}")
    optimal = all(w > 0 for _, _, w in edges) and cut == total_weight
    status = row["status"]
    if row["previous_best"]:
        difference = cut - int(row["previous_best"])
        if difference != int(row["improvement"]):
            raise ValueError("comparison difference is inconsistent")
        if status == "new best-known" and difference <= 0:
            raise ValueError("nonpositive gain labelled as an improvement")
        if status == "matched best-known" and difference != 0:
            raise ValueError("unequal value labelled as a match")
        if status == "optimal (all edges cut)" and (not optimal or difference != 0):
            raise ValueError("all-edges-cut optimality claim failed")
        if status == "below published best" and difference >= 0:
            raise ValueError("nonnegative difference labelled as below published best")
    elif status != "reference value (no published baseline identified)" or row["improvement"]:
        raise ValueError("missing comparison value for this status")
    return dict(instance=row["instance"], K=k, cut=cut, parts_used=len(set(x)),
                largest_single_vertex_gain=largest_gain,
                improving_single_vertex_moves=sum(g > 0 for g in gains),
                all_edges_cut_optimality_proved=optimal, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--csv", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    expected = json.loads((ROOT / "verification/graph_sha256.json").read_text(encoding="utf-8"))
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    graphs, results, seen, failures = {}, [], set(), []
    for row in rows:
        name, key = row["instance"], (row["instance"], row["K"])
        try:
            if key in seen:
                raise ValueError("duplicate instance/k pair")
            seen.add(key)
            if name not in graphs:
                raw = obtain_graph(name, args.graph_dir)
                if digest(raw) != expected[name]:
                    raise ValueError("graph SHA-256 mismatch")
                values = [int(token) for token in raw.split()]
                n, m = values[:2]
                if len(values) != 2 + 3 * m:
                    raise ValueError("edge count differs from graph header")
                edges = [(values[i] - 1, values[i + 1] - 1, values[i + 2])
                         for i in range(2, len(values), 3)]
                if any(not (0 <= u < n and 0 <= v < n and u != v) for u, v, _ in edges):
                    raise ValueError("invalid edge endpoints")
                if len({tuple(sorted((u, v))) for u, v, _ in edges}) != m:
                    raise ValueError("duplicate graph edges")
                graphs[name] = raw, n, edges
            result = check(row, graphs[name], expected[name])
            results.append(result)
            print(f"{name:>3} k={row['K']} cut={result['cut']} max_move_gain={result['largest_single_vertex_gain']} OK")
        except (ValueError, KeyError, OSError) as error:
            message = f"{name} k={row['K']}: {error}"
            failures.append(message)
            print("FAIL " + message, file=sys.stderr)
    if args.report and not failures and results:
        with args.report.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print(f"\n{len(results)}/{len(rows)} certificates passed; {len(failures)} failures.")
    print("Published comparison values are supplied metadata; this program does not search the literature.")
    return 1 if failures or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
