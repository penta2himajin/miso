#!/usr/bin/env python3
"""Summarize norm durations conditioned on their immediately preceding dispatch.

Defaults to the final PR #33 trace and its last 256 complete decode tokens.
Uses only the Python standard library. Run from any directory; optionally pass
--source TRACE.csv.gz --tokens N --output OUTPUT.json.
"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import pathlib
import re
import statistics


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_SOURCE = ROOT / "bench/model/results/2026-09-27-first-principles/profile-short.csv.gz"


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def predecessor_family(name):
    if "embed_q4k_kernel" in name:
        return "Embedding"
    if "moe_down_kernel" in name:
        match = re.search(r"moe_down_kernel<\(miso::kernels::QType\)([01])", name)
        assert match, "unrecognized MoE-down type: " + name
        return "MoE Q4 down" if match[1] == "0" else "MoE Q6 down"
    if "q4k_gemv_kernel" in name:
        return "Q4 GEMV"
    if "q6k_gemv_kernel" in name:
        return "Q6 GEMV"
    if "mixed_gemv_kernel" in name:
        return "Mixed Q4/Q6 GEMV"
    return "Other: " + name


def summarize(source, tokens, dispatches_per_token):
    # Bound memory by retaining only the final token windows. Prefill's embedding
    # also starts a window, but every retained window must validate as decode.
    windows = collections.deque(maxlen=tokens)
    with gzip.open(source, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if "embed_q4k_kernel" in row["KernelName"]:
                windows.append([])
            if windows:
                windows[-1].append(row)
    assert len(windows) == tokens, "not enough embedding boundaries"
    groups = collections.defaultdict(list)
    predecessors = collections.defaultdict(set)
    for token in windows:
        assert len(token) == dispatches_per_token, "incomplete or unexpected token"
        assert "argmax_final_kernel" in token[-1]["KernelName"]
        assert sum("argmax_final_kernel" in row["KernelName"] for row in token) == 1
        assert sum("add_rmsnorm_kernel" in row["KernelName"] for row in token) == 81
        assert not any("prefill" in row["KernelName"] or "qgemm" in row["KernelName"]
                       for row in token)
        for i, row in enumerate(token):
            assert int(row["DurationNs"]) == int(row["EndNs"]) - int(row["BeginNs"])
            if "add_rmsnorm_kernel" not in row["KernelName"]:
                continue
            previous = token[i - 1]
            family = predecessor_family(previous["KernelName"])
            groups[family].append(int(row["DurationNs"]) / 1000)
            predecessors[family].add(previous["KernelName"])

    statistics_by_family = {}
    for family, values in sorted(groups.items()):
        statistics_by_family[family] = {
            "count": len(values),
            "calls_per_token": len(values) / tokens,
            "mean_us": statistics.mean(values),
            "median_us": statistics.median(values),
            "p10_us": percentile(values, 0.10),
            "p90_us": percentile(values, 0.90),
            "sum_us_per_token": math.fsum(values) / tokens,
            "predecessor_kernels": sorted(predecessors[family]),
        }
    norm_count = sum(len(values) for values in groups.values())
    assert norm_count == 81 * tokens
    path = source.resolve()
    return {
        "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "source_gzip_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_engine_work_commit": (
            "8867a79 (PR #33 final combined implementation)"
            if source.resolve() == DEFAULT_SOURCE.resolve() else "Unspecified for alternate source"
        ),
        "selection": "Final complete decode tokens, delimited by embedding dispatches",
        "tokens": tokens,
        "dispatches": tokens * dispatches_per_token,
        "dispatches_per_token": dispatches_per_token,
        "first_index": int(windows[0][0]["Index"]),
        "last_index": int(windows[-1][-1]["Index"]),
        "norm_count": norm_count,
        "norm_us_per_token": math.fsum(value for values in groups.values() for value in values) / tokens,
        "percentile_method": "Linear interpolation at (n - 1) * fraction",
        "groups": statistics_by_family,
        "limitations": [
            "Durations are rocprof BeginNs..EndNs intervals under timestamp tracing.",
            "They are not native HIP-event measurements or pure arithmetic costs.",
            "Predecessor correlation does not identify a causal memory or dispatch mechanism.",
            "These sums are not functional speedup bounds; replacing a kernel changes the chain.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--dispatches-per-token", type=int, default=345)
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "norm-predecessors.json")
    args = parser.parse_args()
    assert args.tokens > 0
    result = summarize(args.source, args.tokens, args.dispatches_per_token)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tokens": result["tokens"], "norm_us_per_token": result["norm_us_per_token"],
                      "groups": {family: {key: value for key, value in stats.items()
                                          if key != "predecessor_kernels"}
                                 for family, stats in result["groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
