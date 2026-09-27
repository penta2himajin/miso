#!/usr/bin/env python3
"""Summarize native/graph Norm controls and model-chain event measurements.

The primary statistic is the median of the three outer-process medians.
Differences pair controls in the same outer process and benchmark round.
Run from any directory; override inputs with --controls and --model-chain.
"""

import argparse
import collections
import csv
import hashlib
import gzip
import json
import math
import pathlib
import re
import statistics


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIMENSIONS = ("execution", "path", "sequence", "delta", "chain")
DIFFERENCES = (("norm", "readwrite"), ("norm", "empty"),
               ("readwrite", "empty"), ("empty", "direct"), ("norm", "direct"))


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def stats(values):
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "p10": quantile(values, 0.1),
            "p90": quantile(values, 0.9), "min": min(values), "max": max(values)}


def aggregate(by_process):
    process_stats = [{"run": run, **stats(values)} for run, values in sorted(by_process.items())]
    medians = [row["median"] for row in process_stats]
    pooled = [value for values in by_process.values() for value in values]
    return {"primary_median_of_process_medians": statistics.median(medians),
            "process_median_statistics": stats(medians),
            "per_process": process_stats, "pooled": stats(pooled)}


def relative(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_metadata(path):
    data = path.read_bytes()
    return {"path": relative(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def parse_measurements(path, expected_processes, expected_rounds):
    processes = {}
    rows = []
    header = None
    run = None
    for line in path.read_text().splitlines():
        marker = re.fullmatch(r"run (\d+) utc (.+)", line)
        if marker:
            run = int(marker[1])
            assert run not in processes, "duplicate outer-process marker"
            processes[run] = {"run": run, "start_utc": marker[2], "metadata": []}
            header = None
        elif line.startswith("exit:"):
            assert run is not None and "exit_code" not in processes[run]
            processes[run]["exit_code"] = int(line.split(":", 1)[1])
        elif line.startswith(("execution,path,sequence,", "chain,execution,control,")):
            header = next(csv.reader([line]))
        elif header and line.startswith(("native,", "graph,", "gemv_norm_router,",
                                         "moe4_norm_qkv,", "moe6_norm_qkv,")):
            cells = next(csv.reader([line]))
            assert len(cells) == len(header), "malformed CSV row: " + line
            raw = dict(zip(header, cells))
            assert run is not None
            row = {"run": run, "round": int(raw["round"]), "execution": raw["execution"],
                   "iterations": int(raw["iterations"])}
            if "control" in raw:
                row.update(path="actual_model_chain", sequence="producer_middle_consumer",
                           delta=1, chain=raw["chain"], variant=raw["control"],
                           device_us=float(raw["device_us_per_step"]),
                           cpu_enqueue_us=float(raw["cpu_enqueue_us_per_step"]),
                           kernels_per_step=2 if raw["control"] == "direct" else 3)
            else:
                row.update(path=raw["path"], sequence=raw["sequence"], delta=int(raw["delta"]),
                           chain=None, variant=raw["variant"], device_us=float(raw["us_per_step"]),
                           cpu_enqueue_us=float(raw["enqueue_us_per_step"]),
                           kernels_per_step=int(raw["kernels_per_step"]),
                           ring_slots=int(raw["ring_slots"]))
            assert all(math.isfinite(row[field]) and row[field] >= 0
                       for field in ("device_us", "cpu_enqueue_us"))
            rows.append(row)
        elif run is not None and line and not line.startswith("command:"):
            processes[run]["metadata"].append(line)
    assert sorted(processes) == list(range(1, expected_processes + 1)), "unexpected process count"
    assert all(proc.get("exit_code") == 0 for proc in processes.values()), "incomplete/failed process"
    groups = collections.defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in DIMENSIONS) + (row["variant"],)
        groups[key].append(row)
    expected_groups = 24 if "control" in header else 72
    assert len(groups) == expected_groups, "unexpected benchmark matrix"
    for group in groups.values():
        for run in processes:
            rounds = [row["round"] for row in group if row["run"] == run]
            assert len(rounds) == expected_rounds and len(set(rounds)) == expected_rounds
            assert sorted(rounds) in (list(range(expected_rounds)), list(range(1, expected_rounds + 1)))
    return rows, groups, list(processes.values())


def summarize_groups(groups):
    result = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        group = dict(zip(DIMENSIONS + ("variant",), key))
        group["samples"] = len(rows)
        for field in ("iterations", "kernels_per_step", "ring_slots"):
            if field in rows[0]:
                values = sorted({row[field] for row in rows})
                assert len(values) == 1, "benchmark configuration changed within group"
                group[field] = values[0]
        for field in ("device_us", "cpu_enqueue_us"):
            per_process = collections.defaultdict(list)
            for row in rows:
                per_process[row["run"]].append(row[field])
            group[field + "_per_step"] = aggregate(per_process)
        result.append(group)
    return result


def summarize_differences(groups):
    pairs = []
    contexts = {key[:-1] for key in groups}
    for context in sorted(contexts, key=str):
        for first, second in DIFFERENCES:
            if context + (first,) not in groups or context + (second,) not in groups:
                continue
            a = {(row["run"], row["round"]): row for row in groups[context + (first,)]}
            b = {(row["run"], row["round"]): row for row in groups[context + (second,)]}
            assert len(a) == len(groups[context + (first,)]) and a.keys() == b.keys()
            item = {**dict(zip(DIMENSIONS, context)), "difference": first + " - " + second}
            for field in ("device_us", "cpu_enqueue_us"):
                per_process = collections.defaultdict(list)
                for run, round_number in sorted(a):
                    per_process[run].append(a[(run, round_number)][field] - b[(run, round_number)][field])
                item[field + "_per_step"] = aggregate(per_process)
            pairs.append(item)
    return pairs


def summarize_execution_differences(groups):
    result = []
    for context in sorted({key[1:] for key in groups}, key=str):
        if ("native",) + context not in groups or ("graph",) + context not in groups:
            continue
        native = {(row["run"], row["round"]): row for row in groups[("native",) + context]}
        graph = {(row["run"], row["round"]): row for row in groups[("graph",) + context]}
        assert native.keys() == graph.keys()
        item = {**dict(zip(DIMENSIONS[1:] + ("variant",), context)), "difference": "graph - native"}
        for field in ("device_us", "cpu_enqueue_us"):
            per_process = collections.defaultdict(list)
            for run, round_number in sorted(native):
                per_process[run].append(graph[(run, round_number)][field] - native[(run, round_number)][field])
            item[field + "_per_step"] = aggregate(per_process)
        result.append(item)
    return result


def summarize_telemetry(path):
    fields = {
        "sclk_mhz": "sclk clock speed:", "mclk_mhz": "mclk clock speed:",
        "edge_c": "Temperature (Sensor edge) (C)",
        "junction_c": "Temperature (Sensor junction) (C)",
        "memory_c": "Temperature (Sensor memory) (C)",
        "socket_power_w": "Current Socket Graphics Package Power (W)",
        "gpu_use_percent": "GPU use (%)",
    }
    by_card = collections.defaultdict(lambda: collections.defaultdict(list))
    timestamps = []
    samples, invalid = 0, 0
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        samples += 1
        try:
            data = json.loads(sample["data"]) if isinstance(sample["data"], str) else sample["data"]
        except (ValueError, KeyError):
            invalid += 1
            continue
        timestamps.append(sample["utc"])
        for card, measurements in data.items():
            for name, field in fields.items():
                match = re.search(r"[-+]?\d+(?:\.\d+)?", str(measurements.get(field, "")))
                if match:
                    by_card[card][name].append(float(match[0]))
    return {"source": source_metadata(path), "samples": samples, "invalid_samples": invalid,
            "first_utc": min(timestamps) if timestamps else None,
            "last_utc": max(timestamps) if timestamps else None,
            "scope": "All samples across the complete outer-process series, including loading, warmup and gaps",
            "variant_attribution": False,
            "cards": {card: {name: {"count": len(values), "min": min(values), "max": max(values)}
                             for name, values in sorted(metrics.items())}
                      for card, metrics in sorted(by_card.items())}}


def summarize_file(path, expected_processes, expected_rounds):
    rows, groups, processes = parse_measurements(path, expected_processes, expected_rounds)
    return {"source": source_metadata(path), "measurement_rows": len(rows), "processes": processes,
            "measurement_groups": summarize_groups(groups),
            "paired_control_differences": summarize_differences(groups),
            "paired_execution_differences": summarize_execution_differences(groups),
            "telemetry": summarize_telemetry(path.with_name(path.name + ".smi.jsonl"))}


def summarize_trace_pairs():
    series = {}
    raw_traces = []
    for label in ("native", "traced"):
        all_rows = []
        sources = []
        for pair in range(1, 4):
            path = HERE / f"matched-{pair}-{label}.txt"
            rows, _, processes = parse_measurements(path, 1, 3)
            for row in rows:
                row["run"] = pair
            all_rows.extend(rows)
            sources.append({"pair": pair, "source": source_metadata(path), "process": processes[0],
                            "telemetry": summarize_telemetry(path.with_name(path.name + ".smi.jsonl"))})
            if label == "traced":
                raw_traces.append({"pair": pair,
                                   "source": source_metadata(HERE / f"matched-{pair}-traced.csv.gz")})
        groups = collections.defaultdict(list)
        for row in all_rows:
            groups[tuple(row[name] for name in DIMENSIONS) + (row["variant"],)].append(row)
        series[label] = {"sources": sources, "rows": all_rows, "groups": groups}
    effects = []
    for key in sorted(series["native"]["groups"], key=str):
        item = {**dict(zip(DIMENSIONS + ("variant",), key)), "difference": "traced - untraced"}
        native = {(row["run"], row["round"]): row for row in series["native"]["groups"][key]}
        traced = {(row["run"], row["round"]): row for row in series["traced"]["groups"][key]}
        assert native.keys() == traced.keys()
        for field in ("device_us", "cpu_enqueue_us"):
            per_pair = collections.defaultdict(list)
            for pair, round_number in sorted(native):
                per_pair[pair].append(traced[(pair, round_number)][field] - native[(pair, round_number)][field])
            item[field + "_per_step"] = aggregate(per_pair)
        effects.append(item)
    return {
        "iterations": 128, "rounds_per_process": 3, "pairs": 3,
        "order": "Untraced then traced in pairs 1/3; traced then untraced in pair 2",
        "pairing": "Equal chain, execution, control and round in adjacent outer-process pairs",
        "native": {"sources": series["native"]["sources"],
                   "measurement_groups": summarize_groups(series["native"]["groups"]),
                   "paired_control_differences": summarize_differences(series["native"]["groups"])},
        "traced": {"sources": series["traced"]["sources"],
                   "measurement_groups": summarize_groups(series["traced"]["groups"]),
                   "paired_control_differences": summarize_differences(series["traced"]["groups"])},
        "paired_trace_effects": effects,
        "raw_timestamp_traces": raw_traces,
        "limitation": "Trace effects compare HIP-event whole-chain times; raw kernel timestamps are not subtracted from native events",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=pathlib.Path, default=HERE / "controls.txt")
    parser.add_argument("--model-chain", type=pathlib.Path, default=HERE / "model-chain.txt")
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "measurements.json.gz")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    result = {
        "primary_statistic": "Median of outer-process medians; seven rounds per process",
        "pairing": "Same context, execution, outer process and benchmark round; subtraction before aggregation",
        "percentile_method": "Linear interpolation at (n - 1) * fraction",
        "units": "Microseconds per benchmark step; a chain step contains two or three kernels",
        "datasets": {
            "controls": summarize_file(args.controls, args.processes, args.rounds),
            "model_chain": summarize_file(args.model_chain, args.processes, args.rounds),
        },
        "matched_trace_pairs": summarize_trace_pairs(),
        "limitations": [
            "Control variants change math, pointer and dependency behavior; they are not inference alternatives.",
            "Paired differences are not an exact additive decomposition or a legal fusion speedup estimate.",
            "CPU enqueue time may overlap GPU work; it is not an additive device-time component.",
            "Graphs reduce host enqueue work but may change runtime scheduling; device comparisons measure whole batches.",
            "Real model weights use synthetic inputs and repeated selected experts, not the complete model workload.",
            "Telemetry covers the whole process series and cannot be assigned to individual variants.",
        ],
    }
    payload = (json.dumps(result, indent=2) + "\n").encode()
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(payload, mtime=0))
    else:
        args.output.write_bytes(payload)
    for name, dataset in result["datasets"].items():
        print(name, "rows", dataset["measurement_rows"], "groups", len(dataset["measurement_groups"]),
              "paired-control groups", len(dataset["paired_control_differences"]))


if __name__ == "__main__":
    main()
