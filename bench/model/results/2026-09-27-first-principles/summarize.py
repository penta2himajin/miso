"""Summarize native runs and active-GPU telemetry without treating profiler times as native."""
import collections
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
data = {"e2e": {}, "contexts": {}, "telemetry": {}}
for v in ("baseline", "mixed", "pipeline", "combined"):
    rates, pp = [], []
    for p in sorted(root.glob(f"e2e-*-{v}.txt")):
        s = p.read_text()
        rates += [float(x) for x in re.findall(r"measured: 256.*?([0-9.]+) tok/s", s)]
        pp += [float(x) for x in re.findall(r"measured: pp512.*?([0-9.]+) tok/s", s)]
    if rates:
        data["e2e"][v] = dict(runs=rates, median=statistics.median(rates),
                             pp512_runs=pp, pp512_median=statistics.median(pp))
for p in sorted(root.glob("ctx*-*-*.txt")):
    matches = re.findall(r"median ctx=(\d+) ms_token=([0-9.]+) tok_s=([0-9.]+)", p.read_text())
    if matches:
        variant = p.stem.split("-")[-1]
        ctx, ms, tps = matches[0]
        data["contexts"].setdefault(ctx, {}).setdefault(variant, []).append(float(tps))
for p in sorted(root.glob("*.smi.jsonl")):
    fields = collections.defaultdict(list)
    clocks = collections.Counter()
    for line in p.read_text().splitlines():
        sample = json.loads(line)
        for card in json.loads(sample["data"]).values():
            if float(card.get("GPU use (%)", 0)) < 90:
                continue
            clocks[card.get("sclk clock speed:", "unknown")] += 1
            for key, value in card.items():
                if "temperature" in key.lower() or "power" in key.lower():
                    try:
                        fields[key].append(float(value))
                    except ValueError:
                        pass
    data["telemetry"][p.name] = dict(
        sclk_samples=dict(clocks),
        measurements={k: dict(min=min(v), median=statistics.median(v), max=max(v))
                      for k, v in fields.items() if v})
data["telemetry_note"] = "GPU use >=90% samples include load, warm-up and prefill; not decode-only."
(root / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps({k: data[k] for k in ("e2e", "contexts")}, indent=2))
