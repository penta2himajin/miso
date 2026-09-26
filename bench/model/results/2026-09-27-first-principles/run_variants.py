"""Sequential, interleaved native decode runs; telemetry is captured by the existing harness."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("binary_dir", type=Path)
p.add_argument("output_dir", type=Path)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--variants", default="baseline,mixed,pipeline,combined")
p.add_argument("--context", type=int)
p.add_argument("--model")
p.add_argument("--golden")
a = p.parse_args()
variants = a.variants.split(",")
harness = Path(__file__).resolve().parents[1] / "2026-09-26-decode-investigation/run_measure.py"
for run in range(a.runs):
    order = variants if run % 2 == 0 else list(reversed(variants))
    for variant in order:
        suffix = "decode" if a.context is None else "context"
        cmd = [str(a.binary_dir / f"{variant}_{suffix}")]
        label = "e2e" if a.context is None else f"ctx{a.context}"
        if a.context is not None:
            if not a.model or not a.golden:
                p.error("--model and --golden are required with --context")
            cmd += [a.model, a.golden, str(a.context)]
        out = a.output_dir / f"{label}-{run + 1}-{variant}.txt"
        subprocess.run([sys.executable, str(harness), str(out), "1", *cmd], check=True)
