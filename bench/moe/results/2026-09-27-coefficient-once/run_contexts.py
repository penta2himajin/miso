"""Serialized, alternating baseline/candidate restored-context measurements."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNNER = REPO / 'bench/model/results/2026-09-27-first-principles/run_variants.py'
MODEL = '/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf'
GOLDEN = str(REPO / 'tests/golden/ornith-layers.gguf')
for context in (17, 4096, 32768):
    subprocess.run([sys.executable, str(RUNNER), '/tmp/miso-moe-softmax', str(HERE),
                    '--runs', '2', '--variants', 'baseline,candidate', '--context', str(context),
                    '--model', MODEL, '--golden', GOLDEN], cwd=REPO, check=True)
