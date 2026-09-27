"""Third 4k pair to check the mixed signs in the first two pairs; never overwrites them."""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
HARNESS = REPO / 'bench/model/results/2026-09-26-decode-investigation/run_measure.py'
MODEL = '/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf'
GOLDEN = str(REPO / 'tests/golden/ornith-layers.gguf')
for variant in ('baseline', 'candidate'):
    out = HERE / ('ctx4096-3-' + variant + '.txt')
    subprocess.run([sys.executable, str(HARNESS), str(out), '1',
                    '/tmp/miso-moe-softmax/' + variant + '_context', MODEL, GOLDEN, '4096'],
                   cwd=REPO, check=True)
