"""Paired native measurement; frozen executables, telemetry and exact output proof."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--binary-dir', type=pathlib.Path, default=pathlib.Path('/tmp/miso-deltanet-register'))
parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
parser.add_argument('--model', default='/home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf')
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--context', type=int)
parser.add_argument('--golden', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
harness = root / 'bench/model/results/2026-09-26-decode-investigation/run_measure.py'
proof_dir = args.binary_dir / 'proof'
proof_dir.mkdir(parents=True, exist_ok=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
executables = ({v: args.binary_dir / (v + '_context') for v in ('baseline', 'half_register')}
               if args.context else {'baseline': args.binary_dir / 'baseline_proof_decode', 'half_register': args.binary_dir / 'half_register_decode'})
label = f'ctx{args.context}' if args.context else 'native'
expected_ids = 65 if args.context else 513
proof = {'executables': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in executables.items()}, 'pairs': []}
for pair in range(1, args.pairs + 1):
    order = ['baseline', 'half_register'] if pair % 2 else ['half_register', 'baseline']
    records = {}
    for variant in order:
        name = f'{label}-{pair}-{variant}'
        output = args.output_dir / (name + '.txt')
        logits = proof_dir / (name + '.logits.bin')
        command = [str(executables[variant]), args.model]
        if args.context:
            command += [str(args.golden or root / 'tests/golden/ornith-layers.gguf'), str(args.context)]
        command += [str(logits)]
        subprocess.run([sys.executable, str(harness), str(output), '1', *command], check=True)
        log = output.read_text()
        ids_match = re.search(r'^all generated ids:(.*)$', log, re.M)
        if not ids_match:
            raise RuntimeError(f'{name}: missing complete generated IDs')
        ids = list(map(int, ids_match.group(1).split()))
        if len(ids) != expected_ids:
            raise RuntimeError(f'{name}: expected {expected_ids} IDs, got {len(ids)}')
        values = logits.read_bytes()
        records[variant] = {'ids': ids, 'bytes': values, 'logits_sha256': hashlib.sha256(values).hexdigest(), 'logits_bytes': len(values)}
    ids_equal = records['baseline']['ids'] == records['half_register']['ids']
    logits_equal = records['baseline']['bytes'] == records['half_register']['bytes']
    proof['pairs'].append({'pair': pair, 'order': order, 'generated_ids_count': expected_ids, 'ids_equal': ids_equal, 'final_logits_bitwise_equal': logits_equal, 'outputs': {k: {f: v[f] for f in ('logits_sha256', 'logits_bytes')} for k, v in records.items()}})
    (args.output_dir / (label + '-proof.json')).write_text(json.dumps(proof, indent=2) + '\n')
    print(f'pair {pair}: IDs equal={ids_equal}, final logits bitwise equal={logits_equal}', flush=True)
    if not ids_equal or not logits_equal:
        raise RuntimeError(f'pair {pair}: native output mismatch')
