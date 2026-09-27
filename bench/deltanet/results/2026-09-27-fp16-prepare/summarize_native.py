"""Summarize matched native throughput pairs and telemetry without discarding runs."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
here = parser.parse_args().output_dir
proof = json.loads((here / 'native-proof.json').read_text())
rows = []
telemetry = collections.defaultdict(list)
for pair in proof['pairs']:
    assert pair['ids_equal'] and pair['final_logits_bitwise_equal']
    row = {'pair': pair['pair'], 'order': pair['order'], 'variants': {}}
    for variant in ('baseline', 'half_only'):
        path = here / f"native-{pair['pair']}-{variant}.txt"
        log = path.read_text()
        assert log.rstrip().endswith('exit: 0')
        pp = re.search(r'measured: pp512 ([\d.]+) ms, ([\d.]+) tok/s', log)
        decode = re.search(r'measured: 256 tokens at ctx 273\.\.529: ([\d.]+) ms/token, ([\d.]+) tok/s', log)
        row['variants'][variant] = {'decode_tok_s': float(decode[2]), 'pp512_tok_s': float(pp[2]), 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for line in pathlib.Path(str(path) + '.smi.jsonl').read_text().splitlines():
            for card in json.loads(json.loads(line)['data']).values():
                if float(card.get('GPU use (%)', 0)) < 90:
                    continue
                for key, value in card.items():
                    if any(word in key.lower() for word in ('temperature', 'power', 'clock speed')):
                        numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', str(value))
                        if numbers:
                            telemetry[key].append(float(numbers[0]))
    for scope in ('decode', 'pp512'):
        a, b = [row['variants'][variant][scope + '_tok_s'] for variant in ('baseline', 'half_only')]
        row[scope + '_paired_percent'] = (b / a - 1) * 100
    rows.append(row)
result = {'pairs': rows, 'telemetry_note': 'GPU-use>=90% samples include load/setup/warmup; not phase-aligned.', 'telemetry': {k: {'min': min(v), 'median': statistics.median(v), 'max': max(v)} for k, v in telemetry.items()}}
for scope in ('decode', 'pp512'):
    changes = [r[scope + '_paired_percent'] for r in rows]
    result[scope] = {'median_tok_s': {v: statistics.median(r['variants'][v][scope + '_tok_s'] for r in rows) for v in ('baseline', 'half_only')}, 'paired_percent_median': statistics.median(changes), 'paired_percent_range': [min(changes), max(changes)], 'faster_pairs': sum(x > 0 for x in changes), 'pair_count': len(rows)}
(here / 'native-summary.json').write_text(json.dumps(result, indent=2) + '\n')
for scope in ('decode', 'pp512'):
    print(scope, json.dumps(result[scope]))

contexts = []
for proof_path in sorted(here.glob('ctx*-proof.json')):
    context = int(re.search(r'ctx(\d+)', proof_path.name)[1])
    context_proof = json.loads(proof_path.read_text())
    pairs = []
    context_telemetry = collections.defaultdict(list)
    for pair in context_proof['pairs']:
        assert pair['ids_equal'] and pair['final_logits_bitwise_equal']
        rates = {}
        for variant in ('baseline', 'half_only'):
            path = here / f"ctx{context}-{pair['pair']}-{variant}.txt"
            text = path.read_text()
            assert text.rstrip().endswith('exit: 0')
            rates[variant] = float(re.search(r'median ctx=\d+ ms_token=[\d.]+ tok_s=([\d.]+)', text)[1])
            for line in pathlib.Path(str(path) + '.smi.jsonl').read_text().splitlines():
                for card in json.loads(json.loads(line)['data']).values():
                    if float(card.get('GPU use (%)', 0)) < 90:
                        continue
                    for key, value in card.items():
                        if any(word in key.lower() for word in ('temperature', 'power', 'clock speed')):
                            numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', str(value))
                            if numbers:
                                context_telemetry[key].append(float(numbers[0]))
        pairs.append({'pair': pair['pair'], 'order': pair['order'], 'tok_s': rates, 'paired_percent': (rates['half_only'] / rates['baseline'] - 1) * 100})
    changes = [p['paired_percent'] for p in pairs]
    contexts.append({'context': context, 'pairs': pairs, 'median_tok_s': {v: statistics.median(p['tok_s'][v] for p in pairs) for v in ('baseline', 'half_only')}, 'paired_percent_median': statistics.median(changes), 'paired_percent_range': [min(changes), max(changes)], 'faster_pairs': sum(d > 0 for d in changes), 'pair_count': len(changes), 'telemetry': {k: {'min': min(v), 'median': statistics.median(v), 'max': max(v)} for k, v in context_telemetry.items()}})
    print(f'context={context}', json.dumps({k: v for k, v in contexts[-1].items() if k not in ('pairs', 'telemetry')}))
if contexts:
    (here / 'context-summary.json').write_text(json.dumps({'contexts': contexts, 'telemetry_note': result['telemetry_note']}, indent=2) + '\n')
