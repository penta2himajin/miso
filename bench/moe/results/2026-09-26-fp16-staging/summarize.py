"""Recompute timing, telemetry and hardware-counter summaries from this directory."""
import csv
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
summary = {}
for prefix, candidate in [('', 'candidate'), ('vector-', 'vector')]:
    timing = {}
    for variant in ['baseline', candidate]:
        decode, prefill = [], []
        for p in sorted(root.glob(f'{prefix}e2e-*-{variant}.txt')):
            text = p.read_text()
            decode += [float(x) for x in re.findall(r'measured: 256 tokens.*?([\d.]+) tok/s', text)]
            prefill += [float(x) for x in re.findall(r'measured: pp512 .*?([\d.]+) tok/s', text)]
        if decode:
            timing[variant] = dict(decode=decode, median_decode=statistics.median(decode),
                                   pp512=prefill, median_pp512=statistics.median(prefill))
    if len(timing) == 2:
        timing['gain_percent'] = 100 * (timing[candidate]['median_decode'] /
                                        timing['baseline']['median_decode'] - 1)
    summary[candidate] = timing

summary['contexts'] = {}
for p in sorted(root.glob('context-*.txt')):
    m = re.search(r'median ctx=(\d+) ms_token=([\d.]+) tok_s=([\d.]+)', p.read_text())
    if m:
        summary['contexts'][p.name] = dict(ctx=int(m[1]), ms_token=float(m[2]), tok_s=float(m[3]))

summary['telemetry'] = {}
for p in sorted(root.glob('*.smi.jsonl')):
    active = []
    for line in p.read_text().splitlines():
        data = json.loads(line)['data']
        if not data.strip().startswith('{'):
            continue
        card = json.loads(data)['card0']
        if float(card['GPU use (%)']) >= 90:
            active.append(card)
    if not active:
        continue
    def values(key):
        return [float(re.search(r'[\d.]+', r[key]).group()) for r in active]
    def span(key):
        v = values(key)
        return [min(v), statistics.median(v), max(v)]
    summary['telemetry'][p.name] = dict(
        samples=len(active), sclk_MHz_min_median_max=span('sclk clock speed:'),
        mclk_MHz_min_median_max=span('mclk clock speed:'),
        junction_max_C=max(values('Temperature (Sensor junction) (C)')),
        power_max_W=max(values('Current Socket Graphics Package Power (W)')))

summary['counters'] = {}
for p in sorted(root.glob('lds-*.csv')):
    groups = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            groups.setdefault(r['KernelName'], []).append(r)
    summary['counters'][p.name] = {
        name: dict(samples=len(rows), **{
            key: statistics.median(float(r[key]) for r in rows)
            for key in ['LDSBankConflict', 'SQ_WAVES', 'SQ_INSTS_LDS', 'SQ_LDS_BANK_CONFLICT']})
        for name, rows in groups.items()}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'telemetry'}, indent=2))
