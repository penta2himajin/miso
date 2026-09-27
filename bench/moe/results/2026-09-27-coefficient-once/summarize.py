"""Native end-to-end medians, matched process comparisons and whole-run telemetry."""
import collections
import hashlib
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
result = {'e2e': {}, 'contexts': {}, 'paired_short': [], 'telemetry': {}}
for variant in ('baseline', 'candidate'):
    rates, pp = [], []
    for p in sorted(HERE.glob('e2e-*-' + variant + '.txt')):
        text = p.read_text()
        rate = re.findall(r'measured: 256.*?([0-9.]+) tok/s', text)
        pre = re.findall(r'measured: pp512.*?([0-9.]+) tok/s', text)
        assert len(rate) == len(pre) == 1 and 'exit: 0' in text, p
        rates.append(float(rate[0])); pp.append(float(pre[0]))
    if rates:
        result['e2e'][variant] = {'runs': rates, 'median': statistics.median(rates),
                                 'pp512_runs': pp, 'pp512_median': statistics.median(pp)}
if len(result['e2e']) == 2:
    baseline = result['e2e']['baseline']['runs']; candidate = result['e2e']['candidate']['runs']
    assert len(baseline) == len(candidate)
    result['paired_short'] = [{'run': i + 1, 'baseline_tok_s': b, 'candidate_tok_s': c,
                               'percent': (c / b - 1) * 100} for i, (b, c) in enumerate(zip(baseline, candidate))]
    result['short_median_change_percent'] = (result['e2e']['candidate']['median'] /
                                             result['e2e']['baseline']['median'] - 1) * 100
for p in sorted(HERE.glob('ctx*-*-*.txt')):
    text = p.read_text()
    matches = re.findall(r'median ctx=(\d+) ms_token=([0-9.]+) tok_s=([0-9.]+)', text)
    if matches:
        assert len(matches) == 1 and 'exit: 0' in text and 'ids_match=no' not in text, p
        ctx, ms, tps = matches[0]
        variant = p.stem.split('-')[-1]
        result['contexts'].setdefault(ctx, {}).setdefault(variant, []).append(float(tps))
for ctx, values in result['contexts'].items():
    baseline, candidate = values['baseline'], values['candidate']
    assert len(baseline) == len(candidate), ctx
    values['baseline_median'] = statistics.median(baseline)
    values['candidate_median'] = statistics.median(candidate)
    values['median_change_percent'] = (values['candidate_median'] / values['baseline_median'] - 1) * 100
    values['paired_changes_percent'] = [(c / b - 1) * 100 for b, c in zip(baseline, candidate)]
    values['paired_change_median_percent'] = statistics.median(values['paired_changes_percent'])
for p in sorted(HERE.glob('*.smi.jsonl')):
    samples, fields = 0, collections.defaultdict(list)
    for line in p.read_text().splitlines():
        sample = json.loads(line)
        for card in json.loads(sample['data']).values():
            if float(card.get('GPU use (%)', 0)) < 90:
                continue
            samples += 1
            for key, value in card.items():
                if any(word in key.lower() for word in ('temperature', 'power', 'clock speed')):
                    nums = re.findall(r'[0-9]+(?:\.[0-9]+)?', str(value))
                    if nums:
                        fields[key].append(float(nums[0]))
    result['telemetry'][p.name] = {'active_samples': samples,
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        'measurements': {k: {'min': min(v), 'median': statistics.median(v), 'max': max(v)}
                         for k, v in fields.items()}}
result['telemetry_note'] = 'GPU use >=90% samples include loading/warmup/prefill; not decode-only or variant-specific clocks.'
(HERE / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('e2e', 'paired_short', 'contexts')}, indent=2))
