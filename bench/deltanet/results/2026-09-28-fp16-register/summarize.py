"""Summarize matched native-event batches; retain every process and round."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import statistics

VARIANTS = ('baseline', 'float_control', 'dual_float', 'dual_half', 'half_only', 'half_register')
SCOPES = ('producer', 'consumer', 'chain')


def expected_order(round_id):
    count = len(SCOPES) * len(VARIANTS)
    indices = [(order + round_id) % count if round_id % 2 == 0
               else (count + round_id - order) % count for order in range(count)]
    return [(SCOPES[index // len(VARIANTS)], VARIANTS[index % len(VARIANTS)])
            for index in indices]


def validate_ordering(rows):
    """Check the seven-round design and each observed process/layer/round sequence."""
    design = [expected_order(round_id) for round_id in range(7)]
    cases = {(scope, variant) for scope in SCOPES for variant in VARIANTS}
    assert all(len(order) == len(cases) and set(order) == cases for order in design)
    comparisons = []
    for scope in SCOPES:
        for reference in ('baseline', 'float_control', 'half_only'):
            for variant in VARIANTS:
                if variant == reference:
                    continue
                before = sum(order.index((scope, variant)) < order.index((scope, reference))
                             for order in design)
                assert sorted((before, 7 - before)) == [3, 4], (scope, reference, variant)
                comparisons.append({'scope': scope, 'reference': reference, 'variant': variant,
                                    'before': before, 'after': 7 - before})
    observed = collections.defaultdict(list)
    for row in rows:
        observed[row['process'], row['layer'], row['round']].append((row['scope'], row['variant']))
    assert {key[2] for key in observed} == set(range(7)), 'this balanced protocol requires seven rounds'
    for key, order in observed.items():
        assert order == expected_order(key[2]), f'unexpected measurement order: {key}'
    return {'design_rounds': 7, 'cases_per_round': len(cases),
            'observed_round_groups': len(observed), 'observed_order_matches': True,
            'comparisons': comparisons}


def summarize(path):
    rows, exits, process = [], [], None
    for line in path.read_text().splitlines():
        start = re.match(r'run (\d+) utc ', line)
        if start:
            process = int(start[1])
        if line.startswith('exit: '):
            exits.append(int(line.split(':')[1]))
        values = dict(re.findall(r'(\w+)=([^\s]+)', line))
        required = ('layer', 'scope', 'variant', 'round', 'gpu_us', 'cpu_enqueue_us')
        if all(key in values for key in required):
            assert process is not None
            rows.append({'process': process, 'layer': int(values['layer']),
                         'scope': values['scope'], 'variant': values['variant'],
                         'round': int(values['round']), 'gpu_us': float(values['gpu_us']),
                         'cpu_enqueue_us': float(values['cpu_enqueue_us'])})
    assert exits and all(code == 0 for code in exits), exits
    assert len(exits) == len({row['process'] for row in rows})
    lookup = {(r['process'], r['layer'], r['scope'], r['variant'], r['round']): r for r in rows}
    assert len(lookup) == len(rows), 'duplicate measurement'
    ordering = validate_ordering(rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['layer'], row['scope']].append(row)
    findings = []
    for (layer, scope), group in sorted(groups.items()):
        assert scope in SCOPES
        assert {r['variant'] for r in group} == set(VARIANTS)
        processes = sorted({r['process'] for r in group})
        rounds = sorted({r['round'] for r in group})
        for process_id in processes:
            for variant in VARIANTS:
                assert all((process_id, layer, scope, variant, n) in lookup for n in rounds)
        for variant in VARIANTS:
            medians, deltas, percents, against_control = [], [], [], []
            against_half, half_percents = [], []
            for process_id in processes:
                chosen = [lookup[process_id, layer, scope, variant, n] for n in rounds]
                medians.append({'process': process_id,
                                'gpu_us': statistics.median(r['gpu_us'] for r in chosen),
                                'cpu_enqueue_us': statistics.median(r['cpu_enqueue_us'] for r in chosen)})
                for n, r in zip(rounds, chosen):
                    base = lookup[process_id, layer, scope, 'baseline', n]['gpu_us']
                    control = lookup[process_id, layer, scope, 'float_control', n]['gpu_us']
                    half_only = lookup[process_id, layer, scope, 'half_only', n]['gpu_us']
                    deltas.append(r['gpu_us'] - base)
                    percents.append((r['gpu_us'] / base - 1) * 100)
                    against_control.append(r['gpu_us'] - control)
                    against_half.append(r['gpu_us'] - half_only)
                    half_percents.append((r['gpu_us'] / half_only - 1) * 100)
            findings.append({'layer': layer, 'scope': scope, 'variant': variant,
                             'process_medians': medians,
                             'median_of_process_medians_us': statistics.median(m['gpu_us'] for m in medians),
                             'paired_gpu_deltas_us': deltas,
                             'paired_delta_median_us': statistics.median(deltas),
                             'paired_percent_changes': percents,
                             'paired_percent_median': statistics.median(percents),
                             'paired_delta_range_us': [min(deltas), max(deltas)],
                             'faster_pairs': sum(d < 0 for d in deltas),
                             'pair_count': len(deltas),
                             'deltas_against_frozen_control_us': against_control,
                             'median_delta_against_frozen_control_us': statistics.median(against_control),
                             'paired_deltas_against_half_only_us': against_half,
                             'paired_delta_against_half_only_median_us': statistics.median(against_half),
                             'paired_delta_against_half_only_range_us': [min(against_half), max(against_half)],
                             'paired_percent_changes_against_half_only': half_percents,
                             'paired_percent_against_half_only_median': statistics.median(half_percents),
                             'paired_percent_against_half_only_range': [min(half_percents), max(half_percents)],
                             'faster_pairs_against_half_only': sum(d < 0 for d in against_half)})
    telemetry_path = pathlib.Path(str(path) + '.smi.jsonl')
    fields = collections.defaultdict(list)
    for line in telemetry_path.read_text().splitlines():
        for card in json.loads(json.loads(line)['data']).values():
            if float(card.get('GPU use (%)', 0)) < 90:
                continue
            for key, value in card.items():
                if any(word in key.lower() for word in ('temperature', 'power', 'clock speed')):
                    numbers = re.findall(r'[0-9]+(?:\.[0-9]+)?', str(value))
                    if numbers:
                        fields[key].append(float(numbers[0]))
    return {'log': path.name, 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'telemetry_sha256': hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
            'row_count': len(rows), 'process_count': len(exits), 'findings': findings,
            'ordering_validation': ordering,
            'telemetry_note': 'GPU-use>=90% samples include load/setup/warmup/checks; not phase-aligned.',
            'telemetry': {k: {'min': min(v), 'median': statistics.median(v), 'max': max(v)}
                          for k, v in fields.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    result = summarize(args.log)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['row_count']} rows / {result['process_count']} processes")
    for finding in result['findings']:
        if finding['scope'] == 'chain':
            print(f"layer={finding['layer']} variant={finding['variant']} "
                  f"median_us={finding['median_of_process_medians_us']:.6f} "
                  f"paired_delta_us={finding['paired_delta_median_us']:.6f} "
                  f"paired_percent={finding['paired_percent_median']:.4f} "
                  f"faster_pairs={finding['faster_pairs']}/{finding['pair_count']}")
