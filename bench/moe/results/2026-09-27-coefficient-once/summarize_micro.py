"""Three process medians and within-round frozen/current MoE comparisons."""
import collections
import csv
import hashlib
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
source = HERE / 'micro.txt'
rows, run = [], 0
for line in source.read_text().splitlines():
    marker = re.fullmatch(r'run (\d+) utc (.+)', line)
    if marker:
        run = int(marker[1])
    cells = next(csv.reader([line]))
    if len(cells) == 8 and cells[0] in ('0', '5'):
        rows.append({'run': run, 'layer': int(cells[0]), 'down_type': cells[1],
            'scope': cells[2], 'variant': cells[3], 'round': int(cells[4]),
            'iterations': int(cells[5]), 'device_us': float(cells[6]), 'enqueue_us': float(cells[7])})
assert len(rows) == 3 * 7 * 2 * 3 * 2
assert source.read_text().count('exit: 0') == 3

def stats(values):
    return {'median': statistics.median(values), 'min': min(values),
            'max': max(values), 'mean': statistics.mean(values)}

def aggregate(groups):
    medians = [statistics.median(v) for _, v in sorted(groups.items())]
    return {'primary_median_of_process_medians': statistics.median(medians),
            'process_medians': medians, 'pooled': stats([x for v in groups.values() for x in v])}

groups = collections.defaultdict(list)
paired = collections.defaultdict(dict)
for row in rows:
    groups[(row['layer'], row['scope'], row['variant'])].append(row)
    paired[(row['run'], row['layer'], row['scope'], row['round'])][row['variant']] = row
result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'rows': len(rows), 'processes': 3, 'rounds_per_process': 7,
          'primary_statistic': 'Median of process medians; deltas paired before aggregation',
          'groups': [], 'paired_differences': []}
for key, series in sorted(groups.items()):
    record = dict(zip(('layer', 'scope', 'variant'), key))
    for field in ('device_us', 'enqueue_us'):
        by_run = collections.defaultdict(list)
        for row in series:
            by_run[row['run']].append(row[field])
        record[field] = aggregate(by_run)
    result['groups'].append(record)
for layer, scope in sorted({(r['layer'], r['scope']) for r in rows}):
    times, percents = collections.defaultdict(list), collections.defaultdict(list)
    for (run, lo, sc, round_number), pair in sorted(paired.items()):
        if (lo, sc) != (layer, scope):
            continue
        assert set(pair) == {'baseline', 'candidate'}
        b, c = pair['baseline']['device_us'], pair['candidate']['device_us']
        times[run].append(c - b); percents[run].append((c / b - 1) * 100)
    result['paired_differences'].append({'layer': layer, 'scope': scope,
        'candidate_minus_baseline_us': aggregate(times), 'percent': aggregate(percents)})
(HERE / 'micro-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['paired_differences'], indent=2))
