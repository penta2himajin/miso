"""Summarize a final contiguous decode window; reject prefill or incomplete tokens.
Usage: python3 summarize_profile.py trace.csv.gz tokens output_prefix
For the context probe use 64 (last measured repeat); for bench_decode use 256.
"""
import collections
import csv
import gzip
import json
import pathlib
import sys

source, tokens, output = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
with gzip.open(source, 'rt') as f:
    rows = list(csv.DictReader(f))
starts = [i for i, r in enumerate(rows) if 'embed_q4k_kernel' in r['KernelName']]
assert len(starts) >= tokens
rows = rows[starts[-tokens]:]
assert len(rows) == 365 * tokens, 'unexpected dispatch count: inspect before summarizing'
assert all('prefill' not in r['KernelName'] and 'qgemm' not in r['KernelName'] for r in rows)
assert sum('argmax_final_kernel' in r['KernelName'] for r in rows) == tokens
assert all(int(r['DurationNs']) == int(r['EndNs']) - int(r['BeginNs']) for r in rows)

def family(r):
    n = r['KernelName']
    if 'gemv_kernel' in n:
        return 'LM head' if int(r['grd']) == 1920 * 256 else 'Dense projections'
    if 'moe_' in n:
        return 'MoE'
    if 'add_rmsnorm' in n:
        return 'Residual RMSNorm'
    if 'deltanet_step' in n:
        return 'DeltaNet recurrence'
    if 'attn_' in n:
        return 'Attention core/prep'
    return 'Embed/argmax'

kernels, families = collections.defaultdict(list), collections.defaultdict(list)
for r in rows:
    v = int(r['DurationNs'])
    kernels[(r['KernelName'], r['grd'])].append(v)
    families[family(r)].append(v)
total = sum(sum(v) for v in families.values())
for label, groups in [('kernels', kernels), ('families', families)]:
    with open(str(output) + '-' + label + '.csv', 'w') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['name', 'calls_per_token', 'us_per_token', 'kernel_time_percent'])
        for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
            w.writerow([k, len(v)/tokens, sum(v)/tokens/1000, sum(v)/total*100])
gaps = [int(b['BeginNs']) - int(a['EndNs']) for a, b in zip(rows, rows[1:])]
span = int(rows[-1]['EndNs']) - int(rows[0]['BeginNs'])
info = dict(source=source, tokens=tokens, dispatches=len(rows),
            first_index=rows[0]['Index'], last_index=rows[-1]['Index'],
            kernel_us_per_token=total/tokens/1000, span_us_per_token=span/tokens/1000,
            positive_gap_ns=sum(max(0, x) for x in gaps),
            negative_overlap_ns=-sum(min(0, x) for x in gaps),
            positive_gap_percent=sum(max(0, x) for x in gaps)/span*100)
assert span == total + sum(gaps)
pathlib.Path(str(output) + '-summary.json').write_text(json.dumps(info, indent=2)+'\n')
print(json.dumps(info, indent=2))
