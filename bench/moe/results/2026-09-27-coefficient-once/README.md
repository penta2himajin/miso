# MoE coefficients only in workgroup 0 (2026-09-27)

Base: main `438fb45`, after PR #34. Decode keeps replicated top-8 ID selection
but computes the routing coefficients only in workgroup0, which is the only
workgroup that exports them. Prefill retains the default full-coefficient path.
Grid120/block256, weight row pipeline, arithmetic order and tolerances unchanged.
No launch, scratch buffer, stream, cross-workgroup dependency or runtime dependency
was added. The final block barrier is unconditional.

## Correctness and ISA

Red: `test_moe` compilation fails the new fourth-argument API (`red-build.txt`).
Green:6 MoE cases/386625 assertions pass, including120-workgroup coefficient
canaries, default full weights, tied/extreme finite logits, all17 real-router
fixtures on layers0/5, and frozen old gate/up bitwise comparisons of all4608
activations, IDs and coefficients. Existing golden/prefill/down tests are unchanged.
Full CTest:21/21 pass,87.97 seconds (`ctest.txt`).

The same-binary benchmark also checks setup IDs, coefficients, all gate/up
activations and whole MoE output bitwise for17 tokens on both layers on an explicit
nonblocking stream. Layer5 has no saved MoE input, so it explicitly uses
`layer0.moe_in` with layer5 weights for both controls.

Production `moe_gate_up_kernel` remains63 VGPR/64B LDS/scratch0; SGPR42→40;
compiled ceiling remains4 waves/SIMD. Prefill top-k VGPR48/LDS64/scratch0 and
ceiling5 remain; SGPR67→63. These are compiled ceilings, not achieved occupancy.

Candidate ISA (`candidate-coefficient-isa-selected.txt`, full gzip archived):
workgroup ID in s6 is compared with0 at0x5110, giving uniform mask s[26:27].
ID LDS store at0x6a7c is outside coefficient gating. Branch0x6a8c skips coefficient
max/exp/reduction/division/weight-store to0x6ec4 for nonzero workgroups.
EXEC is restored, waits complete and the unconditional barrier is at0x6ed4;
block0 then exports IDs/weights. Baseline coefficient exp begins0x6b18 before its
first workgroup-zero comparison0x6ec8. No precision change is involved.

## Micro protocol and findings

GPU initially idle (0% use, junction38C); all GPU jobs serialized. Default DVFS;
clock/memory-clock/temperature/power/use logs accompany every measurement.
Warmup500 launches excluded;2000 timed iterations,7 rotating-order rounds,
3 outer processes. HIP events bracket batches; CPU enqueue time is reported
separately and may overlap GPU work/backpressure. Model loading/router preparation
is outside setup and gate/up timing. Whole MoE includes router→gate/up→down.
Inputs rotate over17 golden activations/logits and phase changes each round.

Primary statistic: median of outer-process medians. Paired candidate-minus-baseline
increments are computed within the same process, layer, scope and round before
aggregation (`micro-summary.json`).

- Layer0 gate/up: paired−0.401us (−1.295%); all21 pairs faster.
- Layer5 gate/up: paired−0.325us (−1.051%); all21 pairs faster.
- Setup: about8.33us, paired−0.016/−0.014us, effectively flat.
- Whole MoE: layer0 paired−1.690us (−2.655%); layer5−0.347us (−0.572%).
  Layer5 has a retained +14.639us candidate outlier; its pooled mean is positive,
  unlike its per-process medians. Do not hide this tail or extrapolate a clean bound.

Setup writes IDs from all120 workgroups to global memory so the compiler cannot
remove ID selection; production exports only block0 IDs and other workgroups use
shared IDs. Thus setup absolute latency is a control, not the production setup
budget. The layer0 whole-MoE saving exceeds gate/up-only saving; cache/dispatch and
compiler scheduling can also contribute. Native end-to-end comparisons decide
adoption; the whole micro difference is not attributed entirely to coefficient math.

## Native comparison

Baseline and candidate decode binaries were frozen before timing; their hashes,
engine archives and source hashes are in `manifest.json`. Every process exits 0.
Variant order alternates within each set; all GPU work is serialized.

Short native `bench_decode`: seven fresh processes per variant, discarded 256-token
warmup and 256 measured tokens (context 273..529). Median 148.8 to 149.8 tok/s
(+0.672%); paired change median +0.535%, range -0.601%..+1.282%. Five pairs improve,
one ties and one regresses. The printed first-eight generated-ID prefixes match
across all fourteen processes; this is not a comparison of all 256 generated IDs.
The pp512 median is 1439.9 to 1439.7 tok/s (-0.014%, effectively flat).

Restored-context controls use one identical probe object linked with each frozen
engine archive. Snapshot prefill/restore is outside timing. Each process discards
one 64-token warmup, then measures three repetitions of the same 64 greedy tokens;
all generated IDs match within repetitions. Primary values below are medians of
process medians, not all repetitions treated as independent observations.

- Context 17, two pairs: 148.8615 to 149.4595 tok/s (+0.402%); mixed pair signs.
- Context 4096, three pairs: 139.625 to 139.886 tok/s (+0.187%). Pair changes are
  +0.113%, -0.769%, +0.786%; effectively flat/inconclusive. Pair 3 was added after
  the first two mixed signs; all results are retained.
- Context 32768, two pairs: 115.8435 to 117.3995 tok/s (+1.343%); both pairs improve
  (+1.608%, +1.078%), but the small sample cannot establish a general long-context gain.

Whole-run active telemetry (GPU use >=90%) includes loading, warmup and prefill,
so its medians are not decode-only clocks. Native runs span sclk 1485..1725 MHz,
mclk 800..1000 MHz, junction 38..83 C and power 33..271 W. The last five active
samples of each short process show 1725/1000 MHz and junction about 57..63 C;
there is no obvious differential throttling in those samples. This is not a
phase-synchronized or fixed-clock experiment. Micro active telemetry has sclk
1725 MHz, mclk 800..1000 MHz, junction 56..66 C and power 84..187 W.

Adoption: keep this small, bit-exact removal of redundant work. The repeatable
gate/up improvement, small positive short median and unchanged launch/VGPR/LDS
budgets support it. Native variability exceeds some observed gains, 4k is
inconclusive and the layer5 whole-MoE outlier remains; no universal decode speedup
or tail-latency bound is claimed.

## Reproduction

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
build/bench/bench_moe_coefficients --check
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py \
  bench/moe/results/2026-09-27-coefficient-once/micro.txt 3 \
  build/bench/bench_moe_coefficients --iterations 2000 --rounds 7 --warmup 500
python3 bench/moe/results/2026-09-27-coefficient-once/summarize_micro.py
python3 bench/moe/results/2026-09-27-coefficient-once/summarize.py
```

For the native comparison, first build main `438fb45` with the same preset and
freeze `build/bench/bench_decode` as `/tmp/miso-moe-softmax/baseline_decode` and
`build/libmiso_engine.a` as `baseline_engine.a`. Freeze the candidate equivalents
after building this branch; preserve `build/libmiso_host.a` as `host.a`. Compile
the archived probe once, then link that same object against both engine archives:

```bash
hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -std=c++20 -O3 \
  --offload-arch=gfx906 -I src -I kernels -c \
  bench/model/results/2026-09-26-decode-investigation/context_probe.hip \
  -o /tmp/miso-moe-softmax/context_probe.o
for variant in baseline candidate; do
  hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -std=c++20 -O3 \
    --offload-arch=gfx906 /tmp/miso-moe-softmax/context_probe.o \
    /tmp/miso-moe-softmax/${variant}_engine.a /tmp/miso-moe-softmax/host.a \
    -o /tmp/miso-moe-softmax/${variant}_context
done
python3 bench/model/results/2026-09-27-first-principles/run_variants.py \
  /tmp/miso-moe-softmax bench/moe/results/2026-09-27-coefficient-once \
  --runs 7 --variants baseline,candidate
python3 bench/moe/results/2026-09-27-coefficient-once/run_contexts.py
python3 bench/moe/results/2026-09-27-coefficient-once/run_extra_4k.py
python3 bench/moe/results/2026-09-27-coefficient-once/summarize.py
```

Run the measurement commands separately from other GPU jobs. The archived
scripts use this host's model/fixture paths; rerunning into this directory replaces
the corresponding logs. `red-bench-build.txt` and the first `green-build.txt`
record an incidental frozen-wrapper ADL ambiguity, fixed by explicit qualification;
the intentional TDD Red is `red-build.txt` and the successful build is `final-build.txt`.
Readable diagnostics have trailing whitespace removed; their original bytes are
preserved in the matching gzip files.
