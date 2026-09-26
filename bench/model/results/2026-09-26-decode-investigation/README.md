# Decode investigation raw data

Engine source: `33915aa`; benchmark binary hashes, compiler and host details in
`manifest.txt`. No production source changes. Analysis:
[`docs/research/decode-optimization-2026-09-26.md`](../../../../docs/research/decode-optimization-2026-09-26.md).

## Reproduce

From repository root, with the GPU idle:

```bash
cmake --preset default
cmake --build --preset default
mkdir -p /tmp/miso-decode-investigation
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py /tmp/baseline.txt 3 ./build/bench/bench_decode
```

`run_measure.py` records the exact command and exit codes, captures stdout/stderr,
and polls rocm-smi at 0.5-second intervals plus command runtime. Its telemetry
includes load/prefill/warm-up. Each benchmark discards its own warm-up.

The archived `context_probe.hip` is a one-off measurement driver linked to the
unchanged production engine. It clones only persistent state needed by decode;
scratch is overwritten by the next step. It verifies identical 64-token
predictions across repetitions and synchronized session/KV positions. Its input
is the repeated 17-token golden prompt; prefill and device-to-device restore are
outside the timed interval. Compile and link in separate steps (hipcc otherwise
interprets archives after a `.hip` input as HIP source):

```bash
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -std=c++20 -O3 --offload-arch=gfx906 -Isrc -Ikernels -c bench/model/results/2026-09-26-decode-investigation/context_probe.hip -o /tmp/miso-decode-investigation/context_probe.o
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --offload-arch=gfx906 /tmp/miso-decode-investigation/context_probe.o build/libmiso_engine.a build/libmiso_host.a -o /tmp/miso-decode-investigation/context_probe
python3 bench/model/results/2026-09-26-decode-investigation/run_measure.py /tmp/context-32768.txt 1 /tmp/miso-decode-investigation/context_probe /home/penta/llm-mi50/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf tests/golden/ornith-layers.gguf 32768
```

Use 17 or 4096 for the other contexts. Timings describe a 64-token interval
starting at the specified context, not a fixed-size-cache loop or ctx 0.

For timestamp profiles, wrap the benchmark command with:

```bash
/opt/rocm/bin/rocprof --timestamp on --stats -o /tmp/miso-decode-investigation/profile.csv ./build/bench/bench_decode
```

For the 32k profile substitute the context probe command above. The committed
`.csv.gz` files preserve every original dispatch row losslessly. Derived files
select the **last** measured repetition and can be regenerated:

```bash
python3 bench/model/results/2026-09-26-decode-investigation/summarize_profile.py bench/model/results/2026-09-26-decode-investigation/profile.csv.gz 256 /tmp/short
python3 bench/model/results/2026-09-26-decode-investigation/summarize_profile.py bench/model/results/2026-09-26-decode-investigation/profile-32k.csv.gz 64 /tmp/long
```

The analyzer asserts kernel counts, absence of prefill and exact span accounting.
A negative control requesting 530 tokens from the short trace is rejected because
it includes prefill. Family percentages have kernel-time, not wall-time,
denominators. The tiny signed overlaps are retained in the summary.

Hardware counters use `rocprof --timestamp on -i counters.txt -o <path>` around
`bench_decode`. The range is specific to this executable/benchmark, not a general
profiling filter; derive it again if dispatch counts change. It selects 1,094
rows starting at the measured short decode window. Counter statistics are
per-kernel medians across two passes; timestamps are not benchmark timings.
`SQ_WAVES` is a launch total, `VALUUtilization` an active-lane percentage and
`VALUBusy` the installed profiler's device-wide ALU utilization metric. Never
interpret any of them as measured resident occupancy.

Telemetry summaries use samples with reported GPU use >=90%, which still include
load transfers and prefill. Raw samples take precedence; a minimum clock below
the active median does not by itself identify throttling during timed decode.
The 32k context is synthetic and no claim of quality or typical prompt routing
is made. All generated performance figures must be compared to unprofiled A/B
runs before accepting a kernel change.

Text console logs and derived CSVs use normalized line endings/trailing whitespace;
measurement values are unchanged. Compressed dispatch CSVs are byte-for-byte raw.
