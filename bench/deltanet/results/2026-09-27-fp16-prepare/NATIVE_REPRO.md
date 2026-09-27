# Reproduce the native experiment

Use a clean checkout of the measurement branch with base `af54ec7`. Do not apply
these temporary patches over unrelated work. The engine patch deliberately fixes
the measured model's Q4_K/K4096/N2048 output dispatch; it is not a production API.
No new output buffer or cast launch is added. Prefill is unchanged.

The sequence below preserves the original source files, freezes both executables
and engine archives, and restores the sources before running measurements.
Do not run concurrent GPU tests or benchmarks. Choose fresh output paths.

```bash
RESULT=bench/deltanet/results/2026-09-27-fp16-prepare
BIN=/tmp/prepare-repeat-binaries
mkdir -p "$BIN"
cp src/engine/deltanet.hip "$BIN/original-deltanet.hip"
cp bench/model/bench_decode.hip "$BIN/original-bench-decode.hip"
cmake --preset default
cmake --build --preset default --target bench_decode
cp build/bench/bench_decode "$BIN/baseline_decode"
cp build/libmiso_engine.a "$BIN/baseline_engine.a"
cp build/libmiso_host.a "$BIN/host.a"
git apply "$RESULT/native-instrument.patch"
cmake --build --preset default --target bench_decode
cp build/bench/bench_decode "$BIN/baseline_proof_decode"
git apply "$RESULT/native-half-only.patch"
cmake --build --preset default --target bench_decode test_deltanet test_model
ctest --preset default -R '^(test_deltanet|test_model)$'
cp build/bench/bench_decode "$BIN/half_only_decode"
cp build/libmiso_engine.a "$BIN/half_only_engine.a"
cp "$BIN/original-deltanet.hip" src/engine/deltanet.hip
cp "$BIN/original-bench-decode.hip" bench/model/bench_decode.hip
cmake --build --preset default --target bench_decode test_deltanet test_model
python3 "$RESULT/run_native.py" --binary-dir "$BIN" \
  --output-dir /tmp/prepare-repeat-results --pairs 7
```

The output instrumentation prints all513 predictions and dumps the final logits
after the timing events. The controller compares actual bytes before recording
hashes. The checked-in `native-proof.json` links fourteen final dumps, which are
identical; `native-final-logits.bin.gz` retains one copy. It does not compare the
full-vocabulary logits at every intermediate token.

Compile one identical restored-context host object and link it against each
frozen engine. Its four repeats restore recurrent/KV state outside events. Round0
warms up; three rounds are measured. Each repeat checks all64 generated IDs and
final logits against round0. Cross-binary comparisons cover65 IDs including the
initial prediction, and final logits.

```bash
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  --offload-arch=gfx906 -std=c++20 -O3 -DNDEBUG -Isrc -Ikernels \
  -c "$RESULT/context_probe.hip" -o "$BIN/context_probe.o"
for variant in baseline half_only; do
  /opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
    --offload-arch=gfx906 "$BIN/context_probe.o" "$BIN/${variant}_engine.a" \
    "$BIN/host.a" -o "$BIN/${variant}_context"
done
for context in 4096 32768; do
  python3 "$RESULT/run_native.py" --binary-dir "$BIN" \
    --output-dir /tmp/prepare-repeat-results --context "$context" --pairs 2
done
python3 "$RESULT/summarize_native.py" --output-dir /tmp/prepare-repeat-results
```

The model and golden fixture paths can be overridden with `--model`/`--golden`.
The controller uses the existing ADR002 telemetry runner. Metadata includes GPU
clocks, temperatures and power throughout each process; samples are not aligned
to the individual event intervals. Raw logs must accompany derived summaries.
