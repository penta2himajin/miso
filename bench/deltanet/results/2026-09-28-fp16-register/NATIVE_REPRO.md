# Reproduce the register-boundary native experiment

Use this branch based on `f258ce1`, with no unrelated working changes. The temporary
integration fixes the target model's Q4_K/K4096/N2048 path; it is not the production
implementation. Do not run concurrent GPU jobs. Pick fresh binary and results paths.

```bash
set -e
RESULT=bench/deltanet/results/2026-09-28-fp16-register
BIN=/tmp/register-repeat-binaries
mkdir -p "$BIN"
cp src/engine/deltanet.hip "$BIN/original-deltanet.hip"
cp bench/model/bench_decode.hip "$BIN/original-bench-decode.hip"
cmake --preset default
cmake --build --preset default
cp build/bench/bench_decode "$BIN/original_decode"
cp build/libmiso_engine.a "$BIN/baseline_engine.a"
cp build/libmiso_host.a "$BIN/host.a"
git apply "$RESULT/native-instrument.patch"
cmake --build --preset default --target bench_decode
cp build/bench/bench_decode "$BIN/baseline_proof_decode"
git apply "$RESULT/native-register.patch"
cmake --build --preset default --target bench_decode test_model test_deltanet
cp build/bench/bench_decode "$BIN/half_register_decode"
cp build/libmiso_engine.a "$BIN/register_engine.a"
cp build/tests/test_model "$BIN/register_test_model"
cp build/tests/test_deltanet "$BIN/register_test_deltanet"
cp "$BIN/original-deltanet.hip" src/engine/deltanet.hip
cp "$BIN/original-bench-decode.hip" bench/model/bench_decode.hip
cmake --build --preset default
```

Compile one identical context host object and link it with each engine. Complete
all CPU builds before timing. Formatting this already formatted source is not needed.

```bash
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  --offload-arch=gfx906 -std=c++20 -O3 -DNDEBUG -Isrc -Ikernels \
  -c "$RESULT/context_probe.hip" -o "$BIN/context_probe.o"
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  --offload-arch=gfx906 "$BIN/context_probe.o" "$BIN/baseline_engine.a" \
  "$BIN/host.a" -o "$BIN/baseline_context"
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 \
  --offload-arch=gfx906 "$BIN/context_probe.o" "$BIN/register_engine.a" \
  "$BIN/host.a" -o "$BIN/half_register_context"
"$BIN/register_test_model"
"$BIN/register_test_deltanet"
python3 "$RESULT/run_native.py" --binary-dir "$BIN" \
  --output-dir /tmp/register-repeat-results --pairs 7
for context in 4096 32768; do
  python3 "$RESULT/run_native.py" --binary-dir "$BIN" \
    --output-dir /tmp/register-repeat-results --context "$context" --pairs 2
done
python3 "$RESULT/summarize_native.py" --output-dir /tmp/register-repeat-results
```

The instrumented short benchmark prints all513 generated IDs and writes final
logits after its events. Each context has one warmup and three restored64-token
measurements; each repeat checks IDs/logits against its warmup. The controller
compares actual baseline/candidate bytes and IDs before recording hashes. It
covers final logits, not every token's full-vocabulary logits. The existing
telemetry runner logs GPU clocks, temperatures and power throughout every process.
Raw logs and proof hashes accompany the summaries. Model/golden paths can be
overridden with `--model`/`--golden`.
