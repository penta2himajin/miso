# REJECT: dual-stream overlap of independent decode GEMVs

Base: post-KEEP counters (K=4096 VALUBusy ~14%; peers: wave-overlap /
latency hiding). Hypothesis: launch independent same-`xn` GEMVs on a side
stream — DeltaNet Q6 `qkv` ∥ Q4 `z_ab`, attention fused-`q` ∥ separate `v` —
to hide memory latency without changing kernels.

## Attempt

Added `gemv_pair` with a process-wide non-blocking side stream. Call sites in
`deltanet_decode` / `attention_decode` overlapped the pairs above.

## Correctness

- Sequential `gemv_pair` (both on the caller's stream): `test_model` **PASS**.
- Concurrent side-stream launch (with `hipStreamWaitEvent` or
  `hipStreamSynchronize(side)`): `test_model` **FAIL**, nondeterministic wrong
  continuation IDs (16/20 assertions). `test_attention` numeric L2 still passed.

Production left sequential. No e2e timing (failed the correctness gate).

## Decision

**REJECT.** On this ROCm 6.3 / gfx906 path, overlapping those GEMVs corrupts
decode under the null stream used by `test_model` / `bench_decode`. Do not
retry without a non-null decode stream design and a dedicated race test.
