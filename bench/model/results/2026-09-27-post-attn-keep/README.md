# Post-#31 decode baseline + peer next-hypotheses

Engine: main `4814fe5` after KEEP #29 (MoE-down FP16) and KEEP #31 (attn K/V LDS overlay).
MI50, ROCm 6.3.x, Release gfx906. Protocol: `run_measure.py`, 3 runs each, smi logged.

## Medians

| probe | median |
|-------|-------:|
| short e2e ctx 273..529 | **146.0 tok/s** |
| pp512 | **1439.8 tok/s** |
| ctx 17 | **144.844 tok/s** |
| ctx 4096 | **134.151 tok/s** |
| ctx 32768 | **112.204 tok/s** |

## Peer consultation

Brief: `brief.md`. Responses: `peer-mimo.txt`, `peer-deepseek.txt`, `peer-glm.txt`.
`peer-astra.txt`: Codex quota exhausted (retry after ~3:55 AM); glm substituted for the
Codex slot in the synthesis.

No fresh rocprof in this batch — peers unanimously ask for re-profile before coding.
