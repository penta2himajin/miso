#!/usr/bin/env python3
"""Produce golden per-layer inputs/outputs for Ornith-1.5-35B-A3B with the transformers reference.

Layers are built one at a time from dequantised GGUF weights (ornith_ref.py), run in FP32 on the
CPU, recorded, and freed, so peak memory stays near one layer (~3.5 GB).

  .venv/bin/python tools/golden/make_golden.py <model.gguf> --gguf-sha256 <sha> \
      --out tests/golden/ornith-layers.gguf
  # Validate the weight mapping end to end: teacher-force llama.cpp's greedy continuation
  .venv/bin/python tools/golden/make_golden.py <model.gguf> --verify "<continuation text>"
"""
import argparse
import os
import sys
import time

import gguf
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

sys.path.insert(0, os.path.dirname(__file__))
import ornith_ref  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. 日本の首都は"


def self_check_reorder():
    x = np.arange(32 * 3).reshape(32, 3)
    grouped_to_tiled = x.reshape(16, 2, 3).transpose(1, 0, 2).reshape(32, 3)
    assert np.array_equal(ornith_ref.tiled_to_grouped(grouped_to_tiled, 0, 16, 2, 1), x)
    assert not np.array_equal(grouped_to_tiled, x)


def squeeze_batch(v):
    return v[0].copy() if v.ndim == 3 and v.shape[0] == 1 else v.copy()


class LazyLayer(torch.nn.Module):
    """Materialises decoder layer `i` from the GGUF for one forward call, recording tensors."""

    def __init__(self, cfg, i, weights, record):
        super().__init__()
        self.cfg, self.i, self.weights, self.record = cfg, i, weights, record

    def forward(self, hidden_states, **kwargs):
        t0 = time.time()
        with torch.device("meta"):
            layer = hf.Qwen3_5MoeDecoderLayer(self.cfg, self.i)
        layer.load_state_dict(ornith_ref.layer_state_dict(self.weights, self.cfg, self.i),
                              strict=True, assign=True)
        rec, hooks = {}, []
        if self.record is not None:
            mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
            hooks.append(mixer.register_forward_hook(
                lambda m, a, o: rec.__setitem__("mixer_out", o[0] if isinstance(o, tuple) else o)))

            def moe_hook(m, a, o):
                rec["moe_in"] = a[0]
                rec["moe_out"] = o[0] if isinstance(o, tuple) else o
            hooks.append(layer.mlp.register_forward_hook(moe_hook))

            def router_hook(m, a, o):
                for j, t in enumerate(o if isinstance(o, tuple) else (o,)):
                    rec[f"router.{j}"] = t
            hooks.append(layer.mlp.gate.register_forward_hook(router_hook))
        with torch.no_grad():
            out = layer(hidden_states, **kwargs)
        for h in hooks:
            h.remove()
        if self.record is not None:
            rec["in"], rec["out"] = hidden_states, out
            self.record[self.i] = {k: squeeze_batch(v.detach().float().numpy()) for k, v in rec.items()}
        del layer
        print(f"  layer {self.i:2d} ({self.cfg.layer_types[self.i]}) {time.time() - t0:5.1f}s",
              flush=True)
        return out


def run(weights, cfg, ids, n_layers, record_layers):
    cfg = cfg.__class__(**cfg.to_dict())
    cfg.num_hidden_layers = n_layers
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.Qwen3_5MoeTextModel(cfg)
    model.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(config=cfg)
    model.norm = hf.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    model.norm.weight.data = torch.from_numpy(weights.get("output_norm.weight") - 1.0)
    record = {}
    model.layers = torch.nn.ModuleList(
        [LazyLayer(cfg, i, weights, record if i in record_layers else None) for i in range(n_layers)])
    emb = torch.from_numpy(weights.rows("token_embd.weight", ids))[None]
    with torch.no_grad():
        out = model(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
    return emb[0].numpy(), out.numpy(), record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--out", default="tests/golden/ornith-layers.gguf")
    ap.add_argument("--gguf-sha256", default="")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--verify", help="llama.cpp greedy continuation of PROMPT; runs all layers")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    self_check_reorder()

    cfg, cfg_path = ornith_ref.load_config()
    tok = AutoTokenizer.from_pretrained(ornith_ref.HF_REPO)
    weights = ornith_ref.Weights(a.gguf)

    if a.verify is not None:
        prompt_ids = tok(PROMPT)["input_ids"]
        ids = tok(PROMPT + a.verify)["input_ids"]
        assert ids[:len(prompt_ids)] == prompt_ids, "continuation changed prompt tokenisation"
        _, hidden, _ = run(weights, cfg, ids, cfg.num_hidden_layers, set())
        logits = hidden @ weights.get("output.weight").T
        pred = logits.argmax(-1)
        n, ok = len(ids) - len(prompt_ids), 0
        for p in range(len(prompt_ids) - 1, len(ids) - 1):
            match = int(pred[p]) == ids[p + 1]
            ok += match
            print(f"pos {p}: ref {tok.decode([int(pred[p])])!r} llama.cpp {tok.decode([ids[p + 1]])!r} "
                  f"{'OK' if match else 'MISMATCH'}")
        print(f"teacher-forced agreement: {ok}/{n}")
        return

    layers = sorted(int(x) for x in a.layers.split(","))
    ids = tok(PROMPT)["input_ids"]
    emb, _, record = run(weights, cfg, ids, max(layers) + 1, set(layers))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    w = gguf.GGUFWriter(a.out, "miso-golden")
    w.add_string("golden.prompt", PROMPT)
    w.add_array("golden.token_ids", ids)
    w.add_array("golden.layers", layers)
    w.add_string("golden.model_gguf_sha256", a.gguf_sha256)
    w.add_string("golden.hf_config", cfg_path.split("snapshots/")[-1])
    w.add_string("golden.transformers_version", transformers.__version__)
    w.add_string("golden.torch_version", torch.__version__)
    w.add_tensor("embeddings", emb.astype(np.float32))
    for i in layers:
        for k, v in record[i].items():
            w.add_tensor(f"layer{i}.{k}", np.ascontiguousarray(v).astype(np.float32))
            print(f"layer{i}.{k}: {v.shape} {v.dtype}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes), {len(ids)} tokens")


if __name__ == "__main__":
    main()
