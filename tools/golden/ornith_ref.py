"""Reference Ornith-1.5-35B-A3B (qwen3_5_moe) layers built from a GGUF (ADR_001 D6, ADR_002 D11).

The GGUF was produced by llama.cpp's converter, which rewrites some HF tensors
(`conversion/qwen.py`, Qwen3NextModel / _LinearAttentionVReorderBase). This module inverts those
rewrites so that the Hugging Face `transformers` modules compute with exactly the dequantised GGUF
weights the engine loads:

- `*norm.weight` except `linear_attn.norm`: stored as 1 + w (HF RMSNorm multiplies by 1 + w).
- `A_log`: stored as `ssm_a = -exp(A_log)`.
- `conv1d.weight`: stored squeezed, (C, K) instead of (C, 1, K).
- DeltaNet V heads: stored in tiled order [K0, K1, ..., K0, K1, ...] instead of HF's grouped order
  [K0_v0, K0_v1, K1_v0, ...] (V rows of qkv, z rows, a/b rows, A_log, dt_bias, V channels of conv1d,
  out_proj columns).
- Experts: HF `gate_up_proj` [E, 2I, H] is stored as separate gate / up tensors.
"""
import json

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import hf_hub_download
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

HF_REPO = "ornith-ai/Ornith-1.5-35B-A3B"


def load_config():
    path = hf_hub_download(HF_REPO, "config.json")
    cfg = Qwen3_5MoeTextConfig(**json.load(open(path))["text_config"])
    cfg._attn_implementation = "eager"
    return cfg, path


class Weights:
    """Dequantised FP32 tensors from the GGUF, in numpy (row-major, GGUF dims reversed) layout."""

    def __init__(self, gguf_path):
        self.reader = GGUFReader(gguf_path)
        self.index = {t.name: t for t in self.reader.tensors}

    def get(self, name):
        t = self.index[name]
        shape = tuple(int(d) for d in reversed(t.shape))
        return dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(shape)

    def rows(self, name, ids):
        """Dequantise selected rows of a 2-D tensor (e.g. token embeddings)."""
        t = self.index[name]
        data = np.asarray(t.data)
        return np.stack([dequantize(data[i], t.tensor_type).astype(np.float32) for i in ids])


def tiled_to_grouped(x, dim, num_k_heads, num_v_per_k, head_dim):
    """Inverse of llama.cpp's _reorder_v_heads along `dim`."""
    shape = list(x.shape)
    new = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(x.reshape(new).transpose(perm)).reshape(shape)


def layer_state_dict(w, cfg, i):
    """HF state dict (FP32 torch tensors) for decoder layer `i`."""
    p = f"blk.{i}."
    sd = {
        "input_layernorm.weight": w.get(p + "attn_norm.weight") - 1.0,
        "post_attention_layernorm.weight": w.get(p + "post_attention_norm.weight") - 1.0,
        "mlp.gate.weight": w.get(p + "ffn_gate_inp.weight"),
        "mlp.experts.gate_up_proj": np.concatenate(
            [w.get(p + "ffn_gate_exps.weight"), w.get(p + "ffn_up_exps.weight")], axis=1),
        "mlp.experts.down_proj": w.get(p + "ffn_down_exps.weight"),
        "mlp.shared_expert.gate_proj.weight": w.get(p + "ffn_gate_shexp.weight"),
        "mlp.shared_expert.up_proj.weight": w.get(p + "ffn_up_shexp.weight"),
        "mlp.shared_expert.down_proj.weight": w.get(p + "ffn_down_shexp.weight"),
        "mlp.shared_expert_gate.weight": w.get(p + "ffn_gate_inp_shexp.weight").reshape(1, -1),
    }
    if cfg.layer_types[i] == "full_attention":
        sd.update({
            "self_attn.q_proj.weight": w.get(p + "attn_q.weight"),
            "self_attn.k_proj.weight": w.get(p + "attn_k.weight"),
            "self_attn.v_proj.weight": w.get(p + "attn_v.weight"),
            "self_attn.o_proj.weight": w.get(p + "attn_output.weight"),
            "self_attn.q_norm.weight": w.get(p + "attn_q_norm.weight") - 1.0,
            "self_attn.k_norm.weight": w.get(p + "attn_k_norm.weight") - 1.0,
        })
    else:
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv, r = cfg.linear_key_head_dim, cfg.linear_value_head_dim, nv // nk
        qk = 2 * nk * dk
        qkv = w.get(p + "attn_qkv.weight")
        qkv[qk:] = tiled_to_grouped(qkv[qk:], 0, nk, r, dv)
        conv = w.get(p + "ssm_conv1d.weight")
        conv[qk:] = tiled_to_grouped(conv[qk:], 0, nk, r, dv)
        a_log = np.log(-w.get(p + "ssm_a").astype(np.float64)).astype(np.float32)
        sd.update({
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_z.weight": tiled_to_grouped(w.get(p + "attn_gate.weight"), 0, nk, r, dv),
            "linear_attn.in_proj_a.weight": tiled_to_grouped(w.get(p + "ssm_alpha.weight"), 0, nk, r, 1),
            "linear_attn.in_proj_b.weight": tiled_to_grouped(w.get(p + "ssm_beta.weight"), 0, nk, r, 1),
            "linear_attn.A_log": tiled_to_grouped(a_log[:, None], 0, nk, r, 1)[:, 0],
            "linear_attn.dt_bias": tiled_to_grouped(w.get(p + "ssm_dt.bias")[:, None], 0, nk, r, 1)[:, 0],
            "linear_attn.conv1d.weight": conv[:, None, :],
            "linear_attn.norm.weight": w.get(p + "ssm_norm.weight"),
            "linear_attn.out_proj.weight": tiled_to_grouped(w.get(p + "ssm_out.weight"), 1, nk, r, dv),
        })
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}
