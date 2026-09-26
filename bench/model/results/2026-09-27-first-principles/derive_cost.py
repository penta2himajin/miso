#!/usr/bin/env python3
"""Header-only decode cost inventory; never accesses GGUF tensor payloads.

Usage: python3 derive_cost.py MODEL.gguf [output.json]
Byte counts distinguish quant payload from allocated padded row spans. Neither
is a measurement of HBM transactions. FLOPs exclude unpacking/nonlinear math.
"""
import collections
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
from gguf_header import Reader, TYPES


def main(path):
    # Map a bounded prefix. Reader touches only metadata/tensor descriptions;
    # mapping does not read tensor payload pages beyond the parsed header.
    with open(path, "rb") as f, mmap.mmap(f.fileno(), min(16 << 20, Path(path).stat().st_size),
                                         access=mmap.ACCESS_READ) as prefix:
        r = Reader(prefix)
        assert r.take("I") == 0x46554747
        version, count, kv_count = r.take("I"), r.take("Q"), r.take("Q")
        meta = {}
        for _ in range(kv_count):
            name = r.string()
            meta[name] = r.value(r.take("I"))
        tensors = []
        for _ in range(count):
            name = r.string()
            dims = [r.take("Q") for _ in range(r.take("I"))]
            ty, offset = r.take("I"), r.take("Q")
            tensors.append((name, dims, ty))
        header_size = r.p
        header_hash = hashlib.sha256(prefix[:header_size]).hexdigest()

    layers = meta["qwen35moe.block_count"] - meta["qwen35moe.nextn_predict_layers"]
    topk, experts = meta["qwen35moe.expert_used_count"], meta["qwen35moe.expert_count"]
    interval = meta["qwen35moe.full_attention_interval"]
    attn_layers, dn_layers = layers // interval, layers - layers // interval
    groups = collections.defaultdict(collections.Counter)
    details = []
    for name, dims, ty in tensors:
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m[1]) >= layers:
            continue
        typename, be, bb = TYPES[ty]
        elements = math.prod(dims)
        row_elements = dims[0]
        rows = elements // row_elements
        row_payload = row_elements // be * bb
        row_span = (row_payload + 15) // 16 * 16 if typename == "Q6_K" else row_payload
        active_rows = rows
        if "_exps" in name:
            assert rows % experts == 0
            active_rows = rows // experts * topk
            category = "moe_down" if "down" in name else "moe_gate_up"
        elif name == "token_embd.weight":
            active_rows, category = 1, "embedding"
        elif "gate_inp" in name:
            category = "router"
            assert typename == "F32"
            row_payload //= 2  # append_bf16 in engine/moe.hip, exact storage conversion
            row_span = row_payload
        elif "shexp" in name:
            category = "moe_down" if "down" in name else "moe_gate_up"
        elif name == "output.weight":
            category = "lm_head"
        elif typename in ("Q4_K", "Q6_K"):
            category = "dense_projections"
        else:
            category = "norm_conv_and_other_small_weights"
        payload, span = active_rows * row_payload, active_rows * row_span
        linear = category in ("router", "moe_gate_up", "moe_down", "lm_head", "dense_projections")
        macs = active_rows * row_elements if linear else 0
        groups[category].update(payload_bytes=payload, row_span_bytes=span, linear_macs=macs)
        details.append(dict(name=name, dims=dims, type=typename, active_rows=active_rows,
                            payload_bytes=payload, row_span_bytes=span, linear_macs=macs))

    dn_dim, dn_vheads = meta["qwen35moe.ssm.state_size"], meta["qwen35moe.ssm.time_step_rank"]
    dn_entries = dn_layers * dn_vheads * dn_dim * dn_dim
    dn_read, dn_write = dn_entries * 4, dn_entries * 4
    conv_channels = 2 * meta["qwen35moe.ssm.group_count"] * dn_dim + meta["qwen35moe.ssm.inner_size"]
    conv_rw = dn_layers * conv_channels * (meta["qwen35moe.ssm.conv_kernel"] - 1) * 4 * 2
    kdim = meta["qwen35moe.attention.key_length"]
    vdim = meta["qwen35moe.attention.value_length"]
    qheads = meta["qwen35moe.attention.head_count"]
    kv_per_pos = attn_layers * meta["qwen35moe.attention.head_count_kv"] * (kdim + vdim) * 2
    weight_payload = sum(x["payload_bytes"] for x in details)
    weight_span = sum(x["row_span_bytes"] for x in details)
    macs = sum(x["linear_macs"] for x in details)
    contexts = []
    for ctx in (0, 4096, 32768):
        # ctx is history before this decode; new position is read after prep writes it.
        kv_read, kv_write = (ctx + 1) * kv_per_pos, kv_per_pos
        total = weight_payload + dn_read + dn_write + conv_rw + kv_read + kv_write
        attention_flops = 2 * attn_layers * qheads * (kdim + vdim) * (ctx + 1)
        contexts.append(dict(history_positions=ctx, core_positions=ctx + 1,
                             kv_read_bytes=kv_read, kv_write_bytes=kv_write,
                             compulsory_payload_plus_state_bytes=total,
                             streaming_797GBps_reference_ms=total / 797e9 * 1e3,
                             streaming_797GBps_reference_tokens_per_s=797e9 / total,
                             attention_QK_PV_flops=attention_flops,
                             attention_12_71TFps_reference_ms=attention_flops / 12.71e12 * 1e3))
    return dict(model_path=str(Path(path).resolve()), gguf_version=version,
                header_bytes_parsed=header_size, header_sha256=header_hash,
                main_layers=layers, deltanet_layers=dn_layers, attention_layers=attn_layers,
                expert_selection=f"{topk}/{experts} plus one shared", groups=dict(groups),
                weight_payload_bytes=weight_payload, active_row_span_bytes=weight_span,
                padding_bytes=weight_span - weight_payload, linear_macs=macs,
                linear_math_flops=2 * macs,
                linear_fp16_dot_flops=2 * (macs - groups["router"]["linear_macs"]),
                router_fp32_math_flops=2 * groups["router"]["linear_macs"],
                linear_peak_math_reference_ms=(2 * (macs - groups["router"]["linear_macs"]) / 25.45e12
                                              + 2 * groups["router"]["linear_macs"] / 12.71e12) * 1e3,
                deltanet_state_read_bytes=dn_read, deltanet_state_write_bytes=dn_write,
                deltanet_conv_state_read_write_bytes=conv_rw,
                deltanet_state_core_flops=dn_entries * 7,
                contexts=contexts, tensors=details,
                limitations=["Unique payload/spans are not measured HBM traffic.",
                             "Row padding is allocated but not read by slot loads.",
                             "Actual load requests duplicate scales and Q6 high bits; caches coalesce them.",
                             "Nonlinear math, reductions, quant unpacking, ISA issue, LDS, and scratch traffic omitted.",
                             "797 GB/s is a measured pure-read microbenchmark reference, not a hard hardware ceiling.",
                             "FP16 dot peak cannot describe mixed integer, FP32, LDS and conversion instructions."])


if __name__ == "__main__":
    result = main(sys.argv[1])
    output = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(output)
    else:
        print(output, end="")
