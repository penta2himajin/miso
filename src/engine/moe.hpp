#pragma once

// Sparse MoE block of one layer (256 experts, top-8, shared expert), decode path.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "engine/weights.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

struct MoeLayer {
  DeviceBuffer<std::uint16_t> router;  // [257][2048] BF16: ffn_gate_inp rows + ffn_gate_inp_shexp
  QMatrix gate_exps, up_exps, down_exps, gate_sh, up_sh, down_sh;

  static MoeLayer load(const gguf::File& f, int layer);
};

struct MoeScratch {
  DeviceBuffer<float> logits{257}, h{9 * 1024}, weights{8};
  DeviceBuffer<int> ids{8};
};

// Chunk buffers of the prefill path, for up to max_tok tokens (9 assignments per token).
struct MoePrefillScratch {
  explicit MoePrefillScratch(unsigned max_tok);
  unsigned max_tok;
  GemmInput in;
  DeviceBuffer<float> logits, weights, as, part;
  DeviceBuffer<int> ids, order, n_tasks;
  DeviceBuffer<int4> tasks;
  DeviceBuffer<_Float16> ah;
};

// y = MoE(x) for one token (x is the post-attention RMSNorm output).
void moe_decode(const MoeLayer& w, MoeScratch& sc, const float* x, float* y, hipStream_t stream);

// The same for n tokens ([n][2048] arrays): batched routing identical to decode, then grouped
// expert GEMMs over the tokens routed to each expert (the shared expert as expert 256).
void moe_prefill(const MoeLayer& w, MoePrefillScratch& sc, const float* x, float* y, unsigned n,
                 hipStream_t stream);

}  // namespace miso
