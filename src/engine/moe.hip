#include "engine/moe.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe_decode.hpp"

namespace miso {

namespace {

// F32 -> BF16, refusing any value that is not exactly representable
// (docs/research/ornith-q4km-gguf.md shows the router tensors are BF16-exact).
void append_bf16(const gguf::File& f, const std::string& name, std::vector<std::uint16_t>& out) {
  const auto* t = f.find_tensor(name);
  if (t == nullptr || t->type != gguf::GgmlType::F32)
    throw std::runtime_error(name + ": missing or not F32");
  const auto src = f.tensor_data(*t);
  for (std::size_t i = 0; i < src.size() / 4; ++i) {
    std::uint32_t bits;
    std::memcpy(&bits, src.data() + 4 * i, 4);
    if ((bits & 0xFFFFu) != 0)
      throw std::runtime_error(name + " is not exactly representable in BF16");
    out.push_back(static_cast<std::uint16_t>(bits >> 16));
  }
}

}  // namespace

MoeLayer MoeLayer::load(const gguf::File& f, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  std::vector<std::uint16_t> router;
  router.reserve(257 * 2048);
  append_bf16(f, p + "ffn_gate_inp.weight", router);
  append_bf16(f, p + "ffn_gate_inp_shexp.weight", router);
  MoeLayer m{DeviceBuffer<std::uint16_t>(router),
             upload_qmatrix(f, p + "ffn_gate_exps.weight"),
             upload_qmatrix(f, p + "ffn_up_exps.weight"),
             upload_qmatrix(f, p + "ffn_down_exps.weight"),
             upload_qmatrix(f, p + "ffn_gate_shexp.weight"),
             upload_qmatrix(f, p + "ffn_up_shexp.weight"),
             upload_qmatrix(f, p + "ffn_down_shexp.weight")};
  for (const QMatrix* q : {&m.gate_exps, &m.up_exps, &m.gate_sh, &m.up_sh}) {
    if (q->type != gguf::GgmlType::Q4_K || q->k != 2048)
      throw std::runtime_error("MoE gate/up must be Q4_K, K = 2048");
  }
  if (m.down_exps.type != m.down_sh.type || m.down_exps.k != 512 || m.down_sh.k != 512) {
    throw std::runtime_error("MoE down projections must share one type with K = 512");
  }
  return m;
}

void moe_decode(const MoeLayer& w, MoeScratch& sc, const float* x, float* y, hipStream_t stream) {
  constexpr int kBlock = 256;
  const kernels::MoeRouterParams rp{w.router.data(), x, sc.logits.data()};
  kernels::moe_router_kernel<kBlock><<<(257 + 3) / 4, kBlock, 0, stream>>>(rp);
  const kernels::MoeGateUpParams gp{
      sc.logits.data(),      w.gate_exps.data.data(), w.up_exps.data.data(),
      w.gate_sh.data.data(), w.up_sh.data.data(),     x,
      sc.h.data(),           sc.ids.data(),           sc.weights.data()};
  kernels::moe_gate_up_kernel<kBlock><<<240, kBlock, 0, stream>>>(gp);
  const kernels::MoeDownParams dp{sc.logits.data(),
                                  sc.ids.data(),
                                  sc.weights.data(),
                                  w.down_exps.data.data(),
                                  w.down_sh.data.data(),
                                  sc.h.data(),
                                  y};
  if (w.down_exps.type == gguf::GgmlType::Q4_K) {
    kernels::moe_down_kernel<kernels::QType::Q4K, kBlock><<<240, kBlock, 0, stream>>>(dp);
  } else {
    kernels::moe_down_kernel<kernels::QType::Q6K, kBlock><<<240, kBlock, 0, stream>>>(dp);
  }
  MISO_HIP_CHECK(hipGetLastError());
}

}  // namespace miso
