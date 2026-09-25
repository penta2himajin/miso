#include "engine/moe.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe_decode.hpp"
#include "moe_prefill.hpp"

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
  // gate/up grid: 80=94/91, 120=88/85, 160=89/86, 240=90/87, 320+=110+ us (layer 0/5).
  kernels::moe_gate_up_kernel<kBlock><<<120, kBlock, 0, stream>>>(gp);
  const kernels::MoeDownParams dp{sc.logits.data(),
                                  sc.ids.data(),
                                  sc.weights.data(),
                                  w.down_exps.data.data(),
                                  w.down_sh.data.data(),
                                  sc.h.data(),
                                  y};
  // down grid: 60=94/93, 120=90/87, 180=102/88, 240=102/95 us (before the gate/up retune).
  if (w.down_exps.type == gguf::GgmlType::Q4_K) {
    kernels::moe_down_kernel<kernels::QType::Q4K, kBlock><<<120, kBlock, 0, stream>>>(dp);
  } else {
    kernels::moe_down_kernel<kernels::QType::Q6K, kBlock><<<120, kBlock, 0, stream>>>(dp);
  }
  MISO_HIP_CHECK(hipGetLastError());
}

MoePrefillScratch::MoePrefillScratch(unsigned n)
    : max_tok(n),
      in(n, 2048),
      logits(std::size_t{n} * 257),
      weights(std::size_t{n} * 9),
      as(std::size_t{n} * 9 * 32),
      part(std::size_t{n} * 9 * 2048),
      ids(std::size_t{n} * 9),
      order(std::size_t{n} * 9),
      n_tasks(1),
      tasks(kernels::moe_max_tasks(n)),
      ah(std::size_t{n} * 9 * 512) {}

void moe_prefill(const MoeLayer& w, MoePrefillScratch& sc, const float* x, float* y, unsigned n,
                 hipStream_t stream) {
  constexpr int kBlock = 256;
  if (n > sc.max_tok)
    throw std::runtime_error("moe_prefill: chunk too large");
  gemm_input(x, 2048, n, sc.in, stream);
  const kernels::MoeRouterBatchParams rp{w.router.data(), x, sc.logits.data(), n};
  kernels::moe_router_batch_kernel<kBlock>
      <<<kernels::moe_router_batch_items(n, kBlock), kBlock, 0, stream>>>(rp);
  const kernels::MoeTopkBatchParams tp{sc.logits.data(), sc.ids.data(), sc.weights.data(), n};
  kernels::moe_topk_batch_kernel<kBlock><<<n, kBlock, 0, stream>>>(tp);
  const kernels::MoeGroupParams gp{sc.ids.data(), sc.order.data(), sc.tasks.data(),
                                   sc.n_tasks.data(), n};
  kernels::moe_group_kernel<1024><<<1, 1024, 0, stream>>>(gp);
  const unsigned max_tasks = kernels::moe_max_tasks(n);
  const kernels::MoeGateUpGroupedParams up{
      w.gate_exps.data.data(), w.up_exps.data.data(), w.gate_sh.data.data(), w.up_sh.data.data(),
      sc.in.xh.data(),         sc.in.xs.data(),       sc.order.data(),       sc.tasks.data(),
      sc.n_tasks.data(),       sc.ah.data(),          sc.as.data()};
  kernels::moe_gate_up_grouped_kernel<kBlock><<<2 * max_tasks, kBlock, 0, stream>>>(up);
  const kernels::MoeDownGroupedParams dp{
      w.down_exps.data.data(), w.down_sh.data.data(), sc.ah.data(),    sc.as.data(),
      sc.weights.data(),       sc.order.data(),       sc.tasks.data(), sc.n_tasks.data(),
      sc.part.data()};
  if (w.down_exps.type == gguf::GgmlType::Q4_K)
    kernels::moe_down_grouped_kernel<kernels::QType::Q4K, kBlock>
        <<<8 * max_tasks, kBlock, 0, stream>>>(dp);
  else
    kernels::moe_down_grouped_kernel<kernels::QType::Q6K, kBlock>
        <<<8 * max_tasks, kBlock, 0, stream>>>(dp);
  const kernels::MoeCombineParams cp{sc.part.data(), y, n};
  kernels::moe_combine_kernel<<<(n * 512 + 255) / 256, 256, 0, stream>>>(cp);
  MISO_HIP_CHECK(hipGetLastError());
}

}  // namespace miso
