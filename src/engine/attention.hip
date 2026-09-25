#include "engine/attention.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention_decode.hpp"
#include "rmsnorm.hpp"

namespace miso {

AttentionLayer AttentionLayer::load(const gguf::File& f, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  return {upload_qmatrix(f, p + "attn_q.weight"), upload_qmatrix(f, p + "attn_k.weight"),
          upload_qmatrix(f, p + "attn_v.weight"), upload_qmatrix(f, p + "attn_output.weight"),
          upload_f32(f, p + "attn_norm.weight"),  upload_f32(f, p + "attn_q_norm.weight"),
          upload_f32(f, p + "attn_k_norm.weight")};
}

RopeConfig RopeConfig::from(const gguf::File& f) {
  const float base = static_cast<float>(f.get<double>("qwen35moe.rope.freq_base"));
  const auto dim = static_cast<int>(f.get<std::int64_t>("qwen35moe.rope.dimension_count"));
  if (dim != kernels::attn::kRopeDim)
    throw std::runtime_error("unexpected rope dimension count");
  // transformers: 1 / base ** (arange(0, dim, 2, float32) / dim), in FP32.
  std::vector<float> inv(dim / 2);
  for (int i = 0; i < dim / 2; ++i) {
    inv[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(dim));
  }
  return {DeviceBuffer<float>(inv)};
}

AttentionCache::AttentionCache(unsigned n)
    : max_ctx(n), k(std::size_t{2} * n * 256), v(std::size_t{2} * n * 256) {}

void attention_decode(const AttentionLayer& w, AttentionCache& cache, AttentionScratch& sc,
                      const RopeConfig& rope, float* h, const float* delta, float* y, float eps,
                      hipStream_t stream) {
  using kernels::attn::half_t;
  constexpr int kBlock = 256;
  if (cache.len >= cache.max_ctx)
    throw std::runtime_error("attention_decode: KV cache is full");
  const kernels::AddRmsNormParams norm{h, delta, w.attn_norm.data(), sc.xn.data(), eps};
  kernels::add_rmsnorm_kernel<2048, kBlock><<<1, kBlock, 0, stream>>>(norm, 1);
  gemv(w.q, sc.xn.data(), sc.qg.data(), stream);
  gemv(w.k, sc.xn.data(), sc.k.data(), stream);
  gemv(w.v, sc.xn.data(), sc.v.data(), stream);
  auto* kc = reinterpret_cast<half_t*>(cache.k.data());
  auto* vc = reinterpret_cast<half_t*>(cache.v.data());
  const kernels::AttnPrepParams prep{
      sc.qg.data(),         sc.k.data(), sc.v.data(), w.q_norm.data(), w.k_norm.data(),
      rope.inv_freq.data(), kc,          vc,          sc.q.data(),     cache.len,
      cache.max_ctx,        eps};
  kernels::attn_prep_kernel<kBlock><<<18, kBlock, 0, stream>>>(prep);
  ++cache.len;
  const unsigned n_splits = kernels::attn_n_splits(cache.len, AttentionScratch::kMaxSplits);
  const kernels::AttnSplitParams split{
      sc.q.data(),   kc,      vc, sc.part_m.data(), sc.part_l.data(), sc.part_o.data(), cache.len,
      cache.max_ctx, n_splits};
  kernels::attn_split_kernel<kBlock><<<dim3(n_splits, 2), kBlock, 0, stream>>>(split);
  const kernels::AttnCombineParams comb{sc.part_m.data(), sc.part_l.data(), sc.part_o.data(),
                                        sc.qg.data(),     sc.core.data(),   n_splits};
  kernels::attn_combine_kernel<kBlock><<<16, kBlock, 0, stream>>>(comb);
  MISO_HIP_CHECK(hipGetLastError());
  gemv(w.o, sc.core.data(), y, stream);
}

}  // namespace miso
