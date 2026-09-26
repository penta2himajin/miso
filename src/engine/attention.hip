#include "engine/attention.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention_decode.hpp"
#include "attention_prefill.hpp"
#include "rmsnorm.hpp"

namespace miso {
namespace {

QMatrix empty_qmatrix() {
  return {DeviceBuffer<std::uint8_t>(1), gguf::GgmlType::F32, 0, 0};
}

const gguf::TensorInfo& require_tensor(const gguf::File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  if (t == nullptr)
    throw std::runtime_error("missing tensor " + name);
  return *t;
}

}  // namespace

AttentionLayer AttentionLayer::load(const gguf::File& f, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto& qt = require_tensor(f, p + "attn_q.weight");
  const auto& kt = require_tensor(f, p + "attn_k.weight");
  const auto& vt = require_tensor(f, p + "attn_v.weight");
  // attn_q and attn_k are both Q4_K in every layer of this model, so their 8704 rows share one
  // GEMV into the [q+gate | k | v] projection buffer; only v may need a launch of its own.
  if (qt.type == kt.type && kt.type == vt.type) {
    return {concat_rows(f, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"}),
            empty_qmatrix(),
            empty_qmatrix(),
            upload_qmatrix(f, p + "attn_output.weight"),
            upload_f32(f, p + "attn_norm.weight"),
            upload_f32(f, p + "attn_q_norm.weight"),
            upload_f32(f, p + "attn_k_norm.weight")};
  }
  if (qt.type == kt.type) {
    return {concat_rows(f, {p + "attn_q.weight", p + "attn_k.weight"}),
            empty_qmatrix(),
            upload_qmatrix(f, p + "attn_v.weight"),
            upload_qmatrix(f, p + "attn_output.weight"),
            upload_f32(f, p + "attn_norm.weight"),
            upload_f32(f, p + "attn_q_norm.weight"),
            upload_f32(f, p + "attn_k_norm.weight")};
  }
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
  gemv(w.q, sc.xn.data(), sc.qg(), stream);
  if (w.k.n != 0)
    gemv(w.k, sc.xn.data(), sc.k(), stream);
  if (w.v.n != 0)
    gemv(w.v, sc.xn.data(), sc.v(), stream);
  auto* kc = reinterpret_cast<half_t*>(cache.k.data());
  auto* vc = reinterpret_cast<half_t*>(cache.v.data());
  const kernels::AttnPrepParams prep{
      sc.qg(), sc.k(), sc.v(),      w.q_norm.data(), w.k_norm.data(), rope.inv_freq.data(),
      kc,      vc,     sc.q.data(), cache.len,       cache.max_ctx,   eps};
  kernels::attn_prep_kernel<kBlock><<<18, kBlock, 0, stream>>>(prep);
  ++cache.len;
  const unsigned n_splits = kernels::attn_n_splits(cache.len, AttentionScratch::kMaxSplits);
  const kernels::AttnSplitParams split{
      sc.q.data(),   kc,      vc, sc.part_m.data(), sc.part_l.data(), sc.part_o.data(), cache.len,
      cache.max_ctx, n_splits};
  kernels::attn_split_kernel<kBlock><<<dim3(n_splits, 2), kBlock, 0, stream>>>(split);
  const kernels::AttnCombineParams comb{sc.part_m.data(), sc.part_l.data(), sc.part_o.data(),
                                        sc.qg(),          sc.core.data(),   n_splits};
  kernels::attn_combine_kernel<kBlock><<<16, kBlock, 0, stream>>>(comb);
  MISO_HIP_CHECK(hipGetLastError());
  gemv(w.o, sc.core.data(), y, stream);
}

void attention_prefill(const AttentionLayer& w, AttentionCache& cache, AttentionScratch& sc,
                       AttentionPrefillScratch& ps, const RopeConfig& rope, float* h,
                       const float* delta, float* y, unsigned n, float eps, hipStream_t stream) {
  using kernels::attn::half_t;
  constexpr int kBlock = 256;
  constexpr unsigned kProj = AttentionScratch::kProj;
  if (cache.len + n > cache.max_ctx)
    throw std::runtime_error("attention_prefill: KV cache is full");
  const kernels::AddRmsNormParams norm{h, delta, w.attn_norm.data(), ps.xn.data(), eps};
  kernels::add_rmsnorm_kernel<2048, kBlock><<<n, kBlock, 0, stream>>>(norm, n);
  gemm_input(ps.xn.data(), 2048, n, ps.in, stream);
  gemm(w.q, ps.in, ps.qg(), kProj, stream);
  if (w.k.n != 0)
    gemm(w.k, ps.in, ps.k(), kProj, stream);
  if (w.v.n != 0)
    gemm(w.v, ps.in, ps.v(), kProj, stream);
  auto* kc = reinterpret_cast<half_t*>(cache.k.data());
  auto* vc = reinterpret_cast<half_t*>(cache.v.data());
  (void)sc;
  const unsigned pos0 = cache.len;
  const kernels::AttnPrepBatchParams prep{
      {ps.qg(), ps.k(), ps.v(), w.q_norm.data(), w.k_norm.data(), rope.inv_freq.data(), kc, vc,
       ps.q.data(), pos0, cache.max_ctx, eps},
      n,
      kProj};
  kernels::attn_prep_batch_kernel<kBlock><<<n * 18, kBlock, 0, stream>>>(prep);
  cache.len += n;
  const kernels::AttnPrefillParams ap{ps.q.data(), ps.qg(),       kc,   vc, ps.core.data(), pos0,
                                      n,           cache.max_ctx, kProj};
  kernels::attn_prefill_kernel<kBlock><<<kernels::attn_prefill_items(n), kBlock, 0, stream>>>(ap);
  MISO_HIP_CHECK(hipGetLastError());
  gemm_input(ps.core.data(), 4096, n, ps.in, stream);
  gemm(w.o, ps.in, y, 2048, stream);
}

}  // namespace miso
