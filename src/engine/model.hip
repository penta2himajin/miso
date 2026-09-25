#include "engine/model.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "argmax.hpp"
#include "embed.hpp"
#include "repack.hpp"
#include "rmsnorm.hpp"

namespace miso {

namespace {

constexpr int kBlock = 256;
constexpr unsigned kHidden = 2048;

bool is_full_attention(const gguf::File& f, unsigned layer) {
  const auto interval =
      static_cast<unsigned>(f.get<std::int64_t>("qwen35moe.full_attention_interval"));
  return (layer + 1) % interval == 0;
}

DeviceBuffer<std::uint8_t> upload_embedding(const gguf::File& f) {
  const auto* t = f.find_tensor("token_embd.weight");
  if (t == nullptr || t->type != gguf::GgmlType::Q4_K)
    throw std::runtime_error("token_embd must be Q4_K");
  const auto src = f.tensor_data(*t);
  std::vector<std::byte> r1(src.size());
  repack_q4k(src, r1);
  DeviceBuffer<std::uint8_t> d(r1.size());
  MISO_HIP_CHECK(hipMemcpy(d.data(), r1.data(), r1.size(), hipMemcpyHostToDevice));
  return d;
}

}  // namespace

Model Model::load(const gguf::File& f) {
  const auto n_layers =
      static_cast<unsigned>(f.get<std::int64_t>("qwen35moe.block_count")) -
      static_cast<unsigned>(f.get<std::int64_t>("qwen35moe.nextn_predict_layers"));
  Model m{upload_embedding(f),
          {},
          upload_f32(f, "output_norm.weight"),
          upload_qmatrix(f, "output.weight"),
          RopeConfig::from(f),
          static_cast<float>(f.get<double>("qwen35moe.attention.layer_norm_rms_epsilon")),
          0};
  m.vocab = m.lm_head.n;
  m.layers.reserve(n_layers);
  for (unsigned i = 0; i < n_layers; ++i) {
    Layer l{std::nullopt, std::nullopt,
            upload_f32(f, "blk." + std::to_string(i) + ".post_attention_norm.weight"),
            MoeLayer::load(f, static_cast<int>(i))};
    if (is_full_attention(f, i)) {
      l.attention.emplace(AttentionLayer::load(f, static_cast<int>(i)));
    } else {
      l.deltanet.emplace(DeltaNetLayer::load(f, static_cast<int>(i)));
    }
    m.layers.push_back(std::move(l));
  }
  return m;
}

Session::Session(const Model& m, unsigned n) : max_ctx(n), logits(m.vocab), tokens(n), pred(n) {
  for (const auto& l : m.layers) {
    deltanet.emplace_back();
    kv.emplace_back();
    if (l.deltanet)
      deltanet.back().emplace();
    if (l.attention)
      kv.back().emplace(n);
  }
}

namespace {

// Final norm of h + delta, LM head and argmax; the argmax goes to s.pred[pos].
void head_argmax(const Model& m, Session& s, float* h, const float* delta, unsigned pos,
                 hipStream_t stream) {
  const kernels::AddRmsNormParams final_norm{h, delta, m.output_norm.data(), s.xn.data(), m.eps};
  kernels::add_rmsnorm_kernel<kHidden, kBlock><<<1, kBlock, 0, stream>>>(final_norm, 1);
  gemv(m.lm_head, s.xn.data(), s.logits.data(), stream);
  const kernels::ArgmaxParams ap{s.logits.data(),       m.vocab,
                                 s.argmax_v.data(),     s.argmax_i.data(),
                                 Session::kArgmaxParts, s.pred.data() + pos};
  kernels::argmax_partial_kernel<kBlock><<<Session::kArgmaxParts, kBlock, 0, stream>>>(ap);
  kernels::argmax_final_kernel<kBlock><<<1, kBlock, 0, stream>>>(ap);
  MISO_HIP_CHECK(hipGetLastError());
}

}  // namespace

void decode_step(const Model& m, Session& s, const int* token, hipStream_t stream) {
  if (s.pos >= s.max_ctx)
    throw std::runtime_error("decode_step: context is full");
  const kernels::EmbedParams ep{m.embed.data(), token, s.h.data()};
  kernels::embed_q4k_kernel<kHidden, kBlock><<<1, kBlock, 0, stream>>>(ep, 1);

  for (std::size_t i = 0; i < m.layers.size(); ++i) {
    const auto& l = m.layers[i];
    const float* delta = i == 0 ? nullptr : s.moe_out.data();  // previous layer's MoE output
    if (l.deltanet) {
      deltanet_decode(*l.deltanet, *s.deltanet[i], s.dn_scratch, s.h.data(), delta,
                      s.mixer_out.data(), m.eps, stream);
    } else {
      attention_decode(*l.attention, *s.kv[i], s.attn_scratch, m.rope, s.h.data(), delta,
                       s.mixer_out.data(), m.eps, stream);
    }
    const kernels::AddRmsNormParams post{s.h.data(), s.mixer_out.data(),
                                         l.post_attention_norm.data(), s.xn.data(), m.eps};
    kernels::add_rmsnorm_kernel<kHidden, kBlock><<<1, kBlock, 0, stream>>>(post, 1);
    moe_decode(l.moe, s.moe_scratch, s.xn.data(), s.moe_out.data(), stream);
  }

  head_argmax(m, s, s.h.data(), s.moe_out.data(), s.pos, stream);
  ++s.pos;
}

void prefill(const Model& m, Session& s, const int* tokens, unsigned n, unsigned max_chunk,
             hipStream_t stream) {
  if (max_chunk == 0 || max_chunk > Session::kMaxChunk)
    throw std::runtime_error("prefill: bad chunk size");
  if (s.pos + n > s.max_ctx)
    throw std::runtime_error("prefill: context is full");
  for (unsigned c0 = 0; c0 < n; c0 += max_chunk) {
    const unsigned c = std::min(max_chunk, n - c0);
    const kernels::EmbedParams ep{m.embed.data(), tokens + c0, s.chunk_h.data()};
    kernels::embed_q4k_kernel<kHidden, kBlock><<<c, kBlock, 0, stream>>>(ep, c);
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
      const auto& l = m.layers[i];
      const float* delta = i == 0 ? nullptr : s.chunk_moe.data();
      if (l.deltanet) {
        deltanet_prefill(*l.deltanet, *s.deltanet[i], s.dn_prefill, s.chunk_h.data(), delta,
                         s.chunk_mixer.data(), c, m.eps, stream);
      } else {
        attention_prefill(*l.attention, *s.kv[i], s.attn_scratch, s.attn_prefill, m.rope,
                          s.chunk_h.data(), delta, s.chunk_mixer.data(), c, m.eps, stream);
      }
      const kernels::AddRmsNormParams post{s.chunk_h.data(), s.chunk_mixer.data(),
                                           l.post_attention_norm.data(), s.chunk_xn.data(), m.eps};
      kernels::add_rmsnorm_kernel<kHidden, kBlock><<<c, kBlock, 0, stream>>>(post, c);
      moe_prefill(l.moe, s.moe_prefill, s.chunk_xn.data(), s.chunk_moe.data(), c, stream);
    }
    s.pos += c;
    if (c0 + c == n) {
      const std::size_t last = std::size_t{c - 1} * kHidden;
      head_argmax(m, s, s.chunk_h.data() + last, s.chunk_moe.data() + last, s.pos - 1, stream);
    }
  }
}

void ingest(const Model& m, Session& s, const int* tokens, unsigned n, hipStream_t stream) {
  if (n < kPrefillMinTokens) {
    for (unsigned t = 0; t < n; ++t)
      decode_step(m, s, tokens + t, stream);
    return;
  }
  prefill(m, s, tokens, n, Session::kMaxChunk, stream);
}

}  // namespace miso
