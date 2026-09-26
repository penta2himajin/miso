#include "engine/deltanet.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "deltanet_step.hpp"
#include "rmsnorm.hpp"

namespace miso {
namespace {

QMatrix empty_qmatrix() {
  return {DeviceBuffer<std::uint8_t>(1), gguf::GgmlType::F32, 0, 0};
}

}  // namespace

DeltaNetLayer DeltaNetLayer::load(const gguf::File& f, int layer, LoadStage* st) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto* qkv_t = f.find_tensor(p + "attn_qkv.weight");
  if (qkv_t == nullptr)
    throw std::runtime_error("missing " + p + "attn_qkv.weight");
  // Same type and K as gate/alpha/beta → one GEMV writes q, k, v, z, a, b.
  if (qkv_t->type == gguf::GgmlType::Q4_K) {
    return {concat_rows(f,
                        {p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_alpha.weight",
                         p + "ssm_beta.weight"},
                        st),
            empty_qmatrix(),
            upload_qmatrix(f, p + "ssm_out.weight", st),
            upload_f32(f, p + "attn_norm.weight", st),
            upload_f32(f, p + "ssm_conv1d.weight", st),
            upload_f32(f, p + "ssm_a", st),
            upload_f32(f, p + "ssm_dt.bias", st),
            upload_f32(f, p + "ssm_norm.weight", st)};
  }
  return {
      upload_qmatrix(f, p + "attn_qkv.weight", st),
      concat_rows(f, {p + "attn_gate.weight", p + "ssm_alpha.weight", p + "ssm_beta.weight"}, st),
      upload_qmatrix(f, p + "ssm_out.weight", st),
      upload_f32(f, p + "attn_norm.weight", st),
      upload_f32(f, p + "ssm_conv1d.weight", st),
      upload_f32(f, p + "ssm_a", st),
      upload_f32(f, p + "ssm_dt.bias", st),
      upload_f32(f, p + "ssm_norm.weight", st)};
}

void DeltaNetState::reset() {
  for (auto& c : conv)
    MISO_HIP_CHECK(hipMemset(c.data(), 0, c.size() * sizeof(float)));
  MISO_HIP_CHECK(hipMemset(s.data(), 0, s.size() * sizeof(float)));
  parity = 0;
}

void deltanet_decode(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetScratch& sc, float* h,
                     const float* delta, float* y, float eps, hipStream_t stream) {
  constexpr int kBlock = 256;
  const kernels::AddRmsNormParams norm{h, delta, w.attn_norm.data(), sc.xn.data(), eps};
  kernels::add_rmsnorm_kernel<2048, kBlock><<<1, kBlock, 0, stream>>>(norm, 1);
  gemv(w.qkv, sc.xn.data(), sc.qkv(), stream);
  if (w.z_ab.n != 0)
    gemv(w.z_ab, sc.xn.data(), sc.z(), stream);
  const kernels::DeltaNetStepParams step{sc.qkv(),
                                         sc.z(),
                                         sc.a(),
                                         sc.b(),
                                         w.conv_w.data(),
                                         st.conv[st.parity].data(),
                                         st.conv[1 - st.parity].data(),
                                         w.ssm_a.data(),
                                         w.dt_bias.data(),
                                         w.ssm_norm.data(),
                                         st.s.data(),
                                         sc.o.data(),
                                         eps};
  kernels::deltanet_step_kernel<kBlock><<<32, kBlock, 0, stream>>>(step);
  MISO_HIP_CHECK(hipGetLastError());
  st.parity = 1 - st.parity;
  gemv(w.out, sc.o.data(), y, stream);
}

void deltanet_prefill(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetPrefillScratch& sc,
                      float* h, const float* delta, float* y, unsigned n, float eps,
                      hipStream_t stream) {
  constexpr int kBlock = 256;
  constexpr unsigned kProj = DeltaNetScratch::kProj;
  const kernels::AddRmsNormParams norm{h, delta, w.attn_norm.data(), sc.xn.data(), eps};
  kernels::add_rmsnorm_kernel<2048, kBlock><<<n, kBlock, 0, stream>>>(norm, n);
  gemm_input(sc.xn.data(), 2048, n, sc.in, stream);
  gemm(w.qkv, sc.in, sc.qkv(), kProj, stream);
  if (w.z_ab.n != 0)
    gemm(w.z_ab, sc.in, sc.z(0), kProj, stream);
  const kernels::DeltaNetSeqParams seq{sc.qkv(),
                                       sc.z(0),
                                       sc.z(0) + 4096,
                                       sc.z(0) + 4128,
                                       w.conv_w.data(),
                                       st.conv[st.parity].data(),
                                       st.conv[1 - st.parity].data(),
                                       w.ssm_a.data(),
                                       w.dt_bias.data(),
                                       w.ssm_norm.data(),
                                       st.s.data(),
                                       sc.o.data(),
                                       n,
                                       eps,
                                       kProj,
                                       kProj};
  kernels::deltanet_seq_kernel<kBlock><<<32, kBlock, 0, stream>>>(seq);
  st.parity = 1 - st.parity;
  MISO_HIP_CHECK(hipGetLastError());
  gemm_input(sc.o.data(), 4096, n, sc.in, stream);
  gemm(w.out, sc.in, y, 2048, stream);
}

}  // namespace miso
