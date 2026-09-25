#include "engine/deltanet.hpp"

#include <string>

#include "deltanet_step.hpp"
#include "rmsnorm.hpp"

namespace miso {

DeltaNetLayer DeltaNetLayer::load(const gguf::File& f, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  return {upload_qmatrix(f, p + "attn_qkv.weight"),  upload_qmatrix(f, p + "attn_gate.weight"),
          upload_qmatrix(f, p + "ssm_alpha.weight"), upload_qmatrix(f, p + "ssm_beta.weight"),
          upload_qmatrix(f, p + "ssm_out.weight"),   upload_f32(f, p + "attn_norm.weight"),
          upload_f32(f, p + "ssm_conv1d.weight"),    upload_f32(f, p + "ssm_a"),
          upload_f32(f, p + "ssm_dt.bias"),          upload_f32(f, p + "ssm_norm.weight")};
}

void DeltaNetState::reset() {
  for (auto& c : conv)
    MISO_HIP_CHECK(hipMemset(c.data(), 0, c.size() * sizeof(float)));
  MISO_HIP_CHECK(hipMemset(s.data(), 0, s.size() * sizeof(float)));
  parity = 0;
}

void deltanet_decode(const DeltaNetLayer& w, DeltaNetState& st, DeltaNetScratch& sc, const float* h,
                     float* y, float eps, hipStream_t stream) {
  constexpr int kBlock = 256;
  const kernels::AddRmsNormParams norm{const_cast<float*>(h), nullptr, w.attn_norm.data(),
                                       sc.xn.data(), eps};
  kernels::add_rmsnorm_kernel<2048, kBlock><<<1, kBlock, 0, stream>>>(norm, 1);
  gemv(w.qkv, sc.xn.data(), sc.qkv.data(), stream);
  gemv(w.z, sc.xn.data(), sc.z.data(), stream);
  gemv(w.a, sc.xn.data(), sc.a.data(), stream);
  gemv(w.b, sc.xn.data(), sc.b.data(), stream);
  const kernels::DeltaNetStepParams step{sc.qkv.data(),
                                         sc.z.data(),
                                         sc.a.data(),
                                         sc.b.data(),
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

}  // namespace miso
