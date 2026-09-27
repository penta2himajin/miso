#include <doctest/doctest.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "../bench/deltanet/activation_prepare.hpp"
#include "engine/deltanet.hpp"
#include "gguf.hpp"
#include "hip_util.hpp"
#include "q4k_gemv.hpp"
#include "rmsnorm.hpp"

namespace prepare = miso::bench::activation_prepare;
namespace k = miso::kernels;

namespace {

// Frozen af54ec7 step: independent FP32 reference retains every operation and barrier.
__global__ void __launch_bounds__(256) frozen_step(k::DeltaNetStepParams p) {
  using namespace k;
  using namespace deltanet_detail;
  constexpr int kBlock = 256;
  __shared__ float q[kDim], key[kDim], v[kDim], red[kBlock / kWave], part[2][kDim];
  const unsigned t = threadIdx.x, vi = t % kDim, half = t / kDim, wave = t / kWave;
  for (unsigned h = blockIdx.x; h < kVHeads; h += gridDim.x) {
    const unsigned kh = h % kKHeads;
    const bool own_qk = h < kKHeads;
    if (t < kDim) {
      q[t] = conv_step(p, kh * kDim + t, own_qk);
      v[t] = conv_step(p, 2 * kKHeads * kDim + h * kDim + t, true);
    } else {
      key[vi] = conv_step(p, kKHeads * kDim + kh * kDim + vi, own_qk);
    }
    __syncthreads();
    const float x = t < kDim ? q[t] : key[vi];
    const float ss = wave_sum(x * x);
    if (t % kWave == kWave - 1)
      red[wave] = ss;
    __syncthreads();
    if (t < kDim)
      q[t] = x * rsqrtf(red[0] + red[1] + 1e-6f) * rsqrtf(static_cast<float>(kDim));
    else
      key[vi] = x * rsqrtf(red[2] + red[3] + 1e-6f);
    __syncthreads();
    const float beta = 1.0f / (1.0f + __expf(-p.b[h]));
    const float decay = __expf(p.ssm_a[h] * softplus(p.a[h] + p.dt_bias[h]));
    float* S = p.state + std::size_t{h} * kDim * kDim;
    float s[kDim / 2], kv = 0;
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j) {
      const unsigned kk = half * (kDim / 2) + j;
      s[j] = S[kk * kDim + vi] * decay;
      kv += s[j] * key[kk];
    }
    part[half][vi] = kv;
    __syncthreads();
    const float delta = (v[vi] - (part[0][vi] + part[1][vi])) * beta;
    __syncthreads();
    float o = 0;
#pragma unroll
    for (int j = 0; j < kDim / 2; ++j) {
      const unsigned kk = half * (kDim / 2) + j;
      s[j] += key[kk] * delta;
      S[kk * kDim + vi] = s[j];
      o += s[j] * q[kk];
    }
    part[half][vi] = o;
    __syncthreads();
    const float oh = t < kDim ? part[0][t] + part[1][t] : 0.0f;
    const float os = wave_sum(oh * oh);
    if (t % kWave == kWave - 1)
      red[wave] = os;
    __syncthreads();
    if (t < kDim) {
      const float r = rsqrtf((red[0] + red[1]) / kDim + p.eps);
      p.out[h * kDim + t] = oh * r * p.norm_w[t] * silu(p.z[h * kDim + t]);
    }
    __syncthreads();
  }
}

__global__ void round_half(const float* x, _Float16* y, unsigned n) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = static_cast<_Float16>(x[i]);
}

using Act = k::q4k_detail::SlotAct<k::ActFormat::Fp16>;
static_assert(sizeof(Act) == 72);
__global__ void compare_slots(const float* x, const _Float16* half, Act* got, Act* want) {
  if (threadIdx.x < 128) {
    got[threadIdx.x] = prepare::load_half_slot(half, threadIdx.x);
    want[threadIdx.x] = k::q4k_detail::load_slot<k::ActFormat::Fp16>(x, threadIdx.x);
  }
}

template <class T>
void exact(const std::vector<T>& actual, const std::vector<T>& expected) {
  REQUIRE(actual.size() == expected.size());
  CHECK(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0);
}

std::vector<float> f32(const miso::gguf::File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  REQUIRE_MESSAGE(t != nullptr, name);
  const auto bytes = f.tensor_data(*t);
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}

std::vector<float> adversarial(unsigned seed) {
  const float special[] = {0.0f,
                           -0.0f,
                           0x1p-25f,
                           -0x1p-25f,
                           0x1p-24f,
                           -0x1p-24f,
                           1.00048828125f,
                           std::nextafter(1.00048828125f, 0.0f),
                           std::nextafter(1.00048828125f, 2.0f),
                           -1.00048828125f,
                           2048.0f,
                           -2048.0f};
  std::vector<float> x(4096);
  for (unsigned i = 0; i < x.size(); ++i)
    x[i] = i % 3 == 0 ? special[(i / 3 + seed) % std::size(special)]
                      : static_cast<float>(static_cast<int>((i * 19 + seed) % 127) - 63) * 0.007f;
  return x;
}

void check_half_gemv(const std::vector<std::uint8_t>& weights, const std::vector<float>& x) {
  constexpr unsigned kK = 4096, kN = 5;
  constexpr float kCanary = -9123.5f;
  REQUIRE(weights.size() == kN * (kK / 256 * 144));
  REQUIRE(x.size() == kK);
  miso::DeviceBuffer<std::uint8_t> w(weights);
  miso::DeviceBuffer<float> xf(x);
  miso::DeviceBuffer<_Float16> half(kK);
  miso::DeviceBuffer<float> got(std::vector<float>(kN + 2, kCanary));
  miso::DeviceBuffer<float> want(std::vector<float>(kN + 2, kCanary));
  round_half<<<16, 256>>>(xf.data(), half.data(), kK);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipDeviceSynchronize());
  const prepare::HalfGemvParams hp{w.data(), half.data(), got.data() + 1, kN};
  const k::Q4kGemvParams fp{w.data(), xf.data(), want.data() + 1, kN};
  prepare::q4_half_gemv_kernel<kK, 256, 4><<<240, 256>>>(hp);
  k::q4k_gemv_kernel<k::ActFormat::Fp16, kK, 256, 4><<<240, 256>>>(fp);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipDeviceSynchronize());
  const auto actual = got.to_host(), expected = want.to_host();
  for (unsigned i = 0; i < actual.size(); ++i)
    CHECK(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]));
}

}  // namespace

TEST_CASE("half slot loader matches FP32→FP16 slot packing for adversarial activations") {
  for (unsigned seed : {0u, 1u, 7u}) {
    CAPTURE(seed);
    const auto x = adversarial(seed);
    miso::DeviceBuffer<float> xf(x);
    miso::DeviceBuffer<_Float16> half(4096);
    miso::DeviceBuffer<Act> got(128), want(128);
    round_half<<<16, 256>>>(xf.data(), half.data(), 4096);
    compare_slots<<<1, 256>>>(xf.data(), half.data(), got.data(), want.data());
    MISO_HIP_CHECK(hipGetLastError());
    MISO_HIP_CHECK(hipDeviceSynchronize());
    exact(got.to_host(), want.to_host());
  }
}

TEST_CASE("bench half activation GEMV matches FP32 reader for zero and adversarial fixtures") {
  constexpr unsigned kK = 4096, kN = 5, kRowBytes = kK / 256 * 144;
  check_half_gemv(std::vector<std::uint8_t>(kN * kRowBytes, 0), std::vector<float>(kK, 0.0f));
  for (unsigned seed : {0u, 3u}) {
    CAPTURE(seed);
    std::vector<std::uint8_t> w(kN * kRowBytes);
    for (unsigned i = 0; i < w.size(); ++i)
      w[i] = static_cast<std::uint8_t>((i * 37 + seed * 11) & 0xff);
    check_half_gemv(w, adversarial(seed));
  }
}

TEST_CASE("dual/fp16 prepare producers match frozen FP32 step and rounded halves" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  constexpr unsigned kHidden = 2048, kCore = 4096;
  const auto model = miso::gguf::File::open(MISO_ORNITH_GGUF);
  const auto golden = miso::gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto layer = miso::DeltaNetLayer::load(model, 0);
  const float eps =
      static_cast<float>(model.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));
  const auto in = f32(golden, "layer0.in");
  REQUIRE(in.size() >= kHidden);

  miso::DeltaNetState state_ref, state_dual, state_half;
  std::vector<float> h0(in.begin(), in.begin() + kHidden);
  miso::DeviceBuffer<float> h(h0);
  miso::DeviceBuffer<float> proj(12352);
  miso::DeviceBuffer<float> out_ref(kCore), out_dual(kCore), out_half_fp32(kCore);
  miso::DeviceBuffer<_Float16> half_dual(kCore), half_only(kCore), half_ref(kCore);

  // One QKV+z_ab projection from the first golden token (same path as production decode).
  {
    miso::DeviceBuffer<float> normed(kHidden);
    const k::AddRmsNormParams norm{h.data(), nullptr, layer.attn_norm.data(), normed.data(), eps};
    k::add_rmsnorm_kernel<kHidden, k::kRmsNormBlock><<<1, k::kRmsNormBlock>>>(norm, 1);
    if (layer.z_ab.n)
      miso::gemv_pair(layer.qkv, layer.z_ab, normed.data(), proj.data(), proj.data() + 8192,
                      nullptr);
    else
      miso::gemv(layer.qkv, normed.data(), proj.data(), nullptr);
    MISO_HIP_CHECK(hipDeviceSynchronize());
  }

  auto make_base = [&](miso::DeltaNetState& state, float* out) {
    return k::DeltaNetStepParams{proj.data(),
                                 proj.data() + 8192,
                                 proj.data() + 12288,
                                 proj.data() + 12320,
                                 layer.conv_w.data(),
                                 state.conv[state.parity].data(),
                                 state.conv[1 - state.parity].data(),
                                 layer.ssm_a.data(),
                                 layer.dt_bias.data(),
                                 layer.ssm_norm.data(),
                                 state.s.data(),
                                 out,
                                 eps};
  };

  frozen_step<<<32, 256>>>(make_base(state_ref, out_ref.data()));
  {
    const prepare::StepParams p{make_base(state_dual, out_dual.data()), half_dual.data()};
    prepare::step_kernel<prepare::Output::Dual><<<32, 256>>>(p);
  }
  {
    const prepare::StepParams p{make_base(state_half, out_half_fp32.data()), half_only.data()};
    prepare::step_kernel<prepare::Output::Fp16><<<32, 256>>>(p);
  }
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipDeviceSynchronize());

  round_half<<<16, 256>>>(out_ref.data(), half_ref.data(), kCore);
  MISO_HIP_CHECK(hipDeviceSynchronize());

  exact(out_dual.to_host(), out_ref.to_host());
  exact(half_dual.to_host(), half_ref.to_host());
  exact(half_only.to_host(), half_ref.to_host());
  exact(state_dual.s.to_host(), state_ref.s.to_host());
  exact(state_half.s.to_host(), state_ref.s.to_host());
  for (int i = 0; i < 2; ++i) {
    exact(state_dual.conv[i].to_host(), state_ref.conv[i].to_host());
    exact(state_half.conv[i].to_host(), state_ref.conv[i].to_host());
  }
}

// The reported live-register rounding mismatch first appeared after token 7. Exercise a
// continued trajectory and a multi-head grid-stride launch, with explicit stream ordering.
TEST_CASE("prepared trajectory preserves all state, rounded activations and output GEMV bits" *
          doctest::skip(!std::filesystem::exists(MISO_ORNITH_GGUF))) {
  constexpr unsigned kHidden = 2048, kCore = 4096, kProj = 12352, kGuard = 4;
  constexpr float kCanary = -9123.5f;
  const auto model = miso::gguf::File::open(MISO_ORNITH_GGUF);
  const auto golden = miso::gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto in = f32(golden, "layer0.in");
  const unsigned tokens = static_cast<unsigned>(in.size() / kHidden);
  REQUIRE(tokens == 17);
  const float eps =
      static_cast<float>(model.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"));
  hipStream_t stream;
  MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  for (int layer_index : {0, 4}) {
    CAPTURE(layer_index);
    const auto layer = miso::DeltaNetLayer::load(model, layer_index);
    REQUIRE(layer.out.k == kCore);
    REQUIRE(layer.out.n == kHidden);
    REQUIRE(layer.out.type == miso::gguf::GgmlType::Q4_K);
    miso::DeviceBuffer<float> h(in), normed(tokens * kHidden), proj(tokens * kProj);
    for (unsigned token = 0; token < tokens; ++token) {
      const k::AddRmsNormParams norm{h.data() + token * kHidden, nullptr, layer.attn_norm.data(),
                                     normed.data() + token * kHidden, eps};
      k::add_rmsnorm_kernel<kHidden, k::kRmsNormBlock><<<1, k::kRmsNormBlock, 0, stream>>>(norm, 1);
      if (layer.z_ab.n)
        miso::gemv_pair(layer.qkv, layer.z_ab, normed.data() + token * kHidden,
                        proj.data() + token * kProj, proj.data() + token * kProj + 8192, stream);
      else
        miso::gemv(layer.qkv, normed.data() + token * kHidden, proj.data() + token * kProj, stream);
    }
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
    for (unsigned grid : {32u, 4u}) {
      CAPTURE(grid);
      miso::DeltaNetState ref, dual, half, reg;
      miso::DeviceBuffer<float> out_ref(std::vector<float>(kCore + 2 * kGuard, kCanary));
      miso::DeviceBuffer<float> out_dual(std::vector<float>(kCore + 2 * kGuard, kCanary));
      const std::vector<_Float16> half_canaries(kCore + 2 * kGuard, static_cast<_Float16>(77.0f));
      miso::DeviceBuffer<_Float16> half_ref(half_canaries), half_dual(half_canaries),
          half_only(half_canaries), half_reg(half_canaries);
      miso::DeviceBuffer<float> y_ref(kHidden), y_dual(kHidden), y_half(kHidden), y_reg(kHidden);
      MISO_HIP_CHECK(hipDeviceSynchronize());
      for (unsigned step = 0; step < 34; ++step) {
        CAPTURE(step);
        const float* x = proj.data() + (step % tokens) * kProj;
        auto base = [&](miso::DeltaNetState& state, float* output) {
          return k::DeltaNetStepParams{x,
                                       x + 8192,
                                       x + 12288,
                                       x + 12320,
                                       layer.conv_w.data(),
                                       state.conv[state.parity].data(),
                                       state.conv[1 - state.parity].data(),
                                       layer.ssm_a.data(),
                                       layer.dt_bias.data(),
                                       layer.ssm_norm.data(),
                                       state.s.data(),
                                       output,
                                       eps};
        };
        frozen_step<<<grid, 256, 0, stream>>>(base(ref, out_ref.data() + kGuard));
        prepare::step_kernel<prepare::Output::Dual><<<grid, 256, 0, stream>>>(
            {base(dual, out_dual.data() + kGuard), half_dual.data() + kGuard});
        prepare::step_kernel<prepare::Output::Fp16>
            <<<grid, 256, 0, stream>>>({base(half, nullptr), half_only.data() + kGuard});
        prepare::step_kernel<prepare::Output::Fp16Register>
            <<<grid, 256, 0, stream>>>({base(reg, nullptr), half_reg.data() + kGuard});
        round_half<<<16, 256, 0, stream>>>(out_ref.data() + kGuard, half_ref.data() + kGuard,
                                           kCore);
        const k::Q4kGemvParams fp{layer.out.data.data(), out_ref.data() + kGuard, y_ref.data(),
                                  kHidden};
        k::q4k_gemv_kernel<k::ActFormat::Fp16, kCore, 256, 4><<<240, 256, 0, stream>>>(fp);
        prepare::q4_half_gemv_kernel<kCore, 256, 4><<<240, 256, 0, stream>>>(
            {layer.out.data.data(), half_dual.data() + kGuard, y_dual.data(), kHidden});
        prepare::q4_half_gemv_kernel<kCore, 256, 4><<<240, 256, 0, stream>>>(
            {layer.out.data.data(), half_only.data() + kGuard, y_half.data(), kHidden});
        prepare::q4_half_gemv_kernel<kCore, 256, 4><<<240, 256, 0, stream>>>(
            {layer.out.data.data(), half_reg.data() + kGuard, y_reg.data(), kHidden});
        MISO_HIP_CHECK(hipGetLastError());
        MISO_HIP_CHECK(hipStreamSynchronize(stream));
        exact(out_dual.to_host(), out_ref.to_host());
        exact(half_dual.to_host(), half_ref.to_host());
        exact(half_only.to_host(), half_ref.to_host());
        exact(half_reg.to_host(), half_ref.to_host());
        exact(y_dual.to_host(), y_ref.to_host());
        exact(y_half.to_host(), y_ref.to_host());
        exact(y_reg.to_host(), y_ref.to_host());
        exact(dual.s.to_host(), ref.s.to_host());
        exact(half.s.to_host(), ref.s.to_host());
        exact(reg.s.to_host(), ref.s.to_host());
        for (int i = 0; i < 2; ++i) {
          exact(dual.conv[i].to_host(), ref.conv[i].to_host());
          exact(half.conv[i].to_host(), ref.conv[i].to_host());
          exact(reg.conv[i].to_host(), ref.conv[i].to_host());
        }
        ref.parity = dual.parity = half.parity = reg.parity = 1 - ref.parity;
      }
    }
  }
  MISO_HIP_CHECK(hipStreamDestroy(stream));
}
