// Diagnostic real DeltaNet update -> K4096 output-GEMV chains. Production is unchanged.
// Each stateful batch starts from the same finite zero snapshot outside timed events.
#include <hip/hip_runtime.h>

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "activation_prepare.hpp"
#include "engine/deltanet.hpp"
#include "q4k_gemv.hpp"
#include "rmsnorm.hpp"

namespace {
using namespace miso;
namespace ap = miso::bench::activation_prepare;
constexpr unsigned kTokens = 17, kHidden = 2048, kCore = 4096, kProj = 12352;
constexpr const char* kVariants[] = {"baseline", "float_control", "dual_float", "dual_half",
                                     "half_only"};
constexpr const char* kScopes[] = {"producer", "consumer", "chain"};

__global__ void round_half_kernel(const float* input, _Float16* output, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = static_cast<_Float16>(input[i]);
}

template <class T>
void require_bits(const std::vector<T>& actual, const std::vector<T>& expected, const char* label) {
  if (actual.size() != expected.size() ||
      std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) != 0)
    throw std::runtime_error(std::string("bit mismatch: ") + label);
}

struct Distribution {
  std::size_t count = 0, zeros = 0, subnormals = 0;
  double absmax = 0;
};

template <class T>
Distribution distribution(const std::vector<T>& values) {
  Distribution result;
  result.count = values.size();
  for (const T value : values) {
    const float v = static_cast<float>(value);
    if (!std::isfinite(v))
      throw std::runtime_error("nonfinite state, conv, output or rounded activation");
    result.zeros += v == 0;
    if constexpr (sizeof(T) == 2) {
      std::uint16_t bits;
      std::memcpy(&bits, &value, 2);
      result.subnormals += (bits & 0x7fffu) > 0 && (bits & 0x7fffu) < 0x0400u;
    } else {
      result.subnormals += std::fpclassify(v) == FP_SUBNORMAL;
    }
    result.absmax = std::max(result.absmax, std::abs(static_cast<double>(v)));
  }
  return result;
}

void print_distribution(int layer, const char* label, const Distribution& d) {
  std::printf(
      "distribution layer=%d values=%s count=%zu zero_fraction=%.9f "
      "subnormal_fraction=%.9f absmax=%.9g\n",
      layer, label, d.count, static_cast<double>(d.zeros) / d.count,
      static_cast<double>(d.subnormals) / d.count, d.absmax);
}

struct Frame {
  DeltaNetState recurrent;
  DeviceBuffer<float> core{kCore}, output{kHidden};
  DeviceBuffer<_Float16> half{kCore};

  void reset(hipStream_t stream) {
    MISO_HIP_CHECK(
        hipMemsetAsync(recurrent.s.data(), 0, recurrent.s.size() * sizeof(float), stream));
    for (auto& conv : recurrent.conv)
      MISO_HIP_CHECK(hipMemsetAsync(conv.data(), 0, conv.size() * sizeof(float), stream));
    MISO_HIP_CHECK(hipMemsetAsync(core.data(), 0, core.size() * sizeof(float), stream));
    MISO_HIP_CHECK(hipMemsetAsync(half.data(), 0, half.size() * sizeof(_Float16), stream));
    MISO_HIP_CHECK(hipMemsetAsync(output.data(), 0, output.size() * sizeof(float), stream));
    recurrent.parity = 0;
  }
};

struct Snapshot {
  std::vector<float> state, conv0, conv1, core, output;
  std::vector<_Float16> half;
  int parity = 0;
};

Snapshot snapshot(const Frame& frame) {
  return {frame.recurrent.s.to_host(),
          frame.recurrent.conv[0].to_host(),
          frame.recurrent.conv[1].to_host(),
          frame.core.to_host(),
          frame.output.to_host(),
          frame.half.to_host(),
          frame.recurrent.parity};
}

struct Probe {
  Probe(const gguf::File& model, const gguf::File& golden, int layer, hipStream_t stream);
  void produce(Frame& frame, unsigned variant, unsigned token, hipStream_t stream);
  void consume(Frame& frame, unsigned variant, const float* f32, const _Float16* f16,
               hipStream_t stream);
  void step(Frame& frame, unsigned variant, unsigned scope, unsigned token, hipStream_t stream);
  void check_tokens(hipStream_t stream);
  Snapshot reference_batch(unsigned iterations, hipStream_t stream);
  void compare(Frame& frame, unsigned variant, unsigned scope, const Snapshot& expected,
               const Snapshot& consumer_expected);

  DeltaNetLayer weights;
  DeviceBuffer<float> projections{kTokens * kProj}, cached_core{kTokens * kCore};
  DeviceBuffer<_Float16> cached_half{kTokens * kCore};
  Frame reference, frames[5];
  float eps;
  int layer_index;
  std::string input_source;
};

Probe::Probe(const gguf::File& model, const gguf::File& golden, int layer, hipStream_t stream)
    : weights(DeltaNetLayer::load(model, layer)),
      eps(static_cast<float>(model.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"))),
      layer_index(layer) {
  if (weights.out.k != kCore || weights.out.n != kHidden ||
      weights.out.type != gguf::GgmlType::Q4_K)
    throw std::runtime_error("expected Q4_K K4096/N2048 output projection");
  const std::string wanted = "layer" + std::to_string(layer) + ".in";
  const auto* input = golden.find_tensor(wanted);
  input_source = input ? wanted : "layer0.in";
  if (!input)
    input = golden.find_tensor(input_source);
  if (!input)
    throw std::runtime_error("golden DeltaNet input missing");
  const auto bytes = golden.tensor_data(*input);
  if (bytes.size() != kTokens * kHidden * sizeof(float))
    throw std::runtime_error("expected 17 golden inputs");
  DeviceBuffer<float> hidden(kTokens * kHidden), normalized(kTokens * kHidden);
  MISO_HIP_CHECK(hipMemcpy(hidden.data(), bytes.data(), bytes.size(), hipMemcpyHostToDevice));
  for (unsigned token = 0; token < kTokens; ++token) {
    const kernels::AddRmsNormParams norm{hidden.data() + token * kHidden, nullptr,
                                         weights.attn_norm.data(),
                                         normalized.data() + token * kHidden, eps};
    kernels::add_rmsnorm_kernel<kHidden, kernels::kRmsNormBlock>
        <<<1, kernels::kRmsNormBlock, 0, stream>>>(norm, 1);
    float* proj = projections.data() + token * kProj;
    if (weights.z_ab.n)
      gemv_pair(weights.qkv, weights.z_ab, normalized.data() + token * kHidden, proj, proj + 8192,
                stream);
    else
      gemv(weights.qkv, normalized.data() + token * kHidden, proj, stream);
  }
  reference.reset(stream);
  for (unsigned token = 0; token < kTokens; ++token) {
    produce(reference, 0, token, stream);
    MISO_HIP_CHECK(hipMemcpyAsync(cached_core.data() + token * kCore, reference.core.data(),
                                  kCore * sizeof(float), hipMemcpyDeviceToDevice, stream));
  }
  round_half_kernel<<<(kTokens * kCore + 255) / 256, 256, 0, stream>>>(
      cached_core.data(), cached_half.data(), kTokens * kCore);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipStreamSynchronize(stream));
}

void Probe::produce(Frame& frame, unsigned variant, unsigned token, hipStream_t stream) {
  const float* proj = projections.data() + token * kProj;
  auto& state = frame.recurrent;
  const kernels::DeltaNetStepParams base{proj,
                                         proj + 8192,
                                         proj + 12288,
                                         proj + 12320,
                                         weights.conv_w.data(),
                                         state.conv[state.parity].data(),
                                         state.conv[1 - state.parity].data(),
                                         weights.ssm_a.data(),
                                         weights.dt_bias.data(),
                                         weights.ssm_norm.data(),
                                         state.s.data(),
                                         frame.core.data(),
                                         eps};
  const ap::StepParams prepared{base, frame.half.data()};
  if (variant == 0)
    kernels::deltanet_step_kernel<256><<<32, 256, 0, stream>>>(base);
  else if (variant == 1)
    ap::step_kernel<ap::Output::Fp32><<<32, 256, 0, stream>>>(prepared);
  else if (variant == 4)
    ap::step_kernel<ap::Output::Fp16><<<32, 256, 0, stream>>>(prepared);
  else
    ap::step_kernel<ap::Output::Dual><<<32, 256, 0, stream>>>(prepared);
  MISO_HIP_CHECK(hipGetLastError());
  state.parity = 1 - state.parity;
}

void Probe::consume(Frame& frame, unsigned variant, const float* f32, const _Float16* f16,
                    hipStream_t stream) {
  if (variant < 3) {
    const kernels::Q4kGemvParams p{weights.out.data.data(), f32, frame.output.data(),
                                   weights.out.n};
    kernels::q4k_gemv_kernel<kernels::ActFormat::Fp16, kCore, 256, 4><<<240, 256, 0, stream>>>(p);
  } else {
    const ap::HalfGemvParams p{weights.out.data.data(), f16, frame.output.data(), weights.out.n};
    ap::q4_half_gemv_kernel<kCore, 256, 4><<<240, 256, 0, stream>>>(p);
  }
  MISO_HIP_CHECK(hipGetLastError());
}

void Probe::step(Frame& frame, unsigned variant, unsigned scope, unsigned token,
                 hipStream_t stream) {
  if (scope != 1)
    produce(frame, variant, token, stream);
  if (scope != 0) {
    const float* f32 = scope == 1 ? cached_core.data() + token * kCore : frame.core.data();
    const _Float16* f16 = scope == 1 ? cached_half.data() + token * kCore : frame.half.data();
    consume(frame, variant, f32, f16, stream);
  }
}

void Probe::compare(Frame& frame, unsigned variant, unsigned scope, const Snapshot& expected,
                    const Snapshot& consumer_expected) {
  const auto got = snapshot(frame);
  if (scope == 1) {
    static const std::vector<float> zero_state(32 * 128 * 128, 0.0f);
    static const std::vector<float> zero_conv(8192 * 3, 0.0f);
    if (got.parity != 0)
      throw std::runtime_error("consumer-only changed conv parity");
    require_bits(got.state, zero_state, "consumer-only state must stay zero");
    require_bits(got.conv0, zero_conv, "consumer-only conv[0] must stay zero");
    require_bits(got.conv1, zero_conv, "consumer-only conv[1] must stay zero");
  } else {
    if (got.parity != expected.parity)
      throw std::runtime_error("conv parity mismatch");
    require_bits(got.state, expected.state, "state after batch");
    require_bits(got.conv0, expected.conv0, "conv[0] after batch");
    require_bits(got.conv1, expected.conv1, "conv[1] after batch");
    if (variant != 4)
      require_bits(got.core, expected.core, "FP32 producer output");
    if (variant >= 2)
      require_bits(got.half, expected.half, "rounded producer output");
  }
  if (scope != 0)
    require_bits(got.output, scope == 1 ? consumer_expected.output : expected.output,
                 "output GEMV after batch");
  // Every defined output equals a finite checked reference. Also check unused state buffers,
  // so a consumer-only test cannot silently corrupt the state or conv allocation.
  (void)distribution(got.state);
  (void)distribution(got.conv0);
  (void)distribution(got.conv1);
  if (scope != 1 && variant != 4)
    (void)distribution(got.core);
  if (scope != 1 && variant >= 2)
    (void)distribution(got.half);
  if (scope != 0)
    (void)distribution(got.output);
}

void Probe::check_tokens(hipStream_t stream) {
  reference.reset(stream);
  for (auto& frame : frames)
    frame.reset(stream);
  for (unsigned token = 0; token < kTokens; ++token) {
    step(reference, 0, 2, token, stream);
    round_half_kernel<<<16, 256, 0, stream>>>(reference.core.data(), reference.half.data(), kCore);
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
    const auto expected = snapshot(reference);
    for (unsigned variant = 0; variant < 5; ++variant) {
      step(frames[variant], variant, 2, token, stream);
      MISO_HIP_CHECK(hipGetLastError());
      MISO_HIP_CHECK(hipStreamSynchronize(stream));
      try {
        compare(frames[variant], variant, 2, expected, expected);
      } catch (const std::exception& error) {
        throw std::runtime_error(std::string("token=") + std::to_string(token) +
                                 " variant=" + kVariants[variant] + " " + error.what());
      }
    }
  }
  std::printf("check layer=%d scope=token_chain tokens=%u bitwise=pass input_source=%s\n",
              layer_index, kTokens, input_source.c_str());
}

Snapshot Probe::reference_batch(unsigned iterations, hipStream_t stream) {
  reference.reset(stream);
  DeviceBuffer<float> tail(34 * kCore);
  for (unsigned i = 0; i < iterations; ++i) {
    step(reference, 0, 2, i % kTokens, stream);
    if (iterations >= 34 && i >= iterations - 34)
      MISO_HIP_CHECK(hipMemcpyAsync(tail.data() + (i - (iterations - 34)) * kCore,
                                    reference.core.data(), kCore * sizeof(float),
                                    hipMemcpyDeviceToDevice, stream));
  }
  round_half_kernel<<<16, 256, 0, stream>>>(reference.core.data(), reference.half.data(), kCore);
  MISO_HIP_CHECK(hipGetLastError());
  MISO_HIP_CHECK(hipStreamSynchronize(stream));
  const auto expected = snapshot(reference);
  print_distribution(layer_index, "state", distribution(expected.state));
  print_distribution(layer_index, "conv0", distribution(expected.conv0));
  print_distribution(layer_index, "conv1", distribution(expected.conv1));
  print_distribution(layer_index, "producer_fp32", distribution(expected.core));
  print_distribution(layer_index, "producer_fp16", distribution(expected.half));
  print_distribution(layer_index, "consumer_fp32", distribution(expected.output));
  if (iterations >= 34) {
    const auto values = tail.to_host();
    std::size_t changed = 0;
    double num = 0, den = 0;
    for (unsigned i = 0; i < 17 * kCore; ++i) {
      const float previous = values[i], last = values[17 * kCore + i];
      changed += std::memcmp(&previous, &last, sizeof(float)) != 0;
      const double delta = static_cast<double>(last) - previous;
      num += delta * delta;
      den += static_cast<double>(previous) * previous;
    }
    std::printf("cycle_check layer=%d iterations=%u lag=17 changed_fraction=%.9f rel_l2=%.9g\n",
                layer_index, iterations, static_cast<double>(changed) / (17 * kCore),
                den > 0 ? std::sqrt(num / den) : (num > 0 ? __builtin_inf() : 0.0));
  } else {
    std::printf("cycle_check layer=%d iterations=%u lag=17 insufficient_updates=1\n", layer_index,
                iterations);
  }
  return expected;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned iterations = 2000, rounds = 7, warmup = 500;
    bool check_only = false;
    std::string model_path = MISO_ORNITH_GGUF;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--check")
        check_only = true;
      else if (i + 1 < argc && arg == "--iterations")
        iterations = std::stoul(argv[++i]);
      else if (i + 1 < argc && arg == "--rounds")
        rounds = std::stoul(argv[++i]);
      else if (i + 1 < argc && arg == "--warmup")
        warmup = std::stoul(argv[++i]);
      else if (i + 1 < argc && arg == "--model")
        model_path = argv[++i];
      else
        throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (!iterations || !rounds)
      throw std::runtime_error("iterations and rounds must be positive");
    const auto model = gguf::File::open(model_path);
    const auto golden = gguf::File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
    hipStream_t stream;
    hipEvent_t begin, end;
    MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    MISO_HIP_CHECK(hipEventCreate(&begin));
    MISO_HIP_CHECK(hipEventCreate(&end));
    hipDeviceProp_t device;
    MISO_HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("process=%ld device=%s arch=%s iterations=%u rounds=%u warmup=%u tokens=%u\n",
                static_cast<long>(getpid()), device.name, device.gcnArchName, iterations, rounds,
                warmup, kTokens);
    std::puts(
        "# Stateful periodic cache-hot single-layer proxy, not natural autoregressive state.");
    std::puts(
        "# Every warmup starts at zero; timed batches reset state/conv/parity outside events.");
    std::puts(
        "# Consumer-only input is an untimed consecutive 17-token production snapshot and device "
        "FP16 cast.");
    for (int layer : {0, 4}) {
      Probe p(model, golden, layer, stream);
      std::printf("input layer=%d source=%s weights_layer=%d matched_fixture=%u\n", layer,
                  p.input_source.c_str(), layer,
                  p.input_source == "layer" + std::to_string(layer) + ".in" ? 1u : 0u);
      p.check_tokens(stream);
      const auto expected = p.reference_batch(iterations, stream);
      const unsigned last_token = (iterations - 1) % kTokens;
      p.consume(p.reference, 0, p.cached_core.data() + last_token * kCore,
                p.cached_half.data() + last_token * kCore, stream);
      MISO_HIP_CHECK(hipStreamSynchronize(stream));
      const auto consumer_expected = snapshot(p.reference);
      for (unsigned round = 0; round < (check_only ? 1u : rounds); ++round) {
        for (unsigned order = 0; order < 15; ++order) {
          // Alternate direction as well as rotating the start. For seven rounds, each
          // candidate runs before its scope's baseline three times and after it four times.
          const unsigned index = round % 2 == 0 ? (order + round) % 15 : (15 + round - order) % 15;
          const unsigned scope = index / 5, variant = index % 5;
          auto& frame = p.frames[variant];
          frame.reset(stream);
          for (unsigned i = 0; i < warmup; ++i)
            p.step(frame, variant, scope, i % kTokens, stream);
          frame.reset(stream);
          MISO_HIP_CHECK(hipStreamSynchronize(stream));
          MISO_HIP_CHECK(hipEventRecord(begin, stream));
          const auto cpu_begin = std::chrono::steady_clock::now();
          for (unsigned i = 0; i < iterations; ++i)
            p.step(frame, variant, scope, i % kTokens, stream);
          const auto cpu_end = std::chrono::steady_clock::now();
          MISO_HIP_CHECK(hipEventRecord(end, stream));
          MISO_HIP_CHECK(hipGetLastError());
          MISO_HIP_CHECK(hipEventSynchronize(end));
          float ms = 0;
          MISO_HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
          try {
            p.compare(frame, variant, scope, expected, consumer_expected);
          } catch (const std::exception& error) {
            throw std::runtime_error("layer=" + std::to_string(layer) + " scope=" + kScopes[scope] +
                                     " variant=" + kVariants[variant] + " " + error.what());
          }
          const double enqueue_us =
              std::chrono::duration<double, std::micro>(cpu_end - cpu_begin).count() / iterations;
          if (check_only)
            std::printf(
                "check layer=%d scope=%s variant=%s iterations=%u bitwise=pass finite=pass\n",
                layer, kScopes[scope], kVariants[variant], iterations);
          else
            std::printf("layer=%d scope=%s variant=%s round=%u gpu_us=%.6f cpu_enqueue_us=%.6f\n",
                        layer, kScopes[scope], kVariants[variant], round, ms * 1000 / iterations,
                        enqueue_us);
          std::fflush(stdout);
        }
      }
    }
    MISO_HIP_CHECK(hipEventDestroy(begin));
    MISO_HIP_CHECK(hipEventDestroy(end));
    MISO_HIP_CHECK(hipStreamDestroy(stream));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
