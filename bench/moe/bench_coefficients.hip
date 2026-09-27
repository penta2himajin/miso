// Native same-binary controls for computing MoE routing coefficients only in workgroup 0.
// Frozen baseline and production gate/up share real weights; inputs rotate over golden tokens.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/moe.hpp"
#include "frozen_gateup.hpp"

namespace {
using namespace miso;
using namespace miso::kernels;
constexpr unsigned kBlock = 256, kGrid = 120, kTokens = 17;

template <bool Candidate>
__global__ void __launch_bounds__(kBlock)
    setup_kernel(const float* logits, int* ids, float* weights) {
  __shared__ int sid[moe::kTopK];
  __shared__ float sw[moe::kTopK];
  if constexpr (Candidate)
    moe_topk<kBlock>(logits, sid, sw, blockIdx.x == 0);
  else
    coeff_bench::frozen::moe_topk<kBlock>(logits, sid, sw);
  if (threadIdx.x < moe::kTopK) {
    ids[blockIdx.x * moe::kTopK + threadIdx.x] = sid[threadIdx.x];
    if (blockIdx.x == 0)
      weights[threadIdx.x] = sw[threadIdx.x];
  }
}

template <class T>
void require_bits(const std::vector<T>& actual, const std::vector<T>& expected, const char* label) {
  if (actual.size() != expected.size() ||
      std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) != 0)
    throw std::runtime_error(std::string("bit mismatch: ") + label);
}

struct Probe {
  Probe(const gguf::File& model, const gguf::File& golden, int layer, hipStream_t stream)
      : w(MoeLayer::load(model, layer)),
        x(kTokens * moe::kHidden),
        cached_logits(kTokens * (moe::kExperts + 1)),
        y0(moe::kHidden),
        y1(moe::kHidden),
        ids0(kGrid * moe::kTopK),
        ids1(kGrid * moe::kTopK),
        weights0(moe::kTopK),
        weights1(moe::kTopK) {
    const std::string wanted = "layer" + std::to_string(layer) + ".moe_in";
    const auto* input = golden.find_tensor(wanted);
    input_source = input ? wanted : "layer0.moe_in";
    if (!input)
      input = golden.find_tensor(input_source);
    if (!input)
      throw std::runtime_error("golden MoE input missing");
    const auto bytes = golden.tensor_data(*input);
    if (bytes.size() != kTokens * moe::kHidden * sizeof(float))
      throw std::runtime_error("expected 17 golden MoE inputs");
    MISO_HIP_CHECK(hipMemcpy(x.data(), bytes.data(), bytes.size(), hipMemcpyHostToDevice));
    for (unsigned token = 0; token < kTokens; ++token) {
      const MoeRouterParams rp{w.router.data(), x.data() + token * moe::kHidden,
                               cached_logits.data() + token * (moe::kExperts + 1)};
      moe_router_kernel<kBlock><<<65, kBlock, 0, stream>>>(rp);
    }
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
  }

  MoeLayer w;
  DeviceBuffer<float> x, cached_logits, y0, y1;
  DeviceBuffer<int> ids0, ids1;
  DeviceBuffer<float> weights0, weights1;
  MoeScratch baseline, candidate;
  std::string input_source;

  void gate_up(bool current, MoeScratch& scratch, const float* logits, const float* input,
               hipStream_t stream) {
    const MoeGateUpParams p{logits,
                            w.gate_exps.data.data(),
                            w.up_exps.data.data(),
                            w.gate_sh.data.data(),
                            w.up_sh.data.data(),
                            input,
                            scratch.h.data(),
                            scratch.ids.data(),
                            scratch.weights.data()};
    if (current)
      moe_gate_up_kernel<kBlock><<<kGrid, kBlock, 0, stream>>>(p);
    else
      coeff_bench::frozen::gate_up_kernel<kBlock><<<kGrid, kBlock, 0, stream>>>(p);
  }

  void step(bool current, unsigned scope, unsigned token, hipStream_t stream) {
    const float* input = x.data() + token * moe::kHidden;
    const float* logits = cached_logits.data() + token * (moe::kExperts + 1);
    if (scope == 0) {
      if (current)
        setup_kernel<true><<<kGrid, kBlock, 0, stream>>>(logits, ids1.data(), weights1.data());
      else
        setup_kernel<false><<<kGrid, kBlock, 0, stream>>>(logits, ids0.data(), weights0.data());
      return;
    }
    auto& scratch = current ? candidate : baseline;
    float* output = current ? y1.data() : y0.data();
    if (scope == 2) {
      const MoeRouterParams rp{w.router.data(), input, scratch.logits.data()};
      moe_router_kernel<kBlock><<<65, kBlock, 0, stream>>>(rp);
      logits = scratch.logits.data();
    }
    gate_up(current, scratch, logits, input, stream);
    if (scope == 2) {
      const MoeDownParams dp{logits,
                             scratch.ids.data(),
                             scratch.weights.data(),
                             w.down_exps.data.data(),
                             w.down_sh.data.data(),
                             scratch.h.data(),
                             output};
      if (w.down_exps.type == gguf::GgmlType::Q4_K)
        moe_down_kernel<QType::Q4K, kBlock><<<kGrid, kBlock, 0, stream>>>(dp);
      else
        moe_down_kernel<QType::Q6K, kBlock><<<kGrid, kBlock, 0, stream>>>(dp);
    }
  }

  void check(hipStream_t stream) {
    for (unsigned token = 0; token < kTokens; ++token) {
      for (unsigned scope = 0; scope < 3; ++scope) {
        step(false, scope, token, stream);
        step(true, scope, token, stream);
        MISO_HIP_CHECK(hipGetLastError());
        MISO_HIP_CHECK(hipStreamSynchronize(stream));
        if (scope == 0) {
          require_bits(ids1.to_host(), ids0.to_host(), "setup all-workgroup IDs");
          require_bits(weights1.to_host(), weights0.to_host(), "setup workgroup-0 weights");
        } else {
          require_bits(candidate.ids.to_host(), baseline.ids.to_host(), "gate/up IDs");
          require_bits(candidate.weights.to_host(), baseline.weights.to_host(), "gate/up weights");
          require_bits(candidate.h.to_host(), baseline.h.to_host(), "all gate/up activations");
          if (scope == 2)
            require_bits(y1.to_host(), y0.to_host(), "whole MoE output");
        }
      }
    }
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    unsigned iterations = 1000, rounds = 7, warmup = 200;
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
    std::printf("device=%s arch=%s iterations=%u rounds=%u warmup=%u tokens=%u baseline=438fb45\n",
                device.name, device.gcnArchName, iterations, rounds, warmup, kTokens);
    std::puts("# Setup and gate/up exclude router; whole_moe includes router/gateup/down.");
    std::puts("# Setup consumes all workgroup IDs and only workgroup-0 coefficients.");
    std::puts(
        "layer,down_type,scope,variant,round,iterations,device_us_per_step,cpu_enqueue_us_per_"
        "step");
    const char* scope_names[] = {"setup", "gate_up", "whole_moe"};
    for (int layer : {0, 5}) {
      Probe p(model, golden, layer, stream);
      p.check(stream);
      std::printf("check=layer%d:pass tokens=%u input=%s\n", layer, kTokens,
                  p.input_source.c_str());
      if (check_only)
        continue;
      for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned order = 0; order < 6; ++order) {
          const unsigned index = (order + round) % 6, scope = index / 2;
          const bool current = index % 2 != 0;
          const unsigned phase = round * 37 % kTokens;
          for (unsigned i = 0; i < warmup; ++i)
            p.step(current, scope, (i + phase) % kTokens, stream);
          MISO_HIP_CHECK(hipStreamSynchronize(stream));
          MISO_HIP_CHECK(hipEventRecord(begin, stream));
          const auto cpu_begin = std::chrono::steady_clock::now();
          for (unsigned i = 0; i < iterations; ++i)
            p.step(current, scope, (i + phase) % kTokens, stream);
          const auto cpu_end = std::chrono::steady_clock::now();
          MISO_HIP_CHECK(hipEventRecord(end, stream));
          MISO_HIP_CHECK(hipGetLastError());
          MISO_HIP_CHECK(hipEventSynchronize(end));
          float ms = 0;
          MISO_HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
          const double enqueue_us =
              std::chrono::duration<double, std::micro>(cpu_end - cpu_begin).count() / iterations;
          std::printf("%d,%s,%s,%s,%u,%u,%.6f,%.6f\n", layer,
                      p.w.down_exps.type == gguf::GgmlType::Q4_K ? "Q4_K" : "Q6_K",
                      scope_names[scope], current ? "candidate" : "baseline", round, iterations,
                      ms * 1000 / iterations, enqueue_us);
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
