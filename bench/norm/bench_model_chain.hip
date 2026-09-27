// Diagnostic real-weight producer -> control -> consumer chains. Control variants deliberately
// change the math; these are timing probes, not valid inference alternatives.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "controls.hpp"
#include "engine/weights.hpp"
#include "moe_decode.hpp"

namespace {
using namespace miso;
using namespace miso::norm_bench;
constexpr Kind kKinds[] = {Kind::Direct, Kind::Empty, Kind::ReadWrite, Kind::Norm};
constexpr const char* kNames[] = {"direct", "empty", "readwrite", "norm"};

struct Probe {
  Probe(const gguf::File& f, int layer, bool is_moe)
      : producer(upload_qmatrix(f, "blk." + std::to_string(layer) +
                                       (is_moe ? ".ffn_down_exps.weight" : ".ssm_out.weight"))),
        shared(upload_qmatrix(f, "blk." + std::to_string(layer) + ".ffn_down_shexp.weight")),
        consumer(upload_qmatrix(f, "blk." + std::to_string(layer + 1) + ".attn_qkv.weight")),
        norm_w(upload_f32(f, "blk." + std::to_string(is_moe ? layer + 1 : layer) +
                                 (is_moe ? ".attn_norm.weight" : ".post_attention_norm.weight"))),
        x(is_moe ? 9 * 512 : 4096),
        residual(kN),
        delta(kN),
        out(kN),
        logits(257),
        next(consumer.n),
        weights(std::vector<float>(8, 0.125f)),
        ids(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}),
        eps(static_cast<float>(f.get<double>("qwen35moe.attention.layer_norm_rms_epsilon"))),
        moe(is_moe) {
    if ((!moe && (producer.n != kN || producer.k != 4096)) ||
        (moe && (producer.k != 512 || shared.n != kN || shared.type != producer.type)) ||
        consumer.k != kN)
      throw std::runtime_error("unexpected chain shape/type");
    // Router tensors are stored as F32 with BF16-exact values in this GGUF.
    std::vector<std::uint16_t> rw;
    for (const char* suffix : {".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight"}) {
      const auto* t = f.find_tensor("blk." + std::to_string(layer) + suffix);
      if (t == nullptr || t->type != gguf::GgmlType::F32)
        throw std::runtime_error("router tensor missing");
      const auto bytes = f.tensor_data(*t);
      for (std::size_t i = 0; i < bytes.size(); i += 4) {
        std::uint32_t bits;
        std::memcpy(&bits, bytes.data() + i, 4);
        if ((bits & 65535u) != 0)
          throw std::runtime_error("router is not BF16 exact");
        rw.push_back(static_cast<std::uint16_t>(bits >> 16));
      }
    }
    router = std::make_unique<DeviceBuffer<std::uint16_t>>(rw);
    std::vector<float> input(x.size());
    for (unsigned i = 0; i < input.size(); ++i)
      input[i] = (static_cast<int>(i % 31) - 15) * 0.015625f;
    MISO_HIP_CHECK(hipMemcpy(x.data(), input.data(), input.size() * 4, hipMemcpyHostToDevice));
    MISO_HIP_CHECK(hipMemset(logits.data(), 0, logits.size() * 4));
  }

  QMatrix producer, shared, consumer;
  DeviceBuffer<float> norm_w, x, residual, delta, out, logits, next, weights;
  DeviceBuffer<int> ids;
  std::unique_ptr<DeviceBuffer<std::uint16_t>> router;
  float eps;
  bool moe;

  void reset(hipStream_t stream) {
    MISO_HIP_CHECK(hipMemsetAsync(residual.data(), 0, kN * 4, stream));
    MISO_HIP_CHECK(hipMemsetAsync(logits.data(), 0, 257 * 4, stream));
  }

  void step(Kind kind, hipStream_t stream) {
    if (!moe) {
      gemv(producer, x.data(), delta.data(), stream);
    } else {
      const kernels::MoeDownParams dp{logits.data(),        ids.data(),         weights.data(),
                                      producer.data.data(), shared.data.data(), x.data(),
                                      delta.data()};
      if (producer.type == gguf::GgmlType::Q4_K)
        kernels::moe_down_kernel<kernels::QType::Q4K, 256><<<120, 256, 0, stream>>>(dp);
      else
        kernels::moe_down_kernel<kernels::QType::Q6K, 256><<<120, 256, 0, stream>>>(dp);
    }
    launch(kind, Params{residual.data(), delta.data(), norm_w.data(), out.data(), eps}, stream);
    // Empty/direct have no middle write. Keep a real producer-to-consumer RAW edge by reading
    // delta in these cases. Pointer/cache/math differ: this is not a legal Norm removal.
    const float* source = kind == Kind::Direct || kind == Kind::Empty ? delta.data() : out.data();
    if (moe) {
      gemv(consumer, source, next.data(), stream);
    } else {
      const kernels::MoeRouterParams rp{router->data(), source, logits.data()};
      kernels::moe_router_kernel<256><<<65, 256, 0, stream>>>(rp);
    }
  }
  std::vector<float> result() const { return moe ? next.to_host() : logits.to_host(); }
};

struct Batch {
  Probe& p;
  Kind kind;
  unsigned iterations;
  hipStream_t stream;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  Batch(Probe& probe, Kind k, unsigned n, hipStream_t s)
      : p(probe), kind(k), iterations(n), stream(s) {
    MISO_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (unsigned i = 0; i < iterations; ++i)
      p.step(kind, stream);
    MISO_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    MISO_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
  ~Batch() {
    (void)hipGraphExecDestroy(executable);
    (void)hipGraphDestroy(graph);
  }
  void enqueue(bool use_graph) {
    if (use_graph)
      MISO_HIP_CHECK(hipGraphLaunch(executable, stream));
    else
      for (unsigned i = 0; i < iterations; ++i)
        p.step(kind, stream);
    MISO_HIP_CHECK(hipGetLastError());
  }
};

void verify(Probe& p, hipStream_t stream) {
  for (Kind kind : kKinds) {
    Batch b(p, kind, 4, stream);
    p.reset(stream);
    b.enqueue(false);
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
    const auto raw = p.result(), residual_raw = p.residual.to_host();
    p.reset(stream);
    b.enqueue(true);
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
    const auto graph = p.result(), residual_graph = p.residual.to_host();
    if (raw.size() != graph.size() || std::memcmp(raw.data(), graph.data(), raw.size() * 4) != 0 ||
        std::memcmp(residual_raw.data(), residual_graph.data(), kN * 4) != 0)
      throw std::runtime_error("native/graph output or residual mismatch");
    for (float v : raw)
      if (!std::isfinite(v))
        throw std::runtime_error("nonfinite chain output");
  }
}
}  // namespace

int main(int argc, char** argv) {
  unsigned iterations = 256, rounds = 7, warmup = 256;
  bool check_only = false;
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
    else
      throw std::runtime_error("unknown or incomplete option: " + arg);
  }
  if (!iterations || !rounds)
    throw std::runtime_error("iterations/rounds must be positive");
  const auto file = miso::gguf::File::open(MISO_ORNITH_GGUF);
  hipStream_t stream;
  hipEvent_t begin, end;
  MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  MISO_HIP_CHECK(hipEventCreate(&begin));
  MISO_HIP_CHECK(hipEventCreate(&end));
  hipDeviceProp_t prop;
  MISO_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device=%s arch=%s iterations=%u rounds=%u warmup=%u synthetic_inputs=1\n", prop.name,
              prop.gcnArchName, iterations, rounds, warmup);
  std::printf(
      "chain,execution,control,round,iterations,device_us_per_step,cpu_enqueue_us_per_step\n");
  // Both source down types are probed; the label is derived from the loaded tensor type.
  for (unsigned probe_i = 0; probe_i < 3; ++probe_i) {
    Probe p(file, probe_i == 2 ? 5 : 0, probe_i != 0);
    const char* chain = probe_i == 0                                    ? "gemv_norm_router"
                        : p.producer.type == miso::gguf::GgmlType::Q4_K ? "moe4_norm_qkv"
                                                                        : "moe6_norm_qkv";
    verify(p, stream);
    std::printf("chain_check=%s:pass producer_n=%u producer_k=%u consumer_n=%u\n", chain,
                p.producer.n, p.producer.k, p.moe ? p.consumer.n : 257);
    if (check_only)
      continue;
    std::vector<std::unique_ptr<Batch>> batches;
    for (Kind kind : kKinds)
      batches.push_back(std::make_unique<Batch>(p, kind, iterations, stream));
    for (unsigned round = 0; round < rounds; ++round) {
      for (unsigned index = 0; index < 8; ++index) {
        const unsigned rotated = (index + round) % 8, variant = rotated / 2;
        const bool graph = rotated % 2 != 0;
        auto& b = *batches[variant];
        p.reset(stream);
        for (unsigned w = 0; w < warmup; ++w)
          p.step(kKinds[variant], stream);
        MISO_HIP_CHECK(hipStreamSynchronize(stream));
        p.reset(stream);
        MISO_HIP_CHECK(hipEventRecord(begin, stream));
        const auto cpu_begin = std::chrono::steady_clock::now();
        b.enqueue(graph);
        const auto cpu_end = std::chrono::steady_clock::now();
        MISO_HIP_CHECK(hipEventRecord(end, stream));
        MISO_HIP_CHECK(hipEventSynchronize(end));
        float ms;
        MISO_HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
        const double enqueue_us =
            std::chrono::duration<double, std::micro>(cpu_end - cpu_begin).count() / iterations;
        std::printf("%s,%s,%s,%u,%u,%.6f,%.6f\n", chain, graph ? "graph" : "native",
                    kNames[variant], round, iterations, ms * 1000.0 / iterations, enqueue_us);
        std::fflush(stdout);
      }
    }
  }
  MISO_HIP_CHECK(hipEventDestroy(begin));
  MISO_HIP_CHECK(hipEventDestroy(end));
  MISO_HIP_CHECK(hipStreamDestroy(stream));
}
