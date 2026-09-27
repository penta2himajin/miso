// Native norm controls; --check validates every control and genuine ping-pong data flow.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "controls.hpp"
#include "hip_util.hpp"

namespace nb = miso::norm_bench;

namespace {

constexpr float kEps = 1e-6f;
const char* name(nb::Kind k) {
  switch (k) {
    case nb::Kind::Empty:
      return "empty";
    case nb::Kind::Copy:
      return "copy";
    case nb::Kind::ReadWrite:
      return "readwrite";
    case nb::Kind::Norm:
      return "norm";
    case nb::Kind::Direct:
      return "direct";
  }
  return "unknown";
}

std::vector<float> values(unsigned slots, unsigned salt) {
  std::vector<float> v(std::size_t{slots} * nb::kN);
  for (unsigned i = 0; i < v.size(); ++i)
    v[i] = static_cast<float>(static_cast<int>((i * 13 + salt) % 101) - 50) / 64.0f;
  return v;
}

std::vector<float> weights() {
  auto v = values(1, 3);
  for (auto& x : v)
    x = 0.5f + x * 0.125f;
  return v;
}

void oracle(nb::Kind kind, std::vector<float>& residual, const std::vector<float>& delta,
            const std::vector<float>& weight, std::vector<float>& out, bool add) {
  if (kind == nb::Kind::Empty || kind == nb::Kind::Direct)
    return;
  if (kind == nb::Kind::Copy) {
    out = residual;
    return;
  }
  std::vector<float> h(residual.size());
  long double ss = 0;
  for (unsigned i = 0; i < h.size(); ++i) {
    h[i] = add ? residual[i] + delta[i] : residual[i];
    ss += static_cast<long double>(h[i]) * h[i];
  }
  const long double r = kind == nb::Kind::Norm ? 1 / std::sqrt(ss / nb::kN + kEps) : 1;
  for (unsigned i = 0; i < h.size(); ++i) {
    if (add)
      residual[i] = h[i];
    out[i] = kind == nb::Kind::Norm ? static_cast<float>(h[i] * r * weight[i]) : h[i] * weight[i];
  }
}

void compare(const std::vector<float>& got, const std::vector<float>& expected, bool exact,
             const char* label) {
  if (got.size() != expected.size())
    throw std::runtime_error("check size mismatch");
  for (unsigned i = 0; i < got.size(); ++i) {
    const bool ok =
        exact ? std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(expected[i])
              : std::isfinite(got[i]) &&
                    std::abs(got[i] - expected[i]) <= 2e-5f * (1 + std::abs(expected[i]));
    if (!ok)
      throw std::runtime_error(std::string(label) + " mismatch at " + std::to_string(i));
  }
}

void check_chain(nb::Kind kind, bool add, bool dependent, bool graph_mode) {
  constexpr unsigned kSteps = 7, kSlots = 3;
  auto a = values(kSlots, 7), b = values(kSlots, 23);
  auto res = values(kSlots, 29), middle = values(kSlots, 31);
  const auto d = values(1, 5), w = weights();
  miso::DeviceBuffer<float> da(a), db(b), dr(res), dm(middle), dd(d), dw(w);
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  if (graph_mode)
    MISO_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (unsigned step = 0; step < kSteps; ++step) {
    const unsigned slot = dependent ? 0 : step % kSlots;
    const unsigned at = slot * nb::kN;
    const bool reverse = dependent && step % 2 != 0;
    const float* source = (reverse ? db.data() : da.data()) + at;
    float* dest = (reverse ? da.data() : db.data()) + at;
    nb::producer_kernel<<<1, nb::kBlock, 0, stream>>>(source, dr.data() + at, dm.data() + at);
    nb::launch(kind, {dr.data() + at, add ? dd.data() : nullptr, dw.data(), dm.data() + at, kEps},
               stream);
    nb::consumer_kernel<<<1, nb::kBlock, 0, stream>>>(dm.data() + at, dest);
    MISO_HIP_CHECK(hipGetLastError());

    const auto& src = reverse ? b : a;
    auto& dst = reverse ? a : b;
    std::vector<float> hr(nb::kN), hm(nb::kN);
    for (unsigned i = 0; i < nb::kN; ++i)
      hr[i] = hm[i] = std::fma(src[at + i], 0.5f, 0.25f);
    oracle(kind, hr, d, w, hm, add);
    for (unsigned i = 0; i < nb::kN; ++i) {
      res[at + i] = hr[i];
      middle[at + i] = hm[i];
      dst[at + i] = std::fma(hm[i], 0.5f, -0.125f);
    }
  }
  if (graph_mode) {
    MISO_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    MISO_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    MISO_HIP_CHECK(hipGraphLaunch(executable, stream));
  }
  MISO_HIP_CHECK(hipStreamSynchronize(stream));
  const bool exact = kind != nb::Kind::Norm;
  compare(da.to_host(), a, exact, "chain A");
  compare(db.to_host(), b, exact, "chain B");
  compare(dr.to_host(), res, exact, "chain residual");
  compare(dm.to_host(), middle, exact, "chain middle");
  if (executable)
    MISO_HIP_CHECK(hipGraphExecDestroy(executable));
  if (graph)
    MISO_HIP_CHECK(hipGraphDestroy(graph));
  MISO_HIP_CHECK(hipStreamDestroy(stream));
}

void check() {
  const auto d = values(1, 5), w = weights();
  miso::DeviceBuffer<float> delta(d), weight(w);
  for (const bool add : {false, true}) {
    for (const auto kind : {nb::Kind::Empty, nb::Kind::Copy, nb::Kind::ReadWrite, nb::Kind::Norm}) {
      auto r = values(1, 7), y = values(1, 11);
      miso::DeviceBuffer<float> residual(r), out(y);
      nb::launch(kind,
                 {residual.data(), add ? delta.data() : nullptr, weight.data(), out.data(), kEps},
                 nullptr);
      MISO_HIP_CHECK(hipGetLastError());
      oracle(kind, r, d, w, y, add);
      compare(residual.to_host(), r, true, "residual");
      compare(out.to_host(), y, kind != nb::Kind::Norm, "output");
    }
  }
  std::puts("# check standalone controls passed");
  for (const bool graph_mode : {false, true})
    for (const bool dependent : {false, true})
      for (const bool add : {false, true})
        for (const auto kind : {nb::Kind::Direct, nb::Kind::Empty, nb::Kind::Copy,
                                nb::Kind::ReadWrite, nb::Kind::Norm})
          check_chain(kind, add, dependent, graph_mode);
  std::puts("# check native/graph producer/control/consumer ping-pong and independent ring passed");
}

struct Options {
  unsigned iterations = 1000, rounds = 7, warmup = 200, slots = 64;
  std::string mode = "native";
  bool check_only = false;
};

Options options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--check")
      o.check_only = true;
    else if (arg == "--graph")
      o.mode = "graph";
    else if (arg == "--help") {
      std::puts(
          "bench_norm [--check] [--iterations N] [--rounds N] [--warmup N] "
          "[--slots N] [--mode native|graph|both] [--graph]");
      std::exit(0);
    } else {
      if (++i == argc)
        throw std::runtime_error("missing value for " + arg);
      if (arg == "--mode") {
        o.mode = argv[i];
        continue;
      }
      std::size_t end = 0;
      const auto value = std::stoul(argv[i], &end);
      if (end != std::string(argv[i]).size() || value == 0 || value > 1000000)
        throw std::runtime_error("invalid numeric option " + arg);
      if (arg == "--iterations")
        o.iterations = value;
      else if (arg == "--rounds")
        o.rounds = value;
      else if (arg == "--warmup")
        o.warmup = value;
      else if (arg == "--slots")
        o.slots = value;
      else
        throw std::runtime_error("unknown option " + arg);
    }
  }
  if (o.mode != "native" && o.mode != "graph" && o.mode != "both")
    throw std::runtime_error("mode must be native, graph, or both");
  if (o.slots < 2)
    throw std::runtime_error("independent ring requires at least two slots");
  return o;
}

struct Case {
  nb::Kind kind;
  bool dependent, chain, add;
  unsigned slots;
  std::vector<float> initial_a, initial_b, initial_r, initial_m;
  miso::DeviceBuffer<float> a, b, residual, middle, delta, weight;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;

  Case(nb::Kind k, bool dep, bool ch, bool ad, unsigned ring)
      : kind(k),
        dependent(dep),
        chain(ch),
        add(ad),
        slots(dep ? 1 : ring),
        initial_a(values(slots, 7)),
        initial_b(values(slots, 23)),
        initial_r(values(slots, 29)),
        initial_m(values(slots, 31)),
        a(initial_a),
        b(initial_b),
        residual(initial_r),
        middle(initial_m),
        delta(values(1, 5)),
        weight(weights()) {}
  ~Case() {
    if (executable)
      (void)hipGraphExecDestroy(executable);
    if (graph)
      (void)hipGraphDestroy(graph);
  }

  void reset(hipStream_t stream) {
    const auto copy = [&](auto& device, const auto& host) {
      MISO_HIP_CHECK(hipMemcpyAsync(device.data(), host.data(), host.size() * sizeof(float),
                                    hipMemcpyHostToDevice, stream));
    };
    copy(a, initial_a);
    copy(b, initial_b);
    copy(residual, initial_r);
    copy(middle, initial_m);
    MISO_HIP_CHECK(hipStreamSynchronize(stream));
  }

  void enqueue(unsigned steps, hipStream_t stream) {
    for (unsigned step = 0; step < steps; ++step) {
      const unsigned at = (dependent ? 0 : step % slots) * nb::kN;
      const bool reverse = dependent && step % 2 != 0;
      float* source = (reverse ? b.data() : a.data()) + at;
      float* dest = (reverse ? a.data() : b.data()) + at;
      nb::Params p{source, add ? delta.data() : nullptr, weight.data(), dest, kEps};
      if (chain) {
        nb::producer_kernel<<<1, nb::kBlock, 0, stream>>>(source, residual.data() + at,
                                                          middle.data() + at);
        p.residual = residual.data() + at;
        p.out = middle.data() + at;
      }
      nb::launch(kind, p, stream);
      if (chain)
        nb::consumer_kernel<<<1, nb::kBlock, 0, stream>>>(middle.data() + at, dest);
    }
    MISO_HIP_CHECK(hipGetLastError());
  }

  void capture(unsigned steps, hipStream_t stream) {
    MISO_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    enqueue(steps, stream);
    MISO_HIP_CHECK(hipStreamEndCapture(stream, &graph));
    MISO_HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  }
};

void benchmark(const Options& o) {
  int device = 0, runtime = 0;
  hipDeviceProp_t props{};
  MISO_HIP_CHECK(hipGetDevice(&device));
  MISO_HIP_CHECK(hipGetDeviceProperties(&props, device));
  MISO_HIP_CHECK(hipRuntimeGetVersion(&runtime));
  std::printf("# device=%s arch=%s CUs=%d HIP_runtime=%d\n", props.name, props.gcnArchName,
              props.multiProcessorCount, runtime);
  std::printf("# N=%u block=%u iterations=%u rounds=%u warmup=%u independent_ring=%u\n", nb::kN,
              nb::kBlock, o.iterations, o.rounds, o.warmup, o.slots);
  std::puts("# Empty has no RAW edge; chain Empty/Direct bypass middle work via producer's copy.");
  std::puts("# Independent uses a ring with no adjacent RAW edges, not infinite disjoint memory.");
  std::puts("# Graph captures an entire iteration batch; capture/instantiate/reset excluded.");
  std::puts("# Differences are controls, not an exact additive decomposition or fusion speedup.");

  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr, end = nullptr;
  MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  MISO_HIP_CHECK(hipEventCreate(&start));
  MISO_HIP_CHECK(hipEventCreate(&end));
  std::vector<std::unique_ptr<Case>> cases;
  for (const bool chain : {false, true})
    for (const bool dependent : {false, true})
      for (const bool add : {false, true})
        for (const auto kind : {nb::Kind::Empty, nb::Kind::Copy, nb::Kind::ReadWrite,
                                nb::Kind::Norm, nb::Kind::Direct}) {
          if (!chain && kind == nb::Kind::Direct)
            continue;
          cases.push_back(std::make_unique<Case>(kind, dependent, chain, add, o.slots));
          if (o.mode != "native")
            cases.back()->capture(o.iterations, stream);
        }
  struct Work {
    Case* item;
    bool graph;
  };
  std::vector<Work> work;
  for (const auto& c : cases) {
    if (o.mode != "graph")
      work.push_back({c.get(), false});
    if (o.mode != "native")
      work.push_back({c.get(), true});
  }
  std::puts(
      "execution,path,sequence,variant,delta,round,iterations,ring_slots,kernels_per_step,us_per_"
      "step,enqueue_us_per_step");
  for (unsigned round = 0; round < o.rounds; ++round) {
    for (unsigned order = 0; order < work.size(); ++order) {
      const unsigned offset = round % 2 == 0 ? order : work.size() - 1 - order;
      const auto& job = work[(offset + round) % work.size()];
      auto& c = *job.item;
      c.reset(stream);
      c.enqueue(o.warmup, stream);
      MISO_HIP_CHECK(hipEventRecord(start, stream));
      const auto enqueue_start = std::chrono::steady_clock::now();
      if (job.graph)
        MISO_HIP_CHECK(hipGraphLaunch(c.executable, stream));
      else
        c.enqueue(o.iterations, stream);
      const auto enqueue_end = std::chrono::steady_clock::now();
      MISO_HIP_CHECK(hipEventRecord(end, stream));
      MISO_HIP_CHECK(hipEventSynchronize(end));
      float ms = 0;
      MISO_HIP_CHECK(hipEventElapsedTime(&ms, start, end));
      const unsigned kernels = c.chain ? (c.kind == nb::Kind::Direct ? 2 : 3) : 1;
      const double enqueue_us =
          std::chrono::duration<double, std::micro>(enqueue_end - enqueue_start).count();
      std::printf("%s,%s,%s,%s,%u,%u,%u,%u,%u,%.6f,%.6f\n", job.graph ? "graph" : "native",
                  c.dependent ? "dependent" : "independent", c.chain ? "chain" : "standalone",
                  name(c.kind), c.add ? 1u : 0u, round + 1, o.iterations, c.slots, kernels,
                  ms * 1000 / o.iterations, enqueue_us / o.iterations);
    }
    std::fflush(stdout);
  }
  cases.clear();
  MISO_HIP_CHECK(hipEventDestroy(start));
  MISO_HIP_CHECK(hipEventDestroy(end));
  MISO_HIP_CHECK(hipStreamDestroy(stream));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = options(argc, argv);
    if (o.check_only) {
      check();
      return 0;
    }
    benchmark(o);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
