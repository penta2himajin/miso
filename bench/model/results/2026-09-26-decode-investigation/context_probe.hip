// Archived measurement probe; links the unchanged production engine.
// One prefilled snapshot per context. Each repeat restores recurrent/KV state and the last
// prediction outside the timed interval, then greedily decodes the same 64 tokens.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "engine/model.hpp"

template <class T>
void copy_buffer(miso::DeviceBuffer<T>& dst, const miso::DeviceBuffer<T>& src) {
  MISO_HIP_CHECK(
      hipMemcpy(dst.data(), src.data(), src.size() * sizeof(T), hipMemcpyDeviceToDevice));
}

void restore(miso::Session& dst, const miso::Session& src) {
  dst.pos = src.pos;
  // Only the last prediction is read before the next decode step writes its prediction.
  MISO_HIP_CHECK(hipMemcpy(dst.pred.data() + src.pos - 1, src.pred.data() + src.pos - 1,
                           sizeof(int), hipMemcpyDeviceToDevice));
  for (std::size_t i = 0; i < src.kv.size(); ++i) {
    if (src.kv[i]) {
      dst.kv[i]->len = src.kv[i]->len;
      copy_buffer(dst.kv[i]->k, src.kv[i]->k);
      copy_buffer(dst.kv[i]->v, src.kv[i]->v);
    }
    if (src.deltanet[i]) {
      dst.deltanet[i]->parity = src.deltanet[i]->parity;
      copy_buffer(dst.deltanet[i]->s, src.deltanet[i]->s);
      for (int j = 0; j < 2; ++j)
        copy_buffer(dst.deltanet[i]->conv[j], src.deltanet[i]->conv[j]);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4)
    return 2;  // model, golden GGUF, context length
  const unsigned ctx = static_cast<unsigned>(std::stoul(argv[3]));
  if (ctx == 0 || ctx > 32768)
    return 2;
  constexpr unsigned gen = 64;
  const auto f = miso::gguf::File::open(argv[1]);
  const auto model = miso::Model::load(f);
  const auto g = miso::gguf::File::open(argv[2]);
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  std::vector<int> prompt(ctx);
  for (unsigned i = 0; i < ctx; ++i)
    prompt[i] = static_cast<int>(ids[i % ids.size()]);
  miso::Session snapshot(model, ctx + gen), s(model, ctx + gen);
  MISO_HIP_CHECK(
      hipMemcpy(snapshot.tokens.data(), prompt.data(), ctx * sizeof(int), hipMemcpyHostToDevice));
  miso::ingest(model, snapshot, snapshot.tokens.data(), ctx, nullptr);
  MISO_HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  MISO_HIP_CHECK(hipEventCreate(&a));
  MISO_HIP_CHECK(hipEventCreate(&b));
  std::vector<int> expected;
  std::vector<float> measured;
  for (int round = 0; round < 4; ++round) {
    restore(s, snapshot);
    MISO_HIP_CHECK(hipEventRecord(a));
    for (unsigned k = 0; k < gen; ++k)
      miso::decode_step(model, s, s.pred.data() + s.pos - 1, nullptr);
    MISO_HIP_CHECK(hipEventRecord(b));
    MISO_HIP_CHECK(hipEventSynchronize(b));
    float ms;
    MISO_HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    const auto all = s.pred.to_host();
    const std::vector<int> pred(all.begin() + ctx, all.begin() + ctx + gen);
    if (round == 0)
      expected = pred;
    else if (pred != expected)
      throw std::runtime_error("restored session predictions differ");
    if (s.pos != ctx + gen)
      throw std::runtime_error("unexpected session position");
    for (const auto& cache : s.kv)
      if (cache && cache->len != s.pos)
        throw std::runtime_error("KV and session positions differ");
    std::printf("%s round=%d ctx=%u..%u tokens=%u ms=%.6f tok_s=%.3f ids_match=yes\n",
                round ? "measured" : "warm-up", round, ctx, s.pos, gen, ms, gen * 1e3 / ms);
    std::fflush(stdout);
    if (round)
      measured.push_back(ms);
  }
  std::sort(measured.begin(), measured.end());
  std::printf("median ctx=%u ms_token=%.6f tok_s=%.3f\n", ctx, measured[1] / gen,
              gen * 1e3 / measured[1]);
  MISO_HIP_CHECK(hipEventDestroy(a));
  MISO_HIP_CHECK(hipEventDestroy(b));
}
