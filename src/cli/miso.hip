// miso: greedy text generation with Ornith-1.5-35B-A3B on one MI50 (v1 CLI, ADR_002 D10).
//
//   miso [-m model.gguf] [-n max_new_tokens] [--no-think] [--raw] "prompt"
//
// --raw feeds the prompt as-is; otherwise it is wrapped in the chat template. Output streams to
// stdout; timings go to stderr.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/model.hpp"
#include "tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

// Length of the longest prefix of s that does not end inside a UTF-8 sequence.
std::size_t complete_utf8(const std::string& s) {
  std::size_t i = s.size(), back = 0;
  while (i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80)
    --i, ++back;
  if (i == 0)
    return s.size();
  const auto lead = static_cast<unsigned char>(s[i - 1]);
  const std::size_t need = lead < 0x80         ? 1
                           : (lead >> 5) == 6  ? 2
                           : (lead >> 4) == 14 ? 3
                           : (lead >> 3) == 30 ? 4
                                               : 1;
  return back + 1 >= need ? s.size() : i - 1;
}

int usage() {
  std::fprintf(stderr,
               "usage: miso [-m model.gguf] [-n max_new_tokens] [--no-think] [--raw] \"prompt\"\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = MISO_ORNITH_GGUF, prompt;
  unsigned max_new = 256;
  bool think = true, raw = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-m" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (a == "-n" && i + 1 < argc) {
      max_new = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    } else if (a == "--no-think") {
      think = false;
    } else if (a == "--raw") {
      raw = true;
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      prompt = a;
    }
  }
  if (prompt.empty() || max_new == 0)
    return usage();

  const auto t_load = Clock::now();
  const auto file = miso::gguf::File::open(model_path);
  const auto tok = miso::Tokenizer::from_gguf(file);
  const auto model = miso::Model::load(file);
  const std::vector<int> ids = tok.encode(raw ? prompt : miso::chat_prompt(prompt, think));
  std::fprintf(stderr, "[miso] loaded in %.1f s; prompt %zu tokens\n", seconds_since(t_load),
               ids.size());

  miso::Session s(model, static_cast<unsigned>(ids.size()) + max_new + 1);
  MISO_HIP_CHECK(
      hipMemcpy(s.tokens.data(), ids.data(), ids.size() * sizeof(int), hipMemcpyHostToDevice));
  int* host_pred = nullptr;
  MISO_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_pred), sizeof(int) * (max_new + 1)));
  std::vector<hipEvent_t> done(max_new + 1);
  for (auto& e : done)
    MISO_HIP_CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming));

  // Step k (k = 0 is the last prompt position) produces generated token k into pred[n + k - 1].
  const unsigned n = static_cast<unsigned>(ids.size());
  auto enqueue = [&](unsigned k) {
    const unsigned pos = s.pos;
    miso::decode_step(model, s, k == 0 ? s.tokens.data() + pos : s.pred.data() + pos - 1, nullptr);
    MISO_HIP_CHECK(hipMemcpyAsync(host_pred + k, s.pred.data() + pos, sizeof(int),
                                  hipMemcpyDeviceToHost, nullptr));
    MISO_HIP_CHECK(hipEventRecord(done[k], nullptr));
  };

  const auto t_prefill = Clock::now();
  miso::prefill(model, s, s.tokens.data(), n - 1, miso::Session::kMaxChunk, nullptr);
  enqueue(0);
  MISO_HIP_CHECK(hipEventSynchronize(done[0]));
  const double prefill_s = seconds_since(t_prefill);

  const auto t_gen = Clock::now();
  std::string pending;
  unsigned generated = 0;
  for (unsigned k = 0; k < max_new; ++k) {
    if (k + 1 < max_new)
      enqueue(k + 1);  // keep one step in flight while k is printed
    MISO_HIP_CHECK(hipEventSynchronize(done[k]));
    const int id = host_pred[k];
    ++generated;
    if (id == tok.eos_id() || id == 248044)
      break;  // <|im_end|>, <|endoftext|>
    pending += tok.token_bytes(id);
    const std::size_t ok = complete_utf8(pending);
    std::fwrite(pending.data(), 1, ok, stdout);
    std::fflush(stdout);
    pending.erase(0, ok);
  }
  MISO_HIP_CHECK(hipDeviceSynchronize());
  std::fwrite(pending.data(), 1, pending.size(), stdout);
  std::printf("\n");
  const double gen_s = seconds_since(t_gen);
  std::fprintf(
      stderr,
      "[miso] prefill %u tokens: %.2f s (%.1f tok/s); generated %u tokens: %.2f s (%.1f tok/s)\n",
      n, prefill_s, n / prefill_s, generated, gen_s, generated / gen_s);
  MISO_HIP_CHECK(hipHostFree(host_pred));
  return 0;
}
