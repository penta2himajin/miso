#pragma once

// Device-resident weights, repacked at load time (ADR_001 D5), and the decode GEMV dispatch.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

// Load-time H2D staging (M7f). Repacked bytes go into pinned buffers and are copied on a dedicated
// stream, so the device read never stalls on a cold pageable buffer. Measured over this model's
// 21.8 GB: 28.3 s copying from fresh pageable buffers, 13.6 s from one reused pageable buffer,
// 7.5 s through this stage. `kSlots` buffers rotate, each gated by its own event, so repacking the
// next tensor overlaps the copy of the previous one.
class LoadStage {
 public:
  // Slots grow to the largest tensor (417 MB here), so 4 of them cap pinned host memory at ~1.7 GB.
  static constexpr int kSlots = 4;

  LoadStage();
  ~LoadStage();
  LoadStage(const LoadStage&) = delete;
  LoadStage& operator=(const LoadStage&) = delete;

  // A writable pinned buffer of at least `n` bytes. Grows the slots once if needed, then blocks on
  // the event of the buffer being reused.
  std::span<std::byte> acquire(std::size_t n);
  // Enqueues the copy of [src, src + n) — a pointer inside the last acquired buffer — into `dst`.
  // The buffer frees when the last enqueued copy lands.
  void release(const std::byte* src, void* dst, std::size_t n);
  // Waits for every enqueued copy. Must run before the device buffers are used.
  void sync();

 private:
  void grow_to(std::size_t n);

  std::uint8_t* slot_[kSlots] = {};
  hipEvent_t empty_[kSlots] = {};
  hipStream_t stream_ = nullptr;
  std::size_t slot_bytes_ = 0;
  int next_ = 0, cur_ = -1;
};

// A quantised matrix of n rows by k columns in its engine layout (Q4_K r1 or Q6_K r1).
struct QMatrix {
  DeviceBuffer<std::uint8_t> data;
  gguf::GgmlType type;
  unsigned n, k;
};

// Rows of same-type, same-K matrices, concatenated into one QMatrix (one GEMV covers them).
QMatrix concat_rows(const gguf::File& f, const std::vector<std::string>& names,
                    LoadStage* st = nullptr);

QMatrix upload_qmatrix(const gguf::File& f, const std::string& name, LoadStage* st = nullptr);
DeviceBuffer<float> upload_f32(const gguf::File& f, const std::string& name,
                               LoadStage* st = nullptr);

// One tensor through the stage, repacked in two halves so the second half's repack overlaps the
// first half's copy. `dst` must hold the tensor's engine-layout size.
void load_split(const gguf::File& f, const gguf::TensorInfo& t, std::byte* dst, LoadStage* st);

// y = W x for one token (decode GEMV family, FP16 activations per ADR_004).
void gemv(const QMatrix& w, const float* x, float* y, hipStream_t stream,
          const _Float16* x_h = nullptr);

// Activations of a token chunk for the prefill GEMM family: FP16-rounded values and per-32 sums.
struct GemmInput {
  GemmInput(unsigned max_tok, unsigned max_k)
      : xh(std::size_t{max_tok} * max_k), xs(std::size_t{max_tok} * max_k / 32) {}
  DeviceBuffer<_Float16> xh;
  DeviceBuffer<float> xs;
  unsigned k = 0, n_tok = 0;
};

// Prepares x ([n_tok][k] floats) as GEMM input.
void gemm_input(const float* x, unsigned k, unsigned n_tok, GemmInput& in, hipStream_t stream);

// Y[t][n] = W x[t] for the prepared chunk; row t of Y starts at y + t * ldy.
void gemm(const QMatrix& w, const GemmInput& in, float* y, unsigned ldy, hipStream_t stream);

}  // namespace miso
