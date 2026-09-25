#pragma once

// Device-resident weights, repacked at load time (ADR_001 D5), and the decode GEMV dispatch.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "gguf.hpp"
#include "hip_util.hpp"

namespace miso {

// A quantised matrix of n rows by k columns in its engine layout (Q4_K r1 or Q6_K r1).
struct QMatrix {
  DeviceBuffer<std::uint8_t> data;
  gguf::GgmlType type;
  unsigned n, k;
};

// Rows of same-type, same-K matrices, concatenated into one QMatrix (one GEMV covers them).
QMatrix concat_rows(const gguf::File& f, const std::vector<std::string>& names);

QMatrix upload_qmatrix(const gguf::File& f, const std::string& name);
DeviceBuffer<float> upload_f32(const gguf::File& f, const std::string& name);

// y = W x for one token (decode GEMV family, FP16 activations per ADR_004).
void gemv(const QMatrix& w, const float* x, float* y, hipStream_t stream);

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
