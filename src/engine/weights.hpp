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

QMatrix upload_qmatrix(const gguf::File& f, const std::string& name);
DeviceBuffer<float> upload_f32(const gguf::File& f, const std::string& name);

// y = W x for one token (decode GEMV family, FP16 activations per ADR_004).
void gemv(const QMatrix& w, const float* x, float* y, hipStream_t stream);

}  // namespace miso
