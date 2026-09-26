#include "engine/weights.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "q4k_gemv.hpp"
#include "q6k_gemv.hpp"
#include "qgemm.hpp"
#include "repack.hpp"

namespace miso {

namespace {

const gguf::TensorInfo& tensor(const gguf::File& f, const std::string& name) {
  const auto* t = f.find_tensor(name);
  if (t == nullptr)
    throw std::runtime_error("missing tensor " + name);
  return *t;
}

DeviceBuffer<std::uint8_t> upload_bytes(const std::vector<std::byte>& b) {
  DeviceBuffer<std::uint8_t> d(b.size());
  MISO_HIP_CHECK(hipMemcpy(d.data(), b.data(), b.size(), hipMemcpyHostToDevice));
  return d;
}

constexpr int kBlock = 256;
constexpr unsigned kGrid = 240;

}  // namespace

// Rows of same-type, same-K matrices, concatenated into one QMatrix (shared by DeltaNet and
// attention, whose gate/alpha/beta and k/v projections are split into separate GGUF tensors).
QMatrix concat_rows(const gguf::File& f, const std::vector<std::string>& names) {
  gguf::GgmlType type = gguf::GgmlType::F32;
  unsigned k = 0, n_total = 0;
  std::vector<std::byte> dst;
  bool first = true;
  for (const auto& name : names) {
    const auto& t = tensor(f, name);
    const unsigned kt = static_cast<unsigned>(t.dims[0]);
    const unsigned nt = static_cast<unsigned>(t.n_elements() / kt);
    if (first) {
      type = t.type, k = kt, first = false;
    } else if (t.type != type || kt != k) {
      throw std::runtime_error("concat_rows: mixed type or K in " + name);
    }
    const auto src = f.tensor_data(t);
    if (t.type == gguf::GgmlType::Q4_K) {
      const std::size_t at = dst.size();
      dst.resize(at + src.size());
      repack_q4k(src, std::span<std::byte>(dst).subspan(at));
    } else if (t.type == gguf::GgmlType::Q6_K) {
      const std::size_t at = dst.size();
      dst.resize(at + std::size_t{nt} * q6k_r1_row_bytes(k));
      repack_q6k(src, k, std::span<std::byte>(dst).subspan(at));
    } else {
      throw std::runtime_error(name + ": cannot concatenate " + gguf::traits(t.type).name);
    }
    n_total += nt;
  }
  return {upload_bytes(dst), type, n_total, k};
}

QMatrix upload_qmatrix(const gguf::File& f, const std::string& name) {
  const auto& t = tensor(f, name);
  const auto src = f.tensor_data(t);
  const unsigned k = static_cast<unsigned>(t.dims[0]);
  const unsigned n = static_cast<unsigned>(t.n_elements() / k);
  std::vector<std::byte> dst;
  if (t.type == gguf::GgmlType::Q4_K) {
    dst.resize(src.size());
    repack_q4k(src, dst);
  } else if (t.type == gguf::GgmlType::Q6_K) {
    dst.resize(std::size_t{n} * q6k_r1_row_bytes(k));
    repack_q6k(src, k, dst);
  } else {
    throw std::runtime_error(name + ": unsupported matrix type " + gguf::traits(t.type).name);
  }
  return {upload_bytes(dst), t.type, n, k};
}

DeviceBuffer<float> upload_f32(const gguf::File& f, const std::string& name) {
  const auto& t = tensor(f, name);
  if (t.type != gguf::GgmlType::F32)
    throw std::runtime_error(name + " is not F32");
  const auto src = f.tensor_data(t);
  DeviceBuffer<float> d(t.n_elements());
  MISO_HIP_CHECK(hipMemcpy(d.data(), src.data(), src.size(), hipMemcpyHostToDevice));
  return d;
}

void gemv(const QMatrix& w, const float* x, float* y, hipStream_t stream) {
  if (w.n == 0)
    return;
  using kernels::ActFormat;
  // Grid and rows-per-wave from decode-sized microbenchmarks (bench/gemv): K = 4096 N = 2048 wants
  // R = 4 at 240 workgroups (18.5 vs 21.9 µs at R = 1); K = 2048 with N >= 2048 likewise prefers
  // R = 4; Q6_K qkv (N = 8192) prefers grid 480. Small N stays on a leaner grid so waves are not
  // mostly idle.
  if (w.type == gguf::GgmlType::Q4_K) {
    const kernels::Q4kGemvParams p{w.data.data(), x, y, w.n};
    if (w.k == 2048) {
      if (w.n >= 2048) {
        kernels::q4k_gemv_kernel<ActFormat::Fp16, 2048, kBlock, 4><<<kGrid, kBlock, 0, stream>>>(p);
      } else {
        const unsigned grid = std::min(kGrid, std::max(60u, (w.n + 3) / 4));
        kernels::q4k_gemv_kernel<ActFormat::Fp16, 2048, kBlock, 1><<<grid, kBlock, 0, stream>>>(p);
      }
    } else if (w.k == 4096) {
      kernels::q4k_gemv_kernel<ActFormat::Fp16, 4096, kBlock, 4><<<kGrid, kBlock, 0, stream>>>(p);
    } else {
      throw std::runtime_error("gemv: no Q4_K kernel for K = " + std::to_string(w.k));
    }
  } else if (w.type == gguf::GgmlType::Q6_K && w.k == 2048) {
    const kernels::Q6kGemvParams p{w.data.data(), x, y, w.n};
    // R = 2 at grid 480 edged out R = 1 for attn_qkv (27.4 vs 28.1 µs); LM head stays R = 1 at a
    // larger grid so occupancy is not cut by the extra row registers.
    if (w.n >= 8192 && w.n < 100000) {
      kernels::q6k_gemv_kernel<2048, kBlock, 2><<<480, kBlock, 0, stream>>>(p);
    } else {
      const unsigned grid = w.n >= 100000 ? 1920u : kGrid;
      kernels::q6k_gemv_kernel<2048, kBlock, 1><<<grid, kBlock, 0, stream>>>(p);
    }
  } else {
    throw std::runtime_error("gemv: no kernel for this type / K = " + std::to_string(w.k));
  }
  MISO_HIP_CHECK(hipGetLastError());
}

void gemm_input(const float* x, unsigned k, unsigned n_tok, GemmInput& in, hipStream_t stream) {
  if (std::size_t{n_tok} * k > in.xh.size() || k % 256 != 0)
    throw std::runtime_error("gemm_input: chunk does not fit");
  in.k = k, in.n_tok = n_tok;
  const kernels::QGemmActParams p{x, in.xh.data(), in.xs.data(), k, n_tok};
  const unsigned groups = n_tok * (k / 32);
  kernels::qgemm_act_kernel<<<(groups + kBlock - 1) / kBlock, kBlock, 0, stream>>>(p);
  MISO_HIP_CHECK(hipGetLastError());
}

void gemm(const QMatrix& w, const GemmInput& in, float* y, unsigned ldy, hipStream_t stream) {
  if (w.k != in.k)
    throw std::runtime_error("gemm: K mismatch");
  const kernels::QGemmParams p{w.data.data(), in.xh.data(), in.xs.data(), y,
                               w.k,           w.n,          in.n_tok,     ldy};
  const unsigned grid = kernels::qgemm_tiles(w.n, in.n_tok);
  if (w.type == gguf::GgmlType::Q4_K)
    kernels::qgemm_kernel<kernels::QFormat::Q4K><<<grid, kBlock, 0, stream>>>(p);
  else if (w.type == gguf::GgmlType::Q6_K)
    kernels::qgemm_kernel<kernels::QFormat::Q6K><<<grid, kBlock, 0, stream>>>(p);
  else
    throw std::runtime_error("gemm: unsupported type");
  MISO_HIP_CHECK(hipGetLastError());
}

}  // namespace miso
