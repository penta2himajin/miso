#include "engine/weights.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>
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

// Engine-layout byte size of one tensor.
std::size_t engine_bytes(const gguf::TensorInfo& t) {
  const unsigned k = static_cast<unsigned>(t.dims[0]);
  const std::size_t n = std::size_t{t.n_elements()} / k;
  switch (t.type) {
    case gguf::GgmlType::Q4_K:
      return t.n_bytes();
    case gguf::GgmlType::Q6_K:
      return n * q6k_r1_row_bytes(k);
    case gguf::GgmlType::F32:
      return t.n_bytes();
    default:
      throw std::runtime_error(std::string("unsupported tensor type ") + gguf::traits(t.type).name);
  }
}

// Repack `t` into `dst`, which must be exactly `engine_bytes(t)` long.
void repack_into(const gguf::File& f, const gguf::TensorInfo& t, std::span<std::byte> dst) {
  const auto src = f.tensor_data(t);
  switch (t.type) {
    case gguf::GgmlType::Q4_K:
      repack_q4k(src, dst.subspan(0, src.size()));
      return;
    case gguf::GgmlType::Q6_K:
      repack_q6k(src, static_cast<unsigned>(t.dims[0]), dst);
      return;
    case gguf::GgmlType::F32:
      std::memcpy(dst.data(), src.data(), src.size());
      return;
    default:
      throw std::runtime_error(std::string("unsupported tensor type ") + gguf::traits(t.type).name);
  }
}

// Repack rows [r0, r1) of `t`. Both layouts are row-contiguous at a fixed stride, so a row range is
// a valid sub-problem; this is what lets one tensor's repack be split across threads (M7f).
void repack_rows(const gguf::File& f, const gguf::TensorInfo& t, std::size_t dst_row,
                 std::size_t r0, std::size_t r1, std::byte* dst) {
  const auto src = f.tensor_data(t);
  const unsigned k = static_cast<unsigned>(t.dims[0]);
  const std::size_t nb = t.n_elements() / k;
  const std::size_t src_row = src.size() / nb;
  const auto ssrc = src.subspan(r0 * src_row, (r1 - r0) * src_row);
  const auto sdst = std::span<std::byte>(dst + r0 * dst_row, (r1 - r0) * dst_row);
  switch (t.type) {
    case gguf::GgmlType::Q4_K:
      repack_q4k(ssrc, sdst);
      return;
    case gguf::GgmlType::Q6_K:
      repack_q6k(ssrc, k, sdst);
      return;
    case gguf::GgmlType::F32:
      std::memcpy(sdst.data(), ssrc.data(), ssrc.size());
      return;
    default:
      throw std::runtime_error(std::string("unsupported tensor type ") + gguf::traits(t.type).name);
  }
}

// Engine-layout row size of one tensor.
std::size_t engine_row(const gguf::TensorInfo& t) {
  const unsigned k = static_cast<unsigned>(t.dims[0]);
  const std::size_t n = t.n_elements() / k;
  return engine_bytes(t) / n;
}

constexpr int kBlock = 256;
constexpr unsigned kGrid = 240;

}  // namespace

// Stage one tensor as a small pipeline: chunk c is repacked on a worker while this thread enqueues
// the copy of chunk c-1. H2D runs at ~3.05 GB/s and repack at ~1.4 GB/s on this host, so
// overlapping the two is what takes the load from serial (repack then copy) to a pipeline. Chunk c
// covers rows [nb*c/chunks, nb*(c+1)/chunks), the split `repack_rows` takes.
void load_split(const gguf::File& f, const gguf::TensorInfo& t, std::byte* dst, LoadStage* st) {
  constexpr int kChunks = 4;
  const std::size_t dst_row = engine_row(t);
  const std::size_t nb = t.n_elements() / static_cast<unsigned>(t.dims[0]);
  const int chunks = nb >= static_cast<std::size_t>(kChunks) ? kChunks : 1;
  auto slot = st->acquire(engine_bytes(t));
  std::size_t prev0 = 0, prev1 = nb;  // rows of the chunk whose copy is pending
  for (int c = 0; c < chunks; ++c) {
    const std::size_t r0 = nb * static_cast<std::size_t>(c) / static_cast<std::size_t>(chunks);
    const std::size_t r1 = nb * static_cast<std::size_t>(c + 1) / static_cast<std::size_t>(chunks);
    std::thread work([&, r0, r1] { repack_rows(f, t, dst_row, r0, r1, slot.data()); });
    if (c > 0)  // chunk c-1 is repacked; its copy overlaps this chunk's repack
      st->release(slot.data() + prev0 * dst_row, dst + prev0 * dst_row, (prev1 - prev0) * dst_row);
    work.join();
    prev0 = r0, prev1 = r1;
  }
  st->release(slot.data() + prev0 * dst_row, dst + prev0 * dst_row, (prev1 - prev0) * dst_row);
}

LoadStage::LoadStage() {
  MISO_HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  grow_to(64u << 20);
}

void LoadStage::grow_to(std::size_t n) {
  for (int i = 0; i < kSlots; ++i)
    if (slot_[i] != nullptr) {
      MISO_HIP_CHECK(hipEventSynchronize(empty_[i]));
      MISO_HIP_CHECK(hipHostFree(slot_[i]));
      slot_[i] = nullptr;
    }
  slot_bytes_ = n;
  for (int i = 0; i < kSlots; ++i) {
    MISO_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&slot_[i]), slot_bytes_, 0));
    if (empty_[i] == nullptr)
      MISO_HIP_CHECK(hipEventCreateWithFlags(&empty_[i], hipEventDisableTiming));
    MISO_HIP_CHECK(hipEventRecord(empty_[i], stream_));  // free from the start
  }
}

LoadStage::~LoadStage() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
    for (int i = 0; i < kSlots; ++i) {
      (void)hipEventDestroy(empty_[i]);
      (void)hipHostFree(slot_[i]);
    }
    (void)hipStreamDestroy(stream_);
  }
}

std::span<std::byte> LoadStage::acquire(std::size_t n) {
  if (n > slot_bytes_)
    grow_to(n);
  const int i = next_;
  next_ = (next_ + 1) % kSlots;
  MISO_HIP_CHECK(hipEventSynchronize(empty_[i]));
  cur_ = i;
  return {reinterpret_cast<std::byte*>(slot_[i]), slot_bytes_};
}

void LoadStage::release(const std::byte* src, void* dst, std::size_t n) {
  MISO_HIP_CHECK(hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, stream_));
  MISO_HIP_CHECK(hipEventRecord(empty_[cur_], stream_));
}

void LoadStage::sync() {
  MISO_HIP_CHECK(hipStreamSynchronize(stream_));
}

// Rows of same-type, same-K matrices, concatenated into one QMatrix (shared by DeltaNet and
// attention, whose gate/alpha/beta and k/v projections are split into separate GGUF tensors).
QMatrix concat_rows(const gguf::File& f, const std::vector<std::string>& names, LoadStage* st) {
  gguf::GgmlType type = gguf::GgmlType::F32;
  unsigned k = 0, n_total = 0;
  std::vector<std::size_t> want;
  std::vector<const gguf::TensorInfo*> ts;
  bool first = true;
  for (const auto& name : names) {
    const auto& t = tensor(f, name);
    const unsigned kt = static_cast<unsigned>(t.dims[0]);
    if (first) {
      type = t.type, k = kt, first = false;
    } else if (t.type != type || kt != k) {
      throw std::runtime_error("concat_rows: mixed type or K in " + name);
    }
    if (t.type != gguf::GgmlType::Q4_K && t.type != gguf::GgmlType::Q6_K)
      throw std::runtime_error(std::string("concat_rows: cannot concatenate ") +
                               gguf::traits(t.type).name);
    want.push_back(engine_bytes(t));
    n_total += static_cast<unsigned>(t.n_elements() / kt);
    ts.push_back(&t);
  }
  std::size_t total = 0;
  for (std::size_t w : want)
    total += w;
  DeviceBuffer<std::uint8_t> out(total);
  if (st == nullptr) {
    std::vector<std::byte> dst(total);
    std::size_t at = 0;
    for (std::size_t i = 0; i < ts.size(); ++i) {
      repack_into(f, *ts[i], std::span<std::byte>(dst).subspan(at, want[i]));
      at += want[i];
    }
    MISO_HIP_CHECK(hipMemcpy(out.data(), dst.data(), total, hipMemcpyHostToDevice));
  } else {
    std::size_t at = 0;
    for (std::size_t i = 0; i < ts.size(); ++i) {
      load_split(f, *ts[i], reinterpret_cast<std::byte*>(out.data()) + at, st);
      at += want[i];
    }
  }
  return {std::move(out), type, n_total, k};
}

QMatrix upload_qmatrix(const gguf::File& f, const std::string& name, LoadStage* st) {
  const auto& t = tensor(f, name);
  const unsigned k = static_cast<unsigned>(t.dims[0]);
  const unsigned n = static_cast<unsigned>(t.n_elements() / k);
  const std::size_t want = engine_bytes(t);
  DeviceBuffer<std::uint8_t> out(want);
  if (st == nullptr) {
    std::vector<std::byte> dst(want);
    repack_into(f, t, dst);
    MISO_HIP_CHECK(hipMemcpy(out.data(), dst.data(), want, hipMemcpyHostToDevice));
  } else {
    load_split(f, t, reinterpret_cast<std::byte*>(out.data()), st);
  }
  return {std::move(out), t.type, n, k};
}

DeviceBuffer<float> upload_f32(const gguf::File& f, const std::string& name, LoadStage* st) {
  const auto& t = tensor(f, name);
  if (t.type != gguf::GgmlType::F32)
    throw std::runtime_error(name + " is not F32");
  DeviceBuffer<float> d(t.n_elements());
  const auto src = f.tensor_data(t);
  if (st == nullptr) {
    MISO_HIP_CHECK(hipMemcpy(d.data(), src.data(), src.size(), hipMemcpyHostToDevice));
    return d;
  }
  auto slot = st->acquire(src.size());
  std::memcpy(slot.data(), src.data(), src.size());
  st->release(slot.data(), d.data(), src.size());
  return d;
}

void gemv(const QMatrix& w, const float* x, float* y, hipStream_t stream, const _Float16* x_h) {
  if (w.n == 0)
    return;
  using kernels::ActFormat;
  // Grid and rows-per-wave from decode-sized microbenchmarks (bench/gemv): K = 4096 N = 2048 wants
  // R = 4 at 240 workgroups (18.5 vs 21.9 µs at R = 1); K = 2048 with N >= 2048 likewise prefers
  // R = 4; Q6_K qkv (N = 8192) prefers grid 480. Small N stays on a leaner grid so waves are not
  // mostly idle.
  if (w.type == gguf::GgmlType::Q4_K) {
    const kernels::Q4kGemvParams p{w.data.data(), x, y, w.n, x_h};
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
    const kernels::Q6kGemvParams p{w.data.data(), x, y, w.n, x_h};
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
