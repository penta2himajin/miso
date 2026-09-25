#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace miso {

inline void hip_check(hipError_t err, const char* expr) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(expr) + " failed: " + hipGetErrorName(err) + " (" +
                             hipGetErrorString(err) + ")");
  }
}

// Owning device allocation of `size()` elements of T.
template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t n) : n_(n) {
    hip_check(hipMalloc(&p_, n * sizeof(T)), "hipMalloc");
  }
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    hip_check(hipMemcpy(p_, host.data(), n_ * sizeof(T), hipMemcpyHostToDevice), "hipMemcpy H2D");
  }
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {
    o.p_ = nullptr;
    o.n_ = 0;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;
  ~DeviceBuffer() {
    if (p_ != nullptr)
      (void)hipFree(p_);
  }

  T* data() const { return p_; }
  std::size_t size() const { return n_; }

  std::vector<T> to_host() const {
    std::vector<T> host(n_);
    hip_check(hipMemcpy(host.data(), p_, n_ * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy D2H");
    return host;
  }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

}  // namespace miso

#define MISO_HIP_CHECK(expr) ::miso::hip_check((expr), #expr)
