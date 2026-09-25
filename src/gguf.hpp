#pragma once

// Read-only GGUF v3 reader over a memory-mapped file.

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace miso::gguf {

enum class GgmlType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
};

struct TypeTraits {
  const char* name;
  std::uint32_t block_elements;
  std::uint32_t block_bytes;
};

// Throws for types without a known block layout.
const TypeTraits& traits(GgmlType type);

// Integers are widened to int64, floats to double; arrays keep their element kind.
using Value = std::variant<std::int64_t, double, bool, std::string, std::vector<std::int64_t>,
                           std::vector<double>, std::vector<bool>, std::vector<std::string>>;

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;  // GGUF order: dims[0] is the contiguous dimension
  GgmlType type;
  std::uint64_t offset;  // relative to File::data_offset()

  std::uint64_t n_elements() const;
  std::uint64_t n_bytes() const;
};

class File {
 public:
  static File open(const std::string& path);

  File(File&& other) noexcept;
  File& operator=(File&&) = delete;
  File(const File&) = delete;
  ~File();

  std::uint32_t version() const { return version_; }
  std::uint32_t alignment() const { return alignment_; }
  std::uint64_t data_offset() const { return data_offset_; }

  const std::map<std::string, Value>& metadata() const { return metadata_; }
  const Value* find(const std::string& key) const;
  template <class T>
  const T& get(const std::string& key) const;

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find_tensor(const std::string& name) const;
  std::span<const std::byte> tensor_data(const TensorInfo& t) const;

 private:
  File() = default;
  void parse(const std::string& path);

  const std::byte* map_ = nullptr;
  std::size_t size_ = 0;
  std::uint32_t version_ = 0;
  std::uint32_t alignment_ = 32;
  std::uint64_t data_offset_ = 0;
  std::map<std::string, Value> metadata_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t> tensor_index_;
};

template <class T>
const T& File::get(const std::string& key) const {
  const Value* v = find(key);
  if (v == nullptr)
    throw std::runtime_error("GGUF key not found: " + key);
  const T* p = std::get_if<T>(v);
  if (p == nullptr)
    throw std::runtime_error("GGUF key has a different type: " + key);
  return *p;
}

}  // namespace miso::gguf
