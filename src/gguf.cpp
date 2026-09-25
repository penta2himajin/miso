#include "gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace miso::gguf {

namespace {

enum ValueType : std::uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};

class Cursor {
 public:
  Cursor(const std::byte* p, std::size_t size) : p_(p), size_(size) {}

  template <class T>
  T read() {
    T v;
    std::memcpy(&v, take(sizeof(T)), sizeof(T));
    return v;
  }

  std::string string() {
    const auto n = read<std::uint64_t>();
    const auto* s = take(n);
    return {reinterpret_cast<const char*>(s), n};
  }

  std::size_t pos() const { return pos_; }

 private:
  const std::byte* take(std::uint64_t n) {
    if (n > size_ - pos_) {
      throw std::runtime_error("truncated GGUF: need " + std::to_string(n) + " bytes at offset " +
                               std::to_string(pos_) + " of " + std::to_string(size_));
    }
    const auto* p = p_ + pos_;
    pos_ += n;
    return p;
  }

  const std::byte* p_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

std::int64_t read_int(Cursor& c, std::uint32_t t) {
  switch (t) {
    case kU8:
      return c.read<std::uint8_t>();
    case kI8:
      return c.read<std::int8_t>();
    case kU16:
      return c.read<std::uint16_t>();
    case kI16:
      return c.read<std::int16_t>();
    case kU32:
      return c.read<std::uint32_t>();
    case kI32:
      return c.read<std::int32_t>();
    case kU64:
      return static_cast<std::int64_t>(c.read<std::uint64_t>());
    case kI64:
      return c.read<std::int64_t>();
  }
  throw std::runtime_error("GGUF: not an integer value type " + std::to_string(t));
}

bool is_int(std::uint32_t t) {
  return t <= kI32 || t == kU64 || t == kI64;
}

double read_float(Cursor& c, std::uint32_t t) {
  return t == kF32 ? static_cast<double>(c.read<float>()) : c.read<double>();
}

template <class T, class F>
std::vector<T> read_n(std::uint64_t n, F&& f) {
  std::vector<T> v;
  v.reserve(n);
  for (std::uint64_t i = 0; i < n; ++i)
    v.push_back(f());
  return v;
}

Value read_value(Cursor& c, std::uint32_t t) {
  if (is_int(t))
    return read_int(c, t);
  switch (t) {
    case kF32:
    case kF64:
      return read_float(c, t);
    case kBool:
      return c.read<std::uint8_t>() != 0;
    case kString:
      return c.string();
    case kArray: {
      const auto et = c.read<std::uint32_t>();
      const auto n = c.read<std::uint64_t>();
      if (is_int(et))
        return read_n<std::int64_t>(n, [&] { return read_int(c, et); });
      if (et == kF32 || et == kF64)
        return read_n<double>(n, [&] { return read_float(c, et); });
      if (et == kBool)
        return read_n<bool>(n, [&] { return c.read<std::uint8_t>() != 0; });
      if (et == kString)
        return read_n<std::string>(n, [&] { return c.string(); });
      throw std::runtime_error("GGUF: unsupported array element type " + std::to_string(et));
    }
  }
  throw std::runtime_error("GGUF: unknown value type " + std::to_string(t));
}

}  // namespace

const TypeTraits& traits(GgmlType type) {
  static const std::map<GgmlType, TypeTraits> table = {
      {GgmlType::F32, {"F32", 1, 4}},       {GgmlType::F16, {"F16", 1, 2}},
      {GgmlType::Q4_0, {"Q4_0", 32, 18}},   {GgmlType::Q4_1, {"Q4_1", 32, 20}},
      {GgmlType::Q5_0, {"Q5_0", 32, 22}},   {GgmlType::Q5_1, {"Q5_1", 32, 24}},
      {GgmlType::Q8_0, {"Q8_0", 32, 34}},   {GgmlType::Q2_K, {"Q2_K", 256, 84}},
      {GgmlType::Q3_K, {"Q3_K", 256, 110}}, {GgmlType::Q4_K, {"Q4_K", 256, 144}},
      {GgmlType::Q5_K, {"Q5_K", 256, 176}}, {GgmlType::Q6_K, {"Q6_K", 256, 210}},
      {GgmlType::Q8_K, {"Q8_K", 256, 292}}, {GgmlType::BF16, {"BF16", 1, 2}},
      {GgmlType::I8, {"I8", 1, 1}},         {GgmlType::I16, {"I16", 1, 2}},
      {GgmlType::I32, {"I32", 1, 4}},       {GgmlType::I64, {"I64", 1, 8}},
      {GgmlType::F64, {"F64", 1, 8}},
  };
  const auto it = table.find(type);
  if (it == table.end()) {
    throw std::runtime_error("unsupported ggml type " +
                             std::to_string(static_cast<std::uint32_t>(type)));
  }
  return it->second;
}

std::uint64_t TensorInfo::n_elements() const {
  std::uint64_t n = 1;
  for (auto d : dims)
    n *= d;
  return n;
}

std::uint64_t TensorInfo::n_bytes() const {
  const auto& tr = traits(type);
  return n_elements() / tr.block_elements * tr.block_bytes;
}

File File::open(const std::string& path) {
  File f;
  f.parse(path);
  return f;
}

File::File(File&& o) noexcept
    : map_(o.map_),
      size_(o.size_),
      version_(o.version_),
      alignment_(o.alignment_),
      data_offset_(o.data_offset_),
      metadata_(std::move(o.metadata_)),
      tensors_(std::move(o.tensors_)),
      tensor_index_(std::move(o.tensor_index_)) {
  o.map_ = nullptr;
  o.size_ = 0;
}

File::~File() {
  if (map_ != nullptr)
    munmap(const_cast<std::byte*>(map_), size_);
}

void File::parse(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size == 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat or empty file: " + path);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  void* m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED)
    throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(errno));
  map_ = static_cast<const std::byte*>(m);

  Cursor c(map_, size_);
  if (c.read<std::uint32_t>() != 0x46554747u)
    throw std::runtime_error("bad GGUF magic: " + path);
  version_ = c.read<std::uint32_t>();
  if (version_ != 3)
    throw std::runtime_error("unsupported GGUF version " + std::to_string(version_));
  const auto n_tensors = c.read<std::uint64_t>();
  const auto n_kv = c.read<std::uint64_t>();

  for (std::uint64_t i = 0; i < n_kv; ++i) {
    auto key = c.string();
    const auto t = c.read<std::uint32_t>();
    metadata_.emplace(std::move(key), read_value(c, t));
  }
  if (find("general.alignment") != nullptr) {
    alignment_ = static_cast<std::uint32_t>(get<std::int64_t>("general.alignment"));
    if (alignment_ == 0)
      throw std::runtime_error("GGUF general.alignment is 0");
  }

  tensors_.reserve(n_tensors);
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo t;
    t.name = c.string();
    const auto n_dims = c.read<std::uint32_t>();
    for (std::uint32_t d = 0; d < n_dims; ++d)
      t.dims.push_back(c.read<std::uint64_t>());
    t.type = static_cast<GgmlType>(c.read<std::uint32_t>());
    t.offset = c.read<std::uint64_t>();
    tensor_index_.emplace(t.name, tensors_.size());
    tensors_.push_back(std::move(t));
  }

  data_offset_ = (c.pos() + alignment_ - 1) / alignment_ * alignment_;
  for (const auto& t : tensors_) {
    if (t.offset % alignment_ != 0)
      throw std::runtime_error("misaligned tensor " + t.name);
    if (data_offset_ + t.offset + t.n_bytes() > size_) {
      throw std::runtime_error("truncated GGUF: tensor " + t.name + " extends past end of file");
    }
  }
}

const Value* File::find(const std::string& key) const {
  const auto it = metadata_.find(key);
  return it == metadata_.end() ? nullptr : &it->second;
}

const TensorInfo* File::find_tensor(const std::string& name) const {
  const auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::span<const std::byte> File::tensor_data(const TensorInfo& t) const {
  return {map_ + data_offset_ + t.offset, t.n_bytes()};
}

}  // namespace miso::gguf
