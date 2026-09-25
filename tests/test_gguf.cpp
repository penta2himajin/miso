#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dequant.hpp"
#include "gguf.hpp"

using miso::gguf::File;
using miso::gguf::GgmlType;

namespace {

std::string fixture(const std::string& name) {
  return std::string(MISO_FIXTURE_DIR) + "/" + name;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

}  // namespace

TEST_CASE("GGUF reader parses every metadata value type") {
  const File f = File::open(fixture("tiny.gguf"));
  CHECK(f.version() == 3);
  CHECK(f.alignment() == 64);
  CHECK(f.get<std::string>("general.architecture") == "miso-test");
  CHECK(f.get<std::int64_t>("test.u8") == 200);
  CHECK(f.get<std::int64_t>("test.i8") == -100);
  CHECK(f.get<std::int64_t>("test.u16") == 60000);
  CHECK(f.get<std::int64_t>("test.i16") == -30000);
  CHECK(f.get<std::int64_t>("test.u32") == 4000000000LL);
  CHECK(f.get<std::int64_t>("test.i32") == -2000000000LL);
  CHECK(f.get<std::int64_t>("test.u64") == (1LL << 40));
  CHECK(f.get<std::int64_t>("test.i64") == -(1LL << 40));
  CHECK(f.get<double>("test.f32") == 0.25);
  CHECK(f.get<double>("test.f64") == -1.5);
  CHECK(f.get<bool>("test.bool"));
  CHECK(f.get<std::string>("test.string") == "miso ミソ");
  CHECK(f.get<std::vector<std::int64_t>>("test.array_i32") == std::vector<std::int64_t>{1, -2, 3});
  CHECK(f.get<std::vector<std::string>>("test.array_str") ==
        std::vector<std::string>{"a", "bc", ""});
  CHECK(f.find("test.missing") == nullptr);
  CHECK_THROWS_WITH(f.get<std::int64_t>("test.missing"), doctest::Contains("test.missing"));
  CHECK_THROWS_WITH(f.get<std::string>("test.u8"), doctest::Contains("test.u8"));
}

TEST_CASE("GGUF reader returns tensor infos at aligned offsets") {
  const File f = File::open(fixture("tiny.gguf"));
  REQUIRE(f.tensors().size() == 5);

  struct Want {
    const char* name;
    GgmlType type;
    std::vector<std::uint64_t> dims;
    std::uint64_t abs_offset;
  };
  const Want want[] = {
      {"f32", GgmlType::F32, {3, 5}, 768},        {"f16", GgmlType::F16, {256}, 832},
      {"bf16", GgmlType::BF16, {64}, 1344},       {"q4_k", GgmlType::Q4_K, {8192, 2}, 1472},
      {"q6_k", GgmlType::Q6_K, {4096, 4}, 10688},
  };
  for (const auto& w : want) {
    CAPTURE(w.name);
    const auto* t = f.find_tensor(w.name);
    REQUIRE(t != nullptr);
    CHECK(t->type == w.type);
    CHECK(t->dims == w.dims);
    CHECK(f.data_offset() + t->offset == w.abs_offset);
    CHECK((f.data_offset() + t->offset) % 64 == 0);
    CHECK(f.tensor_data(*t).size() == t->n_bytes());
  }
  CHECK(f.find_tensor("nope") == nullptr);
}

TEST_CASE("CPU dequantisation matches gguf-py bit-exactly") {
  const File f = File::open(fixture("tiny.gguf"));
  for (const auto& t : f.tensors()) {
    CAPTURE(t.name);
    const auto ref = read_file(fixture("tiny." + t.name + ".f32"));
    REQUIRE(ref.size() == t.n_elements() * sizeof(float));

    std::vector<float> got(t.n_elements());
    miso::dequantize(t.type, f.tensor_data(t), got);
    CHECK(std::memcmp(got.data(), ref.data(), ref.size()) == 0);
  }
}

TEST_CASE("dequantize rejects a source span of the wrong size") {
  const File f = File::open(fixture("tiny.gguf"));
  const auto* t = f.find_tensor("q4_k");
  REQUIRE(t != nullptr);
  std::vector<float> out(t->n_elements());
  CHECK_THROWS(miso::dequantize(t->type, f.tensor_data(*t).first(100), out));
}

TEST_CASE("GGUF reader rejects malformed files") {
  const auto dir = std::filesystem::temp_directory_path();
  const auto bytes = read_file(fixture("tiny.gguf"));

  const auto bad_magic = (dir / "miso-bad-magic.gguf").string();
  {
    std::ofstream o(bad_magic, std::ios::binary);
    o.write("GGUX", 4);
    o.write(bytes.data() + 4, static_cast<std::streamsize>(bytes.size() - 4));
  }
  CHECK_THROWS_WITH(File::open(bad_magic), doctest::Contains("magic"));

  const auto truncated = (dir / "miso-truncated.gguf").string();
  {
    std::ofstream o(truncated, std::ios::binary);
    o.write(bytes.data(), 200);
  }
  CHECK_THROWS_WITH(File::open(truncated), doctest::Contains("truncated"));

  CHECK_THROWS(File::open((dir / "miso-does-not-exist.gguf").string()));
}

TEST_CASE("Ornith Q4_K_M tensor table matches gguf-py" * doctest::skip(!model_present())) {
  const File f = File::open(MISO_ORNITH_GGUF);
  CHECK(f.get<std::string>("general.architecture") == "qwen35moe");

  std::ifstream manifest(fixture("ornith-q4km-tensors.tsv"));
  std::string line;
  std::size_t rows = 0;
  while (std::getline(manifest, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream row(line);
    std::string name, type, dims, offset, bytes;
    std::getline(row, name, '\t');
    std::getline(row, type, '\t');
    std::getline(row, dims, '\t');
    std::getline(row, offset, '\t');
    std::getline(row, bytes, '\t');
    CAPTURE(name);
    const auto* t = f.find_tensor(name);
    REQUIRE(t != nullptr);
    CHECK(static_cast<unsigned>(t->type) == std::stoul(type));
    std::string got_dims;
    for (auto d : t->dims)
      got_dims += (got_dims.empty() ? "" : ",") + std::to_string(d);
    CHECK(got_dims == dims);
    CHECK(f.data_offset() + t->offset == std::stoull(offset));
    CHECK(t->n_bytes() == std::stoull(bytes));
    ++rows;
  }
  CHECK(rows == 753);
  CHECK(f.tensors().size() == 753);
}
