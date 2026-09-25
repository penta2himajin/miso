#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "tokenizer.hpp"
#include "unicode.hpp"

using miso::gguf::File;
namespace uc = miso::unicode;

namespace {

std::string nfc8(const std::string& s) {
  return uc::to_utf8(uc::nfc(uc::from_utf8(s)));
}

bool model_present() {
  return std::filesystem::exists(MISO_ORNITH_GGUF);
}

const miso::Tokenizer& tokenizer() {
  static const File f = File::open(MISO_ORNITH_GGUF);
  static const miso::Tokenizer t = miso::Tokenizer::from_gguf(f);
  return t;
}

}  // namespace

TEST_CASE("UTF-8 round trip, with U+FFFD for invalid input") {
  const std::string s = "aé日👍";
  CHECK(uc::to_utf8(uc::from_utf8(s)) == s);
  CHECK(uc::from_utf8(s).size() == 4);
  CHECK(uc::from_utf8("\xff"
                      "a") == std::u32string{0xFFFD, U'a'});
  CHECK(uc::from_utf8("\xe6\x97") == std::u32string{0xFFFD});  // truncated sequence
}

TEST_CASE("general categories used by the pre-tokeniser") {
  CHECK(uc::is_letter(U'a'));
  CHECK(uc::is_letter(U'日'));
  CHECK(uc::is_mark(0x0301));
  CHECK(uc::is_mark(0x3099));
  CHECK(uc::is_number(U'7'));
  CHECK(uc::is_number(0xFF11));  // fullwidth 1 (Nd)
  CHECK(uc::is_number(0x216B));  // roman numeral twelve (Nl)
  CHECK(uc::is_number(0x00B2));  // superscript two (No)
  CHECK_FALSE(uc::is_letter(U' '));
  CHECK_FALSE(uc::is_number(U'a'));
  CHECK(uc::is_whitespace(0x3000));
  CHECK(uc::is_whitespace(0x00A0));
  CHECK_FALSE(uc::is_whitespace(0x200B));  // zero width space is not White_Space
}

TEST_CASE("NFC normalisation") {
  CHECK(nfc8("e\u0301") == "\u00e9");
  CHECK(nfc8("\u304b\u3099") == "\u304c");                // か + dakuten -> が
  CHECK(nfc8("\u1100\u1161\u11a8") == "\uac01");          // Hangul L V T -> 각
  CHECK(nfc8("\u212b") == "\u00c5");                      // angstrom sign (singleton) -> Å
  CHECK(nfc8("a\u0301\u0323") == nfc8("a\u0323\u0301"));  // canonical reordering
  CHECK(nfc8("\u1e0b\u0323") == "\u1e0d\u0307");          // ḋ + dot below -> ḍ + dot above
  CHECK(nfc8("ﬁ") == "ﬁ");                                // compatibility only: unchanged
  CHECK(nfc8("plain ascii") == "plain ascii");
}

TEST_CASE("tokenizer reproduces the Hugging Face tokenizer ids exactly" *
          doctest::skip(!model_present())) {
  const File fx = File::open(std::string(MISO_FIXTURE_DIR) + "/tokenizer_cases.gguf");
  const auto& texts = fx.get<std::vector<std::string>>("tok.texts");
  const auto& offsets = fx.get<std::vector<std::int64_t>>("tok.offsets");
  const auto* t = fx.find_tensor("tok.ids");
  REQUIRE(t != nullptr);
  const auto* all = reinterpret_cast<const std::int32_t*>(fx.tensor_data(*t).data());
  REQUIRE(offsets.size() == texts.size() + 1);

  std::size_t mismatches = 0;
  for (std::size_t c = 0; c < texts.size(); ++c) {
    const std::vector<int> want(all + offsets[c], all + offsets[c + 1]);
    const auto got = tokenizer().encode(texts[c]);
    if (got != want) {
      ++mismatches;
      if (mismatches <= 5)
        FAIL_CHECK("case " << c << " " << doctest::toString(texts[c].substr(0, 60)));
    }
    CHECK(tokenizer().decode(got) == nfc8(texts[c]));
  }
  CHECK(mismatches == 0);
  MESSAGE(texts.size() << " cases, " << offsets.back() << " ids");
}

TEST_CASE("tokenizer encodes the golden prompt to the golden ids" *
          doctest::skip(!model_present())) {
  const File g = File::open(std::string(MISO_GOLDEN_DIR) + "/ornith-layers.gguf");
  const auto& ids = g.get<std::vector<std::int64_t>>("golden.token_ids");
  const auto got = tokenizer().encode(g.get<std::string>("golden.prompt"));
  CHECK(std::vector<std::int64_t>(got.begin(), got.end()) == ids);
  CHECK(tokenizer().eos_id() == 248046);
  CHECK(tokenizer().encode("<|im_end|>") == std::vector<int>{248046});
}

TEST_CASE("single-turn chat prompt equals the HF chat template output") {
  const File fx = File::open(std::string(MISO_FIXTURE_DIR) + "/tokenizer_cases.gguf");
  const auto& msg = fx.get<std::string>("chat.message");
  CHECK(miso::chat_prompt(msg, /*thinking=*/true) == fx.get<std::string>("chat.think"));
  CHECK(miso::chat_prompt(msg, /*thinking=*/false) == fx.get<std::string>("chat.no_think"));
}
