#include "tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "unicode.hpp"

namespace miso {

namespace {

namespace uc = unicode;

bool is_crlf(char32_t c) {
  return c == U'\r' || c == U'\n';
}
// The HF tokenizer's regex engine matches nothing for \p{M} in this pattern (measured: "हिन्दी"
// splits as ह | िन | ्द | ी), so marks are neither part of [\p{L}\p{M}]+ nor excluded from
// [^\s\p{L}\p{M}\p{N}]. Reproduce that.
bool is_lm(char32_t c) {
  return uc::is_letter(c);
}
// [^\s\p{L}\p{M}\p{N}]
bool is_other(char32_t c) {
  return !uc::is_whitespace(c) && !uc::is_letter(c) && !uc::is_number(c);
}
// Case folding for the (?i:...) contraction alternative; U+017F (long s) folds to 's'.
char32_t fold(char32_t c) {
  return c >= U'A' && c <= U'Z' ? c + 32 : c == 0x17F ? U's' : c;
}

// End of the match of the `qwen35` pre-tokeniser regex at position i (always > i):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Alternatives are tried in order, each with the backtracking a regex engine would do.
std::size_t match_at(const std::u32string& s, std::size_t i) {
  const std::size_t n = s.size();
  auto run = [&](std::size_t j, auto pred) {
    while (j < n && pred(s[j]))
      ++j;
    return j;
  };
  if (s[i] == U'\'' && i + 1 < n) {
    for (const std::u32string_view c : {U"s", U"t", U"re", U"ve", U"m", U"ll", U"d"}) {
      if (i + 1 + c.size() > n)
        continue;
      bool ok = true;
      for (std::size_t k = 0; k < c.size() && ok; ++k)
        ok = fold(s[i + 1 + k]) == c[k];
      if (ok)
        return i + 1 + c.size();
    }
  }
  if (!is_crlf(s[i]) && !uc::is_letter(s[i]) && !uc::is_number(s[i]) && i + 1 < n &&
      is_lm(s[i + 1])) {
    return run(i + 1, is_lm);
  }
  if (is_lm(s[i]))
    return run(i, is_lm);
  if (uc::is_number(s[i]))
    return i + 1;
  if (s[i] == U' ' && i + 1 < n && is_other(s[i + 1]))
    return run(run(i + 1, is_other), is_crlf);
  if (is_other(s[i]))
    return run(run(i, is_other), is_crlf);
  // Whitespace from here on.
  const std::size_t j = run(i, uc::is_whitespace);
  for (std::size_t q = j; q > i; --q) {
    if (is_crlf(s[q - 1]))
      return q;  // \s*[\r\n]+ ends at the run's last newline
  }
  if (j == n || j - i == 1)
    return j;    // \s+(?!\S) at the end of text, else \s+
  return j - 1;  // \s+(?!\S): leave the last space for the next token
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const gguf::File& f) {
  if (f.get<std::string>("tokenizer.ggml.model") != "gpt2" ||
      f.get<std::string>("tokenizer.ggml.pre") != "qwen35") {
    throw std::runtime_error("unsupported tokenizer (need gpt2 / qwen35)");
  }
  Tokenizer t;
  t.tokens_ = f.get<std::vector<std::string>>("tokenizer.ggml.tokens");
  const auto& types = f.get<std::vector<std::int64_t>>("tokenizer.ggml.token_type");
  const auto& merges = f.get<std::vector<std::string>>("tokenizer.ggml.merges");
  t.special_.resize(t.tokens_.size());
  for (std::size_t i = 0; i < t.tokens_.size(); ++i) {
    t.vocab_.emplace(t.tokens_[i], static_cast<int>(i));
    t.special_[i] = types[i] == 3 || types[i] == 4;  // control, user-defined
    if (t.special_[i])
      t.specials_.emplace_back(t.tokens_[i], static_cast<int>(i));
  }
  std::sort(t.specials_.begin(), t.specials_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  for (std::size_t r = 0; r < merges.size(); ++r)
    t.merge_rank_.emplace(merges[r], static_cast<int>(r));
  // GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the rest to U+0100 + n.
  int next = 0;
  for (int b = 0; b < 256; ++b) {
    const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    const char32_t c = keep ? static_cast<char32_t>(b) : static_cast<char32_t>(256 + next++);
    uc::append_utf8(t.byte_symbol_[b], c);
    t.symbol_byte_.emplace(c, static_cast<unsigned char>(b));
  }
  t.eos_ = static_cast<int>(f.get<std::int64_t>("tokenizer.ggml.eos_token_id"));
  return t;
}

void Tokenizer::encode_piece(const std::string& bytes, std::vector<int>& out) const {
  if (const auto it = cache_.find(bytes); it != cache_.end()) {
    out.insert(out.end(), it->second.begin(), it->second.end());
    return;
  }
  std::vector<std::string> sym;
  for (unsigned char b : bytes)
    sym.push_back(byte_symbol_[b]);
  while (sym.size() > 1) {
    int best = INT_MAX;
    std::size_t at = 0;
    for (std::size_t k = 0; k + 1 < sym.size(); ++k) {
      const auto it = merge_rank_.find(sym[k] + " " + sym[k + 1]);
      if (it != merge_rank_.end() && it->second < best)
        best = it->second, at = k;
    }
    if (best == INT_MAX)
      break;
    sym[at] += sym[at + 1];
    sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(at) + 1);
  }
  std::vector<int> ids;
  for (const auto& s : sym) {
    const auto it = vocab_.find(s);
    if (it == vocab_.end())
      throw std::runtime_error("tokenizer: symbol missing from vocab");
    ids.push_back(it->second);
  }
  out.insert(out.end(), ids.begin(), ids.end());
  cache_.emplace(bytes, std::move(ids));
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
  std::vector<int> out;
  auto encode_text = [&](std::string_view seg) {
    const std::u32string s = uc::nfc(uc::from_utf8(seg));
    for (std::size_t i = 0; i < s.size();) {
      const std::size_t j = match_at(s, i);
      std::string piece;
      for (std::size_t k = i; k < j; ++k)
        uc::append_utf8(piece, s[k]);
      encode_piece(piece, out);
      i = j;
    }
  };
  std::size_t start = 0;
  for (std::size_t i = 0; i < text.size();) {
    const auto hit = std::find_if(specials_.begin(), specials_.end(), [&](const auto& sp) {
      return text.compare(i, sp.first.size(), sp.first) == 0;
    });
    if (hit == specials_.end()) {
      ++i;
      continue;
    }
    encode_text(text.substr(start, i - start));
    out.push_back(hit->second);
    i += hit->first.size();
    start = i;
  }
  encode_text(text.substr(start));
  return out;
}

std::string Tokenizer::token_bytes(int id) const {
  const std::string& t = tokens_.at(static_cast<std::size_t>(id));
  if (special_[static_cast<std::size_t>(id)])
    return t;
  std::string out;
  for (char32_t c : uc::from_utf8(t)) {
    const auto it = symbol_byte_.find(c);
    if (it == symbol_byte_.end()) {
      uc::append_utf8(out, c);  // not byte-level (e.g. unused [PAD] entries): raw text
    } else {
      out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

std::string Tokenizer::decode(std::span<const int> ids) const {
  std::string out;
  for (int id : ids)
    out += token_bytes(id);
  return out;
}

std::string chat_prompt(std::string_view user_message, bool thinking) {
  std::string p = "<|im_start|>user\n";
  p += user_message;
  p += "<|im_end|>\n<|im_start|>assistant\n";
  p += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return p;
}

}  // namespace miso
