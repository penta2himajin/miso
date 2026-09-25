#include "unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>

namespace miso::unicode {

namespace {

struct CategoryRange {
  char32_t lo, hi;
  std::uint8_t flags;  // 1 = L, 2 = M, 4 = N
};
struct CombiningClass {
  char32_t cp;
  std::uint8_t ccc;
};
struct Decomposition {
  char32_t cp;
  std::uint16_t offset;
  std::uint8_t length;
};
struct Composition {
  char32_t first, second, composite;
};

#include "unicode_tables.inc"

constexpr char32_t kHangulS = 0xAC00, kHangulL = 0x1100, kHangulV = 0x1161, kHangulT = 0x11A7;
constexpr int kHangulLCount = 19, kHangulVCount = 21, kHangulTCount = 28;
constexpr int kHangulN = kHangulVCount * kHangulTCount, kHangulSCount = kHangulLCount * kHangulN;

std::uint8_t flags(char32_t c) {
  const auto* end = std::end(kCategoryRanges);
  const auto* it = std::upper_bound(std::begin(kCategoryRanges), end, c,
                                    [](char32_t v, const CategoryRange& r) { return v < r.lo; });
  if (it == std::begin(kCategoryRanges))
    return 0;
  --it;
  return c <= it->hi ? it->flags : 0;
}

std::uint8_t ccc(char32_t c) {
  const auto* it = std::lower_bound(std::begin(kCombiningClasses), std::end(kCombiningClasses), c,
                                    [](const CombiningClass& e, char32_t v) { return e.cp < v; });
  return it != std::end(kCombiningClasses) && it->cp == c ? it->ccc : 0;
}

void decompose(char32_t c, std::u32string& out) {
  if (c >= kHangulS && c < kHangulS + kHangulSCount) {
    const int s = static_cast<int>(c - kHangulS);
    out.push_back(kHangulL + s / kHangulN);
    out.push_back(kHangulV + (s % kHangulN) / kHangulTCount);
    if (s % kHangulTCount != 0)
      out.push_back(kHangulT + s % kHangulTCount);
    return;
  }
  const auto* it = std::lower_bound(std::begin(kDecompositions), std::end(kDecompositions), c,
                                    [](const Decomposition& e, char32_t v) { return e.cp < v; });
  if (it != std::end(kDecompositions) && it->cp == c) {
    out.append(kDecompositionData + it->offset, it->length);
  } else {
    out.push_back(c);
  }
}

// Primary composite of (a, b), or 0.
char32_t compose(char32_t a, char32_t b) {
  if (a >= kHangulL && a < kHangulL + kHangulLCount && b >= kHangulV &&
      b < kHangulV + kHangulVCount) {
    return kHangulS + ((a - kHangulL) * kHangulVCount + (b - kHangulV)) * kHangulTCount;
  }
  if (a >= kHangulS && a < kHangulS + kHangulSCount && (a - kHangulS) % kHangulTCount == 0 &&
      b > kHangulT && b < kHangulT + kHangulTCount) {
    return a + (b - kHangulT);
  }
  const auto* it = std::lower_bound(std::begin(kCompositions), std::end(kCompositions), a,
                                    [](const Composition& e, char32_t v) { return e.first < v; });
  for (; it != std::end(kCompositions) && it->first == a; ++it) {
    if (it->second == b)
      return it->composite;
  }
  return 0;
}

}  // namespace

bool is_letter(char32_t c) {
  return flags(c) & 1;
}
bool is_mark(char32_t c) {
  return flags(c) & 2;
}
bool is_number(char32_t c) {
  return flags(c) & 4;
}

bool is_whitespace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

std::u32string nfc(std::u32string_view s) {
  if (std::all_of(s.begin(), s.end(), [](char32_t c) { return c < 0x80; }))
    return std::u32string(s);
  std::u32string d;
  d.reserve(s.size());
  for (char32_t c : s)
    decompose(c, d);
  // Canonical ordering: stable sort of each run of non-starters by combining class.
  for (std::size_t i = 0; i < d.size();) {
    if (ccc(d[i]) == 0) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < d.size() && ccc(d[j]) != 0)
      ++j;
    std::stable_sort(d.begin() + i, d.begin() + j,
                     [](char32_t a, char32_t b) { return ccc(a) < ccc(b); });
    i = j;
  }
  // Canonical composition.
  // Canonical composition: c combines with the last starter unless blocked, i.e. unless a character
  // between them has ccc 0 or ccc >= ccc(c). Everything after the starter is a non-starter in
  // ascending ccc order, so only the last appended character matters.
  std::u32string out;
  out.reserve(d.size());
  constexpr std::size_t kNone = static_cast<std::size_t>(-1);
  std::size_t starter = kNone;
  int last_cc = 0;
  for (char32_t c : d) {
    const int cc = ccc(c);
    if (starter != kNone) {
      const bool adjacent = out.size() - 1 == starter;
      if (adjacent || (last_cc != 0 && last_cc < cc)) {
        if (const char32_t comp = compose(out[starter], c)) {
          out[starter] = comp;
          continue;
        }
      }
    }
    if (cc == 0)
      starter = out.size();
    last_cc = cc;
    out.push_back(c);
  }
  return out;
}

std::u32string from_utf8(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    int n = b0 < 0x80 ? 1 : (b0 >> 5) == 6 ? 2 : (b0 >> 4) == 14 ? 3 : (b0 >> 3) == 30 ? 4 : 0;
    char32_t c = n == 1 ? b0 : n == 2 ? b0 & 0x1F : n == 3 ? b0 & 0x0F : b0 & 0x07;
    bool ok = n > 0 && i + n <= s.size();
    for (int k = 1; ok && k < n; ++k) {
      const auto b = static_cast<unsigned char>(s[i + k]);
      ok = (b >> 6) == 2;
      c = (c << 6) | (b & 0x3F);
    }
    const char32_t min[] = {0, 0, 0x80, 0x800, 0x10000};
    if (ok && (c < min[n] || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)))
      ok = false;
    if (ok) {
      out.push_back(c);
      i += n;
    } else {
      out.push_back(0xFFFD);
      ++i;
      while (i < s.size() && (static_cast<unsigned char>(s[i]) >> 6) == 2)
        ++i;
    }
  }
  return out;
}

void append_utf8(std::string& out, char32_t c) {
  if (c < 0x80) {
    out.push_back(static_cast<char>(c));
  } else if (c < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (c >> 6)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (c >> 12)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (c >> 18)));
    out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
}

std::string to_utf8(std::u32string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t c : s)
    append_utf8(out, c);
  return out;
}

}  // namespace miso::unicode
