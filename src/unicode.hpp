#pragma once

// Unicode support for the tokenizer: UTF-8 conversion, the general categories the pre-tokeniser
// regex uses, and NFC normalisation. Data: src/unicode_tables.inc (tools/gen_unicode_tables.py).

#include <string>
#include <string_view>

namespace miso::unicode {

// Invalid or truncated UTF-8 sequences decode to U+FFFD, one per maximal invalid subsequence start.
std::u32string from_utf8(std::string_view s);
std::string to_utf8(std::u32string_view s);
void append_utf8(std::string& out, char32_t c);

bool is_letter(char32_t c);      // L*
bool is_mark(char32_t c);        // M*
bool is_number(char32_t c);      // N*
bool is_whitespace(char32_t c);  // White_Space property (what the tokenizer regex's \s matches)

std::u32string nfc(std::u32string_view s);

}  // namespace miso::unicode
