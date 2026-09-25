#pragma once

// Byte-level BPE tokenizer of Ornith / Qwen3.5 from GGUF metadata (ADR_002 D10), reproducing the
// Hugging Face tokenizer: special tokens split out first, NFC, the `qwen35` pre-tokeniser regex,
// GPT-2 byte-to-unicode mapping, BPE by merge rank.
//
// Known divergence: control / user-defined tokens that exist in the GGUF but not in the HF
// tokenizer's added tokens (the audio / TTS markers, ids 248070-248076) are matched as specials.

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf.hpp"

namespace miso {

class Tokenizer {
 public:
  static Tokenizer from_gguf(const gguf::File& f);

  std::vector<int> encode(std::string_view utf8) const;
  std::string decode(std::span<const int> ids) const;
  // Bytes of one token (byte-level tokens can end inside a UTF-8 sequence).
  std::string token_bytes(int id) const;

  int eos_id() const { return eos_; }
  std::size_t vocab_size() const { return tokens_.size(); }

 private:
  void encode_piece(const std::string& bytes, std::vector<int>& out) const;

  std::vector<std::string> tokens_;
  std::vector<bool> special_;
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merge_rank_;          // "left right" -> rank
  std::vector<std::pair<std::string, int>> specials_;        // longest first
  std::string byte_symbol_[256];                             // GPT-2 byte -> UTF-8 of mapped char
  std::unordered_map<char32_t, unsigned char> symbol_byte_;  // inverse
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
  int eos_ = -1;
};

// Single-turn chat prompt with the generation prompt appended, as the model's chat template renders
// a lone user message (thinking = true: open <think> block; false: empty think block).
std::string chat_prompt(std::string_view user_message, bool thinking);

}  // namespace miso
