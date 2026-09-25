#!/usr/bin/env python3
"""Tokenizer reference ids from the Hugging Face tokenizer (ADR_002 D10) -> tests/fixtures/tokenizer_cases.gguf.

The C++ tokenizer must reproduce these ids exactly. Stored as GGUF (read by src/gguf.hpp):
  tok.texts     string array, one case per entry
  tok.ids       int32 array, all cases' ids concatenated
  tok.offsets   int32 array, start of each case in tok.ids (len = cases + 1)

Usage: .venv/bin/python tools/make_tokenizer_fixtures.py
"""
import os
import random

import numpy as np
from gguf import GGUFWriter
from transformers import AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHAT_MESSAGE = "日本の首都は？\nOne line, please."

CASES = [
    "", "Hello world", "Hello  world", "  leading", "trailing  ", "a", " ", "   ", "\n", "\n\n", " \n ",
    "tabs\tand\ttabs", "line1\nline2\r\nline3\n\n\nx", "a \n\n b", "x \t\n y", "end with space ",
    "don't I'LL we'Re you've it's he'd THEY'D o'clock", "it'ſ Kelvin \u212a", "'s alone", "''s",
    "numbers 1234567 3.14159 1e10 -42 0x1F", "１２３ fullwidth ４５", "Ⅻ roman ⅷ, x² y³ ½",
    "punct!!! ??? ... --- ***", "a+b=c; x*=2; y->z && w||v", "email@example.com (a) [b] {c}",
    "https://github.com/penta2himajin/miso/pull/10?x=1&y=2#frag",
    "def f(x):\n    return x * 2  # comment\n\n\nclass A:\n\tpass\n",
    "for (int i = 0; i < n; ++i) {\n  y[i] += a * x[i];\n}\n",
    '{"key": [1, 2, 3], "nested": {"a": null, "b": true}}',
    "The capital of France is Paris. The capital of Japan is Tokyo. 日本の首都は",
    "東京です。 The capital of Japan is", "日本語のテキストと English が混ざった文。改行も\nある。",
    "中文句子，包含标点符号。", "한국어 문장입니다.", "\u1100\u1161\u11a8 decomposed hangul",
    "e\u0301 cafe\u0301 café", "か\u3099 が は\u309a ぱ", "A\u030a Å \u212b angstrom",
    "Ω\u2126 ohm", "ﬁ ligature", "x\u00a0y nbsp", "\u3000全角スペース\u3000", "zero\u200bwidth",
    "emoji 👍🏽 👨‍👩‍👧 🇯🇵 ✈️", "العربية نص", "हिन्दी पाठ", "ภาษาไทย", "Ελληνικά Кириллица",
    "math ∑∫√∞ ≤≥≠", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\n\n</think>\n\nAnswer", "<|im_start partial <|im_end|", "<tool_call>{}</tool_call>",
    "mixed<|endoftext|>tail", "  \n\t \r\n  ", "a" * 300, "ab " * 100,
]


def random_cases(n, seed=20260926):
    rng = random.Random(seed)
    pools = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF),
             (0x3040, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00), (0x1F300, 0x1F64F), (0x2000, 0x206F)]
    out = []
    for _ in range(n):
        s = []
        for _ in range(rng.randint(1, 40)):
            lo, hi = rng.choice(pools)
            s.append(chr(rng.randint(lo, hi)))
            if rng.random() < 0.15:
                s.append(rng.choice([" ", "  ", "\n", "\t", "'s", "'LL", "1", "12"]))
        out.append("".join(s))
    return out


def main():
    tok = AutoTokenizer.from_pretrained("ornith-ai/Ornith-1.5-35B-A3B")
    docs = [open(os.path.join(ROOT, p), encoding="utf-8").read()
            for p in ("AGENTS.md", "docs/research/mi50.md", "tests/golden/README.md")]
    texts = CASES + docs + random_cases(200)
    ids, offsets = [], [0]
    for t in texts:
        e = tok(t, add_special_tokens=False)["input_ids"]
        ids += e
        offsets.append(len(ids))
    path = os.path.join(ROOT, "tests", "fixtures", "tokenizer_cases.gguf")
    w = GGUFWriter(path, "miso-tokenizer-test")
    w.add_string("tok.source", "ornith-ai/Ornith-1.5-35B-A3B tokenizer.json via transformers")
    w.add_array("tok.texts", texts)
    w.add_array("tok.offsets", offsets)
    msgs = [{"role": "user", "content": CHAT_MESSAGE}]
    w.add_string("chat.message", CHAT_MESSAGE)
    w.add_string("chat.think", tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    w.add_string("chat.no_think", tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False))
    w.add_tensor("tok.ids", np.asarray(ids, dtype=np.int32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}: {len(texts)} cases, {len(ids)} ids")


if __name__ == "__main__":
    main()
