#pragma once

// Oracle comparison for prefill (ADR_003 D14): the state and next-token logits a Session holds
// after a prompt, whichever path produced them. Relative L2 differences per component; 0 means
// bit-identical.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "engine/model.hpp"

namespace miso::test {

struct SessionDiff {
  double logits = 0, kv = 0, conv = 0, state = 0;
  bool same_pos = true, same_pred = true;
  double max() const { return std::max({logits, kv, conv, state}); }
};

inline double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = double{a[i]} - b[i];
    num += d * d, den += double{b[i]} * b[i];
  }
  return den == 0 ? (num == 0 ? 0 : INFINITY) : std::sqrt(num / den);
}

inline std::vector<float> half_to_float(const std::vector<std::uint16_t>& h) {
  std::vector<float> f(h.size());
  for (std::size_t i = 0; i < h.size(); ++i)
    f[i] = static_cast<float>(__builtin_bit_cast(_Float16, h[i]));
  return f;
}

// Compares `got` against the reference session `want` (same model, same max_ctx).
inline SessionDiff compare(const Session& got, const Session& want) {
  SessionDiff d;
  d.same_pos = got.pos == want.pos;
  if (!d.same_pos || got.pos == 0)
    return d;
  const auto pg = got.pred.to_host(), pw = want.pred.to_host();
  d.same_pred = pg[got.pos - 1] == pw[want.pos - 1];
  d.logits = rel_l2(got.logits.to_host(), want.logits.to_host());
  for (std::size_t i = 0; i < want.kv.size(); ++i) {
    if (!want.kv[i])
      continue;
    const std::size_t n = std::size_t{want.kv[i]->len} * 256;  // valid positions of each KV head
    for (int kvh = 0; kvh < 2; ++kvh) {
      auto slice = [&](const DeviceBuffer<std::uint16_t>& b, unsigned max_ctx) {
        auto all = half_to_float(b.to_host());
        const auto* p = all.data() + static_cast<std::size_t>(kvh) * max_ctx * 256;
        return std::vector<float>(p, p + n);
      };
      d.kv = std::max({d.kv,
                       rel_l2(slice(got.kv[i]->k, got.kv[i]->max_ctx),
                              slice(want.kv[i]->k, want.kv[i]->max_ctx)),
                       rel_l2(slice(got.kv[i]->v, got.kv[i]->max_ctx),
                              slice(want.kv[i]->v, want.kv[i]->max_ctx))});
    }
  }
  for (std::size_t i = 0; i < want.deltanet.size(); ++i) {
    if (!want.deltanet[i])
      continue;
    const auto& g = *got.deltanet[i];
    const auto& w = *want.deltanet[i];
    d.conv = std::max(d.conv, rel_l2(g.conv[g.parity].to_host(), w.conv[w.parity].to_host()));
    d.state = std::max(d.state, rel_l2(g.s.to_host(), w.s.to_host()));
  }
  return d;
}

}  // namespace miso::test
