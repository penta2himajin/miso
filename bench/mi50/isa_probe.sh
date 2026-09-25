#!/usr/bin/env bash
# Compile one tiny kernel per instruction builtin for a target and report
# whether the compiler accepts it and which ISA mnemonic it lowers to.
# Usage: bench/mi50/isa_probe.sh [gfx906]
set -u
ARCH="${1:-gfx906}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
# clang picks the newest GCC in /usr/lib/gcc; override when its libstdc++-dev is absent.
HIPFLAGS="${HIPFLAGS:-}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# name | expected mnemonic | kernel body (in: a,b ints / h half2 / bf short2; out: int* o / float* f)
PROBES=(
  "sdot4_i32_i8|v_dot4_i32_i8|o[0] = __builtin_amdgcn_sdot4(a, b, o[1], false);"
  "udot4_u32_u8|v_dot4_u32_u8|o[0] = __builtin_amdgcn_udot4(a, b, o[1], false);"
  "sdot8_i32_i4|v_dot8_i32_i4|o[0] = __builtin_amdgcn_sdot8(a, b, o[1], false);"
  "udot8_u32_u4|v_dot8_u32_u4|o[0] = __builtin_amdgcn_udot8(a, b, o[1], false);"
  "sdot2_i32_i16|v_dot2_i32_i16|o[0] = __builtin_amdgcn_sdot2(s2a, s2b, o[1], false);"
  "fdot2_f32_f16|v_dot2_f32_f16|f[0] = __builtin_amdgcn_fdot2(h, h2, f[1], false);"
  "fdot2_f32_bf16|v_dot2_f32_bf16|f[0] = __builtin_amdgcn_fdot2_f32_bf16(bf, bf2, f[1], false);"
  "pk_fma_f16|v_pk_fma_f16|hp[0] = __builtin_elementwise_fma(h, h2, hp[1]);"
  "pk_fma_f32|v_pk_fma_f32|f2p[0] = __builtin_elementwise_fma(f2a, f2b, f2p[1]);"
  "fmac_f32|v_fmac_f32|f[0] = __builtin_fmaf(f[2], f[3], f[0]);"
  "mfma_f32_32x32x8f16|v_mfma_f32_32x32x8f16|acc[0] = __builtin_amdgcn_mfma_f32_32x32x8f16(h4, h4, acc[1], 0, 0, 0);"
  "global_atomic_add_f32|global_atomic_add_f32|atomicAdd(f, f[1]);"
  "global_atomic_pk_add_f16|global_atomic_pk_add_f16|__builtin_amdgcn_global_atomic_fadd_v2f16((__attribute__((address_space(1))) half2_t*)hp, h);"
  "ds_bpermute|ds_bpermute_b32|o[0] = __builtin_amdgcn_ds_bpermute(a, b);"
  "mov_dpp|_dpp|o[0] = __builtin_amdgcn_update_dpp(0, a, 0x111, 0xf, 0xf, false);"
)

printf "%-26s %-8s %s\n" "probe" "result" "detail"
for p in "${PROBES[@]}"; do
  IFS='|' read -r name mnem body <<<"$p"
  cat >"$WORK/k.hip" <<EOF
#include <hip/hip_runtime.h>
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef short short2_t __attribute__((ext_vector_type(2)));
typedef __bf16 bf162_t __attribute__((ext_vector_type(2)));
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float16_t __attribute__((ext_vector_type(16)));
extern "C" __global__ void k(int* o, float* f, half2_t* hp, float2_t* f2p, float16_t* acc,
                             int a, int b, half2_t h, half2_t h2, short2_t s2a, short2_t s2b,
                             bf162_t bf, bf162_t bf2, float2_t f2a, float2_t f2b, half4_t h4) {
  $body
}
EOF
  if "$HIPCC" $HIPFLAGS --offload-arch="$ARCH" -O3 --cuda-device-only -S -o "$WORK/k.s" "$WORK/k.hip" >"$WORK/err" 2>&1; then
    if grep -q "$mnem" "$WORK/k.s"; then
      printf "%-26s %-8s %s\n" "$name" "OK" "$(grep -m1 -o "${mnem}[a-z0-9_]*" "$WORK/k.s")"
    else
      printf "%-26s %-8s %s\n" "$name" "EMULATED" "compiled, but no $mnem in ISA"
    fi
  else
    printf "%-26s %-8s %s\n" "$name" "NO" "$(grep -m1 -oE "error: .{0,90}" "$WORK/err")"
  fi
done
