#!/usr/bin/env bash
# Invoke from the repository root. Compiles only; never runs a GPU executable.
set -euo pipefail
register_audit_results=bench/deltanet/results/2026-09-28-fp16-register
register_audit_tmp=$(mktemp -d /tmp/miso-register-audit-repeat.XXXXXX)
for register_audit_variant in volatile live register; do
  cp "$register_audit_results/audit-$register_audit_variant.hpp" "$register_audit_tmp/$register_audit_variant.hpp"
done
gzip -dc "$register_audit_results/audit-source-compiled.hip.gz" > "$register_audit_tmp/audit.hip"
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --offload-arch=gfx906 --cuda-device-only -O3 -DNDEBUG -std=gnu++20 -Ikernels -I"$register_audit_tmp" -S "$register_audit_tmp/audit.hip" -o "$register_audit_tmp/audit.s"
/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 --offload-arch=gfx906 --cuda-device-only -O3 -DNDEBUG -std=gnu++20 -Ikernels -I"$register_audit_tmp" -S -emit-llvm "$register_audit_tmp/audit.hip" -o "$register_audit_tmp/audit.ll"
printf '%s\n' "$register_audit_tmp"
