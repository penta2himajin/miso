// MI50 (gfx906) microbenchmarks: HBM / cache bandwidth, ALU peaks, PCIe, launch overhead.
// Build: see bench/mi50/README.md
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------- bandwidth
__global__ void k_read(const float4* __restrict__ p, size_t n, float* out) {
  float4 s = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float4 v = p[i];
    s.x += v.x;
    s.y += v.y;
    s.z += v.z;
    s.w += v.w;
  }
  if (s.x + s.y + s.z + s.w == -1.0f)
    out[0] = s.x;
}
__global__ void k_write(float4* __restrict__ p, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_float4(1, 2, 3, 4);
}
__global__ void k_copy(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = a[i];
}
// Working-set sweep: each block streams coalesced through the whole power-of-two window
// from a block-specific start, so every CU's footprint equals the window size.
__global__ void k_sweep(const float4* __restrict__ p, unsigned mask, int iters, float* out) {
  unsigned base = blockIdx.x * 7919u * blockDim.x + threadIdx.x;
  float4 s = {0, 0, 0, 0};
  for (int r = 0; r < iters; ++r) {
    float4 v = p[(base + r * blockDim.x) & mask];
    s.x += v.x;
    s.y += v.y;
    s.z += v.z;
    s.w += v.w;
  }
  if (s.x + s.y + s.z + s.w == -1.0f)
    out[0] = s.x;
}
__global__ void k_lds(float* out, int iters) {
  __shared__ float4 buf[1024];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x)
    buf[i] = make_float4(i, 1, 2, 3);
  __syncthreads();
  float4 s = {0, 0, 0, 0};
  for (int r = 0; r < iters; ++r) {
    float4 v = buf[(threadIdx.x + r * 64) & 1023];
    s.x += v.x;
    s.y += v.y;
    s.z += v.z;
    s.w += v.w;
  }
  if (s.x + s.y + s.z + s.w == -1.0f)
    out[0] = s.x;
}

// ---------------------------------------------------------------- ALU peaks
// 8 independent chains per lane to cover ALU latency.
#define CHAINS 8
__global__ void k_fp32(float* out, float b, float c, int iters) {
  float a[CHAINS];
  for (int j = 0; j < CHAINS; ++j)
    a[j] = threadIdx.x + j;
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < CHAINS; ++j)
      a[j] = __builtin_fmaf(a[j], b, c);
  float s = 0;
  for (int j = 0; j < CHAINS; ++j)
    s += a[j];
  if (s == -1.0f)
    out[0] = s;
}
__global__ void k_pkf16(float* out, half2_t b, half2_t c, int iters) {
  half2_t a[CHAINS];
  for (int j = 0; j < CHAINS; ++j)
    a[j] = (half2_t){(_Float16)j, (_Float16)threadIdx.x};
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < CHAINS; ++j)
      a[j] = __builtin_elementwise_fma(a[j], b, c);
  float s = 0;
  for (int j = 0; j < CHAINS; ++j)
    s += (float)a[j].x + (float)a[j].y;
  if (s == -1.0f)
    out[0] = s;
}
__global__ void k_dot2f16(float* out, half2_t b, half2_t c, int iters) {
  float a[CHAINS];
  for (int j = 0; j < CHAINS; ++j)
    a[j] = threadIdx.x + j;
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < CHAINS; ++j)
      a[j] = __builtin_amdgcn_fdot2(b, c, a[j], false);
  float s = 0;
  for (int j = 0; j < CHAINS; ++j)
    s += a[j];
  if (s == -1.0f)
    out[0] = s;
}
__global__ void k_dot4i8(int* out, int b, int c, int iters) {
  int a[CHAINS];
  for (int j = 0; j < CHAINS; ++j)
    a[j] = threadIdx.x + j;
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < CHAINS; ++j)
      a[j] = __builtin_amdgcn_sdot4(b, c, a[j], false);
  int s = 0;
  for (int j = 0; j < CHAINS; ++j)
    s += a[j];
  if (s == -1)
    out[0] = s;
}
__global__ void k_dot8i4(int* out, int b, int c, int iters) {
  int a[CHAINS];
  for (int j = 0; j < CHAINS; ++j)
    a[j] = threadIdx.x + j;
  for (int i = 0; i < iters; ++i)
    for (int j = 0; j < CHAINS; ++j)
      a[j] = __builtin_amdgcn_sdot8(b, c, a[j], false);
  int s = 0;
  for (int j = 0; j < CHAINS; ++j)
    s += a[j];
  if (s == -1)
    out[0] = s;
}
__global__ void k_empty() {}

// ---------------------------------------------------------------- host
template <class F>
static float time_ms(F launch, int reps = 10) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  launch();
  CHECK(hipDeviceSynchronize());
  std::vector<float> t(reps);
  for (int r = 0; r < reps; ++r) {
    CHECK(hipEventRecord(a));
    launch();
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    CHECK(hipEventElapsedTime(&t[r], a, b));
  }
  std::sort(t.begin(), t.end());
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return t[reps / 2];  // median
}

int main(int argc, char** argv) {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  printf(
      "# device: %s (%s) CUs=%d clock=%dkHz memclk=%dkHz busw=%dbit L2=%dB LDS/block=%zuB "
      "regs/block=%d\n",
      p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate, p.memoryClockRate,
      p.memoryBusWidth, p.l2CacheSize, p.sharedMemPerBlock, p.regsPerBlock);
  const int CU = p.multiProcessorCount, TPB = 256;
  float* dout;
  CHECK(hipMalloc(&dout, 64));

  // HBM bandwidth (1 GiB buffers)
  const size_t bytes = 1ull << 30, n = bytes / 16;
  float4 *A, *B;
  CHECK(hipMalloc(&A, bytes));
  CHECK(hipMalloc(&B, bytes));
  CHECK(hipMemset(A, 0, bytes));
  CHECK(hipMemset(B, 0, bytes));
  for (int bpc : {4, 8, 16, 32}) {
    int grid = CU * bpc;
    float r = time_ms([&] { k_read<<<grid, TPB>>>(A, n, dout); });
    float w = time_ms([&] { k_write<<<grid, TPB>>>(B, n); });
    float c = time_ms([&] { k_copy<<<grid, TPB>>>(A, B, n); });
    printf("hbm blocks/CU=%-3d read %7.1f GB/s  write %7.1f GB/s  copy %7.1f GB/s\n", bpc,
           bytes / r / 1e6, bytes / w / 1e6, 2 * bytes / c / 1e6);
  }

  // Working-set sweep (cache hierarchy)
  const int grid = CU * 16, iters = 4096;
  for (size_t ws = 16 << 10; ws <= (256ull << 20); ws *= 2) {
    unsigned mask = (unsigned)(ws / 16 - 1);
    float t = time_ms([&] { k_sweep<<<grid, TPB>>>(A, mask, iters, dout); });
    double moved = (double)grid * TPB * iters * 16;
    printf("sweep ws=%8zu KiB  %8.1f GB/s\n", ws >> 10, moved / t / 1e6);
  }
  {
    float t = time_ms([&] { k_lds<<<grid, TPB>>>(dout, iters); });
    printf("lds ds_read_b128              %8.1f GB/s\n", (double)grid * TPB * iters * 16 / t / 1e6);
  }

  // ALU peaks
  const int it = 1 << 14,
            agrid = CU * 40;  // 40 waves/CU at TPB=64 would be max occupancy; use 256 x 40
  double lanes = (double)agrid * TPB * it * CHAINS;
  half2_t h = {(_Float16)1.0f, (_Float16)0.5f};
  float t;
  t = time_ms([&] { k_fp32<<<agrid, TPB>>>(dout, 1.0001f, 0.0001f, it); });
  printf("peak fp32 fma           %7.2f TFLOP/s\n", lanes * 2 / t / 1e9);
  t = time_ms([&] { k_pkf16<<<agrid, TPB>>>(dout, h, h, it); });
  printf("peak v_pk_fma_f16       %7.2f TFLOP/s\n", lanes * 4 / t / 1e9);
  t = time_ms([&] { k_dot2f16<<<agrid, TPB>>>(dout, h, h, it); });
  printf("peak v_dot2_f32_f16     %7.2f TFLOP/s\n", lanes * 4 / t / 1e9);
  t = time_ms([&] { k_dot4i8<<<agrid, TPB>>>((int*)dout, 0x01020304, 0x05060708, it); });
  printf("peak v_dot4_i32_i8      %7.2f TOP/s\n", lanes * 8 / t / 1e9);
  t = time_ms([&] { k_dot8i4<<<agrid, TPB>>>((int*)dout, 0x12345678, 0x11223344, it); });
  printf("peak v_dot8_i32_i4      %7.2f TOP/s\n", lanes * 16 / t / 1e9);

  // Host <-> device
  const size_t hb = 512ull << 20;
  void* pinned;
  CHECK(hipHostMalloc(&pinned, hb));
  void* pageable = malloc(hb);
  memset(pageable, 1, hb);
  float h2d = time_ms([&] { CHECK(hipMemcpy(A, pinned, hb, hipMemcpyHostToDevice)); }, 5);
  float d2h = time_ms([&] { CHECK(hipMemcpy(pinned, A, hb, hipMemcpyDeviceToHost)); }, 5);
  float h2dp = time_ms([&] { CHECK(hipMemcpy(A, pageable, hb, hipMemcpyHostToDevice)); }, 5);
  printf("pcie pinned H2D %6.2f GB/s  D2H %6.2f GB/s  pageable H2D %6.2f GB/s\n", hb / h2d / 1e6,
         hb / d2h / 1e6, hb / h2dp / 1e6);

  // Launch overhead
  const int L = 10000;
  float tl = time_ms(
      [&] {
        for (int i = 0; i < L; ++i)
          k_empty<<<1, 64>>>();
      },
      5);
  printf("launch back-to-back (stream)  %6.2f us/kernel\n", tl * 1e3 / L);
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  hipGraph_t g;
  hipGraphExec_t ge;
  const int GK = 400;  // ~ 40 layers x 10 kernels
  CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  for (int i = 0; i < GK; ++i)
    k_empty<<<1, 64, 0, s>>>();
  CHECK(hipStreamEndCapture(s, &g));
  CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  float tg = time_ms(
      [&] {
        CHECK(hipGraphLaunch(ge, s));
        CHECK(hipStreamSynchronize(s));
      },
      20);
  printf("hipGraph %d empty kernels   %6.2f us/kernel (%.1f us/graph)\n", GK, tg * 1e3 / GK,
         tg * 1e3);
  float ts = time_ms(
      [&] {
        k_empty<<<1, 64>>>();
        CHECK(hipDeviceSynchronize());
      },
      200);
  printf("launch+sync round trip        %6.2f us\n", ts * 1e3);
  return 0;
}
