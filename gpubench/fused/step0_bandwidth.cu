// Step 0 of the fused-kernel test (gpubench/LOAD_OVERLAP_LEDGER.md, O2-0): on one GPU, how fast does a batch-1
// matrix-vector product read expert-shaped Q4_0 weights that are already in VRAM, how fast does one expert
// arrive from pinned host RAM, and what is the ratio. Standalone: no engine, no model file, random weights.
// Shapes of the 1.26 B model: n_embd 800, expert width 192, 192 experts, 18 active, 13 MoE layers.
//   nvcc -O3 -arch=sm_86 -o step0_bandwidth step0_bandwidth.cu && ./step0_bandwidth
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct __attribute__((packed)) q4_0 { uint16_t d; uint8_t qs[16]; };   // 32 weights in 18 bytes
struct q8_1 { float d; float s; int8_t qs[32]; };                      // activation block: scale, sum, 32 int8

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__device__ inline int load4(const uint8_t * p) { int v; memcpy(&v, p, 4); return v; }
__device__ inline float half2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// one Q4_0 block times one Q8_1 block, the arithmetic of llama.cpp's vec_dot_q4_0_q8_1 (dp4a on nibbles)
__device__ inline float dot_block(const q4_0 * w, const q8_1 * x) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = load4(w->qs + 4*i);
        sumi = __dp4a(v & 0x0F0F0F0F, load4((const uint8_t *) x->qs + 4*i), sumi);
        sumi = __dp4a((v >> 4) & 0x0F0F0F0F, load4((const uint8_t *) x->qs + 16 + 4*i), sumi);
    }
    return half2f(w->d) * (x->d * (float) sumi - 8.0f * x->s);
}

// y[e][r] = W[id[e]][r] . x ; grid (nrows / RPB, nexp), one warp per block, RPB rows per block.
// stride_e = bytes between experts. Reads nexp * nrows * nblk * 18 bytes of weights.
template <int RPB>
__global__ void k_gemv(const uint8_t * __restrict__ W, size_t stride_e, const int * __restrict__ ids, const q8_1 * __restrict__ x,
                       float * __restrict__ y, int nrows, int nblk) {
    const int e = blockIdx.y, r0 = blockIdx.x * RPB;
    const q4_0 * w = (const q4_0 *) (W + (size_t) ids[e] * stride_e) + (size_t) r0 * nblk;
    float acc[RPB];
#pragma unroll
    for (int i = 0; i < RPB; ++i) acc[i] = 0.0f;
    for (int b = threadIdx.x; b < nblk; b += 32) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) acc[i] += dot_block(w + (size_t) i * nblk + b, x + b);
    }
#pragma unroll
    for (int i = 0; i < RPB; ++i) {
        float a = acc[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
        if (threadIdx.x == 0) y[(size_t) e * nrows + r0 + i] = a;
    }
}
// raw read: xor of every 16-byte vector (upper bound on what any kernel can read)
__global__ void k_read(const uint4 * __restrict__ p, long nv, unsigned * out) {
    unsigned a = 0;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < nv; i += (long) gridDim.x * blockDim.x) { const uint4 v = p[i]; a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) out[0] = a;
}
// the engine's copy kernel: 16-byte vectors from the host-mapped pool to a device slot
__global__ void k_copy(const uint4 * __restrict__ src, uint4 * __restrict__ dst, long nv) {
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < nv; i += (long) gridDim.x * blockDim.x) dst[i] = src[i];
}

template <typename F> static double timeit(F f, int reps) {     // seconds per call, back-to-back launches, median of 5
    std::vector<double> t;
    for (int k = 0; k < 5; ++k) { CK(cudaDeviceSynchronize()); const double a = now(); for (int i = 0; i < reps; ++i) f(i); CK(cudaDeviceSynchronize()); t.push_back((now() - a) / reps); }
    std::sort(t.begin(), t.end()); return t[2];
}

int main() {
    const int D = 800, F = 192, E = 192, K = 18;
    const size_t be_gu = (size_t) F * (D / 32) * 18, be_dn = (size_t) D * (F / 32) * 18;     // 86,400 each
    const size_t be = be_gu;
    printf("expert tensor: gate/up %zu bytes, down %zu bytes, one load %zu bytes\n", be_gu, be_dn, 2*be_gu + be_dn);
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("GPU: %s\n", pr.name);

    // device: all E experts of one tensor for NL layers (so reads are not served from one hot region)
    const int NL = 13;
    const size_t tot = (size_t) NL * E * be;
    uint8_t * dW; CK(cudaMalloc(&dW, tot + 64));
    void * hpool; CK(cudaHostAlloc(&hpool, tot, cudaHostAllocMapped));
    { uint32_t s = 12345u; uint8_t * h = (uint8_t *) hpool; for (size_t i = 0; i < tot; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t) (s >> 24); }
      for (size_t i = 0; i < tot; i += 18) { h[i] = 0x66; h[i+1] = 0x2E; } }                // d = a small finite half
    CK(cudaMemcpy(dW, hpool, tot, cudaMemcpyHostToDevice));
    void * hdev; CK(cudaHostGetDevicePointer(&hdev, hpool, 0));
    q8_1 * dx; CK(cudaMalloc(&dx, 32 * sizeof(q8_1))); CK(cudaMemset(dx, 1, 32 * sizeof(q8_1)));
    float * dy; CK(cudaMalloc(&dy, (size_t) E * NL * 800 * sizeof(float)));
    int * dids; CK(cudaMalloc(&dids, (size_t) E * NL * sizeof(int)));
    { std::vector<int> id((size_t) E * NL); for (size_t i = 0; i < id.size(); ++i) id[i] = (int) i; CK(cudaMemcpy(dids, id.data(), id.size() * sizeof(int), cudaMemcpyHostToDevice)); }
    unsigned * dout; CK(cudaMalloc(&dout, 4));
    uint8_t * dslot; CK(cudaMalloc(&dslot, 64 * be));

    printf("\n(a) compute: batch-1 Q4_0 matrix-vector product over weights in VRAM\n");
    auto rep = [&](const char * what, double sec, size_t bytes) { printf("  %-62s %8.2f us  %8.1f GB/s\n", what, sec * 1e6, bytes / sec / 1e9); return bytes / sec / 1e9; };
    double bw_layer = 0, bw_big = 0, bw_read = 0;
    // one launch = the gate (or up) projection of K experts of one layer, experts cycling through the pool
    for (int rpb : {1, 2, 4}) {
        auto f = [&](int i) {
            const int * ids = dids + (size_t) ((i * K) % (E * NL - K));
            if (rpb == 1) k_gemv<1><<<dim3(F, K), 32>>>(dW, be, ids, dx, dy, F, D / 32);
            if (rpb == 2) k_gemv<2><<<dim3(F / 2, K), 32>>>(dW, be, ids, dx, dy, F, D / 32);
            if (rpb == 4) k_gemv<4><<<dim3(F / 4, K), 32>>>(dW, be, ids, dx, dy, F, D / 32);
        };
        char b[128]; snprintf(b, sizeof b, "18 experts, gate shape 192x800, one launch, %d row(s) per block", rpb);
        bw_layer = std::max(bw_layer, rep(b, timeit(f, 4000), (size_t) K * be));
    }
    { auto f = [&](int i) { k_gemv<1><<<dim3(D, K), 32>>>(dW, be, dids + (size_t) ((i * K) % (E * NL - K)), dx, dy, D, F / 32); };
      rep("18 experts, down shape 800x192, one launch", timeit(f, 4000), (size_t) K * be); }
    { auto f = [&](int i) { k_gemv<2><<<dim3(F / 2, 3 * K), 32>>>(dW, be, dids + (size_t) ((i * 3 * K) % (E * NL - 3 * K)), dx, dy, F, D / 32); };
      rep("54 expert tensors (one layer's 18 x 3) in one launch", timeit(f, 4000), (size_t) 3 * K * be); }
    for (int rpb : {1, 2}) {   // asymptote: every expert of every layer in one launch (216 MB)
        auto f = [&](int) { if (rpb == 1) k_gemv<1><<<dim3(F, E * NL), 32>>>(dW, be, dids, dx, dy, F, D / 32);
                            else          k_gemv<2><<<dim3(F / 2, E * NL), 32>>>(dW, be, dids, dx, dy, F, D / 32); };
        char b[128]; snprintf(b, sizeof b, "2,496 experts (216 MB) in one launch, %d row(s) per block", rpb);
        bw_big = std::max(bw_big, rep(b, timeit(f, 50), tot));
    }
    { auto f = [&](int) { k_read<<<1024, 256>>>((const uint4 *) dW, (long) (tot / 16), dout); };
      bw_read = rep("raw read of the same 216 MB (xor of 16-byte vectors)", timeit(f, 50), tot); }
    { auto f = [&](int i) { k_read<<<64, 256>>>((const uint4 *) (dW + (size_t) ((i * K) % (E * NL - K)) * be), (long) (K * be / 16), dout); };
      rep("raw read of 18 expert tensors (1.56 MB), one launch", timeit(f, 4000), (size_t) K * be); }
    { auto f = [&](int) { k_gemv<1><<<dim3(1, 1), 32>>>(dW, be, dids, dx, dy, 1, 1); };
      printf("  %-62s %8.2f us\n", "an almost empty launch (one block, one row)", timeit(f, 20000) * 1e6); }

    printf("\n(b) load: one expert from pinned host RAM\n");
    double bw_load1 = 0, bw_load3 = 0, bw_loadbig = 0;
    for (int blocks : {32, 64, 128, 256}) {
        auto f = [&](int i) { k_copy<<<blocks, 256>>>((const uint4 *) ((const uint8_t *) hdev + (size_t) ((i * 37) % (E * NL)) * be), (uint4 *) (dslot + (size_t) (i % 16) * be), (long) (be / 16)); };
        char b[128]; snprintf(b, sizeof b, "copy kernel, host-mapped pool, 86,400 bytes, %d blocks", blocks);
        bw_load1 = std::max(bw_load1, rep(b, timeit(f, 4000), be));
    }
    for (int blocks : {64, 256}) {
        auto f = [&](int i) { k_copy<<<blocks, 256>>>((const uint4 *) ((const uint8_t *) hdev + (size_t) ((i * 37) % (E * NL - 3)) * be), (uint4 *) (dslot + (size_t) (i % 16) * 3 * be), (long) (3 * be / 16)); };
        char b[128]; snprintf(b, sizeof b, "copy kernel, host-mapped pool, 259,200 bytes (one load), %d blocks", blocks);
        bw_load3 = std::max(bw_load3, rep(b, timeit(f, 4000), 3 * be));
    }
    { auto f = [&](int i) { CK(cudaMemcpyAsync(dslot + (size_t) (i % 16) * 3 * be, (const uint8_t *) hpool + (size_t) ((i * 37) % (E * NL - 3)) * be, 3 * be, cudaMemcpyHostToDevice, 0)); };
      rep("cudaMemcpyAsync from pinned memory, 259,200 bytes", timeit(f, 4000), 3 * be); }
    { auto f = [&](int) { k_copy<<<1024, 256>>>((const uint4 *) hdev, (uint4 *) dW, (long) (tot / 16)); };
      bw_loadbig = rep("copy kernel, host-mapped pool, 216 MB", timeit(f, 10), tot); }
    { auto f = [&](int) { CK(cudaMemcpyAsync(dW, hpool, tot, cudaMemcpyHostToDevice, 0)); };
      rep("cudaMemcpyAsync from pinned memory, 216 MB", timeit(f, 10), tot); }
    { auto f = [&](int i) { k_gemv<2><<<dim3(F / 2, 1), 32>>>((const uint8_t *) hdev, be, dids + (i * 37) % (E * NL), dx, dy, F, D / 32); };
      rep("matrix-vector product reading one expert straight from host", timeit(f, 4000), be); }
    { auto f = [&](int i) { k_gemv<2><<<dim3(F / 2, 3), 32>>>((const uint8_t *) hdev, be, dids + (i * 37) % (E * NL - 3), dx, dy, F, D / 32); };
      rep("the same for three tensors (one load) in one launch", timeit(f, 4000), 3 * be); }

    printf("\n(c) ratio compute / load, against k - 1 = 17 and 17 + 2 (shared expert) = 19\n");
    printf("  per-layer launches : compute %.1f GB/s / load %.1f GB/s = %.1f\n", bw_layer, bw_load3, bw_layer / bw_load3);
    printf("  large launches     : compute %.1f GB/s / load %.1f GB/s = %.1f\n", bw_big, bw_loadbig, bw_big / bw_loadbig);
    printf("  raw read / load    : %.1f GB/s / %.1f GB/s = %.1f\n", bw_read, bw_loadbig, bw_read / bw_loadbig);
    return 0;
}
