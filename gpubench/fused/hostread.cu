// How fast a kernel copies one expert's bytes from host-mapped pinned memory into VRAM, by launch shape.
// For gpubench/LOAD_OVERLAP_LEDGER.md (Part 5). Each thread copies 16-byte vectors gid, gid + B*T, ... .
//   nvcc -O3 -arch=sm_86 -o hostread hostread.cu && ./hostread
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void k_copy(const uint4 * __restrict__ src, uint4 * __restrict__ dst, long nv) {
    const long n = (long) gridDim.x * blockDim.x;
    for (long v = (long) blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += n) dst[v] = src[v];
}
// the same bytes, but a thread's vectors are consecutive (chunk per thread) instead of strided
__global__ void k_copy_chunk(const uint4 * __restrict__ src, uint4 * __restrict__ dst, long nv) {
    const long n = (long) gridDim.x * blockDim.x, per = (nv + n - 1) / n, g = (long) blockIdx.x * blockDim.x + threadIdx.x;
    for (long v = g * per; v < min(nv, (g + 1) * per); ++v) dst[v] = src[v];
}
__global__ void k_empty(const uint4 *, uint4 *, long) {}
int main() {
    const size_t tot = 1ull << 30;
    void * h; CK(cudaHostAlloc(&h, tot, cudaHostAllocMapped)); memset(h, 1, tot);
    void * hd; CK(cudaHostGetDevicePointer(&hd, h, 0));
    uint8_t * d; CK(cudaMalloc(&d, 64 << 20));
    auto run = [&](int kind, int B, int T, size_t bytes) {
        std::vector<double> ts;
        for (int k = 0; k < 5; ++k) {
            CK(cudaDeviceSynchronize()); const double t0 = now();
            for (int i = 0; i < 400; ++i) {
                const uint4 * s = (const uint4 *) ((const uint8_t *) hd + ((size_t) (i * 7919 + k * 131) % 1500) * bytes);
                if (kind == 0) k_copy<<<B, T>>>(s, (uint4 *) d, (long) (bytes / 16));
                else if (kind == 1) k_copy_chunk<<<B, T>>>(s, (uint4 *) d, (long) (bytes / 16));
                else k_empty<<<B, T>>>(s, (uint4 *) d, 0);
            }
            CK(cudaDeviceSynchronize()); ts.push_back((now() - t0) / 400 * 1e6);
        }
        std::sort(ts.begin(), ts.end()); return ts[2];
    };
    for (size_t bytes : {(size_t) 172800, (size_t) 442368, (size_t) 663552}) {
        printf("bytes %zu  (us per launch, GB/s after subtracting an empty launch of the same shape)\n", bytes);
        for (int T : {64, 256, 1024}) for (int B : {1, 2, 4, 8, 12, 16, 27, 32, 54, 84, 128, 256}) {
            if ((size_t) B * T * 16 > bytes * 2) continue;
            const double e = run(2, B, T, bytes), a = run(0, B, T, bytes), c = run(1, B, T, bytes);
            printf("  T=%4d B=%3d reads/thread %7.1f  strided %7.2f us %6.1f GB/s   chunked %7.2f us %6.1f GB/s   empty %5.2f\n",
                   T, B, (double) bytes / 16 / B / T, a, bytes / ((a - e) * 1e-6) / 1e9, c, bytes / ((c - e) * 1e-6) / 1e9, e);
        }
    }
    return 0;
}
