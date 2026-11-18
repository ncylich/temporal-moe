// Does a copy from host-mapped memory overlap with VRAM-bound work in the same launch, or do they add?
// One launch: the first CB blocks copy `cbytes` host -> VRAM (strided, 1024 threads), the other RB blocks read
// `rbytes` of VRAM (16-byte vectors, xor-reduced), as a stand-in for the op's weight reads. Also: the two as
// separate launches on one stream, and on two streams.  For gpubench/LOAD_OVERLAP_LEDGER.md (Part 5).
//   nvcc -O3 -arch=sm_86 -o hostmix hostmix.cu && ./hostmix
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void k_mix(const uint4 * __restrict__ hsrc, uint4 * __restrict__ slot, long cv, int CB,
                      const uint4 * __restrict__ w, long rv, unsigned * out, int first) {
    const int T = blockDim.x, RB = gridDim.x - CB;
    const bool copy = first ? (int) blockIdx.x < CB : (int) blockIdx.x >= RB;      // copy blocks at the front or at the back
    if (copy) {
        const long b = first ? blockIdx.x : blockIdx.x - RB, n = (long) CB * T;
        for (long v = b * T + threadIdx.x; v < cv; v += n) slot[v] = hsrc[v];
    } else {
        const long b = first ? blockIdx.x - CB : blockIdx.x, n = (long) RB * T; unsigned a = 0;
        for (long v = b * T + threadIdx.x; v < rv; v += n) { const uint4 q = w[v]; a ^= q.x ^ q.y ^ q.z ^ q.w; }
        if (threadIdx.x == 0) out[b] = a;
    }
}
int main() {
    const size_t tot = 1ull << 30;
    void * h; CK(cudaHostAlloc(&h, tot, cudaHostAllocMapped)); memset(h, 1, tot);
    void * hd; CK(cudaHostGetDevicePointer(&hd, h, 0));
    uint8_t * slot; CK(cudaMalloc(&slot, 64 << 20));
    uint8_t * w; CK(cudaMalloc(&w, 1ull << 30)); CK(cudaMemset(w, 3, 1ull << 30));
    unsigned * out; CK(cudaMalloc(&out, 1 << 20));
    cudaStream_t s2; CK(cudaStreamCreate(&s2)); cudaEvent_t ev; CK(cudaEventCreate(&ev));
    auto med = [&](auto f) { std::vector<double> ts; for (int k = 0; k < 5; ++k) { CK(cudaDeviceSynchronize()); const double t0 = now();
        for (int i = 0; i < 300; ++i) f(i); CK(cudaDeviceSynchronize()); ts.push_back((now() - t0) / 300 * 1e6); } std::sort(ts.begin(), ts.end()); return ts[2]; };
    auto hs = [&](int i, size_t cb) { return (const uint4 *) ((const uint8_t *) hd + ((size_t) (i * 7919) % 1500) * cb); };
    auto ws = [&](int i, size_t rb) { return (const uint4 *) (w + ((size_t) (i * 104729) % 100) * rb); };
    for (size_t cb : {(size_t) 442368, (size_t) 172800}) for (size_t rb : {(size_t) 8 << 20, (size_t) 4 << 20}) for (int CB : {12, 27}) for (int RB : {204, 84}) {
        const long cv = cb / 16, rv = rb / 16;
        const double tc = med([&](int i) { k_mix<<<CB, 1024>>>(hs(i, cb), (uint4 *) slot, cv, CB, ws(i, rb), 0, out, 1); });
        const double tr = med([&](int i) { k_mix<<<RB, 1024>>>(hs(i, cb), (uint4 *) slot, 0, 0, ws(i, rb), rv, out, 1); });
        const double tf = med([&](int i) { k_mix<<<CB + RB, 1024>>>(hs(i, cb), (uint4 *) slot, cv, CB, ws(i, rb), rv, out, 1); });
        const double tb = med([&](int i) { k_mix<<<CB + RB, 1024>>>(hs(i, cb), (uint4 *) slot, cv, CB, ws(i, rb), rv, out, 0); });
        const double t2 = med([&](int i) { k_mix<<<CB, 1024, 0, s2>>>(hs(i, cb), (uint4 *) slot, cv, CB, ws(i, rb), 0, out, 1); cudaEventRecord(ev, s2);
                                           k_mix<<<RB, 1024>>>(hs(i, cb), (uint4 *) slot, 0, 0, ws(i, rb), rv, out, 1); cudaStreamWaitEvent(0, ev, 0); });
        printf("copy %6zu B in %2d blocks, read %zu MB in %3d blocks:  copy alone %6.2f  read alone %6.2f (%5.0f GB/s)  one launch, copy first %6.2f  copy last %6.2f  two streams %6.2f us\n",
               cb, CB, rb >> 20, RB, tc, tr, rb / ((tr - 2.0) * 1e-6) / 1e9, tf, tb, t2);
    }
    return 0;
}
