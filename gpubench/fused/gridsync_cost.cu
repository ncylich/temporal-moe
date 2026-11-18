// What a grid-wide barrier inside one kernel costs (cooperative groups, grid.sync()), for the one-launch-per-layer
// idea in gpubench/LOAD_OVERLAP_LEDGER.md, O2-4. A cooperative launch needs every block resident at once, so the
// grid is limited to what the occupancy query allows.
//   nvcc -O3 -arch=sm_86 -o gridsync_cost gridsync_cost.cu && ./gridsync_cost
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace cg = cooperative_groups;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__global__ void k_sync(float * buf, int nsync) {
    cg::grid_group g = cg::this_grid();
    float a = (float) threadIdx.x;
    for (int s = 0; s < nsync; ++s) { a += 1.0f; g.sync(); }
    if (threadIdx.x == 0) buf[blockIdx.x] = a;
}
__global__ void k_plain(float * buf) { if (threadIdx.x == 0) buf[blockIdx.x] = (float) threadIdx.x; }

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("GPU %s, %d SMs, cooperative launch %s\n", pr.name, pr.multiProcessorCount, pr.cooperativeLaunch ? "supported" : "NOT supported");
    float * buf; CK(cudaMalloc(&buf, 1 << 20));
    for (int threads : {32, 128, 256, 1024}) {
        int per_sm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_sync, threads, 0));
        const int maxb = per_sm * pr.multiProcessorCount;
        for (int blocks : {maxb / 4, maxb}) {
            printf("blocks of %4d threads, grid %5d (limit %5d):", threads, blocks, maxb);
            double base = 0;
            for (int nsync : {0, 1, 2, 4, 8}) {
                void * args[] = { &buf, &nsync };
                std::vector<double> ts;
                for (int k = 0; k < 5; ++k) {
                    CK(cudaDeviceSynchronize()); const double t0 = now();
                    for (int i = 0; i < 2000; ++i) CK(cudaLaunchCooperativeKernel((void *) k_sync, dim3(blocks), dim3(threads), args, 0, 0));
                    CK(cudaDeviceSynchronize()); ts.push_back((now() - t0) / 2000 * 1e6);
                }
                std::sort(ts.begin(), ts.end());
                if (nsync == 0) base = ts[2];
                printf("  %d sync %.2f us", nsync, ts[2]);
                if (nsync == 8) printf("  => %.2f us per barrier", (ts[2] - base) / 8);
            }
            printf("\n");
        }
    }
    std::vector<double> ts;
    for (int k = 0; k < 5; ++k) { CK(cudaDeviceSynchronize()); const double t0 = now(); for (int i = 0; i < 2000; ++i) k_plain<<<84, 1024>>>(buf); CK(cudaDeviceSynchronize()); ts.push_back((now() - t0) / 2000 * 1e6); }
    std::sort(ts.begin(), ts.end()); printf("ordinary launch, 84 blocks of 1024 threads: %.2f us\n", ts[2]);
    return 0;
}
