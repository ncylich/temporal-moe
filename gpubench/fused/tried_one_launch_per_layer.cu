// Layer-level benchmark for the fused-kernel test (gpubench/LOAD_OVERLAP_LEDGER.md, O2-1 and O2-2).
// Standalone: no engine, no model file, no attention. The 1.26 B model's MoE shapes with random Q4_0 weights:
// 13 MoE layers, 192 experts, 18 active, expert width 192, shared expert width 384, n_embd 800. All 192 experts
// of a layer sit in pinned host RAM (host-mapped, as the engine's pool); 18 slots per layer sit in VRAM.
// Routing is replayed from a trace (loads per token-layer: pinned n, exactly one, or drawn from a histogram),
// not computed. Per token: x = X0[t]; for each layer x = rmsnorm(x + shared(x) + sum_j w * expert_j(x)).
//
// Two implementations of the same function:
//   unfused  the engine's sequence of kernels per layer (router, routing op, copies on a second stream with
//            events, gate-weight ops, shared expert, routed experts, sum), one launch per engine op
//   fused    one launch per layer: one block per routed expert, one for the shared expert, one for the router;
//            the block of an incoming expert first copies its three tensors from the host pool into its slot
//            and then computes, while the other blocks compute; the combine of a layer runs in the prologue of
//            the next layer's launch
// Every timed run is checked against the unfused all-in-VRAM path on the same trace (max relative difference
// of the final activations).
//   nvcc -O3 -arch=sm_86 -o harness harness.cu && ./harness [tokens=2000] [skliar histogram "w0,w1,..."]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

constexpr int NL = 13, E = 192, K = 18, D = 800, F = 192, FS = 384, MAXLD = 8;
constexpr int DB = D / 32;                       // q8/q4 blocks per n_embd row
constexpr size_t BE = (size_t) F * DB * 18;      // bytes of one expert tensor (gate, up and down are all 86,400)
constexpr size_t BS = (size_t) FS * DB * 18;     // shared expert tensor
static_assert(BE == (size_t) D * (F / 32) * 18, "down has the same size");
constexpr float WGT = 1.0f / K;

struct __attribute__((packed)) q4_0 { uint16_t d; uint8_t qs[16]; };
struct q8_1 { float d; float s; int8_t qs[32]; };

__device__ inline int load4(const uint8_t * p) { int v; memcpy(&v, p, 4); return v; }
__device__ inline float dot_block(const q4_0 * w, const q8_1 * x) {     // llama.cpp's vec_dot_q4_0_q8_1
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = load4(w->qs + 4*i);
        sumi = __dp4a(v & 0x0F0F0F0F, load4((const uint8_t *) x->qs + 4*i), sumi);
        sumi = __dp4a((v >> 4) & 0x0F0F0F0F, load4((const uint8_t *) x->qs + 16 + 4*i), sumi);
    }
    return __half2float(__ushort_as_half(w->d)) * (x->d * (float) sumi - 8.0f * x->s);
}
__device__ inline void quant_block(const float * x, q8_1 * q) {         // llama.cpp's quantize_row_q8_1, one block
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f; int s = 0;
    for (int i = 0; i < 32; ++i) { const int v = (int) roundf(x[i] * id); q->qs[i] = (int8_t) v; s += v; }
    q->d = d; q->s = (float) s;
}
__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ inline float warp_sum(float a) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
    return a;
}

// ------------------------------------------------------------------ unfused: one kernel per engine op
__global__ void k_quant(const float * __restrict__ x, q8_1 * __restrict__ q, int nblk, int nch, int stride) {   // nch channels
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nblk * nch) return;
    const int c = i / nblk, b = i % nblk;
    quant_block(x + (size_t) c * stride + 32 * b, q + (size_t) c * nblk + b);
}
// y[c][r] = W[ids[c]][r] . x[c or 0]; grid (nrows, nch), one warp per row
__global__ void k_gemv(const uint8_t * __restrict__ W, size_t stride_e, const int * __restrict__ ids, const q8_1 * __restrict__ x, int x_per_ch,
                       float * __restrict__ y, int nrows, int nblk) {
    const int c = blockIdx.y, r = blockIdx.x;
    const q4_0 * w = (const q4_0 *) (W + (size_t) (ids ? ids[c] : c) * stride_e) + (size_t) r * nblk;
    const q8_1 * xc = x + (x_per_ch ? (size_t) c * nblk : 0);
    float a = 0.0f;
    for (int b = threadIdx.x; b < nblk; b += 32) a += dot_block(w + b, xc + b);
    a = warp_sum(a);
    if (threadIdx.x == 0) y[(size_t) c * nrows + r] = a;
}
// fused gate/up/SwiGLU (the engine's all-in-VRAM path): act[c][r] = silu(G.x) * (U.x)
__global__ void k_gemv_glu(const uint8_t * __restrict__ G, const uint8_t * __restrict__ U, size_t stride_e, const int * __restrict__ ids,
                           const q8_1 * __restrict__ x, float * __restrict__ y, int nrows, int nblk) {
    const int c = blockIdx.y, r = blockIdx.x; const size_t o = (size_t) (ids ? ids[c] : c) * stride_e;
    const q4_0 * g = (const q4_0 *) (G + o) + (size_t) r * nblk, * u = (const q4_0 *) (U + o) + (size_t) r * nblk;
    float a = 0.0f, b2 = 0.0f;
    for (int b = threadIdx.x; b < nblk; b += 32) { a += dot_block(g + b, x + b); b2 += dot_block(u + b, x + b); }
    a = warp_sum(a); b2 = warp_sum(b2);
    if (threadIdx.x == 0) y[(size_t) c * nrows + r] = silu(a) * b2;
}
__global__ void k_glu(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = silu(g[i]) * u[i];
}
__global__ void k_router(const float * __restrict__ Wr, const float * __restrict__ x, float * __restrict__ lg) {   // F32 [E x D]
    const float * w = Wr + (size_t) blockIdx.x * D; float a = 0.0f;
    for (int i = threadIdx.x; i < D; i += 32) a += w[i] * x[i];
    a = warp_sum(a); if (threadIdx.x == 0) lg[blockIdx.x] = a;
}
__global__ void k_small(const float * __restrict__ in, float * __restrict__ out, int n) {   // stand-in for a small elementwise op
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = in[i] * 0.5f + 1.0f;
}
// the routing op, replayed: copies this token-layer's swap list from the trace (the engine's kernel decides it)
__global__ void k_route_replay(const float * lg, const int * tr_n, const int * tr_e, const int * tr_s, int * nswp, int * swp_e, int * swp_s, float * sink) {
    const int i = threadIdx.x;
    if (i == 0) { *nswp = *tr_n; sink[0] = lg[0]; }
    if (i < MAXLD) { swp_e[i] = tr_e[i]; swp_s[i] = tr_s[i]; }
}
__global__ void k_swapcopy(const uint4 * __restrict__ pool, uint4 * __restrict__ slot, const int * __restrict__ swp_e, const int * __restrict__ swp_s, const int * __restrict__ nswp) {
    const int ns = *nswp; if (ns <= 0) return;
    const long nv = (long) (BE / 16), total = (long) ns * nv;
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long) gridDim.x * blockDim.x) {
        const int j = (int) (i / nv); const long off = i - (long) j * nv;
        slot[(size_t) swp_s[j] * nv + off] = pool[(size_t) swp_e[j] * nv + off];
    }
}
// acc[i] = (first ? 0 : acc[i]) + sum over up to 8 channels of w * y[c][i]   (the engine fuses up to 8 adds)
__global__ void k_addn(const float * __restrict__ y, int c0, int nc, float w, float * __restrict__ acc, int first) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= D) return;
    float a = first ? 0.0f : acc[i];
    for (int c = c0; c < c0 + nc; ++c) a += w * y[(size_t) c * D + i];
    acc[i] = a;
}
__global__ void k_add2(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ o) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < D) o[i] = a[i] + b[i];
}
__global__ void k_norm(const float * __restrict__ in, float * __restrict__ out) {     // one block of 1024 threads
    __shared__ float part[32];
    const int i = threadIdx.x; float v = i < D ? in[i] : 0.0f;
    float s = warp_sum(v * v);
    if ((i & 31) == 0) part[i >> 5] = s;
    __syncthreads();
    float t = 0.0f; for (int w = 0; w < 32; ++w) t += part[w];
    if (i < D) out[i] = v * rsqrtf(t / D + 1e-6f);
}

// ------------------------------------------------------------------ fused: one launch per layer
struct FusedArgs {
    const float * xprev;        // input of the previous layer (or X0[t] for layer 0)
    const float * outprev;      // previous layer's block outputs [(K+1) x D]: K routed, then shared; null for layer 0
    float * xcur;               // this layer's input, written by block 0 (read by the next launch)
    float * out;                // this layer's block outputs [(K+1) x D]
    const uint8_t * W[3];       // gate, up, down: slot tensors [K] (streamed) or the full tables [E] (all in VRAM)
    const int * ids;            // expert (or slot) index of each routed position
    const uint8_t * pool[3];    // host-mapped pool [E] per tensor (streamed), null in VRAM
    const int * inc;            // [K]: expert to load into this position's slot before computing, or -1
    const uint8_t * S[3];       // shared expert gate, up, down
    const float * Wr; float * lg;   // router weights and logits (computed for cost; routing is replayed)
    float * xfinal;             // non-null: combine only, write the token's final activations
};
constexpr int FT = 256;         // threads per block
template <int FFv>              // rows of this block's expert (192 routed, 384 shared)
__device__ void expert_block(const uint8_t * G, const uint8_t * U, const uint8_t * Dn, const q8_1 * xq, float * act, q8_1 * aq, float * out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = FT / 32;
    for (int r = warp; r < FFv; r += nw) {                                 // gate and up rows: one warp per row
        const q4_0 * g = (const q4_0 *) G + (size_t) r * DB, * u = (const q4_0 *) U + (size_t) r * DB;
        float a = 0.0f, b = 0.0f;
        if (lane < DB) { a = dot_block(g + lane, xq + lane); b = dot_block(u + lane, xq + lane); }
        a = warp_sum(a); b = warp_sum(b);
        if (lane == 0) act[r] = silu(a) * b;
    }
    __syncthreads();
    if (threadIdx.x < FFv / 32) quant_block(act + 32 * threadIdx.x, aq + threadIdx.x);
    __syncthreads();
    for (int r = threadIdx.x; r < D; r += FT) {                            // down rows: one thread per row
        const q4_0 * d = (const q4_0 *) Dn + (size_t) r * (FFv / 32); float a = 0.0f;
        for (int b = 0; b < FFv / 32; ++b) a += dot_block(d + b, aq + b);
        out[r] = a;
    }
}
__global__ void k_fused(FusedArgs A) {
    __shared__ float x[D + 32]; __shared__ q8_1 xq[DB]; __shared__ float act[FS]; __shared__ q8_1 aq[FS / 32]; __shared__ float part[FT / 32];
    const int tid = threadIdx.x, blk = blockIdx.x;
    const bool incoming = !A.xfinal && blk < K && A.pool[0] && A.inc[blk] >= 0;
    // an incoming expert's block starts its read first: host pool -> slot, 16-byte vectors, three tensors
    if (incoming) {
        const long nv = (long) (BE / 16); const size_t eo = (size_t) A.inc[blk] * nv, so = (size_t) A.ids[blk] * nv;
#pragma unroll
        for (int w = 0; w < 3; ++w) {
            const uint4 * src = (const uint4 *) A.pool[w] + eo; uint4 * dst = (uint4 *) A.W[w] + so;
            for (long i = tid; i < nv; i += FT) dst[i] = src[i];
        }
    }
    // prologue: this layer's input = rmsnorm(xprev + shared + sum_j w * out_j), same order in every block
    float ss = 0.0f;
    for (int i = tid; i < D; i += FT) {
        float v = A.xprev[i];
        if (A.outprev) {
            float a = 0.0f;
            for (int c = 0; c < K; ++c) a += WGT * A.outprev[(size_t) c * D + i];
            v = (a + A.outprev[(size_t) K * D + i]) + v;
        }
        x[i] = v; ss += v * v;
    }
    if (A.outprev) {
        ss = warp_sum(ss);
        if ((tid & 31) == 0) part[tid >> 5] = ss;
        __syncthreads();
        float t = 0.0f; for (int w = 0; w < FT / 32; ++w) t += part[w];
        const float sc = rsqrtf(t / D + 1e-6f);
        for (int i = tid; i < D; i += FT) x[i] *= sc;
    }
    __syncthreads();
    if (A.xfinal) { for (int i = tid; i < D; i += FT) A.xfinal[i] = x[i]; return; }
    if (blk == 0) for (int i = tid; i < D; i += FT) A.xcur[i] = x[i];
    if (blk == K + 1) {                                                    // router logits
        const int lane = tid & 31, warp = tid >> 5;
        for (int r = warp; r < E; r += FT / 32) {
            const float * w = A.Wr + (size_t) r * D; float a = 0.0f;
            for (int i = lane; i < D; i += 32) a += w[i] * x[i];
            a = warp_sum(a); if (lane == 0) A.lg[r] = a;
        }
        return;
    }
    if (tid < DB) quant_block(x + 32 * tid, xq + tid);
    __syncthreads();                                                       // also orders the incoming copy before its use
    if (blk == K) expert_block<FS>(A.S[0], A.S[1], A.S[2], xq, act, aq, A.out + (size_t) K * D);
    else {
        const size_t o = (size_t) A.ids[blk] * BE;
        expert_block<F>(A.W[0] + o, A.W[1] + o, A.W[2] + o, xq, act, aq, A.out + (size_t) blk * D);
    }
}

// ------------------------------------------------------------------ host side
struct Trace { std::string name; std::vector<int> n, ld_e, ld_s, sel, inc, slotid; double mean = 0; std::vector<long> hist; };
static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

// loads per token-layer: pin >= 0, or drawn from the weights. Loaded experts are cold (cycled, never resident),
// victims are slots in round-robin order. Every repeat starts from slots holding experts 0..K-1.
static Trace make_trace(const std::string & name, int T, int pin, const std::vector<double> & w) {
    Trace tr; tr.name = name; tr.n.resize((size_t) T * NL); tr.ld_e.assign((size_t) T * NL * MAXLD, 0); tr.ld_s.assign((size_t) T * NL * MAXLD, 0);
    tr.sel.resize((size_t) T * NL * K); tr.inc.assign((size_t) T * NL * K, -1); tr.slotid.resize(K); tr.hist.assign(MAXLD + 1, 0);
    for (int s = 0; s < K; ++s) tr.slotid[s] = s;
    std::vector<int> slot_e((size_t) NL * K), rr(NL, 0), cyc(NL, K);
    for (int l = 0; l < NL; ++l) for (int s = 0; s < K; ++s) slot_e[(size_t) l * K + s] = s;
    double tot = 0; for (double x : w) tot += x;
    rng_state = 0x9E3779B9u; long loads = 0;
    for (int t = 0; t < T; ++t) for (int l = 0; l < NL; ++l) {
        int n = pin;
        if (pin < 0) { double u = (rnd() >> 8) * (1.0 / 16777216.0) * tot, acc = 0; n = 0; for (size_t i = 0; i < w.size(); ++i) { acc += w[i]; if (u < acc) { n = (int) i; break; } n = (int) i; } }
        if (n > MAXLD) n = MAXLD;
        const size_t tl = (size_t) t * NL + l; tr.n[tl] = n; tr.hist[n]++; loads += n;
        int * se = &slot_e[(size_t) l * K];
        for (int i = 0; i < n; ++i) {
            int e; for (;;) { e = cyc[l]; cyc[l] = (cyc[l] + 1) % E; bool res = false; for (int s = 0; s < K; ++s) res |= se[s] == e; if (!res) break; }
            const int s = rr[l]; rr[l] = (rr[l] + 1) % K;
            se[s] = e; tr.ld_e[tl * MAXLD + i] = e; tr.ld_s[tl * MAXLD + i] = s; tr.inc[tl * K + s] = e;
        }
        for (int s = 0; s < K; ++s) tr.sel[tl * K + s] = se[s];
    }
    tr.mean = (double) loads / ((double) T * NL);
    return tr;
}

template <typename T> static T * dev(const std::vector<T> & v) { T * p; CK(cudaMalloc(&p, v.size() * sizeof(T))); CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2000;
    std::vector<double> skw = {16413, 50220, 7956, 2391, 732, 186, 96, 6};      // Skliar, 1.26 B model, lambda 0.739453125 (O1-5)
    if (argc > 2) { skw.clear(); for (char * p = argv[2]; *p; ) { char * e; skw.push_back(strtod(p, &e)); p = *e == ',' ? e + 1 : e; } }
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("GPU %s; %d tokens per run, %d layers, %d of %d experts, load = %zu bytes\n", pr.name, T, NL, K, E, 3 * BE);

    // weights: host pool [layer][tensor][expert] (pinned, mapped), the same on the device (all-in-VRAM arm), slots, shared, router
    const size_t pool_bytes = (size_t) NL * 3 * E * BE;
    void * hpool; CK(cudaHostAlloc(&hpool, pool_bytes, cudaHostAllocMapped));
    { uint32_t s = 777u; uint8_t * h = (uint8_t *) hpool;
      for (size_t i = 0; i < pool_bytes; ++i) { s = s * 1664525u + 1013904223u; h[i] = (uint8_t) (s >> 24); }
      for (size_t i = 0; i < pool_bytes; i += 18) { h[i] = 0x1F; h[i + 1] = 0x25; } }        // d = 0x251F, about 0.02 as a half
    void * hdev; CK(cudaHostGetDevicePointer(&hdev, hpool, 0));
    uint8_t * dfull; CK(cudaMalloc(&dfull, pool_bytes + 64)); CK(cudaMemcpy(dfull, hpool, pool_bytes, cudaMemcpyHostToDevice));
    const size_t slot_bytes = (size_t) NL * 3 * K * BE;
    uint8_t * dslot; CK(cudaMalloc(&dslot, slot_bytes + 64));
    std::vector<uint8_t> hs((size_t) NL * 3 * BS); { uint32_t s = 4242u; for (size_t i = 0; i < hs.size(); ++i) { s = s * 1664525u + 1013904223u; hs[i] = (uint8_t) (s >> 24); } for (size_t i = 0; i < hs.size(); i += 18) { hs[i] = 0x1F; hs[i + 1] = 0x25; } }
    uint8_t * dshared = dev(hs);
    std::vector<float> hr((size_t) NL * E * D), hx0((size_t) T * D);
    for (auto & v : hr) v = ((int) (rnd() >> 16) - 32768) * (1.0f / 32768.0f) * 0.05f;
    for (auto & v : hx0) v = ((int) (rnd() >> 16) - 32768) * (1.0f / 32768.0f) * 1.7f;
    float * dWr = dev(hr), * dX0 = dev(hx0);
    auto P  = [&](const uint8_t * base, int l, int w, int per) { return base + ((size_t) l * 3 + w) * per * BE; };
    auto Sh = [&](int l, int w) { return dshared + ((size_t) l * 3 + w) * BS; };

    // scratch
    float * dxbuf, * dout, * dXF, * dlg, * dgu, * dact, * dy, * dacc, * dtmp, * dsg, * dsu, * dsa, * dso, * dsmall;
    q8_1 * dxq, * daq, * dsq;
    int * dnswp, * dswp_e, * dswp_s;
    CK(cudaMalloc(&dxbuf, (size_t) (NL + 1) * D * 4)); CK(cudaMalloc(&dout, (size_t) NL * (K + 1) * D * 4)); CK(cudaMalloc(&dXF, (size_t) T * D * 4));
    CK(cudaMalloc(&dlg, (size_t) NL * E * 4)); CK(cudaMalloc(&dgu, (size_t) 2 * K * F * 4)); CK(cudaMalloc(&dact, (size_t) K * F * 4)); CK(cudaMalloc(&dy, (size_t) K * D * 4));
    CK(cudaMalloc(&dacc, D * 4)); CK(cudaMalloc(&dtmp, D * 4)); CK(cudaMalloc(&dsg, FS * 4)); CK(cudaMalloc(&dsu, FS * 4)); CK(cudaMalloc(&dsa, FS * 4)); CK(cudaMalloc(&dso, D * 4)); CK(cudaMalloc(&dsmall, 4 * E * 4));
    CK(cudaMalloc(&dxq, DB * sizeof(q8_1))); CK(cudaMalloc(&daq, (size_t) K * (F / 32) * sizeof(q8_1))); CK(cudaMalloc(&dsq, (FS / 32) * sizeof(q8_1)));
    CK(cudaMalloc(&dnswp, (size_t) NL * 4)); CK(cudaMalloc(&dswp_e, (size_t) NL * MAXLD * 4)); CK(cudaMalloc(&dswp_s, (size_t) NL * MAXLD * 4));
    cudaStream_t cs; CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    cudaEvent_t fork[NL], ev[NL][3];
    for (int l = 0; l < NL; ++l) { CK(cudaEventCreateWithFlags(&fork[l], cudaEventDisableTiming)); for (int w = 0; w < 3; ++w) CK(cudaEventCreateWithFlags(&ev[l][w], cudaEventDisableTiming)); }

    auto reset_slots = [&]() {      // slots of every layer hold experts 0..K-1
        for (int l = 0; l < NL; ++l) for (int w = 0; w < 3; ++w) CK(cudaMemcpy((void *) P(dslot, l, w, K), P(dfull, l, w, E), (size_t) K * BE, cudaMemcpyDeviceToDevice));
        CK(cudaDeviceSynchronize());
    };
    struct DT { int * n, * ld_e, * ld_s, * sel, * inc, * slotid; };

    // ---- unfused, one token. mode 0: all in VRAM (fused gate/up as the engine's VRAM path, no copies);
    //      mode 1: streamed (slots, copies at the routing op on a second stream, unfused gate/up)
    auto unfused_token = [&](int t, const DT & d, int mode, int t0) {
        const float * x = dX0 + (size_t) t * D;
        for (int l = 0; l < NL; ++l) {
            const size_t tl = (size_t) t * NL + l;
            k_router<<<E, 32>>>(dWr + (size_t) l * E * D, x, dlg + (size_t) l * E);
            k_small<<<1, 256>>>(dlg + (size_t) l * E, dsmall, E);                                    // softmax
            k_route_replay<<<1, 32>>>(dlg + (size_t) l * E, d.n + tl, d.ld_e + tl * MAXLD, d.ld_s + tl * MAXLD, dnswp + l, dswp_e + (size_t) l * MAXLD, dswp_s + (size_t) l * MAXLD, dsmall + 2 * E);
            if (mode == 1) {
                CK(cudaEventRecord(fork[l], 0)); CK(cudaStreamWaitEvent(cs, fork[l], 0));
                for (int w = 0; w < 3; ++w) {
                    k_swapcopy<<<256, 256, 0, cs>>>((const uint4 *) P((const uint8_t *) hdev, l, w, E), (uint4 *) P(dslot, l, w, K), dswp_e + (size_t) l * MAXLD, dswp_s + (size_t) l * MAXLD, dnswp + l);
                    CK(cudaEventRecord(ev[l][w], cs));
                }
            }
            k_small<<<1, 256>>>(dsmall, dsmall + E, K);                                              // gather of the gate weights
            k_small<<<1, 256>>>(dsmall + E, dsmall, K); k_small<<<1, 256>>>(dsmall, dsmall + E, K); k_small<<<1, 256>>>(dsmall + E, dsmall, K);   // sum, clamp, divide
            const uint8_t * base = mode ? dslot : dfull; const int per = mode ? K : E;
            const int * ids = mode ? d.slotid : d.sel + tl * K;
            // shared expert, then routed experts (the engine's order with the switch off)
            k_quant<<<1, 32>>>(x, dxq, DB, 1, 0);
            if (mode == 0) {
                k_gemv_glu<<<dim3(FS, 1), 32>>>(Sh(l, 0), Sh(l, 1), 0, nullptr, dxq, dsa, FS, DB);
            } else {
                k_gemv<<<dim3(FS, 1), 32>>>(Sh(l, 0), 0, nullptr, dxq, 0, dsg, FS, DB);
                k_quant<<<1, 32>>>(x, dxq, DB, 1, 0);
                k_gemv<<<dim3(FS, 1), 32>>>(Sh(l, 1), 0, nullptr, dxq, 0, dsu, FS, DB);
                k_glu<<<2, 256>>>(dsg, dsu, dsa, FS);
            }
            k_quant<<<1, 32>>>(dsa, dsq, FS / 32, 1, 0);
            k_gemv<<<dim3(D, 1), 32>>>(Sh(l, 2), 0, nullptr, dsq, 0, dso, D, FS / 32);
            k_quant<<<1, 32>>>(x, dxq, DB, 1, 0);
            if (mode == 0) {
                k_gemv_glu<<<dim3(F, K), 32>>>(P(base, l, 0, per), P(base, l, 1, per), BE, ids, dxq, dact, F, DB);
            } else {
                CK(cudaStreamWaitEvent(0, ev[l][0], 0));
                k_gemv<<<dim3(F, K), 32>>>(P(base, l, 0, per), BE, ids, dxq, 0, dgu, F, DB);
                k_quant<<<1, 32>>>(x, dxq, DB, 1, 0);
                CK(cudaStreamWaitEvent(0, ev[l][1], 0));
                k_gemv<<<dim3(F, K), 32>>>(P(base, l, 1, per), BE, ids, dxq, 0, dgu + (size_t) K * F, F, DB);
                k_glu<<<(K * F + 255) / 256, 256>>>(dgu, dgu + (size_t) K * F, dact, K * F);
            }
            k_quant<<<1, 128>>>(dact, daq, F / 32, K, F);
            if (mode == 1) CK(cudaStreamWaitEvent(0, ev[l][2], 0));
            k_gemv<<<dim3(D, K), 32>>>(P(base, l, 2, per), BE, ids, daq, 1, dy, D, F / 32);
            k_small<<<1, 256>>>(dsmall, dsmall + E, K);                                              // experts * weights
            k_addn<<<4, 256>>>(dy, 0, 8, WGT, dacc, 1); k_addn<<<4, 256>>>(dy, 8, 8, WGT, dacc, 0); k_addn<<<4, 256>>>(dy, 16, K - 16, WGT, dacc, 0);
            k_add2<<<4, 256>>>(dacc, dso, dtmp);                                                     // + shared
            k_add2<<<4, 256>>>(dtmp, x, dacc);                                                       // + residual
            float * xn = l == NL - 1 ? dXF + (size_t) (t - t0) * D : dxbuf + (size_t) l * D;
            k_norm<<<1, 1024>>>(dacc, xn);
            x = xn;
        }
    };
    // ---- fused, one token: one launch per layer plus the final combine
    auto fused_token = [&](int t, const DT & d, int mode, int t0) {
        FusedArgs A; memset(&A, 0, sizeof A);
        for (int l = 0; l <= NL; ++l) {
            const size_t tl = (size_t) t * NL + l;
            A.xprev = l == 0 ? dX0 + (size_t) t * D : dxbuf + (size_t) (l - 1) * D;
            A.outprev = l == 0 ? nullptr : dout + (size_t) (l - 1) * (K + 1) * D;
            if (l == NL) { A.xfinal = dXF + (size_t) (t - t0) * D; k_fused<<<1, FT>>>(A); break; }
            A.xcur = dxbuf + (size_t) l * D; A.out = dout + (size_t) l * (K + 1) * D;
            const uint8_t * base = mode ? dslot : dfull; const int per = mode ? K : E;
            for (int w = 0; w < 3; ++w) { A.W[w] = P(base, l, w, per); A.pool[w] = mode ? P((const uint8_t *) hdev, l, w, E) : nullptr; A.S[w] = Sh(l, w); }
            A.ids = mode ? d.slotid : d.sel + tl * K; A.inc = d.inc + tl * K;
            A.Wr = dWr + (size_t) l * E * D; A.lg = dlg + (size_t) l * E;
            k_fused<<<K + 2, FT>>>(A);
        }
    };

    std::vector<float> ref((size_t) T * D), got((size_t) T * D);
    auto run = [&](const DT & d, int impl, int mode, int n_tok, int reps, std::vector<float> * outv) {
        std::vector<double> ts;
        for (int r = 0; r < reps; ++r) {
            if (mode == 1) reset_slots();
            CK(cudaDeviceSynchronize());
            const double a = now();
            for (int t = 0; t < n_tok; ++t) { if (impl == 0) unfused_token(t, d, mode, 0); else fused_token(t, d, mode, 0); }
            CK(cudaDeviceSynchronize());
            ts.push_back((now() - a) / n_tok * 1e6);
        }
        CK(cudaGetLastError());
        if (outv) CK(cudaMemcpy(outv->data(), dXF, (size_t) n_tok * D * 4, cudaMemcpyDeviceToHost));
        std::sort(ts.begin(), ts.end());
        struct R { double med, lo, hi; }; return R{ts[ts.size() / 2], ts.front(), ts.back()};
    };
    auto maxrel = [&](const std::vector<float> & a, const std::vector<float> & b) { double m = 0; for (size_t i = 0; i < a.size(); ++i) { const double dd = fabs((double) a[i] - b[i]) / (fabs((double) a[i]) + 1e-3); if (!(dd <= m)) m = dd; } return m; };

    std::vector<Trace> traces;
    traces.push_back(make_trace("exactly one load per layer (ours)", T, 1, {}));
    traces.push_back(make_trace("Skliar's recorded histogram", T, -1, skw));
    for (int n = 0; n <= 3; ++n) traces.push_back(make_trace("pinned " + std::to_string(n), T, n, {}));

    printf("\n%-36s %-9s %10s %10s %10s %10s  %s\n", "trace", "loads", "vram", "streamed", "vram", "streamed", "max rel diff vs unfused VRAM (unf-str / fus-vram / fus-str)");
    printf("%-36s %-9s %10s %10s %10s %10s\n", "", "per t-l", "unfused", "unfused", "fused", "fused");
    std::vector<std::vector<double>> tab;
    for (auto & tr : traces) {
        DT d{dev(tr.n), dev(tr.ld_e), dev(tr.ld_s), dev(tr.sel), dev(tr.inc), dev(tr.slotid)};
        run(d, 0, 0, 200, 1, nullptr); run(d, 0, 1, 200, 1, nullptr); run(d, 1, 0, 200, 1, nullptr); run(d, 1, 1, 200, 1, nullptr);     // warm-up
        auto a = run(d, 0, 0, T, 5, &ref);
        auto b = run(d, 0, 1, T, 5, &got); const double e1 = maxrel(ref, got);
        auto c = run(d, 1, 0, T, 5, &got); const double e2 = maxrel(ref, got);
        auto f = run(d, 1, 1, T, 5, &got); const double e3 = maxrel(ref, got);
        printf("%-36s %-9.4f %10.1f %10.1f %10.1f %10.1f  %.1e / %.1e / %.1e   [min-max us: %.1f-%.1f | %.1f-%.1f | %.1f-%.1f | %.1f-%.1f]\n", tr.name.c_str(), tr.mean,
               a.med, b.med, c.med, f.med, e1, e2, e3, a.lo, a.hi, b.lo, b.hi, c.lo, c.hi, f.lo, f.hi);
        tab.push_back({tr.mean, a.med, b.med, c.med, f.med, b.lo, b.hi, f.lo, f.hi});
        fflush(stdout);
    }
    printf("\nmicroseconds per token (13 MoE layers, no attention), median of 5 repeats of %d tokens\n", T);
    const double lb = 3.0 * BE;   // bytes per load
    for (int im = 0; im < 2; ++im) {
        const int c = im ? 4 : 2; const char * nm = im ? "fused" : "unfused";
        const double p0 = tab[2][c], p1 = tab[3][c], p2 = tab[4][c], p3 = tab[5][c];
        printf("%-8s pinned loads: 0 = %.1f, 1st +%.1f, 2nd +%.1f, 3rd +%.1f us per token (%.2f, %.2f, %.2f us per layer); a later load moves at %.1f GB/s\n",
               nm, p0, p1 - p0, p2 - p1, p3 - p2, (p1 - p0) / NL, (p2 - p1) / NL, (p3 - p2) / NL, lb / ((p3 - p2) / NL * 1e-6) / 1e9);
        const double full = p3 - p2;
        printf("%-8s share of the first load hidden (against a later load): %.0f %%\n", nm, 100.0 * (1.0 - (p1 - p0) / full));
        printf("%-8s exactly one %.1f us, Skliar-shaped (mean %.4f) %.1f us: ours against Skliar %+.2f %% (range over repeats %+.2f to %+.2f %%)\n", nm,
               tab[0][c], tab[1][0], tab[1][c], 100.0 * (tab[1][c] / tab[0][c] - 1.0),
               100.0 * (tab[1][im ? 7 : 5] / tab[0][im ? 8 : 6] - 1.0), 100.0 * (tab[1][im ? 8 : 6] / tab[0][im ? 7 : 5] - 1.0));
    }
    printf("all in VRAM: unfused %.1f us, fused %.1f us per token\n", tab[0][1], tab[0][3]);
    return 0;
}
