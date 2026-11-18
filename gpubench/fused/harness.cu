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
//   fused    four launches per layer (gate and up products of all experts plus the router; SwiGLU; down projection
//            plus the sum; norm), every weight read one step deep; an incoming expert is moved from the host pool
//            to its slot by the same blocks that then compute it, next to the resident experts' rows
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
#include <functional>
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
__global__ void k_copy1(const uint4 * __restrict__ src, uint4 * __restrict__ dst, long nv) {
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < nv; i += (long) gridDim.x * blockDim.x) dst[i] = src[i];
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

// ------------------------------------------------------------------ fused: four launches per layer
// A   gate and up products of every routed expert and of the shared expert, and the router logits: one block per
//     32 rows of one tensor (one warp per row, one lane per Q4_0 block), so every weight read is one step deep.
//     The 12 blocks of an incoming expert first move their sixth of the gate or up tensor from the host-mapped
//     pool to the slot, one 16-byte read per thread, then compute like the others: the load starts with the
//     launch and runs next to the resident experts' rows. An incoming expert's blocks come first in the grid.
// A2  SwiGLU and activation quantisation.
// B   down projection, weighted sum over experts, shared expert, residual: one block per output row, one
//     thread per Q4_0 block. An incoming expert's down row is moved from the host pool by extra threads of the
//     same block while the other experts' products are computed, and its own products follow.
// C   rmsnorm and the next layer's input quantisation (one block).
// The activations of an expert, and then every output row, have to be complete before the next stage can use
// them, and blocks of one launch cannot wait for each other; that is why this is four launches and not one
// (the ledger has the one-launch and three-launch versions that were tried first).
struct FusedA {
    const q8_1 * xq; const float * x;
    const uint8_t * W[3]; const int * ids; const uint8_t * pool[3]; const int * inc;     // gate, up, (down: unused here; with two-entry
                                        // arrays this kernel ran 5x slower for a reason not found, see the ledger)
    const uint8_t * SG, * SU; const float * Wr;
    float * gu, * sg, * su, * lg;       // gate and up products [2][K][F], shared [FS] each, router logits
    const int * perm;                   // routed positions, incoming ones first, so their blocks are scheduled first
};
constexpr int GR = F / 32, NA_ROUTED = K * GR * 2, NA_SHARED = (FS / 32) * 2, NA = NA_ROUTED + NA_SHARED + E / 32;
__global__ void k_fused_a(FusedA A) {
    const int lane = threadIdx.x & 31, row = threadIdx.x >> 5, blk = blockIdx.x;
    if (blk >= NA_ROUTED + NA_SHARED) {                                    // router rows
        const int r = (blk - NA_ROUTED - NA_SHARED) * 32 + row;
        const float * w = A.Wr + (size_t) r * D; float a = 0.0f;
#pragma unroll
        for (int i = 0; i < D / 32; ++i) a += w[lane + 32 * i] * A.x[lane + 32 * i];
        a = warp_sum(a); if (lane == 0) A.lg[r] = a;
        return;
    }
    if (blk >= NA_ROUTED) {                                                // shared expert: gate or up, 32 rows
        const int sb = blk - NA_ROUTED, w = sb & 1, r = (sb >> 1) * 32 + row; float a = 0.0f;
        if (lane < DB) { const q4_0 q = ((const q4_0 *) (w ? A.SU : A.SG))[(size_t) r * DB + lane]; a = dot_block(&q, A.xq + lane); }
        a = warp_sum(a); if (lane == 0) (w ? A.su : A.sg)[r] = a;
        return;
    }
    // routed expert c, tensor w (0 gate, 1 up), sixth g of the tensor (32 rows)
    const int w = blk % 2, g = (blk / 2) % GR, c = A.perm[blk / (2 * GR)];
    constexpr int NV = (int) (BE / 16 / GR);                               // 900 16-byte vectors per sixth
    const bool incoming = A.pool[0] && A.inc[c] >= 0;
    if (incoming) {                                                        // host pool -> slot, one read per thread
        if (threadIdx.x < NV) {
            const size_t o = (size_t) g * NV + threadIdx.x;
            ((uint4 *) A.W[w])[(size_t) A.ids[c] * (BE / 16) + o] = ((const uint4 *) A.pool[w])[(size_t) A.inc[c] * (BE / 16) + o];
        }
        __syncthreads();
    }
    const int r = g * 32 + row; float a = 0.0f;
    if (lane < DB) { const q4_0 q = ((const q4_0 *) A.W[w])[(size_t) A.ids[c] * (BE / 18) + (size_t) r * DB + lane]; a = dot_block(&q, A.xq + lane); }
    a = warp_sum(a);
    if (lane == 0) A.gu[((size_t) w * K + c) * F + r] = a;
}
// A2: SwiGLU and activation quantisation, one warp per Q8_1 block (108 routed, 12 shared)
__global__ void k_fused_a2(const float * __restrict__ gu, const float * __restrict__ sg, const float * __restrict__ su, q8_1 * __restrict__ aq, q8_1 * __restrict__ aqs) {
    __shared__ float act[32];
    const int q = blockIdx.x, i = threadIdx.x;
    if (q < K * GR) act[i] = silu(gu[(size_t) q * 32 + i]) * gu[(size_t) K * F + (size_t) q * 32 + i];
    else            act[i] = silu(sg[(q - K * GR) * 32 + i]) * su[(q - K * GR) * 32 + i];
    __syncthreads();
    if (i == 0) quant_block(act, q < K * GR ? aq + q : aqs + (q - K * GR));
}
struct FusedB {
    const q8_1 * aq, * aqs; const float * x;
    const uint8_t * Dn; const int * ids; const uint8_t * poolD; const int * inc; const int * perm; const int * ninc; const uint8_t * SD;
    float * y;
};
constexpr int NQ = K * (F / 32) + FS / 32;                                 // Q4_0 blocks one output row reads: 108 + 12
constexpr int BC = 128, BT = BC + 32;                                      // compute threads, then 8 copy threads for each of 4 incoming experts
__global__ void k_fused_b(FusedB B) {                                      // one block per output row, one thread per Q4_0 block
    __shared__ float v[BC]; __shared__ float part[K + 1];
    const int i = threadIdx.x, r = blockIdx.x;
    constexpr int RB = (F / 32) * 18;                                      // bytes of one down row (108)
    float a = 0.0f; bool late = false;
    if (i < K * (F / 32)) {
        const int c = i / (F / 32), b = i % (F / 32);
        late = B.poolD && B.inc[c] >= 0;                                   // an incoming expert's row is computed after its copy
        if (!late) { const q4_0 w = ((const q4_0 *) B.Dn)[(size_t) B.ids[c] * (BE / 18) + (size_t) r * (F / 32) + b]; a = dot_block(&w, B.aq + i); }
    } else if (i < NQ) {
        const int b = i - K * (F / 32);
        const q4_0 w = ((const q4_0 *) B.SD)[(size_t) r * (FS / 32) + b];
        a = dot_block(&w, B.aqs + b);
    } else if (i >= BC && B.poolD) {
        // copy threads: the 16-byte vectors covering row r of an incoming expert's down tensor, host pool -> slot,
        // one read per thread. Vectors on a row boundary are written by both neighbours with the same bytes.
        // (a fifth and further incoming expert is taken by the same threads in a second round)
        const int j = i - BC, vi = j % 8, n = *B.ninc;
        const size_t first = (size_t) r * RB / 16, last = ((size_t) r * RB + RB - 1) / 16;
        for (int k = j / 8; k < n; k += 4) {
            const int c = B.perm[k];
            if (first + vi <= last) ((uint4 *) B.Dn)[(size_t) B.ids[c] * (BE / 16) + first + vi] = ((const uint4 *) B.poolD)[(size_t) B.inc[c] * (BE / 16) + first + vi];
        }
    }
    __syncthreads();
    if (late) { const int c = i / (F / 32), b = i % (F / 32); const q4_0 w = ((const q4_0 *) B.Dn)[(size_t) B.ids[c] * (BE / 18) + (size_t) r * (F / 32) + b]; a = dot_block(&w, B.aq + i); }
    if (i < BC) v[i] = a;
    __syncthreads();
    // per-expert sums in the order a 32-lane xor-shuffle reduction gives (the reference kernel's order)
    if (i < K) { const float * l = v + i * 6; part[i] = ((l[0] + l[4]) + l[2]) + ((l[1] + l[5]) + l[3]); }
    else if (i == K) { const float * l = v + K * 6; part[K] = (((l[0] + l[8]) + l[4]) + ((l[2] + l[10]) + l[6])) + (((l[1] + l[9]) + l[5]) + ((l[3] + l[11]) + l[7])); }
    __syncthreads();
    if (i == 0) {
        float s2 = 0.0f;
        for (int c = 0; c < K; ++c) s2 += WGT * part[c];
        B.y[r] = (s2 + part[K]) + B.x[r];
    }
}
__global__ void k_fused_c(const float * __restrict__ y, float * __restrict__ xout, q8_1 * __restrict__ xq) {   // one block of 1024 threads
    __shared__ float part[32]; __shared__ float xs[D + 32];
    const int i = threadIdx.x; float v = i < D ? y[i] : 0.0f;
    float s = warp_sum(v * v);
    if ((i & 31) == 0) part[i >> 5] = s;
    __syncthreads();
    float t = 0.0f; for (int w = 0; w < 32; ++w) t += part[w];
    if (i < D) { const float o = v * rsqrtf(t / D + 1e-6f); xs[i] = o; xout[i] = o; }
    __syncthreads();
    if (i < DB) quant_block(xs + 32 * i, xq + i);
}

// ------------------------------------------------------------------ host side
struct Trace { std::string name; std::vector<int> n, ld_e, ld_s, sel, inc, slotid, perm; double mean = 0; std::vector<long> hist; };
static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

// loads per token-layer: pin >= 0, or drawn from the weights. Loaded experts are cold (cycled, never resident),
// victims are slots in round-robin order. Every repeat starts from slots holding experts 0..K-1.
static Trace make_trace(const std::string & name, int T, int pin, const std::vector<double> & w) {
    Trace tr; tr.name = name; tr.n.resize((size_t) T * NL); tr.ld_e.assign((size_t) T * NL * MAXLD, 0); tr.ld_s.assign((size_t) T * NL * MAXLD, 0);
    tr.sel.resize((size_t) T * NL * K); tr.perm.resize((size_t) T * NL * K); tr.inc.assign((size_t) T * NL * K, -1); tr.slotid.resize(K); tr.hist.assign(MAXLD + 1, 0);
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
        { int k = 0; for (int s = 0; s < K; ++s) if (tr.inc[tl * K + s] >= 0) tr.perm[tl * K + k++] = s;       // incoming positions first
          for (int s = 0; s < K; ++s) if (tr.inc[tl * K + s] < 0) tr.perm[tl * K + k++] = s; }
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
    struct DT { int * n, * ld_e, * ld_s, * sel, * inc, * slotid, * perm; };

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
    // ---- fused, one token: one input quantisation, then four launches per layer
    auto fused_token = [&](int t, const DT & d, int mode, int t0) {
        const float * x = dX0 + (size_t) t * D;
        k_quant<<<1, 32>>>(x, dxq, DB, 1, 0);
        for (int l = 0; l < NL; ++l) {
            const size_t tl = (size_t) t * NL + l;
            const uint8_t * base = mode ? dslot : dfull; const int per = mode ? K : E;
            const int * ids = mode ? d.slotid : d.sel + tl * K;
            FusedA A{dxq, x, {P(base, l, 0, per), P(base, l, 1, per), P(base, l, 2, per)}, ids,
                     {mode ? P((const uint8_t *) hdev, l, 0, E) : nullptr, mode ? P((const uint8_t *) hdev, l, 1, E) : nullptr, mode ? P((const uint8_t *) hdev, l, 2, E) : nullptr},
                     d.inc + tl * K, Sh(l, 0), Sh(l, 1), dWr + (size_t) l * E * D, dgu, dsg, dsu, dlg + (size_t) l * E, d.perm + tl * K};
            k_fused_a<<<NA, 1024>>>(A);
            k_fused_a2<<<K * GR + FS / 32, 32>>>(dgu, dsg, dsu, daq, dsq);
            FusedB B{daq, dsq, x, P(base, l, 2, per), ids, mode ? P((const uint8_t *) hdev, l, 2, E) : nullptr, d.inc + tl * K, d.perm + tl * K, d.n + tl, Sh(l, 2), dacc};
            k_fused_b<<<D, BT>>>(B);
            float * xn = l == NL - 1 ? dXF + (size_t) (t - t0) * D : dxbuf + (size_t) l * D;
            k_fused_c<<<1, 1024>>>(dacc, xn, dxq);
            x = xn;
        }
    };

    // ---- each fused launch on its own, back to back (argv[3] = "parts"): where a layer's time goes
    if (argc > 3 && !strcmp(argv[3], "parts")) {
        Trace tr = make_trace("parts", 64, 0, {});
        DT d{dev(tr.n), dev(tr.ld_e), dev(tr.ld_s), dev(tr.sel), dev(tr.inc), dev(tr.slotid), dev(tr.perm)};
        reset_slots();
        k_quant<<<1, 32>>>(dX0, dxq, DB, 1, 0);
        std::vector<int> hinc(K, -1); int * dinc = dev(hinc);
        auto tm = [&](const char * what, int reps, const std::function<void(int)> & f) {
            std::vector<double> ts;
            for (int k = 0; k < 5; ++k) { CK(cudaDeviceSynchronize()); const double a = now(); for (int i = 0; i < reps; ++i) f(i); CK(cudaDeviceSynchronize()); ts.push_back((now() - a) / reps * 1e6); }
            std::sort(ts.begin(), ts.end()); printf("  %-58s %7.2f us\n", what, ts[2]);
        };
        printf("\neach launch alone, back to back, layer 0 (median of 5 x 3000)\n");
        tm("empty launch", 3000, [&](int) { k_small<<<1, 32>>>(dsmall, dsmall + E, 1); });
        for (int n = 0; n <= 3; ++n) {
            // n incoming positions: the expert id changes every launch so the host pages read are cold
            auto setinc = [&](int i) { for (int s = 0; s < K; ++s) hinc[s] = s < n ? K + ((i * 3 + s) % (E - K)) : -1; CK(cudaMemcpyAsync(dinc, hinc.data(), K * sizeof(int), cudaMemcpyHostToDevice, 0)); };
            char b[96];
            FusedA A{dxq, dX0, {P(dslot, 0, 0, K), P(dslot, 0, 1, K), P(dslot, 0, 2, K)}, d.slotid,
                     {P((const uint8_t *) hdev, 0, 0, E), P((const uint8_t *) hdev, 0, 1, E), P((const uint8_t *) hdev, 0, 2, E)}, dinc, Sh(0, 0), Sh(0, 1), dWr, dgu, dsg, dsu, dlg, d.slotid};
            std::vector<int> hn(1, n); int * dn1 = dev(hn);
            FusedB B{daq, dsq, dX0, P(dslot, 0, 2, K), d.slotid, P((const uint8_t *) hdev, 0, 2, E), dinc, d.slotid, dn1, Sh(0, 2), dacc};
            snprintf(b, sizeof b, "A (gate/up/SwiGLU, router), %d incoming", n); tm(b, 3000, [&](int i) { setinc(i); k_fused_a<<<NA, 1024>>>(A); });
            snprintf(b, sizeof b, "B (down, sum), %d incoming", n);              tm(b, 3000, [&](int i) { setinc(i); k_fused_b<<<D, BT>>>(B); });
            snprintf(b, sizeof b, "the id upload alone (pageable memcpy), %d", n); tm(b, 3000, [&](int i) { setinc(i); });
        }
        tm("A2 (SwiGLU, quantise activations)", 3000, [&](int) { k_fused_a2<<<K * GR + FS / 32, 32>>>(dgu, dsg, dsu, daq, dsq); });
        tm("C (norm, quantise)", 3000, [&](int) { k_fused_c<<<1, 1024>>>(dacc, dxbuf, dxq); });
        for (int w = 0; w < 3; ++w) { char b[96]; snprintf(b, sizeof b, "engine-style copy kernel, one tensor of one expert (%d)", w);
            tm(b, 3000, [&](int i) { k_copy1<<<256, 256>>>((const uint4 *) (P((const uint8_t *) hdev, 0, w, E) + (size_t) (K + (i * 3) % (E - K)) * BE), (uint4 *) P(dslot, 0, w, K), (long) (BE / 16)); }); }
        tm("one copy kernel for a whole load (3 tensors)", 3000, [&](int i) { k_copy1<<<256, 256>>>((const uint4 *) (P((const uint8_t *) hdev, 0, 0, E) + (size_t) ((i * 7) % (E - 3)) * BE), (uint4 *) P(dslot, 0, 0, K), (long) (3 * BE / 16)); });
        return 0;
    }
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
        DT d{dev(tr.n), dev(tr.ld_e), dev(tr.ld_s), dev(tr.sel), dev(tr.inc), dev(tr.slotid), dev(tr.perm)};
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
