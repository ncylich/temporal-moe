#include "temporal.cuh"
#include <cstdlib>
#include <cstring>

// ================= Phase-3b UNIFIED: ON-DEVICE residency + graph-capturable swap ===============
// Everything the swap needs is on-device with FIXED buffer pointers: the CPU pool is host-registered
// (mapped) so a copy KERNEL can read it over PCIe; per-layer device tables hold residency; a single
// residency kernel scans the selected ids, evicts + records swaps and updates the tables; a copy kernel
// executes the swaps by device-computed indices; the remap kernel maps ids -> slots via the device loc
// table. No host sync, no per-token graph-node change => CUDA-graph-capturable AND host-sync-free.
#include <unordered_map>
#include <vector>
struct TU {
    const uint8_t* pool_dev[3] = {nullptr,nullptr,nullptr};   // host-mapped device ptr of the CPU pool
    const uint8_t* pool_host[3]= {nullptr,nullptr,nullptr};   // host ptr of the mapped CPU pool (prefill prefetch)
    uint8_t*       slot[3]     = {nullptr,nullptr,nullptr};    // device R-slot base
    size_t         be[3]       = {0,0,0};
    int            E=0, R=0, ready=0, primed=0;   // primed: router-early prime issued this layer (capture-time)
    int *d_slot=nullptr, *d_loc=nullptr, *d_sel=nullptr;      // residency tables (device)
    int *d_swp_e=nullptr, *d_swp_s=nullptr, *d_swp_p=nullptr, *d_nswp=nullptr;  // swap list (device): expert, slot, position
    cudaEvent_t ev[3] = {nullptr,nullptr,nullptr};           // gate/up/down swap-copy done (overlap)
    cudaEvent_t fork = nullptr;                              // fork point on compute stream
    int32_t *remap_ids = nullptr; int64_t remap_words = 0;   // persistent remapped-ids buffer, shared gate/up/down
};
static std::vector<TU> g_L;
static std::unordered_map<const void*, std::pair<int,int>> g_tptr;
static cudaStream_t g_copy = nullptr;                        // shared swap-copy stream

int ggml_cuda_temporal_unified() { static int v=[](){const char*s=getenv("TEMPORAL_UNIFIED");return s?atoi(s):0;}(); return v; }
// FORCE1 caps the swap to <=1 expert/layer (emulates a trained model's temporal locality). OPT-IN only:
// it is faster (~160 tok/s) but only CORRECT when consecutive tokens route to mostly-resident experts; on
// random-locality weights it is numerically approximate (see bench notes). Default 0 = correct on any model.
static int tu_force1() { static int v=[](){const char*s=getenv("TEMPORAL_UNIFIED_FORCE1");return s?atoi(s):0;}(); return v; }
// overlap: stagger gate/up/down swap-copies on a copy stream so up-copy+down-copy hide behind the gate/up
// expert GEMMs (compute stream), each GEMM gated on its own tensor's copy. DEFAULT ON (correctness-safe:
// full swap budget, just overlapped); set TEMPORAL_UNIFIED_NOOVERLAP=1 to disable for debugging.
static int tu_overlap() { static int v=[](){ if (getenv("TEMPORAL_UNIFIED_NOOVERLAP")) return 0; const char*s=getenv("TEMPORAL_UNIFIED_OVERLAP"); return s?atoi(s):1; }(); return v; }
static int tu_nocopy() { static int v=[](){const char*s=getenv("TEMPORAL_UNIFIED_NOCOPY");return s?atoi(s):0;}(); return v; }  // decomposition: skip swap-copy (wrong output)
// Swap-rate EMULATION. The real mechanism swaps <=1 expert/layer/token; the trained models' measured
// mean_swap_rate (probe_replay e1) is p in [0,1] (shipped policy ~1.0). Emulate a global rate p on this
// big random-weight model by doing the (exactly-1) swap on the first round(p*nlayers) layers and ZERO
// swaps on the rest (all swaps are identical 840KiB cost, so total copy work = p*nlayers swaps/token
// regardless of which layers -> timing faithful). TEMPORAL_SWAP_PROB=p (unset => disabled).
static int tu_swap_prob_x1000(){ static int v=[](){const char*s=getenv("TEMPORAL_SWAP_PROB"); return s?(int)(atof(s)*1000.0+0.5):-1;}(); return v; }
// ROUTER-EARLY experiment (TEMPORAL_ROUTER_EARLY=1): the qwen3moe graph computes the router selection
// from the PRE-attention input and emits a "prime" expert op there. On that prime call, unified_remap
// issues the swap-copy on the copy stream WITHOUT waiting, so it overlaps the attention kernels; the
// real experts (post-attention) wait the already-issued copy. Use with OVERLAP=1. Read once.
int ggml_cuda_temporal_router_early(){ static int v=[](){const char*s=getenv("TEMPORAL_ROUTER_EARLY");return s?atoi(s):0;}(); return v; }
// swap-copy grid width: router-early needs a SMALL grid so the copy kernel frees SMs for attention to
// overlap it (see below). PCIe-bandwidth-bound, so a handful of blocks saturate the link. TEMPORAL_COPY_BLOCKS=n.
static int tu_copy_blocks(){ static int v=[](){const char*s=getenv("TEMPORAL_COPY_BLOCKS");return s?atoi(s):256;}(); return v; }

void ggml_cuda_temporal_register(const void* rslot_data, const void* pool_host, size_t bytes_per_expert,
                                 int n_expert, int R, int layer, int which) {
    if ((int)g_L.size() <= layer) g_L.resize(layer+1);
    TU& L = g_L[layer];
    // Allocate our OWN mapped pinned pool and copy the real expert weights in (the -ncmoe buffer may be
    // CUDA-managed, so cudaHostRegister on it is rejected -> invalid device ptr). This mapped buffer's
    // device ptr is what the swap-copy kernel reads over PCIe. One-time host-RAM duplicate of experts.
    void* mapped = nullptr;
    cudaHostAlloc(&mapped, (size_t)n_expert*bytes_per_expert, cudaHostAllocMapped);
    memcpy(mapped, pool_host, (size_t)n_expert*bytes_per_expert);
    void* pdev = nullptr;
    cudaHostGetDevicePointer(&pdev, mapped, 0);
    L.pool_dev[which] = (const uint8_t*)pdev;
    L.pool_host[which] = (const uint8_t*)mapped;
    L.slot[which] = (uint8_t*)rslot_data; L.be[which] = bytes_per_expert; L.E=n_expert; L.R=R;
    if (!L.ready) {
        std::vector<int> hs(R), hl(n_expert,-1);
        for (int s=0;s<R && s<n_expert;++s){ hs[s]=s; hl[s]=s; }
        cudaMalloc(&L.d_slot,R*sizeof(int)); cudaMemcpy(L.d_slot,hs.data(),R*sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&L.d_loc,n_expert*sizeof(int)); cudaMemcpy(L.d_loc,hl.data(),n_expert*sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&L.d_sel,n_expert*sizeof(int));
        cudaMalloc(&L.d_swp_e,R*sizeof(int)); cudaMalloc(&L.d_swp_s,R*sizeof(int)); cudaMalloc(&L.d_swp_p,R*sizeof(int)); cudaMalloc(&L.d_nswp,sizeof(int));
        for (int w=0;w<3;++w) cudaEventCreateWithFlags(&L.ev[w], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&L.fork, cudaEventDisableTiming);
        if (!g_copy) cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking);
        L.ready=1;
    }
    g_tptr[rslot_data]=std::make_pair(layer,which);
    static int announced=0; if(!announced++) fprintf(stderr,"[temporal] UNIFIED (on-device, graph-capturable) ACTIVE: R=%d + CPU pool\n", R);
}

// Residency scan (single block, blockDim threads): mark selected, evict resident-not-selected, record
// swaps, update tables. The clear+mark phases run in PARALLEL across the block; the assignment keeps the
// swap-in experts in expert-index order -> evictable slots in slot-index order (bit-identical), with a
// single monotonic evict pointer (slots only ever become non-evictable within a call, so a forward scan
// is equivalent to restarting from 0 each time). force1 caps the swap count at 1 (the real mechanism).
__global__ void k_residency(const int32_t* __restrict__ ids, int n_used, int stride1, int n_tokens,
                            int* slot, int* loc, int* sel, int E, int R, int force1,
                            int* swp_e, int* swp_s, int* swp_p, int* nswp, int32_t* remap_out) {
    if (blockIdx.x) return;
    extern __shared__ int sh_sel[];                       // sel[] in SHARED (was global d_sel) -> no global
    (void)sel;                                            // round-trips across the barriers. size E*4 bytes.
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int e=tid; e<E; e+=nt) sh_sel[e]=0;              // parallel clear
    __syncthreads();
    for (int i=tid; i<n_used*n_tokens; i+=nt) {           // parallel mark selected (set-to-1 is race-free)
        int t=i/n_used, k=i%n_used; int e=ids[(size_t)t*stride1+k]; if(e>=0&&e<E) sh_sel[e]=1;
    }
    __syncthreads();
    if (force1) {
        // FAST budget=1 path (the deploy): the whole block reads loc[]/slot[] in PARALLEL (global-load
        // latency hidden across threads) to find nominee = first selected expert not resident, and
        // victim = first evictable slot -- via atomicMin over shared, giving the SAME lowest-index result
        // the old serial scan produced (bit-identical). Only the single swap update stays on thread 0.
        __shared__ int s_nom, s_vic;
        if (tid==0){ s_nom=E; s_vic=R; }
        __syncthreads();
        for (int e=tid; e<E; e+=nt) if (sh_sel[e] && loc[e]<0) atomicMin(&s_nom, e);            // parallel find nominee
        for (int s=tid; s<R; s+=nt){ int re=slot[s]; if (re<0 || !sh_sel[re]) atomicMin(&s_vic, s); } // parallel find victim
        __syncthreads();
        if (tid==0){
            int ns=0;
            if (s_nom<E && s_vic<R){
                int old=slot[s_vic]; if(old>=0) loc[old]=-1;
                slot[s_vic]=s_nom; loc[s_nom]=s_vic;
                int pos=-1; for (int k=0;k<n_used;++k){ if(ids[k]==s_nom){pos=k;break;} }
                swp_e[0]=s_nom; swp_s[0]=s_vic; swp_p[0]=pos; ns=1;
            }
            *nswp=ns;
        }
    } else if (tid==0) {                                  // budget=R fallback (rare): original serial scan
        int ns=0, budget = R, sp=0;                       // sp = monotonic evictable-slot pointer
        for (int e=0;e<E && budget>0;++e){
            if (!sh_sel[e] || loc[e]>=0) continue;
            int victim=-1;
            while (sp<R){ int re=slot[sp]; if(re<0 || !sh_sel[re]){victim=sp;break;} sp++; }
            if (victim<0) break;
            sp++;                                         // consume this slot
            int old=slot[victim]; if(old>=0) loc[old]=-1;
            slot[victim]=e; loc[e]=victim;
            int pos=-1; for (int k=0;k<n_used;++k){ if(ids[k]==e){pos=k;break;} }
            swp_e[ns]=e; swp_s[ns]=victim; swp_p[ns]=pos; ns++; budget--;
        }
        *nswp=ns;
    }
    // FUSED remap: loc[] is now updated -> write the id->slot remap in the SAME kernel (drops the separate
    // k_remap_loc launch + its dependency handoff). remap_out==nullptr (router-early prime) skips it.
    __syncthreads();
    if (remap_out) {
        for (int i=tid; i<n_used*n_tokens; i+=nt) {
            int t=i/n_used, k=i%n_used; size_t off=(size_t)t*stride1+k;
            int e=ids[off]; remap_out[off] = loc[e] >= 0 ? loc[e] : 0;
        }
    }
}
// device-indexed swap copy: read host-mapped pool[expert] -> device slot[victim], for each recorded swap.
// Vectorized uint4 (16B/thread) path: host-mapped PCIe reads want wide transactions; byte-per-thread
// starves the link. Expert byte-sizes are quant-row multiples (16B-aligned) so nv=be/16 is exact and the
// per-expert base (swp*be) stays 16-aligned. Tail-byte fallback kept for the (unused) non-16-aligned case.
static __device__ inline void swapcopy_one(const uint8_t* pool_dev, uint8_t* slotbuf, size_t be, int ns,
                                           const int* swp_e, const int* swp_s) {
    long nv = (long)be >> 4;                        // 16-byte vectors per expert
    long total = (long)ns * nv;
    const uint4* psrc = (const uint4*)pool_dev; uint4* pdst = (uint4*)slotbuf;
    for (long i = (long)blockIdx.x*blockDim.x+threadIdx.x; i < total; i += (long)gridDim.x*blockDim.x) {
        int  j   = (int)(i / nv);
        long off = i - (long)j*nv;
        pdst[(size_t)swp_s[j]*nv + off] = psrc[(size_t)swp_e[j]*nv + off];
    }
}
__global__ void k_swapcopy(const uint8_t* __restrict__ pool_dev, uint8_t* __restrict__ slotbuf, size_t be,
                           const int* __restrict__ swp_e, const int* __restrict__ swp_s, const int* __restrict__ nswp) {
    int ns = *nswp; if (ns<=0) return;
    swapcopy_one(pool_dev, slotbuf, be, ns, swp_e, swp_s);
}
// remap ids[k] -> loc[ids[k]] (device residency table).
__global__ void k_remap_loc(const int32_t* __restrict__ ids_in, int32_t* __restrict__ ids_out,
                            const int* __restrict__ loc, int n_used, int n_tokens, int stride1) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x, total=n_used*n_tokens; if(idx>=total) return;
    int t=idx/n_used, k=idx%n_used, off=t*stride1+k; int e=ids_in[off];
    ids_out[off] = loc[e] >= 0 ? loc[e] : 0;
}

extern "C" int ggml_cuda_temporal_unified_remap(const void* src0_data, const int32_t* ids_in, int32_t** ids_out_ptr,
                                                int n_used, int n_tokens, int stride1, cudaStream_t stream) {
    auto it = g_tptr.find(src0_data); if (it==g_tptr.end()) return 0;
    TU& L = g_L[it->second.first]; int which = it->second.second;
    const int overlap = tu_overlap();
    const int router_early = ggml_cuda_temporal_router_early();
    // gate/up/down share the SAME selection + residency table -> the expert->slot remap is IDENTICAL for
    // all three. Compute it ONCE at the gate op into a persistent per-layer buffer; up/down reuse it (drops
    // 2 of every 3 remap kernels, 96/token). Persistent (fixed-addr) buffer keeps it graph-capturable.
    int64_t need = (int64_t)stride1 * n_tokens;
    if (L.remap_words < need) { if (L.remap_ids) cudaFree(L.remap_ids); cudaMalloc(&L.remap_ids, need*sizeof(int32_t)); L.remap_words = need; }
    // swap-rate emulation: this layer does the 1-swap iff layer < round(p*nlayers); else 0 swaps.
    const int sp1000 = tu_swap_prob_x1000();
    const bool emulate = sp1000 >= 0;
    bool layer_swaps = true;
    if (emulate) {
        const int nl = (int)g_L.size();
        const int n_swap_layers = (sp1000 * nl + 500) / 1000;   // round(p*nlayers)
        layer_swaps = it->second.first < n_swap_layers;
    }
    // ROUTER-EARLY: the FIRST which==0 call for a layer is the PRIME (emitted from the pre-attention
    // router). It issues the swap-copy but does NOT wait, so the copy overlaps the attention kernels; the
    // real gate/up/down (post-attention) wait the already-issued copy. Non-router-early: issue at the real
    // gate as before, and every op waits its own copy.
    const bool is_prime = router_early && which==0 && !L.primed;
    const bool do_issue = router_early ? is_prime : (which==0);
    if (do_issue) {
        // on-device residency scan -> records swaps; the slot indices are shared across gate/up/down, so
        // once residency has run we can issue ALL THREE tensors' swap-copies.
        k_residency<<<1,256,(size_t)L.E*sizeof(int),stream>>>(ids_in, n_used, stride1, n_tokens,
                                      L.d_slot, L.d_loc, L.d_sel, L.E, L.R, (emulate ? 1 : tu_force1()),
                                      L.d_swp_e, L.d_swp_s, L.d_swp_p, L.d_nswp,
                                      is_prime ? nullptr : L.remap_ids);   // fused remap (prime skips it)
        // emulated non-swap layer: zero the swap count so every k_swapcopy early-returns (no bytes moved).
        if (emulate && !layer_swaps) cudaMemsetAsync(L.d_nswp, 0, sizeof(int), stream);
        if (overlap) {
            // fork a copy stream from compute after residency, then queue the gate/up/down swap-copies on
            // it. fork + per-op join are ordinary events inside the captured region -> graph-capturable.
            cudaEventRecord(L.fork, stream);
            cudaStreamWaitEvent(g_copy, L.fork, 0);
            for (int w = 0; w < 3; ++w) {   // per-tensor copy + event -> copy-pipelining (gate GEMM waits
                k_swapcopy<<<tu_copy_blocks(),256,0,g_copy>>>(L.pool_dev[w], L.slot[w], L.be[w], L.d_swp_e, L.d_swp_s, L.d_nswp);
                cudaEventRecord(L.ev[w], g_copy);   // only its own copy; up/down hide behind gate/up GEMVs)
            }
        }
    }
    // WAIT: the prime does NOT wait (lets the copy overlap attention); every real op waits its own copy.
    if (overlap) {
        if (!is_prime) cudaStreamWaitEvent(stream, L.ev[which], 0);   // this op waits its own copy (pipelined)
    } else if (!tu_nocopy()) {
        k_swapcopy<<<tu_copy_blocks(),256,0,stream>>>(L.pool_dev[which], L.slot[which], L.be[which], L.d_swp_e, L.d_swp_s, L.d_nswp);
    }
    if (router_early && which==0) L.primed = is_prime ? 1 : 0;   // prime sets it; real gate clears it (per capture)
    // remap is FUSED into k_residency at do_issue. Only the router-early REAL gate (which==0, not the prime,
    // and not itself a do_issue) needs a standalone remap of its post-attention ids; up/down reuse it.
    if (which == 0 && !is_prime && !do_issue) {
        int total=n_used*n_tokens, nb=(total+255)/256;
        k_remap_loc<<<nb,256,0,stream>>>(ids_in, L.remap_ids, L.d_loc, n_used, n_tokens, stride1);
    }
    *ids_out_ptr = L.remap_ids;
    return is_prime ? 2 : 1;   // 2 = prime: caller skips the expert GEMM (copy already issued)
}

// ===== EXPERT-MAJOR STREAMING PREFILL support =====
int ggml_cuda_temporal_slotinfo(const void* src0_data, void** slot_base, const void** pool_host,
                                size_t* bytes_per_expert, int* n_expert, int* R) {
    auto it = g_tptr.find(src0_data); if (it==g_tptr.end()) return 0;
    TU& L = g_L[it->second.first]; int which = it->second.second;
    if (slot_base)       *slot_base       = L.slot[which];
    if (pool_host)       *pool_host       = L.pool_host[which];
    if (bytes_per_expert)*bytes_per_expert= L.be[which];
    if (n_expert)        *n_expert        = L.E;
    if (R)               *R               = L.R;
    return 1;
}
cudaStream_t ggml_cuda_temporal_copy_stream() {
    if (!g_copy) cudaStreamCreateWithFlags(&g_copy, cudaStreamNonBlocking);
    return g_copy;
}
