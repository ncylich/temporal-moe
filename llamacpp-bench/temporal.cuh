#pragma once
#include "common.cuh"

// Phase-3b UNIFIED run (env TEMPORAL_UNIFIED=1): R VRAM expert slots + 192-expert CPU pool + real swap.
// On-device residency + graph-capturable swap: registered R-slot expert tensors have their selected
// expert ids remapped to slots (and the <=1 swap/layer copy issued) before the expert GEMM.
int  ggml_cuda_temporal_unified();
// router-before-attention experiment (env TEMPORAL_ROUTER_EARLY=1)
int  ggml_cuda_temporal_router_early();
// register an R-slot expert weight tensor (keyed by its device data ptr) + its CPU pool source.
void ggml_cuda_temporal_register(const void* rslot_data, const void* pool_host, size_t bytes_per_expert,
                                 int n_expert, int R, int layer, int which /*0=gate,1=up,2=down*/);
extern "C" int ggml_cuda_temporal_unified_remap(const void* src0_data, const int32_t* ids_in, int32_t** ids_out_ptr,
                                                int n_used, int n_tokens, int stride1, cudaStream_t stream);
