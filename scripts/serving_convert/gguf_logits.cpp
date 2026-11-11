// Step 3 check: run a GGUF through the engine's own libllama on given token ids and dump logits.
//
//   gguf_logits logits   model.gguf tokens.i32 n_tokens out.f32 [fa]  -> float32 [n_tokens, n_vocab]
//   gguf_logits tokenize model.gguf text.txt out.i32               -> int32 ids (no BOS, specials parsed)
//   gguf_logits nll model.gguf rows.i32 n_rows seq out.f32 [fa] [kvf16] [threads=N]
//                                                                  -> float32 [n_rows, seq] per-token NLL
//
// CPU only (n_gpu_layers = 0) unless GGUF_LOGITS_GPU=1, one decode call over the whole sequence, so every position goes
// through the stock top-k router. Build line is in README.md.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: see header comment\n");
        return 2;
    }
    const std::string mode = argv[1];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    // GGUF_LOGITS_GPU=1: offload everything to the default GPU backend (the CUDA routing check)
    const bool gpu = getenv("GGUF_LOGITS_GPU") != nullptr;
    mp.n_gpu_layers = gpu ? 999 : 0;
    ggml_backend_dev_t no_devices[] = { nullptr };   // keep Metal out entirely
    if (!gpu) mp.devices = no_devices;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (mode == "tokenize") {
        std::ifstream in(argv[3], std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<llama_token> ids(text.size() + 16);
        int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        FILE * f = fopen(argv[4], "wb");
        fwrite(ids.data(), sizeof(llama_token), n, f);
        fclose(f);
        fprintf(stderr, "tokenized %zu bytes -> %d tokens\n", text.size(), n);
        return 0;
    }

    if (mode == "nll") {
        // gguf_logits nll model.gguf rows.i32 n_rows seq out.f32 [fa] [kvf16] [threads=N]
        // rows.i32 holds n_rows x (seq+1) ids. Each row is one sequence starting at position 0
        // (the KV cache is cleared between rows); writes float32 [n_rows, seq] per-token
        // negative log-likelihoods of ids[1..seq] given the prefix.
        if (argc < 7) { fprintf(stderr, "nll mode: rows.i32 n_rows seq out.f32\n"); return 2; }
        const int n_rows = atoi(argv[4]);
        const int seq = atoi(argv[5]);
        bool fa = false, kvf16 = false; int nth = 8;
        for (int i = 7; i < argc; i++) {
            const std::string o = argv[i];
            if (o == "fa") fa = true;
            else if (o == "kvf16") kvf16 = true;
            else if (o.rfind("threads=", 0) == 0) nth = atoi(o.c_str() + 8);
        }
        std::vector<llama_token> rows((size_t) n_rows * (seq + 1));
        FILE * f = fopen(argv[3], "rb");
        if (!f || fread(rows.data(), sizeof(llama_token), rows.size(), f) != rows.size()) { fprintf(stderr, "row read failed\n"); return 1; }
        fclose(f);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = seq; cp.n_batch = seq; cp.n_ubatch = seq;
        cp.offload_kqv = gpu; cp.op_offload = gpu;
        cp.type_k = kvf16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.type_v = kvf16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.n_threads = cp.n_threads_batch = nth;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }
        llama_batch batch = llama_batch_init(seq, 0, 1);
        FILE * o = fopen(argv[6], "wb");
        std::vector<float> nll(seq);
        double tot = 0.0;
        for (int r = 0; r < n_rows; r++) {
            llama_memory_clear(llama_get_memory(ctx), true);
            const llama_token * ids = rows.data() + (size_t) r * (seq + 1);
            for (int i = 0; i < seq; i++) {
                batch.token[i] = ids[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0; batch.logits[i] = 1;
            }
            batch.n_tokens = seq;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            double rs = 0.0;
            for (int i = 0; i < seq; i++) {
                const float * lg = llama_get_logits_ith(ctx, i);
                float mx = lg[0];
                for (int v = 1; v < n_vocab; v++) mx = lg[v] > mx ? lg[v] : mx;
                double se = 0.0;
                for (int v = 0; v < n_vocab; v++) se += std::exp((double) lg[v] - mx);
                nll[i] = (float) (std::log(se) + mx - lg[ids[i + 1]]);
                rs += nll[i];
            }
            fwrite(nll.data(), sizeof(float), seq, o); fflush(o);
            tot += rs;
            fprintf(stderr, "row %d/%d nll %.5f running %.5f\n", r + 1, n_rows, rs / seq, tot / ((double) (r + 1) * seq));
        }
        fclose(o);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }

    if (argc < 6) { fprintf(stderr, "logits mode needs n_tokens and out path\n"); return 2; }
    const int n = atoi(argv[4]);
    std::vector<llama_token> ids(n);
    FILE * f = fopen(argv[3], "rb");
    if (!f || fread(ids.data(), sizeof(llama_token), n, f) != (size_t) n) { fprintf(stderr, "token read failed\n"); return 1; }
    fclose(f);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.type_k = GGML_TYPE_F32;                        // default F16 KV cache adds rounding noise
    cp.type_v = GGML_TYPE_F32;
    // The flash-attention path casts K and V to F16 regardless of the cache type, which costs
    // about 1e-4 relative accuracy. Off by default so the comparison is exact; pass "fa" as a
    // 6th argument to measure the engine's default behaviour.
    const bool fa = argc > 6 && std::string(argv[6]) == "fa";
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    FILE * o = fopen(argv[5], "wb");
    for (int i = 0; i < n; i++) {
        fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, o);
    }
    fclose(o);
    fprintf(stderr, "wrote %d x %d logits\n", n, n_vocab);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
