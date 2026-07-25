// ram_probe: sustained RAM streaming rate with T threads (the rate Q4 GEMVs see).
// Each thread NEON-sums its own large anonymous buffer repeatedly; reports aggregate
// GB/s. Matches decode's access pattern (streaming weight reads, no reuse).
//   ./ram_probe <threads> <mib_per_thread> <passes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

static uint64_t now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

struct arg_t { size_t bytes; int passes; uint64_t sink; };

static void * worker(void * p) {
    arg_t * a = (arg_t *) p;
    uint8_t * buf = (uint8_t *) malloc(a->bytes);
    memset(buf, 0x5a, a->bytes);            // fault in
    uint64x2_t acc = vdupq_n_u64(0);
    for (int pass = 0; pass < a->passes; pass++) {
        const uint8_t * s = buf;
        for (size_t i = 0; i + 64 <= a->bytes; i += 64) {
            uint8x16_t v0 = vld1q_u8(s + i);
            uint8x16_t v1 = vld1q_u8(s + i + 16);
            uint8x16_t v2 = vld1q_u8(s + i + 32);
            uint8x16_t v3 = vld1q_u8(s + i + 48);
            acc = vaddq_u64(acc, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(veorq_u8(v0, v2)))));
            acc = vaddq_u64(acc, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(veorq_u8(v1, v3)))));
        }
    }
    a->sink = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
    free(buf);
    return NULL;
}

int main(int argc, char ** argv) {
    int    T      = argc > 1 ? atoi(argv[1]) : 4;
    size_t mib    = argc > 2 ? (size_t) atol(argv[2]) : 256;
    int    passes = argc > 3 ? atoi(argv[3]) : 8;
    pthread_t th[16]; arg_t args[16];
    uint64_t t0 = now_ns();
    for (int i = 0; i < T; i++) {
        args[i] = { mib * 1048576, passes, 0 };
        pthread_create(&th[i], NULL, worker, &args[i]);
    }
    uint64_t sink = 0;
    for (int i = 0; i < T; i++) { pthread_join(th[i], NULL); sink ^= args[i].sink; }
    uint64_t wall = now_ns() - t0;
    double bytes = (double) T * mib * 1048576 * passes;
    printf("threads=%d buf_mib=%zu passes=%d agg_gbps=%.2f (sink %llu)\n",
           T, mib, passes, bytes / wall, (unsigned long long) sink);
    return 0;
}
