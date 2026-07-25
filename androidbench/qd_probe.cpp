// qd_probe: does UFS random-read throughput scale with queue depth?
// Reads N random 216 KiB (or --sz) O_DIRECT chunks from a file using T threads,
// each thread its own fd. Reports aggregate GB/s and per-read latency.
//   ./qd_probe <file> <threads> <reads_per_thread> [chunk_bytes]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>

static uint64_t now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

struct arg_t {
    const char * path;
    size_t file_sz, chunk;
    int    n_reads;
    uint64_t seed, total_ns_sum;   // sum of per-read latencies
    int    ok;
};

static void * worker(void * p) {
    arg_t * a = (arg_t *) p;
    int fd = open(a->path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); a->ok = 0; return NULL; }
    uint8_t * buf;
    if (posix_memalign((void **) &buf, 4096, a->chunk)) { a->ok = 0; return NULL; }
    uint64_t rng = a->seed | 1;
    size_t span = (a->file_sz - a->chunk) & ~(size_t) 4095;
    for (int i = 0; i < a->n_reads; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        off_t off = (off_t)((rng % span) & ~(uint64_t) 4095);
        uint64_t t0 = now_ns();
        size_t left = a->chunk; uint8_t * bp = buf; off_t fo = off;
        while (left > 0) {
            ssize_t r = pread(fd, bp, left, fo);
            if (r <= 0) { perror("pread"); a->ok = 0; return NULL; }
            bp += r; fo += r; left -= (size_t) r;
        }
        a->total_ns_sum += now_ns() - t0;
    }
    close(fd); free(buf);
    a->ok = 1;
    return NULL;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <file> <threads> <reads_per_thread> [chunk]\n", argv[0]); return 1; }
    const char * path = argv[1];
    int T = atoi(argv[2]), N = atoi(argv[3]);
    size_t chunk = argc > 4 ? (size_t) atol(argv[4]) : 221184;
    chunk = (chunk + 4095) & ~(size_t) 4095;
    struct stat st; if (stat(path, &st)) { perror("stat"); return 1; }

    pthread_t th[64]; arg_t args[64];
    uint64_t t0 = now_ns();
    for (int i = 0; i < T; i++) {
        args[i] = {path, (size_t) st.st_size, chunk, N, 0x9E3779B97F4A7C15ull * (i + 1), 0, 0};
        pthread_create(&th[i], NULL, worker, &args[i]);
    }
    uint64_t lat_sum = 0; int ok = 1;
    for (int i = 0; i < T; i++) { pthread_join(th[i], NULL); lat_sum += args[i].total_ns_sum; ok &= args[i].ok; }
    uint64_t wall = now_ns() - t0;
    if (!ok) { fprintf(stderr, "FAILED\n"); return 1; }
    double bytes = (double) T * N * chunk;
    printf("qd=%d chunk_kib=%zu reads=%d wall_ms=%.1f agg_gbps=%.3f avg_lat_us=%.1f eff_per_read_us=%.1f\n",
           T, chunk / 1024, T * N, wall / 1e6, bytes / wall,
           lat_sum / 1e3 / (T * N), wall / 1e3 / (T * N));
    return 0;
}
