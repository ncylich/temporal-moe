// Page-cache instrumentation and eviction for the Android benchmark, usable without root.
//
// The Android analogue of "VRAM-resident vs streamed" is "page-cache-resident vs faulted
// from UFS". Since /proc/sys/vm/drop_caches needs root, we use posix_fadvise(DONTNEED),
// and we PROVE it worked with mincore(), which reports exact per-page residency.
//
//   resident <file>          -> how many of the file's pages are in the page cache
//   evict <file>             -> fadvise(DONTNEED) + report residency before/after
//   read <file>              -> timed full sequential read: bytes, ms, MB/s
//   coldread <file>          -> evict, verify eviction, then timed read (the cold number)
//
// Positive control for the whole approach: `coldread` must be measurably slower than a
// second `read` on the same file. If it is not, eviction did not work and every number
// downstream must be labelled warm-cache.
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const size_t PAGE = 4096;

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Fraction of the file currently held in the page cache, via mincore() on a temporary map.
static long resident_pages(const char *path, long *total_pages_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return -1; }
    long npages = (st.st_size + PAGE - 1) / PAGE;
    if (total_pages_out) *total_pages_out = npages;

    void *map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    unsigned char *vec = (unsigned char *)malloc(npages);
    long resident = 0;
    if (mincore(map, st.st_size, vec) == 0) {
        for (long i = 0; i < npages; i++) if (vec[i] & 1) resident++;
    } else {
        perror("mincore");
        resident = -1;
    }
    free(vec);
    munmap(map, st.st_size);
    close(fd);
    return resident;
}

static void report(const char *tag, const char *path) {
    long total = 0;
    long res = resident_pages(path, &total);
    printf("%s resident_pages=%ld total_pages=%ld resident_mib=%.1f pct=%.2f\n",
           tag, res, total, res * PAGE / 1048576.0,
           total ? 100.0 * res / total : 0.0);
    fflush(stdout);
}

// posix_fadvise(DONTNEED) only drops CLEAN, unmapped pages -- so this must run while no
// other process has the file mapped, or it silently does nothing. That is exactly why the
// mincore() readback below is mandatory rather than decorative.
static int evict(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    // sync first: dirty pages are not droppable
    if (posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED) != 0) perror("fadvise");
    close(fd);
    return 0;
}

static void timed_read(const char *path, const char *tag) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return; }
    const size_t BUF = 8 << 20;
    char *buf = (char *)malloc(BUF);
    double t0 = now_ms();
    ssize_t n, total = 0;
    while ((n = read(fd, buf, BUF)) > 0) total += n;
    double ms = now_ms() - t0;
    free(buf);
    close(fd);
    printf("%s bytes_read=%zd ms=%.1f MB_per_s=%.1f\n",
           tag, total, ms, (total / 1048576.0) / (ms / 1000.0));
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <resident|evict|read|coldread> <file>\n", argv[0]);
        return 2;
    }
    const char *cmd = argv[1], *path = argv[2];

    if (!strcmp(cmd, "resident")) {
        report("resident", path);
    } else if (!strcmp(cmd, "evict")) {
        report("before_evict", path);
        evict(path);
        report("after_evict", path);
    } else if (!strcmp(cmd, "read")) {
        report("before_read", path);
        timed_read(path, "read");
        report("after_read", path);
    } else if (!strcmp(cmd, "coldread")) {
        evict(path);
        report("after_evict", path);        // must be ~0 or the cold claim is void
        timed_read(path, "coldread");
    } else {
        fprintf(stderr, "unknown command %s\n", cmd);
        return 2;
    }
    return 0;
}
