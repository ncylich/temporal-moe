// Memory balloon: hold N MiB of dirty anonymous memory to squeeze the page cache.
//
// Why this exists. On this device the model is mmap'd, so "resident" means "in the page
// cache", which is the same RAM the balloon occupies. To test the temporal-MoE premise --
// that only the top-k experts per token need to be resident -- we need to force the model
// to NOT fit, then see how decode degrades as residency falls. Without root we cannot
// drop caches or set cgroup limits, but we CAN create genuine memory pressure and let the
// kernel evict the page cache for us.
//
//   balloon <MiB> <hold_seconds>
//
// Pages are written (not just allocated) so they are dirty anonymous memory the kernel
// cannot simply discard -- it must evict page cache instead, which is the point.
// Prints residency-relevant meminfo before and after inflating so the effect is recorded,
// not assumed. Exits on its own after hold_seconds so a crashed benchmark cannot leave
// the device wedged.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void dump_meminfo(const char *tag) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    printf("%s ", tag);
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "MemAvailable:", 13) || !strncmp(line, "Cached:", 7) ||
            !strncmp(line, "MemFree:", 8)) {
            char *p = line;
            while (*p && *p != '\n') p++;
            *p = 0;
            printf("[%s] ", line);
        }
    }
    printf("\n");
    fflush(stdout);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <MiB> <hold_seconds>\n", argv[0]);
        return 2;
    }
    size_t mib = strtoul(argv[1], nullptr, 10);
    int hold = atoi(argv[2]);
    size_t bytes = mib * 1048576UL;

    dump_meminfo("before_balloon");

    char *p = (char *)malloc(bytes);
    if (!p) { fprintf(stderr, "malloc of %zu MiB failed\n", mib); return 1; }
    // Touch every page so the memory is genuinely committed and dirty.
    for (size_t off = 0; off < bytes; off += 4096) p[off] = (char)(off >> 12);

    dump_meminfo("after_balloon");
    printf("holding %zu MiB for %d s\n", mib, hold);
    fflush(stdout);

    sleep(hold);

    // Re-touch before releasing so the kernel could not have quietly swapped us out
    // and left the page cache intact -- that would make the pressure fictitious.
    volatile long sum = 0;
    for (size_t off = 0; off < bytes; off += 4096) sum += p[off];
    dump_meminfo("before_release");
    free(p);
    printf("released (checksum %ld)\n", (long)sum);
    return 0;
}
