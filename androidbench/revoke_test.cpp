// Can we REVOKE residency of one expert after loading it, and is the revocation real?
//
// "Real" = the next access must fetch from storage again. Two mechanisms differ:
//   madvise(MADV_DONTNEED)  - zaps our PTEs; the page may remain in page cache (minor
//                             fault on next touch, no disk I/O, memory NOT reclaimed)
//   posix_fadvise(DONTNEED) - drops the page cache, but usually refuses while the range
//                             is mapped by a live process
// The engine holds the model mmap'd for its lifetime, so this distinction decides whether
// dynamic expert eviction is actually possible at all.
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static long io_read_bytes() {
    FILE* f = fopen("/proc/self/io", "r"); if (!f) return -1;
    char k[64]; long v, out = -1;
    while (fscanf(f, "%63s %ld", k, &v) == 2) if (!strcmp(k, "read_bytes:")) out = v;
    fclose(f); return out;
}
static long resident_pages(void* p, size_t len) {
    size_t n = (len + 4095) / 4096;
    unsigned char* vec = (unsigned char*)malloc(n);
    long r = 0;
    if (mincore(p, len, vec) == 0) for (size_t i = 0; i < n; i++) r += vec[i] & 1;
    else r = -1;
    free(vec); return r;
}
static long touch(volatile char* p, size_t len) {
    long s = 0; for (size_t i = 0; i < len; i += 4096) s += p[i]; return s;
}

int main(int argc, char** argv) {
    const char* path = argv[1];
    size_t off = argc > 2 ? strtoul(argv[2], 0, 10) : (2UL << 30);
    size_t len = argc > 3 ? strtoul(argv[3], 0, 10) : (216 * 1024);   // one fine expert
    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    char* m = (char*)mmap(0, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    off &= ~4095UL;
    char* slice = m + off;

    printf("slice: offset %zu len %zu (%zu pages)\n", off, len, (len+4095)/4096);
    long b0 = io_read_bytes(); touch(slice, len); long b1 = io_read_bytes();
    printf("1. first touch      : resident=%ld  disk_read=%ld B\n", resident_pages(slice,len), b1-b0);

    madvise(slice, len, MADV_DONTNEED);
    printf("2. after MADV_DONTNEED: resident=%ld\n", resident_pages(slice, len));
    long b2 = io_read_bytes(); touch(slice, len); long b3 = io_read_bytes();
    printf("3. re-touch         : resident=%ld  disk_read=%ld B  <-- 0 means page stayed cached\n",
           resident_pages(slice,len), b3-b2);

    posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
    printf("4. after FADV_DONTNEED (while mapped): resident=%ld\n", resident_pages(slice, len));
    long b4 = io_read_bytes(); touch(slice, len); long b5 = io_read_bytes();
    printf("5. re-touch         : resident=%ld  disk_read=%ld B\n", resident_pages(slice,len), b5-b4);

    madvise(slice, len, MADV_DONTNEED);
    posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED);
    long b6 = io_read_bytes(); touch(slice, len); long b7 = io_read_bytes();
    printf("6. BOTH, then touch : resident=%ld  disk_read=%ld B  <-- >0 means real eviction\n",
           resident_pages(slice,len), b7-b6);
    return 0;
}
