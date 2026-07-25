// uring_probe: does io_uring (SQPOLL / fixed buffers / IOPOLL) beat pread for our
// 216 KiB O_DIRECT random reads on THIS device? Raw syscalls, no liburing.
//   ./uring_probe <file> <mode> <reads> [chunk]
//   mode: 0=pread baseline  1=io_uring basic  2=+fixed buffers  3=+SQPOLL  4=+IOPOLL
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/stat.h>
#include <linux/io_uring.h>
#include <time.h>
#include <atomic>

static uint64_t now_ns() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static int io_uring_setup(unsigned entries, struct io_uring_params * p) {
    return (int) syscall(__NR_io_uring_setup, entries, p);
}
static int io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}
static int io_uring_register(int fd, unsigned op, void * arg, unsigned nr) {
    return (int) syscall(__NR_io_uring_register, fd, op, arg, nr);
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <file> <mode 0-4> <reads> [chunk]\n", argv[0]); return 1; }
    const char * path = argv[1];
    int mode  = atoi(argv[2]);
    int reads = atoi(argv[3]);
    size_t chunk = argc > 4 ? ((size_t) atol(argv[4]) + 4095) & ~(size_t) 4095 : 221184;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t span = ((size_t) st.st_size - chunk) & ~(size_t) 4095;
    uint8_t * buf;
    if (posix_memalign((void **) &buf, 4096, chunk)) return 1;
    uint64_t rng = 0x9E3779B97F4A7C15ull;

    if (mode == 0) {
        uint64_t t0 = now_ns();
        for (int i = 0; i < reads; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            off_t off = (off_t)((rng % span) & ~(uint64_t) 4095);
            ssize_t r = pread(fd, buf, chunk, off);
            if (r != (ssize_t) chunk) { perror("pread"); return 1; }
        }
        printf("mode=0 pread     avg_us=%.1f\n", (now_ns() - t0) / 1e3 / reads);
        return 0;
    }

    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    if (mode == 3) { p.flags |= IORING_SETUP_SQPOLL; p.sq_thread_idle = 2000; }
    if (mode == 4) { p.flags |= IORING_SETUP_IOPOLL; }
    int ring = io_uring_setup(8, &p);
    if (ring < 0) { fprintf(stderr, "io_uring_setup failed (mode %d): %s\n", mode, strerror(errno)); return 2; }

    size_t sq_sz  = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz  = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);
    uint8_t * sq = (uint8_t *) mmap(NULL, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQ_RING);
    uint8_t * cq = (uint8_t *) mmap(NULL, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_CQ_RING);
    struct io_uring_sqe * sqes = (struct io_uring_sqe *) mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe),
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) { perror("mmap"); return 2; }
    unsigned * sq_tail  = (unsigned *)(sq + p.sq_off.tail);
    unsigned * sq_mask  = (unsigned *)(sq + p.sq_off.ring_mask);
    unsigned * sq_array = (unsigned *)(sq + p.sq_off.array);
    unsigned * cq_head  = (unsigned *)(cq + p.cq_off.head);
    unsigned * cq_tail  = (unsigned *)(cq + p.cq_off.tail);
    unsigned * cq_mask  = (unsigned *)(cq + p.cq_off.ring_mask);
    struct io_uring_cqe * cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);

    bool fixed_buf = mode >= 2;
    if (fixed_buf) {
        struct iovec iov = { buf, chunk };
        if (io_uring_register(ring, IORING_REGISTER_BUFFERS, &iov, 1) < 0) {
            fprintf(stderr, "register buffers failed: %s\n", strerror(errno)); return 2;
        }
    }
    if (mode >= 3) {
        if (io_uring_register(ring, IORING_REGISTER_FILES, &fd, 1) < 0) {
            fprintf(stderr, "register files failed: %s\n", strerror(errno)); return 2;
        }
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < reads; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        off_t off = (off_t)((rng % span) & ~(uint64_t) 4095);
        unsigned tail = *sq_tail;
        unsigned idx  = tail & *sq_mask;
        struct io_uring_sqe * s = &sqes[idx];
        memset(s, 0, sizeof(*s));
        s->opcode = fixed_buf ? IORING_OP_READ_FIXED : IORING_OP_READ;
        s->fd     = (mode >= 3) ? 0 : fd;
        s->flags  = (mode >= 3) ? IOSQE_FIXED_FILE : 0;
        s->addr   = (uint64_t) buf;
        s->len    = (unsigned) chunk;
        s->off    = (uint64_t) off;
        s->buf_index = 0;
        sq_array[idx] = idx;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        if (mode == 3) {
            // SQPOLL: kernel thread picks it up; only enter if it went to sleep
            unsigned * flags = (unsigned *)(sq + p.sq_off.flags);
            if (__atomic_load_n(flags, __ATOMIC_ACQUIRE) & IORING_SQ_NEED_WAKEUP) {
                io_uring_enter(ring, 1, 0, IORING_ENTER_SQ_WAKEUP);
            }
        } else if (mode == 4) {
            io_uring_enter(ring, 1, 0, 0);
        } else {
            io_uring_enter(ring, 1, 0, 0);
        }
        // completion: poll the CQ (IOPOLL requires enter(GETEVENTS); others just spin)
        for (;;) {
            if (__atomic_load_n(cq_tail, __ATOMIC_ACQUIRE) != *cq_head) break;
            if (mode == 4) { io_uring_enter(ring, 0, 1, IORING_ENTER_GETEVENTS); }
        }
        struct io_uring_cqe * c = &cqes[*cq_head & *cq_mask];
        if (c->res != (int) chunk) { fprintf(stderr, "cqe res=%d\n", c->res); return 1; }
        __atomic_store_n(cq_head, *cq_head + 1, __ATOMIC_RELEASE);
    }
    printf("mode=%d io_uring%s%s%s avg_us=%.1f\n", mode,
           fixed_buf ? "+fixedbuf" : "", mode >= 3 ? "+sqpoll" : "", mode == 4 ? "+iopoll" : "",
           (now_ns() - t0) / 1e3 / reads);
    return 0;
}
