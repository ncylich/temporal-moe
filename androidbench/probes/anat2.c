#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
// Anatomy of one read. Decomposes: per-request FIXED cost vs per-byte STREAMING rate,
// how much fixed cost concurrency hides, device-cache effect, and pure kernel path cost.
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static const char*P; static long long FSZ;
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y?1:0;}

// median latency of a single QD1 O_DIRECT read of `sz`, random offsets (cold) or fixed (hot)
static double lat1(size_t sz,int hot,int direct,int n){
  int fd=open(P,O_RDONLY|(direct?O_DIRECT:0)); if(fd<0)return -1;
  void*b; if(posix_memalign(&b,4096,sz+4096))return -1;
  double*v=malloc(n*sizeof(double)); unsigned s=7;
  const long long LO=300LL<<20;                 // past the sparse header region
  long long HI=FSZ-(long long)sz-(4LL<<20); if(HI<=LO) HI=LO+1;
  long long span=(HI-LO)/1048576LL;             // in MiB
  long long fixed=(LO+((HI-LO)/2))&~1048575LL;
  for(int i=0;i<n;i++){
    s=s*1103515245+12345;
    long long off = hot? fixed : (LO + ((long long)(s%span))*1048576LL);
    off &= ~4095LL;
    if(off<LO) off=LO;
    double t0=now(); ssize_t r=pread(fd,b,sz,off); double dt=(now()-t0)*1e6;
    if(r!=(ssize_t)sz){ /* short read */ }
    v[i]=dt;
  }
  qsort(v,n,sizeof(double),cmp); double m=v[n/2]; free(v); free(b); close(fd); return m;
}
// N concurrent QD readers, wall time for the whole batch of N reads of sz
typedef struct{size_t sz;int n;pthread_barrier_t*bar;int*fds;void**bufs;long long*offs;} arg_t;
static void*rd(void*a){
  arg_t*A=a; int id=(int)(intptr_t)pthread_getspecific(*(pthread_key_t*)0);
  return 0;
}
int main(int c,char**v){
  P=v[1]; int fd=open(P,O_RDONLY); FSZ=lseek(fd,0,SEEK_END); close(fd);
  printf("file=%s size=%.1f GiB\n\n",P,FSZ/1073741824.0);

  printf("A. PER-REQUEST COST LADDER  (QD1, O_DIRECT, cold random offsets in WRITTEN data, median of 200)\n");
  printf("   %10s %10s %12s %12s\n","size","latency","marginal","implied GB/s");
  size_t sizes[]={4096,8192,16384,32768,65536,110592,221184,331776,663552,1048576};
  double prev_l=0; size_t prev_s=0;
  for(int i=0;i<10;i++){
    double l=lat1(sizes[i],0,1,200);
    double marg = prev_s? (l-prev_l)/((double)(sizes[i]-prev_s)/1024.0) : 0;
    printf("   %8zuKiB %8.1fus %10s %12.2f\n", sizes[i]/1024, l,
           prev_s? ({static char b[32]; snprintf(b,32,"%.3fus/KiB",marg); b;}) : "-",
           sizes[i]/l/1e3);
    prev_l=l; prev_s=sizes[i];
  }
  printf("\nB. DEVICE/CACHE EFFECT  (same 4 KiB offset repeatedly vs random)\n");
  printf("   cold random 4KiB : %7.1f us\n", lat1(4096,0,1,300));
  printf("   hot same   4KiB : %7.1f us\n", lat1(4096,1,1,300));
  printf("   hot same 648KiB : %7.1f us\n", lat1(663552,1,1,200));
  printf("\nC. PURE KERNEL PATH  (buffered read, page-cache HOT: no device involved)\n");
  printf("   4 KiB   buffered hot: %7.2f us\n", lat1(4096,1,0,2000));
  printf("   108 KiB buffered hot: %7.2f us\n", lat1(110592,1,0,1000));
  printf("   648 KiB buffered hot: %7.2f us\n", lat1(663552,1,0,500));
  return 0;
}
