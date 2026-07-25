#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <sys/uio.h>
// Burst wall-time for ONE expert (648 KiB), with an optional memory-bandwidth load to
// emulate the compute threads. Reproduces the idle-vs-in-engine inversion outside the engine.
static const char*PATH; static long long FSIZE;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
#define EXPERT (3*221184)
static int NPART,USE_IOV,NLOAD; static long long BASEOFF=0; static size_t PSZ; static volatile long long base; static volatile int stop=0;
static pthread_barrier_t bar; static int fds[16]; static void*bufs[16]; static void*sep[16][3]; static int SEP=0;
static int SHARED=0;
static void*loadthr(void*a){
  volatile long long acc=0;
  if(SHARED){
    // emulate the engine: compute threads stream the OTHER experts out of the very
    // tensors the DMA is writing into, so the DMA's cache invalidations collide with
    // active reads on the same pages.
    while(!stop){
      for(int k=0;k<3;k++){ char*p=(char*)sep[0][k]; if(!p) continue;
        for(size_t i=0;i<(24u<<20);i+=64) { acc+=p[i]; if(stop) break; } }
    }
  } else {
    size_t N=32u<<20; char*p=malloc(N); if(!p) return 0; memset(p,1,N);
    while(!stop){ for(size_t i=0;i<N;i+=64) acc+=p[i]; }
    free(p);
  }
  (void)acc; return 0;
}
static void*wk(void*a){
  int id=(int)(long)a;
  for(int r=0;r<220;r++){
    pthread_barrier_wait(&bar);
    if(id<NPART){
      long long off=base+(long long)id*PSZ;
      if(USE_IOV){
        struct iovec iov[3]; int nv=3;
        for(int k=0;k<3;k++){
          iov[k].iov_base = SEP ? sep[id][k] : (void*)((char*)bufs[id]+k*221184);
          iov[k].iov_len=221184;
        }
        preadv(fds[id],iov,nv,off);
      } else pread(fds[id],bufs[id],PSZ,off);
    }
    pthread_barrier_wait(&bar);
  }
  return 0;
}
int main(int c,char**v){
  PATH=v[1]; NPART=atoi(v[2]); USE_IOV=atoi(v[3]); NLOAD=atoi(v[4]); SEP=(c>5)?atoi(v[5]):0; BASEOFF=(c>6)?atoll(v[6]):0; SHARED=(c>7)?atoi(v[7]):0;
  PSZ=USE_IOV?EXPERT:EXPERT/NPART;
  int fd=open(PATH,O_RDONLY); FSIZE=lseek(fd,0,SEEK_END); close(fd);
  for(int i=0;i<NPART;i++){fds[i]=open(PATH,O_RDONLY|O_DIRECT); if(posix_memalign(&bufs[i],4096,PSZ+8192))return 1;}
  if(SEP){ for(int i=0;i<NPART;i++) for(int k=0;k<3;k++){
      // 24 MiB apart, mimicking three distinct expert tensors
      if(posix_memalign(&sep[i][k],4096,24u<<20)) return 1; memset(sep[i][k],0,4096);
  } }
  
  pthread_t ld[8]; for(int i=0;i<NLOAD;i++) pthread_create(&ld[i],0,loadthr,0);
  pthread_barrier_init(&bar,0,NPART+1);
  pthread_t th[16]; for(int i=0;i<NPART;i++) pthread_create(&th[i],0,wk,(void*)(long)i);
  double tot=0,worst=0; unsigned seed=999; int n=0;
  for(int r=0;r<220;r++){
    seed=seed*1103515245+12345;
    base=BASEOFF+((long long)(seed%4000))*EXPERT; base&=~4095LL;
    if(base+EXPERT>FSIZE) base=BASEOFF;
    double t0=now();
    pthread_barrier_wait(&bar); pthread_barrier_wait(&bar);
    double dt=(now()-t0)*1e6;
    if(r>=40){tot+=dt;n++;if(dt>worst)worst=dt;}
  }
  stop=1;
  for(int i=0;i<NPART;i++) pthread_join(th[i],0);
  for(int i=0;i<NLOAD;i++) pthread_join(ld[i],0);
  printf("  base=%lldMiB load=%d %s %d x %6zuKiB %-8s : %6.0f us mean  %6.0f us worst  (%.2f GB/s)\n",
     BASEOFF>>20,NLOAD,SHARED?"SHARED-dst":"private   ",NPART,PSZ/1024,USE_IOV?"preadv3":"pread",tot/n,worst,EXPERT/(tot/n)/1e3);
  return 0;
}
