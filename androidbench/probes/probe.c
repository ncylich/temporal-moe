#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
static const char *PATH; static size_t SZ; static int QD, NREQ; static long long FSIZE;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void*wk(void*a){
  int id=(int)(long)a; int fd=open(PATH,O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 0;}
  void*b; if(posix_memalign(&b,4096,SZ)) return 0;
  unsigned seed=id*7919+13;
  for(int i=0;i<NREQ/QD;i++){
    seed=seed*1103515245+12345;
    long long off=((long long)(seed%100000))*221184; off&=~4095LL;
    if(off+ (long long)SZ > FSIZE) off = FSIZE - SZ - 4096, off&=~4095LL;
    if(pread(fd,b,SZ,off)<0){perror("pread");break;}
  }
  close(fd); free(b); return 0;
}
int main(int c,char**v){
  PATH=v[1]; SZ=atoi(v[2]); QD=atoi(v[3]); NREQ=atoi(v[4]);
  int fd=open(PATH,O_RDONLY); FSIZE=lseek(fd,0,SEEK_END); close(fd);
  pthread_t th[64]; double t0=now();
  for(int i=0;i<QD;i++) pthread_create(&th[i],0,wk,(void*)(long)i);
  for(int i=0;i<QD;i++) pthread_join(th[i],0);
  double dt=now()-t0; double mb=(double)SZ*(NREQ/QD)*QD/1e6;
  printf("  size=%6zuKiB QD=%-2d  %7.1f MB read in %6.3f s  =  %.2f GB/s   (%.0f us/req)\n",
         SZ/1024,QD,mb,dt,mb/1e3/dt, dt*1e6/(NREQ/QD));
  return 0;
}
