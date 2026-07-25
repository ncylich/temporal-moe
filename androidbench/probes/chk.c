#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
int main(int c,char**v){
  int fd=open(v[1],O_RDONLY|O_DIRECT);
  long long base=atoll(v[2]);
  void*b[3]; for(int k=0;k<3;k++) posix_memalign(&b[k],4096,221184);
  struct iovec iov[3]; for(int k=0;k<3;k++){iov[k].iov_base=b[k];iov[k].iov_len=221184;}
  ssize_t r=preadv(fd,iov,3,base);
  printf("  preadv(3 iovecs, 648 KiB requested) returned %zd bytes  (%.0f KiB)\n", r, r/1024.0);
  void*big; posix_memalign(&big,4096,663552);
  ssize_t r2=pread(fd,big,663552,base);
  printf("  pread(648 KiB requested)             returned %zd bytes  (%.0f KiB)\n", r2, r2/1024.0);
  ssize_t r3=pread(fd,big,331776,base);
  printf("  pread(324 KiB requested)             returned %zd bytes  (%.0f KiB)\n", r3, r3/1024.0);
  return 0;
}
