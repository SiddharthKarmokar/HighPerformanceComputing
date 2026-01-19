#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

typedef struct {int N; int t; int tid; int nthreads; double *A,*B,*C; int pattern; int block; } arg_t;

static inline uint64_t now_ns(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (uint64_t)ts.tv_sec*1000000000ULL + ts.tv_nsec; }

void *worker(void *v){ arg_t *a = (arg_t*)v; int N=a->N; int tid=a->tid; int T=a->nthreads; double *A=a->A,*B=a->B,*C=a->C; int p=a->pattern; int bsz=a->block;
    if(p==0){ // row contiguous: each thread handles contiguous set of rows
        int rows_per = (N + T -1)/T; int r0 = tid*rows_per; int r1 = r0+rows_per; if(r1>N) r1=N;
        for(int i=r0;i<r1;i++){
            double *arow = A + (size_t)i*N; double *brow = B + (size_t)i*N; double *crow = C + (size_t)i*N;
            for(int j=0;j<N;j++) crow[j]=arow[j]+brow[j];
        }
    } else if(p==1){ // column-major: threads handle column ranges
        int cols_per = (N + T -1)/T; int c0 = tid*cols_per; int c1 = c0+cols_per; if(c1>N) c1=N;
        for(int j=c0;j<c1;j++){
            size_t idx=j;
            for(int i=0;i<N;i++){ C[idx]=A[idx]+B[idx]; idx += N; }
        }
    } else if(p==2){ // blocked tiling by rows and cols
        int rows_per = (N + T -1)/T; int r0 = tid*rows_per; int r1 = r0+rows_per; if(r1>N) r1=N;
        for(int ii=r0; ii<r1; ii+=bsz){
            int iend = ii+bsz; if(iend>r1) iend=r1;
            for(int jj=0;jj<N;jj+=bsz){
                int jend = jj+bsz; if(jend>N) jend=N;
                for(int i=ii;i<iend;i++){
                    double *arow = A + (size_t)i*N; double *brow = B + (size_t)i*N; double *crow = C + (size_t)i*N;
                    for(int j=jj;j<jend;j++) crow[j]=arow[j]+brow[j];
                }
            }
        }
    } else if(p==3){ // linear flattened: each thread handles contiguous chunk of N*N elements
        size_t total = (size_t)N*N; size_t per = (total + T -1)/T; size_t s = per*tid; size_t e = s+per; if(e>total) e=total;
        for(size_t idx=s; idx<e; idx++) C[idx]=A[idx]+B[idx];
    } else if(p==4){ // cyclic rows: thread processes every T-th row starting from tid
        for(int i=tid;i<N;i+=T){
            double *arow = A + (size_t)i*N; double *brow = B + (size_t)i*N; double *crow = C + (size_t)i*N;
            for(int j=0;j<N;j++) crow[j]=arow[j]+brow[j];
        }
    } else if(p==5){ // unrolled inner loop by 4, contiguous rows
        int rows_per = (N + T -1)/T; int r0 = tid*rows_per; int r1 = r0+rows_per; if(r1>N) r1=N;
        for(int i=r0;i<r1;i++){
            double *arow = A + (size_t)i*N; double *brow = B + (size_t)i*N; double *crow = C + (size_t)i*N;
            int j=0; for(; j+3<N; j+=4){ crow[j]=arow[j]+brow[j]; crow[j+1]=arow[j+1]+brow[j+1]; crow[j+2]=arow[j+2]+brow[j+2]; crow[j+3]=arow[j+3]+brow[j+3]; }
            for(;j<N;j++) crow[j]=arow[j]+brow[j];
        }
    }
    return NULL;
}

int main(int argc,char **argv){
    if(argc<4){ printf("Usage: %s N nthreads pattern\nPatterns: 0=row,1=col,2=block,3=linear,4=cyclic,5=unroll\n",argv[0]); return 1; }
    int N=atoi(argv[1]); int T=atoi(argv[2]); int pat=atoi(argv[3]); int repeats=3; int bsz=32;
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
    printf("N=%d threads=%d pattern=%d cores=%d\n",N,T,pat,cores);
    size_t total = (size_t)N*N;
    // allocate aligned
    double *A, *B, *C;
    if(posix_memalign((void**)&A, 64, total*sizeof(double)) || posix_memalign((void**)&B,64,total*sizeof(double)) || posix_memalign((void**)&C,64,total*sizeof(double))){ perror("memalign"); return 1; }
    // init
    for(size_t i=0;i<total;i++){ A[i]=1.0; B[i]=2.0; C[i]=0.0; }

    pthread_t *ths = malloc(sizeof(pthread_t)*T);
    arg_t *args = malloc(sizeof(arg_t)*T);

    // warmup
    for(int r=0;r<1;r++){
        for(int t=0;t<T;t++){ args[t].N=N; args[t].tid=t; args[t].nthreads=T; args[t].A=A; args[t].B=B; args[t].C=C; args[t].pattern=pat; args[t].block=bsz; }
        for(int t=0;t<T;t++) pthread_create(&ths[t],NULL,worker,&args[t]);
        for(int t=0;t<T;t++) pthread_join(ths[t],NULL);
    }

    uint64_t t0=now_ns();
    for(int rep=0; rep<repeats; rep++){
        for(int t=0;t<T;t++){ args[t].N=N; args[t].tid=t; args[t].nthreads=T; args[t].A=A; args[t].B=B; args[t].C=C; args[t].pattern=pat; args[t].block=bsz; }
        for(int t=0;t<T;t++) pthread_create(&ths[t],NULL,worker,&args[t]);
        for(int t=0;t<T;t++) pthread_join(ths[t],NULL);
    }
    uint64_t t1=now_ns();
    double elapsed = (t1 - t0)/1e9 / repeats;
    // verify simple checksum
    double s=0; for(size_t i=0;i<total;i+= (total/16>0?total/16:1)) s+=C[i];
    printf("elapsed=%f sec checksum=%f\n", elapsed, s);
    fflush(stdout);
    // print CSV line
    printf("CSV,%d,%d,%d,%.9f,%f\n", N,T,pat,elapsed,s);
    return 0;
}
