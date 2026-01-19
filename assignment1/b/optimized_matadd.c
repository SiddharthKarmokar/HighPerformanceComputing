#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>

typedef struct {
    int N;
    int tid;
    int nthreads;
    int pattern;
    int block;
    int repeats;
    double *restrict A;
    double *restrict B;
    double *restrict C;
    pthread_barrier_t *barrier;
    char pad[64];   // avoid false sharing
} arg_t;

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline void pin_thread(int tid) {
    cpu_set_t set;
    CPU_ZERO(&set);
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_SET(tid % cores, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

void *worker(void *v) {
    arg_t *a = (arg_t *)v;
    pin_thread(a->tid);

    int N = a->N;
    int tid = a->tid;
    int T = a->nthreads;
    int p = a->pattern;
    int bsz = a->block;

    double *restrict A = a->A;
    double *restrict B = a->B;
    double *restrict C = a->C;

    pthread_barrier_t *bar = a->barrier;

    /* synchronize start */
    pthread_barrier_wait(bar);

    for (int rep = 0; rep < a->repeats; rep++) {

        if (p == 0) { /* row contiguous */
            int rows = (N + T - 1) / T;
            int r0 = tid * rows;
            int r1 = r0 + rows; if (r1 > N) r1 = N;

            for (int i = r0; i < r1; i++) {
                double *ar = A + (size_t)i * N;
                double *br = B + (size_t)i * N;
                double *cr = C + (size_t)i * N;
                for (int j = 0; j < N; j++)
                    cr[j] = ar[j] + br[j];
            }

        } else if (p == 1) { /* column major */
            int cols = (N + T - 1) / T;
            int c0 = tid * cols;
            int c1 = c0 + cols; if (c1 > N) c1 = N;

            for (int j = c0; j < c1; j++) {
                size_t idx = j;
                for (int i = 0; i < N; i++) {
                    C[idx] = A[idx] + B[idx];
                    idx += N;
                }
            }

        } else if (p == 2) { /* blocked */
            int rows = (N + T - 1) / T;
            int r0 = tid * rows;
            int r1 = r0 + rows; if (r1 > N) r1 = N;

            for (int ii = r0; ii < r1; ii += bsz) {
                int ie = ii + bsz; if (ie > r1) ie = r1;
                for (int jj = 0; jj < N; jj += bsz) {
                    int je = jj + bsz; if (je > N) je = N;
                    for (int i = ii; i < ie; i++) {
                        double *ar = A + (size_t)i * N;
                        double *br = B + (size_t)i * N;
                        double *cr = C + (size_t)i * N;
                        for (int j = jj; j < je; j++)
                            cr[j] = ar[j] + br[j];
                    }
                }
            }

        } else if (p == 3) { /* linear */
            size_t total = (size_t)N * N;
            size_t per = (total + T - 1) / T;
            size_t s = tid * per;
            size_t e = s + per; if (e > total) e = total;

            for (size_t k = s; k < e; k++)
                C[k] = A[k] + B[k];

        } else if (p == 4) { /* cyclic rows */
            for (int i = tid; i < N; i += T) {
                double *ar = A + (size_t)i * N;
                double *br = B + (size_t)i * N;
                double *cr = C + (size_t)i * N;
                for (int j = 0; j < N; j++)
                    cr[j] = ar[j] + br[j];
            }

        } else if (p == 5) { /* unroll 4 */
            int rows = (N + T - 1) / T;
            int r0 = tid * rows;
            int r1 = r0 + rows; if (r1 > N) r1 = N;

            for (int i = r0; i < r1; i++) {
                double *ar = A + (size_t)i * N;
                double *br = B + (size_t)i * N;
                double *cr = C + (size_t)i * N;
                int j = 0;
                for (; j + 3 < N; j += 4) {
                    cr[j]   = ar[j]   + br[j];
                    cr[j+1] = ar[j+1] + br[j+1];
                    cr[j+2] = ar[j+2] + br[j+2];
                    cr[j+3] = ar[j+3] + br[j+3];
                }
                for (; j < N; j++)
                    cr[j] = ar[j] + br[j];
            }
        }

        pthread_barrier_wait(bar);  // end of this iteration
    }

    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s N threads pattern repeats\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    int pattern = atoi(argv[3]);
    int repeats = atoi(argv[4]);

    size_t total = (size_t)N * N;

    double *A, *B, *C;
    if (posix_memalign((void**)&A, 64, total * sizeof(double)) ||
        posix_memalign((void**)&B, 64, total * sizeof(double)) ||
        posix_memalign((void**)&C, 64, total * sizeof(double))) {
        perror("posix_memalign");
        return 1;
    }

    for (size_t i = 0; i < total; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    pthread_t *ths = malloc(sizeof(pthread_t) * T);
    arg_t *args = malloc(sizeof(arg_t) * T);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, T);

    for (int t = 0; t < T; t++) {
        args[t].N = N;
        args[t].tid = t;
        args[t].nthreads = T;
        args[t].pattern = pattern;
        args[t].block = 32;
        args[t].repeats = repeats;
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].barrier = &barrier;
        pthread_create(&ths[t], NULL, worker, &args[t]);
    }

    uint64_t t0 = now_ns();
    for (int t = 0; t < T; t++)
        pthread_join(ths[t], NULL);
    uint64_t t1 = now_ns();

    double sec = (t1 - t0) / 1e9 / repeats;

    double checksum = 0.0;
    for (size_t i = 0; i < total; i += (total / 16 + 1))
        checksum += C[i];

    printf("CSV,%d,%d,%d,%.9f,%f\n", N, T, pattern, sec, checksum);

    return 0;
}
