#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RUNS 5     // number of repetitions per pattern
#define BLOCK 64  // tile size

// High-resolution timer
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 0: Row-major (i, j)
void pattern0(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// 1: Column-major (j, i)
void pattern1(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) y[i] = 0.0;
    for (int j = 0; j < N; j++) {
        double xj = x[j];
        for (int i = 0; i < N; i++) {
            y[i] += A[i * N + j] * xj;
        }
    }
}

// 2: Row-major unrolled (4x)
void pattern2(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        int j = 0;
        for (; j <= N - 4; j += 4) {
            sum += A[i * N + j]     * x[j];
            sum += A[i * N + j + 1] * x[j + 1];
            sum += A[i * N + j + 2] * x[j + 2];
            sum += A[i * N + j + 3] * x[j + 3];
        }
        for (; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// 3: Column-major unrolled (4x)
void pattern3(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) y[i] = 0.0;
    for (int j = 0; j < N; j++) {
        double xj = x[j];
        int i = 0;
        for (; i <= N - 4; i += 4) {
            y[i]     += A[i * N + j]     * xj;
            y[i + 1] += A[(i + 1) * N + j] * xj;
            y[i + 2] += A[(i + 2) * N + j] * xj;
            y[i + 3] += A[(i + 3) * N + j] * xj;
        }
        for (; i < N; i++) {
            y[i] += A[i * N + j] * xj;
        }
    }
}

// 4: Blocked row-major
void pattern4(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) y[i] = 0.0;

    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            int imax = (ii + BLOCK < N) ? ii + BLOCK : N;
            int jmax = (jj + BLOCK < N) ? jj + BLOCK : N;

            for (int i = ii; i < imax; i++) {
                double sum = y[i];
                for (int j = jj; j < jmax; j++) {
                    sum += A[i * N + j] * x[j];
                }
                y[i] = sum;
            }
        }
    }
}

// 5: Pointer arithmetic row-major (FIXED)
void pattern5(int N, double *A, double *x, double *y) {
    for (int i = 0; i < N; i++) {
        double *pA = A + i * N;   // reset pointer per row
        double *px = x;
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += (*pA++) * (*px++);
        }
        y[i] = sum;
    }
}

int main() {
    int sizes[] = {256, 512, 1024, 2048};
    int patterns = 6;

    printf("N,threads,pattern,time_sec,checksum\n");

    for (int s = 0; s < 4; s++) {
        int N = sizes[s];

        double *A = (double*)malloc(N * N * sizeof(double));
        double *x = (double*)malloc(N * sizeof(double));
        double *y = (double*)malloc(N * sizeof(double));

        for (int i = 0; i < N * N; i++) A[i] = 1.0 / N;
        for (int i = 0; i < N; i++) x[i] = 48.0 / N;

        for (int p = 0; p < patterns; p++) {

            /* Warm-up */
            pattern0(N, A, x, y);

            double best_time = 1e9;

            for (int r = 0; r < RUNS; r++) {
                for (int i = 0; i < N; i++) y[i] = 0.0;

                double start = get_time();

                switch (p) {
                    case 0: pattern0(N, A, x, y); break;
                    case 1: pattern1(N, A, x, y); break;
                    case 2: pattern2(N, A, x, y); break;
                    case 3: pattern3(N, A, x, y); break;
                    case 4: pattern4(N, A, x, y); break;
                    case 5: pattern5(N, A, x, y); break;
                }

                double end = get_time();
                double elapsed = end - start;

                if (elapsed < best_time)
                    best_time = elapsed;
            }

            double checksum = 0.0;
            for (int i = 0; i < N; i++) checksum += y[i];

            printf("%d,1,%d,%.9f,%.6f\n", N, p, best_time, checksum);
        }

        free(A);
        free(x);
        free(y);
    }

    return 0;
}
