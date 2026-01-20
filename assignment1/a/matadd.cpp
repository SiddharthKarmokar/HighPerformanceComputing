#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <algorithm> 
#include <thread>
#include <fstream>
#include <cmath>

using namespace std;

inline int idx(int r, int c, int N) {
    return r * N + c;
}

double get_checksum(const vector<double>& C, int N) {
    double sum = 0;
    int limit = min(N, 4);
    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < limit; j++) {
            sum += C[idx(i, j, N)];
        }
    }
    return sum;
}

enum PatternID {
    BLOCKED_32 = 0,
    COL_MAJOR = 1,
    CYCLIC_ROWS = 2,
    LINEAR_FLAT = 3,
    ROW_MAJOR_CHUNKS = 4,
    UNROLL_4 = 5
};

void add_blocked_32(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    int blockSize = 32;
    int rows_per_thread = N / n_threads;
    int start_row = t_id * rows_per_thread;
    int end_row = (t_id == n_threads - 1) ? N : start_row + rows_per_thread;

    for (int ii = start_row; ii < end_row; ii += blockSize) {
        for (int jj = 0; jj < N; jj += blockSize) {
            for (int i = ii; i < min(ii + blockSize, end_row); i++) {
                for (int j = jj; j < min(jj + blockSize, N); j++) {
                    int id = idx(i, j, N);
                    C[id] = A[id] + B[id];
                }
            }
        }
    }
}

void add_col_major(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    int cols_per_thread = N / n_threads;
    int start_col = t_id * cols_per_thread;
    int end_col = (t_id == n_threads - 1) ? N : start_col + cols_per_thread;

    for (int j = start_col; j < end_col; j++) {
        for (int i = 0; i < N; i++) {
            int id = idx(i, j, N);
            C[id] = A[id] + B[id];
        }
    }
}

void add_cyclic_rows(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    for (int i = t_id; i < N; i += n_threads) {
        for (int j = 0; j < N; j++) {
            int id = idx(i, j, N);
            C[id] = A[id] + B[id];
        }
    }
}

void add_linear_flat(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    long long total = (long long)N * N;
    long long chunk = total / n_threads;
    long long start = t_id * chunk;
    long long end = (t_id == n_threads - 1) ? total : start + chunk;

    for (long long k = start; k < end; k++) {
        C[k] = A[k] + B[k];
    }
}
void add_row_major_chunks(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    int rows_per_thread = N / n_threads;
    int start_row = t_id * rows_per_thread;
    int end_row = (t_id == n_threads - 1) ? N : start_row + rows_per_thread;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            int id = idx(i, j, N);
            C[id] = A[id] + B[id];
        }
    }
}

void add_unroll_4(const vector<double>& A, const vector<double>& B, vector<double>& C, int N, int t_id, int n_threads) {
    int rows_per_thread = N / n_threads;
    int start_row = t_id * rows_per_thread;
    int end_row = (t_id == n_threads - 1) ? N : start_row + rows_per_thread;

    for (int i = start_row; i < end_row; i++) {
        int j = 0;
        for (; j <= N - 4; j += 4) {
            int id0 = idx(i, j, N);
            int id1 = idx(i, j+1, N);
            int id2 = idx(i, j+2, N);
            int id3 = idx(i, j+3, N);
            C[id0] = A[id0] + B[id0];
            C[id1] = A[id1] + B[id1];
            C[id2] = A[id2] + B[id2];
            C[id3] = A[id3] + B[id3];
        }
        for (; j < N; j++) {
            int id = idx(i, j, N);
            C[id] = A[id] + B[id];
        }
    }
}


typedef void (*MatrixFunc)(const vector<double>&, const vector<double>&, vector<double>&, int, int, int);

struct PatternInfo {
    int id;
    string name;
    MatrixFunc func;
};

int main() {
    vector<int> dimensions = {256, 512, 1024, 2048};
    vector<int> thread_counts = {1}; 
    vector<PatternInfo> patterns = {
        {BLOCKED_32,      " ",        add_blocked_32},
        {COL_MAJOR,       " ",         add_col_major},
        {CYCLIC_ROWS,     " ",       add_cyclic_rows},
        {LINEAR_FLAT,     " ",       add_linear_flat},
        {ROW_MAJOR_CHUNKS," ",   add_row_major_chunks},
        {UNROLL_4,        " ",          add_unroll_4}
    };

    ofstream csv("results.csv");
    csv << "N,threads,pattern,sec,checksum" << endl;

    cout << left 
         << setw(8) << "N" 
         << setw(10) << "threads" 
         << setw(10) << "pattern" 
         << setw(15) << "sec" 
         << setw(15) << "checksum" << endl;
    cout << string(60, '-') << endl;

    for (int N : dimensions) {
        vector<double> A(N * N, 1.0);
        vector<double> B(N * N, 2.0);
        vector<double> C(N * N, 0.0);

        for (int t_num : thread_counts) {
            for (const auto& p : patterns) {
                
                fill(C.begin(), C.end(), 0.0);

                auto start = chrono::high_resolution_clock::now();

                vector<thread> threads;
                for(int t = 0; t < t_num; t++) {
                    threads.emplace_back(p.func, ref(A), ref(B), ref(C), N, t, t_num);
                }
                for(auto& th : threads) {
                    th.join();
                }

                auto end = chrono::high_resolution_clock::now();
                double time_sec = chrono::duration<double>(end - start).count();
                double chk = get_checksum(C, N);
                cout << left 
                     << setw(8) << N 
                     << setw(10) << t_num 
                     << setw(10) << p.id 
                     << setw(15) << fixed << setprecision(9) << time_sec 
                     << setw(15) << fixed << setprecision(6) << chk << endl;

                csv << N << "," 
                    << t_num << "," 
                    << p.id << "," 
                    << fixed << setprecision(9) << time_sec << "," 
                    << fixed << setprecision(6) << chk << endl;
            }
        }
        cout << string(70, '-') << endl;
    }

    csv.close();
    return 0;
}
