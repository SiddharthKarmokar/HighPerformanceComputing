/**
 * Matrix Multiplication - 5 Access Pattern Comparison with Multithreading Analysis
 * 
 * This program benchmarks 5 different matrix element access patterns
 * for matrix multiplication (C = A × B) using multiple threads.
 * 
 * FEATURES:
 * - Warmup runs to avoid cold cache effects
 * - Multiple iterations with averaging for consistent results
 * - Proper speedup and efficiency calculations
 * 
 * Access Patterns:
 * 1. IJK - Standard row-major traversal (baseline)
 * 2. IKJ - Optimized row-major (cache-friendly)
 * 3. JIK - Column-major traversal for result matrix C
 * 4. JKI - Column-major for both A and C (worst case)
 * 5. Blocked/Tiled - Cache-optimized with blocking
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <string>
#include <map>

using namespace std;

// ============================================================================
// CONFIGURATION
// ============================================================================
const int WARMUP_RUNS = 2;      // Warmup runs before timing
const int TIMED_RUNS = 5;       // Number of timed runs (take minimum)
const int BLOCK_SIZE = 32;      // Block size for tiled algorithm

// ============================================================================
// GLOBAL DATA
// ============================================================================
int N;                              // Matrix dimension
int NUM_THREADS;                    // Current thread count
vector<vector<double>> A, B, C;     // Matrices

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================
void initialize_matrices(int size) {
    N = size;
    A.assign(N, vector<double>(N));
    B.assign(N, vector<double>(N));
    C.assign(N, vector<double>(N, 0.0));
    
    // Initialize with deterministic values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i + j) % 10 * 0.1;
            B[i][j] = (i - j + N) % 10 * 0.1;
        }
    }
}

void reset_result() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
        }
    }
}

// ============================================================================
// ACCESS PATTERN 1: IJK (Standard/Naive)
// ============================================================================
void worker_ijk(int tid) {
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start = tid * chunk;
    int end = (start + chunk < N) ? start + chunk : N;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// ============================================================================
// ACCESS PATTERN 2: IKJ (Optimized Row-Major)
// ============================================================================
void worker_ikj(int tid) {
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start = tid * chunk;
    int end = (start + chunk < N) ? start + chunk : N;

    for (int i = start; i < end; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
}

// ============================================================================
// ACCESS PATTERN 3: JIK (Column-Major for C)
// ============================================================================
void worker_jik(int tid) {
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start = tid * chunk;
    int end = (start + chunk < N) ? start + chunk : N;

    for (int j = start; j < end; j++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// ============================================================================
// ACCESS PATTERN 4: JKI (Worst Case)
// ============================================================================
void worker_jki(int tid) {
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start = tid * chunk;
    int end = (start + chunk < N) ? start + chunk : N;

    for (int j = start; j < end; j++) {
        for (int k = 0; k < N; k++) {
            double r = B[k][j];
            for (int i = 0; i < N; i++) {
                C[i][j] += A[i][k] * r;
            }
        }
    }
}

// ============================================================================
// ACCESS PATTERN 5: Blocked/Tiled (Cache-Optimized)
// ============================================================================
void worker_blocked(int tid) {
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start = tid * chunk;
    int end = (start + chunk < N) ? start + chunk : N;

    for (int ii = start; ii < end; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                
                int i_max = (ii + BLOCK_SIZE < end) ? ii + BLOCK_SIZE : end;
                int k_max = (kk + BLOCK_SIZE < N) ? kk + BLOCK_SIZE : N;
                int j_max = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;

                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double r = A[i][k];
                        for (int j = jj; j < j_max; j++) {
                            C[i][j] += r * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// BENCHMARK STRUCTURES
// ============================================================================
struct Method {
    string name;
    void (*func)(int);
};

struct BenchmarkResult {
    int size;
    int threads;
    string method;
    double time_seconds;
    double gflops;
    double speedup;
    double efficiency;
};

// ============================================================================
// EXECUTE ONE RUN (helper function)
// ============================================================================
void execute_once(void (*func)(int), int num_threads) {
    NUM_THREADS = num_threads;
    reset_result();
    
    if (num_threads == 1) {
        func(0);
    } else {
        vector<thread> pool;
        for (int t = 0; t < num_threads; t++) {
            pool.push_back(thread(func, t));
        }
        for (int t = 0; t < num_threads; t++) {
            pool[t].join();
        }
    }
}

// ============================================================================
// RUN BENCHMARK WITH WARMUP AND MINIMUM TIME
// ============================================================================
double run_benchmark(void (*func)(int), int num_threads) {
    // Warmup runs (results discarded)
    for (int w = 0; w < WARMUP_RUNS; w++) {
        execute_once(func, num_threads);
    }
    
    // Timed runs - take MINIMUM (standard benchmarking practice)
    // Minimum filters out OS interference; you can't go faster than possible
    double min_time = 1e9;
    
    for (int r = 0; r < TIMED_RUNS; r++) {
        NUM_THREADS = num_threads;
        reset_result();
        
        auto start_time = chrono::high_resolution_clock::now();
        
        if (num_threads == 1) {
            func(0);
        } else {
            vector<thread> pool;
            for (int t = 0; t < num_threads; t++) {
                pool.push_back(thread(func, t));
            }
            for (int t = 0; t < num_threads; t++) {
                pool[t].join();
            }
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> diff = end_time - start_time;
        double elapsed = diff.count();
        
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }
    
    return min_time;  // Return minimum (best case)
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    // Configuration
    vector<int> sizes = {256, 512, 1024, 2048};
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    // All 5 access patterns
    vector<Method> methods = {
        {"IJK",     worker_ijk},
        {"IKJ",     worker_ikj},
        {"JIK",     worker_jik},
        {"JKI",     worker_jki},
        {"Blocked", worker_blocked}
    };

    // Storage for results
    vector<BenchmarkResult> all_results;
    
    // Store single-thread times for speedup calculation
    map<pair<int, string>, double> single_thread_times;

    // Open output files
    ofstream csv_out("matmul_results.csv");
    csv_out << "MatrixSize,Threads,Method,TimeSeconds,GFLOPS,Speedup,Efficiency\n";
    
    ofstream speedup_csv("speedup_analysis.csv");
    speedup_csv << "MatrixSize,Method,Threads,Speedup,Efficiency\n";

    // Print header
    cout << "================================================================\n";
    cout << "  MATRIX MULTIPLICATION - MULTITHREADING PERFORMANCE ANALYSIS\n";
    cout << "================================================================\n";
    cout << "  Comparing 5 Access Patterns with varying thread counts\n";
    cout << "  Warmup runs: " << WARMUP_RUNS << ", Timed runs: " << TIMED_RUNS << " (minimum taken)\n";
    cout << "================================================================\n\n";
    cout << fixed << setprecision(4);

    // ========================================================================
    // PHASE 1: Collect single-thread baselines first
    // ========================================================================
    cout << "Collecting single-thread baselines...\n";
    for (int size : sizes) {
        initialize_matrices(size);
        for (auto& m : methods) {
            double time_1thread = run_benchmark(m.func, 1);
            single_thread_times[{size, m.name}] = time_1thread;
            cout << "  " << size << "x" << size << " " << m.name << ": " << time_1thread << "s\n";
        }
    }
    cout << "\n";

    // ========================================================================
    // PHASE 2: Run all benchmarks
    // ========================================================================
    for (int size : sizes) {
        initialize_matrices(size);
        
        cout << ">>> Matrix Size: " << size << " x " << size << endl;
        cout << string(70, '-') << endl;
        cout << left << setw(10) << "Threads" 
             << setw(10) << "Method" 
             << setw(12) << "Time(s)" 
             << setw(10) << "GFLOPS"
             << setw(10) << "Speedup"
             << "Efficiency" << endl;
        cout << string(70, '-') << endl;
        
        for (auto& m : methods) {
            double time_1thread = single_thread_times[{size, m.name}];
            
            for (int threads : thread_counts) {
                double time_taken;
                
                if (threads == 1) {
                    time_taken = time_1thread;
                } else {
                    time_taken = run_benchmark(m.func, threads);
                }
                
                // Calculate metrics
                double gflops = (2.0 * size * size * size) / (time_taken * 1e9);
                double speedup = time_1thread / time_taken;
                double efficiency = (speedup / threads) * 100.0;
                
                // Store result
                BenchmarkResult result;
                result.size = size;
                result.threads = threads;
                result.method = m.name;
                result.time_seconds = time_taken;
                result.gflops = gflops;
                result.speedup = speedup;
                result.efficiency = efficiency;
                all_results.push_back(result);
                
                // Write to CSV
                csv_out << size << "," << threads << "," << m.name << ","
                        << time_taken << "," << gflops << "," 
                        << speedup << "," << efficiency << "\n";
                
                speedup_csv << size << "," << m.name << "," << threads << ","
                            << speedup << "," << efficiency << "\n";
                
                // Print to console
                cout << left << setw(10) << threads 
                     << setw(10) << m.name 
                     << setw(12) << time_taken 
                     << setw(10) << gflops
                     << setw(10) << speedup
                     << efficiency << "%" << endl;
            }
        }
        cout << endl;
    }

    csv_out.close();
    speedup_csv.close();

    // ========================================================================
    // PHASE 3: Summary and Analysis
    // ========================================================================
    cout << "\n================================================================\n";
    cout << "  MULTITHREADING IMPROVEMENT ANALYSIS\n";
    cout << "================================================================\n\n";
    
    // Find best configuration for largest matrix
    int largest_size = sizes.back();
    double best_time = 1e9;
    string best_method = "";
    int best_threads = 0;
    double best_speedup = 0;
    
    for (auto& r : all_results) {
        if (r.size == largest_size && r.time_seconds < best_time) {
            best_time = r.time_seconds;
            best_method = r.method;
            best_threads = r.threads;
            best_speedup = r.speedup;
        }
    }
    
    cout << "OPTIMAL CONFIGURATION for " << largest_size << "x" << largest_size << ":\n";
    cout << "  Best Access Pattern: " << best_method << endl;
    cout << "  Optimal Thread Count: " << best_threads << endl;
    cout << "  Execution Time: " << best_time << " seconds\n";
    cout << "  Speedup: " << best_speedup << "x (vs single thread)\n\n";
    
    // ========================================================================
    // COMPARISON METHODS SUMMARY
    // ========================================================================
    cout << "================================================================\n";
    cout << "  COMPARISON METRICS EXPLAINED\n";
    cout << "================================================================\n\n";
    
    cout << "1. EXECUTION TIME COMPARISON\n";
    cout << "   - Direct measurement of computation time\n";
    cout << "   - Lower is better\n";
    cout << "   - Shows absolute performance difference\n\n";
    
    cout << "2. SPEEDUP COMPARISON\n";
    cout << "   - Speedup = Time(1 thread) / Time(N threads)\n";
    cout << "   - Measures how much faster with more threads\n";
    cout << "   - Ideal speedup = N (linear scaling)\n";
    cout << "   - Values < N indicate overhead or bottlenecks\n\n";
    
    cout << "3. EFFICIENCY COMPARISON\n";
    cout << "   - Efficiency = (Speedup / N) x 100%\n";
    cout << "   - Shows how well threads are utilized\n";
    cout << "   - 100% = perfect utilization\n";
    cout << "   - Lower values indicate diminishing returns\n\n";
    
    cout << "4. GFLOPS COMPARISON\n";
    cout << "   - GFLOPS = 2 x N^3 / Time / 10^9\n";
    cout << "   - Measures computational throughput\n";
    cout << "   - Higher is better\n";
    cout << "   - Allows comparison across different matrix sizes\n\n";
    
    cout << "5. SCALABILITY ANALYSIS\n";
    cout << "   - How performance changes with increasing threads\n";
    cout << "   - Strong scaling: fixed problem size, varying threads\n";
    cout << "   - Helps identify optimal thread count\n\n";

    // ========================================================================
    // Per-Method Analysis
    // ========================================================================
    cout << "================================================================\n";
    cout << "  SPEEDUP BY ACCESS PATTERN (for " << largest_size << "x" << largest_size << ")\n";
    cout << "================================================================\n\n";
    
    for (auto& m : methods) {
        cout << m.name << " Pattern:\n";
        cout << "  Threads  |  Speedup  |  Efficiency\n";
        cout << "  ---------|-----------|------------\n";
        
        for (auto& r : all_results) {
            if (r.size == largest_size && r.method == m.name) {
                cout << "     " << setw(2) << r.threads 
                     << "    |   " << setw(6) << r.speedup 
                     << "x |   " << setw(6) << r.efficiency << "%\n";
            }
        }
        cout << endl;
    }

    cout << "================================================================\n";
    cout << "  OUTPUT FILES GENERATED\n";
    cout << "================================================================\n";
    cout << "  1. matmul_results.csv    - Complete benchmark results\n";
    cout << "  2. speedup_analysis.csv  - Speedup and efficiency data\n";
    cout << "  \n";
    cout << "  Run 'python plot_results.py' to generate comparison plots.\n";
    cout << "================================================================\n";

    return 0;
}
