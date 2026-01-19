#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>  // For file output
#include <algorithm> // For min, max
#include <string>
#include <iomanip>  // For std::setw, std::fixed

using namespace std;

// --- Global Data to reduce allocation overhead ---
int N;
int NUM_THREADS;
vector<vector<int>> MATRIX;
vector<long long> PARTIAL_SUMS;

// --- Helper 1: Initialize Matrix ---
void initialize_matrix(int size) {
    N = size;
    // Fill with 1s so we don't waste time on random generation during benchmark
    // and sums are easy to verify.
    MATRIX.assign(N, vector<int>(N, 1)); 
}

// --- Helper 2: Z-Curve Coordinate Decoder ---
// Converts a linear index 'z' into (row, col) coordinates using bit interleaving
void z_to_xy(int z, int &y, int &x) {
    x = 0; y = 0;
    for (int i = 0; i < 32; i++) {
        if (z & (1 << (2*i)))     x |= (1 << i);
        if (z & (1 << (2*i + 1))) y |= (1 << i);
    }
}

// ==========================================
// 5 ACCESS PATTERNS (WORKER FUNCTIONS)
// ==========================================

// 1. SNAKE (Serpentine): Left-Right then Right-Left
void worker_snake(int tid) {
    long long sum = 0;
    for (int i = tid; i < N; i += NUM_THREADS) {
        if (i % 2 == 0) {
            for (int j = 0; j < N; j++) sum += MATRIX[i][j];
        } else {
            for (int j = N - 1; j >= 0; j--) sum += MATRIX[i][j];
        }
    }
    PARTIAL_SUMS[tid] = sum;
}

// 2. CHECKERBOARD (Red-Black): Even sums then Odd sums
void worker_checkerboard(int tid) {
    long long sum = 0;
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS; 
    int start = tid * chunk;
    int end = min(start + chunk, N);
    
    if (start >= N) return;

    // Pass 1: Even (Red)
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            if ((i + j) % 2 == 0) sum += MATRIX[i][j];
        }
    }
    // Pass 2: Odd (Black)
    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            if ((i + j) % 2 != 0) sum += MATRIX[i][j];
        }
    }
    PARTIAL_SUMS[tid] = sum;
}

// 3. STRIDED: Skip elements (Step = 2)
void worker_strided(int tid) {
    long long sum = 0;
    int stride = 2;
    for (int i = tid; i < N; i += NUM_THREADS) {
        for (int j = 0; j < N; j += stride) sum += MATRIX[i][j];
    }
    PARTIAL_SUMS[tid] = sum;
}

// 4. SPIRAL: Process by concentric rings
void worker_spiral(int tid) {
    long long sum = 0;
    int total_rings = N / 2;
    for (int ring = tid; ring < total_rings; ring += NUM_THREADS) {
        int start = ring;
        int end = N - 1 - ring;
        // Top, Right, Bottom, Left
        for (int j = start; j < end; j++) sum += MATRIX[start][j];
        for (int i = start; i < end; i++) sum += MATRIX[i][end];
        for (int j = end; j > start; j--) sum += MATRIX[end][j];
        for (int i = end; i > start; i--) sum += MATRIX[i][start];
    }
    PARTIAL_SUMS[tid] = sum;
}

// 5. Z-CURVE (Morton Order): Space-filling curve
void worker_z_curve(int tid) {
    long long sum = 0;
    long long total_elements = (long long)N * N;
    long long chunk = (total_elements + NUM_THREADS - 1) / NUM_THREADS;
    
    long long start_z = tid * chunk;
    long long end_z = min(start_z + chunk, total_elements);

    for (long long z = start_z; z < end_z; z++) {
        int r, c;
        z_to_xy(z, r, c); 
        if (r < N && c < N) sum += MATRIX[r][c];
    }
    PARTIAL_SUMS[tid] = sum;
}

// ==========================================
// MAIN EXECUTION LOOP
// ==========================================

int main() {
    // Open CSV File for writing
    ofstream csvFile("benchmark_results.csv");
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open benchmark_results.csv for writing.\n";
        return 1;
    }
    
    // Write CSV Headers
    csvFile << "MatrixSize,Threads,Method,TimeSeconds\n"; 

    // Configuration
    vector<int> sizes = {256, 512, 1024, 2048};
    vector<int> thread_counts = {1, 2, 4, 16, 32, 64, 128, 256};
    
    // Method Definition Wrapper
    struct Method { string name; void (*func)(int); };
    vector<Method> methods = {
        {"Snake", worker_snake},
        {"Checkerboard", worker_checkerboard},
        {"Strided", worker_strided},
        {"Spiral", worker_spiral},
        {"ZCurve", worker_z_curve}
    };

    cout << "==========================================================\n";
    cout << " MATRIX TRAVERSAL BENCHMARK \n";
    cout << " Output will be saved to: benchmark_results.csv\n";
    cout << "==========================================================\n";

    // Loop 1: Matrix Sizes
    for (int size : sizes) {
        initialize_matrix(size);
        
        cout << "\n>>> Processing Size: " << size << " x " << size << endl;
        cout << string(60, '-') << endl;
        cout << left << setw(10) << "Threads" << setw(15) << "Method" << "Time (s)" << endl;
        cout << string(60, '-') << endl;

        string best_method_name = "";
        double best_time = 99999.0;
        int best_threads = 0;

        // Loop 2: Thread Counts
        for (int t_count : thread_counts) {
            NUM_THREADS = t_count;
            PARTIAL_SUMS.assign(NUM_THREADS, 0);

            // Loop 3: Access Methods
            for (auto& m : methods) {
                // Reset Sums
                fill(PARTIAL_SUMS.begin(), PARTIAL_SUMS.end(), 0);

                // Start Timer
                auto start = chrono::high_resolution_clock::now();
                
                // Launch Threads
                vector<thread> pool;
                for(int t=0; t<NUM_THREADS; t++) {
                    pool.push_back(thread(m.func, t));
                }
                // Join Threads
                for(auto& th : pool) th.join();

                // Stop Timer
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> diff = end - start;
                double duration = diff.count();

                // 1. Print to Console
                cout << left << setw(10) << t_count 
                     << setw(15) << m.name 
                     << fixed << setprecision(5) << duration << " s" << endl;

                // 2. Save to CSV
                csvFile << size << "," << t_count << "," << m.name << "," << duration << "\n";

                // Track Best Performance for this size
                if (duration < best_time) {
                    best_time = duration;
                    best_method_name = m.name;
                    best_threads = t_count;
                }
            }
        }
        cout << string(60, '-') << endl;
        cout << "WINNER for " << size << "x" << size << " -> " 
             << best_method_name << " (" << best_threads << " threads) : " 
             << best_time << " s\n";
        cout << string(60, '-') << endl;
    }

    csvFile.close();
    cout << "\nDone! Results saved to 'benchmark_results.csv'.\n";
    return 0;
}
