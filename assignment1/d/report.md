# Matrix Multiplication: Access Pattern and Multithreading Analysis

## 1. Introduction

This report analyzes **5 different matrix element access patterns** for matrix multiplication using multiple threads. We evaluate performance across matrix sizes (256×256, 512×512, 1024×1024, 2048×2048) and demonstrate the benefits of multithreading through various comparison methods.

---

## 2. How to Compile and Run

### Compilation (No Special Flags Required)

```bash
# Windows
g++ matmul_patterns.cpp -o matmul_patterns.exe

# Linux/Mac
g++ matmul_patterns.cpp -o matmul_patterns
```

### Running the Benchmark

```bash
./matmul_patterns
```

### Generating Plots

```bash
python plot_results.py
```

---

## 3. The 5 Access Patterns

### Pattern 1: IJK (Standard/Naive)

**Loop Order:** `i → j → k`

```cpp
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

| Matrix | Access | Cache Behavior |
|--------|--------|----------------|
| A[i][k] | Row-wise | ✅ Good |
| B[k][j] | Column-wise | ❌ Bad (stride-N) |
| C[i][j] | Single element | ✅ Good |

**Benefits:** Simple, intuitive, matches mathematical definition.

**Limitations:** Poor cache utilization for B; each B access may cause cache miss.

---

### Pattern 2: IKJ (Optimized Row-Major)

**Loop Order:** `i → k → j`

```cpp
for (i = 0; i < N; i++)
    for (k = 0; k < N; k++) {
        r = A[i][k];
        for (j = 0; j < N; j++)
            C[i][j] += r * B[k][j];
    }
```

| Matrix | Access | Cache Behavior |
|--------|--------|----------------|
| A[i][k] | Element reuse | ✅ Excellent |
| B[k][j] | Row-wise | ✅ Excellent |
| C[i][j] | Row-wise | ✅ Excellent |

**Benefits:** All matrices accessed sequentially; enables prefetching; best for medium matrices.

**Limitations:** Requires accumulation (+=).

---

### Pattern 3: JIK (Column-Major for C)

**Loop Order:** `j → i → k`

```cpp
for (j = 0; j < N; j++)
    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

| Matrix | Access | Cache Behavior |
|--------|--------|----------------|
| A[i][k] | Row-wise | ⚠️ Moderate |
| B[k][j] | Single column | ⚠️ Moderate |
| C[i][j] | Column-wise | ❌ Bad |

**Benefits:** Good for column-major storage (Fortran).

**Limitations:** C accessed column-wise in row-major storage causes cache misses.

---

### Pattern 4: JKI (Worst Case)

**Loop Order:** `j → k → i`

```cpp
for (j = 0; j < N; j++)
    for (k = 0; k < N; k++) {
        r = B[k][j];
        for (i = 0; i < N; i++)
            C[i][j] += A[i][k] * r;
    }
```

| Matrix | Access | Cache Behavior |
|--------|--------|----------------|
| A[i][k] | Column-wise | ❌ Bad |
| B[k][j] | Register reuse | ✅ Good |
| C[i][j] | Column-wise | ❌ Bad |

**Benefits:** B value kept in register.

**Limitations:** Both A and C have stride-N access; maximum cache misses.

---

### Pattern 5: Blocked/Tiled (Cache-Optimized)

**Loop Order:** Block iteration with IKJ inside

```cpp
for (ii = 0; ii < N; ii += BLOCK)
    for (kk = 0; kk < N; kk += BLOCK)
        for (jj = 0; jj < N; jj += BLOCK)
            // Mini-multiply within block (IKJ)
            for (i = ii; i < ii+BLOCK; i++)
                for (k = kk; k < kk+BLOCK; k++)
                    for (j = jj; j < jj+BLOCK; j++)
                        C[i][j] += A[i][k] * B[k][j];
```

| Matrix | Access | Cache Behavior |
|--------|--------|----------------|
| A block | Fits in cache | ✅ Excellent |
| B block | Fits in cache | ✅ Excellent |
| C block | Fits in cache | ✅ Excellent |

**Benefits:** Blocks fit in L1/L2 cache; maximum reuse before eviction; optimal for large matrices.

**Limitations:** More complex; block size needs tuning.

---

## 4. Ways to Compare Multithreading Improvement

### Method 1: Execution Time Comparison

**Formula:** Direct measurement in seconds

**Interpretation:**
- Lower time = Better performance
- Compare same configuration with different thread counts
- Shows absolute performance difference

**Example:**
| Pattern | 1 Thread | 8 Threads | Improvement |
|---------|----------|-----------|-------------|
| IKJ | 10.5s | 1.8s | 83% faster |

---

### Method 2: Speedup Analysis

**Formula:** 
$$\text{Speedup} = \frac{T_1}{T_n}$$

Where:
- $T_1$ = Time with 1 thread
- $T_n$ = Time with n threads

**Interpretation:**
- Speedup of 4× means 4 times faster
- **Ideal speedup = n** (linear scaling)
- Speedup < n indicates overhead or bottlenecks
- Speedup > n is rare (super-linear, due to cache effects)

**Example:**
| Pattern | Speedup (8 threads) | vs Ideal |
|---------|---------------------|----------|
| IKJ | 5.83× | 73% of ideal |
| JKI | 3.2× | 40% of ideal |

---

### Method 3: Efficiency Analysis

**Formula:**
$$\text{Efficiency} = \frac{\text{Speedup}}{n} \times 100\%$$

**Interpretation:**
- 100% = Perfect utilization (all threads fully productive)
- <100% = Some overhead or idle time
- Efficiency drops as thread count increases (diminishing returns)

**Example:**
| Threads | Speedup | Efficiency |
|---------|---------|------------|
| 2 | 1.9× | 95% |
| 4 | 3.5× | 87.5% |
| 8 | 5.8× | 72.5% |
| 16 | 8.2× | 51.25% |

---

### Method 4: GFLOPS (Throughput) Comparison

**Formula:**
$$\text{GFLOPS} = \frac{2 \times N^3}{\text{Time} \times 10^9}$$

The factor of 2 counts both multiply and add operations.

**Interpretation:**
- Higher GFLOPS = More computations per second
- Allows fair comparison across different matrix sizes
- Hardware-independent performance metric

**Example:**
| Pattern | GFLOPS (1 thread) | GFLOPS (8 threads) |
|---------|-------------------|---------------------|
| IKJ | 1.2 | 7.0 |
| Blocked | 1.4 | 8.2 |

---

### Method 5: Scalability Analysis (Strong Scaling)

**What it measures:** How performance improves as threads increase for a fixed problem size.

**Graph:** Speedup vs Thread Count (compared to ideal linear line)

**Interpretation:**
- Lines close to ideal = good scalability
- Flattening curve = diminishing returns
- Helps identify optimal thread count

---

## 5. Expected Performance Ranking

Based on cache behavior analysis:

| Rank | Pattern | Why |
|------|---------|-----|
| 1 | Blocked | Explicit cache management |
| 2 | IKJ | All sequential access |
| 3 | IJK | Only B has issues |
| 4 | JIK | C accessed poorly |
| 5 | JKI | Both A and C accessed poorly |

---

## 6. Finding Optimal Thread Count

The optimal thread count depends on:

1. **Physical cores:** Using more threads than cores adds context switching overhead
2. **Memory bandwidth:** Memory-bound patterns may saturate bandwidth
3. **Cache contention:** Threads compete for shared L3 cache

**General guidelines:**
- Start with number of physical cores
- Test 1, 2, 4, 8, 16 threads
- Look for point where efficiency drops below 50%

---

## 7. Output Files

| File | Description |
|------|-------------|
| `matmul_patterns.cpp` | Main benchmark program |
| `plot_results.py` | Plotting script |
| `matmul_results.csv` | Complete results (Time, GFLOPS, Speedup, Efficiency) |
| `speedup_analysis.csv` | Focused speedup data |
| `plots/` | Generated comparison plots |

### Generated Plots

1. `01_time_comparison.png` - Execution time comparison
2. `02_speedup_analysis.png` - Speedup vs threads and bar chart
3. `03_efficiency_analysis.png` - Thread efficiency
4. `04_gflops_comparison.png` - GFLOPS performance
5. `05_scalability_analysis.png` - Scalability across all sizes
6. `06_summary_dashboard.png` - Comprehensive dashboard

---

## 8. Conclusion

### Optimal Access Pattern
**IKJ or Blocked** - Both provide excellent cache utilization. IKJ is simpler; Blocked is better for very large matrices.

### Optimal Thread Count
Typically equals the number of physical CPU cores. Beyond that, efficiency drops due to:
- Thread creation overhead
- Cache contention
- Memory bandwidth saturation

### Key Insights
1. Access pattern can impact performance by 10×+ for large matrices
2. Multithreading provides near-linear speedup up to core count
3. Efficiency drops significantly beyond physical core count
4. Blocked pattern provides best scalability for large matrices
