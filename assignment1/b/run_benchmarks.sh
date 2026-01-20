#!/usr/bin/env bash
set -e

############################
# CONFIGURATION
############################
CC=gcc
CFLAGS="-O3 -pthread -march=native"
BIN=matadd_opt
OUT=results_optimized.csv

# matrix sizes (powers of 2)
NS=(256 512 1024 2048)

# thread counts
THREADS=(1 2 4 8 16)

# patterns to test
PATTERNS=(0 1 2 3 4 5)

# repeats inside program
REPEATS=5

# enable perf? (0/1)
USE_PERF=0

############################
# BUILD
############################
echo "Compiling optimized binary..."
$CC $CFLAGS optimized_matadd.c -o $BIN

############################
# CSV HEADER
############################
echo "N,threads,pattern,sec,checksum" > $OUT

############################
# RUN BENCHMARKS
############################
for T in "${THREADS[@]}"; do
  echo "==== THREADS = $T ===="

  for N in "${NS[@]}"; do
    echo "  N = $N"

    for P in "${PATTERNS[@]}"; do
      echo "    pattern = $P"

      if [[ $USE_PERF -eq 1 ]]; then
        perf stat -x, \
          -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
          ./$BIN $N $T $P $REPEATS \
          2>> perf_T${T}_N${N}_P${P}.csv \
          | grep "^CSV" | sed 's/^CSV,//' >> $OUT
      else
        ./$BIN $N $T $P $REPEATS \
          | grep "^CSV" | sed 's/^CSV,//' >> $OUT
      fi

    done
  done
done

echo
echo "======================================"
echo "Benchmark complete."
echo "Results written to $OUT"
echo "Run: python plot_result.py"
echo "======================================"
