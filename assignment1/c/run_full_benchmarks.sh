#!/bin/bash

# Compile the program with optimization flags
gcc -O2 c.c -o c

# CSV file for results
CSV_FILE="benchmark_results_full.csv"

# Clear and initialize CSV file
echo "N,Pattern,PatternName,Time" > "$CSV_FILE"

# Matrix sizes to test
SIZES=(256 512 1024 2048)

# Pattern names
PATTERNS=(0 1 2 3 4)
PATTERN_NAMES=("i-j-k" "i-k-j" "k-i-j" "j-i-k" "blocked")

echo "Running comprehensive benchmarks..."
echo "================================================"

# Run benchmarks for each size and pattern
for size in "${SIZES[@]}"; do
    echo "Testing N=$size"
    for i in "${!PATTERNS[@]}"; do
        pattern=${PATTERNS[$i]}
        pattern_name=${PATTERN_NAMES[$i]}
        
        # Run the program and capture full output
        output=$(./c "$size" "$pattern" 2>&1)
        
        # Extract the time value from CSV output format (CSV line)
        time_value=$(echo "$output" | grep "^CSV" | awk -F',' '{print $3}')
        
        echo "  Pattern $pattern ($pattern_name): $time_value seconds"
        
        # Append to CSV file with the extracted time
        if [ ! -z "$time_value" ]; then
            echo "$size,$pattern,$pattern_name,$time_value" >> "$CSV_FILE"
        fi
    done
    echo "---"
done

echo "================================================"
echo "Benchmarks complete! Results saved to $CSV_FILE"
