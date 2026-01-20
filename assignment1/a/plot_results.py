import matplotlib.pyplot as plt
import csv
import os


filename = 'results.csv'
if not os.path.exists(filename):
    print("Error: results.csv not found")
    exit(1)

data = []
try:
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['N'] = int(row['N'])
            row['threads'] = int(row['threads'])
            row['sec'] = float(row['sec'])
            data.append(row)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

data_by_thread = {}
patterns = set()

for row in data:
    t = row['threads']
    p = row['pattern'] 
    n = row['N']
    s = row['sec']
    
    if t not in data_by_thread:
        data_by_thread[t] = {}
    if p not in data_by_thread[t]:
        data_by_thread[t][p] = []
    
    data_by_thread[t][p].append((n, s))
    patterns.add(p)

sorted_patterns = sorted(list(patterns))

pattern_labels = {
    0: "blocked_32",
    1: "col_major",
    2: "cyclic_rows",
    3: "linear_flat",
    4: "row_major_rows_per_thread",
    5: "unroll4"
}
if 1 in data_by_thread:
    plt.figure(figsize=(10, 6))
    for p in sorted_patterns:
        if p in data_by_thread[1]:
            pts = sorted(data_by_thread[1][p]) 
            xs = [x[0] for x in pts]
            ys = [x[1] for x in pts]
            
            
            try:
                p_id = int(p)
                label_name = pattern_labels.get(p_id, p)
            except:
                label_name = p
                
            plt.plot(xs, ys, marker='o', label=label_name)
            
    plt.title('Execution Time vs Matrix Size (Single Thread)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Matrix Dimension (N)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('plot_time_vs_size_single_thread.png')
    plt.close()
else:
    print("Warning: No single thread data found.")

n_target = 2048
best_times = {}

for row in data:
    if row['N'] == n_target:
        p = row['pattern'] 
        s = row['sec']
        if p not in best_times or s < best_times[p]:
            best_times[p] = s

if best_times:
    plt.figure(figsize=(10, 6))
    p_names = sorted(best_times.keys())
    times = [best_times[p] for p in p_names]
    
    plt.bar(p_names, times)
    plt.title(f'Best Observed Time per Pattern (N={n_target})')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('plot_best_time_per_pattern.png')
    plt.close()

print("Plots generated successfully.")
