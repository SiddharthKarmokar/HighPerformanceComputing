"""
Matrix Multiplication - Performance Analysis Plotting Script

Generates comparison plots showing multithreading improvements:
1. Time vs Matrix Size
2. Time vs Threads (Scalability)
3. Speedup Comparison
4. Efficiency Analysis
5. GFLOPS Performance
6. Comprehensive Summary

Usage: python plot_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    """Load benchmark results from CSV files."""
    if not os.path.exists('matmul_results.csv'):
        print("Error: matmul_results.csv not found.")
        print("Please run the matmul_patterns benchmark first.")
        return None
    
    df = pd.read_csv('matmul_results.csv')
    print(f"Loaded {len(df)} benchmark records.")
    return df

def create_output_directory():
    """Create plots directory."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    print("Plots will be saved to 'plots/' directory.\n")

def plot_time_comparison(df):
    """Plot 1: Execution Time vs Matrix Size for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    # Left: Best time per method (using optimal threads)
    ax1 = axes[0]
    for idx, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        best_times = method_data.groupby('MatrixSize')['TimeSeconds'].min()
        ax1.plot(best_times.index, best_times.values, 
                marker='o', linewidth=2, markersize=8,
                label=method, color=colors[idx])
    
    ax1.set_title('Execution Time vs Matrix Size\n(Best Thread Configuration)', fontweight='bold')
    ax1.set_xlabel('Matrix Size (N)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(sorted(df['MatrixSize'].unique()))
    
    # Right: Single thread vs Multi-thread comparison for largest matrix
    ax2 = axes[1]
    largest_size = df['MatrixSize'].max()
    subset = df[df['MatrixSize'] == largest_size]
    
    single_thread = subset[subset['Threads'] == 1]['TimeSeconds'].values
    max_threads = subset['Threads'].max()
    multi_thread = subset[subset['Threads'] == max_threads]['TimeSeconds'].values
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, single_thread, width, label='1 Thread', color='lightcoral')
    bars2 = ax2.bar(x + width/2, multi_thread, width, label=f'{max_threads} Threads', color='steelblue')
    
    ax2.set_title(f'Single vs Multi-Thread Comparison\n({largest_size}×{largest_size} Matrix)', fontweight='bold')
    ax2.set_xlabel('Access Pattern')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/01_time_comparison.png', dpi=150)
    print("  ✓ plots/01_time_comparison.png")
    plt.close()

def plot_speedup_analysis(df):
    """Plot 2: Speedup comparison - how much faster with more threads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'D', 'v']
    
    largest_size = df['MatrixSize'].max()
    subset = df[df['MatrixSize'] == largest_size]
    
    # Left: Speedup vs Threads
    ax1 = axes[0]
    threads = sorted(subset['Threads'].unique())
    
    # Plot ideal speedup line
    ax1.plot(threads, threads, 'k--', linewidth=2, label='Ideal (Linear)')
    
    for idx, method in enumerate(methods):
        method_data = subset[subset['Method'] == method].sort_values('Threads')
        ax1.plot(method_data['Threads'], method_data['Speedup'], 
                marker=markers[idx], linewidth=2, markersize=8,
                label=method, color=colors[idx])
    
    ax1.set_title(f'Speedup vs Thread Count\n({largest_size}×{largest_size} Matrix)', fontweight='bold')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup (×)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_xticks(threads)
    
    # Right: Speedup bar chart at max threads
    ax2 = axes[1]
    max_threads = subset['Threads'].max()
    max_thread_data = subset[subset['Threads'] == max_threads]
    
    bars = ax2.bar(max_thread_data['Method'], max_thread_data['Speedup'], 
                   color=colors, edgecolor='black')
    
    # Add ideal line
    ax2.axhline(y=max_threads, color='red', linestyle='--', linewidth=2, label=f'Ideal ({max_threads}×)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}×', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title(f'Speedup with {max_threads} Threads\n({largest_size}×{largest_size} Matrix)', fontweight='bold')
    ax2.set_xlabel('Access Pattern')
    ax2.set_ylabel('Speedup (×)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/02_speedup_analysis.png', dpi=150)
    print("  ✓ plots/02_speedup_analysis.png")
    plt.close()

def plot_efficiency_analysis(df):
    """Plot 3: Thread efficiency - how well threads are utilized."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    largest_size = df['MatrixSize'].max()
    subset = df[df['MatrixSize'] == largest_size]
    
    # Left: Efficiency vs Threads
    ax1 = axes[0]
    threads = sorted(subset['Threads'].unique())
    
    for idx, method in enumerate(methods):
        method_data = subset[subset['Method'] == method].sort_values('Threads')
        ax1.plot(method_data['Threads'], method_data['Efficiency'], 
                marker='o', linewidth=2, markersize=8,
                label=method, color=colors[idx])
    
    ax1.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% Efficiency')
    ax1.set_title(f'Thread Efficiency vs Thread Count\n({largest_size}×{largest_size} Matrix)', fontweight='bold')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Efficiency (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xticks(threads)
    ax1.set_ylim(0, 120)
    
    # Right: Efficiency heatmap
    ax2 = axes[1]
    
    sizes = sorted(df['MatrixSize'].unique())
    efficiency_matrix = np.zeros((len(methods), len(sizes)))
    
    max_threads = df['Threads'].max()
    for i, method in enumerate(methods):
        for j, size in enumerate(sizes):
            subset_data = df[(df['Method'] == method) & 
                           (df['MatrixSize'] == size) & 
                           (df['Threads'] == max_threads)]
            if len(subset_data) > 0:
                efficiency_matrix[i, j] = subset_data['Efficiency'].values[0]
    
    im = ax2.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax2, label='Efficiency (%)')
    
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(sizes)):
            ax2.text(j, i, f'{efficiency_matrix[i, j]:.0f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_title(f'Efficiency Heatmap ({max_threads} Threads)', fontweight='bold')
    ax2.set_xlabel('Matrix Size')
    
    plt.tight_layout()
    plt.savefig('plots/03_efficiency_analysis.png', dpi=150)
    print("  ✓ plots/03_efficiency_analysis.png")
    plt.close()

def plot_gflops_comparison(df):
    """Plot 4: GFLOPS performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    largest_size = df['MatrixSize'].max()
    max_threads = df['Threads'].max()
    
    # Left: GFLOPS vs Threads
    ax1 = axes[0]
    subset = df[df['MatrixSize'] == largest_size]
    
    for idx, method in enumerate(methods):
        method_data = subset[subset['Method'] == method].sort_values('Threads')
        ax1.plot(method_data['Threads'], method_data['GFLOPS'], 
                marker='o', linewidth=2, markersize=8,
                label=method, color=colors[idx])
    
    ax1.set_title(f'GFLOPS vs Thread Count\n({largest_size}×{largest_size} Matrix)', fontweight='bold')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('GFLOPS (Higher is Better)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(sorted(subset['Threads'].unique()))
    
    # Right: Peak GFLOPS comparison
    ax2 = axes[1]
    peak_data = df[(df['MatrixSize'] == largest_size) & (df['Threads'] == max_threads)]
    
    bars = ax2.bar(peak_data['Method'], peak_data['GFLOPS'], 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(peak_data['GFLOPS']),
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_title(f'Peak GFLOPS ({largest_size}×{largest_size}, {max_threads} Threads)', fontweight='bold')
    ax2.set_xlabel('Access Pattern')
    ax2.set_ylabel('GFLOPS')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/04_gflops_comparison.png', dpi=150)
    print("  ✓ plots/04_gflops_comparison.png")
    plt.close()

def plot_scalability_analysis(df):
    """Plot 5: Scalability analysis across all matrix sizes."""
    sizes = sorted(df['MatrixSize'].unique())
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        subset = df[df['MatrixSize'] == size]
        threads = sorted(subset['Threads'].unique())
        
        # Plot ideal speedup
        ax.plot(threads, threads, 'k--', linewidth=2, alpha=0.5, label='Ideal')
        
        for midx, method in enumerate(methods):
            method_data = subset[subset['Method'] == method].sort_values('Threads')
            ax.plot(method_data['Threads'], method_data['Speedup'], 
                   marker='o', linewidth=2, markersize=6,
                   label=method, color=colors[midx])
        
        ax.set_title(f'Matrix Size: {size}×{size}', fontweight='bold')
        ax.set_xlabel('Threads')
        ax.set_ylabel('Speedup')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks(threads)
    
    plt.suptitle('Scalability Analysis: Speedup vs Threads by Matrix Size', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('plots/05_scalability_analysis.png', dpi=150)
    print("  ✓ plots/05_scalability_analysis.png")
    plt.close()

def plot_summary_dashboard(df):
    """Plot 6: Comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    largest_size = df['MatrixSize'].max()
    max_threads = df['Threads'].max()
    methods = df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    # 1. Time comparison (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    subset = df[df['MatrixSize'] == largest_size]
    single = subset[subset['Threads'] == 1]['TimeSeconds'].values
    multi = subset[subset['Threads'] == max_threads]['TimeSeconds'].values
    
    x = np.arange(len(methods))
    ax1.bar(x - 0.2, single, 0.4, label='1 Thread', color='salmon')
    ax1.bar(x + 0.2, multi, 0.4, label=f'{max_threads} Threads', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Execution Time', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Speedup (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    peak_data = subset[subset['Threads'] == max_threads]
    bars = ax2.bar(peak_data['Method'], peak_data['Speedup'], color=colors)
    ax2.axhline(y=max_threads, color='red', linestyle='--', label='Ideal')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title(f'Speedup ({max_threads} Threads)', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Efficiency (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    bars = ax3.bar(peak_data['Method'], peak_data['Efficiency'], color=colors)
    ax3.axhline(y=100, color='green', linestyle='--', label='100%')
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Thread Efficiency', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. GFLOPS (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    bars = ax4.bar(peak_data['Method'], peak_data['GFLOPS'], color=colors)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('GFLOPS')
    ax4.set_title('Performance (GFLOPS)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Speedup vs Threads (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    threads = sorted(subset['Threads'].unique())
    ax5.plot(threads, threads, 'k--', linewidth=2, label='Ideal')
    for idx, method in enumerate(methods):
        method_data = subset[subset['Method'] == method].sort_values('Threads')
        ax5.plot(method_data['Threads'], method_data['Speedup'], 
                marker='o', linewidth=2, label=method, color=colors[idx])
    ax5.set_xlabel('Threads')
    ax5.set_ylabel('Speedup')
    ax5.set_title('Scalability', fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary text (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Find best configuration
    best_row = peak_data.loc[peak_data['TimeSeconds'].idxmin()]
    
    summary_text = f"""
    SUMMARY FOR {largest_size}×{largest_size} MATRIX
    {'='*40}
    
    Best Access Pattern: {best_row['Method']}
    Optimal Threads: {max_threads}
    Execution Time: {best_row['TimeSeconds']:.4f} s
    Speedup: {best_row['Speedup']:.2f}×
    Efficiency: {best_row['Efficiency']:.1f}%
    Performance: {best_row['GFLOPS']:.2f} GFLOPS
    
    {'='*40}
    COMPARISON METHODS:
    
    1. Time: Direct execution measurement
    2. Speedup: T₁/Tₙ (how much faster)
    3. Efficiency: Speedup/n × 100%
    4. GFLOPS: Computational throughput
    5. Scalability: Performance vs threads
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Matrix Multiplication: Multithreading Performance Dashboard', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/06_summary_dashboard.png', dpi=150)
    print("  ✓ plots/06_summary_dashboard.png")
    plt.close()

def generate_text_report(df):
    """Generate a text-based comparison report."""
    largest_size = df['MatrixSize'].max()
    max_threads = df['Threads'].max()
    methods = df['Method'].unique()
    
    with open('plots/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  MULTITHREADING PERFORMANCE COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Matrix Size: {largest_size} × {largest_size}\n")
        f.write(f"Thread Counts Tested: {sorted(df['Threads'].unique())}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("  COMPARISON METHOD 1: EXECUTION TIME\n")
        f.write("-" * 70 + "\n")
        f.write("  Lower time = Better performance\n\n")
        f.write(f"  {'Method':<12} {'1 Thread':<12} {f'{max_threads} Threads':<12} {'Improvement':<12}\n")
        f.write("  " + "-" * 48 + "\n")
        
        subset = df[df['MatrixSize'] == largest_size]
        for method in methods:
            t1 = subset[(subset['Method'] == method) & (subset['Threads'] == 1)]['TimeSeconds'].values[0]
            tn = subset[(subset['Method'] == method) & (subset['Threads'] == max_threads)]['TimeSeconds'].values[0]
            improvement = ((t1 - tn) / t1) * 100
            f.write(f"  {method:<12} {t1:<12.4f} {tn:<12.4f} {improvement:.1f}%\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("  COMPARISON METHOD 2: SPEEDUP\n")
        f.write("-" * 70 + "\n")
        f.write("  Speedup = Time(1 thread) / Time(N threads)\n")
        f.write(f"  Ideal speedup with {max_threads} threads = {max_threads}×\n\n")
        f.write(f"  {'Method':<12} {'Speedup':<12} {'vs Ideal':<12}\n")
        f.write("  " + "-" * 36 + "\n")
        
        for method in methods:
            speedup = subset[(subset['Method'] == method) & (subset['Threads'] == max_threads)]['Speedup'].values[0]
            vs_ideal = (speedup / max_threads) * 100
            f.write(f"  {method:<12} {speedup:<12.2f}× {vs_ideal:.1f}%\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("  COMPARISON METHOD 3: EFFICIENCY\n")
        f.write("-" * 70 + "\n")
        f.write("  Efficiency = (Speedup / N) × 100%\n")
        f.write("  100% = perfect thread utilization\n\n")
        f.write(f"  {'Method':<12} {'Efficiency':<12}\n")
        f.write("  " + "-" * 24 + "\n")
        
        for method in methods:
            eff = subset[(subset['Method'] == method) & (subset['Threads'] == max_threads)]['Efficiency'].values[0]
            f.write(f"  {method:<12} {eff:.1f}%\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("  COMPARISON METHOD 4: GFLOPS\n")
        f.write("-" * 70 + "\n")
        f.write("  GFLOPS = 2 x N^3 / Time / 10^9\n")
        f.write("  Higher GFLOPS = Better computational throughput\n\n")
        f.write(f"  {'Method':<12} {'GFLOPS':<12}\n")
        f.write("  " + "-" * 24 + "\n")
        
        for method in methods:
            gflops = subset[(subset['Method'] == method) & (subset['Threads'] == max_threads)]['GFLOPS'].values[0]
            f.write(f"  {method:<12} {gflops:.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        
    print("  ✓ plots/comparison_report.txt")

def main():
    print("=" * 60)
    print("  Matrix Multiplication - Performance Analysis")
    print("=" * 60)
    
    df = load_data()
    if df is None:
        return
    
    create_output_directory()
    
    print("Generating comparison plots...")
    
    plot_time_comparison(df)
    plot_speedup_analysis(df)
    plot_efficiency_analysis(df)
    plot_gflops_comparison(df)
    plot_scalability_analysis(df)
    plot_summary_dashboard(df)
    generate_text_report(df)
    
    print("\n" + "=" * 60)
    print("  All plots generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
