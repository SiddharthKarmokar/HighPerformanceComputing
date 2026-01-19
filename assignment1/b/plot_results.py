import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_optimized.csv")

# Pattern ID → name
pattern_names = {
    0: "row_major_rows_per_thread",
    1: "col_major",
    2: "blocked_32",
    3: "linear_flat",
    4: "cyclic_rows",
    5: "unroll4"
}
df["pattern_name"] = df["pattern"].map(pattern_names)

assets_dir = "./assets"
os.makedirs(assets_dir, exist_ok=True)

for threads in sorted(df["threads"].unique()):
    sub = df[df["threads"] == threads]

    plt.figure(figsize=(8, 5))

    for pattern, g in sub.groupby("pattern_name"):
        g = g.sort_values("N")
        plt.plot(g["N"], g["sec"], marker="o", label=pattern)

    plt.xlabel("Matrix size N (NxN)")
    plt.ylabel("Time (seconds)")
    plt.title(f"N vs Time (threads = {threads}) (Optimized)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize="small")
    plt.tight_layout()

    plt.savefig(f"{assets_dir}/N_vs_Time_threads_{threads}_optimized.png")
    plt.close()

print("Plots saved in ./assets/")
