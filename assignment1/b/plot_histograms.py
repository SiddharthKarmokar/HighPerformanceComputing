import os
import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs (DO NOT MODIFY THEM)
df_base = pd.read_csv("results.csv")
df_opt  = pd.read_csv("results_optimized.csv")

# Pattern ID → name
pattern_names = {
    0: "row_major_rows_per_thread",
    1: "col_major",
    2: "blocked_32",
    3: "linear_flat",
    4: "cyclic_rows",
    5: "unroll4"
}

df_base["pattern_name"] = df_base["pattern"].map(pattern_names)
df_opt["pattern_name"]  = df_opt["pattern"].map(pattern_names)

assets_dir = "./assets/histograms"
os.makedirs(assets_dir, exist_ok=True)

Ns = [256, 512, 1024, 2048]
patterns = sorted(pattern_names.values())

# --------- HISTOGRAMS ----------
for N in Ns:
    for pattern in patterns:

        base_vals = df_base[
            (df_base["N"] == N) &
            (df_base["pattern_name"] == pattern)
        ]["sec"]

        opt_vals = df_opt[
            (df_opt["N"] == N) &
            (df_opt["pattern_name"] == pattern)
        ]["sec"]

        # Skip if data missing
        if len(base_vals) == 0 or len(opt_vals) == 0:
            continue

        plt.figure(figsize=(7, 4))

        plt.hist(
            base_vals,
            bins=10,
            alpha=0.6,
            label="Unoptimized",
            edgecolor="black"
        )

        plt.hist(
            opt_vals,
            bins=10,
            alpha=0.6,
            label="Optimized",
            edgecolor="black"
        )

        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Frequency")
        plt.title(f"Histogram: N={N}, Pattern={pattern}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out = f"{assets_dir}/hist_N{N}_{pattern}.png"
        plt.savefig(out)
        plt.close()

print("Histogram plots saved in ./assets/histograms/")
