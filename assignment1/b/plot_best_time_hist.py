import os
import pandas as pd
import matplotlib.pyplot as plt

# Load summary (generated earlier)
df = pd.read_csv("best_times_summary.csv")

assets_dir = "./assets/best_time_plots"
os.makedirs(assets_dir, exist_ok=True)

Ns = sorted(df["N"].unique())

# Color mapping for version
color_map = {
    "optimized": "tab:blue",
    "unoptimized": "tab:orange"
}

for N in Ns:
    sub = df[df["N"] == N].sort_values("best_time_sec")

    plt.figure(figsize=(9, 5))

    colors = [color_map[v] for v in sub["best_version"]]

    plt.bar(
        sub["pattern_name"],
        sub["best_time_sec"],
        color=colors,
        edgecolor="black"
    )

    plt.ylabel("Best Execution Time (seconds)")
    plt.xlabel("Access Pattern")
    plt.title(f"Best Observed Time per Access Pattern (N = {N})")

    plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="tab:blue", edgecolor="black", label="Optimized"),
        Patch(facecolor="tab:orange", edgecolor="black", label="Unoptimized")
    ]
    plt.legend(handles=legend_elems)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = f"{assets_dir}/best_time_N{N}.png"
    plt.savefig(out)
    plt.close()

print("Best-time plots saved in ./assets/best_time_plots/")
