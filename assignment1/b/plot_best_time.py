import pandas as pd

# Load CSVs (DO NOT MODIFY)
df_base = pd.read_csv("results.csv")
df_opt  = pd.read_csv("results_optimized.csv")

# Add label to distinguish source
df_base["version"] = "unoptimized"
df_opt["version"]  = "optimized"

# Combine
df = pd.concat([df_base, df_opt], ignore_index=True)

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

# Only consider required N values
Ns = [256, 512, 1024, 2048]
df = df[df["N"].isin(Ns)]

# ---- CORRECT BEST-TIME EXTRACTION ----
rows = []
for (N, pattern), g in df.groupby(["N", "pattern_name"]):
    best_idx = g["sec"].idxmin()
    rows.append({
        "N": N,
        "pattern_name": pattern,
        "best_time_sec": g.loc[best_idx, "sec"],
        "best_version": g.loc[best_idx, "version"]
    })

best_times = pd.DataFrame(rows).sort_values(["N", "best_time_sec"])

# Print nicely
print("\nBest time observed per access pattern (lower is better):\n")
print(best_times.to_string(index=False))

# Save for report
best_times.to_csv("best_times_summary.csv", index=False)
