# save as: make_slide_artifacts.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_COLS = ["num_groups", "num_players", "num_weeks"]
REQ_COLS = set(KEY_COLS + ["initial_violations", "final_violations", "improvement", "time_seconds"])


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    # enforce deterministic order for reproducibility
    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    # detect duplicates 
    dup = df.duplicated(KEY_COLS, keep=False)
    if dup.any():
        d = df.loc[dup, KEY_COLS].value_counts().head(10)
        raise ValueError(
            "Duplicate instance keys found (same num_groups/num_players/num_weeks appears multiple times). "
            "Fix upstream or deduplicate explicitly.\n"
            f"Examples:\n{d.to_string()}"
        )
    return df


def summary_dict(df: pd.DataFrame) -> dict:
    perfect_n = int((df["final_violations"] == 0).sum())
    n = int(len(df))
    return {
        "n": n,
        "perfect_n": perfect_n,
        "perfect_rate": float(perfect_n / n) if n > 0 else float("nan"),
        "avg_improvement": float(df["improvement"].mean()) if n > 0 else float("nan"),
        "avg_final_violations": float(df["final_violations"].mean()) if n > 0 else float("nan"),
        "avg_runtime_s": float(df["time_seconds"].mean()) if n > 0 else float("nan"),
    }


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def write_metrics(outdir: Path, metrics: dict) -> None:
    (outdir / "slide_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # quick copy/paste format for slides
    lines = []
    lines.append("=== OVERALL ===")
    o = metrics["overall"]
    lines.append(f'MAB: n={o["mab"]["n"]}, perfect={o["mab"]["perfect_n"]} ({o["mab"]["perfect_rate"]:.3f}), '
                 f'avg_impr={o["mab"]["avg_improvement"]:.3f}, avg_final={o["mab"]["avg_final_violations"]:.3f}, '
                 f'avg_time_s={o["mab"]["avg_runtime_s"]:.3f}')
    lines.append(f'RL : n={o["rl"]["n"]}, perfect={o["rl"]["perfect_n"]} ({o["rl"]["perfect_rate"]:.3f}), '
                 f'avg_impr={o["rl"]["avg_improvement"]:.3f}, avg_final={o["rl"]["avg_final_violations"]:.3f}, '
                 f'avg_time_s={o["rl"]["avg_runtime_s"]:.3f}')

    lines.append("")
    lines.append("=== FAIR (COMMON INSTANCES) ===")
    c = metrics["common_set"]
    lines.append(f'MAB(common): n={c["mab"]["n"]}, perfect={c["mab"]["perfect_n"]} ({c["mab"]["perfect_rate"]:.3f}), '
                 f'avg_impr={c["mab"]["avg_improvement"]:.3f}, avg_final={c["mab"]["avg_final_violations"]:.3f}, '
                 f'avg_time_s={c["mab"]["avg_runtime_s"]:.3f}')
    lines.append(f'RL(common) : n={c["rl"]["n"]}, perfect={c["rl"]["perfect_n"]} ({c["rl"]["perfect_rate"]:.3f}), '
                 f'avg_impr={c["rl"]["avg_improvement"]:.3f}, avg_final={c["rl"]["avg_final_violations"]:.3f}, '
                 f'avg_time_s={c["rl"]["avg_runtime_s"]:.3f}')

    lines.append("")
    lines.append("=== HEAD-TO-HEAD (COMMON) ===")
    h = metrics["head_to_head"]
    lines.append(f'MAB wins: {h["mab_wins"]}')
    lines.append(f'RL  wins: {h["rl_wins"]}')
    lines.append(f'Ties    : {h["ties"]}')
    lines.append(f'Mean gap (RL - MAB): {h["gap_mean"]:.3f}')
    lines.append(f'Median gap (RL - MAB): {h["gap_median"]:.3f}')

    (outdir / "slide_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="task1_1_results.csv", help="Task 1.1 results CSV (MAB)")
    ap.add_argument("--rl", default="task1_2_results.csv", help="Task 1.2 results CSV (RL)")
    ap.add_argument("--out", default="plots", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    mab = load_results(args.mab)
    rl = load_results(args.rl)

    # common key set (fair comparison)
    mab_common = pd.merge(mab, rl[KEY_COLS], on=KEY_COLS, how="inner")
    rl_common = pd.merge(rl, mab[KEY_COLS], on=KEY_COLS, how="inner")

    if len(mab_common) == 0 or len(rl_common) == 0:
        raise RuntimeError("No overlapping instances between MAB and RL on KEY_COLS.")

    # per-instance join for head-to-head and gap distribution
    paired = pd.merge(
        rl_common,
        mab_common,
        on=KEY_COLS,
        how="inner",
        suffixes=("_rl", "_mab"),
    )

    # head-to-head counts (quality only)
    mab_wins = int((paired["final_violations_mab"] < paired["final_violations_rl"]).sum())
    rl_wins = int((paired["final_violations_rl"] < paired["final_violations_mab"]).sum())
    ties = int((paired["final_violations_rl"] == paired["final_violations_mab"]).sum())

    gap = (paired["final_violations_rl"] - paired["final_violations_mab"]).astype(float)

    metrics = {
        "overall": {
            "mab": summary_dict(mab),
            "rl": summary_dict(rl),
        },
        "common_set": {
            "mab": summary_dict(mab_common),
            "rl": summary_dict(rl_common),
        },
        "head_to_head": {
            "mab_wins": mab_wins,
            "rl_wins": rl_wins,
            "ties": ties,
            "gap_mean": float(gap.mean()),
            "gap_median": float(gap.median()),
        },
    }

    # export per-instance table for appendices / debugging
    cols = KEY_COLS + [
        "initial_violations_mab", "final_violations_mab", "improvement_mab", "time_seconds_mab",
        "initial_violations_rl", "final_violations_rl", "improvement_rl", "time_seconds_rl",
    ]
    paired[cols].to_csv(outdir / "head_to_head_common.csv", index=False)

    # Plot 1: Fair comparison summary (bars)
    summ_mab = metrics["common_set"]["mab"]
    summ_rl = metrics["common_set"]["rl"]
    plot_metrics = ["avg_improvement", "avg_final_violations", "avg_runtime_s", "perfect_rate"]
    labels = ["Avg improvement", "Avg final violations", "Avg runtime (s)", "Perfect rate"]

    x = np.arange(len(plot_metrics))
    width = 0.38

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, [summ_mab[m] for m in plot_metrics], width, label=f"MAB (n={summ_mab['n']})")
    plt.bar(x + width / 2, [summ_rl[m] for m in plot_metrics], width, label=f"RL (n={summ_rl['n']})")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("Fair comparison on the same instances")
    plt.legend()
    savefig(outdir / "fair_comparison_summary.png")

    # Plot 2: Runtime distribution (boxplot)
    plt.figure(figsize=(8.5, 4.5))
    try:
        plt.boxplot([mab_common["time_seconds"], rl_common["time_seconds"]],
                    tick_labels=["MAB", "RL"],
                    showfliers=True)
    except TypeError:
        plt.boxplot([mab_common["time_seconds"], rl_common["time_seconds"]],
                    labels=["MAB", "RL"],
                    showfliers=True)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime distribution (same instances)")
    savefig(outdir / "runtime_boxplot.png")

    # Plot 3: Quality vs runtime trade-off (scatter, log x)
    plt.figure(figsize=(8.5, 5.0))
    plt.scatter(mab_common["time_seconds"], mab_common["final_violations"], label="MAB", alpha=0.8)
    plt.scatter(rl_common["time_seconds"], rl_common["final_violations"], label="RL", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Runtime (s, log scale)")
    plt.ylabel("Final violations")
    plt.title("Quality vs runtime trade-off (same instances)")
    plt.legend()
    savefig(outdir / "quality_vs_runtime_scatter.png")

    # Plot 4: Per-instance gap histogram
    plt.figure(figsize=(8.2, 4.6))
    plt.hist(gap, bins=15)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Final violations difference (RL − MAB)")
    plt.ylabel("Count")
    plt.title("Per-instance quality gap (same instances)")
    savefig(outdir / "per_instance_gap_hist.png")

    # export metrics for slides
    write_metrics(outdir, metrics)

    print(f"\nSaved artifacts to: {outdir.resolve()}")
    for p in sorted(outdir.glob("*")):
        if p.is_file():
            print(" -", p.name)


if __name__ == "__main__":
    main()
