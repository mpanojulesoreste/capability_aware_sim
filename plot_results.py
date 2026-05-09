"""
plot_results.py — generate summary.png and sweep_chart.png from sweep_results.csv.

Usage
-----
  python plot_results.py sweep_results.csv
  python plot_results.py sweep_results.csv --summary-only
  python plot_results.py sweep_results.csv --sweep-only
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_BASELINE   = "#8B9DAE"   # slate
COLOR_ADAPTIVE   = "#2C7A7B"   # teal
COLOR_STIGMERGIC = "#C8956A"   # amber

ALLOCATOR_COLORS = {
    "baseline":   COLOR_BASELINE,
    "adaptive":   COLOR_ADAPTIVE,
    "stigmergic": COLOR_STIGMERGIC,
}
ALLOCATOR_LABELS = {
    "baseline":   "Baseline",
    "adaptive":   "Adaptive",
    "stigmergic": "Stigmergic",
}


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("reach_right", "reach_left", "seed", "total_time",
                        "repositions", "unreachable", "deliveries",
                        "strong_side_fraction", "mean_handoff_excess",
                        "total_robot_distance", "h_idle", "r_idle", "c_act", "f_del"):
                try:
                    row[key] = float(row[key])
                except (ValueError, KeyError):
                    pass
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def _agg(rows, allocator, profile=None):
    """Return dict of metric → list-of-values for one allocator (optionally filtered to one profile)."""
    subset = [r for r in rows if r["allocator"] == allocator]
    if profile:
        subset = [r for r in subset if r["profile_name"] == profile]
    out = defaultdict(list)
    for r in subset:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                out[k].append(v)
    return out



def plot_summary_three(rows, profile="baseline_test",
                       save_path="summary.png"):
    """
    Three-bar chart (Baseline | Adaptive | Stigmergic) for the baseline_test
    profile.  Stigmergic bars show mean across seeds with min/max error bars.
    """
    allocators = ["baseline", "adaptive", "stigmergic"]

    metrics_def = [
        ("total_time",   "Total time (s)",        "↓ lower is better",  "{:.1f}"),
        ("repositions",  "User repositions",       "↓ lower is better",  "{:.0f}"),
        ("unreachable",  "Unreachable handoffs",   "↓ lower is better",  "{:.0f}"),
        ("deliveries",   "Successful deliveries",  "↑ higher is better", "{:.0f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=150)
    fig.suptitle(
        f"Three-allocator comparison  ·  profile: {profile}\n"
        "5 tasks  ·  3 robots  ·  asymmetric reach",
        fontsize=11, y=0.99,
    )

    for ax, (metric, title, direction, fmt) in zip(axes.flat, metrics_def):
        means, mins, maxs = [], [], []
        for alloc in allocators:
            vals = _agg(rows, alloc, profile=profile)[metric]
            means.append(float(np.mean(vals)))
            mins.append(float(np.min(vals)))
            maxs.append(float(np.max(vals)))

        x      = np.arange(len(allocators))
        colors = [ALLOCATOR_COLORS[a] for a in allocators]
        labels = [ALLOCATOR_LABELS[a] for a in allocators]

        bars = ax.bar(x, means, color=colors, width=0.55)

        for i, alloc in enumerate(allocators):
            if alloc == "stigmergic":
                ax.errorbar(x[i], means[i],
                            yerr=[[means[i] - mins[i]], [maxs[i] - means[i]]],
                            fmt="none", color="#555", capsize=4, linewidth=1.2)

        for i, (bar, val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{title}\n{direction}", fontsize=10, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCC")
        ax.spines["bottom"].set_color("#CCC")
        ax.tick_params(colors="#555", labelsize=9)
        ax.yaxis.grid(True, color="#EEE", linewidth=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Summary chart → {save_path}")
    plt.close()



def plot_sweep_chart(rows, save_path="sweep_chart.png"):
    """
    2×2 subplot line chart.  X-axis: reach asymmetry (reach_right − reach_left).
    One line per allocator.  Each point is the mean across seeds for that profile.
    """
    metrics_def = [
        ("total_time",           "Total time (s)",          "↓",  None,   None),
        ("repositions",          "User repositions",        "↓",  0.03,  -0.10),
        ("strong_side_fraction", "Strong-side fraction",    "↑",  None,   None),
        ("mean_handoff_excess",  "Mean handoff excess (m)", "↓",  0.003, -0.01),
    ]

    ALLOC_STYLE = {
        "baseline":   dict(linestyle="-",  marker="o", label="Baseline (solid)"),
        "adaptive":   dict(linestyle="--", marker="s", label="Adaptive (dashed)"),
        "stigmergic": dict(linestyle=":",  marker="^", label="Stigmergic (dotted)"),
    }
    ZERO_OFFSET = {"baseline": 0, "adaptive": 1, "stigmergic": 2}

    profile_asym = {}
    for r in rows:
        asym = r["reach_right"] - r["reach_left"]
        profile_asym[r["profile_name"]] = asym

    profiles_sorted = sorted(profile_asym.keys(), key=lambda p: profile_asym[p])
    x_vals = [profile_asym[p] for p in profiles_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    fig.suptitle(
        "Capability-profile sweep  ·  allocator comparison across reach asymmetry\n"
        "(asymmetry = reach_right − reach_left;  each point = mean over 10 seeds)",
        fontsize=11, y=0.99,
    )

    for ax, (metric, title, direction, eps, ylim_bottom) in zip(axes.flat, metrics_def):
        zero_panel = eps is not None

        for alloc in ["baseline", "adaptive", "stigmergic"]:
            y_means, y_mins, y_maxs = [], [], []
            for profile in profiles_sorted:
                vals = _agg(rows, alloc, profile=profile)[metric]
                if vals:
                    y_means.append(float(np.mean(vals)))
                    y_mins.append(float(np.min(vals)))
                    y_maxs.append(float(np.max(vals)))
                else:
                    y_means.append(float("nan"))
                    y_mins.append(float("nan"))
                    y_maxs.append(float("nan"))

            plot_y = y_means
            if zero_panel and all(v == 0.0 for v in y_means if not np.isnan(v)):
                offset = -eps * ZERO_OFFSET[alloc]
                plot_y = [v + offset if not np.isnan(v) else v for v in y_means]

            color  = ALLOCATOR_COLORS[alloc]
            style  = ALLOC_STYLE[alloc]
            ax.plot(x_vals, plot_y,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    color=color,
                    label=style["label"],
                    linewidth=2.0,
                    markersize=7)
            if alloc == "stigmergic":
                ax.fill_between(x_vals, y_mins, y_maxs, color=color, alpha=0.15)

        if zero_panel:
            ax.set_ylim(bottom=ylim_bottom)
            ax.axhline(0, color="#999", linewidth=0.8, linestyle="--", zorder=1)
            ax.annotate(
                "Adaptive & Stigmergic = 0 across all profiles",
                xy=(0.55, 0.12), xycoords="axes fraction",
                fontsize=9, color="#555555", style="italic",
            )

        ax.set_xlabel("Reach asymmetry  (right − left, m)", fontsize=9)
        ax.set_title(f"{title}  {direction}", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCC")
        ax.spines["bottom"].set_color("#CCC")
        ax.tick_params(colors="#555", labelsize=8)
        ax.yaxis.grid(True, color="#EEE", linewidth=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Sweep chart → {save_path}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("csv_path", help="Path to sweep_results.csv")
    parser.add_argument("--summary-out",   default="summary.png")
    parser.add_argument("--sweep-out",     default="sweep_chart.png")
    parser.add_argument("--summary-only",  action="store_true")
    parser.add_argument("--sweep-only",    action="store_true")
    parser.add_argument("--profile",       default="baseline_test",
                        help="Profile to use for the summary bar chart")
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    print(f"Loaded {len(rows)} rows from {args.csv_path}")

    if not args.sweep_only:
        plot_summary_three(rows, profile=args.profile, save_path=args.summary_out)

    if not args.summary_only:
        plot_sweep_chart(rows, save_path=args.sweep_out)


if __name__ == "__main__":
    main()
