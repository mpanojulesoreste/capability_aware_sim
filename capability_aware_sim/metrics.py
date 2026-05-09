"""Event logger, summary statistics, and matplotlib summary chart."""

from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt

COLOR_BASELINE = "#8B9DAE"
COLOR_ADAPTIVE = "#2C7A7B"


def plot_summary(baseline_metrics, adaptive_metrics,
                 save_path_png="summary.png", save_path_pdf="summary.pdf"):
    """
    Each *_metrics arg is a dict with keys:
        'total_time', 'repositions', 'unreachable', 'deliveries'
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)
    fig.suptitle(
        "Capability-blind vs. capability-aware allocation\n"
        "5 tasks  \u00b7  3 robots  \u00b7  asymmetric reach (R=0.8m, L=0.25m)",
        fontsize=11, y=0.98,
    )

    panels = [
        (axes[0, 0], "Total time (s)",        "\u2193 lower is better",
         baseline_metrics["total_time"],  adaptive_metrics["total_time"],  "{:.1f}", None),
        (axes[0, 1], "User repositions",      "\u2193 lower is better",
         baseline_metrics["repositions"], adaptive_metrics["repositions"], "{:d}",   3),
        (axes[1, 0], "Unreachable handoffs",  "\u2193 lower is better",
         baseline_metrics["unreachable"], adaptive_metrics["unreachable"], "{:d}",   3),
        (axes[1, 1], "Successful deliveries", "\u2191 higher is better",
         baseline_metrics["deliveries"],  adaptive_metrics["deliveries"],  "{:d}",   None),
    ]

    for ax, title, direction, b_val, a_val, fmt, ylim_floor in panels:
        bars = ax.bar(
            ["Baseline", "Adaptive"],
            [b_val, a_val],
            color=[COLOR_BASELINE, COLOR_ADAPTIVE],
            width=0.55,
        )

        ax.set_title(f"{title}\n{direction}", fontsize=10, pad=8)

        for bar, val in zip(bars, [b_val, a_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(val),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
            )

        if b_val != 0:
            delta_pct = ((a_val - b_val) / b_val) * 100
            delta_str = f"{delta_pct:+.0f}%"
        elif a_val == 0:
            delta_str = "0"
        else:
            delta_str = "+\u221e"

        ax.annotate(
            delta_str,
            xy=(1, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -22), textcoords="offset points",
            ha="center", fontsize=9, color="#555",
        )

        if ylim_floor is not None:
            current_top = ax.get_ylim()[1]
            ax.set_ylim(0, max(current_top, ylim_floor))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCC")
        ax.spines["bottom"].set_color("#CCC")
        ax.tick_params(colors="#555", labelsize=9)
        ax.yaxis.grid(True, color="#EEE", linewidth=0.7)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path_png, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")
    print(f"Chart saved -> {save_path_png}, {save_path_pdf}")
    plt.show()


@dataclass
class MetricsLog:
    label: str

    completion_times: List[float] = field(default_factory=list)
    reposition_count: int = 0
    unreachable_count: int = 0
    delivery_count:    int = 0
    _start_time:       float = 0.0

    def set_start(self, t: float):
        self._start_time = t

    def log_delivery(self, t: float, reachable: bool):
        self.delivery_count += 1
        self.completion_times.append(t - self._start_time)
        if not reachable:
            self.unreachable_count += 1

    def log_reposition(self):
        self.reposition_count += 1

    def total_time(self) -> float:
        return max(self.completion_times) if self.completion_times else 0.0

    def summary(self) -> dict:
        return {
            "label":            self.label,
            "total_time":       round(self.total_time(), 2),
            "repositions":      self.reposition_count,
            "unreachable":      self.unreachable_count,
            "deliveries":       self.delivery_count,
        }
