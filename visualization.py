"""可视化模块：输出中文综合统计图。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_summary(
    count_rows: List[Dict[str, object]],
    flow_counts: Dict[str, int],
    speed_direction_mode: str,
    measured_n: int,
    avg_speed: float,
    max_speed: float,
    save_png: str,
) -> None:
    sec_buckets: Dict[int, Dict[str, float]] = defaultdict(lambda: {"vehicle_sum": 0.0, "n": 0.0})
    for row in count_rows:
        sec = int(float(row["timestamp"]))
        sec_buckets[sec]["vehicle_sum"] += float(row["vehicle_count"])
        sec_buckets[sec]["n"] += 1.0

    secs = sorted(sec_buckets.keys())
    vehicle_series = []
    for s in secs:
        n = max(1.0, sec_buckets[s]["n"])
        vehicle_series.append(sec_buckets[s]["vehicle_sum"] / n)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("监控数据综合统计图", fontsize=14)

    ax1 = axes[0]
    if secs:
        ax1.plot(secs, vehicle_series, label="目标数量")
        ax1.set_title("目标数量变化")
        ax1.set_xlabel("时间（秒）")
        ax1.set_ylabel("数量")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "无目标数据", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("目标数量变化")
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax2 = axes[1]
    if speed_direction_mode == "vertical":
        labels = ["上行（下->上）", "下行（上->下）"]
        vals = [flow_counts.get("down_to_up", 0), flow_counts.get("up_to_down", 0)]
    else:
        labels = ["左行（右->左）", "右行（左->右）"]
        vals = [flow_counts.get("right_to_left", 0), flow_counts.get("left_to_right", 0)]
    ax2.bar(labels, vals)
    ax2.set_title("车流方向统计")
    ax2.set_ylabel("车辆数")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    if max(vals) == 0:
        ax2.text(0.5, 0.5, "无方向统计数据", ha="center", va="center", transform=ax2.transAxes)

    ax3 = axes[2]
    speed_labels = ["平均车速", "最高车速"]
    speed_vals = [avg_speed, max_speed]
    ax3.bar(speed_labels, speed_vals)
    ax3.set_title("车速统计")
    ax3.set_ylabel("km/h")
    if measured_n <= 0:
        ax3.text(0.5, 0.5, "无测速数据", ha="center", va="center", transform=ax3.transAxes)

    for ax in axes:
        ax.grid(False)

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(save_png, dpi=200)
    plt.close(fig)
