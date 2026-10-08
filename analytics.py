"""车流量统计与测速可视化模块。"""

from __future__ import annotations

import csv
import os
from collections import defaultdict, deque
from os import makedirs, path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# 服务器和沙箱环境通常没有图形界面，Agg 后端可直接把统计图写成文件。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from speed_estimator import SpeedEstimator
from visualization import plot_summary

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

Ltrb = Tuple[int, int, int, int]
Track = Tuple[int, Ltrb]
Point = Tuple[float, float]
CLASS_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class TrajectoryAnalyzer:
    """目标轨迹记录与基础摘要。"""

    def __init__(self, maxlen: int = 50) -> None:
        self.maxlen = max(2, int(maxlen))
        # 只保留最近 maxlen 个中心点用于轨迹显示，累计长度单独保存，避免内存持续增长。
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.maxlen))
        self.total_length: Dict[int, float] = defaultdict(float)
        self.first_point: Dict[int, Point] = {}
        self.last_point: Dict[int, Point] = {}
        self.last_class_name: Dict[int, str] = {}
        self.last_timestamp: Dict[int, float] = {}

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _direction(start: Point, end: Point) -> str:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if abs(dx) >= abs(dy):
            return "right" if dx >= 0 else "left"
        return "down" if dy >= 0 else "up"

    def update(
        self,
        track_id: int,
        class_name: str,
        center: Point,
        timestamp: float,
    ) -> None:
        tid = int(track_id)
        center = (float(center[0]), float(center[1]))
        hist = self.track_history[tid]
        if tid not in self.first_point:
            self.first_point[tid] = center
        if hist:
            self.total_length[tid] += self._dist(hist[-1], center)
        hist.append(center)
        self.last_point[tid] = center
        self.last_class_name[tid] = str(class_name)
        self.last_timestamp[tid] = float(timestamp)

    def summary_rows(self) -> List[Dict[str, object]]:
        """生成结果页和 JSON 导出使用的轨迹摘要。"""
        rows = []
        for tid in sorted(self.last_point.keys()):
            start = self.first_point.get(tid, self.last_point[tid])
            end = self.last_point[tid]
            rows.append(
                {
                    "track_id": int(tid),
                    "class_name": self.last_class_name.get(tid, "unknown"),
                    "total_length": float(self.total_length.get(tid, 0.0)),
                    "direction": self._direction(start, end),
                    "last_timestamp": float(self.last_timestamp.get(tid, 0.0)),
                }
            )
        return rows


class AnalyticsEngine:
    """车流统计、双线测速和统计行缓存。

    main.py 每处理一帧都会调用 update()。本类不直接处理视频帧，只接收
    tracker 输出的目标框、类别和 track_id，再维护 CSV 所需的统计数据。
    """

    VEHICLE_CLS = {1, 2, 3, 5, 7}

    def __init__(
        self,
        fps: float,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int],
        count_line: Tuple[int, int, int, int] = None,
        line_distance_m: float = 10.0,
        speed_mode: str = "double_line",
        speed_direction_mode: str = "vertical",
        cross_tolerance: int = 8,
        pixel_to_meter: float = 0.05,
        min_speed_time_s: float = 0.25,
        max_speed_kmh: float = 200.0,
    ) -> None:
        self.fps = float(max(1e-6, fps))
        self.line1 = line1
        self.line2 = line2
        self.count_line = count_line or line1
        self.speed_mode = speed_mode
        self.speed_direction_mode = speed_direction_mode

        # SpeedEstimator 负责逐 track_id 的穿线状态，本类负责把结果组织成网页和 CSV 需要的形式。
        self.estimator = SpeedEstimator(
            line1=line1,
            line2=line2,
            count_line=self.count_line,
            line_distance_m=line_distance_m,
            direction_mode=speed_direction_mode,
            cross_tolerance=cross_tolerance,
            speed_mode=speed_mode,
            pixel_to_meter=pixel_to_meter,
            min_speed_time_s=min_speed_time_s,
            max_speed_kmh=max_speed_kmh,
        )

        self.count_rows: List[Dict[str, object]] = []
        self.event_rows: List[Dict[str, object]] = self.estimator.event_rows
        self.traffic_rows: List[Dict[str, object]] = self.estimator.traffic_rows

    @staticmethod
    def center_of_ltrb(bbox: Ltrb) -> Point:
        l, t, r, b = bbox
        return (0.5 * (l + r), 0.5 * (t + b))

    @property
    def track_speed_kmh(self) -> Dict[int, float]:
        """返回已测速目标的速度映射，供画框时叠加速度文本。"""
        out = {}
        for tid, state in self.estimator.speed_state.items():
            v = state.get("speed_kmh")
            if v is not None:
                out[int(tid)] = float(v)
        return out

    @property
    def line_counts(self) -> Dict[str, int]:
        """按当前方向模式返回可读的车流方向计数。"""
        if self.speed_direction_mode == "vertical":
            return {
                "up": int(self.estimator.flow_counts["down_to_up"]),
                "down": int(self.estimator.flow_counts["up_to_down"]),
            }
        return {
            "left": int(self.estimator.flow_counts["right_to_left"]),
            "right": int(self.estimator.flow_counts["left_to_right"]),
        }

    def speed_summary(self) -> Tuple[int, float, float]:
        return self.estimator.measured_stats()

    def update(
        self,
        frame_id: int,
        timestamp: float,
        detections,
        tracks: Sequence[Track],
        track_class_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """输入一帧检测与跟踪结果并更新累计统计。"""
        vehicle_count = 0
        if detections is not None and getattr(detections, "size", 0) > 0:
            # detections 来自 YOLO 原始检测结果，用于记录当前帧车辆目标数量。
            for row in detections:
                cls_id = int(row[5])
                if cls_id in self.VEHICLE_CLS:
                    vehicle_count += 1

        cls_map = track_class_map or {}
        for track_id, bbox in tracks:
            # tracks 来自 ByteTrack，track_id 是计数、测速和轨迹输出的稳定主键。
            tid = int(track_id)
            cls_id = cls_map.get(tid)
            center = self.center_of_ltrb(bbox)
            self.estimator.update_track(tid, center, float(timestamp), cls_id)

        measured_n, avg_speed, max_speed = self.speed_summary()
        if self.speed_direction_mode == "vertical":
            flow_up = int(self.line_counts["up"])
            flow_down = int(self.line_counts["down"])
        else:
            flow_up = int(self.line_counts["left"])
            flow_down = int(self.line_counts["right"])

        self.count_rows.append(
            {
                "frame_id": int(frame_id),
                "timestamp": float(timestamp),
                "vehicle_count": int(vehicle_count),
                "flow_total": int(flow_up + flow_down),
                "dir_a_count": int(flow_up),
                "dir_b_count": int(flow_down),
                "measured_count": int(measured_n),
                "avg_speed_kmh": float(avg_speed),
                "max_speed_kmh": float(max_speed),
            }
        )

    def export_counts_csv(self, save_csv: str) -> None:
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_id",
                    "timestamp",
                    "vehicle_count",
                    "flow_total",
                    "dir_a_count",
                    "dir_b_count",
                    "measured_count",
                    "avg_speed_kmh",
                    "max_speed_kmh",
                ],
            )
            writer.writeheader()
            for row in self.count_rows:
                writer.writerow(row)

    def export_events_csv(self, save_csv: str) -> None:
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["track_id", "event_type", "direction", "speed_kmh", "timestamp"],
            )
            writer.writeheader()
            for row in self.event_rows:
                writer.writerow(row)

    def export_traffic_csv(self, save_csv: str) -> None:
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "track_id",
                    "class_name",
                    "direction",
                    "speed_kmh",
                    "cross_line1_time",
                    "cross_line2_time",
                ],
            )
            writer.writeheader()
            for row in self.traffic_rows:
                writer.writerow(row)

    def export_plots(self, out_dir: str) -> None:
        makedirs(out_dir, exist_ok=True)
        measured_n, avg_speed, max_speed = self.speed_summary()
        plot_summary(
            count_rows=self.count_rows,
            flow_counts=self.estimator.flow_counts,
            speed_direction_mode=self.speed_direction_mode,
            measured_n=measured_n,
            avg_speed=avg_speed,
            max_speed=max_speed,
            save_png=path.join(out_dir, "综合统计图.png"),
        )
