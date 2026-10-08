"""车流统计与双线测速状态机。

说明：
1) 统计线 count_line 用于车流量计数。
2) speed_line_1 / speed_line_2 用于估算速度。
3) 默认采用双线测速（double_line），更稳定，适合固定摄像头交通场景。
4) pixel_estimate 为备用近似方案，依赖像素标定，误差较大。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

from config import MAX_SPEED_KMH, MIN_SPEED_TIME_S

Point = Tuple[float, float]
Line = Tuple[int, int, int, int]

CLS_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLS = {1, 2, 3, 5, 7}


class SpeedEstimator:
    """按 track_id 维护车流计数和双线测速状态。"""

    def __init__(
        self,
        line1: Line,
        line2: Line,
        count_line: Line = None,
        line_distance_m: float = 10.0,
        direction_mode: str = "vertical",
        cross_tolerance: int = 8,
        speed_mode: str = "double_line",
        pixel_to_meter: float = 0.05,
        min_speed_time_s: float = MIN_SPEED_TIME_S,
        max_speed_kmh: float = MAX_SPEED_KMH,
    ) -> None:
        self.line1 = line1
        self.line2 = line2
        self.count_line = count_line or line1
        self.line_distance_m = float(max(1e-6, line_distance_m))
        self.direction_mode = direction_mode if direction_mode in {"vertical", "horizontal"} else "vertical"
        self.cross_tolerance = int(max(1, cross_tolerance))
        self.speed_mode = speed_mode if speed_mode in {"double_line", "pixel_estimate"} else "double_line"
        self.pixel_to_meter = float(max(1e-6, pixel_to_meter))
        self.min_speed_time_s = float(max(1e-6, min_speed_time_s))
        self.max_speed_kmh = float(max(1.0, max_speed_kmh))

        self.speed_state: Dict[int, Dict[str, object]] = defaultdict(
            lambda: {
                "cross_line1_time": None,
                "cross_line2_time": None,
                "cross_line1_point": None,
                "cross_line2_point": None,
                "last_coord": None,
                "speed_kmh": None,
                "direction": None,
                "class_id": None,
                "done": False,
                "flow_counted": False,
            }
        )
        self.traffic_rows = []
        self.event_rows = []
        self.flow_counts = {
            "up_to_down": 0,
            "down_to_up": 0,
            "left_to_right": 0,
            "right_to_left": 0,
        }

    def _coord(self, pt: Point) -> float:
        return float(pt[1]) if self.direction_mode == "vertical" else float(pt[0])

    def _line_coord(self, line: Line) -> float:
        if self.direction_mode == "vertical":
            return 0.5 * (float(line[1]) + float(line[3]))
        return 0.5 * (float(line[0]) + float(line[2]))

    @staticmethod
    def _dist(p1: Point, p2: Point) -> float:
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _crossed_line(self, prev_c: Optional[float], cur_c: float, line_c: float) -> bool:
        """判断目标中心点是否穿过一条虚拟线。

        当前算法按 vertical/horizontal 方向使用一维坐标判断，因此适合
        近似水平或近似竖直的统计线/测速线；倾斜很大的线会产生近似误差。
        """
        if prev_c is None:
            return False
        prev_d = float(prev_c - line_c)
        cur_d = float(cur_c - line_c)
        if abs(cur_d) <= self.cross_tolerance and abs(prev_d) > self.cross_tolerance:
            return True
        return prev_d * cur_d < 0.0

    def _direction_label(self, from_c: float, to_c: float) -> str:
        if self.direction_mode == "vertical":
            return "up_to_down" if to_c >= from_c else "down_to_up"
        return "left_to_right" if to_c >= from_c else "right_to_left"

    def update_track(
        self,
        track_id: int,
        center: Point,
        timestamp: float,
        cls_id: Optional[int],
    ) -> None:
        """更新单个车辆目标的计数和测速状态。"""
        tid = int(track_id)
        if cls_id is None or int(cls_id) not in VEHICLE_CLS:
            return

        state = self.speed_state[tid]
        state["class_id"] = int(cls_id)

        cur_c = self._coord(center)
        prev_c = state["last_coord"]
        line1_c = self._line_coord(self.line1)
        line2_c = self._line_coord(self.line2)
        count_line_c = self._line_coord(self.count_line)

        crossed_l1 = self._crossed_line(prev_c, cur_c, line1_c)
        crossed_l2 = self._crossed_line(prev_c, cur_c, line2_c)
        crossed_count = self._crossed_line(prev_c, cur_c, count_line_c)

        if crossed_count and not bool(state["flow_counted"]):
            # 车流统计只依赖统计线，每个 track_id 只累计一次，避免反复抖动重复计数。
            if prev_c is not None:
                flow_dir = self._direction_label(float(prev_c), cur_c)
                self.flow_counts[flow_dir] += 1
            state["flow_counted"] = True

        if crossed_l1 and state["cross_line1_time"] is None:
            # 记录穿过测速线1的时刻；不要求它一定先于测速线2发生。
            state["cross_line1_time"] = float(timestamp)
            state["cross_line1_point"] = center

        if crossed_l2 and state["cross_line2_time"] is None:
            # 记录穿过测速线2的时刻；两条线都穿过后即可估算速度。
            state["cross_line2_time"] = float(timestamp)
            state["cross_line2_point"] = center

        if (
            state["cross_line1_time"] is not None
            and state["cross_line2_time"] is not None
            and not bool(state["done"])
        ):
            t1 = float(state["cross_line1_time"])
            t2 = float(state["cross_line2_time"])
            dt = abs(t2 - t1)
            # 过线时间过短通常来自抖动误触发，直接过滤。
            if dt < self.min_speed_time_s:
                state["last_coord"] = cur_c
                state["done"] = True
                return
            p1 = state["cross_line1_point"]
            p2 = state["cross_line2_point"]

            if self.speed_mode == "double_line":
                speed_kmh = self.line_distance_m / dt * 3.6
            else:
                # 备用近似方案：依赖像素标定，误差较大。
                dist_px = self._dist(p1, p2) if isinstance(p1, tuple) and isinstance(p2, tuple) else 0.0
                speed_kmh = (dist_px * self.pixel_to_meter / dt) * 3.6

            speed_kmh = float(max(0.0, speed_kmh))
            if speed_kmh > self.max_speed_kmh:
                state["last_coord"] = cur_c
                state["done"] = True
                return
            state["speed_kmh"] = speed_kmh

            first_point, second_point = (p1, p2) if t1 <= t2 else (p2, p1)
            from_c = self._coord(first_point) if isinstance(first_point, tuple) else cur_c
            to_c = self._coord(second_point) if isinstance(second_point, tuple) else cur_c
            direction = self._direction_label(from_c, to_c)
            state["direction"] = direction
            state["done"] = True

            class_name = CLS_NAME.get(int(state["class_id"]), str(state["class_id"]))
            self.traffic_rows.append(
                {
                    "track_id": tid,
                    "class_name": class_name,
                    "direction": direction,
                    "speed_kmh": speed_kmh,
                    "cross_line1_time": t1,
                    "cross_line2_time": t2,
                }
            )
            self.event_rows.append(
                {
                    "track_id": tid,
                    "event_type": "speed_measured",
                    "direction": direction,
                    "speed_kmh": speed_kmh,
                    "timestamp": max(t1, t2),
                }
            )

        state["last_coord"] = cur_c

    def measured_stats(self) -> Tuple[int, float, float]:
        if not self.traffic_rows:
            return 0, 0.0, 0.0
        speeds = [float(r["speed_kmh"]) for r in self.traffic_rows]
        return len(speeds), float(sum(speeds) / len(speeds)), float(max(speeds))
