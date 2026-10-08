"""规则型异常风险提示模块。

本模块只使用现有检测跟踪结果，不训练新模型。输出结果定位为“疑似风险
提示”，用于补充系统功能闭环，不作为交通违法或事故的严格判定。
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


class AnomalyDetector:
    """基于 track_id、bbox 和中心点的轻量异常风险检测器。

    当前只保留 rear_end_risk：图像空间内前后车距离过近。
    """

    VEHICLE_CLS = {"bicycle", "car", "motorcycle", "bus", "truck"}
    FIELDS = [
        "frame_id",
        "timestamp",
        "event_type",
        "track_id",
        "related_track_id",
        "class_name",
        "description",
        "evidence_path",
        "evidence_paths",
        "qwen_image_paths",
        "evidence_offsets_sec",
        "evidence_times_sec",
        "qwen_result",
        "qwen_reason",
        "qwen_error",
    ]

    def __init__(
        self,
        rear_end_hold_frames: int = 3,
        rear_end_x_overlap_ratio: float = 0.45,
        rear_end_y_gap_ratio: float = 0.35,
        rear_end_confirm_frames: int = 8,
        rear_end_stop_window_frames: int = 5,
        rear_end_stop_move_px: float = 6.0,
        rear_end_context_distance_px: float = 220.0,
        rear_end_other_motion_px: float = 14.0,
        rear_end_other_lateral_px: float = 12.0,
    ) -> None:
        self.rear_end_hold_frames = max(1, int(rear_end_hold_frames))
        self.rear_end_x_overlap_ratio = float(max(0.0, rear_end_x_overlap_ratio))
        self.rear_end_y_gap_ratio = float(max(0.0, rear_end_y_gap_ratio))
        self.rear_end_confirm_frames = max(1, int(rear_end_confirm_frames))
        self.rear_end_stop_window_frames = max(2, int(rear_end_stop_window_frames))
        self.rear_end_stop_move_px = float(max(0.0, rear_end_stop_move_px))
        self.rear_end_context_distance_px = float(max(1.0, rear_end_context_distance_px))
        self.rear_end_other_motion_px = float(max(0.0, rear_end_other_motion_px))
        self.rear_end_other_lateral_px = float(max(0.0, rear_end_other_lateral_px))

        self.rear_end_hits: Dict[Tuple[int, int], int] = defaultdict(int)
        self.track_points: Dict[int, Deque[Point]] = defaultdict(
            lambda: deque(maxlen=max(self.rear_end_confirm_frames, self.rear_end_stop_window_frames) + 2)
        )
        self.rear_end_candidates: Dict[Tuple[int, int], Dict[str, object]] = {}
        self.reported = set()
        self.events: List[Dict[str, object]] = []

    @staticmethod
    def center_of_bbox(bbox: BBox) -> Point:
        l, t, r, b = bbox
        return 0.5 * (l + r), 0.5 * (t + b)

    @staticmethod
    def _x_overlap_ratio(a: BBox, b: BBox) -> float:
        al, _, ar, _ = a
        bl, _, br, _ = b
        overlap = max(0, min(ar, br) - max(al, bl))
        min_width = max(1, min(ar - al, br - bl))
        return float(overlap) / float(min_width)

    @staticmethod
    def _vertical_gap(a: BBox, b: BBox) -> float:
        _, at, _, ab = a
        _, bt, _, bb = b
        if ab < bt:
            return float(bt - ab)
        if bb < at:
            return float(at - bb)
        return 0.0

    @staticmethod
    def _bbox_height(bbox: BBox) -> float:
        return float(max(1, bbox[3] - bbox[1]))

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _recent_motion(self, track_id: int, window: int) -> Optional[float]:
        points = self.track_points.get(int(track_id))
        if points is None or len(points) < window:
            return None
        pts = list(points)[-window:]
        return self._distance(pts[0], pts[-1])

    def _recent_lateral_motion(self, track_id: int, window: int) -> Optional[float]:
        points = self.track_points.get(int(track_id))
        if points is None or len(points) < window:
            return None
        pts = list(points)[-window:]
        return abs(float(pts[-1][0] - pts[0][0]))

    def _pair_is_stopped(self, pair: Tuple[int, int]) -> bool:
        motions = [self._recent_motion(tid, self.rear_end_stop_window_frames) for tid in pair]
        return all(move is not None and move <= self.rear_end_stop_move_px for move in motions)

    def _has_moving_context(self, pair: Tuple[int, int], object_by_id: Dict[int, Dict[str, object]]) -> bool:
        pair_objects = [object_by_id.get(tid) for tid in pair]
        if any(obj is None for obj in pair_objects):
            return False
        pair_center = (
            0.5 * (float(pair_objects[0]["center"][0]) + float(pair_objects[1]["center"][0])),
            0.5 * (float(pair_objects[0]["center"][1]) + float(pair_objects[1]["center"][1])),
        )
        for tid, obj in object_by_id.items():
            if tid in pair:
                continue
            center = tuple(float(v) for v in obj["center"])
            if self._distance(pair_center, center) > self.rear_end_context_distance_px:
                continue
            motion = self._recent_motion(tid, self.rear_end_stop_window_frames)
            lateral = self._recent_lateral_motion(tid, self.rear_end_stop_window_frames)
            if motion is not None and motion >= self.rear_end_other_motion_px:
                return True
            if lateral is not None and lateral >= self.rear_end_other_lateral_px:
                return True
        return False

    def _has_nearby_context(self, pair: Tuple[int, int], object_by_id: Dict[int, Dict[str, object]]) -> bool:
        pair_objects = [object_by_id.get(tid) for tid in pair]
        if any(obj is None for obj in pair_objects):
            return False
        pair_center = (
            0.5 * (float(pair_objects[0]["center"][0]) + float(pair_objects[1]["center"][0])),
            0.5 * (float(pair_objects[0]["center"][1]) + float(pair_objects[1]["center"][1])),
        )
        for tid, obj in object_by_id.items():
            if tid in pair:
                continue
            center = tuple(float(v) for v in obj["center"])
            if self._distance(pair_center, center) <= self.rear_end_context_distance_px:
                return True
        return False

    def _append_event(
        self,
        frame_id: int,
        timestamp: float,
        event_type: str,
        track_id: int,
        related_track_id: Optional[int],
        class_name: str,
        description: str,
    ) -> Dict[str, object]:
        row = {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "event_type": event_type,
            "track_id": int(track_id),
            "related_track_id": "" if related_track_id is None else int(related_track_id),
            "class_name": class_name,
            "description": description,
            "evidence_path": "",
            "evidence_paths": "[]",
            "qwen_image_paths": "[]",
            "evidence_offsets_sec": "[]",
            "evidence_times_sec": "[]",
            "qwen_result": "",
            "qwen_reason": "",
            "qwen_error": "",
        }
        self.events.append(row)
        return row

    def update(self, frame_id: int, timestamp: float, tracked_objects: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        """输入当前帧目标列表，返回本帧新增异常风险事件。"""
        objects = [obj for obj in tracked_objects if str(obj.get("class_name", "")) in self.VEHICLE_CLS]
        frame_events: List[Dict[str, object]] = []
        object_by_id: Dict[int, Dict[str, object]] = {}

        for obj in objects:
            tid = int(obj["track_id"])
            bbox = tuple(int(v) for v in obj["bbox"])
            center = obj.get("center") or self.center_of_bbox(bbox)
            normalized = dict(obj)
            normalized["bbox"] = bbox
            normalized["center"] = (float(center[0]), float(center[1]))
            object_by_id[tid] = normalized
            self.track_points[tid].append(normalized["center"])

        for i, front in enumerate(objects):
            for rear in objects[i + 1 :]:
                tid_a = int(front["track_id"])
                tid_b = int(rear["track_id"])
                bbox_a = tuple(int(v) for v in front["bbox"])
                bbox_b = tuple(int(v) for v in rear["bbox"])
                overlap = self._x_overlap_ratio(bbox_a, bbox_b)
                gap = self._vertical_gap(bbox_a, bbox_b)
                height_ref = min(self._bbox_height(bbox_a), self._bbox_height(bbox_b))
                pair = tuple(sorted((tid_a, tid_b)))
                if overlap >= self.rear_end_x_overlap_ratio and gap <= height_ref * self.rear_end_y_gap_ratio:
                    self.rear_end_hits[pair] += 1
                else:
                    self.rear_end_hits[pair] = 0

                key = ("rear_end_risk", pair)
                if self.rear_end_hits[pair] >= self.rear_end_hold_frames and key not in self.reported:
                    self.rear_end_candidates.setdefault(
                        pair,
                        {
                            "start_frame": int(frame_id),
                            "start_timestamp": float(timestamp),
                            "overlap": float(overlap),
                            "gap": float(gap),
                            "class_name": str(front.get("class_name") or "vehicle"),
                        },
                    )

        for pair, candidate in list(self.rear_end_candidates.items()):
            key = ("rear_end_risk", pair)
            if key in self.reported:
                self.rear_end_candidates.pop(pair, None)
                continue
            if pair[0] not in object_by_id or pair[1] not in object_by_id:
                self.rear_end_candidates.pop(pair, None)
                continue
            elapsed_frames = int(frame_id) - int(candidate["start_frame"])
            pair_stopped = self._pair_is_stopped(pair)
            has_nearby = self._has_nearby_context(pair, object_by_id)
            has_moving_context = self._has_moving_context(pair, object_by_id)
            if pair_stopped and (has_moving_context or not has_nearby):
                self.reported.add(key)
                self.rear_end_candidates.pop(pair, None)
                overlap = float(candidate["overlap"])
                gap = float(candidate["gap"])
                context_text = "周边车辆仍在运动或存在横向绕行趋势" if has_moving_context else "周边未检测到可用于辅助判断的其他车辆"
                frame_events.append(
                    self._append_event(
                        frame_id,
                        timestamp,
                        "rear_end_risk",
                        pair[0],
                        pair[1],
                        str(candidate.get("class_name") or "vehicle"),
                        (
                            f"两个车辆横向重叠 {overlap:.2f}，纵向间距约 {gap:.1f}px；"
                            f"候选追尾后两车在观察窗口后段基本停止，{context_text}，提示疑似追尾风险"
                        ),
                    )
                )
            elif elapsed_frames > self.rear_end_confirm_frames:
                self.rear_end_candidates.pop(pair, None)

        return frame_events

    def export_csv(self, save_csv: str) -> None:
        """导出 anomalies.csv；即使没有事件也写入表头。"""
        path = Path(save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            rows = []
            for event in self.events:
                rows.append({field: event.get(field, "") for field in self.FIELDS})
            writer.writerows(rows)
