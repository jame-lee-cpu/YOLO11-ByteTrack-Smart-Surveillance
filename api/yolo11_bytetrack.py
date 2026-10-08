"""YOLO11 + ByteTrack 统一封装。"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

Ltrb = Tuple[int, int, int, int]
TrackOutput = Tuple[int, Ltrb]

PERSON_VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]
VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def parse_classes_spec(classes_spec: Optional[Union[str, Sequence[int]]]) -> Optional[List[int]]:
    """解析类别配置字符串。

    支持：
    - person_vehicle / mixed: 人车混合
    - vehicle: 仅车辆
    - person: 仅行人
    - all / none: 不做类别过滤
    - "0,1,2,3": 自定义类别 ID 列表
    """
    if classes_spec is None:
        return PERSON_VEHICLE_CLASSES
    if isinstance(classes_spec, (list, tuple)):
        return [int(v) for v in classes_spec] if classes_spec else None

    text = str(classes_spec).strip().lower()
    if text in {"person_vehicle", "mixed", "person+vehicle", "pv"}:
        return PERSON_VEHICLE_CLASSES
    if text in {"vehicle", "vehicles", "car"}:
        return VEHICLE_CLASSES
    if text in {"person", "people", "human"}:
        return [0]
    if text in {"all", "none", ""}:
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    parsed = []
    for p in parts:
        try:
            parsed.append(int(p))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --classes value: {classes_spec}. "
                "Use person_vehicle|vehicle|person|all or comma-separated IDs like 0,1,2,3,5,7."
            ) from exc
    return parsed or None


class YOLO11ByteTracker:
    """YOLO11 + ByteTrack 跟踪封装。"""

    def __init__(
        self,
        model_path: str = "yolo11x.pt",
        tracker: str = "trackers/bytetrack_stable.yaml",
        conf: float = 0.06,
        iou: float = 0.8,
        imgsz: int = 1536,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
        veh_conf: float = 0.08,
        moto_conf: float = 0.05,
        person_conf: float = 0.30,
        person_min_frames: int = 1,
        vehicle_min_frames: int = 1,
        vehicle_min_area_ratio: float = 0.0,
        edge_ignore_px: int = 0,
        vehicle_dedupe_iou: float = 0.9,
        lost_hold: int = 20,
        stable_keep_ratio: float = 0.55,
        stable_keep_floor_veh: float = 0.08,
        stable_keep_floor_moto: float = 0.05,
        stable_keep_floor_person: float = 0.08,
    ) -> None:
        self.model_path = model_path
        self.tracker = tracker
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = list(classes) if classes is not None else PERSON_VEHICLE_CLASSES
        self.device = device
        self.veh_conf = float(veh_conf)
        self.moto_conf = float(moto_conf)
        self.person_conf = float(person_conf)
        self.person_min_frames = int(max(1, person_min_frames))
        self.vehicle_min_frames = int(max(1, vehicle_min_frames))
        self.vehicle_min_area_ratio = float(max(0.0, vehicle_min_area_ratio))
        self.edge_ignore_px = int(max(0, edge_ignore_px))
        self.vehicle_dedupe_iou = float(np.clip(vehicle_dedupe_iou, 0.1, 0.95))
        self.lost_hold = int(max(0, lost_hold))
        self.stable_keep_ratio = float(np.clip(stable_keep_ratio, 0.3, 1.0))
        self.stable_keep_floor_veh = float(max(0.0, stable_keep_floor_veh))
        self.stable_keep_floor_moto = float(max(0.0, stable_keep_floor_moto))
        self.stable_keep_floor_person = float(max(0.0, stable_keep_floor_person))
        self.model = YOLO(self.model_path)
        self.frame_idx = -1
        self.track_hits: Dict[int, int] = defaultdict(int)
        self.last_bbox: Dict[int, Ltrb] = {}
        self.last_seen_frame: Dict[int, int] = {}
        self.emitted_ids = set()

    def _pass_class_conf(self, cls_id: int, score: float, track_id: Optional[int] = None) -> bool:
        # 对已建立轨迹启用更低“保活阈值”，减少阴影/半出镜导致的ID断裂。
        is_emitted = track_id is not None and int(track_id) in self.emitted_ids
        ratio = self.stable_keep_ratio if is_emitted else 1.0
        if cls_id == 0:
            th = self.person_conf * ratio
            if is_emitted:
                th = max(th, self.stable_keep_floor_person)
            return score >= th
        if cls_id == 3:
            th = self.moto_conf * ratio
            if is_emitted:
                th = max(th, self.stable_keep_floor_moto)
            return score >= th
        if cls_id in VEHICLE_CLASSES:
            th = self.veh_conf * ratio
            if is_emitted:
                th = max(th, self.stable_keep_floor_veh)
            return score >= th
        return False

    @staticmethod
    def _bbox_area(bbox: Ltrb) -> float:
        l, t, r, b = bbox
        return float(max(0, r - l) * max(0, b - t))

    def _touches_border(self, bbox: Ltrb, frame_w: int, frame_h: int) -> bool:
        l, t, r, b = bbox
        m = self.edge_ignore_px
        return l <= m or t <= m or r >= (frame_w - 1 - m) or b >= (frame_h - 1 - m)

    @staticmethod
    def _iou(a: Ltrb, b: Ltrb) -> float:
        al, at, ar, ab = a
        bl, bt, br, bb = b
        il = max(al, bl)
        it = max(at, bt)
        ir = min(ar, br)
        ib = min(ab, bb)
        iw = max(0, ir - il)
        ih = max(0, ib - it)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(0, ar - al) * max(0, ab - at))
        area_b = float(max(0, br - bl) * max(0, bb - bt))
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    @classmethod
    def _containment_ratio(cls, a: Ltrb, b: Ltrb) -> float:
        """返回两框交集占较小框面积的比例，用于抑制局部车辆框。"""
        al, at, ar, ab = a
        bl, bt, br, bb = b
        il = max(al, bl)
        it = max(at, bt)
        ir = min(ar, br)
        ib = min(ab, bb)
        iw = max(0, ir - il)
        ih = max(0, ib - it)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        return inter / max(1e-6, min(cls._bbox_area(a), cls._bbox_area(b)))

    def _is_duplicate_vehicle_box(self, box_a: Ltrb, box_b: Ltrb) -> bool:
        return (
            self._iou(box_a, box_b) >= self.vehicle_dedupe_iou
            or self._containment_ratio(box_a, box_b) >= 0.82
        )

    def _vehicle_dedupe(
        self, candidates: List[Tuple[int, Ltrb, float, int]]
    ) -> List[Tuple[int, Ltrb, float, int]]:
        """同帧车辆去重：高重叠时优先保留更大且更稳定的框。"""
        if len(candidates) <= 1:
            return candidates
        keep = [True] * len(candidates)
        for i in range(len(candidates)):
            if not keep[i]:
                continue
            tid_i, box_i, score_i, cls_i = candidates[i]
            if cls_i not in VEHICLE_CLASSES:
                continue
            area_i = self._bbox_area(box_i)
            for j in range(i + 1, len(candidates)):
                if not keep[j]:
                    continue
                tid_j, box_j, score_j, cls_j = candidates[j]
                if cls_j not in VEHICLE_CLASSES:
                    continue
                if not self._is_duplicate_vehicle_box(box_i, box_j):
                    continue
                area_j = self._bbox_area(box_j)
                # 保留面积更大或置信度更高的框，抑制货物/局部重复框。
                score_rank_i = score_i + 0.03 * min(self.track_hits.get(tid_i, 0), 20)
                score_rank_j = score_j + 0.03 * min(self.track_hits.get(tid_j, 0), 20)
                if area_i >= area_j:
                    if score_rank_i >= score_rank_j or area_i >= area_j * 1.2:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                else:
                    if score_rank_j >= score_rank_i or area_j >= area_i * 1.2:
                        keep[i] = False
                        break
                    keep[j] = False
        return [c for k, c in zip(keep, candidates) if k]

    def track_frame(self, frame_bgr: np.ndarray) -> Tuple[List[TrackOutput], np.ndarray]:
        """处理单帧，返回 `(tracks, detections)`。

        - tracks: `[(track_id, (l, t, r, b)), ...]`
        - detections: `Nx6`，列顺序为 `[x1, y1, x2, y2, conf, cls]`
        """
        self.frame_idx += 1
        results = self.model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        raw_boxes = results[0].boxes if results else None

        frame_h, frame_w = frame_bgr.shape[:2]
        candidates: List[Tuple[int, Ltrb, float, int]] = []
        active_filtered_ids = set()

        if raw_boxes is not None and len(raw_boxes) > 0 and raw_boxes.id is not None:
            xyxy = raw_boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf = raw_boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1)
            cls = raw_boxes.cls.detach().cpu().numpy().astype(np.int32).reshape(-1)
            ids = raw_boxes.id.detach().cpu().numpy().astype(np.int32).reshape(-1)
            n = min(len(ids), len(xyxy), len(conf), len(cls))

            for i in range(n):
                cls_id = int(cls[i])
                score = float(conf[i])
                track_id = int(ids[i])
                if self.classes is not None and cls_id not in self.classes:
                    continue
                if not self._pass_class_conf(cls_id, score, track_id):
                    continue

                x1, y1, x2, y2 = xyxy[i]
                bbox: Ltrb = (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                )
                active_filtered_ids.add(track_id)
                self.last_bbox[track_id] = bbox
                self.last_seen_frame[track_id] = self.frame_idx
                self.track_hits[track_id] += 1

                if cls_id in VEHICLE_CLASSES:
                    # 过滤过小车辆框（常见于货物/局部误检）。
                    min_area = self.vehicle_min_area_ratio * float(frame_w * frame_h)
                    if self._bbox_area(bbox) < min_area:
                        continue

                # 类别稳定确认：person 和 vehicle 都要求短时连续命中。
                required_frames = self.person_min_frames if cls_id == 0 else self.vehicle_min_frames
                if self.track_hits[track_id] < required_frames:
                    continue

                # 车辆入画边缘抑制：未完整出现时暂不输出，减少“半车多框”。
                if cls_id in VEHICLE_CLASSES and self._touches_border(bbox, frame_w, frame_h):
                    continue

                self.emitted_ids.add(track_id)
                candidates.append((track_id, bbox, score, cls_id))

        candidates = self._vehicle_dedupe(candidates)
        tracks: List[TrackOutput] = []
        det_rows: List[List[float]] = []
        for track_id, bbox, score, cls_id in candidates:
            tracks.append((track_id, bbox))
            l, t, r, b = bbox
            det_rows.append([float(l), float(t), float(r), float(b), float(score), float(cls_id)])

        # 丢失补偿：短时未匹配时复用最近 bbox，降低“突然消失”的感知。
        for track_id, bbox in list(self.last_bbox.items()):
            if track_id in active_filtered_ids:
                continue
            last_seen = self.last_seen_frame.get(track_id, -10**9)
            if track_id in self.emitted_ids and (self.frame_idx - last_seen) <= self.lost_hold:
                if any(self._is_duplicate_vehicle_box(bbox, current_bbox) for _, current_bbox in tracks):
                    continue
                tracks.append((track_id, bbox))

        # 清理长期失活轨迹，避免状态无限膨胀。
        stale_ids = []
        max_stale = self.lost_hold + 4
        for track_id, last_seen in self.last_seen_frame.items():
            if (self.frame_idx - last_seen) > max_stale:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            self.last_seen_frame.pop(track_id, None)
            self.last_bbox.pop(track_id, None)
            self.track_hits.pop(track_id, None)
            self.emitted_ids.discard(track_id)

        detections = (
            np.asarray(det_rows, dtype=np.float32)
            if det_rows
            else np.empty((0, 6), dtype=np.float32)
        )
        return tracks, detections

    def process_video(
        self,
        input_vid: str,
        save_path: Optional[str] = None,
        on_frame: Optional[
            Callable[[int, np.ndarray, List[TrackOutput], np.ndarray], np.ndarray]
        ] = None,
    ) -> int:
        """视频处理辅助接口。

        `on_frame` 回调签名：`(frame_idx, frame_bgr, tracks, detections) -> frame_bgr`。
        """
        cap = cv2.VideoCapture(input_vid)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_vid}")

        writer = None
        if save_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 25.0
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_idx = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                tracks, detections = self.track_frame(frame)
                out_frame = frame
                if on_frame is not None:
                    out_frame = on_frame(frame_idx, frame, tracks, detections)
                if writer is not None:
                    writer.write(out_frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        return frame_idx + 1
