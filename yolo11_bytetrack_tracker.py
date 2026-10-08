"""YOLO11 + ByteTrack 跟踪器构建模块。

本模块负责把系统配置整理成 YOLO11ByteTracker 所需参数，避免主流程
直接依赖过多模型和跟踪器细节。
"""

from __future__ import annotations

from api.yolo11_bytetrack import YOLO11ByteTracker, parse_classes_spec


def build_tracker(
    model_path: str,
    tracker_cfg: str,
    yolo_conf: float,
    yolo_iou: float,
    imgsz: int,
    classes_spec: str,
    device: str,
    veh_conf: float,
    moto_conf: float,
    person_conf: float,
    person_min_frames: int,
    vehicle_min_frames: int,
    vehicle_min_area_ratio: float,
    edge_ignore_px: int,
    vehicle_dedupe_iou: float,
    lost_hold: int,
    stable_keep_ratio: float,
):
    """根据统一配置创建一帧一帧调用的检测跟踪器。"""
    classes = parse_classes_spec(classes_spec)
    return YOLO11ByteTracker(
        model_path=model_path,
        tracker=tracker_cfg,
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=imgsz,
        classes=classes,
        device=device,
        veh_conf=veh_conf,
        moto_conf=moto_conf,
        person_conf=person_conf,
        person_min_frames=person_min_frames,
        vehicle_min_frames=vehicle_min_frames,
        vehicle_min_area_ratio=vehicle_min_area_ratio,
        edge_ignore_px=edge_ignore_px,
        vehicle_dedupe_iou=vehicle_dedupe_iou,
        lost_hold=lost_hold,
        stable_keep_ratio=stable_keep_ratio,
    )
