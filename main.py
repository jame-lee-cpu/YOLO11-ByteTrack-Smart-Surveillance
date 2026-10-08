"""主流程入口：YOLO11 检测 + ByteTrack 跟踪 + 交互式目标选择 + 轨迹导出。"""

import argparse
import csv
import json
import os
from os import path
from time import perf_counter, sleep
from collections import defaultdict, deque
from urllib.parse import urlparse

import cv2
import numpy as np

from analytics import AnalyticsEngine
from analytics import TrajectoryAnalyzer
from anomaly_detector import AnomalyDetector
from qwen_reviewer import review_rear_end_with_qwen
from llm_payload import export_llm_violation_payload
from config import (
    ANOMALY_EVIDENCE_OFFSETS_SEC,
    ANOMALY_QWEN_EXIT_CHECK_SEC,
    ANOMALY_QWEN_FULL_IMAGE_MAX_SIZE_PX,
    ANOMALY_QWEN_OFFSETS_SEC,
    ANOMALY_ENABLED,
    ANOMALY_MAX_SNAPSHOTS,
    ANOMALY_REAR_END_HOLD_FRAMES,
    ANOMALY_REAR_END_CONFIRM_FRAMES,
    ANOMALY_REAR_END_CONTEXT_DISTANCE_PX,
    ANOMALY_REAR_END_MAX_POST_MOVE_PX,
    ANOMALY_REAR_END_MOTION_CHECK_OFFSETS_SEC,
    ANOMALY_REAR_END_OTHER_LATERAL_PX,
    ANOMALY_REAR_END_OTHER_MOTION_PX,
    ANOMALY_REAR_END_STOP_MOVE_PX,
    ANOMALY_REAR_END_STOP_WINDOW_FRAMES,
    ANOMALY_REAR_END_X_OVERLAP_RATIO,
    ANOMALY_REAR_END_Y_GAP_RATIO,
    CLASSES_SPEC,
    COUNT_LINE,
    CROSS_TOLERANCE,
    QWEN_API_KEY_ENV,
    QWEN_API_URL,
    QWEN_IMAGE_JPEG_QUALITY,
    QWEN_MAX_PROMPT_CHARS,
    QWEN_MODEL,
    QWEN_REVIEW_ENABLED,
    QWEN_TIMEOUT_SEC,
    EDGE_IGNORE_PX,
    IMGSZ,
    LINE1,
    LINE2,
    LINE_DISTANCE_M,
    LLM_MAX_EVENTS,
    LLM_PAYLOAD_NAME,
    LOST_HOLD,
    MOTO_CONF,
    MAX_SPEED_KMH,
    MIN_SPEED_TIME_S,
    PERSON_CONF,
    PERSON_MIN_FRAMES,
    PIXEL_TO_METER,
    SPEED_DIRECTION_MODE,
    SPEED_MODE,
    STABLE_KEEP_RATIO,
    STRIDE,
    TRAJECTORY_RELINK_ENABLED,
    TRAJECTORY_RELINK_MAX_DIST,
    TRAJECTORY_RELINK_MAX_GAP,
    TRACKER_CFG,
    TRAIL_LEN,
    VEH_CONF,
    VEHICLE_DEDUPE_IOU,
    VEHICLE_MIN_AREA_RATIO,
    VEHICLE_MIN_FRAMES,
    WEB_STREAM_OPEN_TIMEOUT_MS,
    WEB_STREAM_READ_TIMEOUT_MS,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MODEL,
    DEFAULT_REPORT_DIR,
)
from yolo11_bytetrack_tracker import build_tracker

# 默认参数别名，供命令行入口和兼容函数复用。
DEFAULT_TRACKER_CFG = TRACKER_CFG
DEFAULT_YOLO_CONF = YOLO_CONF
DEFAULT_YOLO_IOU = YOLO_IOU
DEFAULT_IMGSZ = IMGSZ
DEFAULT_CLASSES = CLASSES_SPEC
DEFAULT_VEH_CONF = VEH_CONF
DEFAULT_MOTO_CONF = MOTO_CONF
DEFAULT_PERSON_CONF = PERSON_CONF
DEFAULT_PERSON_MIN_FRAMES = PERSON_MIN_FRAMES
DEFAULT_VEHICLE_MIN_FRAMES = VEHICLE_MIN_FRAMES
DEFAULT_VEHICLE_MIN_AREA_RATIO = VEHICLE_MIN_AREA_RATIO
DEFAULT_EDGE_IGNORE_PX = EDGE_IGNORE_PX
DEFAULT_VEHICLE_DEDUPE_IOU = VEHICLE_DEDUPE_IOU
DEFAULT_TRAIL_LEN = TRAIL_LEN
DEFAULT_LOST_HOLD = LOST_HOLD
DEFAULT_STABLE_KEEP_RATIO = STABLE_KEEP_RATIO
WINDOW_NAME = "Detections"
DEFAULT_COUNTS_CSV = "./counts.csv"
DEFAULT_EVENTS_CSV = "./events.csv"
DEFAULT_TRAFFIC_CSV = "./traffic.csv"
DEFAULT_ANOMALIES_CSV = "./anomalies.csv"
DEFAULT_LLM_PAYLOAD_JSON = f"./{LLM_PAYLOAD_NAME}"
DEFAULT_PIXEL_TO_METER = PIXEL_TO_METER


def is_stream_source(value: str) -> bool:
    """判断输入视频是否为网络流地址。"""
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() in {"http", "https", "rtmp"} and bool(parsed.netloc)


def open_input_video(value: str) -> cv2.VideoCapture:
    """打开本地视频或网络流；网络流设置超时，避免不可达时阻塞太久。"""
    source = str(value)
    if is_stream_source(source):
        params = []
        # OpenCV 的 FFmpeg 后端支持打开/读取超时；旧版本没有属性时自动回退。
        open_timeout_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout_prop is not None:
            params.extend([open_timeout_prop, int(WEB_STREAM_OPEN_TIMEOUT_MS)])
        if read_timeout_prop is not None:
            params.extend([read_timeout_prop, int(WEB_STREAM_READ_TIMEOUT_MS)])
        if params:
            try:
                capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture
            except Exception:
                pass
    capture = cv2.VideoCapture(source)
    if is_stream_source(source):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class AnalysisCancelled(RuntimeError):
    """网页任务请求提前停止时抛出的内部异常。"""


def parse_xyxy_arg(value: str):
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("坐标格式应为 x1,y1,x2,y2")
    try:
        return tuple(int(float(p)) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("坐标必须为数字") from exc


def parse_ids_arg(value: str) -> set[int]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    try:
        return {int(p.strip()) for p in raw.split(",") if p.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ID格式应为 12 或 12,15") from exc


# =========================
# 车流+测速配置（手动修改区）
# =========================
# 可直接修改下面配置以适配不同视频：
# 坐标及测速配置已统一放入 config.py，可在其中集中调参。

CLASS_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def bbox_center_ltrb(bbox_ltrb):
    """根据 ltrb=(左, 上, 右, 下) 边框计算中心点。"""
    l, t, r, b = bbox_ltrb
    return int((l + r) * 0.5), int((t + b) * 0.5)


def point_in_bbox(x, y, bbox_ltrb, pad=12):
    """判断鼠标点击点是否落在目标框内（含边缘容差）。"""
    l, t, r, b = bbox_ltrb
    return (l - pad) <= x <= (r + pad) and (t - pad) <= y <= (b + pad)


def point_near_frame_edge(center_xy, frame_shape, edge_margin=24):
    """判断中心点是否靠近画面边缘，避免车辆离场后被强行重连到新框。"""
    if frame_shape is None:
        return False
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return False
    x, y = center_xy
    margin = max(0, int(edge_margin))
    return x <= margin or y <= margin or x >= (w - margin) or y >= (h - margin)


def pick_track_id_from_click(click_xy, tracks):
    """根据点击坐标选中 track_id（框内优先，未命中时按最近中心吸附）。"""
    if click_xy is None:
        return None

    x, y = click_xy
    best_id = None
    best_area = None
    best_dist = None
    nearest_id = None
    for track_id, bbox_ltrb in tracks:
        l, t, r, b = bbox_ltrb
        cx, cy = bbox_center_ltrb(bbox_ltrb)
        dist = float(np.hypot(x - cx, y - cy))

        if point_in_bbox(x, y, bbox_ltrb, pad=12):
            area = max(1, (r - l) * (b - t))
            if best_area is None or area < best_area:
                best_area = area
                best_id = int(track_id)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            nearest_id = int(track_id)

    if best_id is not None:
        return best_id

    # 点击没落在框内时，允许对最近目标做一次“吸附”，提升操作手感。
    if nearest_id is not None and best_dist is not None and best_dist <= 40.0:
        return nearest_id
    return best_id


def on_mouse_click(event, x, y, flags, state):
    """OpenCV 鼠标事件回调：记录待处理的点击位置。"""
    _ = flags
    if event == cv2.EVENT_LBUTTONDOWN:
        state["pending_left_click"] = (int(x), int(y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["pending_right_click"] = (int(x), int(y))


def draw_selected_trails(frame, selected_track_points):
    """绘制所有已选中目标的历史轨迹线。"""
    for track_id, points in selected_track_points.items():
        if len(points) < 2:
            continue
        pts = list(points)
        color = (0, 0, 255) if (track_id % 2 == 0) else (0, 165, 255)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 3)


def draw_track_trails(frame, track_history, color=(255, 190, 0), thickness=2):
    """绘制全目标轨迹线（统一颜色）。"""
    for _, points in track_history.items():
        if len(points) < 2:
            continue
        pts = list(points)
        for i in range(1, len(pts)):
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(frame, p1, p2, color, thickness)


def remap_track_for_trail(
    raw_tid: int,
    cls_id,
    center_xy,
    frame_idx: int,
    prev_raw_to_logical,
    logical_last_center,
    logical_last_frame,
    logical_class,
    used_logical_ids,
    frame_shape=None,
):
    """仅用于轨迹显示的轻量ID重连，降低短时丢失后的轨迹断裂。"""
    if raw_tid in prev_raw_to_logical:
        lid = int(prev_raw_to_logical[raw_tid])
        if lid not in used_logical_ids:
            return lid

    if not TRAJECTORY_RELINK_ENABLED:
        return int(raw_tid)

    best_lid = None
    best_dist = None
    for lid, last_pt in logical_last_center.items():
        if lid in used_logical_ids:
            continue
        last_f = int(logical_last_frame.get(lid, -10**9))
        if frame_idx - last_f > int(TRAJECTORY_RELINK_MAX_GAP):
            continue
        if point_near_frame_edge(last_pt, frame_shape, edge_margin=24):
            continue
        if cls_id is not None and logical_class.get(lid) is not None and int(logical_class.get(lid)) != int(cls_id):
            continue
        dx = float(center_xy[0] - last_pt[0])
        dy = float(center_xy[1] - last_pt[1])
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist > float(TRAJECTORY_RELINK_MAX_DIST):
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_lid = int(lid)

    if best_lid is not None:
        return best_lid
    return int(raw_tid)


def draw_scale(frame) -> float:
    """根据视频分辨率缩放叠加文字和线宽，小画面避免文字占满屏幕。"""
    h, w = frame.shape[:2]
    return max(0.38, min(1.0, min(float(w) / 1280.0, float(h) / 720.0)))


def draw_track_boxes(
    frame,
    tracks,
    selected_ids=None,
    rear_end_ids=None,
    only_selected=False,
    speed_map=None,
    track_class_map=None,
):
    """按当前选择状态绘制跟踪框。"""
    selected_ids = selected_ids or set()
    rear_end_ids = rear_end_ids or set()
    speed_map = speed_map or {}
    track_class_map = track_class_map or {}
    scale = draw_scale(frame)
    box_thin = max(1, int(round(2 * scale)))
    box_thick = max(2, int(round(4 * scale)))
    text_scale = max(0.30, 0.55 * scale)
    text_thickness = max(1, int(round(1 * scale)))
    for track_id, bbox_ltrb in tracks:
        is_selected = int(track_id) in selected_ids
        is_rear_end = int(track_id) in rear_end_ids
        if only_selected and not is_selected:
            continue

        l, t, r, b = bbox_ltrb
        if is_rear_end:
            color = (255, 0, 255)
            thickness = box_thick
        elif is_selected:
            color = (0, 0, 255)
            thickness = box_thick
        else:
            color = (0, 220, 0)
            thickness = box_thin
        cls_id = track_class_map.get(int(track_id))
        cls_name = CLASS_NAME.get(int(cls_id), "unknown") if cls_id is not None else "unknown"
        speed_kmh = float(speed_map.get(int(track_id), 0.0))
        if speed_kmh > 0:
            label = f"ID:{track_id} {cls_name} {speed_kmh:.1f}km/h"
        else:
            label = f"ID:{track_id} {cls_name}"
        if is_rear_end:
            label = f"REAR-END RISK {label}"
        cv2.rectangle(frame, (l, t), (r, b), color, thickness)
        cv2.putText(
            frame,
            label,
            (l, max(10, t - max(4, int(8 * scale)))),
            cv2.FONT_HERSHEY_DUPLEX,
            text_scale,
            color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )


def draw_traffic_status(frame, analytics):
    """绘制车流量信息。"""
    measured_n, avg_speed, max_speed = analytics.speed_summary()
    scale = draw_scale(frame)
    font_scale = max(0.32, 0.65 * scale)
    thickness = max(1, int(round(2 * scale)))
    margin = max(4, int(12 * scale))
    line_gap = max(12, int(24 * scale))
    y0 = max(14, int(28 * scale))
    if analytics.speed_direction_mode == "vertical":
        up_count = int(analytics.line_counts.get("up", 0))
        down_count = int(analytics.line_counts.get("down", 0))
        lines = [
            f"Traffic Total: {up_count + down_count}",
            f"Up (Down->Up): {up_count}",
            f"Down (Up->Down): {down_count}",
            f"Measured: {measured_n}",
            f"Avg/Max: {avg_speed:.1f}/{max_speed:.1f} km/h",
        ]
    else:
        left_count = int(analytics.line_counts.get("left", 0))
        right_count = int(analytics.line_counts.get("right", 0))
        lines = [
            f"Traffic Total: {left_count + right_count}",
            f"Left (Right->Left): {left_count}",
            f"Right (Left->Right): {right_count}",
            f"Measured: {measured_n}",
            f"Avg/Max: {avg_speed:.1f}/{max_speed:.1f} km/h",
        ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (margin, y0 + i * line_gap),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

def draw_analysis_guides(frame, line1_xyxy, line2_xyxy, count_line_xyxy=None):
    """绘制统计线与双测速线。"""
    scale = draw_scale(frame)
    line_thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.32, 0.65 * scale)
    text_thickness = max(1, int(round(2 * scale)))
    offset = max(8, int(24 * scale))
    x_offset = max(3, int(6 * scale))
    if count_line_xyxy is not None:
        cx1, cy1, cx2, cy2 = count_line_xyxy
        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 165, 255), line_thickness)
        cv2.putText(
            frame,
            "Count Line",
            (cx1 + x_offset, min(frame.shape[0] - 4, cy1 + offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 165, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    x1, y1, x2, y2 = line1_xyxy
    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), line_thickness)
    cv2.putText(
        frame,
        "Speed Line 1",
        (x1 + x_offset, min(frame.shape[0] - 4, y1 + offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 0),
        text_thickness,
        lineType=cv2.LINE_AA,
    )
    x3, y3, x4, y4 = line2_xyxy
    cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 255), line_thickness)
    cv2.putText(
        frame,
        "Speed Line 2",
        (x3 + x_offset, min(frame.shape[0] - 4, y3 + offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        text_thickness,
        lineType=cv2.LINE_AA,
    )


def bbox_iou(a, b):
    """计算两个 ltrb 边框之间的 IoU。"""
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
    return inter / max(1e-6, union)


def build_track_class_map(tracks, detections, cache):
    """按 IoU 将当前帧检测类别绑定到 track_id（无匹配时用缓存）。"""
    track_class_map = {}
    if detections is not None and getattr(detections, "size", 0) > 0:
        det_boxes = [tuple(int(round(v)) for v in row[:4]) for row in detections]
        det_cls = [int(row[5]) for row in detections]
    else:
        det_boxes = []
        det_cls = []

    for tid, box in tracks:
        best_iou = 0.0
        best_cls = None
        for dbox, dcls in zip(det_boxes, det_cls):
            iou = bbox_iou(box, dbox)
            if iou > best_iou:
                best_iou = iou
                best_cls = dcls
        if best_cls is not None and best_iou >= 0.1:
            track_class_map[int(tid)] = int(best_cls)
            cache[int(tid)] = int(best_cls)
        elif int(tid) in cache:
            track_class_map[int(tid)] = int(cache[int(tid)])
    return track_class_map


def clip_line_to_frame(line_xyxy, frame_w, frame_h):
    """将统计线坐标裁剪到画面范围内，避免坐标越界。"""
    x1, y1, x2, y2 = line_xyxy
    return (
        int(np.clip(x1, 0, max(0, frame_w - 1))),
        int(np.clip(y1, 0, max(0, frame_h - 1))),
        int(np.clip(x2, 0, max(0, frame_w - 1))),
        int(np.clip(y2, 0, max(0, frame_h - 1))),
    )


def apply_click_actions(
    click_state,
    tracks,
    selected_ids_runtime,
    selected_lost_frames,
    selected_track_points,
):
    """处理鼠标点击动作（左键追加、右键取消）。

    说明：
    - 左键命中目标：将该 track_id 加入选中集合
    - 右键命中目标：从选中集合移除该 track_id
    - 右键未命中目标：清空全部选中目标
    """
    # 左键：追加框选目标
    if click_state["pending_left_click"] is not None:
        new_selected_id = pick_track_id_from_click(click_state["pending_left_click"], tracks)
        if new_selected_id is not None and new_selected_id not in selected_ids_runtime:
            selected_ids_runtime.add(new_selected_id)
            selected_lost_frames[new_selected_id] = 0
            selected_track_points[new_selected_id].clear()
            print(f"<< Target Added: ID {new_selected_id} >>")
        click_state["pending_left_click"] = None

    # 右键：取消框选（命中则取消该目标，否则清空全部）
    if click_state["pending_right_click"] is not None:
        remove_id = pick_track_id_from_click(click_state["pending_right_click"], tracks)
        if remove_id is not None and remove_id in selected_ids_runtime:
            selected_ids_runtime.remove(remove_id)
            selected_lost_frames.pop(remove_id, None)
            selected_track_points.pop(remove_id, None)
            print(f"<< Target Removed: ID {remove_id} >>")
        else:
            if selected_ids_runtime:
                print("<< All selected targets cleared >>")
            selected_ids_runtime.clear()
            selected_lost_frames.clear()
            selected_track_points.clear()
        click_state["pending_right_click"] = None


def export_tracks_csv(records, save_csv):
    """导出目标轨迹记录到 CSV。"""
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "track_id", "class_name", "cx", "cy", "timestamp"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def video_writer_same_codec(video: cv2.VideoCapture, save_path: str, stride: int = 1) -> cv2.VideoWriter:
    """创建与输入视频分辨率一致、帧率按 stride 折算的 VideoWriter。

    优先尝试兼容性和质量更好的 avc1；如果当前环境不可用，则回退到
    mp4v，避免因为编码器缺失直接中断分析。
    """
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    stride = max(1, int(stride))
    out_fps = max(1.0, fps / stride)
    codec = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(save_path, codec, out_fps, (w, h))
    if not writer.isOpened():
        # 部分环境没有 avc1 编码器，回退到 mp4v 保证结果视频可写。
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, codec, out_fps, (w, h))
    return writer


def scale_window_for_stride(value: int, stride: int, minimum: int = 1) -> int:
    """把基于源帧调好的窗口折算到当前处理帧间隔。"""
    stride = max(1, int(stride))
    return max(int(minimum), int(np.ceil(float(value) / float(stride))))


def run_surveillance_pipeline(
    input_vid: str,
    save_path: str,
    show: bool = True,
    log_interval: int = 30,
    stride: int = 3,
    trail_len: int = DEFAULT_TRAIL_LEN,
    selected_id: int = None,
    highlight_ids=None,
    trail_mode: str = "all",
    tracks_csv: str = "tracks.csv",
    selected_csv: str = "selected_track.csv",
    counts_csv: str = DEFAULT_COUNTS_CSV,
    events_csv: str = DEFAULT_EVENTS_CSV,
    traffic_csv: str = DEFAULT_TRAFFIC_CSV,
    anomalies_csv: str = DEFAULT_ANOMALIES_CSV,
    llm_payload_json: str = DEFAULT_LLM_PAYLOAD_JSON,
    reports_dir: str = DEFAULT_REPORT_DIR,
    only_selected: bool = False,
    lost_hold: int = DEFAULT_LOST_HOLD,
    pixel_to_meter: float = DEFAULT_PIXEL_TO_METER,
    model_path: str = YOLO_MODEL,
    tracker_cfg: str = DEFAULT_TRACKER_CFG,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    yolo_iou: float = DEFAULT_YOLO_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    classes_spec: str = DEFAULT_CLASSES,
    device: str = None,
    veh_conf: float = DEFAULT_VEH_CONF,
    moto_conf: float = DEFAULT_MOTO_CONF,
    person_conf: float = DEFAULT_PERSON_CONF,
    person_min_frames: int = DEFAULT_PERSON_MIN_FRAMES,
    vehicle_min_frames: int = DEFAULT_VEHICLE_MIN_FRAMES,
    vehicle_min_area_ratio: float = DEFAULT_VEHICLE_MIN_AREA_RATIO,
    edge_ignore_px: int = DEFAULT_EDGE_IGNORE_PX,
    vehicle_dedupe_iou: float = DEFAULT_VEHICLE_DEDUPE_IOU,
    stable_keep_ratio: float = DEFAULT_STABLE_KEEP_RATIO,
    show_guides: bool = True,
    count_line_xyxy=COUNT_LINE,
    line1_xyxy=LINE1,
    line2_xyxy=LINE2,
    line_distance_m: float = LINE_DISTANCE_M,
    speed_mode: str = SPEED_MODE,
    speed_direction_mode: str = SPEED_DIRECTION_MODE,
    cross_tolerance: int = CROSS_TOLERANCE,
    min_speed_time_s: float = MIN_SPEED_TIME_S,
    max_speed_kmh: float = MAX_SPEED_KMH,
    anomaly_enabled: bool = ANOMALY_ENABLED,
    frame_callback=None,
    tracking_callback=None,
    stats_callback=None,
    rear_end_review_callback=None,
    runtime_control_callback=None,
    trajectory_summary_json=None,
    qwen_review_enabled=None,
    qwen_api_key: str = "",
):
    """
    Run one complete video analysis task.

    该函数集中承载视频分析流程，Flask 后台任务和命令行入口都会调用它，
    从而复用同一套 YOLO11 + ByteTrack 检测跟踪、车流统计、测速、
    轨迹导出和统计图导出逻辑。
    """
    # 每个任务只初始化一次 YOLO11 检测器和 ByteTrack 关联封装。
    tracker = build_tracker(
        model_path=model_path,
        tracker_cfg=tracker_cfg,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        imgsz=imgsz,
        classes_spec=classes_spec,
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

    # 先打开输入/输出视频流，失败时尽早报错，便于网页任务进入 failed 状态。
    video = open_input_video(input_vid)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_vid}")
    src_fps = float(video.get(cv2.CAP_PROP_FPS))
    if src_fps <= 0:
        src_fps = 25.0
    stride = max(1, int(stride))
    total_source_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration_sec = float(total_source_frames) / max(1e-6, src_fps) if total_source_frames > 0 else 0.0
    output = video_writer_same_codec(video, save_path, stride=stride)
    if not output.isOpened():
        video.release()
        raise RuntimeError(f"Cannot create output video: {save_path}")
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count_line_xyxy = clip_line_to_frame(count_line_xyxy, frame_w, frame_h)
    line1_xyxy = clip_line_to_frame(line1_xyxy, frame_w, frame_h)
    line2_xyxy = clip_line_to_frame(line2_xyxy, frame_w, frame_h)
    # AnalyticsEngine 负责计数、双线测速和 CSV 行缓存。网页端保存的是原始
    # 视频坐标，这里仍做一次边界裁剪，避免线条靠近画面边缘时越界。
    analytics = AnalyticsEngine(
        fps=src_fps,
        count_line=count_line_xyxy,
        line1=line1_xyxy,
        line2=line2_xyxy,
        line_distance_m=line_distance_m,
        speed_mode=speed_mode,
        speed_direction_mode=speed_direction_mode,
        cross_tolerance=cross_tolerance,
        pixel_to_meter=pixel_to_meter,
        min_speed_time_s=min_speed_time_s,
        max_speed_kmh=max_speed_kmh,
    )
    trajectory_analyzer = TrajectoryAnalyzer(maxlen=trail_len)
    anomaly_detector = None
    if anomaly_enabled:
        # 异常风险提示只做保守规则，不参与检测/跟踪主流程决策。
        anomaly_detector = AnomalyDetector(
            rear_end_hold_frames=scale_window_for_stride(ANOMALY_REAR_END_HOLD_FRAMES, stride, minimum=2),
            rear_end_x_overlap_ratio=ANOMALY_REAR_END_X_OVERLAP_RATIO,
            rear_end_y_gap_ratio=ANOMALY_REAR_END_Y_GAP_RATIO,
            rear_end_confirm_frames=scale_window_for_stride(ANOMALY_REAR_END_CONFIRM_FRAMES, stride, minimum=6),
            rear_end_stop_window_frames=scale_window_for_stride(ANOMALY_REAR_END_STOP_WINDOW_FRAMES, stride, minimum=4),
            rear_end_stop_move_px=ANOMALY_REAR_END_STOP_MOVE_PX,
            rear_end_context_distance_px=ANOMALY_REAR_END_CONTEXT_DISTANCE_PX,
            rear_end_other_motion_px=ANOMALY_REAR_END_OTHER_MOTION_PX,
            rear_end_other_lateral_px=ANOMALY_REAR_END_OTHER_LATERAL_PX,
        )
    output_base_dir = path.dirname(path.abspath(anomalies_csv))
    anomaly_snapshot_dir = path.join(output_base_dir, "anomaly_snapshots")
    qwen_snapshot_dir = path.join(output_base_dir, "qwen_snapshots")
    qwen_review_report_jsonl = path.join(output_base_dir, "qwen_review_report.jsonl")
    anomaly_snapshot_count = 0
    evidence_offsets_sec = tuple(float(v) for v in ANOMALY_EVIDENCE_OFFSETS_SEC)
    qwen_offsets_sec = tuple(float(v) for v in ANOMALY_QWEN_OFFSETS_SEC)
    motion_check_offsets_sec = tuple(float(v) for v in ANOMALY_REAR_END_MOTION_CHECK_OFFSETS_SEC)
    max_buffer_seconds = max(
        abs(min([0.0, *evidence_offsets_sec, *qwen_offsets_sec])),
        max([0.0, *evidence_offsets_sec, *qwen_offsets_sec, *motion_check_offsets_sec]),
    )
    frame_buffer = deque(maxlen=max(8, int(max_buffer_seconds * src_fps / stride) + 12))
    raw_frame_buffer = deque(maxlen=max(8, int(max_buffer_seconds * src_fps / stride) + 12))
    tracks_buffer = deque(maxlen=max(8, int(max_buffer_seconds * src_fps / stride) + 12))
    pending_rear_end_reviews = []
    first_processed_frame_idx = None
    # 网页端可为单个任务覆盖千问开关和 key；命令行仍沿用全局环境变量。
    qwen_enabled_flag = bool(QWEN_REVIEW_ENABLED) if qwen_review_enabled is None else bool(qwen_review_enabled)
    qwen_key_value = str(qwen_api_key or "").strip()
    qwen_review_active = bool(qwen_enabled_flag) and bool(
        qwen_key_value
        or os.environ.get(QWEN_API_KEY_ENV, "").strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
    )

    def _ensure_qwen_review_fields(event: dict) -> None:
        """确保千问复核字段存在，避免 CSV/页面读取时出现缺列。"""
        event.setdefault("qwen_image_paths", "[]")
        event.setdefault("qwen_result", "")
        event.setdefault("qwen_reason", "")
        event.setdefault("qwen_error", "")

    def _append_qwen_review_report(event: dict, review: dict, input_image_paths) -> None:
        """把 main.py 收到的千问复核输入输出独立落盘，便于排查报告链路。"""
        try:
            os.makedirs(output_base_dir, exist_ok=True)
            record = {
                "event_frame_id": event.get("frame_id", ""),
                "event_timestamp": event.get("timestamp", ""),
                "event_type": event.get("event_type", ""),
                "track_id": event.get("track_id", ""),
                "related_track_id": event.get("related_track_id", ""),
                "input_image_paths": list(input_image_paths or []),
                "qwen_result": review.get("rear_end_risk", ""),
                "qwen_reason": review.get("reason", ""),
                "qwen_error": review.get("error", ""),
                "qwen_image_paths": review.get("image_paths", []),
                "qwen_raw_response": review.get("raw_response", ""),
                "anomalies_csv": path.abspath(anomalies_csv),
                "llm_payload_json": path.abspath(llm_payload_json),
            }
            with open(qwen_review_report_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"Qwen review report write failed: {exc}")

    def _log_qwen_skip(event: dict, reason: str, error: str = "", input_image_paths=None) -> None:
        """记录千问跳过原因，不能影响主流程。"""
        _ensure_qwen_review_fields(event)
        review = {
            "rear_end_risk": event.get("qwen_result", ""),
            "reason": reason,
            "error": error,
            "image_paths": list(input_image_paths or []),
            "raw_response": "",
        }
        _append_qwen_review_report(event, review, input_image_paths or [])
        print(
            "Qwen review skipped: "
            f"frame={event.get('frame_id')} "
            f"track={event.get('track_id')} "
            f"related={event.get('related_track_id')} "
            f"reason={reason} "
            f"error={error}"
        )

    def _nearest_processed_frame(target_frame: int) -> int:
        target_frame = max(0, int(target_frame))
        return max(0, int(round(target_frame / max(1, stride))) * max(1, stride))

    def _evidence_target_frame(event_frame: int, offset_sec: float, final_frame: int = None) -> int:
        """计算证据截图帧号，处理视频开头/结尾不足 3 秒的兜底。"""
        target = _nearest_processed_frame(int(event_frame) + int(round(float(offset_sec) * src_fps)))
        if float(offset_sec) <= -3.0 and first_processed_frame_idx is not None:
            target = max(int(first_processed_frame_idx), target)
        if float(offset_sec) >= 3.0 and final_frame is not None:
            target = min(int(final_frame), target)
        return target

    def _motion_check_targets(event_frame: int, final_frame: int = None):
        targets = []
        for offset_sec in motion_check_offsets_sec:
            if 0.0 < source_duration_sec < float(offset_sec):
                continue
            targets.append((float(offset_sec), _evidence_target_frame(event_frame, float(offset_sec), final_frame)))
        if final_frame is not None and targets:
            final_offset = round((float(final_frame) - float(event_frame)) / max(1e-6, src_fps), 3)
            if final_offset > targets[-1][0] and int(final_frame) != int(targets[-1][1]):
                targets.append((float(final_offset), int(final_frame)))
        if len(targets) < 2:
            fallback_offsets = (3.0, 5.0, 8.0)
            existing_frames = {int(frame_no) for _, frame_no in targets}
            for offset_sec in fallback_offsets:
                if 0.0 < source_duration_sec < float(offset_sec):
                    continue
                frame_no = _evidence_target_frame(event_frame, float(offset_sec), final_frame)
                if int(frame_no) in existing_frames:
                    continue
                targets.append((float(offset_sec), frame_no))
                existing_frames.add(int(frame_no))
                if len(targets) >= 2:
                    break
        return sorted(targets, key=lambda item: float(item[0]))

    def _frame_from_buffer(target_frame: int):
        for buffered_frame_id, buffered_frame in frame_buffer:
            if int(buffered_frame_id) == int(target_frame):
                return buffered_frame
        return None

    def _raw_frame_from_buffer(target_frame: int):
        for buffered_frame_id, buffered_frame in raw_frame_buffer:
            if int(buffered_frame_id) == int(target_frame):
                return buffered_frame
        return None

    def _tracks_from_buffer(target_frame: int):
        for buffered_frame_id, buffered_tracks in tracks_buffer:
            if int(buffered_frame_id) == int(target_frame):
                return buffered_tracks
        return None

    def _event_pair_ids(event: dict):
        try:
            return [int(event["track_id"]), int(event["related_track_id"])]
        except (TypeError, ValueError):
            return []

    def _bbox_center_float(bbox_ltrb):
        l, t, r, b = (float(v) for v in bbox_ltrb)
        return 0.5 * (l + r), 0.5 * (t + b)

    def _bbox_iou(a, b) -> float:
        al, at, ar, ab = (float(v) for v in a)
        bl, bt, br, bb = (float(v) for v in b)
        inter_w = max(0.0, min(ar, br) - max(al, bl))
        inter_h = max(0.0, min(ab, bb) - max(at, bt))
        inter = inter_w * inter_h
        area_a = max(1.0, (ar - al) * (ab - at))
        area_b = max(1.0, (br - bl) * (bb - bt))
        return inter / max(1.0, area_a + area_b - inter)

    def _bbox_diag(bbox_ltrb) -> float:
        l, t, r, b = (float(v) for v in bbox_ltrb)
        return float(((r - l) ** 2 + (b - t) ** 2) ** 0.5)

    def _pair_bboxes_from_tracks(event: dict, tracks_like) -> dict:
        pair_ids = set(_event_pair_ids(event))
        bboxes = {}
        for track_id, bbox_ltrb in tracks_like or []:
            tid = int(track_id)
            if tid in pair_ids:
                bboxes[tid] = tuple(int(v) for v in bbox_ltrb)
        return bboxes

    def _match_pair_bboxes_at_frame(event: dict, target_frame: int, base_pair_bboxes: dict = None):
        pair_ids = _event_pair_ids(event)
        if not pair_ids:
            return None
        tracks_at_frame = _tracks_from_buffer(target_frame)
        if tracks_at_frame is None:
            return None
        current = [(int(track_id), tuple(int(v) for v in bbox)) for track_id, bbox in tracks_at_frame]
        matched = {}
        used_indices = set()
        for pair_id in pair_ids:
            for idx, (track_id, bbox) in enumerate(current):
                if track_id == pair_id:
                    matched[pair_id] = bbox
                    used_indices.add(idx)
                    break
        if len(matched) == len(pair_ids):
            return matched
        base_pair_bboxes = base_pair_bboxes or _pair_bboxes_from_tracks(event, _tracks_from_buffer(int(event["frame_id"])))
        for pair_id in pair_ids:
            if pair_id in matched:
                continue
            base_bbox = base_pair_bboxes.get(pair_id)
            if base_bbox is None:
                return None
            base_cx, base_cy = _bbox_center_float(base_bbox)
            base_area = max(1.0, float((base_bbox[2] - base_bbox[0]) * (base_bbox[3] - base_bbox[1])))
            max_center_dist = max(24.0, min(70.0, _bbox_diag(base_bbox) * 0.45))
            best = None
            for idx, (_track_id, bbox) in enumerate(current):
                if idx in used_indices:
                    continue
                cx, cy = _bbox_center_float(bbox)
                dist = float(((cx - base_cx) ** 2 + (cy - base_cy) ** 2) ** 0.5)
                area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
                area_ratio = area / base_area
                iou = _bbox_iou(base_bbox, bbox)
                if not (0.45 <= area_ratio <= 2.2):
                    continue
                if iou < 0.18 and dist > max_center_dist:
                    continue
                score = (iou, -dist)
                if best is None or score > best[0]:
                    best = (score, idx, bbox)
            if best is None:
                return None
            _, idx, bbox = best
            matched[pair_id] = bbox
            used_indices.add(idx)
        return matched if len(matched) == len(pair_ids) else None

    def _full_frame_for_qwen(raw_frame, event: dict, tracks_at_frame, target_frame: int = None):
        """生成千问用完整画面，只用矩形框标出需要复核的两辆车。"""
        if raw_frame is None:
            return None
        full_frame = raw_frame.copy()
        pair_ids = _event_pair_ids(event)
        scale = draw_scale(full_frame)
        thickness = max(2, int(round(4 * scale)))
        label_scale = max(0.32, 0.62 * scale)
        label_thickness = max(1, int(round(2 * scale)))
        colors = [(255, 0, 255), (0, 255, 255)]
        matched_bboxes = None
        if target_frame is not None:
            matched_bboxes = _match_pair_bboxes_at_frame(event, int(target_frame), event.get("_base_pair_bboxes"))
        if matched_bboxes is None:
            matched_bboxes = _pair_bboxes_from_tracks(event, tracks_at_frame)
        for matched, track_id in enumerate(pair_ids):
            bbox_ltrb = matched_bboxes.get(int(track_id)) if matched_bboxes else None
            if bbox_ltrb is None:
                continue
            l, t, r, b = (int(v) for v in bbox_ltrb)
            color = colors[matched % len(colors)]
            cv2.rectangle(full_frame, (l, t), (r, b), color, thickness)
            cv2.putText(
                full_frame,
                f"TARGET {matched + 1}",
                (l, max(16, t - 8)),
                cv2.FONT_HERSHEY_DUPLEX,
                label_scale,
                color,
                label_thickness,
                lineType=cv2.LINE_AA,
            )
        h, w = full_frame.shape[:2]
        max_size = int(ANOMALY_QWEN_FULL_IMAGE_MAX_SIZE_PX)
        if max(w, h) > max_size:
            resize_scale = float(max_size) / float(max(w, h))
            full_frame = cv2.resize(
                full_frame,
                (max(1, int(w * resize_scale)), max(1, int(h * resize_scale))),
                interpolation=cv2.INTER_AREA,
            )
        return full_frame

    def _pair_present_at_frame(event: dict, target_frame: int) -> bool:
        return _match_pair_bboxes_at_frame(event, target_frame, event.get("_base_pair_bboxes")) is not None

    def _pair_centers_at_frame(event: dict, target_frame: int):
        bboxes = _match_pair_bboxes_at_frame(event, target_frame, event.get("_base_pair_bboxes"))
        if bboxes is None:
            return None
        return {tid: _bbox_center_float(bbox) for tid, bbox in bboxes.items()}

    def _max_pair_move_px(base_centers: dict, check_centers: dict) -> float:
        moves = []
        for tid, base in base_centers.items():
            check = check_centers.get(tid)
            if check is None:
                return float("inf")
            dx = float(check[0] - base[0])
            dy = float(check[1] - base[1])
            moves.append(float((dx * dx + dy * dy) ** 0.5))
        return max(moves) if moves else float("inf")

    def _post_motion_review(event: dict, check_frames) -> tuple[bool, str]:
        observations = []
        seen_frames = set()
        for offset_sec, target_frame in sorted(check_frames or [], key=lambda item: float(item[0])):
            frame_no = int(target_frame)
            if frame_no in seen_frames:
                continue
            seen_frames.add(frame_no)
            centers = _pair_centers_at_frame(event, frame_no)
            if centers is None:
                return False, f"规则触发后约{float(offset_sec):.0f}s目标框已丢失，按正常离开处理，跳过千问复核"
            observations.append((float(offset_sec), frame_no, centers))
        if len(observations) < 2:
            return True, "后续观察点不足，已跳过位移复核"

        start_offset, _start_frame, start_centers = observations[-2]
        end_offset, _end_frame, end_centers = observations[-1]
        move_px = _max_pair_move_px(start_centers, end_centers)
        detail = f"+{start_offset:.0f}s到+{end_offset:.0f}s位移{move_px:.1f}px"
        if move_px > float(ANOMALY_REAR_END_MAX_POST_MOVE_PX):
            return False, f"规则触发后目标在后续观察点之间仍有明显位移（{detail}），未达到持续静止条件，跳过千问复核"
        return True, f"后续位移检查通过：{detail}"

    def _qwen_evidence_metadata(event: dict, review_item: dict, ordered_indices) -> list:
        metadata = []
        original_targets = review_item.get("original_qwen_targets") or review_item.get("qwen_targets") or []
        current_targets = review_item.get("qwen_targets") or []
        final_indices = set(review_item.get("used_final_frame_indices") or set())
        event_frame = int(event.get("frame_id") or 0)
        for idx in ordered_indices:
            planned_offset = float(original_targets[idx - 1][0]) if idx - 1 < len(original_targets) else ""
            target_frame = int(current_targets[idx - 1][1]) if idx - 1 < len(current_targets) else event_frame
            actual_offset = round((float(target_frame) - float(event_frame)) / max(1e-6, src_fps), 3)
            timestamp_sec = round(float(target_frame) / max(1e-6, src_fps), 3)
            if planned_offset == "":
                label = "后续证据"
            elif abs(float(planned_offset)) < 1e-6:
                label = "触发时刻"
            elif idx in final_indices:
                label = "后续证据（视频最后一帧替代）"
            else:
                label = f"触发后约{float(planned_offset):.0f}秒"
            metadata.append(
                {
                    "label": label,
                    "planned_offset_sec": planned_offset,
                    "actual_offset_sec": actual_offset,
                    "timestamp_sec": timestamp_sec,
                    "used_final_frame": idx in final_indices,
                }
            )
        return metadata

    def _mark_review_false(event: dict, reason: str, review_item: dict = None) -> None:
        event["qwen_result"] = "false"
        event["qwen_reason"] = reason
        event["qwen_error"] = ""
        _ensure_qwen_review_fields(event)
        if review_item is not None:
            review_item["skip_qwen"] = True
        input_image_paths = []
        if review_item is not None:
            qwen_paths = review_item.get("qwen_paths") or {}
            ordered_qwen_paths = [qwen_paths[idx] for idx in sorted(qwen_paths)]
            input_image_paths = [path.join(path.dirname(path.abspath(anomalies_csv)), p) for p in ordered_qwen_paths]
        _log_qwen_skip(event, reason, input_image_paths=input_image_paths)
        if rear_end_review_callback is not None:
            try:
                rear_end_review_callback(dict(event))
            except Exception:
                pass
        if anomaly_detector is not None:
            try:
                anomaly_detector.export_csv(path.abspath(anomalies_csv))
            except Exception:
                pass

    def _finalize_partial_rear_end_evidence(review_item: dict, note: str) -> bool:
        """视频结束时尽量保留已生成的证据图，避免结果页完全无图。"""
        event = review_item["event"]
        saved_paths = review_item.get("saved_paths") or {}
        qwen_paths = review_item.get("qwen_paths") or {}
        if not saved_paths:
            return False
        ordered_indices = sorted(saved_paths)
        ordered_paths = [saved_paths[idx] for idx in ordered_indices]
        ordered_qwen_paths = [qwen_paths[idx] for idx in sorted(qwen_paths)]
        evidence_targets = review_item["evidence_targets"]
        ordered_offsets = [float(evidence_targets[idx - 1][0]) for idx in ordered_indices]
        ordered_times = [
            round(float(evidence_targets[idx - 1][1]) / max(1e-6, src_fps), 3)
            for idx in ordered_indices
        ]
        event["evidence_path"] = ordered_paths[min(2, len(ordered_paths) - 1)]
        event["evidence_paths"] = json.dumps(ordered_paths, ensure_ascii=False)
        event["qwen_image_paths"] = json.dumps(ordered_qwen_paths, ensure_ascii=False)
        event["evidence_offsets_sec"] = json.dumps(ordered_offsets, ensure_ascii=False)
        event["evidence_times_sec"] = json.dumps(ordered_times, ensure_ascii=False)
        motion_ok, motion_reason = _post_motion_review(event, review_item.get("motion_check_frames") or [])
        qwen_input_complete = len(ordered_qwen_paths) >= len(review_item["qwen_targets"])
        skip_reason = ""
        if not motion_ok:
            skip_reason = motion_reason
        elif not qwen_input_complete:
            skip_reason = "千问输入截图生成不完整，已跳过复核"
        elif motion_reason:
            note = f"{note}；{motion_reason}"
        event["qwen_result"] = "false" if skip_reason else ""
        event["qwen_reason"] = skip_reason or note
        if skip_reason:
            event["qwen_error"] = ""
            _ensure_qwen_review_fields(event)
            _log_qwen_skip(
                event,
                event["qwen_reason"],
                "",
                [path.join(path.dirname(path.abspath(anomalies_csv)), p) for p in ordered_qwen_paths],
            )
        else:
            event["qwen_error"] = ""
            _ensure_qwen_review_fields(event)
        if rear_end_review_callback is not None:
            try:
                rear_end_review_callback(dict(event))
            except Exception:
                pass
        return True

    def _save_rear_end_evidence(review_item: dict) -> bool:
        nonlocal anomaly_snapshot_count
        event = review_item["event"]
        saved_paths = review_item["saved_paths"]
        qwen_paths = review_item["qwen_paths"]
        if review_item.get("skip_qwen"):
            return True
        exit_check_frame = review_item.get("exit_check_frame")
        if exit_check_frame is not None and int(frame_idx) >= int(exit_check_frame):
            if not _pair_present_at_frame(event, int(exit_check_frame)):
                if saved_paths and not review_item.get("evidence_finalized"):
                    ordered_saved_indices = sorted(saved_paths)
                    ordered_paths = [saved_paths[idx] for idx in ordered_saved_indices]
                    evidence_targets = review_item["evidence_targets"]
                    ordered_offsets = [float(evidence_targets[idx - 1][0]) for idx in ordered_saved_indices]
                    ordered_times = [
                        round(float(evidence_targets[idx - 1][1]) / max(1e-6, src_fps), 3)
                        for idx in ordered_saved_indices
                    ]
                    event["evidence_path"] = ordered_paths[2] if len(ordered_paths) >= 3 else ordered_paths[0]
                    event["evidence_paths"] = json.dumps(ordered_paths, ensure_ascii=False)
                    event["evidence_offsets_sec"] = json.dumps(ordered_offsets, ensure_ascii=False)
                    event["evidence_times_sec"] = json.dumps(ordered_times, ensure_ascii=False)
                    _ensure_qwen_review_fields(event)
                    anomaly_snapshot_count += 1
                    review_item["evidence_finalized"] = True
                _mark_review_false(event, "规则触发后约8秒目标框已丢失，按正常离开处理，跳过千问复核", review_item)
                return True
        os.makedirs(anomaly_snapshot_dir, exist_ok=True)
        os.makedirs(qwen_snapshot_dir, exist_ok=True)
        for idx, (offset_sec, target_frame) in enumerate(review_item["evidence_targets"], start=1):
            if idx in saved_paths:
                continue
            if int(frame_idx) < int(target_frame):
                continue
            evidence_frame = _frame_from_buffer(target_frame)
            if evidence_frame is None:
                continue
            related_id = event.get("related_track_id") or "none"
            offset_label = f"{offset_sec:+.0f}s".replace("+", "p").replace("-", "m")
            filename = (
                f"rear_end_risk_event_{int(event['frame_id'])}_"
                f"{int(event['track_id'])}_{related_id}_{idx}_{offset_label}.jpg"
            )
            snapshot_path = path.join(anomaly_snapshot_dir, filename)
            if cv2.imwrite(snapshot_path, evidence_frame):
                saved_paths[idx] = path.relpath(snapshot_path, path.dirname(path.abspath(anomalies_csv)))
        for idx, (offset_sec, target_frame) in enumerate(review_item["qwen_targets"], start=1):
            if idx in qwen_paths:
                continue
            if int(frame_idx) < int(target_frame):
                continue
            raw_evidence_frame = _raw_frame_from_buffer(target_frame)
            tracks_at_frame = _tracks_from_buffer(target_frame)
            qwen_frame = _full_frame_for_qwen(raw_evidence_frame, event, tracks_at_frame, target_frame)
            if qwen_frame is not None:
                related_id = event.get("related_track_id") or "none"
                offset_label = f"{offset_sec:+.0f}s".replace("+", "p").replace("-", "m")
                qwen_filename = (
                    f"rear_end_risk_event_{int(event['frame_id'])}_"
                    f"{int(event['track_id'])}_{related_id}_qwen{idx}_{offset_label}_full.jpg"
                )
                qwen_path = path.join(qwen_snapshot_dir, qwen_filename)
                if cv2.imwrite(
                    qwen_path,
                    qwen_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(QWEN_IMAGE_JPEG_QUALITY)],
                ):
                    qwen_paths[idx] = path.relpath(qwen_path, path.dirname(path.abspath(anomalies_csv)))
        if len(saved_paths) < len(review_item["evidence_targets"]):
            return False
        if len(qwen_paths) < len(review_item["qwen_targets"]):
            return False

        ordered_paths = [saved_paths[idx] for idx in sorted(saved_paths)]
        ordered_qwen_paths = [qwen_paths[idx] for idx in sorted(qwen_paths)]
        ordered_offsets = [float(review_item["evidence_targets"][idx - 1][0]) for idx in sorted(saved_paths)]
        ordered_times = [
            round(float(review_item["evidence_targets"][idx - 1][1]) / max(1e-6, src_fps), 3)
            for idx in sorted(saved_paths)
        ]
        event["evidence_path"] = ordered_paths[2] if len(ordered_paths) >= 3 else ordered_paths[0]
        event["evidence_paths"] = json.dumps(ordered_paths, ensure_ascii=False)
        event["qwen_image_paths"] = json.dumps(ordered_qwen_paths, ensure_ascii=False)
        event["evidence_offsets_sec"] = json.dumps(ordered_offsets, ensure_ascii=False)
        event["evidence_times_sec"] = json.dumps(ordered_times, ensure_ascii=False)
        _ensure_qwen_review_fields(event)
        final_frame_note = "视频在后续证据点前结束，已使用最后一帧作为后续证据"
        anomaly_snapshot_count += 1
        motion_ok, motion_reason = _post_motion_review(event, review_item.get("motion_check_frames") or [])
        if not motion_ok:
            _mark_review_false(event, motion_reason, review_item)
            return True
        if qwen_review_active:
            ordered_qwen_indices = sorted(qwen_paths)
            abs_paths = [path.join(path.dirname(path.abspath(anomalies_csv)), p) for p in ordered_qwen_paths]
            evidence_metadata = _qwen_evidence_metadata(event, review_item, ordered_qwen_indices)
            review = review_rear_end_with_qwen(
                image_paths=abs_paths,
                api_key_env=QWEN_API_KEY_ENV,
                api_url=QWEN_API_URL,
                model=QWEN_MODEL,
                timeout_sec=QWEN_TIMEOUT_SEC,
                max_prompt_chars=QWEN_MAX_PROMPT_CHARS,
                evidence_metadata=evidence_metadata,
                api_key_value=qwen_key_value,
            )
            _append_qwen_review_report(event, review, abs_paths)
            print(
                "Qwen review finished: "
                f"frame={event.get('frame_id')} "
                f"track={event.get('track_id')} "
                f"related={event.get('related_track_id')} "
                f"result={review.get('rear_end_risk', '')} "
                f"error={review.get('error', '')} "
                f"report={qwen_review_report_jsonl}"
            )
            result = review.get("rear_end_risk", "")
            event["qwen_result"] = "" if result == "" else str(bool(result)).lower()
            event["qwen_reason"] = str(review.get("reason") or "")
            event["qwen_error"] = str(review.get("error") or "")
            if review.get("debug_report_path") and event["qwen_error"]:
                event["qwen_error"] = f"{event['qwen_error']}；debug={review.get('debug_report_path')}"
            _ensure_qwen_review_fields(event)
            if event["qwen_result"] == "true":
                event["description"] = f"{event['description']}；千问复核结果：true，{event['qwen_reason']}"
            if rear_end_review_callback is not None:
                try:
                    rear_end_review_callback(dict(event))
                except Exception:
                    pass
        else:
            event["qwen_result"] = ""
            event["qwen_reason"] = "千问复核已禁用，当前显示规则检测结果"
            if review_item.get("used_final_frame"):
                event["qwen_reason"] = f"{event['qwen_reason']}；{final_frame_note}"
            if motion_reason:
                event["qwen_reason"] = f"{event['qwen_reason']}；{motion_reason}"
            event["qwen_error"] = ""
            _ensure_qwen_review_fields(event)
            _log_qwen_skip(
                event,
                event["qwen_reason"],
                "",
                [path.join(path.dirname(path.abspath(anomalies_csv)), p) for p in ordered_qwen_paths],
            )
            if rear_end_review_callback is not None:
                try:
                    rear_end_review_callback(dict(event))
                except Exception:
                    pass
        if anomaly_detector is not None:
            try:
                anomaly_detector.export_csv(path.abspath(anomalies_csv))
            except Exception:
                pass
        return True

    # 单任务处理循环。循环内持续轮询运行时控制，网页端可在不中断任务的
    # 情况下暂停/继续处理或修改高亮 track_id。
    frame_i = 0
    frame_idx = -1
    last_processed_frame_idx = None
    time_taken = 0
    last_vis_frame = None
    trail_len = max(1, int(trail_len))
    lost_hold = max(1, int(lost_hold))
    selected_track_points = defaultdict(lambda: deque(maxlen=trail_len))
    # 选中高亮按逻辑车辆对象维护，避免 ByteTrack 短时换 ID 后高亮断开。
    selected_lost_frames = {}
    selected_ids_runtime = set()
    desired_highlight_raw_ids = set()
    pending_selected_raw_ids = set()
    if selected_id is not None:
        desired_highlight_raw_ids.add(int(selected_id))
        pending_selected_raw_ids.add(int(selected_id))
    if highlight_ids:
        for tid in highlight_ids:
            desired_highlight_raw_ids.add(int(tid))
            pending_selected_raw_ids.add(int(tid))
    trail_mode = str(trail_mode or "all").lower()
    if trail_mode not in {"none", "all", "selected"}:
        trail_mode = "all"
    latest_tracks = []
    track_records = []
    selected_track_records = []
    track_class_cache = {}
    # 轨迹显示专用：raw id -> logical id 映射，降低短暂丢失后的轨迹断裂。
    prev_raw_to_logical = {}
    logical_last_center = {}
    logical_last_frame = {}
    logical_class = {}
    draw_track_history = defaultdict(lambda: deque(maxlen=trail_len))
    rear_end_risk_logical_ids = set()

    def apply_runtime_controls() -> None:
        nonlocal trail_mode, trail_len, selected_track_points, draw_track_history
        nonlocal desired_highlight_raw_ids, pending_selected_raw_ids
        if runtime_control_callback is None:
            return
        try:
            control = runtime_control_callback() or {}
        except Exception:
            return

        if bool(control.get("cancel", False)):
            raise AnalysisCancelled("Analysis task cancelled")

        while bool(control.get("paused", False)):
            sleep(0.1)
            try:
                control = runtime_control_callback() or {}
            except Exception:
                control = {"paused": False}
            if bool(control.get("cancel", False)):
                raise AnalysisCancelled("Analysis task cancelled")

        new_mode = str(control.get("trail_mode", trail_mode) or trail_mode).lower()
        if new_mode in {"none", "all", "selected"}:
            trail_mode = new_mode

        try:
            new_len = max(1, int(control.get("trail_len", trail_len)))
        except (TypeError, ValueError):
            new_len = trail_len
        if new_len != trail_len:
            trail_len = new_len
            selected_track_points = defaultdict(
                lambda: deque(maxlen=trail_len),
                {tid: deque(points, maxlen=trail_len) for tid, points in selected_track_points.items()},
            )
            draw_track_history = defaultdict(
                lambda: deque(maxlen=trail_len),
                {tid: deque(points, maxlen=trail_len) for tid, points in draw_track_history.items()},
            )

        if "highlight_ids" in control:
            try:
                desired_ids = {int(tid) for tid in control.get("highlight_ids", set())}
            except (TypeError, ValueError):
                desired_ids = set()
            if desired_ids != desired_highlight_raw_ids:
                desired_highlight_raw_ids = set(desired_ids)
                pending_selected_raw_ids = set(desired_ids)
                selected_ids_runtime.clear()
                selected_lost_frames.clear()
                selected_track_points.clear()

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            start = perf_counter()
            apply_runtime_controls()
            if show:
                # 先轮询一次按键/窗口事件，降低 UI 交互延迟。
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("<< User has terminated the process >>")
                    break

            # 读取输入视频帧
            ret, frame = video.read()
            if not ret:
                break
            raw_frame = frame.copy()
            frame_idx += 1
            apply_runtime_controls()

            # 跳帧处理：保持显示响应，同时只在每 stride 帧运行一次模型。
            if frame_idx % stride != 0:
                if show:
                    cv2.imshow(WINDOW_NAME, last_vis_frame if last_vis_frame is not None else frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("<< User has terminated the process >>")
                        break
                continue

            # YOLO11 在采样帧上检测目标，ByteTrack 负责关联稳定 track_id。
            tracks, detections = tracker.track_frame(frame)
            last_processed_frame_idx = int(frame_idx)
            if first_processed_frame_idx is None:
                first_processed_frame_idx = int(frame_idx)
            latest_tracks = tracks
            if tracking_callback is not None:
                try:
                    tracking_callback(tracks, frame.shape)
                except Exception:
                    # 网页点击选择是附加能力，即使保存最新 tracks 失败也不能影响分析主流程。
                    pass
            n_objects = int(detections.shape[0]) if detections is not None else 0
            track_class_map = build_track_class_map(tracks, detections, track_class_cache)
            current_raw_to_logical = {}
            used_logical_ids = set()

            # 记录所有轨迹中心点，供 CSV 导出、轨迹分析和运行时高亮使用。
            active_ids = set()
            tracked_objects = []
            for track_id, bbox_ltrb in tracks:
                active_ids.add(int(track_id))
                cx, cy = bbox_center_ltrb(bbox_ltrb)
                cls_id = track_class_map.get(int(track_id))
                cls_name = CLASS_NAME.get(int(cls_id), "unknown") if cls_id is not None else "unknown"
                timestamp = float(frame_idx) / max(1e-6, src_fps)
                tracked_objects.append(
                    {
                        "track_id": int(track_id),
                        "bbox": tuple(int(v) for v in bbox_ltrb),
                        "center": (float(cx), float(cy)),
                        "class_name": cls_name,
                        "class_id": cls_id,
                    }
                )
                logical_tid = remap_track_for_trail(
                    raw_tid=int(track_id),
                    cls_id=cls_id,
                    center_xy=(float(cx), float(cy)),
                    frame_idx=int(frame_idx),
                    prev_raw_to_logical=prev_raw_to_logical,
                    logical_last_center=logical_last_center,
                    logical_last_frame=logical_last_frame,
                    logical_class=logical_class,
                    used_logical_ids=used_logical_ids,
                    frame_shape=frame.shape,
                )
                current_raw_to_logical[int(track_id)] = int(logical_tid)
                used_logical_ids.add(int(logical_tid))
                if int(track_id) in pending_selected_raw_ids:
                    selected_ids_runtime.add(int(logical_tid))
                    selected_lost_frames[int(logical_tid)] = 0
                    pending_selected_raw_ids.discard(int(track_id))
                draw_track_history[int(logical_tid)].append((float(cx), float(cy)))
                logical_last_center[int(logical_tid)] = (float(cx), float(cy))
                logical_last_frame[int(logical_tid)] = int(frame_idx)
                logical_class[int(logical_tid)] = int(cls_id) if cls_id is not None else None
                row = {
                    "frame_id": int(frame_idx),
                    "track_id": int(track_id),
                    "class_name": cls_name,
                    "cx": int(cx),
                    "cy": int(cy),
                    "timestamp": timestamp,
                }
                track_records.append(row)
                trajectory_analyzer.update(
                    track_id=int(track_id),
                    class_name=cls_name,
                    center=(float(cx), float(cy)),
                    timestamp=timestamp,
                )
                if int(logical_tid) in selected_ids_runtime:
                    selected_track_points[int(logical_tid)].append((cx, cy))
                    selected_lost_frames[int(logical_tid)] = 0
                    selected_track_records.append(row)
            prev_raw_to_logical = current_raw_to_logical

            # 车流与测速分析。
            timestamp = float(frame_idx) / max(1e-6, src_fps)
            analytics.update(
                frame_id=frame_idx,
                timestamp=timestamp,
                detections=detections,
                tracks=tracks,
                track_class_map=track_class_map,
            )
            if stats_callback is not None:
                try:
                    measured_n, avg_speed, max_speed = analytics.speed_summary()
                    stats_callback(
                        {
                            "frame_id": int(frame_idx),
                            "timestamp": float(timestamp),
                            "total_frames": int(total_source_frames),
                            "source_duration_sec": float(source_duration_sec),
                            "progress_percent": (
                                min(100.0, max(0.0, (float(frame_idx + 1) / float(total_source_frames)) * 100.0))
                                if total_source_frames > 0
                                else None
                            ),
                            "flow_total": int(sum(analytics.line_counts.values())),
                            "avg_speed_kmh": float(avg_speed),
                            "max_speed_kmh": float(max_speed),
                            "measured_count": int(measured_n),
                        }
                    )
                except Exception:
                    pass
            anomaly_events = []
            if anomaly_detector is not None:
                anomaly_events = anomaly_detector.update(
                    frame_id=frame_idx,
                    timestamp=timestamp,
                    tracked_objects=tracked_objects,
                )
                for event in anomaly_events:
                    if event.get("event_type") != "rear_end_risk":
                        continue
                    try:
                        event_ids = [int(event["track_id"])]
                        if event.get("related_track_id") != "":
                            event_ids.append(int(event["related_track_id"]))
                        for raw_tid in event_ids:
                            rear_end_risk_logical_ids.add(int(current_raw_to_logical.get(raw_tid, raw_tid)))
                    except (TypeError, ValueError):
                        pass

            # 选中目标丢失状态管理：每个目标独立计数，超时后自动移除。
            lost_remove_ids = []
            for tid in list(selected_ids_runtime):
                if tid in used_logical_ids:
                    continue
                selected_lost_frames[tid] = selected_lost_frames.get(tid, 0) + 1
                if selected_lost_frames[tid] > lost_hold:
                    lost_remove_ids.append(tid)
            for tid in lost_remove_ids:
                selected_ids_runtime.remove(tid)
                selected_lost_frames.pop(tid, None)
                selected_track_points.pop(tid, None)
                print(f"<< Target Auto-Removed: ID {tid} (lost > {lost_hold} frames) >>")

            # 绘制叠加信息并写入输出视频，保持 output.mp4 生成逻辑不变。
            selected_raw_ids_for_draw = {
                int(raw_tid)
                for raw_tid, logical_tid in current_raw_to_logical.items()
                if int(logical_tid) in selected_ids_runtime
            }
            rear_end_raw_ids_for_draw = {
                int(raw_tid)
                for raw_tid, logical_tid in current_raw_to_logical.items()
                if int(logical_tid) in rear_end_risk_logical_ids
            }
            draw_track_boxes(
                frame,
                tracks,
                selected_ids=selected_raw_ids_for_draw,
                rear_end_ids=rear_end_raw_ids_for_draw,
                only_selected=only_selected,
                speed_map=analytics.track_speed_kmh,
                track_class_map=track_class_map,
            )
            if trail_mode == "all":
                draw_track_trails(frame, draw_track_history)
                draw_selected_trails(frame, selected_track_points)
            elif trail_mode == "selected":
                draw_selected_trails(frame, selected_track_points)
            draw_traffic_status(frame, analytics)
            if show_guides:
                draw_analysis_guides(frame, line1_xyxy, line2_xyxy, count_line_xyxy)
            if anomaly_events:
                for event in anomaly_events:
                    if int(ANOMALY_MAX_SNAPSHOTS) > 0 and anomaly_snapshot_count >= int(ANOMALY_MAX_SNAPSHOTS):
                        continue
                    if event.get("event_type") != "rear_end_risk":
                        continue
                    evidence_targets = []
                    for offset_sec in evidence_offsets_sec:
                        target = _evidence_target_frame(int(event["frame_id"]), float(offset_sec))
                        evidence_targets.append((float(offset_sec), target))
                    qwen_targets = []
                    for offset_sec in qwen_offsets_sec:
                        target = _evidence_target_frame(int(event["frame_id"]), float(offset_sec))
                        qwen_targets.append((float(offset_sec), target))
                    exit_check_frame = None
                    if not (0.0 < source_duration_sec < float(ANOMALY_QWEN_EXIT_CHECK_SEC)):
                        exit_check_frame = _evidence_target_frame(
                            int(event["frame_id"]),
                            float(ANOMALY_QWEN_EXIT_CHECK_SEC),
                        )
                    motion_check_frames = _motion_check_targets(int(event["frame_id"]))
                    pending_rear_end_reviews.append(
                        {
                            "event": event,
                            "evidence_targets": evidence_targets,
                            "qwen_targets": qwen_targets,
                            "original_qwen_targets": list(qwen_targets),
                            "exit_check_frame": exit_check_frame,
                            "motion_check_frames": motion_check_frames,
                            "saved_paths": {},
                            "qwen_paths": {},
                        }
                    )
                    event["_base_pair_bboxes"] = _pair_bboxes_from_tracks(event, tracks)
                    if rear_end_review_callback is not None:
                        live_event = dict(event)
                        live_event["qwen_result"] = ""
                        live_event["qwen_reason"] = "规则检测已触发，证据截图正在生成"
                        live_event["qwen_error"] = ""
                        _ensure_qwen_review_fields(live_event)
                        try:
                            rear_end_review_callback(live_event)
                        except Exception:
                            pass
                cv2.putText(
                    frame,
                    f"Anomaly: {anomaly_events[0]['event_type']}",
                    (12, min(frame.shape[0] - 16, 160)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            raw_frame_buffer.append((int(frame_idx), raw_frame))
            frame_buffer.append((int(frame_idx), frame.copy()))
            tracks_buffer.append((int(frame_idx), [(int(tid), tuple(int(v) for v in bbox)) for tid, bbox in tracks]))
            if pending_rear_end_reviews:
                remaining_reviews = []
                for review_item in pending_rear_end_reviews:
                    if _save_rear_end_evidence(review_item):
                        continue
                    remaining_reviews.append(review_item)
                pending_rear_end_reviews = remaining_reviews
            output.write(frame)
            if frame_callback is not None:
                try:
                    frame_callback(frame)
                except Exception:
                    # 网页预览回调失败不应中断分析主流程。
                    pass
            last_vis_frame = frame.copy()

            # 统计单帧耗时，并在桌面模式下显示画面。
            frame_time = perf_counter() - start
            if show:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("<< User has terminated the process >>")
                    break
            time_taken += frame_time
            frame_i += 1
            if log_interval > 0 and frame_i % log_interval == 0:
                print(
                    f"Frame {frame_i}: "
                    f"{n_objects} objects - {int(frame_time*1000)} ms = {1/max(frame_time, 1e-6):.2f} Hz"
                )
    finally:
        video.release()
        output.release()
        if show:
            cv2.destroyAllWindows()

    # 打印处理性能摘要。
    print(
        f"\nTotal frames processed: {frame_i}"
        f"\nVideo processing time: {time_taken:.2f} s"
        f"\nAverage FPS: {frame_i/max(time_taken, 1e-6):.2f} Hz"
    )
    if pending_rear_end_reviews:
        for review_item in pending_rear_end_reviews:
            final_processed_frame = int(last_processed_frame_idx) if last_processed_frame_idx is not None else _nearest_processed_frame(max(0, frame_idx))
            adjusted_evidence_targets = []
            used_final_frame = False
            for offset_sec, target_frame in review_item["evidence_targets"]:
                adjusted_frame = _evidence_target_frame(
                    int(review_item["event"]["frame_id"]),
                    float(offset_sec),
                    final_processed_frame,
                )
                if float(offset_sec) > 0 and int(adjusted_frame) != int(target_frame):
                    used_final_frame = True
                adjusted_evidence_targets.append((float(offset_sec), adjusted_frame))
            review_item["evidence_targets"] = adjusted_evidence_targets
            adjusted_qwen_targets = []
            used_final_frame_indices = set()
            for offset_sec, target_frame in review_item["qwen_targets"]:
                adjusted_frame = _evidence_target_frame(
                    int(review_item["event"]["frame_id"]),
                    float(offset_sec),
                    final_processed_frame,
                )
                if float(offset_sec) > 0 and int(adjusted_frame) != int(target_frame):
                    used_final_frame = True
                    used_final_frame_indices.add(len(adjusted_qwen_targets) + 1)
                adjusted_qwen_targets.append((float(offset_sec), adjusted_frame))
            review_item["qwen_targets"] = adjusted_qwen_targets
            if used_final_frame_indices:
                review_item["used_final_frame_indices"] = used_final_frame_indices
            adjusted_motion_check_frames = _motion_check_targets(
                int(review_item["event"]["frame_id"]),
                final_processed_frame,
            )
            if adjusted_motion_check_frames != (review_item.get("motion_check_frames") or []):
                used_final_frame = True
            review_item["motion_check_frames"] = adjusted_motion_check_frames
            if used_final_frame:
                review_item["used_final_frame"] = True
            if not _save_rear_end_evidence(review_item):
                _finalize_partial_rear_end_evidence(
                    review_item,
                    "视频结束时只生成了部分追尾证据截图，已展示可用截图",
                )
    # 处理结束后导出轨迹 CSV。
    export_tracks_csv(track_records, path.abspath(tracks_csv))
    print(f"Track CSV exported: {path.abspath(tracks_csv)} ({len(track_records)} rows)")
    if selected_track_records:
        export_tracks_csv(selected_track_records, path.abspath(selected_csv))
        print(
            f"Selected track CSV exported: {path.abspath(selected_csv)} "
            f"({len(selected_track_records)} rows)"
        )
    analytics.export_counts_csv(path.abspath(counts_csv))
    analytics.export_events_csv(path.abspath(events_csv))
    analytics.export_traffic_csv(path.abspath(traffic_csv))
    if anomaly_detector is not None:
        anomaly_detector.export_csv(path.abspath(anomalies_csv))
        export_llm_violation_payload(
            anomalies_csv=path.abspath(anomalies_csv),
            output_json=path.abspath(llm_payload_json),
            output_video=path.abspath(save_path),
            source_video=input_vid,
            max_events=LLM_MAX_EVENTS,
        )
    analytics.export_plots(path.abspath(reports_dir))
    trajectory_summary = trajectory_analyzer.summary_rows()
    if trajectory_summary_json:
        with open(path.abspath(trajectory_summary_json), "w", encoding="utf-8") as f:
            json.dump(trajectory_summary, f, ensure_ascii=False, indent=2)
    print(f"Trajectory summary: tracks={len(trajectory_summary)}")
    print(f"Counts CSV exported: {path.abspath(counts_csv)} ({len(analytics.count_rows)} rows)")
    print(f"Events CSV exported: {path.abspath(events_csv)} ({len(analytics.event_rows)} rows)")
    print(f"Traffic CSV exported: {path.abspath(traffic_csv)} ({len(analytics.traffic_rows)} rows)")
    if anomaly_detector is not None:
        print(f"Anomalies CSV exported: {path.abspath(anomalies_csv)} ({len(anomaly_detector.events)} rows)")
        print(f"LLM payload JSON exported: {path.abspath(llm_payload_json)}")
        print(f"Qwen review report JSONL: {path.abspath(qwen_review_report_jsonl)}")
    print(f"Report images exported under: {path.abspath(reports_dir)}")


# 兼容旧脚本中的 track_people 调用；新代码应直接使用 run_surveillance_pipeline。
track_people = run_surveillance_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="YOLO11 + ByteTrack Smart Surveillance",
        description="YOLO11 + ByteTrack video tracking pipeline for smart surveillance.",
    )
    parser.add_argument(
        "--input-vid",
        type=str,
        default="./video.mp4",
        help="path to the input video file for surveillance analysis",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./output/output.mp4",
        help="path to save file the output video",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=YOLO_MODEL,
        help="path to YOLO11 model weights, e.g. yolo11x.pt / yolo11s.pt / yolo11n.pt",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER_CFG,
        help='tracker config, default "bytetrack.yaml"',
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=DEFAULT_YOLO_CONF,
        help="confidence threshold for YOLO11 track mode (ByteTrack usually prefers low conf)",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=DEFAULT_YOLO_IOU,
        help="IoU threshold for YOLO11 NMS in track mode",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="inference image size for YOLO11 track mode",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=DEFAULT_CLASSES,
        help='class preset: person_vehicle | vehicle | person | all, or custom csv like "0,1,2,3,5,7"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='inference device, e.g. "cpu", "0", "0,1"',
    )
    parser.add_argument(
        "--veh-conf",
        type=float,
        default=DEFAULT_VEH_CONF,
        help="vehicle class confidence threshold (1,2,3,5,7)",
    )
    parser.add_argument(
        "--moto-conf",
        type=float,
        default=DEFAULT_MOTO_CONF,
        help="motorcycle class confidence threshold (3)",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=DEFAULT_PERSON_CONF,
        help="person class confidence threshold (0)",
    )
    parser.add_argument(
        "--person-min-frames",
        type=int,
        default=DEFAULT_PERSON_MIN_FRAMES,
        help="minimum consecutive frames before person track is emitted",
    )
    parser.add_argument(
        "--vehicle-min-frames",
        type=int,
        default=DEFAULT_VEHICLE_MIN_FRAMES,
        help="minimum consecutive frames before vehicle track is emitted",
    )
    parser.add_argument(
        "--vehicle-min-area-ratio",
        type=float,
        default=DEFAULT_VEHICLE_MIN_AREA_RATIO,
        help="minimum vehicle bbox area ratio (bbox_area / frame_area)",
    )
    parser.add_argument(
        "--edge-ignore-px",
        type=int,
        default=DEFAULT_EDGE_IGNORE_PX,
        help="suppress vehicle emission when bbox touches frame edges within this margin",
    )
    parser.add_argument(
        "--vehicle-dedupe-iou",
        type=float,
        default=DEFAULT_VEHICLE_DEDUPE_IOU,
        help="same-frame vehicle dedupe IoU threshold",
    )
    parser.add_argument(
        "--stable-keep-ratio",
        type=float,
        default=DEFAULT_STABLE_KEEP_RATIO,
        help="keep threshold ratio for already-established tracks",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="disable realtime window display for better stability/performance",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="print status every N frames",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="run detection/tracking every N frames",
    )
    parser.add_argument(
        "--trail-len",
        type=int,
        default=DEFAULT_TRAIL_LEN,
        help="max cached points per track for drawing trajectory",
    )
    parser.add_argument(
        "--trail-mode",
        choices=["none", "all", "selected"],
        default="all",
        help="trajectory display mode: none / all / selected",
    )
    parser.add_argument(
        "--highlight-ids",
        type=parse_ids_arg,
        default=set(),
        help='highlight track ids, e.g. "12" or "12,15"',
    )
    parser.add_argument(
        "--selected-id",
        type=int,
        default=None,
        help="highlight and separately persist this track id",
    )
    parser.add_argument(
        "--tracks-csv",
        type=str,
        default="./tracks.csv",
        help="path to export all trajectory points CSV",
    )
    parser.add_argument(
        "--selected-csv",
        type=str,
        default="./selected_track.csv",
        help="path to export selected track trajectory CSV",
    )
    parser.add_argument(
        "--trajectory-summary-json",
        type=str,
        default="./trajectory_summary.json",
        help="path to export trajectory analysis summary JSON",
    )
    parser.add_argument(
        "--counts-csv",
        type=str,
        default=DEFAULT_COUNTS_CSV,
        help="path to export count statistics CSV",
    )
    parser.add_argument(
        "--events-csv",
        type=str,
        default=DEFAULT_EVENTS_CSV,
        help="path to export behavior event CSV",
    )
    parser.add_argument(
        "--traffic-csv",
        type=str,
        default=DEFAULT_TRAFFIC_CSV,
        help="path to export traffic speed CSV",
    )
    parser.add_argument(
        "--anomalies-csv",
        type=str,
        default=DEFAULT_ANOMALIES_CSV,
        help="path to export anomaly risk CSV",
    )
    parser.add_argument(
        "--llm-payload-json",
        type=str,
        default=DEFAULT_LLM_PAYLOAD_JSON,
        help="path to export LLM-friendly anomaly payload JSON",
    )
    parser.add_argument(
        "--disable-anomaly",
        action="store_true",
        help="disable rule-based anomaly risk hints",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=DEFAULT_REPORT_DIR,
        help="directory to save generated plots",
    )
    parser.add_argument(
        "--only-selected",
        action="store_true",
        help="only render selected target bbox (others still tracked internally)",
    )
    parser.add_argument(
        "--lost-hold",
        type=int,
        default=DEFAULT_LOST_HOLD,
        help="keep selected target in LOST state for N frames before clearing",
    )
    parser.add_argument(
        "--pixel-to-meter",
        type=float,
        default=DEFAULT_PIXEL_TO_METER,
        help="pixel to meter scale for speed estimation",
    )
    parser.add_argument(
        "--count-line",
        type=parse_xyxy_arg,
        default=COUNT_LINE,
        help='counting line coordinates, e.g. "100,360,900,360"',
    )
    parser.add_argument(
        "--speed-line-1",
        type=parse_xyxy_arg,
        default=LINE1,
        help='speed line 1 coordinates, e.g. "100,300,900,300"',
    )
    parser.add_argument(
        "--speed-line-2",
        type=parse_xyxy_arg,
        default=LINE2,
        help='speed line 2 coordinates, e.g. "100,420,900,420"',
    )
    parser.add_argument(
        "--line-distance-m",
        type=float,
        default=LINE_DISTANCE_M,
        help="real-world distance between two speed lines in meters",
    )
    parser.add_argument(
        "--speed-direction-mode",
        choices=["vertical", "horizontal"],
        default=SPEED_DIRECTION_MODE,
        help="crossing direction mode for double-line speed estimation",
    )
    parser.add_argument(
        "--cross-tolerance",
        type=int,
        default=CROSS_TOLERANCE,
        help="pixel tolerance when detecting line crossing",
    )
    parser.add_argument(
        "--min-speed-time-s",
        type=float,
        default=MIN_SPEED_TIME_S,
        help="minimum valid crossing time between two speed lines",
    )
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=MAX_SPEED_KMH,
        help="maximum valid speed; larger values are filtered as outliers",
    )
    parser.add_argument(
        "--hide-guides",
        action="store_true",
        help="hide line overlays on output video",
    )

    args = parser.parse_args()
    args.input_vid = path.abspath(args.input_vid)
    args.save_path = path.abspath(args.save_path)
    start = perf_counter()

    # 启动主分析流程。
    run_surveillance_pipeline(
        args.input_vid,
        args.save_path,
        show=not args.no_show,
        log_interval=args.log_interval,
        stride=args.stride,
        trail_len=args.trail_len,
        selected_id=args.selected_id,
        highlight_ids=args.highlight_ids,
        trail_mode=args.trail_mode,
        tracks_csv=args.tracks_csv,
        selected_csv=args.selected_csv,
        trajectory_summary_json=args.trajectory_summary_json,
        counts_csv=args.counts_csv,
        events_csv=args.events_csv,
        traffic_csv=args.traffic_csv,
        anomalies_csv=args.anomalies_csv,
        llm_payload_json=args.llm_payload_json,
        reports_dir=args.reports_dir,
        only_selected=args.only_selected,
        lost_hold=args.lost_hold,
        pixel_to_meter=args.pixel_to_meter,
        model_path=args.model_path,
        tracker_cfg=args.tracker,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        imgsz=args.imgsz,
        classes_spec=args.classes,
        device=args.device,
        veh_conf=args.veh_conf,
        moto_conf=args.moto_conf,
        person_conf=args.person_conf,
        person_min_frames=args.person_min_frames,
        vehicle_min_frames=args.vehicle_min_frames,
        vehicle_min_area_ratio=args.vehicle_min_area_ratio,
        edge_ignore_px=args.edge_ignore_px,
        vehicle_dedupe_iou=args.vehicle_dedupe_iou,
        stable_keep_ratio=args.stable_keep_ratio,
        show_guides=not args.hide_guides,
        count_line_xyxy=args.count_line,
        line1_xyxy=args.speed_line_1,
        line2_xyxy=args.speed_line_2,
        line_distance_m=args.line_distance_m,
        speed_mode=SPEED_MODE,
        speed_direction_mode=args.speed_direction_mode,
        cross_tolerance=args.cross_tolerance,
        min_speed_time_s=args.min_speed_time_s,
        max_speed_kmh=args.max_speed_kmh,
        anomaly_enabled=not args.disable_anomaly,
    )

    print(f"Total time: {perf_counter()-start:.2f} s")
