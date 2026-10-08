"""轻量 Flask 网页系统：登录、上传、实时预览分析、结果展示。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from auth import ADMIN_PASSWORD, ADMIN_USERNAME, AuthService
from config import (
    CLASSES_SPEC,
    COUNT_LINE,
    CROSS_TOLERANCE,
    DEFAULT_DB_PATH,
    EDGE_IGNORE_PX,
    IMGSZ,
    LINE1,
    LINE2,
    LINE_DISTANCE_M,
    QWEN_REVIEW_ENABLED,
    QWEN_API_KEY_ENV,
    LLM_MAX_EVENTS,
    LLM_PAYLOAD_NAME,
    LOST_HOLD,
    MAX_SPEED_KMH,
    MIN_SPEED_TIME_S,
    MJPEG_PLACEHOLDER_H,
    MJPEG_PLACEHOLDER_W,
    MJPEG_QUALITY,
    MJPEG_SLEEP_SEC,
    WEB_PREVIEW_MIN_INTERVAL_SEC,
    MOTO_CONF,
    PERSON_CONF,
    PERSON_MIN_FRAMES,
    PIXEL_TO_METER,
    SPEED_DIRECTION_MODE,
    SPEED_MODE,
    STABLE_KEEP_RATIO,
    STRIDE,
    SUMMARY_PLOT_NAME,
    TASK_STATUS_DONE,
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    TRACKER_CFG,
    TRAIL_LEN,
    VEH_CONF,
    VEHICLE_DEDUPE_IOU,
    VEHICLE_MIN_AREA_RATIO,
    VEHICLE_MIN_FRAMES,
    WEB_ALLOWED_EXTENSIONS,
    WEB_DEBUG,
    WEB_HOST,
    WEB_MAX_UPLOAD_BYTES,
    WEB_PORT,
    WEB_PREVIEW_ENABLED,
    WEB_SECRET_KEY,
    WEB_STREAM_OPEN_TIMEOUT_MS,
    WEB_STREAM_READ_TIMEOUT_MS,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MODEL,
)
from llm_payload import export_llm_violation_payload
from main import AnalysisCancelled, pick_track_id_from_click, run_surveillance_pipeline
from task_db import TaskDB

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DEFAULT_DB_PATH
UPLOAD_DIR = BASE_DIR / "web_data" / "uploads"
RESULTS_ROOT = BASE_DIR / "web_data" / "results"
ALLOWED_EXTENSIONS = WEB_ALLOWED_EXTENSIONS
STREAM_SCHEMES = {"http", "https", "rtmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("WEB_SECRET_KEY", WEB_SECRET_KEY)
app.config["MAX_CONTENT_LENGTH"] = WEB_MAX_UPLOAD_BYTES

auth_service = AuthService(db_path=str(DB_PATH))
task_db = TaskDB(db_path=str(DB_PATH))

STREAM_LOCK = threading.Lock()
# 三个内存字典只保存运行期状态，服务重启后可由数据库任务记录重新进入页面。
STREAM_STATES: Dict[int, dict] = {}
RUNNING_THREADS: Dict[int, threading.Thread] = {}
RUNTIME_CONTROLS: Dict[int, dict] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_stream_source(value: str) -> bool:
    """判断输入源是否为网页端配置的流地址。"""
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme.lower() in STREAM_SCHEMES and bool(parsed.netloc)


def is_supported_flv_stream(value: str) -> bool:
    """校验 HTTP-FLV / RTMP-FLV 地址，避免普通文本进入 VideoCapture。"""
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme == "rtmp":
        return bool(parsed.netloc)
    if scheme in {"http", "https"}:
        return bool(parsed.netloc) and parsed.path.lower().endswith(".flv")
    return False


def resolve_video_source(value: str) -> str:
    """把任务输入源转换为 VideoCapture 可用的字符串。"""
    raw = str(value or "").strip()
    if is_stream_source(raw):
        return raw
    return str(Path(raw).resolve())


def open_video_capture(value: str):
    """打开本地视频或网络流；网络流使用超时参数，避免不可达时长时间阻塞。"""
    source = resolve_video_source(value)
    if is_stream_source(source):
        params = []
        # OpenCV 只有 FFmpeg 后端支持这些超时参数；旧版本不支持时自动回退。
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


def video_source_label(value: str) -> str:
    """模板中展示视频来源名称，本地文件显示文件名，流地址显示主机和路径。"""
    raw = str(value or "")
    if is_stream_source(raw):
        parsed = urlparse(raw)
        path_part = unquote(parsed.path.rsplit("/", 1)[-1] or parsed.path or "live.flv")
        return f"FLV流：{parsed.netloc}/{path_part}".rstrip("/")
    return Path(raw).name


app.jinja_env.filters["video_source_label"] = video_source_label


def login_required(view_func):
    @wraps(view_func)
    def _wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("请先登录。", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return _wrapped


def read_task_summary(counts_csv: Path) -> tuple[int, float, float, int]:
    """读取 counts.csv 最后一行，回填任务列表和结果页的摘要指标。"""
    flow_total = 0
    avg_speed = 0.0
    max_speed = 0.0
    measured_count = 0
    if not counts_csv.exists():
        return flow_total, avg_speed, max_speed, measured_count

    last_row = None
    with counts_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    if last_row:
        flow_total = int(float(last_row.get("flow_total", 0) or 0))
        avg_speed = float(last_row.get("avg_speed_kmh", 0) or 0)
        max_speed = float(last_row.get("max_speed_kmh", 0) or 0)
        measured_count = int(float(last_row.get("measured_count", 0) or 0))
    return flow_total, avg_speed, max_speed, measured_count


def build_task_outputs(task_output_dir: Path) -> dict[str, Path]:
    """统一生成单个网页任务的输出文件路径。"""
    reports_dir = task_output_dir / "output_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_video": task_output_dir / "output.mp4",
        "counts_csv": task_output_dir / "counts.csv",
        "events_csv": task_output_dir / "events.csv",
        "traffic_csv": task_output_dir / "traffic.csv",
        "anomalies_csv": task_output_dir / "anomalies.csv",
        "llm_payload_json": task_output_dir / LLM_PAYLOAD_NAME,
        "tracks_csv": task_output_dir / "tracks.csv",
        "selected_csv": task_output_dir / "selected_track.csv",
        "trajectory_summary_json": task_output_dir / "trajectory_summary.json",
        "report_png": reports_dir / SUMMARY_PLOT_NAME,
        "preview_jpg": task_output_dir / "preview.jpg",
        "reports_dir": reports_dir,
    }


def xyxy_to_line_dict(xyxy) -> dict[str, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def line_dict_to_xyxy(value, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        try:
            return (
                int(round(float(value["x1"]))),
                int(round(float(value["y1"]))),
                int(round(float(value["x2"]))),
                int(round(float(value["y2"]))),
            )
        except (KeyError, TypeError, ValueError):
            return default
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(round(float(v))) for v in value)
        except (TypeError, ValueError):
            return default
    return default


def get_video_info(video_source: str | Path) -> tuple[int, int]:
    """读取本地视频或流地址的画面尺寸，失败时返回网页预览占位尺寸。"""
    cap = open_video_capture(str(video_source))
    try:
        if not cap.isOpened():
            return MJPEG_PLACEHOLDER_W, MJPEG_PLACEHOLDER_H
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or MJPEG_PLACEHOLDER_W
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or MJPEG_PLACEHOLDER_H
        return width, height
    finally:
        cap.release()


def ensure_preview(task) -> tuple[Path, int, int]:
    """生成并返回当前任务的视频首帧预览。

    配置页使用 preview.jpg 作为调线背景。保存时前端会把显示坐标换算回
    原始视频坐标，因此这里同时返回真实视频宽高。
    """
    input_source = resolve_video_source(task["input_video"])
    output_dir = Path(task["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "preview.jpg"
    cap = open_video_capture(input_source)
    try:
        if not cap.isOpened():
            # 流地址可能暂时不可达，仍生成占位图，让用户可以先配置线条。
            if not preview_path.exists():
                frame = np.full((MJPEG_PLACEHOLDER_H, MJPEG_PLACEHOLDER_W, 3), 24, dtype=np.uint8)
                cv2.putText(
                    frame,
                    "Stream Preview Unavailable",
                    (30, MJPEG_PLACEHOLDER_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(preview_path), frame)
            return preview_path, MJPEG_PLACEHOLDER_W, MJPEG_PLACEHOLDER_H
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or MJPEG_PLACEHOLDER_W
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or MJPEG_PLACEHOLDER_H
        if not preview_path.exists():
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(preview_path), frame)
        return preview_path, width, height
    finally:
        cap.release()


def default_line_config(video_width: int, video_height: int) -> dict[str, Any]:
    """根据视频尺寸生成适合首帧配置页展示的默认线条。"""
    def clipped(line):
        x1, y1, x2, y2 = line
        return xyxy_to_line_dict((
            max(0, min(video_width - 1, x1)),
            max(0, min(video_height - 1, y1)),
            max(0, min(video_width - 1, x2)),
            max(0, min(video_height - 1, y2)),
        ))

    if max(LINE1 + LINE2) >= max(video_width, video_height) * 1.5:
        count_y = int(video_height * 0.55)
        speed_y1 = int(video_height * 0.58)
        speed_y2 = int(video_height * 0.70)
        x1 = int(video_width * 0.15)
        x2 = int(video_width * 0.85)
        count_line = xyxy_to_line_dict((x1, count_y, x2, count_y))
        speed_line_1 = xyxy_to_line_dict((x1, speed_y1, x2, speed_y1))
        speed_line_2 = xyxy_to_line_dict((x1, speed_y2, x2, speed_y2))
    else:
        count_line = clipped(COUNT_LINE)
        speed_line_1 = clipped(LINE1)
        speed_line_2 = clipped(LINE2)

    return {
        "count_line": count_line,
        "speed_line_1": speed_line_1,
        "speed_line_2": speed_line_2,
        "line_distance_m": LINE_DISTANCE_M,
        "speed_direction_mode": SPEED_DIRECTION_MODE,
        "cross_tolerance": CROSS_TOLERANCE,
        "trail_mode": "all",
        "highlight_ids": [],
        "trail_len": TRAIL_LEN,
        "video_width": video_width,
        "video_height": video_height,
    }


def parse_line_xyxy(value: str, default: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], bool]:
    raw = (value or "").strip()
    if not raw:
        return default, True
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        return default, False
    try:
        return tuple(int(float(p)) for p in parts), True
    except ValueError:
        return default, False


def parse_float(value: str, default: float, min_value: float | None = None) -> tuple[float, bool]:
    raw = (value or "").strip()
    if not raw:
        return default, True
    try:
        parsed = float(raw)
        if min_value is not None and parsed < min_value:
            return default, False
        return parsed, True
    except ValueError:
        return default, False


def parse_int(value: str, default: int, min_value: int | None = None) -> tuple[int, bool]:
    raw = (value or "").strip()
    if not raw:
        return default, True
    try:
        parsed = int(float(raw))
        if min_value is not None and parsed < min_value:
            return default, False
        return parsed, True
    except ValueError:
        return default, False


def parse_highlight_ids(value: str) -> tuple[list[int], bool]:
    raw = (value or "").strip()
    if not raw:
        return [], True
    ids: list[int] = []
    try:
        for part in raw.split(","):
            item = part.strip()
            if item:
                ids.append(int(item))
        return sorted(set(ids)), True
    except ValueError:
        return [], False


def parse_checkbox(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_task_config(form) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    model_path = (form.get("model_path") or YOLO_MODEL).strip() or YOLO_MODEL
    if model_path not in {"yolo11n.pt", "yolo11s.pt", "yolo11x.pt", "./yolo11n.pt", "./yolo11s.pt", "./yolo11x.pt"}:
        warnings.append("模型选择无效，已使用默认模型。")
        model_path = YOLO_MODEL

    conf, ok = parse_float(form.get("conf", ""), YOLO_CONF, min_value=0.0)
    if not ok:
        warnings.append("置信度阈值格式错误，已使用默认值。")
    imgsz, ok = parse_int(form.get("imgsz", ""), IMGSZ, min_value=32)
    if not ok:
        warnings.append("输入尺寸格式错误，已使用默认值。")
    line1, ok = parse_line_xyxy(form.get("speed_line_1", ""), LINE1)
    if not ok:
        warnings.append("测速线1坐标格式错误，已使用默认值。")
    line2, ok = parse_line_xyxy(form.get("speed_line_2", ""), LINE2)
    if not ok:
        warnings.append("测速线2坐标格式错误，已使用默认值。")
    distance, ok = parse_float(form.get("line_distance_m", ""), LINE_DISTANCE_M, min_value=0.01)
    if not ok:
        warnings.append("两线实际距离格式错误，已使用默认值。")
    direction = (form.get("speed_direction_mode") or SPEED_DIRECTION_MODE).strip()
    if direction not in {"vertical", "horizontal"}:
        warnings.append("方向模式无效，已使用默认值。")
        direction = SPEED_DIRECTION_MODE
    tolerance, ok = parse_int(form.get("cross_tolerance", ""), CROSS_TOLERANCE, min_value=0)
    if not ok:
        warnings.append("穿线容差格式错误，已使用默认值。")

    trail_mode = (form.get("trail_mode") or "all").strip()
    if trail_mode not in {"none", "all", "selected"}:
        warnings.append("轨迹显示模式无效，已使用默认值。")
        trail_mode = "all"
    highlight_ids, ok = parse_highlight_ids(form.get("highlight_ids", ""))
    if not ok:
        warnings.append("高亮目标ID格式错误，已忽略该配置。")
    trail_len, ok = parse_int(form.get("trail_len", ""), TRAIL_LEN, min_value=1)
    if not ok:
        warnings.append("轨迹长度格式错误，已使用默认值。")
    # 千问 key 只保存在任务配置中，运行该任务时传入分析流程；不写入源码或 README。
    qwen_review_enabled = parse_checkbox(form.get("qwen_review_enabled"))
    qwen_api_key = (form.get("qwen_api_key") or "").strip()

    return {
        "model_path": model_path,
        "conf": conf,
        "imgsz": imgsz,
        "classes_spec": CLASSES_SPEC,
        "count_line": xyxy_to_line_dict(COUNT_LINE),
        "speed_line_1": list(line1),
        "speed_line_2": list(line2),
        "line_distance_m": distance,
        "speed_direction_mode": direction,
        "cross_tolerance": tolerance,
        "trail_mode": trail_mode,
        "highlight_ids": highlight_ids,
        "trail_len": trail_len,
        "qwen_review_enabled": qwen_review_enabled,
        "qwen_api_key": qwen_api_key,
    }, warnings


def load_task_config(task) -> dict[str, Any]:
    video_width = MJPEG_PLACEHOLDER_W
    video_height = MJPEG_PLACEHOLDER_H
    if task is not None:
        video_width, video_height = get_video_info(task["input_video"])
    defaults = {
        "model_path": YOLO_MODEL,
        "conf": YOLO_CONF,
        "imgsz": IMGSZ,
        "classes_spec": CLASSES_SPEC,
        "qwen_review_enabled": False,
        "qwen_api_key": "",
        **default_line_config(video_width, video_height),
    }
    if task is None:
        return defaults
    raw = task["config_json"] if "config_json" in task.keys() else None
    if not raw:
        return defaults
    try:
        data = json.loads(raw)
    except Exception:
        return defaults
    if isinstance(data, dict):
        defaults.update(data)
    defaults["count_line"] = xyxy_to_line_dict(line_dict_to_xyxy(defaults.get("count_line"), COUNT_LINE))
    defaults["speed_line_1"] = xyxy_to_line_dict(line_dict_to_xyxy(defaults.get("speed_line_1"), LINE1))
    defaults["speed_line_2"] = xyxy_to_line_dict(line_dict_to_xyxy(defaults.get("speed_line_2"), LINE2))
    defaults["qwen_review_enabled"] = bool(defaults.get("qwen_review_enabled", False))
    defaults["qwen_api_key"] = str(defaults.get("qwen_api_key") or "")
    return defaults


def task_file_state(task) -> dict[str, bool]:
    output_dir = Path(task["output_dir"])
    paths = {
        "input": "" if is_stream_source(task["input_video"]) else task["input_video"],
        "output": task["output_video"],
        "counts": task["counts_csv"],
        "events": task["events_csv"],
        "traffic": task["traffic_csv"],
        "anomalies": str(output_dir / "anomalies.csv"),
        "report": task["report_png"],
        "tracks": str(output_dir / "tracks.csv"),
    }
    return {key: bool(value and Path(value).exists()) for key, value in paths.items()}


def qwen_api_key_available() -> bool:
    return bool(os.environ.get(QWEN_API_KEY_ENV, "").strip() or os.environ.get("DASHSCOPE_API_KEY", "").strip())


def task_qwen_active(task_config: dict[str, Any]) -> bool:
    """任务级千问开关；API key 可来自页面输入或环境变量。"""
    if not bool(QWEN_REVIEW_ENABLED):
        return False
    if not bool(task_config.get("qwen_review_enabled", False)):
        return False
    return bool(str(task_config.get("qwen_api_key") or "").strip() or qwen_api_key_available())


def clean_model_review_reason(value) -> str:
    """结果页只展示模型复核结论，过滤内部规则复核补充说明。"""
    text = str(value or "").strip()
    if not text:
        return ""
    internal_markers = (
        "触发时刻目标ID缓存不完整",
        "已跳过位移复核",
        "后续位移检查通过",
        "视频在后续证据点前结束，已使用最后一帧作为后续证据",
    )
    parts = [part.strip() for part in text.split("；") if part.strip()]
    kept = [part for part in parts if not any(marker in part for marker in internal_markers)]
    return "；".join(kept)


def build_rear_end_review_item(task_id: int, row: dict) -> dict:
    """把 anomalies.csv 行或运行时事件转换成模板/API 使用的数据结构。"""
    evidence_paths = []
    raw_paths = (row.get("evidence_paths") or "").strip()
    if raw_paths:
        try:
            parsed = json.loads(raw_paths)
            if isinstance(parsed, list):
                evidence_paths = [str(item) for item in parsed if str(item)]
        except json.JSONDecodeError:
            evidence_paths = [item.strip() for item in raw_paths.split(";") if item.strip()]
    elif row.get("evidence_path"):
        evidence_paths = [str(row["evidence_path"])]
    qwen_image_paths = []
    raw_qwen_paths = (row.get("qwen_image_paths") or "").strip()
    if raw_qwen_paths:
        try:
            parsed = json.loads(raw_qwen_paths)
            if isinstance(parsed, list):
                qwen_image_paths = [str(item) for item in parsed if str(item)]
        except json.JSONDecodeError:
            qwen_image_paths = [item.strip() for item in raw_qwen_paths.split(";") if item.strip()]
    evidence_offsets = parse_json_number_list(row.get("evidence_offsets_sec"))
    evidence_times = parse_json_number_list(row.get("evidence_times_sec"))

    downloads = []
    for idx, rel_path in enumerate(evidence_paths):
        prefix = "anomaly_snapshots/"
        if rel_path.startswith(prefix):
            offset_value = evidence_offsets[idx] if idx < len(evidence_offsets) else None
            time_value = evidence_times[idx] if idx < len(evidence_times) else None
            downloads.append(
                {
                    "path": rel_path,
                    "url": url_for("anomaly_snapshot_file", task_id=task_id, filename=rel_path[len(prefix) :]),
                    "label": format_evidence_offset(offset_value, idx),
                    "timestamp": time_value,
                }
            )
    qwen_reason = clean_model_review_reason(row.get("qwen_reason", ""))
    qwen_error = row.get("qwen_error", "")
    qwen_result = row.get("qwen_result", "")
    return {
        "frame_id": row.get("frame_id", ""),
        "timestamp": row.get("timestamp", ""),
        "track_id": row.get("track_id", ""),
        "related_track_id": row.get("related_track_id", ""),
        "description": row.get("description", ""),
        "qwen_result": qwen_result,
        "qwen_reason": qwen_reason,
        "qwen_error": qwen_error,
        "qwen_image_paths": qwen_image_paths,
        "downloads": downloads,
    }


def read_rear_end_reviews(task_id: int, output_dir: Path) -> list[dict]:
    """读取追尾风险复核结果，给结果页展示 true 事件和截图下载列表。"""
    anomalies_path = output_dir / "anomalies.csv"
    if not anomalies_path.exists():
        return []
    reviews = []
    with anomalies_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") != "rear_end_risk":
                continue
            reviews.append(build_rear_end_review_item(task_id, row))
    return reviews


def filter_rear_end_reviews_for_display(
    reviews: list[dict],
    qwen_only: bool = False,
    allow_pending: bool = False,
) -> list[dict]:
    """过滤结果页展示事件。

    保留千问实际复核过的事件和规则疑似事件；规则层已判定正常
    离开/目标丢失/明显位移并跳过千问的事件，不在结果页作为
    追尾风险报告展示。
    """
    filtered = []
    skip_markers = (
        "跳过千问复核",
        "按正常离开处理",
        "目标ID已丢失",
        "目标框已丢失",
        "未达到持续静止条件",
        "千问输入截图生成不完整",
    )
    for item in reviews:
        reason = str(item.get("qwen_reason") or "")
        error = str(item.get("qwen_error") or "")
        text = f"{reason} {error}"
        result = str(item.get("qwen_result") or "").lower()
        is_pending = "规则检测已触发" in text
        if result == "false" and any(marker in text for marker in skip_markers):
            continue
        if qwen_only and not item.get("downloads") and not error:
            continue
        if qwen_only and not result and not error and not (allow_pending and is_pending):
            continue
        filtered.append(item)
    return filtered


def parse_json_number_list(value) -> list[float]:
    """解析 anomalies.csv 中保存的 JSON 数字数组。"""
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    numbers = []
    for item in parsed:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def format_evidence_offset(offset_value, index: int) -> str:
    """将截图相对时间格式化为给人看的标签。"""
    if offset_value is None:
        return f"截图 {index + 1}"
    offset = float(offset_value)
    if abs(offset) < 1e-6:
        return "触发时刻"
    if offset < 0:
        return f"前 {abs(offset):.0f}s"
    return f"后 {offset:.0f}s"


DOWNLOAD_NAMES = {
    "input": "原始视频",
    "output": "标注结果视频.mp4",
    "counts": "车流统计表.csv",
    "events": "测速事件表.csv",
    "traffic": "测速明细表.csv",
    "anomalies": "追尾风险记录表.csv",
    "report": "综合统计图.png",
    "tracks": "轨迹记录表.csv",
}


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), MJPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes()


def _status_frame(text: str, w: int = MJPEG_PLACEHOLDER_W, h: int = MJPEG_PLACEHOLDER_H) -> np.ndarray:
    frame = np.full((h, w, 3), 24, dtype=np.uint8)
    cv2.putText(frame, text, (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    return frame


def _set_stream_state(task_id: int, **kwargs) -> None:
    """线程安全地更新 MJPEG 预览、最新 tracks 和任务提示信息。"""
    with STREAM_LOCK:
        state = STREAM_STATES.get(task_id, {})
        state.update(kwargs)
        STREAM_STATES[task_id] = state


def _update_preview_frame(task_id: int, frame: np.ndarray) -> None:
    """把分析线程产生的 OpenCV 帧编码为 JPEG，供 /video_feed 持续推送。"""
    now = time.time()
    with STREAM_LOCK:
        last_ts = float(STREAM_STATES.get(task_id, {}).get("preview_encode_ts") or 0.0)
    if now - last_ts < float(WEB_PREVIEW_MIN_INTERVAL_SEC):
        return
    jpg = _encode_jpeg(frame)
    if jpg is None:
        return
    _set_stream_state(task_id, latest_jpeg=jpg, ts=now, preview_encode_ts=now)


def _update_tracking_state(task_id: int, tracks, frame_shape) -> None:
    """保存当前帧跟踪框，供网页点击选择接口复用 OpenCV 选中逻辑。"""
    frame_h = int(frame_shape[0]) if frame_shape is not None and len(frame_shape) >= 1 else 0
    frame_w = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) >= 2 else 0
    normalized_tracks = []
    for track_id, bbox_ltrb in tracks or []:
        try:
            l, t, r, b = [int(round(float(v))) for v in bbox_ltrb]
            normalized_tracks.append((int(track_id), (l, t, r, b)))
        except (TypeError, ValueError):
            continue
    _set_stream_state(
        task_id,
        latest_tracks=normalized_tracks,
        frame_width=frame_w,
        frame_height=frame_h,
        tracks_ts=time.time(),
    )


def _update_runtime_stats(task_id: int, stats: dict) -> None:
    """保存分析线程实时统计，供长视频和 FLV 运行中轮询展示。"""
    safe_stats = {
        "flow_total": int(stats.get("flow_total") or 0),
        "avg_speed_kmh": float(stats.get("avg_speed_kmh") or 0.0),
        "max_speed_kmh": float(stats.get("max_speed_kmh") or 0.0),
        "measured_count": int(stats.get("measured_count") or 0),
        "frame_id": int(stats.get("frame_id") or 0),
        "timestamp": float(stats.get("timestamp") or 0.0),
        "total_frames": int(stats.get("total_frames") or 0),
        "source_duration_sec": float(stats.get("source_duration_sec") or 0.0),
        "progress_percent": None if stats.get("progress_percent") is None else float(stats.get("progress_percent") or 0.0),
    }
    _set_stream_state(task_id, runtime_stats=safe_stats, ts=time.time())


def _append_live_rear_end_review(task_id: int, event: dict) -> None:
    """保存运行中的追尾提示，让页面不必等任务结束。"""
    with STREAM_LOCK:
        state = STREAM_STATES.get(task_id, {})
        reviews = list(state.get("rear_end_reviews") or [])
        event_key = (
            str(event.get("frame_id", "")),
            str(event.get("track_id", "")),
            str(event.get("related_track_id", "")),
        )
        updated = False
        for idx, item in enumerate(reviews):
            item_key = (
                str(item.get("frame_id", "")),
                str(item.get("track_id", "")),
                str(item.get("related_track_id", "")),
            )
            if item_key == event_key:
                merged = dict(item)
                merged.update(dict(event))
                reviews[idx] = merged
                updated = True
                break
        if not updated:
            reviews.append(dict(event))
        state["rear_end_reviews"] = reviews
        result = str(event.get("qwen_result") or "").lower()
        if result == "true":
            state["message"] = "千问已确认追尾风险"
        elif result == "false":
            state["message"] = "千问已复核追尾风险"
        else:
            state["message"] = "规则检测到疑似追尾风险"
        STREAM_STATES[task_id] = state


def _start_placeholder(task_id: int, text: str) -> None:
    jpg = _encode_jpeg(_status_frame(text))
    _set_stream_state(task_id, latest_jpeg=jpg, ts=time.time())


def _run_analysis_task(task_id: int) -> None:
    """后台线程入口：读取任务配置，运行分析流程，并把结果写回数据库。"""
    try:
        task = task_db.get_task(task_id)
        if task is None:
            _start_placeholder(task_id, "Task Not Found")
            _set_stream_state(task_id, done=True, message="任务不存在")
            return

        input_video = resolve_video_source(task["input_video"])
        task_output_dir = Path(task["output_dir"]).resolve()
        outputs = build_task_outputs(task_output_dir)
        task_config = load_task_config(task)
        model_path = task_config.get("model_path") or YOLO_MODEL
        model_arg = str(BASE_DIR / model_path) if not Path(str(model_path)).is_absolute() else str(model_path)
        qwen_review_enabled = bool(task_config.get("qwen_review_enabled", False))
        qwen_api_key = str(task_config.get("qwen_api_key") or "")

        _start_placeholder(task_id, "Analyzing... Please Wait")

        def frame_callback(frame):
            _update_preview_frame(task_id, frame)

        def tracking_callback(tracks, frame_shape):
            _update_tracking_state(task_id, tracks, frame_shape)

        def stats_callback(stats):
            _update_runtime_stats(task_id, stats)

        def rear_end_review_callback(event):
            _append_live_rear_end_review(task_id, event)

        def runtime_control_callback():
            # 主分析循环会反复读取该回调，实现暂停、取消、轨迹模式和高亮 ID 的动态更新。
            with STREAM_LOCK:
                return dict(RUNTIME_CONTROLS.get(task_id, {}))

        run_surveillance_pipeline(
            input_vid=input_video,
            save_path=str(outputs["output_video"]),
            show=False,
            log_interval=30,
            stride=STRIDE,
            tracks_csv=str(outputs["tracks_csv"]),
            selected_csv=str(outputs["selected_csv"]),
            trajectory_summary_json=str(outputs["trajectory_summary_json"]),
            counts_csv=str(outputs["counts_csv"]),
            events_csv=str(outputs["events_csv"]),
            traffic_csv=str(outputs["traffic_csv"]),
            anomalies_csv=str(outputs["anomalies_csv"]),
            llm_payload_json=str(outputs["llm_payload_json"]),
            reports_dir=str(outputs["reports_dir"]),
            model_path=model_arg,
            tracker_cfg=TRACKER_CFG,
            yolo_conf=float(task_config.get("conf", YOLO_CONF)),
            yolo_iou=YOLO_IOU,
            imgsz=int(task_config.get("imgsz", IMGSZ)),
            classes_spec=str(task_config.get("classes_spec", CLASSES_SPEC)),
            veh_conf=VEH_CONF,
            moto_conf=MOTO_CONF,
            person_conf=PERSON_CONF,
            person_min_frames=PERSON_MIN_FRAMES,
            vehicle_min_frames=VEHICLE_MIN_FRAMES,
            vehicle_min_area_ratio=VEHICLE_MIN_AREA_RATIO,
            edge_ignore_px=EDGE_IGNORE_PX,
            vehicle_dedupe_iou=VEHICLE_DEDUPE_IOU,
            lost_hold=LOST_HOLD,
            stable_keep_ratio=STABLE_KEEP_RATIO,
            pixel_to_meter=PIXEL_TO_METER,
            speed_mode=SPEED_MODE,
            speed_direction_mode=str(task_config.get("speed_direction_mode", SPEED_DIRECTION_MODE)),
            cross_tolerance=int(task_config.get("cross_tolerance", CROSS_TOLERANCE)),
            count_line_xyxy=line_dict_to_xyxy(task_config.get("count_line"), COUNT_LINE),
            line1_xyxy=line_dict_to_xyxy(task_config.get("speed_line_1"), LINE1),
            line2_xyxy=line_dict_to_xyxy(task_config.get("speed_line_2"), LINE2),
            line_distance_m=float(task_config.get("line_distance_m", LINE_DISTANCE_M)),
            min_speed_time_s=MIN_SPEED_TIME_S,
            max_speed_kmh=MAX_SPEED_KMH,
            trail_mode=str(task_config.get("trail_mode", "all")),
            highlight_ids=set(int(x) for x in task_config.get("highlight_ids", [])),
            trail_len=int(task_config.get("trail_len", TRAIL_LEN)),
            frame_callback=frame_callback if WEB_PREVIEW_ENABLED else None,
            tracking_callback=tracking_callback,
            stats_callback=stats_callback,
            rear_end_review_callback=rear_end_review_callback,
            runtime_control_callback=runtime_control_callback,
            qwen_review_enabled=qwen_review_enabled,
            qwen_api_key=qwen_api_key,
        )

        flow_total, avg_speed, max_speed, measured_count = read_task_summary(outputs["counts_csv"])
        task_db.mark_done(
            task_id=task_id,
            output_video=str(outputs["output_video"]),
            counts_csv=str(outputs["counts_csv"]),
            events_csv=str(outputs["events_csv"]),
            traffic_csv=str(outputs["traffic_csv"]),
            report_png=str(outputs["report_png"]),
            flow_total=flow_total,
            avg_speed_kmh=avg_speed,
            max_speed_kmh=max_speed,
            measured_count=measured_count,
        )
        _start_placeholder(task_id, "Analysis Completed")
        _set_stream_state(task_id, done=True, message="分析完成")
    except AnalysisCancelled:
        _start_placeholder(task_id, "Analysis Cancelled")
        _set_stream_state(task_id, done=True, message="任务已取消")
    except Exception as exc:
        task_db.mark_failed(task_id, str(exc))
        _start_placeholder(task_id, "Analysis Failed")
        _set_stream_state(task_id, done=True, message=f"分析失败: {exc}")
    finally:
        with STREAM_LOCK:
            RUNNING_THREADS.pop(task_id, None)
            RUNTIME_CONTROLS.pop(task_id, None)


def _start_task_thread(task_id: int) -> None:
    with STREAM_LOCK:
        t = RUNNING_THREADS.get(task_id)
        if t is not None and t.is_alive():
            return
        STREAM_STATES[task_id] = {"latest_jpeg": None, "done": False, "message": "running", "ts": time.time()}
        task = task_db.get_task(task_id)
        cfg = load_task_config(task) if task is not None else {}
        RUNTIME_CONTROLS[task_id] = {
            "paused": False,
            "trail_mode": str(cfg.get("trail_mode", "all")),
            "highlight_ids": set(int(x) for x in cfg.get("highlight_ids", [])),
            "trail_len": int(cfg.get("trail_len", TRAIL_LEN)),
        }
        t = threading.Thread(target=_run_analysis_task, args=(task_id,), daemon=True)
        RUNNING_THREADS[task_id] = t
        t.start()


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        ok, msg = auth_service.register(username, password, confirm_password)
        flash(msg, "success" if ok else "danger")
        if ok:
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        ok, msg, user = auth_service.login(username, password)
        flash(msg, "success" if ok else "danger")
        if ok and user is not None:
            session["user_id"] = user.id
            session["username"] = user.username
            session["role"] = user.role
            return redirect(url_for("dashboard"))
    return render_template("login.html", admin_default=f"{ADMIN_USERNAME}/{ADMIN_PASSWORD}")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("已退出登录。", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    tasks = task_db.list_tasks(session["user_id"], session.get("role", "user"))
    return render_template("dashboard.html", tasks=tasks, username=session.get("username"), role=session.get("role"))


@app.route("/tasks/<int:task_id>/delete", methods=["POST"])
@login_required
def delete_task_record(task_id: int):
    """从控制台删除一条任务记录。

    删除范围仅限任务表记录，不删除上传视频和结果文件。运行中任务会先
    写入 cancel 控制信号，后台分析线程在下一次轮询时停止处理。
    """
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        flash("无权限删除该任务。", "danger")
        return redirect(url_for("dashboard"))

    task = task_db.get_task(task_id)
    if task is None:
        flash("任务记录不存在或已删除。", "warning")
        return redirect(url_for("dashboard"))
    was_running = task["status"] == TASK_STATUS_RUNNING
    active_running = False

    if was_running:
        with STREAM_LOCK:
            thread = RUNNING_THREADS.get(task_id)
            if thread is not None and thread.is_alive():
                RUNTIME_CONTROLS.setdefault(task_id, {})["cancel"] = True
                active_running = True
    task_db.delete_task(task_id)
    if not active_running:
        with STREAM_LOCK:
            STREAM_STATES.pop(task_id, None)
            RUNTIME_CONTROLS.pop(task_id, None)
    if active_running:
        flash(f"任务 {task_id} 的网页记录已删除，运行中的后台分析会自动停止。", "success")
    else:
        flash(f"任务 {task_id} 的网页记录已删除。", "success")
    return redirect(url_for("dashboard"))


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_video():
    if request.method == "POST":
        source_type = (request.form.get("source_type") or "file").strip()
        config, warnings = parse_task_config(request.form)
        if source_type == "flv_stream":
            stream_url = (request.form.get("stream_url") or "").strip()
            if not is_supported_flv_stream(stream_url):
                flash("请输入有效的 HTTP-FLV 地址（http/https 且以 .flv 结尾）或 RTMP 地址。", "danger")
                return redirect(url_for("upload_video"))
            input_source = stream_url
        else:
            file = request.files.get("video")
            if file is None or file.filename == "":
                flash("请选择视频文件，或切换为 FLV 流地址。", "warning")
                return redirect(url_for("upload_video"))
            if not allowed_file(file.filename):
                flash("仅支持 mp4 / avi / mov / flv 文件。", "danger")
                return redirect(url_for("upload_video"))

            filename = secure_filename(file.filename)
            user_dir = UPLOAD_DIR / str(session["user_id"])
            user_dir.mkdir(parents=True, exist_ok=True)
            input_path = user_dir / filename
            file.save(str(input_path))
            input_source = str(input_path)

        task_output_dir = RESULTS_ROOT / f"task_{session['user_id']}_{time.time_ns()}"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        task_id = task_db.create_task(
            user_id=int(session["user_id"]),
            username=str(session.get("username", "")),
            input_video=input_source,
            output_dir=str(task_output_dir),
            config_json=json.dumps(config, ensure_ascii=False),
        )
        task = task_db.get_task(task_id)
        if task is not None:
            _, video_width, video_height = ensure_preview(task)
            cfg = load_task_config(task)
            cfg.update(default_line_config(video_width, video_height))
            cfg["model_path"] = config.get("model_path", cfg["model_path"])
            cfg["conf"] = config.get("conf", cfg["conf"])
            cfg["imgsz"] = config.get("imgsz", cfg["imgsz"])
            cfg["classes_spec"] = config.get("classes_spec", cfg["classes_spec"])
            cfg["qwen_review_enabled"] = config.get("qwen_review_enabled", False)
            cfg["qwen_api_key"] = config.get("qwen_api_key", "")
            task_db.update_config(task_id, json.dumps(cfg, ensure_ascii=False))
        flash("视频来源已保存，请在首帧配置页调整统计线和测速线。", "success")
        for msg in warnings:
            flash(msg, "warning")
        return redirect(url_for("config_lines", task_id=task_id))

    return render_template(
        "upload.html",
        defaults={
            "model_path": "yolo11x.pt",
            "conf": 0.15,
            "imgsz": 960,
            "speed_line_1": ",".join(str(x) for x in LINE1),
            "speed_line_2": ",".join(str(x) for x in LINE2),
            "line_distance_m": LINE_DISTANCE_M,
            "speed_direction_mode": SPEED_DIRECTION_MODE,
            "cross_tolerance": CROSS_TOLERANCE,
            "trail_mode": "all",
            "highlight_ids": "",
            "trail_len": TRAIL_LEN,
            "qwen_review_enabled": False,
            "qwen_api_key": "",
        },
    )


@app.route("/tasks/<int:task_id>/config-lines", methods=["GET", "POST"])
@login_required
def config_lines(task_id: int):
    """显示或保存可视化线条配置。

    POST 接收的坐标必须已经是原始视频坐标，后端只做格式归一化和持久化。
    """
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        if request.method == "POST":
            return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
        flash("无权限访问该任务。", "danger")
        return redirect(url_for("dashboard"))

    task = task_db.get_task(task_id)
    if task is None:
        if request.method == "POST":
            return jsonify({"ok": False, "message": "任务不存在。"}), 404
        flash("任务不存在。", "danger")
        return redirect(url_for("dashboard"))

    preview_path, video_width, video_height = ensure_preview(task)
    current = load_task_config(task)
    defaults = default_line_config(video_width, video_height)

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        cfg = load_task_config(task)
        for key, default in (
            ("count_line", COUNT_LINE),
            ("speed_line_1", LINE1),
            ("speed_line_2", LINE2),
        ):
            cfg[key] = xyxy_to_line_dict(line_dict_to_xyxy(payload.get(key), default))

        line_distance_m, _ = parse_float(str(payload.get("line_distance_m", "")), LINE_DISTANCE_M, min_value=0.01)
        cross_tolerance, _ = parse_int(str(payload.get("cross_tolerance", "")), CROSS_TOLERANCE, min_value=0)
        trail_len, _ = parse_int(str(payload.get("trail_len", "")), TRAIL_LEN, min_value=1)
        direction = str(payload.get("speed_direction_mode") or SPEED_DIRECTION_MODE)
        if direction not in {"vertical", "horizontal"}:
            direction = SPEED_DIRECTION_MODE
        trail_mode = str(payload.get("trail_mode") or "all")
        if trail_mode not in {"none", "all", "selected"}:
            trail_mode = "all"
        qwen_review_enabled = bool(payload.get("qwen_review_enabled", False))
        qwen_api_key = str(payload.get("qwen_api_key") or "").strip()
        ids_value = payload.get("highlight_ids", [])
        if isinstance(ids_value, list):
            try:
                highlight_ids = sorted({int(v) for v in ids_value})
            except (TypeError, ValueError):
                highlight_ids = []
        else:
            highlight_ids, _ = parse_highlight_ids(str(ids_value))

        cfg.update(
            {
                "video_width": int(video_width),
                "video_height": int(video_height),
                "line_distance_m": float(line_distance_m),
                "speed_direction_mode": direction,
                "cross_tolerance": int(cross_tolerance),
                "trail_mode": trail_mode,
                "highlight_ids": highlight_ids,
                "trail_len": int(trail_len),
                "qwen_review_enabled": qwen_review_enabled,
                "qwen_api_key": qwen_api_key,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        task_db.update_config(task_id, json.dumps(cfg, ensure_ascii=False))
        return jsonify({"ok": True, "message": "配置已保存"})

    return render_template(
        "config_lines.html",
        task=task,
        task_config=current,
        default_config=defaults,
        preview_url=url_for("media_file", task_id=task_id, file_key="preview"),
        video_width=video_width,
        video_height=video_height,
    )


@app.route("/tasks/<int:task_id>/start", methods=["POST"])
@login_required
def start_task(task_id: int):
    """启动或重新启动分析任务。"""
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "任务不存在。"}), 404
    if task["status"] == TASK_STATUS_RUNNING:
        return jsonify({"ok": False, "message": "任务正在处理中", "redirect_url": url_for("analyze_task", task_id=task_id)})
    if task_db.mark_running_if_possible(task_id):
        _start_task_thread(task_id)
        message = "任务已重新启动" if task["status"] == TASK_STATUS_DONE else "任务已启动"
        return jsonify({"ok": True, "message": message, "redirect_url": url_for("analyze_task", task_id=task_id)})
    return jsonify({"ok": False, "message": "任务状态不允许启动，请刷新页面。"}), 409


@app.route("/tasks/<int:task_id>/trail-control", methods=["POST"])
@login_required
def trail_control(task_id: int):
    """更新运行时轨迹控制参数，后续帧立即读取新设置。"""
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
    payload = request.get_json(silent=True) or {}
    trail_mode = str(payload.get("trail_mode") or "all")
    if trail_mode not in {"none", "all", "selected"}:
        return jsonify({"ok": False, "message": "轨迹模式无效。"}), 400
    highlight_ids, ok = parse_highlight_ids(str(payload.get("highlight_ids", "")))
    if not ok:
        return jsonify({"ok": False, "message": "高亮ID格式错误。"}), 400
    trail_len, ok = parse_int(str(payload.get("trail_len", "")), TRAIL_LEN, min_value=1)
    if not ok:
        return jsonify({"ok": False, "message": "轨迹长度格式错误。"}), 400
    with STREAM_LOCK:
        control = RUNTIME_CONTROLS.setdefault(task_id, {})
        control.update(
            {
                "trail_mode": trail_mode,
                "highlight_ids": set(highlight_ids),
                "trail_len": int(trail_len),
            }
        )
    return jsonify({"ok": True, "message": "轨迹设置已应用"})


@app.route("/tasks/<int:task_id>/click-select", methods=["POST"])
@login_required
def click_select_track(task_id: int):
    """网页端点击实时预览后，按当前帧 tracks 选中或取消高亮目标。"""
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "任务不存在。"}), 404
    if task["status"] != TASK_STATUS_RUNNING:
        return jsonify({"ok": False, "message": "只有处理中任务支持点击选择。"}), 409

    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "select").strip().lower()
    if action not in {"select", "unselect"}:
        return jsonify({"ok": False, "message": "点击操作无效。"}), 400
    try:
        x = int(round(float(payload.get("x"))))
        y = int(round(float(payload.get("y"))))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "点击坐标无效。"}), 400

    with STREAM_LOCK:
        state = dict(STREAM_STATES.get(task_id, {}))
        tracks = list(state.get("latest_tracks") or [])
        frame_width = int(state.get("frame_width") or 0)
        frame_height = int(state.get("frame_height") or 0)

    if frame_width > 0 and frame_height > 0:
        x = max(0, min(frame_width - 1, x))
        y = max(0, min(frame_height - 1, y))
    if not tracks:
        return jsonify({"ok": False, "message": "当前帧暂无可选目标，请等检测框出现后再点击。"}), 404

    selected_id = pick_track_id_from_click((x, y), tracks)
    if selected_id is None:
        return jsonify({"ok": False, "message": "未命中目标，请点击检测框附近。"}), 404

    with STREAM_LOCK:
        control = RUNTIME_CONTROLS.setdefault(task_id, {})
        highlight_ids = set(int(tid) for tid in control.get("highlight_ids", set()))
        if action == "unselect":
            highlight_ids.discard(int(selected_id))
        else:
            highlight_ids.add(int(selected_id))
        control["highlight_ids"] = highlight_ids
        if action == "select" and str(control.get("trail_mode", "all")) == "none":
            control["trail_mode"] = "selected"
    if action == "unselect":
        return jsonify({"ok": True, "message": f"已取消 track_id {selected_id}", "track_id": int(selected_id), "action": action})
    return jsonify({"ok": True, "message": f"已选中 track_id {selected_id}", "track_id": int(selected_id), "action": action})


@app.route("/tasks/<int:task_id>/pause", methods=["POST"])
@login_required
def pause_task(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "任务不存在。"}), 404
    if task["status"] != TASK_STATUS_RUNNING:
        return jsonify({"ok": False, "message": "只有处理中任务可以暂停。"}), 409
    with STREAM_LOCK:
        RUNTIME_CONTROLS.setdefault(task_id, {})["paused"] = True
    _set_stream_state(task_id, message="任务已暂停")
    return jsonify({"ok": True, "message": "任务已暂停"})


@app.route("/tasks/<int:task_id>/resume", methods=["POST"])
@login_required
def resume_task(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "无权限访问该任务。"}), 403
    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "任务不存在。"}), 404
    if task["status"] != TASK_STATUS_RUNNING:
        return jsonify({"ok": False, "message": "只有处理中任务可以继续。"}), 409
    with STREAM_LOCK:
        RUNTIME_CONTROLS.setdefault(task_id, {})["paused"] = False
    _set_stream_state(task_id, message="running")
    return jsonify({"ok": True, "message": "任务已继续"})


@app.route("/analyze/<int:task_id>", methods=["GET", "POST"])
@login_required
def analyze_task(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        flash("无权限访问该任务。", "danger")
        return redirect(url_for("dashboard"))

    task = task_db.get_task(task_id)
    if task is None:
        flash("任务不存在。", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        if task["status"] == TASK_STATUS_DONE:
            return redirect(url_for("result", task_id=task_id))
        if task["status"] == TASK_STATUS_RUNNING:
            flash("任务已在运行中。", "warning")
            return redirect(url_for("analyze_task", task_id=task_id))

        if task_db.mark_running_if_possible(task_id):
            _start_task_thread(task_id)
            flash("任务已启动，正在实时预览检测过程。", "info")
        else:
            flash("任务状态不允许启动，请刷新页面。", "warning")
        return redirect(url_for("analyze_task", task_id=task_id))

    task = task_db.get_task(task_id)
    return render_template("analyze.html", task=task, task_config=load_task_config(task))


@app.route("/task_status/<int:task_id>")
@login_required
def task_status(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"ok": False, "message": "forbidden"}), 403
    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"ok": False, "message": "not found"}), 404
    with STREAM_LOCK:
        state = dict(STREAM_STATES.get(task_id, {}))
        live_stats = dict(state.get("runtime_stats") or {})
        live_review_rows = [dict(item) for item in state.get("rear_end_reviews") or []]
    stats = {
        "flow_total": int(task["flow_total"] or 0),
        "avg_speed_kmh": float(task["avg_speed_kmh"] or 0.0),
        "max_speed_kmh": float(task["max_speed_kmh"] or 0.0),
        "measured_count": int(task["measured_count"] or 0),
    }
    if live_stats:
        stats.update(
            {
                "flow_total": int(live_stats.get("flow_total") or 0),
                "avg_speed_kmh": float(live_stats.get("avg_speed_kmh") or 0.0),
                "max_speed_kmh": float(live_stats.get("max_speed_kmh") or 0.0),
                "measured_count": int(live_stats.get("measured_count") or 0),
                "frame_id": int(live_stats.get("frame_id") or 0),
                "timestamp": float(live_stats.get("timestamp") or 0.0),
                "total_frames": int(live_stats.get("total_frames") or 0),
                "source_duration_sec": float(live_stats.get("source_duration_sec") or 0.0),
                "progress_percent": (
                    None
                    if live_stats.get("progress_percent") is None
                    else float(live_stats.get("progress_percent") or 0.0)
                ),
            }
        )
    qwen_active = task_qwen_active(load_task_config(task))
    rear_end_reviews = []
    for row in live_review_rows:
        rear_end_reviews.append(build_rear_end_review_item(task_id, row))
    rear_end_reviews = filter_rear_end_reviews_for_display(
        rear_end_reviews,
        qwen_only=qwen_active,
        allow_pending=True,
    )
    return jsonify(
        {
            "ok": True,
            "task_id": task_id,
            "status": task["status"],
            "error_message": task["error_message"],
            "result_url": url_for("result", task_id=task_id),
            "paused": bool(RUNTIME_CONTROLS.get(task_id, {}).get("paused", False)),
            "stats": stats,
            "rear_end_reviews": rear_end_reviews,
            "qwen_active": qwen_active,
        }
    )


@app.route("/video_feed/<int:task_id>")
@login_required
def video_feed(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return "forbidden", 403
    if not WEB_PREVIEW_ENABLED:
        def disabled():
            jpg = _encode_jpeg(_status_frame("Preview Disabled"))
            if jpg is not None:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        return Response(disabled(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def generate():
        sent_done = False
        while True:
            task = task_db.get_task(task_id)
            if task is None:
                jpg = _encode_jpeg(_status_frame("Task Not Found"))
                if jpg is not None:
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                break

            with STREAM_LOCK:
                state = STREAM_STATES.get(task_id, {})
                jpg = state.get("latest_jpeg")
                done = bool(state.get("done", False)) or task["status"] in (TASK_STATUS_DONE, TASK_STATUS_FAILED)

            if jpg is None and task["status"] == TASK_STATUS_RUNNING:
                jpg = _encode_jpeg(_status_frame("Waiting for frames..."))
            elif jpg is None:
                txt = "Task Completed" if task["status"] == TASK_STATUS_DONE else "Task Not Running"
                jpg = _encode_jpeg(_status_frame(txt))
                done = True

            if jpg is not None:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

            if done:
                if sent_done:
                    break
                sent_done = True

            time.sleep(MJPEG_SLEEP_SEC)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/result/<int:task_id>")
@login_required
def result(task_id: int):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        flash("无权限访问该任务。", "danger")
        return redirect(url_for("dashboard"))

    task = task_db.get_task(task_id)
    if task is None:
        flash("任务不存在。", "danger")
        return redirect(url_for("dashboard"))

    summary = {
        "tracks_total": 0,
        "avg_track_length": 0.0,
        "top_tracks": [],
    }
    summary_path = Path(task["output_dir"]) / "trajectory_summary.json"
    if summary_path.exists():
        try:
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(rows, list) and rows:
                summary["tracks_total"] = len(rows)
                summary["avg_track_length"] = sum(float(r.get("total_length", 0.0)) for r in rows) / max(1, len(rows))
                summary["top_tracks"] = sorted(
                    rows,
                    key=lambda r: float(r.get("total_length", 0.0)),
                    reverse=True,
                )[:5]
        except Exception:
            pass

    qwen_enabled = task_qwen_active(load_task_config(task))
    rear_end_reviews = read_rear_end_reviews(task_id, Path(task["output_dir"]))
    return render_template(
        "result.html",
        task=task,
        traj_summary=summary,
        task_config=load_task_config(task),
        files=task_file_state(task),
        rear_end_reviews=filter_rear_end_reviews_for_display(rear_end_reviews, qwen_only=qwen_enabled),
        qwen_key_enabled=qwen_enabled,
    )


@app.route("/download/<int:task_id>/<string:file_key>")
@login_required
def download_file(task_id: int, file_key: str):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        flash("无权限访问该任务。", "danger")
        return redirect(url_for("dashboard"))

    task = task_db.get_task(task_id)
    if task is None:
        flash("任务不存在。", "danger")
        return redirect(url_for("dashboard"))

    allowed = {
        "input": "input_video",
        "output": "output_video",
        "counts": "counts_csv",
        "events": "events_csv",
        "traffic": "traffic_csv",
        "anomalies": "anomalies_csv",
        "report": "report_png",
        "tracks": "tracks_csv",
    }
    if file_key not in allowed:
        flash("无效文件类型。", "danger")
        return redirect(url_for("result", task_id=task_id))

    if file_key == "tracks":
        file_path = str(Path(task["output_dir"]) / "tracks.csv")
    elif file_key == "anomalies":
        file_path = str(Path(task["output_dir"]) / "anomalies.csv")
    else:
        file_path = task[allowed[file_key]]
    if not file_path:
        flash("文件不存在。", "warning")
        return redirect(url_for("result", task_id=task_id))
    if file_key == "input" and is_stream_source(file_path):
        flash("流地址不能作为本地文件下载。", "warning")
        return redirect(url_for("result", task_id=task_id))

    path_obj = Path(file_path)
    if not path_obj.exists():
        flash("文件不存在。", "warning")
        return redirect(url_for("result", task_id=task_id))

    download_name = DOWNLOAD_NAMES.get(file_key)
    if file_key == "input":
        suffix = path_obj.suffix
        download_name = f"原始视频{suffix}" if suffix else "原始视频"
    return send_file(str(path_obj), as_attachment=True, download_name=download_name)


def _add_snapshot_download_urls(payload: dict, task_id: int) -> dict:
    """给大模型载荷补充网页可访问的截图下载地址列表。"""
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        download_urls = []
        for item in event.get("evidence", []):
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("path") or "")
            prefix = "anomaly_snapshots/"
            if rel_path.startswith(prefix):
                filename = rel_path[len(prefix) :]
                item["download_url"] = url_for("anomaly_snapshot_file", task_id=task_id, filename=filename)
                item["view_url"] = item["download_url"]
                download_urls.append(item["download_url"])
        event["evidence_download_urls"] = download_urls
    return payload


@app.route("/api/tasks/<int:task_id>/llm-payload")
@login_required
def task_llm_payload(task_id: int):
    """返回面向大模型复核的结构化 JSON，核心证据是异常时刻截图。"""
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return jsonify({"error": "forbidden"}), 403

    task = task_db.get_task(task_id)
    if task is None:
        return jsonify({"error": "not found"}), 404

    output_dir = Path(task["output_dir"])
    payload_path = output_dir / LLM_PAYLOAD_NAME
    if payload_path.exists():
        with payload_path.open("r", encoding="utf-8") as f:
            return jsonify(_add_snapshot_download_urls(json.load(f), task_id))

    payload = export_llm_violation_payload(
        anomalies_csv=output_dir / "anomalies.csv",
        output_json=payload_path,
        output_video=task["output_video"],
        source_video=task["input_video"],
        task_id=task_id,
        max_events=LLM_MAX_EVENTS,
    )
    return jsonify(_add_snapshot_download_urls(payload, task_id))


@app.route("/media/<int:task_id>/anomaly-snapshot/<path:filename>")
@login_required
def anomaly_snapshot_file(task_id: int, filename: str):
    """下载单张异常证据截图。"""
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return "forbidden", 403

    task = task_db.get_task(task_id)
    if task is None:
        return "not found", 404

    snapshot_dir = (Path(task["output_dir"]) / "anomaly_snapshots").resolve()
    file_path = (snapshot_dir / filename).resolve()
    if snapshot_dir not in file_path.parents or not file_path.exists():
        return "not found", 404
    return send_file(str(file_path), as_attachment=False)


@app.route("/media/<int:task_id>/<string:file_key>")
@login_required
def media_file(task_id: int, file_key: str):
    if not task_db.user_can_access(task_id, session["user_id"], session.get("role", "user")):
        return "forbidden", 403

    task = task_db.get_task(task_id)
    if task is None:
        return "not found", 404

    allowed = {
        "input": "input_video",
        "output": "output_video",
        "report": "report_png",
        "preview": "preview_jpg",
    }
    if file_key not in allowed:
        return "bad request", 400

    if file_key == "preview":
        file_path = str(Path(task["output_dir"]) / "preview.jpg")
        if not Path(file_path).exists():
            ensure_preview(task)
    else:
        file_path = task[allowed[file_key]]
    if not file_path:
        return "not found", 404

    path_obj = Path(file_path)
    if not path_obj.exists():
        return "not found", 404

    return send_file(str(path_obj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量智能监控 Flask Web")
    parser.add_argument("--host", type=str, default=os.environ.get("WEB_HOST", WEB_HOST), help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", str(WEB_PORT))), help="监听端口")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("WEB_DEBUG", "1" if WEB_DEBUG else "0") == "1",
        help="开启调试模式",
    )
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
