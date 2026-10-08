"""系统统一配置。

说明：
- 所有模块尽量从这里读取默认参数，避免散落硬编码。
- 坐标请根据视频分辨率手动调优。
"""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# -------------------------
# 通用路径/文件
# -------------------------
DEFAULT_DB_PATH = "users.db"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_DIR = "output_reports"
SUMMARY_PLOT_NAME = "综合统计图.png"

# -------------------------
# 模型与跟踪（YOLO11 + ByteTrack）
# -------------------------
YOLO_MODEL = "./yolo11x.pt"
TRACKER_CFG = "./trackers/bytetrack_stable.yaml"
YOLO_CONF = 0.06
YOLO_IOU = 0.80
IMGSZ = 1536
CLASSES_SPEC = "person_vehicle"

# 类别二阶段阈值
VEH_CONF = 0.10
MOTO_CONF = 0.05
PERSON_CONF = 0.30
PERSON_MIN_FRAMES = 1
VEHICLE_MIN_FRAMES = 1
VEHICLE_MIN_AREA_RATIO = 0.0
EDGE_IGNORE_PX = 0
VEHICLE_DEDUPE_IOU = 0.90
STABLE_KEEP_RATIO = 0.70

# 轨迹/帧处理
STRIDE = 3
TRAIL_LEN = 120
LOST_HOLD = 8
PIXEL_TO_METER = 0.05
TRAJECTORY_RELINK_ENABLED = True
TRAJECTORY_RELINK_MAX_DIST = 60.0
TRAJECTORY_RELINK_MAX_GAP = 18

# -------------------------
# 异常风险提示（规则型，不作为违法/事故判定）
# -------------------------
ANOMALY_ENABLED = True
ANOMALY_REAR_END_HOLD_FRAMES = 3
ANOMALY_REAR_END_X_OVERLAP_RATIO = 0.75
ANOMALY_REAR_END_Y_GAP_RATIO = 0.18
ANOMALY_REAR_END_CONFIRM_FRAMES = 36
ANOMALY_REAR_END_STOP_WINDOW_FRAMES = 12
ANOMALY_REAR_END_STOP_MOVE_PX = 6.0
ANOMALY_REAR_END_CONTEXT_DISTANCE_PX = 220.0
ANOMALY_REAR_END_OTHER_MOTION_PX = 14.0
ANOMALY_REAR_END_OTHER_LATERAL_PX = 12.0
ANOMALY_MAX_SNAPSHOTS = 0  # 0 表示不限制追尾证据截图事件数量
ANOMALY_EVIDENCE_OFFSETS_SEC = (-3.0, -1.0, 0.0, 1.0, 3.0)
ANOMALY_QWEN_OFFSETS_SEC = (-3.0, 0.0, 3.0, 8.0)
ANOMALY_REAR_END_MOTION_CHECK_OFFSETS_SEC = (5.0, 8.0)
ANOMALY_REAR_END_MAX_POST_MOVE_PX = 10.0
ANOMALY_QWEN_EXIT_CHECK_SEC = 8.0
ANOMALY_QWEN_FULL_IMAGE_MAX_SIZE_PX = 960

# 大模型复核接口只消费结构化事件和截图引用，避免直接依赖完整视频输入。
LLM_PAYLOAD_NAME = "llm_payload.json"
LLM_MAX_EVENTS = 50

# 千问/Qwen 复核配置。网页端可为单个任务开启复核并输入 API key；
# 也可以继续通过环境变量提供默认 key。
QWEN_REVIEW_ENABLED = env_bool("QWEN_REVIEW_ENABLED", True)
QWEN_API_KEY_ENV = "QWEN_API_KEY"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-vl-plus"
QWEN_TIMEOUT_SEC = 30
QWEN_IMAGE_JPEG_QUALITY = 68
QWEN_MAX_PROMPT_CHARS = 650000


# -------------------------
# 统计线 / 双线测速（按当前项目默认）
# -------------------------
LINE1 = (1690, 1080, 1984, 1080)
LINE2 = (1592, 1180, 1984, 1180)
COUNT_LINE = LINE1
LINE_DISTANCE_M = 20.0
SPEED_DIRECTION_MODE = "vertical"  # vertical 表示上下方向，horizontal 表示左右方向
CROSS_TOLERANCE = 8
SPEED_MODE = "double_line"  # double_line 为双线测速，pixel_estimate 为像素比例估算

# 测速稳定性
MIN_SPEED_TIME_S = 0.25
MAX_SPEED_KMH = 200.0

# -------------------------
# 网页配置
# -------------------------
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_DEBUG = True
WEB_PREVIEW_ENABLED = True
WEB_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1GB
WEB_ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "flv"}
WEB_SECRET_KEY = "graduation-demo-secret-key"
# 网络流超时用于首帧预览和后台分析，避免 FLV/RTMP 地址不可达时长期阻塞。
WEB_STREAM_OPEN_TIMEOUT_MS = 8000
WEB_STREAM_READ_TIMEOUT_MS = 8000

# MJPEG 推流
MJPEG_SLEEP_SEC = 0.08
MJPEG_QUALITY = 72
WEB_PREVIEW_MIN_INTERVAL_SEC = 0.12
MJPEG_PLACEHOLDER_W = 960
MJPEG_PLACEHOLDER_H = 540

# 任务状态
TASK_STATUS_UPLOADED = "uploaded"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_DONE = "done"
TASK_STATUS_FAILED = "failed"

# 命令行认证
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
