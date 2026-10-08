# 基于 YOLO11 + ByteTrack 的智能监控系统

本项目用于毕业设计《基于视频目标跟踪的智能监控系统设计与实现》。系统以 YOLO11 完成目标检测，以 ByteTrack 完成多目标跟踪，并接入千问/Qwen 视觉模型复核追尾风险，围绕交通视频提供车辆检测、轨迹显示、车流统计、双线估算测速、网页任务管理和结果导出能力。

当前主线代码统一为 YOLO11 + ByteTrack，仓库保留系统运行和答辩演示需要的核心代码；千问/Qwen 作为追尾风险复核能力接入。

## 功能概览

- 用户注册、登录与角色区分：admin / user。
- 网页端支持本地视频上传，也支持 HTTP-FLV / RTMP-FLV 流地址。
- 系统自动生成首帧预览图，用于后续可视化配置。
- 可视化配置统计线、测速线 1、测速线 2。
- YOLO11 + ByteTrack 检测跟踪车辆和行人。
- MJPEG 实时预览后台分析过程。
- 网页端运行时控制：暂停、继续、最大化预览、轨迹模式切换。
- 网页端点击选择车辆：单击选中，双击取消选中。
- 车流量统计、双线估算测速、轨迹摘要分析。
- 规则型异常风险提示：疑似追尾风险，并可接入千问/Qwen 视觉模型复核。
- 任务记录管理：待分析、运行中、失败、已完成任务都可删除网页记录。
- 导出结果视频、CSV 数据和综合统计图。
- 保留命令行运行方式，便于快速测试算法流程。

## 目录结构

```text
.
├── web_app.py                  # Flask 网页入口
├── run.py                      # 命令行入口
├── main.py                     # YOLO11 + ByteTrack 主分析流程
├── qwen_reviewer.py            # 千问/Qwen 追尾风险复核接口
├── yolo11_bytetrack_tracker.py # 跟踪器构建
├── api/yolo11_bytetrack.py     # YOLO11 + ByteTrack 底层封装
├── analytics.py                # 统计、测速摘要、CSV 行缓存
├── speed_estimator.py          # 穿线计数与双线测速状态机
├── config.py                   # 统一默认参数
├── task_db.py                  # 网页任务数据库
├── auth.py / user_db.py        # 用户认证与用户数据库
├── templates/                  # Jinja2 网页模板
├── static/style.css            # 页面样式
├── static/common.js            # 公共前端交互逻辑
├── trackers/bytetrack_stable.yaml
└── requirements.txt
```

运行时数据默认写入：

```text
web_data/uploads/     # 网页上传视频
web_data/results/     # 网页任务输出
output/               # 命令行默认输出视频和 CSV
output_reports/       # 命令行默认统计图
```

这些目录属于运行产物，已加入 `.gitignore`。

## 环境安装

建议使用 Python 虚拟环境：

```bash
cd YOLO11-ByteTrack-Smart-Surveillance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

本 GitHub 发布包不包含预训练模型权重。首次运行前，可用已安装的 Ultralytics 下载官方 YOLO11 权重到项目根目录：

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11x.pt')"
```

对应文件：

```text
yolo11n.pt
yolo11x.pt
```

网页端默认使用 `yolo11x.pt`；如果只想快速测试，可以仅下载 `yolo11n.pt` 并在上传页切换。项目集成现有 YOLO11 和 ByteTrack，不将这两种算法作为自主研发成果。

## 启动网页端

```bash
cd YOLO11-ByteTrack-Smart-Surveillance
source .venv/bin/activate
WEB_DEBUG=0 python web_app.py --host 127.0.0.1 --port 5001
```

浏览器访问：

```text
http://127.0.0.1:5001
```

默认管理员账号：

```text
admin / admin123
```

## 命令行运行

默认读取 `video.mp4`：

```bash
source .venv/bin/activate
python run.py --no-auth
```

指定视频：

```bash
python run.py video.mp4 --no-auth
```

常用参数示例：

```bash
python run.py video.mp4 --trail-mode selected --highlight-ids 12,15 --line-distance-m 10 --no-auth
python run.py video.mp4 --count-line 100,360,900,360 --speed-line-1 100,300,900,300 --speed-line-2 100,420,900,420 --no-auth
```

## 网页使用流程

1. 登录系统。
2. 进入“上传视频”页面。
3. 选择视频来源：本地视频文件或 FLV 流地址。
4. 系统提取首帧并生成 `preview.jpg`。
5. 在配置页拖动统计线和两条测速线。
6. 保存配置并开始分析。
7. 在实时预览页观察检测跟踪效果。
8. 单击车辆可高亮该车辆，双击车辆可取消高亮。
9. 分析完成后查看结果视频、CSV 文件和综合统计图。

FLV 流地址支持：

- HTTP-FLV：`http://host:port/live/stream.flv`
- HTTPS-FLV：`https://host/live/stream.flv`
- RTMP：`rtmp://host/app/stream`

HTTP/HTTPS 地址要求路径以 `.flv` 结尾；RTMP 地址可直接填写。流能否读取取决于当前 OpenCV/FFmpeg 环境是否支持对应协议和网络是否可达。若首帧暂时读取失败，系统会生成占位预览图，仍可进入配置页。

项目对网络流设置了打开和读取超时，默认由 [config.py](config.py) 中的 `WEB_STREAM_OPEN_TIMEOUT_MS` 和 `WEB_STREAM_READ_TIMEOUT_MS` 控制。这样在流不可达时不会长时间卡住上传页或分析线程。

在还没有合适 FLV 流时，可以先验证功能链路：

1. 上传页切换到“FLV 流地址”。
2. 输入形如 `http://127.0.0.1:8080/live/test.flv` 的地址。
3. 系统会接受格式正确的地址并创建任务。
4. 如果地址暂时不可达，配置页会显示占位预览；真正开始分析时任务会在超时后失败并显示错误原因。

拿到真实流后，建议先用 `ffplay` 或 VLC 打开同一个 URL，确认本机能播放，再填入网页端。

已完成任务可以重新配置线条并再次开始分析。重新分析会覆盖该任务的旧输出文件和旧统计字段。

## 网页点击选择车辆

桌面版 OpenCV 窗口可以通过 `cv2.setMouseCallback()` 直接收到鼠标坐标。网页端使用 `<img src="/video_feed/...">` 播放 MJPEG 流，浏览器点击不会自动进入 Python，因此项目增加了前后端桥接：

1. 前端监听实时预览图的单击和双击。
2. 前端按图片实际显示区域把浏览器坐标换算为原始视频坐标。
3. 前端把坐标发送到 `/tasks/<task_id>/click-select`。
4. 后端读取当前帧最新 tracks，并复用 `pick_track_id_from_click()` 选择目标。
5. 单击时把当前命中的原始 `track_id` 加入 `RUNTIME_CONTROLS[task_id]["highlight_ids"]`。
6. 主分析流程会把原始 `track_id` 映射到逻辑车辆对象，再反查当前帧实际框绘制高亮。
7. 双击时把 `track_id` 从 `highlight_ids` 移除。

这样既保留了原来的点击吸附逻辑，也能在 ByteTrack 短时换 ID 时尽量保持同一辆车的高亮连续。车辆靠近画面边缘离开后，系统不会强行把附近新框关联为同一辆车。

## 线条配置说明

配置页前端使用：

```json
{"center_x": 500, "center_y": 300, "length": 800, "angle": 0}
```

保存时转换为后端使用的端点坐标：

```json
{"x1": 100, "y1": 300, "x2": 900, "y2": 300}
```

网页显示尺寸通常小于原始视频尺寸，例如 1920×1080 的视频在网页中可能显示为 960×540。保存配置时会按比例换算回原始视频坐标，避免网页上看起来正确但实际分析时线条偏移。

任务配置保存在 `tasks.config_json`，主要字段包括：

```json
{
  "count_line": {"x1": 100, "y1": 360, "x2": 900, "y2": 360},
  "speed_line_1": {"x1": 100, "y1": 300, "x2": 900, "y2": 300},
  "speed_line_2": {"x1": 100, "y1": 420, "x2": 900, "y2": 420},
  "video_width": 1920,
  "video_height": 1080,
  "line_distance_m": 10.0,
  "speed_direction_mode": "vertical",
  "cross_tolerance": 8,
  "trail_mode": "all",
  "highlight_ids": [],
  "trail_len": 120,
  "qwen_review_enabled": false,
  "qwen_api_key": ""
}
```

## 统计与测速逻辑

车流统计：

- 车辆类别包括 bicycle、car、motorcycle、bus、truck。
- 每个目标由 ByteTrack 分配 `track_id`。
- 目标中心点穿过统计线时累计车流量。
- 每个 `track_id` 只计数一次，减少统计线附近抖动造成的重复计数。

双线测速：

- 目标中心点分别穿过测速线 1 和测速线 2。
- 两条测速线都被穿过后，按 `线间实际距离 / 时间差 * 3.6` 估算 km/h。
- 支持先过线 1 再过线 2，也支持反向通过。
- `MIN_SPEED_TIME_S` 过滤过短时间差。
- `MAX_SPEED_KMH` 过滤明显异常速度。
- 当前估算速度受透视误差影响，只用于演示和统计参考，不作为精确执法测速依据。

## 输出文件

网页任务输出位于：

```text
web_data/results/task_<user_id>_<timestamp>/
```

常见输出：

- `output.mp4`：叠加检测框、轨迹、统计线和测速线后的结果视频。
- `counts.csv`：逐帧累计车流量和速度摘要。
- `events.csv`：测速事件。
- `traffic.csv`：已测速车辆明细。
- `tracks.csv`：目标轨迹中心点。
- `trajectory_summary.json`：轨迹摘要。
- `anomalies.csv`：疑似异常风险提示事件。
- `llm_payload.json`：面向大模型复核的异常事件 JSON，主要引用异常时刻截图。
- `anomaly_snapshots/`：异常事件对应时刻截图。
- `qwen_snapshots/`：发送给千问的完整画面标注图和接口调试记录。
- `qwen_review_report.jsonl`：千问复核输入输出链路调试记录。
- `output_reports/综合统计图.png`：综合统计图。

命令行默认输出：

- `output/output.mp4`
- `output/counts.csv`
- `output/events.csv`
- `output/traffic.csv`
- `output/anomalies.csv`
- `output/llm_payload.json`
- `output/anomaly_snapshots/`
- `output/qwen_snapshots/`
- `output/qwen_review_report.jsonl`
- `output_reports/综合统计图.png`

## 异常风险提示

项目新增了独立的 [anomaly_detector.py](anomaly_detector.py)，用于在已有 YOLO11 + ByteTrack 检测跟踪结果上做轻量规则判断。该模块不重新训练模型，不改变主检测流程，输出定位是“疑似异常风险提示”。

当前只保留一类保守规则：

- `rear_end_risk`：先用横向重叠和纵向近距离规则筛选候选车辆对，再观察后续较长一段采样帧。候选后允许两车短暂运动或滑行，但需要在后续观察窗口内基本停止；触发后还会检查约 5 秒和 8 秒后的位移，若车辆已经正常离开或位移明显，则跳过千问复核。输出后结果视频中相关车辆框会用紫色标出，紫色框同样按逻辑车辆对象显示，不只依赖触发时刻的原始 ID。

输出文件：

```text
anomalies.csv
```

字段：

```text
frame_id,timestamp,event_type,track_id,related_track_id,class_name,description,evidence_path,evidence_paths,qwen_image_paths,evidence_offsets_sec,evidence_times_sec,qwen_result,qwen_reason,qwen_error
```

阈值集中在 [config.py](config.py)：

- `ANOMALY_REAR_END_HOLD_FRAMES`
- `ANOMALY_REAR_END_X_OVERLAP_RATIO`
- `ANOMALY_REAR_END_Y_GAP_RATIO`
- `ANOMALY_REAR_END_CONFIRM_FRAMES`
- `ANOMALY_REAR_END_STOP_WINDOW_FRAMES`
- `ANOMALY_REAR_END_STOP_MOVE_PX`
- `ANOMALY_REAR_END_CONTEXT_DISTANCE_PX`
- `ANOMALY_REAR_END_OTHER_MOTION_PX`
- `ANOMALY_REAR_END_OTHER_LATERAL_PX`
- `ANOMALY_MAX_SNAPSHOTS`
- `ANOMALY_EVIDENCE_OFFSETS_SEC`
- `ANOMALY_QWEN_OFFSETS_SEC`
- `ANOMALY_REAR_END_MOTION_CHECK_OFFSETS_SEC`
- `ANOMALY_REAR_END_MAX_POST_MOVE_PX`

追尾风险触发时会额外保存一组截图到 `anomaly_snapshots/`，默认时间点为确认触发前 3 秒、前 1 秒、确认触发时刻、确认触发后 1 秒、确认触发后 3 秒；千问复核会使用另一组完整画面标注图，默认时间点为触发前 3 秒、触发时刻、触发后 3 秒、触发后约 8 秒。短于 8 秒的视频不会把最后一帧当作真实 +8 秒，只会在提示词中标明“最后一帧替代”。`anomalies.csv` 的 `evidence_paths` 字段保存截图相对路径，`evidence_times_sec` 保存每张图对应的视频时间，`evidence_path` 保留触发时刻截图作为兼容字段。`ANOMALY_MAX_SNAPSHOTS=0` 表示不限制追尾证据截图事件数量。

千问复核使用另一套截图，保存在 `qwen_snapshots/`。这套图来自原始完整画面，只用矩形框标出需要复核的两辆车，不包含轨迹、统计文字和紫色提示框。这样让模型看到完整路况，同时问题聚焦在框选区域车辆是否追尾。对应路径写入 `qwen_image_paths`；网页端仍展示 `anomaly_snapshots/` 中带紫色框的正常证据图。

这些规则只基于图像空间距离和连续帧状态，不能替代真实交通违法检测或事故检测，适合作为系统中的异常风险提示功能说明。

## 大模型复核与调试

考虑到当前大模型对完整视频的直接理解和接收成本都不稳定，项目不会把 `output.mp4` 作为主要模型输入，而是把追尾风险前后截图和结构化事件整理为：

```text
llm_payload.json
```

该 JSON 包含：

- `summary`：事件数量、事件类型统计、截图数量。
- `events`：每条异常事件的类型、帧号、时间戳、车辆 ID、文字说明。
- `events[].evidence`：按时间顺序排列的网页展示截图路径、相对时间和视频时间。
- `events[].qwen_image_paths`：发送给千问的原始完整画面标注图路径。
- `events[].qwen_review`：千问复核结果，包含 `result / reason / error`。
- `llm_prompt_template`：后续接入大模型时可复用的保守复核提示词。

开发调试阶段可以通过接口查看同一份结构化 JSON：

- `GET /api/tasks/<task_id>/llm-payload`：返回当前任务的大模型输入 JSON。

千问复核已经接入到追尾风险处理链路，不需要额外提交外部违规数据。网页上传页和首帧配置页都提供“开启千问AI追尾复核”和 API key 输入框；关闭时不消耗额度，只显示规则检测结果。分析流程会在检测到疑似追尾后保存证据截图、执行后续位移复核、按任务配置调用千问，并把结果写入 `anomalies.csv` 和 `llm_payload.json`。FLV 流或长视频不需要等待全量分析结束；千问判定为 `true` 后，检测界面可以实时显示追尾风险报告。

结果页展示规则：

- 任务未开启千问复核时：展示规则检测到的疑似追尾风险，方便测试检测阈值和查看证据图。
- 任务开启千问复核且提供页面 API key 或环境变量 key 时：显示千问复核结果；规则层直接排除的正常离开事件不会作为追尾风险报告展示。
- 设置全局环境变量 `QWEN_REVIEW_ENABLED=0` 时：即使页面开启复核，也会完全跳过千问调用。

结果页下载区使用中文业务名称展示文件，例如“原始视频”“标注结果视频”“追尾风险记录表”“综合统计图”。开发阶段使用的 `llm_payload.json` 仍会生成，但不再作为结果页下载按钮展示。

千问 API key 可以在网页任务配置中输入，也可以通过环境变量提供：

```bash
export QWEN_API_KEY="你的key"
# 或使用 DashScope 官方环境变量：
export DASHSCOPE_API_KEY="你的key"
```

当前千问复核调用集中在 [qwen_reviewer.py](qwen_reviewer.py)。发送给千问的图片使用 OpenAI-compatible 多模态 `image_url` 输入，结果页给人看的追尾风险报告会展示正常 JPEG 图片，并标注每张图片对应的视频时间。模型会返回 `rear_end_risk=true/false`；如果接口不可用、图片路径无效或网络不可用，错误会写入 `qwen_error`，不会中断视频分析。

排查千问链路时优先看两个 JSONL 文件：

- `qwen_snapshots/qwen_review_debug.jsonl`：千问接口层输入、错误和原始返回。
- `qwen_review_report.jsonl`：主流程收到的千问复核输入输出，以及对应的 `anomalies.csv` / `llm_payload.json` 路径。

## 任务记录删除

控制台删除的是网页任务记录，不会自动删除上传视频和已经生成的结果文件。

运行中任务点击“取消并删除”时，系统会先向后台分析线程写入取消信号，再删除网页记录；后台线程会在下一次轮询运行时控制状态时停止。

如果需要清理磁盘空间，可以手动删除对应的：

```text
web_data/uploads/
web_data/results/
output/
output_reports/
runs/
```

## 关键配置

核心默认参数集中在 [config.py](config.py)：

- 模型：`YOLO_MODEL / TRACKER_CFG / YOLO_CONF / YOLO_IOU / IMGSZ`
- 类别：`CLASSES_SPEC / VEH_CONF / MOTO_CONF / PERSON_CONF`
- 跟踪：`STRIDE / TRAIL_LEN / LOST_HOLD / STABLE_KEEP_RATIO / TRAJECTORY_RELINK_*`
- 统计测速：`COUNT_LINE / LINE1 / LINE2 / LINE_DISTANCE_M / SPEED_DIRECTION_MODE / CROSS_TOLERANCE`
- 异常提示：`ANOMALY_*`
- 速度过滤：`MIN_SPEED_TIME_S / MAX_SPEED_KMH`
- 网页：`WEB_PORT / WEB_PREVIEW_ENABLED / WEB_ALLOWED_EXTENSIONS`
- 网络流：`WEB_STREAM_OPEN_TIMEOUT_MS / WEB_STREAM_READ_TIMEOUT_MS`
- 千问：`QWEN_REVIEW_ENABLED / QWEN_API_KEY_ENV / QWEN_API_URL / QWEN_MODEL`

## 常见问题

1. 网页打不开：确认命令中端口是 `5001`，浏览器访问 `http://127.0.0.1:5001`。
2. 模型加载失败：确认所选 `yolo11*.pt` 文件在项目根目录。
3. 本地视频上传失败：确认格式为 `mp4 / avi / mov / flv`，并且文件未损坏。
4. FLV 流无法读取：确认地址以 `.flv` 结尾或使用 `rtmp://`，并确认本机能访问该流。
5. 实时预览无画面：确认 `WEB_PREVIEW_ENABLED=True`，并查看终端是否有 OpenCV / Ultralytics 报错。
6. 点击车辆无效：等待检测框出现后再点击，点击位置尽量靠近目标框或目标中心。
7. 端口冲突：改用 `python web_app.py --host 127.0.0.1 --port 5002`。

## 项目说明

本版本以毕业设计演示为目标，保持 Flask + Jinja2 + 原生 JavaScript 的轻量结构，不引入 Vue / React。系统重点展示视频目标检测跟踪、车流统计、估算测速、网页任务管理和可视化结果导出。


工程来源为学校交付压缩包；代码、依赖列表和原有 GPL 许可证保留。公开仓库不包含毕业论文、模型权重、视频、数据库、历史任务产物或个人 API Key。
