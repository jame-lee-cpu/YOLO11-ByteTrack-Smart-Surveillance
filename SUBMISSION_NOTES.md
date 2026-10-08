# 学校交付说明

本目录为《基于视频目标跟踪的智能监控系统设计与实现》学校交付版本整理出的 GitHub 发布包。

## 包含内容

- Flask 网页端：登录、上传、任务管理、实时预览、结果页。
- YOLO11 + ByteTrack 检测跟踪主流程。
- 车流统计、双线估算测速、轨迹摘要。
- 疑似追尾风险规则检测。
- 千问/Qwen 视觉模型追尾风险复核接口。

## 未包含内容

- 毕业论文：包含个人信息，单独保存在私有仓库。

- Python 虚拟环境 `.venv/`。
- 历史上传视频和任务结果 `web_data/`。
- 本地数据库 `users.db`。
- 输出目录 `output/`、`output_reports/`、`runs/`。
- 大模型 API key。
- 全部模型权重；默认 `yolo11x.pt` 和快速测试用 `yolo11n.pt` 的获取方式见 README。

## 快速运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11x.pt')"
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

## 千问复核

上传视频页面和首帧配置页面都有“开启千问AI追尾复核”开关和 API key 输入框。关闭时只运行规则检测，不消耗千问额度；开启并填写 key 后，疑似追尾事件会发送完整标注截图给千问复核。

如需全局禁用千问调用，可用：

```bash
QWEN_REVIEW_ENABLED=0 WEB_DEBUG=0 python web_app.py --host 127.0.0.1 --port 5001
```

## 注意

追尾风险输出定位为“疑似风险提示”，不作为交通违法或事故责任的自动判定结论。
