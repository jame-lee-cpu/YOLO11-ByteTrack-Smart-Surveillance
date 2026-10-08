"""千问/Qwen 追尾风险复核接口。

接口只在检测到疑似追尾风险后使用，不参与 YOLO11 检测和 ByteTrack 跟踪。
API key 从环境变量读取，避免把密钥写入代码或 README。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


REAR_END_REVIEW_PROMPT = """你是交通监控追尾风险复核助手。系统会提供 4 张按时间顺序排列的完整监控截图，通常为：
1. 规则触发前约 3 秒
2. 规则触发时刻
3. 触发后约 3 秒
4. 触发后约 8 秒

每张图是完整画面，不是局部裁剪。系统会用彩色矩形框和 TARGET 1 / TARGET 2 标签标出需要复核的一对目标车辆。你的注意力应主要集中在彩色框及其附近区域：先判断框内两辆车之间的前后关系、距离、接触/近接触、是否互相遮挡或压在一起。画面中其他车辆、车流和道路只作为辅助背景，不要因为未框选车辆的正常运动而否定框内两车的追尾风险，也不要把未框选车辆当成事件主体。

注意：如果视频在后续证据点前结束，某些后续图片可能使用视频最后一帧代替。最后一帧只表示视频结束时的画面，不代表车辆真实持续静止到了 +3 秒或 +8 秒。因此，当后续图片是最后一帧替代、或实际观察时间不足时，不得仅凭“后续图像变化不明显”判断为追尾风险。

你的任务不是认定事故或责任，只判断该事件是否值得保留为“疑似追尾风险提示”。

请优先判断触发时刻本身是否有足够强的视觉证据：
- 触发前约 3 秒的图只作为辅助，用于判断两车是否已经接近、是否存在靠近趋势或本来就异常贴近；
- 两车是否明确处于同一车道或同一运动方向的前后关系；
- 两车是否已经明显接触、几乎接触，或异常贴近；远景监控里不要求看清碰撞细节，只要框内两车贴近程度明显异常即可作为强证据；
- 两个彩色框是否相互贴近、重叠、前后压住，或其中一车车头/车尾已经贴到另一车；
- 是否能排除并排行驶、相邻车道、透视重叠、远距离小目标造成的误判。

请按以下标准判断：
- **返回 true**：触发前约 3 秒到触发时刻能看出两车接近或已经异常贴近；触发时刻能看出两车处于前后关系或高度疑似同向前后关系；两车距离异常近，疑似接触、几乎接触，或贴近程度明显超过正常跟车距离；彩色框附近能看出两车已经贴住、压住、重叠明显，或车头/车尾几乎相接；后续图像进一步支持两车保持异常近距离或停住。即使后续证据不足，只要触发时刻框内局部区域本身已经有较强证据，也可以返回 true。
- **返回 false**：车辆距离摄像头较远、目标占比很小、细节不足；只是透视造成重叠，不能确认同车道前后关系；只是并排行驶或相邻车道接近；触发时刻没有明显接触或近接触；后续图片是视频最后一帧替代，不能证明车辆持续停住；只能看到短时间内运动不明显，不能确认追尾风险；证据主要依赖 +8 秒图片，但 +8 秒图片实际是视频最后一帧。

特别规则：如果证据信息中说明“视频在后续证据点前结束”或“使用最后一帧作为后续证据”，请更加保守，但不要一概否定。除非触发时刻已经有明确或较强的接触/近接触证据，否则返回 false。

reason 要明确说明判断依据：
- 如果返回 false，应说明“后续证据不足”或“最后一帧不能证明持续停住”等原因。
- 如果返回 true，应说明“触发时刻已经存在明确接触/近接触证据”，不能只写“后续未移动”。

请仅返回 JSON，不要输出其他文字。格式如下：
{"rear_end_risk": true或false, "reason": "一句话说明理由"}
"""


def _image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _validate_image_paths(image_paths: List[str | Path]) -> Tuple[bool, str, List[str]]:
    """检查千问复核输入截图是否真实存在，便于排查报告输入输出链路。"""
    if not image_paths:
        return False, "未收到任何千问复核截图路径", []

    normalized_paths = [str(Path(path)) for path in image_paths]
    missing_paths = [path for path in normalized_paths if not Path(path).is_file()]
    if missing_paths:
        return False, f"存在 {len(missing_paths)} 张复核截图路径无效: {missing_paths[:3]}", normalized_paths

    return True, "", normalized_paths


def _guess_debug_report_path(image_paths: List[str]) -> str:
    """根据第一张输入截图推断千问调试报告路径。"""
    if not image_paths:
        return ""
    return str(Path(image_paths[0]).parent / "qwen_review_debug.jsonl")


def _append_debug_record(record: Dict[str, Any]) -> None:
    """追加写入千问调试记录，避免上游报告未写入时无法排查。"""
    debug_report_path = str(record.get("debug_report_path") or "")
    if not debug_report_path:
        return

    try:
        path = Path(debug_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_record = {
            "enabled": record.get("enabled", ""),
            "rear_end_risk": record.get("rear_end_risk", ""),
            "reason": record.get("reason", ""),
            "error": record.get("error", ""),
            "raw_response": record.get("raw_response", ""),
            "image_count": record.get("image_count", 0),
            "image_paths": record.get("image_paths", []),
            "api_url": record.get("api_url", ""),
            "model": record.get("model", ""),
            "created_at": record.get("created_at", ""),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _with_debug_record(result: Dict[str, Any], debug_report_path: str) -> Dict[str, Any]:
    """补充通用调试字段并写入 jsonl。"""
    result.setdefault("debug_report_path", debug_report_path)
    result.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    result.setdefault("raw_response", "")
    result.setdefault("api_url", "")
    result.setdefault("model", "")
    result.setdefault("image_count", len(result.get("image_paths") or []))
    _append_debug_record(result)
    return result


def _format_evidence_metadata(evidence_metadata: List[Dict[str, Any]] | None) -> str:
    if not evidence_metadata:
        return ""
    lines = ["\n证据信息："]
    for idx, item in enumerate(evidence_metadata, start=1):
        label = str(item.get("label") or f"第{idx}张")
        planned_offset = item.get("planned_offset_sec", "")
        actual_offset = item.get("actual_offset_sec", "")
        timestamp = item.get("timestamp_sec", "")
        final_note = "，使用视频最后一帧替代" if item.get("used_final_frame") else ""
        lines.append(
            f"- 第{idx}张：{label}，计划 offset={planned_offset}s，"
            f"实际 offset={actual_offset}s，视频时间={timestamp}s{final_note}"
        )
    if any(item.get("used_final_frame") for item in evidence_metadata):
        lines.append("注意：上述部分后续证据由视频最后一帧替代，不代表车辆真实持续静止到该计划 offset。")
    return "\n".join(lines)


def _build_multimodal_content(image_paths: List[str | Path], evidence_metadata: List[Dict[str, Any]] | None = None) -> list:
    """按 OpenAI-compatible 多模态格式组织千问视觉模型输入。"""
    text = "\n".join(
        part
        for part in [REAR_END_REVIEW_PROMPT, _format_evidence_metadata(evidence_metadata)]
        if part
    )
    content = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
    return content


def _parse_review_result(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        data = content
    else:
        raw = str(content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        if not raw:
            return {
                "rear_end_risk": False,
                "reason": "千问未返回有效内容",
                "error": "千问返回内容为空",
                "raw_response": content,
            }
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "rear_end_risk": False,
                "reason": "千问返回内容不是合法 JSON",
                "error": "千问返回内容不是合法 JSON",
                "raw_response": content,
            }

    if isinstance(data, bool):
        return {
            "rear_end_risk": data,
            "reason": "",
            "error": "",
            "raw_response": content,
        }
    if isinstance(data, str):
        data = {"rear_end_risk": data.strip().lower() == "true", "reason": ""}

    rear_end_risk = data.get("rear_end_risk", False)
    if isinstance(rear_end_risk, bool):
        pass
    elif isinstance(rear_end_risk, str):
        rear_end_risk = rear_end_risk.strip().lower() == "true"
    else:
        rear_end_risk = bool(rear_end_risk)

    return {
        "rear_end_risk": rear_end_risk,
        "reason": str(data.get("reason") or ""),
        "error": "",
        "raw_response": content,
    }


def review_rear_end_with_qwen(
    image_paths: List[str | Path],
    api_key_env: str,
    api_url: str,
    model: str,
    timeout_sec: int = 30,
    max_prompt_chars: int = 650000,
    evidence_metadata: List[Dict[str, Any]] | None = None,
    api_key_value: str | None = None,
) -> Dict[str, Any]:
    """把追尾证据截图发送给千问视觉模型，返回 true/false 复核结果。"""
    paths_ok, paths_error, normalized_image_paths = _validate_image_paths(image_paths)
    debug_report_path = _guess_debug_report_path(normalized_image_paths)
    if not paths_ok:
        return _with_debug_record(
            {
                "enabled": True,
                "rear_end_risk": False,
                "reason": "千问复核输入截图路径无效，已跳过复核",
                "error": paths_error,
                "image_count": len(image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": "",
            },
            debug_report_path,
        )

    api_key = str(api_key_value or "").strip() or os.environ.get(api_key_env, "").strip()
    if not api_key and api_key_env != "DASHSCOPE_API_KEY":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        return _with_debug_record(
            {
                "enabled": False,
                "rear_end_risk": "",
                "reason": "",
                "error": f"环境变量 {api_key_env} 或 DASHSCOPE_API_KEY 未设置，已跳过千问复核",
                "image_count": len(normalized_image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": "",
            },
            debug_report_path,
        )

    text_part = "\n".join(
        part
        for part in [REAR_END_REVIEW_PROMPT, _format_evidence_metadata(evidence_metadata)]
        if part
    )
    if len(text_part) > int(max_prompt_chars):
        return _with_debug_record(
            {
                "enabled": True,
                "rear_end_risk": "",
                "reason": "",
                "error": f"千问文本输入过大，已跳过复核: {len(text_part)} chars > {int(max_prompt_chars)} chars",
                "image_count": len(normalized_image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": "",
            },
            debug_report_path,
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _build_multimodal_content(image_paths, evidence_metadata=evidence_metadata),
            }
        ],
        "temperature": 0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=int(timeout_sec)) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return _with_debug_record(
            {
                "enabled": True,
                "rear_end_risk": "",
                "reason": "",
                "error": f"千问 HTTP {exc.code}: {detail}",
                "image_count": len(normalized_image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": detail,
            },
            debug_report_path,
        )
    except Exception as exc:
        return _with_debug_record(
            {
                "enabled": True,
                "rear_end_risk": "",
                "reason": "",
                "error": f"千问请求失败: {exc}",
                "image_count": len(normalized_image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": "",
            },
            debug_report_path,
        )

    def _extract_content(response: Any) -> Any:
        if isinstance(response, dict):
            if "choices" in response:
                try:
                    choice = response["choices"][0]
                except (IndexError, TypeError):
                    return None
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        return choice["message"].get("content")
                    return choice.get("content")
            if "rear_end_risk" in response:
                return response
        return None

    content_text = _extract_content(response_data)
    if content_text is None:
        return _with_debug_record(
            {
                "enabled": True,
                "rear_end_risk": "",
                "reason": "",
                "error": f"千问返回结构异常: {response_data}",
                "image_count": len(normalized_image_paths),
                "image_paths": normalized_image_paths,
                "api_url": api_url,
                "model": model,
                "raw_response": response_data,
            },
            debug_report_path,
        )

    parsed = _parse_review_result(content_text)
    return _with_debug_record(
        {
            "enabled": True,
            "rear_end_risk": parsed["rear_end_risk"],
            "reason": parsed["reason"],
            "error": parsed.get("error", ""),
            "raw_response": parsed.get("raw_response", ""),
            "image_count": len(normalized_image_paths),
            "image_paths": normalized_image_paths,
            "api_url": api_url,
            "model": model,
        },
        debug_report_path,
    )
