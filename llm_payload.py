"""大模型违规/异常数据载荷导出模块。

本模块只做数据整理，不直接调用任何大模型服务。这样可以先把系统输出稳定
成结构化 JSON，后续接入通义、文心、OpenAI 或本地模型时，只需要在接口层
读取该 JSON 并追加模型调用逻辑。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LLM_PAYLOAD_SCHEMA_VERSION = "1.0"


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_number_list(value: Any) -> List[float]:
    """解析 CSV/JSON 中保存的数字列表。"""
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            raw_items = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            raw_items = [item.strip() for item in value.split(";") if item.strip()]
    else:
        raw_items = []
    numbers = []
    for item in raw_items:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers


def read_anomaly_rows(anomalies_csv: str | Path) -> List[Dict[str, str]]:
    """读取 anomalies.csv，文件不存在时返回空列表。"""
    csv_path = Path(anomalies_csv)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_anomaly_event(row: Dict[str, Any], evidence_root: str | Path, event_index: int) -> Dict[str, Any]:
    """把 CSV 行转换为更适合大模型读取的单条事件。"""
    event_type = str(row.get("event_type") or "unknown")
    frame_id = _to_int(row.get("frame_id"), 0) or 0
    timestamp = _to_float(row.get("timestamp", row.get("timestamp_sec")), 0.0)
    track_id = _to_int(row.get("track_id", row.get("primary_track_id")))
    related_track_id = _to_int(row.get("related_track_id"))
    evidence_path = str(row.get("evidence_path") or "").strip()
    evidence_paths: List[str] = []
    raw_evidence_paths = row.get("evidence_paths")
    if isinstance(raw_evidence_paths, str) and raw_evidence_paths.strip():
        try:
            parsed_paths = json.loads(raw_evidence_paths)
            if isinstance(parsed_paths, list):
                evidence_paths.extend(str(item) for item in parsed_paths if str(item).strip())
        except json.JSONDecodeError:
            evidence_paths.extend(item.strip() for item in raw_evidence_paths.split(";") if item.strip())
    elif isinstance(raw_evidence_paths, list):
        evidence_paths.extend(str(item) for item in raw_evidence_paths if str(item).strip())
    if evidence_path and evidence_path not in evidence_paths:
        evidence_paths.insert(0, evidence_path)
    qwen_image_paths: List[str] = []
    raw_qwen_paths = row.get("qwen_image_paths")
    if isinstance(raw_qwen_paths, str) and raw_qwen_paths.strip():
        try:
            parsed_paths = json.loads(raw_qwen_paths)
            if isinstance(parsed_paths, list):
                qwen_image_paths.extend(str(item) for item in parsed_paths if str(item).strip())
        except json.JSONDecodeError:
            qwen_image_paths.extend(item.strip() for item in raw_qwen_paths.split(";") if item.strip())
    elif isinstance(raw_qwen_paths, list):
        qwen_image_paths.extend(str(item) for item in raw_qwen_paths if str(item).strip())
    qwen_result = row.get("qwen_result", "")
    qwen_reason = row.get("qwen_reason", "")
    qwen_error = row.get("qwen_error", "")
    evidence_offsets = _parse_number_list(row.get("evidence_offsets_sec"))
    evidence_times = _parse_number_list(row.get("evidence_times_sec"))
    evidence_items: List[Dict[str, Any]] = []
    raw_evidence = row.get("evidence")
    if isinstance(raw_evidence, list):
        evidence_items.extend(item for item in raw_evidence if isinstance(item, dict))
    for idx, item_path in enumerate(evidence_paths, start=1):
        abs_path = (Path(evidence_root) / item_path).resolve()
        evidence_items.append(
            {
                "type": "image",
                "path": item_path,
                "exists": abs_path.exists(),
                "frame_id": frame_id,
                "timestamp_sec": round(evidence_times[idx - 1], 3) if idx - 1 < len(evidence_times) else round(timestamp, 3),
                "relative_offset_sec": round(evidence_offsets[idx - 1], 3) if idx - 1 < len(evidence_offsets) else None,
                "sequence": idx,
                "note": "追尾风险复核截图，按发生前后时间顺序排列",
            }
        )

    return {
        "event_id": f"{event_type}_{frame_id}_{track_id or 'na'}_{related_track_id or 'na'}_{event_index}",
        "event_type": event_type,
        "risk_level": "medium" if event_type == "rear_end_risk" else "low",
        "frame_id": frame_id,
        "timestamp_sec": round(timestamp, 3),
        "primary_track_id": track_id,
        "related_track_id": related_track_id,
        "class_name": str(row.get("class_name") or "vehicle"),
        "description": str(row.get("description") or ""),
        "evidence": evidence_items,
        "qwen_image_paths": qwen_image_paths,
        "qwen_review": {
            "result": qwen_result,
            "reason": qwen_reason,
            "error": qwen_error,
            "image_paths": qwen_image_paths,
        },
        "model_review_hint": "请结合截图、事件类型和描述判断是否仅为风险提示，避免直接定性为违法或事故。",
    }


def build_llm_violation_payload(
    anomalies_csv: str | Path,
    output_video: str | Path | None = None,
    source_video: str | Path | None = None,
    task_id: int | None = None,
    max_events: int = 50,
) -> Dict[str, Any]:
    """生成面向大模型接口的 JSON 载荷。

    载荷以事件和截图为核心，不把完整视频作为主要输入。`output_video` 只作为
    回溯定位字段保留，便于人工或后续工具定位原始结果。
    """
    csv_path = Path(anomalies_csv)
    evidence_root = csv_path.parent
    rows = read_anomaly_rows(csv_path)
    rows = rows[: max(0, int(max_events))]
    events = [normalize_anomaly_event(row, evidence_root, idx + 1) for idx, row in enumerate(rows)]
    event_counter = Counter(event["event_type"] for event in events)
    evidence_count = sum(len(event.get("evidence", [])) for event in events)

    return {
        "schema_version": LLM_PAYLOAD_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_system": "YOLO11-ByteTrack-Smart-Surveillance",
        "task_id": task_id,
        "purpose": "疑似异常风险提示复核输入，不作为违法或事故自动判定结论",
        "input_preference": "image_snapshots",
        "summary": {
            "event_count": len(events),
            "event_types": dict(event_counter),
            "evidence_image_count": evidence_count,
            "truncated": len(read_anomaly_rows(csv_path)) > len(events),
        },
        "files": {
            "anomalies_csv": str(csv_path),
            "output_video_reference": "" if output_video is None else str(output_video),
            "source_video_reference": "" if source_video is None else str(source_video),
        },
        "events": events,
        "llm_prompt_template": (
            "你是交通监控复核助手。请基于事件描述和对应截图进行保守判断："
            "1. 说明是否存在风险迹象；2. 给出不确定因素；3. 不要把规则提示直接定性为违法或事故。"
        ),
    }


def export_llm_violation_payload(
    anomalies_csv: str | Path,
    output_json: str | Path,
    output_video: str | Path | None = None,
    source_video: str | Path | None = None,
    task_id: int | None = None,
    max_events: int = 50,
) -> Dict[str, Any]:
    """导出 llm_payload.json，并返回同一份字典。"""
    payload = build_llm_violation_payload(
        anomalies_csv=anomalies_csv,
        output_video=output_video,
        source_video=source_video,
        task_id=task_id,
        max_events=max_events,
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
