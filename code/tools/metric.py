from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MetricTool:
    name = "metric"

    def __init__(self, *, metric_dir: str, metric_names: List[str], default_ts: Optional[float]):
        self.metric_dir = metric_dir
        self.metric_names = metric_names
        self.default_ts = default_ts

    def prompt_spec(self) -> str:
        available_metrics = ", ".join(self.metric_names) if self.metric_names else "None"
        default_time_text = _format_timestamp(self.default_ts)
        return f"""
- Metric tool
  Request: {{"tool": "metric", "metric_names": ["cpu.used.percent","memory.used.percent"], "timestamp": "<ISO or epoch>", "window_minutes": 10}}
  Use it to inspect metric values around the incident time. Choose up to 3 metrics per call; window_minutes is clamped to 1-60.
  Example: {{"tool": "metric", "metric_names": ["cpu.used.percent"], "timestamp": "{default_time_text}", "window_minutes": 10}}
  Available metric files: {available_metrics}
""".strip()

    def run(self, request: Dict[str, Any]) -> str:
        metric_names = self._normalize_metric_names(request.get("metric_names") or request.get("metrics") or [])
        if not metric_names:
            return "No metric names provided. Available metrics: " + ", ".join(self.metric_names[:20])

        center_ts = _parse_timestamp(
            request.get("timestamp") or request.get("ts") or request.get("time") or self.default_ts
        )
        window_min = self._normalize_window(request.get("window_minutes") or request.get("window") or 10)
        low_ts = center_ts - window_min * 60 if center_ts is not None else None
        high_ts = center_ts + window_min * 60 if center_ts is not None else None

        outputs: List[str] = []
        for name in metric_names:
            file_path = self._resolve_metric_file(name)
            if not file_path:
                outputs.append(f"[metric] {name}: file not found in telemetry metrics")
                continue

            rows = self._load_metric_rows(file_path)
            filtered = rows
            if center_ts is not None:
                filtered = [row for row in rows if row["ts"] is None or (low_ts <= row["ts"] <= high_ts)]
            if not filtered:
                filtered = sorted(rows, key=lambda row: row["ts"] or 0)[-5:]

            outputs.append(self._format_metric_result(file_path, filtered, center_ts, window_min))

        return "\n".join(outputs)

    @staticmethod
    def _normalize_metric_names(metric_names: Any) -> List[str]:
        if isinstance(metric_names, str):
            metric_names = [name.strip() for name in metric_names.split(",") if name.strip()]
        return list(metric_names)[:3]

    @staticmethod
    def _normalize_window(window: Any) -> int:
        try:
            window_min = int(window)
        except Exception:
            window_min = 10
        return max(1, min(window_min, 60))

    def _resolve_metric_file(self, name: str) -> Optional[str]:
        if not name:
            return None
        name_lower = name.lower()
        expected_file = name_lower if name_lower.endswith(".csv") else f"{name_lower}.csv"

        exact = [metric for metric in self.metric_names if metric.lower() == expected_file]
        if exact:
            return os.path.join(self.metric_dir, exact[0])

        candidates = [metric for metric in self.metric_names if name_lower in metric.lower()]
        if not candidates:
            return None
        return os.path.join(self.metric_dir, candidates[0])

    def _load_metric_rows(self, file_path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_timestamp(row.get("timestamp"))
                val_raw = row.get("value")
                try:
                    val = float(val_raw) if val_raw not in (None, "") else None
                except Exception:
                    val = None
                row_clean = {
                    key: value
                    for key, value in row.items()
                    if key not in {"expression", "normal_min", "normal_max", "application"}
                    and value not in (None, "")
                }
                if val is not None:
                    row_clean["value"] = val
                rows.append({"ts": ts, "value": val, "row": row_clean})
        return rows

    def _format_metric_result(
        self,
        file_path: str,
        rows: List[Dict[str, Any]],
        center_ts: Optional[float],
        window_min: int,
    ) -> str:
        values = [row["value"] for row in rows if row["value"] is not None]
        stats = ""
        if values:
            stats = (
                f"stats(min/avg/max): {min(values):.4f}/{sum(values) / len(values):.4f}/{max(values):.4f}; "
                f"samples={len(values)}"
            )

        samples = sorted(
            rows,
            key=lambda row: abs((row["ts"] or 0) - (center_ts or 0)) if center_ts is not None else (row["ts"] or 0),
        )[:10]
        sample_lines = [self._format_sample(sample) for sample in samples]
        return (
            f"[metric] {os.path.basename(file_path)} window=+/-{window_min}m around {_format_timestamp(center_ts)} "
            f"{stats}\n" + "\n".join(sample_lines)
        )

    def _format_sample(self, sample: Dict[str, Any]) -> str:
        row = sample["row"]
        ts_text = _format_timestamp(sample["ts"])
        value = row.get("value") or (f"{sample['value']:.4f}" if sample["value"] is not None else "n/a")
        instance = row.get("instance") or row.get("resourceTy") or row.get("resource") or ""
        return f"- {ts_text} value={value} instance={instance}"


def _parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts / 1000 if ts > 1e12 else ts
    if isinstance(value, str):
        try:
            ts_num = float(value)
            return ts_num / 1000 if ts_num > 1e12 else ts_num
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return None
    return None


def _format_timestamp(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return str(ts)





