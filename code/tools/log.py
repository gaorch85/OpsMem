from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class LogTool:
    name = "log"

    def __init__(self, *, log_dir: str):
        self.log_dir = log_dir

    def prompt_spec(self) -> str:
        return """
- Log tool
  Request: {"tool": "log", "keywords": ["ERROR","kubelet"], "limit": 15}
  Use it to inspect ERROR log lines. Keywords narrow the result; limit is clamped to 1-50.
  Example: {"tool": "log", "keywords": ["ERROR","kubelet"], "limit": 15}
""".strip()

    def run(self, request: Dict[str, Any]) -> str:
        keywords = self._normalize_keywords(request.get("keywords") or request.get("terms") or [])
        limit = self._normalize_limit(request.get("limit") or 15)

        entries: List[Tuple[float, str]] = []
        if not os.path.isdir(self.log_dir):
            return "Log directory missing."

        for log_file in os.listdir(self.log_dir):
            if not log_file.endswith(".csv"):
                continue
            entries.extend(self._collect_log_entries(log_file, keywords))

        if not entries:
            return "No matching ERROR log lines (filtered by keywords)."

        entries.sort(key=lambda item: item[0])
        picked = [entry[1] for entry in entries[-limit:]]
        return "\n".join(picked)

    def _collect_log_entries(self, log_file: str, keywords: List[str]) -> List[Tuple[float, str]]:
        entries: List[Tuple[float, str]] = []
        full_path = os.path.join(self.log_dir, log_file)
        with open(full_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = row.get("message", "")
                level = row.get("level", "")
                if not self._is_error_log(level, msg):
                    continue
                if not self._matches_keywords(msg, keywords):
                    continue
                ts = _parse_timestamp(row.get("timestamp"))
                entries.append(
                    (
                        ts if ts is not None else 0,
                        f"[log] {log_file} {_format_timestamp(ts)} level={level} message={msg}",
                    )
                )
        return entries

    @staticmethod
    def _normalize_keywords(keywords: Any) -> List[str]:
        if isinstance(keywords, str):
            return [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
        return list(keywords)

    @staticmethod
    def _normalize_limit(limit: Any) -> int:
        try:
            parsed_limit = int(limit)
        except Exception:
            parsed_limit = 15
        return max(1, min(parsed_limit, 50))

    @staticmethod
    def _is_error_log(level: str, message: str) -> bool:
        return "ERROR" in level.upper() or "ERROR" in message.upper()

    @staticmethod
    def _matches_keywords(text: str, keywords: List[str]) -> bool:
        if not keywords:
            return True
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)


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




