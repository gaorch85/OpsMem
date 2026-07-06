from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from tools.log import LogTool
from tools.metric import MetricTool
from tools.shell import ShellTool, load_shell_records


class DiagnosticTool(Protocol):
    name: str

    def prompt_spec(self) -> str:
        ...

    def run(self, request: Dict[str, Any]) -> str:
        ...


@dataclass
class DiagnosticTools:
    case_id: int
    telemetry_dir: str
    metric_dir: str
    log_dir: str
    shell_path: str
    metric_names: List[str] = field(default_factory=list)
    default_ts: Optional[float] = None
    shell_records: List[Dict[str, str]] = field(default_factory=list)
    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.tools: Dict[str, DiagnosticTool] = {}
        self.register_tool(ShellTool(shell_records=self.shell_records, model_path=self.model_path))
        self.register_tool(LogTool(log_dir=self.log_dir))
        self.register_tool(
            MetricTool(
                metric_dir=self.metric_dir,
                metric_names=self.metric_names,
                default_ts=self.default_ts,
            )
        )

    @classmethod
    def from_case_resources(
        cls,
        case_id: int,
        resources: Dict[str, Any],
        model_path: Optional[str] = None,
    ) -> "DiagnosticTools":
        telemetry_dir = resources.get("telemetry_dir", "")
        metric_dir = os.path.join(telemetry_dir, "metrics")
        log_dir = os.path.join(telemetry_dir, "logs")
        shell_path = os.path.join(telemetry_dir, "shell", "shell.csv")
        metric_names = (
            sorted([name for name in os.listdir(metric_dir) if name.endswith(".csv")])
            if os.path.isdir(metric_dir)
            else []
        )

        return cls(
            case_id=case_id,
            telemetry_dir=telemetry_dir,
            metric_dir=metric_dir,
            log_dir=log_dir,
            shell_path=shell_path,
            metric_names=metric_names,
            default_ts=resources.get("default_ts"),
            shell_records=load_shell_records(shell_path),
            model_path=model_path,
        )

    def register_tool(self, tool: DiagnosticTool) -> None:
        self.tools[tool.name] = tool

    def build_tool_prompt(self) -> str:
        default_time_text = _format_timestamp(self.default_ts)
        tool_specs = "\n\n".join(tool.prompt_spec() for tool in self.tools.values())
        prompt = f"""
You can access offline diagnostic tools. These tools work on provided telemetry snapshots only (no real host access).
Default incident time (use if not specified): {default_time_text}.

Tool call format: always one JSON object with a "tool" field.

Available tools:
{tool_specs}

Only JSON format is accepted.
"""
        return prompt.strip()

    def dispatch_tool(self, raw_request: Any) -> str:
        request = self._normalize_tool_request(raw_request)
        tool_name = request.get("tool")
        if not tool_name:
            return f"Missing tool field in request: {raw_request}"

        tool = self.tools.get(tool_name)
        if tool is None:
            supported_tools = ", ".join(self.tools)
            return f"Unsupported tool '{tool_name}' in request: {raw_request}. Supported tools: {supported_tools}"
        return tool.run(request)

    def _normalize_tool_request(self, raw_request: Any) -> Dict[str, Any]:
        if isinstance(raw_request, dict):
            request = dict(raw_request)
        elif isinstance(raw_request, str):
            try:
                request = json.loads(raw_request)
            except Exception:
                return {"raw": raw_request}
            if not isinstance(request, dict):
                return {"raw": raw_request}
        else:
            return {"raw": str(raw_request)}

        tool = request.get("tool") or request.get("tool_name")
        request["tool"] = str(tool).strip().lower() if tool else ""
        return request


def _format_timestamp(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except Exception:
        return str(ts)




