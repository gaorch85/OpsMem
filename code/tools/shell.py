from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from utils.llm import llm


class ShellTool:
    name = "shell"

    def __init__(
        self,
        *,
        shell_records: List[Dict[str, str]],
        model_path: Optional[str],
        max_snapshot_chars: int = 12000,
        enable_fuzzy: bool = True,
    ):
        self.shell_records = shell_records
        self.model_path = model_path
        self.max_snapshot_chars = max_snapshot_chars
        self.enable_fuzzy = enable_fuzzy

    def prompt_spec(self) -> str:
        return """
- Shell tool
  Request: {"tool": "shell", "command": "<shell command to inspect>"}
  Use it to inspect recorded shell.csv command snapshots.
  Example: {"tool": "shell", "command": "ps aux"}
""".strip()

    def run(self, request: Dict[str, Any]) -> str:
        command = self._extract_command(request)
        if not command:
            return "Shell tool needs a 'command' field."

        if not self.shell_records:
            return "No shell snapshots available."

        exact_match = self._match_exact(command)
        if exact_match:
            return exact_match

        if not self.enable_fuzzy:
            return "No exact shell snapshot matched your command."

        substring_match = self._match_substring(command)
        if substring_match:
            return substring_match

        evidence_texts = self._build_llm_evidence_texts()
        match = _llm_match_shell_snapshot(query=command, evidence_texts=evidence_texts, model_path=self.model_path)
        return match or "No relevant shell snapshot matched your command."

    @staticmethod
    def _extract_command(request: Any) -> str:
        if isinstance(request, dict):
            command = (
                request.get("command")
                or request.get("cmd")
                or request.get("query")
                or request.get("content")
                or ""
            )
        else:
            command = str(request)
        return command.strip()

    def _match_exact(self, command: str) -> str:
        for record in self.shell_records:
            if command == record["command"]:
                return _format_shell_snapshot(record)
        return ""

    def _match_substring(self, command: str) -> str:
        for record in self.shell_records:
            if command in record["command"]:
                return _format_shell_snapshot(record)
        return ""

    def _build_llm_evidence_texts(self) -> List[str]:
        evidence_texts: List[str] = []
        total_chars = 0
        for record in self.shell_records:
            text = f"command: {record['command']}\noutput:\n{record['output']}"
            remaining = self.max_snapshot_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                evidence_texts.append(text[:remaining])
                break
            evidence_texts.append(text)
            total_chars += len(text)
        return evidence_texts


def load_shell_records(shell_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not os.path.isfile(shell_path):
        return records
    with open(shell_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            command = (row.get("command") or "").strip()
            output = (row.get("output") or "").strip()
            if not command and not output:
                continue
            records.append({"command": command, "output": output})
    return records


def _format_shell_snapshot(record: Dict[str, str]) -> str:
    return f"[shell snapshot]\ncommand: {record['command']}\noutput:\n{record['output']}"


def _llm_match_shell_snapshot(query: str, evidence_texts: List[str], model_path: Optional[str] = None) -> str:
    system_prompt = """
You are a precise evidence retriever. Extract all text fragments from the provided shell snapshots that answer or relate to the query.

Rules:
1. Output only matched shell snapshot text, joined by newlines when multiple fragments match.
2. Return an empty string if no relevant fragment exists.
3. Preserve the original wording of matched evidence.
4. Do not add explanations, markdown, or summaries.
""".strip()

    user_prompt = f"""
Shell snapshots:
{evidence_texts}

Query:
{query}
""".strip()
    response, _meta = llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_path=model_path,
        temperature=0,
        max_tokens=2048,
        return_meta=True,
    )
    return response




