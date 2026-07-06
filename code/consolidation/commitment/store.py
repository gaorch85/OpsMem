from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from consolidation.schema import to_jsonable


class MemoryConsolidationLogStore:
    def __init__(self, log_dir: str | Path, detailed_logging: bool = False):
        self.log_dir = Path(log_dir)
        self.detailed_logging = detailed_logging
        self._detail_subdirs = {
            "diagnosis_artifacts": self.log_dir / "diagnosis_artifacts",
            "feedback": self.log_dir / "feedback",
            "memory_packages": self.log_dir / "memory_packages",
            "summaries": self.log_dir / "summaries",
            "reflections": self.log_dir / "reflections",
            "locate_results": self.log_dir / "locate_results",
            "proposals": self.log_dir / "proposals",
            "reviews": self.log_dir / "reviews",
            "relation_synthesis": self.log_dir / "relation_synthesis",
            "pattern_recovery": self.log_dir / "pattern_recovery",
            "pending_commit": self.log_dir / "pending_commit",
            "final_approvals": self.log_dir / "final_approvals",
            "approval_rejections": self.log_dir / "approval_rejections",
            "commit_logs": self.log_dir / "commit_logs",
            "committed_bundles": self.log_dir / "committed_bundles",
        }
        self._compact_subdirs = {
            "cases": self.log_dir / "cases",
            "commits": self.log_dir / "commits",
        }
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        subdirs = self._detail_subdirs if self.detailed_logging else self._compact_subdirs
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def path_for(self, bucket: str, file_name: str) -> Path:
        if self.detailed_logging:
            return self._detail_subdirs[bucket] / file_name
        if bucket == "commit_logs":
            return self._compact_subdirs["commits"] / file_name
        return self._case_record_path(file_name)

    def save_json(self, bucket: str, file_name: str, payload: Any) -> Path:
        if self.detailed_logging:
            path = self.path_for(bucket, file_name)
            path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            return path

        case_path = self._merge_case_record(bucket, file_name, payload)
        if bucket == "commit_logs":
            commit_path = self._compact_subdirs["commits"] / file_name
            commit_path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return case_path

    def load_json(self, bucket: str, file_name: str) -> Any:
        path = self.path_for(bucket, file_name)
        return json.loads(path.read_text(encoding="utf-8"))

    def metadata_path(self) -> Path:
        return self.log_dir / "run_metadata.json"

    def save_metadata(self, payload: Any) -> Path:
        path = self.metadata_path()
        path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_metadata(self) -> Any:
        return json.loads(self.metadata_path().read_text(encoding="utf-8"))

    def _merge_case_record(self, bucket: str, file_name: str, payload: Any) -> Path:
        path = self._case_record_path(file_name)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = {"incident_id": self._incident_id_from_filename(file_name)}

        key = self._compact_record_key(bucket, file_name)
        value = to_jsonable(payload)
        if key in record and record[key] != value:
            existing = record[key]
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(value)
            record[key] = existing
        else:
            record[key] = value

        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _case_record_path(self, file_name: str) -> Path:
        incident_id = self._incident_id_from_filename(file_name)
        return self._compact_subdirs["cases"] / f"{incident_id}.json"

    @staticmethod
    def _incident_id_from_filename(file_name: str) -> str:
        stem = Path(file_name).stem
        match = re.match(r"(case_\d+)", stem)
        if match:
            return match.group(1)
        return stem

    @staticmethod
    def _compact_record_key(bucket: str, file_name: str) -> str:
        stem = Path(file_name).stem
        incident_id = MemoryConsolidationLogStore._incident_id_from_filename(file_name)
        suffix = stem.removeprefix(incident_id).strip("_")
        if not suffix:
            return bucket
        return f"{bucket}.{suffix}"




