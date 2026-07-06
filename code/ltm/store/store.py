from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PatternRecord:
    pattern_id: str
    signals: list[str]
    root_cause: str
    content: str


@dataclass
class ProcedureRecord:
    procedure_id: str
    symptoms: list[str]
    content: str


@dataclass
class PatternProcedureEdge:
    pattern_id: str
    procedure_id: str
    weight: float


@dataclass
class CaseRecord:
    case_id: str
    symptoms: list[str]
    root_cause: str
    content: str


@dataclass
class PatternCaseEdge:
    pattern_id: str
    case_id: str
    weight: float


@dataclass
class MemoryStore:
    store_dir: Path
    patterns: list[PatternRecord]
    procedures: list[ProcedureRecord]
    cases: list[CaseRecord]
    pattern_procedure_edges: list[PatternProcedureEdge]
    pattern_case_edges: list[PatternCaseEdge]

    @classmethod
    def load(cls, store_dir: str | Path) -> "MemoryStore":
        root = Path(store_dir)
        return cls(
            store_dir=root,
            patterns=_read_patterns(root / "patterns.jsonl"),
            procedures=_read_procedures(root / "procedures.jsonl"),
            cases=_read_cases(root / "cases.jsonl"),
            pattern_procedure_edges=_read_pattern_procedure_edges(root / "pattern_procedure_edges.csv"),
            pattern_case_edges=_read_pattern_case_edges(root / "pattern_case_edges.csv"),
        )


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    else:
        items = [] if value is None else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _read_patterns(path: Path) -> list[PatternRecord]:
    patterns: list[PatternRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            signals = _as_list(payload.get("signals"))
            if not signals:
                raise ValueError(f"Pattern {payload.get('pattern_id')} must contain a non-empty signals list.")
            patterns.append(
                PatternRecord(
                    pattern_id=str(payload["pattern_id"]),
                    signals=signals,
                    root_cause=str(payload["root_cause"]).strip(),
                    content=str(payload.get("content") or "").strip(),
                )
            )
    return patterns


def _read_procedures(path: Path) -> list[ProcedureRecord]:
    procedures: list[ProcedureRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            procedures.append(
                ProcedureRecord(
                    procedure_id=str(payload["procedure_id"]),
                    symptoms=_as_list(payload.get("symptoms")),
                    content=str(payload.get("content") or "").strip(),
                )
            )
    return procedures


def _read_cases(path: Path) -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                CaseRecord(
                    case_id=str(payload["case_id"]),
                    symptoms=_as_list(payload.get("symptoms")),
                    root_cause=str(payload["root_cause"]).strip(),
                    content=str(payload.get("content") or "").strip(),
                )
            )
    return cases


def _read_pattern_procedure_edges(path: Path) -> list[PatternProcedureEdge]:
    edges: list[PatternProcedureEdge] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                PatternProcedureEdge(
                    pattern_id=row["pattern_id"],
                    procedure_id=row["procedure_id"],
                    weight=float(row["weight"]),
                )
            )
    return edges


def _read_pattern_case_edges(path: Path) -> list[PatternCaseEdge]:
    edges: list[PatternCaseEdge] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                PatternCaseEdge(
                    pattern_id=row["pattern_id"],
                    case_id=row["case_id"],
                    weight=float(row["weight"]),
                )
            )
    return edges




