from __future__ import annotations

from pathlib import Path

from ltm.store import MemoryStore
from consolidation.commitment.id_utils import get_object_id_field, is_uuid


def validate_store_dir(store_dir: str | Path) -> None:
    root = Path(store_dir)
    store = MemoryStore.load(root)

    _assert_unique_ids(store.patterns, "pattern")
    _assert_unique_ids(store.procedures, "procedure")
    _assert_unique_ids(store.cases, "case")

    pattern_ids = {item.pattern_id for item in store.patterns}
    procedure_ids = {item.procedure_id for item in store.procedures}
    case_ids = {item.case_id for item in store.cases}

    for edge in store.pattern_procedure_edges:
        if edge.pattern_id not in pattern_ids:
            raise ValueError(f"Invalid pattern_procedure edge: missing pattern_id={edge.pattern_id}")
        if edge.procedure_id not in procedure_ids:
            raise ValueError(f"Invalid pattern_procedure edge: missing procedure_id={edge.procedure_id}")

    for edge in store.pattern_case_edges:
        if edge.pattern_id not in pattern_ids:
            raise ValueError(f"Invalid pattern_case edge: missing pattern_id={edge.pattern_id}")
        if edge.case_id not in case_ids:
            raise ValueError(f"Invalid pattern_case edge: missing case_id={edge.case_id}")


def _assert_unique_ids(records: list[object], knowledge_type: str) -> None:
    id_field = get_object_id_field(knowledge_type)
    seen: set[str] = set()
    for record in records:
        value = str(getattr(record, id_field))
        if not is_uuid(value):
            raise ValueError(f"{knowledge_type} id is not UUID: {value}")
        if value in seen:
            raise ValueError(f"Duplicate {knowledge_type} id detected: {value}")
        seen.add(value)




