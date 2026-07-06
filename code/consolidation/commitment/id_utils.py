from __future__ import annotations

import re
import uuid
from datetime import datetime


def new_uuid() -> str:
    return str(uuid.uuid4())


def is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return text.lower() or "run"


def make_timestamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y_%m%d_%H%M")


def make_model_run_name(model_name: str, dt: datetime | None = None) -> str:
    return f"{sanitize_name(model_name)}_{make_timestamp(dt)}"


def with_model_run_suffix(prefix: str | None, model_name: str, dt: datetime | None = None) -> str:
    suffix = make_model_run_name(model_name, dt)
    if prefix:
        return f"{sanitize_name(prefix)}_{suffix}"
    return suffix


def get_object_id_field(knowledge_type: str) -> str:
    mapping = {
        "pattern": "pattern_id",
        "procedure": "procedure_id",
        "case": "case_id",
    }
    if knowledge_type not in mapping:
        raise KeyError(f"Unsupported knowledge_type: {knowledge_type}")
    return mapping[knowledge_type]





