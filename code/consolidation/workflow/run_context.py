from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from consolidation.commitment.id_utils import sanitize_name, with_model_run_suffix
from consolidation.schema import RunMetadata
from consolidation.commitment.store import MemoryConsolidationLogStore


DEFAULT_STORE_DIR = "ltm/knowledgebase"
DEFAULT_LOG_ROOT = "logs/OpsMem/consolidation/runs"
DEFAULT_COPY_ROOT = "ltm/runs"


def _resolve_path(base_dir: Path, raw_path: str | None, default: str) -> Path:
    candidate = Path(raw_path or default)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _to_metadata_path(base_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_store_paths(base_dir: Path, cfg_ltm: dict, consolidation_cfg: dict | None = None) -> tuple[Path, Path]:
    consolidation_cfg = consolidation_cfg if consolidation_cfg is not None else cfg_ltm.get("consolidation") or {}
    default_store = cfg_ltm.get("store_dir", DEFAULT_STORE_DIR)
    root_store_dir = _resolve_path(base_dir, consolidation_cfg.get("root_store_dir"), default_store)
    active_store_dir = _resolve_path(base_dir, consolidation_cfg.get("active_store_dir"), str(root_store_dir))
    return root_store_dir, active_store_dir


def prepare_consolidation_run_context(base_dir: Path, consolidation_cfg: dict, model_name: str) -> RunMetadata:
    resume_run = bool(consolidation_cfg.get("resume_run", False))
    copy_on_run = bool(consolidation_cfg.get("copy_on_run", True))
    immediate_runtime_refresh = bool(consolidation_cfg.get("immediate_runtime_refresh", False))

    root_store_dir = _resolve_path(base_dir, consolidation_cfg.get("root_store_dir"), DEFAULT_STORE_DIR)
    active_store_dir = _resolve_path(base_dir, consolidation_cfg.get("active_store_dir"), str(root_store_dir))
    if not active_store_dir.exists():
        raise FileNotFoundError(f"Active long-term memory store does not exist: {active_store_dir}")

    if resume_run:
        resume_log_dir = _resolve_path(base_dir, consolidation_cfg.get("resume_log_dir"), DEFAULT_LOG_ROOT)
        metadata_path = resume_log_dir / "run_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot resume run because run_metadata.json is missing: {resume_log_dir}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = RunMetadata(**payload)
        if metadata.model_name != model_name:
            raise ValueError(f"Resume run model mismatch: expected {metadata.model_name}, got {model_name}")
        if metadata.copy_on_run != copy_on_run:
            raise ValueError(
                f"Resume run copy_on_run mismatch: expected {metadata.copy_on_run}, got {copy_on_run}"
            )
        if _resolve_path(base_dir, consolidation_cfg.get("active_store_dir"), metadata.active_store_dir) != _resolve_path(base_dir, metadata.active_store_dir, metadata.active_store_dir):
            raise ValueError("Resume run active_store_dir does not match stored run metadata.")
        resolved_active_store_dir = _resolve_path(base_dir, metadata.resolved_active_store_dir, metadata.resolved_active_store_dir)
        if not resolved_active_store_dir.exists():
            raise FileNotFoundError(f"Resolved active store is missing for resumed run: {resolved_active_store_dir}")
        return metadata

    exact_run_name = consolidation_cfg.get("exact_run_name")
    custom_log_name = consolidation_cfg.get("log_name")
    custom_copy_name = consolidation_cfg.get("copied_store_name")
    if custom_log_name and custom_copy_name and custom_log_name != custom_copy_name:
        raise ValueError("log_name and copied_store_name must match when both are set.")

    run_name = sanitize_name(exact_run_name) if exact_run_name else with_model_run_suffix(custom_log_name or custom_copy_name, model_name)
    log_root_dir = _resolve_path(base_dir, consolidation_cfg.get("log_root_dir"), DEFAULT_LOG_ROOT)
    log_dir = log_root_dir / run_name
    if log_dir.exists():
        raise FileExistsError(f"Consolidation log directory already exists for new run: {log_dir}")

    if copy_on_run:
        copied_store_root = _resolve_path(base_dir, consolidation_cfg.get("copied_store_root"), DEFAULT_COPY_ROOT)
        resolved_active_store_dir = copied_store_root / run_name
        if resolved_active_store_dir.exists():
            raise FileExistsError(f"Copied long-term memory store already exists for new run: {resolved_active_store_dir}")
        shutil.copytree(active_store_dir, resolved_active_store_dir)
    else:
        resolved_active_store_dir = active_store_dir

    metadata = RunMetadata(
        run_name=run_name,
        model_name=model_name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        root_store_dir=_to_metadata_path(base_dir, root_store_dir),
        active_store_dir=_to_metadata_path(base_dir, active_store_dir),
        resolved_active_store_dir=_to_metadata_path(base_dir, resolved_active_store_dir),
        log_dir=_to_metadata_path(base_dir, log_dir),
        resume_run=resume_run,
        copy_on_run=copy_on_run,
        immediate_runtime_refresh=immediate_runtime_refresh,
    )
    store = MemoryConsolidationLogStore(log_dir)
    store.save_metadata(metadata)
    return metadata







