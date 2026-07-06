from __future__ import annotations

from pathlib import Path

from ltm.store import MemoryStore


class LongTermMemory:
    """Long-term memory store loaded from persistent memory files."""

    def __init__(self, store_dir: str | Path):
        self.store = MemoryStore.load(store_dir)


def build_long_term_memory(cfg_ltm: dict, store_dir: str | Path | None = None) -> LongTermMemory:
    resolved_store_dir = store_dir or cfg_ltm.get("store_dir", "ltm/knowledgebase")
    return LongTermMemory(store_dir=resolved_store_dir)

