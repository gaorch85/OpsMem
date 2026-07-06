"""Long-term memory consolidation for OpsMem."""

from consolidation.consolidator import MemoryConsolidator
from consolidation.workflow import MemoryConsolidationPipeline, get_consolidation_feedback

__all__ = [
    "MemoryConsolidationPipeline",
    "MemoryConsolidator",
    "get_consolidation_feedback",
]

