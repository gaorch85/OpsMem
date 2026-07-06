"""MetaAgent, SubAgents, prompts, and approval helpers for consolidation."""

from consolidation.agents.approval import get_commit_approval
from consolidation.agents.meta_agent import MetaAgent
from consolidation.agents.subagents import MemoryConsolidationSubagents

__all__ = ["MemoryConsolidationSubagents", "MetaAgent", "get_commit_approval"]




