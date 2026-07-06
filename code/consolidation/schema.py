from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


@dataclass
class FeedbackDecision:
    incident_id: str
    decision: str
    comment: str
    reviewer: str


@dataclass
class IncidentSummary:
    incident_id: str
    signals: list[str]
    decisive_evidence: list[str]
    root_cause: str
    case_summary: str


@dataclass
class ReflectionDecision:
    incident_id: str
    update_required: bool
    reflection: str
    subagent_calls: list[str]


@dataclass
class LocateCandidate:
    candidate_id: str
    candidate_type: str
    score: float
    summary: str
    payload: dict[str, Any]


@dataclass
class LocateResult:
    incident_id: str
    agent_type: str
    query: str
    candidates: list[LocateCandidate]
    notes: str




@dataclass
class Proposal:
    proposal_id: str
    incident_id: str
    agent_type: str
    knowledge_type: str
    action: str
    target_id: str | None
    object: dict[str, Any] | None
    rationale: str
    confidence: float
    proposal_source: str = "standard"


@dataclass
class ReviewDecision:
    proposal_id: str
    verdict: str
    reason: str


@dataclass
class ReviewResult:
    incident_id: str
    approved_proposals: list[Proposal]
    rejected_proposals: list[Proposal]
    decisions: list[ReviewDecision]
    review_notes: str


@dataclass
class RelationSynthesisUnlinkedTarget:
    proposal_id: str
    knowledge_type: str
    recovery_comment: str


@dataclass
class RelationSynthesisResult:
    incident_id: str
    relation_proposals: list[Proposal]
    unlinked_targets: list[RelationSynthesisUnlinkedTarget]
    notes: str


@dataclass
class PatternRecoveryResult:
    incident_id: str
    recovery_proposals: list[Proposal]
    triggered_from_proposal_ids: list[str]
    notes: str


@dataclass
class PendingApprovalBundle:
    incident_id: str
    status: str
    approved_proposals: list[Proposal]
    review_notes: str


@dataclass
class ApprovalDecision:
    proposal_id: str
    decision: str
    comment: str


@dataclass
class ApprovalResult:
    incident_id: str
    reviewer: str
    proposal_decisions: list[ApprovalDecision]


@dataclass
class CommitLog:
    commit_id: str
    incident_id: str
    target_store_dir: str
    approved_proposal_ids: list[str]
    rejected_proposal_ids: list[str]
    affected_files: list[str]
    committed_at: str


@dataclass
class RunMetadata:
    run_name: str
    model_name: str
    created_at: str
    root_store_dir: str
    active_store_dir: str
    resolved_active_store_dir: str
    log_dir: str
    resume_run: bool
    copy_on_run: bool
    immediate_runtime_refresh: bool


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value





