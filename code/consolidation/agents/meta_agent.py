from __future__ import annotations

from typing import Any

from consolidation.agents.prompts import (
    build_reflection_prompts,
    build_relation_synthesis_prompts,
    build_review_prompts,
    build_summary_prompts,
)
from consolidation.commitment.id_utils import new_uuid
from consolidation.schema import (
    IncidentSummary,
    LocateResult,
    Proposal,
    ReflectionDecision,
    RelationSynthesisResult,
    RelationSynthesisUnlinkedTarget,
    ReviewDecision,
    ReviewResult,
    to_jsonable,
)
from utils.llm import llm, parse_json_response


ALLOWED_SUBAGENTS = ("PatternAgent", "CaseAgent", "ProcedureAgent")


class MetaAgent:
    """Coordinator for long-term memory consolidation."""

    def __init__(self, llm_args: dict[str, Any] | None = None, config: dict[str, Any] | None = None):
        self.llm_args = llm_args or {}
        self.config = config or {}

    def summarize(self, memory_package: dict) -> IncidentSummary:
        self._require_llm_args("memory package summarization")
        system_prompt, user_prompt = build_summary_prompts(memory_package)
        response, _ = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args.get("temperature", 0.2),
            max_tokens=self.llm_args.get("max_tokens", 2048),
            return_meta=True,
        )
        payload = parse_json_response(response)
        return IncidentSummary(
            incident_id=payload["incident_id"],
            signals=payload.get("signals") or [],
            decisive_evidence=payload.get("decisive_evidence") or [],
            root_cause=payload.get("root_cause", ""),
            case_summary=payload.get("case_summary", ""),
        )

    def reflect(self, summary: IncidentSummary, related_memory: dict) -> ReflectionDecision:
        self._require_llm_args("memory reflection")
        system_prompt, user_prompt = build_reflection_prompts(to_jsonable(summary), related_memory)
        response, _ = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args.get("temperature", 0.2),
            max_tokens=self.llm_args.get("max_tokens", 2048),
            return_meta=True,
        )
        payload = parse_json_response(response)
        return ReflectionDecision(
            incident_id=payload["incident_id"],
            update_required=bool(payload.get("update_required", False)),
            reflection=payload.get("reflection", ""),
            subagent_calls=self._filter_subagent_calls(payload.get("subagent_calls") or []),
        )

    def review(self, summary: dict, reflection: dict, proposals: list[Proposal], related_memory: dict) -> ReviewResult:
        if not proposals:
            return ReviewResult(
                incident_id=summary["incident_id"],
                approved_proposals=[],
                rejected_proposals=[],
                decisions=[],
                review_notes="No proposals to review.",
            )

        self._require_llm_args("proposal review")
        system_prompt, user_prompt = build_review_prompts(
            summary,
            reflection,
            to_jsonable(proposals),
            related_memory,
        )
        response, _ = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args.get("temperature", 0.2),
            max_tokens=self.llm_args.get("max_tokens", 2048),
            return_meta=True,
        )
        payload = parse_json_response(response)
        decisions = payload.get("decisions") or []
        decision_map = {item["proposal_id"]: item for item in decisions if item.get("proposal_id")}

        approved: list[Proposal] = []
        rejected: list[Proposal] = []
        normalized_decisions: list[ReviewDecision] = []
        for proposal in proposals:
            raw = decision_map.get(proposal.proposal_id, {})
            verdict = raw.get("verdict", "reject")
            reason = raw.get("reason", "No reason provided.")
            normalized_decisions.append(ReviewDecision(proposal_id=proposal.proposal_id, verdict=verdict, reason=reason))
            if verdict == "approve":
                approved.append(proposal)
            else:
                rejected.append(proposal)
        return ReviewResult(
            incident_id=summary["incident_id"],
            approved_proposals=approved,
            rejected_proposals=rejected,
            decisions=normalized_decisions,
            review_notes=payload.get("review_notes", ""),
        )

    def synthesize_relations(
        self,
        summary: IncidentSummary,
        pattern_locate_result: LocateResult,
        approved_node_proposals: list[Proposal],
    ) -> RelationSynthesisResult:
        if not approved_node_proposals:
            return RelationSynthesisResult(
                incident_id=summary.incident_id,
                relation_proposals=[],
                unlinked_targets=[],
                notes="No approved node proposals to link.",
            )

        self._require_llm_args("relation synthesis")
        system_prompt, user_prompt = build_relation_synthesis_prompts(
            summary=to_jsonable(summary),
            pattern_locate_result=to_jsonable(pattern_locate_result),
            approved_node_proposals=[self._proposal_to_prompt_dict(proposal) for proposal in approved_node_proposals],
        )
        response, _ = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args.get("temperature", 0.2),
            max_tokens=self.llm_args.get("max_tokens", 2048),
            return_meta=True,
        )
        payload = parse_json_response(response)
        relation_proposals = [
            self._normalize_relation_proposal(summary.incident_id, item)
            for item in (payload.get("relation_proposals") or [])
            if item
        ]
        unlinked_targets = [
            RelationSynthesisUnlinkedTarget(
                proposal_id=str(item["proposal_id"]),
                knowledge_type=str(item["knowledge_type"]),
                recovery_comment=str(item.get("recovery_comment", "")).strip() or "No matching pattern could be synthesized.",
            )
            for item in (payload.get("unlinked_targets") or [])
            if item.get("proposal_id") and item.get("knowledge_type")
        ]
        return RelationSynthesisResult(
            incident_id=summary.incident_id,
            relation_proposals=self._dedupe_relation_proposals(relation_proposals),
            unlinked_targets=unlinked_targets,
            notes=payload.get("notes", ""),
        )

    def _filter_subagent_calls(self, subagent_calls: list[str]) -> list[str]:
        allowed_flags = {
            "PatternAgent": self.config.get("allow_pattern_agent", True),
            "CaseAgent": self.config.get("allow_case_agent", True),
            "ProcedureAgent": self.config.get("allow_procedure_agent", True),
        }
        filtered: list[str] = []
        for name in subagent_calls:
            if name not in ALLOWED_SUBAGENTS:
                continue
            if not allowed_flags.get(name, False):
                continue
            if name not in filtered:
                filtered.append(name)
        return filtered

    def _normalize_relation_proposal(self, incident_id: str, payload: dict) -> Proposal:
        object_payload = dict(payload.get("object") or {})
        return Proposal(
            proposal_id=payload.get("proposal_id") if isinstance(payload.get("proposal_id"), str) else new_uuid(),
            incident_id=incident_id,
            agent_type="MetaAgent",
            knowledge_type="relation",
            action="create",
            target_id=None,
            object=object_payload,
            rationale=payload.get("rationale", ""),
            confidence=float(payload.get("confidence", 0.0)),
        )

    def _dedupe_relation_proposals(self, proposals: list[Proposal]) -> list[Proposal]:
        deduped: list[Proposal] = []
        seen: set[tuple[str, str, str]] = set()
        for proposal in proposals:
            if proposal.knowledge_type != "relation" or not proposal.object:
                continue
            key = (
                str(proposal.object.get("edge_type") or ""),
                str(proposal.object.get("source_id") or ""),
                str(proposal.object.get("target_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(proposal)
        return deduped

    def _require_llm_args(self, operation: str) -> None:
        if not self.llm_args:
            raise ValueError(f"MetaAgent requires configured LLM args for {operation}.")

    @staticmethod
    def _proposal_to_prompt_dict(proposal: Proposal) -> dict:
        return {
            "proposal_id": proposal.proposal_id,
            "incident_id": proposal.incident_id,
            "agent_type": proposal.agent_type,
            "knowledge_type": proposal.knowledge_type,
            "action": proposal.action,
            "target_id": proposal.target_id,
            "object": proposal.object,
            "rationale": proposal.rationale,
            "confidence": proposal.confidence,
            "proposal_source": proposal.proposal_source,
        }
