from __future__ import annotations

from typing import Any

from cmr import CasePropagation, PatternActivation, ProcedurePropagation
from consolidation.agents.prompts import (
    build_locate_filter_prompts,
    build_pattern_recovery_prompts,
    build_proposal_prompts,
)
from consolidation.commitment.id_utils import get_object_id_field, is_uuid, new_uuid
from consolidation.schema import LocateCandidate, LocateResult, Proposal
from utils.llm import llm, parse_json_response


class MemoryConsolidationSubagents:
    def __init__(self, long_term_memory=None, cmr_engine=None, llm_args: dict[str, Any] | None = None):
        self.long_term_memory = long_term_memory
        self.cmr_engine = cmr_engine
        self.llm_args = llm_args or {}

    def locate(self, agent_type: str, summary: dict) -> LocateResult:
        incident_id = summary["incident_id"]
        query = " | ".join(summary.get("signals") or [summary.get("root_cause", "")]).strip()
        raw_candidates = self._retrieve_candidates(agent_type, summary)
        filtered_candidates = self._filter_candidates(agent_type, summary, raw_candidates)
        notes = f"{agent_type} retained {len(filtered_candidates)} related objects after CMR retrieval and LLM filtering."
        return LocateResult(
            incident_id=incident_id,
            agent_type=agent_type,
            query=query,
            candidates=filtered_candidates,
            notes=notes,
        )

    def propose(self, agent_type: str, summary: dict, locate_result: LocateResult) -> list[Proposal]:
        return self._propose_by_llm(agent_type, summary, locate_result)

    def recover_patterns(
        self,
        summary: dict,
        target_proposal: Proposal,
        pattern_locate_result: LocateResult,
        recovery_comment: str,
        rejected_pattern_reviews: list[dict],
    ) -> list[Proposal]:
        return self._recover_patterns_by_llm(
            summary=summary,
            target_proposal=target_proposal,
            pattern_locate_result=pattern_locate_result,
            recovery_comment=recovery_comment,
            rejected_pattern_reviews=rejected_pattern_reviews,
        )

    def _retrieve_candidates(self, agent_type: str, summary: dict) -> list[LocateCandidate]:
        if self.cmr_engine is None:
            return []

        resonance_result = self.cmr_engine.run(summary.get("signals") or [])
        if agent_type == "PatternAgent":
            return [self._pattern_candidate(hit) for hit in resonance_result.pattern_activation]
        if agent_type == "ProcedureAgent":
            return [self._procedure_candidate(hit) for hit in resonance_result.memory_propagation.procedures]
        if agent_type == "CaseAgent":
            return [self._case_candidate(hit) for hit in resonance_result.memory_propagation.cases]
        return []

    def _filter_candidates(self, agent_type: str, summary: dict, candidates: list[LocateCandidate]) -> list[LocateCandidate]:
        if not candidates:
            return []
        return self._filter_candidates_by_llm(agent_type, summary, candidates)

    def _filter_candidates_by_llm(self, agent_type: str, summary: dict, candidates: list[LocateCandidate]) -> list[LocateCandidate]:
        self._require_llm_args("candidate filtering")
        serialized = [candidate.__dict__ for candidate in candidates]
        system_prompt, user_prompt = build_locate_filter_prompts(agent_type, summary, serialized)
        response, _ = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args.get("temperature", 0.2),
            max_tokens=self.llm_args.get("max_tokens", 2048),
            return_meta=True,
        )
        payload = parse_json_response(response)
        related_ids = [str(item).strip() for item in (payload.get("related_candidate_ids") or []) if str(item).strip()]
        candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
        filtered: list[LocateCandidate] = []
        for candidate_id in related_ids:
            candidate = candidate_map.get(candidate_id)
            if candidate is not None and candidate not in filtered:
                filtered.append(candidate)
        return filtered

    def _pattern_candidate(self, hit: PatternActivation) -> LocateCandidate:
        return LocateCandidate(
            candidate_id=hit.pattern.pattern_id,
            candidate_type="pattern",
            score=hit.score,
            summary=hit.pattern.content,
            payload={
                "pattern_id": hit.pattern.pattern_id,
                "signals": hit.pattern.signals,
                "root_cause": hit.pattern.root_cause,
                "content": hit.pattern.content,
            },
        )

    def _procedure_candidate(self, hit: ProcedurePropagation) -> LocateCandidate:
        return LocateCandidate(
            candidate_id=hit.procedure.procedure_id,
            candidate_type="procedure",
            score=hit.score,
            summary=hit.procedure.content,
            payload={
                "procedure_id": hit.procedure.procedure_id,
                "symptoms": hit.procedure.symptoms,
                "content": hit.procedure.content,
                "via_pattern_id": hit.via_pattern_id,
            },
        )

    def _case_candidate(self, hit: CasePropagation) -> LocateCandidate:
        return LocateCandidate(
            candidate_id=hit.case.case_id,
            candidate_type="case",
            score=hit.score,
            summary=hit.case.content,
            payload={
                "case_id": hit.case.case_id,
                "symptoms": hit.case.symptoms,
                "root_cause": hit.case.root_cause,
                "content": hit.case.content,
                "via_pattern_id": hit.via_pattern_id,
            },
        )

    def _propose_by_llm(self, agent_type: str, summary: dict, locate_result: LocateResult) -> list[Proposal]:
        self._require_llm_args("proposal generation")
        system_prompt, user_prompt = build_proposal_prompts(
            agent_type,
            summary,
            {
                "incident_id": locate_result.incident_id,
                "agent_type": locate_result.agent_type,
                "query": locate_result.query,
                "candidates": [candidate.__dict__ for candidate in locate_result.candidates],
                "notes": locate_result.notes,
            },
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
        raw_proposals = payload.get("proposals") or []
        return [self._normalize_proposal(agent_type, summary["incident_id"], item) for item in raw_proposals if item]

    def _recover_patterns_by_llm(
        self,
        summary: dict,
        target_proposal: Proposal,
        pattern_locate_result: LocateResult,
        recovery_comment: str,
        rejected_pattern_reviews: list[dict],
    ) -> list[Proposal]:
        self._require_llm_args("pattern recovery")
        system_prompt, user_prompt = build_pattern_recovery_prompts(
            summary=summary,
            target_proposal=self._proposal_to_prompt_dict(target_proposal),
            pattern_locate_result={
                "incident_id": pattern_locate_result.incident_id,
                "agent_type": pattern_locate_result.agent_type,
                "query": pattern_locate_result.query,
                "candidates": [candidate.__dict__ for candidate in pattern_locate_result.candidates],
                "notes": pattern_locate_result.notes,
            },
            recovery_comment=recovery_comment,
            rejected_pattern_reviews=rejected_pattern_reviews,
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
        raw_proposals = payload.get("proposals") or []
        proposals = [
            self._normalize_proposal("PatternAgent", summary["incident_id"], item, proposal_source="pattern_recovery")
            for item in raw_proposals
            if item
        ]
        return [proposal for proposal in proposals if proposal.action == "create" and proposal.knowledge_type == "pattern"]

    def _normalize_proposal(
        self,
        agent_type: str,
        incident_id: str,
        payload: dict,
        proposal_source: str = "standard",
    ) -> Proposal:
        object_payload = payload.get("object")
        knowledge_type = payload["knowledge_type"]
        if object_payload and knowledge_type != "relation":
            object_payload = dict(object_payload)
            id_field = get_object_id_field(knowledge_type)
            raw_object_id = str(object_payload.get(id_field) or "").strip()
            if not is_uuid(raw_object_id):
                object_payload[id_field] = new_uuid()
        elif object_payload:
            object_payload = dict(object_payload)
        return Proposal(
            proposal_id=payload.get("proposal_id") if is_uuid(payload.get("proposal_id")) else new_uuid(),
            incident_id=incident_id,
            agent_type=agent_type,
            knowledge_type=knowledge_type,
            action=payload["action"],
            target_id=payload.get("target_id"),
            object=object_payload,
            rationale=payload.get("rationale", ""),
            confidence=float(payload.get("confidence", 0.0)),
            proposal_source=payload.get("proposal_source", proposal_source),
        )

    def _require_llm_args(self, operation: str) -> None:
        if not self.llm_args:
            raise ValueError(f"MemoryConsolidationSubagents requires configured LLM args for {operation}.")

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
