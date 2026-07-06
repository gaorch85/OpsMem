from __future__ import annotations

from typing import Any

from cmr import CrossMemoryResonance
from consolidation.agents.approval import get_commit_approval
from consolidation.agents.meta_agent import MetaAgent
from consolidation.commitment.commit import MemoryCommitter
from consolidation.workflow.memory_package import build_resolved_memory_package
from consolidation.schema import (
    ApprovalResult,
    FeedbackDecision,
    IncidentSummary,
    LocateResult,
    PatternRecoveryResult,
    PendingApprovalBundle,
    Proposal,
    ReflectionDecision,
    RelationSynthesisResult,
    ReviewResult,
    to_jsonable,
)
from consolidation.commitment.store import MemoryConsolidationLogStore
from consolidation.agents.subagents import MemoryConsolidationSubagents


class MemoryConsolidationPipeline:
    def __init__(self, config: dict[str, Any], llm_args: dict[str, Any] | None = None, long_term_memory=None):
        self.config = config or {}
        self.llm_args = llm_args or {}
        self.long_term_memory = long_term_memory
        self.cmr_config = self.config.get("cmr") or self.config
        self.embedding_provider = None
        self.cmr_engine = self._build_cmr_engine(long_term_memory)
        self.store = MemoryConsolidationLogStore(
            self.config.get("log_dir", "logs/OpsMem/consolidation"),
            detailed_logging=bool(self.config.get("detailed_logging", False)),
        )
        target_store_dir = self.config.get("resolved_active_store_dir") or self.config.get("active_store_dir")
        if not target_store_dir:
            raise ValueError("MemoryConsolidationPipeline requires resolved_active_store_dir or active_store_dir in config.")
        self.committer = MemoryCommitter(store_dir=target_store_dir, log_store=self.store)
        self.subagents = MemoryConsolidationSubagents(
            long_term_memory=long_term_memory,
            cmr_engine=self.cmr_engine,
            llm_args=self.llm_args,
        )
        self.meta_agent = MetaAgent(llm_args=self.llm_args, config=self.config)

    def _build_cmr_engine(self, long_term_memory):
        if long_term_memory is None:
            return None
        cross_memory_resonance = CrossMemoryResonance(
            long_term_memory=long_term_memory,
            llm_args=self.llm_args,
            cmr_config=self.cmr_config,
            embedding_provider=self.embedding_provider,
        )
        self.embedding_provider = cross_memory_resonance.embedding_provider
        return cross_memory_resonance

    def set_long_term_memory(self, long_term_memory) -> None:
        self.long_term_memory = long_term_memory
        self.cmr_engine = self._build_cmr_engine(long_term_memory)
        self.subagents.long_term_memory = long_term_memory
        self.subagents.cmr_engine = self.cmr_engine

    def set_target_store_dir(self, store_dir: str) -> None:
        self.committer = MemoryCommitter(store_dir=store_dir, log_store=self.store)

    def run(self, diagnosis_artifact: dict, feedback: FeedbackDecision) -> PendingApprovalBundle | None:
        incident_id = diagnosis_artifact["incident_id"]
        self.store.save_json("diagnosis_artifacts", f"{incident_id}.json", diagnosis_artifact)
        self.store.save_json("feedback", f"{incident_id}.json", feedback)

        memory_package = build_resolved_memory_package(diagnosis_artifact, feedback)
        self.store.save_json("memory_packages", f"{incident_id}.json", memory_package)

        summary = self.meta_agent.summarize(memory_package)
        self.store.save_json("summaries", f"{incident_id}.json", summary)

        related_memory = self._build_related_memory(summary)
        reflection = self.meta_agent.reflect(summary, related_memory)
        self.store.save_json("reflections", f"{incident_id}.json", reflection)
        if not reflection.update_required:
            return None

        locate_results = self._collect_locate_results(summary, reflection)
        for agent_type, locate_result in locate_results.items():
            self.store.save_json("locate_results", f"{incident_id}_{agent_type}.json", locate_result)

        node_proposals = self._collect_node_proposals(summary, reflection, locate_results)
        if node_proposals:
            self.store.save_json("proposals", f"{incident_id}_nodes.json", node_proposals)

        review_result = self.meta_agent.review(to_jsonable(summary), to_jsonable(reflection), node_proposals, related_memory)
        self.store.save_json("reviews", f"{incident_id}.json", review_result)

        approved_node_proposals = list(review_result.approved_proposals)
        pattern_locate_result = locate_results.get("PatternAgent") or LocateResult(
            incident_id=incident_id,
            agent_type="PatternAgent",
            query="",
            candidates=[],
            notes="PatternAgent locate was not executed.",
        )

        relation_result = self.meta_agent.synthesize_relations(summary, pattern_locate_result, approved_node_proposals)
        self.store.save_json("relation_synthesis", f"{incident_id}_round1.json", relation_result)

        recovery_result = PatternRecoveryResult(
            incident_id=incident_id,
            recovery_proposals=[],
            triggered_from_proposal_ids=[],
            notes="Pattern recovery was not triggered.",
        )
        final_relation_result = relation_result
        if relation_result.unlinked_targets:
            recovery_result = self._recover_patterns(summary, relation_result, pattern_locate_result, review_result)
            self.store.save_json("pattern_recovery", f"{incident_id}.json", recovery_result)
            approved_node_proposals.extend(recovery_result.recovery_proposals)
            final_relation_result = self.meta_agent.synthesize_relations(summary, pattern_locate_result, approved_node_proposals)
            self.store.save_json("relation_synthesis", f"{incident_id}_round2.json", final_relation_result)

        final_proposals = self._merge_proposals(
            review_result.approved_proposals
            + recovery_result.recovery_proposals
            + final_relation_result.relation_proposals
        )
        if not final_proposals:
            return None

        notes = " | ".join(
            part
            for part in [
                review_result.review_notes,
                final_relation_result.notes,
                recovery_result.notes,
            ]
            if part
        )
        bundle = PendingApprovalBundle(
            incident_id=incident_id,
            status="pending_commit",
            approved_proposals=final_proposals,
            review_notes=notes,
        )
        self.store.save_json("pending_commit", f"{incident_id}.json", bundle)
        return bundle

    def finalize_pending_bundle(self, bundle: PendingApprovalBundle, provider: str | None = None) -> tuple[ApprovalResult, object | None]:
        approval_result = get_commit_approval(
            bundle=bundle,
            provider=provider or self.config.get("approval_provider", "terminal"),
        )
        commit_log = self.committer.commit(bundle=bundle, approval_result=approval_result)
        return approval_result, commit_log

    def _build_related_memory(self, summary: IncidentSummary) -> dict:
        related = {
            "memory_context": "",
            "patterns": [],
            "procedures": [],
            "cases": [],
        }
        if self.cmr_engine is None:
            return related

        cmr_output = self.cmr_engine.run(summary.signals)
        pattern_hits = cmr_output.pattern_hits
        procedure_hits = cmr_output.procedure_hits
        case_hits = cmr_output.case_hits
        related["memory_context"] = cmr_output.context
        related["patterns"] = [
            {
                "pattern_id": hit.pattern.pattern_id,
                "score": hit.score,
                "signals": hit.pattern.signals,
                "root_cause": hit.pattern.root_cause,
                "content": hit.pattern.content,
            }
            for hit in pattern_hits
        ]
        related["procedures"] = [
            {
                "procedure_id": hit.procedure.procedure_id,
                "score": hit.score,
                "content": hit.procedure.content,
                "symptoms": hit.procedure.symptoms,
                "via_pattern_id": hit.via_pattern_id,
            }
            for hit in procedure_hits
        ]
        related["cases"] = [
            {
                "case_id": hit.case.case_id,
                "score": hit.score,
                "symptoms": hit.case.symptoms,
                "root_cause": hit.case.root_cause,
                "content": hit.case.content,
                "via_pattern_id": hit.via_pattern_id,
            }
            for hit in case_hits
        ]
        return related

    def _collect_locate_results(self, summary: IncidentSummary, reflection: ReflectionDecision) -> dict[str, LocateResult]:
        locate_results: dict[str, LocateResult] = {}
        called_agents = set(reflection.subagent_calls)
        if "CaseAgent" in called_agents or "ProcedureAgent" in called_agents:
            locate_results["PatternAgent"] = self.subagents.locate("PatternAgent", to_jsonable(summary))

        for agent_type in reflection.subagent_calls:
            if agent_type in locate_results:
                continue
            locate_results[agent_type] = self.subagents.locate(agent_type, to_jsonable(summary))
        return locate_results

    def _collect_node_proposals(
        self,
        summary: IncidentSummary,
        reflection: ReflectionDecision,
        locate_results: dict[str, LocateResult],
    ) -> list[Proposal]:
        node_proposals: list[Proposal] = []
        for agent_type in reflection.subagent_calls:
            locate_result = locate_results[agent_type]
            proposals = self.subagents.propose(agent_type, to_jsonable(summary), locate_result)
            if proposals:
                self.store.save_json("proposals", f"{summary.incident_id}_{agent_type}.json", proposals)
                node_proposals.extend(proposals)
        return node_proposals

    def _recover_patterns(
        self,
        summary: IncidentSummary,
        relation_result: RelationSynthesisResult,
        pattern_locate_result: LocateResult,
        review_result: ReviewResult,
    ) -> PatternRecoveryResult:
        proposal_map = {proposal.proposal_id: proposal for proposal in review_result.approved_proposals}
        rejected_pattern_reviews = [
            {"proposal_id": decision.proposal_id, "reason": decision.reason}
            for decision in review_result.decisions
            for proposal in review_result.rejected_proposals
            if proposal.proposal_id == decision.proposal_id and proposal.knowledge_type == "pattern"
        ]
        recovery_proposals: list[Proposal] = []
        triggered_ids: list[str] = []
        for target in relation_result.unlinked_targets:
            proposal = proposal_map.get(target.proposal_id)
            if proposal is None:
                continue
            triggered_ids.append(target.proposal_id)
            recovery_proposals.extend(
                self.subagents.recover_patterns(
                    summary=to_jsonable(summary),
                    target_proposal=proposal,
                    pattern_locate_result=pattern_locate_result,
                    recovery_comment=target.recovery_comment,
                    rejected_pattern_reviews=rejected_pattern_reviews,
                )
            )
        return PatternRecoveryResult(
            incident_id=summary.incident_id,
            recovery_proposals=self._merge_proposals(recovery_proposals),
            triggered_from_proposal_ids=triggered_ids,
            notes="Pattern recovery executed for unlinked approved case/procedure proposals.",
        )

    @staticmethod
    def _merge_proposals(proposals: list[Proposal]) -> list[Proposal]:
        merged: list[Proposal] = []
        seen: set[str] = set()
        for proposal in proposals:
            if proposal.proposal_id in seen:
                continue
            seen.add(proposal.proposal_id)
            merged.append(proposal)
        return merged















