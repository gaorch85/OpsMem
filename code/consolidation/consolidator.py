from __future__ import annotations

from typing import Any

from consolidation.workflow import MemoryConsolidationPipeline, get_consolidation_feedback


class MemoryConsolidator:
    """OpsMem long-term memory consolidation stage."""

    def __init__(self, config: dict[str, Any], llm_args: dict[str, Any] | None = None, long_term_memory=None):
        self.config = config or {}
        self.pipeline = MemoryConsolidationPipeline(
            config=self.config,
            llm_args=llm_args,
            long_term_memory=long_term_memory,
        )

    @property
    def store(self):
        return self.pipeline.store

    def set_long_term_memory(self, long_term_memory) -> None:
        self.pipeline.set_long_term_memory(long_term_memory)

    def consolidate(self, diagnosis_artifact: dict) -> None:
        incident_id = diagnosis_artifact["incident_id"]
        self.store.save_json("diagnosis_artifacts", f"{incident_id}.json", diagnosis_artifact)

        feedback = self._collect_feedback(diagnosis_artifact)
        if feedback is None:
            return

        self.store.save_json("feedback", f"{incident_id}.json", feedback)
        if feedback.decision != "accept":
            print(f"[Memory Consolidation] {incident_id}: feedback rejected, consolidation skipped.")
            return

        if not self.config.get("auto_trigger_on_accept_feedback", True):
            print(f"[Memory Consolidation] {incident_id}: accepted feedback recorded; auto trigger disabled.")
            return

        pending_bundle = self.pipeline.run(diagnosis_artifact, feedback)
        if pending_bundle is None:
            print(f"[Memory Consolidation] {incident_id}: no approved proposals were generated.")
            return

        print(f"[Memory Consolidation] {incident_id}: pending approval bundle generated.")
        approval_result, commit_log = self.pipeline.finalize_pending_bundle(
            pending_bundle,
            provider=self.config.get("approval_provider", "terminal"),
        )
        if commit_log is None:
            print(f"[Memory Consolidation] {incident_id}: all pending proposals were rejected by human approval.")
        else:
            print(f"[Memory Consolidation] {incident_id}: committed to {commit_log.target_store_dir}.")

    def _collect_feedback(self, diagnosis_artifact: dict):
        if not self.config.get("prompt_feedback_at_case_end", True):
            return None
        return get_consolidation_feedback(
            diagnosis_artifact=diagnosis_artifact,
            provider=self.config.get("feedback_provider", "terminal"),
            llm_args=self.pipeline.llm_args,
            config=self.config,
        )




