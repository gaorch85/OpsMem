from __future__ import annotations

from copy import deepcopy

from consolidation.schema import FeedbackDecision


def build_resolved_memory_package(diagnosis_artifact: dict, feedback: FeedbackDecision) -> dict:
    package = deepcopy(diagnosis_artifact)
    package["feedback"] = {
        "decision": feedback.decision,
        "comment": feedback.comment,
        "reviewer": feedback.reviewer,
    }
    return package




