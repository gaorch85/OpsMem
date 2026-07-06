from __future__ import annotations

from typing import Any

from eval import _build_user_prompt, _eval_once
from consolidation.schema import FeedbackDecision
from utils.llm import get_current_model_name


def get_consolidation_feedback(
    diagnosis_artifact: dict[str, Any],
    provider: str = "terminal",
    llm_args: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> FeedbackDecision | None:
    if provider == "terminal":
        return _get_terminal_feedback(diagnosis_artifact["incident_id"])
    if provider == "eval":
        return _get_eval_feedback(diagnosis_artifact, llm_args=llm_args, config=config)
    raise ValueError(f"Unsupported feedback provider: {provider}")


def _get_terminal_feedback(incident_id: str) -> FeedbackDecision:
    while True:
        decision = input(f"[Memory Consolidation] {incident_id} decision (accept/reject): ").strip().lower()
        if decision in {"accept", "reject"}:
            break
        print("Please input 'accept' or 'reject'.")

    comment = input(f"[Memory Consolidation] {incident_id} comment (optional): ").strip()
    return FeedbackDecision(
        incident_id=incident_id,
        decision=decision,
        comment=comment,
        reviewer="terminal_human",
    )


def _get_eval_feedback(
    diagnosis_artifact: dict[str, Any],
    llm_args: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> FeedbackDecision:
    llm_args = llm_args or {}
    config = config or {}
    incident_id = diagnosis_artifact["incident_id"]
    groundtruth = str(diagnosis_artifact.get("groundtruth") or "").strip()
    if not groundtruth:
        return FeedbackDecision(
            incident_id=incident_id,
            decision="reject",
            comment="Eval feedback skipped because groundtruth is empty.",
            reviewer="eval_judge",
        )

    result = _eval_once(
        user_prompt=_build_user_prompt(
            prediction=str(diagnosis_artifact.get("prediction") or ""),
            report=str(diagnosis_artifact.get("report") or ""),
            answer=groundtruth,
            case_id=diagnosis_artifact.get("case_index", incident_id),
        ),
        model_name=config.get("feedback_eval_model") or llm_args.get("model_path") or get_current_model_name(),
        temperature=float(config.get("feedback_eval_temperature", 1.0)),
        max_tokens=int(config.get("feedback_eval_max_tokens", 4096)),
    )
    score = int(result["score"])
    decision = "accept" if score == 2 else "reject"
    return FeedbackDecision(
        incident_id=incident_id,
        decision=decision,
        comment=f"Eval score={score}. Reasoning: {result.get('reasoning', '')}",
        reviewer="eval_judge",
    )





