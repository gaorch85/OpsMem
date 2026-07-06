from __future__ import annotations

import json


def build_summary_prompts(package: dict) -> tuple[str, str]:
    system_prompt = (
        "You are MetaAgent. Summarize a resolved incident into compact memory-consolidation-ready JSON. "
        "Output JSON only with keys: incident_id, signals, decisive_evidence, root_cause, case_summary."
    )
    user_prompt = f"Resolved memory package:\n{json.dumps(package, ensure_ascii=False, indent=2)}"
    return system_prompt, user_prompt


def build_reflection_prompts(summary: dict, related_memory: dict) -> tuple[str, str]:
    system_prompt = (
        "You are MetaAgent. Decide whether the resolved incident should be consolidated into long-term memory. "
        "Output JSON only with keys: incident_id, update_required, reflection, subagent_calls. "
        "Allowed subagent_calls: PatternAgent, CaseAgent, ProcedureAgent."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Related long-term memory context:\n{json.dumps(related_memory, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def build_locate_filter_prompts(agent_type: str, summary: dict, candidates: list[dict]) -> tuple[str, str]:
    system_prompt = (
        f"You are {agent_type}. Filter retrieved memory objects and keep only the related ones. "
        "Output JSON only with key 'related_candidate_ids', which must be a list of candidate_id strings. "
        "Do not create new objects. Do not explain unselected candidates."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Retrieved candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def build_proposal_prompts(agent_type: str, summary: dict, locate_result: dict) -> tuple[str, str]:
    system_prompt = (
        f"You are {agent_type}. Based on the incident summary and located memory objects, propose long-term memory changes. "
        "Output JSON only with key 'proposals'. Each proposal must contain: proposal_id, knowledge_type, "
        "action(create/delete), target_id, object, rationale, confidence. "
        "Do not output edges. Proposals in this step are node proposals only."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Locate result:\n{json.dumps(locate_result, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def build_review_prompts(summary: dict, reflection: dict, proposals: list[dict], related_memory: dict) -> tuple[str, str]:
    system_prompt = (
        "You are MetaAgent. Review long-term memory consolidation proposals. Output JSON only with keys: decisions and review_notes. "
        "Each item in decisions must contain: proposal_id, verdict(approve/reject), reason."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Reflection decision:\n{json.dumps(reflection, ensure_ascii=False, indent=2)}\n\n"
        f"Proposals:\n{json.dumps(proposals, ensure_ascii=False, indent=2)}\n\n"
        f"Related long-term memory:\n{json.dumps(related_memory, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def build_relation_synthesis_prompts(
    summary: dict,
    pattern_locate_result: dict,
    approved_node_proposals: list[dict],
) -> tuple[str, str]:
    system_prompt = (
        "You are MetaAgent. Synthesize pattern relations for approved case and procedure proposals. "
        "Output JSON only with keys: relation_proposals, unlinked_targets, notes. "
        "Each relation_proposal must contain: proposal_id, knowledge_type=relation, action=create, object, rationale, confidence. "
        "Each relation object must contain: edge_type(pattern_case or pattern_procedure), source_id, target_id, weight. "
        "Each unlinked_targets item must contain: proposal_id, knowledge_type, recovery_comment."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Filtered pattern locate result:\n{json.dumps(pattern_locate_result, ensure_ascii=False, indent=2)}\n\n"
        f"Approved node proposals:\n{json.dumps(approved_node_proposals, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def build_pattern_recovery_prompts(
    summary: dict,
    target_proposal: dict,
    pattern_locate_result: dict,
    recovery_comment: str,
    rejected_pattern_reviews: list[dict],
) -> tuple[str, str]:
    system_prompt = (
        "You are PatternAgent. A case or procedure proposal cannot be linked to any existing pattern. "
        "Propose recovery patterns. Output JSON only with key 'proposals'. "
        "Each proposal must contain: proposal_id, knowledge_type=pattern, action=create, target_id, object, rationale, confidence. "
        "Do not output edges. At least one recovery pattern proposal should be produced."
    )
    user_prompt = (
        f"Incident summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Unlinked approved proposal:\n{json.dumps(target_proposal, ensure_ascii=False, indent=2)}\n\n"
        f"Filtered pattern locate result:\n{json.dumps(pattern_locate_result, ensure_ascii=False, indent=2)}\n\n"
        f"Recovery comment:\n{recovery_comment}\n\n"
        f"Rejected pattern review reasons:\n{json.dumps(rejected_pattern_reviews, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt





