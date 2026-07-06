from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


PROMPT_ROOT = Path(__file__).resolve().parent


def _load_system_prompt(expert_name: str) -> str:
    prompt_path = PROMPT_ROOT / "system_prompts.py"
    spec = importlib.util.spec_from_file_location("opsmem_system_prompts", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load system prompts from {prompt_path}")

    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)

    attr_name = f"{expert_name}_system_prompt"
    if not hasattr(prompt_module, attr_name):
        raise KeyError(f"Missing system prompt: {attr_name}")
    return getattr(prompt_module, attr_name).strip()


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _memory_block(memory_context: str) -> str:
    if not memory_context:
        return "No long-term memory context is available for this step."
    return memory_context


def construct_ingest_prompt_basic_node(expert_name, symptom):
    system_prompt = f"""
{_load_system_prompt(expert_name)}

Task: extract the initial OpsMem graph nodes from one operational incident description.

Return only valid JSON with this exact schema:
{{
  "symptom_node": "one concise user-visible or externally observable failure signal",
  "isolated_evidence": ["objective evidence item 1", "objective evidence item 2"]
}}

Rules:
- Create exactly one symptom_node.
- Extract only facts stated in the input; do not infer a root cause here.
- Keep isolated_evidence objective and compact. Merge closely related facts to avoid over-fragmentation.
- Prefer 1-5 evidence items. Use an empty list only when the input contains no objective evidence.
- Output JSON only, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Incident description:
{symptom}
""".strip()
    return system_prompt, user_prompt


def construct_ingest_prompt_L1_Hypo(expert_name, graph, symptom_node_id, isolated_evidence_ids):
    evidence = []
    for evidence_id in isolated_evidence_ids:
        evidence.append({
            "id": evidence_id,
            "label": graph.nodes[evidence_id]["label"],
        })

    system_prompt = f"""
{_load_system_prompt(expert_name)}

Task: generate level-1 root-cause hypotheses from the Symptom and Evidence nodes.

Return only valid JSON with this exact schema:
{{
  "candidates": [
    {{"id": "h001", "label": "short hypothesis", "confidence": 0.0, "why": "brief evidence-grounded rationale"}}
  ],
  "edges": [
    {{"src": "evidence node id", "dst": "candidate id", "relation": "support"}}
  ]
}}

Rules:
- Generate 2-4 coarse level-1 hypotheses.
- confidence must be a float from 0.0 to 1.0.
- relation must be either "support" or "refute".
- Every edge src must be one of the provided evidence IDs, and every dst must be one of your candidate IDs.
- Use only operational evidence provided in the input. Do not invent telemetry.
- Output JSON only, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Symptom node:
{_json_block({"id": symptom_node_id, "label": graph.nodes[symptom_node_id]["label"]})}

Evidence nodes:
{_json_block(evidence)}
""".strip()
    return system_prompt, user_prompt


def construct_call_expert_prompt(expert_name, frontier, expert_descriptions, memory_context: str = ""):
    system_prompt = f"""
{_load_system_prompt(expert_name)}

Task: decide which expert agents should inspect telemetry for the current frontier hypothesis.

Return only valid JSON as a list with this exact item schema:
[
  {{"expert_name": "Agent_Name", "task": "specific telemetry question for that agent"}}
]

Rules:
- Return [] if no expert call is needed.
- Select only agents listed in expert_descriptions.
- Prefer 1-3 calls that directly test, support, or refute the frontier.
- The task must be concrete and telemetry-oriented, not a generic request.
- Use memory context as supporting context only; the current frontier remains primary.
- Output JSON only, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Current frontier hypothesis:
{_json_block(frontier)}

Available expert agents:
{_json_block(expert_descriptions)}

Long-term memory context:
{_memory_block(memory_context)}
""".strip()
    return system_prompt, user_prompt


def construct_expert_analyze_prompt(
    expert_name,
    task,
    tool_prompt,
    belief,
    history,
    frontier,
    task_stage,
    decision_result=None,
    memory_context: str = "",
):
    if task_stage == "decision_making":
        task_prompt = """
Task: decide whether more telemetry is required before producing your expert analysis.

Return only valid JSON with this exact schema:
{"type": 2, "decision": "tool_call"}

Rules:
- type is 2 because this open-source example exposes telemetry through the provided tool interface.
- decision must be one of: "tool_call", "retrieve", "analyze".
- Use "tool_call" when a listed telemetry query can test the assigned task/current frontier and has not already been used.
- Use "retrieve" only if the tool interface explicitly provides a retrieval-style command.
- Use "analyze" when existing belief, memory context, frontier, task, and history are sufficient for a concise expert conclusion.
- Do not repeat equivalent historical tool calls.
- Output JSON only, with no markdown or explanation.
""".strip()
    elif task_stage == "content_generation":
        if not decision_result:
            raise ValueError("decision_result is required for content_generation")
        task_prompt = f"""
Task: generate content consistent with the previous decision.

Previous decision:
{_json_block(decision_result)}

Return only valid JSON with this exact schema:
{{"type": {decision_result["type"]}, "decision": "{decision_result["decision"]}", "content": "..."}}

Rules:
- Keep type and decision exactly the same as the previous decision.
- If decision is "tool_call" or "retrieve", content must be one executable/request string accepted by the provided tool interface.
- If decision is "analyze", content must be a concise expert analysis grounded in the assigned task, belief, memory context, frontier, and telemetry history.
- Do not fabricate telemetry or claim checks that are not present in history.
- Output JSON only, with no markdown or explanation.
""".strip()
    else:
        raise ValueError(f"Unknown task_stage: {task_stage}")

    system_prompt = f"""
{_load_system_prompt(expert_name)}

{task_prompt}
""".strip()

    user_prompt = f"""
Tool interface:
{tool_prompt}

Current belief:
{belief}

Current frontier hypothesis:
{_json_block(frontier)}

Telemetry history:
{_json_block(history) if history else "No telemetry has been queried yet."}

Long-term memory context:
{_memory_block(memory_context)}
""".strip()
    return system_prompt, user_prompt


def construct_generate_proposal_prompt(expert_name, belief, graph_description, analyses=None, memory_context: str = ""):
    analyses = analyses or []
    system_prompt = f"""
{_load_system_prompt(expert_name)}

Task: update the short-term memory graph after expert analysis.

Return only valid JSON with this exact schema:
{{
  "edit": [
    {{"node_id": "existing hypothesis node id", "confidence": 0.0, "why": "updated rationale"}}
  ],
  "nodes": [
    {{"id": "n001", "node_type": "Evidence", "label": "new evidence"}},
    {{"id": "h101", "node_type": "Hypothesis", "label": "new hypothesis", "confidence": 0.0, "why": "brief rationale"}}
  ],
  "edges": [
    {{"src": "source id", "dst": "target id", "relation": "support"}}
  ]
}}

Rules:
- edit may update confidence/why for existing hypothesis nodes only.
- nodes may contain only Evidence or Hypothesis.
- relation must be one of: "support", "refute", "refines".
- Use temporary ids for new nodes, and use existing ids from graph_description for existing nodes.
- Add new Evidence only when it comes from expert analysis or telemetry history.
- Add new Hypothesis only when it is a meaningful refinement or alternative supported by the evidence.
- Output empty lists when no graph update is justified.
- Output JSON only, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Current belief:
{belief}

Current STM graph:
{_json_block(graph_description)}

Expert analyses:
{_json_block(analyses)}

Long-term memory context:
{_memory_block(memory_context)}
""".strip()
    return system_prompt, user_prompt


def construct_report_or_refine_prompt(expert_name, belief, graph_description, Report_Flag, frontier, memory_context: str = ""):
    force_report = str(Report_Flag).lower() == "true"
    system_prompt = f"""
{_load_system_prompt(expert_name)}

Task: decide whether OpsMem should produce the final report or refine the frontier into deeper hypotheses.

If reporting, return only valid JSON with this exact schema:
{{
  "type": 1,
  "answer": "short root-cause answer",
  "report": "concise evidence-grounded incident report"
}}

If refining, return only valid JSON with this exact schema:
{{
  "type": 2,
  "candidates": [
    {{"label": "deeper hypothesis", "confidence": 0.0, "why": "brief rationale"}}
  ]
}}

Rules:
- type 1 means final report; type 2 means refine.
- If force_report is true, return type 1.
- Return type 1 when the current frontier is sufficiently supported and no important competing explanation remains.
- Return type 2 when the frontier is still broad and should be split into 2-4 more specific hypotheses.
- confidence must be a float from 0.0 to 1.0.
- Do not fabricate evidence. Reports must cite only provided graph/belief/memory context.
- Output JSON only, with no markdown or explanation.
""".strip()

    user_prompt = f"""
Force final report:
{force_report}

Current belief:
{belief}

Current frontier hypothesis:
{_json_block(frontier)}

Current STM graph:
{_json_block(graph_description)}

Long-term memory context:
{_memory_block(memory_context)}
""".strip()
    return system_prompt, user_prompt
