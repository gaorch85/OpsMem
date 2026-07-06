from __future__ import annotations

from typing import Any, Callable

from prompts.prompt_generation import (
    construct_call_expert_prompt,
    construct_generate_proposal_prompt,
    construct_ingest_prompt_L1_Hypo,
    construct_ingest_prompt_basic_node,
    construct_report_or_refine_prompt,
)
from utils.llm import llm, parse_json_response
from utils.logging import log_to_file


class CentralActions:
    """Prompt construction, LLM invocation, logging, and JSON parsing for CentralAgent."""

    def __init__(self, agent_name: str, llm_args: dict[str, Any], log_path: str):
        self.agent_name = agent_name
        self.llm_args = llm_args
        self.log_path = log_path

    def ingest_basic_nodes(self, symptom: str) -> dict[str, Any]:
        return self._call_json(
            lambda: construct_ingest_prompt_basic_node(
                expert_name=self.agent_name,
                symptom=symptom,
            ),
            "#### Ingest: generating basic symptom and evidence nodes",
        )

    def ingest_l1_hypotheses(self, graph, symptom_node_id: str, isolated_evidence_ids: list[str]) -> dict[str, Any]:
        return self._call_json(
            lambda: construct_ingest_prompt_L1_Hypo(
                expert_name=self.agent_name,
                graph=graph,
                symptom_node_id=symptom_node_id,
                isolated_evidence_ids=isolated_evidence_ids,
            ),
            "#### Ingest: generating L1 hypotheses and evidence links",
        )

    def plan_expert_calls(self, frontier: dict[str, Any], expert_descriptions: dict[str, str], memory_context: str) -> list[dict[str, Any]]:
        return self._call_json(
            lambda: construct_call_expert_prompt(
                expert_name=self.agent_name,
                frontier=frontier,
                expert_descriptions=expert_descriptions,
                memory_context=memory_context,
            ),
            "#### Central plan: expert calls",
        )

    def generate_proposal(self, analyses: list[dict[str, str]], graph_description: dict, belief: str, memory_context: str) -> dict[str, Any]:
        return self._call_json(
            lambda: construct_generate_proposal_prompt(
                expert_name=self.agent_name,
                analyses=analyses,
                graph_description=graph_description,
                belief=belief,
                memory_context=memory_context,
            ),
            "#### Central proposal: STM update",
        )

    def report_or_refine(
        self,
        report_flag: str,
        graph_description: dict,
        belief: str,
        frontier: dict[str, Any],
        memory_context: str,
    ) -> dict[str, Any]:
        return self._call_json(
            lambda: construct_report_or_refine_prompt(
                expert_name=self.agent_name,
                Report_Flag=report_flag,
                graph_description=graph_description,
                belief=belief,
                frontier=frontier,
                memory_context=memory_context,
            ),
            "#### Central decision: report or refine",
        )

    def _call_json(self, prompt_builder: Callable[[], tuple[str, str]], log_title: str):
        system_prompt, user_prompt = prompt_builder()
        response, _meta = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args["temperature"],
            max_tokens=self.llm_args["max_tokens"],
            return_meta=self.llm_args["return_meta"],
        )
        log_to_file(f"{log_title}\n\n{response}\n\n", log_path=self.log_path)
        return parse_json_response(response)





