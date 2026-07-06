from __future__ import annotations

import json
from typing import Any

from prompts.prompt_generation import construct_expert_analyze_prompt
from utils.llm import llm, parse_json_response
from utils.logging import log_to_file


class ExpertActions:
    """Prompt construction, LLM invocation, logging, and JSON parsing for ExpertAgent."""

    def __init__(self, expert_name: str, llm_args: dict[str, Any], log_path: str):
        self.expert_name = expert_name
        self.llm_args = llm_args
        self.log_path = log_path

    def decide(
        self,
        *,
        step_index: int,
        task: str,
        tool_prompt: str,
        belief: str,
        history: list,
        frontier: dict,
        memory_context: str,
    ) -> dict[str, Any] | None:
        log_to_file(f"\n[ExpertAction] Step {step_index} - Decision Making", log_path=self.log_path)

        response = self._call(
            task=task,
            tool_prompt=tool_prompt,
            belief=belief,
            history=history,
            frontier=frontier,
            task_stage="decision_making",
            memory_context=memory_context,
        )
        log_to_file(f"[ExpertAction] Step {step_index} decision output: {response}\n\n", log_path=self.log_path)

        try:
            decision_result = parse_json_response(response.strip())
            if not all(k in decision_result for k in ["type", "decision"]):
                raise ValueError("Missing mandatory fields (type/decision) in decision output")
            if decision_result["decision"] not in ["retrieve", "tool_call", "analyze"]:
                raise ValueError(f"Invalid decision value: {decision_result['decision']}")
            return decision_result
        except (json.JSONDecodeError, ValueError) as e:
            log_to_file(f"Decision-stage response parsing failed: {e}; skipping this retrieval iteration.\n\n", log_path=self.log_path)
            return None

    def generate_content(
        self,
        *,
        step_index: int,
        task: str,
        tool_prompt: str,
        belief: str,
        history: list,
        frontier: dict,
        decision_result: dict[str, Any],
        memory_context: str,
    ) -> dict[str, Any] | None:
        log_to_file(f"[ExpertAction] Step {step_index} - Content Generation\n\n", log_path=self.log_path)

        response = self._call(
            task=task,
            tool_prompt=tool_prompt,
            belief=belief,
            history=history,
            frontier=frontier,
            task_stage="content_generation",
            decision_result=decision_result,
            memory_context=memory_context,
        )
        log_to_file(f"[ExpertAction] Step {step_index} content output: {response}\n\n", log_path=self.log_path)

        try:
            final_response_dict = parse_json_response(response.strip())
            if not all(k in final_response_dict for k in ["type", "decision", "content"]):
                raise ValueError("Missing mandatory fields (type/decision/content) in final response")
            if (
                final_response_dict["type"] != decision_result["type"]
                or final_response_dict["decision"] != decision_result["decision"]
            ):
                raise ValueError(
                    "Content stage result inconsistent with decision stage: "
                    f"expected (type={decision_result['type']}, decision={decision_result['decision']}), "
                    f"got (type={final_response_dict['type']}, decision={final_response_dict['decision']})"
                )
            final_response_dict["content"] = final_response_dict["content"].strip()
            return final_response_dict
        except (json.JSONDecodeError, ValueError) as e:
            log_to_file(f"Content-stage response parsing failed: {e}; skipping this retrieval iteration.\n\n", log_path=self.log_path)
            return None

    def _call(
        self,
        *,
        task: str,
        tool_prompt: str,
        belief: str,
        history: list,
        frontier: dict,
        task_stage: str,
        memory_context: str,
        decision_result: dict[str, Any] | None = None,
    ) -> str:
        system_prompt, user_prompt = construct_expert_analyze_prompt(
            expert_name=self.expert_name,
            task=task,
            tool_prompt=tool_prompt,
            belief=belief,
            history=history,
            frontier=frontier,
            task_stage=task_stage,
            decision_result=decision_result,
            memory_context=memory_context,
        )
        response, _meta = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=self.llm_args["temperature"],
            max_tokens=self.llm_args["max_tokens"],
            return_meta=self.llm_args["return_meta"],
        )
        return response





