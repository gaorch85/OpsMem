from __future__ import annotations

from agents.expert_actions import ExpertActions
from utils.logging import log_to_file
from tools.diagnostic_tools import DiagnosticTools


class ExpertAgent:
    """Expert agent that inspects telemetry for a planned diagnosis task."""

    def __init__(
        self,
        name,
        short_term_memory,
        case_id,
        log_path,
        llm_args,
        evidence_text,
        max_retrieval_steps,
        memory_resonator,
    ):
        self.name = name
        self.short_term_memory = short_term_memory
        self.graph = short_term_memory.graph
        self.fsm = short_term_memory.fsm
        self.case_id = case_id
        self.log_path = log_path
        self.llm_args = llm_args
        self.evidence_text = evidence_text
        self.max_retrieval_steps = max_retrieval_steps
        self.retrieval_history = []
        self.last_analysis = ""
        self.last_task = ""
        self.actions = ExpertActions(expert_name=name, llm_args=llm_args, log_path=log_path)
        self.diagnostic_tools = None
        self.memory_resonator = memory_resonator
        self.memory_context = ""
        self.memory_resonance_traces = []

        if isinstance(evidence_text, dict):
            try:
                self.diagnostic_tools = DiagnosticTools.from_case_resources(
                    case_id=case_id,
                    resources=evidence_text,
                    model_path=llm_args.get("model_path"),
                )
            except Exception as exc:
                log_to_file(
                    f"[DiagnosticTools] Failed to initialize tools for case {case_id}: {exc}\n",
                    log_path=self.log_path,
                )
                self.diagnostic_tools = None

    def run(self, frontier, task: str = ""):
        self.last_task = task
        return self._run_with_tools(frontier, task=task)

    def resonate_memory(self) -> str:
        resonance = self.memory_resonator.resonate(short_term_memory=self.short_term_memory)
        self.memory_context = resonance.context
        self.memory_resonance_traces.append(resonance.to_dict())
        return self.memory_context

    def _run_with_tools(self, frontier, task: str):
        diagnostic_tools = self.diagnostic_tools
        if diagnostic_tools is None:
            return "Diagnostic tools are missing for this case."

        self.resonate_memory()

        retrieve_step = 0
        final_analysis = None
        retrieval_history = self.retrieval_history

        while retrieve_step < self.max_retrieval_steps and final_analysis is None:
            tool_prompt = diagnostic_tools.build_tool_prompt()
            if retrieve_step >= self.max_retrieval_steps - 1:
                tool_prompt += "\nYou are at the final retrieval step; conclude if the available information is sufficient."

            decision_result = self.actions.decide(
                step_index=retrieve_step + 1,
                task=task,
                tool_prompt=tool_prompt,
                belief=self.graph.belief,
                history=retrieval_history,
                frontier=frontier,
                memory_context=self.memory_context,
            )
            if decision_result is None:
                retrieve_step += 1
                continue

            current_decision = decision_result["decision"]
            final_response_dict = self.actions.generate_content(
                step_index=retrieve_step + 1,
                task=task,
                tool_prompt=tool_prompt,
                belief=self.graph.belief,
                history=retrieval_history,
                frontier=frontier,
                decision_result=decision_result,
                memory_context=self.memory_context,
            )
            if final_response_dict is None:
                retrieve_step += 1
                continue
            content = final_response_dict["content"]

            if current_decision in ["tool_call", "retrieve"]:
                retrieve_step += 1
                if not content:
                    log_to_file("Generated retrieval/tool request is empty; stopping retrieval loop.\n", log_path=self.log_path)
                    break
                tool_output = diagnostic_tools.dispatch_tool(content)
                retrieval_history.append((content, tool_output))
                log_to_file(f"[DiagnosticTools] Step {retrieve_step} tool output: {tool_output}\n\n", log_path=self.log_path)
                continue

            if current_decision == "analyze":
                final_analysis = content
                break

            log_to_file(f"Invalid decision value: {current_decision}; stopping retrieval loop.\n\n", log_path=self.log_path)
            break

        if final_analysis is None:
            final_analysis = "Insufficient information to complete the analysis (max retrieval steps reached or invalid decision/response)."

        self.last_analysis = final_analysis
        return final_analysis

    def export_trace(self) -> dict:
        return {
            "expert_name": self.name,
            "task": self.last_task,
            "retrieval_history": list(self.retrieval_history),
            "last_analysis": self.last_analysis,
            "memory_context": self.memory_context,
            "memory_resonance_traces": self.memory_resonance_traces,
        }





