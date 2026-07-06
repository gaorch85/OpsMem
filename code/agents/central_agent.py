from __future__ import annotations

from agents.central_actions import CentralActions
from utils.logging import log_to_file


class CentralAgent:
    """Central coordinator for one OpsMem diagnosis session."""

    def __init__(
        self,
        name,
        short_term_memory,
        log_path,
        case_id,
        case_symptom,
        llm_args,
        agent_team,
        experts,
        evidence_text,
        memory_resonator,
    ):
        self.name = name
        self.short_term_memory = short_term_memory
        self.graph = self.short_term_memory.graph
        self.graph.model_path = llm_args.get("model_path")
        self.fsm = self.short_term_memory.fsm
        self.log_path = log_path
        self.case_id = case_id
        self.case_symptom = case_symptom
        self.llm_args = llm_args
        self.session_max_steps = llm_args["session_max_steps"]
        self.agent_team = agent_team
        self.experts = experts
        self.evidence_text = evidence_text
        self.actions = CentralActions(agent_name=name, llm_args=llm_args, log_path=log_path)
        self.memory_resonator = memory_resonator
        self.memory_context = ""
        self.memory_resonance_traces = []
        self.frontier = {}
        self.last_answer = ""
        self.last_report = ""

    def resonate_memory(self) -> str:
        resonance = self.memory_resonator.resonate(short_term_memory=self.short_term_memory)
        self.memory_context = resonance.context
        self.memory_resonance_traces.append(resonance.to_dict())
        return self.memory_context

    def ingest(self) -> None:
        response_dict = self.actions.ingest_basic_nodes(self.case_symptom)
        symptom_node = response_dict["symptom_node"]
        isolated_evidence = response_dict["isolated_evidence"]
        self.symptom_node_id, self.isolated_evidence_ids, evidence_id_map = (
            self.short_term_memory.initialize_from_ingest_response(symptom_node, isolated_evidence)
        )

        response_dict = self.actions.ingest_l1_hypotheses(
            graph=self.graph,
            symptom_node_id=self.symptom_node_id,
            isolated_evidence_ids=self.isolated_evidence_ids,
        )
        candidates = response_dict["candidates"]
        edges = response_dict["edges"]

        self.short_term_memory.add_initial_hypotheses(
            symptom_node_id=self.symptom_node_id,
            evidence_id_map=evidence_id_map,
            candidates=candidates,
            edges=edges,
        )
        self.resonate_memory()

    def run(self):
        step = 0
        self.graph.generate_belief_text()
        self.resonate_memory()

        while True:
            step += 1
            self.short_term_memory.tick_step(1)

            self.frontier = self.extract_frontier()
            log_to_file(
                f"#### Step {step}: current frontier\n\n{self.frontier}\n",
                log_path=self.log_path,
            )

            plan = self.plan()
            analyses = self.act(plan) if plan else []

            response_dict = self.actions.generate_proposal(
                analyses=analyses,
                graph_description=self.graph.to_dict(),
                belief=self.graph.belief,
                memory_context=self.memory_context,
            )
            warnings = self.short_term_memory.apply_proposal(
                symptom_node_id=self.symptom_node_id,
                frontier=self.frontier,
                edits=response_dict.get("edit", []),
                nodes=response_dict["nodes"],
                edges=response_dict["edges"],
            )
            for warning in warnings:
                log_to_file(f"{warning}\n", log_path=self.log_path)

            self.backtrack_frontier_chain()

            self.frontier = self.extract_frontier()
            log_to_file(
                f"#### Step {step}: frontier after graph update\n\n{self.frontier}",
                log_path=self.log_path,
            )

            self.graph.generate_belief_text()
            self.resonate_memory()

            advance = self.short_term_memory.maybe_advance()
            if step >= self.session_max_steps:
                advance = True

            if advance:
                report_flag = "True" if step >= self.session_max_steps else "False"
                response_dict = self.actions.report_or_refine(
                    report_flag=report_flag,
                    graph_description=self.graph.to_dict(),
                    belief=self.graph.belief,
                    frontier=self.frontier,
                    memory_context=self.memory_context,
                )
                decision_type = response_dict["type"]

                if decision_type == 1:
                    answer = response_dict["answer"]
                    report = response_dict["report"]
                    self.last_answer = answer
                    self.last_report = report
                    return answer, report

                self.short_term_memory.refine_frontier(self.frontier, response_dict["candidates"])

    def extract_frontier(self):
        return self.short_term_memory.extract_frontier()

    def plan(self):
        expert_descriptions = self.agent_team.get("expert_descriptions", {})
        return self.actions.plan_expert_calls(
            frontier=self.frontier,
            expert_descriptions=expert_descriptions,
            memory_context=self.memory_context,
        )

    def act(self, plan):
        analyses = []
        for item in plan:
            expert_name = item["expert_name"]
            matching_experts = [agent for agent in self.experts if agent.name == expert_name]
            if not matching_experts:
                available = [agent.name for agent in self.experts]
                raise ValueError(f"Expert '{expert_name}' not found. Available experts: {available}")

            task = str(item.get("task") or "").strip()
            expert = matching_experts[0]
            analysis = expert.run(self.frontier, task=task)
            analyses.append({"expert_name": expert_name, "task": task, "analysis": analysis})

        return analyses

    def backtrack_frontier_chain(self) -> None:
        for message in self.short_term_memory.backtrack_frontier_chain(self.frontier):
            log_to_file(f"{message}\n\n", log_path=self.log_path)

    def export_diagnosis_artifact(self, groundtruth: str = "", expert_traces: list | None = None) -> dict:
        return {
            "incident_id": f"case_{self.case_id}",
            "case_index": self.case_id,
            "case_symptom": self.case_symptom,
            "observables": self.graph.get_observable_labels(),
            "final_belief": self.graph.belief,
            "frontier": self.frontier,
            "graphs": self.short_term_memory.snapshots,
            "prediction": self.last_answer,
            "report": self.last_report,
            "groundtruth": groundtruth,
            "memory_context": self.memory_context,
            "memory_resonance_traces": self.memory_resonance_traces,
            "expert_traces": expert_traces or [],
            "telemetry_refs": self.evidence_text,
        }





