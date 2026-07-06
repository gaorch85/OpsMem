from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stm.fsm import BeliefFSM
from stm.graph import BeliefGraph


@dataclass
class ShortTermMemory:
    """OpsMem short-term memory: a belief graph plus its diagnostic state machine."""

    graph: BeliefGraph
    fsm: BeliefFSM
    snapshots: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, thresholds: dict | None = None) -> "ShortTermMemory":
        return cls(
            graph=BeliefGraph(),
            fsm=BeliefFSM(thresholds=thresholds or {}),
        )

    def tick_step(self, k: int = 1) -> None:
        self.fsm.tick_step(k)

    def maybe_advance(self) -> bool:
        return self.fsm.maybe_transit(G=self.graph)

    def observable_labels(self) -> list[str]:
        return self.graph.get_observable_labels()

    def snapshot(self) -> dict[str, Any]:
        return self.graph.to_dict()

    def record_snapshot(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        self.snapshots.append(snapshot)
        return snapshot

    def initialize_from_ingest_response(self, symptom_node: str, isolated_evidence: list[str]) -> tuple[str, list[str], dict[str, str]]:
        symptom_node_id = self.graph.add_node(node_type="Symptom", label=symptom_node)
        isolated_evidence_ids = [
            self.graph.add_node(node_type="Evidence", label=evidence)
            for evidence in isolated_evidence
        ]
        evidence_id_map: dict[str, str] = {}
        for idx, real_id in enumerate(isolated_evidence_ids, start=1):
            evidence_id_map[real_id] = real_id
            evidence_id_map[f"ev{idx:03d}"] = real_id
        self.record_snapshot()
        return symptom_node_id, isolated_evidence_ids, evidence_id_map

    def add_initial_hypotheses(
        self,
        symptom_node_id: str,
        evidence_id_map: dict[str, str],
        candidates: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, str]:
        hypo_id_to_real_id: dict[str, str] = {}
        for hypo in candidates:
            real_node_id = self.graph.add_node(
                node_type="Hypothesis",
                label=hypo["label"],
                score=hypo["confidence"],
                attrs={"why": hypo["why"], "level": 1},
            )
            hypo_id_to_real_id[hypo["id"]] = real_node_id

        for hypo_id in hypo_id_to_real_id.values():
            self.graph.add_edge(src=symptom_node_id, dst=hypo_id, edge_type="derive")

        for edge in edges:
            real_src_id = evidence_id_map.get(edge["src"], edge["src"])
            real_dst_id = hypo_id_to_real_id[edge["dst"]]
            self.graph.add_edge(src=real_src_id, dst=real_dst_id, edge_type=edge["relation"])

        self.graph.update_level_nodes()
        self.record_snapshot()
        return hypo_id_to_real_id

    def apply_proposal(
        self,
        symptom_node_id: str,
        frontier: dict[str, Any],
        edits: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> list[str]:
        warnings: list[str] = []
        for edit_item in edits:
            target_node_id = edit_item["node_id"]
            if not self.graph.has_node(target_node_id):
                warnings.append(f"Warning: node {target_node_id} does not exist; skipping update.")
                continue
            self.graph.update_node(
                node_id=target_node_id,
                score=edit_item["confidence"],
                why=edit_item["why"],
            )

        temp_id_to_real_id: dict[str, str] = {}
        for node in nodes:
            temp_node_id = node["id"]
            node_type = node["node_type"]
            node_label = node["label"]
            if node_type == "Evidence":
                real_node_id = self.graph.add_node(node_type=node_type, label=node_label)
            elif node_type == "Hypothesis":
                real_node_id = self.graph.add_node(
                    node_type=node_type,
                    label=node_label,
                    score=node["confidence"],
                    attrs={"why": node["why"], "level": frontier["level"]},
                )
                self.graph.add_edge(src=symptom_node_id, dst=real_node_id, edge_type="refines")
            else:
                warnings.append(f"Warning: unknown node type {node_type}; skipping node {temp_node_id}.")
                continue
            temp_id_to_real_id[temp_node_id] = real_node_id

        self.record_snapshot()

        for edge in edges:
            src_real_id = temp_id_to_real_id.get(edge["src"], edge["src"])
            dst_real_id = temp_id_to_real_id.get(edge["dst"], edge["dst"])
            self.graph.add_edge(src=src_real_id, dst=dst_real_id, edge_type=edge["relation"])

        self.graph.update_level_nodes()
        return warnings

    def refine_frontier(self, frontier: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
        frontier_id = frontier["node_id"]
        candidate_ids: list[str] = []
        for candidate in candidates:
            candidate_id = self.graph.add_node(
                node_type="Hypothesis",
                label=candidate["label"],
                score=candidate["confidence"],
                attrs={"why": candidate["why"], "level": frontier["level"] + 1},
            )
            candidate_ids.append(candidate_id)
        for candidate_id in candidate_ids:
            self.graph.add_edge(frontier_id, candidate_id, edge_type="refines")
        self.fsm.set_state(self.fsm.state + 1)
        self.record_snapshot()
        return candidate_ids

    def extract_frontier(self) -> dict[str, Any]:
        """Return the highest-scoring hypothesis in the current STM level."""
        if not getattr(self.graph, "nodes", None):
            return {}

        try:
            cur_level = self.fsm.get_state()
        except Exception:
            cur_level = None

        hypos: list[tuple[str, dict[str, Any]]] = []
        for nid, node in self.graph.nodes.items():
            if node.get("type") != "Hypothesis":
                continue
            level = node.get("attrs", {}).get("level")
            if cur_level is not None and level != cur_level:
                continue
            hypos.append((nid, node))

        if not hypos:
            return {}

        hypos.sort(key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
        node_id, node = hypos[0]
        supports = 0
        refutes = 0
        for (_src, dst), edge in self.graph.edges.items():
            if dst != node_id:
                continue
            if edge.get("type") == "support":
                supports += 1
            if edge.get("type") == "refute":
                refutes += 1

        return {
            "node_id": node_id,
            "label": node.get("label"),
            "why": node.get("attrs", {}).get("why"),
            "score": float(node.get("score", 0.0)),
            "level": node.get("attrs", {}).get("level"),
            "supports": supports,
            "refutes": refutes,
        }

    def backtrack_frontier_chain(self, frontier: dict[str, Any]) -> list[str]:
        """Remove deeper STM levels when an ancestor is no longer top-ranked."""
        if not frontier:
            return []

        current_frontier_id = frontier.get("node_id")
        current_level = frontier.get("level")
        if current_frontier_id is None or current_level is None:
            return []
        if not self.graph.has_node(current_frontier_id):
            return []

        node_info = self.graph.nodes[current_frontier_id]
        if node_info.get("type") != "Hypothesis":
            return []

        def find_ancestor_node(target_level: int, current_node_id: str):
            if target_level == current_level:
                return current_node_id
            parent_edges = [
                {"src": src, "dst": dst, "type": attrs["type"], "attrs": attrs["attrs"]}
                for (src, dst), attrs in self.graph.edges.items()
                if dst == current_node_id and attrs["type"] in {"derive", "refines"}
            ]
            if not parent_edges:
                return None
            parent_id = parent_edges[0]["src"]
            parent_level = self.graph.nodes[parent_id]["attrs"].get("level") if parent_id in self.graph.nodes else -1
            if parent_level == target_level:
                return parent_id
            return find_ancestor_node(target_level, parent_id)

        level_to_node_id = {}
        for level in range(1, int(current_level) + 1):
            node_id = find_ancestor_node(level, current_frontier_id)
            if node_id:
                level_to_node_id[level] = node_id

        invalid_start_level = None
        for level in range(1, int(current_level) + 1):
            node_id = level_to_node_id.get(level)
            if not node_id:
                continue
            if not self.graph.is_highest_conf_in_level(node_id):
                invalid_start_level = level
                break

        if invalid_start_level is None:
            return []

        messages = [
            f"Node {level_to_node_id[invalid_start_level]} at level {invalid_start_level} is no longer top-ranked; pruning deeper hypotheses."
        ]
        self.fsm.set_state(invalid_start_level)
        for delete_level in range(invalid_start_level + 1, int(current_level) + 1):
            self.graph.delete_level_nodes(delete_level)
            messages.append(f"Deleted all nodes and related edges at level {delete_level}.")
        return messages
