from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import llm, parse_json_response


class BeliefGraph:
    """Short-term belief graph for one OpsMem diagnosis session."""

    VALID_NODE_TYPES = {"Symptom", "Evidence", "Hypothesis"}
    VALID_EDGE_TYPES = {"derive", "support", "refute", "refines"}

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}
        self.start_symptom_id: Optional[str] = None
        self.belief = ""
        self.idx = 1
        self.level_nodes: dict[int, list[tuple[str, float]]] = {}

    def add_node(self, node_type: str, label: str, score: float = 1.0, attrs: Optional[Dict] = None) -> str:
        if node_type not in self.VALID_NODE_TYPES:
            raise ValueError(f"Node type must be one of {self.VALID_NODE_TYPES}, got {node_type}.")

        node_id = f"node-{self.idx}"
        self.idx += 1
        self.nodes[node_id] = {
            "type": node_type,
            "label": label,
            "score": score,
            "attrs": attrs or {},
        }
        return node_id

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self.nodes.get(node_id)

    def update_node(self, node_id: str, score: float, why: str | dict[str, Any]):
        if node_id not in self.nodes:
            return
        self.nodes[node_id]["score"] = score
        if isinstance(why, dict) and set(why) == {"why"}:
            why = why["why"]
        self.nodes[node_id]["attrs"]["why"] = why

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def add_edge(self, src: str, dst: str, edge_type: str, attrs: Optional[Dict] = None):
        if edge_type not in self.VALID_EDGE_TYPES:
            raise ValueError(f"Edge type must be one of {self.VALID_EDGE_TYPES}, got {edge_type}.")
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Cannot add edge with unknown node: {src} -> {dst}")
        self.edges[(src, dst)] = {
            "type": edge_type,
            "attrs": attrs or {},
        }

    def get_edge(self, src: str, dst: str) -> Optional[Dict]:
        return self.edges.get((src, dst))

    def update_edge_conf(self, src: str, dst: str, new_conf: float):
        if (src, dst) in self.edges:
            self.edges[(src, dst)]["conf"] = max(1e-6, min(1 - 1e-6, new_conf))

    def apply_evidence(
        self,
        evidence_id: str,
        hypo_id: str,
        relation: str,
        strength: float,
        provenance: Optional[Dict] = None,
    ):
        if self.nodes[evidence_id]["type"] != "Evidence":
            raise TypeError("evidence_id must reference an Evidence node.")
        if self.nodes[hypo_id]["type"] != "Hypothesis":
            raise TypeError("hypo_id must reference a Hypothesis node.")

        if relation == "support":
            self.link_support(evidence_id, hypo_id, strength, provenance)
        elif relation == "refute":
            self.link_refute(evidence_id, hypo_id, strength, provenance)
        else:
            raise ValueError("relation must be 'support' or 'refute'.")

    def link_support(
        self,
        evidence_id: str,
        hypo_id: str,
        strength: float = 1.0,
        provenance: Optional[Dict] = None,
    ):
        edge_attrs = dict(provenance or {})
        edge_attrs["strength"] = strength
        self.add_edge(evidence_id, hypo_id, "support", edge_attrs)
        self.nodes[hypo_id]["attrs"]["has_evidence"] = True

    def link_refute(
        self,
        evidence_id: str,
        hypo_id: str,
        strength: float,
        provenance: Optional[Dict] = None,
    ):
        edge_attrs = dict(provenance or {})
        edge_attrs["strength"] = strength
        self.add_edge(evidence_id, hypo_id, "refute", edge_attrs)
        self.nodes[hypo_id]["attrs"]["has_evidence"] = True

    def link_refines(
        self,
        src_id: str,
        dst_id: str,
        prob: float | None = None,
        base_prior: float = 1.0,
        attrs: Dict | None = None,
    ):
        src_type = self.nodes[src_id]["type"]
        if src_type not in {"Symptom", "Hypothesis"}:
            raise TypeError("src_id must reference a Symptom or Hypothesis node.")
        if self.nodes[dst_id]["type"] != "Hypothesis":
            raise TypeError("dst_id must reference a Hypothesis node.")

        edge_attrs = dict(attrs or {})
        if prob is not None:
            edge_attrs["prob"] = prob
        if base_prior != 1.0:
            edge_attrs["base_prior"] = base_prior
        self.add_edge(src_id, dst_id, "refines", edge_attrs)


    def to_dict(self) -> Dict:
        edge_list = []
        for (src, dst), edge in self.edges.items():
            edge_list.append({"src": src, "dst": dst, **edge})
        return {
            "nodes": self.nodes,
            "edges": edge_list,
            "start_symptom_id": self.start_symptom_id,
        }

    def pretty_lines(self, max_label_len: int = 60, max_attrs_len: int = 80) -> List[str]:
        def _short(value: Any, max_len: int) -> str:
            text = str(value)
            return text if len(text) <= max_len else f"{text[:max_len - 1]}..."

        def _attrs_str(obj: dict, max_len: int) -> str:
            if not obj:
                return ""
            try:
                payload = json.dumps(obj, ensure_ascii=False)
            except Exception:
                payload = str(obj)
            return _short(payload, max_len)

        lines: List[str] = ["=== Nodes ==="]
        for node_id, node in self.nodes.items():
            attrs = node.get("attrs", {}) or {}
            level = attrs.get("level", "-")
            label = _short(node.get("label") or "-", max_label_len)
            score = float(node.get("score", 0.0))
            line = f"[{node_id[:6]}] {node['type']}[{level}] | {label} | score={score:.3f}"
            attrs_part = _attrs_str(attrs, max_attrs_len)
            if attrs_part:
                line += f" | attrs={attrs_part}"
            lines.append(line)

        lines.append("=== Edges ===")
        for (src, dst), edge in self.edges.items():
            src_label = _short(self.nodes.get(src, {}).get("label") or "-", max_label_len)
            dst_label = _short(self.nodes.get(dst, {}).get("label") or "-", max_label_len)
            edge_type = edge.get("type", "?")
            attrs_part = _attrs_str(edge.get("attrs", {}) or {}, max_attrs_len)
            line = f"{src[:6]}[{src_label}] -> {dst[:6]}[{dst_label}] | {edge_type}"
            if attrs_part:
                line += f" | attrs={attrs_part}"
            lines.append(line)

        return lines

    def get_observable_labels(self, node_types: Optional[set[str]] = None) -> List[str]:
        target_types = node_types or {"Symptom", "Evidence"}
        labels: List[str] = []
        for node in self.nodes.values():
            if node.get("type") not in target_types:
                continue
            label = str(node.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def generate_belief_text(self):
        system_prompt = """Generate a concise reasoning consensus from the current OpsMem inference graph.
Return JSON only with this exact schema: {"belief": "..."}.
Base the belief only on the provided Symptom, Evidence, and Hypothesis nodes and their edges."""
        user_prompt = f"""Current inference graph:
{json.dumps(self.to_dict(), ensure_ascii=False, indent=2)}"""
        response, _meta = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=getattr(self, "model_path", None),
            temperature=0.5,
            max_tokens=2048,
            return_meta=True,
        )
        response_dict = parse_json_response(response)
        self.belief = response_dict["belief"]

    def update_level_nodes(self):
        self.level_nodes = {}
        for node_id, node_info in self.nodes.items():
            if node_info["type"] != "Hypothesis" or "level" not in node_info["attrs"]:
                continue
            level = node_info["attrs"]["level"]
            confidence = node_info.get("score", 0.0)
            self.level_nodes.setdefault(level, []).append((node_id, confidence))

        for level in self.level_nodes:
            self.level_nodes[level].sort(key=lambda item: item[1], reverse=True)

    def delete_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]
        self.edges = {
            (src, dst): attrs
            for (src, dst), attrs in self.edges.items()
            if src != node_id and dst != node_id
        }
        self.update_level_nodes()

    def delete_level_nodes(self, target_level):
        level_node_ids = []
        if target_level in self.level_nodes:
            level_node_ids = [node_id for node_id, _ in self.level_nodes[target_level]]

        for node_id in level_node_ids:
            self.delete_node(node_id)

        self.update_level_nodes()

    def is_highest_conf_in_level(self, node_id):
        if node_id not in self.nodes:
            return False
        node_info = self.nodes[node_id]
        if node_info["type"] != "Hypothesis" or "level" not in node_info["attrs"]:
            return False

        level = node_info["attrs"]["level"]
        if level not in self.level_nodes or len(self.level_nodes[level]) == 0:
            return False

        highest_node_id = self.level_nodes[level][0][0]
        return highest_node_id == node_id

