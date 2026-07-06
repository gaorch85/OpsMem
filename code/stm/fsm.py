from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union


class BeliefFSM:
    """Finite-state controller for layered hypothesis refinement.

    The FSM does not decide the final root cause by itself. It decides when the
    central agent should ask the LLM to either report the current frontier or
    refine it into a deeper hypothesis layer.
    """

    def __init__(self, thresholds: Optional[Dict] = None):
        default_thresholds = {
            "gap_delta": 0.15,
            "min_support": 1,
            "max_steps": 5,
        }
        self.thresholds = {**default_thresholds, **(thresholds or {})}
        self.state: Union[int, str] = 1
        self.history: List[Union[int, str]] = [self.state]
        self._stage_steps: Dict[int, int] = {1: 0}

    def get_state(self) -> Union[int, str]:
        return self.state

    def is_final_state(self) -> bool:
        return self.state == "report"

    def reset(self) -> None:
        self.state = 1
        self.history = [self.state]
        self._stage_steps = {1: 0}

    def set_state(self, new_state: Union[int, str]) -> None:
        if isinstance(new_state, int):
            if new_state < 1:
                raise ValueError(f"FSM state must be a positive integer or 'report', got {new_state}.")
            self._stage_steps.setdefault(new_state, 0)
        elif new_state != "report":
            raise ValueError(f"FSM state must be a positive integer or 'report', got {new_state}.")
        self.state = new_state
        self.history.append(new_state)

    def tick_step(self, k: int = 1) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"Step increment must be a positive integer, got {k}.")
        if isinstance(self.state, int):
            self._stage_steps.setdefault(self.state, 0)
            self._stage_steps[self.state] += k

    def maybe_transit(self, G) -> bool:
        """Return True when the central agent should report or refine."""
        if self.state == "report":
            return True
        candidates = self._top_hypos(G, level=int(self.state))
        if not candidates:
            self.set_state("report")
            return True

        top1_id, _top1_score = candidates[0]
        gap, _ = self._gap_and_top1(candidates)
        support_count = self._count_support_edges(G, top1_id)
        current_steps = self._stage_steps.get(int(self.state), 0)

        if gap >= self.thresholds.get("gap_delta", 0.0) and support_count >= self.thresholds.get("min_support", 0):
            return True
        if current_steps >= self.thresholds.get("max_steps", 10**9):
            return True
        return False

    def _top_hypos(self, G, level: int, k: Optional[int] = None) -> List[Tuple[str, float]]:
        candidates: List[Tuple[str, float]] = []
        for nid, node in G.nodes.items():
            if node.get("type") != "Hypothesis":
                continue
            node_level = int(node.get("attrs", {}).get("level", -1))
            if node_level != level:
                continue
            candidates.append((nid, float(node.get("score", 0.0))))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates if k is None else candidates[:k]

    def _gap_and_top1(self, candidates: List[Tuple[str, float]]) -> Tuple[float, Tuple[str, float]]:
        if not candidates:
            return 0.0, ("", 0.0)
        top1_id, top1_score = candidates[0]
        if len(candidates) == 1:
            return top1_score, (top1_id, top1_score)
        return top1_score - candidates[1][1], (top1_id, top1_score)

    def _count_support_edges(self, G, hypo_id: str) -> int:
        support_count = 0
        for (src_id, dst_id), edge in G.edges.items():
            if dst_id != hypo_id or edge.get("type") != "support":
                continue
            src_node = G.nodes.get(src_id, {})
            if src_node.get("type") == "Evidence":
                support_count += 1
        return support_count
