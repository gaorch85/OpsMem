from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from ltm.store import CaseRecord, MemoryStore, PatternRecord, ProcedureRecord
from utils.embedding import EmbeddingConfig, EmbeddingProvider, build_embedding_provider
from utils.llm import llm, parse_json_response


@dataclass
class SignalCoupling:
    pattern_id: str
    signal: str
    score: float
    active: bool
    matched_query_signal: str

@dataclass
class PatternActivation:
    pattern: PatternRecord
    score: float
    alignment_score: float
    coverage_score: float
    active_signal_count: int
    total_signal_count: int
    signal_couplings: list[SignalCoupling]

@dataclass
class ProcedurePropagation:
    procedure: ProcedureRecord
    score: float
    via_pattern_id: str
    edge_weight: float

@dataclass
class CasePropagation:
    case: CaseRecord
    score: float
    via_pattern_id: str
    edge_weight: float

@dataclass
class MemoryPropagation:
    procedures: list[ProcedurePropagation]
    cases: list[CasePropagation]

@dataclass
class CrossMemoryResonanceResult:
    observables: list[str]
    query_signals: list[str]
    signal_coupling: dict[str, list[SignalCoupling]]
    pattern_activation: list[PatternActivation]
    memory_propagation: MemoryPropagation
    context: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pattern_hits(self) -> list[PatternActivation]:
        return self.pattern_activation

    @property
    def procedure_hits(self) -> list[ProcedurePropagation]:
        return self.memory_propagation.procedures

    @property
    def case_hits(self) -> list[CasePropagation]:
        return self.memory_propagation.cases

    def to_metadata(self) -> dict:
        return {
            "query_signals": self.query_signals,
            "signal_coupling": {
                pattern_id: [coupling.__dict__ for coupling in couplings]
                for pattern_id, couplings in self.signal_coupling.items()
            },
            "pattern_activation": [
                {
                    "pattern_id": item.pattern.pattern_id,
                    "signals": item.pattern.signals,
                    "root_cause": item.pattern.root_cause,
                    "score": item.score,
                    "alignment_score": item.alignment_score,
                    "coverage_score": item.coverage_score,
                    "active_signal_count": item.active_signal_count,
                    "total_signal_count": item.total_signal_count,
                }
                for item in self.pattern_activation
            ],
            "memory_propagation": {
                "procedures": [
                    {
                        "procedure_id": item.procedure.procedure_id,
                        "via_pattern_id": item.via_pattern_id,
                        "edge_weight": item.edge_weight,
                        "score": item.score,
                    }
                    for item in self.memory_propagation.procedures
                ],
                "cases": [
                    {
                        "case_id": item.case.case_id,
                        "via_pattern_id": item.via_pattern_id,
                        "edge_weight": item.edge_weight,
                        "score": item.score,
                    }
                    for item in self.memory_propagation.cases
                ],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "observables": self.observables,
            "query_signals": self.query_signals,
            "context": self.context,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class CrossMemoryResonance:
    def __init__(
        self,
        long_term_memory=None,
        llm_args: dict[str, Any] | None = None,
        cmr_config: dict[str, Any] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.long_term_memory = None
        self.store: MemoryStore | None = None
        self.llm_args = llm_args or {}
        self.cmr_config = cmr_config or {}
        self.embedding_provider = embedding_provider
        self.traces: list[CrossMemoryResonanceResult] = []

        cfg = self.cmr_config
        thresholds = cfg.get("thresholds") or {}
        self.signal_top_k = cfg.get("signal_top_k", 10)
        self.pattern_top_k = cfg.get("pattern_top_k", 3)
        self.procedure_top_k = cfg.get("procedure_top_k", 3)
        self.case_top_k = cfg.get("case_top_k", 3)
        self.signal_threshold = thresholds.get("signal", 0.6)
        self.pattern_threshold = thresholds.get("pattern", 0.6)
        self.procedure_threshold = thresholds.get("procedure", 0.6)
        self.case_threshold = thresholds.get("case", 0.6)
        self.pattern_alignment_weight = cfg.get("pattern_alignment_weight", 0.5)
        self.pattern_coverage_weight = cfg.get("pattern_coverage_weight", 0.5)

        self._procedure_by_id: dict[str, ProcedureRecord] = {}
        self._case_by_id: dict[str, CaseRecord] = {}
        self._signal_index: list[tuple[str, str]] = []
        self._signal_embeddings: np.ndarray | None = None
        self.set_long_term_memory(long_term_memory)

    def set_long_term_memory(self, long_term_memory) -> None:
        self.long_term_memory = long_term_memory
        if long_term_memory is None:
            self.store = None
            self._procedure_by_id = {}
            self._case_by_id = {}
            self._signal_index = []
            self._signal_embeddings = None
            return

        self.store = long_term_memory.store
        self._procedure_by_id = {item.procedure_id: item for item in self.store.procedures}
        self._case_by_id = {item.case_id: item for item in self.store.cases}
        self._signal_index = [
            (pattern.pattern_id, signal)
            for pattern in self.store.patterns
            for signal in pattern.signals
        ]

        if self.embedding_provider is None:
            self.embedding_provider = build_embedding_provider(EmbeddingConfig())
        self._signal_embeddings = (
            self.embedding_provider.encode([signal for _pattern_id, signal in self._signal_index])
            if self._signal_index
            else np.empty((0, 0), dtype=float)
        )

    def set_llm_args(self, llm_args: dict[str, Any] | None) -> None:
        self.llm_args = llm_args or {}

    def resonate(self, short_term_memory=None, graph=None) -> CrossMemoryResonanceResult:
        active_graph = graph
        if active_graph is None and short_term_memory is not None:
            active_graph = short_term_memory.graph

        observables = []
        if active_graph is not None:
            observables = active_graph.get_observable_labels({"Symptom", "Evidence"})

        if self.store is None:
            result = CrossMemoryResonanceResult(
                observables=observables,
                query_signals=[],
                signal_coupling={},
                pattern_activation=[],
                memory_propagation=MemoryPropagation(procedures=[], cases=[]),
                context="",
                created_at=datetime.now().isoformat(timespec="seconds"),
                metadata={"enabled": False},
            )
            self.traces.append(result)
            return result

        result = self.run(observables)
        metadata = result.to_metadata()
        result.metadata = {
            "enabled": True,
            "cmr_steps": {
                "step1_signal_coupling": {
                    "observables": observables,
                    "query_signals": result.query_signals,
                    "couplings": metadata["signal_coupling"],
                },
                "step2_pattern_activation": metadata["pattern_activation"],
                "step3_memory_propagation": metadata["memory_propagation"],
            },
        }
        self.traces.append(result)
        return result

    def run(self, observables: list[str]) -> CrossMemoryResonanceResult:
        query_signals, signal_coupling = self.signal_coupling(observables)
        pattern_activation = self.pattern_activation(signal_coupling)
        memory_propagation = self.memory_propagation(pattern_activation)
        context = _build_memory_context(query_signals, signal_coupling, pattern_activation, memory_propagation)
        return CrossMemoryResonanceResult(
            observables=observables,
            query_signals=query_signals,
            signal_coupling=signal_coupling,
            pattern_activation=pattern_activation,
            memory_propagation=memory_propagation,
            context=context,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    def signal_coupling(self, observables: Iterable[str]) -> tuple[list[str], dict[str, list[SignalCoupling]]]:
        query_signals = self.standardize_observables(observables)[: self.signal_top_k]
        grouped: dict[str, list[SignalCoupling]] = {
            pattern.pattern_id: [] for pattern in self.store.patterns
        } if self.store is not None else {}
        if not query_signals:
            return query_signals, grouped

        raw_scores = self._embedding_signal_scores(query_signals)

        for pattern_id, signal, score, matched_query_signal in raw_scores:
            active_score = score if score >= self.signal_threshold else 0.0
            grouped.setdefault(pattern_id, []).append(
                SignalCoupling(
                    pattern_id=pattern_id,
                    signal=signal,
                    score=float(active_score),
                    active=active_score > 0.0,
                    matched_query_signal=matched_query_signal,
                )
            )
        return query_signals, grouped

    def standardize_observables(self, observables: Iterable[str]) -> list[str]:
        raw_observables = [str(item).strip() for item in observables if str(item).strip()]
        if not raw_observables:
            return []
        if not self.llm_args:
            raise ValueError("CrossMemoryResonance requires configured LLM args for signal standardization.")

        system_prompt = """
You normalize short-term diagnostic memory into reusable operational query signals.
Given symptom/evidence texts from the current incident, output concise normalized signals for long-term memory matching.
Rules:
- Preserve concrete components, metrics, log symptoms, and failure surfaces.
- Remove duplicate wording and incidental prose.
- Do not infer a root cause unless it is explicitly stated by the evidence.
- Return JSON only: {"signals": ["..."]}.
"""
        user_prompt = "Short-term memory observables:\n" + "\n".join(f"- {item}" for item in raw_observables)
        response = llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_path=self.llm_args.get("model_path"),
            temperature=min(float(self.llm_args.get("temperature", 0.2)), 0.3),
            max_tokens=min(int(self.llm_args.get("max_tokens", 1024)), 1024),
            return_meta=False,
        )
        payload = parse_json_response(response)
        signals = payload.get("signals") or []
        return _dedupe([str(signal).strip() for signal in signals if str(signal).strip()])

    def pattern_activation(self, signal_coupling: dict[str, list[SignalCoupling]]) -> list[PatternActivation]:
        if self.store is None:
            return []
        pattern_activation: list[PatternActivation] = []
        for pattern in self.store.patterns:
            couplings = signal_coupling.get(pattern.pattern_id, [])
            total_signal_count = max(len(pattern.signals), 1)
            active_scores = [coupling.score for coupling in couplings if coupling.active]
            active_signal_count = len(active_scores)
            alignment_score = sum(active_scores) / active_signal_count if active_scores else 0.0
            coverage_score = active_signal_count / total_signal_count
            score = (
                self.pattern_alignment_weight * alignment_score
                + self.pattern_coverage_weight * coverage_score
            )
            if score < self.pattern_threshold:
                continue
            pattern_activation.append(
                PatternActivation(
                    pattern=pattern,
                    score=float(score),
                    alignment_score=float(alignment_score),
                    coverage_score=float(coverage_score),
                    active_signal_count=active_signal_count,
                    total_signal_count=total_signal_count,
                    signal_couplings=couplings,
                )
            )

        pattern_activation.sort(key=lambda item: item.score, reverse=True)
        return pattern_activation[: self.pattern_top_k]


    def memory_propagation(self, pattern_activation: Iterable[PatternActivation]) -> MemoryPropagation:
        return MemoryPropagation(
            procedures=self.propagate_to_procedures(pattern_activation),
            cases=self.propagate_to_cases(pattern_activation),
        )

    def propagate_to_procedures(self, pattern_activation: Iterable[PatternActivation]) -> list[ProcedurePropagation]:
        pattern_score_map = {item.pattern.pattern_id: item.score for item in pattern_activation}
        procedure_propagation: list[ProcedurePropagation] = []
        for edge in self.store.pattern_procedure_edges if self.store is not None else []:
            pattern_score = pattern_score_map.get(edge.pattern_id)
            if pattern_score is None:
                continue
            procedure = self._procedure_by_id.get(edge.procedure_id)
            if procedure is None:
                continue
            score = pattern_score * edge.weight
            if score < self.procedure_threshold:
                continue
            procedure_propagation.append(
                ProcedurePropagation(
                    procedure=procedure,
                    score=float(score),
                    via_pattern_id=edge.pattern_id,
                    edge_weight=edge.weight,
                )
            )
        procedure_propagation.sort(key=lambda item: item.score, reverse=True)
        return procedure_propagation[: self.procedure_top_k]

    def propagate_to_cases(self, pattern_activation: Iterable[PatternActivation]) -> list[CasePropagation]:
        pattern_score_map = {item.pattern.pattern_id: item.score for item in pattern_activation}
        case_propagation: list[CasePropagation] = []
        for edge in self.store.pattern_case_edges if self.store is not None else []:
            pattern_score = pattern_score_map.get(edge.pattern_id)
            if pattern_score is None:
                continue
            case = self._case_by_id.get(edge.case_id)
            if case is None:
                continue
            score = pattern_score * edge.weight
            if score < self.case_threshold:
                continue
            case_propagation.append(
                CasePropagation(
                    case=case,
                    score=float(score),
                    via_pattern_id=edge.pattern_id,
                    edge_weight=edge.weight,
                )
            )
        case_propagation.sort(key=lambda item: item.score, reverse=True)
        return case_propagation[: self.case_top_k]

    def _embedding_signal_scores(self, query_signals: list[str]) -> list[tuple[str, str, float, str]]:
        if not self._signal_index:
            return []
        query_embeddings = self.embedding_provider.encode(query_signals)
        if query_embeddings.size == 0 or self._signal_embeddings is None or self._signal_embeddings.size == 0:
            return []
        score_matrix = query_embeddings @ self._signal_embeddings.T
        best_query_indices = score_matrix.argmax(axis=0)
        best_scores = score_matrix.max(axis=0)
        results: list[tuple[str, str, float, str]] = []
        for idx, (pattern_id, signal) in enumerate(self._signal_index):
            query_idx = int(best_query_indices[idx])
            results.append((pattern_id, signal, float(best_scores[idx]), query_signals[query_idx]))
        return results

    def export_traces(self) -> list[dict[str, Any]]:
        return [trace.to_dict() for trace in self.traces]


def _dedupe(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _build_memory_context(
    query_signals: list[str],
    signal_coupling: dict[str, list[SignalCoupling]],
    pattern_activation: list[PatternActivation],
    memory_propagation: MemoryPropagation,
) -> str:
    if not query_signals or (not pattern_activation and not memory_propagation.procedures and not memory_propagation.cases):
        return ""

    lines: list[str] = [
        "Cross-memory resonance results:",
        "Step 1 - standardized STM query signals:",
    ]
    for signal in query_signals:
        lines.append(f"- {signal}")

    if pattern_activation:
        lines.append("Step 2 - activated patterns:")
        for item in pattern_activation:
            active_signals = [coupling.signal for coupling in item.signal_couplings if coupling.active]
            active_text = "; ".join(active_signals) if active_signals else "none"
            lines.append(
                f"- [{item.pattern.pattern_id}] {item.pattern.content} "
                f"root_cause={item.pattern.root_cause}; "
                f"score={item.score:.3f}, alignment={item.alignment_score:.3f}, "
                f"coverage={item.coverage_score:.3f}, active_signals={active_text}"
            )

    if memory_propagation.procedures:
        lines.append("Step 3 - propagated procedures:")
        for item in memory_propagation.procedures:
            symptom_text = ", ".join(item.procedure.symptoms) if item.procedure.symptoms else "N/A"
            lines.append(
                f"- [{item.procedure.procedure_id}] {item.procedure.content} "
                f"(via={item.via_pattern_id}, edge_weight={item.edge_weight:.3f}, "
                f"score={item.score:.3f}, symptoms={symptom_text})"
            )

    if memory_propagation.cases:
        lines.append("Step 3 - propagated cases:")
        for item in memory_propagation.cases:
            symptom_text = ", ".join(item.case.symptoms) if item.case.symptoms else "N/A"
            lines.append(
                f"- [{item.case.case_id}] root_cause={item.case.root_cause}; content={item.case.content} "
                f"(via={item.via_pattern_id}, edge_weight={item.edge_weight:.3f}, "
                f"score={item.score:.3f}, symptoms={symptom_text})"
            )

    return "\n".join(lines)








