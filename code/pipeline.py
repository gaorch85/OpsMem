from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agents.central_agent import CentralAgent
from agents.expert_agent import ExpertAgent
from cmr import CrossMemoryResonance
from consolidation import MemoryConsolidator
from ltm import build_long_term_memory
from consolidation.workflow.run_context import prepare_consolidation_run_context, resolve_store_paths
from stm import ShortTermMemory
from utils.llm import print_usage_summary


ANSWER_COLUMNS = ["prediction", "report", "answer"]
CSV_WRITE_OPTIONS = {
    "index": False,
    "encoding": "utf-8",
    "quoting": csv.QUOTE_ALL,
    "quotechar": '"',
    "escapechar": '"',
    "sep": ",",
}

CONFIG_FILE_NAME = "config.yaml"
REQUIRED_LLM_ARGS = ["temperature", "max_tokens", "return_meta", "session_max_steps", "max_retrieval_steps"]
DEFAULT_AGENT_TEAM = {
    "head_name": "SRE_Commander",
    "expert_names": ["Linux_Agent", "DBA_Agent", "Kubernetes_Agent", "Network_Agent"],
    "expert_descriptions": {
        "Linux_Agent": "Investigates host, process, resource, shell, log, and metric evidence.",
        "DBA_Agent": "Investigates database availability, connection pools, slow queries, and DB-related symptoms.",
        "Kubernetes_Agent": "Investigates pod/container restarts, readiness, scheduling, and resource pressure.",
        "Network_Agent": "Investigates timeout, packet loss, DNS, connectivity, and dependency symptoms.",
    },
}


def _load_pipeline_config(base_dir: Path, model_name: str | None = None) -> dict[str, Any]:
    raw_config = _read_yaml(base_dir / CONFIG_FILE_NAME)
    model_section = _require_section(raw_config, "model")
    agent_section = _require_section(raw_config, "agent")
    stm = _require_section(raw_config, "stm")
    ltm = dict(_require_section(raw_config, "ltm"))
    cmr = raw_config.get("cmr") or {}
    if not isinstance(cmr, dict):
        raise ValueError(f"{CONFIG_FILE_NAME} must define 'cmr' as a mapping when present.")
    consolidation = raw_config.get("consolidation") or {}
    if not isinstance(consolidation, dict):
        raise ValueError(f"{CONFIG_FILE_NAME} must define 'consolidation' as a mapping when present.")
    llm_args = dict(_require_section(agent_section, "llm_args"))
    _require_keys(llm_args, REQUIRED_LLM_ARGS, "agent.llm_args")

    resolved_model_name = str(model_name or model_section.get("current_model") or "").strip()
    if not resolved_model_name:
        raise ValueError(f"model.current_model is required in {CONFIG_FILE_NAME}.")
    os.environ["OPSMEM_MODEL_NAME"] = resolved_model_name
    llm_args["model_path"] = resolved_model_name

    return {
        "agent_team": DEFAULT_AGENT_TEAM,
        "stm": stm,
        "agent": {"llm_args": llm_args},
        "ltm": ltm,
        "cmr": cmr,
        "consolidation": dict(consolidation),
        "model_name": resolved_model_name,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return payload


def _require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"{CONFIG_FILE_NAME} must define a '{key}' mapping.")
    return section


def _require_keys(config: dict[str, Any], keys: list[str], section_name: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{CONFIG_FILE_NAME} missing required keys in {section_name}: {joined}")




def load_microservice_cases(dataset_dir: str | Path = "datasets") -> tuple[list[tuple[int, str]], list[dict[str, Any]], list[str]]:
    """Load sanitized operational cases as symptoms, telemetry references, and ground-truth answers."""
    dataset_path = Path(dataset_dir)
    case_dirs = sorted(
        [path for path in dataset_path.iterdir() if path.is_dir() and path.name.startswith("case_")],
        key=lambda path: int(re.search(r"\d+$", path.name).group()),
    )

    symptoms: list[tuple[int, str]] = []
    evidence_texts: list[dict[str, Any]] = []
    answers: list[str] = []

    for idx, case_root in enumerate(case_dirs):
        alarm_path = case_root / "alarm.json"
        answer_path = case_root / "answer.txt"
        telemetry_dir = case_root / "telemetry"

        with alarm_path.open("r", encoding="utf-8") as f:
            alarms = json.load(f)
        symptom_text, default_ts = _build_alarm_description(alarms)

        answers.append(answer_path.read_text(encoding="utf-8").strip())
        symptoms.append((idx, symptom_text))
        evidence_texts.append(
            {
                "telemetry_dir": str(telemetry_dir),
                "alarms": alarms,
                "default_ts": default_ts,
            }
        )

    return symptoms, evidence_texts, answers


def _build_alarm_description(alarms: list[dict[str, Any]]) -> tuple[str, float | None]:
    lines: list[str] = []
    default_ts = None
    for idx, alarm in enumerate(alarms, start=1):
        labels = alarm.get("labels", {}) or {}
        alert_name = alarm.get("alertName") or labels.get("alertname") or "unknown"
        level = alarm.get("level") or labels.get("level") or "P?"
        instance = labels.get("instance") or alarm.get("instance") or "unknown"
        app = labels.get("app_name") or alarm.get("appName") or ""
        desc = (
            labels.get("description")
            or labels.get("description_en")
            or alarm.get("description")
            or alarm.get("descriptionEn")
            or ""
        )
        value = labels.get("itemvalue") or alarm.get("currentValue")
        unit = labels.get("metric_unit") or alarm.get("valueUnit") or ""
        start_at = alarm.get("startAt")
        if start_at and default_ts is None:
            try:
                default_ts = float(datetime.fromisoformat(start_at.replace("Z", "+00:00")).timestamp())
            except Exception:
                default_ts = None

        line = (
            f"[Alert#{idx}] {alert_name} level {level} on instance {instance}"
            f"{' (app ' + app + ')' if app else ''} started at {start_at or 'unknown'}"
            f"{': ' + desc if desc else ''}"
        )
        if value is not None:
            line += f" | observed value: {value}{unit}"
        metric_key = labels.get("metricKey") or labels.get("metric_key")
        if metric_key:
            line += f" | metric: {metric_key}"
        lines.append(line)

    symptom = "Incoming infrastructure alarms:\n" + "\n".join(lines)
    symptom += "\nTelemetry snapshots (metrics/logs/shell) are available for expert agents to investigate."
    return symptom, default_ts

class OpsMemPipeline:
    """End-to-end OpsMem diagnosis runner."""

    def __init__(
        self,
        base_dir: Path,
        model_name: str | None = None,
        consolidation_enabled: bool | None = None,
        output_experiment: str | None = None,
        output_answer_subdir: str | None = None,
        case_pause_seconds: float | None = None,
    ):
        self.base_dir = base_dir
        self.config = _load_pipeline_config(base_dir, model_name=model_name)
        self.output_experiment = self._resolve_output_experiment(output_experiment)
        self.output_answer_subdir = self._resolve_output_answer_subdir(output_answer_subdir)
        self.case_pause_seconds = self._resolve_case_pause_seconds(case_pause_seconds)
        self.model_name = self.config["model_name"]
        self.agent_team = self.config["agent_team"]
        self.cfg_stm = self.config["stm"]
        self.cfg_agent = self.config["agent"]
        self.cfg_ltm = self.config["ltm"]
        self.cfg_cmr = self.config["cmr"]
        self.cfg_consolidation = self.config["consolidation"]
        self._override_consolidation_enabled(consolidation_enabled)
        self.long_term_memory = None
        self.runtime_store_dir = None
        self.consolidation_config: dict[str, Any] = {}
        self.memory_consolidator = None
        self.memory_resonator = None
        self.runtime_memory_needs_refresh = False

        if model_name:
            print(f"[RUN] Override model from CLI: {model_name}")
        if consolidation_enabled is not None:
            state = "enabled" if consolidation_enabled else "disabled"
            print(f"[RUN] Override memory consolidation from CLI: {state}")
        if self.output_experiment != "opsmem":
            print(f"[RUN] Output experiment: {self.output_experiment}")
        if self.output_answer_subdir:
            print(f"[RUN] Output answer subdir: {self.output_answer_subdir}")
        if self.case_pause_seconds > 0:
            print(f"[RUN] Pause between cases: {self.case_pause_seconds:.2f}s")

    def _override_consolidation_enabled(self, consolidation_enabled: bool | None) -> None:
        if consolidation_enabled is None:
            return
        self.cfg_consolidation["enabled"] = consolidation_enabled

    def _resolve_output_experiment(self, output_experiment: str | None) -> str:
        experiment = output_experiment or os.environ.get("OPSMEM_OUTPUT_EXPERIMENT") or "opsmem"
        experiment = experiment.strip().strip("/")
        if not experiment:
            raise ValueError("Output experiment cannot be empty.")
        experiment_path = Path(experiment)
        if experiment_path.is_absolute() or ".." in experiment_path.parts:
            raise ValueError(f"Invalid output experiment: {experiment}")
        return experiment

    def _resolve_output_answer_subdir(self, output_answer_subdir: str | None) -> str:
        subdir = output_answer_subdir or os.environ.get("OPSMEM_OUTPUT_ANSWER_SUBDIR") or ""
        subdir = subdir.strip().strip("/")
        if not subdir:
            return ""
        subdir_path = Path(subdir)
        if subdir_path.is_absolute() or ".." in subdir_path.parts:
            raise ValueError(f"Invalid output answer subdir: {subdir}")
        return subdir

    def _resolve_case_pause_seconds(self, case_pause_seconds: float | None) -> float:
        raw_value = case_pause_seconds
        if raw_value is None:
            raw_value = os.environ.get("OPSMEM_CASE_PAUSE_SECONDS", 0)
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid case pause seconds: {raw_value}") from exc

    def run(self, start_case: int | None = None, end_case: int | None = None) -> None:
        self._initialize_long_term_memory()
        symptoms, evidence_texts, groundtruths = load_microservice_cases()
        self.memory_resonator = CrossMemoryResonance(
            self.long_term_memory,
            llm_args=self.cfg_agent.get("llm_args"),
            cmr_config=self.cfg_cmr,
        )

        csv_save_path = self._answer_csv_path()
        self._ensure_output_file(csv_save_path)

        if start_case is None:
            start_case = read_start_case(filename=self._start_case_path())

        for idx, symptom in symptoms:
            if idx < start_case:
                continue
            if end_case is not None and idx > end_case:
                break

            self._refresh_runtime_memory_if_needed()
            answer, report, central_agent, expert_agents = self._run_case(
                case_id=idx,
                symptom=symptom,
                evidence_text=evidence_texts[idx],
            )
            self._consolidate_case(
                central_agent=central_agent,
                expert_agents=expert_agents,
                groundtruth=groundtruths[idx],
            )
            self._append_answer(
                csv_save_path=csv_save_path,
                answer=answer,
                report=report,
                groundtruth=groundtruths[idx],
            )
            write_start_case(idx + 1, filename=self._start_case_path())

        print_usage_summary()

    def _initialize_long_term_memory(self) -> None:
        if not self.cfg_ltm:
            return

        self.consolidation_config = dict(self.cfg_consolidation or {})
        configured_root_store_dir, configured_active_store_dir = resolve_store_paths(
            self.base_dir,
            self.cfg_ltm,
            self.consolidation_config,
        )
        self.consolidation_config = {
            **self.consolidation_config,
            "root_store_dir": str(configured_root_store_dir),
            "active_store_dir": str(configured_active_store_dir),
            "thresholds": self.cfg_cmr.get("thresholds") or {},
            "cmr": self.cfg_cmr,
        }
        self.runtime_store_dir = configured_active_store_dir

        if self.consolidation_config.get("enabled", False):
            run_metadata = prepare_consolidation_run_context(
                self.base_dir,
                self.consolidation_config,
                self.model_name,
            )
            self.consolidation_config = {
                **self.consolidation_config,
                "log_dir": _resolve_runtime_path(self.base_dir, run_metadata.log_dir),
                "root_store_dir": _resolve_runtime_path(self.base_dir, run_metadata.root_store_dir),
                "active_store_dir": _resolve_runtime_path(self.base_dir, run_metadata.active_store_dir),
                "resolved_active_store_dir": _resolve_runtime_path(self.base_dir, run_metadata.resolved_active_store_dir),
                "run_name": run_metadata.run_name,
                "model_name": run_metadata.model_name,
            }
            self.runtime_store_dir = Path(self.consolidation_config["resolved_active_store_dir"])
            print(f"[RUN] Memory consolidation logs: {self.consolidation_config['log_dir']}")
            print(f"[RUN] LTM target store: {self.runtime_store_dir}")

        if self.cfg_ltm.get("enabled", False):
            self.long_term_memory = build_long_term_memory(self.cfg_ltm, store_dir=self.runtime_store_dir)
            self.runtime_memory_needs_refresh = False
            print(f"[RUN] Long-term memory enabled: {self.runtime_store_dir}")

        if self.consolidation_config.get("enabled", False):
            self.memory_consolidator = MemoryConsolidator(
                config=self.consolidation_config,
                llm_args=self.cfg_agent.get("llm_args"),
                long_term_memory=self.long_term_memory,
            )

    def _refresh_runtime_memory_if_needed(self) -> None:
        if not (
            self.cfg_ltm.get("enabled", False)
            and self.consolidation_config.get("enabled", False)
            and self.consolidation_config.get("immediate_runtime_refresh", False)
            and self.runtime_memory_needs_refresh
        ):
            return

        self.long_term_memory = build_long_term_memory(self.cfg_ltm, store_dir=self.runtime_store_dir)
        self.runtime_memory_needs_refresh = False
        self.memory_resonator.set_long_term_memory(self.long_term_memory)
        if self.memory_consolidator is not None:
            self.memory_consolidator.set_long_term_memory(self.long_term_memory)

    def _run_case(self, case_id: int, symptom: str, evidence_text):
        print(f"\n\n===============[RUN] Case {case_id} =================")
        if self.case_pause_seconds > 0:
            time.sleep(self.case_pause_seconds)

        case_log_file = self._create_case_log_file(case_id)
        short_term_memory = ShortTermMemory.create(
            thresholds=self.cfg_stm.get("fsm", {}).get("thresholds", {})
        )
        expert_agents = self._create_expert_agents(
            case_id=case_id,
            short_term_memory=short_term_memory,
            case_log_file=case_log_file,
            evidence_text=evidence_text,
        )
        central_agent = self._create_central_agent(
            case_id=case_id,
            symptom=symptom,
            short_term_memory=short_term_memory,
            case_log_file=case_log_file,
            evidence_text=evidence_text,
            expert_agents=expert_agents,
        )

        central_agent.ingest()
        answer, report = central_agent.run()
        return answer, report, central_agent, expert_agents

    def _create_expert_agents(
        self,
        case_id: int,
        short_term_memory: ShortTermMemory,
        case_log_file: str,
        evidence_text,
    ) -> list[ExpertAgent]:
        expert_agents = []
        llm_args = self.cfg_agent.get("llm_args", {})
        for expert_name in self.agent_team.get("expert_names", []):
            expert_agents.append(
                ExpertAgent(
                    expert_name,
                    short_term_memory=short_term_memory,
                    case_id=case_id,
                    log_path=case_log_file,
                    llm_args=llm_args,
                    evidence_text=evidence_text,
                    max_retrieval_steps=llm_args["max_retrieval_steps"],
                    memory_resonator=self.memory_resonator,
                )
            )
        return expert_agents

    def _create_central_agent(
        self,
        case_id: int,
        symptom: str,
        short_term_memory: ShortTermMemory,
        case_log_file: str,
        evidence_text,
        expert_agents: list[ExpertAgent],
    ) -> CentralAgent:
        return CentralAgent(
            self.agent_team.get("head_name"),
            short_term_memory=short_term_memory,
            log_path=case_log_file,
            case_id=case_id,
            case_symptom=symptom,
            llm_args=self.cfg_agent.get("llm_args"),
            agent_team=self.agent_team,
            experts=expert_agents,
            evidence_text=evidence_text,
            memory_resonator=self.memory_resonator,
        )

    def _create_case_log_file(self, case_id: int) -> str:
        case_log_dir = self.base_dir / "logs" / "OpsMem" / f"case_{case_id}"
        case_log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        return str(case_log_dir / log_file_name)

    def _answer_csv_path(self) -> Path:
        output_model_name = _sanitize_filename(self.model_name)
        output_dir = self.base_dir / "output" / self.output_experiment / "answers"
        if self.output_answer_subdir:
            output_dir = output_dir / self.output_answer_subdir
        return output_dir / f"{output_model_name}.csv"

    def _append_answer(self, csv_save_path: Path, answer: str, report: str, groundtruth: str) -> None:
        current_row = {
            "prediction": answer.strip(),
            "report": report.strip(),
            "answer": groundtruth.strip(),
        }
        pd.DataFrame([current_row], columns=ANSWER_COLUMNS).to_csv(
            csv_save_path,
            **CSV_WRITE_OPTIONS,
            mode="a",
            header=False,
        )

    def _consolidate_case(self, central_agent: CentralAgent, expert_agents: list[ExpertAgent], groundtruth: str) -> None:
        if self.memory_consolidator is None:
            return

        expert_traces = [expert.export_trace() for expert in expert_agents]
        diagnosis_artifact = central_agent.export_diagnosis_artifact(
            groundtruth=groundtruth.strip(),
            expert_traces=expert_traces,
        )
        self.memory_consolidator.consolidate(diagnosis_artifact)
        self.runtime_memory_needs_refresh = True

    def _ensure_output_file(self, csv_save_path: Path) -> None:
        csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_save_path.exists():
            pd.DataFrame(columns=ANSWER_COLUMNS).to_csv(csv_save_path, **CSV_WRITE_OPTIONS)

    def _start_case_path(self) -> Path:
        return self.base_dir / "start_case.tmp"


def write_start_case(case_id: int, filename: str | Path = "start_case.tmp") -> None:
    Path(filename).write_text(str(case_id), encoding="utf-8")


def read_start_case(default: int = 0, filename: str | Path = "start_case.tmp") -> int:
    path = Path(filename)
    if not path.exists():
        return default
    content = path.read_text(encoding="utf-8").strip()
    return int(content) if content else default


def _resolve_runtime_path(base_dir: Path, raw_path: str | Path) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)















