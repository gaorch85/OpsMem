from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from consolidation.commitment.id_utils import get_object_id_field, is_uuid, new_uuid
from consolidation.schema import ApprovalResult, CommitLog, PendingApprovalBundle, Proposal, to_jsonable
from consolidation.commitment.store import MemoryConsolidationLogStore
from consolidation.commitment.validator import validate_store_dir


class MemoryCommitter:
    def __init__(self, store_dir: str | Path, log_store: MemoryConsolidationLogStore):
        self.store_dir = Path(store_dir)
        self.log_store = log_store

    def commit(self, bundle: PendingApprovalBundle, approval_result: ApprovalResult) -> CommitLog | None:
        decision_map = {item.proposal_id: item for item in approval_result.proposal_decisions}
        approved = [
            proposal
            for proposal in bundle.approved_proposals
            if decision_map.get(proposal.proposal_id) and decision_map[proposal.proposal_id].decision == "approve"
        ]
        approved_ids = {proposal.proposal_id for proposal in approved}
        rejected = [proposal for proposal in bundle.approved_proposals if proposal.proposal_id not in approved_ids]

        self.log_store.save_json("final_approvals", f"{bundle.incident_id}.json", approval_result)
        if rejected:
            self.log_store.save_json(
                "approval_rejections",
                f"{bundle.incident_id}.json",
                {"incident_id": bundle.incident_id, "rejected_proposals": rejected},
            )

        if not approved:
            return None

        commit_id = new_uuid()
        affected_files = self._apply_commit(approved)
        commit_log = CommitLog(
            commit_id=commit_id,
            incident_id=bundle.incident_id,
            target_store_dir=str(self.store_dir),
            approved_proposal_ids=[proposal.proposal_id for proposal in approved],
            rejected_proposal_ids=[proposal.proposal_id for proposal in rejected],
            affected_files=affected_files,
            committed_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.log_store.save_json("commit_logs", f"{bundle.incident_id}_{commit_id}.json", commit_log)
        self.log_store.save_json(
            "committed_bundles",
            f"{bundle.incident_id}_{commit_id}.json",
            {
                "incident_id": bundle.incident_id,
                "commit_id": commit_id,
                "approved_proposals": approved,
                "review_notes": bundle.review_notes,
            },
        )
        return commit_log

    def _apply_commit(self, approved: list[Proposal]) -> list[str]:
        data = self._load_store_data()
        local_id_map: dict[str, str] = {}

        for proposal in approved:
            if proposal.action == "delete":
                self._apply_delete(data, proposal)

        pending_edge_additions: list[dict] = []
        for proposal in approved:
            if proposal.action != "create":
                continue
            if proposal.knowledge_type == "relation":
                relation_object = dict(proposal.object or {})
                pending_edge_additions.append(
                    {
                        "edge_type": relation_object["edge_type"],
                        "source_id": local_id_map.get(str(relation_object["source_id"]), str(relation_object["source_id"])),
                        "target_id": local_id_map.get(str(relation_object["target_id"]), str(relation_object["target_id"])),
                        "weight": relation_object["weight"],
                    }
                )
                continue

            normalized_object = self._normalize_object_payload(proposal)
            original_id = self._get_original_object_id(proposal)
            final_id = normalized_object[get_object_id_field(proposal.knowledge_type)]
            if original_id and original_id != final_id:
                local_id_map[original_id] = final_id
            self._apply_create(data, proposal.knowledge_type, normalized_object)

        self._apply_edge_additions(data, pending_edge_additions)
        return self._write_validated_store(data)

    def _load_store_data(self) -> dict:
        return {
            "patterns": self._read_jsonl(self.store_dir / "patterns.jsonl"),
            "procedures": self._read_jsonl(self.store_dir / "procedures.jsonl"),
            "cases": self._read_jsonl(self.store_dir / "cases.jsonl"),
            "pattern_procedure_edges": self._read_csv_rows(self.store_dir / "pattern_procedure_edges.csv"),
            "pattern_case_edges": self._read_csv_rows(self.store_dir / "pattern_case_edges.csv"),
        }

    def _apply_delete(self, data: dict, proposal: Proposal) -> None:
        target_id = proposal.target_id
        if not target_id:
            raise ValueError(f"Delete proposal missing target_id: {proposal.proposal_id}")

        if proposal.knowledge_type == "pattern":
            _require_existing(data["patterns"], "pattern_id", target_id)
            data["patterns"] = [row for row in data["patterns"] if str(row["pattern_id"]) != target_id]
            data["pattern_procedure_edges"] = [row for row in data["pattern_procedure_edges"] if str(row["pattern_id"]) != target_id]
            data["pattern_case_edges"] = [row for row in data["pattern_case_edges"] if str(row["pattern_id"]) != target_id]
            return

        if proposal.knowledge_type == "procedure":
            _require_existing(data["procedures"], "procedure_id", target_id)
            data["procedures"] = [row for row in data["procedures"] if str(row["procedure_id"]) != target_id]
            data["pattern_procedure_edges"] = [row for row in data["pattern_procedure_edges"] if str(row["procedure_id"]) != target_id]
            return

        if proposal.knowledge_type == "case":
            _require_existing(data["cases"], "case_id", target_id)
            data["cases"] = [row for row in data["cases"] if str(row["case_id"]) != target_id]
            data["pattern_case_edges"] = [row for row in data["pattern_case_edges"] if str(row["case_id"]) != target_id]
            return

        raise ValueError(f"Unsupported knowledge_type for delete: {proposal.knowledge_type}")

    def _normalize_object_payload(self, proposal: Proposal) -> dict:
        payload = dict(proposal.object or {})
        if proposal.knowledge_type == "relation":
            return payload
        id_field = get_object_id_field(proposal.knowledge_type)
        original_id = str(payload.get(id_field) or "").strip()
        if not is_uuid(original_id):
            payload[id_field] = new_uuid()
        allowed_fields = {
            "pattern": ("pattern_id", "signals", "root_cause", "content"),
            "procedure": ("procedure_id", "symptoms", "content"),
            "case": ("case_id", "symptoms", "root_cause", "content"),
        }[proposal.knowledge_type]
        return {key: payload[key] for key in allowed_fields if key in payload}

    def _get_original_object_id(self, proposal: Proposal) -> str | None:
        if not proposal.object or proposal.knowledge_type == "relation":
            return None
        id_field = get_object_id_field(proposal.knowledge_type)
        value = str(proposal.object.get(id_field) or "").strip()
        return value or None

    def _apply_create(self, data: dict, knowledge_type: str, object_payload: dict) -> None:
        bucket_map = {
            "pattern": ("patterns", "pattern_id"),
            "procedure": ("procedures", "procedure_id"),
            "case": ("cases", "case_id"),
        }
        if knowledge_type not in bucket_map:
            raise ValueError(f"Unsupported knowledge_type for create: {knowledge_type}")
        bucket, id_field = bucket_map[knowledge_type]
        object_id = str(object_payload[id_field])
        if any(str(row[id_field]) == object_id for row in data[bucket]):
            raise ValueError(f"Create proposal conflicts with existing {knowledge_type} id: {object_id}")
        data[bucket].append(object_payload)

    def _apply_edge_additions(self, data: dict, edge_rows: list[dict]) -> None:
        for edge in edge_rows:
            if edge["edge_type"] == "pattern_case":
                row = {"pattern_id": edge["source_id"], "case_id": edge["target_id"], "weight": edge["weight"]}
                if not any(
                    str(item["pattern_id"]) == str(row["pattern_id"]) and str(item["case_id"]) == str(row["case_id"])
                    for item in data["pattern_case_edges"]
                ):
                    data["pattern_case_edges"].append(row)
                continue

            if edge["edge_type"] == "pattern_procedure":
                row = {"pattern_id": edge["source_id"], "procedure_id": edge["target_id"], "weight": edge["weight"]}
                if not any(
                    str(item["pattern_id"]) == str(row["pattern_id"]) and str(item["procedure_id"]) == str(row["procedure_id"])
                    for item in data["pattern_procedure_edges"]
                ):
                    data["pattern_procedure_edges"].append(row)
                continue

            raise ValueError(f"Unsupported edge_type during commit: {edge['edge_type']}")

    def _write_validated_store(self, data: dict) -> list[str]:
        temp_parent = self.log_store.log_dir / "_tmp_commits"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_store_dir = temp_parent / "stage"
        if temp_store_dir.exists():
            shutil.rmtree(temp_store_dir, ignore_errors=True)
        temp_store_dir.mkdir(parents=True, exist_ok=True)

        affected_files = [
            "patterns.jsonl",
            "procedures.jsonl",
            "cases.jsonl",
            "pattern_procedure_edges.csv",
            "pattern_case_edges.csv",
        ]
        try:
            self._write_jsonl(temp_store_dir / "patterns.jsonl", data["patterns"])
            self._write_jsonl(temp_store_dir / "procedures.jsonl", data["procedures"])
            self._write_jsonl(temp_store_dir / "cases.jsonl", data["cases"])
            self._write_csv_rows(temp_store_dir / "pattern_procedure_edges.csv", ["pattern_id", "procedure_id", "weight"], data["pattern_procedure_edges"])
            self._write_csv_rows(temp_store_dir / "pattern_case_edges.csv", ["pattern_id", "case_id", "weight"], data["pattern_case_edges"])
            validate_store_dir(temp_store_dir)
            for name in affected_files:
                shutil.copy2(temp_store_dir / name, self.store_dir / name)
            return affected_files
        finally:
            shutil.rmtree(temp_store_dir, ignore_errors=True)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _require_existing(rows: list[dict], id_field: str, target_id: str) -> None:
    if not any(str(row[id_field]) == target_id for row in rows):
        raise ValueError(f"Delete proposal target {id_field} does not exist: {target_id}")





