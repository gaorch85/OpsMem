from __future__ import annotations

from consolidation.schema import ApprovalDecision, ApprovalResult, PendingApprovalBundle


def get_commit_approval(bundle: PendingApprovalBundle, provider: str = "terminal") -> ApprovalResult:
    if provider in {"meta", "auto"}:
        return ApprovalResult(
            incident_id=bundle.incident_id,
            reviewer="meta_agent",
            proposal_decisions=[
                ApprovalDecision(
                    proposal_id=proposal.proposal_id,
                    decision="approve",
                    comment="Approved automatically because MetaAgent review already accepted this proposal.",
                )
                for proposal in bundle.approved_proposals
            ],
        )

    if provider != "terminal":
        raise ValueError(f"Unsupported approval provider: {provider}")

    proposal_decisions: list[ApprovalDecision] = []
    for proposal in bundle.approved_proposals:
        while True:
            print(
                f"[Memory Approval] incident={bundle.incident_id} "
                f"proposal={proposal.proposal_id} action={proposal.action} memory_type={proposal.knowledge_type}"
            )
            decision = input("Decision (approve/reject): ").strip().lower()
            if decision in {"approve", "reject"}:
                break
            print("Please input 'approve' or 'reject'.")
        comment = input("Comment (optional): ").strip()
        proposal_decisions.append(
            ApprovalDecision(
                proposal_id=proposal.proposal_id,
                decision=decision,
                comment=comment,
            )
        )

    return ApprovalResult(
        incident_id=bundle.incident_id,
        reviewer="terminal_human",
        proposal_decisions=proposal_decisions,
    )




