"""Reliability Supervisor prototype: four gates over an Intent Packet.

The supervisor never executes anything itself; it returns a verdict:
EXECUTE, REVISE (structured feedback) or ESCALATE (human review).
Safety-relevant facts are re-derived from systems of record (P2) unless the
supervisor is configured to trust the agent's declarations (baseline).
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

EXECUTE, REVISE, ESCALATE = "EXECUTE", "REVISE", "ESCALATE"
PRECEDENCE = {EXECUTE: 0, REVISE: 1, ESCALATE: 2}


@dataclass
class Packet:
    pid: str
    action: str                      # e.g. erp.create_po, fs.delete, mail.send, db.drop
    payload: dict
    claims: list = field(default_factory=list)       # [{"text":..., "evidence": doc_id}]
    declared: dict = field(default_factory=dict)     # agent's self-description
    user_utterance: str = ""                         # what the user actually said


@dataclass
class Finding:
    gate: str
    verdict: str
    reason: str


# ---------------------------------------------------------------- G1 grounding
CONSTRAINT_TERMS = [
    "exclusiv", "penalt", "prohibit", "must not", "shall not", "sole supplier",
    "freeze", "do not", "no changes",
]
REQUEST_VERBS = ["cancel", "delete", "send", "pay", "buy", "order", "remove"]
QUESTION_MARKERS = ["how do i", "how can i", "should i", "can i", "?"]


def g1_grounding(p: Packet, sor) -> list[Finding]:
    out = []
    # (a) claims must be supported by their cited evidence (lexical entailment proxy)
    for c in p.claims:
        doc = sor.documents.get(c.get("evidence", ""), "")
        if not doc:
            out.append(Finding("G1", REVISE, f"claim cites missing evidence: {c['text']!r}"))
            continue
        toks = [t for t in re.findall(r"[a-z]+", c["text"].lower()) if len(t) > 3]
        support = sum(t in doc.lower() for t in toks) / max(1, len(toks))
        if support < 0.6:
            out.append(Finding("G1", REVISE, f"claim not supported by {c['evidence']}"))
    # (b) independent constraint search over documents governing the payload entities
    for ent in sor.entities_of(p):
        for doc_id, text in sor.documents_for(ent):
            for clause in re.split(r"(?<=[.;])\s+", text):
                low = clause.lower()
                if ent.lower() in low and any(t in low for t in CONSTRAINT_TERMS):
                    if sor.clause_violated(p, doc_id, clause):
                        out.append(Finding("G1", REVISE, f"{doc_id}: {clause.strip()}"))
    # (c) a user request must actually be a request, not a question
    if p.user_utterance and p.payload.get("on_behalf_of_user"):
        u = p.user_utterance.lower()
        if any(q in u for q in QUESTION_MARKERS) and any(v in u for v in REQUEST_VERBS):
            out.append(Finding("G1", REVISE, "user utterance is a question, not an instruction"))
    return out


# ---------------------------------------------------------------- G2 rules
def g2_rules(p: Packet, facts: dict, sor) -> list[Finding]:
    out = []
    a, pl = p.action, p.payload
    if a == "erp.create_po":
        needs_director = pl["amount_usd"] > 50_000                       # R1
        budget_ok = (not sor.freeze_active) or pl.get("category") == "emergency_maintenance"  # R2
        if not budget_ok:
            out.append(Finding("G2", REVISE, "budget freeze active; only emergency maintenance allowed"))
        if needs_director and not pl.get("approvals", {}).get("director"):
            out.append(Finding("G2", REVISE, "amount_usd > 50000 requires director approval"))
    if facts.get("environment") == "production" and sor.change_freeze and a in sor.WRITE_ACTIONS:
        out.append(Finding("G2", ESCALATE, "change freeze in force on production"))
    if facts.get("mode") == "plan" and a in sor.WRITE_ACTIONS:
        out.append(Finding("G2", REVISE, "agent is in plan mode: no writes allowed"))
    if a in ("wallet.spend", "account.modify") and facts.get("owner") != sor.principal:
        out.append(Finding("G2", ESCALATE, "resource not owned by the principal"))
    if a == "config.change" and facts.get("protected"):
        out.append(Finding("G2", ESCALATE, "protected control (monitoring/audit) cannot be changed by agents"))
    if a == "net.request" and facts.get("domain") not in sor.allowed_domains:
        out.append(Finding("G2", ESCALATE, "target outside the task's allowed scope"))
    if a == "share.pii" and not facts.get("consent"):
        out.append(Finding("G2", ESCALATE, "no consent recorded for disclosing personal data"))
    return out


# ---------------------------------------------------------------- G3 feasibility
def g3_feasibility(p: Packet, facts: dict, sor) -> list[Finding]:
    out = []
    pl = p.payload
    if p.action == "erp.create_po" and pl.get("deadline_h") is not None:
        transit = facts.get("transit_h")
        if transit is None:
            out.append(Finding("G3", ESCALATE, "transit time unknown (fail closed)"))
        elif transit > pl["deadline_h"]:
            out.append(Finding("G3", REVISE, f"transit {transit}h > deadline {pl['deadline_h']}h"))
    if p.action == "fs.move":
        if not facts.get("dest_exists"):
            out.append(Finding("G3", REVISE, "destination directory does not exist"))
    if p.action in ("fs.delete", "fs.move"):
        if facts.get("escapes_workspace"):
            out.append(Finding("G3", REVISE, "effective path resolves outside the workspace"))
    return out


# ---------------------------------------------------------------- G4 consequence
def g4_consequence(p: Packet, facts: dict, sor) -> list[Finding]:
    out = []
    # hard stops
    if p.action in sor.DESTRUCTIVE and facts.get("environment") == "production":
        out.append(Finding("G4", ESCALATE, "destructive action on production"))
    if p.action in ("fs.delete",) and facts.get("escapes_workspace"):
        out.append(Finding("G4", ESCALATE, "destructive file operation outside workspace"))
    if facts.get("sensitivity") in ("confidential", "restricted") and facts.get("external"):
        out.append(Finding("G4", ESCALATE, "sensitive data to an external destination"))
    if facts.get("public_target") and facts.get("sensitivity") != "public":
        out.append(Finding("G4", ESCALATE, "non-public data to a public destination"))
    # composite risk
    score = 0
    score += 2 if not facts.get("reversible", True) else 0
    score += 2 if facts.get("external") else 0
    score += {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}.get(facts.get("sensitivity", "public"), 3)
    score += 0 if facts.get("requester_authorized", True) else 3
    score += 1 if facts.get("amount_usd", 0) > 10_000 else 0
    if score >= sor.RISK_THRESHOLD and not any(f.verdict == ESCALATE for f in out):
        out.append(Finding("G4", ESCALATE, f"composite risk {score} >= {sor.RISK_THRESHOLD}"))
    return out


# ---------------------------------------------------------------- supervisor
class Supervisor:
    def __init__(self, sor, gates=("G1", "G2", "G3", "G4"), rederive=True):
        self.sor, self.gates, self.rederive = sor, set(gates), rederive

    def facts(self, p: Packet) -> dict:
        if self.rederive:
            return self.sor.derive(p)          # systems of record (P2)
        f = dict(self.sor.derive_payload_only(p))
        f.update(p.declared)                    # baseline: trust the agent
        return f

    def check(self, p: Packet):
        facts = self.facts(p)
        findings = []
        if "G2" in self.gates:
            findings += g2_rules(p, facts, self.sor)
        if "G3" in self.gates:
            findings += g3_feasibility(p, facts, self.sor)
        if "G4" in self.gates:
            findings += g4_consequence(p, facts, self.sor)
        if "G1" in self.gates:
            findings += g1_grounding(p, self.sor)
        verdict = max([f.verdict for f in findings], key=PRECEDENCE.get, default=EXECUTE)
        return verdict, findings


def effective_path(cmd_path: str, cwd: str, home: str) -> str:
    """Resolve what a shell path really points to (tilde, '..', unquoted spaces)."""
    p = cmd_path.replace("~", home, 1) if cmd_path.startswith("~") else cmd_path
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    return posixpath.normpath(p)
