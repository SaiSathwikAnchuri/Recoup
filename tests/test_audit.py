"""Phase 8 — decision audit trail + (optional, offline) LLM polish."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from agent.classifier import CauseClassifier
from agent.liquidity import LiquidityModel
from harness.engine import HarnessConfig

pytestmark = pytest.mark.skipif(
    not (CauseClassifier.default_exists() and LiquidityModel.default_exists()),
    reason="models not trained",
)

PRIORS = yaml.safe_load(Path("config/priors.yaml").read_text())
CFG = HarnessConfig(rev_cfg=PRIORS["revocation"])


@pytest.fixture(scope="module")
def bits():
    from agent.audit import build_record
    from agent.ev_policy import EVPolicy
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    truth = {t["case_id"]: t for t in (json.loads(x) for x in open("data/truth.jsonl"))}
    pol = EVPolicy()
    recs = [build_record(c, truth[c["case_id"]], pol, CFG, 42) for c in cases[:120]]
    return recs


# -- structured record ------------------------------------------------
def test_explain_is_structured_and_observable_only():
    from agent.ev_policy import EVPolicy
    cases = [json.loads(x) for x in open("data/cases.jsonl")]
    d = EVPolicy().explain(cases[0])
    assert set(d) >= {"cause_posterior", "funding_window", "candidates_top", "decision",
                      "ev_floor", "ltv_estimate"}
    assert abs(sum(d["cause_posterior"].values()) - 1.0) < 1e-6
    evs = [c["ev"] for c in d["candidates_top"]]
    assert evs == sorted(evs, reverse=True)          # ranked best-first
    assert "true_cause" not in d                     # explain() never sees the label


def test_record_joins_decision_and_outcome(bits):
    for r in bits:
        assert "decision" in r and "outcome" in r and r["narration"]
        o = r["outcome"]
        assert not (o["recovered"] and o["revoked"])
        assert o["attempts_used"] <= CFG.constraints.max_retries


# -- narration --------------------------------------------------------
def test_narration_is_deterministic_and_faithful(bits):
    from agent.audit import narrate
    for r in bits:
        assert narrate(r) == r["narration"]                       # pure function
        top = next(iter(r["cause_posterior"]))
        assert top.replace("_", " ") in r["narration"]
        if r["outcome"]["recovered"]:
            assert "recovered" in r["narration"]
        elif r["outcome"]["revoked"]:
            assert "revoked" in r["narration"]


def test_dead_mandate_records_show_reauth_or_escalation(bits):
    dead = [r for r in bits if r["true_cause"] == "mandate_dead"]
    assert dead
    for r in dead:
        acts = r["decision"]["actions"]
        assert (not acts) or acts[0]["kind"] == "reauth"


# -- persistence -----------------------------------------------------
def test_sqlite_roundtrip(bits, tmp_path):
    from agent.audit import write_sqlite
    p = tmp_path / "a.db"
    write_sqlite(bits, p)
    con = sqlite3.connect(p)
    n = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n == len(bits)
    row = con.execute("SELECT narration, record_json FROM decisions LIMIT 1").fetchone()
    assert row[0] and json.loads(row[1])["case_id"]


# -- LLM layer degrades gracefully with no network ----------------
def test_llm_polish_falls_back_to_template(bits, monkeypatch):
    import agent.llm_explain as m
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = bits[0]
    assert m.polish(r) == r["narration"]

    # even with a key, a network failure must fall back, not raise
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(m, "_call_api", lambda rec: (_ for _ in ()).throw(OSError("no net")))
    monkeypatch.setattr(m, "_load_cache", lambda: {})
    assert m.polish(r) == r["narration"]
