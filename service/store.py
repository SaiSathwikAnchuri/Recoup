"""SQLite state for the running service — cases, decisions, actions, escalations,
plus the Recoup 2.0 tables: an append-only `events` log, folded `customers` /
`mandates`, closed-loop `outcomes`, and `idempotency_keys`.

One file, no ORM. `Store(path)` opens it and creates the schema. Everything is
JSON-in-a-column where the shape is rich (the case record, the decision record,
a folded state snapshot) and plain columns where we filter or sort.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .events import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    source      TEXT,                -- 'webhook' | 'demo' | 'manual'
    created_at  REAL,
    amount      REAL,
    failure     TEXT,
    status      TEXT,                -- 'open' | 'recovered' | 'revoked' | 'escalated' | 'closed'
    record_json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    case_id     TEXT,
    created_at  REAL,
    terminal    TEXT,
    top_cause   TEXT,
    top_cause_p REAL,
    note        TEXT,
    record_json TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT,
    kind        TEXT,                -- 'retry' | 'reauth' | 'sms' | 'nudge' | 'escalate'
    due_at      REAL,
    executed_at REAL,
    status      TEXT,                -- 'pending' | 'done' | 'skipped'
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS escalations (
    case_id     TEXT PRIMARY KEY,
    created_at  REAL,
    reason      TEXT,
    resolved    INTEGER DEFAULT 0
);

-- Recoup 2.0 -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    dedup_key   TEXT UNIQUE,          -- second event with the same key is ignored
    type        TEXT,
    customer_id TEXT,
    mandate_id  TEXT,
    case_id     TEXT,
    at          REAL,
    payload     TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_case ON events(case_id, at);
CREATE INDEX IF NOT EXISTS ix_events_cust ON events(customer_id, at);

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    mandate_id  TEXT,
    updated_at  REAL,
    stage       TEXT,
    churn_risk  REAL,
    state_json  TEXT
);
CREATE TABLE IF NOT EXISTS mandates (
    mandate_id  TEXT PRIMARY KEY,
    customer_id TEXT,
    category    TEXT,
    amount      REAL,
    status      TEXT,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id       TEXT,
    action        TEXT,
    action_at     REAL,
    result        TEXT,
    recovered_amount REAL,
    recovery_delay   REAL,
    reward        REAL,
    state_before  TEXT,
    state_after   TEXT,
    created_at    REAL
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key        TEXT PRIMARY KEY,
    kind       TEXT,
    ref        TEXT,
    created_at REAL
);
"""


class Store:
    def __init__(self, path: str | Path = "service/recoup.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._c = sqlite3.connect(self.path, check_same_thread=False)
        self._c.row_factory = sqlite3.Row
        self._c.executescript(_SCHEMA)
        self._c.commit()

    # -- cases ---------------------------------------------------------------
    def upsert_case(self, case: dict, source: str, status: str = "open") -> None:
        self._c.execute(
            "INSERT INTO cases (case_id, source, created_at, amount, failure, status, record_json) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(case_id) DO UPDATE SET status=excluded.status",
            (case["case_id"], source, time.time(), float(case["mandate"]["amount"]),
             case["failure"].get("token"), status, json.dumps(case)),
        )
        self._c.commit()

    def set_case_status(self, case_id: str, status: str) -> None:
        self._c.execute("UPDATE cases SET status=? WHERE case_id=?", (status, case_id))
        self._c.commit()

    def get_case(self, case_id: str) -> dict | None:
        r = self._c.execute("SELECT record_json FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return json.loads(r["record_json"]) if r else None

    def case_status(self, case_id: str) -> str | None:
        r = self._c.execute("SELECT status FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return r["status"] if r else None

    def list_cases(self, limit: int = 100) -> list[dict]:
        # join only the *latest* decision per case, so a re-plan doesn't duplicate the row
        rows = self._c.execute(
            "SELECT c.case_id, c.source, c.created_at, c.amount, c.failure, c.status, "
            "d.top_cause, d.terminal, d.note "
            "FROM cases c LEFT JOIN decisions d ON d.case_id = c.case_id AND d.created_at = ("
            "  SELECT MAX(created_at) FROM decisions d2 WHERE d2.case_id = c.case_id) "
            "ORDER BY c.created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- decisions ---------------------------------------------------------
    def add_decision(self, case_id: str, rec: dict) -> None:
        tc, tcp = next(iter(rec["cause_posterior"].items()))
        self._c.execute(
            "INSERT INTO decisions (case_id, created_at, terminal, top_cause, top_cause_p, note, record_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (case_id, time.time(), rec["decision"]["terminal"], tc, tcp,
             rec["decision"]["note"], json.dumps(rec)),
        )
        self._c.commit()

    def latest_decision(self, case_id: str) -> dict | None:
        r = self._c.execute(
            "SELECT record_json FROM decisions WHERE case_id=? ORDER BY created_at DESC LIMIT 1",
            (case_id,)).fetchone()
        return json.loads(r["record_json"]) if r else None

    # -- actions ---------------------------------------------------------
    def schedule_actions(self, case_id: str, actions: list[dict]) -> None:
        for a in actions:
            self._c.execute(
                "INSERT INTO actions (case_id, kind, due_at, status) VALUES (?,?,?,?)",
                (case_id, a["kind"], a["due_at"], "pending"))
        self._c.commit()

    def due_actions(self, now: float | None = None) -> list[dict]:
        now = now if now is not None else time.time()
        rows = self._c.execute(
            "SELECT * FROM actions WHERE status='pending' AND due_at<=? ORDER BY due_at", (now,)).fetchall()
        return [dict(r) for r in rows]

    def claim_action(self, action_id: int) -> bool:
        """Atomically move one action from 'pending' to 'running' before executing it.
        Returns False if another caller already claimed it — `due_actions` SELECTs
        rows without locking them, so two concurrent /tick calls (uvicorn runs sync
        routes in a thread pool, and the sqlite3 connection is opened with
        check_same_thread=False for exactly that reason) could otherwise both see
        the same row as pending and run its side effect (a real Payment Link, an
        SMS) twice. This turns that race into a single UPDATE...WHERE that only one
        caller can win."""
        cur = self._c.execute(
            "UPDATE actions SET status='running' WHERE id=? AND status='pending'", (action_id,))
        self._c.commit()
        return cur.rowcount == 1

    def mark_action(self, action_id: int, status: str, result: dict) -> None:
        self._c.execute(
            "UPDATE actions SET status=?, executed_at=?, result_json=? WHERE id=?",
            (status, time.time(), json.dumps(result), action_id))
        self._c.commit()

    def actions_for(self, case_id: str) -> list[dict]:
        rows = self._c.execute(
            "SELECT id, kind, due_at, executed_at, status, result_json FROM actions "
            "WHERE case_id=? ORDER BY due_at", (case_id,)).fetchall()
        return [dict(r) for r in rows]

    # -- escalations ----------------------------------------------------
    def escalate(self, case_id: str, reason: str) -> None:
        self._c.execute(
            "INSERT OR REPLACE INTO escalations (case_id, created_at, reason) VALUES (?,?,?)",
            (case_id, time.time(), reason))
        self.set_case_status(case_id, "escalated")

    def open_escalations(self) -> list[dict]:
        rows = self._c.execute(
            "SELECT case_id, created_at, reason FROM escalations WHERE resolved=0 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- stats ---------------------------------------------------------
    def stats(self) -> dict:
        row = self._c.execute(
            "SELECT COUNT(*) n, "
            "SUM(status='recovered') recovered, SUM(status='revoked') revoked, "
            "SUM(status='escalated') escalated, SUM(status='open') open_, "
            "SUM(CASE WHEN status='recovered' THEN amount ELSE 0 END) recovered_rs "
            "FROM cases").fetchone()
        return {k: (v or 0) for k, v in dict(row).items()}

    # -- idempotency (Recoup 2.0) ----------------------------------------
    def seen_key(self, key: str) -> str | None:
        """Return the ref this idempotency key was first bound to, or None."""
        r = self._c.execute("SELECT ref FROM idempotency_keys WHERE key=?", (key,)).fetchone()
        return r["ref"] if r else None

    def remember_key(self, key: str, kind: str, ref: str) -> bool:
        """Bind `key` -> `ref`. False if it was already bound (caller should treat
        the incoming request as a duplicate)."""
        try:
            self._c.execute(
                "INSERT INTO idempotency_keys (key, kind, ref, created_at) VALUES (?,?,?,?)",
                (key, kind, ref, time.time()))
            self._c.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # -- events (Recoup 2.0) -------------------------------------------
    def append_event(self, e: Event) -> bool:
        """Append one event. False (and no write) if an event with the same
        `dedup_key` already exists — the log stays idempotent."""
        r = e.to_row()
        try:
            self._c.execute(
                "INSERT INTO events (event_id, dedup_key, type, customer_id, mandate_id, "
                "case_id, at, payload) VALUES (?,?,?,?,?,?,?,?)",
                (r["event_id"], r["dedup_key"], r["type"], r["customer_id"],
                 r["mandate_id"], r["case_id"], r["at"], json.dumps(r["payload"])))
            self._c.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def _events(self, where: str, args: tuple) -> list[Event]:
        rows = self._c.execute(
            f"SELECT event_id, dedup_key, type, customer_id, mandate_id, case_id, at, payload "
            f"FROM events WHERE {where} ORDER BY at, event_id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
            out.append(Event.from_row(d))
        return out

    def events_for_case(self, case_id: str) -> list[Event]:
        return self._events("case_id=?", (case_id,))

    def events_for_customer(self, customer_id: str) -> list[Event]:
        return self._events("customer_id=?", (customer_id,))

    # -- folded customer / mandate state ------------------------------
    def save_state(self, state) -> None:
        d = state.to_dict()
        self._c.execute(
            "INSERT INTO customers (customer_id, mandate_id, updated_at, stage, churn_risk, state_json) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(customer_id) DO UPDATE SET "
            "mandate_id=excluded.mandate_id, updated_at=excluded.updated_at, "
            "stage=excluded.stage, churn_risk=excluded.churn_risk, state_json=excluded.state_json",
            (state.customer_id, state.mandate_id, time.time(), state.recovery_stage,
             state.churn_risk, json.dumps(d)))
        self._c.execute(
            "INSERT INTO mandates (mandate_id, customer_id, category, amount, status, updated_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(mandate_id) DO UPDATE SET "
            "status=excluded.status, amount=excluded.amount, updated_at=excluded.updated_at",
            (state.mandate_id, state.customer_id, state.category, state.amount,
             state.recovery_stage, time.time()))
        self._c.commit()

    def get_state(self, customer_id: str) -> dict | None:
        r = self._c.execute("SELECT state_json FROM customers WHERE customer_id=?",
                            (customer_id,)).fetchone()
        return json.loads(r["state_json"]) if r else None

    # -- closed-loop outcomes ----------------------------------------
    def record_outcome(self, case_id: str, *, action: str, action_at: float, result: str,
                       recovered_amount: float = 0.0, recovery_delay: float | None = None,
                       reward: float = 0.0, state_before: dict | None = None,
                       state_after: dict | None = None) -> None:
        self._c.execute(
            "INSERT INTO outcomes (case_id, action, action_at, result, recovered_amount, "
            "recovery_delay, reward, state_before, state_after, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (case_id, action, action_at, result, recovered_amount, recovery_delay, reward,
             json.dumps(state_before or {}), json.dumps(state_after or {}), time.time()))
        self._c.commit()

    def outcomes_for(self, case_id: str) -> list[dict]:
        rows = self._c.execute(
            "SELECT action, action_at, result, recovered_amount, recovery_delay, reward "
            "FROM outcomes WHERE case_id=? ORDER BY created_at", (case_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_outcomes(self, limit: int = 5000) -> list[dict]:
        rows = self._c.execute(
            "SELECT case_id, action, result, recovered_amount, recovery_delay, reward, state_before "
            "FROM outcomes ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def pending_actions_for(self, case_id: str) -> list[dict]:
        rows = self._c.execute(
            "SELECT id, kind, due_at FROM actions WHERE case_id=? AND status='pending' ORDER BY due_at",
            (case_id,)).fetchall()
        return [dict(r) for r in rows]

    def cancel_pending_actions(self, case_id: str) -> int:
        cur = self._c.execute(
            "UPDATE actions SET status='cancelled' WHERE case_id=? AND status='pending'", (case_id,))
        self._c.commit()
        return cur.rowcount

    def reset_case(self, case_id: str) -> None:
        """Wipe every trace of one case_id — events, folded state, decisions,
        actions, outcomes, escalations, idempotency bindings, the case row itself.

        Final-audit fix: `/demo/random?seed=N` derives a deterministic case_id from
        the seed, but a demo case is a live record like any other — a second call
        with the same seed used to keep appending to the first call's event log
        (a second PAYMENT_FAILED, a reauth already marked "tried" from last time,
        ...), so the *displayed decision* silently changed run to run for the exact
        same seed. Calling this before replaying a demo scenario makes "same seed"
        actually mean "same outcome, every time" — required for a reproducible demo
        script, not just nice-to-have. Never called on the webhook-driven path.
        """
        for table in ("events", "outcomes", "actions", "decisions", "escalations", "cases"):
            self._c.execute(f"DELETE FROM {table} WHERE case_id=?", (case_id,))
        self._c.execute("DELETE FROM customers WHERE customer_id=?", (case_id,))
        self._c.execute("DELETE FROM mandates WHERE mandate_id=?", (case_id,))
        self._c.execute("DELETE FROM idempotency_keys WHERE ref=?", (case_id,))
        self._c.commit()

    def close(self) -> None:
        self._c.close()
