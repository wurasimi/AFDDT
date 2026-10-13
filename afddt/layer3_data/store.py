"""Layer 3.1: Raw data store.

SQLite stands in for a time-series database: it needs no installation and the
file (data/afddt.db) can be opened with any SQLite viewer.  Four tables hold
the raw telemetry, the findings (with their enrichment fields), the mitigation
actions and the audit log of every agent decision (NFR8).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER, tier TEXT, twin_id TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_tel_twin ON telemetry(twin_id, tick);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY, tick INTEGER, ts REAL, tier TEXT, source_agent TEXT, twin_ids TEXT,
    confidence REAL, severity TEXT, summary TEXT, explanation TEXT, fused_from TEXT,
    enrichment_status TEXT, enrichment_result TEXT, decision TEXT, latency_ms REAL);
CREATE TABLE IF NOT EXISTS actions (
    action_id TEXT PRIMARY KEY, finding_id TEXT, ts REAL, type TEXT, target TEXT, risk TEXT,
    simulation TEXT, status TEXT, approved_by TEXT, outcome TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, tick INTEGER, agent TEXT, event TEXT, detail TEXT);
"""


class DataStore:
    def __init__(self, path: Path, fresh: bool = True):
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
            if fresh and path.exists():
                path.unlink()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.rows_written = 0

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self.lock:
            self.conn.execute(sql, args)
            self.rows_written += 1

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()

    def write_telemetry(self, tick: int, tier: str, twin_id: str, payload: dict) -> None:
        self._exec("INSERT INTO telemetry(tick,tier,twin_id,payload) VALUES (?,?,?,?)",
                   (tick, tier, twin_id, json.dumps(payload, default=str)))

    def prune(self, keep_ticks: int, current_tick: int) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM telemetry WHERE tick < ?", (current_tick - keep_ticks,))

    def upsert_finding(self, f) -> None:
        self._exec(
            """INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f.finding_id, f.tick, f.ts, f.tier, f.source_agent, json.dumps(f.twin_ids), f.confidence,
             f.severity, f.summary, json.dumps(f.explanation, default=str), json.dumps(f.fused_from),
             f.enrichment_status, json.dumps(f.enrichment, default=str), f.decision, f.latency_ms))

    def upsert_action(self, a) -> None:
        self._exec("INSERT OR REPLACE INTO actions VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (a.action_id, a.finding_id, a.ts, a.type, a.target, a.risk,
                    json.dumps(a.simulation, default=str), a.status, a.approved_by, a.outcome))

    def audit(self, tick: int, agent: str, event: str, detail: dict | str = "") -> None:
        self._exec("INSERT INTO audit(ts,tick,agent,event,detail) VALUES (?,?,?,?,?)",
                   (time.time(), tick, agent, event, json.dumps(detail, default=str)))

    def count(self, table: str) -> int:
        with self.lock:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def recent_audit(self, n: int = 30) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT ts,tick,agent,event,detail FROM audit ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"ts": r[0], "tick": r[1], "agent": r[2], "event": r[3], "detail": json.loads(r[4])} for r in rows]
