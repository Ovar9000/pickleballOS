"""Append-only SQLite event log (the subset of the DESIGN.md schema used so far).

Phase 3 writes sessions, ~10 Hz player positions and camera events. Later phases add rallies,
shots and claims to the same file.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, court_id TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT,
  fps REAL, mode TEXT CHECK (mode IN ('offline','live')), config TEXT
);
CREATE TABLE IF NOT EXISTS positions (
  session_id TEXT NOT NULL, ts REAL NOT NULL, slot TEXT NOT NULL,
  court_x REAL, court_y REAL, conf REAL, estimated INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (session_id, ts, slot)
);
CREATE TABLE IF NOT EXISTS rallies (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, game_id INTEGER, rally_no INTEGER,
  start_ts REAL, end_ts REAL, hits INTEGER, bounces INTEGER, server_slot TEXT,
  winner_team TEXT CHECK (winner_team IN ('A','B')),
  outcome_source TEXT CHECK (outcome_source IN ('scorer','auto')), end_reason TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, rally_id INTEGER,
  ts REAL NOT NULL, frame INTEGER, type TEXT NOT NULL, slot TEXT,
  court_x REAL, court_y REAL, img_u REAL, img_v REAL,
  conf REAL, estimated INTEGER NOT NULL DEFAULT 1, source TEXT,
  supersedes INTEGER REFERENCES events(id), extra TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def start_session(self, court_id: str, fps: float, mode: str = "offline", config: dict | None = None) -> str:
        sid = str(uuid.uuid4())
        self.conn.execute("INSERT INTO sessions (id, court_id, started_at, fps, mode, config) VALUES (?,?,?,?,?,?)",
                          (sid, court_id, _now(), fps, mode, json.dumps(config or {})))
        self.conn.commit()
        return sid

    def end_session(self, session_id: str) -> None:
        self.conn.execute("UPDATE sessions SET ended_at=? WHERE id=?", (_now(), session_id))
        self.conn.commit()

    def add_position(self, session_id: str, ts: float, slot: str, x: float, y: float, conf: float = 1.0) -> None:
        self.conn.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,1)",
                          (session_id, round(ts, 3), slot, x, y, conf))

    def add_event(self, session_id: str, ts: float, type: str, frame: int | None = None, slot: str | None = None,
                  court_xy=None, img_uv=None, source: str = "vision", conf: float | None = None,
                  extra: dict | None = None) -> int:
        x, y = (float(court_xy[0]), float(court_xy[1])) if court_xy is not None else (None, None)
        u, v = (float(img_uv[0]), float(img_uv[1])) if img_uv is not None else (None, None)
        cur = self.conn.execute(
            "INSERT INTO events (session_id, ts, frame, type, slot, court_x, court_y, img_u, img_v, conf, source, extra)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, ts, frame, type, slot, x, y, u, v, conf, source, json.dumps(extra) if extra else None))
        return int(cur.lastrowid)

    def add_rally(self, session_id: str, start: float, end: float, hits: int, bounces: int) -> int:
        n = self.conn.execute("SELECT COUNT(*) FROM rallies WHERE session_id=?", (session_id,)).fetchone()[0]
        cur = self.conn.execute(
            "INSERT INTO rallies (session_id, rally_no, start_ts, end_ts, hits, bounces) VALUES (?,?,?,?,?,?)",
            (session_id, n + 1, start, end, hits, bounces))
        return int(cur.lastrowid)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
