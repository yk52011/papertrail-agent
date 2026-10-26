"""Local job catalogue. Graph checkpoints are stored separately by LangGraph."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class JobStore:
    def __init__(self, root: Path = Path(".papertrail/jobs")):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, title TEXT, "
                       "status TEXT, created TEXT, updated TEXT, summary TEXT)")

    def connect(self):
        db = sqlite3.connect(self.root / "catalog.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def directory(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid job ID")
        return self.root / job_id

    def create(self, title: str) -> str:
        job_id = uuid4().hex
        self.directory(job_id).mkdir()
        now = datetime.now(UTC).isoformat()
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?)",
                       (job_id, title, "created", now, now, "{}"))
        return job_id

    def update(self, job_id: str, status: str, summary: dict):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?, updated=?, summary=? WHERE id=?",
                       (status, datetime.now(UTC).isoformat(),
                        json.dumps(summary, ensure_ascii=False), job_id))

    def list(self, limit: int = 30) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY updated DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "summary": json.loads(row["summary"])} for row in rows]

    def load(self, job_id: str) -> dict:
        directory = self.directory(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown job")
        state_path = directory / "state.json"
        return {**dict(row), "summary": json.loads(row["summary"]),
                "state": json.loads(state_path.read_text()) if state_path.exists() else {}}
