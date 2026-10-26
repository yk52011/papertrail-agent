"""Local run history; stores outputs, never API keys."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def save_run(payload: dict, directory: Path = Path(".papertrail")) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    with sqlite3.connect(directory / "history.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs "
                   "(id TEXT PRIMARY KEY, created TEXT, payload TEXT)")
        db.execute("INSERT INTO runs VALUES (?, ?, ?)",
                   (run_id, datetime.now(UTC).isoformat(),
                    json.dumps(payload, ensure_ascii=False)))
    return run_id
