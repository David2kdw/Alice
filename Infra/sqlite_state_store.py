# alice/infra/sqlite_state_store.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Domain.models import StateBundle


@dataclass
class SQLiteStateStore:
    """
    SQLite-backed StateBundle store (single-slot).

    Stores exactly one StateBundle under a fixed key.
    - load(): returns None if no state is stored yet
    - save(): upserts the latest StateBundle

    Notes:
    - Uses JSON serialization via StateBundle.to_dict()/from_dict()
    - Timestamps are already ISO strings inside the dict (Domain.utils.isoformat)
    """

    db_path: str
    key: str = "default"
    schema_version: int = 1

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_bundle (
                    key TEXT PRIMARY KEY,
                    json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def load(self) -> Optional[StateBundle]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT json, schema_version FROM state_bundle WHERE key = ? LIMIT 1;",
            (self.key,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        json_str, sv = row
        # If you ever bump schema_version, you can add migrations here.
        _ = sv  # reserved
        try:
            data = json.loads(json_str)
            return StateBundle.from_dict(data)
        except Exception:
            # Corrupted row or incompatible schema => treat as no state
            return None

    def save(self, bundle: StateBundle) -> None:
        data = bundle.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        updated_at = data.get("last_tick_time")  # already ISO string

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO state_bundle(key, json, schema_version, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    json=excluded.json,
                    schema_version=excluded.schema_version,
                    updated_at=excluded.updated_at;
                """,
                (self.key, json_str, int(self.schema_version), str(updated_at)),
            )
