"""用户数据库模块：SQLite 初始化与基础用户管理。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import DEFAULT_DB_PATH


@dataclass
class UserRecord:
    id: int
    username: str
    role: str
    created_at: str


class UserDB:
    """用户数据访问层。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_user(self, username: str, password_hash: str, role: str = "user") -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, role, created_at),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_user_by_username(self, username: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
                (username,),
            )
            return cur.fetchone()

    def list_users(self) -> List[UserRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY id ASC"
            ).fetchall()
        return [
            UserRecord(
                id=int(row["id"]),
                username=str(row["username"]),
                role=str(row["role"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def delete_user(self, username: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            return int(cur.rowcount) > 0
