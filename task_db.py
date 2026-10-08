"""任务数据库模块：保存视频分析任务与结果。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

from config import (
    DEFAULT_DB_PATH,
    TASK_STATUS_DONE,
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_UPLOADED,
)


class TaskDB:
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
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    input_video TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    output_video TEXT,
                    counts_csv TEXT,
                    events_csv TEXT,
                    traffic_csv TEXT,
                    report_png TEXT,
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    flow_total INTEGER DEFAULT 0,
                    avg_speed_kmh REAL DEFAULT 0,
                    max_speed_kmh REAL DEFAULT 0,
                    measured_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    config_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "config_json" not in existing_cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN config_json TEXT")
            conn.commit()

    def create_task(
        self,
        user_id: int,
        username: str,
        input_video: str,
        output_dir: str,
        config_json: str | None = None,
    ) -> int:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (user_id, username, input_video, output_dir, status, created_at, config_json)
                VALUES (?, ?, ?, ?, 'uploaded', ?, ?)
                """,
                (int(user_id), str(username), str(input_video), str(output_dir), now, config_json),
            )
            conn.commit()
            return int(cur.lastrowid)

    def mark_running_if_possible(self, task_id: int) -> bool:
        """将任务切换为 running。

        已完成任务也允许重新运行：用户在可视化调线页修改线条后再次
        点击“开始分析”，系统会覆盖当前任务的旧输出和旧统计字段。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?,
                    started_at=?,
                    finished_at=NULL,
                    error_message=NULL,
                    output_video=NULL,
                    counts_csv=NULL,
                    events_csv=NULL,
                    traffic_csv=NULL,
                    report_png=NULL,
                    flow_total=0,
                    avg_speed_kmh=0,
                    max_speed_kmh=0,
                    measured_count=0
                WHERE id=? AND status IN (?, ?, ?)
                """,
                (
                    TASK_STATUS_RUNNING,
                    now,
                    int(task_id),
                    TASK_STATUS_UPLOADED,
                    TASK_STATUS_FAILED,
                    TASK_STATUS_DONE,
                ),
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def mark_done(
        self,
        task_id: int,
        output_video: str,
        counts_csv: str,
        events_csv: str,
        traffic_csv: str,
        report_png: str,
        flow_total: int,
        avg_speed_kmh: float,
        max_speed_kmh: float,
        measured_count: int,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status=?,
                    output_video=?, counts_csv=?, events_csv=?, traffic_csv=?, report_png=?,
                    flow_total=?, avg_speed_kmh=?, max_speed_kmh=?, measured_count=?,
                    finished_at=?
                WHERE id=? AND status=?
                """,
                (
                    TASK_STATUS_DONE,
                    str(output_video),
                    str(counts_csv),
                    str(events_csv),
                    str(traffic_csv),
                    str(report_png),
                    int(flow_total),
                    float(avg_speed_kmh),
                    float(max_speed_kmh),
                    int(measured_count),
                    now,
                    int(task_id),
                    TASK_STATUS_RUNNING,
                ),
            )
            conn.commit()

    def mark_failed(self, task_id: int, error_message: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, error_message=?, finished_at=? WHERE id=?",
                (TASK_STATUS_FAILED, str(error_message), now, int(task_id)),
            )
            conn.commit()

    def update_config(self, task_id: int, config_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET config_json=? WHERE id=?",
                (str(config_json), int(task_id)),
            )
            conn.commit()

    def delete_task(self, task_id: int) -> None:
        """删除一条任务记录。

        这里只删除数据库中的网页记录，不主动删除上传视频和分析结果文件，
        避免用户误点后无法找回实验数据。运行中的任务由 Flask 路由拦截。
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE id=?", (int(task_id),))
            conn.commit()

    def get_task(self, task_id: int):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM tasks WHERE id=?", (int(task_id),)).fetchone()

    def list_tasks(self, user_id: int, role: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            if role == "admin":
                rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE user_id=? ORDER BY id DESC",
                    (int(user_id),),
                ).fetchall()
        return rows

    def user_can_access(self, task_id: int, user_id: int, role: str) -> bool:
        row = self.get_task(task_id)
        if row is None:
            return False
        if role == "admin":
            return True
        return int(row["user_id"]) == int(user_id)
