from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.models import (
    Answer,
    Contestant,
    DiplomacyMessage,
    JudgeEvaluation,
    Question,
    RoundScore,
    RoundSummary,
    VoteRecord,
    utc_now,
)


class Database:
    """SQLite persistence layer for AI Hunger Games experiment runs."""

    def __init__(self, db_path: str = "data/hunger_games.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Runs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    seed INTEGER,
                    config_json TEXT,
                    status TEXT,
                    created_at TIMESTAMP
                )
                """
            )

            # Contestants table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS contestants (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    name TEXT,
                    generation INTEGER,
                    parent_ids_json TEXT,
                    base_model TEXT,
                    temperature REAL,
                    system_prompt TEXT,
                    status TEXT,
                    replacement_strategy TEXT,
                    created_at TIMESTAMP,
                    eliminated_at TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Seasons table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS seasons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    season_number INTEGER,
                    eliminated_contestant_id TEXT,
                    replacement_contestant_id TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Rounds table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rounds (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    season_number INTEGER,
                    round_number INTEGER,
                    question_id TEXT,
                    question_category TEXT,
                    question_prompt TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Answers table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    id TEXT PRIMARY KEY,
                    round_id TEXT,
                    run_id TEXT,
                    contestant_id TEXT,
                    contestant_name TEXT,
                    text TEXT,
                    token_count INTEGER,
                    latency_ms REAL,
                    created_at TIMESTAMP,
                    FOREIGN KEY(round_id) REFERENCES rounds(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Votes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS votes (
                    id TEXT PRIMARY KEY,
                    round_id TEXT,
                    run_id TEXT,
                    season_number INTEGER,
                    voter_id TEXT,
                    voter_name TEXT,
                    votee_id TEXT,
                    votee_name TEXT,
                    vote_label TEXT,
                    ranking_labels_json TEXT,
                    ranking_contestant_ids_json TEXT,
                    reason TEXT,
                    is_valid INTEGER,
                    raw_response TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY(round_id) REFERENCES rounds(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Scores table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id TEXT,
                    run_id TEXT,
                    season_number INTEGER,
                    round_number INTEGER,
                    contestant_id TEXT,
                    contestant_name TEXT,
                    plurality_points INTEGER,
                    borda_points INTEGER,
                    total_points INTEGER,
                    FOREIGN KEY(round_id) REFERENCES rounds(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Diplomacy messages table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS diplomacy_messages (
                    id TEXT PRIMARY KEY,
                    round_id TEXT,
                    run_id TEXT,
                    season_number INTEGER,
                    sender_id TEXT,
                    sender_name TEXT,
                    recipient_id TEXT,
                    recipient_name TEXT,
                    content TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY(round_id) REFERENCES rounds(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            # Neutral judge evaluations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_evaluations (
                    round_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    winner_contestant_id TEXT,
                    winner_contestant_name TEXT,
                    ranking_json TEXT,
                    reasoning TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY(round_id) REFERENCES rounds(id),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
                """
            )

            conn.commit()

    def create_run(self, run_id: str, seed: int, config_dict: dict) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs (id, seed, config_json, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, seed, json.dumps(config_dict), "running", utc_now().isoformat()),
            )
            conn.commit()

    def update_run_status(self, run_id: str, status: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
            conn.commit()

    def save_contestant(self, run_id: str, c: Contestant) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO contestants (
                    id, run_id, name, generation, parent_ids_json, base_model,
                    temperature, system_prompt, status, replacement_strategy,
                    created_at, eliminated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.id,
                    run_id,
                    c.name,
                    c.generation,
                    json.dumps(c.parent_ids),
                    c.base_model,
                    c.temperature,
                    c.system_prompt,
                    c.status,
                    c.replacement_strategy,
                    c.created_at.isoformat() if isinstance(c.created_at, datetime) else str(c.created_at),
                    c.eliminated_at.isoformat() if isinstance(c.eliminated_at, datetime) else None,
                ),
            )
            conn.commit()

    def save_round_summary(self, run_id: str, summary: RoundSummary) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert Round
            cursor.execute(
                """
                INSERT OR REPLACE INTO rounds (
                    id, run_id, season_number, round_number,
                    question_id, question_category, question_prompt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.id,
                    run_id,
                    summary.season_number,
                    summary.round_number,
                    summary.question.id,
                    summary.question.category,
                    summary.question.prompt,
                    summary.created_at.isoformat(),
                ),
            )

            # Insert Answers
            for ans in summary.answers:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO answers (
                        id, round_id, run_id, contestant_id, contestant_name,
                        text, token_count, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ans.id,
                        summary.id,
                        run_id,
                        ans.contestant_id,
                        ans.contestant_name,
                        ans.text,
                        ans.token_count,
                        ans.latency_ms,
                        ans.created_at.isoformat(),
                    ),
                )

            # Insert Votes
            for v in summary.votes:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO votes (
                        id, round_id, run_id, season_number, voter_id, voter_name,
                        votee_id, votee_name, vote_label, ranking_labels_json,
                        ranking_contestant_ids_json, reason, is_valid, raw_response, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        v.id,
                        summary.id,
                        run_id,
                        summary.season_number,
                        v.voter_id,
                        v.voter_name,
                        v.votee_id,
                        v.votee_name,
                        v.vote_label,
                        json.dumps(v.ranking_labels),
                        json.dumps(v.ranking_contestant_ids),
                        v.reason,
                        1 if v.is_valid else 0,
                        v.raw_response,
                        v.created_at.isoformat(),
                    ),
                )

            # Insert Scores
            for sc in summary.scores:
                cursor.execute(
                    """
                    INSERT INTO scores (
                        round_id, run_id, season_number, round_number,
                        contestant_id, contestant_name, plurality_points,
                        borda_points, total_points
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary.id,
                        run_id,
                        summary.season_number,
                        summary.round_number,
                        sc.contestant_id,
                        sc.contestant_name,
                        sc.plurality_points,
                        sc.borda_points,
                        sc.total_points,
                    ),
                )

            # Insert Diplomacy Messages if any
            for msg in summary.messages:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO diplomacy_messages (
                        id, round_id, run_id, season_number,
                        sender_id, sender_name, recipient_id, recipient_name,
                        content, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg.id,
                        summary.id,
                        run_id,
                        summary.season_number,
                        msg.sender_id,
                        msg.sender_name,
                        msg.recipient_id,
                        msg.recipient_name,
                        msg.content,
                        msg.created_at.isoformat(),
                    ),
                )

            # Insert Judge Evaluation if any
            if summary.judge_eval:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO judge_evaluations (
                        round_id, run_id, winner_contestant_id, winner_contestant_name,
                        ranking_json, reasoning, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary.id,
                        run_id,
                        summary.judge_eval.winner_contestant_id,
                        summary.judge_eval.winner_contestant_name,
                        json.dumps(summary.judge_eval.ranking_contestant_ids),
                        summary.judge_eval.reasoning,
                        summary.judge_eval.created_at.isoformat(),
                    ),
                )

            conn.commit()

    def record_season_elimination(
        self,
        run_id: str,
        season_number: int,
        eliminated_id: str,
        replacement_id: str,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO seasons (run_id, season_number, eliminated_contestant_id, replacement_contestant_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, season_number, eliminated_id, replacement_id, utc_now().isoformat()),
            )
            cursor.execute(
                "UPDATE contestants SET status = 'eliminated', eliminated_at = ? WHERE id = ? AND run_id = ?",
                (utc_now().isoformat(), eliminated_id, run_id),
            )
            conn.commit()

    def get_runs(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_contestants(self, run_id: str) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM contestants WHERE run_id = ? ORDER BY generation, created_at", (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_rounds(self, run_id: str, season_number: Optional[int] = None) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if season_number is not None:
                cursor.execute(
                    "SELECT * FROM rounds WHERE run_id = ? AND season_number = ? ORDER BY season_number, round_number",
                    (run_id, season_number),
                )
            else:
                cursor.execute("SELECT * FROM rounds WHERE run_id = ? ORDER BY season_number, round_number", (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_votes(self, run_id: str, season_number: Optional[int] = None) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if season_number is not None:
                cursor.execute(
                    "SELECT * FROM votes WHERE run_id = ? AND season_number = ? ORDER BY created_at",
                    (run_id, season_number),
                )
            else:
                cursor.execute("SELECT * FROM votes WHERE run_id = ? ORDER BY created_at", (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_scores(self, run_id: str) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM scores WHERE run_id = ? ORDER BY season_number, round_number", (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_cumulative_season_scores(self, run_id: str, season_number: int) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    contestant_id,
                    contestant_name,
                    SUM(borda_points) as total_borda,
                    SUM(plurality_points) as total_plurality,
                    SUM(total_points) as cumulative_points,
                    COUNT(DISTINCT round_id) as rounds_played
                FROM scores
                WHERE run_id = ? AND season_number = ?
                GROUP BY contestant_id, contestant_name
                ORDER BY cumulative_points DESC
                """,
                (run_id, season_number),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_seasons(self, run_id: str) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM seasons WHERE run_id = ? ORDER BY season_number", (run_id,))
            return [dict(row) for row in cursor.fetchall()]
