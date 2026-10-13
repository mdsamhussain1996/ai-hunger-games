from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Contestant(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    name: str
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)
    base_model: str = "llama3.2:3b"
    temperature: float = 0.7
    system_prompt: str
    created_at: datetime = Field(default_factory=utc_now)
    eliminated_at: Optional[datetime] = None
    status: Literal["active", "eliminated", "archived"] = "active"
    replacement_strategy: Optional[str] = "seed"


class Question(BaseModel):
    id: str
    category: str
    prompt: str


class Answer(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    round_id: str
    contestant_id: str
    contestant_name: str
    question_id: str
    text: str
    token_count: Optional[int] = None
    latency_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=utc_now)


class CandidateAnswer(BaseModel):
    label: str  # "A", "B", "C"...
    contestant_id: str
    contestant_name: str
    text: str


class VotePayload(BaseModel):
    vote: str = Field(description="The single best answer label, e.g. 'A' or 'B'")
    ranking: List[str] = Field(description="Full ranking of all candidate labels from best to worst, e.g. ['B', 'A']")
    reason: str = Field(description="Short rationale explaining the vote and ranking")


class VoteRecord(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    round_id: str
    season_id: int = 1
    voter_id: str
    voter_name: str
    votee_id: Optional[str] = None
    votee_name: Optional[str] = None
    vote_label: str
    ranking_labels: List[str] = Field(default_factory=list)
    ranking_contestant_ids: List[str] = Field(default_factory=list)
    reason: str
    is_valid: bool = True
    raw_response: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class RoundScore(BaseModel):
    contestant_id: str
    contestant_name: str
    plurality_points: int = 0
    borda_points: int = 0
    total_points: int = 0


class DiplomacyMessage(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    round_id: str
    season_id: int
    sender_id: str
    sender_name: str
    recipient_id: str
    recipient_name: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class JudgeEvaluation(BaseModel):
    round_id: str
    winner_contestant_id: Optional[str] = None
    winner_contestant_name: Optional[str] = None
    ranking_contestant_ids: List[str] = Field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class RoundSummary(BaseModel):
    id: str = Field(default_factory=generate_uuid)
    round_number: int
    season_number: int
    question: Question
    answers: List[Answer]
    votes: List[VoteRecord]
    scores: List[RoundScore]
    messages: List[DiplomacyMessage] = Field(default_factory=list)
    judge_eval: Optional[JudgeEvaluation] = None
    created_at: datetime = Field(default_factory=utc_now)


class GeneralConfig(BaseModel):
    seed: int = 42
    run_id: str = "default_run"
    db_path: str = "data/hunger_games.db"


class ModelsConfig(BaseModel):
    default_model: str = "llama3.2:3b"
    mix_models: bool = False
    model_mapping: Dict[str, str] = Field(default_factory=dict)
    embedding_model: str = "nomic-embed-text"
    judge_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    timeout_seconds: int = 45
    max_retries: int = 2
    concurrency_limit: int = 1


class GameConfig(BaseModel):
    num_contestants: int = 8
    rounds_per_season: int = 8
    max_answer_tokens: int = 120
    scoring_system: Literal["borda", "plurality"] = "borda"
    replacement_strategy: Literal["random", "mutation", "crossover"] = "crossover"


class ResearchModeConfig(BaseModel):
    identity_mode: Literal["anonymous", "named"] = "anonymous"
    diplomacy_phase: bool = False
    memory: Literal["none", "season_summary"] = "none"
    game_master_judge: bool = False


class AppConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    research_mode: ResearchModeConfig = Field(default_factory=ResearchModeConfig)
