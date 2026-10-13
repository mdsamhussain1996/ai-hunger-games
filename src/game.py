from __future__ import annotations

import glob
import json
import logging
import os
import random
import string
from typing import Dict, List, Optional, Tuple

from src.models import (
    Answer,
    CandidateAnswer,
    Contestant,
    Question,
    RoundScore,
    RoundSummary,
    VoteRecord,
)
from src.llm import LLMClient

logger = logging.getLogger(__name__)


class QuestionBank:
    """Loads and samples questions from JSONL files."""

    def __init__(self, questions_dir: str = "questions"):
        self.questions_dir = questions_dir
        self.questions: List[Question] = []
        self._load_questions()

    def _load_questions(self) -> None:
        pattern = os.path.join(self.questions_dir, "*.jsonl")
        files = glob.glob(pattern)
        loaded = []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        loaded.append(
                            Question(
                                id=data["id"],
                                category=data.get("category", "general"),
                                prompt=data["prompt"],
                            )
                        )
                    except Exception as e:
                        logger.warning("Error parsing question in %s: %s", fpath, e)
        self.questions = loaded
        if not self.questions:
            # Fallback question if none found
            self.questions.append(
                Question(
                    id="fallback-01",
                    category="general",
                    prompt="What is the most significant technological challenge of the 21st century?",
                )
            )

    def sample(self, rng: Optional[random.Random] = None) -> Question:
        if rng:
            return rng.choice(self.questions)
        return random.choice(self.questions)

    def get_by_category(self, category: str) -> List[Question]:
        return [q for q in self.questions if q.category == category]


def calculate_round_scores(
    contestants: List[Contestant],
    votes: List[VoteRecord],
    scoring_system: str = "borda",
) -> List[RoundScore]:
    """Calculate plurality and Borda scores for each contestant in the round."""
    plurality_map: Dict[str, int] = {c.id: 0 for c in contestants}
    borda_map: Dict[str, int] = {c.id: 0 for c in contestants}
    name_map: Dict[str, str] = {c.id: c.name for c in contestants}

    for v in votes:
        if not v.is_valid:
            continue

        # Plurality point to the single top choice
        if v.votee_id and v.votee_id in plurality_map:
            plurality_map[v.votee_id] += 1

        # Borda points: for K ranked items, top gets K-1, next K-2, ... bottom gets 0
        k = len(v.ranking_contestant_ids)
        for rank_idx, cid in enumerate(v.ranking_contestant_ids):
            if cid in borda_map:
                points = max(0, k - 1 - rank_idx)
                borda_map[cid] += points

    scores = []
    for c in contestants:
        p_pts = plurality_map.get(c.id, 0)
        b_pts = borda_map.get(c.id, 0)
        tot = b_pts if scoring_system == "borda" else p_pts
        scores.append(
            RoundScore(
                contestant_id=c.id,
                contestant_name=name_map.get(c.id, c.id),
                plurality_points=p_pts,
                borda_points=b_pts,
                total_points=tot,
            )
        )

    # Sort descending by total points
    scores.sort(key=lambda s: s.total_points, reverse=True)
    return scores


def run_single_round(
    round_number: int,
    season_number: int,
    contestants: List[Contestant],
    question: Question,
    llm: LLMClient,
    scoring_system: str = "borda",
    is_named: bool = False,
    max_answer_tokens: int = 120,
    rng: Optional[random.Random] = None,
) -> RoundSummary:
    """Execute one complete round: answers, voting with position-bias cancellation, and scoring."""
    r = rng or random.Random()
    round_id = f"s{season_number}_r{round_number}_{question.id}"

    # Phase 1: Answers
    answers: List[Answer] = []
    for c in contestants:
        ans_text, latency, tokens = llm.generate_answer(
            contestant=c,
            question=question,
            max_tokens=max_answer_tokens,
        )
        answers.append(
            Answer(
                round_id=round_id,
                contestant_id=c.id,
                contestant_name=c.name,
                question_id=question.id,
                text=ans_text,
                token_count=tokens,
                latency_ms=latency,
            )
        )

    # Phase 2: Voting
    # Build a lookup for answers by contestant id
    ans_by_cid = {a.contestant_id: a for a in answers}
    votes: List[VoteRecord] = []

    for voter in contestants:
        # Exclude voter's own answer to prevent self-voting
        other_contestants = [c for c in contestants if c.id != voter.id]
        # Shuffle other answers specifically for this voter to cancel position bias
        shuffled_others = list(other_contestants)
        r.shuffle(shuffled_others)

        # Label candidate answers as A, B, C...
        candidates: List[CandidateAnswer] = []
        label_to_cid: Dict[str, str] = {}
        for idx, other_c in enumerate(shuffled_others):
            label = string.ascii_uppercase[idx]
            other_ans = ans_by_cid[other_c.id]
            candidates.append(
                CandidateAnswer(
                    label=label,
                    contestant_id=other_c.id,
                    contestant_name=other_c.name,
                    text=other_ans.text,
                )
            )
            label_to_cid[label] = other_c.id

        # Ask voter for vote and ranking
        payload, raw_resp, is_valid = llm.generate_vote(
            contestant=voter,
            question=question,
            candidates=candidates,
            is_named=is_named,
        )

        if payload and is_valid:
            vote_label = payload.vote
            votee_id = label_to_cid.get(vote_label)
            votee_name = next((c.name for c in contestants if c.id == votee_id), None)
            ranking_cids = [label_to_cid[lbl] for lbl in payload.ranking if lbl in label_to_cid]

            votes.append(
                VoteRecord(
                    round_id=round_id,
                    season_id=season_number,
                    voter_id=voter.id,
                    voter_name=voter.name,
                    votee_id=votee_id,
                    votee_name=votee_name,
                    vote_label=vote_label,
                    ranking_labels=payload.ranking,
                    ranking_contestant_ids=ranking_cids,
                    reason=payload.reason,
                    is_valid=True,
                    raw_response=raw_resp,
                )
            )
        else:
            # Record an abstention
            votes.append(
                VoteRecord(
                    round_id=round_id,
                    season_id=season_number,
                    voter_id=voter.id,
                    voter_name=voter.name,
                    votee_id=None,
                    votee_name=None,
                    vote_label="ABSTAIN",
                    ranking_labels=[],
                    ranking_contestant_ids=[],
                    reason="Abstention: malformed vote response after maximum retries.",
                    is_valid=False,
                    raw_response=raw_resp,
                )
            )

    # Phase 3: Scoring
    scores = calculate_round_scores(
        contestants=contestants,
        votes=votes,
        scoring_system=scoring_system,
    )

    return RoundSummary(
        id=round_id,
        round_number=round_number,
        season_number=season_number,
        question=question,
        answers=answers,
        votes=votes,
        scores=scores,
    )


class TournamentEngine:
    """Manages full tournament execution across multiple rounds and seasons."""

    def __init__(
        self,
        config: any,
        llm: LLMClient,
        db: any,
        run_id: str,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.run_id = run_id
        self.rng = random.Random(config.general.seed)
        self.q_bank = QuestionBank("questions")
        self.active_contestants: List[Contestant] = []
        self.current_season: int = 1
        self.current_round: int = 0
        self.generation_counter: int = 0

    def initialize_contestants(self, seeds: List[Contestant]) -> None:
        """Initialize contestants for a new run."""
        self.active_contestants = seeds[: self.config.game.num_contestants]
        self.db.create_run(self.run_id, self.config.general.seed, self.config.model_dump())
        for c in self.active_contestants:
            self.db.save_contestant(self.run_id, c)

    def play_round(self, question: Optional[Question] = None) -> RoundSummary:
        """Play the next round in the current season."""
        self.current_round += 1
        q = question or self.q_bank.sample(self.rng)

        summary = run_single_round(
            round_number=self.current_round,
            season_number=self.current_season,
            contestants=self.active_contestants,
            question=q,
            llm=self.llm,
            scoring_system=self.config.game.scoring_system,
            is_named=self.config.research_mode.identity_mode == "named",
            max_answer_tokens=self.config.game.max_answer_tokens,
            rng=self.rng,
        )

        # Save to database
        self.db.save_round_summary(self.run_id, summary)
        return summary

    def resolve_season_elimination(self) -> Tuple[Contestant, Contestant]:
        """Determine lowest scorer across the season, eliminate them, and evolve a replacement."""
        from src.evolution import generate_new_contestant

        season_scores = self.db.get_cumulative_season_scores(self.run_id, self.current_season)

        # Map to active contestants
        c_by_id = {c.id: c for c in self.active_contestants}

        # Tie breaking:
        # 1. Lower cumulative points
        # 2. Fewer 1st-place plurality votes
        # 3. Random deterministic seed
        def sort_key(s):
            cid = s["contestant_id"]
            pts = s["cumulative_points"]
            plurality = s["total_plurality"]
            rnd_tie = self.rng.random()
            return (pts, plurality, rnd_tie)

        sorted_scores = sorted(season_scores, key=sort_key)

        eliminated_id = sorted_scores[0]["contestant_id"]
        eliminated_c = c_by_id[eliminated_id]

        # Top scorers for reproduction
        top_scorers = [c_by_id[s["contestant_id"]] for s in reversed(sorted_scores) if s["contestant_id"] != eliminated_id]

        # Evolve replacement
        self.generation_counter += 1
        replacement_c = generate_new_contestant(
            strategy=self.config.game.replacement_strategy,
            active_contestants=self.active_contestants,
            top_scorers=top_scorers,
            generation=self.generation_counter,
            base_model=self.config.models.default_model,
            llm=self.llm,
        )

        # Record in DB
        self.db.save_contestant(self.run_id, replacement_c)
        self.db.record_season_elimination(
            run_id=self.run_id,
            season_number=self.current_season,
            eliminated_id=eliminated_id,
            replacement_id=replacement_c.id,
        )

        # Update local active roster
        self.active_contestants = [c for c in self.active_contestants if c.id != eliminated_id] + [replacement_c]

        # Advance season
        self.current_season += 1
        self.current_round = 0

        return eliminated_c, replacement_c

