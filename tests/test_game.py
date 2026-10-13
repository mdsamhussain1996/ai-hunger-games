from __future__ import annotations

import random
import pytest

from src.game import calculate_round_scores, run_single_round
from src.llm import LLMClient, extract_json_object
from src.models import Contestant, Question, VotePayload, VoteRecord


def test_extract_json_object_markdown():
    text = """
    Here is my vote:
    ```json
    {
      "vote": "B",
      "ranking": ["B", "A", "C"],
      "reason": "Clear empirical clarity."
    }
    ```
    Thank you.
    """
    res = extract_json_object(text)
    assert res is not None
    assert res["vote"] == "B"
    assert res["ranking"] == ["B", "A", "C"]
    assert "clarity" in res["reason"]


def test_extract_json_object_raw():
    text = '{"vote": "A", "ranking": ["A", "B"], "reason": "Superior logic"}'
    res = extract_json_object(text)
    assert res is not None
    assert res["vote"] == "A"


def test_borda_and_plurality_scoring():
    c1 = Contestant(id="c1", name="Alpha", system_prompt="Test")
    c2 = Contestant(id="c2", name="Beta", system_prompt="Test")
    c3 = Contestant(id="c3", name="Gamma", system_prompt="Test")
    contestants = [c1, c2, c3]

    # Voter 1 ranks c2 > c3 > c1
    # Voter 2 ranks c2 > c1 > c3
    # Voter 3 ranks c1 > c2 > c3
    votes = [
        VoteRecord(
            round_id="r1",
            voter_id="c1",
            voter_name="Alpha",
            votee_id="c2",
            votee_name="Beta",
            vote_label="A",
            ranking_labels=["A", "B"],
            ranking_contestant_ids=["c2", "c3"],
            reason="Good",
        ),
        VoteRecord(
            round_id="r1",
            voter_id="c2",
            voter_name="Beta",
            votee_id="c1",
            votee_name="Alpha",
            vote_label="A",
            ranking_labels=["A", "B"],
            ranking_contestant_ids=["c1", "c3"],
            reason="Good",
        ),
        VoteRecord(
            round_id="r1",
            voter_id="c3",
            voter_name="Gamma",
            votee_id="c2",
            votee_name="Beta",
            vote_label="A",
            ranking_labels=["A", "B"],
            ranking_contestant_ids=["c2", "c1"],
            reason="Good",
        ),
    ]

    scores = calculate_round_scores(contestants, votes, scoring_system="borda")
    score_by_id = {s.contestant_id: s for s in scores}

    # In each 2-candidate vote: rank 0 gets 1 pt, rank 1 gets 0 pts
    # c2 got rank 0 from voter 1 (1 pt) and voter 3 (1 pt) = 2 borda pts, 2 plurality votes
    # c1 got rank 0 from voter 2 (1 pt) and rank 1 from voter 3 (0 pt) = 1 borda pt, 1 plurality vote
    # c3 got rank 1 from voter 1 (0 pt) and rank 1 from voter 2 (0 pt) = 0 borda pts, 0 plurality votes
    assert score_by_id["c2"].borda_points == 2
    assert score_by_id["c2"].plurality_points == 2
    assert score_by_id["c1"].borda_points == 1
    assert score_by_id["c1"].plurality_points == 1
    assert score_by_id["c3"].borda_points == 0
    assert score_by_id["c3"].plurality_points == 0


def test_shuffle_position_mapping():
    """Verify that candidate answers are mapped correctly back to the votee ID."""
    c1 = Contestant(id="c1", name="Alpha", system_prompt="Test")
    c2 = Contestant(id="c2", name="Beta", system_prompt="Test")
    c3 = Contestant(id="c3", name="Gamma", system_prompt="Test")
    q = Question(id="q1", category="general", prompt="Test Question")

    llm = LLMClient(force_mock=True)
    summary = run_single_round(
        round_number=1,
        season_number=1,
        contestants=[c1, c2, c3],
        question=q,
        llm=llm,
        rng=random.Random(42),
    )

    assert len(summary.answers) == 3
    assert len(summary.votes) == 3
    for v in summary.votes:
        assert v.votee_id is not None
        assert v.votee_id != v.voter_id  # Self-voting impossible
        assert v.votee_id in ["c1", "c2", "c3"]
