from __future__ import annotations

import argparse
import random
import sys
import yaml

from src.contestants import load_seed_personalities
from src.game import QuestionBank, run_single_round
from src.llm import LLMClient
from src.models import AppConfig


def print_banner():
    banner = """
================================================================================
           🏹  A I   H U N G E R   G A M E S  :  T O U R N A M E N T  🏹
================================================================================
    """
    print(banner)


def run_demo_round(num_contestants: int = 3, category: str = None, force_mock: bool = False):
    print_banner()

    # 1. Load config
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
            config = AppConfig.model_validate(raw_cfg)
    except Exception:
        config = AppConfig()

    rng = random.Random(config.general.seed)

    # 2. Initialize LLM client
    llm = LLMClient(
        host=config.models.ollama_host,
        timeout_seconds=config.models.timeout_seconds,
        max_retries=config.models.max_retries,
        force_mock=force_mock,
    )

    ollama_online = llm.is_ollama_available()
    if ollama_online:
        print(f" [OLLAMA STATUS]: ONLINE (Connected to {config.models.ollama_host})")
    else:
        print(f" [OLLAMA STATUS]: OFFLINE / SIMULATION MODE (Persona-grounded fallback active)")

    # 3. Select Contestants
    all_seeds = load_seed_personalities("personalities/seed.yaml", default_model=config.models.default_model)
    active = all_seeds[:num_contestants]
    print(f"\n Active Contestants ({len(active)}):")
    for idx, c in enumerate(active, 1):
        print(f"   {idx}. {c.name} [Model: {c.base_model}, Temp: {c.temperature}]")

    # 4. Question Selection
    q_bank = QuestionBank("questions")
    if category:
        cat_questions = q_bank.get_by_category(category)
        question = rng.choice(cat_questions) if cat_questions else q_bank.sample(rng)
    else:
        question = q_bank.sample(rng)

    print(f"\n" + "-" * 80)
    print(f" ROUND 1 QUESTION [{question.category.upper()}] (ID: {question.id}):")
    print(f" \"{question.prompt}\"")
    print("-" * 80)

    # 5. Execute Round
    print("\n Generating answers from contestants...")
    summary = run_single_round(
        round_number=1,
        season_number=1,
        contestants=active,
        question=question,
        llm=llm,
        scoring_system=config.game.scoring_system,
        is_named=config.research_mode.identity_mode == "named",
        max_answer_tokens=config.game.max_answer_tokens,
        rng=rng,
    )

    # 6. Display Answers
    print("\n" + "=" * 80)
    print(" PHASE 1: CONTESTANT ANSWERS")
    print("=" * 80)
    for ans in summary.answers:
        tokens_info = f"{ans.token_count} tokens" if ans.token_count else ""
        latency_info = f"{ans.latency_ms:.0f}ms" if ans.latency_ms else ""
        meta = f"({', '.join(filter(None, [tokens_info, latency_info]))})"
        print(f"\n [{ans.contestant_name}] {meta}:")
        print(f"   \"{ans.text}\"")

    # 7. Display Voting & Reasoning
    print("\n" + "=" * 80)
    print(" PHASE 2: VOTING & REASONING (Self-voting excluded, positions shuffled)")
    print("=" * 80)
    for v in summary.votes:
        votee_display = v.votee_name if v.votee_name else "ABSTAIN"
        print(f"\n Voter: {v.voter_name}")
        print(f"   Voted For  : [{v.vote_label}] -> {votee_display}")
        print(f"   Full Rank  : {' > '.join(v.ranking_labels)}")
        print(f"   Reason     : {v.reason}")

    # 8. Display Round Leaderboard
    print("\n" + "=" * 80)
    print(f" PHASE 3: ROUND SCOREBOARD (Scoring: {config.game.scoring_system.upper()})")
    print("=" * 80)
    print(f" {'Rank':<6} {'Contestant':<26} {'Borda Pts':<12} {'1st-Place Votes':<18} {'Round Points':<12}")
    print("-" * 80)
    for rank, score in enumerate(summary.scores, 1):
        print(
            f" {rank:<6} {score.contestant_name:<26} {score.borda_points:<12} "
            f"{score.plurality_points:<18} {score.total_points:<12}"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Hunger Games CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: round (demo single round)
    round_parser = subparsers.add_parser("round", help="Run a single round demonstration")
    round_parser.add_argument("--contestants", type=int, default=3, help="Number of contestants (default: 3)")
    round_parser.add_argument("--category", type=str, default=None, help="Question category filter")
    round_parser.add_argument("--mock", action="store_true", help="Force offline mock mode")

    args = parser.parse_args()

    if args.command == "round" or args.command is None:
        num_c = getattr(args, "contestants", 3)
        cat = getattr(args, "category", None)
        mock = getattr(args, "mock", False)
        run_demo_round(num_contestants=num_c, category=cat, force_mock=mock)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
