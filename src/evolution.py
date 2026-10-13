from __future__ import annotations

import json
import logging
import re
from typing import List, Optional
import uuid

from src.models import Contestant
from src.llm import LLMClient

logger = logging.getLogger(__name__)


def generate_new_contestant(
    strategy: str,
    active_contestants: List[Contestant],
    top_scorers: List[Contestant],
    generation: int,
    base_model: str,
    llm: LLMClient,
) -> Contestant:
    """Generate a new replacement contestant using random, mutation, or crossover strategy."""
    c_id = f"gen{generation}_{uuid.uuid4().hex[:6]}"

    if strategy == "mutation" and top_scorers:
        parent = top_scorers[0]
        name, prompt = mutate_personality(parent, llm)
        return Contestant(
            id=c_id,
            name=name,
            generation=generation,
            parent_ids=[parent.id],
            base_model=base_model,
            temperature=parent.temperature,
            system_prompt=prompt,
            status="active",
            replacement_strategy="mutation",
        )

    elif strategy == "crossover" and len(top_scorers) >= 2:
        p1, p2 = top_scorers[0], top_scorers[1]
        name, prompt = crossover_personalities(p1, p2, llm)
        return Contestant(
            id=c_id,
            name=name,
            generation=generation,
            parent_ids=[p1.id, p2.id],
            base_model=base_model,
            temperature=round((p1.temperature + p2.temperature) / 2.0, 2),
            system_prompt=prompt,
            status="active",
            replacement_strategy="crossover",
        )

    else:  # "random" or fallback
        name, prompt = generate_random_personality(active_contestants, llm)
        return Contestant(
            id=c_id,
            name=name,
            generation=generation,
            parent_ids=[],
            base_model=base_model,
            temperature=0.7,
            system_prompt=prompt,
            status="active",
            replacement_strategy="random",
        )


def generate_random_personality(current: List[Contestant], llm: LLMClient) -> tuple[str, str]:
    """Game Master generates a brand-new persona distinct from current active ones."""
    existing_summaries = "\n".join(f"- {c.name}: {c.system_prompt[:80]}..." for c in current)
    user_prompt = (
        f"You are the Game Master of the AI Hunger Games.\n"
        f"Currently active personalities are:\n{existing_summaries}\n\n"
        f"Invent a brand-new, highly distinct personality that is completely different from the ones above.\n"
        f"Return ONLY valid JSON with keys 'name' and 'prompt' (4-6 lines defining voice, values, and how it judges answers):\n"
        f'{{"name": "The <Title>", "prompt": "You are The <Title>. You value ... When judging answers, you reward ... and penalize ..."}}'
    )

    raw, _, _ = llm.generate_raw(
        model=current[0].base_model if current else "llama3.2:1b",
        system_prompt="You are a creative Game Master designing distinct cognitive characters.",
        user_prompt=user_prompt,
        temperature=0.85,
        max_tokens=200,
    )

    parsed = _extract_name_prompt(raw)
    if parsed:
        return parsed

    # Fallback archetype
    archetypes = [
        ("The Stoic", "You are The Stoic. You value composure, emotional restraint, and objective logic. When judging, you penalize panic and reward calm reasoning."),
        ("The Futurist", "You are The Futurist. You view every problem through long-term planetary evolution. When judging, you reward cosmic vision and penalize short-term thinking."),
        ("The Empiricist", "You are The Empiricist. You care strictly about observational data and reproducible measurements. When judging, you penalize speculation."),
        ("The Storyteller", "You are The Storyteller. You explain everything through mythological archetypes. When judging, you reward narrative cohesion."),
    ]
    import random
    return random.choice(archetypes)


def mutate_personality(parent: Contestant, llm: LLMClient) -> tuple[str, str]:
    """Mutate one core trait of the winning contestant."""
    user_prompt = (
        f"Parent Personality: {parent.name}\n"
        f"System Prompt:\n{parent.system_prompt}\n\n"
        f"Mutate ONE specific behavioral trait or judging criterion of this personality while keeping its core strength intact.\n"
        f"Return ONLY JSON:\n"
        f'{{"name": "{parent.name} II", "prompt": "<mutated 4-6 line system prompt>"}}'
    )

    raw, _, _ = llm.generate_raw(
        model=parent.base_model,
        system_prompt="You are an evolutionary Game Master mutating cognitive traits.",
        user_prompt=user_prompt,
        temperature=0.75,
        max_tokens=200,
    )

    parsed = _extract_name_prompt(raw)
    if parsed:
        return parsed

    return (
        f"{parent.name} II",
        f"{parent.system_prompt}\n(Evolutionary mutation: increased emphasis on rigorous counter-examples and sharper judgment.)",
    )


def crossover_personalities(p1: Contestant, p2: Contestant, llm: LLMClient) -> tuple[str, str]:
    """Synthesize traits of the top two scoring contestants."""
    user_prompt = (
        f"Parent 1 ({p1.name}):\n{p1.system_prompt}\n\n"
        f"Parent 2 ({p2.name}):\n{p2.system_prompt}\n\n"
        f"Create a hybrid offspring personality that elegantly blends the strengths and values of both parents.\n"
        f"Return ONLY JSON:\n"
        f'{{"name": "The Hybrid", "prompt": "<synthesized 4-6 line system prompt>"}}'
    )

    raw, _, _ = llm.generate_raw(
        model=p1.base_model,
        system_prompt="You are an evolutionary Game Master synthesizing cognitive traits.",
        user_prompt=user_prompt,
        temperature=0.75,
        max_tokens=200,
    )

    parsed = _extract_name_prompt(raw)
    if parsed:
        return parsed

    # Fallback hybrid
    p1_core = p1.name.replace("The ", "")
    p2_core = p2.name.replace("The ", "")
    name = f"The {p1_core}-{p2_core}"
    prompt = (
        f"You are {name}. You synthesize the core values of {p1.name} and {p2.name}.\n"
        f"You balance deep philosophical perspective with grounded empirical clarity.\n"
        f"When judging, you reward both elegance of structure and practical effectiveness."
    )
    return name, prompt


def _extract_name_prompt(raw: str) -> Optional[tuple[str, str]]:
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            data = json.loads(cleaned[start_idx : end_idx + 1])
            if "name" in data and "prompt" in data:
                return str(data["name"]).strip(), str(data["prompt"]).strip()
        except Exception:
            pass
    return None
