from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import ollama
from pydantic import ValidationError

from src.models import (
    CandidateAnswer,
    Contestant,
    Question,
    VotePayload,
)

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> Optional[dict]:
    """Extract a JSON object from text, handling markdown fences and pre/post-ambles."""
    cleaned = text.strip()
    # Remove markdown code block fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Find the outermost JSON object bounds
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class LLMClient:
    """Ollama wrapper with retries, timeouts, JSON parsing, and graceful offline fallback."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        force_mock: bool = False,
    ):
        self.host = host
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.force_mock = force_mock
        self._client: Optional[ollama.Client] = None
        self._available_models: Optional[List[str]] = None
        self._is_online: Optional[bool] = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(
                host=self.host,
                timeout=float(self.timeout_seconds),
            )
        return self._client

    def is_ollama_available(self) -> bool:
        """Check if local Ollama daemon is active and responsive."""
        if self.force_mock:
            return False
        if self._is_online is not None:
            return self._is_online
        try:
            res = self.client.list()
            models_list = getattr(res, "models", [])
            # Model objects have a .model or .name attribute
            self._available_models = [
                getattr(m, "model", getattr(m, "name", str(m))) for m in models_list
            ]
            self._is_online = True
            logger.info("Connected to Ollama on %s. Available models: %s", self.host, self._available_models)
            return True
        except Exception as e:
            logger.warning(
                "Ollama daemon unreachable at %s (%s). Engaging graceful offline simulation.",
                self.host,
                str(e),
            )
            self._is_online = False
            return False

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model has been pulled."""
        if not self.is_ollama_available():
            return False
        if self._available_models is None:
            return False
        # Match exact or base tag, e.g. "llama3.2:3b" or "llama3.2:latest"
        base_name = model_name.split(":")[0]
        return any(
            model_name == m or m.startswith(f"{base_name}:") for m in self._available_models
        )

    def generate_raw(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ) -> Tuple[str, float, int]:
        """Perform raw generation with latency measurement and error handling."""
        start_time = time.perf_counter()

        if self.is_ollama_available():
            try:
                response = self.client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": float(temperature),
                        "num_predict": int(max_tokens),
                    },
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                message = response.get("message", {})
                content = message.get("content", "").strip()
                tokens = response.get("eval_count", len(content.split()))
                return content, latency_ms, tokens
            except Exception as e:
                logger.warning(
                    "Ollama call for model '%s' failed (%s). Falling back to persona generator.",
                    model,
                    str(e),
                )

        # Fallback offline generator (persona-tailored)
        latency_ms = (time.perf_counter() - start_time) * 1000
        content = self._mock_generate(system_prompt, user_prompt, max_tokens)
        tokens = len(content.split())
        return content, latency_ms, tokens

    def generate_answer(
        self,
        contestant: Contestant,
        question: Question,
        memory: Optional[str] = None,
        max_tokens: int = 120,
    ) -> Tuple[str, float, int]:
        """Generate contestant answer according to its personality and memory."""
        prompt_parts = [
            f"Question: {question.prompt}\n",
            f"Provide your answer in 2 to 4 concise sentences, speaking completely in your unique voice and personality.",
            f"Do not exceed {max_tokens} words. Avoid introductory fluff.",
        ]
        if memory:
            prompt_parts.insert(0, f"Your Memory of Past Rounds:\n{memory}\n")

        user_prompt = "\n".join(prompt_parts)
        return self.generate_raw(
            model=contestant.base_model,
            system_prompt=contestant.system_prompt,
            user_prompt=user_prompt,
            temperature=contestant.temperature,
            max_tokens=max_tokens,
        )

    def generate_vote(
        self,
        contestant: Contestant,
        question: Question,
        candidates: List[CandidateAnswer],
        is_named: bool = False,
    ) -> Tuple[Optional[VotePayload], str, bool]:
        """Ask a contestant to evaluate candidate answers and return a strict JSON vote."""
        labels = [c.label for c in candidates]
        labels_str = ", ".join(labels)

        candidates_formatted = []
        for c in candidates:
            if is_named:
                candidates_formatted.append(f"Answer [{c.label}] (by {c.contestant_name}):\n\"{c.text}\"")
            else:
                candidates_formatted.append(f"Answer [{c.label}]:\n\"{c.text}\"")

        candidates_block = "\n\n".join(candidates_formatted)

        vote_instructions = (
            f"Question asked: \"{question.prompt}\"\n\n"
            f"Below are the candidate answers from other contestants (your own answer is excluded):\n\n"
            f"{candidates_block}\n\n"
            f"Your Task:\n"
            f"1. Judge the answers according to your personal values and philosophy.\n"
            f"2. Select the single best answer ('vote'). Must be one of: {labels_str}.\n"
            f"3. Rank ALL answers from best to worst ('ranking'). Must contain every label in [{labels_str}] exactly once.\n"
            f"4. Give a brief reason explaining your decision in character.\n\n"
            f"CRITICAL: Respond ONLY with valid JSON matching this schema, with no markdown fences or other text:\n"
            f'{{"vote": "<label>", "ranking": ["<label>", ...], "reason": "<short explanation>"}}'
        )

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            retry_note = ""
            if attempt > 0:
                retry_note = f"\n\nPrevious attempt was invalid. Return ONLY pure JSON with valid keys 'vote', 'ranking', and 'reason'."

            raw_resp, _, _ = self.generate_raw(
                model=contestant.base_model,
                system_prompt=contestant.system_prompt,
                user_prompt=vote_instructions + retry_note,
                temperature=max(0.2, contestant.temperature - 0.2),  # Lower temp for JSON
                max_tokens=150,
            )
            last_raw = raw_resp

            parsed_data = extract_json_object(raw_resp)
            if parsed_data:
                try:
                    payload = VotePayload.model_validate(parsed_data)
                    # Validate label membership
                    payload.vote = str(payload.vote).strip().upper()
                    payload.ranking = [str(lbl).strip().upper() for lbl in payload.ranking]

                    if payload.vote in labels and set(payload.ranking) == set(labels):
                        return payload, last_raw, True
                    # If ranking has correct labels but vote is off or ranking is partial:
                    if payload.vote in labels:
                        # Fix ranking order by appending missing labels
                        clean_rank = [lbl for lbl in payload.ranking if lbl in labels]
                        for lbl in labels:
                            if lbl not in clean_rank:
                                clean_rank.append(lbl)
                        payload.ranking = clean_rank
                        return payload, last_raw, True
                except ValidationError:
                    pass

        # If in offline/mock mode or all live retries failed, generate a persona-grounded vote
        fallback_vote = self._mock_vote(contestant, candidates)
        if fallback_vote:
            raw_json = json.dumps(fallback_vote.model_dump())
            return fallback_vote, raw_json, True

        # Abstention if impossible
        return None, last_raw, False

    def _mock_generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Personality-grounded offline response generator."""
        sys_lower = system_prompt.lower()
        if "skeptic" in sys_lower:
            return (
                "We must demand verifiable empirical data before jumping to hasty conclusions. "
                "Any hypothesis without clear falsifiability or controlled testing is premature speculation."
            )
        elif "poet" in sys_lower:
            return (
                "Truth is not merely a ledger of dry facts, but the melody beneath the noise. "
                "In our search for understanding, we are travelers gazing into the mirror of existence."
            )
        elif "engineer" in sys_lower:
            return (
                "Under practical operational constraints, the simplest working implementation wins. "
                "Measure latency, verify trade-offs, and iterate based on actual telemetry rather than theoretical ideals."
            )
        elif "contrarian" in sys_lower:
            return (
                "The orthodox consensus has this completely backwards. "
                "If everyone agrees on the premise, you should suspect groupthink and interrogate the foundational blind spots."
            )
        elif "diplomat" in sys_lower:
            return (
                "A balanced perspective synthesizes the merits of both viewpoints while bridging ideological divides. "
                "Constructive progress requires mutual respect and principled compromise."
            )
        elif "know-it-all" in sys_lower:
            return (
                "The literature on this question is extensive, spanning classical epistemological treatises to contemporary empirical studies. "
                "To be historically precise, the foundational axioms require rigorous conceptual demarcation."
            )
        elif "minimalist" in sys_lower:
            return "Eliminate the noise. Focus exclusively on the single fundamental cause."
        elif "showman" in sys_lower:
            return (
                "Ladies and gentlemen, behold the true drama of the question! "
                "Ideas only conquer the world when they seize our imagination with audacious passion and undeniable spectacle!"
            )
        else:
            return "From this perspective, thoughtful analysis and clear reasoning reveal the core insight."

    def _mock_vote(self, contestant: Contestant, candidates: List[CandidateAnswer]) -> VotePayload:
        """Personality-informed deterministic voting heuristic for offline testing."""
        sys_lower = contestant.system_prompt.lower()
        # Heuristic scoring based on candidate length, keywords, and personality alignment
        scored: List[Tuple[float, CandidateAnswer]] = []
        for c in candidates:
            score = 1.0
            word_count = len(c.text.split())
            text_lower = c.text.lower()

            if "minimalist" in sys_lower:
                score += 10.0 / max(5, word_count)
            elif "know-it-all" in sys_lower:
                score += word_count * 0.1
                if any(w in text_lower for w in ["precise", "data", "studies", "empirical"]):
                    score += 2.0
            elif "poet" in sys_lower:
                if any(w in text_lower for w in ["truth", "beauty", "melody", "mirror", "human"]):
                    score += 3.0
            elif "engineer" in sys_lower:
                if any(w in text_lower for w in ["practical", "measure", "trade-off", "works", "implement"]):
                    score += 3.0
            elif "skeptic" in sys_lower:
                if any(w in text_lower for w in ["empirical", "verifiable", "data", "testing"]):
                    score += 3.0
            elif "contrarian" in sys_lower:
                # Favors non-standard answers
                score += (hash(c.text) % 5)

            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_labels = [c.label for _, c in scored]
        best_label = ranked_labels[0]
        reason = f"Ranked {best_label} highest based on adherence to {contestant.name}'s standards."

        return VotePayload(
            vote=best_label,
            ranking=ranked_labels,
            reason=reason,
        )
