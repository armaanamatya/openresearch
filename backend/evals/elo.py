"""Elo tournament system for comparing agent versions via pairwise matches.

Run the same paper through the pipeline N times with different agent versions,
then have an LLM judge compare pairs of Research Maps. Compute Elo ratings.

References:
- Google AI Co-Scientist: Elo ratings from self-play tournaments
- Standard Elo: K=32, starting rating 1500
"""

from __future__ import annotations

import math
import random
from typing import Any

from backend.evals.schemas import EloMatchResult, EloRating


ELO_K_FACTOR = 32
ELO_START_RATING = 1500.0


JUDGE_PROMPT = """You are comparing two Research Maps produced by different AI agent versions
for the same paper. Determine which is better overall.

Paper context: {paper_context}

=== Research Map A (version: {version_a}) ===
{map_a}

=== Research Map B (version: {version_b}) ===
{map_b}

Evaluate on:
1. Are dead ends and promising directions correctly classified?
2. Are the suggested next experiments novel and actionable?
3. Is the synthesis coherent and honest about failures?
4. Would a researcher find this useful?

Respond with EXACTLY one line:
WINNER: A
or
WINNER: B
or
WINNER: DRAW

Then explain your reasoning briefly.
"""


class EloTournament:
    """Elo tournament for ranking agent versions.

    Usage:
        tournament = EloTournament()
        tournament.add_competitor("v1.0")
        tournament.add_competitor("v1.1")
        tournament.add_competitor("v2.0")

        # Record match results (from LLM judge or manual evaluation)
        tournament.record_match("v1.0", "v1.1", winner="v1.1", paper_id="ppo")
        tournament.record_match("v1.1", "v2.0", winner="v2.0", paper_id="ppo")

        # Get rankings
        rankings = tournament.get_rankings()
    """

    def __init__(self, k_factor: float = ELO_K_FACTOR):
        self.k_factor = k_factor
        self._ratings: dict[str, EloRating] = {}
        self._matches: list[EloMatchResult] = []

    def add_competitor(self, version: str, starting_rating: float = ELO_START_RATING) -> None:
        if version not in self._ratings:
            self._ratings[version] = EloRating(
                version=version, rating=starting_rating,
            )

    @property
    def competitors(self) -> list[str]:
        return list(self._ratings.keys())

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for A given ratings."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def record_match(
        self,
        version_a: str,
        version_b: str,
        *,
        winner: str | None = None,
        paper_id: str = "",
        score_a: float = 0.0,
        score_b: float = 0.0,
        judge_rationale: str = "",
    ) -> None:
        """Record a match result and update ratings."""
        if version_a not in self._ratings:
            self.add_competitor(version_a)
        if version_b not in self._ratings:
            self.add_competitor(version_b)

        # Determine actual scores (1=win, 0.5=draw, 0=loss)
        if winner == version_a:
            actual_a, actual_b = 1.0, 0.0
        elif winner == version_b:
            actual_a, actual_b = 0.0, 1.0
        elif winner is None:
            actual_a, actual_b = 0.5, 0.5
        else:
            raise ValueError(f"Winner must be {version_a}, {version_b}, or None")

        # Compute expected scores
        ra = self._ratings[version_a].rating
        rb = self._ratings[version_b].rating
        expected_a = self.expected_score(ra, rb)
        expected_b = self.expected_score(rb, ra)

        # Update ratings
        self._ratings[version_a].rating += self.k_factor * (actual_a - expected_a)
        self._ratings[version_b].rating += self.k_factor * (actual_b - expected_b)

        # Update stats
        self._ratings[version_a].matches_played += 1
        self._ratings[version_b].matches_played += 1
        if winner == version_a:
            self._ratings[version_a].wins += 1
            self._ratings[version_b].losses += 1
        elif winner == version_b:
            self._ratings[version_b].wins += 1
            self._ratings[version_a].losses += 1
        else:
            self._ratings[version_a].draws += 1
            self._ratings[version_b].draws += 1

        # Store match
        self._matches.append(EloMatchResult(
            version_a=version_a,
            version_b=version_b,
            paper_id=paper_id,
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            judge_rationale=judge_rationale,
        ))

    def get_rankings(self) -> list[EloRating]:
        """Get all ratings sorted by rating (highest first)."""
        return sorted(self._ratings.values(), key=lambda r: r.rating, reverse=True)

    def get_rating(self, version: str) -> EloRating | None:
        return self._ratings.get(version)

    @property
    def matches(self) -> list[EloMatchResult]:
        return self._matches

    def run_round_robin(
        self,
        match_fn: Any,  # Callable[[str, str], str | None]
        paper_ids: list[str] | None = None,
        *,
        n_rounds: int = 1,
        seed: int | None = None,
    ) -> None:
        """Run a full round-robin tournament.

        match_fn(version_a, version_b) -> winner (version_a, version_b, or None for draw)
        """
        rng = random.Random(seed)
        versions = list(self._ratings.keys())

        for _ in range(n_rounds):
            # Generate all pairs
            pairs = [(a, b) for i, a in enumerate(versions) for b in versions[i+1:]]
            rng.shuffle(pairs)

            for version_a, version_b in pairs:
                papers = paper_ids or ["default"]
                for paper_id in papers:
                    winner = match_fn(version_a, version_b)
                    self.record_match(
                        version_a, version_b,
                        winner=winner,
                        paper_id=paper_id,
                    )

    def generate_judge_prompt(
        self,
        version_a: str,
        version_b: str,
        map_a_text: str,
        map_b_text: str,
        paper_context: str = "",
    ) -> str:
        """Generate the LLM judge prompt for a pairwise comparison."""
        return JUDGE_PROMPT.format(
            paper_context=paper_context,
            version_a=version_a,
            version_b=version_b,
            map_a=map_a_text,
            map_b=map_b_text,
        )

    def parse_judge_response(
        self,
        response: str,
        version_a: str,
        version_b: str,
    ) -> str | None:
        """Parse LLM judge response into winner."""
        response_upper = response.strip().upper()
        for line in response_upper.split("\n"):
            if "WINNER:" in line:
                if "WINNER: A" in line:
                    return version_a
                elif "WINNER: B" in line:
                    return version_b
                elif "DRAW" in line:
                    return None
        # Fallback: look for A or B anywhere in first line
        first_line = response_upper.split("\n")[0]
        if " A" in first_line or first_line.endswith("A"):
            return version_a
        if " B" in first_line or first_line.endswith("B"):
            return version_b
        return None  # draw if unclear


def compute_confidence(matches_played: int) -> float:
    """Confidence in Elo rating based on number of matches.

    Returns 0-1. Roughly: 10 matches = 0.7, 30 matches = 0.9, 50+ = 0.95+.
    """
    return 1.0 - math.exp(-matches_played / 20.0)
