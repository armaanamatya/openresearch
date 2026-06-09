"""SkillBank and four retrieval strategies (Section 2.2 + Table 3).

Strategies:
  - KM  (Keyword Matching, paper default): match query keywords to skill tags
  - UCB (Upper Confidence Bound, Eq 1):   balance exploration/exploitation
  - Full: return all skills concatenated (truncated if needed)
  - Random: sample k skills uniformly

The teacher = same model + retrieved skill context (Section 2.1).
During SDAR training: skills are retrieved and prepended to teacher prompts.
During SDAR eval: {skill_context} is EMPTY (no skills at inference time).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Built-in skill bank (used when ZJU-REAL/SkillBank is unavailable)
# Constructed from paper domain knowledge: Search-QA, ALFWorld, WebShop
# ──────────────────────────────────────────────────────────────────────────────

_BUILTIN_SKILLS: Dict[str, Dict] = {
    # Search-QA / Multi-hop QA skills
    "decompose": {
        "name": "decompose",
        "tags": ["multi-hop", "complex", "multiple", "steps", "what", "who", "which"],
        "domain": "search_qa",
        "text": (
            "Decomposition Skill: Break complex questions into sub-questions. "
            "For multi-hop questions, identify intermediate entities first. "
            "Example: 'Who directed the film starring X?' → (1) Find films starring X; "
            "(2) Find director of that film."
        ),
    },
    "numeric": {
        "name": "numeric",
        "tags": ["number", "year", "date", "count", "how many", "when", "calculate"],
        "domain": "search_qa",
        "text": (
            "Numeric Reasoning Skill: Extract relevant numbers and perform arithmetic. "
            "When asked about dates, years, or quantities, locate the specific values in "
            "the retrieved passages before computing. Present final numeric answer directly."
        ),
    },
    "comparison": {
        "name": "comparison",
        "tags": ["compare", "more", "less", "larger", "smaller", "between", "versus", "or"],
        "domain": "search_qa",
        "text": (
            "Comparison Skill: When comparing entities, first retrieve facts about each "
            "entity separately. Then compare on the relevant attribute. State clearly "
            "which entity satisfies the comparison criterion."
        ),
    },
    "verification": {
        "name": "verification",
        "tags": ["true", "false", "correct", "verify", "is", "does", "did", "was"],
        "domain": "search_qa",
        "text": (
            "Verification Skill: For yes/no or true/false questions, locate the relevant "
            "claim in the passage and check whether it matches the question. "
            "Answer with a clear True/False before explaining."
        ),
    },
    "entity_linking": {
        "name": "entity_linking",
        "tags": ["named", "entity", "person", "place", "organization", "what is", "who is"],
        "domain": "search_qa",
        "text": (
            "Entity Linking Skill: Identify the main entity in the question. "
            "Use the retrieved passage to find attributes of that entity. "
            "Be precise about entity boundaries."
        ),
    },
    # ALFWorld skills
    "navigation": {
        "name": "navigation",
        "tags": ["go", "move", "walk", "find", "look", "where", "room", "location"],
        "domain": "alfworld",
        "text": (
            "Navigation Skill: To find an object, systematically search the environment. "
            "Start with the most likely container or location based on the task. "
            "Use 'go to' for locations and 'look' to observe the surroundings."
        ),
    },
    "pick_place": {
        "name": "pick_place",
        "tags": ["pick", "take", "put", "place", "move", "carry", "drop"],
        "domain": "alfworld",
        "text": (
            "Pick and Place Skill: (1) Navigate to the object location. "
            "(2) Pick up the object using 'take <object> from <location>'. "
            "(3) Navigate to the target location. "
            "(4) Place the object using 'put <object> in/on <receptacle>'."
        ),
    },
    "heat_cool": {
        "name": "heat_cool",
        "tags": ["heat", "cool", "microwave", "fridge", "hot", "cold", "temperature"],
        "domain": "alfworld",
        "text": (
            "Heat/Cool Skill: To heat an object, put it in the microwave and turn it on. "
            "To cool an object, put it in the fridge. After heating/cooling, "
            "take the object out before placing it at the final destination."
        ),
    },
    "clean": {
        "name": "clean",
        "tags": ["clean", "wash", "dirty", "sink", "basin"],
        "domain": "alfworld",
        "text": (
            "Cleaning Skill: To clean an object, take it to the sink/basin and "
            "use 'clean <object> with sink'. "
            "Ensure the faucet is accessible. Pick up the object first."
        ),
    },
    # WebShop skills
    "product_search": {
        "name": "product_search",
        "tags": ["buy", "purchase", "search", "find product", "item", "price"],
        "domain": "webshop",
        "text": (
            "Product Search Skill: (1) Search with specific keywords from the task. "
            "(2) Filter by relevant attributes (color, size, price). "
            "(3) Select the product that best matches all requirements. "
            "(4) Verify attributes before adding to cart."
        ),
    },
    "attribute_matching": {
        "name": "attribute_matching",
        "tags": ["color", "size", "material", "type", "feature", "specification"],
        "domain": "webshop",
        "text": (
            "Attribute Matching Skill: Parse the task for required attributes. "
            "Check each product attribute against requirements. "
            "Prefer exact matches over partial. Use the 'Options' to select "
            "specific variants (e.g., size, color)."
        ),
    },
}


class SkillBank:
    """Container for skills with retrieval methods.

    Attempts to load the real ZJU-REAL/SkillBank; falls back to the
    built-in representative bank constructed from paper domain knowledge.
    """

    def __init__(
        self,
        domain: str = "search_qa",
        hf_cache: Optional[str] = None,
        max_skill_tokens: int = 512,
    ):
        self.domain = domain
        self.max_skill_tokens = max_skill_tokens
        self._skills: List[Dict] = []
        self._ucb_values: Dict[str, float] = {}
        self._ucb_counts: Dict[str, int] = {}
        self._ucb_total: int = 0
        self._provenance: str = "unknown"

        self._load(hf_cache)

    def _load(self, hf_cache: Optional[str]) -> None:
        """Try to load from HF hub; fall back to built-in bank."""
        loaded = False

        # Try to load from HF
        if not loaded:
            try:
                from datasets import load_dataset
                cache = hf_cache or os.environ.get(
                    "HF_HOME", "/home/sww35/openresearch/runs/.cache/hf"
                )
                ds = load_dataset(
                    "ZJU-REAL/SkillBank",
                    split="train",
                    cache_dir=cache,
                )
                self._skills = [
                    {
                        "name": row.get("skill_name", f"skill_{i}"),
                        "tags": row.get("tags", []) if isinstance(row.get("tags"), list) else
                                row.get("tags", "").split(","),
                        "domain": row.get("domain", self.domain),
                        "text": row.get("skill_text", row.get("text", "")),
                    }
                    for i, row in enumerate(ds)
                ]
                self._provenance = "ZJU-REAL/SkillBank (HuggingFace)"
                loaded = True
                print(f"[SkillBank] Loaded {len(self._skills)} skills from ZJU-REAL/SkillBank")
            except Exception as e:
                print(f"[SkillBank] ZJU-REAL/SkillBank unavailable ({type(e).__name__}: {str(e)[:100]}); "
                      f"using built-in representative bank")

        if not loaded:
            self._skills = [
                s for s in _BUILTIN_SKILLS.values()
                if s["domain"] == self.domain or s["domain"] == "search_qa"
            ]
            self._provenance = (
                "Built-in representative bank (constructed from paper domain knowledge; "
                "ZJU-REAL/SkillBank was unavailable)"
            )
            print(f"[SkillBank] Using built-in bank: {len(self._skills)} skills for domain={self.domain}")

        # Initialize UCB state
        for skill in self._skills:
            sid = skill["name"]
            self._ucb_values[sid] = 0.5  # optimistic init
            self._ucb_counts[sid] = 1

    @property
    def skills(self) -> List[Dict]:
        return self._skills

    @property
    def provenance(self) -> str:
        return self._provenance

    def retrieve(
        self,
        query: str,
        strategy: str = "km",
        k: int = 3,
        c: float = 1.0,
    ) -> List[Dict]:
        """Retrieve k skills for a given query using the specified strategy.

        Args:
            query: The input query string.
            strategy: "km", "ucb", "full", or "random"
            k: number of skills to return (ignored for "full")
            c: UCB exploration constant (Eq 1 in paper)

        Returns:
            List of skill dicts
        """
        if not self._skills:
            return []

        strategy = strategy.lower()
        if strategy == "km":
            return self._km_retrieve(query, k)
        elif strategy == "ucb":
            return self._ucb_retrieve(query, k, c)
        elif strategy == "full":
            return self._full_retrieve(query)
        elif strategy == "random":
            return self._random_retrieve(k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy!r}. "
                             f"Choose from: km, ucb, full, random")

    def _km_retrieve(self, query: str, k: int) -> List[Dict]:
        """Keyword matching: score each skill by keyword overlap with query."""
        query_lower = query.lower()
        query_tokens = set(re.findall(r'\w+', query_lower))

        scored = []
        for skill in self._skills:
            tags = [t.lower() for t in skill.get("tags", [])]
            # Score = fraction of tags that appear in query
            if tags:
                overlap = sum(1 for tag in tags if any(t in query_lower for t in tag.split()))
                score = overlap / len(tags)
            else:
                score = 0.0
            scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def _ucb_retrieve(self, query: str, k: int, c: float = 1.0) -> List[Dict]:
        """UCB retrieval (Eq 1): score_i = value_i + c * sqrt(log(N) / n_i).

        Balances exploitation (value_i) with exploration (uncertainty term).
        After retrieval, values are updated based on next reward signal via
        update_ucb().
        """
        N = max(self._ucb_total, 1)
        ucb_scores = []
        for skill in self._skills:
            sid = skill["name"]
            v_i = self._ucb_values.get(sid, 0.5)
            n_i = max(self._ucb_counts.get(sid, 1), 1)
            ucb = v_i + c * math.sqrt(math.log(N + 1) / n_i)
            ucb_scores.append((ucb, skill))

        ucb_scores.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in ucb_scores[:k]]

    def update_ucb(self, skill_names: List[str], reward: float) -> None:
        """Update UCB statistics after observing a reward."""
        self._ucb_total += 1
        for name in skill_names:
            if name not in self._ucb_values:
                self._ucb_values[name] = 0.5
                self._ucb_counts[name] = 1
            n = self._ucb_counts[name]
            # Running average update
            self._ucb_values[name] = self._ucb_values[name] + (reward - self._ucb_values[name]) / (n + 1)
            self._ucb_counts[name] = n + 1

    def _full_retrieve(self, query: str) -> List[Dict]:
        """Return all skills (truncated by token budget in format_skills)."""
        return list(self._skills)

    def _random_retrieve(self, k: int) -> List[Dict]:
        """Uniformly sample k skills."""
        k = min(k, len(self._skills))
        return random.sample(self._skills, k)

    def format_skills(self, skills: List[Dict], max_tokens: Optional[int] = None) -> str:
        """Format retrieved skills into a skill context string."""
        if not skills:
            return ""
        max_t = max_tokens or self.max_skill_tokens
        parts = []
        total_len = 0
        for skill in skills:
            text = skill.get("text", skill.get("name", ""))
            name = skill.get("name", "")
            entry = f"[Skill: {name}] {text}"
            # Rough token estimate: 4 chars ≈ 1 token
            estimated_tokens = len(entry) // 4
            if total_len + estimated_tokens > max_t and parts:
                break
            parts.append(entry)
            total_len += estimated_tokens
        return "\n".join(parts)
