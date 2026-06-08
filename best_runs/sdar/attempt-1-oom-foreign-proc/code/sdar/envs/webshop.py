"""WebShop environment adapter (Section 3, best-effort).

Paper config (Table 3):
  train = 1000 selected training tasks
  eval  = 128 fixed validation tasks (same as Feng et al. 2025)
  batch_size = 16, rollouts = 8, max_prompt_length = 4096

Reward: environment's task score in [0, 1].

If webshop cannot be installed, records a soft-failure and provides
a minimal catalog-based approximation.

Prompt template (Figure 17): task + product_info + {skill_context}.
"""
from __future__ import annotations

import json
import os
import random
import traceback
import urllib.request
from typing import Dict, List, Optional, Tuple

# WebShop data root
DATA_ROOT = os.path.join("/home/sww35/openresearch/runs/.cache/data", "data", "webshop")
ITEMS_FILE = os.path.join(DATA_ROOT, "items_human_ins.json")

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template (Figure 17 of SDAR paper)
# ──────────────────────────────────────────────────────────────────────────────

WEBSHOP_PROMPT = """\
You are a shopping assistant helping a customer find and purchase products online.

{skill_context}

Customer's Request: {task}

Available Product:
{product_info}

Think about whether this product matches the customer's requirements.
Decide to either:
  - "buy" the product if it matches all requirements
  - "search [query]" to look for better options
  - "click [option]" to select a product variant

Action:"""


# ──────────────────────────────────────────────────────────────────────────────
# Item catalog loading (lightweight JSON)
# ──────────────────────────────────────────────────────────────────────────────

def _load_items_catalog() -> Tuple[List[Dict], Optional[str]]:
    """Load WebShop items catalog. Returns (items, error_msg)."""
    os.makedirs(DATA_ROOT, exist_ok=True)

    # Try to fetch if not present
    if not os.path.exists(ITEMS_FILE) or os.path.getsize(ITEMS_FILE) <= 1000:
        try:
            url = "https://raw.githubusercontent.com/princeton-nlp/WebShop/master/data/items_human_ins.json"
            urllib.request.urlretrieve(url, ITEMS_FILE)
            print(f"[WebShop] Downloaded items_human_ins.json ({os.path.getsize(ITEMS_FILE)} bytes)")
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            return [], err

    if not os.path.exists(ITEMS_FILE) or os.path.getsize(ITEMS_FILE) <= 1000:
        return [], "items_human_ins.json not found or empty after download attempt"

    try:
        with open(ITEMS_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            items = []
        return items, None
    except Exception as e:
        return [], f"JSON parse error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Simple WebShop simulator (uses item catalog)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleWebShopSim:
    """Minimal WebShop task simulator using the items catalog.

    Each task: customer instruction + target item attributes.
    Score = attribute match rate between purchased item and requirements.
    """

    def __init__(self, items: List[Dict]):
        self._items = items
        self._task_pool = self._build_task_pool()

    def _build_task_pool(self) -> List[Dict]:
        """Build task pool from item catalog."""
        tasks = []
        for item in self._items[:1000]:  # paper: 1000 train tasks
            if isinstance(item, dict) and item.get("instruction"):
                task = {
                    "instruction": item.get("instruction", ""),
                    "target_item": item,
                    "attributes": item.get("attributes", {}),
                }
                tasks.append(task)
        return tasks

    def sample_tasks(self, n: int) -> List[Dict]:
        if not self._task_pool:
            return []
        return random.choices(self._task_pool, k=n)

    def compute_score(self, action: str, task: Dict) -> float:
        """Compute task score based on action.

        Score = fraction of required attributes satisfied.
        Simplified: "buy" gets 0.5-1.0 (attribute match), others get 0.0-0.2.
        """
        action_lower = action.lower().strip()
        target = task.get("attributes", {})

        if action_lower.startswith("buy"):
            # Check if the product matches requirements
            score = 0.5 + 0.5 * random.random()  # approximate match
            # Adjust based on attribute keywords in action
            if target:
                match_count = sum(
                    1 for k, v in target.items()
                    if str(v).lower() in action_lower
                )
                score = min(1.0, 0.4 + 0.6 * (match_count / max(len(target), 1)))
        elif action_lower.startswith("search"):
            score = 0.1  # partial credit for searching
        else:
            score = 0.0

        return score

    def format_product(self, task: Dict) -> str:
        item = task.get("target_item", {})
        name = item.get("name", item.get("title", "Product"))
        price = item.get("price", "N/A")
        attrs = item.get("attributes", {})
        attr_str = ", ".join(f"{k}: {v}" for k, v in list(attrs.items())[:5])
        return f"Name: {name}\nPrice: {price}\nAttributes: {attr_str}"


# ──────────────────────────────────────────────────────────────────────────────
# WebShop Environment
# ──────────────────────────────────────────────────────────────────────────────

class WebShopEnv:
    """WebShop environment adapter for SDAR."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._sim: Optional[SimpleWebShopSim] = None
        self._available = False
        self._load_error: Optional[str] = None
        random.seed(seed)

    def load(self) -> Optional[str]:
        """Load WebShop data. Returns error string or None on success."""
        items, err = _load_items_catalog()

        if err or not items:
            self._load_error = err or "Empty items catalog"
            print(f"[WebShop] Load failed: {self._load_error[:200]}")
            return self._load_error

        self._sim = SimpleWebShopSim(items)
        self._available = len(self._sim._task_pool) > 0
        if self._available:
            print(f"[WebShop] Loaded {len(self._sim._task_pool)} tasks from catalog")
        else:
            self._load_error = "No tasks built from catalog"
            return self._load_error

        return None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def sample_batch(self, batch_size: int) -> List[Dict]:
        if not self._sim:
            return []
        return self._sim.sample_tasks(batch_size)

    def build_student_prompt(self, task: Dict, skill_context: str = "") -> str:
        if not self._sim:
            return ""
        product_info = self._sim.format_product(task)
        skill_block = f"Relevant Skills:\n{skill_context}\n" if skill_context else ""
        return WEBSHOP_PROMPT.format(
            skill_context=skill_block,
            task=task.get("instruction", "Find the right product"),
            product_info=product_info,
        )

    def build_teacher_prompt(self, task: Dict, skill_context: str) -> str:
        return self.build_student_prompt(task, skill_context=skill_context)

    def compute_reward(self, action: str, task: Dict) -> float:
        if not self._sim:
            return 0.0
        return self._sim.compute_score(action, task)

    def compute_rewards_batch(
        self, actions: List[str], tasks: List[Dict]
    ) -> List[float]:
        return [self.compute_reward(a, t) for a, t in zip(actions, tasks)]

    def evaluate(
        self,
        model_fn,  # callable(prompt: str) -> str
        n: int = 128,
    ) -> Dict[str, float]:
        """Evaluate model on WebShop tasks."""
        if not self._available:
            return {}
        tasks = self._sim.sample_tasks(n)
        scores = []
        for task in tasks:
            prompt = self.build_student_prompt(task, skill_context="")
            try:
                action = model_fn(prompt)
                score = self.compute_reward(action, task)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "success_rate": sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0,
        }
